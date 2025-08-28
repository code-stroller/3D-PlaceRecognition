"""
SER-MinkPTC (rev3):
- Overlapped serialization tokens (half-shift)
- x1-KNN tokenization for stable queries
- Dual-scale (x1/x2) cross-attention + Gaussian Relative Positional Encoding
- Sparse SE channel attention
- Output: padded (B, C, P) where C = 4 * base_channels (default 256), GeM-friendly

Python 3.8 / MinkowskiEngine 0.5.4
Coordinate format for ME: [batch, x, y, z] (int32 CPU)
Refs:
  - PTv3 / SFC (Hilbert, Z-order): Wu et al., 2023/2024. 
  - MinkLoc3Dv2: sparse CNN + channel attention + GeM. 
  - GeM pooling: Radenović et al. (2017–2019).
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

# -----------------------------------------------------------------------------
# Import user's serialization modules (dispatch only; encoding logic is external)
# -----------------------------------------------------------------------------
_hil_ok = _z_ok = _hz_ok = False

try:
    from hilbert import encode as hilbert_encode  # type: ignore
    _hil_ok = True
except Exception:
    pass


try:
    from z_order import xyz2key as morton_xyz2key  # type: ignore
    _z_ok = True
except Exception:
    pass


try:
    from hz_curve import xyz2key as hz_xyz2key  # type: ignore
    _hz_ok = True
except Exception:
    pass


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def _infer_depth_from_grid(grid_coord: torch.Tensor, max_bits: int = 16) -> int:
    if grid_coord.numel() == 0:
        return 1
    maxv = int(grid_coord.max().item())
    bitlen = max(1, maxv.bit_length())
    return max(1, min(max_bits, bitlen))


@torch.no_grad()
def _serialize_codes_dispatch_one(
    grid_coord: torch.Tensor,  # (N,3) long
    order: str,
    depth: int,
) -> torch.Tensor:
    """
    Return serialization key codes for ALL points (no per-batch split).
    Supports: "z", "z-trans", "hilbert", "hilbert-trans", "hz", "hz-trans".
    """
    assert grid_coord.dim() == 2 and grid_coord.size(1) == 3
    device = grid_coord.device
    trans = order.endswith("-trans")
    mode = order.replace("-trans", "")
    x, y, z = grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]
    if trans:
        x, y = y, x

    if mode == "z":
        assert _z_ok, "z_order module not available"
        codes = morton_xyz2key(x, y, z, b=None, depth=depth)
    elif mode == "hilbert":
        assert _hil_ok, "hilbert module not available"
        coords = torch.stack([x, y, z], dim=-1).to(torch.long)
        codes = hilbert_encode(coords, num_dims=3, num_bits=depth)
    elif mode == "hz":
        assert _hz_ok, "hz_curve module not available"
        codes = hz_xyz2key(x, y, z, b=None, depth=depth)
    else:
        raise ValueError(f"Unsupported order: {order}")
    return codes.to(device=device, dtype=torch.long)


# -----------------------------------------------------------------------------
# Sparse building blocks (ME 0.5.x)
# -----------------------------------------------------------------------------
def me_block(in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=k, stride=s, dilation=1, bias=False, dimension=3
        ),
        ME.MinkowskiBatchNorm(out_ch),
        ME.MinkowskiReLU(inplace=True),
    )


class SparseSE(nn.Module):
    """
    Lightweight Squeeze-and-Excitation for SparseTensor.
    Batch-wise channel squeeze → FC-ReLU-FC-Sigmoid → broadcast to points.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x: ME.SparseTensor, B: int) -> ME.SparseTensor:
        F_in = x.F  # (N, C)
        if F_in.numel() == 0:
            return x
        C = F_in.size(1)
        bidx = x.C[:, 0].long().to(F_in.device)  # [b,x,y,z] -> b
        sums = F_in.new_zeros(B, C)
        sums.scatter_add_(0, bidx.view(-1, 1).expand(-1, C), F_in)
        cnt = torch.bincount(bidx, minlength=B).clamp_min_(1).to(F_in.dtype).view(-1, 1)
        mean = sums / cnt  # (B, C)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(mean), inplace=True)))  # (B, C)
        F_out = F_in * w[bidx]
        return ME.SparseTensor(
            features=F_out,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )


class TokenRefiner1D(nn.Module):
    """
    1D token refiner: DWConv1d + GLU + PWConv (residual).
    Input/Output: (G, D)
    """
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=True)
        self.pw = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=True)  # GLU
        self.out = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.numel() == 0:
            return x_seq
        x = x_seq.transpose(0, 1).unsqueeze(0)  # (1, D, G)
        x = self.dw(x)
        x = self.pw(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        x = self.out(x)
        x = self.drop(x)
        return x.squeeze(0).transpose(0, 1) + x_seq  # residual


# -----------------------------------------------------------------------------
# Main backbone
# -----------------------------------------------------------------------------
class PointTransformerV3(nn.Module):
    """
    Input dict:
      'coord':  (N,3) float  (metric coordinates)
      'feat':   (N,in_channels) float
      'offset': (B+1,) long
      'grid_size': float (quantization size)
    Output: padded (B, C_out, P) with C_out = 4 * base_channels
    """
    def __init__(
        self,
        in_channels: int = 4,               # SI 4ch: intensity, z, range_xy, ones
        base_channels: int = 64,            # => C_out = 256
        attn_dim: int = 128,
        q_patch: int = 48,                  # smaller -> more tokens
        max_keys_l1: int = 256,             # #keys at x1 per batch
        max_keys_l2: int = 384,             # #keys at x2 per batch
        serialization_orders: Optional[List[str]] = None,  # e.g., ["hilbert","z"]
        serialization_depth: Optional[int] = None,         # num_bits; if None -> infer
        se_reduction: int = 8,
        q_from_x1: bool = True,             # build Query from x1 KNN features
        q_knn: int = 48,                    # KNN size for token features
        shifts: Optional[List[int]] = None  # overlapped tokens, e.g., [0, q_patch//2]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.q_patch = q_patch
        self.max_keys_l1 = max_keys_l1
        self.max_keys_l2 = max_keys_l2
        self.serialization_depth = serialization_depth
        self.serialization_orders = serialization_orders or ["hilbert"]
        self.q_from_x1 = q_from_x1
        self.q_knn = q_knn
        self.shifts = shifts if shifts is not None else [0, q_patch // 2]

        # Channels
        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4
        self.c_out = c2

        # Sparse pyramid + SE (MinkLoc3Dv2 style)
        self.conv0 = me_block(in_channels, c0, k=3, s=1)
        self.se0   = SparseSE(c0, reduction=se_reduction)

        self.conv1 = me_block(c0, c1, k=3, s=2)
        self.se1   = SparseSE(c1, reduction=se_reduction)

        self.conv2 = me_block(c1, c2, k=3, s=2)
        self.se2   = SparseSE(c2, reduction=se_reduction)

        # Projections
        self.query_proj_raw = nn.Linear(in_channels, attn_dim, bias=True)
        self.query_proj_x1  = nn.Linear(c1, attn_dim, bias=True)

        self.key1_proj  = nn.Linear(c1, attn_dim, bias=True)
        self.val1_proj  = nn.Linear(c1, attn_dim, bias=True)
        self.key2_proj  = nn.Linear(c2, attn_dim, bias=True)
        self.val2_proj  = nn.Linear(c2, attn_dim, bias=True)
        self.out_proj   = nn.Linear(attn_dim, c2, bias=True)

        # Token refiner
        self.refiner = TokenRefiner1D(attn_dim, kernel_size=5, dropout=0.1)

        # Norms
        self.q_ln_raw = nn.LayerNorm(attn_dim)
        self.q_ln_x1  = nn.LayerNorm(attn_dim)
        self.k1_ln = nn.LayerNorm(attn_dim); self.v1_ln = nn.LayerNorm(attn_dim)
        self.k2_ln = nn.LayerNorm(attn_dim); self.v2_ln = nn.LayerNorm(attn_dim)

        # Scale gating (x1 vs x2)
        self.scale_logits = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

        # Order fusion (if we fuse orders; here we simply concatenate sets, but keep for future)
        self.order_logits = nn.Parameter(torch.zeros(len(self.serialization_orders), dtype=torch.float32))

        # Gaussian RPE parameter: inv_sigma^2 (positive via softplus)
        self.raw_pos_inv_sigma2 = nn.Parameter(torch.tensor(1.0))

    # ----------------- utilities -----------------
    @torch.no_grad()
    def _build_batch_index(self, offset: torch.Tensor) -> torch.Tensor:
        counts = (offset[1:] - offset[:-1]).long()
        B = counts.numel()
        return torch.repeat_interleave(torch.arange(B, device=offset.device, dtype=torch.long), counts)

    @torch.no_grad()
    def _int_grid_coords(self, coords: torch.Tensor, offset: torch.Tensor, grid_size: float) -> torch.Tensor:
        """
        For each batch: subtract min, divide by grid_size, floor -> integer grid coords (N,3)
        """
        B = offset.numel() - 1
        out = torch.empty_like(coords, dtype=torch.long)
        for b in range(B):
            s, e = int(offset[b].item()), int(offset[b + 1].item())
            if e <= s:
                continue
            pts = coords[s:e]
            mn = pts.min(dim=0).values
            out[s:e] = torch.floor((pts - mn) / grid_size).to(torch.long)
        return out

    @staticmethod
    def _grouped_means(values: torch.Tensor, groups: torch.Tensor, num_groups: int) -> torch.Tensor:
        D = values.size(1)
        out = values.new_zeros(num_groups, D)
        out.scatter_add_(0, groups.view(-1, 1).expand(-1, D), values)
        cnt = torch.bincount(groups, minlength=num_groups).clamp_min_(1).view(-1, 1).to(values.dtype)
        return out / cnt

    def _select_topk_keys(
        self, K_feat_all: torch.Tensor, K_coor_all: torch.Tensor, b: int, max_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (K_coor_all[:, 0] == b)
        if not mask.any():
            return K_feat_all.new_zeros(0, K_feat_all.size(1)), K_coor_all.new_zeros(0, 4)
        Kf = K_feat_all[mask]
        Kc = K_coor_all[mask]
        if max_k > 0 and Kf.size(0) > max_k:
            score = (Kf * Kf).sum(dim=1)
            topk = torch.topk(score, k=max_k, sorted=False).indices
            Kf = Kf[topk]
            Kc = Kc[topk]
        return Kf, Kc

    @staticmethod
    def _knn_aggregate(q_xyz: torch.Tensor, k_xyz: torch.Tensor, k_feat: torch.Tensor, k: int = 32, eps: float = 1e-6) -> torch.Tensor:
        """
        Build token features from k-NN of x1 around token center.
        q_xyz: (G,3), k_xyz: (M,3), k_feat: (M,C1) -> (G,C1)
        """
        if q_xyz.numel() == 0 or k_xyz.numel() == 0:
            return q_xyz.new_zeros(q_xyz.size(0), k_feat.size(1))
        q2 = (q_xyz * q_xyz).sum(dim=1, keepdim=True)               # (G,1)
        k2 = (k_xyz * k_xyz).sum(dim=1).view(1, -1)                 # (1,M)
        qk = q_xyz @ k_xyz.t()                                      # (G,M)
        dist2 = (q2 + k2 - 2.0 * qk).clamp_min_(0.0)                # (G,M)
        k_eff = min(k, k_feat.size(0))
        nn_idx = torch.topk(-dist2, k=k_eff, dim=1).indices         # (G,k)
        nn_dist = torch.gather(dist2, 1, nn_idx).sqrt_().clamp_min_(eps)  # (G,k)
        w = 1.0 / nn_dist
        w = w / (w.sum(dim=1, keepdim=True) + eps)
        nn_feat = k_feat[nn_idx]                                    # (G,k,C1)
        q_feat = (w.unsqueeze(-1) * nn_feat).sum(dim=1)             # (G,C1)
        return q_feat

    def _cross_attend(self, Q: torch.Tensor, Qxyz: torch.Tensor,
                      Kf: torch.Tensor, Kc: torch.Tensor,
                      key_proj: nn.Linear, val_proj: nn.Linear,
                      k_ln: nn.LayerNorm, v_ln: nn.LayerNorm,
                      grid: float) -> torch.Tensor:
        """
        Single-head attention with Gaussian positional bias.
        Returns: (G, A) in attention space (before out_proj)
        """
        if Kf.numel() == 0:
            return Q.new_zeros(Q.size(0), Q.size(1))
        A = Q.size(1)
        K = k_ln(key_proj(Kf))              # (M, A)
        V = v_ln(val_proj(Kf))              # (M, A)
        k_xyz = Kc[:, 1:4].to(torch.float32) * grid   # (M,3)

        # Dot-product + Gaussian RPE: -0.5 * inv_sigma^2 * ||Δ||^2
        scores = (Q @ K.t()) / (A ** 0.5)           # (G, M)
        q2 = (Qxyz * Qxyz).sum(dim=1, keepdim=True) # (G,1)
        k2 = (k_xyz * k_xyz).sum(dim=1).view(1, -1) # (1,M)
        qk = Qxyz @ k_xyz.t()                       # (G,M)
        dist2 = (q2 + k2 - 2.0 * qk).clamp_min_(0.0)
        inv_sigma2 = F.softplus(self.raw_pos_inv_sigma2) + 1e-6
        scores = scores - 0.5 * inv_sigma2 * dist2
        attn = F.softmax(scores, dim=-1)
        out = attn @ V                               # (G, A)
        return out

    # ----------------- forward -----------------
    def forward(self, point_dict: dict) -> torch.Tensor:
        """
        Input: {'coord','feat','offset','grid_size'}  ->  Output: padded (B, C, P)
        """
        coords_f: torch.Tensor = point_dict["coord"].to(torch.float32)   # (N,3)
        feats_f:  torch.Tensor = point_dict["feat"].to(torch.float32)    # (N,in_ch)
        offset:   torch.Tensor = point_dict["offset"].long()             # (B+1,)
        grid     = float(point_dict["grid_size"])
        device   = feats_f.device

        B = offset.numel() - 1
        if B <= 0:
            return feats_f.new_zeros(0, self.c_out, 0)

        # --- batch index & grid coords & depth ---
        bidx = self._build_batch_index(offset)                           # (N,)
        grid_coord = self._int_grid_coords(coords_f, offset, grid)       # (N,3) long
        depth = self.serialization_depth if self.serialization_depth is not None else _infer_depth_from_grid(grid_coord)

        # --- ME SparseTensor: [b, x, y, z] on CPU (int) ---
        me_coords = torch.cat([bidx.view(-1, 1).int(), grid_coord.int()], dim=1).cpu().int()
        x = ME.SparseTensor(features=feats_f, coordinates=me_coords, device=device)

        # Sparse pyramid + SE
        x0 = self.conv0(x);  x0 = self.se0(x0, B)
        x1 = self.conv1(x0); x1 = self.se1(x1, B)     # mid
        x2 = self.conv2(x1); x2 = self.se2(x2, B)     # coarse

        # x1/x2 features/coords
        K1_feat_all, K1_coor_all = x1.F, x1.C.int().to(device)  # (M1, C1), (M1, 4)
        K2_feat_all, K2_coor_all = x2.F, x2.C.int().to(device)  # (M2, C2), (M2, 4)

        # --- Precompute serialization codes for all orders (N-based) ---
        orders: List[str] = list(self.serialization_orders)
        all_codes = []
        for order in orders:
            codes = _serialize_codes_dispatch_one(grid_coord, order=order, depth=depth)  # (N,)
            all_codes.append(codes)

        token_out_list: List[torch.Tensor] = []
        token_counts: List[int] = []

        for b in range(B):
            s, e = int(offset[b].item()), int(offset[b + 1].item())
            if e <= s:
                token_out_list.append(None)
                token_counts.append(0)
                continue

            idx_b = torch.arange(s, e, device=device, dtype=torch.long)
            n_b = idx_b.numel()

            feats_b  = feats_f[idx_b]           # (n_b, in_ch)
            coords_b = coords_f[idx_b]          # (n_b, 3)

            # Gather keys for this batch
            K1_feat_b, K1_coor_b = self._select_topk_keys(K1_feat_all, K1_coor_all, b, self.max_keys_l1)
            K2_feat_b, K2_coor_b = self._select_topk_keys(K2_feat_all, K2_coor_all, b, self.max_keys_l2)

            # Fallback if no keys (empty cloud corner-case)
            if K1_feat_b.numel() == 0 and K2_feat_b.numel() == 0:
                # Just mean-pool raw feats by q_patch to keep shapes consistent
                groups = torch.arange(n_b, device=device) // self.q_patch
                G = int(groups[-1].item()) + 1 if n_b > 0 else 0
                Q = self.q_ln_raw(self.query_proj_raw(self._grouped_means(feats_b, groups, G)))
                Q = self.refiner(Q)
                fused = self.out_proj(Q)
                token_out_list.append(fused)
                token_counts.append(G)
                continue

            # For each order and each shift, build tokens and attend
            fused_tokens_sets = []
            G_total = 0

            for ord_i, order in enumerate(orders):
                codes_b = all_codes[ord_i][idx_b]                  # (n_b,)
                order_idx = torch.argsort(codes_b)                 # (n_b,)
                idx_ord = idx_b[order_idx]
                f_ord = feats_f[idx_ord]                           # (n_b, in_ch)
                x_ord = coords_f[idx_ord]                          # (n_b, 3)

                for sh in self.shifts:
                    if sh >= n_b:  # shift too large
                        continue
                    # sub-sequence with shift
                    f_sub  = f_ord[sh:]
                    x_sub  = x_ord[sh:]
                    n_sub  = f_sub.size(0)
                    if n_sub <= 0:
                        continue
                    groups = torch.arange(n_sub, device=device) // self.q_patch
                    G = int(groups[-1].item()) + 1
                    # token center and raw token features
                    Q_xyz = self._grouped_means(x_sub, groups, G)      # (G,3)

                    if self.q_from_x1 and K1_feat_b.numel() > 0:
                        # Build token features from x1 via KNN around Q_xyz
                        k1_xyz_m = K1_coor_b[:, 1:4].to(torch.float32) * grid
                        Q_feat_x1 = self._knn_aggregate(Q_xyz, k1_xyz_m, K1_feat_b, k=self.q_knn)  # (G,C1)
                        Q = self.q_ln_x1(self.query_proj_x1(Q_feat_x1))                            # (G,A)
                    else:
                        # Raw SI features
                        Q_raw = self._grouped_means(f_sub, groups, G)                              # (G,in_ch)
                        Q = self.q_ln_raw(self.query_proj_raw(Q_raw))                               # (G,A)

                    Q = self.refiner(Q)  # local smoothing along token sequence

                    # Attend to x1 / x2 with learned scale gate
                    outs = []
                    scales = F.softmax(self.scale_logits, dim=0)  # (2,)
                    if K1_feat_b.numel() > 0:
                        O1 = self._cross_attend(Q, Q_xyz, K1_feat_b, K1_coor_b, self.key1_proj, self.val1_proj,
                                                self.k1_ln, self.v1_ln, grid)  # (G,A)
                        outs.append(scales[0] * O1)
                    if K2_feat_b.numel() > 0:
                        O2 = self._cross_attend(Q, Q_xyz, K2_feat_b, K2_coor_b, self.key2_proj, self.val2_proj,
                                                self.k2_ln, self.v2_ln, grid)  # (G,A)
                        outs.append(scales[1] * O2)

                    if len(outs) == 0:
                        fused_attn = Q
                    elif len(outs) == 1:
                        fused_attn = outs[0]
                    else:
                        fused_attn = torch.stack(outs, dim=0).sum(dim=0)

                    fused_tokens = self.out_proj(fused_attn)  # (G, C_out)
                    fused_tokens_sets.append(fused_tokens)
                    G_total += G

            if len(fused_tokens_sets) == 0:
                token_out_list.append(None)
                token_counts.append(0)
            else:
                fused_all = torch.cat(fused_tokens_sets, dim=0)  # (G_total, C_out)
                token_out_list.append(fused_all)
                token_counts.append(G_total)

        # --- Pack to (B, C, P) ---
        token_counts_t = torch.tensor(token_counts, device=device, dtype=torch.long)
        P = int(token_counts_t.max().item()) if token_counts_t.numel() > 0 else 0
        C = self.c_out
        padded = feats_f.new_zeros(B, C, P)
        for b in range(B):
            fused = token_out_list[b]
            if fused is None or fused.numel() == 0:
                continue
            g = fused.size(0)
            padded[b, :, :g] = fused.t()
        return padded
