# model.py

import torch
import torch.nn as nn
from .point_transformer_v3 import PointTransformerV3
from .netvlad import NetVladWrapper
from .pooling import MAC, SPoC, GeM

class PTV3BackboneWrapper(nn.Module):
    """
    Wraps PointTransformerV3 so that
      input:  (B, N, 3) raw point cloud
      output: (B, D, N) per-point feature
    Parameters for PTv3 (in_channels, enc_depths, etc.) are all
    passed via cfg.backbone_cfg.
    """
    def __init__(self, cfg):
        super().__init__()
        # pick PTv3 init args out of cfg.backbone_cfg
        pt_keys = {
            'in_channels','order','stride','enc_depths','enc_channels',
            'enc_num_head','enc_patch_size','dec_depths','dec_channels',
            'dec_num_head','dec_patch_size','mlp_ratio','qkv_bias',
            'qk_scale','attn_drop','proj_drop','drop_path','pre_norm',
            'shuffle_orders','enable_rpe','enable_flash',
            'upcast_attention','upcast_softmax','cls_mode',
            'pdnorm_bn','pdnorm_ln','pdnorm_decouple','pdnorm_adaptive',
            'pdnorm_affine','pdnorm_conditions'
        }
        pt_cfg = {k: v for k, v in dict(cfg).items() if k in pt_keys}
        self.grid_size = cfg.grid_size
        self.model = PointTransformerV3(**pt_cfg)

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        coords = x.view(-1, 3)  # (B*N, 3)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)
        feats = coords  # use coords as input feature
        data = {
            'coord': coords,
            'grid_size': self.grid_size,
            'batch': batch,
            'feat': feats
        }
        out = self.model(data)      # returns a Point dict
        pf = out['feat']            # (B*N, D)
        return pf.view(B, N, -1)    # (B, N, D)

class Ours(nn.Module):
    """
    Only PTv3-backbone + NetVLAD (or MAC/SPoC/GeM) pooling.
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.backbone_cfg.type == 'PTV3'
        self.backbone = PTV3BackboneWrapper(cfg.backbone_cfg)

        pool_cfg = cfg.pool_cfg
        # make sure in_channels matches backbone output_dim
        # (you may want to read backbone output at runtime or store it in cfg)
        if pool_cfg.type == 'Max':
            self.pool = MAC(pool_cfg)
        elif pool_cfg.type == 'Avg':
            self.pool = SPoC(pool_cfg)
        elif pool_cfg.type == 'GeM':
            self.pool = GeM(pool_cfg)
        elif pool_cfg.type == 'NetVlad':
            self.pool = NetVladWrapper(pool_cfg)
        else:
            raise ValueError(f"Unknown pool type {pool_cfg.type}")

    def forward(self, data):
        # data is (pcd_sparse_list, raw_pcd) but we only need raw_pcd
        _, raw_pcd = data
        # raw_pcd: (B, N, 3)
        x = self.backbone(raw_pcd)   # -> (B, N, D)
        x = x.permute(0, 2, 1)       # -> (B, D, N) for NetVLAD / MAC / etc.
        return self.pool(x)          # -> (B, out_channels)
