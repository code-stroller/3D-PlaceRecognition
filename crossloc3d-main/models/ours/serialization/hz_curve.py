import torch
from typing import Optional, Tuple
from .z_order import xyz2key as morton_encode_, key2xyz as morton_decode_
from .hilbert import encode as hilbert_encode_, decode as hilbert_decode_

def xyz2key(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            b: Optional[torch.Tensor] = None, depth: int = 16) -> torch.Tensor:
    """
    Encodes 3D coordinates (x, y, z) to HZ-curve keys (integrating Hilbert and Z-order).
    Optionally prefixes a batch index `b` into the upper bits of the key.
    """
    # Ensure integer type
    x, y, z = x.long(), y.long(), z.long()
    # Determine macro vs micro bit-length (use roughly half for micro-level)
    if depth <= 1:
        micro_bits = depth  # for depth=1, use micro_bits=1 (macro_bits=0)
    else:
        micro_bits = depth // 2  # choose half of the bits for intra-block Morton/Hilbert
    macro_bits = depth - micro_bits  # remaining bits for Hilbert macro-block indexing

    # Split coordinates into macro (high bits) and micro (low bits)
    if micro_bits > 0:
        mask = (1 << micro_bits) - 1
        mx, my, mz = x >> micro_bits, y >> micro_bits, z >> micro_bits          # macro-block coords
        lx, ly, lz = x & mask, y & mask, z & mask                              # intra-block coords
    else:
        # If micro_bits == 0, the entire space is one macro block
        mx, my, mz = x, y, z
        lx = ly = lz = torch.zeros_like(x)

    # Encode macro coordinate with a Hilbert curve (block index in Hilbert order)
    if macro_bits > 0:
        macro_coords = torch.stack([mx, my, mz], dim=-1)  # shape (N, 3)
        macro_key = hilbert_encode_(macro_coords, num_dims=3, num_bits=macro_bits)
    else:
        # If no macro bits, all points are in a single block (index 0)
        macro_key = torch.zeros_like(x)

    # Encode local coordinate in Morton (Z-order) and Hilbert (for potential use)
    local_key_morton = morton_encode_(lx, ly, lz, b=None, depth=micro_bits)
    if micro_bits > 0:
        local_coords = torch.stack([lx, ly, lz], dim=-1)
        local_key_hilbert = hilbert_encode_(local_coords, num_dims=3, num_bits=micro_bits)
    else:
        local_key_hilbert = torch.zeros_like(local_key_morton)

    # Choose Morton vs Hilbert for local key based on macro Hilbert parity (even=Morton, odd=Hilbert)
    use_hilbert = (macro_key & 1).bool()               # boolean mask: True for odd macro_key
    local_key = torch.where(use_hilbert, local_key_hilbert, local_key_morton)

    # Combine macro and local keys: macro_key occupies top 3*micro_bits, local_key in lower bits
    if micro_bits > 0:
        key = (macro_key << (3 * micro_bits)) | local_key
    else:
        key = macro_key  # if micro_bits=0, key is just macro (trivial case)

    # If batch indices are provided, prepend them to the key (shift by total coordinate bits)
    if b is not None:
        b = b.long()
        key = (b << (depth * 3)) | key
    return key

def key2xyz(key: torch.Tensor, depth: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decodes HZ-curve keys back into 3D coordinates (x, y, z) and batch indices.
    """
    # Separate batch bits from the key (upper bits beyond 3*depth)
    b = key >> (depth * 3)
    coord_key = key & ((1 << (depth * 3)) - 1)

    # Compute macro/micro bit split (same as used in encoding)
    if depth <= 1:
        micro_bits = depth
    else:
        micro_bits = depth // 2
    macro_bits = depth - micro_bits

    # Separate macro and local parts of the coordinate key
    if micro_bits > 0:
        macro_key = coord_key >> (3 * micro_bits)
        local_key = coord_key & ((1 << (3 * micro_bits)) - 1)
    else:
        macro_key = coord_key
        local_key = torch.zeros_like(coord_key)

    # Decode macro Hilbert key to get macro-block coordinates (mx, my, mz)
    if macro_bits > 0:
        macro_coords = hilbert_decode_(macro_key, num_dims=3, num_bits=macro_bits)
        mx, my, mz = macro_coords[:, 0], macro_coords[:, 1], macro_coords[:, 2]
    else:
        # All points in one macro block
        mx = my = mz = torch.zeros_like(macro_key)

    # Decode local key to get intra-block coords. Use Morton or Hilbert decoder based on parity
    # (We'll decode both patterns for all points, then select per-point results)
    lx_m, ly_m, lz_m, _ = morton_decode_(local_key, depth=micro_bits)       # Morton decode (returns x,y,z,...)
    if micro_bits > 0:
        local_coords_h = hilbert_decode_(local_key, num_dims=3, num_bits=micro_bits)
        lx_h, ly_h, lz_h = local_coords_h[:, 0], local_coords_h[:, 1], local_coords_h[:, 2]
    else:
        lx_h = ly_h = lz_h = torch.zeros_like(local_key)

    use_hilbert = (macro_key & 1).bool()  # True where block index is odd (used Hilbert locally)
    # Select the correct local coords for each point
    lx = torch.where(use_hilbert, lx_h, lx_m)
    ly = torch.where(use_hilbert, ly_h, ly_m)
    lz = torch.where(use_hilbert, lz_h, lz_m)

    # Reconstruct full coordinates by combining macro (high bits) and local (low bits)
    if micro_bits > 0:
        x = (mx << micro_bits) | lx
        y = (my << micro_bits) | ly
        z = (mz << micro_bits) | lz
    else:
        x, y, z = mx, my, mz  # if no micro bits, macro coords are the full coords

    return x, y, z, b
