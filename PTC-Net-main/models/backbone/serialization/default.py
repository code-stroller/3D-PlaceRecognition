import torch
from .z_order import xyz2key as z_order_encode_, key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_, decode as hilbert_decode_
from .hz_curve import xyz2key as hz_encode_, key2xyz as hz_decode_


@torch.inference_mode()
def encode(
    grid_coord: torch.Tensor,
    batch: torch.Tensor = None,
    depth: int = 16,
    order: str = "z"
) -> torch.Tensor:
    """
    Encode 3D coords into 64-bit serialization keys.
    Supported orders: 'z', 'z-trans', 'hilbert', 'hilbert-trans', 'hz', 'hz-trans'.
    """
    assert order in {
        "z", "z-trans",
        "hilbert", "hilbert-trans",
        "hz", "hz-trans"
    }, f"Unknown serialization order: {order!r}"

    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode_(grid_coord[:, [1, 0, 2]], num_dims=3, num_bits=depth)
    elif order == "hz":
        x = grid_coord[:, 0].long()
        y = grid_coord[:, 1].long()
        z = grid_coord[:, 2].long()
        code = hz_encode_(x, y, z, b=None, depth=depth)
    elif order == "hz-trans":
        x = grid_coord[:, 1].long()
        y = grid_coord[:, 0].long()
        z = grid_coord[:, 2].long()
        code = hz_encode_(x, y, z, b=None, depth=depth)
    else:
        raise NotImplementedError(f"Serialization order {order!r} not implemented")

    if batch is not None:
        batch = batch.long()
        code = (batch << (depth * 3)) | code

    return code


@torch.inference_mode()
def decode(
    key: torch.Tensor,
    depth: int = 16,
    order: str = "z"
) -> (torch.Tensor, torch.Tensor):
    """
    Decode 64-bit keys back into 3D coords and batch indices.
    Supported orders: 'z', 'hilbert', 'hz', 'hz-trans'.
    """
    assert order in {"z", "hilbert", "hz", "hz-trans"}, \
        f"Unknown serialization order: {order!r}"

    batch = key >> (depth * 3)
    coord_key = key & ((1 << (depth * 3)) - 1)

    if order == "z":
        coords = z_order_decode(coord_key, depth=depth)
    elif order == "hilbert":
        coords = hilbert_decode_(coord_key, num_dims=3, num_bits=depth)
    elif order == "hz":
        x, y, z, _ = hz_decode_(coord_key, depth=depth)
        coords = torch.stack([x, y, z], dim=-1)
    elif order == "hz-trans":
        x, y, z, _ = hz_decode_(coord_key, depth=depth)
        coords = torch.stack([y, x, z], dim=-1)
    else:
        raise NotImplementedError(f"Serialization order {order!r} not implemented")

    return coords, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    x = grid_coord[:, 0].long()
    y = grid_coord[:, 1].long()
    z = grid_coord[:, 2].long()
    return z_order_encode_(x, y, z, b=None, depth=depth)


def z_order_decode(code: torch.Tensor, depth: int) -> torch.Tensor:
    x, y, z, _ = z_order_decode_(code, depth=depth)
    return torch.stack([x, y, z], dim=-1)


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16) -> torch.Tensor:
    return hilbert_decode_(code, num_dims=3, num_bits=depth)


def hz_curve_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    x = grid_coord[:, 0].long()
    y = grid_coord[:, 1].long()
    z = grid_coord[:, 2].long()
    return hz_encode_(x, y, z, b=None, depth=depth)


def hz_curve_decode(code: torch.Tensor, depth: int = 16) -> torch.Tensor:
    x, y, z, _ = hz_decode_(code, depth=depth)
    return torch.stack([x, y, z], dim=-1)
