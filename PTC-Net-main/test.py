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

print(_z_ok, _hil_ok, _hz_ok)