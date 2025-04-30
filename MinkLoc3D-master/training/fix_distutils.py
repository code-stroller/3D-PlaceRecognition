# fix_distutils.py
try:
    import distutils.version
except ImportError:
    from setuptools._distutils import version as distutils_version
    import distutils
    distutils.version = distutils_version
else:
    if not hasattr(distutils.version, "LooseVersion"):
        from setuptools._distutils.version import LooseVersion
        distutils.version.LooseVersion = LooseVersion
