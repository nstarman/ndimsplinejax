"""N-dimensional splines in JAX."""

__all__ = ["__version__", "__version_tuple__", "compute_coeffs", "SplineInterpolant"]

import jaxtyping

with jaxtyping.install_import_hook("ndimsplinejax", "beartype.beartype"):
    from ._version import __version__, __version_tuple__
    from .coeffs import compute_coeffs
    from .core import SplineInterpolant


# Clean up namespace
del jaxtyping
