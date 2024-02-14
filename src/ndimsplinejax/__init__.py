"""N-dimensional splines in JAX."""

__all__ = [
    "__version__",
    "__version_tuple__",
    "compute_coeffs",
    "AbstractSplineInterpolant",
    "spline_interpolant",
    "Spline1DInterpolant",
    "Spline2DInterpolant",
    "Spline3DInterpolant",
    "Spline4DInterpolant",
    "Spline5DInterpolant",
]

import jaxtyping

with jaxtyping.install_import_hook("ndimsplinejax", "beartype.beartype"):
    from ._version import __version__, __version_tuple__
    from .coeffs import compute_coeffs
    from .core import (
        AbstractSplineInterpolant,
        Spline1DInterpolant,
        Spline2DInterpolant,
        Spline3DInterpolant,
        Spline4DInterpolant,
        Spline5DInterpolant,
        spline_interpolant,
    )


# Clean up namespace
del jaxtyping
