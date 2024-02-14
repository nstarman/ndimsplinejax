"""Test 5-dimensional grid.

A example script for illustaring the usage of "SplineCoefs_from_GriddedData" and
"SplineInterpolant modules" to obtain jittable and auto-differentible
multidimentional spline interpolant.

Created on Fri Oct 21 12:29:47 2022

@author: moteki
"""

import jax.numpy as jnp
from jax import grad

from ndimsplinejax import compute_coeffs, spline_interpolant


def test_5d() -> None:
    """Test 5-dimensional grid."""
    #### synthetic data for demonstration (5-dimension) ####
    a = [0, 0, 0, 0, 0]
    b = [1, 2, 3, 4, 5]
    n = [10, 10, 10, 10, 10]
    N = len(a)

    x_grid = tuple(jnp.linspace(a[j], b[j], n[j] + 1) for j in range(N))
    grid_shape = tuple(n[j] + 1 for j in range(N))

    # Assuming x_grid is a list of 1D arrays
    grids = jnp.meshgrid(*x_grid, indexing="ij")

    # Apply the sin function to each grid and reduce by multiplication
    y_data = jnp.prod(jnp.asarray([jnp.sin(grid) for grid in grids]), axis=0)

    # compute spline coefficients from the gridded data
    coeffs = compute_coeffs(a, b, y_data)

    # compute the jittable and auto-differentiable spline interpolant using the
    # coefficient.
    spline = spline_interpolant(a, b, n, coeffs)

    # give a particular x-coordinate for function evaluation
    x = jnp.array([0.7, 1.0, 1.5, 2.0, 2.5])

    y = spline(x)
    assert jnp.isfinite(y).all()

    ds5d = grad(spline)
    grady = ds5d(x)
    assert jnp.isfinite(grady).all()
