"""Module for computing the spline coefficients from gridded data.

Compute the coefficients of the N-dimensitonal natural-cubic spline interpolant
defined by Habermann and Kindermann 2007.
"""

__all__ = ["compute_coeffs"]

from itertools import product
from typing import Any, no_type_check

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from scipy import linalg

__author__ = "N.Moteki, (The University of Tokyo, NOAA Earth System Research Lab)."
__maintainer__ = "Nathaniel Starkman @nstarman"


class GridDataInfo(eqx.Module):  # type: ignore[misc]
    """Compute the coefficients.

    Compute the coefficients of the N-dimensitonal natural-cubic spline
    interpolant defined by Habermann and Kindermann 2007 Current code supports
    up to 5 dimensions (N can be either of 1,2,3,4,5).

    Author:
        N.Moteki, (The University of Tokyo, NOAA Earth System Research Lab).

    Assumptions:
        x space (independent variables) is N-dimension Equidistant x-grid in
        each dimension y datum (single real value) is given at each grid point

    User's Inputs:
        a: N-list of lower boundary of x-space [1st-dim, 2nd-dim,...].  b:
        N-list of upper boundary of x-space [1st-dim, 2nd-dim,...].  y_data:
        N-dimensional numpy array of data (the value of dependent variable y) on
        the x-gridpoints.

    Output:
        c_i1...iN: N-dimensional numpy array (dtype=float) of spline
        coeffcieints defined as HK2007 p161.

    Usage:
        >>> from ``SplineCoefs_from_GriddedData`` import SplineCoefs_from_GriddedData
        >>> spline_coeffs = compute_coeffs(a,b,n,y_data)


    Ref.  Habermann, C., & Kindermann, F. (2007). Multidimensional spline
    interpolation: Theory and applications. Computational Economics, 30(2),
    153-169.  Notation is modified by N.Moteki as Note of 2022 September 23-27th

    Created on Fri Oct 21 13:41:11 2022
    """

    a: Float[np.ndarray, "N"] = eqx.field(
        converter=lambda x: np.array(x, dtype=float), static=True
    )
    b: Float[np.ndarray, "N"] = eqx.field(
        converter=lambda x: np.array(x, dtype=float), static=True
    )
    y_data: Float[np.ndarray, "*shape"] = eqx.field(
        converter=lambda x: np.array(x, dtype=float), static=True
    )

    @property
    def N(self) -> int:
        """Dimension of the problem."""
        return len(self.a)

    @property
    def nintervals(self) -> Int[np.ndarray, "{self.N}"]:
        """Number of grid interval n in each dimension."""
        return np.asarray(self.y_data.shape, dtype=int) - 1

    def get_c_shape(self, k: int) -> Int[np.ndarray, "{self.N}"]:
        """Get the shape of the coefficient array.

        Parameters
        ----------
        k : int
            The coefficient index.

        Returns
        -------
        Int[np.ndarray, "{self.N}"]
            The shape of the coefficient array.
        """
        n = self.nintervals
        return np.array([int(n[j]) + (3 if j <= k else 1) for j in range(self.N)])


def compute_coeffs(a: Any, b: Any, y_data: Any) -> Float[Array, "..."]:
    """Compute the coefficients for the spline interpolation."""
    gridinfo = GridDataInfo(a, b, y_data)

    match gridinfo.N:
        case 1:
            coeffs = compute_coeffs_1d(gridinfo)
        case 2:
            coeffs = compute_coeffs_2d(gridinfo)
        case 3:
            coeffs = compute_coeffs_3d(gridinfo)
        case 4:
            coeffs = compute_coeffs_4d(gridinfo)
        case 5:
            coeffs = compute_coeffs_5d(gridinfo)
        case _:
            msg = "N>=6 is unsupported!"
            raise ValueError(msg)

    return coeffs


@no_type_check
def _compute_A(N: np.integer | int) -> Float[np.ndarray, "{N}-1 {N}-1"]:
    """Compute the matrix A."""
    return np.eye(N - 1) * 4 + np.eye(N - 1, k=1) + np.eye(N - 1, k=-1)


@no_type_check
def _compute_B(
    N: np.integer | int, arr: Float[np.ndarray, "..."], cfs: Float[np.ndarray, "..."]
) -> Float[np.ndarray, "{N}-1"]:
    """Compute the vector B."""
    B = np.zeros(N - 1)
    B[0] = arr[1] - cfs[1]
    B[N - 2] = arr[N - 1] - cfs[N + 1]
    B[1 : N - 2] = arr[2 : N - 1]
    return B


@no_type_check
def _compute_coeffs_helper(
    out: Float[np.ndarray, "out"],
    arr: Float[np.ndarray, "..."],
    A: Float[np.ndarray, "{N}-1 {N}-1"],
    N: np.integer | int,
) -> Float[np.ndarray, "out"]:
    """Compute the coefficients helper."""
    out[1] = arr[0] / 6  # c_{2}
    out[N + 1] = arr[N] / 6  # c_{n+2}

    B = _compute_B(N, arr, out)

    out[2 : N + 1] = linalg.solve(A, B)
    out[0] = 2 * out[1] - out[2]
    out[N + 2] = 2 * out[N + 1] - out[N]

    return out


def compute_coeffs_1d(gridinfo: GridDataInfo) -> Float[Array, "..."]:
    """Compute the coefficients for the 1D spline interpolation.

    Parameters
    ----------
    gridinfo : GridDataInfo
        The instance of the class.

    Returns
    -------
    Float[Array, "..."]
        The coefficients.
    """
    n = gridinfo.nintervals[0]
    A = _compute_A(n)
    c_i1 = np.zeros(gridinfo.get_c_shape(0))
    c_i1 = _compute_coeffs_helper(c_i1, gridinfo.y_data, A, n)
    return jnp.asarray(c_i1)


def compute_coeffs_2d(gridinfo: GridDataInfo) -> Float[Array, "..."]:
    """Compute the coefficients for the 2D spline interpolation."""
    nintvl = gridinfo.nintervals

    # 1st dimension
    k = 0
    n = nintvl[k]
    A = _compute_A(n)
    c_i1q2 = np.zeros(gridinfo.get_c_shape(k))
    for q2 in range(nintvl[1] + 1):
        c_i1q2[:, q2] = _compute_coeffs_helper(
            c_i1q2[:, q2], gridinfo.y_data[:, q2], A, n
        )

    # 2nd dimension
    k = 1
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2 = np.zeros(gridinfo.get_c_shape(k))
    for i1 in range(nintvl[0] + 3):
        c_i1i2[i1, :] = _compute_coeffs_helper(c_i1i2[i1, :], c_i1q2[i1, :], A, n)

    return jnp.asarray(c_i1i2)


def compute_coeffs_3d(gridinfo: GridDataInfo) -> Float[Array, "..."]:
    """Compute the coefficients for the 3D spline interpolation."""
    nintvl = gridinfo.nintervals

    # 1st dimension
    k = 0
    n = nintvl[k]
    A = _compute_A(n)
    c_i1q2q3 = np.zeros(gridinfo.get_c_shape(k))
    for q2, q3 in product(range(nintvl[1] + 1), range(nintvl[2] + 1)):
        c_i1q2q3[:, q2, q3] = _compute_coeffs_helper(
            c_i1q2q3[:, q2, q3], gridinfo.y_data[:, q2, q3], A, n
        )

    # 2nd dimension
    k = 1
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2q3 = np.zeros(gridinfo.get_c_shape(k))
    for i1, q3 in product(range(nintvl[0] + 3), range(nintvl[2] + 1)):
        c_i1i2q3[i1, :, q3] = _compute_coeffs_helper(
            c_i1i2q3[i1, :, q3], c_i1q2q3[i1, :, q3], A, n
        )

    # 3rd dimension
    k = 2
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2i3 = np.zeros(gridinfo.get_c_shape(k))
    for i1, i2 in product(range(nintvl[0] + 3), range(nintvl[1] + 3)):
        c_i1i2i3[i1, i2, :] = _compute_coeffs_helper(
            c_i1i2i3[i1, i2, :], c_i1i2q3[i1, i2, :], A, n
        )

    return jnp.asarray(c_i1i2i3)


def compute_coeffs_4d(gridinfo: GridDataInfo) -> Float[Array, "..."]:
    """Compute the coefficients for the 4D spline interpolation."""
    nintvl = gridinfo.nintervals

    k = 0  # 1st dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1q2q3q4 = np.zeros(gridinfo.get_c_shape(k))
    for q2, q3, q4 in product(
        range(nintvl[1] + 1), range(nintvl[2] + 1), range(nintvl[3] + 1)
    ):
        c_i1q2q3q4[:, q2, q3, q4] = _compute_coeffs_helper(
            c_i1q2q3q4[:, q2, q3, q4], gridinfo.y_data[:, q2, q3, q4], A, n
        )

    k = 1  # 2nd dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2q3q4 = np.zeros(gridinfo.get_c_shape(k))
    for i1, q3, q4 in product(
        range(nintvl[0] + 3), range(nintvl[2] + 1), range(nintvl[3] + 1)
    ):
        c_i1i2q3q4[i1, :, q3, q4] = _compute_coeffs_helper(
            c_i1i2q3q4[i1, :, q3, q4], c_i1q2q3q4[i1, :, q3, q4], A, n
        )

    k = 2  # 3rd dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2i3q4 = np.zeros(gridinfo.get_c_shape(k))
    for i1, i2, q4 in product(
        range(nintvl[0] + 3), range(nintvl[1] + 3), range(nintvl[3] + 1)
    ):
        c_i1i2i3q4[i1, i2, :, q4] = _compute_coeffs_helper(
            c_i1i2i3q4[i1, i2, :, q4], c_i1i2q3q4[i1, i2, :, q4], A, n
        )

    k = 3  # 4th dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2i3i4 = np.zeros(gridinfo.get_c_shape(k))
    for i1, i2, i3 in product(
        range(nintvl[0] + 3), range(nintvl[1] + 3), range(nintvl[2] + 3)
    ):
        c_i1i2i3i4[i1, i2, i3, :] = _compute_coeffs_helper(
            c_i1i2i3i4[i1, i2, i3, :], c_i1i2i3q4[i1, i2, i3, :], A, n
        )

    return jnp.asarray(c_i1i2i3i4)


def compute_coeffs_5d(gridinfo: GridDataInfo) -> Float[Array, "..."]:
    """Compute the coefficients for the 5D spline interpolation."""
    nintvl = gridinfo.nintervals

    k = 0  # 1st dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1q2q3q4q5 = np.zeros(gridinfo.get_c_shape(k))
    for q2, q3, q4, q5 in product(
        range(nintvl[1] + 1),
        range(nintvl[2] + 1),
        range(nintvl[3] + 1),
        range(nintvl[4] + 1),
    ):
        c_i1q2q3q4q5[:, q2, q3, q4, q5] = _compute_coeffs_helper(
            c_i1q2q3q4q5[:, q2, q3, q4, q5], gridinfo.y_data[:, q2, q3, q4, q5], A, n
        )

    k = 1  # 2nd dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2q3q4q5 = np.zeros(gridinfo.get_c_shape(k))
    for i1, q3, q4, q5 in product(
        range(nintvl[0] + 3),
        range(nintvl[2] + 1),
        range(nintvl[3] + 1),
        range(nintvl[4] + 1),
    ):
        c_i1i2q3q4q5[i1, :, q3, q4, q5] = _compute_coeffs_helper(
            c_i1i2q3q4q5[i1, :, q3, q4, q5], c_i1q2q3q4q5[i1, :, q3, q4, q5], A, n
        )

    k = 2  # 3rd dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2i3q4q5 = np.zeros(gridinfo.get_c_shape(k))
    for i1, i2, q4, q5 in product(
        range(nintvl[0] + 3),
        range(nintvl[1] + 3),
        range(nintvl[3] + 1),
        range(nintvl[4] + 1),
    ):
        c_i1i2i3q4q5[i1, i2, :, q4, q5] = _compute_coeffs_helper(
            c_i1i2i3q4q5[i1, i2, :, q4, q5], c_i1i2q3q4q5[i1, i2, :, q4, q5], A, n
        )

    k = 3  # 4th dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2i3i4q5 = np.zeros(gridinfo.get_c_shape(k))
    for i1, i2, i3, q5 in product(
        range(nintvl[0] + 3),
        range(nintvl[1] + 3),
        range(nintvl[2] + 3),
        range(nintvl[4] + 1),
    ):
        c_i1i2i3i4q5[i1, i2, i3, :, q5] = _compute_coeffs_helper(
            c_i1i2i3i4q5[i1, i2, i3, :, q5], c_i1i2i3q4q5[i1, i2, i3, :, q5], A, n
        )

    k = 4  # 5th dimension
    n = nintvl[k]
    A = _compute_A(n)
    c_i1i2i3i4i5 = np.zeros(gridinfo.get_c_shape(k))
    for i1, i2, i3, i4 in product(
        range(nintvl[0] + 3),
        range(nintvl[1] + 3),
        range(nintvl[2] + 3),
        range(nintvl[3] + 3),
    ):
        c_i1i2i3i4i5[i1, i2, i3, i4, :] = _compute_coeffs_helper(
            c_i1i2i3i4i5[i1, i2, i3, i4, :],
            c_i1i2i3i4q5[i1, i2, i3, i4, :],
            A,
            n,
        )

    return jnp.asarray(c_i1i2i3i4i5)
