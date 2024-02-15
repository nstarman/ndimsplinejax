"""N-dimensional spline interpolant."""

__all__ = [
    "AbstractSplineInterpolant",
    "spline_interpolant",
    "Spline1DInterpolant",
    "Spline2DInterpolant",
    "Spline3DInterpolant",
    "Spline4DInterpolant",
    "Spline5DInterpolant",
]

from functools import partial
from typing import Any, ClassVar, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

FloatScalar: TypeAlias = Float[Array, ""]
IntScalar: TypeAlias = Int[Array, ""]


def _u(
    ii: IntScalar,
    aa: FloatScalar | IntScalar,
    hh: FloatScalar | IntScalar,
    xx: FloatScalar,
) -> FloatScalar:
    t = jnp.abs((xx - aa) / hh + 2 - ii)
    return lax.cond(
        t <= 1,
        lambda t: 4.0 - 6.0 * t**2 + 3.0 * t**3,
        lambda t: (2.0 - t) ** 3,
        t,
    ) * jnp.heaviside(2.0 - t, 1.0)


def _float_array(x: Any) -> Float[Array, "..."]:
    return jnp.asarray(x, dtype=jax.dtypes.canonicalize_dtype(float))


class AbstractSplineInterpolant(eqx.Module):  # type: ignore[misc]
    """Abstract Spline Interpolant.

    Auto-differencible and Jittable N-dimensitonal spline interpolant using
    Google/JAX Current code supports only 3 and 4 dimensions (N=3 or 4), which
    are used for CAS data analysis

    Author:
        N.Moteki, (The University of Tokyo, NOAA Earth System Research Lab).

    Assumptions:
        x space (independent variables) is N-dimension Equidistant x-grid in
        each dimension y datum (single real value) is given at each grid point

    User's Inputs:
        a: N-list of lower boundary of x-space [1st-dim, 2nd-dim,...].  b:
        N-list of upper boundary of x-space [1st-dim, 2nd-dim,...].  n: N-list
        of the number of grid intervals in each dimension.  c: N-dimensional
        numpy array (dtype=float) of spline coeffcieints computed using
        "SplineCoefs_from_GriddedData" module

    Output:
        s3D(x): Autodifferencible and jittable spline interpolant for 3-dim x
        input vector s4D(x): Autodifferencible and jittable spline interpolant
        for 4-dim x input vector.

    Usage:
        from SplineInterpolant import SplineInterpolant  # import this module
        spline= SplineInterpolant(a,b,n,c_i1...iN) # constructor y=
        spline.sND(x) # evaluate the interpolated y value at the input x vector,
        where the sND is s3D (if N=3) or s4D (if N=4).  spline.sND is a jittable
        and auto-differentiable function with respect to x

    Ref.  Habermann, C., & Kindermann, F. (2007). Multidimensional spline
    interpolation: Theory and applications. Computational Economics, 30(2),
    153-169.  Notation is modified by N.Moteki as Note of 2022 September 23-27th

    Created on Fri Oct 21 13:41:11 2022

    @author: moteki
    """

    a: eqx.AbstractVar[Float[Array, "{self.ndim}"]]
    b: eqx.AbstractVar[Float[Array, "{self.ndim}"]]
    n: eqx.AbstractVar[Float[Array, "{self.ndim}"]]

    ndim: eqx.AbstractClassVar[int]

    @property
    def h(self) -> Float[Array, "{self.ndim}"]:
        """Grid intervals."""
        return (self.b - self.a) / self.n

    @partial(jax.jit, static_argnames=("method",))
    def __call__(
        self,
        x: Float[Array, "5"],
        method: Literal["vmap", "nested_scan", "single_scan"] = "vmap",
    ) -> Float[Array, ""]:
        """5D-spline interpolation."""
        methods = ("vmap", "nested_scan", "single_scan")
        return lax.switch(
            methods.index(method),
            (
                self._evaluate_vmap,
                self._evaluate_nested_scan,
                self._evaluate_single_scan,
            ),
            x,
        )


def spline_interpolant(
    a: Any, b: Any, n: Any, coeffs: Any
) -> AbstractSplineInterpolant:
    """Return spline interpolant."""
    N = len(coeffs.shape)
    match N:
        case 1:
            return Spline1DInterpolant(a, b, n, coeffs)
        case 2:
            return Spline2DInterpolant(a, b, n, coeffs)
        case 3:
            return Spline3DInterpolant(a, b, n, coeffs)
        case 4:
            return Spline4DInterpolant(a, b, n, coeffs)
        case 5:
            return Spline5DInterpolant(a, b, n, coeffs)
        case _:
            msg = f"Unsupported number of dimensions: {N}"
            raise ValueError(msg)


#####################################################################


class Spline1DInterpolant(AbstractSplineInterpolant):
    """1D-spline interpolant."""

    a: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    b: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    n: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    ndim: ClassVar[int] = 1  # type: ignore[misc]

    @partial(jax.jit)
    def _evaluate_nested_scan(self, x: Float[Array, "{self.ndim}"]) -> jax.Array:
        """1D-spline interpolation.

        Parameters
        ----------
        x: Array[float, (N,)]
            1-dim x vector (float) at which interplated y-value is evaluated
        """
        a, h = self.a, self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar, idx: tuple[IntScalar], x: FloatScalar
        ) -> tuple[FloatScalar, FloatScalar]:
            val = self.c[idx[0]] * _u(idx[0] + 1, a[0], h[0], x[0])
            carry += val
            return carry, val

        carry, _ = lax.scan(
            lambda s1, i0: single(s1, idx=(i0,), x=x), 0.0, jnp.arange(shape[0])
        )

        return carry

    _evaluate_single_scan = _evaluate_nested_scan  # only for 1D spline

    @partial(jax.jit)
    def _evaluate_vmap(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """1D-spline interpolation."""
        h = self.h
        shape = self.c.shape

        @partial(jax.vmap, in_axes=(0, None))  # Vectorize over the indices
        def value_at_index(flat_index: IntScalar, x: Float[Array, "1"]) -> FloatScalar:
            return self.c[flat_index] * _u(flat_index + 1, self.a[0], h[0], x[0])

        return jnp.sum(value_at_index(jnp.arange(self.c.size), x))


class Spline2DInterpolant(AbstractSplineInterpolant):
    """2D-spline interpolant."""

    a: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    b: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    n: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    ndim: ClassVar[int] = 2  # type: ignore[misc]

    @partial(jax.jit)
    def _evaluate_nested_scan(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """2D-spline interpolation."""
        a, h = self.a, self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar, idx: tuple[IntScalar, IntScalar], x: FloatScalar
        ) -> tuple[FloatScalar, FloatScalar]:
            val = (
                self.c[idx]
                * _u(idx[0] + 1, a[0], h[0], x[0])
                * _u(idx[1] + 1, a[1], h[1], x[1])
            )
            carry += val
            return carry, val

        carry, _ = lax.scan(
            lambda s0, i0: lax.scan(
                lambda s1, i1: single(s1, idx(i0, i1), x=x), s0, jnp.arange(shape[1])
            ),
            0.0,
            jnp.arange(shape[0]),
        )

        return carry

    @partial(jax.jit)
    def _evaluate_single_scan(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """2D-spline interpolation.

        .. warning::

            This method can slower than the `evaluate` method, which uses
            a nested `lax.scan` function.
        """
        a, h = self.a, self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar, flat_index: IntScalar
        ) -> tuple[FloatScalar, FloatScalar]:
            # Calculate the original multi-dimensional indices
            idx = jnp.unravel_index(flat_index, shape)
            # Calculate the value of the spline at these indices
            val = (
                self.c[idx]
                * _u(idx[0] + 1, a[0], h[0], x[0])
                * _u(idx[1] + 1, a[1], h[1], x[1])
            )
            carry += val
            return carry, val

        carry, _ = lax.scan(single, 0.0, jnp.arange(self.c.size))
        return carry

    @partial(jax.jit)
    def _evaluate_vmap(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """5D-spline interpolation."""
        a, h = self.a, self.h
        shape = self.c.shape

        @partial(jax.vmap, in_axes=(0, None))  # Vectorize over the indices
        def value_at_index(flat_index: IntScalar, x: Float[Array, "2"]) -> FloatScalar:
            # Calculate the original multi-dimensional indices
            idx = jnp.unravel_index(flat_index, shape)
            # Calculate the value of the spline at these indices
            return (
                self.c[idx]
                * _u(idx[0] + 1, a[0], h[0], x[0])
                * _u(idx[1] + 1, a[1], h[1], x[1])
            )

        return jnp.sum(value_at_index(jnp.arange(self.c.size), x))


class Spline3DInterpolant(AbstractSplineInterpolant):
    """3D-spline interpolant."""

    a: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    b: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    n: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    ndim: ClassVar[int] = 3  # type: ignore[misc]

    @partial(jax.jit)
    def _evaluate_nested_scan(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """3D-spline interpolation."""
        a, h = self.a, self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar,
            idx: tuple[IntScalar, IntScalar, IntScalar],
            x: FloatScalar,
        ) -> tuple[FloatScalar, FloatScalar]:
            val = (
                self.c[idx]
                * _u(idx[0] + 1, a[0], h[0], x[0])
                * _u(idx[1] + 1, a[1], h[1], x[1])
                * _u(idx[2] + 1, a[2], h[2], x[2])
            )
            carry += val
            return carry, val

        carry, _ = lax.scan(
            lambda s1, i0: lax.scan(
                lambda s2, i1: lax.scan(
                    lambda s3, i2: single(s3, idx=(i0, i1, i2), x=x),
                    s2,
                    jnp.arange(shape[2]),
                ),
                s1,
                jnp.arange(shape[1]),
            ),
            0.0,
            jnp.arange(shape[0]),
        )

        return carry

    @partial(jax.jit)
    def _evaluate_single_scan(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """3D-spline interpolation.

        .. warning::

            This method can slower than the `evaluate` method, which uses
            a nested `lax.scan` function.
        """
        h = self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar, flat_index: IntScalar
        ) -> tuple[FloatScalar, FloatScalar]:
            # Calculate the original multi-dimensional indices
            idx = jnp.unravel_index(flat_index, shape)
            # Calculate the value of the spline at these indices
            val = (
                self.c[idx]
                * _u(idx[0] + 1, self.a[0], h[0], x[0])
                * _u(idx[1] + 1, self.a[1], h[1], x[1])
                * _u(idx[2] + 1, self.a[2], h[2], x[2])
            )
            carry += val
            return carry, val

        carry, _ = lax.scan(single, 0.0, jnp.arange(self.c.size))
        return carry

    @partial(jax.jit)
    def _evaluate_vmap(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """3D-spline interpolation."""
        h = self.h
        shape = self.c.shape

        @partial(jax.vmap, in_axes=(0, None))  # Vectorize over the indices
        def value_at_index(flat_index: IntScalar, x: Float[Array, "5"]) -> FloatScalar:
            # Calculate the original multi-dimensional indices
            idx = jnp.unravel_index(flat_index, shape)
            # Calculate the value of the spline at these indices
            return (
                self.c[idx]
                * _u(idx[0] + 1, self.a[0], h[0], x[0])
                * _u(idx[1] + 1, self.a[1], h[1], x[1])
                * _u(idx[2] + 1, self.a[2], h[2], x[2])
            )

        return jnp.sum(value_at_index(jnp.arange(self.c.size), x))


class Spline4DInterpolant(AbstractSplineInterpolant):
    """4D-spline interpolant."""

    a: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    b: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    n: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    ndim: ClassVar[int] = 4  # type: ignore[misc]

    @partial(jax.jit)
    def _evaluate_nested_scan(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """4D-spline interpolation."""
        a, h = self.a, self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar,
            idx: tuple[IntScalar, IntScalar, IntScalar, IntScalar],
            x: Float[Array, "5"],
        ) -> tuple[FloatScalar, FloatScalar]:
            val = (
                self.c[idx]
                * _u(idx[0] + 1, a[0], h[0], x[0])
                * _u(idx[1] + 1, a[1], h[1], x[1])
                * _u(idx[2] + 1, a[2], h[2], x[2])
                * _u(idx[3] + 1, a[3], h[3], x[3])
            )
            carry += val
            return carry, val

        carry, _ = lax.scan(
            lambda s1, i0: lax.scan(
                lambda s2, i1: lax.scan(
                    lambda s3, i2: lax.scan(
                        lambda s4, i3: single(s4, idx=(i0, i1, i2, i3), x=x),
                        s3,
                        jnp.arange(shape[3]),
                    ),
                    s2,
                    jnp.arange(shape[2]),
                ),
                s1,
                jnp.arange(shape[1]),
            ),
            0.0,
            jnp.arange(shape[0]),
        )

        return carry

    @partial(jax.jit)
    def _evaluate_single_scan(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """4D-spline interpolation.

        .. warning::

            This method can slower than the `evaluate` method, which uses
            a nested `lax.scan` function.
        """
        h = self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar, flat_index: IntScalar
        ) -> tuple[FloatScalar, FloatScalar]:
            # Calculate the original multi-dimensional indices
            idx = jnp.unravel_index(flat_index, shape)
            # Calculate the value of the spline at these indices
            val = (
                self.c[idx]
                * _u(idx[0] + 1, self.a[0], h[0], x[0])
                * _u(idx[1] + 1, self.a[1], h[1], x[1])
                * _u(idx[2] + 1, self.a[2], h[2], x[2])
                * _u(idx[3] + 1, self.a[3], h[3], x[3])
            )
            carry += val
            return carry, val

        carry, _ = lax.scan(single, 0.0, jnp.arange(self.c.size))
        return carry

    @partial(jax.jit)
    def _evaluate_vmap(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """4D-spline interpolation."""
        h = self.h
        shape = self.c.shape

        @partial(jax.vmap, in_axes=(0, None))  # Vectorize over the indices
        def value_at_index(flat_index: IntScalar, x: Float[Array, "4"]) -> FloatScalar:
            # Calculate the original multi-dimensional indices
            idx = jnp.unravel_index(flat_index, shape)
            # Calculate the value of the spline at these indices
            return (
                self.c[idx]
                * _u(idx[0] + 1, self.a[0], h[0], x[0])
                * _u(idx[1] + 1, self.a[1], h[1], x[1])
                * _u(idx[2] + 1, self.a[2], h[2], x[2])
                * _u(idx[3] + 1, self.a[3], h[3], x[3])
            )

        return jnp.sum(value_at_index(jnp.arange(self.c.size), x))


class Spline5DInterpolant(AbstractSplineInterpolant):
    """5D-spline interpolant."""

    a: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    b: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    n: Float[Array, "{self.ndim}"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    ndim: ClassVar[int] = 5  # type: ignore[misc]

    @partial(jax.jit)
    def _evaluate_nested_scan(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """5D-spline interpolation."""
        a, h = self.a, self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar,
            idx: tuple[IntScalar, IntScalar, IntScalar, IntScalar, IntScalar],
            x: Float[Array, "5"],
        ) -> tuple[FloatScalar, FloatScalar]:
            val = (
                self.c[idx]
                * _u(idx[0] + 1, a[0], h[0], x[0])
                * _u(idx[1] + 1, a[1], h[1], x[1])
                * _u(idx[2] + 1, a[2], h[2], x[2])
                * _u(idx[3] + 1, a[3], h[3], x[3])
                * _u(idx[4] + 1, a[4], h[4], x[4])
            )
            carry += val
            return carry, val

        carry, _ = lax.scan(
            lambda s1, i0: lax.scan(
                lambda s2, i1: lax.scan(
                    lambda s3, i2: lax.scan(
                        lambda s4, i3: lax.scan(
                            lambda s5, i4: single(s5, idx=(i0, i1, i2, i3, i4), x=x),
                            s4,
                            jnp.arange(shape[4]),
                        ),
                        s3,
                        jnp.arange(shape[3]),
                    ),
                    s2,
                    jnp.arange(shape[2]),
                ),
                s1,
                jnp.arange(shape[1]),
            ),
            0.0,
            jnp.arange(shape[0]),
        )

        return carry

    @partial(jax.jit)
    def _evaluate_single_scan(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """5D-spline interpolation.

        .. warning::

            This method can slower than the `evaluate` method, which uses
            a nested `lax.scan` function.
        """
        h = self.h
        shape = self.c.shape

        def single(
            carry: FloatScalar, flat_index: IntScalar
        ) -> tuple[FloatScalar, FloatScalar]:
            # Calculate the original multi-dimensional indices
            idx = jnp.unravel_index(flat_index, shape)
            # Calculate the value of the spline at these indices
            val = (
                self.c[idx]
                * _u(idx[0] + 1, self.a[0], h[0], x[0])
                * _u(idx[1] + 1, self.a[1], h[1], x[1])
                * _u(idx[2] + 1, self.a[2], h[2], x[2])
                * _u(idx[3] + 1, self.a[3], h[3], x[3])
                * _u(idx[4] + 1, self.a[4], h[4], x[4])
            )
            carry += val
            return carry, val

        carry, _ = lax.scan(single, 0.0, jnp.arange(self.c.size))
        return carry

    @partial(jax.jit)
    def _evaluate_vmap(self, x: Float[Array, "{self.ndim}"]) -> Float[Array, ""]:
        """5D-spline interpolation."""
        h = self.h
        shape = self.c.shape

        @partial(jax.vmap, in_axes=(0, None))  # Vectorize over the indices
        def value_at_index(flat_index: IntScalar, x: Float[Array, "5"]) -> FloatScalar:
            # Calculate the original multi-dimensional indices
            idx = jnp.unravel_index(flat_index, shape)
            # Calculate the value of the spline at these indices
            return (
                self.c[idx]
                * _u(idx[0] + 1, self.a[0], h[0], x[0])
                * _u(idx[1] + 1, self.a[1], h[1], x[1])
                * _u(idx[2] + 1, self.a[2], h[2], x[2])
                * _u(idx[3] + 1, self.a[3], h[3], x[3])
                * _u(idx[4] + 1, self.a[4], h[4], x[4])
            )

        return jnp.sum(value_at_index(jnp.arange(self.c.size), x))
