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
from typing import Any, TypeAlias

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

    a: Float[Array, "N"] = eqx.field(converter=_float_array)
    b: Float[Array, "N"] = eqx.field(converter=_float_array)
    n: Float[Array, "N"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    @property
    def h(self) -> Float[Array, "N"]:
        """Grid intervals."""
        return (self.b - self.a) / self.n

    @partial(jax.jit)
    def __call__(self, x: Float[Array, "N"]) -> jax.Array:
        """1D-spline interpolation.

        Parameters
        ----------
        x: Array[float, (N,)]
            1-dim x vector (float) at which interplated y-value is evaluated
        """
        h = self.h

        # TODO: consolidate all the f functions into one
        @jax.jit  # type: ignore[misc]
        def f(
            carry: FloatScalar, i1: IntScalar, x: FloatScalar
        ) -> tuple[FloatScalar, FloatScalar]:
            val = self.c[i1 - 1] * _u(i1, self.a[0], h[0], x[0])
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)

        carry, val = lax.scan(lambda s1, i1: f(s1, i1=i1, x=x), 0.0, i1arr)

        return carry


class Spline2DInterpolant(AbstractSplineInterpolant):
    """2D-spline interpolant."""

    a: Float[Array, "N"] = eqx.field(converter=_float_array)
    b: Float[Array, "N"] = eqx.field(converter=_float_array)
    n: Float[Array, "N"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    @property
    def h(self) -> Float[Array, "N"]:
        """Grid intervals."""
        return (self.b - self.a) / self.n

    @partial(jax.jit)
    def __call__(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        """2D-spline interpolation."""
        h = self.h

        # TODO: consolidate all the f functions into one
        @jax.jit  # type: ignore[misc]
        def f(
            carry: FloatScalar,
            i1: IntScalar,
            i2: IntScalar,
            x: FloatScalar,
        ) -> tuple[FloatScalar, FloatScalar]:
            val = (
                self.c[i1 - 1, i2 - 1]
                * _u(i1, self.a[0], h[0], x[0])
                * _u(i2, self.a[1], h[1], x[1])
            )
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)
        i2arr = jnp.arange(1, self.c.shape[1] + 1)

        carry, val = lax.scan(
            lambda s1, i1: lax.scan(lambda s2, i2: f(s2, i1=i1, i2=i2, x=x), s1, i2arr),
            0.0,
            i1arr,
        )

        return carry


class Spline3DInterpolant(AbstractSplineInterpolant):
    """3D-spline interpolant."""

    a: Float[Array, "N"] = eqx.field(converter=_float_array)
    b: Float[Array, "N"] = eqx.field(converter=_float_array)
    n: Float[Array, "N"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    @property
    def h(self) -> Float[Array, "N"]:
        """Grid intervals."""
        return (self.b - self.a) / self.n

    @partial(jax.jit)
    def __call__(self, x: jax.Array) -> jax.Array:
        """3D-spline interpolation."""
        h = self.h

        # TODO: consolidate all the f functions into one
        @jax.jit  # type: ignore[misc]
        def f(
            carry: FloatScalar,
            i1: IntScalar,
            i2: IntScalar,
            i3: IntScalar,
            x: FloatScalar,
        ) -> tuple[FloatScalar, FloatScalar]:
            val = (
                self.c[i1 - 1, i2 - 1, i3 - 1]
                * _u(i1, self.a[0], h[0], x[0])
                * _u(i2, self.a[1], h[1], x[1])
                * _u(i3, self.a[2], h[2], x[2])
            )
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)
        i2arr = jnp.arange(1, self.c.shape[1] + 1)
        i3arr = jnp.arange(1, self.c.shape[2] + 1)

        carry, val = lax.scan(
            lambda s1, i1: lax.scan(
                lambda s2, i2: lax.scan(
                    lambda s3, i3: f(s3, i1=i1, i2=i2, i3=i3, x=x), s2, i3arr
                ),
                s1,
                i2arr,
            ),
            0.0,
            i1arr,
        )

        return carry


class Spline4DInterpolant(AbstractSplineInterpolant):
    """4D-spline interpolant."""

    a: Float[Array, "N"] = eqx.field(converter=_float_array)
    b: Float[Array, "N"] = eqx.field(converter=_float_array)
    n: Float[Array, "N"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    @property
    def h(self) -> Float[Array, "N"]:
        """Grid intervals."""
        return (self.b - self.a) / self.n

    @partial(jax.jit)
    def __call__(self, x: jax.Array) -> jax.Array:
        """4D-spline interpolation."""
        h = self.h

        # TODO: consolidate all the f functions into one
        @jax.jit  # type: ignore[misc]
        def f(
            carry: FloatScalar,
            i1: IntScalar,
            i2: IntScalar,
            i3: IntScalar,
            i4: IntScalar,
            x: FloatScalar,
        ) -> tuple[FloatScalar, FloatScalar]:
            val = (
                self.c[i1 - 1, i2 - 1, i3 - 1, i4 - 1]
                * _u(i1, self.a[0], h[0], x[0])
                * _u(i2, self.a[1], h[1], x[1])
                * _u(i3, self.a[2], h[2], x[2])
                * _u(i4, self.a[3], h[3], x[3])
            )
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)
        i2arr = jnp.arange(1, self.c.shape[1] + 1)
        i3arr = jnp.arange(1, self.c.shape[2] + 1)
        i4arr = jnp.arange(1, self.c.shape[3] + 1)

        carry, val = lax.scan(
            lambda s1, i1: lax.scan(
                lambda s2, i2: lax.scan(
                    lambda s3, i3: lax.scan(
                        lambda s4, i4: f(s4, i1=i1, i2=i2, i3=i3, i4=i4, x=x), s3, i4arr
                    ),
                    s2,
                    i3arr,
                ),
                s1,
                i2arr,
            ),
            0.0,
            i1arr,
        )

        return carry


class Spline5DInterpolant(AbstractSplineInterpolant):
    """5D-spline interpolant."""

    a: Float[Array, "5"] = eqx.field(converter=_float_array)
    b: Float[Array, "5"] = eqx.field(converter=_float_array)
    n: Float[Array, "5"] = eqx.field(converter=_float_array)
    c: Float[Array, "*shape"] = eqx.field(converter=_float_array)

    @property
    def h(self) -> Float[Array, "5"]:
        """Grid intervals."""
        return (self.b - self.a) / self.n

    @partial(jax.jit)
    def __call__(self, x: Float[Array, "5"]) -> Float[Array, ""]:
        """5D-spline interpolation."""
        h = self.h

        # TODO: consolidate all the f functions into one
        @jax.jit  # type: ignore[misc]
        def f(
            carry: FloatScalar,
            i1: IntScalar,
            i2: IntScalar,
            i3: IntScalar,
            i4: IntScalar,
            i5: IntScalar,
            x: Float[Array, "5"],
        ) -> tuple[FloatScalar, FloatScalar]:
            val = (
                self.c[i1 - 1, i2 - 1, i3 - 1, i4 - 1, i5 - 1]
                * _u(i1, self.a[0], h[0], x[0])
                * _u(i2, self.a[1], h[1], x[1])
                * _u(i3, self.a[2], h[2], x[2])
                * _u(i4, self.a[3], h[3], x[3])
                * _u(i5, self.a[4], h[4], x[4])
            )
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)
        i2arr = jnp.arange(1, self.c.shape[1] + 1)
        i3arr = jnp.arange(1, self.c.shape[2] + 1)
        i4arr = jnp.arange(1, self.c.shape[3] + 1)
        i5arr = jnp.arange(1, self.c.shape[4] + 1)

        carry, val = lax.scan(
            lambda s1, i1: lax.scan(
                lambda s2, i2: lax.scan(
                    lambda s3, i3: lax.scan(
                        lambda s4, i4: lax.scan(
                            lambda s5, i5: f(
                                s5, i1=i1, i2=i2, i3=i3, i4=i4, i5=i5, x=x
                            ),
                            s4,
                            i5arr,
                        ),
                        s3,
                        i4arr,
                    ),
                    s2,
                    i3arr,
                ),
                s1,
                i2arr,
            ),
            0.0,
            i1arr,
        )

        return carry
