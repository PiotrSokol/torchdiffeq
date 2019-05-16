from .solvers import FixedGridODESolver, SymplecticODESolver
from . import rk_common
from .misc import _scaled_dot_product, _convert_to_tensor
# import collections


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in func(t, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4


class DIRK(FixedGridODESolver, SymplecticODESolver):

    def __init__(self, func, y0, tableau, step_size=None, grid_constructor=None, **unused_kwargs):
        super.__init__(func, y0, step_size, grid_constructor, tableau=tableau, **unused_kwargs)

    def step(self, func, y0, f0, t0, dt, tableau):
        """Take an arbitrary Runge-Kutta step and estimate error.

        Args:
            func: Function to evaluate like `func(t, y)` to compute the time derivative
                of `y`.
            y0: Tensor initial value for the state.
            f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
            t0: float64 scalar Tensor giving the initial time.
            dt: float64 scalar Tensor giving the size of the desired time step.
            tableau: optional _ButcherTableau describing how to take the Runge-Kutta
                step.
            name: optional name for the operation.

        Returns:
            Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
            the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
            estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
            calculating these terms.
        """
        dtype = y0[0].dtype
        device = y0[0].device

        t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
        dt = _convert_to_tensor(dt, dtype=dtype, device=device)

        k = tuple(map(lambda x: [x], f0))
        for alpha_i, beta_i in zip(tableau.alpha, tableau.beta):
            ti = t0 + alpha_i * dt
            yi = tuple(y0_ + _scaled_dot_product(dt, beta_i, k_) for y0_, k_ in zip(y0, k))
            tuple(k_.append(f_) for k_, f_ in zip(k, func(ti, yi)))

        if not (tableau.c_sol[-1] == 0 and tableau.c_sol[:-1] == tableau.beta[-1]):
            # This property (true for Dormand-Prince) lets us save a few FLOPs.
            y