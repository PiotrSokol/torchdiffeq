from .solvers import FixedGridODESolver, RKSymplecticODESolver
from . import rk_common
from .misc import _scaled_dot_product, _convert_to_tensor
from .mathops import explicit_newton, newton_krylov

import torch


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


class DIRK(FixedGridODESolver, RKSymplecticODESolver):

    def __init__(self, func, y0, tableau, step_size=None, grid_constructor=None, **unused_kwargs):
        super.__init__(func, y0, step_size, grid_constructor, tableau=tableau, **unused_kwargs)

    def step_func(self, func, t, dt, y):
        raise NotImplementedError  # TODO

    def root_finder(self, f, x, dtype, device, reltol=torch.tensor(1e-6), atol=torch.tensor(1e-6), maxit=40):
        return newton_krylov(f, x, reltol=torch.tensor(1e-6), atol=torch.tensor(1e-6, ), maxit=40)


class ExplicitJacobianNewton(DIRK):
    def __new__(cls, decoratee):
        cls = type('ExplicitJacobianNewton'+decoratee.__class__.__name__,
                   (ExplicitJacobianNewton, decoratee.__class__),
                   decoratee.__dict__)
        return object.__new__(cls)

    def root_finder(self, f, x, dtype, device, reltol=torch.tensor(1e-6), atol=torch.tensor(1e-6), maxit=40):
        return explicit_newton(f, x, reltol.type(dtype).device, atol.type(dtype).device, maxit)
