from .solvers import FixedGridODESolver, RKSymplecticODESolver
from . import rk_common
from .misc import _scaled_dot_product, _convert_to_tensor
from .mathops import explicit_newton, newton_krylov
from .rk_common import _ButcherTableau
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

    def step_func(self, func, y0, t0, dt, f0=None):
        tableau = self.tableau
        dtype = y0[0].dtype
        device = y0[0].device

        t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
        dt = _convert_to_tensor(dt, dtype=dtype, device=device)

        if f0 is None:
            k = func(t0, y0)
        else:
            k = tuple(map(lambda x: [x], f0))
        for st in range(len(tableau)):  # loop over stages
            alpha_i, beta_i = tableau.alpha[st], tableau.beta[st, :-1].tolist()
            ti = t0 + alpha_i * dt
            yi = tuple(y0_ + _scaled_dot_product(dt, beta_i, k_) for y0_, k_ in zip(y0, k))
            with torch.set_grad_enabled(True):
                Fun = lambda x: tuple(x - yi_ - _scaled_dot_product(dt, tableau.beta[st, st].tolist(), f_)
                                      for yi_, f_ in zip(yi, func(ti, x)))
                yi = self.root_finder(Fun, y0)
                """ the function F to be minimized here is of the form
                F = y - yi - alpha_i*beta[st,st]*f(t, y);
                and the minimization will be performed over y
                """
            tuple(k_.append(f_) for k_, f_ in zip(k, func(ti, yi)))

        yi = tuple(y0_ + _scaled_dot_product(dt, tableau.c_sol, k_) for y0_, k_ in zip(y0, k))

        y1 = yi
        f1 = tuple(k_[-1] for k_ in k)
        # y1_error = tuple(_scaled_dot_product(dt, tableau.c_error, k_) for k_ in k)
        return (y1, f1, k)

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


class ImplicitMidpoint(DIRK):
    """
    The implicit midpoint method.
    Has global error O(h^2)
    Is alpha stable, good for stiff ODEs.
    """

    def step_func(self, func, t, dt, y0):

        dtype = y0[0].dtype
        device = y0[0].device


        yi = tuple(map(lambda x: [x], y0))
        with torch.set_grad_enabled(True):
            Fun = lambda x: tuple(x - yi_ - _scaled_dot_product(dt, [1/2], f_)
                                  for yi_, f_ in zip(yi, func(t + dt/2, x)))
            yi = self.root_finder(Fun, y0, dtype=dtype, device=device)
        return tuple(yi)

    @property
    def order(self):
        return 2

class DIRK3(DIRK):
    """
    A foruth order Diagonally Implicit, symmetric RK method.
    Tableau taken from
    [1]K. Feng and M. Qin, Symplectic Geometric Algorithms for Hamiltonian Systems.
    Berlin, Heidelberg: Springer Berlin Heidelberg, 2010.

    Pages 285-286

    """
    def __init__(self, func, y0, step_size=None, grid_constructor=None, **unused_kwargs):
        a = 1.351207
        tableau =_ButcherTableau(
                                alpha=[1/2*a, 1/2, 1 - a/2],
                                beta=[
                                    [1 / 2 * a],
                                    [a, 1/2 -a],
                                    [a, 1- 2*a, 1/2*a],
                                ],
                                c_sol=[a, 1 - 2*a, a],
                                )
        super.__init__(func, y0, step_size, grid_constructor, tableau=tableau, **unused_kwargs)

    @property
    def order(self):
        return 4
