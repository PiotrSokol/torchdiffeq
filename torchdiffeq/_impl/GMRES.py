from krypy import utils
import warnings
import numpy
import scipy


class LinearSystem(object):
    def __init__(self, A, b,
                 M=None,
                 Minv=None,
                 Ml=None,
                 Mr=None,
                 ip_B=None,
                 normal=None,
                 self_adjoint=False,
                 positive_definite=False,
                 exact_solution=None
                 ):
        r'''Representation of a (preconditioned) linear system.

        Represents a linear system

        .. math::

          Ax=b

        or a preconditioned linear system

        .. math::

          M M_l A M_r y = M M_l b
          \quad\text{with}\quad x=M_r y.

        :param A: a linear operator on :math:`\mathbb{C}^N` (has to be
          compatible with :py:meth:`~krypy.utils.get_linearoperator`).
        :param b: the right hand side in :math:`\mathbb{C}^N`, i.e.,
          ``b.shape == (N, 1)``.
        :param M: (optional) a self-adjoint and positive definite
          preconditioner, linear operator on :math:`\mathbb{C}^N` with respect
          to the inner product defined by ``ip_B``. This preconditioner changes
          the inner product to
          :math:`\langle x,y\rangle_M = \langle Mx,y\rangle` where
          :math:`\langle \cdot,\cdot\rangle` is the inner product defined by
          the parameter ``ip_B``. Defaults to the identity.
        :param Minv: (optional) the inverse of the preconditioner provided by
          ``M``. This operator is needed, e.g., for orthonormalizing vectors
          for the computation of Ritz vectors in deflated methods.
        :param Ml: (optional) left preconditioner, linear operator on
          :math:`\mathbb{C}^N`. Defaults to the identity.
        :param Mr: (optional) right preconditioner, linear operator on
          :math:`\mathbb{C}^N`. Defaults to the identity.
        :param ip_B: (optional) defines the inner product, see
          :py:meth:`~krypy.utils.inner`.
        :param normal: (bool, optional) Is :math:`M_l A M_r` normal
          in the inner product defined by ``ip_B``? Defaults to ``False``.
        :param self_adjoint: (bool, optional) Is :math:`M_l A M_r` self-adjoint
          in the inner product defined by ``ip_B``? ``self_adjoint=True``
          also sets ``normal=True``. Defaults to ``False``.
        :param positive_definite: (bool, optional) Is :math:`M_l A M_r`
          positive (semi-)definite with respect to the inner product defined by
          ``ip_B``? Defaults to ``False``.
        :param exact_solution: (optional) If an exact solution :math:`x` is
          known, it can be provided as a ``numpy.array`` with
          ``exact_solution.shape == (N,1)``. Then error norms can be computed
          (for debugging or research purposes). Defaults to ``None``.
        '''
        self.N = N = len(b)
        '''Dimension :math:`N` of the space :math:`\mathbb{C}^N` where the
        linear system is defined.'''
        shape = (N, N)

        # init linear operators
        self.A = utils.get_linearoperator(shape, A)
        self.M = utils.get_linearoperator(shape, M)
        self.Minv = utils.get_linearoperator(shape, Minv)
        self.Ml = utils.get_linearoperator(shape, Ml)
        self.Mr = utils.get_linearoperator(shape, Mr)
        self.MlAMr = self.Ml*self.A*self.Mr
        try:
            self.ip_B = utils.get_linearoperator(shape, ip_B)
        except TypeError:
            self.ip_B = ip_B

        # process vectors
        self.flat_vecs, (self.b, self.exact_solution) = \
            utils.shape_vecs(b, exact_solution)

        # store properties of operators
        self.self_adjoint = self_adjoint

        # automatically set normal=True if self_adjoint==True
        if self_adjoint:
            if normal is not None and not normal:
                warnings.warn('Setting normal=True because '
                              'self_adjoint=True is provided.')
            normal = True
        if normal is None:
            normal = False
        self.normal = normal

        self.positive_definite = positive_definite
        if self_adjoint and not normal:
            raise utils.ArgumentError('self-adjointness implies normality')

        # get common dtype
        self.dtype = utils.find_common_dtype(self.A, self.b, self.M,
                                             self.Ml, self.Mr, self.ip_B)

        # Compute M^{-1}-norm of M*Ml*b.
        self.Mlb = self.Ml*self.b
        self.MMlb = self.M*self.Mlb
        self.MMlb_norm = utils.norm(self.Mlb, self.MMlb, ip_B=self.ip_B)
        '''Norm of the right hand side.

        .. math::

          \|M M_l b\|_{M^{-1}}
        '''

    def get_residual(self, z, compute_norm=False):
        r'''Compute residual.

        For a given :math:`z\in\mathbb{C}^N`, the residual

        .. math::

          r = M M_l ( b - A z )

        is computed. If ``compute_norm == True``, then also the absolute
        residual norm

        .. math::

          \| M M_l (b-Az)\|_{M^{-1}}

        is computed.

        :param z: approximate solution with ``z.shape == (N, 1)``.
        :param compute_norm: (bool, optional) pass ``True`` if also the norm
          of the residual should be computed.
        '''
        if z is None:
            if compute_norm:
                return self.MMlb, self.Mlb, self.MMlb_norm
            return self.MMlb, self.Mlb
        r = self.b - self.A*z
        Mlr = self.Ml*r
        MMlr = self.M*Mlr
        if compute_norm:
            return MMlr, Mlr, utils.norm(Mlr, MMlr, ip_B=self.ip_B)
        return MMlr, Mlr

    def get_ip_Minv_B(self):
        '''Returns the inner product that is implicitly used with the positive
        definite preconditioner ``M``.'''
        if not isinstance(self.M, utils.IdentityLinearOperator):
            if isinstance(self.Minv, utils.IdentityLinearOperator):
                raise utils.ArgumentError(
                    'Minv has to be provided for the evaluation of the inner '
                    'product that is implicitly defined by M.')
            if isinstance(self.ip_B, utils.LinearOperator):
                return self.Minv*self.ip_B
            else:
                return lambda x, y: self.ip_B(x, self.Minv*y)
        return self.ip_B

    def __repr__(self):
        ret = 'LinearSystem {\n'

        def add(k):
            op = self.__dict__[k]
            if op is not None and\
                    not isinstance(op, utils.IdentityLinearOperator):
                return '  ' + k + ': ' + op.__repr__() + '\n'
            return ''
        for k in ['A', 'b', 'M', 'Minv', 'Ml', 'Mr', 'ip_B',
                  'normal', 'self_adjoint', 'positive_definite',
                  'exact_solution']:
            ret += add(k)
        return ret+'}'


class TimedLinearSystem(LinearSystem):
    def __init__(self, A, b,
                 M=None,
                 Minv=None,
                 Ml=None,
                 Mr=None,
                 ip_B=None,
                 normal=None,
                 self_adjoint=False,
                 positive_definite=False,
                 exact_solution=None
                 ):
        self.timings = utils.Timings()

        # get shape
        N = len(b)
        shape = (N, N)

        # time inner product
        try:
            _ip_B = utils.get_linearoperator(shape, ip_B,
                                             timer=self.timings['ip_B'])
        except TypeError:
            def _ip_B(X, Y):
                (_, m) = X.shape
                (_, n) = Y.shape
                if m == 0 or n == 0:
                    return ip_B(X, Y)
                with self.timings['ip_B']:
                    ret = ip_B(X, Y)
                self.timings['ip_B'][-1] /= m*n
                return ret

        super(TimedLinearSystem, self).__init__(
            A=utils.get_linearoperator(shape, A,
                                       self.timings['A']),
            b=b,
            M=utils.get_linearoperator(shape, M,
                                       self.timings['M']),
            Minv=utils.get_linearoperator(shape, Minv,
                                          self.timings['Minv']),
            Ml=utils.get_linearoperator(shape, Ml,
                                        self.timings['Ml']),
            Mr=utils.get_linearoperator(shape, Mr,
                                        self.timings['Mr']),
            ip_B=_ip_B,
            normal=normal,
            self_adjoint=self_adjoint,
            positive_definite=positive_definite,
            exact_solution=exact_solution
            )


class ConvertedTimedLinearSystem(TimedLinearSystem):
    def __init__(self, linear_system):
        # pass through all properties of linear_system as arguments
        kwargs = {k: linear_system.__dict__[k]
                  for k in ['A', 'b', 'M', 'Minv', 'Ml', 'Mr', 'ip_B',
                            'normal', 'self_adjoint', 'positive_definite',
                            'exact_solution'
                            ]
                  }
        super(ConvertedTimedLinearSystem, self).__init__(**kwargs)


class _KrylovSolver(object):
    '''Prototype of a Krylov subspace method for linear systems.'''
    def __init__(self, linear_system,
                 x0=None,
                 tol=1e-5,
                 maxiter=None,
                 explicit_residual=False,
                 store_arnoldi=False,
                 dtype=None
                 ):
        r'''Init standard attributes and perform checks.

        All Krylov subspace solvers in this module are applied to a
        :py:class:`LinearSystem`.  The specific methods may impose further
        restrictions on the operators

        :param linear_system: a :py:class:`LinearSystem`.
        :param x0: (optional) the initial guess to use. Defaults to zero
          vector. Unless you have a good reason to use a nonzero initial guess
          you should use the zero vector, cf. chapter 5.8.3 in *Liesen,
          Strakos. Krylov subspace methods. 2013*. See also
          :py:meth:`~krypy.utils.hegedus`.
        :param tol: (optional) the tolerance for the stopping criterion with
          respect to the relative residual norm:

          .. math::

             \frac{ \| M M_l (b-A (x_0+M_r y_k))\|_{M^{-1}} }
             { \|M M_l b\|_{M^{-1}}}
             \leq \text{tol}

        :param maxiter: (optional) maximum number of iterations. Defaults to N.
        :param explicit_residual: (optional)
          if set to ``False`` (default), the updated residual norm from the
          used method is used in each iteration. If set to ``True``, the
          residual is computed explicitly in each iteration and thus requires
          an additional application of ``M``, ``Ml``, ``A`` and ``Mr`` in each
          iteration.
        :param store_arnoldi: (optional)
          if set to ``True`` then the computed Arnoldi basis and the Hessenberg
          matrix are set as attributes ``V`` and ``H`` on the returned object.
          If ``M`` is not ``None``, then also ``P`` is set where ``V=M*P``.
          Defaults to ``False``. If the method is based on the Lanczos method
          (e.g., :py:class:`Cg` or :py:class:`Minres`), then ``H`` is
          real, symmetric and tridiagonal.
        :param dtype: (optional)
          an optional dtype that is used to determine the dtype for the
          Arnoldi/Lanczos basis and matrix.

        Upon convergence, the instance contains the following attributes:

          * ``xk``: the approximate solution :math:`x_k`.
          * ``resnorms``: relative residual norms of all iterations, see
            parameter ``tol``.
          * ``errnorms``: the error norms of all iterations if
            ``exact_solution`` was provided.
          * ``V``, ``H`` and ``P`` if ``store_arnoldi==True``, see
            ``store_arnoldi``

        If the solver does not converge, a
        :py:class:`~krypy.utils.ConvergenceError` is thrown which can be used
        to examine the misconvergence.
        '''
        # sanitize arguments
        if not isinstance(linear_system, LinearSystem):
            raise utils.ArgumentError('linear_system is not an instance of '
                                      'LinearSystem')
        self.linear_system = linear_system
        N = linear_system.N
        self.maxiter = N if maxiter is None else maxiter
        self.flat_vecs, (self.x0,) = utils.shape_vecs(x0)
        self.explicit_residual = explicit_residual
        self.store_arnoldi = store_arnoldi

        # get initial guess
        self.x0 = self._get_initial_guess(self.x0)

        # get initial residual
        self.MMlr0, self.Mlr0, self.MMlr0_norm = \
            self._get_initial_residual(self.x0)

        # sanitize initial guess
        if self.x0 is None:
            self.x0 = numpy.zeros((N, 1))
        self.tol = tol

        self.xk = None
        '''Approximate solution.'''

        # find common dtype
        self.dtype = numpy.find_common_type([linear_system.dtype,
                                             self.x0.dtype,
                                             dtype
                                             ], [])

        # store operator (can be modified in derived classes)
        self.MlAMr = linear_system.MlAMr

        # TODO: reortho
        self.iter = 0
        '''Iteration number.'''

        self.resnorms = []
        '''Relative residual norms as described for parameter ``tol``.'''

        # if rhs is exactly(!) zero, return zero solution.
        if self.linear_system.MMlb_norm == 0:
            self.xk = self.x0 = numpy.zeros((N, 1))
            self.resnorms.append(0.)
        else:
            # initial relative residual norm
            self.resnorms.append(self.MMlr0_norm/self.linear_system.MMlb_norm)

        # compute error?
        if self.linear_system.exact_solution is not None:
            self.errnorms = []
            '''Error norms.'''

            self.errnorms.append(utils.norm(
                self.linear_system.exact_solution - self._get_xk(None),
                ip_B=self.linear_system.ip_B))

        self._solve()
        self._finalize()

    def _get_initial_guess(self, x0):
        '''Get initial guess.

        Can be overridden by derived classes in order to preprocess the
        initial guess.
        '''
        return x0

    def _get_initial_residual(self, x0):
        '''Compute the residual and its norm.

        See :py:meth:`krypy.linsys.LinearSystem.get_residual` for return
        values.
        '''
        return self.linear_system.get_residual(x0, compute_norm=True)

    def _get_xk(self, yk):
        '''Compute approximate solution from initial guess and approximate
        solution of the preconditioned linear system.'''
        if yk is not None:
            return self.x0 + self.linear_system.Mr * yk
        return self.x0

    def _finalize_iteration(self, yk, resnorm):
        '''Compute solution, error norm and residual norm if required.

        :return: the residual norm or ``None``.
        '''
        self.xk = None
        # compute error norm if asked for
        if self.linear_system.exact_solution is not None:
            self.xk = self._get_xk(yk)
            self.errnorms.append(utils.norm(
                self.linear_system.exact_solution - self.xk,
                ip_B=self.linear_system.ip_B))

        rkn = None

        # compute explicit residual if asked for or if the updated residual
        # is below the tolerance or if this is the last iteration
        if self.explicit_residual or \
            resnorm/self.linear_system.MMlb_norm <= self.tol \
                or self.iter+1 == self.maxiter:
            # compute xk if not yet done
            if self.xk is None:
                self.xk = self._get_xk(yk)

            # compute residual norm
            _, _, rkn = self.linear_system.get_residual(self.xk,
                                                        compute_norm=True)

            # store relative residual norm
            self.resnorms.append(rkn/self.linear_system.MMlb_norm)

            # no convergence?
            if self.resnorms[-1] > self.tol:
                # no convergence in last iteration -> raise exception
                # (approximate solution can be obtained from exception)
                if self.iter+1 == self.maxiter:
                    self._finalize()
                    raise utils.ConvergenceError(
                        ('No convergence in last iteration '
                         '(maxiter: {0}, residual: {1}).')
                        .format(self.maxiter, self.resnorms[-1]), self)
                # updated residual was below but explicit is not: warn
                elif not self.explicit_residual \
                        and resnorm/self.linear_system.MMlb_norm <= self.tol:
                    warnings.warn(
                        'updated residual is below tolerance, explicit '
                        'residual is NOT! (upd={0} <= tol={1} < exp={2})'
                        .format(resnorm, self.tol, self.resnorms[-1]))
        else:
            # only store updated residual
            self.resnorms.append(resnorm/self.linear_system.MMlb_norm)

        return rkn

    def _finalize(self):
        pass

    @staticmethod
    def operations(nsteps):
        '''Returns the number of operations needed for nsteps of the solver.

        :param nsteps: number of steps.

        :returns: a dictionary with the same keys as the timings parameter.
          Each value is the number of operations of the corresponding type for
          ``nsteps`` iterations of the method.
        '''
        raise NotImplementedError('operations() has to be overridden by '
                                  'the derived solver class.')

    def _solve(self):
        '''Abstract method that solves the linear system.
        '''
        raise NotImplementedError('_solve has to be overridden by '
                                  'the derived solver class.')



class Gmres(_KrylovSolver):
    '''Preconditioned GMRES method.

    The *preconditioned generalized minimal residual method* can be used to
    solve a system of linear algebraic equations. Let the following linear
    algebraic system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y`.
    The preconditioned GMRES method then computes (in exact arithmetics!)
    iterates :math:`x_k \in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \|M M_l(b - A x_k)\|_{M^{-1}} =
      \min_{z \in x_0 + M_r K_k} \|M M_l (b - A z)\|_{M^{-1}}.

    The Arnoldi alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
    The initial vector for Arnoldi is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is about maxiter+1 vectors for the Arnoldi basis.
    If :math:`M` is used the memory consumption is 2*(maxiter+1).

    If the operator :math:`M_l A M_r` is self-adjoint then consider using
    the MINRES method :py:class:`Minres`.
    '''
    def __init__(self, linear_system, ortho='mgs', **kwargs):
        '''
        All parameters of :py:class:`_KrylovSolver` are valid in this solver.
        '''
        self.ortho = ortho
        super(Gmres, self).__init__(linear_system, **kwargs)

    def _get_xk(self, y):
        if y is None:
            return self.x0
        k = self.arnoldi.iter
        if k > 0:
            yy = scipy.linalg.solve_triangular(self.R[:k, :k], y)
            yk = self.V[:, :k].dot(yy)
            return self.x0 + self.linear_system.Mr * yk
        return self.x0

    def _solve(self):
        # initialize Arnoldi
        self.arnoldi = utils.Arnoldi(self.MlAMr,
                                     self.Mlr0,
                                     maxiter=self.maxiter,
                                     ortho=self.ortho,
                                     M=self.linear_system.M,
                                     Mv=self.MMlr0,
                                     Mv_norm=self.MMlr0_norm,
                                     ip_B=self.linear_system.ip_B
                                     )
        # Givens rotations:
        G = []
        # QR decomposition of Hessenberg matrix via Givens and R
        self.R = numpy.zeros([self.maxiter+1, self.maxiter],
                             dtype=self.dtype)
        y = numpy.zeros((self.maxiter+1, 1), dtype=self.dtype)
        # Right hand side of projected system:
        y[0] = self.MMlr0_norm

        # iterate Arnoldi
        while self.resnorms[-1] > self.tol \
                and self.arnoldi.iter < self.arnoldi.maxiter \
                and not self.arnoldi.invariant:
            k = self.iter = self.arnoldi.iter
            self.arnoldi.advance()

            # Copy new column from Arnoldi
            self.V = self.arnoldi.V
            self.R[:k+2, k] = self.arnoldi.H[:k+2, k]

            # Apply previous Givens rotations.
            for i in range(k):
                self.R[i:i+2, k] = G[i].apply(self.R[i:i+2, k])

            # Compute and apply new Givens rotation.
            G.append(utils.Givens(self.R[k:k+2, [k]]))
            self.R[k:k+2, k] = G[k].apply(self.R[k:k+2, k])
            y[k:k+2] = G[k].apply(y[k:k+2])

            self._finalize_iteration(y[:k+1], abs(y[k+1, 0]))

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._get_xk(y[:self.arnoldi.iter])

    def _finalize(self):
        super(Gmres, self)._finalize()
        # store arnoldi?
        if self.store_arnoldi:
            if not isinstance(self.linear_system.M,
                              utils.IdentityLinearOperator):
                self.V, self.H, self.P = self.arnoldi.get()
            else:
                self.V, self.H = self.arnoldi.get()

    @staticmethod
    def operations(nsteps):
        '''Returns the number of operations needed for nsteps of GMRES'''
        return {'A': 1 + nsteps,
                'M': 2 + nsteps,
                'Ml': 2 + nsteps,
                'Mr': 1 + nsteps,
                'ip_B': 2 + nsteps + nsteps*(nsteps+1)/2,
                'axpy': 4 + 2*nsteps + nsteps*(nsteps+1)/2
                }
