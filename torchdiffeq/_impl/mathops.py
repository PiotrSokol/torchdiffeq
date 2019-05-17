import torch as torch
from torch import jit
from .gradops import explicit_jacobian
from .gradops import Rop
import warnings


@jit.script
def fgmres(jv, b, x0, tol=torch.tensor(1e-6), restart_limit=20, maxiter=50, orthotol=torch.tensor(1e-5)):
    """
        Implements an inner-outer preconditioned Flexible GMRES
    to solve a linear problem  Jx = b
    where A is implicitly given via matrix vector products through the function jv
    :param nonlinear function
    :param b: torch.tensor n by 1 tensor
    :param x0: torch.tensor, initial guess
    :param tol: relative tolerance for terminating the loop
    :param restart_limit: number of iterations of Flexible GMRES before restarts
    :param maxiter: maximal number of restarts for the algorithm
    :param orthotol: Brown Hindmarsh condition on loss of orthogonality
    :return: x: the solution to the linear problem
    """
    dtype = b.dtype
    device = b.device
    n = x0.shape[0]
    x = x0
    for iii in range(maxiter):
        H = torch.zeros((maxiter+1, maxiter), dtype=dtype).device
        deltaH = torch.zeros((1, 1), dtype=dtype).device
        V = torch.zeros((n, maxiter+1), dtype=dtype).device
        Z = torch.zeros((n, maxiter), dtype=dtype).device
        bnorm = b.norm()
        r0 = b - jv(x0)
        beta = r0.norm()

        V[:, [0]] = r0/beta
        # Arnoldi process starts here
        for ii in range(restart_limit):
            for j in range(n):
                Z[:, [j]] = gmres(jv, V[:, [j]], x, device=device, dtype=dtype, tol=tol)
                w = jv(Z[:, [j]])
                wnormold = w.norm()
                for reortho in range(2):
                    for i in range(j+1):
                        deltaH = (w.t() @ V[:, [i]])[0, 0]
                        w -= deltaH * V[:, [i]]
                        H[i, [j]] += deltaH
                    wnorm = w.norm()
                    if wnorm > orthotol*wnormold:  # Double Modified Gram Schmidt condition
                        # Detects if the norm of w after the GS loop is sufficiently close to the original
                        continue
                H[j+1, [j]] = wnorm
                # Arnoldi process ends
                V[:, [j + 1]] = w / H[[j+1], [j]]
                e1 = torch.zeros(torch.Size([j+1, 1]))
                e1[0, 0] = beta
                # y,_ = torch.solve(e1, H[:j+1, :j+1].unsqueeze(0))  # H^-1 beta e_1
                sols = torch.triangular_solve(e1, H[:j+1, :j+1], upper=True, transpose=False, unitriangular=False)
                # x += V[:, :j].t() @ y.squeeze(dim=0)
                x += Z[:, :j+1] @ sols.solution
                resnorm = torch.norm(b - jv(x))
                if resnorm < tol*bnorm:
                    return x

    # warnings.warn('FGMRES did not converge with a tolerance of {}'.format(tol))
    return x


@jit.script
def gmres(jv, b, x0, device, tol=torch.tensor(1e-6), restart_limit= 5, maxiter=5, orthotol=torch.tensor(1e-3), dtype=torch.float32):
    """
    Implements a simple GMRES method with restart and no preconditioner

    see description in fgmres above
    :param jv:
    :param b:
    :param x0:
    :param tol:
    :param maxiter:
    :param orthotol:
    :return:
    """
    n = x0.shape[0]
    x = x0

    for iii in range(maxiter):
        H = torch.zeros((maxiter + 1, maxiter), dtype=dtype)
        deltaH = torch.zeros((1, 1), dtype=dtype)
        V = torch.zeros((n, maxiter + 1), dtype=dtype)
        bnorm = b.norm()
        r0 = b - jv(x0)
        beta = r0.norm()

        V[:, [0]] = r0 / beta
        # Arnoldi process starts here

        for ii in range(restart_limit):
            for j in range(n):
                w = jv(V[:, [j]])
                wnormold = w.norm()
                for reortho in range(2):
                    for i in range(j+1):
                        deltaH = (w.t() @ V[:, [i]])[0, 0]
                        w -= deltaH * V[:, [i]]
                        H[i, [j]] += deltaH
                    wnorm = w.norm()
                    if wnorm > orthotol * wnormold:  # Double Modified Gram Schmidt condition
                        # Detects if the norm of w after the GS loop is sufficiently close to the original
                        continue
                H[j+1, [j]] = wnorm
                # Arnoldi process ends
                V[:, [j + 1]] = w / H[[j+1], [j]]
                e1 = torch.zeros(torch.Size([j+1, 1]), dtype=dtype)
                e1[0, 0] = beta
                # y,_ = torch.solve(e1, H[:j+1, :j+1].unsqueeze(0))  # H^-1 beta e_1
                sols = torch.triangular_solve(e1, H[:j+1, :j+1], upper=True, transpose=False, unitriangular=False)
                x += V[:, :j+1] @ sols.solution
                resnorm = torch.norm(b - jv(x))
                if resnorm < tol * bnorm or H[j + 1, [j]] / torch.norm(H[:j + 2, :j + 1], 2) <= 1e-14:
                    return x

    return x


def test_gmres():
    A = torch.tensor([[2., 3.],[4., 5.]]).type(torch.float64)
    b = torch.tensor([[5.],[6.]]).type(torch.float64)
    jv = lambda x: A@x
    MyX = gmres(jv, b, torch.zeros_like(b), tol=torch.tensor(1e-6), maxiter=5)

@jit.script
def newton_krylov(f, x, atol, rtol, maxit=40, maxit_fgmres=40,
                  restart_limit=20, etamax=torch.tensor(0.9)):
        """
        Inexact Newton-Armijo iteration with Eisenstat-Walker forcing
        Implements a line search with a parabolic trust region
        Reimplemented from C.T. Kelly's Matlab implementation
        https://archive.siam.org/books/fa01/nsoli.m
        :param f: function
        :param x: initial guess
        :param atol: absolute tolerance
        :param rtol: relative tolerance
        :param maxit: maximum number of nonlinear iterations
        :param maxit_fgmres: maximum number of inner iterations for fgmres
        :param lmaxit:
        :param restart_limit:
        :param etamax:
        :return: solution x, code ierr
        # ierr = False for successful termination
        # ierr = True after maxit or linesearch failure
        """
        dtype = x.type
        device = x.device
        alpha = torch.tensor(1e-4, dtype=dtype).device  # sufficient decrease for line search
        sigma0 = torch.tensor(1e-1, dtype=dtype).device  # line search trust region param
        sigma1 = torch.tensor(0.5, dtype=dtype).device  # line search trust region param
        gamma = torch.tensor(0.9, dtype=dtype).device
        maxarm = 20  # number of step length reductions before failure reported
        itc = 0  # iteration counter

        gmres_eta = etamax

        f0 = f(x)
        fnrm = f0.norm()

        stop_tol = atol + rtol*fnrm
        while fnrm > stop_tol and itc < maxit:
            fnrmo = fnrm
            itc += 1

            # step, errstep, inner_it_count, inner_f_evals  = \
            # dkrylov(f0, f, x, gmres_eta, lmaxit, restart_limit)
            step = fgmres(f, f0, x, tol=gmres_eta, restart_limit=restart_limit, maxiter=maxit_fgmres)
            # line search starts here
            xold = x
            lbd = 1
            lamm = 1
            lamc = lbd
            iarm = 0

            xt = x + lbd*step
            ft = f(xt)
            nft = ft.norm()
            nf0 = f0.norm()
            ff0 = nf0**2
            ffc = nft ** 2
            ffm = nft ** 2  #TODO why are there two?
            while nft >= (1 - alpha * lbd) * nf0:
                if iarm == 0:
                    lbd = sigma1 * lbd
                else:
                    lbd = parab3p(lamc, lamm, ff0, ffc, ffm, sigma0, sigma1)
                xt = x+lbd*step
                lamm = lamc
                lamc = lbd

                ft = f(f, xt)
                nft = ft.norm()
                ffm = ffc
                ffc = nft**2
                iarm += 1

                if iarm > maxarm:
                    return xold, True
                # End of Armijo line search
            x = xt
            f0 = ft
            fnrm = f0.norm()

            rat = fnrm/fnrmo

            #Adjust eta as per Eisenstat-Walker
            etaold = etamax.abs()
            etanew = gamma*rat**2
            if gamma*etaold**2 > torch.tensor(0.1, dtype=dtype):
                etanew = torch.max(etanew, gamma*etaold**2)
            gmres_eta = torch.max( torch.min(etanew, etamax), 0.5*stop_tol/fnrm)
        if fnrm > stop_tol:
            return x, True
        else:
            return x, False


def dkrylov(f0, f, x, etamax, lmaxit, restart_limit):
    raise NotImplementedError

@jit.script
def parab3p(lambdac=torch.DoubleTensor(1.), lambdam=torch.DoubleTensor(1.), ff0=torch.DoubleTensor(1.),
            ffc=torch.DoubleTensor(1.), ffm=torch.DoubleTensor(1.), sigma0=torch.DoubleTensor(1.),
            sigma1=torch.DoubleTensor(1.)):
    """
    Apply three point safeguarded parabolic model to line search.

    Reimplemented from C.T. Kelly's Matlab implementation
    https://archive.siam.org/books/fa01/nsoli.m
    :param lambdac: current steplength
    :param lambdam: previous step length
    :param ff0: norm squared of F(x_c)
    :param ffc: norm squared of F(x_c + lambdac * d)
    :param ffm: norm squared of F(x_c + lambdam * d)
    :param sigma0: trust region parameter
    :param sigma1: trust region parameter
    :return: new step length given by parabolic model
    """
    c2 = lambdam*(ffc - ff0) - lambdac*(ffm - ff0)
    if c2 >= 0:
        return sigma1 * lambdac
    c1 = lambdac**2 * (ffm - ff0) - lambdam**2*(ffc - ff0)

    lambdap = -c1 * .5/c2

    if lambdap < sigma0 * lambdac:
        return sigma0*lambdac
    else:
        return sigma1*lambdac


@jit.script
def explicit_newton(f, x, reltol=torch.tensor(1e-6), atol=torch.tensor(1e-6), maxit=40):
    """

    :param f: f(x) is a function that evaluates the residual of the current stage
     x - sum_j^{i-1}\delta t k_j -  \delta t *A_{i,i}*\dot{x}(t,x)
     where k is the value of the previous i-1 stages
    :param x: initial guess
    :param reltol:
    :param atol:
    :param maxit:
    :return: solution x
    ierr = False for successful termination
    ierr = True if the Newton algorithm did not converge
    """
    # inspired by https://bitbucket.org/drreynolds/rklab/src/master/newton.m
    s = torch.ones_like(x)
    for i in range(maxit):
        F = f(x)
        if torch.norm(s, p=float('inf')) < reltol or torch.norm(F, p=float('inf')) < atol:
            return x, False
        A = explicit_jacobian(x, F)# might be slow;
        # for Diagonally Implicit Runge Kutta savings might be made if we consider
        # the special structure of the Jacobian of the residual
        #  I - h*A_{i,i} * \frac{\partial \dot{x}(x,t){\partial x}
        s = torch.solve(F.unsqueeze(0), A.unsqueeze(0))
        x -= s.squeeze(0)
    return x, True