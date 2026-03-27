import numpy as np
import cvxpy as cp
import random as random
from scipy import sparse

def vec_kron_sum(Xs: list) -> np.array:
    """Compute the Kronecker vector-sum"""
    if len(Xs) == 1:
        return Xs[0]
    elif len(Xs) == 2:
        return np.kron(Xs[0], np.ones(Xs[1].shape[0])) + np.kron(np.ones(Xs[0].shape[0]), Xs[1])
    else:
        d_slash0 = np.prod([X.shape[0] for X in Xs[1:]])
        return (
            np.kron(Xs[0], np.ones(d_slash0))
            + np.kron(np.ones(Xs[0].shape[0]), vec_kron_sum(Xs[1:]))
        )

def fast_kronecker_sum_model(
    Psi_1: np.ndarray,
    Psi_2: np.ndarray,
) -> np.ndarray:
    """
    Generates random sample from given parameters
    of the strong product model.
    """
    Lam1, V1 = np.linalg.eigh(Psi_1)
    Lam2, V2 = np.linalg.eigh(Psi_2)
    core = 1 / vec_kron_sum([Lam1, Lam2])

    # Generate sample
    z = np.random.normal(size=Psi_1.shape[0] * Psi_2.shape[0])
    z *= np.sqrt(core)
    Z = V1 @ z.reshape(Psi_1.shape[0], Psi_2.shape[0]) @ V2.T

    return Z

def ks_vec(lams: list[np.ndarray]):
    ds = [lam.shape[0] for lam in lams]
    out = lams[0].reshape([-1] + [1] * (len(ds) - 1))
    for i, lam in enumerate(lams[1:]):
        shape = [1 if i+1 != j else -1 for j, _ in enumerate(ds)]
        out = out + lam.reshape(shape)
    return out.reshape(-1)

def blockwise_trace(Lam: np.ndarray, ds, axis):
    return np.apply_over_axes(
        np.sum,
        Lam.reshape(ds),
        [a for a in range(len(ds)) if a != axis]
    ).reshape(-1)

def proximal(lams, min_eigenvalue=1e-6, max_eigenvalue=1e6):
    return [np.minimum(np.maximum(lam, min_eigenvalue), max_eigenvalue) for lam in lams]

def gmgm_gradient(es: list[np.ndarray], lams: list[np.ndarray], regularization=0.0) -> list[np.ndarray]:
    ds = [lam.shape[0] for lam in lams]
    results = [None] * len(ds)
    for axis in range(len(ds)):
        left = -blockwise_trace(1/ks_vec(lams), ds, axis)
        right = es[axis]
        reg = regularization * 2 * lams[axis]
        results[axis] = left + right + reg
    return results

def gmgm_objective(es: list[np.ndarray], lams: list[np.ndarray], regularization=0.0) -> float:
    left = sum(e @ lam for e, lam in zip(es, lams))
    right = -np.log(ks_vec(lams)).sum()
    return left + right + regularization * sum((lam**2).sum() for lam in lams)

def gmgm_kernel(
    es,
    lams_init,
    max_iter=1000,
    tol=1e-5,
    min_eigenvalue=1e-6,
    step_size=1,
    line_search_shrinkage=0.5,
    line_search_max_iter=40,
    regularization=0.0,
    verbose=False,
    diagnostics=False
):
    ds = [lam.shape[0] for lam in lams_init]
    K = len(ds)
    lams = lams_init

    if diagnostics:
        objectives = []
        changes = []

    prev_objective = gmgm_objective(es, lams, regularization=regularization)

    for iteration in range(max_iter):
        # Get gradients
        grads = gmgm_gradient(es, lams, regularization=regularization)

        # Map to identifiable gradient
        grads_trace = sum([grad.sum() for grad in grads]) / K
        grads = [grad - grad.sum() / len(grad) + grads_trace / len(grad) for grad in grads]

        # Backtracking line search
        for i in range(line_search_max_iter):
            new_lams = [lam - step_size * grad for lam, grad in zip(lams, grads)]
            new_lams = proximal(new_lams, min_eigenvalue)
            new_objective = gmgm_objective(es, new_lams, regularization=regularization)
            if (new_objective > prev_objective):
                step_size *= line_search_shrinkage
            else:
                break
        else:
            new_objective = prev_objective
            step_size = 0
            if verbose:
                print("Line search failed to find a suitable step size.")
        
        # Take a step along the line
        new_lams = [lam - step_size * grad for lam, grad in zip(lams, grads)]
        new_lams = proximal(new_lams, min_eigenvalue)

        # Check convergence
        max_change = max(np.linalg.norm(new - old) for new, old in zip(new_lams, lams))

        if diagnostics:
            objectives.append(new_objective)
            changes.append(max_change)

        prev_objective = new_objective
        lams = new_lams

        if max_change < tol:
            if verbose:
                print(
                    f'Converged in {iteration+1} iterations '
                    + f'with objective value {new_objective}.'
                )
            break
    else:
        if verbose:
            print(
                f'Max iterations reached ({max_iter}) '
                + f'with objective value {new_objective}.'
            )

    if diagnostics:
        return lams, (np.array(objectives), np.array(changes))
    else:
        return lams

def get_Omega_tilde(Omega1, Omega2, Y, s2):
    YS = Y * s2
    
    diag_term = np.diag(np.diag(YS @ Omega2 @ YS.T))
    full_term = Omega1 * (YS @ YS.T)
    result = full_term + diag_term

    # A no-op to help ensure posdeffness against numerical issues
    # result = (result + result.T) / 2
    # eps = 1e-6 * np.trace(result) / result.shape[0]
    # result += eps * np.eye(result.shape[0])

    return result

def flip(Omega1, Omega2, Y, s2, tol=1e-6):
    d1, d2 = Omega1.shape[0], Omega2.shape[0]
    Omega_tilde = get_Omega_tilde(Omega1, Omega2, Y, s2.reshape(1, d2))

    s1 = cp.Variable(d1)
    prob = cp.Problem(
        cp.Minimize(
            cp.quad_form(s1, Omega_tilde, assume_PSD=True)
        ),
        [
            s1 >= tol,
            s1.sum() == 1
        ]
    )
    prob.solve()
    return s1.value

# I've heard that scipy.minimize or conjugate gradient may vastly outperform CVXPY here;
# if I end up runtime-limited, it may be worth switching over to scipy.minimize...
def flop(Omega1, Omega2, Y, s1, tol=1e-6):
    d1, d2 = Omega1.shape[0], Omega2.shape[0]
    Omega_tilde = get_Omega_tilde(Omega2, Omega1, Y.T, s1.reshape(1, d1))

    s2 = cp.Variable(d2)
    prob = cp.Problem(
        cp.Minimize(
            cp.quad_form(s2, Omega_tilde, assume_PSD=True)
        ),
        [
            s2 >= tol,
            s2.sum() == 1
        ]
    )
    prob.solve()
    return s2.value

def flipflop(Omega1, Omega2, Y, s1, s2, max_iter=20, convergence_tol=1e-10, zero_tol=1e-4, verbose=False):
    previous_s1 = s1
    previous_s2 = s2

    for _ in range(max_iter):
        new_s2 = flop(Omega1, Omega2, Y, previous_s1, tol=zero_tol)
        new_s1 = flip(Omega1, Omega2, Y, new_s2, tol=zero_tol)

        diff1 = np.abs(new_s1 - previous_s1).sum()
        diff2 = np.abs(new_s2 - previous_s2).sum()
        if verbose:
            print(f"Diffs: {diff1}, {diff2}")

        if (
            diff1 < convergence_tol
            and diff2 < convergence_tol
        ):
            break

        previous_s1 = new_s1
        previous_s2 = new_s2
    else:
        if verbose:
            print("Warning: failed to converge")

    return new_s1, new_s2

def get_zstar(Omega1, Omega2, Y, s1, s2, max_iter=200, convergence_tol=1e-10, zero_tol=1e-4, verbose=False):
    s1, s2 = flipflop(Omega1, Omega2, Y, s1, s2, max_iter=max_iter, convergence_tol=convergence_tol, zero_tol=zero_tol, verbose=verbose)
    return s1.reshape(-1, 1) * Y * s2.reshape(1, -1), s1, s2

def get_poptinv_final(A, B, K):
    K_red = K[:, :-1]
    B_red = B[:-1, :-1]

    # Schur complement
    A_inv = np.linalg.inv(A)
    KA = K_red.T @ A_inv
    S = B_red - KA @ K_red
    S_inv = np.linalg.inv(S)

    # Blocks of the inverse
    AKS = KA.T @ S_inv
    top_left = A_inv + AKS @ KA
    top_right = - AKS
    bottom_right = S_inv

    return [top_left, top_right, bottom_right]

def trace_final(L, K, M):
    d2 = M.shape[0]+1
    d1 = K.shape[0]
    M_full = np.zeros((d2,d2))
    M_full[:-1,:-1] = M
    K_full = np.zeros((d1,d2))
    K_full[:,:-1] = K

    term1 = 2 * np.trace(L)
    term2 = d1 * (M_full + M_full.T)
    s = K_full.sum(axis=0)
    term3 = 2 * (s[None,:] + s[:,None])
    
    return term1 + term2 + term3

def axis_curvature_final(Omega1, Omega2, d1, d2):
    A = d2 * Omega1 + Omega2.sum() * np.eye(d1)
    B = d1 * Omega2 + Omega1.sum() * np.eye(d2)
    C = Omega1.sum(axis=1).reshape(-1, 1) + Omega2.sum(axis=0).reshape(1, -1)

    [L, K, M] = get_poptinv_final(A, B, C)

    return trace_final(L, K, M)

def curvatures_final(Omega1, Omega2, d1, d2):
    C2 = axis_curvature_final(Omega1, Omega2, d1, d2)
    C1 = axis_curvature_final(Omega2, Omega1, d2, d1)
    return C1, C2

def laplace_approximation(Omega1, Omega2, Y, s1, s2):
    d1 = Omega1.shape[0]
    d2 = Omega2.shape[0]

    zstar, s1, s2 = get_zstar(Omega1, Omega2, Y, s1, s2, max_iter=200, convergence_tol=1e-10, zero_tol=1e-5)

    C1, C2 = curvatures_final(Omega1, Omega2, d1, d2)
    
    outer1 = zstar @ zstar.T - C1 / 2
    outer2 = zstar.T @ zstar - C2 / 2

    return outer1, outer2, s1, s2

def EM_step(Omega1, Omega2, Y, verbose=False, regularization=0, *, s1=None, s2=None):
    outer1, outer2, s1, s2 = laplace_approximation(Omega1, Omega2, Y, s1, s2)

    e1, V1 = np.linalg.eigh(outer1)
    e2, V2 = np.linalg.eigh(outer2)
    es = [e1, e2]

    lams = [
        np.ones_like(e1),
        np.ones_like(e2)
    ]

    new_lams = gmgm_kernel(es, lams, min_eigenvalue=1e-6, verbose=verbose, max_iter=1000, regularization=regularization)

    Omega1 = V1 @ np.diag(new_lams[0]) @ V1.T
    Omega2 = V2 @ np.diag(new_lams[1]) @ V2.T

    return Omega1, Omega2, s1, s2

def EM_algorithm(
    Y,
    max_iter=100,
    stop_tol=1e-3,
    verbose=False,
    very_verbose=False,
    regularization=0,
    return_s_values=False
):
    d1, d2 = Y.shape
    Omega1_init = np.eye(d1)
    Omega2_init = np.eye(d2)
    s1_init = np.random.random(d1)
    s2_init = np.random.random(d2)
    magnitudes = []
    
    Omega1 = Omega1_init
    Omega2 = Omega2_init
    s1 = s1_init
    s2 = s2_init
    for iteration in range(max_iter):
        if very_verbose:
            mag = f"with magnitude {magnitudes[-1]:1.1e}" if iteration > 0 else ""
            print(f"Beginning iteration {iteration} {mag}")

        new_Omega1, new_Omega2, s1, s2 = EM_step(Omega1, Omega2, Y, regularization=regularization, s1=s1, s2=s2)
        magnitudes.append(np.sqrt(
            ((new_Omega1 - Omega1)**2).sum() / d1**2
            + ((new_Omega2 - Omega2)**2).sum() / d2**2
        ))
        Omega1, Omega2 = new_Omega1, new_Omega2

        if magnitudes[-1] < stop_tol:
            if verbose:
                print(f"Converged on iteration {iteration+1}")
            break
    else:
        if verbose:
            print(f"Did not converge after {max_iter} iterations")

    if return_s_values:
        return (Omega1, Omega2), magnitudes, (s1, s2)
    else:
        return (Omega1, Omega2), magnitudes

def quotient(X):
    if np.any(X == 0):
        raise ValueError("Input contains zeros; use sparse_quotient instead")
    d1, d2 = X.shape
    signs = np.sign(X)
    absvals = np.abs(X)
    logvals = np.log(absvals)

    demeaned = (np.eye(d1) - np.ones((d1, d1)) / d1) @ logvals @ (np.eye(d2) - np.ones((d2, d2)) / d2)

    expvals = np.exp(demeaned)
    return signs * expvals
    
def sparse_quotient(X, densify_result=True):
    d1, d2 = X.shape
    sparse_X = sparse.csr_matrix(X)

    rows, cols = sparse_X.nonzero()
    data = sparse_X.data

    signs = np.sign(data)
    L = np.log(np.abs(data))

    # Calculate row means
    row_sums = np.bincount(rows, weights=L, minlength=d1)
    row_counts = np.bincount(rows, minlength=d1)
    row_means = np.zeros(d1)
    row_means[row_counts > 0] = row_sums[row_counts > 0] / row_counts[row_counts > 0]

    # Calculate column means
    col_sums = np.bincount(cols, weights=L, minlength=d2)
    col_counts = np.bincount(cols, minlength=d2)
    col_means = np.zeros(d2)
    col_means[col_counts > 0] = col_sums[col_counts > 0] / col_counts[col_counts > 0]

    # Global mean
    global_mean = L.mean()

    # Double center only on nonzeros
    w = L - row_means[rows] - col_means[cols] + global_mean

    sparse_X.data = signs * np.exp(w)

    if densify_result:
        return sparse_X.toarray()
    else:
        return sparse_X