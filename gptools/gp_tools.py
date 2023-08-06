
import os
from pathlib import Path

import numpy as np
from joblib import Memory

DISABLE_CACHE = False
memory = Memory(location=None if DISABLE_CACHE else Path(__file__).parents[0] / "cache", verbose=0)
filename = os.path.basename(__file__).replace('.py', '')

# create logger
import logging
logger = logging.getLogger(__name__)

def get_GP_samples(X_train=None, N_samples=1, K=None, k_dx=None, seed=1235, reg=0., dX_mode ="norm", pdify=False, check_valid="warn"):
    if K is None:
        if X_train.ndim == 1:
            X_train_vec = X_train[:, None]
        else:
            X_train_vec = X_train

        if dX_mode == "norm":
            dX = np.linalg.norm(X_train_vec[None, :, :] - X_train_vec[:, None, :], axis=-1)
        elif dX_mode == "cosine":
            dX = np.einsum('ki,li->kl', X_train_vec[:, :], X_train_vec[:, :]) / np.linalg.norm(X_train_vec, axis=-1, keepdims=True)**2
            clipped_dX = np.clip(dX, -1, 1)
            if np.allclose(dX, clipped_dX, atol=1e-5):
                dX = clipped_dX
            else:
                raise ValueError

        else:
            raise ValueError
        K = k_dx(dX)
    else:
        pass

    if pdify:
        K = nearestPD(K)
    samples = get_GP_samples_(K, N_samples, K.shape[0], seed, reg, check_valid=check_valid)
    return np.squeeze(samples)


@memory.cache
def get_GP_samples_(K, N_samples, N_supp, seed, reg, check_valid="warn"):
    # sample manually to get continuously transformed samples off the kernel
    # https://juanitorduz.github.io/multivariate_normal/
    # get uncorrelated vectors
    # d = len(x_train)
    # u = rng.normal(loc=0, scale=1, size=d * N_samples).reshape(N_samples, d)
    # L = np.linalg.cholesky(K)
    # m = np.zeros(d)
    # samples = m + np.dot(L, u)

    # passing the argument "cholesky" reproduces this behavior out-of-the-box
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal([0] * N_supp, K+ np.eye(K.shape[0])*reg, size=N_samples, check_valid=check_valid)
    return samples


def filter_samples_by_observations(x_all, y_all, X_obs, Y_obs):
    match_conditions = []
    for x_obs, y_obs in zip(X_obs, Y_obs):
        i_obs = np.argmin((x_all - x_obs) ** 2)
        atol = 10e-2
        match_conditions += [[np.allclose(sample[i_obs], y_obs, atol=atol) for sample in y_all]]

    close_samples = np.array(match_conditions).all(axis=0)

    return y_all[close_samples], y_all[~close_samples]


def plot_posterior(ax, x_all, y_match, y_fail, X_obs, Y_obs, x_tst, cmap, color_seed=123, **plot_kwargs):
    max_fail_samples = 10
    lines_unmatch = []
    lines_match = []
    rng_color = np.random.default_rng(color_seed)
    alpha = plot_kwargs.pop("alpha", 1.0)
    for i, sample in enumerate(y_match):
        l, = ax.plot(x_all.squeeze(), sample, lw=1, color=cmap(rng_color.random()), alpha=alpha)
        lines_match.append(l)

    for i, sample in enumerate(y_fail[:max_fail_samples]):
        l, = ax.plot(x_all.squeeze(), sample, lw=1, color=cmap(rng_color.random()), alpha=alpha)
        lines_unmatch.append(l)

    return lines_match, lines_unmatch


# @memory.cache
def get_Y_pr(K_xX, K_XX, y_tr, nearest_PD=False, inv_mode="cholesky", reg=0., verbose=0):
    K_XX += reg*np.eye(K_XX.shape[0])

    if nearest_PD:
        K_XX = nearestPD(K_XX)
        inv_mode = "cholesky"

    if inv_mode == "cholesky":
        try:
            L = np.linalg.cholesky(K_XX)
        except LinAlgError:
            logger.warning("Matrix not positive definite for cholesky, calling PD-ifier... This can result in poor condition number!")
            K_XX = nearestPD(K_XX)
            L = np.linalg.cholesky(K_XX)

        y_pr = K_xX @ np.linalg.inv(L).T @ np.linalg.inv(L) @ y_tr
    elif inv_mode == "pinv":
        y_pr = K_xX @ np.linalg.pinv(K_XX)@y_tr
    elif inv_mode == "inv":
        y_pr = K_xX @ np.linalg.inv(K_XX)@y_tr
    else:
        raise ValueError
    
    if verbose > 0:
        eigvals = np.linalg.eigvalsh(K_XX)
        print(f"Condition number is {(np.max(eigvals)/np.min(eigvals)):.2f}")
    return y_pr

def vectors_from_theta(theta, d, norm=1., model='rate', full_circle=False):
    """ Returns vectors of angle theta on the norm-sphere. """
    theta = np.abs(theta)
    theta = np.atleast_1d(theta)
    if model == "rate" or model == None:
        x1 = np.zeros(shape=(len(theta), d))
        x1[:, -1] = 1
        x2 = np.empty(shape=(len(theta), d))
        x2[:, :] = (((1 - np.cos(theta) ** 2) / (d - 1)) ** 0.5)[:, None]
        x2[:, -1] = np.cos(theta)
        x1 = norm * x1
        x2 = norm * x2
        assert np.allclose(np.linalg.norm(x1, axis=-1), norm)
        assert np.allclose(np.linalg.norm(x2, axis=-1), norm)

    elif model == "ising" or model == "binary":
        x1 = np.full(shape=(len(theta), d), fill_value=-1.)
        x2 = np.full(shape=(len(theta), d), fill_value=-1.)
        for i, th in enumerate(theta):
            N_flips = np.round((1 - np.cos(th)) * d / 2).astype(int)
            x2[i, :N_flips] = 1.
    else:
        raise ValueError("model not found")

    # normalize
    x1_ = x1
    x2_ = x2
    x1_ /= np.linalg.norm(x1, axis=-1, keepdims=True)
    x2_ /= np.linalg.norm(x2, axis=-1, keepdims=True)
    acos_x1x2 = np.arccos(np.einsum('...i,...i', x1_, x2_))
    if model != "ising" and not full_circle: assert np.allclose(theta, acos_x1x2, atol=1e-3)

    return np.squeeze(x1), np.squeeze(x2)

def nearestPD(A, mode="matlab"):
    if mode == "matlab":
        """Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
    elif mode == "clip":
        # NOTE: the backtransform can numerically yield still negative eigenvalues
        C = (A + A.T)/2
        eigval, eigvec = np.linalg.eig(C)
        eigval[eigval < 0] = 0

        A3 = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
    else:
        raise ValueError

    
    assert np.allclose(A, A3, atol=1e-2)
    return A3

def get_kappa(lmbd, spec, P):
    """
    Canatar21, eq. (4)
    """
    from scipy.optimize import fixed_point, root_scalar

    assert np.ndim(spec) == 1
    assert np.isscalar(P)
    assert lmbd >= 0

    def fix(kappa):
        return lmbd + np.sum(kappa*spec/(kappa+P*spec))
    
    fix = np.vectorize(fix)
    zero = lambda kappa: fix(kappa) - kappa
    
    
    kappa = root_scalar(zero, bracket=[0., 10.], method='brentq').root
    # kappa = fixed_point(fix, 1e-3)
    logger.info(f"Found self-consistent kappa={kappa}, rel. fixpoint discrepancy is {np.abs(fix(kappa)-kappa)/kappa:.4f}")
    # # plot
    # import matplotlib.pyplot as plt
    # plt.close()
    # kappas = np.linspace(-10, 10, 1000)
    # plt.plot(kappas, fix(kappas))
    # plt.plot(kappas, (kappas))
    # plt.plot([kappa], [kappa], marker="o", color="k")
    # plt.show()
    
    assert kappa >= 0

    return kappa

def get_gamma(kappa, spec, P):
    """
    Canatar21, eq. (4)
    """
    assert np.ndim(spec) == 1
    assert np.isscalar(P)

    gamma = np.sum(P*spec**2/(kappa + P*spec)**2)
    assert 0 <= gamma < 1
    return gamma

def get_var(l, kappa, gamma, P, sigma, spec, w_bar=None):
    """
    Canatar21, **ARXIV VERSION**, eq. (71)

    w_bar: ground truth expansion coefficients
    """

    # assume a typical target function sampled from the GP
    if w_bar is None:
        w_bar = np.array([np.random.normal(0, scale=spec[l]**0.5) for l, _ in enumerate(spec)])

    var = 1/(1-gamma)*(sigma**2 + kappa**2*np.sum(spec*w_bar**2/(P*spec + kappa)**2)) 
    var *= P*spec[l]**2 / (P*spec[l] + kappa)**2
    var /= np.sqrt(spec[l]**2)
    return var

get_var = np.vectorize(get_var, excluded=["kappa", "gamma", "P", "sigma", "spec", "w_bar"])