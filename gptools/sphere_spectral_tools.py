
# Gaussfrom pathlib import Path
from joblib import Memory
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy import integrate
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import curve_fit
from scipy.special import gamma, factorial, gegenbauer
from scipy.special import gamma, binom
import logging
import numpy as np

from bayesianchaos.CONST import DISABLE_CACHE, JOBLIB_VERB, PRJ_ROOT, SRC_ROOT
from joblib import Memory

memory = Memory(
    location=None if DISABLE_CACHE else SRC_ROOT / "cache", verbose=0, 
)


logger = logging.getLogger(__name__)

# set verbosity to warning
logger.setLevel(logging.INFO)

def shell_measure(d_emb):
    assert d_emb >= 3
    d = d_emb
    return 2*np.pi**(d/2)/gamma(d/2)

def get_uniform_measure(d):
    # get the xx measure for uniformly distributed vectors on the d-1 sphere
    # https://djalil.chafai.net/blog/2021/05/22/the-funk-hecke-formula/
    Z_inv = np.pi**-0.5 * frac_gamma(x=d/2, da=0, db=-1/2)
    return lambda t: Z_inv*(1-t**2)**((d-3)/2)


def get_Y_from_spectrum(var_lmbds, d, seed):
    from scipy.special import sph_harm
    rng_GP = np.random.default_rng(seed+1)

    Omega = shell_measure(d_emb=d)
    # get the index order right, and use the convention where theta is the polar angle and phi is the azimuthal angle
    Ylm = lambda l, m, theta, phi: sph_harm(m,l, phi, theta) * Omega**0.5

    def Ylm_real(l, m, phi, theta):
        """
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        """
        
        if m == 0:
            return Ylm(l, m, theta, phi).real
        elif m > 0:
            return 2**0.5*(-1)**m*Ylm(l, m, theta, phi).real
        elif m < 0:
            return 2**0.5*(-1)**m*Ylm(l, abs(m), theta, phi).imag
        
    pref = lambda l, m: ((2*l+1) / (4*np.pi) * (factorial(l-m) / factorial(l+m)))**0.5
    
    def Y(PHI, THETA):
        Y_ = Ylm_real(0, 0, PHI, THETA)*0
        
        for l in range(0, len(var_lmbds)):
            sqrtlmbds = rng_GP.normal(0, var_lmbds[l]**0.5, size=(2*l+1,))
            sqrtlmbds_d = {m: sqrtlmbd for sqrtlmbd, m in zip(sqrtlmbds, range(-l, l+1))}

            for im, m in enumerate(range(-l, l+1)):
                # correlated weights enable contructive interference between modes
                sqrtlmbd = sqrtlmbds_d[m]  # whether or not use the same lmbd at a given frequency. This will make the sample smoother. 
                # scipy convention flips theta and phi compared to standard spherical coordinates
                #* frac_fac**-0.5
                Y_ += sqrtlmbd*Ylm_real(l, m, PHI, THETA)

        return Y_
    
    return Y

def frac_gamma(x, da, db):
    """
    Prevents overflow if dimension x=d/2 is very large
    """
    if x < 50:
        return gamma(x + da) / gamma(x + db)
    else:
        # use stirling, see https://math.stackexchange.com/questions/117977/quotient-of-gamma-functions
        return x ** (da - db)

def get_lambda_taylor(bn, d):
    lmd = []
    for l in range(len(bn)):
        # db += 1/2 is correction from Azevedo 2015, also pi ** ((d-1)/2) and 2**(l+1)
        fac = factorial
        f1 = 1 / (np.pi**((d-1)/2)  * 2 ** (l+1))
        f2 = 0

        for s in range(np.floor((len(bn) - l) / 2).astype(int)):

            f2 += bn[2 * s + l] * fac(2 * s + l) / fac(2 * s) * gamma(s + 1 / 2) * frac_gamma(x=d/2, da=0, db=s+l+1/2)
        lmbd_l = f1 * f2
        lmd += [lmbd_l]

    return np.array(lmd)


def get_bn_fn(fn, deg, x0=0):
    bn_poly = approximate_taylor_polynomial(fn, x0, deg, scale=1)
    return bn_poly

code = """
def f(x, {vars}):
    coeff = [{vars}]
    r = 0
    for d, c in enumerate(coeff):
        r += c*(x-{x0})**d

    return r
"""


def get_bn_disc(x,y,deg, x0=0, omit_0=True):
    """
    Fits a function of the form 
    c*(x-{x0})**d
    to the data, where d can be non-integer (i.e. roots).
    """
    # def a function
    vars = ", ".join([f"a{i}" for i in range(deg+1)])
    code_ = code.replace("{vars}", vars).replace("{x0}", str(x0))
    l = dict()
    g = dict()
    exec(code_, g, l)
    f = list(l.values())[0]
    if omit_0:
        x = x[y!=1]
        y = y[y != 1]
    p_opt, p_err = curve_fit(f, x, y, bounds=(0,np.inf))
    assert (p_opt >= 0).all()
    bn_poly = Polynomial(p_opt)
    # coef = np.polynomial.polynomial.polyfit(x, y, deg, full=False)
    # bn_poly = Polynomial(coef)
    return bn_poly


def reconstruct_from_spec(lmbds, d):
    """
    Generates a callable from a given spectrum. 
    """
    w_d = np.pi**-0.5 * frac_gamma(x=d/2, da=0, db=-1/2)
    alpha = (d-2)/2
    ggb = lambda l: gegenbauer(alpha=alpha, n=l)
    C_alpha_l = ggb(1)
    mu = lambda xx: (1-xx**2)**((d-3)/2)

    def k(xx):
        # because the ggb polynomials are orthonormal wrt (1-t^2)^(alpha-1/2), we can omit this factor in the reconstruction
        return np.sum([lmbds[l]*ggb(l)(xx) for l in range(len(lmbds))], axis=0) / len(lmbds)
    
    return k

def get_k(a):
    return lambda xx: np.sign(xx)*np.abs((np.arcsin(xx)/(np.pi/2)))**a

def get_A_spectral_projection(X, lmbd_out=None, alpha_proj=None):    
    # do a PCA via SKLearn
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(X.shape[-1], 512))

    # n_samples, n_features
    pca.fit(X)

    # n_components, n_features
    U = pca.components_
    lmbds_is = pca.explained_variance_

    if alpha_proj is not None:
        lmbds_out = (np.arange(1, len(lmbds_is)+1)**-alpha_proj)**0.5
    
    A = np.einsum("ni,n,nj->ij", U, lmbds_out[:len(lmbds_is)], U)

    return A



def kernel_from_alpha(alpha, d, p=None, X=None, opt_p=None, fit_args=dict()):
    assert p is not None or X is not None, "Either p or X must be given."
    assert not (p is not None and X is not None), "Either p or X must be given, not both."

    p = get_uniform_measure(d) if p is None else p

    logger.info(f"Searching kernel with fit params {fit_args}.")

    def get_alpha(k, p):
        ls = np.arange(0, 10) + 1
        _, lmbds = get_lambda_int(ls, d, shape_fn=k, p=p, repeat_degenerate=True, error_level="warn")
        lmbds = lmbds / lmbds[0]

        alpha_decay_is = fit_spectrum_decay(np.arange(len(lmbds)) + 1, lmbds, N_rescale=1, **fit_args)[0]
        return alpha_decay_is


    # make an optimization loop that takes a kernel and returns the alpha that minimizes the error
    def trafo(x, g):
        return np.sign(x)*np.abs(x)**g

    def get_p_fit(x):
        if not opt_p:
            p_fit = p
        elif opt_p == 'proj':
            A = get_A_spectral_projection(X, lmbd_out=None, alpha_proj=x[1])
            X_out = np.einsum("ij,...j->...i", A, X)
            bins = np.linspace(-1,1,100)
            hist, _ = np.histogram((X_out@X_out.T).flatten(), bins=bins, density=True)
            p_fit = lambda xx: np.interp(xx, bins[:-1], hist)
        elif opt_p == 'trafo':
            p_fit_ = lambda xx: p(trafo((xx), x[1]))
            # normalize
            p_fit = lambda xx: p_fit_(xx) / integrate.quad(p_fit_, -1, 1)[0]
        else:
            raise ValueError(f"opt_p={opt_p} not recognized.")
        
        return p_fit

    def zero(x):
        a = x[0]
        p_fit = get_p_fit(x)
        k = get_k(a)
        alpha_decay_is = get_alpha(k, p_fit)

        return (alpha_decay_is - alpha)**2


    from scipy.optimize import minimize
    callback = lambda x: logger.info(f"Current params: {x}")
    options = {"disp": True, "maxiter": 100}
    from types import SimpleNamespace
    misc = SimpleNamespace()

    if not opt_p:
        res = minimize(zero, x0=(1), bounds=[(1, 10)], tol=1e-1, method="L-BFGS-B", callback=callback, options=options)
    elif opt_p == 'proj':
        res = minimize(zero, x0=[1, 1], bounds=[(1, 10), (.1, 2)], tol=1e-3, method="L-BFGS-B", callback=callback, options=options)
        misc.alpha_proj = res.x[1]
        misc.A = get_A_spectral_projection(X, lmbd_out=None, alpha_proj=misc.alpha_proj)
    elif opt_p == 'trafo':
        res = minimize(zero, x0=(3, 2), bounds=[(1, 10), (.1, 10)], tol=1e-3, method="L-BFGS-B", callback=callback, options=options)
        misc.g = res.x[1]

    misc.param = res.x[0]
    k = get_k(misc.param)
    p_fit = get_p_fit(res.x)
    misc.p_fit = p_fit

    alpha_decay_fit = get_alpha(k, p_fit)
    logger.info(f"Found powerlaw a={misc.param:.2f} at alpha={alpha_decay_fit:.2f}, {misc}")

    return k, misc

# get the dimension of the Hilbert space in dimension d and at mode l
def z_dl(d, l, apx=False): 
    """
    Multiplicity of the spherical harmonic of degree l in dimension d
    """
    z_canatar = np.round(binom(d+l - 1 , l) - binom(d+l-3, l-2)).astype(int)
    z_canatar_apx = np.round(d**l/factorial(l)).astype(int)

    # incorrect
    z_dutordoir = np.round((2*l+d-2) * (gamma(l+d-2)) / (gamma(l+1)*gamma(d-2))).astype(int)
    z_wiki = np.round(binom(d+l - 1 , d-1) - binom(d+l-3, d-1)).astype(int)

    assert z_canatar == z_wiki
    # assert z_canatar == z_wiki

    if d > 30:
        logger.warning(f"Large degeneracy at dimension {d}.") 

    return z_canatar if not apx else z_canatar_apx

@np.vectorize
def num_l_to_num_ml(num_l, d):
    return np.sum([z_dl(d, l) for l in range(num_l)])

def num_ml_to_num_l(num_ml, d):
    num_ml_cand = np.arange(100)
    return np.where(num_ml < num_l_to_num_ml(num_ml_cand, d))[0][0]
    
# print(num_l_to_num_ml(5, 3))
# print(num_ml_to_num_l(2048, 3))


def get_lambda_int(l, d, shape_fn, p=None, xx_p=None, repeat_degenerate=False, error_level="warn"):
    """
    see Dutordoir2020, Funk-Hecke
    """
    p_uni = get_uniform_measure(d)
    if p is None:
        logger.warning("No measure p given, using uniform measure on the sphere.")
        p = p_uni
    
    if not hasattr(p, "__call__"):
        p_vals = p
        p = lambda xx: np.interp(xx, xx_p, p_vals)

    assert not d < 3, "Spherical harmonics undefined for d<3"

    @np.vectorize
    def get_lambda_int_(l, d):
        """
        Note: This follows the prescription at 
        https://djalil.chafai.net/blog/2021/05/22/the-funk-hecke-formula/

        Another working account is Canatar21. 
        """
        alpha = (d-2)/2
        ggb = gegenbauer(alpha=alpha, n=l)
        C_alpha_l = ggb(1)
        integrand = lambda xx: C_alpha_l**-1 * shape_fn(xx) * (ggb(xx) * p_uni(xx)) * (p(xx) / p_uni(xx))
        # custom integration routine
        assert np.allclose(funk_hecke_integrator(p), 1., atol=1e-2)
        integral = funk_hecke_integrator(integrand)


        assert not np.isnan(integral).any()
        lmbd_l = integral

        if lmbd_l < 0:
            if error_level:
                assert lmbd_l > -1e-2, (f"lmbd_l={lmbd_l} is not positive definite.")
            lmbd_l = 0.

        return lmbd_l, integrand


    lmbd_l, f = get_lambda_int_(l, d)

    PLOT = False
    if PLOT:
        l = np.atleast_1d(l)
        fig, axes = plt.subplots(np.ceil(len(l)**0.5).astype(int), np.ceil(len(l)**0.5).astype(int), constrained_layout=True, figsize=(20, 20))
        for iax, ax in enumerate((axes.flat)[:len(l)]):
            cths = np.linspace(-1,1,1000)
            ax.plot(cths, f[iax](cths), label=f"alk={lmbd_l[iax]:.2E}")
            ax.set_title(f"l={l[iax]}")
            ax.legend()
        fig.savefig("test_lambda_int.png", dpi=200)
            
    if repeat_degenerate:
        lmbd_l_ = []
        ls_= []
        for il, l_ in enumerate(np.atleast_1d(l)):
            ls_ += [l_] * z_dl(d, l_)
            lmbd_l_ += [lmbd_l[il]] * z_dl(d, l_)
        ls_ = np.array(ls_)
        lmbd_l = np.array(lmbd_l_)
    else:
        ls_ = l

    
    return ls_, lmbd_l

def funk_hecke_integrator(f):
    """
    Alternatively, a quadrature rule could be used, as detailed in 
    Canatar21, Supplementary Material, eq. (121f)
    """
    eps = 1e-4  # to avoid critical points in the 1 / ( 1 - xx**2 ) integrand
    xx_uni = np.linspace(-1 + eps, 1 - eps, int(1e3 + 1))
    a = 1.
    if a != 1: raise NotImplementedError
    xx_int = np.sign(xx_uni)*np.abs(np.sin(np.pi/2 * xx_uni))**a
    assert 0. in xx_int


    dx = np.diff(xx_int)
    assert (dx > 0).all()
    f_int = f(xx_int)
    f_int = np.array(f_int)

    # handle divergences at boundaries
    if not np.isfinite(f_int[-1]):
        f_int[0] = f_int[1]
    if not np.isfinite(f_int[-1]):
        f_int[-1] = f_int[-2]

    f_int = (f_int[1:] + f_int[:-1])/2
    overlap = np.sum(f_int*dx)
    assert np.isfinite(overlap)
    return overlap



def kernel_hist(d_opts, nbins=100, ax=None):
    (X, Y, x, y) = d_opts.data
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    YY = np.einsum("kp,lp->klp", Y, Y)

    YY = YY.mean(axis=-1).flatten()
    XX = (X @ X.T).flatten()

    YY_hist, bin_edg, show, w = bin_XX_YY_with_std(XX, YY, ax, nbins)

    XX_centers = bin_edg[:-1] + w / 2
    return XX_centers, YY_hist


def bin_XX_YY_with_std(XX, YY, bins=None, separate_1=False, exclude_1=False, interpolate_nan=False, return_empty=False, error_type="standard_deviation"):
    """
    Turns flattened input and output tensors into bins and values at bins, thus effectively discretizing the data.
    """
    # assert (XX <= 1).all() and (XX >= -1).all()

    assert error_type in ["standard_deviation", "standard_error"]
    one_thsd = 1e-6
    if exclude_1:
        YY = YY[np.abs(XX - 1) > one_thsd]
        XX = XX[np.abs(XX - 1) > one_thsd]
        separate_1 = False

    if separate_1:
        _, bins = np.histogram(XX.flatten(), bins=bins)
        bins = list(bins)
        bins = bins[:-1] + [1. - one_thsd, 1., ]

    XX_hist, bin_edg = np.histogram(XX, bins=bins)
    w = bin_edg[1:] - bin_edg[:-1]
    XX_centers = (bin_edg[1:] + bin_edg[:-1]) / 2
    YY_hist_mean = np.empty_like(XX_hist, dtype=float)
    YY_hist_std = np.empty_like(XX_hist, dtype=float)
    YY_counts = np.zeros_like(XX_hist, dtype=int)

    for i in range(len(bin_edg) - 1):
        hits = YY[(XX >= bin_edg[i]) & (XX <= bin_edg[i + 1])]
        if len(hits) > 0:
            YY_counts[i] = len(hits)
            YY_hist_mean[i] = hits.mean()
            YY_hist_std[i] = hits.std() / (len(hits) ** 0.5 if error_type == "standard_error" else 1)
            
        else:
            YY_hist_mean[i] = np.nan
            YY_hist_std[i] = np.nan

    if not return_empty:
        idx_nonempty = ~(np.isnan(YY_hist_mean) | np.isnan(YY_hist_std))
        XX_centers = XX_centers[idx_nonempty]
        YY_hist_mean = YY_hist_mean[idx_nonempty]
        YY_hist_std = YY_hist_std[idx_nonempty]
        YY_counts = YY_counts[idx_nonempty]

    # ax.bar(bin_edg[:-1] - 1, YY_hist, width=w, align='edge', color="gray", alpha=0.3)
    YY_hist_mean = (YY_hist_mean)
    YY_hist_std = (YY_hist_std)

    if interpolate_nan:
        nans, x_ = nan_helper(YY_hist_mean)
        YY_hist_mean[nans] = np.interp(x_(nans), x_(~nans), YY_hist_mean[~nans])
        nans, x_ = nan_helper(YY_hist_std)
        YY_hist_std[nans] = np.interp(x_(nans), x_(~nans), YY_hist_std[~nans])
    # ax.bar(-np.flip(bin_edg[1:]) + 1, YY_hist, width=np.flip(w), align='edge', edgecolor="gray", color="none",
    #        alpha=0.3, zorder=1)
    # ax.bar(-np.flip(bin_edg[1:]) + 1, height=2 * YY_hist_std, bottom=YY_hist - YY_hist_std, width=np.flip(w),
    #        align='edge', color="tab:blue", alpha=0.1, zorder=-1)

    return XX_centers, YY_hist_mean, YY_hist_std, YY_counts


def test_funk_hecke_integrator():
    f = lambda t: np.where(t<0, 1-np.tanh((t+1)**2), 1-np.tanh((t-1)**2))
    overlap1 = funk_hecke_integrator(f)
    from scipy.integrate import quad
    overlap2 = quad(f, -1, 1, limit=1000)
    print(overlap1)
    print(overlap2)

# @memory.cache
def fit_spectrum_decay(ranks, eigenvalues, where_fit=slice(None, None), N_rescale=1,**kwargs): 

    ranks = np.array(ranks)
    eigenvalues = np.array(eigenvalues)   

    if not np.isfinite(eigenvalues).all():
        logger.warning("Eigenvalues contain non-finite values.")
        return np.nan, lambda ranks: np.full_like(ranks, np.nan), where_fit

    if ranks.min() == 0:
        ranks += 1

    where_nonepsilon = eigenvalues > 1e-8

    # use the Kaiser criterion to determine the knee point of the spectrum
    if where_fit == "kaiser":
        logger.info("Using Kaiser criterion to determine knee point of spectrum.")
        where_fit = np.where(eigenvalues > 1 / N_rescale)[0]
        # ax.set_ylim(0.1 * 1 / N_rescale, None)
    elif where_fit == "knee":
        from kneed import KneeLocator
        log10_ranks = np.log10(ranks)
        # interpolate the eigenvalues on a log10 scale
        xx = np.linspace(log10_ranks.min(), log10_ranks.max(), 1000)
        yy = np.interp(xx, log10_ranks[where_nonepsilon], np.log10(eigenvalues[where_nonepsilon]))
        kl = KneeLocator(xx, yy, curve='concave', direction='decreasing', online=True)
        # plt.close("all")
        # plt.plot(kl.x_normalized, kl.y_normalized)
        # plt.plot(kl.x_difference, kl.y_difference)
        # plt.axvline(kl.knee, c="k", ls="--", lw=1)
        # plt.show()
        knee = kl.knee
        knee_index = np.argmin(np.abs(log10_ranks - knee))
        # ax.axvline(ranks[knee_index], c="k", ls="--", lw=1) 
        where_fit = ranks < ranks[knee_index]

        # ax.set_ylim(eigenvalues[knee_index] / 100, None)

    elif type(where_fit) == slice:
        where_fit_ = np.full_like(ranks, False, dtype=bool)
        where_fit_[where_fit] = True
        where_fit = where_fit_
    else:
        raise ValueError("where_fit must be either 'kaiser', 'knee', or a slice object.")
    
    where_fit_tot = where_fit & where_nonepsilon

    x = np.log(ranks[where_fit_tot])
    y = np.log(eigenvalues[where_fit_tot])

    slope, intercept = np.polyfit(x, y, 1)
    f_fit = lambda ranks: np.exp(intercept) * np.power(ranks, slope)

    alpha = -slope
    
    return alpha, f_fit, where_fit_tot

def l_to_ml(ls, N):
    """
    Move the indices l so that they reflect the multiplicities
    0    111   22222   
    0     2      6
    """
    ml = 0
    mls = [0]
    for l in ls[1:]:
        mls += [mls[-1] + (z_dl(N, l-1) - 1)//2 + (z_dl(N, l) + 1)//2]
    return np.array(mls)