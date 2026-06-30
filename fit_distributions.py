# ============================================================
# fit_distributions.py  (DISCRETE + CSN)
# ============================================================

import numpy as np
import scipy.stats as st
from scipy.special import zeta as hurwitz_zeta
from scipy.optimize import minimize, minimize_scalar
import scipy.optimize as opt
from scipy.optimize import brentq
from mpmath import zeta as mp_zeta


def profile_likelihood_CI(nll_func, params_mle, data, delta=3.84):
    """
    Compute 95% profile-likelihood confidence intervals for each parameter.
    nll_func: function(params, data) → NLL
    params_mle: array of MLE parameter values
    data: the dataset
    delta: chi-square cutoff (3.84 for 95% CI with 1 d.o.f.)
    """
    params_mle = np.array(params_mle, dtype=float)
    k = len(params_mle)
    CIs = []

    nll_min = nll_func(params_mle, data)

    for i in range(k):
        theta_hat = params_mle[i]

        def objective(theta):
            p = params_mle.copy()
            p[i] = theta
            return nll_func(p, data)

        # Search bounds – you may customize depending on parameter type
        lower, upper = None, None

        # ----------------------- LOWER BOUND -----------------------
        def root_low(theta):
            return 2 * (objective(theta) - nll_min) - delta

        try:
            lower = brentq(root_low, theta_hat * 0.01, theta_hat)
        except:
            lower = np.nan

        # ----------------------- UPPER BOUND -----------------------
        def root_high(theta):
            return 2 * (objective(theta) - nll_min) - delta

        try:
            upper = brentq(root_high, theta_hat, theta_hat * 50.0)
        except:
            upper = np.nan

        CIs.append((lower, upper))

    return CIs


def zeta_float(alpha, xmin):
    """Return Hurwitz zeta(alpha, xmin) as a float instead of mpf."""
    return float(mp_zeta(alpha, xmin))


def loglik_discrete_pl(alpha, data, xmin):
    """
    Log-likelihood of discrete power law:
        P(k) ∝ k^{-alpha}, k >= xmin
    """
    data = data[data >= xmin]
    n = len(data)
    if n == 0:
        return -np.inf

    # Convert mpf -> float
    z = zeta_float(alpha, xmin)
    return -n * np.log(z) - alpha * np.sum(np.log(data))


def ci_alpha_discrete(alpha_hat, xmin, data):
    """
    95% CI for alpha in a discrete power law, using the likelihood ratio method
    recommended in Clauset-Shalizi-Newman (2009).
    Solves: 2*(LL(a_hat) - LL(a)) = 3.84
    """
    data = np.asarray(data)
    data = data[data >= xmin]

    LL_hat = loglik_discrete_pl(alpha_hat, data, xmin)

    def lr(a):
        return 2 * (LL_hat - loglik_discrete_pl(a, data, xmin)) - 3.84

    # Lower bound search bracket
    try:
        lower = brentq(lr, max(1.01, alpha_hat * 0.1), alpha_hat)
    except:
        lower = np.nan

    # Upper bound search bracket
    try:
        upper = brentq(lr, alpha_hat, alpha_hat * 10.0)
    except:
        upper = np.nan

    return (lower, upper)



# ============================================================
# 0) Discrete Power-Law Likelihood
# ============================================================

def nll_powerlaw_discrete_fixed(alpha, xmin, data):
    if alpha <= 1:
        return np.inf

    x = data[data >= xmin]
    if len(x) == 0:
        return np.inf

    return -( -alpha * np.sum(np.log(x))
              - len(x) * np.log(hurwitz_zeta(alpha, xmin)) )


def mle_alpha_discrete(data, xmin):
    xtail = data[data >= xmin]
    if len(xtail) == 0:
        return None

    def objective(alpha):
        return nll_powerlaw_discrete_fixed(alpha, xmin, data)

    res = minimize_scalar(objective, bounds=(1.01, 10), method="bounded")
    return res.x


def estimate_xmin_discrete(data):
    data = np.sort(data)
    unique_vals = np.unique(data)

    best_ks = np.inf
    best_xmin = None
    best_alpha = None

    for xmin in unique_vals:
        xtail = data[data >= xmin]
        if len(xtail) < 50:
            continue

        alpha = mle_alpha_discrete(data, xmin)
        if alpha is None:
            continue

        ecdf = np.arange(1, len(xtail)+1) / len(xtail)

        zmin = hurwitz_zeta(alpha, xmin)
        model_cdf = (zmin - hurwitz_zeta(alpha, xtail+1)) / zmin

        ks = np.max(np.abs(model_cdf - ecdf))

        if ks < best_ks:
            best_ks = ks
            best_xmin = xmin
            best_alpha = alpha

    return best_alpha, best_xmin, best_ks


def fit_powerlaw_discrete_CSN(data):
    alpha_hat, xmin_hat, ks = estimate_xmin_discrete(data)
    nll = nll_powerlaw_discrete_fixed(alpha_hat, xmin_hat, data)
    return np.array([alpha_hat, xmin_hat]), nll


# ============================================================
# 1) Discrete Exponential
# ============================================================

def nll_exponential(params, data):
    lam, xmin = params
    if lam <= 0 or xmin < 1:
        return np.inf

    x = data[data >= xmin].astype(int)
    if len(x) == 0:
        return np.inf

    # geometric-like pmf
    log_p = np.log1p(-np.exp(-lam)) - lam * (x - xmin)
    return -np.sum(log_p)


# ============================================================
# 2) Discrete Lognormal (correct tail normalization)
# ============================================================

def nll_lognormal(params, data):
    mu, sigma, xmin = params
    if sigma <= 0 or xmin < 1:
        return np.inf

    x = data[data >= xmin].astype(int)
    if len(x) == 0:
        return np.inf

    s = sigma
    scale = np.exp(mu)

    lower = x - 0.5
    upper = x + 0.5
    p_raw = st.lognorm.cdf(upper, s=s, scale=scale) - st.lognorm.cdf(lower, s=s, scale=scale)
    p_raw = np.clip(p_raw, 1e-300, None)

    # ★ truncation normalizer
    Z = 1 - st.lognorm.cdf(xmin - 0.5, s=s, scale=scale)
    if Z <= 0:
        return np.inf

    p = p_raw / Z
    return -np.sum(np.log(p))


# ============================================================
# 3) Discrete Truncated Power Law (FIXED NORMALIZATION)
# ============================================================

def nll_truncated_powerlaw(params, data):
    alpha, lamb, xmin  = params
    if alpha <= 1 or lamb < 0 or xmin < 1:
        return np.inf

    x = data[data >= xmin].astype(int)
    if len(x) == 0:
        return np.inf

    # unnormalized log pmf
    log_unnorm = -alpha * np.log(x) - lamb * x

    # Normalization constant Z (truncated PL)
    Z = 0.0
    k = int(xmin)
    cutoff = x.max() + 20000
    while k <= cutoff:
        term = (k ** -alpha) * np.exp(-lamb * k)
        if term < 1e-300:
            break
        Z += term
        k += 1

    if Z <= 0:
        return np.inf

    return -(np.sum(log_unnorm - np.log(Z)))



# ============================================================
# 4) Fitting Wrapper
# ============================================================

def fit_distribution(data, model, use_csn_xmin=False):
    """
    Fit distributions. By default (use_csn_xmin=True) this uses the CSN
    xmin from the discrete PL and fits all models on the same tail x >= xmin.
    Returns (params_array, nll)
    Param convention:
      - powerlaw_discrete -> [alpha, xmin]
      - exponential -> [lam, xmin]
      - lognormal -> [mu, sigma, xmin]
      - truncated_powerlaw -> [alpha, lamb, xmin]
    """
    data = np.array(data)

    # get csn xmin/alpha for consistency
    if use_csn_xmin:
        # estimate_xmin_discrete returns (alpha_hat, xmin_hat, ks)
        alpha_csn, xmin_csn, ksn = estimate_xmin_discrete(data)
        if xmin_csn is None:
            xmin_csn = int(np.min(data))
    else:
        xmin_csn = int(np.min(data))

    xmin0 = int(xmin_csn)

    # --------------------------
    # POWER-LAW (use CSN function)
    # --------------------------
    if model == "powerlaw_discrete":
        # Use CSN routine which already searches xmin and alpha
        params, nll = fit_powerlaw_discrete_CSN(data)
        ks_emp, pval = goodness_of_fit_pl(data, alpha_hat, params[1], n_sims=500)
        print("PL KS:", ks_emp, "p-value:", pval)

        # ensure correct types
        params = np.array([float(params[0]), int(params[1])])
        nll = float(nll)
        return model, params, nll

    # --------------------------
    # TRUNCATED POWER-LAW
    # --------------------------
    elif model == "truncated_powerlaw":
        # obtain starting guesses from CSN
        alpha0, xmin_from_csn, ks = estimate_xmin_discrete(data)
        xmin_use = int(xmin_from_csn) if xmin_from_csn is not None else xmin0

        # We will minimize over [alpha, lamb] and pass xmin to the objective
        def obj_trpl(p):
            # p = [alpha, lamb]
            # call the nll you defined which expects params=[alpha,lamb,xmin]
            return nll_truncated_powerlaw([p[0], p[1], xmin_use], data)

        res = minimize(
            obj_trpl,
            x0=np.array([alpha0 if alpha0 is not None else 2.0, 0.01]),
            bounds=[(1.01, 50.0), (1e-12, 10.0)]
        )

        alpha_hat, lambda_hat = float(res.x[0]), float(res.x[1])
        params = np.array([alpha_hat, lambda_hat, xmin_use])
        nll = float(res.fun)
        return model, params, nll

    # --------------------------
    # EXPONENTIAL (discrete tail)
    # --------------------------
    elif model == "exponential":
        xmin_use = xmin0
        tail = data[data >= xmin_use]
        if len(tail) == 0:
            return np.array([np.nan, xmin_use]), np.inf

        # minimize over lambda only; nll_exponential expects params = [lam, xmin]
        def obj_lam(lam):
            return nll_exponential([lam, xmin_use], data)

        res = minimize_scalar(obj_lam, bounds=(1e-12, 10.0), method="bounded")
        lam_hat = float(res.x)
        params = np.array([lam_hat, xmin_use])
        nll = float(res.fun)
        return model, params, nll

    # --------------------------
    # LOGNORMAL (discretized tail)
    # --------------------------
    elif model == "lognormal":
        xmin_use = xmin0
        tail = data[data >= xmin_use]
        if len(tail) == 0:
            return np.array([np.nan, np.nan, xmin_use]), np.inf

        mu0 = float(np.mean(np.log(tail)))
        sigma0 = float(np.std(np.log(tail)))
        # nll_lognormal expects params = [mu, sigma, xmin]
        res = minimize(
            lambda p: nll_lognormal([p[0], p[1], xmin_use], data),
            x0=np.array([mu0, max(1e-3, sigma0)]),
            bounds=[(None, None), (1e-6, None)]
        )
        mu_hat, sigma_hat = float(res.x[0]), float(res.x[1])
        params = np.array([mu_hat, sigma_hat, xmin_use])
        nll = float(res.fun)
        return model, params, nll

    else:
        raise ValueError(f"Unknown distribution: {model}")



# ============================================================
# 5) G-test / AIC / CI (unchanged)
# ============================================================

def g_test(nll1, nll2, df=1):
    G = 2*((-nll1) - (-nll2))
    p = st.chi2.sf(G, df)
    return G, p

def compute_aic(nll, k):
    return 2*k + 2*nll

def akaike_weights(aic_values):
    rel = np.exp(-0.5*(aic_values - np.min(aic_values)))
    return rel / np.sum(rel)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.special import zeta as hurwitz_zeta

def _discrete_pl_pmf_array(alpha, xmin, kmax):
    ks = np.arange(xmin, kmax+1)
    pmf = ks.astype(np.float64) ** (-alpha)
    Z = pmf.sum()
    pmf /= Z
    return ks, pmf

def sample_discrete_pl(alpha, xmin, size, kmax=None, rng=None):
    """Sample `size` integers >= xmin from discrete PL using direct pmf over [xmin, kmax]."""
    if rng is None:
        rng = np.random.default_rng()
    if kmax is None:
        kmax = max(xmin + 10000, xmin + 5*size)  # generous bound
    ks, pmf = _discrete_pl_pmf_array(alpha, xmin, kmax)
    return rng.choice(ks, size=size, p=pmf)

def ks_stat_discrete_pl(alpha, xmin, data_tail):
    """KS between empirical tail CDF and discrete PL CDF (x >= xmin)."""
    data_tail = np.sort(np.array(data_tail))
    n = len(data_tail)
    if n == 0:
        return np.nan
    # Empirical CDF at each unique value
    uniq, counts = np.unique(data_tail, return_counts=True)
    ecdf = np.cumsum(counts) / n
    # The discrete PL CDF: P(X <= x) = 1 - zeta(alpha, x+1)/zeta(alpha, xmin)
    zmin = hurwitz_zeta(alpha, xmin)
    model_cdf = 1.0 - hurwitz_zeta(alpha, uniq + 1) / zmin
    ks = np.max(np.abs(ecdf - model_cdf))
    return ks

def goodness_of_fit_pl(data, alpha_hat, xmin_hat, n_sims=500, rng=None):
    """
    Clauset-style goodness-of-fit p-value for discrete PL.
    data: full data array (integers)
    alpha_hat, xmin_hat: from MLE/CSN
    Returns: (ks_emp, p_value)
    """
    if rng is None:
        rng = np.random.default_rng()
    data = np.array(data)
    below = data[data < xmin_hat]
    tail = data[data >= xmin_hat]
    n_tail = len(tail)
    if n_tail == 0:
        return np.nan, np.nan

    # Empirical KS
    ks_emp = ks_stat_discrete_pl(alpha_hat, xmin_hat, tail)

    ks_sims = []
    for _ in range(n_sims):
        # Sample new below-xmin by sampling with replacement from empirical below
        if len(below) > 0:
            below_sim = rng.choice(below, size=len(below), replace=True)
        else:
            below_sim = np.array([], dtype=int)

        # Sample tail synthetic from fitted discrete PL
        tail_sim = sample_discrete_pl(alpha_hat, xmin_hat, size=n_tail, rng=rng)

        # combine
        synth = np.concatenate([below_sim, tail_sim])
        # re-fit alpha on the synthetic tail using same xmin (Clauset suggests re-estimating alpha)
        alpha_sim = mle_alpha_discrete(synth, xmin_hat)
        if alpha_sim is None:
            ks_sims.append(np.nan); continue
        ks_sim = ks_stat_discrete_pl(alpha_sim, xmin_hat, synth[synth >= xmin_hat])
        ks_sims.append(ks_sim)

    ks_sims = np.array(ks_sims)
    ks_sims = ks_sims[~np.isnan(ks_sims)]
    p_value = np.mean(ks_sims >= ks_emp) if len(ks_sims) > 0 else np.nan
    return ks_emp, p_value

