#%% ============================================================================
# Why the DATA is sensitive to eligibility form and the MODEL analysis isn't
# ============================================================================
"""
Eligibility forms (all r_pre * [post factor]):
    raw        = r_pre * r_post
    fluct      = r_pre * (r_post - avg)        <- matches the true rule
    mean_drive = r_pre * avg(r_post)

True rule:  dW = sum_t RPE(t) * (r_post - avg)(t) (x) r_pre(t)   [fluctuation-based]

Hypothesis:
  - SENSITIVITY is the expected behavior. When the analyzed activity is NOT
    co-generated with RPE (a genuine, independent test), only the matching form
    (fluct) recovers the rule; raw and mean_drive fail. == DATA.
  - INSENSITIVITY is the anomaly. When the activity is co-generated with RPE
    (as in a network trained by RPE-gated updates), the baseline/avg ALSO carries
    RPE, so mean_drive and raw work too. == MODEL.
  - RPE is autocorrelated (like the real reward-prediction-error, which relaxes
    over ~10 trials), so a running baseline can pick it up.
  - The co-generated mode is zero-mean ACROSS NEURONS, so population mean rate
    stays ~flat vs RPE (matching our null mean-rate-vs-RPE probe on both sides).
"""
import numpy as np
from scipy.stats import spearmanr

N, T, NDIV, TAU = 120, 8000, 40, 60


def traces(rng, dim):
    L = rng.standard_normal((T, dim))
    M = rng.standard_normal((dim, N))
    return np.maximum(L @ M + 0.3 * rng.standard_normal((T, N)), 0.0)


def running_base(x, tau=TAU):
    b = np.zeros(x.shape[1]); out = np.zeros_like(x); g = 1.0 - 1.0 / tau
    for t in range(x.shape[0]):
        b = g * b + (1.0 - g) * x[t]; out[t] = b
    return out


def smooth_rpe(rng, tau=40):
    # autocorrelated RPE (AR1), like reward-prediction-error relaxing over trials
    r = np.zeros(T); g = 1.0 - 1.0 / tau
    for t in range(1, T):
        r[t] = g * r[t - 1] + rng.standard_normal()
    return (r - r.mean()) / r.std()


def run(dim, rpe_coupled, seed=0):
    rng = np.random.default_rng(seed)
    pre = traces(rng, dim)
    post = traces(rng, dim)
    rpe = smooth_rpe(rng)
    if rpe_coupled:
        # activity co-generated with RPE: a zero-neuron-mean mode whose amplitude
        # tracks RPE (keeps population mean rate ~flat vs RPE)
        w = rng.standard_normal(N); w = w - w.mean()
        post = post + 1.5 * rpe[:, None] * w[None, :]
    base = running_base(post)
    dev = post - base
    dW = np.einsum('t,ti,tj->ij', rpe, dev, pre)   # true fluctuation rule

    edges = np.linspace(0, T, NDIV + 1).astype(int)
    keys = ('raw', 'fluct', 'mean_drive')
    hi = {k: np.zeros(NDIV) for k in keys}
    rd = np.zeros(NDIV)
    for d in range(NDIV):
        s = slice(edges[d], edges[d + 1])
        rd[d] = rpe[s].sum()
        E = {'raw': post[s].T @ pre[s],
             'fluct': dev[s].T @ pre[s],
             'mean_drive': base[s].T @ pre[s]}
        for k in keys:
            hi[k][d] = np.polyfit(E[k].ravel(), dW.ravel(), 1)[0]
    return {k: spearmanr(hi[k], rd)[0] for k in keys}


def avg(dim, rpe_coupled, nseed=8):
    v = [run(dim, rpe_coupled, s) for s in range(nseed)]
    return {k: np.mean([x[k] for x in v]) for k in v[0]}


print("corr(HI, RPE) for the three eligibility forms;  dW = fluctuation rule\n")
for label, coupled in [("DATA-like  (activity independent of RPE)", False),
                       ("MODEL-like (activity co-generated with RPE)", True)]:
    print(label)
    print("  {:>6s} {:>8s} {:>8s} {:>11s}".format("dim", "raw", "fluct", "mean_drive"))
    for dim in (5, 20, 100):
        r = avg(dim, coupled)
        print("  {:>6d} {:+8.2f} {:+8.2f} {:+11.2f}".format(
            dim, r['raw'], r['fluct'], r['mean_drive']))
    print()
