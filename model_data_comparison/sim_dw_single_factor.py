#%% ============================================================================
# Dirt-simple simulation: single factors leak when ΔW is BUILT from the traces
# ============================================================================
"""
Point to demonstrate: the model's single-factor "success" (pre_only / post_only
tracking RPE as well as the coactivity product) is a consequence of ΔW being
constructed as the product rule from the same traces you analyze -- NOT of
activity statistics (mean rate, dimensionality, pre-post correlation).

Setup:
  - Synthetic pre/post activity, RPE drawn INDEPENDENTLY of the activity
    (so corr(activity, RPE) = 0, matching our null mean-rate-vs-RPE probe).
  - ΔW = Σ_t RPE(t) · outer(dev_post(t), pre(t))   [the three-factor product]
  - Compute HI (slope of ΔW vs eligibility, per division) for the product and
    for each single factor; correlate the HI series with RPE across divisions.

Predictions:
  1. Single factors (pre_only, post_only, dev_only) track RPE ~ as well as the
     product, even though the activity is uncorrelated with RPE -> leakage is
     from the ΔW construction.
  2. Sweeping the latent dimensionality of the activity does NOT remove it ->
     dimensionality is orthogonal.
  3. If ΔW is instead built from an INDEPENDENT realization (real-plasticity
     analog), everything -- product included -- goes to ~0.
"""
import numpy as np
from scipy.stats import spearmanr

N, T, NDIV, TAU = 100, 6000, 40, 50


def _traces(rng, dim):
    L = rng.standard_normal((T, dim))
    M = rng.standard_normal((dim, N))
    return np.maximum(L @ M + 0.3 * rng.standard_normal((T, N)), 0.0)  # rectified


def _running_dev(post, tau=TAU):
    base = np.zeros_like(post)
    b = np.zeros(post.shape[1])
    g = 1.0 - 1.0 / tau
    for t in range(post.shape[0]):
        b = g * b + (1 - g) * post[t]
        base[t] = b
    return post - base


def run(dim, self_generated=True, seed=0, imbalance=0.0):
    rng = np.random.default_rng(seed)
    pre = _traces(rng, dim)
    post = _traces(rng, dim)          # independent pre/post populations
    rpe = rng.standard_normal(T)       # RPE INDEPENDENT of activity
    dev = _running_dev(post)
    # imbalance: shift every neuron's deviation so Σ_i dev_i > 0 systematically
    # (imbalanced deviations, as the manuscript notes for the model / Miconi).
    if imbalance:
        dev = dev + imbalance

    if self_generated:
        dW = np.einsum('t,ti,tj->ij', rpe, dev, pre)
    else:  # ΔW from a DIFFERENT realization -> not the product of these traces
        pre2 = _traces(rng, dim)
        post2 = _traces(rng, dim)
        dev2 = _running_dev(post2)
        dW = np.einsum('t,ti,tj->ij', rng.standard_normal(T), dev2, pre2)
    dWf = dW.ravel()

    edges = np.linspace(0, T, NDIV + 1).astype(int)
    hi = {k: np.zeros(NDIV) for k in ('product', 'pre_only', 'post_only', 'dev_only')}
    rpe_div = np.zeros(NDIV)
    for d in range(NDIV):
        s = slice(edges[d], edges[d + 1])
        rpe_div[d] = rpe[s].sum()
        P = pre[s].sum(0)
        Q = post[s].sum(0)
        DV = dev[s].sum(0)
        elig = {
            'product':   dev[s].T @ pre[s],   # Σ_t outer(dev_t, pre_t) -- correct
            'pre_only':  np.broadcast_to(P[None, :], (N, N)),
            'post_only': np.broadcast_to(Q[:, None], (N, N)),
            'dev_only':  np.broadcast_to(DV[:, None], (N, N)),
        }
        for k, E in elig.items():
            hi[k][d] = np.polyfit(np.asarray(E).ravel(), dWf, 1)[0]
    return {k: spearmanr(hi[k], rpe_div)[0] for k in hi}


def avg(dim, self_generated=True, imbalance=0.0, nseed=10):
    vals = [run(dim, self_generated, seed=s, imbalance=imbalance) for s in range(nseed)]
    return {k: np.mean([v[k] for v in vals]) for k in vals[0]}


def show(title, **kw):
    print(title)
    print("{:>10s}  {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "latent dim", "product", "pre_only", "post_only", "dev_only"))
    for dim in (2, 5, 20, 100):
        r = avg(dim, **kw)
        print("{:>10d}  {:+8.2f} {:+8.2f} {:+8.2f} {:+8.2f}".format(
            dim, r['product'], r['pre_only'], r['post_only'], r['dev_only']))
    print()


print("corr(HI, RPE)  --  dW = sum_t RPE * dev_post * pre  (activity independent of RPE)\n")
show("(A) BALANCED deviations (imbalance=0): product-rule dW, no single-factor leak",
    self_generated=True, imbalance=0.0)
show("(B) IMBALANCED deviations (imbalance=1.0): learning-like upward ramp",
    self_generated=True, imbalance=1.0)
print("Control: dW built from an INDEPENDENT realization (real-plasticity analog)")
r = avg(20, self_generated=False, imbalance=1.0)
print("{:>10s}  {:+8.2f} {:+8.2f} {:+8.2f} {:+8.2f}".format(
    "dim=20", r['product'], r['pre_only'], r['post_only'], r['dev_only']))
