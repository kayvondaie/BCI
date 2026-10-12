"""
Baseline-injection manipulation (called by run_model_hi_multiseed.py).

Question: what change to the model shrinks corr(dW, r_pre*r_post) [raw] while
leaving corr(dW, r_pre*(r_post - avg)) [fluct] intact?

Test: the rule y = dW is fixed (already trained). Add large, heterogeneous,
task-INDEPENDENT per-neuron baseline offsets to the POST activity used in the
eligibility. Because the running-average deviation (r_post - avg) is invariant
to a per-neuron baseline shift, this simulates neurons with larger tonic firing
WITHOUT changing the plasticity: y and the fluctuation eligibility are untouched
by construction; only the raw (and mean-drive) forms feel it.

Prediction: as the baseline grows, corr(dW, raw) collapses while corr(dW, fluct)
holds -- i.e. the model becomes sensitive to eligibility form, like the data.
"""
import numpy as np
from scipy.stats import spearmanr, linregress
import kyle_pipeline as kp   # for get_div_idxs (trial-based divisions)


def _running_mean(x, n_window):
    # matches net_helpers.accumulate_decay: gamma = 1 - 1/n_window
    g = 1.0 - 1.0 / n_window
    b = np.zeros(x.shape[1])
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        b = g * b + (1.0 - g) * x[t]
        out[t] = b
    return out


def sweep_baseline(train_outputs, task_hist, params, hi_params, scales, seed=0):
    """
    Returns {scale: (corr_raw, corr_fluct, corr_mean_drive)} where each corr is
    spearman(HI(division), RPE) with a per-neuron post baseline of that scale
    (in units of the activity std) injected into the eligibility.
    """
    task_params, train_params, net_params = params
    output = train_outputs['output']              # (n_steps, n_neu); pre = post = output
    W = train_outputs['W_rec_vals']
    n_spl = train_params['n_steps_per_loss']
    loss_steps = train_outputs['loss_steps']
    total_rpes = train_outputs['total_rpes']
    n_win_base = train_params['n_window_baseline']

    loss_step_divs = kp.get_div_idxs(task_hist, loss_steps, hi_params)
    n_div = int(len(loss_step_divs) - 1)
    dWf = (W[loss_step_divs[-1]] - W[0]).ravel()  # full-session dW (fixed target)

    dev = output - _running_mean(output, n_win_base)   # baseline-invariant deviation
    n_neu = output.shape[1]
    act_scale = float(np.nanstd(output))
    rng = np.random.default_rng(seed)

    # per-division raw step ranges + RPE
    ranges = []
    rpe_div = np.zeros(n_div)
    for d in range(n_div):
        s0 = max(0, loss_steps[loss_step_divs[d]] - n_spl + 1)
        s1 = loss_steps[loss_step_divs[d + 1]] + 1
        ranges.append((s0, s1))
        rpe_div[d] = np.sum(total_rpes[loss_step_divs[d]:loss_step_divs[d + 1]])

    def _corr(h):
        ok = np.isfinite(h) & np.isfinite(rpe_div)
        return spearmanr(h[ok], rpe_div[ok])[0] if ok.sum() >= 3 else np.nan

    out = {}
    for scale in scales:
        # heterogeneous, positive, task-independent per-neuron baseline (post side)
        c = scale * act_scale * np.abs(rng.standard_normal(n_neu))
        hi_raw = np.full(n_div, np.nan)
        hi_fluct = np.full(n_div, np.nan)
        hi_md = np.full(n_div, np.nan)
        for d, (s0, s1) in enumerate(ranges):
            pre = output[s0:s1]                       # (win, n_neu)
            post_mod = output[s0:s1] + c[None, :]     # baseline injected into POST
            E_raw = (post_mod.T @ pre).ravel()        # r_pre * r_post (raw)
            E_fluct = (dev[s0:s1].T @ pre).ravel()    # r_pre * (r_post - avg): unchanged
            E_md = E_raw - E_fluct                     # r_pre * avg (mean-drive)
            if np.std(E_raw) > 0:
                hi_raw[d] = linregress(E_raw, dWf).slope
            if np.std(E_fluct) > 0:
                hi_fluct[d] = linregress(E_fluct, dWf).slope
            if np.std(E_md) > 0:
                hi_md[d] = linregress(E_md, dWf).slope
        out[scale] = (_corr(hi_raw), _corr(hi_fluct), _corr(hi_md))
    return out
