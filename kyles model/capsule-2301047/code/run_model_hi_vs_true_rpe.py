#%% ============================================================================
# The clean question: for each of the 4 eligibility forms, does its HI track the
# TRUE RPE, and which form wins?  (No behavioral proxies -- true RPE directly.)
# Two baseline conventions for the analysis eligibility:
#   'native'  = Kyle's running-EMA baseline (compute_local_hebbian_indexes)
#   'fixed'   = data-style first-20-trial fixed baseline (per-trial epoch acts)
# Prediction: native collapses the deviation forms (all ~equal); fixed separates.
import os, sys
import numpy as np
from scipy.stats import spearmanr, linregress, wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

ELIG = ('hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre')
FORM_LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
NPD = 5; N_BASELINE = 20
hi_params = {'div_mode': 'trials', 'n_trials_per_div': NPD, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(12))


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]; task_hist = task.hists[0]
    rpes_divs, _dl, ds_full, _fs, _e, _m = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params)
    true_rpe = np.asarray(rpes_divs.get('true'), float)

    # --- native (EMA baseline) HI vs true RPE ---
    r_native = np.full(len(ELIG), np.nan)
    for fi in range(len(ELIG)):
        hi = np.asarray(ds_full[fi], float)
        ok = np.isfinite(hi) & np.isfinite(true_rpe)
        if ok.sum() >= 3 and np.std(hi[ok]) > 0 and np.std(true_rpe[ok]) > 0:
            r_native[fi] = spearmanr(hi[ok], true_rpe[ok])[0]

    # --- fixed (first-20-trial) baseline HI vs true RPE ---
    out = np.asarray(to['output'], float)
    ts = np.asarray(task_hist['trial_starts'], int)
    bounds = list(ts) + [out.shape[0]]
    A = np.array([np.nanmean(out[bounds[k]:bounds[k+1]], axis=0)
                  for k in range(len(ts)) if bounds[k+1] > bounds[k]])
    D = A - A[:min(N_BASELINE, A.shape[0])].mean(0)
    Wv = to['W_rec_vals']; dW = (Wv[-1] - Wv[0]).flatten()
    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    n_div = loss_step_divs.shape[0] - 1
    forms_fn = [lambda Ad, Dd: Ad.T @ Ad, lambda Ad, Dd: Dd.T @ Ad,
                lambda Ad, Dd: Dd.T @ Dd, lambda Ad, Dd: Ad.T @ Dd]
    HIf = {fi: np.full(n_div, np.nan) for fi in range(len(ELIG))}
    for d in range(n_div):
        s, e = d * NPD, (d + 1) * NPD
        if e > A.shape[0]:
            break
        Ad, Dd = A[s:e], D[s:e]
        for fi in range(len(ELIG)):
            ef = forms_fn[fi](Ad, Dd).flatten()
            if np.std(ef) > 0:
                HIf[fi][d] = linregress(ef, dW).slope
    r_fixed = np.full(len(ELIG), np.nan)
    for fi in range(len(ELIG)):
        hi = HIf[fi][:len(true_rpe)]; tr = true_rpe[:len(hi)]
        ok = np.isfinite(hi) & np.isfinite(tr)
        if ok.sum() >= 3 and np.std(hi[ok]) > 0 and np.std(tr[ok]) > 0:
            r_fixed[fi] = spearmanr(hi[ok], tr[ok])[0]
    return r_native, r_fixed


NAT, FIX = [], []
for s in SEEDS:
    try:
        rn, rf = run_seed(s); NAT.append(rn); FIX.append(rf)
        print("seed {} | native {} | fixed {}".format(
            s, np.round(rn, 2), np.round(rf, 2)))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
NAT = np.array(NAT); FIX = np.array(FIX)
np.save(os.path.join(OUT, 'model_hi_vs_true_rpe.npy'),
        {'native': NAT, 'fixed': FIX, 'labels': FORM_LABEL}, allow_pickle=True)


def star(v):
    v = v[np.isfinite(v)]
    p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else ''


print("\ncorr(HI, TRUE RPE) per form (mean over {} seeds):".format(len(NAT)))
for name, R in [('native (EMA)', NAT), ('fixed (first-20)', FIX)]:
    print("  {:18s}".format(name) + "".join("{:+6.2f}{:<3s}".format(
        np.nanmean(R[:, fi]), star(R[:, fi])) for fi in range(len(ELIG))))

fig = plt.figure(figsize=(4.6, 3.0)); fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.0/fw, 0.75/fh, 3.3/fw, 1.9/fh])
x = np.arange(len(ELIG))
for j, (name, R, c) in enumerate([('native (EMA baseline)', NAT, '#c0392b'),
                                  ('fixed (first-20 baseline)', FIX, '#2c3e50')]):
    m = np.nanmean(R, 0); se = np.nanstd(R, 0) / np.sqrt(len(R))
    ax.bar(x + (j - 0.5) * 0.4, m, 0.38, yerr=se, color=c, capsize=2, label=name)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(FORM_LABEL, fontsize=7)
ax.set_ylabel('corr(HI, true RPE)')
ax.set_title('Which form''s HI best matches true RPE? (n={})'.format(len(NAT)))
ax.legend(frameon=False, fontsize=6.5)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'model_hi_vs_true_rpe.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
