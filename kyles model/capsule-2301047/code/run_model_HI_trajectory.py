#%% ============================================================================
# Diagnostic: does the model's HI(window) trend over the session?
# Same data-matched setup (trailing-20 baseline, 10-trial windows step 5, 1 seed).
# Plot HI(window) per form and behavior(window) vs window index; quantify the
# within-session trend (Spearman of each vs window index).
import os, sys
import numpy as np
from scipy.stats import spearmanr, linregress
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

FORMS = ['hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre']
FLAB = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
        '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
FCOL = ['#888', '#1baf7a', '#d4537e', '#378add']
WIN_SIZE, WIN_STEP, N_BASELINE, SEED = 10, 5, 20, 0
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ('hebb',), 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']


def trailing_baseline(A, w):
    bl = np.empty_like(A)
    for t in range(A.shape[0]):
        bl[t] = A[max(0, t - w + 1):t + 1].mean(0)
    return bl


params = kp.default_toy_params(seed=SEED, verbose=False)
kp.task_params, kp.train_params, kp.net_params = params
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
_po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
to = toa[0]; task_hist = task.hists[0]
rpe_est = kp.get_rpe_estimates(task_hist)
hit_t = np.asarray(rpe_est['hits'], float)
spd_t = np.asarray(rpe_est['speed_rpe'], float)

out = np.asarray(to['output'], float)
ts = np.asarray(task_hist['trial_starts'], int)
bounds = list(ts) + [out.shape[0]]
A = np.array([np.nanmean(out[bounds[k]:bounds[k + 1]], axis=0)
              for k in range(len(ts)) if bounds[k + 1] > bounds[k]])
D = A - trailing_baseline(A, N_BASELINE)
n_trials = min(A.shape[0], len(hit_t))
loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
Wv = to['W_rec_vals']
dW = (Wv[loss_step_divs[-1]] - Wv[0]).flatten()

win_starts = list(range(0, n_trials - WIN_SIZE + 1, WIN_STEP))
HI = {f: [] for f in FORMS}
hitw, spdw = [], []
for ws in win_starts:
    tr = slice(ws, ws + WIN_SIZE)
    Ad, Dd = A[tr], D[tr]
    forms = {'hebb': Ad.T @ Ad, 'dpost_pre': Dd.T @ Ad, 'dpost_dpre': Dd.T @ Dd, 'post_dpre': Ad.T @ Dd}
    for f, E in forms.items():
        ef = E.flatten()
        HI[f].append(linregress(ef, dW).slope if np.std(ef) > 0 else np.nan)
    hitw.append(np.nanmean(hit_t[tr])); spdw.append(np.nanmean(spd_t[tr]))
x = np.arange(len(win_starts))
hitw = np.array(hitw); spdw = np.array(spdw)

print("Within-session trend (Spearman vs window index), n_windows={}:".format(len(x)))
print("  {:16s} rho={:+.2f}".format('hit rate', spearmanr(x, hitw)[0]))
print("  {:16s} rho={:+.2f}".format('dSpeed(RPE)', spearmanr(x, spdw)[0]))
for f, l in zip(FORMS, ['r r', 'r dr', 'dr dr', 'dr r']):
    h = np.array(HI[f], float)
    print("  HI[{:6s}]       rho={:+.2f}".format(l, spearmanr(x[np.isfinite(h)], h[np.isfinite(h)])[0]))

fig = plt.figure(figsize=(6.6, 2.8))
fw, fh = fig.get_size_inches()
# panel A: HI trajectories (z-scored for overlay)
axA = fig.add_axes([0.75 / fw, 0.7 / fh, 2.7 / fw, 1.85 / fh])
for f, l, c in zip(FORMS, FLAB, FCOL):
    h = np.array(HI[f], float)
    hz = (h - np.nanmean(h)) / (np.nanstd(h) + 1e-12)
    axA.plot(x, hz, '-o', color=c, ms=3, lw=1, label=l)
axA.axhline(0, color='k', lw=0.6, alpha=0.4)
axA.set_xlabel('window index (session time ->)'); axA.set_ylabel('HI (z-scored)')
axA.set_title('Model HI(window) trajectories', fontsize=8)
axA.legend(fontsize=6, loc='best', frameon=False)
# panel B: behavior trajectories
axB = fig.add_axes([4.35 / fw, 0.7 / fh, 1.9 / fw, 1.85 / fh])
axB.plot(x, hitw, '-o', color='#c0392b', ms=3, lw=1.2, label='hit rate')
axB.set_xlabel('window index'); axB.set_ylabel('hit rate', color='#c0392b')
axB.set_title('Behavior over session', fontsize=8)
axB.tick_params(axis='y', labelcolor='#c0392b')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_HI_trajectory.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
