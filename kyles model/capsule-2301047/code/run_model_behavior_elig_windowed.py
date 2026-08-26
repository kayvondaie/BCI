#%% ============================================================================
# Model HI vs behavior x eligibility, fully data-matched convention, ONE run:
#   - per-trial activity, TRAILING-20 baseline
#   - 10-trial SLIDING windows, step 5 (like the data: WIN_SIZE=10, WIN_STEP=5)
#   - HI(window) = slope(whole-session dW, per-window elig)
#   - behavior(window) = mean per-trial outcome over the window
#   - corr(HI, outcome) across windows (single session).
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
FORM_LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
BEH = ['hits', 'speed_rpe', 'speed', 'hits_rpe']
BEH_LABEL = ['Hit rate', '$\\Delta$Speed (RPE)', 'Speed', 'Hit $\\times$ RPE']
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
beh_trial = {b: np.asarray(rpe_est[b], float) for b in BEH}

out = np.asarray(to['output'], float)
ts = np.asarray(task_hist['trial_starts'], int)
bounds = list(ts) + [out.shape[0]]
A = np.array([np.nanmean(out[bounds[k]:bounds[k + 1]], axis=0)
              for k in range(len(ts)) if bounds[k + 1] > bounds[k]])
D = A - trailing_baseline(A, N_BASELINE)
n_trials = min(A.shape[0], min(len(beh_trial[b]) for b in BEH))

loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
Wv = to['W_rec_vals']
dW = (Wv[loss_step_divs[-1]] - Wv[0]).flatten()

win_starts = list(range(0, n_trials - WIN_SIZE + 1, WIN_STEP))
HI = {f: [] for f in FORMS}
BW = {b: [] for b in BEH}
for ws in win_starts:
    tr = slice(ws, ws + WIN_SIZE)
    Ad, Dd = A[tr], D[tr]
    forms = {'hebb': Ad.T @ Ad, 'dpost_pre': Dd.T @ Ad, 'dpost_dpre': Dd.T @ Dd, 'post_dpre': Ad.T @ Dd}
    for f, E in forms.items():
        ef = E.flatten()
        HI[f].append(linregress(ef, dW).slope if np.std(ef) > 0 else np.nan)
    for b in BEH:
        BW[b].append(np.nanmean(beh_trial[b][tr]))
print("n_windows = {} (WIN_SIZE={}, WIN_STEP={})".format(len(win_starts), WIN_SIZE, WIN_STEP))

mat = np.full((4, 4), np.nan); matp = np.full((4, 4), np.nan)
for bi, b in enumerate(BEH):
    for fi, f in enumerate(FORMS):
        h = np.asarray(HI[f], float); bv = np.asarray(BW[b], float)
        ok = np.isfinite(h) & np.isfinite(bv)
        if ok.sum() >= 4 and np.std(h[ok]) > 0 and np.std(bv[ok]) > 0:
            mat[bi, fi], matp[bi, fi] = spearmanr(h[ok], bv[ok])

vmax = max(0.2, np.nanmax(np.abs(mat)))
fig = plt.figure(figsize=(4.4, 3.4))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.35 / fw, 1.05 / fh, 2.4 / fw, 2.0 / fh])
im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
for bi in range(4):
    for fi in range(4):
        if np.isnan(mat[bi, fi]):
            continue
        p = matp[bi, fi]
        st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
        ax.text(fi, bi, '{:+.2f}\n{}'.format(mat[bi, fi], st), ha='center', va='center',
                fontsize=7, fontweight='bold' if st else 'normal')
ax.set_xticks(range(4)); ax.set_xticklabels(FORM_LABEL, fontsize=7)
ax.set_yticks(range(4)); ax.set_yticklabels(BEH_LABEL)
ax.set_title('MODEL, data-matched (trailing-20, 10-trial windows), 1 run')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04); cb.set_label('Spearman $\\rho$ (across windows)')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_4x4_windowed.' + ext), dpi=200, bbox_inches='tight')

print("\nMODEL (data-matched binning + trailing-20) corr(HI, outcome):")
print("  {:14s}".format('outcome') + "".join("{:>10s}".format(l.replace('$', '').replace('\\', '')) for l in ['r r', 'r dr', 'dr dr', 'dr r']))
for bi, b in enumerate(BEH):
    print("  {:14s}".format(BEH_LABEL[bi].split(' ')[0]) + "".join("{:+10.2f}".format(mat[bi, fi]) for fi in range(4)))
print("\nSaved to", OUT)
