#%% ============================================================================
# Model eligibility-form 4x4 correlation, computed with the DATA's convention:
#   - per-trial activity (mean over each trial's steps)  [like the data's per-trial epoch]
#   - FIRST-20-trial fixed baseline (the "weird" one the data uses)
#   - products summed over all trials, then correlate the 4 forms across pairs.
# Also: session-mean baseline (should collapse to ~1.0) and the drift check that
# explains the difference (Sum_trials Dr != 0 for the first-20 baseline).
import os, sys
import numpy as np
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

LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
         '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
N_BASELINE = 20
SEED = 0
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ('hebb',), 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']

params = kp.default_toy_params(seed=SEED, verbose=False)
kp.task_params, kp.train_params, kp.net_params = params
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
_po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
to = toa[0]; task_hist = task.hists[0]

out = np.asarray(to['output'], float)                       # (n_steps, n_neurons)
ts = np.asarray(task_hist['trial_starts'], int)
bounds = list(ts) + [out.shape[0]]
A = []
for k in range(len(ts)):
    s, e = bounds[k], bounds[k + 1]
    if e > s:
        A.append(np.nanmean(out[s:e], axis=0))
A = np.array(A)                                             # (n_trials, n_neurons)
n_trials, n_neu = A.shape
off = ~np.eye(n_neu, dtype=bool)
print("n_trials = {}, n_neurons = {}".format(n_trials, n_neu))


def corr4(baseline):
    D = A - baseline                                       # (n_trials, n_neurons)
    hebb = A.T @ A          # r_pre  r_post
    rd = D.T @ A            # r_pre  dr_post  (post deviated)
    dd = D.T @ D            # dr_pre dr_post
    dr = A.T @ D            # dr_pre r_post   (pre deviated)
    E = np.stack([hebb[off], rd[off], dd[off], dr[off]], axis=0)
    return np.corrcoef(E)


bl_first20 = A[:min(N_BASELINE, n_trials)].mean(0)
bl_session = A.mean(0)
C_first20 = corr4(bl_first20)
C_session = corr4(bl_session)

np.set_printoptions(precision=3, suppress=True)
for name, C in [('FIRST-20-trial baseline (data-matched)', C_first20),
                ('session-mean baseline (sanity: ->1.0)', C_session)]:
    print("\n=== {} ===".format(name))
    print(C)
    print("  deviation off-diagonals (rd.dd, rd.dr, dd.dr) = {:.3f} {:.3f} {:.3f}  mean {:.3f}".format(
        C[1, 2], C[1, 3], C[2, 3], np.mean([C[1, 2], C[1, 3], C[2, 3]])))

# drift check: Sum_trials Dr, which must be ~0 for collapse
print("\nDrift check  mean_neuron |sum_trials Dr|:")
print("  first-20 baseline = {:.4f}".format(np.mean(np.abs((A - bl_first20).sum(0)))))
print("  session  baseline = {:.4f}  (0 by construction)".format(np.mean(np.abs((A - bl_session).sum(0)))))
print("  mean per-trial activity magnitude = {:.4f}".format(np.mean(np.abs(A))))

np.save(os.path.join(OUT, 'model_elig_corr_datamatched.npy'),
        {'first20': C_first20, 'session': C_session}, allow_pickle=True)

fig = plt.figure(figsize=(3.4, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.25 / fw, 1.0 / fh, 1.8 / fw, 1.8 / fh])
im = ax.imshow(C_first20, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
for i in range(4):
    for j in range(4):
        ax.text(j, i, '{:.2f}'.format(C_first20[i, j]), ha='center', va='center',
                fontsize=8, color='white' if abs(C_first20[i, j]) > 0.6 else 'k')
ax.set_xticks(range(4)); ax.set_xticklabels(LABEL, fontsize=7, rotation=40, ha='right')
ax.set_yticks(range(4)); ax.set_yticklabels(LABEL, fontsize=7)
ax.set_title('MODEL, data-matched convention\n(per-trial, first-20 baseline)', fontsize=8)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04); cb.set_label('Pearson r')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_elig_corr_datamatched.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
