#%% ============================================================================
# How similar are the eligibility forms themselves? 4x4 correlation matrix of the
# per-pair full-session eligibility across the four forms (model, one run).
#   corr( (r_pre r_post).flatten(), (r_pre dr_post).flatten() ), etc.
# If two forms are ~identical across pairs, no fit can separate them.
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

ELIG = ('hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre')     # user order
LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
         '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEED = 0

params = kp.default_toy_params(seed=SEED, verbose=False)
kp.task_params, kp.train_params, kp.net_params = params
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
_po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
to = toa[0]; task_hist = task.hists[0]
_rd, _ds, _dsf, _fs, elig_divs, _mlr = kp.compute_local_hebbian_indexes(
    to, task_hist, params, hi_params, return_elig_divs=True)

n_form, n_div, n_post, n_pre = elig_divs.shape
# full-session eligibility per pair per form, exclude self-pairs (diagonal)
offdiag = ~np.eye(n_post, n_pre, dtype=bool) if n_post == n_pre else np.ones((n_post, n_pre), bool)
E = np.stack([elig_divs[k].sum(axis=0)[offdiag] for k in range(n_form)], axis=0)  # (4, n_pairs)

C = np.corrcoef(E)                                            # 4x4
np.save(os.path.join(OUT, 'model_elig_correlation.npy'), C, allow_pickle=True)

fig = plt.figure(figsize=(3.4, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.25 / fw, 1.0 / fh, 1.8 / fw, 1.8 / fh])
im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
for i in range(4):
    for j in range(4):
        ax.text(j, i, '{:.2f}'.format(C[i, j]), ha='center', va='center', fontsize=8,
                color='white' if abs(C[i, j]) > 0.6 else 'k')
ax.set_xticks(range(4)); ax.set_xticklabels(LABEL, fontsize=7, rotation=40, ha='right')
ax.set_yticks(range(4)); ax.set_yticklabels(LABEL, fontsize=7)
ax.set_title('MODEL: eligibility-form similarity\n(corr across pairs, 1 run)', fontsize=8)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04); cb.set_label('Pearson r')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_elig_correlation.' + ext), dpi=200, bbox_inches='tight')

print("MODEL 4x4 eligibility correlation (order: r_pre r_post, r_pre dr_post, dr_pre dr_post, dr_pre r_post):")
print(np.round(C, 3))
print("\nSaved to", OUT)
