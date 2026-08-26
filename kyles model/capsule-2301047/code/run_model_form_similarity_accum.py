#%% ============================================================================
# Is the ELIGIBILITY TRACE (accumulation) why the model's 4 coactivity forms are
# so similar to each other (~0.83) while the data's are not (~0.13)?
# Build the 4 forms' per-loss-step eligibility WITH the EMA trace vs WITHOUT it
# (fresh per loss step), same fixed baseline, and compare their 4x4 similarity.
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
mpl.rcParams.update({'font.size': 8, 'svg.fonttype': 'none'})
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
from net_helpers import accumulate_decay
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

output_vars = ['W_rec_vals', 'output', 'total_rpes', 'loss_steps']
SEED = 0
FORM_LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']


def build_trace(E_step, loss_idx, n_win, mode, n_spl=5):
    n = E_step.shape[1]; cols = []
    tot = np.zeros((n, n)); g = 1 - 1.0 / n_win
    ls = set(loss_idx.tolist())
    for t in range(E_step.shape[0]):
        tot = (g * tot + (1 - g) * E_step[t]) if mode == 'ema' else (tot + E_step[t] / n_spl)
        if t in ls:
            cols.append(tot.ravel().copy())
            if mode != 'ema':
                tot = np.zeros((n, n))
    return np.array(cols, dtype=np.float32).T          # (n*n, n_loss)


params = kp.default_toy_params(seed=SEED, verbose=False)
task_params, train_params, net_params = params
kp.task_params, kp.train_params, kp.net_params = params
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
_po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
to = toa[0]; task_hist = task.hists[0]
n_el = int(train_params.get('n_window_elig', 40)); n_spl = int(train_params['n_steps_per_loss'])
rr = np.asarray(to['output'], float)
T, n = rr.shape
loss_idx = np.asarray(to['loss_steps'], int)

# fixed (first-20-trial) baseline, per neuron
ts = np.asarray(task_hist['trial_starts'], int)
bl_end = ts[min(20, len(ts) - 1)] if len(ts) > 20 else T // 5
baseline = rr[:bl_end].mean(0)
dev = rr - baseline
pre_prev = np.vstack([np.zeros((1, n)), rr[:-1]])       # r_pre(t-1)
dev_prev = np.vstack([np.zeros((1, n)), dev[:-1]])      # dr_pre(t-1)


def E_of(post_fac, pre_fac):
    return post_fac[:, :, None] * pre_fac[:, None, :]


# 4 coactivity forms (post_factor, pre_factor), order matches the figure
FORMS = [(rr, pre_prev),      # r_pre r_post
         (dev, pre_prev),     # r_pre dr_post
         (dev, dev_prev),     # dr_pre dr_post
         (rr, dev_prev)]      # dr_pre r_post


def sim_matrix(mode, sub=8):
    # build each form's per-loss-step eligibility, subsample loss steps for memory
    cols = loss_idx[::sub]
    V = []
    for post_f, pre_f in FORMS:
        A = build_trace(E_of(post_f, pre_f), cols, n_el, mode, n_spl)   # (n*n, n_cols)
        V.append(A.ravel())
    V = np.array(V, dtype=np.float32)
    return np.corrcoef(V)


M_no = sim_matrix('wipe')      # no accumulation (fresh per loss step)
M_acc = sim_matrix('ema')      # WITH accumulation (EMA trace, n_window_elig)


def offdiag_mean(M):
    return (M.sum() - np.trace(M)) / (M.size - M.shape[0])


print("NO-accumulation form similarity (fresh per loss step):")
print(np.round(M_no, 2), "  off-diag mean =", round(offdiag_mean(M_no), 3))
print("\nWITH-accumulation form similarity (EMA trace, n_window_elig={}):".format(n_el))
print(np.round(M_acc, 2), "  off-diag mean =", round(offdiag_mean(M_acc), 3))

x = np.linspace(-1, 1, 256)
bwr = LinearSegmentedColormap.from_list('bwr', np.column_stack(
    [np.minimum(1, 1 + x), 1 - np.abs(x), np.minimum(1, 1 - x)]))
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, M, ttl in [(axes[0], M_no, 'NO trace (fresh per loss step)\noff-diag {:.2f}'.format(offdiag_mean(M_no))),
                   (axes[1], M_acc, 'WITH trace (EMA, n_elig={})\noff-diag {:.2f}'.format(n_el, offdiag_mean(M_acc)))]:
    im = ax.imshow(M, cmap=bwr, vmin=-1, vmax=1)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, '{:.2f}'.format(M[a, b]), ha='center', va='center',
                    fontsize=9, color='w' if abs(M[a, b]) > 0.6 else 'k')
    ax.set_xticks(range(4)); ax.set_xticklabels(FORM_LABEL, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(FORM_LABEL, fontsize=8)
    ax.set_title(ttl, fontsize=9)
fig.suptitle('Does the eligibility trace drive model form-similarity? (model, 1 run)', fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.7, label='Pearson r')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'model_form_similarity_accum.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
