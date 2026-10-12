#%% ============================================================================
# Fix Kyle's model (its true rule) and test DIFFERENT ANALYSIS eligibilities:
# reconstruct each candidate from recorded per-step activity, then recover RPE from
# the total dW.  Anchor: candidate 'true' should match the recorded eligibility
# (W_rec_elg_vals) and recover ~1.0; the others show what breaks recovery.
#
# True rule (networks.py 3factor):  E_ij(t) = (r_i(t)-ema_i(t))*phi'_i(t) * r_j(t-1)
#   accumulated as EMA trace (n_window_elig); baseline ema = EMA(n_window_baseline).
import os, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 7, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
from net_helpers import accumulate_decay
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes',
               'loss_steps', 'act_fn_p_pre_act_vals']
SEED = 0


def build_trace(E_step, loss_idx, n_win, mode='ema', n_spl=5):
    """Accumulate per-step eligibility E_step (T,n,n) and sample at loss steps.
    mode 'ema' = running EMA trace (n_win); 'wipe' = sum within loss window, reset."""
    n = E_step.shape[1]; cols = []
    tot = np.zeros((n, n)); g = 1 - 1.0 / n_win
    ls = set(loss_idx.tolist())
    for t in range(E_step.shape[0]):
        if mode == 'ema':
            tot = g * tot + (1 - g) * E_step[t]
        else:
            tot = tot + E_step[t] / n_spl
        if t in ls:
            cols.append(tot.ravel().copy())
            if mode != 'ema':
                tot = np.zeros((n, n))
    return np.array(cols).T                      # (n*n, n_loss)


params = kp.default_toy_params(seed=SEED, verbose=False)
task_params, train_params, net_params = params
kp.task_params, kp.train_params, kp.net_params = params
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
_po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
to = toa[0]
eta = train_params['eta']; n_be = int(train_params['n_window_baseline'])
n_el = int(train_params.get('n_window_elig', 40)); n_spl = int(train_params['n_steps_per_loss'])

rr = np.asarray(to['output'], float)                 # (T, n) rate
phi = np.asarray(to['act_fn_p_pre_act_vals'], float)  # (T, n) phi'
T, n = rr.shape
loss_idx = np.asarray(to['loss_steps'], int)
rpe = np.asarray(to['total_rpes'], float)
Wv = np.asarray(to['W_rec_vals'], float)
dW = (Wv[-1] - Wv[0]).ravel()
L = min(len(loss_idx), len(rpe), len(Wv))

# baselines
ema = np.zeros_like(rr); prev = np.zeros(n)
for t in range(T):
    prev = accumulate_decay(prev, rr[t], n_window=n_be); ema[t] = prev
dev_ema = rr - ema
fixed_bl = rr[:min(500, T)].mean(0)                  # fixed early baseline
dev_fix = rr - fixed_bl
pre_prev = np.vstack([np.zeros((1, n)), rr[:-1]])     # r_j(t-1)
pre_same = rr                                          # r_j(t)  (no lag)


def E_of(post_fac, pre_fac):
    # returns (T,n,n): post index i (rows), pre index j (cols)
    return post_fac[:, :, None] * pre_fac[:, None, :]


# ---- candidate per-step eligibilities (post_factor, pre_factor, trace-mode) ----
CANDS = {
    'true (dev*phi, prev, ema-trace)':  (dev_ema * phi, pre_prev, 'ema'),
    'no phi (dev, prev, ema-trace)':    (dev_ema,       pre_prev, 'ema'),
    'fixed baseline (dev_fix*phi)':     (dev_fix * phi, pre_prev, 'ema'),
    'no trace (dev*phi, per-loss wipe)':(dev_ema * phi, pre_prev, 'wipe'),
    'no trace no phi (dev, per-loss wipe)':(dev_ema,     pre_prev, 'wipe'),
    'WRONG raw (r_post*phi, prev)':     (rr * phi,      pre_prev, 'ema'),
    'WRONG both-dev (dev*phi, dev-pre)':(dev_ema * phi, dev_ema,  'ema'),
    'WRONG pre-dev (r_post*phi, dev-pre)':(rr * phi,    dev_ema,  'ema'),
    'WRONG raw no-tr no-phi':           (rr,            pre_prev, 'wipe'),
    'WRONG both-dev no-tr no-phi':      (dev_ema,       dev_ema,  'wipe'),
    'WRONG pre-dev no-tr no-phi':       (rr,            dev_ema,  'wipe'),
}

# anchor: recorded true eligibility
Erec = np.asarray(to['W_rec_elg_vals'], float)[:L].reshape(L, -1).T
c_rec, *_ = np.linalg.lstsq(Erec, dW, rcond=None)
print("anchor recorded-elig recovery corr:", round(spearmanr(c_rec / eta, rpe[:L])[0], 4))

print("\n{:38s} {:>10s} {:>10s}".format("analysis eligibility", "recov corr", "dW R^2"))
names, corrs, kinds = [], [], []
for name, (postf, pref, mode) in CANDS.items():
    A = build_trace(E_of(postf, pref), loss_idx[:L], n_el, mode, n_spl)   # (n*n, L)
    m = min(A.shape[1], L)
    A = A[:, :m]; y = dW; rp = rpe[:m]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    rhat = c / eta
    ok = np.isfinite(rhat) & np.isfinite(rp)
    corr = spearmanr(rhat[ok], rp[ok])[0] if ok.sum() > 3 else np.nan
    r2 = 1 - np.sum((y - A @ c) ** 2) / np.sum((y - y.mean()) ** 2)
    print("  {:36s} {:+10.3f} {:10.3f}".format(name, corr, r2))
    names.append(name); corrs.append(corr)
    kinds.append('true' if name.startswith('true') else
                 'wrong2' if name.startswith('WRONG') and 'no-tr' in name else
                 'wrong' if name.startswith('WRONG') else 'ablate')
print("\nrecov corr ~1 => that analysis eligibility recovers RPE; low => it does not.")

# ---- bar graph ----
short = [
    r"$r_{pre}\phi'\Delta r_{post}$" + "\n(true rule)",
    r"$r_{pre}\Delta r_{post}$" + "\n(no $\phi'$)",
    r"$r_{pre}\phi'\Delta r_{post}$" + "\n(fixed base.)",
    r"$r_{pre}\phi'\Delta r_{post}$" + "\n(no trace)",
    r"$r_{pre}\Delta r_{post}$" + "\n(no trace, no $\phi'$)",
    r"$r_{pre}\phi' r_{post}$" + "\n(wrong)",
    r"$\Delta r_{pre}\phi'\Delta r_{post}$" + "\n(wrong)",
    r"$\Delta r_{pre}\phi' r_{post}$" + "\n(wrong)",
    r"$r_{pre} r_{post}$" + "\n(wrong, no tr/$\phi'$)",
    r"$\Delta r_{pre}\Delta r_{post}$" + "\n(wrong, no tr/$\phi'$)",
    r"$\Delta r_{pre} r_{post}$" + "\n(wrong, no tr/$\phi'$)",
]
col = {'true': '#2c7d3f', 'ablate': '#7f7f7f', 'wrong': '#b0392b', 'wrong2': '#e0917f'}
fig = plt.figure(figsize=(13.0, 4.0)); fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.05 / fw, 1.45 / fh, 11.6 / fw, 2.05 / fh])
x = np.arange(len(corrs))
ax.bar(x, corrs, color=[col[k] for k in kinds], width=0.72)
for xi, v in zip(x, corrs):
    ax.text(xi, v + 0.025, '{:.2f}'.format(v), ha='center', fontsize=9)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel('corr( HI (MLR) , true RPE )', fontsize=9); ax.set_ylim(0, 1.1)
ax.set_title("Which analysis eligibility recovers the reward signal (RPE) from $\\Delta W$?\n"
             "green = true plasticity rule    |    gray = true form, one ingredient removed    |"
             "    red = a different coactivity form", fontsize=8.5)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'model_test_analysis_eligs.' + ext), dpi=200, bbox_inches='tight')
print("Saved bar graph to", OUT)
