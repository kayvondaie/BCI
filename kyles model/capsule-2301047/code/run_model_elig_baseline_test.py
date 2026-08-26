#%% ============================================================================
# Is the model's eligibility-form degeneracy a BASELINE-CONVENTION artifact?
# Kyle's forms use a RUNNING (EMA) baseline -> Dr is locally zero-mean -> the
# mean-drive cross terms vanish -> deviation forms collapse (~0.996).
# The DATA uses a FIXED early-trial baseline -> Dr keeps session drift -> distinct.
# Recompute the 4 forms from the model activity under 3 baseline conventions and
# compare their 4x4 correlations across pairs.
import os, sys
import numpy as np
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

ELIG = ('hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre')
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

np.set_printoptions(precision=3, suppress=True)
n_post = elig_divs.shape[2]
offN = ~np.eye(n_post, dtype=bool)


def corr4(E):   # E: (4, n_pairs)
    return np.corrcoef(E)


def offdiag_stats(C):
    o = C[np.triu_indices(4, 1)]
    # deviation-form off-diagonals are pairs among indices 1,2,3
    dev = [C[1, 2], C[1, 3], C[2, 3]]
    return np.mean(dev), dev


# --- 1) Kyle's running-EMA forms (from elig_divs) ---
E_ema = np.stack([elig_divs[k].sum(axis=0)[offN] for k in range(4)], axis=0)
C_ema = corr4(E_ema)

# --- activity-based recomputation with fixed baselines ---
act = np.asarray(to['output'], float)                 # (T, N)
ok = np.all(np.isfinite(act), axis=1)
act = act[ok]
T = act.shape[0]


def forms_fixed(bl):
    D = act - bl                                      # (T, N)
    hebb = act.T @ act
    rd = D.T @ act        # r_pre  dr_post  (post deviated)
    dd = D.T @ D          # dr_pre dr_post
    dr = act.T @ D        # dr_pre r_post   (pre deviated)
    return np.stack([hebb[offN], rd[offN], dd[offN], dr[offN]], axis=0)


# --- 2) fixed SESSION-mean baseline ---
C_sess = corr4(forms_fixed(act.mean(0)))

# --- 3) fixed EARLY-portion baseline (data-matched: first ~23% of the session) ---
n_early = int(0.23 * T)
C_early = corr4(forms_fixed(act[:n_early].mean(0)))

for name, C in [('Kyle running-EMA (current)', C_ema),
                ('fixed SESSION-mean baseline', C_sess),
                ('fixed EARLY-trial baseline (data-matched)', C_early)]:
    m, dev = offdiag_stats(C)
    print("\n=== {} ===".format(name))
    print(C)
    print("  deviation-form off-diagonals (rd.dd, rd.dr, dd.dr) = {:.3f} {:.3f} {:.3f}  (mean {:.3f})".format(
        dev[0], dev[1], dev[2], m))
