#%% ============================================================================
# Probe: mean hit rate in the model (fraction of rewarded trials).
# model 'hits' = ~isnan(rew_idxs) per trial. Report overall + early/late thirds.
import os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp

if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))


def hit_rate(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, _toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    hits = np.asarray(kp.get_rpe_estimates(task.hists[0])['hits'], float)
    n = len(hits)
    third = n // 3
    return np.mean(hits), np.mean(hits[:third]), np.mean(hits[-third:]), n


rows = []
for s in SEEDS:
    try:
        rows.append(hit_rate(s))
        print("seed {:2d}: hit rate = {:.3f}  (early {:.3f} -> late {:.3f}, {} trials)".format(
            s, rows[-1][0], rows[-1][1], rows[-1][2], rows[-1][3]))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))

r = np.array(rows)
print("\nMODEL mean hit rate (n={} seeds):".format(len(r)))
print("  overall = {:.3f} +/- {:.3f} sem".format(np.mean(r[:, 0]), np.std(r[:, 0]) / np.sqrt(len(r))))
print("  early third = {:.3f}    late third = {:.3f}".format(np.mean(r[:, 1]), np.mean(r[:, 2])))
print("  range across seeds = [{:.3f}, {:.3f}]".format(np.min(r[:, 0]), np.max(r[:, 0])))
