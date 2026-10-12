#%% ============================================================================
# Confirm the TRUE eligibility works: Kyle's model records its own eligibility
# trace (W_rec_elg_vals) and per-loss-step RPE (total_rpes). The update is
#   delta_W[k] = eta * rpe_k * elig_k   (networks.py loss_step).
# Two checks:
#   (1) PER-STEP identity (guaranteed): slope of delta_W[k] vs elig_k == eta*rpe_k,
#       so slope/eta recovers rpe_k with corr ~ 1.0.  (sanity that we read it right)
#   (2) TOTAL-dW joint recovery (the real test, like the data): from
#       dW_total = W[-1]-W[0] = eta*sum_k rpe_k*elig_k, stack elig_k as columns and
#       solve for rpe_k. Does the trace collinearity still let us recover RPE?
import os, sys
import numpy as np
from scipy.stats import spearmanr, linregress

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(3))


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    task_params, train_params, net_params = params
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]
    eta = train_params['eta']
    Wv = np.asarray(to['W_rec_vals'], float)           # (n_loss, n, n)
    Ev = np.asarray(to['W_rec_elg_vals'], float)        # (n_loss, n, n)  recorded eligibility trace
    rpe = np.asarray(to['total_rpes'], float)           # (n_loss,)  per-loss-step RPE (avg, clipped)
    L = min(len(Wv), len(Ev), len(rpe))
    Wv, Ev, rpe = Wv[:L], Ev[:L], rpe[:L]

    # (1) per-step identity: delta_W[k] = W[k]-W[k-1] should be eta*rpe_k*E_k
    dW_step = np.diff(Wv, axis=0)                        # (L-1, n, n) -> uses E[1:], rpe[1:]
    slope = np.full(L - 1, np.nan)
    for k in range(L - 1):
        e = Ev[k + 1].ravel(); d = dW_step[k].ravel()
        if np.std(e) > 0:
            slope[k] = linregress(e, d).slope           # == eta*rpe_{k+1} if identity holds
    rpe_step = slope / eta
    ok = np.isfinite(rpe_step) & np.isfinite(rpe[1:])
    r_perstep = spearmanr(rpe_step[ok], rpe[1:][ok])[0]
    # also the exact per-step ratio error
    med_relerr = np.nanmedian(np.abs(rpe_step[ok] - rpe[1:][ok]) / (np.abs(rpe[1:][ok]) + 1e-9))

    # (2) total-dW joint recovery from the true eligibility
    dW_tot = (Wv[-1] - Wv[0]).ravel()                   # = eta*sum_{k>=1} rpe_k*E_k
    A = Ev[1:].reshape(L - 1, -1).T                     # (n*n, L-1) columns = elig per step
    c, *_ = np.linalg.lstsq(A, dW_tot, rcond=None)
    rpe_hat = c / eta
    ok2 = np.isfinite(rpe_hat) & np.isfinite(rpe[1:])
    r_joint = spearmanr(rpe_hat[ok2], rpe[1:][ok2])[0] if ok2.sum() > 3 else np.nan
    # reconstruction quality of dW itself
    r2 = 1 - np.sum((dW_tot - A @ c) ** 2) / np.sum((dW_tot - dW_tot.mean()) ** 2)

    return r_perstep, med_relerr, r_joint, r2, L


print("seed | per-step corr | per-step rel.err | JOINT corr(rpe_hat,rpe) | dW R^2 | n_loss")
for s in SEEDS:
    try:
        rp, err, rj, r2, L = run_seed(s)
        print("  {}  |    {:+.3f}     |     {:.1e}     |        {:+.3f}          | {:.3f}  | {}".format(
            s, rp, err, rj, r2, L))
        sys.stdout.flush()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("seed {} FAILED: {}".format(s, e))
print("\nper-step corr ~1.0 confirms we read the true eligibility correctly (it's the update rule).")
print("JOINT corr is the real test: can the recorded (true) eligibility recover RPE from total dW?")
