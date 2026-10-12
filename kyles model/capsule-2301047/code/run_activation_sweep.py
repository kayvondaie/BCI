#%% ============================================================================
# Activation sweep: does moving off ReTanh break the model's form-insensitivity?
# ============================================================================
# Real controlled experiment (retrain under each activation, 1 seed for a first
# look). For each act_fn report raw / fluct / mean_drive HI-RPE, the cross-neuron
# mean-variance coupling, and whether it learned (final reward).
# Hypothesis: ReTanh's mean-variance coupling is why the model is insensitive;
# breaking it should make raw fail while fluct holds.
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

elig_types = ('true', 'hebb', 'dpost_pre')
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': elig_types, 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward',
               'total_rpes', 'loss_steps']


def run_activation(act_fn, seed=0, eta_scale=1.0):
    params = kp.default_toy_params(seed=seed, verbose=False)
    task_params, train_params, net_params = params
    net_params['act_fn_type'] = act_fn
    train_params['eta'] = train_params['eta'] * eta_scale
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})   # activity-only CN

    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]
    task_hist = task.hists[0]

    rpes_divs, ds_local, ds_full, _fs, elig_divs, _m = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params, return_elig_divs=True)
    rpe = rpes_divs['true']
    n_div = ds_full.shape[1]
    i_h, i_d = elig_types.index('hebb'), elig_types.index('dpost_pre')

    # mean_drive = hebb - dpost_pre (from elig_divs), HI vs full-session dW
    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    W = to['W_rec_vals']
    dWf = (W[loss_step_divs[-1]] - W[0]).flatten()
    md_full = np.full(n_div, np.nan)
    for d in range(n_div):
        md_elig = (elig_divs[i_h, d] - elig_divs[i_d, d]).flatten()
        if np.std(md_elig) > 0:
            md_full[d] = linregress(md_elig, dWf).slope

    def corr(h):
        ok = np.isfinite(h) & np.isfinite(rpe)
        return spearmanr(h[ok], rpe[ok])[0] if ok.sum() >= 3 else np.nan

    # cross-neuron mean-variance coupling
    stab = params[0]['n_steps_stabilize']
    act = to['output'][stab:]
    nmean, nstd = np.nanmean(act, 0), np.nanstd(act, 0)
    okn = np.isfinite(nmean) & np.isfinite(nstd) & (nstd > 0)
    coup = spearmanr(nmean[okn], nstd[okn])[0]

    final_reward = float(np.nanmean(to['reward'][-2000:]))
    return dict(true=corr(ds_full[0]), raw=corr(ds_full[i_h]),
                fluct=corr(ds_full[i_d]), mean_drive=corr(md_full),
                coupling=coup, n_div=n_div, reward=final_reward)


print("{:16s} {:>6s} {:>6s} {:>6s} {:>10s} {:>10s} {:>5s} {:>9s}".format(
    "act_fn(eta)", "true", "raw", "fluct", "mean_drive", "mean-var", "ndiv", "reward"))
for act_fn, eta_scale in (('ReTanh', 1.0), ('Tanh', 1.0),
                          ('linear', 1.0), ('linear', 0.2)):
    label = "{}(x{:.1f})".format(act_fn, eta_scale)
    try:
        r = run_activation(act_fn, seed=0, eta_scale=eta_scale)
        print("{:16s} {:+6.2f} {:+6.2f} {:+6.2f} {:+10.2f} {:+10.2f} {:5d} {:9.1e}".format(
            label, r['true'], r['raw'], r['fluct'], r['mean_drive'],
            r['coupling'], r['n_div'], r['reward']))
    except Exception as e:
        print("{:16s} FAILED: {}".format(label, e))
        import traceback
        traceback.print_exc()
