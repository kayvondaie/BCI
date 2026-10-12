#%% ============================================================================
# Task-irrelevant drive: give the model activity the BCI outcome doesn't touch.
# ============================================================================
# Real controlled experiment. Patch SimpleRNN.forward to add a slow, external
# AR(1) current (task-IRRELEVANT: independent of reward) to the pre-activation,
# so neurons carry large ongoing activity the network must operate amid -- like
# real cortex, unlike this ~all-task RNN. Train through it, restore forward.
#
# Hypothesis: task-irrelevant activity inflates the baseline with RPE-blind
# structure, so mean_drive / raw should fall while fluct holds. 1 seed for hints.
import os, sys
import numpy as np
from scipy.stats import spearmanr, linregress

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
import networks

if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

elig_types = ('true', 'hebb', 'dpost_pre')
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': elig_types, 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward',
               'total_rpes', 'loss_steps']

_ORIG_FORWARD = networks.SimpleRNN.forward


def _install_drive(scale, seed):
    """Patch forward to add a fixed, heterogeneous, positive TONIC current per
    neuron (task-irrelevant) -- raises baselines without injecting noise."""
    if scale == 0.0:
        networks.SimpleRNN.forward = _ORIG_FORWARD
        return
    rng = np.random.default_rng(10_000 + seed)

    def forward(self, input_val, prev_activity, prev_activity_pre_act, net_params,
                perturbation_preact=None, perturbation=None):
        if not hasattr(self, '_tir'):
            self._tir = scale * np.abs(rng.standard_normal(self.n_neurons))
        d = self._tir
        pp = d if perturbation_preact is None else perturbation_preact + d
        return _ORIG_FORWARD(self, input_val, prev_activity, prev_activity_pre_act,
                             net_params, perturbation_preact=pp, perturbation=perturbation)

    networks.SimpleRNN.forward = forward


def run(scale, seed=0):
    _install_drive(scale, seed)
    try:
        params = kp.default_toy_params(seed=seed, verbose=False)
        kp.task_params, kp.train_params, kp.net_params = params
        kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
        _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    finally:
        networks.SimpleRNN.forward = _ORIG_FORWARD

    to = toa[0]
    task_hist = task.hists[0]
    rpes_divs, ds_local, ds_full, _fs, elig_divs, _m = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params, return_elig_divs=True)
    rpe = rpes_divs['true']
    n_div = ds_full.shape[1]
    i_h, i_d = elig_types.index('hebb'), elig_types.index('dpost_pre')

    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    W = to['W_rec_vals']
    dWf = (W[loss_step_divs[-1]] - W[0]).flatten()
    md_full = np.full(n_div, np.nan)
    for d in range(n_div):
        me = (elig_divs[i_h, d] - elig_divs[i_d, d]).flatten()
        if np.std(me) > 0:
            md_full[d] = linregress(me, dWf).slope

    def corr(h):
        ok = np.isfinite(h) & np.isfinite(rpe)
        return spearmanr(h[ok], rpe[ok])[0] if ok.sum() >= 3 else np.nan

    stab = params[0]['n_steps_stabilize']
    act = to['output'][stab:]
    act_std = float(np.nanmean(np.nanstd(act, 0)))
    act_mean = float(np.nanmean(act))
    nmean, nstd = np.nanmean(act, 0), np.nanstd(act, 0)
    okn = np.isfinite(nmean) & np.isfinite(nstd) & (nstd > 0)
    coup = spearmanr(nmean[okn], nstd[okn])[0]
    reward = float(np.nanmean(to['reward'][-2000:]))
    return dict(true=corr(ds_full[0]), raw=corr(ds_full[i_h]), fluct=corr(ds_full[i_d]),
                mean_drive=corr(md_full), coupling=coup, act_std=act_std,
                act_mean=act_mean, reward=reward, n_div=n_div)


print("{:>6s} {:>4s} {:>6s} {:>6s} {:>6s} {:>10s} {:>8s} {:>9s}".format(
    "drive", "ndiv", "true", "raw", "fluct", "mean_drive", "act_mean", "reward"))
for scale in (0.0, 0.05, 0.1, 0.2, 0.4):
    try:
        r = run(scale, seed=0)
        print("{:6.2f} {:4d} {:+6.2f} {:+6.2f} {:+6.2f} {:+10.2f} {:8.3f} {:9.1e}".format(
            scale, r['n_div'], r['true'], r['raw'], r['fluct'], r['mean_drive'],
            r['act_mean'], r['reward']))
    except Exception as e:
        print("{:6.1f} FAILED: {}".format(scale, e))
        import traceback
        traceback.print_exc()
