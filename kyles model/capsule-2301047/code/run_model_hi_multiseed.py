#%% ============================================================================
# CELL 1: Imports and paths
# ============================================================================
"""
Multi-seed aggregation of the model HI-vs-RPE result, to compare against Kyle's
capsule figure (which aggregates ~15 seeds). Single-seed full-ΔW is pure noise;
the aggregate is the meaningful quantity.

For each seed: train one session, then for each eligibility form compute
corr(HI(division), true RPE) two ways --
   full  = HI(division) fit to whole-session ΔW   (data-matched target)
   local = HI(division) fit to within-division ΔW (model-only target)

Reuses kyle_pipeline.py (run build_kyle_pipeline.py first). Uses the same
activity-only CN-selection deviation as run_model_hi_analysis.py.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon, linregress

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import bci_analysis
import kyle_pipeline as kp

# Kyle's env ran numpy 2.0 (his code uses np.astype, a 2.0 module-level fn).
# We're on numpy<2 for dependency stability; shim it. np.astype(x, dtype) ==
# np.asarray(x).astype(dtype).
if not hasattr(np, 'astype'):
    def _np_astype(x, dtype, copy=True, device=None):
        return np.asarray(x).astype(dtype, copy=copy)
    np.astype = _np_astype

RESULTS_NPY = os.path.join(CODE_DIR, 'model_hi_multiseed.npy')

# Shared folder for model+data comparison figures (both scripts write here).
COMPARE_DIR = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'
os.makedirs(COMPARE_DIR, exist_ok=True)
print("Imports OK")

#%% ============================================================================
# CELL 2: Sweep seeds
# ============================================================================
SEEDS = list(range(15))
elig_types = ('true', 'hebb', 'dpost_pre')   # forms compute_local_hebbian_indexes knows
# + mean-drive (from elig_divs) + single-factor controls (no coactivity product)
forms = elig_types + ('mean_drive', 'pre_only', 'post_only', 'dev_only')
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward',
               'total_rpes', 'loss_steps', 'act_fn_p_pre_act_vals', 'session_summary']

# Match Kyle's bci_hebbian_index scan: divide the session BY TRIALS (5/div),
# not by loss-step index bins. Use his notebook compute_local_hebbian_indexes
# (kp.*), which takes task_hist + hi_params and returns rpes_divs as a dict.
hi_params = {
    'div_mode': 'trials',
    'n_trials_per_div': 5,
    'n_divisions': 20,          # only used if div_mode == 'idxs'
    'elig_types': elig_types,
    'rpe_types': ('true',),
}


def run_one_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    # Same flagged deviation as the single-seed script: activity-only CN.
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(
        params, output_vars=output_vars, verbose=False)
    to = toa[0]
    task_hist = task.hists[0]
    # return_elig_divs=True so we can build the mean-drive eligibility ourselves
    rpes_divs, ds_local, ds_full, _fs, elig_divs, _m = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params, return_elig_divs=True)
    rpe = rpes_divs['true']
    n_div = ds_local.shape[1]

    # --- mean-drive HI ---------------------------------------------------
    # mean_drive_elig = hebb_elig - dpost_pre_elig (same identity the data
    # script uses: raw_cc - dev2_cc). Regress it on within-division and
    # whole-session ΔW to get its HI(division), matching how the function does
    # it for the other forms.
    i_h, i_d = elig_types.index('hebb'), elig_types.index('dpost_pre')
    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    W = to['W_rec_vals']
    dW_full = (W[loss_step_divs[-1]] - W[0]).flatten()
    md_local = np.full(n_div, np.nan)
    md_full = np.full(n_div, np.nan)
    for d in range(n_div):
        md_elig = (elig_divs[i_h, d] - elig_divs[i_d, d]).flatten()
        if np.std(md_elig) == 0:
            continue
        dW_div = (W[loss_step_divs[d + 1]] - W[loss_step_divs[d]]).flatten()
        md_local[d] = linregress(md_elig, dW_div).slope
        md_full[d] = linregress(md_elig, dW_full).slope

    hi_full = {'true': ds_full[0], 'hebb': ds_full[i_h],
               'dpost_pre': ds_full[i_d], 'mean_drive': md_full}
    hi_local = {'true': ds_local[0], 'hebb': ds_local[i_h],
                'dpost_pre': ds_local[i_d], 'mean_drive': md_local}

    # --- SINGLE-FACTOR CONTROLS: r_pre only, r_post only, deviation only ------
    # Not a coactivity product. If these track RPE too, the model's HI-RPE
    # correlation never needed the Hebbian product -> can't test the rule.
    for k in ('pre_only', 'post_only', 'dev_only'):
        hi_full[k] = np.full(n_div, np.nan)
        hi_local[k] = np.full(n_div, np.nan)
    try:
        n_spl = params[1]['n_steps_per_loss']
        loss_steps = to['loss_steps']
        out_act = to['output']
        n_neu = out_act.shape[1]
        Wv = to['W_rec_vals']
        dWf = (Wv[loss_step_divs[-1]] - Wv[0]).flatten()
        actsum = np.full((n_div, n_neu), np.nan)
        for d in range(n_div):
            s0 = max(0, loss_steps[loss_step_divs[d]] - n_spl + 1)
            s1 = loss_steps[loss_step_divs[d + 1]] + 1
            actsum[d] = np.nansum(out_act[s0:s1], axis=0)
        sess_mean = np.nanmean(actsum, axis=0)
        for d in range(n_div):
            dWd = (Wv[loss_step_divs[d + 1]] - Wv[loss_step_divs[d]]).flatten()
            facs = {
                'pre_only': np.broadcast_to(actsum[d][None, :], (n_neu, n_neu)),
                'post_only': np.broadcast_to(actsum[d][:, None], (n_neu, n_neu)),
                'dev_only': np.broadcast_to((actsum[d] - sess_mean)[:, None], (n_neu, n_neu)),
            }
            for k, E in facs.items():
                e = np.array(E).flatten()
                if np.std(e) > 0:
                    hi_full[k][d] = linregress(e, dWf).slope
                    hi_local[k][d] = linregress(e, dWd).slope
    except Exception as _e:
        print("  (single-factor probe failed: {})".format(_e))

    out = {}
    for et in forms:
        okf = np.isfinite(hi_full[et]) & np.isfinite(rpe)
        okl = np.isfinite(hi_local[et]) & np.isfinite(rpe)
        rf = spearmanr(hi_full[et][okf], rpe[okf])[0] if okf.sum() >= 3 else np.nan
        rl = spearmanr(hi_local[et][okl], rpe[okl])[0] if okl.sum() >= 3 else np.nan
        out[et] = (rf, rl)

    # --- CRUX PROBE: does mean post activity per division track RPE? ----------
    # Hypothesis: in the model the population mean rate is RPE-contaminated
    # (activity is ~all outcome-driven); in the data it should be ~0.
    # Wrapped so a probe error can't discard this seed's HI results.
    meanact_r = np.nan
    try:
        n_spl = params[1]['n_steps_per_loss']
        loss_steps = to['loss_steps']
        out_act = to['output']
        mean_act = np.full(n_div, np.nan)
        for d in range(n_div):
            s0 = max(0, loss_steps[loss_step_divs[d]] - n_spl + 1)
            s1 = loss_steps[loss_step_divs[d + 1]] + 1
            mean_act[d] = np.nanmean(out_act[s0:s1])
        okm = np.isfinite(mean_act) & np.isfinite(rpe)
        if okm.sum() >= 3:
            meanact_r = spearmanr(mean_act[okm], rpe[okm])[0]
    except Exception as _e:
        print("  (mean-rate probe failed: {})".format(_e))

    # --- CROSS-NEURON COUPLING: does mean rate predict fluctuation size? ------
    # This is the derived crux: mean-drive HI ~ cross-neuron corr(<r_post_i>,
    # deviation_i). Predict positive in the model (active neurons = task neurons),
    # ~0 in the data (baseline rate decoupled from task modulation).
    coupling_r = np.nan
    try:
        stab = params[0]['n_steps_stabilize']
        act = to['output'][stab:]                 # (train steps, n_neurons)
        nmean = np.nanmean(act, axis=0)
        nstd = np.nanstd(act, axis=0)             # per-neuron fluctuation size
        okn = np.isfinite(nmean) & np.isfinite(nstd)
        if okn.sum() >= 5:
            coupling_r = spearmanr(nmean[okn], nstd[okn])[0]
    except Exception as _e:
        print("  (coupling probe failed: {})".format(_e))

    # --- REFINED (confound-free): corr(mean rate, RPE-modulation) across neurons
    # RPE-modulation_i = |corr(neuron i's per-division activity, RPE)|, not raw
    # std -> strips the mean-variance mechanics. Predict model high, data ~0.
    coupling_rpe = np.nan
    try:
        n_spl = params[1]['n_steps_per_loss']
        loss_steps = to['loss_steps']
        out_act = to['output']
        stab = params[0]['n_steps_stabilize']
        n_neu = out_act.shape[1]
        act_div = np.full((n_div, n_neu), np.nan)
        for d in range(n_div):
            s0 = max(0, loss_steps[loss_step_divs[d]] - n_spl + 1)
            s1 = loss_steps[loss_step_divs[d + 1]] + 1
            act_div[d] = np.nanmean(out_act[s0:s1], axis=0)
        rpe_mod = np.full(n_neu, np.nan)
        for i in range(n_neu):
            col = act_div[:, i]
            okc = np.isfinite(col) & np.isfinite(rpe)
            if okc.sum() >= 3 and np.std(col[okc]) > 0:
                rpe_mod[i] = abs(spearmanr(col[okc], rpe[okc])[0])
        nmean2 = np.nanmean(out_act[stab:], axis=0)
        okn2 = np.isfinite(nmean2) & np.isfinite(rpe_mod)
        if okn2.sum() >= 5 and np.std(nmean2[okn2]) > 0:
            coupling_rpe = spearmanr(nmean2[okn2], rpe_mod[okn2])[0]
    except Exception as _e:
        print("  (rpe-coupling probe failed: {})".format(_e))

    # --- DIMENSIONALITY: pre-post (neuron-neuron) correlation + participation
    # ratio. The mechanism for single factors proxying the product: low-D /
    # redundant activity. Predict model high pairwise corr, low PR/N.
    prepost_corr = np.nan
    pr = np.nan
    pr_frac = np.nan
    try:
        act = to['output'][params[0]['n_steps_stabilize']:]   # (steps, n_neu)
        C = np.corrcoef(act.T)
        prepost_corr = np.nanmean(C[~np.eye(C.shape[0], dtype=bool)])
        cov = np.cov(act.T)
        ev = np.linalg.eigvalsh(cov)
        ev = ev[ev > 1e-12]
        pr = float((ev.sum() ** 2) / (ev ** 2).sum())
        pr_frac = pr / act.shape[1]
    except Exception as _e:
        print("  (dim probe failed: {})".format(_e))

    probes = {'meanact': meanact_r, 'coupling': coupling_r,
              'coupling_rpe': coupling_rpe, 'prepost_corr': prepost_corr,
              'pr': pr, 'pr_frac': pr_frac}
    return out, probes

res = {et: {'full': [], 'local': []} for et in forms}
PROBE_KEYS = ('meanact', 'coupling', 'coupling_rpe', 'prepost_corr', 'pr', 'pr_frac')
for k in PROBE_KEYS:
    res[k] = []
for s in SEEDS:
    try:
        o, pd = run_one_seed(s)
        for et in forms:
            res[et]['full'].append(o[et][0])
            res[et]['local'].append(o[et][1])
        for k in PROBE_KEYS:
            res[k].append(pd[k])
        print("seed {:2d}:  ".format(s) + "   ".join(
            "{} full={:+.2f}".format(et, o[et][0]) for et in forms)
            + "   prepost_corr={:+.2f}  PR/N={:.3f}".format(pd['prepost_corr'], pd['pr_frac']))
    except Exception as e:
        print("seed {:2d} FAILED: {}".format(s, e))

np.save(RESULTS_NPY, res, allow_pickle=True)
print("\nSaved per-seed correlations to", RESULTS_NPY)

#%% ============================================================================
# CELL 3: Aggregate + compare
# ============================================================================
res = np.load(RESULTS_NPY, allow_pickle=True).item()

print("\nAggregate corr(HI, RPE) across seeds  (mean +/- sem, Wilcoxon vs 0)")
for tgt, label in [('local', 'within-division dW (model-only)'),
                   ('full', 'full-session dW (data-matched)')]:
    print("\n  target = {}".format(label))
    for et in forms:
        v = np.array(res[et][tgt], dtype=float)
        v = v[np.isfinite(v)]
        p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
        print("    {:10s}: mean={:+.3f}  sem={:.3f}  n={:2d}  p={:.2g}".format(
            et, np.mean(v), np.std(v) / np.sqrt(len(v)), len(v), p))

# CRUX PROBE aggregate: corr(mean post activity, RPE) across seeds
ma = np.array(res.get('meanact', []), dtype=float)
ma = ma[np.isfinite(ma)]
if len(ma) >= 2:
    pma = wilcoxon(ma)[1] if np.any(ma != 0) else np.nan
    print("\n  CRUX PROBE  corr(<r_post> per division, RPE):")
    print("    mean={:+.3f}  sem={:.3f}  n={:2d}  p={:.2g}".format(
        np.mean(ma), np.std(ma) / np.sqrt(len(ma)), len(ma), pma))

# Probe figure: distribution of per-seed corr(mean activity, RPE)
figp, axp = plt.subplots(figsize=(4.5, 4))
axp.axhline(0, color='0.6', lw=0.8)
axp.scatter(np.random.uniform(-0.06, 0.06, len(ma)), ma, s=22,
            color='tab:red', alpha=0.5)
axp.scatter([0], [np.mean(ma)], s=110, color='tab:red', edgecolor='w', zorder=5)
axp.set_xlim(-0.4, 0.4); axp.set_xticks([])
axp.set_ylabel('corr(mean post activity, RPE)')
axp.set_title('MODEL: does the mean rate track RPE?\nmean={:+.2f} (n={})'.format(
    np.mean(ma) if len(ma) else np.nan, len(ma)))
figp.tight_layout()
for d in (CODE_DIR, COMPARE_DIR):
    figp.savefig(os.path.join(d, 'fig_model_meanrate_vs_rpe.png'), dpi=150,
                 bbox_inches='tight')
plt.show()

# CROSS-NEURON COUPLING aggregate: corr(mean rate, fluct size) across neurons
cp = np.array(res.get('coupling', []), dtype=float)
cp = cp[np.isfinite(cp)]
if len(cp) >= 2:
    pcp = wilcoxon(cp)[1] if np.any(cp != 0) else np.nan
    print("\n  COUPLING  cross-neuron corr(mean rate, fluctuation size):")
    print("    mean={:+.3f}  sem={:.3f}  n={:2d}  p={:.2g}".format(
        np.mean(cp), np.std(cp) / np.sqrt(len(cp)), len(cp), pcp))

figc, axc = plt.subplots(figsize=(4.5, 4))
axc.axhline(0, color='0.6', lw=0.8)
axc.scatter(np.random.uniform(-0.06, 0.06, len(cp)), cp, s=22,
            color='tab:green', alpha=0.5)
axc.scatter([0], [np.mean(cp)], s=110, color='tab:green', edgecolor='w', zorder=5)
axc.set_xlim(-0.4, 0.4); axc.set_xticks([]); axc.set_ylim(-1.05, 1.05)
axc.set_ylabel('corr(mean rate, fluct size) across neurons')
axc.set_title('MODEL: is mean rate coupled to deviation?\nmean={:+.2f} (n={})'.format(
    np.mean(cp) if len(cp) else np.nan, len(cp)))
figc.tight_layout()
for d in (CODE_DIR, COMPARE_DIR):
    figc.savefig(os.path.join(d, 'fig_model_meanrate_vs_dev_coupling.png'), dpi=150,
                 bbox_inches='tight')
plt.show()

# REFINED COUPLING aggregate: corr(mean rate, RPE-modulation) across neurons
cr = np.array(res.get('coupling_rpe', []), dtype=float)
cr = cr[np.isfinite(cr)]
if len(cr) >= 2:
    pcr = wilcoxon(cr)[1] if np.any(cr != 0) else np.nan
    print("\n  REFINED COUPLING  cross-neuron corr(mean rate, RPE-modulation):")
    print("    mean={:+.3f}  sem={:.3f}  n={:2d}  p={:.2g}".format(
        np.mean(cr), np.std(cr) / np.sqrt(len(cr)), len(cr), pcr))

figr, axr = plt.subplots(figsize=(4.5, 4))
axr.axhline(0, color='0.6', lw=0.8)
axr.scatter(np.random.uniform(-0.06, 0.06, len(cr)), cr, s=22,
            color='tab:purple', alpha=0.5)
axr.scatter([0], [np.mean(cr)], s=110, color='tab:purple', edgecolor='w', zorder=5)
axr.set_xlim(-0.4, 0.4); axr.set_xticks([]); axr.set_ylim(-1.05, 1.05)
axr.set_ylabel('corr(mean rate, RPE-modulation) across neurons')
axr.set_title('MODEL: mean rate vs RPE-modulation\nmean={:+.2f} (n={})'.format(
    np.mean(cr) if len(cr) else np.nan, len(cr)))
figr.tight_layout()
for d in (CODE_DIR, COMPARE_DIR):
    figr.savefig(os.path.join(d, 'fig_model_meanrate_vs_rpemod_coupling.png'),
                 dpi=150, bbox_inches='tight')
plt.show()

# DIMENSIONALITY aggregate: pre-post correlation + participation ratio
ppc = np.array(res.get('prepost_corr', []), dtype=float); ppc = ppc[np.isfinite(ppc)]
prr = np.array(res.get('pr', []), dtype=float); prr = prr[np.isfinite(prr)]
prf = np.array(res.get('pr_frac', []), dtype=float); prf = prf[np.isfinite(prf)]
print("\n  DIMENSIONALITY (model):")
if len(ppc):
    print("    mean pairwise neuron corr = {:+.3f}  sem={:.3f}  n={}".format(
        np.mean(ppc), np.std(ppc) / np.sqrt(len(ppc)), len(ppc)))
if len(prr):
    print("    participation ratio       = {:.1f}  sem={:.1f}   PR/N = {:.3f}".format(
        np.mean(prr), np.std(prr) / np.sqrt(len(prr)), np.mean(prf)))

# Strip plot: per-seed points + mean, both targets (mirrors Kyle's middle-left)
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(forms))
for j, (tgt, dx, clr) in enumerate([('local', -0.12, 'k'), ('full', 0.12, 'tab:blue')]):
    for i, et in enumerate(forms):
        v = np.array(res[et][tgt], dtype=float); v = v[np.isfinite(v)]
        ax.scatter(np.full(len(v), x[i] + dx), v, s=18, color=clr, alpha=0.35)
        ax.scatter([x[i] + dx], [np.mean(v)], s=90, color=clr,
                   edgecolor='w', zorder=5,
                   label=('local ΔW' if i == 0 else None) if tgt == 'local'
                   else ('full ΔW' if i == 0 else None))
ax.axhline(0, color='0.6', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(forms)
ax.set_ylabel('corr(HI, RPE)')
ax.set_title('Model HI vs RPE across {} seeds'.format(len(SEEDS)))
ax.legend(loc='lower right')
plt.tight_layout()
for d in (CODE_DIR, COMPARE_DIR):
    plt.savefig(os.path.join(d, 'fig_model_hi_multiseed.png'), dpi=150,
                bbox_inches='tight')
plt.show()
print("Figures saved (incl. comparison dir:", COMPARE_DIR, ")")
