#%% ============================================================================
# Decisive test: is the model's form-insensitivity an artifact of using its
# privileged TRUE RPE? Redo all forms + single-factor controls against BOTH the
# true internal RPE and the behavioral RPE (dSpeed = speed_rpe, what the data uses).
# Prediction: vs true RPE everything works (insensitive); vs dSpeed only the
# fluctuation survives (sensitive, data-like) and pre_only/raw/mean_drive fail.
import os, sys
import numpy as np
from scipy.stats import spearmanr, linregress, wilcoxon
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

ELIG = ('true', 'hebb', 'dpost_pre')
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true', 'speed_rpe')}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
FORMS = ('fluct', 'raw', 'mean_drive', 'pre_only', 'post_only', 'dev_only')
SEEDS = list(range(15))


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]
    task_hist = task.hists[0]
    rpes_divs, _dl, ds_full, _fs, elig_divs, _m = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params, return_elig_divs=True)
    rpe_true = np.asarray(rpes_divs['true'], float)
    rpe_beh = np.asarray(rpes_divs['speed_rpe'], float)
    n_div = ds_full.shape[1]
    i_h, i_d = ELIG.index('hebb'), ELIG.index('dpost_pre')

    n_spl = kp.train_params['n_steps_per_loss']
    loss_steps = to['loss_steps']
    out_act = to['output']
    n_neu = out_act.shape[1]
    loss_step_divs = kp.get_div_idxs(task_hist, loss_steps, hi_params)
    Wv = to['W_rec_vals']
    dWf = (Wv[loss_step_divs[-1]] - Wv[0]).flatten()
    actsum = np.full((n_div, n_neu), np.nan)
    for d in range(n_div):
        s0 = max(0, loss_steps[loss_step_divs[d]] - n_spl + 1)
        s1 = loss_steps[loss_step_divs[d + 1]] + 1
        actsum[d] = np.nansum(out_act[s0:s1], axis=0)
    sess_mean = np.nanmean(actsum, axis=0)

    hi = {'fluct': ds_full[i_d], 'raw': ds_full[i_h]}
    for k in ('mean_drive', 'pre_only', 'post_only', 'dev_only'):
        hi[k] = np.full(n_div, np.nan)
    for d in range(n_div):
        md = (elig_divs[i_h, d] - elig_divs[i_d, d]).flatten()
        if np.std(md) > 0:
            hi['mean_drive'][d] = linregress(md, dWf).slope
        facs = {'pre_only': np.broadcast_to(actsum[d][None, :], (n_neu, n_neu)),
                'post_only': np.broadcast_to(actsum[d][:, None], (n_neu, n_neu)),
                'dev_only': np.broadcast_to((actsum[d] - sess_mean)[:, None], (n_neu, n_neu))}
        for k, E in facs.items():
            e = np.array(E).flatten()
            if np.std(e) > 0:
                hi[k][d] = linregress(e, dWf).slope

    def c(h, rv):
        ok = np.isfinite(h) & np.isfinite(rv)
        return spearmanr(h[ok], rv[ok])[0] if (ok.sum() >= 3 and np.std(h[ok]) > 0 and np.std(rv[ok]) > 0) else np.nan
    out = {}
    for f in FORMS:
        out[(f, 'true')] = c(hi[f], rpe_true)
        out[(f, 'beh')] = c(hi[f], rpe_beh)
    return out


allres = []
for s in SEEDS:
    try:
        allres.append(run_seed(s))
        print("seed {} done".format(s))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_true_vs_beh_rpe.npy'), allres, allow_pickle=True)


def agg(f, rlab):
    v = np.array([r[(f, rlab)] for r in allres], float)
    v = v[np.isfinite(v)]
    p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return np.mean(v), np.std(v) / np.sqrt(len(v)), p


def star(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


print("\nMODEL corr(HI, RPE) by form -- true RPE vs behavioral dSpeed (n={}):".format(len(allres)))
print("  {:12s} {:>16s} {:>16s}".format("form", "vs true RPE", "vs dSpeed (beh)"))
for f in FORMS:
    mt, st, pt = agg(f, 'true')
    mb, sb, pb = agg(f, 'beh')
    print("  {:12s} {:+.2f} {:>6s}      {:+.2f} {:>6s}".format(f, mt, star(pt), mb, star(pb)))

# figure: grouped bars per form, true vs behavioral
fig = plt.figure(figsize=(5.2, 3.2))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([0.95 / fw, 0.95 / fh, 4.0 / fw, 1.95 / fh])
x = np.arange(len(FORMS))
for j, (rlab, dx, col, lab) in enumerate([('true', -0.2, '#888888', 'vs true RPE'),
                                          ('beh', 0.2, '#d62728', 'vs $\\Delta$Speed (behavioral)')]):
    ms = [agg(f, rlab) for f in FORMS]
    ax.bar(x + dx, [m[0] for m in ms], 0.36, yerr=[m[1] for m in ms], color=col,
           capsize=2, label=lab)
    for i, m in enumerate(ms):
        ax.text(x[i] + dx, 0.92, star(m[2]).replace('n.s.', ''), ha='center', va='top', fontsize=7)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylim(-0.9, 1.0)
ax.set_xticks(x); ax.set_xticklabels(FORMS, rotation=20, ha='right')
ax.set_ylabel('corr(HI, RPE)')
ax.set_title('MODEL: form-insensitivity is specific to the TRUE RPE')
ax.legend(loc='lower left', frameon=False)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_true_vs_beh_rpe.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
