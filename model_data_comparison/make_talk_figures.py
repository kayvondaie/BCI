#%% ============================================================================
# Talk figures: model vs data eligibility-form sensitivity
# ============================================================================
# Loads the saved model (15-seed) and data (44-session) HI results and makes
# clean, shared-axis panels for the core contrast. Standalone (numpy/scipy/mpl).
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'

MODEL_NPY = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model\capsule-2301047\code\model_hi_multiseed.npy'
DATA_NPY = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI\meta_analysis_results\sliding_window_eligibility_decomposition.npy'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

model = np.load(MODEL_NPY, allow_pickle=True).item()
data = np.load(DATA_NPY, allow_pickle=True).item()

C_DATA, C_MODEL = '#2c7fb8', '#e6772e'


def model_vals(form):
    return np.array([v for v in model[form]['full'] if np.isfinite(v)])


def data_vals(mode):
    out = []
    for s in data[mode]:
        hi = s['hi_with_int'][:, 0]
        rpe = s['win_rpe']
        ok = np.isfinite(hi) & np.isfinite(rpe)
        if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(rpe[ok]) > 0:
            out.append(spearmanr(hi[ok], rpe[ok])[0])
    return np.array(out)


def stars(v):
    v = v[np.isfinite(v)]
    if len(v) < 2 or not np.any(v != 0):
        return ''
    p = wilcoxon(v)[1]
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


def strip(ax, x, v, color):
    v = v[np.isfinite(v)]
    ax.scatter(x + np.random.uniform(-0.08, 0.08, len(v)), v, s=9, color=color, alpha=0.35)
    m = np.mean(v)
    se = np.std(v) / np.sqrt(len(v))
    ax.errorbar([x], [m], yerr=[se], fmt='o', color=color, ms=6, capsize=3,
                mec='w', mew=0.6, zorder=5)
    ax.text(x, ax.get_ylim()[1] * 0.92, stars(v), ha='center', va='top', fontsize=8)
    return m


def ax_inches(fig, w_in, h_in, left=0.9, bottom=0.75):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, w_in / fw, h_in / fh])


# --- forms ---
FORMS3 = [('dpost_pre', 'full_trial', 'fluctuation\n$r_{pre}(r_{post}-\\overline{r})$'),
          ('hebb', 'full_trial_dot_prod', 'raw\n$r_{pre}\\,r_{post}$'),
          ('mean_drive', 'full_trial_mean_drive', 'mean-drive\n$r_{pre}\\,\\overline{r}$')]
SF = [('dpost_pre', 'full_trial', 'product\n(fluct)'),
      ('pre_only', 'full_trial_pre_only', 'pre only'),
      ('post_only', 'full_trial_post_only', 'post only'),
      ('dev_only', 'full_trial_dev_only', 'dev only')]

YL = (-0.9, 1.0)

# ============================================================================
# FIG 1 -- DATA: form sensitivity
# ============================================================================
fig = plt.figure(figsize=(3.4, 3.0))
ax = ax_inches(fig, 2.0, 1.9)
ax.set_ylim(*YL)
for i, (_, dk, lab) in enumerate(FORMS3):
    strip(ax, i, data_vals(dk), C_DATA)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xticks(range(len(FORMS3)))
ax.set_xticklabels([f[2] for f in FORMS3])
ax.set_ylabel('corr(HI, RPE)   [per session]')
ax.set_title('DATA (n=44): only the deviation form tracks RPE')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig1_data_forms.' + ext), dpi=200, bbox_inches='tight')

# ============================================================================
# FIG 2 -- MODEL: form insensitivity (same axes)
# ============================================================================
fig = plt.figure(figsize=(3.4, 3.0))
ax = ax_inches(fig, 2.0, 1.9)
ax.set_ylim(*YL)
for i, (mk, _, lab) in enumerate(FORMS3):
    strip(ax, i, model_vals(mk), C_MODEL)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xticks(range(len(FORMS3)))
ax.set_xticklabels([f[2] for f in FORMS3])
ax.set_ylabel('corr(HI, RPE)   [per seed]')
ax.set_title('MODEL vs TRUE RPE (the puzzle): every form works')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig2_model_forms.' + ext), dpi=200, bbox_inches='tight')

# ============================================================================
# FIG 3 -- Single-factor controls: data vs model
# ============================================================================
fig = plt.figure(figsize=(4.6, 3.2))
ax = ax_inches(fig, 3.2, 2.0)
ax.set_ylim(*YL)
x = np.arange(len(SF))
for i, (mk, dk, lab) in enumerate(SF):
    dm = np.mean(data_vals(dk)); dse = np.std(data_vals(dk)) / np.sqrt(len(data_vals(dk)))
    mm = np.mean(model_vals(mk)); mse = np.std(model_vals(mk)) / np.sqrt(len(model_vals(mk)))
    ax.bar(i - 0.2, dm, 0.36, yerr=dse, color=C_DATA, capsize=2,
           label='data' if i == 0 else None)
    ax.bar(i + 0.2, mm, 0.36, yerr=mse, color=C_MODEL, capsize=2,
           label='model' if i == 0 else None)
    ax.text(i - 0.2, YL[1] * 0.9, stars(data_vals(dk)), ha='center', va='top', fontsize=7)
    ax.text(i + 0.2, YL[1] * 0.9, stars(model_vals(mk)), ha='center', va='top', fontsize=7)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f[2] for f in SF])
ax.set_ylabel('corr(HI, RPE)')
ax.set_title('Single-factor controls: data needs the coincidence, model does not')
ax.legend(loc='lower left', frameon=False)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig3_single_factor.' + ext), dpi=200, bbox_inches='tight')

# ============================================================================
# FIG 0 -- HI x behavioral-variable x epoch matrix (deviation-based HI)
#          matches the manuscript / prior-talk figure so the lab can compare.
# ============================================================================
MODE_MAT = 'dev2_fulltrial_baseline'          # epoch-resolved, deviation-based HI
BEHAV = [('win_hit', 'Hit rate'), ('win_rpe', 'RPE ($\\Delta$Speed)'),
         ('win_rt', 'Reaction time'), ('win_hit_rpe', 'Hit $\\times$ RPE')]
EPOCHS = ['Pre', 'Go cue', 'Late', 'Reward']

mat_mean = np.full((len(BEHAV), len(EPOCHS)), np.nan)
mat_p = np.full((len(BEHAV), len(EPOCHS)), np.nan)
for bi, (bkey, _) in enumerate(BEHAV):
    for ei in range(len(EPOCHS)):
        vals = []
        for s in data[MODE_MAT]:
            hi = s['hi_with_int'][:, ei]
            bv = s[bkey]
            ok = np.isfinite(hi) & np.isfinite(bv)
            if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                vals.append(spearmanr(hi[ok], bv[ok])[0])
        vals = np.array(vals)
        if len(vals):
            mat_mean[bi, ei] = np.mean(vals)
            mat_p[bi, ei] = wilcoxon(vals)[1] if len(vals) >= 2 and np.any(vals != 0) else np.nan

vmax = max(0.12, np.nanmax(np.abs(mat_mean)))
fig = plt.figure(figsize=(3.8, 3.2))
ax = ax_inches(fig, 2.1, 1.9, left=1.2, bottom=0.9)
im = ax.imshow(mat_mean, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
for bi in range(len(BEHAV)):
    for ei in range(len(EPOCHS)):
        if np.isnan(mat_mean[bi, ei]):
            continue
        p = mat_p[bi, ei]
        st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
        ax.text(ei, bi, '{:+.2f}\n{}'.format(mat_mean[bi, ei], st),
                ha='center', va='center', fontsize=7,
                fontweight='bold' if st else 'normal')
ax.set_xticks(range(len(EPOCHS))); ax.set_xticklabels(EPOCHS, rotation=30, ha='right')
ax.set_yticks(range(len(BEHAV))); ax.set_yticklabels([b[1] for b in BEHAV])
ax.set_title('HI vs behavior (deviation-based HI, n=44)')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cb.set_label('mean Spearman $\\rho$')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig0_HI_behavior_matrix.' + ext), dpi=200, bbox_inches='tight')

# ============================================================================
# FIG 0b -- DATA HI x behavior x eligibility-form (raw vs fluct)
#           parallels the model matrix so they can sit side by side.
# ============================================================================
DBEH = [('win_hit', 'Hit rate'), ('win_rpe', 'RPE ($\\Delta$Speed)'),
        ('win_rt', 'Reaction time'), ('win_hit_rpe', 'Hit $\\times$ RPE')]
DFORMS = [('full_trial_dot_prod', 'raw\n$r_{pre}r_{post}$'),
          ('full_trial', 'fluct\n$r_{pre}(r_{post}{-}\\overline{r})$')]
dmat = np.full((len(DBEH), len(DFORMS)), np.nan)
dmatp = np.full((len(DBEH), len(DFORMS)), np.nan)
for bi, (bkey, _) in enumerate(DBEH):
    for fi, (mode, _) in enumerate(DFORMS):
        vv = []
        for s in data[mode]:
            hi = s['hi_with_int'][:, 0]; bv = s[bkey]
            ok = np.isfinite(hi) & np.isfinite(bv)
            if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                vv.append(spearmanr(hi[ok], bv[ok])[0])
        vv = np.array(vv)
        if len(vv):
            dmat[bi, fi] = np.mean(vv)
            dmatp[bi, fi] = wilcoxon(vv)[1] if len(vv) >= 2 and np.any(vv != 0) else np.nan
vmax_d = max(0.12, np.nanmax(np.abs(dmat)))
fig = plt.figure(figsize=(3.4, 3.2))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.4 / fw, 1.0 / fh, 1.3 / fw, 2.0 / fh])
im = ax.imshow(dmat, cmap='coolwarm', vmin=-vmax_d, vmax=vmax_d, aspect='auto')
for bi in range(len(DBEH)):
    for fi in range(len(DFORMS)):
        if np.isnan(dmat[bi, fi]):
            continue
        p = dmatp[bi, fi]
        st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
        ax.text(fi, bi, '{:+.2f}\n{}'.format(dmat[bi, fi], st), ha='center', va='center',
                fontsize=7, fontweight='bold' if st else 'normal')
ax.set_xticks(range(len(DFORMS))); ax.set_xticklabels([f[1] for f in DFORMS])
ax.set_yticks(range(len(DBEH))); ax.set_yticklabels([b[1] for b in DBEH])
ax.set_title('DATA: HI vs behavior, by eligibility form (n=44)')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cb.set_label('mean Spearman $\\rho$')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_data_behavior_matrix.' + ext), dpi=200, bbox_inches='tight')

# ============================================================================
# FIG 4 (ACT 3) -- data vs model against the BEHAVIORAL RPE (dSpeed).
#                  The cell-for-cell match: only the fluctuation form survives.
# ============================================================================
mbeh = list(np.load(os.path.join(OUT, 'model_true_vs_beh_rpe.npy'), allow_pickle=True))
ACT3 = [('fluct', 'full_trial', 'fluct\n$r_{pre}(r_{post}{-}\\overline{r})$'),
        ('raw', 'full_trial_dot_prod', 'raw\n$r_{pre}r_{post}$'),
        ('mean_drive', 'full_trial_mean_drive', 'mean-drive'),
        ('pre_only', 'full_trial_pre_only', 'pre only')]


def model_beh(form):
    return np.array([r[(form, 'beh')] for r in mbeh if np.isfinite(r[(form, 'beh')])])


fig = plt.figure(figsize=(4.8, 3.2))
ax = ax_inches(fig, 3.4, 2.0)
ax.set_ylim(-0.3, 0.55)
xx = np.arange(len(ACT3))
for i, (form, mode, lab) in enumerate(ACT3):
    dv = data_vals(mode)          # corr(HI, win_rpe) per session; win_rpe = dSpeed
    mv = model_beh(form)
    ax.bar(xx[i] - 0.2, np.mean(dv), 0.36, yerr=np.std(dv) / np.sqrt(len(dv)),
           color=C_DATA, capsize=2, label='data (n=44)' if i == 0 else None)
    ax.bar(xx[i] + 0.2, np.mean(mv), 0.36, yerr=np.std(mv) / np.sqrt(len(mv)),
           color=C_MODEL, capsize=2, label='model (n=15)' if i == 0 else None)
    ax.text(xx[i] - 0.2, 0.5, stars(dv), ha='center', va='top', fontsize=7)
    ax.text(xx[i] + 0.2, 0.5, stars(mv), ha='center', va='top', fontsize=7)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xticks(xx); ax.set_xticklabels([a[2] for a in ACT3])
ax.set_ylabel('corr(HI, $\\Delta$Speed)')
ax.set_title('Behavioral RPE: data and model agree — only fluctuation survives')
ax.legend(loc='lower right', frameon=False)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig4_behavioral_match.' + ext), dpi=200, bbox_inches='tight')

# ---- console summary ----
print("DATA (per-session mean corr(HI,RPE)):")
for _, dk, lab in FORMS3 + [s for s in SF if s not in FORMS3]:
    v = data_vals(dk)
    print("  {:22s} {:+.3f}  {}".format(dk, np.mean(v), stars(v)))
print("\nMODEL (per-seed mean corr(HI,RPE)):")
for mk, _, lab in FORMS3 + [s for s in SF if s not in FORMS3]:
    v = model_vals(mk)
    print("  {:22s} {:+.3f}  {}".format(mk, np.mean(v), stars(v)))
print("\nFigures written to", OUT)
plt.show()
