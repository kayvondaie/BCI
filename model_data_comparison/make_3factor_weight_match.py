#%% ============================================================================
# Do the fitted 3rd-factor (outcome) weights recover the TRUE eligibility's?
# From run_model_3factor_fit: HI(w) = sum_k a_k O_k(w); O = [const, true RPE,
# hits, dSpeed, hitxRPE]. The 3rd factor = the outcome weights (drop const).
# For each eligibility form: (a) its RPE weight, (b) cosine similarity of its
# 3rd-factor weight vector to the TRUE eligibility's (per seed).
# Post-processes model_3factor_fit.npy; no retraining.
import os
import numpy as np
from scipy.stats import wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

res = list(np.load(os.path.join(OUT, 'model_3factor_fit.npy'), allow_pickle=True))
OUTC = ['const', 'true', 'hits', 'speed_rpe', 'hits_rpe']   # weight vector order
THIRD = [1, 2, 3, 4]                                        # drop const -> 3rd factor
RPE_IDX = 1
ELIG = ['hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre']
LABEL = {'hebb': '$r_{pre}r_{post}$', 'dpost_pre': '$r_{pre}\\Delta r_{post}$\n(model rule)',
         'dpost_dpre': '$\\Delta r_{pre}\\Delta r_{post}$', 'post_dpre': '$\\Delta r_{pre}r_{post}$'}
MODEL_FORM = 'dpost_pre'


def wvecs(form):
    return np.array([r[(form, 'w_dc')] for r in res], float)   # (seeds, 5)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.dot(a, b) / (na * nb) if na > 0 and nb > 0 else np.nan


# per-seed cosine similarity of 3rd-factor weights to the TRUE eligibility's
truew = wvecs('true')
sim = {f: np.array([cos(wvecs(f)[s][THIRD], truew[s][THIRD]) for s in range(len(res))]) for f in ELIG}
rpe = {f: wvecs(f)[:, RPE_IDX] for f in ELIG}
rpe['true'] = truew[:, RPE_IDX]


def ms(v):
    v = v[np.isfinite(v)]
    return np.mean(v), np.std(v) / np.sqrt(len(v))


def star(v):
    v = v[np.isfinite(v)]
    p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


fig = plt.figure(figsize=(6.4, 3.0))
fw, fh = fig.get_size_inches()

# panel A: RPE weight per form (+ true as reference)
axA = fig.add_axes([0.95 / fw, 0.95 / fh, 2.5 / fw, 1.8 / fh])
formsA = ['true'] + ELIG
for i, f in enumerate(formsA):
    v = rpe[f]; m, se = ms(v)
    col = '#444' if f == 'true' else ('#1baf7a' if f == MODEL_FORM else '#9a9a95')
    axA.bar(i, m, 0.66, yerr=se, color=col, capsize=3)
    axA.text(i, m + (se + 0.01 if m >= 0 else -(se + 0.01)), star(v), ha='center',
             va='bottom' if m >= 0 else 'top', fontsize=7, color='#555')
axA.axhline(0, color='k', lw=0.8)
axA.set_xticks(range(len(formsA)))
axA.set_xticklabels(['true'] + [LABEL[f] for f in ELIG], fontsize=6.5)
axA.set_ylabel('fitted RPE weight')
axA.set_title('RPE (3rd-factor) weight', fontsize=8)

# panel B: cosine similarity of 3rd-factor weights to TRUE elig
axB = fig.add_axes([4.05 / fw, 0.95 / fh, 2.0 / fw, 1.8 / fh])
for i, f in enumerate(ELIG):
    v = sim[f]; m, se = ms(v)
    col = '#1baf7a' if f == MODEL_FORM else '#9a9a95'
    axB.bar(i, m, 0.66, yerr=se, color=col, capsize=3)
    axB.text(i, m + (se + 0.03 if m >= 0 else -(se + 0.03)), star(v), ha='center',
             va='bottom' if m >= 0 else 'top', fontsize=7, color='#555')
axB.axhline(0, color='k', lw=0.8)
axB.set_ylim(-1.05, 1.15)
axB.set_xticks(range(len(ELIG))); axB.set_xticklabels([LABEL[f] for f in ELIG], fontsize=6.5)
axB.set_ylabel('cosine similarity of 3rd-factor\nweights to TRUE elig')
axB.set_title('match to true eligibility', fontsize=8)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_3factor_weight_match.' + ext), dpi=200, bbox_inches='tight')

print("Fitted weight vectors [const, RPE, hits, dSpeed, hitxRPE] (mean over seeds):")
for f in ['true'] + ELIG:
    print("  {:12s} ".format(f) + "  ".join("{:+.2f}".format(x) for x in np.mean(wvecs(f), axis=0)))
print("\nRPE weight (mean +/- sem, sig):")
for f in ['true'] + ELIG:
    m, se = ms(rpe[f]); print("  {:12s} {:+.3f} +/- {:.3f}  {}".format(f, m, se, star(rpe[f])))
print("\nCosine similarity of 3rd-factor weights to TRUE elig:")
for f in ELIG:
    m, se = ms(sim[f]); print("  {:12s} {:+.3f} +/- {:.3f}  {}".format(f, m, se, star(sim[f])))
print("\nSaved to", OUT)
