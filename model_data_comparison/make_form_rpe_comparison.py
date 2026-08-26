#%% ============================================================================
# Close-to-original discrimination: which eligibility form's HI tracks behavior?
# Post-processes sliding_window_four_elig.npy (robust HI-vs-behavior analysis, 4
# forms x 4 behaviors). Per session, corr(HI(window), behavior) for each form
# (pre epoch), then PAIRED tests across sessions: does r_pre*dr_post (post-dev)
# beat the pre-deviated / both-deviated forms, for each behavioral variable?
import os
import numpy as np
from scipy.stats import spearmanr, wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'
DATA = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI\meta_analysis_results\sliding_window_four_elig.npy'

MODES = [('dot_prod_lag', '$r_{pre}r_{post}$'),
         ('dev2_lag', '$r_{pre}\\Delta r_{post}$'),
         ('dpost_dpre_lag', '$\\Delta r_{pre}\\Delta r_{post}$'),
         ('post_dpre_lag', '$\\Delta r_{pre}r_{post}$')]
# behavior field -> (label, sign)  [sign flips RT so higher=faster]
BEH = [('win_hit', 'Hit rate', 1),
       ('win_rpe', '$\\Delta$Speed (RPE)', 1),
       ('win_rt', 'Speed ($-$RT)', -1),
       ('win_hit_rpe', 'Hit $\\times$ RPE', 1)]
EI_PRE = 0
POST = 'dev2_lag'
mode_keys = [m for m, _ in MODES]

data = np.load(DATA, allow_pickle=True).item()


def _dedup(sessions):
    seen, out = set(), []
    for s in sessions:
        key = (s.get('mouse'), s.get('session'))
        if key not in seen:
            seen.add(key); out.append(s)
    return out


_before = len(data[mode_keys[0]])
data = {m: _dedup(data[m]) for m in mode_keys}
n_sess = len(data[mode_keys[0]])
if n_sess != _before:
    print("Deduplicated: {} -> {} unique sessions".format(_before, n_sess))


def star(v):
    v = v[np.isfinite(v)]
    if len(v) < 2 or not np.any(v != 0):
        return 'n.s.'
    p = wilcoxon(v)[1]
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


# R[beh][mode] = per-session corr(HI, behavior)
R = {}
for bkey, blab, bsign in BEH:
    R[bkey] = {m: np.full(n_sess, np.nan) for m in mode_keys}
    for m in mode_keys:
        for si, s in enumerate(data[m]):
            hi = s['hi_with_int'][:, EI_PRE]
            bv = bsign * np.asarray(s[bkey], float)
            ok = np.isfinite(hi) & np.isfinite(bv)
            if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                R[bkey][m][si] = spearmanr(hi[ok], bv[ok])[0]

print("\ncorr(HI, behavior) per form, mean (Wilcoxon vs 0), n={}:".format(n_sess))
hdr = "  {:16s}".format('behavior') + "".join("{:>14s}".format(m.replace('_lag', '')) for m in mode_keys)
print(hdr)
for bkey, blab, _ in BEH:
    row = "  {:16s}".format(blab.split(' (')[0].replace('$', '').replace('\\', ''))
    for m in mode_keys:
        v = R[bkey][m]; vv = v[np.isfinite(v)]
        row += "{:+7.3f}{:>3s} ".format(vv.mean(), star(v).replace('n.s.', ''))[:14]
    print(row)

print("\nPAIRED: post-dev (dev2) MINUS each other form, per behavior:")
for bkey, blab, _ in BEH:
    print("  {}:".format(blab.split(' (')[0]))
    for m, lab in MODES:
        if m == POST:
            continue
        d = R[bkey][POST] - R[bkey][m]
        ok = np.isfinite(d)
        print("     vs {:14s} d={:+.3f} +/- {:.3f}  {}".format(
            m.replace('_lag', ''), d[ok].mean(), d[ok].std() / np.sqrt(ok.sum()), star(d[ok])))

# figure: behavior x form matrix of mean corr, star = post-dev sig > that form (paired)
mat = np.full((len(BEH), len(MODES)), np.nan)
for bi, (bkey, _, _) in enumerate(BEH):
    for mi, (m, _) in enumerate(MODES):
        v = R[bkey][m]; mat[bi, mi] = np.nanmean(v)
vmax = max(0.05, np.nanmax(np.abs(mat)))
fig = plt.figure(figsize=(4.4, 3.4))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.5 / fw, 1.15 / fh, 2.3 / fw, 1.9 / fh])
im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
for bi, (bkey, _, _) in enumerate(BEH):
    for mi, (m, _) in enumerate(MODES):
        s0 = star(R[bkey][m])                       # vs 0
        ax.text(mi, bi, '{:+.2f}\n{}'.format(mat[bi, mi], '' if s0 == 'n.s.' else s0),
                ha='center', va='center', fontsize=7,
                fontweight='bold' if s0 not in ('n.s.', '') else 'normal',
                color='white' if abs(mat[bi, mi]) > 0.6 * vmax else 'k')
ax.set_xticks(range(len(MODES))); ax.set_xticklabels([l for _, l in MODES], fontsize=7)
ax.set_yticks(range(len(BEH))); ax.set_yticklabels([l for _, l, _ in BEH])
ax.set_title('DATA: corr(HI, behavior) by form (n={})\nstars = vs 0'.format(n_sess), fontsize=8)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04); cb.set_label('mean Spearman $\\rho$')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_form_behavior_comparison.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
