#%% ============================================================================
# figure_cn_learning.py
#
# Clean rebuild of the BCI gain-change figure (manuscript fig:behavior),
# centered on the robust late-trial CN learning signal.
#
# Loads two pre-computed pickles (NO network access):
#   1. session_metrics_compiled.pkl   (from session_metrics.py CELL 1/2)
#        -> F_cn_list, rt_mat, switches_list, bci_thresholds_list,
#           cn_peak_mat, dt_si_list, session_keys, ...
#   2. all_epoch_stats.pkl            (from threshold_analysis2.py CELL 5)
#        -> all_epoch_stats, all_session_trials  (behavioral panels j/k)
#
# Panels (each its own cell, added incrementally):
#   N1  pre-reward AUC rises over trials
#   N2  switch-aligned pre-reward AUC
#   N3  CN-vs-population specificity (slope variant)
#   N4  early-vs-peak reward-aligned transient
#   B1  RT improvement vs expected dRT          (panel j)
#   B2  actual vs expected hit rate, headroom    (panel k)
#
# Style: all fonts size 8, axes sized explicitly in inches, png+svg to PANEL_DIR.
# ============================================================================

#%% ============================================================================
# CELL 0: Setup + load pickles
# ============================================================================
import os, sys, pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon

mpl.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'font.family': 'sans-serif', 'svg.fonttype': 'none',
})

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, 'meta_analysis_results')
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')

# Time convention: F[:, :, i] starts PRE_TRIAL_S before the trial-start go cue.
PRE_TRIAL_S = 2.0

# --- Load session metrics pickle ---
_sm_path = os.path.join(RESULTS_DIR, 'session_metrics_compiled.pkl')
if not os.path.exists(_sm_path):
    raise FileNotFoundError(
        f"{_sm_path} not found. Run session_metrics.py CELL 1 + CELL 2 first.")
with open(_sm_path, 'rb') as fh:
    _sm = pickle.load(fh)
globals().update(_sm)
print(f"Loaded session metrics: {len(session_keys)} sessions")

# --- Load epoch stats pickle (behavioral panels) ---
_es_path = os.path.join(RESULTS_DIR, 'all_epoch_stats.pkl')
if os.path.exists(_es_path):
    with open(_es_path, 'rb') as fh:
        _es = pickle.load(fh)
    all_epoch_stats = _es['all_epoch_stats']
    all_session_trials = _es['all_session_trials']
    print(f"Loaded epoch stats: {len(all_epoch_stats)} epoch records")
else:
    all_epoch_stats = None
    print(f"WARNING: {_es_path} not found — behavioral panels (B1/B2) will skip."
          f"\n  Run threshold_analysis2.py CELL 0-5 to generate it.")

#%% ============================================================================
# CELL 1: Recompute derived quantities (fast, in-memory)
# ============================================================================
# Single principled pre-reward AUC used by N1 & N2:
#   per hit trial, mean CN F over the AUC_WIN_S seconds before reward,
#   minus that trial's own baseline (first BASELINE_S of a pre-reward window).
AUC_WIN_S  = 1.0    # integrate this many seconds before reward
BASELINE_S = 0.5    # per-trial baseline window length
SMOOTH_WIN = 10     # trial smoothing for the peak-trial detection

n_sess = len(session_keys)

# --- Per-session per-trial pre-reward AUC (NaN for miss / out-of-bounds) ---
auc_per_session = []      # list of (n_trials,) arrays
for i in range(n_sess):
    F_cn = F_cn_list[i]
    n_fr_s, n_tr = F_cn.shape
    dt_i = dt_si_list[i]
    rt_i = rt_mat[:n_tr, i]
    pre_trial_f = int(PRE_TRIAL_S / dt_i)
    win_n = int(AUC_WIN_S / dt_i)
    bl_n = max(1, int(BASELINE_S / dt_i))
    auc = np.full(n_tr, np.nan)
    for t in range(n_tr):
        if not np.isfinite(rt_i[t]):
            continue
        rf = int(rt_i[t] / dt_i) + pre_trial_f
        lo, hi = rf - win_n, rf
        if lo < 0 or hi > n_fr_s:
            continue
        # per-trial baseline = first BASELINE_S of the pre-reward window
        bl = np.nanmean(F_cn[lo:lo + bl_n, t])
        auc[t] = np.nanmean(F_cn[lo:hi, t]) - bl
    auc_per_session.append(auc)

# --- peak_trial: trial of max smoothed session-z'd pre-reward AUC ---
def _smooth_nan(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    valid = np.isfinite(x).astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)
    num = np.convolve(x0, k, mode='same')
    den = np.convolve(valid, k, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den > 0, num / den, np.nan)
    return out

peak_trial = np.full(n_sess, -1, dtype=int)
for i in range(n_sess):
    a = auc_per_session[i].astype(float)
    if np.nanstd(a) > 1e-9:
        az = (a - np.nanmean(a)) / np.nanstd(a)
    else:
        az = a
    asm = _smooth_nan(az, SMOOTH_WIN)
    if np.any(np.isfinite(asm)):
        peak_trial[i] = int(np.nanargmax(asm))

print(f"Recomputed: auc_per_session ({n_sess}), peak_trial")
print(f"  peak_trial: median = {int(np.median(peak_trial[peak_trial>=0]))}, "
      f"IQR = [{int(np.percentile(peak_trial[peak_trial>=0],25))}, "
      f"{int(np.percentile(peak_trial[peak_trial>=0],75))}]")
print(f"  AUC_WIN_S={AUC_WIN_S}, BASELINE_S={BASELINE_S}, SMOOTH_WIN={SMOOTH_WIN}")
print("\nScaffold ready. Panels will be added one cell at a time.")

#%% ============================================================================
# CELL B: Behavioral dose-response — RT (panel j) and HR (panel k, reframed)
# ============================================================================
# Both panels: x = size of the predicted perturbation (no-adaptation),
#              y = how much the animal beat the prediction.
# Restricted to threshold-INCREASE transitions (upper went up).
#   B1 (RT):  x = expected dRT  = expected_rt - prev_actual_rt
#             y = RT improvement = expected_rt - actual_rt   (positive = beat)
#   B2 (HR):  x = expected HR drop = prev_actual_hr - expected_hr_correct
#             y = HR improvement   = actual_hr - expected_hr_correct
# Origin = "step predicted no change", so no ceiling artifact.
# ============================================================================
if all_epoch_stats is None:
    print("CELL B skipped: all_epoch_stats not loaded.")
else:
    # Group by session, ordered by epoch
    _by_sess = {}
    for r in all_epoch_stats:
        _by_sess.setdefault((r['mouse'], r['session']), []).append(r)
    for k in _by_sess:
        _by_sess[k].sort(key=lambda r: r['epoch'])

    exp_drt, rt_imp = [], []
    exp_hr_drop, hr_imp = [], []
    for key, recs in _by_sess.items():
        for j in range(1, len(recs)):
            cur, prev = recs[j], recs[j - 1]
            if cur['upper'] <= prev['upper']:        # threshold increases only
                continue
            # RT panel
            e_rt = cur.get('expected_rt', np.nan)
            a_rt = cur.get('actual_rt', np.nan)
            p_rt = prev.get('actual_rt', np.nan)
            if np.all(np.isfinite([e_rt, a_rt, p_rt])):
                exp_drt.append(e_rt - p_rt)
                rt_imp.append(e_rt - a_rt)
            # HR panel
            e_hr = cur.get('expected_hr_correct', np.nan)
            a_hr = cur.get('actual_hr', np.nan)
            p_hr = prev.get('actual_hr', np.nan)
            if np.all(np.isfinite([e_hr, a_hr, p_hr])):
                exp_hr_drop.append(p_hr - e_hr)
                hr_imp.append(a_hr - e_hr)

    exp_drt = np.array(exp_drt); rt_imp = np.array(rt_imp)
    exp_hr_drop = np.array(exp_hr_drop); hr_imp = np.array(hr_imp)

    # No magnitude filtering — show all threshold-increase transitions.
    # The full-range correlation is the honest dose-response; the
    # near-origin points are real (small perturbation -> small response).
    fig_w, fig_h = 5.0, 2.3
    axw, axh = 1.5, 1.5
    figB = plt.figure(figsize=(fig_w, fig_h))

    def _dose_panel(left_frac, x, y, xlabel, ylabel):
        ax = figB.add_axes([left_frac, 0.22, axw / fig_w, axh / fig_h])
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=10, c='k', alpha=0.45, edgecolors='none')
        if ok.sum() >= 5:
            r, p = pearsonr(x[ok], y[ok])
            m, b = np.polyfit(x[ok], y[ok], 1)
            xf = np.array([x[ok].min(), x[ok].max()])
            ax.plot(xf, m * xf + b, color='crimson', linewidth=1.0)
            ax.set_title(f'r={r:.2f}, p={p:.1e}, n={int(ok.sum())}', fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    _dose_panel(0.10, exp_drt, rt_imp,
                'Expected ΔRT (s)', 'RT improvement (s)')
    _dose_panel(0.60, exp_hr_drop, hr_imp,
                'Expected HR drop', 'HR improvement')

    fnameB = 'behavioral_dose_response'
    figB.savefig(os.path.join(PANEL_DIR, f'{fnameB}.png'), dpi=300)
    figB.savefig(os.path.join(PANEL_DIR, f'{fnameB}.svg'))
    plt.show()
    print(f"Saved {fnameB}")
