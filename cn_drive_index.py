#%% ============================================================================
# cn_drive_index.py
#
# Counterfactual CN-drive index vs trial, with the matched behavioral
# actual-minus-expected hit-rate panel.  Ported from the Code Ocean capsule
# (cn_drive_trace.py) to this dataset.
#
# Drive index (the key, confound-free CN-adaptation metric):
#   d[i] = mean over [go cue, go + W s] of  max(roi_cn - lower1, 0)/(upper1 - lower1)
# evaluated against the FIXED epoch-1 thresholds (lower1, upper1) and a FIXED
# post-cue window.  Holding BOTH fixed isolates CN adaptation from the threshold
# staircase (using the per-trial escalating threshold or the to-crossing window
# would be ~circular, since the supra-threshold integral to crossing is pinned to
# the target distance).  roi_cn is the live Bonsai readout in THRESHOLD units
# (roi_csv[:, cn_csv_index+2]) -> correct scale, no dF/F normalization needed.
#
# Loads session_metrics_compiled.pkl (cn_fluor_list = per-trial roi_csv CN), so
# no network reload.  Run CELL 0 -> CELL 1 -> CELL 2.
# ============================================================================

#%% ============================================================================
# CELL 0: setup
# ============================================================================
import os, sys, pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'svg.fonttype': 'none',
})
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')
os.makedirs(PANEL_DIR, exist_ok=True)
PKL = os.path.join(_THIS_DIR, 'meta_analysis_results', 'session_metrics_compiled.pkl')

# ---- config -----------------------------------------------------------------
WINDOW      = 'precross'  # 'precross' (last LATE_WIN_S s before crossing = the
                          # near-crossing peak), 'crossing' (trial start ->
                          # crossing, trial-mean; diluted by lengthening trials),
                          # or 'fixed' (first W s post-cue, capsule-style)
LATE_WIN_S  = 2.0    # pre-crossing window (s) for WINDOW == 'precross'
W           = 5.0    # post-cue window (s) for WINDOW == 'fixed'
MA          = 10     # moving-average window (trials)
MIN_SESS    = 10     # hide trials with fewer than this many contributing sessions
DROP_FIRST  = 1      # first trial is an anomalous high-drive "freebie"
BASE_N      = 20     # baseline (zero) = mean over first BASE_N trials (smoothed)
PRE_TRIAL_S = 2.0    # roi_csv per-trial chunk starts this many s before the cue
MISS_T      = 10.0   # crossing-time cutoff (s) for a hit, for the expected rate

def save_panel(name, fig=None):
    fig = plt.gcf() if fig is None else fig
    fig.savefig(os.path.join(PANEL_DIR, f'{name}.png'), dpi=300)
    fig.savefig(os.path.join(PANEL_DIR, f'{name}.svg'))

def two_panel(axw=1.5, axh=1.2, left=0.75, gap=1.0, bottom=0.5, right=0.2, top=0.3):
    fig_w = left + 2 * axw + gap + right
    fig_h = bottom + axh + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax1 = fig.add_axes([left / fig_w, bottom / fig_h, axw / fig_w, axh / fig_h])
    ax2 = fig.add_axes([(left + axw + gap) / fig_w, bottom / fig_h,
                        axw / fig_w, axh / fig_h])
    return fig, (ax1, ax2)

def mov(x, k):
    """Edge-normalized, NaN-aware moving average (divides by # valid terms in
    the window, so endpoints aren't dragged toward zero by zero-padding)."""
    x = np.asarray(x, float)
    v = (~np.isnan(x)).astype(float)
    xf = np.where(v > 0, x, 0.0)
    num = np.convolve(xf, np.ones(k), mode='same')
    den = np.convolve(v,  np.ones(k), mode='same')
    return num / np.where(den > 0, den, np.nan)

def _stars(p):
    return ('***' if p < 1e-3 else '**' if p < 1e-2 else
            '*' if p < 0.05 else 'n.s.')

MOUSE_LOW_FLOOR = {'BCI102': 0.36, 'BCI103': 0.16, 'BCI104': 0.22,
                   'BCI105': 0.17, 'BCI106': 0.24, 'BCI109': 0.24}
def low_floor_for(mouse):
    return MOUSE_LOW_FLOOR.get(mouse, 0.23)

def transfer_fun(f, lower, upper, max_speed=3.3, low_floor=0.0):
    """BCI transfer function: threshold-linear with vmax saturation + noise floor."""
    gain = upper - lower
    if gain <= 0:
        return np.zeros_like(f)
    spd = np.clip((f - lower) / gain * max_speed, 0.0, max_speed)
    return np.maximum(spd, low_floor)

#%% ============================================================================
# CELL 1: load compiled workspace -> per-session (hit_diff, drive_diff) traces
# ============================================================================
with open(PKL, 'rb') as fh:
    D = pickle.load(fh)
session_keys = D['session_keys']
cn_fluor_list = D['cn_fluor_list']          # per-trial roi_csv CN (threshold units)
switches_list = D['switches_list']
dt_si_list = D['dt_si_list']
tc_mat = D['tc_mat']                         # crossing time (s from trial start)
thr_lower_mat = D['thr_lower_mat']
thr_upper_mat = D['thr_upper_mat']
trials_per_session = D['trials_per_session']
print(f"Loaded {len(session_keys)} sessions from compiled pickle.")

def session_traces(i):
    dt = dt_si_list[i]
    W_f = int(W / dt)   # roi_csv chunk starts at the trial start (no 2s buffer),
                        # so the window is [trial start, +W], no pre_f offset
    fl = cn_fluor_list[i]
    n = int(min(trials_per_session[i], len(fl)))
    if n < BASE_N + 5:
        return None
    thr_l = thr_lower_mat[:n, i]; thr_u = thr_upper_mat[:n, i]
    tc = tc_mat[:n, i]
    sw = np.asarray(switches_list[i], int); sw = sw[sw < n]
    if len(sw) < 2:
        return None
    fl_ = thr_l[np.isfinite(thr_l)]; fu_ = thr_u[np.isfinite(thr_u)]
    if len(fl_) == 0 or len(fu_) == 0:
        return None
    lower = float(fl_[0])
    upr = thr_u[sw]                          # threshold at each epoch's start
    upper1 = float(upr[0])
    if not np.isfinite(upper1) or upper1 <= lower:
        return None
    fe = int(sw[1])                          # first-epoch end

    # behavior: actual (MA) minus expected hit rate, using the SATURATION-CORRECT
    # expected (replay epoch-0 CN through the epoch-k transfer fn, clip included,
    # as in threshold_analysis2's expected_hr_correct) -- NOT the linear gain
    # scaling, which over-predicts the slowdown wherever the CN saturates.
    floor = low_floor_for(session_keys[i][0])
    hit = np.isfinite(tc).astype(float)
    e0 = [t for t in range(fe) if np.isfinite(tc[t])]      # epoch-0 hit trials
    exp = np.full(n, np.nan)
    for k in range(len(sw)):
        s = int(sw[k]); e = int(sw[k + 1]) if k + 1 < len(sw) else n
        uk = upr[k]
        hk = []
        for t in e0:
            fu = np.asarray(fl[t][:int(tc[t] / dt)], float)
            so = np.nanmean(transfer_fun(fu, lower, upper1, low_floor=floor))
            sn = np.nanmean(transfer_fun(fu, lower, uk,     low_floor=floor))
            if so > 0 and np.isfinite(sn) and sn > 0:
                hk.append(tc[t] / (sn / so) < MISS_T)   # scale RT by speed ratio
        exp[s:e] = np.mean(hk) if hk else np.nan

    # drive index vs FIXED epoch-1 threshold; window = trial start -> crossing
    # (WINDOW='crossing') or fixed first-W-s post-cue (WINDOW='fixed').
    d = np.full(n, np.nan)
    for t in range(n):
        f = np.asarray(fl[t], float)
        cr = int(tc[t] / dt) if np.isfinite(tc[t]) else len(f)
        if WINDOW == 'precross':
            w = f[max(0, cr - int(LATE_WIN_S / dt)):cr]    # near-crossing peak
        elif WINDOW == 'crossing':
            w = f[:cr]                                     # whole trial (diluted)
        else:                                              # fixed post-cue
            w = f[:W_f]
        w = w[np.isfinite(w)]
        if len(w) >= 3:
            d[t] = np.mean(np.maximum(w - lower, 0.0) / (upper1 - lower))

    # drop anomalous first trial(s), smooth, baseline on the first BASE_N trials
    hit = hit[DROP_FIRST:]; exp = exp[DROP_FIRST:]; d = d[DROP_FIRST:]
    hit_diff = mov(hit, MA) - exp
    sd = mov(d, MA)
    base = np.nanmean(sd[0:min(BASE_N, len(sd))])
    return hit_diff, sd - base

hit_traces, drive_traces = [], []
for i in range(len(session_keys)):
    tr = session_traces(i)
    if tr is not None:
        hit_traces.append(tr[0]); drive_traces.append(tr[1])
print(f"Usable sessions: {len(hit_traces)}")

#%% ============================================================================
# CELL 2: pooled two-panel figure (behavior + CN drive) vs trial
# ============================================================================
def pool(traces):
    L = max(len(t) for t in traces)
    M = np.full((len(traces), L), np.nan)
    for i, t in enumerate(traces):
        M[i, :len(t)] = t
    n = np.sum(~np.isnan(M), axis=0)
    mean = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0) / np.sqrt(np.maximum(n, 1))
    return mean, sem, n

def late_third(t):
    t = np.asarray(t, float)
    return np.nanmean(t[len(t) * 2 // 3:])

fig, (axB, axD) = two_panel()
for ax, traces, ylab, title in [
        (axB, hit_traces, 'actual - expected', 'Hit rate'),
        (axD, drive_traces, 'CN drive - baseline', 'CN drive (fixed thr)')]:
    mean, sem, n = pool(traces)
    cut = 0
    for i in range(len(n)):
        if n[i] >= MIN_SESS:
            cut = i
        else:
            break
    x = np.arange(cut + 1)
    ax.fill_between(x, (mean - sem)[:cut + 1], (mean + sem)[:cut + 1],
                    color='crimson', alpha=0.2, lw=0)
    ax.plot(x, mean[:cut + 1], color='crimson', lw=1.2)
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    lt = np.array([late_third(t) for t in traces])
    lt = lt[np.isfinite(lt)]
    p = wilcoxon(lt).pvalue if len(lt) > 1 else np.nan
    ax.text(0.03, 0.97, f'{_stars(p)}\np={p:.3f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=7)
    ax.set_xlabel('Trial #'); ax.set_ylabel(ylab); ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
save_panel('cn_drive_index_trace')
print(f"behavior late-third vs 0: p={wilcoxon([late_third(t) for t in hit_traces]).pvalue:.4f}")
print(f"CN drive late-third vs 0: p={wilcoxon([late_third(t) for t in drive_traces]).pvalue:.4f}")
