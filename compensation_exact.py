#%% ============================================================================
# compensation_exact.py
#
# The CORRECT compensation number, using the EXACT-integral no-adaptation null
# (replays the actual CN forward through the real transfer function -- saturation
# clip AND low_floor -- and integrates the port to the target distance D, instead
# of assuming speed scales by 1/alpha). This fixes both regimes that break the
# linear diagonal in lickport_speed_analysis.py CELL 5: peak fluorescence above
# upper (saturation) and the discrete step just above lower (the floor).
#
# NO reload: reads all_session_trials from threshold_analysis_062826.py (run that
# first), so the per-trial CN (roi_csv, threshold units, no buffer), crossing
# indices, thresholds and dt all come from that script's verified loading.
#
# Reference is EPOCH 0 (fixed baseline) for every later epoch, so both the
# actual/expected ratio and dCN compare the current epoch to the FIRST epoch
# (cumulative compensation from baseline), matching the per-session figure.
# Per current epoch e (vs epoch 0):
#   D          = median over epoch-0 hit trials of  integral transfer_fun(CN)dt
#                to crossing   (self-consistent port distance)
#   noadapt_RT = replay each epoch-0 CN trace through the epoch-e transfer fn
#                until the port integral reaches D (floor-creep extrapolation
#                beyond the chunk; capped at the 10 s miss timeout)
#   f          = 1 - (art_cur - art_0) / (noadapt_RT - art_0)
#                1 = full compensation, 0 = matched the exact no-adapt null, <0 worse
#   dCN        = mean CN(epoch e) - mean CN(epoch 0)
#
# SANITY: a genuine CN increase (dCN>0) MUST cross faster than the (lower)
# epoch-0 CN, so dCN>0 paired with no compensation flags a bug, not a real null.
#
# Run: threshold_analysis_062826.py (through the all-sessions loop) -> this file.
# ============================================================================
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 7, 'svg.fonttype': 'none'})
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')

MISS_RT   = 10.0    # s; miss timeout
MIN_GAP   = 0.3     # s; minimum predicted slowdown to define a fraction
MIN_REF   = 3       # min reference-epoch hits to calibrate D / no-adapt
D_MIN     = 0.3     # min port distance; below this the transition is degenerate
                    # (gated floor -> D~0 when reference CN barely clears lower),
                    # so the expected crossing and actual/expected ratio blow up
MAX_SPEED = 3.3

MOUSE_LOW_FLOOR = {'BCI102': 0.36, 'BCI103': 0.16, 'BCI104': 0.22,
                   'BCI105': 0.17, 'BCI106': 0.24, 'BCI109': 0.24}
def low_floor_for(m):
    return MOUSE_LOW_FLOOR.get(m, 0.23)

def transfer_fun(f, lower, upper, low_floor):
    """Real BCI transfer: threshold-linear + saturation clip, minimum step
    (low_floor) WHEN moving, and the port FROZEN (speed 0) when F <= lower
    (the lickport cannot move while fluorescence is below the lower threshold)."""
    f = np.asarray(f, float)
    if upper <= lower:
        return np.where(f > lower, low_floor, 0.0)
    spd = np.maximum(np.clip((f - lower) / (upper - lower) * MAX_SPEED, 0.0, MAX_SPEED),
                     low_floor)
    return np.where(f > lower, spd, 0.0)

def noadapt_rt(cn_full, lo_c, up_c, floor, dt, D):
    """No-adapt crossing time: integrate the recorded trace; if it runs out before
    reaching D, assume the animal SUSTAINS its mean drive (not the floor) over the
    longer trial -> extrapolate the remaining distance at the trace's mean speed."""
    cum = np.cumsum(transfer_fun(cn_full, lo_c, up_c, floor)) * dt
    j = int(np.searchsorted(cum, D))
    if j < len(cum):
        return (j + 1) * dt
    mean_spd = cum[-1] / (len(cum) * dt)                     # sustained mean drive
    if mean_spd <= 0:
        return MISS_RT
    extra = (D - cum[-1]) / mean_spd
    return min(len(cum) * dt + extra, MISS_RT)

#%% --- per-transition exact-integral compensation --------------------------------
try:
    all_session_trials
except NameError:
    raise RuntimeError("Run threshold_analysis_062826.py (through the all-sessions "
                       "loop) first -- this script reads all_session_trials.")

recs = []
for (mouse, session), st in all_session_trials.items():
    if 'cn_fluor' not in st:
        continue   # re-run the loop after adding the raw-data fields
    cn_f = st['cn_fluor']; stp = st['cn_stp']; dt = float(st['dt'])
    thr_l = st['thr_l']; thr_u = st['thr_u']
    hit = np.asarray(st['hit']); rt = np.asarray(st['rt'], float)
    sw = np.asarray(st['switches'], int)
    floor = low_floor_for(mouse)
    n = len(hit); ends = np.concatenate((sw[1:], [n]))

    for k in range(1, len(sw)):
        e0, e1 = int(sw[0]), int(sw[1])            # FIRST epoch = fixed baseline
        s0, s1 = int(sw[k]), int(ends[k])          # current epoch [s0, s1)
        lo_p, up_p = float(thr_l[e0]), float(thr_u[e0])   # epoch-0 thresholds
        lo_c, up_c = float(thr_l[s0]), float(thr_u[s0])
        if not (up_c > up_p) or up_p <= lo_p or up_c <= lo_c:
            continue                               # threshold INCREASES only
        alpha = (up_c - lo_c) / (up_p - lo_p)

        ref = [t for t in range(e0, e1)
               if t < len(cn_f) and hit[t] and stp[t] >= 3]
        if len(ref) < MIN_REF:
            continue
        Ds = [np.sum(transfer_fun(cn_f[t][:stp[t]], lo_p, up_p, floor)) * dt
              for t in ref]
        D = float(np.median(Ds))
        if not np.isfinite(D) or D < D_MIN:
            continue   # degenerate port distance -> expected crossing undefined
        na = [noadapt_rt(cn_f[t], lo_c, up_c, floor, dt, D) for t in ref]
        na_rt = float(np.median(na))
        frac_miss = float(np.mean([x >= MISS_RT for x in na]))

        art_prev = float(np.median([rt[t] if hit[t] else MISS_RT for t in range(e0, e1)]))
        art_cur  = float(np.median([rt[t] if hit[t] else MISS_RT for t in range(s0, s1)]))

        cnp = float(np.nanmean([np.nanmean(cn_f[t][:stp[t]]) for t in ref]))
        cur = [t for t in range(s0, s1) if t < len(cn_f) and hit[t] and stp[t] >= 3]
        cnc = float(np.nanmean([np.nanmean(cn_f[t][:stp[t]]) for t in cur])) if cur else np.nan

        # CN DRIVE change: current vs epoch-0 CN, both through the CURRENT transfer
        # -- this is the neural quantity f is built on, so it MUST be >0 where f>0
        # (unlike raw dCN, which is a window-diluted amplitude mean).
        drv_ref = np.nanmean([np.nanmean(transfer_fun(cn_f[t][:stp[t]], lo_c, up_c, floor))
                              for t in ref])
        drv_cur = (np.nanmean([np.nanmean(transfer_fun(cn_f[t][:stp[t]], lo_c, up_c, floor))
                               for t in cur]) if cur else np.nan)

        pred_slow = na_rt - art_prev
        f = (1.0 - (art_cur - art_prev) / pred_slow) if pred_slow > MIN_GAP else np.nan
        recs.append(dict(mouse=mouse, session=session, alpha=alpha, D=D, D_iqr=np.subtract(*np.percentile(Ds, [75, 25])),
                         noadapt_rt=na_rt, art_prev=art_prev, art_cur=art_cur,
                         dCN=cnc - cnp, d_drive=drv_cur - drv_ref, frac_miss=frac_miss, f=f))

print(f"\n{len(recs)} threshold-increase transitions "
      f"({len(set((r['mouse'], r['session']) for r in recs))} sessions)")

#%% --- aggregate + report -------------------------------------------------------
alpha = np.array([r['alpha'] for r in recs])
f     = np.array([r['f'] for r in recs])
dCN   = np.array([r['dCN'] for r in recs])
na_rt = np.array([r['noadapt_rt'] for r in recs])
aprev = np.array([r['art_prev'] for r in recs])
acur  = np.array([r['art_cur'] for r in recs])
okf   = np.isfinite(f)

by = {}
for r in recs:
    if np.isfinite(r['f']):
        by.setdefault((r['mouse'], r['session']), []).append(r['f'])
sess_f = np.array([np.median(v) for v in by.values()])
p_sess = wilcoxon(sess_f).pvalue if len(sess_f) > 1 else np.nan

print("\n================ COMPENSATION (exact-integral null) ================")
print(f"  per-transition median f = {np.median(f[okf]):+.3f}  (n={okf.sum()})")
print(f"  per-session   median f = {np.median(sess_f):+.3f}  "
      f"(n={len(sess_f)}, Wilcoxon p={p_sess:.4f})")
print(f"  [f = 1 full compensation, 0 = matched exact no-adapt, <0 worse]")

# consistency: if f > 0 (sig), the CN change should be too -- but on which measure?
def _sess_med_p(key):
    by = {}
    for r in recs:
        if np.isfinite(r.get(key, np.nan)):
            by.setdefault((r['mouse'], r['session']), []).append(r[key])
    m = np.array([np.median(v) for v in by.values()])
    return np.median(m), (wilcoxon(m).pvalue if len(m) > 1 else np.nan), len(m)
print("\n---- is the CN change also > 0 on average? (must be, where f > 0) ----")
for _k, _lab in [('dCN', 'raw mean CN change (active window)'),
                 ('d_drive', 'CN DRIVE change (mean transfer speed)')]:
    _m, _p, _n = _sess_med_p(_k)
    print(f"  per-session median {_lab:38s} = {_m:+.4f}, p={_p:.4f} (n={_n})")

ratio_ae = na_rt / acur          # expected / actual crossing; > 1 = compensation
okd = np.isfinite(dCN) & np.isfinite(ratio_ae) & (acur > 0)
r_d, p_d = pearsonr(ratio_ae[okd], dCN[okd])
comp_side = okd & (ratio_ae > 1.0)      # transitions where the animal beat the null
print("\n----------- SANITY: does CN rise where the animal compensates? --------")
print(f"  dCN vs expected/actual ratio: r = {r_d:+.3f}, p = {p_d:.4f}")
print(f"  median dCN, compensated (ratio>1) = {np.median(dCN[comp_side]):+.3f} | "
      f"not (ratio<=1) = {np.median(dCN[okd & ~comp_side]):+.3f}")

# real difficulty (~alpha, transfer-function based): how much the no-adapt port
# would slow relative to the epoch-0 baseline crossing
real_diff = na_rt / aprev
okrd = np.isfinite(dCN) & np.isfinite(real_diff) & (aprev > 0)
r_rd, p_rd = pearsonr(real_diff[okrd], dCN[okrd])
print(f"  dCN vs real difficulty (expected/epoch0-actual): r = {r_rd:+.3f}, p = {p_rd:.4f}")
inc = okf & np.isfinite(dCN) & (dCN > 0)
if inc.sum() > 0:
    print(f"  transitions where CN rose (dCN>0): n={inc.sum()}, "
          f"median f = {np.median(f[inc]):+.3f}")
    if inc.sum() >= 5 and np.median(f[inc]) < 0.05:
        print("  *** WARNING: CN rises but compensation ~0 -> suspect a bug ***")
print(f"\n  median no-adapt RT = {np.median(na_rt):.2f}s | actual RT = "
      f"{np.median(acur):.2f}s | prev RT = {np.median(aprev):.2f}s")
print(f"  median frac of no-adapt replays that miss (>{MISS_RT}s) = "
      f"{np.median([r['frac_miss'] for r in recs]):.2f}")
print(f"  median D-IQR/D (model self-consistency, want small) = "
      f"{np.median([r['D_iqr']/r['D'] for r in recs if r['D']>0]):.2f}")

# --- name the most extreme actual/expected transitions (candidate outliers) ---
_fin = np.where(np.isfinite(ratio_ae) & (ratio_ae > 0))[0]
_ord = _fin[np.argsort(-np.abs(np.log(ratio_ae[_fin])))]
print("\n  most extreme expected/actual transitions (candidate outliers):")
for j in _ord[:6]:
    r = recs[j]
    print(f"    {r['mouse']} {r['session']}  ratio={ratio_ae[j]:6.2f}  "
          f"act={r['art_cur']:5.2f}/exp={r['noadapt_rt']:5.2f}  alpha={r['alpha']:.2f}  "
          f"dCN={r['dCN']:+6.1f}  frac_miss={r['frac_miss']:.2f}  D={r['D']:.2f}")

#%% --- figure: null | compensation fraction | CN sanity -------------------------
fig, ax = plt.subplots(1, 4, figsize=(11.0, 2.6))

# (A) actual vs exact no-adapt slowdown; below y=x = compensation
axA = ax[0]
xs, ys = (na_rt - aprev)[okf], (acur - aprev)[okf]
lim = [min(0, np.nanpercentile(np.r_[xs, ys], 2)),
       np.nanpercentile(np.r_[xs, ys], 98) * 1.05]
axA.scatter(xs, ys, s=14, c='k', alpha=0.5, edgecolors='none')
axA.plot(lim, lim, color='gray', lw=0.8, ls='--', label='no adaptation')
axA.axhline(0, color='cornflowerblue', lw=0.8, ls=':', label='full comp.')
axA.set_xlim(lim); axA.set_ylim(lim); axA.set_aspect('equal')
axA.set_xlabel('exact no-adapt slowdown (s)'); axA.set_ylabel('actual slowdown (s)')
axA.set_title('Actual vs exact-null slowdown', fontsize=8)
axA.legend(frameon=False, fontsize=6, loc='upper left')
axA.spines[['top', 'right']].set_visible(False)

# (B) compensation fraction box (the headline number)
axB = ax[1]
axB.boxplot(sess_f, widths=0.6, showfliers=False,
            medianprops=dict(color='crimson', linewidth=1.5))
axB.scatter(np.random.uniform(0.82, 1.18, len(sess_f)), sess_f, s=12, c='k',
            alpha=0.4, edgecolors='none')
axB.axhline(0, color='gray', lw=0.8, ls='--', label='no comp.')
axB.axhline(1, color='cornflowerblue', lw=0.8, ls=':', label='full comp.')
axB.set_xticks([1]); axB.set_xticklabels([f'sessions\n(n={len(sess_f)})'])
axB.set_ylabel('compensation fraction (exact null)')
axB.set_title(f'median {np.median(sess_f):.2f}\nvs 0: p={p_sess:.1e}', fontsize=8)
axB.legend(frameon=False, fontsize=6, loc='lower right')
axB.spines[['top', 'right']].set_visible(False)

# (C) CN change vs COMPENSATION (expected/actual crossing, not the fake alpha)
axC = ax[2]
axC.scatter(ratio_ae[okd], dCN[okd], s=14, c='k', alpha=0.5, edgecolors='none')
if okd.sum() >= 3:
    m, b = np.polyfit(ratio_ae[okd], dCN[okd], 1)
    xf = np.linspace(ratio_ae[okd].min(), ratio_ae[okd].max(), 50)
    axC.plot(xf, m * xf + b, color='crimson', lw=1.2)
axC.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
axC.axvline(1.0, color='gray', lw=0.6, ls='--', alpha=0.5)   # expected = actual
axC.set_xlabel('expected / actual crossing  (>1 = comp.)')
axC.set_ylabel('CN change (current - epoch 0)')
axC.set_title(f'CN change vs compensation\nr={r_d:+.2f}, p={p_d:.3f}', fontsize=8)
axC.spines[['top', 'right']].set_visible(False)

# (D) CN change vs REAL difficulty (expected no-adapt / epoch-0 actual ~ alpha)
axD = ax[3]
axD.scatter(real_diff[okrd], dCN[okrd], s=14, c='k', alpha=0.5, edgecolors='none')
if okrd.sum() >= 3:
    m2, b2 = np.polyfit(real_diff[okrd], dCN[okrd], 1)
    xf2 = np.linspace(real_diff[okrd].min(), real_diff[okrd].max(), 50)
    axD.plot(xf2, m2 * xf2 + b2, color='crimson', lw=1.2)
axD.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
axD.axvline(1.0, color='gray', lw=0.6, ls='--', alpha=0.5)   # no difficulty change
axD.set_xlabel('expected / epoch-0 actual  (real difficulty)')
axD.set_ylabel('CN change (current - epoch 0)')
axD.set_title(f'CN change vs difficulty\nr={r_rd:+.2f}, p={p_rd:.3f}', fontsize=8)
axD.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fname = 'compensation_exact'
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
print(f"\nsaved {fname}.png/.svg")
