#%% ============================================================================
# lickport_speed_analysis.py
#
# Actual vs expected LICKPORT SPEED (steps/s), per epoch — the speed-space
# version of the actual-vs-expected RT scatter. Speed has no ceiling, so it
# avoids the RT 20 s miss-cap entirely: a big gain change just predicts a low
# expected speed, no clipping.
#
#   actual speed (per hit trial) = n_steps / threshold_crossing_time
#   expected speed (per epoch)   = mean_hit_steps / expected_RT (UNCLIPPED)
#
# where expected_RT replays epoch-0 reference fluorescence through the NEW
# transfer function (uncapped). Compensation = actual speed ABOVE the
# no-adaptation diagonal (port moved faster than predicted).
#
# Borrows only the needed pieces from threshold_analysis2.py (transfer_fun,
# epoch detection, roi_csv reconstruction, expected-RT replay).
# ============================================================================

#%% ============================================================================
# CELL 0: Setup
# ============================================================================
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, wilcoxon

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
from BCI_data_helpers import parse_hdf5_array_string

mpl.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'svg.fonttype': 'none',
})
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')

def panel_fig(ax_w=1.25, ax_h=1.25, left=0.7, bottom=0.5, right=0.2, top=0.25):
    """Figure with ONE axis of size (ax_w, ax_h) inches (margins in inches);
    no tight_layout, so the axis box stays exactly that size. Returns (fig, ax)."""
    fig_w, fig_h = left + ax_w + right, bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h])
    return fig, ax

mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
_qc_fail = {('BCI104', '012325'), ('BCI105', '012125'), ('BCI105', '012425')}

# Per-mouse noise floor (matches threshold_analysis2.py)
MOUSE_LOW_FLOOR = {'BCI102': 0.36, 'BCI103': 0.16, 'BCI104': 0.22,
                   'BCI105': 0.17, 'BCI106': 0.24, 'BCI109': 0.24}
DEFAULT_LOW_FLOOR = 0.23
def low_floor_for(m):
    return MOUSE_LOW_FLOOR.get(m, DEFAULT_LOW_FLOOR)

def transfer_fun(fluorescence, lower, upper, max_speed=3.3, low_floor=0.0):
    gain = upper - lower
    if gain <= 0:
        return np.zeros_like(fluorescence)
    speed = np.clip((fluorescence - lower) / gain * max_speed, 0.0, max_speed)
    speed = np.maximum(speed, low_floor)                 # minimum step WHEN moving
    return np.where(fluorescence > lower, speed, 0.0)    # port FROZEN below lower thr

N_REF = 10   # reference trials from epoch 0 for the expected-RT replay

#%% ============================================================================
# CELL 1: Per-session loop — per-epoch actual & expected lickport speed
# ============================================================================
list_of_dirs = session_counting.counter()

epoch_speed = []     # list of dicts, one per epoch (e > 0)
session_speed_data = []   # per-session arrays for switch-aligned plot
load_errors = []

for mouse in mice:
    inds = np.where((list_of_dirs['Mouse'] == mouse) &
                    (list_of_dirs['Has data_main.npy'] == True))[0]
    for si in inds:
        session = list_of_dirs['Session'][si]
        if (mouse, session) in _qc_fail:
            continue
        folder = ('//allen/aind/scratch/BCI/2p-raw/'
                  + mouse + '/' + session + '/pophys/')
        try:
            data = ddct.load_hdf5(folder,
                ['conditioned_neuron', 'dt_si', 'threshold_crossing_time',
                 'reward_time', 'step_time', 'BCI_thresholds',
                 'roi_csv', 'cn_csv_index'], [])
            dt_si = float(np.asarray(data['dt_si']).ravel()[0])
            cn_idx_csv = int(np.asarray(data['cn_csv_index']).ravel()[0])
        except Exception as e:
            load_errors.append(((mouse, session), str(e)))
            continue

        # --- roi_csv -> continuous frame-aligned trace + per-trial chunks ---
        try:
            ops = np.load(folder + 'suite2p_BCI/plane0/ops.npy',
                          allow_pickle=True).tolist()
            fpf = ops['frames_per_file']
            roi = np.copy(data['roi_csv'])
            wraps = np.where(np.diff(roi[:, 1]) < 0)[0]
            for ww in wraps:
                roi[ww + 1:, 1] += roi[ww, 1]
                roi[ww + 1:, 0] += roi[ww, 0]
            frm = np.arange(1, int(np.max(roi[:, 1])) + 1)
            roi_i = interp1d(roi[:, 1], roi, axis=0, kind='linear',
                             fill_value='extrapolate')(frm)
        except Exception as e:
            load_errors.append(((mouse, session), 'roi_csv: ' + str(e)))
            continue

        trl = len(fpf)

        # --- behavioral arrays ---
        tc_p = parse_hdf5_array_string(data['threshold_crossing_time'], trl)
        tc = np.array([x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
                       for x in tc_p], dtype=float)
        rt_p = parse_hdf5_array_string(data['reward_time'], trl)
        rt = np.array([x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
                       for x in rt_p], dtype=float)
        hit = np.isfinite(rt)
        # Step counts per trial — ONLY steps before threshold crossing.
        # Steps logged after tc are motor commands with the port already at
        # the end of its range (CN still active but no movement), so they must
        # be excluded or the step count / speed is inflated.
        st_p = parse_hdf5_array_string(data['step_time'], trl)
        n_steps = np.full(trl, np.nan)
        for i in range(trl):
            s = (np.asarray(st_p[i], dtype=float)
                 if hasattr(st_p[i], '__len__') else np.array([]))
            if np.isfinite(tc[i]):
                n_steps[i] = np.sum(s <= tc[i])
            elif len(s) > 0:
                n_steps[i] = len(s)        # miss: no crossing, keep all (rare use)

        # --- thresholds (forward-fill) + epochs ---
        thr = np.asarray(data['BCI_thresholds'], dtype=float)
        thr_l, thr_u = thr[0, :].copy(), thr[1, :].copy()
        for i in range(1, thr_u.size):
            if np.isnan(thr_u[i]): thr_u[i] = thr_u[i - 1]
            if np.isnan(thr_l[i]): thr_l[i] = thr_l[i - 1]
        if np.isnan(thr_u[0]) and np.any(np.isfinite(thr_u)):
            thr_u[0] = thr_u[np.isfinite(thr_u)][0]
        if np.isnan(thr_l[0]) and np.any(np.isfinite(thr_l)):
            thr_l[0] = thr_l[np.isfinite(thr_l)][0]
        if len(thr_u) < trl:
            thr_u = np.concatenate([thr_u, np.full(trl - len(thr_u), thr_u[-1])])
            thr_l = np.concatenate([thr_l, np.full(trl - len(thr_l), thr_l[-1])])
        d_up = np.diff(thr_u)
        sw = np.concatenate(([0], np.where((d_up != 0) & np.isfinite(d_up))[0] + 1))
        n_ep = len(sw)
        ep_end = np.concatenate((sw[1:], [trl]))

        # --- per-trial CN fluorescence (up to threshold crossing) ---
        cn_fluor, cn_stp = [], []
        strt = 0
        for i in range(min(trl, len(fpf))):
            ind = np.clip(np.arange(strt, strt + fpf[i]), 0, len(roi_i) - 1)
            fl = roi_i[ind, cn_idx_csv + 2]
            cn_fluor.append(fl)
            if hit[i] and np.isfinite(tc[i]):
                t_tr = roi_i[ind, 0] - roi_i[ind[0], 0]
                stp = min(np.searchsorted(t_tr, tc[i]), len(fl))
            else:
                stp = len(fl)
            cn_stp.append(stp)
            strt += fpf[i]

        floor = low_floor_for(mouse)
        TIMEOUT_S = 10.0
        D_steps = np.nanmean(n_steps[hit]) if np.any(hit) else np.nan

        # Actual per-trial lickport speed (steps/s) over ALL trials — this is
        # the whole point: misses get a real, low, continuous speed (port got
        # partway in the timeout), no cap, no 20s fill.
        #   hit  -> steps before crossing / crossing time
        #   miss -> steps achieved / timeout
        actual_speed_tr = np.full(trl, np.nan)
        for i in range(trl):
            if not np.isfinite(n_steps[i]):
                continue
            if hit[i] and np.isfinite(tc[i]) and tc[i] > 0:
                actual_speed_tr[i] = n_steps[i] / tc[i]
            elif not hit[i]:
                actual_speed_tr[i] = n_steps[i] / TIMEOUT_S

        # Reference window = first N_REF trials of epoch 0
        n_ref = min(N_REF, ep_end[0])
        ref_trials = [t for t in range(n_ref) if t < len(cn_fluor)]
        lo_ref, up_ref = thr_l[0], thr_u[0]

        # Epoch-0 actual speed + epoch-0 transfer-fn speed, to calibrate the
        # transfer function (arbitrary units) into measured steps/s.
        ep0_actual_speed = np.nanmean(actual_speed_tr[ref_trials])
        tr_ep0 = [np.nanmean(transfer_fun(cn_fluor[t][:cn_stp[t]],
                                          lo_ref, up_ref, low_floor=floor))
                  for t in ref_trials if cn_stp[t] > 0]
        spd_ref_ep0 = np.nanmean(tr_ep0) if tr_ep0 else np.nan

        ep_expected = {}   # epoch index -> expected (no-adapt) speed
        for ei in range(1, n_ep):
            t0, t1 = sw[ei], ep_end[ei]
            lo_cur, up_cur = thr_l[t0], thr_u[t0]

            # Expected speed (no adaptation): epoch-0 activity through the NEW
            # transfer function, scaled to measured steps/s via epoch 0.
            # Bounded (no RT round-trip / extrapolation), and naturally low for
            # hard thresholds (the "would-be miss" prediction).
            tr_new = [np.nanmean(transfer_fun(cn_fluor[t][:cn_stp[t]],
                                              lo_cur, up_cur, low_floor=floor))
                      for t in ref_trials if cn_stp[t] > 0]
            spd_ref_new = np.nanmean(tr_new) if tr_new else np.nan
            if (np.isfinite(ep0_actual_speed) and np.isfinite(spd_ref_ep0)
                    and spd_ref_ep0 > 0 and np.isfinite(spd_ref_new)):
                expected_speed = ep0_actual_speed * (spd_ref_new / spd_ref_ep0)
            else:
                expected_speed = np.nan

            # Actual speed = mean over ALL of this epoch's trials (hits+misses)
            actual_speed = np.nanmean(actual_speed_tr[t0:t1])
            ep_expected[ei] = expected_speed

            epoch_speed.append({
                'mouse': mouse, 'session': session, 'epoch': ei,
                'actual_speed': actual_speed,
                'expected_speed': expected_speed,
                'D_steps': D_steps,
                'hit_rate': float(np.mean(hit[t0:t1])),
                'gain_ratio': ((up_cur - lo_cur) / (up_ref - lo_ref)
                               if (up_ref - lo_ref) > 0 else np.nan),
                'n_trials': int(t1 - t0),
            })
        session_speed_data.append({
            'mouse': mouse, 'session': session,
            'actual_speed_tr': actual_speed_tr.copy(),
            'sw': sw.copy(), 'thr_u': thr_u.copy(), 'thr_l': thr_l.copy(),
            'ep_expected': ep_expected,
            'ep0_actual_speed': ep0_actual_speed,
        })
        print(f"  {mouse} {session}: {n_ep} epochs, D={D_steps:.1f} steps, "
              f"ep0 speed={ep0_actual_speed:.2f} steps/s")

print(f"\nCollected {len(epoch_speed)} epochs (>0) from "
      f"{len(set((e['mouse'], e['session']) for e in epoch_speed))} sessions; "
      f"{len(load_errors)} load errors")
for k, e in load_errors[:6]:
    print(f"  err {k}: {e[:110]}")

#%% ============================================================================
# CELL 2: Actual vs expected lickport speed scatter
# ============================================================================
act = np.array([e['actual_speed'] for e in epoch_speed], dtype=float)
exp = np.array([e['expected_speed'] for e in epoch_speed], dtype=float)
ok = np.isfinite(act) & np.isfinite(exp)

fig, ax = panel_fig()
ax.scatter(exp[ok], act[ok], s=16, c='k', alpha=0.5, edgecolors='none')
lim = [0, np.nanpercentile(np.concatenate([act[ok], exp[ok]]), 98) * 1.05]
ax.plot(lim, lim, color='gray', linewidth=0.8, linestyle='--',
        label='no adaptation (y = x)')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
n_above = int(np.sum(act[ok] > exp[ok]))
stat, p = wilcoxon(act[ok] - exp[ok])
ax.set_xlabel('Expected lickport speed\n(epoch-0 speed x transfer ratio)')
ax.set_ylabel('Actual lickport speed\n(steps/s; hits & misses)')
ax.set_title(f'{n_above}/{int(ok.sum())} above diagonal (faster than '
             f'predicted)\nWilcoxon p = {p:.1e}', fontsize=8)
ax.legend(frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fname = 'lickport_speed_actual_vs_expected'
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {fname}: {n_above}/{int(ok.sum())} above diagonal, p={p:.4f}")

#%% ============================================================================
# CELL 3: Switch-aligned lickport-speed recovery (miss-clean)
# ============================================================================
# The recovery plot in speed units. Aligned to each threshold INCREASE:
#   actual speed (steps/s, hits+misses, no cap) recovers after the drop;
#   expected speed (no-adaptation) is a step down to the predicted level.
# Compensation = actual recovering ABOVE the expected post-switch level.
PRE, POST = 5, 14

aln_act, aln_exp = [], []
for rec in session_speed_data:
    asp = rec['actual_speed_tr']
    sw = np.asarray(rec['sw'], dtype=int)
    thr_u = rec['thr_u']
    n_tr = len(asp)
    ep0 = rec['ep0_actual_speed']
    for ei in range(1, len(sw)):
        s = int(sw[ei])
        if s >= n_tr or thr_u[s] <= thr_u[s - 1]:   # increases only
            continue
        exp_post = rec['ep_expected'].get(ei, np.nan)
        prev_sw = sw[ei - 1]
        next_sw = sw[ei + 1] if ei + 1 < len(sw) else n_tr
        a_row = np.full(PRE + POST, np.nan)
        e_row = np.full(PRE + POST, np.nan)
        for k in range(PRE + POST):
            t = s - PRE + k
            if t < 0 or t >= n_tr or t < prev_sw or t >= next_sw:
                continue
            a_row[k] = asp[t]
            e_row[k] = ep0 if k < PRE else exp_post   # no-adapt step
        aln_act.append(a_row)
        aln_exp.append(e_row)
aln_act = np.array(aln_act)
aln_exp = np.array(aln_exp)

def _ms(M):
    n = np.sum(np.isfinite(M), axis=0)
    return (np.nanmean(M, axis=0),
            np.nanstd(M, axis=0) / np.sqrt(np.clip(n, 1, None)))

xa = np.arange(-PRE, POST)
am, asem = _ms(aln_act)
em, _ = _ms(aln_exp)

fig3, ax3 = panel_fig()
ax3.fill_between(xa, am - asem, am + asem, color='k', alpha=0.15, linewidth=0)
ax3.plot(xa, am, 'k', linewidth=1.4, label='Actual (steps/s)')
ax3.plot(xa, em, color='cornflowerblue', linewidth=1.4, linestyle='--',
         label='Expected (no adaptation)')
ax3.axvline(0, color='r', linewidth=0.6, linestyle='--')
ax3.set_xlabel('Trials from threshold change')
ax3.set_ylabel('Lickport speed (steps/s)')
ax3.set_xlim(-PRE, POST - 1)
ax3.set_title(f'Switch-aligned speed recovery (n={aln_act.shape[0]} '
              f'increases)', fontsize=8)
ax3.legend(frameon=False, loc='lower right')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
fname3 = 'lickport_speed_switch_aligned'
fig3.savefig(os.path.join(PANEL_DIR, f'{fname3}.png'), dpi=300)
fig3.savefig(os.path.join(PANEL_DIR, f'{fname3}.svg'))
plt.show()
print(f"Saved {fname3}: n={aln_act.shape[0]} transitions")

#%% ============================================================================
# CELL 4: Switch-aligned recovery in COMPENSATION-FRACTION units
# ============================================================================
# Normalize each transition's speed recovery the way the compensation analysis
# does, so the y-axis has context:
#   f_k = (actual_speed_k − expected_speed) / (baseline_speed − expected_speed)
#   f = 0 : at the no-adaptation level (no compensation)
#   f = 1 : back to pre-switch baseline (full compensation)
# baseline = pre-switch actual speed; expected = no-adapt post-switch speed.
# Require a real perturbation (baseline − expected > MIN_GAP) so the
# denominator doesn't blow up for tiny gain changes.
MIN_GAP = 0.5   # steps/s; minimum predicted slowdown to define a fraction

comp_rows = []
for i in range(aln_act.shape[0]):
    a = aln_act[i]
    base = np.nanmean(a[:PRE])
    exp_post = aln_exp[i, PRE]          # post-switch no-adapt speed
    gap = base - exp_post
    if not np.isfinite(gap) or gap < MIN_GAP:
        continue
    comp_rows.append((a - exp_post) / gap)
comp_rows = np.array(comp_rows)

nC = np.sum(np.isfinite(comp_rows), axis=0)
mC = np.nanmean(comp_rows, axis=0)
sC = np.nanstd(comp_rows, axis=0) / np.sqrt(np.clip(nC, 1, None))

fig4, ax4 = panel_fig()
ax4.fill_between(xa, mC - sC, mC + sC, color='k', alpha=0.15, linewidth=0)
ax4.plot(xa, mC, 'k', linewidth=1.4)
ax4.axvline(0, color='r', linewidth=0.6, linestyle='--')
ax4.axhline(0, color='gray', linewidth=0.8, linestyle='--',
            label='no compensation')
ax4.axhline(1, color='cornflowerblue', linewidth=0.8, linestyle=':',
            label='full compensation')
ax4.set_xlabel('Trials from threshold change')
ax4.set_ylabel('Compensation fraction\n(0 = no adapt, 1 = full)')
ax4.set_xlim(-PRE, POST - 1)
ax4.set_title(f'Switch-aligned compensation (n={comp_rows.shape[0]} '
              f'increases, gap>{MIN_GAP})', fontsize=8)
ax4.legend(frameon=False, fontsize=7, loc='lower right')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
fname4 = 'lickport_speed_switch_aligned_compfrac'
fig4.savefig(os.path.join(PANEL_DIR, f'{fname4}.png'), dpi=300)
fig4.savefig(os.path.join(PANEL_DIR, f'{fname4}.svg'))
plt.show()
print(f"Saved {fname4}: n={comp_rows.shape[0]} transitions")

#%% ============================================================================
# CELL 5: Gain ratio vs SPEED ratio — miss-clean compensation dose-response
# ============================================================================
# Speed version of the RT-ratio-vs-alpha plot. Per threshold-increase
# transition (epoch e vs e-1):
#   x = alpha = gain_new / gain_old
#   y = prev_speed / current_speed  (mean step-speeds, hits+misses)
# No-adaptation: current = prev/alpha -> y = alpha -> diagonal. Below diagonal
# = compensation. Anchored fit through (1,1): compensated = 1 - beta.
sp_alpha, sp_ratio, sp_sess = [], [], []
for ri, rec in enumerate(session_speed_data):
    asp = rec['actual_speed_tr']
    sw = np.asarray(rec['sw'], dtype=int)
    thr_l, thr_u = rec['thr_l'], rec['thr_u']
    n_tr = len(asp)
    ep_end = np.concatenate((sw[1:], [n_tr]))
    msp = [np.nanmean(asp[sw[e]:min(ep_end[e], n_tr)]) for e in range(len(sw))]
    gain = [thr_u[sw[e]] - thr_l[sw[e]] for e in range(len(sw))]
    for e in range(1, len(sw)):
        if thr_u[sw[e]] <= thr_u[sw[e] - 1]:        # increases only
            continue
        if not (msp[e] > 0 and msp[e - 1] > 0 and gain[e - 1] > 0):
            continue
        sp_alpha.append(gain[e] / gain[e - 1])
        sp_ratio.append(msp[e - 1] / msp[e])        # prev/current speed
        sp_sess.append(ri)
sp_alpha = np.array(sp_alpha); sp_ratio = np.array(sp_ratio)
sp_sess = np.array(sp_sess)

def _anchored_beta(x, y):
    xs = x - 1.0
    d = np.sum(xs * xs)
    return np.sum(xs * (y - 1.0)) / d if d > 0 else np.nan

oks = np.isfinite(sp_alpha) & np.isfinite(sp_ratio)
beta_s = _anchored_beta(sp_alpha[oks], sp_ratio[oks])
comp_s = 1.0 - beta_s
frac_below = np.mean(sp_ratio[oks] < sp_alpha[oks])
# session-clustered bootstrap CI
uq = np.unique(sp_sess[oks])
rng = np.random.default_rng(0)
bb = []
for _ in range(5000):
    ch = rng.choice(uq, size=len(uq), replace=True)
    xa = np.concatenate([sp_alpha[oks][sp_sess[oks] == c] for c in ch])
    ya = np.concatenate([sp_ratio[oks][sp_sess[oks] == c] for c in ch])
    if len(xa) >= 3:
        bb.append(_anchored_beta(xa, ya))
bb = np.array(bb)
ci_lo, ci_hi = np.percentile(bb, [2.5, 97.5])

# --- Per-session compensation fraction (for the box panel) ---
ALPHA_MIN_BOX = 1.2
f_tr = (sp_alpha - sp_ratio) / (sp_alpha - 1.0)   # (alpha - y)/(alpha - 1)
okb = oks & (sp_alpha > ALPHA_MIN_BOX) & np.isfinite(f_tr)
sess_f = {}
for v, sid in zip(f_tr[okb], sp_sess[okb]):
    sess_f.setdefault(sid, []).append(v)
sess_f_med = np.array([np.median(v) for v in sess_f.values()])
med_box = np.median(sess_f_med)
_, p_box = wilcoxon(sess_f_med)

# --- Anchored linear fit + session-clustered bootstrap CI band ---
xf = np.linspace(1.0, sp_alpha[oks].max(), 100)
band = 1.0 + np.outer(bb, xf - 1.0)          # (nboot, 100) predicted y
fit_lo = np.percentile(band, 2.5, axis=0)
fit_hi = np.percentile(band, 97.5, axis=0)
fit_y = 1.0 + beta_s * (xf - 1.0)

# --- Two-panel figure with explicit axis dimensions (inches) ---
fig_w, fig_h = 6.2, 2.9
axh = 1.9
axw_sc, axw_bx = 1.9, 0.85          # scatter / box axis widths
bot = 0.55
fig5 = plt.figure(figsize=(fig_w, fig_h))
axL = fig5.add_axes([0.55 / fig_w, bot / fig_h, axw_sc / fig_w, axh / fig_h])
axR = fig5.add_axes([(0.55 + axw_sc + 1.15) / fig_w, bot / fig_h,
                     axw_bx / fig_w, axh / fig_h])

# Left: dose-response scatter
axL.scatter(sp_alpha[oks], sp_ratio[oks], s=14, c='k', alpha=0.5,
            edgecolors='none')
lim = [0.9, np.nanpercentile(sp_alpha[oks], 98) * 1.05]
axL.plot(lim, lim, color='gray', linewidth=0.8, linestyle='--',
         label='no adaptation')
axL.axhline(1.0, color='cornflowerblue', linewidth=0.8, linestyle=':',
            label='full compensation')
axL.fill_between(xf, fit_lo, fit_hi, color='crimson', alpha=0.2,
                 linewidth=0, zorder=4)
axL.plot(xf, fit_y, color='crimson', linewidth=1.2, zorder=5,
         label='anchored fit ± 95% CI')
axL.set_xlim(lim)
axL.set_xlabel('gain ratio (gain$_{new}$/gain$_{old}$)')
axL.set_ylabel('speed ratio (prev/current)')
axL.set_title(f'compensated {comp_s:.0%} '
              f'[{1-ci_hi:.0%}, {1-ci_lo:.0%}], n={int(oks.sum())}', fontsize=8)
axL.legend(frameon=False, fontsize=6, loc='lower right')
axL.spines['top'].set_visible(False)
axL.spines['right'].set_visible(False)

# Right: per-session compensation-fraction box
axR.boxplot(sess_f_med, widths=0.6, showfliers=False,
            medianprops=dict(color='crimson', linewidth=1.5))
axR.scatter(np.random.uniform(0.82, 1.18, len(sess_f_med)), sess_f_med,
            s=12, c='k', alpha=0.4, edgecolors='none')
axR.axhline(0.0, color='gray', linewidth=0.8, linestyle='--')
axR.axhline(1.0, color='cornflowerblue', linewidth=0.8, linestyle=':')
axR.set_xticks([1])
axR.set_xticklabels([f'sessions\n(n={len(sess_f_med)})'])
axR.set_ylabel('compensation fraction')
axR.set_title(f'median {med_box:.2f}\nvs 0: p={p_box:.1e}', fontsize=8)
axR.spines['top'].set_visible(False)
axR.spines['right'].set_visible(False)

fname5 = 'compensation_speed_two_panel'
fig5.savefig(os.path.join(PANEL_DIR, f'{fname5}.png'), dpi=300)
fig5.savefig(os.path.join(PANEL_DIR, f'{fname5}.svg'))
plt.show()
print(f"Saved {fname5}: per-transition compensated={comp_s:.3f} "
      f"[{1-ci_hi:.3f},{1-ci_lo:.3f}] (n={int(oks.sum())}); "
      f"per-session median={med_box:.3f} (n={len(sess_f_med)}, p={p_box:.4f})")
