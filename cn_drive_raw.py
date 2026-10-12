#%% ============================================================================
# cn_drive_raw.py  -- self-contained, loads RAW data only
#
# Deliberately re-implements the loading overhead (roi_csv interpolation,
# frames_per_file chunking, thresholds, switches) from the raw h5, so nothing
# depends on session_metrics' pickled intermediates.
#
# Key conventions (verified, not inherited):
#   - roi_csv is interpolated onto the integer imaging-frame grid via its frame
#     column roi[:,1] (fills dropped frames); CN = roi_i[:, cn_csv_index+2]
#     (threshold units, the signal that actually drives the port).
#   - Per-trial chunks come from ops['frames_per_file']; each chunk starts at
#     the trial start (NO 2 s pre-cue buffer on the continuous roi_csv).
#   - threshold_crossing_time is seconds from trial start (relative).
#   - transfer_fun = threshold-linear with vmax saturation AND the per-mouse
#     low_floor (the port creeps at the floor regardless of the bar).
#
# Outputs (both from raw data):
#   CELL 2: CN drive index vs trial (rectified CN vs FIXED epoch-1 threshold).
#   CELL 3: behavior, floor-correct -- actual crossing time vs EXACT-integral
#           no-adapt crossing (replay epoch-0 CN through each epoch's transfer
#           fn until the port integral reaches the same distance D). This is the
#           honest compensation test; the linear tc/al baseline is NOT used.
#
# Run CELL 0 -> CELL 1 (slow network load) -> CELL 2 / CELL 3.
# ============================================================================

#%% ============================================================================
# CELL 0: setup
# ============================================================================
import os, sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

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
os.makedirs(PANEL_DIR, exist_ok=True)

# ---- config -----------------------------------------------------------------
LATE_WIN_S     = 2.0     # pre-crossing window (s) for the drive index
MA             = 10      # moving-average window (trials)
MIN_SESS       = 10      # hide trials with fewer than this many sessions
DROP_FIRST     = 1       # anomalous first trial
BASE_N         = 20      # baseline (zero) = mean over first BASE_N trials
MISS_T         = 10.0    # crossing-time timeout (s)
MAX_CN_DIST_UM = 5.0     # drop sessions with CN ROI farther than this
N_TRIALS       = 250
MAX_SPEED      = 3.3     # transfer-function vmax

MOUSE_LOW_FLOOR = {'BCI102': 0.36, 'BCI103': 0.16, 'BCI104': 0.22,
                   'BCI105': 0.17, 'BCI106': 0.24, 'BCI109': 0.24}
def low_floor_for(mouse):
    return MOUSE_LOW_FLOOR.get(mouse, 0.23)

def transfer_fun(f, lower, upper, low_floor=0.0):
    """Threshold-linear with vmax saturation + noise floor (the port speed)."""
    gain = upper - lower
    if gain <= 0:
        return np.full_like(np.asarray(f, float), low_floor)
    spd = np.clip((np.asarray(f, float) - lower) / gain * MAX_SPEED, 0.0, MAX_SPEED)
    return np.maximum(spd, low_floor)

mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
_qc_fail = {('BCI104', '012325'), ('BCI105', '012125'), ('BCI105', '012425')}

def save_panel(name, fig=None):
    fig = plt.gcf() if fig is None else fig
    fig.savefig(os.path.join(PANEL_DIR, f'{name}.png'), dpi=300)
    fig.savefig(os.path.join(PANEL_DIR, f'{name}.svg'))

def panel_fig(ax_w=1.6, ax_h=1.25, left=0.75, bottom=0.55, right=0.2, top=0.3):
    fig_w, fig_h = left + ax_w + right, bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h])
    return fig, ax

def mov(x, k):
    x = np.asarray(x, float); v = (~np.isnan(x)).astype(float)
    num = np.convolve(np.where(v > 0, x, 0.0), np.ones(k), mode='same')
    den = np.convolve(v, np.ones(k), mode='same')
    return num / np.where(den > 0, den, np.nan)

def _stars(p):
    return ('***' if p < 1e-3 else '**' if p < 1e-2 else
            '*' if p < 0.05 else 'n.s.')

def _ffill(a):
    a = a.astype(float).copy()
    if np.isnan(a[0]) and np.any(np.isfinite(a)):
        a[0] = a[np.isfinite(a)][0]
    for k in range(1, len(a)):
        if np.isnan(a[k]):
            a[k] = a[k - 1]
    return a

#%% ============================================================================
# CELL 1: per-session RAW load -> per-trial CN drive + per-epoch crossing tests
# ============================================================================
list_of_dirs = session_counting.counter()

drive_rows = []                                   # (N_TRIALS,) per session
keys, dt_list = [], []
epoch_recs = []                                   # per (session, escalated epoch)
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
                ['conditioned_neuron', 'dist', 'dt_si', 'threshold_crossing_time',
                 'reward_time', 'BCI_thresholds', 'roi_csv', 'cn_csv_index'], [])
            ops = np.load(folder + 'suite2p_BCI/plane0/ops.npy',
                          allow_pickle=True).tolist()
        except Exception as e:
            load_errors.append(((mouse, session), str(e)[:100]))
            continue

        dt = float(np.asarray(data['dt_si']).ravel()[0])
        cn_csv = int(np.asarray(data['cn_csv_index']).ravel()[0])
        cn_su = int(np.asarray(data['conditioned_neuron']).ravel()[0])
        fpf = np.asarray(ops['frames_per_file']).astype(int)

        # CN-to-target distance QC
        try:
            dist_arr = np.asarray(data['dist']).ravel()
            dist_cn = float(dist_arr[cn_su]) if cn_su < len(dist_arr) else np.nan
        except (KeyError, IndexError, TypeError, ValueError):
            dist_cn = np.nan
        if np.isfinite(dist_cn) and dist_cn > MAX_CN_DIST_UM:
            continue

        # roi_csv -> integer imaging-frame grid (fills dropped frames)
        roi = np.copy(data['roi_csv']).astype(float)
        for ww in np.where(np.diff(roi[:, 1]) < 0)[0]:    # unwrap counter wraps
            roi[ww + 1:, 1] += roi[ww, 1]
            roi[ww + 1:, 0] += roi[ww, 0]
        frm = np.arange(1, int(np.max(roi[:, 1])) + 1)
        roi_i = interp1d(roi[:, 1], roi, axis=0, kind='linear',
                         fill_value='extrapolate')(frm)
        cn_cont = roi_i[:, cn_csv + 2]                    # CN, threshold units

        # thresholds (forward-filled), switches (upper changes)
        thr = np.asarray(data['BCI_thresholds'], float)
        thr_l = _ffill(thr[0, :]); thr_u = _ffill(thr[1, :])
        d_up = np.diff(thr_u)
        sw = np.concatenate(([0], np.where((d_up != 0) & np.isfinite(d_up))[0] + 1))

        # crossing time (seconds from trial start), one per trial
        tcp = parse_hdf5_array_string(data['threshold_crossing_time'], len(fpf))
        tc = np.array([x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
                       for x in tcp], dtype=float)

        n = int(min(len(fpf), thr.shape[1], len(tc)))
        if n < BASE_N + 5 or len(sw) < 2:
            continue

        # per-trial CN segments (chunk starts at trial start; no buffer)
        seg, strt = [], 0
        for t in range(n):
            seg.append(cn_cont[strt:strt + fpf[t]]); strt += fpf[t]

        floor = low_floor_for(mouse)
        lo0 = float(thr_l[sw[0]]); up0 = float(thr_u[sw[0]])   # epoch-1 (fixed)
        if up0 <= lo0:
            continue

        # --- per-trial drive index: rectified CN vs FIXED epoch-1, pre-crossing
        lw = int(LATE_WIN_S / dt)
        drive = np.full(n, np.nan)
        for t in range(n):
            cr = int(tc[t] / dt) if np.isfinite(tc[t]) else len(seg[t])
            w = np.asarray(seg[t][max(0, cr - lw):cr], float)
            w = w[np.isfinite(w)]
            if len(w) >= 3:
                drive[t] = np.mean(np.maximum(w - lo0, 0.0) / (up0 - lo0))

        # --- port distance D (credit) from epoch-0 hit trials ----------------
        fe = int(sw[1])
        Ds = []
        for t in range(fe):
            if np.isfinite(tc[t]):
                spd = transfer_fun(seg[t][:int(tc[t] / dt)], lo0, up0, floor)
                if len(spd) >= 3:
                    Ds.append(np.sum(spd) * dt)
        if not Ds:
            continue
        D = float(np.median(Ds))

        # --- per escalated epoch: actual vs EXACT-integral no-adapt crossing --
        for k in range(1, len(sw)):
            s = int(sw[k]); e = int(sw[k + 1]) if k + 1 < len(sw) else n
            lo_k = float(thr_l[s]); up_k = float(thr_u[s])
            if up_k <= lo_k:
                continue
            act = np.nanmedian(tc[s:e])
            preds = []
            for t in range(fe):                  # replay epoch-0 CN @ epoch-k thr
                cum = np.cumsum(transfer_fun(seg[t], lo_k, up_k, floor)) * dt
                j = np.searchsorted(cum, D)
                if j < len(cum):
                    preds.append((j + 1) * dt)
            if preds and np.isfinite(act):
                epoch_recs.append({'mouse': mouse, 'session': session,
                                   'alpha': (up_k - lo_k) / (up0 - lo0),
                                   'actual_tc': act,
                                   'noadapt_tc': float(np.median(preds))})

        # store drive trace (drop anomalous first trial, pad)
        d = drive[DROP_FIRST:]
        out = np.full(N_TRIALS, np.nan); out[:min(len(d), N_TRIALS)] = d[:N_TRIALS]
        drive_rows.append(out); keys.append((mouse, session)); dt_list.append(dt)

drive_mat = np.column_stack(drive_rows) if drive_rows else np.empty((N_TRIALS, 0))
print(f"Loaded {len(keys)} sessions ({len(load_errors)} load errors), "
      f"{len(epoch_recs)} escalated-epoch transitions.")

#%% ============================================================================
# CELL 2: CN drive index vs trial (pooled across sessions)
# ============================================================================
SHOW = N_TRIALS
M = drive_mat[:SHOW, :]
# session-z each column, baseline-subtract first BASE_N, smooth, pool
C = np.full((M.shape[1], SHOW), np.nan)
for j in range(M.shape[1]):
    x = M[:, j].astype(float); sd = np.nanstd(x)
    if np.isfinite(sd) and sd > 0:
        x = (x - np.nanmean(x)) / sd
    b = np.nanmean(x[:BASE_N])
    C[j] = (x - b) if np.isfinite(b) else x
Csm = np.vstack([mov(C[j], MA) for j in range(C.shape[0])])
nT = np.sum(np.isfinite(Csm), axis=0)
m = np.nanmean(Csm, axis=0)
sem = np.nanstd(Csm, axis=0) / np.sqrt(np.clip(nT, 1, None))
keep = nT >= MIN_SESS
x = np.arange(SHOW)
fig, ax = panel_fig()
ax.fill_between(x[keep], (m - sem)[keep], (m + sem)[keep], color='crimson',
                alpha=0.2, lw=0)
ax.plot(x[keep], m[keep], color='crimson', lw=1.2)
ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
ax.set_xlabel('Trial #'); ax.set_ylabel('CN drive (session-z)')
ax.set_title('CN drive vs fixed epoch-1 thr', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
save_panel('cn_drive_raw_vs_trial')

#%% ============================================================================
# CELL 3: behavior -- actual vs EXACT-integral no-adapt crossing (floor-correct)
# ============================================================================
import collections
act = np.array([r['actual_tc'] for r in epoch_recs])
nad = np.array([r['noadapt_tc'] for r in epoch_recs])
alpha = np.array([r['alpha'] for r in epoch_recs])
ok = np.isfinite(act) & np.isfinite(nad) & (nad > 0)
act, nad, alpha = act[ok], nad[ok], alpha[ok]
# per-session medians for an honest paired test (session = unit)
by = collections.defaultdict(list)
for r, a, nn in zip([epoch_recs[i] for i in np.where(ok)[0]], act, nad):
    by[(r['mouse'], r['session'])].append(nn - a)        # noadapt - actual (>0 = comp)
sess_gap = np.array([np.median(v) for v in by.values()])
_, p = wilcoxon(sess_gap)

fig, ax = panel_fig(1.4, 1.4)
lim = [0, np.nanpercentile(np.r_[act, nad], 98) * 1.05]
ax.scatter(nad, act, s=12, c='k', alpha=0.5, edgecolors='none')
ax.plot(lim, lim, color='gray', lw=0.8, ls='--', label='no adaptation')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
ax.set_xlabel('exact no-adapt crossing (s)')
ax.set_ylabel('actual crossing (s)')
ax.set_title(f'comp {_stars(p)} (median gap {np.median(sess_gap):+.2f}s, '
             f'n={len(sess_gap)})', fontsize=8)
ax.legend(frameon=False, fontsize=6, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
save_panel('cn_drive_raw_compensation')
print(f"actual={np.median(act):.2f}s  exact no-adapt={np.median(nad):.2f}s  "
      f"median per-session gap (noadapt-actual)={np.median(sess_gap):+.2f}s  p={p:.4f}")
print(f"  (points BELOW the diagonal = faster than no-adapt = real compensation)")
