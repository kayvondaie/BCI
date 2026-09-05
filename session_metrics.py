#%% ============================================================================
# session_metrics.py
#
# Compile per-session per-trial metrics into a single workspace for by-hand
# exploration. Borrows just what's needed from threshold_analysis2.py
# (per-session loader) — no expected-RT replay, no aggregation, no figures.
#
# After running CELL 1 you'll have, in workspace:
#
#   N_TRIALS_MAX                 (int, default 150)
#   session_keys                 (list of (mouse, session) tuples)
#   trials_per_session           (int array, length N_sessions)
#
#   Per-trial scalars as (N_TRIALS_MAX x N_sessions), NaN-padded:
#     hit_mat            1 = hit, 0 = miss, NaN = no trial
#     rt_mat             reward time, seconds from trial start (NaN for miss)
#     tc_mat             threshold-crossing time, seconds from trial start
#     thr_lower_mat      lower threshold for that trial
#     thr_upper_mat      upper threshold for that trial
#     cn_mean_mat        mean roi_csv CN over active window
#                          (trial-start -> threshold crossing)
#     cn_peak_mat        99th percentile of roi_csv CN over active window
#     cn_tuning_mat      mean Suite2P CN (post-trial-start late window)
#                          minus pre-trial baseline
#     cursor_speed_mat   mean cursor speed under each session's HARDEST thresholds
#
# *** TIME CONVENTION ***
# F[:, :, i] and the per-trial roi_csv chunks both START AT t = -PRE_TRIAL_S
# (default -2 s) relative to the trial-start go cue. So to map a behavioral
# timestamp (seconds from trial start, e.g. rt_mat[i, j], tc_mat[i, j]) to a
# frame index inside F[:, :, j] or cn_fluor_list[j][i]:
#     frame = int((t + PRE_TRIAL_S) / dt_si_list[j])
#
#   Per-session lists (indexed by session order, parallel to session_keys):
#     F_cn_list          Suite2P per-trial CN trace, shape (frames, trials)
#     cn_fluor_list      list-of-arrays: roi_csv per-trial CN trace (variable len)
#     cn_fluor_stp_list  list of ints: frame index of reward per trial
#     roi_csv_list       raw roi_csv arrays (unwrapped, interpolated)
#     bci_thresholds_list   raw BCI_thresholds arrays
#     switches_list      np.array of epoch boundary trial indices (incl 0)
#     dt_si_list         frame interval, seconds
#     cn_idx_list        CN index into F (Suite2P)
#     cn_csv_idx_list    CN column index into roi_csv (live)
#     frames_per_file_list   per-trial frame counts
#
# Also imported into workspace:
#     transfer_fun(F, lower, upper, max_speed=3.3, low_floor=0.0)
#     MOUSE_LOW_FLOOR, low_floor_for(mouse)
#
# ============================================================================

#%% ============================================================================
# CELL 0: Setup
# ============================================================================
import os, sys, pickle
import numpy as np
from scipy.interpolate import interp1d

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
from BCI_data_helpers import parse_hdf5_array_string

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'svg.fonttype': 'none',
})

# Assembled-figure panel directory. Every figure below is written here as both
# .png and .svg (editable text) via save_panel(); filenames are prefixed 'sm_'
# to namespace session_metrics outputs in the shared folder.
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')
os.makedirs(PANEL_DIR, exist_ok=True)

def save_panel(name, fig=None):
    fig = plt.gcf() if fig is None else fig
    fig.savefig(os.path.join(PANEL_DIR, f'sm_{name}.png'), dpi=300)
    fig.savefig(os.path.join(PANEL_DIR, f'sm_{name}.svg'))

def panel_fig(ax_w=1.25, ax_h=1.25, left=0.7, bottom=0.5, right=0.2, top=0.25):
    """Figure with ONE axis of size (ax_w, ax_h) in inches (margins also in
    inches); no tight_layout, so the axis box stays exactly that size. The SVG
    ends up (left+ax_w+right) x (bottom+ax_h+top). Returns (fig, ax)."""
    fig_w, fig_h = left + ax_w + right, bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h])
    return fig, ax

def panel_row(n, ax_w=1.25, ax_h=1.25, gap=0.55, left=0.7, bottom=0.5,
              right=0.2, top=0.3):
    """n side-by-side axes, each (ax_w, ax_h) inches, separated by `gap` inches
    (margins in inches); no tight_layout. Returns (fig, axes list)."""
    fig_w = left + n * ax_w + (n - 1) * gap + right
    fig_h = bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_axes([(left + i * (ax_w + gap)) / fig_w, bottom / fig_h,
                          ax_w / fig_w, ax_h / fig_h]) for i in range(n)]
    return fig, axes

def panel_grid(nrows, ncols, ax_w=1.25, ax_h=1.25, wgap=0.55, hgap=0.5,
               left=0.7, bottom=0.5, right=0.2, top=0.3):
    """nrows x ncols axes, each (ax_w, ax_h) inches; no tight_layout. Returns
    (fig, axes) with axes a (nrows, ncols) object array, [r, c] top-to-bottom."""
    fig_w = left + ncols * ax_w + (ncols - 1) * wgap + right
    fig_h = bottom + nrows * ax_h + (nrows - 1) * hgap + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            b = bottom + (nrows - 1 - r) * (ax_h + hgap)
            axes[r, c] = fig.add_axes([(left + c * (ax_w + wgap)) / fig_w,
                                       b / fig_h, ax_w / fig_w, ax_h / fig_h])
    return fig, axes

from scipy.stats import wilcoxon, pearsonr, spearmanr

def _stars(p):
    return ('***' if p < 1e-3 else '**' if p < 1e-2 else
            '*' if p < 0.05 else 'n.s.')

# --- Trace smoothing + session-clustered bootstrap band ---------------------
SMOOTH_W = 5            # trials; light smoothing for the coarse-time traces
BOOT_PCT = (16, 84)     # band percentiles (~±1 SE; use (2.5, 97.5) for 95% CI)
N_BOOT_CURVE = 5000

def _smooth(y, w):
    """NaN-aware centered moving average (window w trials)."""
    yv = np.asarray(y, float)
    if w <= 1:
        return yv
    k = np.ones(w) / w
    valid = np.isfinite(yv).astype(float)
    y0 = np.where(np.isfinite(yv), yv, 0.0)
    num = np.convolve(y0, k, mode='same')
    den = np.convolve(valid, k, mode='same')
    return np.where(den > 0, num / den, np.nan)

def _boot_ci_curve(curves, smooth_w=1, pct=BOOT_PCT, n_boot=N_BOOT_CURVE,
                   seed=0):
    """Session-clustered bootstrap of a mean curve. `curves` is (n_units, n_t),
    one row per session. Each row is smoothed first, then the mean across units
    is bootstrapped by resampling units with replacement. Returns (mean, lo, hi)."""
    C = np.vstack([_smooth(curves[u], smooth_w) for u in range(curves.shape[0])])
    rng = np.random.default_rng(seed)
    U = C.shape[0]
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(C, axis=0)
        boots = np.full((n_boot, C.shape[1]), np.nan)
        for b in range(n_boot):
            boots[b] = np.nanmean(C[rng.integers(0, U, U)], axis=0)
        lo, hi = np.nanpercentile(boots, pct, axis=0)
    return mean, lo, hi

def _msem_curve(curves, smooth_w=1):
    """Smooth each unit's curve, then mean +/- SEM across units (sessions)."""
    C = np.vstack([_smooth(curves[u], smooth_w) for u in range(curves.shape[0])])
    with np.errstate(invalid='ignore'):
        n = np.sum(np.isfinite(C), axis=0)
        mean = np.nanmean(C, axis=0)
        sem = np.nanstd(C, axis=0) / np.sqrt(np.clip(n, 1, None))
    return mean, mean - sem, mean + sem

mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
_qc_fail = {
    ('BCI104', '012325'),
    ('BCI105', '012125'),
    ('BCI105', '012425'),
}

N_TRIALS_MAX = 150

# IMPORTANT: F[:, :, i] and per-trial roi_csv chunks (via frames_per_file)
# BOTH start at t = -PRE_TRIAL_S relative to trial start. Frame 0 is the
# start of a pre-trial buffer, NOT the go cue. To map a time t (in seconds
# from trial start, e.g. reward_time or threshold_crossing_time) to a frame
# index inside F[:, :, i] or fluor:
#     frame = int((t + PRE_TRIAL_S) / dt_si)
PRE_TRIAL_S = 2.0

# Per-mouse noise floor for the empirical transfer function (matches
# threshold_analysis2.py).
MOUSE_LOW_FLOOR = {
    'BCI102': 0.36, 'BCI103': 0.16, 'BCI104': 0.22,
    'BCI105': 0.17, 'BCI106': 0.24, 'BCI109': 0.24,
}
DEFAULT_LOW_FLOOR = 0.23

def low_floor_for(mouse):
    return MOUSE_LOW_FLOOR.get(mouse, DEFAULT_LOW_FLOOR)

def transfer_fun(fluorescence, lower, upper, max_speed=3.3, low_floor=0.0):
    """Linear ramp [lower, upper] -> [0, max_speed], plus noise-floor min."""
    gain = upper - lower
    if gain <= 0:
        return np.zeros_like(fluorescence)
    speed = (fluorescence - lower) / gain * max_speed
    speed = np.clip(speed, 0.0, max_speed)
    return np.maximum(speed, low_floor)

#%% ============================================================================
# CELL 1: Loop over sessions and populate workspace
# ============================================================================
list_of_dirs = session_counting.counter()

# Collect the list of (mouse, session) to load
to_load = []
for m in mice:
    inds = np.where(
        (list_of_dirs['Mouse'] == m) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]
    for si in inds:
        s = list_of_dirs['Session'][si]
        if (m, s) not in _qc_fail:
            to_load.append((m, s))

n_sess_attempt = len(to_load)
print(f"Will attempt to load {n_sess_attempt} sessions")

# Per-session storage
session_keys = []
trials_per_session_list = []
F_cn_list = []
cn_fluor_list = []
cn_fluor_stp_list = []
roi_csv_list = []
bci_thresholds_list = []
switches_list = []
dt_si_list = []
cn_idx_list = []
cn_csv_idx_list = []
frames_per_file_list = []
mouse_arr_list = []

# Holders for per-trial scalar arrays (grow then stack)
hit_rows, rt_rows, tc_rows = [], [], []
thr_l_rows, thr_u_rows = [], []
cn_mean_rows, cn_peak_rows, cn_tuning_rows = [], [], []
cursor_speed_rows = []

load_errors = []

for k, (mouse, session) in enumerate(to_load):
    folder = ('//allen/aind/scratch/BCI/2p-raw/'
              + mouse + '/' + session + '/pophys/')
    try:
        data = ddct.load_hdf5(folder,
            ['F', 'conditioned_neuron', 'dt_si',
             'threshold_crossing_time', 'reward_time',
             'BCI_thresholds', 'roi_csv', 'cn_csv_index'], [])
        F = data['F']                          # (frames, neurons, trials)
        trl = F.shape[2]
        cn_idx = int(np.asarray(data['conditioned_neuron']).ravel()[0])
        dt_si = float(np.asarray(data['dt_si']).ravel()[0])
        cn_csv_idx = int(np.asarray(data['cn_csv_index']).ravel()[0])

        # Behavioral arrays (parse strings → numbers)
        rt_parsed = parse_hdf5_array_string(data['reward_time'], trl)
        rt_v = np.array(
            [x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
             for x in rt_parsed], dtype=float)
        tc_parsed = parse_hdf5_array_string(data['threshold_crossing_time'], trl)
        tc_v = np.array(
            [x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
             for x in tc_parsed], dtype=float)
        hit_v = np.isfinite(rt_v).astype(float)

        # Thresholds (forward-fill NaNs, pad to trl)
        thr_raw = np.asarray(data['BCI_thresholds'], dtype=float)
        thr_l = thr_raw[0, :].copy()
        thr_u = thr_raw[1, :].copy()
        for i in range(1, thr_u.size):
            if np.isnan(thr_u[i]):
                thr_u[i] = thr_u[i - 1]
            if np.isnan(thr_l[i]):
                thr_l[i] = thr_l[i - 1]
        if np.isnan(thr_u[0]) and np.any(np.isfinite(thr_u)):
            thr_u[0] = thr_u[np.isfinite(thr_u)][0]
        if np.isnan(thr_l[0]) and np.any(np.isfinite(thr_l)):
            thr_l[0] = thr_l[np.isfinite(thr_l)][0]
        if len(thr_u) < trl:
            thr_u = np.concatenate([thr_u, np.full(trl - len(thr_u), thr_u[-1])])
            thr_l = np.concatenate([thr_l, np.full(trl - len(thr_l), thr_l[-1])])

        # Switches: epoch boundary trial indices (0 = epoch 0 start)
        d_up = np.diff(thr_u)
        sw = np.where((d_up != 0) & np.isfinite(d_up))[0] + 1
        switches = np.concatenate(([0], sw))

        # roi_csv → continuous frame-aligned trace (mirrors threshold_analysis2 C2)
        ops = np.load(folder + 'suite2p_BCI/plane0/ops.npy',
                      allow_pickle=True).tolist()
        fpf = ops['frames_per_file']
        roi = np.copy(data['roi_csv'])
        wraps = np.where(np.diff(roi[:, 1]) < 0)[0]
        for ww in wraps:
            roi[ww + 1:, 1] = roi[ww + 1:, 1] + roi[ww, 1]
            roi[ww + 1:, 0] = roi[ww + 1:, 0] + roi[ww, 0]
        frm_ind = np.arange(1, int(np.max(roi[:, 1])) + 1)
        ifunc = interp1d(roi[:, 1], roi, axis=0, kind='linear',
                         fill_value='extrapolate')
        roi_i = ifunc(frm_ind)

        # Build per-trial CN fluorescence (live signal, roi_csv-based)
        cn_fluor_per_trial = []
        cn_fluor_stp = []
        cn_mean_per_trial = np.full(trl, np.nan)
        cn_peak_per_trial = np.full(trl, np.nan)
        cursor_speed_per_trial = np.full(trl, np.nan)
        hardest_lower = float(np.nanmax(thr_l))
        hardest_upper = float(np.nanmax(thr_u))
        floor_m = low_floor_for(mouse)

        # Pre-trial buffer in frames (fluor[0] is PRE_TRIAL_S seconds before
        # the trial-start go cue).
        pre_buffer_f = int(PRE_TRIAL_S / dt_si)

        strt = 0
        for i in range(min(trl, len(fpf))):
            ind = np.arange(strt, strt + fpf[i], dtype=int)
            ind = np.clip(ind, 0, len(roi_i) - 1)
            fluor = roi_i[ind, cn_csv_idx + 2]
            cn_fluor_per_trial.append(fluor)
            # Active window: trial start -> threshold crossing.
            # tc_v[i] is in seconds from trial start, but fluor starts
            # PRE_TRIAL_S seconds before trial start, so we add the offset.
            if hit_v[i] and np.isfinite(tc_v[i]):
                t_trial = roi_i[ind, 0] - roi_i[ind[0], 0]   # sec from fluor[0]
                target_t = tc_v[i] + PRE_TRIAL_S
                stp = min(np.searchsorted(t_trial, target_t), len(fluor))
            else:
                stp = len(fluor)
            cn_fluor_stp.append(stp)
            # Active window starts at the end of the pre-trial buffer
            active = fluor[pre_buffer_f:stp]
            if len(active) > 0:
                cn_mean_per_trial[i] = np.nanmean(active)
                cn_peak_per_trial[i] = np.nanpercentile(active, 99)
                cursor_speed_per_trial[i] = np.nanmean(
                    transfer_fun(active, hardest_lower, hardest_upper,
                                 low_floor=floor_m))
            strt += fpf[i]

        # CN tuning (Suite2P): late-frame mean minus pre-trial baseline.
        # Baseline = pre-trial buffer (frames 0..pre_buffer_f), which IS the
        # ~2 s before the go cue. Late window = (PRE_TRIAL_S + 0.4) s onward,
        # i.e. starting 0.4 s after the go cue.
        ff = F[:, cn_idx, :].astype(float).copy()
        bl_n = max(1, pre_buffer_f - 5)  # leave a few frames slack
        late_start = pre_buffer_f + int(0.4 / dt_si)
        ff -= np.nanmean(ff[:bl_n, :], axis=0, keepdims=True)
        if late_start < ff.shape[0]:
            cn_tuning_per_trial = np.nanmean(ff[late_start:, :], axis=0)
        else:
            cn_tuning_per_trial = np.full(trl, np.nan)

        # Pad / truncate to N_TRIALS_MAX
        def _pad(v):
            n = min(len(v), N_TRIALS_MAX)
            out = np.full(N_TRIALS_MAX, np.nan)
            out[:n] = v[:n]
            return out

        hit_rows.append(_pad(hit_v))
        rt_rows.append(_pad(rt_v))
        tc_rows.append(_pad(tc_v))
        thr_l_rows.append(_pad(thr_l[:trl]))
        thr_u_rows.append(_pad(thr_u[:trl]))
        cn_mean_rows.append(_pad(cn_mean_per_trial))
        cn_peak_rows.append(_pad(cn_peak_per_trial))
        cn_tuning_rows.append(_pad(cn_tuning_per_trial))
        cursor_speed_rows.append(_pad(cursor_speed_per_trial))

        session_keys.append((mouse, session))
        mouse_arr_list.append(mouse)
        trials_per_session_list.append(trl)
        F_cn_list.append(np.copy(F[:, cn_idx, :]))
        cn_fluor_list.append(cn_fluor_per_trial)
        cn_fluor_stp_list.append(cn_fluor_stp)
        roi_csv_list.append(roi_i)
        bci_thresholds_list.append(np.copy(thr_raw))
        switches_list.append(switches)
        dt_si_list.append(dt_si)
        cn_idx_list.append(cn_idx)
        cn_csv_idx_list.append(cn_csv_idx)
        frames_per_file_list.append(np.array(fpf))

        print(f"  [{k+1:2d}/{n_sess_attempt}] {mouse} {session}: "
              f"{trl} trials, CN={cn_idx}")
    except Exception as e:
        load_errors.append(((mouse, session), str(e)))
        print(f"  [{k+1:2d}/{n_sess_attempt}] {mouse} {session}: "
              f"ERROR {str(e)[:120]}")

n_sessions = len(session_keys)
print(f"\nLoaded {n_sessions} sessions; {len(load_errors)} load errors")

# Stack per-trial rows into (N_TRIALS_MAX, N_sessions) — trials on rows
hit_mat = np.column_stack(hit_rows)         if hit_rows else np.empty((N_TRIALS_MAX, 0))
rt_mat = np.column_stack(rt_rows)           if rt_rows else np.empty((N_TRIALS_MAX, 0))
tc_mat = np.column_stack(tc_rows)           if tc_rows else np.empty((N_TRIALS_MAX, 0))
thr_lower_mat = np.column_stack(thr_l_rows) if thr_l_rows else np.empty((N_TRIALS_MAX, 0))
thr_upper_mat = np.column_stack(thr_u_rows) if thr_u_rows else np.empty((N_TRIALS_MAX, 0))
cn_mean_mat = np.column_stack(cn_mean_rows) if cn_mean_rows else np.empty((N_TRIALS_MAX, 0))
cn_peak_mat = np.column_stack(cn_peak_rows) if cn_peak_rows else np.empty((N_TRIALS_MAX, 0))
cn_tuning_mat = np.column_stack(cn_tuning_rows) if cn_tuning_rows else np.empty((N_TRIALS_MAX, 0))
cursor_speed_mat = np.column_stack(cursor_speed_rows) if cursor_speed_rows else np.empty((N_TRIALS_MAX, 0))

trials_per_session = np.array(trials_per_session_list)
mouse_arr = np.array(mouse_arr_list)

print(f"\nPer-trial scalar matrices: shape ({N_TRIALS_MAX}, {n_sessions})")
print(f"  hit_mat, rt_mat, tc_mat, thr_lower_mat, thr_upper_mat,")
print(f"  cn_mean_mat, cn_peak_mat, cn_tuning_mat, cursor_speed_mat")
print(f"Per-session lists (length {n_sessions}):")
print(f"  F_cn_list, cn_fluor_list, cn_fluor_stp_list,")
print(f"  roi_csv_list, bci_thresholds_list, switches_list,")
print(f"  dt_si_list, cn_idx_list, cn_csv_idx_list, frames_per_file_list")
print(f"\nIndex into a session via:")
print(f"  i = session_keys.index(('BCI102', '020725'))")
print(f"  cn_mean_mat[:, i]   # this session's per-trial CN mean")
print(f"  roi_csv_list[i]     # this session's continuous live signal")

#%% ============================================================================
# CELL 2 (optional): save compiled workspace to a pickle for fast reload
# ============================================================================
SAVE_PATH = os.path.join(_THIS_DIR, 'meta_analysis_results',
                         'session_metrics_compiled.pkl')
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

_to_save = {
    'N_TRIALS_MAX': N_TRIALS_MAX,
    'session_keys': session_keys,
    'mouse_arr': mouse_arr,
    'trials_per_session': trials_per_session,
    'hit_mat': hit_mat, 'rt_mat': rt_mat, 'tc_mat': tc_mat,
    'thr_lower_mat': thr_lower_mat, 'thr_upper_mat': thr_upper_mat,
    'cn_mean_mat': cn_mean_mat, 'cn_peak_mat': cn_peak_mat,
    'cn_tuning_mat': cn_tuning_mat, 'cursor_speed_mat': cursor_speed_mat,
    # Per-session lists (note: roi_csv_list and F_cn_list can be large)
    'F_cn_list': F_cn_list,
    'cn_fluor_list': cn_fluor_list,
    'cn_fluor_stp_list': cn_fluor_stp_list,
    'roi_csv_list': roi_csv_list,
    'bci_thresholds_list': bci_thresholds_list,
    'switches_list': switches_list,
    'dt_si_list': dt_si_list,
    'cn_idx_list': cn_idx_list,
    'cn_csv_idx_list': cn_csv_idx_list,
    'frames_per_file_list': frames_per_file_list,
    'load_errors': load_errors,
}
with open(SAVE_PATH, 'wb') as fh:
    pickle.dump(_to_save, fh)
print(f"\nSaved {SAVE_PATH}")
#%%
A = np.full_like(cn_peak_mat, np.nan)
for i in range(cn_peak_mat.shape[1]):
    a = cn_peak_mat[:, i].astype(float)
    a = (a - np.nanmean(a[0:10])) / np.nanstd(a)   # whole-session z-score
    
    A[:, i] = a

m = np.nanmean(A, axis=1)
sm = np.convolve(m, np.ones(10)/10, mode='valid')
m = np.nanmean(A, axis=1)
n = 45
s = np.nanstd(A, axis=1) / np.sqrt(np.clip(n, 1, None))

w = 10
k = np.ones(w) / w
m_sm = np.convolve(m, k, mode='valid')
s_sm = np.convolve(s, k, mode='valid')
x = np.arange(len(m_sm)) + (w - 1) / 2   # center the smoothing window on the trial axis

fig, ax = panel_fig()
plt.fill_between(x, m_sm - s_sm, m_sm + s_sm, color='k', alpha=0.25, linewidth=0)
plt.plot(x, m_sm, 'k')
plt.axhline(0, color='gray', linewidth=0.5, linestyle=':')
plt.xlabel('Trial')
plt.ylabel('CN peak (session z, smoothed)')
plt.xlim((0,80))
plt.ylim((-.2,.4))
save_panel('cn_peakz_smoothed_vs_trial')
#%%
from scipy.stats import wilcoxon

# Per-session: late mean - early mean (whole-session z'd)
early = np.nanmean(A[0:10, :], axis=0)    # skip first 5 trials (settle)
late  = np.nanmean(A[10:80, :], axis=0)
delta = late - early

ok = np.isfinite(delta)
print(f"n = {ok.sum()}, "
      f"median Δ = {np.nanmedian(delta):.3f}, "
      f"fraction > 0 = {np.mean(delta[ok] > 0):.2f}, "
      f"Wilcoxon p = {wilcoxon(delta[ok]).pvalue:.3e}")

fig, ax = panel_fig()
plt.hist(delta[ok], bins=20, color='gray', edgecolor='white')
plt.axvline(0, color='k', linewidth=0.5, linestyle=':')
plt.axvline(np.nanmedian(delta), color='crimson', linewidth=1.5,
            label=f'median = {np.nanmedian(delta):.2f}')
plt.xlabel('CN peak (session z): late (40–80) − early (5–25)')
plt.ylabel('# sessions')
plt.legend(frameon=False)
save_panel('cn_peakz_late_minus_early_hist')
#%%
from scipy.stats import wilcoxon

# Smooth each session (NaN-aware)
w = 10
k = np.ones(w) / w
def _sm_col(col):
    valid = np.isfinite(col).astype(float)
    c0 = np.where(np.isfinite(col), col, 0.0)
    num = np.convolve(c0, k, mode='same')
    den = np.convolve(valid, k, mode='same')
    return np.where(den > 0, num / den, np.nan)

A_sm = np.column_stack([_sm_col(A[:, i]) for i in range(A.shape[1])])

# Per-session: smoothed peak, when it happens, and peak above baseline
peak_val = np.nanmax(A_sm, axis=0)
peak_trial = np.array([np.nanargmax(A_sm[:, i]) if np.any(np.isfinite(A_sm[:, i])) else -1
                       for i in range(A_sm.shape[1])])
baseline = np.nanmean(A_sm[:10, :], axis=0)
peak_minus_base = peak_val - baseline

ok = np.isfinite(peak_val) & np.isfinite(peak_minus_base)
print(f"n = {ok.sum()}")
print(f"  median peak (z):       {np.nanmedian(peak_val):.2f}")
print(f"  median peak - base:    {np.nanmedian(peak_minus_base):.2f}")
print(f"  frac sessions peak > 0.5: {np.mean(peak_val > 0.5):.2f}")
print(f"  frac peak - base > 0:  {np.mean(peak_minus_base > 0):.2f}")
print(f"  Wilcoxon p (peak-base > 0): {wilcoxon(peak_minus_base[ok]).pvalue:.2e}")
print(f"  peak trial: median = {int(np.median(peak_trial[ok]))}, "
      f"IQR = [{int(np.percentile(peak_trial[ok], 25))}, "
      f"{int(np.percentile(peak_trial[ok], 75))}]")

fig, axes = panel_row(3)
axes[0].hist(peak_val[ok], bins=15, color='gray', edgecolor='white')
axes[0].axvline(0, color='k', lw=0.5, ls=':')
axes[0].set_xlabel('Per-session peak (smoothed z)')
axes[0].set_ylabel('# sessions')
axes[1].hist(peak_minus_base[ok], bins=15, color='gray', edgecolor='white')
axes[1].axvline(0, color='k', lw=0.5, ls=':')
axes[1].set_xlabel('Peak − early baseline (z)')
axes[2].hist(peak_trial[ok], bins=15, color='gray', edgecolor='white')
axes[2].set_xlabel('Trial of peak')
save_panel('cn_peak_smoothed_peakstats')
#%%
PEAK_WIN = 5            # ± trials around per-session peak
EARLY_WIN = (0, 10)     # trials 0-9

# F_cn_list contains the Suite2P CN trace per session, shape (frames, trials)
n_sess = A.shape[1]
n_frames_min = min(F.shape[0] for F in F_cn_list)

early_avg = np.full((n_frames_min, n_sess), np.nan)
peak_avg  = np.full((n_frames_min, n_sess), np.nan)

for i in range(n_sess):
    F_cn = F_cn_list[i][:n_frames_min, :]
    n_tr = F_cn.shape[1]
    pk = int(peak_trial[i])

    early_avg[:, i] = np.nanmean(F_cn[:, EARLY_WIN[0]:EARLY_WIN[1]], axis=1)
    lo = max(0, pk - PEAK_WIN)
    hi = min(n_tr, pk + PEAK_WIN + 1)
    if hi > lo:
        peak_avg[:, i] = np.nanmean(F_cn[:, lo:hi], axis=1)

# Subtract per-session pre-trial baseline (first 20 frames ≈ pre-go-cue)
def _baseline_subtract(M):
    return M - np.nanmean(M[:20, :], axis=0, keepdims=True)

early_bs = _baseline_subtract(early_avg)
peak_bs  = _baseline_subtract(peak_avg)

def _ms(M):
    n = np.sum(np.isfinite(M), axis=1)
    m = np.nanmean(M, axis=1)
    s = np.nanstd(M, axis=1) / np.sqrt(np.clip(n, 1, None))
    return m, s

me, se = _ms(early_bs)
mp, sp = _ms(peak_bs)

dt = float(np.median(dt_si_list))
t = np.arange(n_frames_min) * dt

fig, ax = panel_fig()
plt.fill_between(t, me - se, me + se, color='gray',    alpha=0.3, linewidth=0)
plt.plot(t, me, 'gray',    linewidth=1.4, label='Early trials (0–9)')
plt.fill_between(t, mp - sp, mp + sp, color='crimson', alpha=0.3, linewidth=0)
plt.plot(t, mp, 'crimson', linewidth=1.4, label=f'Peak trials (±{PEAK_WIN})')
plt.axhline(0, color='gray', linewidth=0.5, linestyle=':')
plt.xlabel('Time from trial start (s)')
plt.ylabel('CN F (baseline-subtracted)')
plt.legend(frameon=False)
save_panel('cn_trace_trialstart_early_vs_peak')
#%%
RTA_PRE  = 3.0   # seconds before reward
RTA_POST = 1.5   # seconds after reward
PEAK_WIN = 10     # ± trials around per-session peak
EARLY_WIN = 10   # first N trials

# Baseline subtraction strategy:
#   'per_trial'   — subtract each individual trial's own pre-reward baseline
#                   BEFORE averaging within session. Removes within-trial
#                   baseline drift (good if baseline falls as transients grow).
#   'per_session' — subtract the per-session AVERAGED trace's baseline
#                   AFTER averaging trials. (Original behavior.)
#   'none'        — no baseline subtraction.
BASELINE_MODE = 'per_trial'     # 'per_trial' | 'per_session' | 'none'
BASELINE_S    = 0.5             # seconds at the start of the pre-reward window
                                # to use as the baseline

# Common time axis: use median dt_si across sessions (they're nearly identical)
dt_med = float(np.median(dt_si_list))
pre_f  = int(RTA_PRE  / dt_med)
post_f = int(RTA_POST / dt_med)
n_fr   = pre_f + post_f
t_rta  = (np.arange(n_fr) - pre_f) * dt_med   # axis: time from reward
bl_n   = max(1, int(BASELINE_S / dt_med))     # # baseline frames

PRE_TRIAL_S = 2.0   # F[0, :, t] is 2 s before trial start (the buffer)

def _align_one_session(F_cn, rt_v, dt_i, trial_idx, n_frames_total):
    pre_i  = int(RTA_PRE  / dt_i)
    post_i = int(RTA_POST / dt_i)
    pre_trial_f = int(PRE_TRIAL_S / dt_i)   # ← number of frames in the buffer
    traces = []
    for t in trial_idx:
        if t < 0 or t >= F_cn.shape[1] or not np.isfinite(rt_v[t]):
            continue
        # rt_v[t] is seconds from trial start, but F starts -PRE_TRIAL_S
        # earlier, so add the buffer before converting to a frame index.
        rf = int(rt_v[t] / dt_i) + pre_trial_f
        lo, hi = rf - pre_i, rf + post_i
        if lo < 0 or hi > F_cn.shape[0]:
            continue
        tr = F_cn[lo:hi, t].astype(float)
        if len(tr) != n_frames_total:
            tr = np.interp(np.linspace(0, 1, n_frames_total),
                           np.linspace(0, 1, len(tr)), tr)
        if BASELINE_MODE == 'per_trial':
            tr = tr - np.nanmean(tr[:bl_n])
        traces.append(tr)
    return (np.nanmean(np.column_stack(traces), axis=1)
            if traces else np.full(n_frames_total, np.nan))

n_sess = A.shape[1]
early_rta = np.full((n_fr, n_sess), np.nan)
peak_rta  = np.full((n_fr, n_sess), np.nan)

for i in range(n_sess):
    F_cn = F_cn_list[i]
    n_tr = F_cn.shape[1]
    rt_i = rt_mat[:n_tr, i]
    dt_i = dt_si_list[i]
    pk = int(peak_trial[i])

    early_idx = np.arange(min(EARLY_WIN, n_tr))
    peak_idx  = np.arange(max(0, pk - PEAK_WIN), min(n_tr, pk + PEAK_WIN + 1))

    early_rta[:, i] = _align_one_session(F_cn, rt_i, dt_i, early_idx, n_fr)
    peak_rta[:, i]  = _align_one_session(F_cn, rt_i, dt_i, peak_idx,  n_fr)

# Per-session baseline subtraction only if we didn't already do per-trial.
# (per_trial already starts each trial at ~0, so the session average does too.)
if BASELINE_MODE == 'per_session':
    early_bs = early_rta - np.nanmean(early_rta[:bl_n, :], axis=0, keepdims=True)
    peak_bs  = peak_rta  - np.nanmean(peak_rta[:bl_n,  :], axis=0, keepdims=True)
else:
    early_bs = early_rta
    peak_bs  = peak_rta

def _ms(M):
    n = np.sum(np.isfinite(M), axis=1)
    m = np.nanmean(M, axis=1)
    s = np.nanstd(M, axis=1) / np.sqrt(np.clip(n, 1, None))
    return m, s

me, se = _ms(early_bs)
mp, sp = _ms(peak_bs)

# Axis box sized explicitly in inches (tweak AX_W, AX_H); no tight_layout.
AX_W, AX_H = 1.25, 1.25
_L, _B, _R, _T = 0.7, 0.5, 0.2, 0.25       # margins (inches)
fig_w, fig_h = _L + AX_W + _R, _B + AX_H + _T
fig = plt.figure(figsize=(fig_w, fig_h))
ax = fig.add_axes([_L / fig_w, _B / fig_h, AX_W / fig_w, AX_H / fig_h])
ax.fill_between(t_rta, me - se, me + se, color='gray',    alpha=0.3, linewidth=0)
ax.plot(t_rta, me, 'gray',    linewidth=1.4, label='Early trials (0–9)')
ax.fill_between(t_rta, mp - sp, mp + sp, color='crimson', alpha=0.3, linewidth=0)
ax.plot(t_rta, mp, 'crimson', linewidth=1.4, label=f'Peak trials (±{PEAK_WIN})')
ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Time from reward (s)')
ax.set_ylabel(f'CN F (baseline mode = {BASELINE_MODE})')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_panel('cn_reward_aligned_early_vs_peak')
#%%
# Z-score each session's traces by the session's own pooled std (computed
# across both groups, on the baseline-subtracted traces). Same divisor for
# early and peak within a session, so the relative amplitude is preserved.
n_sess = early_bs.shape[1]
sess_std = np.full(n_sess, np.nan)
for i in range(n_sess):
    pooled = np.concatenate([early_bs[:, i], peak_bs[:, i]])
    pooled = pooled[np.isfinite(pooled)]
    if len(pooled) > 5 and pooled.std() > 0:
        sess_std[i] = pooled.std()

early_z = early_bs / sess_std[np.newaxis, :]
peak_z  = peak_bs  / sess_std[np.newaxis, :]

me_z, se_z = _ms(early_z)
mp_z, sp_z = _ms(peak_z)

fig, ax = panel_fig()
plt.fill_between(t_rta, me_z - se_z, me_z + se_z, color='gray',    alpha=0.3, linewidth=0)
plt.plot(t_rta, me_z, 'gray',    linewidth=1.4, label='Early trials (0–9)')
plt.fill_between(t_rta, mp_z - sp_z, mp_z + sp_z, color='crimson', alpha=0.3, linewidth=0)
plt.plot(t_rta, mp_z, 'crimson', linewidth=1.4, label=f'Peak trials (±{PEAK_WIN})')
plt.axvline(0, color='k', linewidth=0.5, linestyle='--')
plt.axhline(0, color='gray', linewidth=0.5, linestyle=':')
plt.xlabel('Time from reward (s)')
plt.ylabel('CN F (baseline-subtracted, session-z)')
plt.legend(frameon=False)
save_panel('cn_reward_aligned_early_vs_peak_sessionz')
#%%
RTA_PRE  = 2.0   # seconds before reward
RTA_POST = 2.0   # seconds after reward
PRE_TRIAL_S = 2.0   # F[0] is at -2s from trial start

# Common time axis (use median dt_si — sessions are nearly identical)
dt_med = float(np.median(dt_si_list))
pre_f  = int(RTA_PRE  / dt_med)
post_f = int(RTA_POST / dt_med)
n_fr   = pre_f + post_f
t_rta  = (np.arange(n_fr) - pre_f) * dt_med   # time from reward

# rta_list[i] = (n_frames, n_hits) — one column per rewarded trial.
# rta_trial_idx_list[i] = trial indices (into the session's 0..n_trials-1)
#   that contributed columns; useful for cross-referencing with cn_peak_mat,
#   rt_mat, switches_list[i], etc.
rta_list = []
rta_trial_idx_list = []

for s in range(len(F_cn_list)):
    F_cn = F_cn_list[s]
    n_tr = F_cn.shape[1]
    rt_s = rt_mat[:n_tr, s]
    dt_s = dt_si_list[s]
    pre_trial_f_s = int(PRE_TRIAL_S / dt_s)

    cols = []
    trial_idx = []
    for t in range(n_tr):
        if not np.isfinite(rt_s[t]):
            continue
        rf = int(rt_s[t] / dt_s) + pre_trial_f_s
        lo_s = rf - int(RTA_PRE  / dt_s)
        hi_s = rf + int(RTA_POST / dt_s)
        if lo_s < 0 or hi_s > F_cn.shape[0]:
            continue
        tr = F_cn[lo_s:hi_s, t]
        # Resample to the common length if this session's dt differs
        if len(tr) != n_fr:
            tr = np.interp(np.linspace(0, 1, n_fr),
                           np.linspace(0, 1, len(tr)), tr)
        cols.append(tr)
        trial_idx.append(t)

    if cols:
        rta_list.append(np.column_stack(cols))            # (n_fr, n_hits)
        rta_trial_idx_list.append(np.array(trial_idx))
    else:
        rta_list.append(np.full((n_fr, 0), np.nan))
        rta_trial_idx_list.append(np.array([], dtype=int))

print(f"Built RTA list for {len(rta_list)} sessions.")
print(f"  per-session #rewarded trials: "
      f"median={int(np.median([m.shape[1] for m in rta_list]))}, "
      f"range=[{min(m.shape[1] for m in rta_list)}, "
      f"{max(m.shape[1] for m in rta_list)}]")
print(f"  time axis: {n_fr} frames, dt={dt_med:.4f}s, "
      f"t_rta = [{t_rta[0]:.2f}, {t_rta[-1]:.2f}]s")

# Trial-by-trial heatmap for one session
i = session_keys.index(('BCI105', '020425'))
plt.imshow(rta_list[i].T, aspect='auto', extent=[t_rta[0], t_rta[-1], 0, rta_list[i].shape[1]])

# Per-trial reward-window AUC, sorted by trial number
auc = np.nanmean(rta_list[i][pre_f - int(1/dt_med):pre_f, :], axis=0)
plt.plot(rta_trial_idx_list[i], auc, '.')

# Mean response by epoch, within one session
sw = switches_list[i]
for ei in range(len(sw)):
    a = sw[ei]; b = sw[ei+1] if ei+1 < len(sw) else 999
    mask = (rta_trial_idx_list[i] >= a) & (rta_trial_idx_list[i] < b)
    plt.plot(t_rta, np.nanmean(rta_list[i][:, mask], axis=1), label=f'epoch {ei}')

#%% ============================================================================
# Pre-reward AUC aligned to threshold changes
# ============================================================================
# For each threshold-increase transition, take the per-trial pre-reward AUC
# in a window around the switch and average across transitions.
# ============================================================================
AUC_WIN_S = 2.0     # seconds before reward to integrate over
PRE_TR  = 15        # trials before switch
POST_TR = 30        # trials after switch
INC_ONLY = True     # restrict to threshold-INCREASE transitions
# NaN-mask trials from neighboring epochs vs. include all in-window trials.
# Default off: constant composition across lags (avoids the survivorship bias
# where long/worse epochs dominate at long post-switch lags). Double-counting
# of shared trials is accepted.
MASK_NEIGHBOR_SWITCHES = False

# Baseline corrections (independent — can use either, both, or neither)
PER_TRIAL_BASELINE = False   # subtract each trial's own pre-ramp baseline
                            # (first BASELINE_S of the reward-aligned window)
                            # before computing AUC. Removes within-trial drift.
PER_SWITCH_BASELINE = True  # subtract pre-switch mean AUC from each aligned
                            # row before averaging. Shows CHANGE in AUC at the
                            # switch, not absolute AUC.

# Per-session per-trial pre-reward AUC, NaN for miss trials
auc_per_session = []
for i in range(len(rta_list)):
    dt_i = dt_si_list[i]
    pre_f_i = int(RTA_PRE / dt_i)
    win_n = int(AUC_WIN_S / dt_i)
    bl_n_i = max(1, int(BASELINE_S / dt_i))
    rta_i = rta_list[i]
    # AUC over the last AUC_WIN_S seconds before reward
    auc_hits = np.nanmean(rta_i[pre_f_i - win_n:pre_f_i, :], axis=0)
    if PER_TRIAL_BASELINE:
        # Subtract each trial's first BASELINE_S of the reward-aligned window
        bl_hits = np.nanmean(rta_i[:bl_n_i, :], axis=0)
        auc_hits = auc_hits - bl_hits
    # Map back to full-trial-length array (NaN for misses / dropped trials)
    n_tr = F_cn_list[i].shape[1]
    auc_full = np.full(n_tr, np.nan)
    auc_full[rta_trial_idx_list[i]] = auc_hits
    auc_per_session.append(auc_full)

# Align each switch's AUC window
aligned = []       # (n_transitions, PRE_TR + POST_TR) — per-switch baseline
alpha_arr = []     # INCREMENTAL gain ratio: new_gain / prev_epoch_gain
                   #   > 1 when this single step widened the gain.
alpha_cum_arr = [] # CUMULATIVE gain ratio:  new_gain / epoch_0_gain
                   #   absolute task widening since session start.
sess_idx_arr = []  # session index per transition (for ep0-relative response)
switch_trial_arr = []   # trial index of the switch (for ep0-relative response)
for i, sw in enumerate(switches_list):
    thr_l_i = bci_thresholds_list[i][0, :].astype(float)
    thr_u_i = bci_thresholds_list[i][1, :].astype(float)
    # First element may be NaN — fill it with the first finite value.
    if np.isnan(thr_u_i[0]) and np.any(np.isfinite(thr_u_i)):
        thr_u_i[0] = thr_u_i[np.isfinite(thr_u_i)][0]
    if np.isnan(thr_l_i[0]) and np.any(np.isfinite(thr_l_i)):
        thr_l_i[0] = thr_l_i[np.isfinite(thr_l_i)][0]
    # forward-fill remaining
    for k in range(1, len(thr_u_i)):
        if np.isnan(thr_u_i[k]):
            thr_u_i[k] = thr_u_i[k - 1]
        if np.isnan(thr_l_i[k]):
            thr_l_i[k] = thr_l_i[k - 1]
    gain_ep0 = thr_u_i[0] - thr_l_i[0]   # session's starting gain
    auc_i = auc_per_session[i]
    n_tr = len(auc_i)
    for s_idx in range(1, len(sw)):
        s = int(sw[s_idx])
        if s >= n_tr:
            continue
        gain_old = thr_u_i[s - 1] - thr_l_i[s - 1]
        gain_new = thr_u_i[s]     - thr_l_i[s]
        if not (np.isfinite(gain_old) and gain_old > 0 and
                np.isfinite(gain_new) and gain_new > 0):
            continue
        alpha = gain_new / gain_old        # incremental
        alpha_cum = (gain_new / gain_ep0
                    if np.isfinite(gain_ep0) and gain_ep0 > 0 else np.nan)
        if INC_ONLY and not (alpha > 1):
            continue
        # Optionally restrict to the epochs adjacent to this switch.
        if MASK_NEIGHBOR_SWITCHES:
            prev_sw = max([int(x) for x in sw if x < s], default=0)
            next_sw = min([int(x) for x in sw if x > s], default=n_tr)
        else:
            prev_sw, next_sw = 0, n_tr
        row = np.full(PRE_TR + POST_TR, np.nan)
        for k in range(PRE_TR + POST_TR):
            t = s - PRE_TR + k
            if 0 <= t < n_tr and prev_sw <= t < next_sw:
                row[k] = auc_i[t]
        if PER_SWITCH_BASELINE:
            pre_mean = np.nanmean(row[:PRE_TR])
            if np.isfinite(pre_mean):
                row = row - pre_mean
        aligned.append(row)
        alpha_arr.append(alpha)
        alpha_cum_arr.append(alpha_cum)
        sess_idx_arr.append(i)
        switch_trial_arr.append(s)

aligned = np.array(aligned)
alpha_arr = np.array(alpha_arr)
alpha_cum_arr = np.array(alpha_cum_arr)
sess_idx_arr = np.array(sess_idx_arr)
switch_trial_arr = np.array(switch_trial_arr)

# Per-session epoch-0 mean AUC (reference for cumulative-relative response)
ep0_mean_per_session = np.full(len(switches_list), np.nan)
for i in range(len(switches_list)):
    sw = switches_list[i]
    ep0_end = int(sw[1]) if len(sw) > 1 else len(auc_per_session[i])
    ep0_end = max(1, min(ep0_end, len(auc_per_session[i])))
    if ep0_end > 0:
        ep0_mean_per_session[i] = np.nanmean(auc_per_session[i][:ep0_end])
print(f"Aligned {aligned.shape[0]} transitions "
      f"({'increases only' if INC_ONLY else 'all'}) "
      f"from {len(switches_list)} sessions")
print(f"  per_trial_baseline  = {PER_TRIAL_BASELINE}")
print(f"  per_switch_baseline = {PER_SWITCH_BASELINE}")

# Aggregate transitions to a per-session curve first (session = unit of
# replication), then a session-clustered bootstrap band (+ light smoothing).
uniq_s = np.unique(sess_idx_arr)
sess_curves = np.full((len(uniq_s), PRE_TR + POST_TR), np.nan)
for j, si in enumerate(uniq_s):
    with np.errstate(invalid='ignore'):
        sess_curves[j] = np.nanmean(aligned[sess_idx_arr == si], axis=0)
x = np.arange(-PRE_TR, POST_TR)
# Transition-pooled mean +/- SEM across transitions (the gradual version):
# pooling many transitions smooths the cross-epoch leakage into a gradual rise
# rather than the choppy per-session jump. The star below stays session-level
# (honest), computed from sess_curves above.
m, lo, hi = _msem_curve(aligned, smooth_w=SMOOTH_W)

# Build a label that documents both corrections
bl_tag_parts = []
if PER_TRIAL_BASELINE:
    bl_tag_parts.append('per-trial')
if PER_SWITCH_BASELINE:
    bl_tag_parts.append('per-switch')
bl_tag = ' + '.join(bl_tag_parts) if bl_tag_parts else 'none'

fig, ax = panel_fig()
plt.fill_between(x, lo, hi, color='k', alpha=0.25, linewidth=0)
plt.plot(x, m, 'k', linewidth=1.4)
plt.axvline(0, color='r', linewidth=0.6, linestyle='--')
plt.axhline(0, color='gray', linewidth=0.5, linestyle=':')
plt.xlabel('Trials from threshold change')
plt.ylabel(f'Pre-reward AUC ({AUC_WIN_S:.1f}s window)\n'
          f'baselines: {bl_tag}')
plt.title(f'n = {aligned.shape[0]} transitions'
          f'{" (threshold increases)" if INC_ONLY else ""}')
# Per-session significance of the post-switch rise + star over the window
_pc = np.nanmean(sess_curves[:, PRE_TR + 1:PRE_TR + POST_TR], axis=1)
_pc = _pc[np.isfinite(_pc)]
_p2 = wilcoxon(_pc).pvalue
_ystar = np.nanmax(hi) * 1.08
plt.plot([1, POST_TR - 1], [_ystar, _ystar], color='k', linewidth=0.8)
plt.text(POST_TR / 2, _ystar, _stars(_p2), ha='center', va='bottom')
save_panel('prereward_auc_switch_aligned')

#%% ============================================================================
# Scatter: post-switch late-activity change vs. alpha (incremental & cumulative)
# ============================================================================
# y axis matches x semantically:
#   Left  panel (incremental): response measured RELATIVE TO PRE-SWITCH BASELINE
#                              ("did this step push the response up?")
#   Right panel (cumulative):  response measured RELATIVE TO EPOCH 0
#                              ("how much has the response grown since session start?")
# ============================================================================
from scipy.stats import pearsonr

LATE_LO, LATE_HI = 5, 20
late_cols = slice(PRE_TR + LATE_LO, PRE_TR + LATE_HI)

# Incremental view: use the per-switch-baselined `aligned`.
late_change_per_switch = np.nanmean(aligned[:, late_cols], axis=1)

# Cumulative view: late raw AUC minus session's epoch-0 mean AUC.
# Clip the late window at the next switch so it stays within this epoch.
late_change_per_ep0 = np.full(aligned.shape[0], np.nan)
for ti in range(aligned.shape[0]):
    si = sess_idx_arr[ti]
    s = switch_trial_arr[ti]
    auc_i = auc_per_session[si]
    sw_si = switches_list[si]
    next_sw = (min([int(x) for x in sw_si if x > s], default=len(auc_i))
               if MASK_NEIGHBOR_SWITCHES else len(auc_i))
    lo = s + LATE_LO
    hi = min(s + LATE_HI, len(auc_i), next_sw)
    if hi > lo and np.isfinite(ep0_mean_per_session[si]):
        late_change_per_ep0[ti] = (np.nanmean(auc_i[lo:hi])
                                   - ep0_mean_per_session[si])

def _scatter(ax, x_arr, y_arr, xlabel, ylabel):
    ok = np.isfinite(x_arr) & np.isfinite(y_arr)
    ax.scatter(x_arr[ok], y_arr[ok], s=18, c='k', alpha=0.55,
               edgecolors='none')
    if ok.sum() >= 5:
        a, b = np.polyfit(x_arr[ok], y_arr[ok], 1)
        xf = np.array([x_arr[ok].min(), x_arr[ok].max()])
        ax.plot(xf, a * xf + b, color='crimson', linewidth=1.0)
        r, p = pearsonr(x_arr[ok], y_arr[ok])
    else:
        r, p = np.nan, np.nan
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(1, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'r = {r:+.2f}, p = {p:.2e}, n = {int(ok.sum())}',
                 fontsize=9)

fig_a, axes_a = panel_row(2)
_scatter(axes_a[0], alpha_arr, late_change_per_switch,
         'INCREMENTAL alpha\n(gain_new / prev_epoch_gain)',
         f'Late AUC change vs PRE-SWITCH\n(trials {LATE_LO}–{LATE_HI})')
_scatter(axes_a[1], alpha_cum_arr, late_change_per_ep0,
         'CUMULATIVE alpha\n(gain_new / epoch_0_gain)',
         f'Late AUC change vs EPOCH 0\n(trials {LATE_LO}–{LATE_HI})')
save_panel('lateAUC_vs_alpha_incr_and_cumulative')

#%% ============================================================================
# Per-SESSION late-AUC rise vs gain change — which sessions show it, and does
# it fall off when the change is TOO big? (inverted-U test)
# ============================================================================
# Reuses the lateAUC per-transition arrays (alpha_cum_arr, late_change_per_ep0,
# sess_idx_arr). Per session, take the LARGEST cumulative gain change and the
# late-AUC change there: "at the hardest the task got this session, did the late
# tuning rise?" A single linear fit can wash out a rise-then-fall relationship.
_okt = np.isfinite(alpha_cum_arr) & np.isfinite(late_change_per_ep0)
_srow = {}
for _ti in np.where(_okt)[0]:
    _si = int(sess_idx_arr[_ti])
    _a, _y = alpha_cum_arr[_ti], late_change_per_ep0[_ti]
    if _si not in _srow or _a > _srow[_si][0]:
        _srow[_si] = (_a, _y)
_rows = sorted([(a, y, session_keys[si]) for si, (a, y) in _srow.items()],
               reverse=True)
_nup = sum(y > 0 for a, y, k in _rows)
print("\nPer-session late-AUC change at the session's LARGEST cumulative gain change:")
print(f"  {'alpha_cum':>9} {'dLateAUC':>9}  session")
for a, y, k in _rows:
    print(f"  {a:9.2f} {y:+9.3f}  {k[0]}_{k[1]}{'   UP' if y > 0 else ''}")
print(f"  --> {_nup}/{len(_rows)} sessions raised late-AUC at their hardest point")

_a_all = alpha_cum_arr[_okt]; _y_all = late_change_per_ep0[_okt]
fig_ps, ax_ps = panel_row(2)
# left: all transitions, binned by cumulative alpha (median +/- SEM per bin)
_edges = np.unique(np.nanpercentile(_a_all, [0, 20, 40, 60, 80, 100]))
_cx, _cy, _ce = [], [], []
for _lo, _hi in zip(_edges[:-1], _edges[1:]):
    _m = (_a_all >= _lo) & (_a_all <= _hi)
    if _m.sum() >= 3:
        _cx.append(np.median(_a_all[_m])); _cy.append(np.median(_y_all[_m]))
        _ce.append(np.std(_y_all[_m]) / np.sqrt(_m.sum()))
ax_ps[0].scatter(_a_all, _y_all, s=10, c='0.75', edgecolors='none')
ax_ps[0].errorbar(_cx, _cy, yerr=_ce, fmt='o-', color='crimson', ms=4, lw=1.2, capsize=2)
ax_ps[0].axhline(0, color='gray', lw=0.5, ls=':')
ax_ps[0].set_xlabel('cumulative alpha (binned)')
ax_ps[0].set_ylabel(f'Late AUC change vs epoch 0\n(trials {LATE_LO}-{LATE_HI})')
ax_ps[0].set_title('rise then fall for big changes?', fontsize=9)
# right: one dot per session (max cumulative alpha vs late change there)
_sa = np.array([a for a, y, k in _rows]); _sy = np.array([y for a, y, k in _rows])
ax_ps[1].scatter(_sa, _sy, s=18, c='k', alpha=0.6, edgecolors='none')
ax_ps[1].axhline(0, color='gray', lw=0.5, ls=':')
ax_ps[1].set_xlabel('session max cumulative alpha')
ax_ps[1].set_ylabel('late AUC change at that point')
ax_ps[1].set_title(f'{_nup}/{len(_rows)} sessions up', fontsize=9)
save_panel('lateAUC_per_session_vs_gain')

#%% ============================================================================
# Per-switch ΔCN vs ΔRT — direct neural<->behavioral coupling
# ============================================================================
# For each threshold-increase transition: ΔCN = post-window mean pre-reward AUC
# minus pre-window mean; ΔRT = post-window mean RT minus pre-window mean
# (hits-only; rt_mat is NaN for misses). Tests whether transitions with a
# bigger CN increase show a bigger RT change, without the expected-RT model.
DCN_PRE, DCN_POST = 10, 10   # pre / post window lengths (trials)

# Load epoch stats (for expected RT) and build a (mouse,session,epoch) lookup.
if 'all_epoch_stats' not in globals():
    _es_path = os.path.join(_THIS_DIR, 'meta_analysis_results',
                            'all_epoch_stats.pkl')
    with open(_es_path, 'rb') as fh:
        all_epoch_stats = pickle.load(fh)['all_epoch_stats']
_es_lookup = {(r['mouse'], r['session'], r['epoch']): r
              for r in all_epoch_stats}

dcn_per_tr = np.full(len(sess_idx_arr), np.nan)
drt_per_tr = np.full(len(sess_idx_arr), np.nan)
rt_vs_exp_per_tr = np.full(len(sess_idx_arr), np.nan)   # actual_rt − expected_rt
for ti in range(len(sess_idx_arr)):
    si = sess_idx_arr[ti]
    s = switch_trial_arr[ti]
    auc_i = auc_per_session[si]
    rt_i = rt_mat[:len(auc_i), si]
    a0, a1 = max(0, s - DCN_PRE), s
    b0, b1 = s, min(s + DCN_POST, len(auc_i))
    if a1 > a0 and b1 > b0:
        dcn_per_tr[ti] = np.nanmean(auc_i[b0:b1]) - np.nanmean(auc_i[a0:a1])
        drt_per_tr[ti] = np.nanmean(rt_i[b0:b1]) - np.nanmean(rt_i[a0:a1])
    # actual − expected RT for this epoch (epoch index = position of s in sw)
    sw_si = np.asarray(switches_list[si])
    epoch_match = np.where(sw_si == s)[0]
    if len(epoch_match) > 0:
        rec = _es_lookup.get((session_keys[si][0], session_keys[si][1],
                              int(epoch_match[0])))
        if rec is not None and np.isfinite(rec.get('expected_rt', np.nan)) \
                and np.isfinite(rec.get('actual_rt', np.nan)):
            rt_vs_exp_per_tr[ti] = rec['actual_rt'] - rec['expected_rt']

def _dcn_panel(ax, y, ylabel):
    ok = np.isfinite(dcn_per_tr) & np.isfinite(y)
    ax.scatter(dcn_per_tr[ok], y[ok], s=16, c='k', alpha=0.5, edgecolors='none')
    if ok.sum() >= 5:
        r, p = pearsonr(dcn_per_tr[ok], y[ok])
        m, b = np.polyfit(dcn_per_tr[ok], y[ok], 1)
        xf = np.array([dcn_per_tr[ok].min(), dcn_per_tr[ok].max()])
        ax.plot(xf, m * xf + b, color='crimson', linewidth=1.0)
        ax.set_title(f'r = {r:+.2f}, p = {p:.2e}, n = {int(ok.sum())}',
                     fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel(f'ΔCN (post−pre pre-reward AUC, ±{DCN_POST} trials)')
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig_d, axes_d = panel_row(2)
_dcn_panel(axes_d[0], drt_per_tr, 'ΔRT (post−pre, s; hits only)')
_dcn_panel(axes_d[1], rt_vs_exp_per_tr,
           'RT − expected (epoch, s)\n(neg = faster than no-adapt prediction)')
save_panel('dCN_vs_dRT_incremental')
print(f"ΔCN vs ΔRT: n={int((np.isfinite(dcn_per_tr)&np.isfinite(drt_per_tr)).sum())}; "
      f"ΔCN vs (RT−exp): n={int((np.isfinite(dcn_per_tr)&np.isfinite(rt_vs_exp_per_tr)).sum())}")

#%% ============================================================================
# Per-switch ΔCN vs ΔRT — referenced to EPOCH 0 (cumulative)
# ============================================================================
# Like the cumulative-alpha view: measure CN and RT in the post-switch window
# RELATIVE TO EPOCH 0 (session start) instead of to the immediate pre-switch
# window. Tests whether cumulative CN rise tracks cumulative behavioral change.

# Per-session epoch-0 means (pre-reward AUC and RT)
ep0_auc_mean = np.full(n_sess, np.nan)
ep0_rt_mean = np.full(n_sess, np.nan)
for i in range(n_sess):
    sw_i = np.asarray(switches_list[i])
    ep0_end = int(sw_i[1]) if len(sw_i) > 1 else len(auc_per_session[i])
    ep0_end = max(1, min(ep0_end, len(auc_per_session[i])))
    ep0_auc_mean[i] = np.nanmean(auc_per_session[i][:ep0_end])
    ep0_rt_mean[i] = np.nanmean(rt_mat[:ep0_end, i])

dcn_ep0 = np.full(len(sess_idx_arr), np.nan)   # post CN − epoch-0 CN
drt_ep0 = np.full(len(sess_idx_arr), np.nan)   # post RT − epoch-0 RT
for ti in range(len(sess_idx_arr)):
    si = sess_idx_arr[ti]
    s = switch_trial_arr[ti]
    auc_i = auc_per_session[si]
    rt_i = rt_mat[:len(auc_i), si]
    b0, b1 = s, min(s + DCN_POST, len(auc_i))
    if b1 > b0:
        dcn_ep0[ti] = np.nanmean(auc_i[b0:b1]) - ep0_auc_mean[si]
        drt_ep0[ti] = np.nanmean(rt_i[b0:b1]) - ep0_rt_mean[si]

def _ep0_panel(ax, y, ylabel):
    ok = np.isfinite(dcn_ep0) & np.isfinite(y)
    ax.scatter(dcn_ep0[ok], y[ok], s=16, c='k', alpha=0.5, edgecolors='none')
    if ok.sum() >= 5:
        r, p = pearsonr(dcn_ep0[ok], y[ok])
        m, b = np.polyfit(dcn_ep0[ok], y[ok], 1)
        xf = np.array([dcn_ep0[ok].min(), dcn_ep0[ok].max()])
        ax.plot(xf, m * xf + b, color='crimson', linewidth=1.0)
        ax.set_title(f'r = {r:+.2f}, p = {p:.2e}, n = {int(ok.sum())}',
                     fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('ΔCN vs epoch 0 (post pre-reward AUC − ep0 mean)')
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig_e, axes_e = panel_row(2)
_ep0_panel(axes_e[0], drt_ep0, 'ΔRT vs epoch 0 (post − ep0 mean, s)')
_ep0_panel(axes_e[1], rt_vs_exp_per_tr, 'RT − expected (epoch, s)')
save_panel('dCN_vs_dRT_vs_epoch0')
print(f"ΔCN(vs ep0) vs ΔRT(vs ep0): "
      f"n={int((np.isfinite(dcn_ep0)&np.isfinite(drt_ep0)).sum())}")

#%% ============================================================================
# Same coupling, but ΔCN over the FULL trial-start→reward window
# ============================================================================
# Alternative CN measure: mean F_cn over [trial start, reward] (the whole
# active period), baseline-subtracted by the pre-trial buffer — instead of
# just the pre-reward AUC. Lets us compare which window carries the coupling.

# Per-trial trial-start->reward mean CN (Suite2P), pre-trial-buffer baseline.
cn_full_per_session = []
for i in range(n_sess):
    F_cn = F_cn_list[i]
    n_fr_s, n_tr = F_cn.shape
    dt_i = dt_si_list[i]
    rt_i = rt_mat[:n_tr, i]
    pre_trial_f = int(PRE_TRIAL_S / dt_i)
    cn_full = np.full(n_tr, np.nan)
    for t in range(n_tr):
        if not np.isfinite(rt_i[t]):
            continue
        rf = min(int(rt_i[t] / dt_i) + pre_trial_f, n_fr_s)
        if rf > pre_trial_f:
            bl = np.nanmean(F_cn[:pre_trial_f, t])
            cn_full[t] = np.nanmean(F_cn[pre_trial_f:rf, t]) - bl
    cn_full_per_session.append(cn_full)

# Epoch-0 mean of the full-window CN, and post − epoch0 per transition
ep0_cnfull_mean = np.array([
    np.nanmean(cn_full_per_session[i][:(int(np.asarray(switches_list[i])[1])
               if len(switches_list[i]) > 1 else len(cn_full_per_session[i]))])
    for i in range(n_sess)])

dcn_full_ep0 = np.full(len(sess_idx_arr), np.nan)
for ti in range(len(sess_idx_arr)):
    si = sess_idx_arr[ti]
    s = switch_trial_arr[ti]
    cnf = cn_full_per_session[si]
    b0, b1 = s, min(s + DCN_POST, len(cnf))
    if b1 > b0:
        dcn_full_ep0[ti] = np.nanmean(cnf[b0:b1]) - ep0_cnfull_mean[si]

def _full_panel(ax, y, ylabel):
    ok = np.isfinite(dcn_full_ep0) & np.isfinite(y)
    ax.scatter(dcn_full_ep0[ok], y[ok], s=16, c='k', alpha=0.5, edgecolors='none')
    if ok.sum() >= 5:
        r, p = pearsonr(dcn_full_ep0[ok], y[ok])
        m, b = np.polyfit(dcn_full_ep0[ok], y[ok], 1)
        xf = np.array([dcn_full_ep0[ok].min(), dcn_full_ep0[ok].max()])
        ax.plot(xf, m * xf + b, color='crimson', linewidth=1.0)
        ax.set_title(f'r = {r:+.2f}, p = {p:.2e}, n = {int(ok.sum())}',
                     fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('ΔCN vs epoch 0 (trial-start→reward mean − ep0)')
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig_f, axes_f = panel_row(2)
_full_panel(axes_f[0], drt_ep0, 'ΔRT vs epoch 0 (post − ep0 mean, s)')
_full_panel(axes_f[1], rt_vs_exp_per_tr, 'RT − expected (epoch, s)')
save_panel('dCN_fullwindow_vs_epoch0')
print(f"ΔCN_full(vs ep0): "
      f"n={int((np.isfinite(dcn_full_ep0)&np.isfinite(rt_vs_exp_per_tr)).sum())}")

#%% ============================================================================
# RT ratio vs alpha — non-circular behavioral dose-response
# ============================================================================
# x = alpha = gain_new / gain_old  (imposed perturbation size)
# y = actual_rt / prev_rt          (observed RT ratio across the switch)
# No-adaptation prediction is the y = x diagonal (expected_rt/prev_rt ≈ alpha).
# Points below diagonal = compensation. Slope of y on x < 1 = sub-proportional
# RT growth = compensation that scales with perturbation. No expected_rt in y,
# so not circular with alpha.
rt_ratio_per_tr = np.full(len(sess_idx_arr), np.nan)
for ti in range(len(sess_idx_arr)):
    si = sess_idx_arr[ti]
    s = switch_trial_arr[ti]
    m_, sess_ = session_keys[si]
    sw_si = np.asarray(switches_list[si])
    em = np.where(sw_si == s)[0]
    if len(em) == 0 or em[0] == 0:
        continue
    ep = int(em[0])
    rec_cur = _es_lookup.get((m_, sess_, ep))
    rec_prev = _es_lookup.get((m_, sess_, ep - 1))
    if (rec_cur is not None and rec_prev is not None
            and np.isfinite(rec_cur.get('actual_rt', np.nan))
            and np.isfinite(rec_prev.get('actual_rt', np.nan))
            and rec_prev['actual_rt'] > 0):
        rt_ratio_per_tr[ti] = rec_cur['actual_rt'] / rec_prev['actual_rt']

ok = np.isfinite(alpha_arr) & np.isfinite(rt_ratio_per_tr)
a_ok = alpha_arr[ok]
y_ok = rt_ratio_per_tr[ok]
sess_ok = sess_idx_arr[ok]
frac_below = np.mean(y_ok < a_ok)

# Anchored fit through (1, 1): no gain change (alpha=1) -> no RT change (ratio=1).
# Regress (y - 1) on (alpha - 1) with NO intercept. beta is the slope through
# that physically-required point; compensated fraction = 1 - beta, with no
# offset ambiguity. (beta=1 no adaptation, beta=0 full compensation.)
def _anchored_beta(x, y):
    xs = x - 1.0
    denom = np.sum(xs * xs)
    return np.sum(xs * (y - 1.0)) / denom if denom > 0 else np.nan

beta = _anchored_beta(a_ok, y_ok)
comp_frac = 1.0 - beta

# Free-intercept fit, kept only to report the baseline drift at alpha=1.
m_free, b_free = np.polyfit(a_ok, y_ok, 1)
y_at_1 = m_free + b_free

# --- Session-clustered bootstrap on the anchored beta ---
uniq_sess = np.unique(sess_ok)
rng = np.random.default_rng(0)
N_BOOT = 5000
boot_beta = []
for _ in range(N_BOOT):
    chosen = rng.choice(uniq_sess, size=len(uniq_sess), replace=True)
    xa = np.concatenate([a_ok[sess_ok == cs] for cs in chosen])
    ya = np.concatenate([y_ok[sess_ok == cs] for cs in chosen])
    bb = _anchored_beta(xa, ya)
    if np.isfinite(bb):
        boot_beta.append(bb)
boot_beta = np.array(boot_beta)
ci_lo, ci_hi = np.percentile(boot_beta, [2.5, 97.5])
p_vs1 = min(1.0, 2 * min(np.mean(boot_beta >= 1.0), np.mean(boot_beta <= 1.0)))
comp_lo, comp_hi = 1.0 - ci_hi, 1.0 - ci_lo

fig_g, ax_g = panel_fig()
ax_g.scatter(a_ok, y_ok, s=16, c='k', alpha=0.5, edgecolors='none')
lim = [0.9, a_ok.max() * 1.05]
ax_g.plot(lim, lim, color='gray', linewidth=0.8, linestyle='--',
          label='no adaptation (y = x)')
ax_g.axhline(1.0, color='cornflowerblue', linewidth=0.8, linestyle=':',
             label='full compensation (y = 1)')
# Anchored fit line: passes through (1, 1) by construction
xf = np.array([1.0, a_ok.max()])
ax_g.plot(xf, 1.0 + beta * (xf - 1.0), color='crimson', linewidth=1.0)
ax_g.set_title(
    f'compensated {comp_frac:.0%}  (95% CI [{comp_lo:.0%}, {comp_hi:.0%}])\n'
    f'β = {beta:.2f} [{ci_lo:.2f}, {ci_hi:.2f}] (through (1,1)), '
    f'vs 1: p={p_vs1:.3f}; {frac_below:.0%} below diag; n={int(ok.sum())}',
    fontsize=8)
ax_g.set_xlabel('alpha = gain_new / gain_old')
ax_g.set_ylabel('RT ratio (actual_rt / prev_rt)')
ax_g.legend(frameon=False, fontsize=7, loc='upper left')
ax_g.spines['top'].set_visible(False)
ax_g.spines['right'].set_visible(False)
save_panel('rt_ratio_vs_alpha_dose_response')
print(f"RT ratio vs alpha (anchored at (1,1)): beta={beta:.3f} "
      f"[{ci_lo:.3f}, {ci_hi:.3f}], compensated={comp_frac:.3f} "
      f"[{comp_lo:.3f}, {comp_hi:.3f}], p(vs1)={p_vs1:.4f}, n={int(ok.sum())}")
print(f"  (free-fit baseline check: RT ratio at alpha=1 = {y_at_1:.3f}; "
      f">1 = epoch-to-epoch drift), n_sessions={len(uniq_sess)}")

#%% ============================================================================
# Expected ΔRT vs actual ΔRT — additive compensation (seconds)
# ============================================================================
# x = expected ΔRT (predicted slowdown vs epoch 0), y = actual ΔRT (observed).
# Both from all_epoch_stats (relative to epoch 0). Diagonal y=x = no adaptation;
# y=0 = full compensation (RT unchanged from ep0). slope = 1 − compensation.
# Caveat: both deltas share −ep0_rt (session-level shared term), so mildly less
# clean than RT-ratio-vs-alpha (whose x carries no RT at all).
exp_drt_tr = np.full(len(sess_idx_arr), np.nan)
act_drt_tr = np.full(len(sess_idx_arr), np.nan)
for ti in range(len(sess_idx_arr)):
    si = sess_idx_arr[ti]
    s = switch_trial_arr[ti]
    m_, sess_ = session_keys[si]
    sw_si = np.asarray(switches_list[si])
    em = np.where(sw_si == s)[0]
    if len(em) == 0:
        continue
    rec = _es_lookup.get((m_, sess_, int(em[0])))
    if rec is not None:
        exp_drt_tr[ti] = rec.get('expected_delta_rt', np.nan)
        act_drt_tr[ti] = rec.get('actual_delta_rt', np.nan)

ok2 = np.isfinite(exp_drt_tr) & np.isfinite(act_drt_tr)
x2, y2, sess2 = exp_drt_tr[ok2], act_drt_tr[ok2], sess_idx_arr[ok2]
m2, b2 = np.polyfit(x2, y2, 1)
frac_below2 = np.mean(y2 < x2)

uniq2 = np.unique(sess2)
rng2 = np.random.default_rng(0)
boot2 = []
for _ in range(5000):
    chosen = rng2.choice(uniq2, size=len(uniq2), replace=True)
    xb = np.concatenate([x2[sess2 == cs] for cs in chosen])
    yb = np.concatenate([y2[sess2 == cs] for cs in chosen])
    if len(xb) >= 3 and np.std(xb) > 1e-9:
        boot2.append(np.polyfit(xb, yb, 1)[0])
boot2 = np.array(boot2)
ci2_lo, ci2_hi = np.percentile(boot2, [2.5, 97.5])
p2_vs1 = min(1.0, 2 * min(np.mean(boot2 >= 1.0), np.mean(boot2 <= 1.0)))
comp2 = 1.0 - m2

fig_h, ax_h = panel_fig()
ax_h.scatter(x2, y2, s=16, c='k', alpha=0.5, edgecolors='none')
lim2 = [min(x2.min(), y2.min()), max(x2.max(), y2.max())]
ax_h.plot(lim2, lim2, color='gray', linewidth=0.8, linestyle='--',
          label='no adaptation (y = x)')
ax_h.axhline(0, color='cornflowerblue', linewidth=0.8, linestyle=':',
             label='full compensation (y = 0)')
xf2 = np.array([x2.min(), x2.max()])
ax_h.plot(xf2, m2 * xf2 + b2, color='crimson', linewidth=1.0)
ax_h.set_title(
    f'compensated {comp2:.0%}  (95% CI [{1-ci2_hi:.0%}, {1-ci2_lo:.0%}])\n'
    f'slope = {m2:.2f} [{ci2_lo:.2f}, {ci2_hi:.2f}], vs 1: p={p2_vs1:.3f}; '
    f'{frac_below2:.0%} below diag; n={int(ok2.sum())}', fontsize=8)
ax_h.set_xlabel('Expected ΔRT vs epoch 0 (s)')
ax_h.set_ylabel('Actual ΔRT vs epoch 0 (s)')
ax_h.legend(frameon=False, fontsize=7, loc='upper left')
ax_h.spines['top'].set_visible(False)
ax_h.spines['right'].set_visible(False)
save_panel('expected_vs_actual_deltaRT')
print(f"exp ΔRT vs act ΔRT: slope={m2:.3f} [{ci2_lo:.3f}, {ci2_hi:.3f}], "
      f"compensated={comp2:.3f}, p(vs1)={p2_vs1:.4f}, n={int(ok2.sum())}")

#%% ============================================================================
# Flipped (speed-space) convention + starting-RT confound check
# ============================================================================
# Gain (speed per unit fluorescence) DROPS as the task hardens, so the natural
# convention is gain_new/gain_old < 1. Reciprocal axes:
#   x = width_old/width_new = speed-gain ratio (<1 = harder)
#   y = rt_old/rt_new = prev_rt/actual_rt = speed ratio (new/old)
# No-adapt: y = x.  Full comp: y = 1.  ABOVE diagonal = compensation.
# Points colored by prev_rt; corr(alpha, prev_rt) tests whether small
# perturbations were chosen when the animal started fast (the floor worry).
rt_prev_tr = np.full(len(sess_idx_arr), np.nan)
rt_cur_tr = np.full(len(sess_idx_arr), np.nan)
for ti in range(len(sess_idx_arr)):
    si = sess_idx_arr[ti]; s = switch_trial_arr[ti]
    m_, sess_ = session_keys[si]
    sw_si = np.asarray(switches_list[si]); em = np.where(sw_si == s)[0]
    if len(em) == 0 or em[0] == 0:
        continue
    rc = _es_lookup.get((m_, sess_, int(em[0])))
    rp = _es_lookup.get((m_, sess_, int(em[0]) - 1))
    if (rc and rp and np.isfinite(rc.get('actual_rt', np.nan))
            and np.isfinite(rp.get('actual_rt', np.nan)) and rc['actual_rt'] > 0):
        rt_prev_tr[ti] = rp['actual_rt']
        rt_cur_tr[ti] = rc['actual_rt']

x_g = 1.0 / alpha_arr               # speed-gain ratio (<1 = harder)
y_s = rt_prev_tr / rt_cur_tr        # speed ratio (old/new)
okf = np.isfinite(x_g) & np.isfinite(y_s) & np.isfinite(rt_prev_tr)
xf_, yf_, pf_ = x_g[okf], y_s[okf], rt_prev_tr[okf]

beta_f = _anchored_beta(xf_, yf_)
comp_f = 1.0 - beta_f
# Clean confound test: perturbation size (alpha) vs starting RT (no shared term)
r_conf, p_conf = pearsonr(alpha_arr[okf], pf_)

fig_i, ax_i = panel_fig()
sc = ax_i.scatter(xf_, yf_, s=20, c=pf_, cmap='viridis', alpha=0.75,
                  edgecolors='none')
cb = fig_i.colorbar(sc, ax=ax_i); cb.set_label('prev_rt (s)')
lim = [min(xf_.min(), yf_.min()) * 0.95, 1.15]
ax_i.plot(lim, lim, color='gray', linewidth=0.8, linestyle='--',
          label='no adaptation (y = x)')
ax_i.axhline(1.0, color='crimson', linewidth=0.8, linestyle=':',
             label='full compensation (y = 1)')
xfit = np.array([xf_.min(), 1.0])
ax_i.plot(xfit, 1.0 + beta_f * (xfit - 1.0), color='k', linewidth=1.0)
ax_i.set_title(f'compensated {comp_f:.0%} (β={beta_f:.2f}); '
               f'corr(alpha, prev_rt) r={r_conf:+.2f}, p={p_conf:.3f}',
               fontsize=8)
ax_i.set_xlabel('speed-gain ratio = width_old / width_new  (<1 = harder)')
ax_i.set_ylabel('speed ratio = prev_rt / actual_rt')
ax_i.legend(frameon=False, fontsize=7, loc='lower right')
ax_i.spines['top'].set_visible(False)
ax_i.spines['right'].set_visible(False)
save_panel('speedspace_compensation_confound')
print(f"flipped speed-space: compensated={comp_f:.3f} (beta={beta_f:.3f}), "
      f"n={int(okf.sum())}")
print(f"  confound corr(alpha, prev_rt): r={r_conf:+.3f}, p={p_conf:.4f} "
      f"(negative = small perturbations chosen at fast starts)")

#%% ============================================================================
# MAIN behavioral panel: per-session compensation FRACTION box plot
# ============================================================================
# Perturbation-adjusted compensation fraction (perturbation-independent):
#   f = (predicted_change − actual_change)/predicted_change
#     = (alpha − rt_ratio) / (alpha − 1)
#   f = 0 : no compensation,  f = 1 : full,  f > 1 : overcompensation
# Equals 1 − beta when data follows the anchored line, so it ties to the slope.
# Undefined as alpha -> 1 (can't define a fraction with no perturbation), so
# require a real perturbation (alpha > ALPHA_MIN) and aggregate by per-session
# median. (Scatter/slope version goes to supplement.)
from scipy.stats import wilcoxon

ALPHA_MIN = 1.2   # minimum gain change to define a compensation fraction

rt_ratio_comp = rt_cur_tr / rt_prev_tr
comp_fraction = (alpha_arr - rt_ratio_comp) / (alpha_arr - 1.0)
okf2 = (np.isfinite(comp_fraction) & np.isfinite(alpha_arr)
        & (alpha_arr > ALPHA_MIN))

# Per-transition values and per-session medians.
f_trans = comp_fraction[okf2]                  # one per transition
sess_f = {}
for ti in np.where(okf2)[0]:
    sess_f.setdefault(sess_idx_arr[ti], []).append(comp_fraction[ti])
sess_f_med = np.array([np.median(v) for v in sess_f.values()])

# Stats vs 0 (no compensation) for each unit
med_t = np.median(f_trans)
_, w_p_t = wilcoxon(f_trans)
med_s = np.median(sess_f_med)
_, w_p_s = wilcoxon(sess_f_med)

def _comp_box(ax, vals, label, med, p):
    ax.boxplot(vals, widths=0.5, showfliers=False,
               medianprops=dict(color='crimson', linewidth=1.5))
    ax.scatter(np.random.uniform(0.85, 1.15, len(vals)), vals,
               s=14, c='k', alpha=0.4, edgecolors='none')
    ax.axhline(0.0, color='gray', linewidth=0.8, linestyle='--',
               label='no compensation')
    ax.axhline(1.0, color='cornflowerblue', linewidth=0.8, linestyle=':',
               label='full compensation')
    ax.set_xticks([1]); ax.set_xticklabels([f'{label}\n(n={len(vals)})'])
    ax.set_ylabel('Compensation fraction\n(0 = none, 1 = full)')
    ax.set_title(f'median = {med:.2f}, vs 0: p = {p:.1e}', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig_b, (ax_t, ax_s) = panel_row(2)
_comp_box(ax_t, f_trans, 'per transition', med_t, w_p_t)
_comp_box(ax_s, sess_f_med, 'per session', med_s, w_p_s)
ax_s.legend(frameon=False, fontsize=7, loc='upper right')
save_panel('compensation_fraction_box')
print(f"Compensation FRACTION (alpha>{ALPHA_MIN}):")
print(f"  per transition: median={med_t:.3f}, n={len(f_trans)}, p={w_p_t:.4f}")
print(f"  per session:    median={med_s:.3f}, n={len(sess_f_med)}, p={w_p_s:.4f}")

#%% ============================================================================
# Compensation fraction vs late-trial ΔCN (neural <-> behavior, fraction units)
# ============================================================================
# Same x as the ΔCN-vs-(RT-expected) panel: post-switch pre-reward AUC minus
# the epoch-0 mean. y = behavioral compensation FRACTION for that transition
# (0 = none, 1 = full), restricted to real perturbations (alpha > ALPHA_MIN).
# Positive slope = bigger CN increase -> more of the imposed slowdown cancelled.
# Pearson in the title to match the sibling panels; Spearman also reported
# since the fraction has heavy tails when alpha is near the cutoff.
okc = (np.isfinite(dcn_ep0) & np.isfinite(comp_fraction)
       & np.isfinite(alpha_arr) & (alpha_arr > ALPHA_MIN))
xc, yc = dcn_ep0[okc], comp_fraction[okc]
r_c, p_c = pearsonr(xc, yc)
rho_c, prho_c = spearmanr(xc, yc)
mc, bc = np.polyfit(xc, yc, 1)

fig_cf, ax_cf = panel_fig()
ax_cf.scatter(xc, yc, s=16, c='k', alpha=0.5, edgecolors='none')
xf_cf = np.array([xc.min(), xc.max()])
ax_cf.plot(xf_cf, mc * xf_cf + bc, color='crimson', linewidth=1.0)
ax_cf.axhline(0.0, color='gray', linewidth=0.8, linestyle='--')           # none
ax_cf.axhline(1.0, color='cornflowerblue', linewidth=0.8, linestyle=':')  # full
ax_cf.axvline(0.0, color='gray', linewidth=0.5, linestyle=':')
ax_cf.set_xlabel('ΔCN vs epoch 0 (post pre-reward AUC − ep0 mean)')
ax_cf.set_ylabel('Compensation fraction (0 = none, 1 = full)')
ax_cf.set_title(f'r = {r_c:+.2f}, p = {p_c:.2e}, n = {int(okc.sum())}'
                f'  (rho = {rho_c:+.2f}, p = {prho_c:.1e})', fontsize=8)
ax_cf.spines['top'].set_visible(False)
ax_cf.spines['right'].set_visible(False)
save_panel('compfrac_vs_dCN_epoch0')
print(f"comp-fraction vs dCN(ep0): r={r_c:+.3f} p={p_c:.3e}, "
      f"rho={rho_c:+.3f} p={prho_c:.3e}, n={int(okc.sum())}")

#%% ============================================================================
# Fractional CN change vs epoch 0 — pooled across epochs (like RT analysis)
# ============================================================================
# Each epoch x>0 vs epoch 0 is one comparison: frac = (CN_x − CN_0)/CN_0,
# where CN_e = mean active-window CN over epoch e's trials (cn_mean_mat, raw
# roi_csv → stable positive denominator). Pool all epoch-vs-0 comparisons
# (per "transition"), and per-session median. Two box plots, vs 0 = no change.
cnfrac_trans = []          # one value per (session, epoch>0)
cnfrac_sess_id = []
for i in range(n_sess):
    # Raw pre-reward CN per trial (no baseline subtraction → positive
    # denominator), same window/source as the featured pre-reward AUC.
    F_cn = F_cn_list[i]
    n_fr_s, n_tr = F_cn.shape
    dt_i = dt_si_list[i]
    rt_i = rt_mat[:n_tr, i]
    pre_trial_f = int(PRE_TRIAL_S / dt_i)
    win_n = int(AUC_WIN_S / dt_i)
    cn_i = np.full(n_tr, np.nan)
    for t in range(n_tr):
        if not np.isfinite(rt_i[t]):
            continue
        rf = int(rt_i[t] / dt_i) + pre_trial_f
        lo, hi = rf - win_n, rf
        if lo < 0 or hi > n_fr_s:
            continue
        cn_i[t] = np.nanmean(F_cn[lo:hi, t])

    sw = np.asarray(switches_list[i], dtype=int)
    n_ep = len(sw)
    ntr = n_tr
    e0_end = sw[1] if n_ep > 1 else ntr
    cn0 = np.nanmean(cn_i[:min(e0_end, ntr)])
    if not np.isfinite(cn0) or cn0 <= 0:
        continue
    for e in range(1, n_ep):
        a = sw[e]
        b = sw[e + 1] if e + 1 < n_ep else ntr
        if a >= ntr:
            continue
        cn_e = np.nanmean(cn_i[a:min(b, ntr)])
        if np.isfinite(cn_e):
            cnfrac_trans.append((cn_e - cn0))
            cnfrac_sess_id.append(i)
cnfrac_trans = np.array(cnfrac_trans)
cnfrac_sess_id = np.array(cnfrac_sess_id)

# Per-session median across that session's epoch-vs-0 comparisons
cn_by_sess = {}
for v, sid in zip(cnfrac_trans, cnfrac_sess_id):
    cn_by_sess.setdefault(sid, []).append(v)
cnfrac_sess_med = np.array([np.median(v) for v in cn_by_sess.values()])

med_ct = np.median(cnfrac_trans); _, p_ct = wilcoxon(cnfrac_trans)
med_cs = np.median(cnfrac_sess_med); _, p_cs = wilcoxon(cnfrac_sess_med)

def _cn_box(ax, vals, label, med, p):
    ax.boxplot(vals, widths=0.5, showfliers=False,
               medianprops=dict(color='crimson', linewidth=1.5))
    ax.scatter(np.random.uniform(0.85, 1.15, len(vals)), vals,
               s=14, c='k', alpha=0.4, edgecolors='none')
    ax.axhline(0.0, color='gray', linewidth=0.8, linestyle='--',
               label='no change')
    ax.set_xticks([1]); ax.set_xticklabels([f'{label}\n(n={len(vals)})'])
    ax.set_ylabel('Fractional CN change vs epoch 0\n(CN_x − CN_0)/CN_0')
    ax.set_title(f'median = {med:+.2f}, vs 0: p = {p:.1e}', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig_cn, (axct, axcs) = panel_row(2)
_cn_box(axct, cnfrac_trans, 'per epoch-vs-0', med_ct, p_ct)
_cn_box(axcs, cnfrac_sess_med, 'per session', med_cs, p_cs)
axcs.legend(frameon=False, fontsize=7, loc='upper right')
save_panel('cn_change_vs_epoch0_fraction_box')
print("Fractional CN change vs epoch 0:")
print(f"  per epoch-vs-0: median={med_ct:+.3f}, n={len(cnfrac_trans)}, p={p_ct:.4f}")
print(f"  per session:    median={med_cs:+.3f}, n={len(cnfrac_sess_med)}, p={p_cs:.4f}")

#%% ============================================================================
# CN change vs epoch 0 in SESSION-Z units (normalizes cross-session F scale)
# ============================================================================
# z-score each session's raw pre-reward CN by its own whole-session mean/std,
# then epoch_x − epoch_0 in z units. Denominator is the session std (stable,
# no blow-up), and cross-session F-scale heterogeneity is removed.
cnz_trans = []
cnz_sess_id = []
for i in range(n_sess):
    F_cn = F_cn_list[i]
    n_fr_s, n_tr = F_cn.shape
    dt_i = dt_si_list[i]
    rt_i = rt_mat[:n_tr, i]
    pre_trial_f = int(PRE_TRIAL_S / dt_i)
    win_n = int(AUC_WIN_S / dt_i)
    cn_i = np.full(n_tr, np.nan)
    for t in range(n_tr):
        if not np.isfinite(rt_i[t]):
            continue
        rf = int(rt_i[t] / dt_i) + pre_trial_f
        lo, hi = rf - win_n, rf
        if lo < 0 or hi > n_fr_s:
            continue
        cn_i[t] = np.nanmean(F_cn[lo:hi, t])

    mu, sd = np.nanmean(cn_i), np.nanstd(cn_i)
    if not np.isfinite(sd) or sd <= 1e-9:
        continue
    cn_z = (cn_i - mu) / sd

    sw = np.asarray(switches_list[i], dtype=int)
    n_ep = len(sw)
    e0_end = sw[1] if n_ep > 1 else n_tr
    z0 = np.nanmean(cn_z[:min(e0_end, n_tr)])
    if not np.isfinite(z0):
        continue
    for e in range(1, n_ep):
        a = sw[e]
        b = sw[e + 1] if e + 1 < n_ep else n_tr
        if a >= n_tr:
            continue
        z_e = np.nanmean(cn_z[a:min(b, n_tr)])
        if np.isfinite(z_e):
            cnz_trans.append(z_e - z0)
            cnz_sess_id.append(i)
cnz_trans = np.array(cnz_trans)
cnz_sess_id = np.array(cnz_sess_id)

cnz_by_sess = {}
for v, sid in zip(cnz_trans, cnz_sess_id):
    cnz_by_sess.setdefault(sid, []).append(v)
cnz_sess_med = np.array([np.median(v) for v in cnz_by_sess.values()])

med_zt = np.median(cnz_trans); _, p_zt = wilcoxon(cnz_trans)
med_zs = np.median(cnz_sess_med); _, p_zs = wilcoxon(cnz_sess_med)

def _cnz_box(ax, vals, label, med, p):
    ax.boxplot(vals, widths=0.5, showfliers=False,
               medianprops=dict(color='crimson', linewidth=1.5))
    ax.scatter(np.random.uniform(0.85, 1.15, len(vals)), vals,
               s=14, c='k', alpha=0.4, edgecolors='none')
    ax.axhline(0.0, color='gray', linewidth=0.8, linestyle='--', label='no change')
    ax.set_xticks([1]); ax.set_xticklabels([f'{label}\n(n={len(vals)})'])
    ax.set_ylabel('CN change vs epoch 0 (session-z units)')
    ax.set_title(f'median = {med:+.2f}, vs 0: p = {p:.1e}', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig_cz, (azt, azs) = panel_row(2)
_cnz_box(azt, cnz_trans, 'per epoch-vs-0', med_zt, p_zt)
_cnz_box(azs, cnz_sess_med, 'per session', med_zs, p_zs)
azs.legend(frameon=False, fontsize=7, loc='upper right')
save_panel('cn_change_vs_epoch0_sessionz_box')
print("CN change vs epoch 0 (session-z):")
print(f"  per epoch-vs-0: median={med_zt:+.3f}, n={len(cnz_trans)}, p={p_zt:.4f}")
print(f"  per session:    median={med_zs:+.3f}, n={len(cnz_sess_med)}, p={p_zs:.4f}")

# Concatenate ALL trials across all sessions (if you want a flat per-trial pool)
all_rta = np.column_stack(rta_list)   # shape (n_fr, total_hits_across_sessions)
#%%
N = len(rta_list)
A = np.zeros((300,N))
A = np.full((300,N), np.nan)
for i in range(N):
    auc = np.nanmean(rta_list[i][0:100, :], axis=0)
    A[0:len(auc),i] = auc - np.nanmean(auc[1:10])
plt.plot(np.nanmean(A[0:82,:],1))
N = len(rta_list)
A = np.full((300, N), np.nan)
for i in range(N):
    auc = np.nanmean(rta_list[i][0:100, :], axis=0)
    A[:len(auc), i] = auc - np.nanmean(auc[1:10])

x = np.arange(82)
M = A[:82, :]
# Mean ± SEM across sessions (+ light smoothing).
m, lo, hi = _msem_curve(M.T, smooth_w=SMOOTH_W)

# Axis box sized explicitly in inches (tweak AX_W, AX_H).
AX_W, AX_H = 1.25, 1.25
fig, ax = panel_fig(AX_W, AX_H)
ax.fill_between(x, lo, hi, color='k', alpha=0.25, linewidth=0)
ax.plot(x, m, 'k')
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Trial')
ax.set_ylabel('Pre-reward AUC (baseline-subtracted)')
# Per-session slope significance + star
_sl = np.full(M.shape[1], np.nan)
for _i in range(M.shape[1]):
    _y = M[:, _i]; _okm = np.isfinite(_y)
    if _okm.sum() >= 20:
        _sl[_i] = np.polyfit(x[_okm], _y[_okm], 1)[0]
_sl = _sl[np.isfinite(_sl)]
_p1 = wilcoxon(_sl).pvalue
ax.text(0.03, 0.97, _stars(_p1), transform=ax.transAxes,
        ha='left', va='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_panel('prereward_auc_over_trials')
#%% ============================================================================
# CELL 3 (optional): reload from the pickle (skip the slow network load)
# ============================================================================
# Use this to repopulate workspace without re-running CELL 1.
# with open(SAVE_PATH, 'rb') as fh:
#     _d = pickle.load(fh)
# globals().update(_d)
# print(f"Loaded {len(session_keys)} sessions from {SAVE_PATH}")

#%% ============================================================================
# CELL 4: Per-session CN late-trial change vs. population
# ============================================================================
# For each session, compute the per-neuron pre-reward AUC over each trial,
# then "late minus early" change per neuron, then where CN's value sits in
# the population distribution (percentile + z-score).
#
# Requires re-loading F (which CELL 1 doesn't keep). One-shot loop; results
# are scalar per session.
# ============================================================================
PRE_REWARD_S    = 1.0    # seconds before reward to integrate (AUC window)
EARLY_LO_C4, EARLY_HI_C4 = 0, 10    # trials defining the "early" window
LATE_LO_C4,  LATE_HI_C4  = 50, 80   # trials defining the "late"  window

cn_late_change_per_session = []   # CN's own late-minus-early
cn_pctile_per_session      = []   # CN's percentile (0-100) in pop distribution
cn_zscore_per_session      = []   # CN's z-score relative to pop
pop_mean_per_session_c4    = []   # population mean of delta
pop_std_per_session_c4     = []   # population std of delta
# Same three quantities for the slope-based delta:
cn_slope_per_session         = []
cn_slope_pctile_per_session  = []
cn_slope_zscore_per_session  = []
pop_slope_mean_per_session   = []
pop_slope_std_per_session    = []
n_neurons_per_session      = []
c4_keys = []

def _cn_vs_pop(delta, cn_idx):
    """Return (cn_value, pctile, z, pop_mean, pop_std) for a per-neuron delta."""
    cn_val = (delta[cn_idx] if 0 <= cn_idx < len(delta) else np.nan)
    ok = np.isfinite(delta)
    if ok.sum() < 10 or not np.isfinite(cn_val):
        return cn_val, np.nan, np.nan, np.nan, np.nan
    pm = float(np.nanmean(delta[ok]))
    ps = float(np.nanstd(delta[ok]))
    pc = float(np.mean(delta[ok] < cn_val) * 100)
    zs = (cn_val - pm) / ps if ps > 0 else np.nan
    return float(cn_val), pc, zs, pm, ps

for k, (mouse, session) in enumerate(to_load):
    folder = ('//allen/aind/scratch/BCI/2p-raw/'
              + mouse + '/' + session + '/pophys/')
    try:
        data = ddct.load_hdf5(folder,
            ['F', 'conditioned_neuron', 'dt_si', 'reward_time'], [])
        F = data['F']     # (frames, neurons, trials)
        n_frames_s, n_neurons_s, n_trials_s = F.shape
        dt_s = float(np.asarray(data['dt_si']).ravel()[0])
        cn_idx_s = int(np.asarray(data['conditioned_neuron']).ravel()[0])
        rt_parsed = parse_hdf5_array_string(data['reward_time'], n_trials_s)
        rt_s = np.array(
            [x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
             for x in rt_parsed], dtype=float)
    except Exception as e:
        print(f"  [{k+1:2d}/{len(to_load)}] {mouse} {session}: "
              f"ERROR {str(e)[:120]}")
        continue

    pre_trial_f_s = int(PRE_TRIAL_S / dt_s)
    win_n_s = int(PRE_REWARD_S / dt_s)

    # Per-trial per-neuron pre-reward AUC: shape (n_neurons, n_trials)
    per_trial_auc = np.full((n_neurons_s, n_trials_s), np.nan)
    for t in range(n_trials_s):
        if not np.isfinite(rt_s[t]):
            continue
        rf = int(rt_s[t] / dt_s) + pre_trial_f_s
        lo, hi = rf - win_n_s, rf
        if lo < 0 or hi > n_frames_s:
            continue
        per_trial_auc[:, t] = np.nanmean(F[lo:hi, :, t], axis=0)

    # --- Delta variant 1: per-neuron late-minus-early ---
    e_hi = min(EARLY_HI_C4, n_trials_s)
    l_hi = min(LATE_HI_C4, n_trials_s)
    early_mean = np.nanmean(per_trial_auc[:, EARLY_LO_C4:e_hi], axis=1)
    late_mean  = np.nanmean(per_trial_auc[:, LATE_LO_C4:l_hi],  axis=1)
    delta_lme = late_mean - early_mean

    # --- Delta variant 2: per-neuron slope of AUC vs trial number ---
    trial_idx_s = np.arange(n_trials_s, dtype=float)
    delta_slope = np.full(n_neurons_s, np.nan)
    for n in range(n_neurons_s):
        y = per_trial_auc[n, :]
        valid = np.isfinite(y)
        if valid.sum() >= 10:
            delta_slope[n] = np.polyfit(trial_idx_s[valid], y[valid], 1)[0]

    # CN-vs-pop stats for each delta variant
    cn_d, pct, zsc, pm, ps = _cn_vs_pop(delta_lme, cn_idx_s)
    cn_late_change_per_session.append(cn_d)
    cn_pctile_per_session.append(pct)
    cn_zscore_per_session.append(zsc)
    pop_mean_per_session_c4.append(pm)
    pop_std_per_session_c4.append(ps)

    cn_sl, pct_sl, zsc_sl, pm_sl, ps_sl = _cn_vs_pop(delta_slope, cn_idx_s)
    cn_slope_per_session.append(cn_sl)
    cn_slope_pctile_per_session.append(pct_sl)
    cn_slope_zscore_per_session.append(zsc_sl)
    pop_slope_mean_per_session.append(pm_sl)
    pop_slope_std_per_session.append(ps_sl)

    n_neurons_per_session.append(int(np.sum(np.isfinite(delta_lme))))
    c4_keys.append((mouse, session))

    print(f"  [{k+1:2d}/{len(to_load)}] {mouse} {session}: "
          f"LME Δ={cn_d:+.3f} pct={pct:5.1f} z={zsc:+.2f}  |  "
          f"slope={cn_sl:+.4f} pct={pct_sl:5.1f} z={zsc_sl:+.2f}")

cn_late_change_per_session = np.array(cn_late_change_per_session)
cn_pctile_per_session = np.array(cn_pctile_per_session)
cn_zscore_per_session = np.array(cn_zscore_per_session)
pop_mean_per_session_c4 = np.array(pop_mean_per_session_c4)
pop_std_per_session_c4 = np.array(pop_std_per_session_c4)
cn_slope_per_session = np.array(cn_slope_per_session)
cn_slope_pctile_per_session = np.array(cn_slope_pctile_per_session)
cn_slope_zscore_per_session = np.array(cn_slope_zscore_per_session)
pop_slope_mean_per_session = np.array(pop_slope_mean_per_session)
pop_slope_std_per_session = np.array(pop_slope_std_per_session)
n_neurons_per_session = np.array(n_neurons_per_session)

def _summary(label, z_arr, p_arr):
    ok_z = np.isfinite(z_arr); ok_p = np.isfinite(p_arr)
    print(f"  {label}:")
    print(f"    z-score:   median = {np.median(z_arr[ok_z]):.2f}, "
          f"frac > 0 = {np.mean(z_arr[ok_z] > 0):.2f}, "
          f"frac > 1 = {np.mean(z_arr[ok_z] > 1):.2f}  (n = {ok_z.sum()})")
    print(f"    percentile: median = {np.median(p_arr[ok_p]):.1f}, "
          f"frac > 50 = {np.mean(p_arr[ok_p] > 50):.2f}, "
          f"frac > 90 = {np.mean(p_arr[ok_p] > 90):.2f}")

print(f"\nSummary across sessions:")
_summary('Late-minus-early', cn_zscore_per_session, cn_pctile_per_session)
_summary('Slope (AUC vs trial)',
         cn_slope_zscore_per_session, cn_slope_pctile_per_session)

# Plots: 2 rows (LME / slope) × 2 cols (z-score / percentile)
fig_c4, axes_c4 = panel_grid(2, 2)
for row, (lbl, z_arr, p_arr) in enumerate([
    ('Late-minus-early', cn_zscore_per_session, cn_pctile_per_session),
    ('Slope (AUC vs trial)', cn_slope_zscore_per_session, cn_slope_pctile_per_session),
]):
    ok_z = np.isfinite(z_arr); ok_p = np.isfinite(p_arr)
    ax = axes_c4[row, 0]
    ax.hist(z_arr[ok_z], bins=15, color='gray', edgecolor='white')
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
    ax.axvline(np.median(z_arr[ok_z]), color='crimson', linewidth=1.2,
               label=f'median = {np.median(z_arr[ok_z]):.2f}')
    ax.set_xlabel(f'{lbl}: CN z-score vs population')
    ax.set_ylabel('# sessions')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes_c4[row, 1]
    ax.hist(p_arr[ok_p], bins=np.linspace(0, 100, 21),
            color='gray', edgecolor='white')
    ax.axvline(50, color='k', linewidth=0.5, linestyle=':')
    ax.axvline(np.median(p_arr[ok_p]), color='crimson', linewidth=1.2,
               label=f'median = {np.median(p_arr[ok_p]):.1f}')
    ax.set_xlabel(f'{lbl}: CN percentile (vs population)')
    ax.set_ylabel('# sessions')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

save_panel('cn_vs_population_specificity')

#%% ============================================================================
# Per-session CN learning RATE (slope) vs behavioral compensation
# ============================================================================
# Pairs the CELL 4 per-session CN slope (slope of the CN's pre-reward AUC vs
# trial #, the slope-based learning delta) with that session's median
# compensation fraction (from the MAIN behavioral panel). Session is the unit:
# one CN learning rate vs one compensation value per session. Positive => the
# sessions with faster CN learning cancelled more of the imposed slowdown.
# Requires CELL 4 (cn_slope_per_session, c4_keys) AND the compensation-box cell
# (sess_f, session_keys) to have run.
comp_by_key = {session_keys[si]: np.median(v) for si, v in sess_f.items()}
xs_sl, ys_cp = [], []
for k, key in enumerate(c4_keys):
    if key in comp_by_key and np.isfinite(cn_slope_per_session[k]):
        xs_sl.append(cn_slope_per_session[k])
        ys_cp.append(comp_by_key[key])
xs_sl = np.array(xs_sl); ys_cp = np.array(ys_cp)

r_sc, p_sc = pearsonr(xs_sl, ys_cp)
rho_sc, prho_sc = spearmanr(xs_sl, ys_cp)
m_sc, b_sc = np.polyfit(xs_sl, ys_cp, 1)

fig_sc, ax_sc = panel_fig()
ax_sc.scatter(xs_sl, ys_cp, s=20, c='k', alpha=0.6, edgecolors='none')
xf_sc = np.array([xs_sl.min(), xs_sl.max()])
ax_sc.plot(xf_sc, m_sc * xf_sc + b_sc, color='crimson', linewidth=1.0)
ax_sc.axhline(0.0, color='gray', linewidth=0.8, linestyle='--')           # none
ax_sc.axhline(1.0, color='cornflowerblue', linewidth=0.8, linestyle=':')  # full
ax_sc.axvline(0.0, color='gray', linewidth=0.5, linestyle=':')
ax_sc.set_xlabel('CN learning rate (slope of pre-reward AUC vs trial)')
ax_sc.set_ylabel('Compensation fraction (per-session median)')
ax_sc.set_title(f'r = {r_sc:+.2f}, p = {p_sc:.2e}, n = {len(xs_sl)}'
                f'  (rho = {rho_sc:+.2f}, p = {prho_sc:.1e})', fontsize=8)
ax_sc.spines['top'].set_visible(False)
ax_sc.spines['right'].set_visible(False)
save_panel('compfrac_vs_cn_slope_per_session')
print(f"per-session CN slope vs compensation: r={r_sc:+.3f} p={p_sc:.3e}, "
      f"rho={rho_sc:+.3f} p={prho_sc:.3e}, n={len(xs_sl)}")

#%% ============================================================================
# Per-session CN slope PERCENTILE (vs population) vs behavioral compensation
# ============================================================================
# Same as above but x = the CN's slope percentile within the session's
# population (CELL 4 cn_slope_pctile_per_session): how much of an outlier the
# CN's learning RATE is (high = sparse / CN-specific learning). y = per-session
# median compensation fraction. Positive => sessions where CN learning is more
# CN-specific compensated more of the imposed slowdown.
comp_by_key = {session_keys[si]: np.median(v) for si, v in sess_f.items()}
xs_pc, ys_pc = [], []
for k, key in enumerate(c4_keys):
    if key in comp_by_key and np.isfinite(cn_slope_pctile_per_session[k]):
        xs_pc.append(cn_slope_pctile_per_session[k])
        ys_pc.append(comp_by_key[key])
xs_pc = np.array(xs_pc); ys_pc = np.array(ys_pc)

r_pc, p_pc = pearsonr(xs_pc, ys_pc)
rho_pc, prho_pc = spearmanr(xs_pc, ys_pc)
m_pc, b_pc = np.polyfit(xs_pc, ys_pc, 1)

fig_pc, ax_pc = panel_fig()
ax_pc.scatter(xs_pc, ys_pc, s=20, c='k', alpha=0.6, edgecolors='none')
xf_pc = np.array([xs_pc.min(), xs_pc.max()])
ax_pc.plot(xf_pc, m_pc * xf_pc + b_pc, color='crimson', linewidth=1.0)
ax_pc.axhline(0.0, color='gray', linewidth=0.8, linestyle='--')           # none
ax_pc.axhline(1.0, color='cornflowerblue', linewidth=0.8, linestyle=':')  # full
ax_pc.axvline(50.0, color='gray', linewidth=0.5, linestyle=':')           # pop median
ax_pc.set_xlabel('CN slope percentile (vs population)')
ax_pc.set_ylabel('Compensation fraction (per-session median)')
ax_pc.set_title(f'r = {r_pc:+.2f}, p = {p_pc:.2e}, n = {len(xs_pc)}'
                f'  (rho = {rho_pc:+.2f}, p = {prho_pc:.1e})', fontsize=8)
ax_pc.spines['top'].set_visible(False)
ax_pc.spines['right'].set_visible(False)
save_panel('compfrac_vs_cn_slope_percentile')
print(f"per-session CN slope percentile vs compensation: r={r_pc:+.3f} "
      f"p={p_pc:.3e}, rho={rho_pc:+.3f} p={prho_pc:.3e}, n={len(xs_pc)}")
