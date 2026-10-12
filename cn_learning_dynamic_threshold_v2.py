#%% ============================================================================
# cn_learning_dynamic_threshold.py
#
# Discovery view of CN activity across trials 0-100, all sessions.
# Sessions with fewer than 100 trials are NaN-padded.
#
# REBUILT FROM PRIMARY DATA: per-trial CN measure is computed here, directly
# from data['F']. The window is trial-start -> threshold-crossing-time, i.e.
# the active task period before reward delivery. For miss trials (no crossing)
# the full trial window is used.
#
# Standalone — no workspace dependencies. Run cells in order.
# ============================================================================

#%% ============================================================================
# CELL 0: Setup
# ============================================================================
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
from BCI_data_helpers import parse_hdf5_array_string

mpl.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'svg.fonttype': 'none',
})

PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')

# QC failures — mirror threshold_analysis2.py
_qc_fail = {
    ('BCI104', '012325'),
    ('BCI105', '012125'),
    ('BCI105', '012425'),
}
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

N_TRIALS = 100
BASE_N = 10            # first-N trials used as session baseline for subtraction
MAX_CN_DIST_UM = 5.0   # exclude sessions whose CN ROI is > this far (um) from
                       # the experimenter's drawn target. NaN distances pass
                       # through (older sessions without this field).

#%% ============================================================================
# CELL 1: Load CN per-trial from primary data (data['F'])
# ============================================================================
# Per-trial CN measure: mean of F[:, cn, trial] across all frames in the trial.
# F shape: (frames, neurons, trials). All trials share the same frame count.
# This does NOT truncate at reward time, so the measure is decoupled from RT.
# ============================================================================
list_of_dirs = session_counting.counter()

cn_by_session = {}     # (mouse, session) -> np.array shape (n_trials,)
dist_by_session = {}   # (mouse, session) -> float, distance (um) of CN ROI
                       # from experimenter's drawn target. NaN if unavailable.
noncn_by_session = {}  # (mouse, session) -> np.array shape (n_trials,)
                       # per-trial mean across all non-CN ROIs in F
frame_profile_early = {}  # (mouse, session) -> np.array shape (n_frames,)
                          # CN trace averaged across trials 0..9
frame_profile_rest = {}   # (mouse, session) -> np.array shape (n_frames,)
                          # CN trace averaged across trials 10..end
rt_by_session = {}        # (mouse, session) -> np.array (n_trials,) reward
                          # times in s; NaN for misses
first_switch_by_session = {}   # (mouse, session) -> int, first trial after
                               # epoch 0; equals n_trials if no switch
epoch0_mean_rt = {}       # (mouse, session) -> float, mean reward time over
                          # epoch 0, miss trials filled with 10s
epoch0_mean_rt_hits = {}  # (mouse, session) -> float, mean reward time over
                          # hit trials only in epoch 0
epoch0_hit_rate = {}      # (mouse, session) -> float, hit rate in epoch 0
load_errors = []

for mouse in mice:
    inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]
    for si in inds:
        session = list_of_dirs['Session'][si]
        if (mouse, session) in _qc_fail:
            continue
        folder = ('//allen/aind/scratch/BCI/2p-raw/'
                  + mouse + '/' + session + '/pophys/')
        try:
            data = ddct.load_hdf5(folder,
                                  ['F', 'conditioned_neuron', 'dist',
                                   'threshold_crossing_time', 'dt_si',
                                   'reward_time', 'BCI_thresholds'], [])
            F = data['F']
            cn = int(np.asarray(data['conditioned_neuron']).ravel()[0])
            dt_si_s = float(np.asarray(data['dt_si']).ravel()[0])
        except Exception as e:
            load_errors.append(((mouse, session), str(e)))
            continue

        # Parse threshold-crossing time (seconds) per trial. Defensive:
        # try direct numeric conversion first, fall back to the hdf5
        # array-string parser. Miss trials -> NaN.
        n_trials_s = F.shape[2]
        try:
            tc_arr = np.asarray(data['threshold_crossing_time'],
                                dtype=float).ravel()
            if tc_arr.size != n_trials_s:
                raise ValueError("shape mismatch")
            tc_per_trial = tc_arr
        except (TypeError, ValueError):
            tc_parsed = parse_hdf5_array_string(
                data['threshold_crossing_time'], n_trials_s)
            tc_per_trial = np.array(
                [x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
                 for x in tc_parsed], dtype=float)

        # CN-to-target distance (um). dist is a 1D array indexed by ROI.
        # Defensive: handle missing field, wrong shape, out-of-range index.
        try:
            dist_arr = np.asarray(data['dist']).ravel()
            dist_cn = float(dist_arr[cn]) if cn < len(dist_arr) else np.nan
        except (KeyError, IndexError, TypeError, ValueError):
            dist_cn = np.nan

        # Per-trial window: trial-start go cue -> threshold crossing.
        # F[0] is PRE_TRIAL_S before the go cue (pre-trial buffer), so both
        # the window start and the crossing frame are offset by the buffer.
        PRE_TRIAL_S = 2.0
        pre_trial_f = int(PRE_TRIAL_S / dt_si_s)
        n_frames_per_trial = F.shape[0]
        end_frames = np.where(
            np.isfinite(tc_per_trial) & (tc_per_trial > 0),
            np.minimum(((tc_per_trial + PRE_TRIAL_S) / dt_si_s).astype(int),
                       n_frames_per_trial),
            n_frames_per_trial,
        ).astype(int)

        # CN mean over [trial start, threshold crossing]
        n_neurons = F.shape[1]
        non_cn_mask = np.ones(n_neurons, dtype=bool)
        if 0 <= cn < n_neurons:
            non_cn_mask[cn] = False

        cn_per_trial = np.full(n_trials_s, np.nan, dtype=float)
        non_cn_per_trial = np.full(n_trials_s, np.nan, dtype=float)
        for ti in range(n_trials_s):
            ef = max(pre_trial_f + 1, end_frames[ti])
            cn_per_trial[ti] = np.nanmean(F[pre_trial_f:ef, cn, ti])
            non_cn_per_trial[ti] = np.nanmean(F[pre_trial_f:ef, non_cn_mask, ti])

        cn_by_session[(mouse, session)] = cn_per_trial
        dist_by_session[(mouse, session)] = dist_cn
        noncn_by_session[(mouse, session)] = non_cn_per_trial

        # --- Reward time per trial (parse like threshold_analysis.py) ---
        try:
            rt_parsed = parse_hdf5_array_string(data['reward_time'],
                                                n_trials_s)
            rt_per_trial = np.array(
                [x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
                 for x in rt_parsed], dtype=float)
        except Exception:
            rt_per_trial = np.full(n_trials_s, np.nan, dtype=float)
        rt_by_session[(mouse, session)] = rt_per_trial

        # --- First-switch trial from BCI_thresholds (forward-fill NaN) ---
        try:
            thr_raw = np.asarray(data['BCI_thresholds'], dtype=float)
            thr_upper = thr_raw[1, :].copy()
            for ii in range(1, thr_upper.size):
                if np.isnan(thr_upper[ii]):
                    thr_upper[ii] = thr_upper[ii - 1]
            if np.isnan(thr_upper[0]) and np.any(np.isfinite(thr_upper)):
                thr_upper[0] = thr_upper[np.isfinite(thr_upper)][0]
            d_up = np.diff(thr_upper)
            sw = np.where((d_up != 0) & np.isfinite(d_up))[0] + 1
            first_switch = int(sw[0]) if len(sw) > 0 else n_trials_s
        except Exception:
            first_switch = n_trials_s
        first_switch_by_session[(mouse, session)] = first_switch

        # --- Epoch-0 behavioral summaries ---
        rt_ep0 = rt_per_trial[:first_switch]
        hit_ep0 = np.isfinite(rt_ep0)
        # Mean RT with misses filled to 10s (captures both speed AND miss rate)
        rt_filled = rt_ep0.copy()
        rt_filled[~hit_ep0] = 10.0
        epoch0_mean_rt[(mouse, session)] = (float(np.nanmean(rt_filled))
                                             if len(rt_filled) > 0
                                             else np.nan)
        # Mean RT over hits only
        epoch0_mean_rt_hits[(mouse, session)] = (
            float(np.nanmean(rt_ep0[hit_ep0])) if np.any(hit_ep0) else np.nan)
        epoch0_hit_rate[(mouse, session)] = (float(np.mean(hit_ep0))
                                              if len(hit_ep0) > 0
                                              else np.nan)

        # Within-trial frame profile: CN trace averaged across trials 0-9
        # vs trials 10+. Different shapes between the two would indicate
        # the trial-0 spike has a specific within-trial origin.
        n_trials_s = F.shape[2]
        if n_trials_s >= 10:
            frame_profile_early[(mouse, session)] = np.nanmean(
                F[:, cn, 0:10], axis=1).astype(float)
            frame_profile_rest[(mouse, session)] = np.nanmean(
                F[:, cn, 10:], axis=1).astype(float)

        flag = '  ' if not np.isfinite(dist_cn) else (
            '!!' if dist_cn > MAX_CN_DIST_UM else '  ')
        n_cross = int(np.sum(np.isfinite(tc_per_trial) & (tc_per_trial > 0)))
        med_tc = (float(np.nanmedian(tc_per_trial[
            np.isfinite(tc_per_trial) & (tc_per_trial > 0)]))
            if n_cross > 0 else np.nan)
        print(f"  {flag} {mouse} {session}: {F.shape[2]:4d} trials, "
              f"CN={cn:3d}, dist={dist_cn:5.2f} um, "
              f"cross n={n_cross:3d}/{n_trials_s:3d} med_t={med_tc:4.1f}s, "
              f"mean={np.nanmean(cn_per_trial):+.3f}")

print(f"\nLoaded {len(cn_by_session)} sessions; {len(load_errors)} load errors")
for k, e in load_errors[:5]:
    print(f"  load error {k}: {e[:140]}")

# Summary of CN-to-target distance
_d = np.array(list(dist_by_session.values()), dtype=float)
_d_finite = _d[np.isfinite(_d)]
if len(_d_finite):
    print(f"\nCN-to-target distance: n_finite={len(_d_finite)}/{len(_d)}, "
          f"median={np.median(_d_finite):.2f} um, "
          f"max={_d_finite.max():.2f} um, "
          f"> {MAX_CN_DIST_UM} um: {int(np.sum(_d_finite > MAX_CN_DIST_UM))}")
else:
    print("\nNo finite CN-to-target distances loaded.")

#%% ============================================================================
# CELL 2: Build (n_sessions, N_TRIALS) matrix and plot
# ============================================================================
# Filter out sessions whose CN ROI is too far from the experimenter target.
# NaN distances pass through (older sessions without this field).
all_keys = sorted(cn_by_session.keys())
dropped = []
keys = []
for k in all_keys:
    d = dist_by_session.get(k, np.nan)
    if np.isfinite(d) and d > MAX_CN_DIST_UM:
        dropped.append((k, d))
    else:
        keys.append(k)

if dropped:
    print(f"Dropped {len(dropped)} sessions with CN dist > {MAX_CN_DIST_UM} um:")
    for (m, s), d in dropped:
        print(f"    {m} {s}: dist = {d:.2f} um")
else:
    print(f"No sessions dropped at MAX_CN_DIST_UM = {MAX_CN_DIST_UM} um")

n_sess = len(keys)
cn_mat = np.full((n_sess, N_TRIALS), np.nan)
trial_counts = np.zeros(n_sess, dtype=int)
for i, k in enumerate(keys):
    cn = cn_by_session[k]
    trial_counts[i] = len(cn)
    n = min(len(cn), N_TRIALS)
    cn_mat[i, :n] = cn[:n]

# Baseline-subtract per session (mean of first BASE_N trials)
cn_base = np.nanmean(cn_mat[:, :BASE_N], axis=1, keepdims=True)
cn_sub = cn_mat - cn_base

# Mean +/- SEM across sessions at each trial
n_per_trial = np.sum(np.isfinite(cn_sub), axis=0)
mean_sub = np.nanmean(cn_sub, axis=0)
sem_sub = np.nanstd(cn_sub, axis=0) / np.sqrt(np.clip(n_per_trial, 1, None))

# Sanity prints
print(f"n_sessions = {n_sess}")
print(f"trial-count quartiles: "
      f"min={trial_counts.min()}, 25%={int(np.percentile(trial_counts,25))}, "
      f"50%={int(np.percentile(trial_counts,50))}, "
      f"75%={int(np.percentile(trial_counts,75))}, max={trial_counts.max()}")
print(f"sessions reaching {N_TRIALS} trials: "
      f"{int(np.sum(trial_counts >= N_TRIALS))} / {n_sess}")
print(f"n contributing at trial 0:  {n_per_trial[0]}")
print(f"n contributing at trial 50: {n_per_trial[50]}")
print(f"n contributing at trial 99: {n_per_trial[-1]}")

# --- Figure: heatmap (left) + mean +/- SEM trace (right) ---
fig_w, fig_h = 5.5, 2.4
ax_w_in, ax_h_in = 1.9, 1.6   # explicit axis box (inches)

fig = plt.figure(figsize=(fig_w, fig_h))
ax1 = fig.add_axes([0.09, 0.18, ax_w_in / fig_w, ax_h_in / fig_h])
vlim = np.nanpercentile(np.abs(cn_sub), 95)
im = ax1.imshow(cn_sub, aspect='auto', interpolation='nearest',
                cmap='RdBu_r', vmin=-vlim, vmax=vlim)
ax1.set_xlabel('Trial')
ax1.set_ylabel('Session')
cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cb.set_label('CN - baseline (F)')

ax2 = fig.add_axes([0.62, 0.18, ax_w_in / fig_w, ax_h_in / fig_h])
trial_axis = np.arange(N_TRIALS)
ax2.fill_between(trial_axis, mean_sub - sem_sub, mean_sub + sem_sub,
                 color='k', alpha=0.20, linewidth=0)
ax2.plot(trial_axis, mean_sub, 'k', linewidth=1.0)
ax2.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax2.set_xlabel('Trial')
ax2.set_ylabel('CN - baseline (F)')
ax2.set_title(f'n = {n_sess} sessions (baseline = first {BASE_N} trials)',
              fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fname = 'cn_learning_dynamic_threshold'
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {fname}.png and {fname}.svg")

#%% ============================================================================
# CELL 3: Diagnostics — find the sessions driving the mean
# ============================================================================
# (1) Rank sessions by mean(CN - baseline) over trials 30..end, and by baseline
#     level itself. Big-negative sessions are the ones dragging the mean down.
# (2) Overlay every per-session trace under the population mean + median.
# No file is saved.
# ============================================================================
late_dev = np.nanmean(cn_sub[:, 30:], axis=1)
baseline_lvl = cn_base.ravel()

print("\nMost-NEGATIVE late deviation (likely outliers / bleaching):")
print(f"  {'rank':>4} {'mouse':>7} {'session':>8} {'n_trl':>6} "
      f"{'baseline':>9} {'late_dev':>9}")
order_neg = np.argsort(late_dev)
for r, i in enumerate(order_neg[:10]):
    m, s = keys[i]
    print(f"  {r+1:>4} {m:>7} {s:>8} {trial_counts[i]:>6d} "
          f"{baseline_lvl[i]:>+9.3f} {late_dev[i]:>+9.3f}")

print("\nMost-POSITIVE late deviation (clean learners):")
for r, i in enumerate(order_neg[::-1][:10]):
    m, s = keys[i]
    print(f"  {r+1:>4} {m:>7} {s:>8} {trial_counts[i]:>6d} "
          f"{baseline_lvl[i]:>+9.3f} {late_dev[i]:>+9.3f}")

fig2 = plt.figure(figsize=(5.5, 2.4))
ax_w_in, ax_h_in = 1.9, 1.6
axA = fig2.add_axes([0.10, 0.20, ax_w_in / 5.5, ax_h_in / 2.4])
for i in range(n_sess):
    axA.plot(trial_axis, cn_sub[i], color='0.75', linewidth=0.4)
axA.plot(trial_axis, mean_sub, 'k', linewidth=1.2, label='mean')
axA.plot(trial_axis, np.nanmedian(cn_sub, axis=0), color='crimson',
         linewidth=1.0, label='median')
axA.axhline(0, color='gray', linewidth=0.5, linestyle=':')
axA.set_xlabel('Trial')
axA.set_ylabel('CN - baseline (F)')
axA.set_title('All sessions overlaid', fontsize=8)
axA.legend(frameon=False, loc='lower left')
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)

axB = fig2.add_axes([0.60, 0.20, ax_w_in / 5.5, ax_h_in / 2.4])
axB.hist(late_dev, bins=20, color='0.6', edgecolor='white')
axB.axvline(0, color='gray', linewidth=0.5, linestyle=':')
axB.axvline(np.nanmedian(late_dev), color='crimson', linewidth=1.0,
            label=f'median = {np.nanmedian(late_dev):+.3f}')
axB.set_xlabel('Mean (CN - baseline), trials 30+')
axB.set_ylabel('# sessions')
axB.legend(frameon=False)
axB.spines['top'].set_visible(False)
axB.spines['right'].set_visible(False)

plt.show()

# --- Dump diagnostics to a text file for easy sharing ---
diag_path = os.path.join(PANEL_DIR,
                         'cn_learning_dynamic_threshold_diagnostics.txt')
with open(diag_path, 'w') as fh:
    fh.write("cn_learning_dynamic_threshold.py — diagnostics\n")
    fh.write(f"N_TRIALS={N_TRIALS}, BASE_N={BASE_N}, "
             f"MAX_CN_DIST_UM={MAX_CN_DIST_UM}\n")
    fh.write(f"Per-trial CN measure: np.nanmean(F[:, cn, trial]) "
             f"over all frames\n\n")

    # Section: load summary
    fh.write(f"--- Load summary ---\n")
    fh.write(f"n_sessions_loaded     = {len(cn_by_session)}\n")
    fh.write(f"n_sessions_kept       = {n_sess}\n")
    fh.write(f"n_dropped_by_dist     = {len(dropped)}\n")
    fh.write(f"n_load_errors         = {len(load_errors)}\n\n")

    # Section: CN-to-target distance distribution
    _d_all = np.array([dist_by_session[k] for k in all_keys], dtype=float)
    _df = _d_all[np.isfinite(_d_all)]
    fh.write(f"--- CN-to-target distance (um) ---\n")
    if len(_df):
        fh.write(f"n_finite={len(_df)}/{len(_d_all)}, "
                 f"median={np.median(_df):.2f}, "
                 f"mean={np.mean(_df):.2f}, "
                 f"max={_df.max():.2f}, "
                 f"n_over_{MAX_CN_DIST_UM}um={int(np.sum(_df > MAX_CN_DIST_UM))}\n\n")
    else:
        fh.write("no finite distances\n\n")

    # Section: dropped sessions
    fh.write(f"--- Sessions dropped (dist > {MAX_CN_DIST_UM} um) ---\n")
    if dropped:
        for (m, s), d in dropped:
            fh.write(f"  {m} {s}: dist = {d:.2f} um\n")
    else:
        fh.write("  (none)\n")
    fh.write("\n")

    # Section: load errors
    fh.write(f"--- Load errors ---\n")
    if load_errors:
        for k, e in load_errors:
            fh.write(f"  {k}: {e[:200]}\n")
    else:
        fh.write("  (none)\n")
    fh.write("\n")

    # Section: trial-count info
    fh.write(f"--- Trial counts per (kept) session ---\n")
    fh.write(f"min={trial_counts.min()}, "
             f"25%={int(np.percentile(trial_counts,25))}, "
             f"median={int(np.percentile(trial_counts,50))}, "
             f"75%={int(np.percentile(trial_counts,75))}, "
             f"max={trial_counts.max()}\n")
    fh.write(f"n reaching {N_TRIALS} trials: "
             f"{int(np.sum(trial_counts >= N_TRIALS))} / {n_sess}\n\n")

    # Section: per-trial n contributing
    fh.write(f"--- n contributing per trial position ---\n")
    for t in [0, 10, 25, 50, 75, 99]:
        fh.write(f"  trial {t:>3}: n = {n_per_trial[t]}\n")
    fh.write("\n")

    # Section: full ranked list of sessions by late_dev
    fh.write(f"--- All kept sessions, ranked by late_dev "
             f"(mean of CN-baseline, trials 30+) ---\n")
    fh.write(f"  {'rank':>4} {'mouse':>7} {'session':>8} {'n_trl':>6} "
             f"{'dist_um':>8} {'baseline':>9} {'late_dev':>9}\n")
    for r, i in enumerate(order_neg):
        m, s = keys[i]
        d = dist_by_session.get(keys[i], np.nan)
        fh.write(f"  {r+1:>4} {m:>7} {s:>8} {trial_counts[i]:>6d} "
                 f"{d:>8.2f} {baseline_lvl[i]:>+9.3f} {late_dev[i]:>+9.3f}\n")
    fh.write("\n")

    # Section: mean trace values (so we can plot in the chat if needed)
    fh.write(f"--- Population mean (CN - baseline) per trial ---\n")
    fh.write(f"  {'trial':>5} {'n':>4} {'mean':>9} {'sem':>9} {'median':>9}\n")
    median_sub = np.nanmedian(cn_sub, axis=0)
    for t in range(N_TRIALS):
        fh.write(f"  {t:>5} {n_per_trial[t]:>4d} "
                 f"{mean_sub[t]:>+9.4f} {sem_sub[t]:>9.4f} "
                 f"{median_sub[t]:>+9.4f}\n")

print(f"\nWrote diagnostics to {diag_path}")

#%% ============================================================================
# CELL 4: Baseline-definition comparison
# ============================================================================
# The trials 0-9 baseline introduces regression-to-the-mean (sessions whose
# first 10 trials happen to be high show "drops"; sessions whose first 10
# trials happen to be low show "rises"). Compare 4 baseline definitions to
# see how robust any learning trend is to the choice. The whole-session
# baseline (D) eliminates the regression-to-mean artifact entirely.
# ============================================================================
from scipy.stats import pearsonr

baselines = [
    ('A: trials 0-9 (current)',   np.nanmean(cn_mat[:,  0:10], axis=1,
                                              keepdims=True)),
    ('B: trials 3-12 (drop t=0)', np.nanmean(cn_mat[:,  3:13], axis=1,
                                              keepdims=True)),
    ('C: trials 10-29 (wider)',   np.nanmean(cn_mat[:, 10:30], axis=1,
                                              keepdims=True)),
    ('D: whole session mean',     np.nanmean(cn_mat, axis=1, keepdims=True)),
]

# --- Figure: 4 mean traces side by side ---
fig_w, fig_h = 8.0, 2.0
ax_w_in, ax_h_in = 1.5, 1.4
left_pad_in, bottom_in, gap_in = 0.5, 0.4, 0.3

fig4 = plt.figure(figsize=(fig_w, fig_h))
for ai, (label, base) in enumerate(baselines):
    sub = cn_mat - base
    n_t = np.sum(np.isfinite(sub), axis=0)
    m = np.nanmean(sub, axis=0)
    s = np.nanstd(sub, axis=0) / np.sqrt(np.clip(n_t, 1, None))
    med = np.nanmedian(sub, axis=0)
    left_frac = (left_pad_in + ai * (ax_w_in + gap_in)) / fig_w
    ax = fig4.add_axes([left_frac, bottom_in / fig_h,
                        ax_w_in / fig_w, ax_h_in / fig_h])
    ax.fill_between(trial_axis, m - s, m + s, color='k', alpha=0.20,
                    linewidth=0)
    ax.plot(trial_axis, m, 'k', linewidth=1.0, label='mean')
    ax.plot(trial_axis, med, color='crimson', linewidth=0.8, label='median')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Trial')
    if ai == 0:
        ax.set_ylabel('CN - baseline (F)')
    ax.set_title(label, fontsize=8)
    if ai == len(baselines) - 1:
        ax.legend(frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fname4 = 'cn_learning_baseline_comparison'
fig4.savefig(os.path.join(PANEL_DIR, f'{fname4}.png'), dpi=300)
fig4.savefig(os.path.join(PANEL_DIR, f'{fname4}.svg'))
plt.show()
print(f"Saved {fname4}")

# --- Scatter: baseline A vs late_dev under baseline A ---
late_dev_A = np.nanmean((cn_mat - baselines[0][1])[:, 30:], axis=1)
baseline_A = baselines[0][1].ravel()
ok_A = np.isfinite(baseline_A) & np.isfinite(late_dev_A)
r_A, p_A = pearsonr(baseline_A[ok_A], late_dev_A[ok_A])

# --- Scatter: session-mean F vs late_dev under baseline A ---
session_mean = np.nanmean(cn_mat, axis=1)
ok_S = np.isfinite(session_mean) & np.isfinite(late_dev_A)
r_S, p_S = pearsonr(session_mean[ok_S], late_dev_A[ok_S])

fig4b_w, fig4b_h = 5.0, 2.4
ax_w_in, ax_h_in = 1.6, 1.6
left_pad_in, bottom_in, gap_in = 0.7, 0.45, 0.5

fig4b = plt.figure(figsize=(fig4b_w, fig4b_h))
for ai, (xv, ok, r, p, xlabel) in enumerate([
    (baseline_A, ok_A, r_A, p_A, 'Baseline F (trials 0-9 mean)'),
    (session_mean, ok_S, r_S, p_S, 'Session-mean F'),
]):
    left_frac = (left_pad_in + ai * (ax_w_in + gap_in)) / fig4b_w
    ax = fig4b.add_axes([left_frac, bottom_in / fig4b_h,
                         ax_w_in / fig4b_w, ax_h_in / fig4b_h])
    ax.scatter(xv[ok], late_dev_A[ok], s=14, c='k', alpha=0.5,
               edgecolors='none')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel(xlabel)
    if ai == 0:
        ax.set_ylabel('Late dev (trials 30+, baseline A)')
    ax.set_title(f'r = {r:+.2f}, p = {p:.1e}, n = {int(ok.sum())}',
                 fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fname4b = 'cn_learning_baseline_regression'
fig4b.savefig(os.path.join(PANEL_DIR, f'{fname4b}.png'), dpi=300)
fig4b.savefig(os.path.join(PANEL_DIR, f'{fname4b}.svg'))
plt.show()
print(f"Saved {fname4b}")

# --- Append summary to diagnostics text file ---
with open(diag_path, 'a') as fh:
    fh.write("\n=== CELL 4: Baseline-definition comparison ===\n")
    fh.write(f"\n--- Per-baseline trace at key trial positions ---\n")
    fh.write(f"  {'baseline':33s} {'trial':>5s} {'n':>4s} "
             f"{'mean':>9s} {'sem':>9s} {'median':>9s}\n")
    for label, base in baselines:
        sub = cn_mat - base
        n_t = np.sum(np.isfinite(sub), axis=0)
        m = np.nanmean(sub, axis=0)
        s = np.nanstd(sub, axis=0) / np.sqrt(np.clip(n_t, 1, None))
        med = np.nanmedian(sub, axis=0)
        for t in [0, 10, 25, 50, 75]:
            fh.write(f"  {label:33s} {t:>5d} {n_t[t]:>4d} "
                     f"{m[t]:>+9.4f} {s[t]:>9.4f} {med[t]:>+9.4f}\n")
        fh.write("\n")
    fh.write(f"--- Regression-to-mean diagnostics (late_dev under baseline A) ---\n")
    fh.write(f"baseline A (trials 0-9) vs late_dev: "
             f"r = {r_A:+.3f}, p = {p_A:.2e}, n = {int(ok_A.sum())}\n")
    fh.write(f"session-mean F        vs late_dev: "
             f"r = {r_S:+.3f}, p = {p_S:.2e}, n = {int(ok_S.sum())}\n")
    fh.write(f"(strong negative r under baseline A == regression-to-mean artifact)\n")

print(f"Appended baseline comparison to {diag_path}")

#%% ============================================================================
# CELL 5: Session-z-scored CN, then average across sessions
# ============================================================================
# Following the old-paper protocol of z-scoring each session before combining.
# Normalize using whole-session mean and std (long, stable window) rather than
# the first-10-trial window, which we just showed is biased and creates a
# regression-to-mean artifact under dynamic thresholds.
# ============================================================================

SMOOTH_WIN = 10   # moving-average window applied per session before averaging

def _smooth_nan(x, w):
    """NaN-aware moving average with edge reflection (output same length)."""
    if w <= 1:
        return x
    k = np.ones(w) / w
    valid = np.isfinite(x).astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)
    num = np.convolve(x0, k, mode='same')
    den = np.convolve(valid, k, mode='same')
    return np.where(den > 0, num / den, np.nan)

sess_mean = np.nanmean(cn_mat, axis=1, keepdims=True)
sess_std = np.nanstd(cn_mat, axis=1, keepdims=True)
cn_z = (cn_mat - sess_mean) / np.where(sess_std > 1e-6, sess_std, np.nan)

# Smooth each session's z trace before pooling
cn_z_smooth = np.vstack([_smooth_nan(cn_z[i], SMOOTH_WIN)
                         for i in range(cn_z.shape[0])])

n_per_trial_z = np.sum(np.isfinite(cn_z_smooth), axis=0)
mean_z = np.nanmean(cn_z_smooth, axis=0)
sem_z = np.nanstd(cn_z_smooth, axis=0) / np.sqrt(np.clip(n_per_trial_z, 1, None))
median_z = np.nanmedian(cn_z_smooth, axis=0)

# --- Figure: heatmap (left) + mean +/- SEM trace (right) ---
fig_w, fig_h = 5.5, 2.4
ax_w_in, ax_h_in = 1.9, 1.6

fig5 = plt.figure(figsize=(fig_w, fig_h))

ax1 = fig5.add_axes([0.09, 0.18, ax_w_in / fig_w, ax_h_in / fig_h])
vlim = np.nanpercentile(np.abs(cn_z), 95)
im = ax1.imshow(cn_z, aspect='auto', interpolation='nearest',
                cmap='RdBu_r', vmin=-vlim, vmax=vlim)
ax1.set_xlabel('Trial')
ax1.set_ylabel('Session')
cb = fig5.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cb.set_label('CN (session z-score)')

ax2 = fig5.add_axes([0.62, 0.18, ax_w_in / fig_w, ax_h_in / fig_h])
ax2.fill_between(trial_axis, mean_z - sem_z, mean_z + sem_z,
                 color='k', alpha=0.20, linewidth=0)
ax2.plot(trial_axis, mean_z, 'k', linewidth=1.0, label='mean')
ax2.plot(trial_axis, median_z, color='crimson', linewidth=0.8, label='median')
ax2.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax2.set_xlabel('Trial')
ax2.set_ylabel('CN (session z-score)')
ax2.set_title(f'n = {n_sess} sessions; z by whole-session mean/std; '
              f'smooth w={SMOOTH_WIN}', fontsize=8)
ax2.legend(frameon=False, loc='upper right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fname5 = 'cn_learning_session_zscore'
fig5.savefig(os.path.join(PANEL_DIR, f'{fname5}.png'), dpi=300)
fig5.savefig(os.path.join(PANEL_DIR, f'{fname5}.svg'))
plt.show()
print(f"Saved {fname5}")

# Append to diagnostics
with open(diag_path, 'a') as fh:
    fh.write("\n=== CELL 5: Session-z-scored CN (whole-session mean/std) ===\n")
    fh.write(f"  {'trial':>5s} {'n':>4s} {'mean':>9s} {'sem':>9s} "
             f"{'median':>9s}\n")
    for t in range(N_TRIALS):
        fh.write(f"  {t:>5d} {n_per_trial_z[t]:>4d} "
                 f"{mean_z[t]:>+9.4f} {sem_z[t]:>9.4f} "
                 f"{median_z[t]:>+9.4f}\n")
print(f"Appended z-score trace to {diag_path}")

#%% ============================================================================
# CELL 6: Trial-0 spike diagnostics
# ============================================================================
# (a) Compare CN per-trial mean vs non-CN per-trial mean, both session-z'd.
#     If non-CN also spikes at trial 0 -> global imaging artifact (laser
#     warm-up / pre-bleach). If only CN spikes -> CN-specific.
# (b) Within-trial frame profile of CN, averaged across trials 0-9 vs 10+.
#     Different shapes would indicate the trial-0 spike has a within-trial
#     origin (e.g., the very first frames are elevated).
# ============================================================================

# Build aligned non-CN matrix (n_sess x N_TRIALS), NaN-padded — only sessions
# that also passed the dist filter (keys list).
noncn_mat = np.full((n_sess, N_TRIALS), np.nan)
for i, k in enumerate(keys):
    arr = noncn_by_session.get(k)
    if arr is None:
        continue
    nn = min(len(arr), N_TRIALS)
    noncn_mat[i, :nn] = arr[:nn]

# Session z-score both
def _sess_z(M):
    m = np.nanmean(M, axis=1, keepdims=True)
    s = np.nanstd(M, axis=1, keepdims=True)
    return (M - m) / np.where(s > 1e-6, s, np.nan)

cn_z_for6 = _sess_z(cn_mat)
noncn_z = _sess_z(noncn_mat)

# Smooth each row before pooling (reuse helper + window from CELL 5)
cn_z_sm = np.vstack([_smooth_nan(cn_z_for6[i], SMOOTH_WIN)
                     for i in range(n_sess)])
noncn_z_sm = np.vstack([_smooth_nan(noncn_z[i], SMOOTH_WIN)
                        for i in range(n_sess)])

def _m_s(M):
    n = np.sum(np.isfinite(M), axis=0)
    m = np.nanmean(M, axis=0)
    s = np.nanstd(M, axis=0) / np.sqrt(np.clip(n, 1, None))
    return m, s, n

m_cn, s_cn, n_cn = _m_s(cn_z_sm)
m_nc, s_nc, n_nc = _m_s(noncn_z_sm)

# --- Figure: (a) CN vs non-CN trial profile, (b) within-trial frame profile ---
fig_w, fig_h = 6.5, 2.4
ax_w_in, ax_h_in = 2.0, 1.6
left_pad_in, bottom_in, gap_in = 0.7, 0.5, 0.7

fig6 = plt.figure(figsize=(fig_w, fig_h))

# Panel a: CN vs non-CN per-trial means (session-z'd, smoothed)
axA = fig6.add_axes([left_pad_in / fig_w, bottom_in / fig_h,
                     ax_w_in / fig_w, ax_h_in / fig_h])
axA.fill_between(trial_axis, m_cn - s_cn, m_cn + s_cn,
                 color='k', alpha=0.20, linewidth=0)
axA.plot(trial_axis, m_cn, 'k', linewidth=1.0, label='CN')
axA.fill_between(trial_axis, m_nc - s_nc, m_nc + s_nc,
                 color='cornflowerblue', alpha=0.20, linewidth=0)
axA.plot(trial_axis, m_nc, color='cornflowerblue', linewidth=1.0,
         label='non-CN mean')
axA.axhline(0, color='gray', linewidth=0.5, linestyle=':')
axA.set_xlabel('Trial')
axA.set_ylabel('Session z-score')
axA.set_title('CN vs non-CN (smoothed)', fontsize=8)
axA.legend(frameon=False, loc='upper right')
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)

# Panel b: within-trial frame profile of CN, trials 0-9 vs trials 10+
# Average across sessions of frame-by-frame CN trace; pad shorter sessions.
common_keys = [k for k in keys
               if k in frame_profile_early and k in frame_profile_rest]
if common_keys:
    n_frames_max = max(len(frame_profile_early[k]) for k in common_keys)
    fp_early = np.full((len(common_keys), n_frames_max), np.nan)
    fp_rest = np.full((len(common_keys), n_frames_max), np.nan)
    for i, k in enumerate(common_keys):
        ae = frame_profile_early[k]
        ar = frame_profile_rest[k]
        fp_early[i, :len(ae)] = ae
        fp_rest[i, :len(ar)] = ar
    # Session-normalize: subtract each row's mean so different absolute F
    # levels don't dominate. Don't divide by std here -- we want to compare
    # within-trial shape, not amplitude.
    fp_early -= np.nanmean(fp_early, axis=1, keepdims=True)
    fp_rest -= np.nanmean(fp_rest, axis=1, keepdims=True)
    me, se, _ = _m_s(fp_early)
    mr, sr, _ = _m_s(fp_rest)
    frame_axis = np.arange(n_frames_max)

    axB = fig6.add_axes([(left_pad_in + ax_w_in + gap_in) / fig_w,
                         bottom_in / fig_h,
                         ax_w_in / fig_w, ax_h_in / fig_h])
    axB.fill_between(frame_axis, me - se, me + se, color='crimson',
                     alpha=0.20, linewidth=0)
    axB.plot(frame_axis, me, color='crimson', linewidth=1.0,
             label='trials 0-9')
    axB.fill_between(frame_axis, mr - sr, mr + sr, color='k',
                     alpha=0.20, linewidth=0)
    axB.plot(frame_axis, mr, 'k', linewidth=1.0, label='trials 10+')
    axB.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    axB.set_xlabel('Frame within trial')
    axB.set_ylabel('CN F (row-mean subtracted)')
    axB.set_title('Within-trial CN profile', fontsize=8)
    axB.legend(frameon=False, loc='upper right')
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)

fname6 = 'cn_trial0_spike_diagnostics'
fig6.savefig(os.path.join(PANEL_DIR, f'{fname6}.png'), dpi=300)
fig6.savefig(os.path.join(PANEL_DIR, f'{fname6}.svg'))
plt.show()
print(f"Saved {fname6}")

# Append to diagnostics file
with open(diag_path, 'a') as fh:
    fh.write("\n=== CELL 6: Trial-0 spike diagnostics ===\n")
    fh.write("\n--- CN vs non-CN at early trials (smoothed, session-z) ---\n")
    fh.write(f"  {'trial':>5s} {'n':>4s} "
             f"{'CN_mean':>9s} {'CN_sem':>9s} "
             f"{'nCN_mean':>9s} {'nCN_sem':>9s}\n")
    for t in [0, 1, 2, 3, 5, 10, 25, 50]:
        fh.write(f"  {t:>5d} {n_cn[t]:>4d} "
                 f"{m_cn[t]:>+9.4f} {s_cn[t]:>9.4f} "
                 f"{m_nc[t]:>+9.4f} {s_nc[t]:>9.4f}\n")

    if common_keys:
        fh.write("\n--- Within-trial CN frame profile (row-mean subtracted) ---\n")
        fh.write(f"  {'frame':>5s} {'early_mean':>11s} {'early_sem':>10s} "
                 f"{'rest_mean':>10s} {'rest_sem':>9s}\n")
        # Show first 20 frames + a few late ones
        for f_idx in list(range(0, 20)) + [40, 80, 160, 240]:
            if f_idx < n_frames_max:
                fh.write(f"  {f_idx:>5d} "
                         f"{me[f_idx]:>+11.4f} {se[f_idx]:>10.4f} "
                         f"{mr[f_idx]:>+10.4f} {sr[f_idx]:>9.4f}\n")
print(f"Appended trial-0 diagnostics to {diag_path}")

#%% ============================================================================
# CELL 7: Does epoch-0 mean RT explain cross-session differences in CN?
# ============================================================================
# For each kept session, build per-session CN summary stats and scatter them
# against epoch-0 mean reward time (misses filled to 10s). Also split sessions
# into terciles by epoch-0 mean RT and overlay the average smoothed CN traces.
# ============================================================================

# Per-session arrays, in `keys` order
ep0_rt = np.array([epoch0_mean_rt.get(k, np.nan) for k in keys], dtype=float)
ep0_rt_hits = np.array([epoch0_mean_rt_hits.get(k, np.nan)
                        for k in keys], dtype=float)
ep0_hr = np.array([epoch0_hit_rate.get(k, np.nan) for k in keys], dtype=float)
first_sw = np.array([first_switch_by_session.get(k, np.nan)
                     for k in keys], dtype=float)

# CN summary stats (matched to keys order)
sess_mean_cn = np.nanmean(cn_mat, axis=1)
trial0_val = cn_mat[:, 0]
early_val = np.nanmean(cn_mat[:, 0:10], axis=1)   # baseline A
late_val = np.nanmean(cn_mat[:, 30:], axis=1)
late_minus_early = late_val - early_val           # raw "learning" delta

# Linear slope of CN vs trial index per session (over available trials)
cn_slope = np.full(n_sess, np.nan)
for i in range(n_sess):
    y = cn_mat[i]
    ok = np.isfinite(y)
    if ok.sum() >= 10:
        x = np.arange(len(y))[ok]
        cn_slope[i] = np.polyfit(x, y[ok], 1)[0]

# Scatter helper
def _scatter(ax, x, y, xlabel, ylabel):
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=14, c='k', alpha=0.55, edgecolors='none')
    if ok.sum() >= 5:
        r, p = pearsonr(x[ok], y[ok])
        m, b = np.polyfit(x[ok], y[ok], 1)
        xf = np.array([x[ok].min(), x[ok].max()])
        ax.plot(xf, m * xf + b, color='crimson', linewidth=0.8)
        title = f'r={r:+.2f}, p={p:.1e}, n={int(ok.sum())}'
    else:
        title = f'n={int(ok.sum())}'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.4, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return (pearsonr(x[ok], y[ok]) if ok.sum() >= 5 else (np.nan, np.nan))

# --- Figure 7a: scatters of CN summary stats vs epoch-0 mean RT ---
panels_a = [
    (sess_mean_cn,      'Session-mean CN (F)'),
    (trial0_val,        'CN at trial 0 (F)'),
    (late_minus_early,  'Late - early CN (F)'),
    (cn_slope,          'CN slope (F / trial)'),
]
fig7a_w, fig7a_h = 8.0, 2.2
ax_w_in, ax_h_in = 1.5, 1.4
left_pad_in, bottom_in, gap_in = 0.55, 0.5, 0.35

fig7a = plt.figure(figsize=(fig7a_w, fig7a_h))
correlations = []
for ai, (yv, ylabel) in enumerate(panels_a):
    left_frac = (left_pad_in + ai * (ax_w_in + gap_in)) / fig7a_w
    ax = fig7a.add_axes([left_frac, bottom_in / fig7a_h,
                         ax_w_in / fig7a_w, ax_h_in / fig7a_h])
    r, p = _scatter(ax, ep0_rt, yv, 'Epoch-0 mean RT (s)', ylabel)
    correlations.append((ylabel, r, p))

fname7a = 'cn_vs_epoch0_rt_scatter'
fig7a.savefig(os.path.join(PANEL_DIR, f'{fname7a}.png'), dpi=300)
fig7a.savefig(os.path.join(PANEL_DIR, f'{fname7a}.svg'))
plt.show()
print(f"Saved {fname7a}")

# --- Figure 7b: CN traces split by epoch-0 mean RT tercile ---
ok_ep0 = np.isfinite(ep0_rt)
if ok_ep0.sum() >= 9:
    q1, q2 = np.nanpercentile(ep0_rt[ok_ep0], [33.3, 66.7])
    terc = np.full(n_sess, -1, dtype=int)
    terc[ok_ep0 & (ep0_rt < q1)] = 0   # fast
    terc[ok_ep0 & (ep0_rt >= q1) & (ep0_rt < q2)] = 1   # mid
    terc[ok_ep0 & (ep0_rt >= q2)] = 2   # slow

    # Session-z'd, smoothed (matches CELL 5 conventions)
    sm = np.nanmean(cn_mat, axis=1, keepdims=True)
    ss = np.nanstd(cn_mat, axis=1, keepdims=True)
    cn_z7 = (cn_mat - sm) / np.where(ss > 1e-6, ss, np.nan)
    cn_z7_sm = np.vstack([_smooth_nan(cn_z7[i], SMOOTH_WIN)
                          for i in range(n_sess)])

    fig7b_w, fig7b_h = 4.0, 2.4
    ax_w_in, ax_h_in = 2.4, 1.7
    fig7b = plt.figure(figsize=(fig7b_w, fig7b_h))
    ax = fig7b.add_axes([0.18, 0.22,
                         ax_w_in / fig7b_w, ax_h_in / fig7b_h])
    colors = ['#1b7837', '#999999', '#762a83']
    labels = [f'Fast (RT<{q1:.1f}s)', f'Mid', f'Slow (RT>={q2:.1f}s)']
    for ti in range(3):
        rows = cn_z7_sm[terc == ti]
        if rows.shape[0] == 0:
            continue
        n_t = np.sum(np.isfinite(rows), axis=0)
        m = np.nanmean(rows, axis=0)
        s = np.nanstd(rows, axis=0) / np.sqrt(np.clip(n_t, 1, None))
        ax.fill_between(trial_axis, m - s, m + s,
                        color=colors[ti], alpha=0.15, linewidth=0)
        ax.plot(trial_axis, m, color=colors[ti], linewidth=1.1,
                label=f'{labels[ti]} (n={rows.shape[0]})')
    ax.axhline(0, color='gray', linewidth=0.4, linestyle=':')
    ax.set_xlabel('Trial')
    ax.set_ylabel('CN (session z, smoothed)')
    ax.set_title(f'Split by epoch-0 mean RT (w={SMOOTH_WIN})', fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fname7b = 'cn_traces_split_by_epoch0_rt'
    fig7b.savefig(os.path.join(PANEL_DIR, f'{fname7b}.png'), dpi=300)
    fig7b.savefig(os.path.join(PANEL_DIR, f'{fname7b}.svg'))
    plt.show()
    print(f"Saved {fname7b}")
else:
    print("Not enough sessions with finite epoch-0 RT to tercile-split")

# --- Append to diagnostics file ---
with open(diag_path, 'a') as fh:
    fh.write("\n=== CELL 7: Epoch-0 RT vs CN summary stats ===\n")
    fh.write(f"\n--- Per-session epoch-0 behavioral summary ---\n")
    fh.write(f"  {'mouse':>7} {'session':>8} {'first_sw':>9} "
             f"{'ep0_n':>6} {'ep0_HR':>7} "
             f"{'ep0_RT_all':>11} {'ep0_RT_hits':>12}\n")
    for i, k in enumerate(keys):
        m, s = k
        fsw = first_sw[i]
        ep0_n_trials = int(fsw) if np.isfinite(fsw) else -1
        fh.write(f"  {m:>7} {s:>8} {ep0_n_trials:>9d} "
                 f"{ep0_n_trials:>6d} {ep0_hr[i]:>7.2f} "
                 f"{ep0_rt[i]:>11.2f} {ep0_rt_hits[i]:>12.2f}\n")

    fh.write(f"\n--- Correlations: epoch-0 mean RT vs CN summary stats ---\n")
    for label, r, p in correlations:
        fh.write(f"  {label:25s}  r = {r:+.3f}, p = {p:.2e}\n")

print(f"Appended CELL 7 diagnostics to {diag_path}")

#%% ============================================================================
# CELL 8: Per-session learning demand from the threshold protocol
# ============================================================================
# Aggregate the per-transition expected_delta_rt and expected_hr drop across
# threshold-increase transitions in each session. Sessions that imposed more
# demand for CN increase should show bigger CN responses if learning is real.
#
# Requires all_epoch_stats. Loaded from disk if not already in workspace —
# threshold_analysis2.py CELL 5 saves it to a pickle on completion.
# ============================================================================
_eps_pkl = os.path.join(_THIS_DIR, 'meta_analysis_results',
                        'all_epoch_stats.pkl')
if 'all_epoch_stats' not in globals():
    if os.path.exists(_eps_pkl):
        import pickle as _pickle
        with open(_eps_pkl, 'rb') as _fh:
            _saved = _pickle.load(_fh)
        all_epoch_stats = _saved['all_epoch_stats']
        all_session_trials = _saved['all_session_trials']
        print(f"Loaded all_epoch_stats from {_eps_pkl}")
        print(f"  {len(all_epoch_stats)} epoch records, "
              f"{len(all_session_trials)} sessions")
    else:
        print(f"CELL 8 skipped: all_epoch_stats not in workspace and no "
              f"pickle at {_eps_pkl}")
        print("  Run threshold_analysis2.py through CELL 5 first.")

if 'all_epoch_stats' not in globals():
    pass   # nothing to do
else:
    # Build per-session demand summaries
    sess_recs = {}
    for r in all_epoch_stats:
        sess_recs.setdefault((r['mouse'], r['session']), []).append(r)
    for k in sess_recs:
        sess_recs[k].sort(key=lambda r: r['epoch'])

    total_expected_drt = np.full(n_sess, np.nan)
    total_hr_drop = np.full(n_sess, np.nan)
    max_expected_drt = np.full(n_sess, np.nan)
    n_increases = np.full(n_sess, 0, dtype=int)

    for i, k in enumerate(keys):
        recs = sess_recs.get(k, [])
        if len(recs) < 2:
            total_expected_drt[i] = 0.0
            total_hr_drop[i] = 0.0
            max_expected_drt[i] = 0.0
            continue
        drts, hr_drops = [], []
        for j in range(1, len(recs)):
            if recs[j]['upper'] > recs[j - 1]['upper']:
                drt = recs[j].get('expected_delta_rt', np.nan)
                if np.isfinite(drt):
                    drts.append(max(drt, 0.0))
                ehr = recs[j].get('expected_hr_correct', np.nan)
                if np.isfinite(ehr):
                    hr_drops.append(max(1.0 - ehr, 0.0))
        n_increases[i] = max(len(drts), len(hr_drops))
        total_expected_drt[i] = float(np.sum(drts)) if drts else 0.0
        max_expected_drt[i] = float(np.max(drts)) if drts else 0.0
        total_hr_drop[i] = float(np.sum(hr_drops)) if hr_drops else 0.0

    n_sess_with_inc = int(np.sum(n_increases > 0))
    print(f"CELL 8: {n_sess_with_inc}/{n_sess} kept sessions have >=1 "
          f"threshold increase")
    print(f"  total_expected_drt: median={np.nanmedian(total_expected_drt):.2f}s, "
          f"max={np.nanmax(total_expected_drt):.2f}s")
    print(f"  total_hr_drop:      median={np.nanmedian(total_hr_drop):.2f}, "
          f"max={np.nanmax(total_hr_drop):.2f}")

    # --- Scatters: demand vs CN summary stats ---
    cn_stats = [
        (sess_mean_cn,     'Session-mean CN (F)'),
        (trial0_val,       'CN at trial 0 (F)'),
        (late_minus_early, 'Late - early CN (F)'),
        (cn_slope,         'CN slope (F / trial)'),
    ]
    demand_predictors = [
        (total_expected_drt, 'Total expected DRT (s)'),
        (total_hr_drop,      'Total expected HR drop'),
    ]

    fig_w, fig_h = 8.0, 4.4
    ax_w_in, ax_h_in = 1.5, 1.4
    left_pad_in, bottom_in, gap_in_x, gap_in_y = 0.55, 0.45, 0.35, 0.85

    fig8 = plt.figure(figsize=(fig_w, fig_h))
    cell8_correlations = []
    for ri, (xv, xlabel) in enumerate(demand_predictors):
        for ci, (yv, ylabel) in enumerate(cn_stats):
            left_frac = (left_pad_in + ci * (ax_w_in + gap_in_x)) / fig_w
            bottom_frac = (bottom_in + (1 - ri) * (ax_h_in + gap_in_y)) / fig_h
            ax = fig8.add_axes([left_frac, bottom_frac,
                                ax_w_in / fig_w, ax_h_in / fig_h])
            # Restrict to sessions with at least one increase
            mask = n_increases > 0
            xx = np.where(mask, xv, np.nan)
            r, p = _scatter(ax, xx, yv, xlabel, ylabel)
            cell8_correlations.append((xlabel, ylabel, r, p))

    fname8 = 'cn_vs_threshold_demand_scatter'
    fig8.savefig(os.path.join(PANEL_DIR, f'{fname8}.png'), dpi=300)
    fig8.savefig(os.path.join(PANEL_DIR, f'{fname8}.svg'))
    plt.show()
    print(f"Saved {fname8}")

    # Append diagnostics
    with open(diag_path, 'a') as fh:
        fh.write("\n=== CELL 8: Threshold-protocol demand vs CN response ===\n")
        fh.write(f"\n--- Per-session demand (threshold-increase aggregates) ---\n")
        fh.write(f"  {'mouse':>7} {'session':>8} {'n_inc':>5} "
                 f"{'tot_DRT':>9} {'max_DRT':>9} {'tot_HRdrop':>11}\n")
        for i, k in enumerate(keys):
            m, s = k
            fh.write(f"  {m:>7} {s:>8} {n_increases[i]:>5d} "
                     f"{total_expected_drt[i]:>9.2f} "
                     f"{max_expected_drt[i]:>9.2f} "
                     f"{total_hr_drop[i]:>11.2f}\n")
        fh.write(f"\n--- Correlations: demand vs CN summary stats "
                 f"(sessions with >=1 increase) ---\n")
        for xlabel, ylabel, r, p in cell8_correlations:
            fh.write(f"  {xlabel:25s} vs {ylabel:25s}  "
                     f"r = {r:+.3f}, p = {p:.2e}\n")
    print(f"Appended CELL 8 diagnostics to {diag_path}")

#%% ============================================================================
# CELL 9: Multiple linear regression — best predictor combo for CN slope
# ============================================================================
# Outcome: cn_slope (linear slope of CN F per trial, computed in CELL 7).
# Predictors: protocol-demand variables (CELL 8) + epoch-0 behavior (CELL 7).
# All predictors and outcome z-scored within the analysis sample so betas are
# standardized (partial correlations in z-units). Reports both multivariate
# and univariate betas with t-test p-values to surface collinearity.
# ============================================================================
from scipy.stats import t as _tdist

# Predictor matrix
_predictors = [
    ('total_expected_drt', total_expected_drt),
    ('max_expected_drt',   max_expected_drt),
    ('total_hr_drop',      total_hr_drop),
    ('n_increases',        n_increases.astype(float)),
    ('epoch0_mean_rt',     ep0_rt),
    ('epoch0_hit_rate',    ep0_hr),
    ('first_switch',       first_sw),
]
_pred_names = [p[0] for p in _predictors]
X_full = np.column_stack([p[1] for p in _predictors])
y_full = cn_slope.copy()

# Restrict to sessions with all-finite predictors + outcome + >=1 increase
_valid = np.all(np.isfinite(X_full), axis=1) & np.isfinite(y_full) & (n_increases > 0)
Xv = X_full[_valid]
yv = y_full[_valid]
n9, p9 = Xv.shape

# Z-score both predictors and outcome (so betas are standardized)
Xz = (Xv - Xv.mean(axis=0)) / Xv.std(axis=0, ddof=1)
yz = (yv - yv.mean()) / yv.std(ddof=1)

def _ols(X, y):
    """OLS with standard errors / t-tests / R^2."""
    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ b
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    df = n - p - 1
    sigma2 = rss / df
    XtXi = np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(sigma2 * np.diag(XtXi))
    tstat = b / se
    pvals = 2 * (1 - _tdist.cdf(np.abs(tstat), df=df))
    r2 = 1 - rss / tss
    adj_r2 = 1 - (rss / df) / (tss / (n - 1))
    f_stat = ((tss - rss) / p) / sigma2
    f_pval = 1 - _tdist.cdf(np.sqrt(np.clip(f_stat, 0, None)), df=df) * 2  # rough
    return {
        'b': b[1:], 'se': se[1:], 't': tstat[1:], 'p': pvals[1:],
        'b0': b[0], 'r2': r2, 'adj_r2': adj_r2, 'n': n, 'df': df,
    }

# Multivariate fit
mlr = _ols(Xz, yz)

# Univariate fits (one predictor at a time)
uni_b, uni_se, uni_p = (np.full(p9, np.nan) for _ in range(3))
for j in range(p9):
    res_j = _ols(Xz[:, [j]], yz)
    uni_b[j] = res_j['b'][0]
    uni_se[j] = res_j['se'][0]
    uni_p[j] = res_j['p'][0]

# Predictor correlation matrix (for collinearity inspection)
corr_mat = np.corrcoef(Xz.T)

# --- Figure 9: side-by-side multivariate vs univariate standardized betas ---
fig9_w, fig9_h = 7.0, 2.6
ax_w_in, ax_h_in = 2.6, 1.8
fig9 = plt.figure(figsize=(fig9_w, fig9_h))

def _bar_panel(ax, b, se, p, title):
    xpos = np.arange(p9)
    ax.bar(xpos, b, yerr=se, color='#888', edgecolor='black',
           linewidth=0.7, capsize=3)
    ax.axhline(0, color='k', linewidth=0.5)
    for j, pv in enumerate(p):
        tag = ('***' if pv < 0.001 else '**' if pv < 0.01 else
               '*' if pv < 0.05 else '')
        if tag:
            off = (se[j] * 1.3 + 0.02) * (1 if b[j] >= 0 else -1)
            ax.text(xpos[j], b[j] + off, tag, ha='center', fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(_pred_names, rotation=35, ha='right')
    ax.set_ylabel('Std. β (predict cn_slope)')
    ax.set_title(title, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axL = fig9.add_axes([0.08, 0.30, ax_w_in / fig9_w, ax_h_in / fig9_h])
_bar_panel(axL, mlr['b'], mlr['se'], mlr['p'],
           f"Multivariate (n={n9}, R²={mlr['r2']:.2f}, "
           f"adj R²={mlr['adj_r2']:.2f})")

axR = fig9.add_axes([0.56, 0.30, ax_w_in / fig9_w, ax_h_in / fig9_h])
_bar_panel(axR, uni_b, uni_se, uni_p,
           f"Univariate (n={n9})")

fname9 = 'cn_slope_mlr'
fig9.savefig(os.path.join(PANEL_DIR, f'{fname9}.png'), dpi=300)
fig9.savefig(os.path.join(PANEL_DIR, f'{fname9}.svg'))
plt.show()
print(f"Saved {fname9}")

# --- Append to diagnostics ---
with open(diag_path, 'a') as fh:
    fh.write("\n=== CELL 9: MLR predicting cn_slope ===\n")
    fh.write(f"n = {n9}, all-finite predictors + outcome + n_increases > 0\n")
    fh.write(f"Predictors and outcome z-scored within sample.\n\n")

    fh.write(f"--- Multivariate (all predictors jointly) ---\n")
    fh.write(f"R2 = {mlr['r2']:.3f}, adj R2 = {mlr['adj_r2']:.3f}, "
             f"df = {mlr['df']}\n")
    fh.write(f"  {'predictor':22s} {'beta':>9s} {'SE':>8s} {'t':>7s} {'p':>9s}\n")
    for j, name in enumerate(_pred_names):
        fh.write(f"  {name:22s} {mlr['b'][j]:>+9.3f} {mlr['se'][j]:>8.3f} "
                 f"{mlr['t'][j]:>+7.2f} {mlr['p'][j]:>9.3e}\n")

    fh.write(f"\n--- Univariate (each predictor alone) ---\n")
    fh.write(f"  {'predictor':22s} {'beta':>9s} {'SE':>8s} {'p':>9s}\n")
    for j, name in enumerate(_pred_names):
        fh.write(f"  {name:22s} {uni_b[j]:>+9.3f} {uni_se[j]:>8.3f} "
                 f"{uni_p[j]:>9.3e}\n")

    fh.write(f"\n--- Predictor correlation matrix (collinearity check) ---\n")
    fh.write(f"  {'':22s}")
    for name in _pred_names:
        fh.write(f" {name[:10]:>10s}")
    fh.write("\n")
    for i, name in enumerate(_pred_names):
        fh.write(f"  {name:22s}")
        for j in range(p9):
            fh.write(f" {corr_mat[i, j]:>+10.2f}")
        fh.write("\n")

print(f"Appended CELL 9 MLR to {diag_path}")
