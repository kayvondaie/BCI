#%% ============================================================================
# CELL 1: Imports and helpers
# ============================================================================
"""
Continuous-time eligibility-trace HI analysis, matching the modeling section
of the 3-factor learning paper.

For each pair, integrate the instantaneous eligibility
    E[p, t] = pre_g(t - PRE_LAG) * (df[post(p), t] - h_bar[post(p), t])
with an exponential leaky filter of timescale tau_elig, where h_bar is itself
an EMA of df_closedloop with timescale tau_bl.  The phi' gain term in the
paper's E_{ij,t} is dropped.

Eligibility accumulates continuously across trial boundaries.  Optionally,
instantaneous E is zeroed during intertrial frames (frames after the trial's
first reward, before the next trial_start).

Per-window CC (sum over frames in the window of E_bar) is regressed across
pairs against dW = AMP[1] - AMP[0] to yield HI slope and intercept per window.
Candidate RPE signals (trial-based and frame-based, with separate time
constants for steps and rewards) are computed per window for post-hoc
correlation with HI slope.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon
from scipy.signal import lfilter
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *
import bci_time_series as bts

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
# Mirror figures to the OneDrive panels folder used by the paper figure scripts.
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')
os.makedirs(PANEL_DIR, exist_ok=True)


def _save_panel(fig_path):
    """Mirror a saved figure into the OneDrive panel directory."""
    try:
        import shutil
        shutil.copy(fig_path, os.path.join(PANEL_DIR, os.path.basename(fig_path)))
    except Exception as e:
        print(f"  (could not mirror to panels: {e})")

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
})


def ema_causal(x, tau_frames, axis=-1, init=None):
    """Causal EMA along axis via scipy.signal.lfilter.

    Recurrence:  y[t] = (1 - 1/tau_frames) y[t-1] + (1/tau_frames) x[t]
    init: value of y at the first index (default: x at first index).
    """
    x = np.asarray(x, dtype=float)
    alpha = 1.0 / float(tau_frames)
    b = np.array([alpha])
    a = np.array([1.0, -(1.0 - alpha)])
    x_first = np.take(x, 0, axis=axis)
    if init is None:
        init_arr = x_first
    else:
        init_arr = np.broadcast_to(np.asarray(init, dtype=float), x_first.shape)
    # For a 1st-order filter, zi has length 1 along the filtering axis.
    # Choosing zi = init - alpha * x[0] yields y[0] = init exactly.
    zi = np.expand_dims(init_arr - alpha * x_first, axis=axis)
    y, _ = lfilter(b, a, x, axis=axis, zi=zi)
    return y


def compute_iti_flag(trial_start_vector, reward_vector):
    """Per-frame ITI flag = True for frames after the first reward in a trial
    (up to but not including the next trial start)."""
    total_frames = len(trial_start_vector)
    iti = np.zeros(total_frames, dtype=bool)
    trial_starts = np.where(trial_start_vector > 0)[0]
    for i, ts in enumerate(trial_starts):
        te = trial_starts[i + 1] if i + 1 < len(trial_starts) else total_frames
        rewards_in_trial = np.where(reward_vector[ts:te] > 0)[0]
        if len(rewards_in_trial) > 0:
            rf = ts + rewards_in_trial[0]
            iti[rf + 1:te] = True
    return iti


print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

WIN_SIZE = 10
WIN_STEP = 5

# Time constants (seconds).  Defaults inspired by the paper's modeling section.
TAU_ELIG = 2.0
TAU_BL   = 360.0
TAU_REW  = 30.0
TAU_STEP = 2.0

# Pre lag in frames (paper uses h_{j, t-1}).
PRE_LAG_FRAMES = 1

# Zero out instantaneous eligibility during intertrial frames.  Eligibility
# accumulator (E_bar) still leaks during ITI but receives no contribution.
ZERO_ITI = False

# For the trial-based RPE signals (kept for v2 comparability).
TAU_ELIG_TRIAL = 10

# Tag appended to all output filenames so parameter sweeps don't overwrite.
CONFIG_TAG = (f"_te{TAU_ELIG:g}_tb{TAU_BL:g}_tr{TAU_REW:g}_ts{TAU_STEP:g}"
              f"_lag{PRE_LAG_FRAMES}_iti{int(ZERO_ITI)}")

print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"tau_elig={TAU_ELIG}s, tau_bl={TAU_BL}s, "
      f"tau_rew={TAU_REW}s, tau_step={TAU_STEP}s")
print(f"pre_lag={PRE_LAG_FRAMES} frame(s), zero_iti={ZERO_ITI}")
print(f"Output tag: {CONFIG_TAG}")

all_results = []

#%% ============================================================================
# CELL 3: Main loop
# ============================================================================
import csv
_qc_csv = os.path.join(RESULTS_DIR, 'qc', 'qc_summary.csv')
_qc_fail = set()
if os.path.exists(_qc_csv):
    with open(_qc_csv) as _f:
        for _r in csv.DictReader(_f):
            if _r['pass_qc'] != 'True':
                _qc_fail.add((_r['mouse'], _r['session']))
    print(f"QC filter: {len(_qc_fail)} sessions excluded")
else:
    print("WARNING: qc_summary.csv not found, no sessions excluded")

for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]

    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            if (mouse, session) in _qc_fail:
                print(f"  Skipping {mouse} {session} -- failed QC")
                continue
            folder = (r'//allen/aind/scratch/BCI/2p-raw/'
                      + mouse + r'/' + session + '/pophys/')
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = [
                'df_closedloop', 'F', 'mouse', 'session',
                'conditioned_neuron', 'dt_si', 'step_time',
                'reward_time', 'BCI_thresholds',
            ]
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"  Skipping -- file not found.")
                continue

            BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
            thr = BCI_thresholds[1, :]
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i - 1]
            if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
                thr[0] = thr[np.isfinite(thr)][0]
            BCI_thresholds[1, :] = thr

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # Per-trial behavior (kept for trial-based RPEs)
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=TAU_ELIG_TRIAL, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0,
                                  tau=TAU_ELIG_TRIAL, fill_value=0.0)

            # ---- Continuous closed-loop data ----
            df_full = np.asarray(data['df_closedloop'], dtype=float)
            df_full[np.isnan(df_full)] = 0.0
            n_neurons_df, total_frames = df_full.shape
            if n_neurons_df != n_neurons:
                print(f"  WARNING: df_closedloop has {n_neurons_df} neurons "
                      f"but F has {n_neurons}; skipping.")
                continue

            step_vector, reward_vector, trial_start_vector = (
                bts.bci_time_series_fun(folder, data, rt_filled, dt_si))
            if len(step_vector) != total_frames:
                print(f"  WARNING: vector length {len(step_vector)} != "
                      f"df_closedloop frames {total_frames}; truncating.")
                tf = min(len(step_vector), total_frames)
                step_vector = step_vector[:tf]
                reward_vector = reward_vector[:tf]
                trial_start_vector = trial_start_vector[:tf]
                df_full = df_full[:, :tf]
                total_frames = tf

            # ---- Pair selection (same as v2) ----
            dist_target_lt = 10
            dist_nontarg_min = 30
            dist_nontarg_max = 1000
            amp0_thr = 0.1
            amp1_thr = 0.1

            dw_list = []
            pair_cl_list = []
            pair_nt_list = []

            for gi in range(stimDist.shape[1]):
                cl = np.where(
                    (stimDist[:, gi] < dist_target_lt) &
                    (AMP[0][:, gi] > amp0_thr) &
                    (AMP[1][:, gi] > amp1_thr)
                )[0]
                if cl.size == 0:
                    continue
                nontarg = np.where(
                    (stimDist[:, gi] > dist_nontarg_min) &
                    (stimDist[:, gi] < dist_nontarg_max)
                )[0]
                if nontarg.size == 0:
                    continue
                dw = AMP[1][nontarg, gi] - AMP[0][nontarg, gi]
                dw_list.append(dw)
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            n_pairs = len(Y_T)

            # ---- Sliding windows (trial-based, mapped to frame ranges) ----
            trial_start_frames = np.where(trial_start_vector > 0)[0]
            if len(trial_start_frames) < trl:
                print(f"  Warning: only {len(trial_start_frames)} trial starts "
                      f"in vector vs {trl} trials in F; clamping.")
                trl_eff = len(trial_start_frames)
            else:
                trl_eff = trl

            win_starts_trial = np.arange(0, trl_eff - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts_trial)
            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            win_frame_ranges = []
            for ws in win_starts_trial:
                we = ws + WIN_SIZE
                f0 = int(trial_start_frames[ws])
                f1 = (int(trial_start_frames[we])
                      if we < len(trial_start_frames) else total_frames)
                win_frame_ranges.append((f0, f1))

            # ---- Continuous baselines (post and behavior) ----
            tau_bl_frames = TAU_BL / dt_si
            tau_rew_frames = TAU_REW / dt_si
            tau_step_frames = TAU_STEP / dt_si
            tau_elig_frames = TAU_ELIG / dt_si

            h_bar = ema_causal(df_full, tau_bl_frames, axis=-1,
                               init=df_full[:, 0])
            dev = df_full - h_bar

            reward_baseline = ema_causal(
                reward_vector.astype(float), tau_rew_frames, init=0.0)
            step_baseline = ema_causal(
                step_vector.astype(float), tau_step_frames, init=0.0)
            rew_rpe_frame = reward_vector.astype(float) - reward_baseline
            step_rpe_frame = step_vector.astype(float) - step_baseline

            # ---- ITI flag ----
            iti_flag = (compute_iti_flag(trial_start_vector, reward_vector)
                        if ZERO_ITI
                        else np.zeros(total_frames, dtype=bool))

            # ---- Per-group eligibility accumulation ----
            cc_arr = np.zeros((n_wins, n_pairs))
            pair_offset = 0
            for gi_idx in range(len(dw_list)):
                cl = pair_cl_list[gi_idx][0]
                nontarg = pair_nt_list[gi_idx]
                n_nt = len(nontarg)

                # Pre activity per frame (averaged over this group's cl set)
                pre_g = df_full[cl, :].mean(axis=0)
                # Lag pre by PRE_LAG_FRAMES (paper uses h_{j, t-1})
                pre_g_lag = np.empty_like(pre_g)
                pre_g_lag[:PRE_LAG_FRAMES] = 0.0
                pre_g_lag[PRE_LAG_FRAMES:] = pre_g[:-PRE_LAG_FRAMES] if PRE_LAG_FRAMES > 0 else pre_g

                # Instantaneous eligibility per nontarget per frame
                dev_nt = dev[nontarg, :]                       # (n_nt, T)
                E_g = pre_g_lag[None, :] * dev_nt
                if ZERO_ITI:
                    E_g[:, iti_flag] = 0.0

                # Accumulated eligibility (causal EMA along time)
                E_bar_g = ema_causal(E_g, tau_elig_frames, axis=-1, init=0.0)

                # Sum E_bar over each window's frame range
                for wi, (f0, f1) in enumerate(win_frame_ranges):
                    cc_arr[wi, pair_offset:pair_offset + n_nt] = (
                        E_bar_g[:, f0:f1].sum(axis=1))

                pair_offset += n_nt

            # ---- Per-window behavioral aggregates ----
            win_hit = np.full(n_wins, np.nan)
            win_rpe = np.full(n_wins, np.nan)
            win_rt = np.full(n_wins, np.nan)
            win_hit_rpe = np.full(n_wins, np.nan)
            win_rew_rpe = np.full(n_wins, np.nan)
            win_step_rpe = np.full(n_wins, np.nan)
            win_center = np.full(n_wins, np.nan)

            for wi, ws in enumerate(win_starts_trial):
                we = ws + WIN_SIZE
                trial_idx = np.arange(ws, we)
                win_center[wi] = (ws + we) / 2.0
                win_hit[wi] = np.nanmean(hit[trial_idx])
                win_rt[wi] = np.nanmean(rt_filled[trial_idx])
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx])
                win_hit_rpe[wi] = np.nanmean(hit_rpe[trial_idx])
                f0, f1 = win_frame_ranges[wi]
                win_rew_rpe[wi] = np.nanmean(rew_rpe_frame[f0:f1])
                win_step_rpe[wi] = np.nanmean(step_rpe_frame[f0:f1])

            # ---- HI fits per window ----
            hi_no_int = np.full(n_wins, np.nan)
            hi_with_int = np.full(n_wins, np.nan)
            hi_intercept = np.full(n_wins, np.nan)
            hi_corr = np.full(n_wins, np.nan)

            for wi in range(n_wins):
                cc_pair = cc_arr[wi, :]
                if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                    continue
                hi_no_int[wi] = (
                    np.dot(cc_pair, Y_T) / np.dot(cc_pair, cc_pair))
                A = np.column_stack([np.ones(n_pairs), cc_pair])
                coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                hi_intercept[wi] = coeffs[0]
                hi_with_int[wi] = coeffs[1]
                hi_corr[wi], _ = pearsonr(cc_pair, Y_T)

            result = {
                'mouse': mouse,
                'session': session,
                'n_pairs': n_pairs,
                'n_trials': trl,
                'n_trials_used': trl_eff,
                'n_windows': n_wins,
                'n_frames': total_frames,
                'dt_si': dt_si,
                'tau_elig': TAU_ELIG,
                'tau_bl': TAU_BL,
                'tau_rew': TAU_REW,
                'tau_step': TAU_STEP,
                'zero_iti': ZERO_ITI,
                'pre_lag_frames': PRE_LAG_FRAMES,
                'win_centers': win_center,
                'hit_rate': float(np.nanmean(hit)),
                'hi_no_int': hi_no_int,
                'hi_with_int': hi_with_int,
                'hi_intercept': hi_intercept,
                'hi_corr': hi_corr,
                'win_hit': win_hit,
                'win_rpe': win_rpe,
                'win_rt': win_rt,
                'win_hit_rpe': win_hit_rpe,
                'win_rew_rpe': win_rew_rpe,
                'win_step_rpe': win_step_rpe,
                'cc_arr': cc_arr,
                'Y_T': Y_T,
                # Raw signals retained for fast tau_rew / tau_step sweeps
                'step_vector': step_vector.astype(np.uint8),
                'reward_vector': reward_vector.astype(np.uint8),
                'win_frame_ranges': np.asarray(win_frame_ranges, dtype=np.int64),
            }
            all_results.append(result)
            print(f"  {n_wins} windows, {n_pairs} pairs, "
                  f"{total_frames} frames")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nTotal sessions: {len(all_results)}")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR,
        f'sliding_window_continuous_eligibility{CONFIG_TAG}.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR,
                 f'sliding_window_continuous_eligibility{CONFIG_TAG}.npy'),
    allow_pickle=True).tolist()
print(f"Loaded {len(all_results)} sessions")

#%% ============================================================================
# CELL 5b: Sweep tau_rew and tau_step (post-hoc, no main-loop rerun)
# ============================================================================
# Requires step_vector, reward_vector, win_frame_ranges saved per session.
# For each (tau_rew, tau_step) combination, recompute per-frame RPE baselines,
# aggregate into per-window means, then compute the session-wise Spearman
# correlation of HI slope with (rew_RPE, step_RPE) and the Wilcoxon p-value
# across sessions.  Plots a heatmap over the tau grid.

TAU_REW_SWEEP = [2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0]
TAU_STEP_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]

# Match Cell 9's truncation (set to None for all windows).
SWEEP_TRIAL_CUTOFF = None


def _win_means(frame_signal, win_frame_ranges):
    out = np.empty(len(win_frame_ranges), dtype=float)
    for wi, (f0, f1) in enumerate(win_frame_ranges):
        if f1 > f0:
            out[wi] = float(np.mean(frame_signal[f0:f1]))
        else:
            out[wi] = np.nan
    return out


def _session_corr(slope, beh, win_starts_trial, trial_cutoff):
    if trial_cutoff is not None:
        mask = win_starts_trial < trial_cutoff
        slope = slope[mask]
        beh = beh[mask]
    ok = np.isfinite(slope) & np.isfinite(beh)
    if np.sum(ok) < 5 or np.std(slope[ok]) == 0 or np.std(beh[ok]) == 0:
        return np.nan
    rho, _ = spearmanr(slope[ok], beh[ok])
    return rho


# Precompute per session: dt_si, win_starts (trial), step/reward as float
_sess = []
for s in all_results:
    sv = np.asarray(s['step_vector'], dtype=float)
    rv = np.asarray(s['reward_vector'], dtype=float)
    wfr = np.asarray(s['win_frame_ranges'], dtype=np.int64)
    ws = s['win_centers'] - WIN_SIZE / 2.0
    _sess.append({
        'sv': sv, 'rv': rv, 'wfr': wfr,
        'ws': ws, 'dt_si': float(s['dt_si']),
        'slope': np.asarray(s['hi_with_int'], dtype=float),
    })

rew_rho_mat = np.full((len(TAU_REW_SWEEP), len(TAU_STEP_SWEEP)), np.nan)
rew_p_mat = np.full_like(rew_rho_mat, np.nan)
step_rho_mat = np.full_like(rew_rho_mat, np.nan)
step_p_mat = np.full_like(rew_rho_mat, np.nan)

for ri, tau_r in enumerate(TAU_REW_SWEEP):
    for ci, tau_s in enumerate(TAU_STEP_SWEEP):
        rew_rhos, step_rhos = [], []
        for sess in _sess:
            tau_r_frames = tau_r / sess['dt_si']
            tau_s_frames = tau_s / sess['dt_si']
            rew_bl = ema_causal(sess['rv'], tau_r_frames, init=0.0)
            step_bl = ema_causal(sess['sv'], tau_s_frames, init=0.0)
            rew_rpe = sess['rv'] - rew_bl
            step_rpe = sess['sv'] - step_bl
            win_rew = _win_means(rew_rpe, sess['wfr'])
            win_step = _win_means(step_rpe, sess['wfr'])
            rew_rhos.append(
                _session_corr(sess['slope'], win_rew, sess['ws'],
                              SWEEP_TRIAL_CUTOFF))
            step_rhos.append(
                _session_corr(sess['slope'], win_step, sess['ws'],
                              SWEEP_TRIAL_CUTOFF))
        rr = np.asarray(rew_rhos)
        sr = np.asarray(step_rhos)
        rr = rr[np.isfinite(rr)]
        sr = sr[np.isfinite(sr)]
        if len(rr) >= 3:
            rew_rho_mat[ri, ci] = float(np.mean(rr))
            try:
                _, rew_p_mat[ri, ci] = wilcoxon(rr)
            except Exception:
                pass
        if len(sr) >= 3:
            step_rho_mat[ri, ci] = float(np.mean(sr))
            try:
                _, step_p_mat[ri, ci] = wilcoxon(sr)
            except Exception:
                pass

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, mat, pmat, title in [
    (axes[0], rew_rho_mat, rew_p_mat, 'slope vs rew_RPE'),
    (axes[1], step_rho_mat, step_p_mat, 'slope vs step_RPE'),
]:
    vmax = np.nanmax(np.abs(mat)) if np.any(np.isfinite(mat)) else 0.1
    vmax = max(vmax, 0.05)
    im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                   aspect='auto', interpolation='nearest')
    for ri in range(len(TAU_REW_SWEEP)):
        for ci in range(len(TAU_STEP_SWEEP)):
            v = mat[ri, ci]
            p = pmat[ri, ci]
            if np.isnan(v):
                continue
            sig = ('***' if p < 0.001 else
                   '**'  if p < 0.01  else
                   '*'   if p < 0.05  else '')
            txt = f'{v:+.3f}'
            if sig:
                txt += f'\n{sig}'
            ax.text(ci, ri, txt, ha='center', va='center',
                    fontsize=9, fontweight='bold' if sig else 'normal')
    ax.set_xticks(range(len(TAU_STEP_SWEEP)))
    ax.set_xticklabels([f'{t:g}' for t in TAU_STEP_SWEEP])
    ax.set_yticks(range(len(TAU_REW_SWEEP)))
    ax.set_yticklabels([f'{t:g}' for t in TAU_REW_SWEEP])
    ax.set_xlabel('tau_step (s)')
    ax.set_ylabel('tau_rew (s)')
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='mean rho')

cutoff_label = (f', first {SWEEP_TRIAL_CUTOFF} trials'
                if SWEEP_TRIAL_CUTOFF is not None else '')
fig.suptitle(
    f'tau sweep over RPE baselines (n={len(_sess)} sessions{cutoff_label})\n'
    f'cc_arr from tau_elig={TAU_ELIG}s, tau_bl={TAU_BL}s, '
    f'zero_iti={ZERO_ITI}, lag={PRE_LAG_FRAMES}',
    fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
sweep_tag = (f'_cut{SWEEP_TRIAL_CUTOFF}'
             if SWEEP_TRIAL_CUTOFF is not None else '')
_p = os.path.join(RESULTS_DIR,
        f'fig_continuous_eligibility_tausweep{CONFIG_TAG}{sweep_tag}.png')
plt.savefig(_p, dpi=150, bbox_inches='tight')
_save_panel(_p)
plt.show()
print("Sweep figure saved.")

# ---- Report optimum tau for each predictor ----
def _report_optimum(mat, pmat, taus_y, taus_x, label):
    flat_idx = np.nanargmax(mat)
    ri, ci = np.unravel_index(flat_idx, mat.shape)
    rho = mat[ri, ci]
    p = pmat[ri, ci]
    print(f"  {label:24s} max rho = {rho:+.4f}  "
          f"(tau_rew={taus_y[ri]:g}s, tau_step={taus_x[ci]:g}s)  "
          f"Wilcoxon p={p:.4f}")

print("\nOptimum tau combinations:")
_report_optimum(rew_rho_mat, rew_p_mat,
                TAU_REW_SWEEP, TAU_STEP_SWEEP, "slope vs rew_RPE")
_report_optimum(step_rho_mat, step_p_mat,
                TAU_REW_SWEEP, TAU_STEP_SWEEP, "slope vs step_RPE")

#%% ============================================================================
# CELL 6: Within-session correlations (HI slope/intercept vs candidate RPEs)
# ============================================================================
beh_names = ['hit_rate', 'RT', 'RPE_trial', 'hit_RPE_trial',
             'rew_RPE_frame', 'step_RPE_frame',
             'reward_rate', 'step_rate']
beh_labels = ['Hit rate', 'Reaction time', 'RPE (trial)', 'Hit RPE (trial)',
              'Reward RPE (frame)', 'Step RPE (frame)',
              'Reward rate', 'Step rate']
n_beh = len(beh_names)

# Compute raw frame-level reward and step rates per window from the stored
# step/reward vectors (tau-independent, no main-loop rerun needed).  These get
# attached to each result dict for use by get_beh below.
for _s in all_results:
    _sv = np.asarray(_s['step_vector'], dtype=float)
    _rv = np.asarray(_s['reward_vector'], dtype=float)
    _wfr = np.asarray(_s['win_frame_ranges'], dtype=np.int64)
    _ws_rate = np.empty(len(_wfr), dtype=float)
    _rw_rate = np.empty(len(_wfr), dtype=float)
    for _wi, (_f0, _f1) in enumerate(_wfr):
        if _f1 > _f0:
            _ws_rate[_wi] = float(np.mean(_sv[_f0:_f1]))
            _rw_rate[_wi] = float(np.mean(_rv[_f0:_f1]))
        else:
            _ws_rate[_wi] = np.nan
            _rw_rate[_wi] = np.nan
    _s['win_step_rate'] = _ws_rate
    _s['win_reward_rate'] = _rw_rate


def get_beh(s, bname):
    return {
        'hit_rate': s['win_hit'],
        'RT': s['win_rt'],
        'RPE_trial': s['win_rpe'],
        'hit_RPE_trial': s['win_hit_rpe'],
        'rew_RPE_frame': s['win_rew_rpe'],
        'step_RPE_frame': s['win_step_rpe'],
        'reward_rate': s['win_reward_rate'],
        'step_rate': s['win_step_rate'],
    }[bname]


n_s = len(all_results)
corr_slope = np.full((n_s, n_beh), np.nan)
corr_intercept = np.full((n_s, n_beh), np.nan)

for si, s in enumerate(all_results):
    for bi, bname in enumerate(beh_names):
        bvar = get_beh(s, bname)
        if np.sum(np.isfinite(bvar)) < 5 or np.std(bvar[np.isfinite(bvar)]) == 0:
            continue
        slope = s['hi_with_int']
        intercept = s['hi_intercept']
        ok = np.isfinite(bvar) & np.isfinite(slope)
        if np.sum(ok) >= 5 and np.std(slope[ok]) > 0:
            corr_slope[si, bi], _ = spearmanr(bvar[ok], slope[ok])
        ok2 = np.isfinite(bvar) & np.isfinite(intercept)
        if np.sum(ok2) >= 5 and np.std(intercept[ok2]) > 0:
            corr_intercept[si, bi], _ = spearmanr(bvar[ok2], intercept[ok2])

print("Within-session correlations computed.")

#%% ============================================================================
# CELL 6.5: Override tau_rew / tau_step and refresh win_*rpe + correlations
# ============================================================================
# Recomputes the frame-based reward and step RPE baselines per session using
# user-supplied tau values, updates each result dict's win_rew_rpe and
# win_step_rpe in place, then re-runs the within-session correlation arrays.
# Subsequent cells (7, 9) will then plot using the override values.
#
# Set both to None to skip and keep whatever was loaded.

OVERRIDE_TAU_REW = 600.0   # seconds; e.g., set to the sweep optimum
OVERRIDE_TAU_STEP = 2.0   # seconds; e.g., set to the sweep optimum

# Always-defined effective values + a tag that downstream cells will pick up.
eff_tau_rew = OVERRIDE_TAU_REW if OVERRIDE_TAU_REW is not None else TAU_REW
eff_tau_step = (OVERRIDE_TAU_STEP if OVERRIDE_TAU_STEP is not None
                else TAU_STEP)
override_tag = ""
if OVERRIDE_TAU_REW is not None:
    override_tag += f"_otr{OVERRIDE_TAU_REW:g}"
if OVERRIDE_TAU_STEP is not None:
    override_tag += f"_ots{OVERRIDE_TAU_STEP:g}"

if OVERRIDE_TAU_REW is not None or OVERRIDE_TAU_STEP is not None:
    print(f"Overriding tau_rew={eff_tau_rew}s, tau_step={eff_tau_step}s")

    for s in all_results:
        dt = float(s['dt_si'])
        sv = np.asarray(s['step_vector'], dtype=float)
        rv = np.asarray(s['reward_vector'], dtype=float)
        wfr = np.asarray(s['win_frame_ranges'], dtype=np.int64)

        if OVERRIDE_TAU_REW is not None:
            rew_bl = ema_causal(rv, OVERRIDE_TAU_REW / dt, init=0.0)
            rew_rpe = rv - rew_bl
            wr = np.empty(len(wfr), dtype=float)
            for wi, (f0, f1) in enumerate(wfr):
                wr[wi] = float(np.mean(rew_rpe[f0:f1])) if f1 > f0 else np.nan
            s['win_rew_rpe'] = wr

        if OVERRIDE_TAU_STEP is not None:
            step_bl = ema_causal(sv, OVERRIDE_TAU_STEP / dt, init=0.0)
            step_rpe = sv - step_bl
            ws = np.empty(len(wfr), dtype=float)
            for wi, (f0, f1) in enumerate(wfr):
                ws[wi] = (float(np.mean(step_rpe[f0:f1]))
                          if f1 > f0 else np.nan)
            s['win_step_rpe'] = ws

    # Re-run the within-session correlation loop with updated win_*rpe
    corr_slope = np.full((n_s, n_beh, n_epochs if False else 1), np.nan)
    # (Cell 6's arrays were 2D (n_s, n_beh); restore that shape)
    corr_slope = np.full((n_s, n_beh), np.nan)
    corr_intercept = np.full((n_s, n_beh), np.nan)

    for si, s in enumerate(all_results):
        for bi, bname in enumerate(beh_names):
            bvar = get_beh(s, bname)
            if (np.sum(np.isfinite(bvar)) < 5
                    or np.std(bvar[np.isfinite(bvar)]) == 0):
                continue
            slope = s['hi_with_int']
            intercept = s['hi_intercept']
            ok = np.isfinite(bvar) & np.isfinite(slope)
            if np.sum(ok) >= 5 and np.std(slope[ok]) > 0:
                corr_slope[si, bi], _ = spearmanr(bvar[ok], slope[ok])
            ok2 = np.isfinite(bvar) & np.isfinite(intercept)
            if np.sum(ok2) >= 5 and np.std(intercept[ok2]) > 0:
                corr_intercept[si, bi], _ = spearmanr(
                    bvar[ok2], intercept[ok2])

    # Print quick before/after for the two affected rows
    for bi, bname in enumerate(beh_names):
        if bname not in ('rew_RPE_frame', 'step_RPE_frame'):
            continue
        v = corr_slope[:, bi]
        v = v[np.isfinite(v)]
        if len(v) < 3:
            continue
        try:
            _, p = wilcoxon(v)
        except Exception:
            p = 1.0
        print(f"  slope vs {bname:18s} "
              f"mean rho={np.mean(v):+.4f}, p={p:.4f}")
else:
    print("OVERRIDE_TAU_REW and OVERRIDE_TAU_STEP both None; skipping.")

#%% ============================================================================
# CELL 7: Slope/intercept summary (behavior x {slope, intercept})
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(8, max(5, 0.7 * n_beh + 1)))

for col, (corr_arr, row_label) in enumerate([
    (corr_slope, 'Slope'),
    (corr_intercept, 'Intercept'),
]):
    ax = axes[col]
    vec_mean = np.full(n_beh, np.nan)
    vec_p = np.full(n_beh, np.nan)
    for bi in range(n_beh):
        vals = corr_arr[:, bi]
        v = vals[np.isfinite(vals)]
        if len(v) < 3:
            continue
        vec_mean[bi] = np.mean(v)
        try:
            _, p = wilcoxon(v)
        except Exception:
            p = 1.0
        vec_p[bi] = p

    vmax = np.nanmax(np.abs(vec_mean)) if np.any(np.isfinite(vec_mean)) else 0.2
    vmax = max(vmax, 0.05)
    im = ax.imshow(vec_mean[:, None], cmap='coolwarm',
                   vmin=-vmax, vmax=vmax,
                   aspect='auto', interpolation='nearest')
    for bi in range(n_beh):
        val = vec_mean[bi]
        p = vec_p[bi]
        if np.isnan(val):
            continue
        sig = ('***' if p < 0.001 else
               '**'  if p < 0.01  else
               '*'   if p < 0.05  else '')
        txt = f'{val:+.3f}'
        if sig:
            txt += f'\n{sig}'
        ax.text(0, bi, txt, ha='center', va='center',
                fontsize=10, fontweight='bold' if sig else 'normal')

    ax.set_xticks([0])
    ax.set_xticklabels(['Continuous'], rotation=0)
    ax.set_yticks(range(n_beh))
    ax.set_yticklabels(beh_labels)
    plt.colorbar(im, ax=ax, shrink=0.7, label='Mean rho')
    ax.set_title(f'{row_label}', fontsize=13, fontweight='bold')

_eff_tau_rew = globals().get('eff_tau_rew', TAU_REW)
_eff_tau_step = globals().get('eff_tau_step', TAU_STEP)
_override_tag = globals().get('override_tag', '')
fig.suptitle(
    f'Continuous-time eligibility HI (n={n_s} sessions)\n'
    f'tau_elig={TAU_ELIG}s, tau_bl={TAU_BL}s, '
    f'tau_rew={_eff_tau_rew}s, tau_step={_eff_tau_step}s, '
    f'zero_iti={ZERO_ITI}',
    fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
_p = os.path.join(RESULTS_DIR,
        f'fig_continuous_eligibility_matrix{CONFIG_TAG}{_override_tag}.png')
plt.savefig(_p, dpi=150, bbox_inches='tight')
_save_panel(_p)
plt.show()
print("Figure saved.")

#%% ============================================================================
# CELL 8: Text report
# ============================================================================
_eff_tau_rew = globals().get('eff_tau_rew', TAU_REW)
_eff_tau_step = globals().get('eff_tau_step', TAU_STEP)
_override_tag = globals().get('override_tag', '')
report_path = os.path.join(
    RESULTS_DIR,
    f'continuous_eligibility_report{CONFIG_TAG}{_override_tag}.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("CONTINUOUS-TIME ELIGIBILITY HI ANALYSIS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"tau_elig={TAU_ELIG}s, tau_bl={TAU_BL}s, "
            f"tau_rew={_eff_tau_rew}s, tau_step={_eff_tau_step}s\n")
    f.write(f"pre_lag={PRE_LAG_FRAMES} frame(s), zero_iti={ZERO_ITI}\n")
    f.write("=" * 70 + "\n\n")
    f.write("CC per window = sum_{frames in window} E_bar[p, t], where\n")
    f.write("  E[p, t]     = pre_g(t - PRE_LAG) * (df[post, t] - h_bar[post, t])\n")
    f.write("  h_bar[n, t] = EMA(df_closedloop[n, t], tau_bl), init at first frame\n")
    f.write("  E_bar[p, t] = EMA(E[p, t], tau_elig), init at 0\n")
    f.write("If ZERO_ITI: E[:, t] = 0 for frames after the trial's first reward.\n\n")

    for target, corr_arr, label in [
        ('slope', corr_slope, 'BEHAVIOR vs SLOPE'),
        ('intercept', corr_intercept, 'BEHAVIOR vs INTERCEPT'),
    ]:
        f.write(f"\n{label}\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'behavior':25s} {'mean':>7s} {'median':>7s} "
                f"{'%>0':>5s} {'Wilcoxon p':>10s} {'sig':>4s}\n")
        for bi, bname in enumerate(beh_names):
            vals = corr_arr[:, bi]
            v = vals[np.isfinite(vals)]
            m = np.mean(v) if len(v) > 0 else np.nan
            md = np.median(v) if len(v) > 0 else np.nan
            fpos = np.mean(v > 0) * 100 if len(v) > 0 else np.nan
            try:
                _, p = wilcoxon(v)
            except Exception:
                p = 1.0
            sig = ('***' if p < 0.001 else
                   '**'  if p < 0.01  else
                   '*'   if p < 0.05  else '')
            f.write(f"  {bname:25s} {m:+7.3f} {md:+7.3f} "
                    f"{fpos:4.0f}% {p:10.4f} {sig:>4s}\n")
        f.write("\n")
print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 9: Binned HI slope vs each candidate behavior (within-session z)
# ============================================================================
n_bins = 3
# Restrict to early trials (set to None to use all windows).
# Includes windows whose start trial < TRIAL_CUTOFF.
TRIAL_CUTOFF = None

_n_cols = 4
_n_rows = int(np.ceil(len(beh_names) / _n_cols))
fig, axes = plt.subplots(_n_rows, _n_cols, figsize=(5 * _n_cols, 4 * _n_rows))
axes = np.atleast_1d(axes).flatten()
# Hide any unused subplots
for _ax in axes[len(beh_names):]:
    _ax.set_axis_off()

for bi, (bname, blabel) in enumerate(zip(beh_names, beh_labels)):
    ax = axes[bi]
    all_beh_z = []
    all_slope_z = []
    for s in all_results:
        bvar = get_beh(s, bname)
        slope = s['hi_with_int']
        if TRIAL_CUTOFF is not None:
            # win_centers[wi] = (ws + ws + WIN_SIZE) / 2 = ws + WIN_SIZE/2
            ws = s['win_centers'] - WIN_SIZE / 2.0
            trial_mask = ws < TRIAL_CUTOFF
            bvar = bvar[trial_mask]
            slope = slope[trial_mask]
        ok = np.isfinite(bvar) & np.isfinite(slope)
        if np.sum(ok) < 3:
            continue
        bvar_ok = bvar[ok]
        slope_ok = slope[ok]
        if np.std(bvar_ok) == 0 or np.std(slope_ok) == 0:
            continue
        all_beh_z.append((bvar_ok - np.mean(bvar_ok)) / np.std(bvar_ok))
        all_slope_z.append((slope_ok - np.mean(slope_ok)) / np.std(slope_ok))
    if len(all_beh_z) == 0:
        ax.set_axis_off()
        continue
    all_beh_z = np.concatenate(all_beh_z)
    all_slope_z = np.concatenate(all_slope_z)

    # ---- Pooled stats on the within-session z-scored data ----
    # Spearman rho is naive (treats every (session, window) point as
    # independent; in reality adjacent windows within a session share trials).
    rho_pool, p_pool = spearmanr(all_beh_z, all_slope_z)
    # Permutation null: shuffle the *behavior* time series within each session
    # while preserving the slope time series, then recompute the pooled rho.
    # This breaks the within-session alignment without destroying the
    # within-session autocorrelation of either signal.
    n_perm = 2000
    rng = np.random.default_rng(0)
    perm_rhos = np.empty(n_perm)
    # Need per-session boundaries to do the within-session shuffle.
    boundaries = np.cumsum([len(b) for b in
                            [(np.full_like(_b, np.nan)) for _b in []]])  # placeholder
    # Recover boundaries from the original concat order:
    _lens = []
    for s in all_results:
        bvar = get_beh(s, bname)
        slope = s['hi_with_int']
        if TRIAL_CUTOFF is not None:
            ws = s['win_centers'] - WIN_SIZE / 2.0
            tm = ws < TRIAL_CUTOFF
            bvar = bvar[tm]; slope = slope[tm]
        ok = np.isfinite(bvar) & np.isfinite(slope)
        if np.sum(ok) < 3:
            continue
        if np.std(bvar[ok]) == 0 or np.std(slope[ok]) == 0:
            continue
        _lens.append(int(np.sum(ok)))
    sess_offsets = np.r_[0, np.cumsum(_lens)]
    for pi in range(n_perm):
        beh_shuf = all_beh_z.copy()
        for k in range(len(_lens)):
            seg = slice(sess_offsets[k], sess_offsets[k + 1])
            rng.shuffle(beh_shuf[seg])
        perm_rhos[pi], _ = spearmanr(beh_shuf, all_slope_z)
    p_perm = float(np.mean(np.abs(perm_rhos) >= abs(rho_pool)))

    bin_edges = np.percentile(all_beh_z, np.linspace(0, 100, n_bins + 1))
    bc, bm, bs = [], [], []
    for bbi in range(n_bins):
        if bbi < n_bins - 1:
            mask = (all_beh_z >= bin_edges[bbi]) & (all_beh_z < bin_edges[bbi + 1])
        else:
            mask = (all_beh_z >= bin_edges[bbi]) & (all_beh_z <= bin_edges[bbi + 1])
        if np.sum(mask) < 3:
            continue
        bc.append(np.mean(all_beh_z[mask]))
        bm.append(np.mean(all_slope_z[mask]))
        bs.append(np.std(all_slope_z[mask]) / np.sqrt(np.sum(mask)))

    ax.errorbar(bc, bm, yerr=bs, fmt='o-', color='#2c3e50',
                capsize=5, linewidth=2, markersize=7)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.axvline(0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel(f'{blabel} (within-session z)')
    ax.set_ylabel('Slope (within-session z)')

    # Annotate with pooled Spearman rho and permutation p-value (within-session shuffle).
    sig_perm = ('***' if p_perm < 0.001 else
                '**'  if p_perm < 0.01  else
                '*'   if p_perm < 0.05  else '')
    ax.set_title(
        f'{blabel}\nrho={rho_pool:+.3f}, p_perm={p_perm:.4f} {sig_perm}',
        fontsize=11, fontweight='bold')

cutoff_label = f', first {TRIAL_CUTOFF} trials' if TRIAL_CUTOFF is not None else ''
_eff_tau_rew = globals().get('eff_tau_rew', TAU_REW)
_eff_tau_step = globals().get('eff_tau_step', TAU_STEP)
_override_tag = globals().get('override_tag', '')
fig.suptitle(
    f'Continuous-time eligibility HI slope vs candidate behaviors '
    f'(n={n_s} sessions, {n_bins} bins{cutoff_label})\n'
    f'tau_elig={TAU_ELIG}s, tau_bl={TAU_BL}s, '
    f'tau_rew={_eff_tau_rew}s, tau_step={_eff_tau_step}s, zero_iti={ZERO_ITI}',
    fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
cutoff_tag = f'_cut{TRIAL_CUTOFF}' if TRIAL_CUTOFF is not None else ''
_p = os.path.join(RESULTS_DIR,
        f'fig_continuous_eligibility_binned{CONFIG_TAG}{_override_tag}{cutoff_tag}.png')
plt.savefig(_p, dpi=150, bbox_inches='tight')
_save_panel(_p)
plt.show()
print("Binned figure saved.")

#%% ============================================================================
# CELL 10: Lasso regression on pooled (within-session z-scored) data
# ============================================================================
# Tests which behavior variables independently predict HI slope after pooling
# across all sessions.  Pipeline:
#   1. Per session, gather windows where slope and all 8 behaviors are finite.
#   2. Within-session z-score every column (target and each behavior) so that
#      session-level means/scales don't drive the fit.
#   3. Concatenate across sessions; track session id for grouped CV.
#   4. LassoCV with GroupKFold(groups=session_id) picks alpha (groups keep all
#      windows from one session in the same fold — prevents leakage from the
#      within-session autocorrelation of sliding windows).
#   5. Optional session-bootstrap: resample sessions WITH replacement, refit at
#      the chosen alpha, count how often each coefficient is selected and the
#      distribution of its magnitudes.  "Selection frequency" is a robust
#      stability measure for which predictors actually matter.
#
# Set LASSO_TRIAL_CUTOFF to e.g. 40 to restrict to early-learning windows.

from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import GroupKFold

LASSO_TRIAL_CUTOFF = None
LASSO_N_BOOTSTRAPS = 500
LASSO_CV_FOLDS = 5

all_X, all_y, all_groups = [], [], []
for si, s in enumerate(all_results):
    slope = np.asarray(s['hi_with_int'], dtype=float)
    if LASSO_TRIAL_CUTOFF is not None:
        ws = s['win_centers'] - WIN_SIZE / 2.0
        trial_mask = ws < LASSO_TRIAL_CUTOFF
    else:
        trial_mask = np.ones_like(slope, dtype=bool)

    sess_X = np.column_stack([get_beh(s, b) for b in beh_names]).astype(float)
    sess_y = slope.astype(float)

    sess_X = sess_X[trial_mask]
    sess_y = sess_y[trial_mask]

    ok = np.isfinite(sess_y) & np.all(np.isfinite(sess_X), axis=1)
    sess_X = sess_X[ok]
    sess_y = sess_y[ok]
    if len(sess_y) < 5:
        continue

    # Within-session z-score.  Replace zero-std columns with all-zeros so they
    # contribute nothing.
    X_std = sess_X.std(axis=0)
    X_std_safe = np.where(X_std > 0, X_std, 1.0)
    sess_X_z = (sess_X - sess_X.mean(axis=0)) / X_std_safe
    sess_X_z[:, X_std == 0] = 0.0

    y_std = sess_y.std()
    if y_std == 0:
        continue
    sess_y_z = (sess_y - sess_y.mean()) / y_std

    all_X.append(sess_X_z)
    all_y.append(sess_y_z)
    all_groups.append(np.full(len(sess_y), si))

X = np.concatenate(all_X)
y = np.concatenate(all_y)
groups = np.concatenate(all_groups)
unique_sessions = np.unique(groups)
print(f"Pooled: {X.shape[0]} windows from {len(unique_sessions)} sessions, "
      f"{X.shape[1]} features")

# Grouped CV for LassoCV (precomputed splits since LassoCV doesn't accept groups).
n_splits = min(LASSO_CV_FOLDS, len(unique_sessions))
gkf = GroupKFold(n_splits=n_splits)
cv_splits = list(gkf.split(X, y, groups))

lassocv = LassoCV(cv=cv_splits, alphas=None, max_iter=20000, n_jobs=-1,
                  fit_intercept=False)
lassocv.fit(X, y)

# Held-out cross-validated predictions: for each fold, train on the rest and
# predict on this fold's test rows at the LassoCV-chosen alpha.
y_pred_cv = np.full_like(y, np.nan, dtype=float)
for tr_idx, te_idx in cv_splits:
    fold_model = Lasso(alpha=lassocv.alpha_, max_iter=20000,
                       fit_intercept=False)
    fold_model.fit(X[tr_idx], y[tr_idx])
    y_pred_cv[te_idx] = fold_model.predict(X[te_idx])
ok = np.isfinite(y_pred_cv)
ss_res = np.sum((y[ok] - y_pred_cv[ok]) ** 2)
ss_tot = np.sum((y[ok] - y[ok].mean()) ** 2)
cv_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
cv_r, cv_p = pearsonr(y_pred_cv[ok], y[ok])

print(f"Selected alpha:     {lassocv.alpha_:.4g}")
print(f"In-sample R^2:      {lassocv.score(X, y):.4f}")
print(f"CV (held-out) R^2:  {cv_r2:.4f}")
print(f"CV Pearson r:       {cv_r:.4f}  (p = {cv_p:.3e})")

# ---- Coefficient table ----
print("\nLasso coefficients (sorted by |coef|):")
for name, c in sorted(zip(beh_labels, lassocv.coef_),
                      key=lambda x: -abs(x[1])):
    star = '*' if abs(c) > 1e-8 else ''
    print(f"  {name:25s} {c:+.4f} {star}")

# ---- Session-bootstrap for stability ----
rng_lasso = np.random.default_rng(0)
boot_coefs = np.zeros((LASSO_N_BOOTSTRAPS, X.shape[1]))
sess_to_rows = {sid: np.where(groups == sid)[0] for sid in unique_sessions}
for bi in range(LASSO_N_BOOTSTRAPS):
    boot_sess = rng_lasso.choice(unique_sessions, size=len(unique_sessions),
                                 replace=True)
    rows = np.concatenate([sess_to_rows[sid] for sid in boot_sess])
    X_b = X[rows]; y_b = y[rows]
    lasso_b = Lasso(alpha=lassocv.alpha_, max_iter=20000, fit_intercept=False)
    lasso_b.fit(X_b, y_b)
    boot_coefs[bi] = lasso_b.coef_

selection_freq = np.mean(np.abs(boot_coefs) > 1e-8, axis=0)
boot_mean = np.mean(boot_coefs, axis=0)
boot_lo = np.percentile(boot_coefs, 2.5, axis=0)
boot_hi = np.percentile(boot_coefs, 97.5, axis=0)

print(f"\nSession bootstrap (n={LASSO_N_BOOTSTRAPS}):")
print(f"  {'behavior':25s} {'fit_coef':>10s} {'boot_mean':>10s} "
      f"{'95% CI':>22s} {'sel_freq':>10s}")
order = np.argsort(-np.abs(lassocv.coef_))
for bi in order:
    print(f"  {beh_labels[bi]:25s} "
          f"{lassocv.coef_[bi]:+10.4f} {boot_mean[bi]:+10.4f} "
          f"[{boot_lo[bi]:+.4f}, {boot_hi[bi]:+.4f}] "
          f"{selection_freq[bi]:>10.2%}")

# ---- Bar plot + CV fit panel ----
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

ax = axes[0]
order = np.argsort(-np.abs(lassocv.coef_))
colors = ['#3498db' if c > 0 else '#e74c3c' for c in lassocv.coef_[order]]
ax.bar(range(len(order)), lassocv.coef_[order], color=colors, edgecolor='k')
for xi, bi in enumerate(order):
    ax.errorbar(xi, boot_mean[bi],
                yerr=[[boot_mean[bi] - boot_lo[bi]],
                      [boot_hi[bi] - boot_mean[bi]]],
                fmt='o', color='k', markersize=4, capsize=4)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([beh_labels[i] for i in order], rotation=30, ha='right')
ax.set_ylabel('Lasso coefficient (z-scored)')
ax.axhline(0, color='k', lw=0.5)
ax.set_title(f'Lasso coefficients (alpha={lassocv.alpha_:.3g}, '
             f'in-sample R²={lassocv.score(X, y):.3f})',
             fontsize=12, fontweight='bold')

ax = axes[1]
ax.bar(range(len(order)), selection_freq[order],
       color='#7f8c8d', edgecolor='k')
ax.axhline(0.5, color='k', ls='--', alpha=0.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([beh_labels[i] for i in order], rotation=30, ha='right')
ax.set_ylabel('Selection frequency')
ax.set_ylim(0, 1)
ax.set_title(f'Bootstrap selection frequency (n={LASSO_N_BOOTSTRAPS}, '
             f'GroupKFold by session)', fontsize=12, fontweight='bold')

# ---- CV held-out predictions (binned) ----
ax = axes[2]
# Quantile bins on predictions
n_b = 10
edges_b = np.percentile(y_pred_cv[ok], np.linspace(0, 100, n_b + 1))
bxs, bys, bes = [], [], []
for bbi in range(n_b):
    if bbi < n_b - 1:
        m = (y_pred_cv[ok] >= edges_b[bbi]) & (y_pred_cv[ok] < edges_b[bbi + 1])
    else:
        m = (y_pred_cv[ok] >= edges_b[bbi]) & (y_pred_cv[ok] <= edges_b[bbi + 1])
    if np.sum(m) < 3:
        continue
    bxs.append(np.mean(y_pred_cv[ok][m]))
    bys.append(np.mean(y[ok][m]))
    bes.append(np.std(y[ok][m]) / np.sqrt(np.sum(m)))
ax.errorbar(bxs, bys, yerr=bes, fmt='o-', color='#2c3e50',
            capsize=4, linewidth=2, markersize=7)
# Identity line for reference
lo = min(min(bxs), min(bys))
hi = max(max(bxs), max(bys))
ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='identity')
ax.axhline(0, color='k', lw=0.3, alpha=0.5)
ax.axvline(0, color='k', lw=0.3, alpha=0.5)
ax.set_xlabel('Predicted HI slope (CV, z)')
ax.set_ylabel('Observed HI slope (z, binned)')
ax.legend(fontsize=9, loc='best')
ax.set_title(f'Cross-validated fit\nR²={cv_r2:.3f}, '
             f'r={cv_r:.3f} (p={cv_p:.2e})',
             fontsize=12, fontweight='bold')

_lasso_cutoff_tag = (f'_cut{LASSO_TRIAL_CUTOFF}'
                     if LASSO_TRIAL_CUTOFF is not None else '')
fig.suptitle(
    f'Pooled lasso of HI slope on behavior candidates '
    f'(n={X.shape[0]} windows, {len(unique_sessions)} sessions)\n'
    f'tau_elig={TAU_ELIG}s, tau_bl={TAU_BL}s, '
    f'tau_rew={_eff_tau_rew}s, tau_step={_eff_tau_step}s, zero_iti={ZERO_ITI}',
    fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
_p = os.path.join(RESULTS_DIR,
        f'fig_continuous_eligibility_lasso{CONFIG_TAG}{_override_tag}{_lasso_cutoff_tag}.png')
plt.savefig(_p, dpi=150, bbox_inches='tight')
_save_panel(_p)
plt.show()
print("Lasso figure saved.")
