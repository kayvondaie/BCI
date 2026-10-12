#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Test for sensory prediction error signals in pan-neuronal recordings.

The BCI task maps conditioned neuron (CN) fluorescence to lickport speed via a
transfer function whose gain = upper - lower threshold.  When the mouse performs
well, the gain is decreased (thresholds widen).

Three hypothesized response types relative to port movements:
  1. CN-like:  correlated with steps, cross-corr peak slightly LEADS the port
  2. Sensory:  correlated with steps, cross-corr peak slightly LAGS the port;
               slope of neuron-vs-step correlation does NOT change with gain
  3. Prediction error: correlated with steps, cross-corr peak LAGS the port;
               slope SHIFTS when gain changes (reafference cancellation breaks)

Approach:
  - For each session, split trials into gain epochs using BCI_thresholds
  - Convolve step_vector with a short exponential to get a continuous step signal
  - For each neuron, compute:
      (a) cross-correlation with step signal (to find lag)
      (b) neuron-vs-step regression slope in EACH gain epoch
  - Compare slopes across consecutive epoch pairs where gain changes
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import linregress
import traceback
import plotting_functions as pf

import session_counting
import data_dict_create_module_test as ddct
import bci_time_series as bts
from BCI_data_helpers import (
    compute_amp_from_photostim,
    parse_hdf5_array_string,
)

# Environment toggle: 'local' or 'codeocean'
RUN_ENV = 'local'

if RUN_ENV == 'codeocean':
    DATA_ROOT = r'/root/capsule/data/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal'
    RESULTS_DIR = '/root/capsule/results'
else:
    DATA_ROOT = r'//allen/aind/scratch/BCI/2p-raw'
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

session_counting._BASE_DIR_OVERRIDE = DATA_ROOT
list_of_dirs = session_counting.counter()

os.makedirs(RESULTS_DIR, exist_ok=True)
print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI88", "BCI93", "BCI107"]
n_sd_threshold = 2         # outlier classification
TAU_STEP = 0.2             # exponential decay for step convolution (seconds)
MAX_LAG_SEC = 2.0          # max lag for cross-correlation (seconds)
MIN_TRIALS_PER_EPOCH = 3   # minimum trials to compute slope in an epoch

#%% ============================================================================
# CELL 3: Helper functions
# ============================================================================
def forward_fill_thresholds(thr_lower, thr_upper, trl):
    """Forward-fill NaNs and pad to trl length."""
    thr_lower = thr_lower.copy()
    thr_upper = thr_upper.copy()
    for i in range(1, thr_upper.size):
        if np.isnan(thr_upper[i]):
            thr_upper[i] = thr_upper[i - 1]
        if np.isnan(thr_lower[i]):
            thr_lower[i] = thr_lower[i - 1]
    if np.isnan(thr_upper[0]) and np.any(np.isfinite(thr_upper)):
        thr_upper[0] = thr_upper[np.isfinite(thr_upper)][0]
    if np.isnan(thr_lower[0]) and np.any(np.isfinite(thr_lower)):
        thr_lower[0] = thr_lower[np.isfinite(thr_lower)][0]
    if len(thr_upper) < trl:
        thr_upper = np.concatenate([thr_upper,
            np.full(trl - len(thr_upper), thr_upper[-1])])
        thr_lower = np.concatenate([thr_lower,
            np.full(trl - len(thr_lower), thr_lower[-1])])
    return thr_lower, thr_upper


def detect_gain_epochs(thr_upper, trl):
    """Detect threshold switch points, return (switches, epoch_ends, n_epochs)."""
    d_upper = np.diff(thr_upper)
    switches = np.where((d_upper != 0) & np.isfinite(d_upper))[0] + 1
    switches = np.concatenate(([0], switches))
    epoch_ends = np.concatenate((switches[1:], [trl]))
    return switches, epoch_ends, len(switches)


def build_step_conv(step_vector, dt_si, tau):
    """Convolve binary step_vector with causal decaying exponential."""
    n_kernel = int(5 * tau / dt_si)
    t_kernel = np.arange(n_kernel) * dt_si
    kernel = np.exp(-t_kernel / tau)
    kernel = kernel / np.sum(kernel)
    return np.convolve(step_vector.astype(float), kernel, mode='full')[:len(step_vector)]


def compute_xcorr(trace, step_sig, max_lag_frames, dt_si):
    """Compute normalized cross-correlation, return (lags, xcorr_window, peak_lag, peak_r)."""
    valid = np.isfinite(trace) & np.isfinite(step_sig)
    if np.sum(valid) < 200:
        return None
    t_v = trace.copy()
    s_v = step_sig.copy()
    t_v[~valid] = 0
    s_v[~valid] = 0
    t_v = t_v - np.mean(t_v[valid])
    s_v = s_v - np.mean(s_v[valid])

    xcorr = correlate(t_v, s_v, mode='full')
    mid = len(t_v) - 1
    xcorr_window = xcorr[mid - max_lag_frames:mid + max_lag_frames + 1]
    norm = np.sqrt(np.sum(t_v**2) * np.sum(s_v**2))
    if norm <= 0:
        return None
    xcorr_window = xcorr_window / norm

    lags = np.arange(-max_lag_frames, max_lag_frames + 1) * dt_si
    peak_idx = np.argmax(np.abs(xcorr_window))
    return lags, xcorr_window, lags[peak_idx], xcorr_window[peak_idx]


def epoch_slope(trace, step_sig, frame_inds):
    """Regression slope of neuron ~ step_conv for given frame indices."""
    x = step_sig[frame_inds]
    y = trace[frame_inds]
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 50:
        return np.nan
    return linregress(x[valid], y[valid]).slope


#%% ============================================================================
# CELL 4: Main loop -- cross-correlation and per-epoch slopes
# ============================================================================
# Per-neuron, per-epoch-pair results
all_peak_lag = []        # peak xcorr lag (s); positive = step leads neuron
all_peak_r = []          # peak xcorr value
all_slope_pre = []       # slope in epoch before gain change
all_slope_post = []      # slope in epoch after gain change
all_is_outlier = []
all_gain_ratio = []      # gain_post / gain_pre
all_gain_pre = []
all_gain_post = []

# Store per-session data for diagnostic plotting
session_data_store = {}
session_labels = []

for mi, mouse_name in enumerate(mice):
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse_name) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]

    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            folder = os.path.join(DATA_ROOT, mouse, session, 'pophys') + '/'
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            # Load data
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

            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]
            df = data['df_closedloop']
            n_cells = df.shape[0]

            # ---- Classify outlier neurons ----
            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            amp_ep0 = AMP[0]
            amp_masked = amp_ep0.copy()
            amp_masked[stimDist < 30] = np.nan
            mean_amp_nontarg = np.nanmean(amp_masked, axis=1)

            mu_amp = np.nanmean(mean_amp_nontarg)
            sd_amp = np.nanstd(mean_amp_nontarg)
            outlier_threshold = mu_amp + n_sd_threshold * sd_amp
            is_outlier = mean_amp_nontarg > outlier_threshold
            is_valid = np.isfinite(mean_amp_nontarg)
            print(f"  Outlier: {np.sum(is_outlier)}, Rest: {np.sum(is_valid & ~is_outlier)}")

            # ---- BCI thresholds ----
            BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
            thr_lower, thr_upper = forward_fill_thresholds(
                BCI_thresholds[0, :], BCI_thresholds[1, :], trl)
            gain = thr_upper - thr_lower
            switches, epoch_ends, n_epochs = detect_gain_epochs(thr_upper, trl)

            if n_epochs < 2:
                print(f"  Only {n_epochs} gain epoch(s) -- skipping.")
                continue

            # Print all epochs
            for ei in range(n_epochs):
                t0, t1 = switches[ei], epoch_ends[ei]
                print(f"  Epoch {ei}: trials {t0}-{t1-1}, gain={gain[t0]:.2f}")

            # ---- Build step vector and convolve ----
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0

            step_vector, reward_vector, trial_start_vector = \
                bts.bci_time_series_fun(folder, data, rt_filled, dt_si)
            step_conv = build_step_conv(step_vector, dt_si, TAU_STEP)

            # ---- Map trial indices to frame ranges ----
            ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy',
                          allow_pickle=True).tolist()
            frames_per_file = ops['frames_per_file']
            trial_starts_frame = np.cumsum([0] + list(frames_per_file[:-1]))

            def get_epoch_frames(trial_inds):
                frames = []
                for ti in trial_inds:
                    if ti < len(trial_starts_frame) and ti < len(frames_per_file):
                        t0 = trial_starts_frame[ti]
                        t1 = t0 + frames_per_file[ti]
                        frames.extend(range(t0, min(t1, df.shape[1])))
                return np.array(frames, dtype=int)

            n_common = min(df.shape[1], len(step_conv))

            # Build frame indices for each epoch (full, odd-trial, even-trial)
            epoch_frames = []
            epoch_frames_odd = []   # odd trials — for xcorr (peak_r)
            epoch_frames_even = []  # even trials — for slopes (ds/dg)
            epoch_gains = []
            for ei in range(n_epochs):
                ep_trials = np.arange(switches[ei], epoch_ends[ei])
                if len(ep_trials) < MIN_TRIALS_PER_EPOCH:
                    epoch_frames.append(None)
                    epoch_frames_odd.append(None)
                    epoch_frames_even.append(None)
                    epoch_gains.append(gain[switches[ei]])
                    continue
                fr = get_epoch_frames(ep_trials)
                fr = fr[fr < n_common]
                # Split trials within this epoch into odd/even
                local_trials = ep_trials
                odd_trials = local_trials[0::2]
                even_trials = local_trials[1::2]
                fr_odd = get_epoch_frames(odd_trials)
                fr_odd = fr_odd[fr_odd < n_common]
                fr_even = get_epoch_frames(even_trials)
                fr_even = fr_even[fr_even < n_common]
                epoch_frames.append(fr if len(fr) >= 30 else None)
                epoch_frames_odd.append(fr_odd if len(fr_odd) >= 30 else None)
                epoch_frames_even.append(fr_even if len(fr_even) >= 30 else None)
                epoch_gains.append(gain[switches[ei]])

            # Relative gain: each epoch's gain / first epoch's gain
            gain0 = epoch_gains[0]
            epoch_gains_rel = [g / gain0 if gain0 > 0 else np.nan
                               for g in epoch_gains]

            # ---- Per-neuron: xcorr + slope in every epoch ----
            max_lag_frames = int(MAX_LAG_SEC / dt_si)

            # Store per-neuron results for diagnostic cell
            session_neuron_data = []  # list of dicts per valid neuron

            for ci in range(n_cells):
                if not is_valid[ci]:
                    continue

                trace_full = df[ci, :n_common]

                # Cross-correlation (full session)
                xc_result = compute_xcorr(trace_full, step_conv[:n_common],
                                          max_lag_frames, dt_si)
                if xc_result is None:
                    continue
                lags, xcorr_window, peak_lag, peak_r = xc_result

                # Slope in each epoch (full data)
                slopes = []
                for ei in range(n_epochs):
                    if epoch_frames[ei] is not None:
                        slopes.append(epoch_slope(trace_full, step_conv, epoch_frames[ei]))
                    else:
                        slopes.append(np.nan)

                # --- Split-half for circularity control ---
                # peak_r_odd: xcorr from ODD trials only (all epochs)
                peak_r_odd = np.nan
                all_odd_frames = []
                for ei in range(n_epochs):
                    if epoch_frames_odd[ei] is not None:
                        all_odd_frames.append(epoch_frames_odd[ei])
                if len(all_odd_frames) > 0:
                    odd_fr = np.concatenate(all_odd_frames)
                    if len(odd_fr) >= 200:
                        xc_odd = compute_xcorr(trace_full[odd_fr],
                                               step_conv[odd_fr],
                                               max_lag_frames, dt_si)
                        if xc_odd is not None:
                            _, _, _, peak_r_odd = xc_odd

                # slopes_even: slopes from EVEN trials only (per epoch)
                slopes_even = []
                for ei in range(n_epochs):
                    if epoch_frames_even[ei] is not None:
                        slopes_even.append(epoch_slope(trace_full, step_conv,
                                                       epoch_frames_even[ei]))
                    else:
                        slopes_even.append(np.nan)

                session_neuron_data.append(dict(
                    ci=ci, xcorr=xcorr_window, peak_lag=peak_lag,
                    peak_r=peak_r, peak_r_odd=peak_r_odd,
                    slopes=slopes, slopes_even=slopes_even,
                    is_outlier=is_outlier[ci],
                ))

            # Accumulate across sessions -- all valid epoch pairs (not just consecutive)
            for nd in session_neuron_data:
                valid_ei = [ei for ei in range(n_epochs)
                            if np.isfinite(nd['slopes'][ei])]
                for ii in range(len(valid_ei)):
                    for jj in range(ii + 1, len(valid_ei)):
                        ei0, ei1 = valid_ei[ii], valid_ei[jj]
                        g_pre = epoch_gains[ei0]
                        g_post = epoch_gains[ei1]
                        if g_pre > 0:
                            all_peak_lag.append(nd['peak_lag'])
                            all_peak_r.append(nd['peak_r'])
                            all_slope_pre.append(nd['slopes'][ei0])
                            all_slope_post.append(nd['slopes'][ei1])
                            all_is_outlier.append(nd['is_outlier'])
                            all_gain_ratio.append(g_post / g_pre)
                            all_gain_pre.append(g_pre)
                            all_gain_post.append(g_post)

            # Store session data for diagnostics
            session_data_store[f"{mouse} {session}"] = dict(
                df=df, step_conv=step_conv, n_common=n_common,
                is_outlier=is_outlier, is_valid=is_valid,
                epoch_frames=epoch_frames, epoch_gains=epoch_gains,
                epoch_gains_rel=epoch_gains_rel, gain0=gain0,
                switches=switches, epoch_ends=epoch_ends, n_epochs=n_epochs,
                lags=lags,
                neuron_data=session_neuron_data,
                dt_si=dt_si, trl=trl, gain=gain,
            )
            session_labels.append(f"{mouse} {session}")

        except Exception:
            traceback.print_exc()
            continue

all_peak_lag = np.array(all_peak_lag)
all_peak_r = np.array(all_peak_r)
all_slope_pre = np.array(all_slope_pre)
all_slope_post = np.array(all_slope_post)
all_is_outlier = np.array(all_is_outlier)
all_gain_ratio = np.array(all_gain_ratio)
all_gain_pre = np.array(all_gain_pre)
all_gain_post = np.array(all_gain_post)

print(f"\nTotal neuron-epoch-pairs: {len(all_peak_lag)} "
      f"({np.sum(all_is_outlier)} outlier, {np.sum(~all_is_outlier)} rest)")
print(f"Sessions: {len(session_labels)}")

#%% ============================================================================
# CELL 5: Diagnostic -- example neurons of the 3 response types
# ============================================================================
"""
Pick a session with multiple gain epochs. Classify each neuron into one of:
  1. CN-like:  peak lag < 0 (neuron leads steps)
  2. Sensory:  peak lag > 0 (steps lead neuron), slope tracks gain
               (slope_post ≈ slope_pre * gain_post/gain_pre)
  3. Pred. error: peak lag > 0, slope does NOT track gain
               (gain-corrected residual is large)

Classification uses gain-corrected slope residual:
  predicted_slope_post = slope_pre * (gain_post / gain_pre)
  residual = |slope_post - predicted| / max(|slope_pre|, |slope_post|)
A pure sensory neuron has residual ≈ 0; prediction error neuron deviates.

For each type, show:
  Left column:  population-average cross-correlation
  Right column: population-average mean_bin_plot of F (x) vs step_conv (y) per epoch
"""
# Change this index to browse sessions (0, 1, 2, ...). Sorted by number of gain epochs.
DIAG_INDEX = 1

sess_keys_sorted = sorted(session_data_store.keys(),
                           key=lambda k: -session_data_store[k]['n_epochs'])
for i, k in enumerate(sess_keys_sorted):
    n_ep = session_data_store[k]['n_epochs']
    print(f"  [{i}] {k}  ({n_ep} epochs)")
DIAG_INDEX = min(DIAG_INDEX, len(sess_keys_sorted) - 1)
best_sess = sess_keys_sorted[DIAG_INDEX]
sd = session_data_store[best_sess]
print(f"Diagnostic session: {best_sess} ({sd['n_epochs']} gain epochs)")

lags_sec = sd['lags']
epoch_colors_map = plt.cm.coolwarm(np.linspace(0, 1, sd['n_epochs']))

# Classify neurons into 4 groups:
#   1. CN-like: lag <= 0
#   2-4. Step-following (lag > 0), split by mean slope change direction:
#        - slope increases with gain
#        - slope decreases with gain
#        - slope stable (no systematic change)
cn_like = []
sf_all = []         # (nd, mean_ds_per_dg) for step-following, anticorrelated neurons
sf_increase = []
sf_decrease = []
sf_stable = []

n_pos_xcorr_skipped = 0

for nd in sd['neuron_data']:
    valid_slopes_nd = [s for s in nd['slopes'] if np.isfinite(s)]
    if len(valid_slopes_nd) < 2:
        continue

    # Row 1: neuron leads (negative lag, positive peak)
    if nd['peak_lag'] <= 0 and nd['peak_r'] > 0:
        cn_like.append(nd)
        continue

    # Port leads: only include positively correlated neurons (peak_r > 0)
    if nd['peak_lag'] > 0 and nd['peak_r'] <= 0:
        n_pos_xcorr_skipped += 1
        continue

    if nd['peak_lag'] <= 0:
        # negative peak_r with negative lag -- skip
        continue

    # Compute mean slope change per unit gain change across all valid epoch pairs
    delta_slopes = []
    delta_gains = []
    valid_ei = [ei for ei in range(len(nd['slopes']))
                if np.isfinite(nd['slopes'][ei])]
    for ii in range(len(valid_ei)):
        for jj in range(ii + 1, len(valid_ei)):
            ei0, ei1 = valid_ei[ii], valid_ei[jj]
            s0, s1 = nd['slopes'][ei0], nd['slopes'][ei1]
            g0, g1 = sd['epoch_gains'][ei0], sd['epoch_gains'][ei1]
            if g0 > 0 and g1 > 0 and (g1 - g0) != 0:
                delta_slopes.append(s1 - s0)
                delta_gains.append(g1 - g0)

    if len(delta_slopes) == 0:
        sf_all.append((nd, 0.0))
        continue

    ds = np.array(delta_slopes)
    dg = np.array(delta_gains)
    mean_ds_per_dg = np.mean(ds / dg)
    sf_all.append((nd, mean_ds_per_dg))

# Split step-following by terciles of mean_ds_per_dg
if len(sf_all) > 0:
    all_metrics = np.array([m for _, m in sf_all])
    abs_metrics = np.abs(all_metrics)
    p10_abs = np.percentile(abs_metrics, 20)  # small |change| = stable
    p90 = np.percentile(all_metrics, 80)      # top 10% signed = increase
    p10 = np.percentile(all_metrics, 20)      # bottom 10% signed = decrease
    print(f"  Step-following ds/dg: min={np.min(all_metrics):.4f}, "
          f"10th={p10:.4f}, 90th={p90:.4f}, "
          f"|ds/dg| 10th={p10_abs:.4f}, max={np.max(all_metrics):.4f}")
    for nd, metric in sf_all:
        if metric <= p10:
            sf_decrease.append(nd)
        elif metric >= p90:
            sf_increase.append(nd)
        elif abs(metric) <= p10_abs:
            sf_stable.append(nd)

print(f"  Neurons lead: {len(cn_like)}, Port leads stable: {len(sf_stable)}, "
      f"slope down: {len(sf_decrease)}, slope up: {len(sf_increase)}")
print(f"  Skipped {n_pos_xcorr_skipped} negative-xcorr port-leads neurons")
print(f"  Epoch gains: {sd['epoch_gains']}")
# Debug: which epochs have valid frames?
for ei in range(sd['n_epochs']):
    ef = sd['epoch_frames'][ei]
    n_fr = len(ef) if ef is not None else 0
    print(f"    Epoch {ei}: gain={sd['epoch_gains'][ei]:.1f}, "
          f"frames={'None' if ef is None else len(ef)}")
# Debug: sample neurons from each group
for grp_name, grp in [('sf_increase', sf_increase), ('sf_decrease', sf_decrease),
                       ('sf_stable', sf_stable)]:
    if len(grp) > 0:
        nd0 = grp[0]
        print(f"  {grp_name} example cell {nd0['ci']}: slopes={nd0['slopes']}")
    else:
        print(f"  {grp_name}: EMPTY")
# Debug: check df shape and n_common
print(f"  df shape: {sd['df'].shape}, n_common: {sd['n_common']}")
print(f"  step_conv len: {len(sd['step_conv'])}")

# Collect outlier neurons (regardless of lag/peak sign)
outlier_neurons = [nd for nd in sd['neuron_data']
                   if nd['is_outlier'] and
                   len([s for s in nd['slopes'] if np.isfinite(s)]) >= 2]
print(f"  Outlier neurons: {len(outlier_neurons)}")

type_labels = ['Neurons lead',
               'Port leads, slopes don\'t change',
               'Port leads, slope goes down\nas BCI_threshold goes up',
               'Port leads, slope goes up\nas BCI_threshold goes up',
               'Outlier neurons']
type_groups = [cn_like, sf_stable, sf_decrease, sf_increase, outlier_neurons]
type_colors = ['steelblue', 'black', 'darkorange', 'red', 'magenta']

fig, axes = plt.subplots(5, 2, figsize=(10, 17))

for row, (group, label, col) in enumerate(zip(type_groups, type_labels, type_colors)):
    if len(group) == 0:
        for c in range(2):
            axes[row, c].text(0.5, 0.5, f'No {label} found', ha='center',
                              va='center', transform=axes[row, c].transAxes)
        continue

    # --- Col 0: population-average cross-correlation ---
    ax = axes[row, 0]
    xcorr_stack = np.array([nd['xcorr'] for nd in group])
    xcorr_mean = np.nanmean(xcorr_stack, axis=0)
    xcorr_sem = np.nanstd(xcorr_stack, axis=0) / np.sqrt(len(group))
    ax.fill_between(lags_sec, xcorr_mean - xcorr_sem, xcorr_mean + xcorr_sem,
                    color=col, alpha=0.2)
    ax.plot(lags_sec, xcorr_mean, color=col, lw=1.5)
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Lag (s)  [+ = steps lead]')
    ax.set_ylabel('Cross-correlation')
    ax.set_title(f'{label}\nn={len(group)} neurons (mean ± SEM)')
    ax.set_xlim(-MAX_LAG_SEC, MAX_LAG_SEC)

    # --- Col 1: population-average F vs step, per epoch ---
    ax = axes[row, 1]
    cell_inds = [nd['ci'] for nd in group]
    print(f"  Row {row} ({label.split(chr(10))[0]}): {len(cell_inds)} cells, "
          f"cell_inds[:5]={cell_inds[:5]}")
    pop_slopes = []
    for ei in range(sd['n_epochs']):
        if sd['epoch_frames'][ei] is None:
            pop_slopes.append(np.nan)
            print(f"    Epoch {ei}: frames=None, skipped")
            continue
        fr = sd['epoch_frames'][ei]
        traces = sd['df'][cell_inds, :][:, :sd['n_common']]
        f_vals = np.nanmean(traces[:, fr], axis=0)
        s_vals = sd['step_conv'][fr]
        valid = np.isfinite(f_vals) & np.isfinite(s_vals)
        print(f"    Epoch {ei}: {len(fr)} frames, {np.sum(valid)} valid, "
              f"f range=[{np.nanmin(f_vals):.3f}, {np.nanmax(f_vals):.3f}], "
              f"s range=[{np.nanmin(s_vals):.3f}, {np.nanmax(s_vals):.3f}]")
        if np.sum(valid) > 50:
            xv = s_vals[valid]   # step_conv on x
            yv = f_vals[valid]   # fluorescence on y
            n_bins = 5
            bin_edges = np.linspace(np.min(xv), np.max(xv), n_bins + 1)
            bin_idx = np.digitize(xv, bin_edges) - 1
            X, Y, SE = [], [], []
            for bi in range(n_bins):
                mask_bi = bin_idx == bi
                if np.sum(mask_bi) > 5:
                    X.append(np.mean(xv[mask_bi]))
                    Y.append(np.mean(yv[mask_bi]))
                    SE.append(np.std(yv[mask_bi]) / np.sqrt(np.sum(mask_bi)))
            if len(X) > 0:
                ax.errorbar(X, Y, SE, marker='o', markersize=5,
                            color=epoch_colors_map[ei],
                            markerfacecolor=epoch_colors_map[ei])
        ep_slopes = [nd['slopes'][ei] for nd in group if np.isfinite(nd['slopes'][ei])]
        pop_slopes.append(np.mean(ep_slopes) if len(ep_slopes) > 0 else np.nan)
    ax.set_xlabel('Step signal')
    ax.set_ylabel('Pop avg fluorescence (DF/F)')
    slopes_str = ', '.join([f'{s:.2f}' if np.isfinite(s) else 'nan'
                            for s in pop_slopes])
    ax.set_title(f'F vs step by epoch  (slopes: {slopes_str})')
    for ei in range(sd['n_epochs']):
        rel_g = sd['epoch_gains_rel'][ei]
        ax.plot([], [], color=epoch_colors_map[ei], lw=2,
                label=f'Ep{ei} g={sd["epoch_gains"][ei]:.0f} ({rel_g:.1f}x)')
    ax.legend(fontsize=6, loc='best')

plt.suptitle(f'Diagnostic: {best_sess} — 5 neuron types', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_diagnostic_session.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 5b: Per-epoch cross-correlations
# ============================================================================
"""
Cross-correlation between neuron activity and step_conv computed WITHIN each
gain epoch. Compare xcorr shape/amplitude across epochs.

Top:    best single neuron (highest |peak_r|), one xcorr curve per epoch
Bottom: population average xcorr per epoch (all neurons)
"""
dt = sd['dt_si']
max_lag_frames = int(MAX_LAG_SEC / dt)
step_sig = sd['step_conv'][:sd['n_common']]

# Best single neuron
best_ni = max(range(len(sd['neuron_data'])),
              key=lambda i: abs(sd['neuron_data'][i]['peak_r']))
nd_best = sd['neuron_data'][best_ni]
ci = nd_best['ci']
trace = sd['df'][ci, :sd['n_common']]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# --- Top: single best neuron, per-epoch xcorr ---
ax = axes[0]
for ei in range(sd['n_epochs']):
    if sd['epoch_frames'][ei] is None:
        continue
    fr = sd['epoch_frames'][ei]
    xc = compute_xcorr(trace[fr], step_sig[fr], max_lag_frames, dt)
    if xc is not None:
        lags_ep, xcorr_ep, _, _ = xc
        ax.plot(lags_ep, xcorr_ep, color=epoch_colors_map[ei], lw=1.5,
                label=f'Ep{ei} g={sd["epoch_gains"][ei]:.0f}')
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Lag (s)  [+ = steps lead]')
ax.set_ylabel('Cross-correlation')
ax.set_title(f'Cell {ci} (|r|={abs(nd_best["peak_r"]):.3f}) — xcorr per gain epoch')
ax.set_xlim(-MAX_LAG_SEC, MAX_LAG_SEC)
ax.legend(fontsize=8)

# --- Bottom: population average per-epoch xcorr ---
ax = axes[1]
for ei in range(sd['n_epochs']):
    if sd['epoch_frames'][ei] is None:
        continue
    fr = sd['epoch_frames'][ei]
    epoch_xcorrs = []
    for nd in sd['neuron_data']:
        t = sd['df'][nd['ci'], :sd['n_common']]
        xc = compute_xcorr(t[fr], step_sig[fr], max_lag_frames, dt)
        if xc is not None:
            epoch_xcorrs.append(xc[1])
    if len(epoch_xcorrs) > 0:
        xcorr_stack = np.array(epoch_xcorrs)
        xcorr_mean = np.nanmean(xcorr_stack, axis=0)
        xcorr_sem = np.nanstd(xcorr_stack, axis=0) / np.sqrt(len(epoch_xcorrs))
        lags_ep = np.arange(-max_lag_frames, max_lag_frames + 1) * dt
        ax.fill_between(lags_ep, xcorr_mean - xcorr_sem, xcorr_mean + xcorr_sem,
                        color=epoch_colors_map[ei], alpha=0.15)
        ax.plot(lags_ep, xcorr_mean, color=epoch_colors_map[ei], lw=1.5,
                label=f'Ep{ei} g={sd["epoch_gains"][ei]:.0f} (n={len(epoch_xcorrs)})')
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Lag (s)  [+ = steps lead]')
ax.set_ylabel('Cross-correlation')
ax.set_title('Population avg xcorr per gain epoch — all neurons')
ax.set_xlim(-MAX_LAG_SEC, MAX_LAG_SEC)
ax.legend(fontsize=7)

plt.suptitle(f'{best_sess} — epoch-wise cross-correlations', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_epoch_xcorr.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 6: Cross-correlation lag distributions (all sessions)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sig_mask = np.abs(all_peak_r) > 0.02

ax = axes[0]
bins = np.linspace(-MAX_LAG_SEC, MAX_LAG_SEC, 50)
ax.hist(all_peak_lag[sig_mask & ~all_is_outlier], bins=bins, color='black',
        alpha=0.5, density=True, label='Rest')
ax.hist(all_peak_lag[sig_mask & all_is_outlier], bins=bins, color='red',
        alpha=0.5, density=True, label='Outlier')
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('Peak xcorr lag (s)\n(positive = steps lead neuron)')
ax.set_ylabel('Density')
ax.set_title('Cross-correlation lag with step signal')
ax.legend(fontsize=8)

ax = axes[1]
bins_r = np.linspace(-0.3, 0.3, 50)
ax.hist(all_peak_r[~all_is_outlier], bins=bins_r, color='black',
        alpha=0.5, density=True, label='Rest')
ax.hist(all_peak_r[all_is_outlier], bins=bins_r, color='red',
        alpha=0.5, density=True, label='Outlier')
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('Peak xcorr value')
ax.set_ylabel('Density')
ax.set_title('Cross-correlation strength')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_xcorr_lag_distribution.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7: Slope change across all consecutive gain epoch pairs
# ============================================================================
valid_slopes = np.isfinite(all_slope_pre) & np.isfinite(all_slope_post)
slope_change = all_slope_post - all_slope_pre

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Slope pre vs post scatter
ax = axes[0]
mask_out = valid_slopes & all_is_outlier
mask_rest = valid_slopes & ~all_is_outlier
ax.scatter(all_slope_pre[mask_rest], all_slope_post[mask_rest],
           s=5, alpha=0.3, color='black', label='Rest')
ax.scatter(all_slope_pre[mask_out], all_slope_post[mask_out],
           s=15, alpha=0.7, color='red', label='Outlier')
lim = np.nanpercentile(np.abs(np.concatenate([all_slope_pre[valid_slopes],
                                               all_slope_post[valid_slopes]])), 98)
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.5)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel('Slope (pre gain change)')
ax.set_ylabel('Slope (post gain change)')
ax.set_title('Neuron-step slope across gain changes')
ax.legend(fontsize=8)
ax.set_aspect('equal')

# Slope change histogram
ax = axes[1]
sc_lim = np.nanpercentile(np.abs(slope_change[valid_slopes]), 98)
bins_sc = np.linspace(-sc_lim, sc_lim, 40)
ax.hist(slope_change[valid_slopes & ~all_is_outlier], bins=bins_sc, color='black',
        alpha=0.5, density=True, label='Rest')
ax.hist(slope_change[valid_slopes & all_is_outlier], bins=bins_sc, color='red',
        alpha=0.5, density=True, label='Outlier')
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('Slope change (post - pre)')
ax.set_ylabel('Density')
ax.set_title('Slope change distribution')
ax.legend(fontsize=8)

# Slope change vs peak lag
ax = axes[2]
ax.scatter(all_peak_lag[valid_slopes & ~all_is_outlier],
           slope_change[valid_slopes & ~all_is_outlier],
           s=5, alpha=0.2, color='black', label='Rest')
ax.scatter(all_peak_lag[valid_slopes & all_is_outlier],
           slope_change[valid_slopes & all_is_outlier],
           s=15, alpha=0.7, color='red', label='Outlier')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Peak xcorr lag (s)\n(positive = steps lead)')
ax.set_ylabel('Slope change (post - pre)')
ax.set_title('Lag vs slope change')
ax.legend(fontsize=8)

plt.suptitle(f'Sensory prediction error ({len(session_labels)} sessions, '
             f'{len(all_peak_lag)} neuron-epoch pairs)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_slope_change_analysis.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 8: Normalize slope change by gain ratio
# ============================================================================
"""
Pure sensory neuron: slope should scale with gain (slope_post/slope_pre ~ gain_post/gain_pre)
Prediction error neuron: slope deviates from this scaling.
"""
slope_predicted = all_slope_pre * all_gain_ratio
slope_residual = all_slope_post - slope_predicted

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

valid_res = valid_slopes & np.isfinite(slope_residual)

ax = axes[0]
res_lim = np.nanpercentile(np.abs(slope_residual[valid_res]), 98)
bins_res = np.linspace(-res_lim, res_lim, 40)
ax.hist(slope_residual[valid_res & ~all_is_outlier], bins=bins_res, color='black',
        alpha=0.5, density=True, label='Rest')
ax.hist(slope_residual[valid_res & all_is_outlier], bins=bins_res, color='red',
        alpha=0.5, density=True, label='Outlier')
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('Slope residual (actual - gain-predicted)')
ax.set_ylabel('Density')
ax.set_title('Deviation from pure sensory model')
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(all_peak_lag[valid_res & ~all_is_outlier],
           slope_residual[valid_res & ~all_is_outlier],
           s=5, alpha=0.2, color='black', label='Rest')
ax.scatter(all_peak_lag[valid_res & all_is_outlier],
           slope_residual[valid_res & all_is_outlier],
           s=15, alpha=0.7, color='red', label='Outlier')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Peak xcorr lag (s)\n(positive = steps lead)')
ax.set_ylabel('Slope residual')
ax.set_title('Lag vs gain-corrected slope change')
ax.legend(fontsize=8)

plt.suptitle('Prediction error test: slope residual after gain correction', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_gain_corrected_residual.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"Outlier slope residual: mean={np.nanmean(slope_residual[valid_res & all_is_outlier]):.4f}")
print(f"Rest slope residual:    mean={np.nanmean(slope_residual[valid_res & ~all_is_outlier]):.4f}")

#%% ============================================================================
# CELL 9: Cross-session summary — classify all neurons, pool by group
# ============================================================================
"""
Apply the Cell 5 classification to ALL sessions:
  1. Neurons lead (peak_lag <= 0, peak_r > 0)
  2. Port leads, slopes don't change (|ds/dg| < 20th pctl of |ds/dg|)
  3. Port leads, slope goes down as threshold goes up (ds/dg < 20th pctl)
  4. Port leads, slope goes up as threshold goes up (ds/dg > 80th pctl)
  5. Outlier neurons

Compile:
  - Neuron counts per group per session
  - Per-group distribution of ds/dg across all sessions
  - Per-group xcorr (pooled across sessions)
  - Per-group F vs step binned by normalized gain (low/med/high terciles)
"""

def classify_session_neurons(sd):
    """Classify neurons in one session. Returns dict of group lists."""
    cn_like = []
    sf_all = []  # (nd, mean_ds_per_dg)
    outlier_all = []

    for nd in sd['neuron_data']:
        valid_slopes_nd = [s for s in nd['slopes'] if np.isfinite(s)]

        # Track outliers separately
        if nd['is_outlier'] and len(valid_slopes_nd) >= 2:
            outlier_all.append(nd)

        if len(valid_slopes_nd) < 2:
            continue

        if nd['peak_lag'] <= 0 and nd['peak_r'] > 0:
            cn_like.append(nd)
            continue

        if nd['peak_lag'] > 0 and nd['peak_r'] > 0:
            # Compute ds/dg
            delta_slopes = []
            delta_gains = []
            valid_ei = [ei for ei in range(len(nd['slopes']))
                        if np.isfinite(nd['slopes'][ei])]
            for ii in range(len(valid_ei)):
                for jj in range(ii + 1, len(valid_ei)):
                    ei0, ei1 = valid_ei[ii], valid_ei[jj]
                    s0, s1 = nd['slopes'][ei0], nd['slopes'][ei1]
                    g0, g1 = sd['epoch_gains'][ei0], sd['epoch_gains'][ei1]
                    if g0 > 0 and g1 > 0 and (g1 - g0) != 0:
                        delta_slopes.append(s1 - s0)
                        delta_gains.append(g1 - g0)

            if len(delta_slopes) == 0:
                sf_all.append((nd, 0.0))
            else:
                ds = np.array(delta_slopes)
                dg = np.array(delta_gains)
                sf_all.append((nd, np.mean(ds / dg)))

    return cn_like, sf_all, outlier_all


# --- Collect ds/dg metrics across ALL sessions ---
all_sf_metrics = []  # (session_key, nd, metric)
all_cn_like = []     # (session_key, nd)
all_outlier = []     # (session_key, nd)

for sess_key, sd in session_data_store.items():
    cn, sf, outliers = classify_session_neurons(sd)
    for nd in cn:
        all_cn_like.append((sess_key, nd))
    for nd, metric in sf:
        all_sf_metrics.append((sess_key, nd, metric))
    for nd in outliers:
        all_outlier.append((sess_key, nd))

# Split step-following neurons using pooled percentiles
all_metrics_arr = np.array([m for _, _, m in all_sf_metrics])
sf_decrease_all = []
sf_stable_all = []
sf_increase_all = []

if len(all_metrics_arr) > 0:
    abs_metrics = np.abs(all_metrics_arr)
    p_abs_stable = np.percentile(abs_metrics, 20)
    p_decrease = np.percentile(all_metrics_arr, 20)
    p_increase = np.percentile(all_metrics_arr, 80)

    for sess_key, nd, metric in all_sf_metrics:
        if metric <= p_decrease:
            sf_decrease_all.append((sess_key, nd))
        elif metric >= p_increase:
            sf_increase_all.append((sess_key, nd))
        elif abs(metric) <= p_abs_stable:
            sf_stable_all.append((sess_key, nd))

print(f"=== Cross-session summary ===")
print(f"  Neurons lead: {len(all_cn_like)}")
print(f"  Port leads, stable: {len(sf_stable_all)}")
print(f"  Port leads, slope down: {len(sf_decrease_all)}")
print(f"  Port leads, slope up: {len(sf_increase_all)}")
print(f"  Outlier: {len(all_outlier)}")
print(f"  Total step-following: {len(all_sf_metrics)}")
print(f"  ds/dg percentiles: 20th={p_decrease:.4f}, 80th={p_increase:.4f}, "
      f"|ds/dg| 20th={p_abs_stable:.4f}")

# --- Figure 1: ds/dg distribution ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.hist(all_metrics_arr, bins=50, color='gray', alpha=0.7, edgecolor='k', lw=0.3)
ax.axvline(p_decrease, color='darkorange', lw=1.5, ls='--', label=f'20th pctl={p_decrease:.4f}')
ax.axvline(p_increase, color='red', lw=1.5, ls='--', label=f'80th pctl={p_increase:.4f}')
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('mean(Δslope / Δgain)')
ax.set_ylabel('Count')
ax.set_title(f'ds/dg distribution (n={len(all_metrics_arr)} neurons)')
ax.legend(fontsize=8)

ax = axes[1]
# Counts per session
sess_counts = {}
for sess_key in session_data_store:
    sess_counts[sess_key] = {'cn': 0, 'stable': 0, 'down': 0, 'up': 0, 'outlier': 0}
for sk, _ in all_cn_like:
    sess_counts[sk]['cn'] += 1
for sk, _ in sf_stable_all:
    sess_counts[sk]['stable'] += 1
for sk, _ in sf_decrease_all:
    sess_counts[sk]['down'] += 1
for sk, _ in sf_increase_all:
    sess_counts[sk]['up'] += 1
for sk, _ in all_outlier:
    sess_counts[sk]['outlier'] += 1

labels_short = ['Neuron\nleads', 'Stable', 'Slope\ndown', 'Slope\nup', 'Outlier']
colors_bar = ['steelblue', 'black', 'darkorange', 'red', 'magenta']
means = [np.mean([sess_counts[sk][k] for sk in sess_counts])
         for k in ['cn', 'stable', 'down', 'up', 'outlier']]
sems = [np.std([sess_counts[sk][k] for sk in sess_counts]) /
        np.sqrt(len(sess_counts))
        for k in ['cn', 'stable', 'down', 'up', 'outlier']]
ax.bar(labels_short, means, yerr=sems, color=colors_bar, alpha=0.7,
       capsize=4, edgecolor='k', lw=0.5)
ax.set_ylabel('Neurons per session (mean ± SEM)')
ax.set_title(f'{len(session_data_store)} sessions')

plt.suptitle('Cross-session neuron classification', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_summary_classification.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 1b: ds/dg vs peak_r — circular vs split-half control ---
# For the split-half: peak_r from ODD trials, ds/dg from EVEN trials
# These use completely non-overlapping data.

port_lag_metrics = []       # ds/dg from full slopes (for classification/histogram)
port_lag_peak_r = []        # full-session xcorr peak
port_lag_peak_r_odd = []    # odd-trial xcorr peak
port_lag_dsdg_even = []     # ds/dg from even-trial slopes

for sess_key, nd, metric in all_sf_metrics:
    port_lag_metrics.append(metric)
    port_lag_peak_r.append(nd['peak_r'])
    port_lag_peak_r_odd.append(nd.get('peak_r_odd', np.nan))

    # Compute ds/dg from even-trial slopes
    slopes_even = nd.get('slopes_even', nd['slopes'])
    sd_s = session_data_store[sess_key]
    delta_slopes_e = []
    delta_gains_e = []
    valid_ei = [ei for ei in range(len(slopes_even))
                if np.isfinite(slopes_even[ei])]
    for ii in range(len(valid_ei)):
        for jj in range(ii + 1, len(valid_ei)):
            ei0, ei1 = valid_ei[ii], valid_ei[jj]
            s0, s1 = slopes_even[ei0], slopes_even[ei1]
            g0, g1 = sd_s['epoch_gains'][ei0], sd_s['epoch_gains'][ei1]
            if g0 > 0 and g1 > 0 and (g1 - g0) != 0:
                delta_slopes_e.append(s1 - s0)
                delta_gains_e.append(g1 - g0)
    if len(delta_slopes_e) > 0:
        port_lag_dsdg_even.append(np.mean(np.array(delta_slopes_e) /
                                          np.array(delta_gains_e)))
    else:
        port_lag_dsdg_even.append(np.nan)

port_lag_metrics = np.array(port_lag_metrics)
port_lag_peak_r = np.array(port_lag_peak_r)
port_lag_peak_r_odd = np.array(port_lag_peak_r_odd)
port_lag_dsdg_even = np.array(port_lag_dsdg_even)

from scipy.stats import pearsonr
n_bins_br = 8

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) Histogram of ds/dg
ax = axes[0]
ax.hist(port_lag_metrics, bins=50, color='gray', alpha=0.7, edgecolor='k', lw=0.3)
ax.axvline(0, color='k', lw=0.8)
ax.axvline(np.nanmean(port_lag_metrics), color='red', lw=1.5, ls='-',
           label=f'mean={np.nanmean(port_lag_metrics):.4f}')
ax.axvline(np.nanmedian(port_lag_metrics), color='blue', lw=1.5, ls='--',
           label=f'median={np.nanmedian(port_lag_metrics):.4f}')
ax.set_xlabel('ds/dg (Δslope / Δgain)')
ax.set_ylabel('Count')
ax.set_title(f'Port-lagging neurons (n={len(port_lag_metrics)})')
ax.legend(fontsize=8)

# (b) ds/dg (full) vs full-session peak_r (circular — for reference)
ax = axes[1]
valid_both = np.isfinite(port_lag_peak_r) & np.isfinite(port_lag_metrics)
xv = port_lag_peak_r[valid_both]
yv = port_lag_metrics[valid_both]
order = np.argsort(xv)
xv, yv = xv[order], yv[order]
row = len(xv) // n_bins_br
length = row * n_bins_br
xb = xv[:length].reshape(n_bins_br, row)
yb = yv[:length].reshape(n_bins_br, row)
ax.errorbar(np.mean(xb, axis=1), np.mean(yb, axis=1),
            np.std(yb, axis=1) / np.sqrt(row),
            marker='o', markersize=5, color='gray',
            markerfacecolor='gray', capsize=3)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Peak xcorr (full session)')
ax.set_ylabel('ds/dg (mean ± SEM)')
r_full, p_full = pearsonr(port_lag_peak_r[valid_both],
                           port_lag_metrics[valid_both])
ax.set_title(f'Circular: r={r_full:.3f}, p={p_full:.1e}')

# (c) ds/dg (EVEN trials) vs peak_r (ODD trials) — independent split-half
ax = axes[2]
valid_sh = (np.isfinite(port_lag_peak_r_odd) &
            np.isfinite(port_lag_dsdg_even))
n_valid_sh = np.sum(valid_sh)
if n_valid_sh > n_bins_br * 5:
    xv = port_lag_peak_r_odd[valid_sh]
    yv = port_lag_dsdg_even[valid_sh]
    order = np.argsort(xv)
    xv, yv = xv[order], yv[order]
    row = len(xv) // n_bins_br
    length = row * n_bins_br
    xb = xv[:length].reshape(n_bins_br, row)
    yb = yv[:length].reshape(n_bins_br, row)
    ax.errorbar(np.mean(xb, axis=1), np.mean(yb, axis=1),
                np.std(yb, axis=1) / np.sqrt(row),
                marker='o', markersize=5, color='k',
                markerfacecolor='k', capsize=3)
    r_sh, p_sh = pearsonr(port_lag_peak_r_odd[valid_sh],
                           port_lag_dsdg_even[valid_sh])
    ax.set_title(f'Split-half: r={r_sh:.3f}, p={p_sh:.1e}\n(n={n_valid_sh})')
else:
    ax.set_title(f'Split-half (n={n_valid_sh} — too few)')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Peak xcorr (odd trials)')
ax.set_ylabel('ds/dg from even trials (mean ± SEM)')

plt.suptitle('Gain modulation vs step coupling — circular vs split-half control',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_dsdg_vs_peak_r.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"Port-lagging neurons: n={len(port_lag_metrics)}")
print(f"  ds/dg (full): mean={np.nanmean(port_lag_metrics):.4f}, "
      f"median={np.nanmedian(port_lag_metrics):.4f}")
if n_valid_sh > 10:
    print(f"  Split-half: r={r_sh:.3f}, p={p_sh:.1e} (n={n_valid_sh})")

# --- Figure 2: pooled xcorr per group ---
group_names = ['Neurons lead', 'Port leads, stable',
               'Port leads, slope down', 'Port leads, slope up', 'Outlier']
group_data = [all_cn_like, sf_stable_all, sf_decrease_all, sf_increase_all, all_outlier]
group_colors = ['steelblue', 'black', 'darkorange', 'red', 'magenta']

fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)
for gi, (gname, gdata, gcol) in enumerate(zip(group_names, group_data, group_colors)):
    ax = axes[gi]
    if len(gdata) == 0:
        ax.set_title(f'{gname}\nn=0')
        continue
    xcorr_stack = np.array([nd['xcorr'] for _, nd in gdata])
    xcorr_mean = np.nanmean(xcorr_stack, axis=0)
    xcorr_sem = np.nanstd(xcorr_stack, axis=0) / np.sqrt(len(gdata))
    # Use lags from first session that has data
    first_sd = session_data_store[gdata[0][0]]
    lags_plot = first_sd['lags']
    ax.fill_between(lags_plot, xcorr_mean - xcorr_sem, xcorr_mean + xcorr_sem,
                    color=gcol, alpha=0.2)
    ax.plot(lags_plot, xcorr_mean, color=gcol, lw=1.5)
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.axhline(0, color='k', lw=0.3)
    ax.set_xlabel('Lag (s)')
    ax.set_title(f'{gname}\nn={len(gdata)}')
    ax.set_xlim(-MAX_LAG_SEC, MAX_LAG_SEC)
axes[0].set_ylabel('Cross-correlation')
plt.suptitle('Pooled cross-correlations by neuron type (all sessions)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_summary_xcorr.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 3: Regression slope vs relative gain per group ---
# Instead of pooling raw F/step values (which suffer from inter-session offsets),
# use the per-neuron per-epoch regression slopes already computed.
# For each neuron in each epoch, we have a slope and a relative gain.
# Plot mean slope (± SEM) as a function of relative gain bin.

REL_GAIN_BINS = [(0.5, 0.8, '0.5-0.8x'),
                 (0.8, 1.2, '~1x'),
                 (1.2, 2.0, '1.2-2x'),
                 (2.0, 3.0, '2-3x'),
                 (3.0, 10.0, '3x+')]
n_rel_bins = len(REL_GAIN_BINS)

fig, axes = plt.subplots(1, len(group_data), figsize=(3.5 * len(group_data), 4), sharey=True)

for gi, (gname, gdata, gcol) in enumerate(zip(group_names, group_data, group_colors)):
    ax = axes[gi]
    if len(gdata) == 0:
        ax.set_title(f'{gname}\nn=0')
        continue

    # Collect slopes per relative gain bin
    bin_slopes = {bi: [] for bi in range(n_rel_bins)}

    for sess_key, nd in gdata:
        sd_s = session_data_store[sess_key]
        for ei in range(len(nd['slopes'])):
            s = nd['slopes'][ei]
            if not np.isfinite(s):
                continue
            if ei >= len(sd_s['epoch_gains_rel']):
                continue
            rel_g = sd_s['epoch_gains_rel'][ei]
            if not np.isfinite(rel_g):
                continue
            for bi, (lo, hi, _) in enumerate(REL_GAIN_BINS):
                if lo <= rel_g < hi:
                    bin_slopes[bi].append(s)
                    break

    # Plot mean ± SEM per bin
    X, Y, SE = [], [], []
    for bi in range(n_rel_bins):
        vals = np.array(bin_slopes[bi])
        if len(vals) >= 5:
            X.append((REL_GAIN_BINS[bi][0] + REL_GAIN_BINS[bi][1]) / 2)
            Y.append(np.nanmean(vals))
            SE.append(np.nanstd(vals) / np.sqrt(len(vals)))

    if len(X) > 0:
        ax.errorbar(X, Y, SE, marker='o', markersize=6, color=gcol,
                    markerfacecolor=gcol, capsize=4, lw=1.5)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Relative gain (gain / gain₀)')
    ax.set_title(f'{gname}\nn={len(gdata)}')

    # Annotate bin counts
    for bi in range(n_rel_bins):
        n_bi = len(bin_slopes[bi])
        if n_bi > 0:
            mid_x = (REL_GAIN_BINS[bi][0] + REL_GAIN_BINS[bi][1]) / 2
            ax.text(mid_x, ax.get_ylim()[0], f'n={n_bi}', ha='center',
                    va='bottom', fontsize=6, color='gray')

axes[0].set_ylabel('F-vs-step regression slope')
plt.suptitle('Regression slope vs relative gain — pooled across sessions', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'spe_summary_slope_vs_gain.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 4: Summary statistics table ---
from scipy.stats import ttest_ind, mannwhitneyu

print("\n=== Summary statistics ===")
print(f"{'Group':<25} {'N':>5} {'mean ds/dg':>12} {'peak_lag':>10} {'peak_r':>10}")
print("-" * 65)
for gname, gdata in zip(group_names, group_data):
    if len(gdata) == 0:
        print(f"{gname:<25} {'0':>5}")
        continue
    lags = np.array([nd['peak_lag'] for _, nd in gdata])
    rs = np.array([nd['peak_r'] for _, nd in gdata])
    print(f"{gname:<25} {len(gdata):>5} "
          f"{'—':>12} "
          f"{np.mean(lags):>10.3f} {np.mean(rs):>10.4f}")

# ds/dg stats for the 3 step-following groups
for gname, gdata, key in [('Stable', sf_stable_all, 'stable'),
                           ('Slope down', sf_decrease_all, 'down'),
                           ('Slope up', sf_increase_all, 'up')]:
    if len(gdata) == 0:
        continue
    # recover metrics for these neurons
    metrics = []
    for sess_key, nd in gdata:
        for sk2, nd2, m in all_sf_metrics:
            if sk2 == sess_key and nd2['ci'] == nd['ci']:
                metrics.append(m)
                break
    if len(metrics) > 0:
        print(f"  {gname} ds/dg: mean={np.mean(metrics):.4f}, "
              f"std={np.std(metrics):.4f}")
