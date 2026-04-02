#%% ============================================================================
# Figure: Gain modulation schematic (standalone)
#
# Introduces how threshold changes modulate the mapping from CN fluorescence
# to reward port speed. Uses example session BCI105 020425.
#
# Panel A: Transfer function for each epoch — fluorescence vs reward port speed
#          with fluorescence distribution from reference trials overlaid
# Panel B: Example hit trial from epoch 0 — CN dF/F + cumulative lickport steps
# Panel C: Example hit trial from hard epoch — same layout, matched y-axes
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib
matplotlib.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'font.family': 'sans-serif',
    'svg.fonttype': 'none',
})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import data_dict_create_module_test as ddct
from BCI_data_helpers import parse_hdf5_array_string
from bci_time_series import bci_time_series_fun

# --- Output directory ---
FIG_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
           r'\3-factor learning paper\claude code 032226'
           r'\meta_analysis_results\panels')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# Load data for BCI105 020425
# ============================================================================
EX_MOUSE = 'BCI105'
EX_SESSION = '020425'
folder = (r'//allen/aind/scratch/BCI/2p-raw/'
          + EX_MOUSE + '/' + EX_SESSION + '/pophys/')
print(f"Loading {EX_MOUSE} {EX_SESSION}")

bci_keys = ['F', 'df_closedloop', 'conditioned_neuron', 'dt_si',
            'step_time', 'reward_time', 'BCI_thresholds',
            'roi_csv', 'cn_csv_index']
data = ddct.load_hdf5(folder, bci_keys, [])

F = data['F']
trl = F.shape[2]
dt_si = data['dt_si']
cn = data['conditioned_neuron'][0][0]

# Continuous CN dF/F trace (all neurons x all frames)
df_cl = data['df_closedloop']      # shape: [n_neurons, n_total_frames]
cn_df = df_cl[cn, :]               # CN continuous dF/F

# ops for frames_per_file
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy',
              allow_pickle=True).tolist()
frames_per_file = ops['frames_per_file']

# Parse behavioral arrays
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
rt = np.array([x[0] if len(x) > 0 else np.nan
               for x in data['reward_time']], dtype=float)
hit = np.isfinite(rt)

# Build step_vector (binary, frame resolution)
step_vector, reward_vector, trial_start_vector = bci_time_series_fun(
    folder, data, rt, dt_si)

# --- BCI thresholds: forward-fill NaNs, pad to trl ---
BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
thr_lower = BCI_thresholds[0, :].copy()
thr_upper = BCI_thresholds[1, :].copy()
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

# Detect threshold switch trials
d_upper = np.diff(thr_upper)
switches = np.where((d_upper != 0) & np.isfinite(d_upper))[0] + 1
switches = np.concatenate(([0], switches))
n_epochs = len(switches)
epoch_ends = np.concatenate((switches[1:], [trl]))

print(f"{trl} trials, {n_epochs} threshold epochs")
for ei in range(n_epochs):
    t0, t1 = switches[ei], epoch_ends[ei]
    print(f"  Epoch {ei}: trials {t0}-{t1-1}, "
          f"lower={thr_lower[t0]:.1f}, upper={thr_upper[t0]:.1f}")

# --- Transfer function ---
def transfer_fun(fluorescence, lower, upper, max_speed=3.3):
    gain = upper - lower
    if gain <= 0:
        return np.zeros_like(fluorescence)
    speed = (fluorescence - lower) / gain * max_speed
    speed = np.clip(speed, 0, max_speed)
    return speed

# --- Raw CN fluorescence from roi_csv (needed for transfer function x-axis) ---
from scipy.interpolate import interp1d
roi = np.copy(data['roi_csv'])
inds_wrap = np.where(np.diff(roi[:, 1]) < 0)[0]
for i in range(len(inds_wrap)):
    ind = inds_wrap[i]
    roi[ind+1:, 1] += roi[ind, 1]
    roi[ind+1:, 0] += roi[ind, 0]
frm_ind = np.arange(1, int(np.max(roi[:, 1])) + 1)
interp_func = interp1d(roi[:, 1], roi, axis=0, kind='linear',
                        fill_value='extrapolate')
roi_interp = interp_func(frm_ind)
cn_ind = data['cn_csv_index'][0]
cn_raw = roi_interp[:, cn_ind + 2]  # raw online fluorescence (for transfer fun)

# ============================================================================
# BUILD FIGURE
# ============================================================================
fig = plt.figure(figsize=(5.5, 1.3))
gs = GridSpec(1, 2, figure=fig, left=0.10, right=0.97, bottom=0.25, top=0.90,
              wspace=0.4, width_ratios=[1, 3.5])

# Shades of gray: light (early) to dark (late)
epoch_colors = [plt.cm.Greys(v) for v in np.linspace(0.3, 0.9, n_epochs)]

# --- Panel A: Transfer function curves ---
ax_a = fig.add_subplot(gs[0, 0])

f_range = np.linspace(0, np.percentile(cn_raw, 99.5), 300)
for ei in range(n_epochs):
    t0 = switches[ei]
    speed_e = transfer_fun(f_range, thr_lower[t0], thr_upper[t0])
    ax_a.plot(f_range, speed_e, color=epoch_colors[ei], linewidth=1.2,
              label=f'Epoch {ei + 1}')

ax_a.set_xlabel('CN fluorescence (a.u.)')
ax_a.set_ylabel('Reward port speed')
ax_a.set_ylim(-0.1, 3.5)
ax_a.legend(frameon=False, loc='upper left', fontsize=5)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# --- Helper: get per-trial traces from df_closedloop and step_vector ---
trial_starts = np.cumsum([0] + list(frames_per_file[:-1]))

def get_trial_traces(trial_idx, smooth_tau=20):
    """Return (time_s, cn_dff, step_rate) for a trial.
    step_rate is the binary step vector smoothed with a causal exponential kernel."""
    t_start = int(trial_starts[trial_idx])
    n_frames = int(frames_per_file[trial_idx])

    t_s = np.arange(n_frames) * dt_si
    dff = cn_df[t_start:t_start + n_frames]
    steps_binary = step_vector[t_start:t_start + n_frames].astype(float)

    # Smooth with causal exponential kernel to get step rate
    kernel = np.exp(-np.arange(int(5 * smooth_tau)) / smooth_tau)
    kernel /= kernel.sum()
    step_rate = np.convolve(steps_binary, kernel, mode='full')[:n_frames]

    return t_s, dff, step_rate

# --- Pick one example hit trial per epoch ---
def pick_trial(epoch_idx, target_pctile=50):
    t0, t1 = switches[epoch_idx], epoch_ends[epoch_idx]
    trials_in_epoch = np.arange(t0, t1)
    hit_trials = trials_in_epoch[hit[trials_in_epoch]]
    if len(hit_trials) == 0:
        return None
    rts = rt[hit_trials]
    target = np.nanpercentile(rts[np.isfinite(rts)], target_pctile)
    return hit_trials[np.argmin(np.abs(rts - target))]

# --- Panel B: One trial per epoch, CN dF/F above speed ---
ax_b = fig.add_subplot(gs[0, 1])

TRIAL_DUR = 10.0   # max duration per trial
GAP = 0.5          # seconds of blank between trials
REWARD_DIST_MM = 7.0  # lickport travel distance to reward

# Compute step size (mm) from a reference hit trial: 7mm / total steps
ref_hit = np.where(hit)[0]
n_steps_ref = []
for ri in ref_hit[:20]:  # use first 20 hit trials
    steps_i = data['step_time'][ri]
    if steps_i is not None and len(steps_i) > 0:
        n_steps_ref.append(len(steps_i))
step_size_mm = REWARD_DIST_MM / np.median(n_steps_ref)
print(f"Step size: {step_size_mm:.3f} mm ({int(np.median(n_steps_ref))} steps to reward)")

# First pass: collect traces, truncated at reward time
# step rate is converted to mm/s
all_traces = []
for ei in range(n_epochs):
    ti = pick_trial(ei, 2)
    if ti is not None:
        t_s, dff, srate = get_trial_traces(ti)
        # Convert step rate from steps/frame to mm/s
        speed_mm_s = srate * step_size_mm / dt_si
        # Truncate at reward (or TRIAL_DUR for misses)
        t_end = rt[ti] if np.isfinite(rt[ti]) else TRIAL_DUR
        t_end = min(t_end, TRIAL_DUR)
        mask = t_s <= t_end
        all_traces.append((ei, ti, t_s[mask], dff[mask], speed_mm_s[mask]))

# Compute real-unit ranges for scale bars
all_dff = np.concatenate([d for _, _, _, d, _ in all_traces])
all_spd = np.concatenate([s for _, _, _, _, s in all_traces])
dff_lo, dff_hi = np.percentile(all_dff, 1), np.percentile(all_dff, 99)
spd_lo, spd_hi = 0, np.percentile(all_spd, 99)
dff_range = dff_hi - dff_lo if dff_hi > dff_lo else 1
spd_range = spd_hi - spd_lo if spd_hi > spd_lo else 1

BAND_HEIGHT = 1.0   # each trace band spans this in plot units
GAP_BAND = 0.3      # vertical gap between speed and dF/F bands

# Plot each epoch's trial sequentially
x_cursor = 0.0
epoch_centers = []
for ei, ti, t_s, dff, spd in all_traces:
    col = epoch_colors[ei]
    # Speed (bottom band) — normalized to [0, BAND_HEIGHT]
    spd_norm = (spd - spd_lo) / spd_range * BAND_HEIGHT
    ax_b.plot(x_cursor + t_s, spd_norm, color=col, linewidth=0.8, alpha=0.8)
    # CN dF/F (top band) — normalized to [0, BAND_HEIGHT] then offset
    dff_norm = (dff - dff_lo) / dff_range * BAND_HEIGHT + BAND_HEIGHT + GAP_BAND
    ax_b.plot(x_cursor + t_s, dff_norm, color=col, linewidth=0.5, alpha=0.7)
    trial_len = t_s[-1] if len(t_s) > 0 else TRIAL_DUR
    epoch_centers.append(x_cursor + trial_len / 2)
    x_cursor += trial_len + GAP

# Axis formatting — epoch labels on x-axis
ax_b.set_xlim(-0.5, x_cursor - GAP + 0.5)
ax_b.set_ylim(-0.25, 2 * BAND_HEIGHT + GAP_BAND + 0.15)
ax_b.set_xticks(epoch_centers)
ax_b.set_xticklabels([str(e + 1) for e, _, _, _, _ in all_traces])
ax_b.set_xlabel('Epoch')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.spines['bottom'].set_visible(False)
ax_b.spines['left'].set_visible(False)
ax_b.tick_params(axis='x', length=0)

# Horizontal 1s scale bar under the first epoch
sb_x0 = 0.0
sb_x1 = 1.0
sb_y = -0.15
ax_b.plot([sb_x0, sb_x1], [sb_y, sb_y], 'k', linewidth=1.5, clip_on=False)
ax_b.text((sb_x0 + sb_x1) / 2, sb_y - 0.06, '1 s', ha='center', va='top',
          fontsize=7)

# Vertical scale bars with labels — placed at left edge of first trace
sb_x = -0.8  # x position in data coords, just left of traces

# Speed scale bar
spd_per_unit = spd_range / BAND_HEIGHT
spd_bar_val = round(spd_per_unit * 0.4, 1)
if spd_bar_val < 0.1:
    spd_bar_val = 0.1
spd_bar_height = spd_bar_val / spd_per_unit
spd_bar_y0 = BAND_HEIGHT / 2 - spd_bar_height / 2
ax_b.plot([sb_x, sb_x], [spd_bar_y0, spd_bar_y0 + spd_bar_height],
          'k', linewidth=1.5, clip_on=False)
ax_b.text(sb_x - 0.2, spd_bar_y0 + spd_bar_height / 2,
          f'{spd_bar_val:.1f} mm/s',
          ha='right', va='center', fontsize=6, clip_on=False)

# dF/F scale bar
dff_per_unit = dff_range / BAND_HEIGHT
dff_bar_val = 3.0
dff_bar_height = dff_bar_val / dff_per_unit
dff_band_mid = BAND_HEIGHT + GAP_BAND + BAND_HEIGHT / 2
dff_bar_y0 = dff_band_mid - dff_bar_height / 2
ax_b.plot([sb_x, sb_x], [dff_bar_y0, dff_bar_y0 + dff_bar_height],
          'k', linewidth=1.5, clip_on=False)
ax_b.text(sb_x - 0.2, dff_bar_y0 + dff_bar_height / 2,
          f'{dff_bar_val:.1f} dF/F',
          ha='right', va='center', fontsize=6, clip_on=False)

ax_b.set_yticks([])

# ============================================================================
# Save
# ============================================================================
fname = 'figure_gain_modulation'
fig.savefig(os.path.join(FIG_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(FIG_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {fname}.png and {fname}.svg")
