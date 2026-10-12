#%% ============================================================================
# Figure: Threshold-driven learning in the BCI task
#
# This script assumes threshold_analysis.py has been run through Cell 10,
# populating the following workspace variables:
#   - all_epoch_stats, all_session_trials
#   - hr_aligned, rt_aligned, cn_aligned, rt_rpe_aligned
#   - exp_hr_aligned, exp_rt_aligned, thr_direction, mask_inc
#   - exp_rt_change, rt_improvement, switch_epochs_rt
#   - trial_axis, PRE, POST
#
# Layout (7.5 x 4 inches):
#   Row 1 (4 panels): Example session (BCI105 020425)
#     A: Hit rate + expected    B: Raw fluorescence + 1/gain overlay
#     C: CN heatmap (trials x time)   D: Actual vs expected RT bars
#   Row 2 (5 panels): Population summary (threshold increases only)
#     E: Switch-aligned RT with expected    F: Switch-aligned CN (z-score)
#     G: RPE aligned to transitions         H: RT improvement vs expected dRT
#     I: Actual vs expected hit rate
# ============================================================================
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
import os

# --- Output directory ---
FIG_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written\3-factor learning paper\claude code 032226\meta_analysis_results\panels'

# ============================================================================
# Example session: BCI105 020425
# ============================================================================
EX_MOUSE = 'BCI105'
EX_SESSION = '020425'

# --- Load example session data from workspace ---
ex_data = all_session_trials[(EX_MOUSE, EX_SESSION)]
ex_hit = ex_data['hit']
ex_rt = ex_data['rt'].copy()
ex_rt[~np.isfinite(ex_rt)] = 10.0
ex_switches = ex_data['switches']
ex_thr_u = ex_data['thr_u']
ex_trl = len(ex_hit)

# Get epoch stats for this session
ex_epoch_stats = [r for r in all_epoch_stats
                  if r['mouse'] == EX_MOUSE and r['session'] == EX_SESSION]
ex_epoch_stats.sort(key=lambda r: r['epoch'])

# Build expected HR trace
ex_exp_hr = np.full(ex_trl, np.nan)
for r in ex_epoch_stats:
    if np.isfinite(r.get('expected_hr_correct', np.nan)):
        ex_exp_hr[r['trial_start']:r['trial_end']] = r['expected_hr_correct']

# Build 1/gain trace normalized to epoch 0 (so it starts at 1)
g0 = ex_epoch_stats[0]['upper'] - ex_epoch_stats[0]['lower']
ex_gain = np.full(ex_trl, np.nan)
for r in ex_epoch_stats:
    g = r['upper'] - r['lower']
    ex_gain[r['trial_start']:r['trial_end']] = g0 / g if g > 0 else np.nan

# CN activity per trial
ex_cn = ex_data['cn']

# ============================================================================
# BUILD FIGURE
# ============================================================================
fig = plt.figure(figsize=(6, 2.5))
gs_top = GridSpec(1, 4, figure=fig,
                  left=0.07, right=0.95, bottom=0.56, top=0.95,
                  wspace=0.85)
gs_bot = GridSpec(1, 5, figure=fig,
                  left=0.07, right=0.95, bottom=0.10, top=0.46,
                  wspace=0.85)

win = 10  # smoothing window

# --- Panel A: Hit rate + expected ---
ax_a = fig.add_subplot(gs_top[0, 0])
hr_smooth = np.convolve(ex_hit.astype(float), np.ones(win)/win, mode='valid')
ax_a.plot(np.arange(win-1, win-1+len(hr_smooth)), hr_smooth, 'k', linewidth=0.8)
ax_a.plot(ex_exp_hr, color='gray', linewidth=1.0)
for sw in ex_switches[1:]:
    ax_a.axvline(sw, ymin=0, ymax=0.06, color='k', linewidth=0.8)
ax_a.set_xlim(win-1, ex_trl)
ax_a.set_xlabel('Trial')
ax_a.set_ylabel('Hit rate')
ax_a.set_ylim(-0.05, 1.05)
ax_a.legend(['Actual', 'Expected'], frameon=False, loc='lower left')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# --- Panel B: CN dF/F + 1/gain (difficulty) overlay ---
ax_b = fig.add_subplot(gs_top[0, 1])

# Build continuous raw fluorescence from roi_csv
from scipy.interpolate import interp1d
folder_ex = (r'//allen/aind/scratch/BCI/2p-raw/'
             + EX_MOUSE + '/' + EX_SESSION + '/pophys/')
import data_dict_create_module_test as ddct_b
data_b = ddct_b.load_hdf5(folder_ex, ['roi_csv', 'cn_csv_index'], [])
roi_b = np.copy(data_b['roi_csv'])
wraps_b = np.where(np.diff(roi_b[:, 1]) < 0)[0]
for wi in range(len(wraps_b)):
    ww = wraps_b[wi]
    roi_b[ww+1:, 1] += roi_b[ww, 1]
    roi_b[ww+1:, 0] += roi_b[ww, 0]
frm_b = np.arange(1, int(np.max(roi_b[:, 1])) + 1)
ifunc_b = interp1d(roi_b[:, 1], roi_b, axis=0, kind='linear', fill_value='extrapolate')
roi_interp_b = ifunc_b(frm_b)
cn_idx_b = data_b['cn_csv_index'][0]
t_cont_b = roi_interp_b[:, 0]
cn_cont_b = roi_interp_b[:, cn_idx_b + 2]

# Convert to dF/F using 20th percentile as baseline
f0_b = np.percentile(cn_cont_b, 20)
cn_dfof_b = (cn_cont_b - f0_b) / f0_b

ax_b.plot(t_cont_b, cn_dfof_b, 'k', linewidth=0.04)
ax_b.set_xlabel('Time (s)')
ax_b.set_ylabel('CN dF/F')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# Overlay 1/gain (difficulty) on twin axis
ops_b = np.load(folder_ex + r'/suite2p_BCI/plane0/ops.npy',
                allow_pickle=True).tolist()
fpf_b = ops_b['frames_per_file']
diff_time_b = np.full(len(t_cont_b), np.nan)
strt_b = 0
for ti_b in range(min(ex_trl, len(fpf_b))):
    idx_b = np.arange(strt_b, strt_b + fpf_b[ti_b], dtype=int)
    idx_b = np.clip(idx_b, 0, len(diff_time_b) - 1)
    diff_time_b[idx_b] = ex_gain[ti_b]
    strt_b += fpf_b[ti_b]

ax_b2 = ax_b.twinx()
ax_b2.plot(t_cont_b, diff_time_b, color='cornflowerblue', linewidth=0.8, alpha=0.8)
ax_b2.set_ylabel('1/gain', color='cornflowerblue', labelpad=1)
ax_b2.tick_params(axis='y', colors='cornflowerblue', pad=1)
ax_b2.spines['top'].set_visible(False)
ax_b2.spines['left'].set_visible(False)

# --- Panel C: CN heatmap ---
ax_c = fig.add_subplot(gs_top[0, 2])
import data_dict_create_module_test as ddct_c
data_ex = ddct_c.load_hdf5(folder_ex, ['F', 'conditioned_neuron', 'dt_si'], [])
F_ex = data_ex['F']
cn_ex = data_ex['conditioned_neuron'][0][0]
dt_si_ex = data_ex['dt_si']

im_c = ax_c.imshow(F_ex[:, cn_ex, :].T, aspect='auto', interpolation='nearest')
ax_c.set_xlabel('Time (s)')
ax_c.set_ylabel('Trial')
frames_10s = int(10.0 / dt_si_ex)
n_fr_trial = F_ex.shape[0]
ax_c.set_xticks([0, min(frames_10s, n_fr_trial-1)])
ax_c.set_xticklabels(['0', '10'])
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
cb_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
cb_c.set_label('dF/F', fontsize=7)

# --- Panel D: Actual vs expected RT bars ---
ax_d = fig.add_subplot(gs_top[0, 3])
n_ep = len(ex_epoch_stats)
x_bar = np.arange(n_ep)
act_rt_ep = np.array([r['actual_rt'] for r in ex_epoch_stats])
exp_rt_ep = np.array([r.get('expected_rt', np.nan) for r in ex_epoch_stats])
w = 0.35
ax_d.bar(x_bar - w/2, act_rt_ep, width=w, color='k', label='Actual')
ax_d.bar(x_bar + w/2, exp_rt_ep, width=w, color='gray', label='Expected')
ax_d.set_xlabel('Epoch')
ax_d.set_ylabel('Reward time (s)')
ax_d.set_xticks(x_bar)
ax_d.legend(frameon=False)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

# ============================================================================
# Population panels (threshold increases only)
# ============================================================================

def _mean_sem(arr, axis=0):
    m = np.nanmean(arr, axis=axis)
    n = np.sum(np.isfinite(arr), axis=axis)
    s = np.nanstd(arr, axis=axis) / np.sqrt(np.clip(n, 1, None))
    return m, s

# Filter to threshold increases
hr_sub = hr_aligned[mask_inc]
rt_sub = rt_aligned[mask_inc]
cn_sub = cn_aligned[mask_inc]
rt_rpe_sub = rt_rpe_aligned[mask_inc]
exp_hr_sub = exp_hr_aligned[mask_inc]
exp_rt_sub = exp_rt_aligned[mask_inc]
n_inc = np.sum(mask_inc)

# --- Panel E: Switch-aligned RT with expected ---
ax_e = fig.add_subplot(gs_bot[0, 0])
rt_m, rt_s = _mean_sem(rt_sub)
exp_rt_m = np.nanmean(exp_rt_sub, axis=0)
ax_e.fill_between(trial_axis, rt_m - rt_s, rt_m + rt_s, color='k', alpha=0.15)
ax_e.plot(trial_axis, rt_m, 'k', linewidth=1.0)
ax_e.plot(trial_axis, exp_rt_m, color='cornflowerblue', linewidth=1.0)
ax_e.axvline(0, color='r', linewidth=0.6, linestyle='--')
ax_e.set_xlabel('Trials from thr. change')
ax_e.set_ylabel('Reward time (s)')
ax_e.set_xlim(-PRE, POST - 1)
ax_e.legend(['Actual', 'Expected'], frameon=False)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)

# --- Panel F: Switch-aligned CN activity (z-scored) ---
ax_f = fig.add_subplot(gs_bot[0, 1])
cn_pre_mean = np.nanmean(cn_sub[:, :PRE], axis=1, keepdims=True)
cn_pre_std = np.nanstd(cn_sub[:, :PRE], axis=1, keepdims=True)
cn_z = (cn_sub - cn_pre_mean) / np.where(cn_pre_std > 1e-6, cn_pre_std, np.nan)
cn_z_m, cn_z_s = _mean_sem(cn_z)
ax_f.fill_between(trial_axis, cn_z_m - cn_z_s, cn_z_m + cn_z_s, color='k', alpha=0.15)
ax_f.plot(trial_axis, cn_z_m, 'k', linewidth=1.0)
ax_f.axvline(0, color='r', linewidth=0.6, linestyle='--')
ax_f.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax_f.set_xlabel('Trials from thr. change')
ax_f.set_ylabel('CN activity (z)')
ax_f.set_xlim(-PRE, POST - 1)
ax_f.spines['top'].set_visible(False)
ax_f.spines['right'].set_visible(False)

# --- Panel G: RPE aligned to transitions ---
ax_g = fig.add_subplot(gs_bot[0, 2])
rt_rpe_m, rt_rpe_s = _mean_sem(rt_rpe_sub)
ax_g.fill_between(trial_axis, rt_rpe_m - rt_rpe_s, rt_rpe_m + rt_rpe_s,
                  color='k', alpha=0.15)
ax_g.plot(trial_axis, rt_rpe_m, 'k', linewidth=1.0)
ax_g.axvline(0, color='r', linewidth=0.6, linestyle='--')
ax_g.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax_g.set_xlabel('Trials from thr. change')
ax_g.set_ylabel('RT RPE (s)')
ax_g.set_xlim(-PRE, POST - 1)
ax_g.spines['top'].set_visible(False)
ax_g.spines['right'].set_visible(False)

# --- Panel H: RT improvement vs expected dRT ---
ax_h = fig.add_subplot(gs_bot[0, 3])
ax_h.scatter(exp_rt_change, rt_improvement, s=12, c='k', alpha=0.4, edgecolors='none')
ax_h.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax_h.axvline(0, color='gray', linewidth=0.5, linestyle=':')
ax_h.set_xlabel('Expected \u0394RT (s)')
ax_h.set_ylabel('RT improvement (s)')
ax_h.spines['top'].set_visible(False)
ax_h.spines['right'].set_visible(False)

# --- Panel I: Actual vs expected hit rate (threshold increases) ---
ax_i = fig.add_subplot(gs_bot[0, 4])
# Collect per-epoch actual and expected HR for threshold increases (epoch > 0)
actual_hrs = []
expected_hrs = []
# Group epoch stats by session to identify threshold increases
from collections import defaultdict
_sess_epochs = defaultdict(list)
for r in all_epoch_stats:
    _sess_epochs[(r['mouse'], r['session'])].append(r)
for key in _sess_epochs:
    eps = sorted(_sess_epochs[key], key=lambda x: x['epoch'])
    for j in range(1, len(eps)):
        if eps[j]['upper'] > eps[j-1]['upper']:  # threshold increase
            if np.isfinite(eps[j].get('expected_hr_correct', np.nan)):
                actual_hrs.append(eps[j]['actual_hr'])
                expected_hrs.append(eps[j]['expected_hr_correct'])
actual_hrs = np.array(actual_hrs)
expected_hrs = np.array(expected_hrs)

ax_i.scatter(expected_hrs, actual_hrs, s=12, c='k', alpha=0.4, edgecolors='none')
lims = [0, 1.05]
ax_i.plot(lims, lims, color='gray', linewidth=0.5, linestyle=':')
ax_i.set_xlim(lims)
ax_i.set_ylim(lims)
ax_i.set_xlabel('Expected HR')
ax_i.set_ylabel('Actual HR')
ax_i.spines['top'].set_visible(False)
ax_i.spines['right'].set_visible(False)

# ============================================================================
# Save
# ============================================================================
fname = 'figure_threshold_learning'
fig.savefig(os.path.join(FIG_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(FIG_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {fname}.png and {fname}.svg")
