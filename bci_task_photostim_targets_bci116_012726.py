"""
What are the photostim-identified target cells doing during the BCI task?

For BCI116/012826, identify the cells whose direct photostim response was
> 0.05 dF/F (responsive targets, the same set we used in the I→I matrix
analysis), then look at their activity during the closed-loop BCI session
aligned to trial start and reward.

Modeled on pan_neuronal_analysis2/outlier_event_aligned.py but for a single
session and using direct photostim targets instead of statistical outliers.
"""

#%% ============================================================================
# CELL 1: Imports + setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() \
            else r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI'
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import pearsonr

import data_dict_create_module_test as ddct
import bci_time_series as bts
from BCI_data_helpers import (
    compute_amp_from_photostim_artifact_free,
    parse_hdf5_array_string,
    get_reward_aligned_df_truncated,
    get_trial_aligned_df_padded,
    compute_rpe,
)


def _load_stim_params_from_h5(folder, ps_epoch):
    """Pull the stim_params group from a photostim h5 (handles scalar datasets)."""
    all_h5 = sorted(glob.glob(os.path.join(folder, '*.h5')))
    if ps_epoch == 'photostim':
        cand = [p for p in all_h5
                if 'photostim' in os.path.basename(p).lower()
                and 'photostim2' not in os.path.basename(p).lower()]
    else:
        cand = [p for p in all_h5
                if 'photostim2' in os.path.basename(p).lower()]
    if not cand:
        raise FileNotFoundError(f'No h5 for {ps_epoch} in {folder}')
    sp = {}
    with h5py.File(cand[0], 'r') as f:
        if 'stim_params' in f:
            for k in f['stim_params'].keys():
                try:
                    sp[k] = f['stim_params'][k][()]
                except Exception:
                    pass
    return sp

# -----------------------------
# Session
# -----------------------------
mouse   = 'BCI116'
session = '012826'
folder  = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'

SAVE_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(SAVE_DIR, exist_ok=True)

REWARD_WINDOW     = (-4, 10)
TRIAL_WINDOW      = (-2, 4)
MEDFILT_KERNEL    = 11
TARGET_AMP_THRESH = 0.05         # dF/F threshold for "responsive target"


#%% ============================================================================
# CELL 2: Load via ddct.load_hdf5 (mirrors sliding_window_temporal_offset_v2.py
# line 109 exactly). Pull stim_params from h5 separately so the artifact-free
# amp function can find it.
# ============================================================================
photostim_keys = ['stimDist', 'favg_raw']
bci_keys = [
    'df_closedloop', 'F', 'mouse', 'session',
    'conditioned_neuron', 'dt_si', 'step_time', 'reward_time',
]
data = ddct.load_hdf5(folder, bci_keys, photostim_keys)

if 'photostim' in data:
    data['photostim']['stim_params']  = _load_stim_params_from_h5(folder, 'photostim')
if 'photostim2' in data:
    data['photostim2']['stim_params'] = _load_stim_params_from_h5(folder, 'photostim2')

dt_si = data['dt_si']
F = data['F']
trl = F.shape[2]
print(f'Loaded {mouse} {session}: F shape = {F.shape}, dt_si = {dt_si:.4f}')


#%% ============================================================================
# CELL 3: Identify responsive target cells
# ============================================================================
AMP, stimDist = compute_amp_from_photostim_artifact_free(mouse, data, folder)
amp_ep = AMP[-1]                          # photostim2 if present, else photostim
n_cells  = F.shape[1]
n_groups = amp_ep.shape[1]

target_cell_idx = np.argmin(stimDist, axis=0)
target_amp = amp_ep[target_cell_idx, np.arange(n_groups)]

is_target = np.zeros(n_cells, dtype=bool)
for gi in range(n_groups):
    if np.isfinite(target_amp[gi]) and target_amp[gi] > TARGET_AMP_THRESH:
        is_target[target_cell_idx[gi]] = True
is_other = ~is_target
print(f'Responsive target cells: {is_target.sum()}')
print(f'Other cells:             {is_other.sum()}')


#%% ============================================================================
# CELL 4: Event vectors + aligned activity (verbatim pattern from outlier_event_aligned.py)
# ============================================================================
import pickle

def _decode_event_times(v, trl):
    """For BCI116-era data: step_time / reward_time come out of ddct.load_hdf5
    as a 0-d ndarray containing pickled bytes. The pickle decodes to a sequence
    of length `trl` where each entry is a numpy array of shape (1, n_events).
    Flatten each to 1-d so downstream `x[0]` gives a scalar."""
    raw = v.item() if hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0 else v
    if isinstance(raw, (bytes, np.bytes_)):
        try:
            obj = pickle.loads(raw)
            return [np.asarray(x).ravel() for x in obj]
        except Exception:
            pass
    return parse_hdf5_array_string(v, trl)

data['step_time']   = _decode_event_times(data['step_time'],   trl)
data['reward_time'] = _decode_event_times(data['reward_time'], trl)

rt = np.array([x[0] if x.size > 0 else np.nan
               for x in data['reward_time']], dtype=float)
rt_filled = rt.copy()
rt_filled[~np.isfinite(rt_filled)] = 30.0
hit = np.isfinite(rt)

print(f'reward_time first 5 (after decode): '
      f'{[x.tolist() for x in data["reward_time"][:5]]}')
print(f'step_time   first 3 (after decode): '
      f'{[x.tolist() for x in data["step_time"][:3]]}')
print(f'rt: hits={int(hit.sum())}/{len(rt)}, '
      f'first 10 rt: {rt[:10]}')

step_vector, reward_vector, trial_start_vector = \
    bts.bci_time_series_fun(folder, data, rt_filled, dt_si)

# ---- Diagnostics so we can sanity-check the event vectors ----
def _summary(name, v):
    v = np.asarray(v)
    nz = int(np.count_nonzero(v))
    print(f'  {name}: shape={v.shape}, nonzero={nz}, '
          f'first nonzero={int(np.argmax(v != 0)) if nz else "n/a"}, '
          f'sum={float(np.nansum(v)):.3f}')

print(f'trl (n trials from F): {trl}')
print(f'len(step_time)  : {len(data["step_time"])}')
print(f'len(reward_time): {len(data["reward_time"])}')
print(f'rt: total={len(rt)}, hits={int(hit.sum())}, '
      f'min={np.nanmin(rt):.2f}, max={np.nanmax(rt):.2f}')
_summary('step_vector       ', step_vector)
_summary('reward_vector     ', reward_vector)
_summary('trial_start_vector', trial_start_vector)

df = data['df_closedloop']
rta, t_reward = get_reward_aligned_df_truncated(
    df, reward_vector, trial_start_vector, dt_si, window=REWARD_WINDOW)
sta, t_trial = get_trial_aligned_df_padded(
    df, trial_start_vector, reward_vector, dt_si, window=TRIAL_WINDOW)
print(f'rta shape: {rta.shape}  (time, cells, rewards)')
print(f'sta shape: {sta.shape}  (time, cells, trials)')


#%% ============================================================================
# CELL 5: Per-target traces (stacked, trial-start + reward aligned)
# ============================================================================
target_idx_list = np.where(is_target)[0]
n_targ = len(target_idx_list)

rr_targ = np.nanmean(rta[:, target_idx_list, :], axis=2)   # (time, n_targ)
ss_targ = np.nanmean(sta[:, target_idx_list, :], axis=2)

fig, axes = plt.subplots(1, 2, figsize=(12, max(3, 0.25 * n_targ)), sharey=True)

ax = axes[0]
for i in range(n_targ):
    ax.plot(t_trial, medfilt(ss_targ[:, i], MEDFILT_KERNEL) + i * 0.05,
            lw=0.7, color='black', alpha=0.7)
ax.axvline(0, color='k', lw=0.8, alpha=0.5)
ax.set_xlabel('Time from trial start (s)')
ax.set_ylabel('Target cell  (offset for visibility)')
ax.set_title(f'{mouse} {session}: target cells, trial-start aligned (n={n_targ})')

ax = axes[1]
for i in range(n_targ):
    ax.plot(t_reward, medfilt(rr_targ[:, i], MEDFILT_KERNEL) + i * 0.05,
            lw=0.7, color='black', alpha=0.7)
ax.axvline(0, color='k', lw=0.8, alpha=0.5)
ax.set_xlabel('Time from reward (s)')
ax.set_title('reward aligned')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_target_cells_traces.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%% ============================================================================
# CELL 6: Population averages -- targets vs other cells
# ============================================================================
def _pop_mean_sem(arr, mask):
    """arr: (time, cells, trials). Average over trials per cell, then over cells."""
    per_cell = np.nanmean(arr[:, mask, :], axis=2)     # (time, n_in_mask)
    m  = np.nanmean(per_cell, axis=1)
    se = np.nanstd(per_cell, axis=1, ddof=1) / np.sqrt(np.sum(mask))
    return m, se

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Trial-start aligned
ax = axes[0]
for mask, label, color in [
    (is_target, f'Targets (n={is_target.sum()})', '#d62728'),
    (is_other,  f'Other (n={is_other.sum()})',     'black'),
]:
    m, se = _pop_mean_sem(sta, mask)
    ax.plot(t_trial, medfilt(m, MEDFILT_KERNEL), color=color, label=label)
    ax.fill_between(t_trial, medfilt(m - se, MEDFILT_KERNEL),
                    medfilt(m + se, MEDFILT_KERNEL), color=color, alpha=0.2)
ax.axvline(0, color='k', lw=0.8, alpha=0.5)
ax.set_xlabel('Time from trial start (s)')
ax.set_ylabel('ΔF/F')
ax.set_title('Trial-start aligned')
ax.legend(frameon=False)

# Reward aligned
ax = axes[1]
for mask, label, color in [
    (is_target, f'Targets (n={is_target.sum()})', '#d62728'),
    (is_other,  f'Other (n={is_other.sum()})',     'black'),
]:
    m, se = _pop_mean_sem(rta, mask)
    ax.plot(t_reward, medfilt(m, MEDFILT_KERNEL), color=color, label=label)
    ax.fill_between(t_reward, medfilt(m - se, MEDFILT_KERNEL),
                    medfilt(m + se, MEDFILT_KERNEL), color=color, alpha=0.2)
ax.axvline(0, color='k', lw=0.8, alpha=0.5)
ax.set_xlabel('Time from reward (s)')
ax.set_ylabel('ΔF/F')
ax.set_title('Reward aligned')
ax.legend(frameon=False)

plt.suptitle(f'{mouse} {session}: target vs other cells', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_target_vs_other_population.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%% ============================================================================
# CELL 7: Heatmap of all target cells (sorted by reward-aligned peak time)
# ============================================================================
sta_targ_mean = np.nanmean(sta[:, target_idx_list, :], axis=2).T   # (n_targ, time)
rta_targ_mean = np.nanmean(rta[:, target_idx_list, :], axis=2).T

# Single baseline per cell from the pre-trial-start window of the
# trial-start-aligned trace; subtract from both panels.
sta_base_mask = (t_trial >= TRIAL_WINDOW[0]) & (t_trial < TRIAL_WINDOW[0] + 1.0)
if sta_base_mask.any():
    bl = np.nanmean(sta_targ_mean[:, sta_base_mask], axis=1, keepdims=True)
    sta_targ_mean -= bl
    rta_targ_mean -= bl

# Clip the reward-aligned panel to [-4, +4] s for display.
_rta_mask = t_reward <= 4.0
t_reward_disp = t_reward[_rta_mask]
rta_targ_mean = rta_targ_mean[:, _rta_mask]

# Harvey-style sequence:
#   1. smooth each row (denoises argmax)
#   2. peak-normalize each row to its max abs value (one cell = one row of equal
#      intensity, so low-amplitude cells aren't crushed in the heatmap)
#   3. concatenate trial-start || reward; sort cells by the argmax of the
#      smoothed-normalized combined trace
def _row_smooth(arr, k=MEDFILT_KERNEL):
    return np.array([medfilt(r, k) for r in arr])

def _peak_norm(arr):
    scale = np.nanmax(np.abs(arr), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return arr / 1

# ---- Targets (row 0) ----
sta_targ_norm = _peak_norm(_row_smooth(sta_targ_mean))
rta_targ_norm = _peak_norm(_row_smooth(rta_targ_mean))
sort_targ = np.argsort(
    np.nanargmax(np.concatenate([sta_targ_norm, rta_targ_norm], axis=1), axis=1)
)

# ---- Clean non-targets (row 1) ----
# Exclude any cell that was ever within NONTARG_DIST of a stim site, so we
# don't include "targeted but not responsive" cells in the non-target group.
NONTARG_DIST = 30
min_dist_per_cell = np.nanmin(stimDist, axis=1)
nontarg_idx_list = np.where(min_dist_per_cell > NONTARG_DIST)[0]
n_nontarg = len(nontarg_idx_list)

sta_nt_mean = np.nanmean(sta[:, nontarg_idx_list, :], axis=2).T
rta_nt_mean = np.nanmean(rta[:, nontarg_idx_list, :], axis=2).T
if sta_base_mask.any():
    bl_nt = np.nanmean(sta_nt_mean[:, sta_base_mask], axis=1, keepdims=True)
    sta_nt_mean -= bl_nt
    rta_nt_mean -= bl_nt
rta_nt_mean = rta_nt_mean[:, _rta_mask]

sta_nt_norm = _peak_norm(_row_smooth(sta_nt_mean))
rta_nt_norm = _peak_norm(_row_smooth(rta_nt_mean))
sort_nt = np.argsort(
    np.nanargmax(np.concatenate([sta_nt_norm, rta_nt_norm], axis=1), axis=1)
)

vmax = .3

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Row 0: targets
ax = axes[0, 0]
im = ax.imshow(sta_targ_norm[sort_targ, :], aspect='auto', cmap='bwr',
               vmin=-vmax, vmax=vmax,
               extent=[t_trial[0], t_trial[-1], n_targ, 0])
ax.axvline(0, color='k', lw=0.8, alpha=0.7)
ax.set_xlabel('Time from trial start (s)')
ax.set_ylabel(f'Target cell (n={n_targ})\nsorted by combined peak time')
ax.set_title('Trial-start aligned')
plt.colorbar(im, ax=ax, label='ΔF/F')

ax = axes[0, 1]
im = ax.imshow(rta_targ_norm[sort_targ, :], aspect='auto', cmap='bwr',
               vmin=-vmax, vmax=vmax,
               extent=[t_reward_disp[0], t_reward_disp[-1], n_targ, 0])
ax.axvline(0, color='k', lw=0.8, alpha=0.7)
ax.set_xlabel('Time from reward (s)')
ax.set_title('Reward aligned')
plt.colorbar(im, ax=ax, label='ΔF/F')

# Row 1: clean non-targets (>NONTARG_DIST µm from every stim site)
ax = axes[1, 0]
im = ax.imshow(sta_nt_norm[sort_nt, :], aspect='auto', cmap='bwr',
               vmin=-vmax, vmax=vmax,
               extent=[t_trial[0], t_trial[-1], n_nontarg, 0])
ax.axvline(0, color='k', lw=0.8, alpha=0.7)
ax.set_xlabel('Time from trial start (s)')
ax.set_ylabel(f'Non-target cell (n={n_nontarg})\nsorted by combined peak time')
plt.colorbar(im, ax=ax, label='ΔF/F')

ax = axes[1, 1]
im = ax.imshow(rta_nt_norm[sort_nt, :], aspect='auto', cmap='bwr',
               vmin=-vmax, vmax=vmax,
               extent=[t_reward_disp[0], t_reward_disp[-1], n_nontarg, 0])
ax.axvline(0, color='k', lw=0.8, alpha=0.7)
ax.set_xlabel('Time from reward (s)')
plt.colorbar(im, ax=ax, label='ΔF/F')

plt.suptitle(f'{mouse} {session}: target vs non-target cell heatmaps', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_target_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%% ============================================================================
# CELL 8: Per-target reward modulation vs RT and RPE
# ============================================================================
rew_mask  = (t_reward >= -0.5) & (t_reward <= 0.5)
base_mask = (t_reward >= -4)   & (t_reward <= -3)

n_rewards = rta.shape[2]
hit_idx = np.where(hit)[0]
if len(hit_idx) != n_rewards:
    hit_idx = hit_idx[:n_rewards]
rt_hit = rt[hit_idx]

rpe_all = -compute_rpe(rt_filled, baseline=3.0, tau=10, fill_value=30.0)
rpe_hit = rpe_all[hit_idx]

per_cell_corr_rt  = np.full(n_targ, np.nan)
per_cell_corr_rpe = np.full(n_targ, np.nan)

for ki, ci in enumerate(target_idx_list):
    rew_act  = np.nanmean(rta[rew_mask,  ci, :], axis=0)
    base_act = np.nanmean(rta[base_mask, ci, :], axis=0)
    rm = rew_act - base_act
    valid = np.isfinite(rm) & np.isfinite(rt_hit) & np.isfinite(rpe_hit)
    if valid.sum() < 10:
        continue
    per_cell_corr_rt[ki],  _ = pearsonr(rm[valid], rt_hit[valid])
    per_cell_corr_rpe[ki], _ = pearsonr(rm[valid], rpe_hit[valid])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
vals = per_cell_corr_rt[np.isfinite(per_cell_corr_rt)]
ax.hist(vals, bins=15, color='steelblue', edgecolor='k', alpha=0.7)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axvline(np.nanmean(vals), color='red', lw=1.5,
           label=f'mean={np.nanmean(vals):.3f}')
ax.set_xlabel('Correlation (reward mod. vs RT)')
ax.set_ylabel('# target cells')
ax.set_title('Per-cell corr with RT')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
vals = per_cell_corr_rpe[np.isfinite(per_cell_corr_rpe)]
ax.hist(vals, bins=15, color='coral', edgecolor='k', alpha=0.7)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axvline(np.nanmean(vals), color='red', lw=1.5,
           label=f'mean={np.nanmean(vals):.3f}')
ax.set_xlabel('Correlation (reward mod. vs RPE)')
ax.set_ylabel('# target cells')
ax.set_title('Per-cell corr with RPE')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(f'{mouse} {session}: target cells, reward modulation vs behavior',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_target_reward_mod_vs_behavior.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%% ============================================================================
# CELL 9: Population fast vs slow trials (target cells, reward-aligned)
# ============================================================================
mean_rta_targ = np.nanmean(rta[:, is_target, :], axis=1)   # (time, n_rewards)

med_rt = np.nanmedian(rt_hit)
fast_inds = np.where(rt_hit < med_rt)[0]
slow_inds = np.where(rt_hit > med_rt)[0]

fig, ax = plt.subplots(figsize=(5, 4))
for inds, color, label in [
    (slow_inds, 'b', f'Slow (n={len(slow_inds)})'),
    (fast_inds, 'r', f'Fast (n={len(fast_inds)})'),
]:
    if len(inds) == 0:
        continue
    a = mean_rta_targ[:, inds]
    m  = np.nanmean(a, axis=1)
    se = np.nanstd(a, axis=1, ddof=1) / np.sqrt(a.shape[1])
    ax.plot(t_reward, m, color=color, label=label)
    ax.fill_between(t_reward, m - se, m + se, color=color, alpha=0.3,
                    edgecolor='none')

ax.axvline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Time from reward (s)')
ax.set_ylabel('Population avg ΔF/F (target cells)')
ax.set_title(f'{mouse} {session}: target population, fast vs slow trials')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_target_fast_slow_reward.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%% ============================================================================
# CELL 10: Split target cells by reward response, plot photostim amp vs stimDist
# for each subgroup
# ============================================================================
# Reward response per cell (vectorized across all n_cells)
rew_mask  = (t_reward >= -0.5) & (t_reward <= 0.5)
base_mask = (t_reward >= -4)   & (t_reward <= -3)

reward_resp_all = (np.nanmean(rta[rew_mask,  :, :], axis=(0, 2))
                   - np.nanmean(rta[base_mask, :, :], axis=(0, 2)))   # (n_cells,)

excited_target_cells   = np.where(is_target & (reward_resp_all > 0))[0]
inhibited_target_cells = np.where(is_target & (reward_resp_all < 0))[0]
excited_other_cells    = np.where(is_other  & (reward_resp_all > 0))[0]
inhibited_other_cells  = np.where(is_other  & (reward_resp_all < 0))[0]
print(f'Reward-excited targets:    {len(excited_target_cells)}')
print(f'Reward-inhibited targets:  {len(inhibited_target_cells)}')
print(f'Reward-excited non-targs:  {len(excited_other_cells)}')
print(f'Reward-inhibited non-targs:{len(inhibited_other_cells)}')

edges   = np.array([30, 50, 75, 100, 150, 200, 300, 500])
centers = 0.5 * (edges[:-1] + edges[1:])

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey='row')

panels = [
    (axes[0, 0], excited_target_cells,   'Reward-excited TARGETS',    '#d62728'),
    (axes[0, 1], inhibited_target_cells, 'Reward-inhibited TARGETS',  '#1f77b4'),
    (axes[1, 0], excited_other_cells,    'Reward-excited non-targets',   '#d62728'),
    (axes[1, 1], inhibited_other_cells,  'Reward-inhibited non-targets', '#1f77b4'),
]

# Pre-collect each panel's data so we can compute tight per-row y-limits
panel_data = []
for ax, cells, label, color in panels:
    if len(cells) == 0:
        panel_data.append((ax, label, color, np.array([]), np.array([])))
        continue
    a = amp_ep[cells, :].ravel()
    d = stimDist[cells, :].ravel()
    valid = np.isfinite(a) & np.isfinite(d) & (d > 30)
    panel_data.append((ax, label, color, a[valid], d[valid]))

def _row_ylim(*entries, pctile=90):
    pooled = np.concatenate([a for (_, _, _, a, _) in entries if a.size > 0]) \
             if any(a.size > 0 for (_, _, _, a, _) in entries) else np.array([])
    if pooled.size == 0:
        return 0.1
    return np.nanpercentile(np.abs(pooled), pctile)

top_ylim = _row_ylim(panel_data[0], panel_data[1], pctile=90)
bot_ylim = _row_ylim(panel_data[2], panel_data[3], pctile=90)

for ax, label, color, a, d in panel_data:
    if a.size == 0:
        ax.text(0.5, 0.5, 'no cells', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(f'{label} (n=0)', fontsize=11, color=color)
        continue

    ax.scatter(d, a, s=18, alpha=0.25, color=color, edgecolor='none')

    mean_b = np.full(len(centers), np.nan)
    sem_b  = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        sel = (d >= edges[k]) & (d < edges[k + 1])
        if sel.sum() > 0:
            mean_b[k] = np.mean(a[sel])
            sem_b[k]  = np.std(a[sel], ddof=1) / np.sqrt(sel.sum())
    ax.errorbar(centers, mean_b, yerr=sem_b, fmt='o-', color='black',
                linewidth=1.8, markersize=6, capsize=3, zorder=5)
    ax.axhline(0, color='gray', linewidth=0.5)
    n_cells_panel = sum(1 for c in [a] for _ in [c])  # length isn't right; use the cells list size
    ax.set_title(f'{label}', fontsize=11, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Apply tightened symmetric y-limits per row
axes[0, 0].set_ylim(-top_ylim, top_ylim)
axes[1, 0].set_ylim(-bot_ylim, bot_ylim)

# Re-set per-panel titles with cell counts (n) since we lost them above
for (ax, cells, label, color), (_, _, _, a, _) in zip(panels, panel_data):
    ax.set_title(f'{label} (n={len(cells)})', fontsize=11, color=color)

for ax in axes[1, :]:
    ax.set_xlabel('Distance from stim site (µm)')
axes[0, 0].set_ylabel('I-cell AMP (ΔF/F)')
axes[1, 0].set_ylabel('Non-target AMP (ΔF/F)')

plt.suptitle(
    f'{mouse} {session}: photostim amp vs distance (>30 µm), '
    f'split by reward response',
    fontsize=12,
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_target_amp_vs_dist_by_reward.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%% ============================================================================
# CELL 11: Network response when targeting reward-responsive vs non-responsive cells
# Split STIM GROUPS by their target cell's reward response, then plot amp vs
# distance for all OTHER cells (>30 µm) — i.e., the network response to
# stimming responsive vs non-responsive I cells.
# ============================================================================
# Each stim group's target cell's reward response
group_target_resp = reward_resp_all[target_cell_idx]   # (n_groups,)
group_target_isI  = is_target[target_cell_idx]         # only consider responsive-target groups

# Threshold for "reward responsive" target cell:
# top half by |reward response| among I-cell targets, bottom half = non-responsive.
abs_resp = np.abs(group_target_resp)
median_thresh = np.nanmedian(abs_resp[group_target_isI])

groups_resp_excited = np.where(group_target_isI &
                                (group_target_resp >  median_thresh))[0]
groups_resp_inhibited = np.where(group_target_isI &
                                  (group_target_resp < -median_thresh))[0]
groups_nonresp = np.where(group_target_isI &
                           (np.abs(group_target_resp) <= median_thresh))[0]
print(f'Stim groups w/ excited target:        {len(groups_resp_excited)}')
print(f'Stim groups w/ inhibited target:      {len(groups_resp_inhibited)}')
print(f'Stim groups w/ non-responsive target: {len(groups_nonresp)}')
print(f'  |resp| threshold (median): {median_thresh:.4f}')


def _amp_vs_dist_for_groups(group_inds):
    a = amp_ep[:, group_inds].ravel()
    d = stimDist[:, group_inds].ravel()
    valid = np.isfinite(a) & np.isfinite(d) & (d > 30)
    return a[valid], d[valid]


fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey='row')

splits = [
    (groups_resp_excited,   'Targeting reward-EXCITED I cells',         '#d62728'),
    (groups_resp_inhibited, 'Targeting reward-INHIBITED I cells',       '#1f77b4'),
    (groups_nonresp,        'Targeting reward-NON-RESPONSIVE I cells',  'gray'),
]

# Pre-collect amp/dist per split
collected = []
for gs, label, color in splits:
    a, d = _amp_vs_dist_for_groups(gs)
    collected.append((gs, label, color, a, d))

pooled         = np.concatenate([a for _, _, _, a, _ in collected if a.size > 0])
pooled_abs     = np.abs(pooled)
ylim_signed    = np.nanpercentile(np.abs(pooled),  90) if pooled.size else 0.1
ylim_abs       = np.nanpercentile(pooled_abs,     90) if pooled.size else 0.1

for col, (gs, label, color, a, d) in enumerate(collected):
    ax_signed = axes[0, col]
    ax_abs    = axes[1, col]

    if a.size == 0:
        for ax in (ax_signed, ax_abs):
            ax.text(0.5, 0.5, 'no groups', transform=ax.transAxes,
                    ha='center', va='center')
        ax_signed.set_title(f'{label}\n(n_groups=0)', fontsize=10, color=color)
        continue

    # ----- Top: signed amp -----
    ax_signed.scatter(d, a, s=14, alpha=0.2, color=color, edgecolor='none')
    mean_b = np.full(len(centers), np.nan)
    sem_b  = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        sel = (d >= edges[k]) & (d < edges[k + 1])
        if sel.sum() > 0:
            mean_b[k] = np.mean(a[sel])
            sem_b[k]  = np.std(a[sel], ddof=1) / np.sqrt(sel.sum())
    ax_signed.errorbar(centers, mean_b, yerr=sem_b, fmt='o-', color='black',
                       linewidth=1.8, markersize=6, capsize=3, zorder=5)
    ax_signed.axhline(0, color='gray', linewidth=0.5)
    ax_signed.set_title(f'{label}\n(n_groups={len(gs)})', fontsize=10, color=color)
    ax_signed.spines['top'].set_visible(False)
    ax_signed.spines['right'].set_visible(False)

    # ----- Bottom: |amp| -----
    a_abs = np.abs(a)
    ax_abs.scatter(d, a_abs, s=14, alpha=0.2, color=color, edgecolor='none')
    mean_ba = np.full(len(centers), np.nan)
    sem_ba  = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        sel = (d >= edges[k]) & (d < edges[k + 1])
        if sel.sum() > 0:
            mean_ba[k] = np.mean(a_abs[sel])
            sem_ba[k]  = np.std(a_abs[sel], ddof=1) / np.sqrt(sel.sum())
    ax_abs.errorbar(centers, mean_ba, yerr=sem_ba, fmt='o-', color='black',
                    linewidth=1.8, markersize=6, capsize=3, zorder=5)
    ax_abs.set_xlabel('Distance from stim site (µm)')
    ax_abs.spines['top'].set_visible(False)
    ax_abs.spines['right'].set_visible(False)

axes[0, 0].set_ylabel('Non-target AMP (ΔF/F)')
axes[1, 0].set_ylabel('Non-target |AMP| (ΔF/F)')
axes[0, 0].set_ylim(-ylim_signed, ylim_signed)
axes[1, 0].set_ylim(0,            ylim_abs)

plt.suptitle(
    f'{mouse} {session}: non-target network responses, '
    f'grouped by target cell reward response',
    fontsize=12,
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_nontarget_response_by_target_reward.png'),
            dpi=150, bbox_inches='tight')
plt.show()
