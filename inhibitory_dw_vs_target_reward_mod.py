"""
Per-pair dW vs target-cell reward modulation, single session (BCI116/012826).

For each (responsive target, non-target ≥30 µm) pair:
  x  = target cell's reward modulation during the BCI task
       = mean rta in [-0.5, 0.5] s minus mean rta in [-4, -3] s
  y  = dW = AMP[photostim2] − AMP[photostim] for that non-target / target group

Many points share an x (one target → many non-targets), so we also overlay a
binned mean ± SEM and report Pearson r and a per-target Wilcoxon test on dW.
"""

import os
import glob
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon

import data_dict_create_module_test as ddct
import bci_time_series as bts
from BCI_data_helpers import (
    compute_amp_from_photostim_artifact_free,
    parse_hdf5_array_string,
    get_reward_aligned_df_truncated,
    get_trial_aligned_df_padded,
)


# -----------------------------
# Session + parameters
# -----------------------------
mouse   = 'BCI116'
session = '012826'
folder  = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
SAVE_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(SAVE_DIR, exist_ok=True)

REWARD_WINDOW     = (-4, 10)
TRIAL_WINDOW      = (-2, 4)
TARGET_AMP_THR    = 0.05
DIST_TARGET_LT    = 10
DIST_NONTARG_LO   = 30
DIST_NONTARG_HI   = 1000
REW_WIN_S         = (-0.5,  0.5)
BASE_WIN_S        = (-4.0, -3.0)

# Regress out per-group target dTarget = mean(AMP[1][cl] - AMP[0][cl]) from dW
# before binning, so changes in target driving don't masquerade as plasticity.
CONTROL_DTARGET   = True


# -----------------------------
# Helpers (same as the heatmap-minimal script)
# -----------------------------
def _load_stim_params_from_h5(folder, ps_epoch):
    all_h5 = sorted(glob.glob(os.path.join(folder, '*.h5')))
    if ps_epoch == 'photostim':
        cand = [p for p in all_h5
                if 'photostim' in os.path.basename(p).lower()
                and 'photostim2' not in os.path.basename(p).lower()]
    else:
        cand = [p for p in all_h5
                if 'photostim2' in os.path.basename(p).lower()]
    if not cand:
        return None
    sp = {}
    with h5py.File(cand[0], 'r') as f:
        if 'stim_params' in f:
            for k in f['stim_params'].keys():
                try:
                    sp[k] = f['stim_params'][k][()]
                except Exception:
                    pass
    return sp


def _decode_event_times(v, trl):
    raw = v.item() if hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0 else v
    if isinstance(raw, (bytes, np.bytes_)):
        try:
            obj = pickle.loads(raw)
            return [np.asarray(x).ravel() for x in obj]
        except Exception:
            pass
    return parse_hdf5_array_string(v, trl)


# -----------------------------
# Load BCI + photostim
# -----------------------------
photostim_keys = ['stimDist', 'favg_raw']
bci_keys = [
    'df_closedloop', 'F', 'mouse', 'session',
    'conditioned_neuron', 'dt_si', 'step_time', 'reward_time',
]
data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
for ep in ('photostim', 'photostim2'):
    if ep in data:
        sp = _load_stim_params_from_h5(folder, ep)
        if sp is not None:
            data[ep]['stim_params'] = sp

dt_si = data['dt_si']
F = data['F']
trl = F.shape[2]


# -----------------------------
# Per-(cell, group) AMP for both photostim epochs (in dF/F)
# -----------------------------
AMP, stimDist = compute_amp_from_photostim_artifact_free(mouse, data, folder)
assert len(AMP) >= 2, 'need both photostim and photostim2 to compute dW'


# -----------------------------
# Reward-aligned activity → per-cell reward modulation
# -----------------------------
data['step_time']   = _decode_event_times(data['step_time'],   trl)
data['reward_time'] = _decode_event_times(data['reward_time'], trl)
rt = np.array([x[0] if x.size > 0 else np.nan
               for x in data['reward_time']], dtype=float)
rt_filled = rt.copy()
rt_filled[~np.isfinite(rt_filled)] = 30.0

step_v, reward_v, trial_start_v = bts.bci_time_series_fun(
    folder, data, rt_filled, dt_si)
df = data['df_closedloop']
rta, t_reward = get_reward_aligned_df_truncated(
    df, reward_v, trial_start_v, dt_si, window=REWARD_WINDOW)

rew_mask  = (t_reward >= REW_WIN_S[0])  & (t_reward <= REW_WIN_S[1])
base_mask = (t_reward >= BASE_WIN_S[0]) & (t_reward <= BASE_WIN_S[1])
# reward modulation per cell: mean(reward window) − mean(baseline window),
# averaged over rewards
reward_mod = (np.nanmean(rta[rew_mask,  :, :], axis=(0, 2))
              - np.nanmean(rta[base_mask, :, :], axis=(0, 2)))


# -----------------------------
# Build (target, non-target) pairs and gather x, y
# -----------------------------
xs = []          # target reward modulation, repeated per non-target
ys = []          # dW per non-target
dts = []         # per-pair dTarget (target's own dW), broadcast per group
group_id = []    # for the per-target Wilcoxon
target_id = []   # cell index of the target
n_groups = stimDist.shape[1]

for gi in range(n_groups):
    cl = np.where(
        (stimDist[:, gi] < DIST_TARGET_LT) &
        (AMP[0][:, gi] > TARGET_AMP_THR) &
        (AMP[1][:, gi] > TARGET_AMP_THR)
    )[0]
    if cl.size == 0:
        continue
    nontarg = np.where(
        (stimDist[:, gi] > DIST_NONTARG_LO) &
        (stimDist[:, gi] < DIST_NONTARG_HI)
    )[0]
    if nontarg.size == 0:
        continue
    rm = np.nanmean(reward_mod[cl])
    if not np.isfinite(rm):
        continue
    dw = AMP[1][nontarg, gi] - AMP[0][nontarg, gi]
    dt = float(np.mean(AMP[1][cl, gi] - AMP[0][cl, gi]))
    valid = np.isfinite(dw)
    xs.extend([rm] * int(valid.sum()))
    ys.extend(dw[valid].tolist())
    dts.extend([dt] * int(valid.sum()))
    group_id.extend([gi] * int(valid.sum()))
    target_id.extend([int(cl[0])] * int(valid.sum()))

xs = np.array(xs)
ys = np.array(ys)
dts = np.array(dts)
group_id = np.array(group_id)
target_id = np.array(target_id)
print(f'{mouse} {session}: {len(np.unique(group_id))} target groups, '
      f'{len(ys)} (target, non-target) pairs')

if CONTROL_DTARGET and dts.size and np.std(dts) > 0:
    dtc = dts - np.mean(dts)
    beta_dt = float(np.dot(dtc, ys) / np.dot(dtc, dtc))
    ys = ys - beta_dt * dtc
    print(f'CONTROL_DTARGET on: regressed out dTarget (β={beta_dt:+.3f})')

if xs.size:
    r, p = pearsonr(xs, ys)
else:
    r, p = (np.nan, np.nan)
print(f'Pooled Pearson r = {r:+.3f}, p = {p:.2e}')


# -----------------------------
# Per-target dW: Wilcoxon test that mean dW != 0 across target cells
# -----------------------------
mean_dw_per_target = []
mean_x_per_target  = []
for gi in np.unique(group_id):
    sel = group_id == gi
    mean_dw_per_target.append(np.mean(ys[sel]))
    mean_x_per_target.append(np.mean(xs[sel]))
mean_dw_per_target = np.array(mean_dw_per_target)
mean_x_per_target  = np.array(mean_x_per_target)
try:
    _, p_w = wilcoxon(mean_dw_per_target)
except Exception:
    p_w = np.nan
print(f'Per-target Wilcoxon (mean dW != 0): n={len(mean_dw_per_target)}, p={p_w:.3f}')


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 5))

bx, by, be = [], [], []
if xs.size > 10:
    n_bins = 5
    edges = np.percentile(xs, np.linspace(0, 100, n_bins + 1))
    for k in range(n_bins):
        if k < n_bins - 1:
            sel = (xs >= edges[k]) & (xs < edges[k + 1])
        else:
            sel = (xs >= edges[k]) & (xs <= edges[k + 1])
        if sel.sum() < 3:
            continue
        bx.append(np.mean(xs[sel]))
        by.append(np.mean(ys[sel]))
        be.append(np.std(ys[sel], ddof=1) / np.sqrt(sel.sum()))
ax.errorbar(bx, by, yerr=be, fmt='o-', color='black', capsize=4,
            linewidth=1.8, markersize=6)

ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Target reward modulation (ΔF/F)', fontsize=11)
_y_lbl = ('dW − β·dTarget for non-targets (ΔF/F)' if CONTROL_DTARGET
          else 'dW = AMP[post] − AMP[pre] for non-targets (ΔF/F)')
ax.set_ylabel(_y_lbl, fontsize=11)
ax.set_title(
    f'{mouse} {session}: dW vs target reward modulation\n'
    f'pooled r={r:+.3f}, p={p:.2e}   |   per-target Wilcoxon p={p_w:.3f}',
    fontsize=11,
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'inhibitory_dw_vs_target_reward_mod.png'),
            dpi=150, bbox_inches='tight')
plt.show()
