"""
Per-pair dW vs Δlate of the post-synaptic (non-target) neuron, single
session (BCI116/012826).

For each non-target cell:
  Δlate = slope of (late-trial activity) regressed against absolute trial
          index, where late-trial activity = mean dF/F in [-1, 0] s before
          reward, taken across all rewarded trials.

For each (responsive target, non-target ≥30 µm) pair:
  x = Δlate of the post-synaptic neuron
  y = dW = AMP[photostim2] − AMP[photostim] for that pair
      (optionally with per-group dTarget regressed out — same correction
      as the sliding-window analysis)

Many points share an x (one post-synaptic neuron has many incoming targets),
so we plot only the binned mean ± SEM and report Pearson r and a Wilcoxon
test on the per-cell mean dW.
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
)


# -----------------------------
# Session + parameters
# -----------------------------
mouse   = 'BCI116'
session = '010826'
folder  = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
SAVE_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(SAVE_DIR, exist_ok=True)

REWARD_WINDOW     = (-4, 10)
TARGET_AMP_THR    = 0.05
DIST_TARGET_LT    = 10
DIST_NONTARG_LO   = 30
DIST_NONTARG_HI   = 1000
LATE_WIN_S        = (-1.0, 0.0)   # 1 s pre-reward
N_BINS_PLOT       = 5
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
# AMP per (cell, group) for both photostim epochs (in dF/F)
# -----------------------------
AMP, stimDist = compute_amp_from_photostim_artifact_free(mouse, data, folder)
assert len(AMP) >= 2, 'need both photostim and photostim2 to compute dW'


# -----------------------------
# Reward-aligned activity → late-trial activity per (cell, reward)
# -----------------------------
data['step_time']   = _decode_event_times(data['step_time'],   trl)
data['reward_time'] = _decode_event_times(data['reward_time'], trl)
rt = np.array([x[0] if x.size > 0 else np.nan
               for x in data['reward_time']], dtype=float)
rt_filled = rt.copy()
rt_filled[~np.isfinite(rt_filled)] = 30.0
hit = np.isfinite(rt)

step_v, reward_v, trial_start_v = bts.bci_time_series_fun(
    folder, data, rt_filled, dt_si)
df = data['df_closedloop']
rta, t_reward = get_reward_aligned_df_truncated(
    df, reward_v, trial_start_v, dt_si, window=REWARD_WINDOW)

late_mask = (t_reward >= LATE_WIN_S[0]) & (t_reward <= LATE_WIN_S[1])
late_act = np.nanmean(rta[late_mask, :, :], axis=0)         # (n_cells, n_rewards)


# Map reward index → absolute trial index in the session.
# rta and the BCI rt array can differ by 1 (extra/missing reward at the end);
# truncate both to the shorter so the regression has aligned arrays.
hit_idx = np.where(hit)[0]
n_use = min(len(hit_idx), late_act.shape[1])
if n_use < max(len(hit_idx), late_act.shape[1]):
    print(f'rta has {late_act.shape[1]} rewards but hit array has {len(hit_idx)}; '
          f'using first {n_use}')
hit_idx = hit_idx[:n_use]
late_act = late_act[:, :n_use]
trial_idx = hit_idx.astype(float)


# -----------------------------
# Δlate per cell: slope of late_act vs trial index across rewarded trials
# -----------------------------
n_cells = late_act.shape[0]
delta_late = np.full(n_cells, np.nan)
for ci in range(n_cells):
    y = late_act[ci, :]
    valid = np.isfinite(y) & np.isfinite(trial_idx)
    if valid.sum() < 5 or np.std(trial_idx[valid]) == 0:
        continue
    delta_late[ci] = np.polyfit(trial_idx[valid], y[valid], 1)[0]

print(f'Δlate computed for {int(np.isfinite(delta_late).sum())} / {n_cells} cells')


# -----------------------------
# Build (target, non-target) pairs and gather x, y, dTarget
# -----------------------------
xs = []          # Δlate of the post-synaptic non-target
ys = []          # dW per pair
dts = []         # per-pair dTarget (target's own dW), broadcast per group
nt_id = []       # cell index of the non-target (post-synaptic)
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
    dw = AMP[1][nontarg, gi] - AMP[0][nontarg, gi]
    dt = float(np.mean(AMP[1][cl, gi] - AMP[0][cl, gi]))
    dl = delta_late[nontarg]
    valid = np.isfinite(dw) & np.isfinite(dl)
    xs.extend(dl[valid].tolist())
    ys.extend(dw[valid].tolist())
    dts.extend([dt] * int(valid.sum()))
    nt_id.extend(nontarg[valid].tolist())

xs = np.array(xs)
ys = np.array(ys)
dts = np.array(dts)
nt_id = np.array(nt_id)
print(f'{mouse} {session}: {len(ys)} (target, non-target) pairs across '
      f'{len(np.unique(nt_id))} unique non-targets')


# -----------------------------
# dTarget correction (regress out)
# -----------------------------
if CONTROL_DTARGET and dts.size and np.std(dts) > 0:
    dtc = dts - np.mean(dts)
    beta_dt = float(np.dot(dtc, ys) / np.dot(dtc, dtc))
    ys = ys - beta_dt * dtc
    print(f'CONTROL_DTARGET on: regressed out dTarget (β={beta_dt:+.3f})')


# -----------------------------
# Stats
# -----------------------------
if xs.size:
    r, p = pearsonr(xs, ys)
else:
    r, p = (np.nan, np.nan)

# Per-non-target Wilcoxon: mean dW per non-target (collapsing across its targets)
mean_dw_per_nt = []
mean_dl_per_nt = []
for ci in np.unique(nt_id):
    sel = nt_id == ci
    mean_dw_per_nt.append(np.mean(ys[sel]))
    mean_dl_per_nt.append(np.mean(xs[sel]))
mean_dw_per_nt = np.array(mean_dw_per_nt)
mean_dl_per_nt = np.array(mean_dl_per_nt)
try:
    _, p_w = wilcoxon(mean_dw_per_nt)
except Exception:
    p_w = np.nan
print(f'Pooled Pearson r = {r:+.3f}, p = {p:.2e}')
print(f'Per-non-target Wilcoxon (mean dW != 0): n={len(mean_dw_per_nt)}, p={p_w:.3f}')


# -----------------------------
# Plot — binned mean ± SEM
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 5))

bx, by, be = [], [], []
if xs.size > 10:
    edges = np.percentile(xs, np.linspace(0, 100, N_BINS_PLOT + 1))
    for k in range(N_BINS_PLOT):
        if k < N_BINS_PLOT - 1:
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
ax.set_xlabel('Δlate (slope of late-trial activity vs trial)', fontsize=11)
_ylab = ('dW − β·dTarget for non-targets (ΔF/F)' if CONTROL_DTARGET
         else 'dW = AMP[post] − AMP[pre] for non-targets (ΔF/F)')
ax.set_ylabel(_ylab, fontsize=11)
ax.set_title(
    f'{mouse} {session}: dW vs Δlate (post-synaptic non-targets)\n'
    f'pooled r={r:+.3f}, p={p:.2e}   |   per-non-target Wilcoxon p={p_w:.3f}',
    fontsize=11,
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'inhibitory_dw_vs_delta_late.png'),
            dpi=150, bbox_inches='tight')
plt.show()
