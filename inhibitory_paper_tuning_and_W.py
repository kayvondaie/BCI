"""
Step 1+2 toward replicating Figs 5h & 6b of the BCI paper for the inhibitory data.

Builds the per-cell building blocks used by the paper's MLR:
  • 4-epoch task tuning T_i  = (Pre_i, Early_i, Late_i, Rew_i)  — mean across trials.
  • Δtuning           ΔT_i  = (ΔPre_i, ΔEarly_i, ΔLate_i, ΔRew_i) — slope vs trial index.
  • Causal connectivity W_{i,g} = mean_a / std_a of per-repeat PS responses.
  • Δcausal connectivity ΔW_{i,g} = (μ_post − μ_pre) / sqrt(σ_post² + σ_pre²).

Single session (BCI116/012826). Saves a dict to disk so the MLR script can
just load `T`, `dT`, `T_g`, `dT_g`, `W`, `dW` without recomputing.

Sanity plots:
  • Distributions of all 8 per-cell tuning features.
  • Distribution of W (one PS epoch) and ΔW (across the two epochs).
  • Two Fig-5-style example scatters: W vs non-target Early tuning,
    W vs target-group Pre tuning.
"""

import os
import sys
import glob
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import data_dict_create_module_test as ddct
import bci_time_series as bts
from BCI_data_helpers import (
    parse_hdf5_array_string,
    get_reward_aligned_df_truncated,
    get_trial_aligned_df_padded,
)

sys.path.append(r'C:\Users\christina.wang\Downloads\BCI_code_local')
import data_dict_create_module_iscell as ddc


# -----------------------------
# Session + parameters
# -----------------------------
mouse   = 'BCI116'
session = '012826'
folder  = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
SAVE_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(SAVE_DIR, exist_ok=True)

REWARD_WINDOW   = (-4, 10)
TRIAL_WINDOW    = (-2, 4)

# Task epoch windows (paper definitions; Early uses a fixed 0–1 s window for
# simplicity — paper uses variable trial-start to reward−1 s)
EPOCHS = {
    'Pre':   ('sta', -2.0, -1.0),     # -2 to -1 s pre trial start
    'Early': ('sta',  0.0,  1.0),     # 0 to 1 s post trial start
    'Late':  ('rta', -1.0,  0.0),     # -1 to 0 s pre reward
    'Rew':   ('rta',  0.0,  3.0),     # 0 to 3 s post reward
}
EPOCH_NAMES = list(EPOCHS.keys())

# Photostim
PRE_FRAMES  = 8     # baseline frames pre stim (per repeat)
POST_AFTER  = 0.4   # post-window length, in s, after stim ends (similar to paper's 300 ms)
TARGET_AMP_THR = 0.05
DIST_TARGET_LT = 10


# -----------------------------
# Load BCI + photostim h5 (for stim_params)
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


photostim_keys = ['stimDist', 'favg_raw']
bci_keys = [
    'df_closedloop', 'F', 'mouse', 'session',
    'conditioned_neuron', 'dt_si', 'step_time', 'reward_time',
]
data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
dt_si = data['dt_si']
F = data['F']
trl = F.shape[2]


# -----------------------------
# Event vectors + reward-aligned and trial-start-aligned activity
# -----------------------------
data['step_time']   = _decode_event_times(data['step_time'],   trl)
data['reward_time'] = _decode_event_times(data['reward_time'], trl)
rt = np.array([x[0] if x.size > 0 else np.nan
               for x in data['reward_time']], dtype=float)
hit = np.isfinite(rt)
rt_filled = rt.copy()
rt_filled[~np.isfinite(rt_filled)] = 30.0

step_v, reward_v, trial_start_v = bts.bci_time_series_fun(
    folder, data, rt_filled, dt_si)
df = data['df_closedloop']
rta, t_reward = get_reward_aligned_df_truncated(
    df, reward_v, trial_start_v, dt_si, window=REWARD_WINDOW)
sta, t_trial = get_trial_aligned_df_padded(
    df, trial_start_v, reward_v, dt_si, window=TRIAL_WINDOW)
n_cells = sta.shape[1]


# -----------------------------
# Per-trial epoch activity matrices  (cell × trial; NaN for non-hit when needed)
# -----------------------------
def _epoch_per_trial(epoch_name):
    src, t_lo, t_hi = EPOCHS[epoch_name]
    if src == 'sta':
        mask = (t_trial >= t_lo) & (t_trial <= t_hi)
        per_trial = np.nanmean(sta[mask, :, :], axis=0)         # (n_cells, n_trials)
        # paper computes Pre / Early on hit trials only (cleaner alignment)
        per_trial[:, ~hit] = np.nan
    else:
        mask = (t_reward >= t_lo) & (t_reward <= t_hi)
        # rta has one column per reward; map back to absolute trial index
        rew_to_trial = np.where(hit)[0]
        n_rew = rta.shape[2]
        if len(rew_to_trial) != n_rew:
            n_use = min(len(rew_to_trial), n_rew)
            rew_to_trial = rew_to_trial[:n_use]
            n_rew = n_use
        per_rew = np.nanmean(rta[mask, :, :n_rew], axis=0)      # (n_cells, n_rewards)
        per_trial = np.full((n_cells, trl), np.nan)
        per_trial[:, rew_to_trial] = per_rew
    return per_trial


per_trial_epoch = {ep: _epoch_per_trial(ep) for ep in EPOCH_NAMES}


# -----------------------------
# T_i (mean across trials) and ΔT_i (slope vs trial index)
# -----------------------------
T  = np.zeros((n_cells, len(EPOCH_NAMES)))
dT = np.zeros((n_cells, len(EPOCH_NAMES)))

for ei, ep in enumerate(EPOCH_NAMES):
    M = per_trial_epoch[ep]                                     # (n_cells, n_trials)
    T[:, ei]  = np.nanmean(M, axis=1)
    trial_idx = np.arange(trl, dtype=float)
    for ci in range(n_cells):
        y = M[ci, :]
        v = np.isfinite(y)
        if v.sum() < 5 or np.std(trial_idx[v]) == 0:
            dT[ci, ei] = np.nan
            continue
        dT[ci, ei] = np.polyfit(trial_idx[v], y[v], 1)[0]

print(f'Tuning vectors: T={T.shape}, dT={dT.shape} '
      f'(finite frac per epoch: T={(np.isfinite(T).mean(0)).round(2)}, '
      f'dT={(np.isfinite(dT).mean(0)).round(2)})')


# -----------------------------
# Per-repeat photostim responses → W_{i,g} and ΔW_{i,g}
# Loads suite2p F.npy + ops + stat for each photostim epoch and runs
# ddc.stimDist_single_cell to get Fstim_raw (per-repeat).
# -----------------------------
def _per_repeat_W(ps_epoch):
    """Return mu_ig (n_cells, n_groups), sig_ig (n_cells, n_groups),
       stimDist (n_cells, n_groups, µm). Mean/std across repeats of the
       per-repeat (post − pre) / baseline."""
    suite2p = ('suite2p_photostim_single' if ps_epoch == 'photostim'
               else 'suite2p_photostim_single2')
    ps_dir = os.path.join(folder, suite2p, 'plane0')
    if not os.path.exists(ps_dir):
        return None
    iscell_i = np.load(os.path.join(ps_dir, 'iscell.npy'),   allow_pickle=True)
    stat_i   = np.load(os.path.join(ps_dir, 'stat.npy'),     allow_pickle=True)
    F_i      = np.load(os.path.join(ps_dir, 'F.npy'),        allow_pickle=True)
    ops_i    = np.load(os.path.join(ps_dir, 'ops.npy'),      allow_pickle=True).tolist()
    siHeader = np.load(os.path.join(ps_dir, 'siHeader.npy'), allow_pickle=True).tolist()

    out = ddc.stimDist_single_cell(ops_i, F_i, siHeader, stat_i, 0)
    Fstim_raw = out[9]                  # (n_frames, n_cells_full, n_trials)
    seq_arr   = np.asarray(out[1]).ravel() - 1
    stimDist_pix = out[3]
    umPerPix = (
        1000
        / float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])
        / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
    )
    dt_ps = 1.0 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
    stimDist_um = stimDist_pix * umPerPix

    # Stim window from h5
    sp = _load_stim_params_from_h5(folder, ps_epoch)
    h5_time = np.asarray(sp['time']).ravel()
    h5_total = float(np.asarray(sp['total_duration']).ravel()[0])
    start = int(np.where(np.isclose(h5_time, 0.0))[0][0])
    end_hits = np.where(h5_time >= h5_total)[0]
    end = int(end_hits[0]) if len(end_hits) > 0 else start + 16
    after_n = int(np.floor(POST_AFTER / dt_ps))
    pre_start = max(0, start - PRE_FRAMES)
    post_end  = min(Fstim_raw.shape[0], end + after_n)

    # Baseline F per (cell, group), used to normalize repeat responses
    n_full = Fstim_raw.shape[1]
    n_groups = stimDist_um.shape[1]
    bl_per_g = np.full((n_full, n_groups), np.nan)
    mu  = np.full((n_full, n_groups), np.nan)
    sig = np.full((n_full, n_groups), np.nan)
    for gi in range(n_groups):
        idx = np.where(seq_arr == gi)[0]
        if len(idx) < 3:
            continue
        pre_per_rep  = np.nanmean(Fstim_raw[pre_start:start, :, idx], axis=0)   # (cells, repeats)
        post_per_rep = np.nanmean(Fstim_raw[end:post_end, :, idx], axis=0)
        bl = np.nanmean(pre_per_rep, axis=1)                                    # (cells,)
        bl_safe = np.where(np.isfinite(bl) & (bl > 0), bl, np.nan)
        repeat_W = (post_per_rep - pre_per_rep) / bl_safe[:, None]              # (cells, repeats)
        bl_per_g[:, gi] = bl
        mu[:, gi]  = np.nanmean(repeat_W, axis=1)
        sig[:, gi] = np.nanstd(repeat_W, axis=1, ddof=1)

    return dict(mu=mu, sig=sig, stimDist=stimDist_um,
                iscell=iscell_i, n_cells_full=n_full)


W_pre_pkg  = _per_repeat_W('photostim')
W_post_pkg = _per_repeat_W('photostim2')

assert W_pre_pkg is not None and W_post_pkg is not None, \
    'need both photostim and photostim2 suite2p folders'

# Z-scored W per epoch
W_pre  = W_pre_pkg['mu']  / np.where(W_pre_pkg['sig']  > 0, W_pre_pkg['sig'],  np.nan)
W_post = W_post_pkg['mu'] / np.where(W_post_pkg['sig'] > 0, W_post_pkg['sig'], np.nan)
# ΔW with pooled-noise normalization (paper eq.)
sig_pool = np.sqrt(W_pre_pkg['sig'] ** 2 + W_post_pkg['sig'] ** 2)
dW = (W_post_pkg['mu'] - W_pre_pkg['mu']) / np.where(sig_pool > 0, sig_pool, np.nan)
stimDist = W_post_pkg['stimDist']        # photostim2 distances

# Sanity
print(f'W_pre  finite: {np.isfinite(W_pre).mean():.3f}')
print(f'W_post finite: {np.isfinite(W_post).mean():.3f}')
print(f'ΔW     finite: {np.isfinite(dW).mean():.3f}')

# Cell-index alignment: BCI F (and tuning vectors T/dT) come from the
# BCI suite2p; W comes from suite2p_photostim_single*. They have different
# n_cells. For now, only use cells that exist in both — use min length and
# warn the user if they differ. (A proper fix is suite2p ROI matching.)
n_W = W_pre.shape[0]
if n_W != n_cells:
    n_match = min(n_W, n_cells)
    print(f'WARNING: BCI suite2p has {n_cells} cells, photostim suite2p has '
          f'{n_W}. Using first {n_match} for now — proper ROI matching needed '
          f'before publication.')
    T  = T[:n_match]
    dT = dT[:n_match]
    W_pre  = W_pre[:n_match]
    W_post = W_post[:n_match]
    dW     = dW[:n_match]
    stimDist = stimDist[:n_match]


# -----------------------------
# Target-group tuning vectors T_g, dT_g
# T_g = mean of T_i over the responsive target cells of group g (using W_post
# > some threshold near the stim site as the responsiveness criterion).
# -----------------------------
n_groups = stimDist.shape[1]
T_g  = np.full((n_groups, len(EPOCH_NAMES)), np.nan)
dT_g = np.full((n_groups, len(EPOCH_NAMES)), np.nan)
for gi in range(n_groups):
    cl = np.where(
        (stimDist[:, gi] < DIST_TARGET_LT) &
        np.isfinite(W_pre[:, gi])  & (W_pre[:, gi]  > 0) &
        np.isfinite(W_post[:, gi]) & (W_post[:, gi] > 0)
    )[0]
    if cl.size == 0:
        continue
    T_g[gi]  = np.nanmean(T[cl],  axis=0)
    dT_g[gi] = np.nanmean(dT[cl], axis=0)


# -----------------------------
# Save building blocks for the next-step MLR script
# -----------------------------
out_path = os.path.join(SAVE_DIR, f'paper_tuning_W_{mouse}_{session}.npz')
np.savez(out_path,
         T=T, dT=dT, T_g=T_g, dT_g=dT_g,
         W_pre=W_pre, W_post=W_post, dW=dW, stimDist=stimDist,
         epoch_names=np.array(EPOCH_NAMES))
print(f'Saved building blocks → {out_path}')


# -----------------------------
# Sanity plots
# -----------------------------
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ei, ep in enumerate(EPOCH_NAMES):
    ax = axes[0, ei]
    ax.hist(T[:, ei][np.isfinite(T[:, ei])], bins=30, color='#0096FF', alpha=0.7)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title(f'T_i: {ep}', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax = axes[1, ei]
    ax.hist(dT[:, ei][np.isfinite(dT[:, ei])], bins=30, color='#d62728', alpha=0.7)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title(f'ΔT_i: {ep}', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle(f'{mouse} {session}: per-cell task tuning building blocks',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'paper_step1_tuning_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, M, ttl in [
    (axes[0], W_pre,  'W (photostim epoch)'),
    (axes[1], W_post, 'W (photostim2 epoch)'),
    (axes[2], dW,     'ΔW = (μ_post - μ_pre) / sqrt(σ_post² + σ_pre²)'),
]:
    v = M[np.isfinite(M)]
    lim = np.nanpercentile(np.abs(v), 99) if v.size else 1.0
    bins = np.linspace(-lim, lim, 51)
    ax.hist(v, bins=bins, color='gray', alpha=0.7, density=True)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(f'{ttl}\n(n={v.size}, median={np.nanmedian(v):+.3f})',
                 fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle(f'{mouse} {session}: causal connectivity (Z-scored)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'paper_step2_W_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.show()


# Fig-5-style example scatters: every (non-target i, group g) pair is its own point.
# Same non-target appears in many points (once per group); same group appears in many
# points (once per non-target). No averaging across the i or g axis.
nt_dist_mask = stimDist > 30
ii_idx, gg_idx = np.where(nt_dist_mask & np.isfinite(W_pre))   # all non-target pairs
W_pairs   = W_pre[ii_idx, gg_idx]
Early_i   = T[ii_idx, EPOCH_NAMES.index('Early')]
Pre_g     = T_g[gg_idx, EPOCH_NAMES.index('Pre')]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) W_{i,g} vs Early tuning of non-target i — one dot per (i, g) pair
ax = axes[0]
ok = np.isfinite(Early_i) & np.isfinite(W_pairs)
ax.scatter(Early_i[ok], W_pairs[ok], s=4, alpha=0.2, color='black',
           edgecolor='none')
# binned mean ± SEM (no averaging — just bin x and average y within each bin)
if ok.sum() > 20:
    edges = np.percentile(Early_i[ok], np.linspace(0, 100, 8))
    bx, by, be = [], [], []
    for k in range(len(edges) - 1):
        m = ok & (Early_i >= edges[k]) & (Early_i <= edges[k + 1])
        if m.sum() < 5:
            continue
        bx.append(np.mean(Early_i[m]))
        by.append(np.mean(W_pairs[m]))
        be.append(np.std(W_pairs[m], ddof=1) / np.sqrt(m.sum()))
    ax.errorbar(bx, by, yerr=be, fmt='o-', color='#d62728', capsize=3,
                linewidth=1.5, markersize=5, zorder=5)
if ok.sum() >= 3:
    r, p = pearsonr(Early_i[ok], W_pairs[ok])
    ax.set_title(f'W_{{i,g}} vs Early_i (n={ok.sum()} pairs)\n'
                 f'r={r:+.3f}, p={p:.2e}', fontsize=11)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Early_i (non-target)')
ax.set_ylabel('W_{i,g}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) W_{i,g} vs Pre tuning of target group g — one dot per (i, g) pair
ax = axes[1]
ok = np.isfinite(Pre_g) & np.isfinite(W_pairs)
ax.scatter(Pre_g[ok], W_pairs[ok], s=4, alpha=0.2, color='black',
           edgecolor='none')
if ok.sum() > 20:
    edges = np.percentile(Pre_g[ok], np.linspace(0, 100, 8))
    bx, by, be = [], [], []
    for k in range(len(edges) - 1):
        m = ok & (Pre_g >= edges[k]) & (Pre_g <= edges[k + 1])
        if m.sum() < 5:
            continue
        bx.append(np.mean(Pre_g[m]))
        by.append(np.mean(W_pairs[m]))
        be.append(np.std(W_pairs[m], ddof=1) / np.sqrt(m.sum()))
    ax.errorbar(bx, by, yerr=be, fmt='o-', color='#d62728', capsize=3,
                linewidth=1.5, markersize=5, zorder=5)
if ok.sum() >= 3:
    r, p = pearsonr(Pre_g[ok], W_pairs[ok])
    ax.set_title(f'W_{{i,g}} vs Pre_g (n={ok.sum()} pairs)\n'
                 f'r={r:+.3f}, p={p:.2e}', fontsize=11)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Pre_g (target group)')
ax.set_ylabel('W_{i,g}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(f'{mouse} {session}: Fig-5-style sanity scatters', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'paper_step2_W_examples.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print('\nStep 1+2 complete.')
print(f'  T, dT, T_g, dT_g, W_pre, W_post, dW, stimDist saved to: {out_path}')
print('  Next step: load this .npz and run the MLR (Fig 5h / 6b).')
