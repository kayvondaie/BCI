# -*- coding: utf-8 -*-
"""
Hub neuron identification from single-cell 2p photostimulation.

Builds a functional connectivity matrix from photostim-evoked responses,
compares degree distributions to an Erdos-Renyi null, and flags hub neurons
whose in-degree or out-degree exceeds the ER expectation.

@author: kayvon.daie
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import data_dict_create_module_test as ddc

#%% ---- Parameters ----
mouse = "BCI120"
session = '010526'
folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'

target_dist_thresh = 10    # um — max distance to call a neuron the "target"
nontarget_dist_min = 30    # um — min distance for non-target neurons
sig_alpha = 0.1            # significance threshold for connection detection (t-test)
hub_alpha = 0.005          # significance threshold for hub classification (vs ER null)

# Output directory
save_dir = os.path.join(_THIS_DIR, 'results', f'{mouse}_{session}')
os.makedirs(save_dir, exist_ok=True)

#%% ---- Load data & compute photostim amplitudes ----
data = ddc.load_data_dict(folder)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

# Single-trial dF/F responses and stim identity
Fstim = data['photostim']['Fstim']        # (time, neurons, trials) — already dF/F
seq = np.array(data['photostim']['seq'])   # (trials,) — stim group per trial (1-indexed)

# Distance in microns
stimDist = data['photostim']['stimDist'] * umPerPix  # (neurons, stims)

N_neurons = Fstim.shape[1]
N_stims = stimDist.shape[1]

# Per-trial amplitude: mean(frames 15:20) - mean(frames 0:6)
amp_trials = np.nanmean(Fstim[15:20, :, :], axis=0) - np.nanmean(Fstim[0:6, :, :], axis=0)  # (neurons, trials)

#%% ---- Identify target neuron for each stim ----
# Target = closest neuron to stim site, must be within threshold
target_idx = np.argmin(stimDist, axis=0)  # (stims,)
target_dist = stimDist[target_idx, np.arange(N_stims)]

# Only keep stim groups where we have a clear single-cell target
valid_stim = target_dist < target_dist_thresh
print(f"Valid stim groups with identified target: {valid_stim.sum()} / {N_stims}")

#%% ---- Build connectivity matrix via t-test on single trials ----
# For each stim group, collect single-trial amplitudes for each non-target neuron.
# t-test: are the trial responses significantly different from zero?

unique_targets = np.unique(target_idx[valid_stim])
N_targets = len(unique_targets)

conn_amp = np.full((N_neurons, N_neurons), np.nan)
conn_pval = np.full((N_neurons, N_neurons), np.nan)

for gi in range(N_stims):
    if not valid_stim[gi]:
        continue
    t_neuron = target_idx[gi]
    trial_mask = seq == (gi + 1)  # seq is 1-indexed
    if trial_mask.sum() < 3:
        continue

    for j in range(N_neurons):
        if j == t_neuron:
            continue
        if stimDist[j, gi] < nontarget_dist_min:
            continue

        trial_amps = amp_trials[j, trial_mask]
        trial_amps = trial_amps[np.isfinite(trial_amps)]
        if len(trial_amps) < 3:
            continue

        t_stat, pval = stats.ttest_1samp(trial_amps, 0)
        mean_amp = np.nanmean(trial_amps)

        # If multiple stim groups target the same neuron, keep the most significant
        if np.isnan(conn_pval[t_neuron, j]) or pval < conn_pval[t_neuron, j]:
            conn_amp[t_neuron, j] = mean_amp
            conn_pval[t_neuron, j] = pval

# Binary connectivity: significant connections
conn_binary = np.zeros((N_neurons, N_neurons), dtype=int)
sig_mask = conn_pval < sig_alpha
conn_binary[sig_mask] = 1
# Only count connections between stimulated targets and valid responders
conn_binary[np.isnan(conn_pval)] = 0

print(f"Unique stimulated neurons: {N_targets}")
print(f"Total significant connections: {conn_binary.sum()}")

#%% ---- Degree distributions ----
# Out-degree: number of neurons that respond when neuron i is stimulated
# (only defined for neurons that were actually stimulated)
out_degree = np.full(N_neurons, np.nan)
in_degree = np.full(N_neurons, np.nan)

# Out-degree: for each stimulated neuron, count significant responders
for t_neuron in unique_targets:
    # Number of testable pairs for this source
    testable = ~np.isnan(conn_pval[t_neuron, :])
    out_degree[t_neuron] = conn_binary[t_neuron, testable].sum()

# In-degree: for each neuron, count how many stim sources drive it
for j in range(N_neurons):
    testable = ~np.isnan(conn_pval[:, j])
    if testable.sum() > 0:
        in_degree[j] = conn_binary[testable, j].sum()

#%% ---- Erdos-Renyi null model ----
# Connection probability = total edges / total testable pairs
testable_pairs = np.sum(~np.isnan(conn_pval))
total_edges = conn_binary.sum()
p_conn = total_edges / testable_pairs if testable_pairs > 0 else 0
print(f"Connection probability p = {p_conn:.4f} ({total_edges}/{testable_pairs})")

# For each neuron, compare its degree to Binomial(n_testable, p_conn)
out_hub = np.zeros(N_neurons, dtype=bool)
in_hub = np.zeros(N_neurons, dtype=bool)
out_pval = np.full(N_neurons, np.nan)
in_pval = np.full(N_neurons, np.nan)

for t_neuron in unique_targets:
    n_testable = np.sum(~np.isnan(conn_pval[t_neuron, :]))
    if n_testable == 0:
        continue
    # P(X >= observed) under Binomial(n_testable, p_conn)
    k = int(out_degree[t_neuron])
    pval = stats.binom.sf(k - 1, n_testable, p_conn)  # sf = 1 - cdf(k-1)
    out_pval[t_neuron] = pval
    out_hub[t_neuron] = pval < hub_alpha

for j in range(N_neurons):
    n_testable = np.sum(~np.isnan(conn_pval[:, j]))
    if n_testable == 0:
        continue
    k = int(in_degree[j])
    pval = stats.binom.sf(k - 1, n_testable, p_conn)
    in_pval[j] = pval
    in_hub[j] = pval < hub_alpha

hub_neurons = out_hub | in_hub
print(f"\nOut-hubs: {out_hub.sum()}  |  In-hubs: {in_hub.sum()}  |  Total hubs: {hub_neurons.sum()}")

#%% ---- Plot: Degree distributions vs ER null ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Out-degree
ax = axes[0]
valid_out = out_degree[~np.isnan(out_degree)].astype(int)
if len(valid_out) > 0:
    max_k = max(valid_out.max(), 1)
    bins = np.arange(-0.5, max_k + 1.5, 1)
    ax.hist(valid_out, bins=bins, density=True, alpha=0.6, color='steelblue', label='Observed')

    # ER expected (use median n_testable for the theoretical curve)
    n_test_out = [np.sum(~np.isnan(conn_pval[t, :])) for t in unique_targets]
    n_median = int(np.median(n_test_out))
    k_range = np.arange(0, max_k + 2)
    er_pmf = stats.binom.pmf(k_range, n_median, p_conn)
    ax.plot(k_range, er_pmf, 'r-o', ms=4, label=f'ER (n={n_median}, p={p_conn:.3f})')

    # Mark hubs
    hub_out_vals = out_degree[out_hub].astype(int)
    if len(hub_out_vals) > 0:
        for v in np.unique(hub_out_vals):
            ax.axvline(v, color='red', ls='--', alpha=0.5)

ax.set_xlabel('Out-degree')
ax.set_ylabel('Density')
ax.set_title('Out-degree distribution')
ax.legend()

# In-degree
ax = axes[1]
valid_in = in_degree[~np.isnan(in_degree)].astype(int)
if len(valid_in) > 0:
    max_k = max(valid_in.max(), 1)
    bins = np.arange(-0.5, max_k + 1.5, 1)
    ax.hist(valid_in, bins=bins, density=True, alpha=0.6, color='coral', label='Observed')

    n_test_in = [np.sum(~np.isnan(conn_pval[:, j])) for j in range(N_neurons) if not np.isnan(in_degree[j])]
    n_median = int(np.median(n_test_in)) if len(n_test_in) > 0 else 1
    k_range = np.arange(0, max_k + 2)
    er_pmf = stats.binom.pmf(k_range, n_median, p_conn)
    ax.plot(k_range, er_pmf, 'r-o', ms=4, label=f'ER (n={n_median}, p={p_conn:.3f})')

    hub_in_vals = in_degree[in_hub].astype(int)
    if len(hub_in_vals) > 0:
        for v in np.unique(hub_in_vals):
            ax.axvline(v, color='red', ls='--', alpha=0.5)

ax.set_xlabel('In-degree')
ax.set_ylabel('Density')
ax.set_title('In-degree distribution')
ax.legend()

plt.suptitle(f'{mouse} {session} — Hub neuron analysis', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(save_dir, 'degree_distributions.png'), dpi=200, bbox_inches='tight')

#%% ---- Plot: Connectivity matrix (sorted by degree) ----
fig, ax = plt.subplots(figsize=(6, 5))

# Build a submatrix: rows = stimulated neurons, cols = all neurons
# Sort rows by out-degree, cols by in-degree
row_order = unique_targets[np.argsort(-out_degree[unique_targets])]
col_valid = np.where(~np.isnan(in_degree))[0]
col_order = col_valid[np.argsort(-in_degree[col_valid])]

display_mat = conn_binary[np.ix_(row_order, col_order)].astype(float)
display_mat[np.isnan(conn_pval[np.ix_(row_order, col_order)])] = np.nan

cmap = plt.cm.Blues.copy()
cmap.set_bad('lightgray')
ax.imshow(display_mat, aspect='auto', cmap=cmap, interpolation='none', vmin=0, vmax=1)
ax.set_xlabel('Responding neuron (sorted by in-degree)')
ax.set_ylabel('Stimulated neuron (sorted by out-degree)')
ax.set_title('Binary connectivity matrix')

# Mark hub rows/cols
for i, neuron in enumerate(row_order):
    if out_hub[neuron]:
        ax.annotate('*', (display_mat.shape[1] + 0.5, i), color='red', fontsize=12,
                     ha='left', va='center', fontweight='bold')
for j, neuron in enumerate(col_order):
    if in_hub[neuron]:
        ax.annotate('*', (j, -0.8), color='red', fontsize=10,
                     ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(save_dir, 'connectivity_matrix.png'), dpi=200, bbox_inches='tight')

#%% ---- Plot: Hub neuron spatial locations ----
fig, ax = plt.subplots(figsize=(6, 6))
centroidX = np.array(data['photostim']['centroidX'], dtype=float)
centroidY = np.array(data['photostim']['centroidY'], dtype=float)

# All neurons
ax.scatter(centroidX, centroidY, s=10, c='gray', alpha=0.3, label='Non-hub')

# Stimulated neurons (non-hub)
stim_nonhub = unique_targets[~out_hub[unique_targets] & ~in_hub[unique_targets]]
ax.scatter(centroidX[stim_nonhub], centroidY[stim_nonhub], s=30, c='steelblue',
           alpha=0.6, label='Stimulated (non-hub)')

# Hub neurons
hub_idx = np.where(hub_neurons)[0]
labels_used = set()
for h in hub_idx:
    if out_hub[h] and in_hub[h]:
        c, lab = 'purple', 'Hub (in+out)'
    elif out_hub[h]:
        c, lab = 'red', 'Hub (out)'
    else:
        c, lab = 'orange', 'Hub (in)'
    lbl = lab if lab not in labels_used else None
    labels_used.add(lab)
    ax.scatter(centroidX[h], centroidY[h], s=80, c=c, edgecolors='k', linewidths=0.8, label=lbl)

ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_title(f'{mouse} {session} — Hub neuron locations')
ax.legend(loc='best', fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(save_dir, 'hub_spatial_map.png'), dpi=200, bbox_inches='tight')

#%% ---- Summary table ----
print("\n=== HUB NEURON SUMMARY ===")
print(f"{'Neuron':>8}  {'Out-deg':>8}  {'In-deg':>8}  {'Out-p':>8}  {'In-p':>8}  {'Type':>10}")
print("-" * 60)
for h in hub_idx:
    od = int(out_degree[h]) if not np.isnan(out_degree[h]) else '-'
    id_ = int(in_degree[h]) if not np.isnan(in_degree[h]) else '-'
    op = f"{out_pval[h]:.4f}" if not np.isnan(out_pval[h]) else '-'
    ip = f"{in_pval[h]:.4f}" if not np.isnan(in_pval[h]) else '-'
    htype = []
    if out_hub[h]:
        htype.append('out')
    if in_hub[h]:
        htype.append('in')
    print(f"{h:>8}  {od:>8}  {id_:>8}  {op:>8}  {ip:>8}  {'+'.join(htype):>10}")

#%% ---- Save results ----
results = {
    'mouse': mouse,
    'session': session,
    'conn_amp': conn_amp,
    'conn_pval': conn_pval,
    'conn_binary': conn_binary,
    'out_degree': out_degree,
    'in_degree': in_degree,
    'out_pval': out_pval,
    'in_pval': in_pval,
    'out_hub': out_hub,
    'in_hub': in_hub,
    'hub_neurons': hub_neurons,
    'hub_idx': hub_idx,
    'unique_targets': unique_targets,
    'p_conn': p_conn,
    'target_dist_thresh': target_dist_thresh,
    'nontarget_dist_min': nontarget_dist_min,
    'sig_alpha': sig_alpha,
    'hub_alpha': hub_alpha,
}
np.save(os.path.join(save_dir, 'hub_results.npy'), results)
print(f"\nResults saved to {save_dir}")
