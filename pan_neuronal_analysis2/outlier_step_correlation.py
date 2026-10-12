#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Compute per-neuron correlation with lickport step signal (step_vector convolved
with a 1s decaying exponential).  Compare outlier vs rest neurons.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib.pyplot as plt
import traceback

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
n_sd_threshold = 2   # outlier = mean + n_sd * SD of non-target amp
TAU_STEP = 0.2       # exponential decay time constant (seconds)

#%% ============================================================================
# CELL 3: Loop over sessions -- compute step correlation per neuron
# ============================================================================
all_corr_outlier = []   # per-neuron correlation with step signal (outlier cells)
all_corr_rest = []      # per-neuron correlation with step signal (rest cells)
session_labels = []

for mi, mouse in enumerate(mice):
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
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
                'conditioned_neuron', 'dt_si', 'step_time', 'reward_time',
            ]
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"  Skipping -- file not found.")
                continue

            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]

            # ---- Classify outlier neurons ----
            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            amp_ep0 = AMP[0]
            amp_masked = amp_ep0.copy()
            amp_masked[stimDist < 30] = np.nan
            mean_amp_nontarg = np.nanmean(amp_masked, axis=1)

            mu = np.nanmean(mean_amp_nontarg)
            sd = np.nanstd(mean_amp_nontarg)
            outlier_threshold = mu + n_sd_threshold * sd
            is_outlier = mean_amp_nontarg > outlier_threshold
            is_rest = ~is_outlier & np.isfinite(mean_amp_nontarg)
            print(f"  Outlier: {np.sum(is_outlier)}, Rest: {np.sum(is_rest)} "
                  f"(threshold={outlier_threshold:.4f})")

            if np.sum(is_outlier) == 0:
                print("  No outliers -- skipping.")
                continue

            # ---- Build step vector ----
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0

            step_vector, reward_vector, trial_start_vector = \
                bts.bci_time_series_fun(folder, data, rt_filled, dt_si)

            # ---- Convolve step_vector with decaying exponential ----
            # Kernel: exp(-t/tau) for t >= 0, sampled at dt_si
            n_kernel = int(5 * TAU_STEP / dt_si)  # 5 time constants
            t_kernel = np.arange(n_kernel) * dt_si
            kernel = np.exp(-t_kernel / TAU_STEP)
            kernel = kernel / np.sum(kernel)  # normalize

            step_conv = np.convolve(step_vector.astype(float), kernel, mode='full')[:len(step_vector)]

            # ---- Correlate each neuron's closed-loop activity with step_conv ----
            df = data['df_closedloop']  # (nCells, nFrames)
            n_cells = df.shape[0]

            # Truncate to common length
            n_common = min(df.shape[1], len(step_conv))
            step_sig = step_conv[:n_common]
            df_trunc = df[:, :n_common]

            corr_per_cell = np.full(n_cells, np.nan)
            for ci in range(n_cells):
                trace = df_trunc[ci, :]
                valid = np.isfinite(trace) & np.isfinite(step_sig)
                if np.sum(valid) > 100:
                    corr_per_cell[ci] = np.corrcoef(trace[valid], step_sig[valid])[0, 1]

            all_corr_outlier.extend(corr_per_cell[is_outlier])
            all_corr_rest.extend(corr_per_cell[is_rest])
            session_labels.append(f"{mouse} {session}")

            print(f"  Outlier mean corr: {np.nanmean(corr_per_cell[is_outlier]):.4f}, "
                  f"Rest mean corr: {np.nanmean(corr_per_cell[is_rest]):.4f}")

        except Exception:
            traceback.print_exc()
            continue

all_corr_outlier = np.array(all_corr_outlier)
all_corr_rest = np.array(all_corr_rest)
print(f"\nTotal: {len(all_corr_outlier)} outlier cells, {len(all_corr_rest)} rest cells")

#%% ============================================================================
# CELL 4: Plot -- histograms and comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Overlapping histograms
ax = axes[0]
bins = np.linspace(-0.5, 0.5, 40)
ax.hist(all_corr_rest, bins=bins, color='black', alpha=0.5, density=True, label='Rest')
ax.hist(all_corr_outlier, bins=bins, color='red', alpha=0.5, density=True, label='Outlier')
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axvline(np.nanmean(all_corr_outlier), color='red', lw=1.5, ls='-',
           label=f'Outlier mean={np.nanmean(all_corr_outlier):.3f}')
ax.axvline(np.nanmean(all_corr_rest), color='black', lw=1.5, ls='-',
           label=f'Rest mean={np.nanmean(all_corr_rest):.3f}')
ax.set_xlabel('Correlation with step signal')
ax.set_ylabel('Density')
ax.set_title('Neuron-step correlation')
ax.legend(fontsize=8)

# Bar plot of means
ax = axes[1]
means = [np.nanmean(all_corr_outlier), np.nanmean(all_corr_rest)]
sems = [np.nanstd(all_corr_outlier) / np.sqrt(np.sum(np.isfinite(all_corr_outlier))),
        np.nanstd(all_corr_rest) / np.sqrt(np.sum(np.isfinite(all_corr_rest)))]
bars = ax.bar(['Outlier', 'Rest'], means, yerr=sems, color=['red', 'black'],
              alpha=0.7, capsize=5, edgecolor='k', linewidth=0.8)
ax.set_ylabel('Mean correlation with step signal')
ax.set_title(f'{len(session_labels)} sessions')
ax.axhline(0, color='k', lw=0.5)

plt.suptitle(f'Correlation with convolved step vector (tau={TAU_STEP}s)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outlier_step_correlation.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"Outlier: mean={np.nanmean(all_corr_outlier):.4f}, "
      f"median={np.nanmedian(all_corr_outlier):.4f}")
print(f"Rest:    mean={np.nanmean(all_corr_rest):.4f}, "
      f"median={np.nanmedian(all_corr_rest):.4f}")
