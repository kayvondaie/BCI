#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()

QC_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results', 'qc')
os.makedirs(QC_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'svg.fonttype': 'none',
})

mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

print("Setup complete!")

#%% ============================================================================
# CELL 2: Loop through all sessions and generate QC PNGs
# ============================================================================

from scipy.stats import pearsonr

summary_rows = []

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
            folder = (r'//allen/aind/scratch/BCI/2p-raw/'
                      + mouse + r'/' + session + '/pophys/')
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['dt_si', 'mouse', 'session']
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"  Skipping -- file not found.")
                continue

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            dt_si = data['dt_si']
            favg_raw = data['photostim']['favg_raw']  # (timepoints, neurons, groups)
            n_tp, n_neurons, n_groups = favg_raw.shape
            t_axis = np.arange(n_tp) * dt_si

            # --- Identify target neurons per group (stimDist < 10 um) ---
            target_traces = []
            for gi in range(n_groups):
                targ = np.where(stimDist[:, gi] < 10)[0]
                if targ.size > 0:
                    for ti in targ:
                        trace = favg_raw[:, ti, gi]
                        bl = np.nanmean(trace[0:3])
                        if bl > 0:
                            target_traces.append((trace - bl) / bl)

            # --- Figure ---
            fig, axes = plt.subplots(1, 3, figsize=(8, 2.2),
                                     gridspec_kw={'left': 0.07, 'right': 0.97,
                                                  'bottom': 0.22, 'top': 0.82,
                                                  'wspace': 0.35})

            # Panel 1: Average response of targeted neurons (from favg_raw)
            ax = axes[0]
            if len(target_traces) > 0:
                traces_arr = np.array(target_traces)  # (n_targets, n_tp)
                mean_trace = np.nanmean(traces_arr, axis=0)
                sem_trace = np.nanstd(traces_arr, axis=0) / np.sqrt(traces_arr.shape[0])
                ax.fill_between(t_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                                alpha=0.3, color='k')
                ax.plot(t_axis, mean_trace, 'k-', linewidth=1)
                ax.set_title(f'Target response (n={len(target_traces)})')
            else:
                ax.text(0.5, 0.5, 'No targets', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title('Target response')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(r'$\Delta F/F$')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Panel 2: AMP[0] vs stimDist
            ax = axes[1]
            amp0_flat = AMP[0].ravel()
            dist_flat = stimDist.ravel()
            ok = np.isfinite(amp0_flat) & np.isfinite(dist_flat) & (dist_flat < 500)
            if np.sum(ok) > 0:
                ax.scatter(dist_flat[ok], amp0_flat[ok], s=3, alpha=0.3, color='k',
                           edgecolors='none')
                ax.axhline(0, color='k', ls='--', alpha=0.3, linewidth=0.5)
                ax.axvline(10, color='r', ls='--', alpha=0.5, linewidth=0.5, label='10 µm')
            ax.set_xlabel('Distance (µm)')
            ax.set_ylabel('AMP[0]')
            ax.set_title('AMP[0] vs distance')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Panel 3: AMP[1] vs AMP[0] for target neurons (stimDist < 10)
            ax = axes[2]
            targ_mask = stimDist < 10
            amp0_targ = AMP[0][targ_mask]
            amp1_targ = AMP[1][targ_mask]
            ok = np.isfinite(amp0_targ) & np.isfinite(amp1_targ)
            if np.sum(ok) > 0:
                ax.scatter(amp0_targ[ok], amp1_targ[ok], s=8, alpha=0.5, color='k',
                           edgecolors='none')
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                        max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.5)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
            ax.set_xlabel('AMP[0] (early)')
            ax.set_ylabel('AMP[1] (late)')
            ax.set_title(f'Targets: AMP[1] vs AMP[0] (n={np.sum(ok)})')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.suptitle(f'{mouse}  {session}', fontsize=9, fontweight='bold')
            out_path = os.path.join(QC_DIR, f'qc_{mouse}_{session}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  Saved {out_path}")

            # --- Collect summary metrics ---
            mean_resp = np.nan
            if len(target_traces) > 0:
                traces_arr = np.array(target_traces)
                # Peak of mean trace as summary of response amplitude
                mean_resp = np.nanmax(np.nanmean(traces_arr, axis=0))

            r_epochs = np.nan
            p_epochs = np.nan
            n_targ = 0
            targ_mask = stimDist < 10
            amp0_targ = AMP[0][targ_mask]
            amp1_targ = AMP[1][targ_mask]
            ok_ep = np.isfinite(amp0_targ) & np.isfinite(amp1_targ)
            n_targ = int(np.sum(ok_ep))
            if n_targ >= 3:
                r_epochs, p_epochs = pearsonr(amp0_targ[ok_ep], amp1_targ[ok_ep])

            summary_rows.append({
                'mouse': mouse, 'session': session,
                'n_targets': n_targ,
                'mean_peak_dff': mean_resp,
                'r_amp0_amp1': r_epochs,
                'p_amp0_amp1': p_epochs,
            })

        except Exception:
            traceback.print_exc()
            print(f"  ERROR in {mouse} {session}")
            continue

# --- Write summary CSV with pass/fail ---
import csv

# QC criteria: need targets and significant epoch correlation
for row in summary_rows:
    row['pass_qc'] = (row['n_targets'] > 0
                      and np.isfinite(row['r_amp0_amp1'])
                      and row['p_amp0_amp1'] < 0.05)

csv_path = os.path.join(QC_DIR, 'qc_summary.csv')
fields = ['mouse', 'session', 'n_targets', 'mean_peak_dff',
          'r_amp0_amp1', 'p_amp0_amp1', 'pass_qc']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary_rows)

n_pass = sum(r['pass_qc'] for r in summary_rows)
n_fail = len(summary_rows) - n_pass
print(f"\nQC summary: {n_pass} pass, {n_fail} fail")
for row in summary_rows:
    if not row['pass_qc']:
        print(f"  FAIL: {row['mouse']} {row['session']} "
              f"(n_targ={row['n_targets']}, r={row['r_amp0_amp1']:.3f}, p={row['p_amp0_amp1']:.3f})")

print(f"\nCSV saved to {csv_path}")
print("Done! All QC PNGs saved to:", QC_DIR)
