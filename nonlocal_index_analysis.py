#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Non-Local Index (NLI) analysis.
#
# From the gradient: dW_{i,j} ∝ (R - R̄) * Σ_k (r_k - r̄_k) * r_i * w_{j,k}
#
# Fix a stim group g targeting neuron j.
#   - ΔW_{:,j} = change in connection from each nontarget i to j  (1 × N_nt)
#   - CC_{i,k}^t = r_i^t * (r_k^t - r̄_k)  for each nontarget i, neuron k, trial t
#   - Fit: ΔW_{:,j} = Σ_w β_w * CC_{:,k}^w   (windowed, CV across nontargets)
#   - NLI(g,k) = test r from this fit
#   - w_{j,k} = AMP[0][k, g]  (directly measured outgoing weight)
#
# Weight transport predicts: NLI(g,k) correlates with w_{j,k} across k.
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr, wilcoxon
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *
import plotting_functions as pf

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
})

print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
N_BASELINE = 20

all_sessions = []

# Grand-average accumulators: per (group, k) observations
grand_nli = []
grand_w = []

print(f"Config: simple correlation NLI (no CV)")

#%% ============================================================================
# CELL 3: Main loop — per stim group, per downstream neuron k
# ============================================================================

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

            BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
            thr = BCI_thresholds[1, :]
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i - 1]
            if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
                thr[0] = thr[np.isfinite(thr)][0]
            BCI_thresholds[1, :] = thr

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # ---- Pre-epoch activity ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)
            # epoch_act: (n_neurons, trl)

            bl_trials = np.arange(min(N_BASELINE, trl))
            bl_mean = np.nanmean(epoch_act[:, bl_trials], axis=1)
            pop_dev = epoch_act - bl_mean[:, np.newaxis]  # (n_neurons, trl)

            # ---- Loop over stim groups ----
            n_groups = stimDist.shape[1]
            dist_target_lt = 10
            dist_nontarg_min = 30

            sess_nli = []  # NLI values for this session
            sess_w = []    # corresponding w_{j,k}
            n_groups_used = 0

            for gi in range(n_groups):
                # Targets of this group
                cl = np.where(
                    (stimDist[:, gi] < dist_target_lt) &
                    (AMP[0][:, gi] > 0.1) &
                    (AMP[1][:, gi] > 0.1)
                )[0]
                if cl.size == 0:
                    continue

                # Nontargets
                nontargets = np.where(
                    (stimDist[:, gi] > dist_nontarg_min) &
                    (stimDist[:, gi] < 1000)
                )[0]
                n_nt = len(nontargets)
                if n_nt < 10:
                    continue

                # ΔW_{:,j} for this group
                dw = AMP[1][nontargets, gi] - AMP[0][nontargets, gi]
                dw = np.nan_to_num(dw, nan=0.0)

                # Outgoing weights from group g: w_{j,k} = AMP[0][k, g]
                w_out = AMP[0][:, gi].copy()
                w_out[stimDist[:, gi] < dist_nontarg_min] = np.nan  # NaN close neurons

                # Valid downstream neurons k
                valid_k = np.where(np.isfinite(w_out))[0]
                n_valid_k = len(valid_k)
                if n_valid_k < 10:
                    continue

                # Activity of nontargets: (n_nt, trl)
                r_nt = epoch_act[nontargets, :]

                # CC_{i,k} = Σ_t r_i(t) * (r_k(t) - r̄_k), summed over all trials
                # cc_mat: (n_nt, n_valid_k)
                cc_mat = r_nt @ pop_dev[valid_k, :].T

                # NLI(k) = pearson(CC_{:,k}, ΔW) across nontargets
                # Vectorized across all k
                dw_m = dw - dw.mean()
                cc_m = cc_mat - cc_mat.mean(axis=0, keepdims=True)
                num = (cc_m * dw_m[:, np.newaxis]).sum(axis=0)
                den = np.sqrt((cc_m**2).sum(axis=0) *
                              (dw_m**2).sum()) + 1e-30
                NLI_k = num / den

                w_k = w_out[valid_k]
                sess_nli.append(NLI_k)
                sess_w.append(w_k)
                n_groups_used += 1

            if n_groups_used == 0:
                print("  No valid groups.")
                continue

            # Pool across groups within session
            sess_nli = np.concatenate(sess_nli)
            sess_w = np.concatenate(sess_w)

            rho, p_rho = spearmanr(sess_w, sess_nli)
            print(f"  {n_groups_used} groups, {len(sess_nli)} (group,k) obs | "
                  f"rho(w, NLI) = {rho:.4f}, p = {p_rho:.2e}")

            all_sessions.append({
                'mouse': mouse,
                'session': session,
                'n_groups': n_groups_used,
                'n_obs': len(sess_nli),
                'rho': rho,
                'p_rho': p_rho,
            })

            grand_nli.append(sess_nli)
            grand_w.append(sess_w)

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

grand_nli = np.concatenate(grand_nli)
grand_w = np.concatenate(grand_w)
print(f"\n{len(all_sessions)} sessions, {len(grand_nli)} total (group,k) observations")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'nonlocal_index_results.npy'),
        all_sessions, allow_pickle=True)
np.savez(os.path.join(RESULTS_DIR, 'nonlocal_index_grand.npz'),
         nli=grand_nli, w=grand_w)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_sessions = np.load(
    os.path.join(RESULTS_DIR, 'nonlocal_index_results.npy'),
    allow_pickle=True).tolist()
gd = np.load(os.path.join(RESULTS_DIR, 'nonlocal_index_grand.npz'))
grand_nli = gd['nli']
grand_w = gd['w']
print(f"Loaded: {len(all_sessions)} sessions, {len(grand_nli)} obs")

#%% ============================================================================
# CELL 6: Summary figure
# ============================================================================

rho_all = np.array([s['rho'] for s in all_sessions])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Panel A: Grand average — NLI vs w_{j,k} binned ---
ax = axes[0]
plt.sca(ax)
pf.mean_bin_plot(grand_w, grand_nli, 10, 1, 1, 'k')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel(r'Outgoing weight $w_{j,k}$')
ax.set_ylabel('NLI (CV test r)')
ax.set_title(r'NLI vs $w_{j,k}$ (grand average)', fontweight='bold')
rho_grand, p_grand = spearmanr(grand_w, grand_nli)
ax.text(0.05, 0.95, f'rho={rho_grand:.4f}, p={p_grand:.1e}\nn={len(grand_w)}',
        transform=ax.transAxes, va='top', fontsize=11)

# --- Panel B: Per-session rho distribution ---
ax = axes[1]
ax.hist(rho_all, bins=15, color='#3498db', edgecolor='k', alpha=0.8)
ax.axvline(0, color='k', ls='--', alpha=0.5)
ax.axvline(np.median(rho_all), color='r', ls='-', lw=2,
           label=f'median={np.median(rho_all):.3f}')
try:
    _, p_pop = wilcoxon(rho_all)
except Exception:
    p_pop = 1.0
n_pos = np.sum(rho_all > 0)
ax.set_xlabel(r'Spearman $\rho$($w_{j,k}$, NLI)')
ax.set_ylabel('Sessions')
ax.set_title('Per-session weight transport test', fontweight='bold')
ax.text(0.05, 0.95, f'{n_pos}/{len(rho_all)} > 0\nWilcoxon p={p_pop:.4f}',
        transform=ax.transAxes, va='top', fontsize=11)
ax.legend(fontsize=10)

# --- Panel C: Raw scatter (all sessions, subsampled) ---
ax = axes[2]
rng = np.random.default_rng(0)
n_show = min(50000, len(grand_w))
idx = rng.choice(len(grand_w), n_show, replace=False)
ax.scatter(grand_w[idx], grand_nli[idx], s=2, alpha=0.05, c='gray')
ax.axhline(0, color='k', ls='--', alpha=0.3)
ax.set_xlabel(r'$w_{j,k}$')
ax.set_ylabel('NLI')
ax.set_title('All (group, k) observations', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig20_nonlocal_index.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 20 saved.")

#%% ============================================================================
# CELL 7: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'nonlocal_index_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("NON-LOCAL INDEX ANALYSIS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("NLI(g,k) = Pearson r(CC_{:,k}, dW) across nontargets\n")
    f.write("CC_{i,k} = sum_t r_i(t) * (r_k(t) - r_bar_k)\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL:\n")
    f.write("  For each stim group g (targeting neuron j):\n")
    f.write("    dW_{:,j} = sum_w beta_w * CC_{:,k}^w\n")
    f.write("    CC_{i,k}^t = r_i^t * (r_k^t - r_bar_k)\n")
    f.write("    NLI(g,k) = CV test r (across nontargets i)\n")
    f.write("    w_{j,k} = AMP[0][k, g] (outgoing weight from j)\n")
    f.write("  Weight transport: NLI(g,k) should increase with w_{j,k}\n\n")

    f.write("GRAND AVERAGE\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Total (group, k) observations: {len(grand_w)}\n")
    f.write(f"  Spearman rho(w, NLI): {rho_grand:.4f}, p={p_grand:.2e}\n\n")

    f.write("POPULATION (per-session rho)\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Sessions: {len(rho_all)}\n")
    f.write(f"  Median rho: {np.median(rho_all):.4f}\n")
    f.write(f"  Mean rho:   {np.mean(rho_all):.4f}\n")
    f.write(f"  %>0: {n_pos}/{len(rho_all)}\n")
    f.write(f"  Wilcoxon p: {p_pop:.6f}\n")
    sig = '***' if p_pop < 0.001 else '**' if p_pop < 0.01 else '*' if p_pop < 0.05 else 'ns'
    f.write(f"  Significance: {sig}\n\n")

    f.write("PER-SESSION DETAIL\n")
    f.write("-" * 70 + "\n")
    f.write(f"  {'mouse':8s} {'session':8s} {'groups':>6s} {'n_obs':>7s} "
            f"{'rho':>7s} {'p':>10s}\n")
    for s in all_sessions:
        f.write(f"  {s['mouse']:8s} {s['session']:8s} {s['n_groups']:6d} "
                f"{s['n_obs']:7d} "
                f"{s['rho']:+7.4f} {s['p_rho']:10.2e}\n")

print(f"Report saved to: {report_path}")
