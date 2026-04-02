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
from scipy.stats import pearsonr, spearmanr
import traceback
import importlib
import csv
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *
import plotting_functions as pf

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')
os.makedirs(PANEL_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'svg.fonttype': 'none',
})

mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

# QC filter
_qc_csv = os.path.join(RESULTS_DIR, 'qc', 'qc_summary.csv')
_qc_fail = set()
if os.path.exists(_qc_csv):
    with open(_qc_csv) as _f:
        for _r in csv.DictReader(_f):
            if _r['pass_qc'] != 'True':
                _qc_fail.add((_r['mouse'], _r['session']))
    print(f"QC filter: {len(_qc_fail)} sessions excluded")

print("Setup complete!")

#%% ============================================================================
# CELL 2: Loop through sessions — measure CN activity across trials
# ============================================================================
all_learning = []

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
            if (mouse, session) in _qc_fail:
                print(f"  Skipping {mouse} {session} -- failed QC")
                continue
            folder = (r'//allen/aind/scratch/BCI/2p-raw/'
                      + mouse + r'/' + session + '/pophys/')
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            bci_keys = ['df_closedloop', 'F', 'mouse', 'session',
                        'conditioned_neuron', 'dt_si', 'step_time',
                        'reward_time', 'BCI_thresholds']
            try:
                data = ddct.load_hdf5(folder, bci_keys, [])
            except FileNotFoundError:
                print(f"  Skipping -- file not found.")
                continue

            dt_si = data['dt_si']
            F = data['F']  # (frames, neurons, trials)
            trl = F.shape[2]
            n_frames = F.shape[0]
            tsta = np.arange(n_frames) * dt_si
            # Align so t=0 is trial start (~2s into the trace)
            t0_idx = min(int(round(2.0 / dt_si)), n_frames - 1)
            tsta = tsta - tsta[t0_idx]

            # Get conditioned neuron index
            cn_raw = data['conditioned_neuron']
            try:
                if hasattr(cn_raw, '__len__') and len(cn_raw) > 0:
                    if hasattr(cn_raw[0], '__len__') and len(cn_raw[0]) > 0:
                        cn = int(cn_raw[0][0])
                    else:
                        cn = int(cn_raw[0])
                else:
                    cn = int(cn_raw)
                if np.isnan(cn) or cn < 0 or cn >= F.shape[1]:
                    print(f"  Skipping -- invalid CN index: {cn_raw}")
                    continue
            except (ValueError, TypeError, IndexError) as e:
                print(f"  Skipping -- can't parse CN: {cn_raw} ({e})")
                continue

            # Parse reward times for hit detection
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)

            # CN activity: trial-by-trial
            # F[frames, neurons, trials] — extract CN trace
            cn_traces = F[:, cn, :]  # (frames, trials)

            # Define time windows
            # Baseline: pre-trial (tsta < 0)
            t_baseline = np.where(tsta < 0)[0]
            # Full trial: tsta >= 0
            t_trial = np.where(tsta >= 0)[0]
            # Late epoch: 1s before typical reward time (~last 1s of trial)
            # Use tsta between max(tsta)-1 and max(tsta), or
            # reward is at variable times, so use a fixed late window
            t_late = np.where((tsta > (tsta[-1] - 1)) & (tsta <= tsta[-1]))[0]
            if len(t_late) == 0:
                t_late = t_trial[-int(1/dt_si):]

            # Mean CN activity per trial in each window
            cn_baseline = np.nanmean(cn_traces[t_baseline, :], axis=0)  # (trials,)
            cn_trial = np.nanmean(cn_traces[t_trial, :], axis=0)        # (trials,)
            cn_late = np.nanmean(cn_traces[t_late, :], axis=0)          # (trials,)

            # Baseline-subtracted
            cn_trial_sub = cn_trial - cn_baseline
            cn_late_sub = cn_late - cn_baseline

            # Hit rate in sliding window
            win = 20
            hit_rate = np.convolve(hit.astype(float),
                                   np.ones(win)/win, mode='same')

            # Summary stats
            # Early vs late thirds
            n3 = trl // 3
            early_trials = np.arange(n3)
            late_trials = np.arange(trl - n3, trl)

            cn_trial_early = np.nanmean(cn_trial_sub[early_trials])
            cn_trial_late = np.nanmean(cn_trial_sub[late_trials])
            cn_late_early = np.nanmean(cn_late_sub[early_trials])
            cn_late_late = np.nanmean(cn_late_sub[late_trials])
            hit_rate_early = np.nanmean(hit[early_trials])
            hit_rate_late = np.nanmean(hit[late_trials])

            rec = {
                'mouse': mouse, 'session': session,
                'n_trials': trl, 'cn_idx': cn,
                'cn_trial_sub': cn_trial_sub,    # per-trial array
                'cn_late_sub': cn_late_sub,       # per-trial array
                'hit': hit,
                'hit_rate': hit_rate,
                'cn_trial_early': cn_trial_early,
                'cn_trial_late': cn_trial_late,
                'cn_late_early': cn_late_early,
                'cn_late_late': cn_late_late,
                'hit_rate_early': hit_rate_early,
                'hit_rate_late': hit_rate_late,
                'dt_si': dt_si,
                'tsta': tsta,
            }
            all_learning.append(rec)
            print(f"  CN={cn}, trials={trl}, "
                  f"hit early={hit_rate_early:.2f} late={hit_rate_late:.2f}, "
                  f"CN_late early={cn_late_early:.3f} late={cn_late_late:.3f}")

        except Exception as e:
            print(f"  FAILED {mouse} {session}: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(all_learning)} sessions.")

#%% ============================================================================
# CELL 3: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'bci_learning.npy'),
        all_learning, allow_pickle=True)
print("Saved bci_learning.npy")

#%% ============================================================================
# CELL 4: Load
# ============================================================================
all_learning = np.load(
    os.path.join(RESULTS_DIR, 'bci_learning.npy'),
    allow_pickle=True).tolist()
print(f"Loaded {len(all_learning)} sessions.")

#%% ============================================================================
# CELL 5: Figure — CN activity vs trial number (grand mean ± SEM)
# ============================================================================
n_s = len(all_learning)
mice_unique = sorted(set(r['mouse'] for r in all_learning))
mouse_colors = {m: plt.cm.tab10(i) for i, m in enumerate(mice_unique)}

# Find max trial count across sessions, pad shorter sessions with NaN
max_trl = max(len(r['cn_trial_sub']) for r in all_learning)

cn_trial_mat = np.full((n_s, max_trl), np.nan)
cn_late_mat = np.full((n_s, max_trl), np.nan)
hit_mat = np.full((n_s, max_trl), np.nan)

for si, r in enumerate(all_learning):
    n_t = len(r['cn_trial_sub'])
    cn_trial_mat[si, :n_t] = r['cn_trial_sub']
    cn_late_mat[si, :n_t] = r['cn_late_sub']
    hit_mat[si, :n_t] = r['hit'].astype(float)

trial_ax = np.arange(1, max_trl + 1)

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5),
                         gridspec_kw={'left': 0.08, 'right': 0.97,
                                      'bottom': 0.22, 'top': 0.85,
                                      'wspace': 0.35})

for ax, data_mat, ylabel, title in [
    (axes[0], cn_trial_mat, 'CN activity\n(trial - baseline)', 'CN during trial'),
    (axes[1], cn_late_mat, 'CN activity\n(late epoch - baseline)', 'CN late epoch'),
    (axes[2], hit_mat, 'Hit rate', 'Hit rate'),
]:
    # Number of sessions contributing at each trial
    n_valid = np.sum(np.isfinite(data_mat), axis=0)
    # Only plot where at least half the sessions contribute
    mask = n_valid >= n_s // 2

    from scipy.signal import medfilt
    grand_mean = medfilt(np.nanmean(data_mat, axis=0),5)
    grand_sem = medfilt(np.nanstd(data_mat, axis=0) / np.sqrt(n_valid),5)

    ax.fill_between(trial_ax[mask], (grand_mean - grand_sem)[mask],
                    (grand_mean + grand_sem)[mask], alpha=0.3, color='k')
    ax.plot(trial_ax[mask], grand_mean[mask], 'k-', linewidth=1.5)
    ax.axhline(0, color='k', ls='--', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Trial')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(PANEL_DIR, 'bci_learning_vs_trial.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, 'bci_learning_vs_trial.svg'))
plt.show()
print("Saved bci_learning_vs_trial")

#%% ============================================================================
# CELL 6: Figure — CN activity increase early vs late
# ============================================================================
n_s = len(all_learning)
mice_unique = sorted(set(r['mouse'] for r in all_learning))
mouse_colors = {m: plt.cm.tab10(i) for i, m in enumerate(mice_unique)}

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5),
                         gridspec_kw={'left': 0.08, 'right': 0.97,
                                      'bottom': 0.22, 'top': 0.85,
                                      'wspace': 0.35})

# --- Panel 1: CN activity (full trial) early vs late ---
ax = axes[0]
for r in all_learning:
    c = mouse_colors[r['mouse']]
    ax.plot([0, 1], [r['cn_trial_early'], r['cn_trial_late']],
            '-o', color=c, markersize=3, linewidth=0.8, alpha=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Early', 'Late'])
ax.set_ylabel('CN activity (trial mean - baseline)')
ax.set_title('CN during trial')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel 2: CN activity (late epoch, 1s pre-reward) early vs late ---
ax = axes[1]
for r in all_learning:
    c = mouse_colors[r['mouse']]
    ax.plot([0, 1], [r['cn_late_early'], r['cn_late_late']],
            '-o', color=c, markersize=3, linewidth=0.8, alpha=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Early', 'Late'])
ax.set_ylabel('CN activity (late epoch - baseline)')
ax.set_title('CN during late epoch')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel 3: Hit rate early vs late ---
ax = axes[2]
for r in all_learning:
    c = mouse_colors[r['mouse']]
    ax.plot([0, 1], [r['hit_rate_early'], r['hit_rate_late']],
            '-o', color=c, markersize=3, linewidth=0.8, alpha=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Early', 'Late'])
ax.set_ylabel('Hit rate')
ax.set_title('Hit rate')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=mouse_colors[m], marker='o', linewidth=0.8,
                  markersize=3, label=m) for m in mice_unique]
axes[2].legend(handles=handles, loc='lower right', frameon=False, fontsize=6)

fig.savefig(os.path.join(PANEL_DIR, 'bci_learning_early_vs_late.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, 'bci_learning_early_vs_late.svg'))
plt.show()

#%% ============================================================================
# CELL 6: Figure — CN activity time course per mouse (across sessions)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(7, 4),
                         gridspec_kw={'left': 0.08, 'right': 0.97,
                                      'bottom': 0.12, 'top': 0.92,
                                      'wspace': 0.30, 'hspace': 0.45})
axes = axes.ravel()

for mi, m in enumerate(mice_unique):
    ax = axes[mi] if mi < len(axes) else None
    if ax is None:
        break
    sessions = [r for r in all_learning if r['mouse'] == m]
    # Sort by session date
    sessions.sort(key=lambda r: r['session'])

    for si, r in enumerate(sessions):
        # Normalize trial index to [0, 1] for comparison across sessions
        n_t = len(r['cn_late_sub'])
        x = np.linspace(0, 1, n_t)
        # Smooth with running average
        win = min(20, n_t // 3)
        if win > 1:
            kernel = np.ones(win) / win
            smoothed = np.convolve(r['cn_late_sub'], kernel, mode='same')
        else:
            smoothed = r['cn_late_sub']
        alpha = 0.3 + 0.7 * si / max(len(sessions) - 1, 1)
        ax.plot(x, smoothed, linewidth=0.8, alpha=alpha, color='k')

    ax.axhline(0, color='k', ls='--', alpha=0.3, linewidth=0.5)
    ax.set_title(f'{m} ({len(sessions)} sessions)')
    ax.set_xlabel('Trial (normalized)')
    ax.set_ylabel('CN late epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide unused axes
for j in range(len(mice_unique), len(axes)):
    axes[j].set_visible(False)

fig.savefig(os.path.join(PANEL_DIR, 'bci_learning_cn_timecourse.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, 'bci_learning_cn_timecourse.svg'))
plt.show()

#%% ============================================================================
# CELL 7: Summary stats — save txt
# ============================================================================
from scipy.stats import wilcoxon

txt_path = os.path.join(RESULTS_DIR, 'bci_learning_summary.txt')
with open(txt_path, 'w') as f:
    f.write("BCI Learning Summary\n")
    f.write("=" * 70 + "\n\n")

    # Per-session table
    f.write(f"{'Mouse':8s} {'Session':8s} {'nTrl':>5s} "
            f"{'CN_trial_E':>10s} {'CN_trial_L':>10s} "
            f"{'CN_late_E':>10s} {'CN_late_L':>10s} "
            f"{'hitR_E':>7s} {'hitR_L':>7s}\n")
    f.write("-" * 85 + "\n")
    for r in all_learning:
        f.write(f"{r['mouse']:8s} {r['session']:8s} {r['n_trials']:5d} "
                f"{r['cn_trial_early']:+10.4f} {r['cn_trial_late']:+10.4f} "
                f"{r['cn_late_early']:+10.4f} {r['cn_late_late']:+10.4f} "
                f"{r['hit_rate_early']:7.3f} {r['hit_rate_late']:7.3f}\n")

    # Population stats (filter NaN)
    cn_trial_diff = np.array([r['cn_trial_late'] - r['cn_trial_early'] for r in all_learning])
    cn_late_diff = np.array([r['cn_late_late'] - r['cn_late_early'] for r in all_learning])
    hit_diff = np.array([r['hit_rate_late'] - r['hit_rate_early'] for r in all_learning])

    f.write(f"\nPOPULATION (late third - early third), n={n_s}\n")
    f.write("-" * 70 + "\n")
    for label, diff in [('CN trial', cn_trial_diff),
                        ('CN late epoch', cn_late_diff),
                        ('Hit rate', hit_diff)]:
        ok = np.isfinite(diff)
        d = diff[ok]
        n_ok = len(d)
        n_pos = np.sum(d > 0)
        try:
            _, p = wilcoxon(d)
        except:
            p = 1.0
        f.write(f"  {label:15s}: mean={np.mean(d):+.4f}, median={np.median(d):+.4f}, "
                f"{n_pos}/{n_ok} increase, Wilcoxon p={p:.4f}\n")

    # Per-mouse summary
    f.write(f"\nPER-MOUSE SUMMARY\n")
    f.write("-" * 70 + "\n")
    for m in mice_unique:
        recs = [r for r in all_learning if r['mouse'] == m]
        d = np.array([r['cn_late_late'] - r['cn_late_early'] for r in recs])
        hr = np.array([r['hit_rate_late'] - r['hit_rate_early'] for r in recs])
        f.write(f"  {m}: {len(recs)} sessions, "
                f"CN_late diff={np.mean(d):+.4f}, "
                f"hit rate diff={np.mean(hr):+.3f}\n")

print(f"Saved to {txt_path}")
