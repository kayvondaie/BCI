"""
Three-factor variance explained, inhibitory-photostim version — CODE OCEAN / PORTABLE.

Same analysis as three_factor_variance_explained_inhibitory.py; only the data root,
results dir, and QC-CSV path are made portable so it reproduces from the PACKAGED
inhibitory bundle (F:\inhibitory_data_2026 locally, or a Code Ocean data-asset mount)
instead of the //allen share. Use it to VALIDATE the F: bundle before uploading to CO.

------------------------------------------------------------------------------
SET DATA_ROOT below for your environment:
  - Local validation:  r'F:\inhibitory_data_2026'
  - Code Ocean:        '/root/capsule/data/<inhibitory_asset_name>'
Expected layout (produced by copy_sessions_inhibitory.py):
  {DATA_ROOT}/{mouse}/{session}/pophys/*.h5              (ddct.load_hdf5 reads these)
  {DATA_ROOT}/{mouse}/{session}/pophys/suite2p*/...      (full suite2p folders)
Requires counter22 in session_counting.py (now folded into the canonical module).
------------------------------------------------------------------------------
"""

#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import csv as _csv
import re  as _re
import glob as _glob
import h5py as _h5py
import pickle as _pickle
import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, wilcoxon
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

# ==== CONFIG: set DATA_ROOT for your environment ====
#   Local validation:  r'F:\inhibitory_data_2026'
#   Code Ocean:        '/root/capsule/data/<inhibitory_asset_name>'
DATA_ROOT   = r'F:\inhibitory_data_2026'
RESULTS_DIR = os.path.join(_THIS_DIR, 'results')   # headless-safe; figures saved here
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# QC: Type=Inhibitory, Pre OR Post = Good/Ok, no non-zero offset notes
# -----------------------------
_QC_CSV        = os.path.join(_THIS_DIR, 'Inhibitory & Pan-neuronal BCI summary - Sheet1.csv')
_OK_QUALITY    = {'good', 'ok'}
_BAD_OFFSET_RE = _re.compile(r'offset\s*=\s*[1-9]', _re.IGNORECASE)

_qc_keep = set()
_qc_dropped_offset = []
_cur_m = _cur_t = None
with open(_QC_CSV, encoding='utf-8') as _f:
    _rows = list(_csv.reader(_f))
for _row in _rows[2:]:
    if not _row:
        continue
    _subj = _row[0].strip() if len(_row) > 0 else ''
    _type = _row[1].strip() if len(_row) > 1 else ''
    if _subj: _cur_m = _subj
    if _type: _cur_t = _type
    if (_cur_t or '').lower() != 'inhibitory':
        continue
    _date  = _row[2].strip()        if len(_row) > 2 else ''
    _pre   = _row[4].strip().lower() if len(_row) > 4 else ''
    _post  = _row[5].strip().lower() if len(_row) > 5 else ''
    _notes = _row[6]                 if len(_row) > 6 else ''
    if (not _date or _cur_m is None or
        (_pre not in _OK_QUALITY and _post not in _OK_QUALITY)):
        continue
    try:
        _m, _d, _y = _date.split('/')
        _session = f'{int(_m):02d}{int(_d):02d}{int(_y):02d}'
    except Exception:
        continue
    if _BAD_OFFSET_RE.search(_notes or ''):
        _qc_dropped_offset.append((_cur_m, _session, (_notes or '').strip()))
        continue
    _qc_keep.add((_cur_m, _session))

mice = sorted({m for m, _ in _qc_keep})
print(f'QC CSV: kept {len(_qc_keep)} inhibitory sessions across mice {mice}')
if _qc_dropped_offset:
    print(f'QC CSV: dropped {len(_qc_dropped_offset)} sessions for non-zero offset:')
    for _m, _s, _n in _qc_dropped_offset:
        print(f'   {_m} {_s}  — "{_n}"')
else:
    print('QC CSV: no sessions dropped for non-zero offset (regex did not match anything)')

session_counting._BASE_DIR_OVERRIDE = DATA_ROOT
list_of_dirs = session_counting.counter22(mice)

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'figure.titlesize': 18,
})
print("Setup complete!")


#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
tau_elig = 10
N_BASELINE = 20

# Sliding window parameters
WIN_SIZE = 10
WIN_STEP = 5

# Fitting
fit_type = 'pinv'
n_cv_folds = 5

# Target-cell inclusion threshold (artifact-free amp is in ΔF/F units)
TARGET_AMP_THR  = 0.05    # was 0.1 in raw-F-amp version
NONTARG_DIST_LO = 30
NONTARG_DIST_HI = 1000

# Pairwise coactivity mode:
#   'dev2'          — pre × (post − baseline)    (baseline-subtracted, default)
#   'dot_prod'      — pre × post                 (raw product, no baseline subtract)
#   'target_only'   — pre only                   (target activity, no non-target term)
#   'nontarget_only'— (post − baseline) only     (non-target activity, no target term)
CC_MODE = 'dev2'

# If True, append the per-group change in target neuron's direct response
# (dTarget = mean(AMP[1][cl] - AMP[0][cl]) per group, broadcast across pairs)
# as an extra feature column in X. Same correction as the sliding-window
# script: controls for the confound where a target driven harder post
# inflates dW for its non-target neighbors.
CONTROL_DTARGET = True

# If True, restrict the non-target pool to cells that are themselves responsive
# targets in some other group — i.e. focus on I→I connections only. The
# headline results suggest these are the cells with the strongest connections,
# so restricting can sharpen the signal.
RESTRICT_TO_RESPONSIVE_TARGETS = False

all_results = []
print(f"Config: win={WIN_SIZE}, step={WIN_STEP}, {fit_type}, {n_cv_folds}-fold CV, "
      f"CC_MODE={CC_MODE}, CONTROL_DTARGET={CONTROL_DTARGET}, "
      f"RESTRICT_TO_RESPONSIVE_TARGETS={RESTRICT_TO_RESPONSIVE_TARGETS}")


#%% ============================================================================
# CELL 3: Main loop — per-(mouse, session), single group (all non-targets)
# ============================================================================
def _load_stim_params_from_h5(folder, ps_epoch):
    """stim_params is an h5 Group of sub-datasets (incl. scalar `total_duration`)
    that ddct.load_hdf5 can't read. Pull it directly via h5py."""
    all_h5 = sorted(_glob.glob(os.path.join(folder, '*.h5')))
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
    with _h5py.File(cand[0], 'r') as f:
        if 'stim_params' in f:
            for k in f['stim_params'].keys():
                try:
                    sp[k] = f['stim_params'][k][()]
                except Exception:
                    pass
    return sp


def _decode_event_times(v, trl):
    """For BCI111/BCI116-era data: ddct.load_hdf5 returns step_time / reward_time
    as a 0-d ndarray of pickled bytes. Decode and flatten each (1, n_events)
    per-trial entry. Fall back to parse_hdf5_array_string for older sessions."""
    raw = v.item() if hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0 else v
    if isinstance(raw, (bytes, np.bytes_)):
        try:
            obj = _pickle.loads(raw)
            return [np.asarray(x).ravel() for x in obj]
        except Exception:
            pass
    return parse_hdf5_array_string(v, trl)


for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where(list_of_dirs['Mouse'] == mouse)[0]

    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            if (mouse, session) not in _qc_keep:
                continue                                # silent: not in QC
            folder = os.path.join(DATA_ROOT, mouse, session, 'pophys') + '/'
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

            # Attach stim_params for the artifact-free amp function
            for ep in ('photostim', 'photostim2'):
                if ep in data:
                    sp = _load_stim_params_from_h5(folder, ep)
                    if sp is not None:
                        data[ep]['stim_params'] = sp

            BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
            thr = BCI_thresholds[1, :]
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i - 1]
            if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
                thr[0] = thr[np.isfinite(thr)][0]
            BCI_thresholds[1, :] = thr

            AMP, stimDist = compute_amp_from_photostim_artifact_free(
                mouse, data, folder)
            if len(AMP) < 2:
                print(f"  Need 2 photostim epochs for dW; got {len(AMP)}. Skipping.")
                continue

            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]
            n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time']   = _decode_event_times(data['step_time'],   trl)
            data['reward_time'] = _decode_event_times(data['reward_time'], trl)

            # ---- Pair selection (all non-targets, no outlier split) ----
            # Optionally precompute the per-cell mask of cells that are
            # responsive targets in some other group (I→I-only restriction).
            n_cells_sess = AMP[0].shape[0]
            is_responsive_target = np.zeros(n_cells_sess, dtype=bool)
            if RESTRICT_TO_RESPONSIVE_TARGETS:
                for _gi in range(stimDist.shape[1]):
                    _cl = np.where(
                        (stimDist[:, _gi] < 10) &
                        (AMP[0][:, _gi] > TARGET_AMP_THR) &
                        (AMP[1][:, _gi] > TARGET_AMP_THR)
                    )[0]
                    is_responsive_target[_cl] = True

            dw_list = []
            pair_cl_list = []
            pair_nt_list = []
            dtarget_list = []

            for gi in range(stimDist.shape[1]):
                cl = np.where(
                    (stimDist[:, gi] < 10) &
                    (AMP[0][:, gi] > TARGET_AMP_THR) &
                    (AMP[1][:, gi] > TARGET_AMP_THR)
                )[0]
                if cl.size == 0:
                    continue
                nontarg = np.where(
                    (stimDist[:, gi] > NONTARG_DIST_LO) &
                    (stimDist[:, gi] < NONTARG_DIST_HI)
                )[0]
                if RESTRICT_TO_RESPONSIVE_TARGETS:
                    nontarg = nontarg[is_responsive_target[nontarg]]
                if nontarg.size == 0:
                    continue
                dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)
                dt = np.mean(AMP[1][cl, gi] - AMP[0][cl, gi])
                dtarget_list.append(np.full(len(nontarg), dt))

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            dTarget = np.nan_to_num(np.concatenate(dtarget_list), nan=0.0)
            n_pairs = len(Y)

            cl_weights = np.zeros((n_pairs, n_neurons))
            offset = 0
            for gi_idx in range(len(dw_list)):
                n_nt = len(dw_list[gi_idx])
                cl_arr = pair_cl_list[gi_idx]
                for qi in range(n_nt):
                    cl_neurons = cl_arr[qi]
                    cl_weights[offset + qi, cl_neurons] = 1.0 / len(cl_neurons)
                offset += n_nt

            # ---- Pre-epoch activity ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            epoch_act = np.nanmean(
                F_nan[ts_pre[0]:ts_pre[-1] + 1, :, :], axis=0)

            bl_trials = np.arange(min(N_BASELINE, trl))
            bl_mean = np.nanmean(epoch_act[:, bl_trials], axis=1)

            # ---- Sliding window CC ----
            r_pre = cl_weights @ epoch_act
            if CC_MODE == 'dev2':
                r_post = epoch_act[all_nt, :] - bl_mean[all_nt, np.newaxis]
                cc_trial = r_pre * r_post
            elif CC_MODE == 'dot_prod':
                r_post = epoch_act[all_nt, :]
                cc_trial = r_pre * r_post
            elif CC_MODE == 'target_only':
                cc_trial = r_pre
            elif CC_MODE == 'nontarget_only':
                cc_trial = epoch_act[all_nt, :] - bl_mean[all_nt, np.newaxis]
            else:
                raise ValueError(
                    f"CC_MODE must be 'dev2', 'dot_prod', 'target_only', or "
                    f"'nontarget_only', got {CC_MODE!r}"
                )

            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)
            if n_wins < 3:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            X = np.zeros((n_pairs, n_wins))
            for wi, ws in enumerate(win_starts):
                X[:, wi] = np.sum(cc_trial[:, ws:ws + WIN_SIZE], axis=1)

            mu_x = X.mean(axis=0)
            sig_x = X.std(axis=0)
            sig_x[sig_x == 0] = 1.0
            X = (X - mu_x) / sig_x

            if CONTROL_DTARGET:
                # Append standardized dTarget as one extra feature column.
                # Lets the linear model use dTarget to absorb the per-group
                # "target driven harder post" confound; CC features then
                # capture only the part of dW unexplained by dTarget.
                dt_mu, dt_sig = dTarget.mean(), dTarget.std()
                if dt_sig == 0 or not np.isfinite(dt_sig):
                    dt_sig = 1.0
                dT_z = ((dTarget - dt_mu) / dt_sig).reshape(-1, 1)
                X = np.column_stack([X, dT_z])

            # ---- Cross-validated fitting ----
            cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            corr_test_folds  = []
            corr_train_folds = []
            p_test_folds     = []
            Y_test_all       = []
            Y_pred_all       = []
            beta_first       = None

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

                mu_y, sig_y = Y_train.mean(), Y_train.std()
                if sig_y == 0 or not np.isfinite(sig_y):
                    sig_y = 1.0
                Y_train_z = (Y_train - mu_y) / sig_y
                Y_test_z  = (Y_test  - mu_y) / sig_y

                if fit_type == 'pinv':
                    beta = np.linalg.pinv(X_train) @ Y_train_z
                elif fit_type == 'ridge':
                    from sklearn.linear_model import RidgeCV
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import Pipeline
                    ridge = RidgeCV(alphas=np.logspace(-10, -4, 10),
                                    fit_intercept=True)
                    pipe = Pipeline([('scaler', StandardScaler()),
                                     ('ridge', ridge)])
                    pipe.fit(X_train, Y_train_z)
                    beta = pipe.named_steps['ridge'].coef_

                Y_train_pred = X_train @ beta
                Y_test_pred  = X_test @ beta

                r_tr = pearsonr(Y_train_pred, Y_train_z)[0] \
                    if np.std(Y_train_pred) > 0 else 0.0
                if np.std(Y_test_pred) > 0:
                    r_te, p_te = pearsonr(Y_test_pred, Y_test_z)
                else:
                    r_te, p_te = 0.0, 1.0

                corr_train_folds.append(r_tr)
                corr_test_folds.append(r_te)
                p_test_folds.append(p_te)
                Y_test_all.append(Y_test_z)
                Y_pred_all.append(Y_test_pred)
                if fold_idx == 0:
                    beta_first = beta.copy()

            r_test_mean  = np.mean(corr_test_folds)
            r_train_mean = np.mean(corr_train_folds)
            p_test_combined = np.exp(np.mean(np.log(
                np.clip(p_test_folds, 1e-300, 1.0))))

            all_results.append({
                'mouse': mouse, 'session': session,
                'n_pairs': n_pairs, 'n_trials': trl, 'n_windows': n_wins,
                'r_test': r_test_mean, 'r_train': r_train_mean,
                'p_test': p_test_combined,
                'r_test_folds': corr_test_folds,
                'betas': beta_first,
                'Y_test_all': np.concatenate(Y_test_all),
                'Y_pred_all': np.concatenate(Y_pred_all),
            })
            sig = '*' if p_test_combined < 0.05 else ''
            print(f"  {n_pairs} pairs, {n_wins} wins | "
                  f"train r={r_train_mean:.3f}, test r={r_test_mean:.3f}, "
                  f"p={p_test_combined:.4f} {sig}")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(all_results)} sessions")


#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'three_factor_ve_inhibitory.npy'),
        {'all': all_results}, allow_pickle=True)
print("Saved.")


#%% ============================================================================
# CELL 5: Load
# ============================================================================
_saved = np.load(
    os.path.join(RESULTS_DIR, 'three_factor_ve_inhibitory.npy'),
    allow_pickle=True).item()
all_results = _saved['all']
print(f"Loaded {len(all_results)} sessions")


#%% ============================================================================
# CELL 6: Summary figure (single group: all non-targets)
# ============================================================================
n_s = len(all_results)
if n_s == 0:
    print("No sessions to plot.")
else:
    r_test  = np.array([s['r_test']  for s in all_results])
    r_train = np.array([s['r_train'] for s in all_results])
    p_test  = np.array([s['p_test']  for s in all_results])
    mice_arr = np.array([s['mouse'] for s in all_results])
    n_sig = int(np.sum(p_test < 0.05))
    color = '#c0392b'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel A: Test r per session (color by mouse) ---
    ax = axes[0]
    mouse_list = sorted(set(mice_arr))
    cmap = plt.cm.Set2
    mouse_colors = {m: cmap(i) for i, m in enumerate(mouse_list)}

    for i in range(n_s):
        ec = 'k' if p_test[i] < 0.05 else 'none'
        lw = 1.5 if p_test[i] < 0.05 else 0
        ax.bar(i, r_test[i], color=mouse_colors[mice_arr[i]], alpha=0.7,
               edgecolor=ec, linewidth=lw)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.axhline(np.median(r_test), color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Session')
    ax.set_ylabel('Test r (5-fold CV)')
    ax.set_title(f'{n_sig}/{n_s} p<0.05', fontweight='bold')
    for m in mouse_list:
        ax.scatter([], [], color=mouse_colors[m], label=m, s=40)
    ax.legend(loc='best', fontsize=8, ncol=2)

    # --- Panel B: Pooled binned prediction ---
    ax = axes[1]
    Y_pred_pool = np.concatenate([s['Y_pred_all'] for s in all_results])
    Y_test_pool = np.concatenate([s['Y_test_all'] for s in all_results])

    n_bins_plot = 5
    edges = np.percentile(Y_pred_pool, np.linspace(0, 100, n_bins_plot + 1))
    bx, by, be = [], [], []
    for bi in range(n_bins_plot):
        if bi < n_bins_plot - 1:
            mask = (Y_pred_pool >= edges[bi]) & (Y_pred_pool < edges[bi + 1])
        else:
            mask = (Y_pred_pool >= edges[bi]) & (Y_pred_pool <= edges[bi + 1])
        if np.sum(mask) < 3:
            continue
        bx.append(np.mean(Y_pred_pool[mask]))
        by.append(np.mean(Y_test_pool[mask]))
        be.append(np.std(Y_test_pool[mask]) / np.sqrt(np.sum(mask)))

    ax.errorbar(bx, by, yerr=be, fmt='o-', color=color, capsize=5,
                linewidth=2, markersize=7)
    ax.axhline(0, color='k', ls='-', alpha=0.2)
    r_pool, p_pool = pearsonr(Y_pred_pool, Y_test_pool)
    ax.set_xlabel('Predicted dW (z)')
    ax.set_ylabel('Actual dW (z)')
    ax.set_title(f'pooled: r={r_pool:.3f}, p={p_pool:.2e}',
                 fontweight='bold')

    # --- Panel C: Train vs test r ---
    ax = axes[2]
    ax.scatter(r_train, r_test, c=color, s=30, alpha=0.6, zorder=3)
    lim = max(np.max(np.abs(r_train)), np.max(np.abs(r_test))) * 1.1
    lim = max(lim, 0.1)
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlabel('Train r')
    ax.set_ylabel('Test r')
    ax.set_title('overfitting check', fontweight='bold')

    try:
        _, p_wilcox = wilcoxon(r_test)
    except Exception:
        p_wilcox = 1.0
    print(f"All non-targets: median r_test={np.median(r_test):.4f}, "
          f"mean={np.mean(r_test):.4f}, Wilcoxon p={p_wilcox:.4f}, "
          f"sig={n_sig}/{n_s}")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_variance_explained_inhibitory.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved.")
