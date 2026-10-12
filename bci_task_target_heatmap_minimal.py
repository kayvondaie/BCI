"""
Minimal script for sharing:
  1. Heatmaps (one example session) of photostim-identified target cells and
     clean non-target cells during the BCI task, sorted Harvey-style by peak
     activity time across trial-start- and reward-aligned epochs.
  2. Pie charts of how cells distribute across four task epochs (pre-trial,
     early trial, late trial, reward), aggregated across ALL sessions that
     pass the QC CSV filter (Type=Inhibitory, pre OR post = Good/Ok, and no
     non-zero offset notes).

Two output figures, both saved to SAVE_DIR.
"""

import sys
import os
import csv
import re
import glob
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

import data_dict_create_module_test as ddct
import bci_time_series as bts
from BCI_data_helpers import (
    compute_amp_from_photostim_artifact_free,
    parse_hdf5_array_string,
    get_reward_aligned_df_truncated,
    get_trial_aligned_df_padded,
)


# -----------------------------
# Parameters
# -----------------------------
DATA_ROOT = r'//allen/aind/scratch/BCI/2p-raw'
QC_CSV    = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI\Inhibitory & Pan-neuronal BCI summary - Sheet1.csv'
SAVE_DIR  = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(SAVE_DIR, exist_ok=True)

EXAMPLE_MOUSE   = 'BCI116'
EXAMPLE_SESSION = '012826'

REWARD_WINDOW     = (-4, 10)
TRIAL_WINDOW      = (-2, 4)
MEDFILT_KERNEL    = 11
TARGET_AMP_THRESH = 0.05    # ΔF/F threshold for "responsive target"
NONTARG_DIST      = 30      # µm; cells must be > this from every stim site
RTA_DISPLAY_MAX   = 4.0     # clip reward panel display to <= this many s post-reward
VMAX              = 0.3     # heatmap color saturation (ΔF/F)

EPOCH_DEFS = [
    # label, alignment, t_lo, t_hi, color
    ('Pre-trial\n(-2 to -1 s)',  'sta', -2.0, -1.0, '#33b983'),  # teal
    ('Early trial\n(0 to 1 s)',  'sta',  0.0,  1.0, '#1077f3'),  # blue
    ('Late trial\n(-1 to 0 s)',  'rta', -1.0,  0.0, '#0050ae'),  # dark blue
    ('Reward\n(0 to 2 s)',       'rta',  0.0,  2.0, '#bf8cfc'),  # lavender
]


# -----------------------------
# Helpers
# -----------------------------
def _load_stim_params_from_h5(folder, ps_epoch):
    """stim_params is an h5 Group of sub-datasets (incl. scalar `total_duration`)
    that ddct.load_hdf5 can't read. Pull it directly via h5py."""
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
    """step_time / reward_time come out of ddct.load_hdf5 as a 0-d ndarray of
    pickled bytes. Decode and flatten each (1, n_events) per-trial entry."""
    raw = v.item() if hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0 else v
    if isinstance(raw, (bytes, np.bytes_)):
        try:
            obj = pickle.loads(raw)
            return [np.asarray(x).ravel() for x in obj]
        except Exception:
            pass
    return parse_hdf5_array_string(v, trl)


def _row_smooth(arr, k=MEDFILT_KERNEL):
    return np.array([medfilt(r, k) for r in arr])


def _epoch_of_peak(sta_arr, rta_arr, t_trial, t_reward_disp):
    """For each row, pick the epoch (out of EPOCH_DEFS) that contains the
    row's maximum value. Returns (n_cells,) int in [0 .. len(EPOCH_DEFS)-1]."""
    cols = []
    for _, src, t_lo, t_hi, _c in EPOCH_DEFS:
        if src == 'sta':
            mask = (t_trial >= t_lo) & (t_trial <= t_hi)
            arr = sta_arr
        else:
            mask = (t_reward_disp >= t_lo) & (t_reward_disp <= t_hi)
            arr = rta_arr
        if not mask.any():
            cols.append(np.full(arr.shape[0], -np.inf))
            continue
        cols.append(np.nanmax(arr[:, mask], axis=1))
    return np.argmax(np.column_stack(cols), axis=1)


# -----------------------------
# QC filter — pull session list from the inhibitory-photostim summary CSV
# -----------------------------
_OK_QUALITY    = {'good', 'ok'}
_BAD_OFFSET_RE = re.compile(r'offset\s*=\s*[1-9]', re.IGNORECASE)

def load_qc_sessions():
    """Return sorted list of (mouse, session) tuples that pass QC:
       Type == 'Inhibitory'
       Pre  ∈ {'Good','Ok'}  OR  Post ∈ {'Good','Ok'}
       Notes do NOT contain 'offset = N' with N >= 1
    """
    keep = set()
    cur_mouse = cur_type = None
    with open(QC_CSV, encoding='utf-8') as f:
        rows = list(csv.reader(f))
    for row in rows[2:]:                                       # skip 2 header rows
        if not row:
            continue
        subj = row[0].strip() if len(row) > 0 else ''
        type_= row[1].strip() if len(row) > 1 else ''
        if subj:
            cur_mouse = subj
        if type_:
            cur_type = type_
        if (cur_type or '').lower() != 'inhibitory':
            continue
        date  = row[2].strip()        if len(row) > 2 else ''
        pre   = row[4].strip().lower() if len(row) > 4 else ''
        post  = row[5].strip().lower() if len(row) > 5 else ''
        notes = row[6]                 if len(row) > 6 else ''
        if (not date or cur_mouse is None or
            (pre not in _OK_QUALITY and post not in _OK_QUALITY)):
            continue
        if _BAD_OFFSET_RE.search(notes or ''):
            continue
        try:
            m, d, y = date.split('/')
            session = f'{int(m):02d}{int(d):02d}{int(y):02d}'
        except Exception:
            continue
        keep.add((cur_mouse, session))
    return sorted(keep)


# -----------------------------
# Per-session pipeline
# -----------------------------
def process_session(mouse, session):
    """Load one session, identify target / clean non-target cells, build the
    smoothed sta / rta arrays per group. Returns dict with everything needed
    for both the heatmap and the epoch-of-peak classification."""
    folder = os.path.join(DATA_ROOT, mouse, session, 'pophys') + '/'

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
    F     = data['F']
    trl   = F.shape[2]

    AMP, stimDist = compute_amp_from_photostim_artifact_free(mouse, data, folder)
    amp_ep = AMP[-1]
    n_cells, n_groups = amp_ep.shape

    target_cell_idx = np.argmin(stimDist, axis=0)
    target_amp = amp_ep[target_cell_idx, np.arange(n_groups)]
    is_target = np.zeros(n_cells, dtype=bool)
    for gi in range(n_groups):
        if np.isfinite(target_amp[gi]) and target_amp[gi] > TARGET_AMP_THRESH:
            is_target[target_cell_idx[gi]] = True
    is_clean_nontarget = np.nanmin(stimDist, axis=1) > NONTARG_DIST

    data['step_time']   = _decode_event_times(data['step_time'],   trl)
    data['reward_time'] = _decode_event_times(data['reward_time'], trl)
    rt = np.array([x[0] if x.size > 0 else np.nan
                   for x in data['reward_time']], dtype=float)
    rt_filled = rt.copy()
    rt_filled[~np.isfinite(rt_filled)] = 30.0

    step_v, reward_v, trial_start_v = bts.bci_time_series_fun(
        folder, data, rt_filled, dt_si)
    df_cl = data['df_closedloop']
    rta, t_reward = get_reward_aligned_df_truncated(
        df_cl, reward_v, trial_start_v, dt_si, window=REWARD_WINDOW)
    sta, t_trial = get_trial_aligned_df_padded(
        df_cl, trial_start_v, reward_v, dt_si, window=TRIAL_WINDOW)

    sta_base_mask = (t_trial >= TRIAL_WINDOW[0]) & (t_trial < TRIAL_WINDOW[0] + 1.0)
    rta_mask      = t_reward <= RTA_DISPLAY_MAX
    t_reward_disp = t_reward[rta_mask]

    def _build(cell_idx):
        if len(cell_idx) == 0:
            n_t = sta.shape[0]; n_r = int(rta_mask.sum())
            return (np.zeros((0, n_t)), np.zeros((0, n_r)))
        sta_m = np.nanmean(sta[:, cell_idx, :], axis=2).T
        rta_m = np.nanmean(rta[:, cell_idx, :], axis=2).T
        if sta_base_mask.any():
            bl = np.nanmean(sta_m[:, sta_base_mask], axis=1, keepdims=True)
            sta_m -= bl
            rta_m -= bl
        rta_m = rta_m[:, rta_mask]
        return _row_smooth(sta_m), _row_smooth(rta_m)

    target_idx_list  = np.where(is_target)[0]
    nontarg_idx_list = np.where(is_clean_nontarget)[0]
    sta_t, rta_t = _build(target_idx_list)
    sta_n, rta_n = _build(nontarg_idx_list)

    return dict(
        sta_t=sta_t, rta_t=rta_t, n_t=len(target_idx_list),
        sta_n=sta_n, rta_n=rta_n, n_n=len(nontarg_idx_list),
        t_trial=t_trial, t_reward_disp=t_reward_disp,
    )


# -----------------------------
# 1) Example-session heatmap
# -----------------------------
S = process_session(EXAMPLE_MOUSE, EXAMPLE_SESSION)
print(f'{EXAMPLE_MOUSE} {EXAMPLE_SESSION}: '
      f'{S["n_t"]} target, {S["n_n"]} clean non-target cells')

sort_t = np.argsort(np.nanargmax(np.concatenate([S['sta_t'], S['rta_t']], axis=1), axis=1))
sort_n = np.argsort(np.nanargmax(np.concatenate([S['sta_n'], S['rta_n']], axis=1), axis=1))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
panels = [
    (axes[0, 0], S['sta_t'][sort_t], S['t_trial'],       'Time from trial start (s)',
     f'Target cell (n={S["n_t"]})\nsorted by combined peak time', 'Trial-start aligned'),
    (axes[0, 1], S['rta_t'][sort_t], S['t_reward_disp'], 'Time from reward (s)',
     None, 'Reward aligned'),
    (axes[1, 0], S['sta_n'][sort_n], S['t_trial'],       'Time from trial start (s)',
     f'Non-target cell (n={S["n_n"]})\nsorted by combined peak time', None),
    (axes[1, 1], S['rta_n'][sort_n], S['t_reward_disp'], 'Time from reward (s)',
     None, None),
]
for ax, M, t_axis, xlabel, ylabel, title in panels:
    im = ax.imshow(M, aspect='auto', cmap='bwr', vmin=-VMAX, vmax=VMAX,
                   extent=[t_axis[0], t_axis[-1], M.shape[0], 0])
    ax.axvline(0, color='k', lw=0.8, alpha=0.7)
    ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    plt.colorbar(im, ax=ax, label='ΔF/F')
plt.suptitle(f'{EXAMPLE_MOUSE} {EXAMPLE_SESSION}: target vs non-target cell heatmaps',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_target_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.show()


# -----------------------------
# 2) Multi-session epoch-of-peak pie charts
# -----------------------------
qc_sessions = load_qc_sessions()
print(f'\nQC CSV: {len(qc_sessions)} sessions to aggregate for pie chart')

all_epoch_t = []
all_epoch_n = []
for mouse_i, session_i in qc_sessions:
    try:
        Si = process_session(mouse_i, session_i)
    except Exception as e:
        print(f'  Skipping {mouse_i} {session_i}: {type(e).__name__}: {e}')
        continue
    if Si['n_t'] > 0:
        all_epoch_t.extend(_epoch_of_peak(Si['sta_t'], Si['rta_t'],
                                          Si['t_trial'], Si['t_reward_disp']).tolist())
    if Si['n_n'] > 0:
        all_epoch_n.extend(_epoch_of_peak(Si['sta_n'], Si['rta_n'],
                                          Si['t_trial'], Si['t_reward_disp']).tolist())
    print(f'  {mouse_i} {session_i}: targets={Si["n_t"]}, non-targets={Si["n_n"]}')

all_epoch_t = np.array(all_epoch_t)
all_epoch_n = np.array(all_epoch_n)
counts_t = np.bincount(all_epoch_t, minlength=len(EPOCH_DEFS))
counts_n = np.bincount(all_epoch_n, minlength=len(EPOCH_DEFS))

labels = [d[0] for d in EPOCH_DEFS]
colors = [d[4] for d in EPOCH_DEFS]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, counts, title in [
    (axes[0], counts_t, f'Target cells (n={counts_t.sum()})'),
    (axes[1], counts_n, f'Non-target cells (n={counts_n.sum()})'),
]:
    ax.pie(
        counts, labels=labels, colors=colors,
        autopct=lambda p, c=counts: f'{int(round(p * c.sum() / 100))}\n({p:.0f}%)',
        startangle=90, counterclock=False,
        textprops={'fontsize': 9},
    )
    ax.set_title(title, fontsize=12)
plt.suptitle(
    f'Epoch of peak activity — aggregated across {len(qc_sessions)} QC sessions',
    fontsize=13,
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'bci_task_target_epoch_pie.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f'\nTarget counts:     {dict(zip([l.replace(chr(10), " ") for l in labels], counts_t.tolist()))}')
print(f'Non-target counts: {dict(zip([l.replace(chr(10), " ") for l in labels], counts_n.tolist()))}')
