#%% ============================================================================
# population_cn_activity.py
#
# Two quantities of interest, computed for the CN and for the POPULATION
# (both from F = Suite2P), as a function of absolute trial number:
#
#   trial tuning       = mean F over [trial start, threshold crossing]
#   late-trial tuning  = mean F over [reward - LATE_WIN_S, reward]
#                        (= pre-reward AUC; called "late trial tuning" in
#                         prior work and matches session_metrics' version)
#
# For each quantity, CELL 2 makes three takes on CN vs population over trials:
#   {tag}_ztrace       - the picture (CN vs pop, session-z)
#   {tag}_cn_minus_pop - the divergence on one line
#   {tag}_slope_strip  - the stat (per-session CN vs pop slopes)
#
# Run CELL 0 -> CELL 1 (slow network load) -> CELL 2.  Saves popcn_*.png/.svg.
# ============================================================================

#%% ============================================================================
# CELL 0: setup
# ============================================================================
import os, sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
from BCI_data_helpers import parse_hdf5_array_string

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
mpl.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'svg.fonttype': 'none',
})
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')
os.makedirs(PANEL_DIR, exist_ok=True)

# ---- config -----------------------------------------------------------------
PRE_TRIAL_S    = 2.0     # F[0] is at -PRE_TRIAL_S relative to trial start
LATE_WIN_S     = 2.0     # pre-reward window for late-trial tuning
POP_EXCLUDE_CN = True    # exclude CN from the population mean
MAX_CN_DIST_UM = 5.0     # drop sessions whose CN ROI is farther than this
N_TRIALS       = 150
EARLY          = (0, 10) # baseline trials (defines zero); (5,10) skips onset
SHOW_TR        = 100     # trials on the x-axis
MIN_SESS       = 15      # hide trials with fewer than this many sessions
SLOPE_HI       = 80      # per-session slope is fit over trials [0, SLOPE_HI)
SMOOTH_W       = 5       # light NaN-aware smoothing for the traces
NORMALIZE      = 'session_z'  # 'session_z' (per-session z) | 'raw' (dF/F)

mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
_qc_fail = {('BCI104', '012325'), ('BCI105', '012125'), ('BCI105', '012425')}

def save_panel(name, fig=None):
    fig = plt.gcf() if fig is None else fig
    fig.savefig(os.path.join(PANEL_DIR, f'popcn_{name}.png'), dpi=300)
    fig.savefig(os.path.join(PANEL_DIR, f'popcn_{name}.svg'))

def panel_fig(ax_w=1.25, ax_h=1.25, left=0.7, bottom=0.5, right=0.2, top=0.3):
    fig_w, fig_h = left + ax_w + right, bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h])
    return fig, ax

def _smooth(y, w):
    yv = np.asarray(y, float)
    if w <= 1:
        return yv
    k = np.ones(w) / w
    valid = np.isfinite(yv).astype(float)
    y0 = np.where(np.isfinite(yv), yv, 0.0)
    num = np.convolve(y0, k, mode='same')
    den = np.convolve(valid, k, mode='same')
    return np.where(den > 0, num / den, np.nan)

def _msem_curve(curves, smooth_w=1):
    C = np.vstack([_smooth(curves[u], smooth_w) for u in range(curves.shape[0])])
    with np.errstate(invalid='ignore'):
        n = np.sum(np.isfinite(C), axis=0)
        mean = np.nanmean(C, axis=0)
        sem = np.nanstd(C, axis=0) / np.sqrt(np.clip(n, 1, None))
    return mean, mean - sem, mean + sem, n

def _stars(p):
    return ('***' if p < 1e-3 else '**' if p < 1e-2 else
            '*' if p < 0.05 else 'n.s.')

def _zcols(MAT):
    """Per-session z-score each column over its finite trials."""
    Z = np.full(MAT.shape, np.nan)
    for j in range(MAT.shape[1]):
        x = MAT[:, j].astype(float); sd = np.nanstd(x)
        if np.isfinite(sd) and sd > 0:
            Z[:, j] = (x - np.nanmean(x)) / sd
    return Z

def _norm(MAT):
    return MAT.astype(float) if NORMALIZE == 'raw' else _zcols(MAT)

def _bsub_curves(Z):
    """Baseline-subtract the EARLY window, truncate to SHOW_TR. (n_sess, SHOW_TR)"""
    out = np.full((Z.shape[1], SHOW_TR), np.nan)
    for j in range(Z.shape[1]):
        x = Z[:, j].copy()
        b = np.nanmean(x[EARLY[0]:EARLY[1]])
        if np.isfinite(b):
            x = x - b
        out[j] = x[:SHOW_TR]
    return out

def _paired_slopes(CNz, POPz):
    """Per-session slope (z/trial) over [0, SLOPE_HI) for CN and pop, paired."""
    xs = np.arange(SLOPE_HI); sC, sP = [], []
    for j in range(CNz.shape[1]):
        xc = CNz[:SLOPE_HI, j]; xp = POPz[:SLOPE_HI, j]
        oc = np.isfinite(xc); op = np.isfinite(xp)
        if oc.sum() >= 20 and op.sum() >= 20:
            sC.append(np.polyfit(xs[oc], xc[oc], 1)[0])
            sP.append(np.polyfit(xs[op], xp[op], 1)[0])
    return np.array(sC), np.array(sP)

#%% ============================================================================
# CELL 1: per-session loop -> CN/pop trial- and late-tuning (N_TRIALS x n_sess)
# ============================================================================
list_of_dirs = session_counting.counter()

CN_trial, CN_late, POP_trial, POP_late = [], [], [], []
dt_list, keys, dist_by_session, load_errors = [], [], {}, []

for mouse in mice:
    inds = np.where((list_of_dirs['Mouse'] == mouse) &
                    (list_of_dirs['Has data_main.npy'] == True))[0]
    for si in inds:
        session = list_of_dirs['Session'][si]
        if (mouse, session) in _qc_fail:
            continue
        folder = ('//allen/aind/scratch/BCI/2p-raw/'
                  + mouse + '/' + session + '/pophys/')
        try:
            data = ddct.load_hdf5(folder,
                ['F', 'conditioned_neuron', 'dist', 'dt_si',
                 'threshold_crossing_time', 'reward_time'], [])
            F = data['F']                          # (frames, neurons, trials)
            n_fr, n_neu, n_tr = F.shape
            cn = int(np.asarray(data['conditioned_neuron']).ravel()[0])
            dt = float(np.asarray(data['dt_si']).ravel()[0])
        except Exception as e:
            load_errors.append(((mouse, session), str(e)))
            continue

        # crossing & reward times per trial (s from trial start; NaN for miss)
        def _parse(field):
            p = parse_hdf5_array_string(data[field], n_tr)
            return np.array([x[0] if hasattr(x, '__len__') and len(x) > 0
                             else np.nan for x in p], dtype=float)
        tc_v = _parse('threshold_crossing_time')
        rw_v = _parse('reward_time')

        # CN-to-target distance QC (um); dist is 1D indexed by ROI
        try:
            dist_arr = np.asarray(data['dist']).ravel()
            dist_cn = float(dist_arr[cn]) if cn < len(dist_arr) else np.nan
        except (KeyError, IndexError, TypeError, ValueError):
            dist_cn = np.nan
        dist_by_session[(mouse, session)] = dist_cn
        if np.isfinite(dist_cn) and dist_cn > MAX_CN_DIST_UM:
            continue

        pre_f = int(PRE_TRIAL_S / dt)
        late_n = int(LATE_WIN_S / dt)
        others = ([j for j in range(n_neu) if j != cn] if POP_EXCLUDE_CN
                  else list(range(n_neu)))
        ct = np.full(n_tr, np.nan); cl = np.full(n_tr, np.nan)
        pt = np.full(n_tr, np.nan); pl = np.full(n_tr, np.nan)
        for t in range(n_tr):
            # trial tuning: trial start -> threshold crossing
            if np.isfinite(tc_v[t]):
                tcf = min(int(tc_v[t] / dt) + pre_f, n_fr)
                if tcf > pre_f:
                    m = np.nanmean(F[pre_f:tcf, :, t], axis=0)
                    ct[t] = m[cn]; pt[t] = np.nanmean(m[others])
            # late-trial tuning: 2 s before reward (pre-reward AUC)
            if np.isfinite(rw_v[t]):
                rwf = min(int(rw_v[t] / dt) + pre_f, n_fr)
                lo = max(pre_f, rwf - late_n)
                if rwf > lo:
                    m = np.nanmean(F[lo:rwf, :, t], axis=0)
                    cl[t] = m[cn]; pl[t] = np.nanmean(m[others])

        def _pad(v):
            out = np.full(N_TRIALS, np.nan)
            out[:min(n_tr, N_TRIALS)] = v[:N_TRIALS]
            return out
        CN_trial.append(_pad(ct)); CN_late.append(_pad(cl))
        POP_trial.append(_pad(pt)); POP_late.append(_pad(pl))
        dt_list.append(dt); keys.append((mouse, session))

CN_trial = np.column_stack(CN_trial); CN_late = np.column_stack(CN_late)
POP_trial = np.column_stack(POP_trial); POP_late = np.column_stack(POP_late)
print(f"Loaded {len(keys)} sessions ({len(load_errors)} load errors); "
      f"POP_EXCLUDE_CN={POP_EXCLUDE_CN}, MAX_CN_DIST_UM={MAX_CN_DIST_UM}")

#%% ============================================================================
# CELL 2: CN vs population over trials, for both quantities
# ============================================================================
def over_trials(CN_mat, POP_mat, tag):
    CNz, POPz = _norm(CN_mat), _norm(POP_mat)
    x = np.arange(SHOW_TR)
    sC, sP = _paired_slopes(CNz, POPz)
    pC = wilcoxon(sC).pvalue if len(sC) > 1 else np.nan
    pP = wilcoxon(sP).pvalue if len(sP) > 1 else np.nan
    pD = wilcoxon(sC - sP).pvalue if len(sC) > 1 else np.nan

    mC, loC, hiC, nC = _msem_curve(_bsub_curves(CNz), SMOOTH_W)
    mP, loP, hiP, _ = _msem_curve(_bsub_curves(POPz), SMOOTH_W)
    keep = nC >= MIN_SESS                       # hide low-n tail
    for a in (mC, loC, hiC, mP, loP, hiP):
        a[~keep] = np.nan

    # (1) two-trace picture
    fig, ax = panel_fig()
    ax.fill_between(x, loP, hiP, color='0.6', alpha=0.30, lw=0)
    ax.plot(x, mP, color='0.45', lw=1.2, label='pop')
    ax.fill_between(x, loC, hiC, color='crimson', alpha=0.25, lw=0)
    ax.plot(x, mC, color='crimson', lw=1.2, label='CN')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Trial'); ax.set_ylabel(f'{tag} tuning ({NORMALIZE})')
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_panel(f'{tag}_ztrace')

    # (2) CN - pop difference
    mD, loD, hiD, _ = _msem_curve(_bsub_curves(CNz - POPz), SMOOTH_W)
    for a in (mD, loD, hiD):
        a[~keep] = np.nan
    fig, ax = panel_fig()
    ax.fill_between(x, loD, hiD, color='purple', alpha=0.20, lw=0)
    ax.plot(x, mD, color='purple', lw=1.4)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Trial'); ax.set_ylabel(f'{tag}: CN - pop ({NORMALIZE})')
    ax.set_title(f'CN vs pop slope {_stars(pD)} (n={len(sC)})', fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_panel(f'{tag}_cn_minus_pop')

    # (3) per-session slope strip
    fig, ax = panel_fig(1.2, 1.25)
    for i, (s, col) in enumerate([(sC, 'crimson'), (sP, '0.45')]):
        ax.scatter(np.random.uniform(i - 0.15, i + 0.15, len(s)), s,
                   s=8, c='k', alpha=0.4, edgecolors='none')
        ax.plot([i - 0.25, i + 0.25], [np.median(s)] * 2, color=col, lw=2)
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['CN', 'pop'])
    ax.set_ylabel(f'{tag} slope ({NORMALIZE}/trial)')
    ax.set_title(f'CN {_stars(pC)} pop {_stars(pP)} diff {_stars(pD)}', fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_panel(f'{tag}_slope_strip')

    print(f"{tag}: CN slope med={np.median(sC):+.4f} p={pC:.3f}; "
          f"pop med={np.median(sP):+.4f} p={pP:.3f}; "
          f"CN-pop p={pD:.3f}; n={len(sC)} sessions")

over_trials(CN_trial, POP_trial, 'trial')
over_trials(CN_late,  POP_late,  'late')
