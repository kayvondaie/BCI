#%% ============================================================================
# CELL 1: Imports and helpers
# ============================================================================
"""
Diagnostic for the (null) data eligibility fingerprint.

Question: the fingerprint's I block (~0.003) fails to reproduce the ROBUST HI
(coactivity->dW, Fig 3d, +0.11). The ONLY difference is the fine per-frame
RPE weighting w(t). So: is the weighting washing out the signal?

This script, per session, builds I/S_post/S_pre/M under a SWEEP of weightings
and reports standardized betas:
  - 'uniform'   w=1                      <- SANITY: I should recover the HI signal
  - 'step_rpe'  step - EMA(step)         <- current fingerprint weighting
  - 'step'      raw lickport position
  - 'reward'    reward_vector
  - 'step_vel'  |d/dt lickport|          <- movement speed
  - 'step_rpe1' step - EMA(step), tau=1s
  - 'step_rpe5' step - EMA(step), tau=5s
Read: does 'uniform' give nonzero I (pipeline OK) while RPE-weighted gives 0?
      does ANY weighting revive S_post? If none -> null is robust.
Mirrors fingerprint_eligibility_data.py loading/pair-selection exactly.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from scipy.signal import lfilter
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *
import bci_time_series as bts

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')


def ema_causal(x, tau_frames, axis=-1, init=None):
    x = np.asarray(x, dtype=float)
    alpha = 1.0 / float(tau_frames)
    b = np.array([alpha]); a = np.array([1.0, -(1.0 - alpha)])
    x_first = np.take(x, 0, axis=axis)
    init_arr = x_first if init is None else np.broadcast_to(np.asarray(init, float), x_first.shape)
    zi = np.expand_dims(init_arr - alpha * x_first, axis=axis)
    y, _ = lfilter(b, a, x, axis=axis, zi=zi)
    return y


print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

TAU_BL = 360.0          # seconds; EMA baseline for deviation dev = df - h_bar
PRE_LAG_FRAMES = 1
N_SHUF = 0              # FAST first pass: 0 = no shuffles (true betas only, ~6x faster).
                        # Set to 10-20 later to get a null for whichever weighting looks nonzero.
MICE_SUBSET = None      # e.g. ["BCI102"] for a quick look at the strongest mouse first; None = all
BLOCKS = ['I', 'S_post', 'S_pre', 'M']
WEIGHTINGS = ['uniform', 'step_rpe', 'step', 'reward', 'step_vel', 'step_rpe1', 'step_rpe5']

all_results = []   # list of dicts: {mouse, session, n_pairs, betas{w: beta4}, shuf{w: beta4}}
print("weightings:", WEIGHTINGS)


def _zc(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def _fit_blocks(I, S_post, S_pre, M, Y):
    X = np.column_stack([_zc(I), _zc(S_post), _zc(S_pre), _zc(M)])
    b, *_ = np.linalg.lstsq(X, _zc(Y), rcond=None)
    return b


#%% ============================================================================
# CELL 3: Main loop
# ============================================================================
import csv
_qc_csv = os.path.join(RESULTS_DIR, 'qc', 'qc_summary.csv')
_qc_fail = set()
if os.path.exists(_qc_csv):
    with open(_qc_csv) as _f:
        for _r in csv.DictReader(_f):
            if _r['pass_qc'] != 'True':
                _qc_fail.add((_r['mouse'], _r['session']))
    print(f"QC filter: {len(_qc_fail)} sessions excluded")
else:
    print("WARNING: qc_summary.csv not found, no sessions excluded")

_mice = mice if MICE_SUBSET is None else [m for m in mice if m in MICE_SUBSET]
for mi in range(len(_mice)):
    mouse = _mice[mi]
    session_inds = np.where((list_of_dirs['Mouse'] == mouse) &
                            (list_of_dirs['Has data_main.npy'] == True))[0]
    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            if (mouse, session) in _qc_fail:
                continue
            folder = (r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/')
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['df_closedloop', 'F', 'mouse', 'session', 'conditioned_neuron',
                        'dt_si', 'step_time', 'reward_time', 'BCI_thresholds']
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print("  file not found."); continue

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            dt_si = data['dt_si']; F = data['F']; trl = F.shape[2]; n_neurons = F.shape[1]
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']], float)
            rt_filled = rt.copy(); rt_filled[~np.isfinite(rt_filled)] = 30.0

            df_full = np.asarray(data['df_closedloop'], float); df_full[np.isnan(df_full)] = 0.0
            n_df, total_frames = df_full.shape
            if n_df != n_neurons:
                print("  neuron mismatch; skipping."); continue

            step_vector, reward_vector, trial_start_vector = (
                bts.bci_time_series_fun(folder, data, rt_filled, dt_si))
            tf = min(len(step_vector), total_frames)
            step_vector = np.asarray(step_vector[:tf], float)
            reward_vector = np.asarray(reward_vector[:tf], float)
            df_full = df_full[:, :tf]; total_frames = tf

            # ---- build weighting dictionary ----
            def _rpe(sv, tau_s):
                tf_ = tau_s / dt_si
                return sv - ema_causal(sv, tf_, init=float(sv.mean()))
            W = {}
            W['uniform'] = np.ones(total_frames, float)
            W['step_rpe'] = _rpe(step_vector, 2.0)
            W['step'] = step_vector.copy()
            W['reward'] = reward_vector.copy()
            sv_vel = np.zeros(total_frames, float)
            sv_vel[1:] = np.abs(np.diff(step_vector))
            W['step_vel'] = sv_vel
            W['step_rpe1'] = _rpe(step_vector, 1.0)
            W['step_rpe5'] = _rpe(step_vector, 5.0)

            # ---- deviation (EMA baseline) and mean activity ----
            tau_bl_frames = TAU_BL / dt_si
            dev = df_full - ema_causal(df_full, tau_bl_frames, axis=-1, init=df_full[:, 0])
            m_neuron = df_full.mean(axis=1)

            # ---- pair selection ----
            dw_list, pair_cl_list, pair_nt_list = [], [], []
            for gi in range(stimDist.shape[1]):
                cl = np.where((stimDist[:, gi] < 10) & (AMP[0][:, gi] > 0.1) & (AMP[1][:, gi] > 0.1))[0]
                if cl.size == 0:
                    continue
                nt = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))[0]
                if nt.size == 0:
                    continue
                dw_list.append(AMP[1][nt, gi] - AMP[0][nt, gi])
                pair_cl_list.append(cl); pair_nt_list.append(nt)
            if len(dw_list) == 0:
                print("  no pairs."); continue

            # precompute per-group pre traces / dpre (weighting-independent)
            pre_traces, dpre_traces, m_pre_list = [], [], []
            for gi in range(len(dw_list)):
                cl = pair_cl_list[gi]
                pre_g = df_full[cl, :].mean(axis=0)
                pre_lag = np.empty_like(pre_g)
                pre_lag[:PRE_LAG_FRAMES] = 0.0
                pre_lag[PRE_LAG_FRAMES:] = pre_g[:-PRE_LAG_FRAMES] if PRE_LAG_FRAMES > 0 else pre_g
                dpre = pre_lag - ema_causal(pre_lag, tau_bl_frames, init=pre_lag[0])
                pre_traces.append(pre_lag); dpre_traces.append(dpre); m_pre_list.append(pre_lag.mean())
            Y = np.nan_to_num(np.concatenate(dw_list))

            def build(w_use):
                q_neuron = (w_use[None, :] * dev).sum(axis=1)
                I_all, Sp_all, Sq_all, M_all = [], [], [], []
                for gi in range(len(dw_list)):
                    nt = pair_nt_list[gi]
                    dpre = dpre_traces[gi]; m_pre = m_pre_list[gi]
                    q_pre = float(np.sum(w_use * dpre))
                    wd = w_use * dpre
                    I_all.append(dev[nt, :] @ wd)
                    Sp_all.append(m_pre * q_neuron[nt])
                    Sq_all.append(m_neuron[nt] * q_pre)
                    M_all.append(m_pre * m_neuron[nt])
                return (np.concatenate(I_all), np.concatenate(Sp_all),
                        np.concatenate(Sq_all), np.concatenate(M_all))

            betas, shufs = {}, {}
            rng = np.random.default_rng(0)
            for wname in WEIGHTINGS:
                w = W[wname]
                betas[wname] = _fit_blocks(*build(w), Y)
                if wname == 'uniform' or N_SHUF == 0:
                    shufs[wname] = np.full(4, np.nan)
                else:
                    shufs[wname] = np.mean(
                        [_fit_blocks(*build(np.roll(w, rng.integers(total_frames // 10, total_frames))), Y)
                         for _ in range(N_SHUF)], axis=0)

            all_results.append({'mouse': mouse, 'session': session, 'n_pairs': len(Y),
                                'betas': betas, 'shufs': shufs})
            print("  {} pairs | I(unif)={:+.3f} I(rpe)={:+.3f} | S_post: ".format(
                len(Y), betas['uniform'][0], betas['step_rpe'][0])
                  + "  ".join("{}={:+.3f}".format(wn, betas[wn][1]) for wn in WEIGHTINGS))

        except Exception as e:
            print(f"  FAILED: {e}"); traceback.print_exc(); continue

print(f"\nTotal sessions: {len(all_results)}")
np.save(os.path.join(RESULTS_DIR, 'fingerprint_data_weighting_sweep.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 4: Aggregate table
# ============================================================================
all_results = np.load(os.path.join(RESULTS_DIR, 'fingerprint_data_weighting_sweep.npy'),
                      allow_pickle=True).tolist()


def _star(v):
    v = v[np.isfinite(v)]
    p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else 'ns'


n = len(all_results)
print(f"\nDATA fingerprint weighting sweep, n={n} sessions (standardized beta, mean+/-sem):")
print("  {:10s} {:>16s} {:>16s} {:>16s} {:>16s}".format('weighting', 'I', 'S_post', 'S_pre', 'S_post-S_pre'))
for wn in WEIGHTINGS:
    BI = np.array([s['betas'][wn][0] for s in all_results])
    BP = np.array([s['betas'][wn][1] for s in all_results])
    BQ = np.array([s['betas'][wn][2] for s in all_results])
    D = BP - BQ
    print("  {:10s} {:+.3f}+/-{:.3f} {:2s} {:+.3f}+/-{:.3f} {:2s} {:+.3f}+/-{:.3f} {:2s} {:+.3f} {:2s}".format(
        wn, BI.mean(), BI.std() / n**.5, _star(BI),
        BP.mean(), BP.std() / n**.5, _star(BP),
        BQ.mean(), BQ.std() / n**.5, _star(BQ),
        D.mean(), _star(D)))
    if wn != 'uniform':
        SP = np.array([s['shufs'][wn][1] for s in all_results])
        if np.isfinite(SP).any():
            dsh = BP - SP
            print("  {:10s} {:>16s} S_post-shuffle={:+.3f}+/-{:.3f} {}".format(
                '', '', dsh.mean(), dsh.std() / n**.5, _star(dsh)))
print("\nRead: if 'uniform' I is nonzero (sanity) but RPE-weighted I~0 => weighting kills signal.")
print("      if some weighting's S_post > shuffle AND > S_pre => real post-dev fingerprint.")
