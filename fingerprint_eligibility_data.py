#%% ============================================================================
# CELL 1: Imports and helpers
# ============================================================================
"""
Eligibility-FORM fingerprint on the data (ported from the model analysis).

Model result: with a CONTINUOUS behavioral weighting, dW decomposes as
    dW(pair) = b_I*I + b_Spost*S_post + b_Spre*S_pre + b_M*M
where, per pair (pre = CN group, post = nontarget neuron):
    q_x  = sum_t w(t) * dev_x(t)          (RPE-weighted fluctuation)
    m_x  = mean_t act_x(t)                (overall mean activity)
    I      = sum_t w(t) * dpre(t)*dpost(t)   (interaction)
    S_post = m_pre * q_post                  (POST deviated, pre raw)  <- model's rule
    S_pre  = m_post * q_pre                   (PRE deviated)
    M      = m_pre * m_post                   (mean-mean; RPE-independent background)
w(t) = continuous lickport/step prediction-error (bts step_vector), per frame.
Read: S_post >> S_pre, collapsing under a time-shuffle of w, => the biological
eligibility deviates the POSTsynaptic factor and is behavior/reward-gated.
dev uses an EMA baseline (tau_bl); m uses the overall time-mean (as in the model).
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
    """Causal EMA: y[t] = (1-1/tau) y[t-1] + (1/tau) x[t]."""
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

TAU_BL = 360.0          # seconds; EMA baseline for the deviation dev = df - h_bar
TAU_STEP = 2.0          # seconds; baseline for the lickport-RPE weighting
PRE_LAG_FRAMES = 1      # pre leads post by this many frames (paper convention)
N_SHUF = 20             # circular-shift shuffles of w(t) for the null
WEIGHT_MODE = 'step_rpe'   # 'step_rpe' (lickport prediction error) or 'step' (raw)

BLOCKS = ['I', 'S_post', 'S_pre', 'M']
print(f"tau_bl={TAU_BL}s, tau_step={TAU_STEP}s, pre_lag={PRE_LAG_FRAMES}, "
      f"weight={WEIGHT_MODE}, n_shuf={N_SHUF}")

all_results = []

#%% ============================================================================
# CELL 3: Main loop -- per session, fit dW ~ [I, S_post, S_pre, M]
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


def _zc(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def _fit_blocks(I, S_post, S_pre, M, Y):
    X = np.column_stack([_zc(I), _zc(S_post), _zc(S_pre), _zc(M)])
    b, *_ = np.linalg.lstsq(X, _zc(Y), rcond=None)
    return b


for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == True))[0]

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

            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['df_closedloop', 'F', 'mouse', 'session',
                        'conditioned_neuron', 'dt_si', 'step_time',
                        'reward_time', 'BCI_thresholds']
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print("  Skipping -- file not found."); continue

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']], float)
            rt_filled = rt.copy(); rt_filled[~np.isfinite(rt_filled)] = 30.0

            df_full = np.asarray(data['df_closedloop'], float)
            df_full[np.isnan(df_full)] = 0.0
            n_df, total_frames = df_full.shape
            if n_df != n_neurons:
                print("  neuron mismatch; skipping."); continue

            step_vector, reward_vector, trial_start_vector = (
                bts.bci_time_series_fun(folder, data, rt_filled, dt_si))
            tf = min(len(step_vector), total_frames)
            step_vector = np.asarray(step_vector[:tf], float)
            df_full = df_full[:, :tf]; total_frames = tf

            # ---- continuous behavioral weighting w(t) (lickport / step) ----
            tau_step_frames = TAU_STEP / dt_si
            if WEIGHT_MODE == 'step_rpe':
                w = step_vector - ema_causal(step_vector, tau_step_frames, init=float(step_vector.mean()))
            else:
                w = step_vector.copy()

            # ---- deviation (EMA baseline) and mean activity ----
            tau_bl_frames = TAU_BL / dt_si
            dev = df_full - ema_causal(df_full, tau_bl_frames, axis=-1, init=df_full[:, 0])
            m_neuron = df_full.mean(axis=1)                       # overall mean per neuron

            # ---- pair selection (pre = CN group cl, post = nontarget) ----
            dw_list, pair_cl_list, pair_nt_list = [], [], []
            for gi in range(stimDist.shape[1]):
                cl = np.where((stimDist[:, gi] < 10) &
                              (AMP[0][:, gi] > 0.1) & (AMP[1][:, gi] > 0.1))[0]
                if cl.size == 0:
                    continue
                nt = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))[0]
                if nt.size == 0:
                    continue
                dw_list.append(AMP[1][nt, gi] - AMP[0][nt, gi])
                pair_cl_list.append(cl)
                pair_nt_list.append(nt)
            if len(dw_list) == 0:
                print("  no pairs."); continue

            def build(w_use):
                q_neuron = (w_use[None, :] * dev).sum(axis=1)      # per-neuron RPE-weighted fluct
                I_all, Sp_all, Sq_all, M_all, Y_all = [], [], [], [], []
                for gi in range(len(dw_list)):
                    cl, nt = pair_cl_list[gi], pair_nt_list[gi]
                    pre_g = df_full[cl, :].mean(axis=0)
                    pre_lag = np.empty_like(pre_g)
                    pre_lag[:PRE_LAG_FRAMES] = 0.0
                    pre_lag[PRE_LAG_FRAMES:] = pre_g[:-PRE_LAG_FRAMES] if PRE_LAG_FRAMES > 0 else pre_g
                    dpre = pre_lag - ema_causal(pre_lag, tau_bl_frames, init=pre_lag[0])
                    m_pre = pre_lag.mean()
                    q_pre = float(np.sum(w_use * dpre))
                    wd = w_use * dpre
                    I_p = dev[nt, :] @ wd                          # sum_t w*dpre*dpost per nt
                    q_post = q_neuron[nt]; m_post = m_neuron[nt]
                    I_all.append(I_p)
                    Sp_all.append(m_pre * q_post)                  # S_post
                    Sq_all.append(m_post * q_pre)                  # S_pre
                    M_all.append(m_pre * m_post)                   # M
                    Y_all.append(dw_list[gi])
                return (np.concatenate(I_all), np.concatenate(Sp_all),
                        np.concatenate(Sq_all), np.concatenate(M_all),
                        np.nan_to_num(np.concatenate(Y_all)))

            I, Sp, Sq, M, Y = build(w)
            b_true = _fit_blocks(I, Sp, Sq, M, Y)
            rng = np.random.default_rng(0)
            b_shuf = np.mean([_fit_blocks(*build(np.roll(w, rng.integers(total_frames // 10, total_frames)))[:4], Y)
                              for _ in range(N_SHUF)], axis=0)

            all_results.append({'mouse': mouse, 'session': session, 'n_pairs': len(Y),
                                'beta_true': b_true, 'beta_shuf': b_shuf})
            print("  {} pairs | S_post={:+.3f} (shuf {:+.3f}) | S_pre={:+.3f}".format(
                len(Y), b_true[1], b_shuf[1], b_true[2]))

        except Exception as e:
            print(f"  FAILED: {e}"); traceback.print_exc(); continue

print(f"\nTotal sessions: {len(all_results)}")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'fingerprint_eligibility_data.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Aggregate + figure
# ============================================================================
all_results = np.load(os.path.join(RESULTS_DIR, 'fingerprint_eligibility_data.npy'),
                      allow_pickle=True).tolist()
BT = np.array([s['beta_true'] for s in all_results])
BS = np.array([s['beta_shuf'] for s in all_results])


def _star(v):
    v = v[np.isfinite(v)]
    p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


print(f"\nDATA fingerprint across {len(BT)} sessions (standardized beta):")
print("  {:10s} {:>18s} {:>18s}".format('block', WEIGHT_MODE, 'shuffled'))
for i, b in enumerate(BLOCKS):
    print("  {:10s} {:+.3f}+/-{:.3f} {:>4s}   {:+.3f}+/-{:.3f} {:>4s}".format(
        b, BT[:, i].mean(), BT[:, i].std() / np.sqrt(len(BT)), _star(BT[:, i]),
        BS[:, i].mean(), BS[:, i].std() / np.sqrt(len(BS)), _star(BS[:, i])))
d_sh = BT[:, 1] - BS[:, 1]
d_ps = BT[:, 1] - BT[:, 2]
print("\n  S_post - S_post(shuffle): {:+.3f} +/- {:.3f}  {}".format(
    d_sh.mean(), d_sh.std() / np.sqrt(len(d_sh)), _star(d_sh)))
print("  S_post - S_pre:           {:+.3f} +/- {:.3f}  {}".format(
    d_ps.mean(), d_ps.std() / np.sqrt(len(d_ps)), _star(d_ps)))

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(4)
for j, (B, c, lab) in enumerate([(BT, '#1baf7a', WEIGHT_MODE), (BS, '#b0b0b0', 'shuffled')]):
    ax.bar(x + (j - 0.5) * 0.4, B.mean(0), 0.38, yerr=B.std(0) / np.sqrt(len(B)),
           color=c, capsize=3, label=lab)
    for i in range(4):
        ax.scatter(np.full(len(B), x[i] + (j - 0.5) * 0.4) + np.random.uniform(-0.06, 0.06, len(B)),
                   B[:, i], s=12, color='k', alpha=0.3, zorder=3)
ax.axhline(0, color='k', lw=1)
ax.set_xticks(x); ax.set_xticklabels(['I', 'S_post\n(POST dev)', 'S_pre\n(PRE dev)', 'M'])
ax.set_ylabel('standardized beta')
ax.set_title(f'DATA eligibility fingerprint (n={len(BT)} sessions)')
ax.legend(frameon=False)
plt.tight_layout()
_p = os.path.join(RESULTS_DIR, 'fig_fingerprint_eligibility_data.png')
plt.savefig(_p, dpi=150, bbox_inches='tight'); plt.show()
print("Figure saved:", _p)
