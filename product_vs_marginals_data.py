#%% ============================================================================
# CELL 1: Imports
# ============================================================================
"""
Product-vs-marginals eligibility test on the DATA (ported from
run_model_product_vs_marginals.py), on the WORKING HI substrate:
  - trial-epoch activity F (pre-epoch, tsta in (-10,0)), OFFSET_SEC=0 (no lag)
  - pre = stimulated target neuron's epoch activity per trial (cl group)
  - post = nontarget neuron's epoch activity per trial
  - dev2 baseline = mean over first 20 trials (post side)
Per pair (i = nontarget/post, j = target/pre), summed over ALL trials:
  pre_x_dpost = sum_t pre_j(t) * (post_i(t) - bl_i)     (product; the HI / Fig 3d form)
  pre         = sum_t pre_j(t)                          (pre marginal; target activity)
  dpost       = sum_t (post_i(t) - bl_i)                (post-dev marginal)
  post        = sum_t post_i(t)                         (post raw marginal)
Target: dW = AMP[1]-AMP[0] (whole-session photostim connectivity change).
Test: does the coactivity PRODUCT survive controlling for the marginals?
Model reference (uniform): pre beta +0.55 >> product +0.16; product +0.16***.
No RPE weighting (uniform) -- matches the model's uniform column.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
print("Setup complete!")

#%% ============================================================================
# CELL 2: Config
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
N_BASELINE = 20            # dev2 baseline: first 20 trials
TERMS = ['pre_x_dpost', 'pre', 'dpost', 'post']
JOINT = ['pre_x_dpost', 'pre', 'post']     # joint fit (drop dpost: collinear with post)
all_results = []


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


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

for mi in range(len(mice)):
    mouse = mice[mi]
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
            dt_si = data['dt_si']
            F = data['F']
            n_frames, n_neurons, trl = F.shape
            tsta = np.arange(0, 12, dt_si); tsta = tsta - tsta[int(2 / dt_si)]

            # ---- pair selection (i = nontarget/post, j = target/pre) ----
            dw_list, pair_cl_list, pair_nt_list = [], [], []
            for gi in range(stimDist.shape[1]):
                cl = np.where((stimDist[:, gi] < 10) & (AMP[0][:, gi] > 0.1) & (AMP[1][:, gi] > 0.1))[0]
                if cl.size == 0:
                    continue
                nt = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))[0]
                if nt.size == 0:
                    continue
                dw_list.append(AMP[1][nt, gi] - AMP[0][nt, gi])
                pair_cl_list.append(np.tile(cl, (len(nt), 1)))
                pair_nt_list.append(nt)
            if len(dw_list) == 0:
                print("  no pairs."); continue
            Y = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            n_pairs = len(Y)

            cl_weights = np.zeros((n_pairs, n_neurons))
            off = 0
            for gi_idx in range(len(dw_list)):
                for qi in range(len(dw_list[gi_idx])):
                    cln = pair_cl_list[gi_idx][qi]
                    cl_weights[off + qi, cln] = 1.0 / len(cln)
                off += len(dw_list[gi_idx])

            # ---- pre-epoch activity (N, trl); OFFSET_SEC=0 so pre & post share epoch ----
            F_nan = F.copy(); F_nan[np.isnan(F_nan)] = 0
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            epoch = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1] + 1, :, :], axis=0)   # (N, trl)

            pre_pt = cl_weights @ epoch            # (n_pairs, trl) target activity per pair
            post_pt = epoch[all_nt, :]             # (n_pairs, trl) nontarget activity
            bl = post_pt[:, :min(N_BASELINE, trl)].mean(axis=1)     # dev2 baseline
            post_dev = post_pt - bl[:, None]

            E = {
                'pre_x_dpost': np.sum(pre_pt * post_dev, axis=1),
                'pre':   np.sum(pre_pt, axis=1),
                'dpost': np.sum(post_dev, axis=1),
                'post':  np.sum(post_pt, axis=1),
            }
            uni = {}
            for t in TERMS:
                ok = np.isfinite(E[t]) & np.isfinite(Y)
                uni[t] = spearmanr(E[t][ok], Y[ok])[0] if ok.sum() >= 5 and np.std(E[t][ok]) > 0 else np.nan
            X = np.column_stack([_z(E[t]) for t in JOINT])
            beta, *_ = np.linalg.lstsq(X, _z(Y), rcond=None)
            beta = dict(zip(JOINT, beta))

            all_results.append({'mouse': mouse, 'session': session, 'n_pairs': n_pairs,
                                'n_groups': len(dw_list), 'uni': uni, 'beta': beta})
            print("  {} pairs / {} targets | uni prod={:+.3f} pre={:+.3f} | joint prod={:+.3f} pre={:+.3f} post={:+.3f}".format(
                n_pairs, len(dw_list), uni['pre_x_dpost'], uni['pre'],
                beta['pre_x_dpost'], beta['pre'], beta['post']))

        except Exception as e:
            print(f"  FAILED: {e}"); traceback.print_exc(); continue

print(f"\nTotal sessions: {len(all_results)}")
np.save(os.path.join(RESULTS_DIR, 'product_vs_marginals_data.npy'), all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 4: Aggregate
# ============================================================================
all_results = np.load(os.path.join(RESULTS_DIR, 'product_vs_marginals_data.npy'),
                      allow_pickle=True).tolist()
n = len(all_results)


def _star(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else 'ns'


print(f"\nDATA product-vs-marginals, n={n} sessions (uniform, no RPE):")
print("  {:14s} {:>18s} {:>18s}".format('term', 'univariate rho', 'joint beta'))
for t in TERMS:
    u = np.array([s['uni'][t] for s in all_results])
    b = np.array([s['beta'][t] for s in all_results]) if t in JOINT else None
    bs = "{:+.3f}+/-{:.3f} {:3s}".format(b[np.isfinite(b)].mean(), b[np.isfinite(b)].std() / n**.5, _star(b)) if b is not None else "     (not in joint)"
    print("  {:14s} {:+.3f}+/-{:.3f} {:3s}   {}".format(
        t, np.nanmean(u), np.nanstd(u) / n**.5, _star(u), bs))
print("\nModel reference (uniform): joint pre +0.55***, product +0.16***  (pre dominates)")
print("Read: if data product survives controlling for pre & post -> coactivity eligibility;")
print("      compare whether pre-marginal predicts data dW like it does in the model.")

fig = plt.figure(figsize=(4.4, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.0 / fw, 0.75 / fh, 3.0 / fw, 1.9 / fh])
x = np.arange(len(TERMS))
U = np.array([[s['uni'][t] for t in TERMS] for s in all_results])
ax.bar(x - 0.2, np.nanmean(U, 0), 0.38, yerr=np.nanstd(U, 0) / n**.5, color='#888', capsize=2, label='univariate')
Bj = np.array([[s['beta'].get(t, np.nan) for t in TERMS] for s in all_results])
ax.bar(x + 0.2, np.nanmean(Bj, 0), 0.38, yerr=np.nanstd(Bj, 0) / n**.5, color='#1b6faf', capsize=2, label='joint beta')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(['$r_{pre}\\Delta r_{post}$', '$r_{pre}$', '$\\Delta r_{post}$', '$r_{post}$'], fontsize=7)
ax.set_ylabel('corr / beta with $\\Delta W$')
ax.set_title('DATA: product vs marginals (n={})'.format(n))
ax.legend(frameon=False, fontsize=7)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(RESULTS_DIR, 'product_vs_marginals_data.' + ext), dpi=200, bbox_inches='tight')
print("Figure saved.")
