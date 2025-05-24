import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# === Load Data ===
data_dict = np.load(
    r"C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\Data\Figures 2025\Hebbian index fits\hebbian_fit_inputs.npy",
    allow_pickle=True
).item()

HIb = data_dict["HIb"]
THR = data_dict["THR"]
RT  = data_dict["RT"]
RPE = data_dict["RPE"]
HIT = data_dict["HIT"]

# === Helper: Exponential kernel smoothing ===
def exp_kernel_smooth(x, tau):
    kernel = np.exp(-np.arange(0, 10*tau) / tau)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode='same')

# === Use best taus from full fit ===
tau_rpe = 10
tau_hit = 10
tau_thr = 10

X_list = []
y_list = []
session_ids = []
session_id_nums = []

# === Build full dataset with best taus ===
for i in range(len(HIb)):
    try:
        n = len(HIb[i])
        if n == 0 or np.nanstd(HIb[i]) == 0:
            print(f"Skipping session {i}: empty or zero variance")
            continue

        rt = np.asarray(RT[i])[:n]
        rpe = np.nan_to_num(RPE[i])[:n]
        hit = np.asarray(HIT[i])[:n]
        thr = np.asarray(THR[i])[:n]
        if len(thr) <= 2 or thr[2] == 0:
            print(f"Skipping session {i}: invalid threshold data")
            continue
        thr_norm = (thr - thr[2]) / thr[2]

        delta_hit = np.concatenate([[0], np.diff(hit)])[:n]
        delta_thr = np.concatenate([[0], np.diff(thr_norm)])[:n]

        rpe_smooth = exp_kernel_smooth(rpe, tau_rpe)[:n]
        dhit_smooth = exp_kernel_smooth(delta_hit, tau_hit)[:n]
        dthr_smooth = exp_kernel_smooth(delta_thr, tau_thr)[:n]

        if not all(len(arr) == n for arr in [rt, rpe_smooth, dhit_smooth, dthr_smooth, hit, thr_norm]):
            print(f"Skipping session {i}: mismatched feature lengths")
            continue

        df = pd.DataFrame({
            'RT': rt,
            'RPE': rpe_smooth,
            'Delta_HIT': dhit_smooth,
            'Delta_THR': dthr_smooth,
            'HIT': hit,
            'THR': thr_norm
        })
        X_list.append(df)

        hi = np.asarray(HIb[i])
        hi_zscore = (hi - np.nanmean(hi)) / np.nanstd(hi)
        y_list.append(hi_zscore[:n])
        session_ids.append(i)
        session_id_nums.append(i)

    except Exception as e:
        print(f"Skipping session {i}: {str(e)}")

# === Leave-One-Session-Out CV ===
all_preds = []
all_true = []
session_metrics = []
all_p, all_r = [],[]

for test_session in session_id_nums:
    train_idx = [i for i, s in enumerate(session_id_nums) if s != test_session]
    test_idx = [i for i, s in enumerate(session_id_nums) if s == test_session]

    if not test_idx:
        continue

    X_train = pd.concat([X_list[i] for i in train_idx], ignore_index=True)
    y_train = np.concatenate([y_list[i] for i in train_idx])
    X_test = pd.concat([X_list[i] for i in test_idx], ignore_index=True)
    y_test = np.concatenate([y_list[i] for i in test_idx])

    train_mean = X_train.mean()
    train_std = X_train.std().replace(0, 1).fillna(1)
    train_mean = train_mean.fillna(0)

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    if len(X_test) == 0 or len(X_train) == 0:
        continue

    model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r, p = pearsonr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Held-out session {test_session:2d}: Pearson r = {r:.3f}, p = {p:.2e}, R² = {r2:.3f}")

    session_metrics.append({'session': test_session, 'r': r, 'p': p, 'r2': r2})
    all_preds.append(y_pred)
    all_true.append(y_test)
    all_p.append(p)
    all_r.append(r)

if all_preds:
    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_preds)
    r2 = r2_score(y_true_all, y_pred_all)
    r, p = pearsonr(y_true_all, y_pred_all)
    print(f"\nLeave-One-Session-Out CV R² = {r2:.3f}")
    print(f"Leave-One-Session-Out CV Pearson r = {r:.3f}, p = {p:.2e}")
