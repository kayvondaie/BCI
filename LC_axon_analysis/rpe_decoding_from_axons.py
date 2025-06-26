from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import session_counting
import data_dict_create_module_test as ddc
sessions = session_counting.counter2(["BCINM_017"],'010112',has_pophys=False)
#%%
import sys
sys.path.append('./LC_axon_analysis')  # or use full path if needed
from scipy.signal import medfilt
from scipy.signal import correlate
from axon_helper_module import *
from BCI_data_helpers import *

processing_mode = 'all'
si = 4
if processing_mode == 'all':
    inds = np.arange(0,len(sessions))
else:
    inds = np.arange(si,si+1)
AXON_REW, AXON_TS = [], []
SESSION = []
import bci_time_series as bts
for i in inds:
    print(i)
    data, *_ = get_axon_data_dict(sessions,i)
    
    # --- New cell: Correlation matrix, SVD, and distribution of correlations ---
    try:
        rt = data['reward_time'];dt_si = data['dt_si']

        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        rt[np.isnan(rt)] = 20
        numsteps = np.array([len(x) for x in data['step_time']])
        rpe = compute_rpe(rt, baseline=1, window=20, fill_value=50)
        #rpe = compute_rpe_standard(numsteps, baseline=1, window=20, fill_value=50)
        #rpe = compute_rpe(rt!=20, baseline=1, window=20, fill_value=50)
        # Keep neurons x trials by averaging over time (not space)
        axons = np.nanmean(data['ch1']['F'][40:, :, :], axis=0)  # shape: (neurons, trials)
        X = axons.T  # shape: (trials, neurons)

        y = rpe      # shape (82,)
        
        # Remove any samples with NaNs
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X = X[valid]
        y = y[valid]
        
        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        y_pred = np.full_like(y, np.nan, dtype=np.float64)

        coefs = []  # Store model coefficients
        
        # Use fixed regularization alpha (or you can CV it separately)
        alpha = 1.0
        for train_idx, test_idx in kf.split(X):
            model = Ridge(alpha=alpha)
            model.fit(X[train_idx], y[train_idx])
            y_pred[test_idx] = model.predict(X[test_idx])
            coefs.append(model.coef_)  # save weights
        
        # Compute R² on held-out predictions
        r2 = r2_score(y, y_pred)
        print(f"Cross-validated R²: {r2:.3f}")
        
        # Plot: true vs predicted (held out)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel('True RPE')
        plt.ylabel('Predicted RPE (held-out)')
        plt.title('CV Ridge regression: Axons → RPE')
        plt.subplot(122)
        plt.plot(y, label='True RPE', linewidth=2)
        plt.plot(y_pred, label='Predicted RPE (held-out)', linewidth=2, linestyle='--')
        plt.xlabel('Time index')
        plt.ylabel('RPE')
        plt.title('Time series: True vs Predicted RPE')
        plt.legend()
        plt.tight_layout()
        plt.show()

        from sklearn.utils import shuffle

        y_shuffled = shuffle(y, random_state=0)
        y_pred_shuff = np.full_like(y, np.nan)
        
        for train_idx, test_idx in kf.split(X):
            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], y_shuffled[train_idx])
            y_pred_shuff[test_idx] = model.predict(X[test_idx])
        
        r2_shuff = r2_score(y_shuffled, y_pred_shuff)
        print(f'Shuffled R²: {r2_shuff:.3f}')
        
        # --- New cell: Analyze model coefficients ---
        coefs = np.stack(coefs)  # shape: (n_folds, n_neurons)
        mean_coefs = np.mean(coefs, axis=0)

        # Bar plot of mean coefficients
        # plt.figure(figsize=(10, 4))
        # plt.bar(np.arange(len(mean_coefs)), mean_coefs)
        # plt.xlabel('Neuron index')
        # plt.ylabel('Mean Ridge coefficient')
        # plt.title('Neuron contributions to RPE prediction (mean across folds)')
        # plt.tight_layout()
        # plt.show()

       
    except:
        print('error')
        continue
#%%
from scipy.stats import pearsonr
import numpy as np

# Initialize arrays
r_vals = np.zeros(X.shape[1])
p_vals = np.zeros(X.shape[1])

# Loop over neurons
for i in range(X.shape[1]):
    r, p = pearsonr(X[:, i], y)
    r_vals[i] = r
    p_vals[i] = p

# Optionally: print top correlated neurons
top_idx = np.argsort(np.abs(r_vals))[::-1]
print("\nTop 10 axons by |correlation| with RPE:")
for j in range(10):
    i = top_idx[j]
    print(f"  Axon {i:2d}  r = {r_vals[i]:+.3f}   p = {p_vals[i]:.2e}")
#%%
import sys
sys.path.append('./LC_axon_analysis')  # or use full path if needed
from scipy.signal import medfilt
from scipy.signal import correlate
from axon_helper_module import *
from BCI_data_helpers import *

processing_mode = 'one'
si = 4
if processing_mode == 'all':
    inds = np.arange(0,len(sessions))
else:
    inds = np.arange(si,si+1)
AXON_REW, AXON_TS = [], []
SESSION = []
import bci_time_series as bts
for i in inds:
    print(i)
    data, *_ = get_axon_data_dict(sessions,i)

    try:
        rt = data['reward_time']; dt_si = data['dt_si']

        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        rt[np.isnan(rt)] = 20
        numsteps = np.array([len(x) for x in data['step_time']])

        # Compute different types of RPE
        rpe = compute_rpe(rt, baseline=1, window=20, fill_value=50)
        rpe_steps = compute_rpe_standard(numsteps, baseline=1, window=20)
        rpe_hit = compute_rpe_standard(rt != 20, baseline=1, window=20)

        # Axon activity: average over time, keep neurons x trials
        axons = np.nanmean(data['ch1']['F'][40:, :, :], axis=0)  # shape (neurons, trials)
        X = axons.T  # shape (trials, neurons)

        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        alpha = 1.0

        # --- New cell: Compare RPE variants ---
        rpe_types = {
            'RT-based RPE': rpe,
            'Step count RPE': rpe_steps,
            'Hit-based RPE': rpe_hit
        }
        rpe_labels = {
            'RT-based RPE': 'RPE (based on RT)',
            'Step count RPE': 'RPE (based on steps)',
            'Hit-based RPE': 'RPE (based on hits)'
        }

        coefs = []
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, (label, rpe_var) in zip(axes, rpe_types.items()):
            y = rpe_var
            valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X_valid = X[valid]
            y_valid = y[valid]

            y_pred = np.full_like(y_valid, np.nan, dtype=np.float64)
            for train_idx, test_idx in kf.split(X_valid):
                model = Ridge(alpha=alpha)
                model.fit(X_valid[train_idx], y_valid[train_idx])
                y_pred[test_idx] = model.predict(X_valid[test_idx])
                coefs.append(model.coef_)  # save weights
            

            r2 = r2_score(y_valid, y_pred)
            ax.plot(y_valid, label='True', linewidth=2)
            ax.plot(y_pred, label='Predicted', linewidth=2, linestyle='--')
            ax.set_title(f'{label}\nR² = {r2:.2f}')
            ax.set_xlabel('Trial')
            ax.set_ylabel('RPE')
            ax.autoscale(enable=True, axis='both', tight=True)  # <-- axis tight equivalent
            ax.set_ylabel(rpe_labels[label])

            ax.legend()

        plt.suptitle(f'Session {i}: RPE prediction from axons')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f'Error in session {i}: {e}')
        continue
    plt.figure(figsize = (5,2))
    plt.subplot(121);
    plt.plot(coefs[0],coefs[1],'k.')
    plt.xlabel('RPE_RT coef');plt.ylabel('RPE_step coef')
    plt.ylabel('RPE_HIT coef');plt.xlabel('RPE_step coef')
    plt.subplot(122);
    plt.plot(coefs[1],coefs[2],'k.')

#%%
import sys
sys.path.append('./LC_axon_analysis')  # or use full path if needed
from scipy.signal import medfilt
from scipy.signal import correlate
from axon_helper_module import *
from BCI_data_helpers import *

processing_mode = 'one'
regress_out_steps = False  # <-- Set this flag to enable/disable step regression
si = 6
if processing_mode == 'all':
    inds = np.arange(0,len(sessions))
else:
    inds = np.arange(si,si+1)
AXON_REW, AXON_TS = [], []
SESSION = []
import bci_time_series as bts
for i in inds:
    print(i)
    data, *_ = get_axon_data_dict(sessions,i)

    try:
        rt = data['reward_time']; dt_si = data['dt_si']

        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        rt[np.isnan(rt)] = 20
        numsteps = np.array([len(x) for x in data['step_time']])

        # Compute different types of RPE
        rpe = compute_rpe(rt, baseline=1, window=20, fill_value=50)
        rpe_steps = compute_rpe_standard(numsteps, baseline=1, window=20)
        rpe_hit = compute_rpe_standard(rt != 20, baseline=1, window=20)

        # Axon activity: average over time, keep neurons x trials
        axons = np.nanmean(data['ch1']['F'][40:, :, :], axis=0)  # shape (neurons, trials)
        X = axons.T

        if regress_out_steps:
            from sklearn.linear_model import LinearRegression
        
            step_valid = np.isfinite(numsteps) & np.all(np.isfinite(X), axis=1)
            X_clean = X[step_valid]
            steps_clean = numsteps[step_valid].reshape(-1, 1)
        
            # Regress out steps from each neuron
            X_detrended = np.zeros_like(X_clean)
            for j in range(X.shape[1]):
                model = LinearRegression().fit(steps_clean, X_clean[:, j])
                X_detrended[:, j] = X_clean[:, j] - model.predict(steps_clean)
        
            X = np.full_like(X, np.nan)
            X[step_valid] = X_detrended


        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        alpha = 1.0

        # --- New cell: Compare RPE variants ---
        rpe_types = {
            'RT-based RPE': rpe,
            'Step count RPE': rpe_steps,
            'Hit-based RPE': rpe_hit
        }
        rpe_labels = {
            'RT-based RPE': 'RPE (based on RT)',
            'Step count RPE': 'RPE (based on steps)',
            'Hit-based RPE': 'RPE (based on hits)'
        }

        coefs = []
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, (label, rpe_var) in zip(axes, rpe_types.items()):
            y = rpe_var
            valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X_valid = X[valid]
            y_valid = y[valid]

            y_pred = np.full_like(y_valid, np.nan, dtype=np.float64)
            for train_idx, test_idx in kf.split(X_valid):
                model = Ridge(alpha=alpha)
                model.fit(X_valid[train_idx], y_valid[train_idx])
                y_pred[test_idx] = model.predict(X_valid[test_idx])
                coefs.append(model.coef_)  # save weights
            

            r2 = r2_score(y_valid, y_pred)
            ax.plot(y_valid, label='True', linewidth=2)
            ax.plot(y_pred, label='Predicted', linewidth=2, linestyle='--')
            c,p = pearsonr(y_pred,y_valid)
            print(f'p = {p:.3f}' + '   ' + f'c = {c:.3f}')
            ax.set_title(f'{label}\nR² = {r2:.2f}')
            ax.set_xlabel('Trial')
            ax.set_ylabel('RPE')
            ax.autoscale(enable=True, axis='both', tight=True)  # <-- axis tight equivalent
            ax.set_ylabel(rpe_labels[label])

            ax.legend()

        plt.suptitle(f'Session {i}: RPE prediction from axons')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f'Error in session {i}: {e}')
        continue
    


#%%
from BCI_data_helpers import *
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import bci_time_series as bts

# Containers for pooled results
R_AXON, R_RPE, RPE_COEFF = [], [], []
TRACES = {'Low': [],'Medium': [], 'High': []}
TIMES = None

def residuals(y, x):
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    return y - model.predict(x.reshape(-1, 1))

# --- Loop over sessions ---
for i in range(4,5):
    print(f"Session {i}")
    try:
        mouse = sessions['Mouse'][i]
        session = sessions['Session'][i]
        folder_base = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/'
        folder = folder_base + 'pophys/' if os.path.exists(folder_base + 'pophys/') else folder_base
        data = np.load(os.path.join(folder, f"data_main_{mouse}_{session}_BCI.npy"), allow_pickle=True)

        # Trial-level variables
        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        rt[np.isnan(rt)] = 20
        numsteps = np.array([len(x) for x in data['step_time']])
        rpe = compute_rpe(numsteps, baseline=1, window=20, fill_value=50)

        # Axon signal (mean over time and space)
        axons = np.nanmean(np.nanmean(data['ch1']['F'][40:,:,:], axis=1), axis=0)
        
        df = pd.DataFrame({'axons': axons, 'steps': numsteps, 'RPE': rpe}).dropna()
        if len(df) < 10:
            print(f"Skipping session {i}: too few trials")
            continue

        # Regress out steps to get residuals
        resid_axons = residuals(df['axons'].values, df['steps'].values)
        resid_rpe = residuals(df['RPE'].values, df['steps'].values)

        R_AXON.append(resid_axons)
        R_RPE.append(resid_rpe)

        # Regression just for RPE beta
        X = sm.add_constant(df[['steps', 'RPE']])
        y = df['axons']
        model = sm.OLS(y, X).fit()
        RPE_COEFF.append(model.params['RPE'])

        # Bin trials by residual RPE
        try:
            bins = pd.qcut(resid_rpe, q=3, labels=["Low","Medium", "High"])
        except ValueError as ve:
            print(f"Skipping session {i}: {ve}")
            continue

        # Time vector
        t_bci = data['t_bci'][:-1]
        if TIMES is None:
            TIMES = t_bci

        # Trial-averaged traces per RPE bin
        for b in ["Low", "High"]:
            trial_inds = (bins == b)
            if np.sum(trial_inds) < 5:
                continue
            mean_trace = np.nanmean(data['ch1']['F'][:, :, trial_inds], axis=(1, 2))
            TRACES[b].append(mean_trace)

    except Exception as e:
        print(f"Skipping session {i}: {e}")
        continue

# --- Pooled partial correlation (residual-residual) ---
x = np.concatenate(R_RPE)
y = np.concatenate(R_AXON)
r, p = pearsonr(x, y)
print(f"\nPooled partial correlation: r = {r:.3f}, p = {p:.3g}")

# --- Plot: partial correlation (binned scatter) ---
pf.mean_bin_plot(x, y, 5, 1, 1, 'k')
plt.xlabel('Residual RPE (steps regressed out)')
plt.ylabel('Residual LC axon activity')
plt.title(f'Pooled partial corr: r = {r:.2f}, p = {p:.3g}')
plt.show()

# --- Plot: group-averaged traces by residual RPE ---
plt.figure(figsize=(6, 4))
for b, color in zip(["Low", "High"], ['blue', 'red']):
    if len(TRACES[b]) == 0:
        continue
    try:
        traces = np.stack(TRACES[b])  # shape: (n_sessions, n_timepoints)
        mean_trace = np.nanmean(traces, axis=0)
        sem_trace = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])

        plt.plot(t_bci, mean_trace[:], label=f'{b} residual RPE', color=color)
        plt.fill_between(TIMES, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3, color=color)
    except ValueError as ve:
        print(f"Skipping {b} bin in plot due to: {ve}")
        continue

plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('LC activity (dF/F)')
plt.title('Session-averaged LC activity by residual RPE')
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot raw axons vs steps and RPE ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(df['steps'], df['axons'], alpha=0.6)
plt.xlabel('Steps')
plt.ylabel('Axon signal')
plt.title('Raw axons vs steps')

plt.subplot(1, 2, 2)
plt.scatter(df['RPE'], df['axons'], alpha=0.6)
plt.xlabel('RPE')
plt.ylabel('Axon signal')
plt.title('Raw axons vs RPE')

plt.tight_layout()
plt.show()
