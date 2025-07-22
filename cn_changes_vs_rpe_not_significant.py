# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:13:36 2025

@author: kayvon.daie
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 12:39:48 2025

@author: kayvon.daie
"""


import session_counting
import data_dict_create_module_test as ddct

list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
import bci_time_series as bts
from BCI_data_helpers import *
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.stats import ttest_1samp
import re


mice = ["BCI102","BCI103","BCI105","BCI106","BCI109","BCI104","BCI107","BCI88","BCI93"]
mice = ["BCI102","BCI103"]
PVAL, COEFS,ALLX,ALLCN = [],[],[],[]
for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) & (list_of_dirs['Has data_main.npy']==True))[0]
    si = 7
    
    for sii in range(0,len(session_inds)):        
    #for sii in range(si,si+1):
        try:
            num_bins      =  2000         # number of bins to calculate correlations
            print(sii)
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
            photostim_keys = []
            bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si','step_time','reward_time','BCI_thresholds']
            try:
                data = ddct.load_hdf5(folder,bci_keys,photostim_keys )
            except:
                continue        
            BCI_thresholds = data['BCI_thresholds']    
            try:
                cn = data['conditioned_neuron'][0][0]
            except:
                cn = 0;
            dfcn = data['df_closedloop'][cn,:]
            
        
            dt_si = data['dt_si']
            trl = F.shape[2]
            tsta = np.arange(0,12,data['dt_si'])
            tsta=tsta-tsta[int(2/dt_si)]
            
            # Initialize arrays
            kstep = np.zeros((F.shape[1], trl))
            Q = np.zeros((F.shape[1], trl))
            krewards = np.zeros((F.shape[1], trl))    
            step_raw = data['step_time']
        
            # --- Replace step_time and reward_time with parsed versions if needed ---
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
        
          
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
            hit = np.isnan(rt)==0;
            rt[np.isnan(rt)] = 30;
            BCI_thresholds = data['BCI_thresholds']
    
            step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
                folder, data, rt, dt_si)
            df = data['df_closedloop']
            ind = np.where(trial_start_vector == 1)[0];        
            Fepoch = []
            two = np.floor((2,1)/dt_si).astype(int)
            cn = data['conditioned_neuron'][0][0]
            for j in range(2):
                Fts = []
                for i in range(len(ind)-1):
                    if hit[i] == 1:
                        if j == 0:
                            stp = np.where(tsta > rt[i])[0]
                            if len(stp)>0:
                                stp = stp[0]
                            else:
                                stp = ind[i+1]
                            inds = np.arange(ind[i],ind[i]+stp);
                            inds[inds > df.shape[1]] = df.shape[1]
                            inds[inds >= df.shape[1]] = df.shape[1]-1
                        elif j == 1:
                            stp = np.where(tsta > rt[i])[0]
                            if len(stp)>0:
                                stp = stp[0]
                            else:
                                stp = ind[i+1]
                            inds = np.arange(ind[i]-two[0],ind[i]-two[1]);
                            inds[inds >= df.shape[1]] = df.shape[1]-1
                        Fts.append(df[:,inds]);
                    else:
                        Fts.append(df[:,ind[i]:ind[i+1]])
                Fepoch.append(Fts)
            tun = np.stack([np.nanmean(x,1) for x in Fepoch[0]])
            cn_tuning = tun[:,cn]
    
            import numpy as np
            import statsmodels.api as sm
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler
            
            # Define action and regressors
            a_t = cn_tuning
            delta_a = np.diff(a_t)
            thr = BCI_thresholds[1,:]
            if len(thr) < len(rt):
                thr2 = 0*rt;
                thr2[0:len(thr)] = thr
                thr2[len(thr):] = thr[-1]
            else:
                thr2 = thr
                
            rpe = compute_rpe(rt,4,1,np.nan)
            dhit = -compute_rpe(hit,1,5,np.nan)
            X = np.column_stack([
                rt[:-1],                  # reward time
                np.diff(rt),              # change in reward time
                hit[:-1].astype(int),     # hit on trial t
                dhit[:-1],  # miss on trial t
                thr2[:-1]    # threshold
            ])[:-1, :]  # Drop last trial to match delta_a
            X[np.isnan(X)] = 0
            
            # Normalize
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            X_with_intercept = sm.add_constant(X_normalized)
            
            # Shifted model: Predict **future ΔCN**
            future_delta_a = delta_a[1:]
            X_future = X_with_intercept[:-1, :]  # Align with future_delta_a
            
            # Fit model
            model = sm.OLS(future_delta_a, X_future).fit()
            if future_delta_a.ndim == 1 and future_delta_a.size > 1:
                ALLX.append(X_future)
                ALLCN.append(future_delta_a)
            else:
                print(f"Skipping session {mouse}, {session} due to insufficient data (future_delta_a size: {future_delta_a.size})")

            
            # Plot coefficients
            coefs = model.params[1:]  # Skip intercept
            pvals = model.pvalues[1:]
            var_names = ['rt', 'Δrt', 'hit', 'dhit', 'threshold']
            
            plt.figure(figsize=(8, 5))
            bars = plt.bar(var_names, coefs, color='gray')
            COEFS.append(coefs)
            for i, p in enumerate(pvals):
                if p < 0.05:
                    bars[i].set_color('red')
            plt.axhline(0, color='black', linewidth=1)
            plt.ylabel('Standardized Regression Coefficient')
            plt.title('Predictors of Future ΔCN Activity (Significant in Red)')
            plt.show()
            
            C, P = [], []
            for lag in range(10):
                if lag == 0:
                    model = sm.OLS(future_delta_a, X_future).fit()
                else:
                    model = sm.OLS(future_delta_a[lag:], X_future[:-lag, :]).fit()
                C.append(model.rsquared)       # Overall model fit
                P.append(model.f_pvalue)      # Overall p-value
            
            PVAL.append(-np.log(P))
            plt.figure(figsize=(8, 5))
            plt.plot(range(10), -np.log(P))
            plt.xlabel('Trials back (lag)')
            plt.ylabel('-log(P)')
            plt.title('Significance Across Lags (Future ΔCN)')
            plt.show()
        except Exception as e:
            print(f"Error in session {mouse}, {session} (sii={sii}): {e}")
            continue
#%%
# Stack all sessions together
# Stack all sessions together
X = np.vstack(ALLX)
CN = np.concatenate(ALLCN)

# Replace NaNs (optional)
CN[np.isnan(CN)] = 0
X[np.isnan(X)] = 0

import statsmodels.api as sm
import matplotlib.pyplot as plt

C_agg, P_agg = [], []
max_lag = 10

for lag in range(max_lag):
    if lag == 0:
        model = sm.OLS(CN, X).fit()
    else:
        model = sm.OLS(CN[lag:], X[:-lag, :]).fit()
    C_agg.append(model.rsquared)
    P_agg.append(model.f_pvalue)

plt.figure(figsize=(8, 5))
plt.plot(range(max_lag), -np.log(P_agg),'k.-')
plt.xlabel('Trials back (lag)')
plt.ylabel('-log(P)')
plt.title('Aggregate Significance Across Lags (Future ΔCN)')
plt.plot(plt.xlim(),(-np.log(.05),-np.log(.05)),'k:')
plt.show()

# If desired, you can also print zero-lag model summary:
print(sm.OLS(CN, X).fit().summary())
