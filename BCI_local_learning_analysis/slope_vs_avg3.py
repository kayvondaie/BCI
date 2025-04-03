import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'BCI_local_learning_analysis'))
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import scipy.stats as stats
import io
from helper_functions1 import *
from helper_functions2 import *

try:
    import mat73
except:
    !pip install mat73
    import mat73

mypath = 'H:/My Drive/Learning rules/BCI_data/combined_new_old_060524.mat'
data_dict = mat73.loadmat(mypath)



#%%
from sklearn.decomposition import PCA  # unused
import copy

session_idx = 12
exemplar_group_idx = 17

shuffle_events = True  # unused

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1,
    'direct_predictor_mode': 'sum',
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans',
    'direct_input_mode': 'average',
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

ps_events_group_idxs = data_dict['data']['seq'][session_idx]
ps_fs = data_dict['data']['Fstim'][session_idx]

n_ps_times = ps_fs.shape[0]
n_neurons = ps_fs.shape[1]
n_groups = int(np.max(ps_events_group_idxs))

d_ps_flat = data_dict['data']['x'][session_idx]
d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons)

resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=False
)
resp_ps_events = resp_ps_extras['resp_ps_events']

group_event_slope = np.zeros((n_groups,))
group_event_rsquared = np.zeros((n_groups,))

group_idx = exemplar_group_idx
direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > 30, d_ps[:, group_idx] < 2000))[0]

dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :]
indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :]

n_direct = dir_resp_ps_events.shape[0]
n_indirect = indir_resp_ps_events.shape[0]

sum_dir_resp_ps_events = np.nansum(dir_resp_ps_events, axis=0, keepdims=True)
sum_dir_resp_ps_events = np.repeat(sum_dir_resp_ps_events, n_indirect, axis=0)

fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4,))
plot_ax = ax5

slope, _, rvalue, pvalue, _ = add_regression_line(
    sum_dir_resp_ps_events.flatten(), indir_resp_ps_events.flatten(), 
    fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=plot_ax, color='k', zorder=5
)

group_event_slope[group_idx] = slope
group_event_rsquared[group_idx] = rvalue**2

ps_stats_params_copy = copy.deepcopy(ps_stats_params)
ps_stats_params_copy['direct_predictor_mode'] = 'sum'
ps_stats_params_copy['n_direct_predictors'] = 1

direct_predictors, direct_shift, _ = find_photostim_variation_predictors(
    dir_resp_ps_events, ps_stats_params_copy, return_extras=True
)

indirect_params, pvals, _ = fit_photostim_variation(
    dir_resp_ps_events,
    indir_resp_ps_events,
    direct_predictors,
    direct_shift,
    ps_stats_params_copy
)

slope_idx = 1 if ps_stats_params_copy['direct_predictor_intercept_fit'] else 0
slopes = indirect_params[:, slope_idx]
pvals = pvals[:, slope_idx]
mean_indir_response = np.nanmean(indir_resp_ps_events, axis=1)

valid_mask = ~np.isnan(slopes) & ~np.isnan(mean_indir_response)
slopes = slopes[valid_mask]
mean_indir_response = mean_indir_response[valid_mask]

b = np.argsort(slopes)       
cl = b[-5]
bb = np.argsort(sum_dir_resp_ps_events[cl, :]) 
ind = np.where(ps_events_group_idxs - 1 == exemplar_group_idx)[0]
f_all = ps_fs[:, cl, ind].copy()

f_all[5:8, :] = np.nan

f_all[:10, :] = np.apply_along_axis(
    lambda m: np.interp(
        np.arange(len(m)),
        np.where(~np.isnan(m))[0] if np.any(~np.isnan(m)) else [0],
        m[~np.isnan(m)] if np.any(~np.isnan(m)) else [0]
    ),
    axis=0,
    arr=f_all[:10, :]
)

bl = np.nanmean(f_all[0:4, :])
f_all = (f_all - bl) / bl

num_top = 9
f_big = f_all[:, bb[-num_top:]]
bl = np.nanmean(f_big[0:4, :])
f_big = (f_big - bl)

dt_si = data_dict['data']['dt_si'][-1]
t_sta = np.arange(0, dt_si * f_all.shape[0], dt_si)
t_sta = t_sta - t_sta[8]

plt.show()
plt.subplot(121)
plt.plot(t_sta[0:18], np.nanmean(f_all[0:18], axis=1), 'b')       
plt.plot(t_sta[0:18], np.nanmean(f_big[0:18], axis=1), 'k')    
plt.xlabel('Time (s)')
plt.ylabel('$\Delta$F/F')   

plt.subplot(122)
plt.plot(sum_dir_resp_ps_events[cl, :], indir_resp_ps_events[cl, :], 'b.')
plt.plot(sum_dir_resp_ps_events[cl, bb[-num_top:]], indir_resp_ps_events[cl, bb[-num_top:]], 'k.')
plt.ylabel('$\Delta$F/F indir.')   
plt.xlabel('$\Delta$F/F dir.')   
plt.tight_layout()
