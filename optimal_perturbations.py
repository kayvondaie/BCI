# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:11:35 2023

@author: scanimage
"""

import os 
os.chdir('C:/Users/scanimage/Documents/Python Scripts/BCI_analysis/')

import data_dict_create_module as ddc
import numpy as np
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300

folder = r'D:/KD/BCI_data/BCI_2022/BCI45/050123/'
data = ddc.load_data_dict(folder)
rewT = data['reward_time']
steps = data['step_time']
#%%
dt = data['dt_si']
iscell = data['iscell']
df_closedloop = data['df_closedloop'].T
df_closedloop = df_closedloop[:,iscell[:,0]==1]
t = np.arange(0, dt * df_closedloop.shape[0], dt)
#t = t[0:-1]
vel = 0*t;
window_width_one_side = 0.3 # in sec. window_width before and after event.

for i in range(len(steps)):
    vel[np.where(t>steps[i])[0][0]] = 1

n_timesteps = len(vel)
# Correlation at time windows around vel events.
vel_events = np.nonzero(vel)[0]
vel_events_window = np.zeros(n_timesteps, dtype=bool)
dw = np.around(window_width_one_side/dt).astype(int)
print('dw:', dw)
for k in vel_events:
    vel_events_window[k-dw:k+dw+1] = True
vel_events_window[vel_events_window>len(df_closedloop)] = []

trace_corr = np.corrcoef(df_closedloop.T)
trace_corr[np.arange(len(trace_corr)),np.arange(len(trace_corr))] = 0

trace_corr_at_vel = np.corrcoef(df_closedloop[vel_events_window,:].T)
trace_corr_at_vel[np.arange(len(trace_corr_at_vel)),np.arange(len(trace_corr_at_vel))]=0

scale_coef = np.abs(trace_corr).sum()/np.abs(trace_corr_at_vel).sum()

J = (trace_corr - scale_coef*trace_corr_at_vel).T.dot(trace_corr - scale_coef*trace_corr_at_vel)

eigval, eigvec = np.linalg.eigh(J)


# Take top 5 opt vecs.
n_vec = 5
opt_vec = eigvec[:,::-1][:,:n_vec]

'''
1. Take pos (or neg) part of opt vecs.
2. we truncate each of them to the top 40 neurons with highest opt_vec weights.
3. Apply k-means clustering to the top 40 neurons with n_cluster = 4 (which seems appropriate for spatial 
constraint that stim neurons be within 500 micron distance given 1mm x 1mm FOV). 
Note that then each cluster is expected to have ~10 neurons, which is a reaonable number for holographic stim.
'''

n_clusters = 4
n_top = 40


'''
We first do this for pos part of opt vecs. We will repeat the same for neg part below.
'''

opt_vec_pos_trun = opt_vec.copy()
opt_vec_pos_trun[opt_vec_pos_trun < 0] = 0

for k in range(opt_vec_pos_trun.shape[1]):
    tmp = np.argsort(opt_vec_pos_trun[:,k])[::-1][n_top:]
    opt_vec_pos_trun[tmp,k:k+1] =  0
        
opt_vec_pos_trun /= np.linalg.norm(opt_vec_pos_trun, axis=0)



n_neurons = len(opt_vec)
opt_vec_pos_spat_near = np.zeros((n_neurons, n_vec, n_clusters))

for k in range(n_vec):
       
       mask = (opt_vec_pos_trun[:,k] > 0)
       mask_idx = np.nonzero(mask)[0]
       N = df_closedloop.shape[1]
       roi = np.zeros((N,2))
       roi[:,0] = data['centroidX']
       roi[:,1] = data['centroidY']     
       X = roi[mask,:]
   
       kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
       
       for l in range(n_clusters):
           
           opt_vec_pos_spat_near[mask_idx[kmeans.labels_==l],k:k+1,l:l+1] = \
           opt_vec_pos_trun[mask_idx[kmeans.labels_==l],k:k+1,None]
           
opt_vec_pos_spat_near /= np.linalg.norm(opt_vec_pos_spat_near, axis=0)


'''
Repeat the above for neg part.
'''

opt_vec_neg_trun = opt_vec.copy()
opt_vec_neg_trun[opt_vec_neg_trun > 0] = 0
opt_vec_neg_trun = -opt_vec_neg_trun

for k in range(opt_vec_neg_trun.shape[1]):
    tmp = np.argsort(opt_vec_neg_trun[:,k])[::-1][n_top:]
    opt_vec_neg_trun[tmp,k:k+1] =  0
    
opt_vec_neg_trun /= np.linalg.norm(opt_vec_neg_trun, axis=0)



n_neurons = len(opt_vec)
opt_vec_neg_spat_near = np.zeros((n_neurons, n_vec, n_clusters))

for k in range(n_vec):
    
    mask = (opt_vec_neg_trun[:,k] > 0)
    mask_idx = np.nonzero(mask)[0]
    X = roi[mask]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    
    for l in range(n_clusters):
        
        opt_vec_neg_spat_near[mask_idx[kmeans.labels_==l],k:k+1,l:l+1] = \
        opt_vec_neg_trun[mask_idx[kmeans.labels_==l],k:k+1,None]
        
opt_vec_neg_spat_near /= np.linalg.norm(opt_vec_neg_spat_near, axis=0)

'''
We will choose top few vectors from opt_vec_pos_spat and opt_vec_neg_trun based on their dot product
discriminability on non-directly stimulated neurons.
'''

opt_vec_spat_near = np.stack([opt_vec_pos_spat_near, opt_vec_neg_spat_near], -1).reshape(n_neurons, -1)

n_opt_vec_spat_near = opt_vec_spat_near.shape[1]



'''
dist_to_opt_vec[k], of length (n_neurons), represents the minimum distance of a neuron to k-th's stim
group. This will later be used for identifying non-directly stimulated neurons (i.e. those that are farther
than 20 micron from stim group), which are then used to compute the dot product discriminability.
'''
pixel_to_micron_const = 1.25

dist_mat = np.linalg.norm(roi[None] - roi[:,None], axis=2)*pixel_to_micron_const

near_thr = 20   

dist_to_opt_vec = np.zeros((n_opt_vec_spat_near, n_neurons))
for k in range(n_opt_vec_spat_near):
    mask = (opt_vec_spat_near[:,k] > 0)
    dist_to_opt_vec[k] = dist_mat[mask].min(0)


'''
y represents the neural responses to opt vecs when connectivity matrix is trace_corr 
or scale_coef*trace_corr_at_vel.
'''
y_trace_corr_opt_vec_spat_near = trace_corr.dot(opt_vec_spat_near)
y_trace_corr_at_vel_opt_vec_spat_near = scale_coef*trace_corr_at_vel.dot(opt_vec_spat_near)

'''
dstim_mask is True only on directly stimulated neurons. So we will use ~dstim_mask 
to identify non-directly stimulated neurons
'''
dstim_mask = dist_to_opt_vec < near_thr


'''
In y_{...}_non_stim, only neural responses of non-directly stimulated neurons are kept, and those of
directly stimulated neurons are set to zero.
'''
y_trace_corr_opt_vec_spat_near_non_stim = np.zeros_like(y_trace_corr_opt_vec_spat_near)
for k in range(n_opt_vec_spat_near):
    y_trace_corr_opt_vec_spat_near_non_stim[~dstim_mask[k],k:k+1] = \
    y_trace_corr_opt_vec_spat_near[~dstim_mask[k],k:k+1]

y_trace_corr_at_vel_opt_vec_spat_near_non_stim = np.zeros_like(y_trace_corr_at_vel_opt_vec_spat_near)
for k in range(n_opt_vec_spat_near):
    y_trace_corr_at_vel_opt_vec_spat_near_non_stim[~dstim_mask[k],k:k+1] = \
    y_trace_corr_at_vel_opt_vec_spat_near[~dstim_mask[k],k:k+1]

n_top_to_keep  = 10

'''
Normalize y_{...}_non_stim.
'''
y_trace_corr_opt_vec_spat_near_non_stim_normed = \
y_trace_corr_opt_vec_spat_near_non_stim/np.linalg.norm(y_trace_corr_opt_vec_spat_near_non_stim, axis=0)

y_trace_corr_at_vel_opt_vec_spat_near_non_stim_normed = \
y_trace_corr_at_vel_opt_vec_spat_near_non_stim/np.linalg.norm(y_trace_corr_at_vel_opt_vec_spat_near_non_stim, axis=0)    
        

'''
Compute dot_prod distance btw neural responses to opt perts (i.e. y) when connectivity matrix is 
trace_corr or trace_corr_at_vel. We want to maximize this, which is a measure of discriminability of the two 
hypotheses under opt perts. This is achieved by argsort(dot_prod)[::-1], which sorts opt perts in decreasing
order of their values of dot_prod.
'''
dot_prod = 1 - (y_trace_corr_opt_vec_spat_near_non_stim_normed*y_trace_corr_at_vel_opt_vec_spat_near_non_stim_normed).sum(0)

sorted_opt_vec_idxs = np.argsort(dot_prod)[::-1]

opt_pert_all_sess = opt_vec_spat_near[:,sorted_opt_vec_idxs[:n_top_to_keep]]
opt_pert_dot_prod_all_sess = dot_prod[sorted_opt_vec_idxs[:n_top_to_keep]]

#%%
'''
Visualize opt_pert.
'''    
for k in range(n_top_to_keep):
    print('<opt vec {}>'.format(k+1))
    cur_idx = sorted_opt_vec_idxs[k]
    print('dot prod: {:.4f}'.format(dot_prod[cur_idx]))
    print('')
    
    fig, ax = plt.subplots(figsize=(5, 5))

    min_coord = roi.min()
    max_coord = roi.max()

    pad = (max_coord - min_coord)*0.05


    ax.set_aspect('equal')

    ax.set_xlim(min_coord-pad, max_coord+pad)
    ax.set_ylim(min_coord-pad, max_coord+pad)


    cur_opt_vec = opt_vec_spat_near[:,cur_idx]

    tmp = cur_opt_vec/cur_opt_vec.max()*0.7 # tmp=1 is too dark that it looks black.

    for m in np.nonzero(tmp)[0]:
        ax.scatter(roi[m,0], roi[m,1],\
            marker='o', s=25, facecolor=plt.get_cmap('Reds')(tmp[m]), edgecolor='none')

    plt.show()
    new_folder = folder + 'optimal_perturbation/'      
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    filename = new_folder + r'optimal_perturbation_' + str(k)
    fig.savefig(filename)
    
from scipy.io import savemat
savemat(new_folder + 'vectors.mat',{'opt_vec_spat_near': opt_vec_spat_near})
np.save(new_folder + 'vectors.npy',opt_vec_spat_near);