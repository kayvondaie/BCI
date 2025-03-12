import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats
from sklearn.cluster import KMeans
import os, time

# --- Assume data is already loaded into a dictionary called 'data' ---

# Construct ROI from centroids.
roi = np.column_stack((data['centroidX'], data['centroidY']))

# Sampling interval (in seconds)
dt_si = data['dt_si']

F = data['F']
df_closedloop = data['df_closedloop'].T
tsta = np.arange(0,F.shape[0]*dt_si,dt_si)
tsta = tsta - tsta[117]
F = data['F']
trl = F.shape[2]
vel = []
rew = []
fstack = []
for ti in range(trl):
    steps = data['step_time'][ti]
    indices = np.searchsorted(tsta, steps)
    indices = indices[indices<800]
    v = np.zeros(len(tsta))
    v[indices] = 1;    
    vel.append(v[117:])
    
    steps = data['reward_time'][ti]
    indices = np.searchsorted(tsta, steps)
    indices = indices[indices<800]
    v = np.zeros(len(tsta))
    v[indices] = 1;    
    rew.append(v[117:])
    fstack.append(F[117:,:,ti])
df_closedloop = np.concatenate(fstack).T
vel = np.concatenate(vel)
rew = np.concatenate(rew)


# Session and subject information (assuming these are strings or arrays of strings)
sess_date = data['session']
mouse = data['mouse']
print("Session:", sess_date)
print("Mouse:", mouse)

# Use provided connectivity matrices.
# If you want to recompute trace_corr from df_closedloop, you could do so.
# Here we assume trace_corr is precomputed.
trace_corr = data['trace_corr']
# Ensure the diagonal is zero.
np.fill_diagonal(trace_corr, 0)



# --- Identify velocity events and construct a time window around them ---
window_width_one_side = 0.3  # seconds
vel_events = np.nonzero(vel)[0]
n_timesteps = len(vel)
vel_events_window = np.zeros(n_timesteps, dtype=bool)
dw = int(round(window_width_one_side / dt_si))
print('dw:', dw)
for k in vel_events:
    start = max(0, k - dw)
    end = min(n_timesteps, k + dw + 1)
    vel_events_window[start:end] = True

# --- Compute correlation matrices ---
# trace_corr is already available above.
# Compute trace_corr_at_vel using only timesteps around velocity events.
trace_corr_at_vel = np.corrcoef(df_closedloop[:, vel_events_window])
np.fill_diagonal(trace_corr_at_vel, 0)
trace_corr_at_vel[np.isnan(trace_corr_at_vel)] = 0
trace_corr[np.isnan(trace_corr)] = 0
# Match magnitudes between the two connectivity matrices.
scale_coef = np.abs(trace_corr).sum() / np.abs(trace_corr_at_vel).sum()


# Form matrix J.
J = (trace_corr - scale_coef * trace_corr_at_vel).T.dot(trace_corr - scale_coef * trace_corr_at_vel)

# --- Eigen decomposition and selection of optimal vectors ---
eigval, eigvec = np.linalg.eigh(J)
n_vec = 5
# Get the top n_vec eigenvectors (largest eigenvalues)
opt_vec = eigvec[:, ::-1][:, :n_vec]

# --- Process optimal vectors ---
n_clusters = 4
n_top = 40
n_neurons = opt_vec.shape[0]




# --- Process positive parts ---
opt_vec_pos_trun = opt_vec.copy()
opt_vec_pos_trun[opt_vec_pos_trun < 0] = 0
for k in range(opt_vec_pos_trun.shape[1]):
    tmp = np.argsort(opt_vec_pos_trun[:, k])[::-1][n_top:]
    opt_vec_pos_trun[tmp, k:k+1] = 0
opt_vec_pos_trun /= np.linalg.norm(opt_vec_pos_trun, axis=0)

# Cluster the positive parts spatially.
opt_vec_pos_spat_near = np.zeros((n_neurons, n_vec, n_clusters))
for k in range(n_vec):
    mask = opt_vec_pos_trun[:, k] > 0
    mask_idx = np.nonzero(mask)[0]
    if mask_idx.size == 0:
        print(f"No neurons with positive values for eigenvector {k}. Skipping clustering for this vector.")
        continue  # or assign zeros and move on
    X = roi[mask]
    
    # If the number of available neurons is less than n_clusters, adjust n_clusters accordingly.
    n_clusters_current = n_clusters if X.shape[0] >= n_clusters else X.shape[0]
    
    # Run kmeans only if we have at least 1 sample (which we do by this check)
    kmeans = KMeans(n_clusters=n_clusters_current, random_state=0).fit(X)
    
    for l in range(n_clusters_current):
        indices = mask_idx[kmeans.labels_ == l]
        opt_vec_pos_spat_near[indices, k, l] = opt_vec_pos_trun[indices, k]

# Normalize if there was any clustering done; otherwise, leave as zeros.
norm_factors = np.linalg.norm(opt_vec_pos_spat_near, axis=0)
norm_factors[norm_factors == 0] = 1  # avoid division by zero
opt_vec_pos_spat_near /= norm_factors






# --- Process negative parts ---
opt_vec_neg_trun = opt_vec.copy()
opt_vec_neg_trun[opt_vec_neg_trun > 0] = 0
opt_vec_neg_trun = -opt_vec_neg_trun
for k in range(opt_vec_neg_trun.shape[1]):
    tmp = np.argsort(opt_vec_neg_trun[:, k])[::-1][n_top:]
    opt_vec_neg_trun[tmp, k:k+1] = 0
opt_vec_neg_trun /= np.linalg.norm(opt_vec_neg_trun, axis=0)

opt_vec_neg_spat_near = np.zeros((n_neurons, n_vec, n_clusters))
for k in range(n_vec):
    mask = opt_vec_neg_trun[:, k] > 0
    mask_idx = np.nonzero(mask)[0]
    if mask_idx.size == 0:
        print(f"No neurons with negative values for eigenvector {k}. Skipping clustering for this vector.")
        continue
    X = roi[mask]
    
    n_clusters_current = n_clusters if X.shape[0] >= n_clusters else X.shape[0]
    kmeans = KMeans(n_clusters=n_clusters_current, random_state=0).fit(X)
    
    for l in range(n_clusters_current):
        indices = mask_idx[kmeans.labels_ == l]
        opt_vec_neg_spat_near[indices, k, l] = opt_vec_neg_trun[indices, k]

norm_factors = np.linalg.norm(opt_vec_neg_spat_near, axis=0)
norm_factors[norm_factors == 0] = 1
opt_vec_neg_spat_near /= norm_factors



# --- Combine positive and negative parts ---
opt_vec_spat_near = np.stack([opt_vec_pos_spat_near, opt_vec_neg_spat_near], axis=-1).reshape(n_neurons, -1)
n_opt_vec_spat_near = opt_vec_spat_near.shape[1]

# --- Compute spatial distance metrics ---
pixel_to_micron_const = 1.25
dist_mat = np.linalg.norm(roi[:, None, :] - roi[None, :, :], axis=2) * pixel_to_micron_const
near_thr = 20  # microns

dist_to_opt_vec = np.zeros((n_opt_vec_spat_near, n_neurons))
for k in range(n_opt_vec_spat_near):
    mask = opt_vec_spat_near[:, k] > 0
    if np.any(mask):
        dist_to_opt_vec[k] = dist_mat[mask].min(axis=0)
    else:
        dist_to_opt_vec[k] = np.inf

# --- Compute responses to the optimal vectors ---
y_trace_corr_opt_vec_spat_near = trace_corr.dot(opt_vec_spat_near)
y_trace_corr_at_vel_opt_vec_spat_near = scale_coef * trace_corr_at_vel.dot(opt_vec_spat_near)

# Identify non-directly stimulated neurons (distance >= near_thr).
dstim_mask = dist_to_opt_vec < near_thr

y_trace_corr_opt_vec_spat_near_non_stim = np.zeros_like(y_trace_corr_opt_vec_spat_near)
for k in range(n_opt_vec_spat_near):
    y_trace_corr_opt_vec_spat_near_non_stim[~dstim_mask[k], k:k+1] = \
        y_trace_corr_opt_vec_spat_near[~dstim_mask[k], k:k+1]

y_trace_corr_at_vel_opt_vec_spat_near_non_stim = np.zeros_like(y_trace_corr_at_vel_opt_vec_spat_near)
for k in range(n_opt_vec_spat_near):
    y_trace_corr_at_vel_opt_vec_spat_near_non_stim[~dstim_mask[k], k:k+1] = \
        y_trace_corr_at_vel_opt_vec_spat_near[~dstim_mask[k], k:k+1]

# Normalize responses.
y_trace_corr_opt_vec_spat_near_non_stim_normed = (
    y_trace_corr_opt_vec_spat_near_non_stim /
    np.linalg.norm(y_trace_corr_opt_vec_spat_near_non_stim, axis=0)
)
y_trace_corr_at_vel_opt_vec_spat_near_non_stim_normed = (
    y_trace_corr_at_vel_opt_vec_spat_near_non_stim /
    np.linalg.norm(y_trace_corr_at_vel_opt_vec_spat_near_non_stim, axis=0)
)

# --- Compute dot product discriminability metric ---
dot_prod = 1 - (y_trace_corr_opt_vec_spat_near_non_stim_normed *
                y_trace_corr_at_vel_opt_vec_spat_near_non_stim_normed).sum(0)
sorted_opt_vec_idxs = np.argsort(dot_prod)[::-1]
n_top_to_keep = 10
opt_pert = opt_vec_spat_near[:, sorted_opt_vec_idxs[:n_top_to_keep]]
opt_pert_dot_prod = dot_prod[sorted_opt_vec_idxs[:n_top_to_keep]]

# --- Visualize the optimal perturbations ---
for k in range(n_top_to_keep):
    print('<opt vec {}>'.format(k+1))
    cur_idx = sorted_opt_vec_idxs[k]
    print('dot prod: {:.4f}'.format(dot_prod[cur_idx]))
    print('')
    
    fig, ax = plt.subplots(figsize=(5, 5))
    min_coord = roi.min()
    max_coord = roi.max()
    pad = (max_coord - min_coord) * 0.05
    ax.set_aspect('equal')
    ax.set_xlim(min_coord - pad, max_coord + pad)
    ax.set_ylim(min_coord - pad, max_coord + pad)
    
    cur_opt_vec = opt_vec_spat_near[:, cur_idx]
    tmp = cur_opt_vec / cur_opt_vec.max() * 0.7  # scale for color intensity
    for m in np.nonzero(tmp)[0]:
        ax.scatter(roi[m, 0], roi[m, 1],
                   marker='o', s=25,
                   facecolor=plt.get_cmap('Reds')(tmp[m]),
                   edgecolor='none')
    plt.show()

# --- Save results if desired ---
np.save(folder + '/opt_pert_all_sess.npy', opt_pert)
np.save(folder + './opt_pert_dot_prod_all_sess.npy', opt_pert_dot_prod)

# # --- Plot dot product discriminability metric ---
# fig, ax = plt.subplots()
# ax.errorbar(np.arange(n_top_to_keep), opt_pert_dot_prod.mean(0),
#             yerr=opt_pert_dot_prod.std(0))
# ax.set_xlabel('top pert')
# ax.set_ylabel('dot prod')
# plt.show()
