import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def unflatted_neurons_by_groups(flattened, n_neurons):
    return flattened.reshape((-1, n_neurons)).T

def compute_photostim_distances(x_flat, n_neurons):
    return unflatted_neurons_by_groups(x_flat, n_neurons)

def find_first_valid_pair(data_dict):
    for i in range(len(data_dict['data']['x']) - 1):
        try:
            x1 = data_dict['data']['x'][i]
            x2 = data_dict['data']['x'][i+1]
            Fstim1 = data_dict['data']['Fstim'][i]
            Fstim2 = data_dict['data']['Fstim'][i+1]
            if Fstim1.shape[2] != Fstim2.shape[2]:
                continue
            seq1 = data_dict['data']['seq'][i]
            seq2 = data_dict['data']['seq'][i+1]
            trace_corr1 = data_dict['data']['trace_corr'][i]
            trace_corr2 = data_dict['data']['trace_corr'][i+1]
            if all(obj is not None for obj in [x1, x2, Fstim1, Fstim2, seq1, seq2, trace_corr1, trace_corr2]):
                return i, i + 1
        except:
            continue
    raise ValueError("No valid session pairs found.")

def extract_analysis_variables(data_dict):
    D_NEAR = 30
    D_FAR = 100
    D_DIRECT = 30

    exemplar_group_idx = 7

    day_1_idx, day_2_idx = 10, 11  # manually specified to ensure 716 neurons

    ps_fs_1 = data_dict['data']['Fstim'][day_1_idx]
    ps_fs_2 = data_dict['data']['Fstim'][day_2_idx]

    d_ps_flat_1 = data_dict['data']['x'][day_1_idx]
    d_ps_flat_2 = data_dict['data']['x'][day_2_idx]

    d_ps_1 = compute_photostim_distances(d_ps_flat_1, ps_fs_1.shape[1])
    d_ps_2 = compute_photostim_distances(d_ps_flat_2, ps_fs_2.shape[1])

    d_ps = d_ps_1
    ps_events_group_idxs_1 = data_dict['data']['seq'][day_1_idx]
    ps_events_group_idxs_2 = data_dict['data']['seq'][day_2_idx]

    ps_resp_1 = np.mean(ps_fs_1[4:9, :, :], axis=0)
    ps_resp_2 = np.mean(ps_fs_2[4:9, :, :], axis=0)

    resp_ps_plot_1 = ps_resp_1[:, ps_events_group_idxs_1 == (exemplar_group_idx + 1)]
    resp_ps_plot_2 = ps_resp_2[:, ps_events_group_idxs_2 == (exemplar_group_idx + 1)]

    if resp_ps_plot_1.shape[1] == 0 or resp_ps_plot_2.shape[1] == 0:
        raise ValueError(f"Group {exemplar_group_idx} is not present or trial counts do not match.")

    change_in_resp_ps = np.mean(resp_ps_plot_2, axis=1) - np.mean(resp_ps_plot_1, axis=1)

    pairwise_corrs_1 = data_dict['data']['trace_corr'][day_1_idx]
    pairwise_corrs_2 = data_dict['data']['trace_corr'][day_2_idx]
    pairwise_corrs_1 = np.nan_to_num(pairwise_corrs_1)
    pairwise_corrs_2 = np.nan_to_num(pairwise_corrs_2)

    direct_mask = d_ps[:, exemplar_group_idx] < D_DIRECT
    indirect_mask = (d_ps[:, exemplar_group_idx] > D_NEAR) & (d_ps[:, exemplar_group_idx] < D_FAR)
    indirect_idxs = np.where(indirect_mask)[0]

    group_cors_1 = np.mean(pairwise_corrs_1[:, direct_mask], axis=1)
    group_cors_2 = np.mean(pairwise_corrs_2[:, direct_mask], axis=1)
    change_in_cors = group_cors_2 - group_cors_1

    x = change_in_cors[indirect_idxs]
    y = change_in_resp_ps[indirect_idxs]

    slopes = np.full(d_ps.shape[0], np.nan)
    for i in indirect_idxs:
        xi = change_in_cors[i]
        yi = change_in_resp_ps[i]
        if np.isfinite(xi) and np.isfinite(yi):
            slopes[i] = yi / xi if xi != 0 else np.nan

    mean_indir_response = np.full(d_ps.shape[0], np.nan)
    mean_indir_response[indirect_idxs] = y

    indirect_params = d_ps[indirect_idxs, exemplar_group_idx]
    return d_ps, indirect_idxs, exemplar_group_idx, slopes, mean_indir_response, x, y, indirect_params

def make_summary_plot(d_ps, indirect_idxs, exemplar_group_idx, slopes, mean_indir_response, x, y):
    indirect_params = d_ps[indirect_idxs, exemplar_group_idx]

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(indirect_params, slopes[indirect_idxs])
    plt.xlabel('Distance to stim group')
    plt.ylabel('Slope')
    plt.title('Distance vs slope')

    plt.subplot(122)
    plt.scatter(indirect_params, mean_indir_response[indirect_idxs])
    plt.xlabel('Distance to stim group')
    plt.ylabel('Mean indirect response')
    plt.title('Distance vs mean response')

    plt.tight_layout()
    plt.show()
