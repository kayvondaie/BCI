import numpy as np
import matplotlib.pyplot as plt


def compute_connectivity_matrices(bci_data):
    """
    Processes BCI data to compute overall and velocity-specific
    functional connectivity matrices.

    Args:
        bci_data (dict): The loaded BCI data dictionary.

    Returns:
        tuple: A tuple containing:
            - trace_corr (np.ndarray): Overall (N,N) correlation matrix.
            - trace_corr_at_vel (np.ndarray): (N,N) correlation matrix during movement.
            - scale_coef (float): Scaling coefficient to match matrix magnitudes.
    """
    print("--- Computing connectivity matrices from BCI data ---")

    df_closedloop = bci_data['df_closedloop']
    dt_si = bci_data['dt_si']

    # --- Compute trace_corr ---
    trace_corr = np.corrcoef(df_closedloop)
    np.fill_diagonal(trace_corr, 0)
    trace_corr[np.isnan(trace_corr)] = 0

    trial_start = bci_data['trial_start']
    step_time = bci_data['step_time']
    num_trials = len(trial_start)

    step_time_from_sess_start = []
    for ti in range(num_trials):
        step_time_from_sess_start.extend(trial_start[ti] + step_time[ti])

    step_idx_from_sess_start = np.round(np.array(step_time_from_sess_start) / dt_si).astype(int)

    window_width_sec = 0.3
    dw = int(round(window_width_sec / dt_si))
    n_neurons, n_frames = df_closedloop.shape

    vel_events_window = np.zeros(n_frames, dtype=bool)
    for k in step_idx_from_sess_start:
        start = max(0, k - dw)
        end = min(n_frames, k + dw + 1)
        vel_events_window[start:end] = True

    trace_corr_at_vel = np.corrcoef(df_closedloop[:, vel_events_window])
    np.fill_diagonal(trace_corr_at_vel, 0)
    trace_corr_at_vel[np.isnan(trace_corr_at_vel)] = 0

    scale_coef = np.abs(trace_corr).sum() / np.abs(trace_corr_at_vel).sum()

    print("Connectivity matrices computed successfully.\n")
    return trace_corr, trace_corr_at_vel, scale_coef


def analyze_photostim_responses(
    stim_data, dt_si, trace_corr, trace_corr_at_vel, scale_coef, mouse_id=None
):
    """
    Analyzes photostimulation responses with consistent calculation methods.
    - applied_pert_vectors: Response of directly stimulated neurons.
    - actual_response: Response of non-directly stimulated neurons.

    Args:
        stim_data (dict): The loaded photostimulation data dictionary.
        dt_si (float): The sampling interval from the BCI data.
        trace_corr (np.ndarray): Overall connectivity matrix.
        trace_corr_at_vel (np.ndarray): Movement-specific connectivity matrix.
        scale_coef (float): Scaling coefficient.
        mouse_id (str, optional): Mouse ID for stimulus timing adjustments (e.g., 'BCI103').
    """
    print("--- Analyzing photostimulation responses (consistent calculation) ---")

    favg = stim_data['favg']  # shape: (Time, Neurons, Groups)
    umPerPix = 1.63
    stim_dist = stim_data['stimDist'] * umPerPix

    num_neurons, num_groups = stim_dist.shape
    num_frames = favg.shape[0]

    direct_stim_mask = stim_dist < 10  # shape (N, G)
    indirect_stim_mask = stim_dist > 30  # shape (N, G)

    applied_pert_vectors = np.zeros((num_neurons, num_groups))
    actual_response = np.zeros((num_neurons, num_groups))

    print("Calculating stimulation responses per group...")

    for k in range(num_groups):
        favg_k = favg[:, :, k].copy()

        artifact_indices = np.arange(10, 16)
        after_frames = int(np.floor(0.2 / dt_si))
        before_frames = int(np.floor(0.2 / dt_si))

        pre_win = (int(artifact_indices[0] - before_frames), int(artifact_indices[0] - 2))
        post_win = (int(artifact_indices[-1] + 2), int(artifact_indices[-1] + after_frames))

        if pre_win[0] < 0 or post_win[1] > num_frames:
            print(f"Warning: Analysis window out of bounds for group {k}. Skipping.")
            applied_pert_vectors[:, k] = np.nan
            actual_response[:, k] = np.nan
            continue

        # Interpolate through artifact
        favg_k[artifact_indices, :] = np.nan
        trace_to_interp = favg_k[0:30, :]
        interpolated_trace = np.apply_along_axis(
            lambda m: np.interp(
                np.arange(len(m)),
                np.where(~np.isnan(m))[0] if np.any(~np.isnan(m)) else [0],
                m[~np.isnan(m)] if np.any(~np.isnan(m)) else [0]
            ),
            axis=0,
            arr=trace_to_interp
        )
        favg_k[0:30, :] = interpolated_trace

        mean_pre = np.nanmean(favg_k[pre_win[0]:pre_win[1], :], axis=0)
        mean_post = np.nanmean(favg_k[post_win[0]:post_win[1], :], axis=0)
        amp_response_all_neurons = mean_post - mean_pre

        mask_k = direct_stim_mask[:, k]
        indirect_mask_k = indirect_stim_mask[:, k]
        applied_pert_vectors[mask_k, k] = amp_response_all_neurons[mask_k]
        actual_response[indirect_mask_k, k] = amp_response_all_neurons[indirect_mask_k]

    pred_resp_overall = trace_corr.dot(applied_pert_vectors)
    pred_resp_at_vel = scale_coef * trace_corr_at_vel.dot(applied_pert_vectors)
    print("Computed predicted responses.")

    nan_mask = np.isnan(actual_response) | np.isnan(applied_pert_vectors)
    pred_resp_overall[nan_mask] = 0
    pred_resp_at_vel[nan_mask] = 0
    actual_response[nan_mask] = 0

    pred_resp_overall[direct_stim_mask] = 0
    pred_resp_at_vel[direct_stim_mask] = 0

    def normalize_responses(response_matrix):
        norm = np.linalg.norm(response_matrix, axis=0)
        norm[norm == 0] = 1
        return response_matrix / norm

    pred_overall_normed = normalize_responses(pred_resp_overall)
    pred_at_vel_normed = normalize_responses(pred_resp_at_vel)
    actual_normed = normalize_responses(actual_response)

    print("Filtered and normalized all responses.")

    similarity_overall = np.sum(pred_overall_normed * actual_normed, axis=0)
    similarity_at_vel = np.sum(pred_at_vel_normed * actual_normed, axis=0)

    opt_indices = slice(0, 10)
    rand_indices = slice(10, 20)

    print("\n--- Comparison Results (Cosine Similarity) ---")
    print("Higher values indicate better prediction of the actual response pattern.\n")
    print("Model 1: Prediction based on overall connectivity (trace_corr)")
    print(f" - Avg. Similarity for Optimal Perturbations: {np.nanmean(similarity_overall[opt_indices]):.4f}")
    print(f" - Avg. Similarity for Random Perturbations: {np.nanmean(similarity_overall[rand_indices]):.4f}\n")
    print("Model 2: Prediction based on movement connectivity (trace_corr_at_vel)")
    print(f" - Avg. Similarity for Optimal Perturbations: {np.nanmean(similarity_at_vel[opt_indices]):.4f}")
    print(f" - Avg. Similarity for Random Perturbations: {np.nanmean(similarity_at_vel[rand_indices]):.4f}\n")

    labels = ['Optimal', 'Random']
    overall_means = [np.nanmean(similarity_overall[opt_indices]), np.nanmean(similarity_overall[rand_indices])]
    at_vel_means = [np.nanmean(similarity_at_vel[opt_indices]), np.nanmean(similarity_at_vel[rand_indices])]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, overall_means, width, label='Model: Overall Corr.')
    rects2 = ax.bar(x + width/2, at_vel_means, width, label='Model: At Step Corr.')

    ax.set_ylabel('Avg. Cosine Similarity')
    ax.set_title('Prediction Accuracy of Connectivity Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    fig.tight_layout()
    plt.show()


# Full paths to the data files on your I: drive
bci_data_path = 'I:/My Drive/Learning rules/BCI_data/optimal_perturbation_data/data_main_BCI109_040225_BCI.npy'
photostim_data_path = 'I:/My Drive/Learning rules/BCI_data/optimal_perturbation_data/data_photostimBCI109_040225.npy'

# Load the data with pickle enabled
bci_data = np.load(bci_data_path, allow_pickle=True)
stim_data = np.load(photostim_data_path, allow_pickle=True)


print("Successfully loaded BCI and photostimulation data.\n")


print("Successfully loaded BCI and photostimulation data.\n")

dt_si = bci_data['dt_si']
trace_corr, trace_corr_at_vel, scale_coef = compute_connectivity_matrices(bci_data)

analyze_photostim_responses(stim_data, dt_si, trace_corr, trace_corr_at_vel, scale_coef)
