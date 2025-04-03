import matplotlib.pyplot as plt
import numpy as np

def plot_cell_68():
    # def basic_paired_photostim_analysis(
    #     ps_stats_params, session_idxs, data_dict, 
    #     verbose=False,
    # ):
    #     """
    #     Conducts some basic analysis on photostim data

    #     INPUTS:
    #     session_idx_pairs: list of session pairs
    #     data_dict: loaded data file
    #     return_dict: compact dictionary for easy saving an futher plotting.

    #     """
    #     records = { # Fill with defaults used no matter what metrics
    #         'mice': [],
    #         'session_idxs': [],
    #         'ps_CC': [],
    #     }

    #     n_sessions = len(session_idxs)

    #     for session_idx_idx, session_idx in enumerate(session_idxs):

    #         records['session_idxs'].append(session_idx)
    #         records['mice'].append(data_dict['data']['mouse'][session_idx])

    #         data_to_extract = ('d_ps', 'resp_ps', 'trial_start_metrics', 'trial_start_fs', 'd_masks',)
    #         data = extract_session_data(session_idx, data_dict, data_to_extract, ps_stats_params)

    ps_stats_params = {
        'trial_average_mode': 'trials_first', 
        'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
        'mask_mode': 'constant', # constant, each_day, kayvon_match
        'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
        'neuron_metrics_adjust': None, # Normalize things like tuning before computing metrics

        ### Plotting/fitting parameters
        'fit_individual_sessions': True, # Fit each individual session too
        'connectivity_metrics': None,
        'plot_pairs': None,

        'plot_up_mode': None, # all, significant, None; what to actually create plots for 
    }

    which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
    session_idx_pairs = find_valid_ps_pairs(
        ps_stats_params, which_sessions_to_include, data_dict,
        verbose=False
    )

    if (20, 21) in session_idx_pairs:
        print('Removing session to match Kayvons sessions')
        session_idx_pairs.remove((20, 21))

    print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

    exemplar_pair_idx = 0
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))

    P_VALUE_THRESH = 0.05 # Threshold for significance

    # See how consistent direct responses are across pairs
    bin_mode = True
    d_direct_temps = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    #     d_direct_temps = [5, 10, 15, 20, 25, 30, 35, 40]

    n_direct_temps = len(d_direct_temps)
    if bin_mode: n_direct_temps -= 1 # Skip last

    d_direct_bins_rsquares = np.zeros((len(session_idx_pairs), n_direct_temps,))
    d_direct_bins_slopes = np.zeros((len(session_idx_pairs), n_direct_temps,))

    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
        day_2_idx = session_idx_pair[1]
        day_1_idx = session_idx_pair[0]

        print(f'Pair {pair_idx} - Sessions {day_1_idx} and {day_2_idx}.')

        ps_events_group_idxs_1 = data_dict['data']['seq'][day_1_idx] 
        ps_fs_1 = data_dict['data']['Fstim'][day_1_idx] # (ps_times, n_neurons, n_ps_events,)
        ps_events_group_idxs_2 = data_dict['data']['seq'][day_2_idx]
        ps_fs_2 = data_dict['data']['Fstim'][day_2_idx] # (ps_times, n_neurons, n_ps_events,)

    #     assert ps_fs_1.shape[0] == ps_fs_2.shape[0] # Same ps_times
        assert ps_fs_1.shape[1] == ps_fs_2.shape[1] # Same number of neurons

        n_neurons = ps_fs_1.shape[1]
        n_groups = int(np.max(ps_events_group_idxs_1))

        d_ps_flat_1 = data_dict['data']['x'][day_1_idx]
        d_ps_1 = unflatted_neurons_by_groups(d_ps_flat_1, n_neurons,)
        d_ps_flat_2 = data_dict['data']['x'][day_2_idx]
        d_ps_2 = unflatted_neurons_by_groups(d_ps_flat_2, n_neurons,)

        resp_ps_1, resp_ps_extras_1 = compute_resp_ps_mask_prevs(
            ps_fs_1, ps_events_group_idxs_1, d_ps_1, ps_stats_params, return_extras=False
        )
        resp_ps_2, resp_ps_extras_2 = compute_resp_ps_mask_prevs(
            ps_fs_2, ps_events_group_idxs_2, d_ps_2, ps_stats_params, return_extras=False
        )

        # Scan over distances
        for d_direct_temp_idx in range(n_direct_temps):

            d_direct_temp = d_direct_temps[d_direct_temp_idx]

            direct_resp_ps_1 = []
            direct_resp_ps_2 = []
            direct_d_ps_mean = []

            for neuron_idx in range(n_neurons):

                if bin_mode:
                    direct_idxs = np.where(np.logical_and(
                        np.logical_and(d_ps_1[neuron_idx, :] >= d_direct_temp, d_ps_1[neuron_idx, :] < d_direct_temps[d_direct_temp_idx+1]),
                        np.logical_and(d_ps_2[neuron_idx, :] >= d_direct_temp, d_ps_2[neuron_idx, :] < d_direct_temps[d_direct_temp_idx+1]),
                    ))[0]
                else:
                    direct_idxs = np.where(np.logical_and(
                        d_ps_1[neuron_idx, :] < d_direct_temp, d_ps_2[neuron_idx, :] < d_direct_temp
                    ))[0]

                # Take distance to be mean between two sessions
                d_ps_mean = np.mean(np.concatenate(
                    (d_ps_1[neuron_idx:neuron_idx+1, direct_idxs], d_ps_2[neuron_idx:neuron_idx+1, direct_idxs]), 
                    axis=0), axis=0)

                direct_resp_ps_1.append(resp_ps_1[neuron_idx, direct_idxs])
                direct_resp_ps_2.append(resp_ps_2[neuron_idx, direct_idxs])
                direct_d_ps_mean.append(d_ps_mean)

            direct_resp_ps_1 = np.concatenate(direct_resp_ps_1, axis=0)
            direct_resp_ps_2 = np.concatenate(direct_resp_ps_2, axis=0)
            direct_d_ps_mean = np.concatenate(direct_d_ps_mean, axis=0)

            ax = None
            if exemplar_pair_idx == pair_idx:
                ax = ax1
                ax1.scatter(direct_resp_ps_1, direct_resp_ps_2, c=direct_d_ps_mean, marker='.', 
                            vmin=0.0, vmax=30, cmap='viridis', alpha=0.5)

            slope, _, rvalue, _, _ = add_regression_line(direct_resp_ps_1, direct_resp_ps_2,  ax=ax, color='k', zorder=1)

            d_direct_bins_rsquares[pair_idx, d_direct_temp_idx] = rvalue**2
            d_direct_bins_slopes[pair_idx, d_direct_temp_idx] = slope
    #         if bin_mode:
    #             print('D_dir {} to {} um - slope {:.2f}\tr*2: {:.2f}'.format(
    #                 d_direct_temp, d_direct_temps[d_direct_temp_idx+1], slope, rvalue**2
    #             ))
    #         else:
    #             print('D_dir {} um - slope {:.2f}\tr*2: {:.2f}'.format(d_direct_temp, slope, rvalue**2))

    ax1.set_xlabel('Day 1 Direct Resp.')
    ax1.set_ylabel('Day 2 Direct Resp.')
    # ax1.legend()

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    fig2, ax3 = plt.subplots(1, 1, figsize=(6, 4))

    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):

        if pair_idx > 9:
            linestyle = 'dashed'
        else:
            linestyle = 'solid'

        ax2.plot(d_direct_bins_rsquares[pair_idx], color=c_vals[pair_idx%9], linestyle=linestyle)
        ax3.plot(d_direct_bins_slopes[pair_idx], color=c_vals[pair_idx%9], linestyle=linestyle)

    for ax in (ax2, ax3):    
        ax.set_xlabel('Distance to photostim group (um)')
        ax.set_xticks(np.arange(n_direct_temps))
        ax.set_xticklabels((
            '0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40',
        ), rotation=45)

        ax.axvline(5.5, color='grey', zorder=-5)

    ax2.set_ylabel('$r^2$ of Day 1 vs. Day 2')
    ax3.set_ylabel('slope of Day 1 vs. Day 2')

    ax3.axhline(1.0, color='grey', zorder=-5)

if __name__ == '__main__':
    plot_cell_68()
