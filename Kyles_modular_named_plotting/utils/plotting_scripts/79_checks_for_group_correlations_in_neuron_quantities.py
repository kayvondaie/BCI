import matplotlib.pyplot as plt
import numpy as np

def plot_cell_79():
    def check_for_group_corrs(
        neuron_metric, ps_stats_params, session_idx_pairs, data_dict, 
        verbose=False,
    ):
        """
        Create a compact function call to extract data from an intermediate processing step 
        to a recreation of the explainers of change of tuning.

        INPUTS:
        session_idx_pairs: list of session pairs
        data_dict: loaded data file
        return_dict: compact dictionary for easy saving an futher plotting.

        """
        records = { # Fill with defaults used no matter what metrics
            'ps_pair_idx': [],
            'ps_CC': [],
        }

        n_pairs = len(session_idx_pairs)

    #     fig3, (ax7, ax8, ax9) = plt.subplots(1, 3, figsize=(12, 4))
        fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))
        fig2, ((ax11, ax22), (ax33, ax44), (ax55, ax66)) = plt.subplots(3, 2, figsize=(12, 12))

        all_sums_x = []
        all_means_x = []
        all_sum_diffs_x = []
        all_mean_diffs_x = []
        all_stds_x = []

        all_sums_y = []
        all_means_y = []
        all_sum_diffs_y = []
        all_mean_diffs_y = []
        all_stds_y = []

        for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
            day_2_idx = session_idx_pair[1]
            day_1_idx = session_idx_pair[0]

            assert day_2_idx > day_1_idx
            assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

            data_to_extract = ('d_ps', 'resp_ps', 'trial_start_metrics', 'trial_start_fs', 'd_masks',)
            data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)
            data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params)

            indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
                ps_stats_params, data_1, data_2, verbose=verbose
            )

            n_neurons = data_1['resp_ps'].shape[0]
            n_groups = data_1['resp_ps'].shape[1]

            if neuron_metric == 'tuning':
                metric = 'tuning'
                neuron_metric_1 = data_1['tuning']
                neuron_metric_2 = data_2['tuning']
            elif neuron_metric == 'trial_resp':
                metric = 'trial_resp'
                neuron_metric_1 = data_1['trial_resp']
                neuron_metric_2 = data_2['trial_resp']
            elif neuron_metric == 'post':
                metric = 'post'
                neuron_metric_1 = data_1['post']
                neuron_metric_2 = data_2['post']
            elif neuron_metric == 'pre':
                metric = 'pre'
                neuron_metric_1 = data_1['pre']
                neuron_metric_2 = data_2['pre']
            else:
                raise NotImplementedError('Neuron metric {} not recognized.'.format(neuron_metric))

    #         print('Max combined:', np.max(indir_mask_weighted + dir_mask_weighted))
    #         dir_sim = np.matmul(dir_mask_weighted, dir_mask_weighted.T)
    #         indir_sim = np.matmul(indir_mask_weighted, indir_mask_weighted.T)
    #         far_sim = np.matmul(indir_mask_weighted + dir_mask_weighted, 
    #                             (indir_mask_weighted + dir_mask_weighted).T)

    #         dir_sim_flat = []
    #         indir_sim_flat = []
    #         far_sim_flat = []
    #         tuning_diff_flat = []
    #         for neuron_idx1 in range(dir_sim.shape[0]):
    #             for neuron_idx2 in range(neuron_idx1): # Not inclusive of same index
    #                 dir_sim_flat.append(dir_sim[neuron_idx1, neuron_idx2])
    #                 indir_sim_flat.append(indir_sim[neuron_idx1, neuron_idx2])
    #                 far_sim_flat.append(far_sim[neuron_idx1, neuron_idx2])
    #                 tuning_diff_flat.append(np.abs(neuron_metric_1[neuron_idx1] - neuron_metric_1[neuron_idx2]))

    #         for flat, ax in zip(
    #             (dir_sim_flat, indir_sim_flat, far_sim_flat), (ax7, ax8, ax9)
    #         ):
    #             ax.scatter(flat, tuning_diff_flat, marker='.', color='k')
    #             add_regression_line(flat, tuning_diff_flat, ax=ax, color=c_vals[pair_idx % 8])

    #         for ax in (ax7, ax8, ax9):
    #             ax.legend()

    #         fig3.show()
    #         print(sdfdsfsd)

            sum_dir_resp_1 = np.zeros((n_groups,))
            mean_dir_resp_1 = np.zeros((n_groups,))
            sum_indir_resp_1 = np.zeros((n_groups,))
            mean_indir_resp_1 = np.zeros((n_groups,))
            sum_dir_resp_2 = np.zeros((n_groups,))
            mean_dir_resp_2 = np.zeros((n_groups,))
            sum_indir_resp_2 = np.zeros((n_groups,))
            mean_indir_resp_2 = np.zeros((n_groups,))

            std_dir_resp_1 = np.zeros((n_groups,))
            std_dir_resp_2 = np.zeros((n_groups,))
            std_indir_resp_1 = np.zeros((n_groups,))
            std_indir_resp_2 = np.zeros((n_groups,))

            for group_idx in range(n_groups):
                group_mask_dir = dir_mask_weighted[:, group_idx] > 0
                group_mask_indir = indir_mask_weighted[:, group_idx] > 0

    #             print('Group {} counts - dir: {} indir: {}'.format(
    #                 group_idx, neuron_metric_1[group_mask_dir].shape[0],
    #                 neuron_metric_1[group_mask_indir].shape[0]
    #             ))
                if group_mask_dir.shape[0] == 0: # No neuron catch
                    sum_dir_resp_1[group_idx] = np.nan
                    mean_dir_resp_1[group_idx] = np.nan
                    sum_dir_resp_2[group_idx] = np.nan
                    mean_dir_resp_2[group_idx] = np.nan
                    std_dir_resp_1 = np.nan
                    std_dir_resp_2 = np.nan
                else:
                    sum_dir_resp_1[group_idx] = np.sum(neuron_metric_1[group_mask_dir])
                    mean_dir_resp_1[group_idx] = np.mean(neuron_metric_1[group_mask_dir])
                    sum_dir_resp_2[group_idx] = np.sum(neuron_metric_2[group_mask_dir])
                    mean_dir_resp_2[group_idx] = np.mean(neuron_metric_2[group_mask_dir])
                    std_dir_resp_1[group_idx] = np.std(neuron_metric_1[group_mask_dir])
                    std_dir_resp_2[group_idx] = np.std(neuron_metric_2[group_mask_dir])
                if group_mask_indir.shape[0] == 0: # No neuron catch
                    sum_indir_resp_1[group_idx] = np.nan
                    mean_indir_resp_1[group_idx] = np.nan
                    sum_indir_resp_2[group_idx] = np.nan
                    mean_indir_resp_2[group_idx] = np.nan
                    std_indir_resp_1[group_idx] = np.nan
                    std_indir_resp_2[group_idx] = np.nan
                else:
                    sum_indir_resp_1[group_idx] = np.sum(neuron_metric_1[group_mask_indir])
                    mean_indir_resp_1[group_idx] = np.mean(neuron_metric_1[group_mask_indir])
                    sum_indir_resp_2[group_idx] = np.sum(neuron_metric_2[group_mask_indir])
                    mean_indir_resp_2[group_idx] = np.mean(neuron_metric_2[group_mask_indir])
                    std_indir_resp_1[group_idx] = np.std(neuron_metric_1[group_mask_indir])
                    std_indir_resp_2[group_idx] = np.std(neuron_metric_2[group_mask_indir])

            if pair_idx == 0: # Plot these only for the very first pair index

                ax11.scatter(sum_dir_resp_1, sum_indir_resp_1, marker='.', color='k')
                ax22.scatter(sum_dir_resp_2, sum_indir_resp_2, marker='.', color='k')
                ax33.scatter(mean_dir_resp_1, mean_indir_resp_1, marker='.', color='k')
                ax44.scatter(mean_dir_resp_2, mean_indir_resp_2, marker='.', color='k')
                ax55.scatter(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, marker='.', color='k')
                ax66.scatter(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_indir_resp_1, marker='.', color='k')

                _ = add_regression_line(sum_dir_resp_1, sum_indir_resp_1, ax=ax11, color=c_vals[pair_idx])
                _ = add_regression_line(sum_dir_resp_2, sum_indir_resp_2, ax=ax22, color=c_vals[pair_idx])
                _ = add_regression_line(mean_dir_resp_1, mean_indir_resp_1, ax=ax33, color=c_vals[pair_idx])
                _ = add_regression_line(mean_dir_resp_2, mean_indir_resp_2, ax=ax44, color=c_vals[pair_idx])
                _ = add_regression_line(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, ax=ax55, color=c_vals[pair_idx])
                _ = add_regression_line(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_indir_resp_1, ax=ax66, color=c_vals[pair_idx])

            _ = add_regression_line(sum_dir_resp_1, sum_indir_resp_1, ax=ax1, color=c_vals[pair_idx % 8], label=None)
            _ = add_regression_line(sum_dir_resp_2, sum_indir_resp_2, ax=ax1, color=c_vals_l[pair_idx % 8], label=None)
            _ = add_regression_line(mean_dir_resp_1, mean_indir_resp_1, ax=ax3, color=c_vals[pair_idx % 8], label=None)
            _ = add_regression_line(mean_dir_resp_2, mean_indir_resp_2, ax=ax3, color=c_vals_l[pair_idx % 8], label=None)
            _ = add_regression_line(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, ax=ax5, color=c_vals[pair_idx % 8], label=None)
            _ = add_regression_line(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_indir_resp_1, ax=ax6, color=c_vals[pair_idx % 8], label=None)

            all_sums_x.extend(sum_dir_resp_1)
            all_sums_x.extend(sum_dir_resp_2)
            all_means_x.extend(mean_dir_resp_1)
            all_means_x.extend(mean_dir_resp_2)
            all_sums_y.extend(sum_indir_resp_1)
            all_sums_y.extend(sum_indir_resp_2)
            all_means_y.extend(mean_indir_resp_1)
            all_means_y.extend(mean_indir_resp_2)
            all_sum_diffs_x.extend(sum_dir_resp_2 - sum_dir_resp_1)
            all_sum_diffs_y.extend(sum_indir_resp_2 - sum_indir_resp_1)
            all_mean_diffs_x.extend(mean_dir_resp_2 - mean_dir_resp_1)
            all_mean_diffs_y.extend(mean_indir_resp_2 - mean_dir_resp_1)

            _ = add_regression_line(std_dir_resp_1, std_indir_resp_1, ax=ax2, color=c_vals[pair_idx % 8], label=None)
            _ = add_regression_line(std_dir_resp_2, std_indir_resp_2, ax=ax2, color=c_vals_l[pair_idx % 8], label=None)
            all_stds_x.extend(std_dir_resp_1)
            all_stds_x.extend(std_dir_resp_1)
            all_stds_y.extend(std_indir_resp_1)
            all_stds_y.extend(std_indir_resp_2)

        _ = add_regression_line(all_sums_x, all_sums_y, ax=ax1, color='k')
        _ = add_regression_line(all_stds_x, all_stds_y, ax=ax2, color='k')
        _ = add_regression_line(all_means_x, all_means_y, ax=ax3, color='k')
        _ = add_regression_line(all_sum_diffs_x, all_sum_diffs_y, ax=ax5, color='k')
        _ = add_regression_line(all_mean_diffs_x, all_mean_diffs_y, ax=ax6, color='k')

        ax1.set_xlabel('Sum Dir. {}'.format(metric))
        ax1.set_ylabel('Sum Indir. {}'.format(metric))
        ax2.set_xlabel('Std Dir. {}'.format(metric))
        ax2.set_ylabel('Std Indir. {}'.format(metric))
        ax3.set_xlabel('Mean Dir. {}'.format(metric))
        ax3.set_ylabel('Mean Indir. {}'.format(metric))
        ax5.set_xlabel('$\Delta$ (Sum Dir. {})'.format(metric))
        ax5.set_ylabel('$\Delta$ (Sum Indir. {})'.format(metric))
        ax6.set_xlabel('$\Delta$ (Mean Dir. {})'.format(metric))
        ax6.set_ylabel('$\Delta$ (Mean Indir. {})'.format(metric))

        ax11.set_xlabel('Sum Dir. {} (Day 1)'.format(metric))
        ax11.set_ylabel('Sum Indir. {} (Day 1)'.format(metric))
        ax22.set_xlabel('Sum Dir. {} (Day 2)'.format(metric))
        ax22.set_ylabel('Sum Indir. {} (Day 2)'.format(metric))
        ax33.set_xlabel('Mean Dir. {} (Day 1)'.format(metric))
        ax33.set_ylabel('Mean Indir. {} (Day 1)'.format(metric))
        ax44.set_xlabel('Mean Dir. {} (Day 2)'.format(metric))
        ax44.set_ylabel('Mean Indir. {} (Day 2)'.format(metric))
        ax55.set_xlabel('$\Delta$ (Sum Dir. {})'.format(metric))
        ax55.set_ylabel('$\Delta$ (Sum Indir. {})'.format(metric))
        ax66.set_xlabel('$\Delta$ (Mean Dir. {})'.format(metric))
        ax66.set_ylabel('$\Delta$ (Mean Indir. {})'.format(metric))

        for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax11, ax22, ax33, ax44, ax55, ax66):
            ax.axhline(0.0, color='grey', linestyle='dashed')
            ax.axvline(0.0, color='grey', linestyle='dashed')
    #     for ax in (ax11, ax22, ax33, ax44, ax55, ax66):
            ax.legend()

        fig2.show()

        return records

    which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)

    neuron_metric = 'pre' # tuning, trial_resp, pre, post

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

    session_idx_pairs.remove((20, 21)) # Kayvon tosses out this session

    print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

    records = check_for_group_corrs(
        neuron_metric, ps_stats_params, session_idx_pairs, data_dict,
        verbose=False
    )

if __name__ == '__main__':
    plot_cell_79()
