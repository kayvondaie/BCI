import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'steps'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from steps.load_data import run as run_load_data
run_load_data()

from steps.setup_analysis import run as run_setup_analysis
run_setup_analysis()

from steps.find_valid_pairs import run as run_find_valid_pairs
run_find_valid_pairs()

from steps.select_exemplars import run as run_select_exemplars
run_select_exemplars()

from steps.plot_direct_prs import run as run_plot_direct_prs
run_plot_direct_prs()

from steps.plot_fit_summary import run as run_plot_fit_summary
run_plot_fit_summary()

from steps.weighting_and_session_selection import run as run_weighting_and_session_selection
run_weighting_and_session_selection()

from steps.assign_validation_fits import run as run_assign_validation_fits
run_assign_validation_fits()

from steps.find_valid_pairs_again import run as run_find_valid_pairs_again
run_find_valid_pairs_again()

from steps.filter_pairs_for_special_case import run as run_filter_pairs_for_special_case
run_filter_pairs_for_special_case()

from steps.plot_bootstrapped_metrics import run as run_plot_bootstrapped_metrics
run_plot_bootstrapped_metrics()

from steps.compare_cn_strategies import run as run_compare_cn_strategies
run_compare_cn_strategies()

from steps.compare_cn_strategies_2 import run as run_compare_cn_strategies_2
run_compare_cn_strategies_2()

from steps.compare_cn_strategies_3 import run as run_compare_cn_strategies_3
run_compare_cn_strategies_3()

from steps.evaluate_fit_consistency import run as run_evaluate_fit_consistency
run_evaluate_fit_consistency()

from steps.plot_fit_vs_delta import run as run_plot_fit_vs_delta
run_plot_fit_vs_delta()

from steps.refit_all_sessions import run as run_refit_all_sessions
run_refit_all_sessions()

from steps.plot_fit_accuracy import run as run_plot_fit_accuracy
run_plot_fit_accuracy()

from steps.plot_fit_error_dists import run as run_plot_fit_error_dists
run_plot_fit_error_dists()

from steps.debug_fit_weights import run as run_debug_fit_weights
run_debug_fit_weights()

from steps.plot_exemplar_session import run as run_plot_exemplar_session
run_plot_exemplar_session()

from steps.plot_exemplar_ps_traces import run as run_plot_exemplar_ps_traces
run_plot_exemplar_ps_traces()

from steps.session_idx_group_filtering import run as run_session_idx_group_filtering
run_session_idx_group_filtering()

from steps.load_behav_data import run as run_load_behav_data
run_load_behav_data()

from steps.plot_trial_timing import run as run_plot_trial_timing
run_plot_trial_timing()

from steps.compute_timeseries_window import run as run_compute_timeseries_window
run_compute_timeseries_window()

from steps.align_trial_starts_and_rewards import run as run_align_trial_starts_and_rewards
run_align_trial_starts_and_rewards()

from steps.compare_tuning_behav_vs_raw import run as run_compare_tuning_behav_vs_raw
run_compare_tuning_behav_vs_raw()

from steps.plot_exemplar_trace import run as run_plot_exemplar_trace
run_plot_exemplar_trace()

from steps.compare_behav_and_raw_traces import run as run_compare_behav_and_raw_traces
run_compare_behav_and_raw_traces()

from steps.compute_behavioral_correlations import run as run_compute_behavioral_correlations
run_compute_behavioral_correlations()

from steps.initialize_direct_response_array import run as run_initialize_direct_response_array
run_initialize_direct_response_array()
