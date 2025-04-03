# Cell 31
def eval_over_neurons(
    neuron_metric_1, neuron_cc_1, neuron_metric_2, neuron_cc_2, ps_stats_params, weights=None,
    over_neuron_mode='matrix_multiply_sanity', debug=False, second_weights=None, keep_days_separate=False,
):
    """
    Various ways of computing "A" or "W" for a given group OR group/indirect neuron
    combination (e.g. in computing correlation). 
    
    An example usage would be the computation of 5c, where the y-axis's W, for a 
    given group, is given by the matrix multiplication of delta tuning and 
    indirect change in photostimulation.

    This function gives added functionality to decompose various contributions 
    to said matrix multiplication including things like doing mean subtraction
    or compuating the pearson correlation instead of matrix multiplication. All
    of these still function as taking something that is shape (n_neurons,) and 
    another quantity that is (n_neurons, n_groups,) to something that is 
    (n_groups,)

    Note that many of these measures require being more careful than simply doing
    a matrix multiplication, so the masks for each group are done by hand
    rather than relying on zeros in the neuron_cc_1 to properly mask things.
    
    Note this function can return NaNs for certain elements and they will be omitted
    by a nan mask later.
    
    Finally note that over_neuron_mode can be a list of modes, in which case it 
    iterates over the various modes
    
    INPUTS:
    - neuron_metric_1: string OR np.array of shape: (n_neurons,) OR (n_neurons2, n_neurons)
    - neuron_cc_1: the weight metric (n_neurons, n_groups,)
    - neuron_metric_2: None OR string OR np.array of shape: (n_neurons,) OR (n_neurons2, n_neurons)
    - neuron_cc_2: None OR the weight metric (n_neurons, n_groups,)
    - over_neuron_mode: string or list of strings
        - matrix_multiply: usual way of doing matrix multiplication that relies
            on zeros to mask groups
        - matrix_multiply_sanity: sanity check to make sure everything is done 
            correctly, should yield same results as matrix-multiply, but 
            eliminates group members by hand
        - means: isolate mean contribution to matrix_multiply_sanity
        - matrix_multiply_centered: isolate mean subtraction contribution to matrix_multiply_sanity
    - second_weights: optional additional weights to mask n_neurons2 if neuron_metric_1 is 2d
        - Generally used to properly select down to indirect neurons
    """
    
    if ps_stats_params['ps_analysis_type'] == 'single_session': # Doesnt make sense to have two days in this setting
        assert neuron_metric_2 is None
        assert neuron_cc_2 is None
        assert ~keep_days_separate
    elif ps_stats_params['ps_analysis_type'] == 'paired': # Doesnt make sense to have two days in this setting
        assert neuron_metric_2 is not None
        assert neuron_cc_2 is not None
        if type(neuron_metric_1) != str:
            assert np.all(neuron_metric_1.shape == neuron_metric_2.shape)
        assert np.all(neuron_cc_1.shape == neuron_cc_2.shape)
    
    group_vals_all = []
    if type(over_neuron_mode) == str: # Default case, creates single element tuple to iterate over
        on_modes = (over_neuron_mode,)
    else:
        on_modes = over_neuron_mode
        
    neuron_metric_1, neuron_cc_1, weights = validation_tests(
        neuron_metric_1, neuron_cc_1, ps_stats_params, weights=weights, second_weights=second_weights,
        on_modes=on_modes
    )
    
    if type(neuron_metric_1) != str:
        # These work when neuron metric is either (n_neurons,) OR (n_neurons2, n_neurons)
        if ps_stats_params['neuron_metrics_adjust'] in ('normalize',):
            raise NotImplementedError('Update to work with two neuron_metrics')
            neuron_metric_1 /= np.linalg.norm(neuron_metric_1, axis=-1, keepdims=True)
        elif ps_stats_params['neuron_metrics_adjust'] in ('standardize',):
            raise NotImplementedError('Update to work with two neuron_metrics')
            neuron_metric_1 = (
                (neuron_metric_1 - np.nanmean(neuron_metric_1, axis=-1, keepdims=True)) / 
                np.nanstd(neuron_metric_1, axis=-1, keepdims=True)
            )
        elif ps_stats_params['neuron_metrics_adjust'] != None:
            raise NotImplementedError('neuron_metrics_adjust {} not recognized.'.format(ps_stats_params['neuron_metrics_adjust']))
        
    for on_mode in on_modes: 
        # This is the slow way of doing things but treats things more carefully than just explicit matrix multiplication
        if on_mode in ('pearson', 'means', 'matrix_multiply_centered', 
                       'matrix_multiply_standardized', 'neuron_centered_conn_standardized'):
            raise ValueError('Depricated in new eval_over_neurons function')
        if on_mode in ('matrix_multiply_sanity', 'product_nc',) or type(neuron_metric_1) == str:
            assert weights is not None
            
            weights_mask = weights > 0 # (n_neurons, n_groups,)
            
            n_groups = neuron_cc_1.shape[-1]
            
            if type(neuron_metric_1) == str:
                if neuron_metric_1 in ('raw_nc',): # Do not collapse over indirect neurons
                    group_vals = [] # (n_groups, n_indirect_neurons)
                else:
                    group_vals = np.zeros((n_groups,)) # (n_groups,)
            elif on_mode in ('product_nc',): # Metrics that do not collapse over indirect neurons
                group_vals = [] # (n_groups, n_indirect_neurons)
            else:
                if len(neuron_metric_1.shape) == 1: # (n_neurons,)
                    group_vals = np.zeros((n_groups,))
                elif len(neuron_metric_1.shape) == 2: # For when things like correlation are used, (n_neurons2, n_neurons,)
                    group_vals = []
                    assert second_weights is not None
                    second_weights_mask = second_weights > 0 # (n_neurons, n_groups)

            for group_idx in range(n_groups):
                
                masked_neuron_cc_1 = neuron_cc_1[weights_mask[:, group_idx], group_idx]
                if neuron_cc_2 is not None:
                    masked_neuron_cc_2 = neuron_cc_2[weights_mask[:, group_idx], group_idx]

                if masked_neuron_cc_1.shape[0]== 0: # Default values if no neurons make it past the mask
                    if type(group_vals) == list: 
                        if len(neuron_metric_1.shape) == 1: # (n_neurons = 0)
                            group_vals.append(np.array(()))
                        elif len(neuron_metric_1.shape) == 2: # (n_neurons2,)
                            masked_neuron_metric_1 = neuron_metric_1[second_weights_mask[:, group_idx], :]
                            group_vals.append(np.nan * np.ones(masked_neuron_metric_1.shape[0]))
                    else:
                        group_vals[group_idx] = np.nan
                    continue

                if type(neuron_metric_1) == str: # Special sum, all are independent of over_neuron_mode
                    if neuron_cc_2 is not None: # Distinct ways of combining don't matter in these cases
                        masked_neuron_cc = masked_neuron_cc_1 + masked_neuron_cc_2
                    else:
                        masked_neuron_cc = np.copy(masked_neuron_cc_1)
                    
                    if neuron_metric_1 in ('raw_nc',):
                        assert on_mode in ('product_nc',)
                        group_vals.append(masked_neuron_cc) # n_indirect
                    elif neuron_metric_1 in ('sum',):
                        group_vals[group_idx] = np.sum(masked_neuron_cc)
                    elif neuron_metric_1 in ('abs_sum',):
                        group_vals[group_idx] = np.sum(np.abs(masked_neuron_cc))
                    elif neuron_metric_1 in ('mask_sum',): # Zeros eliminated in sum
                        group_vals[group_idx] = np.sum(weights[:, group_idx])
                else:
                    # Handle masking for the various possible shapes of the neuron metric
                    if len(neuron_metric_1.shape) == 1: # (n_neurons,)
                        masked_neuron_metric_1 = neuron_metric_1[weights_mask[:, group_idx]]
                        if neuron_metric_2 is not None:
                            masked_neuron_metric_2 = neuron_metric_2[weights_mask[:, group_idx]]
                    elif len(neuron_metric_1.shape) == 2: # (n_neurons2, n_neurons,)
                        if on_mode not in ('matrix_multiply_sanity',):
                            NotImplementedError('neuron_metric_1 shape not implemented for on_mode:', on_mode)
                        masked_neuron_metric_1 = neuron_metric_1[:, weights_mask[:, group_idx]]
                        masked_neuron_metric_1 = masked_neuron_metric_1[second_weights_mask[:, group_idx], :]
                        if neuron_metric_2 is not None:
                            masked_neuron_metric_2 = neuron_metric_2[:, weights_mask[:, group_idx]]
                            masked_neuron_metric_2 = masked_neuron_metric_2[second_weights_mask[:, group_idx], :]
                    else:
                        raise NotImplementedError('neuron_metric_1 shape not recognized:', neuron_metric_1.shape)                
                        
                    if on_mode in ('matrix_multiply_sanity',): # Sanity check that this yields same as mm above
                        if len(masked_neuron_metric_1.shape) == 1: # yields scale
                            if ps_stats_params['ps_analysis_type'] == 'single_session':
                                group_vals[group_idx] = np.dot(masked_neuron_metric_1, masked_neuron_cc_1)
                            elif ps_stats_params['ps_analysis_type'] == 'paired':
                                if not keep_days_separate:
                                    group_vals[group_idx] = np.dot( # Combine values between days first, then product
                                        masked_neuron_metric_1 + masked_neuron_metric_2,
                                        masked_neuron_cc_1 + masked_neuron_cc_2
                                    )
                                else:
                                    group_vals[group_idx] = ( # Product first, then combine between days
                                        np.dot(masked_neuron_metric_1, masked_neuron_cc_1) +
                                        np.dot(masked_neuron_metric_2, masked_neuron_cc_2)
                                    )
                        elif len(masked_neuron_metric_1.shape) == 2: # yields (n_neurons2,)
                            if ps_stats_params['ps_analysis_type'] == 'single_session':
                                group_vals.append(np.matmul(masked_neuron_metric_1, masked_neuron_cc_1))
                            elif ps_stats_params['ps_analysis_type'] == 'paired':
                                if not keep_days_separate:
                                    group_vals.append(np.matmul( # Combine values between days first, then product
                                        masked_neuron_metric_1 + masked_neuron_metric_2,
                                        masked_neuron_cc_1 + masked_neuron_cc_2
                                    ))
                                else:
                                    group_vals.append( # Product first, then combine between days
                                        np.matmul(masked_neuron_metric_1, masked_neuron_cc_1) +
                                        np.matmul(masked_neuron_metric_2, masked_neuron_cc_2)
                                    )
                    elif on_mode in ('product_nc',):
                        if ps_stats_params['ps_analysis_type'] == 'single_session':
                            group_vals.append(masked_neuron_metric_1 * masked_neuron_cc_1) # n_indirect
                        elif ps_stats_params['ps_analysis_type'] == 'paired':
                            if not keep_days_separate: # Combine values between days first, then product
                                group_vals.append(
                                    (masked_neuron_metric_1 + masked_neuron_metric_2) *
                                    (masked_neuron_cc_1 + masked_neuron_cc_2)
                                )
                            else:
                                group_vals.append( # Product first, then combine between days
                                    (masked_neuron_metric_1 * masked_neuron_cc_1) +
                                    (masked_neuron_metric_2 * masked_neuron_cc_2)
                                )
                                
            group_vals_all.append(group_vals)
        elif on_mode == 'matrix_multiply':
            # Mask sum is already incorporated into neuron_cc_1 by setting 
            # corresponding elements to zero
            if weights is not None:
                raise ValueError('Weights dont make sense here.')

            group_vals_all.append(np.matmul(neuron_metric_1.T, neuron_cc_1))
        else:
            raise ValueError('Over_neuron_mode {} not recognized.'.format(on_mode))
            
    if type(over_neuron_mode) == str: # Default case, returns first element in list
        return group_vals_all[0]
    else: # Retruns entire list instead
        return group_vals_all
    
def validation_tests(
    neuron_metric_1, neuron_cc_1, ps_stats_params, weights=None,
    second_weights=None, on_modes=(),
):
    if 'matrix_multiply' in on_modes:
        raise NotImplementedError('Validation tests not yet implemented for on_modes:', on_modes)
    
    weights_mask = weights > 0 # (n_neurons, n_groups,)
    if 'shuffle_X_within_session' in ps_stats_params['validation_types']: # Shuffle the neuron metric
        raise NotImplementedError('Update to work with two neuron_metrics')
        if type(neuron_metric_1) != str: # Skip if string, because neuron_metric_1 does nothing for these types
            # Neuron metric is either (n_neurons,) OR (n_neurons2, n_neurons), so always shuffle along -1th axis
            neuron_metric_1 = shuffle_along_axis(neuron_metric_1, axis=-1)
    if 'mean_X_within_session' in ps_stats_params['validation_types']:
        raise NotImplementedError('Update to work with two neuron_metrics')
        if type(neuron_metric_1) != str: # Skip if string, because neuron_metric_1 does nothing for these types
            # Neuron metric is either (n_neurons,) OR (n_neurons2, n_neurons), so take mean along -1th axis
            neuron_metric_1 = np.nanmean(neuron_metric_1, axis=-1, keepdims=True) * np.ones_like(neuron_metric_1)
    if 'shuffle_X_within_group' in ps_stats_params['validation_types']: # Shuffle current group
        raise NotImplementedError('Update to work with two neuron_metrics')
        if type(neuron_metric_1) != str: # Skip if string, because neuron_metric_1 does nothing for these types
            raise NotImplementedError()
            temp_neuron_metric_1 = np.copy(neuron_metric_1)
    
    return neuron_metric_1, neuron_cc_1, weights

# Cell 33
MAX_P_LOG10 = 5 # Maximum significance to plot for various plots below

def scan_over_connectivity_pairs(ps_stats_params, records, exemplar_session_idx=None, verbose=False):
    """
    Scans over various pairs of connectivity metrics that have been
    collected in records. This is used to look for significant trends
    for both single sessions and session pairs.
    
    If ps_stats_params['plot_pairs'] is None, this does a brute force
    scan over all possible pair combinations for all connectivity 
    metrics that end in '_x' or '_y'.
    """
    
    # This can either be session pairs or individual sessions
    n_sessions = len(records[ps_stats_params['connectivity_metrics'][0]])

    full_fits = {} # Dict of fits across all sessions
    session_fits = {} # Dict of fits to individual sessions (optional)
    
    # Create an exhaustive list of all possible fit pairs from connectivity_metrics
    plot_pairs_significance_plot = False
    if ps_stats_params['plot_pairs'] is None:
        print('Generating all possible combinations of plot_pairs...')
        connectivity_metrics_xs = []
        connectivity_metrics_ys = []
        
        for key in ps_stats_params['connectivity_metrics']:
            if key in ('mask_counts_x', 'mask_counts_y',):
                continue # Don't add these ones to the scan
            
            if key[-2:] == '_x':
                connectivity_metrics_xs.append(key)
            elif key[-2:] == '_y':
                connectivity_metrics_ys.append(key)
            elif key[-2:] not in ('_w',): # Things like weights are not included in this scan
                print('Ending of {} is {}, not recognized.'.format(key, key[-2:]))
        plot_pairs = list(itertools.product(connectivity_metrics_xs, connectivity_metrics_ys))
        plot_pairs_idxs = list(itertools.product(
            np.arange(len(connectivity_metrics_xs)), np.arange(len(connectivity_metrics_ys))
        ))
        
        ps_stats_params['plot_pairs'] = plot_pairs
        
        plot_pairs_xs = np.zeros((len(connectivity_metrics_xs), len(connectivity_metrics_ys),))
        plot_pairs_ys = np.zeros_like(plot_pairs_xs)
        plot_pairs_ps = np.zeros_like(plot_pairs_xs)
        plot_pairs_slopes = np.zeros_like(plot_pairs_xs)
        plot_pairs_r_squareds = np.zeros_like(plot_pairs_xs)
        
        plot_pairs_significance_plot = True
    
    plot_pairs = ps_stats_params['plot_pairs']
    
    ### Iteration over all possible things we want to plot against one another ###
    for plot_pairs_idx, pair in enumerate(plot_pairs):
        
        session_fits[pair] = [] # Only fillted in if fitting individual sessions
        
        # For the given pair, flattens records across sessions to plot all at once. 
        # Note this is not the most efficient way to do this because it
        # will have to flatten the same thing many times, but was the 
        # easiest to code up.
        records_x_flat = []
        records_y_flat = []
        
        records_w_flat = []
        
        if ps_stats_params['plot_up_mode'] in ('all',):
            fig, (ax, axp) = plt.subplots(1, 2, figsize=(12, 4))
        elif ps_stats_params['plot_up_mode'] == None:
            ax, axp = None, None
        else:
            raise NotImplementedError('plot_up_mode {} not recognized or depricated'.format(ps_stats_params['plot_up_mode']))
        
        # Somtimes x and y will be different shapes, so this code handles casting them to same shape
        for session_idx, (session_records_x, session_records_y) in enumerate(zip(records[pair[0]], records[pair[1]])):
            if type(session_records_x) == np.ndarray and type(session_records_y) == np.ndarray: # No modification needed
                session_records_x_flat = session_records_x
                session_records_y_flat = session_records_y
            elif type(session_records_x) == list or type(session_records_y) == list: # either x or y for each indirect, may need to extend the other
                session_records_x_flat = []
                session_records_y_flat = []
                for group_idx in range(len(session_records_y)):
                    if type(session_records_x) == np.ndarray: # Need to extend x for each y
                        session_records_x_flat.append(
                            session_records_x[group_idx] * np.ones((len(session_records_y[group_idx]),))
                        )
                        session_records_y_flat.append(session_records_y[group_idx])
                    elif type(session_records_y) == np.ndarray: # Need to extend y for each x
                        session_records_x_flat.append(session_records_x[group_idx])
                        session_records_y_flat.append(
                            session_records_y[group_idx] * np.ones((len(session_records_x[group_idx]),))
                        )
                    elif type(session_records_x) == list and type(session_records_y) == list: # x and y already extended
                        session_records_x_flat.append(session_records_x[group_idx])
                        session_records_y_flat.append(session_records_y[group_idx])
                    else:
                        raise NotImplementedError('This combination shouldnt occur')
                    
                    if 'shuffle_A_within_group' in ps_stats_params['validation_types']: # Shuffle most recent group
                        if plot_pairs_idx==0 and session_idx == 0 and group_idx ==0: print('Shuffling A within group!')
                        np.random.shuffle(session_records_y_flat[-1])
                    
                session_records_x_flat = np.concatenate(session_records_x_flat, axis=0)
                session_records_y_flat = np.concatenate(session_records_y_flat, axis=0)
                
                if ps_stats_params['indirect_weight_type'] is not None: # Does the same for weights
                    session_records_w_flat = []
                    for group_idx in range(len(session_records_y)):
                        session_records_w_flat.append(records['indirect_weights_w'][session_idx][group_idx])
                    session_records_w_flat = np.concatenate(session_records_w_flat, axis=0)
            else:
                raise NotImplementedError('This combination shouldnt occur')
            
            # Append session records to across-session collection
            records_x_flat.append(session_records_x_flat)
            records_y_flat.append(session_records_y_flat)
            if ps_stats_params['indirect_weight_type'] is not None:
                records_w_flat.append(session_records_w_flat)
                
            if ps_stats_params['fit_individual_sessions']:
            
                nan_mask = np.where(~np.isnan(session_records_x_flat) * ~np.isnan(session_records_y_flat))[0]
                session_records_x_flat = session_records_x_flat[nan_mask]
                session_records_y_flat = session_records_y_flat[nan_mask]
                if ps_stats_params['indirect_weight_type'] is not None:
                    session_weights = session_records_w_flat[nan_mask]
                else:
                    session_weights = None

                if ps_stats_params['ps_analysis_type'] in ('single_session',):
                    session_color = c_vals[SESSION_COLORS[session_idx]]
                elif ps_stats_params['ps_analysis_type'] in ('paired',):
                    session_color = c_vals[PAIR_COLORS[session_idx]]
                
                session_fit = {}
                session_fit['slope'], _, session_fit['rvalue'], session_fit['pvalue'], _ = add_regression_line(
                    session_records_x_flat, session_records_y_flat, weights=session_weights, ax=ax, 
                    color=session_color, label=None,
                )
                session_fit['pvalue'] = np.where(session_fit['pvalue'] < MIN_P_VALUE, MIN_P_VALUE, session_fit['pvalue'])
                
                session_fits[pair].append(session_fit)
                
                if axp is not None:
                    axp.scatter(session_fit['slope'], -np.log10(session_fit['pvalue']), marker='.', color=session_color)
                    
                if exemplar_session_idx is not None: # Exemplar session plot of fit
                    if session_idx == exemplar_session_idx:
                        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
                        
                        ax2.scatter(
                            session_records_x_flat, session_records_y_flat, color=session_color, alpha=0.05,
                            marker='.'
                        )
                        add_regression_line(
                            session_records_x_flat, session_records_y_flat, weights=session_weights, ax=ax2, 
                            color=session_color,
                        )
                        ax2.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
                        ax2.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')

                        ax2.set_xlabel(pair[0])
                        ax2.set_ylabel(pair[1])
                        ax2.legend()
            
        records_x_flat = np.concatenate(records_x_flat, axis=0)
        records_y_flat = np.concatenate(records_y_flat, axis=0)
        
        # Catch nans in records that can result from invalid group masks
        nonnan_mask = np.where(~np.isnan(records_x_flat) * ~np.isnan(records_y_flat))[0]
        records_x_flat = records_x_flat[nonnan_mask]
        records_y_flat = records_y_flat[nonnan_mask]
        
        if ps_stats_params['normalize_masks']: # Trivial relation in this setting
            if pair[0] == 'mask_counts_x' and pair[1] == 'mask_counts_y':
                continue
        
        if ps_stats_params['indirect_weight_type'] is not None: 
            records_w_flat = np.concatenate(records_w_flat, axis=0)
            records_w_flat = records_w_flat[nonnan_mask]
            weights = records_w_flat
        elif ps_stats_params['group_weights_type'] in ('direct_resp', 'direct_resp_norm_sessions',):
            raise NotImplemntedError('This is depricated now, need to update to new name conventions \
                                      and the fact that flat records is not computed ahead of time.')
            
            if ps_stats_params['ps_analysis_type'] == 'single_session':
                weight_key = 'cc_x'  # Total direct response for each group
            elif ps_stats_params['ps_analysis_type'] == 'paired':
                # Construct a special weight vector
                flat_records['laser_resp_special_x'] = []
                for session_cc_x, session_laser_resp in zip(records['cc_x'], records['laser_resp_x']):
                    # 0. if either session's cc_x < 0., otherwise mean laser resp.
                    for group_idx in range(session_cc_x[0].shape[0]):
                        if session_cc_x[0][group_idx] < 0. or session_cc_x[1][group_idx] < 0.:
                            flat_records['laser_resp_special_x'].append(0.0)
                        else:
                            flat_records['laser_resp_special_x'].append(session_laser_resp[group_idx])
                flat_records['laser_resp_special_x'] = np.array(flat_records['laser_resp_special_x'])
                weight_key = 'laser_resp_special_x' 
#                 weight_key = 'laser_resp_x'  # Mean direct response across pair for each group 
            else:
                raise ValueError('No weight key specified.')
            
            weights = np.where(flat_records[weight_key] > 0., flat_records[weight_key], 0.) # Zero out negatives
            if ps_stats_params['group_weights_type'] in ('direct_resp_norm_sessions',):
                raise NotImplementedError()
            # Apply same nan mask used above 
            weights = weights[nonnan_mask]
        elif ps_stats_params['group_weights_type'] == None and ps_stats_params['indirect_weight_type'] == None:
            weights = None
        else:
            raise NotImplementedError('Group weights type {} not recognized.'.format(ps_stats_params['group_weights_type']))
        
        aggregate_fit = {}
        aggregate_fit['slope'], _, aggregate_fit['rvalue'], aggregate_fit['pvalue'], _ = add_regression_line(
            records_x_flat, records_y_flat, weights=weights, ax=ax, color='k', label='',
        )
        aggregate_fit['pvalue'] = np.where(aggregate_fit['pvalue'] < MIN_P_VALUE, MIN_P_VALUE, aggregate_fit['pvalue'])
        
        if verbose:
            print_str = '{} - {}\tvs. {}\tp={:.2e}\t{}'.format(
                plot_pairs_idx, pair[0], pair[1], aggregate_fit['pvalue'], 
                '*' * np.min((MAX_P_LOG10, int(-np.log10(aggregate_fit['pvalue'])),))
            )
            print(print_str)

        full_fits[pair] = aggregate_fit
        
        if ax is not None:
            ax.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            
            ax.set_xlabel(pair[0])
            ax.set_ylabel(pair[1])
            ax.legend()
        if axp is not None:
            axp.scatter(aggregate_fit['slope'], -np.log10(aggregate_fit['pvalue']), marker='o', color='k')
            
            axp.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            axp.axhline(2.0, color='lightgrey', zorder=-5, linestyle='dashed')
            
            axp.set_xlabel('fit slope')
            axp.set_ylabel('fit -log10(p)')
        
        if plot_pairs_significance_plot:
            pair_idxs = plot_pairs_idxs[plot_pairs_idx]
            plot_pairs_xs[pair_idxs[0], pair_idxs[1]] = pair_idxs[0]
            plot_pairs_ys[pair_idxs[0], pair_idxs[1]] = pair_idxs[1]
            plot_pairs_ps[pair_idxs[0], pair_idxs[1]] = aggregate_fit['pvalue'] 
            plot_pairs_slopes[pair_idxs[0], pair_idxs[1]] = aggregate_fit['slope']
            plot_pairs_r_squareds[pair_idxs[0], pair_idxs[1]] = aggregate_fit['rvalue']**2
    
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    if plot_pairs_significance_plot:
        fig, (ax, axp) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Avoids errors where p_values are too low to be meaningful
        plot_pairs_ps = np.where(plot_pairs_ps < MIN_P_VALUE, MIN_P_VALUE, plot_pairs_ps)
        
        plot_pairs_ps_log10 = np.where(
            -1 * np.log10(plot_pairs_ps) > MAX_P_LOG10, MAX_P_LOG10, -1 * np.log10(plot_pairs_ps)
        )
        # Color plot_piar_ps by sign of slope
        plot_pairs_ps_log10 = np.sign(plot_pairs_slopes) * plot_pairs_ps_log10
        
        # Transpose needed since matshow has first index run vertically by default, but first
        # index is xs so want it to run horizontally
        ax.matshow(plot_pairs_ps_log10.T, cmap='bwr', vmin=-MAX_P_LOG10, vmax=MAX_P_LOG10)
        axp.matshow(plot_pairs_r_squareds.T, cmap='Reds', vmin=0)
        
        for (i, j), z in np.ndenumerate(plot_pairs_ps): # Above transpose accounted for here too
            ax.text(i, j, '({:0.0e})'.format(z), ha='center', va='center')
        for (i, j), z in np.ndenumerate(plot_pairs_r_squareds): # Above transpose accounted for here too
            axp.text(i, j, '({:0.1e})'.format(z), ha='center', va='center')
        
#         # Old plots that use dots
#         plot_pairs_colors = ['k' if s > 0 else 'r' for s in plot_pairs_slopes.flatten()]

#         ax.scatter(plot_pairs_xs.flatten(), plot_pairs_ys.flatten(), s=50*plot_pairs_ps_log10.flatten(),
#                    color=plot_pairs_colors)
        for ax_plot in (ax, axp,):
            ax_plot.set_xticks(np.arange(len(connectivity_metrics_xs)))
            ax_plot.set_xticklabels(connectivity_metrics_xs, rotation=90)
            ax_plot.set_yticks(np.arange(len(connectivity_metrics_ys)))
            ax_plot.set_yticklabels(connectivity_metrics_ys) 
        
    if ps_stats_params['fit_individual_sessions']:
        for plot_pairs_idx, pair in enumerate(plot_pairs):
            
            full_fits[pair]['pvalue'] = np.where(full_fits[pair]['pvalue'] < MIN_P_VALUE, MIN_P_VALUE, full_fits[pair]['pvalue'])
            full_p_val = np.min((-1 * np.log10(full_fits[pair]['pvalue']), MAX_P_LOG10))
            ax1.scatter(plot_pairs_idx, full_p_val, color='k', marker='_')
            
            ax2.scatter(plot_pairs_idx, full_fits[pair]['slope'] / full_fits[pair]['slope'], color='k', marker='_')
            
            pair_ps = np.zeros((n_sessions,))
            pair_slopes = np.zeros((n_sessions,))
            
            for session_idx in range(n_sessions):
                if ps_stats_params['ps_analysis_type'] in ('single_session',):
                    session_color = c_vals[SESSION_COLORS[session_idx]]
                elif ps_stats_params['ps_analysis_type'] in ('paired',):
                    session_color = c_vals[PAIR_COLORS[session_idx]]
                
                session_p_val = np.min((-1 * np.log10(session_fits[pair][session_idx]['pvalue']), MAX_P_LOG10))
                ax1.scatter(plot_pairs_idx, session_p_val, color=session_color, zorder=-5,
                            alpha=0.5, marker='.')
                pair_ps[session_idx] = session_p_val
                
                # Normalize slopes by full fit slope to see consistency
                pair_slopes[session_idx] = session_fits[pair][session_idx]['slope'] / full_fits[pair]['slope']
                ax2.scatter(plot_pairs_idx, pair_slopes[session_idx], color=session_color, 
                            zorder=-5, alpha=0.5, marker='.')
                
            ax1.scatter(plot_pairs_idx, pair_ps.mean(), color='k', zorder=-1, marker='.')
            ax2.scatter(plot_pairs_idx, pair_slopes.mean(), color='k', zorder=-1, marker='.')
            ax1.errorbar(plot_pairs_idx, pair_ps.mean(), pair_ps.std() / np.sqrt(n_sessions), 
                         color='k', zorder=-2, marker='.', alpha=0.8)
            ax2.errorbar(plot_pairs_idx, pair_slopes.mean(), pair_slopes.std() / np.sqrt(n_sessions), 
                         color='k', zorder=-2, marker='.', alpha=0.8)
            
        ax1.set_xlabel('Plot_pairs')
        ax1.set_ylabel('-log10(ps)')
        
        ax2.set_xlabel('Plot_pairs')
        ax2.set_ylabel('Norm. slope')
        
        ax2.set_ylim((-2, 2))
        ax2.axhline(1.0, color='grey', linestyle='dashed', zorder=-10)
        ax2.axhline(0.0, color='grey', linestyle='dashed', zorder=-10)
        
        for ax in (ax1, ax2,):
            ax.set_xticks(np.arange(len(plot_pairs)))
            ax.set_xticklabels([None if x % 5 else x for x in range(len(plot_pairs))]) # Label every 5th
            for val in range(0, len(plot_pairs), 5):
                ax.axvline(val, color='lightgrey', alpha=0.5, zorder=-20)
            
        fig2.show()
                
    return full_fits, session_fits, ps_stats_params

def enumerate_plot_pairs(ps_stats_params):
    """
    Given a list of all possible connectivity_metrics, separates them into independent/dependent variables.
    """
    connectivity_metrics_xs = []
    connectivity_metrics_ys = []

    for key in ps_stats_params['connectivity_metrics']:
        if key in ('mask_counts_x', 'mask_counts_y',):
            continue # Don't add these ones to the scan

        if key[-2:] == '_x':
            connectivity_metrics_xs.append(key)
        elif key[-2:] == '_y':
            connectivity_metrics_ys.append(key)
        elif key[-2:] not in ('_w',): # Things like weights are not included in this scan
            print('Ending of {} is {}, not recognized.'.format(key, key[-2:]))
            
    return connectivity_metrics_xs, connectivity_metrics_ys

def get_all_xs(records, connectivity_metrics_xs, standardize_xs=False):
    """
    Get all x predictors that are going to be used later for MLR fits.
    Output is in a form that can be used for full fit and for hierarchical
    bootstrapping too.
    
    OUTPUTS:
    - records_x_flat_all: list of lists of numpy arrays
        - First list idx: session_idx
        - Second list idx: group_idx
        - Each numpy array is shape (n_cm_x, n_indirect) for the given session/group
    """
    
    if 'raw_cc_y' in records.keys():
        exemplar_key = 'raw_cc_y'
    elif 'raw_delta_cc_y' in records.keys():
        exemplar_key = 'raw_delta_cc_y'
    else:
        raise ValueError()

    n_sessions = len(records[exemplar_key])

    records_x_flat_all = [] # session_idx, group_idx, (n_cm_x, n_indirect)

    ### Iteration over all possible things we want to plot against one another ###
    for session_idx in range(n_sessions):
        
        exemplar_session_records_y = records[exemplar_key][session_idx]
        n_groups = len(exemplar_session_records_y)
        session_records_x_flat = [[] for group_idx in range(n_groups)] # group_idx, (n_cm_x, n_indirect)
        
        for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):

            session_records_x = records[connectivity_metrics_x][session_idx]

            # Expand all xs so that they're the same shape 
            for group_idx in range(len(session_records_x)):
                if type(session_records_x) == np.ndarray: # Need to extend x for each y
                    session_records_x_flat[group_idx].append(
                        session_records_x[group_idx] * np.ones((len(exemplar_session_records_y[group_idx]),))
                    )
                elif type(session_records_x) == list: # x already extended
                    session_records_x_flat[group_idx].append(session_records_x[group_idx])

        
        # Now for each group, stack the different possible cm_x
        for group_idx in range(len(session_records_x)):
            try:
                session_records_x_flat[group_idx] = np.array(session_records_x_flat[group_idx]) # (n_cm_x, n_indirect)
            except ValueError:
                print('session', session_idx, 'group', group_idx)
                for val_idx, val in enumerate(session_records_x_flat[group_idx]):
                    print(val_idx, len(val))
                
                print(connectivity_metrics_xs)
                
                print(session_records_x_flat[group_idx])
                print(sdfdsfds)
    
        if standardize_xs: # Standardize the x across the session  
            temp_session_records_x_flat = np.concatenate(session_records_x_flat, axis=-1) # (n_cm_x, n_total_indirect)
            session_mean = np.nanmean(temp_session_records_x_flat, axis=-1, keepdims=True) # (n_cm_x, 1)
            session_std = np.nanstd(temp_session_records_x_flat, axis=-1, keepdims=True) # (n_cm_x, 1)
            
            for group_idx in range(len(session_records_x_flat)): # Now adjust each group
                session_records_x_flat[group_idx] = (session_records_x_flat[group_idx] - session_mean) / session_std

        records_x_flat_all.append(session_records_x_flat)
        
    return records_x_flat_all

def fit_all_xs_at_once(records, ps_stats_params, fit_intercept=True, standardize_xs=False,
                       standardize_ys=False, bootstrap_type=None, n_bootstrap=1, 
                       verbose=False):
    """
    Fit all _xs at once for each _y using multiple linear regression. 
    
    INPUTS:
    
    - bootstrap_type: None, hierarcy
    - n_bootstrap: Number of bootstraps to run
    
    OUTPUTS:
    - full_fits: Dict with key for each connectivity_metrics_y, containing the full LS fit
        - e.g. full_fits[connectivity_metrics_y]
    - session_fits: Same as above, with each dict key containing a list of sessions
        - e.g. full_fits[connectivity_metrics_y][session_idx]
    """

    def full_fit(records_x_all, records_y_all, records_w_all=[]):
        """ This is sepearated into a function becauses its repeated when bootstrapping """
        
        records_x_all_temp = [None for _ in range(len(records_x_all))]
        records_y_all_temp = [None for _ in range(len(records_y_all))]
        records_w_all_temp = [None for _ in range(len(records_y_all))]
        
        # For each session, need to flatten across groups
        for session_idx in range(len(records_y_flat)):
            records_x_all_temp[session_idx] = np.concatenate(records_x_all[session_idx], axis=-1)
            records_y_all_temp[session_idx] = np.concatenate(records_y_all[session_idx], axis=0)
            if ps_stats_params['indirect_weight_type'] is not None: 
                records_w_all_temp[session_idx] = np.concatenate(records_w_all[session_idx], axis=0)
        
        # Now that all sessions are collected, concatenate sessions together
        records_x_all_temp = np.concatenate(records_x_all_temp, axis=-1) # Temporary concat to fit everything
        records_y_all_temp = np.concatenate(records_y_all_temp, axis=0)

        nonnan_mask = np.where(np.all(~np.isnan(records_x_all_temp), axis=0) * ~np.isnan(records_y_all_temp))[0]

        records_x_all_temp = records_x_all_temp[:, nonnan_mask]
        records_y_all_temp = records_y_all_temp[nonnan_mask]
        if ps_stats_params['indirect_weight_type'] is not None: 
            weights = np.concatenate(records_w_all_temp, axis=0)[nonnan_mask]
        elif ps_stats_params['group_weights_type'] == None and ps_stats_params['indirect_weight_type'] == None:
            weights = None
        else:
            raise NotImplementedError('Group weights type {} not recognized.'.format(ps_stats_params['group_weights_type']))

        X = records_x_all_temp.T # (n_regressors, n_neurons) -> (n_neurons, n_regressors)
        Y = records_y_all_temp[:, np.newaxis]

        if fit_intercept:
            X = sm.add_constant(X)

        if weights is None:
            fit_model = sm.OLS(Y, X, missing='none') # nans should be caught above
        else:
            fit_model = sm.WLS(Y, X, weights=weights, missing='drop') # nans should be caught above
        
        return fit_model.fit()
    
    param_idx_offset = 1 if fit_intercept else 0

    # This can either be session pairs or individual sessions
    n_sessions = len(records[ps_stats_params['connectivity_metrics'][0]])

    full_fits = {} # Dict of fits across all sessions
    session_fits = {} # Dict of fits to individual sessions (optional)

    if standardize_xs and verbose:
        print('Standardizing _x\'s!')
    if standardize_ys and verbose:
        print('Standardizing _y\'s!')

    # Create lists of all _x's and _y's from connectivity_metrics
    if ps_stats_params['plot_pairs'] is not None:
        raise NotImplementedError('All possible pairs is hard coded for now.')

    # Get list of all possible 
    connectivity_metrics_xs, connectivity_metrics_ys = enumerate_plot_pairs(ps_stats_params)

    # Get all _x's used to fit the _y's below
    records_x_flat_all = get_all_xs(records, connectivity_metrics_xs, standardize_xs=standardize_xs)
    
    if bootstrap_type is None:
        bs_params = None
    else:
        bs_params = np.zeros((len(connectivity_metrics_xs), len(connectivity_metrics_ys), n_bootstrap))

    if 'raw_cc_y' in records.keys():
        exemplar_key = 'raw_cc_y'
    elif 'raw_delta_cc_y' in records.keys():
        exemplar_key = 'raw_delta_cc_y'
    else:
        raise ValueError()

    # Now use the _x to predict each possible _y  
    for cm_y_idx, connectivity_metrics_y in enumerate(connectivity_metrics_ys):

        session_fits[connectivity_metrics_y] = [] # Initialize, only filled in if fitting individual sessions

        records_y_flat = []
        records_w_flat = []

        for session_idx, session_records_y in enumerate(records[connectivity_metrics_y]):
            exemplar_session_records_y = records[exemplar_key][session_idx]

            # Expand all ys so that they're the same shape 
            session_records_y_flat = []
            for group_idx in range(len(session_records_y)):
                if type(session_records_y) == np.ndarray: # Need to extend y for each x
                    session_records_y_flat.append(
                        session_records_y[group_idx] * np.ones((len(exemplar_session_records_y[group_idx]),))
                    )
                elif type(session_records_y) == list: # y already extended
                    session_records_y_flat.append(session_records_y[group_idx])

#             # Concatenate across all groups, to get total number of indirect in session
#             session_records_y_flat = np.concatenate(session_records_y_flat, axis=0)

            if standardize_ys:
                temp_session_records_y_flat = np.concatenate(session_records_y_flat, axis=0)
                session_mean = np.nanmean(temp_session_records_y_flat)
                session_std = np.nanstd(temp_session_records_y_flat)
                for group_idx in range(len(session_records_y_flat)): # Now adjust each group
                    session_records_y_flat[group_idx] = (session_records_y_flat[group_idx] - session_mean) / session_std

            if ps_stats_params['indirect_weight_type'] is not None: # Does the same for weights
                session_records_w_flat = []
                for group_idx in range(len(session_records_y)): # Already extended
                    session_records_w_flat.append(records['indirect_weights_w'][session_idx][group_idx])
#                 session_records_w_flat = np.concatenate(session_records_w_flat, axis=0) # Across groups

            # Append session records to across-session collection
            records_y_flat.append(session_records_y_flat)
            if ps_stats_params['indirect_weight_type'] is not None:
                records_w_flat.append(session_records_w_flat)

            if ps_stats_params['fit_individual_sessions']:
                
                # Temporary concat, leave separated into groups for potential bootstrapping later on
                session_records_x_flat_temp = np.concatenate(records_x_flat_all[session_idx], axis=-1) # Concat across groups
                session_records_y_flat_temp = np.concatenate(session_records_y_flat, axis=0)
                
                nonnan_mask = np.where(np.all(~np.isnan(session_records_x_flat_temp), axis=0) * ~np.isnan(session_records_y_flat_temp))[0]
                session_records_x_flat_temp = session_records_x_flat_temp[:, nonnan_mask]
                session_records_y_flat_temp = session_records_y_flat_temp[nonnan_mask]
                if ps_stats_params['indirect_weight_type'] is not None:
                    session_records_w_flat_temp = np.concatenate(session_records_w_flat, axis=0)
                    session_weights = session_records_w_flat_temp[nonnan_mask]
                else:
                    session_weights = None

                session_fit = {}

                X = session_records_x_flat_temp.T # (n_regressors, n_neurons) -> (n_neurons, n_regressors)
                Y = session_records_y_flat_temp[:, np.newaxis]

                if fit_intercept:
                    X = sm.add_constant(X)

                if session_weights is None:
                    fit_model = sm.OLS(Y, X, missing='none') # nans should be caught above
                else:
                    fit_model = sm.WLS(Y, X, weights=session_weights, missing='none') # nans should be caught above
                session_results = fit_model.fit()

                session_fits[connectivity_metrics_y].append(session_results)

        # Fit on all data at once
        full_fits[connectivity_metrics_y] = full_fit(
            records_x_flat_all, records_y_flat, records_w_flat
        )
        
        # Bootstrap fittings
        if bootstrap_type is not None:    
            for bs_idx in range(n_bootstrap):
    
                bs_session_idxs, bs_group_idxs = get_hier_bootstrap_shuffle(
                    records, bootstrap_groups=bootstrap_groups, seed=BASE_SEED+bs_idx
                )

                bs_records_x_all = []
                bs_records_y_all = []
                bs_records_w_all = []

                for session_idx, bs_session_idx in enumerate(bs_session_idxs):
                    bs_records_x_all.append(
                           [records_x_flat_all[bs_session_idx][bs_group_idx] for bs_group_idx in bs_group_idxs[session_idx]]
                    )
                    bs_records_y_all.append(
                           [records_y_flat[bs_session_idx][bs_group_idx] for bs_group_idx in bs_group_idxs[session_idx]]
                    )
                    if ps_stats_params['indirect_weight_type'] is not None:
                        bs_records_w_all.append(
                               [records_w_flat[bs_session_idx][bs_group_idx] for bs_group_idx in bs_group_idxs[session_idx]]
                        )

                bs_fit = full_fit(
                    bs_records_x_all, bs_records_y_all, bs_records_w_all
                )
                bs_params[:, cm_y_idx, bs_idx] = bs_fit.params[param_idx_offset:] # Don't include intercept
            
    return full_fits, session_fits, (connectivity_metrics_xs, connectivity_metrics_ys), bs_params

N_GROUPS = 100 # Hardcoding this in for now, since it is true for all sessions we are interested in

def get_hier_bootstrap_shuffle(records, bootstrap_groups=True, seed=None):
    """
    Seed is used here to have uniform bootstrap shuffles across all fits.
    
    First chooses same number of mice from full list of mice, with recplacement.
    Given mice, creates list of all sessions from the chosen mice (if the same 
    mouse is chosen twice, its sessions will be in the twice). Then chooses
    
    
    """
    if seed is not None:
        np.random.seed(seed)
    
    # This code could be made more efficient if we didn't need to reproduce these every time
    n_sessions = len(records[ps_stats_params['connectivity_metrics'][0]])
    mouse_idx_to_session_idx_map = []
    mouse_idx_to_mouse_map = []
    mouse_idx = 0
    for session_idx in range(n_sessions):
        if records['mice'][session_idx] not in mouse_idx_to_mouse_map: # New mouse
            mouse_idx_to_session_idx_map.append([]) 
            mouse_idx_to_mouse_map.append(records['mice'][session_idx])
            mouse_idx += 1
        mouse_idx_to_session_idx_map[mouse_idx_to_mouse_map.index(records['mice'][session_idx])].append(session_idx)
    
    n_mice = len(mouse_idx_to_mouse_map)

    bs_mouse_idxs = np.random.choice(np.arange(n_mice), n_mice, replace=True)
    bs_possible_session_idxs = [] # This might not be the same total number of session idxs, so sample below
    for mouse_idx in bs_mouse_idxs: 
        bs_possible_session_idxs.extend(mouse_idx_to_session_idx_map[mouse_idx])
    
    bs_session_idxs = np.random.choice(bs_possible_session_idxs, n_sessions, replace=True)
    
    bs_group_idxs = [] # For each session, which groups to include
    
    for session_idx in bs_session_idxs:
        groups_idxs = np.arange(N_GROUPS)
        if bootstrap_groups:
            bs_group_idxs.append(np.random.choice(groups_idxs, N_GROUPS, replace=True))
        else:
            bs_group_idxs.append(groups_idxs)
            
    return bs_session_idxs, bs_group_idxs

