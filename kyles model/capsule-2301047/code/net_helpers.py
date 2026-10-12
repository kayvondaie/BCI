import numpy as np

from scipy.stats import lognorm

import time # For debugging

import os
import pickle

import copy

def rand_weight_init(n_inputs, n_outputs=None, init_type='gaussian', cell_types=None,
                     ei_balance_val=None, sparsity=None, weight_norm=None, self_couplings=True):
    """
    Returns a random weight initialization of the specified type: 
    If n_outputs is None: (n_inputs,) 
    else: (n_outputs, n_inputs,)

    Note: uses numpy throughout, convert to tensor after call if needed.

    ei_balance_val: balances strength of excitation and inhibition
    spasity: override sparsity 

    self_couplings: whether or not diagonal couplings can be nonzero

    """
    sparsity_p = 1.0 if sparsity is None else sparsity

    if n_outputs is not None: # 2d case
        weight_shape = (n_outputs, n_inputs,)
    else: # 1d case
        weight_shape = (n_inputs,)

    if init_type == 'xavier':
        if n_outputs is not None: # 2d case
            xavier_bound = np.sqrt(6/(n_inputs + n_outputs)) if weight_norm is None else weight_norm
        else: # 1d case
            xavier_bound = np.sqrt(6/(n_inputs)) if weight_norm is None else weight_norm
        rand_weights = np.random.uniform(low=-xavier_bound, high=xavier_bound, size=weight_shape)
    elif init_type in ('gaussian', 'sparse_gaussian', 'sparse_gaussian_ln_in'):
        norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm
        rand_weights = norm_factor * np.random.normal(scale=1.0, size=weight_shape)
    elif init_type in ('mirror_gaussian',): # Two gaussians peaks centered at opposite means (fixed mean/scale ratio for now)
        norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm
        rand_weights = norm_factor * (
            np.random.choice([-1, 1], size=weight_shape) * np.random.normal(loc=1.0, scale=0.5, size=weight_shape)
        )
    elif init_type in ('log_normal', 'sparse_log_normal'):
        norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm # if init_type == 'log_normal' else n_outputs * sparsity_p
        rand_weights = norm_factor * np.random.lognormal(mean=0.0, sigma=1.0, size=weight_shape)
    elif init_type in ('ones', 'sparse_ones') or type(init_type) == float:
        if n_outputs is not None: # 2d case
            if n_outputs > 1:
                raise NotImplementedError('Normalization is weird for this, should think about it if we use this again')
            norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm #if init_type == 'ones' else n_outputs * sparsity_p
        else: # 1d case
            norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm #if init_type == 'ones' else n_inputs * n_outputs * sparsity_p
        if init_type in ('ones', 'sparse_ones'):
            rand_weights = norm_factor * np.ones(weight_shape)
        else:
            rand_weights = init_type * np.ones(weight_shape)
    elif init_type in ('rand_one_hot',):
        assert len(weight_shape) == 2
        shuffle_idxs = np.random.permutation(max(weight_shape))
        # Shuffle an identity matrix that is the larger of the two weight dimensions
        square_weights = np.eye(max(weight_shape))[shuffle_idxs, :]
        # Clip to appropriate size
        rand_weights = square_weights[:weight_shape[0], :weight_shape[1]]
    elif init_type in ('zeros',):
        rand_weights = np.zeros(weight_shape)
    else:
        raise NotImplementedError('Random weight init type {} not recognized!'.format(init_type))

    # Creates sparsification masks (masks that determine which elements are zero)
    if init_type in ('sparse_gaussian', 'sparse_log_normal', 'sparse_ones', 'sparse_gaussian_ln_in'):

        if init_type in ('sparse_gaussian', 'sparse_ones',): # Note this creates an in-degree distribution that is Gaussian distributed
            weight_mask = np.random.uniform(0, 1, size=rand_weights.shape) > sparsity_p # Which weights to zero
        elif init_type in ('sparse_log_normal', 'sparse_gaussian_ln_in',): 
            LOG_NORM_SHAPE = 0.25 # This could be adjusted to experimental data eventually

            # Adjusts scale relative to standard log normal to set desired mean
            standard_ln = lognorm(s=LOG_NORM_SHAPE)
            scaled_ln = lognorm(s=LOG_NORM_SHAPE, scale=sparsity_p / standard_ln.mean())

            # Draws twice as many as needed since will be truncating all > 1.0
            in_degrees = scaled_ln.rvs(size=2*n_outputs)
            in_degrees = in_degrees[in_degrees <=1.0] # Truncates to <1.0 in degrees

            if in_degrees.shape[0] < n_outputs:
                raise ValueError('Did not draw enough in-degrees that met threshold (only {}).'.format(
                    in_degrees.shape[0]
                ))

            weight_mask = np.zeros(weight_shape, dtype=bool)
            for out_idx in range(n_outputs): # Generates mask one output at a time
                in_degree_mask = np.zeros((n_inputs,), dtype=bool) # True for elements set to zero
                # Set all elements beyond index to be True, corresponding to no connection
                # (small in_degree[out_idx] corresponds to majority 1s)
                in_degree_mask[int(np.floor(in_degrees[out_idx] * n_inputs)):] = True
                np.random.shuffle(in_degree_mask)

                weight_mask[out_idx] = in_degree_mask

        if not self_couplings: # Sets all diagonal elements of mask so weights are set to zero
            assert n_inputs == n_outputs
            for neuron_idx in range(n_inputs):
                weight_mask[neuron_idx, neuron_idx] = True

        rand_weights[weight_mask] = 0.0

    if (cell_types is not None) and (n_outputs is not None): # Assigns cell types
        cell_types = cell_types.detach().cpu().numpy()
        assert cell_types.shape[0] == 1
        assert cell_types.shape[1] == n_inputs

        if ei_balance_val is not None:
             # Signed based on cell type, but also enhances strength appropriately
            cell_weights = np.where(cell_types < 0, ei_balance_val * cell_types, cell_types)
        else:
            cell_weights = cell_types # Just 1s or -1s to correct sign

        rand_weights = cell_weights * np.abs(rand_weights)

    # print(' Init type:', init_type)
    # print(' Cell types:', cell_types)
    # perc_pos = np.sum(rand_weights > 0) / np.prod(rand_weights.shape)
    # perc_neg = np.sum(rand_weights < 0) / np.prod(rand_weights.shape)
    # print(' Perc pos: {:.2f} perc neg: {:.2f}'.format(perc_pos, perc_neg))

    return rand_weights

def get_stimulus(task_params, stim_type='uniform', rel_stimulus_scale=1.0):
    """ Generates stimulus inputs for several tasks. """
    if 'n_inp' in task_params: # Defaults to number of inputs 
        n_neurons = task_params['n_inp']
    elif 'n_neurons' in task_params:
        n_neurons = task_params['n_neurons']
    else:
        raise NotImplementedError('Unknown stimulus size')

    if stim_type in ('uniform',):
        return np.random.uniform(
            -rel_stimulus_scale * task_params['task_scale'], rel_stimulus_scale * task_params['task_scale'], size=(1, n_neurons,)
        )
    elif stim_type in ('normal',):
        # I messed around with the scale of this input so that mean activity didn't look too huge compared to baseline
        return np.random.normal(
            0, rel_stimulus_scale * task_params['task_scale'], size=(1, n_neurons,)
        )

def append_to_average(avg_raw, raw, n_window=1):
    """ Rolling average """
    if len(raw) < n_window:
        avg_raw.append(np.mean(raw))
    else:
        avg_raw.append(np.mean(raw[-n_window:]))
    return avg_raw

def accumulate_decay(prev, new, n_window=10, nan_mode='persist'):
    """
    Running average of quantities, with special behavior for when nans are
    encountered (nans can arise in rewards)

    nan_mode:
    - persist: when new == nan, just returns prev
    - decay: when new == nan, treats new = 0.0
    """
    gamma = 1. - 1./n_window

    if np.prod(new.shape) == 1:
        if np.isnan(new):
            if nan_mode in ('persist',):
                return prev
            elif nan_mode in ('decay',):
                return gamma * prev
            else:
                raise ValueError('Nan mode {} not recognized.'.format(nan_mode))

    return gamma * prev + (1 - gamma) * new

# def accumulate_decay(decay_raw, raw, n_window=10, base_val=0.0):
#   """ Average by accumulation and decay """
#   gamma = 1. - 1./n_window
    
#   if len(decay_raw) == 0: # Use base_val for previous value
#       if np.isnan(raw[-1]): # Skips nans and just set to base value
#           decay_raw.append(base_val)
#       else:
#           decay_raw.append(gamma*base_val + 1/n_window * raw[-1])
#   else:
#       if np.isnan(raw[-1]): # Skips nans and just holds decay_raw constant
#           decay_raw.append(decay_raw[-1])
#       else:
#           decay_raw.append(gamma*decay_raw[-1] + 1/n_window * raw[-1])
#       # print('Raw {} new avg {}'.format(decay_raw[-1], decay_raw[-1]))
#   return decay_raw

def relu_fn(x):
    return torch.maximum(x, torch.zeros_like(x))
def relu_fn_np(x):
    return np.maximum(x, np.zeros_like(x))
def relu_fn_p(x):
    return torch.heaviside(x, torch.zeros_like(x)) # For x = 0, return 0 just like pytorch default

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))    
def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_p(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def tanh_p(x):
    return (1/torch.cosh(x))**2 # Sech**2

def tanh_re(x):
    return torch.maximum(torch.tanh(x), torch.zeros_like(x))
def tanh_re_np(x):
    return np.maximum(np.tanh(x), np.zeros_like(x))    
def tanh_re_p(x): # Have to use where here instead of maximum because we want this to return 0 for x = 0, like ReLU (and sech(0) = 1)
    return torch.where(x > 1e-6, (1/torch.cosh(x))**2, 0.) # Sech**2

def tanh_re_super(x, alpha=2.0):
    return torch.maximum(torch.tanh(alpha*x), torch.zeros_like(x))
def tanh_re_super_np(x, alpha=2.0):
    return np.maximum(np.tanh(alpha*x), np.zeros_like(x))    
def tanh_re_super_p(x): # Have to use where here instead of maximum because we want this to return 0 for x = 0, like ReLU (and sech(0) = 1)
    raise NotImplementedError()
    return torch.where(x > 0., (1/torch.cosh(x))**2, 0.) # Sech**2

def linear_fn(x):
    return x     
def linear_fn_p(x):
    return torch.ones_like(x)

def cubed_re(x):
    return torch.maximum(x**3, torch.zeros_like(x))
def cubed_re_p(x):
    return torch.maximum(3*x**2, torch.zeros_like(x))

def tukey_fn(x):
    raw = 1/2 - 1/2 * torch.cos(np.pi * x) 
    raw[x < 0.] = 0.0
    raw[x > 1.] = 1.0
    return raw
def tukey_fn_np(x):
    raw = 1/2 - 1/2 * np.cos(np.pi * x) 
    raw[x < 0.] = 0.0
    raw[x > 1.] = 1.0
    return raw
def tukey_fn_p(x):
    raw = np.pi / 2 * torch.sin(np.pi * x)
    raw[x < 0.] = 0.0
    raw[x > 1.] = 0.0

def heaviside_p(x):
    raise NotImplementedError()

def get_activation_function(act_fn):
    """ Returns pytorch version, numpy version, and pytorch derivative functions """
    if act_fn == 'ReLU':
        return relu_fn, relu_fn_np, relu_fn_p
    elif act_fn == 'sigmoid':
        return sigmoid, sigmoid_np, sigmoid_p
    elif act_fn == 'tanh_re':
        return tanh_re, tanh_re_np, tanh_re_p
    elif act_fn == 'tanh_re_super': # Supra linear version of Tanh
        return tanh_re_super, tanh_re_super_np, tanh_re_super_p
    elif act_fn == 'tanh':
        return torch.tanh, np.tanh, tanh_p
    elif act_fn == 'tukey':
        return tukey_fn, tukey_fn_np, tukey_fn_p
    elif act_fn == 'linear':
        return linear_fn, linear_fn, linear_fn_p
    elif act_fn == 'cubed_re':
        return cubed_re, None, cubed_re_p
    elif act_fn == 'heaviside':
        return lambda x : torch.heaviside(x, torch.tensor(0.5)), lambda x : np.heaviside(x, 0.5), heaviside_p
    else:
        raise ValueError('Activation function: {} not recoginized!'.format(act_fn))

def cosine_similarity_loss(output, pred):
    """
    Cosine similiarty loss, 
    expects output and pred to both be: (B, Ny)
    """
    cosine_sim = nn.CosineSimilarity(dim=-1)

    return torch.mean(torch.abs(cosine_sim(output, pred)))

def mse_loss_weighted(output, pred, weight):
    """
    Weighted version of MSE loss. For fitting curves weighted by mean squared error.
    """  
    return torch.mean(weight * (output - pred) ** 2)
    

def shuffle_dataset(inputs, labels, masks=None):
    """ Shuffles a dataset over its batch index """
    
    assert inputs.shape[0] == labels.shape[0] # Checks batch indexes are equal
    shuffle_idxs = np.arange(inputs.shape[0])
    np.random.shuffle(shuffle_idxs)
    inputs = inputs[shuffle_idxs, :, :]
    labels = labels[shuffle_idxs, :, :]
    
    if masks is not None:
        assert inputs.shape[0] == masks.shape[0]
        masks = masks[shuffle_idxs, :, :]
    else:
        masks = None

    return inputs, labels, masks

def round_to_values(array, round_vals):
    """ Round all elements of an array to closest value in round_vals, numpy version """

    assert len(round_vals.shape) == 1

    dims_unsqueeze = [i for i in range(len(array.shape))] # How many times to unsqueeze round_vals based on array shape
    dists = np.abs(
        np.expand_dims(array, axis=-1) - # shape: (array.shape*, 1)
        np.expand_dims(round_vals, axis=dims_unsqueeze) 
    )

    return round_vals[np.argmin(dists, axis=-1)]

def save_run(params, train_outputs_all, net, task, task_ps, root_path, path_mode='raw_path', overwrite=False, verbose=True):
    """
    Saves the network, training information (via train_outputs_all), and the
    task.
    """

    task_params, train_params, net_params = params

    if path_mode == 'raw_path': # Just add the seed information
        path_end = '_seed' + str(task_params['seed']) + '.pkl'
    else:
        raise NotImplementedError('Path mode {} not recognized'.format(path_mode))

    filename = root_path + path_end

    save_file = True

    directory = os.path.split(filename)[0]
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(filename):
        print('  File already exists at:', filename)
        override = input('Override? (Y/N):')
        if override == 'N':
            save_file = False
        elif override != 'Y':
            raise ValueError(f'Input {override} not recognized!')

    if save_file:
        if verbose:
            print('  Filename:', filename)
        with open(filename, 'wb') as save_file:
            # Saves additional parameters so the network can be re-initialized
            pickle.dump(params, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_outputs_all, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(net, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(task, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(task_ps, save_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('  File not saved!')

def load_run(filename, seed):
    """
    Loads params, train_outputs_all, and net

    filename: should be the filename of the pkl file WITHOUT SEED OR .pkl
    """
    filename = filename + '_seed{}.pkl'.format(seed)

    if not os.path.exists(filename):
        raise ValueError('No file at path:', filename)
    else:
        with open(filename, 'rb') as load_file:
            try:
                params_load = pickle.load(load_file)
            except AttributeError:
                print('Attribute error in params_load!')
                params_load = None
            train_outputs_all_load = pickle.load(load_file)
            try:
                net_load = pickle.load(load_file)
            except AttributeError:
                print('Attribute error in net_load!')
                net_load = None
            task_load = pickle.load(load_file)
            task_ps_load = pickle.load(load_file)

        return params_load, train_outputs_all_load, net_load, task_load, task_ps_load

def save_session(session_idx, params, train_outputs, net, task, task_ps, root_path, path_mode='raw_path', overwrite=False, verbose=True):
    """
    Saves the network, training information (via train_outputs), and the
    task. Just uses save_run setup, but ensures that this is a single session
    and not a list of sessions. 

    Also wipes information that would be repeated in every save from task and task_ps.
    Note that some of this information is already being wiped in the training code,
    so the clearing of task_copy and task_pc_copy hists is repetitive, but kept for
    simplicity.

    This is helpful for saving data from runs that require many sessions, so saves 
    it as sessions progress rather than all at once at the end.
    """

    assert type(train_outputs) == dict

    root_path += '_session_idx{}'.format(session_idx)

    task_copy = copy.deepcopy(task) # Wipe all previous sessions
    for prev_session_idx in range(task_copy.session_idx - 1): # Don't include current session
        task_copy.hists[prev_session_idx] = {} 

    task_ps_copy = copy.deepcopy(task_ps) # Wipe all previous sessions
    if task_ps_copy is not None:
        for prev_session_idx in range(task_ps_copy.session_idx - 1): # Don't include current session
            task_ps_copy.hists[prev_session_idx] = {}

    save_run(params, train_outputs, net, task_copy, task_ps_copy, root_path, path_mode=path_mode, overwrite=overwrite, verbose=verbose)

def load_session(session_idx, filename, seed):
    """
    Loads a single BCI session's data. Note this uses load_run at its core
    but just has some additional checks and file changes to ensure it is 
    only loading a single session.
    """

    filename += '_session_idx{}'.format(session_idx)

    params_load, train_outputs_load, net_load, task_load, task_ps_load = load_run(filename, seed)

    assert type(train_outputs_load) == dict

    return params_load, train_outputs_load, net_load, task_load, task_ps_load

def save_bci_activity(task_params, bci_activity, task, root_path, path_mode='raw_path', overwrite=False, verbose=True):
    """
    Saves just the bci_activity to be used to run a test session to drive the
    network. Used to compute various neuron tunings during one-off sessions.
    """

    if path_mode == 'raw_path': # Just add the seed information
        path_end = '_seed' + str(task_params['seed']) + '.pkl'
    else:
        raise NotImplementedError('Path mode {} not recognized'.format(path_mode))

    filename = root_path + path_end

    save_file = True

    directory = os.path.split(filename)[0]
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(filename):
        print('  File already exists at:', filename)
        override = input('Override? (Y/N):')
        if override == 'N':
            save_file = False
        elif override != 'Y':
            raise ValueError(f'Input {override} not recognized!')

    if save_file:
        if verbose:
            print('  Filename:', filename)
        with open(filename, 'wb') as save_file:
            # Saves additional parameters so the network can be re-initialized
            pickle.dump(task_params, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(bci_activity, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(task, save_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('  File not saved!')

def load_bci_activity(filename, seed):
    """
    Loads bci_activity to be used to run a test session. Also loads task_params
    to check some consistency conditions.

    filename: should be the filename of the pkl file WITHOUT SEED OR .pkl
    """
    filename = filename + '_seed{}.pkl'.format(seed)

    if not os.path.exists(filename):
        raise ValueError('No file at path:', filename)
    else:
        with open(filename, 'rb') as load_file:
            task_params_load = pickle.load(load_file)
            bci_activity_load = pickle.load(load_file)
            task_load = pickle.load(load_file)

        return task_params_load, bci_activity_load, task_load
