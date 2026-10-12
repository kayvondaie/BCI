import numpy as np
import time

from net_helpers import accumulate_decay
from net_helpers import rand_weight_init

class SimpleRNN():
    """
    Numpy implementation of a simple Vanilla RNN. This could probably be
    made a lot more efficient. Note currently just implemented as a 
    feedforward network, so hidden activity needs to be kept track of
    externally and passed every forward pass.

    Includes function to do backward pass using various methods of weight
    adjustment, including loca
    """

    def __init__(self, net_params, task_params, train_params):

        self.n_neurons = net_params['n_neurons']
        self.n_inp = net_params.get('n_inp', self.n_neurons)

        self.last_reward = None # Used to transfer reward between sessions
        self.rng_state = None
        
        self.act_fn_type = net_params.get('act_fn_type', 'ReTanh')
        self.act_fn, self.act_fn_p = self.get_activation_function(self.act_fn_type)

        self.W_inp_type = net_params.get('W_inp_type', 'gaussian')
        self.W_inp_weight_norm = net_params.get('W_inp_weight_norm', 1/np.sqrt(self.n_inp))
        self.W_inp_adjust = net_params.get('W_inp_adjust', False)
        self.W_inp_adjust_mask = None # None by default, just adjust everything if enabled
        self.W_rec_type = net_params.get('W_rec_type', 'gaussian')
        self.W_rec_weight_norm = net_params.get('W_rec_weight_norm', 1/np.sqrt(self.n_neurons))
        self.W_rec_adjust = net_params.get('W_rec_adjust', False)
        self.W_rec_adjust_mask = None # None by default, just adjust everything if enabled
        
        if self.act_fn_type in ('Tanh', 'linear',):
            print('Lowering recurrent spectral radius for stability!!')
            self.W_rec_weight_norm = 0.8 * self.W_rec_weight_norm
        
        # if self.act_fn_type not in ('ReTanh',):
        #     raise NotImplementedError('Assuming activation function is ReTanh')

        self.W_sparsity = net_params.get('W_sparsity', 0.1) # Number of nonzero, 1.0 is equivalent to no sparsity
        self.W_n_low_dim = net_params.get('W_n_low_dim', 10) # Amount of correlation, 0.0 is equivalent to no correlation

        self.adjust_type = train_params['adjust_type']
        self.eligibility_acc_type = train_params['eligibility_acc_type']
        self.n_window_elig = train_params.get('n_window_elig', None)
        self.n_steps_per_loss = train_params.get('n_steps_per_loss', 5)

        self.task_scale = task_params['task_scale'] # Used to set size of Miconi's eligibility

        self.leak_type = net_params.get('leak_type', 'membrane')
        self.leak_term = net_params.get('leak_term', 0.8)

        if net_params['direct_input'] and self.W_inp_adjust:
            raise NotImplementedError('Unclear how this would work if W_inp can be adjusted, assumes it is constant.')

        self.W_inp = rand_weight_init(
            self.n_inp, n_outputs=self.n_neurons, init_type=self.W_inp_type, weight_norm=self.W_inp_weight_norm,
        )
        if self.W_inp_type in ('sparse_gaussian',):
            raise NotImplementedError('Write code to determine sparsity mask')
            # self.W_inp_adjust_mask = 

        self.W_rec = rand_weight_init(
            self.n_neurons, n_outputs=self.n_neurons, init_type=self.W_rec_type, weight_norm=self.W_rec_weight_norm
        )
        if self.W_rec_type in ('sparse_gaussian',):
            raise NotImplementedError('Write code to determine sparsity mask')
            # self.W_inp_adjust_mask = 
        # if self.W_inp_type in ('gaussian',):
        #     self.W_inp_mean = 0.0
        #     self.W_inp = 1 / np.sqrt(self.n_inp) * np.random.normal(loc=self.W_inp_mean, size=(self.n_neurons, self.n_inp))
        #     # if task_params['task_type'] in ('noise_multi_stim', 'noise_constant_rpe_two_stim',): # Makes W_inp block diagonal, so only subset of neurons excited
        #     #     n_inp_over2 = int(np.round(self.n_inp / 2))
        #     #     self.W_inp = 2 * self.W_inp # Renormalizes for halved input unumber
        #     #     self.W_inp[:n_inp_over2, n_inp_over2:] = 0.0
        #     #     self.W_inp[n_inp_over2:, :n_inp_over2] = 0.0
        #     # self.input_mult = 0.25
        #     # print('Adjusting input by {:.2f}'.format(self.input_mult))
        #     # self.W_inp = self.input_mult * self.W_inp
        # elif self.W_inp_type in ('gaussian_low',):
        #     if not hasattr(self, 'low_dim_weights'): # Uses same weights if they've already been generated
        #         print('Generating low-dim weights...')
        #         # self.low_dim_weights = np.random.lognormal(size=(self.n_neurons, self.W_n_low_dim))
        #         # self.low_dim_weights = self.low_dim_weights / np.sum(self.low_dim_weights, axis=-1, keepdims=True) # Each neuron sums to one
        #         self.low_dim_weights = 1 / np.sqrt(self.W_n_low_dim) * np.random.normal(size=(self.n_neurons, self.W_n_low_dim))

        #     W_inp_low = 1 / np.sqrt(self.n_inp) * np.random.normal(size=(self.W_n_low_dim, self.n_inp))
        #     self.W_inp = np.matmul(self.low_dim_weights, W_inp_low)
        #     print('W_inp is low_dimensional {}'.format(self.W_n_low_dim))
        #     # W_correlation = 1 / np.sqrt(self.n_inp) * np.random.normal(size=(1, self.n_inp))
        #     # W_correlation = np.repeat(W_correlation, self.n_neurons, 0) # Same for every single postsynaptic neuron
        #     # self.W_inp = (1 - self.W_correlate) * self.W_inp + self.W_correlate * W_correlation
        # elif self.W_inp_type in ('gaussian_sparse',):
        #     n_inp_mean = self.W_sparsity * self.n_inp # Sets normalization
        #     sparsity_mask = (np.random.uniform(size=(self.n_neurons, self.n_inp)) < self.W_sparsity).astype(np.int32)
        #     self.W_inp_adjust_mask = np.copy(sparsity_mask)
        #     self.W_inp = 1 / np.sqrt(n_inp_mean) * sparsity_mask * np.random.normal(size=(self.n_neurons, self.n_inp))
        # elif self.W_inp_type in ('zeros',):
        #     self.W_inp = np.zeros((self.n_neurons, self.n_inp,))
        # else:
        #     raise ValueError('W_inp_type {} not recognized.'.format(self.W_inp_type))

        # if self.W_rec_type in ('gaussian',):
        #     self.W_rec_mean = 0.0
        #     # self.W_rec_mean = 0.03
        #     # print('Setting nonzero recurrent mean: {:.2f}'.format(self.W_rec_mean))
        #     self.W_rec = 1 / np.sqrt(self.n_neurons) * np.random.normal(loc=self.W_rec_mean, size=(self.n_neurons, self.n_neurons))
        #     # self.rec_mult = 1.25
        #     # print('Adjusting recurrent input by {:.2f}'.format(self.rec_mult))
        #     # self.W_rec = self.rec_mult * self.W_rec
        # elif self.W_rec_type in ('gaussian_low',):
        #     if not hasattr(self, 'low_dim_weights'): # Uses same weights if they've already been generated
        #         print('Generating low-dim weights...')
        #         # self.low_dim_weights = np.random.lognormal(size=(self.n_neurons, self.W_n_low_dim))
        #         # self.low_dim_weights = self.low_dim_weights / np.sum(self.low_dim_weights, axis=-1, keepdims=True) # Each neuron sums to one
        #         self.low_dim_weights = 1 / np.sqrt(self.W_n_low_dim) * np.random.normal(size=(self.n_neurons, self.W_n_low_dim))

        #     W_rec_low = 1 / np.sqrt(self.n_neurons) * np.random.normal(size=(self.W_n_low_dim, self.n_neurons))
        #     self.W_rec = np.matmul(self.low_dim_weights, W_rec_low)
        #     print('W_rec is low_dimensional {}'.format(self.W_n_low_dim))
        #     # W_correlation = 1 / np.sqrt(self.n_neurons) * np.random.normal(size=(1, self.n_neurons))
        #     # W_correlation = np.repeat(W_correlation, self.n_neurons, 0) # Same for every single postsynaptic neuron
        #     # self.W_rec = (1 - self.W_correlate) * self.W_rec + self.W_correlate * W_correlation
        # elif self.W_rec_type in ('gaussian_sparse',):
        #     n_inp_mean = self.W_sparsity * self.n_neurons # Sets normalization
        #     sparsity_mask = (np.random.uniform(size=(self.n_neurons, self.n_neurons)) < self.W_sparsity).astype(np.int32)
        #     self.W_rec_adjust_mask = np.copy(sparsity_mask)
        #     self.W_rec = 1 / np.sqrt(n_inp_mean) * sparsity_mask * np.random.normal(size=(self.n_neurons, self.n_neurons))
        # elif self.W_rec_type in ('log_normal',):
        #     self.W_rec = np.random.lognormal(size=(self.n_neurons, self.n_neurons))
        #     self.W_rec = np.random.choice([1., -1.], size=(self.n_neurons, self.n_neurons)) * self.W_rec
        #     self.W_rec = 1 / np.sqrt(self.n_neurons) * (self.W_rec / np.std(self.W_rec))
        # elif self.W_rec_type in ('perm',):
        #     self.W_rec = np.diag(np.ones((self.n_neurons,)), k=1)[:-1, :-1]
        #     self.W_rec[-1, 0] = 1.
        # elif self.W_rec_type in ('zeros',):
        #     self.W_rec = np.zeros((self.n_neurons, self.n_neurons,))
        # else:
        #     raise ValueError('W_rec_type {} not recognized.'.format(self.W_rec_type))

        # Cumulative quantities used in gradient adjustment
        self.total_W_inp_elg = np.zeros_like(self.W_inp)
        self.total_W_rec_elg = np.zeros_like(self.W_rec)
        self.total_rpe = np.nan # Overriden by first non-NaN RPE

        # Initializes some special eligibility parameters only needed for certain types of adjustment
        if self.adjust_type in ('backprop_two_step',):
            self.dh_dW_inp = np.zeros_like(self.W_inp) # Because of analytic shortcut, this is now just a 2-tensor
            self.dh_dW_rec = np.zeros_like(self.W_rec)
            self.prev_W_inp_elg = None # No leak
            self.prev_W_rec_elg = None
        elif self.adjust_type in ('backprop', 'backprop_sl',): # Needs full eligibility expressions
            self.dh_dW_inp = np.zeros((self.n_neurons, *self.W_inp.shape))
            self.dh_dW_rec = np.zeros((self.n_neurons, *self.W_rec.shape))
            self.delta_neurons = np.eye(self.n_neurons)
            self.prev_W_inp_elg = np.zeros_like(self.W_inp)
            self.prev_W_rec_elg = np.zeros_like(self.W_rec)
        elif self.adjust_type in ('3factor_leak',): # In this case need to keep track of prev elig to compute leak
            self.dh_dW_inp = None
            self.dh_dW_rec = None
            self.prev_W_inp_elg = np.zeros_like(self.W_inp)
            self.prev_W_rec_elg = np.zeros_like(self.W_rec)
        else:
            self.dh_dW_inp = None
            self.dh_dW_rec = None
            self.prev_W_inp_elg = None
            self.prev_W_rec_elg = None

    def forward(self, input_val, prev_activity, prev_activity_pre_act, net_params, 
                perturbation_preact=None, perturbation=None):
        """
        RNN forward pass.

        Note that hidden activity is NOT stored internally, everything 
        is just kept track of externally and passed directly to the network (including the 
        hidden state).

        INPUTS:
        - perturbation_preact: Optional direct injection into hidden state neuron when using
            an input layer. Only used for photostim current injection and certain types of
            node perturbation at the moment. 
            Shape: (self.n_neurons,)
        - perturbation: Optional direction injection onto current activity. Differs from
            direct_input above in that it is applied post-activation. Used for node-perturbation.
            Note this is applied before any leak so on equal footing with noise perturbations.
            Shape: (self.n_neurons,)
        """ 

        W_rec_h = np.matmul(self.W_rec, prev_activity)

        if net_params['direct_input']: # input already takes into account W_inp
            output_pre_act = np.copy(input_val + W_rec_h) 
        else:
            W_inp_x = np.matmul(self.W_inp, input_val)
            output_pre_act = np.copy(W_inp_x + W_rec_h)

            if perturbation_preact is not None: # Optional direction injection 
                output_pre_act += perturbation_preact

        if self.leak_type in ('activity',):
            output_pre_leak = self.act_fn(output_pre_act)
            
            if perturbation is not None: # Node perturbation
                output_pre_leak = output_pre_leak + perturbation
                
            output = self.leak_term * prev_activity + (1. - self.leak_term) * output_pre_leak
        elif self.leak_type in ('membrane',):
            output_pre_act = self.leak_term * prev_activity_pre_act + (1. - self.leak_term) * output_pre_act
            output = self.act_fn(output_pre_act)
            
            if perturbation is not None: # Node perturbation
                output = output + perturbation

        return output, output_pre_act

    def backward(self, task_params, rpe, output_deviation, current_input, prev_activity,
                 act_fn_p_pre_act, prev_avg_activity, output_pre_act, output,
                 output_preact_deviation, current_act_perturbation, current_preact_perturbation,
                 bci_masks,):
        """
        Updates eligibiltiy traces of network. 
        """
        
        if self.adjust_type in ('3factor_sl', 'backprop_sl',):
            if bci_masks.shape[0] > 1:
                raise NotImplementedError('Need to rewrite this code for training with multiple BCI masks!')
            assert np.all(task_params['activity_subtract'] == np.zeros_like(task_params['activity_subtract']))
            assert task_params['z_score_activities'] == False
            bci_mask_idx = 0
            bci_mask = bci_masks[bci_mask_idx, :]
        else:
            bci_mask = None

        # start = time.time()
        if self.W_inp_adjust or self.W_rec_adjust: # Computations for any type of adjustment
            if ~np.isnan(rpe): # NaNs represent time steps where RPE should just be ignored
                # Accumulation of RPE just represents rewards coming in more sparsely from recent time steps
                if np.isnan(self.total_rpe): # Initialization condition, first non-NaN RPE
                    self.total_rpe = np.copy(rpe)
                else:
                    self.total_rpe += rpe

        if task_params['non_bci_deviation_zero']: # Special test to see how non-BCI deviations influence training
            if self.adjust_type in ('3factor_sl', 'backprop_sl',):
                raise ValueError('Using some redundant variable names here, rewrite code for this case.')
            raise NotImplementedError()

            # Project deviations onto BCI mask manifold
            bci_masks = task_params['bci_masks'] # (n_bci_masks, n_neurons)

            # This only works for learnign rules that use deviations currently
            assert self.adjust_type in (
                '3factor', '3factor_leak', '3factor_nophiprime', '3factor_eh',
                '3factor_miconi', '3factor_miconi_insp', '3factor_two_devs',
            )

            bci_proj = np.matmul(np.matmul(bci_masks.T, np.linalg.inv(np.matmul(
                bci_masks, bci_masks.T
            ))), bci_masks) # P_A = A (A.T A)^(-1) A.T (n_neurons, n_neurons)

            output_deviation = np.matmul(bci_proj, output_deviation)
            output_preact_deviation = np.matmul(bci_proj, output_preact_deviation)

        if self.W_inp_adjust:
            input_deviation = None
            if self.adjust_type in ('3factor_predevs', '3factor_two_devs',):
                raise NotImplementedError('Presynaptic deviations for W_inp not yet implemented because mean inputs activity not tracked.')

            W_inp_elg, self.dh_dW_inp, self.prev_W_inp_elg = self.compute_eligibility_trace(
                output_deviation, current_input, act_fn_p_pre_act,
                prev_dh_dW=self.dh_dW_inp, output_activity=output,
                output_pre_act_deviation=output_preact_deviation,
                prev_elig=self.prev_W_inp_elg, act_perturbation=current_act_perturbation,
                preact_perturbation=current_preact_perturbation, input_deviation=input_deviation,
                bci_mask=bci_mask,
            ) # Contribution to current time step eligibility from current activity

            # Add current eligibility to accumulation
            if self.eligibility_acc_type in ('acc_and_wipe',): # BPTT-like
                # Norm by number of steps taken (ensures different modes of accumulation are roughly equal mags)
                self.total_W_inp_elg += W_inp_elg / self.n_steps_per_loss
            elif self.eligibility_acc_type in ('running_average',): # RTRL-like
                self.total_W_inp_elg = accumulate_decay(self.total_W_inp_elg, W_inp_elg, n_window=self.n_window_elig)

        if self.W_rec_adjust:
            input_deviation = prev_activity - prev_avg_activity

            W_rec_elg, self.dh_dW_rec, self.prev_W_rec_elg = self.compute_eligibility_trace(
                output_deviation, prev_activity, act_fn_p_pre_act,
                prev_dh_dW=self.dh_dW_rec, output_activity=output,
                output_pre_act_deviation=output_preact_deviation,
                prev_elig=self.prev_W_rec_elg, act_perturbation=current_act_perturbation,
                preact_perturbation=current_preact_perturbation, input_deviation=input_deviation,
                bci_mask=bci_mask,
            ) # Contribution to current time step eligibility from current activity

            # Add current eligibility to accumulation
            if self.eligibility_acc_type in ('acc_and_wipe',): # BPTT-like
                # Norm by number of steps taken (ensures different modes of accumulation are roughly equal mags)
                self.total_W_rec_elg += W_rec_elg / self.n_steps_per_loss
            elif self.eligibility_acc_type in ('running_average',): # RTRL-like
                self.total_W_rec_elg = accumulate_decay(self.total_W_rec_elg, W_rec_elg, n_window=self.n_window_elig)

    def compute_eligibility_trace(self, output_deviation, input_activity, phi_prime_term,
                              prev_dh_dW=None, output_activity=None,
                              output_pre_act_deviation=None, prev_elig=None, act_perturbation=None,
                              preact_perturbation=None, input_deviation=None, bci_mask=None):
        """
        Computes the current time step eligibility trace.

        Note that if there is a membrane/firing rate time constant, so that these
        quantities depend on the previous quantities, this is implemented here as
        well (though most learning rates assume this contribution is negligible
        and treat the current update as completely due to the current step)
        
        INPUTS:
        - prev_dh_dW: 3-tensor
        - prev_elig: Handles leak-like terms in eligbility computation
            - In the case this is a 3-factor learning rule, 2-tensor that is last eligibility for leak term
            - In the case of deeper assignment, 3-tensor equal to dhtilde_dW
        - bci_masks: Needed for SL setups where the network has knowledge of the BCI mask
        

        """
        # These aren't needed unless doing some multisynapse credit assignment backprop
        dh_dW = None 
        current_elig = None 
        
        if self.adjust_type in ('backprop_two_step',): # This is a more optimized version of two-step backprop

            if self.leak_type not in ('membrane',): # Phi prime only multiplies (1 - alpha) term
                raise NotImplementedError('Wrote all this code assuming membrane leak.')
            
            # Order zero contribution, the true local term which is passed forward in time
            dhtilde_dW_local = phi_prime_term[:, np.newaxis] * input_activity[np.newaxis, :]

            # This is sloppy, but prev_dh_dW is phi^prime_{i, t-1} times h_{j, t-2} 
            # which is really prev_dhtile_dW_local, the delta function contribution has 
            # already been accounted for analytically, which means there is no matmul anymore! 
            # This is still a 3-tensor (kij), but much easier to compute (DEPRICATED, FASTER METHOD BELOW)
            # dhtilde_dW_W_rec_prev = phi_prime_term[:, None, None] * self.W_rec[:, :, None] * prev_dh_dW[None, :, :]
            # W_elig_nonlocal = np.sum(output_deviation[:, None, None] * dhtilde_dW_W_rec_prev, axis=0)
            
            # The most efficient way to do this computation is to sum over k before creating a 3-tensor
            i_tensor = np.sum((output_deviation * phi_prime_term)[:, None] * self.W_rec, axis=0)
            W_elig_nonlocal = i_tensor[:, None] * prev_dh_dW
            
            # Note this assumes uniform credit assignment
            W_elig_local = output_deviation[:, None] * dhtilde_dW_local
            
            W_elig = W_elig_local + W_elig_nonlocal
            
            # Passed forward to next time step to multiply by the recurrent weights 
            dh_dW = dhtilde_dW_local
            # No leak, so no need to pass forward current_elig, only "dh_dW"
            current_elig = None
        elif self.adjust_type in ('backprop', 'backprop_sl', 'backprop_two_step',):
            
            if self.adjust_type in ('backprop_two_step',):
                raise NotImplementedError('Depricated, shortcut code above that yields same results much quicker.')
            
            if self.leak_type not in ('membrane',): # Phi prime only multiplies (1 - alpha) term
                raise NotImplementedError('Wrote all this code assuming membrane leak.')
            
            # dhtilde_dW_local = np.einsum('ki, J -> kiJ',
            #     self.delta_neurons, # delta_ij
            #     input_activity
            # )
            dhtilde_dW_local = self.delta_neurons[:, :, None] * input_activity[None, None, :]

            # prev_dh_dW is derivative of the hidden activity with respect to W (i.e.
            # it is not h tilde), so already includes phi^prime contribution
            # Doing matmul here with transposes (for correct matmul format) is way quicker than equivalent einsum,
            # which is effectively doing kl, lij -> kij
            # Transpose does lij -> ilj, matmul does kl, ilj -> ikj, final transpose does ikj -> kij
            dhtilde_dW_W_rec_prev = np.transpose(np.matmul(self.W_rec, np.transpose(prev_dh_dW, (1, 0, 2))), (1, 0, 2))
            # dh_dW_W_prev = np.einsum('kl, lij -> kij', W, dh_dW_rec_test)

            if self.adjust_type in ('backprop_two_step',): # No leak, doesn't use prev_elig but still assigns for below
                dhtilde_dW = dhtilde_dW_local + dhtilde_dW_W_rec_prev
            else: # Leak contribution
                dhtilde_dW_no_leak = (1 - self.leak_term) * (dhtilde_dW_local + dhtilde_dW_W_rec_prev)
                dhtilde_dW_leak = self.leak_term * prev_elig # prev elig is prev P_{kij}

                dhtilde_dW = dhtilde_dW_leak + dhtilde_dW_no_leak # Equiv to P_{kij} for membrane leak

            # Now convert to dh_dW by multiplying by current phi_prime
            # dh_dW = np.einsum('k, kiJ -> kiJ',
            #     phi_prime_term,
            #     dhtilde_dW,
            # )
            dh_dW = phi_prime_term[:, None, None] * dhtilde_dW
            
            if self.adjust_type in ('backprop', 'backprop_two_step',): # Multiply by activity deviation
                # Note the sum here assumes uniform credit assignment
                # W_elig = np.sum(np.einsum('k, kij -> kij', output_deviation, dh_dW), axis=0)
                W_elig = np.sum(output_deviation[:, None, None] * dh_dW, axis=0)

                if self.adjust_type in ('backprop_two_step',): # Replaces dh_dW that is passed to next iteration with just local term
                    # dh_dW = np.einsum('k, kiJ -> kiJ',
                    #     phi_prime_term,
                    #     dhtilde_dW_local,
                    # )
                    dh_dW = phi_prime_term[:, None, None] * dhtilde_dW_local

                current_elig = dhtilde_dW # Somtimes current_elig is 2-tensor below, so can't just call output dhtilde_dW
            elif  self.adjust_type in ('backprop_sl',):
                W_elig = np.sum(bci_mask[:, None, None] * dh_dW, axis=0) # Sum weighted by actual BCI mask instead of activity deviations
                current_elig = dhtilde_dW # Somtimes current_elig is 2-tensor below, so can't just call output dhtilde_dW
                
        elif self.adjust_type in ('3factor',):
            W_elig = output_deviation[:, np.newaxis] * (
                phi_prime_term[:, np.newaxis] * input_activity[np.newaxis, :]
            )
        elif self.adjust_type in ('3factor_rawpost',):
            # CONTROL variant: RAW post factor (no baseline subtraction) instead of
            # the deviation -- i.e. train with r_pre * phi' * r_post.
            W_elig = output_activity[:, np.newaxis] * (
                phi_prime_term[:, np.newaxis] * input_activity[np.newaxis, :]
            )
        elif self.adjust_type in ('3factor_leak',):
            if net_params['leak_type'] in ('activity',): # Phi prime only multiplies (1 - alpha) term
                current_elig = net_params['leak_term'] * prev_elig + (1. - net_params['leak_term']) * (
                    phi_prime_term[:, np.newaxis] * input_activity[np.newaxis, :]
                )
                W_elig = output_deviation[:, np.newaxis] * current_elig
            elif net_params['leak_type'] in ('membrane',): # Phi prime multiplies all terms
                current_elig = net_params['leak_term'] * prev_elig + (1. - net_params['leak_term']) * input_activity[np.newaxis, :]
                W_elig = output_deviation[:, np.newaxis] * phi_prime_term[:, np.newaxis] * current_elig
        elif self.adjust_type in ('3factor_nophiprime',):
            W_elig = output_deviation[:, np.newaxis] * input_activity[np.newaxis, :]
        elif self.adjust_type in ('3factor_two_devs',):
            W_elig = output_deviation[:, np.newaxis] * input_deviation[np.newaxis, :]
        elif self.adjust_type in ('3factor_predevs',):
            W_elig = output_activity[:, np.newaxis] * input_deviation[np.newaxis, :]
        elif self.adjust_type in ('3factor_nodev',):
            W_elig = output_activity[:, np.newaxis] * (
                phi_prime_term[:, np.newaxis] * input_activity[np.newaxis, :]
            )
        elif self.adjust_type in ('3factor_vanilla_hebbian',):
            W_elig = output_activity[:, np.newaxis] * input_activity[np.newaxis, :]
        elif self.adjust_type in ('3factor_eh',):
            W_elig = output_pre_act_deviation[:, np.newaxis] * input_activity[np.newaxis, :]
        elif self.adjust_type in ('3factor_miconi',): # Noramlization puts it on roughly the same scale as other learning rules, but still need some adjustment
            W_elig = (output_pre_act_deviation[:, np.newaxis] * input_activity[np.newaxis, :])**3 / (self.task_scale**4)
        elif self.adjust_type in ('3factor_miconi_insp',): # Noramlization puts it on roughly the same scale as other learning rules, but still need some adjustment
            W_elig = (output_deviation[:, np.newaxis] * input_activity[np.newaxis, :])**3 / (self.task_scale**4)
        elif self.adjust_type in ('3factor_node_pert',):
            W_elig = act_perturbation[:, np.newaxis] * input_activity[np.newaxis, :]
        elif self.adjust_type in ('3factor_node_pert_pre',):
            W_elig = preact_perturbation[:, np.newaxis] * input_activity[np.newaxis, :]
        elif self.adjust_type in ('3factor_sl',):
            W_elig = bci_mask[:, None] * phi_prime_term[:, np.newaxis] * input_activity[np.newaxis, :]
        elif self.adjust_type in ('3factor_sl_uniform',): # Same as above, just no BCI mask
            W_elig = phi_prime_term[:, np.newaxis] * input_activity[np.newaxis, :]
        elif self.adjust_type in ('backprop_sl',):
            raise NotImplementedError('Still need to implement this, its quite complex')
        else:
            raise ValueError('Adjust type {} not recognized.'.format(self.adjust_type))

        return W_elig, dh_dW, current_elig

    def loss_step(self, task_params, train_params):

        # Normalize RPE by number of steps taken
        self.total_rpe = self.total_rpe / train_params['n_steps_per_loss']

        # RPE clip, stabilizes training a bit (don't do this if holding RPE constant)
        if train_params['rpe_clip'] is not None and  task_params['task_type'] not in ('noise_constant_rpe', 'noise_constant_rpe_two_stim', 'noise_constant_rpe_two_stim_alt', 'noise_multi_stim'):
            self.total_rpe = np.clip(self.total_rpe, -1 * train_params['rpe_clip'], train_params['rpe_clip'])

        if self.W_inp_adjust:
            if np.isnan(self.total_rpe): # No RPE-relevant weight updates
                delta_W_inp = np.zeros_like(self.W_inp)
            else:
                delta_W_inp = train_params['eta'] * self.total_rpe * self.total_W_inp_elg

            if train_params['weight_reg'] in ('L2',):
                delta_W_inp += -1 * train_params['reg_lambda'] * self.W_inp

            if self.W_inp_adjust_mask is not None:
                delta_W_inp = self.W_inp_adjust_mask * delta_W_inp

            self.W_inp = self.W_inp + delta_W_inp

        if self.W_rec_adjust:
            if np.isnan(self.total_rpe): # No RPE-relevant weight updates
                delta_W_rec = np.zeros_like(self.W_rec)
            else:
                delta_W_rec = train_params['eta'] * self.total_rpe * self.total_W_rec_elg

            if train_params['weight_reg'] in ('L2',):
                delta_W_rec += -1 * train_params['reg_lambda'] * self.W_rec

            # if net_params['W_rec_adjust_mask'] is not None:
            #     delta_W_rec = net_params['W_rec_adjust_mask'] * delta_W_rec
            if self.W_rec_adjust_mask is not None:
                delta_W_rec = self.W_rec_adjust_mask * delta_W_rec

            self.W_rec = self.W_rec + delta_W_rec

    def set_grads_for_next_step(self):
        """ Reset some values now that credit assignment is finished """

        self.total_rpe = np.nan
        if self.eligibility_acc_type in ('acc_and_wipe',):
            self.total_W_inp_elg = np.zeros_like(self.W_inp)
            self.total_W_rec_elg = np.zeros_like(self.W_rec)

        # # Commenting this out so this is as close to 3-factor as possible,
        # # weird if this just wipes these quantities every couple of steps
        # if train_params['adjust_type'] in ('backprop',): # Resets these so not calculating with respect to old weights
        #     self.dh_dW_inp = np.zeros((task_params['n_neurons'], *self.W_inp.shape))
        #     self.dh_dW_rec = np.zeros((task_params['n_neurons'], *self.W_rec.shape))
    
    
    def tanh(self, x):
        return np.tanh(x)
    def tanh_p(self, x):
        return 1 / np.cosh(x)
    def retanh(self, x):
        return np.maximum(np.tanh(x), np.zeros_like(x))
    def retanh_p(self, x):
        return np.where(x > 1e-6, (1/np.cosh(x))**2, 0.)
    def relu(self, x):
        return np.maximum(x, np.zeros_like(x))
    def relu_p(self, x):
        return np.heaviside(x, np.zeros_like(x))
    def linear_fn(self, x):
        return x     
    def linear_fn_p(self, x):
        return np.ones_like(x)
    
    def get_activation_function(self, act_fn_type):
        """ 
        
        OUTPUTS:
        - self.act_fn
        - self.act_fn_p
        """
        
        if act_fn_type in ('ReTanh',):
            return self.retanh, self.retanh_p
        elif act_fn_type in ('Tanh',):
            return self.tanh, self.tanh_p
        elif act_fn_type in ('ReLU',):
            return self.relu, self.relu_p
        elif act_fn_type in ('linear',):
            return self.linear_fn, self.linear_fn_p
        else:
            raise ValueError('act_fn_type {} not recognized!'.format(act_fn_type))


    