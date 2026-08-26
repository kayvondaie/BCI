import numpy as np
import gymnasium as gym

from scipy.signal import fftconvolve

from net_helpers import get_stimulus

class Noise_Base(gym.Env):
    """
    Noise base for several distinct BCI environment setups to inject noise into networks.
    Can optionally generate time-correlated noise.

    Here the "state" of the environment is mutli-dimensional noise input. The "action" 
    of the agent has no effect on the state, but see children for actual interactions.
    """

    def __init__(self, task_params):
        """
        
        INPUTS:
        - task_params
            - noise_type: iid, tc_weight, None
                - iid: equal weight across neurons, no time-correlation
                - tc_weight: time_correlated and potentially distinctly weighted across neurons
                - None: no noise injected
            - task_scale: size of noise
            - stim_to_noise_ratio: size of noise relative to stimulus, further scales size of noise
            - n_inp/n_neurons: determines number of neurons noise will be injected into
            - steps_stored: number of noise steps to generate at once, needed to correlated noise over time-steps

        OUTPUTS:
        None

        """

        self.noise_type = task_params.get('noise_type', 'iid') # type of noise injected into each noise_dim

        self.task_scale = task_params.get('task_scale', 0.1) # size of noise
        self.stim_to_noise_ratio = task_params.get('stim_to_noise_ratio', 1.0) # size of noise relative to stimuluss

        if 'n_inp' in task_params: # Default to n_inp
            self.noise_dim = task_params['n_inp']
        else:
            self.noise_dim = task_params['n_neurons']

        self.steps_stored = task_params.get('steps_stored', 1000) # Number of noise steps to generate internally at once

        if self.noise_type in ('iid',):
            self.kernel_len = 1 # Effective kernel length is one, but won't actually use a kernel
        elif self.noise_type in ('tc_weight',):
            self.noise_timescale = task_params.get('noise_timescale', 100) # Units of ms

            self.n_corr_scale = int(np.round(self.noise_timescale /  task_params['t_step']))  # Translates to units of idxs
            kernel_steps = np.arange(-self.n_corr_scale, self.n_corr_scale+1)
            if self.n_corr_scale > 0:
                self.smoothing_kernel = ( # Normalization here is so that correlated noise has same variance as uncorrelated
                    1/(np.pi*self.n_corr_scale)**(1/4)*np.exp(-kernel_steps**2 / (2 * self.n_corr_scale))
                 ) 
            else:
                self.smoothing_kernel = np.array((1.,))

            self.kernel_len = 2 * self.n_corr_scale + 1
        elif self.noise_type in (None,):
            self.noise_dim = 0
        else:
            raise NotImplementedError('Noise type {} not recognized.'.format(self.noise_type))

    def reset(self):
        """
        This only resets the noise-relevant parts, should be called before child resets.

        INPUT:
        None

        OUTPUT:
        initial_state: initial noise fed into network
        """

        if self.noise_type in (None,): # Exit if no noise is generated
            return None

        # Determines amount of noise to inject into each neuron
        if self.noise_type in ('iid',):
            self.input_noise_weights = np.ones((self.noise_dim,), dtype=np.int32)
        elif self.noise_type in ('tc_weight',):
            self.input_noise_weights = np.ones((self.noise_dim,), dtype=np.int32) # For now just uniform, can change later

        steps_to_generate = self.steps_stored + self.kernel_len - 1

        if self.stim_to_noise_ratio > 1.0: # Reduces noise amplitude relative to stim amplitude
            self.noise_scale = 1 / self.stim_to_noise_ratio * self.task_scale
        else: # Reduces stim amplitude (elsewhere) relative to noise amplitude
            self.noise_scale = self.task_scale

        self.raw_noise = self.input_noise_weights * np.random.normal(
            loc=0.0, scale=self.noise_scale, 
            size=(steps_to_generate, self.noise_dim,)
        )

        self.seq_idx_noise = 0

        if self.noise_type in ('iid',):
            self.stored_noise = np.copy(self.raw_noise)
            del self.raw_noise
        elif self.noise_type in ('tc_weight',):
            self.stored_noise = fftconvolve(
                self.raw_noise, self.smoothing_kernel[:, np.newaxis], mode='valid', axes=0
            )
            # Now store raw_noise that could be used in future, delete rest
            self.raw_noise_stored = self.raw_noise[-self.kernel_len + 1:, :]
            del self.raw_noise

            # print('Raw noise shape:', self.raw_noise.shape)
            # print('Kernel shape:', self.smoothing_kernel.shape, 'n_corr_scale:', self.n_corr_scale)
            # print('Time correlation noise shape:', self.stored_noise.shape)

        return self.stored_noise[self.seq_idx_noise, :], {}


    def step(self, state):
        """
        Generate noise for next "state" to pass to agent. 

        INPUTS:
        - state: since noise generated does not depend on state, not used, but follows usual step(state) format

        OUTPUS:
        - usual gym output format
        """

        if self.noise_type in (None,): # Exit if no noise is generated
            return None, None, None, None, None

        if self.seq_idx_noise == self.steps_stored - 1: # Need to generate more noise

            self.seq_idx_noise = 0 # Reset index

            # Generate new raw noise: put stored at front, then regenerate rest
            self.raw_noise = np.zeros((self.steps_stored + self.kernel_len - 1, self.noise_dim,))

            self.raw_noise[self.kernel_len-1:, :] = self.input_noise_weights * np.random.normal(
                loc=0.0, scale=self.noise_scale, 
                size=(self.steps_stored, self.noise_dim)
            )

            if self.noise_type in ('iid',):
                self.stored_noise = np.copy(self.raw_noise)
                del self.raw_noise
            elif self.noise_type in ('tc_weight',):
                self.raw_noise[:self.kernel_len-1, :] = self.raw_noise_stored # put stored at front if needed

                self.stored_noise = fftconvolve(
                    self.raw_noise, self.smoothing_kernel[:, np.newaxis], mode='valid', axes=0
                )

                # Now store raw_noise that could be used in future, delete rest
                self.raw_noise_stored = self.raw_noise[-self.kernel_len + 1:, :]
                del self.raw_noise
        else:
            self.seq_idx_noise += 1 # First step call, will return seq_idx = 1 

        return (
            self.stored_noise[self.seq_idx_noise, :],
            None, 
            None, 
            None,
            None,
        )

class BCI_Base(Noise_Base):
    """
    BCI Base for several distinct BCI environment setups. Primarily:
    1. Initializes a lot of attributes that are shared across the classes.
    2. Contains all the attributes/functions for pre-session observations
    3. Contains function for iterating through sessions
    """
    
    def __init__(self, task_params):
        """

        INPUTS:
        - task_params
            - n_inp: sets size of state space
            - n_neurons: sets size of action space
            - n_steps_stabilize: number of pre-session steps
            - reward_mode: distinct types of reward generation
                - water_only: rewards only when water is given
                - water_and_spout: water + rewards when spout moves
                - thirst: default + increasing reward for longer time periods between rewards
                - spout_and_thirst: default + both the above
            - dt: time of each step of environment
        """

        super().__init__(task_params) # Noise initialization
        
        self.session_idx = 0 # Keep track of multiple sessions at once within the same environment

        self.n_inp = task_params.get('n_inp', 1) # Contributes to size of state space (input activity)
        self.n_neurons = task_params.get('n_neurons', 1) # Size of action space (hidden activity)

        # Initiates presession period where certain parameters tracked
        self.n_steps_stabilize =  task_params.get('n_steps_stabilize', False) 
        
        self.activity_offset = task_params.get('activity_offset', 0.0) # Manual offset of treshold/max_activity in units of (max_activity - median)
        self.reward_mode = task_params.get('reward_mode', 'water_only') # water_only, water_and_spout, thirst, spout_and_thirst
        self.nan_mask_mode = task_params.get('nan_mask_mode', None) # None, all_but_reward

        self.dt = task_params.get('t_step', 50) / 1000 # ms -> sec
        
        np.random.seed(task_params.get('seed', 0))
        self.seq_idx = 0 # Sequence index within session
        self.seq_since_reward = 0 # Used for thirst-based rewards
        self.States = None
        
        if self.n_steps_stabilize > 0:
            self.threshold = None
            self.threshold_upper = None
        else:
            raise NotImplementedError()
            self.threshold = task_params.get('threshold', 10)        # df/f threshold for lickspout to move 
            self.threshold_upper = task_params.get('max_activity', 1.) # max velocity occurs at max activity
         
        # This needs to be determined here in order to figure out how to initialize history
        self.threshold_change_type = task_params.get('threshold_change_type', 'fixed')
        if self.threshold_change_type in ('fixed',):
            self.dynamic_threshold = False # Whether or not thresholds change during the task
        elif self.threshold_change_type in ('seq_idxs',): # For all settings where threhsold can change
            self.dynamic_threshold = True
            
            ### Determines conditions for a threshold change ###
            if self.threshold_change_type in ('seq_idxs',):
                self.threshold_change_idxs = list(task_params.get('threshold_change_idxs')) # Somtimes this is modified
                assert type(self.threshold_change_idxs) == list
                self.threshold_change_intertrial = task_params.get('threshold_change_intertrial', False) 
            
            ### Determines amount of threshold change ###
            # This will be used in a function call to determine threshold change given current upper/lower thresholds
            self.threshold_change_mag_type = task_params.get('threshold_change_mag_type', 'fixed')
            self.threshold_change_mag_params = task_params.get('threshold_change_magnitude', (1.5, 0.0))
            assert type(self.threshold_change_mag_params) == tuple
            assert len(self.threshold_change_mag_params) == 2
        else:
            raise ValueError('Threshold change type {} not recognized.'.format(self.threshold_change_type))

        self.reward = None         # reward signal (internal sense of reward, not literal water rewards)

        # Presession state attr
        self.ps = None
        
        # Init history trackers
        self.hists = [self.init_new_hist(),]
        self.hist = self.hists[self.session_idx] # Sets current hist

        # These currently just stay false at all times
        self.done = False 
        self.terminated = False

    def init_new_hist(self):
        """ Create a base history dict for a new session. """

        session_hist = {
            'world_trajectory': [],
        }

        return session_hist

    def set_new_session(self):
        """ Iterates to a new session. Note reset needs to be called separately after this """

        # Checks to make sure the previous session actually ran
        assert len(self.hist['world_trajectory']) > 0

        self.session_idx += 1
        self.hists.append(self.init_new_hist())

        self.hist = self.hists[self.session_idx] # Update current hist

        if self.n_steps_stabilize > 0:
            self.threshold = None
            self.threshold_upper = None
        else:
            raise NotImplementedError()

        
    def render(self):
        raise NotImplementedError()


class BCI_Env(BCI_Base):
    """
    Main BCI enviroment for spout movement task. 

    In our setups the "state space" is the size of the input into the network, which contains noise 
    and encodings of stimulus representations. The "action space" of the agent is the BCI activity,
    which is the activity of a particular layer of neurons, which have been converted into the BCI 
    activity via the BCI mask. This activity alone drives the task. The form of the reward is 
    generated interally and directly passed to the agent, with various options for what exactly 
    encodes reward (e.g. spout movments).

    The same enviroment is used to track multiple sessions, because things like the mouse's conversion 
    of the state input are assumed to remain constant across multiple sessions.

    Task structure:
    - presession -> trial -> hit/miss
        - hit: trial -> reward -> pre-trial -> intertrial -> spout reset -> trial
        - miss: intertrial -> spout reset -> trial

    States:
    - presession: Used to measure "spontaneous activity" and set difficulty of the task
    - trial: Requires activity to go high to move spout, can result in hit or miss
    - reward: Water reward for the mouse, always a source of reward feedback
    - pre-trial: Requires activity to go low to reset task
    - intertrial: Preset waiting time
    - spout-rest: Spout moves backwards ot reset task

    """
    
    def __init__(self, task_params):

        super().__init__(task_params)  

        self.States = ['pretrial', 'trial', 'reward', 'inter-trial', 'reset_spout', 'presession']

        ### Various ways the stimuli are coded into the input signal ###
        # - mix_... indicates the stimuli signal is added to the noise input
        # - sep_... indicates the stimuli signal is one-hot and has its own input neurons (old way of doing input)
        #
        # - mix_spout_loc: responses based on spout location, 2d input allows for all locations to have the 
        #     same stimulus input magnitude
        # - mix_spout_loc_1d: responses based on spout location, so fully yields largest response, goes to zero 
        #     as it approaches reward location. No signal during backwards movement to mirror Matt's setup
        # - mix_spout_movement: response based on magnitude of spout movement, but signed so forward/backward
        #     yield opposite responses
        # - mix_spout_movement_abs: response based on magnitude of spout movement, only speed dependent so
        #     forward/backward yield the same response
        self.state_mode = task_params.get('state_mode', 'sep_spout_loc') # mix_spout_loc, mix_spout_movement, sep_spout_loc, sep_spout_movement

        # Simplified version of task that does not require oscillation
        # This will simply make it so after a reward the task transitions
        # directly to inter-trial, instead of having pre-trial (which 
        # requires low activity) before
        self.simple_states = task_params.get('simple_states', False) 

        self.track_raw_states = task_params.get('track_raw_states', False) # Tracks stimulus representations, useful for mix_... state modes.

        assert task_params['reward_structure'] == 'trapezoid'
        self.reward_structure = task_params.get('reward_structure', 'trapezoid')
        
        if self.state_mode in ('mix_spout_loc', 'mix_spout_loc_1d', 'mix_spout_movement', 'mix_spout_movement_abs',): # Stimuli high-dimensional and added to noise
            if self.state_mode in ('mix_spout_loc',):
                self.n_stim = 4 
                self.l_dl_to_state = self.l_to_theta_to_state
            elif self.state_mode in ('mix_spout_loc_1d',):
                self.n_stim = 3 
                self.l_dl_to_state = self.l_to_state
            elif self.state_mode in ('mix_spout_movement',):
                self.n_stim = 3
                self.l_dl_to_state = self.dl_to_state 
            elif self.state_mode in ('mix_spout_movement_abs',):
                self.n_stim = 3 
                self.l_dl_to_state = self.dl_to_state_abs 
            self.state_dim = np.copy(self.noise_dim)

            self.stim_input = np.zeros((self.n_stim, self.n_inp,))
            for stim_idx in range(self.n_stim): # Generate stimuli corresponding to distinct inputs
                # The size of this determines how much stimuli dominate activity over noise inputs
                if self.stim_to_noise_ratio < 1.0: # Ensures maximum scale of stim and noise is roughly task_scale
                    stim_mult = self.stim_to_noise_ratio # Reduces stim relative to noise
                else:
                    stim_mult = 1.0 # Reduces noise relative to stim
                
                self.stim_input[stim_idx:stim_idx+1, :] = stim_mult * get_stimulus(task_params)

        elif self.state_mode in ('sep_spout_loc', 'sep_spout_movement',): # Stimuli are one-hots, separate from noise  
            self.stim_input = None
            if self.state_mode in ('sep_spout_loc'):
                self.n_stim = 4
                self.l_dl_to_state = self.l_to_theta_to_state
            elif self.state_mode in ('sep_spout_movement'):
                self.n_stim = 3
                self.l_dl_to_state = self.dl_to_state     
            self.state_dim = self.n_stim + self.noise_dim

        else:
            raise NotImplementedError('State mode {} not recognized.'.format(self.state_mode))
            
        ### Various internally tracked state attributes ###
        # Pretrial state attr
        self.p = None             # pretrial below-threshold time (sec)
        self.pretrial_wait  = 0.2 # amount of time activity needs to be below threshold to leave pretrial (sec) (200 ms in original experiments, upped to 500 ms in blocking exp)

        # Trial state attr
        self.t = None             # time within a trial (sec)
        self.l = None             # location of lickspout (units of distance, l < 0 reward, L0 = inital condition)
        self.dl = None            # change in location of lickspout
        self.aborttrial = 10      # maximum time within a trial (sec, 10 seconds in experiment)
        self.L0 = 3               # initial loation of spout (a.u.'s of distance)
        
        self.max_velocity = self.L0 / 0.7           # dist/sec (Kayvon/Marton: at max velocity spout takes about 700 ms to reach mouse)
        self.max_velocity_back = self.L0 / 0.5      # dist/sec (Kayvon/Marton: speed backwards is faster because it doesn't have to start/stop)
        self.tonelength = 0.05                      # time of tone at start of trial, sec (50 ms in experiment)
        self.tone = None                            # binary of whether or not tone is playing

        # Reward state attr
        self.reward_present = None                      # binary of whether reward is present (part of state)
        self.w = None                                   # time within a reward (sec)
        self.t_reward = 1.                              # total time of reward period (sec, 1 second in experiment)
        self.reward_per_step = 0.2 * self.task_scale    # 0.2 here is arbitrary, but effectively scaled by learning rate too so okay. Chosen to match toy task.
        self.total_trial_reward = self.reward_per_step * self.t_reward / self.dt  # Total reward recieved in a successful trial
        
        # This parameter is used to set the scale of spout movement rewards. It is set relative to the
        # total reward the agent recieves for a water reward (time-integrated). A full spout movement
        # from reset to reward will result in this multiple of the water reward.
        self.spout_movement_reward_scale = task_params.get('spout_movement_reward_scale', 0.25)
        
        # Amount of seconds where time integrated negative reward equals total trial reward
        self.thirst_reward_timescale = task_params.get('thirst_reward_timescale', 500) 
        # Inst. reward at thirst_reward_timescale, when total negative reward = total_trial_reward (solve total rew = 1/2 * time_steps * inst rew)
        self.thirst_reward_at_timescale =  2 * self.total_trial_reward / (self.thirst_reward_timescale / self.dt)

        # Inter-trial state attr
        self.it = None              # time within an inter-trial (sec)
        self.it_time = 1.0          # inter-trial length (sec, 1 second in experiment)

    def init_new_hist(self):
        """ Reset history for new session """
        
        session_hist = super().init_new_hist() 

        session_hist['trial_starts'] = []
        session_hist['trial_ends'] = [] # seq_idxs of trial ends, where hit or miss is determined
        session_hist['rewards'] = [] # seq_idxs of rewards
        session_hist['trial_outcomes'] = [] # for each trial, +1 for reward, 0 for no reward
        session_hist['pretrial_starts'] = []
        session_hist['pretrial_ends'] = []
        session_hist['raw_states'] = []
        
        if self.dynamic_threshold:
            session_hist['threshold_changes'] = []
            session_hist['threshold_vals'] = []
            session_hist['threshold_trials'] = [] # Trial where threshold change is first used (could be mid-trial)

        return session_hist
        
    def input_to_velocity(self, bci_activity): 
        """ 
        Convert BCI activity to spout velocity. 

        Spout velocity does not move below threshold and moves at maximum velocity
        when above threshold_upper, otherwise linear interpolation between these two.

        """
        if bci_activity >= self.threshold_upper:
            return self.max_velocity
        elif bci_activity < self.threshold:
            return 0.
        else:
            return self.max_velocity * (bci_activity - self.threshold) / (self.threshold_upper - self.threshold)

    def copy_attributes(self, bci_env):
        """
        Copy all state-relvant attributes from one bci_env to this environment so 
        that it can be run by the BCI acitivty and produce the same state transitions.
        """
        self.threshold = bci_env.threshold
        self.threshold_upper = bci_env.threshold_upper

    ### Various types of reward functions ###
    def l_to_theta_to_state(self, l, dl): # Remap the lickspout position into state observation
        theta = l / self.L0 * np.pi/2
        return (np.sin(theta), np.cos(theta))
    def l_to_state(self, l, dl): # 1 when at max position, 0 when at reward position
        if dl > 0: # Backwards spout movement don't give any input
            return (0. * l,)
        else:
            return (l / self.L0,)
    def dl_to_state(self, l, dl): # 1 when at max velocity forward, 0 when not moving, slightly larger than -1 when moving backwards
        return (-1 * dl / (self.max_velocity * self.dt),)
    def dl_to_state_abs(self, l, dl): # 1 when at max velocity forward, 0 when not moving, slightly larger than 1 when moving backwards
        return (np.abs(dl) / (self.max_velocity * self.dt),)

    def build_full_state(self, state, noise):
        """
        Combines the noise and the state to build the full state. Note there are
        several different types of state representations. One large factor is
        whether or not the variables representing the lick spout position/movement,
        trial start tone, or reward indicator are mixed with the noise dimensions
        or not.

        state.shape: 3 or 4 - (tone, reward_present, *l_dl_to_state)
        """

        if self.state_mode in ('mix_spout_loc', 'mix_spout_loc_1d', 'mix_spout_movement', 'mix_spout_movement_abs',): # Stimuli are added to noise
            # Convert stimuli variables to their representations using matrix multiplication
            state = np.matmul(state, self.stim_input)

            if self.noise_type in (None,):
                return state
            else:
                return state + noise
        elif self.state_mode in ('sep_spout_loc', 'sep_spout_movement',): # States unchanged, just concatenate if needed
            if self.noise_type in (None,):
                return state
            else:
                return np.concatenate((state, noise), axis=0)

    def get_nan_mask(self):
        """ Different ways to mask reward with nans to trigger no plasticity at certain time steps """

        if self.nan_mask_mode is None:
            return 1.
        elif self.nan_mask_mode in ('all_but_reward',):
            if self.state in (self.States[2],): # reward
                return 1.
            else:
                return np.nan
        else:
            raise NotImplementedError(
                'nan_mask_mode {} not recognized.'.format(self.nan_mask_mode)
            )
        
    def reset(self):

        noise, _ = super().reset()  

        self.seq_idx = 0
        self.l = self.L0
        self.dl = 0.
        self.tone = 0.
        self.reward_present = 0.

        self.seq_since_reward = 0

        if self.n_steps_stabilize > 0: # Start in presession
            self.state = self.States[-1]
            self.statenum = -1
            self.ps = 0
            self.reward = np.nan
        else:
            self.state = self.States[1] # Start in trial
            self.statenum = 1
            self.t = 0 # Reset trial time
            self.hist['trial_starts'].append(1) 
            self.reward = 0.

        state = np.array([self.tone, self.reward_present, *self.l_dl_to_state(self.l, self.dl)])

        if self.track_raw_states:
            self.hist['raw_states'].append(np.array([self.tone, self.reward_present, self.l]))

        full_state = self.build_full_state(state, noise)

        return full_state, {'oracle_signal': np.nan,}
    
    def step(self, bci_activity):
        """    
         
        INPUTS:
        - bci_activity: 1d BCI activity
            - Note that in presession, before the BCI mask has been determined, bci_activity 
              is not defined and so is just None. This is okay becasue presession is not
              dependent upon bci_activity anyway, so it is not relevant.
        """

        noise, _, _, _, _ = super().step(None)  

        self.hist['world_trajectory'].append(self.statenum) # store the computer state at everytime point for future analysis

        self.evaluate_threshold_change() # See if the task needs to change or not
        
        # Do this before trial updates
        nan_mask = self.get_nan_mask()

        oracle_signal = np.nan # supervised-like-training hack of this enviroment
        
        if self.state == self.States[0]: # pretrial
            self.p += self.dt*(bci_activity < self.threshold) - self.p*(bci_activity > self.threshold)   # resets p if above threshold, otherwise progresses time
            if self.p > self.pretrial_wait: # transition to inter-trial
                self.state = self.States[3] 
                self.statenum = 3
                self.it = 0.
                self.hist['pretrial_ends'].append(self.seq_idx)

            self.seq_since_reward += 1
            self.reward = 0
            self.reward_present = 0

            oracle_signal = max(0., self.threshold - self.threshold_upper) # used only for supervised-like-training hack of this enviroment
                
        elif self.state == self.States[1]: # trial
            self.t += self.dt # Iterate trial time
            self.tone = float(self.t<=self.tonelength) # Tone plays at start of trial
            self.dl = -self.dt * self.input_to_velocity(bci_activity) # integrates velocity of lickspout with Euler integration
            if self.l + self.dl < 0: # Catches movement beyond limit
                self.dl = -1 * self.l
            self.l += self.dl    

            if self.l <= 0: # Reward condition
                self.state = self.States[2] # Transition to reward
                self.statenum = 2
                self.l = 0 # Lickspout set to reward position
                self.w = 0
                self.hist['rewards'].append(self.seq_idx)
                self.hist['trial_ends'].append(self.seq_idx)
                self.hist['trial_outcomes'].append(1)
            elif self.t > self.aborttrial: # Fail trial condition, transition to inter-trial
                self.state = self.States[3]
                self.statenum = 3
                self.it = 0.
                self.hist['trial_ends'].append(self.seq_idx)
                self.hist['trial_outcomes'].append(0)                
            
            self.seq_since_reward += 1 
            if self.reward_mode in ('water_and_spout', 'spout_and_thirst',): # Reward for spout movement during trial
                self.reward = -self.dl / self.L0 * self.spout_movement_reward_scale * self.total_trial_reward
            else:
                self.reward = 0.
            self.reward_present = 0.

            oracle_signal = self.threshold_upper # used only for supervised-like-training hack of this enviroment
        
        elif self.state == self.States[2]: # reward
            self.dl = 0.
            self.w += self.dt

            self.tone = 0 # Turn off trial tone if needed

            # Reward is given during the reward period
            self.seq_since_reward = 0
            self.reward = self.reward_per_step
            self.reward_present = 1
            
            if self.w > self.t_reward: # Reward time finished
                if self.simple_states: # Simplified setup, simply go directly to inter-trial
                    self.state = self.States[3] 
                    self.statenum = 3
                    self.it = 0.
                else: # Standard setup that transitions to pre-trial
                    self.p = 0
                    self.state = self.States[0] 
                    self.statenum = 0
                    self.reward = 0
                    self.hist['pretrial_starts'].append(self.seq_idx)

        elif self.state == self.States[3]: # inter-trial
            self.dl = 0.
            self.it += self.dt # Iterate inter-trial time
            
            if self.it > self.it_time: # Transition to reset_spout
                self.state = self.States[4]
                self.statenum = 4

            self.seq_since_reward += 1
            self.reward = 0
            self.reward_present = 0

        elif self.state == self.States[4]: # reset_spout
            self.dl = self.dt * self.max_velocity_back # Move spout backwards at max velocity
            if self.l + self.dl  > self.L0: # Catches movement beyond limit
                self.dl = self.L0 - self.l
            self.l += self.dl
            
            if self.l >= self.L0: # Transition to trial
                self.state = self.States[1]
                self.statenum = 1
                self.l = self.L0 # Reset lickspout location
                self.t = 0 # Reset trial time
                self.hist['trial_starts'].append(self.seq_idx+1)

            self.seq_since_reward += 1
            self.reward = 0
            self.reward_present = 0

        elif self.state == self.States[-1]: # presession, similar state to pretrial, but will not transition for low activity
            
            if self.seq_idx == self.n_steps_stabilize - 1: # Presession is over, should analyze activtiy to set various parameters externally

                self.state = self.States[1] # Transition to trial
                self.statenum = 1
                self.t = 0 # Reset trial time
                self.hist['trial_starts'].append(self.seq_idx+1)

            self.seq_since_reward = 0
            self.reward = np.nan # Nan triggers running averages to remain constant
            self.reward_present = 0
        
        if self.reward_mode in ('thirst', 'spout_and_thirst',): # Negative reward for time since last drink
            self.reward += -1 * (self.seq_since_reward / (self.thirst_reward_timescale / self.dt)) * self.thirst_reward_at_timescale # When seq_since_reward = thirst_reward_timescale, equal to height

        self.seq_idx += 1

        # print(self.tone)
        # print(self.reward_present)
        # print(self.l)
        # print(self.dl)
        # print(self.l_dl_to_state(self.l, self.dl))

        # State which encodes the stimulus representations is 3 to 4-dimensional:
        # - 0: Tone signal to indicate start of trail (binary)
        # - 1: Reward signal to indicate spout is in licking position (binary)
        # - 2/3: Spout position (cosine/sine) / movement 
        state = np.array([self.tone, self.reward_present, *self.l_dl_to_state(self.l, self.dl)])

        if self.track_raw_states:
            self.hist['raw_states'].append(np.array([self.tone, self.reward_present, self.l]))

        full_state = self.build_full_state(state, noise)

        return full_state, nan_mask * self.reward, self.done, self.terminated, {'oracle_signal': oracle_signal,}
    
    def evaluate_threshold_change(self):
        """
        Determines if it is time for a threshold change and then determines how large of
        a threshold change will occur
        """
        
        if not self.dynamic_threshold:
            return None
        
        if self.threshold_change_type in ('seq_idxs',):
            
            change_condition_met = True # Default is True, overridden above in special scenarios
            if self.threshold_change_intertrial: # Trial must be in intertrial in order to evaluate change
                change_condition_met = True if self.state == self.States[3] else False
            
            if len(self.threshold_change_idxs) > 0: # Still some thresholds to change
                if self.seq_idx >= self.threshold_change_idxs[0] and change_condition_met: # Evaluate first threshold to change
                    # Just modify the upper threshold for now
                    new_threshold, new_threshold_upper = self.determine_threshold_change_mag()
                    self.change_threshold(new_threshold, new_threshold_upper)
                    self.threshold_change_idxs.pop(0) # Remove the threshold change index
                    return None
            # elif self.seq_idx in self.threshold_change_idxs: 
            #     # Just modify the upper threshold for now
            #     THRESHOLD_MULT = 2.0
            #     self.change_threshold(
            #         0.0, (THRESHOLD_MULT - 1.) * self.threshold_upper
            #     )
            #     return None
    
    def set_threshold(self, threshold, threshold_upper):
        """
        Modifies the thresholds which will change how the "input_to_velocity" 
        function behaves
        """
        
        self.threshold = threshold
        self.threshold_upper = threshold_upper
        
        if self.dynamic_threshold: # Stores these because they will change
            self.hist['threshold_changes'].append(np.copy(self.seq_idx))
            self.hist['threshold_vals'].append(
                (np.copy(self.threshold), np.copy(self.threshold_upper))
            )
            # When change occurs during the trial, gives current trial index (since len(trial_ends) = trial_idx b/c off by 1)
            # When change occurs after trial, gives next trial index
            self.hist['threshold_trials'].append(len(self.hist['trial_ends']))
        
    def change_threshold(self, new_threshold, new_threshold_upper):
        """
        Small wrapper around set_threshold function to change thresholds. 
        """
        print('Threshold change at idx {} - lower: {:.1e} to {:.1e}, upper: : {:.1e} to {:.1e}'.format(
            self.seq_idx, self.threshold, new_threshold, self.threshold_upper, new_threshold_upper,
        ))
        self.set_threshold(new_threshold, new_threshold_upper)
        
    def determine_threshold_change_mag(self):
        """
        OUTPUTS:
        - new_threshold
        - new_threshold_upper
        """
        
        if self.threshold_change_mag_type in ('fixed',): # Fixed gain_new / gain_old value by changing upper threshold
            gain_old =  self.threshold_upper - self.threshold
            new_threshold_upper = self.threshold_change_mag_params[0] * gain_old + self.threshold
            return np.copy(self.threshold), new_threshold_upper
        else:
            raise ValueError('Threhsold change magnitude type {} not recognized!'.format(self.threshold_change_mag_type))
        
        
class Photostim_Env(Noise_Base):
    """
    Photostimulation environment for stimulating the network and measuring responses. Note this is
    currently setup to not train the network and thus does not provide any reward signal. 

    bci_env is passed at initialization and many features are copied into this environment to
    ensure things like the same stimulus representations.
    """
    
    def __init__(self, task_params, bci_env):

        super().__init__(task_params) # Noise initialization

        self.dt = bci_env.dt # sec
        self.L0 = bci_env.L0
        self.n_inp = bci_env.n_inp
        self.n_neurons = bci_env.n_neurons
        self.state_mode = bci_env.state_mode 

        # These are state_mode dependent, but important for generating similar input activity in photostim setting
        self.build_full_state = bci_env.build_full_state # Function to determine state
        self.state_dim = bci_env.state_dim
        self.n_stim = bci_env.n_stim
        self.l_dl_to_state = bci_env.l_dl_to_state
        self.stim_input = bci_env.stim_input
   
        self.session_idx = 0 # Keep track of multiple sessions at once within the same environment

        self.States = ['photostim']

        # # No internal seed setting to avoid the same noise input into multiple photostim sessions
        # np.random.seed(task_params.get('seed', 0))

        self.seq_idx = 0 # Sequence index within photostim session

        # Photostim state attr
        self.t_ps = None                            # keeps track of time within a single stimulus flash occurence
        self.ps_idx = None                          # keeps track of current photostim index
        self.t_max_stim_on = 0.100 #0.15            # photostimulus time (sec), experiment uses 100 ms
        self.t_between_stim  = 0.600 # 1.0          # between stimulus time (sec), experiment uses 600 ms
        self.n_groups = task_params.get('n_groups', 100)
        self.n_neurons_per_group = task_params.get('n_neurons_per_group', 10)

        if np.abs(np.round(self.t_between_stim / self.dt) - self.t_between_stim / self.dt) > 1e-2:
            raise ValueError('Time between stims and dt are not integer multiples of one another, this will cause problems!')
        if np.abs(np.round(self.t_max_stim_on / self.dt) - self.t_max_stim_on / self.dt) > 1e-2:
            raise ValueError('Photostim on time and dt are not integer multiples of one another, this will cause problems!')

        # Two ways of determining length of photostimulation session:
        # 1. Given n_repeats_per_group and n_groups, determine length
        # 2. Given length and n_groups, determine n_repeats_per_group
        if 'n_repeats_per_group' in task_params:
            self.n_repeats_per_group = task_params.get('n_repeats_per_group')
            self.n_steps_photostim = self.n_groups * self.n_repeats_per_group * int(np.round(self.t_between_stim / self.dt))
            self.t_photostim = self.n_steps_photostim * self.dt # in seconds
        else:
            assert 'n_steps_photostim' in task_params
            self.n_steps_photostim = task_params.get('n_steps_photostim')
            self.t_photostim = self.n_steps_photostim * self.dt # in seconds
            self.n_repeats_per_group = int(np.ceil(self.t_photostim / (self.n_groups * self.t_between_stim)))

        # These parameters determine how each neuron is excited by the photostim
        # self.neuron_fidelities is the amount of (pre-activity) value that gets added to the network's activity
        # Note this is outside of the reset because this should be a fixed property of the neurons across sessions
        self.neuron_fidelity_mode = task_params.get('neuron_fidelity_mode', 'ones')
        self.max_fidelty = task_params.get('max_fidelty', 0.8)
        if self.neuron_fidelity_mode in ('copy',): # Copy previous fidelities
            self.neuron_fidelities = task_params.get('neuron_fidelities')
        elif self.neuron_fidelity_mode in ('ones',):
            self.neuron_fidelities = self.max_fidelty * np.ones((self.n_neurons,))
        elif self.neuron_fidelity_mode in ('uniform_random',):
            self.neuron_fidelities = self.max_fidelty * np.random.uniform(0, 1, size=(self.n_neurons,))
        else:
            raise ValueError('Neuron fidelity mode {} not recognized.'.format(self.neuron_fidelity_mode))

        # Creates groups of neurons, this is outside the reset because should be fixed across the sessions
        self.groups = np.zeros((self.n_groups, self.n_neurons_per_group), dtype=np.int32)
        for group_idx in range(self.n_groups):
            self.groups[group_idx] = np.random.choice(self.n_neurons, size=(self.n_neurons_per_group,), replace=False)

        # Init history trackers
        self.hists = [self.init_new_hist(),]
        self.hist = self.hists[self.session_idx] # Sets current hist

        # These currently just stay false at all times
        self.reward = np.nan       # reward signal (used to adjust gradients)
        self.done = False 
        self.terminated = False

    # def copy_prev_task_ps(self, task_ps):
    #     """
    #     Overrides a few internal parameters with a previous photostim task to mimic the experiment,
    #     where things like neuron groups and photostim order are held constant across photostim 
    #     days.
    #     """
    #     print('Copying parameters from previous photostim task...')

    #     self.neuron_fidelities = task_ps.neuron_fidelities
    #     self.groups = task_ps.groups
    #     self.group_order = task_ps.group_order

    def init_new_hist(self):
        
        session_hist = {}

        session_hist['stim_starts'] = [] # seq_idxs of start of stimulation
        session_hist['stim_ends'] = [] # seq_idxs of end of stimulation
        session_hist['stim_group_idxs'] = [] # group idx of stimuli in current session

        return session_hist

    def set_new_session(self):
        """ Iterates to a new session. Note reset needs to be called separately after this """

        # Checks to make sure the previous session actually ran
        assert len(self.hist['stim_starts']) > 0

        self.session_idx += 1
        self.hists.append(self.init_new_hist())

        self.hist = self.hists[self.session_idx] # Update current hist

    def reset(self):

        noise, _ = super().reset() 
        # noise *= 0.
        
        self.state = self.States[0] # Start in photostim session

        # These are the same state quantities used in the BCI environment, but they are always fixed.
        self.tone = 0. 
        self.reward_present = 0.
        self.l = self.L0
        self.dl = 0

        self.statenum = 0
        self.t_ps = 0.
        self.ps_idx = 0
        
        self.seq_idx = 0
        self.reward = np.nan # Nan triggers running averages to remain constant

        # Create a new group order if one does not exist already, but do not override
        # if one does exist (from copying another photostim's group order)
        # Note that even if we do not use the new group order, we create it to keep random seeds in sync
        candidate_group_order = np.array([[group_idx for _ in range(self.n_repeats_per_group) for group_idx in range(self.n_groups)]]).reshape(-1)
        np.random.shuffle(candidate_group_order)
        if not hasattr(self, 'group_order'): 
            self.group_order = candidate_group_order

        # # Init history tracker
        # self.hist = self.init_new_hist() # Creates a new history
        # self.hist['stim_starts'].append(self.seq_idx) # First stimulus starts right away
        # self.hist['stim_group_idxs'].append(self.group_order[self.ps_idx])

        photostim_input = np.zeros((self.n_neurons,))

        state = np.array([self.tone, self.reward_present, *self.l_dl_to_state(self.l, self.dl)])
        full_state = self.build_full_state(state, noise)

        return (full_state, photostim_input), {}
    
    def step(self, state):
        """
        'state' not used, but needs to follow usual step(state) format.

        Photostim_input kept separate from the rest of the state because it can be directly injected
        into hidden activity.
        
        OUTPUTS:
        - full_state: stimulus representation of what is being seen by network during photostim, passed through input layer
        - photostim_input: noise directly injected into (pre-activation) hidden activity to represent stimulation
        
        """

        noise, _, _, _, _ = super().step(None)  
        # noise *= 0.
        
        photostim_input = np.zeros((self.n_neurons,))
        if self.state == self.States[0]: # Photostim
            if self.t_ps < self.t_max_stim_on: # Stimulus is showing, add group's fidelities to noise
                current_group_idxs = self.groups[self.group_order[self.ps_idx]] 
                photostim_input[current_group_idxs] += self.neuron_fidelities[current_group_idxs]

                # print('Ps_idx {}, Group idx: {}, Stim_group_idxs entry: {}'.format(
                #     self.ps_idx, self.group_order[self.ps_idx], self.hist['stim_group_idxs'][-1]
                # ))
                # print('  Group idxs:', current_group_idxs)
                # noise[current_group] += self.neuron_fidelities[current_group]

                if self.t_ps + self.dt >= self.t_max_stim_on: # Last time step of stimulus
                    self.hist['stim_ends'].append(self.seq_idx + 1)

            self.t_ps += self.dt # Iterate stimulus showing time

            if self.t_ps >= self.t_between_stim: # Transition to the next stimulus
                self.t_ps = 0.
                self.ps_idx += 1 # Move to next stimulus set

                # Skip this on the very last sequence index because it doesn't matter and causes indexing errors
                if self.seq_idx < self.n_steps_photostim - 1:
                    self.hist['stim_starts'].append(self.seq_idx + 1)
                    self.hist['stim_group_idxs'].append(self.group_order[self.ps_idx])

        else:
            raise ValueError('This shouldnt happen')

        self.seq_idx += 1

        state = np.array([self.tone, self.reward_present, *self.l_dl_to_state(self.l, self.dl)]) 
        full_state = self.build_full_state(state, noise)

        return (full_state, photostim_input), self.reward, self.done, self.terminated, {}