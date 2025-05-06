# bci_time_series_module.py

import numpy as np

def bci_time_series_fun(folder, data, rt, dt_si):
    ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
    frames_per_file = ops['frames_per_file']
    df = data['df_closedloop']

    total_frames = sum(frames_per_file)
    trial_starts = np.cumsum([0] + list(frames_per_file[:-1]))

    # Trial start vector
    trial_start_vector = np.zeros(total_frames, dtype=int)
    trial_start_vector[trial_starts] = 1

    # Reward vector
    reward_vector = np.zeros(total_frames, dtype=int)
    for start, dur, rtime in zip(trial_starts, frames_per_file, rt):
        if np.isnan(rtime):
            continue
        reward_frame = int(round(rtime / dt_si))
        reward_idx = start + reward_frame
        if reward_idx < start + dur:
            reward_vector[reward_idx] = 1

    # Step vector
    step_vector = np.zeros(total_frames, dtype=int)
    for start, dur, steps in zip(trial_starts, frames_per_file, data['step_time']):
        if steps is None or len(steps) == 0:
            continue
        step_frames = np.round(np.array(steps) / dt_si).astype(int)
        for f in step_frames:
            if 0 <= f < dur:
                step_vector[start + f] = 1

    return step_vector, reward_vector, trial_start_vector


def compute_ema(signal, tau, start_val=None):
    alpha = 1 / tau
    ema = np.zeros_like(signal, dtype=float)

    # Use provided start_val or default to global mean
    if start_val is None:
        ema[0] = np.mean(signal)
    else:
        ema[0] = start_val

    for t in range(1, len(signal)):
        ema[t] = alpha * signal[t] + (1 - alpha) * ema[t - 1]
    
    return ema