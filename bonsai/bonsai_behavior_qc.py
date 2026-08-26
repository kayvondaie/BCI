"""
Standalone Bonsai behavior QC.

Reads only the Bonsai SoftwareEvents JSON files plus OperationControl/SpoutPosition.csv
(no ScanImage, no harp triggers, no suite2p, no alignment). Per trial:
  - hit / miss
  - time to reward  : GiveReward time, measured from ResponsePeriod onset (the
                      "trial start" once quiescence is satisfied; capped at
                      response_period.duration, typically 10 s)
  - 1st action time : PerformedAction == True time from ResponsePeriod onset
  - threshold xing  : time the spout first reaches its max position, from
                      OperationControl/SpoutPosition.csv (lags 1st action by
                      spout actuator dynamics)

Usage in a kernel that already has `folder` defined (the pophys folder ddc loads from):
    exec(open(r'.../bonsai_behavior_qc.py').read())

Usage from CLI:
    python bonsai_behavior_qc.py <path-to-pophys-or-behavior-dir>
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_events(events_dir, name, optional=False):
    path = os.path.join(events_dir, f'{name}.json')
    if not os.path.isfile(path):
        if optional:
            return []
        raise FileNotFoundError(path)
    out = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _scalar(v):
    """Unwrap Bonsai distribution-parameter dicts to a plain number."""
    if isinstance(v, dict):
        return float(v['distribution_parameters']['value'])
    return float(v)


def resolve_behavior_dir(path):
    """Accept either the pophys folder (ddc's `folder`) or the behavior folder.
    Returns the behavior folder containing SoftwareEvents/.
    """
    path = path.rstrip('/\\')
    if os.path.isdir(os.path.join(path, 'SoftwareEvents')):
        return path
    # ddc convention: behavior is a sibling of pophys
    sibling = os.path.join(os.path.dirname(path), 'behavior')
    if os.path.isdir(os.path.join(sibling, 'SoftwareEvents')):
        return sibling
    raise FileNotFoundError(
        f'Could not find SoftwareEvents/ under {path!r} or its sibling behavior/'
    )


def qc(path):
    behavior_dir = resolve_behavior_dir(path)
    events_dir = os.path.join(behavior_dir, 'SoftwareEvents')
    op_dir = os.path.join(behavior_dir, 'OperationControl')

    trials = _load_events(events_dir, 'Trial')
    responses = _load_events(events_dir, 'ResponsePeriod')
    rewards = _load_events(events_dir, 'GiveReward', optional=True)
    actions = _load_events(events_dir, 'PerformedAction', optional=True)
    outcomes = _load_events(events_dir, 'IsValidRewardOutcome', optional=True)

    trials.sort(key=lambda e: e['timestamp'])
    responses.sort(key=lambda e: e['timestamp'])
    rewards.sort(key=lambda e: e['timestamp'])
    actions.sort(key=lambda e: e['timestamp'])

    trial_ts = np.array([e['timestamp'] for e in trials])
    response_ts = np.array([e['timestamp'] for e in responses])
    reward_ts = np.array([e['timestamp'] for e in rewards])
    action_ts = np.array([e['timestamp'] for e in actions if e['data'] is True])

    rp_dur = np.array([_scalar(e['data']['response_period']['duration']) for e in trials])
    quiescence_dur = np.array([_scalar(e['data']['quiescence_period']['duration']) for e in trials])

    # Spout position (for threshold-crossing detection on the actuator side)
    sp_path = os.path.join(op_dir, 'SpoutPosition.csv')
    has_spout = os.path.isfile(sp_path)
    if has_spout:
        sp = pd.read_csv(sp_path, skipinitialspace=True)
        sp.columns = sp.columns.str.strip()
        t_sp = sp['Seconds'].to_numpy()
        pos_sp = sp['Value'].to_numpy()
        p_max = float(pos_sp.max())
    else:
        t_sp = pos_sp = np.array([])
        p_max = np.nan

    n = len(trial_ts)
    print(f'behavior_dir: {behavior_dir}')
    print(f'  n_trials       = {n}')
    print(f'  n_responses    = {len(response_ts)}')
    print(f'  n_rewards      = {len(reward_ts)}')
    print(f'  n_actions      = {len(action_ts)}  (PerformedAction == True)')
    print(f'  n_outcomes     = {len(outcomes)}')
    print(f'  spout samples  = {len(t_sp)}  (P_max = {p_max if has_spout else float("nan"):.3f})')
    print(f'  response_period.duration: min={rp_dur.min():.2f} max={rp_dur.max():.2f} mean={rp_dur.mean():.2f}')
    print(f'  quiescence.duration:      min={quiescence_dur.min():.2f} max={quiescence_dur.max():.2f}')
    print()

    # All latencies are measured from ResponsePeriod onset (= "trial start" after
    # quiescence is achieved, which is the user-facing trial start).
    time_to_reward = np.full(n, np.nan)
    time_to_action = np.full(n, np.nan)
    time_to_thr_xing = np.full(n, np.nan)
    hit = np.zeros(n, dtype=bool)
    rp_for_trial = np.full(n, np.nan)

    for i in range(n):
        t_trial = trial_ts[i]
        t_next_trial = trial_ts[i + 1] if i + 1 < n else np.inf

        # Response-period onset for this trial: the first ResponsePeriod in [t_trial, t_next_trial)
        rp_mask = (response_ts >= t_trial) & (response_ts < t_next_trial)
        if not np.any(rp_mask):
            continue
        rp0 = response_ts[rp_mask][0]
        rp_for_trial[i] = rp0
        cap = rp0 + rp_dur[i]  # ignore anything past the response-period cap

        # Time to reward
        r_mask = (reward_ts >= rp0) & (reward_ts < min(cap, t_next_trial))
        if np.any(r_mask):
            time_to_reward[i] = reward_ts[r_mask][0] - rp0
            hit[i] = True

        # Time to 1st action
        a_mask = (action_ts >= rp0) & (action_ts < min(cap, t_next_trial))
        if np.any(a_mask):
            time_to_action[i] = action_ts[a_mask][0] - rp0

        # Time to spout threshold crossing
        if has_spout:
            sp_mask = (t_sp >= rp0) & (t_sp < min(cap, t_next_trial))
            seg_pos = pos_sp[sp_mask]
            seg_t = t_sp[sp_mask]
            hit_idx = np.where(seg_pos >= p_max - 0.1)[0]
            if len(hit_idx):
                time_to_thr_xing[i] = seg_t[hit_idx[0]] - rp0

    print('--- per-trial summary (times in seconds, from ResponsePeriod onset) ---')
    print(f'{"trl":>4} {"hit":>4} {"reward":>8} {"action":>8} {"thr_xing":>9} {"rp_dur":>7}')
    for i in range(n):
        def fmt(x, w):
            return f'{x:>{w}.3f}' if not np.isnan(x) else f'{"--":>{w}}'
        print(f'{i:>4} {("Y" if hit[i] else "N"):>4} {fmt(time_to_reward[i], 8)} {fmt(time_to_action[i], 8)} {fmt(time_to_thr_xing[i], 9)} {rp_dur[i]:>7.2f}')

    print()
    print('--- aggregate ---')
    print(f'  hit rate     : {hit.mean():.3f}  ({hit.sum()}/{n})')
    for name, arr in (('reward  ', time_to_reward),
                      ('action  ', time_to_action),
                      ('thr_xing', time_to_thr_xing)):
        if np.any(~np.isnan(arr)):
            print(f'  {name}   : min={np.nanmin(arr):.3f}  max={np.nanmax(arr):.3f}  median={np.nanmedian(arr):.3f}')

    print()
    print('--- invariant checks (cap = response_period.duration + 50 ms slack) ---')
    cap_per = rp_dur + 0.05
    for name, arr in (('reward', time_to_reward),
                      ('action', time_to_action),
                      ('thr_xing', time_to_thr_xing)):
        over = np.where(arr > cap_per)[0]
        if len(over):
            print(f'  FAIL: {name} exceeds cap on trials: {over.tolist()}')
            for i in over:
                print(f'    trial {i}: {name}={arr[i]:.3f}s  cap={cap_per[i]:.3f}s')
        else:
            n_valid = int(np.sum(~np.isnan(arr)))
            print(f'  OK: {name} within cap on all {n_valid} trials with a {name} event')

    _plot(behavior_dir, hit, time_to_reward, time_to_action, time_to_thr_xing, rp_dur)

    return {
        'trial_ts': trial_ts,
        'rp_ts': rp_for_trial,
        'hit': hit,
        'time_to_reward': time_to_reward,
        'time_to_action': time_to_action,
        'time_to_thr_xing': time_to_thr_xing,
        'rp_dur': rp_dur,
    }


def _plot(behavior_dir, hit, time_to_reward, time_to_action, time_to_thr_xing, rp_dur):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    n = len(hit)
    trials = np.arange(n)
    cap = float(np.nanmedian(rp_dur))

    ax = axes[0, 0]
    ax.plot(trials, time_to_reward, 'o-', color='tab:blue', label='time to reward', markersize=3)
    ax.plot(trials, time_to_action, 'x', color='tab:orange', label='1st action', markersize=4)
    ax.plot(trials, time_to_thr_xing, '.', color='tab:green', label='spout thr crossing', markersize=4)
    ax.axhline(cap, color='k', linestyle='--', linewidth=0.8, label=f'response_period = {cap:.1f}s')
    ax.set_xlabel('Trial #'); ax.set_ylabel('Time from trial start (s)')
    ax.set_title('Per-trial latencies')
    ax.legend(loc='best', frameon=False, fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    window = 10 if n >= 10 else max(1, n // 2)
    rolling = np.convolve(hit.astype(float), np.ones(window) / window, mode='valid')
    ax.plot(np.arange(len(rolling)) + window - 1, rolling, 'k')
    ax.scatter(trials, hit.astype(float), marker='|', color='gray', s=40)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Trial #'); ax.set_ylabel('Hit rate')
    ax.set_title(f'Hit rate (rolling {window}-trial)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Shared histogram bins so the three latency dists are directly comparable.
    all_vals = np.concatenate([
        time_to_reward[~np.isnan(time_to_reward)],
        time_to_action[~np.isnan(time_to_action)],
        time_to_thr_xing[~np.isnan(time_to_thr_xing)],
    ])
    upper = max(cap + 1.0, float(all_vals.max()) + 0.5) if len(all_vals) else cap + 1.0
    bins = np.linspace(0, upper, 40)

    ax = axes[1, 0]
    for arr, color, label in ((time_to_reward, 'tab:blue', 'reward'),
                              (time_to_action, 'tab:orange', '1st action'),
                              (time_to_thr_xing, 'tab:green', 'thr xing')):
        vals = arr[~np.isnan(arr)]
        if len(vals):
            ax.hist(vals, bins=bins, color=color, alpha=0.5, label=label)
    ax.axvline(cap, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time from trial start (s)'); ax.set_ylabel('Count')
    ax.set_title('Latency distributions')
    ax.legend(loc='best', frameon=False, fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    ax.axis('off')
    session = os.path.basename(os.path.dirname(behavior_dir.rstrip(os.sep)))
    lines = [
        f'session: {session}',
        '',
        f'trials       : {n}',
        f'hits         : {int(hit.sum())}  ({hit.mean():.1%})',
        f'rp duration  : {cap:.2f} s',
        '',
        'medians (s):',
        f'  reward     : {_fmt_stat(time_to_reward, np.nanmedian)}',
        f'  action     : {_fmt_stat(time_to_action, np.nanmedian)}',
        f'  thr xing   : {_fmt_stat(time_to_thr_xing, np.nanmedian)}',
        '',
        'max (s):',
        f'  reward     : {_fmt_stat(time_to_reward, np.nanmax)}',
        f'  action     : {_fmt_stat(time_to_action, np.nanmax)}',
        f'  thr xing   : {_fmt_stat(time_to_thr_xing, np.nanmax)}',
        '',
        f'over cap rew : {int(np.nansum(time_to_reward > cap + 0.05))}',
        f'over cap act : {int(np.nansum(time_to_action > cap + 0.05))}',
        f'over cap xng : {int(np.nansum(time_to_thr_xing > cap + 0.05))}',
    ]
    ax.text(0.0, 1.0, '\n'.join(lines), va='top', ha='left', family='monospace', fontsize=8)

    fig.tight_layout()
    plt.show()


def _fmt_stat(arr, fn):
    valid = arr[~np.isnan(arr)]
    return f'{fn(valid):.2f}' if len(valid) else '(none)'


# --- entry point ---
# Picks `folder` straight out of the calling namespace (same convention as
# bonsai_npy_threshold_calculator.py). Falls back to CLI arg, then to the
# example session in this repo.
try:
    folder  # type: ignore[name-defined]
    _path = folder
except NameError:
    if len(sys.argv) >= 2:
        _path = sys.argv[1]
    else:
        _path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '123456_2026-04-21T222754Z', 'behavior')
        print(f'no `folder` in scope and no CLI arg; defaulting to {_path}')

qc(_path)
