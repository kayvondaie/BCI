"""
Debug script for aligning Bonsai trials to ScanImage files.
Edit folder and bonsai_behav paths, then run.
"""
import sys
sys.path.insert(0, r'C:\Users\kayvon.daie\Documents\claude_code\bonsai')

import json
import numpy as np
import os

folder = '//allen/aind/scratch/BCI/2p-raw/BCI124/042326/pophys/'
parent = os.path.dirname(folder.rstrip('/\\'))
bonsai_behav = os.path.join(parent, 'behavior')

# --- Load SI info from suite2p_BCI ---
ops = np.load(os.path.join(folder, 'suite2p_BCI', 'plane0', 'ops.npy'), allow_pickle=True).tolist()
siHeader = np.load(os.path.join(folder, 'suite2p_BCI', 'plane0', 'siHeader.npy'), allow_pickle=True).tolist()
dt_si = 1 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])

si_durations = np.array(ops['frames_per_file']) * dt_si
n_si = len(si_durations)

print(f'SI files: {n_si}')
print(f'SI dt_si: {dt_si:.4f} s')
print(f'SI durations: min={si_durations.min():.2f} max={si_durations.max():.2f} mean={si_durations.mean():.2f}')
print(f'SI first 10: {np.round(si_durations[:10], 2)}')
print(f'SI total time: {si_durations.sum():.1f} s')
print()

# --- Load Bonsai trial timestamps ---
trial_timestamps = []
with open(os.path.join(bonsai_behav, 'SoftwareEvents', 'Trial.json'), 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        trial_timestamps.append(json.loads(line)['timestamp'])
trial_timestamps = np.array(sorted(trial_timestamps))
n_bonsai = len(trial_timestamps)

bonsai_intervals = np.diff(trial_timestamps)

print(f'Bonsai trials: {n_bonsai}')
print(f'Bonsai intervals: min={bonsai_intervals.min():.2f} max={bonsai_intervals.max():.2f} mean={bonsai_intervals.mean():.2f}')
print(f'Bonsai first 10: {np.round(bonsai_intervals[:10], 2)}')
print(f'Bonsai total time: {trial_timestamps[-1] - trial_timestamps[0]:.1f} s')
print()

# --- Compare ---
print(f'SI total imaging time: {si_durations.sum():.1f} s')
print(f'Bonsai session duration: {trial_timestamps[-1] - trial_timestamps[0]:.1f} s')
print()

# --- Cross-correlation on intervals ---
# SI: intervals between file starts = durations of all but last file
# (si_durations[i] = time from file i start to file i+1 start)
# But SI files may not be 1:1 with trials. Let's check.

print('--- Diagnosis ---')
print(f'Are SI files 1:1 with trials? SI has {n_si} files, Bonsai has {n_bonsai} trials')
print(f'SI duration std: {si_durations.std():.4f} (if ~0, SI is fixed-length continuous recording)')
print()

if si_durations.std() < 0.1:
    print('SI files appear to be FIXED LENGTH (continuous acquisition, not trial-triggered)')
    print('Cross-correlation on durations will not work.')
    print()
    print('Alternative: use cumulative frame count to convert Bonsai harp timestamps to SI frames')
    total_frames = sum(ops['frames_per_file'])
    print(f'Total SI frames: {total_frames}')
    print(f'Total SI time: {total_frames * dt_si:.1f} s')
else:
    print('SI files have VARIABLE length — likely one file per trial trigger')
    print()
    # Greedy matching: walk through SI files and Bonsai intervals,
    # skipping Bonsai trials that don't match (gaps/pauses where SI wasn't recording)
    tol = 0.5  # seconds tolerance for matching durations

    # First find the starting offset: which Bonsai trial matches SI file 0?
    best_start = 0
    best_start_err = np.inf
    for start in range(n_bonsai - 1):
        err = abs(si_durations[0] - bonsai_intervals[start])
        if err < best_start_err:
            best_start_err = err
            best_start = start

    print(f'First match: SI file 0 -> Bonsai trial {best_start} (err={best_start_err:.3f} s)')

    # Now greedily match from there
    si_to_bonsai = {}  # SI file index -> Bonsai trial index
    bi = best_start
    for si_idx in range(n_si):
        si_d = si_durations[si_idx]
        # Search forward in Bonsai for a matching interval
        found = False
        for search in range(bi, min(bi + 10, n_bonsai - 1)):  # look up to 10 trials ahead
            bn_d = bonsai_intervals[search]
            if abs(si_d - bn_d) < tol:
                si_to_bonsai[si_idx] = search
                bi = search + 1
                found = True
                break
        if not found:
            print(f'  WARNING: no match for SI file {si_idx} (dur={si_d:.2f}) searching Bonsai trials {bi}-{min(bi+10, n_bonsai-1)}')

    print(f'\nMatched {len(si_to_bonsai)} / {n_si} SI files to Bonsai trials')

    # Show matches
    print(f'\n{"SI file":>8} {"Bonsai trial":>12} {"SI dur":>8} {"Bonsai int":>10} {"diff":>8}')
    for si_idx in range(min(30, n_si)):
        si_d = si_durations[si_idx]
        if si_idx in si_to_bonsai:
            bi = si_to_bonsai[si_idx]
            bn_d = bonsai_intervals[bi]
            print(f'{si_idx:>8} {bi:>12} {si_d:>8.2f} {bn_d:>10.2f} {si_d-bn_d:>8.2f}')
        else:
            print(f'{si_idx:>8} {"---":>12} {si_d:>8.2f}       ---      ---')
    if n_si > 30:
        print(f'  ... ({n_si - 30} more)')

    # Show any gaps (where Bonsai trials were skipped)
    matched_bonsai = sorted(si_to_bonsai.values())
    skipped = [i for i in range(matched_bonsai[0], matched_bonsai[-1]+1) if i not in matched_bonsai]
    if skipped:
        print(f'\nSkipped Bonsai trials (no SI file): {skipped}')
        for s in skipped:
            print(f'  trial {s}: interval={bonsai_intervals[s]:.2f} s')
