"""
Diagnostic: confirm whether create_bonsai_info actually ran with the cap fix,
and show where the >20s windows in `data` are coming from.

Run AFTER ddc.main(folder). Picks up `data` and `folder` from the kernel.
"""
import os
import json
import struct
import numpy as np


def diag(folder, data):
    # 1) confirm the loaded module actually has the cap line ---
    import data_dict_create_module_bruker as ddc
    src = open(ddc.__file__).read()
    has_cap = 'all_bonsai_trial_ts' in src and 't_next_bonsai' in src
    print(f'ddc.__file__              = {ddc.__file__}')
    print(f'fix present in source?    = {has_cap}')
    print()

    # 2) what windows did the cached data actually use? ---
    sit = np.array([x[0] if len(x) > 0 else np.nan for x in data['SI_start_times']])
    tct = np.array([x[0] if len(x) > 0 else np.nan for x in data['threshold_crossing_time']])
    rt = tct - sit
    gaps = np.diff(sit)
    print(f'#trials in data           = {len(sit)}')
    print(f'max gap between SI_start  = {np.nanmax(gaps):.2f} s')
    print(f'median gap                = {np.nanmedian(gaps):.2f} s')
    print(f'max rt                    = {np.nanmax(rt):.2f} s')
    print(f'#trials with rt > 20      = {int(np.nansum(rt > 20))}')
    print()

    # 3) load the raw Bonsai sources and recompute what t_next *should* have been
    parent = os.path.dirname(folder.rstrip('/\\'))
    bonsai_behav = os.path.join(parent, 'behavior')
    events_dir = os.path.join(bonsai_behav, 'SoftwareEvents')

    trial_ts = []
    with open(os.path.join(events_dir, 'Trial.json')) as f:
        for line in f:
            line = line.strip()
            if line:
                trial_ts.append(json.loads(line)['timestamp'])
    trial_ts = np.array(sorted(trial_ts))

    harp_bin = os.path.join(bonsai_behav, 'Behavior.harp', 'Behavior_34.bin')
    sit_harp = []
    with open(harp_bin, 'rb') as f:
        b = f.read()
    o = 0
    while o < len(b):
        msg_len = b[o + 1]
        total = 2 + msg_len
        ts = struct.unpack_from('<I', b, o + 5)[0] + struct.unpack_from('<H', b, o + 9)[0] * 32e-6
        payload = b[o + 11:o + 13]
        if payload == b'\x01\x00':
            sit_harp.append(ts)
        o += total
    sit_harp = np.array(sorted(sit_harp))

    print(f'n bonsai trials (Trial.json)        = {len(trial_ts)}')
    print(f'n harp SI triggers (Behavior_34)    = {len(sit_harp)}')
    print(f'n SI files (data["SI_start_times"]) = {len(sit)}')
    print()

    # 4) show the worst offenders
    bad = np.argsort(rt)[::-1][:10]
    print('top 10 trials by rt:')
    print(f'{"i":>4} {"rt":>8} {"sit[i]":>14} {"sit[i+1]":>14} {"gap":>8} {"#bonsai in win":>14}')
    for i in bad:
        if np.isnan(rt[i]):
            continue
        s0 = sit[i]
        s1 = sit[i + 1] if i + 1 < len(sit) else np.nan
        gap = (s1 - s0) if not np.isnan(s1) else np.nan
        n_bonsai_in_win = int(np.sum((trial_ts > s0 + 0.5) & (trial_ts < s1))) if not np.isnan(s1) else -1
        print(f'{i:>4} {rt[i]:>8.2f} {s0:>14.2f} {s1 if not np.isnan(s1) else float("nan"):>14.2f} {gap if not np.isnan(gap) else float("nan"):>8.2f} {n_bonsai_in_win:>14}')


if __name__ == '__main__' or True:
    try:
        diag(folder, data)  # type: ignore[name-defined]
    except NameError:
        print('define `folder` and `data` in the kernel first, then re-run')
