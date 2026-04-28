"""
Loop through multiple BCI116 sessions, compute far-cell dF/F for each,
and plot the average across sessions.
"""
import sys
sys.path.insert(0, r'C:\Users\kayvon.daie\Documents\claude_code\bonsai')

import numpy as np
import os
import matplotlib.pyplot as plt
import data_dict_create_module_bruker as ddc

mouse = 'BCI116'
sessions = [
    '010526',  # Jan 5
    '010626',  # Jan 6
    '010826',  # Jan 8
    '012026',  # Jan 20
    '012726',  # Jan 27
    '012826',  # Jan 28
    '013026',  # Jan 30
]

root = f'//allen/aind/scratch/BCI/2p-raw/{mouse}'

per_session_far_avg = []
per_session_t = []

for sess in sessions:
    folder = f'{root}/{sess}/pophys/'
    ps_dir = folder + r'/suite2p_photostim_single/plane0/'
    if not os.path.isfile(ps_dir + 'iscell.npy'):
        print(f'[{sess}] skip — no suite2p_photostim_single folder')
        continue
    print(f'[{sess}] loading...')

    iscell = np.load(ps_dir + 'iscell.npy', allow_pickle=True)
    stat = np.load(ps_dir + 'stat.npy', allow_pickle=True)
    Ftrace = np.load(ps_dir + 'F.npy', allow_pickle=True)
    ops = np.load(ps_dir + 'ops.npy', allow_pickle=True).tolist()
    siHeader = np.load(ps_dir + 'siHeader.npy', allow_pickle=True).tolist()

    Fstim, seq, favg, stimDist, stimPosition, centroidX, centroidY, slmDist, stimID, Fstim_raw, favg_raw, stim_params = \
        ddc.stimDist_single_cell(ops, Ftrace, siHeader, stat, 0)

    dt_si = 1 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
    t = np.arange(favg.shape[0]) * dt_si

    F0 = np.nanmean(np.nanmean(Fstim_raw[:8, :, :], axis=0), axis=-1)
    favg_raw_bs = favg_raw - np.nanmean(favg_raw[:8, :, :], axis=0, keepdims=True)
    favg_dff = favg_raw_bs / F0[None, :, None]

    dim_thresh = np.nanpercentile(F0, 10)
    dim_mask = F0 <= dim_thresh
    not_cell_mask = iscell[:, 0] == 0

    far_avg = np.full((favg.shape[0], favg.shape[2]), np.nan)
    for gi in range(favg.shape[2]):
        far_cells = np.where((stimDist[:, gi] > 100) & dim_mask & not_cell_mask)[0]
        if len(far_cells) > 0:
            far_avg[:, gi] = np.nanmean(favg_dff[:, far_cells, gi], axis=1)

    # Average across target groups for this session
    per_session_far_avg.append(np.nanmean(far_avg, axis=1))
    per_session_t.append(t)
    print(f'[{sess}] done — {favg.shape[2]} target groups')

# Align all sessions to a common time base (truncate to min length)
min_len = min(60, min(len(x) for x in per_session_far_avg))
all_traces = np.array([x[:min_len] for x in per_session_far_avg])
t_common = per_session_t[0][:min_len]

plt.figure(figsize=(6, 4))
for i, sess in enumerate([s for s in sessions if len(per_session_far_avg) > 0][:len(all_traces)]):
    plt.plot(t_common, all_traces[i], color='gray', alpha=0.5, linewidth=0.8)
plt.plot(t_common, np.nanmean(all_traces, axis=0), 'k', linewidth=2, label=f'mean (n={len(all_traces)})')
plt.xlabel('Time (s)')
plt.ylabel('Mean dF/F (far, dim, iscell=0)')
plt.title(f'{mouse} — {len(all_traces)} sessions')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.show()
