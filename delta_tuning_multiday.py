import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import data_dict_create_module_test as ddc  # your module

# --- Settings ---
base_dir = '//allen/aind/scratch/BCI/2p-raw'
mice = ["BCI106"]
#mice = ["BCI88", "BCI93", "BCI102", "BCI103", "BCI104", "BCI105","BCI106","BCI107","BCI109"]
cutoff_date = datetime.strptime('2025-01-08', '%Y-%m-%d')
pre_sec = 1
ms = 10

# --- Iterate over mice ---
for mouse in mice:
    mouse_dir = os.path.join(base_dir, mouse)
    if not os.path.isdir(mouse_dir):
        continue

    sessions = sorted(os.listdir(mouse_dir))
    sessions_filtered = []

    for session in sessions:
        try:
            session_date = datetime.strptime(session, '%m%d%y')
            if session_date >= cutoff_date:
                sessions_filtered.append(session)
        except ValueError:
            continue

    for i in range(1, len(sessions_filtered)):
        session0 = sessions_filtered[i - 1]
        session1 = sessions_filtered[i]

        def get_iscell_count(session):
            try:
                iscell_path = os.path.join(base_dir, mouse, session, 'pophys', 'suite2p_BCI', 'plane0', 'iscell.npy')
                iscell = np.load(iscell_path, allow_pickle=True)
                return len(iscell)
            except:
                return -1

        if get_iscell_count(session0) != get_iscell_count(session1):
            print(f"⚠️ Skipping {session0} → {session1}: ROI mismatch based on iscell")
            continue

        def load_session(mouse, session):
            folder = os.path.join(base_dir, mouse, session, 'pophys')
            suite2p_dir = os.path.join(folder, 'suite2p_BCI', 'plane0')

            iscell = np.load(os.path.join(suite2p_dir, 'iscell.npy'), allow_pickle=True)
            stat = np.load(os.path.join(suite2p_dir, 'stat.npy'), allow_pickle=True)
            F = np.load(os.path.join(suite2p_dir, 'F.npy'), mmap_mode='r')
            ops = np.load(os.path.join(suite2p_dir, 'ops.npy'), allow_pickle=True).tolist()
            siHeader = np.load(os.path.join(suite2p_dir, 'siHeader.npy'), allow_pickle=True).tolist()
            umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

            cells = np.where(np.asarray(iscell)[:, 0] > -1)[0]
            Ftrace = F[cells, :]
            stat = stat[cells]

            dt_vol = 1 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
            if dt_vol < 0.05:
                post = round(10 / 0.05 * 0.05 / dt_vol)
                pre = round(3  / 0.05 * 0.05 / dt_vol)
            else:
                post = round(10 / 0.05)
                pre = round(3  / 0.05)

            F_filtered, *_ = ddc.create_BCI_F(Ftrace, ops, stat, pre, post)
            dist, _, conditioned_neuron, _ = ddc.find_conditioned_neurons(siHeader, stat)
            dt_si = 1 / float(siHeader['metadata']['hRoiManager']['scanFrameRate'])

            return F_filtered, dist * umPerPix, conditioned_neuron, dt_si, iscell

        try:
            F0, dist0, _, dt0, _ = load_session(mouse, session0)
            F1, dist1, cn1, dt1, iscell = load_session(mouse, session1)
        except Exception as e:
            print(f"❌ Error loading {session0} or {session1}: {e}")
            continue

        two = np.round(2 / dt0).astype(int)
        t_bci0 = np.arange(0, dt0 * F0.shape[0], dt0) - 2
        t_bci1 = np.arange(0, dt1 * F1.shape[0], dt1) - 2

        cell_inds = np.where(np.asarray(iscell)[:, 0] == 1)[0]
        cn1_index = cn1[0][0]
        cells = np.unique(np.concatenate([cell_inds, [cn1_index]]))

        F1 = F1[:, cells, :]
        F0 = F0[:, cells, :]
        dist1 = dist1[cells]
        cn1_plot_index = np.where(cells == cn1_index)[0][0]

        pre = int(round(pre_sec / dt1))
        f0 = np.nanmean(F0, axis=2)
        f1 = np.nanmean(F1, axis=2)

        f0 -= np.nanmean(f0[:pre], axis=0)
        f1 -= np.nanmean(f1[:pre], axis=0)

        delt = np.nanmean(f1[pre + 1:], axis=0) - np.nanmean(f0[pre + 1:], axis=0)

        # Plot
        fig = plt.figure(figsize=(6, 3))
        plt.subplot(121)
        plt.plot(t_bci0, f0[:, cn1_plot_index], 'k')
        plt.plot(t_bci1, f1[:, cn1_plot_index], 'm')
        plt.xlabel('Time from trial start (s)')
        plt.ylabel('$\Delta$F/F')
        plt.title("Conditioned neuron response")

        plt.subplot(122)
        plt.plot(dist1, delt, 'k.', markersize=ms)
        plt.plot(dist1[cn1_plot_index], delt[cn1_plot_index], 'm.', markersize=ms)
        plt.title(f"{mouse} {session1}")
        plt.xlabel("Distance ($\mu$m)")
        plt.ylabel("Δ F")
        plt.tight_layout()
        plt.show()

        name = mouse + '_delta_tuning_multiday'
        folder = f'C:/Users/kayvon.daie/OneDrive - Allen Institute/Documents/Data/Figures 2025/{name}.png'
        fig.savefig(folder, format='png')
