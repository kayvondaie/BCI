def analyze_all_photostim_groups(data, folder, epoch='photostim'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind

    # Load imaging scale info
    siHeader = np.load(folder + r'/suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).tolist()
    umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

    # Load data
    stimDist = data[epoch]['stimDist'] * umPerPix  # shape: (cell, group)
    Fstim = data[epoch]['Fstim']  # shape: (time, neuron, trial)
    favg = data[epoch]['favg_raw']  # shape: (time, neuron, group)
    seq = np.asarray(data[epoch]['seq']).flatten() - 1  # MATLAB to Python index

    # Define bins
    bins = np.concatenate((np.arange(0, 100, 10), np.arange(100,300, 25)))

    # Define time windows
    t = data[epoch]['stim_params']['time']
    strt = np.where(t < 0)[0][-1] - 2
    stop = np.where(t > data[epoch]['stim_params']['total_duration'])[0][0] + 2
    dt = np.nanmedian(np.diff(t))
    window = int(np.floor(0.2 / dt))
    bef = np.arange(strt - window, strt).astype(int)
    aft = np.arange(stop, stop + window).astype(int)

    # Clean up seq to avoid indexing errors
    n_trials = Fstim.shape[2]
    seq_clean = np.where((seq >= 0) & (seq < n_trials), seq, -1)

    def run_analysis(group_range, label):
        favg_groups = favg.shape[2]
        stimDist_subset = stimDist[:, group_range]
        n_cells = Fstim.shape[1]
        pv = np.full((n_cells, len(group_range)), np.nan)
        amp = np.full((n_cells, len(group_range)), np.nan)

        for idx, gi in enumerate(group_range):
            if gi >= favg_groups:
                print(f"Skipping group {gi}: not found in favg")
                continue

            trial_inds = np.where(seq_clean == gi)[0]
            print(f"{label} - group {gi}: {len(trial_inds)} valid trials")

            if len(trial_inds) == 0:
                continue

            # Group-averaged response for amp
            amp[:, idx] = (
                np.nanmean(favg[aft[0]:aft[-1]+1, :, gi], axis=0) -
                np.nanmean(favg[bef[0]:bef[-1]+1, :, gi], axis=0)
            )

            # Trial-resolved response for stats
            post = np.nanmean(Fstim[aft[0]:aft[-1]+1, :, trial_inds], axis=0)
            pre = np.nanmean(Fstim[bef[0]:aft[-1]+1, :, trial_inds], axis=0)
            t_stat, p_value = ttest_ind(post, pre, axis=1, nan_policy='omit')
            pv[:, idx] = p_value

        # Compute fraction significant
        frac_e = np.zeros((len(bins) - 1,))
        frac_i = np.zeros((len(bins) - 1,))

        for i in range(len(bins) - 1):
            in_bin = (stimDist_subset > bins[i]) & (stimDist_subset < bins[i+1])
            in_bin_flat = in_bin.flatten()
            sig = (pv.flatten() < 0.05)
            pos = (amp.flatten() > 0)
            neg = (amp.flatten() < 0)

            denom = np.sum(in_bin_flat)
            if denom > 0:
                frac_e[i] = np.sum(in_bin_flat & sig & pos) / denom
                frac_i[i] = np.sum(in_bin_flat & sig & neg) / denom

        # Plot
        plt.figure()
        plt.bar(bins[0:3], frac_e[0:3], color=[.7, .7, .7], width=9)
        plt.bar(bins[3:-1], frac_e[3:], color='k', width=9)
        plt.bar(bins[0:-1], -frac_i, width=9, color='w', edgecolor='k')
        plt.xlabel('Distance from photostim. target (um)')
        plt.ylabel('Fraction significant')
        plt.title(f"{folder} - {label}")
        plt.ylim(-1, 1)
        plt.axhline(0, color='k')
        plt.tight_layout()
        plt.show()

    # Run both halves
    run_analysis(range(9), "First 9 photostim groups")
    run_analysis(range(9, stimDist.shape[1]), "Remaining photostim groups")
analyze_all_photostim_groups(data, folder)
#%%
favg[strt+1:stop-1, :, :] = np.nan

favg = np.apply_along_axis(
lambda m: np.interp(
    np.arange(len(m)),
    np.where(~np.isnan(m))[0] if np.where(~np.isnan(m))[0].size > 0 else [0],  # Ensure sample points are non-empty
    m[~np.isnan(m)] if np.where(~np.isnan(m))[0].size > 0 else [0]             # Use a fallback value if empty
),
axis=0,
arr=favg
)