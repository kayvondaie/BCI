import numpy as np
from scipy.stats import spearmanr, pearsonr
#%%
mice = ["BCI102"]

for mi in range(len(mice)):

    pairwise_mode = 'dot_prod'
    fit_type      = 'pinv'
    alpha         = 100
    lasso_alphas  = np.logspace(-6, 0, 60)

    num_bins      = 10
    tau_elig      = 10
    shuffle       = 0
    plotting      = 1
    fitting       = 1
    correct_direct = True

    mouse = mice[mi]
    session_inds = np.where((list_of_dirs['Mouse'] == mouse) & (list_of_dirs['Has data_main.npy'] == True))[0]
    si = 5

    for sii in range(si, si + 1):
        try:
            print(sii)
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si','step_time','reward_time','BCI_thresholds']
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except:
                continue
        except:
            continue

        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        cn = int(data["conditioned_neuron"][0][0])

        N, n_groups = stimDist.shape
        W0 = np.full((N, N), np.nan)
        W1 = np.full((N, N), np.nan)
        targs = np.argmin(stimDist, axis=0)

        for gi in range(n_groups):
            targ = targs[gi]
            if not (
                (stimDist[targ, gi] < 10)
                and (AMP[0][targ, gi] > 0)
                and (AMP[1][targ, gi] > 0)
            ):
                continue

            nontarg = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))[0]
            if nontarg.size == 0:
                continue

            W0[nontarg, targ] = AMP[0][nontarg, gi]
            W1[nontarg, targ] = AMP[1][nontarg, gi]

        dW = W1 - W0

        F = data["F"]
        one = max(1, int(np.round(1 / data["dt_si"])))
        pre = np.nanmean(np.nanmean(F, 2)[:one, :], 0).astype(float)
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)

        W0_eff = np.nan_to_num(W0, nan=0.0, posinf=0.0, neginf=0.0)
        dW_eff = np.nan_to_num(dW, nan=0.0, posinf=0.0, neginf=0.0)

        T = 10

        x = np.zeros((T + 1, N))
        x[0, :] = pre
        for t in range(T):
            x[t + 1, :] = W0_eff @ x[t, :]

        t1, t2 = 1, T

        s = np.zeros((T + 1, N))
        for tt in range(t1, t2 + 1):
            s[tt, cn] += x[tt, cn]
        for t in range(T - 1, -1, -1):
            s[t, :] = W0_eff.T @ s[t + 1, :]

        Grad = np.zeros((N, N))
        for t in range(T):
            Grad += np.outer(s[t + 1, :], x[t, :])

        obs_mask = np.isfinite(W0) & np.isfinite(W1)
        obs_mask[:, cn] = False

        g = Grad[obs_mask].ravel()
        dw = dW_eff[obs_mask].ravel()

        if g.size > 3:
            r_p, p_p = pearsonr(g, dw)
            r_s, p_s = spearmanr(g, dw)
        else:
            r_p = p_p = r_s = p_s = np.nan

        mpos = g > 0
        mneg = g < 0

        keep = (g != 0) & np.isfinite(dw)
        

        r_p, p_p = pearsonr(g[keep], dw[keep]) if np.sum(keep) > 3 else (np.nan, np.nan)
        r_s, p_s = spearmanr(g[keep], dw[keep]) if np.sum(keep) > 3 else (np.nan, np.nan)


        w0_edge = W0[obs_mask].ravel()
        dw_edge = (W1 - W0)[obs_mask].ravel()
        A = np.vstack([w0_edge, np.ones_like(w0_edge)]).T
        beta = np.linalg.lstsq(A, dw_edge, rcond=None)[0]
        dw_res = dw_edge - (beta[0]*w0_edge + beta[1])

        g_use = g.copy()
        dw_use = dw_res.copy()

        m = np.isfinite(g_use) & np.isfinite(dw_use)
        print("corr(dw_res, g):", pearsonr(g_use[m], dw_use[m])[0], spearmanr(g_use[m], dw_use[m])[0])
        pf.mean_bin_plot(dw_use[m],g_use[m],5)
        pf.mean_bin_plot(g_use[m],dw_use[m],5)
