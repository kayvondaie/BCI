import session_counting
import data_dict_create_module_test as ddct
import numpy as np
import re

list_of_dirs = session_counting.counter()
#%%
session_inds = np.where((list_of_dirs['Mouse'] == 'BCI102') & (list_of_dirs['Has data_main.npy'] == True))[0]
X_list = []
Y_list = []
Yo_list = []
si = 5;
#for sii in range(len(session_inds)):
for sii in range(si,si+1):
    mouse = list_of_dirs['Mouse'][session_inds[sii]]
    session = list_of_dirs['Session'][session_inds[sii]]
    folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'

    photostim_keys = ['stimDist', 'favg_raw']
    bci_keys = ['F', 'dt_si', 'BCI_thresholds']
    data = ddct.load_hdf5(folder, bci_keys, photostim_keys)

    siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
    umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

    AMP = []
    for epoch_i in range(2):
        stim_key = 'photostim' if epoch_i == 0 else 'photostim2'
        stimDist = data[stim_key]['stimDist'] * umPerPix
        favg_raw = data[stim_key]['favg_raw']
        favg = (favg_raw - np.nanmean(favg_raw[0:3, :], axis=0)) / np.nanmean(favg_raw[0:3, :], axis=0)

        dt_si = data['dt_si']
        after = int(np.floor(0.2 / dt_si))
        before = int(np.floor(0.2 / dt_si))

        artifact = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
        artifact = artifact - np.nanmean(artifact[0:4])
        artifact = np.where(artifact > 0.5)[0]
        artifact = artifact[artifact < 40]

        pre = (int(artifact[0] - before), int(artifact[0] - 2))
        post = (int(artifact[-1] + 2), int(artifact[-1] + after))

        favg[artifact, :, :] = np.nan
        favg[0:30, :] = np.apply_along_axis(
            lambda m: np.interp(
                np.arange(len(m)),
                np.where(~np.isnan(m))[0] if np.any(~np.isnan(m)) else [0],
                m[~np.isnan(m)] if np.any(~np.isnan(m)) else [0]
            ),
            axis=0,
            arr=favg[0:30, :]
        )

        amp = np.nanmean(favg[post[0]:post[1], :, :], axis=0) - np.nanmean(favg[pre[0]:pre[1], :, :], axis=0)
        AMP.append(amp)

    F = data['F']
    trl = F.shape[2]
    tsta = np.arange(0, 12, data['dt_si']) - 2  # Centered around 0

    ts = np.where((tsta > 0) & (tsta < 2))[0]
    k = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)
    k[np.isnan(k)] = 0
    
    CC = np.corrcoef(k)

    X, Y, Yo = [], [], []
    for gi in range(stimDist.shape[1]):
        cl = np.where((stimDist[:, gi] < 10) & (AMP[0][:, gi] > 0.1) & (AMP[1][:, gi] > 0.1))[0]
        if len(cl) > 0:
            x = np.nanmean(CC[cl, :], axis=0)
            nontarg = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))[0]
            y = AMP[1][nontarg, gi]
            yo = AMP[0][nontarg, gi]
            X.append(x[nontarg])
            Y.append(y)
            Yo.append(yo)
    if X:
        X_list.append(np.concatenate(X))
        Y_list.append(np.concatenate(Y))
        Yo_list.append(np.concatenate(Yo))
#%%
from scipy.stats import pearsonr
import numpy as np

cc = []
cco= []
for x, y, yo in zip(X, Y, Yo):
    x = np.asarray(x)
    y = np.asarray(y)
    yo = np.asarray(yo)

    # Filter out pairs where either x or y is NaN or inf
    valid = np.isfinite(x) & np.isfinite(y)
    
    if np.sum(valid) > 1:  # need at least 2 points to compute correlation
        r, _ = pearsonr(x[valid], y[valid])
        cc.append(r)
        
        r, _ = pearsonr(x[valid], yo[valid])
        cco.append(r)
    else:
        cc.append(np.nan)
        cco.append(np.nan)

b = np.argsort(np.asarray(cco)-np.asarray(cc))
i = b[-7]
pf.mean_bin_plot(X[i],Y[i],5,1,1,'m')
pf.mean_bin_plot(X[i],Yo[i],5,1,1,'k')
#%%
CC = np.corrcoef(k[:,-40:])
numCells = favg.shape[1]
stim_cells = np.argmin(stimDist,axis=0)
direct_amp = amp[stim_cells,np.arange(0,100)]
rec = []
reco = []
for ci in range(numCells):
    ind = np.where((stimDist[ci,:] > 30) & (direct_amp > 0.1))[0]
    #plt.scatter(CC[ci,ind],amp[ci,ind])
    a = np.corrcoef(CC[ci,ind],AMP[0][ci,ind])[0,1]
    b = np.corrcoef(CC[ci,ind],AMP[1][ci,ind])[0,1]
    rec.append(b)
    reco.append(a)
    #pf.mean_bin_plot(CC[ci,ind],AMP[0][ci,ind],3,1,1,'k')
    #pf.mean_bin_plot(CC[ci,ind],AMP[1][ci,ind],3,1,1,'m')
rec = np.asarray(rec)
reco = np.asarray(reco)
b = np.argsort(reco - rec)
ci = b[13]
ci = 160
ind = np.where((stimDist[ci,:] > 30) & (direct_amp > 0.1))[0]
pf.mean_bin_plot(CC[ci,ind],AMP[0][ci,ind],3,1,1,'k')
pf.mean_bin_plot(CC[ci,ind],AMP[1][ci,ind],3,1,1,'m')
plt.plot(plt.xlim(),(0,0),'k:')
plt.ylim(tuple(np.max(np.abs(plt.ylim())) * np.array([-1, 1])))

#%%
from scipy.stats import ttest_ind
b = np.asarray(reco)
a = np.asarray(rec)
ind = (np.isnan(rec)==0) * (np.isnan(reco)==0)
t_stat, p_val = ttest_ind(np.abs(a[ind]),np.abs(b[ind]), equal_var=False)  # Welch's t-test (safer if variances differ)
print(f"t = {t_stat:.3f}, p = {p_val:.3g}")


#%%
# Final outputs
X = np.array(X_list)
Y = np.array(Y_list)
Yo = np.array(Yo_list)

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0
Yo[np.isnan(Yo)] = 0
