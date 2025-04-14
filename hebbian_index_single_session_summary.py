
import session_counting
import data_dict_create_module_test as ddct

list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
mouse = 'BCI106'
session = '020725'
folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
#%%
photostim_keys = ['stimDist', 'favg_raw']
bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si']
data = ddct.load_hdf5(folder,bci_keys,photostim_keys )


iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True);
cells = np.where(np.asarray(iscell)[:,0]==1)[0]
spont_pre = np.load(folder +r'/suite2p_spont_pre/plane0/F.npy', allow_pickle=True)[cells,:]
spont_post = np.load(folder +r'/suite2p_spont_post/plane0/F.npy', allow_pickle=True)[cells,:]
#%% 
AMP = []
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
umPerPix = 1000/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
for epoch_i in range(2):
    if epoch_i == 0:
        stimDist = data['photostim']['stimDist']*umPerPix 
        favg_raw = data['photostim']['favg_raw']
    else:
        stimDist = data['photostim2']['stimDist']*umPerPix 
        favg_raw = data['photostim2']['favg_raw']
    favg = favg_raw*0
    for i in range(favg.shape[1]):
        favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[0:3, i]))/np.nanmean(favg_raw[0:3, i])
    favg[18:26, :, :] = np.nan
    
    favg[0:30,:,:] = np.apply_along_axis(
    lambda m: np.interp(
        np.arange(len(m)),
        np.where(~np.isnan(m))[0] if np.where(~np.isnan(m))[0].size > 0 else [0],  # Ensure sample points are non-empty
        m[~np.isnan(m)] if np.where(~np.isnan(m))[0].size > 0 else [0]             # Use a fallback value if empty
    ),
    axis=0,
    arr=favg[0:30,:,:]
    )

    amp = np.nanmean(favg[26:35, :, :], axis=0) - np.nanmean(favg[10:19, :, :], axis=0)
    AMP.append(amp)
    plt.plot(np.nanmean(np.nanmean(favg[0:40,:,:],axis=2),axis=1))
#%%
import plotting_functions as pf



stimDist = data['photostim']['stimDist']*umPerPix
plt.figure(figsize=(8,4))  # Set figure size to 10x10 inches
df = data['df_closedloop']
cc = np.corrcoef(df)
F = data['F']
ko = np.nanmean(F[120:360,:,0:40],axis=0)
kn = np.nanmean(F[120:360,:,40:],axis=0)
k = np.nanmean(F[120:360,:,:],axis=0)
cc = np.corrcoef(k)
cco = np.corrcoef(ko)
ccn = np.corrcoef(kn)


ei = 1;
X = []
X2 = []
Y = []
Yo = []
for gi in range(stimDist.shape[1]):
    cl = np.where((stimDist[:,gi]<15) & (AMP[0][:,gi]> .1) * ((AMP[1][:,gi]> .1)))[0]
    
    x = np.nanmean(cc[cl,:],axis=0)    
    x2 = np.nanmean(ccn[cl,:] - cco[cl,:],axis=0)
    nontarg = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<10000))
    y = AMP[1][nontarg,gi]
    yo = AMP[0][nontarg,gi]
    
    # y = AMP[1][nontarg,gi]/np.nanmean(AMP[1][cl,gi])*np.nanmean(AMP[0][cl,gi])
    # yo = AMP[0][nontarg,gi]
    
    #plt.scatter(x[nontarg],amp[nontarg,gi])
    X.append(x[nontarg])
    X2.append(x2[nontarg])
    Y.append(y)
    Yo.append(yo)



X = np.concatenate(X)
X2 = np.concatenate(X2)
Y = np.concatenate(Y,axis=1)
Yo = np.concatenate(Yo,axis=1)
plt.subplot(231)
pf.mean_bin_plot(X,Yo,5,1,1,'k')
plt.xlabel('$correlation_{i,j}$')
plt.ylabel('$W_{i,j}$')
plt.title('Before learning')

plt.subplot(234)
pf.mean_bin_plot(X,Y,5,1,1,'k')
plt.xlabel('$correlation_{i,j}$')
plt.ylabel('$W_{i,j}$')
plt.title('After learning')

plt.subplot(132)
pf.mean_bin_plot(X,Y-Yo,5,1,1,'k')
plt.xlabel('$correlation_{i,j}$')
plt.ylabel('$\Delta W_{i,j}$')
plt.tight_layout()

plt.subplot(133)
pf.mean_bin_plot(X2,Y-Yo,6,1,1,'k')
plt.xlabel('$\Delta correlation_{i,j}$')
plt.ylabel('$\Delta W_{i,j}$')
plt.tight_layout()
plt.title(data['mouse'] + ' ' + data['session'])
plt.title(data['mouse'] + ' ' + data['session'])
