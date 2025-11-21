data = dict()
data['photostim'] = np.load('//allen/aind/scratch/BCI/2p-raw/BCI116/101625/pophys/data_photostimBCI116_101625.npy',allow_pickle=True)
#%%
folder = '//allen/aind/scratch/BCI/2p-raw/BCI116/101625/pophys/suite2p_photostim_single/plane0/'
iscell = np.load(os.path.join(folder, 'iscell.npy'), allow_pickle=True)
stat = np.load(os.path.join(folder, 'stat.npy'), allow_pickle=True)
F = np.load(os.path.join(folder, 'F.npy'), allow_pickle=True)
ops = np.load(os.path.join(folder, 'ops.npy'), allow_pickle=True).tolist()
siHeader = np.load(os.path.join(folder, 'siHeader.npy'), allow_pickle=True).tolist()

_, _, _, _, _, _, _, _, _, data['photostim']['Fstim_raw'], _, _ = ddc.stimDist_single_cell(ops,F,siHeader,stat)
#%%
Fstim2 = Fstim * 0
Fstim_raw = data['photostim']['Fstim_raw']
bl = np.nanstd(F,1)
for i in range(Fstim_raw.shape[1]):
    a = Fstim_raw[:,i,:]
    pre = np.nanmean(a[0:8,:],0)    
    Fstim2[:,i,:] = (a-pre)/bl[i]

#%%
def kd_filt(x, bins):
    t = np.arange(0, 2*bins)
    kernel = np.exp(-t / bins)
    kernel /= kernel.sum()
    a = np.convolve(x, kernel, mode='full')[:len(x)]
    return a


epoch = 'photostim'
from scipy.stats import ttest_ind
stimDist = data[epoch]['stimDist']
#Ftsim = Fstim_raw.copy()
Fstim = data['photostim']['Fstim']
#Fstim = Fstim2.copy()
#plt.scatter(stimDist[:,gi],-np.log(p_value))
bins = [30,100]
favg = data['photostim']['favg']
bins = np.linspace(0,200,10)
bins = np.arange(0,300,10)
bins = np.concatenate((np.arange(0, 100, 10), np.arange(100, 300, 25)))
G = stimDist.shape[1]
num_connect = np.zeros((len(bins)-1,G))
frac_e = np.zeros((len(bins)-1,G))
frac_i = np.zeros((len(bins)-1,G))
frac_connect = np.zeros((len(bins)-1,G))
pv = np.zeros((Fstim.shape[1],favg.shape[2]))
pv_reb = np.zeros((Fstim.shape[1],favg.shape[2]))
favg = data[epoch]['favg_raw']
t = data['photostim']['stim_params']['time']
t = t - t[10]
strt = np.where(t < 0)[0][-1]
stop = np.where(t > data['photostim']['stim_params']['total_duration'])[0][0]
amp = np.nanmean(favg[10:25, :, :], axis=0) - np.nanmean(favg[0:8, :, :], axis=0)
Fstim_sort = []
for gi in range(G):    
    seq = data[epoch]['seq']-1
    ind = np.where(seq == gi)[0][::]
    post = np.nanmean(Fstim[10:25, :, ind], axis=0) 
    reb = np.nanmean(Fstim[25:40, :, ind], axis=0) 
    pre  = np.nanmean(Fstim[0:8, :, ind], axis=0)
    Fstim_sort.append(Fstim[:,:,ind])
    ind = np.where(np.sum(np.isnan(pre[0:10,:]),axis=0)==0)[0]
    t_stat, p_value = ttest_ind(post[:,ind], pre[:,ind], axis=1)
    t_stat, p_value_reb = ttest_ind(reb[:,ind], pre[:,ind], axis=1)
    cl = np.argmin(stimDist[:,gi])
    cc = np.corrcoef(post[:,ind]-pre[:,ind])[:,cl]
    
    
    pv[:,gi] = p_value
    pv_reb[:,gi] = p_value_reb
#%%
frac_e = np.zeros((len(bins)-1,))
frac_i = np.zeros((len(bins)-1,))
for i in range(len(bins)-1):
    ind = np.where((stimDist.flatten() > bins[i]) & (stimDist.flatten()<bins[i+1]))[0]
    inde = np.where((stimDist.flatten() > bins[i]) & (stimDist.flatten()<bins[i+1]) & (pv.flatten() < 0.05) & (amp.flatten()>0))[0]    
    indi = np.where((stimDist.flatten() > bins[i]) & (stimDist.flatten()<bins[i+1]) & (pv.flatten() < 0.05) & (amp.flatten()<0))[0]    
     
    frac_i[i] = len(indi)/len(ind)
    frac_e[i] = len(inde)/len(ind)
    
plt.bar(bins[0:2],frac_e[0:2],color=[.7,.7,.7],width=9)
plt.bar(bins[2:-1],frac_e[2:],color='k',width=9)
plt.bar(bins[0:-1],-frac_i,width=9,color='w',edgecolor='k')
plt.xlabel('Distance from photostim. target (um)')
plt.ylabel('Fraction significant')
plt.title(folder)
#%%
T = data['photostim']['stim_params']['time']
T = T - T[10]
def kd_filt(x, bins):
    t = np.arange(0,2*bins)
    kernel = np.exp(-t/bins)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode='same')

b = np.argsort(pv,0)
#b = np.argsort(pv_reb,0)
plt.figure(figsize=(10,5))
favg = data['photostim']['favg']
amp = np.nanmean(favg[10:25,:,:],0) - np.nanmean(favg[0:9,:,:],0)
gi = 8
nums = 21
stim_start = T[10]
stim_end   = T[25]
rows = 3
for i in range(nums):
    plt.subplot(rows,int(nums/rows),i+1)
    
    # shape: (48 frames, n_trials)
    aa = Fstim_sort[gi][0:48, b[i,gi], :]
    
    # mean and SEM across trials
    m = np.nanmean(aa, axis=1)
    s = np.nanstd(aa, axis=1) / np.sqrt(aa.shape[1])
    
    # smooth
    m_s = kd_filt(m, 3)[:-2]
    s_s = kd_filt(s, 3)[:-2]

    t = T[0:len(m_s)]

    # shaded error
    plt.fill_between(
        t,
        m_s - s_s,
        m_s + s_s,
        alpha=0.3,
        color='k',
        linewidth=0
    )
    plt.plot(t, m_s, 'k')

    # photostim bar (frames 10:25)
    ymin, ymax = plt.ylim()
    plt.axvspan(stim_start, stim_end, ymin=0.03, ymax=1,  # small band near bottom
                color='cyan', alpha=0.4)

    plt.title(f" {np.round(stimDist[b[i,gi],gi])} $\mu$m")

plt.tight_layout()


#%%
targs = np.argmin(stimDist,0)
gi = 16;
f = np.nanmean(Fstim_sort[gi][0:70,:,:],2)
pop = np.where(np.nanvar(f,0) < 100)[0]
f = f[:,pop]
forig = f.copy()
f[np.isnan(f)] = 0
f2 = f.copy()
win = 11;
for i in range(f.shape[1]):
    f[:,i] = medfilt(f[:,i],win)
    f2[:,i] = kd_filt(f[:,i],win)
u,s,v = np.linalg.svd(f[0:50,:])
v = v.T;
a = 1
vp = v[:,a] * (v[:,a]>0)
vn = -v[:,a] * (v[:,a]<0)
plt.subplot(311)
plt.plot(f @ vp,'k.-')

plt.subplot(312)
plt.plot(f @ vp,'b.-')
plt.plot(f @ vn,'r.-')

plt.subplot(313)
plt.plot(forig @ vp,'b.-')
plt.plot(forig @ vn,'r.-')







