photostim_keys = ['stimDist', 'seq','Fstim','favg','favg_raw']
bci_keys = []
data = ddct.load_hdf5(folder,bci_keys,photostim_keys )
#%%
epoch = 'photostim2'
from scipy.stats import ttest_ind
stimDist = data[epoch]['stimDist']
Fstim = data[epoch]['Fstim'];
#plt.scatter(stimDist[:,gi],-np.log(p_value))
bins = [30,100]
bins = np.linspace(0,200,10)
bins = np.arange(0,300,10)
bins = np.concatenate((np.arange(0, 100, 10), np.arange(100, 300, 25)))
G = stimDist.shape[1]
num_connect = np.zeros((len(bins)-1,G))
frac_e = np.zeros((len(bins)-1,G))
frac_i = np.zeros((len(bins)-1,G))
frac_connect = np.zeros((len(bins)-1,G))
pv = np.zeros((Fstim.shape[1],favg.shape[2]))
favg = data[epoch]['favg_raw']
amp = np.nanmean(favg[26:35, :, :], axis=0) - np.nanmean(favg[10:19, :, :], axis=0)
for gi in range(G):    
    seq = data[epoch]['seq']-1
    ind = np.where(seq == gi)[0][::]
    post = np.nanmean(Fstim[27:35, :, ind], axis=0) 
    pre  = np.nanmean(Fstim[10:16, :, ind], axis=0)
    ind = np.where(np.sum(np.isnan(pre[0:10,:]),axis=0)==0)[0]
    t_stat, p_value = ttest_ind(post[:,ind], pre[:,ind], axis=1)
    cl = np.argmin(stimDist[:,gi])
    cc = np.corrcoef(post[:,ind]-pre[:,ind])[:,cl]
    
    
    pv[:,gi] = p_value
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
gi = 39
ci = a[0]
cl = np.argmin(stimDist[:,gi])
seq = data[epoch]['seq']-1
ind = np.where(seq == gi)[0][::]
post = np.nanmean(Fstim[27:35, :, ind], axis=0)
b = np.argsort(stimDist[:,gi])
plt.scatter(post[cl,:],post[b[ci],:])
aa = np.where(np.isnan(post[cl,:])==0)[0]
m,p = pearsonr(post[cl,aa].T,post[[ci],aa].T)
plt.title(str(stimDist[[ci],gi]) + ' p= ' + str(p))
plt.show()

cl = np.argmin(stimDist[:,gi])
direct = post[cl,:]
direct[np.isnan(direct)==1] = 0
small = np.where(direct<np.percentile(direct,70))[0]
big = np.where(direct>=np.percentile(direct,70))[0]
amp_big = np.nanmean(post[:,big],axis=1)
amp_small = np.nanmean(post[:,small],axis=1)
clss = np.where(stimDist[:,gi]>30)
pf.mean_bin_plot(stimDist[clss,gi],amp_big[clss]-amp_small[clss],9,1,1,'k')

#%%
plt.figure(figsize=(6, 2))  # Adjust the width and height as needed
dt = 1/58
tstim = np.arange(0,Fstim.shape[0]*dt,dt)
tstim = tstim - tstim[26]
Fstim = data[epoch]['Fstim'];
Fstim[18:26, :, :] = np.nan

seq = data[epoch]['seq']-1
indd = np.where(seq == gi)[0][::]
post = np.nanmean(Fstim[27:35, :, indd], axis=0) 
pre  = np.nanmean(Fstim[10:16, :, indd], axis=0)
plt.subplot(221)
plt.plot(tstim[0:50],Fstim[0:50,cl,indd],linewidth=.2)

ind = np.where(np.sum(np.isnan(pre[0:10,:]),axis=0)==0)[0]
t_stat, p_value = ttest_ind(post[:,ind], pre[:,ind], axis=1)
cl = np.argmin(stimDist[:,gi])
amps = (post - pre)[:,ind]

cc = np.corrcoef(amps)[:,cl]
ci = np.where(cc<-.3)[0]
cii = 11
plt.subplot(223)
plt.plot(tstim[0:50],Fstim[0:50,ci[cii],indd],linewidth=.2)
plt.subplot(122)
plt.plot(amps[cl,:],amps[ci[cii],:],'k.')
plt.tight_layout()







