# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:33:48 2025

@author: kayvon.daie
"""

data = dict()
data['photostim'] = np.load('//allen/aind/scratch/BCI/2p-raw/BCI116/101525/pophys/data_photostimBCI116_101525.npy',allow_pickle=True)
siHeader = np.load(os.path.join(folder, 'siHeader.npy'), allow_pickle=True).tolist()
umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

#%%
data['dt_si'] = 1/38
favg_raw = data['photostim']['favg_raw']
Ftrace = data['photostim']['Ftrace']
bl = np.nanstd(Ftrace,1)
favg = favg_raw*0
for i in range(favg_raw.shape[1]):
    for j in range(favg_raw.shape[2]):
        try:
            a = favg_raw[:,i,j].copy()
            aa = np.nanmean(a[0:10])
            favg[:,i,j] = (a-aa)/bl[i]
        except:
            continue
gi = 2;
#favg  = data['photostim']['favg']

stimDist = data['photostim']['stimDist']
targs = np.argmin(stimDist,0)
plt.subplot(121);
plt.plot(favg[:,targs[gi],gi])
#%%
for gi in range(favg.shape[2]):
    ind = np.where(stimDist[:,gi] > 200)[0];
    b = np.nanmean(favg[:,ind,gi],1)
    favg[:, :, gi] = favg[:, :, gi] - b[:, None]


#%%

gi = 15;
#favg  = data['photostim']['favg']
def kd_filt(x, bins):
    t = np.arange(0, 2*bins)
    kernel = np.exp(-t / bins)
    kernel /= kernel.sum()
    a = np.convolve(x, kernel, mode='full')[:len(x)]
    return a
stim_start = 10;
stim_end = 25
targs = np.argmin(stimDist,0)
stimDist = data['photostim']['stimDist']
plt.subplot(121);
plt.plot(favg[:,targs[gi],gi])
plt.show()
from scipy.signal import medfilt
f = favg[0:70,:,gi]
f[~np.isfinite(f)] = 0
f[np.isnan(f)] = 0
pop = np.where(np.var(f,0) < 100)[0]
pop = pop[pop<500]
f = f[:,pop]
forig = f.copy()
f[np.isnan(f)] = 0
f2 = f.copy()
win = 11;
for i in range(f.shape[1]):
    f[:,i] = medfilt(f[:,i],win)
    f2[:,i] = kd_filt(f[:,i],win)
f[:,targs[gi]] = 0
u,s,v = np.linalg.svd(f[0:50,:])
v = v.T;
a = 1
v[:,a] = -v[:,a] * np.sign((forig @ v[:,a])[15])
vp = v[:,a] * (v[:,a]>0)
vn = -v[:,a] * (v[:,a]<0)
T = data['photostim']['stim_params']['time']
T = T - T[10]
plt.subplot(311)
plt.plot(T[0:70],forig @ v[:,a],'k.-')
plt.axvspan(stim_start, stim_end, ymin=0.03, ymax=1,color='cyan', alpha=0.4)

plt.subplot(312)
plt.plot(f @ vp,'b.-')
plt.plot(f @ vn,'r.-')

plt.subplot(313)
plt.plot(forig @ vp,'b.-')
plt.plot(forig @ vn,'r.-')
plt.show()



#%%
b = np.argsort(-v[:,1]);
i = 9
plt.plot(f[0:,b[i]])
plt.plot(f2[0:,b[i]])
plt.plot(forig[0:,b[i]])
plt.show()
#%%
b = np.argsort(-v[:,1]);

plt.figure(figsize = (9,3))

plt.subplot(131)
plt.plot(T[0:70],favg[0:70,targs[gi],gi])
plt.axvspan(stim_start, stim_end, ymin=0.03, ymax=1,color='cyan', alpha=0.4)
plt.xlabel('Time from photostim start (s)')
plt.ylabel('DF/F')
plt.title('Target neuron')

plt.subplot(132)
plt.plot(T[0:70],forig @ v[:,1],'k-')
plt.axvspan(stim_start, stim_end, ymin=0.03, ymax=1,color='cyan', alpha=0.4)
plt.xlabel('Time from photostim start (s)')
plt.ylabel('DF/F')
plt.title('Network response (SVD 1)')

plt.subplot(133)
plt.imshow(f[:,b].T,aspect = 'auto',cmap = 'bwr',vmin = -.25,vmax = .25)
plt.xticks([10, 48], ['0', '1'])
plt.ylabel('Neuron')
plt.xlabel('Time from photostim start (s)')
plt.colorbar()
plt.tight_layout()
#%%
plt.figure(figsize=(10,5))
for i in range(1,11):
    plt.subplot(2,5,i)
    T = data['photostim']['stim_params']['time']
    T = T - T[10]
    # mean and SEM across trials
    aa = Fstim_sort[gi][0:70,pop[b[i]],:]
    m = np.nanmean(aa, axis=1)
    s = np.nanstd(aa, axis=1) / np.sqrt(aa.shape[1])
    
    # smooth
    m_s = kd_filt(m, 5)[:-2]
    s_s = kd_filt(s, 5)[:-2]
    
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
    plt.axvspan(t[stim_start], t[stim_end], ymin=0.03, ymax=1,  # small band near bottom
                color='cyan', alpha=0.4)
plt.tight_layout()
#%%
from scipy.signal import medfilt
favg = data['photostim']['favg']
stimDist = data['photostim']['stimDist']
targs = np.argmin(stimDist,0)
F = []
V = []
for gi in range(favg.shape[2]):
    
    f = favg[0:70,:,gi]
    f[:,targs[gi]] = 0
    f[~np.isfinite(f)] = 0
    f[np.isnan(f)] = 0
    pop = np.where(np.nanvar(f,0) < 100)[0]
    f = f[:,pop]
    forig = f.copy()
    f[np.isnan(f)] = 0
    for i in range(f.shape[1]):
        f[:,i] = medfilt(f[:,i],win)
    u,s,v = np.linalg.svd(f[0:50,0:])
    v = v.T;
    F.append(forig[:,0:] @ v[:,1])
    v[:,1] = v[:,1] * np.sign(F[gi][15])
    F[gi] = -F[gi] * np.sign(F[gi][15])
    
    k = np.zeros()
    V.append(v[:,1])

#%%
ff = np.stack(F)
#plt.plot(np.nanmean(ff[0:40,:],0))
plt.plot(t,np.nanmean(ff[:,:-2],0))
ymin, ymax = plt.ylim()
plt.axvspan(stim_start, stim_end, ymin=0.03, ymax=1,  # small band near bottom
            color='cyan', alpha=0.4)
plt.xlabel('Time from photostim start (s)')
plt.ylabel('DF/F')
plt.title('Network response (SVD 1)')
#%%
targs = np.argmin(stimDist,0)
a = np.stack(V)
cc = np.corrcoef(a)
plt.imshow(cc,vmin = -.8,vmax = .8,cmap = 'bwr')
targ_dist = np.zeros((50,50))
for i in range(50):
    for j in range(50):
        x1 = np.nanmean(stat[targs[i]]['xpix'])
        x2 = np.nanmean(stat[targs[i]]['ypix'])
        y1 = np.nanmean(stat[targs[j]]['xpix'])
        y2 = np.nanmean(stat[targs[j]]['ypix'])
        targ_dist[i,j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        
#%%
A = []
for gi in range(stimDist.shape[1]):
    ind = np.where((stimDist[:,gi]>20)&(stimDist[:,gi]<100))[0];
    a = np.nanmedian(favg[:,ind,gi],1);
    A.append(a)
A = np.stack(A)
plt.plot(np.nanmedian(A[0:40,0:50],0));plt.plot(np.nanmedian(A[41:,0:50],0))
#%%
A = []
for gi in range(stimDist.shape[1]):
    A.append(favg[:,targs[gi],gi])
A = np.stack(A)
plt.plot(np.nanmean(A[0:40,0:50],0));
plt.plot(np.nanmean(A[41:,0:50],0))    
#%%
b = np.argsort(v[:,1])
plt.imshow(f[0:50,b].T,aspect = 'auto',vmin = -.5,vmax = .5,cmap='bwr');plt.colorbar()


#%%
plt.figure(figsize=(8,4))
T = data['photostim']['stim_params']['time']
T = T - T[10]
t = T[0:50]
from scipy.signal import medfilt
stimDist = data['photostim']['stimDist']
targs = np.argmin(stimDist,0)
V,FF = [],[]
bins = (0,10,30,1000)
bins = (0,5,15,30,50,80,100,150,1000)

for bi in range(len(bins)-1):
    plt.subplot(2,int((len(bins)/2)),bi + 1)
    F = []    
    for gi in range(favg.shape[2]):
        
        f = favg[0:70,:,gi]
        f[:,targs[gi]] = 0
        f[~np.isfinite(f)] = 0
        f[np.isnan(f)] = 0
        pop = np.where(np.nanvar(f,0) < 100)[0]
        f = f[:,pop]
        ind = np.where((stimDist[pop,gi]>bins[bi]) & (stimDist[pop,gi]<bins[bi+1]))[0]
        a = np.nanmean(f[:,ind],1)
        
        ind = np.where(stimDist[pop,gi]>200)[0]
        b = np.nanmean(f[:,ind],1)
        
        F.append(a)
    ff = np.stack(F)
    FF.append(ff)
    #plt.plot(np.nanmean(ff[0:40,:],0))
    plt.plot(t,np.nanmean(ff[:,0:50],0),'k')
    ymin, ymax = plt.ylim()
    plt.axvspan(t[stim_start], t[stim_end], ymin=0.03, ymax=1,  # small band near bottom
                color='cyan', alpha=0.4)
    plt.title(f" {bins[bi]} $\mu$m")

    if bi == 1:
        plt.xlabel('Time from photostim start (s)')
    if bi == 0:
        plt.ylabel('DF/F')
plt.tight_layout()    
#%%
amp = np.nanmean(favg[10:25,:,:],0) - np.nanmean(favg[0:9,:,:],0)
pf.fixed_bin_plot(stimDist[0:,:].flatten(),amp[0:,:].flatten(),15,1,1,'k',(0,5,10,15,20,25,30,40,50,60,70,80,90,100,150,200,300,2000));plt.plot(plt.xlim(),(0,0),'k:')
plt.xlabel('Distance from target $\mu m$')
plt.ylabel('$\Delta F/F$')
#%%
amp_reb = np.nanmean(favg[30:48,:,:],0) - np.nanmean(favg[0:9,:,:],0)












