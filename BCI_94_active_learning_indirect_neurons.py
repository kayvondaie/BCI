import os
from scipy.io import loadmat
import numpy as np

# Set the root directory where the search begins
root_dir = r'\\allen\aind\scratch\BCI\2p-raw\BCI94\120524'

# List to store matched paths
matched_paths = []
siHeader = np.load(root_dir + r'/suite2p_spont/plane0/siHeader.npy', allow_pickle=True).tolist()
# Loop through the directory to find directories starting with 'suite2p_al'
for entry in os.listdir(root_dir):
    if entry.startswith('suite2p_al'):
        # Add the full path to the matched_paths list using os.path.join
        full_path = os.path.join(root_dir, entry)
        matched_paths.append(full_path)

# Standard MATLAB-style loop
FAVG = []
SD = []
AMP = []
FSTIM = []
SEQ = []
offset = 0
for p_i in range(len(matched_paths)):
    # Construct the file path dynamically for .npy files
    file_name = f"data_BCI94_120524_i{p_i}.npy"
    file_path = os.path.join(matched_paths[p_i], file_name)

    # Normalize the path for consistency
    file_path = os.path.normpath(file_path)

        # If it's an .npy file, load and process it
    data = np.load(file_path, allow_pickle=True).tolist()
    
    mat_file = r'\\allen\aind\scratch\BCI\2p-raw\BCI94\120524\backup\patterns_i' + str(p_i) + '.mat'
    mat_data = loadmat(mat_file)
    
    Fstim = data['photostim']['Fstim']
    favg_raw = data['photostim']['favg_raw']
    favg = np.zeros(favg_raw.shape)
    stimDist = data['photostim']['stimDist']
    #bl = np.percentile(Ftrace, 50, axis=1)
    N = stimDist.shape[0]
    
    # Process photostimulation data
    for i in range(N):
        favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[0:4, i]))/np.nanmean(favg_raw[0:4, i])
    
    amp = np.nanmean(favg[8:16, :, :], axis=0) - np.nanmean(favg[0:4, :, :], axis=0)
    FAVG.append(favg);
    SD.append(data['photostim']['stimDist'])
    AMP.append(amp)
    FSTIM.append(Fstim)
    seq = data['photostim']['seq']-1
    seq = seq + offset
    offset = len(seq) + offset
    SEQ.append(seq)

#%%
stat = np.load(root_dir + '/suite2p_spont/plane0/' + 'stat.npy', allow_pickle=True)
Ftrace = np.load(root_dir +r'/suite2p_spont/plane0/F.npy', allow_pickle=True)
dummy = np.zeros((Ftrace.shape[0],100))
for ci in range(Ftrace.shape[0]):
    for ii in range(100):
        ind = int(np.random.rand()*(Ftrace.shape[1]-20) + 6)
        aft = np.nanmean(Ftrace[ci,ind+10])
        bef = np.nanmean(Ftrace[ci,ind-5])
        dummy[ci,ii] = (aft-bef)/bef
dummy = np.nanstd(dummy,axis=1)
amp_z = amp*0       
for i in range(amp.shape[0]):
    amp_z[i,:] = amp[i,:]/dummy[i]
#%%

from scipy.stats import ttest_ind
stimDist = np.concatenate(SD,axis=1);
Fstim = np.concatenate(FSTIM,axis=2);
amp = np.concatenate(AMP,axis=1)
seq = np.concatenate(SEQ)
p_value = np.zeros((100,100))
dist = np.zeros((100,100))
w = np.zeros((100,100))
for post in range(100):
    for pre in range(100):
        direct = 10;
        near = 30;
        grps = np.where((stimDist[pre,:]<direct) & (stimDist[post,:]>near));
        if len(grps[0]) > 0:
            bef = np.nanmean(Fstim[0:4,post,grps[0]],axis=0);
            aft = np.nanmean(Fstim[8:16,post,grps[0]],axis=0)
            t_stat, p_value[post,pre] = ttest_ind(bef, aft)
            dx = (np.mean(stat[pre]['xpix']) - np.mean(stat[post]['xpix']))**2
            dy = (np.mean(stat[pre]['ypix']) - np.mean(stat[post]['ypix']))**2
            dist[post,pre] = np.sqrt(dx+dy)
            w[post,pre] = np.nanmean(amp[post,pre])
            
#%%
bins = np.concatenate((np.arange(0, 100, 10), np.arange(100, 300, 25)))
bins = np.arange(0, 200, 10)

umPerPix = 1000/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

x = stimDist.flatten()*umPerPix;
y = amp_z.flatten();
frac_e = np.zeros(len(bins))
frac_i = np.zeros(len(bins))
for bi in range(len(bins)-1):
    ind = np.where((x>bins[bi]) & (x<bins[bi+1]))[0]
    frac_e[bi] = np.nansum(y[ind]>1)/max(seq)
    frac_i[bi] = np.nansum(y[ind]<-1)/max(seq)

plt.bar(bins[0:3],frac_e[0:3],color=[.7,.7,.7],width=9)
plt.bar(bins[3:],frac_e[3:],color='k',width=9)
plt.bar(bins[:],-frac_i[:],color='w',width=9,edgecolor='k')
plt.xlabel('Distance from photostim (um)')
plt.ylabel('Number of responsive neurons (z-score > 1)')

        
    



    