

#folder = '//allen/aind/scratch/BCI/2p-raw/BCI88/081924/'
#file_path = '//allen/aind/scratch/BCI/2p-raw/BCI88/081924/two_color_00001.tif'

folder = '//allen/aind/scratch/BCI/2p-raw/BCI87/081924/'
file_path = '//allen/aind/scratch/BCI/2p-raw/BCI87/082024/two_color_00002.tif'

folder = '//allen/aind/scratch/BCI/2p-raw/BCI87/082224/'
file_path = '//allen/aind/scratch/BCI/2p-raw/BCI87/082224/two_color2_00001.tif'

stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)#note that this is 
data = ddc.load_data_dict(folder)
Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)

#%%
import tifffile as tf
import numpy as np


# Read the TIF file
img = tf.imread(file_path)

# Ensure the data is a 3D array (frames x height x width)
if img.ndim != 3:
    raise ValueError("Expected a 3D array with interleaved channels.")

# Split channels
red_channel = img[1::2,:,:]  # Every other frame starting from index 0
green_channel = img[0::2,:,:]  # Every other frame starting from index 1

# Compute the average for each channel
red_avg = np.mean(red_channel, axis=0)
green_avg = np.mean(green_channel, axis=0)
#%%
import numpy as np
import matplotlib.pyplot as plt

N = len(stat)
cc = []
G = []
R = []

# Calculate G and cc
for i in range(N):
    r = red_avg[stat[i]['ypix'], stat[i]['xpix']]
    g = green_avg[stat[i]['ypix'], stat[i]['xpix']]
    G.append(np.nanmean(g))
    R.append(np.nanmean(r))
    cc.append(np.corrcoef(g, r)[0, 1])

# Convert lists to NumPy arrays
G = np.array(G)
R = np.array(R)
cc = np.array(cc)

# Filter for cells
#cells = np.where(data['iscell'][:, 1] > .2)[0]
# from sklearn.mixture import GaussianMixture

# prob_perc = .33 # cut-off for being inhibitory or excitatory
# n_clusters = 2

# dd = np.column_stack((G, cc))
# dd[np.isnan(dd)] = 1
# dd[np.isinf(dd)] = 1
# gmm = GaussianMixture(n_components=n_clusters, random_state=0)
# gmm.fit(dd)
# labels = gmm.predict(dd)
# probs = gmm.predict_proba(dd)
# probs = probs[:,0]
# if sum(labels)/len(labels)>.5:
#     labels = np.abs(labels-1)
#     probs = probs[:,1]

# #prob_perc
# putative_pyr = probs>=(1-prob_perc)
# putative_inh = probs<(prob_perc)
# unidentified = (probs<(1-prob_perc)) & (probs>=(prob_perc))

cells = np.where(data['iscell'][:, 0] ==1)[0]
i_cells = np.where((G[cells]/R[cells]>.1) | (cc[cells]<0))[0]
e_cells = np.where((G[cells]/R[cells]>.0) & (cc[cells]>0))[0]
# Plot
plt.plot(G[cells][e_cells]/R[cells][e_cells], cc[cells][e_cells], 'k.', markersize=4)
plt.plot(G[cells][i_cells]/R[cells][i_cells], cc[cells][i_cells], 'r.', markersize=4)
plt.xlim(0,.2)
plt.xlabel("Green/Red")
plt.ylabel("Pixelwise RG correlation")
plt.show()
#%%
cl = i_cells[36]
win = 15;
x = np.round(data['centroidX'][cl])
y = np.round(data['centroidY'][cl])
x = int(x)
y = int(y)
plt.subplot(223)
plt.imshow(green_avg[y - win-2:y + win-2, x - win:x + win])
plt.subplot(224)
plt.imshow(red_avg[y - win-2:y + win-2, x - win:x + win])

plt.subplot(222)
plt.scatter(stimDist[cl,:],amp[cl,:])
plt.subplot(221)
plt.plot(np.nanmean(favg[0:30,cl,:],axis=1))
#%%

favg_raw = data['photostim']['favg_raw']
favg = np.zeros(favg_raw.shape)
stimDist = data['photostim']['stimDist']
bl = np.percentile(Ftrace,50,axis=1)
bl = np.nanstd(Ftrace,axis=1)
N = stimDist.shape[0]
for i in range(N):
    favg[:,i] = (favg_raw[:,i] - np.nanmean(favg_raw[0:4,i]))/bl[i]

amp = np.nanmean(favg[11:15,:,:],axis = 0)-np.nanmean(favg[0:4,:,:],axis = 0);

non_targ = np.zeros((favg.shape[0],favg.shape[1]))
targ = np.zeros((favg.shape[0],favg.shape[1]))
for ci in range(favg.shape[1]):
    ind = np.where((stimDist[ci,:]>30) & (stimDist[ci,:]<100))[0]
    non_targ[:,ci] = np.nanmean(favg[:,ci,ind],axis=1);
    ind = np.where(stimDist[ci,:]<10)[0]
    targ[:,ci] = np.nanmean(favg[:,ci,ind],axis=1);

numcells = 40;

plt.subplot(222)
plt.imshow(non_targ[0:25,cells[e_cells[0:numcells]]].T,vmin = -1,vmax=1)
plt.subplot(224)
plt.imshow(non_targ[0:25,cells[i_cells[0:numcells]]].T,vmin = -1,vmax=1)
plt.subplot(221)
plt.imshow(targ[0:25,cells[e_cells[0:numcells]]].T,vmin = -1,vmax=1)
plt.subplot(223)
plt.imshow(targ[0:25,cells[i_cells[0:numcells]]].T,vmin = -1,vmax=1)
plt.show()

plt.subplot(221)
plt.plot(np.nanmean(targ[0:25,cells[e_cells[0:numcells]]],axis=1),'k')
plt.ylim(-.5,1)
plt.subplot(222)
plt.plot(np.nanmean(non_targ[0:25,cells[e_cells[0:numcells]]],axis=1),'k')
plt.ylim(-.1,.2)
plt.subplot(223)
plt.plot(np.nanmean(targ[0:25,cells[i_cells[0:numcells]]],axis=1),'r')
plt.ylim(-.5,1)
plt.subplot(224)
plt.plot(np.nanmean(non_targ[0:25,cells[i_cells[0:numcells]]],axis=1),'r')
plt.ylim(-.1,.2)
plt.show()

plt.subplot(121)
plt.plot(stimDist[cells[e_cells[0:numcells]],:],amp[cells[e_cells[0:numcells]],:],'k.',markersize=2);
plt.ylim(-1,3)
plt.subplot(122)
plt.plot(stimDist[cells[i_cells[0:numcells]],:],amp[cells[i_cells[0:numcells]],:],'r.',markersize=2);
plt.ylim(-1,3)
#%%
cn = data['conditioned_neuron'][0][0]
f = data['F'];
f = np.nanmean(f,axis = 2)
N = f.shape[1]
for i in range(N):
    bl = np.nanmean(f[0:19,i])
    f[:,i] = f[:,i] - bl    
tune = np.mean(f[40:100,:],axis = 0)
#%%
sorted_indices = np.concatenate((cells[e_cells], cells[i_cells]))
cc_sorted = cc[np.ix_(sorted_indices, sorted_indices)]
np.fill_diagonal(cc_sorted, 0)
plt.imshow(cc_sorted, cmap='seismic', interpolation='none',vmin=-1,vmax=1)
plt.colorbar()
e_cells_count = len(e_cells)
plt.show()
#%%
# Extract correlations
e_e_corr = cc[np.ix_(e_cells, e_cells)].flatten()  # E-E correlations
i_i_corr = cc[np.ix_(i_cells, i_cells)].flatten()  # I-I correlations
e_i_corr = cc[np.ix_(e_cells, i_cells)].flatten()  # E-I correlations

# Remove diagonal elements (self-correlations) from E-E and I-I
e_e_corr = e_e_corr[e_e_corr != 0]
i_i_corr = i_i_corr[i_i_corr != 0]

# Sort correlations for cumulative distributions
e_e_sorted = np.sort(e_e_corr)
i_i_sorted = np.sort(i_i_corr)
e_i_sorted = np.sort(e_i_corr)

# Compute cumulative distributions
e_e_cdf = np.arange(1, len(e_e_sorted) + 1) / len(e_e_sorted)
i_i_cdf = np.arange(1, len(i_i_sorted) + 1) / len(i_i_sorted)
e_i_cdf = np.arange(1, len(e_i_sorted) + 1) / len(e_i_sorted)

# Plot the cumulative distributions
plt.figure(figsize=(8, 6))
plt.plot(e_e_sorted, e_e_cdf, label="E-E", linewidth=2)
plt.plot(i_i_sorted, i_i_cdf, label="I-I", linewidth=2)
plt.plot(e_i_sorted, e_i_cdf, label="E-I", linewidth=2)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distributions of Correlations")
plt.legend()
plt.grid(True)
plt.show()

