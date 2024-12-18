

#folder = '//allen/aind/scratch/BCI/2p-raw/BCI88/081924/'
#file_path = '//allen/aind/scratch/BCI/2p-raw/BCI88/081924/two_color_00001.tif'

folder = '//allen/aind/scratch/BCI/2p-raw/BCI87/082024/'
#file_path = '//allen/aind/scratch/BCI/2p-raw/BCI87/082024/two_color_00002.tif'
file_path = '//allen/aind/scratch/BCI/2p-raw/BCI87/082024/pophys/two_color_00002.tif'

#folder = '//allen/aind/scratch/BCI/2p-raw/BCI87/082224/'
#file_path = '//allen/aind/scratch/BCI/2p-raw/BCI87/082224/two_color2_00001.tif'

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
wcc, stimDist, amp, pairwiseDist, favg = stim.causal_connectivity(data)
wcc_old, stimDist_old, amp_old, pairwiseDist_old, favg_old = stim.causal_connectivity(old)
#%%
dt = data['dt_si']
t = np.arange(0, favg.shape[0] * dt, dt)
t = t - t[10]
cl = i_cells[1]
cl = np.argsort(cc)[15]
win = 25;
x = np.round(data['centroidX'][cl])
y = np.round(data['centroidY'][cl])
x = int(x)
y = int(y)
plt.subplot(223)
plt.imshow(green_avg[y - win:y + win, x - win:x + win],cmap='gray')
plt.axis('off')  # Hide the axes

plt.subplot(224)
plt.imshow(red_avg[y - win:y + win, x - win:x + win],cmap='gray')
plt.axis('off')  # Hide the axes

plt.subplot(222)
plt.plot(stimDist[cl,:]*2,amp[cl,:],'k.')
plt.xlabel('Distance from photostim target (um)')
plt.tight_layout()
plt.subplot(221)
plt.plot(t[0:30],np.nanmean(favg[0:30,cl,:],axis=1))
#%%
t = np.arange(0, favg.shape[0] * dt, dt)
t = t - t[5]
Ftrace = np.load(folder +r'/suite2p_photostim_single/plane0/F.npy', allow_pickle=True)
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
plt.plot(t[0:25],np.nanmean(targ[0:25,cells[e_cells[0:numcells]]],axis=1),'k')
plt.ylim(-.5,1.5)
plt.subplot(222)
plt.plot(t[0:25],np.nanmean(non_targ[0:25,cells[e_cells[0:numcells]]],axis=1),'k')
plt.ylim(-.1,.2)
plt.subplot(223)
plt.plot(t[0:25],np.nanmean(targ[0:25,cells[i_cells[0:numcells]]],axis=1),'r')
plt.ylim(-.5,1.5)
plt.subplot(224)
plt.plot(t[0:25],np.nanmean(non_targ[0:25,cells[i_cells[0:numcells]]],axis=1),'r')
plt.ylim(-.1,.2)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))  # Set a larger figure size for better spacing
plt.subplot(121)
plt.plot(stimDist[cells[e_cells[0:numcells]],:],amp[cells[e_cells[0:numcells]],:],'k.',markersize=2);
plt.ylim(-1,3)
plt.subplot(122)
plt.plot(stimDist[cells[i_cells[0:numcells]],:],amp[cells[i_cells[0:numcells]],:],'r.',markersize=2);
plt.tight_layout()
plt.ylim(-1,3)
#%%
x=stimDist[cells[i_cells[0:numcells]],:]
y=amp[cells[i_cells[0:numcells]],:]
bins = [0,2,5,10,20,25,35,55,100]
Y = np.zeros(len(bins)-1)
for i in range(len(bins)-1):
    ind = np.where((x>bins[i]) &(x<bins[i+1]));
    Y[i] = np.nanmean(y[ind])
plt.plot(bins[0:-1],Y,'k.-')
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

lens = 30;
evts = []
B = Ftrace*0
for ci in range(Ftrace.shape[0]):
    a = Ftrace[ci,:];
    b = np.convolve(a,np.ones(lens),'same')/lens;
    noise = np.std(b-a)*3;
    bl = np.percentile(a,20);
    evts.append(np.mean(a > (bl + noise)))
    B[ci,:] = b
cc = np.corrcoef(B);
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
#%%
folder = r'//allen/aind/scratch/BCI/2p-raw/BCI93/091224/'
old_folder = r'//allen/aind/scratch/BCI/2p-raw/BCI93/091124/'
data = ddc.main(folder)
old = ddc.main(old_folder)
wcc, stimDist, amp, pairwiseDist, favg = stim.causal_connectivity(data)
wcc_old, stimDist_old, amp_old, pairwiseDist_old, favg_old = stim.causal_connectivity(old)

#%%
dt = data['dt_si']
t = np.arange(0, favg.shape[0] * dt, dt)
t = t-t[10]
df = data['df_closedloop']
cc = np.corrcoef(df);
plt.figure(figsize=(12, 4))  # Set a larger figure size for better spacing
ci = 53
plt.subplot(141)
plt.plot(stimDist[ci,:],amp[ci,:],'k.')
#pf.mean_bin_plot(stimDist[ci,:],amp[ci,:],6,1,1,'k');plt.ylim((-.1,1))
plt.xlabel('Dist. from target')
plt.ylabel('response amp')
plt.title('neuron' + str(ci))
plt.subplot(142)
ind = np.where((np.diag(amp) > 1) & (stimDist[ci,:]>15))[0]
#plt.plot(cc[ci,ind],amp[ci,ind],'k.')
#pf.mean_bin_plot(cc[ci,ind],amp[ci,ind],6,1,1,'k')
pf.mean_bin_plot(cc[ci,ind]*np.diag(amp)[ind],amp[ci,ind],6,1,1,'k')
plt.xlabel('Corr with pre synaptic neuron')
plt.ylabel('Pairwise connection')

plt.subplot(143)
plt.plot(t[0:45],np.nanmean(favg[0:45,ci,:],axis=1))


plt.subplot(144)
pf.mean_bin_plot(pairwiseDist[:,ci],cc[:,ci],5,1,1,'k')
plt.xlabel('Pairwise dist')
plt.ylabel('Pairwise corr')

plt.tight_layout()  # Automatically adjust subplot parameters
# Remove top and right spines for all subplots
for ax in plt.gcf().get_axes():  # Get all axes from the current figure
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
#%%
tot = np.nanmean(np.nanmean(favg[50:75,:,:],axis=2),axis=0)
i_cells = np.where(tot>.02)[0]
e_cells = np.where(tot<0.02)[0]

w_corr_corr = np.zeros(favg.shape[1],)
noise_dist_corr = np.zeros(favg.shape[1],)
for ci in range(favg.shape[1]):
    ind = np.where((np.diag(amp) > 1) & (stimDist[ci,:]>15))[0]    
    w_corr_corr[ci] = np.corrcoef(cc[ci,ind],amp[ci,ind])[0,1]
    noise_dist_corr[ci] = np.corrcoef(pairwiseDist[:,ci],cc[:,ci])[0,1]
plt.plot(noise_dist_corr[i_cells],w_corr_corr[i_cells],'k.')


#%%
plt.figure(figsize=(12, 12))  # Set a larger figure size for better spacing
plt.imshow(Ftrace[0:600,0:17000],aspect = 'auto',vmin = 0,vmax = 100)
#%%
from scipy.interpolate import interp1d

ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
nums = [2,5,7,8,9,10,20,30,31,50,100,150,160,171]
x = [0, 1]  # Length 2
y = [[0, 0, 1], [0.8, 0.8,1]]  # Length 2 along the interpolation axis
xi = np.linspace(0, 1, len(nums))

col = interp1d(x, y, axis=0)(xi)  # Interpolate along the correct axis
plt.figure(figsize=(12, 8))  # Set a larger figure size for better spacing
ci = 23;

b = np.argsort(stimDist[ci,:])
a=[]
t = np.arange(0, favg.shape[0] * dt, dt)
strt = 0;
plt.subplot(222);
for i in range(len(nums)):    
    plt.plot(t[0:45]+i*t[50],favg_old[0:45,ci,b[nums[i]]],color=col[i,:])    
    plt.ylim(-.5,2.5)
plt.subplot(224);
for i in range(len(nums)):
    plt.plot(t[0:45]+i*t[50],favg[0:45,ci,b[nums[i]]],color=col[i,:])    
    plt.ylim(-.5,2.5)
plt.xlabel('Time (s)')
plt.ylabel('DF/F')    
plt.subplot(221)
ax = plt.gca()
im = ax.imshow(ops['meanImg'], cmap='gray', vmin=0, vmax=100)
plt.axis('off')
plt.title('neuron' + str(ci))
x = data['centroidX'][ci]
y = data['centroidY'][ci]
ax.plot(x, y, 'ro',markerfacecolor='none', markersize=8)
for i in range(len(nums)):
    x = data['centroidX'][b[nums[i]]]
    y = data['centroidY'][b[nums[i]]]
    ax.plot(x, y, 'o',color=col[i,:], markerfacecolor='none', markersize=8)
    
    
#%%
fr = bci.reward_triggered_average(data)
f = bci.trial_start_response(data)
plt.figure(figsize=(12, 4))  # Set a larger figure size for better spacing
plt.subplot(141)

rewtun = np.nanmean(fr[90:120,:],axis=0) - np.nanmean(fr[0:20,:],axis=0)
tun = np.nanmean(f[120:240,:],axis=0)
ci = 4;
plt.subplot(142)
ind = np.where((np.diag(amp) > 1) & (stimDist[ci,:]>15))[0]
#plt.plot(cc[ci,ind],amp[ci,ind],'k.')
#pf.mean_bin_plot(cc[ci,ind],amp[ci,ind],6,1,1,'k')
pf.mean_bin_plot(rewtun[ind]*np.diag(amp)[ind],amp[ci,ind],6,1,1,'k')
plt.xlabel('Corr with pre synaptic neuron')
plt.ylabel('Pairwise connection')
#%%
tot = np.nanmean(np.nanmean(favg[50:75,:,:],axis=2),axis=0)
b = np.argsort(-tot)
plt.plot(fr[:,b[6]])
#%%
ts = np.arange(0, df.shape[1] * dt, dt)
# Plot with imshow
plt.imshow(df[i_cells, 0:7000], aspect='auto', vmin=0, vmax=5)

# Label x-axis with time values
num_ticks = 10  # Number of ticks you want on the x-axis
tick_positions = np.linspace(0, 7000, num_ticks, dtype=int)  # Tick positions
tick_labels = [f"{int(ts[pos])}" for pos in tick_positions]  # Time labels without decimals

plt.xticks(tick_positions, tick_labels)  # Set the ticks and labels
plt.xlabel('Time (s)')
#%%
t = np.arange(0,F.shape[0]*dt,dt);
t = t - t[120]
plt.figure(figsize=(6,2))
fr,frew = bci.reward_triggered_average(data)
cc = np.corrcoef(df);
b = np.argsort(-cc[:,7]);
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(F[:,b[i],:].T,vmin=0,vmax=4,aspect='auto')
    num_ticks = 3  # Number of ticks you want on the x-axis
    tick_positions = [120,620]
    tick_labels = [f"{int(t[pos])}" for pos in tick_positions]  # Time labels without decimals
    plt.xticks(tick_positions, tick_labels)  # Set the ticks and labels
    if i == 0:
        plt.xlabel('Time from trial start (s)')
        plt.ylabel('Trial #')
    plt.title('Neuron ' + str(b[i]))
plt.show()

tr = np.arange(0,fr.shape[0]*dt,dt);
tr = tr - tr[60]
plt.figure(figsize=(6,2))
fr,frew = bci.reward_triggered_average(data)
cc = np.corrcoef(df);
b = np.argsort(-cc[:,7]);
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(frew[:,b[i],:].T,vmin=0,vmax=4,aspect='auto')
    num_ticks = 3  # Number of ticks you want on the x-axis
    tick_positions = [60,180]
    tick_labels = [f"{int(tr[pos])}" for pos in tick_positions]  # Time labels without decimals
    plt.xticks(tick_positions, tick_labels)  # Set the ticks and labels
    if i == 0:
        plt.xlabel('Time from reward (s)')
        plt.ylabel('Trial #')
    plt.title('Neuron ' + str(b[i]))
    
plt.show()

#%%
plt.figure(figsize=(4,2))
plt.subplot(121)
plt.imshow(f[:,i_cells].T,aspect='auto',vmin=-.5,vmax=.5)
num_ticks = 3  # Number of ticks you want on the x-axis
tick_positions = [120,620]
tick_labels = [f"{int(t[pos])}" for pos in tick_positions]  # Time labels without decimals
plt.xticks(tick_positions, tick_labels)  # Set the ticks and labels
plt.xlabel('Time from trial start (s)')
plt.ylabel('I neuron #')

plt.subplot(122)
plt.imshow(fr[:,i_cells].T,aspect='auto',vmin=-.5,vmax=.5)
num_ticks = 3  # Number of ticks you want on the x-axis
tick_positions = [60,180]
tick_labels = [f"{int(tr[pos])}" for pos in tick_positions]  # Time labels without decimals
plt.xticks(tick_positions, tick_labels)  # Set the ticks and labels
plt.xlabel('Time from reward (s)')
plt.ylabel('I neuron #')
plt.tight_layout()
#%%
tun_i = np.zeros(F.shape[2])
for ti in range(F.shape[2]):
    a = np.nanmean(F[:,i_cells,ti],axis=1)
    tun_i[ti] = np.nanmean(a[60:])-np.nanmean(a[0:60])
plt.plot(tun_i)
#%%
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(f[:,i_cells].T,aspect='auto',vmin=-.5,vmax=.5)
tick_positions = [120,620]
tick_labels = [f"{int(t[pos])}" for pos in tick_positions]  # Time labels without decimals
plt.xticks(tick_positions, tick_labels)  # Set the ticks and labels
plt.xlabel('Time from trial start (s)')
plt.ylabel('Neuron #')
plt.title('Neuron ' + str(b[i]))
plt.subplot(122)
plt.imshow(fo[:,i_cells].T,aspect='auto',vmin=-.5,vmax=.5)
tick_positions = [120,620]
tick_labels = [f"{int(t[pos])}" for pos in tick_positions]  # Time labels without decimals
plt.xticks(tick_positions, tick_labels)  # Set the ticks and labels
plt.show()

#%%
plt.plot(amp_old[i_cells,:].flatten(),amp[i_cells,:].flatten(),'k.',markersize=1)
plt.xlabel('W Day 1')
plt.ylabel('W Day 2')
plt.title('Connections to inhibitory neurons')
xl = plt.xlim()
plt.plot(xl,xl,'k')
#%%
cno = old['conditioned_neuron'][0][0]
f = bci.trial_start_response(data)
fo = bci.trial_start_response(old)
fro,frewo = bci.reward_triggered_average(old)
fr,frew = bci.reward_triggered_average(data)
tun = np.nanmean(f[120:,:],axis=0)
tuno = np.nanmean(fo[120:,:],axis=0)

tunr = np.nanmean(fr[0:60,:],axis=0)
tunro = np.nanmean(fro[0:60,:],axis=0)
plt.figure(figsize=(4, 8))  # Set a larger figure size for better spacing
plt.plot(tuno[e_cells],tun[e_cells],'k.')
plt.plot(tuno[i_cells],tun[i_cells],'r.')
plt.plot(tuno[cn],tun[cn],'m.',markersize=20)
plt.xlabel('Tuning day 1')
plt.ylabel('Tuning day 2')
plt.show()
delt = tun - tuno
deltr = tunr - tunro
dinp = amp - amp_old;

plt.figure(figsize=(10,3))  # Set a larger figure size for better spacing
plt.subplot(131)
plt.plot(delt[i_cells],np.nanmean(dinp[i_cells,:],axis=1),'k.')
plt.xlabel('$\Delta$ Tuning')
plt.ylabel('$\Delta$ W')
plt.subplot(132)
plt.plot(cc[i_cells,cn],np.nanmean(dinp[i_cells,:],axis=1),'k.')
plt.xlabel('Correlation with Day2 CN')
plt.ylabel('$\Delta$ W')
plt.subplot(133)
plt.plot(deltr[i_cells],np.nanmean(dinp[i_cells,:],axis=1),'k.')
plt.xlabel('$\Delta$ Tuning to reward')
plt.ylabel('$\Delta$ W')

plt.tight_layout()
plt.show()
#%%
b = np.argsort(-np.nanmean(amp,axis=1))
plt.figure(figsize=(8,4))

plt.subplot(121)
plt.imshow(wcc_old[np.ix_(b, b)], aspect='auto', cmap='bwr',vmin=-.5,vmax=.5)
plt.xlabel('Post')
plt.ylabel('Pre')
plt.title('Day 1')

plt.subplot(122)
plt.imshow(wcc[np.ix_(b, b)], aspect='auto', cmap='bwr',vmin=-.5,vmax=.5)
plt.xlabel('Post')
plt.ylabel('Pre')
plt.title('Day 2')
cbar=plt.colorbar()  # Add a colorbar
cbar.set_label('$\Delta$F/F', rotation=270, labelpad=15)  # Add label, rotate, and add padding
plt.tight_layout()

plt.show()


cco = np.corrcoef(dfo)
cc = np.corrcoef(df)

F = new['F']
Fo = old['F']
k = np.nanmean(F[60:120,:,:],axis= 0)
ko = np.nanmean(Fo[60:120,:,:],axis=0)
cc = np.corrcoef(k)
cco = np.corrcoef(ko)

b = np.argsort(-np.nanmean(amp,axis=1))
plt.figure(figsize=(8,4))

plt.subplot(121)
plt.imshow(cco[np.ix_(b, b)], aspect='auto', cmap='bwr',vmin=-1,vmax=1)
plt.xlabel('Post')
plt.ylabel('Pre')
plt.title('Day 1')

plt.subplot(122)
plt.imshow(cc[np.ix_(b, b)], aspect='auto', cmap='bwr',vmin=-1,vmax=1)
plt.xlabel('Post')
plt.ylabel('Pre')
plt.title('Day 2')
cbar=plt.colorbar()  # Add a colorbar
cbar.set_label('$\Delta$F/F', rotation=270, labelpad=15)  # Add label, rotate, and add padding
plt.tight_layout()

#%%
t = np.arange(0,Fstim.shape[1]*dt,dt)
plt.figure(figsize=(7,4))
Fstim = np.load(folder +r'/suite2p_photostim_single/plane0/F.npy', allow_pickle=True)
b = np.argsort(-np.corrcoef(Fstim[:,0:1000])[:,7])
num = 10
for i in range(num):
    plt.subplot(num,1,i+1)
    plt.plot(t[0:1000],Fstim[b[i],0:1000],'k',linewidth=.3)
for ax in plt.gcf().get_axes():  # Get all axes from the current figure
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.show()
bci.roi_show_circles(folder,data,b[0:num],show_numbers=False)