ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
from scipy.interpolate import interp1d
len_files = ops['frames_per_file'];
cn_ind = data['cn_csv_index'][0]
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['threshold_crossing_time']])
st = np.array([x[0] if len(x) > 0 else np.nan for x in data['SI_start_times']])
rt = rt - st;

rew = ~np.isnan(rt)
# Define frm_ind (assuming ROI is already loaded)
roi = np.copy(data['roi_csv'])
frm_ind = np.arange(1, int(np.max(roi[:, 1])) + 1)

inds = np.where(np.diff(roi[:,1])<0)[0]
for i in range(len(inds)):
    ind = inds[i]
    roi[ind+1:,1] = roi[ind+1:,1] + roi[ind,1]
    roi[ind+1:,0] = roi[ind+1:,0] + roi[ind,0]

# Interpolate roi data
interp_func = interp1d(roi[:, 1], roi, axis=0, kind='linear', fill_value='extrapolate')
roi_interp = interp_func(frm_ind)


# Define the function based on BCI_threshold
fun = lambda x: min(3.3,(x > BCI_threshold[0]) * ((x- BCI_threshold[0]) / np.diff(BCI_threshold)) * 3.3)

# Initialize variables
strt = 0  # Python is 0-indexed, adjust accordingly
dt_si = np.median(np.diff(roi[:, 0]))
fcn = np.empty((350, len(len_files) - 1))
FCN = np.empty((350, len(len_files) - 1))
t_si = np.empty((350, len(len_files) - 1))


# Flatten BCI_threshold to ensure it's 1D
BCI_thresholds = data['BCI_thresholds']
if (data['mouse'] == 'BCI106') & (data['session'] == '020725'):    
    BCI_thresholds[:,0] = BCI_thresholds[:,3]
    BCI_thresholds[:,1] = BCI_thresholds[:,3]
    BCI_thresholds[:,2] = BCI_thresholds[:,3]
ind = np.where(~np.isnan(BCI_thresholds[0,:]))[0][-1]
k = np.diff(BCI_thresholds[1,:]);
switchesu = np.where((k!=0) & (~np.isnan(k)))[0]
k = np.diff(BCI_thresholds[0,:]);
switchesl = np.where((k!=0) & (~np.isnan(k)))[0]
switches = np.unique(np.concatenate((switchesu, switchesl)))
switches = switchesu
switches  = np.concatenate(([0],switches))
avg = np.empty((len(len_files) - 1,len(switches)))
avg_raw = np.empty((len(len_files) - 1,len(switches)))

# Define the function based on the flattened BCI_threshold
for si in range(len(switches)):
    # Initialize strt at the start of the loop
    strt = 0  # Python uses 0-based indexing, corresponding to MATLAB's strt = 1
    switch = switches[si]
    # Initialize strts array to hold values
    strts = np.empty(len(len_files) - 1, dtype=int)  # Initialize with the correct length
    
    BCI_threshold = BCI_thresholds[:,switch+2]


    fun = lambda x: np.minimum((x > BCI_threshold[0]) * ((x-BCI_threshold[0]) / np.diff(BCI_threshold)[0]) * 3.3, 3.3)
    t = roi_interp[:,0]
    trl_frm = np.zeros(len(t),)
    thr_time = np.full((len(t),2),np.nan);
    # Loop through the trials
    for i in range(len(rew) - 1):
        strts[i] = strt  # Literal translation of strts(i) = strt
        ind = np.arange(strt, strt + len_files[i], dtype=int)  # Ensure ind is an array of integers
        ind = np.clip(ind, 0, len(roi_interp) - 1)
        # Extract and process roi_interp data for fcn and t_si
        a = roi_interp[ind.astype(int), cn_ind + 2]
        thr_time[ind,0] = BCI_thresholds[0,i]
        thr_time[ind,1] = BCI_thresholds[1,i]
        # Pad with 300 NaNs as in the MATLAB code, and then select the first 250 elements
        a_padded = np.concatenate([a, np.full(400, np.nan)])
        fcn[:, i] = a_padded[:350]
        FCN[:, i] = a_padded[:350]
    
        # Repeat for t_si (first column of roi_interp)
        a = roi_interp[ind.astype(int), 0]
        a = a - a[0]  # Shift time values
    
        # Pad with 300 NaNs as in the MATLAB code, and then select the first 250 elements
        a_padded = np.concatenate([a, np.full(400, np.nan)])
        t_si[:, i] = a_padded[:350]
    
        strt = strt + len_files[i]  # Update strt for the next trial
    
        # Determine the stopping point
        if rew[i]:
            stp = np.max(np.where(t_si[:, i] < rt[i])[0])
        else:
            stp = t_si.shape[0]
        
        # Calculate average for this trial
        avg[i,si] = np.nanmean(fun(fcn[:stp, i]))
        avg_raw[i,si] = np.nanmean((fcn[:stp, i]))
        FCN[stp:,i] = np.nan
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()    
fig = plt.figure(figsize=(6, 4))  # Adjust the width and height as needed
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8  # Set this to the desired font size
plt.subplot(231)
epochs = np.concatenate((switches, [len(rew)]))
dummy_hit = np.zeros(len(rew),)
dummy_rt = np.zeros(len(switches),)
actual_rt = np.zeros(len(switches),)
upr = np.unique(BCI_thresholds[1,:])[0:-1]
upr = BCI_thresholds[1,switches+1]
lwr = np.unique(BCI_thresholds[0,:])[0]
for si in range(len(switches)):
    ind = np.arange(epochs[si], epochs[si+1])
    min_activity = float(siHeader['metadata']['hRoiManager']['linesPerFrame'])/800*.35
    dummy_hit[ind] = np.nanmean(avg[0:10,si] > min_activity)
    alpha = (upr[0]-lwr)/(upr[si]-lwr)
    print('alpha = ' + str(1/alpha))
    dummy_hit[ind] = np.mean(rt[0:switches[1]]/alpha<10)
    dummy_rt[si] = np.nanmean(rt[0:switches[1]]/alpha);
    actual_rt[si] = np.nanmean(rt[switches[si-1]:switches[si]]);
plt.plot(np.convolve(rew[:],np.ones(10,))/10,'k');plt.xlim(8,len(rew))
plt.plot(dummy_hit,color = 'gray')    
plt.xlabel('Trial #')
plt.ylabel('Hit rate')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
for i in range(len(switches)):
    plt.plot((switches[i],switches[i]),(0,.1),'k')
#plt.title(folder)

plt.subplot(232)
switch_frame = np.cumsum(len_files)[switch]
plt.plot(t,roi_interp[:,cn_ind+2],'k',linewidth=.04)
plt.plot(t,thr_time,'b')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('Time (s)')
plt.ylabel('Raw fluorescence')
plt.title(data['mouse'] + '  ' + data['session'])

plt.subplot(233);
F = data['F'];
cn = data['conditioned_neuron'][0][0]
#plt.imshow(F[:,cn,:].T,vmin = np.nanmin(BCI_thresholds),vmax=np.nanmax(BCI_thresholds), aspect='auto')
#plt.imshow(F[:,cn,:].T,vmin = np.nanmin(BCI_thresholds)/4,vmax=np.nanmax(BCI_thresholds)/4, aspect='auto')
plt.imshow(F[:,cn,:].T,aspect='auto')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
plt.xticks([120,720], ['0', '10'])
plt.xlabel('Time from trial start (s)')
plt.ylabel('Trial #')

plt.subplot(234)
plt.plot(np.convolve(np.nanmean(F[:,cn,:],axis=0),np.ones(10,))[10:-10]/10,'k')
plt.xlabel('Trial #')
plt.ylabel('CN activity')
for i in range(len(switches)):
    plt.plot((switches[i],switches[i]),(0,.1),'k')

plt.subplot(235)    
ff = F[:,cn,:]
for ti in range(ff.shape[1]):        
     ff[:,ti] = ff[:,ti] - np.nanmean(ff[0:20,ti]);    
n = switches[1]
plt.plot(np.convolve(np.nanmean(ff[60:,:],axis=0),np.ones(n,))[n:-n]/n,'k')
plt.xlabel('Trial #')
plt.ylabel('CN Tuning')
for i in range(len(switches)):
     plt.plot((switches[i],switches[i]),(0,.1),'k')

plt.subplot(236)
x = np.arange(0,len(actual_rt)*3,3)
plt.bar(x,actual_rt,color = 'k')
x = np.arange(1,len(actual_rt)*3+1,3)
plt.bar(x,dummy_rt, color ='gray')
plt.legend(['Real','expected'])
plt.xlabel('Epoch')
plt.ylabel('Time to reward (s)')
plt.tight_layout()



fig = plt.figure(figsize=(5,2.5))  # Adjust the width and height as needed
rt_epoch = np.zeros((50,len(switches)))
tuning_epoch = np.zeros((50,len(switches)))
tuning = np.nanmean(ff[60:,:],axis=0)
for i in range(len(switches)-1):
    ind = np.arange(switches[i]-3,switches[i+1])
    ind[ind<0] = 0
    b = rt[ind];
    b = b - np.nanmean(b[0:3])
    if len(b) > 50:
        b = b[0:50]
    a = np.zeros(50,)
    a[0:len(b)] = b;
    rt_epoch[:,i] = a
    
    b = tuning[ind];
    b = b - np.nanmean(b[0:3])
    a = np.zeros(50,)
    if len(b) > 50:
        b = b[0:50]
    a[0:len(b)] = b;
    tuning_epoch[:,i] = a
rt_epoch[rt_epoch==0] = np.nan
tuning_epoch[tuning_epoch==0] = np.nan


x = np.arange(0,rt_epoch.shape[0])
x = x - 3
plt.subplot(121)
plt.plot(x,np.nanmean(rt_epoch,axis=1),'k.-')    
plt.xlabel('Trials since Thr change')
plt.ylabel('$\Delta$ Time to reward (s)')
# delta_rt.append(np.nanmean(rt_epoch,axis=1))

plt.subplot(122)
plt.plot(x,np.nanmean(tuning_epoch,axis=1),'k.-')
plt.xlabel('Trials since Thr change')
plt.ylabel('$\Delta$ Time to reward (s)')
plt.tight_layout()


#name = mouse +'_'+ session + '_epoch_reward_time'
#folder = 'C:/Users/kayvon.daie/OneDrive - Allen Institute/Documents/Data/Figures 2025/' + name +'.png'
#fig.savefig(folder, format='png')