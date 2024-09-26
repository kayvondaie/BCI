import os
import re
import numpy as np
import pandas as pd
import tifffile as tiff
import scipy.io
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300

ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
from scipy.interpolate import interp1d
len_files = ops['frames_per_file'];
cn_ind = data['cn_csv_index'][0]
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['threshold_crossing_time']])
rew = ~np.isnan(rt)
# Define frm_ind (assuming ROI is already loaded)
roi = np.copy(data['roi_csv'])
frm_ind = np.arange(1, int(np.max(roi[:, 1])) + 1)

inds = np.where(np.diff(roi[:,1])>1)[0]
for i in range(len(inds)):
    ind = inds[i]
    roi[ind+1:,1] = roi[ind+1:,1] + roi[ind,1]
    roi[ind+1:,0] = roi[ind+1:,0] + roi[ind,0]

# Interpolate roi data
interp_func = interp1d(roi[:, 1], roi, axis=0, kind='linear', fill_value='extrapolate')
roi_interp = interp_func(frm_ind)


# Define the function based on BCI_threshold
fun = lambda x: (x > BCI_threshold[0]) * (x / np.diff(BCI_threshold)) * 3.3

# Initialize variables
strt = 0  # Python is 0-indexed, adjust accordingly
dt_si = np.median(np.diff(roi[:, 0]))
fcn = np.empty((350, len(len_files) - 1))
FCN = np.empty((350, len(len_files) - 1))
t_si = np.empty((350, len(len_files) - 1))


# Flatten BCI_threshold to ensure it's 1D
BCI_thresholds = data['BCI_thresholds']
ind = np.where(~np.isnan(BCI_thresholds[0,:]))[0][-1]
k = np.diff(BCI_thresholds[1,:]);
switchesu = np.where((k!=0) & (~np.isnan(k)))[0]
k = np.diff(BCI_thresholds[0,:]);
switchesl = np.where((k!=0) & (~np.isnan(k)))[0]
switches = np.unique(np.concatenate((switchesu, switchesl)))
switches  = np.concatenate(([0],switches))
avg = np.empty((len(len_files) - 1,len(switches)))

# Define the function based on the flattened BCI_threshold
for si in range(len(switches)):
    # Initialize strt at the start of the loop
    strt = 0  # Python uses 0-based indexing, corresponding to MATLAB's strt = 1
    switch = switches[si]
    # Initialize strts array to hold values
    strts = np.empty(len(len_files) - 1, dtype=int)  # Initialize with the correct length
    
    BCI_threshold = BCI_thresholds[:,switch+4]


    fun = lambda x: np.minimum((x > BCI_threshold[0]) * (x / np.diff(BCI_threshold)[0]) * 3.3, 3.3)
    
    
    # Loop through the trials
    for i in range(len(len_files) - 1):
        strts[i] = strt  # Literal translation of strts(i) = strt
        ind = np.arange(strt, strt + len_files[i], dtype=int)  # Ensure ind is an array of integers
        ind = np.clip(ind, 0, len(roi_interp) - 1)
        # Extract and process roi_interp data for fcn and t_si
        a = roi_interp[ind.astype(int), cn_ind + 2]
    
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
#%%
epochs = np.concatenate((switches, [len(rew)]))
dummy_hit = np.zeros(len(rew),)
for si in range(len(switches)):
    ind = np.arange(epochs[si], epochs[si+1])
    min_activity = 0.35
    dummy_hit[ind] = np.nanmean(avg[0:switches[1],si] > min_activity)
plt.plot(np.convolve(rew[:],np.ones(10,))/10,'k');plt.xlim(10,len(rew))
plt.plot(dummy_hit)    
plt.title(folder)
#%%
switch_frame = np.cumsum(len_files)[switch]
t = roi_interp[:,0]
plt.plot(t,roi_interp[:,cn_ind+2],'k',linewidth=.3)
plt.plot((0,t[switch_frame]),(BCI_thresholds[0,switch-1],BCI_thresholds[0,switch-1]),color = [.5,.5,1])
plt.plot((0,t[switch_frame]),(BCI_thresholds[1,switch-1],BCI_thresholds[1,switch-1]),color = [.5,.5, 1])
plt.plot((t[switch_frame],t[-1]),(BCI_thresholds[0,switch+1],BCI_thresholds[0,switch+1]),color = [0,0,1])
plt.plot((t[switch_frame],t[-1]),(BCI_thresholds[1,switch+1],BCI_thresholds[1,switch+1]),color = [0,0, 1])
plt.xlabel('Time (s)')
plt.ylabel('Raw fluorescence')

#%%
t = roi_interp[:,0]
plt.plot(t,roi_interp[:,cn_ind+2],'k',linewidth=.3)
plt.plot((0,t[switch_frame]),(BCI_thresholds[0,switch-1],BCI_thresholds[0,switch-1]),color = [0,0,1])
plt.plot((0,t[switch_frame]),(BCI_thresholds[1,switch-1],BCI_thresholds[1,switch-1]),color = [0,0, 1])
plt.plot((0,t[switch_frame]),(BCI_thresholds[0,switch+1],BCI_thresholds[0,switch+1]),color = [.5,.5,1])
plt.plot((0,t[switch_frame]),(BCI_thresholds[1,switch+1],BCI_thresholds[1,switch+1]),color = [.5,.5,1])

plt.xlabel('Time (s)')
plt.ylabel('Raw fluorescence')
plt.xlim((0,80))
plt.show()

plt.plot(1 + np.random.rand(10)/2,avg[0:switch,0],'o',color = [0,0,1])
plt.plot(2 + np.random.rand(10)/2,avg[0:switch,1],'o',color = [.5,.5,1])
plt.plot(plt.xlim(), [min_activity,min_activity], 'k:')  # 'k' for black line
plt.ylabel('Avg. voltage')
plt.xticks([1,2], ['true', 'new_thr'])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

plt.plot(avg[0:10,0],rew[0:10],'ko');
plt.plot((min_activity,min_activity),(0,1),'k:')
plt.yticks((0,1),['Miss','Hit'])
plt.xlabel
plt.xlim(0,2)
plt.ylim(-.1,1.1)
#%%




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the decaying exponential model
def exp_model(x, a, l, c):
    return a * np.exp(-x / l) + c

# Clean the data by removing rows where rt or avg are NaN
ind = np.where(~np.isnan(rt[:len(avg)]) & ~np.isnan(avg))[0]

# Prepare the x-axis values for plotting the fit
x = np.linspace(0, np.max(avg), 1000)

# Fit the exponential model to the data
popt, pcov = curve_fit(exp_model, avg[ind], rt[ind], p0=(1, 1, 1))  # Initial guess (1, 1, 1)

# Extract the fit parameters
A, l, c = popt

# Create the fitted values using the model
fit = exp_model(x, A, l, c)

# Plot the data points and the fitted curve
plt.scatter(avg[ind], rt[ind], color='k', facecolor='w', label='Data')
plt.plot(x, fit, label=f'Fit: A={A:.2f}, l={l:.2f}, c={c:.2f}')
plt.xlabel('avg "voltage"')
plt.ylabel('Time to reward (s)')

# Calculate and plot the minimum activity where fit > 10
min_activity = x[np.max(np.where(fit > 10))]
plt.axvline(min_activity, color='k', linestyle=':')

# Annotate the plot with the minimum activity value
plt.text(min_activity + 0.1, 14, f'V = {round(min_activity, 2)}')

# Add final touches and title
plt.title(folder)
plt.legend()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt

# Initialize new_thr and new_slp
pc = np.linspace(0, 90, 100)
new_thr = np.percentile(roi_interp[:, cn + 2], pc)  # Similar to MATLAB's prctile
new_thr = BCI_threshold[0] * np.linspace(0.5, 1.5, 100)
new_slp = BCI_threshold[1] * np.linspace(0.5, 2, 100)

# Initialize avg_new
avg_new = np.zeros((122, 100, 100))
new_rt = np.zeros((122, 100, 100))

# Loop over thresholds and slopes
for ti in range(len(new_thr)):
    for si in range(len(new_slp)):
        # Define the function to compute activity
        fun = lambda x: np.minimum((x > new_thr[ti]) * (x / (new_slp[si] - new_thr[ti])) * 3.3, 3.3)

        
        strt = 0  # Initialize start index
        for i in range(122):  # 122 iterations, based on your final size of avg_new
            strts = strt
            ind = np.arange(strt, strt + len_files[i], dtype=int)  # Indexes as in MATLAB
            
            # Get roi_interp data for fcn and t_si
            a = roi_interp[ind, cn + 2]
            a_padded = np.concatenate([a, np.full(300, np.nan)])  # Pad with 300 NaNs
            fcn = a_padded[:250]
            
            a = roi_interp[ind, 0] - roi_interp[ind, 0][0]  # Time-shifted
            a_padded = np.concatenate([a, np.full(300, np.nan)])  # Pad with 300 NaNs
            t_si = a_padded[:250]
            
            strt += len_files[i]  # Update strt for the next iteration
            
            # Determine stopping point
            if rew[i]:
                stp = np.max(np.where(t_si < rt[i])[0])
            else:
                stp = len(t_si)

            # Compute avg_new
            avg_new[i, ti, si] = np.nanmean(fun(fcn[:stp]))
            new_rt[i,ti,si] = exp_model(np.nanmean(fun(fcn[:stp])), A, l, c)

# Apply the model function to avg_new
#new_rt = model(avg_new)

# Compute hit rate
new_hit = np.mean(new_rt < 10, axis=0)

# Plotting the results
plt.figure()
plt.imshow(new_hit, aspect='auto', origin='upper')
plt.colorbar(label='Hit rate')

# Set Y ticks (Lower threshold)
y_inds = np.arange(0, 100, 20)
y_values = np.round(new_thr[::20])
plt.yticks(y_inds, y_values)

# Set X ticks (Upper threshold)
x_inds = np.arange(0, 100, 20)
x_values = np.round(new_slp[::20] * 100) / 100
plt.xticks(x_inds, x_values)

# Mark the current thresholds
currentx = np.min(np.where(new_slp >= BCI_threshold[1]))
currenty = np.min(np.where(new_thr >= BCI_threshold[0]))
plt.plot(currentx, currenty, 'wo', markersize=10)

# Labels
plt.xlabel('Upper threshold')
plt.ylabel('Lower threshold')
plt.title('Hit Rate Heatmap')
plt.show()
#%%
import matplotlib.pyplot as plt
BCI_threshold = threshold_data['BCI_threshold']
BCI_threshold = BCI_threshold.flatten()
t = np.arange(0, dt_si * FCN.shape[0], dt_si)
plt.figure(figsize=(6, 3))  # Adjust the width and height as needed
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Set this to the desired font size
HIT = []
for ii in range(3):
# Increase the figure size to give more space for each subplot
    
    # Loop to create subplots and plot data
    strt = 3
    stp = 4
    if ii == 0:
        AA = BCI_threshold*0
        AA[0] = new_thr[currenty]
        AA[1] = new_slp[currentx]
        hit = avg_new[:,currenty,currentx] 
        color = 'b'
    elif ii == 1:
        AA = BCI_threshold*0
        AA[0] = new_thr[80]
        AA[1] = new_slp[currentx]
        hit = avg_new[:,80,currentx] 
        color = [.5,.5,1]
    elif ii == 2:
        AA = BCI_threshold*0
        AA[0] = new_thr[currenty]
        AA[1] = new_slp[99]
        hit = avg_new[:,currenty,99] 
        color = [.9,.9,1]
    HIT.append(hit)
    fun = lambda x: np.minimum((x > AA[0]) * (x / np.diff(AA)[0]) * 3.3, 3.3)
    
    

    for i in range(stp-strt):
        # First row of subplots
        
        clr = 'k'
        
        
        plt.subplot(2, 3, ii+1)
        plt.xlim(0,10)
        plt.plot(t,FCN[:, i+strt], color=clr, linewidth=1)
        plt.plot(plt.xlim(), [AA[0], AA[0]], color=color)  # 'k' for black line
        plt.plot(plt.xlim(), [AA[1], AA[1]], color=color)  # 'k' for black line
        plt.ylim(50, 1000)
        plt.tight_layout()
            
    
    
        # Remove y-axis labels for all but the first plot in the row
        if i > 0:
            plt.gca().set_yticks([])
            plt.gca().set_ylabel('')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # Second row of subplots
        plt.subplot(2, 3, ii+4)
        plt.xlim(0,10)
        plt.plot(t,fun(FCN[:, i+strt]), color = clr, linewidth=1)        
        plt.ylim(0, 4)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    
        # Remove y-axis labels for all but the first plot in the row
        if i > 0:
            plt.gca().set_yticks([])
            plt.gca().set_ylabel('')
    
    # Optionally, you can adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust these values to change the spacing
    
plt.show()

plt.figure(figsize=(3, 3))  # Adjust the width and height as needed
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Set this to the desired font size

for i in range(3):
    if i == 0:
        color = 'b'
    elif i == 1:
        color = [.5,.5,1]
    elif i == 2:
        color = [.9,.9,1]
    y = HIT[i]
    x = i + np.random.randn(*y.shape) / 10
    plt.plot(x,y,'.',color=color)
plt.plot(plt.xlim(), [min_activity,min_activity], 'k:')  # 'k' for black line
plt.ylabel('cumulative activity')
plt.xticks([0,1,2], ['true', 'medium', 'hard'])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)



































