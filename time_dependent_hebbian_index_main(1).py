
import session_counting
import data_dict_create_module_test as ddct

list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
from bci_time_series import *

HI  = []
RT  = []
HIT = []
HIa = []
HIb = []
HIc = []
DOT = []
TRL = []
THR = []
RPE = []
CN  = []
RPE_FIT = []
AVG_RPE = []
CC_RPE, CC_RT, HIT_RATE, CN_SNR, D_HIT_RATE, RPE_VAR, CORR_RPE, CORR_RT, RT_WINDOW, HIT_WINDOW, THR_WINDOW = [],[],[],[],[],[],[],[],[],[],[]
PTRL, PVAL, RVAL,Ddirect,Dindirect,CCdirect,MOUSE,SESSION = [], [], [], [], [],[],[],[]
mice = ["BCI102","BCI105","BCI106","BCI109"]
for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) & (list_of_dirs['Has data_main.npy']==True))[0]
    #session_inds = np.where((list_of_dirs['Mouse'] == 'BCI103') & (list_of_dirs['Session']=='012225'))[0]
    si = 6
    
    pairwise_mode = 'dot_prod'  #dot_prod, noise_corr,dot_prod_no_mean
    fit_type      = 'ridge'     #ridge, pinv
    alpha         =  .1        #only used for ridge
    epoch         =  'reward'  # reward, step, trial_start
    for sii in range(0,len(session_inds)):        
    #for sii in range(si,si+1):
        num_bins      =  2000         # number of bins to calculate correlations
        print(sii)
        mouse = list_of_dirs['Mouse'][session_inds[sii]]
        session = list_of_dirs['Session'][session_inds[sii]]
        folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
        photostim_keys = ['stimDist', 'favg_raw']
        bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si','step_time','reward_time','BCI_thresholds']
        try:
            data = ddct.load_hdf5(folder,bci_keys,photostim_keys )
        except:
            continue        
        BCI_thresholds = data['BCI_thresholds']
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
            dt_si = data['dt_si']
            after = np.floor(0.2/dt_si)
            before = np.floor(0.2/dt_si)
            if mouse == "BCI103":
                after = np.floor(0.5/dt_si)
            artifact = np.nanmean(np.nanmean(favg_raw,axis=2),axis=1)
            artifact = artifact - np.nanmean(artifact[0:4])
            artifact = np.where(artifact > .5)[0]
            artifact = artifact[artifact<40]
            pre = (int(artifact[0]-before),int(artifact[0]-2))
            post = (int(artifact[-1]+2),int(artifact[-1]+after))
            favg[artifact, :, :] = np.nan
            
            favg[0:30,:] = np.apply_along_axis(
            lambda m: np.interp(
                np.arange(len(m)),
                np.where(~np.isnan(m))[0] if np.where(~np.isnan(m))[0].size > 0 else [0],  # Ensure sample points are non-empty
                m[~np.isnan(m)] if np.where(~np.isnan(m))[0].size > 0 else [0]             # Use a fallback value if empty
            ),
            axis=0,
            arr=favg[0:30,:]
            )
        
            amp = np.nanmean(favg[post[0]:post[1], :, :], axis=0) - np.nanmean(favg[pre[0]:pre[1], :, :], axis=0)
            AMP.append(amp)
            #plt.plot(np.nanmean(np.nanmean(favg[0:40,:,:],axis=2),axis=1))
        
        from scipy.stats import pearsonr
        import numpy as np
    




#%%
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt

X1_list, X2_list, Y1_list, Y2_list,RTHIr = [], [], [], [], []
# Get indices of significant sessions
valid_sess = [i for i in range(len(PVAL)) if PVAL[i] < 0.05]

# Then bootstrap from only those

for si in range(100):
    X_list = []
    y_list = []
    rt_list = []
    #sess_inds = np.random.choice(len(HIb), size=len(HIb), replace=True)
    sess_inds = np.random.choice(valid_sess, size=len(valid_sess), replace=True)


    for ii in range(len(valid_sess)):
        i = sess_inds[ii]
        n = len(HIb[i])
        if n == 0 or np.nanstd(HIb[i]) == 0:
            continue
        hi = np.asarray(HIb[i])
        thr = np.asarray(THR[i])[:n]
        if len(thr) <= 2 or thr[2] == 0:
            continue
        thr_norm = thr / thr[2]
        X_list.append(thr_norm)
        y_list.append(hi / np.nanstd(hi))
        rt_list.append(RT[i])

    x = np.concatenate(X_list)
    y = np.concatenate(y_list)
    z = np.concatenate(rt_list)
    ind = np.where(np.isnan(x) == 0)[0]
    x = x[ind]
    y = y[ind]
    z = z[ind]

    trl_inds = np.random.choice(len(x), size=len(x), replace=True)
    x = x[trl_inds]
    y = y[trl_inds]
    z = z[trl_inds]

    ind = np.where(z < 20)[0]
    r,p = pearsonr(z[ind], y[ind])
    RTHIr.append(r)
    X1, Y1, _ = pf.mean_bin_plot(z[ind], y[ind], 5, 110, 1, 'k')
    X2, Y2, _ = pf.mean_bin_plot(x, y, 3, 110, 1, 'k')

    X1_list.append(X1)
    X2_list.append(X2)
    Y1_list.append(Y1)
    Y2_list.append(Y2)

X1 = np.stack(X1_list, axis=0)
X2 = np.stack(X2_list, axis=0)
Y1 = np.stack(Y1_list, axis=0)
Y2 = np.stack(Y2_list, axis=0)

# Compute means and empirical 95% confidence intervals
Y1_mean = np.nanmean(Y1, axis=0)
Y1_low = np.nanpercentile(Y1, 2.5, axis=0)
Y1_high = np.nanpercentile(Y1, 97.5, axis=0)

Y2_mean = np.nanmean(Y2, axis=0)
Y2_low = np.nanpercentile(Y2, 2.5, axis=0)
Y2_high = np.nanpercentile(Y2, 97.5, axis=0)

X1_mean = np.nanmean(X1, axis=0)
X2_mean = np.nanmean(X2, axis=0)


plt.figure(figsize=(7, 3))

plt.subplot(121)
plt.plot(X1_mean, Y1_mean, color='k')
plt.fill_between(X1_mean, Y1_low, Y1_high, color='k', alpha=0.3)
plt.xlabel('Time to reward (s)')
plt.ylabel('Hebbian Index')

plt.subplot(122)
plt.plot(X2_mean, Y2_mean, color='k')
plt.fill_between(X2_mean, Y2_low, Y2_high, color='k', alpha=0.3)
plt.xlabel('THR')
plt.ylabel('Hebbian Index')

plt.tight_layout()
plt.show()

1 - np.nanmean(np.array(RTHIr)>0)


#%%
from scipy.stats import ttest_rel
import numpy as np
import matplotlib.pyplot as plt

pre = 5
post = 26
n_boot = 1000

A_boot = []
B_boot = []

valid_sess = [i for i in range(len(PVAL)) if PVAL[i] < 0.05]

for si in range(n_boot):
    A_list = []
    B_list = []

    # Bootstrap resample session indices with replacement
    sess_inds = np.random.choice(valid_sess, size=len(valid_sess), replace=True)

    for si_ in sess_inds:
        x = THR[si_]
        y = CORR_RPE[si_]
        z = HIT_WINDOW[si_]

        ind = np.where(np.diff(x) > 0)[0]
        for i in range(len(ind)):
            dx = (x[ind[i]+1] - x[ind[i]]) / x[ind[i]]
            a = y[ind[i]-pre:ind[i]+post] / dx
            b = z[ind[i]-pre:ind[i]+post]

            if len(a) == pre + post and not np.any(np.isnan(a)):
                A_list.append(a)
                B_list.append(b)

    # Stack into arrays (aligned trials × events)
    if len(A_list) > 0 and len(B_list) > 0:
        A_arr = np.column_stack(A_list)
        B_arr = np.column_stack(B_list)

        A_boot.append(np.nanmean(A_arr, axis=1))
        B_boot.append(np.nanmean(B_arr, axis=1))

A_boot = np.stack(A_boot, axis=0)  # shape: (n_boot, time)
B_boot = np.stack(B_boot, axis=0)

xaxis = np.arange(pre + post) - pre

# Compute mean + 95% CI
A_mean = np.nanmean(A_boot, axis=0)
A_low = np.nanpercentile(A_boot, 2.5, axis=0)
A_high = np.nanpercentile(A_boot, 97.5, axis=0)

B_mean = np.nanmean(B_boot, axis=0)
B_low = np.nanpercentile(B_boot, 2.5, axis=0)
B_high = np.nanpercentile(B_boot, 97.5, axis=0)

# === Plot ===
plt.figure(figsize=(9,3))

plt.subplot(131)
plt.plot(xaxis, A_mean, color='k')
plt.fill_between(xaxis, A_low, A_high, color='k', alpha=0.3)
plt.axvline(0, linestyle='--', color='gray')
plt.ylabel('RPE weight')
plt.xlabel('Trials from threshold increase')

plt.subplot(132)
plt.plot(xaxis, B_mean, color='blue')
plt.fill_between(xaxis, B_low, B_high, color='blue', alpha=0.3)
plt.axvline(0, linestyle='--', color='gray')
plt.ylabel('Hit rate')
plt.xlabel('Trials from threshold increase')

plt.tight_layout()

# Optional: bootstrap test for early vs. late RPE
early = np.nanmean(A_boot[:, 0:3], axis=1)
late = np.nanmean(A_boot[:, -20:], axis=1)
pval = 2*np.mean(late < early) if np.mean(late) > np.mean(early) else np.mean(early > late)
print(f"Bootstrap p-value (early vs. late RPE): {pval:.6f}")

#%%
from scipy.stats import pearsonr
from sklearn.utils import resample

dhit = []
drt  = []
for i in range(len(THR)):
    ind = np.where(np.diff(THR[i]) > 0)[0]
    if len(ind) > 0:
        ind = ind[0]
        d = np.nanmean(HIT[i][ind-2:ind]) - np.nanmean(HIT[i][ind+1:ind+15])
        dd = np.nanmean(RT[i][ind-2:ind]) - np.nanmean(RT[i][ind+1:ind+15])
        dhit.append(dd)
        drt.append(dd)
    else:
        dhit.append(np.nan)
        drt.append(np.nan)


x = np.asarray(dhit)
y = np.asarray(CC_RPE)
ind = np.where((np.isnan(x)==0) & (np.isnan(y)==0))[0]
plt.scatter(-x,y)
pf.mean_bin_plot(-x,y,4,1,1,'k')
plt.xlabel('D Hit after thr change')
plt.ylabel('RPE weight')
pearsonr(-x[ind],y[ind])
#%%

# Original data preparation
x = np.asarray(dhit)
y = np.asarray(CC_RPE)
ind = np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
x = -x[ind]  # Flip x here to match scatter plot
y = y[ind]

# Fit line using bootstrapping
n_boot = 1000
x_fit = np.linspace(np.min(x), np.max(x), 100)
y_boot = np.zeros((n_boot, len(x_fit)))
c_boot = np.zeros((n_boot,))

for i in range(n_boot):
    xb, yb = resample(x, y)
    coef = np.polyfit(xb, yb, 1)
    c_boot[i] = coef[0]
    y_boot[i] = np.polyval(coef, x_fit)

# Compute central fit and confidence bounds
y_median = np.median(y_boot, axis=0)
y_lower = np.percentile(y_boot, 2.5, axis=0)
y_upper = np.percentile(y_boot, 97.5, axis=0)

# Plot
plt.subplot(133)
plt.scatter(x, y, alpha=0.7, edgecolor='k')
plt.plot(x_fit, y_median, color='k', linewidth=2, label='Bootstrapped fit')
plt.fill_between(x_fit, y_lower, y_upper, color='gray', alpha=0.4, label='95% CI')
plt.xlabel('Δ Hit after threshold change')
plt.ylabel('RPE weight')
plt.legend(loc='lower left')  # or use `bbox_to_anchor=(1, 1)`
plt.title(f"r = {pearsonr(x, y)[0]:.2f}, p = {pearsonr(x, y)[1]:.3g}")
plt.show()

#%%
plt.plot(CC_RT,CC_RPE,'k.')
plt.xlabel('Correlation between HI and RT')
plt.ylabel('Correlation between HI and RPE')

#%%
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

map_sorting = np.argsort(np.nansum(np.nanmean(CCts, axis=2), axis=0))
inds = np.arange(0, X.shape[0])[2::3]
dy = (Y - Yo).flatten()
y = Y.flatten()
yo = Yo.flatten()

plt.figure(figsize=(12, 6))

# Top left heatmap (HI > 0)
plt.subplot(231)
tr = np.where((np.array(ptrl) < .001) & (np.array(b) > 0))[0]
cc_slice = CCts[:, :, tr[2]]
cc_sorted = cc_slice[np.ix_(map_sorting, map_sorting)]
plt.imshow(cc_sorted, aspect='auto', cmap='jet', interpolation='none',
           vmin=0, vmax=5000)
plt.title(fr'$C_{{ij}}^{{tr={tr[2]}}}$')  # <-- this line
plt.xlabel('Neuron')
plt.ylabel('Neuron')
plt.colorbar()

# Bottom left heatmap (HI < 0)
plt.subplot(234)
tr = np.where((np.array(ptrl) < .01) & (np.array(b) < 0))[0]
cc_slice = CCts[:, :, tr[2]]
cc_sorted = cc_slice[np.ix_(map_sorting, map_sorting)]
plt.imshow(cc_sorted, aspect='auto', cmap='jet', interpolation='none',
           vmin=0, vmax=5000)
plt.title(fr'$C_{{ij}}^{{tr={tr[2]}}}$')  # <-- this line
plt.xlabel('Neuron')
plt.ylabel('Neuron')
plt.colorbar()

# Top middle: ΔW vs coactivity (HI > 0)
plt.subplot(232)
tr = np.where((np.array(ptrl) < .001) & (np.array(b) > 0))[0]
xvals = X[inds[tr[2]], :].reshape(-1, 1)
pf.mean_bin_plot(xvals, dy, 5, 1, 1, 'k')
r, p = pearsonr(xvals.flatten(), dy)
plt.text(0.95, 0.95, f"r = {r:.2f}\np = {p:.1e}",
         ha='right', va='top', transform=plt.gca().transAxes)
plt.xlabel(fr'$C_{{ij}}^{{tr={tr[2]}}}$')  # <-- this line
plt.ylabel(r'$\Delta W_{ij}$')

# Bottom middle: ΔW vs coactivity (HI < 0)
plt.subplot(235)
tr = np.where((np.array(ptrl) < .01) & (np.array(b) < 0))[0]
xvals = X[inds[tr[2]], :].reshape(-1, 1)
pf.mean_bin_plot(xvals, dy, 5, 1, 1, 'k')
r, p = pearsonr(xvals.flatten(), dy)
plt.text(0.95, 0.95, f"r = {r:.2f}\np = {p:.1e}",
         ha='right', va='top', transform=plt.gca().transAxes)
plt.xlabel(fr'$C_{{ij}}^{{tr={tr[2]}}}$')  # <-- this line
plt.ylabel(r'$\Delta W_{ij}$')

# Top right: Post vs Pre (HI > 0)
plt.subplot(233)
tr = np.where((np.array(ptrl) < .001) & (np.array(b) > 0))[0]
xvals = X[inds[tr[2]], :].reshape(-1, 1)
pf.mean_bin_plot(xvals, y, 5, 1, 1, 'm')
pf.mean_bin_plot(xvals, yo, 5, 1, 1, 'k')
plt.xlabel(fr'$C_{{ij}}^{{tr={tr[2]}}}$')  # <-- this line
plt.ylabel('$W_{i,j}$')
plt.plot([], [], 'm', label='Post')
plt.plot([], [], 'k', label='Pre')
plt.legend(loc='lower right', frameon=False)

# Bottom right: Post vs Pre (HI < 0)
plt.subplot(236)
tr = np.where((np.array(ptrl) < .01) & (np.array(b) < 0))[0]
xvals = X[inds[tr[2]], :].reshape(-1, 1)
pf.mean_bin_plot(xvals, y, 5, 1, 1, 'm')
pf.mean_bin_plot(xvals, yo, 5, 1, 1, 'k')
plt.xlabel(fr'$C_{{ij}}^{{tr={tr[2]}}}$')  # <-- this line
plt.ylabel('$W_{i,j}$')
plt.plot([], [], 'm', label='Post')
plt.plot([], [], 'k', label='Pre')
plt.legend(loc='lower right', frameon=False)

plt.tight_layout()

#%%
plt.figure(figsize = (5,3))
xrt = rt.copy();xrt[xrt>10] = 10
plt.plot(xrt,b,'k.')
plt.xlabel('Time to reward (s)')
plt.ylabel('Hebbian index (tr)')

plt.figure(figsize = (5,3))
xrt = rt.copy();xrt[xrt>10] = 10
plt.plot(rpe,b,'k.')
plt.xlabel('RPE (s)')
plt.ylabel('Hebbian index (tr)')

plt.show()
plt.figure(figsize = (8,6))
plt.subplot(211)
plt.plot(trial_bins, np.asarray(b) / max(np.abs(np.asarray(b)))*10, 'k', label='HI')
plt.plot(72,np.asarray(b[72]) / max(np.abs(np.asarray(b)))*10,'g.',markersize = 20)
plt.plot(110,np.asarray(b[110]) / max(np.abs(np.asarray(b)))*10,'b.',markersize = 20)
plt.ylabel('Hebbian index (tr)')
plt.title(mouse + ' ' + session)
plt.subplot(212)
plt.plot(trial_bins, rt_bin, 'r', label='rew time')
plt.plot(trial_bins, rpe_bin, 'b', label='RPE')
plt.ylabel('Time (s)')
plt.legend()
plt.xlabel('Trial #')
plt.show()

#%%
plt.figure(figsize = (10,6))
plt.subplot(221)
xrt = rt.copy();xrt[xrt>10] = 10
plt.plot(xrt[15:40],b[15:40],'k.')
plt.xlabel('Time to reward (s)')
plt.ylabel('Hebbian index (tr)')
plt.subplot(223)
xrt = rt.copy();xrt[xrt>10] = 10
i = 31
plt.plot(xrt[0+i:10+i],b[0+i:10+i],'k.')
plt.xlabel('Time to reward (s)')
plt.ylabel('Hebbian index (tr)')
#plt.plot(trial_bins[:-window], corr_rpe,'k')
plt.subplot(122)
plt.plot(trial_bins[:-window], corr_rpe,'k');
plt.plot(trial_bins[:-window],(thr_window-np.nanmean(thr_window))/np.nanmean(thr_window))
plt.xlabel('Trial #')
plt.ylabel('HI corr with RPE')

plt.tight_layout()
#%%
for i in range(40):
   plt.plot(rpe[0+i:10+i],b[0+i:10+i],'k.')
   plt.title(str(i))
   plt.show()

#%%

i = 0;
plt.plot()
