

from scipy.signal import medfilt, correlate
from axon_helper_module import *
import bci_time_series as bts

processing_mode = 'all'
si = 6
inds = np.arange(len(sessions)) if processing_mode == 'all' else np.arange(si, si + 1)
XCORR, LAGS, SESSION, LOW, HIGH = [], [], [], [], []
num = 1000
plot = 1
I = 0
time = np.arange(0,2000)*data['dt_si']
plt.figure(figsize=(3,3))
for i in inds:
    try:
        print(i)
        mouse = sessions['Mouse'][i]
        session = sessions['Session'][i]
    
        # Load data
        try:
            folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
            main_npy_filename = f"data_main_{mouse}_{session}_BCI.npy"
            data = np.load(os.path.join(folder, main_npy_filename), allow_pickle=True)
        except:
            folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/'
            main_npy_filename = f"data_main_{mouse}_{session}_BCI.npy"
            data = np.load(os.path.join(folder, main_npy_filename), allow_pickle=True)
    
        # Timing and behavioral signals
        dt_si = data['dt_si']
        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        rt[np.isnan(rt)] = 20
        dfaxon = data['ch1']['df_closedloop']
    
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)
      # Compute RPE
        rpe = compute_rpe(rt == 20, baseline=1, window=20, fill_value=50)
        df = data['ch1']['df_closedloop']
        df = np.nanmean(df,0);
        ind = np.where(trial_start_vector == 1)[0]
        g = np.zeros((7000,len(ind)))
        ts = np.zeros((7000,len(ind)))
        for ii in range(len(ind)-1):
            a = df[ind[ii]:ind[ii+1]+60]
            g[0:len(a),ii] = a;
            ts[0:len(a),ii] = trial_start_vector[ind[ii]:ind[ii+1]+60]

        miss = np.where(rt == 20)[0]
        a = np.nanmean(g[0:290,miss],1)
        bl = np.nanmedian(df)
        a = a-bl
        plt.plot(time[0:290]-10,a,'b',label = 'Miss')
        plt.plot(time[0:290]-10,np.nanmean(ts[0:290,miss],1)/10,'k')
        plt.xlim((-2,4))
        rta = reward_aligned_responses(dfaxon, reward_vector, dt_si, window=(-2, 4))
        n_timepoints = mean_rta.shape[0]
        time = np.linspace(-2, 4, n_timepoints)
        b = np.nanmean(np.nanmean(rta,2),1) 
        b = b - bl
        plt.plot(time,b,'r',label = 'Hit')
        plt.xlabel('Time from (expected) reward (s)')
        plt.ylabel('Avg. axons')
        plt.legend()
        HIGH.append(b);
        LOW.append(a)

    except:
        print('error: {e}')
        continue