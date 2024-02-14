import numpy as np
import matplotlib.pyplot as plt
import plotting_functions as pf
import pandas as pd

folder = r'H:/My Drive/Learning rules/BCI_data/dataset ai229 & ai228/'

# Load the CSV file
df = pd.read_csv(folder + r'/metadata.csv')

# Custom function to determine if a row should be 1 or 0 based on your conditions
def custom_filter(row):
    # Handle the first row since there's no row i-1 for it
    if row.name == 0:
        return 0
    prev_row = df.iloc[row.name - 1]
    
    # Check conditions
    if (pd.isna(row['BCI_2 CN']) and
            row['mouse'] == prev_row['mouse'] and
            not pd.isna(row['photostim group number']) and
            not pd.isna(prev_row['photostim group number'])):  # Additional condition
        return 1
    return 0

# Apply the custom function row-wise
df['custom_value'] = df.apply(custom_filter, axis=1)
satisfying_rows_indices = df[df['custom_value'] == 1].index.values
hebb_index = []
cn_tuning = []
vel_cor = []
delta_rew = []
for di in range(len(satisfying_rows_indices)):
    
    # Filter rows where custom_value is 1
    filtered_rows = df[df['custom_value'] == 1]
    
    file = df['filename'].iloc[satisfying_rows_indices[di]]
    old_file = df['filename'].iloc[satisfying_rows_indices[di]-1]
    data = np.load(folder+file,allow_pickle=True).tolist()
    old = np.load(folder+old_file,allow_pickle=True).tolist()
    
    favg = data['photostim']['favg']
    favgo = old['photostim']['favg']
    X = data['photostim']['stimDist']
    Xo = old['photostim']['stimDist']
    f = np.nanmean(data['BCI_1']['F'],axis = 2);
    fo = np.nanmean(old['BCI_1']['F'],axis = 2);
    FO = old['BCI_1']['F'];
    ko = np.nanmean(FO[20:80,:,:],axis = 0)
    cco = np.corrcoef(ko)
    cco = old['BCI_1']['trace_corr']
    for i in range(f.shape[1]):    
        f[:,i] = f[:,i] - np.nanmean(f[0:5,i])
        fo[:,i] = fo[:,i] - np.nanmean(fo[0:5,i])
    
    delt = np.nanmean(f[39:-1,:],axis = 0) - np.nanmean(fo[39:-1,:],axis = 0)
    
    Y = np.nanmean(favg[9:15,:,:],axis = 0)
    Yo = np.nanmean(favgo[9:15,:,:],axis = 0)
    
    stm = np.dot(cco,Y*(X<30))
    stmo = np.dot(cco,Yo*(Xo<30))
    
    eff = Y*(X>30)*(X<10000)
    effo = Yo*(Xo>30)*(Xo<1000)
    # pf.mean_bin_plot(X.flatten(),Y.flatten(),11,1,1,'k')
    # plt.show()
        
    vel = old['BCI_1']['step_time']
    cn = data['BCI_1']['conditioned_neuron']

    dfcn = old['BCI_1']['df_closedloop'][cn,:].T
    vel_cor.append(np.corrcoef(vel,dfcn)[0][1])
    #cn_tuning.append(np.nanmean(fo[20:100,cn])) 
    cn_tuning.append(np.nanmean(cco[:,cn]))
    a = (stm+stmo).flatten()
    b = (eff-effo).flatten()
    ind = np.where(np.abs(b)<10)[0]
    c = np.corrcoef(a[ind],b[ind])[0][1]
    hebb_index.append(c)
    
    
    rew_rate = np.mean(data['BCI_1']['reward_time'][0:15000])
    rew_rateo = np.mean(old['BCI_1']['reward_time'][0:15000])
    delta_rew.append(rew_rate-rew_rateo)    

    plt.subplot(121)
    plt.plot(fo[:,cn])
    plt.subplot(122)
    aa,bb = pf.mean_bin_plot(a[ind],b[ind],11,1,1,'k')
    hebb_index.append(bb[-1])
    plt.show()

ind = [0,1,4,5,6,7,8]
plt.scatter(np.asarray(cn_tuning)[ind],np.asarray(hebb_index)[ind])
np.corrcoef(np.asarray(cn_tuning)[ind],np.asarray(hebb_index)[ind])[0][1]
#plt.scatter(np.asarray(vel_cor)[ind],np.asarray(hebb_index)[ind])
#plt.scatter(np.asarray(delta_rew)[ind],np.asarray(hebb_index)[ind])
#np.corrcoef(np.asarray(delta_rew)[ind],np.asarray(hebb_index)[ind])[0][1]