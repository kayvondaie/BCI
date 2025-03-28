from scipy import stats
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import plotting_functions as pf
import scipy
import io
try:
    import mat73
except:
    !pip install mat73
    import mat73
 
mypath2 = 'H:/My Drive/Learning rules/BCI_data/combined_new_old_060524.mat' 
data_dict = mat73.loadmat(mypath2)
#data = data_combined['data']
#%%
import numpy as np

def mask_prev_trial_decay_fun(data, indd, trialsback):
    # Check if 'indd' is a list (cell in MATLAB)
    if isinstance(indd, list):
        indo = indd[1]
        indd = indd[0]
    else:
        indo = indd - 1

    seq = data['seq'][indd].copy()
    Fstim = data['Fstim'][indd].copy()
    GRP = data['GRP'][indd].astype(int)

    seqo = data['seq'][indo].copy()
    Fstimo = data['Fstim'][indo].copy()
    GRPo = data['GRP'][indo].astype(int)

    xo = data['x'][indo].copy()
    yo = data['y'][indo].copy()
    x = data['x'][indd].copy()
    y = data['y'][indd].copy()

    # Convert as in the original code
    num = int(np.max(GRP))
    # Y = np.reshape(y, (-1, num))
    # X = np.reshape(x, (-1, num))
    # Yo = np.reshape(yo, (-1, num))
    # Xo = np.reshape(xo, (-1, num))
    
    X = np.reshape(x, (-1, num), order='F')
    Y = np.reshape(y, (-1, num), order='F')
    Xo = np.reshape(xo, (-1, num), order='F')
    Yo = np.reshape(yo, (-1, num), order='F')

    

    # Iterate twice as in MATLAB (iter = 1:2)
    for iter_val in [1, 2]:
        if iter_val == 1:
            fs = Fstim.copy()
            sq = seq.copy()
        else:
            fs = Fstimo.copy()
            sq = seqo.copy()

        # Loop over trials as in MATLAB (ti = 2:size(fs,3))
        for ti in range(1, fs.shape[2]):
            grp = int(sq[ti-1])
            tb = trialsback
            if ti < trialsback + 1:
                tb = ti

            clss = np.where(X[:, int(sq[ti-1]) - 1] < 30)[0]
            fs[:, clss, ti] = np.nan

        amp = np.zeros((fs.shape[1], num))  # Initialize amp with the correct dimensions
        for si in range(1, num+1):
            ind = np.where(sq == si)[0]
            fff = np.nanmean(fs[:, :, ind], axis=2)
            baseline = np.nanmean(fff[0:4, :], axis=0)  # (1:4 in MATLAB)
            response = np.nanmean(fff[9:16, :], axis=0)  # (10:16 in MATLAB)
            amp[:, si-1] = (response - baseline) / baseline


        if iter_val == 1:
            Y = amp
            Fstim = fs
        else:
            Yo = amp
            Fstimo = fs

    return Y, Yo
#%%
I = 0
maxtrials = 40
near = 30
far = 100
direct = 30
delT = []
WDT = []
DWT = []
DWDT = []
CC = []
sess = []
prefx1 = []
prefx2 = []
prefy1 = []
prefy2 = []
in3 = []
ino3 = []
out3 = []
outo3 = []
ex = []
tune = []
tind = np.arange(40,240)
before = np.arange(20)
mouse = data_dict['data']['mouse']
#for di in range(len(data_dict['data']['F'])-1):
for di in range(2,len(data_dict['data'])):
    
    current_idx = None
    past_idx = None
    xo, yo, x, y = None, None, None, None
    F, Fo, seq, seqo = None, None, None, None
    GRP, GRPo = None, None
    in2, out2, ino2, outo2 = [], [], [], []
    
    current_idx = di
    past_idx = di-1
    #print(di)
    if (mouse[current_idx] == mouse[past_idx]) and (data_dict['data']['x'][past_idx] is not None):
        xo = data_dict['data']['x'][past_idx].copy()
        yo = data_dict['data']['y'][past_idx].copy()
        x = data_dict['data']['x'][current_idx].copy()
        y = data_dict['data']['y'][current_idx].copy()
        F = data_dict['data']['F'][current_idx].copy()
        Fo = data_dict['data']['F'][past_idx].copy()
        Fstim = data_dict['data']['Fstim'][current_idx].copy()
        Fstimo = data_dict['data']['Fstim'][past_idx].copy()
        seq = data_dict['data']['seq'][current_idx].copy()
        seqo = data_dict['data']['seq'][past_idx].copy()

        f = np.nanmean(F[:, :, 0:maxtrials], axis = 2)
        f = f - np.nanmean(f[before, :], axis = 0)[None, :]
        fo = np.nanmean(Fo[:, :, 0:], axis = 2)
        fo = fo - np.nanmean(fo[before, :], axis = 0)[None, :]
        fs = np.nanmean(f[tind,:], axis = 0)
        fso = np.nanmean(fo[tind,:], axis = 0)

        #plt.plot(f)
        #plt.show()

        #***Y, Y0 = mask_previous_trial_decay()
        if f.shape[0] == fo.shape[0]:
            cn = int(data_dict['data']['conditioned_neuron'][current_idx]-1)
            GRP = data_dict['data']['GRP'][current_idx].copy()
            GRPo = data_dict['data']['GRP'][past_idx].copy()
            corr0 = data_dict['data']['trace_corr'][past_idx].copy()
            corr1 = data_dict['data']['trace_corr'][current_idx].copy()
            I+=1
            sess.append(I)
            num = int(np.max(GRP))
            try:
                """
                X = x.reshape([-1,num])
                Xo = xo.reshape([-1,num])
                Y = y.reshape([-1,num])
                Yo = yo.reshape([-1,num])
                """
                Y,Yo = mask_prev_trial_decay_fun(data_dict['data'],current_idx,trialsback=1)
                #X = x.reshape([num, -1]).T
                #Xo = xo.reshape([num, -1]).T
                Xo = np.reshape(xo, (-1, num), order='F')
                X = np.reshape(x, (-1, num), order='F')
                #Y = y.reshape([num, -1]).T
                #Yo = yo.reshape([num, -1]).T
                #Y = a[1]
                #Yo = a[0]

                #plt.imshow(Y, cmap='seismic', vmin = -1, vmax = 1)
                #plt.show()
            except:
                print('skip')
                continue
            delt = fs-fso
            vcd = delt / np.linalg.norm(delt)
            
            import numpy as np
            
            # Initialize lists for each structure
            in2 = []
            ino2 = []
            out2 = []
            outo2 = []
            
            # Assuming `I` is a loop variable or index and the loops handle it correctly
            for gi in range(Y.shape[1]):
                ind = np.where((X[:, gi] > near) & (X[:, gi] < far))[0]
                in2.append(np.dot(vcd[ind],Y[ind, gi]))
                ino2.append(np.dot(vcd[ind],Yo[ind, gi]))
            
                ind = np.where(X[:, gi] < direct)[0]
                out2.append(np.dot(vcd[ind], Y[ind, gi]))
                outo2.append(np.dot(vcd[ind], Yo[ind, gi]))
            in3.append(np.asarray(in2))
            ino3.append(np.asarray(ino2))
            out3.append(np.asarray(out2))
            outo3.append(np.asarray(outo2))
#%%            
ind = list(range(0, 6)) + [7, 8] + list(range(10, 13))
a = [out3[i] for i in ind]
ao = [outo3[i] for i in ind]
b = [in3[i] for i in ind]
bo = [ino3[i] for i in ind]
a = np.asarray(a).flatten();
ao = np.asarray(ao).flatten();
b = np.asarray(b).flatten();
bo = np.asarray(bo).flatten();
pf.mean_bin_plot(a + ao,b - bo,6,1,1,'k')

