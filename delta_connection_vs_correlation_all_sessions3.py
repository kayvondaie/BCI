# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:58:35 2024

@author: kayvon.daie
"""

from scipy.io import loadmat
import h5py

data_file = 'H:/My Drive/Learning rules/BCI_data/combined_new_old_090622.mat'
mat_data = h5py.File(data_file, 'r')
data = mat_data['data']

data_file = 'H:/My Drive/Learning rules/BCI_data/combined_behavior.mat'
mat_data = h5py.File(data_file, 'r')
data_behav = mat_data['data_behav']

def keep(*variables_to_keep):
    current_variables = locals()

    existing_variables = set(current_variables.keys())

    for var_name in variables_to_keep:
        if var_name in existing_variables:
            del current_variables[var_name]

#%%
import numpy as np
import matplotlib.pyplot as plt

# Define necessary data structures and variables here
I = 0
plt.figure(400)
# Define figure_initialize function here
plt.figure(figsize=(8, 3))
plt.gca().set_position([-10, 4, 8, 3])

for di in range(2, len(data)):
    variables_to_keep = ['I', 'di', 'data', 'data_behav', 'BO', 'STM', 'STMO', 'EFF', 'EFFO', 'AO', 'del_var', 'ccx', 'CC', 'tun_co', 'avg_cn_corr']
    keep(*variables_to_keep)
    
    indd = di
    indo = indd - 1
    
    if data[indd]['mouse'] == data[indo]['mouse'] and data[indd]['x'] is not None and data[indo]['x'] is not None:
        xo = data[indo]['x']
        yo = data[indo]['y']
        x = data[indd]['x']
        y = data[indd]['y']
        GRP = data[indd]['GRP']
        num = max(GRP)
        Y = np.reshape(y, (-1, num))
        X = np.reshape(x, (-1, num))
        Yo = np.reshape(yo, (-1, num))
        Xo = np.reshape(xo, (-1, num))
        
        if X.shape[0] == Xo.shape[0] and np.corrcoef(X.flatten(), Xo.flatten())[0, 1] > 0.8:
            F = data[indd]['F']
            Fo = data[indo]['F']
            f = np.nanmean(F[0:240, :, 0:], axis=2)
            f = f - np.tile(np.mean(f[0:20, :], axis=0), (f.shape[0], 1))
            fo = np.nanmean(Fo[0:240, :, 0:], axis=2)
            fo = fo - np.tile(np.mean(fo[0:20, :], axis=0), (fo.shape[0], 1))
            Frewo = data_behav[indo]['F_rew']
            Frew = data_behav[indd]['F_rew']
            dist = data[indd]['dist']
            tsta = data[indd]['tsta']

            if f.shape[1] == fo.shape[1]:
                cn = data[indd]['conditioned_neuron']
                GRP = data[indd]['GRP']
                GRPo = data[indo]['GRP']

                I += 1
                num = max(GRP)
                Y = np.reshape(y, (-1, num))
                X = np.reshape(x, (-1, num))
                Yo = np.reshape(yo, (-1, num))
                Xo = np.reshape(xo, (-1, num))
                ccx[I] = np.corrcoef(X.flatten(), Xo.flatten())[0, 1]

                near = 30
                far = 1000
                direct = 30
                tind = range(40, 151)
                a = np.argsort(data[indd]['dist'])
                b = data[indd]['dist'][a]
                del_var[I] = np.mean(f[tind, b]) - np.mean(fo[tind, b])
                dist = b
                cnn = np.where(b == cn)[0]
                ind = np.arange(X.shape[1])
                delCon[I] = ((Y[b, ind] * (X[b, ind] > 30) & (X[b, ind] < 8000)) - (Yo[b, ind] * (Xo[b, ind] > 30) & (Xo[b, ind] < 8000)))
                delConNear[I] = ((Y[b, ind] * (X[b, ind] > 30) & (X[b, ind] < 100)) - (Yo[b, ind] * (Xo[b, ind] > 30) & (Xo[b, ind] < 100)))
                delConFar[I] = ((Y[b, ind] * (X[b, ind] > 100) & (X[b, ind] < 1000)) - (Yo[b, ind] * (Xo[b, ind] > 100) & (Xo[b, ind] < 1000)))
                delConDir[I] = ((Y[b, ind] * (X[b, ind] > 0) & (X[b, ind] < 20)) - (Yo[b, ind] * (Xo[b, ind] > 0) & (Xo[b, ind] < 20)))

                delConFar[I][delConFar[I] == 0] = np.nan
                delConNear[I][delConNear[I] == 0] = np.nan
                delCon[I][delCon[I] == 0] = np.nan
                delConDir[I][delConDir[I] == 0] = np.nan
                tind = range(40, 80)
                k = np.nanmean(Fo[tind, :, 0:np.floor(end)], axis=2)
                cc = np.corrcoef(k.T)
                cor[I] = cc[:, cnn]

                marg = 0.5
                k = np.nanmean(F[tind, :, 0:np.floor(end)], axis=2)
                k[np.isnan(k)] = 0
                ko = np.nanmean(Fo[tind, :, 0:np.floor(end)], axis=2)
                cc = np.corrcoef(k.T)
                cco = np.corrcoef(ko.T)
                grps = np.unique(GRP)
                num = max(GRP)
                Y = np.reshape(y, (-1, num))
                X = np.reshape(x, (-1, num))
                Yo = np.reshape(yo, (-1, num))
                Xo = np.reshape(xo, (-1, num))

                direct = 30
                near = 30
                far = 2000
                cls = np.where(cc[:, cn] > -10000.15)
                grp = np.arange(1, X.shape[1] + 1)

                stm = (Y * (X < direct)).T @ cco
                eff = ((Y * (X > near) * (X < far)).T)
                stm = stm[grp, cls]
                eff = eff[grp, cls]

                stmo = (Yo * (Xo < direct)).T @ cco
                effo = ((Yo * (Xo > near) * (Xo < far)).T)
                stmo = stmo[grp, cls]
                effo = effo[grp, cls]

                row = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
                col = np.arange(1, 6).tolist() + np.arange(1, 6).tolist()

                ind = eff != 0
                ind = effo != 0 + eff != 0
                CC[I] = np.corrcoef(stmo[ind] + stm[ind], eff[ind] - effo[ind])
                tun_co[I] = np.mean(fo[40:80, cn])
                avg_cn_corr[I] = np.mean(cco[cn, :])
                rew = data_behav[indo]['rew']
                vel = data_behav[indo]['vel']
                dt_si = data_behav[indo]['dt_si']
                t = np.arange(0, dt_si * len(rew), dt_si)
                rew = np.convolve(np.exp(-t / 0.1), rew, mode='valid') * 0 + np.convolve(np.exp(-t / 1), vel, mode='valid') * 1
                rew = rew[:len(t)]
                avg_cn_corr[I] = np.corrcoef(rew, data_behav[indo]['df_closedLoop'][:, cn])[0, 1]

                STM.append(stm)
                STMO.append(stmo)
                EFF.append(eff)
                EFFO.append(effo)

                f_cn[:, :, I] = np.column_stack([fo[:, cn], f[:, cn]])
                plt.box(False)
                plt.title(data[di]['mouse'])

# Additional code for ttest, mean_bin_plot, scatter, and finalizing figures

plt.show()

# Plotting the last figure
plt.figure(1)
plt.clf()
plt.scatter(avg_cn_corr, CC, s=100, c='k')
plt.xlabel('Day 1 CN correlation with lickport', fontsize=12)
plt.ylabel('ρ_{ΔW_{i,j},ρ_{i,j}}', fontsize=18)
