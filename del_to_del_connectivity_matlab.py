from scipy import stats
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
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
ex = []
tune = []
tind = np.arange(40,240)
before = np.arange(40)
mouse = data_dict['data']['mouse']
for di in range(len(data_dict['data']['F'])-1):
    current_idx = di+1
    past_idx = di
    #print(di)
    if mouse[current_idx] == mouse[past_idx]:
        xo = data_dict['data']['x'][past_idx]
        yo = data_dict['data']['y'][past_idx]
        x = data_dict['data']['x'][current_idx]
        y = data_dict['data']['y'][current_idx]
        F = data_dict['data']['F'][current_idx]
        Fo = data_dict['data']['F'][past_idx]
        Fstim = data_dict['data']['Fstim'][current_idx]
        Fstimo = data_dict['data']['Fstim'][past_idx]
        seq = data_dict['data']['seq'][current_idx]
        seqo = data_dict['data']['seq'][past_idx]

        f = np.nanmean(F[:, :, 0:maxtrials], axis = 2)
        f = f - np.nanmean(f[before, :], axis = 0)[None, :]
        fo = np.nanmean(Fo[:, :, 0:maxtrials], axis = 2)
        fo = fo - np.nanmean(fo[before, :], axis = 0)[None, :]
        fs = np.nanmean(f[tind], axis = 0)
        fso = np.nanmean(fo[tind], axis = 0)

        #plt.plot(f)
        #plt.show()

        #***Y, Y0 = mask_previous_trial_decay()
        if f.shape[0] == fo.shape[0]:
            cn = int(data_dict['data']['conditioned_neuron'][current_idx]-1)
            GRP = data_dict['data']['GRP'][current_idx]
            GRPo = data_dict['data']['GRP'][past_idx]
            corr0 = data_dict['data']['trace_corr'][past_idx]
            corr1 = data_dict['data']['trace_corr'][current_idx]
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
                X = x.reshape([num, -1]).T
                Xo = xo.reshape([num, -1]).T
                Y = y.reshape([num, -1]).T
                Yo = yo.reshape([num, -1]).T

                #plt.imshow(Y, cmap='seismic', vmin = -1, vmax = 1)
                #plt.show()
            except:
                print('skip')
                continue
            delf = fs-fso
            tun = 0.5*(fso+fs)  
            Z1 = (Y*(X>near)*(X<far)).dot((Y*(X<direct)).T)
            Zo = (Yo*(X>near)*(X<far)).dot((Yo*(X<direct)).T)
            Z = Z1
            #print(Z.shape)
            wdt = Z.T.dot(delf)
            delT.append(delf);
            WDT.append(wdt)
            delZ = Z1-Zo
            dwt = delZ.T.dot(tun)
            #print(delZ, tun)
            DWT.append(dwt)

            #a = ((Y-Yo)*(X>near)*(X<far))
            #b = ((Y+Yo)*(X<direct))

            dwdt = delZ.dot(delf);
            DWDT.append(dwdt.ravel())

            Z1d = (Y).dot((Y*(X<direct)).T)
            Zod = (Yo).dot((Yo*(Xo<direct)).T)
            Zd = 0.5*(Z1d+Zod)
            prefx1.append(delf.dot(delZ))
            prefy1.append(np.diagonal(Zd)*delf)
            ex.append(np.diagonal(Z1d-Zod))

            CC.append(np.corrcoef(X.ravel(), Xo.ravel())[1,0])

ind = np.where(np.array(CC)>.6)[0]
cts = 0
for keep in ind:
    if cts == 0:
        cts+=1
        dwt = DWT[keep]
        dwdt = DWDT[keep]
        wdt = WDT[keep]
        dT = delT[keep]
        px = prefx1[keep]
        py = prefy1[keep]
        exc= ex[keep]
    else:
        cts+=1
        dwt = np.hstack([dwt, DWT[keep]])
        dwdt = np.hstack([dwdt,DWDT[keep]])
        wdt = np.hstack([wdt,WDT[keep]])
        dT = np.hstack([dT, delT[keep]])
        px = np.hstack([px, prefx1[keep]])
        py = np.hstack([py, prefy1[keep]])
        exc = np.hstack([exc, ex[keep]])


def plotup(X1,Y1, ax, name,color, keepscatter = False, equalcount = False, upperc = 90, lowperc = 10):
    if keepscatter:
        ax.scatter(X1, Y1, c='k', alpha = 0.1)
    res = stats.linregress(X1,Y1)
    Nbins = 10
    xlimUp = np.percentile(X1,upperc)
    xlimLow = np.percentile(X1,lowperc)
    if equalcount:
        percbin = np.linspace(lowperc, upperc,Nbins)
        bins = [np.percentile(X1, perc) for perc in percbin]

    else:
        bins = np.linspace(xlimLow,xlimUp, Nbins)
    #xbincenter = bins[1::]-np.diff(bins)[0]/2
    meaninbin = []
    stdmeaninbin = []
    countbin = []
    xbincenter = []
    for i in range(Nbins-1):
      meaninbin.append(np.mean(Y1[np.logical_and(X1>bins[i], X1<bins[i+1])]))
      stdmeaninbin.append(np.std(Y1[np.logical_and(X1>bins[i], X1<bins[i+1])]))
      countbin.append(np.sum(np.logical_and(X1>bins[i], X1<bins[i+1])))
      xbincenter.append(np.mean(X1[np.logical_and(X1 > bins[i], X1 < bins[i+1])]))
    ax.errorbar(xbincenter, meaninbin, np.array(stdmeaninbin)/np.sqrt(countbin), color=color, lw = 2, label = 'Pearson = ' + str(np.round(res.rvalue,3)))
    fitx = np.array(xbincenter).ravel()
    ax.plot(fitx, res.intercept + res.slope*fitx, color, label='slope = ' + str(np.round(res.slope, 3))+ ', log(pvalue) = '+str(np.round(np.log10(res.pvalue),1)))
    if name == 'Preferential':
        ax.set_ylabel('$\Delta T_{i} \cdot W_{ii}$')
        ax.set_xlabel('$\Delta T_i \cdot \Delta W$')
    else:
        ax.set_ylabel('$\Delta T_{i}$')
        ax.set_xlabel(name)
    #ax.set_ylim([-0.03, 0.03])
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc='best')


xs = [dwt,wdt,dwdt, exc, px]        
keeps = [np.where(np.isnan(X)==False)[0] for X in xs]
Xin = [xs[i][keeps[i]] for i in range(len(keeps))]
Yin = [dT[keeps[0]],dT[keeps[1]],dT[keeps[2]], dT[keeps[3]], py[keeps[4]]]
colors = ['g', 'b', 'r','m', 'k']
names = ['$\Delta W_{ij} \cdot T_{j}$','$W_{ij} \cdot \Delta T_{j}$','$\Delta W_{ij} \cdot \Delta T_{j}$', '$\Delta W_{ii}$','Preferential']

fig,ax = plt.subplots(1,5,figsize = [20,3])
for zzz in range(len(Xin)):
    plotup(Xin[zzz],Yin[zzz], ax[zzz], names[zzz], colors[zzz], keepscatter = False)
plt.tight_layout()
plt.show()

fig,ax = plt.subplots(1,5,figsize = [20,3])
for zzz in range(len(Xin)):
    plotup(Xin[zzz],Yin[zzz], ax[zzz], names[zzz], colors[zzz], keepscatter = False, equalcount = True)
plt.tight_layout()
plt.show()