# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:30:46 2023

@author: scanimage
"""
import os
rootDir = r'C:/Users/scanimage/Documents/Python Scripts/BCI_analysis'
os.chdir(rootDir)

import scipy.io as spio
import numpy as np
import extract_scanimage_metadata
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import data_dict_create_module as ddc
folder = r'D:/KD/BCI_data/BCI_2022/BCI54/'

folders = [name for name in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, name))]

I = 0
data = dict()
for fi in range(2,4):
    #var_name = r'data' + folders[fi]
    var_name = r'data' + str(I)
    data[I] = ddc.load_data_dict(folder+folders[fi]+r'/')
    I = I + 1;
    #locals()[var_name] = data
iscell = data[0]['iscell']
cl_ind = np.where(iscell[:,0]==1)[0]

#%%
N = data[0]['F'].shape[1]
ntrl = data[0]['favg'].shape[2]
nt = data[0]['favg'].shape[0]
amps = np.zeros((N,ntrl,len(data)))
for si in range(len(data)):
    favg = data[si]['favg']    
    amp = np.nanmean(favg[9:15,:,:],axis = 0) - np.mean(favg[1:5,:,:],axis = 0)
    amps[:,:,si] = amp

stimDist = data[0]['stimDist'][cl_ind,:]
amps = amps[cl_ind,:,:]
gi = 89;
cl = np.where((stimDist[:,gi]>0) & (stimDist[:,gi]<100050))
plt.plot(amps[cl,gi,0],amps[cl,gi,1],'ko',markerfacecolor = 'w')

#%%
cn = data[1]['conditioned_neuron'][0][0]
fold = data[0]['F'];fold = np.nanmean(fold,axis = 2)
fnew = data[1]['F'];fnew = np.nanmean(fnew,axis = 2)
for i in range(N):
    bl = np.nanmean(fold[0:19,i])
    fold[:,i] = fold[:,i] - bl
    bl = np.nanmean(fnew[0:19,i])
    fnew[:,i] = fnew[:,i] - bl
plt.subplot(1,2,1)
plt.plot(fnew[:,cn])
plt.plot(fold[:,cn])
plt.subplot(1,2,2)
dist = data[1]['dist']
delt = np.nanmean(fnew[39:100,:],axis = 0) - np.nanmean(fold[39:100,:],axis = 0)
delt = delt[cl_ind]
plt.plot(dist[cl_ind],delt,'ko',markerfacecolor = 'w')
plt.plot(dist[cn],delt[cn],'ko',markerfacecolor = 'm')
plt.show()

#%%
ind = np.where((stimDist[cn,:]>30)&(stimDist[cn,:]<250))[0]
plt.plot(np.mean(data[0]['favg'][:,cn,ind],axis = 1))
plt.plot(np.mean(data[1]['favg'][:,cn,ind],axis = 1))
plt.show()
#%%
numGrps = favg.shape[2]
x = np.zeros((numGrps,2))
y = np.zeros((numGrps,2))
tuno = np.nanmean(fold[39:100,:],axis = 0);
tun = np.nanmean(fnew[39:100,:],axis = 0);
tc = data[0]['trace_corr']
#delt = tc[cn,:]
for di in range(2):
    for i in range(numGrps):
        near = np.where(stimDist[:,i]<20)
        far = np.where((stimDist[:,i]>30)&(stimDist[:,i]<1000))
        x[i,di] = np.dot(amps[near,i,di],delt[near])
        y[i,di] = np.dot(amps[far,i,di],delt[far])
plt.scatter(x[:,0]+x[:,1],y[:,1]-y[:,0])        
a=np.corrcoef(x[:,0]+x[:,1],y[:,1]-y[:,0])
print(a)
plt.show()
#%% connection vs correlation
import scipy.stats as stats
day = 1
N = amps.shape[0]
r = np.zeros((N,))
c = 2;
for cl in range(N):
    X = np.zeros((100,))
    for gi in range(100):
        cells = np.where(stimDist[:,gi]<10)[0]
        cc = data[1]['trace_corr'][cl_ind,cl]-data[0]['trace_corr'][cl_ind,cl]
        cc = cc[cells]
        s = amps[cells,gi,day]    
        X[gi] = np.dot(cc,s)
        X[gi] = np.mean(cc)
    Y = amps[cl,:,day]-amps[cl,:,0]
    ind = np.where(stimDist[cl,:]>30)
    #plt.plot(X[ind],Y[ind],'ko',markerfacecolor='w')
    r[cl], p_value = stats.pearsonr(X[ind].T, Y[ind].T)
    plt.show()
    #print(r[cl])
#%%
import scipy.stats as stats
day = 1
r = np.zeros((N,))
F = data[day]['F'][:,cl_ind,:]
cll = cn;
for cl in range(cll,cll+1):
    X = np.zeros((100,))
    for gi in range(100):
        cells = np.where(stimDist[:,gi]<10)[0]
        cc = data[1]['trace_corr'][cl_ind,cl]-data[0]['trace_corr'][cl_ind,cl]
        cc = cc[cells]
        s = amps[cells,gi,day]    
        X[gi] = np.dot(cc,s)
        X[gi] = np.mean(cc)
    Y = amps[cl,:,day]-amps[cl,:,0]
    ind = np.where(stimDist[cl,:]>30)
    plt.plot(X[ind],Y[ind],'ko',markerfacecolor='w')
    r[cl], p_value = stats.pearsonr(X[ind].T, Y[ind].T)
    plt.show()
    print(r[cl])
#%%
day = 1
gi = 63
seq = data[day]['seq']-1
cells = np.where(stimDist[:,gi]<10)[0]
#cells = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<1000))[0]
f = data[day]['Fstim'][:,:,seq==gi]
f = f[:,cl_ind,:]
f = f[1:20,cells,:]

plt.subplot(3,1,1)
plt.plot(np.mean(f,axis=1),'k',linewidth=1)
plt.plot(np.mean(np.mean(f,axis=1),axis=1),'k',linewidth=1)

plt.subplot(3,1,2)
plt.plot(stimDist[:,gi],amps[:,gi,day],'ko',markerfacecolor = 'w',markersize = 2)

plt.subplot(3,1,3)
cells = np.where(stimDist[:,gi]<10)[0]
cc = data[day]['trace_corr'][cells,:]
s = amps[cells,gi,day] 
x = np.dot(cc.T,s) 
x =np.mean(data[0]['trace_corr'][cells,:],axis = 0)
far = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<100))[0]
plt.plot(x[far],amps[far,gi,day],'ko',markerfacecolor = 'w',markersize = 2)
r, p_value = stats.pearsonr(x[far],amps[far,gi,day])
print(r)
xl = plt.xlim()
plt.plot(xl,(0,0),'k:')
plt.show()
#%%
N = amps.shape[0]
r = np.zeros((N,))
dcc = np.zeros((N,))
for cl in range(N):
    ind = np.where((stimDist[cl,:]>30)&(stimDist[cl,:]<100))[0]
    cc = data[1]['trace_corr'][cl,:]-data[0]['trace_corr'][cl,:]
    dcc[cl] = np.sum(cc**2)
    r[cl], p_value = stats.spearmanr(amps[cl,ind,1],amps[cl,ind,0])
plt.scatter(np.abs(delt),r)
rsq = stats.pearsonr(np.abs(delt),r)
print(rsq)
#%%
from scipy.stats import ttest_ind
import scipy.stats as stats

day = 1
num_far = np.zeros((100,2))
num_inh = np.zeros((100,2))
num_ex = np.zeros((100,2))
amp_near = np.zeros((100,2))
cn_corr = np.zeros((100,1))
N = amps.shape[0]
pval = np.zeros((N,100,2))
for day in range(2):
    for gi in range(100):
        seq = data[day]['seq']-1
        cells = np.where((stimDist[:,gi]<20))[0]
        cn_corr[gi] = np.mean(data[0]['trace_corr'][cells,cn])
        f = data[day]['Fstim'][:,:,seq==gi]
        f = f[:,cl_ind,:]
        bef = np.mean(f[1:5,:,:],axis =0)
        aft = np.mean(f[10:15,:,:],axis =0)
        t_statistic, p_value = ttest_ind(bef.T,aft.T)
        t_statistic, p_inh = ttest_ind(bef.T,aft.T,alternative = 'less')
        t_statistic, p_ex = ttest_ind(bef.T,aft.T,alternative = 'greater')
        pval[:,gi,day] = p_value
        ind = np.where(stimDist[:,gi]>30)[0]
        num_far[gi,day]=(sum(p_value[ind]<.05))
        num_inh[gi,day]=sum(p_inh[ind]<.05)
        num_ex[gi,day]=sum(p_ex[ind]<.05)
        ind = np.where(stimDist[:,gi]<20)[0]
        amp_near[gi,day] = np.mean(amps[ind,gi,day],axis = 0)
plt.subplot(1,2,1)
plt.plot(cn_corr,amp_near[:,1]-amp_near[:,0],'.')
plt.subplot(1,2,2)
plt.plot(cn_corr,num_ex[:,1]-num_ex[:,0],'.')
a=stats.pearsonr(cn_corr[:,0],num_ex[:,1]-num_ex[:,0])
print(a)

num_in = np.zeros((N,2))
num_out = np.zeros((N,2))
amp_in = np.zeros((N,2))
amp_out = np.zeros((N,2))
for ci in range(N):
    ind = np.where(stimDist[ci,:]>30)[0]
    num_in[ci,:] = np.mean(pval[ci,ind,:],axis=0)
    amp_in[ci,:] = np.mean(amps[ci,ind,:],axis=0)
    ind = np.where(stimDist[ci,:]<30)[0]
    num_out[ci,:] = np.mean(pval[ci,ind,:],axis=0)
    amp_out[ci,:] = np.mean(amps[ci,ind,:],axis=0)