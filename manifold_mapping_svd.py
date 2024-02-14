# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:24:49 2023

@author: scanimage
"""
import numpy as np
import data_dict_create_module as ddc
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
mpl.rcParams['figure.dpi'] = 300
data = ddc.main(folder)
folder = r'D:/KD/BCI_data/BCI_2022/BCI54/072423/'
#%%
df = data['df_closedloop'].T
df = np.nan_to_num(df, nan=0)
iscell = data['iscell']
numCells = 100;

ind = np.where(iscell[:,0])[0]
ind = ind[0:numCells]


F = data['F'][:,ind,:]
f = np.nanmean(F,axis = 2)
for i in range(f.shape[1]):
    bl = np.nanmean(f[1:20,i])
    f[:,i] = f[:,i] - bl
f = np.nan_to_num(f, nan=0)

df = df[:,ind];
df = np.nan_to_num(df, nan=0)
U,S,vec = np.linalg.svd(df, full_matrices=False)
vec = vec.T
#%%
from scipy.linalg import orth
plt.rcParams.update({'font.size': 8})  # Change the value to your desired font size
fig, axes = plt.subplots(10,2)

U,S,bci = np.linalg.svd(f, full_matrices=False)
bci = bci.T
nums = 4
v2 = vec*0
for i in range(vec.shape[1]):
    a = np.dot(vec[:,i],bci[:,0:nums])
    a = vec[:,i] - (np.matmul(bci[:,0:nums],a))        
    v2[:,i] = a    
#v2 = vec
for num in range(10):
    ax = axes[num,0]
    proj = np.matmul(f,v2[:,num])
    ax.plot(proj,'k',linewidth = .5)
    
    ax = axes[num,1]
    proj = np.matmul(df[35000:45000,:],v2[:,num])
    ax.plot(proj,'k',linewidth = .2)
plt.show()
np.save(folder + r'manifolds_'+data['mouse']+r'_'+data['session']+r'.npy',v2,ind)
#%%
import pandas as pd

csv_data = pd.read_csv('D:/KD/BCI_data/BCI_2022/BCI54/072023/manifold1_again_IntegrationRois_00001.csv')
csv_data = csv_data.values
proj = np.matmul(csv_data[:,2:102],vec[:,1]);
#plt.subplot(1,2,1)
#plt.plot(csv_data[:,0],proj,'k',linewidth=.3)
csv_dff = np.zeros((csv_data.shape[0], csv_data.shape[1]))
for i in range(csv_data.shape[1]):
    a = csv_data[:,i]
    bl = np.percentile(a,50)
    csv_dff[:,i] = (a-bl)/bl
#plt.subplot(1,2,2)
projn = np.matmul(csv_dff[:,2:102],vec[:,1]);
plt.plot(csv_data[:,0],projn,'k',linewidth=.3)    

#%%
import folder_props_fun
import os
import tifffile
folder_props = folder_props_fun.folder_props_fun(folder)
bases = folder_props['bases']
print(bases)
ind = input('pick indices of bases for BCI, photostim and spont in that order')
ind = np.fromstring(ind[1:-1], sep=',')
for ei in range(0,len(ind)):
    #base = {base for i, base in enumerate(bases) if str(i) in ind}
    base = bases[int(ind[ei])]
    siFiles = folder_props['siFiles']
    files = os.listdir(folder)
    good = np.zeros([1,np.shape(files)[0]])
    for fi in range(0,np.shape(files)[0]):
        str = files[fi]
        a = str.find('.tif')
        if a > -1:
            #b = str.find('_')
            #b2 = str.find('_',b+1)
            #b = max([b,b2]);
            b = max([i for i, char in enumerate(str) if char == '_'])
            b = str[0:b]
            if b == base:
                good[0][fi] = 1
    #        if b == base2:
    #            good[0][fi] = 1
    
    good = np.where(good == 1)
    good = good[1]
files = [files[i] for i in good]
#%%
num = np.zeros([1,len(files)])
for i in range(len(files)):
    file = folder + files[i]    
    with tifffile.TiffFile(file) as tiff:
        num_frames = len(tiff.pages)
    num[0][i] = num_frames    
num = num[0]
#%%
proj = np.matmul(csv_data[:,2:102],vec[:,1]);
dfff = np.full((6000, len(num)), np.nan)
for i in range(len(num)):
    ind = np.arange(0,int(num[i]))
    a = proj[0:int(num[i])]
    proj = np.delete(proj,ind,axis=0)
    dfff[0:len(a),i] = a
dfff = np.nanmean(dfff,axis = 1)   