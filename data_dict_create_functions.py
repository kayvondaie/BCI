# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:10:45 2023

@author: scanimage
"""
import numpy as np
def create_BCI_F(Ftrace,ops):
    F_trial_strt = [];
    strt = 0;
    dff = 0*F
    for i in range(np.shape(F)[0]):
        #bl = np.percentile(F[i,:],50)
        bl = np.std(Ftrace[i,:])
        dff[i,:] = (Ftrace[i,:] - bl)/bl
    for i in range(len(ops['frames_per_file'])):
        ind = list(range(strt,strt+ops['frames_per_file'][i]))    
        f = dff[:,ind]
        F_trial_strt.append(f)
        strt = ind[-1]+1
        

    f_first_ten = np.full((240,np.shape(F)[0],len(ops['frames_per_file'])),np.nan)
    pre = np.full((np.shape(F)[0],40),np.nan)
    for i in range(len(ops['frames_per_file'])):
        f = F_trial_strt[i]
        if i > 0:
            pre = F_trial_strt[i-1][:,-40:]
        pad = np.full((np.shape(F)[0],200),np.nan)
        f = np.concatenate((pre,f),axis = 1)
        f = np.concatenate((f,pad),axis = 1)
        f = f[:,0:240]
        f_first_ten[:,:,i] = np.transpose(f)
    return F, Fraw