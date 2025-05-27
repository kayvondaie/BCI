# -*- coding: utf-8 -*-
"""
Created on Tue May 27 00:25:01 2025

@author: kayvon.daie
"""

b = np.array([np.nanmean(hi[:-int(np.round(len(hi)/2))]/np.std(hi[0:])) for hi in HIb])
a = np.array([np.nanmean(hit[0:15]) for hit in RT])
pf.mean_bin_plot(a,b,5,1,1,'k')
ind = np.where(np.isnan(b+a)==0)[0]
pearsonr(a[ind],b[ind])
plt.xlabel('RT (s) session start')
plt.ylabel('HI session start')
#%%
a = np.array([np.nanmean(hit[0:10]) for hit in RT])
ind = np.where(np.isnan(b+a)==0)[0]
pf.mean_bin_plot(a[ind],np.array(CC_RPE)[ind],5,1,1,'k')
pearsonr(a[ind],np.array(CC_RPE)[ind])