# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:37:35 2024

@author: kayvon.daie
"""
import numpy as np
import pandas as pd

# Load the CSV file, skip the first 6 rows which might be metadata or headers
df = pd.read_csv(csvfilename, delimiter=';', skiprows=6)

# Remove rows where 'TYPE' or 'MSG' contains specific unwanted strings or are NaN
df = df[df['TYPE'].notna()]  # Keep rows where 'TYPE' is not NaN
df = df[df['TYPE'] != '|']  # Remove rows where 'TYPE' is '|'
df = df[df['MSG'].notna()]  # Keep rows where 'MSG' is not NaN
df = df[df['MSG'].str.strip() != '']  # Remove rows where 'MSG' is empty
df = df[df['MSG'] != '|']  # Remove rows where 'MSG' is '|'

# Reset the index after deletion
df = df.reset_index(drop=True)
#%%
import numpy as np

# Find indices of "Reward_L" and trial start
reward_l_indices = df.index[(df['MSG'] == 'Reward_L') & (df['TYPE'] == 'TRANSITION')].tolist()
trial_start_indices = df.index[df['TYPE'] == 'TRIAL'].tolist()
threshold_crossing_ind = df.index[(df['MSG'] == 'ResponseInRewardZone') & (df['TYPE'] == 'TRANSITION')].tolist()
threshold_crossing_time = df.loc[(df['MSG'] == 'ResponseInRewardZone') & (df['TYPE'] == 'TRANSITION'),'BPOD-INITIAL-TIME'].values#[0]#.index.to_numpy()[0]

# Convert the list of reward indices to a numpy array
rew_ind = np.array(reward_l_indices)

# Initialize a 1D array to store reward counts
rew = np.zeros(len(trial_start_indices))
thr = np.zeros(len(trial_start_indices))
# Iterate through the trial start indices to check for rewards within each trial
for i in range(len(trial_start_indices) - 1):
    # Find rewards between the current trial start and the next
    ind = np.where((rew_ind > trial_start_indices[i]) & (rew_ind < trial_start_indices[i + 1]))[0]
    # Record the number of rewards found in this interval
    rew[i] = len(ind)
    if len(ind)>0:
        thr[i] = threshold_crossing_time[ind[0]]

# Handle the last trial separately
ind_last = np.where(rew_ind > trial_start_indices[-1])[0]
if len(ind_last) > 0:
    rew[-1] = len(ind_last)

# Display the result
plt.plot(rew[0:],'.')
