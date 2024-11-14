dt_si = data['dt_si']
t = np.arange(0, dt * (df.shape[1]), dt)
trial_strt = 0*t;
rew = 0*t
steps = data['step_time']
strt = data['trial_start']
rewT = data['reward_time']
F = data['F']
vel = np.zeros((F.shape[0],F.shape[2]))
offset = int(np.round(data['SI_start_times'][0]/dt_si)[0])
for i in range(len(steps)):
    if np.isnan(F[-1,0,i]):
        l = np.where(np.isnan(F[40:,0,i])==1)[0][0]+39;
    else:
        l = F.shape[0]
    v = np.zeros(l,)
    for si in range(len(steps[i])):
        ind = np.where(t>steps[i][si])[0][0]
        v[ind] = 1
    vel[0:l,i] = v
for i in range(len(strt)):
    ind = np.where(t>strt[i])[0][0]
    trial_strt[ind] = 1

rew = np.zeros((F.shape[0],F.shape[2]))
for i in range(len(rewT)):
    try:
        ind = np.where(t>rewT[i])[0][0]
        rew[ind,i] = 1
    except:
        pass
        

pos = 0*vel;
for ti in range(pos.shape[1]):
    for i in range(pos.shape[0]):
        pos[i,ti] = pos[i-1,ti]
        if vel[i,ti] == 1:
            pos[i,ti] = pos[i-1,ti] + 1; 
#%%
frew = np.full((140, F.shape[1], F.shape[2]), np.nan)
for i in range(F.shape[2]):
    ind = np.where(rew[:, i] == 1)[0]  # Find indices where rew[:, i] == 1
    if len(ind) == 1:  # Only proceed if there is exactly one index
        j = ind[0]  # Extract the single index
        start = max(j - 40, 0)  # Ensure the range doesn't go below 0
        end = min(j + 100, F.shape[0])  # Ensure the range doesn't exceed array bounds
        frew[:(end - start), :, i] = F[start:end, :, i]

fr = np.nanmean(frew,axis=2)    
fr_mean = np.nanmean(fr[0:40, :], axis=0)  # Compute the column-wise mean of the first 40 rows, ignoring NaNs
fr = fr - fr_mean[np.newaxis, :]  # Subtract the mean from every row of 'fr'

