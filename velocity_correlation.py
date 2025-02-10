



F = data['F']
vel = []
fstack = []
for ti in range(trl):
    steps = data['step_time'][ti]
    indices = np.searchsorted(tsta, steps)
    indices = indices[indices<690]
    v = np.zeros(len(tsta))
    v[indices] = 1;
    vel.append(v)
    fstack.append(F[:,:,ti])
fstack = np.concatenate(fstack)
vel = np.concatenate(vel)
ind = np.nanmean(fstack,axis=1);
ind = np.where(np.isnan(ind)==0)[0]
fstack = fstack[ind,:]
vel = vel[ind]

num = 20;
ker = np.convolve(vel,np.ones(num,))[num-1:]
vel_cor = np.array([np.corrcoef(fstack[:, i], ker.ravel())[0, 1] for i in range(fstack.shape[1])])
