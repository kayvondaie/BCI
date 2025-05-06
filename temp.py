



from bci_time_series import *
step_vector, hit_vector, trial_start_vector = bci_time_series_fun(folder, data, rt, dt_si)

# Define time constant in frames (60 seconds)
tau = int(round(60 / dt_si))



step_rate = compute_ema(step_vector, round(1/dt_si), np.nanmean(step_vector))
step_rate_slow = compute_ema(step_vector, tau, np.nanmean(step_vector))
hit_rate  = compute_ema(hit_vector,round(3/dt_si),0)
hit_rate_slow  = compute_ema(hit_vector,tau,0)

# RPE-like signal: reward minus expected reward (proxied by step rate)
rpe_step = step_rate * (1 - 35*step_rate_slow)
rpe_hit = hit_rate * (1 - 500*hit_rate_slow)

plt.subplot(211)
plt.plot(rpe_step[0:])
plt.subplot(212)
plt.plot(rpe_hit[0:])
#%%
df_steps = df * rpe_step
df_step_corr = df_steps @ df_steps.T
plt.plot(df_steps[cn,0:1000]);plt.plot(df_steps[0,0:1000])



