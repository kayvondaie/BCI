s = reward_vector.copy()
dur = len(s) * dt_si
per_min = 60.0 / dt_si                     # frames-per-count -> per-minute
mu = float(s.mean())                        # session mean (in events/frame)

# --- Two exponential timescales, both in rewards/minute ---
TAU_FAST = dt_si.copy()     # sec  — smoothed instantaneous rate
TAU_SLOW = 60.0    # sec  — running baseline rate

rate_fast = ema_causal(s, TAU_FAST / dt_si, init=mu) * per_min   # rew/min
rate_slow = ema_causal(s, TAU_SLOW / dt_si, init=mu) * per_min   # rew/min

t_min = np.arange(len(s)) * dt_si / 60.0

plt.figure(figsize=(12, 5))
plt.subplot(211)
plt.plot(t_min, rate_fast, lw=0.6, color='#e74c3c',
         label=f'fast EMA (τ={TAU_FAST:g}s)')
plt.plot(t_min, rate_slow, lw=1.2, color='#27ae60',
         label=f'slow EMA (τ={TAU_SLOW:g}s)')
plt.axhline(mu * per_min, color='k', ls='--', lw=0.5,
            label=f'session mean = {mu*per_min:.2f}/min')
plt.ylabel('rate (rewards/min)')
plt.legend(fontsize=9, loc='upper right')

plt.subplot(212)
plt.plot(t_min, rate_fast - rate_slow, lw=0.6, color='#2c3e50')
plt.axhline(0, color='k', lw=0.3)
plt.ylabel('fast − slow (rew/min)')
plt.xlabel('Time (min)')
plt.tight_layout()
