from utils.helper_functions1 import get_data_dict

def run():
    # Cell 112
fig, ((ax1, ax2,), (ax3, ax4,)) = plt.subplots(2, 2, figsize=(8, 8))

ax1.scatter(tuning_n, tuning, marker='.', color=c_vals[0])
ax1.set_xlabel('Tuning from df_closedLoop')
ax1.set_ylabel('Tuning from data_dict[F]')

ax2.scatter(trial_resp_n, trial_resp, marker='.', color=c_vals[0])
ax2.set_xlabel('Trial resp. from df_closedLoop')
ax2.set_ylabel('Trial resp. from data_dict[F]')

ax3.scatter(post_n, post, marker='.', color=c_vals[0])
ax3.set_xlabel('Post from df_closedLoop')
ax3.set_ylabel('Post from data_dict[F]')

ax4.scatter(pre_n, pre, marker='.', color=c_vals[0])
ax4.set_xlabel('Pre from df_closedLoop')
ax4.set_ylabel('Pre from data_dict[F]')

for ax in (ax1, ax2, ax3, ax4):
    ax.axhline(0.0, color='lightgrey', zorder=5, linestyle='dashed')
    ax.axvline(0.0, color='lightgrey', zorder=5, linestyle='dashed')

    print("✅ compare_tuning_behav_vs_raw ran successfully.")
if __name__ == '__main__':
    run()