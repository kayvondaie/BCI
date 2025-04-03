def run():
    # Cell 20
fig2, (ax2pp, ax2, ax2p) = plt.subplots(1, 3, figsize=(10, 4))

n_direct_for_pr = np.array(n_direct_for_pr)
prs_direct_1_2 = np.array(prs_direct_1_2)
prs_direct_1 = np.array(prs_direct_1)
prs_direct_2 = np.array(prs_direct_2)

ax2pp.scatter(prs_direct_1_2, n_direct_for_pr, marker='.', color=c_vals[4], alpha=0.3, label='Day 1+2')
ax2pp.scatter(prs_direct_1, n_direct_for_pr, marker='.', color=c_vals[2], alpha=0.3, label='Day 1')
ax2pp.scatter(prs_direct_2, n_direct_for_pr, marker='.', color=c_vals[3], alpha=0.3, label='Day 2')

add_identity(ax2pp, color='k', zorder=5)

ax2pp.set_xlabel('Direct participation ratio')
ax2pp.set_ylabel('# Direct')
ax2pp.legend()

ax2pp.set_xlim((0, 1.05 * np.max(n_direct_for_pr)))
ax2pp.set_ylim((0, 1.05 * np.max(n_direct_for_pr)))

_, bins, _ = ax2.hist(prs_direct_1_2, bins=30, color=c_vals[4], alpha=0.3, label='Day 1+2')
_ = ax2.hist(prs_direct_1, bins=bins, color=c_vals[2], alpha=0.3, label='Day 1')
_ = ax2.hist(prs_direct_2, bins=bins, color=c_vals[3], alpha=0.3, label='Day 2')
ax2.axvline(np.nanmean(prs_direct_1_2), color=c_vals_d[4], zorder=5)
ax2.axvline(np.nanmean(prs_direct_1), color=c_vals_d[2], zorder=5)
ax2.axvline(np.nanmean(prs_direct_2), color=c_vals_d[3], zorder=5)

ax2.set_xlabel('Direct participation ratio')
ax2.legend()

pr_ratio = prs_direct_1_2 / (1/2. * (np.array(prs_direct_1) + np.array(prs_direct_1)))

ax2p.hist(pr_ratio, bins=40, color=c_vals[4])
ax2p.axvline(np.nanmean(pr_ratio), color=c_vals_d[4], zorder=5)
ax2p.axvline(1.0, color='k', zorder=5)

ax2p.set_xlabel('Relative PR size (combined/individual)')

    print("✅ plot_direct_prs ran successfully.")
if __name__ == '__main__':
    run()