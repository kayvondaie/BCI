import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Parameters
thresh = 0.15
min_dist = 30.0   # exclude cells within 30 µm
epochs = [0, 1, "diff"]

# Extract centroid positions for each ROI
xc = np.array([np.mean(s['xpix']) for s in stat])
yc = np.array([np.mean(s['ypix']) for s in stat])
target_idx = np.argmin(stimDist, axis=0)

# Prepare amplitude matrices
amp0 = AMP[0]
amp1 = AMP[1]
amp_diff = amp1 - amp0

amp_dict = {0: amp0, 1: amp1, "diff": amp_diff}

# --- Compute global color normalization across all epochs ---
all_strengths = []
for key in amp_dict:
    mat = amp_dict[key]
    mask_all = (stimDist > min_dist) & (mat > thresh)
    if np.any(mask_all):
        all_strengths.append(mat[mask_all])
all_strengths = np.concatenate(all_strengths)

vmin = np.min(all_strengths)
vmax = np.percentile(all_strengths, 90)   # robust upper bound

norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("Reds")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")

for i, ep in enumerate(epochs):
    ax = axes[i]
    amp_mat = amp_dict[ep]

    ax.set_title(f"Epoch {ep}")
    ax.set_xlim(0, np.max(xc) + 20)
    ax.set_ylim(np.max(yc) + 20, 0)
    ax.axis("off")
    ax.set_facecolor("white")

    for gi, targ in enumerate(target_idx):
        tx, ty = xc[targ], yc[targ]

        # find non-targets above threshold and >30 µm away
        mask = (stimDist[:, gi] > min_dist) & (amp_mat[:, gi] > thresh)
        non_targets = np.where(mask)[0]
        strengths = amp_mat[non_targets, gi]

        # plot lines with color ∝ connection strength
        for nt, strength in zip(non_targets, strengths):
            nx, ny = xc[nt], yc[nt]
            color = cmap(norm(strength))
            ax.plot([tx, nx], [ty, ny], '-', color=color, linewidth=1.2)

        # dot size ∝ number of outgoing connections
        dot_size = 30 + 10 * len(non_targets)
        ax.plot(tx, ty, 'o', color='magenta',
                markersize=np.sqrt(dot_size),
                markeredgecolor='k', markeredgewidth=0.5)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
cbar.set_label("Connection strength (AMP)")

plt.tight_layout()
plt.show()
