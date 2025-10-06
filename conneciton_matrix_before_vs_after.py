import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

conn_mats = []

# Build matrices for both epochs
for epoch in range(2):
    amp_mat = AMP[epoch]            # shape (nCells, nGroups)
    dist_mat = stimDist             # shape (nCells, nGroups)

    # Identify each group's target cell
    target_idx = np.argmin(dist_mat, axis=0)

    # Mask definition: exclude any neuron within 30 µm of its group's target
    mask = dist_mat > 30.0
    conn_matrix_nan = np.where(mask, amp_mat, np.nan)

    # Exclude target cells themselves
    for gi, targ in enumerate(target_idx):
        conn_matrix_nan[targ, gi] = np.nan

    conn_mats.append(conn_matrix_nan)

    # Plot epoch connectivity
    im = axes[epoch].imshow(
        conn_matrix_nan[0:, :],
        cmap='bwr',
        aspect='auto',
        vmin=-0.5,
        vmax=0.5
    )
    im.cmap.set_bad(color='white')
    axes[epoch].set_title(f"Epoch {epoch}")
    axes[epoch].set_xlabel("Target neuron (group)")
    axes[epoch].set_ylabel("Neuron index")

# Difference = Epoch 1 − Epoch 0
diff_mat = conn_mats[1] - conn_mats[0]

im_diff = axes[2].imshow(
    diff_mat[0:, :],
    cmap='bwr',
    aspect='auto',
    vmin=-0.5,
    vmax=0.5
)
im_diff.cmap.set_bad(color='white')
axes[2].set_title("Difference (Epoch1 − Epoch0)")
axes[2].set_xlabel("Target neuron (group)")
axes[2].set_ylabel("Neuron index")

# Shared colorbar
fig.colorbar(im_diff, ax=axes.ravel().tolist(), shrink=0.8, label="Connection strength (AMP)")

plt.show()
