import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals, svd

# --- Helper function: mean_bin_plot ---
def mean_bin_plot(x, y, nbins=20, plot=True):
    """
    Bin the (x,y) data into nbins, compute the mean of y in each bin,
    and compute the correlation between binned x and y.
    Returns:
        bin_centers: centers of the bins
        y_means: mean y for each bin
        counts: number of points per bin
        corr: correlation coefficient between binned x and y
    If plot=True, a scatter of the raw data and binned means is plotted.
    """
    # Bin data based on x
    bins = np.linspace(np.min(x), np.max(x), nbins+1)
    bin_indices = np.digitize(x, bins) - 1  # bins indices 0..nbins-1
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    y_means = np.zeros(nbins)
    counts = np.zeros(nbins)
    for i in range(nbins):
        mask = bin_indices == i
        if np.any(mask):
            y_means[i] = np.mean(y[mask])
            counts[i] = np.sum(mask)
        else:
            y_means[i] = np.nan
            counts[i] = 0

    # Compute correlation between binned values (ignoring bins with no data)
    valid = ~np.isnan(y_means)
    if np.sum(valid) > 1:
        corr = np.corrcoef(bin_centers[valid], y_means[valid])[0, 1]
    else:
        corr = np.nan

    if plot:
        plt.figure()
        plt.scatter(x, y, color='gray', alpha=0.5, label='Data')
        plt.plot(bin_centers, y_means, 'ro-', label='Binned mean')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Binned Plot (corr = {corr:.3f})')
        plt.show()

    return bin_centers, y_means, counts, corr

# --- Define nonlinearity ---
thr = 0.15
# The MATLAB code shows successive definitions; we use the final one:
# fun(x) = ((x-thr).*(x>thr)).^1.5
def fun(x):
    # Ensure using numpy arrays for elementwise operations.
    return np.power((x - thr) * (x > thr), 1.5)

# --- Set simulation parameters ---
thr2 = 4      # not used in the final definition
N = 50
#np.random.seed(0)  # for reproducibility
w_in = np.random.randn(1, N)

w = np.random.randn(N, N)
# Scale w so that its maximum real eigenvalue is 0.4
max_eval = np.max(np.real(eigvals(w)))
w = w / max_eval * 0.4

tau = 0.1
dt = 0.01
t = np.arange(0, 1+dt, dt)  # time vector

# --- Create input matrix ---
# Initially create a random input matrix then “project” it using SVD
inp = np.random.rand(len(t), N)
inp = inp * (np.random.rand(*inp.shape) > 0)  # effectively same as inp
U, S, Vt = svd(inp, full_matrices=False)
# project onto the first 'dim' singular vectors
dim = N
inp = inp @ (Vt.T[:, :dim] @ Vt[:dim, :])
inp = inp / 10
inp = inp * 0   # set to zero, as in MATLAB
# Optionally, you could set one row of input:
# inp[19, :] = 1   # MATLAB: in(20,:) = 1

# --- Offsets and pre-allocation ---
offsets = np.linspace(-0.5, 1, 50)
n_offsets = len(offsets)

# Pre-allocate arrays to store results
R_all = []  # list to store R for each offset, shape: (len(t), N, n_iter)
RR = np.zeros((len(t), N, n_offsets))
# Off-diagonal elements: there are N*N - N elements.
n_off = N*N - N
CC = np.zeros((n_off, n_offsets))
WCC = np.zeros((n_off, n_offsets))
direct = np.zeros((N, n_offsets))
ins = np.zeros((N, n_offsets))
outs = np.zeros((N, n_offsets))
tot = np.zeros((N, n_offsets))
C_corr = np.zeros(n_offsets)

n_iter = 10  # number of iterations for simulation

# --- Main loop over offsets ---
for oi, off_val in enumerate(offsets):
    # Create an offset vector (shape: (N,))
    offset = np.ones(N) * off_val
    R_iter = np.zeros((len(t), N, n_iter))
    
    # Run dynamics for each iteration
    for iter_idx in range(n_iter):
        r = np.zeros((len(t), N))
        for i in range(len(t) - 1):
            # Compute network input at time i:
            net_input = r[i, :] @ w + inp[i, :] + offset + np.random.randn(N)*0.1
            r[i+1, :] = r[i, :] + dt/tau * (-r[i, :] + fun(net_input))
        R_iter[:, :, iter_idx] = r
    # Average over iterations (averaging over time indices 50:end)
    k = np.nanmean(R_iter[50:, :, :], axis=0)  # shape: (N, n_iter)
    # Compute correlation matrix among neurons (each row corresponds to one neuron across iterations)
    cc_mat = np.corrcoef(k)  # shape: (N, N)
    # stim_amp simulation: measure response when a stimulus pulse is applied at time index 50
    stim_amp = 10
    wcc = np.zeros((N, N))
    for ci in range(N):
        p = np.zeros(N)
        p[ci] = stim_amp
        r = np.zeros((len(t), N))
        for i in range(len(t) - 1):
            # At time step 50 (i==50) add stimulus p
            pulse = p if (i == 50) else 0
            net_input = r[i, :] @ w + pulse + offset
            r[i+1, :] = r[i, :] + dt/tau * (-r[i, :] + fun(net_input))
        # Subtract baseline computed as mean of r at time steps 48 and 49 (MATLAB indexing)
        # Note: Python indices: MATLAB r(48:49,:) -> use indices 47:49 (zero-indexed)
        baseline = np.mean(r[47:49, :], axis=0)
        # Average response from time index 50 onward minus the baseline
        wcc[ci, :] = np.mean(r[49:, :] - baseline, axis=0)
    
    # Store the mean response over iterations for this offset
    RR[:, :, oi] = np.mean(R_iter, axis=2)
    
    # Get off-diagonal indices
    ind = np.where(~np.eye(N, dtype=bool))
    # Extract off-diagonal correlation and wcc values as vectors
    cc_off = cc_mat[ind]
    wcc_off = wcc[ind]
    # Compute a binned correlation using the mean_bin_plot helper
    _, _, _, corr_val = mean_bin_plot(cc_off, wcc_off, nbins=20, plot=False)
    C_corr[oi] = corr_val
    WCC[:, oi] = wcc_off
    CC[:, oi] = cc_off
    direct[:, oi] = np.diag(wcc)
    # Remove the diagonal from wcc:
    wcc_no_diag = wcc - np.diag(np.diag(wcc))
    ins[:, oi] = np.sum(wcc_no_diag, axis=0)
    outs[:, oi] = np.sum(wcc_no_diag, axis=1)
    tot[:, oi] = np.sum(RR[:, :, oi], axis=0)

# --- Plotting ---
plt.figure(1)
plt.suptitle('Non-linearity')

# Subplot 231: plot nonlinearity function over offsets
plt.subplot(2,3,1)
plt.plot(offsets, fun(offsets), 'k')
plt.xlabel('Offset')
plt.ylabel('fun(offset)')
    
# Choose two offset indices for detailed time-course plots:
bef = 15
aft = 45

# Subplot 232: plot RR for offset index 'bef'
plt.subplot(2,3,2)
plt.plot(t, RR[:, :, bef])
plt.title(f'Offset = {offsets[bef]:.2f}')
plt.xlabel('Time')
plt.ylabel('Activity')

# Subplot 233: plot RR for offset index 'aft'
plt.subplot(2,3,3)
plt.plot(t, RR[:, :, aft])
plt.title(f'Offset = {offsets[aft]:.2f}')
plt.xlabel('Time')
plt.ylabel('Activity')

# Subplot 235: mean_bin_plot of correlation vs. wcc for offset 'bef'
plt.subplot(2,3,5)
pf.mean_bin_plot(CC[:, bef], WCC[:, bef],5,1,1,'k')
plt.xlabel('CC_{i,j}')
plt.ylabel('Wcc_{i,j}')

# Subplot 236: mean_bin_plot of correlation vs. wcc for offset 'aft'
plt.subplot(2,3,6)
pf.mean_bin_plot(CC[:, aft], WCC[:, aft], 5,1,1,'k')
plt.xlabel('CC_{i,j}')
plt.ylabel('Wcc_{i,j}')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(2)

# Subplot 122: Plot binned differences between offsets 'aft' and 'bef'
plt.subplot(1,2,2)
diff_CC = CC[:, aft] - CC[:, bef]
diff_WCC = WCC[:, aft] - WCC[:, bef]
pf.mean_bin_plot(diff_CC, diff_WCC, 5,1,1,'k')
plt.xlabel('Δ correlation')
plt.ylabel('Δ W_{cc}')

# Subplot 121: Bar plot of C_corr vs offsets, and vertical line at thr
plt.subplot(1,2,1)
plt.bar(offsets, C_corr, color='r', width=(offsets[1]-offsets[0])*0.8)
plt.axvline(x=thr, color='k', linestyle=':')
plt.xlabel('Offset')
plt.ylabel('Corr(C_{i,j}, Wcc_{i,j})')
plt.tight_layout()
plt.show()
