import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------
# BlueWhiteRed colormap (matches MATLAB)
# ------------------------------------------------------------
bluewhitered = LinearSegmentedColormap.from_list(
    "bluewhitered",
    [(0,0,1), (1,1,1), (1,0,0)]
)

# ------------------------------------------------------------
# Parameters and initial conditions
# ------------------------------------------------------------
dt = 0.01
tau = 0.1
t = np.arange(0, 3 + dt, dt)

inp = np.zeros_like(t)
inp[20:200] = 1
inp[200:220] = -2

# MATLAB: w_in = [1 -5 -5 -5 1 -5 -5 -5];
w_in = np.array([1,-5,-5,-5, 1,-5,-5,-5])

ff = 0.5
fb = 1
amp = 3

def fun(x):
    return np.maximum(x, 0)    # MATLAB x.*(x>0)

R = np.zeros((len(t), 8, 2))
WW = np.zeros((8, 8, 2))

# ------------------------------------------------------------
# Main loop for j = 1,2
# ------------------------------------------------------------
plt.figure(figsize=(12,6))
plot_counter = 1

for j in range(1, 3):

    # base 4×4 block
    w4 = np.array([
        [fb, ff, 0,   0],
        [0,  0,  1,  -1],
        [0,  0,  amp, amp],
        [0,  0, -amp,-amp]
    ])

    # build 8×8 W
    W = np.zeros((8,8))
    W[0:4,0:4] = w4
    W[4:8,4:8] = w4

    if j == 1:
        # MATLAB: W(5,5)=0
        W[4,4] = 0
    else:
        a = 0
        W[0,0] = a
        W[4,4] = 0
        W[0,4] = np.sqrt(1-a)
        W[4,0] = np.sqrt(1-a)

    w_big = W

    # --------------------------------------------------------
    # Run dynamics
    # --------------------------------------------------------
    r = np.zeros((len(t), 8))
    for i in range(len(t)-1):
        dr = (-r[i] + fun(r[i]) @ w_big + w_in * inp[i]) * (dt/tau)
        r[i+1] = r[i] + dr

    r = fun(r)
    R[:,:,j-1] = r
    WW[:,:,j-1] = w_big

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plt.subplot(2, 3, plot_counter)
    plt.plot(t, r)
    plt.title(f"r (j={j})")
    plot_counter += 1

    plt.subplot(2, 3, plot_counter)
    plt.imshow(np.corrcoef(r), cmap='viridis')
    plt.title("corr(r')")
    plot_counter += 1

    plt.subplot(2, 3, plot_counter)
    plt.imshow(r.T, aspect='auto')
    plt.title("r'")
    plot_counter += 1

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# wcc response matrix
# ------------------------------------------------------------
cells = np.arange(8)
cells = cells[(cells != 3) & (cells != 7)]  # remove MATLAB 4 and 8 → Python 3 and 7

t2 = np.arange(0, 0.4 + dt, dt)
wcc = np.zeros((8,8,2))

for j in range(2):
    W = WW[:,:,j]
    for ci in range(8):
        inp2 = np.zeros(8)
        inp2[ci] = 2

        r = np.zeros((len(t2), 8))
        for i in range(len(t2)-1):
            pulse = (i == 10)
            dr = (-r[i] + fun(r[i]) @ W + inp2 * pulse) * (dt/tau)
            r[i+1] = r[i] + dr
        r = fun(r)

        wcc[ci,:,j] = np.mean(r, axis=0)
        wcc[ci,ci,j] = 0

# Subselect cells
wcc_sub = wcc[cells][:,cells,:]

plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(wcc_sub[:,:,0], vmin=-0.3, vmax=0.3, cmap='viridis')
plt.title("wcc j=1")

plt.subplot(1,3,2)
plt.imshow(wcc_sub[:,:,1], vmin=-0.3, vmax=0.3, cmap='viridis')
plt.title("wcc j=2")

plt.subplot(1,3,3)
plt.imshow(wcc_sub[:,:,1] - wcc_sub[:,:,0], vmin=-0.3, vmax=0.3, cmap='viridis')
plt.title("difference")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Regression section
# ------------------------------------------------------------
dwcc = wcc_sub[:,:,1] - wcc_sub[:,:,0]

pre  = np.nanmean(R[0:200, :, 0], axis=0)
dpre = np.nanmean(R[0:200, :, 1], axis=0) - pre
trl  = np.nanmean(R[200:, :, 0], axis=0)
dtrl = np.nanmean(R[200:, :, 1], axis=0) - trl

vecs = [np.ones(8), pre, dpre, trl, dtrl]

# Build design matrix X
X = []
for a in vecs:
    for b in vecs:
        M = np.outer(a, b)
        M = M[cells][:,cells]
        X.append(M.flatten(order='F'))

X = np.column_stack(X)
X[:,0] = 0  # zero out constant-constant regressor

y = dwcc.flatten(order='F')

beta = np.linalg.pinv(X) @ y
beta = beta.reshape(len(vecs), len(vecs))

plt.figure(figsize=(5,5))
plt.imshow(beta.T, cmap=bluewhitered, vmin=-1/500, vmax=1/500)
plt.colorbar()
plt.title("beta")
plt.tight_layout()
plt.show()
