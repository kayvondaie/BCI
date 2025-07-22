from suite2p.io import save_mat as save_mat_module
import numpy as np
import os


ops = np.load(os.path.join(folder, 'ops.npy'), allow_pickle=True).item()

# Load all files manually, handling optional ones
def try_load(fname):
    path = os.path.join(folder, fname)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else []

stat = try_load('stat.npy')
F = try_load('F.npy')
Fneu = try_load('Fneu.npy')
spks = try_load('spks.npy')
iscell = try_load('iscell.npy')
redcell = try_load('redcell.npy')
F_chan2 = try_load('F_chan2.npy')
Fneu_chan2 = try_load('Fneu_chan2.npy')

save_mat_module(ops, stat, F, Fneu, spks, iscell, redcell)
