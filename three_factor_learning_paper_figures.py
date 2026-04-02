#%% ============================================================================
# Three-Factor Learning Paper — Master Figure Script
# ============================================================================
# Run each cell to regenerate the corresponding panel(s).
# Each cell runs both the computation and figure-generation scripts.
# All panels save PNG + SVG to the OneDrive panels directory.
#
# Pipeline per cell:
#   Cell 0: Common setup + session QC
#   Cell 1: Example pair (loads single session from network)
#   Cell 2: Hebbian index didactic (loads single session from network)
#   Cell 3: three_factor_variance_explained.py → .npy → _figure.py → panels
#   Cell 4: sliding_window_temporal_offset.py → .npy → _figure.py → panels
# ============================================================================

#%% ============================================================================
# CELL 0: Common setup + session QC
# ============================================================================
import sys, os, importlib
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, wilcoxon

import session_counting
import data_dict_create_module_test as ddct
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *
import plotting_functions as pf

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(_THIS_DIR, 'meta_analysis_results')
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')
os.makedirs(PANEL_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'svg.fonttype': 'none',
})

# Run session QC if csv doesn't exist yet
_qc_csv = os.path.join(RESULTS_DIR, 'qc', 'qc_summary.csv')
if not os.path.exists(_qc_csv):
    print("QC summary not found — running session_qc.py ...")
    exec(open(os.path.join(_THIS_DIR, 'session_qc.py')).read())
else:
    print(f"QC summary exists: {_qc_csv}")

print("=== CELL 0 done: common setup ===")

#%% ============================================================================
# CELL 1: Example pair — average top N pairs
# Source: example_pair_simple.py (loads single session from network)
# Output: {mouse}_{session}_avg_top{N}.png/.svg
# ============================================================================
exec(open(os.path.join(_THIS_DIR, 'example_pair_simple.py')).read())
print("=== CELL 1 done: example pair figures ===")

#%% ============================================================================
# CELL 2: Hebbian index didactic — HI time series + CC vs dW scatter
# Source: hebbian_index_didactic.py (loads single session from network)
# Output: hebbian_index_didactic_*.png/.svg
#         hebbian_index_matrices_*.png/.svg
#         hebbian_index_pooled_*.png/.svg
# ============================================================================
exec(open(os.path.join(_THIS_DIR, 'hebbian_index_didactic.py')).read())
print("=== CELL 2 done: hebbian index didactic figures ===")

#%% ============================================================================
# CELL 3: 2-factor vs 3-factor variance explained
# Step 1: Run computation across all sessions → saves .npy
# Step 2: Run figure script → loads .npy → saves panels
# Output: three_factor_variance_explained.png/.svg
#         two_factor_variance_explained.png/.svg
#         two_vs_three_factor_comparison.png/.svg
#         two_vs_three_factor_overlay.png/.svg
# ============================================================================
print("--- Running three_factor_variance_explained.py (computation) ---")
exec(open(os.path.join(_THIS_DIR, 'three_factor_variance_explained.py')).read())
print("--- Running three_factor_variance_explained_figure.py (figures) ---")
exec(open(os.path.join(_THIS_DIR, 'three_factor_variance_explained_figure.py')).read())
print("=== CELL 3 done: variance explained figures ===")

#%% ============================================================================
# CELL 4: Sliding window temporal offset — coefficient matrices + RPE scatter
# Step 1: Run computation across all sessions → saves .npy
# Step 2: Run figure script → loads .npy → saves panels
# Output: temporal_offset_matrices.png/.svg
#         temporal_offset_matrices_logp.png/.svg
#         rpe_vs_hi_slope.png/.svg
#         dw_vs_cc_rpe_split.png/.svg
# ============================================================================
print("--- Running sliding_window_temporal_offset.py (computation) ---")
exec(open(os.path.join(_THIS_DIR, 'sliding_window_temporal_offset.py')).read())
print("--- Running sliding_window_temporal_offset_figure.py (figures) ---")
exec(open(os.path.join(_THIS_DIR, 'sliding_window_temporal_offset_figure.py')).read())
print("=== CELL 4 done: temporal offset figures ===")

#%% ============================================================================
# CELL 5: Threshold-driven learning — example session + population summary
# Step 1: Run threshold_analysis.py (computation + per-session figs)
# Step 2: Run figure_threshold_learning.py (composite figure)
# Output: figure_threshold_learning.png/.svg
# ============================================================================
print("--- Running threshold_analysis.py (computation) ---")
exec(open(os.path.join(_THIS_DIR, 'threshold_analysis.py')).read())
print("--- Running figure_threshold_learning.py (figure) ---")
exec(open(os.path.join(_THIS_DIR, 'figure_threshold_learning.py')).read())
print("--- Running figure_gain_modulation.py (figure) ---")
exec(open(os.path.join(_THIS_DIR, 'figure_gain_modulation.py')).read())
print("=== CELL 5 done: threshold learning figures ===")
