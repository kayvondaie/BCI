# plot_config.py
import matplotlib.pyplot as plt
from cycler import cycler

def set_default_plot_style():
    plt.rcParams.update({
        'figure.figsize': (1, 1),
        'font.size': 4,
        'axes.titlesize': 6,
        'axes.labelsize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'legend.fontsize': 5,
        'figure.dpi': 300,
        'lines.linewidth': 0.5,
        'lines.color': 'k',
        'axes.prop_cycle': cycler('color', ['k']),
        'scatter.marker': '.',
        'lines.markersize': 2,
        'lines.linewidth': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.2,
        'xtick.major.width': 0.3,
        'ytick.major.width': 0.3,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
    })
