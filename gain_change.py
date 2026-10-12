"""
Two-neuron context-dependent sensory processing model.

  r1 : sensory neuron, context-independent (r1 in {0, 1}).
  r2 : responder, r2 = g * phi(w * r1 + I - inh), with phi a fixed sigmoid.
  resp = r2(stim) - r2(baseline).

Four context mechanisms (one per column):
  1. Weight change      : w changes between contexts.
  2. Setpoint change    : tonic input I shifts where the cell sits on phi.
  3. Gain change        : multiplicative output gain g scales phi.
  4. Inhibitory circuit : inh neuron driven tonically only in context A,
                          clamping r2 low; silent in context B.

Photostim: tonic input I_stim added to the responder's input (not inh).
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'

# ---------- model ----------
def phi(x):
    return 1.0 / (1.0 + np.exp(-x))

def responder(r1, w, I, g, inh=0.0):
    return g * phi(w * r1 + I - inh)

r1_base, r1_stim = 0.0, 1.0
I_stim = 1.0                    # photostim: tonic input added to I in context A

# Params tuned so Delta_r_A ~ 0.20 and Delta_r_B ~ 0.60 in every mechanism.
mechanisms = [
    ('Weight change',                                # vary w, fix I, g
     dict(w=1.0, I=-1.5, g=1.0),
     dict(w=2.8, I=-1.5, g=1.0)),
    ('Setpoint change',                              # vary I, fix w, g
     dict(w=3.0, I=-4.3, g=1.0),                     # A: low end of phi (subthreshold)
     dict(w=3.0, I=-1.0, g=1.0)),                    # B: steep part of phi
    ('Gain change',                                  # vary g (multiplicative), fix w, I
     dict(w=1.0, I=-1.5, g=1.0),
     dict(w=1.0, I=-1.5, g=3.0)),
    ('Inhibitory circuit',                           # inh neuron on only in context A
     dict(w=3.0, I=-1.0, g=1.0, inh=3.3),            # A: tonic inhibition clamps r2 low
     dict(w=3.0, I=-1.0, g=1.0, inh=0.0)),           # B: inh neuron silent
]

# ---------- figure ----------
ax_w, ax_h = 1.4, 1.1
col_gap, row_gap = 0.55, 0.55
left_pad, right_pad = 0.6, 0.2
top_pad, bot_pad = 0.35, 0.55
n_cols, n_rows = 4, 3

fig_w = left_pad + n_cols * ax_w + (n_cols - 1) * col_gap + right_pad
fig_h = bot_pad  + n_rows * ax_h + (n_rows - 1) * row_gap + top_pad
fig = plt.figure(figsize=(fig_w, fig_h))

r1_grid = np.linspace(-0.5, 2.0, 251)
ctx_colors = ['#1f77b4', '#d62728', '#2ca02c']   # A, B, A+photostim

for j, (name, pA, pB) in enumerate(mechanisms):
    pA_stim = {**pA, 'I': pA['I'] + I_stim}
    pB_stim = {**pB, 'I': pB['I'] + I_stim}

    # per-panel ylim from the actual max responder value across conditions
    ymax = 1.05 * max(responder(r1_grid.max(), **p)
                      for p in (pA, pB, pA_stim))

    L = (left_pad + j * (ax_w + col_gap)) / fig_w

    # --- row 0 (top): transfer function r2 vs sensory drive r1 ---
    B = (bot_pad + 2 * (ax_h + row_gap)) / fig_h
    ax = fig.add_axes([L, B, ax_w / fig_w, ax_h / fig_h])

    ax.axvline(0, color='0.7', lw=0.4, ls='--', zorder=0)
    ax.axvline(1, color='0.7', lw=0.4, ls='--', zorder=0)

    for params, color, label in [(pA,      ctx_colors[0], 'A'),
                                 (pB,      ctx_colors[1], 'B'),
                                 (pA_stim, ctx_colors[2], 'A+stim')]:
        curve = responder(r1_grid, **params)
        ax.plot(r1_grid, curve, color=color, lw=1.3, label=label)
        y_base = responder(r1_base, **params)
        y_stim = responder(r1_stim, **params)
        ax.plot([r1_base, r1_stim], [y_base, y_stim],
                '-', color=color, lw=0.8, alpha=0.4)
        ax.plot(r1_base, y_base, 'o', color=color, ms=4.5, mfc='white', mew=1.1)
        ax.plot(r1_stim, y_stim, 'o', color=color, ms=4.5)

    ax.set_title(name, pad=4)
    ax.set_xlabel(r'Sensory drive $r_1$')
    if j == 0:
        ax.set_ylabel(r'Responder $r_2$')
    ax.set_xlim(-0.5, 2.0); ax.set_ylim(-0.05, ymax)
    ax.set_xticks([0, 1, 2])
    ax.legend(frameon=False, loc='lower right', handlelength=1.0,
              borderaxespad=0.2, handletextpad=0.4)

    # --- row 1 (middle): sensory-evoked Delta r ---
    B2 = (bot_pad + ax_h + row_gap) / fig_h
    ax2 = fig.add_axes([L, B2, ax_w / fig_w, ax_h / fig_h])
    dA      = responder(r1_stim, **pA)      - responder(r1_base, **pA)
    dB      = responder(r1_stim, **pB)      - responder(r1_base, **pB)
    dA_stim = responder(r1_stim, **pA_stim) - responder(r1_base, **pA_stim)
    ax2.bar([0, 1, 2], [dA, dB, dA_stim], color=ctx_colors, width=0.6)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['A', 'B', 'A+stim'])
    ax2.set_xlabel('Condition')
    if j == 0:
        ax2.set_ylabel(r'$\Delta r_2$ (sensory)')
    ax2.set_ylim(0, 0.8)
    ax2.axhline(0, color='k', lw=0.5)

    # --- row 2 (bottom): photostim-evoked Delta r (r1 held at baseline) ---
    B3 = bot_pad / fig_h
    ax3 = fig.add_axes([L, B3, ax_w / fig_w, ax_h / fig_h])
    dA_photo = responder(r1_base, **pA_stim) - responder(r1_base, **pA)
    dB_photo = responder(r1_base, **pB_stim) - responder(r1_base, **pB)
    ax3.bar([0, 1], [dA_photo, dB_photo],
            color=[ctx_colors[0], ctx_colors[1]], width=0.6)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['A', 'B'])
    ax3.set_xlabel('Context')
    if j == 0:
        ax3.set_ylabel(r'$\Delta r_2$ (photostim)')
    ax3.set_ylim(0, 0.8)
    ax3.axhline(0, color='k', lw=0.5)

# ---------- save ----------
save_dir = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
            r'\3-factor learning paper\claude code 032226\meta_analysis_results\panels')
fname = 'gain_change'
if os.path.isdir(save_dir):
    fig.savefig(os.path.join(save_dir, fname + '.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, fname + '.svg'))

plt.show()
