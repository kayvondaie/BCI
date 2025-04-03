import matplotlib.pyplot as plt
import numpy as np

def plot_cell_87():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(1, 1, figsize=(12, 8))

    mice = []

    # for ax, connectivity_metric, y_label in zip(
    #     (ax1, ax2, ax3, ax4),
    #     ('cc_x', 'tuning_x', 'trial_resp_x', 'post_resp_x'),
    #     ('Sum direct neuron W_CC', 'Direct neuron tuning', )
    # )

    for idx, cc_x in enumerate(records['cc_x']):
        cc_x_below_zero = np.where(cc_x < 0)[0]
    #     print(cc_x_below_zero.shape[0])
    #     print('{} - num cc_x below zero: {}/{}'.format(
    #         idx, cc_x_below_zero.shape[0], cc_x.shape[0]
    #     ))
        print('Mouse {}, session {} - '.format(records['mice'][idx], records['session_idxs'][idx]), cc_x_below_zero)

        if records['mice'][idx] not in mice:
            mice.append(records['mice'][idx])

        ax1.scatter(records['mask_counts_x'][idx], cc_x, marker='.', color=c_vals[len(mice)-1], alpha=0.3)

    ax1.axhline(0.0, color='grey', zorder=-5, linestyle='dashed')
    ax1.set_xlabel('# of direct neurons')
    ax1.set_ylabel('Sum direct neuron W_CC')    

if __name__ == '__main__':
    plot_cell_87()
