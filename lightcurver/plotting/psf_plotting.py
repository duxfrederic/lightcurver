import matplotlib.pyplot as plt
import numpy as np


def plot_psf_diagnostic(datas, noisemaps, residuals, full_psf,
                        loss_curve=None, masks=None, names=None,
                        diagnostic_text=None,
                        save_path=None):
    """
    a utility to plot a summary of a psf modelling: plots the stars, noisemaps, and residuals.
    optionally, one can pass the loss curve (to make sure it looks like the model converged),
    names (one name per star), masks (for visualization, will appear as holes in the noisemaps),
    and some diagnostic text to be displayed at the top right.
    Args:
        datas: 3d array of shape (N, nx, ny): N stars, one slice per star.
        noisemaps: same as above but for the noisemaps
        residuals: same as above but for the residuals
        full_psf: full pixelated psf model
        loss_curve: optional, 1D array containing the evolution of the loss during optimization
        masks: optional, 3d array of shape (N, nx, ny): the masks that were used during optimization.
        names: optional, list of strings: for identifying stars.
        diagnostic_text: optional, string, with line breaks (max ~ 20 chars per line)
        save_path: optional, string or path, where to save the plot.

    Returns: None

    """
    plt.style.use('dark_background')
    cmap = 'viridis'
    cmap_residuals = 'coolwarm'
    text_color = 'white'
    text_size = 11
    single_letter_text_size = 14
    info_box_text_size = 6

    N = len(datas)
    if names is not None:
        assert N == len(names)

    sub_size = 3
    fig, ax = plt.subplots(3, N+1, figsize=((N+1)*sub_size, 3*sub_size))

    for i in range(N):
        for j in range(3):
            ax[j, i].axis('off')
            ax[j, i].set_aspect('equal')
            if j == 0:
                ax[j, i].imshow(datas[i], cmap=cmap)
                if names is not None:
                    ax[j, i].text(0.5, 0.02, names[i],
                                  horizontalalignment='center',
                                  verticalalignment='bottom',
                                  transform=ax[j, i].transAxes,
                                  color=text_color, fontsize=single_letter_text_size,
                                  weight='bold')
            elif j == 1:
                ax[j, i].imshow(noisemaps[i], cmap=cmap)
                ax[j, i].text(0.5, 0.02, 'noisemap, mask',
                              horizontalalignment='center',
                              verticalalignment='bottom',
                              transform=ax[j, i].transAxes,
                              color=text_color, fontsize=text_size,
                              weight='bold')
            elif j == 2:
                res = np.array(residuals[i])  # explicit casting, jax stuff
                if masks is not None:
                    mask = np.array(masks[i]).astype(bool)
                    res[np.where(~mask)] = np.nan
                ax[j, i].imshow(res, cmap=cmap_residuals)
                ax[j, i].text(0.5, 0.02, 'residuals',
                              horizontalalignment='center',
                              verticalalignment='bottom',
                              transform=ax[j, i].transAxes,
                              color=text_color, fontsize=text_size,
                              weight='bold')

    # info box
    ax[0, N].text(0.1, 0.99, diagnostic_text,
                  horizontalalignment='left',
                  verticalalignment='top',
                  fontsize=info_box_text_size,
                  color='white')
    ax[0, N].axis('off')

    # loss plot
    if loss_curve is not None:
        ax[1, N].plot(loss_curve, color='white')
        ax[1, N].text(0.5, 0.99, 'loss',
                      horizontalalignment='center',
                      verticalalignment='top',
                      transform=ax[1, N].transAxes,
                      color='white', fontsize=text_size,
                      weight='bold')

        ax[1, N].axis('off')
    # psf model plot
    ax[2, N].imshow(full_psf, cmap=cmap, aspect='auto')
    ax[2, N].axis('off')
    ax[2, N].text(0.5, 0.01, 'Full PSF',
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  transform=ax[2, N].transAxes,
                  color=text_color, fontsize=text_size,
                  weight='bold')
    ax[2, N].set_aspect('equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    if save_path is not None:
        plt.savefig(save_path, pad_inches=0, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
