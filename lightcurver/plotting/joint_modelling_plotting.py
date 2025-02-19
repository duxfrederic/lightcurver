import matplotlib.pyplot as plt
import numpy as np


def plot_joint_modelling_diagnostic(datas, noisemaps, residuals, loss_curve, chi2_per_frame, starlet_background=None,
                                    save_path=None):
    """
    a diagnostic of how well a joint modelling went. We will plot a stack of the data, and a stack of
    the residuals. We will show how the loss behaved during optimization.
    It is important to avoid systematics, as this is the core of this package: doing precise photometry,
    so convergence when modelling the pixels is essential.
    Args:
        datas: 3d array of shape (N, nx, ny): N stars, one slice per star.
        noisemaps: same as above but for the noisemaps
        residuals: same as above but for the residuals
        loss_curve: 1D array containing the evolution of the loss during optimization
        chi2_per_frame: 1D array, the chi2 value of the fit, one per slice.
        starlet_background: 2D array, in case we included a regularized pixelated background in the model. default None.
        save_path: optional, string or path, where to save the plot.

    Returns: None

    """
    plt.style.use('dark_background')
    cmap = 'viridis'
    cmap_residuals = 'coolwarm'
    text_color = 'red'
    text_size = 11

    data_stack = np.mean(datas, axis=0)  # mean and not median, we wanna see them outliers
    residuals_stack = np.mean(residuals, axis=0)
    rel_residuals_stack = np.mean(residuals / noisemaps, axis=0)

    sub_size = 3
    ncols = 5 if starlet_background is None else 6
    fig_size_mult = 4.6 if starlet_background is None else 5.6
    fig, ax = plt.subplots(1, ncols, figsize=(fig_size_mult * sub_size, sub_size))
    ax = ax.flatten()
    # data stack
    ax[0].imshow(data_stack, cmap=cmap, aspect='auto', origin='lower')
    ax[0].axis('off')
    ax[0].text(0.5, 0.01, 'Data stack',
               horizontalalignment='center',
               verticalalignment='bottom',
               transform=ax[0].transAxes,
               color=text_color, fontsize=text_size,
               weight='bold')
    # residuals stack
    ax[1].imshow(residuals_stack, cmap=cmap_residuals, aspect='auto', origin='lower')
    ax[1].axis('off')
    ax[1].text(0.5, 0.01, 'residuals stack',
               horizontalalignment='center',
               verticalalignment='bottom',
               transform=ax[1].transAxes,
               color=text_color, fontsize=text_size,
               weight='bold')

    # rel residuals stack
    ax[2].imshow(rel_residuals_stack, cmap=cmap_residuals, aspect='auto', origin='lower')
    ax[2].axis('off')
    ax[2].text(0.5, 0.01, 'rel. residuals stack',
               horizontalalignment='center',
               verticalalignment='bottom',
               transform=ax[2].transAxes,
               color=text_color, fontsize=text_size,
               weight='bold')

    # loss plot
    ax[3].plot(loss_curve, color='white')
    ax[3].text(0.5, 0.99, 'loss',
               horizontalalignment='center',
               verticalalignment='top',
               transform=ax[3].transAxes,
               color='white', fontsize=text_size,
               weight='bold')

    ax[3].axis('off')
    # chi2 plot
    ax[4].hist(chi2_per_frame, color='white', bins=len(chi2_per_frame))
    ax[4].text(0.5, 0.99, 'chi2 per frame',
               horizontalalignment='center',
               verticalalignment='top',
               transform=ax[4].transAxes,
               color='white', fontsize=text_size,
               weight='bold')
    # and view of the background common to all epochs if we included one, just to make sure it isn't nonsense.
    if starlet_background is not None:
        ax[5].imshow(starlet_background, origin='lower')
        ax[5].axis('off')
        ax[5].text(0.5, 0.99, 'regularized background',
                   horizontalalignment='center',
                   verticalalignment='top',
                   transform=ax[5].transAxes,
                   color='white', fontsize=text_size,
                   weight='bold')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, pad_inches=0, bbox_inches='tight', dpi=130)
        plt.close()
    else:
        plt.show()
