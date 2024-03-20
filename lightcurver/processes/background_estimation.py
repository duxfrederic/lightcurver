import sep
import numpy as np


def subtract_background(image, mask_sources_first=False, n_boxes=10):
    """
    subtracts an estimated smooth background from the 2d array "image".
    basic, for complex cases do your own background subtraction.
    Background subtraction is particularly sensitive if we want to achieve
    high precision on the relative zeropoints of the images.
    (we will not include a background in our deconvolutions, because it can always be
    degenerate with other sources or the object itself).

    Here we will do this in two steps if mask_sources_first is True.
    1. roughly subtract the background, extract the sources.
    2. mask the sources, estimate the background again.

    :param image: 2d numpy array
    :param mask_sources_first: bool, whether we identify sources and mask before doing background estimation.
    :param n_boxes: int, in how many boxes do we divide the side of the image for background estimation?
    :return: 2d numpy array
    """
    # so, first, estimate a background.
    box_size = np.min(image.shape) // n_boxes
    bkg = sep.Background(image, bw=box_size, bh=box_size, fw=3, fh=3)
    image_sub = image - bkg
    if not mask_sources_first:
        # that's it
        return image_sub, bkg

    # find a lot of sources.
    sources, seg_map = sep.extract(data=image_sub, var=bkg.globalrms**2, thresh=2,
                                   minarea=10, segmentation_map=True)
    # estimate again
    bkg = sep.Background(image, bw=box_size, bh=box_size, fw=3, fh=3, mask=(seg_map > 0))
    # sub again
    image_sub = image - bkg
    # and done
    return image_sub, bkg
