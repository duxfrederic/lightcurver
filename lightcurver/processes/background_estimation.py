import sep


def subtract_background(image):
    """
    subtracts an estimated smooth background from the 2d array "image".
    basic, for complex cases do your own background subtraction.

    :param image: 2d numpy array
    :return: 2d numpy array
    """
    bkg = sep.Background(image, bw=64, bh=64, fw=3, fh=3)
    image_sub = image - bkg

    return image_sub, bkg
