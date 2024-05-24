import numpy as np


def rescale_image_coordinates(xy_coordinates_array, image_shape):
    """
    translates and rescales the coordinates in xy_array (origin bottom left of image), such that the new origin
    is the center of the image, and the coordinates go from -1 to 1.
    E.g., bottom left is (-1, -1), center is (0, 0), top left is (-1, 1).
    Function only factored out to keep things consistent with distortion.

    Args:
        xy_coordinates_array: an array of shape (N, 2), denoting a list of coordinate pairs (x,y) with origin
                              at the bottom left of the image.
        image_shape: shape of the image the coordinates refer to, obtained with `image.shape`

    Returns:
`       rescaled_xy_coordinates: an array of shape (N,2) with origin in the center of the image, and values in [-1, 1]
    """
    image_dims = np.array(image_shape)[::-1]  # reversed because y~lines, x~columns
    center = (image_dims - 1) / 2.

    rescaled_xy_coordinates = xy_coordinates_array - center
    rescaled_xy_coordinates /= image_dims

    return rescaled_xy_coordinates
