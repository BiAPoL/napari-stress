from napari.types import ImageData

from skimage import transform

import numpy as np
from ._utils.frame_by_frame import frame_by_frame


@frame_by_frame
def rescale(image: ImageData,
            scale_z: float = 1.0,
            scale_y: float = 1.0,
            scale_x: float = 1.0) -> ImageData:
    """
    Rescale an image by a given set of scale factors.

    Parameters
    ----------
    image : ImageData
    scale_z : float, optional
        factor by which to scale the image along the z axis. The default is 1.0.
    scale_y : float, optional
        factor by which to scale the image along the y dimension. The default is 1.0.
    scale_x : float, optional
        factor by which to scale the image along the x dimension. The default is 1.0.

    Returns
    -------
    ImageData

    """

    scale_factors = np.asarray([scale_z, scale_y, scale_x])

    return transform.rescale(image, scale=scale_factors)
