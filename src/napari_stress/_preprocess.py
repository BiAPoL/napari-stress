from napari_tools_menu import register_function
from napari.types import ImageData

from ._utils import frame_by_frame

from skimage.transform import rescale
from skimage import measure

import pandas as pd
import numpy as np
import vedo


@frame_by_frame
def resample(image: ImageData,
            scale_factors: np.ndarray) -> ImageData:
    """
    Resample an image by a given set of scale factors.

    """

    return rescale(image, scale=scale_factors)


def fit_ellipse_4D(binary_image: np.ndarray,
                   n_rays: np.uint16):

    n_frames = binary_image.shape[0]

    # Fit ellipse
    pts = np.zeros([n_rays * n_frames, 4])
    for t in range(n_frames):
        _pts = fit_ellipse(binary_image=binary_image[t])
        pts[t * n_rays : (t + 1) * n_rays, 1:] = _pts
        pts[t * n_rays : (t + 1) * n_rays, 0] = t

    return pts

def fit_ellipse(binary_image: np.ndarray,
                n_samples: np.uint16 = 256) -> list:
    """
    Fit an ellipse to a binary image.

    Parameters
    ----------
    binary_image : np.ndarray
        DESCRIPTION.
    n_samples : np.uint16, optional
        DESCRIPTION. The default is 256.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if not len(binary_image.shape) == 4:
        binary_image = binary_image[None, :]

    # Allocate points
    n_frames = binary_image.shape[0]
    pts = []

    for t in range(n_frames):

        _binary_image = binary_image[t]
        props = measure.regionprops_table(_binary_image,
                                          properties=(['centroid']))
        CoM = [props['centroid-0'][0],
               props['centroid-1'][0],
               props['centroid-2'][0]]

        # Create coordinate grid:
        ZZ, YY, XX = np.meshgrid(np.arange(_binary_image.shape[0]),
                                  np.arange(_binary_image.shape[1]),
                                  np.arange(_binary_image.shape[2]), indexing='ij')

        # Substract center of mass and mask
        ZZ = (ZZ.astype(float) - CoM[0]) * _binary_image
        YY = (YY.astype(float) - CoM[1]) * _binary_image
        XX = (XX.astype(float) - CoM[2]) * _binary_image

        # Concatenate to single (Nx3) coordinate vector
        ZYX = np.vstack([ZZ.ravel(), YY.ravel(), XX.ravel()]).transpose((1, 0))

        # Calculate orientation matrix
        S = 1/np.sum(_binary_image) * np.dot(ZYX.conjugate().T, ZYX)
        D, R = np.linalg.eig(S)

        # Now create points on the surface of an ellipse
        props = pd.DataFrame(measure.regionprops_table(_binary_image, properties=(['major_axis_length', 'minor_axis_length'])))
        semiAxesLengths = [props.loc[0].major_axis_length/2,
                           props.loc[0].minor_axis_length/2,
                           props.loc[0].minor_axis_length/2]
        _pts = fibonacci_sphere(semiAxesLengths, R.T, CoM, samples=n_samples)
        pts.append(vedo.pointcloud.Points(_pts))

    return pts

# def fit_curvature():
#     """
#     Find curvature for every point
#     """

#     print('\n---- Curvature-----')
#     curv = []
#     for idx, point in tqdm.tqdm(self.points.iterrows(), desc='Measuring mean curvature', total=len(self.points)):
#         sXYZ, sXq = surface.get_patch(self.points, idx, self.CoM)
#         curv.append(curvature.surf_fit(sXYZ, sXq))

#     self.points['Curvature'] = curv
#     self.points = surface.clean_coordinates(self)

#     # Raise flags for provided data
#     self.has_curv = True
