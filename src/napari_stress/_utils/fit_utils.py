import numpy as np
import inspect
from napari.types import  PointsData

from .._spherical_harmonics import lebedev_info_SPB as lebedev_info

def _sigmoid(array: np.ndarray, center:float, amplitude:float, slope:float, offset:float):
    """
    Sigmoidal fit function
    https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
    """
    return amplitude / (1 + np.exp(-slope*(array-center))) + offset

def _gaussian(array: np.ndarray, center:float, sigma:float, amplitude:float):
    """
    Gaussian normal fit function
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    return amplitude/np.sqrt((2*np.pi*sigma**2)) * np.exp(-(array - center)**2 / (2*sigma**2))

def _detect_maxima(array: np.ndarray, center: float = None):
    """
    Function to find the maximum's index of data with a single peak.
    """
    return np.argmax(array)

def _detect_max_gradient(array: np.ndarray, center: float = None):
    """
    Function to find the location of the steepest gradient for sigmoidal data
    """
    return np.argmax(np.diff(array))

def _function_args_to_list(function: callable) -> list:

    sig = inspect.signature(function)
    return list(sig.parameters.keys())

def Least_Squares_Harmonic_Fit(fit_degree: int,
                               points_ellipse_coords: tuple,
                               input_points: PointsData,
                               use_true_harmonics: bool = True) -> np.ndarray:
    """
    Least squares harmonic fit to point cloud, given choice of basis and degree.
    """

    U, V = points_ellipse_coords[0], points_ellipse_coords[1]

    All_Y_mn_pt_in = []

    for n in range(fit_degree + 1):
        for m in range(-1*n, n+1):
            Y_mn_coors_in = []
            Y_mn_coors_in = lebedev_info.Eval_SPH_Basis(m, n, U, V)
            All_Y_mn_pt_in.append(Y_mn_coors_in)

    All_Y_mn_pt_in_mat = np.hstack(( All_Y_mn_pt_in ))
    x1 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 0])[0]
    x2 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 1])[0]
    x3 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 2])[0]

    return np.vstack([x1, x2, x3]).transpose()