# -*- coding: utf-8 -*-
from napari.types import PointsData
import vedo
import numpy as np

from typing import Tuple
from enum import Enum

from .._utils.frame_by_frame import frame_by_frame

import pyshtools
from . import sph_func_SPB as sph_f
from .._utils.fit_utils import Least_Squares_Harmonic_Fit
from .._utils.coordinate_conversion import cartesian_to_elliptical_coordinates


def shtools_spherical_harmonics_expansion(points: PointsData,
                                          max_degree: int = 5
                                          ) -> Tuple[PointsData, np.ndarray]:
    """
    Approximate a surface by spherical harmonics expansion with pyshtools implementation.

    Parameters
    ----------
    points : PointsData
    max_degree : int, optional
        Degree of spherical harmonics to fit to the data. The default is 5.

    Returns
    -------
    PointsData

    """
    # Convert points coordinates relative to center
    center = points.mean(axis=0)
    relative_coordinates = points - center[np.newaxis, :]

    # Convert point coordinates to spherical coordinates (in degree!)
    spherical_coordinates = vedo.cart2spher(relative_coordinates[:, 0],
                                            relative_coordinates[:, 1],
                                            relative_coordinates[:, 2])
    radius = spherical_coordinates[0]
    longitude = np.rad2deg(spherical_coordinates[1])
    latitude = np.rad2deg(spherical_coordinates[2])

    # Find spherical harmonics expansion coefficients until specified degree
    opt_fit_params = pyshtools._SHTOOLS.SHExpandLSQ(radius, latitude, longitude,
                                                    lmax = max_degree)[1]
    # Sample radius values at specified latitude/longitude
    spherical_harmonics_coeffcients = pyshtools.SHCoeffs.from_array(opt_fit_params)
    values = spherical_harmonics_coeffcients.expand(lat=latitude, lon=longitude)

    # Convert points back to cartesian coordinates
    points = vedo.spher2cart(values,
                             spherical_coordinates[1],
                             spherical_coordinates[2]).transpose()

    points = points + center[np.newaxis, :]
    return points, spherical_harmonics_coeffcients.to_array()

def stress_spherical_harmonics_expansion(points: PointsData,
                                         max_degree: int = 5) -> Tuple[PointsData, np.ndarray]:
    """
    Approximate a surface by spherical harmonics expansion with stress implementation.

    Parameters
    ----------
    points : PointsData
    max_degree : int, optional
        Degree of spherical harmonics to fit to the data. The default is 5.

    Returns
    -------
    PointsData
    """

    # get LS Ellipsoid estimate and get point cordinates in elliptical coordinates
    longitude, latitude = cartesian_to_elliptical_coordinates(points)

    # This implementation fits a superposition of three sets of spherical harmonics
    # to the data, one for each cardinal direction (x/y/z).
    optimal_fit_parameters = Least_Squares_Harmonic_Fit(
        fit_degree=max_degree,
        points_ellipse_coords = (longitude, latitude),
        input_points = points)

    X_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 0], max_degree)
    Y_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 1], max_degree)
    Z_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 2], max_degree)

    coefficients = np.stack([X_fit_sph_coef_mat, Y_fit_sph_coef_mat, Z_fit_sph_coef_mat])

    # Create SPH_func to represent X, Y, Z:
    X_fit_sph = sph_f.spherical_harmonics_function(coefficients[0], max_degree)
    Y_fit_sph = sph_f.spherical_harmonics_function(coefficients[1], max_degree)
    Z_fit_sph = sph_f.spherical_harmonics_function(coefficients[2], max_degree)

    X_fit_sph_UV_pts = X_fit_sph.Eval_SPH(longitude, latitude)
    Y_fit_sph_UV_pts = Y_fit_sph.Eval_SPH(longitude, latitude)
    Z_fit_sph_UV_pts = Z_fit_sph.Eval_SPH(longitude, latitude)

    fitted_points = np.hstack((X_fit_sph_UV_pts, Y_fit_sph_UV_pts, Z_fit_sph_UV_pts ))

    return fitted_points, coefficients

class spherical_harmonics_methods(Enum):
    shtools = {'function': shtools_spherical_harmonics_expansion}
    stress = {'function': stress_spherical_harmonics_expansion}

@frame_by_frame
def fit_spherical_harmonics(points: PointsData,
                            max_degree: int = 5,
                            implementation: spherical_harmonics_methods = spherical_harmonics_methods.stress
                            ) -> PointsData:
    """
    Approximate a surface by spherical harmonics expansion

    Parameters
    ----------
    points : PointsData
    max_degree : int
        Order up to which spherical harmonics should be included for the approximation.

    Returns
    -------
    PointsData
        Pointcloud on surface of a spherical harmonics expansion at the same
        latitude/longitude as the input points.

    See also
    --------
    [1] https://en.wikipedia.org/wiki/Spherical_harmonics#/media/File:Spherical_Harmonics.png

    """
    # Parse inputs
    if isinstance(implementation, str):
        fit_function = spherical_harmonics_methods.__members__[implementation].value['function']
    else:
        fit_function = implementation.value['function']
    fitted_points, coefficients = fit_function(points, max_degree=max_degree)

    return fitted_points
