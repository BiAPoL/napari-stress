# -*- coding: utf-8 -*-
"""
This file bundles up spherical harmonics functionality to have all "behind-the-
scenes" functionality in one place. This will allow to have all fron-end-related
functions (e.g., those functions that are visible to napari) in a separated place.

"""

from napari.types import PointsData
from typing import Tuple, Union
import numpy as np
import vedo
import pyshtools
import warnings

from . import sph_func_SPB as sph_f
from .._utils.fit_utils import Least_Squares_Harmonic_Fit
from .._utils.coordinate_conversion import cartesian_to_elliptical_coordinates
from . import manifold_SPB as mnfd
from . import euclidian_k_form_SPB as euc_kf
from . import lebedev_info_SPB as lebedev_info

def shtools_spherical_harmonics_expansion(points: PointsData,
                                          max_degree: int = 5
                                          ) -> Tuple[PointsData, list]:
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
    return points, [spherical_harmonics_coeffcients]

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

def calculate_power_spectrum(coefficients: list) ->np.ndarray:
    """
    Convert coefficients to pyshtools format and calculate power spectrum.

    Parameters
    ----------
    coefficients : Union[np.ndarray, list]
        Coefficients of spherial harmonics expansion as np.ndarray (stress-format)
        or list of pyshtools.SHCoeffs

    Returns
    -------
    power_spectra : list
        list with power spectra for every expansion (i.e., if x/y/z are 
        approximated separately)

    """
    # convert to list[SHCoeffs]
    if isinstance(coefficients, pyshtools.SHCoeffs):


    # Calculate spectra for each set of coefficients
    power_spectra = [coeff.power_spectrum() for coeff in coefficients]
    return power_spectra


def lebedev_quadrature(coefficients: np.ndarray,
                       number_of_quadrature_points: int = 500,
                       use_minimal_point_set: bool = True
                       ) -> Tuple[PointsData, lebedev_info.lbdv_info]:
    """
    Calculate lebedev quadrature points for a given spherical harmonics expansion.

    Parameters
    ----------
    coefficients : np.ndarray
        Spherical harmonics coefficient matrix.
    number_of_quadrature_points : int, optional
        Number of quadrature points to retrieve on the passed set of spherical
        harmonics functions. The default is 500.
    use_minimal_point_set : bool, optional
        Depending on the degree a defined minimal set of quadrature points is
        sufficient to integrate exactly. The default is True.

    Returns
    -------
    lebedev_points : PointsData
    LBDV_Fit : lebedev_info
        Data container storing points and quick access for other parameters.

    """
    # Clip number of quadrature points
    if number_of_quadrature_points > 5810:
        number_of_quadrature_points = 5810

    # An expansion of degree 3 will have an Nx4x4 coefficient matrix
    max_degree = coefficients.shape[-1] - 1

    possible_n_points = np.asarray(list(lebedev_info.pts_of_lbdv_lookup.values()))
    index_correct_n_points = np.argmin(abs(possible_n_points - number_of_quadrature_points))
    number_of_quadrature_points = possible_n_points[index_correct_n_points]

    if use_minimal_point_set:
        number_of_quadrature_points = lebedev_info.look_up_lbdv_pts(max_degree + 1)

    # Only a specific amount of points are needed for spherical harmonics of a
    # given order. Using more points will not give more precise results.
    if number_of_quadrature_points > lebedev_info.look_up_lbdv_pts(max_degree + 1):
        warnings.warn(r'Note: Only {necessary_n_points} are required for exact results.')

    # Create spherical harmonics functions to represent z/y/x
    fit_functions = [
        sph_f.spherical_harmonics_function(x, max_degree) for x in coefficients
        ]

    # Get {Z/Y/X} Coordinates at lebedev points, so we can leverage our code more efficiently (and uniformly) on surface:
    LBDV_Fit = lebedev_info.lbdv_info(max_degree, number_of_quadrature_points)
    lebedev_points  = [
        euc_kf.get_quadrature_points_from_sh_function(f, LBDV_Fit, 'A') for f in fit_functions
        ]
    lebedev_points = np.stack(lebedev_points).squeeze().transpose()

    return lebedev_points, LBDV_Fit

def create_manifold(points: PointsData,
                    lebedev_fit: lebedev_info.lbdv_info,
                    max_degree: int) -> mnfd.manifold:

    # create manifold to calculate H, average H:
    Manny_Dict = {}
    Manny_Name_Dict = {} # sph point cloud at lbdv
    Manny_Name_Dict['coordinates'] = points

    Manny_Dict['Pickle_Manny_Data'] = False
    Manny_Dict['Maniold_lbdv'] = lebedev_fit

    Manny_Dict['Manifold_SPH_deg'] = max_degree
    Manny_Dict['use_manifold_name'] = False # we are NOT using named shapes in these tests
    Manny_Dict['Maniold_Name_Dict'] = Manny_Name_Dict # sph point cloud at lbdv

    return mnfd.manifold(Manny_Dict)

def get_normals_on_manifold(manifold: mnfd.manifold,
                            lebedev_fit: lebedev_info.lbdv_info):

    normal_X_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(manifold.Normal_Vec_X_A_Pts,
                                                       manifold.Normal_Vec_X_B_Pts,
                                                       lebedev_fit)
    normal_Y_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(manifold.Normal_Vec_Y_A_Pts,
                                                       manifold.Normal_Vec_Y_B_Pts,
                                                       lebedev_fit)
    normal_Z_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(manifold.Normal_Vec_Z_A_Pts,
                                                       manifold.Normal_Vec_Z_B_Pts,
                                                       lebedev_fit)

    normals_lbdv_points = np.stack([normal_X_lbdv_pts, normal_Y_lbdv_pts, normal_Z_lbdv_pts])
    return normals_lbdv_points.squeeze().transpose()

def calculate_mean_curvature_on_manifold(lebedev_points: PointsData,
                                         lebedev_fit: lebedev_info.lbdv_info,
                                         max_degree: int) -> np.ndarray:
    """
    Calculate mean curvatures for a set of lebedev points.

    Parameters
    ----------
    lebedev_points : PointsData
    lebedev_fit : lebedev_info.lbdv_info
        lebedev info object - provides info which point is defined by which chart.
    max_degree : int
        Degree of spherical harmonics expansion.

    Returns
    -------
    np.ndarray
        Mean curvature value for every lebedev point.

    """
    manifold = create_manifold(lebedev_points, lebedev_fit, max_degree)
    normals = get_normals_on_manifold(manifold, lebedev_fit)

    # Test orientation:
    points = manifold.get_coordinates()
    centered_lbdv_pts = points - points.mean(axis=0)[None, :]

    # Makre sure orientation is inward, so H is positive (for Ellipsoid, and small deviations):
    Orientations = [np.dot(x, y) for x, y in zip(centered_lbdv_pts,  normals)]
    num_pos_orr = np.sum(np.asarray(Orientations).flatten() > 0)

    Orientation = 1. # unchanged (we want INWARD)
    if(num_pos_orr > .5 * len(centered_lbdv_pts)):
        Orientation = -1.

    return Orientation*euc_kf.Combine_Chart_Quad_Vals(manifold.H_A_pts, manifold.H_B_pts, lebedev_fit).squeeze()
