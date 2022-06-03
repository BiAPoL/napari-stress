# -*- coding: utf-8 -*-

from napari.types import PointsData, LayerDataTuple

from .._utils.frame_by_frame import frame_by_frame

from . import sph_func_SPB as sph_f
from . import euclidian_k_form_SPB as euc_kf
from . import lebedev_info_SPB as lebedev_info
from . import manifold_SPB as mnfd
from .expansion import spherical_harmonics_methods

import numpy as np
import napari
from napari_tools_menu import register_function

import numpy as np
import warnings

@register_function(menu="Measurement > Surface curvature from points (n-STRESS)",
                   number_of_quadrature_points={'min': 6, 'max': 5180})
@frame_by_frame
def measure_curvature(points: PointsData,
                      max_degree: int = 5,
                      implementation: spherical_harmonics_methods = spherical_harmonics_methods.stress,
                      number_of_quadrature_points: int = 500,
                      use_minimal_point_set: bool = False,
                      viewer: napari.Viewer = None
                      ) -> PointsData:
    """
    Measure curvature on pointcloud surface.

    Parameters
    ----------
    points : PointsData
        DESCRIPTION.
    max_degree : int, optional
        DESCRIPTION. The default is 5.
    implementation : spherical_harmonics_methods, optional
        DESCRIPTION. The default is spherical_harmonics_methods.stress.
    number_of_quadrature_points : int, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    PointsData
        DESCRIPTION.

    """
    if isinstance(implementation, str):
        fit_function = spherical_harmonics_methods.__members__[implementation].value['function']
    else:
        fit_function = implementation.value['function']
    fitted_points, coefficients = fit_function(points, max_degree=max_degree)

    # Clip number of quadrature points
    if number_of_quadrature_points > 5810:
        number_of_quadrature_points = 5810

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

    properties, features = {}, {}
    features['curvature'] = _integrate_on_manifold(lebedev_points, LBDV_Fit, max_degree).squeeze()

    properties['features'] = features
    properties['face_color'] = 'curvature'
    properties['size'] = 0.5
    properties['name'] = 'Result of measure curvature'

    if viewer is not None:
        if properties['name'] not in viewer.layers:
            viewer.add_points(lebedev_points, **properties)
        else:
            layer = viewer.layers[properties['name']]
            layer.features = properties
            layer.data = lebedev_points
    return lebedev_points


def _integrate_on_manifold(lebedev_points: PointsData, LBDV_Fit, max_degree: int):

    # create manifold to calculate H, average H:
    Manny_Dict = {}
    Manny_Name_Dict = {} # sph point cloud at lbdv
    Manny_Name_Dict['coordinates'] = lebedev_points

    Manny_Dict['Pickle_Manny_Data'] = False
    Manny_Dict['Maniold_lbdv'] = LBDV_Fit

    Manny_Dict['Manifold_SPH_deg'] = max_degree
    Manny_Dict['use_manifold_name'] = False # we are NOT using named shapes in these tests
    Manny_Dict['Maniold_Name_Dict'] = Manny_Name_Dict # sph point cloud at lbdv

    Manny = mnfd.manifold(Manny_Dict)

    # Test orientation:
    centered_lbdv_pts = lebedev_points - lebedev_points.mean(axis=0)[None, :]

    normal_X_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Manny.Normal_Vec_X_A_Pts, Manny.Normal_Vec_X_B_Pts, LBDV_Fit)
    normal_Y_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Manny.Normal_Vec_Y_A_Pts, Manny.Normal_Vec_Y_B_Pts, LBDV_Fit)
    normal_Z_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Manny.Normal_Vec_Z_A_Pts, Manny.Normal_Vec_Z_B_Pts, LBDV_Fit)

    normals_lbdv_points = np.stack([normal_X_lbdv_pts, normal_Y_lbdv_pts, normal_Z_lbdv_pts]).squeeze().transpose()

    # Makre sure orientation is inward, so H is positive (for Ellipsoid, and small deviations):
    Orientations = [np.dot(x, y) for x, y in zip(centered_lbdv_pts,  normals_lbdv_points)]
    num_pos_orr = np.sum(np.asarray(Orientations).flatten() > 0)

    Orientation = 1. # unchanged (we want INWARD)
    if(num_pos_orr > .5 * len(centered_lbdv_pts)):
        Orientation = -1.

    return Orientation*euc_kf.Combine_Chart_Quad_Vals(Manny.H_A_pts, Manny.H_B_pts, LBDV_Fit)
