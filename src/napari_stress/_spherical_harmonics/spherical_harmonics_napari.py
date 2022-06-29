# -*- coding: utf-8 -*-
from napari.types import LayerDataTuple, PointsData
from napari.layers import Points
import numpy as np
from enum import Enum

from .._utils.frame_by_frame import frame_by_frame
from .toolbox import spherical_harmonics_toolbox
from .spherical_harmonics import shtools_spherical_harmonics_expansion,\
    stress_spherical_harmonics_expansion,\
    lebedev_quadrature,\
    create_manifold,\
    calculate_mean_curvature_on_manifold

import napari
import pathlib, os
from napari_tools_menu import register_function


class spherical_harmonics_methods(Enum):
    """Available methods for spherical harmonics expansion."""

    shtools = {'function': shtools_spherical_harmonics_expansion}
    stress = {'function': stress_spherical_harmonics_expansion}

@register_function(menu="Points > Fit spherical harmonics (n-STRESS")
@frame_by_frame
def fit_spherical_harmonics(points: PointsData,
                            max_degree: int = 5,
                            implementation: spherical_harmonics_methods = spherical_harmonics_methods.stress,
                            run_analysis_toolbox: bool = True
                            ) -> LayerDataTuple:
    """
    Approximate a surface by spherical harmonics expansion.

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

    See Also
    --------
    [1] https://en.wikipedia.org/wiki/Spherical_harmonics#/media/File:Spherical_Harmonics.png

    """
    # Parse inputs
    if isinstance(implementation, str):
        fit_function = spherical_harmonics_methods.__members__[implementation].value['function']
    else:
        fit_function = implementation.value['function']
    fitted_points, coefficients = fit_function(points, max_degree=max_degree)

    properties, features, metadata = {}, {}, {}

    features['error'] = np.linalg.norm(fitted_points - points, axis=1)
    metadata['spherical_harmonics_coefficients'] = coefficients
    metadata['spherical_harmonics_implementation'] = implementation

    properties['features'] = features
    properties['metadata'] = metadata
    properties['face_color'] = 'error'
    properties['size'] = 0.5

    return (fitted_points, properties, 'points')

@register_function(menu="Measurement > Surface curvature from points (n-STRESS)",
                   number_of_quadrature_points={'min': 6, 'max': 5180})
@frame_by_frame
def measure_curvature(points: PointsData,
                      max_degree: int = 5,
                      implementation: spherical_harmonics_methods = spherical_harmonics_methods.stress,
                      number_of_quadrature_points: int = 500,
                      use_minimal_point_set: bool = False,
                      viewer: napari.Viewer = None,
                      run_analysis_toolbox: bool = True
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

    lebedev_points, LBDV_Fit = lebedev_quadrature(coefficients,
                                                  number_of_quadrature_points,
                                                  use_minimal_point_set=use_minimal_point_set)
    curvature = calculate_mean_curvature_on_manifold(lebedev_points,
                                                     lebedev_fit=LBDV_Fit,
                                                     max_degree=max_degree)

    properties, features, metadata = {}, {}, {}

    features['curvature'] = curvature
    metadata['averaged_curvature_H0'] = curvature.mean()

    properties['features'] = features
    properties['metadata'] = metadata
    properties['face_color'] = 'curvature'
    properties['size'] = 0.5
    properties['name'] = 'Result of measure curvature'

    if viewer is not None:
        if properties['name'] not in viewer.layers:
            viewer.add_points(lebedev_points, **properties)
            layer = viewer.layers[properties['name']]
        else:
            layer = viewer.layers[properties['name']]
            layer.features = features
            layer.data = lebedev_points

        if run_analysis_toolbox:
            toolbox = spherical_harmonics_toolbox(viewer, layer)
            viewer.window.add_dock_widget(toolbox)
    return lebedev_points
