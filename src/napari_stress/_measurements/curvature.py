# -*- coding: utf-8 -*-

import napari
from napari.types import PointsData
from .._stress import manifold_SPB as mnfd
from .._stress import euclidian_k_form_SPB as euc_kf
from .._spherical_harmonics.spherical_harmonics import get_normals_on_manifold

from .utils import naparify_measurement


import numpy as np

@naparify_measurement
def calculate_mean_curvature_on_manifold(manifold: mnfd.manifold,
                                         viewer: napari.Viewer = None) -> dict:
    """
    Calculate mean curvatures for a given manifold.

    Parameters
    ----------
    manifold: mnfd.manifold

    Returns
    -------
    np.ndarray
        Mean curvature value for every lebedev point.

    """
    normals = get_normals_on_manifold(manifold)

    # Test orientation:
    points = manifold.get_coordinates()
    centered_lbdv_pts = points - points.mean(axis=0)[None, :]

    # Makre sure orientation is inward, so H is positive (for Ellipsoid, and small deviations):
    Orientations = [np.dot(x, y) for x, y in zip(centered_lbdv_pts,  normals)]
    num_pos_orr = np.sum(np.asarray(Orientations).flatten() > 0)

    Orientation = 1. # unchanged (we want INWARD)
    if(num_pos_orr > .5 * len(centered_lbdv_pts)):
        Orientation = -1.

    mean_curvatures = Orientation*euc_kf.Combine_Chart_Quad_Vals(manifold.H_A_pts, manifold.H_B_pts, manifold.lebedev_info).squeeze()
    H0_arithmetic = averaged_mean_curvature(mean_curvatures)
    H0_surface_integral = surface_integrated_mean_curvature(mean_curvatures,
                                                            manifold)

    # aggregate results in dictionary
    features = {'Mean_curvature_at_lebedev_points': mean_curvatures}
    metadata = {'H0_arithmetic_average': H0_arithmetic,
               'H0_surface_integral': H0_surface_integral}

    return features, metadata

def averaged_mean_curvature(curvatures: np.ndarray) -> float:
    """Calculate arithmetic average of mean curvature."""
    return curvatures.flatten().mean()

def surface_integrated_mean_curvature(mean_curvatures: np.ndarray,
                                      manifold: mnfd.manifold):
    """Calculate mean curvature by integrating surface area."""
    Integral_on_surface = euc_kf.Integral_on_Manny(mean_curvatures,
                                                   manifold,
                                                   manifold.lebedev_info)
    Integral_on_sphere = euc_kf.Integral_on_Manny(np.ones_like(mean_curvatures).astype(float),
                                                  manifold,
                                                  manifold.lebedev_info)

    return Integral_on_surface/Integral_on_sphere
