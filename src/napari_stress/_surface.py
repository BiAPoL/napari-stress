# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData
from ._utils import frame_by_frame

import vedo
import typing

@frame_by_frame
def reconstruct_surface_ball_pivot(points: PointsData,
                                   radius: float = 5,
                                   delta: float = 0) -> SurfaceData:

    surface = nppas.surface_from_point_cloud_ball_pivoting(points, radius,
                                                           delta_radius=delta)
    return surface


@frame_by_frame
def reconstruct_surface_alpha_shape(points: PointsData,
                                    alpha:float = 5.0) -> SurfaceData:
    surf = nppas.surface_from_point_cloud_alpha_shape(points,  alpha=alpha)

    return surf

@frame_by_frame
def reconstruct_surface(points: PointsData,
                        dims: list = [100, 100, 100],
                        radius: float = None,
                        sampleSize: int = None,
                        holeFilling: bool = True) -> SurfaceData:
    """
    Reconstruct a surface from a set of points.

    Parameters
    ----------
    points : PointsData (napari.types.PointsData)

    Returns
    -------
    SurfaceData: napari.types.SurfaceData

    """
    # Catch magicgui default values
    if radius == 0:
        radius = None

    if sampleSize == 0:
        sampleSize = None

    pointcloud = vedo.pointcloud.Points(points)
    surf = pointcloud.reconstructSurface(dims=dims,
                                         radius=radius,
                                         sampleSize=sampleSize,
                                         holeFilling=holeFilling)

    return (surf.points(), np.asarray(surf.faces(), dtype=int))

@frame_by_frame
def smooth_laplacian(surface: SurfaceData,
                     niter: int = 15,
                     relax_factor: float = 0.1,
                     edge_angle: float = 15,
                     feature_angle: float = 60,
                     boundary: bool = False) -> SurfaceData:
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.smoothLaplacian(niter=niter, relaxfact=relax_factor,
                         edgeAngle=edge_angle,
                         featureAngle=feature_angle,
                         boundary=boundary)

    return (mesh.points(), np.asarray(mesh.faces()))

@frame_by_frame
def smooth_sinc(surface: SurfaceData,
                niter: int = 15,
                passBand: float = 0.1,
                edgeAngle: float = 15,
                feature_angle: float = 60,
                boundary: bool = False) -> SurfaceData:

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.smooth(niter=niter, passBand=passBand,
                edgeAngle=edgeAngle, featureAngle=feature_angle,
                boundary=boundary)
    return (mesh.points(), np.asarray(mesh.faces(), dtype=int))

@frame_by_frame
def smoothMLS2D(points: PointsData,
                f: float = 0.2,
                radius: float = None) -> PointsData:

    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.smoothMLS2D(f=f, radius=radius)

    return pointcloud.points()[pointcloud.info['isvalid']]

@frame_by_frame
def surface_from_label(label_image: LabelsData,
                       scale: typing.Union[np.ndarray, list] = np.array([1.0, 1.0, 1.0])
                       ) -> SurfaceData:

    surf = list(nppas.label_to_surface(label_image))
    surf[0] = surf[0] * np.asarray(scale)[None, :]

    return surf

@frame_by_frame
def adjust_surface_density(surface: SurfaceData,
                           density_target: float) -> SurfaceData:
    """Adjust the number of vertices of a surface to a defined density"""
    import open3d

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    n_vertices_target = int(mesh.area() * density_target)

    # sample desired number of vertices from surface
    points = nppas.sample_points_poisson_disk(surface, n_vertices_target)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    # measure distances between points
    distances = np.array(pcd.compute_nearest_neighbor_distance())
    radius = np.median(distances)
    delta = 2 * distances.std()

    # reconstruct the surface
    surface = nppas.surface_from_point_cloud_ball_pivoting(points,
                                                           radius=radius,
                                                           delta_radius=delta)
    # Fix holes
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.fillHoles()

    return (mesh.points(), np.asarray(mesh.faces()))
