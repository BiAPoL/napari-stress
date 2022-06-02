# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData
from napari_stress._utils.frame_by_frame import frame_by_frame

import vedo
import typing


@frame_by_frame
def reconstruct_surface(points: PointsData,
                        radius: float = 1.0,
                        holeFilling: bool = True,
                        padding: float = 0.05
                        ) -> SurfaceData:
    """
    Reconstruct a surface from a given pointcloud.

    Parameters
    ----------
    points : PointsData
    radius : float
        Radius within which to search for neighboring points.
    holeFilling : bool, optional
        The default is True.
    padding : float, optional
        Whether or not to thicken the surface by a given margin.
        The default is 0.05.

    Returns
    -------
    SurfaceData
    """
    pointcloud = vedo.pointcloud.Points(points)

    surface = pointcloud.reconstructSurface(radius=radius,
                                            sampleSize=None,
                                            holeFilling=holeFilling,
                                            padding=padding)

    return (surface.points(), np.asarray(surface.faces(), dtype=int))

@frame_by_frame
def extract_vertex_points(surface: SurfaceData) -> PointsData:
    """
    Return only the vertex points of an input surface.

    Parameters
    ----------
    surface : SurfaceData

    Returns
    -------
    PointsData

    """
    return surface[0]

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
                factor: float = 0.5,
                radius: float = None) -> PointsData:

    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.smoothMLS2D(f=factor, radius=radius)

    if radius is not None:
        return pointcloud.points()[pointcloud.info['isvalid']]
    else:
        return pointcloud.points()

@frame_by_frame
def decimate(surface: SurfaceData,
             fraction: float = 0.1) -> SurfaceData:

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))

    n_vertices = mesh.N()
    n_vertices_target = n_vertices * fraction

    while mesh.N() > n_vertices_target:
        _fraction = n_vertices_target/mesh.N()
        mesh.decimate(fraction=_fraction)

    return (mesh.points(), np.asarray(mesh.faces()))


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
    mesh.fillHoles(size=(radius+delta)**2)

    return (mesh.points(), np.asarray(mesh.faces()))
