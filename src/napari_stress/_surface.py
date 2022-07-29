# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData, VectorsData
from napari_stress._utils.frame_by_frame import frame_by_frame
from napari_tools_menu import register_function

import vedo

@register_function(menu="Points > Fit ellipsoid major axis to points (vedo, n-STRESS)")
@frame_by_frame
def fit_ellipsoid_to_pointcloud_points(points: PointsData,
                                       inside_fraction: float = 0.673) -> PointsData:
    """
    Fit an ellipsoid to a pointcloud an retrieve surface pointcloud.

    Parameters
    ----------
    points : PointsData
    inside_fraction : float, optional
        Fraction of points to be inside the fitted ellipsoid. The default is 0.673.

    Returns
    -------
    PointsData

    """
    ellipsoid = vedo.pcaEllipsoid(vedo.pointcloud.Points(points), pvalue=inside_fraction)

    output_points = ellipsoid.points()

    return output_points

@register_function(menu="Points > Fit ellipsoid points to points (vedo, n-STRESS)")
@frame_by_frame
def fit_ellipsoid_to_pointcloud_vectors(points: PointsData,
                                        inside_fraction: float = 0.673,
                                        normalize: bool = False) -> VectorsData:
    """
    Fit an ellipsoid to a pointcloud an retrieve the major axises as vectors.

    Parameters
    ----------
    points : PointsData
    inside_fraction : float, optional
        Fraction of points to be inside the fitted ellipsoid. The default is 0.673.
    normalize : bool, optional
        Normalize the resulting vectors. The default is False.

    Returns
    -------
    VectorsData

    """
    ellipsoid = vedo.pcaEllipsoid(vedo.pointcloud.Points(points), pvalue=inside_fraction)

    vectors = np.stack([ellipsoid.axis1 * ellipsoid.va,
                        ellipsoid.axis2 * ellipsoid.vb,
                        ellipsoid.axis3 * ellipsoid.vc])

    if normalize:
        vectors = vectors/np.linalg.norm(vectors, axis=0)[None, :]

    base_points = np.stack([ellipsoid.center, ellipsoid.center, ellipsoid.center])
    vectors = np.stack([base_points, vectors]).transpose((1,0,2))

    return vectors


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

@register_function(menu="Points > Create points from surface vertices (n-STRESS)")
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
    points = surface[0]
    return points


@register_function(menu="Surfaces > Smoothing (Windowed Sinc, vedo, n-STRESS)")
@frame_by_frame
def smooth_sinc(surface: SurfaceData,
                n_iterations: int = 15,
                passBand: float = 0.1,
                edgeAngle: float = 15,
                feature_angle: float = 60,
                boundary: bool = False) -> SurfaceData:
    """
    Adjust mesh point positions using the Windowed Sinc function interpolation kernel.

    Parameters
    ----------
    surface : SurfaceData
        DESCRIPTION.
    n_iterations : int, optional
        Number of iteratios. The default is 15.
    passBand : float, optional
        Passband of sinc filter. The default is 0.1.
    edgeAngle : float, optional
        Edge angle to control smoothing along edges. The default is 15.
    feature_angle : float, optional
        Specifies the feature angle for sharp edge identification. The default is 60.
    boundary : bool, optional
        The default is False.

    Returns
    -------
    SurfaceData

    See also
    --------
    https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.smooth

    """

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.smooth(niter=n_iterations,
                passBand=passBand,
                edgeAngle=edgeAngle,
                featureAngle=feature_angle,
                boundary=boundary)
    return (mesh.points(), np.asarray(mesh.faces(), dtype=int))

@register_function(menu="Points > Smoothing (MLS2D, vedo, n-STRESS)")
@frame_by_frame
def smoothMLS2D(points: PointsData,
                factor: float = 0.25,
                radius: float = 1.0) -> PointsData:
    """
    Smooth points with a Moving Least Squares algorithm variant.

    Parameters
    ----------
    points : PointsData
    f : float, optional
        Smoothing factor - typical range is [0,2]. Will be ignored if radius is
        different from 0. The default is 0.25.
    radius : float, optional
        Search radius for neighboring points to identify isolated points.
        Set this value to zero to ignore it. The default is 1.0.

    Returns
    -------
    PointsData

    See also
    --------
    https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points.smoothMLS2D

    """
    if radius == 0:
        radius = None

    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.smoothMLS2D(f=factor, radius=radius)

    if radius is not None:
        return pointcloud.points()[pointcloud.info['isvalid']]
    else:
        return pointcloud.points()

@register_function(menu="Surfaces > Simplify (decimate, vedo, n-STRESS)")
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


@register_function(menu="Surfaces > Surface density adjustment (vedo, n-STRESS)")
@frame_by_frame
def adjust_surface_density(surface: SurfaceData,
                           density_target: float = 1.0) -> SurfaceData:
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
