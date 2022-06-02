# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData
from napari_stress._utils.frame_by_frame import frame_by_frame

import vedo
import typing

import pyshtools

@frame_by_frame
def fit_spherical_harmonics(points: PointsData,
                            max_degree: int = 5) -> PointsData:
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
    # Convert points coordinates relative to center
    center = points.mean(axis=0)
    relative_coordinates = points - center[np.newaxis, :]

    # Convert point coordinates to spherical coordinates (in degree!)
    spherical_coordinates = vedo.cart2spher(relative_coordinates[:, 0],
                                            relative_coordinates[:, 1],
                                            relative_coordinates[:, 2])
    radius = spherical_coordinates[0]
    latitude = np.rad2deg(spherical_coordinates[1])
    longitude = np.rad2deg(spherical_coordinates[2])

    # Find spherical harmonics expansion coefficients until specified degree
    opt_fit_params = pyshtools._SHTOOLS.SHExpandLSQ(radius, latitude, longitude,
                                                    lmax = max_degree)[1]
    # Sample radius values at specified latitude/longitude
    spherical_harmonics_coeffcients = pyshtools.SHCoeffs.from_array(opt_fit_params)
    values = spherical_harmonics_coeffcients.expand(lat=latitude, lon=longitude)

    # Convert points back to cartesian coordinates
    points = vedo.spher2cart(values,
                             np.deg2rad(latitude),
                             np.deg2rad(longitude))
    return points.transpose() + center[np.newaxis, :]

@frame_by_frame
def extract_vertex_points(surface: SurfaceData) -> PointsData:
    """
    Extract the vertices of a surface as points layer.

    Parameters
    ----------
    surface : SurfaceData

    Returns
    -------
    PointsData

    """
    points = surface[0]
    return points


def reconstruct_surface(points: PointsData,
                        radius: float = None,
                        sampleSize: int = None,
                        holeFilling: bool = True,
                        bounds: tuple = (),
                        padding: float = 0.05
                        ):
    pointcloud = vedo.pointcloud.Points(points)

    dims = np.max(pointcloud.points(), axis=0) - np.min(pointcloud.points(), axis=0)
    surf = pointcloud.reconstructSurface(dims=dims.astype(int),
                                         radius=radius,
                                         sampleSize=sampleSize,
                                         holeFilling=holeFilling,
                                         bounds=bounds,
                                         padding=padding)

    return (surf.points(), np.asarray(surf.faces(), dtype=int))


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

def smoothMLS2D(points: PointsData,
                f: float = 0.2,
                radius: float = None) -> PointsData:

    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.smoothMLS2D(f=f, radius=radius)

    return pointcloud.points()[pointcloud.info['isvalid']]

def adjust_surface_density(surface: SurfaceData,
                           density_target: float) -> vedo.mesh.Mesh:

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    n_vertices_target = int(mesh.area() * density_target)

    while mesh.N() < n_vertices_target:
        mesh.subdivide()

    mesh.decimate(N=n_vertices_target)

    return (mesh.points(), np.asarray(mesh.faces(), dtype=int))
