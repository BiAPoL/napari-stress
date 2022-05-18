# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData
from napari_stress._utils.time_slicer import frame_by_frame

import vedo
import typing

import pyshtools

@frame_by_frame
def fit_spherical_harmonics(points: PointsData,
                            max_degree: int) -> PointsData:
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
        DESCRIPTION.

    See also
    --------
    [1] https://en.wikipedia.org/wiki/Spherical_harmonics#/media/File:Spherical_Harmonics.png

    """
    center = points.mean(axis=0)
    pts = points - center[np.newaxis, :]
    pts_spherical = vedo.cart2spher(pts[:, 0], pts[:, 1], pts[:, 2])

    r = pts_spherical[0]
    lat = np.rad2deg(pts_spherical[1])
    lon = np.rad2deg(pts_spherical[2])
    popt = pyshtools._SHTOOLS.SHExpandLSQ(r, lat, lon, lmax = max_degree)[1]
    clm = pyshtools.SHCoeffs.from_array(popt)

    values = clm.expand(lat=lat, lon=lon)

    points = vedo.spher2cart(values, np.deg2rad(lat), np.deg2rad(lon))
    return points.transpose() + center[np.newaxis, :]


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

def surface_from_label(label_image: LabelsData,
                       scale: typing.Union[list, np.ndarray]) -> SurfaceData:

    if isinstance(scale, list):
        scale = np.array(scale)

    n_frames = label_image.shape[0]

    surfs = []
    for t in range(n_frames):
        surf = nppas.label_to_surface(label_image[t])
        surfs.append(vedo.mesh.Mesh((surf[0] * scale[None, :], surf[1])))

    return surfs


def adjust_surface_density(surface: SurfaceData,
                           density_target: float) -> vedo.mesh.Mesh:

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    n_vertices_target = int(mesh.area() * density_target)

    while mesh.N() < n_vertices_target:
        mesh.subdivide()

    mesh.decimate(N=n_vertices_target)

    return (mesh.points(), np.asarray(mesh.faces(), dtype=int))
