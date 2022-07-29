# -*- coding: utf-8 -*-
import numpy as np
import napari_stress
import vedo
import napari

def test_smoothing():
    from napari_stress import smoothMLS2D, smooth_sinc
    import vedo

    sphere = vedo.shapes.Sphere(res=30)
    points = sphere.points() * 10
    faces = np.asarray(sphere.faces())
    points += np.random.uniform(size=points.shape)

    smoothed_points = smoothMLS2D(points, factor=0.25, radius=0)
    smoothed_points = smooth_sinc((points, faces))

    points = napari_stress.get_droplet_point_cloud()[0][0][:, 1:]
    smoothed_points = smoothMLS2D(points, factor=1, radius=0)
    smoothed_points = smoothMLS2D(points, factor=1, radius=5)

def test_reconstruction():
    points = vedo.shapes.Ellipsoid().points() * 100

    surface = napari_stress.reconstruct_surface(points)

def test_surface_to_points():
    ellipse = vedo.shapes.Ellipsoid()

    surface = (ellipse.points(), np.asarray(ellipse.faces()))
    points = napari_stress.extract_vertex_points(surface)

def test_ellipsoid_points():
    pointcloud = np.random.normal(size=(1000, 3)) * 10 * np.array([1,2,3])[None, :]
    ellipse_points = napari_stress.fit_ellipsoid_to_pointcloud_points(pointcloud, inside_fraction=0.5)
    axis = napari_stress.fit_ellipsoid_to_pointcloud_vectors(pointcloud, inside_fraction=0.5)
    axis = napari_stress.fit_ellipsoid_to_pointcloud_vectors(pointcloud, inside_fraction=0.5, normalize=True)

    pointcloud = vedo.shapes.Ellipsoid().points() * 10
    ellipse_points = napari_stress.fit_ellipsoid_to_pointcloud_points(pointcloud, inside_fraction=0.5)

    # test 4d handling
    Converter = napari_stress.TimelapseConverter()
    pointcloud_4d = Converter.list_of_data_to_data([pointcloud, pointcloud + 1], napari.types.PointsData)
    vectors_4d = napari_stress.fit_ellipsoid_to_pointcloud_vectors(pointcloud_4d)

    # directions_4d = np.zeros_like(pointcloud_4d)
    # directions_4d[:, 0] = pointcloud_4d[:, 0]
    # directions_4d[directions_4d[:,0] == 0, 1:] = pointcloud - pointcloud.mean(axis=0)[None, :]
    # directions_4d[directions_4d[:,0] == 1, 1:] = pointcloud - (pointcloud + 1).mean(axis=0)[None, :]

if __name__ == '__main__':
    test_smoothing()
