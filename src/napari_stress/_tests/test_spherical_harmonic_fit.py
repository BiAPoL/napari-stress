# -*- coding: utf-8 -*-
import vedo
import numpy as np
import napari_stress

def test_frontend_spherical_harmonics():

    ellipse = vedo.shapes.Ellipsoid()

    # Test pyshtools implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='shtools')[0]
    assert np.array_equal(ellipse.points().shape, points.shape)

    # Test stress implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='stress')[0]
    assert np.array_equal(ellipse.points().shape, points.shape)

    # Test default implementations
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)[0]
    assert np.array_equal(ellipse.points().shape, points.shape)

def test_frontend_analysis_toolbox(make_napari_viewer):
    from napari_stress import spherical_harmonics_toolbox, measure_curvature

    viewer = make_napari_viewer()

    points = napari_stress.get_droplet_point_cloud()[0]
    viewer.add_points(points[0], **points[1])

    measure_curvature(points[0][:, 1:], viewer=viewer, run_analysis_toolbox=True)


def test_spherical_harmonics():
    from napari_stress._spherical_harmonics import spherical_harmonics as sh

    ellipse_points = vedo.shapes.Ellipsoid().points()

    pts, coeffs_pysh = sh.shtools_spherical_harmonics_expansion(ellipse_points)
    pts, coeffs_stress = sh.stress_spherical_harmonics_expansion(ellipse_points)

    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress)  # with pickle
    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress,
                                                         use_minimal_point_set=False)  # with pickle
    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress)  # without pickle

def test_quadrature(make_napari_viewer):
    points = napari_stress.get_droplet_point_cloud()[0]

    lebedev_points = napari_stress.measure_curvature(points[0])

    viewer = make_napari_viewer()
    lebedev_points = napari_stress.measure_curvature(points[0], viewer=viewer)
    lebedev_points = napari_stress.measure_curvature(points[0],
                                                    use_minimal_point_set=True,
                                                    number_of_quadrature_points=50)

if __name__ == '__main__':
    import napari
    test_frontend_analysis_toolbox(napari.Viewer)
