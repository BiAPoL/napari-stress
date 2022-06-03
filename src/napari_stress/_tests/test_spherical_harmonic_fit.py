# -*- coding: utf-8 -*-
import vedo
import numpy as np
import napari_stress

def test_spherical_harmonics():

    ellipse = vedo.shapes.Ellipsoid()

    # Test pyshtools implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='shtools')
    assert np.array_equal(ellipse.points().shape, points.shape)

    # Test stress implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='stress')
    assert np.array_equal(ellipse.points().shape, points.shape)

    # Test default implementations
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)
    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

    degree = 5
    points, coefficients = stress_spherical_harmonics_expansion(ellipse.points(),
                                                                max_degree=degree)

    assert np.array_equal(coefficients.shape, np.array([3, degree + 1, degree + 1]))

    function_x = spherical_harmonics_function(coefficients[0], degree)
    value = function_x.Eval_SPH(0, 0)
    delta_phi = function_x.Eval_SPH_Der_Phi(1, 1)
    delta_phi_phi = function_x.Eval_SPH_Der_Phi_Phi(1, 1)
    derivative1 = function_x.Eval_SPH_Der_Phi(1, 1)

def test_quadrature(make_napari_viewer):
    points = napari_stress.get_dropplet_point_cloud()[0]

    lebedev_points = napari_stress.measure_curvature(points[0])

    viewer = make_napari_viewer()
    lebedev_points = napari_stress.measure_curvature(points[0], viewer=viewer)
    lebedev_points = napari_stress.measure_curvature(points[0],
                                                    use_minimal_point_set=True,
                                                    number_of_quadrature_points=50)

if __name__ == '__main__':
    test_stress_spherical_harmonics()
    assert np.array_equal(ellipse.points().shape, points.shape)