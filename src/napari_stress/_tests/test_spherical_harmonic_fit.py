# -*- coding: utf-8 -*-
import vedo
import numpy as np
import napari_stress

def test_spherical_harmonics():

    ellipse = vedo.shapes.Ellipsoid()

    # Test pyshtools implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='shtools')
    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

    # Test stress implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='stress')
    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

    # Test default implementations
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)
    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

def test_stress_spherical_harmonics():

    from napari_stress._spherical_harmonics._expansion import stress_spherical_harmonics_expansion
    from napari_stress._spherical_harmonics.sph_func_SPB import spherical_harmonics_function

    ellipse = vedo.shapes.Ellipsoid()

    degree = 5
    points, coefficients = stress_spherical_harmonics_expansion(ellipse.points(),
                                                                max_degree=degree)

    assert np.array_equal(coefficients.shape, np.array([3, degree + 1, degree + 1]))

    function_x = spherical_harmonics_function(coefficients[0], degree)
    value = function_x.Eval_SPH(0, 0)
    delta_phi = function_x.Eval_SPH_Der_Phi(1, 1)
    delta_phi_phi = function_x.Eval_SPH_Der_Phi_Phi(1, 1)
    derivative1 = function_x.Eval_SPH_Der_Phi(1, 1)

if __name__ == '__main__':
    test_stress_spherical_harmonics()
