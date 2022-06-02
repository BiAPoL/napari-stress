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
    ellipse = vedo.shapes.Ellipsoid()

    points, coefficients = stress_spherical_harmonics_expansion(ellipse.points(),
                                                                max_degree=5)

    assert np.array_equal(coefficients.shape, np.array([3, 6, 6]))

if __name__ == '__main__':
    test_stress_spherical_harmonics()
