# -*- coding: utf-8 -*-
import numpy as np

def test_spherical_harmonics():
    import napari_stress
    import vedo

    ellipse = vedo.shapes.Ellipsoid(res=100)

    # Use default options (same number of points)
    points = napari_stress.fit_spherical_harmonics(ellipse.points() * 100, max_degree=3)
    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

    # Use different number of points
    points = napari_stress.fit_spherical_harmonics(ellipse.points() * 100, max_degree=2,
                                                   n_points=1000)

    assert len(points) == 1000
