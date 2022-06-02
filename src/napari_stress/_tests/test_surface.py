# -*- coding: utf-8 -*-
import numpy as np
import vedo
import napari_stress

def test_spherical_harmonics():
    ellipse = vedo.shapes.Ellipsoid()
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)

    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

def test_surface_to_points():
    ellipse = vedo.shapes.Ellipsoid()

    surface = (ellipse.points(), np.asarray(ellipse.faces()))
    points = napari_stress.extract_vertex_points(surface)

if __name__ =='__main__':
    test_surface_to_points()
