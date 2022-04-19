import numpy as np

def test_fit_functions():
    from napari_stress._utils import _sigmoid, _gaussian, _detect_maxima, _detect_drop

    x = np.arange(0, 100, 1)

    y = _sigmoid(x, center=50, amplitude=1.0, slope=0.5, offset=0)
    assert np.max(y) <= 1.0
    assert y[51] > 0.5 and y[49] < 0.5

    argdrop = _detect_drop(y)
    assert 49 <= argdrop <= 51

    y = _gaussian(x, center=50, sigma=2.0, amplitude=2.0)
    assert np.max(y) <= 2.0

    argmax = _detect_maxima(y)
    assert 49 <= argmax <= 51


def test_point_utils():

    from napari_stress._utils import list_of_points_to_points,\
        points_to_list_of_points

    # distribute fibonacci points
    points = np.zeros((100, 4))
    points[:, 0] = np.repeat(np.arange(0, 50, 1), 2)

    list_of_points = points_to_list_of_points(points)
    assert  isinstance(list_of_points, list)

    _points = list_of_points_to_points(list_of_points)
    assert np.array_equal(points, _points)

def test_surf_utils():

    from napari_stress._utils import surface_to_list_of_surfaces,\
    list_of_surfaces_to_surface
    from vedo import Sphere

    list_of_surfaces = [
        (Sphere().points(), np.asarray(Sphere().faces(), dtype=int))
        ] * 10

    surfaces = list_of_surfaces_to_surface(list_of_surfaces)
    _list_of_surfaces = surface_to_list_of_surfaces(surfaces)

    for idx in range(10):
        assert np.array_equal(list_of_surfaces[idx][0], _list_of_surfaces[idx][0])
        assert np.array_equal(list_of_surfaces[idx][1][1], _list_of_surfaces[idx][1][1])

def test_decorator(make_napari_viewer):
    from napari_stress import reconstruct_surface
    from napari_stress._utils import list_of_points_to_points
    from vedo import Sphere

    viewer = make_napari_viewer()

    points = [Sphere().points() * k for k in np.arange(1.9, 2.1, 0.1)]
    points = list_of_points_to_points(points)
    viewer.add_points(points, size=0.05)

    surf = reconstruct_surface(points)
    viewer.add_surface(surf)


if __name__ == '__main__':
    import napari
    test_decorator(napari.Viewer)
