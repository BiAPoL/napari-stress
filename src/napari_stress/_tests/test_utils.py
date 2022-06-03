import numpy as np
from napari.types import LayerData, PointsData, SurfaceData, ImageData

def test_fit_functions():
    from napari_stress._utils.fit_utils import _sigmoid, _gaussian, _detect_maxima, _detect_max_gradient

    x = np.arange(0, 100, 1)

    y = _sigmoid(x, center=50, amplitude=1.0, slope=0.5, offset=0)
    assert np.max(y) <= 1.0
    assert y[51] > 0.5 and y[49] < 0.5

    argdrop = _detect_max_gradient(y)
    assert 49 <= argdrop <= 51

    y = _gaussian(x, center=50, sigma=2.0, amplitude=2.0)
    assert np.max(y) <= 2.0

    argmax = _detect_maxima(y)
    assert 49 <= argmax <= 51

def test_decorator_points():
    from napari_stress import TimelapseConverter
    from vedo import Sphere

    Converter = TimelapseConverter()

    points_list = [Sphere().points() * k for k in np.arange(1.9, 2.1, 0.1)]
    points_array = Converter.list_of_data_to_data(points_list, PointsData)
    points_list_conv = Converter.data_to_list_of_data(points_array, PointsData)

    for pts, _pts in zip(points_list, points_list_conv):
        assert np.array_equal(points_list, points_list_conv)


def test_decorator_surfaces():
    from napari_stress import TimelapseConverter, frame_by_frame
    from napari_process_points_and_surfaces import sample_points_poisson_disk
    from napari.types import SurfaceData
    from vedo import Sphere

    Converter = TimelapseConverter()

    surface_list = [
        (Sphere().points() * k, np.asarray(Sphere().faces())) for k in np.arange(1.9, 2.1, 0.1)
        ]
    surface_array = Converter.list_of_data_to_data(surface_list, SurfaceData)
    surface_list_conv = Converter.data_to_list_of_data(surface_array, SurfaceData)

    for surf, _surf in zip(surface_list, surface_list_conv):
        assert np.array_equal(surf[0], _surf[0])
        assert np.array_equal(surf[1], _surf[1])

    points = frame_by_frame(sample_points_poisson_disk)(surface_array)
    points = frame_by_frame(sample_points_poisson_disk)(surface_list[0])

def test_decorator_images():

    from napari_stress import TimelapseConverter

    Converter = TimelapseConverter()

    image = np.random.random(size=(50,50,50))
    image_list = [k * image for k in np.arange(0.1, 1, 0.1)]

    image_array = Converter.list_of_data_to_data(image_list, ImageData)

    assert image_array.shape[0] == len(image_list)

    image_list_conv = Converter.list_of_data_to_data(image_array, ImageData)
    for img, _img in zip(image_list, image_list_conv):
        assert np.array_equiv(img, _img)

if __name__ =='__main__':
    test_decorator_surfaces()
