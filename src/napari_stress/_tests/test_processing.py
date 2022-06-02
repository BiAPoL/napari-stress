# -*- coding: utf-8 -*-
import numpy as np


def test_rescaling4d():
    from napari_stress import rescale

    # Test for 4D
    image = np.random.random(size=(2, 100, 100, 100))

    resampled_image = rescale(image,
                              scale_z = 1.0,
                              scale_y = 1.0,
                              scale_x = 1.0)
    assert np.array_equal(resampled_image.shape, image.shape)

    resampled_image = rescale(image,
                              scale_z = 0.5,
                              scale_y = 0.5,
                              scale_x = 0.5)
    assert np.array_equal(
        resampled_image.shape[1:], (np.asarray(image.shape)/2)[1:]
        )

    # Test for 3D
    image = np.random.random(size=(100, 100, 100))
    resampled_image = rescale(image,
                              scale_z = 1.0,
                              scale_y = 1.0,
                              scale_x = 1.0)
    assert np.array_equal(resampled_image.squeeze().shape, image.squeeze().shape)

    # Test for 2D
    image = np.ones((100,100))
    resampled_image1 = rescale(image,
                               scale_z = 1.0,
                               scale_y = 1.0,
                               scale_x = 0.5).squeeze()
    resampled_image2 = rescale(image,
                               scale_z = 1.0,
                               scale_y = 0.5,
                               scale_x = 1.0).squeeze()
    resampled_image3 = rescale(image,
                               scale_z = 0.5,
                               scale_y = 1.0,
                               scale_x = 1.0).squeeze()
    assert np.array_equal(resampled_image1.shape, np.asarray([100, 50]))
    assert np.array_equal(resampled_image2.shape, np.asarray([50, 100]))
    assert np.array_equal(resampled_image3.shape, np.asarray([100, 100]))
