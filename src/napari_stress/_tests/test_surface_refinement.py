# -*- coding: utf-8 -*-
import numpy as np

def test_surface_tracing(make_napari_viewer):
    from napari_stress import trace_refinement_of_surface
    from skimage import filters, morphology, io, measure
    from vedo import shapes

    true_radius = 30

    # Make a blurry sphere first
    image = np.zeros([100, 100, 100])
    image[50 - 30:50 + 31,
          50 - 30:50 + 31,
          50 - 30:50 + 31] = morphology.ball(radius=true_radius)
    blurry_sphere = filters.gaussian(image, sigma = 5)

    # Put surface points on a slightly larger radius and add a bit of noise
    surf_points = shapes.Sphere().points()
    surf_points += (surf_points * true_radius + 2) + 50

    # Test different fit methods (fancy/quick)
    fit_type = 'quick'
    traced_points = trace_refinement_of_surface(blurry_sphere, surf_points,
                                                trace_length=20,
                                                sampling_distance=1.0,
                                                selected_fit_type=fit_type,
                                                remove_outliers=False)
    radial_vectors = np.array([50, 50, 50])[None, :] - traced_points
    mean_radii = np.linalg.norm(radial_vectors, axis=1).mean()

    assert np.allclose(true_radius, mean_radii, atol=1)

    fit_type = 'fancy'
    traced_points = trace_refinement_of_surface(blurry_sphere, surf_points,
                                                trace_length=10,
                                                selected_fit_type=fit_type,
                                                remove_outliers=False)
    radial_vectors = np.array([50, 50, 50])[None, :] - traced_points
    mean_radii = np.linalg.norm(radial_vectors, axis=1).mean()

    assert np.allclose(true_radius, mean_radii, atol=1)

    # Test outlier identification
    surf_points[0] += [0, 0, 10]
    traced_points = trace_refinement_of_surface(blurry_sphere, surf_points,
                                                trace_length=10,
                                                selected_fit_type=fit_type,
                                                remove_outliers=True)
    assert len(traced_points.squeeze()) < len(surf_points)

    # Now, let's test surface-labelled data
    blurry_ring = filters.sobel(image)

    surf_points = shapes.Sphere().points()
    surf_points += (surf_points * true_radius + 2) + 50

    fit_type = 'fancy'
    traced_points = trace_refinement_of_surface(blurry_ring, surf_points,
                                                trace_length=10,
                                                sampling_distance=0.1,
                                                selected_fit_type=fit_type,
                                                selected_edge='surface',
                                                remove_outliers=False)

    fit_type = 'quick'
    traced_points = trace_refinement_of_surface(blurry_ring, surf_points,
                                                trace_length=10,
                                                sampling_distance=0.1,
                                                selected_fit_type=fit_type,
                                                selected_edge='surface',
                                                remove_outliers=False)

if __name__ == '__main__':
    import napari
    test_surface_tracing(napari.Viewer)
