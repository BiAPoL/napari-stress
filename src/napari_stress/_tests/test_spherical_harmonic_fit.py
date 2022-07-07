# -*- coding: utf-8 -*-
import vedo
import numpy as np
import napari_stress

def test_frontend_spherical_harmonics():

    ellipse = vedo.shapes.Ellipsoid()

    # Test pyshtools implementation
    points1 = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=10,
                                                   implementation='shtools')
    assert np.array_equal(ellipse.points().shape, points1[0].shape)

    # Test stress implementation
    points2 = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='stress')
    assert np.array_equal(ellipse.points().shape, points2[0].shape)

    # Test default implementations
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)[0]
    assert np.array_equal(ellipse.points().shape, points.shape)

def test_frontend_analysis_toolbox(make_napari_viewer):
    from napari_stress import measure_curvature

    viewer = make_napari_viewer()

    points = napari_stress.get_droplet_point_cloud()[0]
    viewer.add_points(points[0], **points[1])

    measure_curvature(points[0][:, 1:], viewer=viewer, run_analysis_toolbox=True)
    
    results_layer = viewer.layers[-1]
    assert 'averaged_curvature_H0' in results_layer.metadata.keys()
    assert 'spherical_harmonics_coefficients' in results_layer.metadata.keys()
    assert 'gauss_bonnet_relative_error' in results_layer.metadata.keys()    


def test_spherical_harmonics():
    from napari_stress._spherical_harmonics import spherical_harmonics as sh

    ellipse_points = vedo.shapes.Ellipsoid().points()

    pts, coeffs_pysh = sh.shtools_spherical_harmonics_expansion(ellipse_points)
    pts, coeffs_stress = sh.stress_spherical_harmonics_expansion(ellipse_points)

    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress)  # with pickle
    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress,
                                                         use_minimal_point_set=False)  # with pickle
    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress)  # without pickle

def test_interoperatibility():

    from napari_stress._spherical_harmonics import spherical_harmonics as sh
    from napari_stress._spherical_harmonics.sph_func_SPB import convert_coeffcients_stress_to_pyshtools,\
        convert_coefficients_pyshtools_to_stress,\
        spherical_harmonics_function

    deg = 10
    # get stress expansion of pointcloud
    points = napari_stress.get_droplet_point_cloud()[0][0][:, 1:]
    pts_stress, coeffs_stress = sh.stress_spherical_harmonics_expansion(points, max_degree=deg)

    # convert the x/y/z expansions to pysh format
    coeffs_pysh_x = convert_coeffcients_stress_to_pyshtools(coeffs_stress[0])
    coeffs_pysh_y = convert_coeffcients_stress_to_pyshtools(coeffs_stress[1])
    coeffs_pysh_z = convert_coeffcients_stress_to_pyshtools(coeffs_stress[2])

    # convert coeffs back to stress format
    coeffs_str_x = convert_coefficients_pyshtools_to_stress(coeffs_pysh_x)
    coeffs_str_y = convert_coefficients_pyshtools_to_stress(coeffs_pysh_y)
    coeffs_str_z = convert_coefficients_pyshtools_to_stress(coeffs_pysh_z)

    # check if coeffs are still the same
    _coeffs_stress = np.stack([coeffs_str_x, coeffs_str_y, coeffs_str_z])
    assert all((_coeffs_stress - coeffs_stress).flatten() == 0)

    # # Let's check whether these two give the same values when evaluated at the
    # # same coordinates

    # theta = np.linspace(0, 2*np.pi, 50)
    # phi = np.linspace(0, 2*np.pi, 100)
    # THETA, PHI = np.meshgrid(theta, phi)
    # # Stress:
    # SH_stress_x = spherical_harmonics_function(_coeffs_stress[0], SPH_Deg=deg)
    # stress_x = SH_stress_x.Eval_SPH(THETA, PHI)

    # SH_stress_y = spherical_harmonics_function(_coeffs_stress[1], SPH_Deg=deg)
    # stress_y = SH_stress_y.Eval_SPH(THETA, PHI)

    # SH_stress_z = spherical_harmonics_function(_coeffs_stress[2], SPH_Deg=deg)
    # stress_z = SH_stress_z.Eval_SPH(THETA, PHI)

    # stress_pts = np.stack([stress_z.flatten(), stress_y.flatten(), stress_x.flatten()]).transpose()


    # # Pyshtools:
    # pysh_x = coeffs_pysh_x.expand(lon = list(np.rad2deg(THETA.flatten())), lat=list(np.rad2deg(PHI.flatten())))
    # pysh_y = coeffs_pysh_y.expand(lon = list(np.rad2deg(THETA.flatten())), lat=list(np.rad2deg(PHI.flatten())))
    # pysh_z = coeffs_pysh_z.expand(lon = list(np.rad2deg(THETA.flatten())), lat=list(np.rad2deg(PHI.flatten())))
    # pysh_pts = np.stack([pysh_x, pysh_y, pysh_z]).transpose()



def test_quadrature(make_napari_viewer):
    points = napari_stress.get_droplet_point_cloud()[0]

    lebedev_points = napari_stress.measure_curvature(points[0])

    viewer = make_napari_viewer()
    lebedev_points = napari_stress.measure_curvature(points[0], viewer=viewer)
    lebedev_points = napari_stress.measure_curvature(points[0],
                                                    use_minimal_point_set=True,
                                                    number_of_quadrature_points=50)

def test_analysis():
    from napari_stress._spherical_harmonics import spherical_harmonics as sh

    ellipse_points = vedo.shapes.Ellipsoid().points()

    degree = 10
    pts, coeffs_pysh = sh.shtools_spherical_harmonics_expansion(ellipse_points, max_degree=degree)
    pts, coeffs_stress = sh.stress_spherical_harmonics_expansion(ellipse_points, max_degree=degree)

    power_spectra_pysh = sh.calculate_power_spectrum(coeffs_pysh)
    power_spectra_stress = sh.calculate_power_spectrum(coeffs_stress)

    assert len(power_spectra_pysh[0]) == degree +1
    assert len(power_spectra_stress[0]) == degree +1


if __name__ == '__main__':
    import napari
    test_frontend_analysis_toolbox(napari.Viewer)
