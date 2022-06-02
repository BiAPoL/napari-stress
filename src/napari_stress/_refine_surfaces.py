# -*- coding: utf-8 -*-

import vedo
from napari.types import SurfaceData, ImageData, PointsData

from ._utils.fit_utils import _sigmoid, _gaussian, _function_args_to_list, _detect_max_gradient, _detect_maxima
from ._utils.frame_by_frame import frame_by_frame

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

import numpy as np
import tqdm
import pandas as pd

from enum import Enum

class fit_types(Enum):
    quick_edge_fit = 'quick'
    fancy_edge_fit = 'fancy'

class edge_functions(Enum):
    interior = {'fancy': _sigmoid, 'quick': _detect_max_gradient}
    surface = {'fancy': _gaussian, 'quick': _detect_maxima}

@frame_by_frame
def trace_refinement_of_surface(intensity_image: ImageData,
                                points: PointsData,
                                selected_fit_type: fit_types = fit_types.fancy_edge_fit,
                                selected_edge: edge_functions = edge_functions.interior,
                                trace_length: float = 10.0,
                                sampling_distance: float = 0.5,
                                scale_z: float = 1.0,
                                scale_y: float = 1.0,
                                scale_x: float = 1.0,
                                show_progress: bool = True,
                                remove_outliers: bool = False,
                                outlier_tolerance: float = 4.5
                                )-> PointsData:
    """
    Generate intensity profiles along traces.

    This function receives an intensity image and a pointcloud with points on
    the surface of an object in the intensity image. It assumes that the
    pointcloud corresponds to the vertices on a surface around that object.

    As a first step, the function calculates normals on the surface and
    multiplies the length of this vector with `trace_length`. The intensity of
    the input image is then sampled along this vector perpendicular to the
    surface with a distance of `sampling distance` between each point along the
    normal vector.

    The location of the object's surface is then determined by fitting a
    selected function to the intensity profile along the prolonged normal vector.

    Parameters
    ----------
    intensity_image : ImageData
    points : PointsData
    selected_fit_type : fit_types, optional
        Which fit types to choose from. Can be `fit_types.fancy_edge_fit`/`"fancy"` or
        `fit_types.quick_edge_fit`/`"quick"`.
    selected_edge : edge_functions, optional
        Depending on the fluorescence of the intensity image, a different fit
        function is required. Can be either of `edge_functions.interior` or
        edge_functions.surface. The default is `edge_functions.interior`.
    trace_length : float, optional
        Length of the normal vector perpendicular to the surface. The default is 2.0.
    sampling_distance : float, optional
        Distance between two sampled intensity values along the normal vector.
        The default is 0.1.
    scale_z : float
        Voxel size in z
    scale_y: float
        Voxel size in y
    scale_x: float
        Voxel size in x
    show_progress : bool, optional
        The default is False.
    remove_outliers : bool, optional
        If this is set to true, the function will evaluate the fit residues of
        the chosen function and remove points that are classified as outliers.
        The default is True.
    outlier_tolerance : float, optional
        Determines how strict the outlier removal will be. Values with
        `value > Q75 + interquartile_factor * IQR` are classified as outliers,
        whereas `Q75` and `IQR` denote the 75% quartile and the interquartile
        range, respectively.
        The default is 1.5.

    Returns
    -------
    PointsData

    """
    if isinstance(selected_fit_type, str):
        selected_fit_type = fit_types(selected_fit_type)

    if isinstance(selected_edge, str):
        edge_detection_function = edge_functions.__members__[selected_edge].value[selected_fit_type.value]
    else:
        edge_detection_function = selected_edge.value[selected_fit_type.value]

    # Convert to mesh and calculate normals
    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.computeNormalsWithPCA(orientationPoint=pointcloud.centerOfMass())

    # Define start and end points for the surface tracing vectors
    scale = np.asarray([scale_z, scale_y, scale_x])
    n_samples = int(trace_length/sampling_distance)
    start_points = pointcloud.points()/scale[None, :] - 0.5 * trace_length * pointcloud.pointdata['Normals']

    # Define trace vectors (full length and single step
    vectors = trace_length * pointcloud.pointdata['Normals']
    vector_step = vectors/n_samples

    # Create coords for interpolator
    X1 = np.arange(0, intensity_image.shape[0], 1)
    X2 = np.arange(0, intensity_image.shape[1], 1)
    X3 = np.arange(0, intensity_image.shape[2], 1)
    interpolator = RegularGridInterpolator((X1, X2, X3),
                                           intensity_image,
                                           bounds_error=False,
                                           fill_value=intensity_image.min())

    # Allocate arrays for results (location of object border, fit parameters,
    # fit errors, and intensity profiles)
    fit_parameters = _function_args_to_list(edge_detection_function)[1:]
    fit_errors = [p + '_err' for p in fit_parameters]
    columns = ['surface_points'] + ['idx_of_border'] + ['projection_vector'] +\
        fit_parameters + fit_errors + ['profiles']

    if len(fit_parameters) == 1:
        fit_parameters, fit_errors = fit_parameters[0], fit_errors[0]

    optimal_fit_parameters = []
    optimal_fit_errors = []
    new_surface_points = []
    projection_vectors = []
    idx_of_border = []

    # create empty dataframe to keep track of results
    fit_data = pd.DataFrame(columns=columns, index=np.arange(pointcloud.N()))

    if show_progress:
        iterator = tqdm.tqdm(range(pointcloud.N()), desc = 'Processing vertices...')
    else:
        iterator = range(pointcloud.N())

    # Iterate over all provided target points
    for idx in iterator:

        coordinates = [start_points[idx] + k * vector_step[idx] for k in range(n_samples)]
        fit_data.loc[idx, 'profiles'] = interpolator(coordinates)

        # Simple or fancy fit?
        if selected_fit_type == fit_types.quick_edge_fit:
            idx_of_border.append(edge_detection_function(np.array(fit_data.loc[idx, 'profiles'])))
            perror = 0
            popt = 0

        elif selected_fit_type == fit_types.fancy_edge_fit:
            popt, perror = _fancy_edge_fit(np.array(fit_data.loc[idx, 'profiles']),
                                           selected_edge_func=edge_detection_function)
            idx_of_border.append(popt[0])

        optimal_fit_errors.append(perror)
        optimal_fit_parameters.append(popt)

        # get new surface point
        new_point = (start_points[idx] + idx_of_border[idx] * vector_step[idx]) * scale
        new_surface_points.append(new_point)
        projection_vectors.append(idx_of_border[idx] * (-1) * vector_step[idx])

    fit_data['idx_of_border'] = idx_of_border
    fit_data[fit_parameters] = optimal_fit_parameters
    fit_data[fit_errors] = optimal_fit_errors
    fit_data['surface_points'] = new_surface_points
    fit_data['projection_vector'] = projection_vectors

    # NaN rows should be removed either way
    fit_data = fit_data.dropna().reset_index()

    # Filter points to remove points with high fit errors
    if remove_outliers:
        fit_data = _remove_outliers_by_index(fit_data,
                                             column_names=fit_errors,
                                             factor=outlier_tolerance,
                                             which='above')
        fit_data = _remove_outliers_by_index(fit_data,
                                             column_names='idx_of_border',
                                             factor=outlier_tolerance,
                                             which='both')
    output_points = np.stack(fit_data['surface_points'].to_numpy()).astype(float)
    return output_points

def _remove_outliers_by_index(table: pd.DataFrame,
                              column_names: list,
                              which: str = 'above',
                              factor: float = 1.5) -> pd.DataFrame:
    """
    Filter all rows in a dataframe that qualify as outliers based on column-statistics.

    Parameters
    ----------
    table : pd.DataFrame
    on : list
        list of column names that should be taken into account
    which : str, optional
        Can be 'above', 'below' or 'both' and determines which outliers to
        remove - the excessively high or low values or both.
        The default is 'above'.
    factor : float, optional
        Determine how far a datapoint is to be above the interquartile range to
        be classified as outlier. The default is 1.5.

    Returns
    -------
    table : pd.DataFrame

    """
    # Check if list or single string was passed
    if isinstance(column_names, str):
        column_names = [column_names]

    # Remove the offset error from the list of relevant errors - fluorescence
    # intensity offset is not meaningful for distinction of good/bad fit
    if 'offset_err' in column_names:
        column_names.remove('offset_err' )


    # True if values are good, False if outliers
    table = table.dropna().reset_index(drop=True)
    indices = np.ones(len(table), dtype=bool)

    for column in column_names:
        Q1 = table[column].quantile(0.25)
        Q3 = table[column].quantile(0.75)
        IQR = Q3 - Q1
        if which == 'above':
            idx = table[column] > (Q3 + factor * IQR)
        elif which == 'below':
            idx = table[column] < (Q1 - factor * IQR)
        elif which == 'both':
            idx = (table[column] < (Q1 - factor * IQR)) + (table[column] > (Q3 + factor * IQR))
        indices[table[idx].index] = False

    return table[indices]


def _fancy_edge_fit(array: np.ndarray,
                    selected_edge_func: edge_functions = edge_functions.interior
                    ) -> float:
    """
    Fit a line profile with a gaussian normal curve or a sigmoidal function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    Parameters
    ----------
    profile : np.ndarray
        DESCRIPTION.
    mode : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    float
        DESCRIPTION.

    """
    params = _function_args_to_list(selected_edge_func)[1:]
    try:
        if selected_edge_func == _sigmoid:

            # Make sure that intensity goes up along ray so that fit can work
            if array[0] > array[-1]:
                array = array[::-1]

            parameter_estimate = [len(array)/2,
                                  max(array),
                                  np.diff(array).mean(),
                                  min(array)]
            optimal_fit_parameters, _covariance = curve_fit(
                selected_edge_func, np.arange(len(array)), array, parameter_estimate
                )

        elif selected_edge_func == _gaussian:
            parameter_estimate = [len(array)/2,
                                  len(array)/2,
                                  max(array)]
            optimal_fit_parameters, _covariance = curve_fit(
                selected_edge_func, np.arange(0, len(array), 1), array, parameter_estimate
                )

        # retrieve errors from covariance matrix
        parameter_error = np.sqrt(np.diag(_covariance))

    # If fit fails, replace bad values with NaN
    except Exception:
        optimal_fit_parameters = np.repeat(np.nan, len(params))
        parameter_error = np.repeat(np.nan, len(params))

    return optimal_fit_parameters, parameter_error
