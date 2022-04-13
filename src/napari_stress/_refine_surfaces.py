# -*- coding: utf-8 -*-

import vedo
from napari.types import SurfaceData, ImageData, PointsData

from ._utils import _sigmoid,\
    _gaussian,\
    _func_args_to_list,\
    _detect_drop,\
    _detect_maxima,\
    frame_by_frame

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
    interior = {'fancy': _sigmoid, 'quick': _detect_drop}
    surface = {'fancy': _gaussian, 'quick': _detect_maxima}

@frame_by_frame
def trace_refinement_of_surface(intensity_image: ImageData,
                                surface: SurfaceData,
                                trace_length: float = 2.0,
                                sampling_distance: float = 0.1,
                                selected_fit_type: fit_types = fit_types.fancy_edge_fit,
                                selected_edge: edge_functions = edge_functions.interior,
                                scale: np.ndarray = np.array([1.0, 1.0, 1.0]),
                                show_progress: bool = False,
                                remove_outliers: bool = True,
                                interquartile_factor: float = 1.5
                                )-> PointsData:
    """
    Generate intensity profiles along traces.

    The profiles are interpolated from the input image with linear interpolation
    Parameters
    """
    if isinstance(selected_fit_type, str):
        selected_fit_type = fit_types(selected_fit_type)

    edge_func = selected_edge.value[selected_fit_type.value]

    # Convert to mesh and calculate normals
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.computeNormals()

    # Define start and end points for the surface tracing vectors
    n_samples = int(trace_length/sampling_distance)
    start_pts = mesh.points()/scale[None, :] - 0.5 * trace_length * mesh.pointdata['Normals']

    # Define trace vectors for full length and for single step
    vectors = trace_length * mesh.pointdata['Normals']
    v_step = vectors/n_samples

    # Create coords for interpolator
    X1 = np.arange(0, intensity_image.shape[0], 1)
    X2 = np.arange(0, intensity_image.shape[1], 1)
    X3 = np.arange(0, intensity_image.shape[2], 1)
    rgi = RegularGridInterpolator((X1, X2, X3), intensity_image,
                                  bounds_error=False,
                                  fill_value=intensity_image.min())

    # Allocate arrays for results
    fit_params = _func_args_to_list(edge_func)[1:]
    fit_errors = [p + '_err' for p in fit_params]
    columns = ['surface_points'] + ['idx_of_border'] + ['projection_vector'] +\
        fit_params + fit_errors + ['profiles']

    if len(fit_params) == 1:
        fit_params, fit_errors = fit_params[0], fit_errors[0]

    opt_fit_params = []
    opt_fit_errors = []
    new_surf_points = []
    projection_vectors = []
    idx_of_border = []

    fit_data = pd.DataFrame(columns=columns, index=np.arange(mesh.N()))

    if show_progress:
        tk = tqdm.tqdm(range(mesh.N(), desc = 'Processing vertices...'))
    else:
        tk = range(mesh.N())

    # Iterate over all provided target points
    for idx in tk:

        profile_coords = [start_pts[idx] + k * v_step[idx] for k in range(n_samples)]
        fit_data.loc[idx, 'profiles'] = rgi(profile_coords)

        # Simple or fancy fit?
        if selected_fit_type == fit_types.quick_edge_fit:
            idx_of_border.append(
                edge_func(np.array(fit_data.loc[idx, 'profiles']))
                )
            perror = 0
            popt = 0

        elif selected_fit_type == fit_types.fancy_edge_fit:
            popt, perror = _fancy_edge_fit(np.array(fit_data.loc[idx, 'profiles']),
                                           selected_edge_func=edge_func)
            idx_of_border.append(popt[0])

        opt_fit_errors.append(perror)
        opt_fit_params.append(popt)

        # get new surface point
        new_surf_point = (start_pts[idx] + idx_of_border[idx] * v_step[idx]) * scale
        new_surf_points.append(new_surf_point)
        projection_vectors.append(idx_of_border[idx] * (-1) * v_step[idx])

    fit_data['idx_of_border'] = idx_of_border
    fit_data[fit_params] = opt_fit_params
    fit_data[fit_errors] = opt_fit_errors
    fit_data['surface_points'] = new_surf_points
    fit_data['projection_vector'] = projection_vectors

    # Filter points to remove points with high fit errors
    if remove_outliers:
        fit_data = _remove_outliers_by_index(fit_data, on=fit_errors,
                                             factor=interquartile_factor)

    return np.stack(fit_data['surface_points'].to_numpy())

def _remove_outliers_by_index(df, on=list, factor: float = 1.5) -> pd.DataFrame:
    "Filter all rows that qualify as outliers based on column-statistics."
    if isinstance(on, str):
        on = [on]

    # True if values are good, False if outliers
    df = df.dropna().reset_index()
    indices = np.ones(len(df), dtype=bool)

    for col in on:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        indices[df[df[col] > (Q3 + factor * IQR)].index] = False

    return df[indices]


def _fancy_edge_fit(profile: np.ndarray,
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
    params = _func_args_to_list(selected_edge_func)[1:]
    try:
        if selected_edge_func == _sigmoid:

            # Make sure that intensity goes up along ray so that fit can work
            if profile[0] > profile[-1]:
                profile = profile[::-1]

            p0 = [max(profile),
                  len(profile/2),
                  np.diff(profile).mean(),
                  min(profile)]
            popt, _pcov = curve_fit(
                selected_edge_func, np.arange(len(profile)), profile, p0
                )

        elif selected_edge_func == _gaussian:
            p0 = [len(profile)/2, len(profile)/2, max(profile)]
            popt, _pcov = curve_fit(
                selected_edge_func, np.arange(0, len(profile), 1), profile, p0
                )

        # retrieve errors from covariance matrix
        perr = np.sqrt(np.diag(_pcov))

    # If fit fails:
    except Exception:
        popt = np.repeat(np.nan, len(params))
        perr = np.repeat(np.nan, len(params))

    return popt, perr
