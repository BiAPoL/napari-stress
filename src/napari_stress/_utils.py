import numpy as np
import vedo

import napari
from napari.types import PointsData, SurfaceData, ImageData, LabelsData
import inspect

from functools import wraps
from itertools import chain
import tqdm

def pointcloud_to_vertices4D(surfs: list) -> np.ndarray:

    n_vertices = sum([surf.N() for surf in surfs])
    vertices_4d = np.zeros([n_vertices, 4])

    for idx, surf in enumerate(surfs):
        vertices_4d[idx * surf.N() : idx * surf.N() + surf.N(), 1:] = surf.points()
        vertices_4d[idx * surf.N() : idx * surf.N() + surf.N(), 0] = idx

    return vertices_4d

def vertices4d_to_pointcloud(vertices: np.ndarray) -> list:

    assert vertices.shape[1] == 4

    frames = np.unique(vertices[:, 0])

    surfs =  []
    for idx in frames:
        frame = vertices(np.where(vertices[:, 0] == idx))
        surfs.append(vedo.pointcloud.Points(frame))

    return surfs

def list_of_points_to_pointsdata(points: list) -> PointsData:
    """
    Convert list of pointData objects to single pointsdata object

    Parameters
    ----------
    points : list
        DESCRIPTION.

    Returns
    -------
    PointsData
        DESCRIPTION.
    """
    # First, split list in list of cordinates and properties
    list_of_properties = [pt[1]['properties'] for pt in points]
    list_of_verts = [pt[0] for pt in points]

    # Create time-index for each point in each frame
    time_frames = np.concatenate([
        [idx] * len(timepoint) for idx, timepoint in enumerate(list_of_verts)
        ])

    new_points = np.zeros((len(time_frames), 4))
    new_points[:, 0] = time_frames
    new_points[:, 1:] = np.vstack(list_of_verts)

    new_props = {}
    for key in list(list_of_properties[0].keys()):
        new_props[key] = np.vstack([tp[key] for tp in list_of_properties])

    return (new_points, {'properties': new_props}, 'Points')

def _sigmoid(x, center, amplitude, slope, offset):
    "https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python"
    return amplitude / (1 + np.exp(-slope*(x-center))) + offset

def _gaussian(x, center, sigma, amplitude):
    return amplitude/np.sqrt((2*np.pi*sigma**2)) * np.exp(-(x - center)**2 / (2*sigma**2))

def _detect_maxima(profile, center: float = None):
    return np.argmax(profile)

def _detect_drop(profile, center: float = None):
    return np.argmax(np.abs(np.diff(profile)))

def _func_args_to_list(func: callable) -> list:

    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def frame_by_frame(function, progress_bar: bool = False):

    @wraps(function)
    def wrapper(*args, **kwargs):

        sig = inspect.signature(function)
        annotations = [sig.parameters[key].annotation for key in sig.parameters.keys()]
        return_annotation = sig.return_annotation

        # Dictionary of functions that can convert 4D to list of data
        funcs_data_to_list = {
            napari.types.PointsData: points_to_list_of_points,
            napari.types.SurfaceData: surface_to_list_of_surfaces,
            napari.types.ImageData: image_to_list_of_images,
            napari.types.LabelsData: image_to_list_of_images
            }

        funcs_list_to_data = {
            napari.types.PointsData: list_of_points_to_points,
            napari.types.SurfaceData: list_of_surfaces_to_surface,
            napari.types.ImageData: list_of_images_to_image,
            napari.types.LabelsData: list_of_images_to_image
            }

        supported_data = [ImageData, PointsData, SurfaceData, LabelsData]

        args = list(args)
        n_frames = None

        # Convert 4D data to list(s) of 3D data for every supported argument
        #TODO: Check if objects are actually 4D
        ind_of_framed_arg = []  # remember which arguments were converted

        for idx, arg in enumerate(args):
            if annotations[idx] in supported_data:
                args[idx] = funcs_data_to_list[annotations[idx]](arg)
                ind_of_framed_arg.append(idx)
                n_frames = len(args[idx])

        # apply function frame by frame
        #TODO: Put this in a thread by default?
        results = [None] * n_frames
        it = tqdm.tqdm(range(n_frames)) if progress_bar else range(n_frames)
        for t in it:
            _args = args.copy()

            # Replace argument value by frame t of argument value
            for idx in ind_of_framed_arg:
                _args[idx] = _args[idx][t]

            results[t] = function(*_args, **kwargs)

        if return_annotation is tuple:
            ...

        # Turn results into an array with dimension [result, T, data]
        _results = np.array(results)

        list_of_results = []
        for i, ret_ann in enumerate(return_annotation):
            list_of_results.append(funcs_list_to_data[ret_ann](results[:, i]))

        return
    return wrapper

def list_of_points_to_points(points: list) -> np.ndarray:

    n_points = sum([len(frame) for frame in points])
    t = np.concatenate([[idx] * len(frame) for idx, frame in enumerate(points)])

    points_out = np.zeros((n_points, 4))
    points_out[:, 1:] = np.vstack(points)
    points_out[:, 0] = t

    return points_out

def image_to_list_of_images(image: ImageData) -> list:
    """Convert 4D image to list of images"""
    #TODO: Check if it actually is 4D
    return list(image)

def points_to_list_of_points(points: PointsData) -> list:
    """Convert a 4D point array to list of 3D points"""
    #TODO: Check if it actually is 4D
    n_frames = len(np.unique(points[:, 0]))

    points_out = [None] * n_frames
    for t in range(n_frames):
        points_out[t] = points[points[:, 0] == t, 1:]

    return points_out

def surface_to_list_of_surfaces(surface: SurfaceData) -> list:
    """Convert a 4D surface to list of 3D surfaces"""
    #TODO: Check if it actually is 4D
    points = surface[0]
    faces = np.asarray(surface[1], dtype=int)

    n_frames = len(np.unique(points[:, 0]))
    points_per_frame = [sum(points[:, 0] == t) for t in range(n_frames)]

    idx_face_new_frame = []
    t = 0
    for idx, face in enumerate(faces):
        if points[face[0], 0] == t:
          idx_face_new_frame.append(idx)
          t += 1
    idx_face_new_frame.append(len(faces))

    surfaces = [None] * n_frames
    for t in range(n_frames):
        _points = points[points[:, 0] == t, 1:]
        _faces = faces[idx_face_new_frame[t] : idx_face_new_frame[t+1]-1] - sum(points_per_frame[:t])
        surfaces[t] = (_points, _faces)

    return surfaces

def list_of_images_to_image(images: list) -> ImageData:
    return np.stack(images)

def list_of_surfaces_to_surface(surfs: list) -> tuple:
    """
    Convert vedo surface object to napari-diggestable data format.

    Parameters
    ----------
    surfs : typing.Union[vedo.mesh.Mesh, list]
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if isinstance(surfs[0], vedo.mesh.Mesh):
        surfs = [(s.points(), s.faces()) for s in surfs]

    vertices = [surf[0] for surf in surfs]
    faces = [surf[1] for surf in surfs]
    values = None
    if len(surfs[0]) == 3:
        values = np.concatenate([surf[2] for surf in surfs])

    vertices = list_of_points_to_points(vertices)

    n_verts = 0
    for idx, surf in enumerate(surfs):

        # Offset indices in faces list by previous amount of points
        faces[idx] = n_verts + np.array(faces[idx])

        # Add number of vertices in current surface to n_verts
        n_verts += surf[0].shape[0]

    faces = np.vstack(faces)

    if values is None:
        return (vertices, faces)
    else:
        return (vertices, faces, values)
