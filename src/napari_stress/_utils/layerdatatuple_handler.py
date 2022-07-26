# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:10:45 2022

@author: johamuel
"""
import napari
import numpy as np
import inspect

import napari
from toolz import curry
from typing import Callable, Tuple
from functools import wraps
import time

def my_function() -> ("napari.types.PointsData", dict, dict):
    some_data = np.random.random((10,3))
    metadata = {'attribute': 1}
    features = {'attribute2': np.random.random(10)}
    return some_data, features, metadata

@curry
def layerdatatuple_handler(function: Callable) -> Callable:
    has_viewer_parameter = False

    @wraps(function)
    def worker_function(*args, **kwargs):
        args = list(args)

        # Retrieve the viewer parameter so that we can know which current timepoint is selected
        viewer = None
        for key, value in kwargs.items():
            if isinstance(value, napari.Viewer):
                viewer = value
        if viewer is None:
            for value in args:
                if isinstance(value, napari.Viewer):
                    viewer = value
                    break

        if not has_viewer_parameter:
            if "viewer" in kwargs.keys():
                kwargs.pop("viewer")

        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()



        # call the decorated function
        result = function(*bound.args, **bound.kwargs)

        if viewer is not None and isinstance(sig.return_annotation, tuple):
            data, features, metadata = result

            if metadata is None:
                metadata = {}

            if data is not None:

                # retrieve layertype from return annotation and convert to layerdatatuple format
                # For example: 'napari.types.ImageData' -> 'image'
                layertype = str(sig.return_annotation[0]).replace('Data', '').lower().split('.')[-1]

                meta = {'features': features, 'metadata': metadata}

                layer = napari.layers.Layer.create(data, meta = meta, layer_type=layertype)

        return result

    # If the function has now "viewer" parameter, we add one so that we can read out the current timepoint later
    import inspect
    sig = inspect.signature(worker_function)
    parameters = []
    for name, value in sig.parameters.items():
        if name == "viewer" or name == "napari_viewer" or "napari.viewer.Viewer" in str(value.annotation):
            has_viewer_parameter = True
        parameters.append(value)
    if not has_viewer_parameter:
        parameters.append(inspect.Parameter("viewer", inspect.Parameter.KEYWORD_ONLY, annotation="napari.viewer.Viewer", default=None))

    worker_function.__signature__ = inspect.Signature(parameters, return_annotation=napari.types.LayerDataTuple)

    return worker_function

if __name__ == '__main__':
    viewer = napari.Viewer()
    widget = viewer.window.add_function_widget(annotator(my_function))
