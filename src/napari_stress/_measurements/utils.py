# -*- coding: utf-8 -*-
from functools import wraps
import inspect
import napari

def convert(value, new_annotation):
    """Check if an object is of a specific type and convert it if it isn't."""
    from .._stress.manifold_SPB import manifold
    from napari_stress import _METADATAKEY_MANIFOLD

    if isinstance(value, new_annotation):
        return value

    if isinstance(value, napari.layers.Layer) and new_annotation is manifold:
        return value.metadata[_METADATAKEY_MANIFOLD]
    raise ValueError(f'Unsupported conversion {type(value)} -> {new_annotation}')



def naparify_measurement(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        wrapper_sig = inspect.signature(wrapper)

        for key in sig.parameters.keys():
            value = bound.arguments[key]
            original_annotation = sig.parameters[key].annotation
            new_annotation = wrapper_sig.parameters[key].annotation

            print('Old:', original_annotation, 'New:', new_annotation)
            if original_annotation is not new_annotation:
                value = convert(value, original_annotation)
                bound.arguments[key] = value

            result = function(*bound.args, **bound.kwargs)

            return result


    # If we find a manifold parameter in the passed data, we replace its
    # annotation with napari.layers.Points
    import inspect
    from .._stress.manifold_SPB import manifold
    sig = inspect.signature(wrapper)
    parameters = []
    for name, value in sig.parameters.items():
        print(name, str(value.annotation))
        if value.annotation is manifold:
            # replace parameter annotation with napari.layers.Points
            parameters.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="napari.layers.Points"))
        else:
            parameters.append(value)

    wrapper.__signature__ = inspect.Signature(parameters, return_annotation=napari.types.LayerDataTuple)
    return wrapper
