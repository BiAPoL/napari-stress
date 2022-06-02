import numpy as np
import inspect

def _sigmoid(array: np.ndarray, center:float, amplitude:float, slope:float, offset:float):
    """
    Sigmoidal fit function
    https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
    """
    return amplitude / (1 + np.exp(-slope*(array-center))) + offset

def _gaussian(array: np.ndarray, center:float, sigma:float, amplitude:float):
    """
    Gaussian normal fit function
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    return amplitude/np.sqrt((2*np.pi*sigma**2)) * np.exp(-(array - center)**2 / (2*sigma**2))

def _detect_maxima(array: np.ndarray, center: float = None):
    """
    Function to find the maximum's index of data with a single peak.
    """
    return np.argmax(array)

def _detect_max_gradient(array: np.ndarray, center: float = None):
    """
    Function to find the location of the steepest gradient for sigmoidal data
    """
    return np.argmax(np.diff(array))

def _function_args_to_list(function: callable) -> list:

    sig = inspect.signature(function)
    return list(sig.parameters.keys())
