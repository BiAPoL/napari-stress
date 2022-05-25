# -*- coding: utf-8 -*-
import numpy as np
import vedo

def fibonacci_sphere(samples=1000):
    """
    Scatter n points evenly on the surface of a sphere of radius 1 and center (0,0,0)
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """

    latitude = []
    longitude = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        coordinates = vedo.cart2spher(x, y, z)
        latitude.append(np.rad2deg(coordinates[1]))
        longitude.append(np.rad2deg(coordinates[2]))


    return np.asarray(latitude), np.asarray(longitude)
