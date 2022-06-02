# Spherical harmonics expansion

This tutorial will explain how to perform a spherical harmonics surface expansion with napari-stress interactively from the napari viewer. To get started, open your own data or use the provided sample data from napari-stress ([raw data source](https://github.com/campaslab/STRESS)):

<img src="../../imgs/viewer_screenshots/open_sample_droplet.png" width="45%"> <img src="../../imgs/viewer_screenshots/open_sample_droplet1.png" width="45%">

You can then approximate this pointcloud with a [spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics) expansion. In brief, this fits a set of basis functions to the input pointcloud and returns an analytical representation of the points on the surface. This then allows to sample any number of points on the approximated surface and derive further characteristic surface parameters.

Select the `Fit spherical harmonics` from the plugin menu:

<img src="../../imgs/viewer_screenshots/fit_spherical_harmonics.png" width="45%">

This will bring up a dialogue with the available options:

<img src="../../imgs/viewer_screenshots/fit_spherical_harmonics1.png" width="45%">

The `max_degree` controls the accuracy of the approximation. A higher-degree expansion will lead to a better approximation of the input pointcloud, but will eventually pick up noise and lead to [overfitting](https://en.wikipedia.org/wiki/Overfitting). A lower degree will lead to a smoother surface, but higher remaining errors.

## Results

Applying a spherical harmonics expansion of `max_degree = 5` will lead to the following result:

<img src="../../imgs/viewer_screenshots/fit_spherical_harmonics2.png">

Lowering the value to `max_degree = 0` will lead to only the first *mode* of spherical harmonics being used, which corresponds to fitting a sphere to the pointcloud:

<img src="../../imgs/viewer_screenshots/fit_spherical_harmonics3.png">

**Layer display settings**

napari may default to innapropriate settings for the pointcloud (e.g., inproper point size or color). To change this, select the malformated layer, click the `select_points` icon in the top left:

<img src="../../imgs/viewer_screenshots/change_layer_settings.png" width="45%">

Then, drag a box over all points and change the point size and face color with the slider and the color selecter, respectively. Alternatively, you can also open the code terminal with the icon in the bottom-left:

<img src="../../imgs/viewer_screenshots/open_terminal.png" width="45%">

and type:

```Python
viewer.layers[-1].size = 0.5
viewer.layers[-1].face_color = 'cyan'
```
