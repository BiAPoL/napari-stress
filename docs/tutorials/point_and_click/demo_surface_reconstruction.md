# Surface reconstruction

This tutorial will explain how to perform a surface reconstruction with napari-stress interactively from the napari viewer. This plugin implements the [respective function](https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points.reconstructSurface) from the [vedo](https://vedo.embl.es) library. To get started, open your own data or use the provided sample data from napari-stress ([raw data source](https://github.com/campaslab/STRESS)):

<img src="../../imgs/viewer_screenshots/open_sample_droplet.png" width="45%"> <img src="../../imgs/viewer_screenshots/open_sample_droplet1.png" width="45%">

Select the `Reconstruct surface` function from the plugin menu:

<img src="../../imgs/viewer_screenshots/reconstruct_surface.png" width="50%">

This will bring up the plugin widget:

<img src="../../imgs/viewer_screenshots/reconstruct_surface1.png" width="50%">

## Results

The `radius` controls the search radius of the algorithm: To reconstruct a surface, the function finds all neighboring points for a given point in the pointcloud to be considered for a surface. Setting this value too low will result in a leaky surface:

<img src="../../imgs/viewer_screenshots/reconstruct_surface2.png" width="100%">

In creasing the value will fix this issue:

<img src="../../imgs/viewer_screenshots/reconstruct_surface3.png" width="100%">

Such surfaces typically consist of a large ammount of vertices. This behaviour can be control with the `padding` parameter: Increasing it will simplify the obtained surface representation:

<img src="../../imgs/viewer_screenshots/reconstruct_surface4.png" width="100%">







