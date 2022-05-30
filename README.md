# napari-stress

This plugin provides tools for the analysis of surfaces in Napari. It provides utilies to determine and refine the surface-representations of objects with a ray-casting approach and calculate surface curvatures. It re-implements code in Napari that was written for [Gross et al. (2021): STRESS, an automated geometrical characterization of deformable particles for in vivo measurements of cell and tissue mechanical stresses](https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1) and has been made public in [this repository](https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1).

## Installation

Create a new conda environment with

```
conda create -n napari-stress Python=3.9
conda activate napari-stress
```

Install a few necessary plugins:

```
conda install -c conda-forge napari jupyterlab
```

To install the plugin, clone the repository and install it:

```
git clone https://github.com/BiAPoL/napari-stress.git
cd napari-stress
pip install -e .
```

## Usage

Functionality in this repository is divided in two groups: **Recipes** and **plugins**.

### Recipes

Napari-stress provides jupyter notebooks with [complete workflows](./docs/notebooks/recipes) for different types of input data. Napari-stress currently provides notebooks for the following data/image types:

* Confocal data (*.tif*), 3D+t: This type of data can be processed with napari-stressed as show in [this notebook](./docs/notebooks/recipes/Process_confocal.ipynb)
* Lightsheet data (*.czi*), 3D + t: coming soon....

### Plugins & functions

All functions in this repository are documented separately as Jupyter notebooks [here](./docs/notebooks/demo)

Data to be used for this plugin is typically of the form `[TZYX]` (e.g., 3D + time). Napari-stress offers some convenient way to extent other function's functionality (which are often made for 3D data) to timelapse data using the `frame_by_frame` function and the `TimelapseConverter` class, both of which are described in more detail in [this notebook]([url](https://github.com/BiAPoL/napari-stress/blob/add-timelapse-decorator-for-points-and-surfaces/docs/notebooks/demo/TimeLapse_processing.ipynb)).


Depending on the set curvature radius, the calculation captures the global or the local curvature.


