# -*- coding: utf-8 -*-

from qtpy.QtWidgets import QWidget
from qtpy import uic
from napari_matplotlib.base import NapariMPLWidget

import napari
from napari.layers import Points

import pathlib, os

import numpy as np

class spherical_harmonics_toolbox(QWidget):
    """Dockwidget for spherical harmonics analysis."""

    def __init__(self, napari_viewer: napari.viewer.Viewer,
                 points_layer: Points):
        super(spherical_harmonics_toolbox, self).__init__()
        ui_file = os.path.join(pathlib.Path(__file__).parent.absolute(),
                               'toolbox.ui')
        uic.loadUi(ui_file, self) # Load the .ui file
        self.viewer = napari_viewer
        self.layer_name = points_layer.name

        self._get_data_from_viewer()
        self._setup_curvature_histogram()
        self.update_curvature_histogram()

        self._setup_callbacks()

    def _setup_callbacks(self):
        self.viewer.dims.events.current_step.disconnect(self.histogram_curvature._draw)
        self.viewer.layers.selection.events.changed.disconnect(self.histogram_curvature.update_layers)

        self.viewer.dims.events.current_step.connect(self.update_curvature_histogram)

    def _get_data_from_viewer(self):

        self.layer = self.viewer.layers[self.layer_name]

    def _setup_curvature_histogram(self):

        self.histogram_curvature = NapariMPLWidget(self.viewer)
        self.histogram_curvature.axes = self.histogram_curvature.canvas.figure.subplots()
        self.histogram_curvature.n_layers_input = 1

        self.toolBox.setCurrentIndex(0)
        self.toolBox.currentWidget().layout().removeWidget(self.placeholder_curv)
        self.toolBox.currentWidget().layout().addWidget(self.histogram_curvature)

    def update_curvature_histogram(self):

        self._get_data_from_viewer()
        self.histogram_curvature.axes.clear()

        colormapping = self.layer.face_colormap

        N, bins, patches = self.histogram_curvature.axes.hist(
            self.layer.features['curvature'],
            edgecolor='white',
            linewidth=1)

        bins_norm = (bins - bins.min())/(bins.max() - bins.min())
        colors = colormapping.map(bins_norm)
        for idx, patch in enumerate(patches):
            patch.set_facecolor(colors[idx])
        ylims = self.histogram_curvature.axes.get_ylim()
        avg = self.layer.metadata['averaged_curvature_H0']

        self.histogram_curvature.axes.vlines(avg, 0, ylims[1], linewidth = 5, color='white')
        self.histogram_curvature.axes.text(avg, ylims[1] - 10, f'avg. mean curvature $H_0$ = {avg}')
        self.histogram_curvature.axes.set_ylim(ylims)

        self.histogram_curvature.axes.set_xlabel('Curvature H')
        self.histogram_curvature.axes.set_ylabel('Occurrences [#]')

        self.histogram_curvature.canvas.figure.tight_layout()
        self.histogram_curvature.canvas.draw()
