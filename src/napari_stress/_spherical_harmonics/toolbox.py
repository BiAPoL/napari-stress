# -*- coding: utf-8 -*-

from qtpy.QtWidgets import QWidget
from qtpy import uic
from napari_matplotlib.base import NapariMPLWidget

import napari
from napari.layers import Points

import pathlib, os

class spherical_harmonics_toolbox(QWidget):
    """Dockwidget for spherical harmonics analysis."""

    def __init__(self, napari_viewer: napari.viewer.Viewer,
                 points_layer: Points):
        super(spherical_harmonics_toolbox, self).__init__()
        ui_file = os.path.join(pathlib.Path(__file__).parent.absolute(),
                               'toolbox.ui')
        uic.loadUi(ui_file, self) # Load the .ui file
        self.viewer = napari_viewer
        self.points = points_layer.data
        self.features = points_layer.features
        self.metadata = points_layer.metadata

        self._setup_curvature_histogram()
        self.update()

    def _setup_curvature_histogram(self):

        self.histogram_curvature = NapariMPLWidget(self.viewer)
        self.histogram_curvature.axes = self.histogram_curvature.canvas.figure.subplots()

        self.toolBox.setCurrentIndex(0)
        self.toolBox.currentWidget().layout().removeWidget(self.placeholder_curv)
        self.toolBox.currentWidget().layout().addWidget(self.histogram_curvature)

    def update(self):

        self.histogram_curvature.axes.clear()

        self.histogram_curvature.axes.hist(self.features['curvature'], bins=50)
        ylims = self.histogram_curvature.axes.get_ylim()
        avg = self.metadata['averaged_curvature_H0']

        self.histogram_curvature.axes.vlines(avg, 0, ylims[1], linewidth = 5, color='white')
        self.histogram_curvature.axes.text(avg, ylims[1] - 10, f'avg. mean curvature $H_0$ = {avg}')
        self.histogram_curvature.axes.set_ylim(ylims)

        self.histogram_curvature.axes.set_xlabel('Curvature H')
        self.histogram_curvature.axes.set_ylabel('Occurrences [#]')

        self.histogram_curvature.canvas.draw()
