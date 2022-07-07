# -*- coding: utf-8 -*-

from qtpy.QtWidgets import QWidget
from qtpy import uic
from napari_matplotlib.base import NapariMPLWidget

import napari
from napari.layers import Points

from .spherical_harmonics import calculate_power_spectrum

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
        self._first_time_setup()

        self.update_curvature_histogram()
        self.update_power_spectrum()
        self.update_gauss_bonnet()

        self.things_to_update_in_tab = {
            0: self.update_curvature_histogram,
            1: self.update_power_spectrum,
            2: self.update_gauss_bonnet
            }
        
        self._setup_callbacks()
        
    def update_plots(self):
        """Update things in the currently selected tab"""
        selected_tab = self.toolBox.currentIndex()
        print(f'Updating tab {selected_tab}: {str(self.things_to_update_in_tab[selected_tab])}')
        self.things_to_update_in_tab[selected_tab]()

    def _setup_callbacks(self):
        """Disconnect some default signals and hook up own signals"""
        self.viewer.dims.events.current_step.disconnect(self.histogram_curvature._draw)
        self.viewer.layers.selection.events.changed.disconnect(self.histogram_curvature.update_layers)

        self.viewer.dims.events.current_step.connect(self.update_plots)
        self.toolBox.currentChanged.connect(self.update_plots)
        self.button_run_measurement.clicked.connect(self.update_plots)

    def _get_data_from_viewer(self):
        """Find the associated layer with this plugin in the viewer"""

        self.layer = self.viewer.layers[self.layer_name]

    def _first_time_setup(self):
        """First time setup of curvature histogram plot"""

        # Curvature histogram
        self.histogram_curvature = NapariMPLWidget(self.viewer)
        self.histogram_curvature.axes = self.histogram_curvature.canvas.figure.subplots()
        # self.histogram_curvature.n_selected_layers = 0

        self.toolBox.setCurrentIndex(0)
        self.toolBox.currentWidget().layout().removeWidget(self.placeholder_curv)
        self.toolBox.currentWidget().layout().addWidget(self.histogram_curvature)

        # Power spectrum
        self.plot_power_spectrum = NapariMPLWidget(self.viewer)
        self.plot_power_spectrum.axes = self.plot_power_spectrum.canvas.figure.subplots()
        # self.plot_power_spectrum.n_selected_layers = 0

        self.toolBox.setCurrentIndex(1)
        self.toolBox.currentWidget().layout().removeWidget(self.placeholder_spectrum)
        self.toolBox.currentWidget().layout().addWidget(self.plot_power_spectrum)
        
    def update_gauss_bonnet(self):
        self._get_data_from_viewer()
        relative_error = self.layer.metadata['gauss_bonnet_relative_error']
        self.gauss_bonnet_relative_error.setText('{:.5E}'.format(relative_error))        

    def update_power_spectrum(self):
        self._get_data_from_viewer()
        self.plot_power_spectrum.axes.clear()

        coefficients = self.layer.metadata['spherical_harmonics_coefficients']
        power_spectra = calculate_power_spectrum(coefficients)

        for p in power_spectra:
            self.plot_power_spectrum.axes.plot(p)

        self.plot_power_spectrum.axes.set_xlabel('Degree l')
        self.plot_power_spectrum.axes.set_ylabel('Power per degree')
        self.plot_power_spectrum.axes.set_yscale('log')
        self.plot_power_spectrum.axes.grid(which='major', color='white',
                                           linestyle='--', alpha=0.7)

        self.plot_power_spectrum.canvas.draw()

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

        self.histogram_curvature.canvas.draw()
