import os
import sys
import json
import numpy as np

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib
matplotlib.use('QT5Agg')  # Ensure using PyQt5 backend
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
import pyqtgraph
# import singleton

import settings
import main_window_design


# TODO: make singleton
class MainWindow(QtWidgets.QMainWindow, main_window_design.Ui_MainWindow):
    """A singleton representing main window of the app."""
    def __init__(self):
        pyqtgraph.setConfigOption('background', 'w')  # before loading widget
        """Initialize main window."""
        super().__init__()
        self.setupUi(self)

        self.btn_load.clicked.connect(self._load_params)
        self.btn_run.clicked.connect(self._run_computations)
        self.btn_save.clicked.connect(self._save_params)
        self.btn_exit.clicked.connect(self.exit_app)

        self.plot_view.plotItem.showGrid(True, True, 0.7)

        self.freq_data = settings.FreqData(
            lower_fb=self.lower_fb.value(),
            upper_fb=self.upper_fb.value(),
            max_freq=self.max_freq.value(),
            dt=self.dt.value())

        self.opt_set = settings.OptimizerSettings(
            opt_tol=self.opt_tol.value(),
            fun_tol=self.fun_tol.value(),
            stp_tol=self.stp_tol.value(),
            max_its=self.max_its.value(),
            sol_mtd=self.sol_mtd.value(),
            opt_its=self.opt_its.value(),
            opt_mcp=self.opt_mcp.value())

        self.gen_params = settings.GeneratorParameters(
            d_2=self.d_2.value(),
            e_2=self.e_2.value(),
            m_2=self.m_2.value(),
            x_d2=self.x_d2.value(),
            ic_d2=self.ic_d2.value())

        self.osc_params = settings.OscillationParameters(
            osc_amp=self.d_2.value(),
            osc_freq=self.d_2.value())

        self.white_noise_params = settings.WhiteNoise(
            rnd_amp=self.rnd_amp.value())

        self.inf_bus_params = settings.InfBusInitializer(
            ic_v1=self.ic_v1.value(),
            ic_t1=self.ic_t1.value())


    def _load_params(self):
        # Load parameters from file
        pass


    def _run_computations(self):
        # Run computations and drawing plots
        # plot_title = self.title.text()

        x = np.arange(-20.0, 20.0, 0.05)
        y = x**2 - 2*x + 1.0

        plot_color = pyqtgraph.hsvColor(1, alpha=.5)
        pen = pyqtgraph.mkPen(color=plot_color, width=7)

        self.plot_view.plot(x, y, pen=pen, clear=True)

        # self.label_plot.canvas.ax.plot(x, y)
        # self.label_plot.canvas.draw()


    def _save_params(self):
        # Save parameters to file
        print("Freq Data: ", self.freq_data.get_values())
        print("Optimizer Data: ", self.opt_set.get_values())
        print("Generator Parameters: ", self.gen_params.get_values())
        print("Oscillation Parameters: ", self.osc_params.get_values())
        print("White Noise Params: ", self.white_noise_params.get_values())
        print("Inf Bus Parameters: ", self.inf_bus_params.get_values())


    def exit_app(self):
        """Quit the app."""
        # self.close()
        QtWidgets.QApplication.quit()



def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    app.exec_()


if __name__ == '__main__':
    main()

