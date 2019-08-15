import os
import os.path
import sys
import json

import matplotlib
matplotlib.use('QT5Agg')  # Ensure using PyQt5 backend
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
import pyqtgraph

import settings
import designs.main_window
import baseline
import singleton



@singleton.singleton
class MainWindow(QtWidgets.QMainWindow, designs.main_window.Ui_MainWindow):
    """A singleton representing main window of the GUI app.

    Attributes:
        There are a bunch of different generated attributes here,
        but they shouldn't be read or written outside this class.
        Communicate with a single instance of this class
        only via its public methods.
    """

    def __init__(self):
        """Initializes main window."""
        pyqtgraph.setConfigOption('background', 'w')
        super().__init__()
        self.setupUi(self)
        self.plot_view.plotItem.showGrid(True, True, 0.7)

        self.btn_load.clicked.connect(self.load_params)
        self.btn_run.clicked.connect(self.run_computations)
        self.btn_save.clicked.connect(self.save_params)
        self.btn_exit.clicked.connect(self.confirm_exit)


    def get_params_from_gui(self):
        """Returns parameters from GUI.

        Retrieves all parameters listed below directly from GUI
        and returns all of them as an instance of class settings.Settings.

        Returns:
            all_settings (class Settings): all settings from GUI
        """
        return settings.Settings({
            'FreqData': {
                'lower_fb': self.lower_fb.value(),
                'upper_fb': self.upper_fb.value(),
                'max_freq': self.max_freq.value()
            },
            'OptimizerSettings': {
                'opt_tol': self.opt_tol.value(),
                'fun_tol': self.fun_tol.value(),
                'stp_tol': self.stp_tol.value(),
                'max_its': self.max_its.value(),
                'sol_mtd': self.sol_mtd.value(),
                'opt_its': self.opt_its.value(),
                'opt_mcp': self.opt_mcp.value()
            },
            'GeneratorParameters': {
                'd_2': self.d_2.value(),
                'e_2': self.e_2.value(),
                'm_2': self.m_2.value(),
                'x_d2': self.x_d2.value(),
                'ic_d2': self.ic_d2.value()
            },
            'OscillationParameters': {
                'osc_amp': self.osc_amp.value(),
                'osc_freq': self.osc_freq.value()
            },
            'Noise': {
                'rnd_amp': self.rnd_amp.value(),
                'snr': self.snr.value()
            },
            'InfBusInitializer': {
                'ic_v1': self.ic_v1.value(),
                'ic_t1': self.ic_t1.value()
            },
            'IntegrationSettings': {
                'df_length': self.df_length.value(),
                'dt_step': self.dt_step.value()
            }
        })


    def set_params_to_gui(self, new_params):
        """Updates parameters in the GUI.

        Args:
            new_params (class Settings): new values of all parameters
                which will be updated in GUI
        """
        self.df_length.setValue(new_params.integration_settings.df_length)
        self.dt_step.setValue(new_params.integration_settings.dt_step)

        self.lower_fb.setValue(new_params.freq_data.lower_fb)
        self.upper_fb.setValue(new_params.freq_data.upper_fb)
        self.max_freq.setValue(new_params.freq_data.max_freq)

        self.opt_tol.setValue(new_params.optimizer_settings.opt_tol)
        self.fun_tol.setValue(new_params.optimizer_settings.fun_tol)
        self.stp_tol.setValue(new_params.optimizer_settings.stp_tol)
        self.max_its.setValue(new_params.optimizer_settings.max_its)
        self.sol_mtd.setValue(new_params.optimizer_settings.sol_mtd)
        self.opt_its.setValue(new_params.optimizer_settings.opt_its)
        self.opt_mcp.setValue(new_params.optimizer_settings.opt_mcp)

        self.d_2.setValue(new_params.generator_parameters.d_2)
        self.e_2.setValue(new_params.generator_parameters.e_2)
        self.m_2.setValue(new_params.generator_parameters.m_2)
        self.x_d2.setValue(new_params.generator_parameters.x_d2)
        self.ic_d2.setValue(new_params.generator_parameters.ic_d2)

        self.osc_amp.setValue(new_params.oscillation_parameters.osc_amp)
        self.osc_freq.setValue(new_params.oscillation_parameters.osc_freq)

        self.rnd_amp.setValue(new_params.noise.rnd_amp)
        self.snr.setValue(new_params.noise.snr)

        self.ic_v1.setValue(new_params.inf_bus_initializer.ic_v1)
        self.ic_t1.setValue(new_params.inf_bus_initializer.ic_t1)


    def load_params(self):
        """Loads parameters from a json file.

        Constructs an absolute path to the directory '../data/workspaces/'
        and asks the user to choose a .json file which contains a workspace
        (bunch of handpicked parameters).
        """
        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        path_to_loading_file = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Choose a file containing a workspace',
            os.path.join(path_to_this_file, '..', 'data', 'workspaces'),
            '*.json'
        )[0]
        if os.path.isfile(path_to_loading_file):
            with open(path_to_loading_file) as params_file:
                new_params = json.load(params_file)
                self.set_params_to_gui(settings.Settings(new_params))


    # TODO: look here !!!!!! REFAAAAACTOR NEED TO DO
    def run_computations(self):
        """Runs computations and drawing plots (not implemented yet)."""
        params_from_gui = self.get_params_from_gui()
        data_to_gui = baseline.run_all_computations(params_from_gui)

        plot_color = pyqtgraph.hsvColor(1, alpha=.9)
        pen = pyqtgraph.mkPen(color=plot_color, width=0.4)
        # plot_title = self.title.text()
        # self.plot_view.plot(data_to_gui['t_vec'], data_to_gui['w2'], pen=pen, clear=True)


    def save_params(self):
        """Saves parameters to a json file.

        Constructs an absolute path to the directory '../data/workspaces/'
        and asks the user to choose a file for saving his current workspace
        (bunch of parameters) using json format.
        """
        data_to_save = self.get_params_from_gui()
        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        path_to_saving_file = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Please, specify a file to save your current workspace',
            os.path.join(path_to_this_file, '..', 'data', 'workspaces')
        )[0]
        if path_to_saving_file:
            with open(path_to_saving_file, 'w') as params_file:
                json.dump(data_to_save.get_values_as_dict(), params_file)


    def closeEvent(self, event):
        """Handles pushing X button of the main window.

        It overrides the method in the base class.
        When a user clicks the X title bar button
        the main window shouldn't be closed immediately.
        We want to ask the user to confirm exiting.
        """
        event.ignore()
        self.confirm_exit()


    def confirm_exit(self):
        """Opens a dialog asking whether exit or not."""
        btn_reply = QtWidgets.QMessageBox.question(
            self, 'Exiting...', "Are you sure you want to exit?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if btn_reply == QtWidgets.QMessageBox.Yes:
            QtWidgets.QApplication.quit()



def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

