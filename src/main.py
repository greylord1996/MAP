import os
import os.path
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('QT5Agg')  # Ensure using PyQt5 backend
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
import pyqtgraph

import settings
import designs.main_window
import utils
import form_initial_data

import time


@utils.singleton
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

        # params = self.get_params_from_gui()
        # self._freq_data = settings.FreqData(**params['FreqData'])
        # self._opt_set = settings.OptimizerSettings(**params['OptimizerSettings'])
        # self._gen_params = settings.GeneratorParameters(**params['GeneratorParameters'])
        # self._osc_params = settings.OscillationParameters(**params['OscillationParameters'])
        # self._white_noise_params = settings.WhiteNoise(**params['WhiteNoise'])
        # self._inf_bus_params = settings.InfBusInitializer(**params['InfBusInitializer'])


    def get_params_from_gui(self):
        """Returns a dict containing parameters from GUI.

        Retrieves all parameters listed below directly from GUI
        and returns all of them in a two-level dictionary.
        """
        return {
            'FreqData': {
                'lower_fb': self.lower_fb.value(),
                'upper_fb': self.upper_fb.value(),
                'max_freq': self.max_freq.value(),
                'dt': self.dt.value(),
            },
            'OptimizerSettings': {
                'opt_tol': self.opt_tol.value(),
                'fun_tol': self.fun_tol.value(),
                'stp_tol': self.stp_tol.value(),
                'max_its': self.max_its.value(),
                'sol_mtd': self.sol_mtd.value(),
                'opt_its': self.opt_its.value(),
                'opt_mcp': self.opt_mcp.value(),
            },
            'GeneratorParameters': {
                'd_2': self.d_2.value(),
                'e_2': self.e_2.value(),
                'm_2': self.m_2.value(),
                'x_d2': self.x_d2.value(),
                'ic_d2': self.ic_d2.value(),
            },
            'OscillationParameters': {
                'osc_amp': self.osc_amp.value(),
                'osc_freq': self.osc_freq.value(),
            },
            'WhiteNoise': {
                'rnd_amp': self.rnd_amp.value(),
            },
            'InfBusInitializer': {
                'ic_v1': self.ic_v1.value(),
                'ic_t1': self.ic_t1.value(),
            },
            'IntegrationSettings': {
                'df_length': self.df_length.value(),
                'dt_step': self.dt_step.value(),
            }
        }


    def set_params_to_gui(self, new_params):
        """Updates params in GUI.

        Args:
            new_params (dict of dicts): new values of the all parameters
                which will be updated in GUI
        """
        # self._freq_data.set_values(new_params['FreqData'])
        # self._opt_set.set_values(new_params['OptimizerSettings'])
        # self._gen_params.set_values(new_params['GeneratorParameters'])
        # self._osc_params.set_values(new_params['OscillationParameters'])
        # self._white_noise_params.set_values(new_params['WhiteNoise'])
        # self._inf_bus_params.set_values(new_params['InfBusInitializer'])

        self.lower_fb.setValue(new_params['FreqData']['lower_fb'])
        self.upper_fb.setValue(new_params['FreqData']['upper_fb'])
        self.max_freq.setValue(new_params['FreqData']['max_freq'])
        self.dt.setValue(new_params['FreqData']['dt'])

        self.opt_tol.setValue(new_params['OptimizerSettings']['opt_tol'])
        self.fun_tol.setValue(new_params['OptimizerSettings']['fun_tol'])
        self.stp_tol.setValue(new_params['OptimizerSettings']['stp_tol'])
        self.max_its.setValue(new_params['OptimizerSettings']['max_its'])
        self.sol_mtd.setValue(new_params['OptimizerSettings']['sol_mtd'])
        self.opt_its.setValue(new_params['OptimizerSettings']['opt_its'])
        self.opt_mcp.setValue(new_params['OptimizerSettings']['opt_mcp'])

        self.d_2.setValue(new_params['GeneratorParameters']['d_2'])
        self.e_2.setValue(new_params['GeneratorParameters']['e_2'])
        self.m_2.setValue(new_params['GeneratorParameters']['m_2'])
        self.x_d2.setValue(new_params['GeneratorParameters']['x_d2'])
        self.ic_d2.setValue(new_params['GeneratorParameters']['ic_d2'])

        self.osc_amp.setValue(new_params['OscillationParameters']['osc_amp'])
        self.osc_freq.setValue(new_params['OscillationParameters']['osc_freq'])

        self.rnd_amp.setValue(new_params['WhiteNoise']['rnd_amp'])

        self.ic_v1.setValue(new_params['InfBusInitializer']['ic_v1'])
        self.ic_t1.setValue(new_params['InfBusInitializer']['ic_t1'])

        self.df_length.setValue(new_params['IntegrationSettings']['df_length'])
        self.dt_step.setValue(new_params['IntegrationSettings']['dt_step'])


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
            with open(path_to_loading_file) as params_input:
                new_params = json.load(params_input)
                self.set_params_to_gui(new_params)

    # look here !!!!!! REFAAAAACTOR NEED TO DO
    def run_computations(self):
        """Runs computations and drawing plots (not implemented yet)."""
        # self.get_params_from_gui()
        # plot_title = self.title.text()
        a = self.get_params_from_gui()
        print(a['WhiteNoise'])

        b = form_initial_data.OdeSolver(a['WhiteNoise'], a['GeneratorParameters'],
                                        a['OscillationParameters'], a['IntegrationSettings'])
        b.solve()
        #b.show_T1t_in_test_mode()
        b.show_V1t_in_test_mode()
        c = b.get_appropr_data_to_gui()
        #x = np.arange(-20.0, 20.0, 0.05)
        #y = x**2 - 2*x + 1.0
        print("###")
        plot_color = pyqtgraph.hsvColor(1, alpha=.9)
        pen = pyqtgraph.mkPen(color=plot_color, width=0.4)
        self.plot_view.plot(c['t_vec'], c['w2'], pen=pen, clear=True)


    def save_params(self):
        """Saves parameters to a file.

        Constructs an absolute path to the directory '../data/workspaces/'
        and asks the user to choose a file for saving his current workspace
        (bunch of parameters) using json format.
        """
        data_to_save = self.get_params_from_gui()
        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        path_to_saving_file = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Please, specify the file for saving your current workspace',
            os.path.join(path_to_this_file, '..', 'data', 'workspaces')
        )[0]
        if path_to_saving_file:
            with open(path_to_saving_file, 'w') as params_output:
                json.dump(data_to_save, params_output)


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
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()

