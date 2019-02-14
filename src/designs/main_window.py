# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/qt/main_window.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 688)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout.setObjectName("formLayout")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_5)
        self.label_lower_fb = QtWidgets.QLabel(self.centralwidget)
        self.label_lower_fb.setObjectName("label_lower_fb")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_lower_fb)
        self.lower_fb = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.lower_fb.setDecimals(3)
        self.lower_fb.setMaximum(9.999)
        self.lower_fb.setSingleStep(0.001)
        self.lower_fb.setProperty("value", 1.99)
        self.lower_fb.setObjectName("lower_fb")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lower_fb)
        self.label_upper_fb = QtWidgets.QLabel(self.centralwidget)
        self.label_upper_fb.setObjectName("label_upper_fb")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_upper_fb)
        self.upper_fb = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.upper_fb.setDecimals(3)
        self.upper_fb.setMaximum(9.999)
        self.upper_fb.setSingleStep(0.001)
        self.upper_fb.setProperty("value", 2.01)
        self.upper_fb.setObjectName("upper_fb")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.upper_fb)
        self.label_max_freq = QtWidgets.QLabel(self.centralwidget)
        self.label_max_freq.setObjectName("label_max_freq")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_max_freq)
        self.max_freq = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.max_freq.setDecimals(3)
        self.max_freq.setMaximum(99.999)
        self.max_freq.setSingleStep(0.001)
        self.max_freq.setProperty("value", 6.0)
        self.max_freq.setObjectName("max_freq")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.max_freq)
        self.label_dt = QtWidgets.QLabel(self.centralwidget)
        self.label_dt.setObjectName("label_dt")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_dt)
        self.dt = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dt.setDecimals(3)
        self.dt.setMaximum(9.999)
        self.dt.setSingleStep(0.001)
        self.dt.setProperty("value", 0.005)
        self.dt.setObjectName("dt")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.dt)
        self.verticalLayout.addLayout(self.formLayout)
        self.verticalLayout_7.addLayout(self.verticalLayout)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_7.addWidget(self.line_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_osc_amp = QtWidgets.QLabel(self.centralwidget)
        self.label_osc_amp.setObjectName("label_osc_amp")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_osc_amp)
        self.osc_amp = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.osc_amp.setDecimals(3)
        self.osc_amp.setMaximum(9.999)
        self.osc_amp.setSingleStep(0.001)
        self.osc_amp.setProperty("value", 2.0)
        self.osc_amp.setObjectName("osc_amp")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.osc_amp)
        self.label_osc_freq = QtWidgets.QLabel(self.centralwidget)
        self.label_osc_freq.setObjectName("label_osc_freq")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_osc_freq)
        self.osc_freq = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.osc_freq.setDecimals(3)
        self.osc_freq.setMaximum(9.999)
        self.osc_freq.setSingleStep(0.001)
        self.osc_freq.setProperty("value", 0.005)
        self.osc_freq.setObjectName("osc_freq")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.osc_freq)
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_22)
        self.verticalLayout_4.addLayout(self.formLayout_4)
        self.verticalLayout_7.addLayout(self.verticalLayout_4)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_7.addWidget(self.line_2)
        self.formLayout_6 = QtWidgets.QFormLayout()
        self.formLayout_6.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout_6.setObjectName("formLayout_6")
        self.label_ic_v1 = QtWidgets.QLabel(self.centralwidget)
        self.label_ic_v1.setObjectName("label_ic_v1")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_ic_v1)
        self.ic_v1 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.ic_v1.setDecimals(1)
        self.ic_v1.setMaximum(9.9)
        self.ic_v1.setSingleStep(0.1)
        self.ic_v1.setProperty("value", 1.0)
        self.ic_v1.setObjectName("ic_v1")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.ic_v1)
        self.label_ic_t1 = QtWidgets.QLabel(self.centralwidget)
        self.label_ic_t1.setObjectName("label_ic_t1")
        self.formLayout_6.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_ic_t1)
        self.ic_t1 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.ic_t1.setDecimals(1)
        self.ic_t1.setMaximum(9.9)
        self.ic_t1.setSingleStep(0.1)
        self.ic_t1.setProperty("value", 0.5)
        self.ic_t1.setObjectName("ic_t1")
        self.formLayout_6.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.ic_t1)
        self.label_27 = QtWidgets.QLabel(self.centralwidget)
        self.label_27.setAlignment(QtCore.Qt.AlignCenter)
        self.label_27.setObjectName("label_27")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_27)
        self.verticalLayout_7.addLayout(self.formLayout_6)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_7.addWidget(self.line)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout_5.setContentsMargins(-1, -1, -1, 40)
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_23)
        self.label_rnd_amp = QtWidgets.QLabel(self.centralwidget)
        self.label_rnd_amp.setObjectName("label_rnd_amp")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_rnd_amp)
        self.rnd_amp = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.rnd_amp.setDecimals(3)
        self.rnd_amp.setMaximum(9.999)
        self.rnd_amp.setSingleStep(0.001)
        self.rnd_amp.setProperty("value", 0.02)
        self.rnd_amp.setObjectName("rnd_amp")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.rnd_amp)
        self.verticalLayout_6.addLayout(self.formLayout_5)
        self.verticalLayout_7.addLayout(self.verticalLayout_6)
        self.horizontalLayout.addLayout(self.verticalLayout_7)
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.horizontalLayout.addWidget(self.line_5)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_opt_tol = QtWidgets.QLabel(self.centralwidget)
        self.label_opt_tol.setObjectName("label_opt_tol")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_opt_tol)
        self.opt_tol = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.opt_tol.setDecimals(6)
        self.opt_tol.setMinimum(-10.0)
        self.opt_tol.setMaximum(10.0)
        self.opt_tol.setSingleStep(1e-06)
        self.opt_tol.setProperty("value", 1e-06)
        self.opt_tol.setObjectName("opt_tol")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.opt_tol)
        self.label_fun_tol = QtWidgets.QLabel(self.centralwidget)
        self.label_fun_tol.setObjectName("label_fun_tol")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_fun_tol)
        self.fun_tol = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.fun_tol.setDecimals(6)
        self.fun_tol.setMinimum(-10.0)
        self.fun_tol.setMaximum(10.0)
        self.fun_tol.setSingleStep(1e-06)
        self.fun_tol.setProperty("value", 1e-06)
        self.fun_tol.setObjectName("fun_tol")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.fun_tol)
        self.label_stp_tol = QtWidgets.QLabel(self.centralwidget)
        self.label_stp_tol.setObjectName("label_stp_tol")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_stp_tol)
        self.stp_tol = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.stp_tol.setDecimals(6)
        self.stp_tol.setMinimum(-10.0)
        self.stp_tol.setMaximum(10.0)
        self.stp_tol.setSingleStep(1e-06)
        self.stp_tol.setProperty("value", 1e-06)
        self.stp_tol.setObjectName("stp_tol")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.stp_tol)
        self.label_max_its = QtWidgets.QLabel(self.centralwidget)
        self.label_max_its.setObjectName("label_max_its")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_max_its)
        self.max_its = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.max_its.setDecimals(0)
        self.max_its.setMinimum(0.0)
        self.max_its.setMaximum(10000.0)
        self.max_its.setSingleStep(1.0)
        self.max_its.setProperty("value", 1.0)
        self.max_its.setObjectName("max_its")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.max_its)
        self.label_sol_mtd = QtWidgets.QLabel(self.centralwidget)
        self.label_sol_mtd.setObjectName("label_sol_mtd")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_sol_mtd)
        self.sol_mtd = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.sol_mtd.setDecimals(0)
        self.sol_mtd.setMaximum(10000.0)
        self.sol_mtd.setProperty("value", 3.0)
        self.sol_mtd.setObjectName("sol_mtd")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.sol_mtd)
        self.label_opt_its = QtWidgets.QLabel(self.centralwidget)
        self.label_opt_its.setObjectName("label_opt_its")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_opt_its)
        self.opt_its = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.opt_its.setDecimals(0)
        self.opt_its.setMaximum(10000.0)
        self.opt_its.setProperty("value", 50.0)
        self.opt_its.setObjectName("opt_its")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.opt_its)
        self.label_opt_mcp = QtWidgets.QLabel(self.centralwidget)
        self.label_opt_mcp.setObjectName("label_opt_mcp")
        self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_opt_mcp)
        self.opt_mcp = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.opt_mcp.setDecimals(3)
        self.opt_mcp.setMaximum(99.999)
        self.opt_mcp.setSingleStep(0.001)
        self.opt_mcp.setProperty("value", 0.01)
        self.opt_mcp.setObjectName("opt_mcp")
        self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.opt_mcp)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_6)
        self.verticalLayout_2.addLayout(self.formLayout_2)
        self.verticalLayout_8.addLayout(self.verticalLayout_2)
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout_8.addWidget(self.line_4)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_19)
        self.label_d_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_d_2.setObjectName("label_d_2")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_d_2)
        self.d_2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.d_2.setSingleStep(0.01)
        self.d_2.setProperty("value", 0.25)
        self.d_2.setObjectName("d_2")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.d_2)
        self.label_e_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_e_2.setObjectName("label_e_2")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_e_2)
        self.e_2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.e_2.setSingleStep(0.01)
        self.e_2.setProperty("value", 1.0)
        self.e_2.setObjectName("e_2")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.e_2)
        self.label_m_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_m_2.setObjectName("label_m_2")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_m_2)
        self.m_2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.m_2.setSingleStep(0.01)
        self.m_2.setProperty("value", 1.0)
        self.m_2.setObjectName("m_2")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.m_2)
        self.label_x_d2 = QtWidgets.QLabel(self.centralwidget)
        self.label_x_d2.setObjectName("label_x_d2")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_x_d2)
        self.x_d2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.x_d2.setSingleStep(0.01)
        self.x_d2.setProperty("value", 0.01)
        self.x_d2.setObjectName("x_d2")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.x_d2)
        self.label_ic_d2 = QtWidgets.QLabel(self.centralwidget)
        self.label_ic_d2.setObjectName("label_ic_d2")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_ic_d2)
        self.ic_d2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.ic_d2.setSingleStep(0.01)
        self.ic_d2.setProperty("value", 1.0)
        self.ic_d2.setObjectName("ic_d2")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.ic_d2)
        self.verticalLayout_3.addLayout(self.formLayout_3)
        self.verticalLayout_8.addLayout(self.verticalLayout_3)
        self.line_7 = QtWidgets.QFrame(self.centralwidget)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.verticalLayout_8.addWidget(self.line_7)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_14.setContentsMargins(-1, -1, -1, 40)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.formLayout_7 = QtWidgets.QFormLayout()
        self.formLayout_7.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout_7.setObjectName("formLayout_7")
        self.label_df_length = QtWidgets.QLabel(self.centralwidget)
        self.label_df_length.setObjectName("label_df_length")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_df_length)
        self.label_dt_step = QtWidgets.QLabel(self.centralwidget)
        self.label_dt_step.setObjectName("label_dt_step")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_dt_step)
        self.df_length = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.df_length.setMaximum(999.99)
        self.df_length.setProperty("value", 100.0)
        self.df_length.setObjectName("df_length")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.df_length)
        self.dt_step = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dt_step.setSingleStep(0.01)
        self.dt_step.setProperty("value", 0.05)
        self.dt_step.setObjectName("dt_step")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.dt_step)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_3)
        self.verticalLayout_14.addLayout(self.formLayout_7)
        self.verticalLayout_8.addLayout(self.verticalLayout_14)
        self.horizontalLayout.addLayout(self.verticalLayout_8)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.verticalLayout_5.addWidget(self.line_6)
        self.verticalLayout_9.addLayout(self.verticalLayout_5)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setObjectName("btn_load")
        self.horizontalLayout_2.addWidget(self.btn_load)
        self.btn_run = QtWidgets.QPushButton(self.centralwidget)
        self.btn_run.setObjectName("btn_run")
        self.horizontalLayout_2.addWidget(self.btn_run)
        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setObjectName("btn_save")
        self.horizontalLayout_2.addWidget(self.btn_save)
        self.verticalLayout_10.addLayout(self.horizontalLayout_2)
        self.btn_exit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_exit.setObjectName("btn_exit")
        self.verticalLayout_10.addWidget(self.btn_exit)
        self.verticalLayout_9.addLayout(self.verticalLayout_10)
        self.horizontalLayout_3.addLayout(self.verticalLayout_9)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem)
        self.plot_view = PlotWidget(self.centralwidget)
        self.plot_view.setObjectName("plot_view")
        self.horizontalLayout_3.addWidget(self.plot_view)
        self.verticalLayout_11.addLayout(self.horizontalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_5.setText(_translate("MainWindow", "FreqData"))
        self.label_lower_fb.setText(_translate("MainWindow", "lower_fb"))
        self.label_upper_fb.setText(_translate("MainWindow", "upper_fb"))
        self.label_max_freq.setText(_translate("MainWindow", "max_freq"))
        self.label_dt.setText(_translate("MainWindow", "dt"))
        self.label_osc_amp.setText(_translate("MainWindow", "osc_amp"))
        self.label_osc_freq.setText(_translate("MainWindow", "osc_freq"))
        self.label_22.setText(_translate("MainWindow", "OscillationParameters"))
        self.label_ic_v1.setText(_translate("MainWindow", "ic_v1"))
        self.label_ic_t1.setText(_translate("MainWindow", "ic_t1"))
        self.label_27.setText(_translate("MainWindow", "InfBusInitializer"))
        self.label_23.setText(_translate("MainWindow", "WhiteNoise"))
        self.label_rnd_amp.setText(_translate("MainWindow", "rnd_amp"))
        self.label_opt_tol.setText(_translate("MainWindow", "opt_tol"))
        self.label_fun_tol.setText(_translate("MainWindow", "fun_tol"))
        self.label_stp_tol.setText(_translate("MainWindow", "stp_tol"))
        self.label_max_its.setText(_translate("MainWindow", "max_its"))
        self.label_sol_mtd.setText(_translate("MainWindow", "sol_mtd"))
        self.label_opt_its.setText(_translate("MainWindow", "opt_its"))
        self.label_opt_mcp.setText(_translate("MainWindow", "opt_mcp"))
        self.label_6.setText(_translate("MainWindow", "OptimizerSettings"))
        self.label_19.setText(_translate("MainWindow", "GeneratorParameters"))
        self.label_d_2.setText(_translate("MainWindow", "d_2"))
        self.label_e_2.setText(_translate("MainWindow", "e_2"))
        self.label_m_2.setText(_translate("MainWindow", "m_2"))
        self.label_x_d2.setText(_translate("MainWindow", "x_d2"))
        self.label_ic_d2.setText(_translate("MainWindow", "ic_d2"))
        self.label_df_length.setText(_translate("MainWindow", "df_length"))
        self.label_dt_step.setText(_translate("MainWindow", "dt_step"))
        self.label_3.setText(_translate("MainWindow", "IntegrationSettings"))
        self.btn_load.setText(_translate("MainWindow", "Load"))
        self.btn_run.setText(_translate("MainWindow", "Run"))
        self.btn_save.setText(_translate("MainWindow", "Save"))
        self.btn_exit.setText(_translate("MainWindow", "Exit"))


from pyqtgraph import PlotWidget
