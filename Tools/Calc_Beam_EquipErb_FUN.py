import sys
import math as m
import numpy as np
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGridLayout, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QTabWidget, QFileDialog, QMessageBox)
from PyQt5.QtGui import QDoubleValidator, QValidator
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.figure import Figure
# 注意：Vpc_calc模块假设存在，请确保可用
from Vpc_calc import Vpc_calc  

# 物理常量
YITA = 1.7588e11       # 电子电荷质量比
ERB = 8.854e-12        # 真空介电常数 (F/m)
def calculate_dielectric(params, A_list, u_list):
        """
        计算相对介电常数
        
        参数:
        params (dict): 基本参数字典
        A_list (list): A值数组
        u_list (list): u值数组
        
        返回:
        dict: 包含所有计算结果
        """
        # 解包参数
        I = params['I']
        U = params['U']
        f0_GHz = params['f0_GHz']
        Kc = params['Kc']
        w_beam = params['w_beam'] * 1e-3  # mm 转 m
        t_beam = params['t_beam'] * 1e-3  # mm 转 m
        
        # 转换为数组
        A_arr = np.array(A_list)
        u_arr = np.array(u_list)
        
        # 核心计算
        C = m.pow(I * Kc / (4 * U), 1 / 3)
        S = (w_beam * t_beam * np.pi) / 4
        
        numerator = I * np.sqrt(YITA)
        denominator = S * ERB * np.sqrt(2 * U)
        Wp = np.sqrt(numerator / denominator)
        
        Vec = Vpc_calc(U)
        omega = 2 * np.pi * f0_GHz * 1e9
        beta_e = omega / (Vec * 299792458)
        print(beta_e)
        V1z = 2 * C * u_arr * (Vec * 299792458)
        print(V1z)
        Ez = A_arr*np.sqrt(2*C*I*U*2*beta_e**2*Kc)
        print(Ez)
        print(Wp / YITA)
        
        epsilon_rs = 1 - (Wp / YITA) * u_arr/A_arr
        
        # 返回结果
        return {
            'C': C,
            'S': S,
            'Wp': Wp,
            'Vec': Vec,
            'beta_e': beta_e,
            'V1z': V1z,
            'Ez': Ez,
            'epsilon_rs': epsilon_rs,
            'A': A_arr,
            'u': u_arr
        }
