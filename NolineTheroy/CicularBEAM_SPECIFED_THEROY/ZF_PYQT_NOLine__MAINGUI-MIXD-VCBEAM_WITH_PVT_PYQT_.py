import sys
import os
import json
import csv
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QMessageBox, QGridLayout, QHeaderView,
    QSplitter, QTextEdit, QProgressBar, QStackedWidget, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from Noline_GAIN_MAINCALL_VCBEAMCORE_SUPER_MIX_WITH_PVT import calculate_SEGMENT_TWT_NOLINE

font_list = [f.name for f in fm.fontManager.ttflist]
cn_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'FangSong']
available_cn_fonts = [f for f in cn_fonts if any(f in font for font in font_list)]

plt.rcParams['font.sans-serif'] = available_cn_fonts
plt.rcParams['axes.unicode_minus'] = False



# 常量定义
CONFIG_DIR = "config"
RESULTS_DIR = "Results"
TABLE_COLUMNS = ["Kc (Ω)", "Loss_perunit", "Freq (GHz)", "Vpc (c)"]

# 参数配置（添加了参数调整系数）
PARAM_CONFIG = {
    'i': {'label': '电流 I (A)', 'default': '0.3', 
          'tooltip': '电子枪发射电流，单位为安培(A)'},
    'v': {'label': '电压 V (V)', 'default': '23000', 
          'tooltip': '电子枪加速电压，单位为伏特(V)'},
    'p_sws': {'label': '周期长度 p_SWS (mm)\n(逗号分隔)', 'default': '0.5,0.5,0.5,0.5',
              'tooltip': '各段慢波结构的周期长度，以逗号分隔的浮点数列表'},
    'n_unit': {'label': '周期数 N_Unit\n(逗号分隔)', 'default': '25,7,12,13',
               'tooltip': '各段的周期数，与周期长度一一对应'},
    'w': {'label': '束流宽度 w (mm)', 'default': '0.2',
          'tooltip': '电子束在水平方向上的宽度'},
    't': {'label': '束流厚度 t (mm)', 'default': '0.2',
          'tooltip': '电子束在垂直方向上的厚度'},
    'Fn_K': {'label': 'Fn_K 参数', 'default': '1',
             'tooltip': '填充率倒数，对于圆形束流(w=t)此值大于1，对于带状束流用于矫正'},
    'p_in': {'label': '输入功率 P_in (W)', 'default': '0.1',
             'tooltip': '输入信号的功率'},
    'loss_attu': {'label': '衰减量 (dB)', 'default': '20',
                  'tooltip': '衰减段引入的衰减量'},
    'section_seg_idx': {'label': '衰减段索引\n(逗号分隔)', 'default': '1',
                       'tooltip': '指示哪些段是衰减段，以逗号分隔的整数索引（从0开始）'},
    'vpc_coeff': {'label': 'Vpc 调整系数', 'default': '0.3',
                  'tooltip': '用于计算Vpc随慢波结构周期变化的调整系数\n公式: Vpc_new = Vpc + coeff * (p_current - p_first)/p_first * Vpc'},
    'kc_coeff': {'label': 'Kc 调整系数', 'default': '0.9',
                 'tooltip': '用于计算Kc随慢波结构周期变化的调整系数\n公式: Kc_new = Kc + coeff * (p_current - p_first)/p_first * Kc'},
}

class ParamWidget(QWidget):
    """参数输入组件"""
    def __init__(self, param_key, parent=None):
        super().__init__(parent)
        self.param_key = param_key
        config = PARAM_CONFIG[param_key]
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        
        self.label = QLabel(config['label'])
        
        # 添加工具提示
        if 'tooltip' in config:
            self.label.setToolTip(config['tooltip'])
            self.label.setWhatsThis(config['tooltip'])
        
        self.line_edit = QLineEdit(config['default'])
        self.line_edit.setFixedWidth(120)
        
        # 添加工具提示到输入框
        if 'tooltip' in config:
            self.line_edit.setToolTip(config['tooltip'])
            self.line_edit.setWhatsThis(config['tooltip'])
        
        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.line_edit)

    @property
    def value(self):
        """获取当前参数值"""
        return self.line_edit.text().strip()


class WorkerThread(QThread):
    """计算工作线程"""
    progress_updated = pyqtSignal(int, str)
    calculation_finished = pyqtSignal(list, dict)
    calculation_error = pyqtSignal(str)
    message = pyqtSignal(str)

    def __init__(self, fixed_params, var_params, para_func):
        super().__init__()
        self.fixed_params = fixed_params
        self.var_params = var_params
        self.para_func = para_func
        self._is_running = True

    def run(self):
        """执行计算任务"""
        try:
            total = len(self.var_params)
            results = []
            
            for idx, params in enumerate(self.var_params):
                if not self._is_running:
                    self.message.emit("计算已取消")
                    return
                    
                self.message.emit(f"\n计算点 {idx+1}/{total}: ")
                self.message.emit(f"  频率: {params['Freq']} GHz")
                self.message.emit(f"  初始Kc: {params['Kc']} Ω")
                self.message.emit(f"  初始Vpc: {params['Vpc']} c")
                
                progress = int((idx + 1) * 100 / total)
                self.progress_updated.emit(progress, f"计算中... ({idx+1}/{total})")
                
                result = calculate_SEGMENT_TWT_NOLINE(
                    I=self.fixed_params['i'],
                    V=self.fixed_params['v'],
                    Kc=params['Kc'],
                    Loss_perunit=params['Loss_perunit'],
                    SectionedSEGMENT_IDX=self.fixed_params['section_seg_idx'],
                    p_SWS=self.fixed_params['p_sws'],
                    N_unit=self.fixed_params['n_unit'],
                    w=self.fixed_params['w'],
                    t=self.fixed_params['t'],
                    Fn_K=self.fixed_params['Fn_K'],
                    f0_GHz=params['Freq'],
                    Vpc=params['Vpc'],
                    para_func=self.para_func,
                    P_in=self.fixed_params['p_in'],
                    Loss_attu=self.fixed_params['loss_attu']
                )
                
                power = result['输出功率P_out']
                results.append((params['Freq'], power))
                self.message.emit(f"  输出功率: {power:.2f} W")
                
            self.calculation_finished.emit(results, self.fixed_params)
            
        except Exception as e:
            self.calculation_error.emit(f"计算错误: {str(e)}")
        
    def stop(self):
        """停止计算"""
        self._is_running = False


class PlotCanvas(FigureCanvas):
    """优化的绘图画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        
        # 设置初始绘图
        self.reset_plot()
        
    def reset_plot(self):
        """重置绘图区域"""
        self.ax.clear()
        self.ax.set_title("Output Power vs Frequency")
        self.ax.set_xlabel("Frequency (GHz)")
        self.ax.set_ylabel("Output Power (W)")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.draw()
        
    def plot_results(self, results, label=None):
        """绘制结果曲线"""
        freqs, powers = zip(*results)
        self.ax.plot(freqs, powers, 'o-', markersize=4, label=label)
        self.ax.legend()
        self.draw()


class TWTCalculator(QMainWindow):
    """主应用程序窗口 - 优化版本"""
    def __init__(self):
        super().__init__()
        self.history_results = []
        self.worker_thread = None
        self.init_ui()
        self.load_last_config()

    def init_ui(self):
        """初始化用户界面 - 优化布局和结构"""
        self.setWindowTitle('TWT分段计算器')
        self.setGeometry(100, 100, 1600, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局 - 水平分割，带有可调整大小的分隔条
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        
        # ================== 左侧面板 ==================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 标签页布局
        tab_widget = QTabWidget()
        
        # 固定参数标签页
        param_tab = QWidget()
        param_layout = QGridLayout(param_tab)
        param_layout.setSpacing(10)
        
        self.param_widgets = {}
        
        # 参数排列顺序
        params_order = [
            'i', 'v', 
            'p_sws', 'n_unit',
            'w', 't',
            'Fn_K',
            'p_in', 'loss_attu',
            'section_seg_idx',
            'vpc_coeff', 'kc_coeff'
        ]
        
        for idx, param_key in enumerate(params_order):
            widget = ParamWidget(param_key)
            row = idx // 2
            col = (idx % 2) * 2
            param_layout.addWidget(widget, row, col, 1, 2)
            self.param_widgets[param_key] = widget
        
        tab_widget.addTab(param_tab, "固定参数")
        
        # 可变参数标签页
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        
        # 表格说明文本
        table_desc = QLabel(
            "<b>此表格用于输入不同频率点的计算参数</b><br>"
            "Kc: 耦合阻抗，单位为欧姆(Ω)<br>"
            "Loss_perunit: 单位长度的损耗<br>"
            "Freq: 频率，单位为GHz<br>"
            "Vpc: 相位速度与光速的比值(c)"
        )
        table_layout.addWidget(table_desc)
        
        # 表格操作按钮
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加行", clicked=self.add_table_row)
        self.del_btn = QPushButton("删除行", clicked=self.delete_table_row)
        self.import_btn = QPushButton("导入CSV", clicked=self.import_csv)
        self.clear_table_btn = QPushButton("清空表格", clicked=self.clear_table)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.del_btn)
        btn_layout.addWidget(self.import_btn)
        btn_layout.addWidget(self.clear_table_btn)
        
        # 参数表格
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(300)
        
        table_layout.addLayout(btn_layout)
        table_layout.addWidget(self.table)
        
        tab_widget.addTab(table_tab, "可变参数")
        
        left_layout.addWidget(tab_widget)
        
        # 功能按钮组
        func_group = QGroupBox("功能操作")
        func_layout = QGridLayout(func_group)
        
        # 功能按钮
        self.calc_btn = QPushButton("开始计算", clicked=self.calculate)
        self.cancel_btn = QPushButton("取消计算", clicked=self.cancel_calculation)
        self.cancel_btn.setEnabled(False)
        self.save_btn = QPushButton("保存配置", clicked=self.save_config)
        self.export_btn = QPushButton("导出数据", clicked=self.export_data)
        
        func_layout.addWidget(self.calc_btn, 0, 0)
        func_layout.addWidget(self.cancel_btn, 0, 1)
        func_layout.addWidget(self.save_btn, 1, 0)
        func_layout.addWidget(self.export_btn, 1, 1)
        
        left_layout.addWidget(func_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)
        
        splitter.addWidget(left_panel)
        
        # ================== 右侧面板 ==================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # 结果标签页
        result_tabs = QTabWidget()
        
        # 绘图标签页
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        # 绘图控制按钮
        plot_btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清空绘图", clicked=self.clear_plot)
        self.export_plot_btn = QPushButton("导出图像", clicked=self.export_plot)
        plot_btn_layout.addWidget(self.clear_btn)
        plot_btn_layout.addWidget(self.export_plot_btn)
        
        # 绘图区域
        self.plot_canvas = PlotCanvas()
        self.plot_canvas.setMinimumSize(600, 400)
        
        plot_layout.addLayout(plot_btn_layout)
        plot_layout.addWidget(self.plot_canvas)
        result_tabs.addTab(plot_tab, "结果曲线")
        
        # 计算结果标签页
        result_text_tab = QWidget()
        result_text_layout = QVBoxLayout(result_text_tab)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("计算结果和参数调整信息将显示在此处...")
        
        # 添加计算过程说明
        param_func_desc = (
            "<h3>参数调整公式说明：</h3>"
            "<p><b>1. Vpc_adjusted = Vpc + (Vpc系数) × (当前段周期 - 第一段周期) / 第一段周期 × Vpc</b><br>"
            "   示例: 如果Vpc系数=0.3，当前周期=0.55，第一段周期=0.5，Vpc=0.288<br>"
            "   则Vpc_adjusted = 0.288 + 0.3×(0.55-0.5)/0.5 × 0.288 = 0.288 + 0.00864 = 0.29664</p>"
            "<p><b>2. Kc_adjusted = Kc + (Kc系数) × (当前段周期 - 第一段周期) / 第一段周期 × Kc</b><br>"
            "   示例: 如果Kc系数=0.9，当前周期=0.55，第一段周期=0.5，Kc=3.6<br>"
            "   则Kc_adjusted = 3.6 + 0.9×(0.55-0.5)/0.5 × 3.6 = 3.6 + 0.324 = 3.924</p>"
        )
        self.result_text.append(param_func_desc)
        
        result_text_layout.addWidget(self.result_text)
        result_tabs.addTab(result_text_tab, "计算结果")
        
        right_layout.addWidget(result_tabs)
        splitter.addWidget(right_panel)
        
        # 设置分割条初始比例
        splitter.setSizes([400, 1000])
        main_layout.addWidget(splitter)
        
        # 初始化表格
        self.add_table_row()

    def add_table_row(self):
        """添加表格行"""
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        
        # 自动填充默认值
        if row_count == 0:
            self.table.setItem(row_count, 0, QTableWidgetItem("3.6"))
            self.table.setItem(row_count, 1, QTableWidgetItem("0"))
            self.table.setItem(row_count, 2, QTableWidgetItem("211"))
            self.table.setItem(row_count, 3, QTableWidgetItem("0.288"))

    def delete_table_row(self):
        """删除表格行"""
        if self.table.rowCount() > 0:
            self.table.removeRow(self.table.currentRow() if self.table.currentRow() != -1 else self.table.rowCount()-1)

    def clear_table(self):
        """清空表格"""
        if self.table.rowCount() > 0:
            reply = QMessageBox.question(
                self, '确认', 
                '确定要清空表格吗？所有数据将丢失。',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.table.setRowCount(0)
                self.add_table_row()

    def import_csv(self):
        """导入CSV文件"""
        path, _ = QFileDialog.getOpenFileName(self, "打开CSV文件", "", "CSV文件 (*.csv)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader)  # 跳过标题行
                    self.table.setRowCount(0)  # 清空现有数据
                    for row in reader:
                        if len(row) < 4:
                            continue
                        row_num = self.table.rowCount()
                        self.table.insertRow(row_num)
                        for col, value in enumerate(row[:4]):
                            self.table.setItem(row_num, col, QTableWidgetItem(value.strip()))
                QMessageBox.information(self, "导入成功", f"成功导入{self.table.rowCount()}行数据")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"文件读取失败: {str(e)}")

    def get_fixed_params(self):
        """解析固定参数值"""
        params = {}
        for key, widget in self.param_widgets.items():
            value = widget.value
            
            # 特殊处理列表类型参数
            if key in ['p_sws', 'n_unit', 'section_seg_idx']:
                if not value:  # 空字符串处理
                    params[key] = [] if key != 'section_seg_idx' else []
                else:
                    try:
                        if key == 'section_seg_idx':  # 整数列表
                            params[key] = [int(i.strip()) for i in value.split(',') if i.strip()]
                        elif key == 'p_sws':  # 浮点数列表
                            params[key] = [float(p.strip()) for p in value.split(',') if p.strip()]
                        elif key == 'n_unit':  # 整数列表
                            params[key] = [int(n.strip()) for n in value.split(',') if n.strip()]
                    except Exception as e:
                        QMessageBox.warning(self, "错误", f"参数 {key} 格式错误: {str(e)}")
                        return None
            else:  # 常规数值处理
                try:
                    params[key] = float(value)
                except ValueError:
                    QMessageBox.warning(self, "错误", f"参数 {key} 不是有效的数字: {value}")
                    return None
        
        # 验证参数长度一致性
        if 'p_sws' in params and 'n_unit' in params:
            n_p_sws = len(params['p_sws'])
            n_n_unit = len(params['n_unit'])
            
            if n_p_sws != n_n_unit:
                QMessageBox.warning(self, "错误", 
                                  f"p_SWS的长度({n_p_sws})与N_unit的长度({n_n_unit})不一致!")
                return None
                
        if 'p_sws' in params and not params['p_sws']:
            QMessageBox.warning(self, "错误", "p_SWS参数不能为空!")
            return None
            
        if 'n_unit' in params and not params['n_unit']:
            QMessageBox.warning(self, "错误", "N_unit参数不能为空!")
            return None
            
        return params

    def get_var_params(self):
        """获取可变参数"""
        params = []
        for row in range(self.table.rowCount()):
            try:
                params.append({
                    'Kc': float(self.table.item(row, 0).text()),
                    'Loss_perunit': float(self.table.item(row, 1).text()),
                    'Freq': float(self.table.item(row, 2).text()),
                    'Vpc': float(self.table.item(row, 3).text())
                })
            except (AttributeError, ValueError):
                continue
        return params

    def para_func(self, p_SWS, idx, Vpc, Kc):
        """实现参数计算函数"""
        current_p = p_SWS[idx]
        first_p = p_SWS[0]
        
        # 获取用户输入的调整系数
        try:
            vpc_coeff = float(self.param_widgets['vpc_coeff'].value)
            kc_coeff = float(self.param_widgets['kc_coeff'].value)
        except ValueError:
            vpc_coeff = 0.3
            kc_coeff = 0.9
        
        # 计算调整值
        vpc_adjust = vpc_coeff * (current_p - first_p) / first_p * Vpc
        kc_adjust = kc_coeff * (current_p - first_p) / first_p * Kc
        
        Vpc_adjusted = Vpc + vpc_adjust
        Kc_adjusted = Kc + kc_adjust
        
        # 返回调整后的值
        return {"Vpc": Vpc_adjusted, "Kc": Kc_adjusted}

    def calculate(self):
        """启动计算线程"""
        self.result_text.clear()
        self.result_text.append("计算开始...")
        
        fixed_params = self.get_fixed_params()
        if fixed_params is None:
            return
            
        # 显示基本参数
        self.result_text.append(f"\n固定参数设置:")
        self.result_text.append(f"  电流 I: {fixed_params['i']} A")
        self.result_text.append(f"  电压 V: {fixed_params['v']} V")
        self.result_text.append(f"  周期长度: {','.join([str(p) for p in fixed_params['p_sws']])} mm")
        self.result_text.append(f"  周期数: {','.join([str(n) for n in fixed_params['n_unit']])}")
        self.result_text.append(f"  束流 w×t: {fixed_params['w']}×{fixed_params['t']} mm")
        self.result_text.append(f"  Fn_K: {fixed_params['Fn_K']}")
        self.result_text.append(f"  输入功率: {fixed_params['p_in']} W")
        self.result_text.append(f"  衰减量: {fixed_params['loss_attu']} dB")
        self.result_text.append(f"  衰减段索引: {','.join([str(i) for i in fixed_params['section_seg_idx']])}")
        self.result_text.append(f"  Vpc调整系数: {self.param_widgets['vpc_coeff'].value}")
        self.result_text.append(f"  Kc调整系数: {self.param_widgets['kc_coeff'].value}\n")
        
        var_params = self.get_var_params()
        
        if not var_params:
            QMessageBox.warning(self, "错误", "没有可计算的参数！")
            return
            
        # 设置UI状态
        self.calc_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动工作线程
        self.worker_thread = WorkerThread(
            fixed_params, 
            var_params,
            self.para_func
        )
        
        # 连接信号
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.calculation_finished.connect(self.handle_calculation_finished)
        self.worker_thread.calculation_error.connect(self.handle_calculation_error)
        self.worker_thread.message.connect(self.result_text.append)
        
        # 启动线程
        self.worker_thread.start()

    def cancel_calculation(self):
        """取消正在进行的计算"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.cancel_btn.setEnabled(False)
            self.result_text.append("\n计算已取消")
            QMessageBox.information(self, "取消", "计算已被取消")

    def update_progress(self, value, message):
        """更新进度条"""
        self.progress_bar.setValue(value)
        if value == 100:
            self.progress_bar.setFormat("完成")
        else:
            self.progress_bar.setFormat(f"{message}")

    def handle_calculation_finished(self, results, fixed_params):
        """处理计算完成"""
        self.calc_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("计算完成")
        
        if results:
            # 保存结果
            self.save_results(results, fixed_params, self.get_var_params())
            
            # 更新绘图
            self.plot_canvas.plot_results(results, f"参数组 {len(self.history_results)+1}")
            
            self.result_text.append(f"\n计算完成！共计算{len(results)}组数据")
            QMessageBox.information(self, "完成", f"成功计算{len(results)}组数据")
        else:
            self.result_text.append("\n计算完成，但没有有效的结果")
            QMessageBox.warning(self, "错误", "没有有效的计算结果！")

    def handle_calculation_error(self, error_msg):
        """处理计算错误"""
        self.result_text.append(error_msg)
        self.calc_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.critical(self, "计算错误", error_msg)

    def save_results(self, results, fixed_params, var_params):
        """保存结果文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, f"Result_{timestamp}.csv")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            # 写入固定参数
            f.write("固定参数:\n")
            f.write(f"电流 I (A): {fixed_params['i']}\n")
            f.write(f"电压 V (V): {fixed_params['v']}\n")
            f.write(f"周期长度 p_SWS (mm): {','.join(map(str, fixed_params['p_sws']))}\n")
            f.write(f"周期数 N_Unit: {','.join(map(str, fixed_params['n_unit']))}\n")
            f.write(f"束流宽度 w (mm): {fixed_params['w']}\n")
            f.write(f"束流厚度 t (mm): {fixed_params['t']}\n")
            f.write(f"Fn_K 参数: {fixed_params['Fn_K']}\n")
            f.write(f"输入功率 P_in (W): {fixed_params['p_in']}\n")
            f.write(f"衰减量 (dB): {fixed_params['loss_attu']}\n")
            f.write(f"衰减段索引: {','.join(map(str, fixed_params['section_seg_idx']))}\n")
            f.write(f"Vpc调整系数: {self.param_widgets['vpc_coeff'].value}\n")
            f.write(f"Kc调整系数: {self.param_widgets['kc_coeff'].value}\n\n")
            
            # 写入数据表头
            f.write("频率 (GHz),输出功率 (W),Kc (Ω),Loss_perunit,Vpc (c)\n")
            
            # 写入每行数据
            for res, var in zip(results, var_params):
                freq, power = res
                f.write(f"{freq},{power},{var['Kc']},{var['Loss_perunit']},{var['Vpc']}\n")
        
        self.history_results.append((results, fixed_params))
        self.result_text.append(f"结果已保存至: {save_path}")

    def clear_plot(self):
        """清空绘图"""
        self.history_results.clear()
        self.plot_canvas.reset_plot()

    def export_plot(self):
        """导出绘图到文件"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", 
            f"TWT_Output_Power_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "PNG图像 (*.png);;SVG矢量图 (*.svg);;PDF文件 (*.pdf)"
        )
        if path:
            try:
                self.plot_canvas.figure.savefig(path, dpi=300)
                QMessageBox.information(self, "成功", f"图像已保存至: {path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像失败: {str(e)}")

    def export_data(self):
        """导出数据文件（按原始格式）"""
        if not self.history_results:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存数据文件", "", "CSV文件 (*.csv)")
        if not path:
            return

        with open(path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("频率(GHz),输出功率(W)\n")
            
            # 写入每行数据（每个历史结果集单独导出）
            for set_idx, (results, _) in enumerate(self.history_results):
                f.write(f"# 结果集 {set_idx+1}\n")
                for freq, power in sorted(results, key=lambda x: x[0]):
                    f.write(f"{freq},{power}\n")
                f.write("\n")
        
        QMessageBox.information(self, "导出成功", f"数据已保存至：\n{path}")

    def save_config(self):
        """保存当前配置"""
        config = {
            "fixed": {k: widget.value for k, widget in self.param_widgets.items()},
            "variables": [[self.table.item(row, col).text() for col in range(4)] 
                         for row in range(self.table.rowCount())]
        }
        
        os.makedirs(CONFIG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(CONFIG_DIR, f"config_{timestamp}.json")
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        self.result_text.append(f"配置已保存至: {save_path}")
        QMessageBox.information(self, "保存成功", f"配置已保存至：\n{save_path}")

    def load_last_config(self):
        """加载最近配置"""
        try:
            if not os.path.exists(CONFIG_DIR):
                return
            
            config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
            if not config_files:
                return
                
            # 找到最新配置文件
            latest_file = max(
                config_files, 
                key=lambda f: os.path.getctime(os.path.join(CONFIG_DIR, f))
            )
            
            # 读取配置文件
            with open(os.path.join(CONFIG_DIR, latest_file), 'r') as f:
                config = json.load(f)
                
                # 加载固定参数
                for key, value in config["fixed"].items():
                    if key in self.param_widgets:
                        self.param_widgets[key].line_edit.setText(str(value))
                
                # 加载表格数据
                self.table.setRowCount(0)
                for row in config["variables"]:
                    row_num = self.table.rowCount()
                    self.table.insertRow(row_num)
                    for col in range(4):
                        self.table.setItem(row_num, col, QTableWidgetItem(row[col]))
                        
            self.result_text.append(f"已加载配置: {latest_file}")
        except Exception as e:
            self.result_text.append(f"配置加载错误: {str(e)}")
            print(f"配置加载错误: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TWTCalculator()
    window.show()
    sys.exit(app.exec_())