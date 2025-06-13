import sys
import numpy as np
import logging
import time
from multiprocessing import Pool, cpu_count
from sko.PSO import PSO

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTabWidget,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QMessageBox,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from Noline_MAINCALL_VCORE_FORPVTOPT import calculate_SEGMENT_TWT_NOLINE_for_PVTOPT
from Noline_MAINCALL_VCORE_FORPVTOPT import (
    calculate_SEGMENT_TWT_NOLINE_SECTIONED_for_PVTOPT,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PVTFunction:
    def __init__(self, vpc_a=0.05, vpc_b=0.288 - 0.5 * 0.05, kc_a=0.0, kc_b=3.6):
        self.vpc_a = vpc_a
        self.vpc_b = vpc_b
        self.kc_a = kc_a
        self.kc_b = kc_b

    def __call__(self, p_SWS):
        return {
            "Vpc": self.vpc_a * p_SWS + self.vpc_b,
            "Kc": self.kc_a * p_SWS + self.kc_b,
        }

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class FitnessEvaluator:
    def __init__(self, fixed_params, custom_para_func, calculate_func):
        self.fixed_params = fixed_params
        self.custom_para_func = custom_para_func
        self.calculate_func = calculate_func

    def __call__(self, x):
        try:
            params = self.fixed_params
            p_SWS = x.tolist() if isinstance(x, np.ndarray) else x
            twt_result = self.calculate_func(
                params["I"],
                params["V"],
                params["Loss"],
                p_SWS,
                params["N_unit"],
                params["w"],
                params["t"],
                params["Fn_K"],
                params["f0_GHz"],
                self.custom_para_func,
            )
            return -twt_result["输出功率P_out"]
        except Exception as e:
            logger.error(f"无效参数: {np.round(x,3)}, 错误: {str(e)[:50]}")
            return float("inf")


class OptimizationThread(QThread):
    finished = pyqtSignal(object)
    log_message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, fixed_params, pso_config, initial_p_SWS, custom_para_func):
        super().__init__()
        self.fixed_params = fixed_params
        self.pso_config = pso_config
        self.initial_p_SWS = initial_p_SWS
        self.custom_para_func = custom_para_func
        self.calculate_func = (
            calculate_SEGMENT_TWT_NOLINE_for_PVTOPT
            if fixed_params["model_type"] == "non_sectioned"
            else calculate_SEGMENT_TWT_NOLINE_SECTIONED_for_PVTOPT
        )

    def run(self):
        try:
            model_type = (
                "非分段"
                if self.fixed_params["model_type"] == "non_sectioned"
                else "分段"
            )
            self.log_message.emit(f"执行初始参数计算 ({model_type}模型)...")

            # 初始计算
            params = self.fixed_params
            initial_result = self.calculate_func(
                params["I"],
                params["V"],
                params["Loss"],
                self.initial_p_SWS,
                params["N_unit"],
                params["w"],
                params["t"],
                params["Fn_K"],
                params["f0_GHz"],
                self.custom_para_func,
            )
            initial_power = initial_result["输出功率P_out"]
            self.log_message.emit(f"初始参数功率: {initial_power:.2f} W")

            # 优化过程
            self.log_message.emit("\n开始优化过程...")
            optimized_p_SWS = self.optimize_TWT_with_PSO()

            # 验证结果
            verification_result = self.calculate_func(
                params["I"],
                params["V"],
                params["Loss"],
                optimized_p_SWS.tolist(),
                params["N_unit"],
                params["w"],
                params["t"],
                params["Fn_K"],
                params["f0_GHz"],
                self.custom_para_func,
            )
            verified_power = verification_result["输出功率P_out"]

            # 结果对比
            self.log_message.emit("\n优化结果对比:")
            self.log_message.emit(
                f"初始功率: {initial_power:.2f} W → 优化后: {verified_power:.2f} W"
            )

            # 返回结果
            self.finished.emit(
                {
                    "initial_power": initial_power,
                    "optimized_p_SWS": optimized_p_SWS,
                    "verified_power": verified_power,
                    "segment_params": [
                        self.custom_para_func(p) for p in optimized_p_SWS
                    ],
                    "model_type": self.fixed_params["model_type"],
                }
            )

        except Exception as e:
            self.log_message.emit(f"优化过程中发生错误: {str(e)}")
            self.finished.emit(None)

    def optimize_TWT_with_PSO(self):
        n_dim = len(self.fixed_params["N_unit"])
        if len(self.pso_config["lb"]) != n_dim or len(self.pso_config["ub"]) != n_dim:
            raise ValueError("边界条件维度与N_unit长度不一致")

        cpus = self.pso_config.get("cpus", cpu_count())
        self.log_message.emit(f"使用多进程并行计算 (CPU核心数: {cpus})")

        evaluator = FitnessEvaluator(
            self.fixed_params, self.custom_para_func, self.calculate_func
        )
        pso = PSO(
            func=evaluator,
            n_dim=n_dim,
            pop=self.pso_config["pop_size"],
            max_iter=self.pso_config["max_iter"],
            lb=self.pso_config["lb"],
            ub=self.pso_config["ub"],
        )

        with Pool(processes=cpus) as pool:
            original_cal_y = pso.cal_y

            def parallel_cal_y():
                try:
                    results = pool.map(evaluator, pso.X)
                    pso.Y = np.array(results).reshape(-1, 1)
                except Exception:
                    original_cal_y()

            pso.cal_y = parallel_cal_y

            try:
                for i in range(self.pso_config["max_iter"]):
                    pso.run(1)
                    progress = int((i + 1) / self.pso_config["max_iter"] * 100)
                    self.progress_updated.emit(progress)
                    self.log_message.emit(
                        f"迭代 {i+1}/{self.pso_config['max_iter']} 完成, 当前最佳功率: {-pso.gbest_y[0]:.2f} W"
                    )
                    time.sleep(0.01)
            except Exception as e:
                self.log_message.emit(f"优化过程中出错: {str(e)}")

        optimal_p_SWS = np.round(pso.gbest_x, 4)
        self.log_message.emit(f"优化完成! 最优p_SWS: {optimal_p_SWS.tolist()}")
        return optimal_p_SWS


class BaseEditor(QWidget):
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        layout = QFormLayout()
        self.widgets = {}

        for field in fields:
            if len(field) == 2 and field[1] == "label":
                # 纯标签字段
                label = QLabel(field[0])
                layout.addRow(label)
            else:
                # 输入字段
                label_text, widget_type, *args = field
                label = QLabel(label_text)

                if widget_type == "double":
                    widget = QDoubleSpinBox()
                    widget.setRange(args[0], args[1])
                    widget.setValue(args[2])
                    if len(args) > 3:
                        widget.setSingleStep(args[3])
                    if len(args) > 4:
                        widget.setDecimals(args[4])
                elif widget_type == "int":
                    widget = QSpinBox()
                    widget.setRange(args[0], args[1])
                    widget.setValue(args[2])
                elif widget_type == "text":
                    widget = QLineEdit()
                    if args:
                        widget.setText(args[0])
                elif widget_type == "combo":
                    widget = QComboBox()
                    for item in args:
                        widget.addItem(item[0], item[1])

                layout.addRow(label, widget)
                self.widgets[label_text] = widget

        self.setLayout(layout)

    def get_value(self, label_text):
        """获取指定标签对应的控件值"""
        if label_text in self.widgets:
            widget = self.widgets[label_text]
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                return widget.value()
            elif isinstance(widget, QLineEdit):
                return widget.text()
            elif isinstance(widget, QComboBox):
                return widget.currentData()
        return None


class PVTFunctionEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("Vpc = a * p_SWS + b", "label"),
            ("a (斜率):", "double", -10, 10, 0.05, 0.01, 4),
            ("b (截距):", "double", -10, 10, 0.288 - 0.5 * 0.05, 0.01, 4),
            ("Kc = c * p_SWS + d", "label"),
            ("c (斜率):", "double", -10, 10, 0.0, 0.01, 4),
            ("d (截距):", "double", -10, 10, 3.6, 0.01, 4),
        ]
        super().__init__(fields, parent)

    def get_function(self):
        return PVTFunction(
            self.get_value("a (斜率):"),
            self.get_value("b (截距):"),
            self.get_value("c (斜率):"),
            self.get_value("d (截距):"),
        )


class FixedParamsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("束流电流 (A):", "double", 0, 1e2, 0.3, 0.01, 3),
            ("工作电压 (V):", "double", 0, 100000, 23000, 100, 0),
            ("损耗参数:", "double", 0, 1e3, 0.0, 0.01, 3),
            ("各段单元数 (逗号分隔):", "text", "30,5,5,5,5"),
            ("结构参数 w (mm):", "double", 0, 1e1, 0.2, 0.01, 3),
            ("结构参数 t (mm):", "double", 0, 1e1, 0.2, 0.01, 3),
            ("填充系数 Fn_K:", "double", 1, 10, 1.0, 0.01, 3),
            ("工作频率 (GHz):", "double", 0, 2e3, 211.0, 1, 1),
            (
                "模型类型:",
                "combo",
                ("非分段模型", "non_sectioned"),
                ("分段模型", "sectioned"),
            ),#类型: "double" # - 参数: # - 最小值:  # - 最大值: # - 默认值: # - 步长:  # - 小数位数: 
        ]
        super().__init__(fields, parent)

    def get_params(self):
        try:
            n_units_text = self.get_value("各段单元数 (逗号分隔):")
            n_units = [int(x.strip()) for x in n_units_text.split(",")]

            return {
                "I": self.get_value("束流电流 (A):"),
                "V": self.get_value("工作电压 (V):"),
                "Loss": self.get_value("损耗参数:"),
                "N_unit": n_units,
                "w": self.get_value("结构参数 w (mm):"),
                "t": self.get_value("结构参数 t (mm):"),
                "Fn_K": self.get_value("填充系数 Fn_K:"),
                "f0_GHz": self.get_value("工作频率 (GHz):"),
                "model_type": self.get_value("模型类型:"),
            }
        except ValueError:
            QMessageBox.warning(
                self, "输入错误", "各段单元数必须是整数列表（例如：30,5,5,5,5）"
            )
            return None


class PSOConfigEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("种群大小:", "int", 10, 1000, 20),
            ("最大迭代次数:", "int", 10, 1000, 20),
            ("参数下限 (逗号分隔):", "text", "0.4,0.4,0.4,0.4,0.4"),
            ("参数上限 (逗号分隔):", "text", "0.6,0.6,0.6,0.6,0.6"),
            ("使用的CPU核心数:", "int", 1, cpu_count(), cpu_count()),
        ]
        super().__init__(fields, parent)

    def get_config(self, n_dim):
        try:
            lb_text = self.get_value("参数下限 (逗号分隔):")
            ub_text = self.get_value("参数上限 (逗号分隔):")

            lb = [float(x.strip()) for x in lb_text.split(",")]
            ub = [float(x.strip()) for x in ub_text.split(",")]

            # 扩展单值到所有维度
            lb = lb * n_dim if len(lb) == 1 else lb
            ub = ub * n_dim if len(ub) == 1 else ub

            if len(lb) != n_dim or len(ub) != n_dim:
                raise ValueError(f"边界值数量必须与段数({n_dim})匹配")

            return {
                "pop_size": self.get_value("种群大小:"),
                "max_iter": self.get_value("最大迭代次数:"),
                "lb": lb,
                "ub": ub,
                "cpus": self.get_value("使用的CPU核心数:"),
            }
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


class InitialParamsEditor(BaseEditor):
    def __init__(self, parent=None):
        fields = [
            ("初始p_SWS值 (逗号分隔):", "text", "0.50,0.50,0.50,0.50,0.50"),
        ]
        super().__init__(fields, parent)

    def get_initial_p(self, n_dim):
        try:
            initial_p_text = self.get_value("初始p_SWS值 (逗号分隔):")
            initial_p = [float(x.strip()) for x in initial_p_text.split(",")]

            # 扩展单值到所有维度
            initial_p = initial_p * n_dim if len(initial_p) == 1 else initial_p

            if len(initial_p) != n_dim:
                raise ValueError(f"初始p_SWS值数量必须与段数({n_dim})匹配")

            return initial_p
        except Exception as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return None


class TWTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TWT参数优化工具")
        self.setGeometry(100, 100, 800, 700)

        # 主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 选项卡
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # 配置页面
        self.pvt_editor = PVTFunctionEditor()
        self.fixed_params_editor = FixedParamsEditor()
        self.pso_editor = PSOConfigEditor()
        self.initial_editor = InitialParamsEditor()

        tabs.addTab(self.create_tab("PVT函数", self.pvt_editor), "PVT函数")
        tabs.addTab(self.create_tab("固定参数", self.fixed_params_editor), "固定参数")
        tabs.addTab(self.create_tab("PSO配置", self.pso_editor), "PSO配置")
        tabs.addTab(self.create_tab("初始参数", self.initial_editor), "初始参数")

        # 日志区域
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(QLabel("日志输出:"))
        main_layout.addWidget(self.log_area)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.run_button = QPushButton("开始优化")
        self.run_button.clicked.connect(self.start_optimization)
        button_layout.addWidget(self.run_button)

        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_ui)
        button_layout.addWidget(self.reset_button)

        main_layout.addLayout(button_layout)

        # 应用样式
        self.setStyleSheet(
            """
            QMainWindow { 
                background-color: #f0f0f0; 
                font-family: 'Segoe UI', Arial, sans-serif; 
            }
            QTabWidget::pane { 
                border: 1px solid #c0c0c0; 
                background: white; 
            }
            QTabBar::tab { 
                background: #e0e0e0; 
                border: 1px solid #c0c0c0; 
                padding: 8px 15px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { 
                background: white; 
                border-bottom-color: white; 
            }
            QGroupBox { 
                border: 1px solid #c0c0c0; 
                border-radius: 5px; 
                margin-top: 1ex;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QPushButton { 
                background-color: #4a86e8; 
                color: white; 
                border: none; 
                padding: 8px 15px; 
                border-radius: 4px; 
                font-weight: bold; 
                min-width: 100px;
            }
            QPushButton:hover { background-color: #3a76d8; }
            QPushButton:disabled { background-color: #a0a0a0; }
            QTextEdit { 
                background-color: white; 
                border: 1px solid #c0c0c0; 
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QLabel { font-weight: normal; }
        """
        )

        # 全局变量
        self.optimization_thread = None
        self.optimization_result = None
        self.custom_para_func = None

    def create_tab(self, title, widget):
        """创建选项卡内容"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)

        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(widget)
        group_layout.setContentsMargins(10, 20, 10, 10)

        layout.addWidget(group)
        layout.addStretch()
        return tab

    def start_optimization(self):
        """启动优化过程"""
        # 获取自定义PVT函数
        self.custom_para_func = self.pvt_editor.get_function()

        # 获取固定参数
        fixed_params = self.fixed_params_editor.get_params()
        if fixed_params is None:
            return

        # 获取PSO配置
        n_dim = len(fixed_params["N_unit"])
        pso_config = self.pso_editor.get_config(n_dim)
        if pso_config is None:
            return

        # 获取初始参数
        initial_p = self.initial_editor.get_initial_p(n_dim)
        if initial_p is None:
            return

        # 准备UI状态
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_area.clear()
        self.log_area.append("开始优化过程...")

        # 创建并启动优化线程
        self.optimization_thread = OptimizationThread(
            fixed_params, pso_config, initial_p, self.custom_para_func
        )
        self.optimization_thread.finished.connect(self.optimization_finished)
        self.optimization_thread.log_message.connect(self.log_area.append)
        self.optimization_thread.progress_updated.connect(self.progress_bar.setValue)
        self.optimization_thread.start()

    def optimization_finished(self, result):
        """优化完成处理"""
        self.optimization_result = result
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if result is None:
            self.log_area.append("优化失败！")
            QMessageBox.critical(self, "优化失败", "优化过程中发生错误，请检查日志")
        else:
            model_type = "非分段" if result["model_type"] == "non_sectioned" else "分段"
            self.log_area.append(f"\n优化完成! (使用{model_type}模型)")
            self.log_area.append(f"初始功率: {result['initial_power']:.2f} W")
            self.log_area.append(f"优化后功率: {result['verified_power']:.2f} W")
            self.log_area.append(
                f"功率提升: {result['verified_power'] - result['initial_power']:.2f} W"
            )
            self.log_area.append("优化结果已准备就绪，可以保存")
            self.save_button.setEnabled(True)

    def save_results(self):
        """保存优化结果"""
        if self.optimization_result is None:
            QMessageBox.warning(self, "保存失败", "没有可用的优化结果")
            return

        try:
            # 保存优化后的p_SWS值
            np.savetxt(
                "optimized_p_SWS.txt", self.optimization_result["optimized_p_SWS"]
            )

            # 保存PVT段参数
            with open("optimized_pvt_params.txt", "w") as f:
                for i, params in enumerate(self.optimization_result["segment_params"]):
                    f.write(
                        f"段 {i+1}: Vpc={params['Vpc']:.4f}, Kc={params['Kc']:.4f}\n"
                    )

            self.log_area.append("\n优化结果已保存到当前目录:")
            self.log_area.append("- optimized_p_SWS.txt: 最优p_SWS值")
            self.log_area.append("- optimized_pvt_params.txt: PVT段参数")
            QMessageBox.information(self, "保存成功", "优化结果已保存")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存结果时出错: {str(e)}")

    def reset_ui(self):
        """重置UI状态"""
        self.progress_bar.setVisible(False)
        self.save_button.setEnabled(False)
        self.log_area.clear()
        self.optimization_result = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TWTMainWindow()
    window.show()
    sys.exit(app.exec_())
