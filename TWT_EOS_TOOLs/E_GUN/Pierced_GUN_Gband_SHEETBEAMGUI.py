import sys
import numpy as np
from scipy.optimize import fsolve
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QGroupBox, QLabel, QPushButton, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QGridLayout, QMessageBox,
                            QDoubleSpinBox, QTextEdit)
from PyQt5.QtCore import Qt

class PierceGunOptimizer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Pierce 电子枪参数优化器")
        self.setGeometry(100, 100, 1200, 1000)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 参数输入部分
        param_group = QGroupBox("输入参数")
        param_layout = QGridLayout()
        
        # 添加参数输入字段
        self.voltage_input = self.create_spin_input("电压 U (V):", 25000, 1000, 100000, 1000, 0)
        self.current_input = self.create_spin_input("电流 I (A):", 0.04, 0.001, 10.0, 0.001, 4)
        self.jc_input = self.create_spin_input("电流密度 Jc (A/mm²):", 0.16, 0.001, 10.0, 0.01, 3)
        self.tao_input = self.create_spin_input("Tao:", 1.25, 0.1, 10.0, 0.01, 2)
        self.k_input = self.create_spin_input("k:", 0.905, 0.1, 2.0, 0.001, 3)
        self.rw1_input = self.create_spin_input("阴极半径 rw1 (mm):", 0.03, 0.001, 10.0, 0.001, 4)
        self.rw2_ratio_input = self.create_spin_input("阴极半径比例:", 5, 0.1, 20.0, 0.1, 2)
        
        # RD 范围输入
        self.rd_start_input = self.create_spin_input("阳极孔半径起始 (mm):", 0.2, 0.01, 10.0, 0.01, 3)
        self.rd_end_input = self.create_spin_input("阳极孔半径结束 (mm):", 2.0, 0.01, 10.0, 0.01, 3)
        self.rd_step_input = self.create_spin_input("阳极孔半径步长 (mm):", 0.1, 0.001, 1.0, 0.001, 4)
        
        # 容差设置
        self.zac_tolerance_input = self.create_spin_input("Zac容差 (mm):", 0.01, 0.001, 1.0, 0.001, 4)
        self.rc_tolerance_input = self.create_spin_input("rc互易容差 (mm):", 0.01, 0.001, 1.0, 0.001, 4)
        
        # 添加到布局
        row = 0
        param_layout.addWidget(QLabel("电子枪参数:"), row, 0)
        param_layout.addWidget(self.voltage_input, row, 1)
        param_layout.addWidget(self.current_input, row, 2)
        param_layout.addWidget(self.jc_input, row, 3)
        row += 1
        
        param_layout.addWidget(QLabel("常数:"), row, 0)
        param_layout.addWidget(self.tao_input, row, 1)
        param_layout.addWidget(self.k_input, row, 2)
        row += 1
        
        param_layout.addWidget(QLabel("阴极半径设置:"), row, 0)
        param_layout.addWidget(self.rw1_input, row, 1)
        param_layout.addWidget(QLabel("rw2 = rw1 × "), row, 2)
        param_layout.addWidget(self.rw2_ratio_input, row, 3)
        row += 1
        
        param_layout.addWidget(QLabel("阳极孔半径范围:"), row, 0)
        param_layout.addWidget(self.rd_start_input, row, 1)
        param_layout.addWidget(QLabel("到"), row, 2)
        param_layout.addWidget(self.rd_end_input, row, 3)
        param_layout.addWidget(QLabel("步长"), row, 4)
        param_layout.addWidget(self.rd_step_input, row, 5)
        row += 1
        
        param_layout.addWidget(QLabel("优化容差:"), row, 0)
        param_layout.addWidget(self.zac_tolerance_input, row, 1)
        param_layout.addWidget(QLabel("Zac容差"), row, 2)
        param_layout.addWidget(self.rc_tolerance_input, row, 3)
        param_layout.addWidget(QLabel("rc互易容差"), row, 4)
        
        param_group.setLayout(param_layout)
        
        # 计算按钮
        self.calculate_button = QPushButton("优化计算")
        self.calculate_button.clicked.connect(self.calculate)
        
        # 进度标签
        self.progress_label = QLabel("就绪")
        self.progress_label.setAlignment(Qt.AlignCenter)
        
        # 结果表格
        self.first_table = self.create_result_table("第一次运行结果 (rw1)")
        self.second_table = self.create_result_table("第二次运行结果 (rw2)")
        
        # 最佳匹配显示
        self.match_label = QLabel("最佳匹配将显示在这里")
        self.match_label.setAlignment(Qt.AlignCenter)
        self.match_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        
        # 调试输出
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        
        # 表格布局
        tables_layout = QHBoxLayout()
        tables_layout.addWidget(self.first_table)
        tables_layout.addWidget(self.second_table)
        
        # 主布局添加所有部件
        main_layout.addWidget(param_group)
        main_layout.addWidget(self.calculate_button)
        main_layout.addWidget(self.progress_label)
        main_layout.addLayout(tables_layout)
        main_layout.addWidget(self.match_label)
        main_layout.addWidget(QLabel("计算详情:"))
        main_layout.addWidget(self.debug_output)
    
    def create_spin_input(self, label_text, default, min_val, max_val, step, decimals):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        spin = QDoubleSpinBox()
        spin.setMinimum(min_val)
        spin.setMaximum(max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        layout.addWidget(spin)
        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    def create_result_table(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout()
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["rd (mm)", "rc (mm)", "Zac (mm)", "rw (mm)"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(table)
        group.setLayout(layout)
        return group
    
    def get_input_value(self, widget):
        # 从输入部件获取值
        spin = widget.findChild(QDoubleSpinBox)
        if spin:
            return spin.value()
        return None
    
    def validate_inputs(self):
        # 验证所有输入值是否有效
        inputs = {
            "U": self.get_input_value(self.voltage_input),
            "I": self.get_input_value(self.current_input),
            "Jc": self.get_input_value(self.jc_input),
            "Tao_val": self.get_input_value(self.tao_input),
            "k_val": self.get_input_value(self.k_input),
            "rw1": self.get_input_value(self.rw1_input),
            "rw2_ratio": self.get_input_value(self.rw2_ratio_input),
            "rd_start": self.get_input_value(self.rd_start_input),
            "rd_end": self.get_input_value(self.rd_end_input),
            "rd_step": self.get_input_value(self.rd_step_input),
            "zac_tolerance": self.get_input_value(self.zac_tolerance_input),
            "rc_tolerance": self.get_input_value(self.rc_tolerance_input),
        }
        
        # 检查是否有无效输入
        for name, value in inputs.items():
            if value is None:
                QMessageBox.warning(self, "输入错误", f"参数 {name} 无效")
                return None
            
            # 检查非负值
            if name in ["I", "Jc", "rw1", "rw2_ratio", "rd_start", "rd_end", "rd_step", 
                       "zac_tolerance", "rc_tolerance"] and value <= 0:
                QMessageBox.warning(self, "输入错误", f"参数 {name} 必须是正数")
                return None
        
        # 检查范围有效性
        if inputs["rd_start"] >= inputs["rd_end"]:
            QMessageBox.warning(self, "输入错误", "阳极孔半径起始值必须小于结束值")
            return None
        
        if inputs["rd_step"] <= 0:
            QMessageBox.warning(self, "输入错误", "阳极孔半径步长必须大于0")
            return None
        
        return inputs
    
    def debug_print(self, message):
        """打印调试信息"""
        self.debug_output.append(message)
        QApplication.processEvents()
    
    def compute_pierce_gun(self, U, I, rw, Jc, Tao_val, k_val, rd_list):
        """Pierce电子枪参数计算"""
        results = []
        for rd in rd_list:
            uP = 1e6 * I / (U ** 1.5)
            theta = 30 * math.sqrt(uP)
            
            # 迭代求解theta
            iteration = 0
            max_iterations = 100
            while iteration < max_iterations:
                iteration += 1
                
                # 求解Rc
                def equation_Rc(x):
                    return math.pi * x * math.sin(math.radians(theta)) * rd - I / Jc
                
                try:
                    Rc = fsolve(equation_Rc, 1)[0]
                except Exception as e:
                    self.debug_print(f"求解 Rc 失败: {str(e)}")
                    break
                
                rc = Rc * math.sin(math.radians(theta))
                alpha = math.sqrt(14.67 * (1 - math.cos(math.radians(theta))) / uP)
                gamma = alpha - 0.275 * (alpha**2) + 0.06 * (alpha**3) - 0.006 * (alpha**4)
                rb_Za = rc * math.exp(-gamma)
                
                # 计算Tan_PhiA和Tan_PhiB
                numerator = Tao_val * (1 + 0.6*gamma + 0.225*gamma**2 + 0.0573*gamma**3 + 
                                      0.0108*gamma**4 + 0.0021*gamma**5)
                Tan_PhiA = math.sin(math.radians(theta)) * (1 - numerator / (3 * alpha))
                
                R = rb_Za / rw
                if R <= 0:
                    Tan_PhiB = 0
                else:
                    Tan_PhiB = 0.17409 * math.sqrt(uP * abs(math.log(R)))
                
                # 更新theta
                if Tan_PhiA != 0:
                    theta = theta * math.sqrt(Tan_PhiB / abs(Tan_PhiA))
                
                # 检查收敛条件
                if Tan_PhiB != 0 and abs(Tan_PhiA / Tan_PhiB - 1) <= 0.005:
                    break
            
            # 计算最终参数
            Ra = Rc * math.exp(-gamma)
            Zac = Rc - Ra
            
            results.append({
                'rd': rd,
                'rc': rc,
                'Zac': Zac,
                'rw': rw
            })
            
        return results
    
    def calculate(self):
        inputs = self.validate_inputs()
        if not inputs:
            return
            
        # 清空调试输出
        self.debug_output.clear()
        self.progress_label.setText("计算中...")
        self.progress_label.setStyleSheet("color: blue; font-weight: bold;")
        QApplication.processEvents()
        
        try:
            # 生成rd列表
            num_points = int((inputs["rd_end"] - inputs["rd_start"]) / inputs["rd_step"]) + 1
            rd_list = np.linspace(inputs["rd_start"], inputs["rd_end"], num_points)
            
            # 计算rw2
            rw2 = inputs["rw1"] * inputs["rw2_ratio"]
            
            # 打印输入参数
            self.debug_print("=== 输入参数 ===")
            self.debug_print(f"电压 U = {inputs['U']:.1f} V")
            self.debug_print(f"电流 I = {inputs['I']:.4f} A")
            self.debug_print(f"电流密度 Jc = {inputs['Jc']:.3f} A/mm²")
            self.debug_print(f"Tao = {inputs['Tao_val']:.2f}")
            self.debug_print(f"k = {inputs['k_val']:.3f}")
            self.debug_print(f"阴极半径 rw1 = {inputs['rw1']:.3f} mm")
            self.debug_print(f"阴极半径 rw2 = {rw2:.3f} mm")
            self.debug_print(f"阳极孔半径范围: {inputs['rd_start']:.1f} - {inputs['rd_end']:.1f}, 步长: {inputs['rd_step']:.1f}")
            self.debug_print(f"Zac容差: {inputs['zac_tolerance']:.4f} mm")
            self.debug_print(f"rc互易容差: {inputs['rc_tolerance']:.4f} mm")
            
            # 第一次运行 (rw1)
            self.debug_print("\n=== 第一次运行 (rw1) ===")
            first_results = self.compute_pierce_gun(
                inputs["U"], inputs["I"], inputs["rw1"], 
                inputs["Jc"], inputs["Tao_val"], inputs["k_val"], rd_list
            )
            
            # 第二次运行 (rw2)
            self.debug_print("\n=== 第二次运行 (rw2) ===")
            second_results = self.compute_pierce_gun(
                inputs["U"], inputs["I"], rw2, 
                inputs["Jc"], inputs["Tao_val"], inputs["k_val"], rd_list
            )
            
            # 寻找最优解：Zac接近且[rd, rc]互易
            best_solution = None
            min_zac_diff = float('inf')
            
            for i, first in enumerate(first_results):
                for j, second in enumerate(second_results):
                    # 检查互易条件：first.rd ≈ second.rc 且 first.rc ≈ second.rd
                    rd_rc_match = abs(first['rd'] - second['rc']) < inputs["rc_tolerance"]
                    rc_rd_match = abs(first['rc'] - second['rd']) < inputs["rc_tolerance"]
                    
                    if rd_rc_match and rc_rd_match:
                        zac_diff = abs(first['Zac'] - second['Zac'])
                        
                        # 检查Zac差异是否在容差范围内
                        if zac_diff < inputs["zac_tolerance"] and zac_diff < min_zac_diff:
                            min_zac_diff = zac_diff
                            best_solution = {
                                'first': first,
                                'second': second,
                                'zac_diff': zac_diff,
                                'first_index': i,
                                'second_index': j
                            }
            
            # 更新结果表格
            self.update_table(self.first_table, first_results)
            self.update_table(self.second_table, second_results)
            
            # 显示最佳匹配
            if best_solution:
                first = best_solution['first']
                second = best_solution['second']
                
                result_text = (
                    f"找到最优解：\n"
                    f"第一次运行 (rw1={inputs['rw1']:.3f} mm): rd={first['rd']:.3f} mm, rc={first['rc']:.3f} mm, Zac={first['Zac']:.3f} mm\n"
                    f"第二次运行 (rw2={rw2:.3f} mm): rd={second['rd']:.3f} mm, rc={second['rc']:.3f} mm, Zac={second['Zac']:.3f} mm\n"
                    f"互易验证: [rd1={first['rd']:.3f} ≈ rc2={second['rc']:.3f}], [rc1={first['rc']:.3f} ≈ rd2={second['rd']:.3f}]\n"
                    f"Zac差异: {best_solution['zac_diff']:.6f} mm"
                )
                self.match_label.setText(result_text)
                self.debug_print("\n" + result_text)
                
                # 高亮显示表格中的最优解
                self.highlight_solution(self.first_table, best_solution['first_index'])
                self.highlight_solution(self.second_table, best_solution['second_index'])
            else:
                result_text = "未找到满足互易条件的最优解"
                self.match_label.setText(result_text)
                self.debug_print("\n" + result_text)
            
            self.progress_label.setText("计算完成")
            self.progress_label.setStyleSheet("color: green; font-weight: bold;")
        
        except Exception as e:
            self.progress_label.setText("计算错误")
            self.progress_label.setStyleSheet("color: red; font-weight: bold;")
            self.debug_print(f"计算错误: {str(e)}")
            QMessageBox.critical(self, "计算错误", f"计算过程中发生错误: {str(e)}")
    
    def update_table(self, table, results):
        """更新结果显示表格"""
        table_widget = table.findChild(QTableWidget)
        table_widget.setRowCount(len(results))
        
        for row, result in enumerate(results):
            table_widget.setItem(row, 0, QTableWidgetItem(f"{result['rd']:.3f}"))
            table_widget.setItem(row, 1, QTableWidgetItem(f"{result['rc']:.3f}"))
            table_widget.setItem(row, 2, QTableWidgetItem(f"{result['Zac']:.3f}"))
            table_widget.setItem(row, 3, QTableWidgetItem(f"{result['rw']:.3f}"))
    
    def highlight_solution(self, table, index):
        """高亮显示最优解在表格中的行"""
        table_widget = table.findChild(QTableWidget)
        for col in range(table_widget.columnCount()):
            item = table_widget.item(index, col)
            if item:
                item.setBackground(Qt.yellow)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PierceGunOptimizer()
    window.show()
    sys.exit(app.exec_())