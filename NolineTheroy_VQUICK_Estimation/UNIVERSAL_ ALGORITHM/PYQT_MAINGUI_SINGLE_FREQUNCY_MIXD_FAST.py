import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_V2_MIX import solveTWTNOLINE_OUTPUT, solveTWTNOLINE_INIT, solveTWTNOLINE_Drift

class TWT_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("行波管计算器 - 高级版")
        self.root.geometry("1300x800")
        
        # 创建主框架
        mainframe = ttk.Frame(root, padding="10")
        mainframe.pack(fill=tk.BOTH, expand=True)
        
        # 左侧参数区域
        input_frame = ttk.LabelFrame(mainframe, text="参数设置")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # 右侧结果区域
        result_frame = ttk.LabelFrame(mainframe, text="计算结果")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建输入控件
        self.create_input_controls(input_frame)
        
        # 创建结果展示
        self.create_result_display(result_frame)
        
        # 设置默认值
        self.set_default_values()
    
    def create_input_controls(self, parent):
        """创建参数输入控件"""
        # 全局参数
        global_frame = ttk.LabelFrame(parent, text="全局参数")
        global_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 电流 I
        ttk.Label(global_frame, text="电流 I (A):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.I_var = tk.DoubleVar()
        ttk.Entry(global_frame, textvariable=self.I_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # 电压 V
        ttk.Label(global_frame, text="电压 V (V):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.V_var = tk.DoubleVar()
        ttk.Entry(global_frame, textvariable=self.V_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        # 宽度 w
        ttk.Label(global_frame, text="宽度 w (mm):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.w_var = tk.DoubleVar()
        ttk.Entry(global_frame, textvariable=self.w_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # 厚度 t
        ttk.Label(global_frame, text="厚度 t (mm):").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.t_var = tk.DoubleVar()
        ttk.Entry(global_frame, textvariable=self.t_var, width=10).grid(row=1, column=3, padx=5, pady=2)
        
        # Loss_attu
        ttk.Label(global_frame, text="Loss_attu:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.loss_attu_var = tk.DoubleVar()
        ttk.Entry(global_frame, textvariable=self.loss_attu_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # P_in
        ttk.Label(global_frame, text="P_in:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.P_in = tk.DoubleVar()
        ttk.Entry(global_frame, textvariable=self.P_in, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # 分段参数标题
        ttk.Label(parent, text="分段参数配置", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(15, 5))
        
        # 分段参数表
        columns = ["长度(mm)", "Vpc", "螺距(mm)", "耦合阻抗", "频率(GHz)", "每单元损耗", "填充因子", "类型"]
        self.segments_tree = ttk.Treeview(parent, columns=columns, show="headings", height=6)
        
        # 设置列标题
        for col in columns:
            self.segments_tree.heading(col, text=col)
            self.segments_tree.column(col, width=80, anchor=tk.CENTER)
        
        self.segments_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加分段管理按钮
        seg_button_frame = ttk.Frame(parent)
        seg_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(seg_button_frame, text="添加分段", command=self.add_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(seg_button_frame, text="编辑分段", command=self.edit_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(seg_button_frame, text="删除分段", command=self.delete_segment).pack(side=tk.LEFT, padx=5)
        
        # 计算按钮
        calc_frame = ttk.Frame(parent)
        calc_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(calc_frame, text="开始计算", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(calc_frame, text="重置", command=self.reset).pack(side=tk.LEFT, padx=5)
    
    def create_result_display(self, parent):
        """创建结果显示区域"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 文本结果标签
        text_frame = ttk.Frame(notebook)
        self.result_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)
        notebook.add(text_frame, text="计算结果")
        
        # 图形结果标签
        plot_frame = ttk.Frame(notebook)
        self.fig = plt.figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初始化图形
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "计算结果将在此处显示", 
                fontsize=12, ha='center', va='center', 
                transform=ax.transAxes, color='gray')
        ax.axis('off')
        self.canvas.draw()
        notebook.add(plot_frame, text="可视化结果")
    
    def set_default_values(self):
        """设置默认参数值"""
        # 全局参数默认值
        self.I_var.set(0.3)
        self.V_var.set(23000)
        self.w_var.set(0.20)
        self.t_var.set(0.20)
        self.loss_attu_var.set(0)
        
        # 默认分段
        default_segments = [
            {"len": 17, "Vpc": 0.288, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "Loss_perunit": 0, "Fn_K": 1, "type": "initial"},
            {"len": 1, "Vpc": 0.288, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "Loss_perunit": 0, "Fn_K": 1, "type": "attenuator"},
            {"len": 17, "Vpc": 0.288, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "Loss_perunit": 0, "Fn_K": 1, "type": "O"},
            {"len": 1, "Vpc": 0.288, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "Loss_perunit": 0, "Fn_K": 1, "type": "attenuator"},
            {"len": 17, "Vpc": 0.288, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "Loss_perunit": 0, "Fn_K": 1, "type": "O"},
        ]
        
        # 添加默认分段到树视图
        for segment in default_segments:
            values = [
                segment["len"],
                segment["Vpc"],
                segment["p_SWS"],
                segment["Kc"],
                segment["f0_GHz"],
                segment["Loss_perunit"],
                segment["Fn_K"],
                segment["type"],
            ]
            self.segments_tree.insert("", tk.END, values=values)
    
    def reset(self):
        """重置所有参数"""
        self.segments_tree.delete(*self.segments_tree.get_children())
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        
        # 重置图形
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "计算结果将在此处显示", 
                fontsize=12, ha='center', va='center', 
                transform=ax.transAxes, color='gray')
        ax.axis('off')
        self.canvas.draw()
        
        # 恢复默认值
        self.set_default_values()
    
    def add_segment(self):
        """添加新分段"""
        # 创建编辑对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("添加新分段")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 分段参数标签
        fields = [
            ("len", "长度 (mm):"),
            ("Vpc", "Vpc:"),
            ("p_SWS", "螺距 (mm):"),
            ("Kc", "耦合阻抗:"),
            ("f0_GHz", "频率 (GHz):"),
            ("Loss_perunit", "每单元损耗:"),
            ("Fn_K", "填充因子:"),
            ("type", "类型:")
        ]
        
        entries = {}
        
        # 创建输入字段
        for i, (key, label) in enumerate(fields):
            frame = ttk.Frame(dialog)
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            lbl = ttk.Label(frame, text=label, width=15, anchor=tk.E)
            lbl.pack(side=tk.LEFT, padx=5)
            
            if key == "type":  # 类型使用下拉框
                var = tk.StringVar(value="initial")
                combobox = ttk.Combobox(frame, textvariable=var, values=["initial", "attenuator", "O"], width=15)
                combobox.pack(side=tk.LEFT, padx=5)
                entries[key] = var
            else:
                var = tk.StringVar()
                entry = ttk.Entry(frame, textvariable=var, width=15)
                entry.pack(side=tk.LEFT, padx=5)
                entries[key] = var
        
        # 添加按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="添加", command=lambda: self.save_segment(entries, dialog)).pack(side=tk.LEFT, padx=20)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=20)
    
    def edit_segment(self):
        """编辑选中的分段"""
        selected = self.segments_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先选择一个分段")
            return
            
        item = selected[0]
        values = self.segments_tree.item(item, "values")
        
        # 创建编辑对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("编辑分段")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 分段参数标签
        fields = [
            ("len", "长度 (mm):"),
            ("Vpc", "Vpc:"),
            ("p_SWS", "螺距 (mm):"),
            ("Kc", "耦合阻抗:"),
            ("f0_GHz", "频率 (GHz):"),
            ("Loss_perunit", "每单元损耗:"),
            ("Fn_K", "填充因子:"),
            ("type", "类型:")
        ]
        
        entries = {}
        
        # 创建输入字段并填充当前值
        for i, (key, label) in enumerate(fields):
            frame = ttk.Frame(dialog)
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            lbl = ttk.Label(frame, text=label, width=15, anchor=tk.E)
            lbl.pack(side=tk.LEFT, padx=5)
            
            if key == "type":  # 类型使用下拉框
                var = tk.StringVar(value=values[i])
                combobox = ttk.Combobox(frame, textvariable=var, values=["initial", "attenuator", "O"], width=15)
                combobox.pack(side=tk.LEFT, padx=5)
                entries[key] = var
            else:
                var = tk.StringVar(value=values[i])
                entry = ttk.Entry(frame, textvariable=var, width=15)
                entry.pack(side=tk.LEFT, padx=5)
                entries[key] = var
        
        # 添加按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="保存", command=lambda: self.update_segment(item, entries, dialog)).pack(side=tk.LEFT, padx=20)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=20)
    
    def delete_segment(self):
        """删除选中的分段"""
        selected = self.segments_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先选择一个分段")
            return
            
        for item in selected:
            self.segments_tree.delete(item)
    
    def save_segment(self, entries, dialog):
        """保存新分段"""
        values = [
            entries["len"].get(),
            entries["Vpc"].get(),
            entries["p_SWS"].get(),
            entries["Kc"].get(),
            entries["f0_GHz"].get(),
            entries["Loss_perunit"].get(),
            entries["Fn_K"].get(),
            entries["type"].get()
        ]
        
        try:
            # 验证数值型参数
            float(entries["len"].get())
            float(entries["Vpc"].get())
            float(entries["p_SWS"].get())
            float(entries["Kc"].get())
            float(entries["f0_GHz"].get())
            float(entries["Loss_perunit"].get())
            float(entries["Fn_K"].get())
        except ValueError:
            messagebox.showerror("输入错误", "所有参数应为数值类型")
            return
            
        self.segments_tree.insert("", tk.END, values=values)
        dialog.destroy()
    
    def update_segment(self, item, entries, dialog):
        """更新分段参数"""
        values = [
            entries["len"].get(),
            entries["Vpc"].get(),
            entries["p_SWS"].get(),
            entries["Kc"].get(),
            entries["f0_GHz"].get(),
            entries["Loss_perunit"].get(),
            entries["Fn_K"].get(),
            entries["type"].get()
        ]
        
        try:
            # 验证数值型参数
            float(entries["len"].get())
            float(entries["Vpc"].get())
            float(entries["p_SWS"].get())
            float(entries["Kc"].get())
            float(entries["f0_GHz"].get())
            float(entries["Loss_perunit"].get())
            float(entries["Fn_K"].get())
        except ValueError:
            messagebox.showerror("输入错误", "所有参数应为数值类型")
            return
            
        self.segments_tree.item(item, values=values)
        dialog.destroy()
    
    def calculate(self):
        """执行计算"""
        try:
            # 获取全局参数
            COMMON_PARAMS = {
                "I": float(self.I_var.get()),
                "V": float(self.V_var.get()),
                "w": float(self.w_var.get()),
                "t": float(self.t_var.get()),
                "P_in":float(self.P_in.get()),
            }
            
            # 获取Loss_attu
            Loss_attu = float(self.loss_attu_var.get())
            
            # 获取分段参数
            SEGMENTS = []
            for item in self.segments_tree.get_children():
                values = self.segments_tree.item(item, "values")
                segment = {
                    "len": float(values[0]),
                    "Vpc": float(values[1]),
                    "p_SWS": float(values[2]),
                    "Kc": float(values[3]),
                    "f0_GHz": float(values[4]),
                    "Loss_perunit": float(values[5]),
                    "Fn_K": float(values[6]),
                    "type": values[7]
                }
                SEGMENTS.append(segment)
            
            if not SEGMENTS:
                raise ValueError("请添加至少一个分段")
            
            # 准备结果显示
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "开始计算...\n")
            self.result_text.insert(tk.END, f"全局参数: {COMMON_PARAMS}\n")
            self.result_text.insert(tk.END, f"分段数量: {len(SEGMENTS)}\n")
            self.result_text.update()
            
            # ========================= 主计算逻辑 =========================
            results = []
            C_list = []
            
            for seg_idx, seg in enumerate(SEGMENTS):
                # 更新状态
                self.result_text.insert(tk.END, f"\n处理分段 #{seg_idx+1} ({seg['type']})...\n")
                self.result_text.see(tk.END)
                self.result_text.update()
                
                # 参数计算
                input_params = self.build_input_params(COMMON_PARAMS, seg)
                calc_result = simple_calculation(*input_params)
                
                # 缓存公共参数
                C = calc_result["小信号增益因子C"]
                L = 2 * np.pi * calc_result["互作用长度N"] * C
                C_list.append(C)
                params = {
                    "C": C, 
                    "b": calc_result["非同步参量b"],
                    "d": calc_result["损耗因子d"],
                    "wp_w": calc_result["等离子体频率Wp"] / (2 * np.pi * seg["f0_GHz"] * 1e9),
                    "Rn": self.get_plasma_factor(calc_result, COMMON_PARAMS["w"], COMMON_PARAMS["t"]),
                    "m": 100,
                    "y_end": L + (results[-1]["y"][-1] if seg_idx > 0 else 0)
                }

                # 分段处理
                if seg["type"] == "initial":
                    self.handle_initial_segment(params, COMMON_PARAMS, calc_result, results)
                elif seg["type"] == "attenuator":
                    self.handle_attenuator_segment(params, results, seg_idx, Loss_attu, L)
                else:
                    self.handle_normal_segment(params, results, seg_idx)
            
            # ========================= 结果处理与可视化 =========================
            self.process_and_visualize(results, C_list, COMMON_PARAMS, SEGMENTS)
            
        except Exception as e:
            self.result_text.insert(tk.END, f"\n错误: {str(e)}\n")
            self.result_text.see(tk.END)
            self.result_text.config(state=tk.DISABLED)
            messagebox.showerror("计算错误", f"发生错误: {str(e)}")
    
    def build_input_params(self, common_params, seg):
        """构建输入参数列表"""
        return [
            common_params["I"], 
            common_params["V"], 
            seg["Kc"], 
            seg["Loss_perunit"],
            seg["p_SWS"], 
            seg["len"], 
            common_params["w"], 
            common_params["t"],
            seg["Fn_K"], 
            seg["f0_GHz"], 
            seg["Vpc"]
        ]
    
    def get_plasma_factor(self, calc_result, w, t):
        """获取等离子体频率因子"""
        return calc_result["Rowe特征值R"] if w == t else calc_result["等离子体频率降低因子Fn"]
    
    def handle_initial_segment(self, params, common_params, calc_result, results):
        """处理初始段"""
        P_in=common_params["P_in"]
        P_flux = params["C"] * common_params["I"] * common_params["V"] * 2
        params.update({
            "A0": np.sqrt(P_in / P_flux),
            "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"]
        })
        
        # 显示日志
        self.result_text.insert(tk.END, "初始段参数:\n")
        for k, v in params.items():
            self.result_text.insert(tk.END, f"  {k}: {v}\n")
        self.result_text.see(tk.END)
        self.result_text.update()
        
        results.append(solveTWTNOLINE_INIT(**params))
    
    def handle_attenuator_segment(self, params, results, seg_idx, Loss_attu, L):
        """处理衰减段"""
        prev = results[seg_idx-1]
        d_attu = 0.01836 * Loss_attu / (L / (2 * np.pi))
        
        self.result_text.insert(tk.END, f"衰减段损耗系数: {d_attu}\n")
        self.result_text.update()
        
        params.update({
            "result_y_ends": prev["y"][-1],
            "result_A_ends": prev["A_Ends"]* 10**(-Loss_attu/20),
            "result_dA_dy": prev["dA_dy_Ends"]*0,
            "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"],
            "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_Drift(**params))
    
    def handle_normal_segment(self, params, results, seg_idx):
        """处理常规段"""
        prev = results[seg_idx-1]
        params.update({
            "result_y_ends": prev["y"][-1],
            "result_A_ends": prev["A_Ends"],
            "result_dA_dy": prev["dA_dy_Ends"],
            "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"],
            "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_OUTPUT(**params))
    
    def process_and_visualize(self, results, C_list, common_params, segments):
        """结果处理与可视化"""
        # 数据合成
        Y_Finall = np.concatenate([r["y"] for r in results])
        A_Fianll = np.concatenate([r["A"] for r in results])
        theta_Fianll = np.concatenate([r["theta"] for r in results])
        u_Finall = np.concatenate([r["u_now"] for r in results])
        
        # 功率计算
        P_Out = 2 * common_params["I"] * common_params["V"] * np.concatenate(
            [C_list[i] * (results[i]["A"]**2) for i in range(len(segments))]
        )
        
        # 性能指标
        P_max = P_Out.max()
        Eff_max = P_max / (common_params["I"] * common_params["V"]) * 100
        Lenth = Y_Finall / (2 * np.pi * np.mean(C_list))
        
        # 显示最终结果
        self.result_text.insert(tk.END, "\n======== 最终计算结果 ========\n")
        self.result_text.insert(tk.END, f"非线性理论增益: {10 * np.log10(P_Out[-1]/0.1):.4f} dB\n")
        self.result_text.insert(tk.END, f"输出功率: {P_Out[-1]:.4f} W\n")
        self.result_text.insert(tk.END, f"最大效率: {Eff_max:.4f}%\n")
        self.result_text.insert(tk.END, f"最大功率: {P_max:.4f} W\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
        
        # 可视化结果
        self.plot_results(Y_Finall, A_Fianll, theta_Fianll, u_Finall, Lenth, P_Out, results[-1])
    
    def plot_results(self, Y, A, theta, u, Lenth, P_Out, final_seg):
        """可视化绘图"""
        self.fig.clear()
        
        # 振幅演化
        ax1 = self.fig.add_subplot(2, 3, 1)
        ax1.plot(Y, A, 'navy')
        ax1.set_xlabel("Position y", fontsize=10)
        ax1.set_ylabel("Amplitude A(y)", fontsize=10)
        ax1.set_title("Amplitude Growth", fontsize=12)
        ax1.grid(alpha=0.3)

        # 相位演化
        ax2 = self.fig.add_subplot(2, 3, 2)
        ax2.plot(Y, theta, 'maroon')
        ax2.set_xlabel("Position y", fontsize=10)
        ax2.set_ylabel("Phase Shift θ(y)", fontsize=10)
        ax2.set_title("Phase Evolution", fontsize=12)
        ax2.grid(alpha=0.3)

        # 速度分布
        ax3 = self.fig.add_subplot(2, 3, 3)
        scatter = ax3.scatter(
            final_seg["phi_final"], final_seg["u_final"],
            c=final_seg["phi_final"], cmap='hsv', s=20, edgecolor='k', lw=0.5
        )
        self.fig.colorbar(scatter, ax=ax3, label="Final Phase ϕ(y_end)")
        ax3.set_xlabel("Final Phase ϕ(y_end)", fontsize=10)
        ax3.set_ylabel("Final Velocity u(y_end)", fontsize=10)
        ax3.set_title("Velocity Distribution", fontsize=12)
        ax3.grid(alpha=0.3)

        # 相位分布
        ax4 = self.fig.add_subplot(2, 3, 4)
        scatter = ax4.scatter(
            final_seg["phi0_grid"], final_seg["phi_final"],
            c=final_seg["phi0_grid"], cmap='hsv', s=20, edgecolor='k', lw=0.5
        )
        self.fig.colorbar(scatter, ax=ax4, label="Initial Phase")
        ax4.set_xlabel("Initial Phase ϕ₀", fontsize=10)
        ax4.set_ylabel("Final Phase ϕ(y_end)", fontsize=10)
        ax4.set_title("Phase Distribution", fontsize=12)
        ax4.grid(alpha=0.3)

        # 电子相空间
        ax5 = self.fig.add_subplot(2, 3, 5)
        ax5.plot(Lenth, u, 'navy')
        ax5.set_xlabel("Position Z(Interaction Length)", fontsize=10)
        ax5.set_ylabel("Electron Velocity (u)", fontsize=10)
        ax5.set_title("Electron Phase Space", fontsize=12)
        ax5.grid(alpha=0.3)

        # 功率演化
        ax6 = self.fig.add_subplot(2, 3, 6)
        ax6.plot(Lenth, P_Out, 'darkgreen')
        ax6.set_xlabel("Position Z(Interaction Length)", fontsize=10)
        ax6.set_ylabel("Output Power (W)", fontsize=10)
        ax6.set_title("Power Evolution", fontsize=12)
        ax6.grid(alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TWT_GUI(root)
    root.mainloop()