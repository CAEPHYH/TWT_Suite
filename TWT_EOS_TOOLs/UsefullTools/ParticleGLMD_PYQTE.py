import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.neighbors import KernelDensity
import tkinter as tk
from tkinter import filedialog, ttk
import re
import pandas as pd
import os
from datetime import datetime


class ParticleDistributionAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Distribution Analyzer")
        self.sections = []
        self.current_section = 0
        self.setup_ui()

    def setup_ui(self):
        # File control
        self.control_frame = ttk.Frame(self.root)
        self.btn_load = ttk.Button(
            self.control_frame, text="Load Data", command=self.load_data
        )
        self.lbl_status = ttk.Label(self.control_frame, text="Status: Ready")
        self.lbl_rect = ttk.Label(self.control_frame, text="等效矩形尺寸: N/A")
        self.btn_load.pack(side=tk.LEFT, padx=5)
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        self.lbl_rect.pack(side=tk.LEFT, padx=10)
        self.control_frame.pack(fill=tk.X, pady=5)

        # Navigation
        self.nav_frame = ttk.Frame(self.root)
        self.btn_prev = ttk.Button(
            self.nav_frame,
            text="◀ Previous",
            state="disabled",
            command=lambda: self.change_section(-1),
        )
        self.btn_next = ttk.Button(
            self.nav_frame,
            text="Next ▶",
            state="disabled",
            command=lambda: self.change_section(1),
        )
        self.lbl_page = ttk.Label(self.nav_frame, text="Section: 0/0")
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.btn_next.pack(side=tk.LEFT, padx=2)
        self.lbl_page.pack(side=tk.LEFT, padx=10)
        self.nav_frame.pack(fill=tk.X, pady=5)

        # Plot area
        self.fig = Figure(figsize=(12, 9))
        self.gs = self.fig.add_gridspec(3, 2, height_ratios=[2, 2, 1])

        # Top-left: Global average
        self.ax_global = self.fig.add_subplot(self.gs[0, 0])
        self.ax_global.set_title("Global Average Density")

        # Top-right: Contour plot
        self.ax_contour = self.fig.add_subplot(self.gs[0, 1])
        self.ax_contour.set_title("Normalized Contour")

        # Bottom: Current section
        self.ax_current = self.fig.add_subplot(self.gs[1, :])
        self.ax_current.set_title("Current Section Density")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return

        try:
            self.sections = []
            self.parse_file(file_path)
            self.prepare_grid()
            self.compute_densities()
            self.plot_global_average()
            self.plot_contour()
            self.plot_current_section()
            self.update_navigation()
            self.lbl_status.config(
                text=f"Loaded: {file_path}\n Saved global_density.csv"
            )
        except Exception as e:
            self.lbl_status.config(text=f"Error: {str(e)}")

    def parse_file(self, path):
        current_section = None
        all_points = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                # 处理分区标记行（例如： #"Position_x"@ 10.5 mm）
                if line.startswith('#"Position_x'):
                    if current_section:
                        current_section["data"] = np.array(current_section["data"])
                        self.sections.append(current_section)
                    # 使用正则表达式提取z轴位置（单位假设为mm）
                    z_pos = float(re.search(r"@\s*([\d.]+)\s*mm", line).group(1))
                    current_section = {"z": z_pos, "data": []}
                elif line and not line.startswith("#---"):
                    x, y = map(float, line.split())
                    current_section["data"].append([x, y])
                    all_points.append([x, y])

        if current_section:
            current_section["data"] = np.array(current_section["data"])
            self.sections.append(current_section)

        all_points = np.array(all_points)
        self.xlim = (all_points[:, 0].min(), all_points[:, 0].max())
        self.ylim = (all_points[:, 1].min(), all_points[:, 1].max())

    def prepare_grid(self, resolution=100):
        self.x_grid = np.linspace(*self.xlim, resolution)
        self.y_grid = np.linspace(*self.ylim, resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.grid_points = np.vstack([self.X.ravel(), self.Y.ravel()]).T

    def _compute_section_density(self, section, bandwidth):
        """计算单个section的KDE密度"""
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(section["data"])
        log_dens = kde.score_samples(self.grid_points)
        return np.exp(log_dens).reshape(self.X.shape)

    def compute_densities(self):
        # 合并所有样本并计算动态带宽
        data = np.vstack([sec["data"] for sec in self.sections])
        n_samples, n_features = data.shape
        scott_factor = n_samples ** (-1.0 / (n_features + 4))
        bandwidth = scott_factor * np.std(data, axis=0).mean()
        # 预计算网格参数并初始化
        dx, dy = (self.x_grid[1] - self.x_grid[0], self.y_grid[1] - self.y_grid[0])
        density_list = []
        normalized_densities = []

        for sec in self.sections:
            # KDE拟合和预测
            density = self._compute_section_density(sec, bandwidth)
            # 积分校验和记录
            integral = (density.sum() * dx * dy).item()
            print(f"积分校验: {integral:.4f}")
            # 保存结果
            sec["density"] = density
            density_list.append(density)
            normalized_densities.append(density / density.sum())  # 使用积分结果归一化
            # print(normalized_densities)

        # 计算全局密度分布
        self.global_density = np.mean(density_list, axis=0)
        self.normalized_density = np.mean(normalized_densities, axis=0)

        self.calculate_equivalent_rectangle()

    def calculate_equivalent_rectangle(self, coverage=0.9):
        """整合后的等效矩形计算函数"""
        try:
            # 原calculate_principal_axis的内容
            weights = self.global_density / self.global_density.sum()
            x_avg = np.sum(self.X * weights)
            y_avg = np.sum(self.Y * weights)

            # 计算协方差矩阵
            cov = np.cov(
                np.vstack(((self.X - x_avg).ravel(), (self.Y - y_avg).ravel())),
                aweights=weights.ravel(),
            )

            # 特征分解获取主轴方向
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_axis = eigvecs[:, np.argmax(eigvals)]  # 主特征向量

            # 沿主轴投影
            proj = main_axis[0] * (self.X - x_avg) + main_axis[1] * (self.Y - y_avg)

            # 累积概率分布
            sorted_idx = np.argsort(proj.ravel())
            cum_prob = np.cumsum(self.global_density.ravel()[sorted_idx])
            cum_prob /= cum_prob[-1]

            # 查找覆盖指定概率的区间
            low_idx = np.searchsorted(cum_prob, 0.5 * (1 - coverage))
            high_idx = np.searchsorted(cum_prob, 1 - 0.5 * (1 - coverage))
            effective_length = (
                proj.ravel()[sorted_idx[high_idx]] - proj.ravel()[sorted_idx[low_idx]]
            )

            # 原calculate_equivalent_rectangle的内容
            theta = np.arctan2(main_axis[1], main_axis[0])
            L = effective_length
            # 次轴方向基于协方差比例
            W = L * np.sqrt(eigvals[0] / eigvals[1])  # 使用实际特征值

            # 转换到原始坐标系尺寸
            equiv_width = L * np.abs(np.cos(theta)) + W * np.abs(np.sin(theta))
            equiv_height = L * np.abs(np.sin(theta)) + W * np.abs(np.cos(theta))

            self.lbl_rect.config(
                text=f"等效矩形尺寸: {equiv_width:.2f} mm × {equiv_height:.2f} mm"
            )
        except Exception as e:
            print(f"计算错误: {str(e)}")
            self.lbl_rect.config(text="等效尺寸计算失败")

    def plot_global_average(self):
        """Top-left: Global average heatmap"""
        self.ax_global.clear()
        im = self.ax_global.pcolormesh(
            self.X, self.Y, self.global_density, cmap="viridis", shading="auto"
        )
        self.fig.colorbar(im, ax=self.ax_global, label="Density")
        self.ax_global.set_xlabel("X (mm)")
        self.ax_global.set_ylabel("Y (mm)")
        # 正确导出数据（直接展平现有网格）
        ploteddata = {
            "X (mm)": self.X.flatten(),  # 展平二维网格
            "Y (mm)": self.Y.flatten(),  # 展平二维网格
            "Density": self.global_density.flatten(),  # 展平二维密度
        }

        dataframe = pd.DataFrame(ploteddata)
        os.makedirs("./Results", exist_ok=True)  # 确保目录存在
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"global_density_{timestamp}.csv"
        dataframe.to_csv(os.path.join(".", "Results", filename), index=False)

    def plot_contour(self):
        """Top-right: 全局密度累积分布等高线图"""
        self.ax_contour.clear()

        # 生成 p 值（1.0 到 0.0，步长 0.05）
        p_values = np.linspace(1.0, 0.0, 21)

        # 计算累积分布阈值
        sorted_density = np.sort(self.normalized_density.flatten())[::-1]
        cumulative = np.cumsum(sorted_density)

        # 向量化计算每个 p 对应的 level
        indices = np.searchsorted(cumulative, p_values, side="left")
        indices = np.clip(indices, 0, len(cumulative) - 1)
        levels = sorted_density[indices]

        # 去重，保留每个 level 的最大 p
        level_to_p = {}
        for p, level in zip(p_values, levels):
            if level not in level_to_p:
                level_to_p[level] = p

        # 按 level 升序排列
        levels = sorted(level_to_p.keys())
        p_for_levels = [level_to_p[lv] for lv in levels]

        # 绘制等高线
        cf = self.ax_contour.contourf(
            self.X, self.Y, self.normalized_density, levels=levels, cmap="plasma"
        )
        cs = self.ax_contour.contour(
            self.X, self.Y, self.normalized_density, 
            levels=levels, colors="white", linewidths=0.5
        )

        # 生成标签
        fmt = {l: f"{p*100:.0f}%" for l, p in zip(levels, p_for_levels)}
        self.ax_contour.clabel(cs, levels=levels[::2], inline=True, fmt=fmt, fontsize=10)

        # 导出等高线数据
        os.makedirs("./Results", exist_ok=True)
        contour_data = [
            {"level": lv, "x": x, "y": y}
            for i, lv in enumerate(cs.levels)
            for seg in cs.allsegs[i]
            for x, y in seg
        ]
        pd.DataFrame(contour_data).to_csv(
            f"./Results/contour_linesQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            index=False
        )

        # 设置图表
        self.fig.colorbar(cf, ax=self.ax_contour, label="Probability Level")
        self.ax_contour.set_title("Global Density Cumulative Contour")
        self.ax_contour.set_xlabel("X (mm)")
        self.ax_contour.set_ylabel("Y (mm)")

    def plot_current_section(self):
        """Bottom: Current section plot"""
        self.ax_current.clear()
        sec = self.sections[self.current_section]

        im = self.ax_current.pcolormesh(
            self.X, self.Y, sec["density"], cmap="viridis", shading="auto"
        )
        self.fig.colorbar(im, ax=self.ax_current, label="Density")
        self.ax_current.set_title(f"Current Section @ {sec['z']:.2f} mm")
        self.ax_current.set_xlabel("X (mm)")
        self.ax_current.set_ylabel("Y (mm)")
        self.canvas.draw()

    def update_navigation(self):
        total = len(self.sections)
        self.lbl_page.config(text=f"Section: {self.current_section+1}/{total}")
        self.btn_prev["state"] = "normal" if self.current_section > 0 else "disabled"
        self.btn_next["state"] = (
            "normal" if self.current_section < total - 1 else "disabled"
        )

    def change_section(self, delta):
        new_index = self.current_section + delta
        if 0 <= new_index < len(self.sections):
            self.current_section = new_index
            self.plot_current_section()
            self.update_navigation()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1300x900")
    app = ParticleDistributionAnalyzer(root)
    root.mainloop()
