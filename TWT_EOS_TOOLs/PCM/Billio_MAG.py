import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import fsolve

# 定义符号变量
B, L = sp.symbols('B L')

# 参数设置
U = 25000
I = 0.45
yita = 1.76e11
erb = 8.85e-12
Vb = np.sqrt(2 * yita * U)
t = 0.12e-3
w = 6 * t
S = w * t * np.pi / 4
p = I / (S * Vb)

# 定义公共表达式
Wo = yita * B
k = 2 * np.pi / (L * 1e-2)  # L单位转换为cm

# 定义所有隐函数方程
fun1 = Wo**2/(Vb*k)**2 - 3.64
fun2 = (Wo**2/(Vb*k)**2)*0.5 - (yita*p)/((Vb*k)**2*erb)*(w/(w+t))
fun3 = (Wo**2/(Vb*k)**2)*0.5 - (yita*p)/((Vb*k)**2*erb)*(w/(w+t)) - 1
fun4 = (Wo**2/(Vb*k)**2)*3.1/4 - (yita*p)/((Vb*k)**2*erb) - 1
fun6 = Wo-8*Vb/(5*t)

# 转换为数值函数
B_vals = np.linspace(-10, 10, 100)
L_vals = np.linspace(0.1, 10, 100)  # 避免除以零
B_grid, L_grid = np.meshgrid(B_vals, L_vals)

def eval_equation(eq):
    """将符号表达式转换为数值计算函数"""
    f = sp.lambdify((B, L), eq, modules='numpy')
    return f(B_grid, L_grid)

# 计算各方程的值
Z1 = eval_equation(fun1)
Z2 = eval_equation(fun2)
Z3 = eval_equation(fun3)
Z4 = eval_equation(fun4)
Z6 = eval_equation(fun6)

# 绘制隐函数曲线
plt.figure(figsize=(10, 6))

# 绘制等高线
plt.contour(B_grid, L_grid, Z1, levels=[0], colors='red', linewidths=2)
plt.contour(B_grid, L_grid, Z2, levels=[0], colors='#808000', linewidths=2)
plt.contour(B_grid, L_grid, Z3, levels=[0], colors='purple', linewidths=2)
plt.contour(B_grid, L_grid, Z4, levels=[0], colors='blue', linewidths=2)
plt.contour(B_grid, L_grid, Z6, levels=[0], colors='green', linewidths=2)

# 创建自定义图例项
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Z1'),
    Line2D([0], [0], color='#808000', lw=2, label='Z2'),
    Line2D([0], [0], color='purple', lw=2, label='Z3'),
    Line2D([0], [0], color='blue', lw=2, label='Z4'),
    Line2D([0], [0], color='green', lw=2, label='Z6')
]

# 计算关键磁场值
Bmax = float(2 * Vb / (yita * t))
Bbri = np.sqrt(1.414 * I / (w * t * erb * yita**1.5 * U**0.5))
Bo = 2 * Bbri

# 求解fun5的周期长度
WoS = Bo * yita
fun5 = (WoS**2/(Vb*k)**2)*3.1/4 - (yita*p)/((Vb*k)**2*erb) - 1
Lp = sp.solve(fun5, L)
Lp_val = abs(float(Lp[0]) * 10)  # 单位转换

# 计算其他参数
Wp = np.sqrt(yita * I / (S * Vb * erb))
lamda_p = 2 * np.pi * float(Vb) / Wp / 3 * 1e3  # 转换为毫米

# 输出结果
print(f'布里渊磁场: {Bbri:.4f} T')
print(f'最小峰值磁场: {Bo:.4f} T, 推荐充磁: {Bo/0.5:.4f} T')
print(f'推荐周期: {lamda_p:.2f} mm')
print(f'最大周期: {Lp_val/2:.2f} mm')

# 图形修饰
plt.xlabel('B (T)')
plt.ylabel('L (cm)')
plt.title('Magnetic Field vs Period Length')
plt.legend(handles=legend_elements, loc='best')
plt.grid(True)
plt.show()