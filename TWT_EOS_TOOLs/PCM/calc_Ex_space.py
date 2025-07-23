import numpy as np
from scipy.integrate import quad
import math

# 预置参数
U = 23000.0  # 电压，单位V
I = 0.45      # 电流，单位A
yita = 1.76e11  # 迁移率，单位 m^2/(V·s)
erb = 8.85e-12  # 真空介电常数，单位 F/m
Vb = np.sqrt(2 * yita * U)  # 基底速度，单位 m/s

# 参数输入
H = 0.15e-3  # 通道高度，单位m
b = 0.12e-3  # 带状束流厚度，单位m

for K_ab in range(6, 9):  # K_ab从6到8
    a = K_ab * b  # 带状束流宽度
    rx = a / H    # 归一化长度
    ry = b / H    # 归一化厚度
    S = math.pi * a * b / 4  # 束流横截面积
    rho = I / (S * Vb)       # 电荷密度
    X0 = rx  # 计算电场的位置（归一化）
    
    sum_total = 0.0  # 初始化总和

    for n in range(10):  # n从0到9
        # 定义被积函数（向量化处理）
        def integrand_Ia(t, rx_val, ry_val, n_val):
            t_sq_norm = (t**2) / (rx_val**2)
            yb = ry_val * np.sqrt(1.0 - t_sq_norm)
            freq = (2 * n_val + 1) * math.pi
            return np.sin(freq * yb) * np.exp(freq * (t - rx_val))

        # 计算Ia的积分（-rx到X0=rx）
        Ia_val, _ = quad(integrand_Ia, -rx, rx, args=(rx, ry, n))
        Ib_val = 0.0  # X0到rx积分区间为0

        # 累加电场分量
        E_x = 2 * (Ia_val - Ib_val) / ((2*n +1)*math.pi) * (rho * H) / erb
        sum_total += E_x

    # 计算并输出磁场
    By = sum_total / Vb
    By_Gs = By * 1e4  # 特斯拉转高斯
    print(f'聚焦带状束流所需磁场By是 {By_Gs:.6f} Gs (K_ab={K_ab})')