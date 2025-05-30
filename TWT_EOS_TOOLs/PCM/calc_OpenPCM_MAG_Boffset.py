import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 定义符号变量（单位：毫米）
Offset_PoleBot_Common = sp.symbols('Offset_PoleBot_Common')

# 转换为米
Offset_PoleBot_Common_m = Offset_PoleBot_Common

# 定义其他参数（单位：米）
x = 0.40#束流边缘位置
Ws = 2 * Offset_PoleBot_Common_m  # 计算宽度（米）
Bm = 2 * 0.5  # 1.0 Tesla

# 计算磁场偏移
By_offset = 1 * (sp.atan((Ws/2 + x)/(Bm/2)) - sp.atan((Ws/2 - x)/(Bm/2)))

# 转换为数值函数
By_offset_func = sp.lambdify(Offset_PoleBot_Common, By_offset, 'numpy')

# 生成数据点（0-10毫米）
offsets = np.linspace(0, 10, 400)
by_offsets = By_offset_func(offsets)

# 绘图
plt.plot(offsets, by_offsets)
plt.xlabel('Offset_PoleBot_Common /mm')
plt.ylabel('By_offset /T')
plt.grid(True)
plt.show()