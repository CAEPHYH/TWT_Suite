import numpy as np
from scipy.optimize import fsolve
import math

# 定义单位：U (V), I (A), J (A/mm²), 长度 (mm), 角度 (度)

print('compute start')
U = 11e3  # 电压，单位V(用户输入)
I = 0.10  # 电流，单位A(用户输入)
Jc = 0.12  # 电流密度，单位A/mm²(用户输入)
rw = 0.05 # 阳极孔半径，单位mm(用户输入)
Tao = 1.25
k = 0.905

for rd in np.arange(0.1, 1, 0.02):  # rd从1到3，步长0.25
    # 初始计算
    uP = 1e6 * I / (U ** 1.5)
    theta = 30 * math.sqrt(uP)
    
    # 迭代求解theta
    while True:
        # 求解Rc的方程
        def equation_Rc(x):
            return math.pi * x * math.sin(math.radians(theta)) * (rd) - I/Jc
        Rc = fsolve(equation_Rc, 1)[0]
        
        rc = Rc * math.sin(math.radians(theta))
        alpha = math.sqrt(14.67 * (1 - math.cos(math.radians(theta))) / uP)
        gamma = alpha - 0.275 * (alpha ** 2) + 0.06 * (alpha ** 3) - 0.006 * (alpha ** 4)
        rb_Za = rc * math.exp(-gamma)
        
        # 计算Tan_PhiA和Tan_PhiB
        numerator = Tao * (1 + 0.6*gamma + 0.225*gamma**2 + 0.0573*gamma**3 + 
                          0.0108*gamma**4 + 0.0021*gamma**5)
        Tan_PhiA = math.sin(math.radians(theta)) * (1 - numerator / (3 * alpha))
        
        R = rb_Za / rw
        Tan_PhiB = 0.17409 * math.sqrt(uP * abs(math.log(R)))
        
        # 更新theta
        theta_prev = theta
        theta = theta * math.sqrt(Tan_PhiB / abs(Tan_PhiA))
        
        # 检查收敛条件
        if abs(Tan_PhiA / Tan_PhiB - 1) <= 0.005:
            break
    
    # 计算最终参数
    Ra = Rc * math.exp(-gamma)
    ra = Ra * math.sin(math.radians(theta))
    Zac = Rc - Ra
    R = rb_Za / rw
    Z = 11.0 * math.sqrt(R - 1) + 1.32 * (R - 1) + 0.0615 * (R - 1)**2 - 0.00167 * (R - 1)**3
    Za = Rc - math.sqrt(Ra**2 - ra**2)
    Zw = Zac + rw * Z / math.sqrt(uP)
    thetaT = math.degrees(math.asin(math.sin(math.radians(theta)) / k))
    RcT = rc / math.sin(math.radians(thetaT))
    
    # 输出结果
    print(f'\ncompute finished')
    print(f'皮尔斯电子枪计算结果如下: 阴极长半径 rd={rd:.2f} mm')
    print(f'注锥角 theta = {theta:.2f}°')
    print(f'阴极球半径 Rc = {Rc:.2f} mm')
    print(f'阳极球半径 Ra = {Ra:.2f} mm')
    print(f'阴极短半径 rc = {rc:.2f} mm')
    print(f'阳极孔半径 ra = {ra:.2f} mm')
    print(f'阴阳极间距 Zac = {Zac:.2f} mm')
    print(f'束腰位置 Zw = {Zw:.2f} mm')
    print(f'Rb_Za = {rb_Za:.2f} mm')

print('\ncompute all finished')