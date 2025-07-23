import numpy as np
from scipy.optimize import fsolve
import math

def compute_pierce_gun(U, I, rw, Jc, Tao_val, k_val, rd_list):
    results = []
    for rd in rd_list:
        uP = 1e6 * I / (U ** 1.5)
        theta = 30 * math.sqrt(uP)
        
        # 迭代求解theta
        while True:
            def equation_Rc(x):
                return math.pi * x * math.sin(math.radians(theta)) * rd - I / Jc
            Rc = fsolve(equation_Rc, 1)[0]
            
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
            theta_prev = theta
            if Tan_PhiA == 0:
                theta = theta
            else:
                theta = theta * math.sqrt(Tan_PhiB / abs(Tan_PhiA))
            
            # 检查收敛条件
            if abs(Tan_PhiA / Tan_PhiB - 1) <= 0.005:
                break
        
        # 计算最终参数
        Ra = Rc * math.exp(-gamma)
        Zac = Rc - Ra
        
        results.append({
            'rd': rd,
            'rc': rc,
            'Zac': Zac
        })
    return results

# 参数设置
U = 25e3  # 电压，单位V(用户输入)
I = 0.04  # 电流，单位A(用户输入)
Jc = 0.16  # 电流密度，单位A/mm²(用户输入)
Tao_val = 1.25  # 参数Tao
k_val = 0.905  # 参数k

# 第一次运行：rw=0.1，遍历rd
rw1 = 0.03#(用户输入)
rd_values_first = np.arange(0.2, 2, 0.1)
print(f"第一次运行：rw={rw1}，遍历rd")
first_results = compute_pierce_gun(U, I, rw1, Jc, Tao_val, k_val, rd_values_first)
rc_list = [result['rc'] for result in first_results]

# 第二次运行：rw=0.4，遍历第一次得到的rc作为rd
rw2 = 5 * rw1#(用户输入)
print(f"\n第二次运行：rw={rw2}，遍历rc作为rd")
second_results = compute_pierce_gun(U, I, rw2, Jc, Tao_val, k_val, rc_list)

# 找到Zac最接近的情况
min_diff = float('inf')
best_pair = None

for i, first in enumerate(first_results):
    for j, second in enumerate(second_results):
        diff = abs(first['Zac'] - second['Zac'])
        if diff < min_diff:
            min_diff = diff
            best_pair = (i, j, first, second)

if best_pair:
    i, j, first, second = best_pair
    print(f"\n最接近的情况：")
    print(f"第一次运行: rd={first['rd']:.1f} mm, Zac={first['Zac']:.2f} mm")
    print(f"第二次运行: rd={second['rd']:.2f} mm（对应第一次的rc={rc_list[i]:.2f} mm）, Zac={second['Zac']:.2f} mm")
    print(f"两次Zac差异: {min_diff:.4f} mm")
else:
    print("未找到匹配的结果。")