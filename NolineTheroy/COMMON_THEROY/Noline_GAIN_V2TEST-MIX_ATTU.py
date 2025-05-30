import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys

from TWT_CORE_SIMP import simple_calculation
import math as m
import matplotlib.pyplot as plt
import numpy as np
from _TWT_CORE_NOLINE_COMPLEX_V2_MIX import solveTWTNOLINE_OUTPUT,solveTWTNOLINE_INIT


if __name__ == "__main__":


    # ========================= 多段参数配置 =========================
    SEGMENTS = [
        {   # 段0（初始段）
            "N_unit": 20,# 单元个数
            "Vpc": 0.285,# 归一化相速
            "Kc": 3.6,# 耦合阻抗
            "f0_GHz": 211,# 工作频率 (GHz)
            "Loss":0# 损耗
        },
        {   # 段1
            "N_unit": 20,# 单元个数
            "Vpc": 0.285,# 归一化相速
            "Kc": 3.6,# 耦合阻抗
            "f0_GHz": 211,# 工作频率 (GHz)
            "Loss":0# 损耗
        },
        # {   # 段2
        #     "N_unit": 20,# 单元个数
        #     "Vpc": 0.288,# 归一化相速
        #     "Kc": 3.6,# 耦合阻抗
        #     "f0_GHz": 211,# 工作频率 (GHz)
        #     "Loss":0# 损耗
        # },
        # {   # 段3
        #     "N_unit": 15,# 单元个数
        #     "Vpc": 0.288,# 归一化相速
        #     "Kc": 3.6,# 耦合阻抗
        #     "f0_GHz": 211,# 工作频率 (GHz)
        #     "Loss":0# 损耗
        # }
    ]
    Loss_attu = 20

    # ========================= 主计算逻辑 =========================
    if __name__ == "__main__":
        # 全局固定参数（各段共用）
        COMMON_PARAMS = {
            "I": 0.30,          # 电流 (A)
            "V": 23000,        # 电压 (V)
            "w": 0.20,         # 波导宽度
            "t": 0.20,         # 波导厚度
            "Fn_K": 1,         # 归一化因子
            "p_SWS": 0.50,     # 周期长度(mm)"
        }

        results = []  # 存储各段计算结果
        
        # 遍历所有段进行分步计算
        for seg_idx, seg in enumerate(SEGMENTS):
            # 构建当前段输入参数
            inputP = [
                COMMON_PARAMS["I"],
                COMMON_PARAMS["V"],
                seg["Kc"],
                seg["Loss"],
                COMMON_PARAMS["p_SWS"],
                seg["N_unit"],
                COMMON_PARAMS["w"],
                COMMON_PARAMS["t"],
                COMMON_PARAMS["Fn_K"],
                seg["f0_GHz"],
                seg["Vpc"]
            ]
            print(inputP)
            
            # 计算基础参数
            calc_result = simple_calculation(*inputP)
            C = calc_result["小信号增益因子C"]
            b = calc_result["非同步参量b"]
            L = 2 * np.pi * calc_result["互作用长度N"] * C
            
            # 首段特殊处理：计算全局参数和初始条件
            if seg_idx == 0:
                d = calc_result["损耗因子d"]
                if (inputP[6] == inputP[7]):  # 行波管等离子体频率降低系数R,ShitBeam用Fn,CicularBeam用Rowe特征值R
                    R = calc_result["Rowe特征值R"]
                else:
                    R = calc_result["等离子体频率降低因子Fn"]
                wp_omega = calc_result["等离子体频率Wp"] / (2 * np.pi * inputP[9] * 1e9)
                P_in=0.1
                P_flux = C * inputP[0] * inputP[1] * 2
                A0 = np.sqrt(P_in / P_flux)  # 初始振幅
                
                # 首段求解器参数
                params = {
                    "C": C, "b": b, "d": d, "wp_w": wp_omega,
                    "Rn": R, "m": 100, "A0": A0, "y_end": L
                }
                print(params)
                result = solveTWTNOLINE_INIT(**params)
            #输出段求解
            else:
                prev_result = results[seg_idx - 1]
                params = {
                    "C": C, "b": b, "d": d, "wp_w": wp_omega, "Rn": R, "m": 100,
                    "result_y_ends": prev_result["y"][-1],
                    "result_A_ends": prev_result["A_Ends"]*10**(-Loss_attu/20),
                    "result_dA_dy": prev_result["dA_dy_Ends"],
                    "result_theta": prev_result["theta_Ends"],
                    "result_dtheta_dy":prev_result["dtheta_dy_Ends"],
                    "result_u_finnal": prev_result["u_final"],
                    "result_phi_finnal": prev_result["phi_final"],
                    "y_end": prev_result["y"][-1] + L
                }
                result = solveTWTNOLINE_OUTPUT(**params)
                
            
            results.append(result)

        # ========================= 结果合成 =========================
        # 拼接各段数据
        Y_Finall = np.concatenate([r["y"] for r in results])
        A_Fianll = np.concatenate([r["A"] for r in results])
        theta_Fianll = np.concatenate([r["theta"] for r in results])
        u_Finall = np.concatenate([r["u_now"] for r in results])

        # 计算输出功率
        C_list = [simple_calculation(*[
            COMMON_PARAMS["I"], COMMON_PARAMS["V"],
            SEGMENTS[i]["Kc"], SEGMENTS[i]["Loss"],
            COMMON_PARAMS["p_SWS"], SEGMENTS[i]["N_unit"],
            COMMON_PARAMS["w"], COMMON_PARAMS["t"],
            COMMON_PARAMS["Fn_K"], seg["f0_GHz"],
            SEGMENTS[i]["Vpc"]
        ])["小信号增益因子C"] for i in range(len(SEGMENTS))]
        
        P_Out = 2 * COMMON_PARAMS["I"] * COMMON_PARAMS["V"] * np.concatenate(
            [C_list[i] * (results[i]["A"]**2) for i in range(len(SEGMENTS))]
        )

        P_max=max(P_Out);
        Eff_max=P_max/(inputP[0]*inputP[1])*100;

        Lenth=Y_Finall/(2*np.pi*np.mean(C_list));

        print('The Gain in Noline Theroy is %.4f \n'%(10*np.log10((P_Out[-1]/P_in))))
        print('The P_out in Noline Theroy is %.4f,The maximum Efficence is %.4f in percent,The maximum POWER is %.4f in Watt'%(P_Out[-1],Eff_max,P_max))
    # ========================= 输出与可视化 ========================= 
    # （与原代码可视化部分相同，此处省略）



# ========================= 结果可视化 =========================
plt.figure(figsize=(12, 8), dpi=100)

# 振幅演化
plt.subplot(2, 3, 1)
plt.plot(Y_Finall, A_Fianll, color='navy')
plt.xlabel('Position y', fontsize=10)
plt.ylabel('Amplitude A(y)', fontsize=10)
plt.title('Amplitude Growth', fontsize=12)
plt.grid(True, alpha=0.3)

# 相位演化
plt.subplot(2, 3, 2)
plt.plot(Y_Finall, theta_Fianll, color='maroon')
plt.xlabel('Position y', fontsize=10)
plt.ylabel('Phase Shift θ(y)', fontsize=10)
plt.title('Phase Evolution', fontsize=12)
plt.grid(True, alpha=0.3)

# 最终速度分布
plt.subplot(2, 3, 3)
plt.scatter(results[-1]['phi_final'], results[-1]['u_final'], 
            c=results[-1]['phi_final'], cmap='hsv', s=20, edgecolor='k', lw=0.5)
plt.colorbar(label='Final Phase ϕ(y_end)')
plt.xlabel('Final Phase ϕ(y_end)', fontsize=10)
plt.ylabel('Final Velocity u(y_end)', fontsize=10)
plt.title('Velocity Distribution', fontsize=12)
plt.grid(True, alpha=0.3)

# 最终相位分布
plt.subplot(2, 3, 4)
plt.scatter(results[-1]['phi0_grid'], results[-1]['phi_final'], 
            c=results[-1]['phi0_grid'], cmap='hsv', s=20, edgecolor='k', lw=0.5)
plt.colorbar(label='Initial Phase')
plt.xlabel('Initial Phase ϕ₀', fontsize=10)
plt.ylabel('Final Phase ϕ(y_end)', fontsize=10)
plt.title('Phase Distribution', fontsize=12)
plt.grid(True, alpha=0.3)

# 电子相空间图
plt.subplot(2, 3, 5)
plt.plot(Lenth, u_Finall, color='navy')
plt.xlabel('Position Z(Interaction Length)', fontsize=10)
plt.ylabel('Electron Velocity (u)', fontsize=10)
plt.title('Electron Phase Space', fontsize=12)
plt.grid(True, alpha=0.3)

# 轴向功率演化
plt.subplot(2, 3, 6)
plt.plot(Lenth, P_Out, color='darkgreen')
plt.xlabel('Position Z(Interaction Length)', fontsize=10)
plt.ylabel('Output Power (W)', fontsize=10)
plt.title('Power Evolution', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()