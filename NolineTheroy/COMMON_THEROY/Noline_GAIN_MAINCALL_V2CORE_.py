import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
# import sys
# 
from TWT_CORE_SIMP import simple_calculation
import math as m
import matplotlib.pyplot as plt
import numpy as np
from _TWT_CORE_NOLINE_COMPLEX_V2_MIX import solveTWTNOLINE_INIT


def Noline_CORE_CALC(I, V, Kc, Loss, p_SWS, N_unit, w, t, Fn_K, f0_GHz, Vpc):
    try:
        inputP = [
            I,
            V,
            Kc,
            Loss,
            p_SWS,
            N_unit,
            w,
            t,
            Fn_K,
            f0_GHz,
            Vpc,
        ]
        print(inputP)  
        # 计算物理参数
        initparamater = simple_calculation(*inputP)
        # return {
        #     "小信号增益因子C": C,
        #     "互作用长度N": N,
        #     "慢波线上损耗L": L,
        #     "损耗因子d": d,
        #     "等离子体频率降低因子Fn": Fn,
        #     "等离子体频率Wp": Wp,
        #     "空间电荷参量4QC": Wq_over_omegaC_sq,
        #     "非同步参量b": b,
        #     "归一化电子速度Vec": Vec,
        #     "束流归一化尺寸r_beam": r_beam,
        #     "beta_e": beta_e,
        #     "Rowe特征值R": R,
        # }


        C = initparamater["小信号增益因子C"]
        # 行波管增益参数C
        b = initparamater["非同步参量b"]
        # 行波管非同步参量b
        d = initparamater["损耗因子d"]
        # 损耗系数d
        wp_omega = initparamater["等离子体频率Wp"] / (2 * np.pi * inputP[9] * 1e9)
        # 相对等离子体频率
        if (inputP[6] == inputP[7]):  # 行波管等离子体频率降低系数R,ShitBeam用Fn,CicularBeam用Rowe特征值R
            R = initparamater["Rowe特征值R"]
        else:
            R = initparamater["等离子体频率降低因子Fn"]
        L = 2 * np.pi * initparamater["互作用长度N"] * C

        P_in = 0.10
        P_flux = C * inputP[0] * inputP[1] * 2
        A_0 = np.sqrt(P_in / P_flux)

        # ================================
        # 参数配置区（根据物理系统调整）
        # ================================
        # 参数设置 (根据实际需求修改)

        params = {
            "C": C,  # 增益参量
            "b": b,  # 速度非同步参量
            "d": d,  # 线路损耗参量
            "wp_w": wp_omega,  # 相对等离子体频率
            "Rn": R,  # 行波管等离子体频率降低系数R
            "m": 100,  # 电子离散数量
            "A0": A_0,  # 初始振幅
            "y_end": L,
        }
        print(params)

        # 调用求解器
        result = solveTWTNOLINE_INIT(
            **params
        )  #     (function) def solveTWTNOLINE_INIT(C: Any,b: Any,d: Any,wp_w: Any,beta_e: Any,r_beam: Any,m: Any,A0: Any,y_end: Any,N_steps: int = 1000# ) -> dict[str, Any]

        # ========================= 结果后处理 =========================
        P_Out = C * inputP[0] * inputP[1] * 2 * (result["A"][-1]) ** 2
        Eff = 2 * C * max(result["A"]) ** 2 * 100
        P_max = C * inputP[0] * inputP[1] * 2 * (max(result["A"])) ** 2
        return {"Pmax": P_max, "P_out": P_Out, "Maxiumum_eff": Eff}

    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")
    
if __name__ == "__main__":
    # 计算物理参数
    inputP = [
        0.3,# I: Any
        22800,# V: Any
        3.509778644,# Kc: Any
        0,# Loss: Any
        0.5,# p_SWS: Any
        50,# N_unit: Any
        0.2,# w: Any
        0.2,# t: Any
        1,# Fill_Rate: Any
        211.77257637942898,# f0_GHz: Any
        0.288209415578382,# Vpc: Any
    ] 
    TWTresult=Noline_CORE_CALC(*inputP)
    print(TWTresult)