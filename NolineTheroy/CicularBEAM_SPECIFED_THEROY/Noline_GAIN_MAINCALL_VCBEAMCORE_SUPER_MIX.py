import numpy as np
import sys

from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_VSUPER_MIX import solveTWTNOLINE_OUTPUT, solveTWTNOLINE_INIT

def calculate_SEGMENT_TWT_NOLINE(I, V, Kc, Loss_perunit, p_SWS, N_unit, w, t, Fn_K, f0_GHz, Vpc,P_in = 0.1,Loss_attu=20):
    # 根据N_unit构建SEGMENTS
    SEGMENTS = [{
        "len": n,
        "Vpc": Vpc,
        "Kc": Kc,
        "Loss_perunit": Loss_perunit,
        "p_SWS": p_SWS,
    } for n in N_unit]

    # 公共参数配置
    COMMON_PARAMS = {
        "I": I,
        "V": V,
        "w": w,
        "t": t,
        "Fn_K": Fn_K,
        "f0_GHz": f0_GHz,
    }

    results = []
    C_list = []  # 保存各段的C值

    for seg_idx, seg in enumerate(SEGMENTS):
        inputP = [
            COMMON_PARAMS["I"],
            COMMON_PARAMS["V"],
            seg["Kc"],
            seg["Loss_perunit"],
            seg["p_SWS"],
            seg["len"],
            COMMON_PARAMS["w"],
            COMMON_PARAMS["t"],
            COMMON_PARAMS["Fn_K"],
            COMMON_PARAMS["f0_GHz"],
            seg["Vpc"]
        ]
        print(inputP)
        calc_result = simple_calculation(*inputP)
        C = calc_result["小信号增益因子C"]
        b = calc_result["非同步参量b"]
        L = 2 * np.pi * calc_result["互作用长度N"] * C
        d = calc_result["损耗因子d"]
        beta_space = calc_result["beta_Space"]
        r_beam=calc_result["束流归一化尺寸r_beam"]
        wp_omega = calc_result["等离子体频率Wp"] / (2 * np.pi * inputP[9] * 1e9)
        Space_cut=10#空间电荷场截断项数

        C_list.append(C)  # 保存当前段的C值

        if seg_idx == 0:
            # 初始段处理
            
            P_flux = C * I * V * 2
            A0 = np.sqrt(P_in / P_flux)
            
            params = {
                "C": C, "b": b, "d": d, "wp_w": wp_omega,
                "beta_space": beta_space,"r_beam":r_beam ,"Fill_Rate":COMMON_PARAMS["Fn_K"],"p_SWS":seg["p_SWS"]*1e-3,
                "m": 50, "A0": A0, "y_end": L,"Space_cut":Space_cut
            }
            print(params)
            result = solveTWTNOLINE_INIT(**params)
        else:
            # 后续段处理
            prev_result = results[seg_idx - 1]
            params = {
                "C": C, "b": b, "d": d, "wp_w": wp_omega,
                "beta_space": beta_space,"r_beam":r_beam ,"Fill_Rate":COMMON_PARAMS["Fn_K"],"p_SWS":seg["p_SWS"]*1e-3,
                "m": 50,
                "result_y_ends": prev_result["y"][-1],
                "result_A_ends": prev_result["A_Ends"] * 10**(-Loss_attu/20),
                "result_dA_dy": prev_result["dA_dy_Ends"],
                "result_theta": prev_result["theta_Ends"],
                "result_dtheta_dy": prev_result["dtheta_dy_Ends"],
                "result_u_finnal": prev_result["u_final"],
                "result_phi_finnal": prev_result["phi_final"],
                "y_end": prev_result["y"][-1] + L,
                "Space_cut":Space_cut
            }
            result = solveTWTNOLINE_OUTPUT(**params)
        
        results.append(result)

    # 计算最终输出功率
    P_Out = 2 * I * V * np.concatenate([
        C_list[i] * (results[i]["A"]**2) for i in range(len(SEGMENTS))
    ])
    P_End=P_Out[-1]

    return {
        "输出功率P_out":P_End
    }

if __name__ == "__main__":
    # 计算物理参数
    inputP = [
        0.3,# I: Any
        23000,# V: Any
        3.8812150306728705,# Kc: Any
        0,# Loss_perunit: Any
        0.5,# p_SWS: Any
        [25,25],# N_unit: Any
        0.2,# w: Any
        0.2,# t: Any
        1,# Fill_Rate: Any
        211,# f0_GHz: Any
        0.2893778968136611,# Vpc: Any
    ]  ##(function) def detailed_calculation(I: Any,V: Any,Kc: Any,Loss_perunit: Any,p_SWS: Any,N_unit: Any,w: Any,t: Any,Fill_Rate: Any,f0_GHz: Any,Vpc: Any# )
    TWTresult = calculate_SEGMENT_TWT_NOLINE(*inputP)
    print(TWTresult)