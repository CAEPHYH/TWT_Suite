import numpy as np
from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_V2_MIX import (
    solveTWTNOLINE_OUTPUT,
    solveTWTNOLINE_INIT,
    solveTWTNOLINE_Drift,
)


def calculate_SEGMENT_TWT_NOLINE_for_PVTOPT(
    I, V, Loss_perunit, p_SWS, N_unit, w, t, Fn_K, f0_GHz, para_func, P_in=0.1
):
    """
    计算分段TWT非线性输出功率

    参数:
        I: 束流电流 (A)
        V: 工作电压 (V)
        Loss_perunit: 损耗参数
        p_SWS: 慢波结构参数 (标量或列表)
        N_unit: 各段单元数 (列表)
        w: 结构参数w (mm)
        t: 结构参数t (mm)
        Fn_K: 填充系数
        f0_GHz: 工作频率 (GHz)
        para_func: PVT段参数计算函数
        P_in: 输入功率 (W), 默认为0.1W

    返回:
        包含输出功率的字典
    """
    num_segments = len(N_unit)
    # 使用传入的para_func计算PVT段参数
    if isinstance(p_SWS, (int, float)):
        p_SWS = [p_SWS] * num_segments
    SEGMENTS = [
        {
            "len": n,
            "Vpc": para_func(p_sws)["Vpc"],
            "Kc": para_func(p_sws)["Kc"],
            "p_SWS": p_sws,
        }
        for n, p_sws in zip(N_unit, p_SWS)
    ]

    # 公共参数配置
    COMMON_PARAMS = {
        "I": I,
        "V": V,
        "w": w,
        "t": t,
        "Fn_K": Fn_K,
        "Loss_perunit": Loss_perunit,
        "f0_GHz": f0_GHz,
    }

    results = []
    C_list = []  # 保存各段的C值

    for seg_idx, seg in enumerate(SEGMENTS):
        inputP = [
            COMMON_PARAMS["I"],
            COMMON_PARAMS["V"],
            seg["Kc"],
            COMMON_PARAMS["Loss_perunit"],
            seg["p_SWS"],
            seg["len"],
            COMMON_PARAMS["w"],
            COMMON_PARAMS["t"],
            COMMON_PARAMS["Fn_K"],
            COMMON_PARAMS["f0_GHz"],
            seg["Vpc"],
        ]
        print(inputP)
        calc_result = simple_calculation(*inputP)
        C = calc_result["小信号增益因子C"]
        b = calc_result["非同步参量b"]
        L = 2 * np.pi * calc_result["互作用长度N"] * C
        d = calc_result["损耗因子d"]
        if (
            inputP[6] == inputP[7]
        ):  # 行波管等离子体频率降低系数R,ShitBeam用Fn,CicularBeam用Rowe特征值R
            R = calc_result["Rowe特征值R"]
        else:
            R = calc_result["等离子体频率降低因子Fn"]
        wp_omega = calc_result["等离子体频率Wp"] / (2 * np.pi * inputP[9] * 1e9)

        C_list.append(C)  # 保存当前段的C值

        if seg_idx == 0:
            # 初始段处理
            P_flux = C * I * V * 2
            A0 = np.sqrt(P_in / P_flux)

            params = {
                "C": C,
                "b": b,
                "d": d,
                "wp_w": wp_omega,
                "Rn": R,
                "m": 100,
                "A0": A0,
                "y_end": L,
            }
            result = solveTWTNOLINE_INIT(**params)
        else:
            # 后续段处理
            prev_result = results[seg_idx - 1]
            params = {
                "C": C,
                "b": b,
                "d": d,
                "wp_w": wp_omega,
                "Rn": R,
                "m": 100,
                "result_y_ends": prev_result["y"][-1],
                "result_A_ends": prev_result["A_Ends"] * 1,
                "result_dA_dy": prev_result["dA_dy_Ends"],
                "result_theta": prev_result["theta_Ends"],
                "result_dtheta_dy": prev_result["dtheta_dy_Ends"],
                "result_u_finnal": prev_result["u_final"],
                "result_phi_finnal": prev_result["phi_final"],
                "y_end": prev_result["y"][-1] + L,
            }
            result = solveTWTNOLINE_OUTPUT(**params)

        results.append(result)

    # 计算最终输出功率
    P_Out = (2* I* V* np.concatenate([C_list[i] * (results[i]["A"] ** 2) for i in range(len(SEGMENTS))]))
    P_End = P_Out[-1]

    return {"输出功率P_out": P_End}


def calculate_SEGMENT_TWT_NOLINE_SECTIONED_for_PVTOPT(
    I,
    V,
    Loss_perunit,
    p_SWS,
    N_unit,
    w,
    t,
    Fn_K,
    f0_GHz,
    para_func,
    P_in=0.1,
    IsZeroSectioned=1,
    Loss_attu=20,
    SectionedSEGMENT_IDX=[1,3]
):
    """
    计算分段TWT非线性输出功率

    参数:
        I: 束流电流 (A)
        V: 工作电压 (V)
        Loss_perunit: 损耗参数
        p_SWS: 慢波结构参数 (标量或列表)
        N_unit: 各段单元数 (列表)
        w: 结构参数w (mm)
        t: 结构参数t (mm)
        Fn_K: 填充系数
        f0_GHz: 工作频率 (GHz)
        para_func: PVT段参数计算函数
        P_in: 输入功率 (W), 默认为0.1W
        IsZeroSectioned:是否是零距离切断，否则采用飘逸段算法

    返回:
        包含输出功率的字典
    """
    # ========================= 参数配置 =========================
    num_segments = len(N_unit)
    # 使用传入的para_func计算PVT段参数
    if isinstance(p_SWS, (int, float)):
        p_SWS = [p_SWS] * num_segments
    SEGMENTS = [
        {
            "len": n,
            "Vpc": para_func(p_sws)["Vpc"],
            "Kc": para_func(p_sws)["Kc"],
            "p_SWS": p_sws,
        }
        for n, p_sws in zip(N_unit, p_SWS)
    ]
    # 公共参数配置
    COMMON_PARAMS = {
        "I": I,
        "V": V,
        "w": w,
        "t": t,
        "Fn_K": Fn_K,
        "Loss_perunit": Loss_perunit,
        "f0_GHz": f0_GHz,
    }

    # ========================= 方法配置 =========================
    def build_input_params(common_params, seg):
        """构建输入参数列表"""
        return [
            common_params["I"],
            common_params["V"],
            seg["Kc"],
            common_params["Loss_perunit"],
            seg["p_SWS"],
            seg["len"],
            common_params["w"],
            common_params["t"],
            common_params["Fn_K"],
            common_params["f0_GHz"],
            seg["Vpc"],
        ]

    def get_plasma_factor(calc_result, w, t):
        """获取等离子体频率因子"""
        return (
            calc_result["Rowe特征值R"]
            if w == t
            else calc_result["等离子体频率降低因子Fn"]
        )

    def handle_initial_segment(params, common_params, calc_result, results, P_in):
        """处理初始段"""
        P_in = P_in  # 输入功率Pin
        P_flux = params["C"] * common_params["I"] * common_params["V"] * 2
        params.update(
            {
                "A0": np.sqrt(P_in / P_flux),
                "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"],
            }
        )
        results.append(solveTWTNOLINE_INIT(**params))

    def handle_attenuator_segment(params, results, seg_idx, Loss_attu, IsZeroSectioned):
        """处理衰减段"""
        prev = results[seg_idx - 1]
        if IsZeroSectioned == 0:
            params.update(
                {
                    "result_y_ends": prev["y"][-1],
                    "result_A_ends": prev["A_Ends"] * 10 ** (-Loss_attu / 20),
                    "result_dA_dy": prev["dA_dy_Ends"] * 0,
                    "result_theta": prev["theta_Ends"],
                    "result_dtheta_dy": prev["dtheta_dy_Ends"],
                    "result_u_finnal": prev["u_final"],
                    "result_phi_finnal": prev["phi_final"],
                }
            )
            results.append(solveTWTNOLINE_Drift(**params))
        else:
            params.update(
                {
                    "result_y_ends": prev["y"][-1],
                    "result_A_ends": prev["A_Ends"] * 10 ** (-Loss_attu / 20),
                    "result_dA_dy": prev["dA_dy_Ends"],
                    "result_theta": prev["theta_Ends"],
                    "result_dtheta_dy": prev["dtheta_dy_Ends"],
                    "result_u_finnal": prev["u_final"],
                    "result_phi_finnal": prev["phi_final"],
                }
            )
            results.append(solveTWTNOLINE_OUTPUT(**params))

    def handle_normal_segment(params, results, seg_idx):
        """处理常规段"""
        prev = results[seg_idx - 1]
        params.update(get_previous_results(prev))
        results.append(solveTWTNOLINE_OUTPUT(**params))

    def get_previous_results(prev_result):
        """获取前一段结果"""
        return {
            "result_y_ends": prev_result["y"][-1],
            "result_A_ends": prev_result["A_Ends"],
            "result_dA_dy": prev_result["dA_dy_Ends"],
            "result_theta": prev_result["theta_Ends"],
            "result_dtheta_dy": prev_result["dtheta_dy_Ends"],
            "result_u_finnal": prev_result["u_final"],
            "result_phi_finnal": prev_result["phi_final"],
        }

    def process_result(results, C_list, common_params, segments):
        # 功率计算
        P_Out = (2* common_params["I"]* common_params["V"]* np.concatenate([C_list[i] * (results[i]["A"] ** 2) for i in range(len(segments))]))
        return P_Out

    # ========================= 主计算逻辑 =========================
    results = []
    C_list = []

    for seg_idx, seg in enumerate(SEGMENTS):
        # 参数计算
        input_params = build_input_params(COMMON_PARAMS, seg)
        print(input_params)
        calc_result = simple_calculation(*input_params)

        # 缓存公共参数
        C = calc_result["小信号增益因子C"]
        L = 2 * np.pi * calc_result["互作用长度N"] * C
        C_list.append(C)
        params = {
            "C": C,
            "b": calc_result["非同步参量b"],
            "d": calc_result["损耗因子d"],
            "wp_w": calc_result["等离子体频率Wp"]
            / (2 * np.pi * COMMON_PARAMS["f0_GHz"] * 1e9),
            "Rn": get_plasma_factor(calc_result, input_params[6], input_params[7]),
            "m": 100,
            "y_end": L + (results[-1]["y"][-1] if seg_idx > 0 else 0),
        }

        # 分段处理
        if seg_idx == 0:
            handle_initial_segment(params, COMMON_PARAMS, calc_result, results, P_in)
        elif seg_idx in SectionedSEGMENT_IDX:
            handle_attenuator_segment(
                params, results, seg_idx, Loss_attu, IsZeroSectioned
            )
        else:
            handle_normal_segment(params, results, seg_idx)

    P_End = process_result(results, C_list, COMMON_PARAMS, SEGMENTS)

    return {"输出功率P_out": P_End[-1]}


# 测试代码
if __name__ == "__main__":
    # 定义默认PVT参数计算函数
    def default_para_func(p_SWS):
        Vpc = 0.262 + p_SWS * 0.05
        Kc = 3.6 + p_SWS * 0
        return {"Vpc": Vpc, "Kc": Kc}

    # 计算物理参数
    inputP = [
        0.30,  # I: Any
        23000,  # V: Any
        0.1,  # Loss_perunit: Any
        [0.5, 0.5, 0.5, 0.5, 0.5],  # p_SWS: Any
        [30,5,5,5,5],  # N_unit: Any
        0.20,  # w: Any
        0.20,  # t: Any
        1,  # Fill_Rate: Any
        211,  # f0_GHz: Any
        default_para_func,  # PVT参数计算函数
    ]

    IsSectioned = int(input("Is Sectioned:"))

    if IsSectioned == 0:
        TWTresult = calculate_SEGMENT_TWT_NOLINE_for_PVTOPT(*inputP)
    else:
        TWTresult = calculate_SEGMENT_TWT_NOLINE_SECTIONED_for_PVTOPT(*inputP)
    print(f"输出功率: {TWTresult['输出功率P_out']:.2f} W")
