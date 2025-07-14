
import numpy as np
from TWT_CORE_SIMP import simple_calculation
from _TWT_CORE_NOLINE_COMPLEX_VCBEAM_MIX import (
    solveTWTNOLINE_OUTPUT,
    solveTWTNOLINE_INIT,
    solveTWTNOLINE_Drift
)

def calculate_SEGMENT_TWT_NOLINE(
    I,
    V,
    Kc,
    Loss_perunit,
    SectionedSEGMENT_IDX,
    p_SWS,
    N_unit,
    w,
    t,
    Fn_K,
    f0_GHz,
    Vpc,
    P_in,
    Loss_attu,
    harmonic_start_idx=None,  # 添加倍频段起始索引参数

):
    """主函数：计算TWT非线性模型的分段输出功率"""
    
    # ===================== 内部辅助函数定义 =====================
    
    def expand_scalar_params(params_dict, segment_count):
        """将单值参数扩展为与段数相同的列表"""
        expanded = {}
        for name, value in params_dict.items():
            if isinstance(value, (int, float)):
                expanded[name] = [value] * segment_count
            else:
                expanded[name] = value
        return expanded
    
    def handle_harmonic_parameters(param_list, harmonic_index, segment_count):
        """
        处理倍频段参数: 
        - 如果参数是列表且长度为2 (表示[前段值, 倍频段值])
        - 并且指定了倍频起始索引
        - 则扩展为完整列表
        """
        if isinstance(param_list, list) and len(param_list) == 2 and harmonic_index is not None:
            # 确保倍频索引在合理范围内
            if harmonic_index < 0 or harmonic_index >= segment_count:
                raise ValueError(f"倍频起始索引 {harmonic_index} 必须在 0 到 {segment_count-1} 之间")
                
            return ([param_list[0]] * harmonic_index + 
                    [param_list[1]] * (segment_count - harmonic_index))
        return param_list
    
    def build_segment_input(common_params, seg):
        """构建单个段的计算输入参数"""
        return [
            common_params["I"],   # 0: 电流
            common_params["V"],   # 1: 电压
            seg["Kc"],            # 2: 耦合系数
            seg["Loss_perunit"],  # 3: 单位损耗
            seg["p_SWS"],         # 4: 慢波结构周期
            seg["len"],           # 5: 段长度
            common_params["w"],   # 6: 宽度
            common_params["t"],   # 7: 厚度
            seg["Fn_K"],          # 8: 填充因子
            common_params["f0_GHz"],  # 9: 中心频率(GHz)
            seg["Vpc"]            # 10: 相位速度
        ]
    
    def initialize_first_segment(params, common, calc_result, results, input_power):
        """初始化并计算第一段"""
        flux = params["C"] * common["I"] * common["V"] * 2
        params.update({
            "A0": np.sqrt(input_power / flux),
            "y_end": 2 * np.pi * calc_result["互作用长度N"] * params["C"],
        })
        print(params)
        results.append(solveTWTNOLINE_INIT(**params))
    
    def process_attenuator_segment(params, results, seg_idx, attenuation):
        """处理有衰减的段"""
        prev = results[seg_idx - 1]
        params.update({
            "result_y_ends": prev["y"][-1],
            "result_A_ends": prev["A_Ends"] * 10 ** (-attenuation / 20),
            "result_dA_dy": prev["dA_dy_Ends"] * 0,
            "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"],
            "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_Drift(**params))
    
    def process_normal_segment(params, results, seg_idx):
        """处理常规段（无衰减）"""
        prev = results[seg_idx - 1]
        params.update({
            "result_y_ends": prev["y"][-1],
            "result_A_ends": prev["A_Ends"],
            "result_dA_dy": prev["dA_dy_Ends"],
            "result_theta": prev["theta_Ends"],
            "result_dtheta_dy": prev["dtheta_dy_Ends"],
            "result_u_finnal": prev["u_final"],
            "result_phi_finnal": prev["phi_final"],
        })
        results.append(solveTWTNOLINE_OUTPUT(**params))
    
    def calculate_final_power(results, c_list, common_params, segment_count):
        """计算并返回最终功率输出"""
        return 2 * common_params["I"] * common_params["V"] * np.concatenate(
            [c_list[i] * (results[i]["A"] ** 2) for i in range(segment_count)]
        )
    
    # ===================== 主计算逻辑 =====================
    
    # 1. 参数准备与校验
    num_segments = len(N_unit)
    
    # 处理倍频段参数（如果指定了倍频起始索引）
    # 对于每个参数：如果是长度为2的列表 [前段值, 倍频段值]，则扩展为完整列表
    if harmonic_start_idx is not None:
        Kc = handle_harmonic_parameters(Kc, harmonic_start_idx, num_segments)
        Loss_perunit = handle_harmonic_parameters(Loss_perunit, harmonic_start_idx, num_segments)
        p_SWS = handle_harmonic_parameters(p_SWS, harmonic_start_idx, num_segments)
        Fn_K = handle_harmonic_parameters(Fn_K, harmonic_start_idx, num_segments)
        Vpc = handle_harmonic_parameters(Vpc, harmonic_start_idx, num_segments)
    
    # 扩展单值参数为列表
    expansion_params = expand_scalar_params({
        "p_SWS": p_SWS,
        "Kc": Kc,
        "Vpc": Vpc,
        "Loss_perunit": Loss_perunit,
        "Fn_K": Fn_K
    }, num_segments)
    
    # 2. 创建分段结构定义
    segments = []
    for i in range(num_segments):
        segments.append({
            "len": N_unit[i],
            "Vpc": expansion_params["Vpc"][i],
            "Kc": expansion_params["Kc"][i],
            "p_SWS": expansion_params["p_SWS"][i],
            "Loss_perunit": expansion_params["Loss_perunit"][i],
            "Fn_K": expansion_params["Fn_K"][i]
        })
    
    # 3. 公共参数配置
    common_params = {
        "I": I,
        "V": V,
        "w": w,
        "t": t,
        "f0_GHz": f0_GHz,
    }
    
    # 4. 分段计算主循环
    results = []  # 存储每段计算结果
    c_values = [] # 存储每段C值
    
    for seg_idx, segment in enumerate(segments):
        # 4.1 准备当前段输入参数
        input_params = build_segment_input(common_params, segment)
        print(input_params)
        calc_result = simple_calculation(*input_params)
        
        # 4.2 计算当前段公共参数
        C = calc_result["小信号增益因子C"]
        interaction_length = 2 * np.pi * calc_result["互作用长度N"] * C
        c_values.append(C)
        
        # 4.3 准备非线性求解器参数
        solver_params = {
            "C": C,
            "b": calc_result["非同步参量b"],
            "d": calc_result["损耗因子d"],
            "wp_w": calc_result["等离子体频率Wp"] / (2 * np.pi * input_params[9] * 1e9),
            "beta_space": calc_result["beta_Space"],
            "r_beam": calc_result["束流归一化尺寸r_beam"],
            "Fill_Rate": segment["Fn_K"],
            "p_SWS": segment["p_SWS"] * 1e-3,  # 转换为米
            "Space_cut": 10,  # 空间电荷场截断项数
            "m": 50,          # 求解器精度参数
            "y_end": interaction_length + (results[-1]["y"][-1] if seg_idx > 0 else 0),
        }
        
        # 4.4 根据段类型处理

        if seg_idx == 0:  # 第一段
            initialize_first_segment(solver_params, common_params, calc_result, results, P_in)
        elif seg_idx in SectionedSEGMENT_IDX:  # 衰减段
            process_attenuator_segment(solver_params, results, seg_idx, Loss_attu)
        else:  # 常规段
            process_normal_segment(solver_params, results, seg_idx)
    
    # 5. 计算最终输出功率
    final_power = calculate_final_power(results, c_values, common_params, num_segments)
    
    return {"输出功率P_out": final_power[-1]}


if __name__ == "__main__":
    # 测试参数（简化输入形式）
    input_params_simple = [
        0.065,   # 电流 (I)
        23250,  # 电压 (V)
        [0.75, 1.31],  # 耦合系数 (Kc)，倍频前和倍频段,对应两组色散*
        [0.1, 0.13],   # 单位损耗，倍频前和倍频段,对应两组色散*
        [1,3],     # 衰减段索引
        [0.455, 0.15],  # 慢波结构周期，倍频前和倍频段!
        [60, 10, 10, 10, 150],  # 段长度 (N_unit)
        0.08,   # 宽度 (w, mm)
        0.08,   # 厚度 (t, mm)
        [1.85, 1.25],  # 填充因子，倍频前和倍频段!
        672,    # 中心频率 (f0_GHz)
        [0.293, 0.292],  # 相位速度，倍频前和倍频段,对应两组色散*
        0.004,
        0
    ]

    # 倍频段起始索引（第3段开始倍频）
    harmonic_start_idx = 2
    
    # 执行计算（传入倍频起始索引）
    result = calculate_SEGMENT_TWT_NOLINE(
        *input_params_simple,
        harmonic_start_idx=harmonic_start_idx
    )

#     def calculate_SEGMENT_TWT_NOLINE(
#     I: Any,
#     V: Any,
#     Kc: Any,
#     Loss_perunit: Any,
#     SectionedSEGMENT_IDX: Any,
#     p_SWS: Any,
#     N_unit: Any,
#     w: Any,
#     t: Any,
#     Fn_K: Any,
#     f0_GHz: Any,
#     Vpc: Any,
#     P_in: Any,
#     Loss_attu: Any,
#     harmonic_start_idx: Any | None = None
# ) -> dict[str, Any]
    
    print("计算结果:", result)