import numpy as np
from _TWT_CORE_NOLINE_CALC_EspDEV import compute_Esp_V2, compute_esc_V2Z, compute_Esp_VR
from TWT_CORE_SIMP import simple_calculation
import time

def test_compute_Esp():
    """针对compute_Esp函数的测试用例"""
    inputP = [
        0.3,  # I: Any
        23000,  # V: Any
        3.6,  # Kc: Any
        0,  # Loss_perunit: Any
        0.5,  # p_SWS: Any
        28,  # N_unit: Any
        0.40,  # w: Any
        0.20,  # t: Any
        1,  # Fill_Rate: Any
        211,  # f0_GHz: Any
        0.288,  # Vpc: Any
    ]  ##(function) def detailed_calculation(I: Any,V: Any,Kc: Any,Loss_perunit: Any,p_SWS: Any,N_unit: Any,w: Any,t: Any,Fn_K: Any,f0_GHz: Any,Vpc: Any# )
    initparamater = simple_calculation(*inputP)
    print(initparamater)

    C = initparamater["小信号增益因子C"]
    b = initparamater["非同步参量b"]
    d = initparamater["损耗因子d"]
    x_b = inputP[6]
    y_b = inputP[7]
    x_d = x_b / 1
    y_d = y_b / 1
    I_beam = inputP[0]
    V_beam = inputP[1]
    freq = inputP[9]
    beam_params = [x_d, y_d, x_b, y_b, I_beam, V_beam, freq]
    Rn_sqr_values = initparamater["等离子体频率降低因子Fn"]
    wp_w = initparamater["等离子体频率Wp"] / (2 * np.pi * inputP[9] * 1e9)
    Space_CUT = 10
    beam_paramsR = [Rn_sqr_values, wp_w, Space_CUT]

    L = 2 * np.pi * initparamater["互作用长度N"] * C
    P_in = 0.1
    P_flux = C * inputP[0] * inputP[1] * 2
    A_0 = np.sqrt(P_in / P_flux)

    # ================================
    # 参数配置区

    m = 32
    delta_phi0 = 2 * np.pi / m

    # 生成相位差矩阵 (示例：线性递增相位差)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    phi_diff_matrix = phi0_grid[:, np.newaxis] - phi0_grid  # (32, 32)

    # 生成denominator (1 + 2*C*u)，假设u为小量
    u = np.zeros(m)  # 随机生成u
    denominator = 1 + 2 * C * u  # (32,)

    # === 输入处理（添加方法有效性校验） ===
    while True:  # 确保输入有效
        try:
            method_input = input("请输入要对比的方法版本（用空格分隔，例如'1 2 3'）: ")
            method_Version = list(map(int, method_input.split()))
            if not all(mv in {1, 2, 3} for mv in method_Version):
                print("输入数字必须为1、2或3，请重新输入。")
                continue
            print("用户输入的列表为:", method_Version)
            break
        except ValueError:
            print("输入包含非数字字符，请重新输入。")

    # === 性能对比和结果存储 ===
    Esp_results = {}
    for mV in method_Version:
        start_time = time.time()  # 记录开始时间

        # 调用待测函数
        if mV == 1:
            Esp_vector = compute_Esp_VR(
                phi_diff_matrix, denominator, delta_phi0, C, b, m, beam_paramsR
            )
        elif mV == 2:
            Esp_vector = compute_esc_V2Z(
                phi_diff_matrix, denominator, C, m, beam_params
            )
        elif mV == 3:
            Esp_vector = compute_Esp_V2(
                phi_diff_matrix, denominator, C, m, beam_params
            )

        exec_time = time.time() - start_time  # 计算耗时

        # 存储结果和时间
        Esp_results[mV] = {"result": Esp_vector, "time": exec_time}
        print(f"方法{mV}耗时: {exec_time:.6f}秒")
        print(f"方法{mV}结果示例（前5元素）:", Esp_vector)

    # === 结果差异对比 ===
    methods = list(Esp_results.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            meth1 = methods[i]
            meth2 = methods[j]
            
            res1 = Esp_results[meth1]["result"]
            res2 = Esp_results[meth2]["result"]
            
            # 检查形状是否一致
            if res1.shape != res2.shape:
                print(f"方法{meth1}和方法{meth2}结果形状不一致，无法对比!")
                continue
                
            # 计算数值差异
            abs_diff = np.abs(res1 - res2)
            max_abs_diff = np.max(abs_diff)
            avg_abs_diff = np.mean(abs_diff)
            max_rel_diff = np.max(np.abs((res1 - res2) / (res2 + 1e-10)))
            
            print(f"\n==== 方法{meth1} vs 方法{meth2} 差异 ====")
            print(f"最大绝对差: {max_abs_diff}")
            print(f"平均绝对差: {avg_abs_diff}")
            print(f"最大相对差: {max_rel_diff * 100:.2f}%")

    return Esp_results

# 运行测试
if __name__ == "__main__":
    Esp_results = test_compute_Esp()
    print("测试完成! 结果已保存")