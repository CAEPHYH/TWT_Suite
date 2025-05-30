import numpy as np
from _TWT_CORE_NOLINE_COMPLEX_VSHEETBEAM import compute_Esp, compute_Esp_V2
from TWT_CORE_SIMP import simple_calculation
import time


def test_compute_Esp():
    """针对compute_Esp函数的测试用例"""
    inputP = [
        0.3,  # I: Any
        23000,  # V: Any
        3.6,  # Kc: Any
        0,  # Loss: Any
        0.5,  # p_SWS: Any
        28,  # N_unit: Any
        0.40,  # w: Any
        0.20,  # t: Any
        0,  # Fill_Rate: Any
        211,  # f0_GHz: Any
        0.288,  # Vpc: Any
    ]  ##(function) def detailed_calculation(I: Any,V: Any,Kc: Any,Loss: Any,p_SWS: Any,N_unit: Any,w: Any,t: Any,Fn_K: Any,f0_GHz: Any,Vpc: Any# )
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
    print(initparamater)

    C = initparamater["小信号增益因子C"]
    # 行波管增益参数C
    b = initparamater["非同步参量b"]
    # 行波管非同步参量b
    d = initparamater["损耗因子d"]
    # 损耗系数d
    x_b = inputP[6]
    y_b = inputP[7]
    x_d = x_b / 1
    y_d = y_b / 1
    I_beam = inputP[0]
    V_beam = inputP[1]
    freq = inputP[9]
    beam_params = [x_d, y_d, x_b, y_b, I_beam, V_beam, freq]
    # 束流参数beam_params
    L = 2 * np.pi * initparamater["互作用长度N"] * C
    # 计算终点位置

    P_in = 0.1
    P_flux = C * inputP[0] * inputP[1] * 2
    A_0 = np.sqrt(P_in / P_flux)

    # ================================
    # 参数配置区

    params = {
        "C": C,  # 增益参量
        "b": b,  # 速度非同步参量
        "d": d,  # 线路损耗参量
        "beam_params": beam_params,  # 带状束流参数
        "A0": A_0,  # 初始振幅
        "y_end": L,
    }
    print(params)
    m = 100

    # 生成相位差矩阵 (示例：线性递增相位差)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    phi_diff_matrix = phi0_grid[:, np.newaxis] - phi0_grid  # (32, 32)

    # 生成denominator (1 + 2*C*u)，假设u为小量
    u = np.zeros(m)  # 随机生成u
    denominator = 1 + 2 * C * u  # (32,)

    # === 输入处理（添加方法有效性校验） ===
    while True:  # 确保输入有效
        try:
            method_input = input("请输入要对比的方法版本（用空格分隔，例如'1 2'）: ")
            method_Version = list(map(int, method_input.split()))
            if not all(mv in {1, 2} for mv in method_Version):
                print("输入数字必须为1或2，请重新输入。")
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
            Esp_vector = compute_Esp(phi_diff_matrix, denominator, C, m, beam_params)
        else:
            Esp_vector = compute_Esp_V2(phi_diff_matrix, denominator, C, m, beam_params)

        exec_time = time.time() - start_time  # 计算耗时

        # 存储结果和时间
        Esp_results[mV] = {"result": Esp_vector, "time": exec_time}
        print(f"方法{mV}耗时: {exec_time:.6f}秒")
        print(f"方法{mV}结果示例（前5元素）:", Esp_vector)

    if 1 in Esp_results and 2 in Esp_results:
        esp1 = Esp_results[1]["result"]
        esp2 = Esp_results[2]["result"]

        # 检查形状是否一致
        if esp1.shape != esp2.shape:
            print("警告：两种方法结果形状不一致，无法直接对比！")
        else:
            # 计算数值差异
            abs_diff = np.abs(esp1 - esp2)
            max_abs_diff = np.max(abs_diff)
            avg_abs_diff = np.mean(abs_diff)

            rel_diff = abs_diff / (np.abs(esp2) + 1e-10)  # 避免除以0
            max_rel_diff = np.max(rel_diff)
            avg_rel_diff = np.mean(rel_diff)

            print("\n==== 方法结果差异对比 ====")
            print(f"最大绝对差: {max_abs_diff}")
            print(f"平均绝对差: {avg_abs_diff}")
            print(f"最大相对差: {max_rel_diff * 100:.2f}%")
            print(f"平均相对差: {avg_rel_diff * 100:.2f}%")

    return Esp_results


# 运行测试
if __name__ == "__main__":
    Esp_results = test_compute_Esp()
    print("示例输出（前5个元素）:", Esp_results)
