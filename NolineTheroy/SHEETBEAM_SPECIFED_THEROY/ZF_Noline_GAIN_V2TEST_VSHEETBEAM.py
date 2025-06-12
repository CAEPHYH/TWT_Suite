import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from TWT_CORE_SIMP import simple_calculation
from TWT_CORE_SIMP import detailed_calculation
import math as m
import matplotlib.pyplot as plt
import numpy as np
from _TWT_CORE_NOLINE_COMPLEX_VSHEETBEAM import solveTWTNOLINE_INIT


if __name__ == "__main__":
    # 计算物理参数
    inputP = [
        0.3,  # I: Any
        23000,  # V: Any
        3.6,  # Kc: Any
        0,  # Loss: Any
        0.5,  # p_SWS: Any
        30,  # N_unit: Any
        0.45,  # w: Any
        0.15,  # t: Any
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
    # 参数配置区（根据物理系统调整）

    params = {
        "C": C,  # 增益参量
        "b": b,  # 速度非同步参量
        "d": d,  # 线路损耗参量
        "beam_params": beam_params,  # 带状束流参数
        "m": 50,  # 电子离散数量
        "A0": A_0,  # 初始振幅
        "y_end": L,
    }
    print(params)

    # 调用求解器
    result = solveTWTNOLINE_INIT(
        **params
    )  #     (function) def solveTWTNOLINEM(C: Any,b: Any,d: Any,wp_w: Any,beta_e: Any,r_beam: Any,m: Any,A0: Any,y_end: Any,N_steps: int = 1000# ) -> dict[str, Any]

    # ========================= 结果后处理 =========================
    P_Out = C * inputP[0] * inputP[1] * 2 * (result["A"]) ** 2
    Eff = 2 * C * max(result["A"]) ** 2 * 100
    P_max = C * inputP[0] * inputP[1] * 2 * (max(result["A"])) ** 2

    resultLineG = detailed_calculation(*inputP)
    print(
        "The Gmax in Noline Theroy is %.4f dB "
        % (20 * np.log10((result["A_Ends"] / params["A0"])))
    )
    print("The Gain in Line Theroy is %.3f dB" % (resultLineG["Gmax"]))
    print(
        "The P_out in Noline Theroy is %.4f,The maximum Efficence is %.4f in percent,The maximum POWER is %.4f in Watt"
        % (P_Out[-1], Eff, P_max)
    )
    print(
        f"The A in the end is {result['A'][-1]}, The u in the end is {np.mean(abs(result['u_final']))}"
    )

    # ========================= 结果可视化 =========================
    plt.figure(figsize=(12, 8), dpi=100)

    # 振幅演化
    plt.subplot(2, 3, 1)
    plt.plot(result["y"], result["A"], color="navy")
    plt.xlabel("Position y", fontsize=10)
    plt.ylabel("Amplitude A(y)", fontsize=10)
    plt.title("Amplitude Growth", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 相位演化
    plt.subplot(2, 3, 2)
    plt.plot(result["y"], result["theta"], color="maroon")
    plt.xlabel("Position y", fontsize=10)
    plt.ylabel("Phase Shift θ(y)", fontsize=10)
    plt.title("Phase Evolution", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 最终速度分布
    plt.subplot(2, 3, 3)
    plt.scatter(
        result["phi_final"],
        result["u_final"],
        c=result["phi_final"],
        cmap="hsv",
        s=20,
        edgecolor="k",
        lw=0.5,
    )
    plt.colorbar(label="Final Phase ϕ(y_end)")
    plt.xlabel("Final Phase ϕ(y_end)", fontsize=10)
    plt.ylabel("Final Velocity u(y_end)", fontsize=10)
    plt.title("Velocity Distribution", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 最终相位分布
    plt.subplot(2, 3, 4)
    final_phi = result["phi_final"]
    plt.scatter(
        result["phi0_grid"],
        final_phi,
        c=result["phi0_grid"],
        cmap="hsv",
        s=20,
        edgecolor="k",
        lw=0.5,
    )
    plt.colorbar(label="Initial Phase")
    plt.xlabel("Initial Phase ϕ₀", fontsize=10)
    plt.ylabel("Final Phase ϕ(y_end)", fontsize=10)
    plt.title("Phase Distribution", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 电子相空间图
    Lenth = result["y"] / (2 * np.pi * C)
    plt.subplot(2, 3, 5)
    plt.plot(Lenth, result["u_now"], color="navy")
    plt.xlabel("Position Z(Interaction Lenth)", fontsize=10)
    plt.ylabel("Velocity of eletron (Z)", fontsize=10)
    plt.title("Velocity Distribution (Z))", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 轴向功率图
    plt.subplot(2, 3, 6)
    plt.plot(Lenth, P_Out, color="navy")
    plt.xlabel("Position Z(Interaction Lenth)", fontsize=10)
    plt.ylabel("Output Power Pout(Z)", fontsize=10)
    plt.title("Power Growth", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
