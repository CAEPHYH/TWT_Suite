import tkinter as tk
from tkinter import messagebox
import math as m
import numpy as np
import Vpc_calc
from scipy.special import jv  # 贝塞尔函数和双曲余切
from scipy.optimize import root_scalar  # 数值求根
import matplotlib.pyplot as plt
from scipy.special import iv, kv, jv  # 仅导入贝塞尔函数
from scipy.optimize import root

def calculate():
    try:
        # 获取输入参数（新增QC手动输入）
        I = float(entry_i.get())
        V = float(entry_v.get())
        Kc = float(entry_kc.get())
        Loss = float(entry_loss.get())
        p_SWS = float(entry_p_SWS.get()) * 1e-3  # 将用户输入周期长度转换为m
        N_unit = int(entry_n_unit.get())
        w = float(entry_w.get()) * 1e-3  # 将用户输入束流尺寸转换为m
        t = float(entry_t.get()) * 1e-3  # 将用户输入束流尺寸转换为m
        f0_GHz = float(entry_f0.get())
        Fn_K = float(entry_Fn_K.get())  # 获取QC手动输入
        Vpc = float(entry_Vpc.get())

        # ========== 输入验证 ==========
        if N_unit <= 0:
            raise ValueError("N_unit必须为正整数")
        if any(v <= 0 for v in [w, t, f0_GHz]):
            raise ValueError("w、t、f0必须为正数")

        # ========== 核心计算逻辑（与图片严格一致） ==========

        # 计算C = (I*Kc/(4V))^(1/3) （图片公式4）
        C = m.pow(I * Kc / (4 * V), 1 / 3)

        # 计算omega和Wq （图片等离子体频率公式）
        f0_Hz = f0_GHz * 1e9
        omega = 2 * m.pi * f0_Hz

        yita = 1.7588e11
        erb = 8.854e-12
        S = (w * t * m.pi) / 4

        numerator = I * np.sqrt(yita)
        denominator = S * erb * np.sqrt(2 * V)
        Wp = np.sqrt(numerator / denominator)

        Vec = Vpc_calc.Vpc_calc(V)
        # 归一化电子速度Vec

        # 计算电子波数Beta_e,gamma0与束流归一化尺寸
        beta_e = omega / (Vec * 299792458)  # 电子波数Beta_e
        K_wave = omega / 299792458
        gamma0 = np.sqrt(beta_e**2 - K_wave**2)

        r_beam = np.sqrt((w * t / 4))  # 束流归一化尺寸

        # 计算非同步参量b
        b = 1 / C * (Vec - Vpc) / Vpc

        # 计算等离子体频率降Fn
        ##圆形束流等离子体频率降特征方程-特征值数值求解
        if w == t:
            if Fn_K == 1:
                Fn_tmp = 2.405 / r_beam  # 圆形束流等离子体频率降特征方程-特征值
            else:
                Fn_tmp = calculate_R(gamma_0=gamma0, a=r_beam * Fn_K, b=r_beam)
            Fn = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))
        else:
            Fn_tmp = np.sqrt((np.pi / w) ** 2 + (np.pi / t) ** 2)
            Fn = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))
            Fn = Fn * Fn_K

        # 计算N=beta_e*p_SWS/(2*m.pi) * N_unit （图片公式3）
        N = beta_e * p_SWS / (2 * m.pi) * N_unit

        # 计算L = Loss/N （图片公式2）
        L = Loss / N
        # 计算d = 0.01836*L/C （图片公式2）
        d = 0.01836 * L / C

        # 空间电荷因子4QC
        Wq_over_omegaC_sq = (Fn * Wp / (omega * C)) ** 2

        # ========== 行波管反波三次方程求解（图片公式） ==========
        # 原方程 δ² = 1/(b+jd-jδ) - 4QC
        # 重排为三次方程: -jδ³ + (b+jd)δ² - (4QC)jδ + (−1+4QCb+4QCjd) = 0
        j = 1j
        coeffs = [
            -j,
            j * d + b,
            -Wq_over_omegaC_sq * j,
            Wq_over_omegaC_sq * (j * d + b) - 1,
        ]
        roots = np.roots(coeffs)
        sorted_roots = sorted(roots, key=lambda x: x.real, reverse=True)
        x1, y1 = sorted_roots[0].real, sorted_roots[0].imag
        x2, y2 = sorted_roots[1].real, sorted_roots[1].imag
        x3, y3 = sorted_roots[2].real, sorted_roots[2].imag

        delta1 = sorted_roots[0]
        delta2 = sorted_roots[1]
        delta3 = sorted_roots[2]

        # ========== 核心增益计算与振荡判断 ==========
        # 反波起征绝对不稳定性条件
        CN_X = np.linspace(0, 1, 51)
        termA = (
            (Wq_over_omegaC_sq + delta1**2)
            / ((delta1 - delta2) * (delta1 - delta3))
            * np.exp(2 * np.pi * delta1 * CN_X)
        )
        termB = (
            (Wq_over_omegaC_sq + delta2**2)
            / ((delta2 - delta3) * (delta2 - delta1))
            * np.exp(2 * np.pi * delta2 * CN_X)
        )
        termC = (
            (Wq_over_omegaC_sq + delta3**2)
            / ((delta3 - delta1) * (delta3 - delta2))
            * np.exp(2 * np.pi * delta3 * CN_X)
        )
        F_backwave = abs(termA + termB + termC)
        plt.plot(CN_X, F_backwave, color="maroon")
        plt.xlabel("CN_X")
        plt.ylabel("F_backwave")
        plt.title("Backwave")
        plt.grid(True)
        plt.show(block=False)  # 非阻塞模式显示图像

        # 反波震荡估计
        NUnit_Backwave = (1 / 2 * (Wq_over_omegaC_sq / 4) ** 0.25) / C
        # ========== 严格按图片格式输出 ==========
        result_text = (
            f"以下是计算结果："
            f"增益因子 C = {C:.3f}\nN = {N:.3f}\n"
            f"L = Loss/N={Loss:.1f}/{N:.3f} = {L:.3f}\n"
            f"损耗因子 d = 0.01836*L/C = 0.01836*{L:.3f}/{C:.3f} = {d:.2f}\n"
            f"圆形束流等离子体频率降低因子计算参量 Gamma*r_beam ={gamma0*r_beam:.3e} \n"
            f"超宽带状束流等离子体频率降低因子计算参量 Gamma*t ={gamma0*t:.3e} \n"
            f"圆形束满填充等离子体频率降低因子 Fn ={Fn:.3e} \n"
            f"等离子体频率 Wp = {Wp:.3e} rad/s\n"
            f"空间电荷参量 4QC = {Wq_over_omegaC_sq:.3e}\n"  # 显示实际使用的QC值
            f"非同步参量 b={b:.3f}\n"
            f"归一化电子速度 Vec={Vec:.3f}\n"
            f"x₁ = {x1:.3f}\n"
            f"以下是反波震荡估计：\n"
            f"起反波震荡长度NUnit_Backwave={NUnit_Backwave:.3f}"
        )
        lbl_result.config(text=result_text)

    except ValueError as e:
        messagebox.showerror("输入错误", str(e))
    except ZeroDivisionError:
        messagebox.showerror("计算错误", "除零错误（请检查输入参数）")


# ==============================================
# ================== GUI界面（新增QC输入框） ==================
root = tk.Tk()
root.title("行波管线性增益计算器-SUPERPLUS版")
root.geometry("500x650")

# 输入参数框架（新增QC输入行）
input_frame = tk.LabelFrame(root, text="输入参数（*为频域变量）", padx=10, pady=10)
input_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

input_labels = [
    ("束流电流 I (A):", "0.30"),
    ("束流电压 U (V):", "23000"),
    ("Kc*:", "3.6"),
    ("Loss*:", "0"),
    ("周期长度p_SWS (mm):", "0.500"),
    ("周期个数N_unit (正整数):", "25"),
    ("宽度 w (mm):", "0.2"),
    ("厚度 t (mm):", "0.2"),
    ("频率 f0 (GHz)*:", "211"),
    (
        "Fn_K 参数:\n w=t表示圆形束流\n 填充率倒数必须大于1\n w!=t,仅为带状束流矫正\n",
        "1",
    ),
    ("Vpc (c)*:", "0.288"),  # 新增Vpc输入行
]

for row, (label_text, default_val) in enumerate(input_labels):
    tk.Label(input_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
    entry = tk.Entry(input_frame)
    entry.insert(0, default_val)
    entry.grid(row=row, column=1, padx=5)

# 获取输入框引用（新增QC）
entries = [
    child for child in input_frame.children.values() if isinstance(child, tk.Entry)
]
(
    entry_i,
    entry_v,
    entry_kc,
    entry_loss,
    entry_p_SWS,
    entry_n_unit,
    entry_w,
    entry_t,
    entry_f0,
    entry_Fn_K,
    entry_Vpc,
) = entries

# 其他组件保持不变...
btn_calculate = tk.Button(root, text="计算", command=calculate)
btn_calculate.pack(pady=10)

lbl_result = tk.Label(root, text="", font=("Courier New", 10), justify=tk.LEFT)
lbl_result.pack()


root.mainloop()

def calculate_R(gamma_0, a, b):
    def equation(T, b, a, gamma0):
        Tb = T * b
        gamma0b = gamma0 * b
        gamma0a = gamma0 * a

        # 左侧计算
        J0 = jv(0, Tb)
        J1 = jv(1, Tb)
        lhs = -Tb * J1 / J0

        # 右侧计算
        numerator = gamma0b * (
            iv(1, gamma0b) * kv(0, gamma0a) + iv(0, gamma0a) * kv(1, gamma0b)
        )
        denominator = iv(0, gamma0b) * kv(0, gamma0a) - iv(0, gamma0a) * kv(0, gamma0b)
        rhs = numerator / denominator

        return lhs - rhs

    initial_guess = 1.0  # 初始猜测值

    # 求解
    sol = root(equation, initial_guess, args=(b, a, gamma_0))
    return sol.x[0]


def calculate_Rowe(beta_space, a, b):
    Bes = iv(1, beta_space * b) * kv(0, beta_space * a) + iv(0, beta_space * a) * kv(
        1, beta_space * b
    )
    Rn = np.sqrt(1 - beta_space * b * Bes / (iv(0, beta_space * a)))

    return Rn

def Vpc_calc(U):
    gamma=U/5.11e5;
    Vp_c=1*m.pow(1-1/m.pow(1+gamma,2),0.5);

    return Vp_c