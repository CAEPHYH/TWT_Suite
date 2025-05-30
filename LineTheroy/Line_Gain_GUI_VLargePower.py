import tkinter as tk
from tkinter import messagebox
import math as m
import numpy as np
import sys

import Vpc_calc
from solve_TWTChara import solveTWT
from scipy.special import jv  # 贝塞尔函数和双曲余切
from scipy.optimize import root_scalar  # 数值求根
from sympy import symbols, expand, sqrt, I, Eq, simplify
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
        Fn_K = float(entry_Fn.get())  # 获取Fn_K手动输入
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
        # 非同步参量b


        # 计算等离子体频率降Fn
        ##圆形束流等离子体频率降特征方程-特征值数值求解
        Fn_tmp = 2.405 / r_beam  # 圆形束流等离子体频率降特征方程-特征值
        Fn = 1 / np.sqrt(1 + m.pow((Fn_tmp / beta_e), 2))*Fn_K #计算等离子体频率降Fn

        # 计算N=beta_e*p_SWS/(2*m.pi) * N_unit （图片公式3）
        N = beta_e * p_SWS / (2 * m.pi) * N_unit

        # 计算L = Loss/N （图片公式2）
        L = Loss / N
        # 计算d = 0.01836*L/C （图片公式2）
        d = 0.01836 * L / C

        # 空间电荷因子4QC
        Wq_over_omegaC_sq = (Fn * Wp / (omega * C)) ** 2
        Q=Wq_over_omegaC_sq/(4*C) #计算空间电荷因子Q

        # ========== 行波管复杂三次方程求解（图片公式） ==========
        # #复杂三次方程: jδ³ + (-b + jd + C²/4)δ² + j[4QC - C/2 - (C²/2)(b - jd)]δ + (-4QCb + QC³ -1 +5C/2 b +2C²b² -2C²d² + jd(4QC -5C/2 -4C²b)) =0
        j = 1j
        # coeffs = [
        #     j, # δ³项系数
        #     (-b + j*d + C**2/4), # δ²项系数 (-b + j*d + C²/4)
        #     j*(4*Q*C - C/2 - (C**2/2)*(b - j*d)), # δ项系数 j*(4*Q*C - C/2 - (C²/2)*(b - j*d))
        #     (Q*C**3 - 1 -4*Q*C*b + (5*C/2)*b + 2*C**2*b**2 - 2*C**2*d**2+ j*d*(4*Q*C - (5*C)/2 -4*C**2*b)), # 常数项 [Q*C³ -1 -4*Q*C*b + (5*C/2)*b + 2*C²*b² - 2*C²*d² + j*d*(4*Q*C -5*C/2 -4*C²*b)]
        # ]
        # roots = np.roots(coeffs)
        roots = solveTWT(Q, C, b, d)
        sorted_roots = sorted(roots, key=lambda x: x.real, reverse=True)
        print(sorted_roots)

        x1, y1 = sorted_roots[0].real, sorted_roots[0].imag
        x2, y2 = sorted_roots[1].real, sorted_roots[1].imag
        x3, y3 = sorted_roots[3].real, sorted_roots[2].imag

        # 截断附加衰减
        delta1 = sorted_roots[0];
        delta2 = sorted_roots[1];
        delta3 = sorted_roots[3];
        Ab=20*m.log10((delta2*delta3-delta1*delta2-delta1*delta3)/((delta1-delta2)*(delta1-delta3)))#零距离截断附加衰减Ab

        # ========== 核心增益计算与振荡判断 ==========

        # 计算Gmax = A + 54.6*x1*C*N （图片公式5）
        numeratorA = ((1 + j*C*delta2) * (1 + j*C*delta3) * (delta1**2 + 4*Q*C*(1 + j*C*delta1)**2))
        denominatorA = (delta1 - delta2) * (delta1 - delta3) 
        A=20 * np.log10(abs(numeratorA / denominatorA))
        # A=-9.54
        Gmax = A + 54.6*x1* C * N
        if Gmax<0:
            Gmax=0
        else:
            Gmax=Gmax

        # 计算Tao = (Gmax - Loss)/2 （图片公式6）
        Tao = (Gmax - Loss) / 2

        #=======计算模块结束

        # ========== 严格按图片格式输出 ==========
        result_text = (
            f"以下是计算结果："
            f"增益因子 C = {C:.3f}\nN = {N:.3f}\n"
            f"L = Loss/N={Loss:.1f}/{N:.3f} = {L:.3f}\n"
            f"损耗因子 d = 0.01836*L/C = 0.01836*{L:.3f}/{C:.3f} = {d:.2f}\n"
            f"等离子体频率降低因子计算参量 Gamma*r_beam ={gamma0*r_beam:.3e} \n"
            f"圆形束满填充等离子体频率降低因子 Fn ={Fn:.3e} \n"
            f"等离子体频率 Wp = {Wp:.3e} rad/s\n"
            f"空间电荷参量 4QC = {Wq_over_omegaC_sq:.3e}\n"  # 显示实际使用的QC值
            f"非同步参量 b={b:.3f}\n"
            f"归一化电子速度 Vec={Vec:.3f}\n"
            f"x₁ = {x1:.3f}\n"
            f"Gmax要来了：\n"
            f"Gmax={A:.3f} + 54.6*{x1:.3f}* {C:.3f} * {N:.3f}\n= {Gmax:.3f} dB\n"
            f"Required Return Loss = {Tao:.3f} dB\n"
        )
        lbl_result.config(text=result_text)

    except ValueError as e:
        messagebox.showerror("输入错误", str(e))
    except ZeroDivisionError:
        messagebox.showerror("计算错误", "除零错误（请检查输入参数）")
    return Gmax


# ==============================================
# ================== GUI界面（新增QC输入框） ==================
root = tk.Tk()
root.title("行波管线性增益计算器-SUPERPLUSPRO版")
root.geometry("500x650")

# 输入参数框架（新增QC输入行）
input_frame = tk.LabelFrame(root, text="输入参数（*为频域变量）", padx=10, pady=10)
input_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

input_labels = [
    ("束流电流 I (A):", "0.30"),
    ("束流电压 U (V):", "23000"),
    ("Kc*:", "3"),
    ("Loss*:", "0"),
    ("周期长度p_SWS (mm):", "0.600"),
    ("周期个数N_unit (正整数):", "60"),
    ("宽度 w (mm):", "0.2"),
    ("厚度 t (mm):", "0.2"),
    ("频率 f0 (GHz)*:", "210"),
    ("Fn_K:", "0"),
    ("Vpc (c)*:", "0.29032"),  # 新增Vpc输入行
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
    entry_Fn,
    entry_Vpc,
) = entries

# 其他组件保持不变...
btn_calculate = tk.Button(root, text="计算", command=calculate)
btn_calculate.pack(pady=10)

lbl_result = tk.Label(root, text="", font=("Courier New", 10), justify=tk.LEFT)
lbl_result.pack()

# --------- 公式说明 ---------
formula_text = (
    "计算公式（来自用户图片）:\n"
    "1. L = Loss / N\n"
    "2. d = 0.01836 * L / C\n"
    "其他公式:\n"
    "3. N=beta_e*p_SWS/(2*pi) * N_unit\n"
    "4. C = (I*Kc/(4V))^(1/3)\n"
    "5. Gmax = -9.54 + 54.6*x1*C*N\n"
    "6. Tao = (Gmax - Loss)/2\n"
    "7. Q = [ηI/(SVeε₀)]/ω² ÷ [KcI/(4V)]\n"
    "8. Wq² = ηI / (SVeε₀)"
)
lbl_formula = tk.Label(root, text=formula_text, justify=tk.LEFT)
lbl_formula.pack(pady=5)

root.mainloop()
