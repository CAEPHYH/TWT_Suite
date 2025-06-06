# twt_core.py（核心计算模块）
import math as m
import numpy as np
import Vpc_calc
from scipy.special import jv  # 仅导入贝塞尔函数
from scipy.optimize import root_scalar
from TWT_CORE_SIMP import calculate_R


def detailed_calculation(
    I, V, Kc, Loss, p_SWS, N_unit, w, t, Fn_K, f0_GHz, Vpc
):
    """返回包含中间结果和最终增益的字典"""
    try:
        # 通用参数计算
            # 单位转换
            p_SWS *= 1e-3  # mm转m
            w *= 1e-3
            t *= 1e-3
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
            Q = Wq_over_omegaC_sq / (4 * C)

        # ========== 行波管三次方程求解（图片公式） ==========
            # 原方程 δ² = 1/(-b+jd +jδ) - 4QC
            # 重排为三次方程: jδ³ + (−b+jd)δ² + (4QC)jδ + (−1−4QCb+4QCjd) = 0
            # ========== 替换4QC为Wq_over_omegaC_sq ==========
            # 重排为三次方程: jδ³ + jdδ² + (Wq_over_omegaC_sq)jδ + (Wq_over_omegaC_sq)*(j*d-b) -1 = 0
            j = 1j
            coeffs = [
                j,
                j * d - b,
                Wq_over_omegaC_sq * j,
                Wq_over_omegaC_sq * (j * d - b) - 1,
            ]
            roots = np.roots(coeffs)
            sorted_roots = sorted(roots, key=lambda x: x.real, reverse=True)
            x1, y1 = sorted_roots[0].real, sorted_roots[0].imag
            x2, y2 = sorted_roots[1].real, sorted_roots[1].imag
            x3, y3 = sorted_roots[2].real, sorted_roots[2].imag

            # 截断附加衰减
            delta1 = sorted_roots[0]
            delta2 = sorted_roots[1]
            delta3 = sorted_roots[2]

        # ========== 核心增益计算与振荡判断 ==========

            # 计算Gmax = A + 54.6*x1*C*N （图片公式5）
            numeratorA = (
                (1 + j * C * delta2)
                * (1 + j * C * delta3)
                * (delta1**2 + 4 * Q * C * (1 + j * C * delta1) ** 2)
            )
            denominatorA = (delta1 - delta2) * (delta1 - delta3)
            A = 20 * np.log10(abs(numeratorA / denominatorA))
            A = -abs(A)
            # A=-9.54
            Gmax = 54.6 * x1 * C * N
            if Gmax < 0:
                Gmax = 0
            else:
                Gmax = Gmax

            # 计算Tao = (Gmax - Loss)/2 （图片公式6）
            Tao = (Gmax - Loss) / 2

        # 截断附加衰减Ab(有限距离Lenth_att)
            Lenth_att = 0
            theta_Ab = beta_e * C * Lenth_att * np.sqrt(Wq_over_omegaC_sq)  # 衰减角
            Amp_Attu = (
                (delta2 * delta3 - delta1 * delta2 - delta1 * delta3 - 4 * Q * C)
                * m.cos(theta_Ab)
                + (
                    (delta2 + delta3 - delta1) * np.sqrt(Wq_over_omegaC_sq)
                    + delta1 * delta2 * delta3 / np.sqrt(Wq_over_omegaC_sq)
                )
                * m.sin(theta_Ab)
            ) / ((delta1 - delta2) * (delta1 - delta3))
            Ab = 20 * m.log10(Amp_Attu)  # 有限距离截断附加衰减Ab
        
        #return
            return {
                "小信号增益因子C": C,
                "互作用长度N": N,
                "慢波线上损耗L": L,
                "损耗因子d": d,
                "等离子体频率降低因子Fn": Fn,
                "等离子体频率Wp": Wp,
                "空间电荷参量4QC": Wq_over_omegaC_sq,
                "非同步参量b": b,
                "归一化电子速度Vec": Vec,
                "增幅波第一实数解x1": x1,
                "线性最大增益Gmax": Gmax,
                "慢波线最大反射Tao": Tao,
                "衰减降低增益量Ab": Ab,
                "初始化调制增益降低量A": A,
            }

    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")
