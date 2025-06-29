import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import math as m
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def compute_Esp(phi_diff_matrix, denominator, C, m, beam_params):
    """
    优化后的Esp计算函数，利用向量化操作减少循环嵌套

    参数:
        phi_diff_matrix (ndarray): 相位差矩阵，形状(m, m)
        denominator (ndarray): 分母项 1 + 2*C*u，形状(m,)
        C: 增益参量
        m: 电子数
        beam_params: 束流参数

    返回:
        ndarray: Esp向量，形状(m,)
    """
    # 根据beam_params初始化参数
    x_d = beam_params[0] * 1e-3
    y_d = beam_params[1] * 1e-3
    x_b = beam_params[2] * 1e-3
    y_b = beam_params[3] * 1e-3
    I = beam_params[4]
    V0 = beam_params[5]
    freq = beam_params[6] * 1e9

    m_max = n_max = 1e1  # 级数截断值，可根据需要调整
    eta = 1.76e11  # 电子荷质比 (C/kg)
    epsilon_0 = 8.854e-12
    u0 = np.sqrt(2 * eta * V0)
    omega = 2 * np.pi * freq
    Q = I / (freq * m)  # 电荷线密度

    # 常数前置系数
    pre_coeff = 2 * x_d * y_d * Q / (np.pi**4 * epsilon_0 * (x_b * y_b) ** 2)

    # 预生成p和q的网格
    p = np.arange(1, m_max + 1)
    q = np.arange(1, n_max + 1)
    P, Q_grid = np.meshgrid(p, q, indexing="ij")  # 形状(m_max, n_max)

    # 预计算cos_p, cos_q及sqrt_term
    cos_p = (
        np.cos(np.pi * P / (2 * x_d) * (x_d + x_b))
        - np.cos(np.pi * P / (2 * x_d) * (x_d - x_b))
    ) ** 2
    cos_q = (
        np.cos(np.pi * Q_grid / (2 * y_d) * (y_d + y_b))
        - np.cos(np.pi * Q_grid / (2 * y_d) * (y_d - y_b))
    ) ** 2
    sqrt_term_pq = np.sqrt((y_d * P) ** 2 + (x_d * Q_grid) ** 2)

    # 预计算公共因子和系数
    precomputed_cos_pq_div_pq_sq = (cos_p * cos_q) / (P**2 * Q_grid**2)
    factor = np.pi / (x_d * y_d * omega) * u0

    # 初始化Esp向量
    Esp = np.zeros(m)

    for i in range(m):
        # 处理当前行的相位差
        phi_diff_row = phi_diff_matrix[i, :]
        sign_i = np.sign(phi_diff_row)
        sign_i[phi_diff_row == 0] = 0.0
        abs_phi_diff_i = np.abs(phi_diff_row)

        # 向量化计算指数项
        exponents = (
            -factor
            * sqrt_term_pq[:, :, np.newaxis]
            * denominator[np.newaxis, np.newaxis, :]
            * abs_phi_diff_i[np.newaxis, np.newaxis, :]
        )
        exp_terms = np.exp(exponents)

        # 计算sum_terms_i并聚合
        sum_terms_i = np.sum(
            precomputed_cos_pq_div_pq_sq[:, :, np.newaxis] * exp_terms, axis=(0, 1)
        )
        sum_total_i = np.dot(sign_i, sum_terms_i)

        # 计算Esp[i]
        Esp[i] = (-eta / (omega * C**2 * u0)) * (pre_coeff * sum_total_i)

    return Esp


def compute_Esp_V2(
    phi_diff_matrix: np.ndarray,  # 形状 (num_electrons, num_electrons)
    denominator: np.ndarray,  # 形状 (num_electrons,)
    C: float,
    num_electrons: int,
    beam_params: list,
) -> np.ndarray:
    # 参数初始化
    x_d, y_d, x_b, y_b, I, V0, freq = (
        beam_params[0] * 1e-3,  # mm -> m
        beam_params[1] * 1e-3,
        beam_params[2] * 1e-3,
        beam_params[3] * 1e-3,
        beam_params[4],  # A
        beam_params[5],  # V
        beam_params[6] * 1e9,  # GHz -> Hz
    )
    epsilon0 = 8.854e-12
    eta = 1.76e11
    omega = 2 * np.pi * freq
    Q0 = I / (freq * num_electrons)
    u0 = np.sqrt(2 * eta * V0)

    # 计算纵向位置差（广播分母项）
    z_diff = phi_diff_matrix * denominator.reshape(1, -1) * (u0 / omega)  # (num, num)
    sgn = np.sign(z_diff)
    abs_z_diff = np.abs(z_diff)

    # 多模级数参数
    m_max = n_max = 10
    p, q = np.arange(1, m_max + 1), np.arange(1, n_max + 1)
    P, Q = np.meshgrid(p, q, indexing="ij")  # (m_max, n_max)

    # 模式参数计算
    sigma_x = x_b / x_d
    sigma_y = y_b / y_d
    zeta = x_b / y_b
    alpha_p = P * (np.pi / 2) * sigma_x  # (m_max, n_max)
    beta_q = Q * (np.pi / 2) * sigma_y
    mu_pq = np.sqrt(alpha_p**2 + (zeta * beta_q) ** 2)

    # 级数项与指数项计算
    series_terms = (2 * np.sin(alpha_p) * np.sin(beta_q)) ** 2 / (alpha_p * beta_q) ** 2
    exponent = (
        -2 * (mu_pq[..., None, None] / zeta) * (abs_z_diff / y_b)
    )  # (m_max, n_max, num, num)
    exponential = np.exp(exponent)

    # 多模求和（爱因斯坦求和）
    sum_series = np.einsum("pq,pqij->ij", series_terms, exponential)  # (num, num)

    # 合成电场矩阵（包含所有i,j对）
    abs_term = np.abs(Q0 / (2 * epsilon0 * x_b * y_b))
    sigma_product = sigma_x * sigma_y
    Esp_matrix = (
        (-eta / (2*omega * C**2 * u0)) * abs_term * sigma_product * sgn * sum_series
    )

    # 对角线归零（排除自相互作用）
    np.fill_diagonal(Esp_matrix, 0)

    # 计算每个电子的净电场（行求和）
    Esp_vector = np.sum(Esp_matrix, axis=1) * 1  # 形状 (num_electrons,)

    return Esp_vector


def compute_esc_V2Z(
    phi_diff_matrix: np.ndarray,  # 形状 (num_electrons, num_electrons)
    denominator: np.ndarray,  # 形状 (num_electrons,)
    C: float,
    num_electrons: int,
    beam_params: list,
    M=100,
    K=100,
):
    """
    计算空间电荷场 E_{sc,z}.

    参数:
        W, H: 束流宽度和厚度 (标量)
        a, b_prime: 通道宽度和高度 (标量)
        q_k: 每个电荷的电荷量 (标量)
        eps0: 真空介电常数 (标量)
        z_positions: 电荷位置数组 (形状 (N,))
        M, K: 级数截断上限 (整数，默认100)

    返回:
        E_sc_z: 每个电荷的电场 (形状 (N,))
    """
    """
    正确计算空间电荷场 E_{sc,z}
    """
    # 参数初始化
    a = beam_params[0] * 1e-3  # mm -> m
    b_prime = beam_params[1] * 1e-3
    W = beam_params[2] * 1e-3
    H = beam_params[3] * 1e-3
    I = beam_params[4]  # A
    V0 = beam_params[5]  # V
    freq = beam_params[6] * 1e9  # GHz -> Hz
    
    epsilon0 = 8.854e-12  # F/m
    eta = 1.76e11  # C/kg (电子荷质比)
    omega = 2 * np.pi * freq
    q_k = I / (freq * num_electrons)  # 每个电子的电荷量
    u0 = np.sqrt(2 * eta * V0)  # 电子初速度
    
    # 1. 计算 z 位置差异 (z - z')
    z_diff = phi_diff_matrix * denominator.reshape(1, -1) * (u0 / omega)
    
    # 2. 计算距离 |z - z'| 和符号 sgn(z - z')
    abs_z_diff = np.abs(z_diff)  # 距离绝对值
    sign_mat = np.sign(z_diff)   # 符号矩阵
    
    # 3. 初始化电场累加矩阵
    E_sc_z_matrix = np.zeros_like(z_diff)
    
    # 4. 只对奇数 m 和 k 求和（因为 sin²(mπ/2) 和 sin²(kπ/2) 在偶数时为0）
    m_vals = np.arange(1, M + 1, 2)
    k_vals = np.arange(1, K + 1, 2)
    
    # 预计算常数
    sin_sq_m = 1.0  # sin²(mπ/2) = 1 (因为 m 是奇数)
    sin_sq_k = 1.0  # sin²(kπ/2) = 1 (因为 k 是奇数)
    
    for m in m_vals:
        # 公式中的 sin(mπW/(2a))
        sin_m = np.sin(m * np.pi * W / (2 * a))
        exp_term_m = np.exp(-np.pi * m * abs_z_diff / b_prime)
        
        for k in k_vals:
            # 公式中的 sin(kπH/(2b'))
            sin_k = np.sin(k * np.pi * H / (2 * b_prime))
            exp_term_k = np.exp(-np.pi * k * abs_z_diff / a)
            
            # 公式中的完整项
            term = (1/(m * k)) * sin_sq_m * sin_sq_k * sin_m * sin_k * exp_term_m * exp_term_k
            E_sc_z_matrix += term
    
    # 5. 应用符号和常数因子
    constant = (8 / (np.pi**2 * epsilon0)) * (q_k / (W * H))
    E_sc_z_matrix = sign_mat * constant * E_sc_z_matrix
    
    # 6. 去除自身作用（对角元素置零）
    np.fill_diagonal(E_sc_z_matrix, 0)
    
    # 7. 对源电荷求和（沿列方向求和）
    E_sc_z = np.sum(E_sc_z_matrix, axis=1)
    
    # 8. 应用归一化校正
    # E_sc_zm = -eta/(2*ω*C²*u0) * E_sc_z
    norm_factor = -eta / (2 * omega * C**2 * u0)
    E_sc_zm = norm_factor * E_sc_z

    return E_sc_zm


def compute_Esp_VR(
    phi_diff_matrix: np.ndarray,  # 形状 (num_electrons, num_electrons)
    denominator: np.ndarray,  # 形状 (num_electrons,)
    delta_phi0: float,
    C,
    b,
    num_electrons: int,
    beam_params: list,
) -> np.ndarray:
    # 参数初始化
    F1z_integral = np.zeros(num_electrons)
    # beam_params初始化
    Rn_sqr_values = beam_params[0]
    wp_w = beam_params[1]
    Space_CUT = beam_params[2]

    n_values = np.arange(1, Space_CUT)  # n ∈ [1, Space_cut)

    for n in n_values:
        Rn_sqr = Rn_sqr_values
        term = (np.sin(n * phi_diff_matrix) * Rn_sqr) / (2 * np.pi * n)
        sum_term = np.sum(term / denominator[np.newaxis, :], axis=0) * delta_phi0
        F1z_integral += sum_term

    Esp_vector = (
        -((wp_w / C) ** 2) / (1 + C * b) * F1z_integral
    )  # 形状 (num_electrons,)

    return Esp_vector
