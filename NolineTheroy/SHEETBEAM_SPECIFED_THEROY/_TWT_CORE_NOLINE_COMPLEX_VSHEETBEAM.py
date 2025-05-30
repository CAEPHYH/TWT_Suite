import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import math as m
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def solveTWTNOLINE_INIT_SHEETBEAM(C, b, d, beam_params, m, A0, y_end, N_steps=1000):
    """四阶龙格库塔法求解行波管非线性方程组

    参数：
        C: 增益参量
        b: 速度非同步参量
        d: 线路损耗参量
        wp_w: 相对等离子体频率
        Rn: 空间电荷系数
        m: 电子离散数量
        A0: 初始振幅
        y_end: 模拟终止位置 (默认10)
        N_steps: 总步数 (默认1000)

    返回：
        result: 包含求解结果的字典
    """
    # ========================= 初始化设置 =========================
    y_start = 0
    h = (y_end - y_start) / N_steps  # 计算步长

    # 离散初始相位ϕ₀ (式18.3.8)
    phi0_grid = np.linspace(0, 2 * np.pi, m, endpoint=False)
    delta_phi0 = 2 * np.pi / m

    # ========================= 初始条件 =========================
    state = np.zeros(4 + 2 * m)
    state[0] = A0  # A(0) = A₀ (式18.3.1)
    state[1] = 0.0  # dA/dy(0) = 0 (式18.3.2)
    state[2] = 0.0  # θ(0) = 0 (式18.3.3)
    state[3] = -b  # dθ/dy(0) = -b (式18.3.4)
    state[4 : 4 + m] = 0.0  # u(y=0, ϕ₀) = 0 (式18.3.6)
    state[4 + m : 4 + 2 * m] = phi0_grid  # ϕ(y=0, ϕ₀) = ϕ₀ (式18.3.7)

    # ========================= 辅助函数 =========================
    def compute_derivatives(y, state):
        """计算状态向量导数的嵌套函数"""
        A, dA_dy, theta, dtheta_dy = state[0], state[1], state[2], state[3]
        u = state[4 : 4 + m]
        phi = state[4 + m : 4 + 2 * m]

        # 计算积分项
        denominator = 1 + 2 * C * u
        sum_cos = np.sum(np.cos(phi) / denominator) * delta_phi0
        sum_sin = np.sum(np.sin(phi) / denominator) * delta_phi0

        # 方程1: d²A/dy²
        term1 = A * ((1 / C - dtheta_dy) ** 2 - ((1 + C * b) / C) ** 2)
        rhs1 = (1 + C * b) / (np.pi * C) * (sum_cos + 2 * C * d * sum_sin)
        d2A_dy2 = term1 + rhs1

        # 方程2: d²θ/dy²
        term_theta = (1 + C * b) / (np.pi * C) * (sum_sin - 2 * C * d * sum_cos)
        d2theta_dy2 = (term_theta - 2 * dA_dy * (dtheta_dy - 1 / C)) / A + 2 * d / C * (
            1 + C * b
        ) ** 2

        # 方程3: ∂u/∂y (含空间电荷项)
        phi_diff = phi[:, np.newaxis] - phi  # 相位差矩阵

        Ez = compute_Esp(phi_diff, denominator, C, m, beam_params)
        term_u = (
            A * (1 - C * dtheta_dy) * np.sin(phi) - C * dA_dy * np.cos(phi) + Ez
        )  # ((wp_w/C)**2)/(1 + C*b) * F1z_integral
        du_dy = term_u / denominator

        # 方程4: ∂ϕ/∂y
        dphi_dy = (2 * u) / (1 + 2 * C * u) - dtheta_dy

        # 组合导数向量
        d_state = np.zeros_like(state)
        d_state[0] = dA_dy
        d_state[1] = d2A_dy2
        d_state[2] = dtheta_dy
        d_state[3] = d2theta_dy2
        d_state[4 : 4 + m] = du_dy
        d_state[4 + m : 4 + 2 * m] = dphi_dy

        return d_state

    # ========================= RK4 求解 =========================
    y = y_start
    y_values = np.linspace(y_start, y_end, N_steps + 1)
    A_values = np.zeros(N_steps + 1)
    theta_values = np.zeros(N_steps + 1)
    dA_dy_values = np.zeros(N_steps + 1)
    dtheta_dy_values = np.zeros(N_steps + 1)
    u_now = np.zeros((N_steps + 1, m))

    # 存储初始值
    A_values[0] = state[0]
    theta_values[0] = state[2]
    dA_dy_values[0] = state[1]
    dtheta_dy_values[0] = state[3]
    u_now[0, :] = state[4 : 4 + m]

    for i in range(1, N_steps + 1):
        # RK4步骤
        k1 = compute_derivatives(y, state)
        k2 = compute_derivatives(y + h / 2, state + h * k1 / 2)
        k3 = compute_derivatives(y + h / 2, state + h * k2 / 2)
        k4 = compute_derivatives(y + h, state + h * k3)

        state += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y += h

        # 存储结果
        A_values[i] = state[0]
        theta_values[i] = state[2]

        dA_dy_values[i] = state[1]
        dtheta_dy_values[i] = state[3]
        u_now[i, :] = state[4 : 4 + m]

    return {
        "y": y_values,
        "A": A_values,
        "theta": theta_values,
        "u_now": u_now,
        "A_Ends": A_values[-1],
        "theta_Ends": theta_values[-1],
        "dA_dy_Ends": dA_dy_values[-1],
        "dtheta_dy_Ends": dtheta_dy_values[-1],
        "u_final": state[4 : 4 + m],
        "phi_final": state[4 + m : 4 + 2 * m],
        "phi0_grid": phi0_grid,
        "dphi_dy": k1[4 + m : 4 + 2 * m],
    }


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
    Q = I / (1*np.pi*freq *m)  # 电荷线密度

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
        Esp[i] = (u0 / (4 * omega * C**2 * V0)) * (pre_coeff * sum_total_i)

    return Esp


def compute_Esp_V2(
    phi_diff_matrix: np.ndarray,  # 形状 (num_electrons, num_electrons)
    denominator: np.ndarray,      # 形状 (num_electrons,)
    C: float,
    num_electrons: int,
    beam_params: list
) -> np.ndarray:
    # 参数初始化
    x_d, y_d, x_b, y_b, I, V0, freq = (
        beam_params[0] * 1e-3,   # mm -> m
        beam_params[1] * 1e-3,
        beam_params[2] * 1e-3,
        beam_params[3] * 1e-3,
        beam_params[4],          # A
        beam_params[5],          # V
        beam_params[6] * 1e9,    # GHz -> Hz
    )
    epsilon0 = 8.854e-12
    eta = 1.76e11 
    omega = 2 * np.pi * freq
    Q0 = I / (1*np.pi*freq *num_electrons) 
    u0=np.sqrt(2*eta*V0)

    # 计算纵向位置差（广播分母项）
    z_diff = phi_diff_matrix * denominator.reshape(1, -1) * (u0 / omega)  # (num, num)
    sgn = np.sign(z_diff)
    abs_z_diff = np.abs(z_diff)

    # 多模级数参数
    m_max = n_max = 10
    p, q = np.arange(1, m_max+1), np.arange(1, n_max+1)
    P, Q = np.meshgrid(p, q, indexing='ij')  # (m_max, n_max)

    # 模式参数计算
    sigma_x = x_b / x_d
    sigma_y = y_b / y_d
    zeta = x_b / y_b
    alpha_p = P * np.pi / (2 * sigma_x)  # (m_max, n_max)
    beta_q = Q * np.pi / (2 * sigma_y)
    mu_pq = np.sqrt(alpha_p**2 + (zeta * beta_q)**2)

    # 级数项与指数项计算
    series_terms = (2*np.sin(alpha_p) * np.sin(beta_q))**2 / (alpha_p * beta_q)**2
    exponent = -2 * (mu_pq[..., None, None] / zeta) * (abs_z_diff / y_b)  # (m_max, n_max, num, num)
    exponential = np.exp(exponent)

    # 多模求和（爱因斯坦求和）
    sum_series = np.einsum('pq,pqij->ij', series_terms, exponential)  # (num, num)

    # 合成电场矩阵（包含所有i,j对）
    abs_term = np.abs(Q0 / (2 * epsilon0 * x_b * y_b))
    sigma_product = sigma_x * sigma_y
    Esp_matrix = (u0 / (4 * omega * C**2 * V0)) * abs_term * sigma_product * sgn * sum_series

    # 对角线归零（排除自相互作用）
    np.fill_diagonal(Esp_matrix, 0)

    # 计算每个电子的净电场（行求和）
    Esp_vector = np.sum(Esp_matrix, axis=1)  # 形状 (num_electrons,)

    return Esp_vector


