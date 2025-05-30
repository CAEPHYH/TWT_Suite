import numpy as np
import sys
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from sko.PSO import PSO


from Noline_MAINCALL_VCORE_FORPVTOPT import calculate_SEGMENT_TWT_NOLINE_for_PVTOPT


def para_Each_PVTSEGMENT_CALC(p_SWS):
    """计算每个PVT段的参数

    参数：
    p_SWS : float - 慢波结构周期参数

    返回：
    dict : 包含Vpc和Kc的字典
    """
    Vpc = 0.288 + (p_SWS - 0.5) * 0.05
    Kc = 3.6 + (p_SWS - 0.5) * 0
    return {"Vpc": Vpc, "Kc": Kc}


class FitnessEvaluator:
    """适应度评估器，封装计算逻辑以支持多进程"""

    def __init__(self, fixed_params, verbose=True):
        self.fixed_params = fixed_params
        self.verbose = verbose

    def __call__(self, x):
        """计算单个个体的适应度值"""
        try:
            # 解包固定参数
            I = self.fixed_params["I"]
            V = self.fixed_params["V"]
            Loss = self.fixed_params["Loss"]
            N_unit = self.fixed_params["N_unit"]
            w = self.fixed_params["w"]
            t = self.fixed_params["t"]
            Fn_K = self.fixed_params["Fn_K"]
            f0_GHz = self.fixed_params["f0_GHz"]

            # 计算TWT性能
            p_SWS = x.tolist() if isinstance(x, np.ndarray) else x
            twt_result = calculate_SEGMENT_TWT_NOLINE_for_PVTOPT(
                I, V, Loss, p_SWS, N_unit, w, t, Fn_K, f0_GHz, para_Each_PVTSEGMENT_CALC
            )
            return -twt_result["输出功率P_out"]  # 负号转为最小化问题
        except Exception as e:
            if self.verbose:
                print(f"无效参数: {np.round(x,3)}, 错误: {str(e)[:50]}")
            return float("inf")


def optimize_TWT_with_PSO(fixed_params, pso_config, verbose=True):
    """
    TWT参数优化函数（PSO算法）

    参数：
    fixed_params : dict - 固定参数配置
    pso_config : dict - PSO算法配置
    verbose : bool - 是否显示优化过程信息

    返回：
    tuple : (最优p_SWS数组, 最大输出功率)
    """
    # 验证参数维度
    n_dim = len(fixed_params["N_unit"])
    if len(pso_config["lb"]) != n_dim or len(pso_config["ub"]) != n_dim:
        raise ValueError("边界条件维度与N_unit长度不一致")

    # 创建适应度评估器
    fitness_evaluator = FitnessEvaluator(fixed_params, verbose)

    # 配置并行处理
    cpus = pso_config.get("cpus", cpu_count())
    if verbose:
        print(f"使用多进程并行计算 (CPU核心数: {cpus})")

    # 初始化PSO
    pso = PSO(
        func=fitness_evaluator,
        n_dim=n_dim,
        pop=pso_config["pop_size"],
        max_iter=pso_config["max_iter"],
        lb=pso_config["lb"],
        ub=pso_config["ub"],
    )

    # 创建进程池
    pool = Pool(processes=cpus)

    # 保存原始cal_y方法
    original_cal_y = pso.cal_y

    # 定义新的并行评估方法
    def parallel_cal_y():
        """并行评估整个种群的适应度"""
        try:
            # 获取当前种群位置
            population = pso.X

            # 并行计算适应度
            results = pool.map(fitness_evaluator, population)
            pso.Y = np.array(results).reshape(-1, 1)
        except Exception as e:
            if verbose:
                print(f"并行评估错误: {str(e)}")
            # 出错时回退到原始方法
            original_cal_y()

    # 覆盖PSO的cal_y方法以实现并行评估
    pso.cal_y = parallel_cal_y

    # 运行优化
    if verbose:
        print("开始PSO优化...")
        print(f"搜索空间维度: {n_dim}")
        print(f"种群大小: {pso_config['pop_size']}")
        print(f"最大迭代次数: {pso_config['max_iter']}")

    try:
        pso.run()
    except Exception as e:
        if verbose:
            print(f"优化过程中出错: {str(e)}")
    finally:
        # 确保优化完成后关闭进程池
        pool.close()
        pool.join()

    # 获取最优结果
    optimal_p_SWS = np.round(pso.gbest_x, 4)

    if verbose:
        print("\n优化完成！")
        print(f"最优p_SWS: {optimal_p_SWS.tolist()}")

    return optimal_p_SWS


def run_optimization_example(enable_optimization=True, save_results=True):
    """运行优化示例

    参数：
    enable_optimization : bool - 是否执行优化过程
    save_results : bool - 是否保存优化结果
    """
    # ================= 参数配置 =================
    # 固定参数
    fixed_params = {
        "I": 0.3,  # 束流电流 (A)
        "V": 23000,  # 工作电压 (V)
        "Loss": 0,  # 损耗参数
        "N_unit": [30, 5, 5, 5, 5],  # 各段单元数
        "w": 0.2,  # 结构参数w (mm)
        "t": 0.2,  # 结构参数t (mm)
        "Fn_K": 1,  # 填充系数
        "f0_GHz": 211,  # 工作频率 (GHz)
    }

    # PSO配置
    pso_config = {
        "pop_size": 20,  # 种群大小
        "max_iter": 20,  # 最大迭代次数
        "lb": [0.4] * 5,  # 参数下限
        "ub": [0.6] * 5,  # 参数上限
        "cpus": 32,  # 使用的CPU核心数
    }

    # 初始参数
    initial_p_SWS = [0.50, 0.50, 0.50, 0.50, 0.50]

    # ================= 初始计算 =================
    print("执行初始参数计算...")
    initial_result = calculate_SEGMENT_TWT_NOLINE_for_PVTOPT(
        fixed_params["I"],
        fixed_params["V"],
        fixed_params["Loss"],
        initial_p_SWS,
        fixed_params["N_unit"],
        fixed_params["w"],
        fixed_params["t"],
        fixed_params["Fn_K"],
        fixed_params["f0_GHz"],
        para_Each_PVTSEGMENT_CALC,
    )
    initial_power = initial_result["输出功率P_out"]
    print(f"初始参数功率: {initial_power:.2f} W")

    # 计算并显示每个PVT段的参数
    print("\n初始PVT段参数计算:")
    for i, p in enumerate(initial_p_SWS):
        segment_params = para_Each_PVTSEGMENT_CALC(p)
        print(
            f"段 {i+1}: p_SWS={p:.3f} -> Vpc={segment_params['Vpc']:.3f}, Kc={segment_params['Kc']:.3f}"
        )

    # 如果禁用优化，直接返回初始结果
    if not enable_optimization:
        print("\n优化已禁用，仅显示初始计算结果")
        return initial_p_SWS, initial_power

    # ================= 执行优化 =================
    print("\n开始优化过程...")
    optimized_p_SWS = optimize_TWT_with_PSO(
        fixed_params=fixed_params, pso_config=pso_config, verbose=True
    )

    # ================= 结果验证 =================
    print("\n验证优化结果...")
    verification_result = calculate_SEGMENT_TWT_NOLINE_for_PVTOPT(
        fixed_params["I"],
        fixed_params["V"],
        fixed_params["Loss"],
        optimized_p_SWS.tolist(),
        fixed_params["N_unit"],
        fixed_params["w"],
        fixed_params["t"],
        fixed_params["Fn_K"],
        fixed_params["f0_GHz"],
        para_Each_PVTSEGMENT_CALC,
    )
    verified_power = verification_result["输出功率P_out"]

    # ================= 结果对比 =================
    print("\n优化结果对比:")
    print(f"初始p_SWS: {initial_p_SWS}")
    print(f"初始功率: {initial_power:.2f} W")
    print(f"优化后p_SWS: {optimized_p_SWS.tolist()}")
    print(f"验证功率: {verified_power:.2f} W")

    # 显示优化后的PVT段参数
    print("\n优化后PVT段参数:")
    for i, p in enumerate(optimized_p_SWS):
        segment_params = para_Each_PVTSEGMENT_CALC(p)
        print(
            f"段 {i+1}: p_SWS={p:.3f} -> Vpc={segment_params['Vpc']:.3f}, Kc={segment_params['Kc']:.3f}"
        )

    # 保存结果
    if save_results:
        # 保存优化结果
        np.save("optimized_p_SWS.txt", optimized_p_SWS)
        print("\n优化结果已保存到 optimized_p_SWS.txt")

        # 保存PVT段参数
        pvt_params = [para_Each_PVTSEGMENT_CALC(p) for p in optimized_p_SWS]
        np.save("optimized_pvt_params.txt", pvt_params)
        print("优化后的PVT段参数已保存到 optimized_pvt_params.txt")

    return optimized_p_SWS


def main():
    """主函数，处理命令行参数和用户交互"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="TWT参数优化工具")
    parser.add_argument("--optimize", action="store_true", help="启用优化过程")
    parser.add_argument("--no-save", action="store_true", help="禁用结果保存")
    parser.add_argument(
        "--batch", action="store_true", help="使用批处理模式（禁用交互）"
    )
    args = parser.parse_args()

    if not any(vars(args).values()) or not args.batch:
        print("\n===== TWT参数优化工具（交互模式）=====")
        response = input("是否执行优化过程? (y/n): ").strip().lower()
        enable_optimization = response == "y"

        if enable_optimization:
            response = input("是否保存优化结果? (y/n): ").strip().lower()
            save_results = response == "y"
        else:
            save_results = False

        print(f"\n配置: 优化={enable_optimization}, 保存结果={save_results}")
        run_optimization_example(enable_optimization, save_results)

    # 命令行参数模式
    else:
        enable_optimization = args.optimize
        save_results = not args.no_save
        print(f"命令行配置: 优化={enable_optimization}, 保存结果={save_results}")
        run_optimization_example(enable_optimization, save_results)


if __name__ == "__main__":
    main()
