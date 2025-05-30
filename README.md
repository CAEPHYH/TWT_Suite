# 行波管计算工具集详细说明文档

## 🖥️ GUI工具模块

## **PYQT-GUI**（带有PYQT后缀的）(3+2=5个)

- `PYQT_Line_Gain_MAIN-MIXD-COMPLEX_VN_PYQT_`  
      📌 功能：多段行波管频谱计算(仅支持零距离切断)
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节

  - `🌟PYQT_Line_Gain_MAIN-MIXD-MORECOMPLEX_VN_PYQT_` `🌟(用这个)`
      📌 功能：多段行波管频谱计算  
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节  

  - `🌟PYQT_Line_Gain_MAIN-SINGLE_COMPLEXM_PYQT_``🌟(用这个)`  
      📌 功能：单段行波管频谱计算  
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节

  - `🌟PYQT_Noline_GAIN_MAINGUI-SINGLE_COMPLEXM_SUPER_PYQT_` `(用这个)`  
      📌 功能：单段行波管频谱计算(非线性理论)  
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节  

    - `PYQT_NoLine_Gain_MAINGUI-MIXD-MORECOMPLEX_VN_PYQT_`  
    📌 功能：多段行波管频谱计算(非线性理论) 开发中，实现过于困难-图一乐

  ## **特殊GUI**（1+1+2=4个）

  - `_Line_Gain_INIT_VEXTREME_COMPLEX`  
    📌 用途：大增益单段行波管单频点计算（GUI模式）

  - `Line_Gain_INIT_VCLASSIC`  
    📌 用途：常规单段行波管单频点计算（基础GUI）

  - `Line_Gain_MAIN-MIXD-SIMPLE_DUAL`  
    📌 用途：常规双段行波管单频点计算（基础GUI）

  - `Line_Gain_MAIN-MIXD-SIMPLE_V3X`  
    📌 用途：常规多段行波管单频点计算（基础GUI）

## 🔧 实用小工具（3+1+2=6个）

- `MERGE_CSV`  
    📌 用途：合并HFSS计算结果文件（支持本征模式/阻抗-phi/相速-phi等）  
- `Vpc_calc`  
    📌 用途：电压→归一化电子速度快速计算  
- `Calc_Beam_EquipErb (2)`  
    📌 用途：电子束等效介电常数计算  

- `Useful_Tool_CN_calc`  
    📌 用途：简单估计反波起震荡
      -`Useless_Tool_Calc_SWSWORKING_V0`  
    📌 用途：简单估计工作点
      -`Useless_Tool_Pediocd_SWS_Calc`  
        📌 用途：简单估计周期长度

---

## `通用参数`一般指：`def simple_calculation( # I: Any,# V: Any,# Kc: Any,# Loss: Any,# p_SWS: Any,# N_unit: Any,# w: Any,# t: Any,# Fill_Rate: Any,# f0_GHz: Any,# Vpc: Any)`

---

## 🔧 TEST工具(对非线性理论来说很重要)（4个）

- `Noline_GAIN_MAINCALL_V2TEST`
  - 📌 用途：顾名思义，实例化`_TWT_CORE_NOLINE_COMPLEX_V2.py`：
  - 传一组`通用参数`转换非线性理论计算所需的参数，在对结果后处理，可以得到:

      ```python
      - print('The Gmax in Noline Theroy is %.4f dB '%(20*np.log10((result['A_Ends']/params['A0']))))
        print('The Gain in Line Theroy is %.3f dB'%(resultLineG['Gmax']))
        print('The P_out in Noline Theroy is %.4f,The maximum Efficence is %.4f in percent,The maximum POWER is %.4f in Watt'%(P_Out[-1],Eff,P_max))
        print(f"The A in the end is {result['A'][-1]}, The u in the end is {np.mean(abs(result['u_final']))}")
        振幅演化，相位演化， 最终速度分布， 最终相位分布，电子相空间图，轴向功率图等结果

- `Noline_GAIN_MAINCALL_V2SUPERTEST` `🌟(用这个)`

  - 📌 用途：顾名思义，实例化`_TWT_CORE_NOLINE_COMPLEX_V2SUPER.py`：
  - 传一组`通用参数`转换非线性理论计算所需的参数，在对结果后处理，可以得到:

    ```python
    - print('The Gmax in Noline Theroy is %.4f dB '%(20*np.log10((result['A_Ends']/params['A0']))))
      print('The Gain in Line Theroy is %.3f dB'%(resultLineG['Gmax']))
      print('The P_out in Noline Theroy is %.4f,The maximum Efficence is %.4f in percent,The maximum POWER is %.4f in Watt'%(P_Out[-1],Eff,P_max))
      print(f"The A in the end is {result['A'][-1]}, The u in the end is {np.mean(abs(result['u_final']))}")
      振幅演化，相位演化， 最终速度分布， 最终相位分布，电子相空间图，轴向功率图等结果

- `Noline_GAIN_MAINCALL_V2TEST-MIX_ATTUSUPER` `🌟(非常NB)`

  - 📌 用途：顾名思义，实例化`_TWT_CORE_NOLINE_COMPLEX_V2.py`以及`_TWT_CORE_NOLINE_COMPLEX_V2_MIX`：
  - 传一组`通用参数`根据相速跳变段的配置情况、截断的配置情况转化为对应`固定参数`+`每段相应参数`:
  - `固定参数`:`COMMON_PARAMS = {
            "I": 0.3,          # 电流 (A)
            "V": 23000,        # 电压 (V)
            "Loss": 0,         # 损耗
            "w": 0.20,         # 波导宽度
            "t": 0.20,         # 波导厚度
            "Fn_K": 1,         # 归一化因子
        }`
  - `每段相应参数`:`{   段X（新增段）
            "len": 5,# 单元个数
            "Vpc": 0.288,# 归一化相速
            "p_SWS": 0.50,# 周期长度(mm)
            "Kc": 3.6,# 耦合阻抗
            "f0_GHz": 211# 工作频率 (GHz)
      }`

  - 并`将上述参数针对每段情况转换成相应非线性理论计算所需的参数`，最后对`结果后处理(合并累加等操作)`，可以得到`总的`:

      ```python
      - print('The Gmax in Noline Theroy is %.4f dB '%(20*np.log10((result['A_Ends']/params['A0']))))
        print('The Gain in Line Theroy is %.3f dB'%(resultLineG['Gmax']))
        print('The P_out in Noline Theroy is %.4f,The maximum Efficence is %.4f in percent,The maximum POWER is %.4f in Watt'%(P_Out[-1],Eff,P_max))
        print(f"The A in the end is {result['A'][-1]}, The u in the end is {np.mean(abs(result['u_final']))}")
        振幅演化，相位演化， 最终速度分布， 最终相位分布，电子相空间图，轴向功率图等结果

  `适用于下列情况`:
  `情况A`:Uniform-截断-PVT-PVT的复杂行波管`
  `理论上情况B`:可以扩展到倍频器这种：`Uniform(W band)-截断-Uniform(340 band)-PVT(340 band)`，
  `理论上情况C`:也可以扩展混合色散这种：`Uniform(W band 强耦合)-截断-Uniform(W band 弱耦合)-PVT(W band 弱耦合)`
  一般用于`情况A`

- `Noline_GAIN_MAINCALL_V2TEST-MIX_UPER` `🌟(非常NB)`
  - 📌 用途：与 `Noline_GAIN_MAINCALL_V2TEST-MIX_ATTUSUPER` 类似，不过做了些微简化(不考虑截断)
-

---

## ⚙️ 核心计算函数库(CORE字样)(4+4=8个左右)

### **线性理论核心计算库**（4个）

- `solve_TWTChara`  
    📦 功能：大增益场景核心求解过程  
- `TWT_CORE_COMPLEX`  
    📦 功能：零距离截断线性理论计算  
- `TWT_CORE_MORE_COMPLEX`  
    📦 功能：非零距离截断线性理论计算  
- `TWT_CORE_SIMP`谨慎修改
  - 📦功能：(
  - 包括多个函数谨慎修改：
  - 其中def detailed_calculation()用于无截断线性理论计算；
  - def simple_calculation()用于将`通用参数`转换为非线性理论计算所需的:

  ```python
  - {
            "小信号增益因子C": C,
            "互作用长度N": N,
            "慢波线上损耗L": L,
            "损耗因子d": d,
            "等离子体频率降低因子Fn": Fn,
            "等离子体频率Wp": Wp,
            "空间电荷参量4QC": Wq_over_omegaC_sq,
            "非同步参量b": b,
            "归一化电子速度Vec": Vec,
            "束流归一化尺寸r_beam": r_beam,
            "beta_Space": beta_Space,
            "Rowe特征值R": R,
        }参数
  - )

### **非线性理论核心计算库**（4个）

- `_TWT_CORE_NOLINE_COMPLEX_V2/_TWT_CORE_NOLINE_COMPLEX_V2SUPER`  
    📦 功能：非线性理论核心计算过程  
- `_TWT_CORE_NOLINE_COMPLEX_V2_MIX`  
    📦 功能：非线性理论核心计算过程(相速度跳变，截断等等)  
- `Noline_GAIN_MAINCALL_V2CORE.py`  
    📦 功能：将`通用参数`转换为_TWT_CORE_NOLINE_COMPLEX_V2.py所需的:

    ```python
          params = {
              "C": C,  # 增益参量
              "b": b,  # 速度非同步参量
              "d": d,  # 线路损耗参量
              "wp_w": wp_omega,  # 相对等离子体频率
              "Rn": R,  # 电子波数beta_e
              "m": 100,  # 电子离散数量
              "A0": A_0,  # 初始振幅
              "y_end": L,
          }等参数,并对结果后处理返回所需的功率P_out,Eff等参数
      适用于无截断单段行波管非线性计算 
       
- `Noline_GAIN_MAINCALL_V2CORE_SUPER.py`  

    📦 功能：将`通用参数`转换为_TWT_CORE_NOLINE_COMPLEX_V2SUPER.py所需的:

    ```python
          params = {
              "C": C,  # 增益参量
              "b": b,  # 速度非同步参量
              "d": d,  # 线路损耗参量
              "wp_w": wp_omega,  # 相对等离子体频率
              "beta_space": beta_Space,  # 电子波数beta_space
              "r_beam": r_beam,
              "Fill_Rate": Fill_Rate,
              "p_SWS": p_SWS,
              "m": 100,  # 电子离散数量
              "A0": A_0,  # 初始振幅
              "y_end": L,
              'Space_cut':2,
          }等参数，并对结果后处理返回所需的功率P_out,Eff等参数
      适用于无截断单段行波管非线性计算

---

⚠️ **重要说明**  

1. 标有`CORE`前缀的函数为内部核心函数，建议开发者谨慎修改  
2. GUI工具中标注`PYQT`的表示基于PyQt框架开发  
3. 非线性理论模块仍在持续开发中，接口可能变动  
