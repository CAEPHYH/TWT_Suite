# 行波管计算工具集详细说明文档

## 🖥️ GUI工具模块

## **PYQT-GUI**（带有PYQT后缀的）
**位于.\LineTheroy目录下，基于线性理论，主要面向圆形束流行波管优化**：

- `PYQT_Line_Gain_MAIN-MIXD-COMPLEX_VN_PYQT_`  
      📌 功能：多段行波管频谱计算(仅支持零距离切断)
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节

- `🌟PYQT_Line_Gain_MAIN-MIXD-MORECOMPLEX_VN_PYQT_` `🌟(用这个)`
📌 功能：多段行波管频谱计算  
🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节  

- `🌟PYQT_Line_Gain_MAIN-SINGLE_COMPLEXM_PYQT_``🌟(用这个)`  
📌 功能：单段行波管频谱计算  
🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节
    
**位于.\NolineTheroy\COMMON_THEROY目录下，主要面向圆形束流行波管优化：**

- `🌟PYQT_Noline_GAIN_MAINGUI-SINGLEOPT_PYQT_` `🌟(用这个)`  
      📌 功能：单段行波管**频谱计算**(非线性理论)  
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节  

- `PYQT_NOLine_Gain_MAINGUI-MIXD_MIX_PYQT_`  
📌 功能：分段行波管**频谱计算**(非线性理论)

- `🌟PYQT_Noline_PVTOPT_MAINGUI_MIX_forPVTOPT_PYQT` `🌟(用这个)`  
      📌 功能：行波管单频点**全周期相速度优化（PVT）**(单段、零距离切断均可[此时默认配置是均匀段-切断-均匀段-切断-{PVT段-PVT段}])(非线性理论)  
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节

**位于.\NolineTheroy\CicularBEAM_SPECIFED_THEROY目录下，只能用于圆形束流行波管优化(SUPER字样)：**
  - `🌟PYQT_NOLine_Gain_MAINGUI-MIXD_SUPER_MIX_PYQT_` `🌟(用这个)`  
      📌 功能：**分段**行波管**频谱计算**(非线性理论)
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节  

    - `PYQT_Noline_GAIN_MAINGUI-SINGLE_SUPER_PYQT_`  
      📌 功能：**单段**行波管**频谱计算**(非线性理论)  
      🌟 特色功能：数据导入导出、曲线叠绘、自动保存/载入参数、滑块参数调节  


## **特殊GUI**

**位于.\LineTheroy目录下，基于线性理论，主要面向圆形束流行波管优化**：

  - `Line_Gain_GUI_VCLASSIC`  
     📌 用途：常规单段行波管单频点计算（基础GUI）

  - `Line_Gain_GUI_VLargePower`  
    📌 用途：大增益单段行波管单频点计算（GUI模式）

  - `Line_Gain_GUI-MIXD_DUAL`  
    📌 用途：常规双段行波管单频点计算（基础GUI）

  - `Line_Gain_GUI-MIXD-SIMPLE_V3X`  
    📌 用途：常规多段行波管单频点计算（基础GUI）

## 🔧 实用小工具

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

## 🔧 TEST工具

**位于.\NolineTheroy\COMMON_THEROY目录下，主要面向圆形束流行波管优化：**

- `Noline_GAIN_V2TEST_COMMON`
  - 📌 用途：顾名思义，实例化`_TWT_CORE_NOLINE_COMPLEX_V2_MIX.py`：
  - 传一组`通用参数`转换非线性理论计算所需的参数，在对结果后处理，可以得到:
振幅演化，相位演化， 最终速度分布， 最终相位分布，电子相空间图，轴向功率图等结果
  -通过def para_SWP(isparaSWP, isSAVERESULT, inputP)方法进行自定义单频点的参数扫描：
        示例：scan_params = {
            1: [23000,23500,24000],  # 扫描电压V
            8: [1],  # 扫描填充效率
        }
    
-`Noline_GAIN_V2TEST-MIX_DriftedATTU`

- 📌 用途：实例化`_TWT_CORE_NOLINE_COMPLEX_V2_MIX.py`，
- 对包含相速度跳变段**有限距离截断**的分段行波管单频点工作状态进行分析
- 示例：
-   ```python ========================= 多段参数配置 =========================
    SEGMENTS = [
        {"len": 15, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "initial"},
        {"len": 15, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "attenuator"},
        {"len": 5, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "attenuator"},
        {"len": 5, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "O"},
        {"len": 5, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "O"},
        {"len": 5, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "O"},
    ]
-**Loss_attu = 20 表示衰减量**
-"type": "attenuator"**表示漂移段(不是SWS)**；"type": "O"表示相速度跳变段也可以配置不进行相速度跳变；"type": "initial"表示起始段

-`Noline_GAIN_V2TEST-MIX_DriftedATTU`

- 📌 用途：实例化`_TWT_CORE_NOLINE_COMPLEX_V2_MIX.py`，
- 对包含相速度跳变段**零距离截断**的分段行波管单频点工作状态进行分析
- 示例：
- ```python
  SEGMENTS = [
        {"len": 20, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "initial"},
        {"len": 20, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "attenuator"},
        {"len": 5, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "attenuator"},
        {"len": 5, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "O"},
        {"len": 5, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "O"},
        {"len": 5, "Vpc": 0.285, "p_SWS": 0.50, "Kc": 3.6, "f0_GHz": 211, "type": "O"},
    ]
 - "type": "attenuator"表示**起点为截断的SWS**；"type": "O"表示相速度跳变段也可以配置不进行相速度跳变；"type": "initial"表示起始段
   

**位于.\NolineTheroy\CicularBEAM_SPECIFED_THEROY目录下，只能用于圆形束流行波管优化：**


- 包括Noline_GAIN_VSUPERTEST_SINGLE、Noline_GAIN_VSUPERTEST-MIX_ATTUPVT、Noline_GAIN_VSUPERTEST-MIX_DriftedATTU
 
- 与.\NolineTheroy\COMMON_THEROY目录下的三个TEST工具功能分别对应：

- Noline_GAIN_VSUPERTEST_SINGLE用于**单段单频点**分析(主要用来检验慢波单元)

- Noline_GAIN_VSUPERTEST-MIX_ATTUPVT用于对包含相速度跳变段**零距离截断**的分段行波管单频点工作状态进行分析

- Noline_GAIN_VSUPERTEST-MIX_DriftedATTU用于对包含相速度跳变段**有限距离截断**的分段行波管单频点工作状态进行分析


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

以**位于.\NolineTheroy\COMMON_THEROY目录**为例：

- `_TWT_CORE_NOLINE_COMPLEX_V2_MIX/_TWT_CORE_NOLINE_COMPLEX_VSUPER_MIX`  
    📦 功能：非线性理论核心计算过程  
- `Noline_GAIN_MAINCALL_V2CORE_.py`  
    📦 功能：将`通用参数`转换为_TWT_CORE_NOLINE_COMPLEX_V2_MIX.py所需的:

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
       
- `Noline_GAIN_MAINCALL_V2CORE_MIX.py`
    📦 功能：将`通用参数`转换为_TWT_CORE_NOLINE_COMPLEX_V2_MIX.py所需的参数
  进行零距离截断多端均匀周期行波管分析
  
**位于.\NolineTheroy\SHEETBEAM_SPECIFED_THEROY目录**：

- _TWT_CORE_NOLINE_COMPLEX_VSHEETBEAM
包含带状束流行波管空间电荷场的核心求解过程


---

⚠️ **重要说明**  

1. 标有`CORE`前缀的函数为内部核心函数，建议开发者谨慎修改  
2. GUI工具中标注`PYQT`的表示基于PyQt框架开发  
3. 非线性理论模块仍在持续开发中，接口可能变动  
