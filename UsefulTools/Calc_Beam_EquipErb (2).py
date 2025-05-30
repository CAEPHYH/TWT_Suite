
import math as m
from Vpc_calc import Vpc_calc
import numpy as np 


# 物理常量
yita = 1.7588e11
erb = 8.854e-12  # 真空介电常数 (F/m)

# 用户输入参数

params=[0.3,23000,0.07,211,1.08,0.715,0.8,0.4,3.6,0.2,0.2]#parama=[I,U,Eff,f0_GHz,A_max,u_max,a_WG,b_WG,Kc,w_beam,t_beam]

I = params[0]
U = params[1]
Eff = params[2]
f0_GHz = params[3]
A_max=params[4]
u_max=params[5]
a_WG=params[6]
b_WG=params[7]
Kc=params[8]
w_beam=params[9]*1e-3
t_beam=params[10]*1e-3

A=np.linspace(0,A_max,10)
u=np.linspace(0,u_max,10)

class WaveGuide:
    def __init__(self, a, b):
        self.a = a*1e-3
        self.b = b*1e-3

# 创建结构体实例
my_WG = WaveGuide(a_WG, b_WG)


f0_Hz=f0_GHz * 1e9

C = m.pow(I * Kc / (4 * U), 1 / 3)

S = (w_beam * t_beam * np.pi) / 4
S_WG=my_WG.a*my_WG.b

numerator = I * np.sqrt(yita)
denominator = S * erb * np.sqrt(2 * U)
Wp = np.sqrt(numerator / denominator)


Vec = Vpc_calc(U)
omega=2 * np.pi * f0_Hz
beta_e = omega / (Vec * 299792458)
V1z=2*C*u*(Vec * 299792458)
P_out=C*I*U*2*(A)**2
Ez=np.sqrt(2*beta_e*Kc*P_out/S_WG)


epsilon_r = 1-Eff#
epsilon_rs=1-(Wp/yita)*(V1z/Ez)

# 结果输出
print(f"计算结果:Wp={Wp},V1z={V1z},Ez={Ez}")

print(f"相对介电常数 ε_re = {epsilon_r:.6f}")
print(f"相对介电常数 ε_res = {epsilon_rs}")