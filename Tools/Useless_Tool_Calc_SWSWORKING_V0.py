import math as m
import Vpc_calc as V

U=23000;I=0.090;
f0=352*1e9*6.28;
yita=1.76e11;erb=8.85e-12;pi=3.1415926;
Ve=m.sqrt(2*yita*U);
t=0.026e-3;
w=1*t;
S=w*t*pi/4;
Vec=V.Vpc_calc(U);

fe=m.sqrt(yita*I/(S*Ve*erb));
V_0=Vec/(1+fe/f0)
print('The Working point Velocity of Vp is %.4f c' %(V_0)) 
print('The Difference between the electron velocity and the SWS phase velocity is smaller than %.4f c\n' %abs(V_0-Vec)) 
print('The Difference between the electron velocity and the SWS phase velocity has best Value of %.4f c\n' %abs(0.67*(V_0-Vec))) 

for U in range(22000,24000,100):
 Vpc=V.Vpc_calc(U)