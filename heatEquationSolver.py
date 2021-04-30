# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:56:40 2021

@author: xainx
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi


k = 1.0 #is this the boltzmann constant?


#analytic solution of 1D rod with ends set at T=0
def HeatEqu_1D_homog(x, t, f_x, L, count):
    
    def coefs(x, n, L, f_x):
        def integrand(x, n, f_x):
            return f_x(x)*np.sin((n*np.pi*x)/L)
        
        intresult = spi.quad(integrand, 0, L, args=(n, f_x))
        return 2*intresult[0] / L
    ret = 0
    for n in range(1, count+1):
        ret += coefs(x, n, L, f_x) * np.sin((n*np.pi*x) / L) * np.exp(-k * ((n*np.pi)/L)**2 * t)
    return ret

def HeatEqu_1D(x, t, f_x, L, T1, T2, count):
    
    def equlib(x):
        return T1 + (T2 - T1)*x/L
    
    def coefs(x, n, L, f_x):
        def integrand(x, n, f_x):
            return (f_x(x) - equlib(x))*np.sin((n*np.pi*x)/L)
        
        intresult = spi.quad(integrand, 0, L, args=(n, f_x))
        return 2*intresult[0] / L
    ret = 0
    for n in range(1, count+1):
        ret += equlib(x) + coefs(x, n, L, f_x) * np.sin((n*np.pi*x) / L) * np.exp(-k * ((n*np.pi)/L)**2 * t)
    return ret
    


t_0 = 0#s
t_f = 40#s
tcount = 40
Len = 2*np.pi #meters
poscount = 30
Temp1 = 0.1
Temp2 = 0

def InitialHeatFunction(x):
    return 20

xs, ts = np.meshgrid(np.linspace(0, Len, poscount), np.linspace(t_0, t_f, tcount))
heatsolngrid = HeatEqu_1D(xs, ts, InitialHeatFunction, Len, Temp1, Temp2, 100)
plt.figure()
plt.pcolormesh(heatsolngrid, cmap='hot')
plt.colorbar()
plt.xlabel('1D postition [0,L]')
plt.ylabel('time [t_0, t_f]')
plt.show()


    