# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 19:17:53 2021

@author: Ryan Neuman
DIFF-TOOLS
"""
import numpy as np

def Fluxs(state, thermalCond): #joules
    grad = np.gradient(state)
    xflux = -thermalCond * grad[1]
    yflux = -thermalCond * grad[0]
    return xflux,yflux

def Mag2D(stateX, stateY):
    return np.sqrt(stateX**2 + stateY**2)

def DiscreteDeriv(xarr, yarr, xisLinspaced=True):    
    dx = -1
    for i in range(len(yarr)):
        if (i == 0): continue
        if (xisLinspaced):
            dx = xarr[1] - xarr[0]
        else:
            dx = np.diff(xarr)
        dy = np.diff(yarr)
        return dy / dx
    
def DerivThis(func, x, epsilon):
    return (func(x+epsilon) - func(x)) / epsilon

def Approxs_perc(val, target, percEpsilon): #epsilon is a percent difference
    if target == 0: return np.abs(target) <= percEpsilon
    return np.abs((target - val) / target) <= percEpsilon

def Approxs(val, target, epsilon): #epsilon is an absolute difference   
    return np.abs(target - val) <= epsilon


def Quadratic(a, b, c, tryCastFloat=True, EpsilonPerc=0.01): #epsilon perc=0.01 -> will cast any complex with a complex part within 1% of 0 to a real number
    r1 = (2.0+0.0j*c) / (-b - np.sqrt(b**2 - (4.0+0.0j)*a*c)) #alternate form of quad is more accurate for r1
    r2 = (-b - np.sqrt(b**2 - (4.0+0.0j)*a*c)) / (2.0+0.0j*a) #regular 
    if (tryCastFloat):
        if (Approxs_perc(r1.imag, 0, EpsilonPerc)):
            r1 = r1.real
        if (Approxs_perc(r2.imag, 0, EpsilonPerc)):
            r2 = r2.real
    return r1, r2


