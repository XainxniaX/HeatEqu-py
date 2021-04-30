# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:26:19 2021

@author: xainx
"""

import numpy as np
import matplotlib.pyplot as plt
import heatEquationSim as sim
import heatEquationSolver as sol
import Utils as util
import time



def phi_uniform(x, y):
    return 10.0

def neuminx_const(t, y):
    return 0.0

def neuminy_const(t, x):
    return 0.0

def neumaxx_const(t, y):
    return 0.0

def neumaxy_const(t, x):
    return 0.0

K = 0.85

xmin = 0.0
xmax = 10.0
ymin = 0.0
ymax = 10.0
bounds = (xmin, xmax, ymin, ymax)

count = 100

condgrid = K*np.ones((count, count))
condgrid[:,0:5] = 0.01
condgrid[:,-6:] = 0.01

thetafinal = 2*np.pi*3
wavenumber = thetafinal/count

for y in range(5,count-6):
    for x in range(100):
        condgrid[x,y] = K*(np.sin(y*wavenumber)**2)

sim2D = sim.HeatSim_2D(bounds, count, phi_uniform, neuminx_const, neuminy_const, neumaxx_const, neumaxy_const, condgrid)

tRate = 500 #tsteps per simulated second
tfinal = 3.0
tcount = tfinal*tRate
deltat = tfinal / tcount

time0 = time.perf_counter()
soln = sim2D.RunTo(tfinal, dt=deltat, outputGrid=True, quiet=False) #CHANGE quiet TO True FOR SLIGHT SPEED INCREASE
time1 = time.perf_counter()

exectime = time1 - time0
netDt = np.sum(soln[0,:,:]) - np.sum(soln[-1,:,:])

resistancesX, resistancesY = sim2D.EnergyResistance_2D_Anisotropic(soln[-1,:,:])
netresx, netresy = np.sum(resistancesX), np.sum(resistancesY)
netres = np.sqrt(netresx**2 + netresy**2)



finalDEx, finalDEy = sim2D.NetEnergyFlux(soln[-1,:,:])
prevDEx, prevDEy = sim2D.NetEnergyFlux(soln[-2,:,:])
deltaDEx, deltaDEy = finalDEx - prevDEx, finalDEy - prevDEy
magdeltaDE = util.Mag2D(deltaDEx, deltaDEy)

print('\n\n:::INFO:::')
print(f'Execution time: {np.round(time1 - time0,3)} sec.')
print(f'Processed {tcount} timesteps over {np.round(sim2D.ElapsedTime,3)} simulated seconds.')
print(f'Process time per timestep: {np.round(exectime/tcount,3)} sec (for {count*count} cells)')
print(f'NET ΔT = {np.round(netDt,3)} K Σ(final T - initial T)')
print(f'Final state NET energy fluxs: ΔE_x = {np.round(finalDEx,6)} W, ΔE_y = {np.round(finalDEy,6)} W/')
print(f'Delta final NET energy fluxs: ΔΔE_x = {np.round(deltaDEx,6)} W/s, ΔΔE_y = {np.round(deltaDEy,6)} W/s')
print(f'Magnitude of delta energy flux: |ΔΔE| = {np.round(magdeltaDE,6)} W/s')
print(f'Computed NET "Energy Resistance" = {np.round(netres,3)} K/W')
print('\n:::ANISOTROPIC RESISTANCES:::')
print(f'NET R_x = {np.round(netresx, 3)} K/W, NET R_y = {np.round(netresy, 3)} K/W')
print('\n:::EXPECTED RESISTANCES:::')
print('TODO : should be 1/condgrid (1/conductance)\n\n')


plt.figure()
plt.title('CONDUCTANCE GRID')
plt.pcolormesh(condgrid, cmap='hot')
plt.colorbar()
plt.show()


for i in range(1, int(np.floor(tfinal))):
    plt.figure()
    plt.title(f'STATE AT t = {i} second(s)')
    plt.pcolormesh(soln[i*tRate-1,:,:], cmap='viridis')
    plt.colorbar()
    plt.show()
    
plt.figure()
plt.title(f'FINAL STATE, t = {np.round(sim2D.ElapsedTime, 3)}')
plt.pcolormesh(soln[-1,:,:], cmap='viridis')
plt.colorbar()
plt.show()

