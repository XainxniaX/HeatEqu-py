# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 03:17:31 2021

@author: xainx
An attempt at a simulation of the heat equation using fourier's law of thermal conductance
"""
import numpy as np
import matplotlib.pyplot as plt
import Utils as util


class HeatSimulation():
    ThermalCells = None
    CellPositions = None
    ThermalConductivity = 1.0
    ElapsedTime = 0.0
    CountedTimesteps = 0
    
    DT_SIZEFACTOR = 10
            
    def __init__(self, cells, positions, thermalCond = 1.0):
        self.ThermalCells = cells
        self.ThermalConductivity = thermalCond
        self.CellPositions = positions
        self.ElapsedTime = 0.0
        
    def StepForward(self, dt):        
        self.ElapsedTime += dt #do nothing but increment time because this should be overridden in child class

    def RunTo(self, steps, dt):
        for i in range(steps):
            self.StepForward(dt)
            self.CountedTimesteps += 1
            

class HeatSim_1D(HeatSimulation):
    xmin = 0.0
    xmax = 10.0
    count = 100
    
    ElapsedTime = 0.0
    
    F_x = None
    
    ThermalCells = None
    CellPositions = None
    
    ThermalConductivity = 1.0
    Area = 1.0
    
    '''
    idf_x: Initial thermal Distrobution Function of x
    '''
    def __init__(self, xmin, xmax, count, idf_x, thermalCond = 1.0, area = 1.0):
        self.xmin = xmin
        self.xmax = xmax
        self.ElapsedTime = 0.0
        self.ThermalConductivity = thermalCond
        self.Area = area
        self.count = count
        self.F_x = idf_x      
        self.CellPositions = np.linspace(xmin, xmax, count)
        self.ThermalCells = idf_x(self.CellPositions)        

    '''
    Steps time forward one step of (dt) while simulating all cells.
    checkConversion simply warns if net flux doesn't approx 0 within the percent given by conservationEpsilon,
    in this case changing the scale of dt may help
    '''
    def StepForward(self, dt, checkConservation = True, conservationEpsilon = 0.01):
        fluxs = -self.ThermalConductivity * np.gradient(self.ThermalCells)
        dQs = fluxs * self.Area * dt
        if checkConservation:
            if not util.Approxs_perc(sum(dQs), 0, conservationEpsilon):
                print('[Warn] Net flux != 0 : net dQ =', sum(dQs))
        self.ThermalCells += dQs #nice syntax sugar
        self.ElapsedTime += dt  
        self.CountedTimesteps += 1
    
    '''
    Runs time forward a set amount of time while simulating all cells.
    checkConversion simply warns if net flux doesn't approx 0 within the percent given by conservationEpsilon,
    in this case changing the scale of dt may help
    '''
    def RunTo(self, endTime, stepcount, checkConservation = True, conservationEpsilon = 0.01, outputGrid = False):
        dt = endTime / stepcount
        if outputGrid:
            ret = np.zeros((stepcount, self.count))
            for i in range(stepcount):
                 self.StepForward(dt, checkConservation, conservationEpsilon)
                 for j in range(self.count):
                     ret[i, j] = self.ThermalCells[j]
            return ret
        else:
            for i in range(stepcount):
                self.StepForward(dt, checkConservation, conservationEpsilon)
    
    

class HeatSim_1D_Closed(HeatSimulation):
    xmin = 0.0
    xmax = 1.0
    count = 10
    
    Dx = (xmax - xmin) / count
    
    PhiFunc = None #idf_x in HeatSim_1D -- Initial Heat Distrobution condition
    
    NeuMin = None #neumann condition for xmin (dT/dx @ xmin = NeuMin) rate of heat flow in/out of xmin-end
    NeuMax = None #neumann condition for xmax (dT/dx @ xmax = NeuMax) rate of heat flow in/out of xmax-end
    
    Forcing = None #outside input F(x, t) applied to 1D bar
    
    def __init__(self, xmin, xmax, count, phi, neumin, neumax, forcing, thermalCond = 1.0):
        self.ThermalConductivity = thermalCond
        self.xmin = xmin
        self.xmax = xmax
        self.count = count
        self.PhiFunc = phi
        self.NeuMin = neumin
        self.NeuMax = neumax
        self.Forcing = forcing
        self.Dx = (xmax - xmin) / count
        self.CellPositions = np.linspace(xmin, xmax, count)
        self.ThermalCells = self.PhiFunc(self.CellPositions)
        
    def StepForward(self, dt, checkConservation = True, conservationEpsilon = 0.01):
        #T_(i,j+1) = T_(i,j) - kdt/dx^2 * (T_(i+1,j) - 2T_(i,j) + T_(i-1,j))
        newtemps = np.zeros(self.count)
        newtemps[0] = self.NeuMin(self.ElapsedTime + dt) + self.ThermalCells[0]
        newtemps[self.count - 1] = self.NeuMax(self.ElapsedTime + dt) + self.ThermalCells[self.count-1]
        for i in range(1, self.count-1):
            newtemps[i] = self.ThermalCells[i] + ((self.ThermalConductivity * dt)/(self.Dx**2)) * (self.ThermalCells[i+1] - 2*self.ThermalCells[i] + self.ThermalCells[i-1])            
        if checkConservation:
            dT = sum(newtemps) - sum(self.ThermalCells)
            if not util.Approxs_perc(dT, 0, conservationEpsilon): 
                print('[Warn] Energy has not been conserved in the system (this could be expected) : net dT =', dT)
        self.ThermalCells = newtemps
        self.ElapsedTime += dt
        self.CountedTimesteps += 1
        
    
    def RunTo(self, t_final, dt='Auto', outputGrid = False, treat_t_final_asAbsolute = False, checkConservation = True, conservationEpsilon = 0.01):
        if (type(dt) is str):
            if dt == 'Auto':
                #find stepcount which will make |kdt/dx^2| > 1 with a margin of self.DT_SIZEFACTOR i.e. |kdt/dx^2| = 1/self.DT_SIZEFACTOR
                dt = np.abs((self.Dx**2)/(self.DT_SIZEFACTOR * self.ThermalConductivity))                        
        stepcount = 10
        if treat_t_final_asAbsolute:
            stepcount = np.int(np.round((t_final - self.ElapsedTime) / dt))            
        else:
            stepcount = np.int(np.round(t_final / dt))                 
        if outputGrid:
            ret = np.zeros((stepcount, self.count))
            for j in range(stepcount):
                self.StepForward(dt, checkConservation, conservationEpsilon)
                for i in range(self.count):
                    ret[j, i] = self.ThermalCells[i]
            return ret
        else:
            for j in range(stepcount):
                self.StepForward(dt, checkConservation, conservationEpsilon)
                
    
    def FlowField(self, grid):
        gradx, gradt = np.gradient(grid)
        grad2x = np.gradient(gradx)
        grad2t = np.gradient(gradt)
        xs, ts = np.meshgrid(np.linspace(self.xmin, self.xmax, self.count), np.linspace(0, self.ElapsedTime, self.CountedTimesteps))
        return [xs, ts, gradx, gradt, grad2x, grad2t]
    

class HeatSim_2D(HeatSimulation):
    '''        
        2D HEAT SIMULATION ON SQUARE/RECTANGULAR GRID WITH PER-CELL THERMAL CONDUCTIVITY
        Parameters
        ----------
        bounds : 4-Tuple of floats : (xmin, xmax, ymin, ymax)
            Defines physical dimensions of 2D-Square grid.
        count : int
            Count of cells in a single dimension.
        phi : function of trace : fun(x, y) => float T
            Function which gives the initial state of temperature in the entire (count x count) grid.
        neuminX : function of trace : fun(t, y) => float dT/dt
            Neumann condition for boundry: left side (x-min). (dT/dt = neuminx @ x=xmin)
        neuminY : function of trace : fun(t, x) => float dT/dt
            Neumann condition for boundry: bottom side (y-min). (dT/dt = neuminy @ y=ymin)
        neumaxX : function of trace : fun(t, y) => float dT/dt
            Neumann condition for boundry: right side (x-max). (dT/dt = neumaxx @ x=xmax)
        neumaxY : function of trace : fun(t, x) => float dT/dt
            Neumann condition for boundry: top side (y-max). (dT/dt = neumaxy @ y=ymax)
        conductivityGrid : 2D (count x count) array/grid of floats : thermal conductivity @ grid positions
            This specifies the individual thermal conductivity of each cell.
        '''
    
    Xmin = 0.0
    Xmax = 0.0
    Ymin = 0.0
    Ymax = 0.0
    CellCount = 10 #count of cells in single dim
    TotalCells = 100 #total count of cells
    Dx = (Xmax - Xmin) / CellCount

    ConductivityGrid = None #each cell's thermal conductivity
    Phi = None #initial heat distrobution
    NeuMinX = None
    NeuMaxX = None
    NeuMinY = None
    NeuMaxY = None

    def __init__(self, bounds, count, phi, neuminX, neuminY, neumaxX, neumaxY, conductivityGrid):
        '''        
        2D HEAT SIMULATION ON SQUARE/RECTANGULAR GRID WITH PER-CELL THERMAL CONDUCTIVITY
        Parameters
        ----------
        bounds : 4-Tuple of floats : (xmin, xmax, ymin, ymax)
            Defines physical dimensions of 2D-Square grid.
        count : int
            Count of cells in a single dimension.
        phi : function of trace : fun(x, y) => float T
            Function which gives the initial state of temperature in the entire (count x count) grid.
        neuminX : function of trace : fun(t, y) => float dT/dt
            Neumann condition for boundry: left side (x-min). (dT/dt = neuminx @ x=xmin)
        neuminY : function of trace : fun(t, x) => float dT/dt
            Neumann condition for boundry: bottom side (y-min). (dT/dt = neuminy @ y=ymin)
        neumaxX : function of trace : fun(t, y) => float dT/dt
            Neumann condition for boundry: right side (x-max). (dT/dt = neumaxx @ x=xmax)
        neumaxY : function of trace : fun(t, x) => float dT/dt
            Neumann condition for boundry: top side (y-max). (dT/dt = neumaxy @ y=ymax)
        conductivityGrid : 2D (count x count) array/grid of floats : thermal conductivity @ grid positions
            This specifies the individual thermal conductivity of each cell.
        '''
        self.Xmin = bounds[0]
        self.Xmax = bounds[1]
        self.Ymin = bounds[2]
        self.Ymax = bounds[3]
        self.CellCount = count
        self.Phi = phi
        self.NeuMinX = neuminX
        self.NeuMaxX = neumaxX
        self.NeuMinY = neuminY
        self.NeuMaxY = neumaxY
        self.ConductivityGrid = conductivityGrid
        self.Dx = (self.Xmax - self.Xmin) / self.CellCount
        self.CellPositions = np.meshgrid(np.linspace(self.Xmin, self.Xmax, self.CellCount), np.linspace(self.Ymin, self.Ymax, self.CellCount))
        self.ThermalCells = np.zeros((self.CellCount, self.CellCount))
        for y in range(self.CellCount):
            for x in range(self.CellCount):
                self.ThermalCells[x, y] = self.Phi(self.CellPositions[0][x, y], self.CellPositions[1][x, y]) #init thermal cells
    
    def StepForward(self, dt, quiet = False):
        newtemps = np.zeros(self.ThermalCells.shape)
        
        newtemps[0, :] = self.NeuMinX(self.ElapsedTime + dt, self.CellPositions[1][0, :])
        newtemps[-1,:] = self.NeuMaxX(self.ElapsedTime + dt, self.CellPositions[1][-1,:])
        newtemps[:, 0] = self.NeuMinY(self.ElapsedTime + dt, self.CellPositions[0][:, 0])
        newtemps[:,-1] = self.NeuMaxY(self.ElapsedTime + dt, self.CellPositions[0][:,-1])
        
        for k in range(1, self.CellCount-1):
            for j in range(1, self.CellCount-1):
                newtemps[j,k] = self.ThermalCells[j,k] + (self.ConductivityGrid[j,k]*dt)/(self.Dx**2)*\
                    (self.ThermalCells[j+1,k] + self.ThermalCells[j-1,k] + self.ThermalCells[j,k+1] + self.ThermalCells[j,k-1] - 4*self.ThermalCells[j,k])
        if not quiet:
            print(f'Step {self.CountedTimesteps}: t = {np.round(self.ElapsedTime + dt, 3)} sec, NET ΔT = {np.round(np.sum(newtemps) - np.sum(self.ThermalCells),3)}K')
        
        self.ThermalCells = newtemps
        self.ElapsedTime += dt
        self.CountedTimesteps += 1


    def RunTo(self, t_final, dt='Auto', outputGrid=False, treat_t_final_asAbsolute=False, quiet=False):
        if (type(dt) is str):
            if dt == 'Auto':
                #find stepcount which will make |kdt/dx^2| > 1 with a margin of self.DT_SIZEFACTOR i.e. |kdt/dx^2| = 1/self.DT_SIZEFACTOR
                dt = np.abs((self.Dx**2)/(self.DT_SIZEFACTOR * self.ThermalConductivity))                        
        stepcount = 10 #arbitrary default value bc python be like that
        if treat_t_final_asAbsolute:
            stepcount = np.int(np.round((t_final - self.ElapsedTime) / dt))            
        else:
            stepcount = np.int(np.round(t_final / dt)) 
        if outputGrid:
            ret = np.zeros((stepcount, self.CellCount, self.CellCount))            
            for i in range(stepcount):
                ret[i,:,:] = self.ThermalCells[:,:]
                self.StepForward(dt, quiet)                
            return ret
        else:
            for i in range(stepcount):
                self.StepForward(dt, quiet)
                
    def Fluxs(self, state):
       return util.Fluxs(state, self.ThermalConductivity)
   
    def GradMag_2D(self, state):
        grad = np.gradient(state)
        mag = np.sqrt(grad[0][:,:]**2 + grad[1][:,:]**2)
        return mag
          
   
    def NetEnergyFlux(self, state):
        fluxx, fluxy = self.Fluxs(state)
        netfluxx = np.sum(fluxx)
        netfluxy = np.sum(fluxy)
        return netfluxx, netfluxy
    
    def EnergyResistance_2D_Anisotropic(self, state):
        grad = np.gradient(state)
        avgdeltaTx = grad[1]
        avgdeltaTy = grad[0]
        fluxx, fluxy = self.Fluxs(state)
        efluxmagx, efluxmagy = np.sum(fluxx), np.sum(fluxy)
        resx = avgdeltaTx / efluxmagx
        resy = avgdeltaTy / efluxmagy
        return resx, resy
    
    
    def EnergyResistance_2D(self, state):
        avgdeltaT = self.GradMag_2D(state)
        fluxx, fluxy = self.Fluxs(state)
        efluxmag = util.Mag2D(fluxx, fluxy)
        res = avgdeltaT / efluxmag
        return res
    
    def NetEnergyResistance_2D(self, state):
        return np.sum(self.EnergyResistance_2D(state))
        
        



    



        
        
        
'''
def f(x):
    return x

heatsim1 = HeatSim_1D(0.0, 1.0, 10, f)
phasegrid = heatsim1.RunTo(5.0, 20, outputGrid=True)

plt.figure()
plt.pcolormesh(phasegrid, cmap='hot')
plt.xlabel('1D position')
plt.ylabel('time steps')
plt.colorbar()
plt.show()
'''
'''

def phi(x):
    return x**3+5

def neuMin_zero(t):
    return 1.0

def neuMax_zero(t):
    return -2.0

def forceFunc(t):
    return 0.0

heatsim2 = HeatSim_1D_Closed(0.0, 10.0, 30, phi, neuMin_zero, neuMax_zero, forceFunc)
xvstGrid = heatsim2.RunTo(10.0, dt=0.05, outputGrid=True, treat_t_final_asAbsolute=True)
'''
'''
L = 1000
xvstGrid = np.zeros((L, 30))
for j in range(L):
    for i in range(30):
        xvstGrid[j, i] = heatsim2.ThermalCells[i]
    heatsim2.StepForward2(0.05)
'''    
'''

plt.figure()
plt.pcolormesh(xvstGrid, cmap='hot')
plt.xlabel('1D position (x cell index = i)')
plt.ylabel('time index = j')
plt.colorbar()


fields = heatsim2.FlowField(xvstGrid)

plt.figure()
plt.pcolormesh(fields[2], cmap='hot')
plt.colorbar()


plt.figure()
plt.pcolormesh(fields[3], cmap='hot')
plt.colorbar()


plt.figure()
plt.pcolormesh(fields[4][0], cmap='hot')
plt.colorbar()


plt.figure()
plt.pcolormesh(fields[4][1], cmap='hot')
plt.colorbar()


plt.figure()
plt.pcolormesh(fields[5][0], cmap='hot')
plt.colorbar()


plt.figure()
plt.pcolormesh(fields[5][1], cmap='hot')
plt.colorbar()



plt.figure()
plt.streamplot(fields[0], fields[1], fields[2], fields[3])


#print(fields[5])


plt.figure()
plt.plot(range(30), xvstGrid[0,:], '-b')
plt.plot(range(30), xvstGrid[-1,:], '-k')

plt.show()
'''
        
        
        