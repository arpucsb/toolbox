'''
A module for nanosquid specific code.
'''
import numpy as np

def cal_attocube_to_um(x=None, y=None, z=None, xmax=40, ymax=40, zmax=20, vmax=7.5, centerxy=True):
    '''
    Calibrate the voltage sent to the attocubes into position in microns based on the extension
    at maximum voltage ouput.

    Args:
        x : The X axis values to calibrate
        y : The Y axis values to calibrate
        z : The Z axis values to calibrate
        xmax : The X axis maximum extension
        ymax : The Y axis maximum extension
        vmax : The voltage to apply for maximum extension
        cetnerxy: if True will place (0,0) in center of voltage range.
    '''
    ret = []
    if x is not None:
        x = xmax*x/vmax
        if centerxy:
            x = x - xmax/2
        ret.append(x)
    if y is not None:
        y = ymax*y/vmax
        if centerxy:
            y = y - ymax/2
        ret.append(y)
    if z is not None:
        z = zmax*z/vmax
        ret.append(z)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret
#

def convertND(p0, n0, C=3.18e15):
    '''
    Convert from gate voltage units (n0, p0) into electron denisty Ne (10^-12 cm^-2) and displacement
    field (V/nm)

    C is Geometric capacitance which Haoxin measured in m^-2 V^-1
    '''
    ee = 1.602176634e-19  # C (Elementary charge)
    epsilon0 = 8.854187817e-12  # F/m
    k = C * ee / (2 * epsilon0) * 1e-9 # convert to nm
    N = n0 * C * 1.0e-16  # 10^-12 cm^-2
    D = p0 * k  # V/nm
    return N, D

def cal_compressability(M, Minf=0.201524, M0=0.301748, c=1.9383e-4):
    '''
    Calibrate the capacitance data into compressability

    c in units of eV/a0^2 # Has been converted to eV and unit cell area
    '''
    kappa = (1/(2*c))*(M-Minf)/(M0-Minf)
    return kappa