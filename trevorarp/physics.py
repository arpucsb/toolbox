'''
physics.py

A module for various physical quantities and functions

Last updated February 2020

by Trevor Arp

Includes the following Physical Constants:

# Fundamental Constants, SI Values

e = 1.602176634e-19 # C (Elementary charge)

c = 299792458 # m/s (speed of light)

h = 6.626070040e-34 #J*s

hbar = 1.05457180e-34 #J*s

Navogadro = 6.022140857e23 # 1/mol (Avogadro's Number)

kb = 1.38064852e-23 # J K−1 (Boltzmann's constant)

# Electromagnetic Constants

mu0 = 4*np.pi*1e-7 # N/A^2

epsilon0 =8.854187817e-12 # F/m

phi0 = 2.067833831e-15 # Wb (Magnetic Flux Quantum)

G0 = 7.748091731e-5 #S (Conductance Quantum)

J_eV = 1.6021766208e-19 # J/eV


# Particle

me = 9.10938356e-31 # kg (electron mass)

mp = 1.672621898e-27 # kg (proton mass)

alphaFS = 7.2973525664e-3 # Electromagnetic Fine Structure constant

Rinf = 10973731.568508 # 1/m (Rydberg Constant)

amu = 1.660539040e-27 # kg (atomic mass unit)

# Physical Constants in other units

kb_eV = 8.6173324e-5 # eV/K

h_eV = 4.135667662e-15 # eV s

hbar_eV = 6.582119514e-16 # eV s

c_nm = 2.99792458e17 # nm/s (speed of light)

# Graphene Constants

G_vf = 1.0e6 # m/s

G_a = 0.142e-9 # m (Graphene lattice constant)

G_Ac = 3*np.sqrt(3)*(G_a**2)/2 # nm^2 (Unit cell area)

'''
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.signal import convolve

# Fundamental Constants, SI Values
e = 1.602176634e-19 # C (Elementary charge)
c = 299792458 # m/s (speed of light)
h = 6.626070040e-34 #J*s
hbar = 1.05457180e-34 #J*s
Navogadro = 6.022140857e23 # 1/mol (Avagadro's Number)
kb = 1.38064852e-23 # J K−1 (Boltzmann's constant)

# Electromagnetic Constants
mu0 = 4*np.pi*1e-7 # N/A^2
epsilon0 =8.854187817e-12 # F/m
phi0 = 2.067833831e-15 # Wb (Magnetic Flux Quantum)
G0 = 7.748091731e-5 #S (Conductance Quantum)
J_eV = 1.6021766208e-19 # J/eV

# Particle
me = 9.10938356e-31 # kg (electron mass)
mp = 1.672621898e-27 # kg (proton mass)
alphaFS = 7.2973525664e-3 # Electromagnetic Fine Structure constant
Rinf = 10973731.568508 # 1/m (Rydberg Constant)
amu = 1.660539040e-27 # kg (atomic mass unit)

# Physical Constants in other units
kb_eV = 8.6173324e-5 # eV/K
h_eV = 4.135667662e-15 # eV s
hbar_eV = 6.582119514e-16 # eV s
c_nm = 2.99792458e17 # nm/s (speed of light)

# Graphene Constants
G_vf = 1.0e6 # m/s
G_a = 0.142e-9 # m (Graphene lattice constant)
G_Ac = 3*np.sqrt(3)*(G_a**2)/2 # nm^2 (Unit cell area)

def f_FD(E, T, E0=0.0):
    '''
    The Fermi-Dirac distribution as a function of energy and temperature.

    Args:
        E : The energy, in eV
        T : The temperature of the distribution, in Kelvin
        E0 (float, optional) : The zero point of the distribution

    Returns:
        The Fermi function at the given energy and temperature
    '''
    return 1/(np.exp((E-E0)/(kb_eV*T)) + 1)
#

def f_BE(E, T, E0=0.0):
    '''
    The Bose-Einstein distribution as a function of energy and temperature.

    Args:
        E : The energy, in eV
        T : The temperature of the distribution, in Kelvin
        E0 (float, optional) : The zero point of the distribution

    Returns:
        The Bose-Einstein distribution function at the given energy and temperature
    '''
    return 1/(np.exp((E-E0)/(kb_eV*T)) - 1)
#


def DOS_Graphene(E, E0=0.0):
    '''
    The Density of States for Graphene
    '''
    return (2/(np.pi*G_vf)**2)*np.abs(E-E0)
#

def magnetic_dipole(r, m):
    '''
    Formula for the magnetic field -- in Tesla -- due to a dipole @ r (in meters)

    Args:
        r: Three component vector of distance to the dipole
        m: Three component vector of magnetic moment of the dipole

    Returns:
        Three component magnetic field (in Tesla)
    '''
    mu0 = 1.256637062e-6 # tesla * ampere / meter
    rhat = np.divide(r, np.linalg.norm(r))
    t1 = np.multiply(3*np.dot(rhat,m), rhat)
    r3 = np.linalg.norm(r)**3
    num = np.subtract(t1, m)
    #numz = num[2]
    return mu0/(4*np.pi) * (num/r3)
#

def field_above_dipoles(m, density, perimeter, height, xrng, yrng, simdensity=10, numx=50, numy=50, holes=None, retpolygon=False):
    """
    Calculate the magnetic field above a sheet of dipoles all with the same magnetic moment vector

    Unless otherwise noted use real units, i.e. 1 micron = 1e-6 m
    Args:
        m: Magnetic moment of an individual dipole (Three component Numpy Array)
        density: The density of dipoles used to normalize the total magnetic moment
        perimeter: A list of points defining a polygon that is the perimeter of the magnetization
        height: The height above the geometry to consider
        xrng: Range (xmin, xmax) to simulate in the geometry.
        yrng: Range (ymin, ymax) to simulate in the geometry.
        simdensity # The spacing of computed dipoles per micron, values will be normalized to the total real density,
            larger results in a more accurate simulation but at the cost of computing time.
        numx: The number of points in X to calculate
        numy: The number of points in X to calculate
        holes: List of objects with the same requirements as perimeter that creates a holes(s) in the polygon. Shee shapely.polygon for details
    Returns:
        X, Y, Bx, By, Bz where each are meshgrids of size (numx, numy). Field is in units of Tesla
    """
    # Define the device as a polygon
    polygon = Polygon(perimeter, holes)
    area = (polygon.area)
    x_, y_ = polygon.exterior.xy

    ## Simulate by assuming dipoles length/2n on each side of grid point add up to form one dipole.
    ## Fine assumption as long as grid spacing << distance to surface
    ## Normalize signal to total number of dipoles
    m_total = np.linalg.norm(m)
    mhat = m / m_total
    total_moment = density * m_total * area

    # Grid of dipoles to simulate
    x = np.arange(np.min(x_), np.max(x_) + 1e-9, 1e-6 / simdensity)
    y = np.arange(np.min(y_), np.max(y_) + 1e-9, 1e-6 / simdensity)
    d = list()
    for i in range(len(y)):
        for j in range(len(x)):
            if polygon.contains(Point(x[j], y[i])):
                d.append([x[j], y[i], 0])
    d = np.array(d)
    moment_per_dipole = total_moment / len(d)
    m_normed = moment_per_dipole * mhat # Normalized magnetic moment

    # Grid of points to calculate
    box_grid_x = np.linspace(xrng[0], xrng[1], numx)
    box_grid_y = np.linspace(yrng[0], yrng[1], numy)
    X, Y = np.meshgrid(box_grid_x, box_grid_y)

    # Perform the computation
    f = np.zeros((numy, numx, 3))
    for i in range(numy):
        for j in range(numx):
            dist = np.subtract([X[i, j], Y[i, j], height], d)
            l = [magnetic_dipole(r, m_normed) for r in dist]
            f[i, j, :] = [sum(i) for i in zip(*l)] #np.sum([magnetic_dipole(r, m_normed) for r in dist])
            # f[i, j,:] = np.sum([magnetic_dipole(r, m_normed) for r in dist], axis=1)
    return X, Y, f[:,:,0], f[:,:,1], f[:,:,2]

def field_above_dipoles_conv(m, density, perimeter, height, xrng, yrng, holes=None, simdensity=10, numx=200, numy=200, fulloutput=True):
    """
    Calculate the magnetic field above a sheet of dipoles all with the same magnetic moment vector.
    Use a convolution based algorithm to be more efficient.

    Unless otherwise noted use real units, i.e. 1 micron = 1e-6 m
    Args:
        m: Magnetic moment of an individual dipole (Three component Numpy Array)
        density: The density of dipoles used to normalize the total magnetic moment
        perimeter: A list of points defining a polygon that is the perimeter of the magnetization
        height: The height above the geometry to consider
        xrng: Range (xmin, xmax) to simulate in the geometry.
        yrng: Range (ymin, ymax) to simulate in the geometry.
        simdensity: Deprecated parameter, included for backwards compatability
        numx: The number of points in X to calculate
        numy: The number of points in X to calculate
        holes: List of objects with the same requirements as perimeter that creates a holes(s) in the polygon. Shee shapely.polygon for details
    Returns:
        If fulloutput:
        X, Y, Bx, By, Bz where each are meshgrids of size (numx, numy). Field is in units of Tesla
        else: # calculate only the z-component
        X, Y,  Bz where each are meshgrids of size (numx, numy). Field is in units of Tesla
    """
    # Define the device as a polygon
    polygon = Polygon(perimeter, holes)
    area = (polygon.area)

    ## Simulate by assuming dipoles length/2n on each side of grid point add up to form one dipole.
    ## Fine assumption as long as grid spacing << distance to surface
    ## Normalize signal to total number of dipoles
    m_total = np.linalg.norm(m)
    mhat = m / m_total
    total_moment = density * m_total * area

    box_grid_x = np.linspace(xrng[0], xrng[1], numx)
    box_grid_y = np.linspace(yrng[0], yrng[1], numy)
    X, Y = np.meshgrid(box_grid_x, box_grid_y)

    # Generate the mesh of sources
    magnetization = np.zeros((numy, numx))
    Ndipoles = 0
    for i in range(numy):
        for j in range(numx):
            if polygon.contains(Point(X[i,j], Y[i,j])):
                magnetization[i,j] = 1.0
                Ndipoles += 1
    moment_per_dipole = total_moment / Ndipoles
    m_normed = moment_per_dipole * mhat  # Normalized magnetic moment
    # Normalize by the total density

    # Generate the field of a single moment
    df = np.zeros((numy, numx, 3))
    for i in range(numy):
        for j in range(numx):
            r = np.subtract([X[i,j], Y[i,j], height], [(xrng[1]+xrng[0])/2,(yrng[1]+yrng[0])/2,0])
            df[i,j,:] = magnetic_dipole(r, m_normed)

    # Convolve them efficiently
    if fulloutput:
        fx = convolve(magnetization, df[:, :, 0], mode='same')
        fy = convolve(magnetization, df[:, :, 1], mode='same')
        fz = convolve(magnetization, df[:, :, 2], mode='same')
        return X, Y, fx, fy, fz
        # return X, Y, df[:,:,0], df[:,:,1], df[:,:,2] # For debugging
    else:
        fz = convolve(magnetization, df[:, :, 2], mode='same')
        return X, Y, fz