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

'''
The Density of States for Graphene
'''
def DOS_Graphene(E, E0=0.0):
    return (2/(np.pi*G_vf)**2)*np.abs(E-E0)
#
