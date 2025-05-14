'''
math.py

A module for various general use mathematical functions

Last updated July 2023

by Trevor Arp
All Rights Reserved
'''
import numpy as np

def symm_exp(x,y0,A,tau, x0):
	'''
	The Symmetric Exponential Function

	y = y0 + A*Exp(-|x-t0/tau|)

	Args:
		x : the independent variable
		y0 (float) : The offset
		A (float) : The Amplitude
		tau (float) : The 1/e value
		x0 (float) : The center x-value
	'''
	return y0 + A*np.exp(-np.abs((x-x0)/tau))
# end symm_exp

def power_law(x, A, g, y0):
	'''
	A Power Law Function

	y = A*x^g + y0

	Args:
		x : the independent variable
		A (float) : The Amplitude
		g (float) : gamma, the power law exponent
		y0 (float) : The offset
	'''
	return A*np.power(x,g) + y0
# end power_law

def gauss(x, A, x0, sigma):
	'''
	A one-dimensional Gaussian function given by

	y = A*Exp[-(x-x0)^2/(2*sigma^2)]

	Args:
		x : the independent variable
		A (float) : The Amplitude
		x0 (float) : The center x-value
		sigma (float) : The standard deviation
	'''
	return A*np.exp(-(x-x0)**2/(2*sigma**2))
# end gauss

def gauss2D(X, A, x0, sigmax, y0, sigmay):
	'''
	A two-dimensional Gaussian function given by

	f(x,y) = A*Exp[ -(x-x0)^2/(2*sigmax^2) - (y-y0)^2/(2*sigmay^2) ]

	Args:
		X : the independent variable, containing both x and y where x = X[0], y = X[1]
		A (float) : The Amplitude
		x0 (float) : The center x-value
		sigmax (float) : The standard deviation in the x direction
		sigmay (float) : The standard deviation in the y direction
	'''
	return A*np.exp(-(X[0]-x0)**2/(2*sigmax**2) - (X[1]-y0)**2/(2*sigmay**2))
# end gauss2D

def lorentzian(x, A, x0, G, y0):
	'''
	A basic normalized Lorentzian function
	               0.5*G
	y(x) = A*---------------------
	         (x-x0)^2 + (0.5*G)^2

	Args:
		x : the independent variable
		A (float) : The Amplitude
		x0 (float) : The center x-value
		G (float) : The full width at half maximum
		y0 (float) : The offset
	'''
	return A*0.5*G/((x-x0)**2 + (0.5*G)**2) + y0
# end lorentzian

def dydx(x, y):
	'''
	A simple numerical derivative using numpy.gradient, computes using 2nd order central differences.
    Certain data may require more thoughtful means of differentiation.

	Args:
		x : The independent variable
		y : The dependent variable, should have same length as x

	Returns:
		A simple numerical derivative, more complex differentiation may be required sometimes.
	'''
	return np.gradient(y)/np.gradient(x)
# end dydx

def divide_handel_zero(a,b):
	'''
	Divides a and b, if b is ever zero will return zero instead of throwing an Error.

	Args:
		a : Numerator (Array Line)
		b : Denominator (Array Line)

	Returns:
		The element wise quotient of a and b, handeling the divide by zero case.
	'''
	return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def HTheta(x):
	'''
	Heaviside Theta function.

	Args:
		x : the independent variable
	'''
	return 0.5*(np.sign(x) + 1)
# end HTheta

def Box(x, start, stop):
	'''
	Boxcar function.
	Composed of two HTheta functions, gives 1.0 from start to stop and is zero everywhere else

	Args:
		x : the independent variable
		start : The beginning of the non-zero domain
		stop : The end of the non-zero domain
	'''
	return HTheta(x-start) - HTheta(x-stop)
# end Box

def logistic_shift(x, k, yl, yr):
	'''
	A logistic function that continuously switches between two values yl and yr

	Smoother version of the step function.
	'''
	return yl + yr/(1+np.exp(-k*x))
# logistic_shift
