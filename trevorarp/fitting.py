'''
fitting.py

A module for fitting data

Last updated August 2020

by Trevor Arp
All Rights Reserved
'''
import numpy as np
import warnings

from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, griddata
from scipy.optimize import minimize_scalar

from trevorarp.math import gauss, power_law, symm_exp, lorentzian
from trevorarp.processing import lowpass

def generic_fit(x, y, p0, fitfunc, warn=True, maxfev=2000):
    '''
    A generic wrapper for curve_fit, accepts data (x,y), a set of initial parameters p0 and a function
    to fit to.

    Args:
        x : The independent variable
        y : The dependent variable
        p0 : The initial parameters for the fitting function
        fitfunc : The function to fit
        warn (bool, optional) : If True (default) will print error message when fitting
        maxfev (int, optional) : The 'maxfev' parameter of scipy.optimize.curve_fit

    Returns:
        p, perr

            p - The fitting parameters that optimize the fit, or the initial parameters if the fit failed.

            perr - Estimated error in the parameters, or zero if the fit failed.
    '''
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, plconv = curve_fit(fitfunc, x, y, p0=p0, maxfev=maxfev)
            perr = np.sqrt(np.abs(np.diag(plconv)))
    except Exception as e:
        p = p0
        perr = [0 for _ in range(len(p0))]
        if warn:
            print("Error fitting.generic_fit: Could not fit, parameters set to initial")
            print(str(e))
    return p, perr
# end generic_fit

def leastsq_2D_fit(x, y, data, p0, fitfunc):
    '''
    A general fitting routine for two-dimensional data using scipy.optimize.leastsq

    Args:
        x : the row variable of the data.
        y : the column variable of the data.
        data : a 2D array as a function of x and y
        p0 : the initial guesses of the parameters
        fitfunc : the function to minimize, takes params func(x, y, \*p) and returns a 2D array
            in the same format as the data.

    Returns:
        popt, pcov

            popt - The solution (or the result of the last iteration of an unsuccessful call.)

            pcov - Estimate of the covariance matrix from the Jacobian around the solution. The diagonals estimate the variance of the parameters.
    '''
    def lsq_func(params):
        r = data - fitfunc(x, y, *params)
        return r.flatten()
    #
    retval = leastsq(lsq_func, p0, full_output=1)
    return retval[0], retval[1]
# end leastsq_2D_fit

def power_law_fit(x, y, p0=None, warn=True):
    '''
    Fits data to a symmetric exponential function defined by math.power_law

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent variable
        p0 (optional): The initial parameters for the fitting function. If None (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p, perr

            p - The fitting parameters that optimize the fit, or the initial parameters if the fit failed.

            perr - Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.power_law_fit: X and Y data must have the same length")
        return
    if p0 is None:
        b, a = np.polyfit(x,y,1.0)
        g = 1.0
        p0=(b, g, 0.0)
    return generic_fit(x, y, p0, power_law, warn=warn)
# end power_law_fit

def symm_exponential_fit(x, y, p0=None, warn=True):
    '''
    Fits data to a symmetric exponential function defined by math.symm_exp

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent variable
        p0 (optional): The initial parameters for the fitting function. If None (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p, perr

            p - The fitting parameters that optimize the fit, or the initial parameters if the fit failed.

            perr - Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.symm_exponential_fit: X and Y data must have the same length")
        return
    if p0 is None:
        a = np.mean(y)
        b = np.mean(y[int(9*l/20):int(11*l/20)]) - a
        t = (x[l-1]-x[0])/4.0
        p0=(a, b, t, 0.0)
    return generic_fit(x, y, p0, symm_exp, warn=warn)
# end symm_exponential_fit

def gauss_fit(x, y, p0=None, warn=True):
    '''
    Fits data to a gaussian function defined by math.gauss

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent variable
        p0 (optional): The initial parameters for the fitting function. If None (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p, perr

            p - The fitting parameters that optimize the fit, or the initial parameters if the fit failed.

            perr - Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.gauss_fit: X and Y data must have the same length")
        return
    if p0 is None:
        a = np.max(y)
        sigma = 1.0
        x0 = x[int(len(x)/2)]
        p0=(a, sigma, x0)
    return generic_fit(x, y, p0, gauss, warn=warn)
# end gauss_fit

def lorentzian_fit(x, y, p0=None, warn=True):
    '''
    Fits data to a Lorentzian function defined by math.lorentzian

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent variable
        p0 (optional): The initial parameters for the fitting function. If None (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p, perr

            p - The fitting parameters that optimize the fit, or the initial parameters if the fit failed.

            perr - Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.gauss_fit: X and Y data must have the same length")
        return
    if p0 is None:
        a = np.max(y) - np.mean(y)
        x0 = np.mean(x)
        y0 = np.mean(y)
        p0=(a, x0, 0.5, y0)
    return generic_fit(x, y, p0, lorentzian, warn=warn)
# end gauss_fit

def interp_maximum(x, y, kind='cubic', warn=True):
	'''
	Estimate the maximum coordinates of some data from an interpolation of the data. Works best on sparse
	data following a slow curve, where the maximum is clearly between points.

	Args:
		x : The x-values of the data to interpolate
		y : The y-values of the data to interpolate
		kind (str, optional): 'cubic' : the kind of intepolation to use, default is 'cubic'.
		warn (bool, optional): If true (default) will print a warning if the optimization fails.

	Returns:
        X0, Y0

             X0 - the estimated maximum location

             Y0 - the estimated maximum value.
	'''
	ifunc = interp1d(x, -1.0*y, kind=kind)
	ix = np.argmin(x)
	res = minimize_scalar(ifunc, x[ix], method='bounded', bounds=(np.min(x), np.max(x)))
	if res.success:
		x0 = res.x
	else:
		x0 = x[ix]
		if warn:
			print('Warning interp_maximum: Optimization did not exit successfully')
	return x0, -1.0*ifunc(x0)
# end interp_maximum

def reduced_chi2(data, fit, Nparams):
    '''
    A generic function of calculating the reduced chi squared value of a fit, to a functions

    The reduced chi squared is the sum of variations per degree of freedom
             1
    chi^2 = --- sum[ (data - fit)^2 / stdev^2 ]
            N-m

    Rule of thumb for goodness of fit:
    chi2 >> 1, = bad fit
    chi2 > 1, fit not fully capturing data
    chi2 = 1, data and fit match within error variance
    chi2 < 1, overfitting

    Args:
        data : the data to calculate
        fit : the corresponding values of the fitted function
        Nparams : the number of parameters that go into the fit
    '''
    N = data.size
    var = np.var(data)
    chi2 = np.sum((data-fit)**2 / var)
    return chi2/(N-Nparams)
# end reduced_chi2

'''
Average fit error
chi =  sum( (obs-expected)^2/expected^2 )
'''
def avg_error(obs, expected):
    N = len(obs)
    s = 0.0
    for i in range(N):
        s += ((obs[i] - expected[i]) / expected[i] )**2
    s = np.sqrt(s/N)
    return s
# avg_error

'''
A symmetric bi-exponential function with a fast and slow component

y = A + B*Exp(-|x/tau_slow|) + C*Exp(-|x/tau_fast|)
'''
def biexponential(x, A, B, tauS, C, tauF, t0):
    return A + B*np.exp(-np.abs((x-t0)/tauS)) + C*np.exp(-np.abs((x-t0)/tauF))
# end biexponential

'''
A symmetric bi-exponential function with a fast and slow component. With a penalization parameter
to guarrentee that the fast and slow components are the correct order. Only needs to be used
for the fit
'''
def biexponential_pen(x, A, B, tauS, C, tauF, t0):
    if 1.5*tauF > tauS:
        penalization = 10000.0*tauF
    else:
        penalization = 0.0
    return A + B*np.exp(-np.abs((x-t0)/tauS)) + C*np.exp(-np.abs((x-t0)/tauF)) + penalization
# end biexponential_pen

def biexponential_fit(x, y, p0=-1, warn=True):
    '''
    Fits data to a biexponential function.

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent varaible
        p0 (optional): The initial parameters for the fitting function. If -1 (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p : The fitting parameters that optimize the fit, or the initial paramters if the fit failed.
        perr : Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.biexponential_fit: X and Y data must have the same length")
        return
    if p0 == -1:
        ts_start = np.abs(x[0] - x[l-1])/2.0
        tf_start = ts_start/50.0
        p0 = (np.min(y), np.max(y)/10.0, ts_start, np.max(y), tf_start, 0.0)
    return generic_fit(x, y, p0, biexponential_pen, warn=warn)
# end biexponential_fit

'''
Logarithmic Power Dependence Function, analytic solution to exciton rate equations

y = (1/A)*log(1+Ax) + I0
'''
def log_analytic(x, B, A, I0):
    return B*np.log(1+A*x) + I0
# end log_analytic

def log_analytic_fit(x, y, p0=-1, warn=True):
    '''
    Fits data to to the logarithm-analytic function.

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent varaible
        p0 (optional): The initial parameters for the fitting function. If -1 (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p : The fitting parameters that optimize the fit, or the initial paramters if the fit failed.
        perr : Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.log_analytic_fit: X and Y data must have the same length")
        return
    if p0 == -1:
        p0=(np.mean(y), 0.1, 0.0)
    return generic_fit(x, y, p0, log_analytic, warn=warn)
# end log_analytic_fit

def mean_residual(data, fit):
    '''
    A generic function of calculating mean absolute value of the residual

    Args:
        data : the data to calculate
        fit : the corresponding values of the fitted function
    '''
    return np.mean(np.abs(data-fit))
# end mean_residual

'''
Interpolates meshgrid data, wrapper for scipy.interpolate.griddata

(x, y, D) are the meshgrid data you have
(meshX and meshY) are the meshgrid to interpolate on.
'''
def interp_meshgrid(x, y, D, meshX, meshY):
    return griddata((x.ravel(), y.ravel()), D.ravel(), (meshX, meshY))


'''
Takes a 2D scan and lowpases the columns of each scan using fitting.lowpass
'''
def lp_scan_cols(data, cutoff=0.05, samprate=1.0):
    rows, cols = data.shape
    original = np.copy(data)
    for i in range(cols):
        data[:,i] = lowpass(original[:,i], cutoff=cutoff, samprate=samprate)
    return data
# end lp_cube_cols

'''
Takes a data cube and lowpases the columns of each scan using fitting.lowpass
'''
def lp_cube_cols(datacube, cutoff=0.05, samprate=1.0):
    rows, cols, N = datacube.shape
    original = np.copy(datacube)
    for j in range(N):
        for i in range(cols):
            datacube[:,i,j] = lowpass(original[:,i,j], cutoff=cutoff, samprate=samprate)
    return datacube
# end lp_cube_cols

'''
Takes a data cube and lowpases the rows, then the columns of each scan using fitting.lowpass
'''
def lp_cube_rows_cols(datacube, cutoff=0.05, samprate=1.0):
    rows, cols, N = datacube.shape
    original = np.copy(datacube)
    for j in range(N):
        for i in range(rows):
            datacube[i,:,j] = lowpass(original[i,:,j], cutoff=cutoff, samprate=samprate)
        for i in range(cols):
            datacube[:,i,j] = lowpass(datacube[:,i,j], cutoff=cutoff, samprate=samprate)
    return datacube
# end lp_cube_cols

'''
Takes a data cube and subtracts out the background from each individual scan,
determines the background from the values of the last $nx columns

$ix is the number of columns at the end of each row to use as background
'''
def subtract_bg_cube(datacube, nx=20):
    rows, cols, N = datacube.shape
    for j in range(N):
        n = np.mean(datacube[:,cols-nx:cols,j], axis=1)
        for i in range(rows):
            datacube[i,:,j] = datacube[i,:,j] - n[i]
    return datacube
# end subtract_bg_cube

def max_from_interp(x, y, kind='cubic', N=200):
    '''
    Find the maximum of y in the range of x while interpolating to increase resolution.

    Args:
        x : The independent variable
        y : The dependent variable
        warn (str, optional) : parameter passed to interp1d
        N (int, optional) : resolution of the interpolated data

    Returns:
        xmax, ymax

            The x coordinate and value of the maximum
    '''
    _x = np.linspace(np.min(x), np.max(x), N)
    Ifunc = interp1d(x, y, kind=kind)
    ixmax = np.argmax(Ifunc(_x))
    xmax = _x[ixmax]
    ymax = Ifunc(xmax)
    return xmax, ymax, _x, Ifunc(_x)
# end max_from_interp
