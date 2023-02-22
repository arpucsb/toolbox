'''
sim.py

A module for simulating various things

Last updated June 2016

by Trevor Arp

'''
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from timeit import default_timer as timer
from scipy.integrate import quad
import datetime
import traceback

'''
For a given profile of $gamma simulate the resulting photocurrent assuming a power law
photoresponse I = A*P^gamma, and a diffraction limited guassian beam with a full with at
half max given by $FWHM which must be in the same units as the spatial profile of gamma.
'''
def conv_PC_gamma(gamma, FWHM):
    sigma2 = (FWHM/2.355)**2
    N = len(gamma)
    x = np.arange(float(N))
    if len(x) != N:
        print("Error in conv_PC_gamma: x and gamma must be the same length")
        return
    f = np.zeros(N)
    mn = np.amin(gamma)
    for i in range(N):
        if gamma[i] > mn:
            f[i] = 1.0
    # The integrade of the convolution integral
    def conv(xp, x, sigma2):
        return np.exp(-(gamma[int(xp)]/(2.0*sigma2))*(x-xp)**2)*f[int(xp)]
    out = np.ones(N)
    for i in range(N):
        y, abserr = quad(conv, 0, N, args=(x[i], sigma2), limit=200)
        out[i] = y
    return out
# end conv_PC_gamma

def multiprocess2D(func, args_array, ncores=4, display=True):
    '''
    Multipurpose parallel processing

    Takes a function and an array of arguments, evaluates the function with the given
    arguments for each point, processing in parallel using ncores number of parallel
    processes.

    WARNING: needs to be protected by a if __name__ == "__main__" block or else
    multiprocessing.pool will throw a hissy fit.

    Args:
        func : The function to evaluate, can only accept one argument but it can be a list
            or tuple
        args_array is the 2D array of arguments to the input function $func.
        ncores : The number of nodes to pass to multiprocessing.Pool
        display : Will display progress if true.

    Returns:
        Returns the results of the calculation as a 2D numpy ndarray.
    '''
    pool = Pool(processes=ncores)
    rows = len(args_array)
    cols = len(args_array[0])
    output = np.zeros((rows, cols))
    if rows > 10:
        disp_rows = np.arange(rows/10, rows, rows/10)
    else:
        disp_rows = np.arange(1, rows, 1)
    if display:
        print("Parallel Processing Started with " + str(ncores) + " subprocesses")
    t0 = timer()
    for i in range(rows):
        worker_args = []
        for j in range(cols):
            worker_args.append(args_array[i][j])
        try:
            out = pool.map(func, worker_args)
            for j in range(cols):
                output[i,j] = out[j]
            if display and i in disp_rows:
                print(str(round(100*i/float(rows))) + "% Complete")
        except:
            print("Exception in sim.multiprocessing2D: Cannot Process")
            print(traceback.print_exc())
            print("sim.multiprocessing2D: Exiting Process")
            break
    tf = timer()
    if display:
        print(" ")
        dt = tf-t0
        print("Computations Completed in: " + str(datetime.timedelta(seconds=dt)))
    return output
# end multiprocess2D

'''
Samples a probability distribution using the Metropolis-Hastings Algorithm

Returns an numpy array of N samples of the probabilty distribution

Parameters:

N is the number of samples to generate

PDF is PDF(x), the Probability Density Function to be sampled, should accept one argument

x0=0.0 is the starting value of the markov chain used to generate the samples, set to approximatly the mean
value of your probability distribution. Set to 0.0 by default

rng=1.0 is the range of innovation for candidate selection, i.e. the next candidate in the Markov chain is
can = x + random.uniform(-rng,rng), make sure this samples your distribution well, 1.0 by default

'''
def metropolis_sample(N, PDF, x0=0.0, rng=1.0):
	vec = np.zeros(N)
	innov = np.random.uniform(-rng, rng, N) #uniform proposal distribution
	x = x0
	vec[0] = x
	ix = 1
	i = 0
	while ix < N:
		can = x + innov[i] #candidate
		aprob = min(1.0, PDF(can)/PDF(x)) #acceptance probability
		u = np.random.uniform(0,1)
		if u < aprob:
			x = can
			vec[ix] = x
			ix += 1
		i += 1
		if i >= N:
			innov = np.random.uniform(-rng,rng,N)
			i = 0
	return vec
# end metropolis_sample
