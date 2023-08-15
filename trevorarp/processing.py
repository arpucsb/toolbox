'''
processing.py

A module of basic data processing and filtering.

Last updated March 2020

by Trevor Arp
'''
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fftpack import fft, fftfreq

from scipy import ndimage as ndi
from skimage import filters
from skimage.morphology import skeletonize, remove_small_objects

def moving_average(a, n=3):
    """
    Calculate the simple moving average of an array, taken from stack overflow:
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    Args:
        a: The array to average
        n: The window to average over

    Returns:

    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def lowpass(data, cutoff=0.05, samprate=1.0):
    '''
    A generic lowpass filter, based on a Butterworth filter.

    Args:
        data : The data to be filtered, considered to be sampled at 1 Hz
        cutoff (float, optional) : The cutoff frequency in units of the nyquist frequency, must be less than 1
        samprate (float, optional) : is the sample rate in Hz
    '''
    b,a = butter(2,cutoff/(samprate/2.0),btype='low',analog=0,output='ba')
    return filtfilt(b,a,data)
# end lowpass

def highpass(data, cutoff=0.05, samprate=1.0):
    '''
    A generic lowpass filter, based on a Butterworth filter.

    Args:
        data : The data to be filtered, considered to be sampled at 1 Hz
        cutoff (float, optional) : The cutoff frequency in units of the nyquist frequency, must be less than 1
        samprate (float, optional) : is the sample rate in Hz
    '''
    b,a = butter(2,cutoff/(samprate/2.0),btype='high',analog=0,output='ba')
    return filtfilt(b,a,data)
# end lowpass

def lowpasspx(data, Npixels=50):
    '''
    A lowpass for data that is definied by the time/time_constant, meaning it is at the full limit of singal
    to noise and you only need to filter out the pixel to pixel variation.

    Args:
        data : The data to pass to the processing.lowpass function
        Npixels : The number of pixels to define the frequency. cutoff = Npixels/len(data)
    '''
    N = len(data)
    return lowpass(data, Npixels / N)
#

def notchfilter(data, frequency, Q=2.0, samplefreq=1.0):
    '''
    A notch filter to remove a specific frequency from a signal.

    Args:
        data (numpy array) : The data to be filtered
        frequency (float) : The frequency to filter
        Q (float) : The quality factor defining the frequency width of the notch filter
        samplefreq (float) : The sampling frequency of the signal
    '''
    b,a = iirnotch(frequency, Q, fs=samplefreq)
    return filtfilt(b,a,data)
# end notchfilter

def normfft(d):
    '''
    Calculates a normalized Fast Fourier Transform (FFT) of the given data

    Args:
        d : The data

    Returns:
        The normalized Fast Fourier Transform of the data.
    '''
    n = len(d)
    f = fft(d)
    return 2.0*np.abs(f)/n
# end normfft

def normfft_freq(t, d):
    '''
    Calculates a normalized Fast Fourier Transform (FFT) of the given data and the frequency samples for
    a given an (evenly sampled) time series

    Args:
        t : An evenly sampled times series for the data
        d : The data

    Returns:
        A tuple containing (freq, fft)

            freq - The sampling frequencies for the FFT, based on the argument t.

            fft - The normalized Fast Fourier Transform of the data.
    '''
    n = len(d)
    f = fft(d)
    f = 2.0*np.abs(f)/n
    freq = fftfreq(n, d=np.mean(np.diff(t)))
    return freq, f
# end normfft

'''
Finds sharp edges in the input image $d using a sobel filter, and morphological operations

if $remove_small is true then small domains will be removed from the filtered image prior to the final
calculation of the edge, has the potential to remove some of the edge
'''
def find_sharp_edges(d, remove_small=False):
    # edge filter
    edge = filters.sobel(d)

    # Convert to binary image
    thresh = filters.threshold_li(edge)
    edge = edge > thresh

    # Close the gaps
    edge = ndi.morphology.binary_closing(edge)

    # If desiered remove small domains
    if remove_small:
        edge = remove_small_objects(edge)

    # Skeletonize the image down to minimally sized features
    edge = skeletonize(edge)
    return edge
# end find_sharp_edges
