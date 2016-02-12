import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from time import sleep
from os import getcwd as cwd

from scans import range_from_log

from new_colormaps import _viridis_data, _plasma_data

matplotlib.rcParams["keymap.fullscreen"] = ''

'''
Returns the viridis colormap,

for use before matplotlib is updated
'''
def get_viridis():
	return mcolors.ListedColormap(_viridis_data, name='Viridis')
#

'''
Returns the plasma colormap,

for use before matplotlib is updated
'''
def get_plasma():
	return mcolors.ListedColormap(_plasma_data, name='Plasma')

def generate_colormap(cpt=0.5, width=0.25):
	'''
	Returns the color map for plotting
	$cpt is the central point of the color transition (0 to 1 scale)
	$width is related to the width of the color transition (0 to 1 scale)
	'''
	low_RGB = (249/255., 250/255., 255/255.)
	cl_RGB = (197/255., 215/255., 239/255.)
	clpt = np.max([0.05, cpt-width])
	center_RGB = (100/255., 170/255., 211/255.)
	ch_RGB = (30/255., 113/255., 180/255.)
	chpt = np.min([0.95, cpt+width])
	high_RGB =  (8/255., 48/255., 110/255.)
	cdict = {'red':((0.0, 0.0, low_RGB[0]),
					(clpt, cl_RGB[0], cl_RGB[0]),
					(cpt, center_RGB[0], center_RGB[0]),
					(chpt, ch_RGB[0], ch_RGB[0]),
					(1.0, high_RGB[0], 0.0)),


			'green': ((0.0, 0.0, low_RGB[1]),
					(clpt, cl_RGB[1], cl_RGB[1]),
					(cpt, center_RGB[1], center_RGB[1]),
					(chpt, ch_RGB[1], ch_RGB[1]),
					(1.0, high_RGB[1], 0.0)),

			'blue':  ((0.0, 0.0, low_RGB[2]),
					  (clpt, cl_RGB[2], cl_RGB[2]),
					  (cpt, center_RGB[2], center_RGB[2]),
					  (chpt, ch_RGB[2], ch_RGB[2]),
					  (1.0, high_RGB[2], 0.0)),
		  }
	return mcolors.LinearSegmentedColormap('GreenBlue', cdict)
# end generate_colormap


def format_plot_axes(ax, fntsize=14, tickfntsize=12):
	'''
	Formats the plot axes in a standard format
	$ax is the axes object for the plot, such as plt.gca()
	'''
	for i in ax.spines.itervalues():
		i.set_linewidth(2)
	ax.tick_params(width=2, labelsize=tickfntsize, direction='out')
	matplotlib.rcParams.update({'font.size': fntsize})
# end format_plot_axes

'''
Sets the x and y ticks for a data image based on the log file
$ax is the current axes
$img is the image object
$log is the log file

$xparam and $yparam are the paramters for the x and y axes if they are strings set the from the
log file, if they are a numpy array they are set from that array

$nticks is the number of ticks to use
$sigfigs is the number of significant figures to round to
'''
def set_img_ticks(ax, img, log, xparam, yparam, nticks=5, sigfigs=2):
	if isinstance(xparam, str):
		xt = np.linspace(0, int(log['nx'])-1, nticks)
		xrng = range_from_log(xparam, log, log['nx'])
	elif isinstance(xparam, np.ndarray):
		xt = np.linspace(0, len(xparam)-1, nticks)
		xrng = xparam
	else:
		print 'Error set_img_ticks: X Paramter must be a string or an array'
		return
	if isinstance(yparam, str):
		yt = np.linspace(0, int(log['ny'])-1, nticks)
		yrng = range_from_log(yparam, log, log['ny'])
	elif isinstance(yparam, np.ndarray):
		yt = np.linspace(0, len(yparam)-1, nticks)
		yrng = yparam
	else:
		print 'Error set_img_ticks: Y Paramter must be a string or an array'
		return
	xl = xrng[xt.astype(int)]
	yl = yrng[yt.astype(int)]
	for i in range(len(xl)):
	    xl[i] = round(xl[i], sigfigs)
	for i in range(len(yl)):
	    yl[i] = round(yl[i], sigfigs)
	img.set_extent((xl[0], xl[len(xt)-1], yl[len(yt)-1], yl[0]))
	ax.set_xticks(xl)
	ax.set_yticks(yl)
	ax.set_xlim(xl[0], xl[len(xt)-1])
	ax.set_ylim(yl[len(yt)-1], yl[0])
# end set_img_ticks

'''
A Class for displaying a data cube, where the user can switch between scans using a slider or using
the arrow keys.

$d is the data cube to be displayed.

CubeFigure.ax gives the axes for manipulation
'''
class CubeFigure(object):
    def __init__(self, d):
        self.d = d

        rows, cols, N = self.d.shape
        self.N = N

        self.fig = plt.figure()
        self.ax = self.fig.gca()

        self.ix = 0
        self.img = self.ax.imshow(self.d[:,:,self.ix], cmap=get_viridis())
        self.cbar = self.fig.colorbar(self.img)

        self.sliderax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(self.sliderax, 'Scan', 0, N, valinit=1)
        self.slider.on_changed(self.update)
        self.slider.drawon = False

        self.slider.valtext.set_text('{}'.format(int(0.0))+'/'+str(self.N))

        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.fig.show()
    # end init

    def onKey(self, event):
        k = event.key
        if (k == 'right' or k == 'up') and self.ix < self.N:
            val = self.ix + 1
            self.slider.set_val(val)
            self.update(val)
        elif (k == 'left' or k == 'down') and self.ix > 0:
            val = self.ix - 1
            self.slider.set_val(val)
            self.update(val)

    def update(self, value):
        self.ix = int(value)
        self.img.set_data(self.d[:,:,self.ix-1])
        self.img.autoscale()
        self.slider.valtext.set_text('{}'.format(int(self.ix))+'/'+str(self.N))
        self.fig.canvas.draw()
    # end update
# end CubeFigure
