'''
display.py

A module for general functions related to displaying information, for functions and classes related
to displaying specific kinds of information see visual.py

Last updated July 2023

by Trevor Arp
All Rights Reserved
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import cmasher as cmr # colorscales

import addcopyfighandler
from datetime import datetime

import numpy as np

from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import shift

from trevorarp.math import power_law
from trevorarp.fitting import power_law_fit

class figure_inches():
    '''
    A class to assist in the layout of figures, handling the conversion from matplotlibs
    0 to 1 units to actual physical units in inches. Will also generate figures and axes
    of various kinds from specifications given in inches.

    Args:
        name (str) : The name of the figure, can't have two figures with the same name in a script.
            If None, will default to the current date and time.
        xinches (float) : The width of the figure in inches. To make several default size
            figures, pass the number of figures per row in a grid layout as a string. For example,
            give xinches="2" to make the figure large enough for two default figures per row.
        yinches (float) : The height of the figure in inches. To make several default size
            figures, pass the number of rows of figures (in a grid layout) as a string. For example,
            give yinches="2" to make the figure large enough for two rows of default figures.
        defaults (dict) : A dictionary of default values (xinches, width, etc.) which will overwrite the
            built in defaults.
        style (str) : The style to use. Options are "notes" style (default) as defined in notes_format
            of the "figure" format for customizing papers as defined in figure_format. Style will not be
            set if style (i.e. rcParams) has already been modified.
        dark (bool) : Use the dark theme
    '''

    def __init__(self, name=None, xinches="1", yinches="1", defaults=None, style='notes', dark=False):
        self.defaults = {
        'xinches':4.5,
        'yinches':4.8,
        'xmargin':0.8,
        'ymargin':0.55,
        'height':3.5,
        'width':4.0,
        'xint':0.8,
        'yint':0.8
        }
        if defaults is not None:
            for k, v in defaults.items():
                if k in self.defaults:
                    self.defaults[k] = defaults[k]

        self.default_figs_x = 0
        if isinstance(xinches,str):
            try:
                self.Nx = int(xinches)
            except:
                raise ValueError("Invalid str to specify number of default figures.")
            self.xinches = self.defaults['xinches'] + (self.Nx-1)*(self.defaults['width']+self.defaults['xint'])
            self.default_xstart = self.defaults['xmargin']
        else:
            self.Nx = 0
            self.xinches = xinches
            self.default_xstart = self.defaults['xmargin']

        if isinstance(yinches,str):
            try:
                self.Ny = int(yinches)
            except:
                raise ValueError("Invalid str to specify number of default figures.")
            self.yinches = self.defaults['yinches'] + (self.Ny-1)*(self.defaults['height']+self.defaults['yint'])
            self.default_ystart = self.defaults['ymargin'] + (self.Ny-1)*(self.defaults['height']+self.defaults['yint'])
        else:
            self.Ny = 0
            self.yinches = yinches
            self.default_ystart = self.defaults['ymargin']
        self.r = self.yinches/self.xinches

        if name is None:
            now = datetime.now()
            self.name = now.strftime("%Y-%m-%d-%H:%M:%S")
        else:
            self.name = name

        if mpl.rcParamsDefault == mpl.rcParams:
            if style == 'notes':
                notes_format()
            elif style == "paper" or style == "figure":
                figure_format()
            else:
                print("Invalid style, using default")

        if dark:
            facecolor='k'
            plt.style.use('dark_background')
        else:
            facecolor='w'
        self.axes = []
        self.fig = plt.figure(self.name, figsize=(self.xinches, self.yinches), facecolor=facecolor)
    # end init

    def stamp(self, iden, fontsize=10, wrapnum=None, xpos=0.05, ypos=0.05):
        '''
        Put a identifier (or list of identifiers) in the top left of the figure
        Args:
            iden (str or list): the Identifier(s) to display. If is a list will display each in series.
            fontsize (int) : The font size
            wrapnum (int) : The number of entries in a list to wrap back on. Automatically determined if None
            xpos (float) : position in inches of left side of text
            ypos (float) : position in inches of top of text
        '''
        if wrapnum is None:
            wrapnum = 3*self.Nx
        if isinstance(iden, str):
            s = iden
        elif isinstance(iden, list):
            s = ''
            for i in range(len(iden)):
                s += iden[i]
                if i != len(iden) - 1:
                    s = s + ", "
                if (i+1) % wrapnum == 0 and i != len(iden)-1:
                    s += "\n"
        else:
            print("Warning figure_axes.stamp, unknown identifier type")
            s = str(iden)
        plt.figtext(xpos/self.xinches, (self.yinches - ypos)/self.yinches, s, fontsize=fontsize, ha='left', va='top')
    #

    def get_fig(self):
        '''
        Returns a figures object with the physical size given
        '''
        return self.fig
    # end make_figure

    def make_axes(self, spec=None, zorder=1):
        '''
        Makes and returns a matplotlib Axes object.

        Args:
            spec : A list of the dimensions (in inches) of the axis can be [left, bottom, width, height]
                to fully specify, or [left, bottom] to set the x and y coordinates and use the default size or
                leave as None for the default size and position.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
        if spec is None:
            self.default_figs_x += 1
            spec = [self.default_xstart, self.default_ystart, self.defaults['width'], self.defaults['height']]
            self.default_xstart = self.default_xstart + self.defaults['width'] + self.defaults['xint']
            if self.default_figs_x >= self.Nx:
                self.default_figs_x = 0
                self.default_xstart = self.defaults['xint']
                self.default_ystart = self.default_ystart - self.defaults['height'] - self.defaults['yint']
        elif len(spec) == 2:
            spec = [spec[0], spec[1], self.defaults['width'], self.defaults['height']]
        plt.figure(self.name)
        ax = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches], zorder=zorder)
        self.axes.append(ax)
        return ax
    # make_axes

    def make_3daxes(self, spec=None, zorder=1):
        '''
        Makes and returns a matplotlib Axes object with a 3D projection

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
        if spec is None:
            self.default_figs_x += 1
            spec = [self.default_xstart, self.default_ystart, self.defaults['width'], self.defaults['height']]
            self.default_xstart = self.default_xstart + self.defaults['width'] + self.defaults['xint']
            if self.default_figs_x >= self.Nx:
                self.default_figs_x = 0
                self.default_xstart = self.defaults['xint']
                self.default_ystart = self.default_ystart - self.defaults['height'] - self.defaults['yint']
        elif len(spec) == 2:
            spec = [spec[0], spec[1], self.defaults['width'], self.defaults['height']]
        plt.figure(self.name)
        ax =  plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches], zorder=zorder, projection='3d')
        self.axes.append(ax)
        return ax
    # make_3daxes

    def make_img_axes(self, spec=None, cbpercent=0.4, cbmargin=0.1, cbheightpercent=0.15, zorder=1):
        '''
        Makes and returns a matplotlib Axes object with a default colorbar.
        To easily make a colorbar in the title area.

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
            cbpercent (float): The percentage of the width of the axes that the colorbar should use to define
                it's width, i.e. cbwidth = cbpercent*width.
        '''
        if spec is None:
            self.default_figs_x += 1
            spec = [self.default_xstart, self.default_ystart, self.defaults['height'], self.defaults['height']]
            self.default_xstart = self.default_xstart + self.defaults['height'] + self.defaults['xint']
            if self.default_figs_x >= self.Nx:
                self.default_figs_x = 0
                self.default_xstart = self.defaults['xint']
                self.default_ystart = self.default_ystart - self.defaults['height'] - self.defaults['yint']
        elif len(spec) == 2:
            spec = [spec[0], spec[1], self.defaults['height'], self.defaults['height']]
        plt.figure(self.name)
        xpos = spec[0]/self.xinches
        ypos = spec[1]/self.yinches
        width = spec[2]/self.xinches
        height = spec[3]/self.yinches
        ax = plt.axes([xpos, ypos, width, height], zorder=zorder)

        margin = cbmargin/self.yinches #min([0.1*height]
        cbwidth = cbpercent*width
        cbheight = cbheightpercent/self.yinches
        cb = plt.axes([xpos+width-cbwidth, ypos+height+margin, cbwidth, cbheight], zorder=zorder)
        xaxis_top(cb)
        cb.__display_default_flag__ = True
        self.axes.append(ax)
        self.axes.append(cb)
        return ax, cb
    # make_axes_and_cb

    def make_dualy_axes(self, spec=None, color_left='k', color_right='k', zorder=1, lefthigher=True):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis.

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_left (str, optional) : The color (in matplotlib notation) of the left y-axis, default black.
            color_right (str, optional) : The color (in matplotlib notation) of the left x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
            lefthigher (bool, optional) : If True (default) the left axis will be on top and provide the x-axis.
        '''
        if spec is None:
            self.default_figs_x += 1
            spec = [self.default_xstart, self.default_ystart, self.defaults['width'], self.defaults['height']]
            self.default_xstart = self.default_xstart + self.defaults['width'] + self.defaults['xint']
            if self.default_figs_x >= self.Nx:
                self.default_figs_x = 0
                self.default_xstart = self.defaults['xint']
                self.default_ystart = self.default_ystart - self.defaults['height'] - self.defaults['yint']
        elif len(spec) == 2:
            spec = [spec[0], spec[1], self.defaults['width'], self.defaults['height']]
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        if lefthigher:
            zorderl = zorder + 1
            zorderr = zorder
        else:
            zorderl = zorder
            zorderr = zorder + 1
        #

        axl = plt.axes([spec[0]+1, spec[1]+1, spec[2]+1, spec[3]+1], zorder=zorderl)
        axl.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axl.patch.set_alpha(0)
        axl.tick_params('y', which='both', colors=color_left)
        axl.spines['left'].set_color(color_left)
        axl.spines['right'].set_color(color_right)

        axr = plt.axes([spec[0]+2, spec[1]+2, spec[2]+2, spec[3]+2], zorder=zorderr)
        axr.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axr.patch.set_alpha(0)
        axr.xaxis.set_visible(False)
        axr.yaxis.tick_right()
        axr.yaxis.set_label_position("right")
        axr.tick_params('y', which='both', colors=color_right)
        self.axes.append(axl)
        self.axes.append(axr)
        return axl, axr
    # make_dualy_axes

    def make_dualx_axes(self, spec=None, color_bottom='k', color_top='k', zorder=1):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis. Note, the
        first axes returned (the bottom x-axis) is "on bottom" and provides the y-axis

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_bottom (str, optional) : The color (in matplotlib notation) of the bottom x-axis, default black.
            color_top (str, optional) : The color (in matplotlib notation) of the top x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
        if spec is None:
            self.default_figs_x += 1
            spec = [self.default_xstart, self.default_ystart, self.defaults['width'], self.defaults['height']]
            self.default_xstart = self.default_xstart + self.defaults['width'] + self.defaults['xint']
            if self.default_figs_x >= self.Nx:
                self.default_figs_x = 0
                self.default_xstart = self.defaults['xint']
                self.default_ystart = self.default_ystart - self.defaults['height'] - self.defaults['yint']
        elif len(spec) == 2:
            spec = [spec[0], spec[1], self.defaults['width'], self.defaults['height']]
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        axb = plt.axes([spec[0]+1, spec[0]+1, spec[0]+1, spec[0]+1], zorder=zorder)
        axb.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axb.patch.set_alpha(0)
        axb.tick_params('x', which='both', colors=color_bottom)
        axb.spines['bottom'].set_color(color_bottom)
        axb.spines['top'].set_color(color_top)

        axt = plt.axes([spec[0]+2, spec[0]+2, spec[0]+2, spec[0]+2], zorder=zorder+1)
        axt.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axt.patch.set_alpha(0)
        axt.yaxis.set_visible(False)
        axt.xaxis.tick_top()
        axt.xaxis.set_label_position("top")
        axt.tick_params('x', which='both', colors=color_top)
        self.axes.append(axb)
        self.axes.append(axt)
        return axb, axt
    # make_dualx_axes

    def make_dualxy_axes(self, spec=None, color_bottom='k', color_top='k', color_left='k', color_right='k', zorder=1):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis. Note: the
        first axis returned (the left y-axis) is "on top"

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_bottom (str, optional) : The color (in matplotlib notation) of the bottom x-axis, default black.
            color_top (str, optional) : The color (in matplotlib notation) of the top x-axis, default black.
            color_left (str, optional) : The color (in matplotlib notation) of the left y-axis, default black.
            color_right (str, optional) : The color (in matplotlib notation) of the left x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.

        Returns:
            axes : Returns the axes in the following order: left, right, bottom, top
        '''
        if spec is None:
            self.default_figs_x += 1
            spec = [self.default_xstart, self.default_ystart, self.defaults['width'], self.defaults['height']]
            self.default_xstart = self.default_xstart + self.defaults['width'] + self.defaults['xint']
            if self.default_figs_x >= self.Nx:
                self.default_figs_x = 0
                self.default_xstart = self.defaults['xint']
                self.default_ystart = self.default_ystart - self.defaults['height'] - self.defaults['yint']
        elif len(spec) == 2:
            spec = [spec[0], spec[1], self.defaults['width'], self.defaults['height']]
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        axl = plt.axes([spec[0]+4, spec[0]+4, spec[0]+4, spec[0]+4], zorder=zorder+3)
        axl.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axl.patch.set_alpha(0)
        axl.xaxis.set_visible(False)
        axl.tick_params('y', which='both', colors=color_left)
        axl.spines['left'].set_color(color_left)
        axl.spines['right'].set_color(color_right)
        axl.spines['top'].set_color(color_top)
        axl.spines['bottom'].set_color(color_bottom)

        axr = plt.axes([spec[0]+3, spec[0]+3, spec[0]+3, spec[0]+3], zorder=zorder+2)
        axr.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axr.patch.set_alpha(0)
        axr.xaxis.set_visible(False)
        axr.yaxis.tick_right()
        axr.yaxis.set_label_position("right")
        axr.tick_params('y', which='both', colors=color_right)

        axb = plt.axes([spec[0]+1, spec[0]+1, spec[0]+1, spec[0]+1], zorder=zorder+1)
        axb.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axb.patch.set_alpha(0)
        axb.yaxis.set_visible(False)
        axb.xaxis.tick_bottom()
        axb.xaxis.set_label_position("bottom")
        axb.tick_params('x', which='both', colors=color_bottom)

        axt = plt.axes([spec[0]+2, spec[0]+2, spec[0]+2, spec[0]+2], zorder=zorder)
        axt.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axt.patch.set_alpha(0)
        axt.yaxis.set_visible(False)
        axt.xaxis.tick_top()
        axt.xaxis.set_label_position("top")
        axt.tick_params('x', which='both', colors=color_top)
        self.axes.append(axb)
        self.axes.append(axt)
        self.axes.append(axl)
        self.axes.append(axr)
        return axl, axr, axb, axt
    # make_dualx_axes

    def figtext(self, xcoord, ycoord, *args, **kwargs):
        '''
        Wrapper for plt.figtext, takes same arguments except in units of inches, and converts then to figure units

        Args:
            xcoord (float) : X Coordinate in inches
            ycoord (float) : Y Coordinate in inches
            args : positional arguments to pass to pyplot.figtext
            kwargs : keyword arguments to pass to pyplot.figtext
        '''
        plt.figtext(xcoord/self.xinches, ycoord/self.yinches, *args, **kwargs)
# end figure_inches

def interactive_image(fig, ax, X, Y, data):
    '''
    Add a standard set of interactive elements to an image plot

    Args:
        fig : Either the matplotlib figure or the figure_axes object to apply effects to.
        ax : The specific axes to apply the effects to.
        X : X (column) axis data, either a 1D array or 2D meshgrid (if meshgrid draws from X[0,:])
        Y : Y (row) axis data, either a 1D array or 2D meshgrid (if meshgrid draws from Y[:,0])
    '''
    if isinstance(fig, figure_inches):
        fig = fig.get_fig()
    if len(X.shape) > 1:
        X = X[0,:]
    if len(Y.shape) > 1:
        Y = Y[:,0]

    def onclick(event):
        if event.inaxes==ax:
            xix = np.searchsorted(X,event.xdata)
            yix = np.searchsorted(Y,event.ydata)
            print('x=%f, y=%f, z=%f'%(event.xdata, event.ydata, data[yix, xix]))
    fig.canvas.mpl_connect('button_press_event', onclick)
#

def yaxis_right(ax):
    '''
    A simple function to move the y-axis of a matplotlib plot to the right.

    Args:
        ax : The matplotlib axes object to manipulate
    '''
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
# end yaxis_right

def xaxis_top(ax):
    '''
    A simple function to move the x-axis of a matplotlib plot to the top.

    Args:
        ax : The matplotlib axes object to manipulate
    '''
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
# end yaxis_right

def figure_format(fntsize=12, font=None, bilinear=True, labelpad=0):
    '''
    Defines a standard format for paper figures and other production quality visualizations,
    can use any font that is in the maplotlib font folder Lib/site-packages/matplotlib/mpl-data/fonts/tff,

    Args:
        fntsize (int, optional) : The fontsize
        font (int, optional) : The font, in None (default) uses Arial
        bilinear (bool, optional) : If true (default) will use 'bilinear' interpolation option in imshow
        labelpad (float, optional) : The default padding between the axes and axes labels.
    '''
    mpl.rcParams.update({'font.family':'sans-serif'})
    if font is not None:
        mpl.rcParams.update({'font.sans-serif':font})
    else:
        mpl.rcParams.update({'font.sans-serif':'Arial'})
    mpl.rcParams.update({'font.size':fntsize})
    mpl.rcParams.update({'axes.labelpad': labelpad})
    mpl.rcParams.update({'axes.titlepad': labelpad})
    mpl.rcParams.update({'xtick.direction':'out'})
    mpl.rcParams.update({'ytick.direction':'out'})
    mpl.rcParams.update({'xtick.major.width':1.0})
    mpl.rcParams.update({'ytick.major.width':1.0})
    mpl.rcParams.update({'axes.linewidth':1.0})
    if bilinear:
        mpl.rcParams.update({'image.interpolation':'bilinear'})
    mpl.rcParams["keymap.fullscreen"] = '' # To prevent f from being fullscreen
# end figure_format

def notes_format(fntsize=12, tickfntsize=9, font=None, bilinear=True, dark=False):
    '''
    A simplistic format meant for notes and easy display, allowing axes to be shown
    without having to worry too much about the formatting of the axes.

    Args:
        fntsize (int, optional) : The fontsize of labels, titles.
        tickfntsize (int, optional) : The fontsize of tick labels
        font (int, optional) : The font, in None (default) uses Arial
        bilinear (bool, optional) : If true (default) will use 'bilinear' interpolation option in imshow
        dark (bool) : Use the dark theme.
    '''
    mpl.rcParams.update({'font.family':'sans-serif'})
    if font is not None:
        mpl.rcParams.update({'font.sans-serif':font})
    else:
        mpl.rcParams.update({'font.sans-serif':'Arial'})
    mpl.rcParams.update({'font.size':fntsize})
    mpl.rcParams.update({'font.size':fntsize})
    mpl.rcParams.update({'xtick.labelsize':tickfntsize})
    mpl.rcParams.update({'ytick.labelsize':tickfntsize})
    mpl.rcParams.update({'axes.labelpad': 5})
    mpl.rcParams.update({'axes.titlepad': 6})
    mpl.rcParams.update({'xtick.direction':'out'})
    mpl.rcParams.update({'ytick.direction':'out'})
    mpl.rcParams.update({'xtick.major.width':1.0})
    mpl.rcParams.update({'ytick.major.width':1.0})
    mpl.rcParams.update({'axes.linewidth':1.0})
    if bilinear:
        mpl.rcParams.update({'image.interpolation':'bilinear'})
    mpl.rcParams["keymap.fullscreen"] = '' # To prevent f from being fullscreen
#

def change_axes_colors(ax, c):
    '''
    Changes the color of a given matplotlib Axes instance

    Note: for colorbars will need to use matplotlib.rcParams['axes.edgecolor'] = c before
    calling colorbarBase because whoever coded that clase bound the colorbar axes to that
    param value rather than having it function like a normal axis, which it is.

    Args:
        ax : The matplotlib axes object to manipulate
        c : The color, in matplotlib notation.
    '''
    ax.yaxis.label.set_color(c)
    ax.xaxis.label.set_color(c)
    ax.tick_params(axis='x', which='both', colors=c)
    ax.tick_params(axis='y', which='both', colors=c)
    ax.spines['bottom'].set_color(c)
    ax.spines['top'].set_color(c)
    ax.spines['left'].set_color(c)
    ax.spines['right'].set_color(c)
# end change_axes_colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    '''
    Takes a color map and returns a new colormap that only uses the part of it between minval and maxval
    on a scale of 0 to 1

    Taken From: http://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib

    Args:
        cmap : The matplotlib colormap object to truncate
        minval (float, optional) : The minimum colorvalue, on a scale of 0 to 1
        maxval (float, optional) : The maximum colorvalue, on a scale of 0 to 1
        n (int, optional) : The number of samples, if -1 (default) uses the sampling from cmap
    '''
    if n == -1:
        n = cmap.N
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
# end truncate_colormap

def colorscale_map(darray, mapname='viridis', cmin=None, cmax=None, centerzero=False, truncate=None):
    '''
    Generates a Colormap, Normalization and ScalarMappable objects for the given data.

    Args:
        darray : The data, If color bounds not specified the min and max of the array are used,
            the mappable is initialized to the data.
        mapname : The name (matplotlib conventions) of the colormap to use.
        cmin (float, optional) : The minimum value of the map
        cmax (float, optional) : The maximum value of the map
        centerzero (bool, optional) : If true will make zero the center value of the colorscale, use
            for diverging colorscales.
        truncate (tuple, optional) : If not None (default) will truncate the colormap with (min, max)
            colorvalues on a scale of 0 to 1.

    Returns:
        Tuple containing (cmap, norm, sm) where cmap is the Colormap, norm is the
        Normalization and sm is the ScalarMappable.
    '''
    if mapname == 'cmbipolar':
        cmap = get_cmbipolar()
    else:
        cmap = plt.get_cmap(mapname)
    if truncate is not None:
        cmap = truncate_colormap(cmap, truncate[0], truncate[1])
    if cmin is None:
        cmin = np.min(darray)
    if cmax is None:
        cmax = np.max(darray)
    if centerzero:
        cmax = max(np.abs([cmin, cmax]))
        cmin =  -1.0*cmax
    cNorm  = colors.Normalize(vmin=cmin, vmax=cmax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    scalarMap.set_array(darray)
    return cmap, cNorm, scalarMap
# end colorscale_map

def make_colorbar(ax, cmap, cnorm, title=None, ttlparams=None, orientation='vertical', ticks=None, ticklabels=None, color='k', alpha=None):
    '''
    Instantiates and returns a colorbar object for the given axes, with a few more options than
    instantiating directly

    Args:
        ax : The axes to make the colorbar on.
        cmap : The Colormap
        norm : The Normalization
        title: Text to place to the left on a horizonal colorbar or at the top of a vertical colorbar
        ttlparams: Dictionary of text parameters to pass to the title annotation
        orientation (str, optional) : 'vertical' (default) or 'horizontal' orientation
        ticks (list, optional) : the locations of the ticks. If None will let matplotlib automatically set them.
        ticklabels (list, optional) : the labels of the ticks. If None will let matplotlib automatically set them.
        color (str, optional) : the color of the colorbar, default black.
        alpha (float, optional) : the transparency of the colorbar

    Returns:
        The matplotlib.ColorbarBase object.
    '''
    if hasattr(ax,"__display_default_flag__"):
        orientation = "horizontal"
        xtop = True
    else:
        xtop = False
    if ticks is None:
        vmin = cnorm.vmin
        vmax = cnorm.vmax
        ticks = [vmin, (vmin+vmax)/2, vmax]
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=cnorm, orientation=orientation, ticks=ticks, alpha=alpha)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    if color != 'k':
        mpl.rcParams['axes.edgecolor'] = color
        change_axes_colors(ax, color)
    if xtop:
        xaxis_top(ax)
        ax.tick_params(pad=0)
    if title is not None:
        if ttlparams is None:
            ttlparams = {}
        if orientation == 'horizontal':
            ax.text(-0.05, 0.5, title, transform=ax.transAxes, va='center', ha='right', **ttlparams)
        else:
            ax.text(0.5, 1.05, title, transform=ax.transAxes, va='bottom', ha='center', **ttlparams)
    return cb
# end make_colorbar

def discrete_colorscale(rngdict, basecolormap='viridis'):
    '''
    Creates a discrete colorscale.

    Args:
        rngdict (dict) : The ranges to show with a color scale with labels as keys for example,
            rngdict={"<1/2":(0.0,0.45), "1/2":(0.45,0.55), ">1/2":(0.55,1.0)}, in increasing order.
        basecolormap (str, optional) : the continuous colormap to base the discrete colormap on, default 'viridis'.

    Returns: tick_locations, tick_labels, colorscale, normalization
    '''
    labels = []
    locations = []
    bounds = []
    bnds = []
    for k,v in rngdict.items():
        labels.append(k)
        locations.append((v[1]+v[0])/2.0)
        bnds.append(v[0])
        bnds.append(v[1])
    for i in bnds:
        if i not in bounds:
            bounds.append(i)
    bounds.sort()
    cmap = cm.get_cmap(basecolormap,len(rngdict))
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return locations, labels, cmap, norm
# end discrete_colorscale

def fix_label_misalign(ax, offset=-0.0045, fixX=True):
	"""
	Fixes the bug where the positive and negative tick labels have an offset

	Displaces the positive labels by the offset paramter

	Fixes the labels on the x axis is fixX is true, the y axis labels otherwise
	"""
	if fixX:
		tks = ax.get_xticklabels()
		xticks = ax.get_xticks().tolist()
		for i in range(len(tks)):
		    if float(xticks[i]) >= 0.0:
		        tks[i].set_y(offset)
	else:
		tks = ax.get_yticklabels()
		yticks = ax.get_yticks().tolist()
		for i in range(len(tks)):
		    if float(yticks[i]) >= 0.0:
		        tks[i].set_x(offset)
# ebd fix_label_misalign

def zoom_in_center(ax, percentage):
	'''
	Zooms in on central $percentage area of a 2D figure
	'''
	ymax, ymin = ax.get_ylim()
	xmin, xmax = ax.get_xlim()
	dy = np.abs(ymax - ymin)
	dx = np.abs(xmax - xmin)
	y0 = max(ax.get_ylim()) - dy/2
	x0 = max(ax.get_xlim()) - dx/2
	sp = np.sqrt(percentage)
	ax.set_ylim(y0+sp*dy/2, y0-sp*dy/2)
	ax.set_xlim(x0-sp*dx/2, x0+sp*dx/2)
# end zoom_in_center

def center_2D_data(d, mode='wrap', cval=0.0):
	'''
	Centers the data in the middle of the map

	$d is a 2D arrawy of data to center

	$mode specifies what do do with the points outside the boundaries, passed to
	scipy.ndimage.interpolation.shift, default is wrap

	$fillvall if the mode is 'constant' fills out of bounds points with this value
	'''
	rows, cols = d.shape
	coords = center_of_mass(np.abs(d))
	dr = rows/2 - coords[0]
	dc = cols/2 - coords[1]
	if mode == 'constant':
		d = shift(d, (dr, dc), mode=mode, cval=0.0)
	else:
		d = shift(d, (dr, dc), mode=mode)
	return d
# end center_2D_data

def show_powerlaw_points(aximg, axplt, log, x, y, power, d, color=None, showerr=False):
	'''
	Plots a power law from a point on a 2D map, puts markers on the map
	'''
	if color is None:
		color=['b','r','m','c']
	fx = log['Fast Axis']
	fc = (fx[1]-fx[0])/log['nx']
	sx = log['Slow Axis']
	sc = (sx[1]-sx[0])/log['ny']
	for i in range(len(x)):
		aximg.plot([fx[0]+fc*x[i]], [sx[0]+sc*y[i]], color[i]+'o')
		pc = d[y[i],x[i],:]
		if len(power.shape) == 1:
			pw = power[:]
		elif len(power.shape) == 2:
			pw = power[y[i],:]
		elif len(power.shape) == 3:
			pw = power[y[i], x[i], :]
		else:
			print("Error show_powerlaw_points: Invalid Power Data")
			return
		params, err = power_law_fit(pw-np.min(pw), pc)
		if showerr:
			lbl = r"$\gamma = $ "+ str(round(params[1],2)) + r' $\pm$' + str(round(err[1],2))
		else:
			lbl = r"$\gamma = $ "+ str(round(params[1],2))
		axplt.plot(pw, power_law(pw-np.min(pw), params[0], params[1], params[2]), color[i]+'-', lw=2 , label=lbl)
		axplt.plot(pw, pc, color[i]+'o', lw=2)
# end show_powerlaw_points

def _wavelength_to_rgb(wavelength, gamma=0.8):
	'''
	This converts a given wavelength of light to an approximate RGB color value. The wavelength must be given in nanometers in the range from 380 nm through 750 nm.

	Based on code by Dan Bruton
	http://www.physics.sfasu.edu/astro/color/spectra.html
	'''
	wavelength = float(wavelength)
	if wavelength >= 380 and wavelength <= 440:
		attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
		R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
		G = 0.0
		B = (1.0 * attenuation) ** gamma
	elif wavelength >= 440 and wavelength <= 490:
		R = 0.0
		G = ((wavelength - 440) / (490 - 440)) ** gamma
		B = 1.0
	elif wavelength >= 490 and wavelength <= 510:
		R = 0.0
		G = 1.0
		B = (-(wavelength - 510) / (510 - 490)) ** gamma
	elif wavelength >= 510 and wavelength <= 580:
		R = ((wavelength - 510) / (580 - 510)) ** gamma
		G = 1.0
		B = 0.0
	elif wavelength >= 580 and wavelength <= 645:
		R = 1.0
		G = (-(wavelength - 645) / (645 - 580)) ** gamma
		B = 0.0
	elif wavelength >= 645 and wavelength <= 750:
		attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
		R = (1.0 * attenuation) ** gamma
		G = 0.0
		B = 0.0
	else:
		R = 0.0
		G = 0.0
		B = 0.0
	#
	return (R, G, B)
# end wavelength_to_rgb

def wavelength_colormap(N=250):
	'''
	Creates a normalized colormap the converts a wavelength (in nm) to approximatly the color of the
	light with that wavelength
	'''
	w = np.linspace(380, 750, N)
	colorvals = []
	for i in range(N):
	    colorvals.append(_wavelength_to_rgb(w[i]))
	cmap = LinearSegmentedColormap.from_list('visiblespectrum', colorvals)
	cnorm = colors.Normalize(vmin=380, vmax=750)
	scalarMap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
	scalarMap.set_array(w)
	return cmap, cnorm, scalarMap
# end wavelength_colormap

def hex_to_rgb(value):
    '''
    From: https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    '''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    From: https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    '''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    '''
    From: https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
    creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def get_cmbipolar(lsp=500):
    vals = ['#00b4ff', '#00b2ff', '#01affe', '#01adfe', '#01aafd', '#02a8fd', '#02a5fc', '#02a3fc', '#03a1fb', '#039efb', '#039cfa', '#0499fa', '#0497f9', '#0494f9', '#0592f8', '#058ff8', '#058df7', '#068af7', '#0688f6', '#0785f6', '#0783f5', '#0780f5', '#087ef4', '#087bf4', '#0879f3', '#0977f3', '#0974f2', '#0972f2', '#0a6ff1', '#0a6df1', '#0a6af0', '#0b68f0', '#0b65f0', '#0b63ef', '#0c60ef', '#0c5eee', '#0c5bee', '#0d59ed', '#0d56ed', '#0d54ec', '#0e52ec', '#0e4feb', '#0e4deb', '#0f4aea', '#0f48ea', '#0f45e9', '#1043e9', '#1040e8', '#103ee8', '#113be7', '#1139e7', '#1136e6', '#1234e6', '#1231e5', '#132fe5', '#132ce4', '#132ae4', '#1428e3', '#1425e3', '#1423e2', '#1520e2', '#151ee1', '#151be1', '#1619e1', '#1616e0', '#1616dd', '#1515d9', '#1515d6', '#1515d2', '#1414cf', '#1414cb', '#1414c8', '#1313c4', '#1313c1', '#1313bd', '#1212ba', '#1212b6', '#1212b3', '#1111af', '#1111ac', '#1111a8', '#1010a5', '#1010a1', '#0f0f9e', '#0f0f9a', '#0f0f97', '#0e0e93', '#0e0e90', '#0e0e8c', '#0d0d89', '#0d0d85', '#0d0d82', '#0c0c7e', '#0c0c7b', '#0c0c77', '#0b0b74', '#0b0b70', '#0b0b6d', '#0a0a69', '#0a0a66', '#0a0a62', '#09095f', '#09095b', '#090958', '#080854', '#080851', '#08084d', '#07074a', '#070746', '#070743', '#06063f', '#06063c', '#060638', '#050535', '#050531', '#04042e', '#04042a', '#040427', '#030323', '#030320', '#03031c', '#020219', '#020215', '#020212', '#01010e', '#01010b', '#010107', '#000004', '#000000', '#040000', '#080000', '#0c0000', '#100000', '#140000', '#180000', '#1c0000', '#200000', '#240000', '#280000', '#2c0000', '#300000', '#350000', '#390000', '#3d0000', '#410000', '#450000', '#490000', '#4d0000', '#510000', '#550000', '#590000', '#5d0000', '#610000', '#650000', '#690000', '#6d0000', '#710000', '#750000', '#790000', '#7d0000', '#810000', '#850000', '#890000', '#8d0000', '#910000', '#960000', '#9a0000', '#9e0000', '#a20000', '#a60000', '#aa0000', '#ae0000', '#b20000', '#b60000', '#ba0000', '#be0000', '#c20000', '#c60000', '#ca0000', '#ce0000', '#d20000', '#d60000', '#da0000', '#de0000', '#e20000', '#e60000', '#ea0000', '#ee0000', '#f20000', '#f70000', '#fb0000', '#ff0000', '#ff0400', '#fe0701', '#fe0b01', '#fe0f02', '#fe1302', '#fd1603', '#fd1a03', '#fd1e04', '#fd2104', '#fc2505', '#fc2905', '#fc2d06', '#fc3006', '#fb3407', '#fb3807', '#fb3b08', '#fa3f08', '#fa4309', '#fa4709', '#fa4a0a', '#f94e0a', '#f9520b', '#f9550b', '#f9590c', '#f85d0c', '#f8610d', '#f8640d', '#f8680e', '#f76c0e', '#f76f0f', '#f7730f', '#f77710', '#f67b10', '#f67e11', '#f68211', '#f58612', '#f58912', '#f58d13', '#f59113', '#f49514', '#f49814', '#f49c15', '#f4a015', '#f3a316', '#f3a716', '#f3ab17', '#f3af17', '#f2b218', '#f2b618', '#f2ba19', '#f1bd19', '#f1c11a', '#f1c51a', '#f1c91b', '#f0cc1b', '#f0d01c', '#f0d41c', '#f0d71d', '#efdb1d', '#efdf1e', '#efe31e', '#efe61f', '#eeea1f', '#eeee20']
    newcmp = get_continuous_cmap(vals)
    return newcmp
#
