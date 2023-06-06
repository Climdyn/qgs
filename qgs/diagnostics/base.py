"""
    Diagnostic base classes
    =======================

    Abstract base classes defining the diagnostics of the model and used to analyze its outputs.

    Description of the classes
    --------------------------

    * :class:`Diagnostic`: General base class.
    * :class:`FieldDiagnostic`: General base class for diagnostics returning model's fields.

    Warnings
    --------

    These are `abstract base class`_, they must be subclassed to create new diagnostics!

    .. _abstract base class: https://docs.python.org/3/glossary.html#term-abstract-base-class

"""

from abc import ABC, abstractmethod
import warnings
import base64

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from ipywidgets import interactive

from sympy import symbols

from qgs.diagnostics.misc import *

# TODO: - need to introduce an oro_basis specific to the orography !!
#       - no orography plot when oro_basis is none


_n = symbols('n', real=True, nonnegative=True)  # only this works with sympy substitution, don't ask me why...


class Diagnostic(ABC):
    """General base class to create diagnostics.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, dimensional):

        self._model_params = None
        self._ocean = None
        self._ground = None
        self._orography = None
        self._heat_exchange = None
        self._newton = None
        self._data = None
        self._diagnostic_data = None
        self._diagnostic_data_dimensional = None
        self._time = None
        self._subs = list()

        self.dimensional = dimensional

        self._plot_title = ""
        self._plot_units = ""

        self.set_params(model_params)

    def __call__(self, time, data):
        self.set_data(time, data)
        return self.diagnostic

    @property
    def diagnostic(self):
        """~numpy.ndarray: The output diagnostic."""
        diag = self._check_diagnostic(self.dimensional)
        if diag is False:
            return self._get_diagnostic(self.dimensional)
        else:
            return diag

    def _check_diagnostic(self, dimensional):

        if self._data is None:
            return None
        elif self._diagnostic_data is not None and self._diagnostic_data_dimensional == dimensional:
            return self._diagnostic_data
        elif self._data.shape[0] != self._model_params.ndim:
            warnings.warn('Problem with the provided data. Expected array with 0-axis shape of '+str(self._model_params.ndim)+' .'
                          + 'Got ' + str(self._data.shape[0]) + ' ! Returning None.')
            return None
        else:
            return False

    def set_params(self, model_params, kwargs=None):
        """Set or replace the current model's parameter to which the diagnostic is attached to.

        Parameters
        ----------

        model_params: QgParams
            An instance of the model parameters.
        kwargs: dict
            Arguments to eventually reconfigure the diagnostic instance. Have the same keywords as the diagnostic instantiation method.
            If `None`, does not reconfigure the instance.
            Default to `None`.
        """

        self._set_params(model_params)
        self._subs = [(_n, model_params.scale_params.n)]
        if kwargs is not None:
            self._configure(**kwargs)

    def _set_params(self, model_params):

        self._model_params = model_params

        # determine some parameters about the model

        self._ocean = model_params.oceanic_basis is not None
        self._ground = model_params.ground_basis is not None
        if model_params.ground_params is not None:
            self._orography = model_params.ground_params.hk is not None
        self._heat_exchange = model_params.atemperature_params.C is not None
        self._newton = model_params.atemperature_params.thetas is not None

    def set_data(self, time, data):
        """Provide the model data to the diagnostic.

        Parameters
        ----------
        time: ~numpy.ndarray
            The time (in nondimensional timeunits) corresponding to the data.
            Its length should match the length of the last axis of the provided `data`.
        data: ~numpy.ndarray
            The model output data that the user want to convert using the diagnostic.
            Should be a 2D array of shape (:attr:`~.params.QgParams.ndim`, number_of_timesteps).
        """

        self._data = data
        self._time = time
        self._diagnostic_data = None

    @abstractmethod
    def _get_diagnostic(self, dimensional):
        pass

    @abstractmethod
    def _configure(self, **kwargs):
        pass


class FieldPointDiagnostic(Diagnostic):
    """General base class to give field values over time at a given point of the domain.

    Warnings
    --------

    Not yet implemented.

    """

    # TODO: WIP, still to be implemented

    def __init__(self, model_params, dimensional):

        Diagnostic.__init__(self, model_params, dimensional)

        self._x = None
        self._y = None

    def set_point_coordinates(self, x, y):

        self._x = x
        self._y = y

    def _get_point_coordinates(self):
        return self._x, self._y

    @property
    def point_coordinates(self):
        return self._get_point_coordinates()


class FieldDiagnostic(Diagnostic):
    """General base class for field diagnostic on the model's domain.
    Should provide a spatial gridded representation of the fields.


    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.
    """

    def __init__(self, model_params, dimensional):

        Diagnostic.__init__(self, model_params, dimensional)

        self._X = None
        self._Y = None

        self._grid_basis = None
        self._oro_basis = None

        self._color_bar_format = True

        self._default_plot_kwargs = {'cmap': plt.get_cmap('jet'), 'interpolation': 'spline36'}

    def __len__(self):
        if self.diagnostic is not None:
            return self.diagnostic.shape[0]
        else:
            return None

    @abstractmethod
    def _compute_grid(self, delta_x=None, delta_y=None):
        pass

    @property
    def grid_shape(self):
        """tuple(int): Return the shape of the grid of points covering the model's domain."""
        if self._Y is not None:
            return self._Y.shape

    def plot(self, time_index, style="image", ax=None, figsize=(16, 9),
             contour_labels=True, color_bar=True, show_time=True, plot_kwargs=None, oro_kwargs=None):
        """Plot the field of the provided data at the given time index.

        Parameters
        ----------
        time_index: int
            The time index of the data.
        style: str, optional
            The style of the plot. Can be:

            * `image`: show the fields as images with a given colormap specified in the `plot_kwargs` argument.
            * `contour`: show the fields as contour superimposed on the image of the orographic height (if it exists, see the `oro_kwargs` below).
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        contour_labels: bool, optional
            If `style` is set to `contour`, specify if the contours must be labelled with their value or not.
            Default to `True`.
        color_bar: bool, optional
            Specify if a color bar must be drawn beside the plot or not.
            Default to `True`.
        show_time: bool, optional
            Show the timestamp of the field on the plot or not.
            Default to `True`.
        plot_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.
        oro_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method plotting the image of the orography if `style` is set to `contour`.

        Returns
        -------
        ~matplotlib.axes.Axes
            An axes where the data were plotted.
        """

        if self.diagnostic is None:
            warnings.warn('No diagnostic data available. Showing nothing.')
            return None

        if time_index > self.__len__() - 1:
            warnings.warn('Time index ' + str(time_index) + ' provided is greater than the largest one possible: ' + str(self.__len__()) + ' . Showing nothing.')
            return None

        if plot_kwargs is None:
            plot_kwargs = self._default_plot_kwargs
        else:
            tmp_dict = dict(self._default_plot_kwargs)
            tmp_dict.update(plot_kwargs)
            plot_kwargs = tmp_dict

        if oro_kwargs is None:
            oro_kwargs = dict()
            oro_kwargs['cmap'] = plt.get_cmap('cividis')
            oro_kwargs['alpha'] = 0.6
            oro_kwargs['interpolation'] = 'spline36'

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        else:
            fig = ax.figure

        if style == "image":
            im = ax.imshow(self.diagnostic[time_index], origin='lower',
                           extent=[0, 2*np.pi/self._model_params.scale_params.n, 0, np.pi], **plot_kwargs)
            if color_bar:
                if self._color_bar_format:
                    cl = fig.colorbar(im, ax=ax, format=strf)
                else:
                    cl = fig.colorbar(im, ax=ax)

        elif style == "contour":
            if self._orography and oro_kwargs is not False and self._oro_basis is not None:
                hk = np.array(self._model_params.ground_params.hk, dtype=float)
                oro = hk @ np.swapaxes(self._oro_basis, 0, 1) * self._model_params.scale_params.Ha
                im = ax.imshow(oro, origin='lower',
                               extent=[0, 2 * np.pi / self._model_params.scale_params.n, 0, np.pi], **oro_kwargs)
                if color_bar:
                    cl = fig.colorbar(im, ax=ax)
                    cl.set_label('Orographic height (in m)', rotation=270, labelpad=20.)
            if 'cmap' in plot_kwargs:
                del plot_kwargs['cmap']
            if 'interpolation' in plot_kwargs:
                del plot_kwargs['interpolation']
            cont = ax.contour(self._X, self._Y, self.diagnostic[time_index], **plot_kwargs)
            if contour_labels:
                ax.clabel(cont, fontsize=10, fmt='%1.f')
        else:
            warnings.warn('Provided style parameter ' + style + ' not supported ! Nothing to plot.')

        title = self._plot_title
        if self.dimensional:
            title += self._plot_units
            if show_time:
                title += " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[time_index]) + " " + self._model_params.time_unit
        else:
            title += r" (in nondim units)"
            if show_time:
                title += " at " + str(self._time[time_index]) + " timeunits"

        ax.set_title(title, pad=20.)

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)

        if oro_kwargs is False:
            color_bar = False

        if color_bar:
            return ax, cl
        else:
            return ax

    def plot_grid_point(self, i, j, ax=None, figsize=(16, 9), plot_kwargs=None):
        """Plot the time serie of the field at a given grid point.

        Attributes
        ----------
        i: int
            Index corresponding to the x-coordinate of the grid point.
        j: int
            Index corresponding to the y-coordinate of the grid point.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        plot_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.plot` method.

        Returns
        -------
        ~matplotlib.axes.Axes
            An axes where the data were plotted.
        """

        if self.diagnostic is None:
            warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
            return None

        if plot_kwargs is None:
            plot_kwargs = dict()

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        else:
            fig = ax.figure

        ax.plot(self._time * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time),
                self.diagnostic[:, j, i], **plot_kwargs)

        title = self._plot_title
        if self.dimensional:
            title += self._plot_units
        else:
            title += r" (in nondim units)"
        title += " at gridpoint " + "({:.2f}".format(self._X[j, i]) + ", {:.2f}) ".format(self._Y[j, i])

        ax.set_title(title, pad=20.)

        if self.dimensional:
            ax.set_xlabel(time_axis_label + ' [' + self._model_params.time_unit + ']')
        else:
            ax.set_xlabel(time_axis_label + ' [timeunits]')

        return ax

    def movie(self, output='html', filename='', style="image", ax=None, figsize=(16, 9),
              contour_labels=True, color_bar=True, show_time=True, plot_kwargs=None, oro_kwargs=None, anim_kwargs=None):
        """ Create and return a movie of the output of the `plot` method animated over time.

        Parameters
        ----------
        output: str, optional
            Define the kind of movie being created. Can be:

            * `jshtml`: Generate an interactive HTML representation of the animation.
            * `html5`: Generate the movie as HTML5 code.
            * `html`: Output the movie as a HTML video tag.
            * `ihtml`: Output the interactive movie as a HTML video tag.
            * `save`: Save the movie in MP4 format (H264 codec).

            Default to `html`.
        filename: str, optional
            Filename (and path) where to save the movie. Needed if `output` is set to `save`.
        style: str, optional
            The style of the plot. Can be:

            * `image`: show the fields as images with a given colormap specified in the `plot_kwargs` argument.
            * `contour`: show the fields as contour superimposed on the image of the orographic height (see the `oro_kwargs` below).
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        contour_labels: bool, optional
            If `style` is set to `contour`, specify if the contours must be labelled with their value or not.
            Default to `True`.
        color_bar: bool, optional
            Specify if a color bar must be drawn beside the plot or not.
            Default to `True`.
        show_time: bool, optional
            Show the timestamp of the field on the plot or not.
            Default to `True`.
        plot_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.
        oro_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method plotting the image of the orography if `style` is set to `contour`.
        anim_kwargs: dict, optional
            Arguments to pass to the :class:`matplotlib.animation.FuncAnimation` instantiation method. Specify the parameters of the animation.

        Returns
        -------
        ~matplotlib.animation.FuncAnimation or HTML code or HTML tag
            The animation object or the HTML code or tag.
        """

        anim = self._make_anim(style, ax, figsize, contour_labels, color_bar, show_time, plot_kwargs, oro_kwargs, anim_kwargs, False)

        if 'html' in output:

            if output == "jshtml" or output == 'ihtml':
                jshtml = anim.to_jshtml()
                if output == "jshtml":
                    return jshtml
                else:
                    return HTML(jshtml)
            else:
                html5 = anim.to_html5_video()
                if output == 'html5':
                    return html5
                else:
                    return HTML(html5)

        elif output == 'save':

            if not filename:
                warnings.warn('No filename provided to the method animate. Video not saved !\n Please provide a filename.')

            html5 = anim.to_html5_video()
            start_index = html5.index('base64,')
            start_index += len('base64,')
            end_index = html5.index('">', start_index)
            video = html5[start_index: end_index]
            with open(filename, 'wb') as f:
                f.write(base64.b64decode(video))
            return html5

        else:
            warnings.warn('Provided output parameter ' + output + ' not supported ! Nothing to plot. Returning None.')
            anim = None

        return anim

    def animate(self, output='animate',  style="image", ax=None, figsize=(16, 9),
                contour_labels=True, color_bar=True, show_time=True, stride=1, plot_kwargs=None, oro_kwargs=None, anim_kwargs=None, show=True):
        """Return the output of the `plot` method animated over time.

        Parameters
        ----------
        output: str, optional
            Define the kind of animation being created. Can be:

            * `animate`: Create and show a :class:`ipywidgets.widgets.interaction.interactive` widget. Works only in Jupyter notebooks.
            * `show`: Create and show an animation with the :mod:`matplotlib.animation` module. Works only in IPython or Python.

        style: str, optional
            The style of the plot. Can be:

            * `image`: show the fields as images with a given colormap specified in the `plot_kwargs` argument.
            * `contour`: show the fields as contour superimposed on the image of the orographic height (see the `oro_kwargs` below).

        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        contour_labels: bool, optional
            If `style` is set to `contour`, specify if the contours must be labelled with their value or not.
            Default to `True`.
        color_bar: bool, optional
            Specify if a color bar must be drawn beside the plot or not.
            Default to `True`.
        show_time: bool, optional
            Show the timestamp of the field on the plot or not.
            Default to `True`.
        stride: int, optional
            Specify the time step of the animation. Works only with `output` set to `animate`.
        plot_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.
        oro_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method plotting the image of the orography if `style` is set to `contour`.
        anim_kwargs: dict, optional
            Arguments to pass to the :class:`matplotlib.animation.FuncAnimation` instantiation method. Specify the parameters of the animation.
            Works only with `output` set to `show`.
        show: bool, optional
            Whether to plot or not the animation.

        Returns
        -------
        ~matplotlib.animation.FuncAnimation or ~IPython.display.DisplayHandle or callable
            The animation object or the callable to update the widget, depending on the value of the `output` and `show` parameters.
        """

        if output == 'animate':

            if self.diagnostic is None:
                warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
                return None

            if style == 'image':
                vmin = self.diagnostic.min() * 1.03
                vmax = self.diagnostic.max() * 1.03
                if plot_kwargs is not None:
                    if 'vmin' not in plot_kwargs:
                        plot_kwargs['vmin'] = vmin
                    if 'vmax' not in plot_kwargs:
                        plot_kwargs['vmax'] = vmax
                else:
                    plot_kwargs = dict()
                    plot_kwargs['vmin'] = vmin
                    plot_kwargs['vmax'] = vmax

            def update(time_index):
                self.plot(time_index, style, ax, figsize, contour_labels, color_bar, show_time, plot_kwargs, oro_kwargs)
                if show:
                    plt.show()

            if show:
                plot = interactive(update, time_index=(0, len(self)-1, stride))
                anim = display(plot)
            else:
                return update

        elif output == 'show':

            anim = self._make_anim(style, ax, figsize, contour_labels, color_bar, show_time, plot_kwargs, oro_kwargs, anim_kwargs, True)
            if show:
                plt.show()

        else:
            warnings.warn('Provided output parameter ' + output + ' not supported ! Nothing to plot. Returning None.')
            anim = None

        return anim

    def _init_anim(self, style="image", ax=None, figsize=(16, 9), contour_labels=True, color_bar=True,
                   show_time_in_title=False, show_time=True, plot_kwargs=None, oro_kwargs=None, anim_kwargs=None):

        if self.diagnostic is None:
            warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
            return None

        if style == 'image':
            vmin = self.diagnostic.min() * 1.03
            vmax = self.diagnostic.max() * 1.03
            if plot_kwargs is not None:
                if 'vmin' not in plot_kwargs:
                    plot_kwargs['vmin'] = vmin
                if 'vmax' not in plot_kwargs:
                    plot_kwargs['vmax'] = vmax
            else:
                plot_kwargs = dict()
                plot_kwargs['vmin'] = vmin
                plot_kwargs['vmax'] = vmax

        pack = self.plot(0, style, ax, figsize, contour_labels, color_bar, show_time_in_title, plot_kwargs, oro_kwargs)
        if hasattr(pack, '__getitem__'):
            ax = pack[0]
        else:
            ax = pack
        fig = ax.figure

        if show_time:
            if self.dimensional:
                tt = " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[0]) + " " + self._model_params.time_unit
            else:
                tt = " at " + str(self._time[0]) + " timeunits"
            ax.text(0.1, 0.9, tt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        fargs = (ax, show_time_in_title, show_time)

        kwargs = {'style': style, 'ax': ax, 'figsize': figsize, 'contour_labels': contour_labels, 'color_bar': color_bar,
                  'show_time_in_title': show_time_in_title, 'show_time': show_time, 'plot_kwargs': plot_kwargs, 'oro_kwargs': oro_kwargs,
                  'anim_kwargs': anim_kwargs}

        return fig, ax, fargs, kwargs

    def _make_update(self, style="image", ax=None, figsize=(16, 9), contour_labels=True, color_bar=True,
                     show_time_in_title=False, show_time=True, plot_kwargs=None, oro_kwargs=None, anim_kwargs=None):
        def update(i, axe, show_t_in_title, show_t):
            axe.clear()
            axe = self.plot(i, style, axe, figsize, contour_labels, False, show_t_in_title, plot_kwargs, oro_kwargs)
            if show_t:
                if self.dimensional:
                    tt = " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[i]) + " " + self._model_params.time_unit
                else:
                    tt = " at " + str(self._time[i]) + " timeunits"
                axe.text(0.1, 0.9, tt, horizontalalignment='center', verticalalignment='center', transform=axe.transAxes)
            return [axe]

        return update

    def _make_anim(self, style="image", ax=None, figsize=(16, 9), contour_labels=True, color_bar=True,
                   show_time=True, plot_kwargs=None, oro_kwargs=None, anim_kwargs=None, blit=True):

        if show_time:
            if blit:
                show_time_in_title = False
            else:
                show_time_in_title = True
                show_time = False
        else:
            show_time_in_title = False

        fig, ax, fargs, kwargs = self._init_anim(style, ax, figsize, contour_labels, color_bar, show_time_in_title, show_time, plot_kwargs, oro_kwargs, anim_kwargs)

        update = self._make_update(**kwargs)

        if anim_kwargs is not None:

            if 'blit' in anim_kwargs:
                del anim_kwargs['blit']

            anim = animation.FuncAnimation(fig, update, fargs=fargs, blit=blit, **anim_kwargs)

        else:
            anim = animation.FuncAnimation(fig, update, fargs=fargs, blit=blit)

        return anim


class ProfileDiagnostic(Diagnostic):
    """General base class for profile diagnostic on the model's domain.
    Should provide a 1D representation of the fields averages or section.


    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.
    """

    def __init__(self, model_params, dimensional):

        Diagnostic.__init__(self, model_params, dimensional)

        self._points_coordinates = None
        self._plot_label = None
        self._axis_label = None

    # self._default_plot_kwargs = {'cmap': plt.get_cmap('jet'), 'interpolation': 'spline36'}

    def __len__(self):
        if self.diagnostic is not None:
            return self.diagnostic.shape[0]
        else:
            return None

    def plot(self, time_index=0, ax=None, figsize=(16, 9), show_time=True, plot_kwargs=None, **kwargs):
        """Plot the multiple profile diagnostic provided.

        Parameters
        ----------
        time_index: int
            The time index of the data. Not used in this subclass.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        show_time: bool, optional
            Show the timestamp of the field on the plot or not.
            Default to `True`.
        plot_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.

        Returns
        -------
        ~matplotlib.axes.Axes
            An axes where the data were plotted.
        """

        if self.diagnostic is None:
            warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
            return None

        if time_index > self.__len__() - 1:
            warnings.warn('Time index ' + str(time_index) + ' provided is greater than the largest one possible: ' + str(self.__len__()) + ' . Showing nothing.')
            return None

        if plot_kwargs is None:
            plot_kwargs = dict()

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.figure

        if 'label' not in plot_kwargs:
            lab = self._plot_label
            if self.dimensional:
                lab += r" [" + self._plot_units + r"]"
            plot_kwargs = {'label': lab}

        ax.plot(self._points_coordinates, self.diagnostic[time_index], **plot_kwargs)
        ax.legend()

        ax.set_xlabel(self._axis_label)

        title = self._plot_title
        if self.dimensional:
            if show_time:
                title += " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[time_index]) + " " + self._model_params.time_unit
        else:
            title += r" (in nondim units)"
            if show_time:
                title += " at " + str(self._time[time_index]) + " timeunits"

        ax.set_title(title, pad=20.)

        return ax

    def movie(self, output='html', filename='', ax=None, figsize=(16, 9), plot_kwargs=None, anim_kwargs=None):
        """ Create and return a movie of the output of the `plot` method animated over time.
        Show a red dot moving and depicting the current value of the model's selected variables.

        Parameters
        ----------
        output: str, optional
            Define the kind of movie being created. Can be:

            * `jshtml`: Generate an interactive HTML representation of the animation.
            * `html5`: Generate the movie as HTML5 code.
            * `html`: Output the movie as a HTML video tag.
            * `ihtml`: Output the interactive movie as a HTML video tag.
            * `save`: Save the movie in MP4 format (H264 codec).

            Default to `html`.
        filename: str, optional
            Filename (and path) where to save the movie. Needed if `output` is set to `save`.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
            If not provided, create a new one.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        plot_kwargs: dict, optional
            Arguments to pass to the background plot.
        anim_kwargs: dict, optional
            Arguments to pass to the :class:`matplotlib.animation.FuncAnimation` instantiation method. Specify the parameters of the animation.

        Returns
        -------
        ~matplotlib.animation.FuncAnimation or HTML code or HTML tag
            The animation object or the HTML code or tag.
        """

        if self.diagnostic is None:
            warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
            return None

        anim = self._make_anim(ax, figsize, True, plot_kwargs, anim_kwargs, False)

        if 'html' in output:

            if output == "jshtml" or output == 'ihtml':
                jshtml = anim.to_jshtml()
                if output == "jshtml":
                    return jshtml
                else:
                    return HTML(jshtml)
            else:
                html5 = anim.to_html5_video()
                if output == 'html5':
                    return html5
                else:
                    return HTML(html5)

        elif output == 'save':

            if not filename:
                warnings.warn('No filename provided to the method animate. Video not saved !\n Please provide a filename.')

            html = anim.to_html5_video()
            start_index = html.index('base64,')
            start_index += len('base64,')
            end_index = html.index('">', start_index)
            video = html[start_index: end_index]
            with open(filename, 'wb') as f:
                f.write(base64.b64decode(video))

        else:
            warnings.warn('Provided output parameter ' + output + ' not supported ! Nothing to plot. Returning None.')
            anim = None

        return anim

    def animate(self, output='animate', ax=None, figsize=(16, 9), show_time=True, stride=1, plot_kwargs=None, anim_kwargs=None, show=True):
        """Return the output of the `plot` method animated over time.

        Parameters
        ----------
        output: str, optional
            Define the kind of animation being created. Can be:

            * `animate`: Create and show a :class:`ipywidgets.widgets.interaction.interactive` widget. Works only in Jupyter notebooks.
            * `show`: Create and show an animation with the :mod:`matplotlib.animation` module. Works only in IPython or Python.

        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        show_time: bool, optional
            Show the timestamp of the field on the plot or not.
            Default to `True`.
        stride: int, optional
            Specify the time step of the animation. Works only with `output` set to `animate`.
        plot_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.
        anim_kwargs: dict, optional
            Arguments to pass to the :class:`matplotlib.animation.FuncAnimation` instantiation method. Specify the parameters of the animation.
            Works only with `output` set to `show`.
        show: bool, optional
            Whether to plot or not the animation.

        Returns
        -------
        ~matplotlib.animation.FuncAnimation or ~IPython.display.DisplayHandle or callable
            The animation object or the callable to update the widget, depending on the value of the `output` and `show` parameters.
        """

        if self.diagnostic is None:
            warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
            return None

        if output == 'animate':

            heat_max = np.max(self.diagnostic)
            heat_min = np.min(self.diagnostic)

            def update(time_index):
                self.plot(time_index, ax, figsize, show_time, plot_kwargs)
                ax.set_ylim([heat_min, heat_max])
                ax.legend(loc=1)
                if show:
                    plt.show()

            if show:
                plot = interactive(update, time_index=(0, len(self)-1, stride))
                anim = display(plot)
            else:
                return update

        elif output == 'show':

            anim = self._make_anim(ax, figsize, show_time, plot_kwargs, anim_kwargs, True)
            if show:
                plt.show()

        else:
            warnings.warn('Provided output parameter ' + output + ' not supported ! Nothing to plot. Returning None.')
            anim = None

        return anim

    def _init_anim(self, ax=None, figsize=(16, 9), show_time_in_title=False, show_time=True, plot_kwargs=None, anim_kwargs=None):

        if self.diagnostic is None:
            warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
            return None

        heat_max = np.max(self.diagnostic)
        heat_min = np.min(self.diagnostic)

        ax = self.plot(0, ax, figsize, show_time_in_title, plot_kwargs)
        fig = ax.figure
        ax.set_ylim([heat_min, heat_max])
        ax.legend(loc=1)

        if show_time:
            if self.dimensional:
                tt = " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[0]) + " " + self._model_params.time_unit
            else:
                tt = " at " + str(self._time[0]) + " timeunits"
            ax.text(0.1, 0.9, tt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        fargs = (ax, show_time_in_title, show_time)

        kwargs = {'ax': ax, 'figsize': figsize, 'show_time_in_title': show_time_in_title, 'show_time': show_time, 'plot_kwargs': plot_kwargs,
                  'anim_kwargs': anim_kwargs}

        return fig, ax, fargs, kwargs

    def _make_update(self, ax=None, figsize=(16, 9), show_time_in_title=False, show_time=True, plot_kwargs=None, anim_kwargs=None):

        heat_max = np.max(self.diagnostic)
        heat_min = np.min(self.diagnostic)

        def update(i, axe, show_t_in_title, show_t):
            axe.clear()
            axe = self.plot(i, axe, figsize, show_t_in_title, plot_kwargs)
            axe.set_ylim([heat_min, heat_max])
            axe.legend(loc=1)
            if show_t:
                if self.dimensional:
                    tt = " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[i]) + " " + self._model_params.time_unit
                else:
                    tt = " at " + str(self._time[i]) + " timeunits"
                axe.text(0.1, 0.9, tt, horizontalalignment='center', verticalalignment='center', transform=axe.transAxes)
            return [axe]

        return update

    def _make_anim(self, ax=None, figsize=(16, 9), show_time=True, plot_kwargs=None, anim_kwargs=None, blit=True):

        if show_time:
            if blit:
                show_time_in_title = False
            else:
                show_time_in_title = True
                show_time = False
        else:
            show_time_in_title = False

        fig, ax, fargs, kwargs = self._init_anim(ax, figsize, show_time_in_title, show_time, plot_kwargs, anim_kwargs)

        update = self._make_update(**kwargs)

        if anim_kwargs is not None:

            if 'blit' in anim_kwargs:
                del anim_kwargs['blit']

            anim = animation.FuncAnimation(fig, update, fargs=fargs, blit=blit, **anim_kwargs)

        else:
            anim = animation.FuncAnimation(fig, update, fargs=fargs, blit=blit)

        return anim
