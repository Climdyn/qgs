"""
    Multidiagnostic class
    ======================

    This class is used to analyze and plot simultaneously several diagnostic together.

"""
import warnings
import base64
import numpy as np
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from ipywidgets import interactive


class MultiDiagnostic(object):
    """class analyze and plot simultaneously several diagnostic together.
    The diagnostics information are plotted in arrays of fixed dimensions, defining the total number of diagnostics that
    the object can hold.

    Parameters
    ----------
    nrows: int
        The number of rows of diagnostic.
    ncols: int
        The number of columns of diagnostic

    Attributes
    ----------
    figure: ~matplotlib.figure.Figure
            The Matplotlib figure instance where the data are plotted.
    """

    def __init__(self, nrows, ncols):

        self._geometry = (nrows, ncols)
        self._diagnostics_list = list()
        self._diagnostics_position = list()
        self._position_to_index = np.zeros(self._geometry, dtype=int)
        self._position_occupied = np.full(self._geometry, False)
        self._diagnostic_kwargs_list = list()
        self._plot_kwargs_list = list()
        self._figures_array = None
        self._axes_array = None
        self.figure = None

    def __len__(self):
        return self._geometry[0] * self._geometry[1]

    def __call__(self, time, data):
        self.set_data(time, data)

    @property
    def _diagnostic_min_len(self):
        diag_len = list()
        for diag in self._diagnostics_list:
            diag_len.append(diag.__len__())
        try:
            min_len = min(diag_len)
            return min_len
        except:
            return 0

    @property
    def ncols(self):
        """int: The number of columns of diagnostic."""
        return self._geometry[1]

    @property
    def nrows(self):
        """int: The number of rows of diagnostic."""
        return self._geometry[0]

    @property
    def _figure_list(self):
        if self._figures_array is None:
            return None
        else:
            return list(self._figures_array.flatten())

    def set_data(self, time, data):
        """Provide the model data to all the diagnostics.

        Parameters
        ----------
        time: ~numpy.ndarray
            The time (in nondimensional timeunits) corresponding to the data.
            Its length should match the length of the last axis of the provided `data`.
        data: ~numpy.ndarray
            The model output data that the user want to convert using the diagnostic.
        """
        for diagnostic in self._diagnostics_list:
            out = diagnostic(time, data)

    @property
    def diagnostic(self):
        """list(~numpy.ndarray): The output diagnostics as a list."""
        ret = list()
        for diagnostic in self._diagnostics_list:
            diag = diagnostic.diagnostic
            ret.append(diag)
        return ret

    @property
    def diagnostics_list(self):
        """list(Diagnostic): The list of stored diagnostics."""
        return self._diagnostics_list

    @property
    def diagnostic_positions(self):
        """list(tuple(int)): Position occupied by each diagnostic in the plotting array."""
        return self._diagnostics_position

    def add_diagnostic(self, diagnostic, position=None, diagnostic_kwargs=None, plot_kwargs=None):
        """Method to add a diagnostic to the list.

        Parameters
        ----------
        diagnostic: Diagnostic
            Diagnostic to add.
        position: tuple(int), optional
            2-tuple specifying the position of the diagnostic in the plotting array. Find a *free* spot in the array if not specified.
            If the plotting array is full, a position **must** be provided to overwrite a already defined diagnostic.
        diagnostic_kwargs: dict, optional
            Dictionary of arguments to pass to the `plot`, `animate` and `movie` method of the provided diagnostic.
        plot_kwargs: dict, optional
            Specific `plot_kwargs` argument to pass to the `plot`, `animate` and `movie` method of the provided diagnostic.
            If provided, overwrite the one possibly present in the `diagnostic_kwargs` above.
        """
        if position is None:
            for i, j in product(range(self._geometry[0]), range(self._geometry[1])):
                if not self._position_occupied[i, j]:
                    position = (i, j)
                    break
            else:
                warnings.warn('All diagnostic position already occupied. Please specify the position argument to overwrite.')
                return None

            self._diagnostics_list.append(diagnostic)
            self._diagnostic_kwargs_list.append(diagnostic_kwargs)
            self._plot_kwargs_list.append(plot_kwargs)
            self._diagnostics_position.append(position)
            self._position_occupied[position[0], position[1]] = True
            self._position_to_index[position[0], position[1]] = len(self._diagnostics_list)-1
        else:
            index = self._position_to_index[position[0], position[1]]
            self._diagnostics_list[index] = diagnostic
            self._diagnostic_kwargs_list[index] = diagnostic_kwargs
            self._plot_kwargs_list[index] = plot_kwargs

    def plot(self, time_index, figure=None, figsize=(16, 9), tight_layout=False):
        """Plot the fields of the provided data at the given time index.

        Parameters
        ----------

        time_index: int
            The time index of the data.
        figure: ~matplotlib.figure.Figure, optional
            The Matplotlib figure instance where the data are plotted. Update the `figure` attribute of the object.
            If not provided, use the `figure` attribute of the object.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple. Used only if a new figure must be created.
        tight_layout: bool, optional
            Enforce a tight layout of the diagnostics axes.

        """

        if figure is None:
            self.figure = plt.figure(figsize=figsize)
            self._figures_array = self.figure.subfigures(*self._geometry, squeeze=False)
            self._axes_array = np.empty_like(self._figures_array)
            for i in range(self._geometry[0]):
                for j in range(self._geometry[1]):
                    index = self._position_to_index[i, j]
                    kwargs = self._diagnostic_kwargs_list[index]
                    if kwargs is None:
                        kwargs = dict()
                    if 'style' in kwargs:
                        if '3D' in kwargs['style']:
                            self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1, projection='3d')
                        else:
                            self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1)
                    else:
                        self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1)
        elif figure is False:
            pass
        else:
            self.figure = figure

        for i, diagnostic in enumerate(self._diagnostics_list):
            position = self._diagnostics_position[i]
            ax = self._axes_array[position[0], position[1]]
            if self._diagnostic_kwargs_list[i] is not None:
                kwargs = self._diagnostic_kwargs_list[i]
                try:
                    del kwargs['ax']
                except:
                    pass
            else:
                kwargs = dict()
            if self._plot_kwargs_list[i] is not None:
                kwargs['plot_kwargs'] = self._plot_kwargs_list[i]
            diagnostic.plot(time_index, ax=ax, **kwargs)
        if tight_layout:
            self.figure.tight_layout()

    def animate(self, output='animate', figure=None, figsize=(16, 9), stride=1, anim_kwargs=None):
        """Return the output of the `plot` method animated over time.

        Parameters
        ----------
        output: str, optional
            Define the kind of animation being created. Can be:

            * `animate`: Create and show a :class:`ipywidgets.widgets.interaction.interactive` widget. Works only in Jupyter notebooks.
            * `show`: Create and show an animation with the :mod:`matplotlib.animation` module. Works only in IPython or Python.

        figure: ~matplotlib.figure.Figure, optional
            The Matplotlib figure instance where the data are plotted. Update the `figure` attribute of the object.
            If not provided, use the `figure` attribute of the object.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple. Used only if a new figure must be created.
        stride: int, optional
            Specify the time step of the animation. Works only with `output` set to `animate`.
        anim_kwargs: dict, optional
            Arguments to pass to the :class:`matplotlib.animation.FuncAnimation` instantiation method. Specify the parameters of the animation.
            Works only with `output` set to `show`.

        Returns
        -------
        ~matplotlib.animation.FuncAnimation or ~IPython.display.DisplayHandle
            The animation object.
        """

        if output == 'animate':
            kwargs_list = list()

            for j, diagnostic in enumerate(self._diagnostics_list):
                if self._diagnostic_kwargs_list[j] is not None:
                    kwargs_list.append(self._diagnostic_kwargs_list[j].copy())
                else:
                    kwargs_list.append(dict())
                kwargs_list[-1]['output'] = output
                kwargs_list[-1]['show'] = False
                if self._plot_kwargs_list[j] is not None:
                    kwargs_list[-1]['plot_kwargs'] = self._plot_kwargs_list[j]
                if anim_kwargs is not None:
                    kwargs_list[-1]['anim_kwargs'] = anim_kwargs

            def update_tot(time_index):
                ## no subfigure clf, so we recreate the figure all the time
                self.figure = plt.figure(figsize=figsize)
                self._figures_array = self.figure.subfigures(*self._geometry, squeeze=False)
                self._axes_array = np.empty_like(self._figures_array)
                for k in range(self._geometry[0]):
                    for j in range(self._geometry[1]):
                        index = self._position_to_index[k, j]
                        kwargs = self._diagnostic_kwargs_list[index]
                        if kwargs is None:
                            kwargs = dict()
                        if 'style' in kwargs:
                            if '3D' in kwargs['style']:
                                self._axes_array[k, j] = self._figures_array[k, j].add_subplot(1, 1, 1, projection='3d')
                            else:
                                self._axes_array[k, j] = self._figures_array[k, j].add_subplot(1, 1, 1)
                        else:
                            self._axes_array[k, j] = self._figures_array[k, j].add_subplot(1, 1, 1)

                for j, diagnostic in enumerate(self._diagnostics_list):
                    position = self._diagnostics_position[j]
                    ax = self._axes_array[position[0], position[1]]
                    kwargs_list[j]['ax'] = ax

                    update = diagnostic.animate(**kwargs_list[j])
                    update(time_index)
                plt.show()

            plot = interactive(update_tot, time_index=(0, self._diagnostic_min_len-1, stride))
            anim = display(plot)

        elif output == 'show':

            if figure is None:
                self.figure = plt.figure(figsize=figsize)
                self._figures_array = self.figure.subfigures(*self._geometry, squeeze=False)
                self._axes_array = np.empty_like(self._figures_array)
                for i in range(self._geometry[0]):
                    for j in range(self._geometry[1]):
                        index = self._position_to_index[i, j]
                        kwargs = self._diagnostic_kwargs_list[index]
                        if kwargs is None:
                            kwargs = dict()
                        if 'style' in kwargs:
                            if '3D' in kwargs['style']:
                                self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1, projection='3d')
                            else:
                                self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1)
                        else:
                            self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1)
            elif figure is False:
                pass
            else:
                self.figure = figure

            fargs_list = list()
            update_list = list()

            for j, diagnostic in enumerate(self._diagnostics_list):
                position = self._diagnostics_position[j]
                ax = self._axes_array[position[0], position[1]]
                if self._diagnostic_kwargs_list[j] is not None:
                    kwargs = self._diagnostic_kwargs_list[j].copy()
                    try:
                        del kwargs['output']
                    except:
                        pass
                    try:
                        del kwargs['show']
                    except:
                        pass
                else:
                    kwargs = dict()
                kwargs['ax'] = ax
                if self._plot_kwargs_list[j] is not None:
                    kwargs['plot_kwargs'] = self._plot_kwargs_list[j]
                if anim_kwargs is not None:
                    kwargs['anim_kwargs'] = anim_kwargs
                fig, axe, fargs, kwargs = diagnostic._init_anim(**kwargs)
                update = diagnostic._make_update(**kwargs)
                fargs_list.append(fargs)
                update_list.append(update)

            def update_tot(i, update_lst, fargs_lst):
                ax_list = list()
                for up, fags in zip(update_lst, fargs_lst):
                    out = up(i, *fags)
                    ax_list.append(out[0])
                return ax_list

            if anim_kwargs is not None:

                if 'blit' in anim_kwargs:
                    del anim_kwargs['blit']

                anim = animation.FuncAnimation(self.figure, update_tot, fargs=[update_list, fargs_list], blit=True, **anim_kwargs)

            else:
                anim = animation.FuncAnimation(self.figure, update_tot, fargs=[update_list, fargs_list], blit=True)

            plt.show()

        else:
            warnings.warn('Provided output parameter ' + output + ' not supported ! Nothing to plot. Returning None.')
            anim = None

        return anim

    def movie(self, output='html', filename='', figure=None, figsize=(16, 9), anim_kwargs=None):
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
        figure: ~matplotlib.figure.Figure, optional
            The Matplotlib figure instance where the data are plotted. Update the `figure` attribute of the object.
            If not provided, use the `figure` attribute of the object.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple. Used only if a new figure must be created.
        anim_kwargs: dict, optional
            Arguments to pass to the :class:`matplotlib.animation.FuncAnimation` instantiation method. Specify the parameters of the animation.
            Works only with `output` set to `show`.

        Returns
        -------
        ~matplotlib.animation.FuncAnimation or HTML code or HTML tag
            The animation object or the HTML code or tag.
        """

        if figure is None:
            self.figure = plt.figure(figsize=figsize)
            self._figures_array = self.figure.subfigures(*self._geometry, squeeze=False)
            self._axes_array = np.empty_like(self._figures_array)
            for i in range(self._geometry[0]):
                for j in range(self._geometry[1]):
                    index = self._position_to_index[i, j]
                    kwargs = self._diagnostic_kwargs_list[index]
                    if kwargs is None:
                        kwargs = dict()
                    if 'style' in kwargs:
                        if '3D' in kwargs['style']:
                            self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1, projection='3d')
                        else:
                            self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1)
                    else:
                        self._axes_array[i, j] = self._figures_array[i, j].add_subplot(1, 1, 1)
        elif figure is False:
            pass
        else:
            self.figure = figure

        fargs_list = list()
        update_list = list()

        for j, diagnostic in enumerate(self._diagnostics_list):
            position = self._diagnostics_position[j]
            ax = self._axes_array[position[0], position[1]]
            if self._diagnostic_kwargs_list[j] is not None:
                kwargs = self._diagnostic_kwargs_list[j].copy()
                try:
                    del kwargs['output']
                except:
                    pass
                try:
                    del kwargs['show']
                except:
                    pass
            else:
                kwargs = dict()
            kwargs['ax'] = ax
            if self._plot_kwargs_list[j] is not None:
                kwargs['plot_kwargs'] = self._plot_kwargs_list[j]
            if anim_kwargs is not None:
                kwargs['anim_kwargs'] = anim_kwargs
            if 'show_time' in kwargs:
                if not kwargs['show_time']:
                    kwargs['show_time_in_title'] = False
                else:
                    kwargs['show_time_in_title'] = True
                    kwargs['show_time'] = False
            else:
                kwargs['show_time_in_title'] = True
                kwargs['show_time'] = False

            fig, axe, fargs, kwargs = diagnostic._init_anim(**kwargs)
            update = diagnostic._make_update(**kwargs)
            fargs_list.append(fargs)
            update_list.append(update)

        def update_tot(i, update_lst, fargs_lst):
            ax_list = list()
            for up, fags in zip(update_lst, fargs_lst):
                out = up(i, *fags)
                ax_list.append(out[0])
            return ax_list

        if anim_kwargs is not None:

            if 'blit' in anim_kwargs:
                del anim_kwargs['blit']

            anim = animation.FuncAnimation(self.figure, update_tot, fargs=[update_list, fargs_list], blit=False, **anim_kwargs)

        else:
            anim = animation.FuncAnimation(self.figure, update_tot, fargs=[update_list, fargs_list], blit=False)

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


class FieldsDiagnosticsList(object):
    """General base class for plotting multiple diagnostics on a single axe. The diagnostics must be provided as a list.
    Assumes that the first diagnostic in the list set the parameters that are not specified.

    Parameters
    ----------
    diagnostics_list: list
        List of initialized auxialiary diagnostics to plot with the main diagnostic.
    """

    def __init__(self, diagnostics_list=None):

        if diagnostics_list is not None:
            self._diagnostics_list = diagnostics_list
        else:
            self._diagnostics_list = list()

    def append_diagnostic(self, diagnostic):
        """Method to add an auxiliary diagnostic to the list.

        Parameters
        ----------
        diagnostic: Diagnostic
            The diagnostic to add to the list.
        """
        self._diagnostics_list.append(diagnostic)

    def __call__(self, time, data, index=None):
        self.set_data(time, data, index)

    def __len__(self):
        diag_len = list()
        for diag in self._diagnostics_list:
            diag_len.append(diag.__len__())
        try:
            min_len = min(diag_len)
            return min_len
        except:
            return 0

    def set_data(self, time, data, index=None):
        """Provide the model data to the index-th diagnostic.

        Parameters
        ----------
        time: ~numpy.ndarray
            The time (in nondimensional timeunits) corresponding to the data.
            Its length should match the length of the last axis of the provided `data`.
        data: ~numpy.ndarray
            The model output data that the user want to convert using the diagnostic.
            Should be a 2D array of shape (:attr:`~.params.QgParams.ndim`, number_of_timesteps).
        index: int or None
            The index of the diagnostic in the list to provide the data to.
        """
        if self._diagnostics_list is None:
            warnings.warn('No diagnostics available. Doing nothing.')
            return None

        if index is None:
            return None
        else:
            self._diagnostics_list[index].set_data(time, data)

    def plot(self, time_index, style="image", ax=None, figsize=(16, 9),
             contour_labels=True, color_bar=True, show_time=True, plot_kwargs=None, oro_kwargs=None):
        """Plot the field of the provided diagnostics at the given time index.
        Almost all the parameters (except `ax` and `figsize`) and fig should be lists corresponding to diagnostics in the list.
        If a single parameter is provided, it applies to all the diagnostics.

        Parameters
        ----------
        time_index: list(int)
            The time index of the data.
        style: list(str), optional
            The style of the plot. Can be:

            * `image`: show the fields as images with a given colormap specified in the `plot_kwargs` argument.
            * `contour`: show the fields as contour superimposed on the image of the orographic height (if it exists, see the `oro_kwargs` below).
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        contour_labels: list(bool), optional
            If `style` is set to `contour`, specify if the contours must be labelled with their value or not.
            Default to `True`.
        color_bar: list(bool), optional
            Specify if a color bar must be drawn beside the plot or not.
            Default to `True`.
        show_time: list(bool), optional
            Show the timestamp of the field on the plot or not.
            Default to `True`.
        plot_kwargs: list(dict), optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.
        oro_kwargs: list(dict), optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method plotting the image of the orography if `style` is set to `contour`.

        Returns
        -------
        ~matplotlib.axes.Axes
            An axes where the data were plotted.
        """

        if self._diagnostics_list is None:
            warnings.warn('No diagnostics available. Showing nothing.')
            return None

        if not isinstance(time_index, (list, tuple)):
            time_index = (len(self._diagnostics_list)) * [time_index]

        if not isinstance(style, (list, tuple)):
            style = (len(self._diagnostics_list)) * [style]

        if not isinstance(contour_labels, (list, tuple)):
            contour_labels = (len(self._diagnostics_list)) * [contour_labels]

        if not isinstance(color_bar, (list, tuple)):
            color_bar = (len(self._diagnostics_list)) * [color_bar]

        if not isinstance(show_time, (list, tuple)):
            show_time = (len(self._diagnostics_list)) * [show_time]

        if not isinstance(plot_kwargs, (list, tuple)):
            plot_kwargs = (len(self._diagnostics_list)) * [plot_kwargs]

        if not isinstance(oro_kwargs, (list, tuple)):
            oro_kwargs = (len(self._diagnostics_list)) * [oro_kwargs]

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()

        for j, diagnostic in enumerate(self._diagnostics_list):
            diagnostic.plot(time_index[j], style[j], ax, figsize, contour_labels[j], color_bar[j], show_time[j], plot_kwargs[j], oro_kwargs[j])

        return ax

    def movie(self, output='html', filename='', style="image", ax=None, figsize=(16, 9),
              contour_labels=True, color_bar=True, show_time=True, plot_kwargs=None, oro_kwargs=None, anim_kwargs=None):
        """ Create and return a movie of the output of the `plot` method animated over time.
        Almost all the parameters (except `output`, `filename`, `ax`, `figsize`, and `anim_kwargs`) and fig should be lists corresponding to diagnostics in the list.
        If a single parameter is provided, it applies to all the diagnostics.

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
        style: lits(str), optional
            The style of the plot. Can be:

            * `image`: show the fields as images with a given colormap specified in the `plot_kwargs` argument.
            * `contour`: show the fields as contour superimposed on the image of the orographic height (see the `oro_kwargs` below).
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        contour_labels: list(bool), optional
            If `style` is set to `contour`, specify if the contours must be labelled with their value or not.
            Default to `True`.
        color_bar: list(bool), optional
            Specify if a color bar must be drawn beside the plot or not.
            Default to `True`.
        show_time: list(bool), optional
            Show the timestamp of the field on the plot or not.
            Default to `True`.
        plot_kwargs: list(dict), optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.
        oro_kwargs: list(dict), optional
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
        Almost all the parameters (except `animate`, `ax`, `figsize`, `stride`, `anim_kwargs` and `show`) and fig should be lists corresponding to diagnostics in the list.
        If a single parameter is provided, it applies to all the diagnostics.

        Parameters
        ----------
        output: str, optional
            Define the kind of animation being created. Can be:

            * `animate`: Create and show a :class:`ipywidgets.widgets.interaction.interactive` widget. Works only in Jupyter notebooks.
            * `show`: Create and show an animation with the :mod:`matplotlib.animation` module. Works only in IPython or Python.

        style: list(str), optional
            The style of the plot. Can be:

            * `image`: show the fields as images with a given colormap specified in the `plot_kwargs` argument.
            * `contour`: show the fields as contour superimposed on the image of the orographic height (see the `oro_kwargs` below).

        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        contour_labels: list(bool), optional
            If `style` is set to `contour`, specify if the contours must be labelled with their value or not.
            Default to `True`.
        color_bar: list(bool), optional
            Specify if a color bar must be drawn beside the plot or not.
            Default to `True`.
        show_time: list(bool), optional
            Show the timestamp of the field on the plot or not.
            Default to `True`.
        stride: int, optional
            Specify the time step of the animation. Works only with `output` set to `animate`.
        plot_kwargs: list(dict), optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.
        oro_kwargs: list(dict), optional
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

        if self._diagnostics_list is None:
            warnings.warn('No diagnostic available. Showing nothing. Returning None.')
            return None

        if not isinstance(style, (list, tuple)):
            style = (len(self._diagnostics_list)) * [style]

        if not isinstance(contour_labels, (list, tuple)):
            contour_labels = (len(self._diagnostics_list)) * [contour_labels]

        if not isinstance(color_bar, (list, tuple)):
            color_bar = (len(self._diagnostics_list)) * [color_bar]

        if not isinstance(show_time, (list, tuple)):
            show_time = (len(self._diagnostics_list)) * [show_time]

        if not isinstance(plot_kwargs, (list, tuple)):
            plot_kwargs = (len(self._diagnostics_list)) * [plot_kwargs]

        if not isinstance(oro_kwargs, (list, tuple)):
            oro_kwargs = (len(self._diagnostics_list)) * [oro_kwargs]

        if output == 'animate':

            if style == 'image':

                for i, diagnostic in enumerate(self._diagnostics_list):

                    vmin = diagnostic.min() * 1.03
                    vmax = diagnostic.max() * 1.03
                    if plot_kwargs[i] is not None:
                        if 'vmin' not in plot_kwargs[i]:
                            plot_kwargs[i]['vmin'] = vmin
                        if 'vmax' not in plot_kwargs[i]:
                            plot_kwargs[i]['vmax'] = vmax
                    else:
                        plot_kwargs[i] = dict()
                        plot_kwargs[i]['vmin'] = vmin
                        plot_kwargs[i]['vmax'] = vmax

            def update(time_index):
                axe = None
                for i, diagnostic in enumerate(self._diagnostics_list):
                    if i == 0:
                        pack = diagnostic.plot(time_index, style[i], ax, figsize, contour_labels[i], color_bar[i], show_time[i], plot_kwargs[i], oro_kwargs[i])
                        if hasattr(pack, '__getitem__'):
                            axe = pack[0]
                        else:
                            axe = pack
                    else:
                        diagnostic.plot(time_index, style[i], axe, figsize, contour_labels[i], color_bar[i], show_time[i], plot_kwargs[i], oro_kwargs[i])
                if show:
                    plt.show()

            if show:
                plot = interactive(update, time_index=(0, len(self._diagnostics_list[0])-1, stride))
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

        if self._diagnostics_list is None:
            warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
            return None

        if not isinstance(style, (list, tuple)):
            style = (len(self._diagnostics_list)) * [style]

        if not isinstance(contour_labels, (list, tuple)):
            contour_labels = (len(self._diagnostics_list)) * [contour_labels]

        if not isinstance(color_bar, (list, tuple)):
            color_bar = (len(self._diagnostics_list)) * [color_bar]

        if not isinstance(show_time, (list, tuple)):
            show_time = (len(self._diagnostics_list)) * [show_time]

        if not isinstance(plot_kwargs, (list, tuple)):
            plot_kwargs = (len(self._diagnostics_list)) * [plot_kwargs]

        if not isinstance(oro_kwargs, (list, tuple)):
            oro_kwargs = (len(self._diagnostics_list)) * [oro_kwargs]

        if style == 'image':

            for i, diagnostic in enumerate(self._diagnostics_list):

                vmin = diagnostic.min() * 1.03
                vmax = diagnostic.max() * 1.03
                if plot_kwargs[i] is not None:
                    if 'vmin' not in plot_kwargs[i]:
                        plot_kwargs[i]['vmin'] = vmin
                    if 'vmax' not in plot_kwargs[i]:
                        plot_kwargs[i]['vmax'] = vmax
                else:
                    plot_kwargs[i] = dict()
                    plot_kwargs[i]['vmin'] = vmin
                    plot_kwargs[i]['vmax'] = vmax

        for i, diagnostic in enumerate(self._diagnostics_list):
            pack = diagnostic.plot(0, style[i], ax, figsize, contour_labels[i], color_bar[i], show_time_in_title, plot_kwargs[i], oro_kwargs[i])
            if hasattr(pack, '__getitem__'):
                ax = pack[0]
            else:
                ax = pack
            fig = ax.figure

            if show_time[i]:
                if self._diagnostics_list[i].dimensional:
                    tt = " at " + "{:.2f}".format(self._diagnostics_list[i]._model_params.dimensional_time * self._diagnostics_list[i]._time[0]) \
                         + " " + self._diagnostics_list[i]._model_params.time_unit
                else:
                    tt = " at " + str(self._diagnostics_list[i]._time[0]) + " timeunits"
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
            for j, diagnostic in enumerate(self._diagnostics_list):
                axe = diagnostic.plot(i, style[j], axe, figsize, contour_labels[j], False, show_t_in_title, plot_kwargs[j], oro_kwargs[j])
                if show_t[j]:
                    if self._diagnostics_list[j].dimensional:
                        tt = " at " + "{:.2f}".format(self._diagnostics_list[j]._model_params.dimensional_time * self._diagnostics_list[j]._time[i]) \
                             + " " + self._diagnostics_list[j]._model_params.time_unit
                    else:
                        tt = " at " + str(self._diagnostics_list[j]._time[i]) + " timeunits"
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


if __name__ == '__main__':
    from qgs.params.params import QgParams
    from qgs.params.params import QgParams
    from qgs.integrators.integrator import RungeKuttaIntegrator
    from qgs.functions.tendencies import create_tendencies
    from qgs.diagnostics.streamfunctions import LowerLayerAtmosphericStreamfunctionDiagnostic, UpperLayerAtmosphericStreamfunctionDiagnostic, \
        MiddleAtmosphericStreamfunctionDiagnostic
    from qgs.diagnostics.temperatures import MiddleAtmosphericTemperatureDiagnostic
    from qgs.diagnostics.variables import VariablesDiagnostic, GeopotentialHeightDifferenceDiagnostic

    pars = QgParams()
    pars.set_atmospheric_channel_fourier_modes(2, 2)
    f, Df = create_tendencies(pars)
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)
    ic = np.random.rand(pars.ndim) * 0.1
    integrator.integrate(0., 200000., 0.1, ic=ic, write_steps=5)
    time, traj = integrator.get_trajectories()
    integrator.terminate()

    psi3 = LowerLayerAtmosphericStreamfunctionDiagnostic(pars)
    psi1 = UpperLayerAtmosphericStreamfunctionDiagnostic(pars)
    psi = MiddleAtmosphericStreamfunctionDiagnostic(pars)
    theta = MiddleAtmosphericTemperatureDiagnostic(pars)

    var_nondim = VariablesDiagnostic([1, 2, 0], pars, False)
    geo_dim = GeopotentialHeightDifferenceDiagnostic([[[np.pi/pars.scale_params.n, np.pi/4], [np.pi/pars.scale_params.n, 3*np.pi/4]],
                                                      [[0, np.pi/4], [0, 3*np.pi/4]]],
                                                     pars, True)

    kw = {'frames': 100, 'interval': 100, 'blit': False}
    pkw = {'ms': 0.1}

    m = MultiDiagnostic(2, 2)
    m.add_diagnostic(var_nondim, diagnostic_kwargs={'style':'2Dscatter'}, plot_kwargs=pkw)
    # m.add_diagnostic(geo_dim, diagnostic_kwargs={'style': 'moving-timeserie'}, plot_kwargs=pkw)
    m.add_diagnostic(psi1)
    m.add_diagnostic(psi3)
    m.add_diagnostic(theta)
    m(time, traj)
