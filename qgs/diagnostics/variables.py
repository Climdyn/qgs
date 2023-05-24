"""
    Diagnostic variables classes
    =============================

    Classes defining multiple scalar diagnostics (variables) of the model and used to analyze its outputs.

    Description of the classes
    --------------------------

    * :class:`VariablesDiagnostic`: General class to get and show the scalar variables of the models.
    * :class:`GeopotentialHeightDifferenceDiagnostic`: Class to compute and show the geopotential height difference
      between points of the model's domain.

"""
import warnings
import base64
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from ipywidgets import interactive

from qgs.diagnostics.base import Diagnostic

from qgs.diagnostics.misc import time_axis_label


class VariablesDiagnostic(Diagnostic):
    """General class to create multiple scalar diagnostics based on the variables of the model.

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

    def __init__(self, variable_list, model_params, dimensional):

        Diagnostic.__init__(self, model_params, dimensional)
        self._variable_list = variable_list
        self._variable_labels = [self._model_params.latex_var_string[var] for var in self._variable_list]
        self._variable_units = [self._model_params.get_variable_units(var) for var in self._variable_list]
        self._plot_title = "Model's variables"

    def __len__(self):
        if self.diagnostic is not None:
            return self.diagnostic.shape[1]
        else:
            return None

    def _configure(self, **kwargs):
        pass

    def _get_diagnostic(self, dimensional):
        if dimensional:
            vr = self._model_params.variables_range
            for i, j in enumerate(self._variable_list):
                v = self._data[j].copy()
                if j < vr[0]:
                    v *= self._model_params.streamfunction_scaling
                if vr[0] <= j < vr[1]:
                    v *= self._model_params.temperature_scaling * 2
                if self._ocean:
                    if vr[1] <= j < vr[2]:
                        v *= self._model_params.streamfunction_scaling
                    if vr[2] <= j < vr[3]:
                        v *= self._model_params.temperature_scaling
                if self._ground:
                    if vr[1] <= j < vr[2]:
                        v *= self._model_params.temperature_scaling

                if i == 0:
                    self._diagnostic_data = v[np.newaxis, :]
                else:
                    self._diagnostic_data = np.vstack((self._diagnostic_data, v))
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = self._data[self._variable_list]
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data

    def plot(self, time_index=0, variables='all', style="timeserie", ax=None, figsize=(16, 9), plot_kwargs=None, **kwargs):
        """Plot the multiple scalar diagnostic provided.

        Parameters
        ----------
        time_index: int
            The time index of the data. Not used in this subclass.
        variables: str or list(int)
            List of the model variables to consider as diagnostics.
            Default to `all`, i.e. select all the variables of the model.
        style: str, optional
            The style of the plot. Can be:

            * `timeserie`: Plot all the selected variables as a function of time.
            * `2Dscatter`: Plot the first two selected variables on a 2D scatter plot.
            * `3Dscatter`: Plot the first three selected variables on a 3D scatter plot.

        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot the fields.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        plot_kwargs: dict, optional
            Arguments to pass to the :meth:`matplotlib.axes.Axes.imshow` method if `style` is set to `image`, or to the :meth:`matplotlib.axes.Axes.contour` method if `style` is set to `contour`.

        Returns
        -------
        ~matplotlib.axes.Axes
            An axes where the data were plotted.
        """

        if variables == 'all':
            variables = list(range(len(self._variable_list)))

        if self.diagnostic is None:
            warnings.warn('No diagnostic data available. Showing nothing. Returning None.')
            return None

        if plot_kwargs is None:
            plot_kwargs = dict()

        if ax is None:
            fig = plt.figure(figsize=figsize)
            if '3D' in style:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.figure

        if style == 'timeserie':

            for var in variables:
                lab = '$' + self._variable_labels[var] + '$'
                if self.dimensional:
                    lab += r" [" + self._variable_units[var] + r"]"
                time = self._time * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time)
                ax.plot(time, self.diagnostic[var], label=lab, **plot_kwargs)
            ax.legend()

            if self.dimensional:
                ax.set_xlabel(time_axis_label + ' [' + self._model_params.time_unit + ']')
            else:
                ax.set_xlabel(time_axis_label + ' [timeunits]')

        elif style == "2Dscatter":
            ax.plot(self.diagnostic[variables[0]], self.diagnostic[variables[1]], marker='o', ls='', **plot_kwargs)

            # TODO: Still to test (and adapt to variables ranges)
            # if self._ocean:
            #     natm = self._model_params.nmod[0]
            #     ngoc = self._model_params.nmod[1]
            #     if 2*natm <= variables[0] < 2*natm+ngoc:
            #         tl = ax.get_xticklabels()
            #         labto, num = _tick_fmt(tl)
            #         ax.set_xticklabels(labto)
            #         til = ax.xaxis.get_major_ticks()
            #         for j in range(0, len(til), 1):
            #             til[j].label1.set_visible(False)
            #         til[num].label1.set_visible(True)
            #         til[0].label1.set_visible(True)

            lab = '$' + self._variable_labels[0] + '$'
            if self.dimensional:
                lab += r" [" + self._variable_units[0] + r"]"
            ax.set_xlabel(lab)
            lab = '$' + self._variable_labels[1] + '$'
            if self.dimensional:
                lab += r" [" + self._variable_units[1] + r"]"
            ax.set_ylabel(lab)

        elif style == "3Dscatter":
            ax.plot(self.diagnostic[variables[0]], self.diagnostic[variables[1]], self.diagnostic[variables[2]], marker='o', ls='', **plot_kwargs)

            lab = '$' + self._variable_labels[0] + '$'
            if self.dimensional:
                lab += r" [" + self._variable_units[0] + r"]"
            ax.set_xlabel(lab, labelpad=20.)
            lab = '$' + self._variable_labels[1] + '$'
            if self.dimensional:
                lab += r" [" + self._variable_units[1] + r"]"
            ax.set_ylabel(lab, labelpad=20.)
            lab = '$' + self._variable_labels[2] + '$'
            if self.dimensional:
                lab += r" [" + self._variable_units[2] + r"]"
            ax.set_zlabel(lab, labelpad=20.)

        else:
            warnings.warn('Provided style parameter ' + style + ' not supported ! Nothing to plot.')

        title = self._plot_title
        if self.dimensional:
            pass
        else:
            title += r" (in nondim units)"

        ax.set_title(title, pad=20.)

        return ax

    def movie(self, variables='all', output='html', filename='', style="2Dscatter", background=None, ax=None, figsize=(16, 9),
              plot_kwargs=None, anim_kwargs=None):
        """ Create and return a movie of the output of the `plot` method animated over time.
        Show a red dot moving and depicting the current value of the model's selected variables.

        Parameters
        ----------
        variables: str or list(int)
            List of the model variables to consider as diagnostics.
            Default to `all`, i.e. select all the variables of the model.
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

            * `timeserie`: Plot all the selected variables as a function of time.
            * `moving-timeserie`: Plot all the selected variables as a function of time. Draw the lines as the time evolves.
            * `2Dscatter`: Plot the first two selected variables on a 2D scatter plot.
            * `3Dscatter`: Plot the first three selected variables on a 3D scatter plot.

        background: VariablesDiagnostic, optional
            The variables diagnostic data used as background for the evolving red dot. If `None`, use the current
            VariablesDiagnostic instance. Default to `None`.
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

        anim = self._make_anim(variables, style, background, ax, figsize, True, plot_kwargs, anim_kwargs, False)

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

    def animate(self, variables='all', output='animate', style="2Dscatter", background=None, ax=None, figsize=(16, 9), show_time=True,
                stride=1, plot_kwargs=None, anim_kwargs=None, show=True):
        """Return the output of the `plot` method animated over time.
        Show a red dot moving and depicting the current value of the model's selected variables.

        Parameters
        ----------
        variables: str or list(int)
            List of the model variables to consider as diagnostics.
            Default to `all`, i.e. select all the variables of the model.
        output: str, optional
            Define the kind of animation being created. Can be:

            * `animate`: Create and show a :class:`ipywidgets.widgets.interaction.interactive` widget. Works only in Jupyter notebooks.
            * `show`: Create and show an animation with the :mod:`matplotlib.animation` module. Works only in IPython or Python.

        style: str, optional
            The style of the plot. Can be:

            * `timeserie`: Plot all the selected variables as a function of time.
            * `moving-timeserie`: Plot all the selected variables as a function of time. Draw the lines as the time evolves.
            * `2Dscatter`: Plot the first two selected variables on a 2D scatter plot.
            * `3Dscatter`: Plot the first three selected variables on a 3D scatter plot.

        background: VariablesDiagnostic, optional
            The variables diagnostic data used as background for the evolving red dot. If `None`, use the current
            VariablesDiagnostic instance. Default to `None`.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
            If not provided, create a new one.
        figsize: tuple(float), optional
            The size of the figure in inches as a 2-tuple.
        show_time: bool, optional
            Show the timestamp on the plot or not. Only valid for scatter plots.
            Default to `True`.
        stride: int, optional
            Specify the time step of the animation. Works only with `output` set to `animate`.
        plot_kwargs: dict, optional
            Arguments to pass to the background plot.
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

            if background is None:
                background = self

            if variables == 'all':
                variables = list(range(len(self._variable_list)))

            def update(time_index):
                if 'moving' not in style:
                    pack = background.plot(variables=variables, style=style, ax=ax, figsize=figsize, plot_kwargs=plot_kwargs)
                    if hasattr(pack, '__getitem__'):
                        axe = pack[0]
                    else:
                        axe = pack
                else:
                    if ax is None:
                        fig = plt.figure(figsize=figsize)
                        if '3D' in style:
                            axe = fig.add_subplot(1, 1, 1, projection='3d')
                        else:
                            axe = fig.add_subplot(1, 1, 1)
                    else:
                        axe = ax
                        fig = axe.figure

                if style == 'timeserie':

                    lines = ax.get_lines()
                    time = self._time[time_index:time_index+1] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time)
                    for i in range(len(variables)):
                        ax.plot(time, self.diagnostic[variables[i]][time_index:time_index+1], marker='o', linestyle='', color=lines[i].get_c())

                elif style == 'moving-timeserie':

                    ii = -1
                    lines = list()
                    var_min = 2.e168
                    var_max = -2.e168
                    end_time = -2.e168
                    start_time = 2.e168
                    for i in range(len(variables)):
                        var_min = min(var_min, np.min(self.diagnostic[i]))
                        var_max = max(var_max, np.max(self.diagnostic[i]))
                        end_time = max(end_time, self._time[ii] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time))
                        start_time = min(start_time, self._time[0] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time))
                        time = self._time[:time_index+1] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time)
                        lab = '$' + self._variable_labels[i] + '$'
                        if self.dimensional:
                            lab += r" [" + self._variable_units[i] + r"]"
                        line, = ax.plot(time, self.diagnostic[variables[i]][:time_index+1], label=lab)  # , **plot_kwargs
                        lines.append(line)
                    if var_min < 0.:
                        var_min *= 1.03
                    else:
                        var_min *= 0.97
                    if var_max > 0.:
                        var_max *= 1.03
                    else:
                        var_max *= 0.97
                    ax.set_xlim(start_time, end_time)
                    ax.set_ylim(var_min, var_max)
                    ax.legend()
                    if self.dimensional:
                        ax.set_xlabel(time_axis_label + ' [' + self._model_params.time_unit + ']')
                    else:
                        ax.set_xlabel(time_axis_label + ' [timeunits]')
                    title = self._plot_title
                    if self.dimensional:
                        pass
                    else:
                        title += r" (in nondim units)"
                    ax.set_title(title, pad=20.)

                elif style == '2Dscatter':
                    axe.plot(self.diagnostic[variables[0]][time_index:time_index + 1], self.diagnostic[variables[1]][time_index:time_index + 1], marker='o', linestyle='', color='r')
                elif style == '3Dscatter':
                    axe.plot(self.diagnostic[variables[0]][time_index:time_index + 1], self.diagnostic[variables[1]][time_index:time_index + 1], zs=self.diagnostic[variables[2]][time_index:time_index + 1],
                             marker='o', linestyle='', color='r')
                if show_time and 'scatter' in style:
                    if self.dimensional:
                        tt = " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[time_index]) + " " + self._model_params.time_unit
                    else:
                        tt = " at " + str(self._time[time_index]) + " timeunits"
                    if style == '2Dscatter':
                        axe.text(0.1, 0.9, tt, horizontalalignment='center', verticalalignment='center', transform=axe.transAxes)
                    elif style == '3Dscatter':
                        axe.text2D(0.1, 0.9, tt, horizontalalignment='center', verticalalignment='center', transform=axe.transAxes)
                if show:
                    plt.show()

            if show:
                plot = interactive(update, time_index=(0, len(self)-1, stride))
                anim = display(plot)
            else:
                return update

        elif output == 'show':

            anim = self._make_anim(variables, style, background, ax, figsize, show_time, plot_kwargs, anim_kwargs, True)
            if show:
                plt.show()

        else:
            warnings.warn('Provided output parameter ' + output + ' not supported ! Nothing to plot. Returning None.')
            anim = None

        return anim

    def _init_anim(self, variables='all', style="2Dscatter", background=None, ax=None, figsize=(16, 9),
                   show_time_in_title=False, show_time=True, plot_kwargs=None, anim_kwargs=None):

        if variables == 'all':
            variables = list(range(len(self._variable_list)))

        if background is None:
            background = self

        if ax is None:
            fig = plt.figure(figsize=figsize)
            if '3D' in style:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.figure

        if 'moving' not in style:
            background.plot(variables=variables, style=style, ax=ax, figsize=figsize, plot_kwargs=plot_kwargs)

        if 'timeserie' in style:
            show_time = False

        if show_time and 'scatter' in style:
            if self.dimensional:
                tt = " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[0]) + " " + self._model_params.time_unit
            else:
                tt = " at " + str(self._time[0]) + " timeunits"
            if style == '2Dscatter':
                time_text = ax.text(0.1, 0.9, tt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            elif style == '3Dscatter':
                time_text = ax.text2D(0.1, 0.9, tt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            time_text = None

        if style == 'timeserie':
            xpoint = list()
            lines = ax.get_lines()
            for i in range(len(variables)):
                pp, = ax.plot([], [], marker=',', linestyle='', color=lines[i].get_c())
                xpoint.append(pp)
        elif style == 'moving-timeserie':
            if 'frames' in anim_kwargs:
                ii = anim_kwargs['frames']
            else:
                ii = -1
            lines = list()
            var_min = 2.e168
            var_max = -2.e168
            end_time = -2.e168
            start_time = 2.e168
            for i in range(len(variables)):
                var_min = min(var_min, np.min(self.diagnostic[i]))
                var_max = max(var_max, np.max(self.diagnostic[i]))
                end_time = max(end_time, self._time[ii] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time))
                start_time = min(start_time, self._time[0] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time))
                time = self._time[:1] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time)
                lab = '$' + self._variable_labels[i] + '$'
                if self.dimensional:
                    lab += r" [" + self._variable_units[i] + r"]"
                line, = ax.plot(time, self.diagnostic[variables[i]][:1], label=lab)  # , **plot_kwargs
                lines.append(line)
            if var_min < 0.:
                var_min *= 1.03
            else:
                var_min *= 0.97
            if var_max > 0.:
                var_max *= 1.03
            else:
                var_max *= 0.97
            ax.set_xlim(start_time, end_time)
            ax.set_ylim(var_min, var_max)
            ax.legend()
            if self.dimensional:
                ax.set_xlabel(time_axis_label + ' [' + self._model_params.time_unit + ']')
            else:
                ax.set_xlabel(time_axis_label + ' [timeunits]')
            title = self._plot_title
            if self.dimensional:
                pass
            else:
                title += r" (in nondim units)"

            ax.set_title(title, pad=20.)
            xpoint = None
        elif style == '2Dscatter':
            xpoint, = ax.plot([], [], marker=',', linestyle='', color='r')
            lines = None
        elif style == '3Dscatter':
            xpoint, = ax.plot([], [], zs=[], marker=',', linestyle='', color='r')
            lines = None
        else:
            warnings.warn('Provided style parameter ' + style + ' not supported ! Nothing to plot. Returning None.')
            return None

        fargs = (ax, show_time)

        kwargs = {'variables': variables, 'style': style, 'background': background,
                  'ax': ax, 'figsize': figsize,
                  'show_time': show_time, 'plot_kwargs': plot_kwargs,
                  'anim_kwargs': anim_kwargs,
                  'xpoint': xpoint,
                  'lines': lines,
                  'time_text': time_text}

        return fig, ax, fargs, kwargs

    def _make_update(self, variables='all', style="2Dscatter", background=None, ax=None, figsize=(16, 9),
                     show_time_in_title=False, show_time=True, plot_kwargs=None, anim_kwargs=None, time_text=None, lines=None, xpoint=None):

        if style == 'timeserie':
            def update(i, axe, show_t):
                if i == 1:
                    for xp in xpoint:
                        xp.set_marker('o')

                time = self._time[i:i+1] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time)
                for j, xp in enumerate(xpoint):
                    xp.set_data(time, self.diagnostic[variables[j]][i:i+1])
                return [axe]

        elif style == 'moving-timeserie':
            def update(i, axe, show_t):

                time = self._time[0:i+1] * (1 - int(self.dimensional) + int(self.dimensional) * self._model_params.dimensional_time)

                for j, line in enumerate(lines):
                    line.set_xdata(time)
                    line.set_ydata(self.diagnostic[variables[j]][0:i+1])

                return [axe]

        elif style == '2Dscatter':
            def update(i, axe, show_t):
                if i == 1:
                    xpoint.set_marker('o')
                xpoint.set_data(self.diagnostic[variables[0]][i:i+1], self.diagnostic[variables[1]][i:i+1])
                if show_t:
                    if self.dimensional:
                        tt = " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[i]) + " " + self._model_params.time_unit
                    else:
                        tt = " at " + str(self._time[i]) + " timeunits"
                    time_text.set_text(tt)
                return [axe]

        elif style == '3Dscatter':
            def update(i, axe, show_t):
                if i == 1:
                    xpoint.set_marker('o')
                xpoint.set_data(self.diagnostic[variables[0]][i:i+1], self.diagnostic[variables[1]][i:i+1])
                xpoint.set_3d_properties(self.diagnostic[variables[2]][i:i+1])
                if show_t:
                    if self.dimensional:
                        tt = " at " + "{:.2f}".format(self._model_params.dimensional_time * self._time[i]) + " " + self._model_params.time_unit
                    else:
                        tt = " at " + str(self._time[i]) + " timeunits"
                    time_text.set_text(tt)
                return [axe]
        else:
            warnings.warn('Provided style parameter ' + style + ' not supported ! Nothing to plot. Returning None.')
            return None

        return update

    def _make_anim(self, variables='all', style="2Dscatter", background=None, ax=None, figsize=(16, 9),
                   show_time=True, plot_kwargs=None, anim_kwargs=None, blit=True):

        fig, ax, fargs, kwargs = self._init_anim(variables, style, background, ax, figsize, show_time, plot_kwargs, anim_kwargs)

        update = self._make_update(**kwargs)

        if 'blit' in anim_kwargs:
            del anim_kwargs['blit']

        if anim_kwargs is not None:
            anim = animation.FuncAnimation(fig, update, fargs=fargs, blit=blit, **anim_kwargs)
        else:
            anim = animation.FuncAnimation(fig, update, fargs=fargs, blit=blit)

        return anim


class GeopotentialHeightDifferenceDiagnostic(VariablesDiagnostic):
    """Class to compute and show the geopotential height difference
    between points of the model's domain.

    Parameters
    ----------

    points_list: list(2-tuple(2-tuple(float)))
        List of couple of point (as 2-tuple of float) of which to compute the geopotential height difference.
    model_params: QgParams
        An instance of the model parameters.
    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """
    def __init__(self, points_list, model_params, dimensional):

        self._point1 = list()
        self._point2 = list()
        self._func_points1 = None
        self._func_points2 = None
        variable_list = list(range(len(points_list)))

        VariablesDiagnostic.__init__(self, variable_list, model_params, dimensional)
        self._configure(points_list)
        self._plt = 'Geopotential height difference between points'
        self._plot_units = ' (in meters)'
        if self.dimensional:
            self._plot_title = self._plt + self._plot_units
        else:
            self._plot_title = self._plt

    def _configure(self, points_list, **kwargs):
        self.set_points(points_list)

    def set_points(self, points_list):
        """Set the couples of points of the domain of which to compute the geopotential height difference.

        Parameters
        ----------

        points_list: list(2-tuple(2-tuple(float)))
            List of couple of point (as 2-tuple of float) of which to compute the geopotential height difference.
        """
        self._point1 = list()
        self._point2 = list()
        for points in points_list:
            self._point1.append(points[0])
            self._point2.append(points[1])
        self._variable_labels = [ r'\mathrm{Points} \, (' + "{:.2f}".format(self._point1[i][0])+','
                                 + "{:.2f}".format(self._point1[i][1]) +r') \, \mathrm{and} \, (' + "{:.2f}".format(self._point2[i][0]) +
                                 ',' + "{:.2f}".format(self._point2[i][1])+')' for i in range(len(self._point1))]
        if self.dimensional:
            for i in range(len(self._variable_labels)):
                self._variable_labels[i] += r'\, \mathrm{in} \, '
        self._variable_units = len(self._point1) * ['m']
        self._compute_functions_value()

    def _compute_functions_value(self):

        if self._model_params.dynamic_T:
            offset = 1
        else:
            offset = 0

        self._func_points1 = list()
        self._func_points2 = list()

        basis = self._model_params.atmospheric_basis

        funcs_list = basis.num_functions(self._subs)

        for point in self._point1:
            self._func_points1.append(list())
            for func in funcs_list[offset:]:
                self._func_points1[-1].append(func(point[0], point[1]))

        for point in self._point2:
            self._func_points2.append(list())
            for func in funcs_list[offset:]:
                self._func_points2[-1].append(func(point[0], point[1]))

        self._func_points1 = np.array(self._func_points1)
        self._func_points2 = np.array(self._func_points2)

    def _get_diagnostic(self, dimensional):
        vr = self._model_params.variables_range
        for i in range(len(self._variable_list)):
            val1 = self._func_points1[i] @ self._data[:vr[0], ...]
            val2 = self._func_points2[i] @ self._data[:vr[0], ...]
            v = val1 - val2
            if i == 0:
                self._diagnostic_data = v[np.newaxis, :]
            else:
                self._diagnostic_data = np.vstack((self._diagnostic_data, v))

        if dimensional:
            self._diagnostic_data *= self._model_params.geopotential_scaling * self._model_params.streamfunction_scaling
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


if __name__ == '__main__':
    from qgs.params.params import QgParams
    from qgs.params.params import QgParams
    from qgs.integrators.integrator import RungeKuttaIntegrator
    from qgs.functions.tendencies import create_tendencies

    pars = QgParams()
    pars.set_atmospheric_channel_fourier_modes(2, 2)
    f, Df = create_tendencies(pars)
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)
    ic = np.random.rand(pars.ndim) * 0.1
    integrator.integrate(0., 200000., 0.1, ic=ic, write_steps=5)
    time, traj = integrator.get_trajectories()
    integrator.terminate()

    var_nondim = VariablesDiagnostic([10, 0, 14], pars, False)
    var_dim = VariablesDiagnostic([10, 0, 14], pars, True)

    var_nondim(time, traj)
    var_dim(time, traj)

    geo_nondim = GeopotentialHeightDifferenceDiagnostic([[[np.pi/pars.scale_params.n, np.pi/4], [np.pi/pars.scale_params.n, 3*np.pi/4]],
                                                         [[0, np.pi/4], [0, 3*np.pi/4]]],
                                                        pars, False)
    geo_nondim.set_data(time, traj)

    geo_dim = GeopotentialHeightDifferenceDiagnostic([[[np.pi/pars.scale_params.n, np.pi/4], [np.pi/pars.scale_params.n, 3*np.pi/4]],
                                                         [[0, np.pi/4], [0, 3*np.pi/4]]],
                                                        pars, True)
    geo_dim.set_data(time, traj)
