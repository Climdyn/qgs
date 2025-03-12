"""
    Plotting utility module
    =======================

    Work in progress...

"""
import matplotlib.pyplot as plt


def std_plot(x, mean, std, ax=None, **kwargs):
    """Plot a mean and a standard deviation time series.

    Parameters
    ----------
    x: ~numpy.ndarray
        1D array of the x-axis values (usually time)
    mean: ~numpy.ndarray
        1D array of mean values to represent, should have the same shape as x.

    std: ~numpy.ndarray
        1D array of standard deviation values to represent, should have the same shape as mean.
    ax: None or ~matplotlib.axes.Axes
        A `matplotlib`_ axes instance to plot the values. If `None`, create one.
    kwargs: dict
        Keyword arguments to be given to the plot routine.

    Returns
    -------
    ~matplotlib.axes.Axes
        The axes instance on which the values where plotted.


    .. _matplotlib: https://matplotlib.org/
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    ax.fill_between(x, mean - std, mean + std, **kwargs)

    return ax
