
import numpy as np
import matplotlib.ticker as ticker

# Module private variables and functions

x_axis_label = r"$x$"
y_axis_label = r"$y$"
time_axis_label = 'time'


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    if x == 0.:
        return u'0'
    elif x < 0.:
        return (r'{}$\times 10^{{{}}}$'.format(a, b)).replace(u'-', u'\u2212', 1)
    else:
        return r'{}$\times 10^{{{}}}$'.format(a, b)


strf = ticker.FuncFormatter(fmt)


def tick_fmt(tl):
    tlabels = [item.get_text() for item in tl]
    ii = len(tlabels) - 1
    labto = []
    jk = 0
    for x in tlabels:
        if x:
            jk += 1
            y = float(x.replace(u'\u2212', u'-'))
            if jk == 1:
                n = _order(y)
            if abs(n) > 2:
                y = y * 10 ** (-n)
                y = round(y, 2)
                # labto.append(unicode(y).replace(u'-',u'\u2212')+u'e'+unicode(n).replace(u'-',u'\u2212'))
                labto.append(str(y).replace(u'-', u'\u2212') + r'$\times 10^{' + str(n) + r'}$')
            else:
                y = round(y, 2)
                labto.append(str(y).replace(u'-', u'\u2212'))
        else:
            ii -= 1
            labto.append(x)

    return labto, ii


def _order(n):
    if n == 0.:
        return 0
    h = np.abs(n)
    if h < 1.:
        return int(np.log10(h))-1
    else:
        return int(np.log10(h))+1


