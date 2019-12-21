
from numba import njit
import numpy as np
from functions.util import reverse


def integrate_runge_kutta(f, t0, t, dt, ic=None, forward=True, write_steps=1, b=None, c=None, a=None):

    if ic is None:
        i = 1
        while True:
            ic = np.zeros(i)
            try:
                x = f(0., ic)
            except:
                i += 1
            else:
                break

        i = len(f(0., ic))
        ic = np.zeros(i)

    if len(ic.shape) == 1:
        ic = ic.reshape((1, -1))

    # Default is RK4
    if a is None and b is None and c is None:
        c = np.array([0., 0.5, 0.5, 1.])
        b = np.array([1./6, 1./3, 1./3, 1./6])
        a = np.zeros((len(c), len(b)))
        a[1, 0] = 0.5
        a[2, 1] = 0.5
        a[3, 2] = 1.

    if forward:
        time_direction = 1
    else:
        time_direction = -1

    time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))

    recorded_traj = integrate_runge_kutta_jit(f, time, ic, time_direction, write_steps, b, c, a)

    if write_steps > 0:
        if forward:
            if time[::write_steps][-1] == time[-1]:
                return time[::write_steps], np.squeeze(recorded_traj)
            else:
                return np.concatenate((time[::write_steps], np.full((1,), t))), np.squeeze(recorded_traj)
        else:
            rtime = reverse(time[::-write_steps])
            if rtime[0] == time[0]:
                return rtime, np.squeeze(recorded_traj)
            else:
                return np.concatenate((np.full((1,), t0), rtime)), np.squeeze(recorded_traj)
    else:
        return time[-1], np.squeeze(recorded_traj)


@njit
def integrate_runge_kutta_jit(f, time, ic, time_direction, write_steps, b, c, a):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    s = len(b)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1

    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    if time_direction == -1:
        directed_time = reverse(time)
    else:
        directed_time = time

    for i_traj in range(n_traj):
        y = ic[i_traj].copy()
        k = np.zeros((s, n_dim))
        iw = 0
        for ti, (tt, dt) in enumerate(zip(directed_time[:-1], np.diff(directed_time))):

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_traj[i_traj, :, iw] = y
                iw += 1

            k.fill(0.)
            for i in range(s):
                y_s = y + dt * a[i] @ k
                k[i] = f(tt + c[i] * dt, y_s)
            y_new = y + dt * b @ k
            y = y_new

        recorded_traj[i_traj, :, -1] = y

    return recorded_traj[:, :, ::time_direction]


@njit
def tangent_linear_system(fjac, t, xs, x, adjoint):
    if adjoint:
        return fjac(t, xs).transpose() @ x
    else:
        return fjac(t, xs) @ x


@njit
def zeros_func(t, x):
    return np.zeros_like(x)


def integrate_runge_kutta_tgls(f, fjac, t0, t, dt, ic=None, tg_ic=None,
                               forward=True, adjoint=False, inverse=False, boundary=None,
                               write_steps=1, b=None, c=None, a=None, method=1):

    if ic is None:
        i = 1
        while True:
            ic = np.zeros(i)
            try:
                x = f(0., ic)
            except:
                i += 1
            else:
                break

        i = len(f(0., ic))
        ic = np.zeros(i)

    if len(ic.shape) == 1:
        ic = ic.reshape((1, -1))

    n_traj = ic.shape[0]

    if tg_ic is None:
        tg_ic = np.eye(ic.shape[1])

    if len(tg_ic.shape) == 1:
        tg_ic = tg_ic.reshape((1, -1, 1))
        ict = tg_ic.copy()
        for i in range(n_traj-1):
            ict = np.concatenate((ict, tg_ic))
        tg_ic = ict
    elif len(tg_ic.shape) == 2:
        if tg_ic.shape[0] == n_traj:
            tg_ic = tg_ic[..., np.newaxis]
        else:
            tg_ic = tg_ic[np.newaxis, ...]
            ict = tg_ic.copy()
            for i in range(n_traj-1):
                ict = np.concatenate((ict, tg_ic))
            tg_ic = ict

    # Default is RK4
    if a is None and b is None and c is None:
        c = np.array([0., 0.5, 0.5, 1.])
        b = np.array([1./6, 1./3, 1./3, 1./6])
        a = np.zeros((len(c), len(b)))
        a[1, 0] = 0.5
        a[2, 1] = 0.5
        a[3, 2] = 1.

    if forward:
        time_direction = 1
    else:
        time_direction = -1

    time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))

    if boundary is None:
        boundary = zeros_func

    inv = 1.
    if inverse:
        inv *= -1.

    if method == 1:
        recorded_traj, recorded_fmatrix = integrate_runge_kutta_tgls_jit1(f, fjac, time, ic, tg_ic,
                                                                          time_direction, write_steps,
                                                                          b, c, a, adjoint, inv, boundary)
    elif method == 2:
        recorded_traj, recorded_fmatrix = integrate_runge_kutta_tgls_jit2(f, fjac, time, ic, tg_ic,
                                                                          time_direction, write_steps,
                                                                          b, c, a, adjoint, inv, boundary)
    else:
        recorded_traj, recorded_fmatrix = integrate_runge_kutta_tgls_jit3(f, fjac, time, ic, tg_ic,
                                                                          time_direction, write_steps,
                                                                          b, c, a, adjoint, inv, boundary)

    if write_steps > 0:
        if forward:
            if time[::write_steps][-1] == time[-1]:
                return time[::write_steps], np.squeeze(recorded_traj), np.squeeze(recorded_fmatrix)
            else:
                return np.concatenate((time[::write_steps], np.full((1,), t))), np.squeeze(recorded_traj),\
                       np.squeeze(recorded_fmatrix)
        else:
            rtime = reverse(time[::-write_steps])
            if rtime[0] == time[0]:
                return rtime, np.squeeze(recorded_traj), np.squeeze(recorded_fmatrix)
            else:
                return np.concatenate((np.full((1,), t0), rtime)), np.squeeze(recorded_traj),\
                       np.squeeze(recorded_fmatrix)

    else:
        return time[-1], np.squeeze(recorded_traj), np.squeeze(recorded_fmatrix)


@njit
def integrate_runge_kutta_tgls_jit1(f, fjac, time, ic, tg_ic, time_direction, write_steps, b, c, a,
                                    adjoint, inverse, boundary):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    s = len(b)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1
    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    recorded_fmatrix = np.zeros((n_traj, tg_ic.shape[1], tg_ic.shape[2], n_records))
    if time_direction == -1:
        directed_time = reverse(time)
    else:
        directed_time = time

    for i_traj in range(n_traj):
        y = ic[i_traj].copy()
        fm = tg_ic[i_traj].copy()
        recorded_traj[i_traj, :, 0] = ic[i_traj]
        recorded_fmatrix[i_traj, :, :, 0] = tg_ic[i_traj]
        k = np.zeros((s, n_dim))
        km = np.zeros((s, tg_ic.shape[1], tg_ic.shape[2]))
        iw = 0
        for ti, (tt, dt) in enumerate(zip(directed_time[:-1], np.diff(directed_time))):

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_traj[i_traj, :, iw] = y
                recorded_fmatrix[i_traj, :, :, iw] = fm
                iw += 1

            k.fill(0.)
            km.fill(0.)
            for i in range(s):
                y_s = y + dt * a[i] @ k
                k[i] = f(tt + c[i] * dt, y_s)
                km_s = fm.copy()
                for j in range(len(a[i])):
                    km_s += dt * a[i, j] * km[j]
                hom = inverse * tangent_linear_system(fjac, tt + c[i] * dt, y_s, km_s, adjoint)
                inhom = boundary(tt + c[i] * dt, y_s)
                km[i] = (hom.T + inhom.T).T
            y_new = y + dt * b @ k
            fm_new = fm.copy()
            for j in range(len(b)):
                fm_new += dt * b[j] * km[j]
            y = y_new
            fm = fm_new

        recorded_traj[i_traj, :, -1] = y
        recorded_fmatrix[i_traj, :, :, -1] = fm

    return recorded_traj[:, :, ::time_direction], recorded_fmatrix[:, :, :, ::time_direction]


@njit
def integrate_runge_kutta_tgls_jit2(f, fjac, time, ic, tg_ic, time_direction, write_steps, b, c, a,
                                    adjoint, inverse, boundary):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    s = len(b)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1
    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    recorded_fmatrix = np.zeros((n_traj, tg_ic.shape[1], tg_ic.shape[2], n_records))
    if time_direction == -1:
        directed_time = reverse(time)
    else:
        directed_time = time

    for i_traj in range(n_traj):
        y = ic[i_traj].copy()
        fm = tg_ic[i_traj].copy()
        recorded_traj[i_traj, :, 0] = ic[i_traj]
        recorded_fmatrix[i_traj, :, :, 0] = tg_ic[i_traj]
        k = np.zeros((s, n_dim))
        km = np.zeros((s, tg_ic.shape[1], tg_ic.shape[2]))
        iw = 0
        for ti, (tt, dt) in enumerate(zip(directed_time[:-1], np.diff(directed_time))):

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_traj[i_traj, :, iw] = y
                recorded_fmatrix[i_traj, :, :, iw] = fm
                iw += 1

            k.fill(0.)
            km.fill(0.)
            for i in range(s):
                y_s = y + dt * a[i] @ k
                k[i] = f(tt + c[i] * dt, y_s)
                km_s = np.eye(n_dim)
                for j in range(len(a[i])):
                    km_s += dt * a[i, j] * km[j]
                hom = inverse * tangent_linear_system(fjac, tt + c[i] * dt, y_s, km_s, adjoint)
                inhom = boundary(tt + c[i] * dt, y_s)
                km[i] = (hom.T + inhom.T).T
            y_new = y + dt * b @ k
            y = y_new
            fmm = np.eye(n_dim)
            for j in range(len(b)):
                fmm += dt * b[j] * km[j]

            fm_new = fmm @ fm
            fm = fm_new

        recorded_traj[i_traj, :, -1] = y
        recorded_fmatrix[i_traj, :, :, -1] = fm

    return recorded_traj[:, :, ::time_direction], recorded_fmatrix[:, :, :, ::time_direction]


@njit
def integrate_runge_kutta_tgls_jit3(f, fjac, time, ic, tg_ic, time_direction, write_steps, b, c, a,
                                    adjoint, inverse, boundary):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    s = len(b)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1
    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    recorded_fmatrix = np.zeros((n_traj, tg_ic.shape[1], tg_ic.shape[2], n_records))
    if time_direction == -1:
        directed_time = reverse(time)
    else:
        directed_time = time

    for i_traj in range(n_traj):
        y = ic[i_traj].copy()
        fm = tg_ic[i_traj].copy()
        k = np.zeros((s, n_dim))
        recorded_traj[i_traj, :, 0] = ic[i_traj]
        recorded_fmatrix[i_traj, :, :, 0] = tg_ic[i_traj]
        iw = 0
        for ti, (tt, dt) in enumerate(zip(directed_time[:-1], np.diff(directed_time))):

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_traj[i_traj, :, iw] = y
                recorded_fmatrix[i_traj, :, :, iw] = fm
                iw += 1

            k.fill(0.)
            for i in range(s):
                y_s = y + dt * a[i] @ k
                k[i] = f(tt + c[i] * dt, y_s)
            y_new = y + dt * b @ k
            y = y_new

            fm_new = (np.eye(n_dim) + dt * fjac(tt+dt, y)) @ fm
            fm = fm_new

        recorded_traj[i_traj, :, -1] = y
        recorded_fmatrix[i_traj, :, :, -1] = fm

    return recorded_traj[:, :, ::time_direction], recorded_fmatrix[:, :, :, ::time_direction]


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.integrate import odeint

    @njit
    def f(t, x):
        return - np.array([1., 2., 3.]) * x

    def fr(x, t):
        return f(t, x)

    ic = np.random.randn(6).reshape(2, 3)

    time, r = integrate_runge_kutta(f, 0., 10., 0.01, ic=ic, write_steps=3)

    t = np.arange(0., 10., 0.01)
    t = np.concatenate((t[::3], np.full((1,), 10.)))
    rl = list()
    for i in range(ic.shape[0]):
        rl.append(odeint(fr, ic[i], t).T)

    plt.figure()
    for i in range(ic.shape[0]):
        p, = plt.plot(time, r[i, 0])
        c = p.get_color()
        plt.plot(t, rl[i][0], color=c, ls='--')
        for j in range(1, ic.shape[1]):
            p, = plt.plot(time, r[i, j], color=c)
            plt.plot(t, rl[i][j], color=c, ls='--')

    timet, rt = integrate_runge_kutta(f, 0., 10., 0.01, ic=ic, forward=False, write_steps=3)
    rlt = list()
    for i in range(ic.shape[0]):
        rlt.append(odeint(fr, ic[i], reverse(t)).T)

    plt.figure()
    for i in range(ic.shape[0]):
        p, = plt.plot(timet, rt[i, 0])
        c = p.get_color()
        plt.plot(t, reverse(rlt[i][0]), color=c, ls='--')
        for j in range(1, ic.shape[1]):
            p, = plt.plot(timet, rt[i, j], color=c)
            plt.plot(t, reverse(rlt[i][j]), color=c, ls='--')

    tt, re = integrate_runge_kutta(f, 0., 10., 0.01, ic=ic, write_steps=0)
    print(tt)
    print(r[0, :, -1], re[0])
    plt.show(block=False)

    a = 0.25
    F = 16.
    G = 3.
    b = 6.


    @njit
    def DfL84(t, x):
        return np.array([[     -a        , -2. * x[1], -2. * x[2]],
                         [x[1] - b * x[2], -1. + x[0], -b * x[0]],
                         [b * x[1] + x[2],  b * x[0], -1. + x[0]]])

    @njit
    def fL84(t, x):
        xx = -x[1] ** 2 - x[2] ** 2 - a * x[0] + a * F
        yy = x[0] * x[1] - b * x[0] * x[2] - x[1] + G
        zz = b * x[0] * x[1] + x[0] * x[2] - x[2]
        return np.array([xx, yy, zz])

    def fL84r(x, t):
        return fL84(t, x)

    tt, ic_L84 = integrate_runge_kutta(fL84, 0., 10000., 0.01, write_steps=0)

    print(ic_L84)

    ttt1, irkt1, fm_irkt1 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 0.01, 0.001, ic=ic_L84, write_steps=1)
    bttt1, birkt1, bfm_irkt1 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 0.01, 0.001, ic=irkt1[:, -1],
                                                          forward=False, write_steps=1, adjoint=True, inverse=True)

    plt.figure()
    for i in range(len(ic_L84)):
        for j in range(len(ic_L84)):
            p, = plt.plot(ttt1, fm_irkt1[i, j])
            c = p.get_color()
            plt.plot(bttt1, bfm_irkt1[i, j], ls='--', color=c)

    plt.figure()
    for i in range(len(ic_L84)):
        p, = plt.plot(ttt1, irkt1[i])
        c = p.get_color()
        plt.plot(bttt1, birkt1[i], ls='--', color=c)

    # ttt1, irkt1, fm_irkt1 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 1., 0.001, ic=ic_L84, write_steps=1)
    # ttt2, irkt2, fm_irkt2 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 10., 0.001, ic=ic_L84, write_steps=100,
    #                                                    method=2)
    # ttt3, irkt3, fm_irkt3 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 10., 0.001, ic=ic_L84, write_steps=100,
    #                                                    method=3)
    # bttt1, birkt1, bfm_irkt1 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 10., 0.001, ic=irkt1[:, -1],
    #                                                       tg_ic=fm_irkt1[:,:,-1], forward=False, write_steps=1)
    # bttt2, birkt2, bfm_irkt2 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 10., 0.001, ic=irkt1[:, -1],
    #                                                       tg_ic=fm_irkt1[:,:,-1], forward=False, write_steps=100,
    #                                                       method=2)
    # bttt3, birkt3, bfm_irkt3 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 10., 0.001, ic=irkt1[:, -1],
    #                                                       tg_ic=fm_irkt1[:,:,-1], forward=False, write_steps=100,
    #                                                       method=3)
    #
    # plt.figure()
    # for i in range(len(ic_L84)):
    #     for j in range(len(ic_L84)):
    #         p, = plt.plot(ttt1, fm_irkt1[i, j])
    #         c = p.get_color()
    #         plt.plot(ttt2, fm_irkt2[i, j], ls='', marker='+', color=c)
    #         plt.plot(ttt3, fm_irkt3[i, j], ls='', marker='x', color=c)
    #         plt.plot(bttt1, bfm_irkt1[i, j], ls='--', color=c)
    #         plt.plot(bttt2, bfm_irkt2[i, j], ls='', marker='D', color=c)
    #         plt.plot(bttt3, bfm_irkt3[i, j], ls='', marker='*', color=c)

    vec = np.random.randn(3)
    vec = vec/np.linalg.norm(vec)
    for i in range(1,13):
        lam = 2.**(-i)

        dy = lam * vec

        tx, dx = integrate_runge_kutta(fL84, 0., 0.001, 0.001, ic=ic_L84, write_steps=0)
        txp, dxp = integrate_runge_kutta(fL84, 0., 0.001, 0.001, ic=ic_L84+dy, write_steps=0)

        dy1 = dxp - dx

        txtl, dxtl, dytl = integrate_runge_kutta_tgls(fL84, DfL84, 0., 0.001, 0.001, ic=ic_L84, tg_ic=dy, write_steps=0)

        print('Perturbation size: ', dy @ dy)
        print('Resulting difference in trajectory: (eps ~ 2^-'+str(i)+')')
        print('diff: ', dy1 @ dy1)
        print('tl: ', dytl @ dytl)
        print('ratio: ', (dy1 @ dy1) / (dytl @ dytl))
