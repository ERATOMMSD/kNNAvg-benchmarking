import numpy as np
import numpy.random as ra
import scipy.integrate as integ


class Simulator:
    def __init__(self, f, n, h=None):
        self.f = f
        self.n = n
        self.h = h
        self.rng = ra.default_rng()

    def simulate(self, x0, N, T):
        dt = 1 / N
        ts = np.arange(0, T + dt, dt)

        def ode_f(t, y):
            n = self.n
            m = np.asmatrix(y).transpose()
            x = m[0:n, 0]
            d = m[n, 0]
            nextd = d
            if self.h is not None:
                nextd = self.h(t, x, d)

            if self.h is None:
                nextx = self.f(t, x)
            else:
                nextx = self.f(
                    t, x, nextd
                )  # use nextd as it is not a dynamic variable

            nexty = np.zeros(n + 1)
            nexty[0:n] = np.asarray(nextx.transpose())[0]
            nexty[n] = nextd
            return nexty

        xs = np.matrix(np.zeros((self.n, ts.shape[0])))

        n = self.n
        y0 = np.zeros(n + 1)
        y0[0:n] = np.asarray(x0.transpose())[0]
        y0[n] = 0
        if self.h is not None:
            y0[n] = self.h(0, x0, 0)

        ys = integ.solve_ivp(ode_f, (0, T), y0, method="RK45", t_eval=ts).y

        for i in range(0, ts.shape[0]):
            xs[:, i] = np.asmatrix(ys[0:n, i]).transpose()
        return xs
