import numpy as np
import simulator as si
import animate as an
import multiprocessing as mu
import costs as co


class PendulumCart:
    # model from Kolmanovsky, Gilbery, 1995.
    # noise coefficients from Steinhardt et al., "Finite-Time Regional Verification of Stochastic Nonlinear Systems", Int. J. Robotics Research, 2012.
    def __init__(self, paramdic):
        self.paramdic = paramdic

        def ff(X, u):
            mp = self.paramdic["mp"]
            mc = self.paramdic["mc"]
            L = self.paramdic["L"]
            g = self.paramdic["g"]
            x = X[0, 0]
            xdot = X[1, 0]
            theta = X[2, 0]
            thetadot = X[3, 0]
            costheta = np.cos(theta)
            sintheta = np.sin(theta)

            xdotdot = (
                u + mp * sintheta * (L * thetadot * thetadot - g * costheta)
            ) / (mc + mp * sintheta * sintheta)
            thetadotdot = (
                -u * costheta
                - mp * L * thetadot * thetadot * costheta * sintheta
                + (mc + mp) * g * sintheta
            ) / (L * (mc + mp * sintheta * sintheta))
            if paramdic["type"] == "linear":
                xdotdot = (-mp * g * theta / mc) + u / mc
                thetadotdot = ((mp + mc) * g / (L * mc)) * theta - u / (L * mc)
            return np.matrix([[xdot], [xdotdot], [thetadot], [thetadotdot]])

        self.ff = ff

        self.f = None

    def get_u(self, simdic):
        upright_threshold = simdic["uprightThreshold"]
        K = np.matrix(
            [[0.5451, 1.8357, 27.2815, 8.6552]]
        )  # from Kolmanovsky et al., 1995

        def u(t, x, d=0):
            if d > 0.5:  # closed-loop active
                return (K * x)[0, 0]
            ps = simdic["ps"]
            us = simdic["us"]
            T = simdic["EMT"]
            p = t / T
            best_index = 0
            best_dist = np.abs(ps[0] - p)
            for i in range(1, ps.shape[0]):
                dist = np.abs(ps[i] - p)
                if dist < best_dist:
                    best_dist = dist
                    best_index = i
            return us[best_index]

        return u

    def get_f(self, simdic, activate_closed_loop):
        u = self.get_u(simdic)

        if activate_closed_loop:

            def f(t, x, d):
                return self.ff(x, u(t, x, d))

            return f
        else:

            def f(t, x):
                return self.ff(x, u(t, x))

            return f

    def get_h(self, simdic):
        upright_threshold = simdic["uprightThreshold"]

        def h(t, x, d):
            if d > 0.5:  # already activated, don't change
                return 1.0

            if co.distance_from_upright(x) <= upright_threshold:
                return 1.0  # activate closed-loop control
            else:
                return 0.0

        return h

    def simulate_once(self, simdic, activate_closed_loop=False):
        f = self.get_f(simdic, activate_closed_loop)
        h = self.get_h(simdic)
        em = si.Simulator(f, 4)  # n = 4 dimensional system
        if activate_closed_loop:
            em = si.Simulator(f, 4, h)
        xs = em.simulate(simdic["x0"], simdic["EMN"], simdic["EMT"])
        for i in range(xs.shape[1]):
            xs[2, i] = (xs[2, i] + 2 * np.pi) % (
                2 * np.pi
            )  # results will be between [0, 2 * pi]
        T = simdic["EMT"]
        dt = 1 / simdic["EMN"]
        ts = np.arange(0, T + dt, dt)
        us = np.zeros(ts.shape[0])
        ds = np.zeros(ts.shape[0])
        u = self.get_u(simdic)
        for i in range(ts.shape[0]):
            if activate_closed_loop:
                prevd = 0
                if i > 0:
                    prevd = ds[i - 1]
                ds[i] = h(ts[i], xs[:, i], prevd)

            us[i] = u(ts[i], xs[:, i], ds[i])
            xs[:, i] = (
                xs[:, i]
                + np.asmatrix(
                    em.rng.normal(0, self.paramdic["w"], 4)
                ).transpose()
            )
        return {"ts": ts, "xs": xs, "us": us, "ds": ds}

    def calculate_J_values(self, simdic):
        simulation_result = self.simulate_once(simdic)
        ts = simulation_result["ts"]
        xs = simulation_result["xs"]
        us = simulation_result["us"]
        return [J(ts, xs, us) for J in simdic["Js"]]


def average_J_values(paramdic, simdic):
    def calculate_J_values(i):
        return PendulumCart(paramdic).calculate_J_values(simdic)

    pool_indices = list(range(simdic["nS"]))
    with mu.Pool(processes=simdic["threadCount"]) as pool:
        all_J_values = pool.map(calculate_J_values, pool_indices)

    avg_J_values = [0 for J in simdic["Js"]]
    for i in range(n):
        for j in range(len(simdic["Js"])):
            avg_J_values[j] = (avg_J_values[j] * i + all_J_values[i][j]) / (
                i + 1
            )

    return avg_J_values


def print_xs(xs):
    for i in range(xs.shape[1]):
        print(
            "{:.3f} - {:.3f} - {:.3f} - {:.3f}".format(
                xs[0, i], xs[1, i], xs[2, i], xs[3, i]
            )
        )


if __name__ == "__main__":
    default_paramdic = {
        "type": "nonlinear",
        "mp": 0.5,
        "mc": 0.5,
        "L": 1.4,
        "g": 10,
        "w": np.array([0.03, 0.1, 0.03, 0.1]),
    }

    default_simdic = {
        "EMN": 100,
        "EMT": 10,
        "ps": np.array([0, 1.0]),
        "us": np.array([0, 0]),
        "x0": np.matrix([[0], [0], [np.pi], [0]]),
        "uprightThreshold": 0.1,
        "umax": 20,
        "umin": -20,
        "xmax": 3,
        "xmin": -3,
    }
    default_simdic["x0"] = np.matrix([[0], [0.0], [0.0], [0]])
    an.animate(
        PendulumCart(default_paramdic).simulate_once(
            default_simdic, activate_closed_loop=True
        )["xs"],
        default_paramdic,
        default_simdic,
        "animations/standard_with_control_activation.mp4",
        save=True,
    )
