import numpy as np


def distance_from_upright(X):
    x = X[0, 0]
    xdot = X[1, 0]
    theta = X[2, 0]
    thetadot = X[3, 0]
    return np.sqrt(x * x + xdot * xdot + theta * theta + thetadot * thetadot)


def time_to_upright(ts, xs, us, threshold=0.1):
    # all values in a ball with radius threshold
    for i in range(ts.shape[0]):
        t = ts[i]
        dist = distance_from_upright(xs[:, i])
        if dist < threshold:
            return t / ts[-1]
    return 1.0


def index_to_upright(ts, xs, us, threshold=0.1):
    # all values in a ball with radius threshold
    for i in range(ts.shape[0]):
        t = ts[i]
        dist = distance_from_upright(xs[:, i])
        if dist < threshold:
            return i
    return ts.shape[0] - 1


def min_distance_from_upright(ts, xs, us):
    min_dist = distance_from_upright(xs[:, 0])
    for i in range(1, ts.shape[0]):
        dist = distance_from_upright(xs[:, i])
        if dist < min_dist:
            min_dist = dist
    return min_dist


def index_min_distance_from_upright(ts, xs, us):
    min_dist = distance_from_upright(xs[:, 0])
    min_index = 0
    for i in range(1, ts.shape[0]):
        dist = distance_from_upright(xs[:, i])
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index


def squared_distance_from_upright(
    ts, xs, us, threshold=0.1
):  # - minus distance of the threshold configuration
    x = xs[0, -1]
    xdot = xs[1, -1]
    theta = xs[2, -1]
    thetadot = xs[3, -1]
    return (
        x * x
        + xdot * xdot
        + theta * theta
        + thetadot * thetadot
        - (threshold * threshold)
    )


def max_distance_from_position_range(ts, xs, us, simdic):
    xmin = simdic["xmin"]
    xmax = simdic["xmax"]

    maxx = 0
    minx = 0

    upright_threshold = simdic["uprightThreshold"]
    # find maximum distance until upright is reached
    index = index_to_upright(ts, xs, us, upright_threshold)
    for i in range(index + 1):
        x = xs[0, i]
        if x > maxx:
            maxx = x
        if x < minx:
            minx = x

    right_distance = np.array([0, maxx - xmax]).max()
    left_distance = np.abs(np.array([0, minx - xmin]).min())
    return np.array([right_distance, left_distance]).max()


def max_distance_from_position_range_until_min_distance_to_upright(
    ts, xs, us, simdic
):
    xmin = simdic["xmin"]
    xmax = simdic["xmax"]

    maxx = 0
    minx = 0

    upright_threshold = simdic["uprightThreshold"]
    # find maximum distance until upright is reached
    index = index_min_distance_from_upright(ts, xs, us)
    for i in range(index + 1):
        x = xs[0, i]
        if x > maxx:
            maxx = x
        if x < minx:
            minx = x

    right_distance = np.array([0, maxx - xmax]).max()
    left_distance = np.abs(np.array([0, minx - xmin]).min())
    return np.array([right_distance, left_distance]).max()


def avg_squared_control(ts, xs, us):
    dt = ts[1] - ts[0]
    return ((us * us).sum() * dt) / ts[-1]


def normalized_avg_squared_control(ts, xs, us, simdic):
    max_sq_u = simdic["umax"] * simdic["umax"]
    sq_u_min = simdic["umin"] * simdic["umin"]
    if sq_u_min > max_sq_u:
        max_sq_u = sq_u_min

    return avg_squared_control(ts, xs, us) / max_sq_u


def set_cost124(simdic):
    if "Js" not in simdic:
        simdic["Js"] = []

    def J(ts, xs, us):
        # max_dist = max_distance_from_position_range(ts, xs, us, simdic)
        max_dist = (
            max_distance_from_position_range_until_min_distance_to_upright(
                ts, xs, us, simdic
            )
        )  # this part can be changed
        if max_dist > 0:
            return 2 + 2 * (1 - np.exp(-max_dist))
        else:
            min_dist_from_upright = min_distance_from_upright(ts, xs, us)
            threshold = simdic["uprightThreshold"]
            if min_dist_from_upright > threshold:
                return 1 + (1 - np.exp(-(min_dist_from_upright - threshold)))
            else:
                return time_to_upright(ts, xs, us, threshold)

    simdic["Js"].append(J)


def set_control_cost(simdic):
    if "Js" not in simdic:
        simdic["Js"] = []

    def J(ts, xs, us):
        return normalized_avg_squared_control(ts, xs, us, simdic)

    simdic["Js"].append(J)
