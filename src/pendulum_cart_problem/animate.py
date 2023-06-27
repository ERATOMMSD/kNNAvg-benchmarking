import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as an
import matplotlib.patches as pa


def new_fig(figsize=(5.65, 4), fontsize=14):
    pl.rc("ps", usedistiller="xpdf")
    pl.rc("text", usetex=True)
    pl.rc("font", size=fontsize)
    pl.rc(
        "text.latex", preamble="\\usepackage{amsmath}\n \\usepackage{amssymb}"
    )
    pl.rcParams["mathtext.fontset"] = "custom"
    fig = pl.figure(figsize=figsize)  # , tight_layout=True)
    return fig


def animate(result, paramdic, simdic, anim_filename, save=True):
    xs = result["xs"]
    xmin = simdic["xmin"]
    xmax = simdic["xmax"]
    ymin = -1.1 * paramdic["L"]
    ymax = 1.5 * paramdic["L"]
    xxrange = xmax - xmin
    yyrange = ymax - ymin
    figwidth = 6
    figheight = yyrange * figwidth / xxrange
    fig = new_fig(figsize=(figwidth, figheight), fontsize=14)
    ax = pl.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))

    cart_width = 1
    cart_height = 0.1
    pendulum_length = 0.5
    if "L" in paramdic:
        pendulum_length = paramdic["L"]
    if "l" in paramdic:
        pendulum_length = paramdic["l"]
    pendulum_thickness = 0.01

    def get_cartxy(i):
        y = 0
        x = -cart_width / 2 + xs[0, i]
        return (x, y)

    def get_pendulumxy(i):
        y = cart_height
        x = xs[0, i]
        return (x, y)

    def get_pendulumangle(i):
        return (180 / np.pi) * (-xs[2, i] + np.pi / 2)

    def get_pendulumwidth(i):
        pxrange = np.abs(np.array(pl.xlim())).sum()
        pyrange = np.abs(np.array(pl.ylim())).sum()

        width = (
            pyrange
            * (pendulum_thickness / pxrange)
            * np.tan((np.pi / 180) * get_pendulumangle(i))
        )
        return width

    ground = pa.Rectangle((xmin, 0), xxrange, 0.01, color="#CCCCCC")
    cart = pa.Rectangle(get_cartxy(0), cart_width, cart_height, color="C1")
    pendulum = pa.Rectangle(
        get_pendulumxy(0),
        pendulum_length,
        pendulum_thickness,
        angle=get_pendulumangle(0),
    )

    def init():
        cart.xy = get_cartxy(0)
        pendulum.xy = get_pendulumxy(0)
        pendulum.angle = get_pendulumangle(0)
        ax.add_patch(ground)
        ax.add_patch(cart)
        ax.add_patch(pendulum)
        return [ground, cart, pendulum]

    def animate(i):
        cart.xy = get_cartxy(i)
        if result["ds"][i] > 0.5:
            cart.set_color("#303030")
        else:
            cart.set_color("C1")
        pendulum.xy = get_pendulumxy(i)
        pendulum.angle = get_pendulumangle(i)
        return [ground, cart, pendulum]

    pl.xlabel("$\\mathrm{Cart\,\, position}\,\, x_1$")
    pl.tight_layout()

    ani = an.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=xs.shape[1],
        interval=1000 * 1 / simdic["EMN"],
        blit=True,
    )
    if save:
        ani.save(anim_filename)
    else:
        pl.show()
