import numpy

"""
Implementation of RK4 in pythono
"""


def rk4(f, x: numpy.array, u: numpy.array, step: float) -> numpy.array:
    """
    RK4 Algorithm
    f - A function f that maps the states, x, to the state derivates xdot
    x - A vector of states
    step - Timestep for the ODE solver

    Not included here is time. (assuming time invariant system)
    """
    f1 = f(x, u)
    # This will probably need fixing once I have a vector coming out of f(x,u)
    f2 = f(x + 0.5 * f1 * step, u)
    f3 = f(x + 0.5 * f2 * step, u)
    f4 = f(x + f3 * step, u)
    x_next = x + (1 / 6) * (f1 + 2 * f2 + 2 * f3 + f4) * step
    return x_next


if __name__ == "__main__":
    import numpy as np

    f = lambda y, t: y**2
    T = np.linspace(0, 20, 160)
    y0 = -1

    res = []
    y = y0
    for t in T:
        y_next = rk4(f, y, 0, step=(20 / 160))
        y = y_next
        res.append(y)

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.show()
    # Yay RK4 works..
