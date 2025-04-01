import numpy

"""
Implementation of RK4 in python
"""


def rk4(t: float, f: callable, x: numpy.matrix, u: numpy.matrix, step: float) -> numpy.matrix:
    """
    RK4 Algorithm
    f - A function f(t, x, u) that maps the states, x, to the state derivates xdot.
    x - A vector of states
    step - Timestep for the ODE solver

    Not included here is time, you have to figure out the step size and give it to the solver.
    Assumes the dynamics in f() are time invariant for the time being...
    """
    f1 = f(t, x, u)
    # This will probably need fixing once I have a vector coming out of f(x,u)
    f2 = f(t, x + 0.5 * f1 * step, u)
    f3 = f(t, x + 0.5 * f2 * step, u)
    f4 = f(t, x + f3 * step, u)
    x_next = x + (1 / 6) * (f1 + 2 * f2 + 2 * f3 + f4) * step
    return x_next


if __name__ == "__main__":
    import numpy as np

    f = lambda t, y, x: y**2
    T = np.linspace(0, 20, 160)
    y0 = -1

    res = []
    y = y0
    for t in T:
        y_next = rk4(t, f, y, 0, step=(20 / 160))
        y = y_next
        res.append(y)

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.show()
    # Yay RK4 works..
