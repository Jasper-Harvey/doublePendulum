import numpy as np


class Pendulum:
    L1 = 0  # Length of pendulum 1 (m)
    L2 = 0  # Length of pendulum 2 (m)
    mc = 0  # Mass of the cart (kg)
    m1 = 0  # Mass of pendulum 1 (kg)
    m2 = 0  # Mass of pendulum 2 (kg)
    g = 9.81  # Gravity.

    xlim_lower = -0.1  # Lower x value limit of the cart (m).
    xlim_upper = 0.1  # Upper x value limit of the cart (m).


PENDULUM_DATA = Pendulum()
PENDULUM_DATA.mc = 0.4
PENDULUM_DATA.m1 = 0.4
PENDULUM_DATA.m2 = 0.4
PENDULUM_DATA.L1 = 0.4
PENDULUM_DATA.L2 = 0.4
PENDULUM_DATA.g = 9.81


def pendulumDynamics(t: float, x: np.matrix, u: np.matrix):
    """
    States for the dynamic equations are:
    x1 = x
    x2 = theta1
    x3 = theta2
    x4 = x_dot
    x5 = theta1_dot
    x6 = theta2_dot

    u is an input force on the cart in the x direction.

    x = [x1, x2, x3, x4, x5, x6]
    u = u1
    """
    mc = PENDULUM_DATA.mc  # kg
    m1 = PENDULUM_DATA.m1  # kg
    m2 = PENDULUM_DATA.m2  # kg
    L1 = PENDULUM_DATA.L1  # m
    L2 = PENDULUM_DATA.L2  # m
    g = PENDULUM_DATA.g  # g

    xc = x[0, 0]
    theta1 = x[1, 0]
    theta2 = x[2, 0]
    xc_dot = x[3, 0]
    theta1_dot = x[4, 0]
    theta2_dot = x[5, 0]

    D11 = mc + m1 + m2
    D12 = (0.5 * m1 + m2) * L1 * np.cos(theta1)
    D13 = 0.5 * m2 * L2 * np.cos(theta2)

    D22 = ((1 / 3) * m1 + m2) * L1 * L1
    D23 = 0.5 * m2 * L1 * L2 * np.cos(theta1 - theta2)

    D33 = (1 / 3) * m2 * L2 * L2

    D = np.array(
        [
            [D11, D12, D13],
            [D12, D22, D23],
            [D13, D23, D33],
        ]
    )

    C12 = -(0.5 * m1 + m2) * L1 * np.sin(theta1) * theta1_dot
    C13 = -0.5 * m2 * L2 * np.sin(theta2) * theta2_dot
    C23 = 0.5 * m2 * L1 * L2 * np.sin(theta1 - theta2) * theta2_dot
    C32 = -0.5 * m2 * L1 * L2 * np.sin(theta1 - theta2) * theta1_dot

    C = np.array(
        [
            [0, C12, C13],
            [0, 0, C23],
            [0, C32, 0],
        ]
    )

    G2 = -0.5 * g * (m1 + m2) * L1 * np.sin(theta1)
    G3 = -0.5 * g * m2 * L2 * np.sin(theta2)

    G = np.array(
        [
            [
                0,
            ],
            [
                G2,
            ],
            [
                G3,
            ],
        ]
    )

    H = np.array(
        [
            [
                1,
            ],
            [
                0,
            ],
            [
                0,
            ],
        ]
    )

    pre_ddx = -(C @ x[3:, :]) - G + (H @ u)
    ddx = np.linalg.solve(D, pre_ddx)

    x_dot = np.block([[x[3:, :]], [ddx]])

    return x_dot


if __name__ == "__main__":
    x0 = np.zeros((6, 1))
    x0[0, 0] = 0.1
    x0[4, 0] = 0.6
    a = pendulumDynamics(0, x0, np.matrix(0))
    print(a)
