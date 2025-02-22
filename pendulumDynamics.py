import numpy as np


class Pendulum:
    L1 = 0
    L2 = 0
    mc = 0
    m1 = 0
    m2 = 0
    g = 9.81  # Gravity. Probably best not to change this...

    xlim_lower = -0.1
    xlim_upper = 0.1


PENDULUM_DATA = Pendulum()
PENDULUM_DATA.mc = 0.4  # kg
PENDULUM_DATA.m1 = 0.4  # kg
PENDULUM_DATA.m2 = 0.4  # kg
PENDULUM_DATA.L1 = 0.2  # m
PENDULUM_DATA.L2 = 0.4  # m
PENDULUM_DATA.g = 9.81  # g


def pendulumDynamics(x, u):
    """
    x - the states
    x1 = x
    x2 = theta1
    x3 = theta2
    x4 = x_dot
    x5 = theta1_dot
    x6 = theta2_dot
    """
    mc = PENDULUM_DATA.mc  # kg
    m1 = PENDULUM_DATA.m1  # kg
    m2 = PENDULUM_DATA.m2  # kg
    L1 = PENDULUM_DATA.L1  # m
    L2 = PENDULUM_DATA.L2  # m
    g = PENDULUM_DATA.g  # g

    xc = x[0, 0]
    # x[0,0] = 0 # Fix the card in place
    theta1 = x[1, 0]
    theta2 = x[2, 0]
    xc_dot = x[3, 0]
    # x[3,0] = 0 # Fix the cart in place
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

    I = np.eye(3)
    D_inv = np.linalg.lstsq(D, I)[0]  # I bet this is breaking shit

    D_inv_C = -(D_inv @ C)
    blk1 = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 3)), D_inv_C]])

    D_inv_G = -(D_inv @ G)
    blk2 = np.block([[np.zeros((3, 1))], [D_inv_G]])

    D_inv_H = D_inv @ H
    blk3 = np.block([[np.zeros((3, 1))], [D_inv_H]])

    # Original one:
    # x_dot = blk1 @ x + blk2 + blk3 @ u

    # test = np.block([[np.zeros((3,3)), np.eye(3)], [np.zeros((3,3)), -C]]) @ x + np.block([[np.zeros((3,1))], [-G]]) + np.block([[np.zeros((3,1))], [H,]]) @ u
    # np.linalg.solve(np.block([[np.zeros((3,6))], [np.zeros((3,3)), D]]), test) # SINGULAR! :(
    # x_dot = np.linalg.lstsq(np.block([[np.zeros((3,6))], [np.zeros((3,3)), D]]), test)[0]

    pre_ddx = -(C @ x[3:, :]) - G + (H @ u)
    ddx = np.linalg.solve(D, pre_ddx)

    x_dot = np.block([[x[3:, :]], [ddx]])

    # Constrain states here?
    if x[0, 0] > PENDULUM_DATA.xlim_upper:
        x[0, 0] = PENDULUM_DATA.xlim_upper
    if x[0, 0] < PENDULUM_DATA.xlim_lower:
        x[0, 0] = PENDULUM_DATA.xlim_lower

    return x_dot


if __name__ == "__main__":
    x0 = np.zeros((6, 1))
    x0[0, 0] = 0.1
    x0[4, 0] = 0.6
    a = pendulumDynamics(x0, 0)
    print(a)
