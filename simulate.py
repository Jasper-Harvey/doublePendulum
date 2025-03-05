from pendulumDynamics import pendulumDynamics, PENDULUM_DATA
from rk4 import rk4

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Pendulum has 6 states
# Numpy is row, col


# Create a vector of times
step = 0.01
T = np.arange(start=0, stop=10, step=step)
states = np.matrix(np.zeros((6, T.shape[0])))

states[:, 0] = np.matrix(
    [
        [0],
        [0],
        [np.pi + 0.7],
        [0],
        [0],
        [0],
    ]
)

for i in range(0, T.shape[0] - 1):
    # # State Constraints:
    # if states[0, i] > PENDULUM_DATA.xlim_upper:
    #     states[0, i] = PENDULUM_DATA.xlim_upper
    #     states[3,i] = 0
    # if states[0, i] < PENDULUM_DATA.xlim_lower:
    #     states[0, i] = PENDULUM_DATA.xlim_lower
    #     states[3,i] = 0
    states[:, i + 1] = rk4(pendulumDynamics, states[:, i], u=np.matrix(1000), step=step)


fig, ax = plt.subplots()
line1 = ax.plot(0, 0, marker=".")[0]
ax.set(
    xlim=[-1, 1],
    ylim=[-1, 1],
)


def update_plot(frame):
    frame = frame * 2
    x = T[frame]
    y1 = states[0, frame]
    y2 = states[1, frame]
    y3 = states[2, frame]

    L1 = PENDULUM_DATA.L1
    L2 = PENDULUM_DATA.L2
    L1_tip_pos_x = L1 * np.sin(y2)
    L1_tip_pos_y = L1 * np.cos(y2)

    L2_tip_pos_x = L2 * np.sin(y3)
    L2_tip_pos_y = L2 * np.cos(y3)
    # print(L1_tip_pos_x)

    line1.set_xdata([y1, y1 + L1_tip_pos_x, y1 + L1_tip_pos_x + L2_tip_pos_x])
    line1.set_ydata([0, L1_tip_pos_y, L1_tip_pos_y + L2_tip_pos_y])

    # print("Updates")
    return line1


ani = animation.FuncAnimation(
    fig=fig, func=update_plot, frames=int(T.shape[0] / 2), interval=150
)
plt.show()
