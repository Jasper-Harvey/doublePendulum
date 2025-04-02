from pendulumDynamics import pendulumDynamics, PENDULUM_DATA
from nlmpc import ControllerParams, errorMPC_energy, errorMPC, nlmpc
from rk4 import rk4

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Assign some parameters for the controller:
par = ControllerParams()
# Dynamics:
par.dynamics = pendulumDynamics
# Cost function:
par.cost = errorMPC
# Setpoint:
par.trajectory = lambda t, x: np.zeros((6,1))
# Prediction and control horizons:
par.Np = 5
par.Nc = 5
# Substeps for the controller predictions:
par.nSubsteps = 1
# Timestep for simulation and controller:
par.dt = 0.025

par.qx = 10 # Position penalty
par.qth = 3000 # Angle penalty
par.qu = 0 # Input penalty

# Input bounds:
par.u_max = 200
par.u_min = -200
# Slew rate constraints:
par.u_dot_max = 2
par.u_dot_min = -2

# Create a vector of times for the simulation:
step = par.dt
start = 0
stop = 5
T = np.arange(start=start, stop=stop, step=step)
states = np.matrix(np.zeros((6, T.shape[0])))

# Start upright:
states[:, 0] = np.matrix(
    [
        [0],
        [0.1],
        [0],
        [0],
        [0],
        [0],
    ]
)
U = np.zeros((par.Nc,)) # Zeros over the whole control horizon for initial guess.
u = np.matrix(U[0])
for t in range(0, T.shape[0] - 1):
    # Based on the states at the current time, compute the next input:
    # Pretend we have perfect state measurements...
    U, terminal_cost = nlmpc(t, states[:,t], U, par)
    u = np.matrix(U[0]) # Have to cast as a matrix becuase numpy is annoying.

    # Compute the next states:
    states[:, t + 1] = rk4(t, pendulumDynamics, states[:, t], u, step)

    print(f"t = {(t/T.shape[0]) * stop : .2f}, u = {u[0,0] : .4f}, cost = {terminal_cost}")




fig, ax = plt.subplots()
line1 = ax.plot(0, 0, marker=".")[0]
ax.set(
    xlim=[-1, 1],
    ylim=[-1, 1],
)

def update_plot(frame):
    frame = frame * 2
    x = T[frame]
    # t = (frame/T.shape[0]) * 10
    y1 = states[0, frame]
    y2 = states[1, frame]
    y3 = states[2, frame]

    L1 = PENDULUM_DATA.L1
    L2 = PENDULUM_DATA.L2
    L1_tip_pos_x = L1 * np.sin(y2)
    L1_tip_pos_y = L1 * np.cos(y2)

    L2_tip_pos_x = L2 * np.sin(y3)
    L2_tip_pos_y = L2 * np.cos(y3)

    line1.set_xdata([y1, y1 + L1_tip_pos_x, y1 + L1_tip_pos_x + L2_tip_pos_x])
    line1.set_ydata([0, L1_tip_pos_y, L1_tip_pos_y + L2_tip_pos_y])

    fig.legend(f"t={x}")

    return line1


ani = animation.FuncAnimation(
    fig=fig, func=update_plot, frames=int(T.shape[0] / 2), interval=150
)
plt.show()


