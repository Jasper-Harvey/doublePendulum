import numpy as np
from rk4 import rk4
from scipy.optimize import least_squares
import pendulumDynamics

# Non-linear MPC control for the double pendulum system
# This seems like a bit of a sledge hammer approach and a more computationally efficent
# approach would probably use linear MPC, with more careful consideration given
# to cost function selection, and optimiser...

# Should also consider deriving the gradients of the cost function for the system to increase
# speed.


# This would also benefit from being made into a class, and generating some form of MPC
# parameter structure to hold the details, instead of passing it all to (and defining in) nlmpc()

class ControllerParams:
    dynamics: callable = None
    cost: callable = None
    trajectory: callable = None # function to evaluate a state trajecetory. Used by the cost function.
    nSubsteps: int = None # Number of substeps to do in the errorMPC evaluation
    dt: float = None # Timestep for the solver?
    Np: int = None # Prediction horizon
    Nc: int = None # Control horizon
    qx: float = None
    qth: float = None
    qu: float = None

    u_max: float = None # Input constraint max
    u_min: float = None # Input constraint min

    u_dot_max: float = None # Slew rate penalty
    u_dot_min: float = None # Slew rate penalty

def trajectoryEval(t):
    # Basically just retuns a vector of zeros... We can make this track something cool later...

    return np.zeros((6,1)) # All states zeros


def errorMPC(x: np.matrix, t: float, U: np.array, params: ControllerParams) -> np.array:
    """
    t1 - Current time of the simulation
    x1 - Current states 
    u0 - Current input 
    U - Vector of inputs over the control horizon

    """

    e = np.zeros((4*params.Np,)) # Error vector

    f = params.dynamics # Get the dynamics.
    dt = params.dt / params.nSubsteps

    for k in range(0,params.Np):
        # Loop over the prediction horizon

        if k <= params.Nc:
            u = np.matrix(U[k]) # This needs updating for more inputs! (if we have them)
        else:
            u = np.matrix(0)
        # print(f"u extracted from U: {u}")

        for j in range(0,params.nSubsteps):
            f1 = f(t,          x,                           u)
            f2 = f(t + dt/3,   x + f1*dt/3,                 u)
            f3 = f(t + dt*2/3, x - f1*dt/3 + f2*dt,         u)
            f4 = f(t + dt,     x + f1*dt   - f2*dt + f3*dt, u)
            x = x + (f1 + 3*f2 + 3*f3 + f4)*dt/8

            t = t + dt

        # Evaluate the error:
        X = params.trajectory(x,t) # Get states at the trajectory(t)

        # scipy.optimise.least_squares() requires an array
        # Having a cost function in this form makes calculating the analytical jacobian of the
        # cost function much easier. Makes optimisation a lot easier to do.
        e_pred = np.array([ 
            np.sqrt(params.qx) * (x[0,0] - X[0,0]),
            np.sqrt(params.qth) * (x[1,0] - X[1,0]),
            np.sqrt(params.qth) * (x[2,0] - X[2,0]),
            np.sqrt(params.qu) * u[0,0]
            ])
        e[k*4:(k*4 + 4)] = e_pred
    return e


def errorMPC_energy(x: np.matrix, t: float, U: np.array, params: ControllerParams) -> np.array:
    """
    Uses the kinetic and potential energy terms as the cost function. 
    Simply using the angle is not so good..

    t1 - Current time of the simulation
    x1 - Current states 
    u0 - Current input (just used to determine the size of the input vector)
    U - Vector of inputs over the control horizon

    """
    e = np.zeros((2*params.Np,)) # Error vector

    f = params.dynamics # Get the dynamics.
    dt = params.dt / params.nSubsteps

    for k in range(0,params.Np):
        # Loop over the prediction horizon

        if k <= params.Nc:
            u = np.matrix(U[k]) # This needs updating for more inputs! (if we have them)
        else:
            u = np.matrix(0)

        for j in range(0,params.nSubsteps):
            x = rk4(t, f, x, u, dt)
            t = t + dt

        # Evaluate the error:
        # X = trajectoryEval(t) # Get states at the trajectory(t)
        # Evaluate the target energy:
        Ek_star = pendulumDynamics.Ek(np.zeros((6,1)))
        Ep_star = pendulumDynamics.Ep(np.zeros((6,1)))

        # Evaluate the current energy:
        Ek = pendulumDynamics.Ek(x)
        Ep = pendulumDynamics.Ep(x)

        # scipy.optimise.least_squares() requires an array
        e_pred = np.array([
            params.qx * (Ek - Ek_star),
            params.qth * (Ep - Ep_star)])
        
        e[k*2:(k*2 + 2)] = e_pred
    
    return e



def nlmpc(t: float, x: np.matrix, U: np.array, params : ControllerParams) -> np.array:
    """
    Computes the optimal control sequence over a horizon.
    Inputs:
        - t : Time
        - x : current state
        - U : vector of control inputs over the horizon
        - params : Controller parameters
    """

    # TODO: Slew rate constraints on the input U

    optim_error = lambda U_opt: params.cost(x, t, U_opt, params)

    result = least_squares(optim_error, U, bounds=(params.u_min, params.u_max))
    U = result.x
    cost = result.cost
    
    return U, cost




if __name__ == "__main__":
    # Test that errorMPC returns something not stupid...
    from pendulumDynamics import pendulumDynamics, PENDULUM_DATA


    # Arbitrarily assign some parameters:
    par = ControllerParams()
    par.dynamics = pendulumDynamics
    par.cost = errorMPC
    par.Np = 50
    par.Nc = 50
    par.nSubsteps = 10
    par.dt = 0.1

    par.qx = 0
    par.qth = 1000000
    par.qu = 0

    x_init = np.zeros((6,1))
    x_init[0,0] = 0 # 0.2m away from the origin
    x_init[1,0] = 0.01
    x_init[2,0] = 0

    U = np.zeros((par.Nc,)) # Zeros over the whole control horizon. (for now)

    e = errorMPC(x=x_init, t=0, u=np.zeros(1,), U=U, params=par)

    from scipy.optimize import least_squares

    optim_error = lambda U_opt: errorMPC(x_init, 0, np.matrix(0), U_opt, par)
    # This needs sorting out with the dimension issue. Can only be a np array, but I am using matrixes...
    test = least_squares(optim_error, U)
    print(test)
    print(test.x[0])

