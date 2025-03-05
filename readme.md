# Double Pendulum Simulation

This repo contains a simulation, and hopefully control of a double pendulum system.

The intention is to do Model Predicitve Control of the system in simulation, and eventually carry that over to a real physical setup.

# Derivation
The full derivation of the system can be found in .\derivation\derivation.pdf
It details the derivation of the Lagrangian, and building that up into a first order matrix differential equation.

Simulation is just then a matter of using Runge-Kutta 4 or similar methods to integrate the ODE.