'''
Pore pressure to failure Monte Carlo Calculation

Inputs:
indatacell : array
    Baseline inputs for mohrs_3D_v2 function
sigcell    : array
    Variance values for each element of indatacell
nruns      : int
    Monte Carlo runs, default: Uniform
dist    : str, default: Uniform
    Distribution ("Uniform", "Normal", or "Lognormal")

Outputs:
out     : Array
    Output values from Monte Carlo simulations
inj     : Array of arrays
    Perturbed input values ffrom Monte Carlo simulations
'''

import matplotlib.pyplot as plt
import monte_carlo
import mohrs_3D_v2

def ppfail_MC(indatacell, sigcell, nruns = 1000, dist = "Uniform"):
    if indatacell is None or sigcell is None:
        raise ValueError(f'Either indatacell or sigcell is missing')

    outs, inj = monte_carlo(mohrs_3D_v2, indatacell, sigcell, nruns, dist)

    # Plot the figure
    plt.figure()
    plt.hist(outs, density = True, alpha = 0.7, color = "steelblue", ec = "black")
    plt.xlabel("Pore Pressure to Failure")
    plt.ylabel("Probability Density")
    plt.title("Probability Density Function of Monte Carlo Simulation Results")
    plt.grid(True)
    plt.show()

    return outs, inj