'''
Run Monte Carlo simulation on 3D Mohr's Circle function with stochastic inputs

Inputs:
f       : function
    Function handle
in0     : array
    Baseline inputs for f
inSig   : array
    Variance values for each element of in0
nruns   : int
    Monte Carlo runs, default: Uniform
dist    : str, default: Uniform
    Distribution ("Uniform", "Normal", or "Lognormal")

Outputs:
out     : Array
    Output values from each run
inj     : Array of arrays
    Perturbed input values for each run
'''

import numpy as np

def monte_carlo(f, in0, inSig, nruns = 1000, dist = "Uniform"):
    # Preallocate output arrays
    out = np.zeros(nruns)
    inj = [in0.copy() for _ in range(nruns)]

    # Run Monte Carlo simulation
    for i in range(nruns):
        # Perturb each input according to specified distribution
        for j in range(len(in0)):
            if dist == "Uniform":
                inj[i][j] = np.random.uniform(in0[j] - inSig[j], in0[j] + inSig[j])
            elif dist == "Normal":
                inj[i][j] = np.random.normal(in0[j], inSig[j])
            elif dist == "Lognormal":
                inj[i][j] = np.random.lognormal(np.log(in0[j]), inSig[j])
            else:
                raise ValueError("Unknown distribution specified.")

        # Run function with perturbed inputs
        result = f(*inj[i])
        out[i] = result[0]

    return out, inj