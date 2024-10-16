import numpy as np
import random

'''
Inputs:
f:     function handle of map to run Monte Carlo On
in0:   cell array of nominal inputs to function, i.e. f(in0) produces the function output
inSig: sigma (gaussian) cell array for the sigma associated with each element of in0 or delta +/- in uniform distribution
nruns: how many Monte Carlo Runs you want to do

Outputs:
Out: desired outputs
inj: inputted variables
'''

def monte_carlo(f, in0, inSig, nruns):
    out = np.zeros(nruns)	        # Preallocate
    inj = np.zeros(nruns, len(in0))	# Preallocate

    for jj in range(nruns):
        inj = in0   # Reassign
        for k in range(len(in0)):
            inj[jj][k] = inj[k] + inSig[k] * random.randint(1, len(inSig[k]))               # Normal Distribution
            inj[jj][k] = in0[k] + inSig[k] * (2 * random.randint(1, len(inSig[k])) - 1)     # Uniform Distribution
            inj[jj][k] = np.random.lognormal(np.log(in0[k]), np.log(1 + inSig[k] / in0[k])) # Lognormal Distribution

    out[jj] = f(inj[jj, :])

    return out, inj