import numpy as np

def monte_carlo(f, in0, inSig, nruns, dist, dist_param):
    in0 = np.array(in0)
    inSig = np.array(inSig)
    dist_param = np.array(dist_param)

    # Run Monte Carlo simulation
    if dist == 'Uniform':
        inj = np.random.uniform(, , size = (nruns, len(in0)))
    elif dist == 'Normal':
        inj = np.random.normal(, , size = (nruns, len(in0)))
    elif dist == 'Lognormal':
        inj = np.random.lognormal(, , size = (nruns, len(in0)))
    else:
        raise ValueError(f'Unknown distribution: {dist}')

    # Run function with perturbed inputs
    results = np.apply_along_axis(lambda params: f(*params, plot = False), 1, inj)
    out = np.asarray(results)[:, 0]

    return out, inj