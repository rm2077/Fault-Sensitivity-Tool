import numpy as np

def monte_carlo(f, in0, idx, std, dist, nruns=1000):
    in0 = np.array(in0)
    inj = np.tile(in0, (nruns, 1))

    if dist == "uniform":
        inj[:, idx] = np.random.uniform(in0[idx] - std, in0[idx] + std, nruns)
    elif dist == "normal":
        inj[:, idx] = np.random.normal(in0[idx], std, nruns)
    elif dist == "lognormal":
        sigma_ln = np.sqrt(np.log(1 + (std / in0[idx]) ** 2))
        mean_ln = np.log(in0[idx]) - 0.5 * sigma_ln**2
        inj[:, idx] = np.random.lognormal(mean_ln, sigma_ln, nruns)
    elif np.isnan(dist):
        return None, None
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    results = np.apply_along_axis(lambda params: f(*params), 1, inj)
    return results, inj