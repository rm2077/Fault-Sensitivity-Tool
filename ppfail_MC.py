import matplotlib.pyplot as plt
from mohrs_3D_v2 import mohrs_3D_v2
from monte_carlo import monte_carlo

def ppfail_MC(indatacell, sigcell, dist_params, nruns = 1000):
    if indatacell is None or sigcell is None or dist_params is None:
        raise ValueError('Missing input data for Monte Carlo simulation')

    dists = {}
    params = {}
    results = {}

    for key in dist_params.keys():
        if key.endswith('_dist'):
            dists[key] = dist_params[key]
            param_key = key.replace('_dist', '_param')

            if param_key in dist_params:
                params[param_key] = dist_params[param_key]

    for var in dists.keys():
        base_var = var.replace('_dist', '')
        dist_type = dists[var]
        param_value = params.get(f"{base_var}_param")

        out, inj = monte_carlo(mohrs_3D_v2, indatacell, sigcell, nruns = nruns, dist = dist_type, dist_param = param_value)
        results[base_var] = out

        # Plot figure
        plt.figure()
        plt.hist(out, density = True, alpha = 0.7, color = "steelblue", ec = "black")
        plt.xlabel(f"Pore Pressure to Failure due to {base_var}")
        plt.ylabel("Probability Density")
        plt.title("Probability Density Function of Monte Carlo Simulation ({base_var})")
        plt.grid(True)
        plt.show()

    return results