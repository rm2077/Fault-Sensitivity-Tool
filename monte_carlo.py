import numpy as np

def monte_carlo(model, variables, dist_type, dist_param, param_name, nruns):
    n_faults = len(variables["strike"])
    data = {k: np.atleast_1d(v) for k, v in variables.items()}
    input_matrix = {}

    for key, val in data.items():
        if key == param_name:
            continue

        val = np.atleast_1d(val)
        if len(val) == 1:
            val = np.repeat(val, n_faults)

        input_matrix[key] = np.tile(val, (nruns, 1))

    param_base = np.broadcast_to(np.atleast_1d(variables[param_name]), (n_faults,))
    dist_type = np.broadcast_to(np.atleast_1d(dist_type), (n_faults,))
    param_std = np.broadcast_to(np.atleast_1d(dist_param), (n_faults,))

    # Generate perturbations
    perturbed = np.zeros((nruns, n_faults))
    for i in range(n_faults):
        mean = param_base[i]
        param = param_std[i] if len(param_std) > 1 else param_std[0]
        dist = dist_type[i] if len(dist_type) > 1 else dist_type[0]

        if dist == "normal":
            perturbed[:, i] = np.random.normal(mean, param, nruns)
        elif dist == "uniform":
            a = mean - param * np.sqrt(3)
            b = mean + param * np.sqrt(3)
            perturbed[:, i] = np.random.uniform(a, b, nruns)
        elif dist == "lognormal":
            zeta = np.sqrt(np.log(1 + (param / mean) ** 2))
            lambdaa = np.log(mean) - 0.5 * zeta ** 2
            perturbed[:, i] = np.random.lognormal(lambdaa, zeta, nruns)
        else:
            raise ValueError(f"Unknown distribution: {dist}")

    input_matrix[param_name] = perturbed

    # Run Mohrs Circle Model
    results = np.zeros((nruns, n_faults))
    for i in range(nruns):
        for j in range(n_faults):
            inputs = {k: [input_matrix[k][i, j]] for k in input_matrix}
            inputs[param_name] = [perturbed[i, j]]
            inputs["name"] = [f"Fault_{j}"]
            results[i, j] = model(**inputs, plot=False, data=False)

    return results, perturbed