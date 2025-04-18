import matplotlib.pyplot as plt
import numpy as np
from mohrs_3D_v2 import mohrs_3D_v2
from monte_carlo import monte_carlo

# Build wrapper for mohrs_3D_v2
def wrapper(fault, param_values, ref_mu, name):
    def f(*params):
        kwargs = fault.copy()

        # Parameters expected by mohrs_3D_v2
        valid_keys = ['strike', 'dip', 'Sv_grad', 'mu', 'Pp', 'SHmax_or', 'Aphi', 'SHmax_mag', 'Shmin_mag']
        kwargs = {key: value for key, value in kwargs.items() if key in valid_keys}

        # Add parameter values to kwargs
        for i, param in enumerate(valid_keys):
            kwargs[param] = param_values[i]

        kwargs['ref_mu'] = ref_mu
        kwargs['name'] = name

        # Call mohrs_3D_v2 with the filtered arguments
        return mohrs_3D_v2(**kwargs, plot=False, data=False)

    return f

def ppfail_MC_single(fault_row, strike, dip, mu, SHmax_or, Pp, Aphi, Sv_grad, SHmax_mag, Shmin_mag, ref_mu, name, nruns=1000):
    fault_dict = fault_row.to_dict()
    f = wrapper(fault_dict, [strike, dip, Sv_grad, mu, Pp, SHmax_or, Aphi, SHmax_mag, Shmin_mag], ref_mu, name)
    in0 = np.array([fault_row[param] for param in ['strike', 'dip', 'Sv_grad', 'mu', 'Pp', 'SHmax_or', 'Aphi', 'SHmax_mag', 'Shmin_mag']])
    results = {}

    for i, param in enumerate(['strike', 'dip', 'Sv_grad', 'mu', 'Pp', 'SHmax_or', 'Aphi', 'SHmax_mag', 'Shmin_mag']):
        dist = fault_row[f"{param}_dist"]
        std = fault_row[f"{param}_param"]

        if dist is None:
            continue

        mc_out, _ = monte_carlo(f, in0, i, std, dist, nruns=nruns)
        if mc_out is not None:
            results[param] = mc_out

    plot_mc_histograms(results, results.keys(), fault_row['name'])
    return results

def ppfail_MC_all(faults_df, strike, dip, mu, SHmax_or, Pp, Aphi, Sv_grad, SHmax_mag, Shmin_mag, ref_mu, name, nruns=1000):
    output = {}

    for idx, row in faults_df.iterrows():
        print(f"Running Monte Carlo for fault: {row['name']}")

        # Pass the individual parameters along with the fault row
        results = ppfail_MC_single(row, strike, dip, mu, SHmax_or, Pp, Aphi, Sv_grad, SHmax_mag, Shmin_mag, ref_mu, name, nruns)
        output[row['name']] = results
    return output

def plot_mc_histograms(results_dict, param_names, fault_name):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]
        ax.hist(results_dict[param], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{param}')
        ax.set_xlabel('ppfail')
        ax.set_ylabel('Frequency')

    fig.suptitle(f'Monte Carlo Histograms for {fault_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# # Plot results
# num_params = len(param_names)
# fig, axes = plt.subplots(2, num_params, figsize=(4 * num_params, 8))
# axes = axes.flatten()

# for j, param in enumerate(param_names):
#     ax_in = axes[j]
#     ax_in.hist(inj[:, j], bins=30, density=True, alpha=0.7, color="orange", edgecolor="black")
#     ax_in.set_title(f"{param} Distribution")
#     ax_in.set_xlabel(param)
#     ax_in.set_ylabel("Density")
#     ax_in.grid(True)

# ax_out = axes[-1]
# ax_out.hist(out, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="black")
# ax_out.set_title(f"ppfail for {fault['name']}")
# ax_out.set_xlabel("Pore Pressure to Failure")
# ax_out.set_ylabel("Density")
# ax_out.grid(True)

# fig.suptitle(f"Monte Carlo Simulation - {fault['name']}")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# return results