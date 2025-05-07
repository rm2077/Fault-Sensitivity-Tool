from mohrs_3D_v2 import mohrs_3D_v2
from monte_carlo import monte_carlo
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

def ppfail_MC(variables, dist_params, unit_type, nruns=1000, plot=True):
    dists = {}
    params = {}
    ppfail_results = []

    units = {
        'Sv_grad': {'SI': 'Mpa/km', 'Imperial': 'PSI/ft'},
        'mu': 'Coefficient of Friction',
        'strike': 'Degrees',
        'dip': 'Degrees',
        'Pp': {'SI': 'Mpa/km', 'Imperial': 'PSI/ft'},
        'SHmax_or': 'Degrees',
        'Aphi': 'Relative Stress Magnitude',
        'SHmax_mag': {'SI': 'Mpa/km', 'Imperial': 'PSI/ft'},
        'Shmin_mag': {'SI': 'Mpa/km', 'Imperial': 'PSI/ft'}
    }

    for key in dist_params:
        if key.endswith("_dist"):
            base_var = key[:-5]
            dists[base_var] = dist_params[key]
        elif key.endswith("_param"):
            base_var = key[:-6]
            params[base_var] = dist_params[key]

    num_faults = len(variables["strike"])
    num_vars = len(dists)

    for fault_idx in range(num_faults):
        ncols = int(np.ceil(np.sqrt(num_vars)))
        nrows = int(np.ceil(num_vars / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
        fig.subplots_adjust(top=0.95)
        fig.suptitle(f"Monte Carlo Histograms - {variables['name'][fault_idx]}", fontsize=20)
        axes = axes.flatten()

        for i, var in enumerate(list(dists.keys())):
            dist_type = dists[var]
            dist_param = params.get(var)
            result, perturbed = monte_carlo(mohrs_3D_v2, variables, dist_type, dist_param, var, nruns)
            ppfail_results.append(result[:, fault_idx])
            data = perturbed[:, fault_idx]

            ax = axes[i]
            if np.any(np.isnan(data)):
                ax.set_visible(False)
                continue

            unit_label = units.get(var, "")
            if isinstance(unit_label, dict):
                unit_label = unit_label.get(unit_type, "")

            ax.hist(data, bins=25, edgecolor="black")
            ax.axvline(np.mean(data), color='orange', linewidth=3, label="Mean")
            counts, _ = np.histogram(data, bins=25)
            ax.axhline(np.mean(counts), color='red', linewidth=3, label="Shape")

            ax.legend()
            ax.set_title(var)
            ax.set_xlabel(unit_label)
            ax.set_ylabel("Number of Realizations")

        if plot:
            plt.show()

    fig, ax = plt.subplots()
    cmap = cm.get_cmap('RdYlGn')
    norm = colors.Normalize(vmin=0, vmax=3500)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    for fault_idx in range(num_faults):
        # Collect the ppfail distribution from the Monte Carlo result
        ppfail_values = np.sort(ppfail_results[fault_idx])
        cum_prob = np.linspace(0, 1, len(ppfail_values))
        color = scalar_map.to_rgba(np.mean(ppfail_values))
        ax.plot(ppfail_values, cum_prob, label=variables["name"][fault_idx], color=color)

    scalar_map.set_array([0, 3500])
    cbar = plt.colorbar(scalar_map, ax=ax, orientation='horizontal')
    cbar.set_label("Delta PP to slip")

    ax.set_xlabel("Delta Pore Pressure to Slip")
    ax.set_ylabel("Probability of Fault Slip")
    ax.set_title("Cumulative Distribution Delta Pore Pressure to Slip")
    ax.grid(True)
    ax.legend(loc="lower right")
    plt.show()