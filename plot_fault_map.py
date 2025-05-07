import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

def plot_fault_map(center_X, center_Y, length, strike, ppfail, names):
    num_faults = len(center_X)
    cmap = cm.get_cmap('RdYlGn')
    norm = colors.Normalize(vmin=0, vmax=3500)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig, ax = plt.subplots()

    for i in range(num_faults):
        x = center_X[i]
        y = center_Y[i]
        l = length[i] / 2
        angle = np.deg2rad(strike[i])

        # Fault endpoints from center using angle
        dx = l * np.sin(angle)
        dy = l * np.cos(angle)
        x_start, x_end = x - dx, x + dx
        y_start, y_end = y - dy, y + dy

        color = scalar_map.to_rgba(ppfail[i])
        ax.plot([x_start, x_end], [y_start, y_end], color=color)
        ax.text(x, y + 0.01, names[i], ha='center', va='bottom', fontsize=9)

    scalar_map.set_array([0, 3500])
    cbar = plt.colorbar(scalar_map, orientation='horizontal', ax=ax, pad=0.15)
    cbar.set_label('Delta PP to slip')

    ax.set_xlabel("x easting (latitude)")
    ax.set_ylabel("y northing (longitude)")
    ax.set_title("Fault Map")
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.show()
