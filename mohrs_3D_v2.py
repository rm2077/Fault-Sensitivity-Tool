import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from get_hor_from_APhi import get_hor_from_APhi

# Get parameter values
def get_value(x, i):
    return x[i] if hasattr(x, '__getitem__') else x

def mohrs_3D_v2(strike, dip, mu, SHmax_or, Pp, Aphi, Sv_grad, SHmax_mag, Shmin_mag, ref_mu, name, biot=1.0, nu=0.5, plot=True, data=True):
    num_faults = len(strike)

    # Store per-fault outputs
    ppfail_list = []
    cff_list = []
    scu_list = []
    sig_fault_list = []
    tau_fault_list = []
    fault_sigmas = []
    fault_taus = []

    # Determine number of plots for Mohr's Circle
    scalar_inputs = [mu, SHmax_or, Pp, Aphi, Sv_grad, ref_mu]
    all_scalar = all(np.isscalar(x) for x in scalar_inputs)
    if Aphi is None and not any(hasattr(x, '__getitem__') for x in [SHmax_mag, Shmin_mag]):
        all_scalar = True

    # Colorbar initialization
    colors_list = []
    cmap = cm.get_cmap('RdYlGn')
    norm = colors.Normalize(vmin=0, vmax=3500)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(num_faults):
        mu_i = get_value(mu, i)
        SHmax_or_i = get_value(SHmax_or, i)
        Pp_i = get_value(Pp, i)
        Aphi_i = get_value(Aphi, i)
        Sv_grad_i = get_value(Sv_grad, i)
        ref_mu_i = get_value(ref_mu, i)

        # Calculate principal stress components
        if not np.isnan(Aphi_i):
            if not (0 <= Aphi_i <= 3):
                raise ValueError(f'Aphi should be in [0,3]. Got: {Aphi_i}')

            SHmax_mag_i, Shmin_mag_i = get_hor_from_APhi(Aphi_i, Sv_grad_i, ref_mu_i, Pp_i)
            if 0 <= Aphi_i < 1:
                Sig0 = [Sv_grad_i, SHmax_mag_i, Shmin_mag_i]
                ixSv = 0
            elif 1 <= Aphi_i < 2:
                Sig0 = [SHmax_mag_i, Sv_grad_i, Shmin_mag_i]
                ixSv = 1
            else:
                Sig0 = [SHmax_mag_i, Shmin_mag_i, Sv_grad_i]
                ixSv = 2
        else:
            SHmax_mag_i = get_value(SHmax_mag, i)
            Shmin_mag_i = get_value(Shmin_mag, i)
            Sig0 = sorted([Sv_grad_i, SHmax_mag_i, Shmin_mag_i], reverse=True)
            ixSv = Sig0.index(Sv_grad_i)

        # Calculate effective stresses and unit vectors
        Sig = np.array(Sig0) + biot * (1 - 2 * nu) / (1 - nu)
        uSH = [np.cos(np.deg2rad(SHmax_or_i)), np.sin(np.deg2rad(SHmax_or_i)), 0]
        uV = [0, 0, 1]

        if ixSv == 0:
            uSh = np.cross(uV, uSH)
            uS3, uS2, uS1 = uSh, uSH, uV
        elif ixSv == 1:
            uSh = np.cross(uSH, uV)
            uS3, uS2, uS1 = uSh, uV, uSH
        elif ixSv == 2:
            uSh = np.cross(uV, uSH)
            uS3, uS2, uS1 = uV, uSh, uSH

        uF = [-np.sin(np.deg2rad(dip[i])) * np.sin(np.deg2rad(strike[i])), 
            np.sin(np.deg2rad(dip[i])) * np.cos(np.deg2rad(strike[i])), 
            -np.cos(np.deg2rad(dip[i]))]

        Sigv = Sig - Pp_i
        l, m, n = np.dot(uF, uS1), np.dot(uF, uS2), np.dot(uF, uS3)

        # Calculate output data
        sig_fault = Sigv[0] * l**2 + Sigv[1] * m**2 + Sigv[2] * n**2
        tau_fault = np.sqrt((Sigv[0] * l)**2 + (Sigv[1] * m)**2 + (Sigv[2] * n)**2 - sig_fault**2)
        sig_fault *= 100
        tau_fault *= 100
        ppfail = sig_fault - tau_fault / mu_i
        cff = tau_fault - mu_i * sig_fault
        scu = tau_fault / (mu_i * sig_fault)

        # Store output data
        ppfail_list.append(ppfail)
        cff_list.append(cff)
        scu_list.append(scu)
        sig_fault_list.append(sig_fault)
        tau_fault_list.append(tau_fault)

        # Plot individual Mohr's Circles
        if all_scalar:
            fault_sigmas.append(sig_fault)
            fault_taus.append(tau_fault)
            colors_list.append(scalar_map.to_rgba(ppfail))
        elif plot:
            angles = np.linspace(0, 2 * np.pi, 100)
            R1 = 0.5 * (Sig0[0] - Sig0[2])
            R2 = 0.5 * (Sig0[1] - Sig0[2])
            R3 = 0.5 * (Sig0[0] - Sig0[1])
            C1 = (R1 * np.exp(1j * angles) + (Sig0[0] + Sig0[2]) / 2 - Pp_i) * 100
            C2 = (R2 * np.exp(1j * angles) + (Sig0[1] + Sig0[2]) / 2 - Pp_i) * 100
            C3 = (R3 * np.exp(1j * angles) + (Sig0[0] + Sig0[1]) / 2 - Pp_i) * 100

            fig, ax = plt.subplots()
            ax.plot(np.real(C1), np.imag(C1), 'k')
            ax.plot(np.real(C2), np.imag(C2), 'k')
            ax.plot(np.real(C3), np.imag(C3), 'k')
            ax.plot([0, Sig0[0] * 100], mu_i * np.array([0, Sig0[0]]) * 100, 'r', linewidth=2, label='Failure Line')
            ax.plot(sig_fault, tau_fault, 'o', color=scalar_map.to_rgba(ppfail), markeredgecolor='k', label='Fault Point')

            scalar_map.set_array([0, 3500])
            cbar = plt.colorbar(scalar_map, orientation='horizontal', ax=ax, pad=0.15)
            cbar.set_label('Delta PP to slip (kPa)')

            ax.grid()
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r'$\sigma$ effective [kPa]')
            ax.set_ylabel(r'$\tau$ [kPa]')
            ax.set_title(f"Mohr's Circle for Fault {i+1}")
            ax.set_aspect('equal', adjustable='box')
            ax.legend()
            plt.show()

    # Plot singular Mohr's Circles
    if all_scalar and plot:
        angles = np.linspace(0, 2 * np.pi, 100)
        R1 = 0.5 * (Sig0[0] - Sig0[2])
        R2 = 0.5 * (Sig0[1] - Sig0[2])
        R3 = 0.5 * (Sig0[0] - Sig0[1])
        C1 = (R1 * np.exp(1j * angles) + (Sig0[0] + Sig0[2]) / 2 - Pp_i) * 100
        C2 = (R2 * np.exp(1j * angles) + (Sig0[1] + Sig0[2]) / 2 - Pp_i) * 100
        C3 = (R3 * np.exp(1j * angles) + (Sig0[0] + Sig0[1]) / 2 - Pp_i) * 100

        fig, ax = plt.subplots()
        ax.plot(np.real(C1), np.imag(C1), 'k')
        ax.plot(np.real(C2), np.imag(C2), 'k')
        ax.plot(np.real(C3), np.imag(C3), 'k')
        ax.plot([0, Sig0[0] * 100], mu * np.array([0, Sig0[0]]) * 100, 'r', linewidth=2, label='Failure Line')
        
        for sig, tau, col in zip(fault_sigmas, fault_taus, colors_list):
            ax.plot(sig, tau, 'o', color=col, markeredgecolor='k')

        scalar_map.set_array([0, 3500])
        cbar = plt.colorbar(scalar_map, orientation='horizontal', ax=ax, pad=0.15)
        cbar.set_label('Delta PP to slip (kPa)')

        ax.grid()
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$\sigma$ effective [kPA]')
        ax.set_ylabel(r'$\tau$ [kPa]')
        ax.set_title("Mohr's Circle for All Faults")
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        plt.show()

    if data:
        # Output verbose results
        for i in range(num_faults):
            print(f"\n{name[i]}:")
            print(f"Pore pressure to failure for fault (ppfail): {ppfail_list[i]:.3f}")
            print(f"Coulomb failure function (cff): {cff_list[i]:.3f}")
            print(f"Shear capacity utilization (scu): {scu_list[i]:.3f}")
            print(f"Effective normal stress projected onto fault (sig_fault): {sig_fault_list[i]:.3f}")
            print(f"Effective shear stress projected onto fault (tau_fault): {tau_fault_list[i]:.3f}")

    return ppfail