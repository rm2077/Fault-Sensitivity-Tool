import matplotlib.pyplot as plt
import numpy as np
from get_hor_from_APhi import get_hor_from_APhi

def mohrs_3D_v2(strike, dip, mu, SHdir, p0, dp, APhi, Sv_mag, SHmax_mag, Shmin_mag, ref_mu, biot, nu, plot = True):
    # Calculate principal stress components using SHmax_mag and Shmin_mag
    if SHmax_mag and Shmin_mag:
        Sig0 = sorted([Sv_mag, SHmax_mag, Shmin_mag], reverse = True)
        ixSv = Sig0.index(Sv_mag)
    # Calculate principal stress components using APhi
    elif APhi:
        if APhi < 0 or APhi > 3:
            raise ValueError(f'APhi should be in range [0,3]. You have APhi = {APhi}.')

        SHmax_mag, Shmin_mag = get_hor_from_APhi(APhi, Sv_mag, ref_mu, p0)

        if 0 <= APhi < 1:
            Sig0 = [Sv_mag, SHmax_mag, Shmin_mag]
            ixSv = 0
        elif 1 <= APhi < 2:
            Sig0 = [SHmax_mag, Sv_mag, Shmin_mag]
            ixSv = 1
        elif 2 <= APhi <= 3:
            Sig0 = [SHmax_mag, Shmin_mag, Sv_mag]
            ixSv = 2
    # Exit program with error if specified values do not exist.
    else:
        raise Exception('Either SHmax magnitude and Shmin magnitude should exist or APhi should exist')

    # Calculate effective stresses and unit vectors
    Sig = np.array(Sig0) + biot * (1 - 2 * nu) / (1 - nu) * dp
    uSH = [np.cos(np.deg2rad(SHdir)), np.sin(np.deg2rad(SHdir)), 0]
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

    uF = [-np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(strike)), 
          np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(strike)), 
          -np.cos(np.deg2rad(dip))]

    Sigv = Sig - (p0 + dp)
    l, m, n = np.dot(uF, uS1), np.dot(uF, uS2), np.dot(uF, uS3)

    sig_fault = Sigv[0] * l**2 + Sigv[1] * m**2 + Sigv[2] * n**2
    tau_fault = np.sqrt((Sigv[0] * l)**2 + (Sigv[1] * m)**2 + (Sigv[2] * n)**2 - sig_fault**2)

    # Calculate output data
    ppfail = sig_fault - tau_fault / mu
    cff = tau_fault - mu * sig_fault
    scu = tau_fault / (mu * sig_fault)

    # Calculate Mohr's Circle data
    R1 = 0.5 * (Sig0[0] - Sig0[2])
    R2 = 0.5 * (Sig0[1] - Sig0[2])
    R3 = 0.5 * (Sig0[0] - Sig0[1])

    angles = np.linspace(0, 2 * np.pi, 100)
    C1 = R1 * np.exp(1j * angles) + (Sig0[0] + Sig0[2]) / 2 - (p0 + dp)
    C2 = R2 * np.exp(1j * angles) + (Sig0[1] + Sig0[2]) / 2 - (p0 + dp)
    C3 = R3 * np.exp(1j * angles) + (Sig0[0] + Sig0[1]) / 2 - (p0 + dp)

    # Plot Mohr's Circle
    if plot:
        plt.figure()
        plt.plot(np.real(C1), np.imag(C1), color = 'k')
        plt.plot(np.real(C2), np.imag(C2), color = 'k')
        plt.plot(np.real(C3), np.imag(C3), color = 'k')
        plt.plot([0, Sig0[0]], mu * np.array([0, Sig0[0]]), 'r', linewidth = 2, label = 'Failure Line')
        plt.plot(sig_fault, tau_fault, 'og', markerfacecolor = 'g', label = 'Fault Point')

        plt.grid()
        plt.xlim(left = 0, right = Sig0[0])
        plt.ylim(bottom = 0, top = 2 * R1)
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.xlabel(r'$\sigma$ effective [bars]')
        plt.ylabel(r'$\tau$ [bars]')
        plt.legend()
        plt.title('Mohr\'s Circle Analysis for a Single Fault')
        plt.show()

    return ppfail, cff, scu, sig_fault, tau_fault, SHmax_mag, Shmin_mag