'''
Calculate 3D Mohr's Circle for stress analysis

Inputs:
Sig01   : int
    Initial maximum principal stresses
Sig02   : int
    Initial intermediate principal stresses
Sig03   : int
    Initial minimum principal stresses
ixSv    : int
    Vertical principal stress index (1, 2, or 3)
strike  : int
    Strike fault orientation in degrees
dip     : int
    Dip fault orientation in degrees
SHdir   : int
    Maximum horizontal stress direction in degrees
p0      : float
    Initial pore pressure
dp      : float
    Pressure pertubation
mu      : float
    Friction coefficient
APhi    : float, optional
    Friction angle parameter, default: None
ref_mu  : float, optional
    Reference friction angle, default: None
biot    : float, optional
    Biot coefficient (Poroelasticity), default: 1.0
nu      : float, optional
    Poisson's ratio (Poroelasticity), default: 0.5

Outputs:
ppfail      : float
    Pore pressure at failure (effective normal stress)
cff         : float
    Couloumb failure function
scu         : float
    Shear capacity utilization
sig_fault   : array
    Effective normal stress on fault
tau_fault   : array
    Effective shear stress on fault 
'''

import matplotlib.pyplot as plt
import numpy as np
import get_hor_from_APhi

def mohrs_3D_v2(Sig01, Sig02, Sig03, ixSv, strike, dip, SHdir, p0, dp, mu, APhi = None, ref_mu = None, biot = 1, nu = 0.5):
    Sig0 = [Sig01, Sig02, Sig03]

    if len(Sig0) != 3:
        raise ValueError(f'Sig0 should include 3 elements. You have {len(Sig0)} elements.')

    if ixSv not in [1, 2, 3]:
        raise ValueError(f'ixSv should be 1, 2, or 3. You have ixSv = {ixSv}.')

    # Sort initial stresses in descending order
    Sig0 = np.sort(Sig0)[::-1]

    # Determine vertical stress
    Sv = Sig0[int(ixSv) - 1]

    # Compute horizontal stresses using APhi (if provided)
    if APhi is not None and ref_mu is not None:
        if APhi < 0 or APhi > 3:
            raise ValueError(f'APhi should be in range [0,3]. You have APhi = {APhi}.')

        SH, Sh = get_hor_from_APhi(APhi, Sv, ref_mu, p0)

        if ixSv == 1:
            Sig0 = [Sv, SH, Sh]
        elif ixSv == 2:
            Sig0 = [SH, Sv, Sh]
        elif ixSv == 3:
            Sig0 = [SH, Sh, Sv]

    # Compute effective stresses
    Sig = np.array(Sig0) + biot * (1 - 2 * nu) / (1 - nu) * dp

    # Unit vectors
    uSH = [np.cos(np.deg2rad(SHdir)), np.sin(np.deg2rad(SHdir)), 0]
    uV = [0, 0, 1]

    # Direction cosines based on vertical stress index
    if ixSv == 1:
        uSh = np.cross(uV, uSH)
        uS3 = uSh
        uS2 = uSH
        uS1 = uV
    elif ixSv == 2:
        uSh = np.cross(uSH, uV)
        uS3 = uSh
        uS2 = uV
        uS1 = uSH
    elif ixSv == 3:
        uSh = np.cross(uV, uSH)
        uS3 = uV
        uS2 = uSh
        uS1 = uSH

    # Fault normal vector
    uF = [-np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(strike)), 
          np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(strike)), 
          -np.cos(np.deg2rad(dip))]

    # Effective stresses on fault
    Sigv = Sig - (p0 + dp)
    l = np.dot(uF, uS1)
    m = np.dot(uF, uS2)
    n = np.dot(uF, uS3)

    sig_fault = Sigv[0] * l**2 + Sigv[1] * m**2 + Sigv[2] * n**2
    tau_fault = np.sqrt((Sigv[0] * l)**2 + (Sigv[1] * m)**2 + (Sigv[2] * n)**2 - sig_fault**2)

    # Calculate outputs
    ppfail = sig_fault - tau_fault / mu
    cff = tau_fault - mu * sig_fault
    scu = tau_fault / (mu * sig_fault)

    # Calculate Mohr's Circle data
    R1 = 0.5 * (Sig0[0] - Sig0[2])
    R2 = 0.5 * (Sig0[1] - Sig0[2])
    R3 = 0.5 * (Sig0[0] - Sig0[1])

    # Circle plots
    angles = np.linspace(0, 2 * np.pi, 100)
    C1 = R1 * np.exp(1j * angles) + (Sig0[0] + Sig0[2]) / 2 - (p0 + dp)
    C2 = R2 * np.exp(1j * angles) + (Sig0[1] + Sig0[2]) / 2 - (p0 + dp)
    C3 = R3 * np.exp(1j * angles) + (Sig0[0] + Sig0[1]) / 2 - (p0 + dp)

    # Plotting
    plt.figure()
    plt.plot(np.real(C1), np.imag(C1), label = 'C1', color = 'blue')
    plt.plot(np.real(C2), np.imag(C2), label = 'C2', color = 'orange')
    plt.plot(np.real(C3), np.imag(C3), label = 'C3', color = 'green')
    plt.plot([0, Sig0[0]], mu * np.array([0, Sig0[0]]), 'r', linewidth = 2, label = 'Friction Line')
    plt.plot(sig_fault, tau_fault, 'og', markerfacecolor = 'g', label = 'Fault Point')

    plt.grid()
    plt.axis('equal')
    plt.xlim([0, Sig0[0]])
    plt.ylim([0, 2 * R1])
    plt.xlabel(r'$\sigma$ effective [bars]')
    plt.ylabel(r'$\tau$ [bars]')
    plt.legend()
    plt.title('Mohr\'s Circle Analysis for a Single Fault')
    plt.show()

    return ppfail, cff, scu, sig_fault, tau_fault