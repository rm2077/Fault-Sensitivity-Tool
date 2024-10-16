import matplotlib.pyplot as plt
import numpy as np

'''
Function calculates horizontal stresses from critical stress assumptions
and input of APhi and reference friction angle mu. Total vertical stress
Sv and pore pressure Pp are the other inputs.

Outputs are the horizontal total stresses SH and Sh
'''
def getHorFromAPhi(APhi, Sv, mu, Pp):  # Is hDv part of GUI?
    # Calculate n and Phi from APhi
    if 0 <= APhi < 1:
        n = 0
    elif 1 <= APhi < 2:
        n = 1
    elif 2 <= APhi <= 3:
        n = 2
    else:
        n = np.nan

    SH = np.nan
    Sh = np.nan

    if not np.isnan(n):
        Phi = (APhi - (n + 0.5)) / (-1) ** n + 0.5
        k = (mu + np.sqrt(1 + mu ** 2)) ** 2

        if n == 0:
            Sh = (Sv - Pp) / k + Pp
            SH = Phi * (Sv - Sh) + Sh
        elif n == 1:
            A = [[1, -k], [Phi, 1 - Phi]]
            b = [Pp - k * Pp, Sv]
            x = np.linalg.solve(A, b)
            SH = x[0]
            Sh = x[1]
        elif n == 2:
            SH = k * (Sv - Pp) + Pp
            Sh = Phi * (SH - Sv) + Sv
    else:
        print('Check Aphi [0,3] range. You have APhi =', APhi)

    return SH, Sh

'''
Mohr's Circle Calculation 3D

Inputs:
Sig0        = Vector of initial total stresses (assume row vector)
ixSv        = Index of vertical stress
p0          = Initial Reference pore pressure corresponding to Sig0
strike, dip = Strikes, dips of faults in degrees
SHdir       = Maximum horizontal stress direction
dp          = Pressure pertubation at each fault
mu          = friction coefficients for faults
aphi if 11th input
reference mu for stresses if 12th input
ax          = axis handle for plotting

Outputs:
C1, C2, C3  = Plotted circles
sig_fault   = Effective normal stress projected onto fault
tau_fault   = Effective shear stress projected onto fault
outs.ppfail = Pore pressure to failure for fault (effective norm stress)
outs.cff    = Couloumb failure function (effective shear stress to fail)
outs.scu    = Shear capacity utilization
failout     = This is the number output that is run as part of Monte Carlo

IMPORTANT! Order of inputs must match ppfail_MC and other MC calcs
'''
def mohrs_3D_v2(indatacell = None, ax = None):
    if indatacell is None and ax is None:
        Sig0 = [250, 200, 100]
        ixSv = 2
        strike = [90]
        dip = 90
        SHdir = 60
        p0 = 10
        dp = 60
        mu = 0.6
        biot = 1
        nu = 0.5
        indatacell = [Sig0, ixSv, p0, strike, dip, SHdir, dp, mu, biot, nu]
        plt.figure()
        ax = plt.axes()

        if True:    # If choosing aphi and mu inputs, set to True
            indatacell.append(1.5)
            indatacell.append(0.6)
    
    elif ax is None:
        ax = []
        
    Sig0 = indatacell[0]
    Sig0.sort(reverse = True)   # Sorting initial total stress
    ixSv = indatacell[1]
    p0 = indatacell[2]
    strike = indatacell[3]
    dip = indatacell[4]
    SHdir = indatacell[5]
    dp = indatacell[6]
    mu = indatacell[7]
    biot = indatacell[8]
    nu = indatacell[9]

    # If more inputs, calculate frictional failure equilibrium assumption
    # inside Monte Carlo algorithm
    if len(indatacell) > 10:
        APhi = indatacell[10]
    if len(indatacell) > 11:    # If no reference mu, take fault friction
        referenceMuForStresses = indatacell[11]
    else:
        referenceMuForStresses = mu

    sVertical = Sig0[ixSv]
    if np.isnan(sVertical): # Make sure not NaN
        sVertical = Sig0[~np.isnan(Sig0)]
        sVertical = sVertical[0]

    SH, Sh = getHorFromAPhi(APhi, sVertical, referenceMuForStresses, p0)

    # Make SHmax and Shmin in appropriate part of principal stress vector
    if ixSv == 0:   # Index of SVertical
        Sig0 = [sVertical, SH, Sh]
    elif ixSv == 1:
        Sig0 = [SH, sVertical, Sh]
    elif ixSv == 2:
        Sig0 = [SH, Sh, sVertical]

    if np.any(np.isnan(dp)):
        print("found DP's = NaN, heads up, setting = 0")

    # dp[np.isnan(dp)] = 0    # Uncertain fix to errors caused on some faults

    N = len(strike)

    # Compute poroelastic total stresses
    Sig = np.kron(Sig0, np.ones((N, 1)))
    Ds = np.kron(biot * (1 - 2 * nu) / (1 - nu) * dp, np.ones((1, 3)))
    Sig += Ds

    # Compute fault stress projection
    tau_fault = np.zeros(N)
    sig_fault = np.zeros(N)

    # Unit vectors
    uSH = [np.cos(SHdir), np.sin(SHdir), 0]
    uV = [0, 0, 1]

    if ixSv == 0:   # Index of SVertical
        uSh = np.cross(uV, uSH)
        uS3, uS2, uS1 = uSh, uSH, uV
    elif ixSv == 1:
        uSh = np.cross(uSH, uV)
        uS3, uS2, uS1 = uSh, uV, uSH
    elif ixSv == 2:
        uSh = np.cross(uV, uSH)
        uS3, uS2, uS1 = uV, uSh, uSH

    phi = strike
    delta = dip
    
    # Effective Stress
    sigv = Sig - (p0 + dp[:, np.newaxis])
    # Fault normal
    uF = [-np.sin(delta) * np.sin(phi),
          np.sin(delta) * np.cos(phi),
          -np.cos(delta)]

    # Direction cosines
    l = np.dot(uF, uS1)
    m = np.dot(uF, uS2)
    n = np.dot(uF, uSh)

    # Normal stress on fault
    sig_fault = np.sum(sigv * [l ** 2, m ** 2, n ** 2], axis = 1)
    # Shear stress on fault
    tau_fault = np.sqrt(np.sum((sigv * [l, m, n]) ** 2, axis = 1) - sig_fault ** 2)

    outs = {}
    outs["pp.fail"] = sig_fault - tau_fault / mu
    outs["cff"] = tau_fault - mu * sig_fault
    outs["scu"] = tau_fault / mu * sig_fault   
    failout = outs["pp.fail"]

    a = np.linspace(0, np.pi)
    c = np.exp(1j * a)
    C1 = np.zeros(N, len(a))
    C2 = np.zeros(N, len(a))
    C3 = np.zeros(N, len(a))

    R1 = 0.5 * (Sig[:, 0] - Sig[:, 2])
    R2 = 0.5 * (Sig[:, 1] - Sig[:, 2])
    R3 = 0.5 * (Sig[:, 0] - Sig[:, 1])

    C1 = R1[:, np.newaxis] * c + (Sig[:, 0] + Sig[:, 2]) / 2 - (p0 + dp)[:, np.newaxis]
    C2 = R2[:, np.newaxis] * c + (Sig[:, 1] + Sig[:, 2]) / 2 - (p0 + dp)[:, np.newaxis]
    C3 = R3[:, np.newaxis] * c + (Sig[:, 0] + Sig[:, 1]) / 2 - (p0 + dp)[:, np.newaxis]

    # Plotting
    if ax is None:
        ax.hold(True)
    else:
        plt.sca(ax)

    if ax == None:
        for i in range(len(N)):
            c1, c2, c3 = C1[i], C2[i], C3[i]

            ax.plot(c1.real, c1.imag)
            ax.plot(c2.real, c2.imag)
            ax.plot(c3.real, c3.imag)

            x = np.linspace(0, Sig0[0])
            ax.plot(x, mu[i] * x, "r", linewidth = 2)

            plt.plot(sig_fault[i], tau_fault[i], "og", markerfacecolor = "g")

        plt.grid(True)
        plt.axis("equal")
        plt.set_xlim([0, Sig0[0]])
        plt.set_ylim([0, 2 * R1])
        plt.xlabel("Sigma effective [bars]")
        plt.ylabel("Tau [bars]")

    return failout, outs, C1, C2, C3, sig_fault, tau_fault