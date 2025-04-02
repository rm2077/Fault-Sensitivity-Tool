import numpy as np

def get_hor_from_APhi(APhi, Sv, ref_mu, p0):
    if 0 <= APhi < 1:
        n = 0
    elif 1 <= APhi < 2:
        n = 1
    elif 2 <= APhi <= 3:
        n = 2

    Phi = (APhi - (n + 0.5)) / (-1)**n + 0.5
    k = (ref_mu + np.sqrt(1 + ref_mu**2))**2

    if n == 0:
        Sh = (Sv - p0) / k + p0
        SH = Phi * (Sv - Sh) + Sh
    elif n == 1:
        A = np.array([[1, -k], [Phi, 1 - Phi]])
        b = np.array([p0 - k * p0, Sv])
        x = np.linalg.solve(A, b)

        SH = x[0]
        Sh = x[1]
    elif n == 2:
        SH = k * (Sv - p0) + p0
        Sh = Phi * (SH - Sv) + Sv

    return SH, Sh