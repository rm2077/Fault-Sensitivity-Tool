import numpy as np

def get_hor_from_APhi(APhi, Sv_grad, Pp, ref_mu):
    if 0 <= APhi < 1:
        n = 0
    elif 1 <= APhi < 2:
        n = 1
    elif 2 <= APhi <= 3:
        n = 2

    Phi = (APhi - (n + 0.5)) / (-1)**n + 0.5
    k = (ref_mu + np.sqrt(1 + ref_mu**2))**2

    if n == 0:
        Sh = (Sv_grad - Pp) / k + Pp
        SH = Phi * (Sv_grad - Sh) + Sh
    elif n == 1:
        A = np.array([[1, -k], [Phi, 1 - Phi]])
        b = np.array([Pp - k * Pp, Sv_grad])
        x = np.linalg.solve(A, b)

        SH = x[0]
        Sh = x[1]
    elif n == 2:
        SH = k * (Sv_grad - Pp) + Pp
        Sh = Phi * (SH - Sv_grad) + Sv_grad

    return SH, Sh