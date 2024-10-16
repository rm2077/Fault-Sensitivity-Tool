import matplotlib.pyplot as plt
import monte_carlo
import mohrs_3D_v2
'''
Pore pressure to failure Monte Carlo Calculation. Runs the mohrs_3D.m code
with stochastic inputs

Inputs:
indatacell: Cell array of baseline inputs into mohrs_3D.m
sigcell: Cell array of uniform standard deviations corresponding to each input in indatacell
dpmax: Maximum dp range over which to run the calculation
nruns: How many runs for each set of inputs to compute results
'''
def ppfail_MC(indatacell = None, sigcell = None, nruns = 1000):
    if indatacell is None or sigcell is None:
        Sig0 = [250, 200, 100]
        ixSv = 2
        strike = 90
        dip = 90
        SHdir = 60
        p0 = 10
        dp = 0
        mu = 0.6
        biot = 1
        nu = 0.25
        indatacell = [Sig0, ixSv, p0, strike, dip, SHdir, dp, mu, biot, nu]
        sigcell = [[3, 3, 3], 0, 0, 0, 0, 0, 0, 0, 0, 0]

    outs, inj = monte_carlo(mohrs_3D_v2, indatacell, sigcell, nruns)

    if indatacell == None:
        plt.figure()
        plt.hist(outs, bins = 50, density = True, cumulative = True, label = "CDF", linewidth = 2)
        plt.xlabel("PP fail")
        plt.ylabel("CDF Probability")
        plt.grid(True)
        plt.show()

    return outs, inj