'''
Import data from CSV file

Inputs:
N/A

Outputs:
Sig0    : array
    Initial principal stresses (contains 3 elements)
ixSv    : int
    Vertical principal stress index (1, 2, or 3)
p0      : float
    Initial pore pressure
strike  : int
    Strike fault orientation in degrees
dip     : int
    Dip fault orientation in degrees
SHdir   : int
    Maximum horizontal stress direction in degrees
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
'''

import pandas as pd

def import_data():
    df = pd.read_csv('input.csv')

    Sig0 = df[df.columns[0]].apply(lambda x: list(map(int, x.split(',')))).tolist()
    ixSv = df[df.columns[1]].tolist()
    strike = df[df.columns[2]].tolist()
    dip = df[df.columns[3]].tolist()
    SHdir = df[df.columns[4]].tolist()
    p0 = df[df.columns[5]].tolist()
    dp = df[df.columns[6]].tolist()
    mu = df[df.columns[7]].tolist()
    biot = df[df.columns[8]].tolist()
    nu = df[df.columns[9]].tolist()

    return Sig0, ixSv, strike, dip, SHdir, p0, dp, mu, biot, nu