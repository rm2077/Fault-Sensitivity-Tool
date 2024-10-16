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
    

    return Sig0,ixSv, strike, dip, SHdir, p0, dp, mu, biot, nu