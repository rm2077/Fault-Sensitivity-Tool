import pandas as pd

def import_data():
    # Import CSV files
    params_df = pd.read_csv('Params Input Table.csv')
    faults_df = pd.read_csv('Faults Input Table.csv')

    # Define column names
    column_names = [
        'seg_x_cntrpt', 'seg_y_cntrpt', 'seg_len', 'strike', 'strike_dist', 'strike_param', 'dip', 
        'dip_dist', 'dip_param', 'mu', 'mu_dist', 'mu_param', 'SHdir', 'SHdir_dist', 'SHdir_param', 
        'p0', 'p0_dist', 'p0_param', 'dp', 'dp_dist', 'dp_param', 'APhi', 'APhi_dist', 'APhi_param', 
        'Sv_mag', 'Sv_mag_dist', 'Sv_mag_param', 'SHmax_mag', 'SHmax_mag_dist', 'SHmax_mag_param', 
        'Shmin_mag', 'Shmin_mag_dist', 'Shmin_mag_param', 'ref_mu', 'biot', 'nu'
    ]

    # Store parameters
    params = {}

    for col, name in zip(df.columns, column_names):
        params[name] = df[col].iloc[0] if not df[col].isnull().all() else None

    # Set default values if any parameter is empty
    defaults = {
        'seg_len': 10000,
        'mu': 0.6,
        'mu_dist': 'Normal',
        'mu_param': 0.05,
        'p0': 0.44 * params.get('seg_len', 10000),
        'Sv_mag': 1.1 * params.get('seg_len', 10000),
        'ref_mu': 0.6,
        'biot': 1.0,
        'nu': 0.5
    }

    # Update parameters with defaults
    for key, value in defaults.items():
        if not params.get(key):
            params[key] = value

    return params

import_data()