import pandas as pd

# Default values
DEFAULTS = {
    'units': 'SI',
    'ref_mu': {'SI': 3.0, 'Imperial': 10000.0},
    'name': 'Fault',
    'Sv_grad': {'SI': 25.0, 'Imperial': 1.1},
    'Sv_grad_dist': 'uniform',
    'Sv_grad_param': {'SI': 0.2, 'Imperial': 0.01},
    'mu': 0.6,
    'mu_dist': 'uniform',
    'mu_param': 0.01,
    'strike_dist': 'uniform',
    'strike_param': 5.0,
    'dip_dist': 'uniform',
    'dip_param': 5.0,
    'Pp': {'SI': 10.0, 'Imperial': 0.44},
    'Pp_dist': 'normal',
    'Pp_param': {'SI': 0.2, 'Imperial': 0.01},
    'SHmax_or_dist': 'normal',
    'SHmax_or_param': 5.0,
    'Aphi_dist': 'normal',
    'Aphi_param': 0.1,
    'SHmax_mag_dist': 'normal',
    'SHmax_mag_param': {'SI': 0.2, 'Imperial': 0.01},
    'Shmin_mag_dist': 'normal',
    'Shmin_mag_param': {'SI': 0.2, 'Imperial': 0.01},
}

# Required values
REQUIRED_FAULT_FIELDS = ['center_X', 'center_Y', 'strike', 'dip', 'length']

# Apply default parameters to Faults Input Table
def apply_fault_defaults(faults_df, units):
    for key, default in DEFAULTS.items():
        if key in faults_df and faults_df[key].isnull().all():
            if isinstance(default, dict):
                faults_df[key] = default.get(units)
            else:
                faults_df[key] = default

    return faults_df

# Load input tables
def load_input_tables(params_path, faults_path):
    # Column mapping for Params Input Table
    param_column_mapping = {
        'units': 'units',
        'ref_depth_km_or_ft': 'ref_mu',
        'Sv_gradient_MPa_km_or_psi_ft': 'Sv_grad',
        'Sv_grad_func_form': 'Sv_grad_dist',
        'Sv_grad_param2_MPa_km_or_psi_ft': 'Sv_grad_param',
        'fault_friction_coefficient': 'mu',
        'fric_coef_functional_form': 'mu_dist',
        'fric_coef_param2': 'mu_param',
        'strike_func_form': 'strike_dist',
        'strike_param2_deg': 'strike_param',
        'dip_func_form': 'dip_dist',
        'dip_param2_deg': 'dip_param',
        'background_pore_pressure_grad_MPa_km_or_psi_ft': 'Pp',
        'pore_press_func_form': 'Pp_dist',
        'pore_press_grad_param2_MPa_km_or_psi_ft': 'Pp_param',
        'SHmax_orientation_deg_clockwise_from_N': 'SHmax_or',
        'SHmax_or_func_form': 'SHmax_or_dist',
        'SHmax_or_param2_deg': 'SHmax_or_param',
        'Aphi': 'Aphi',
        'Aphi_func_form': 'Aphi_dist',
        'Aphi_param2': 'Aphi_param',
        'SHmax_magnitude_gradient_if_no_Aphi_MPa_km_or_psi_ft': 'SHmax_mag',
        'SHmax_mag_func_form': 'SHmax_mag_dist',
        'SHmax_mag_param2_Mpa_km_or_psi_ft': 'SHmax_mag_param',
        'Shmin_mag_if_no_Aphi_MPa_km_or_psi_ft': 'Shmin_mag',
        'Shmin_mag_func_form': 'Shmin_mag_dist',
        'Shmin_mag_param2_MPa_km_or_psi_ft': 'Shmin_mag_param'
    }

    # Column mapping for Faults Input Table
    fault_column_mapping = {
        'center_X': 'center_X',
        'center_Y': 'center_Y',
        'strike_deg_clockwise_from_north': 'strike',
        'dip_deg_from_horiz': 'dip',
        'length': 'length',
        'Sv_gradient_MPa_km_or_psi_ft': 'Sv_grad',
        'Sv_grad_func_form': 'Sv_grad_dist',
        'Sv_grad_param2_MPa_km_or_psi_ft': 'Sv_grad_param',
        'fault_friction_coefficient': 'mu',
        'fric_coef_functional_form': 'mu_dist',
        'fric_coef_param2': 'mu_param',
        'strike_func_form': 'strike_dist',
        'strike_param2_deg': 'strike_param',
        'dip_func_form': 'dip_dist',
        'dip_param2_deg': 'dip_param',
        'background_pore_pressure_gradient_at_fault_center_MPa_km_or_psi_ft': 'Pp',
        'pore_press_func_form': 'Pp_dist',
        'pore_press_grad_param2_MPa_km_or_psi_ft': 'Pp_param',
        'SHmax_orientation_fault_center_deg_clockwise_from_N': 'SHmax_or',
        'SHmax_or_func_form': 'SHmax_or_dist',
        'SHmax_or_param2_deg': 'SHmax_or_param',
        'Aphi_at_fault_center': 'Aphi',
        'Aphi_func_form': 'Aphi_dist',
        'Aphi_param2': 'Aphi_param',
        'SHmax_magnitude_gradient_fault_center_if_no_Aphi_MPa_km_or_psi_ft': 'SHmax_mag',
        'SHmax_mag_func_form': 'SHmax_mag_dist',
        'SHmax_mag_param2_Mpa_km_or_psi_ft': 'SHmax_mag_param',
        'Shmin_mag_fault_center_if_no_Aphi_MPa_km_or_psi_ft': 'Shmin_mag',
        'Shmin_mag_func_form': 'Shmin_mag_dist',
        'Shmin_mag_param2_MPa_km_or_psi_ft': 'Shmin_mag_param'
    }

    # Load Params Input Table
    params_df = pd.read_csv(params_path).rename(columns=param_column_mapping)
    if params_df.shape[0] != 1:
        raise ValueError("Params Input Table must contain exactly one row.")

    params_row = params_df.iloc[0].to_dict()
    units = 'Imperial' if params_row.get('units') == 'Imperial' else 'SI'
    if pd.isna(params_row.get("ref_mu")):
        default_value = DEFAULTS["ref_mu"].get(units) if isinstance(DEFAULTS["ref_mu"], dict) else DEFAULTS["ref_mu"]
        params_row["ref_mu"] = default_value

    # Load Faults Input Table
    faults_df = pd.read_csv(faults_path).rename(columns=fault_column_mapping)
    faults_df = apply_fault_defaults(faults_df, units)

    # Validate required fault fields
    for field in REQUIRED_FAULT_FIELDS:
        if faults_df[field].isnull().any():
            raise ValueError(f"Missing required fault field: {field}")

    # Override empty fault parameters
    override_fields = [
        'mu', 'mu_dist', 'mu_param',
        'strike_dist', 'strike_param',
        'dip_dist', 'dip_param',
        'Pp', 'Pp_dist', 'Pp_param',
        'SHmax_or', 'SHmax_or_dist', 'SHmax_or_param',
        'Aphi', 'Aphi_dist', 'Aphi_param',
        'SHmax_mag', 'SHmax_mag_dist', 'SHmax_mag_param',
        'Shmin_mag', 'Shmin_mag_dist', 'Shmin_mag_param'
    ]

    for field in override_fields:
        if field in params_row:
            faults_df[field] = faults_df.get(field, pd.Series([None]*len(faults_df))).fillna(params_row[field])

    return params_row, faults_df