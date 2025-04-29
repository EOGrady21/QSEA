import pandas as pd
import numpy as np
from file_handling import QC_FLAG_SEVERITY # Import severity mapping

# Import potentially required libraries (ensure installed)
try:
    from global_land_mask import globe
except ImportError:
    print("Warning: 'global_land_mask' library not found. Test 1.4 (Position on Land) will be skipped.")
    globe = None
try:
    import gsw
except ImportError:
     print("Warning: 'gsw' library not found. Seawater-related tests (2.6, 2.10) may be skipped or limited.")
     gsw = None
try:
    from geopy.distance import geodesic
except ImportError:
    print("Warning: 'geopy' library not found. Speed calculation (Test 1.5) might be less accurate or skipped.")
    geodesic = None


# --- Helper Function to Update Flags ---
def update_flags(df, indices, test_name, flag_value):
    """Updates auto_qc_flag and auto_qc_details for given indices."""
    current_flags = df.loc[indices, 'auto_qc_flag']
    current_severity = current_flags.map(QC_FLAG_SEVERITY).fillna(0) # Handle potential NaN
    new_severity = QC_FLAG_SEVERITY.get(flag_value, 0)

    # Update flag only if the new flag is more severe
    update_mask = new_severity > current_severity
    update_indices = indices[update_mask]
    df.loc[update_indices, 'auto_qc_flag'] = flag_value

    # Append test details regardless of severity update (useful for info)
    # But only if the test actually failed (flag > 1 generally, or specifically 7)
    if flag_value > 1 or flag_value == 7:
        details_col = df.loc[indices, 'auto_qc_details']
        # Ensure we are appending to lists
        for i in indices:
             if isinstance(df.loc[i, 'auto_qc_details'], list):
                 if test_name not in df.loc[i, 'auto_qc_details']: # Avoid duplicates
                     df.loc[i, 'auto_qc_details'].append(test_name)
             else: # Initialize if not a list (e.g., first failure)
                  df.loc[i, 'auto_qc_details'] = [test_name]

# --- Individual QC Test Functions (Placeholders - NEED IMPLEMENTATION) ---

def test_1_1_platform_id(df):
    """Test 1.1: GTSPP Platform Identification."""
    test_name = "1.1_PlatformID"
    # Example: Check if MISSION_DESCRIPTOR is null or empty
    fail_indices = df[df['MISSION_DESCRIPTOR'].isnull() | (df['MISSION_DESCRIPTOR'] == '')].index
    update_flags(df, fail_indices, test_name, 4) # Bad flag (4)
    print(f"{test_name}: Found {len(fail_indices)} failures.")
    # No return needed, modifies df in place

def test_1_2_datetime(df):
    """Test 1.2: GTSPP Impossible Date/Time."""
    test_name = "1.2_DateTime"
    # Check for NaT in the parsed DATETIME column
    fail_indices = df[df['DATETIME'].isna()].index
    update_flags(df, fail_indices, test_name, 4) # Bad flag (4)
    # Add checks for dates too far in the past/future if needed
    # future_date_limit = pd.Timestamp.now() + pd.Timedelta(days=1)
    # past_date_limit = pd.Timestamp('1900-01-01')
    # fail_indices_future = df[df['DATETIME'] > future_date_limit].index
    # fail_indices_past = df[df['DATETIME'] < past_date_limit].index
    # update_flags(df, fail_indices_future, test_name + "_Future", 4)
    # update_flags(df, fail_indices_past, test_name + "_Past", 4)
    print(f"{test_name}: Found {len(fail_indices)} failures (parsing or invalid).")


def test_1_3_location(df):
    """Test 1.3: GTSPP Impossible Location."""
    test_name = "1.3_Location"
    # Basic range checks
    lat_fails = df[(df['DIS_HEADER_SLAT'] < -90) | (df['DIS_HEADER_SLAT'] > 90)].index
    lon_fails = df[(df['DIS_HEADER_SLON'] < -180) | (df['DIS_HEADER_SLON'] > 180)].index
    fail_indices = lat_fails.union(lon_fails)
    update_flags(df, fail_indices, test_name, 4) # Bad flag (4)
    print(f"{test_name}: Found {len(fail_indices)} failures.")


def test_1_4_on_land(df):
    """Test 1.4: GTSPP Position on Land."""
    test_name = "1.4_OnLand"
    if globe is None:
        print(f"{test_name}: Skipped (global_land_mask library not available).")
        return
    try:
        # globe.is_land returns True for land points
        # Ensure lat/lon are valid numbers before checking
        valid_coords = df[['DIS_HEADER_SLAT', 'DIS_HEADER_SLON']].notna().all(axis=1)
        lats = df.loc[valid_coords, 'DIS_HEADER_SLAT']
        lons = df.loc[valid_coords, 'DIS_HEADER_SLON']
        if not lats.empty:
             is_land_mask = globe.is_land(lats, lons)
             fail_indices_valid = lats[is_land_mask].index # Get original df indices
             update_flags(df, fail_indices_valid, test_name, 4) # Bad flag (4)
             print(f"{test_name}: Found {len(fail_indices_valid)} failures.")
        else:
            print(f"{test_name}: No valid coordinates to check.")
    except Exception as e:
        print(f"Error during {test_name}: {e}")

def test_1_5_speed(df):
    """Test 1.5: GTSPP Impossible Speed."""
    test_name = "1.5_Speed"
    max_speed_kmh = 80 # Example threshold (approx 43 knots) - adjust as needed!

    fail_indices_list = []
    # Calculate speed between consecutive points *within the same mission*
    # Requires sorting by mission and time
    df_sorted = df.sort_values(by=['MISSION_DESCRIPTOR', 'DATETIME'])
    df_sorted = df_sorted.dropna(subset=['DATETIME', 'DIS_HEADER_SLAT', 'DIS_HEADER_SLON'])

    if len(df_sorted) < 2:
         print(f"{test_name}: Not enough data points to calculate speed.")
         return

    # Calculate differences between consecutive rows *only if mission is the same*
    df_sorted['TIME_DIFF_S'] = df_sorted.groupby('MISSION_DESCRIPTOR')['DATETIME'].diff().dt.total_seconds()
    df_sorted['LAT_PREV'] = df_sorted.groupby('MISSION_DESCRIPTOR')['DIS_HEADER_SLAT'].shift(1)
    df_sorted['LON_PREV'] = df_sorted.groupby('MISSION_DESCRIPTOR')['DIS_HEADER_SLON'].shift(1)

    # Calculate distance only where diff calculation is valid
    valid_diff = df_sorted['TIME_DIFF_S'].notna() & (df_sorted['TIME_DIFF_S'] > 1) # Avoid division by zero/small intervals
    valid_df = df_sorted[valid_diff].copy()

    distances_km = []
    if geodesic is None:
        print(f"{test_name}: Skipped (geopy library not available for accurate distance).")
        return

    for idx, row in valid_df.iterrows():
        try:
            point1 = (row['DIS_HEADER_SLAT'], row['DIS_HEADER_SLON'])
            point2 = (row['LAT_PREV'], row['LON_PREV'])
            dist = geodesic(point1, point2).km
            distances_km.append(dist)
        except Exception:
            distances_km.append(np.nan) # Handle potential errors in distance calc

    valid_df['DISTANCE_KM'] = distances_km
    valid_df = valid_df.dropna(subset=['DISTANCE_KM'])

    if not valid_df.empty:
        # Speed in km/h
        valid_df['SPEED_KMH'] = (valid_df['DISTANCE_KM'] / valid_df['TIME_DIFF_S']) * 3600
        fail_indices = valid_df[valid_df['SPEED_KMH'] > max_speed_kmh].index
        update_flags(df, fail_indices, test_name, 3) # Doubtful flag (3) for speed
        print(f"{test_name}: Found {len(fail_indices)} failures (Speed > {max_speed_kmh} km/h).")
    else:
        print(f"{test_name}: No valid consecutive points with sufficient time difference found.")


def test_1_6_sounding(df):
    """Test 1.6: GTSPP Impossible Sounding."""
    test_name = "1.6_Sounding"
    # Placeholder: Check against a simple max depth or basic bathymetry
    max_reasonable_depth = 12000 # e.g., Marianas Trench depth
    min_reasonable_depth = -5     # Allow slightly negative for sensor offset near surface

    fail_indices = df[(df['DIS_HEADER_START_DEPTH'].notna()) & \
                      ((df['DIS_HEADER_START_DEPTH'] < min_reasonable_depth) | \
                       (df['DIS_HEADER_START_DEPTH'] > max_reasonable_depth))].index
    update_flags(df, fail_indices, test_name, 4) # Bad flag (4)
    # TODO: Implement check against actual bathymetry data (requires external dataset and lookup)
    print(f"{test_name}: Found {len(fail_indices)} failures (Basic depth range check). Needs bathymetry integration.")


def test_2_1_global_range(df, external_data):
    """Test 2.1: Global Impossible Parameter Values."""
    test_name = "2.1_GlobalRange"
    global_ranges = external_data.get('global_ranges')
    if global_ranges is None:
        print(f"{test_name}: Skipped (Global ranges data not loaded).")
        return

    fail_count = 0
    # Assuming global_ranges has columns like 'DATA_TYPE_METHOD', 'MinValue', 'MaxValue'
    # Iterate through parameter types defined in the ranges file
    for _, param_range in global_ranges.iterrows():
        param = param_range['DATA_TYPE_METHOD']
        min_val = param_range['MinValue']
        max_val = param_range['MaxValue']

        # Find rows matching the parameter type
        param_mask = (df['DATA_TYPE_METHOD'] == param) & df['DIS_DETAIL_DATA_VALUE'].notna()
        param_indices = df[param_mask].index

        if not param_indices.empty:
            # Check values outside the defined range
            fail_mask = (df.loc[param_indices, 'DIS_DETAIL_DATA_VALUE'] < min_val) | \
                        (df.loc[param_indices, 'DIS_DETAIL_DATA_VALUE'] > max_val)
            specific_fail_indices = param_indices[fail_mask]

            if not specific_fail_indices.empty:
                update_flags(df, specific_fail_indices, f"{test_name}_{param}", 4) # Bad flag (4)
                fail_count += len(specific_fail_indices)

    print(f"{test_name}: Found {fail_count} total failures.")

def test_2_2_regional_range(df, external_data):
    """Test 2.2: Regional Impossible Parameter Values."""
    test_name = "2.2_RegionalRange"
    regional_ranges = external_data.get('regional_ranges')
    if regional_ranges is None:
        print(f"{test_name}: Skipped (Regional ranges data not loaded).")
        return
    # TODO: Implement logic based on regional definitions (e.g., lat/lon boxes)
    # Needs structure for regional_ranges.csv (e.g., MinLat, MaxLat, MinLon, MaxLon, DATA_TYPE_METHOD, MinValue, MaxValue)
    print(f"{test_name}: Placeholder - Needs Implementation based on regional definitions.")
    pass # Placeholder

def test_2_4_profile_envelope(df, external_data):
    """Test 2.4: Profile Envelope test."""
    test_name = "2.4_ProfileEnvelope"
    envelopes = external_data.get('profile_envelopes')
    if envelopes is None:
        print(f"{test_name}: Skipped (Profile envelope data not loaded).")
        return
    # TODO: Implement logic comparing profile data against depth-dependent envelopes
    # Requires grouping by profile ID (e.g., EVENT_COLLECTOR_EVENT_ID)
    # Needs structure for profile_envelopes.csv (e.g., DATA_TYPE_METHOD, DepthBin, MinValue, MaxValue)
    # Interpolation might be needed if envelope depths don't match data depths exactly.
    print(f"{test_name}: Placeholder - Needs Implementation (group by profile, compare vs depth bins).")
    pass # Placeholder


def test_2_5_constant_profile(df):
    """Test 2.5: Check for constant profiles."""
    test_name = "2.5_ConstantProfile"
    fail_count = 0
    # Group by profile ID and parameter type
    # A profile is constant if std deviation is very close to zero (or min == max)
    grouped = df.dropna(subset=['DIS_DETAIL_DATA_VALUE']).groupby(['EVENT_COLLECTOR_EVENT_ID', 'DATA_TYPE_METHOD'])

    for name, group in grouped:
        # Check only if there's more than one point in the profile for this parameter
        if len(group) > 1:
            # Use a small tolerance for floating point comparisons
            is_constant = group['DIS_DETAIL_DATA_VALUE'].nunique() == 1
            # Alternative using std dev (might need adjustment for scale)
            # is_constant = group['DIS_DETAIL_DATA_VALUE'].std() < 1e-9

            if is_constant:
                # Flag all points in this constant profile group
                fail_indices = group.index
                update_flags(df, fail_indices, f"{test_name}_{name[0]}_{name[1]}", 3) # Doubtful flag (3)
                fail_count += len(fail_indices)

    print(f"{test_name}: Found {fail_count} points belonging to constant profiles.")


def test_2_6_freezing_point(df):
    """Test 2.6: Check Temperature against freezing point."""
    test_name = "2.6_FreezingPoint"
    if gsw is None:
         print(f"{test_name}: Skipped (gsw library not available).")
         return

    fail_count = 0
    # Identify Temperature and Salinity data (adjust method names if needed)
    temp_mask = df['DATA_TYPE_METHOD'].str.contains("TEMP", case=False, na=False)
    # Need salinity (Practical Salinity, PSU) to calculate freezing point
    # Assume salinity is available within the same profile (EVENT_ID)
    # This requires merging/lookup within profiles - can be complex/slow without optimization

    # Iterate through profiles
    for event_id, profile_df in df.groupby('EVENT_COLLECTOR_EVENT_ID'):
        temps = profile_df[profile_df['DATA_TYPE_METHOD'].str.contains("TEMP", case=False, na=False)]
        sals = profile_df[profile_df['DATA_TYPE_METHOD'].str.contains("PSAL", case=False, na=False)] # Assuming PSAL for Practical Salinity

        if temps.empty or sals.empty:
            continue

        # Simple approach: Check each temp against nearest depth salinity
        # More robust: Interpolate salinity to temperature depths
        for idx, temp_row in temps.iterrows():
            depth = temp_row['DIS_HEADER_START_DEPTH']
            temp_val = temp_row['DIS_DETAIL_DATA_VALUE']

            if pd.isna(depth) or pd.isna(temp_val):
                continue

            # Find closest salinity measurement by depth
            sals['depth_diff'] = abs(sals['DIS_HEADER_START_DEPTH'] - depth)
            closest_sal_row = sals.loc[sals['depth_diff'].idxmin()]
            sal_val = closest_sal_row['DIS_DETAIL_DATA_VALUE']
            # pressure = gsw.p_from_z(-depth, temp_row['DIS_HEADER_SLAT']) # Convert depth to pressure

            if pd.notna(sal_val): # and pd.notna(pressure):
                try:
                    # Calculate in-situ freezing point (requires Absolute Salinity, temp, pressure)
                    # Using gsw.CT_freezing for Conservative Temp might be better if CT is available
                    # Simpler gsw.fp_t_exact requires Temp, Salinity, Pressure
                    # Using simplified approximation if pressure calculation is complex:
                    # gsw.SP_from_SK() # if salinity is Knudsen
                    # gsw.SA_from_SP() # Convert Practical Salinity SP to Absolute Salinity SA
                    # gsw.p_from_z()   # Convert depth Z to pressure P
                    # fp = gsw.freezing_point_poly(sal_val, pressure, 0) # Approximation
                    # Using more direct GSW function if possible:
                    abs_sal = gsw.SA_from_SP(sal_val, 0, temp_row['DIS_HEADER_SLON'], temp_row['DIS_HEADER_SLAT']) # p=0 at surface approx
                    freezing_pt = gsw.t_freezing(abs_sal, 0, 0) # t_freezing(SA, p, saturation_fraction=0)

                    # Check if temperature is below freezing point (allow small tolerance)
                    tolerance = 0.05 # Degrees C tolerance
                    if temp_val < (freezing_pt - tolerance):
                        update_flags(df, pd.Index([idx]), f"{test_name}_{event_id}", 4) # Bad flag (4)
                        fail_count += 1
                except Exception as e:
                    # print(f"Warning: Could not calculate freezing point for idx {idx}: {e}")
                    pass # Ignore points where calculation fails

    print(f"{test_name}: Found {fail_count} potential failures (Temp < Freezing Point).")


def test_2_7_replicates(df):
    """Test 2.7: Replicate Comparisons (Chlorophyll, Phaeophytin)."""
    test_name = "2.7_Replicates"
    fail_count = 0

    # Define parameters and their replicate checks
    replicate_checks = {
        'CHLA': { # Assuming 'CHLA' or similar in DATA_TYPE_METHOD for Chlorophyll
            'thresholds': [(0.3, 0.05), (3.0, 0.3), (float('inf'), 0.7)], # (upper_bound, delta)
            'method_pattern': 'CHLA' # Pattern to find in DATA_TYPE_METHOD
        },
        'PHAE': { # Assuming 'PHAE' or similar for Phaeophytin
            'thresholds': [(0.3, 0.05), (3.0, 0.5), (float('inf'), 0.7)],
            'method_pattern': 'PHAE' # Pattern to find in DATA_TYPE_METHOD
        }
        # Add other parameters if needed
    }

    # Identify potential replicates: same event, depth, parameter type, but different sample ID?
    # Group by keys that should be identical for replicates except the sample ID itself
    group_keys = ['EVENT_COLLECTOR_EVENT_ID', 'DATA_TYPE_METHOD', 'DIS_HEADER_START_DEPTH'] # Add others?
    # Filter only relevant parameters first for efficiency
    param_masks = [df['DATA_TYPE_METHOD'].str.contains(p['method_pattern'], na=False, case=False) for p in replicate_checks.values()]
    combined_mask = pd.concat(param_masks, axis=1).any(axis=1)
    relevant_df = df[combined_mask & df['DIS_DETAIL_COLLECTOR_SAMP_ID'].notna()].copy() # Ensure SAMP_ID exists

    if relevant_df.empty:
        print(f"{test_name}: No relevant data for replicate checks found.")
        return

    grouped = relevant_df.groupby(group_keys)

    for name, group in grouped:
        # We need at least two samples to compare
        if len(group['DIS_DETAIL_COLLECTOR_SAMP_ID'].unique()) > 1:
            param_method = name[1] # DATA_TYPE_METHOD from group key
            param_key = None
            for key, config in replicate_checks.items():
                if config['method_pattern'] in param_method:
                    param_key = key
                    break

            if param_key:
                thresholds = replicate_checks[param_key]['thresholds']
                values = group['DIS_DETAIL_DATA_VALUE'].dropna()

                if len(values) > 1:
                    mean_val = values.mean()
                    max_diff = values.max() - values.min()

                    # Find appropriate delta threshold based on mean value
                    delta_limit = None
                    for upper_bound, delta in thresholds:
                        if mean_val < upper_bound:
                            delta_limit = delta
                            break

                    if delta_limit is not None and max_diff > delta_limit:
                        # Flag all points in this replicate group
                        fail_indices = group.index
                        update_flags(df, fail_indices, f"{test_name}_{param_key}_{name[0]}", 7) # Requires Investigation (7)
                        fail_count += len(fail_indices)

    print(f"{test_name}: Found {fail_count} points failing replicate comparison (Flag 7).")


def test_2_8_bottle_ctd(df):
    """Test 2.8: Bottle versus CTD Measurements."""
    test_name = "2.8_BottleVsCTD"
    # Parameters to compare: TEMP, PSAL, DOXY, PHPH (adjust method names)
    params_to_compare = ['TEMP', 'PSAL', 'DOXY', 'PHPH']
    # Depth tolerance for matching Bottle and CTD samples (in meters)
    depth_tolerance = 5 # Adjust as needed

    fail_count = 0

    # Iterate through profiles
    for event_id, profile_df in df.groupby('EVENT_COLLECTOR_EVENT_ID'):
        # Separate Bottle and CTD data for the parameters of interest
        bottle_mask = profile_df['DATA_TYPE_METHOD'].str.contains('BOTTLE', case=False, na=False)
        ctd_mask = profile_df['DATA_TYPE_METHOD'].str.contains('CTD', case=False, na=False)
        param_mask = profile_df['DATA_TYPE_METHOD'].str.contains('|'.join(params_to_compare), case=False, na=False)

        bottle_data = profile_df[bottle_mask & param_mask].copy()
        ctd_data = profile_df[ctd_mask & param_mask].copy()

        if bottle_data.empty or ctd_data.empty:
            continue

        # For each bottle sample, find nearby CTD samples of the same parameter type
        for idx, bottle_row in bottle_data.iterrows():
            bottle_depth = bottle_row['DIS_HEADER_START_DEPTH']
            bottle_param = bottle_row['DATA_TYPE_METHOD'] # Or extract base param type
            bottle_value = bottle_row['DIS_DETAIL_DATA_VALUE']

            if pd.isna(bottle_depth) or pd.isna(bottle_value):
                continue

            # Find matching CTD data
            nearby_ctd = ctd_data[
                (ctd_data['DATA_TYPE_METHOD'].str.contains(bottle_param.split('_')[0], case=False, na=False)) & # Match base parameter
                (abs(ctd_data['DIS_HEADER_START_DEPTH'] - bottle_depth) <= depth_tolerance) &
                (ctd_data['DIS_DETAIL_DATA_VALUE'].notna())
            ]

            if not nearby_ctd.empty:
                # Compare bottle value to the mean (or median) of nearby CTD values
                ctd_mean = nearby_ctd['DIS_DETAIL_DATA_VALUE'].mean()
                # TODO: Define acceptable difference thresholds per parameter (e.g., temp_diff > 0.5, sal_diff > 0.1)
                # This requires specific thresholds based on expected sensor differences.
                # Example placeholder check:
                difference = abs(bottle_value - ctd_mean)
                threshold = 0.5 # Placeholder threshold - DEFINE PROPERLY PER PARAMETER
                param_short_name = bottle_param.split('_')[0] # Rough extraction

                if difference > threshold:
                     # Flag both the bottle sample and the compared CTD samples? Or just bottle? Flagging bottle here.
                     update_flags(df, pd.Index([idx]), f"{test_name}_{param_short_name}_{event_id}", 7) # Requires Investigation (7)
                     # Optionally flag the CTD points as well: update_flags(df, nearby_ctd.index, ...)
                     fail_count += 1


    print(f"{test_name}: Found {fail_count} Bottle samples potentially inconsistent with CTD (Flag 7). Needs parameter-specific thresholds.")

def test_2_9_gradient_inversion(df):
    """Test 2.9: Excessive Gradient or Inversion."""
    test_name = "2.9_GradientInversion"
    # TODO: Define thresholds per parameter (e.g., dTemp/dDepth, dSal/dDepth)
    gradient_thresholds = {
        'TEMP': 1.0, # Max change in deg C per meter (example)
        'PSAL': 0.5, # Max change in PSU per meter (example)
        # Add other parameters
    }
    inversion_thresholds = { # Check for density inversions if possible
        'DENS': -0.01 # kg/m^3 per meter (density should generally increase with depth) - Requires Density calculation
    }

    fail_count = 0
    # Group by profile and parameter, sort by depth
    grouped = df.dropna(subset=['DIS_HEADER_START_DEPTH', 'DIS_DETAIL_DATA_VALUE'])\
                .sort_values(by='DIS_HEADER_START_DEPTH')\
                .groupby(['EVENT_COLLECTOR_EVENT_ID', 'DATA_TYPE_METHOD'])

    for name, group in grouped:
        if len(group) > 1:
            param_method = name[1]
            # Extract base parameter type (e.g., 'TEMP' from 'TEMP_CTD_PRIMARY')
            base_param = param_method.split('_')[0] # Adjust if naming is different

            # Calculate differences between adjacent points
            depth_diff = group['DIS_HEADER_START_DEPTH'].diff()
            value_diff = group['DIS_DETAIL_DATA_VALUE'].diff()

            # Avoid division by zero or near-zero depth difference
            valid_diff = depth_diff.notna() & (depth_diff > 0.1) # Min depth diff in meters
            if valid_diff.any():
                gradients = value_diff[valid_diff] / depth_diff[valid_diff]

                # Check gradient threshold
                if base_param in gradient_thresholds:
                    max_grad = gradient_thresholds[base_param]
                    fail_mask_grad = abs(gradients) > max_grad
                    if fail_mask_grad.any():
                         fail_indices_grad = gradients[fail_mask_grad].index
                         # Flag the point *below* the large gradient
                         update_flags(df, fail_indices_grad, f"{test_name}_Grad_{base_param}_{name[0]}", 3) # Doubtful flag (3)
                         fail_count += len(fail_indices_grad)

                # Check inversion threshold (if applicable, e.g., for density)
                # TODO: Calculate density first if needed (using gsw)
                if base_param in inversion_thresholds:
                     min_grad = inversion_thresholds[base_param] # Typically negative for bad inversion
                     fail_mask_inv = gradients < min_grad
                     if fail_mask_inv.any():
                          fail_indices_inv = gradients[fail_mask_inv].index
                          update_flags(df, fail_indices_inv, f"{test_name}_Inv_{base_param}_{name[0]}", 4) # Bad flag (4) for inversion
                          fail_count += len(fail_indices_inv)


    print(f"{test_name}: Found {fail_count} points failing gradient/inversion checks. Needs parameter-specific thresholds & density calc.")


def test_2_10_surface_doxy_sat(df):
    """Test 2.10: Surface Dissolved Oxygen Data versus Percent Saturation."""
    test_name = "2.10_SurfaceDOXYSat"
    if gsw is None:
         print(f"{test_name}: Skipped (gsw library not available).")
         return

    surface_depth_limit = 20 # meters
    # Define acceptable saturation range (e.g., 80% to 120%)
    min_sat_perc = 80
    max_sat_perc = 120

    fail_count = 0
    # Find surface DOXY data (adjust method name if needed)
    doxy_mask = df['DATA_TYPE_METHOD'].str.contains('DOXY', case=False, na=False) & \
                (df['DIS_HEADER_START_DEPTH'] < surface_depth_limit) & \
                (df['DIS_DETAIL_DATA_VALUE'].notna())
    surface_doxy = df[doxy_mask].copy()

    if surface_doxy.empty:
        print(f"{test_name}: No surface DOXY data found.")
        return

    # Need Temp and Salinity at the same location/time/depth (approx)
    # Iterate through profiles or nearby data lookup (similar complexity to freezing point/bottle-ctd)
    for event_id, profile_df in df.groupby('EVENT_COLLECTOR_EVENT_ID'):
         profile_doxy = profile_df[profile_df.index.isin(surface_doxy.index)]
         if profile_doxy.empty:
             continue

         temps = profile_df[profile_df['DATA_TYPE_METHOD'].str.contains("TEMP", case=False, na=False)]
         sals = profile_df[profile_df['DATA_TYPE_METHOD'].str.contains("PSAL", case=False, na=False)] # Practical Salinity

         if temps.empty or sals.empty:
            continue

         for idx, doxy_row in profile_doxy.iterrows():
            depth = doxy_row['DIS_HEADER_START_DEPTH']
            doxy_val = doxy_row['DIS_DETAIL_DATA_VALUE'] # Units? Assume umol/kg for gsw
            lat = doxy_row['DIS_HEADER_SLAT']
            lon = doxy_row['DIS_HEADER_SLON']

            # Find nearest Temp and Salinity by depth
            temps['depth_diff'] = abs(temps['DIS_HEADER_START_DEPTH'] - depth)
            sals['depth_diff'] = abs(sals['DIS_HEADER_START_DEPTH'] - depth)
            closest_temp_row = temps.loc[temps['depth_diff'].idxmin()]
            closest_sal_row = sals.loc[sals['depth_diff'].idxmin()]
            temp_val = closest_temp_row['DIS_DETAIL_DATA_VALUE'] # In-situ Temp
            sal_val = closest_sal_row['DIS_DETAIL_DATA_VALUE'] # Practical Salinity

            if pd.notna(temp_val) and pd.notna(sal_val) and pd.notna(lat) and pd.notna(lon):
                try:
                    # Calculate Absolute Salinity (SA), Conservative Temperature (CT)
                    abs_sal = gsw.SA_from_SP(sal_val, 0, lon, lat) # p=0 approx for surface
                    cons_temp = gsw.CT_from_t(abs_sal, temp_val, 0) # p=0 approx

                    # Calculate DO solubility (O2sat) in umol/kg
                    doxy_sol = gsw.O2sol(abs_sal, cons_temp, 0, lon, lat)

                    if doxy_sol > 1e-9: # Avoid division by zero
                         percent_sat = (doxy_val / doxy_sol) * 100

                         if not (min_sat_perc <= percent_sat <= max_sat_perc):
                             update_flags(df, pd.Index([idx]), f"{test_name}_{event_id}", 3) # Doubtful flag (3)
                             fail_count += 1
                except Exception as e:
                     # print(f"Warning: Could not calculate DO saturation for idx {idx}: {e}")
                     pass

    print(f"{test_name}: Found {fail_count} surface DOXY points outside {min_sat_perc}-{max_sat_perc}% saturation range.")


def test_3_5_3_6_climatology(df, external_data):
    """Test 3.5/3.6: Monthly Climatology Comparison."""
    test_name = "3.5_3.6_Climatology"
    climatology = external_data.get('climatology')
    if climatology is None:
        print(f"{test_name}: Skipped (Climatology data not loaded).")
        return

    # TODO: Implement comparison against monthly climatology
    # Needs structure for climatology_monthly.csv (e.g., Month, LatBin, LonBin, DepthBin, DATA_TYPE_METHOD, Mean, StdDev)
    # Extract month from DATETIME: df['Month'] = df['DATETIME'].dt.month
    # Find matching climatology bin based on month, lat, lon, depth.
    # Compare data value against Mean +/- N * StdDev (e.g., N=3 or N=4)
    # Requires spatial/temporal indexing or lookup logic.
    print(f"{test_name}: Placeholder - Needs Implementation (lookup climatology values, compare vs Mean/StdDev).")
    pass # Placeholder


# --- Main QC Execution Function ---

def run_all_qc(df, external_data):
    """Runs all automated QC checks and updates the DataFrame."""
    if df is None:
        return df # Return None if input df is None

    print("Starting Automated QC...")
    # --- Preparation ---
    # Ensure initial flag state
    df['auto_qc_flag'] = 1 # Start with 'Good' assumption before tests
    df['auto_qc_details'] = [[] for _ in range(len(df))] # Reset details list

    # Ensure necessary columns exist and have rough types (more conversion in file_handling)
    df['DIS_HEADER_SLAT'] = pd.to_numeric(df['DIS_HEADER_SLAT'], errors='coerce')
    df['DIS_HEADER_SLON'] = pd.to_numeric(df['DIS_HEADER_SLON'], errors='coerce')
    df['DIS_HEADER_START_DEPTH'] = pd.to_numeric(df['DIS_HEADER_START_DEPTH'], errors='coerce')
    df['DIS_DETAIL_DATA_VALUE'] = pd.to_numeric(df['DIS_DETAIL_DATA_VALUE'], errors='coerce')
    # DATETIME column should be created in file_handling.parse_uploaded_file

    # --- Execute Tests ---
    # Pass df and external_data dictionary to tests that need them
    test_1_1_platform_id(df)
    test_1_2_datetime(df)
    test_1_3_location(df)
    test_1_4_on_land(df)
    test_1_5_speed(df)
    test_1_6_sounding(df)
    test_2_1_global_range(df, external_data)
    test_2_2_regional_range(df, external_data)
    test_2_4_profile_envelope(df, external_data)
    test_2_5_constant_profile(df)
    test_2_6_freezing_point(df)
    test_2_7_replicates(df)
    test_2_8_bottle_ctd(df)
    test_2_9_gradient_inversion(df)
    test_2_10_surface_doxy_sat(df)
    test_3_5_3_6_climatology(df, external_data)
    # Add calls to any other implemented tests

    # --- Finalization ---
    # Initialize the final QC code with the automated results
    # Convert auto_qc_details list to comma-separated string for display if needed, or keep as list
    df['auto_qc_details_str'] = df['auto_qc_details'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else '')
    # Set initial final flag based on auto flag
    df['DIS_DETAIL_DATA_QC_CODE'] = df['auto_qc_flag']

    print("Automated QC Finished.")
    return df