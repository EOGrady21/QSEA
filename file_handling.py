import pandas as pd
import base64
import io
import os
import datetime
from pathlib import Path

# Define expected columns precisely based on the prompt
EXPECTED_COLUMNS = [
    'DIS_DATA_NUM', 'MISSION_DESCRIPTOR', 'EVENT_COLLECTOR_EVENT_ID',
    'EVENT_COLLECTOR_STN_NAME', 'DIS_HEADER_START_DEPTH', 'DIS_HEADER_END_DEPTH',
    'DIS_HEADER_SLAT', 'DIS_HEADER_SLON', 'DIS_HEADER_SDATE', 'DIS_HEADER_STIME',
    'DIS_DETAIL_DATA_TYPE_SEQ', 'DATA_TYPE_METHOD', 'DIS_DETAIL_DATA_VALUE',
    'DIS_DETAIL_DATA_QC_CODE', 'DIS_DETAIL_DETECTION_LIMIT',
    'DIS_DETAIL_DETAIL_COLLECTOR', 'DIS_DETAIL_COLLECTOR_SAMP_ID',
    'CREATED_BY', 'CREATED_DATE', 'DATA_CENTER_CODE', 'PROCESS_FLAG',
    'BATCH_SEQ', 'DIS_SAMPLE_KEY_VALUE'
]

QC_FLAG_DEFINITIONS = {
    0: 'NoQC', 1: 'Good', 2: 'Inconsistent', 3: 'Doubtful',
    4: 'Bad', 5: 'Modified', 7: 'RequiresInvestigation', 9: 'Missing'
}
QC_FLAG_SEVERITY = { # Higher number means more severe/takes precedence
    9: 9, 4: 8, 3: 7, 7: 6, 2: 5, 5: 4, 1: 3, 0: 2 # Assign severity levels
}

EXTERNAL_DATA_PATHS = {
    'global_ranges': './qc_data/global_ranges.csv',
    'regional_ranges': './qc_data/regional_ranges.csv',
    'profile_envelopes': './qc_data/profile_envelopes.csv',
    'climatology': './qc_data/climatology_monthly.csv',
    # Add paths for other required external data
}

def parse_uploaded_file(contents, filename):
    """Parses the uploaded CSV file content."""
    if contents is None:
        return None, "Error: No file content received."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), low_memory=False) # Added low_memory=False
        # elif 'xls' in filename: # Add support for excel if needed
        #     df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, f"Error: File type not supported for '{filename}'. Please upload a CSV."

        # --- Column Validation ---
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            return None, f"Error: Missing expected columns: {', '.join(missing_cols)}"
        extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
        if extra_cols:
             print(f"Warning: Extra columns found and ignored: {', '.join(extra_cols)}")
             # Optionally drop extra columns: df = df[EXPECTED_COLUMNS]

        # --- Data Type Conversion (Crucial!) ---
        try:
            df['DIS_HEADER_SLAT'] = pd.to_numeric(df['DIS_HEADER_SLAT'], errors='coerce')
            df['DIS_HEADER_SLON'] = pd.to_numeric(df['DIS_HEADER_SLON'], errors='coerce')
            df['DIS_HEADER_START_DEPTH'] = pd.to_numeric(df['DIS_HEADER_START_DEPTH'], errors='coerce')
            df['DIS_DETAIL_DATA_VALUE'] = pd.to_numeric(df['DIS_DETAIL_DATA_VALUE'], errors='coerce')
            # Combine Date and Time and convert - Robust parsing needed
            # Handle potential variations in time format (HHMMSS, HH:MM:SS, etc.)
            # Assuming DIS_HEADER_STIME might be numeric like HHMMSS or string HH:MM:SS
            df['DIS_HEADER_STIME_STR'] = df['DIS_HEADER_STIME'].astype(str).str.zfill(6) # Pad if numeric like HHMMSS
            df['DIS_HEADER_STIME_STR'] = df['DIS_HEADER_STIME_STR'].str.replace(r'(\d{2})(\d{2})(\d{2})', r'\1:\2:\3', regex=True)
            # Combine date (YYYYMMDD) and formatted time
            df['DATETIME_STR'] = df['DIS_HEADER_SDATE'].astype(str) + ' ' + df['DIS_HEADER_STIME_STR']
            df['DATETIME'] = pd.to_datetime(df['DATETIME_STR'], format='%Y%m%d %H:%M:%S', errors='coerce')

            # Ensure QC code is integer
            df['DIS_DETAIL_DATA_QC_CODE'] = pd.to_numeric(df['DIS_DETAIL_DATA_QC_CODE'], errors='coerce').fillna(0).astype(int)

        except Exception as e:
             return None, f"Error during data type conversion: {e}. Check columns like SLAT, SLON, DEPTH, VALUE, SDATE, STIME."

        # Initialise QC columns if they dont exist (or reset them)
        df['auto_qc_flag'] = 0 # Default to NoQC initially
        df['auto_qc_details'] = [[] for _ in range(len(df))] # Use lists to store multiple failed tests

        return df, f"Successfully loaded and parsed '{filename}'."

    except Exception as e:
        print(e) # Log the full error for debugging
        return None, f"Error processing file '{filename}': {e}. Ensure it is a valid CSV with UTF-8 encoding."


def load_external_data():
    """Loads external data files needed for QC checks."""
    external_data = {}
    errors = []
    for name, path in EXTERNAL_DATA_PATHS.items():
        try:
            if Path(path).is_file():
                external_data[name] = pd.read_csv(path)
                print(f"Successfully loaded {name} from {path}")
            else:
                errors.append(f"Warning: External data file not found: {path}. Some QC checks may not run.")
                external_data[name] = None # Store None if file not found
        except Exception as e:
            errors.append(f"Error loading external data file {path}: {e}")
            external_data[name] = None
    return external_data, errors


def format_dataframe_for_output(df):
    """Ensures the DataFrame has the original columns in the correct order for output."""
    # Make sure all expected columns exist, add if missing (filled with NaN or default)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA # Or appropriate default

    # Select and reorder columns
    output_df = df[EXPECTED_COLUMNS].copy()
    return output_df

def sanitize_filename(filename):
    """Removes or replaces characters unsuitable for filenames."""
    # Remove potentially problematic characters like /, \, :, *, ?, ", <>, |
    sanitized = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
    sanitized = sanitized.replace(' ', '_') # Replace spaces with underscores
    return sanitized

def append_to_log(log_data, mission_descriptor):
    """Appends QC review actions to a mission-specific log file."""
    if not mission_descriptor:
        print("Error: Cannot log review - Mission Descriptor is missing.")
        return

    log_filename_base = sanitize_filename(str(mission_descriptor))
    log_filepath = Path(f"./{log_filename_base}_qclog.csv")
    log_entry_df = pd.DataFrame([log_data]) # Convert single entry to DataFrame

    try:
        if log_filepath.is_file():
            # Append without header
            log_entry_df.to_csv(log_filepath, mode='a', header=False, index=False, encoding='utf-8')
        else:
            # Create new file with header
            log_entry_df.to_csv(log_filepath, mode='w', header=True, index=False, encoding='utf-8')
        # print(f"Successfully logged review action to {log_filepath}") # Optional: confirmation message
    except Exception as e:
        print(f"Error writing to log file {log_filepath}: {e}")


def generate_summary_report(df, filename, sme_username):
    """Generates a text summary report of the QC process."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mission_descriptor = df['MISSION_DESCRIPTOR'].unique()
    mission_str = ', '.join(map(str, mission_descriptor)) if len(mission_descriptor) > 0 else "N/A"

    # Ensure flags are numeric for counting
    df['auto_qc_flag'] = pd.to_numeric(df['auto_qc_flag'], errors='coerce').fillna(0)
    df['DIS_DETAIL_DATA_QC_CODE'] = pd.to_numeric(df['DIS_DETAIL_DATA_QC_CODE'], errors='coerce').fillna(0)

    auto_flags_counts = df['auto_qc_flag'].value_counts().sort_index()
    final_flags_counts = df['DIS_DETAIL_DATA_QC_CODE'].value_counts().sort_index()

    auto_flags_str = "\n".join([f"  - Flag {int(flag)} ({QC_FLAG_DEFINITIONS.get(int(flag), 'Unknown')}): {count}" for flag, count in auto_flags_counts.items()])
    final_flags_str = "\n".join([f"  - Flag {int(flag)} ({QC_FLAG_DEFINITIONS.get(int(flag), 'Unknown')}): {count}" for flag, count in final_flags_counts.items()])

    # Detailed breakdown by test (requires 'auto_qc_details' to be well-populated)
    test_failure_counts = {}
    for details_list in df['auto_qc_details']:
        if isinstance(details_list, list): # Check if it's a list
             for test_name in details_list:
                 test_failure_counts[test_name] = test_failure_counts.get(test_name, 0) + 1
    test_breakdown_str = "\n".join([f"  - {test}: {count} failures" for test, count in sorted(test_failure_counts.items())])
    if not test_breakdown_str:
        test_breakdown_str = "  - No specific test failures recorded in details."


    changed_records = df[df['auto_qc_flag'] != df['DIS_DETAIL_DATA_QC_CODE']].shape[0]

    report = f"""
QC Process Summary Report
--------------------------
Input Filename: {filename}
Mission Descriptor(s): {mission_str}
SME Username: {sme_username if sme_username else 'Not Provided'}
Report Generated: {now}

Automated QC Results (Initial Flags):
{auto_flags_str}

Breakdown by Failing Test (from 'auto_qc_details'):
{test_breakdown_str}

Final QC Results (After Review):
{final_flags_str}

Manual Review Summary:
- Number of records manually changed: {changed_records}
--------------------------
"""
    return report