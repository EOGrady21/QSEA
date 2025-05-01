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

def robust_datetime_parser(date_val, time_val):
    """
    Parses date and time values from various formats into a single datetime object.

    Handles common date formats (YYYYMMDD, DD-MON-YY, DD-MON-YYYY, MM/DD/YYYY)
    and time formats (HHMM, HH:MM, HHMMSS, HH:MM:SS).

    Args:
        date_val: The date value/string.
        time_val: The time value/string.

    Returns:
        pandas.Timestamp: The parsed datetime object, or pd.NaT if parsing fails.
    """
    parsed_datetime = pd.NaT # Initialize with Not a Time

    # --- 1. Parse Date ---
    date_str = str(date_val)
    parsed_date = pd.NaT
    possible_date_formats = [
        '%Y%m%d',        # YYYYMMDD
        '%d-%b-%y',      # DD-MON-YY (e.g., 01-Jan-25) - Case-insensitive month handled by pandas
        '%d-%b-%Y',      # DD-MON-YYYY (e.g., 01-Jan-2025)
        '%m/%d/%Y',      # MM/DD/YYYY
        '%Y/%m/%d',      # YYYY/MM/DD
        '%m-%d-%Y',      # MM-DD-YYYY
        '%Y-%m-%d'       # YYYY-MM-DD (ISO-like)
    ]

    # Try specific formats first
    for fmt in possible_date_formats:
        parsed_date = pd.to_datetime(date_str, format=fmt, errors='coerce')
        if not pd.isna(parsed_date):
            break # Stop if successfully parsed

    # If specific formats failed, try letting pandas infer
    if pd.isna(parsed_date):
         # infer_datetime_format can be faster for consistent formats but less flexible
         # Using errors='coerce' without specific format lets pandas try harder
         parsed_date = pd.to_datetime(date_str, errors='coerce')

    # If date parsing failed, we can't proceed
    if pd.isna(parsed_date):
        return pd.NaT

    # --- 2. Parse Time ---
    time_str = str(time_val).strip()
    parsed_time_str = "00:00:00" # Default to midnight if time is invalid/missing

    # Normalize time string (remove colons, handle potential floats like 1400.0)
    if '.' in time_str: # Handle cases like 1400.0
        time_str = time_str.split('.')[0]

    time_str_cleaned = time_str.replace(':', '')

    try:
        if len(time_str_cleaned) == 4: # HHMM
            hour = int(time_str_cleaned[:2])
            minute = int(time_str_cleaned[2:])
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                 parsed_time_str = f"{hour:02d}:{minute:02d}:00"
        elif len(time_str_cleaned) == 6: # HHMMSS
            hour = int(time_str_cleaned[:2])
            minute = int(time_str_cleaned[2:4])
            second = int(time_str_cleaned[4:])
            if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                parsed_time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        elif ':' in time_str: # Check original string for HH:MM or HH:MM:SS
             parts = time_str.split(':')
             if len(parts) == 2: # HH:MM
                 hour = int(parts[0])
                 minute = int(parts[1])
                 if 0 <= hour <= 23 and 0 <= minute <= 59:
                     parsed_time_str = f"{hour:02d}:{minute:02d}:00"
             elif len(parts) == 3: # HH:MM:SS
                 hour = int(parts[0])
                 minute = int(parts[1])
                 second = int(parts[2])
                 if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                     parsed_time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
    except (ValueError, TypeError):
        # If any conversion fails, keep the default "00:00:00"
        pass


    # --- 3. Combine Date and Time ---
    try:
        # Format date to YYYY-MM-DD before combining
        date_part_str = parsed_date.strftime('%Y-%m-%d')
        datetime_str_to_parse = f"{date_part_str} {parsed_time_str}"
        # Final parse of the standardized combined string
        parsed_datetime = pd.to_datetime(datetime_str_to_parse, format='%Y-%m-%d %H:%M:%S', errors='raise') # Raise error here if final combo is bad
    except Exception as e:
         # print(f"Could not combine date '{date_val}' and time '{time_val}'. Error: {e}") # Optional: for debugging
         parsed_datetime = pd.NaT # Ensure NaT on failure

    return parsed_datetime

def parse_uploaded_file(contents, filename):
    """
    Parses the uploaded CSV file content, performs type conversions,
    and uses a robust parser for date and time columns.
    """
    if contents is None:
        return None, "Error: No file content received."

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
    except Exception as e:
        return None, f"Error decoding file content: {e}. Ensure the file upload format is correct."

    try:
        if 'csv' in filename.lower(): # Use lower() for case-insensitive check
            # Assume the user uploaded a CSV file
            # Try UTF-8 first, then fall back to latin1 or others if needed
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), low_memory=False)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode('latin1')), low_memory=False)
                    print("Warning: File decoded using latin1 encoding.")
                except Exception as decode_err:
                     return None, f"Error decoding CSV file '{filename}' with utf-8 or latin1: {decode_err}"

        # elif 'xls' in filename.lower(): # Example: Add support for excel
        #     try:
        #         df = pd.read_excel(io.BytesIO(decoded))
        #     except Exception as excel_err:
        #         return None, f"Error reading Excel file '{filename}': {excel_err}"
        else:
            return None, f"Error: File type not supported for '{filename}'. Please upload a CSV." # or supported type

        # --- Column Validation ---
        # Ensure case-insensitivity if column names might vary
        df.columns = df.columns.str.strip() # Remove leading/trailing whitespace from headers
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            # Provide more context about available columns for easier debugging
            available_cols = ', '.join(df.columns)
            return None, f"Error: Missing expected columns: {', '.join(missing_cols)}. Available columns: {available_cols}"

        extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
        if extra_cols:
             print(f"Warning: Extra columns found and ignored: {', '.join(extra_cols)}")
             # Optionally drop extra columns if strict adherence is needed:
             # df = df[EXPECTED_COLUMNS]

        # --- Data Type Conversion ---
        print("Starting data type conversion...") # Debugging print
        try:
            # Convert numeric columns, coercing errors to NaN
            df['DIS_HEADER_SLAT'] = pd.to_numeric(df['DIS_HEADER_SLAT'], errors='coerce')
            df['DIS_HEADER_SLON'] = pd.to_numeric(df['DIS_HEADER_SLON'], errors='coerce')
            df['DIS_HEADER_START_DEPTH'] = pd.to_numeric(df['DIS_HEADER_START_DEPTH'], errors='coerce')
            df['DIS_DETAIL_DATA_VALUE'] = pd.to_numeric(df['DIS_DETAIL_DATA_VALUE'], errors='coerce')

            # Ensure QC code is integer, handling potential NaNs from coercion
            df['DIS_DETAIL_DATA_QC_CODE'] = pd.to_numeric(df['DIS_DETAIL_DATA_QC_CODE'], errors='coerce').fillna(0).astype(int)

            # *** Use the robust datetime parser ***
            print("Applying robust datetime parser...") # Debugging print
            if 'DIS_HEADER_SDATE' in df.columns and 'DIS_HEADER_STIME' in df.columns:
                 df['DATETIME'] = df.apply(
                     lambda row: robust_datetime_parser(row['DIS_HEADER_SDATE'], row['DIS_HEADER_STIME']),
                     axis=1
                 )
            else:
                 # Handle case where date/time columns might be missing despite EXPECTED_COLUMNS check
                 # (e.g., if validation logic changes)
                 return None, "Error: Required date/time columns (DIS_HEADER_SDATE, DIS_HEADER_STIME) not found for parsing."

            # Check how many datetimes failed parsing
            failed_parses = df['DATETIME'].isna().sum()
            if failed_parses > 0:
                print(f"Warning: {failed_parses} rows failed datetime parsing and resulted in NaT.")
                # Consider logging specific rows or returning an error if too many fail

            print("Data type conversion finished.") # Debugging print

        except KeyError as ke:
             # This might happen if a column name is misspelled or missing despite initial check
             return None, f"Error during data type conversion: Missing column {ke}."
        except Exception as e:
             # Catch other potential errors during conversion/parsing
             return None, f"Error during data type conversion or parsing: {e}."

        # --- Initialize QC Columns ---
        # Ensure these are always present after processing
        df['auto_qc_flag'] = 0 # Default to NoQC initially
        # Initialize with empty lists using a safe method
        df['auto_qc_details'] = [[] for _ in range(len(df))]

        print(f"Successfully loaded and parsed '{filename}'.") # Debugging print
        return df, f"Successfully loaded and parsed '{filename}'."

    except Exception as e:
        # General catch-all for errors during file reading or initial processing
        import traceback
        print(f"Critical error processing file '{filename}': {e}")
        print(traceback.format_exc()) # Log the full traceback for debugging
        return None, f"Error processing file '{filename}': {e}. Ensure it is a valid CSV with appropriate encoding."



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