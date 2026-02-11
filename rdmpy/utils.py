# ALL UTILITY FUNCTIONS

# IMPORTS
import json
import pickle
import sys
import os
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

from demo.data.schedule import schedule_data
from demo.data.reference import reference_files
from demo.data.incidents import incident_files

# Add the parent directory to the Python path to access the input module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# FOR SCHEDULE PROCESSING

def load_schedule_data(st_code, schedule_data, reference_files):
    """
    Load all necessary data for schedule processing.
    
    Returns:
        tuple: (train_count (deprecated, returns None), tiploc, schedule_data_loaded, stanox_ref, tiploc_to_stanox)
    """
    # Load the all DFT categories stations reference file (JSON format)
    with open(reference_files["all dft categories"], 'r') as f:
        reference_data = json.load(f)
    # Convert to DataFrame
    stanox_ref_df = pd.DataFrame(reference_data)
    
    # Convert DataFrame to list of dictionaries for backward compatibility
    stanox_ref = stanox_ref_df.to_dict('records')

    # Find the TIPLOC for this STANOX
    tiploc = None
    
    # First try to find using DataFrame operations (more efficient)
    matching_rows = stanox_ref_df[stanox_ref_df['stanox'] == str(st_code)]
    if not matching_rows.empty:
        tiploc = matching_rows.iloc[0]['tiploc']
    else:
        # Fallback: try with different type conversions
        try:
            # Try as integer
            matching_rows = stanox_ref_df[stanox_ref_df['stanox'] == int(st_code)]
            if not matching_rows.empty:
                tiploc = matching_rows.iloc[0]['tiploc']
        except (ValueError, TypeError):
            pass
    
    if tiploc is None:
        print(f"Warning: STANOX {st_code} not found in reference data")
        return 0, None, None, stanox_ref, {}

    # Load the schedule file (Pickle format) - use pandas for consistency
    schedule_data_loaded = pd.read_pickle(schedule_data["schedule"])
    
    # Convert to list of dicts if it's a DataFrame
    if isinstance(schedule_data_loaded, pd.DataFrame):
        schedule_data_loaded = schedule_data_loaded.to_dict('records')

    # Create tiploc_to_stanox mapping from DataFrame
    tiploc_to_stanox = dict(zip(stanox_ref_df['tiploc'], stanox_ref_df['stanox']))

    # Return without train_count (deprecated parameter)
    return None, tiploc, schedule_data_loaded, stanox_ref, tiploc_to_stanox


# This code processes schedule files for one station code (st_code).
# st_code, and therefore tiploc, needs to be defined

# Add the parent directory to the Python path to access the input module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# helper 1 for get_english_day_types_from_schedule

def get_day_code_mapping():
    """
    Create a mapping for day codes used throughout the application.
    
    Returns:
        dict: Mapping from day indices to day codes (0=Monday, 1=Tuesday, ..., 6=Sunday)
    """
    return {
        0: "MO",  # Monday
        1: "TU",  # Tuesday  
        2: "WE",  # Wednesday
        3: "TH",  # Thursday
        4: "FR",  # Friday
        5: "SA",  # Saturday
        6: "SU"   # Sunday
    }

# helper 2 for get_english_day_types_from_schedule

def extract_schedule_days_runs(schedule_entry):
    """
    Extract schedule_days_runs from a schedule entry.
    
    Args:
        schedule_entry: Schedule entry dictionary
        
    Returns:
        str: Binary string representing days the schedule runs, or None if not found
    """
    try:
        # schedule_days_runs is at the top level (flattened structure)
        return schedule_entry.get('schedule_days_runs')
    except (KeyError, TypeError, AttributeError):
        return None


def get_english_day_types_from_schedule(schedule_entry):
    """
    Convert schedule_days_runs to list of ENGLISH_DAY_TYPE values.
    
    Args:
        schedule_entry: Schedule entry dictionary
        
    Returns:
        list: List of ENGLISH_DAY_TYPE values that this schedule runs on
    """
    schedule_days_runs = extract_schedule_days_runs(schedule_entry)
    if not schedule_days_runs:
        return []
    
    day_type_mapping = get_day_code_mapping()
    english_day_types = []
    
    # Convert binary string to list of day types
    for i, bit in enumerate(schedule_days_runs):
        if bit == '1' and i < len(day_type_mapping):
            english_day_types.append(day_type_mapping[i])
    
    return english_day_types


# =======================================================================================
# DATA VALIDATION AND CLEANING FUNCTIONS
# =======================================================================================

def is_valid_schedule_entry(schedule_entry):
    """
    Validate that a schedule entry has required structure.
    
    Args:
        schedule_entry: Schedule entry dictionary
        
    Returns:
        bool: True if entry has required fields
    """
    try:
        if not isinstance(schedule_entry, dict):
            return False
        schedule_segment = schedule_entry.get('schedule_segment')
        if not schedule_segment:
            return False
        schedule_locations = schedule_segment.get('schedule_location')
        if not isinstance(schedule_locations, list):
            return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


def validate_schedule_locations(schedule_locations):
    """
    Validate that schedule_locations is iterable and contains valid entries.
    
    Args:
        schedule_locations: List of location dictionaries
        
    Returns:
        bool: True if valid
    """
    try:
        return isinstance(schedule_locations, list) and all(isinstance(loc, dict) for loc in schedule_locations)
    except (TypeError, AttributeError):
        return False


def is_valid_location_entry(location):
    """
    Validate a single location entry has get method (dict-like).
    
    Args:
        location: Location entry to validate
        
    Returns:
        bool: True if location is dict-like
    """
    return isinstance(location, dict) and hasattr(location, 'get')


def has_time_information(location):
    """
    Check if location has either departure or arrival time.
    
    Args:
        location: Location dictionary
        
    Returns:
        bool: True if either departure or arrival exists
    """
    return bool(location.get('departure') or location.get('arrival'))


def extract_location_time(location):
    """
    Extract departure or arrival time from location, preferring departure.
    
    Args:
        location: Location dictionary
        
    Returns:
        str or None: Time in format 'HHMM' or None if unavailable
    """
    if location.get('departure'):
        return location['departure'][:4]
    elif location.get('arrival'):
        return location['arrival'][:4]
    return None


def get_train_service_code(schedule_entry):
    """
    Extract CIF_train_service_code from schedule entry.
    
    Args:
        schedule_entry: Schedule entry dictionary
        
    Returns:
        str or None: Train service code or None if not found
    """
    try:
        return schedule_entry["schedule_segment"]["CIF_train_service_code"]
    except (KeyError, TypeError):
        return None


def clean_dataframe_types(df, columns_to_convert):
    """
    Standardize data types in a DataFrame for consistent merging.
    
    Args:
        df: DataFrame to clean
        columns_to_convert: List of (column_name, target_type) tuples
        
    Returns:
        DataFrame: DataFrame with converted types
    """
    df_copy = df.copy()
    for col, dtype in columns_to_convert:
        if col in df_copy.columns:
            try:
                df_copy[col] = df_copy[col].astype(dtype)
            except (ValueError, TypeError):
                pass
    return df_copy


def filter_valid_delay_entries(delay_df):
    """
    Filter delay entries to keep only those with valid datetime strings.
    
    Args:
        delay_df: DataFrame of delay entries
        
    Returns:
        DataFrame: Filtered DataFrame
    """
    mask = (delay_df['PLANNED_ORIGIN_GBTT_DATETIME'].astype(str).str.len() >= 5) & \
           (delay_df['PLANNED_DEST_GBTT_DATETIME'].astype(str).str.len() >= 5) & \
           (delay_df['EVENT_DATETIME'].astype(str).str.len() >= 5)
    return delay_df[mask].copy()



# =======================================================================================
# MAIN FUNCTION: Process schedule for a specific STANOX code
# =======================================================================================

def process_schedule(st_code, schedule_data=None, reference_files=None, 
                    train_count=None, tiploc=None, schedule_data_loaded=None, 
                    stanox_ref=None, tiploc_to_stanox=None):
    """
    Generate a schedule timeline for all trains that match the specified STANOX code.
    OPTIMIZED VERSION - Accepts pre-loaded data to avoid reloading from files.

    Args:
        st_code (str): STANOX code to process.
        schedule_data (dict, optional): Dictionary containing schedule data file paths.
        reference_files (dict, optional): Dictionary containing reference file paths.
        train_count (int, optional): Expected simple count of number of trains (from pre-loaded data).
        tiploc (str, optional): TIPLOC code corresponding to st_code.
        schedule_data_loaded (list, optional): Pre-loaded schedule data.
        stanox_ref (dict, optional): Pre-loaded STANOX reference data.
        tiploc_to_stanox (dict, optional): Pre-loaded TIPLOC to STANOX mapping.

    Returns:
        list: Sorted schedule timeline.
    """
    # Load data if not pre-loaded
    if schedule_data_loaded is not None and stanox_ref is not None and tiploc_to_stanox is not None:
        print("Using pre-loaded data (much faster!)")
        
        if train_count is None or tiploc is None:
            train_count, tiploc = _extract_tiploc_and_count(st_code, schedule_data_loaded, stanox_ref)
    else:
        print("Loading data from files (this may take a while)...")
        train_count, tiploc, schedule_data_loaded, stanox_ref, tiploc_to_stanox = load_schedule_data(
            st_code, schedule_data, reference_files
        )
    
    if tiploc is None:
        return []
    
    # Convert stanox_ref to dict if needed
    if isinstance(stanox_ref, list):
        stanox_ref = {str(entry.get("stanox")): entry for entry in stanox_ref if "stanox" in entry}

    # Convert data to list format if needed
    if hasattr(schedule_data_loaded, 'iterrows'):
        schedule_data_list = schedule_data_loaded.to_dict('records')
    else:
        schedule_data_list = schedule_data_loaded

    processed_schedule = _process_schedule_entries(
        schedule_data_list, tiploc, tiploc_to_stanox, stanox_ref, st_code
    )
    
    # Sort by planned calls
    processed_schedule.sort(key=lambda x: int(x["PLANNED_CALLS"]))
    
    # Debug output
    _print_schedule_debug_info(processed_schedule, tiploc, tiploc_to_stanox, st_code)

    return processed_schedule


def _extract_tiploc_and_count(st_code, schedule_data_loaded, stanox_ref):
    """
    Extract TIPLOC and count matching trains from pre-loaded data.
    
    Args:
        st_code: STANOX code
        schedule_data_loaded: Pre-loaded schedule data
        stanox_ref: Reference data
        
    Returns:
        tuple: (train_count, tiploc)
    """
    tiploc = None
    train_count = 0
    
    # Convert to list if DataFrame
    if hasattr(stanox_ref, 'to_dict'):
        stanox_ref_list = stanox_ref.to_dict('records')
    else:
        stanox_ref_list = stanox_ref
    
    # Find TIPLOC
    for entry in stanox_ref_list:
        if str(entry.get("stanox")) == str(st_code):
            tiploc = entry.get("tiploc") or entry.get("tiploc_code")
            break
    
    if tiploc:
        train_count = _count_matching_trains(schedule_data_loaded, tiploc)
    
    return train_count, tiploc


def _count_matching_trains(schedule_data_loaded, tiploc):
    """
    Count trains in schedule that pass through target TIPLOC.
    
    Args:
        schedule_data_loaded: Schedule data (list or DataFrame)
        tiploc: Target TIPLOC code
        
    Returns:
        int: Count of matching trains
    """
    train_count = 0
    
    if hasattr(schedule_data_loaded, 'iterrows'):
        for idx, row in schedule_data_loaded.iterrows():
            s = row.to_dict() if hasattr(row, 'to_dict') else row
            if _train_passes_through_tiploc(s, tiploc):
                train_count += 1
    else:
        for s in schedule_data_loaded:
            if _train_passes_through_tiploc(s, tiploc):
                train_count += 1
    
    return train_count


def _train_passes_through_tiploc(schedule_entry, target_tiploc):
    """
    Check if train schedule passes through target TIPLOC.
    
    Args:
        schedule_entry: Schedule entry dictionary
        target_tiploc: Target TIPLOC code
        
    Returns:
        bool: True if train stops at TIPLOC
    """
    try:
        json_schedule = schedule_entry.get("JsonScheduleV1", {})
        schedule_segment = json_schedule.get("schedule_segment", {})
        sched_loc = schedule_segment.get("schedule_location", [])
        tiploc_codes = {loc.get("tiploc_code") for loc in sched_loc}
        return target_tiploc in tiploc_codes
    except (KeyError, TypeError, AttributeError):
        return False


def _process_schedule_entries(schedule_data_list, tiploc, tiploc_to_stanox, stanox_ref, st_code):
    """
    Process all schedule entries to find matching trains.
    
    Args:
        schedule_data_list: List of schedule entries
        tiploc: Target TIPLOC
        tiploc_to_stanox: Mapping dictionary
        stanox_ref: Reference data
        st_code: STANOX code
        
    Returns:
        list: Processed schedule records
    """
    processed_schedule = []
    trains_processed = 0

    print(f"Processing {len(schedule_data_list)} schedule entries for TIPLOC: {tiploc}")

    for idx, s in enumerate(schedule_data_list):
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(schedule_data_list)} entries, found {trains_processed} matching trains")
        
        # Validate entry structure
        if not is_valid_schedule_entry(s):
            continue
        
        schedule_locations = s['schedule_segment']['schedule_location']
        
        # Validate locations
        if not validate_schedule_locations(schedule_locations):
            continue
        
        # Extract locations (early exit optimization)
        relevant_location, origin_location, destination_location = _extract_relevant_locations(
            schedule_locations, tiploc
        )
        
        if relevant_location is None:
            continue
        
        # Extract time
        s_time = extract_location_time(relevant_location)
        if s_time is None:
            continue
        
        # Extract service code
        train_service_code = get_train_service_code(s)
        if train_service_code is None:
            continue
        
        # Get day types
        schedule_day_types = get_english_day_types_from_schedule(s)
        
        # Build record
        train_record = build_train_record(
            train_service_code, origin_location, destination_location,
            relevant_location, s_time, schedule_day_types, tiploc,
            tiploc_to_stanox, stanox_ref, st_code
        )
        
        processed_schedule.append(train_record)
        trains_processed += 1

    print(f"Completed processing. Found {len(processed_schedule)} trains (including all weekly instances)")
    
    return processed_schedule


def _extract_relevant_locations(schedule_locations, target_tiploc):
    """
    Extract all relevant location types for target TIPLOC from schedule.
    Uses early exit optimization.
    
    Args:
        schedule_locations: List of location dictionaries
        target_tiploc: Target TIPLOC code
        
    Returns:
        tuple: (relevant_location, origin_location, destination_location)
    """
    relevant_location = None
    origin_location = None
    destination_location = None
    
    for loc in schedule_locations:
        if not is_valid_location_entry(loc):
            continue
        
        loc_tiploc = loc.get('tiploc_code')
        
        if loc_tiploc == target_tiploc:
            relevant_location = loc
            
            # Check if this is also origin or destination
            loc_type = loc.get('location_type')
            if loc_type in ['LO', 'L0']:
                origin_location = loc
            elif loc_type == 'LT':
                destination_location = loc
        
        # Early exit if found all needed locations
        if relevant_location and origin_location and destination_location:
            break
    
    return relevant_location, origin_location, destination_location


def _print_schedule_debug_info(processed_schedule, tiploc, tiploc_to_stanox, st_code):
    """
    Print debug information about processed schedule.
    
    Args:
        processed_schedule: List of processed train records
        tiploc: Target TIPLOC
        tiploc_to_stanox: Mapping dictionary
        st_code: STANOX code
    """
    role_counts = {}
    for train in processed_schedule:
        role = train.get("STATION_ROLE", "Unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print(f"Station roles for {tiploc} (STANOX {tiploc_to_stanox.get(tiploc, 'Unknown')}):")
    for role, count in role_counts.items():
        print(f"  {role}: {count} trains")


def extract_day_of_week_from_delay(delay_entry):
    """
    Extract the day of the week from PLANNED_ORIGIN_WTT_DATETIME in delay data.
    
    Args:
        delay_entry: Delay entry dictionary
        
    Returns:
        str: Day of week code (MO, TU, WE, TH, FR, SA, SU) or None if parsing fails
    """
    try:
        # Get the datetime string from delay data
        datetime_str = delay_entry.get('PLANNED_ORIGIN_WTT_DATETIME')
        if not datetime_str:
            return None
            
        # Parse the datetime string (format: "01-APR-2024 08:51")
        dt = datetime.strptime(datetime_str, "%d-%b-%Y %H:%M")
        
        # Convert to day of week (0=Monday, 1=Tuesday, ..., 6=Sunday)
        weekday = dt.weekday()
        
        # Use the shared day code mapping
        day_mapping = get_day_code_mapping()
        return day_mapping.get(weekday)
        
    except (ValueError, AttributeError, KeyError):
        return None


def schedule_runs_on_day(schedule_entry, target_day):
    """
    Check if a schedule entry runs on a specific day of the week.
    
    Args:
        schedule_entry: Schedule entry dictionary (from processed schedule)
        target_day: Day code (MO, TU, WE, TH, FR, SA, SU)
        
    Returns:
        bool: True if the schedule runs on the target day
    """
    # Use ENGLISH_DAY_TYPE which now contains the list of day codes
    schedule_day_types = schedule_entry.get('ENGLISH_DAY_TYPE', [])
    return target_day in schedule_day_types


# =======================================================================================
# DATA TRANSFORMATION AND EXTRACTION FUNCTIONS
# =======================================================================================

def find_location_by_tiploc(schedule_locations, target_tiploc):
    """
    Find first location matching target TIPLOC.
    
    Args:
        schedule_locations: List of location dictionaries
        target_tiploc: TIPLOC code to match
        
    Returns:
        dict or None: Matching location or None
    """
    for loc in schedule_locations:
        if not isinstance(loc, dict):
            continue
        if loc.get('tiploc_code') == target_tiploc:
            return loc
    return None


def find_origin_location(schedule_locations, target_tiploc):
    """
    Find origin location (LO or L0) at target TIPLOC.
    
    Args:
        schedule_locations: List of location dictionaries
        target_tiploc: TIPLOC code to match
        
    Returns:
        dict or None: Origin location or None
    """
    for loc in schedule_locations:
        if not isinstance(loc, dict):
            continue
        if loc.get('tiploc_code') == target_tiploc and loc.get('location_type') in ['LO', 'L0']:
            return loc
    return None


def find_destination_location(schedule_locations, target_tiploc):
    """
    Find destination location (LT) at target TIPLOC.
    
    Args:
        schedule_locations: List of location dictionaries
        target_tiploc: TIPLOC code to match
        
    Returns:
        dict or None: Destination location or None
    """
    for loc in schedule_locations:
        if not isinstance(loc, dict):
            continue
        if loc.get('tiploc_code') == target_tiploc and loc.get('location_type') == 'LT':
            return loc
    return None


def determine_station_role(relevant_location, origin_location, destination_location, tiploc):
    """
    Determine the role of station in train's journey.
    
    Args:
        relevant_location: Matched location at target TIPLOC
        origin_location: Origin location (if exists)
        destination_location: Destination location (if exists)
        tiploc: Target TIPLOC code
        
    Returns:
        str: "Origin", "Destination", "Intermediate", or "Unknown"
    """
    if not relevant_location:
        return "Unknown"
    
    if origin_location and origin_location.get("tiploc_code") == tiploc:
        return "Origin"
    elif destination_location and destination_location.get("tiploc_code") == tiploc:
        return "Destination"
    else:
        return "Intermediate"


def build_train_record(train_service_code, origin_location, destination_location, 
                       relevant_location, s_time, schedule_day_types, tiploc, 
                       tiploc_to_stanox, stanox_ref, st_code):
    """
    Build a single train record for processed schedule.
    
    Args:
        train_service_code: CIF service code
        origin_location: Origin location dict
        destination_location: Destination location dict
        relevant_location: Location at target station
        s_time: Time in HHMM format
        schedule_day_types: List of day codes
        tiploc: Target TIPLOC
        tiploc_to_stanox: TIPLOC to STANOX mapping
        stanox_ref: STANOX reference data (dict)
        st_code: Target STANOX code
        
    Returns:
        dict: Train record with all fields populated
    """
    origin_stanox = tiploc_to_stanox.get(origin_location.get("tiploc_code"), "Unknown") if origin_location else "Unknown"
    dest_stanox = tiploc_to_stanox.get(destination_location.get("tiploc_code"), "Unknown") if destination_location else "Unknown"
    station_role = determine_station_role(relevant_location, origin_location, destination_location, tiploc)
    
    return {
        "TRAIN_SERVICE_CODE": train_service_code,
        "PLANNED_ORIGIN_LOCATION_CODE": origin_stanox,
        "PLANNED_ORIGIN_GBTT_DATETIME": origin_location.get("departure", "Unknown") if origin_location else "Unknown",
        "PLANNED_DEST_LOCATION_CODE": dest_stanox,
        "PLANNED_DEST_GBTT_DATETIME": destination_location.get("arrival", "Unknown") if destination_location else "Unknown",
        "PLANNED_CALLS": s_time,
        "ACTUAL_CALLS": s_time,
        "PFPI_MINUTES": 0.0,
        "INCIDENT_REASON": None,
        "INCIDENT_NUMBER": None,
        "EVENT_TYPE": None,
        "SECTION_CODE": None,
        "DELAY_DAY": None,
        "EVENT_DATETIME": None,
        "INCIDENT_START_DATETIME": None,
        "ENGLISH_DAY_TYPE": schedule_day_types,
        "STATION_ROLE": station_role,
        "DFT_CATEGORY": stanox_ref.get(st_code, {}).get("dft_category", None),
        "PLATFORM_COUNT": stanox_ref.get(st_code, {}).get("numeric_platform_count", None)
    }


def extract_time_components_from_delays(delays_df):
    """
    Extract time components from delay DataFrame for matching.
    
    Args:
        delays_df: DataFrame of delays
        
    Returns:
        DataFrame: DataFrame with added origin_time, dest_time, event_time columns
    """
    delays_clean = delays_df.copy()
    delays_clean['origin_time'] = delays_clean['PLANNED_ORIGIN_GBTT_DATETIME'].astype(str).str[-5:].str.replace(":", "")
    delays_clean['dest_time'] = delays_clean['PLANNED_DEST_GBTT_DATETIME'].astype(str).str[-5:].str.replace(":", "")
    delays_clean['event_time'] = delays_clean['EVENT_DATETIME'].astype(str).str[-5:].str.replace(":", "")
    return delays_clean


def expand_schedule_by_days(schedule_df):
    """
    Expand schedule entries for multi-day schedules (one row per day).
    
    Args:
        schedule_df: Schedule DataFrame
        
    Returns:
        DataFrame: Expanded DataFrame with current_day column
    """
    schedule_expanded = []
    for _, sched in schedule_df.iterrows():
        sched_dict = sched.to_dict()
        english_day_types = sched_dict.get('ENGLISH_DAY_TYPE', [])
        if not english_day_types:
            english_day_types = ['MO']
        
        for day in english_day_types:
            sched_copy = sched_dict.copy()
            sched_copy['current_day'] = day
            schedule_expanded.append(sched_copy)
    
    return pd.DataFrame(schedule_expanded)

# PROCESS DELAYS

# This code processes delay files for one station code (st_code).
# st_code needs to be defined

def process_delays(incident_files, st_code, output_dir):
    """
    Processes delay files by converting them to vertical JSON, removing irrelevant columns, and filtering rows.

    Args:
        incident_files (dict): Dictionary with period names as keys and file paths as values.
        output_dir (str): Directory to save the converted JSON files.
        st_code (str): The station code to filter delays.

    Returns:
        dict: Dictionary with period names as keys and processed DataFrames as values.
    """
    columns_to_remove = [
        "ATTRIBUTION_STATUS", "INCIDENT_EQUIPMENT", "APPLICABLE_TIMETABLE_FLAG", "TRACTION_TYPE",
        "TRAILING_LOAD"
    ]
    
    processed_delays = {}
    
    # Ensure st_code is an integer for proper comparison with STANOX columns
    st_code_int = int(st_code)
    
    for period_name, file_path in incident_files.items():
        # Load the delay data
        delay_df = pd.read_csv(file_path)
        
        # Filter for rows where START_STANOX or END_STANOX equals st_code
        delay_df = delay_df[
            (delay_df["START_STANOX"] == st_code_int) | (delay_df["END_STANOX"] == st_code_int)
        ]
        
        # Convert the DataFrame to a vertical JSON file (one JSON object per line)
        # Use the period name for the JSON file instead of the original filename
        json_file_name = os.path.join(output_dir, f"{period_name}.json")
        delay_df.to_json(json_file_name, orient="records", lines=True)
        
        # Remove irrelevant columns
        delay_df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        
        # Keep all EVENT_TYPE values including "C" (cancellations) for analysis
        
        processed_delays[period_name] = delay_df
    
    return processed_delays


# =======================================================================================
# DELAY MATCHING AND PROCESSING FUNCTIONS
# =======================================================================================

def extract_day_from_each_delay(delays_df):
    """
    Extract day of week for each delay entry.
    
    Args:
        delays_df: DataFrame of delays
        
    Returns:
        list: List of day codes (MO, TU, etc.) aligned with rows
    """
    delay_days = []
    for _, delay in delays_df.iterrows():
        day = extract_day_of_week_from_delay(delay.to_dict())
        delay_days.append(day)
    return delay_days


def add_delay_day_column(delays_df):
    """
    Add delay_day column to delays DataFrame, filtering for valid entries.
    
    Args:
        delays_df: DataFrame of delays
        
    Returns:
        DataFrame: DataFrame with delay_day column (filtered to valid entries)
    """
    delays_df = delays_df.copy()
    delay_days = extract_day_from_each_delay(delays_df)
    delays_df['delay_day'] = delay_days
    return delays_df[delays_df['delay_day'].notna()]


def find_matched_delays_info(matched_results_df):
    """
    Extract matched delay information for comparison with unmatched.
    
    Args:
        matched_results_df: Filtered DataFrame of matched results
        
    Returns:
        set: Set of tuples (TRAIN_SERVICE_CODE, DELAY_DAY, PFPI_MINUTES)
    """
    if matched_results_df.empty:
        return set()
    
    matched_info = matched_results_df[['TRAIN_SERVICE_CODE', 'DELAY_DAY', 'PFPI_MINUTES']].copy()
    return set(zip(matched_info['TRAIN_SERVICE_CODE'], 
                   matched_info['DELAY_DAY'], 
                   matched_info['PFPI_MINUTES']))


def identify_unmatched_delays(delays_df, matched_delay_info):
    """
    Identify delays that were not matched with schedule entries.
    
    Args:
        delays_df: DataFrame of all delays
        matched_delay_info: Set of matched delay tuples
        
    Returns:
        DataFrame: DataFrame of unmatched delays
    """
    delays_df = delays_df.copy()
    delays_df['match_tuple'] = list(zip(
        delays_df['TRAIN_SERVICE_CODE'], 
        delays_df['delay_day'], 
        delays_df['PFPI_MINUTES']
    ))
    
    unmatched_mask = ~delays_df['match_tuple'].isin(matched_delay_info)
    return delays_df[unmatched_mask].copy()


def determine_planned_call_time(row, st_code):
    """
    Determine planned call time for unmatched delay based on station role.
    
    Args:
        row: Delay row (Series or dict-like)
        st_code: Station code being analyzed
        
    Returns:
        str: Time in HHMM format
    """
    origin_match = (st_code is not None) and (str(row.get('PLANNED_ORIGIN_LOCATION_CODE')) == str(st_code))
    dest_match = (st_code is not None) and (str(row.get('PLANNED_DEST_LOCATION_CODE')) == str(st_code))
    
    if origin_match:
        return row.get('origin_time', row.get('dest_time', '0000'))
    elif dest_match:
        return row.get('dest_time', row.get('origin_time', '0000'))
    else:
        return row.get('dest_time', '0000')


def build_unmatched_entry(delay_row, st_code):
    """
    Build a record for unmatched delay entry.
    
    Args:
        delay_row: Delay Series/dict
        st_code: Station code being analyzed
        
    Returns:
        dict: Complete delay entry record
    """
    planned_calls = determine_planned_call_time(delay_row, st_code)
    
    return {
        "TRAIN_SERVICE_CODE": delay_row['TRAIN_SERVICE_CODE'],
        "PLANNED_ORIGIN_LOCATION_CODE": delay_row['PLANNED_ORIGIN_LOCATION_CODE'],
        "PLANNED_ORIGIN_GBTT_DATETIME": delay_row.get('origin_time', 'Unknown'),
        "PLANNED_DEST_LOCATION_CODE": delay_row['PLANNED_DEST_LOCATION_CODE'],
        "PLANNED_DEST_GBTT_DATETIME": delay_row.get('dest_time', 'Unknown'),
        "PLANNED_CALLS": planned_calls,
        "ACTUAL_CALLS": delay_row.get('event_time', 'Unknown'),
        "PFPI_MINUTES": float(delay_row['PFPI_MINUTES']),
        "INCIDENT_REASON": delay_row.get('INCIDENT_REASON'),
        "INCIDENT_NUMBER": delay_row.get('INCIDENT_NUMBER'),
        "EVENT_TYPE": delay_row.get('EVENT_TYPE'),
        "SECTION_CODE": delay_row.get('SECTION_CODE'),
        "DELAY_DAY": delay_row.get('delay_day'),
        "EVENT_DATETIME": delay_row.get('EVENT_DATETIME'),
        "INCIDENT_START_DATETIME": delay_row.get('INCIDENT_START_DATETIME'),
        "ENGLISH_DAY_TYPE": [delay_row.get('delay_day')] if delay_row.get('delay_day') else [],
        "DFT_CATEGORY": None,
        "PLATFORM_COUNT": None,
        "STATION_ROLE": None,
        "START_STANOX": delay_row.get('START_STANOX'),
        "END_STANOX": delay_row.get('END_STANOX')
    }


def apply_delays_to_matches(result_df, matched_mask):
    """
    Update actual times and delay info for matched entries.
    
    Args:
        result_df: Result DataFrame (modified in-place)
        matched_mask: Boolean mask of matched entries
        
    Returns:
        DataFrame: Updated DataFrame
    """
    result_df = result_df.copy()
    result_df.loc[matched_mask, 'ACTUAL_CALLS'] = result_df.loc[matched_mask, 'event_time']
    result_df.loc[matched_mask, 'DELAY_DAY'] = result_df.loc[matched_mask, 'delay_day']
    
    # For non-matched, keep planned = actual
    non_matched_mask = ~matched_mask
    result_df.loc[non_matched_mask, 'ACTUAL_CALLS'] = result_df.loc[non_matched_mask, 'PLANNED_CALLS']
    result_df.loc[non_matched_mask, 'PFPI_MINUTES'] = 0
    
    return result_df


def filter_result_columns(combined_df):
    """
    Filter result DataFrame to required columns only.
    
    Args:
        combined_df: Combined results DataFrame
        
    Returns:
        DataFrame: Filtered DataFrame with core columns
    """
    required_columns = [
        'TRAIN_SERVICE_CODE', 'PLANNED_ORIGIN_LOCATION_CODE', 'PLANNED_ORIGIN_GBTT_DATETIME',
        'PLANNED_DEST_LOCATION_CODE', 'PLANNED_DEST_GBTT_DATETIME', 'PLANNED_CALLS',
        'ACTUAL_CALLS', 'PFPI_MINUTES', 'INCIDENT_REASON', 'INCIDENT_NUMBER',
        'EVENT_TYPE', 'SECTION_CODE', 'DELAY_DAY', 'EVENT_DATETIME', 'INCIDENT_START_DATETIME',
        'ENGLISH_DAY_TYPE', 'DFT_CATEGORY', 'PLATFORM_COUNT', 'STATION_ROLE', 'START_STANOX', 'END_STANOX'
    ]
    
    available_cols = [col for col in required_columns if col in combined_df.columns]
    return combined_df[available_cols].copy()


# =======================================================================================
# MATCHING PROCESSED DELAYS AND SCHEDULE
# =======================================================================================
# This code wants to match schedule data with delay data, adjusting the timeline based on delays.
# It processes schedule data, applies delays, and generates an adjusted timeline for further analysis.


def adjust_schedule_timeline(processed_schedule, processed_delays, st_code=None):
    """
    Adjust the schedule timeline based on delays and generate an updated timeline.
    PANDAS OPTIMIZED VERSION: Uses pandas DataFrames for ultra-fast matching operations.

    Args:
        processed_schedule (list): List of processed schedule dictionaries.
        processed_delays (list): List of delay records from all days.
        st_code (str, optional): The station code being analyzed to determine correct planned call times.

    Returns:
        list: Adjusted schedule timeline sorted by actual calls.
    """
    
    print(f"Using pandas for delay matching: {len(processed_schedule)} schedule entries and {len(processed_delays)} delay entries...")
    
    if not processed_schedule or not processed_delays:
        print("No schedule or delay data to process")
        return processed_schedule if processed_schedule else []
    
    # Convert to DataFrames
    schedule_df = pd.DataFrame(processed_schedule)
    delays_df = pd.DataFrame(processed_delays)
    
    print(f"Created DataFrames: schedule ({len(schedule_df)} rows), delays ({len(delays_df)} rows)")
    
    # Standardize types
    print("Standardizing data types...")
    schedule_df = clean_dataframe_types(schedule_df, [
        ('TRAIN_SERVICE_CODE', str),
        ('PLANNED_ORIGIN_LOCATION_CODE', str),
        ('PLANNED_DEST_LOCATION_CODE', str),
        ('PLANNED_ORIGIN_GBTT_DATETIME', str),
        ('PLANNED_DEST_GBTT_DATETIME', str)
    ])
    
    delays_df = clean_dataframe_types(delays_df, [
        ('TRAIN_SERVICE_CODE', str),
        ('PLANNED_ORIGIN_LOCATION_CODE', str),
        ('PLANNED_DEST_LOCATION_CODE', str)
    ])
    
    # Clean and preprocess delays
    print("Cleaning and preprocessing delay data...")
    delays_clean = filter_valid_delay_entries(delays_df)
    print(f"Filtered to {len(delays_clean)} delays with valid datetime strings")
    
    if delays_clean.empty:
        print("No valid delays found after filtering")
        return processed_schedule
    
    # Extract time components
    delays_clean = extract_time_components_from_delays(delays_clean)
    
    # Add delay days
    print("Extracting delay days...")
    delays_clean = add_delay_day_column(delays_clean)
    print(f"Filtered to {len(delays_clean)} delays with valid day information")
    
    if delays_clean.empty:
        print("No delays with valid day information")
        return processed_schedule
    
    # Expand schedules by day
    print("Expanding multi-day schedules...")
    schedule_expanded_df = expand_schedule_by_days(schedule_df)
    print(f"Expanded to {len(schedule_expanded_df)} schedule entries (including multi-day)")
    
    # Perform matching
    print("Performing origin-based matching...")
    origin_matches = _match_by_origin(schedule_expanded_df, delays_clean)
    
    print("Performing destination-based matching...")
    dest_matches = _match_by_destination(schedule_expanded_df, delays_clean)
    
    # Combine matches
    print("Combining matches...")
    combined_matches = _combine_match_results(origin_matches, dest_matches)
    
    # Apply delays
    print("Applying delays to matched entries...")
    matched_mask = combined_matches['PFPI_MINUTES'].notna()
    combined_matches = apply_delays_to_matches(combined_matches, matched_mask)
    
    # Extract results
    result_df = filter_result_columns(combined_matches)
    print(f"Matched {matched_mask.sum()} schedule entries with delays")
    
    # Add unmatched delays
    print("Adding unmatched delays (optimized)...")
    matched_delay_info = find_matched_delays_info(combined_matches[matched_mask])
    unmatched_delays = identify_unmatched_delays(delays_clean, matched_delay_info)
    
    print(f"Found {len(unmatched_delays)} unmatched delays to add")
    
    unmatched_entries = _build_unmatched_entries_list(unmatched_delays, st_code)
    print(f"Added {len(unmatched_entries)} unmatched delays as new entries")
    
    # Combine results
    final_result = result_df.to_dict('records') + unmatched_entries
    
    # Sort by actual calls
    final_result.sort(key=lambda x: int(x["ACTUAL_CALLS"]) if str(x["ACTUAL_CALLS"]).isdigit() else 0)
    
    print(f"Timeline adjustment complete: {len(final_result)} total entries")
    return final_result


def _match_by_origin(schedule_expanded_df, delays_clean):
    """
    Match schedule entries with delays by origin location and time.
    
    Args:
        schedule_expanded_df: Expanded schedule DataFrame
        delays_clean: Cleaned delays DataFrame
        
    Returns:
        DataFrame: Matched results
    """
    return schedule_expanded_df.merge(
        delays_clean,
        left_on=['TRAIN_SERVICE_CODE', 'PLANNED_ORIGIN_LOCATION_CODE', 'PLANNED_ORIGIN_GBTT_DATETIME', 'current_day'],
        right_on=['TRAIN_SERVICE_CODE', 'PLANNED_ORIGIN_LOCATION_CODE', 'origin_time', 'delay_day'],
        how='left',
        suffixes=('', '_delay')
    )


def _match_by_destination(schedule_expanded_df, delays_clean):
    """
    Match schedule entries with delays by destination location and time.
    
    Args:
        schedule_expanded_df: Expanded schedule DataFrame
        delays_clean: Cleaned delays DataFrame
        
    Returns:
        DataFrame: Matched results
    """
    return schedule_expanded_df.merge(
        delays_clean,
        left_on=['TRAIN_SERVICE_CODE', 'PLANNED_DEST_LOCATION_CODE', 'PLANNED_DEST_GBTT_DATETIME', 'current_day'],
        right_on=['TRAIN_SERVICE_CODE', 'PLANNED_DEST_LOCATION_CODE', 'dest_time', 'delay_day'],
        how='left',
        suffixes=('', '_delay')
    )


def _combine_match_results(origin_matches, dest_matches):
    """
    Combine origin and destination matches, prioritizing origin matches.
    
    Args:
        origin_matches: Origin match DataFrame
        dest_matches: Destination match DataFrame
        
    Returns:
        DataFrame: Combined matches
    """
    combined = origin_matches.copy()
    
    # Fill in destination-only matches
    dest_only_mask = combined['PFPI_MINUTES'].isna() & dest_matches['PFPI_MINUTES'].notna()
    
    for col in ['PFPI_MINUTES', 'INCIDENT_REASON', 'INCIDENT_NUMBER', 'EVENT_TYPE', 
                'SECTION_CODE', 'EVENT_DATETIME', 'INCIDENT_START_DATETIME', 'event_time', 'delay_day']:
        if col in dest_matches.columns:
            combined.loc[dest_only_mask, col] = dest_matches.loc[dest_only_mask, col]
    
    return combined


def _build_unmatched_entries_list(unmatched_delays, st_code):
    """
    Build list of unmatched delay entries for addition to results.
    
    Args:
        unmatched_delays: DataFrame of unmatched delays
        st_code: Station code being analyzed
        
    Returns:
        list: List of delay entry dictionaries
    """
    if unmatched_delays.empty:
        return []
    
    unmatched_entries = []
    for _, delay_row in unmatched_delays.iterrows():
        entry = build_unmatched_entry(delay_row, st_code)
        unmatched_entries.append(entry)
    
    return unmatched_entries


# =======================================================================================
# BATCH PROCESSING OPTIMIZATION FUNCTIONS 
# =======================================================================================
# These functions optimize batch processing by loading data once and reusing it

def load_schedule_data_once(schedule_data, reference_files):
    """
    Load schedule data once to avoid reloading for each station.
    
    Args:
        schedule_data (dict): Dictionary containing schedule data file paths
        reference_files (dict): Dictionary containing reference file paths
        
    Returns:
        tuple: (schedule_data_loaded, stanox_ref, tiploc_to_stanox)
    """
    try:
        # Load schedule data
        print("  Loading schedule pickle file...")
        schedule_data_loaded = pd.read_pickle(schedule_data["schedule"])
        
        # Load reference data
        print("  Loading reference data...")
        with open(reference_files["all dft categories"], 'r') as f:
            reference_data = json.load(f)
        # Convert to DataFrame format expected by process_schedule
        stanox_ref = pd.DataFrame(reference_data)
        
        # Create TIPLOC to STANOX mapping
        print("  Creating TIPLOC to STANOX mapping...")
        # Check if the reference data has the expected columns
        if 'tiploc' in stanox_ref.columns and 'stanox' in stanox_ref.columns:
            tiploc_to_stanox = dict(zip(stanox_ref['tiploc'], stanox_ref['stanox']))
        elif 'tiploc_code' in stanox_ref.columns and 'stanox' in stanox_ref.columns:
            tiploc_to_stanox = dict(zip(stanox_ref['tiploc_code'], stanox_ref['stanox']))
        else:
            print(f"  Warning: Expected TIPLOC columns not found. Available columns: {list(stanox_ref.columns)}")
            tiploc_to_stanox = {}
        
        return schedule_data_loaded, stanox_ref, tiploc_to_stanox
        
    except Exception as e:
        print(f"  Error loading schedule data: {e}")
        return None, None, None


def load_incident_data_once(incident_files):
    """
    Load all incident data once to avoid reloading for each station.
    
    Args:
        incident_files (dict): Dictionary with period names as keys and file paths as values
        
    Returns:
        dict: Dictionary with period names as keys and loaded DataFrames as values
    """
    incident_data_loaded = {}
    
    try:
        for period_name, file_path in incident_files.items():
            print(f"  Loading incident data for period: {period_name}")
            df = pd.read_csv(file_path)
            incident_data_loaded[period_name] = df
            
        return incident_data_loaded
        
    except Exception as e:
        print(f"  Error loading incident data: {e}")
        return {}


def process_delays_optimized(incident_data_loaded, st_code, output_dir=None):
    """
    Process delays using pre-loaded incident data to avoid file I/O.
    
    Args:
        incident_data_loaded (dict): Pre-loaded incident data by period
        st_code (str): The station code to filter delays
        output_dir (str, optional): Directory to save converted JSON files (not used in optimized mode)
        
    Returns:
        dict: Dictionary with period names as keys and processed DataFrames as values
    """
    columns_to_remove = [
        "ATTRIBUTION_STATUS", "INCIDENT_EQUIPMENT", "APPLICABLE_TIMETABLE_FLAG", "TRACTION_TYPE",
        "TRAILING_LOAD"
    ]
    
    processed_delays = {}
    
    # Ensure st_code is an integer for proper comparison with STANOX columns
    st_code_int = int(st_code)
    
    for period_name, df in incident_data_loaded.items():
        try:
            # Remove irrelevant columns
            df_cleaned = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
            
            # Filter rows where START_STANOX or END_STANOX matches st_code
            mask = (df_cleaned['START_STANOX'] == st_code_int) | (df_cleaned['END_STANOX'] == st_code_int)
            df_filtered = df_cleaned[mask]
            
            if not df_filtered.empty:
                processed_delays[period_name] = df_filtered
                print(f"    Processed {len(df_filtered)} delay entries for period {period_name}")
            
        except Exception as e:
            print(f"    Error processing period {period_name}: {e}")
            continue
    
    return processed_delays