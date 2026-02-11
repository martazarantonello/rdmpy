
# THIS IS NOT A FINAL VERSION OF THE SCRIPT.
# THIS FILE NEEDS TO BE FULLY OPTIMIZED FOR BATCH PROCESSING OF STATIONS BY CATEGORY.


"""
Save processed schedule and delay data to parquet files for railway stations by DFT category.
This script can process any DFT category (A, B, C1, C2) or all categories at once.
Processes schedule data, applies delays, and saves the results as pandas DataFrames
organized by day of the week for each station.

Usage:
- Single station: python -m preprocess.preprocessor <STANOX_CODE>
- Category A: python -m preprocess.preprocessor --category-A
- Category B: python -m preprocess.preprocessor --category-B
- Category C1: python -m preprocess.preprocessor --category-C1
- Category C2: python -m preprocess.preprocessor --category-C2
- All categories: python -m preprocess.preprocessor --all-categories
- Interactive: python -m preprocess.preprocessor
"""

import json
import os
import pandas as pd
import pickle
from datetime import datetime
from rdmpy.utils import adjust_schedule_timeline
from rdmpy.utils import process_schedule, get_day_code_mapping
from rdmpy.utils import process_delays
from rdmpy.utils import load_schedule_data_once, load_incident_data_once, process_delays_optimized
from demo.data.schedule import schedule_data
from demo.data.reference import reference_files
from demo.data.incidents import incident_files


def get_weekday_from_schedule_entry(entry):
    """
    Extract the primary weekday from a schedule entry for sorting purposes.
    
    Args:
        entry: Schedule entry dictionary
        
    Returns:
        int: Weekday index (0=Monday, 6=Sunday) for sorting
    """
    # If there's a delay day, use that for sorting
    if entry.get('DELAY_DAY'):
        day_mapping = get_day_code_mapping()
        # Reverse mapping: day code to index
        reverse_mapping = {v: k for k, v in day_mapping.items()}
        return reverse_mapping.get(entry['DELAY_DAY'], 0)
    
    # Otherwise, use the first day from ENGLISH_DAY_TYPE
    english_day_types = entry.get('ENGLISH_DAY_TYPE', [])
    if english_day_types:
        day_mapping = get_day_code_mapping()
        # Reverse mapping: day code to index
        reverse_mapping = {v: k for k, v in day_mapping.items()}
        return reverse_mapping.get(english_day_types[0], 0)
    
    return 0  # Default to Monday if no day information


def load_stations(category=None):
    """
    Load stations from the reference JSON file.
    
    Args:
        category (str, optional): DFT category to filter by (e.g., 'A', 'B', 'C1', 'C2'). 
                                 If None, returns all stations.
    
    Returns:
        list: List of STANOX codes for the specified category or all stations
    """
    try:
        with open(reference_files["all dft categories"], 'r') as f:
            stations_data = json.load(f)
        
        # Filter by category if specified
        if category:
            filtered_stations = [
                station for station in stations_data 
                if station.get('dft_category') == category
            ]
        else:
            # When no specific category is requested, only include stations that have ANY category
            # (i.e., exclude stations with no category or empty category)
            filtered_stations = [
                station for station in stations_data 
                if station.get('dft_category') and station.get('dft_category').strip()
            ]
        
        # Extract STANOX codes
        stanox_codes = [str(station.get('stanox', '')) for station in filtered_stations if station.get('stanox')]
        
        return stanox_codes
    except Exception as e:
        print(f"Error loading stations: {e}")
        return []


def _process_schedule_step(st_code, schedule_data_loaded=None, stanox_ref=None, tiploc_to_stanox=None):
    """
    Process schedule data for a station.
    
    Args:
        st_code (str): STANOX code to process
        schedule_data_loaded (pd.DataFrame, optional): Pre-loaded schedule data
        stanox_ref (pd.DataFrame, optional): Pre-loaded STANOX reference data
        tiploc_to_stanox (dict, optional): Pre-loaded TIPLOC to STANOX mapping
    
    Returns:
        list: Processed schedule entries
    """
    print("Step 1: Processing schedule data...")
    
    if schedule_data_loaded is not None and stanox_ref is not None and tiploc_to_stanox is not None:
        print("  Using pre-loaded schedule data (OPTIMIZED)")
        processed_schedule = process_schedule(
            st_code, None, None,
            train_count=None, tiploc=None,
            schedule_data_loaded=schedule_data_loaded,
            stanox_ref=stanox_ref,
            tiploc_to_stanox=tiploc_to_stanox
        )
    else:
        print("  Loading schedule data from files (LEGACY)")
        processed_schedule = process_schedule(st_code, schedule_data, reference_files)
    
    print(f"Processed {len(processed_schedule)} schedule entries")
    return processed_schedule


def _process_delays_step(st_code, incident_data_loaded=None):
    """
    Process delay/incident data for a station.
    
    Args:
        st_code (str): STANOX code to process
        incident_data_loaded (dict, optional): Pre-loaded incident data by period
    
    Returns:
        list: Flattened list of delay records
    """
    print("Step 2: Processing delay data...")
    
    if incident_data_loaded is not None:
        print("  Using pre-loaded incident data (OPTIMIZED)")
        processed_delays_dict = process_delays_optimized(incident_data_loaded, st_code, None)
    else:
        print("  Loading incident data from files (LEGACY)")
        temp_delays_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_delays")
        os.makedirs(temp_delays_path, exist_ok=True)
        processed_delays_dict = process_delays(incident_files, st_code, temp_delays_path)
    
    # Convert dictionary of DataFrames to list of delay records
    processed_delays = []
    for period_name, df in processed_delays_dict.items():
        if hasattr(df, 'to_dict'):  # Check if it's a DataFrame
            processed_delays.extend(df.to_dict('records'))
    
    print(f"Processed {len(processed_delays)} delay entries from {len(processed_delays_dict)} periods")
    return processed_delays


def _deduplicate_timeline(schedule_timeline_adjusted):
    """
    Remove identical duplicate entries from timeline.
    
    Args:
        schedule_timeline_adjusted (list): Schedule entries to deduplicate
    
    Returns:
        list: Deduplicated schedule entries
    """
    print("Step 4: Removing identical duplicate entries...")
    
    seen = set()
    deduplicated = []
    for entry in schedule_timeline_adjusted:
        # Create a hashable representation that allows same train to have different delays on different days
        key_fields = []
        for k, v in sorted(entry.items()):
            if isinstance(v, (str, int, float, type(None))):
                key_fields.append((k, v))
            elif isinstance(v, list):
                key_fields.append((k, tuple(v)))
        
        entry_key = tuple(key_fields)
        
        if entry_key not in seen:
            seen.add(entry_key)
            deduplicated.append(entry)
    
    print(f"Removed {len(schedule_timeline_adjusted) - len(deduplicated)} identical duplicate entries")
    print(f"Deduplicated timeline contains {len(deduplicated)} entries")
    return deduplicated


def _organize_by_weekday(schedule_timeline_adjusted):
    """
    Organize schedule entries by weekday.
    
    Args:
        schedule_timeline_adjusted (list): Schedule entries
    
    Returns:
        dict: Entries organized by day code
    """
    print("Step 5: Organizing data by weekday...")
    
    day_mapping = get_day_code_mapping()
    weekday_data = {day_code: [] for day_code in day_mapping.values()}
    
    for entry in schedule_timeline_adjusted:
        english_day_types = entry.get('ENGLISH_DAY_TYPE', [])
        
        if english_day_types:
            for day_code in english_day_types:
                if day_code in weekday_data:
                    day_specific_entry = entry.copy()
                    day_specific_entry['DATASET_TYPE'] = 'MULTI_DAY' if len(english_day_types) > 1 else 'SINGLE_DAY'
                    day_specific_entry['WEEKDAY'] = day_code
                    weekday_data[day_code].append(day_specific_entry)
        else:
            weekday_index = get_weekday_from_schedule_entry(entry)
            day_code = day_mapping[weekday_index]
            entry['DATASET_TYPE'] = 'FALLBACK'
            entry['WEEKDAY'] = day_code
            weekday_data[day_code].append(entry)
    
    return weekday_data


def _convert_to_dataframes(weekday_data):
    """
    Convert weekday-organized data to pandas DataFrames with sorting.
    
    Args:
        weekday_data (dict): Data organized by day code
    
    Returns:
        dict: DataFrames organized by day code
    """
    print("Step 6: Converting to pandas DataFrames...")
    
    def safe_sort_key(x):
        """Sort key that handles NaN and non-numeric ACTUAL_CALLS values."""
        actual_calls = x.get("ACTUAL_CALLS")
        if actual_calls is None or actual_calls == "NA":
            return 0
        if isinstance(actual_calls, (int, float)):
            if pd.isna(actual_calls):
                return 0
            return int(actual_calls)
        if isinstance(actual_calls, str):
            if actual_calls.isdigit():
                return int(actual_calls)
            return 0
        return 0
    
    weekday_dataframes = {}
    
    for day_code, entries in weekday_data.items():
        if entries:
            entries.sort(key=safe_sort_key)
            df = pd.DataFrame(entries)
            weekday_dataframes[day_code] = df
            print(f"Created DataFrame for {day_code} with {len(df)} entries")
    
    return weekday_dataframes


def save_processed_data_by_weekday_to_dataframe(st_code, output_dir="processed_data", 
                                               schedule_data_loaded=None, stanox_ref=None, 
                                               tiploc_to_stanox=None, incident_data_loaded=None):
    """
    Process schedule and delay data, then save to pandas DataFrame organized by weekday.
    OPTIMIZED VERSION: Accepts pre-loaded data to avoid file I/O for each station.
    
    Args:
        st_code (str): STANOX code to process
        output_dir (str): Directory to save output files
        schedule_data_loaded (pd.DataFrame, optional): Pre-loaded schedule data
        stanox_ref (pd.DataFrame, optional): Pre-loaded STANOX reference data
        tiploc_to_stanox (dict, optional): Pre-loaded TIPLOC to STANOX mapping
        incident_data_loaded (dict, optional): Pre-loaded incident data by period
        
    Returns:
        dict: Dictionary containing processed data as pandas DataFrames organized by weekday
    """
    print(f"Processing data for STANOX: {st_code}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process schedule data
    processed_schedule = _process_schedule_step(st_code, schedule_data_loaded, stanox_ref, tiploc_to_stanox)
    
    if not processed_schedule:
        print(f"No schedule data found for station {st_code}. Skipping...")
        return None
    
    # Step 2: Process delay data
    processed_delays = _process_delays_step(st_code, incident_data_loaded)
    
    # Step 3: Adjust schedule timeline with delays
    print("Step 3: Adjusting schedule timeline with delays...")
    schedule_timeline_adjusted = adjust_schedule_timeline(processed_schedule, processed_delays, st_code)
    print(f"Adjusted timeline contains {len(schedule_timeline_adjusted)} entries")
    
    if not schedule_timeline_adjusted:
        print(f"No adjusted schedule data found for station {st_code}. Skipping...")
        return None
    
    # Step 4: Remove identical duplicates
    schedule_timeline_adjusted = _deduplicate_timeline(schedule_timeline_adjusted)
    
    # Step 5: Organize data by weekday
    weekday_data = _organize_by_weekday(schedule_timeline_adjusted)
    
    # Step 6: Convert to pandas DataFrames
    weekday_dataframes = _convert_to_dataframes(weekday_data)
    
    return weekday_dataframes


def _load_all_reference_data():
    """
    Load all required reference data for batch processing.
    
    Returns:
        tuple: (schedule_data_loaded, stanox_ref, tiploc_to_stanox, incident_data_loaded)
    """
    print("\n" + "="*60)
    print("PRE-LOADING DATA (this will save significant time)")
    print("="*60)
    
    print("Step 1: Loading schedule data...")
    schedule_data_loaded, stanox_ref, tiploc_to_stanox = load_schedule_data_once(schedule_data, reference_files)
    print(f"Loaded schedule data with {len(schedule_data_loaded) if len(schedule_data_loaded) else 0} entries")
    
    print("Step 2: Pre-loading incident data...")
    incident_data_loaded = load_incident_data_once(incident_files)
    print(f"Loaded incident data for {len(incident_data_loaded)} periods")
    
    print("Data pre-loading complete. Now processing stations...")
    print("="*60)
    
    return schedule_data_loaded, stanox_ref, tiploc_to_stanox, incident_data_loaded


def _cleanup_existing_station_folders(stations, output_dir):
    """
    Remove old station folders to clear out existing parquet files.
    
    Args:
        stations (list): List of STANOX codes
        output_dir (str): Output directory path
    
    Returns:
        int: Number of folders removed
    """
    print(f"Cleaning up existing station folders...")
    import shutil
    
    files_removed = 0
    for station_code in stations:
        station_folder = os.path.join(output_dir, station_code)
        if os.path.exists(station_folder):
            try:
                shutil.rmtree(station_folder)
                files_removed += 1
                print(f"  Removed folder: {station_folder}")
            except Exception as e:
                print(f"  Warning: Could not remove {station_folder}: {e}")
    
    if files_removed > 0:
        print(f"Cleaned up {files_removed} existing station folders")
    
    return files_removed


def _process_single_station_and_save(st_code, i, total_stations, output_dir, 
                                      schedule_data_loaded, stanox_ref, 
                                      tiploc_to_stanox, incident_data_loaded):
    """
    Process a single station and save its data as parquet files.
    
    Args:
        st_code (str): STANOX code to process
        i (int): Station number in batch (for display)
        total_stations (int): Total stations in batch (for display)
        output_dir (str): Output directory path
        schedule_data_loaded: Pre-loaded schedule data
        stanox_ref: Pre-loaded STANOX reference
        tiploc_to_stanox: Pre-loaded TIPLOC mapping
        incident_data_loaded: Pre-loaded incident data
    
    Returns:
        tuple: (success: bool, station_data: dict or None, total_entries: int)
    """
    print(f"\n{'='*60}")
    print(f"Processing station {i}/{total_stations}: {st_code}")
    print(f"{'='*60}")
    
    try:
        # Process the station data using pre-loaded data
        station_data = save_processed_data_by_weekday_to_dataframe(
            st_code, output_dir,
            schedule_data_loaded=schedule_data_loaded,
            stanox_ref=stanox_ref,
            tiploc_to_stanox=tiploc_to_stanox,
            incident_data_loaded=incident_data_loaded
        )
        
        if station_data:
            # Create station-specific folder
            station_folder = os.path.join(output_dir, st_code)
            os.makedirs(station_folder, exist_ok=True)
            
            # Save each weekday DataFrame as a separate parquet file
            total_entries = 0
            files_created = []
            for day_code, df in station_data.items():
                filename = os.path.join(station_folder, f"{day_code}.parquet")
                df.to_parquet(filename, index=False)
                files_created.append(filename)
                total_entries += len(df)
                print(f"Saved {len(df)} entries for {day_code} to {filename}")
            
            print(f"Successfully processed station {st_code}: {total_entries} total entries")
            return True, files_created, total_entries
        else:
            print(f"No data found for station {st_code}")
            return False, [], 0
            
    except Exception as e:
        print(f"Error processing station {st_code}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, [], 0


def _print_processing_summary(results):
    """
    Print summary of batch processing results.
    
    Args:
        results (dict): Processing results dictionary
    """
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(results['successful_stations'])} stations")
    print(f"Failed to process: {len(results['failed_stations'])} stations")
    print(f"Total parquet files created: {len(results['files_created'])}")
    print(f"Output directory: {results['output_dir']}")
    
    if results['successful_stations']:
        print(f"\nSuccessful stations: {results['successful_stations']}")
        print(f"\nTotal entries by station:")
        for station, count in results['total_entries_by_station'].items():
            print(f"  {station}: {count} entries")
    
    if results['failed_stations']:
        print(f"\nFailed stations: {results['failed_stations']}")


def save_stations_by_category(category=None, output_dir="processed_data"):
    """
    Process and save data for stations by DFT category as parquet files.
    FULLY OPTIMIZED VERSION: Loads schedule and delay data once, reuses for all stations.
    No redundant file I/O operations during batch processing.
    
    Args:
        category (str, optional): DFT category to process ('A', 'B', 'C1', 'C2'). 
                                 If None, processes all categories.
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Summary of processing results
    """
    # Load stations for the specified category
    if category:
        print(f"Loading Category {category} stations...")
        stations = load_stations(category=category)
        category_label = f"category_{category.lower()}"
    else:
        print("Loading all DFT category stations...")
        stations = load_stations()
        category_label = "all_categories"
    
    if not stations:
        print(f"No stations found for category {category if category else 'all'}. Exiting.")
        return None
    
    print(f"Found {len(stations)} stations: {stations[:10]}{'...' if len(stations) > 10 else ''}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Pre-load all reference data
    schedule_data_loaded, stanox_ref, tiploc_to_stanox, incident_data_loaded = _load_all_reference_data()
    
    # Clean up existing station folders
    _cleanup_existing_station_folders(stations, output_dir)
    
    # Initialize results dictionary
    results = {
        'category': category if category else 'all',
        'successful_stations': [],
        'failed_stations': [],
        'total_entries_by_station': {},
        'files_created': [],
        'output_dir': output_dir
    }
    
    # Process each station
    for i, st_code in enumerate(stations, 1):
        success, files_created, total_entries = _process_single_station_and_save(
            st_code, i, len(stations), output_dir,
            schedule_data_loaded, stanox_ref, tiploc_to_stanox, incident_data_loaded
        )
        
        if success:
            results['successful_stations'].append(st_code)
            results['total_entries_by_station'][st_code] = total_entries
            results['files_created'].extend(files_created)
        else:
            results['failed_stations'].append(st_code)
    
    # Print summary
    _print_processing_summary(results)
    
    return results


def save_all_category_a_stations(output_dir="processed_data"):
    """
    Backward compatibility function for processing Category A stations only.
    
    Args:
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Summary of processing results
    """
    return save_stations_by_category(category='A', output_dir=output_dir)


def main(st_code=None, process_category=None, process_all_categories=False):
    """
    Main function to demonstrate the data processing and saving functionality.
    
    Args:
        st_code (str, optional): STANOX code to process. If not provided and no category processing, will prompt for input.
        process_category (str, optional): DFT category to process ('A', 'B', 'C1', 'C2').
        process_all_categories (bool): If True, process all categories instead of a single station.
    """
    if process_category or process_all_categories:
        category = None if process_all_categories else process_category
        category_desc = "all categories" if process_all_categories else f"Category {process_category}"
        print(f"Processing {category_desc} stations...")
        try:
            result = save_stations_by_category(category=category)
            
            if result:
                print(f"\n" + "="*50)
                print(f"{category_desc.upper()} STATIONS PROCESSING COMPLETE")
                print("="*50)
                print(f"Successfully processed: {len(result['successful_stations'])} stations")
                print(f"Failed to process: {len(result['failed_stations'])} stations")
                print(f"Total files created: {len(result['files_created'])}")
                
                if result['successful_stations']:
                    print(f"\nSuccessful stations: {result['successful_stations']}")
                    
                if result['failed_stations']:
                    print(f"\nFailed stations: {result['failed_stations']}")
            else:
                print(f"No {category_desc} stations were processed.")
                
        except Exception as e:
            print(f"Error processing {category_desc} stations: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        # Process single station (original functionality)
        # Get STANOX code from parameter or user input
        if st_code is None:
            st_code = input("Enter STANOX code to process: ").strip()
            if not st_code:
                print("No STANOX code provided. Exiting.")
                return None
        
        try:
            # For single station processing, we don't pre-load data (it's not worth it for one station)
            result = save_processed_data_by_weekday_to_dataframe(st_code)
            
            if result:
                print(f"\n" + "="*50)
                print("PROCESSING COMPLETE")
                print("="*50)
                total_entries = sum(len(df) for df in result.values())
                print(f"Total entries processed: {total_entries}")
                print(f"DataFrames created: {len(result)}")
                
                print("\nDataFrames created:")
                for day_code, df in result.items():
                    print(f"  {day_code}: {len(df)} entries")
                    
                # Optionally save as parquet files
                save_choice = input("\nSave DataFrames as parquet files? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_dir = "processed_data"
                    station_folder = os.path.join(output_dir, st_code)
                    os.makedirs(station_folder, exist_ok=True)
                    
                    for day_code, df in result.items():
                        filename = os.path.join(station_folder, f"{day_code}.parquet")
                        df.to_parquet(filename, index=False)
                        print(f"Saved {filename}")
                        
            else:
                print(f"No data found for station {st_code}")
                
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all-categories":
            # Process all categories
            main(process_all_categories=True)
        elif sys.argv[1].startswith("--category-"):
            # Process specific category (e.g., --category-A, --category-B, --category-C1, --category-C2)
            category = sys.argv[1].replace("--category-", "").upper()
            main(process_category=category)
        else:
            # Process specific station code
            main(st_code=sys.argv[1])
    else:
        # Interactive mode
        print("Choose processing mode:")
        print("1. Process a single station")
        print("2. Process Category A stations")
        print("3. Process Category B stations")
        print("4. Process Category C1 stations")
        print("5. Process Category C2 stations")
        print("6. Process ALL categories")
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == "2":
            main(process_category='A')
        elif choice == "3":
            main(process_category='B')
        elif choice == "4":
            main(process_category='C1')
        elif choice == "5":
            main(process_category='C2')
        elif choice == "6":
            main(process_all_categories=True)
        else:
            main()
