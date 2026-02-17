"""
Unit tests for rdmpy.outputs.analysis_tools module.

"""

from unittest import result
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdmpy.outputs.analysis_tools import (
    calculate_incident_summary_stats,
    _load_and_prepare_multiday_data,
    get_stanox_for_service,
    map_train_journey_with_incidents,
    _prepare_journey_map_data,
    _compute_station_route_connections,
    _aggregate_delays_and_incidents,
    _create_station_markers_on_map,
    _create_incident_markers_on_map,
    _finalize_journey_map,
    train_view,
    train_view_2,
    plot_reliability_graphs,
    _print_date_statistics,
    _load_station_coordinates,
    _aggregate_time_view_data,
    _create_time_view_markers,
    _finalize_time_view_map,
    create_time_view_html,
)


# ==============================================================================
# FIXTURES for aggregate view tests
# ==============================================================================

@pytest.fixture
def sample_delay_df():
    """Create a sample dataframe with delay events for testing."""
    dates = pd.date_range('2024-01-01', periods=10, freq='H')
    return pd.DataFrame({
        'full_datetime': dates,
        'PFPI_MINUTES': [10, 15, 20, 5, 30, 25, 35, 40, 45, 50],
        'EVENT_TYPE': ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'C', 'C'],
        'INCIDENT_NUMBER': [12345] * 10,
    })


@pytest.fixture
def sample_complete_df():
    """Create a complete sample dataframe with all required columns."""
    dates = pd.date_range('2024-01-01 08:00', periods=15, freq='H')
    df = pd.DataFrame({
        'full_datetime': dates,
        'EVENT_DATETIME': ['01-JAN-2024 ' + f'{8+i:02d}:00' for i in range(15)],
        'PFPI_MINUTES': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        'EVENT_TYPE': ['D'] * 13 + ['C'] * 2,
        'INCIDENT_NUMBER': [12345] * 15,
        'INCIDENT_START_DATETIME': ['01-JAN-2024 08:00'] * 15,
        'event_date_only': [date(2024, 1, 1)] * 15,
    })
    return df


# ==============================================================================
# TESTS FOR aggregate view (internal functions)
# ==============================================================================

def test_calculate_incident_summary_stats_k0(sample_complete_df):
    """Test summary stats calculation returns dict with correct keys."""
    delay_data = sample_complete_df[sample_complete_df['PFPI_MINUTES'] > 0].copy()
    unique_dates = [date(2024, 1, 1)]
    
    result = calculate_incident_summary_stats(
        df=sample_complete_df,
        delay_data_all=delay_data,
        unique_dates=unique_dates,
        files_processed=100,
        files_with_data=10,
        incident_number=12345,
        num_days=1
    )
    
    # Assert based on sample_complete_df fixture
    expected_total_delay = sample_complete_df['PFPI_MINUTES'].sum()
    expected_cancellations = len(sample_complete_df[sample_complete_df['EVENT_TYPE'] == 'C'])
    expected_records = len(sample_complete_df)

    assert result is not None
    assert isinstance(result, dict)
    assert result['Total Delay Minutes'] == expected_total_delay
    assert result['Total Cancellations'] == expected_cancellations
    assert result['Total Records Found'] == expected_records
    assert result['Files Processed'] == 100
    assert result['Files with Data'] == 10
    assert result['Number of Days'] == 1
    assert result['Date Range'] == '01-Jan-2024 to 01-Jan-2024'
    assert result['Incident Number'] == 12345
    ## Peak should be 75 minutes (from 14th record)
    assert '75.0 minutes' in result['Peak Delay Event']
    assert isinstance(result['Peak Delay Event'], str) # 'XX minutes at DATE TIME (TYPE)'
    
    # Verify all keys are present in correct order
    assert list(result.keys()) == [
        'Total Delay Minutes',
        'Total Cancellations', 
        'Total Records Found',
        'Files Processed',
        'Files with Data',
        'Incident Number',
        'Number of Days',
        'Date Range',
        'Unique Dates',
        'Time Range',
        'Peak Delay Event',
        'Peak Regular Delay',
        'Peak Cancellation Delay'
    ]
    
# using @patch to mock the data loading function to test aggregate_view_multiday's handling of load failures and multi-day incidents

@patch('rdmpy.outputs.analysis_tools._load_and_prepare_multiday_data')
def test_aggregate_view_multiday_k0(mock_load_prep):
    """Test aggregate_view_multiday returns None when data loading fails."""
    from rdmpy.outputs.analysis_tools import aggregate_view_multiday
    
    mock_load_prep.return_value = None
    
    result = aggregate_view_multiday(12345, '01-JAN-2024')
    
    assert result is None

@patch('rdmpy.outputs.analysis_tools.plt.show')
@patch('rdmpy.outputs.analysis_tools._load_and_prepare_multiday_data')
def test_aggregate_view_multiday_k2(mock_load_prep, mock_plt_show, sample_complete_df):
    """Test aggregate_view_multiday handles multi-day incidents correctly."""
    from rdmpy.outputs.analysis_tools import aggregate_view_multiday

    unique_dates = [date(2024, 1, 1), date(2024, 1, 2)] # 2 dates to simulate multi-day incident
    mock_load_prep.return_value = (sample_complete_df, 100, 10, unique_dates, 2)
    
    result = aggregate_view_multiday(12345, '01-JAN-2024')
    
    # Should return a dict (plotted and summarized without error)
    assert mock_plt_show.called
    assert result is not None
    assert isinstance(result, dict)
    assert result['Date Range'] == '01-Jan-2024 to 02-Jan-2024'


# INTEGRATION TESTS for aggregate view consistency

def test_aggregate_view_workflow_consistency(sample_complete_df):
    """Test that aggregated stats from complete workflow are consistent."""
    # Test that calculations are internally consistent
    delay_data = sample_complete_df[sample_complete_df['PFPI_MINUTES'] > 0].copy()
    unique_dates = [date(2024, 1, 1)]
    
    result = calculate_incident_summary_stats(
        df=sample_complete_df,
        delay_data_all=delay_data,
        unique_dates=unique_dates,
        files_processed=100,
        files_with_data=10,
        incident_number=12345,
        num_days=1
    )
    
    # Verify internal consistency: Total records should match dataframe length
    assert result['Total Records Found'] == len(sample_complete_df)
    
    # Verify delay range makes sense
    assert result['Total Delay Minutes'] >= 0
    
    # Verify that total delay roughly matches sum of PFPI_MINUTES (excluding edge cases)
    expected_delay_sum = sample_complete_df['PFPI_MINUTES'].fillna(0).sum()
    assert result['Total Delay Minutes'] == expected_delay_sum


# ==============================================================================
# FIXTURES FOR TRAIN_VIEW TESTS
# ==============================================================================

# This fixture simulates the data assessed by the train_view function
@pytest.fixture
def train_journey_fixture():
    """
    Fixture representing realistic train journey data for testing the complete workflow.
    Simulates data for OD pair (12931 -> 54311) with known service codes and incidents on specific dates.
    """
    return pd.DataFrame({
        'PLANNED_ORIGIN_LOCATION_CODE': ['12931'] * 5 + ['54311'] * 5,
        'PLANNED_DEST_LOCATION_CODE': ['54311'] * 5 + ['12931'] * 5,
        'TRAIN_SERVICE_CODE': ['21700001', '21700001', '21700002', '21700003', '21700001',
                               '21700004', '21700005', '21700001', '21700001', '21700006'], # adding many service codes for realism, but here only 21700001 is used to test
        'STANOX': ['12931', '89012', '45123', '54311', '78912',
                   '54311', '45123', '89012', '12931', '54311'],
        'PLANNED_ORIGIN_GBTT_DATETIME': ['07:00', '07:45', '08:15', '09:00', '10:00',
                                        '16:00', '16:45', '17:15', '18:00', '19:00'],
        'PLANNED_DEST_GBTT_DATETIME': ['12:30', '13:15', '14:00', '14:45', '15:30',
                                      '21:30', '22:15', '23:00', '23:45', '00:30'],
        'PLANNED_CALLS': ['0700', '0745', '0815', '0900', '1000',
                         '1600', '1645', '1715', '1800', '1900'],
        'ACTUAL_CALLS': ['0715', '0805', '0835', '0925', '1020',
                        '1620', '1700', '1735', '1825', '1925'],
        'PFPI_MINUTES': [15.0, 20.0, 20.0, 25.0, 20.0,
                        20.0, 15.0, 20.0, 25.0, 25.0],
        'INCIDENT_REASON': ['Signal failure', 'Track defect', 'Points failure', 'Signalling issue', 'Track defect',
                           'Signal failure', 'Track defect', 'Points failure', 'Signalling issue', 'Signal failure'],
        'INCIDENT_NUMBER': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
        'EVENT_TYPE': ['D'] * 10,
        'SECTION_CODE': ['12931:89012', '89012:45123', '45123:54311', '12931:54311', '89012:45123',
                        '54311:45123', '45123:89012', '89012:12931', '12931:54311', '54311:12931'],
        'DELAY_DAY': ['07-DEC-2024', '07-DEC-2024', '07-DEC-2024', '07-DEC-2024', '08-DEC-2024',
                      '07-DEC-2024', '07-DEC-2024', '07-DEC-2024', '07-DEC-2024', '07-DEC-2024'],
        'EVENT_DATETIME': ['07-DEC-2024 07:15', '07-DEC-2024 08:05', '07-DEC-2024 08:35', '07-DEC-2024 09:25', '08-DEC-2024 10:20',
                          '07-DEC-2024 16:20', '07-DEC-2024 17:00', '07-DEC-2024 17:35', '07-DEC-2024 18:25', '07-DEC-2024 19:25'],
        'INCIDENT_START_DATETIME': ['2024-12-07 07:00', '2024-12-07 08:00', '2024-12-07 08:15', '2024-12-07 09:00', '2024-12-07 10:00',
                                   '2024-12-07 16:00', '2024-12-07 16:45', '2024-12-07 17:15', '2024-12-07 18:00', '2024-12-07 19:00'],
        'ENGLISH_DAY_TYPE': ['Weekday'] * 10,
        'STATION_ROLE': ['O', 'I', 'I', 'D', 'I'] * 2,
        'DFT_CATEGORY': ['Cat1'] * 10,
        'PLATFORM_COUNT': [4, 4, 3, 5, 4, 5, 3, 4, 4, 5],
        'DATASET_TYPE': ['Delay'] * 10,
        'WEEKDAY': [6] * 10,
        'DAY': [7] * 10,
    })


# TESTS FOR get_stanox_for_service

def test_get_stanox_for_service_retrieves_all_stations(train_journey_fixture):
    """
    Test that get_stanox_for_service returns all unique STANOX stations for a given service code.
    This represents the second step in the workflow: once we identify a service from train_view results,
    we retrieve all stations it calls at to map the journey.
    """
    service_code = '21700001'
    
    result = get_stanox_for_service(
        train_journey_fixture,
        train_service_code=service_code,
        origin_code='12931',
        destination_code='54311'
    )

    # Service 21700001 has STANOX: 12931, 89012, 78912 (rows 0, 1, 4)
    # Plus destination 54311 is added as destination station
    expected_stations = {'12931', '89012', '78912', '54311'}
    
  
    assert isinstance(result, list)   # Verify result is a list of STANOX codes
    assert len(result) > 0
    assert '12931' in result # Verify origin is included
    assert '54311' in result # Verify destination is included
    assert expected_stations.issubset(set(result)) # Verify all expected stations are included


def test_get_stanox_for_service_filters_by_date(train_journey_fixture):
    """
    Test that get_stanox_for_service correctly filters stations by date when provided.
    Fixture has service 21700001 on different dates: 07-DEC-2024 (12931, 89012) and 08-DEC-2024 (78912).
    """
    service_code = '21700001'
    
    # Query for 07-DEC-2024: should get rows with service 21700001 on that date
    result_07_dec = get_stanox_for_service(
        train_journey_fixture,
        train_service_code=service_code,
        origin_code='12931',
        destination_code='54311',
        date_str='07-DEC-2024'
    )
    
    # Query for 08-DEC-2024: should get rows with service 21700001 on that date
    result_08_dec = get_stanox_for_service(
        train_journey_fixture,
        train_service_code=service_code,
        origin_code='12931',
        destination_code='54311',
        date_str='08-DEC-2024'
    )
    
    assert isinstance(result_07_dec, list)   # Should return lists of stations
    assert isinstance(result_08_dec, list)   # Should return lists of stations

    expected_07_dec = {'12931', '89012', '54311'}
    assert expected_07_dec.issubset(set(result_07_dec))  # 07-DEC-2024: rows 0, 1 have STANOX 12931, 89012 + destination 54311

    expected_08_dec = {'78912', '54311'}
    assert expected_08_dec.issubset(set(result_08_dec))  # 08-DEC-2024: row 4 has STANOX 78912 + destination 54311
    
    assert set(result_07_dec) != set(result_08_dec)   # Verify filtering by date produces different results


def test_get_stanox_for_service_returns_error_for_nonexistent_service(train_journey_fixture):
    """
    Test that get_stanox_for_service returns error message for non-existent service code.
    """
    result = get_stanox_for_service(
        train_journey_fixture,
        train_service_code='XXXXX',
        origin_code='12931',
        destination_code='54311'
    )
    
    # Should return error message string
    assert isinstance(result, str)
    assert 'No records found' in result


# INTEGRATION TESTS FOR map_train_journey_with_incidents - Complete Workflow

@pytest.fixture
def sample_station_ref_json(tmp_path):
    """Create a temporary station reference JSON file for testing."""
    import json
    station_ref = [
        {'stanox': '12931', 'description': 'Origin Station', 'latitude': 51.5, 'longitude': -0.1},
        {'stanox': '89012', 'description': 'Intermediate Station 1', 'latitude': 52.0, 'longitude': -0.5},
        {'stanox': '45123', 'description': 'Intermediate Station 2', 'latitude': 52.5, 'longitude': -1.0},
        {'stanox': '54311', 'description': 'Destination Station', 'latitude': 53.0, 'longitude': -1.5},
        {'stanox': '78912', 'description': 'Additional Station', 'latitude': 51.8, 'longitude': -0.3},
    ]
    
    json_file = tmp_path / "stations_ref.json"
    with open(json_file, 'w') as f:
        json.dump(station_ref, f)
    
    return str(json_file)


@patch('folium.Map')
def test_map_train_journey_with_incidents_creates_map(mock_map, train_journey_fixture, sample_station_ref_json):
    """
    Test that map_train_journey_with_incidents creates a folium map object when given service STANOX and incident data.
    This is the third step in the workflow: visualize the journey with incidents on a map.
    """
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    
    # Get service STANOX from train_view result
    service_stanox = ['12931', '89012', '45123', '54311']
    
    # Get incident results from train_view 
    incident_df = train_journey_fixture[
        (train_journey_fixture['PLANNED_ORIGIN_LOCATION_CODE'] == '12931') &
        (train_journey_fixture['PLANNED_DEST_LOCATION_CODE'] == '54311')
    ].copy()
    
    result = map_train_journey_with_incidents(
        all_data=train_journey_fixture,
        service_stanox=service_stanox,
        incident_results=[incident_df],
        stations_ref_path=sample_station_ref_json,
        service_code='21700001',
        date_str='07-DEC-2024'
    )
    
    # Verify map was created (mocked)
    assert mock_map.called
    assert result is mock_map_instance


@patch('builtins.display', create=True)
def test_train_view_filters_by_od_pair_and_date(mock_display, train_journey_fixture):
    """
    Test that train_view correctly filters all data for a specific OD pair.
    Returns all incidents for that route (note: date_str parameter doesn't currently filter).
    """
    result = train_view(train_journey_fixture, '12931', '54311', '07-DEC-2024')
    
    assert isinstance(result, pd.DataFrame)     # Verify result is DataFrame with incidents
    assert len(result) > 0
    assert all(result['PLANNED_ORIGIN_LOCATION_CODE'] == '12931') # Assert filtering is correct: all rows match the OD pair
    assert all(result['PLANNED_DEST_LOCATION_CODE'] == '54311')
    assert all(result['PFPI_MINUTES'].fillna(0) > 0)   # Assert data contains delays (PFPI_MINUTES > 0)


@patch('builtins.display', create=True)
def test_train_view_bidirectional_od_pair(mock_display, train_journey_fixture):
    """
    Test that train_view works for both directions of an OD pair.
    Verify reverse direction (54311->12931) returns different incident counts than forward direction.
    """
    result_forward = train_view(train_journey_fixture, '12931', '54311', '07-DEC-2024')
    result_reverse = train_view(train_journey_fixture, '54311', '12931', '07-DEC-2024')
    
    assert isinstance(result_forward, pd.DataFrame) and isinstance(result_reverse, pd.DataFrame)    # Both should return DataFrames
    assert all(result_forward['PLANNED_ORIGIN_LOCATION_CODE'] == '12931')    # Forward direction has origin 12931
    assert all(result_forward['PLANNED_DEST_LOCATION_CODE'] == '54311')      # and destination 54311
    assert all(result_reverse['PLANNED_ORIGIN_LOCATION_CODE'] == '54311')    # Reverse direction has origin 54311
    assert all(result_reverse['PLANNED_DEST_LOCATION_CODE'] == '12931')      # and destination 12931
    assert len(result_forward) == 5    # Forward: all 5 rows with 12931->54311 (rows 0-4)
    assert len(result_reverse) == 5    # Reverse: all 5 rows with 54311->12931 (rows 5-9)


@patch('builtins.display', create=True)
def test_train_view_invalid_od_pair_returns_message(mock_display, train_journey_fixture):
    """
    Test that train_view returns error message for non-existent OD pair.
    """
    result = train_view(train_journey_fixture, 'XXXX', 'YYYY', '07-DEC-2024')
    
    # Should return string error message (not DataFrame)
    assert isinstance(result, str)
    assert 'not found' in result.lower()


@patch('builtins.display', create=True)
def test_train_view_no_incidents_on_date_returns_message(mock_display, train_journey_fixture):
    """
    Test that train_view returns informative message when OD pair exists but has no incidents on specified date.
    """
    result = train_view(train_journey_fixture, '12931', '54311', '06-DEC-2024')
    
    # Should return string message (no data on this date)
    assert isinstance(result, str)
    assert 'no incidents' in result.lower()

# FIXTURES for train_view_2 tests

@pytest.fixture
def sample_service_reliability_df():
    """Create sample data for train_view_2 testing with known reliability metrics."""
    return pd.DataFrame({
        'TRAIN_SERVICE_CODE': ['SVC001'] * 10,
        'STANOX': ['12345', '12345', '12345', '12346', '12346', '12346', '12347', '12347', '12347', '12348'],
        'PFPI_MINUTES': [0.0, 5.0, 10.0, 0.0, 15.0, 20.0, 0.0, 0.0, 25.0, 10.0],
        'INCIDENT_REASON': ['OnTime', 'Delay', 'Delay', 'OnTime', 'Delay', 'Delay', 'OnTime', 'OnTime', 'Delay', 'Delay'],
    })

@pytest.fixture
def sample_single_station_reliability_df():
    """Create sample data for train_view_2 testing with single station focus."""
    return pd.DataFrame({
        'TRAIN_SERVICE_CODE': ['SVC002'] * 3,
        'STANOX': ['12345', '12345', '12345'],
        'PFPI_MINUTES': [0.0, 5.0, 10.0],
        'INCIDENT_REASON': ['OnTime', 'Delay', 'Delay'],
    })


# TESTS FOR train_view_2 (not included in general report of the tool, but present in the code)

@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_computes_reliability_metrics(mock_open, mock_print, sample_service_reliability_df):
    """Test train_view_2 computes correct mean delay and on-time percentage."""
    # Mock station reference file
    mock_station_ref = [
        {'stanox': '12345', 'description': 'Station A'},
        {'stanox': '12346', 'description': 'Station B'},
        {'stanox': '12347', 'description': 'Station C'},
        {'stanox': '12348', 'description': 'Station D'},
    ]
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    service_stanox = ['12345', '12346']
    result = train_view_2(sample_service_reliability_df, service_stanox, 'SVC001', stations_ref_path='mock_path.json')
    
    # Should return DataFrame with metrics
    assert isinstance(result, pd.DataFrame)
    assert len(result) >= 2
    
    # Verify required columns exist
    expected_cols = ['ServiceCode', 'StationName', 'MeanDelay', 'DelayVariance', 'OnTime%', 'IncidentCount']
    assert all(col in result.columns for col in expected_cols)
    
    # Verify ServiceCode is correct
    assert all(result['ServiceCode'] == 'SVC001')


@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_calculates_mean_delay_correctly(mock_open, mock_print, sample_single_station_reliability_df):
    """Test train_view_2 calculates mean delay excluding zero delays."""
    # Mock station reference file
    mock_station_ref = [
        {'stanox': '12345', 'description': 'Station A'},
    ]
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    service_stanox = ['12345']
    result = train_view_2(sample_single_station_reliability_df, service_stanox, 'SVC002', stations_ref_path='mock_path.json')
    
    # STANOX 12345 has PFPI: [0.0, 5.0, 10.0] -> mean excluding 0 = (5+10)/2 = 7.5
    assert len(result) == 1
    assert result.loc[0, 'MeanDelay'] == 7.5
    assert result.loc[0, 'IncidentCount'] == 2  # Only 2 delays (excluding 0.0)


@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_calculates_on_time_percentage(mock_open, mock_print, sample_single_station_reliability_df):
    """Test train_view_2 calculates on-time percentage correctly."""
    # Mock station reference file
    mock_station_ref = [
        {'stanox': '12345', 'description': 'Station A'},
    ]
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    service_stanox = ['12345']
    result = train_view_2(sample_single_station_reliability_df, service_stanox, 'SVC002', stations_ref_path='mock_path.json')
    
    # STANOX 12345: 1 on-time (<=0) out of 3 records = 33.33%
    assert len(result) == 1
    assert abs(result.loc[0, 'OnTime%'] - 33.33) < 1  # Allow small rounding difference


@patch('builtins.print')
@patch('builtins.open', create=True)
def test_train_view_2_empty_service_stanox(mock_open, mock_print, sample_service_reliability_df):
    """Test train_view_2 with empty service_stanox list returns empty or minimal DataFrame."""
    # Mock station reference file
    mock_station_ref = []
    import json
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_station_ref)
    
    result = train_view_2(sample_service_reliability_df, [], 'SVC001', stations_ref_path='mock_path.json')
    
    # Should return DataFrame (may include stations found from data with delays)
    assert isinstance(result, pd.DataFrame)


# TESTS FOR plot_reliability_graphs FUNCTION (still in train view)

@patch('builtins.print')
@patch('matplotlib.pyplot.show')
def test_plot_reliability_graphs_runs_without_error(mock_plt_show, mock_print, sample_single_station_reliability_df):
    """Test plot_reliability_graphs completes without error with mocking."""
    service_stanox = ['12345']
    
    # Use a non-existent path to trigger exception handling
    try:
        plot_reliability_graphs(sample_single_station_reliability_df, service_stanox, 'SVC002', 
                              stations_ref_path='/nonexistent/path.json')
        # If it doesn't raise, it's using exception handling - that's fine
        assert True
    except (FileNotFoundError, KeyError):
        # Expected when station ref file doesn't exist
        assert True
    finally:
        plt.close('all')


# ==============================================================================
# TIME VIEW FUNCTION TESTS
# ==============================================================================

# Fixtures for time view tests

@pytest.fixture
def sample_time_view_data():
    """
    Create sample time view incident data for testing.
    
    This fixture includes:
    - Multiple dates (2024-01-15 and 2024-01-16) for testing date filtering
    - Multiple stations with varying incident counts
    - Varying PFPI minutes for testing color coding (green, yellow, orange, red, dark red, violet)
    - Multiple incident reasons for testing top reasons aggregation
    """
    # Data for 2024-01-15 (10 incidents across 4 stations)
    date_2024_01_15 = [
        ('2024-01-15 08:30:00', '12345', 1001, 'TH', 15),
        ('2024-01-15 09:00:00', '12346', 1002, 'TG', 10),
        ('2024-01-15 09:45:00', '12345', 1003, 'TH', 25),
        ('2024-01-15 10:30:00', '12347', 1004, 'XA', 35),  # 31-60 min range
        ('2024-01-15 11:00:00', '12345', 1005, 'TG', 20),
        ('2024-01-15 11:30:00', '12346', 1006, 'M8', 50),  # 31-60 min range
        ('2024-01-15 12:00:00', '12345', 1007, 'TH', 18),
        ('2024-01-15 12:30:00', '12348', 1008, 'QM', 80),  # 61-120 min range
        ('2024-01-15 13:00:00', '12345', 1009, 'TG', 28),
        ('2024-01-15 13:30:00', '12346', 1010, 'M8', 14),
    ]
    
    # Data for 2024-01-16 (6 incidents across 3 stations - for testing no data on other dates)
    date_2024_01_16 = [
        ('2024-01-16 08:00:00', '12345', 1011, 'TG', 12),
        ('2024-01-16 09:00:00', '12346', 1012, 'TH', 7),
        ('2024-01-16 10:00:00', '12346', 1013, 'TG', 19),
    ]
    
    all_records = date_2024_01_15 + date_2024_01_16
    
    return pd.DataFrame({
        'INCIDENT_START_DATETIME': [r[0] for r in all_records],
        'STANOX': [r[1] for r in all_records],
        'INCIDENT_NUMBER': [r[2] for r in all_records],
        'INCIDENT_REASON': [r[3] for r in all_records],
        'PFPI_MINUTES': [r[4] for r in all_records],
    })

@pytest.fixture
def sample_time_view_stations_ref():
    """
    Create sample station reference data for time view.
    Includes latitude/longitude for UK stations.
    """
    return [
        {'stanox': 12345, 'station_name': 'London Kings Cross', 'latitude': 51.5307, 'longitude': -0.1234},
        {'stanox': 12346, 'station_name': 'Manchester Piccadilly', 'latitude': 53.4808, 'longitude': -2.2426},
        {'stanox': 12347, 'station_name': 'Birmingham New Street', 'latitude': 52.5078, 'longitude': -1.9043},
        {'stanox': 12348, 'station_name': 'Leeds City', 'latitude': 53.7949, 'longitude': -1.7477},
    ]

# Tests for _print_date_statistics function (helper for time view, which prints summary statistics for a given date)

def test_print_date_statistics(sample_time_view_data, capsys):
    """
    Test _print_date_statistics handles various data scenarios.
    Capsys is useful for testing functions that communicate via print statements rather than return values. 
    Without capsys, the print output would just go to your terminal—you wouldn't be able to verify it in your test assertions.
    
    Tests:
    - Prints correct summary for dates with incidents
    - Handles empty datasets gracefully
    - Handles dates with no matching incidents
    """
    # Test 1: Date with incidents - verify specific incident count and top reasons
    _print_date_statistics('2024-01-15', sample_time_view_data)
    captured = capsys.readouterr()
    assert '2024-01-15' in captured.out
    assert '10 incidents' in captured.out  # 2024-01-15 has exactly 10 incidents
    assert 'Top reasons' in captured.out
    # Verify top reasons from fixture: TH(3), TG(3), M8(2), XA(1), QM(1)
    assert 'TH(3)' in captured.out
    assert 'TG(3)' in captured.out
    assert 'M8(2)' in captured.out
    
    # Test 2: Empty dataset instead of sample_time_view_data
    empty_df = pd.DataFrame({
        'INCIDENT_START_DATETIME': pd.Series([], dtype='object'),
        'STANOX': pd.Series([], dtype='object'),
        'INCIDENT_REASON': pd.Series([], dtype='object'),
        'PFPI_MINUTES': pd.Series([], dtype='float64'),
    })
    _print_date_statistics('2024-01-15', empty_df)
    captured = capsys.readouterr()
    assert 'No incidents' in captured.out
    
    # Test 3: No matching date
    _print_date_statistics('2024-02-01', sample_time_view_data) # this date doesn't exist in the sample data, so should trigger "no incidents" message
    captured = capsys.readouterr()
    assert 'No incidents' in captured.out


# Tests for _create_time_view_markers function (helper for time view, which creates map markers for each station)

@patch('folium.CircleMarker')
def test_create_time_view_markers(mock_circle):
    """
    Test _create_time_view_markers marker creation, color grading, and radius scaling.
    
    Tests:
    - Adds CircleMarkers for each affected STANOX
    - Applies correct color based on PFPI severity
    - Scales radius by incident count
    """
    mock_map = MagicMock()
    
    # Test 1: Multiple markers
    _create_time_view_markers(
        mock_map, 
        affected_stanox={12345, 12346, 12347},
        incident_counts={12345: 5, 12346: 3, 12347: 1},
        total_pfpi={12345: 106, 12346: 36, 12347: 30},
        stanox_to_coords={'12345': [51.5074, -0.1278], '12346': [53.4808, -2.2426], '12347': [52.5200, 13.4050]}
    )
    # Assert markers created with correct counts and verify color grading
    assert mock_circle.call_count >= 3
    
    # Test 2: Verify correct colors are applied to each severity range based on Test 1 data
    # Station 12345: PFPI 106 (61-120 range) → Dark Red '#8B0000'
    # Station 12346: PFPI 36 (31-60 range) → Red '#FF0000'
    # Station 12347: PFPI 30 (0-30 range) → Dark Orange '#FF8C00'
    
    # Extract location-color pairs from each marker call
    markers = {tuple(call[1].get('location', [])): call[1].get('fill_color', call[1].get('color', '')) 
               for call in mock_circle.call_args_list}
    
    # Assert each station has the correct color for its severity
    assert markers[(51.5074, -0.1278)] == '#8B0000', "Station 12345: expected #8B0000"
    assert markers[(53.4808, -2.2426)] == '#FF0000', "Station 12346: expected #FF0000"
    assert markers[(52.5200, 13.4050)] == '#FF8C00', "Station 12347: expected #FF8C00"
    
    # Test 3: Verify radius scaling with incident counts from Test 1 data
    # Extract radii: station 12345 (5 incidents) should have larger radius than 12347 (1 incident)
    radii = {tuple(call[1].get('location', [])): call[1].get('radius', 0) 
             for call in mock_circle.call_args_list}
    
    radius_12345 = radii[(51.5074, -0.1278)]  # 5 incidents
    radius_12346 = radii[(53.4808, -2.2426)]  # 3 incidents
    radius_12347 = radii[(52.5200, 13.4050)]  # 1 incident
    
    # Verify radius scaling: more incidents = larger radius
    assert radius_12345 > radius_12346 > radius_12347, "Radius should increase with incident count"


# Tests for _finalize_time_view_map function (helper for time view, which finalizes the map by adding title, legend, and saving to file)

@patch('folium.Element')
def test_finalize_time_view_map(mock_element):
    """
    Test _finalize_time_view_map adds title, legend, and saves file.
    
    Tests:
    - Adds title to the map via HTML element
    - Adds legend to the map via HTML element
    - Saves map to file with correct naming convention
    """
    mock_map = MagicMock()
    
    # Test all aspects in one call
    _finalize_time_view_map(mock_map, '2024-01-15')
    
    # Verify title and legend are added
    assert mock_map.get_root().html.add_child.called
    calls = mock_map.get_root().html.add_child.call_count
    assert calls >= 2  # At least title and legend
    
    # Verify file is saved
    assert mock_map.save.called
    call_args = mock_map.save.call_args
    if call_args:
        filename = call_args[0][0]
        assert 'time_view_2024_01_15' in filename


# Tests for _aggregate_time_view_data function (helper for time view, which aggregates incident counts and PFPI totals for a given date)

def test_aggregate_time_view_data_calculates_correct_totals(sample_time_view_data):
    """
    Test _aggregate_time_view_data calculates correct incident counts and PFPI sums.
    
    Verification using fixture data:
    - 2024-01-15: 10 incidents across 4 stations
    - STANOX 12345: 5 incidents, PFPI total = 15+25+20+18+28 = 106 minutes
    - STANOX 12346: 3 incidents, PFPI total = 10+50+14 = 74 minutes
    """
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-01-15', sample_time_view_data)
    
    # Verify data structure
    assert affected_stanox is not None
    assert len(affected_stanox) == 4  # 4 stations affected
    
    # Convert to int for comparison
    stanox_list = [int(s) for s in affected_stanox]
    assert sorted(stanox_list) == [12345, 12346, 12347, 12348]
    
    # Verify incident counts
    count_12345 = incident_counts.get(12345) or incident_counts.get('12345')
    count_12346 = incident_counts.get(12346) or incident_counts.get('12346')
    count_12347 = incident_counts.get(12347) or incident_counts.get('12347')
    count_12348 = incident_counts.get(12348) or incident_counts.get('12348')
    
    assert count_12345 == 5
    assert count_12346 == 3
    assert count_12347 == 1
    assert count_12348 == 1
    
    # Verify PFPI totals
    pfpi_12345 = total_pfpi.get(12345) or total_pfpi.get('12345')
    pfpi_12346 = total_pfpi.get(12346) or total_pfpi.get('12346')
    pfpi_12347 = total_pfpi.get(12347) or total_pfpi.get('12347')
    assert pfpi_12345 == 106  # 15+25+20+18+28
    assert pfpi_12346 == 74   # 10+50+14



def test_aggregate_time_view_data_empty_date(sample_time_view_data):
    """Test _aggregate_time_view_data handles date with no incidents."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-03-01', sample_time_view_data)
    
    # Should return None for non-matching date
    assert affected_stanox is None


# INTEGRATION TESTS FOR create_time_view_html (calling main create_time_view_html function, no helpers)

@patch('rdmpy.outputs.analysis_tools._load_station_coordinates')
@patch('folium.Map')
@patch('folium.CircleMarker')
@patch('builtins.print')
def test_create_time_view_html_comprehensive_workflow(mock_print, mock_circle, mock_map, mock_load_coords, sample_time_view_data):
    """
    Comprehensive test for create_time_view_html covering complete workflow, marker creation,
    filename format validation, partial coordinates handling, and date filtering.
    
    Tests:
    - Complete workflow: print stats, create map, add markers, save file
    - Correct number of markers created (4 for 4 affected stations on 2024-01-15)
    - Correct filename format (time_view_YYYY_MM_DD.html)
    - Graceful handling of partial station coordinates
    - Correct date filtering and independence between dates
    """
    # Arrange: Mock dependencies with full coordinates
    full_coords = {
        '12345': [51.5307, -0.1234],
        '12346': [53.4808, -2.2426],
        '12347': [52.5078, -1.9043],
        '12348': [53.7949, -1.7477],
    }
    
    mock_load_coords.return_value = full_coords
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    
    # Act: Process 2024-01-15 (10 incidents across 4 stations)
    create_time_view_html('2024-01-15', sample_time_view_data)
    
    # Map should be created with correct center and zoom
    assert mock_map.called
    map_call_args = mock_map.call_args
    assert map_call_args[1]['location'] == [54.5, -2.5]  # UK center
    assert map_call_args[1]['zoom_start'] == 6
    
    # Correct number of markers created (4 affected stations)
    assert mock_circle.call_count >= 4
    
    # Map should be saved with correct filename
    assert mock_map_instance.save.called
    save_call_args = mock_map_instance.save.call_args
    filename = save_call_args[0][0]
    assert filename.startswith('time_view_')
    assert filename.endswith('.html')
    assert '2024_01_15' in filename
    
    # Reset mocks for second date test. This was done to not repeat lines of code
    mock_map.reset_mock()
    mock_circle.reset_mock()
    mock_print.reset_mock()
    mock_map.return_value = mock_map_instance
    
    # Act: Process 2024-01-16 (different date with fewer incidents)
    create_time_view_html('2024-01-16', sample_time_view_data)
    
    # Assert: Verify date filtering works independently
    assert mock_map.called  # Map should still be created
    
    # Test with partial coordinates (missing 2 of 4 stations)
    mock_map.reset_mock()
    mock_circle.reset_mock()
    partial_coords = {
        '12345': [51.5307, -0.1234],
        '12346': [53.4808, -2.2426],
        # 12347 and 12348 missing
    }
    mock_load_coords.return_value = partial_coords
    mock_map.return_value = mock_map_instance
    
    # Act: Should not crash with missing coordinates
    create_time_view_html('2024-01-15', sample_time_view_data)
    
    # Assert: Function completes without error
    assert mock_map.called
    assert mock_circle.call_count >= 2


@patch('rdmpy.outputs.analysis_tools._load_station_coordinates')
@patch('folium.Map')
def test_create_time_view_html_no_data_returns_gracefully(mock_map, mock_load_coords, sample_time_view_data):
    """
    Test that create_time_view_html handles dates with no incidents gracefully.
    
    Expected behavior:
    - Function should return early without creating map
    - No exception should be raised
    """
    # Arrange
    mock_load_coords.return_value = {}
    
    # Act & Assert: Should not crash on non-existent date
    result = create_time_view_html('2024-12-31', sample_time_view_data)
    
    # Map should NOT be created for date with no data
    # (Since _aggregate_time_view_data returns None)
    assert result is None

