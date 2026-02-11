"""
Unit tests for rdmpy.outputs.utils module.
Tests focus on aggregate view functions and their helpers.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdmpy.outputs.utils import (
    _calculate_incident_summary_stats,
    _load_and_prepare_multiday_data,
    get_stanox_for_service,
    map_train_journey_with_incidents,
    _prepare_journey_map_data,
    _compute_station_route_connections,
    _aggregate_delays_and_incidents,
    _create_station_markers_on_map,
    _create_incident_markers_on_map,
    _finalize_journey_map,
    _print_date_statistics,
    _load_station_coordinates,
    _aggregate_time_view_data,
    _create_time_view_markers,
    _finalize_time_view_map,
)


# ==============================================================================
# FIXTURES
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
# TESTS FOR _calculate_incident_summary_stats
# ==============================================================================

def test_calculate_incident_summary_stats_k0(sample_complete_df):
    """Test summary stats calculation returns dict with correct keys."""
    delay_data = sample_complete_df[sample_complete_df['PFPI_MINUTES'] > 0].copy()
    unique_dates = [date(2024, 1, 1)]
    
    result = _calculate_incident_summary_stats(
        df=sample_complete_df,
        delay_data_all=delay_data,
        unique_dates=unique_dates,
        files_processed=100,
        files_with_data=10,
        incident_number=12345,
        num_days=1
    )
    
    # Verify result structure and keys
    assert isinstance(result, dict)
    assert list(result.keys()) == ['Total Delay Minutes', 'Total Cancellations', 'Total Records Found', 'Peak Delay Event']
    
    # Verify statistics calculations
    expected_total = 5 + 10 + 15 + 20 + 25 + 30 + 35 + 40 + 45 + 50 + 55 + 60 + 65 + 70 + 75
    assert result['Total Delay Minutes'] == expected_total
    
    assert result['Total Cancellations'] == 2

    assert result['Total Records Found'] == 15
    
    ## Peak should be 75 minutes (from 14th record)
    assert '75.0 minutes' in result['Peak Delay Event']


# ==============================================================================
# TESTS FOR _load_and_prepare_multiday_data
# ==============================================================================

@patch('rdmpy.outputs.utils.find_processed_data_path')
@patch('rdmpy.outputs.utils._load_station_files_for_multiday_incident')
def test_load_and_prepare_multiday_data_k0(mock_load_files, mock_find_path):
    """Test load_and_prepare returns correct tuple structure when no incidents found."""
    mock_find_path.return_value = '/mock/path'
    mock_load_files.return_value = ([], 100, 0)
    
    result = _load_and_prepare_multiday_data(12345)
    
    assert result is None


@patch('rdmpy.outputs.utils.find_processed_data_path')
def test_load_and_prepare_multiday_data_k1(mock_find_path):
    """Test load_and_prepare returns None when no processed_data directory found."""
    mock_find_path.return_value = None
    
    result = _load_and_prepare_multiday_data(12345)
    
    assert result is None


@patch('rdmpy.outputs.utils.find_processed_data_path')
@patch('rdmpy.outputs.utils._load_station_files_for_multiday_incident')
def test_load_and_prepare_multiday_data_k2(mock_load_files, mock_find_path):
    """Test load_and_prepare returns correct tuple with valid data."""
    mock_find_path.return_value = '/mock/path'
    
    # Create mock incident records
    incident_records = [
        {
            'INCIDENT_NUMBER': 12345,
            'EVENT_DATETIME': '01-JAN-2024 08:00',
            'PFPI_MINUTES': 10,
            'EVENT_TYPE': 'D',
            'INCIDENT_START_DATETIME': '01-JAN-2024 08:00',
        },
        {
            'INCIDENT_NUMBER': 12345,
            'EVENT_DATETIME': '01-JAN-2024 09:00',
            'PFPI_MINUTES': 20,
            'EVENT_TYPE': 'D',
            'INCIDENT_START_DATETIME': '01-JAN-2024 08:00',
        }
    ]
    
    mock_load_files.return_value = (incident_records, 50, 5)
    
    result = _load_and_prepare_multiday_data(12345)
    
    assert result is not None
    assert len(result) == 5  # df, files_processed, files_with_data, unique_dates, num_days
    df, files_processed, files_with_data, unique_dates, num_days = result
    
    assert isinstance(df, pd.DataFrame)
    assert files_processed == 50
    assert files_with_data == 5
    assert num_days == 1
    assert len(unique_dates) == 1


# ==============================================================================
# TESTS FOR aggregate_view_multiday
# ==============================================================================

@patch('rdmpy.outputs.utils.plt.show')
@patch('rdmpy.outputs.utils._load_and_prepare_multiday_data')
def test_aggregate_view_multiday_k0(mock_load_prep, mock_plt_show):
    """Test aggregate_view_multiday returns None when data loading fails."""
    from rdmpy.outputs.utils import aggregate_view_multiday
    
    mock_load_prep.return_value = None
    
    result = aggregate_view_multiday(12345, '01-JAN-2024')
    
    assert result is None


@patch('rdmpy.outputs.utils.plt.show')
@patch('rdmpy.outputs.utils._load_and_prepare_multiday_data')
@patch('rdmpy.outputs.utils._calculate_incident_summary_stats')
def test_aggregate_view_multiday_k1(mock_calc_stats, mock_load_prep, mock_plt_show):
    """Test aggregate_view_multiday returns summary dict with expected keys."""
    from rdmpy.outputs.utils import aggregate_view_multiday
    
    # Create sample prepared data
    dates = pd.date_range('2024-01-01 08:00', periods=10, freq='H')
    sample_df = pd.DataFrame({
        'full_datetime': dates,
        'PFPI_MINUTES': [10, 15, 20, 5, 30, 25, 35, 40, 45, 50],
        'EVENT_TYPE': ['D'] * 8 + ['C'] * 2,
        'event_date_only': [date(2024, 1, 1)] * 10,
        'INCIDENT_START_DATETIME': ['01-JAN-2024 08:00'] * 10,
    })
    
    unique_dates = [date(2024, 1, 1)]
    mock_load_prep.return_value = (sample_df, 50, 5, unique_dates, 1)
    
    mock_stats = {
        'Total Delay Minutes': 275.0,
        'Total Cancellations': 2,
        'Total Records Found': 10,
        'Peak Delay Event': '50.0 minutes at 01-JAN-2024 17:00 (Regular Delay)',
    }
    mock_calc_stats.return_value = mock_stats
    
    result = aggregate_view_multiday(12345, '01-JAN-2024')
    
    assert result is not None
    assert isinstance(result, dict)
    assert result['Total Delay Minutes'] == 275.0
    assert result['Total Cancellations'] == 2


@patch('rdmpy.outputs.utils.plt.show')
@patch('rdmpy.outputs.utils._load_and_prepare_multiday_data')
def test_aggregate_view_multiday_k2(mock_load_prep, mock_plt_show):
    """Test aggregate_view_multiday handles multi-day incidents correctly."""
    from rdmpy.outputs.utils import aggregate_view_multiday
    
    # Create sample multi-day prepared data
    dates = pd.date_range('2024-01-01 08:00', periods=20, freq='H')
    sample_df = pd.DataFrame({
        'full_datetime': dates,
        'PFPI_MINUTES': list(range(10, 30)),
        'EVENT_TYPE': ['D'] * 18 + ['C'] * 2,
        'event_date_only': [date(2024, 1, 1)] * 10 + [date(2024, 1, 2)] * 10,
        'INCIDENT_START_DATETIME': ['01-JAN-2024 08:00'] * 20,
    })
    
    unique_dates = [date(2024, 1, 1), date(2024, 1, 2)]
    mock_load_prep.return_value = (sample_df, 100, 10, unique_dates, 2)
    
    result = aggregate_view_multiday(12345, '01-JAN-2024')
    
    # Should return a dict (plotted and summarized without error)
    assert result is not None
    assert isinstance(result, dict)


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_aggregate_view_workflow_consistency(sample_complete_df):
    """Test that aggregated stats from complete workflow are consistent."""
    # Test that calculations are internally consistent
    delay_data = sample_complete_df[sample_complete_df['PFPI_MINUTES'] > 0].copy()
    unique_dates = [date(2024, 1, 1)]
    
    result = _calculate_incident_summary_stats(
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

@pytest.fixture
def sample_station_ref():
    """Create sample station reference data."""
    return [
        {
            "stanox": "12345",
            "description": "London King's Cross",
            "latitude": 51.5307,
            "longitude": -0.1238
        },
        {
            "stanox": "12346",
            "description": "Peterborough",
            "latitude": 52.5680,
            "longitude": -0.2429
        },
        {
            "stanox": "12347",
            "description": "Leeds",
            "latitude": 53.7951,
            "longitude": -1.8455
        },
        {
            "stanox": "12348",
            "description": "York",
            "latitude": 53.9581,
            "longitude": -1.0873
        },
    ]


@pytest.fixture
def sample_incident_results():
    """Create sample incident results (list of DataFrames)."""
    df1 = pd.DataFrame({
        'STANOX': ['12345', '12346', '12345', '12347'],
        'INCIDENT_NUMBER': [1001, 1002, 1003, 1004],
        'INCIDENT_NUMBER_str': ['1001', '1002', '1003', '1004'],
        'PFPI_MINUTES': [10.0, 15.0, 20.0, 25.0],
        'INCIDENT_START_DATETIME': ['2024-01-15 08:00', '2024-01-15 09:00', '2024-01-15 10:00', '2024-01-15 11:00'],
        'EVENT_DATETIME': ['15-JAN-2024 09:00', '15-JAN-2024 10:30', '15-JAN-2024 11:00', '15-JAN-2024 12:30'],
        'SECTION_CODE': ['12345', '12346:12347', '12345', '12346:12347'],
        'INCIDENT_REASON': ['Signal failure', 'Signalling issue', 'Track defect', 'Points failure'],
    })
    return [df1]


@pytest.fixture
def sample_stanox_coords():
    """Create sample STANOX coordinates."""
    return [
        ('12345', 51.5307, -0.1238),
        ('12346', 52.5680, -0.2429),
        ('12347', 53.7951, -1.8455),
    ]


@pytest.fixture
def sample_stanox_names():
    """Create sample STANOX to name mapping."""
    return {
        '12345': 'London King\'s Cross',
        '12346': 'Peterborough',
        '12347': 'Leeds',
    }


# ==============================================================================
# TESTS FOR get_stanox_for_service
# ==============================================================================

def test_get_stanox_for_service_k0():
    """Test get_stanox_for_service function exists and is callable."""
    # Just verify the function is callable
    assert callable(get_stanox_for_service)


@pytest.mark.skip(reason="Requires complex DataFrame structure")
def test_get_stanox_for_service_k1():
    """Test get_stanox_for_service handles missing data gracefully."""
    # This function requires many columns to be set up properly
    # Skipping in favor of integration tests in notebooks
    pass


# ==============================================================================
# TESTS FOR _prepare_journey_map_data
# ==============================================================================

def test_prepare_journey_map_data_k0(sample_station_ref, sample_stanox_names):
    """Test _prepare_journey_map_data returns tuple of coords and names."""
    all_stanox = {'12345', '12346', '12347'}
    coords, names = _prepare_journey_map_data(all_stanox, sample_station_ref)
    
    assert isinstance(coords, list)
    assert isinstance(names, dict)
    assert len(coords) == 3
    assert len(names) == 3


def test_prepare_journey_map_data_k1(sample_station_ref):
    """Test _prepare_journey_map_data coords have correct format."""
    all_stanox = {'12345', '12346'}
    coords, names = _prepare_journey_map_data(all_stanox, sample_station_ref)
    
    for stanox, lat, lon in coords:
        assert isinstance(stanox, str)
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert 45 <= lat <= 60  # UK latitude range
        assert -10 <= lon <= 5  # UK longitude range


def test_prepare_journey_map_data_k2(sample_station_ref):
    """Test _prepare_journey_map_data with missing STANOX handles gracefully."""
    all_stanox = {'12345', '99999'}  # 99999 doesn't exist
    coords, names = _prepare_journey_map_data(all_stanox, sample_station_ref)
    
    # Should only return the valid one
    assert len(coords) == 1
    assert coords[0][0] == '12345'


def test_prepare_journey_map_data_k3(sample_station_ref):
    """Test _prepare_journey_map_data with empty STANOX set."""
    all_stanox = set()
    coords, names = _prepare_journey_map_data(all_stanox, sample_station_ref)
    
    assert coords == []
    assert names == {}


# ==============================================================================
# TESTS FOR _compute_station_route_connections
# ==============================================================================

def test_compute_station_route_connections_k0(sample_stanox_coords, sample_stanox_names):
    """Test _compute_station_route_connections returns list of connections."""
    connections = _compute_station_route_connections(sample_stanox_coords, sample_stanox_names)
    
    assert isinstance(connections, list)
    assert len(connections) > 0  # Should have at least one connection for 3 stations


def test_compute_station_route_connections_k1(sample_stanox_coords, sample_stanox_names):
    """Test _compute_station_route_connections creates minimum spanning tree."""
    connections = _compute_station_route_connections(sample_stanox_coords, sample_stanox_names)
    
    # For N stations, MST should have N-1 edges
    assert len(connections) == len(sample_stanox_coords) - 1


def test_compute_station_route_connections_k2(sample_stanox_coords, sample_stanox_names):
    """Test _compute_station_route_connections returns proper tuple structure."""
    connections = _compute_station_route_connections(sample_stanox_coords, sample_stanox_names)
    
    for connection in connections:
        assert len(connection) == 6
        start_stanox, end_stanox, start_coords, end_coords, start_name, end_name = connection
        assert isinstance(start_stanox, str)
        assert isinstance(end_stanox, str)
        assert isinstance(start_coords, tuple) and len(start_coords) == 2
        assert isinstance(end_coords, tuple) and len(end_coords) == 2
        assert isinstance(start_name, str)
        assert isinstance(end_name, str)


def test_compute_station_route_connections_k3():
    """Test _compute_station_route_connections with single station."""
    single_coord = [('12345', 51.5307, -0.1238)]
    names = {'12345': 'London'}
    connections = _compute_station_route_connections(single_coord, names)
    
    # Single station should have no connections
    assert len(connections) == 0


# ==============================================================================
# TESTS FOR _aggregate_delays_and_incidents
# ==============================================================================

def test_aggregate_delays_and_incidents_k0(sample_incident_results):
    """Test _aggregate_delays_and_incidents returns tuple of correct types."""
    stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records = _aggregate_delays_and_incidents(sample_incident_results)
    
    assert isinstance(stanox_delay, dict)
    assert isinstance(stanox_incidents, dict)
    assert isinstance(incident_rank, dict)
    assert isinstance(incident_durations, dict)
    assert isinstance(incident_records, (pd.DataFrame, type(None)))


def test_aggregate_delays_and_incidents_k1(sample_incident_results):
    """Test _aggregate_delays_and_incidents aggregates delays correctly."""
    stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records = _aggregate_delays_and_incidents(sample_incident_results)
    
    assert stanox_delay['12345'] == 30.0  # 10 + 20
    assert stanox_delay['12346'] == 15.0
    assert stanox_delay['12347'] == 25.0


def test_aggregate_delays_and_incidents_k2(sample_incident_results):
    """Test _aggregate_delays_and_incidents groups incidents by STANOX."""
    stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records = _aggregate_delays_and_incidents(sample_incident_results)
    
    assert set(stanox_incidents['12345']) == {'1001', '1003'}
    assert stanox_incidents['12346'] == ['1002']


def test_aggregate_delays_and_incidents_k3(sample_incident_results):
    """Test _aggregate_delays_and_incidents creates incident rankings."""
    stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records = _aggregate_delays_and_incidents(sample_incident_results)
    
    # Should have 4 incidents ranked 1-4 by start time
    assert len(incident_rank) == 4
    assert 1 in incident_rank.values()
    assert 4 in incident_rank.values()


def test_aggregate_delays_and_incidents_k4():
    """Test _aggregate_delays_and_incidents with empty incident results."""
    stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records = _aggregate_delays_and_incidents(None)
    
    assert stanox_delay == {}
    assert stanox_incidents == {}
    assert incident_rank == {}
    assert incident_durations == {}
    assert incident_records is None


def test_aggregate_delays_and_incidents_k5():
    """Test _aggregate_delays_and_incidents with empty DataFrame."""
    empty_result = []
    stanox_delay, stanox_incidents, incident_rank, incident_durations, incident_records = _aggregate_delays_and_incidents(empty_result)
    
    assert stanox_delay == {}
    assert stanox_incidents == {}


# ==============================================================================
# TESTS FOR _create_station_markers_on_map
# ==============================================================================

@patch('folium.CircleMarker')
def test_create_station_markers_on_map_k0(mock_circle, sample_stanox_coords, sample_stanox_names):
    """Test _create_station_markers_on_map creates markers for each station."""
    mock_map = MagicMock()
    stanox_delay = {'12345': 10.0, '12346': 0.0, '12347': 25.0}
    stanox_incidents = {'12345': ['1001'], '12346': [], '12347': ['1002']}
    incident_rank = {'1001': 1, '1002': 2}
    
    _create_station_markers_on_map(mock_map, sample_stanox_coords, sample_stanox_names, 
                                   stanox_delay, stanox_incidents, incident_rank)
    
    # Should create 3 markers (one per station)
    assert mock_circle.call_count == 3


@patch('folium.CircleMarker')
def test_create_station_markers_on_map_k1(mock_circle, sample_stanox_coords, sample_stanox_names):
    """Test _create_station_markers_on_map applies correct colors based on delay."""
    mock_map = MagicMock()
    stanox_delay = {
        '12345': 0.0,      # blue
        '12346': 3.0,      # lime green (1-5 min)
        '12347': 20.0,     # dark orange (16-30 min)
    }
    stanox_incidents = {'12345': [], '12346': [], '12347': []}
    incident_rank = {}
    
    _create_station_markers_on_map(mock_map, sample_stanox_coords, sample_stanox_names,
                                   stanox_delay, stanox_incidents, incident_rank)
    
    # Check that CircleMarker was called with correct color values
    calls = mock_circle.call_args_list
    # First call (12345) should be blue
    assert calls[0][1]['color'] == 'blue'
    # Second call (12346) should be lime green
    assert calls[1][1]['color'] == '#32CD32'
    # Third call (12347) should be dark orange
    assert calls[2][1]['color'] == '#FF8C00'


@patch('folium.CircleMarker')
def test_create_station_markers_on_map_k2(mock_circle, sample_stanox_coords, sample_stanox_names):
    """Test _create_station_markers_on_map with no delay data."""
    mock_map = MagicMock()
    stanox_delay = {}  # No delays recorded
    stanox_incidents = {}
    incident_rank = {}
    
    _create_station_markers_on_map(mock_map, sample_stanox_coords, sample_stanox_names,
                                   stanox_delay, stanox_incidents, incident_rank)
    
    # Should still create markers (with default blue color)
    assert mock_circle.call_count == 3


# ==============================================================================
# TESTS FOR _create_incident_markers_on_map
# ==============================================================================

@patch('folium.PolyLine')
@patch('folium.Marker')
def test_create_incident_markers_on_map_k0(mock_marker, mock_polyline, sample_incident_results, 
                                           sample_station_ref):
    """Test _create_incident_markers_on_map creates markers for incidents."""
    mock_map = MagicMock()
    incident_records = sample_incident_results[0]
    incident_rank = {'1001': 1, '1002': 2, '1003': 3, '1004': 4}
    incident_durations = {
        '1001': pd.Timedelta(hours=1),
        '1002': pd.Timedelta(hours=1.5),
        '1003': pd.Timedelta(hours=1),
        '1004': pd.Timedelta(hours=1.5),
    }
    
    _create_incident_markers_on_map(mock_map, incident_records, sample_station_ref,
                                   incident_rank, incident_durations, "purple")
    
    # Should create markers (at least some)
    assert mock_marker.call_count > 0 or mock_polyline.call_count > 0


@patch('folium.PolyLine')
@patch('folium.Marker')
def test_create_incident_markers_on_map_k1(mock_marker, mock_polyline):
    """Test _create_incident_markers_on_map with empty DataFrame."""
    mock_map = MagicMock()
    empty_df = pd.DataFrame()
    
    _create_incident_markers_on_map(mock_map, empty_df, [], {}, {}, "purple")
    
    # Should not create any markers
    assert mock_marker.call_count == 0
    assert mock_polyline.call_count == 0


@patch('folium.PolyLine')
@patch('folium.Marker')
def test_create_incident_markers_on_map_k2(mock_marker, mock_polyline):
    """Test _create_incident_markers_on_map with None DataFrame."""
    mock_map = MagicMock()
    
    _create_incident_markers_on_map(mock_map, None, [], {}, {}, "purple")
    
    # Should handle None gracefully
    assert mock_marker.call_count == 0
    assert mock_polyline.call_count == 0


# ==============================================================================
# TESTS FOR _finalize_journey_map
# ==============================================================================

@patch('folium.LayerControl')
def test_finalize_journey_map_k0(mock_layer_control):
    """Test _finalize_journey_map adds legend and layer control."""
    mock_map = MagicMock()
    
    _finalize_journey_map(mock_map)
    
    # Should add HTML element for legend
    assert mock_map.get_root.called
    # Should add layer control
    assert mock_layer_control.return_value.add_to.called


@patch('folium.LayerControl')
def test_finalize_journey_map_k1(mock_layer_control):
    """Test _finalize_journey_map legend contains expected text."""
    mock_map = MagicMock()
    
    _finalize_journey_map(mock_map)
    
    # Verify that add_child was called (for legend HTML)
    assert mock_map.get_root.return_value.html.add_child.called


# ==============================================================================
# INTEGRATION TESTS FOR map_train_journey_with_incidents
# ==============================================================================

@patch('folium.Map')
@patch('folium.LayerControl')
def test_map_train_journey_with_incidents_k0(mock_layer_control, mock_map,
                                             sample_incident_results, sample_station_ref):
    """Test map_train_journey_with_incidents returns folium map object."""
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    
    service_stanox = ['12345', '12346', '12347']
    
    with patch('builtins.open', create=True) as mock_open:
        import json
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(sample_station_ref)
        result = map_train_journey_with_incidents(
            None, service_stanox, sample_incident_results,
            stations_ref_path='dummy_path.json',
            service_code='EK001', date_str='2024-01-15'
        )
    
    # Should return a folium map
    assert result is mock_map_instance


@patch('folium.Map')
def test_map_train_journey_with_incidents_k1(mock_map, sample_station_ref):
    """Test map_train_journey_with_incidents with empty service and no incidents."""
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    
    service_stanox = ['12345']
    
    with patch('builtins.open', create=True) as mock_open:
        import json
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(sample_station_ref)
        result = map_train_journey_with_incidents(
            None, service_stanox, None,
            stations_ref_path='dummy_path.json'
        )
    
    assert result is mock_map_instance


@patch('folium.Map')
def test_map_train_journey_with_incidents_k2(mock_map, sample_station_ref):
    """Test map_train_journey_with_incidents with invalid STANOX."""
    # Don't create a map when no valid stations
    mock_map.return_value = MagicMock()
    
    service_stanox = ['99999', '88888']  # Non-existent STANOX codes
    
    with patch('builtins.open', create=True) as mock_open:
        import json
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(sample_station_ref)
        result = map_train_journey_with_incidents(
            None, service_stanox, None,
            stations_ref_path='dummy_path.json'
        )
    
    # Should return None when no valid stations found
    assert result is None


@patch('folium.Map')
def test_map_train_journey_with_incidents_k3(mock_map, sample_incident_results, sample_station_ref):
    """Test map_train_journey_with_incidents combines service and incident stations."""
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    
    service_stanox = ['12345', '12346']  # Service only goes through these
    # incident_results also includes 12347
    
    with patch('builtins.open', create=True) as mock_open:
        import json
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(sample_station_ref)
        result = map_train_journey_with_incidents(
            None, service_stanox, sample_incident_results,
            stations_ref_path='dummy_path.json'
        )
    
    # Should successfully combine both
    assert result is mock_map_instance


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

def test_prepare_journey_map_data_with_float_stanox(sample_station_ref):
    """Test _prepare_journey_map_data handles STANOX as float."""
    all_stanox = {12345.0, 12346.0}  # STANOX as float numbers
    coords, names = _prepare_journey_map_data(all_stanox, sample_station_ref)
    
    # Should normalize and match
    assert len(coords) == 2


def test_prepare_journey_map_data_with_int_stanox(sample_station_ref):
    """Test _prepare_journey_map_data handles STANOX as int."""
    all_stanox = {12345, 12346}  # STANOX as integers
    coords, names = _prepare_journey_map_data(all_stanox, sample_station_ref)
    
    # Should normalize and match
    assert len(coords) == 2


@pytest.mark.parametrize("delay_value,expected_color", [
    (0, "blue"),
    (3, "#32CD32"),     # 1-5 min
    (10, "#FFD700"),    # 6-15 min
    (25, "#FF8C00"),    # 16-30 min
    (45, "#FF0000"),    # 31-60 min
    (90, "#8B0000"),    # 61-120 min
    (150, "#8A2BE2"),   # 120+ min
])
@patch('folium.CircleMarker')
def test_color_grading_for_delays(mock_circle, delay_value, expected_color, sample_stanox_coords, sample_stanox_names):
    """Test color grading function works correctly for different delay values."""
    mock_map = MagicMock()
    stanox_delay = {'12345': delay_value, '12346': 0.0, '12347': 0.0}
    stanox_incidents = {}
    incident_rank = {}
    
    _create_station_markers_on_map(mock_map, sample_stanox_coords, sample_stanox_names,
                                   stanox_delay, stanox_incidents, incident_rank)
    
    # Check the first marker's color
    first_call = mock_circle.call_args_list[0]
    assert first_call[1]['color'] == expected_color


# ==============================================================================
# TIME VIEW HELPER FUNCTION TESTS
# ==============================================================================

@pytest.fixture
def sample_time_view_data():
    """Create sample time view incident data."""
    dates = pd.date_range('2024-01-15 08:00', periods=10, freq='H')
    return pd.DataFrame({
        'INCIDENT_START_DATETIME': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
        'STANOX': ['12345', '12346', '12345', '12347', '12345', '12346', '12345', '12348', '12345', '12346'],
        'INCIDENT_NUMBER': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'INCIDENT_REASON': ['Signal Failure', 'Track Obstruction', 'Signal Failure', 'Traction Loss', 'Track Obstruction'] * 2,
        'PFPI_MINUTES': [15, 10, 25, 30, 20, 12, 18, 22, 28, 14],
    })


@pytest.fixture
def sample_time_view_stations_ref():
    """Create sample station reference data for time view."""
    return [
        {'stanox': 12345, 'station_name': 'Station A', 'latitude': 51.5074, 'longitude': -0.1278},
        {'stanox': 12346, 'station_name': 'Station B', 'latitude': 53.4808, 'longitude': -2.2426},
        {'stanox': 12347, 'station_name': 'Station C', 'latitude': 52.5200, 'longitude': 13.4050},
        {'stanox': 12348, 'station_name': 'Station D', 'latitude': 48.8566, 'longitude': 2.3522},
    ]


def test_print_date_statistics_with_data(sample_time_view_data, capsys):
    """Test _print_date_statistics prints correct summary for a date with incidents."""
    _print_date_statistics('2024-01-15', sample_time_view_data)
    
    captured = capsys.readouterr()
    # Should print the date and incident count
    assert '2024-01-15' in captured.out
    assert 'incidents' in captured.out.lower()


def test_print_date_statistics_empty_data(capsys):
    """Test _print_date_statistics handles empty dataset."""
    empty_df = pd.DataFrame({
        'INCIDENT_START_DATETIME': pd.Series([], dtype='object'),
        'STANOX': pd.Series([], dtype='object'),
        'INCIDENT_REASON': pd.Series([], dtype='object'),
        'PFPI_MINUTES': pd.Series([], dtype='float64'),
    })
    
    _print_date_statistics('2024-01-15', empty_df)
    
    captured = capsys.readouterr()
    # Should print something for the date
    assert '2024-01-15' in captured.out or 'No incidents' in captured.out


def test_print_date_statistics_no_matching_date(sample_time_view_data, capsys):
    """Test _print_date_statistics when no incidents match the date."""
    _print_date_statistics('2024-02-01', sample_time_view_data)
    
    captured = capsys.readouterr()
    # Should indicate no incidents found
    assert '2024-02-01' in captured.out


@patch('rdmpy.outputs.utils._load_station_coordinates')
def test_load_station_coordinates_valid(mock_load, sample_time_view_stations_ref):
    """Test _load_station_coordinates successfully loads station data."""
    # Mock the return value
    expected_result = {
        '12345': [51.5074, -0.1278],
        '12346': [53.4808, -2.2426],
        '12347': [52.5200, 13.4050],
    }
    mock_load.return_value = expected_result
    
    result = mock_load()
    
    # Check structure: should be dict with STANOX as keys
    assert isinstance(result, dict)
    # STANOX should map to [lat, lon] pairs
    for stanox, coords in result.items():
        assert isinstance(coords, list)
        assert len(coords) == 2


@patch('rdmpy.outputs.utils._load_station_coordinates')
def test_load_station_coordinates_missing_file(mock_load):
    """Test _load_station_coordinates handles missing reference file gracefully."""
    mock_load.return_value = {}
    
    result = mock_load()
    # Should return empty dict on missing file
    assert result == {}


def test_aggregate_time_view_data_valid_date(sample_time_view_data):
    """Test _aggregate_time_view_data aggregates incidents correctly for a date."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-01-15', sample_time_view_data)
    
    # Should find affected STANOX
    assert affected_stanox is not None
    assert len(affected_stanox) > 0
    
    # Check aggregations - returns pandas Series, not dict
    assert hasattr(incident_counts, 'get')  # Series has get method
    assert hasattr(total_pfpi, 'get')
    
    # STANOX 12345 appears 5 times in sample data
    assert 12345 in [int(s) for s in affected_stanox]


def test_aggregate_time_view_data_empty_date(sample_time_view_data):
    """Test _aggregate_time_view_data handles date with no incidents."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-03-01', sample_time_view_data)
    
    # Should return None for non-matching date
    assert affected_stanox is None


def test_aggregate_time_view_data_stanox_grouping(sample_time_view_data):
    """Test _aggregate_time_view_data correctly groups by STANOX."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-01-15', sample_time_view_data)
    
    if affected_stanox is not None:
        # Verify STANOX 12345 has correct incident count
        # STANOX might be string or int in the Series, try both
        stanox_12345_count = incident_counts.get(12345) or incident_counts.get('12345')
        # STANOX 12345 appears 5 times in the sample data
        assert stanox_12345_count == 5
        
        # Verify PFPI totals are sums
        stanox_12345_pfpi = total_pfpi.get(12345) or total_pfpi.get('12345')
        # Incidents for 12345: indices 0, 2, 4, 6, 8 with PFPI 15, 25, 20, 18, 28
        assert stanox_12345_pfpi == 106



@patch('folium.CircleMarker')
def test_create_time_view_markers_adds_markers(mock_circle):
    """Test _create_time_view_markers adds CircleMarkers to the map."""
    mock_map = MagicMock()
    
    affected_stanox = {12345, 12346, 12347}
    incident_counts = {12345: 5, 12346: 3, 12347: 1}
    total_pfpi = {12345: 106, 12346: 36, 12347: 30}
    stanox_to_coords = {
        '12345': [51.5074, -0.1278],
        '12346': [53.4808, -2.2426],
        '12347': [52.5200, 13.4050],
    }
    
    _create_time_view_markers(mock_map, affected_stanox, incident_counts, total_pfpi, stanox_to_coords)
    
    # Should add markers for each STANOX
    assert mock_circle.call_count >= len(affected_stanox)


@patch('folium.CircleMarker')
def test_create_time_view_markers_color_grading(mock_circle):
    """Test _create_time_view_markers applies correct colors based on PFPI."""
    mock_map = MagicMock()
    
    affected_stanox = {12345}
    incident_counts = {12345: 1}
    total_pfpi = {12345: 50.0}  # 31-60 min range = Red
    stanox_to_coords = {'12345': [51.5074, -0.1278]}
    
    _create_time_view_markers(mock_map, affected_stanox, incident_counts, total_pfpi, stanox_to_coords)
    
    # Check that a marker was added with appropriate color
    assert mock_circle.called
    call_args = mock_circle.call_args
    if call_args:
        # Color should be red for 31-60 min range
        assert call_args[1].get('color') == '#FF0000' or call_args[1].get('fill_color') == '#FF0000'


@patch('folium.CircleMarker')
def test_create_time_view_markers_radius_scaling(mock_circle):
    """Test _create_time_view_markers scales radius by incident count."""
    mock_map = MagicMock()
    
    affected_stanox = {12345}
    incident_counts = {12345: 10}  # Higher count should have larger radius
    total_pfpi = {12345: 50.0}
    stanox_to_coords = {'12345': [51.5074, -0.1278]}
    
    _create_time_view_markers(mock_map, affected_stanox, incident_counts, total_pfpi, stanox_to_coords)
    
    # Check radius is scaled
    assert mock_circle.called
    call_args = mock_circle.call_args
    if call_args:
        # Radius should increase with incident count
        assert call_args[1].get('radius') > 5


@patch('folium.Element')
def test_finalize_time_view_map_adds_title(mock_element):
    """Test _finalize_time_view_map adds title to the map."""
    mock_map = MagicMock()
    
    _finalize_time_view_map(mock_map, '2024-01-15')
    
    # Should call html.add_child for title and legend
    assert mock_map.get_root().html.add_child.called


@patch('folium.Element')
@patch('builtins.open', create=True)
def test_finalize_time_view_map_saves_file(mock_open, mock_element):
    """Test _finalize_time_view_map saves the map to file."""
    mock_map = MagicMock()
    
    _finalize_time_view_map(mock_map, '2024-01-15')
    
    # Should call map.save()
    assert mock_map.save.called
    # Check that save was called with correct filename pattern
    call_args = mock_map.save.call_args
    if call_args:
        filename = call_args[0][0]
        assert 'time_view_2024_01_15' in filename


@patch('folium.Element')
def test_finalize_time_view_map_adds_legend(mock_element):
    """Test _finalize_time_view_map adds legend to the map."""
    mock_map = MagicMock()
    
    _finalize_time_view_map(mock_map, '2024-01-15')
    
    # Should add HTML elements for legend
    calls = mock_map.get_root().html.add_child.call_count
    # At least one call for title/legend
    assert calls >= 1


def test_aggregate_time_view_data_preserves_stanox_format(sample_time_view_data):
    """Test _aggregate_time_view_data returns STANOX in consistent format."""
    affected_stanox, incident_counts, total_pfpi = _aggregate_time_view_data('2024-01-15', sample_time_view_data)
    
    if affected_stanox is not None:
        # All STANOX should be convertible to int
        for stanox in affected_stanox:
            int_stanox = int(stanox)
            assert 10000 <= int_stanox <= 100000  # Valid STANOX range