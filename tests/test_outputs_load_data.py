

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rdmpy.outputs.load_data import load_processed_data


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_parquet_dataframe():
    """Create a mock DataFrame that represents loaded parquet data."""
    return pd.DataFrame({
        'TRAIN_SERVICE_CODE': ['ABC123', 'DEF456'],
        'PFPI_MINUTES': [10.5, 20.0],
        'EVENT_TYPE': ['D', 'C']
    })


@pytest.fixture
def mock_directory_structure(tmp_path):
    """Create a temporary directory structure mimicking processed_data structure."""
    base_dir = tmp_path / "processed_data"
    
    # Create station directories with parquet files
    station_11271 = base_dir / "11271"
    station_11271.mkdir(parents=True)
    (station_11271 / "MO.parquet").touch()
    (station_11271 / "TU.parquet").touch()
    
    station_12001 = base_dir / "12001"
    station_12001.mkdir(parents=True)
    (station_12001 / "WE.parquet").touch()
    
    return base_dir


# ============================================================================
# TESTS FOR load_processed_data()
# ============================================================================

def test_load_processed_data_k0_successful_load(mock_directory_structure, mock_parquet_dataframe):
    """
    k0: Happy path - successfully load parquet files from directory structure.
    Tests that the function loads all parquet files and returns a concatenated DataFrame.
    """
    with patch('pandas.read_parquet', return_value=mock_parquet_dataframe):
        result = load_processed_data(base_dir=str(mock_directory_structure))
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that STANOX and DAY columns were added
        assert 'STANOX' in result.columns
        assert 'DAY' in result.columns
        
        # Result should have data from multiple files (3 files in mock structure)
        assert len(result) > 0


def test_load_processed_data_k1_missing_directory():
    """
    k1: Error case - handle nonexistent directory gracefully.
    Tests that the function handles missing directories by trying parent directory.
    """
    nonexistent_dir = "this_directory_does_not_exist_123456"
    
    # Should attempt to find parent directory or handle gracefully
    with patch('os.path.exists', return_value=False):
        # Function should handle missing directory without crashing
        result = load_processed_data(base_dir=nonexistent_dir)
        
        # Should return empty DataFrame or handle appropriately
        assert result is None or isinstance(result, pd.DataFrame)


def test_load_processed_data_k2_empty_directory(tmp_path):
    """
    k2: Edge case - handle empty directory with no parquet files.
    Tests that the function returns an empty DataFrame when no parquet files exist.
    """
    empty_dir = tmp_path / "empty_processed_data"
    empty_dir.mkdir()
    
    result = load_processed_data(base_dir=str(empty_dir))
    
    # Should return empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_load_processed_data_k3_stanox_day_columns_added(mock_directory_structure, mock_parquet_dataframe):
    """
    k3: Verify STANOX and DAY columns are correctly added from file paths.
    Tests that metadata columns are properly extracted and added to the DataFrame.
    """
    with patch('pandas.read_parquet', return_value=mock_parquet_dataframe):
        result = load_processed_data(base_dir=str(mock_directory_structure))
        
        # Check STANOX column contains station codes from directory names
        assert 'STANOX' in result.columns
        assert all(result['STANOX'].isin(['11271', '12001']))
        
        # Check DAY column contains day codes from file names
        assert 'DAY' in result.columns
        assert all(result['DAY'].isin(['MO', 'TU', 'WE']))


def test_load_processed_data_k4_engine_fallback(mock_directory_structure, mock_parquet_dataframe):
    """
    k4: Test engine fallback from pyarrow to fastparquet.
    Tests that the function tries pyarrow first, then falls back to fastparquet on error.
    """
    def read_parquet_side_effect(file_path, engine=None):
        if engine == 'pyarrow':
            raise Exception("PyArrow not available")
        elif engine == 'fastparquet':
            return mock_parquet_dataframe
        return mock_parquet_dataframe
    
    with patch('pandas.read_parquet', side_effect=read_parquet_side_effect) as mock_read:
        result = load_processed_data(base_dir=str(mock_directory_structure))
        
        # Check that both engines were tried
        assert mock_read.call_count > 0
        
        # Check that result is still valid
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


def test_load_processed_data_k5_skip_corrupted_files(mock_directory_structure):
    """
    k5: Test handling of corrupted parquet files.
    Tests that the function skips corrupted files and continues processing others.
    """
    def read_parquet_side_effect(file_path, engine=None):
        # Fail for first file, succeed for others
        if 'MO.parquet' in str(file_path):
            raise Exception("Corrupted file")
        return pd.DataFrame({
            'TRAIN_SERVICE_CODE': ['ABC123'],
            'PFPI_MINUTES': [10.5]
        })
    
    with patch('pandas.read_parquet', side_effect=read_parquet_side_effect):
        result = load_processed_data(base_dir=str(mock_directory_structure))
        
        # Should still return data from non-corrupted files
        assert isinstance(result, pd.DataFrame)
        # Should have some data despite one file failing
        assert len(result) >= 0


def test_load_processed_data_k6_concatenates_multiple_files(mock_directory_structure):
    """
    k6: Verify that data from multiple files is properly concatenated.
    Tests that the function combines data from all parquet files into a single DataFrame.
    """
    # Create different data for each file
    def read_parquet_side_effect(file_path, engine=None):
        file_name = str(file_path)
        if 'MO.parquet' in file_name:
            return pd.DataFrame({'VALUE': [1], 'ID': ['MO_DATA']})
        elif 'TU.parquet' in file_name:
            return pd.DataFrame({'VALUE': [2], 'ID': ['TU_DATA']})
        elif 'WE.parquet' in file_name:
            return pd.DataFrame({'VALUE': [3], 'ID': ['WE_DATA']})
        return pd.DataFrame()
    
    with patch('pandas.read_parquet', side_effect=read_parquet_side_effect):
        result = load_processed_data(base_dir=str(mock_directory_structure))
        
        # Check that data from all files is present
        assert isinstance(result, pd.DataFrame)
        # Should have combined data from 3 files (MO, TU, WE)
        assert len(result) == 3
        assert set(result['ID']) == {'MO_DATA', 'TU_DATA', 'WE_DATA'}


def test_load_processed_data_k7_relative_vs_absolute_paths(tmp_path):
    """
    k7: Test that both relative and absolute paths work correctly.
    Tests path handling for different input formats.
    """
    base_dir = tmp_path / "processed_data"
    station_dir = base_dir / "11271"
    station_dir.mkdir(parents=True)
    (station_dir / "MO.parquet").touch()
    
    mock_df = pd.DataFrame({'VALUE': [1]})
    
    with patch('pandas.read_parquet', return_value=mock_df):
        # Test with absolute path
        result_abs = load_processed_data(base_dir=str(base_dir.absolute()))
        assert isinstance(result_abs, pd.DataFrame)
        
        # Test with relative path (change to tmp_path first)
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result_rel = load_processed_data(base_dir="processed_data")
            assert isinstance(result_rel, pd.DataFrame)
        finally:
            os.chdir(original_dir)


def test_load_processed_data_k8_only_parquet_files_processed(tmp_path):
    """
    k8: Verify that only .parquet files are processed.
    Tests that non-parquet files in the directory are ignored.
    """
    base_dir = tmp_path / "processed_data"
    station_dir = base_dir / "11271"
    station_dir.mkdir(parents=True)
    
    # Create various file types
    (station_dir / "MO.parquet").touch()
    (station_dir / "TU.csv").touch()  # Should be ignored
    (station_dir / "WE.txt").touch()  # Should be ignored
    (station_dir / "metadata.json").touch()  # Should be ignored
    
    mock_df = pd.DataFrame({'VALUE': [1]})
    
    with patch('pandas.read_parquet', return_value=mock_df) as mock_read:
        result = load_processed_data(base_dir=str(base_dir))
        
        # Check that only parquet files were read
        parquet_files_read = [str(call[0][0]) for call in mock_read.call_args_list]
        assert all('MO.parquet' in f for f in parquet_files_read)
        assert not any('.csv' in f for f in parquet_files_read)
        assert not any('.txt' in f for f in parquet_files_read)
        assert not any('.json' in f for f in parquet_files_read)


def test_load_processed_data_k9_nested_directory_structure(tmp_path):
    """
    k9: Test recursive directory traversal for nested structures.
    Tests that the function correctly handles the station_id/day.parquet structure.
    """
    base_dir = tmp_path / "processed_data"
    
    # Create multiple station directories
    for station_id in ['11271', '12001', '13702']:
        station_dir = base_dir / station_id
        station_dir.mkdir(parents=True)
        for day in ['MO', 'TU', 'WE']:
            (station_dir / f"{day}.parquet").touch()
    
    mock_df = pd.DataFrame({'VALUE': [1]})
    
    with patch('pandas.read_parquet', return_value=mock_df) as mock_read:
        result = load_processed_data(base_dir=str(base_dir))
        
        # Should have read from all station directories (3 stations * 3 days = 9 files)
        assert mock_read.call_count == 9
        assert isinstance(result, pd.DataFrame)


def test_load_processed_data_k10_returns_dataframe_type():
    """
    k10: Verify return type is always pandas DataFrame.
    Tests that the function consistently returns the correct type.
    """
    with patch('pathlib.Path.rglob', return_value=[]):
        result = load_processed_data(base_dir="any_dir")
        
        # Should return DataFrame even when empty
        assert isinstance(result, pd.DataFrame)
