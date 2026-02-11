import pytest
from rdmpy.utils import (
    get_english_day_types_from_schedule, 
    process_schedule, 
    process_delays_optimized, 
    adjust_schedule_timeline,
    extract_day_of_week_from_delay,
    schedule_runs_on_day,
    # New validation functions
    is_valid_schedule_entry,
    validate_schedule_locations,
    is_valid_location_entry,
    has_time_information,
    extract_location_time,
    get_train_service_code,
    # New transformation functions
    find_location_by_tiploc,
    find_origin_location,
    find_destination_location,
    determine_station_role,
    build_train_record,
    extract_time_components_from_delays,
    expand_schedule_by_days,
    # New delay matching functions
    build_unmatched_entry,
    apply_delays_to_matches,
    filter_result_columns
)

import pandas as pd
import numpy as np
import json

# testing get_english_day_types_from_schedule

def test_get_english_day_types_from_schedule_k0():
    """Test conversion of all days running to day types"""
    entry = {"schedule_days_runs": "1111111"}

    result = get_english_day_types_from_schedule(entry)

    assert result == ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]

def test_get_english_day_types_from_schedule_k1():
    """Test conversion of weekdays only to day types"""
    entry = {"schedule_days_runs": "1111100"}

    result = get_english_day_types_from_schedule(entry)

    assert result == ["MO", "TU", "WE", "TH", "FR"]

def test_get_english_day_types_from_schedule_k2():
    """Test conversion of weekends only to day types"""
    entry = {"schedule_days_runs": "0000011"}

    result = get_english_day_types_from_schedule(entry)

    assert result == ["SA", "SU"]


# testing extract_day_of_week_from_delay

def test_extract_day_of_week_from_delay_k0():
    """Test extraction of day of week from valid datetime format"""
    delay_entry = {
        "PLANNED_ORIGIN_WTT_DATETIME": "01-JAN-2024 08:00"
    }
    
    result = extract_day_of_week_from_delay(delay_entry)
    
    # 01-JAN-2024 is a Monday
    assert result == "MO"


def test_extract_day_of_week_from_delay_k1():
    """Test handling of invalid datetime format"""
    delay_entry = {
        "PLANNED_ORIGIN_WTT_DATETIME": "invalid-date"
    }
    
    result = extract_day_of_week_from_delay(delay_entry)
    
    # Should return None for invalid format
    assert result is None


# testing schedule_runs_on_day

def test_schedule_runs_on_day_k0():
    """Test that function correctly identifies if schedule runs on target day"""
    schedule_entry = {
        "ENGLISH_DAY_TYPE": ["MO", "TU", "WE", "TH", "FR"]
    }
    
    result = schedule_runs_on_day(schedule_entry, "MO")
    
    assert result is True


def test_schedule_runs_on_day_k1():
    """Test that function correctly identifies when schedule does not run on target day"""
    schedule_entry = {
        "ENGLISH_DAY_TYPE": ["MO", "TU", "WE", "TH", "FR"]
    }
    
    result = schedule_runs_on_day(schedule_entry, "SA")
    
    assert result is False

    

# testing process_schedule

def test_process_schedule_k0():
    """Test basic schedule processing with origin station and valid departure time"""
    st_code = "31510"
    tiploc = "MNCRVIC"

    schedule_data_loaded = [
        {
            "schedule_segment": {
                "CIF_train_service_code": "T123",
                "schedule_location": [
                    {"tiploc_code": "MNCRVIC", "location_type": "L0", "departure": "0815"},
                ]
            },
            "schedule_days_runs": "1111100"
        }
    ]

    stanox_ref = {
        "31510": {
            "dft_category": "B",
        }
    }

    tiploc_to_stanox = {
        "MNCRVIC": "31510",
    }

    result = process_schedule(
        st_code=st_code,
        schedule_data_loaded=schedule_data_loaded,
        stanox_ref=stanox_ref,
        tiploc_to_stanox=tiploc_to_stanox,
        tiploc=tiploc,
        train_count=1
    )

    assert len(result) == 1
    train = result[0]

    assert train["TRAIN_SERVICE_CODE"] == "T123"
    assert train["STATION_ROLE"] == "Origin"
    assert train["ENGLISH_DAY_TYPE"] == ["MO", "TU", "WE", "TH", "FR"]


def test_process_schedule_k1():
    """Test that trains not stopping at target station are filtered out"""
    st_code = "12345"

    schedule_data_loaded = [
        {
            "schedule_segment": {
                "CIF_train_service_code": "T999",
                "schedule_location": [
                    {"tiploc_code": "XYZ", "departure": "1015"}
                ]
            },
            "schedule_days_runs": "1111111"
        }
    ]

    result = process_schedule(
        st_code=st_code,
        schedule_data_loaded=schedule_data_loaded,
        stanox_ref={"12345": {}},
        tiploc_to_stanox={"XYZ": "88888"},
        tiploc="ABC",
        train_count=1
    )

    assert result == []

def test_process_schedule_k2():
    """Test graceful handling of malformed schedule entries"""
    st_code = "12345"
    tiploc = "ABC"

    schedule_data_loaded = [
        {},  # completely broken
        {
            "schedule_segment": {
                "schedule_location": "not a list"
            }
        },
        {
            "schedule_segment": {
                "CIF_train_service_code": "T777",
                "schedule_location": [
                    {"tiploc_code": "ABC", "arrival": "120000"},
                    {"tiploc_code": "ZZZ", "location_type": "LT", "arrival": "130000"}
                ]
            },
            "schedule_days_runs": "1000000"
        }
    ]

    result = process_schedule(
        st_code=st_code,
        schedule_data_loaded=schedule_data_loaded,
        stanox_ref={"12345": {}},
        tiploc_to_stanox={"ABC": "12345", "ZZZ": "54321"},
        tiploc=tiploc,
        train_count=1
    )

    assert len(result) == 1
    assert result[0]["TRAIN_SERVICE_CODE"] == "T777"
    assert result[0]["ENGLISH_DAY_TYPE"] == ["MO"]


def test_process_schedule_k3():
    """Test that destination station role is correctly identified"""
    st_code = "54321"
    tiploc = "DESTTN"

    schedule_data_loaded = [
        {
            "schedule_segment": {
                "CIF_train_service_code": "T456",
                "schedule_location": [
                    {"tiploc_code": "START", "location_type": "LO", "departure": "0900"},
                    {"tiploc_code": "DESTTN", "location_type": "LT", "arrival": "1000"}
                ]
            },
            "schedule_days_runs": "1111111"
        }
    ]

    result = process_schedule(
        st_code=st_code,
        schedule_data_loaded=schedule_data_loaded,
        stanox_ref={"54321": {}},
        tiploc_to_stanox={"START": "11111", "DESTTN": "54321"},
        tiploc=tiploc,
        train_count=1
    )

    assert len(result) == 1
    train = result[0]
    assert train["STATION_ROLE"] == "Destination"
    assert train["PLANNED_CALLS"] == "1000"


def test_process_schedule_k4():
    """Test that intermediate station role is correctly identified"""
    st_code = "22222"
    tiploc = "MIDDLE"

    schedule_data_loaded = [
        {
            "schedule_segment": {
                "CIF_train_service_code": "T789",
                "schedule_location": [
                    {"tiploc_code": "START", "location_type": "LO", "departure": "0800"},
                    {"tiploc_code": "MIDDLE", "departure": "0900", "arrival": "0855"},
                    {"tiploc_code": "END", "location_type": "LT", "arrival": "1000"}
                ]
            },
            "schedule_days_runs": "1111100"
        }
    ]

    result = process_schedule(
        st_code=st_code,
        schedule_data_loaded=schedule_data_loaded,
        stanox_ref={"22222": {}},
        tiploc_to_stanox={"START": "11111", "MIDDLE": "22222", "END": "33333"},
        tiploc=tiploc,
        train_count=1
    )

    assert len(result) == 1
    train = result[0]
    assert train["STATION_ROLE"] == "Intermediate"


def test_process_schedule_k5():
    """Test processing multiple trains in one schedule batch"""
    st_code = "40000"
    tiploc = "MULTI"

    schedule_data_loaded = [
        {
            "schedule_segment": {
                "CIF_train_service_code": "T111",
                "schedule_location": [
                    {"tiploc_code": "MULTI", "location_type": "L0", "departure": "0700"}
                ]
            },
            "schedule_days_runs": "1111111"
        },
        {
            "schedule_segment": {
                "CIF_train_service_code": "T222",
                "schedule_location": [
                    {"tiploc_code": "MULTI", "location_type": "L0", "departure": "0730"}
                ]
            },
            "schedule_days_runs": "1111111"
        },
        {
            "schedule_segment": {
                "CIF_train_service_code": "T333",
                "schedule_location": [
                    {"tiploc_code": "OTHER", "location_type": "L0", "departure": "0800"}
                ]
            },
            "schedule_days_runs": "1111111"
        }
    ]

    result = process_schedule(
        st_code=st_code,
        schedule_data_loaded=schedule_data_loaded,
        stanox_ref={"40000": {}},
        tiploc_to_stanox={"MULTI": "40000", "OTHER": "50000"},
        tiploc=tiploc,
        train_count=3
    )

    assert len(result) == 2
    service_codes = {train["TRAIN_SERVICE_CODE"] for train in result}
    assert service_codes == {"T111", "T222"}


def test_process_schedule_k6():
    """Test that arrival time is used when departure time is not available"""
    st_code = "60000"
    tiploc = "ARRVAL"

    schedule_data_loaded = [
        {
            "schedule_segment": {
                "CIF_train_service_code": "T555",
                "schedule_location": [
                    {"tiploc_code": "START", "location_type": "L0", "departure": "1000"},
                    {"tiploc_code": "ARRVAL", "arrival": "1100"}  # No departure, only arrival
                ]
            },
            "schedule_days_runs": "1111100"
        }
    ]

    result = process_schedule(
        st_code=st_code,
        schedule_data_loaded=schedule_data_loaded,
        stanox_ref={"60000": {}},
        tiploc_to_stanox={"START": "11111", "ARRVAL": "60000"},
        tiploc=tiploc,
        train_count=1
    )

    assert len(result) == 1
    train = result[0]
    assert train["PLANNED_CALLS"] == "1100"


def test_process_schedule_k7():
    """Test that train is skipped when both departure and arrival times are missing"""
    st_code = "70000"
    tiploc = "NOTIME"

    schedule_data_loaded = [
        {
            "schedule_segment": {
                "CIF_train_service_code": "T666",
                "schedule_location": [
                    {"tiploc_code": "START", "location_type": "L0", "departure": "1000"},
                    {"tiploc_code": "NOTIME"}  # No departure or arrival
                ]
            },
            "schedule_days_runs": "1111111"
        }
    ]

    result = process_schedule(
        st_code=st_code,
        schedule_data_loaded=schedule_data_loaded,
        stanox_ref={"70000": {}},
        tiploc_to_stanox={"START": "11111", "NOTIME": "70000"},
        tiploc=tiploc,
        train_count=1
    )

    assert result == []


# testing process_delays_optimized

def test_process_delays_optimized_k0():
    """Test that delays are correctly filtered by STANOX code"""
    st_code = "12345"
    
    # Create sample incident data for multiple periods
    incident_data_loaded = {
        "period1": pd.DataFrame({
            "START_STANOX": [12345],
            "END_STANOX": [54321],
            "INCIDENT_REASON": ["Fault"],
            "ATTRIBUTION_STATUS": ["status1"],
            "INCIDENT_EQUIPMENT": ["equip1"],
            "APPLICABLE_TIMETABLE_FLAG": ["flag1"],
            "TRACTION_TYPE": ["traction1"],
            "TRAILING_LOAD": ["load1"],
            "EVENT_TYPE": ["A"]
        })
    }
    
    result = process_delays_optimized(
        incident_data_loaded=incident_data_loaded,
        st_code=st_code
    )
    
    # Should have one period
    assert "period1" in result
    # Should have 1 row (START_STANOX=12345 matches st_code)
    assert len(result["period1"]) == 1
    # Verify columns are removed
    removed_cols = ["ATTRIBUTION_STATUS", "INCIDENT_EQUIPMENT", "APPLICABLE_TIMETABLE_FLAG", "TRACTION_TYPE", "TRAILING_LOAD"]
    for col in removed_cols:
        assert col not in result["period1"].columns
    # Verify relevant columns remain
    assert "INCIDENT_REASON" in result["period1"].columns
    assert "EVENT_TYPE" in result["period1"].columns


# testing adjust_schedule_timeline

def test_adjust_schedule_timeline_k0():
    """Test that schedule entries without delays remain unchanged"""
    processed_schedule = [
        {
            "TRAIN_SERVICE_CODE": "T001",
            "PLANNED_ORIGIN_LOCATION_CODE": "11111",
            "PLANNED_ORIGIN_GBTT_DATETIME": "08:00",
            "PLANNED_DEST_LOCATION_CODE": "22222",
            "PLANNED_DEST_GBTT_DATETIME": "09:00",
            "PLANNED_CALLS": "0800",
            "ACTUAL_CALLS": None,
            "PFPI_MINUTES": 0.0,
            "INCIDENT_REASON": None,
            "INCIDENT_NUMBER": None,
            "EVENT_TYPE": None,
            "SECTION_CODE": None,
            "DELAY_DAY": None,
            "EVENT_DATETIME": None,
            "INCIDENT_START_DATETIME": None,
            "ENGLISH_DAY_TYPE": ["MO"],
            "DFT_CATEGORY": "A",
            "PLATFORM_COUNT": 2,
            "STATION_ROLE": "Origin"
        }
    ]
    
    # Empty delays - no matching delays
    processed_delays = []
    
    result = adjust_schedule_timeline(processed_schedule, processed_delays)
    
    assert len(result) == 1
    train = result[0]
    # Verify schedule entry is unchanged
    assert train["TRAIN_SERVICE_CODE"] == "T001"
    assert train["PFPI_MINUTES"] == 0.0
    assert train["PLANNED_CALLS"] == "0800"
    assert train["INCIDENT_REASON"] is None


def test_adjust_schedule_timeline_k1():
    """Test that function returns processed_schedule when delays are empty"""
    processed_schedule = [
        {
            "TRAIN_SERVICE_CODE": "T002",
            "PLANNED_ORIGIN_LOCATION_CODE": "11111",
            "PLANNED_ORIGIN_GBTT_DATETIME": "10:00",
            "PLANNED_DEST_LOCATION_CODE": "22222",
            "PLANNED_DEST_GBTT_DATETIME": "11:00",
            "PLANNED_CALLS": "1000",
            "ACTUAL_CALLS": None,
            "PFPI_MINUTES": 0.0,
            "INCIDENT_REASON": None,
            "INCIDENT_NUMBER": None,
            "EVENT_TYPE": None,
            "SECTION_CODE": None,
            "DELAY_DAY": None,
            "EVENT_DATETIME": None,
            "INCIDENT_START_DATETIME": None,
            "ENGLISH_DAY_TYPE": ["MO"],
            "DFT_CATEGORY": "A",
            "PLATFORM_COUNT": 2,
            "STATION_ROLE": "Origin"
        },
        {
            "TRAIN_SERVICE_CODE": "T003",
            "PLANNED_ORIGIN_LOCATION_CODE": "11111",
            "PLANNED_ORIGIN_GBTT_DATETIME": "11:00",
            "PLANNED_DEST_LOCATION_CODE": "22222",
            "PLANNED_DEST_GBTT_DATETIME": "12:00",
            "PLANNED_CALLS": "1100",
            "ACTUAL_CALLS": None,
            "PFPI_MINUTES": 0.0,
            "INCIDENT_REASON": None,
            "INCIDENT_NUMBER": None,
            "EVENT_TYPE": None,
            "SECTION_CODE": None,
            "DELAY_DAY": None,
            "EVENT_DATETIME": None,
            "INCIDENT_START_DATETIME": None,
            "ENGLISH_DAY_TYPE": ["TU"],
            "DFT_CATEGORY": "A",
            "PLATFORM_COUNT": 2,
            "STATION_ROLE": "Origin"
        }
    ]
    
    # No delays
    processed_delays = []
    
    result = adjust_schedule_timeline(processed_schedule, processed_delays)
    
    # All schedule entries should be returned unchanged
    assert len(result) == 2
    assert all(train["PFPI_MINUTES"] == 0.0 for train in result)
    assert result[0]["TRAIN_SERVICE_CODE"] == "T002"
    assert result[1]["TRAIN_SERVICE_CODE"] == "T003"


def test_adjust_schedule_timeline_k2():
    """Test that function handles valid data structures correctly"""
    processed_schedule = [
        {
            "TRAIN_SERVICE_CODE": "T004",
            "PLANNED_ORIGIN_LOCATION_CODE": "11111",
            "PLANNED_ORIGIN_GBTT_DATETIME": "12:00",
            "PLANNED_DEST_LOCATION_CODE": "22222",
            "PLANNED_DEST_GBTT_DATETIME": "13:00",
            "PLANNED_CALLS": "1200",
            "ACTUAL_CALLS": None,
            "PFPI_MINUTES": 0.0,
            "INCIDENT_REASON": None,
            "INCIDENT_NUMBER": None,
            "EVENT_TYPE": None,
            "SECTION_CODE": None,
            "DELAY_DAY": None,
            "EVENT_DATETIME": None,
            "INCIDENT_START_DATETIME": None,
            "ENGLISH_DAY_TYPE": ["MO", "TU"],
            "DFT_CATEGORY": "B",
            "PLATFORM_COUNT": 3,
            "STATION_ROLE": "Intermediate"
        }
    ]
    
    # Valid delay data with proper datetime formats
    # Note: Train route is 11111->22222, but incident occurs on section 33333->44444
    # (different section of the route, or parallel incident tracking)
    processed_delays = [
        {
            "TRAIN_SERVICE_CODE": "T004",
            "PLANNED_ORIGIN_LOCATION_CODE": "11111",
            "PLANNED_ORIGIN_GBTT_DATETIME": "01-JAN-2024 12:00",
            "PLANNED_DEST_LOCATION_CODE": "22222",
            "PLANNED_DEST_GBTT_DATETIME": "01-JAN-2024 13:00",
            "START_STANOX": 33333,
            "END_STANOX": 44444,
            "PFPI_MINUTES": 10.0,
            "INCIDENT_REASON": "Test Incident",
            "INCIDENT_NUMBER": "INC001",
            "EVENT_TYPE": "A",
            "SECTION_CODE": "SEC001",
            "EVENT_DATETIME": "01-JAN-2024 12:10",
            "INCIDENT_START_DATETIME": "01-JAN-2024 12:00"
        }
    ]
    
    result = adjust_schedule_timeline(processed_schedule, processed_delays, st_code="11111")
    
    # Result should contain processed data
    assert len(result) >= 1
    assert isinstance(result, list)
    # Verify all results have required fields
    for entry in result:
        assert "TRAIN_SERVICE_CODE" in entry
        assert "PLANNED_CALLS" in entry
        assert "ACTUAL_CALLS" in entry


def test_adjust_schedule_timeline_k3():
    """Test matching: delays are correctly matched to schedules based on train code, location, time, and day"""
    processed_schedule = [
        {
            "TRAIN_SERVICE_CODE": "T100",
            "PLANNED_ORIGIN_LOCATION_CODE": "11111",
            "PLANNED_ORIGIN_GBTT_DATETIME": "0900",  # Must match origin_time format (colons removed)
            "PLANNED_DEST_LOCATION_CODE": "22222",
            "PLANNED_DEST_GBTT_DATETIME": "1000",   # Must match dest_time format (colons removed)
            "PLANNED_CALLS": "0900",
            "ACTUAL_CALLS": None,
            "PFPI_MINUTES": 0.0,
            "INCIDENT_REASON": None,
            "INCIDENT_NUMBER": None,
            "EVENT_TYPE": None,
            "SECTION_CODE": None,
            "DELAY_DAY": None,
            "EVENT_DATETIME": None,
            "INCIDENT_START_DATETIME": None,
            "ENGLISH_DAY_TYPE": ["MO"],
            "DFT_CATEGORY": "A",
            "PLATFORM_COUNT": 2,
            "STATION_ROLE": "Origin"
        }
    ]
    
    processed_delays = [
        {
            "TRAIN_SERVICE_CODE": "T100",
            "PLANNED_ORIGIN_LOCATION_CODE": "11111",
            "PLANNED_ORIGIN_GBTT_DATETIME": "01-JAN-2024 09:00",
            "PLANNED_ORIGIN_WTT_DATETIME": "01-JAN-2024 09:00",  # Used for day extraction
            "PLANNED_DEST_LOCATION_CODE": "22222",
            "PLANNED_DEST_GBTT_DATETIME": "01-JAN-2024 10:00",
            "START_STANOX": 55555,
            "END_STANOX": 66666,
            "PFPI_MINUTES": 15.0,
            "INCIDENT_REASON": "Matched Delay",
            "INCIDENT_NUMBER": "INC100",
            "EVENT_TYPE": "A",
            "SECTION_CODE": "SEC100",
            "EVENT_DATETIME": "01-JAN-2024 09:15",
            "INCIDENT_START_DATETIME": "01-JAN-2024 09:00"
        }
    ]
    
    result = adjust_schedule_timeline(processed_schedule, processed_delays, st_code="11111")
    
    # Verify the function matched delays to schedule
    assert any(
        e["TRAIN_SERVICE_CODE"] == "T100" 
        and e["INCIDENT_REASON"] == "Matched Delay"
        and e["PFPI_MINUTES"] == 15.0
        and e["INCIDENT_NUMBER"] == "INC100"
        for e in result
    ), "Expected matched delay entry in result"


# =======================================================================================
# TESTS FOR HELPER FUNCTIONS - VALIDATION LAYER
# =======================================================================================

def test_is_valid_schedule_entry_k0():
    """Test that valid schedule entries are accepted"""
    entry = {
        "schedule_segment": {
            "schedule_location": [
                {"tiploc_code": "ABC", "departure": "0900"}
            ]
        }
    }
    
    result = is_valid_schedule_entry(entry)
    
    assert result is True


def test_is_valid_schedule_entry_k1():
    """Test that entries without schedule_segment are rejected"""
    entry = {"other_field": "value"}
    
    result = is_valid_schedule_entry(entry)
    
    assert result is False


def test_is_valid_schedule_entry_k2():
    """Test that entries with non-list schedule_location are rejected"""
    entry = {
        "schedule_segment": {
            "schedule_location": "not a list"
        }
    }
    
    result = is_valid_schedule_entry(entry)
    
    assert result is False


def test_validate_schedule_locations_k0():
    """Test that valid location list is accepted"""
    locations = [
        {"tiploc_code": "ABC", "departure": "0900"},
        {"tiploc_code": "DEF", "arrival": "1000"}
    ]
    
    result = validate_schedule_locations(locations)
    
    assert result is True


def test_validate_schedule_locations_k1():
    """Test that non-list locations are rejected"""
    result = validate_schedule_locations("not a list")
    
    assert result is False


def test_validate_schedule_locations_k2():
    """Test that locations containing non-dicts are rejected"""
    locations = [
        {"tiploc_code": "ABC"},
        "not a dict"
    ]
    
    result = validate_schedule_locations(locations)
    
    assert result is False


def test_is_valid_location_entry_k0():
    """Test that dict with get method is valid"""
    location = {"tiploc_code": "ABC", "departure": "0900"}
    
    result = is_valid_location_entry(location)
    
    assert result is True


def test_is_valid_location_entry_k1():
    """Test that non-dict is invalid"""
    result = is_valid_location_entry("not a dict")
    
    assert result is False


def test_has_time_information_k0():
    """Test location with departure time"""
    location = {"tiploc_code": "ABC", "departure": "0900"}
    
    result = has_time_information(location)
    
    assert result is True


def test_has_time_information_k1():
    """Test location with arrival time"""
    location = {"tiploc_code": "ABC", "arrival": "1000"}
    
    result = has_time_information(location)
    
    assert result is True


def test_has_time_information_k2():
    """Test location without any time information"""
    location = {"tiploc_code": "ABC"}
    
    result = has_time_information(location)
    
    assert result is False


def test_extract_location_time_k0():
    """Test extraction of departure time when present"""
    location = {"departure": "0845", "arrival": "0900"}
    
    result = extract_location_time(location)
    
    # Should prefer departure
    assert result == "0845"


def test_extract_location_time_k1():
    """Test extraction of arrival time when departure missing"""
    location = {"arrival": "1000"}
    
    result = extract_location_time(location)
    
    assert result == "1000"


def test_extract_location_time_k2():
    """Test returns None when no time available"""
    location = {"tiploc_code": "ABC"}
    
    result = extract_location_time(location)
    
    assert result is None


def test_get_train_service_code_k0():
    """Test extraction of valid train service code"""
    entry = {
        "schedule_segment": {
            "CIF_train_service_code": "T12345"
        }
    }
    
    result = get_train_service_code(entry)
    
    assert result == "T12345"


def test_get_train_service_code_k1():
    """Test returns None when code not found"""
    entry = {
        "schedule_segment": {
            "other_field": "value"
        }
    }
    
    result = get_train_service_code(entry)
    
    assert result is None


# =======================================================================================
# TESTS FOR HELPER FUNCTIONS - EXTRACTION LAYER
# =======================================================================================

def test_find_location_by_tiploc_k0():
    """Test finding location by matching TIPLOC code"""
    locations = [
        {"tiploc_code": "ABC", "departure": "0900"},
        {"tiploc_code": "DEF", "arrival": "1000"},
        {"tiploc_code": "GHI", "departure": "1100"}
    ]
    
    result = find_location_by_tiploc(locations, "DEF")
    
    assert result == {"tiploc_code": "DEF", "arrival": "1000"}


def test_find_location_by_tiploc_k1():
    """Test returns None when TIPLOC not found"""
    locations = [
        {"tiploc_code": "ABC", "departure": "0900"},
        {"tiploc_code": "DEF", "arrival": "1000"}
    ]
    
    result = find_location_by_tiploc(locations, "XYZ")
    
    assert result is None


def test_find_origin_location_k0():
    """Test finding origin location with LO type"""
    locations = [
        {"tiploc_code": "ABC", "location_type": "LO", "departure": "0900"},
        {"tiploc_code": "DEF", "arrival": "1000"}
    ]
    
    result = find_origin_location(locations, "ABC")
    
    assert result == {"tiploc_code": "ABC", "location_type": "LO", "departure": "0900"}


def test_find_origin_location_k1():
    """Test finding origin location with L0 type (zero instead of O)"""
    locations = [
        {"tiploc_code": "ABC", "location_type": "L0", "departure": "0900"}
    ]
    
    result = find_origin_location(locations, "ABC")
    
    assert result is not None
    assert result["location_type"] == "L0"


def test_find_destination_location_k0():
    """Test finding destination location with LT type"""
    locations = [
        {"tiploc_code": "ABC", "departure": "0900"},
        {"tiploc_code": "DEF", "location_type": "LT", "arrival": "1000"}
    ]
    
    result = find_destination_location(locations, "DEF")
    
    assert result == {"tiploc_code": "DEF", "location_type": "LT", "arrival": "1000"}


def test_find_destination_location_k1():
    """Test returns None when LT not found for TIPLOC"""
    locations = [
        {"tiploc_code": "ABC", "location_type": "LO", "departure": "0900"}
    ]
    
    result = find_destination_location(locations, "ABC")
    
    assert result is None


def test_determine_station_role_k0():
    """Test station role is Origin when at origin location"""
    relevant_loc = {"tiploc_code": "ABC"}
    origin_loc = {"tiploc_code": "ABC", "location_type": "LO"}
    dest_loc = None
    
    result = determine_station_role(relevant_loc, origin_loc, dest_loc, "ABC")
    
    assert result == "Origin"


def test_determine_station_role_k1():
    """Test station role is Destination when at destination location"""
    relevant_loc = {"tiploc_code": "DEF"}
    origin_loc = None
    dest_loc = {"tiploc_code": "DEF", "location_type": "LT"}
    
    result = determine_station_role(relevant_loc, origin_loc, dest_loc, "DEF")
    
    assert result == "Destination"


def test_determine_station_role_k2():
    """Test station role is Intermediate when at intermediate location"""
    relevant_loc = {"tiploc_code": "MID"}
    origin_loc = {"tiploc_code": "START"}
    dest_loc = {"tiploc_code": "END"}
    
    result = determine_station_role(relevant_loc, origin_loc, dest_loc, "MID")
    
    assert result == "Intermediate"


def test_determine_station_role_k3():
    """Test returns Unknown when no relevant location"""
    result = determine_station_role(None, None, None, "ABC")
    
    assert result == "Unknown"


def test_extract_time_components_from_delays_k0():
    """Test time components are extracted correctly from delays"""
    delays_df = pd.DataFrame({
        "PLANNED_ORIGIN_GBTT_DATETIME": ["01-JAN-2024 09:15"],
        "PLANNED_DEST_GBTT_DATETIME": ["01-JAN-2024 10:30"],
        "EVENT_DATETIME": ["01-JAN-2024 09:20"]
    })
    
    result = extract_time_components_from_delays(delays_df)
    
    assert "origin_time" in result.columns
    assert "dest_time" in result.columns
    assert "event_time" in result.columns
    assert result.iloc[0]["origin_time"] == "0915"
    assert result.iloc[0]["dest_time"] == "1030"
    assert result.iloc[0]["event_time"] == "0920"


def test_expand_schedule_by_days_k0():
    """Test schedule is expanded for multi-day schedules"""
    schedule_df = pd.DataFrame({
        "TRAIN_SERVICE_CODE": ["T001"],
        "ENGLISH_DAY_TYPE": [["MO", "TU", "WE"]],
        "PLANNED_CALLS": ["0900"]
    })
    
    result = expand_schedule_by_days(schedule_df)
    
    # Should have 3 rows (one per day)
    assert len(result) == 3
    # Each should have a current_day
    assert all("current_day" in result.columns for _ in range(len(result)))
    assert set(result["current_day"].tolist()) == {"MO", "TU", "WE"}


def test_expand_schedule_by_days_k1():
    """Test single-day schedules remain single row"""
    schedule_df = pd.DataFrame({
        "TRAIN_SERVICE_CODE": ["T002"],
        "ENGLISH_DAY_TYPE": [["FR"]],
        "PLANNED_CALLS": ["1000"]
    })
    
    result = expand_schedule_by_days(schedule_df)
    
    assert len(result) == 1
    assert result.iloc[0]["current_day"] == "FR"


def test_build_unmatched_entry_k0():
    """Test building unmatched entry from delay row"""
    delay_row = pd.Series({
        "TRAIN_SERVICE_CODE": "T123",
        "PLANNED_ORIGIN_LOCATION_CODE": "11111",
        "PLANNED_DEST_LOCATION_CODE": "22222",
        "origin_time": "0900",
        "dest_time": "1000",
        "event_time": "0910",
        "delay_day": "MO",
        "PFPI_MINUTES": 10.0,
        "INCIDENT_REASON": "Test Reason",
        "INCIDENT_NUMBER": "INC123",
        "EVENT_TYPE": "A",
        "SECTION_CODE": "SEC1",
        "EVENT_DATETIME": "01-JAN-2024 09:10",
        "INCIDENT_START_DATETIME": "01-JAN-2024 09:00",
        "START_STANOX": 77777,
        "END_STANOX": 88888
    })
    
    result = build_unmatched_entry(delay_row, st_code="11111")
    
    assert result["TRAIN_SERVICE_CODE"] == "T123"
    assert result["ACTUAL_CALLS"] == "0910"
    assert result["INCIDENT_REASON"] == "Test Reason"
    assert result["PFPI_MINUTES"] == 10.0
    # Verify incident section is preserved (independent from train route)
    assert result["START_STANOX"] == 77777
    assert result["END_STANOX"] == 88888


def test_apply_delays_to_matches_k0():
    """Test that matched entries receive updated ACTUAL_CALLS and DELAY_DAY"""
    df = pd.DataFrame({
        "TRAIN_SERVICE_CODE": ["T001", "T002"],
        "PLANNED_CALLS": ["0900", "1000"],
        "ACTUAL_CALLS": ["0900", "1000"],
        "PFPI_MINUTES": [15.0, np.nan],
        "event_time": ["0915", np.nan],
        "delay_day": ["MO", np.nan]
    })
    
    matched_mask = pd.Series([True, False])
    
    result = apply_delays_to_matches(df, matched_mask)
    
    # Matched entry should have updated times
    assert result.iloc[0]["ACTUAL_CALLS"] == "0915"
    assert result.iloc[0]["DELAY_DAY"] == "MO"
    # Non-matched should keep planned times
    assert result.iloc[1]["ACTUAL_CALLS"] == "1000"
    assert result.iloc[1]["PFPI_MINUTES"] == 0


def test_filter_result_columns_k0():
    """Test that only required columns are kept in results"""
    # Create DataFrame with all expected final columns PLUS temporary working columns
    df = pd.DataFrame({
        # Required output columns
        "TRAIN_SERVICE_CODE": ["T001"],
        "PLANNED_ORIGIN_LOCATION_CODE": ["11111"],
        "PLANNED_ORIGIN_GBTT_DATETIME": ["08:00"],
        "PLANNED_DEST_LOCATION_CODE": ["22222"],
        "PLANNED_DEST_GBTT_DATETIME": ["09:00"],
        "PLANNED_CALLS": ["0900"],
        "ACTUAL_CALLS": ["0910"],
        "PFPI_MINUTES": [10.0],
        "INCIDENT_REASON": ["Delay"],
        "INCIDENT_NUMBER": ["INC123"],
        "EVENT_TYPE": ["A"],
        "SECTION_CODE": ["SEC1"],
        "DELAY_DAY": ["MO"],
        "EVENT_DATETIME": ["01-JAN-2024 09:10"],
        "INCIDENT_START_DATETIME": ["01-JAN-2024 09:00"],
        "ENGLISH_DAY_TYPE": [["MO"]],
        "DFT_CATEGORY": ["A"],
        "PLATFORM_COUNT": [2],
        "STATION_ROLE": ["Origin"],
        "START_STANOX": [55555],
        "END_STANOX": [66666],
        # Temporary working columns that should be filtered out
        "origin_time": ["0900"],
        "dest_time": ["0910"],
        "event_time": ["0910"],
        "delay_day": ["MO"],
        "current_day": ["MO"],
        "EXTRA_COLUMN": ["should_be_removed"]
    })
    
    result = filter_result_columns(df)
    
    # Verify all required columns are present
    assert "TRAIN_SERVICE_CODE" in result.columns
    assert "PLANNED_ORIGIN_LOCATION_CODE" in result.columns
    assert "PLANNED_CALLS" in result.columns
    assert "ACTUAL_CALLS" in result.columns
    assert "PFPI_MINUTES" in result.columns
    assert "INCIDENT_REASON" in result.columns
    assert "STATION_ROLE" in result.columns
    assert "START_STANOX" in result.columns
    assert "END_STANOX" in result.columns
    # Verify incident section is preserved (independent from train route)
    assert result.iloc[0]["START_STANOX"] == 55555
    assert result.iloc[0]["END_STANOX"] == 66666
    # Verify temporary working columns are removed
    assert "origin_time" not in result.columns
    assert "dest_time" not in result.columns
    assert "event_time" not in result.columns
    assert "current_day" not in result.columns
    assert "EXTRA_COLUMN" not in result.columns


