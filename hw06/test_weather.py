"""
Test file for the weather station functions - Template

Copy this file to `test_weather.py` and replace every TODO with your own tests.
Keep `weather.py` in the same folder so the import below works.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from weather import (
    celsius_to_fahrenheit,
    clean_readings,
    heat_alert,
    run_pipeline,
    summarize_temperatures,
)

# =============================================================================
# PART 1: UNIT TESTS
# =============================================================================


# TODO: add @pytest.mark.parametrize with at least 5 (celsius, expected) cases,
# including absolute zero itself.
@pytest.mark.parametrize("celsius, expected", [
    (0, 32),
    (100, 212),
    (-40, -40),
    (37, 98.6),
    (-273.15, -459.67),
])
def test_celsius_to_fahrenheit(celsius, expected):
    """Check conversion for five temperatures, including absolute zero."""
    # Act
    result = celsius_to_fahrenheit(celsius)

    # Assert
    assert result == pytest.approx(expected)


# TODO: add @pytest.mark.parametrize with at least 4 (bad_input, expected_error) cases.
@pytest.mark.parametrize("bad_input, expected_error", [
    ("37", TypeError),
    (None, TypeError),
    (-273.16, ValueError),
    (-300, ValueError),
])
def test_celsius_to_fahrenheit_invalid(bad_input, expected_error):
    """Check that invalid types and temperatures raise the expected errors."""
    with pytest.raises(expected_error):
        celsius_to_fahrenheit(bad_input)


def test_summarize_temperatures():
    """Check that the summary ignores NaN and counts the missing reading."""
    # Arrange
    temps = [10.0, float("nan"), 20.0]

    # Act
    result = summarize_temperatures(temps)

    # Assert
    assert result["mean"] == pytest.approx(15.0)
    assert result["min"] == pytest.approx(10.0)
    assert result["max"] == pytest.approx(20.0)
    assert result["n_missing"] == 1


# =============================================================================
# PART 2: FIXTURES
# =============================================================================


@pytest.fixture
def raw_readings():
    """Provide unsorted readings with one missing and one implausible temperature."""
    return pd.DataFrame({
        "station": ["JFK", "CPK", "JFK", "CPK", "LGA"],
        "date": [
            "2026-07-02",
            "2026-07-01",
            "2026-07-01",
            "2026-07-02",
            "2026-07-01",
        ],
        "temp_c": [30.0, 25.0, float("nan"), 999.0, 28.0],
    })


def test_raw_readings_fixture(raw_readings):
    """Check the fixture's columns, row count, and missing temperature count."""
    assert list(raw_readings.columns) == ["station", "date", "temp_c"]
    assert len(raw_readings) == 5
    assert raw_readings["temp_c"].isna().sum() == 1


# =============================================================================
# PART 3: TESTING DATAFRAMES
# =============================================================================


def test_clean_readings(raw_readings):
    """Check removal of invalid readings, sorting, and Fahrenheit conversion."""
    # Arrange
    expected = pd.DataFrame({
        "station": ["CPK", "JFK", "LGA"],
        "date": ["2026-07-01", "2026-07-02", "2026-07-01"],
        "temp_c": [25.0, 30.0, 28.0],
        "temp_f": [77.0, 86.0, 82.4],
    })

    # Act
    result = clean_readings(raw_readings)

    # Assert
    assert_frame_equal(result, expected)


# =============================================================================
# PART 4: MOCKING
# =============================================================================


# TODO: add @pytest.mark.parametrize with at least 5 (forecast_high, expected) cases,
# including both thresholds.
@pytest.mark.parametrize("forecast_high, expected", [
    (35.0, "heat warning"),
    (30.0, "heat warning"),
    (15.0, "normal"),
    (0.0, "cold warning"),
    (-5.0, "cold warning"),
])
def test_heat_alert(mocker, forecast_high, expected):
    """Check all three alert levels, including both temperature boundaries."""
    # Arrange: replace the API call with a controlled forecast.
    mocker.patch("weather.fetch_forecast", return_value=forecast_high)

    # Act
    result = heat_alert("New York")

    # Assert
    assert result == expected


# =============================================================================
# PART 5: INTEGRATION TEST
# =============================================================================


def test_run_pipeline_end_to_end(tmp_path, raw_readings):
    """Check that the pipeline reads, cleans, and saves the expected data."""
    # Arrange
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "cleaned.csv"
    raw_readings.to_csv(input_path, index=False)

    expected = pd.DataFrame({
        "station": ["CPK", "JFK", "LGA"],
        "date": ["2026-07-01", "2026-07-02", "2026-07-01"],
        "temp_c": [25.0, 30.0, 28.0],
        "temp_f": [77.0, 86.0, 82.4],
    })

    # Act
    run_pipeline(input_path, output_path)

    # Assert
    assert output_path.exists()
    result = pd.read_csv(output_path)
    assert_frame_equal(result, expected)
