"""Weather station utilities for QMSS G5072 Homework 6."""

import math
from pathlib import Path

import pandas as pd
import requests

ABSOLUTE_ZERO_C = -273.15
MIN_PLAUSIBLE_C = -90.0
MAX_PLAUSIBLE_C = 60.0


def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert a temperature from degrees Celsius to degrees Fahrenheit.

    Args:
        celsius: Temperature in degrees Celsius.

    Returns:
        Temperature in degrees Fahrenheit.

    Raises:
        TypeError: If `celsius` is not an int or a float.
        ValueError: If `celsius` is below absolute zero (-273.15 C).
    """
    if isinstance(celsius, bool) or not isinstance(celsius, (int, float)):
        raise TypeError("celsius must be a number")
    if celsius < ABSOLUTE_ZERO_C:
        raise ValueError("temperature cannot be below absolute zero")
    return celsius * 9 / 5 + 32


def summarize_temperatures(temps: list[float]) -> dict[str, float]:
    """
    Summarize a list of temperature readings, ignoring missing values (NaN).

    Args:
        temps: Temperature readings. Missing readings are stored as float("nan").

    Returns:
        A dictionary with the keys "mean", "min", "max", and "n_missing".

    Raises:
        ValueError: If there are no valid (non-missing) readings.
    """
    valid = [t for t in temps if not math.isnan(t)]
    if not valid:
        raise ValueError("no valid temperature readings")
    return {
        "mean": sum(valid) / len(valid),
        "min": min(valid),
        "max": max(valid),
        "n_missing": len(temps) - len(valid),
    }


def clean_readings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw station readings.

    Steps:
        1. Drop rows where `temp_c` is missing.
        2. Drop physically implausible readings (outside -90 C to 60 C, inclusive bounds are kept).
        3. Add a `temp_f` column with the temperature in Fahrenheit.
        4. Sort by `station`, then `date`, and reset the index.

    Args:
        df: DataFrame with the columns `station`, `date`, and `temp_c`.

    Returns:
        A new, cleaned DataFrame. The input DataFrame is not modified.
    """
    cleaned = df.dropna(subset=["temp_c"])
    in_range = cleaned["temp_c"].between(MIN_PLAUSIBLE_C, MAX_PLAUSIBLE_C)
    cleaned = cleaned[in_range].copy()
    cleaned["temp_f"] = cleaned["temp_c"] * 9 / 5 + 32
    return cleaned.sort_values(["station", "date"]).reset_index(drop=True)


def fetch_forecast(city: str) -> float:
    """
    Fetch tomorrow's forecast high (in Celsius) for a city from the wttr.in web API.

    This function makes a real network request: it is slow, needs internet access,
    and returns a different answer every day.

    Args:
        city: Name of the city, e.g. "New York".

    Returns:
        Forecast maximum temperature in degrees Celsius.
    """
    response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
    response.raise_for_status()
    return float(response.json()["weather"][1]["maxtempC"])


def heat_alert(city: str, heat_threshold: float = 30.0, cold_threshold: float = 0.0) -> str:
    """
    Classify tomorrow's forecast for a city into an alert level.

    Args:
        city: Name of the city.
        heat_threshold: Forecast highs at or above this value trigger a heat warning.
        cold_threshold: Forecast highs at or below this value trigger a cold warning.

    Returns:
        "heat warning", "cold warning", or "normal".
    """
    forecast_high = fetch_forecast(city)
    if forecast_high >= heat_threshold:
        return "heat warning"
    if forecast_high <= cold_threshold:
        return "cold warning"
    return "normal"


def run_pipeline(input_path: Path, output_path: Path) -> None:
    """Load raw readings from a CSV file, clean them, and save the result as a CSV file."""
    raw = pd.read_csv(input_path)
    clean_readings(raw).to_csv(output_path, index=False)
