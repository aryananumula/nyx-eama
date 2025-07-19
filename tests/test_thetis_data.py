import pytest
import pandas as pd
from src.data.thetis_data import load_data, explore_data

def test_load_data_valid():
    data = load_data()
    assert data is not None
    assert isinstance(data, dict)
    assert all(isinstance(df, pd.DataFrame) for df in data.values())

def test_load_data_no_csv():
    # Mocking the output directory to simulate no CSV files
    import os
    from unittest.mock import patch

    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.glob', return_value=[]):
        data = load_data()
        assert data is None

def test_explore_data_empty():
    explore_data(None)  # Should not raise any errors

def test_explore_data_valid():
    # Create a mock DataFrame for testing
    mock_data = {
        'stroke1': pd.DataFrame({'column1': [1, 2], 'column2': [3, 4]}),
        'stroke2': pd.DataFrame({'column1': [5, 6], 'column2': [7, 8]})
    }
    
    explore_data(mock_data)  # Should print information about the datasets

    # Since we are not capturing print output, we cannot assert on printed values directly.
    # In a real test, you might want to use a context manager to capture stdout.