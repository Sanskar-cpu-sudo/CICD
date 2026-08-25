import pandas as pd
import pytest

from src.validate_data import validate_dataframe


def test_valid_data():

    df = pd.read_csv("data.csv")

    validate_dataframe(df)


def test_missing_column_fails():

    df = pd.read_csv("data.csv")

    df = df.drop(columns=["cgpa"])

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_invalid_cgpa_fails():

    df = pd.read_csv("data.csv")

    df.loc[0, "cgpa"] = 15

    with pytest.raises(ValueError):
        validate_dataframe(df)