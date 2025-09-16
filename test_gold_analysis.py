import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame with Date index."""
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values and normalize column names."""
    df = df.dropna()
    df.columns = [col.lower().strip() for col in df.columns]
    return df


class TestGoldAnalysis(unittest.TestCase):

    def test_load_data(self):
        df = load_data("gold_data_2015_25.csv")
        self.assertFalse(df.empty)
        for col in ["SPX", "GLD", "USO", "SLV", "EUR/USD"]:
            self.assertIn(col, df.columns)

    def test_clean_data_removes_nulls(self):
        df = pd.DataFrame({
            "Price": [1200, None, 1250],
            "Date": ["2020-01-01", "2020-01-02", None]
        })
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), 1)
        self.assertListEqual(list(cleaned.columns), ["price", "date"])

    def test_ml_model_prediction(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        model = LinearRegression().fit(X, y)
        preds = model.predict([[5]])
        self.assertAlmostEqual(preds[0], 10.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
    df = load_data("gold_data_2015_25.csv")
    print(df.head())
