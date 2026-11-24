from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from churn_prediction.data import load_dataset


class LoadDatasetTests(TestCase):
    def test_numeric_text_with_blank_values_loads_as_numeric(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "customers.csv"
            pd.DataFrame(
                {
                    "TotalCharges": ["29.85", "1889.5", " "],
                    "Contract": ["month-to-month", "one year", "two year"],
                }
            ).to_csv(csv_path, index=False)

            frame = load_dataset(csv_path)

        self.assertTrue(pd.api.types.is_numeric_dtype(frame["TotalCharges"]))
        self.assertEqual(frame["TotalCharges"].iloc[:2].tolist(), [29.85, 1889.5])
        self.assertTrue(pd.isna(frame["TotalCharges"].iloc[2]))
        self.assertEqual(frame["Contract"].tolist(), ["month-to-month", "one year", "two year"])
