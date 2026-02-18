import unittest

import numpy as np
import pandas as pd

try:
    import preprocess_dataset as pdm
except Exception:  # pragma: no cover - dependency may be unavailable
    pdm = None


class TestPreprocessDataset(unittest.TestCase):
    @unittest.skipIf(pdm is None, "preprocess_dataset import failed (missing deps)")
    def test_parse_columbia(self):
        self.assertEqual(
            pdm.parse_columbia("001_2m_-5P_10V_15H.jpg"),
            (-5, 10, 15),
        )
        self.assertIsNone(pdm.parse_columbia("bad_name.jpg"))

    @unittest.skipIf(pdm is None, "preprocess_dataset import failed (missing deps)")
    def test_remove_outliers_iqr(self):
        df = pd.DataFrame(
            {
                "H": [0, 1, 2, 100],
                "V": [0, 1, 2, 100],
            }
        )
        cleaned = pdm.remove_outliers_iqr(df, ("H", "V"))
        self.assertLess(cleaned["H"].max(), 100)
        self.assertLess(cleaned["V"].max(), 100)

    @unittest.skipIf(pdm is None, "preprocess_dataset import failed (missing deps)")
    def test_clean_df_removes_nan_and_duplicates(self):
        df = pd.DataFrame(
            {
                "person_id": ["1", "1", "2"],
                "H": [0, 0, np.nan],
                "V": [1, 1, 2],
                "P": [0, 0, 0],
                "f0": [0.1, 0.1, 0.2],
                "f1": [0.1, 0.1, 0.2],
                "f2": [0.1, 0.1, 0.2],
                "f3": [0.1, 0.1, 0.2],
                "f4": [0.1, 0.1, 0.2],
                "f5": [0.1, 0.1, 0.2],
                "f6": [0.1, 0.1, 0.2],
                "f7": [0.1, 0.1, 0.2],
                "f8": [0.1, 0.1, 0.2],
                "f9": [0.1, 0.1, 0.2],
                "f10": [0.1, 0.1, 0.2],
            }
        )
        cleaned = pdm.clean_df(df)
        self.assertEqual(len(cleaned), 1)

