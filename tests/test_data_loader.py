import unittest
import pandas as pd
import sys
import os

# Add root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        # Assumes data/example_data.xlsx exists as per discovery
        filepath = 'data/example_data.xlsx'
        if not os.path.exists(filepath):
            print(f"Skipping test: {filepath} not found")
            return
        
        loader = DataLoader(filepath)
        df = loader.load_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Item Code', df.columns)
        self.assertIn('Quarter', df.columns)
        self.assertIn('Incoming', df.columns)
        self.assertFalse(df.empty, "Dataframe should not be empty")
        
        print(f"\nSuccessfully loaded {len(df)} aggregated rows.")
        print(df.head(2))

if __name__ == '__main__':
    unittest.main()
