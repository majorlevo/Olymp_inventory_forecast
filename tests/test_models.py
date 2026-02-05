import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import BaselineModel

class TestBaselineModel(unittest.TestCase):
    def test_forecast(self):
        # Create mock data
        data = pd.DataFrame({
            'Item Code': ['A', 'A', 'B', 'B'],
            'Quarter': ['2024Q1', '2024Q2', '2024Q1', '2024Q2'],
            'Outgoing': [100, 200, 50, 50],
            'Product Description': ['Prod A', 'Prod A', 'Prod B', 'Prod B'],
            'Unit': ['pcs', 'pcs', 'box', 'box'],
            # Add dummy remaining columns to match expectance if any (none strictly needed by logic but good practice)
            'Opening Balance': [0,0,0,0],
            'Incoming': [0,0,0,0],
            'Closing Balance': [0,0,0,0]
        })
        
        model = BaselineModel(window_size=2)
        model.fit(data)
        
        forecast = model.predict(horizon=1)
        
        self.assertEqual(len(forecast), 2) # 2 items
        
        # Check Item A
        # Avg of 100 and 200 is 150
        item_a = forecast[forecast['Item Code'] == 'A'].iloc[0]
        self.assertEqual(item_a['Forecasted_Outgoing'], 150)
        self.assertEqual(item_a['Quarter'], '2024Q3')

        # Check Item B
        # Avg of 50 and 50 is 50
        item_b = forecast[forecast['Item Code'] == 'B'].iloc[0]
        self.assertEqual(item_b['Forecasted_Outgoing'], 50)

if __name__ == '__main__':
    unittest.main()
