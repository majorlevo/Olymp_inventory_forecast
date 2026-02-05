import pandas as pd
import os

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

    def load_data(self) -> pd.DataFrame:
        """
        Loads the Excel data, cleans it, and aggregates it by Quarter.
        
        Returns:
            pd.DataFrame: Aggregated data with columns ['Item Code', 'Quarter', ... sum columns]
        """
        # Load Data
        df = pd.read_excel(self.filepath)
        
        # specific columns expected
        expected_cols = ['Order Date', 'Item Code', 'Product Description', 
                         'Expiration date', 'Unit', 'Opening Balance', 
                         'Incoming', 'Outgoing', 'Closing Balance']
        
        # Normalize columns (strip spaces)
        df.columns = df.columns.str.strip()
        
        # Basic Cleaning
        df.fillna(0, inplace=True)
        
        # Date Conversion
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%Y.%m.%d', errors='coerce')
        # Drop rows with invalid dates if necessary, or handle them. 
        # For now assuming data is clean enough or errors=coerce creates NaT which we can drop
        df = df.dropna(subset=['Order Date'])
        
        # Create Quarter Column
        # Convert date to period 'Q' then back to string or timestamp for grouping
        df['Quarter'] = df['Order Date'].dt.to_period('Q').astype(str)
        
        # Aggregation Logic (FIFO impact: we just sum up outgoing/incoming per item per quarter)
        # We need to keep 'Product Description' so we take the 'first' or 'max'
        agg_rules = {
            'Product Description': 'first',
            'Unit': 'first',
            'Opening Balance': 'sum', # Needs care: Opening of Period vs Sum of daily openings? 
                                      # Wait. Daily "Opening Balance" is the balance at start of day.
                                      # The Quarterly Opening Balance should be the Opening Balance of the FIRST record in that quarter.
                                      # The Quarterly Closing Balance should be the Closing Balance of the LAST record.
                                      # Incoming/Outgoing should be SUM.
            'Incoming': 'sum',
            'Outgoing': 'sum',
            'Closing Balance': 'last' # Logic fix below
        }
        
        # We need to sort by date to get correct first/last
        df = df.sort_values(by=['Item Code', 'Order Date'])
        
        # Group by Item and Quarter
        grouped = df.groupby(['Item Code', 'Quarter']).agg({
            'Product Description': 'first',
            'Unit': 'first',
            'Opening Balance': 'first', # See note above
            'Incoming': 'sum',
            'Outgoing': 'sum',
            'Closing Balance': 'last'
        }).reset_index()
        
        # Calculate calculated closing to verify? 
        # For now trust the data or the aggregation: 
        # Q_Close = Q_Open + Q_In - Q_Out. 
        # The raw data "Closing Balance" is per-day. The "Last" one is the Quarter Close. Correct.
        
        return grouped
