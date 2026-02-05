import sys
import os
import pandas as pd
sys.path.append(os.getcwd())

from src.reporting import PDFReportGenerator
from src.models import LLMModel, LSTMModel

def test_pdf_generation():
    print("Testing PDF Generation...")
    try:
        # Dummy data
        data = {
            'Item Code': ['A', 'B'],
            'Product Description': ['Test A', 'Test B'],
            'Forecasted_Outgoing': [100, 50],
            'Closing Balance': [80, 60],
            'Gap': [20, -10]
        }
        df = pd.DataFrame(data)
        
        pdf_gen = PDFReportGenerator()
        pdf_gen.generate_report("This is a test summary.", df, file_path="test_report.pdf")
        
        if os.path.exists("test_report.pdf"):
            print("PDF created successfully.")
            # os.remove("test_report.pdf") # Keep it for manual check if needed
        else:
            print("PDF file not found.")
            
    except Exception as e:
        print(f"PDF Generation Failed: {e}")

def test_models_exist():
    print("Testing Models Import...")
    try:
        llm = LLMModel()
        lstm = LSTMModel(epochs=1)
        print("Models instantiated successfully.")
        
        if hasattr(llm, 'generate_executive_summary'):
             print("LLM has summary method.")
        else:
             print("LLM missing summary method.")
             
    except Exception as e:
       print(f"Model Test Failed: {e}")

if __name__ == "__main__":
    test_pdf_generation()
    test_models_exist()
