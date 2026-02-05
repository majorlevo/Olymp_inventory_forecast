from fpdf import FPDF
import pandas as pd
import datetime

class PDFReportGenerator(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)

    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Olymp Inventory Forecast - Executive Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_section_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def add_body_text(self, text):
        self.set_font("Arial", size=11)
        self.multi_cell(0, 7, text)
        self.ln(5)

    def add_df_table(self, df: pd.DataFrame):
        """
        Simple table generator for FPDF.
        Assumes df columns fit in page width (A4 ~210mm).
        """
        self.set_font("Arial", "B", 10)
        
        # Calculate column widths (simple logic: divide equally)
        page_width = self.w - 2 * self.l_margin
        col_width = page_width / len(df.columns)
        
        # Header
        for col in df.columns:
            self.cell(col_width, 10, str(col), border=1, align="C")
        self.ln()
        
        # Rows
        self.set_font("Arial", size=10)
        for index, row in df.iterrows():
            for col in df.columns:
                # Truncate content if too long
                content = str(row[col])[:20]
                self.cell(col_width, 10, content, border=1)
            self.ln()
        self.ln(10)

    def generate_report(self, executive_summary: str, risk_data: pd.DataFrame, file_path: str = "Inventory_Report.pdf"):
        """
        Generates the full report.
        """
        # Metadata
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        self.ln(5)

        # 1. Manager's Report
        self.add_section_title("1. Inventory Manager Report & Strategy")
        self.add_body_text(executive_summary)

        # 2. Key Risks
        self.add_section_title("2. Critical Stock Risks (Data)")
        if not risk_data.empty:
            # Prepare display frame
            display_df = risk_data.copy()
            if 'Gap' in display_df.columns:
                display_df.rename(columns={'Gap': 'Rec. Order'}, inplace=True)
            if 'Forecasted_Outgoing' in display_df.columns:
                display_df.rename(columns={'Forecasted_Outgoing': 'Forecast'}, inplace=True)
            if 'Closing Balance' in display_df.columns:
                display_df.rename(columns={'Closing Balance': 'Stock'}, inplace=True)
                
            # Select relevant columns for PDF
            display_cols = ['Item Code', 'Product Description', 'Stock', 'Forecast', 'Rec. Order']
            
            # Filter if columns exist
            cols = [c for c in display_cols if c in display_df.columns]
            
            # Format numbers
            for c in ['Stock', 'Forecast', 'Rec. Order']:
                if c in display_df.columns:
                    display_df[c] = display_df[c].apply(lambda x: f"{x:.0f}")

            self.add_df_table(display_df[cols].head(20)) # Top 20
        else:
            self.add_body_text("No critical risks identified for the forecast period.")

        self.output(file_path)
