import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from src.data_loader import DataLoader
from src.models import BaselineModel, LinearTrendModel, ARIMAModel, EnsembleModel, LSTMModel, LLMModel
from src.reporting import PDFReportGenerator

# Page Config
st.set_page_config(page_title="Blood Lab Inventory Forecast", layout="wide")

st.title("🩸 Blood Lab Inventory Forecast PoC")

# Sidebar Configuration
st.sidebar.header("Configuration")

# 1. File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Inventory Excel", type=['xlsx'])

# 2. Simulation Mode (REMOVED per user request)
# "Always use all of the data to train"
enable_backtest = False 

# 3. Model Selector
model_choice = st.sidebar.selectbox("Select Forecasting Model", 
                                    ["Bubble 1: Baseline (Moving Average)", 
                                     "Bubble 2: Linear Trend (Coming Soon)",
                                     "Bubble 3: ARIMA (Coming Soon)",
                                     "Bubble 4: Ensemble AI (XGBoost)",
                                     "Bubble 5: Deep Learning (LSTM)",
                                     "Bubble 6: LLM Modifier (OpenRouter)"])

run_forecast = st.sidebar.button("Generate Forecast")

# Main Logic
def main():
    # Session State Initialization
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None

    # File Handling
    if uploaded_file is not None:
        filepath = "temp_uploaded.xlsx"
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        # Auto-detect default file in data folder
        data_dir = "data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith(".xlsx")]
            if files:
                filepath = os.path.join(data_dir, files[0])
                st.info(f"Using default dataset: **{files[0]}**")
            else:
                 st.info("Please upload an Excel file to get started (or place a .xlsx in the 'data' folder).")
                 return
        else:
             st.info("Please upload an Excel file to get started.")
             return

    # Load Data (Only on first load or change?)
    # For PoC simplicity, load every time but maybe persist?
    # Let's clean load logic.
    try:
        loader = DataLoader(filepath)
        raw_data = loader.load_data()
        
        # Always use ALL data for training
        train_data = raw_data.copy()
        test_data = pd.DataFrame() # No ground truth if we use all data (predicting unknown future)
        
        simulation_title_suffix = ""

        with st.expander("Raw Aggregated Data Preview"):
            st.dataframe(train_data.head())
        
        # Stats
        total_items = train_data['Item Code'].nunique()
        st.metric("Total Items Tracked", total_items)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Tabs for Main View
    tab1, tab2 = st.tabs(["📊 Forecast & Recommendations", "📈 Data Analytics"])

    with tab1:
        # Forecasting Logic
        if run_forecast:
            st.subheader(f"Forecast Results")
            
            model = None
            if "Baseline" in model_choice:
                model = BaselineModel(window_size=2)
            elif "Linear Trend" in model_choice:
                model = LinearTrendModel()
            elif "ARIMA" in model_choice:
                model = ARIMAModel()
            elif "Ensemble AI" in model_choice:
                model = EnsembleModel()
            elif "Deep Learning" in model_choice:
                model = LSTMModel(epochs=50) # Lightweight for PoC
            elif "LLM Modifier" in model_choice:
                 model = LLMModel()
                
            if model:
                with st.spinner(f"Training {model_choice}..."):
                    model.fit(train_data)
                    forecast = model.predict(horizon=4) 
                
                st.success("Forecast Generated!")
                
                # --- Debug & Warnings ---
                if hasattr(model, 'warning') and model.warning:
                    st.warning(f"⚠️ Model Warning: {model.warning}")
                
                if hasattr(model, 'debug_info') and model.debug_info:
                    with st.expander("🛠️ Developer Pipeline (Prompt & Response)"):
                        st.markdown("**Last Sent Prompt:**")
                        st.code(model.debug_info.get('last_prompt', ''))
                        st.markdown("**Last Raw Response:**")
                        st.code(model.debug_info.get('last_response', ''))
                # ------------------------

                # Save to Session State
                st.session_state.forecast_data = forecast
                st.session_state.raw_data = raw_data
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
            else:
                st.warning(f"Model '{model_choice}' is not yet implemented.")



        # Report Generation Section using AI
        st.markdown("---")
        st.subheader("📄 Export Report")
        
        # Helper function to run forecast if missing
        def ensure_forecast_exists():
            if st.session_state.forecast_data is None:
                with st.spinner(f"Auto-running forecast using {model_choice}..."):
                    model_obj = None
                    if "Baseline" in model_choice:
                        model_obj = BaselineModel(window_size=2)
                    elif "Linear Trend" in model_choice:
                        model_obj = LinearTrendModel()
                    elif "ARIMA" in model_choice:
                        model_obj = ARIMAModel()
                    elif "Ensemble AI" in model_choice:
                        model_obj = EnsembleModel()
                    elif "Deep Learning" in model_choice:
                        model_obj = LSTMModel(epochs=50)
                    elif "LLM Modifier" in model_choice:
                        model_obj = LLMModel()
                        
                    if model_obj:
                        model_obj.fit(train_data)
                        forecast = model_obj.predict(horizon=4)
                        st.session_state.forecast_data = forecast
                        st.session_state.train_data = train_data # Update state
                        return True
            return True

        if st.button("Generate Inventory Risk Report (PDF)"):
            if train_data is None:
                st.error("Please load data first.")
            else:
                # 1. Ensure Forecast Exists
                ensure_forecast_exists()
                
                if st.session_state.forecast_data is not None:
                    with st.spinner("Analyzing risks and generating AI Manager Report..."):
                        # 2. Identify Risks
                        forecast_df = st.session_state.forecast_data
                        
                        # Get last closing balance
                        last_balances = st.session_state.train_data.sort_values('Quarter').groupby('Item Code').last()['Closing Balance']
                        
                        # Merge with forecast (first quarter)
                        first_q = forecast_df['Quarter'].min()
                        risk_df = forecast_df[forecast_df['Quarter'] == first_q].copy()
                        risk_df = risk_df.join(last_balances, on='Item Code')
                        risk_df['Gap'] = risk_df['Forecasted_Outgoing'] - risk_df['Closing Balance']
                        
                        # Filter for positive gap (Shortage)
                        shortages = risk_df[risk_df['Gap'] > 0].sort_values('Gap', ascending=False)
                        
                        # 3. Generate Manager Summary via LLM
                        llm = LLMModel() 
                        # Use new method
                        manager_report = llm.generate_manager_report(shortages)
                        
                        # 4. Create PDF
                        pdf_gen = PDFReportGenerator()
                        report_filename = "Inventory_Risk_Manager_Report.pdf"
                        pdf_gen.generate_report(manager_report, shortages, file_path=report_filename)
                        
                        # 5. Download Button
                        with open(report_filename, "rb") as f:
                            st.download_button("Download Manager Report (PDF)", f, file_name=report_filename, mime="application/pdf")
                        st.success("Manager Report generated successfully! Click download above.")

        # Visualization and Results (Check Session State)
        if st.session_state.forecast_data is not None:
            # Load from state
            forecast = st.session_state.forecast_data
            raw_data = st.session_state.raw_data
            train_data = st.session_state.train_data
            test_data = st.session_state.test_data # likely empty now
            
            visualize_forecast_and_recommend(raw_data, train_data, forecast, test_data)

    with tab2:
        st.header("Data Statistics Dashboard")
        if train_data is not None:
            # 1. High Level Metrics
            col1, col2, col3 = st.columns(3)
            total_items = train_data['Item Code'].nunique()
            total_demand = train_data['Outgoing'].sum()
            avg_quarterly_demand = train_data.groupby('Quarter')['Outgoing'].sum().mean()
            
            col1.metric("Total Items Tracked", total_items)
            col2.metric("Total Lifetime Demand", f"{total_demand:,.0f}")
            col3.metric("Avg. Demand per Quarter", f"{avg_quarterly_demand:,.0f}")
            
            # 2. Top Movers
            st.subheader("Top 5 Fast Moving Items")
            top_items = train_data.groupby('Item Code')['Outgoing'].sum().sort_values(ascending=False).head(5)
            # Merge with description
            top_items_df = top_items.to_frame().reset_index()
            # Get description map
            desc_map = train_data[['Item Code', 'Product Description']].drop_duplicates().set_index('Item Code')
            top_items_df = top_items_df.join(desc_map, on='Item Code')
            st.dataframe(top_items_df, use_container_width=True)
            
            # 3. Demand Distribution
            st.subheader("Quarterly Global Demand Trend")
            global_trend = train_data.groupby('Quarter')['Outgoing'].sum().reset_index()
            # Convert quarter to string for plotting
            global_trend['Quarter'] = global_trend['Quarter'].astype(str)
            fig_trend = px.bar(global_trend, x='Quarter', y='Outgoing', title="Total Lab Consumption per Quarter")
            st.plotly_chart(fig_trend, theme="streamlit", use_container_width=True)
            
            # 4. Model Championship (Leaderboard)
            st.markdown("---")
            st.header("🏆 Model Championship")
            st.markdown("Compare models by backtesting them on the **last 2 quarters** of your data.")
            
            if st.button("Run Championship"):
                with st.spinner("Running tournament... this may take a moment."):
                    # 1. Prepare Split
                    all_quarters = sorted(train_data['Quarter'].unique())
                    if len(all_quarters) < 4:
                        st.error("Not enough data history to run championship (need 4+ quarters).")
                    else:
                        test_quarters = all_quarters[-2:]
                        train_quarters = all_quarters[:-2]
                        
                        train_split = train_data[train_data['Quarter'].isin(train_quarters)].copy()
                        test_split = train_data[train_data['Quarter'].isin(test_quarters)].copy()
                        
                        models_to_test = [
                            ("Baseline (SMA)", BaselineModel(window_size=2)),
                            ("Linear Trend", LinearTrendModel()),
                            ("ARIMA", ARIMAModel()),
                            ("Ensemble (XGB)", EnsembleModel()),
                            ("Deep Learning (LSTM)", LSTMModel(epochs=10)), # Reduced epochs for speed
                            ("LLM Modifier", LLMModel())
                        ]
                        
                        results = []
                        
                        progress_bar = st.progress(0)
                        for idx, (name, model) in enumerate(models_to_test):
                            try:
                                model.fit(train_split)
                                pred_df = model.predict(horizon=2)
                                
                                # Evaluate
                                # Join prediction with ground truth (test_split)
                                merged = pd.merge(test_split, pred_df, on=['Item Code', 'Quarter'], how='inner', suffixes=('_true', '_pred'))
                                
                                if not merged.empty:
                                    y_true = merged['Outgoing']
                                    y_pred = merged['Forecasted_Outgoing']
                                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                                    results.append({"Model": name, "RMSE": rmse})
                                else:
                                    results.append({"Model": name, "RMSE": np.nan})
                            except Exception as e:
                                # st.error(f"Error evaluating {name}: {e}")
                                results.append({"Model": name, "RMSE": np.nan})
                            
                            progress_bar.progress((idx + 1) / len(models_to_test))
                        
                        # Display Leaderboard
                        res_df = pd.DataFrame(results).sort_values("RMSE")
                        st.success(f"Winner: **{res_df.iloc[0]['Model']}**")
                        st.table(res_df)
                        
        else:
            st.info("Load data to view statistics.")

def visualize_forecast_and_recommend(full_history, train_history, forecast, ground_truth=None):
    # Select an item to visualize
    items = sorted(train_history['Item Code'].unique())
    selected_item = st.selectbox("Select Item to Visualize", items)
    
    # Filter data for specific item
    hist_item = full_history[full_history['Item Code'] == selected_item].copy()
    pred_item = forecast[forecast['Item Code'] == selected_item].copy()
    
    # --- Recommendation Logic ---
    # 1. Get current stock (Closing Balance of the LAST VALID quarter)
    last_train_record = train_history[train_history['Item Code'] == selected_item].sort_values('Quarter').iloc[-1]
    current_stock = last_train_record['Closing Balance']
    
    # 2. Select Target Quarter
    forecast_quarters = sorted(pred_item['Quarter'].unique()) # Ensure sorted
    
    if not forecast_quarters:
        st.warning("No forecast generated.")
        return

    target_date = st.selectbox("Planning Order For:", forecast_quarters, index=0)
    
    # Get Forecasted Demand
    target_row = pred_item[pred_item['Quarter'] == target_date]
    if not target_row.empty:
        target_demand = target_row.iloc[0]['Forecasted_Outgoing']
    else:
        target_demand = 0
        
    # 3. Calculate Gap
    needed_quantity = max(0, target_demand - current_stock)
    
    # Display Big Number
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Current Stock ({last_train_record['Quarter']})", f"{current_stock:,.0f}")
    col2.metric(f"Forecasted Demand ({target_date})", f"{target_demand:,.0f}")
    col3.metric("Recommended Order", f"{needed_quantity:,.0f}", delta_color="normal")


    if needed_quantity > 0:
        st.info(f"💡 To meet demand for **{target_date}**, you should order **{needed_quantity:,.0f}** units.")
    else:
        st.success(f"✅ Sufficient stock for {target_date}.")

    # --- Plotting ---
    # We want to show:
    # 1. Historical Data (Green? or Blue)
    # 2. Forecast (Red dashed)
    # 3. If Backtesting: Ground Truth (Gray or Blue solid where forecast is)
    
    # Let's combine for Plotly
    
    # Prepare History (Up to cutoff)
    train_plot = train_history[train_history['Item Code'] == selected_item].copy()
    train_plot['Type'] = 'Historical (Training)'
    
    # Prepare Forecast
    pred_plot = pred_item.copy()
    pred_plot['Type'] = 'Forecast'
    
    # Prepare Ground Truth (if backtesting)
    combined_list = [train_plot, pred_plot]
    
    if ground_truth is not None and not ground_truth.empty:
        truth_plot = ground_truth[ground_truth['Item Code'] == selected_item].copy()
        if not truth_plot.empty:
            truth_plot['Type'] = 'Actual (Ground Truth)'
            combined_list.append(truth_plot)
            
    # Normalize columns
    for df in combined_list:
        if 'Outgoing' in df.columns:
            df.rename(columns={'Outgoing': 'Quantity'}, inplace=True)
        if 'Forecasted_Outgoing' in df.columns:
             df.rename(columns={'Forecasted_Outgoing': 'Quantity'}, inplace=True)
    
    combined = pd.concat(combined_list)
    
    # Plot
    fig = px.line(combined, x='Quarter', y='Quantity', color='Type', markers=True,
                  title=f"Predicted Quarterly Consumption (Required Stock): {selected_item} ({train_plot['Product Description'].iloc[0]})")
    
    # Render Chart
    st.plotly_chart(fig, theme="streamlit", width="stretch")

if __name__ == "__main__":
    main()
