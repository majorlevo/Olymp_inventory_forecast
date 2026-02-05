import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import torch
import torch.nn as nn
import os
from openai import OpenAI
import json
import warnings

class ForecastingModel(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Trains the model on the provided data.
        Data is expected to be aggregated by Item Code and Quarter.
        """
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Generates forecast for the given horizon (number of quarters).
        Returns a DataFrame with columns: ['Item Code', 'Quarter', 'Forecasted_Outgoing', ...].
        """
        pass

class BaselineModel(ForecastingModel):
    def __init__(self, window_size: int = 2):
        self.window_size = window_size
        self.data = None

    def fit(self, data: pd.DataFrame):
        self.data = data

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Model must be trained before prediction.")

        # Logic: 
        # 1. Identify valid items (unique Item Codes)
        # 2. For each item, get the last N quarters of 'Outgoing'
        # 3. Calculate Average
        # 4. Generate future quarters
        
        predictions = []
        
        # Helper to generate next quarters
        def get_next_quarters(start_quarter_str, n):
            start_q = pd.Period(start_quarter_str, freq='Q')
            return [str(start_q + i + 1) for i in range(n)]

        # Determine the global last quarter in data to project from
        # Note: In a real scenario, different items might stop at different times? 
        # Usually inventory data is snapshotted. Let's assume global max quarter.
        last_quarter = self.data['Quarter'].max()
        future_quarters = get_next_quarters(last_quarter, horizon)

        unique_items = self.data['Item Code'].unique()

        for item in unique_items:
            item_data = self.data[self.data['Item Code'] == item].sort_values('Quarter')
            
            # Get last N records
            recent_data = item_data.tail(self.window_size)
            
            # Simple Moving Average
            if len(recent_data) > 0:
                avg_demand = recent_data['Outgoing'].mean()
            else:
                avg_demand = 0
            
            # Create forecast rows
            for q in future_quarters:
                predictions.append({
                    'Item Code': item,
                    'Quarter': q,
                    'Forecasted_Outgoing': avg_demand,
                    'Product Description': item_data['Product Description'].iloc[0] if not item_data.empty else "Unknown",
                    'Unit': item_data['Unit'].iloc[0] if not item_data.empty else "pcs"
                })
        
        return pd.DataFrame(predictions)

class LinearTrendModel(ForecastingModel):
    def __init__(self):
        self.model = None
        self.data = None

    def fit(self, data: pd.DataFrame):
        self.data = data

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Model must be trained before prediction.")

        predictions = []
        
        # Helper for next quarters
        def get_next_quarters(start_quarter_str, n):
            start_q = pd.Period(start_quarter_str, freq='Q')
            return [str(start_q + i + 1) for i in range(n)]

        last_quarter = self.data['Quarter'].max()
        future_quarters = get_next_quarters(last_quarter, horizon)
        
        # Convert quarters to ordinal for regression (0, 1, 2...)
        # We need a consistent time index.
        # Get all unique quarters sorted
        all_quarters = sorted(self.data['Quarter'].unique())
        q_to_idx = {q: i for i, q in enumerate(all_quarters)}
        next_idx_start = len(all_quarters)

        unique_items = self.data['Item Code'].unique()

        for item in unique_items:
            item_data = self.data[self.data['Item Code'] == item].sort_values('Quarter')
            
            if len(item_data) < 2:
                # Fallback to mean if not enough points for trend
                avg = item_data['Outgoing'].mean() if not item_data.empty else 0
                y_pred = [avg] * horizon
            else:
                X = item_data['Quarter'].map(q_to_idx).values.reshape(-1, 1)
                y = item_data['Outgoing'].values
                
                reg = LinearRegression()
                reg.fit(X, y)
                
                # Predict future
                X_future = np.array(range(next_idx_start, next_idx_start + horizon)).reshape(-1, 1)
                y_pred = reg.predict(X_future)
                y_pred = np.maximum(y_pred, 0) # No negative inventory

            # Create forecast rows
            for i, q in enumerate(future_quarters):
                predictions.append({
                    'Item Code': item,
                    'Quarter': q,
                    'Forecasted_Outgoing': y_pred[i],
                    'Product Description': item_data['Product Description'].iloc[0] if not item_data.empty else "Unknown",
                    'Unit': item_data['Unit'].iloc[0] if not item_data.empty else "pcs"
                })
        
        return pd.DataFrame(predictions)

class ARIMAModel(ForecastingModel):
    def __init__(self, order=(1, 1, 0)):
        self.order = order
        self.data = None

    def fit(self, data: pd.DataFrame):
        self.data = data

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Model must be trained before prediction.")

        predictions = []
        
        # Helper
        def get_next_quarters(start_quarter_str, n):
            start_q = pd.Period(start_quarter_str, freq='Q')
            return [str(start_q + i + 1) for i in range(n)]

        last_quarter = self.data['Quarter'].max()
        future_quarters = get_next_quarters(last_quarter, horizon)

        unique_items = self.data['Item Code'].unique()
        
        # Suppress convergence warnings
        warnings.filterwarnings("ignore")

        for item in unique_items:
            item_data = self.data[self.data['Item Code'] == item].sort_values('Quarter')
            
            # ARIMA needs a series. 
            # Ideally we reindex to ensure no gaps, filling 0.
            # Convert to PeriodIndex
            ts = item_data.set_index('Quarter')['Outgoing']
            # Ensure index is sorted PeriodIndex?? 
            # Our 'Quarter' is string. 
            # Let's create a proper Time Series
            try:
                # Convert string '2024Q1' to period
                ts.index = pd.PeriodIndex(ts.index, freq='Q')
                ts = ts.sort_index()
                
                # Handle gaps?
                full_range = pd.period_range(start=ts.index.min(), end=ts.index.max(), freq='Q')
                ts = ts.reindex(full_range, fill_value=0)
                
                if len(ts) < 3:
                     # Fallback for too short history
                    forecast_values = [ts.mean()] * horizon
                else:
                    model = ARIMA(ts, order=self.order)
                    model_fit = model.fit()
                    forecast_res = model_fit.forecast(steps=horizon)
                    forecast_values = forecast_res.values
                    forecast_values = np.maximum(forecast_values, 0)

            except Exception as e:
                # Fallback on error (e.g., stationarity issues)
                # print(f"ARIMA error for {item}: {e}")
                forecast_values = [item_data['Outgoing'].mean()] * horizon

            # Create forecast rows
            for i, q in enumerate(future_quarters):
                predictions.append({
                    'Item Code': item,
                    'Quarter': q,
                    'Forecasted_Outgoing': forecast_values[i],
                    'Product Description': item_data['Product Description'].iloc[0] if not item_data.empty else "Unknown",
                    'Unit': item_data['Unit'].iloc[0] if not item_data.empty else "pcs"
                })
        
        return pd.DataFrame(predictions)

class EnsembleModel(ForecastingModel):
    def __init__(self):
        self.model = None
        self.data = None

    def fit(self, data: pd.DataFrame):
        self.data = data

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Model must be trained before prediction.")

        predictions = []
        
        # Helper
        def get_next_quarters(start_quarter_str, n):
            start_q = pd.Period(start_quarter_str, freq='Q')
            return [str(start_q + i + 1) for i in range(n)]

        last_quarter = self.data['Quarter'].max()
        future_quarters = get_next_quarters(last_quarter, horizon)
        
        # Consistent time index
        all_quarters = sorted(self.data['Quarter'].unique())
        q_to_idx = {q: i for i, q in enumerate(all_quarters)}
        next_idx_start = len(all_quarters)

        unique_items = self.data['Item Code'].unique()

        for item in unique_items:
            item_data = self.data[self.data['Item Code'] == item].sort_values('Quarter')
            
            if len(item_data) < 4:
                # XGBoost needs more data, fallback to Mean
                avg = item_data['Outgoing'].mean() if not item_data.empty else 0
                y_pred = [avg] * horizon
            else:
                # Features: Quarter Index, Previous Value (Lag 1)
                item_data['QuarterIdx'] = item_data['Quarter'].map(q_to_idx)
                item_data['Lag1'] = item_data['Outgoing'].shift(1).fillna(0)
                
                features = ['QuarterIdx', 'Lag1']
                X = item_data[features]
                y = item_data['Outgoing']
                
                model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
                model.fit(X, y)
                
                # Recursive Forecasting
                last_val = item_data['Outgoing'].iloc[-1]
                cur_idx = next_idx_start
                
                y_pred = []
                for _ in range(horizon):
                    # Create input for next step
                    next_input = pd.DataFrame([[cur_idx, last_val]], columns=features)
                    pred = model.predict(next_input)[0]
                    pred = max(0, pred) # significant
                    y_pred.append(pred)
                    
                    last_val = pred
                    cur_idx += 1

            # Create forecast rows
            for i, q in enumerate(future_quarters):
                predictions.append({
                    'Item Code': item,
                    'Quarter': q,
                    'Forecasted_Outgoing': y_pred[i],
                    'Product Description': item_data['Product Description'].iloc[0] if not item_data.empty else "Unknown",
                    'Unit': item_data['Unit'].iloc[0] if not item_data.empty else "pcs"
                })
        
        return pd.DataFrame(predictions)

class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class LSTMModel(ForecastingModel):
    def __init__(self, epochs=100):
        self.model = None
        self.epochs = epochs
        self.data = None

    def fit(self, data: pd.DataFrame):
        self.data = data
        self.warning = None
        self.debug_info = {}

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Model must be trained before prediction.")

        predictions = []
        
        # Helper
        def get_next_quarters(start_quarter_str, n):
            start_q = pd.Period(start_quarter_str, freq='Q')
            return [str(start_q + i + 1) for i in range(n)]

        last_quarter = self.data['Quarter'].max()
        future_quarters = get_next_quarters(last_quarter, horizon)
        
        unique_items = self.data['Item Code'].unique()
        
        for item in unique_items:
            item_data = self.data[self.data['Item Code'] == item].sort_values('Quarter')
            
            # Prepare data
            raw_data = item_data['Outgoing'].values.astype(float)
            
            if len(raw_data) < 4:
                # Fallback
                avg = np.mean(raw_data) if len(raw_data) > 0 else 0
                y_pred = [avg] * horizon
            else:
                # Normalize? For PoC maybe just raw scaling or min-max
                # Let's keep it simple: raw for now, or simple normalization
                max_val = np.max(raw_data)
                if max_val == 0:
                   train_data_norm = raw_data
                else:
                   train_data_norm = raw_data / max_val
                
                train_data_norm = torch.FloatTensor(train_data_norm).view(-1)
                
                # train_window
                train_window = 2 # Reduced from 4 for PoC with small data
                
                def create_inout_sequences(input_data, tw):
                    inout_seq = []
                    if len(input_data) <= tw:
                        return []
                    for i in range(len(input_data)-tw):
                        train_seq = input_data[i:i+tw]
                        train_label = input_data[i+tw:i+tw+1]
                        inout_seq.append((train_seq ,train_label))
                    return inout_seq

                train_inout_seq = create_inout_sequences(train_data_norm, train_window)
                
                if not train_inout_seq:
                     # Not enough data for window
                     self.warning = f"Not enough data for LSTM (Need > {train_window} quarters). Switched to Baseline."
                     avg = np.mean(raw_data)
                     y_pred = [avg] * horizon
                else:
                    model = LSTMNet()
                    loss_function = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    
                    # Training
                    for _ in range(self.epochs):
                        for seq, labels in train_inout_seq:
                            optimizer.zero_grad()
                            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                            torch.zeros(1, 1, model.hidden_layer_size))

                            y_pred_single = model(seq)

                            single_loss = loss_function(y_pred_single, labels)
                            single_loss.backward()
                            optimizer.step()
                    
                    # Predict
                    model.eval()
                    fut_pred_seq = train_data_norm[-train_window:].tolist()

                    for _ in range(horizon):
                        seq = torch.FloatTensor(fut_pred_seq[-train_window:])
                        with torch.no_grad():
                            next_val = model(seq).item()
                            fut_pred_seq.append(next_val)
                    
                    # Denormalize
                    if max_val != 0:
                        y_pred = [x * max_val for x in fut_pred_seq[-horizon:]]
                    else:
                        y_pred = fut_pred_seq[-horizon:]
                    
                    y_pred = [max(0, y) for y in y_pred]

            # Create forecast rows
            for i, q in enumerate(future_quarters):
                predictions.append({
                    'Item Code': item,
                    'Quarter': q,
                    'Forecasted_Outgoing': y_pred[i],
                    'Product Description': item_data['Product Description'].iloc[0] if not item_data.empty else "Unknown",
                    'Unit': item_data['Unit'].iloc[0] if not item_data.empty else "pcs"
                })

        return pd.DataFrame(predictions)

class LLMModel(ForecastingModel):
    def __init__(self):
        self.data = None
        self.client = None
        # Initialize client here or in fit/predict to load env
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
        else:
            print("Warning: OPENROUTER_API_KEY not found.")

    def fit(self, data: pd.DataFrame):
        self.data = data
        self.warning = None
        self.debug_info = {}

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Model must be trained before prediction.")
        
        predictions = []
        
        # Helper
        def get_next_quarters(start_quarter_str, n):
            start_q = pd.Period(start_quarter_str, freq='Q')
            return [str(start_q + i + 1) for i in range(n)]

        last_quarter = self.data['Quarter'].max()
        future_quarters = get_next_quarters(last_quarter, horizon)
        
        unique_items = self.data['Item Code'].unique()
        
        # Limit processing for PoC speed?
        # Let's process all but handle errors gracefully
        
        for item in unique_items:
            item_data = self.data[self.data['Item Code'] == item].sort_values('Quarter')
            history_str = item_data[['Quarter', 'Outgoing']].to_string(index=False)
            
            y_pred = []
            
            # Debug: Capture prompt
            prompt = ""
            response_content = ""

            if self.client:
                try:
                    # Very simple prompt for PoC
                    prompt = f"""
                    Forecast quantitative demand for item '{item}' ({item_data['Product Description'].iloc[0]}) for next {horizon} periods.
                    History:
                    {history_str}
                    
                    Return ONLY a JSON list of {horizon} integers. Example: [10, 11, 10, 12].
                    """
                    
                    model_name = os.getenv("LLM_MODEL_NAME", "deepseek/deepseek-r1:free")
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    response_content = response.choices[0].message.content
                    # Clean markdown code blocks if present
                    content_clean = response_content
                    if "```" in content_clean:
                        content_clean = content_clean.split("```")[1].replace("json", "").strip()
                    
                    y_pred = json.loads(content_clean)
                    
                    if not isinstance(y_pred, list):
                        y_pred = [float(x) for x in y_pred if str(x).isdigit()]
                        
                except Exception as e:
                    self.warning = f"LLM API Failed ({e}). SWITCHED TO SIMULATION MODE."
                    
                    # Generate Mock Prediction (Baseline + Random Noise) so it looks distinct
                    import random
                    avg_val = item_data['Outgoing'].mean() if not item_data.empty else 0
                    # +/- 20% random wobble
                    y_pred = [avg_val * (1 + random.uniform(-0.2, 0.2)) for _ in range(horizon)]
                    
                    self.debug_info = {
                        "last_prompt": prompt,
                        "last_response": f"❌ API ERROR: {e}\n\n🤖 SIMULATION MODE ACTIVE:\nGenerated random variation around mean ({avg_val:.2f}) to demonstrate workflow."
                    }
            
            # Fallback if LLM failed or returned bad data (and we didn't mock it above yet, e.g. json error)
            # If we mocked it above, y_pred is full.
            if len(y_pred) != horizon:
                if not self.warning:
                     self.warning = "LLM returned invalid format. Switched to Baseline."
                avg = item_data['Outgoing'].mean() if not item_data.empty else 0
                y_pred = [avg] * horizon

            # Create forecast rows
            for i, q in enumerate(future_quarters):
                val = y_pred[i] if i < len(y_pred) else 0
                predictions.append({
                    'Item Code': item,
                    'Quarter': q,
                    'Forecasted_Outgoing': float(val),
                    'Product Description': item_data['Product Description'].iloc[0] if not item_data.empty else "Unknown",
                    'Unit': item_data['Unit'].iloc[0] if not item_data.empty else "pcs"
                })

        return pd.DataFrame(predictions)

    def generate_executive_summary(self, risk_data: pd.DataFrame) -> str:
        """
        Generates a qualitative executive summary based on the risk analysis data.
        """
        if not self.client:
            return "AI Summary unavailable (API Key missing)."

        # Prepare context
        # summarize top 5 risks
        top_risks = risk_data.head(5).to_dict(orient='records')
        context_str = json.dumps(top_risks, indent=2)

        prompt = f"""
        Act as a supply chain executive. Review the following inventory risk data (Items with potential stockouts):
        {context_str}

        Write a concise, professional executive summary (max 150 words) suitable for a PDF report. 
        Highlight the most critical item and suggest a general procurement action.
        Do not use markdown formatting (no bold/italics), just plain text.
        """

        try:
            model_name = os.getenv("LLM_MODEL_NAME", "deepseek/deepseek-r1:free")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            # Cleanup deepseek thinking process if included (usually in <think> tags or just raw)
            # DeepSeek R1 often includes <think>...</think>. Let's remove it.
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            
            return content
        except Exception as e:
            return f"Error generating summary: {e}"

    def generate_manager_report(self, risk_data: pd.DataFrame) -> str:
        """
        Generates a detailed Manager-Level text report.
        """
        if not self.client:
            return "AI Report unavailable (API Key missing)."

        # Prepare context - Top 20 Shortages
        # We need to give the LLM enough info to say "Why" (Forecast vs Stock)
        # Format: Item Name (ID): Stock X, Forecast Y -> Shortage Z
        
        top_risks = risk_data.head(20).copy()
        top_risks['Info'] = top_risks.apply(
            lambda x: f"{x['Product Description']} ({x['Item Code']}): Current Stock {x['Closing Balance']:.0f}, Need {x['Forecasted_Outgoing']:.0f} -> BUY {x['Gap']:.0f}", 
            axis=1
        )
        context_str = "\n".join(top_risks['Info'].tolist())

        prompt = f"""
        Act as an Inventory Manager for a medical lab. Write a detailed "Inventory Risk & Procurement Report" based on this shortage data:
        
        {context_str}
        
        Structure the report as follows:
        1. **Executive Overview**: High-level summary of the situation (e.g. "We are facing shortages in X critical items...").
        2. **Critical Action Items**: List the top 3-5 most urgent items to buy, explaining WHY (e.g. "High demand vs low stock"). 
        3. **Procurement Strategy**: Recommendations for the purchasing team (e.g. "Immediate expedite needed for...", "Review safety stock for...").
        
        Keep the tone professional and actionable. Do not use markdown (no bold/italics), just clear plain text with spacing.
        """

        try:
            model_name = os.getenv("LLM_MODEL_NAME", "deepseek/deepseek-r1:free")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            
            # Clean think tags
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
                
            return content
        except Exception as e:
            return f"Error generating report: {e}"
