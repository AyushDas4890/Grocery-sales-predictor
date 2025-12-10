import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from streamlit_lottie import st_lottie
import os

# Page Configuration
st.set_page_config(
    page_title="RetailPulse | Demand Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headings - Force Dark Color */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a202c !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Text - Force Dark Color */
    p, .stMarkdown p, li, .stMarkdown li {
        color: #4a5568 !important;
    }

    /* Cards/Metrics */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2b6cb0 !important;
    }
    .metric-label {
        color: #718096 !important;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2b6cb0;
        color: white !important;
        border-radius: 8px;
        height: 48px;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2c5282;
        box-shadow: 0 4px 12px rgba(43, 108, 176, 0.3);
        color: white !important;
    }
    
    /* Fix for Plotly Chart Background */
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }
    
    /* Banner Image */
    .hero-banner {
        width: 100%;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the groceries dataset."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Aggregating daily sales
    daily_sales = df.groupby('Date').size().reset_index(name='Sales_Count')
    daily_sales = daily_sales.sort_values('Date')
    return daily_sales

def get_season(month):
    if month in [12, 1, 2]: return 3 # Winter
    elif month in [3, 4, 5]: return 2 # Spring
    elif month in [6, 7, 8]: return 1 # Summer
    else: return 0 # Fall

def get_season_name(season_code):
    seasons = {3: 'Winter ❄️', 2: 'Spring 🌸', 1: 'Summer ☀️', 0: 'Fall 🍂'}
    return seasons.get(season_code, 'Unknown')

@st.cache_resource
def train_model(daily_sales):
    """Features engineering and Model training."""
    
    # Feature Engineering
    df_model = daily_sales.copy()
    df_model['Month'] = df_model['Date'].dt.month
    df_model['Day'] = df_model['Date'].dt.day
    df_model['DayOfWeek'] = df_model['Date'].dt.dayofweek
    df_model['Year'] = df_model['Date'].dt.year
    df_model['Quarter'] = df_model['Date'].dt.quarter
    df_model['Is_Weekend'] = df_model['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df_model['Season_Code'] = df_model['Month'].apply(get_season)

    # Lag features
    for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
        df_model[f'Sales_Lag_{lag}'] = df_model['Sales_Count'].shift(lag)

    # Rolling stats
    for window in [3, 7, 14, 30]:
        df_model[f'Rolling_Mean_{window}'] = df_model['Sales_Count'].rolling(window=window).mean()
        df_model[f'Rolling_Std_{window}'] = df_model['Sales_Count'].rolling(window=window).std()

    # Drop NaN
    df_model = df_model.dropna()

    # Create Target
    # High demand if sales > median
    threshold = df_model['Sales_Count'].median()
    df_model['Demand_Level'] = df_model['Sales_Count'].apply(lambda x: 'High' if x > threshold else 'Low')
    
    le = LabelEncoder()
    # High=0 (H < L), Low=1
    df_model['Target'] = le.fit_transform(df_model['Demand_Level'])

    feature_cols = [
        'Month', 'Day', 'DayOfWeek', 'Year', 'Quarter', 'Is_Weekend', 'Season_Code',
        'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Sales_Lag_4', 'Sales_Lag_5', 'Sales_Lag_6', 'Sales_Lag_7', 'Sales_Lag_14',
        'Rolling_Mean_3', 'Rolling_Mean_7', 'Rolling_Mean_14', 'Rolling_Mean_30',
        'Rolling_Std_3', 'Rolling_Std_7', 'Rolling_Std_14', 'Rolling_Std_30'
    ]

    X = df_model[feature_cols]
    y = df_model['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    model.fit(X_scaled, y)

    return model, scaler, feature_cols, threshold, le

# -----------------------------------------------------------------------------
# Main App Layout
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3759/3759997.png", width=60) 
    st.title("Grocery AI")
    
    st.markdown("### 🛠 Configuration")
    data_file = "Groceries_dataset.csv"
    
    if st.checkbox("Show Raw Data"):
        st.session_state['show_data'] = True
    else:
        st.session_state['show_data'] = False
    
    st.write("---")
    
    # Simple logic for sidebar animation
    lottie_sidebar = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5nzkkqwe.json")
    if lottie_sidebar:
        st_lottie(lottie_sidebar, height=150, key="sidebar_anim")
        
    st.info("Gradient Boosting Model Active")
    st.caption("v2.0.0 | RetailPulse AI")

# Main Content
# Header Image
if os.path.exists("banner.png"):
    st.image("banner.png", use_container_width=True, output_format="PNG", caption=None)
else:
    # Fallback to a placeholder gradient div if image not found
    st.markdown("""<div style='background: linear-gradient(90deg, #2b6cb0 0%, #2c5282 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'><h1>Grocery Sales AI</h1></div>""", unsafe_allow_html=True)


col1, col2 = st.columns([3, 1])
with col1:
    st.title("Demand Forecast Dashboard")
    st.markdown("Leverage AI to predict daily sales demand with high precision. Optimize your inventory today.")

with col2:
    # Small animation next to title
    lottie_cart = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json") 
    if lottie_cart:
        st_lottie(lottie_cart, height=100, key="cart_anim")

# Load Data & Train
try:
    with st.spinner("Crunching numbers & Training AI..."):
        daily_sales = load_data(data_file)
        model, scaler, feature_cols, threshold, le = train_model(daily_sales)
        
        last_date = daily_sales['Date'].max()
        
    # KPIs
    st.markdown("### 📊 Historical Snapshot")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(daily_sales)}</div><div class='metric-label'>Total Days</div></div>", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{daily_sales['Sales_Count'].mean():.0f}</div><div class='metric-label'>Avg Daily Sales</div></div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{daily_sales['Sales_Count'].max()}</div><div class='metric-label'>Peak Sales</div></div>", unsafe_allow_html=True)
    with kpi4:
         st.markdown(f"<div class='metric-card'><div class='metric-value'>{threshold:.0f}</div><div class='metric-label'>High Threshold</div></div>", unsafe_allow_html=True)

    # Chart
    st.markdown("### 📈 Intelligent Trend Analysis")
    fig = px.area(daily_sales, x='Date', y='Sales_Count', title='Sales Volume Over Time')
    fig.update_layout(
        xaxis_title="Date", 
        yaxis_title="Sales Count", 
        template="plotly_white", 
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0')
    )
    fig.update_traces(line_color='#2b6cb0', fillcolor='rgba(43, 108, 176, 0.1)')
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get('show_data'):
        st.dataframe(daily_sales.tail(10))

    # Prediction Interface
    st.markdown("---")
    st.subheader("🔮 Predictive Analytics Engine")
    
    p_col1, p_col2 = st.columns([1, 1])
    
    with p_col1:
        st.markdown("##### 📅 Select Forecast Date")
        default_date = last_date + timedelta(days=1)
        target_date = st.date_input("Target Date", value=default_date, min_value=last_date + timedelta(days=1))
        
        st.markdown("")
        if st.button("Generate Forecast 🚀", use_container_width=True):
            # Prediction Logic
            t_date = pd.to_datetime(target_date)
            
            # Prepare features
            month = t_date.month
            day = t_date.day
            day_of_week = t_date.dayofweek
            year = t_date.year
            quarter = t_date.quarter
            is_weekend = 1 if day_of_week >= 5 else 0
            season_code = get_season(month)

            # ---------------------------------------------------------
            # Historical Parallel Mapping (No Synthetic Data)
            # ---------------------------------------------------------
            # To avoid "hallucinating" random numbers, we use actual historical records.
            # We find a date in the past that closely matches the target date's 
            # seasonality (Month) and Day of Week.
            
            # 1. Filter history for same Month and Day of Week
            same_mo_dow = daily_sales[
                (daily_sales['Date'].dt.month == month) & 
                (daily_sales['Date'].dt.day_of_week == day_of_week)
            ]
            
            # 2. If matches found, take the most recent one. Else fallback to same Month only.
            if not same_mo_dow.empty:
                proxy_date = same_mo_dow['Date'].max()
            else:
                # Fallback: Just same month
                same_mo = daily_sales[daily_sales['Date'].dt.month == month]
                if not same_mo.empty:
                    proxy_date = same_mo['Date'].max()
                else:
                    # Fallback: Last available date in dataset
                    proxy_date = daily_sales['Date'].max()
            
            # 3. Get the Sales/Lags associated with that REAL historical date
            # We essentially say "The market conditions for [Target Date] will be similar to [Proxy Date]"
            
            # We need to find the index of this proxy date to get its rolling stats/lags
            proxy_idx = daily_sales[daily_sales['Date'] == proxy_date].index[0]
            
            # We can't just take the row because we need to re-calculate features if we wanted to be 100% precise,
            # but picking the pre-calculated features for that date is the most grounded approach.
            # However, our daily_sales df here only has 'Sales_Count'. We need to re-derive features.
            
            # Let's get the slice of data leading up to that proxy date
            # Ensure we have enough history before the proxy date
            if proxy_idx < 30:
                recent_data_proxy = daily_sales.iloc[0 : proxy_idx+1]
            else:
                recent_data_proxy = daily_sales.iloc[proxy_idx-30 : proxy_idx+1]
                
            sales_values = recent_data_proxy['Sales_Count'].values
            
            lags = {}
            for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
                valid_lag = sales_values[-lag-1] if len(sales_values) > lag else sales_values[0]
                lags[f'Sales_Lag_{lag}'] = valid_lag
                
            rolling_stats = {}
            for window in [3, 7, 14, 30]:
                # Rolling stat up to the day BEFORE the proxy date (since we are predicting for the proxy date equivalent)
                # But actually, input features for time T usually include rolling averages of T-1.
                # So we take the window ending at -1.
                subset = sales_values[:-1] # Exclude the target day itself
                if len(subset) == 0: subset = [0]
                
                rolling_stats[f'Rolling_Mean_{window}'] = np.mean(subset[-window:])
                rolling_stats[f'Rolling_Std_{window}'] = np.std(subset[-window:])
            
            features = {
                'Month': month,
                'Day': day,
                'DayOfWeek': day_of_week,
                'Year': year,
                'Quarter': quarter,
                'Is_Weekend': is_weekend,
                'Season_Code': season_code,
                **lags,
                **rolling_stats
            }
            
            input_df = pd.DataFrame([features])[feature_cols]
            input_scaled = scaler.transform(input_df)
            
            prediction_code = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            predicted_label = le.inverse_transform([prediction_code])[0]
            confidence = prediction_proba[prediction_code]
            
            # Visualize Result
            with p_col2:
                # Gauge Chart for Probability
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    title = {'text': f"Prediction: {predicted_label.upper()} DEMAND"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#10B981" if predicted_label=='High' else "#EF4444"},
                        'steps': [
                            {'range': [0, 50], 'color': "#f7fafc"},
                            {'range': [50, 100], 'color': "#edf2f7"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "#1a202c"})
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Context
                st.markdown(f"**Season:** {get_season_name(season_code)}")
                st.markdown(f"**Recent Trend (7d):** {rolling_stats['Rolling_Mean_7']:.1f} sales/day")
            
            st.success("Analysis Complete.")
            
            with st.expander("Show Technical Feature Vector"):
                st.json(features)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #718096 !important;'>© 2025 RetailPulse AI | Powered by Gradient Boosting</p>", unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Error: 'Groceries_dataset.csv' not found. Please ensure the dataset is in the same directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")
