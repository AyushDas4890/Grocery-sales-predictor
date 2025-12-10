import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

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
        background: linear-gradient(to right bottom, #f8f9fa, #e9ecef);
        font-family: 'Inter', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Cards/Metrics */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        color: #718096;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2b6cb0;
        color: white;
        border-radius: 8px;
        height: 48px;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2c5282;
        box-shadow: 0 4px 12px rgba(43, 108, 176, 0.3);
    }
    
    /* Prediction Box */
    .prediction-box-high {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.4);
    }
    .prediction-box-low {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Processing Functions
# -----------------------------------------------------------------------------

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
    st.image("https://cdn-icons-png.flaticon.com/512/3759/3759997.png", width=80) 
    st.title("Settings")
    
    st.markdown("### 🛠 Configuration")
    data_file = "Groceries_dataset.csv"
    
    if st.checkbox("Show Raw Data"):
        st.session_state['show_data'] = True
    else:
        st.session_state['show_data'] = False
        
    st.info("Uses Gradient Boosting Classifier for prediction.")
    st.write("---")
    st.caption("v1.0.0 | RetailPulse AI")

# Main Content
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Grocery Sales Predictor")
    st.markdown("Forecast daily demand levels with precision using AI.")

# Load Data & Train
try:
    with st.spinner("Initializing AI Engine..."):
        daily_sales = load_data(data_file)
        model, scaler, feature_cols, threshold, le = train_model(daily_sales)
        
        last_date = daily_sales['Date'].max()
        start_date = daily_sales['Date'].min()
        
    # KPIs
    st.markdown("### 📊 Historical Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(daily_sales)}</div><div class='metric-label'>Total Days</div></div>", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{daily_sales['Sales_Count'].mean():.0f}</div><div class='metric-label'>Avg Daily Sales</div></div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{daily_sales['Sales_Count'].max()}</div><div class='metric-label'>Peak Sales</div></div>", unsafe_allow_html=True)
    with kpi4:
         st.markdown(f"<div class='metric-card'><div class='metric-value'>{threshold:.0f}</div><div class='metric-label'>High Demand Threshold</div></div>", unsafe_allow_html=True)

    # Chart
    st.markdown("### 📈 Sales Trends")
    fig = px.line(daily_sales, x='Date', y='Sales_Count', title='Daily Sales Volume')
    fig.update_layout(xaxis_title="Date", yaxis_title="Sales Count", template="plotly_white", height=350)
    fig.update_traces(line_color='#3182ce')
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get('show_data'):
        st.dataframe(daily_sales.tail(10))

    # Prediction Interface
    st.markdown("---")
    st.markdown("### 🔮 Future Prediction")
    
    p_col1, p_col2 = st.columns([1, 1])
    
    with p_col1:
        st.markdown("Select a date to forecast demand:")
        default_date = last_date + timedelta(days=1)
        target_date = st.date_input("Target Date", value=default_date, min_value=last_date + timedelta(days=1))
        
        if st.button("Generate Forecast", use_container_width=True):
            # Prediction Logic
            t_date = pd.to_datetime(target_date)
            
            # Prepare single row input
            # Logic: We imply "persistence" of the last known stats for future dates mostly
            # but getting the correct day/month/season is crucial.
            month = t_date.month
            day = t_date.day
            day_of_week = t_date.dayofweek
            year = t_date.year
            quarter = t_date.quarter
            is_weekend = 1 if day_of_week >= 5 else 0
            season_code = get_season(month)

            # Get recent data for lags (naive: using last available data)
            recent_data = daily_sales.tail(30)
            sales_values = recent_data['Sales_Count'].values
            
            lags = {}
            for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
                valid_lag = sales_values[-lag] if len(sales_values) >= lag else sales_values[-1]
                lags[f'Sales_Lag_{lag}'] = valid_lag
                
            rolling_stats = {}
            for window in [3, 7, 14, 30]:
                rolling_stats[f'Rolling_Mean_{window}'] = np.mean(sales_values[-window:])
                rolling_stats[f'Rolling_Std_{window}'] = np.std(sales_values[-window:])
            
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
            
            with p_col2:
                st.markdown(f"<div style='margin-top: 28px'></div>", unsafe_allow_html=True)
                if predicted_label == 'High':
                    st.markdown(f"""
                        <div class='prediction-box-high'>
                            <h2 style='color: white; margin:0'>HIGH DEMAND</h2>
                            <p style='margin:0; opacity: 0.9'>Confidence: {confidence:.1%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='prediction-box-low'>
                            <h2 style='color: white; margin:0'>LOW DEMAND</h2>
                            <p style='margin:0; opacity: 0.9'>Confidence: {confidence:.1%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
            st.success("Prediction generated successfully.")
            
            # Explainability
            with st.expander("Why this result? (Feature Inputs)"):
                st.write(features)

except FileNotFoundError:
    st.error("Error: 'Groceries_dataset.csv' not found. Please ensure the dataset is in the same directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")
