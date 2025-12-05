
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# 1. Load and Preprocess Data
print("Loading data...")
df = pd.read_csv('Groceries_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
daily_sales = df.groupby('Date').size().reset_index(name='Sales_Count')
daily_sales = daily_sales.sort_values('Date')

# 2. Feature Engineering
print("Engineering features...")
daily_sales['Month'] = daily_sales['Date'].dt.month
daily_sales['Day'] = daily_sales['Date'].dt.day
daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
daily_sales['Year'] = daily_sales['Date'].dt.year
daily_sales['Quarter'] = daily_sales['Date'].dt.quarter
daily_sales['Is_Weekend'] = daily_sales['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

def get_season(month):
    if month in [12, 1, 2]: return 3 # Winter
    elif month in [3, 4, 5]: return 2 # Spring
    elif month in [6, 7, 8]: return 1 # Summer
    else: return 0 # Fall

daily_sales['Season_Code'] = daily_sales['Month'].apply(get_season)

# Lag features
for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
    daily_sales[f'Sales_Lag_{lag}'] = daily_sales['Sales_Count'].shift(lag)

# Rolling stats
for window in [3, 7, 14, 30]:
    daily_sales[f'Rolling_Mean_{window}'] = daily_sales['Sales_Count'].rolling(window=window).mean()
    daily_sales[f'Rolling_Std_{window}'] = daily_sales['Sales_Count'].rolling(window=window).std()

# Drop NaN
daily_sales = daily_sales.dropna()

# Label creation (Target)
# Assuming 'High' (0) if Sales > Threshold, 'Low' (1) otherwise. 
# Need to infer threshold or use median.
# NOTE: notebook_prediction_code.py says: 
# "Decode prediction (0=High, 1=Low based on alphabetical LabelEncoder)"
# But wait, it says "predicted_label = "High" if prediction_code == 0 else "Low""
# This implies 0 maps to High.
# Let's create the target variable.
threshold = daily_sales['Sales_Count'].median()
daily_sales['Demand_Level'] = daily_sales['Sales_Count'].apply(lambda x: 'High' if x > threshold else 'Low')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# If we fit LE on ['High', 'Low'], High(0), Low(1) because H comes before L.
daily_sales['Target'] = le.fit_transform(daily_sales['Demand_Level'])

# Define feature columns
feature_cols = [
    'Month', 'Day', 'DayOfWeek', 'Year', 'Quarter', 'Is_Weekend', 'Season_Code',
    'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Sales_Lag_4', 'Sales_Lag_5', 'Sales_Lag_6', 'Sales_Lag_7', 'Sales_Lag_14',
    'Rolling_Mean_3', 'Rolling_Mean_7', 'Rolling_Mean_14', 'Rolling_Mean_30',
    'Rolling_Std_3', 'Rolling_Std_7', 'Rolling_Std_14', 'Rolling_Std_30'
]

X = daily_sales[feature_cols]
y = daily_sales['Target']

# 3. Train Model
print("Training model...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# We use all data for training to get best prediction for future
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_scaled, y)

print(f"Model trained. Accuracy on training set: {gb_model.score(X_scaled, y):.4f}")


# 4. Prediction Function (adapted from notebook_prediction_code.py)
def predict_future_demand(target_date_str, daily_sales_df, scaler, gb_model, feature_cols):
    print(f"\n{'='*70}")
    print(f" FUTURE DEMAND PREDICTION")
    print(f"{'='*70}\n")
    
    target_date = pd.to_datetime(target_date_str)
    print(f" Target Date: {target_date.strftime('%A, %B %d, %Y')}")
    
    latest_date = daily_sales_df['Date'].max()
    print(f" Latest data available: {latest_date.strftime('%Y-%m-%d')}")
    
    # Feature extraction logic for TARGET DATE
    month = target_date.month
    day = target_date.day
    day_of_week = target_date.dayofweek
    year = target_date.year
    quarter = target_date.quarter
    is_weekend = 1 if day_of_week >= 5 else 0
    season_code = get_season(month)
    
    # We need recent sales values relative to the TARGET DATE.
    # If target date is far in future, this naive approach of taking "last available data" is wrong if we want real forecast.
    # BUT, the user prompt implies: "predict ... 1 january 2016".
    # If the dataset ends in 2015, then 2016-01-01 is truly the "Next" step (or close to it).
    # Let's check the gap.
    days_gap = (target_date - latest_date).days
    
    # Get recent data. If predicting Jan 1 2016, and data ends Dec 30 2015, we use that end data.
    # For Lag features, we need the sales at T-1, T-2... relative to Target Date.
    # If Target=Jan 1, 2016. T-1 = Dec 31, 2015.
    # If Dec 31 is in df, use it. If not?
    # The 'notebook_prediction_code.py' uses `data.tail(30)` which implies it ALWAYS uses the absolute last known data 
    # regardless of how far the prediction date is. This is a simplification ("Last values carry forward for lags").
    # We will stick to that logic as it's what was in the provided code snippet.
    
    recent_data = daily_sales_df.tail(30)
    sales_values = recent_data['Sales_Count'].values
    
    lags = {}
    for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
        # Logic: use the value at index -lag. 
        lags[f'Sales_Lag_{lag}'] = sales_values[-lag] if len(sales_values) >= lag else sales_values[-1]
        
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
    
    prediction_code = gb_model.predict(input_scaled)[0]
    prediction_proba = gb_model.predict_proba(input_scaled)[0]
    
    predicted_label = "High" if prediction_code == 0 else "Low"
    
    print(f"\n Context:")
    print(f"   Season Code: {season_code}")
    print(f"   Recent 7-day average: {rolling_stats['Rolling_Mean_7']:.1f}")
    
    print(f"\n{'='*70}")
    print(f" PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"PREDICTION: {predicted_label}")
    print(f"CONFIDENCE: {prediction_proba}")

# 5. RUN PREDICTION
predict_future_demand('2016-01-01', daily_sales, scaler, gb_model, feature_cols)
print("SCRIPT_COMPLETED")
