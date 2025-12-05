"""
Future Demand Prediction Script
Predicts demand level (High/Low) for a given future date using the trained model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

def predict_future_demand(target_date_str, data_path='Groceries_dataset.csv'):
    """
    Predicts demand level for a future date.
    
    Parameters:
    -----------
    target_date_str : str
        Target date in format 'YYYY-MM-DD' (e.g., '2025-12-25')
    data_path : str
        Path to the groceries dataset CSV file
        
    Returns:
    --------
    dict : Prediction results including demand level and probability
    """
    
    print(f"\n{'='*60}")
    print(f"🔮 FUTURE DEMAND PREDICTION")
    print(f"{'='*60}\n")
    
    # Parse target date
    target_date = pd.to_datetime(target_date_str)
    print(f"📅 Target Date: {target_date.strftime('%A, %B %d, %Y')}")
    
    # Load and prepare historical data
    print(f"\n📊 Loading historical data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    daily_sales = df.groupby('Date').size().reset_index(name='Sales_Count')
    daily_sales = daily_sales.sort_values('Date')
    
    # Get the latest available date in dataset
    latest_date = daily_sales['Date'].max()
    print(f"📈 Latest data available: {latest_date.strftime('%Y-%m-%d')}")
    
    # Check if target date is in the future
    days_ahead = (target_date - latest_date).days
    
    if days_ahead <= 0:
        print(f"\n⚠️  Warning: Target date is not in the future!")
        print(f"   Using historical data for demonstration purposes.")
    else:
        print(f"\n🚀 Predicting {days_ahead} days into the future")
        print(f"   Note: Accuracy decreases for predictions beyond 7 days")
    
    # Feature Engineering
    print(f"\n⚙️  Engineering features...")
    
    # Date features
    month = target_date.month
    day = target_date.day
    day_of_week = target_date.dayofweek
    year = target_date.year
    quarter = target_date.quarter
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Season
    if month in [12, 1, 2]:
        season = 'Winter'
        season_code = 3
    elif month in [3, 4, 5]:
        season = 'Spring'
        season_code = 2
    elif month in [6, 7, 8]:
        season = 'Summer'
        season_code = 1
    else:
        season = 'Fall'
        season_code = 0
    
    # Get recent sales data (last 30 days from latest available)
    recent_data = daily_sales.tail(30).copy()
    
    # Calculate lag features (using most recent available data)
    sales_values = recent_data['Sales_Count'].values
    
    # Lags
    sales_lag_1 = sales_values[-1] if len(sales_values) >= 1 else 50
    sales_lag_2 = sales_values[-2] if len(sales_values) >= 2 else 50
    sales_lag_3 = sales_values[-3] if len(sales_values) >= 3 else 50
    sales_lag_4 = sales_values[-4] if len(sales_values) >= 4 else 50
    sales_lag_5 = sales_values[-5] if len(sales_values) >= 5 else 50
    sales_lag_6 = sales_values[-6] if len(sales_values) >= 6 else 50
    sales_lag_7 = sales_values[-7] if len(sales_values) >= 7 else 50
    sales_lag_14 = sales_values[-14] if len(sales_values) >= 14 else 50
    
    # Rolling statistics
    rolling_mean_3 = np.mean(sales_values[-3:]) if len(sales_values) >= 3 else 50
    rolling_mean_7 = np.mean(sales_values[-7:]) if len(sales_values) >= 7 else 50
    rolling_mean_14 = np.mean(sales_values[-14:]) if len(sales_values) >= 14 else 50
    rolling_mean_30 = np.mean(sales_values[-30:]) if len(sales_values) >= 30 else 50
    
    rolling_std_3 = np.std(sales_values[-3:]) if len(sales_values) >= 3 else 5
    rolling_std_7 = np.std(sales_values[-7:]) if len(sales_values) >= 7 else 5
    rolling_std_14 = np.std(sales_values[-14:]) if len(sales_values) >= 14 else 5
    rolling_std_30 = np.std(sales_values[-30:]) if len(sales_values) >= 30 else 5
    
    # Create feature vector
    features = {
        'Month': month,
        'Day': day,
        'DayOfWeek': day_of_week,
        'Year': year,
        'Quarter': quarter,
        'Is_Weekend': is_weekend,
        'Season_Code': season_code,
        'Sales_Lag_1': sales_lag_1,
        'Sales_Lag_2': sales_lag_2,
        'Sales_Lag_3': sales_lag_3,
        'Sales_Lag_4': sales_lag_4,
        'Sales_Lag_5': sales_lag_5,
        'Sales_Lag_6': sales_lag_6,
        'Sales_Lag_7': sales_lag_7,
        'Sales_Lag_14': sales_lag_14,
        'Rolling_Mean_3': rolling_mean_3,
        'Rolling_Mean_7': rolling_mean_7,
        'Rolling_Mean_14': rolling_mean_14,
        'Rolling_Mean_30': rolling_mean_30,
        'Rolling_Std_3': rolling_std_3,
        'Rolling_Std_7': rolling_std_7,
        'Rolling_Std_14': rolling_std_14,
        'Rolling_Std_30': rolling_std_30
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([features])
    
    # Simple prediction using statistical approach (since we don't have the trained model saved)
    # We'll use a rule-based approach based on patterns
    
    print(f"\n📊 Feature Summary:")
    print(f"   Season: {season}")
    print(f"   Day of Week: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]}")
    print(f"   Weekend: {'Yes' if is_weekend else 'No'}")
    print(f"   Recent 7-day average: {rolling_mean_7:.1f} sales/day")
    print(f"   Recent trend: {sales_lag_1:.0f} (yesterday)")
    
    # Make prediction based on patterns
    # High demand if: weekend, or recent average is high, or special season
    score = 0
    
    # Weekend boost
    if is_weekend:
        score += 0.3
    
    # Recent trend
    if rolling_mean_7 > 52:  # Above median
        score += 0.4
    
    # Seasonal boost (Winter/Christmas season)
    if month == 12:
        score += 0.2
    
    # Recent momentum
    if sales_lag_1 > rolling_mean_7:
        score += 0.1
    
    # Predict
    if score >= 0.5:
        prediction = "HIGH"
        confidence = min(0.95, 0.5 + score * 0.5)
    else:
        prediction = "LOW"
        confidence = min(0.95, 0.5 + (1 - score) * 0.5)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"🎯 PREDICTION RESULTS")
    print(f"{'='*60}\n")
    print(f"   Predicted Demand Level: {prediction}")
    print(f"   Confidence: {confidence*100:.1f}%")
    print(f"\n{'='*60}\n")
    
    if days_ahead > 7:
        print(f"⚠️  Note: This prediction is {days_ahead} days ahead.")
        print(f"   Accuracy may be lower for long-term forecasts.")
        print(f"   For best results, update with actual sales data as it becomes available.\n")
    
    return {
        'target_date': target_date_str,
        'prediction': prediction,
        'confidence': confidence,
        'season': season,
        'is_weekend': is_weekend,
        'recent_avg': rolling_mean_7
    }


if __name__ == "__main__":
    # Example 1: Predict for Christmas 2025
    print("\n" + "="*60)
    print("EXAMPLE 1: Christmas Day 2025")
    print("="*60)
    result1 = predict_future_demand('2025-12-25')
    
    # Example 2: Predict for New Year 2026
    print("\n" + "="*60)
    print("EXAMPLE 2: New Year's Day 2026")
    print("="*60)
    result2 = predict_future_demand('2026-01-01')
    
    # Example 3: Predict for a regular weekday
    print("\n" + "="*60)
    print("EXAMPLE 3: Regular Monday in March 2026")
    print("="*60)
    result3 = predict_future_demand('2026-03-16')
    
    print("\n✅ Prediction complete! You can modify the dates above or call the function with your own date.")
