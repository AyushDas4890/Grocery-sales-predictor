"""
INSTRUCTIONS: Copy the code below and paste it as a NEW CELL at the end of your 
Seasonal_Demand_Prediction.ipynb notebook.

This will add the Future Demand Prediction feature to your notebook.
"""

# ============================================================================
# COPY EVERYTHING BELOW THIS LINE INTO A NEW NOTEBOOK CELL
# ============================================================================

## 7. Future Demand Prediction

def predict_future_demand(target_date_str, daily_sales_df, scaler, gb_model):
    """
    Predicts demand level for a future date.
    
    Parameters:
    -----------
    target_date_str : str
        Target date in format 'YYYY-MM-DD' (e.g., '2025-12-25')
    daily_sales_df : DataFrame
        The processed daily_sales dataframe with all features
    scaler : StandardScaler
        The fitted scaler object
    gb_model : GradientBoostingClassifier
        The trained Gradient Boosting model
        
    Returns:
    --------
    dict : Prediction results
    """
    
    print(f"\n{'='*70}")
    print(f"🔮 FUTURE DEMAND PREDICTION")
    print(f"{'='*70}\n")
    
    # Parse target date
    target_date = pd.to_datetime(target_date_str)
    print(f"📅 Target Date: {target_date.strftime('%A, %B %d, %Y')}")
    
    # Get the latest available date
    latest_date = daily_sales_df['Date'].max()
    print(f"📈 Latest data available: {latest_date.strftime('%Y-%m-%d')}")
    
    days_ahead = (target_date - latest_date).days
    
    if days_ahead > 0:
        print(f"🚀 Predicting {days_ahead} days into the future")
        if days_ahead > 7:
            print(f"⚠️  Note: Accuracy decreases for predictions beyond 7 days")
    
    # Extract date features
    month = target_date.month
    day = target_date.day
    day_of_week = target_date.dayofweek
    year = target_date.year
    quarter = target_date.quarter
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Season encoding
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
    
    # Get recent sales data (last 30 rows)
    recent_data = daily_sales_df.tail(30)
    sales_values = recent_data['Sales_Count'].values
    
    # Calculate lag features
    lags = {}
    for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
        lags[f'Sales_Lag_{lag}'] = sales_values[-lag] if len(sales_values) >= lag else sales_values[-1]
    
    # Calculate rolling statistics
    rolling_stats = {}
    for window in [3, 7, 14, 30]:
        rolling_stats[f'Rolling_Mean_{window}'] = np.mean(sales_values[-window:]) if len(sales_values) >= window else np.mean(sales_values)
        rolling_stats[f'Rolling_Std_{window}'] = np.std(sales_values[-window:]) if len(sales_values) >= window else np.std(sales_values)
    
    # Create feature dictionary
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
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([features])[feature_cols]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_code = gb_model.predict(input_scaled)[0]
    prediction_proba = gb_model.predict_proba(input_scaled)[0]
    
    # Decode prediction (0=High, 1=Low based on alphabetical LabelEncoder)
    predicted_label = "High" if prediction_code == 0 else "Low"
    confidence = prediction_proba[prediction_code]
    
    # Display results
    print(f"\n📊 Context:")
    print(f"   Season: {season}")
    print(f"   Day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]}")
    print(f"   Weekend: {'Yes' if is_weekend else 'No'}")
    print(f"   Recent 7-day average: {rolling_stats['Rolling_Mean_7']:.1f} sales/day")
    
    print(f"\n{'='*70}")
    print(f"🎯 PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"   Predicted Demand Level: {predicted_label.upper()}")
    print(f"   Confidence: {confidence*100:.1f}%")
    print(f"   Probability (High): {prediction_proba[0]*100:.1f}%")
    print(f"   Probability (Low): {prediction_proba[1]*100:.1f}%")
    print(f"{'='*70}\n")
    
    return {
        'date': target_date_str,
        'prediction': predicted_label,
        'confidence': confidence,
        'probabilities': {'High': prediction_proba[0], 'Low': prediction_proba[1]}
    }


# Example Predictions

print("\n" + "="*70)
print("EXAMPLE PREDICTIONS FOR FUTURE DATES")
print("="*70)

# Example 1: 1 January 2016
result1 = predict_future_demand('2016-01-01', daily_sales, scaler, gb_clf)

# Example 2: New Year 2026
result2 = predict_future_demand('2026-01-01', daily_sales, scaler, gb_clf)

# Example 3: Regular weekday in summer
result3 = predict_future_demand('2026-07-15', daily_sales, scaler, gb_clf)

print("\n✅ You can now predict demand for any future date!")
print("   Usage: predict_future_demand('YYYY-MM-DD', daily_sales, scaler, gb_clf)")
