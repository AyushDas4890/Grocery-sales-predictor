# Seasonal Product Demand Prediction 

## 📌 Project Overview
This project is a comprehensive Data Science Proof of Concept (POC) designed to predict the **Seasonal Demand Level** (High/Low) for grocery products. By leveraging historical sales data and advanced machine learning techniques, we aim to provide actionable insights for better inventory management and supply chain optimization.

## 🌐 Live Application
Access the live demand prediction dashboard here: **[Grocery Sales Predictor App](https://ayushdas4890-grocery-sales-predictor-streamlit-app-paeuog.streamlit.app/)**

## 📊 Dataset
**File**: `Groceries_dataset.csv`

The dataset contains transactional data representing daily sales of various grocery items. Key attributes include:
- **Date**: The date of the transaction.
- **itemDescription**: The name of the product sold.
- **Member_number**: Unique identifier for the customer.

*Note: The raw data is aggregated to a daily level to analyze sales trends and demand fluctuations.*

## 🎯 Objective
The primary objective is to build a robust classification model that can accurately predict whether the demand for a specific period will be **"High"** or **"Low"**. This binary classification aids in decision-making processes regarding stock levels and promotional activities.

## 🚀 Vision
To transition from reactive inventory management to a **proactive, data-driven approach**. This model serves as the foundation for a scalable demand forecasting system that adapts to seasonal trends and minimizes both stockouts and overstock scenarios.

## ⚙️ Workflow
The project follows a rigorous Data Science lifecycle:

1.  **Data Loading & Preprocessing**:
    - Aggregating raw transactions into daily sales counts.
    - Handling missing values and ensuring data consistency.
2.  **Feature Engineering**:
    - **Date Features**: Extracted Month, Day, Year, Quarter, DayOfWeek, and Weekend indicators.
    - **Seasonality**: Mapped months to seasons (Winter, Spring, Summer, Fall) to capture cyclical trends.
    - **Lag Features**: Created `Sales_Lag_1` to `Sales_Lag_14` to capture past sales influence.
    - **Rolling Statistics**: Calculated Rolling Means and Standard Deviations (3, 7, 14, 30 days) to smooth noise and capture trends.
3.  **Exploratory Data Analysis (EDA)**:
    - Visualized monthly and seasonal sales distributions.
    - Analyzed the correlation between features and the target variable.
4.  **Model Implementation**:
    - Trained multiple diverse classifiers to find the best fit.
5.  **Evaluation**:
    - Assessed models using Accuracy, Confusion Matrices (TP/TN/FP/FN), and ROC-AUC Curves.
6.  **Validation**:
    - Applied **K-Fold Cross-Validation** to ensure model robustness and prevent overfitting.
7.  **Future Prediction**:
    - Generated predictions for a future target date (Jan 1, 2016) to demonstrate practical application.

## 🤖 Models Used
We implemented and compared a variety of algorithms to ensure the best performance:

1.  **Logistic Regression**: A baseline linear model for binary classification.
2.  **Decision Tree Classifier**: To capture non-linear relationships and simple decision rules.
3.  **Random Forest Classifier**: An ensemble of decision trees to reduce variance and improve accuracy.
4.  **Gradient Boosting Classifier**: A powerful boosting technique that builds models sequentially to correct errors.
5.  **Voting Classifier (Ensemble)**: A **Novelty** in this project, combining Soft Voting from Logistic Regression, Decision Tree, and Random Forest to leverage the strengths of each.

## 🚧 Hurdles Faced & Management
| Hurdle | Strategy / Solution |
| :--- | :--- |
| **Model Generalization** | Initial models might overfit to the training data. We mitigated this by implementing **10-Fold Cross-Validation** on our top performers (Random Forest & Gradient Boosting). |
| **Feature Scale** | The dataset contained features with vastly different scales (e.g., rolling means vs. binary flags). We used `StandardScaler` to normalize all features before training. |
| **Single Model Bias** | Relying on one model can be risky. We introduced a **Voting Classifier** to aggregate predictions, reducing the risk of individual model bias. |
| **Seasonality Complexity** | Simple date features weren't enough. We engineered explicit **Seasonal Codes** and **Rolling Statistics** to capture the "beat" of the business. |

## 🌟 Novelty
- **Ensemble Learning**: The implementation of a `VotingClassifier` demonstrates an advanced approach to improving prediction stability by averaging out errors from diverse models.
- **Rigorous Validation**: Unlike simple train-test splits, our use of **K-Fold Cross-Validation** provides a statistically significant measure of model performance, ensuring the results are reliable.
- **Deep Feature Engineering**: The combination of Lag variations and multiple Rolling Windows created a rich dataset that captures both immediate and long-term trends.

## 🏁 End Goal
The ultimate goal is to deploy this model into a real-time environment where it can ingest live sales data and output daily demand forecasts. The current Proof of Concept (POC) successfully demonstrates this capability by predicting the demand for **January 1, 2016**, paving the way for full-scale production deployment.

---
*Created by the Data Science Team*
