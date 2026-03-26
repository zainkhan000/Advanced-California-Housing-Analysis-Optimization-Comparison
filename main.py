import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def run_pipeline():
    # 1. Load the California Housing dataset
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("HousePrice")], axis=1)
    
    # 2. Separate Features (X) and Target Variable (y)
    X = df.drop("HousePrice", axis=1)
    y = df["HousePrice"]
    
    # 3. Feature Scaling
    # This ensures all features have a common scale for better stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 5. Define and Train Multiple Models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate performance metrics
        # np.sqrt is used for RMSE to ensure version compatibility
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        results[name] = {
            "RMSE": rmse,
            "R2 Score": r2
        }
        
    # 6. Display Performance Comparison Table
    results_df = pd.DataFrame(results).T
    print("\n--- Model Performance Comparison ---")
    print(results_df)
    
    # 7. Visual Performance Validation
    # Plotting the baseline Linear Regression results
    best_model = LinearRegression()
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.4, color='teal')
    plt.xlabel("Actual House Prices")
    plt.ylabel("Predicted House Prices")
    plt.title("Actual vs Predicted House Prices")
    # Perfect prediction reference line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
    plt.show()
    
    # 8. Save the Model
    joblib.dump(best_model, 'house_price_model.pkl')
    print("\nModel saved successfully as 'house_price_model.pkl'.")

if __name__ == "__main__":
    run_pipeline()
