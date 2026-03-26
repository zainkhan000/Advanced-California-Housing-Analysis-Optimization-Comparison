# Advanced House Price Prediction: Optimization & Benchmarking

## Project Overview
This project implements an enhanced House Price Prediction system using the **California Housing Dataset**. It demonstrates a professional Machine Learning workflow by applying **feature scaling** and **multi-model benchmarking** to identify the most effective algorithm for predicting median house values.

## Key Objectives
* **Real-World Improvement**: Move beyond basic training to learn how engineers improve models in real-world projects.
* **Data Preparation**: Ensure data is cleaned, transformed, and refined correctly for Machine Learning.
* **Model Optimization**: Improve performance using preprocessing techniques like feature scaling.
* **Algorithm Comparison**: Train multiple algorithms and select the best-performing model using measurable metrics.

## Dataset Information
The project utilizes the **California Housing Dataset**.
* **Target Variable**: Median House Value.
* **Input Features**: Includes Median Income, House Age, Average Rooms, Population, and location-based attributes.

## Technical Stack
* **Language**: Python.
* **Libraries**: pandas, NumPy, scikit-learn, matplotlib, and seaborn.
* **Environment**: Jupyter Notebook.

## Machine Learning Pipeline
1.  **Library Import**: Loading essential tools for data manipulation and modeling.
2.  **Data Loading**: Fetching and structuring the dataset into a DataFrame.
3.  **Feature Separation**: Isolating input features ($X$) from the target variable ($y$).
4.  **Feature Scaling**: Using `StandardScaler` to ensure fair learning across features and improve stability.
5.  **Train-Test Split**: Allocating 20% of data for evaluation on unseen values.
6.  **Multi-Model Training**: Implementing three distinct models:
    * **Linear Regression**: Serves as the performance baseline.
    * **Ridge Regression**: Helps reduce potential overfitting.
    * **Decision Tree**: Captures complex non-linear relationships.
7.  **Performance Benchmarking**: Comparing results using **RMSE** and **R2 Score**.
8.  **Visual Validation**: Generating scatter plots to compare Actual vs. Predicted values.

