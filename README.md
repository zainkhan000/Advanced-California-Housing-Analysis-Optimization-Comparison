# Advanced House Price Prediction: Optimization & Benchmarking

## Project Overview
This project implements an enhanced House Price Prediction system using the **California Housing Dataset**. [cite_start]It demonstrates a professional Machine Learning workflow by applying **feature scaling** and **multi-model benchmarking** to identify the most effective algorithm for predicting median house values [cite: 30-34, 143].

## Key Objectives
* [cite_start]**Real-World Improvement**: Move beyond basic training to learn how engineers improve models in real-world projects[cite: 12].
* [cite_start]**Data Preparation**: Ensure data is cleaned, transformed, and refined correctly for Machine Learning[cite: 14, 22].
* [cite_start]**Model Optimization**: Improve performance using preprocessing techniques like feature scaling[cite: 15, 80].
* [cite_start]**Algorithm Comparison**: Train multiple algorithms and select the best-performing model using measurable metrics [cite: 16-17, 24].

## Dataset Information
[cite_start]The project utilizes the **California Housing Dataset**[cite: 39].
* [cite_start]**Target Variable**: Median House Value[cite: 41].
* [cite_start]**Input Features**: Includes Median Income, House Age, Average Rooms, Population, and location-based attributes [cite: 43-46].

## Technical Stack
* [cite_start]**Language**: Python[cite: 48].
* [cite_start]**Libraries**: pandas, NumPy, scikit-learn, matplotlib, and seaborn [cite: 50-53].
* [cite_start]**Environment**: Jupyter Notebook[cite: 49].

## Machine Learning Pipeline
1.  [cite_start]**Library Import**: Loading essential tools for data manipulation and modeling [cite: 55-65].
2.  [cite_start]**Data Loading**: Fetching and structuring the dataset into a DataFrame [cite: 67-72].
3.  [cite_start]**Feature Separation**: Isolating input features ($X$) from the target variable ($y$) [cite: 74-79].
4.  [cite_start]**Feature Scaling**: Using `StandardScaler` to ensure fair learning across features and improve stability [cite: 80-88].
5.  [cite_start]**Train-Test Split**: Allocating 20% of data for evaluation on unseen values [cite: 89-95].
6.  [cite_start]**Multi-Model Training**: Implementing three distinct models [cite: 96-103]:
    * [cite_start]**Linear Regression**: Serves as the performance baseline[cite: 105].
    * [cite_start]**Ridge Regression**: Helps reduce potential overfitting[cite: 106].
    * [cite_start]**Decision Tree**: Captures complex non-linear relationships[cite: 107].
7.  [cite_start]**Performance Benchmarking**: Comparing results using **RMSE** and **R2 Score** [cite: 108-126].
8.  [cite_start]**Visual Validation**: Generating scatter plots to compare Actual vs. Predicted values [cite: 127-140].

## Deliverables
* [cite_start]**Jupyter Notebook**: `AI_ML_Task2_Model_Comparison.ipynb`[cite: 148].
* [cite_start]**Comparison Table**: Structured performance metrics for all trained models[cite: 149].
* [cite_start]**Methodology Report**: A 1-2 page PDF explaining results and conclusions[cite: 150].

