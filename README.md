# Obesity Classification: A Machine Learning Approach

## Table of Contents
1. [Overview](#overview)
2. [Project Goals](#project-goals)
3. [Data Description](#data-description)
4. [Variables Table](#variables-table)
5. [Data Cleaning](#data-cleaning)
6. [Methodology](#methodology)
7. [Models Implemented](#models-implemented)
8. [Initial Model Performance](#initial-model-performance)
9. [Model Optimization](#model-optimization)
10. [Results](#results)
11. [Insights and Key Findings](#insights-and-key-findings)
12. [Future Work](#future-work)
13. [Repository Structure](#repository-structure)
14. [How to Run](#how-to-run)
15. [Conclusion](#conclusion)
16. [Acknowledgments](#acknowledgments)
17. [Sources](#Sources)


## Overview
This project aims to classify individuals into different obesity levels based on several health-related factors such as age, height, weight, eating habits, and physical activities. The classification model predicts the obesity category based on these attributes with a high level of accuracy, providing insights for health awareness and intervention planning. Understanding the factors contributing to obesity can help in formulating effective interventions. This project utilizes a synthetic dataset to build and evaluate machine learning models for predicting obesity levels based on various lifestyle and demographic features.

## Project Goals
- **Data Cleaning and Preparation**: The dataset was cleaned and transformed to ensure it is ready for model training.
- **Model Training**: A Random Forest classifier was used to predict obesity levels.
- **Model Optimization**: The model's performance was further optimized using hyperparameter tuning.
- **Evaluation**: The final model achieved a classification accuracy of over 95%.

## Data Description
- **Total Observations:** 2,112 rows
- **Features:** 
  - Gender
  - Age
  - Height
  - Weight
  - Family History
  - Eating habits
  - Physical activity
  - ...and more.
- **Target Variable:** Obesity level (classified into various categories).

The dataset was synthetically generated to represent diverse obesity levels, ensuring a balanced distribution across different features.

## Variables Table

| Variable Name                    | Role    | Type        | Demographic | Description                                                      | Units | Missing Values |
|-----------------------------------|---------|-------------|-------------|------------------------------------------------------------------|-------|----------------|
| **Gender**                        | Feature | Categorical | Gender      |                                                                  |       | No             |
| **Age**                           | Feature | Continuous  | Age         |                                                                  |       | No             |
| **Height**                        | Feature | Continuous  |             |                                                                  |       | No             |
| **Weight**                        | Feature | Continuous  |             |                                                                  |       | No             |
| **family_history_with_overweight**| Feature | Binary      |             | Has a family member suffered or suffers from overweight?          |       | No             |
| **FAVC**                          | Feature | Binary      |             | Do you eat high caloric food frequently?                          |       | No             |
| **FCVC**                          | Feature | Integer     |             | Do you usually eat vegetables in your meals?                      |       | No             |
| **NCP**                           | Feature | Continuous  |             | How many main meals do you have daily?                            |       | No             |
| **CAEC**                          | Feature | Categorical |             | Do you eat any food between meals?                                |       | No             |
| **SMOKE**                         | Feature | Binary      |             | Do you smoke?                                                     |       | No             |
| **CH2O**                          | Feature | Continuous  |             | How much water do you drink daily?                                |       | No             |
| **SCC**                           | Feature | Binary      |             | Do you monitor the calories you eat daily?                        |       | No             |
| **FAF**                           | Feature | Continuous  |             | How often do you have physical activity?                          |       | No             |
| **TUE**                           | Feature | Integer     |             | How much time do you use technological devices such as cell phone, videogames, television, computer and others? | No    |
| **CALC**                          | Feature | Categorical |             | How often do you drink alcohol?                                   |       | No             |
| **MTRANS**                        | Feature | Categorical |             | Which transportation do you usually use?                          |       | No             |
| **NObeyesdad**                    | Target  | Categorical |             | Obesity level                                                     |       | No             |

## Data Cleaning
- Categorical variables were encoded as numerical values.
- Numerical values were scaled using `StandardScaler`.
- The cleaned dataset was saved as a CSV for further use.

## Methodology
1. **Data Cleaning and Preprocessing**
   - Handled missing values.
   - Categorical encoding for variables like Gender, Family History, and Eating habits.
   - Scaled continuous variables (e.g., Age, Weight, Height).
   - Addressed class imbalance during model evaluation.
2. **Exploratory Data Analysis (EDA)**: Conducted histograms and bar plots to visualize key features and analyze the correlation between features and the target variable.
3. **Model Selection**: Implemented a **Random Forest Classifier** due to its robustness and ability to handle categorical and numerical data.
4. **Data Splitting**: The dataset was divided into 80% for training and 20% for testing.

## Models Implemented
- **Random Forest Classifier**: An ensemble method used to predict obesity levels based on various health factors.
- **Logistic Regression:** A linear model used for baseline comparison.
- **K-Nearest Neighbors (KNN):** A distance-based algorithm for comparison.

### Model Performance
| Model               | Accuracy  | Precision | Recall | F1-Score |
|---------------------|-----------|-----------|--------|----------|
| Logistic Regression  | 76.12%    | Varies    | Varies | Varies   |
| Random Forest        | 96.21%    | Varies    | Varies | Varies   |
| K-Nearest Neighbors   | 81.32%    | Varies    | Varies | Varies   |

## Initial Model Performance
- **Accuracy**: 95.5%
- **Precision, Recall, and F1-score** for each obesity class were all high, indicating strong performance across the board.

## Model Optimization
A hyperparameter tuning process was conducted using `GridSearchCV`, which included:
- Varying the number of trees (`n_estimators`), tree depth (`max_depth`), and other relevant parameters.
- The best model was selected based on cross-validation scores.

## Results
- **Final Model Accuracy**: 95.5%
- The model successfully classifies individuals into obesity categories with a high degree of accuracy.
- **Random Forest** achieved the highest accuracy of **96.21%**, effectively handling diverse obesity levels.
- **Logistic Regression** and **KNN** performed moderately, with Logistic Regression struggling with more complex classes.

## Insights and Key Findings
- **Critical Features:** Family history, physical activity, and eating habits were significant predictors of obesity.
- **Model Performance:** Random Forest outperformed other models, making it the most suitable for this dataset due to its ability to handle non-linear data.

## Future Work
- Investigate how model accuracy varies with real-world data.
- Explore advanced feature engineering methods (e.g., calculating BMI).
- Conduct further investigation into the impact of class imbalance and potential oversampling techniques.
- Explore more complex models like XGBoost or Neural Networks for potential further improvements.
- Use real-world data for validation.
- Investigate the impact of other features, such as dietary patterns and medical history, on the prediction of obesity levels.

## Repository Structure
- **data/**: Contains the cleaned dataset (`ObesityDataSet_raw_and_data_sinthetic.csv`).
- **notebooks/**: Jupyter notebooks for data exploration, cleaning, and model training.
- **scripts/**: Python scripts for model training and evaluation.
- **README.md**: Overview of the project.
- **.gitignore**: Specifies files and folders to be ignored in the repository.

## How to Run
1. Clone the repository: `git clone <repo_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python scripts to see the data processing, model training, and evaluation.

## Conclusion
This project demonstrates the use of a machine learning model to classify obesity levels based on lifestyle and physical attributes. With further improvements and data, the model can be used for early obesity diagnosis and prevention strategies.

## Acknowledgments
Special thanks to all participants and contributors who made this project possible.


## Sources

Dataset: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

Kaggle information: https://www.kaggle.com/datasets/suleymansulak/obesity-datasetWhat is BMI: https://www.ncbi.nlm.nih.gov/books/NBK541070/


