# Obesity Classification: A Machine Learning Approach

## Table of Contents
1. [Overview](#Overview)
2. [Project Goals](#Project-Goals)
3. [Variables Table](#Variables-Table)
4. [Data](#Data)
5. [Data Cleaning](#Data-Cleaning)
6. [Methodology](#Methodology)
7. [Models Implemented](#Models-Implemented)
8. [Initial Model Performance](#Initial-Model-Performance)
9. [Model Optimization](#Model-Optimization)
10. [Results](#Results)
11. [Insights and Key Findings](#Insights-and-Key-Findings)
12. [Future Development](#Future-Development)
13. [Repository Structure](#Repository-Structure)
14. [How to Run](#How-to-Run)
15. [Conclusion](#Conclusion)
16. [Acknowledgments](#Acknowledgments)
17. [License](#License)

## Overview
This project aims to classify individuals into different obesity levels based on several health-related factors such as age, height, weight, eating habits, and physical activities. The classification model predicts the obesity category based on these attributes with a high level of accuracy, providing insights for health awareness and intervention planning.

## Project Goals
- **Data Cleaning and Preparation**: The dataset was cleaned and transformed to ensure it is ready for model training.
- **Model Training**: A Random Forest classifier was used to predict obesity levels.
- **Model Optimization**: The model's performance was further optimized using hyperparameter tuning.
- **Evaluation**: The final model achieved a classification accuracy of over 95%.

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

## Data
The dataset used for this project contains synthetic data related to obesity levels with various health-related features. The target variable represents the obesity classification.

## Data Cleaning
- Categorical variables were encoded as numerical values.
- Numerical values were scaled using `StandardScaler`.
- The cleaned dataset was saved as a CSV for further use.

## Methodology
1. **Exploratory Data Analysis (EDA)**: Conducted histograms and bar plots to visualize key features and analyze the correlation between features and the target variable.
2. **Model Selection**: Implemented a **Random Forest Classifier** due to its robustness and ability to handle categorical and numerical data.
3. **Data Splitting**: The dataset was divided into 80% for training and 20% for testing.

## Models Implemented
- **Random Forest Classifier**: An ensemble method used to predict obesity levels based on various health factors.

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

## Insights and Key Findings
- **Critical Features**: Family history, physical activity, and eating habits were significant predictors of obesity.
- **Model Performance**: Random Forest outperformed other models, making it the most suitable for this dataset due to its ability to handle non-linear data.

## Future Development
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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
