## diabetes_project_2


1. [Overview](#Overview)
2. [Project Goals](#Project-Goals)
3. [Data](#data)
4. [Data Cleaning](#Data-Cleaning)
5. [Model](#Model)
6. [Initial Model Performance](#Initial-Model-Performance)
7. [Model Optimization](#Model_Optimization)
8. [Results](#Results)
9. [Future Development](#Future-Development)
10. [Repository Structure](#Repository-Structure)
11. [How to Run](#How-to-Run)
12. [Conclusion](#conclusion)






## **Overview**
This project aims to classify individuals into different obesity levels based on several health-related factors such as age, height, weight, eating habits, and physical activities. The classification model predicts the obesity category based on these attributes with a high level of accuracy.

## **Project Goals**
- **Data Cleaning and Preparation**: The dataset was cleaned and transformed to ensure it is ready for model training.
- **Model Training**: A Random Forest classifier was used to predict obesity levels.
- **Model Optimization**: The model's performance was further optimized using hyperparameter tuning.
- **Evaluation**: The final model achieved a classification accuracy of over 95%.

## **Data**
The dataset used for this project contains synthetic data related to obesity levels with the following key features:
- **Age, Height, Weight**: Numerical values representing physical attributes.
- **FAVC, CAEC, SMOKE, MTRANS, etc.**: Categorical variables representing lifestyle choices and habits.
- **NObeyesdad**: The target variable representing the obesity classification.

## **Data Cleaning**
- Categorical variables were encoded as numerical values.
- Numerical values were scaled using StandardScaler.
- The cleaned dataset was saved as a CSV for further use.

## **Model**
A **Random Forest Classifier** was chosen due to its robustness and ability to handle categorical and numerical data. The model was trained using 80% of the dataset, while 20% was reserved for testing.

## **Initial Model Performance**
- **Accuracy**: 95.5%
- **Precision, Recall, and F1-score** for each obesity class were all high, indicating strong performance across the board.

## **Model Optimization**
A hyperparameter tuning process was conducted using GridSearchCV, which included:
- Varying the number of trees (`n_estimators`), tree depth (`max_depth`), and other relevant parameters.
- The best model was selected based on cross-validation scores.

## **Results**
- **Final Model Accuracy**: 95.5%
- The model successfully classifies individuals into obesity categories with a high degree of accuracy.

## **Future Development**
- Explore more complex models like XGBoost or Neural Networks for potential further improvements.
- Use real-world data for validation.
- Investigate the impact of other features, such as dietary patterns and medical history, on the prediction of obesity levels.

## **Repository Structure**
- **data/**: Contains the cleaned dataset (`ObesityDataSet_raw_and_data_sinthetic.csv`).
- **notebooks/**: Jupyter notebooks for data exploration, cleaning, and model training.
- **scripts/**: Python scripts for model training and evaluation.
- **README.md**: Overview of the project.
- **.gitignore**: Specifies files and folders to be ignored in the repository.

## **How to Run**
1. Clone the repository: `git clone <repo_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python scripts to see the data processing, model training, and evaluation.

## **Conclusion**
This project demonstrates the use of a machine learning model to classify obesity levels based on lifestyle and physical attributes. With further improvements and data, the model can be used for early obesity diagnosis and prevention strategies.
