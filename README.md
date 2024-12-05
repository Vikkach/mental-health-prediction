# Exploring Mental Health Data

## Project Overview

This project aims to predict depression using supervised machine learning techniques. The dataset used is generated from a deep learning model trained on the [Depression Survey/Dataset for Analysis dataset](https://www.kaggle.com/datasets/sumansharmadataworld/depression-surveydataset-for-analysis) and explores factors like sleep duration, dietary habits, academic pressure, etc. to identify depression indicators.

## Dataset

The dataset contains both training and testing sets. The features include:

- **Demographic Information**: Gender, Age, City, Profession
- **Academic/Work-Related**: Working Professional or Student, Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction
- **Lifestyle**: Sleep Duration, Dietary Habits, Work/Study Hours
- **Mental Health**: Have you ever had suicidal thoughts?, Financial Stress, Family History of Mental Illness
- **Target Variable**: Depression (0 or 1)

## Data Preprocessing

The following preprocessing steps are applied to the dataset:

- **Data Cleaning**: Handling missing values, standardizing categorical variables (City, Profession, Degree), fixing inconsistencies in columns like Sleep Duration and Dietary Habits.
- **Feature Engineering**: Splitting the "Working Professional or Student" column into separate binary features for "Student" and "Working Professional".
- **Normalization**: Scaling numerical features to a range between 0 and 1 using Min-Max scaling.
- **Encoding**: One-hot encoding for categorical features and binary conversion for binary features.

## Model Training and Evaluation

- **Model Selection**: Various models are explored, including Logistic Regression, Random Forest, XGBoost, CatBoost, Decision Tree, and K-Nearest Neighbors.
- **Hyperparameter Tuning**: GridSearchCV is used to find the best hyperparameters for each model.
- **Model Evaluation**: Accuracy is used as the primary evaluation metric. Cross-validation is performed to assess model generalization.

## Results

- The ensemble model (VotingClassifier) combining Random Forest, CatBoost, and XGBoost achieved the best test accuracy of ~0.94.
- Key factors associated with depression:
    - Age (negative correlation)
    - Academic Pressure (positive correlation)
    - Have you ever had suicidal thoughts? (positive correlation)
    - Financial Stress (positive correlation)

## Conclusion

This project demonstrates the potential of machine learning to predict depression based on survey data. It highlights important factors that could contribute to depression.

## Usage

To run the notebook:

1.  **Import data source**.
2.  **Install required libraries**: `seaborn`, `matplotlib`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `imblearn`.
3.  **Execute the notebook cells**.
