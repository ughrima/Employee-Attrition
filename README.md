# CodeClause-Internship_Emp-Attrition

# Employee Attrition Prediction

## Project Overview

This project aims to predict the likelihood of employee attrition in a company using HR data. By leveraging advanced classification techniques and feature engineering, the goal is to build a model that can accurately identify employees who are likely to leave the company.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

The project is divided into two main components:

1. **Data Preprocessing**: Handling data cleaning, feature engineering, and scaling.
2. **Model Training**: Building and evaluating classification models.

## Files

- `preprocessing.ipynb`: Contains the code for data preprocessing including handling missing values, encoding categorical variables, and feature scaling.
- `train.ipynb`: Contains the code for building, training, and evaluating the classification models.

## Steps to Run the Project

### 1. Data Preprocessing

Open the `preprocessing.ipynb` notebook and execute the cells to preprocess the HR data. This step involves:
- Loading the dataset.
- **Basic Inspection**: Initial inspection of the data.
- **Dropping Columns**: Removing columns that have the same value for all entries.
- **Feature Relationship Analysis**: Plotting graphs to see the relationship between features and the target variable.
- **Data Encoding**: Encoding categorical features to convert them into numerical values to ensure model accuracy.
- Handling missing values.
- Scaling numerical features.

### 2. Model Training

Open the `train.ipynb` notebook and execute the cells to train and evaluate the models. This step involves:
- Splitting the data into training and testing sets.
- Building and evaluating classification models:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
  - Naive Bayes
- **Observations**:
  - Random Forest and Logistic Regression were the best performing models.
  - Random Forest had the highest accuracy but tended to overfit the data.
  - Logistic Regression provided a good balance of accuracy without overfitting.
- Hyperparameter tuning to optimize the model.

## Results

The model evaluation metrics and visualizations will be generated in the `train.ipynb` notebook. These results help in understanding the performance of the models and identifying the best performing model.

## Future Work

- Collect more data to enhance model robustness.
- Explore other machine learning algorithms like XGBoost or Support Vector Machines.
- Implement cross-validation to ensure model generalizability.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

