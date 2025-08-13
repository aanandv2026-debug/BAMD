# Group 12
MBA/0131/61	SAHIL NEGI
MBA/0161/61	AANAND VISHNU NAMBUDIRIPAD
MBA/0242/61	ABHINAV SINGH UPPERWAL
MBA/0258/61	BUDARAPU SHASHANK
MBA/0406/61	ABHISHEK SOURABH


### Run lending_club.ipynb

### Loan Default Prediction
This project aims to predict whether a borrower will pay back their loan or default. Using a dataset of past loan applicants from Lending Club, three different machine learning models were built and evaluated to classify loan status as either "Fully Paid" or "Charged Off".
Dataset
The project uses the "Lending Club Loan Data" dataset (lending_club_loan_two.csv). This dataset contains information on past loan applicants, including their loan details, financial history, and employment information.
#### Data Dictionary
Feature	Description
loan_amnt	The listed amount of the loan applied for by the borrower.
term	The number of payments on the loan (in months).
int_rate	Interest Rate on the loan.
installment	The monthly payment owed by the borrower.
grade	LC assigned loan grade.
home_ownership	The home ownership status provided by the borrower (RENT, OWN, MORTGAGE).
annual_inc	The self-reported annual income provided by the borrower.
verification_status	Indicates if income was verified by LC, not verified, or if the income source was verified.
purpose	A category provided by the borrower for the loan request.
dti	A ratio of the borrower's total monthly debt payments to their self-reported monthly income.
revol_util	Revolving line utilization rate.
mort_acc	Number of mortgage accounts.
pub_rec_bankruptcies	Number of public record bankruptcies.
loan_status	Target Variable: Current status of the loan (Fully Paid or Charged Off).
### Project Pipeline
Data Cleaning & Preprocessing:
Handled missing values by dropping sparse/uninformative columns (emp_title, emp_length) and imputing others (mort_acc).
Engineered new features by extracting the year from date columns and the zip code from the address string.
Converted categorical features into a numerical format using one-hot encoding.
Exploratory Data Analysis (EDA):
Analyzed the correlation between numerical features and the target variable (loan_status).
Visualized the distribution of key features to understand their relationship with loan default rates.
### Model Training & Evaluation:
The data was split into training (67%) and testing (33%) sets and scaled using MinMaxScaler.
Three different classification models were trained and tuned:
Artificial Neural Network (ANN) using TensorFlow/Keras.
Random Forest Classifier with hyperparameter tuning via HalvingRandomSearchCV.
XGBoost Classifier with hyperparameter tuning via RandomizedSearchCV.
Models were evaluated based on their Accuracy, Classification Report, and ROC AUC Score.
### Results
All models performed well, achieving high accuracy. The XGBoost model showed the best performance on the test set based on the ROC AUC score.
Model	Test Set ROC AUC Score
XGBoost	0.9110
ANN	0.9084
Random Forest	0.8961
### Usage
The entire analysis and model training process is documented in the Jupyter Notebook. To replicate the results, you can run the cells sequentially.
Dependencies
Python 3.x
pandas
scikit-learn
TensorFlow
XGBoost
Matplotlib & Seaborn
joblib
### Saved Artifacts
The following artifacts have been saved for easy deployment and inference:
imputer.joblib: The trained SimpleImputer for handling missing mort_acc values.
scaler.joblib: The trained MinMaxScaler.
model_columns.joblib: A list of the feature columns used for training.
ann_model.h5: The trained Artificial Neural Network model.
random_forest_model.joblib: The tuned Random Forest model.
xgboost_model.joblib: The tuned XGBoost model.
