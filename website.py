import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="wide"
)

st.title("Loan Default Risk Predictor")
st.markdown("Predict the probability of loan default using machine learning models")

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    try:
        # Load preprocessing objects
        model_columns = joblib.load('model_columns.joblib')
        imputer = joblib.load('imputer.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Load models
        ann_model = tf.keras.models.load_model('ann_model.h5')
        rf_model = joblib.load('random_forest_model.joblib')
        xgb_model = joblib.load('xgboost_model.joblib')
        
        return model_columns, imputer, scaler, ann_model, rf_model, xgb_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None

# Load models
model_columns, imputer, scaler, ann_model, rf_model, xgb_model = load_models()

if model_columns is not None:
    # Create input form
    st.header("Loan Application Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Loan Information")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=17500, step=500)
        term = st.selectbox("Term", [36, 60], help="Loan term in months", index=1)
        int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=17.0, step=0.1)
        installment = st.number_input("Monthly Installment ($)", min_value=15.0, max_value=2000.0, value=450.0, step=10.0)
        
        purpose = st.selectbox("Purpose", [
            'debt_consolidation', 'credit_card', 'home_improvement', 'other', 
            'major_purchase', 'small_business', 'car', 'medical', 'moving',
            'vacation', 'house', 'wedding', 'renewable_energy', 'educational'
        ], index=1)
        
    with col2:
        st.subheader("Borrower Information")
        annual_inc = st.number_input("Annual Income ($)", min_value=4000, max_value=500000, value=50000, step=1000)
        dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        
        home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
        verification_status = st.selectbox("Income Verification", ['Verified', 'Source Verified', 'Not Verified'], index=1)
        
        issue_year = st.selectbox("Issue Year", list(range(2007, 2025)), index=7)
        earliest_cr_year = st.selectbox("Earliest Credit Line Year", list(range(1940, 2025)), index=55)
        
    with col3:
        st.subheader("Credit Information")
        open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=14)
        total_acc = st.number_input("Total Credit Lines", min_value=1, max_value=100, value=28)
        pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)
        pub_rec_bankruptcies = st.number_input("Bankruptcies", min_value=0, max_value=5, value=0)
        
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=11000)
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=95.0)
        mort_acc = st.number_input("Mortgage Accounts", min_value=0, max_value=20, value=0)
        
        sub_grade = st.selectbox("Sub Grade", [
            'A1', 'A2', 'A3', 'A4', 'A5',
            'B1', 'B2', 'B3', 'B4', 'B5',
            'C1', 'C2', 'C3', 'C4', 'C5',
            'D1', 'D2', 'D3', 'D4', 'D5',
            'E1', 'E2', 'E3', 'E4', 'E5',
            'F1', 'F2', 'F3', 'F4', 'F5',
            'G1', 'G2', 'G3', 'G4', 'G5'
        ], index=17)
        
        initial_list_status = st.selectbox("Initial Listing Status", ['w', 'f'])
        application_type = st.selectbox("Application Type", ['Individual', 'Joint App'])
        
        # Simplified zip code input
        zip_code = st.selectbox("ZIP Code", ['70466', '22690', '30723', '48052', '00813', '29597', '05113', '11650', '93700', '86630'])
    
    if st.button("Predict Default Risk", type="primary"):
        try:
            # Create input dataframe
            input_data = {
                'loan_amnt': loan_amnt,
                'term': term,
                'int_rate': int_rate,
                'installment': installment,
                'annual_inc': annual_inc,
                'dti': dti,
                'open_acc': open_acc,
                'pub_rec': pub_rec,
                'revol_bal': revol_bal,
                'revol_util': revol_util,
                'total_acc': total_acc,
                'mort_acc': mort_acc,
                'pub_rec_bankruptcies': pub_rec_bankruptcies,
                'issue_d': issue_year,
                'earliest_cr_line': earliest_cr_year
            }
            
            # Create base dataframe
            df_input = pd.DataFrame([input_data])
            
            # Create dummy variables for categorical features
            categorical_mappings = {
                'sub_grade': sub_grade,
                'verification_status': verification_status,
                'purpose': purpose,
                'initial_list_status': initial_list_status,
                'application_type': application_type,
                'home_ownership': home_ownership,
                'zip_code': zip_code
            }
            
            # Create a dataframe with all possible dummy columns set to 0
            dummy_df = pd.DataFrame(0, index=[0], columns=model_columns)
            
            # Set the numerical columns
            for col, val in input_data.items():
                if col in dummy_df.columns:
                    dummy_df[col] = val
            
            # Set the appropriate dummy variables to 1
            for category, value in categorical_mappings.items():
                if category == 'sub_grade':
                    # Skip the first category (drop_first=True in original)
                    if value != 'A1':  # Assuming A1 is the first category that gets dropped
                        col_name = f'sub_grade_{value}'
                        if col_name in dummy_df.columns:
                            dummy_df[col_name] = 1
                            
                elif category == 'verification_status':
                    if value != 'Not Verified':  # Assuming this is the first category
                        col_name = f'verification_status_{value}'
                        if col_name in dummy_df.columns:
                            dummy_df[col_name] = 1
                            
                elif category == 'purpose':
                    if value != 'car':  # Assuming car is the first category alphabetically
                        col_name = f'purpose_{value}'
                        if col_name in dummy_df.columns:
                            dummy_df[col_name] = 1
                            
                elif category == 'initial_list_status':
                    if value != 'f':
                        col_name = f'initial_list_status_{value}'
                        if col_name in dummy_df.columns:
                            dummy_df[col_name] = 1
                            
                elif category == 'application_type':
                    if value != 'Individual':
                        col_name = f'application_type_{value}'
                        if col_name in dummy_df.columns:
                            dummy_df[col_name] = 1
                            
                elif category == 'home_ownership':
                    if value != 'MORTGAGE':  # Assuming MORTGAGE is dropped
                        col_name = f'home_ownership_{value}'
                        if col_name in dummy_df.columns:
                            dummy_df[col_name] = 1
                            
                elif category == 'zip_code':
                    col_name = f'zip_code_{value}'
                    if col_name in dummy_df.columns:
                        dummy_df[col_name] = 1
            
            # Apply preprocessing
            X_processed = imputer.transform(dummy_df)
            X_scaled = scaler.transform(X_processed)
            
            # Make predictions
            ann_pred = 1 - ann_model.predict(X_scaled)[0][0]
            rf_pred = 1 - rf_model.predict_proba(X_scaled)[0][1]
            xgb_pred = 1 - xgb_model.predict_proba(X_scaled)[0][1]
            
            # Display results
            st.header("📈 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🧠 Neural Network",
                    f"{ann_pred*100:.1f}%",
                    help="Probability of default using Artificial Neural Network"
                )
                
            with col2:
                st.metric(
                    "🌲 Random Forest",
                    f"{rf_pred*100:.1f}%",
                    help="Probability of default using Random Forest"
                )
                
            with col3:
                st.metric(
                    "🚀 XGBoost",
                    f"{xgb_pred*100:.1f}%",
                    help="Probability of default using XGBoost"
                )
            
            # Average prediction
            avg_pred = (ann_pred + rf_pred + xgb_pred) / 3
            
            st.header("Overall Risk Assessment")
            
            risk_level = "Low Risk" if avg_pred < 0.2 else "Medium Risk" if avg_pred < 0.5 else "High Risk"
            
            st.metric(
                "Average Default Probability",
                f"{avg_pred*100:.1f}%",
                help="Average prediction across all models"
            )
            
            st.success(f"Risk Level: {risk_level}")
            
            # Risk interpretation
            if avg_pred < 0.2:
                st.info("This loan application shows low default risk. The borrower has strong creditworthiness indicators.")
            elif avg_pred < 0.5:
                st.warning("This loan application shows moderate default risk. Consider additional verification or adjusted terms.")
            else:
                st.error("This loan application shows high default risk. Careful consideration recommended before approval.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check that all model files are properly saved and accessible.")

else:
    st.error("Model files not found. Please ensure the following files are in the app directory:")
    st.write("- model_columns.joblib")
    st.write("- imputer.joblib") 
    st.write("- scaler.joblib")
    st.write("- ann_model.h5")
    st.write("- random_forest_model.joblib")
    st.write("- xgboost_model.joblib")
