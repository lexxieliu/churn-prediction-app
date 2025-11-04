import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD THE TRAINED MODEL ---
# This happens only once when the app starts
try:
    pipeline = joblib.load('churn_model_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file (churn_model_pipeline.pkl) not found. Please train the model first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. DEFINE THE FEATURE COLUMNS ---
# This MUST be in the same order as the model was trained
feature_columns = [
    'gender', 'SeniorCitizen', 'InternetService', 
    'recency_score', 'frequency_score', 'monetary_score', 
    'customer_segment', 'contract_risk_score', 'payment_risk_score',
    'paperless_billing_risk', 'no_protection', 'fiber_no_support',
    'support_services_count', 'has_any_support', 'streaming_count',
    'is_streamer', 'engagement_score', 'has_bundle', 'has_family',
    'price_per_service', 'charges_vs_median', 'spending_tier',
    'value_score', 'MonthlyCharges'
]

# --- 3. BUILD THE STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title('🚀 Customer Churn Prediction & Revenue Platform')

st.write("""
This dashboard predicts the likelihood of a customer churning based on their 
account details and behavioral features. It uses an XGBoost model with an 
**AUC of 0.84+** and a **Recall-optimized threshold**.
""")

# --- Create input fields in the sidebar ---
st.sidebar.header('Enter Customer Data:')

col1, col2 = st.sidebar.columns(2)

# --- Column 1 Inputs ---
contract_risk_score = col1.selectbox('Contract Type (Risk Score)', [3, 2, 1], 
                                     format_func=lambda x: {3: 'Month-to-Month', 2: 'One Year', 1: 'Two Year'}[x])
recency_score = col1.slider('Recency Score (Tenure)', 1, 4, 3, 
                            help="4 = 0-6mo, 3 = 6-12mo, 2 = 1-2yr, 1 = 2+ yr")
payment_risk_score = col1.selectbox('Payment Method (Risk Score)', [0.453, 0.191, 0.167, 0.152], 
                                    format_func=lambda x: {0.453: 'Electronic Check', 0.191: 'Mailed Check', 0.167: 'Bank Transfer', 0.152: 'Credit Card'}[x])
fiber_no_support = col1.selectbox('Fiber w/o Support?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])
InternetService = col1.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])

# --- Column 2 Inputs ---
# Parameters below are based on the .describe() output from the notebook
MonthlyCharges = col2.slider('Monthly Charges ($)', 18.0, 120.0, 75.0, 0.01) # <-- Values confirmed from notebook
support_services_count = col2.slider('Support Services Count', 0, 4, 0)
engagement_score = col2.slider('Total Engagement Score', 0, 8, 3)
has_family = col2.selectbox('Has Family (Partner/Dependents)?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])
paperless_billing_risk = col2.selectbox('Paperless (Non-AutoPay)?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])


# --- 4. CREATE THE DATA DICTIONARY WITH DEFAULTS ---
# These are needed to match the model's expected 24 features
# You can expand the dashboard to include these if you wish
default_data = {
    'gender': 'Female', # Will be one-hot encoded by the pipeline
    'SeniorCitizen': 0,
    'frequency_score': 3,
    'monetary_score': 3,
    'customer_segment': 'Medium-Value',
    'no_protection': 1,
    'has_any_support': 0,
    'streaming_count': 1,
    'is_streamer': 1,
    'has_bundle': 1,
    'price_per_service': 15.0,
    'charges_vs_median': 1.1,
    'spending_tier': 'Medium',
    'value_score': 4.0
}

# Combine inputs into a dictionary
customer_data_dict = {
    **default_data, # Add defaults first
    'contract_risk_score': contract_risk_score, # Overwrite with user inputs
    'recency_score': recency_score,
    'payment_risk_score': payment_risk_score,
    'fiber_no_support': fiber_no_support,
    'InternetService': InternetService,
    'MonthlyCharges': MonthlyCharges,
    'support_services_count': support_services_count,
    'engagement_score': engagement_score,
    'has_family': has_family,
    'paperless_billing_risk': paperless_billing_risk
}

# --- 5. THE PREDICTION LOGIC ---
if st.sidebar.button('Run Risk Assessment'):
    try:
        # Convert the dictionary to a DataFrame in the correct order
        customer_data = pd.DataFrame(customer_data_dict, index=[0])
        customer_data = customer_data[feature_columns]
        
        # Get probability prediction
        # predict_proba returns [[P(No Churn), P(Churn)]]
        probability = pipeline.predict_proba(customer_data)[0][1]
        
        # Apply your optimal threshold
        optimal_threshold = 0.962 # <-- UPDATED FROM NOTEBOOK RESULT
        prediction = 1 if probability >= optimal_threshold else 0
        
        prob_percent = probability * 100
        
        # --- 6. DISPLAY THE RESULT ---
        st.subheader('Prediction Result')
        if prediction == 1:
            st.error(f'**HIGH CHURN RISK** (Probability: {prob_percent:.1f}%)')
            st.markdown(f"""
            This customer has a **{prob_percent:.1f}% chance of churning**, which is above our **{optimal_threshold*100:.1f}%** action threshold.
            
            **Recommended Action:**
            * **Cost to Retain:** $63 (Send retention offer)
            * **Potential LTV Saved:** $1,200
            * **Primary Drivers (based on model):** `contract_risk_score` (Month-to-Month), `recency_score` (New Customer), `fiber_no_support`.
            * **Action:** Immediately enroll in "Fiber Pro" support bundle and offer a 1-year contract discount.
            """)
        else:
            st.success(f'**LOW CHURN RISK** (Probability: {prob_percent:.1f}%)')
            st.markdown(f"""
            This customer has a **{prob_percent:.1f}% chance of churning**, which is below our **{optimal_threshold*100:.1f}%** action threshold.
            
            **Recommended Action:**
            * No retention offer needed.
            * Consider a low-cost "Thank You" email or a standard upsell for a new service.
            """)

        with st.expander("Show Raw Customer Data Sent to Model:"):
            st.dataframe(customer_data)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")