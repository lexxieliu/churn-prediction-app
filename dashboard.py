import streamlit as st
import requests
import json
import pandas as pd # Add pandas for the default data

st.set_page_config(layout="wide")
st.title('🚀 Customer Churn Prediction & Revenue Platform')

st.write("""
This dashboard predicts the likelihood of a customer churning based on their 
account details and behavioral features. It uses an XGBoost model with an 
**AUC of 0.86+** and a **Recall-optimized threshold of 0.30**.
""")

# --- 1. Create input fields in the sidebar ---
st.sidebar.header('Enter Customer Data:')

# Use columns to make it cleaner
col1, col2 = st.sidebar.columns(2)

# --- 2. Create interactive widgets ---
# These match the *most important* features from your SHAP plot.
contract_risk_score = col1.selectbox('Contract Type (Risk Score)', [3, 2, 1], 
                                     format_func=lambda x: {3: 'Month-to-Month', 2: 'One Year', 1: 'Two Year'}[x])
recency_score = col1.slider('Recency Score (Tenure)', 1, 4, 3, 
                            help="4 = 0-6mo, 3 = 6-12mo, 2 = 1-2yr, 1 = 2+ yr")
payment_risk_score = col1.selectbox('Payment Method (Risk Score)', [0.453, 0.191, 0.167, 0.152], 
                                    format_func=lambda x: {0.453: 'Electronic Check', 0.191: 'Mailed Check', 0.167: 'Bank Transfer', 0.152: 'Credit Card'}[x])
fiber_no_support = col1.selectbox('Fiber w/o Support?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])
InternetService_Fiber_optic = col1.selectbox('Internet: Fiber Optic?', [1, 0], format_func=lambda x: {1: 'Yes', 0: 'No'}[x]) # Note: 1 is the 'Yes' option

MonthlyCharges = col2.slider('Monthly Charges ($)', 18.0, 120.0, 75.0, 0.01)
support_services_count = col2.slider('Support Services Count', 0, 4, 0)
engagement_score = col2.slider('Total Engagement Score', 0, 8, 3)
has_family = col2.selectbox('Has Family (Partner/Dependents)?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])
paperless_billing_risk = col2.selectbox('Paperless (Non-AutoPay)?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])

# --- 3. Create the data dictionary to send to the API ---
# We need to fill in *all* 24 features the model expects.
# We'll use the user inputs for the important ones and reasonable defaults for the rest.

# Create a dictionary for all the default/placeholder values
default_data = {
    'gender': 'Female', # Will be one-hot encoded by the pipeline
    'SeniorCitizen': 0,
    'InternetService': 'Fiber optic' if InternetService_Fiber_optic == 1 else 'DSL',
    'frequency_score': 3,
    'monetary_score': 3,
    'customer_segment': 'Medium-Value',
    'no_protection': 1,
    'has_any_support': 0,
    'streaming_count': 0,
    'is_streamer': 0,
    'has_bundle': 1,
    'price_per_service': 15.0,
    'charges_vs_median': 1.1,
    'spending_tier': 'Medium',
    'value_score': 3.5
}

# Create the final JSON payload
customer_data = {
    **default_data, # Start with defaults
    'contract_risk_score': contract_risk_score, # Overwrite with user inputs
    'recency_score': recency_score,
    'payment_risk_score': payment_risk_score,
    'fiber_no_support': fiber_no_support,
    'InternetService': 'Fiber optic' if InternetService_Fiber_optic == 1 else 'DSL', # Update this one too
    'MonthlyCharges': MonthlyCharges,
    'support_services_count': support_services_count,
    'engagement_score': engagement_score,
    'has_family': has_family,
    'paperless_billing_risk': paperless_billing_risk
}

# --- 4. The "Predict" Button ---
if st.sidebar.button('Run Risk Assessment'):
    # API endpoint URL (your 'kitchen's' address)
    api_url = 'http://127.0.0.1:5000/predict'
    
    try:
        # 5. Send the data to the API
        response = requests.post(api_url, json=customer_data)
        response.raise_for_status() # Raise an error for bad responses (like 400 or 500)
        result = response.json()
        
        prob = result['churn_probability'] * 100
        prediction = result['churn_prediction']
        
        # 6. Display the result
        st.subheader('Prediction Result')
        if prediction == 1:
            st.error(f'**HIGH CHURN RISK** (Probability: {prob:.1f}%)')
            st.markdown(f"""
            This customer has a **{prob:.1f}% chance of churning**, which is above our **30.0%** action threshold.
            
            **Recommended Action:**
            * **Cost to Retain:** $63 (Send retention offer)
            * **Potential LTV Saved:** $1,200
            * **Primary Drivers (based on model):** `contract_risk_score`, `recency_score`, `fiber_no_support`.
            * **Action:** Immediately enroll in "Fiber Pro" support bundle and offer a 1-year contract discount.
            """)
        else:
            st.success(f'**LOW CHURN RISK** (Probability: {prob:.1f}%)')
            st.markdown(f"""
            This customer has a **{prob:.1f}% chance of churning**, which is below our **30.0%** action threshold.
            
            **Recommended Action:**
            * No retention offer needed.
            * Consider a low-cost "Thank You" email or a standard upsell for a new service.
            """)

        with st.expander("Show Raw Data Sent to API"):
            st.json(customer_data)
        
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the Flask API. Is it running in its own terminal?")
    except Exception as e:
        st.error(f"An error occurred: {e}")