import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD THE TRAINED MODEL ---
try:
    pipeline = joblib.load('churn_model_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file (churn_model_pipeline.pkl) not found. Please train the model first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. DEFINE THE FEATURE COLUMNS ---
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
account details and behavioral features. It uses an **XGBoost model** with:
- **AUC: 0.839** (Strong discriminative ability)
- **Recall: 79.9%** (Catches 4 out of 5 churners)
- **Precision: 51.4%** (1 in 2 flagged customers will actually churn)
- **Optimal Threshold: 0.387** (Maximizes recall target)
""")

# --- Create input fields in the sidebar ---
st.sidebar.header('Enter Customer Data:')

col1, col2 = st.sidebar.columns(2)

# --- Column 1 Inputs ---
gender = col1.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = col1.selectbox('Senior Citizen?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])
InternetService = col1.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])
contract_risk_score = col1.selectbox('Contract Type', [3, 2, 1], 
                                     format_func=lambda x: {3: 'Month-to-Month (High Risk)', 2: 'One Year (Medium)', 1: 'Two Year (Low Risk)'}[x])
recency_score = col1.slider('Customer Tenure (Recency)', 1, 4, 3, 
                            help="4 = 0-6mo (High Risk), 3 = 6-12mo, 2 = 1-2yr, 1 = 2+ yr (Low Risk)")

# --- Column 2 Inputs ---
MonthlyCharges = col2.slider('Monthly Charges ($)', 18.0, 120.0, 75.0, 0.01)
support_services_count = col2.slider('Support Services (0-4)', 0, 4, 0, help="OnlineSecurity, Backup, DeviceProtection, TechSupport")
engagement_score = col2.slider('Engagement Score (0-8)', 0, 8, 3)
has_family = col2.selectbox('Has Family?', [0, 1], format_func=lambda x: {1: 'Yes (Lower Risk)', 0: 'No'}[x])
has_bundle = col2.selectbox('Phone + Internet Bundle?', [0, 1], format_func=lambda x: {1: 'Yes (Lower Risk)', 0: 'No'}[x])

# Advanced options in expander
with st.sidebar.expander("Advanced Options"):
    payment_risk_score = st.selectbox('Payment Method', 
        [0.453, 0.191, 0.167, 0.152], 
        format_func=lambda x: {0.453: 'Electronic Check (High Risk)', 
                               0.191: 'Mailed Check', 
                               0.167: 'Bank Transfer (Auto)', 
                               0.152: 'Credit Card (Auto)'}[x])
    fiber_no_support = st.selectbox('Fiber Optic Without Support?', [0, 1], 
                                    format_func=lambda x: {1: 'Yes (High Risk)', 0: 'No'}[x])
    paperless_billing_risk = st.selectbox('Paperless Billing (Non-AutoPay)?', [0, 1], 
                                          format_func=lambda x: {1: 'Yes', 0: 'No'}[x])
    no_protection = st.selectbox('No Protection Services?', [0, 1], 
                                 format_func=lambda x: {1: 'Yes (High Risk)', 0: 'No'}[x])
    streaming_count = st.slider('Streaming Services (0-2)', 0, 2, 0)
    is_streamer = st.selectbox('Is Streamer?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])
    has_any_support = st.selectbox('Has Any Support?', [0, 1], format_func=lambda x: {1: 'Yes', 0: 'No'}[x])


# --- 4. CREATE THE DATA DICTIONARY WITH DEFAULTS ---
default_data = {
    'frequency_score': 3,
    'monetary_score': 3,
    'customer_segment': 'Medium-Value',
    'price_per_service': 15.0,
    'charges_vs_median': 1.1,
    'spending_tier': 'Medium',
    'value_score': 4.0
}

# Combine inputs into a dictionary
customer_data_dict = {
    **default_data,
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'InternetService': InternetService,
    'contract_risk_score': contract_risk_score,
    'recency_score': recency_score,
    'payment_risk_score': payment_risk_score,
    'fiber_no_support': fiber_no_support,
    'MonthlyCharges': MonthlyCharges,
    'support_services_count': support_services_count,
    'engagement_score': engagement_score,
    'has_family': has_family,
    'paperless_billing_risk': paperless_billing_risk,
    'no_protection': no_protection,
    'has_bundle': has_bundle,
    'streaming_count': streaming_count,
    'is_streamer': is_streamer,
    'has_any_support': has_any_support
}

# --- 5. THE PREDICTION LOGIC ---
if st.sidebar.button('🔍 Run Risk Assessment', use_container_width=True):
    try:
        # Convert to DataFrame in correct order
        customer_data = pd.DataFrame(customer_data_dict, index=[0])
        customer_data = customer_data[feature_columns]
        
        # Get probability prediction
        probability = pipeline.predict_proba(customer_data)[0][1]
        
        # UPDATED: Use the optimal threshold from your results
        optimal_threshold = 0.387
        prediction = 1 if probability >= optimal_threshold else 0
        
        prob_percent = probability * 100
        
        # --- 6. DISPLAY THE RESULT ---
        st.subheader('📊 Prediction Result')
        
        # Create two columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Churn Probability", f"{prob_percent:.1f}%")
        with metric_col2:
            st.metric("Optimal Threshold", f"{optimal_threshold*100:.1f}%")
        with metric_col3:
            status = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOW RISK"
            st.metric("Risk Level", status)
        
        st.divider()
        
        if prediction == 1:
            st.error(f'### 🔴 HIGH CHURN RISK - Probability: {prob_percent:.1f}%')
            st.markdown(f"""
            This customer has a **{prob_percent:.1f}% chance of churning**.
            
            **Business Impact:**
            - **Retention Cost:** $63
            - **Potential Revenue Saved:** $1,200 (LTV)
            - **Expected ROI:** 12.3x (if 70% retention success rate)
            
            **Recommended Actions:**
            1. **Immediate:** Contact customer within 24 hours
            2. **Offer:** 
               - For Month-to-Month contracts: 1-year contract discount (15-20%)
               - For Fiber users without support: Free 3-month TechSupport trial
               - For new customers: "First Year Loyalty" discount
            3. **Follow-up:** Enroll in support services to increase engagement
            
            **Key Risk Factors for This Customer:**
            """)
            
            # Show which factors are contributing to high risk
            if contract_risk_score == 3:
                st.warning("⚠️ Month-to-Month contract (high flexibility to leave)")
            if recency_score >= 3:
                st.warning("⚠️ Relatively new customer (tenure < 12 months)")
            if fiber_no_support == 1:
                st.warning("⚠️ Fiber optic without support services")
            if payment_risk_score > 0.3:
                st.warning("⚠️ Risky payment method (electronic check)")
            if support_services_count == 0:
                st.warning("⚠️ No protection services subscribed")
                
        else:
            st.success(f'### 🟢 LOW CHURN RISK - Probability: {prob_percent:.1f}%')
            st.markdown(f"""
            This customer has a **{prob_percent:.1f}% chance of churning**.
            
            **Recommended Actions:**
            1. **Standard Care:** Continue regular engagement
            2. **Upsell Opportunity:** 
               - If not already: Offer streaming services
               - If no support: Suggest support bundle for peace of mind
            3. **Retention:** Thank you email with loyalty appreciation
            
            **Positive Factors:**
            """)
            
            if has_family == 1:
                st.success("✅ Has family (lower churn)")
            if has_bundle == 1:
                st.success("✅ Multi-service bundle (higher stickiness)")
            if support_services_count > 0:
                st.success("✅ Multiple support services (engaged customer)")
            if contract_risk_score <= 2:
                st.success("✅ Long-term contract (commitment)")

        # Display probability gauge
        st.divider()
        
        # Create a visual gauge
        col_gauge, col_info = st.columns([2, 1])
        
        with col_gauge:
            # Create a simple bar chart gauge
            gauge_data = pd.DataFrame({
                'Status': ['Churn Risk', 'Retention Safe'],
                'Percentage': [prob_percent, 100 - prob_percent]
            })
            st.bar_chart(gauge_data.set_index('Status'), height=200)
        
        with col_info:
            st.markdown(f"""
            **Model Performance:**
            - AUC: 0.839
            - Recall: 79.9%
            - Precision: 51.4%
            
            **Decision Logic:**
            - If > {optimal_threshold*100:.1f}% → Offer retention
            - If ≤ {optimal_threshold*100:.1f}% → Standard care
            """)
        
        st.divider()
        with st.expander("📋 Show Raw Data Sent to Model"):
            st.dataframe(customer_data)
            
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        st.info("Please ensure all inputs are valid and try again.")

# --- Footer ---
st.divider()
st.markdown("""
---
**Model Details:**
- Algorithm: XGBoost Classifier
- Training Data: 7,032 telecom customers
- Features: 24 engineered behavioral & RFM features
- Optimization: Recall-optimized (80% target)
""")