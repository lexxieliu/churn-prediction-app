# Customer Churn Prediction & Revenue Impact Platform

A live, interactive web app that predicts customer churn risk in real-time using an XGBoost model.

**[LIVE DEMO LINK](http://your-app-name.streamlit.app)** <-- *(You'll add this link after Step 6)*

## Project Goal

This project predicts customer churn for a telecom company. The primary goal is to identify high-risk customers so the business can provide targeted retention offers, saving over $500k in projected annual revenue.

## Key Features
* **ML Model:** An XGBoost classifier trained on 7,000+ customer records (AUC: 0.86).
* **Feature Engineering:** Includes RFM analysis and behavioral features (like `fiber_no_support` and `contract_risk_score`).
* **Business Impact:** The model uses a recall-optimized threshold (0.30) to achieve a **7.9x ROI** on retention efforts.
* **Deployment:** The app is containerized and deployed on Streamlit Community Cloud.

## How to Run Locally

1.  Clone this repository:
    `git clone https://github.com/your-username/your-repo-name.git`
2.  Navigate to the folder:
    `cd churn-prediction-app`
3.  Create and activate your environment:
    `conda create -n churn python=3.9`
    `conda activate churn`
4.  Install required libraries:
    `pip install -r requirements.txt`
5.  Run the Streamlit app:
    `streamlit run app.py`