# Customer Churn Prediction & Revenue Impact Platform

A live, interactive web app that predicts customer churn risk in real-time using an XGBoost model.

**[LIVE DEMO LINK]([http://your-app-name.streamlit.app](https://lexxie-churn-prediction-app.streamlit.app/))

## Project Goal

This project predicts customer churn for a telecom company. The primary goal is to identify high-risk customers so the business can provide targeted retention offers, saving over $500k in projected annual revenue.

## Visualizations 
<img width="787" height="393" alt="Screenshot 2025-11-03 at 6 29 43 PM" src="https://github.com/user-attachments/assets/b6583eac-72a7-4f50-9cd8-2e24e84fdbc8" />
<img width="847" height="547" alt="Screenshot 2025-11-03 at 6 30 32 PM" src="https://github.com/user-attachments/assets/a513b633-f4bc-4971-90e3-8a727d40dbf6" />
<img width="686" height="571" alt="Screenshot 2025-11-03 at 6 31 11 PM" src="https://github.com/user-attachments/assets/b3c100a3-17c6-4cec-9222-ef19fdf9beed" />
<img width="774" height="375" alt="Screenshot 2025-11-03 at 6 31 25 PM" src="https://github.com/user-attachments/assets/7dcb70ee-00f9-4d08-93ee-a35fda4bc302" />
<img width="677" height="438" alt="Screenshot 2025-11-03 at 6 31 41 PM" src="https://github.com/user-attachments/assets/d0263f7d-31b3-433c-9da9-1ebfc4f19e70" />
<img width="594" height="449" alt="Screenshot 2025-11-03 at 6 32 14 PM" src="https://github.com/user-attachments/assets/77b057e2-9907-4e05-94a0-da0cb1417e0e" />
<img width="1085" height="988" alt="image" src="https://github.com/user-attachments/assets/5a86e224-360d-4c93-952b-d8a265513e21" />
<img width="1028" height="589" alt="Screenshot 2025-11-03 at 6 33 16 PM" src="https://github.com/user-attachments/assets/b3e93f3d-8634-4094-8c9a-1d692e92b57f" />
<img width="834" height="760" alt="Screenshot 2025-11-03 at 6 33 49 PM" src="https://github.com/user-attachments/assets/67701407-03cd-4ebd-aef4-39559400fb1f" />
<img width="770" height="752" alt="Screenshot 2025-11-03 at 6 34 03 PM" src="https://github.com/user-attachments/assets/a198eea5-3ecd-48ac-987b-66815abbfdcb" />

## Dashboard App page
<img width="1470" height="802" alt="Screenshot 2025-11-03 at 5 32 12 PM" src="https://github.com/user-attachments/assets/02b172e6-7acc-4e6a-b6aa-6ad7b766885e" />
<img width="1470" height="798" alt="Screenshot 2025-11-03 at 5 33 21 PM" src="https://github.com/user-attachments/assets/d37356ac-5031-461b-b714-d0e42f0d2a8c" />

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
