import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Modeling & Evaluation Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, classification_report, 
                             confusion_matrix, precision_recall_curve, roc_curve, auc)
# Imbalance Handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Model & Interpretability
import xgboost as xgb
import shap

import joblib
# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# 1. RECENCY: How recently customer joined (tenure-based)
# In this dataset, lower tenure = more recent = higher risk
df['recency_score'] = pd.cut(df['tenure'], 
                              bins=[0, 6, 12, 24, 72], 
                              labels=[4, 3, 2, 1],
                              include_lowest=True)  # 4=very recent (high risk)

df['recency_score'] = df['recency_score'].astype(int)

print("\nRECENCY SCORES (Tenure-based):")
print(df.groupby('recency_score')['Churn'].apply(lambda x: (x=='Yes').mean()*100))

# 2. FREQUENCY: Service usage intensity
# Count number of services subscribed
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']

# Create binary service indicators
df['service_count'] = 0
for col in service_cols:
    if col == 'InternetService':
        df['service_count'] += (df[col] != 'No').astype(int)
    else:
        df['service_count'] += (df[col] == 'Yes').astype(int)

df['frequency_score'] = pd.cut(df['service_count'],
                               bins=[-1, 2, 4, 6, 10],
                               labels=[1, 2, 3, 4])  # 4=high engagement
df['frequency_score'] = df['frequency_score'].astype(int)

print("\nFREQUENCY SCORES (Service count):")
print(df.groupby('frequency_score')['Churn'].apply(lambda x: (x=='Yes').mean()*100))

# 3. MONETARY: Revenue contribution
df['monetary_score'] = pd.cut(df['TotalCharges'],
                              bins=[0, 500, 2000, 4000, 10000],
                              labels=[1, 2, 3, 4],
                              include_lowest=True)  # 4=high value
df['monetary_score'] = df['monetary_score'].astype(int)

print("\nMONETARY SCORES (Total charges):")
print(df.groupby('monetary_score')['Churn'].apply(lambda x: (x=='Yes').mean()*100))

# 4. COMPOSITE RFM SCORE
df['rfm_score'] = (df['recency_score'] + 
                   df['frequency_score'] + 
                   df['monetary_score'])

# Create customer segments
df['customer_segment'] = pd.cut(df['rfm_score'],
                               bins=[0, 5, 8, 12],
                               labels=['At-Risk', 'Medium-Value', 'High-Value'])

print("\nCUSTOMER SEGMENTS (RFM-based):")
segment_analysis = df.groupby('customer_segment').agg({
    'Churn': lambda x: f"{(x=='Yes').mean()*100:.1f}%",
    'customerID': 'count',
    'MonthlyCharges': 'mean'
})
segment_analysis.columns = ['Churn_Rate', 'Customer_Count', 'Avg_Monthly_Revenue']
print(segment_analysis)

print("\n" + "="*60)
print("BEHAVIORAL RISK FEATURES")
print("="*60)

# 1. CONTRACT RISK SCORE
contract_risk_map = {
    'Month-to-month': 3,  # High risk
    'One year': 2,        # Medium risk
    'Two year': 1         # Low risk
}
df['contract_risk_score'] = df['Contract'].map(contract_risk_map)

# 2. PAYMENT RISK SCORE (based on observed churn rates)
payment_churn = df.groupby('PaymentMethod')['Churn'].apply(
    lambda x: (x=='Yes').mean()
).to_dict()
df['payment_risk_score'] = df['PaymentMethod'].map(payment_churn)

# 3. SENIOR CITIZEN FLAG (already 0/1)
# Keep as is

# 4. PAPERLESS BILLING RISK
# Customers with paperless billing + auto-pay might be less engaged
df['paperless_billing_risk'] = (
    (df['PaperlessBilling'] == 'Yes') & 
    (df['PaymentMethod'] != 'Bank transfer (automatic)') &
    (df['PaymentMethod'] != 'Credit card (automatic)')
).astype(int)

# 5. NO PROTECTION SERVICES FLAG
df['no_protection'] = (
    ((df['OnlineSecurity'] == 'No') | (df['OnlineSecurity'] == 'No internet service')) &
    ((df['OnlineBackup'] == 'No') | (df['OnlineBackup'] == 'No internet service')) &
    ((df['DeviceProtection'] == 'No') | (df['DeviceProtection'] == 'No internet service')) &
    ((df['TechSupport'] == 'No') | (df['TechSupport'] == 'No internet service'))
).astype(int)

# 6. FIBER OPTIC WITHOUT SUPPORT
df['fiber_no_support'] = (
    (df['InternetService'] == 'Fiber optic') &
    ((df['TechSupport'] == 'No') | (df['OnlineSecurity'] == 'No'))
).astype(int)

# Print churn rates for new features
risk_features = ['contract_risk_score', 'payment_risk_score', 
                'paperless_billing_risk', 'no_protection', 'fiber_no_support']

for feature in risk_features:
    print(f"\n{feature.upper()}:")
    if df[feature].dtype in ['int64', 'float64']:
        churn_by_feature = df.groupby(feature)['Churn'].apply(
            lambda x: (x=='Yes').mean()*100
        )
        for val, rate in churn_by_feature.items():
            print(f"  Value {val}: {rate:.1f}% churn rate")

#=============================================================================
# ENGAGEMENT & VALUE FEATURES
#=============================================================================

print("\n" + "="*60)
print("ENGAGEMENT & VALUE FEATURES")
print("="*60)

# 1. SUPPORT SERVICES ENGAGEMENT
support_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
df['support_services_count'] = 0
for service in support_services:
    df['support_services_count'] += (df[service] == 'Yes').astype(int)

df['has_any_support'] = (df['support_services_count'] > 0).astype(int)

# 2. STREAMING ENGAGEMENT
streaming_services = ['StreamingTV', 'StreamingMovies']
df['streaming_count'] = 0
for service in streaming_services:
    df['streaming_count'] += (df[service] == 'Yes').astype(int)

df['is_streamer'] = (df['streaming_count'] > 0).astype(int)

# 3. OVERALL ENGAGEMENT SCORE
df['engagement_score'] = (
    df['support_services_count'] + 
    df['streaming_count'] +
    (df['PhoneService'] == 'Yes').astype(int) +
    (df['MultipleLines'] == 'Yes').astype(int)
)

# 4. SERVICE DIVERSITY RATIO
total_possible_services = 9
df['service_diversity_ratio'] = df['service_count'] / total_possible_services

# 5. PHONE + INTERNET BUNDLE
df['has_bundle'] = (
    (df['PhoneService'] == 'Yes') & 
    (df['InternetService'] != 'No')
).astype(int)

# 6. FAMILY INDICATOR (Partner OR Dependents)
df['has_family'] = (
    (df['Partner'] == 'Yes') | 
    (df['Dependents'] == 'Yes')
).astype(int)

print("\nENGAGEMENT FEATURES - Churn Rates:")
print(f"\nSupport Services Count:")
print(df.groupby('support_services_count')['Churn'].apply(lambda x: (x=='Yes').mean()*100).round(1))

print(f"\nEngagement Score:")
print(df.groupby('engagement_score')['Churn'].apply(lambda x: (x=='Yes').mean()*100).round(1))

print(f"\nHas Bundle:")
print(df.groupby('has_bundle')['Churn'].apply(lambda x: (x=='Yes').mean()*100).round(1))

print("\n" + "="*60)
print("RATIO & EFFICIENCY FEATURES")
print("="*60)

# 1. PRICE PER SERVICE (Value perception)
df['price_per_service'] = df['MonthlyCharges'] / (df['service_count'] + 1)

# 2. AVERAGE MONTHLY SPEND RATE
df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)

# 3. CHARGES VS MEDIAN (Relative pricing)
median_monthly = df['MonthlyCharges'].median()
df['charges_vs_median'] = df['MonthlyCharges'] / median_monthly

# 4. SPENDING TIER
df['spending_tier'] = pd.cut(df['MonthlyCharges'],
                             bins=[0, 35, 70, 150],
                             labels=['Low', 'Medium', 'High'])

# 5. TENURE TO CHARGES RATIO (Loyalty value)
df['tenure_to_charges_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1)

# 6. VALUE PERCEPTION SCORE
df['value_score'] = df['service_count'] / (df['charges_vs_median'] + 0.1)

print("\nRATIO FEATURES - Statistics:")
ratio_features = ['price_per_service', 'avg_monthly_spend', 'charges_vs_median', 'value_score']
for feat in ratio_features:
    churned_mean = df[df['Churn']=='Yes'][feat].mean()
    retained_mean = df[df['Churn']=='No'][feat].mean()
    print(f"\n{feat}:")
    print(f"  Churned: {churned_mean:.2f}")
    print(f"  Retained: {retained_mean:.2f}")
    print(f"  Difference: {abs(churned_mean - retained_mean):.2f}")


# --- 1. Define Target (y) ---
df_model = df.copy()
df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})
target = 'Churn'

# --- 2. Define Features (X) ---

# Drop original/intermediate columns to use our final engineered features
cols_to_drop = [
    'customerID', 'Churn', 
    'tenure', 'TotalCharges', 'Contract', 'PaymentMethod', # Replaced by risk scores/recency
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', # Used in 'support_services_count' & 'no_protection'
    'StreamingTV', 'StreamingMovies', 'PhoneService', 'MultipleLines', # Used in 'engagement_score' & 'is_streamer'
    'Partner', 'Dependents', 'PaperlessBilling', # Used in 'has_family' & 'paperless_billing_risk'
    'rfm_score', 'service_count', # Intermediate scores, we use the components
    'avg_monthly_spend', 'tenure_to_charges_ratio', 'service_diversity_ratio' # Redundant or less predictive
]

X = df_model.drop(columns=cols_to_drop)
y = df_model[target]

# Define categorical and numerical feature lists for the pipeline
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# --- 3. Train-Test Split ---
# We use 'stratify=y' to ensure the test set has the same churn proportion as the full dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Final Numerical Features: {numerical_features}")
print(f"Final Categorical Features: {categorical_features}")

#=============================================================================
# PIPELINE CREATION & MODEL TRAINING
#=============================================================================
# --- 1. Create Preprocessing Steps ---

# Numerical pipeline: scale data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical pipeline: one-hot encode
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 2. Create the XGBoost Model ---
# We calculate scale_pos_weight to help XGBoost with the imbalance, even before SMOTE
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,  # Handles imbalance
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# --- 3. Create and Fit the Full Pipeline with SMOTE ---

# Use ImbPipeline to include SMOTE in the flow
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb_model)
])

print("Starting pipeline training...")
pipeline.fit(X_train, y_train)
print("✅ Pipeline training complete.")

#=============================================================================
# MODEL EVALUATION
#=============================================================================
# Get predicted probabilities for the positive class (Churn=1)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"✅ Model AUC Score: {auc_score:.3f}")

# Get standard 0.5 threshold predictions
y_pred = pipeline.predict(X_test)
print("\n--- Classification Report (Default 0.5 Threshold) ---")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))


print("Training final pipeline...")
pipeline.fit(X_train, y_train)
print("✅ Training complete.")

# Save the pipeline to a file
joblib.dump(pipeline, 'churn_model_pipeline.pkl')
print("✅ Pipeline saved to churn_model_pipeline.pkl")
