from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np # Make sure to import numpy

app = Flask(__name__)

# 1. Load the trained pipeline
# This happens only ONCE when the server starts, making it fast.
pipeline = joblib.load('churn_model_pipeline.pkl')

# 2. Define the exact columns the model was trained on
# This is CRITICAL to build the DataFrame correctly.
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

# 3. Create a URL "endpoint"
# This tells Flask: "When someone sends a POST request to http://.../predict, run this function."
@app.route('/predict', methods=['POST'])
def predict_churn():
    # 4. Get the data
    # The dashboard will send data in a format called JSON. This line gets it.
    data = request.get_json(force=True)
    
    try:
        # 5. Convert JSON back to a DataFrame
        # Your model was trained on a DataFrame, so we must re-create one.
        customer_data = pd.DataFrame(data, index=[0])
        
        # 6. Ensure column order is correct
        customer_data = customer_data[feature_columns] 
        
        # 7. Run the prediction
        # .predict_proba() gets the [Prob(No Churn), Prob(Churn)]
        probability = pipeline.predict_proba(customer_data)[0][1]
        
        # 8. Apply your business logic
        optimal_threshold = 0.30 
        prediction = 1 if probability >= optimal_threshold else 0
        
        # 9. Send the answer back
        # jsonify() converts your Python dictionary into JSON for the dashboard to read.
        return jsonify({
            'churn_prediction': prediction,
            'churn_probability': float(probability),
            'optimal_threshold': optimal_threshold
        })
        
    except Exception as e:
        # Send a clear error message if something fails (e.g., missing feature)
        return jsonify({'error': str(e)}), 400

# 10. Run the server
# This tells the script to start listening on port 5000.
if __name__ == '__main__':
    app.run(port=5000, debug=True)