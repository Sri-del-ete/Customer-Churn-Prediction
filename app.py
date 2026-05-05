from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model/churn_model.h5")
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Correct feature order (IMPORTANT)
feature_order = [
    "tenure","MonthlyCharges","TotalCharges",
    "Contract","InternetService",
    "TechSupport","OnlineSecurity","PaymentMethod"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(f)) for f in feature_order]

        final_input = np.array([features])
        final_input = scaler.transform(final_input)

        prediction = model.predict(final_input)[0][0]
        prob = round(prediction * 100, 2)

        if prediction > 0.5:
            result = f"⚠️ Likely to Churn ({prob}%)"
        else:
            result = f"✅ Likely to Stay ({100 - prob}%)"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)