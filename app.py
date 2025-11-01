from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Initialize app
app = Flask(__name__)
CORS(app)  # 👈 This line fixes the CORS error!

# Load trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news_text = data.get("text", "")

    # Transform the text
    transformed_text = vectorizer.transform([news_text])

    # Predict
    prediction = model.predict(transformed_text)[0]
    confidence = np.max(model.predict_proba(transformed_text)) * 100

    result = {
        "prediction": "FAKE" if prediction == 0 else "REAL",
        "confidence": round(confidence, 2)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
    