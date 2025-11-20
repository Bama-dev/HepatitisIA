from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load("./models/modelo_regresion_logistica.pkl")
scaler = joblib.load('./models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)  # IMPORTANTE
    prediction = model.predict(features_scaled)
    estado = "Vive" if prediction == 1 else "Muere"
    print(estado)
    return jsonify({"prediction": estado})


if __name__ == '__main__':
    app.run(debug=True)
    