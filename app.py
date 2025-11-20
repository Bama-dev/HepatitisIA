from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("./models/modelo_regresion_logistica.pkl")
scaler = joblib.load('./models/scaler.pkl')

print(model.classes_)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predicción (clase 1 o 2)
    prediction = model.predict(features_scaled)[0]

    probas = model.predict_proba(features_scaled)[0]
    classes = model.classes_  # [1, 2]

    proba_dict = {classes[i]: probas[i] for i in range(len(classes))}

    # Sacar probabilidades según el significado real
    prob_vive = float(proba_dict[1] * 100)   # Clase 1 = Vive
    prob_muere = float(proba_dict[2] * 100)  # Clase 2 = Muere

    # Estado final
    estado = "Vive" if prediction == 1 else "Muere"

    return jsonify({
        "estado": estado,
        "prob_vive": prob_vive,
        "prob_muere": prob_muere
    })


if __name__ == '__main__':
    app.run()
