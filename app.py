
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "model/heatwave_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "latitude", "longitude", "year", "month", "day_of_year", "decade",
    "weather_code", "temp_max_c", "temp_min_c", "temp_range_c",
    "apparent_temp_max_c", "apparent_temp_min_c",
    "precipitation_mm", "rain_mm", "snowfall_cm", "precipitation_hours",
    "sunshine_duration_sec", "daylight_duration_sec",
    "wind_speed_max_kmh", "wind_gusts_max_kmh",
    "wind_direction_dominant_deg", "solar_radiation_mj_m2",
    "reference_evapotranspiration_mm", "city", "state"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Heatwave Prediction API",
        "status": "running",
        "prediction": "next_day_heatwave"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })

@app.route("/predict", methods=["POST"])
def predict():

    try:
        input_data = request.get_json()

        if input_data is None:
            return jsonify({"error": "JSON input required"}), 400

        missing = [
            col for col in FEATURE_COLUMNS
            if col not in input_data
        ]

        if missing:
            return jsonify({
                "error": "Missing features",
                "missing": missing
            }), 400

        input_df = pd.DataFrame([{
            col: input_data[col]
            for col in FEATURE_COLUMNS
        }])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        return jsonify({
            "prediction": prediction,
            "label": "Heatwave" if prediction == 1 else "No Heatwave",
            "heatwave_probability": round(probability, 4),
            "prediction_target": "next_day_heatwave"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
