from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = float(request.form["gender"])
        age = float(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        duration = float(request.form["duration"])
        heart_rate = float(request.form["heart_rate"])
        body_temp = float(request.form["body_temp"])

        # Feature array
        features = np.array([[
            gender,
            age,
            height,
            weight,
            duration,
            heart_rate,
            body_temp
        ]])

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        output = round(prediction, 2)

        return render_template(
            "index.html",
            prediction_text=f"Calories Burned: {output}"
        )

    except Exception:
        return render_template(
            "index.html",
            prediction_text="Error: Invalid Input"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)