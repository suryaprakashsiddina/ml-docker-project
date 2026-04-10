from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# HTML UI
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Model Prediction</title>
</head>
<body style="font-family: Arial; text-align: center; margin-top: 50px;">
    <h2>ML Model Prediction App 🚀</h2>
    <p>This app predicts output based on the trained ML model.</p>
    
    <form action="/predict" method="post">
        <input type="number" name="input_value" placeholder="Enter value" required>
        <br><br>
        <button type="submit">Predict</button>
    </form>

    {% if prediction is not none %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

# Home page
@app.route("/")
def home():
    return render_template_string(html_page, prediction=None)

# Prediction route (handles both UI + API)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # From UI (form)
        if request.form:
            input_value = float(request.form["input_value"])
        
        # From API (JSON)
        else:
            input_value = float(request.json["input"])

        prediction = model.predict(np.array([[input_value]]))[0]

        # If request is from browser form → render page
        if request.form:
            return render_template_string(html_page, prediction=prediction)

        # If request is API → return JSON
        return jsonify({"prediction": prediction})

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
