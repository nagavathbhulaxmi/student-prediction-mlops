
from flask import Flask, request, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Performance Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #eef2ff;
            text-align: center;
            padding: 40px;
        }
        .container {
            background: white;
            padding: 30px;
            margin: auto;
            max-width: 420px;
            border-radius: 15px;
            box-shadow: 0 4px 15px #cccccc;
        }
        input, button {
            width: 90%;
            padding: 12px;
            margin: 10px 0;
            box-sizing: border-box;
        }
        button {
            background: #4f46e5;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 6px;
        }
        .result {
            color: #15803d;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Student Prediction</h1>
        <p>Enter student details below</p>

        <form method="POST">
            <input type="number" name="study_hours"
                placeholder="Study hours per day"
                min="0" max="24" step="any" required>

            <input type="number" name="attendance"
                placeholder="Attendance percentage"
                min="0" max="100" step="any" required>

            <input type="number" name="previous_marks"
                placeholder="Previous marks"
                min="0" max="100" step="any" required>

            <button type="submit">Predict Marks</button>
        </form>

        {% if prediction is not none %}
            <p class="result">
                Predicted Marks: {{ prediction }} / 100
            </p>
            <p>This is an estimate, not a guaranteed result.</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_marks = float(request.form["previous_marks"])

        input_data = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks
        }])

        prediction = round(float(model.predict(input_data)[0]), 2)
        prediction = max(0, min(100, prediction))

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
