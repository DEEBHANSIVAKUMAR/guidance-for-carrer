import numpy as np
import joblib
from flask import Flask, request, render_template,jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import csv

app = Flask(__name__)

CAREER_DETAILS = {
    "Software Engineer": ["Frontend Developer", "Backend Developer", "AI Engineer", "Cybersecurity Expert"],
    "Doctor": ["Surgeon", "Pediatrician", "Dermatologist", "Radiologist"],
    "Entrepreneur": ["Startup Founder", "Business Consultant", "E-commerce Specialist", "Marketing Strategist"],
    "Artist": ["Painter", "Graphic Designer", "Musician", "Animator"],
    "Scientist": ["Biotechnologist", "Astrophysicist", "Data Scientist", "Environmental Scientist"],
    "Lawyer": ["Corporate Lawyer", "Criminal Lawyer", "Human Rights Lawyer", "Intellectual Property Lawyer"],
    "Engineer": ["Mechanical Engineer", "Civil Engineer", "Electrical Engineer", "Chemical Engineer", "Industrial Engineer", "Software Engineer", "Computer Engineer", "Environmental Engineer", "Aeronautical Engineer", "Biomedical Engineer", "Systems Engineer"],
    "Psychologist": ["Clinical Psychologist", "Counseling Psychologist", "Forensic Psychologist", "Industrial Psychologist"],
    "Economist": ["Financial Analyst", "Policy Advisor", "Investment Banker", "Market Research Analyst"],
    "Journalist": ["News Reporter", "Investigative Journalist", "Editor", "Broadcast Journalist"]
}

def load_model():
    try:
        model_pipeline = joblib.load("career_model.pkl")
    except FileNotFoundError:
        X_train = np.array([
            [8, 7, 6, 5, 7, 9],
            [3, 9, 8, 6, 5, 4],
            [7, 6, 9, 8, 6, 7],
            [5, 9, 4, 7, 8, 5],
            [6, 5, 7, 9, 6, 8],
            [4, 6, 5, 8, 7, 9],
            [9, 5, 6, 4, 8, 7],
            [5, 8, 7, 6, 9, 4],
            [6, 7, 9, 5, 8, 6],
            [7, 6, 5, 9, 4, 8]
        ])
        y_train = list(CAREER_DETAILS.keys())

        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', max_iter=2000))
        ])

        model_pipeline.fit(X_train, y_train)
        joblib.dump(model_pipeline, "career_model.pkl")

    return model_pipeline

model_pipeline = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = np.array([[int(request.form[f'question{i}']) for i in range(1, 7)]])
        feedback_data = np.array([[int(request.form[f'feedback{i}']) for i in range(1, 7)]])
        combined_data = (user_data + feedback_data) / 2
        probabilities = model_pipeline.predict_proba(combined_data)[0]
        career_options = model_pipeline.named_steps['mlp'].classes_
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_careers = [(career_options[i], probabilities[i]) for i in top_indices]

        result = "\nTop Career Recommendations:\n"
        for rank, (career, prob) in enumerate(top_careers, 1):
            result += f"{rank}. {career} (Confidence: {prob:.2%})\n"
            if career in CAREER_DETAILS:
                result += "   Possible Roles: " + ", ".join(CAREER_DETAILS[career]) + "\n"

        return render_template('result.html', result=result)

    except ValueError:
        return render_template('index.html', error="Please enter valid integers between 1 and 10.")

@app.route('/exp.html')
def career_navigation():
    return render_template('exp.html')

@app.route('/map.html')
def index():
    return render_template('map.html')

@app.route('/trichy_colleges_data')
def trichy_colleges_data():
    try:
        colleges = pd.read_csv('trichy_colleges.csv')
        return jsonify(colleges.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)