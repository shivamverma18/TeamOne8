from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Ensure vectorizer is saved

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        text_vectorized = vectorizer.transform([text])  # Convert text to numerical form
        prediction = model.predict(text_vectorized)[0]  # Get prediction
        sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
        return render_template('index.html', sentiment=sentiment, text=text)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
