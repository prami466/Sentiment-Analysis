from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the model and vectorizer
cv = pickle.load(open('vectorizer.pkl', 'rb'))
bnb = pickle.load(open('model1.pkl', 'rb'))

# Function to preprocess text
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower(text):
    return text.lower()

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

def preprocess_text(review):
    review = clean(review)
    review = is_special(review)
    review = to_lower(review)
    review = rem_stopwords(review)
    review = stem_txt(review)
    return review

def predict_sentiment(review):
    review = preprocess_text(review)
    review_bow = cv.transform([review]).toarray()
    prediction = bnb.predict(review_bow)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form['text']
        prediction = predict_sentiment(review)
        if prediction == 1:
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'
        return render_template('index.html', translated_text=sentiment)
    return render_template('index.html', translated_text=None)

if __name__ == '__main__':
    app.run(debug=True)
