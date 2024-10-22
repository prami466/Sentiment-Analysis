import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
import pickle

# Download the stopwords dataset
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
data = pd.read_csv(r'C:\Users\prami\Downloads\sentiment_analysys\archive (1).zip')
print("Dataset shape:", data.shape)
data.head()

# Data info
data.info()

# Label encode sentiment to 1 (positive) and 0 (negative)
data.sentiment.replace('positive', 1, inplace=True)
data.sentiment.replace('negative', 0, inplace=True)
print("Sentiment counts:\n", data.sentiment.value_counts())

# Define function to remove HTML tags
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

# Apply HTML tag removal
data.review = data.review.apply(clean)

# Define function to remove special characters
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

# Apply special character removal
data.review = data.review.apply(is_special)

# Define function to convert text to lowercase
def to_lower(text):
    return text.lower()

# Apply lowercase conversion
data.review = data.review.apply(to_lower)

# Define function to remove stopwords
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

# Apply stopwords removal
data.review = data.review.apply(rem_stopwords)

# Define function to stem words
def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

# Apply stemming
data.review = data.review.apply(stem_txt)

# Creating Bag of Words (BOW)
X = np.array(data.iloc[:, 0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(data.review).toarray()

# Train-test split
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=9)
print("Train shapes : X = {}, y = {}".format(trainx.shape, trainy.shape))
print("Test shapes : X = {}, y = {}".format(testx.shape, testy.shape))

# Define models
gnb, mnb, bnb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True)

# Train models
gnb.fit(trainx, trainy)
mnb.fit(trainx, trainy)
bnb.fit(trainx, trainy)

# Prediction and accuracy metrics
ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)

print("Gaussian Accuracy = ", accuracy_score(testy, ypg))
print("Multinomial Accuracy = ", accuracy_score(testy, ypm))
print("Bernoulli Accuracy = ", accuracy_score(testy, ypb))

# Save the best model (BernoulliNB) to disk
pickle.dump(bnb, open('model1.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

# Test the model with a new review input
def predict_review_sentiment(review):
    # Load the vectorizer and model
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
    bnb = pickle.load(open('model1.pkl', 'rb'))

    # Preprocess the new review
    f1 = clean(review)
    f2 = is_special(f1)
    f3 = to_lower(f2)
    f4 = rem_stopwords(f3)
    f5 = stem_txt(f4)

    # Create BOW for the new review
    review_bow = cv.transform([f5]).toarray()

    # Predict sentiment
    y_pred = bnb.predict(review_bow)
    return y_pred[0]
