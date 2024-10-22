# Example usage
from main import predict_review_sentiment
new_review = input("Enter a new review: ")
prediction = predict_review_sentiment(new_review)

if prediction == 1:
    print("Positive")
else:
    print("Negative")