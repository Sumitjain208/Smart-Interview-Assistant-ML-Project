from textblob import TextBlob

def calculate_sentiment(text):

    sentiment = TextBlob(text).sentiment.polarity

    return sentiment