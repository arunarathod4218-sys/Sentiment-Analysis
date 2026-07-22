import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():

    df = pd.read_csv("reviews.csv")

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(df["review"])

    y = df["sentiment"]

    model = MultinomialNB()

    model.fit(X, y)

    return model, vectorizer