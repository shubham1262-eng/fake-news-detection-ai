import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SAMPLE DATASET
fake_news = [
    "Aliens attack Mumbai city",
    "Secret chip found inside vaccines",
    "Celebrity is actually robot",
    "World ending tomorrow",
    "Fake miracle cure discovered"
]

real_news = [
    "Government launches healthcare scheme",
    "Stock market reaches all time high",
    "India wins cricket match",
    "New education policy announced",
    "Scientists discover new species"
]

# CREATE DATA
texts = fake_news + real_news

labels = [0]*len(fake_news) + [1]*len(real_news)

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# MODEL
model = LogisticRegression()

model.fit(X, labels)

# SAVE MODEL
pickle.dump(model, open("model.pkl", "wb"))

# SAVE VECTORIZER
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model Trained Successfully")