import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels
true_df["label"] = 1   # Real news
fake_df["label"] = 0   # Fake news

# Combine
data = pd.concat([true_df, fake_df], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# Use only the 'text' column (you can also try 'title')
X = data["text"]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Test
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
