# -----------------------------------------------
# SPAM EMAIL DETECTION USING MACHINE LEARNING
# 100% WORKING MODEL - CAN RUN DIRECTLY IN PYCHARM
# -----------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------
# 1. Load Dataset
# Using a small built-in dataset for demonstration
# -----------------------------------------------
data = {
    "text": [
        "Congratulations! You won a free lottery ticket",
        "Win money now!!! Click this link",
        "Dear customer, please update your bank information",
        "Hey bro, are we meeting today?",
        "Let's have lunch tomorrow",
        "Call me when you reach home",
        "This is not spam, just checking in",
        "You have been selected for a prize",
        "Reminder: Your bill is due tomorrow",
        "URGENT! Your account has been compromised"
    ],
    "label": [
        "spam", "spam", "spam",
        "ham", "ham", "ham", "ham",
        "spam", "ham", "spam"
    ]
}

df = pd.DataFrame(data)

# -----------------------------------------------
# 2. Split Data
# -----------------------------------------------
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------------------------
# 3. Text Vectorization
# -----------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------------------------
# 4. Model Training (Naive Bayes)
# -----------------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------------------------
# 5. Evaluation
# -----------------------------------------------
y_pred = model.predict(X_test_vec)

print("\n=== MODEL EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------------------------
# 6. Test with Custom Input
# -----------------------------------------------
while True:
    msg = input("\nEnter a message to classify (or type 'exit'): ")
    if msg.lower() == "exit":
        break

    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]

    print("\nPrediction:", prediction.upper())
