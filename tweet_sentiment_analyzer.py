import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from text_processing import extract_features


# Load dataset
df = pd.read_csv("tweets.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Label encoding
le = LabelEncoder()
df['Sentiment'] = le.fit_transform(df['Sentiment'])


# Apply feature extraction
df[['cleanedText', 'Polarity', 'Subjectivity', 'vaderCompound', 'positiveWordCount', 'negativeWordCount']] = \
    df['Text'].apply(lambda x: pd.Series(extract_features(x)))


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df[['cleanedText', 'Polarity', 'Subjectivity', 'vaderCompound', 'positiveWordCount', 'negativeWordCount']],
    df['Sentiment'],
    test_size=0.2,
    random_state=42
)

# ----------------------------
# TF-IDF (fit only on train)
# ----------------------------
vectorizer = TfidfVectorizer()
tfidf_train = vectorizer.fit_transform(X_train['cleanedText']).toarray()
tfidf_test = vectorizer.transform(X_test['cleanedText']).toarray()

# Combine features
numerical_features_train = X_train[['Polarity', 'Subjectivity', 'vaderCompound', 'positiveWordCount', 'negativeWordCount']].values
numerical_features_test = X_test[['Polarity', 'Subjectivity', 'vaderCompound', 'positiveWordCount', 'negativeWordCount']].values

X_train_final = np.hstack([numerical_features_train, tfidf_train])
X_test_final = np.hstack([numerical_features_test, tfidf_test])

# Train model
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ----------------------------
# Save pipeline parts
# ----------------------------
joblib.dump(model, "sentiment_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(le, "label_encoder.joblib")
