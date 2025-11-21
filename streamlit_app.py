import streamlit as st
import pandas as pd
import numpy as np
import joblib
from text_processing import extract_features

# Load models
@st.cache_resource
def load_model_components():
    model = joblib.load("joblib/sentiment_model.joblib")
    vectorizer = joblib.load("joblib/tfidf_vectorizer.joblib")
    le = joblib.load("joblib/label_encoder.joblib")
    return model, vectorizer, le

model, vectorizer, le = load_model_components()


# UI

st.title("Tweet Sentiment Analysis Dashboard")
st.write("A chill and clean  data-driven sentiment analysis dashboard.")

st.set_page_config(layout="wide")  

# User Input 
st.header("Analyze a Tweet!")
user_input = st.text_area("Enter a tweet:", height=120)

button_clicked = st.button("Analyze")

# Tweet Sentiment Analyzing
if button_clicked:
    if user_input.strip():
        st.success("Tweet analyzed!")
        cleaned, polarity, subjectivity, vader, pos_count, neg_count = extract_features(user_input)

        tfidf_vec = vectorizer.transform([cleaned]).toarray()

        numerical = [[polarity, subjectivity, vader, pos_count, neg_count]]

        X_final = np.hstack((numerical, tfidf_vec))

        pred = model.predict(X_final)[0]
        sentiment_label = le.inverse_transform([pred])[0]
        if sentiment_label == "Irrelevant":
            sentiment_label = "Neutral"

        st.subheader("🔎 Sentiment Analysis Result")
        st.metric("Predicted Sentiment", sentiment_label)

        col1, col2, col3 = st.columns(3)
        col1.metric("Polarity", f"{polarity:.3f}")
        col1.caption("Measures how positive or negative the text is. " \
        "Ranges from -1 (very negative) to +1 (very positive).")

        col2.metric("Subjectivity", f"{subjectivity:.3f}")
        col2.caption("Measures how much of the text is based on opinions vs facts. " \
        "0 = very factual, 1 = very opinion-based.")

        col3.metric("VADER Score", f"{vader:.3f}")
        col3.caption("A sentiment score designed for social media text. " \
        "Captures emotion like slang, emojis, caps, and intensity. " \
        "Ranges from -1 (negative) to +1 (positive).")
    else:
        st.warning("Please enter a tweet first!")


st.header("📊 Dataset Sentiment Breakdown")

df = pd.read_csv("tweets.csv")
df = df[df["Sentiment"] != "Irrelevant"]

#Raw data display
st.subheader("Raw Dataset")
st.dataframe(df)

# Sentiment type distribution
st.subheader("Sentiment Distribution")
sentiment_counts = df["Sentiment"].value_counts()

st.bar_chart(sentiment_counts, height=500)