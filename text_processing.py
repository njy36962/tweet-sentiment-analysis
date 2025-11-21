import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

abbreviations = {"thx": "thanks", "brb": "be right back", "ily": "I love you",
                 "lol": "laugh out loud", "idk": "I don't know", "pls": "please",
                 "smh": "shaking my head", "tbh": "to be honest", "imo": "in my opinion",
                 "btw": "by the way", "omg": "oh my god", "fyi": "for your information",
                 "lmk": "let me know", "np": "no problem", "rofl": "rolling on the floor laughing"}

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.,:;()'\-]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()

    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# ----------------------------
# Sentiment (TextBlob + VADER)
# ----------------------------
sia = SentimentIntensityAnalyzer()

def extract_features(text):
    cleaned = preprocess_text(text)

    tb = TextBlob(cleaned)
    polarity = tb.sentiment.polarity
    subjectivity = tb.sentiment.subjectivity

    vader_compound = sia.polarity_scores(cleaned)['compound']

    positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'love', 'like', 'awesome', 'happy']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'worst', 'boring', 'not', 'never', 'trash']

    pos_count = sum(1 for word in cleaned.split() if word in positive_words)
    neg_count = sum(1 for word in cleaned.split() if word in negative_words)

    return cleaned, polarity, subjectivity, vader_compound, pos_count, neg_count