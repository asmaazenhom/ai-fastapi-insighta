
from typing import List, Dict
from collections import Counter
from datetime import date
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import hstack, csr_matrix
import joblib
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def get_wordnet_pos(tag: str):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words
    ]
    return ' '.join(lemmatized_words)

def extract_additional_features(text: str) -> list:
    words = word_tokenize(text)
    num_words = len(words)
    num_chars = len(text)
    sentiment = sia.polarity_scores(text)['compound']
    readability = textstat.flesch_reading_ease(text)
    return [num_words, num_chars, sentiment, readability]

def predict_emotion(text: str, model, vectorizer, label_encoder, scaler) -> str:
    processed = preprocess_text(text)
    vect_text = vectorizer.transform([processed])
    additional = np.array([extract_additional_features(processed)])
    additional_scaled = scaler.transform(additional)
    combined_features = hstack([vect_text, csr_matrix(additional_scaled)])
    prediction = model.predict(combined_features)
    return label_encoder.inverse_transform(prediction)[0]

mood_map = {
    'sadness': ['joy', 'surprise'],
    'anger': ['love', 'joy'],
    'fear': ['love', 'joy'],
    'joy': ['joy','surprise', 'love'],
    'surprise': ['love', 'joy'],
    'love': ['joy', 'love', 'surprise']
}

def recommend_content(emotion: str, top_n: int = 5) -> Dict[str, List[Dict]]:
    try:
        books_df = pd.read_csv("./data/classified_books.csv")
        articles_df = pd.read_csv("./data/classified_articles.csv")
        target_emotions = mood_map.get(emotion, ['joy'])
        books = books_df[books_df['emotion'].isin(target_emotions)].head(top_n)
        articles = articles_df[articles_df['emotion'].isin(target_emotions)].head(top_n)
        return {
            "books": books[["title", "authors", "emotion"]].to_dict(orient="records"),
            "articles": articles[["title", "url", "emotion"]].to_dict(orient="records")
        }
    except Exception as e:
        print(f"⚠️ Error loading recommendations: {e}")
        return {"books": [], "articles": []}

def generate_daily_emotion_report(
    posts: List[str],
    model,
    vectorizer,
    label_encoder
) -> Dict:
    processed_posts = [preprocess_text(p) for p in posts]
    X_tfidf = vectorizer.transform(processed_posts)

    predictions = model.predict(X_tfidf)
    labels = label_encoder.inverse_transform(predictions)

    emotion_counts = dict(Counter(labels))
    total = len(posts)
    emotion_distribution = {emotion: round((count / total) * 100, 2) for emotion, count in emotion_counts.items()}

    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(emotion_distribution.keys()), y=list(emotion_distribution.values()), palette='Set2')
    plt.title("توزيع المشاعر اليومي")
    plt.ylabel("%")
    plt.xlabel("المشاعر")
    plt.tight_layout()
    plt.savefig("daily_emotion_chart.png")
    plt.close()

    negative_emotions = ['anger', 'sadness', 'fear']
    negative_posts = [(text, emo) for text, emo in zip(posts, labels) if emo in negative_emotions]
    top_negative = negative_posts[:3]

    if top_negative:
        dominant_negative = Counter([e for _, e in top_negative]).most_common(1)[0][0]
    else:
        dominant_negative = "joy"

    recommendations = recommend_content(emotion=dominant_negative)

    report = {
        "date": str(date.today()),
        "total_posts": total,
        "emotion_distribution": emotion_distribution,
        "top_negative_posts": [{"text": t, "emotion": e} for t, e in top_negative],
        "recommendations": recommendations
    }

    with open("daily_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("📄 تم إنشاء تقرير اليوم: daily_report.json")
    print("📊 الرسم البياني محفوظ في: daily_emotion_chart.png")
    return report

# تحميل النموذج والمكونات
model = joblib.load("./models/random_forest_model.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("./models/label_encoder.pkl")

# منشورات اليوم
daily_posts = [
    "I feel really anxious about school tomorrow.",
    "Had fun with my friends today!",
    "Why is everything going wrong in my life?",
    "Watched a peaceful documentary about nature.",
    "I feel so loved and appreciated."
]

generate_daily_emotion_report(
    posts=daily_posts,
    model=model,
    vectorizer=vectorizer,
    label_encoder=label_encoder
)
