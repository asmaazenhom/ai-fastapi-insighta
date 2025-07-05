# Emotion-Based Content Recommendation System

This project implements an emotion-based content recommendation system that suggests books and articles based on user emotions. The system uses natural language processing and machine learning to classify user input into emotions and then recommends suitable content accordingly. A FastAPI backend is provided to serve the system as an API.

## 💡 Features

- Emotion classification (joy, sadness, fear, anger, love, surprise)
- Text preprocessing (lemmatization, TF-IDF vectorization)
- Handles imbalanced data using SMOTE
- Sentiment scoring and readability analysis
- Book & article recommendation based on emotional state
- Daily emotional impact report generation
- RESTful API using FastAPI

## 📁 Project Structure

```
project2/
├── data/
│   ├── books.csv
│   ├── articles.csv
│   ├── tweet_emotions.csv
│   ├── classified_books.csv
│   └── classified_articles.csv
│
├── models/
│   ├── random_forest_model.pkl
│   ├── random_forest_model.zip
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── backend/
│   ├── app.py                  # FastAPI application
│   ├── recommender.py          # Filtering logic
│   ├── model_utils.py          # Preprocessing & prediction utils
│   ├── daily_emotion_report.py # Daily analysis logic
│   └── requirements.txt
│
├── grad_proj.ipynb            # Main notebook with experiments
└── README.md                  # You are here!
```

## 📊 Data Files

- books.csv: Raw book data (title, author, content)
- articles.csv: Raw article content
- tweet_emotions.csv: Emotion-labeled text (training data)
- classified_books.csv / classified_articles.csv: Processed & emotion-tagged versions

## 🤖 Model Files

- random_forest_model.pkl: Trained emotion classifier
- tfidf_vectorizer.pkl: Vectorizes text for model input
- label_encoder.pkl: Maps encoded emotion labels

> Note: random_forest_model.zip is a compressed version (~168MB)

## 🚀 API Endpoints (via FastAPI)

Base URL: http://localhost:8000

| Endpoint                  | Method | Description                               |
|---------------------------|--------|-------------------------------------------|
| /                         | GET    | Welcome message                           |
| /predict                  | POST   | Predicts emotion from text                |
| /recommend                | POST   | Recommend content based on emotion        |
| /daily-report             | POST   | Generate daily emotional report           |

Sample /predict input:
```json
{
  "text": "I feel very anxious about my exams."
}
```

Sample /recommend input:
```json
{
  "text": "I feel so stressed lately."
}
```

## 🛠️ Running the Backend

1. Navigate to backend folder:
   ```bash
   cd backend
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn app:app --reload
   ```
4. Visit Swagger docs:
   http://localhost:8000/docs

## 📓 Jupyter Notebook (grad_proj.ipynb)

- Preprocesses data
- Trains & evaluates classifier
- Applies emotion filtering
- Visualizes insights

## 🧪 Example: Daily Report

Inside daily_emotion_report.py you can run:

```python
from daily_emotion_report import generate_daily_emotion_report

posts = [
  "I had a really bad day.",
  "Feeling grateful and happy today!",
  "I miss my family so much.",
  "Loved the book I just finished!"
]

generate_daily_emotion_report(posts, model, tfidf, label_encoder)
```

Creates:
- daily_report.json
- daily_emotion_chart.png

## 📦 Dependencies

Main:
- numpy, pandas, scikit-learn, nltk, textstat, joblib, matplotlib, seaborn, wordcloud

Backend:
- fastapi, uvicorn, pydantic, python-dotenv

## 📄 License

MIT License © 2025 Ahmed Elgabrey

Feel free to fork, star, and contribute! 💖
