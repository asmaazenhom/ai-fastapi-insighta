from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
import zipfile
import gdown
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Emotion-Based Content Recommendation API",
    description="API for recommending books and articles based on emotions",
    version="1.0.0"
)

# -----------------------------
# 🔽 Auto-download + unzip model files if missing
# -----------------------------
MODEL_PATH = Path("../models")
ZIP_PATH = MODEL_PATH / "model_files.zip"
EXTRACTED_FLAG = MODEL_PATH / "random_forest_model.pkl"  # أي ملف بيميز إن الموديل اتفك

if not EXTRACTED_FLAG.exists():
    print("🔽 Downloading model zip from Google Drive...")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Google Drive download
    file_id = "1iVfM4OkuWdAnsSyamD6slBEyxtvAZagw"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(ZIP_PATH), quiet=False)

    print("📦 Extracting model files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_PATH)

    print("✅ Models extracted successfully.")

# -----------------------------
# 📦 Load models
# -----------------------------
try:
    model = joblib.load(MODEL_PATH / "random_forest_model.pkl")
    vectorizer = joblib.load(MODEL_PATH / "tfidf_vectorizer.pkl")
    label_encoder = joblib.load(MODEL_PATH / "label_encoder.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# -----------------------------
# 📊 Load classified data
# -----------------------------
DATA_PATH = Path("../data")
try:
    books_df = pd.read_csv(DATA_PATH / "C:/Users/DELL/Downloads/project2/data/classified_books.csv")
    articles_df = pd.read_csv(DATA_PATH / "C:/Users/DELL/Downloads/project2/data/classified_articles.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# -----------------------------
# 📤 Schemas
# -----------------------------
class EmotionRequest(BaseModel):
    emotions: List[str]
    num_recommendations: Optional[int] = 3

class ContentRequest(BaseModel):
    text: str
    num_recommendations: Optional[int] = 3

# -----------------------------
# 🌐 Routes
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to Emotion-Based Content Recommendation API"}

@app.post("/recommend/emotions")
async def get_emotion_recommendations(request: EmotionRequest):
    try:
        book_recommendations = books_df[books_df['emotion'].isin(request.emotions)].head(request.num_recommendations)
        article_recommendations = articles_df[articles_df['emotion'].isin(request.emotions)].head(request.num_recommendations)

        return {
            "books": book_recommendations.to_dict(orient='records'),
            "articles": article_recommendations.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/content")
async def get_content_recommendations(request: ContentRequest):
    try:
        text_features = vectorizer.transform([request.text])
        emotion_pred = model.predict(text_features)
        emotion = label_encoder.inverse_transform(emotion_pred)[0]

        book_recommendations = books_df[books_df['emotion'] == emotion].head(request.num_recommendations)
        article_recommendations = articles_df[articles_df['emotion'] == emotion].head(request.num_recommendations)

        return {
            "predicted_emotion": emotion,
            "books": book_recommendations.to_dict(orient='records'),
            "articles": article_recommendations.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions")
async def get_available_emotions():
    try:
        emotions = label_encoder.classes_.tolist()
        return {"emotions": emotions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 🚀 Run locally if needed
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
