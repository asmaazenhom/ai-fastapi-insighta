from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import predict
from recommender import recommend_content
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8000"] if you want to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_emotion_endpoint(data: TextInput):
    emotion = predict(data.text)
    return {"emotion": emotion}

@app.post("/recommend")
def recommend_endpoint(data: TextInput):
    emotion = predict(data.text)
    recommendations = recommend_content(emotion)
    return {
        "emotion": emotion,
        "recommendations": recommendations
    }
