# Emotion-Based Content Recommendation System

This project implements an emotion-based content recommendation system that suggests books and articles based on user emotions. The system uses natural language processing and machine learning to classify and recommend content.

## Project Structure

```
project/
├── data/                    # Data directory
│   ├── books.csv           # Raw book dataset
│   ├── articles.csv        # Raw article dataset
│   ├── tweet_emotions.csv  # Emotion dataset from tweets
│   ├── classified_books.csv    # Processed and classified books
│   └── classified_articles.csv # Processed and classified articles
│
├── models/                  # Models directory
│   ├── random_forest_model.pkl     # Trained Random Forest model
│   ├── random_forest_model.zip     # Compressed version of the model
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer for text processing
│   └── label_encoder.pkl           # Label encoder for emotion categories
│
├── backend/                # Backend API
│   ├── main.py            # FastAPI application
│   └── requirements.txt   # Backend dependencies
│
└── grad_proj.ipynb         # Main Jupyter notebook containing the implementation
```

## Data Files

### Raw Data
- `books.csv`: Contains the raw book dataset with titles, authors, and content
- `articles.csv`: Contains the raw article dataset with titles and content
- `tweet_emotions.csv`: Dataset of tweets labeled with emotions, used for training the emotion classifier

### Processed Data
- `classified_books.csv`: Books that have been processed and classified with emotions
- `classified_articles.csv`: Articles that have been processed and classified with emotions

## Model Files

### Random Forest Model
- `random_forest_model.pkl`: The trained Random Forest classifier for emotion classification
- `random_forest_model.zip`: Compressed version of the model for easier storage and transfer
  - Size: ~1.9GB (uncompressed), ~168MB (compressed)
  - Purpose: Classifies text into emotion categories

### Supporting Models
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer used for text feature extraction
  - Size: ~160KB
  - Purpose: Converts text into numerical features for the classifier

- `label_encoder.pkl`: Label encoder for emotion categories
  - Size: ~516B
  - Purpose: Converts emotion labels to numerical values and vice versa

## Backend API

The project includes a FastAPI backend that provides RESTful endpoints for the recommendation system.

### API Endpoints

1. `GET /`: Welcome message
2. `GET /emotions`: Get list of available emotions
3. `POST /recommend/emotions`: Get recommendations based on emotions
   - Request body: `{"emotions": ["happy", "sad"], "num_recommendations": 3}`
4. `POST /recommend/content`: Get recommendations based on content similarity
   - Request body: `{"text": "your text here", "num_recommendations": 3}`

### Running the Backend

1. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Start the server:
   ```bash
   python main.py
   ```

3. Access the API documentation at `http://localhost:8000/docs`

## Implementation

The main implementation is in `grad_proj.ipynb`, which contains:
- Data preprocessing and feature extraction
- Model training and evaluation
- Content recommendation system
- Analysis and visualization of results

## Usage

1. Ensure all data files are in the `data/` directory
2. Load the trained models from the `models/` directory
3. Run the notebook `grad_proj.ipynb` to:
   - Process new content
   - Generate emotion-based recommendations
   - Analyze results
4. Use the backend API to integrate the recommendation system into other applications

## Dependencies

### Main Project
- numpy
- pandas
- scikit-learn
- nltk
- textstat
- scipy
- joblib
- matplotlib
- seaborn
- wordcloud

### Backend
- fastapi
- uvicorn
- pydantic
- python-multipart
- python-dotenv

## Note

The model files are quite large, especially the Random Forest model. The compressed version (`random_forest_model.zip`) is provided for easier storage and transfer. Make sure to unzip it before using the model. 