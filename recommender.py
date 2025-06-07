from typing import List, Dict

# بيانات الكتب والمقالات (ممكن تعدل لتحميل من ملف أو قاعدة بيانات)
books = [
    {"title": "Book 1", "emotion": "joy"},
    {"title": "Book 2", "emotion": "sad"},
    {"title": "Book 3", "emotion": "love"},
]

articles = [
    {"title": "Article 1", "emotion": "joy"},
    {"title": "Article 2", "emotion": "sad"},
    {"title": "Article 3", "emotion": "love"},
]

def recommend_content(emotion: str) -> Dict[str, List[str]]:
    """ترجع توصيات كتب ومقالات بناءً على المشاعر"""
    recommended_books = [b["title"] for b in books if b["emotion"] == emotion]
    recommended_articles = [a["title"] for a in articles if a["emotion"] == emotion]
    return {
        "books": recommended_books,
        "articles": recommended_articles
    }
