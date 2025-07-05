import nltk
from nltk.tokenize import word_tokenize

nltk.data.path.clear()
nltk.data.path.append(r"C:/nltk_data")

print(word_tokenize("Test after restart."))
