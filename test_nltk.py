import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

text = "Hello, how are you doing today?"
tokens = word_tokenize(text.lower())
print(tokens)
