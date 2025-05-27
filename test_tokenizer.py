import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

text = "This is a sample sentence."
tokens = word_tokenize(text.lower())
print(tokens)
