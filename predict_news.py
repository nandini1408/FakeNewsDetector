# predict_news.py
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# --- Download NLTK data (ensure these are downloaded) ---
# These downloads are crucial for text processing.
# They will only download if not already present.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Stop words (must be defined globally or passed) ---
stop_words = set(stopwords.words('english'))

# --- Clean text function (EXACTLY as in your fake_news_predictor.py) ---
def clean_text(text):
    """
    Cleans the input text by lowercasing, tokenizing using NLTK,
    removing non-alphabetic words, and removing stopwords.
    This function must be identical to the one used during model training.
    """
    if not isinstance(text, str): # Handle potential non-string inputs
        return ""
    tokens = word_tokenize(text.lower())
    # Filter out non-alphabetic tokens and stopwords
    return " ".join([word for word in tokens if word.isalpha() and word not in stop_words])

# --- Load model and vectorizer ---
# IMPORTANT: Ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory as this script.
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Error: Model files (model.pkl, vectorizer.pkl) not found.")
    print("Please ensure 'fake_news_predictor.py' has been run to generate these files in the same directory.")
    exit() # Exit the script if files are not found
except Exception as e:
    print(f"An error occurred while loading the model files: {e}")
    print("Please check if 'model.pkl' and 'vectorizer.pkl' are valid pickle files.")
    exit() # Exit the script if loading fails for other reasons

print("\n--- Fake News Detector (Command Line) ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter news article text:\n")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    if not user_input.strip(): # Check if input is empty or just whitespace
        print("Please enter some text.\n")
        continue

    # Clean, vectorize, and predict
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0]

    # Get confidence for 'REAL' and 'FAKE' based on model's class order
    # model.classes_ gives the order of classes (e.g., ['FAKE', 'REAL'] or ['REAL', 'FAKE'])
    real_proba = confidence[list(model.classes_).index('REAL')]
    fake_proba = confidence[list(model.classes_).index('FAKE')]

    # Print output
    print(f"\nCleaned Text: {cleaned}")
    print(f"Prediction: {prediction}")
    print(f"Confidence (REAL): {real_proba:.4f}")
    print(f"Confidence (FAKE): {fake_proba:.4f}\n")

