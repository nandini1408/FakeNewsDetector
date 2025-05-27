# fake_news_predictor.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re

# --- Download NLTK data (should be done once) ---
# These downloads are crucial for text processing.
# They will only download if not already present.
# We'll use a more general exception handling for robustness.
print("Checking NLTK data...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Use LookupError as indicated by NLTK traceback
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
print("NLTK data check complete.")


# --- Load datasets ---
# IMPORTANT: Ensure 'Fake.csv' and 'True.csv' are in the same directory as this script.
# You will need to download these from a fake news dataset (e.g., from Kaggle).
try:
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")
    print("\nDatasets loaded successfully.")
except FileNotFoundError:
    print("Error: 'Fake.csv' or 'True.csv' not found. Please download the datasets and place them in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading datasets: {e}")
    exit()

# Add labels
fake['label'] = 'FAKE'
real['label'] = 'REAL'

# Combine and shuffle the datasets
df = pd.concat([fake, real])
df = df[['text', 'label']] # Select only relevant columns
df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle for randomness

print("\nSample of combined and shuffled data:")
print(df.head())
print(f"\nTotal data points: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# --- Stop words and Lemmatizer ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() # Initialize lemmatizer

# --- Clean text function (uses lemmatization for better accuracy) ---
def clean_text(text):
    """
    Cleans the input text by lowercasing, tokenizing,
    removing non-alphabetic words, removing stopwords, and applying lemmatization.
    This function must be identical to the one used in the Streamlit app (app.py).
    """
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    # Remove non-alphabetic tokens, remove stopwords, and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(cleaned_tokens)

# Clean entire dataset
print("\nCleaning text... (this might take a minute depending on dataset size)")
df['cleaned_text'] = df['text'].fillna('').apply(clean_text)
print("Text cleaning complete.")

# Features and labels
X = df['cleaned_text']
y = df['label']

# --- Vectorize text using TF-IDF ---
# ngram_range=(1,2) includes both single words and two-word phrases (bigrams)
# max_features limits the number of features to the top 10,000 most frequent ones
print("\nVectorizing text using TF-IDF...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000) # Added ngram_range and max_features for better performance
X_tfidf = tfidf.fit_transform(X)
print(f"TF-IDF vectorization complete. Number of features: {X_tfidf.shape[1]}")

# --- Train-test split ---
# Splitting data into training and testing sets for model evaluation.
print("\nSplitting data into training and testing sets (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- Train Logistic Regression model ---
# Logistic Regression is a good baseline classifier for text data.
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, solver='liblinear') # 'liblinear' solver is good for smaller datasets and L1/L2 regularization
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate model accuracy ---
# Evaluate the model's performance on the unseen test data.
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy on test set: {accuracy:.4f}")

# --- Save model and vectorizer ---
# Saving the trained model and vectorizer to disk for later use in the web app.
print("\nSaving trained model and TF-IDF vectorizer...")
try:
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(tfidf, vectorizer_file)
    print("Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'.")
except Exception as e:
    print(f"Error saving files: {e}")

# --- Example Prediction (for verification) ---
print("\n--- Example Prediction ---")
sample_news = """The government announced a new policy today to reduce pollution across major cities, aiming for a 20% decrease in emissions by 2030."""
print(f"Sample news: '{sample_news}'")
cleaned_sample = clean_text(sample_news)
vectorized_sample = tfidf.transform([cleaned_sample]) # Use tfidf (the trained vectorizer) here
prediction_sample = model.predict(vectorized_sample)[0]
confidence_sample = model.predict_proba(vectorized_sample)[0]

# Get confidence for 'REAL' and 'FAKE' based on model's class order
real_proba_index = list(model.classes_).index('REAL')
fake_proba_index = list(model.classes_).index('FAKE')
real_confidence = confidence_sample[real_proba_index]
fake_confidence = confidence_sample[fake_proba_index]

print(f"Cleaned sample text: {cleaned_sample}")
print(f"Prediction: {prediction_sample}")
print(f"Confidence (REAL): {real_confidence:.4f}")
print(f"Confidence (FAKE): {fake_confidence:.4f}")
