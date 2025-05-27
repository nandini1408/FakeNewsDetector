# app.py
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# --- Streamlit App Interface Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Download NLTK data (ensure these are downloaded for the Streamlit environment) ---
# This might take a moment the first time the app runs or if the data isn't cached.
# Changed exception from nltk.downloader.DownloadError to LookupError for compatibility
print("Checking NLTK data for Streamlit app...")
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
print("NLTK data check complete for Streamlit app.")


# --- Load the Saved Model and Vectorizer ---
# Ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory as this script.
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    st.success("Machine Learning Model and Vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model files (model.pkl, vectorizer.pkl) not found. Please ensure 'fake_news_predictor.py' has been run to generate these files in the same directory.")
    st.stop() # Stop the app if files are not found, preventing further errors.
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()

# --- Preprocessing Function (MUST match the one used in fake_news_predictor.py) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() # Initialize lemmatizer

def clean_text(text):
    """
    Cleans the input text by lowercasing, tokenizing,
    removing non-alphabetic words, removing stopwords, and applying lemmatization.
    This function must be identical to the one used in the model training script (fake_news_predictor.py).
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

st.title("📰 Fake News Detector")
st.markdown("""
    Paste the text of a news article below, and our model will predict if it's **REAL** or **FAKE**.
    """)

# Text area for user input
user_input = st.text_area("News Article Content:", height=300, placeholder="Type or paste your news article here...")

# Prediction button
if st.button("Analyze News"):
    if user_input:
        with st.spinner("Analyzing text and making a prediction..."):
            # 1. Clean the user input using the same function as training
            cleaned_input = clean_text(user_input)

            # 2. Vectorize the cleaned input using the loaded TF-IDF vectorizer
            # .transform expects an iterable (list of strings), so pass [cleaned_input]
            vectorized_input = vectorizer.transform([cleaned_input])

            # 3. Make the prediction
            prediction = model.predict(vectorized_input)
            # Get prediction probabilities for confidence
            prediction_proba = model.predict_proba(vectorized_input)

            # 4. Display the result
            st.subheader("Prediction Result:")
            if prediction[0] == 'REAL':
                # The probabilities are ordered by the classes learned during training.
                # To get the probability for 'REAL', we need to know its index.
                # LogisticRegression.classes_ will give the order.
                real_proba_index = list(model.classes_).index('REAL')
                confidence = prediction_proba[0][real_proba_index] * 100
                st.success(f"This news article is predicted as: **REAL NEWS** (Confidence: {confidence:.2f}%)")
                st.balloons() # A little celebration for real news!
            else:
                fake_proba_index = list(model.classes_).index('FAKE')
                confidence = prediction_proba[0][fake_proba_index] * 100
                st.error(f"This news article is predicted as: **FAKE NEWS** (Confidence: {confidence:.2f}%)")
                st.snow() # A subtle effect for fake news

            st.markdown("---")
            st.markdown("### How this prediction works:")
            st.info("""
                This application uses a **Logistic Regression** machine learning model.
                The model was trained on a dataset of real and fake news articles.
                Before prediction, the text is cleaned (lowercased, tokenized, stopwords removed, non-alphabetic words filtered).
                Then, it's converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**,
                which helps the model understand the importance of words in the context of the article and the entire dataset.
                """)
    else:
        st.warning("Please paste some news article text into the box to get a prediction.")

st.markdown("---")
st.markdown("Developed by a CS Student for learning purposes. This is a simplified model and should not be used for critical decisions.")
