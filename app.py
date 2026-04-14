import streamlit as st
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from joblib import load

# ==============================
# LOAD MODEL & VECTORIZER
# ==============================
model = load("model.joblib")
vectorizer = load("vectorizer.joblib")

# ==============================
# TEXT CLEANING FUNCTION
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])

def preprocess(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

# ==============================
# STREAMLIT UI
# ==============================
st.title("🛍️ Women's Clothing Review Predictor")

st.write("Enter a customer review to predict whether the product is recommended.")

# Input box
user_input = st.text_area("Enter Review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess
        processed_text = preprocess(user_input)

        # Vectorize
        vectorized_text = vectorizer.transform([processed_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]

        # Output
        if prediction == 1:
            st.success("✅ Recommended")
        else:
            st.error("❌ Not Recommended")