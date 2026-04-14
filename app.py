import streamlit as st
import joblib
import pandas as pd
import time
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="She Shops And We Pick", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"]).reset_index(drop=True)

# Create Product Names
df["Product Name"] = "Product " + df.index.astype(str)

# ---------------- CSS (FINAL FIXED) ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background-color: #1e3a8a;
}

/* ALL TEXT WHITE */
html, body, p, div, span, label {
    color: white !important;
}

/* INPUT TEXT BLACK */
textarea, input {
    color: black !important;
}

/* SELECTBOX FIX */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

/* DROPDOWN OPTIONS */
div[role="listbox"] {
    background-color: white !important;
    color: black !important;
}

/* BUTTON */
.stButton > button {
    border-radius: 12px;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white !important;
    font-weight: 600;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 70px;
    font-weight: 800;
}

/* CENTER TEXT */
.center {
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= LANDING PAGE =================
if st.session_state.page == "home":

    st.markdown('<p class="title">SHE SHOPS AND WE PICK</p>', unsafe_allow_html=True)

    st.markdown('<p class="center">AI that understands what customers truly feel</p>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.button("✨ Let’s Get Started"):
        st.session_state.page = "app"

# ================= MAIN APP =================
elif st.session_state.page == "app":

    st.markdown("## 🛍️ Choose a Product")

    selected_product = st.selectbox(
        "Select Product",
        df["Product Name"].unique()
    )

    product_data = df[df["Product Name"] == selected_product].iloc[0]
    review_text = product_data["Review Text"]

    st.markdown("### 📝 Customer Review")
    st.info(review_text)

    if st.button("🔍 Understand Customer Decision"):

        with st.spinner("Analyzing customer intention... 🤖"):
            time.sleep(1.5)

        # MODEL
        review_tfidf = vectorizer.transform([review_text])
        prediction = model.predict(review_tfidf)[0]
        prob = model.predict_proba(review_tfidf)[0]

        prob_yes = float(prob[1]) * 100

        # INTENTION LOGIC
        review_lower = review_text.lower()
        negative_words = ["not", "bad", "worst", "poor", "disappointed"]
        positive_words = ["love", "great", "perfect", "amazing", "good"]

        neg_count = sum(word in review_lower for word in negative_words)
        pos_count = sum(word in review_lower for word in positive_words)

        if neg_count > pos_count:
            prediction = 0

        st.markdown("---")

        # RESULT
        if prediction == 1:
            st.markdown(f"""
            <div style="background:#14532d; padding:20px; border-radius:12px;">
            ✅ <b>Recommended</b><br>
            Confidence: {prob_yes:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#7f1d1d; padding:20px; border-radius:12px;">
            ❌ <b>Not Recommended</b><br>
            Confidence: {100 - prob_yes:.1f}%
            </div>
            """, unsafe_allow_html=True)

        # WHY
        st.markdown("### 🧠 Why this decision?")

        if neg_count > pos_count:
            st.write("The customer shows dissatisfaction and negative intent in the review.")
        elif pos_count > neg_count:
            st.write("The customer expresses satisfaction and positive intent.")
        else:
            st.write("The review contains mixed opinions.")

        # CONFIDENCE
        st.markdown("### 📊 Confidence Level")
        score = prob_yes if prediction == 1 else 100 - prob_yes
        st.progress(int(score))
        st.write(f"{score:.1f}% confidence")

        # KEY WORDS
        st.markdown("### 🔑 Key Words Influencing Decision")
        words = review_text.lower().split()
        common = Counter(words).most_common(5)

        for word, count in common:
            st.write(f"{word} ({count})")

        # HIGHLIGHT
        st.markdown("### ✨ Important Words in Review")

        highlighted = review_text

        for word in positive_words:
            highlighted = highlighted.replace(
                word, f"<span style='color:#22c55e; font-weight:bold'>{word}</span>"
            )

        for word in negative_words:
            highlighted = highlighted.replace(
                word, f"<span style='color:#ef4444; font-weight:bold'>{word}</span>"
            )

        st.markdown(highlighted, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
