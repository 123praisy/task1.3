import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Product Recommendation Calculator", layout="wide")

st.plotly_chart(fig, key=f"gauge_{item}")

# ---------------- LOAD ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"])

# ---------------- SESSION STATE ----------------
if "cart" not in st.session_state:
    st.session_state.cart = []

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "best_product" not in st.session_state:
    st.session_state.best_product = None

if "confirm_purchase" not in st.session_state:
    st.session_state.confirm_purchase = False

if "purchased" not in st.session_state:
    st.session_state.purchased = None

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {background:#1e3a8a;color:white;}
textarea {background:white !important;color:black !important;}
button {background:#2563eb !important;color:white !important;border-radius:10px;}
button:hover {transform:scale(1.05);}

.metric-card {
    background:white;
    color:black;
    padding:15px;
    border-radius:12px;
    text-align:center;
    transition:0.3s;
}
.metric-card:hover {
    transform:scale(1.05);
    box-shadow:0 0 15px white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>PRODUCT RECOMMENDATION CALCULATOR</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Review Analysis","Model Performance","EDA Analysis","Dataset"])

# ================= REVIEW ANALYSIS =================
if page == "Review Analysis":

    st.subheader("🛒 Product Selection")

    products = df[['Clothing ID','Class Name']].drop_duplicates().dropna()
    options = products.apply(lambda x: f"{x['Clothing ID']} - {x['Class Name']}", axis=1)

    selected = st.selectbox("Choose Product", options)

    if st.button("Add to Cart"):
        st.session_state.cart.append(selected)
        st.success("Added to cart")

    st.write("### 🧺 Cart Items")
    for i, item in enumerate(st.session_state.cart):
        st.write("✔", item)

    # -------- ANALYZE BUTTON --------
    if st.button("Analyze Cart"):
        st.session_state.analysis_done = True

    # -------- ANALYSIS --------
    if st.session_state.analysis_done and st.session_state.cart:

        best_score = 0

        for i, item in enumerate(st.session_state.cart):

            pid = int(item.split(" - ")[0])
            data = df[df["Clothing ID"] == pid]

            text = " ".join(data["Review Text"])
            tfidf = vectorizer.transform([text])
            prob = model.predict_proba(tfidf)[0][1]

            pos = sum(data["Recommended IND"] == 1)
            neg = sum(data["Recommended IND"] == 0)

            st.markdown(f"## 🛍 {item}")

            c1,c2,c3,c4 = st.columns(4)

            c1.markdown(f"<div class='metric-card'>Recommendation<br><b>{prob:.2f}</b></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'>Positive<br><b>{pos}</b></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'>Negative<br><b>{neg}</b></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'>Total<br><b>{len(data)}</b></div>", unsafe_allow_html=True)

            # -------- CONFIDENCE BAR --------
            st.write("Confidence Level")

            colL, colM, colR = st.columns([1,6,1])
            colL.write("0%")

            progress = int(prob * 100)
            bar = colM.empty()

            for i in range(progress + 1):
                bar.progress(i)
                time.sleep(0.003)

            colR.write(f"{progress}%")

            # -------- GAUGE --------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Confidence"},
                gauge={'axis': {'range': [0, 100]}}
            ))

            st.plotly_chart(fig, key=f"gauge_{item}_{i}")  # ✅ FIXED

            if prob > best_score:
                best_score = prob
                st.session_state.best_product = item

        st.markdown("---")
        st.success(f"🏆 Best Product: {st.session_state.best_product} ⭐⭐⭐⭐⭐")

        purchase_clicked = st.button("Purchase Best Product")

if purchase_clicked:
    st.session_state.confirm_purchase = True

    # -------- PURCHASE FLOW --------
    if st.session_state.confirm_purchase and st.session_state.cart:

        st.subheader("🛍 Confirm Purchase")

        selected_purchase = st.selectbox(
            "Select product to purchase",
            st.session_state.cart
        )

        if st.button("Confirm Purchase"):
    st.session_state.purchased = selected_purchase
    st.session_state.confirm_purchase = False
    st.session_state.analysis_done = True
            st.success("🎉 Thank you for choosing highly recommended product!")

    # -------- REVIEW SECTION --------
    if st.session_state.purchased:

        st.markdown("---")
        st.subheader("📝 Please give your review on the product received")

        pid = st.session_state.purchased.split(" - ")[0]
        pname = st.session_state.purchased.split(" - ")[1]

        st.write(f"Product ID: {pid}")
        st.write(f"Product Name: {pname}")

        review = st.text_area("Write your review")

        if st.button("Submit Review"):

            df.loc[len(df)] = {
                "Clothing ID": int(pid),
                "Class Name": pname,
                "Review Text": review,
                "Recommended IND": 1
            }

            df.to_csv("Womens Clothing E-Commerce Reviews.csv", index=False)

            st.success("✅ Thank you for your review!")
            st.info("📁 New review added to dataset")

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.87")
    col1.metric("Precision", "0.85")
    col2.metric("Recall", "0.83")
    col2.metric("F1 Score", "0.84")

# ================= EDA =================
elif page == "EDA Analysis":

    st.subheader("📈 Class Distribution")
    fig = px.bar(df['Recommended IND'].value_counts())
    st.plotly_chart(fig)

    df['review_length'] = df['Review Text'].apply(len)

    st.subheader("📊 Review Length Distribution")
    fig = px.histogram(df, x='review_length')
    st.plotly_chart(fig)

    st.subheader("🔗 Correlation Matrix")
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

# ================= DATASET =================
elif page == "Dataset":
    st.dataframe(df.head(50))
