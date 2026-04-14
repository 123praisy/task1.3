import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Product Recommendation Calculator", layout="wide")

# ---------------- LOAD ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"])

# SESSION STATE
if "cart" not in st.session_state:
    st.session_state.cart = []
if "purchased" not in st.session_state:
    st.session_state.purchased = None
if "confirm_purchase" not in st.session_state:
    st.session_state.confirm_purchase = False

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

    st.write("### Cart Items")
    for item in st.session_state.cart:
        st.write("✔", item)

    # -------- ANALYZE --------
    if st.button("Analyze Cart"):

        best_product = None
        best_score = 0

        for item in st.session_state.cart:

            pid = int(item.split(" - ")[0])
            data = df[df["Clothing ID"] == pid]

            text = " ".join(data["Review Text"])
            tfidf = vectorizer.transform([text])
            prob = model.predict_proba(tfidf)[0][1]

            pos = sum(data["Recommended IND"]==1)
            neg = sum(data["Recommended IND"]==0)

            st.markdown(f"## {item}")

            c1,c2,c3,c4 = st.columns(4)

            c1.markdown(f"<div class='metric-card'>Recommendation<br><b>{prob:.2f}</b></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'>Positive<br><b>{pos}</b></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'>Negative<br><b>{neg}</b></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'>Total<br><b>{len(data)}</b></div>", unsafe_allow_html=True)

            # -------- PROGRESS BAR --------
            st.write("Confidence Level")

            colL,colM,colR = st.columns([1,6,1])
            colL.write("0%")

            progress = int(prob*100)
            holder = colM.empty()

            for i in range(progress+1):
                holder.progress(i)
                time.sleep(0.005)

            colR.write(f"{progress}%")

            # -------- GAUGE --------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text':"Confidence"},
                gauge={'axis':{'range':[0,100]}}
            ))
            st.plotly_chart(fig)

            if prob>best_score:
                best_score=prob
                best_product=item

        # -------- FINAL --------
        st.markdown("---")
        st.success(f"🏆 Best Product: {best_product} ⭐⭐⭐⭐⭐")

        if st.button("Purchase Best Product"):
            st.session_state.purchased = best_product
            st.session_state.confirm_purchase = True

    # -------- PURCHASE FLOW --------
    if st.session_state.confirm_purchase:

        st.subheader("🛍 Confirm Your Purchase")

        selected_purchase = st.selectbox(
            "Select product you want to purchase from cart",
            st.session_state.cart
        )

        if st.button("Confirm Purchase ✅"):
            st.session_state.purchased = selected_purchase
            st.session_state.confirm_purchase = False
            st.success("🎉 Thank you for choosing highly recommended product!")

    # -------- REVIEW AFTER PURCHASE --------
    if st.session_state.purchased and not st.session_state.confirm_purchase:

        st.markdown("---")
        st.subheader("📝 Please give your review on the product received")

        st.write(f"Product ID: {st.session_state.purchased.split(' - ')[0]}")
        st.write(f"Product Name: {st.session_state.purchased.split(' - ')[1]}")

        review = st.text_area("Write your review")

        if st.button("Submit Review"):

            pid = int(st.session_state.purchased.split(" - ")[0])
            pname = st.session_state.purchased.split(" - ")[1]

            df.loc[len(df)] = {
                "Clothing ID":pid,
                "Class Name":pname,
                "Review Text":review,
                "Recommended IND":1
            }

            df.to_csv("Womens Clothing E-Commerce Reviews.csv", index=False)

            st.success("✅ Thanks for your review!")
            st.info("📁 This review has been added to the dataset")

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.subheader("Model Performance")

    st.metric("Accuracy","0.87")
    st.metric("Precision","0.85")
    st.metric("Recall","0.83")
    st.metric("F1 Score","0.84")

# ================= EDA =================
elif page == "EDA Analysis":

    st.subheader("Class Distribution")
    fig = px.bar(df['Recommended IND'].value_counts())
    st.plotly_chart(fig)

    df['review_length'] = df['Review Text'].apply(len)

    st.subheader("Review Length")
    fig = px.histogram(df, x='review_length')
    st.plotly_chart(fig)

    st.subheader("Correlation Matrix")
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

# ================= DATASET =================
elif page == "Dataset":
    st.dataframe(df.head(50))
