
import streamlit as st
import pickle
import re

# Load model and vectorizer
with open("Models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("Models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean text function
def clean_text(text):
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    return text

# Streamlit app
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter a news headline and article below to check if it's likely *Fake* or *Real*.")

title = st.text_input("News Title")
content = st.text_area("News Content", height=200)

if st.button("Analyze"):
    combined = title + " " + content
    cleaned = clean_text(combined)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 0:
        st.success("✅ This looks like REAL news.")
    else:
        st.error("🚫 This appears to be FAKE news.")
