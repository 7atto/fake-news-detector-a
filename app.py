import streamlit as st
import pickle
import os

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector (Arabic + English)")
st.write("Enter a news text and the AI will detect whether it's **fake** or **real**.")

# Load saved model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "..", "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

news_text = st.text_area("✍️ Write news text here (Arabic or English):")

if st.button("Check"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text!")
    else:
        X = vectorizer.transform([news_text])
        prediction = model.predict(X)[0]

        if prediction == 0:
            st.error("🔴 **Fake News**")
        else:
            st.success("🟢 **Real News**")
