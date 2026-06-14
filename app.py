import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page title
st.title("📧 Spam Email Detector")

# Input box
message = st.text_area("Enter Email Message")

# Predict button
if st.button("Detect Spam"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        data = vectorizer.transform([message])

        try:
            result = model.predict(data)[0]

            if result == 1 or str(result).lower() == "spam":
                st.error("🚨 Spam")
            else:
                st.success("✅ Ham (Not Spam)")

        except Exception as e:
            st.write("Prediction Error:", e)

# About section
st.markdown("---")
st.subheader("About")
st.write(
    "This Spam Email Detector uses Machine Learning and NLP techniques "
    "to classify emails as Spam or Ham."
)
