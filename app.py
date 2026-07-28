import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧"
)

# Title
st.title("📧 Email Spam Detection")

st.write("Enter an email or SMS message below to check whether it is Spam or Not Spam.")

# Text input
message = st.text_area("Enter Message")

# Predict button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Convert text into TF-IDF features
        message_vector = vectorizer.transform([message])

        # Predict
        prediction = model.predict(message_vector)

        # Display result
        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
