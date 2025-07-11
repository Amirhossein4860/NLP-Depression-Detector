import gradio as gr
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Load model
model = joblib.load("LogisticandBoWPipeline(Depression).joblib")

# Preprocessing function


def preprocess_text(text):
    contractions = {
        r"\b(ve)\b": "have", r"\b(ive)\b": "I've", r"\b(im)\b": "I'm",
        r"\b(ill)\b": "I'll", r"\b(it)\b": "it", r"\b(can t)\b": "can't",
        r"\b(wont)\b": "won't", r"\b(dont)\b": "don't", r"\b(isn t)\b": "isn't",
        r"\b(arent)\b": "aren't", r"\b(have)\b": "have", r"\b(willnot)\b": "won't",
        r"\b(wouldve)\b": "would've", r"\b(shouldve)\b": "should've",
        r"\b(mightve)\b": "might've", r"\b(mustve)\b": "must've", r"\b(didnt)\b": "didn't"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered = [lemmatizer.lemmatize(word.lower())
                for word in words if word.lower() not in stop_words]
    return " ".join(filtered)

# Emotion detection


def predict_emotion(text):
    if not text.strip():
        return "Please enter some text."
    processed = preprocess_text(text)
    prediction = model.predict([processed])[0]
    return f"Predicted Emotion: {prediction}"


# Gradio UI
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Type your text here..."),
    outputs="text",
    title="Depression Detector",
    description="Enter a comment and the model will predict user depression (1), or non-depression (0)."
)

iface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
