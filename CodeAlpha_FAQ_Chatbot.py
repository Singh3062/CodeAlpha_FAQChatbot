# CodeAlpha Task 2: FAQ Chatbot
# Fixed version: removed NLTK stopwords download dependency

import tkinter as tk
from tkinter import scrolledtext
import threading
import datetime
import re
import sys
import subprocess

for pkg in ["scikit-learn", "nltk"]:
    try:
        __import__(pkg.replace("-", "") if pkg == "scikit-learn" else pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer


FAQ_DATA = [
    ("what is python", "Python is a high-level programming language used in AI, ML, web development and data science."),
    ("what is artificial intelligence", "AI is the simulation of human intelligence in machines."),
    ("what is machine learning", "Machine Learning allows systems to learn from data."),
    ("what is nlp", "NLP helps computers understand and process human language."),
    ("what is codealpha", "CodeAlpha provides internship programs in software development domains."),
    ("what is cosine similarity", "Cosine similarity measures similarity between text vectors."),
    ("what is tfidf", "TF-IDF measures word importance in documents.")
]

stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)


questions = [q for q, _ in FAQ_DATA]
answers = [a for _, a in FAQ_DATA]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([preprocess(q) for q in questions])


def get_answer(user_input):
    vec = vectorizer.transform([preprocess(user_input)])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    idx = scores.argmax()

    if scores[idx] < 0.15:
        return "I am not sure about that.", scores[idx]

    return answers[idx], scores[idx]


root = tk.Tk()
root.title("FAQ Chatbot - CodeAlpha")
root.geometry("680x620")

chat = scrolledtext.ScrolledText(root, wrap="word")
chat.pack(expand=True, fill="both", padx=10, pady=10)

entry = tk.Entry(root)
entry.pack(fill="x", padx=10)


def send_message(event=None):
    text = entry.get().strip()
    if not text:
        return

    entry.delete(0, "end")
    chat.insert("end", f"You: {text}\n")

    def worker():
        answer, score = get_answer(text)
        root.after(300, lambda: chat.insert("end", f"Assistant: {answer}\n\n"))

    threading.Thread(target=worker, daemon=True).start()


button = tk.Button(root, text="Send", command=send_message)
button.pack(pady=5)

entry.bind("<Return>", send_message)

chat.insert("end", "Assistant: Hi! Ask me about AI, Python, NLP or CodeAlpha.\n")

root.mainloop()
