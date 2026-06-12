"""
CodeAlpha — Task 2: FAQ Chatbot
NLP-based FAQ chatbot using TF-IDF + Cosine Similarity (sklearn + nltk)
"""
import tkinter as tk
from tkinter import scrolledtext
import threading, datetime, re, sys, subprocess

# ── Auto-install deps ──────────────────────────────────────────────────────────
for pkg in ["scikit-learn", "nltk"]:
    try:
        __import__(pkg.replace("-", "") if pkg == "scikit-learn" else pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── FAQ Dataset (College/Tech topic) ──────────────────────────────────────────
FAQ_DATA = [
    # Python
    ("what is python",
     "Python is a high-level, interpreted programming language known for its simplicity and readability. It is widely used in AI, ML, web development, and data science."),
    ("why use python for ai",
     "Python has powerful libraries like TensorFlow, PyTorch, Scikit-learn, and NumPy which make it the top choice for AI and Machine Learning development."),
    ("how to install python",
     "Download Python from python.org, run the installer, and make sure to check 'Add Python to PATH'. Then verify with 'python --version' in terminal."),
    ("what is pip",
     "pip is the package manager for Python. You can install any library using 'pip install library_name' in your terminal."),
    # AI/ML
    ("what is artificial intelligence",
     "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think, learn, and solve problems like humans."),
    ("what is machine learning",
     "Machine Learning is a subset of AI where machines learn from data and improve their performance over time without being explicitly programmed."),
    ("what is deep learning",
     "Deep Learning is a subset of Machine Learning that uses neural networks with many layers to learn complex patterns from large amounts of data."),
    ("what is nlp",
     "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language."),
    ("what is a neural network",
     "A neural network is a series of algorithms that try to recognize patterns in data, inspired by the structure of the human brain with layers of interconnected nodes."),
    ("what is overfitting",
     "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new unseen data. It can be reduced using regularization or more data."),
    # CodeAlpha Internship
    ("what is codealpha",
     "CodeAlpha is a leading software development company offering internship programs in domains like AI, Web Development, and Cybersecurity, providing real-world experience to students."),
    ("how many tasks to complete",
     "You need to complete a minimum of 2 or 3 tasks out of the 4 assigned tasks to be eligible for the internship completion certificate."),
    ("where to submit tasks",
     "Tasks are submitted through the submission form shared in your respective WhatsApp group. Upload source code to GitHub and post a video explanation on LinkedIn."),
    ("what is the github repo format",
     "Name your GitHub repository as 'CodeAlpha_ProjectName'. For example, 'CodeAlpha_LanguageTranslator' or 'CodeAlpha_FAQChatbot'."),
    ("what perks do interns get",
     "Interns receive an Offer Letter, QR-verified Completion Certificate, Unique ID Certificate, Letter of Recommendation (based on performance), and placement support."),
    # General
    ("hello hi hey",
     "Hello! 👋 I'm your FAQ Assistant. Ask me anything about AI, Python, or the CodeAlpha internship!"),
    ("what can you do",
     "I can answer frequently asked questions about Artificial Intelligence, Python programming, Machine Learning concepts, and the CodeAlpha internship program!"),
    ("thank you thanks",
     "You're welcome! 😊 Feel free to ask more questions anytime."),
    ("who made you",
     "I was built as part of the CodeAlpha AI Internship — Task 2: FAQ Chatbot using Python NLP techniques like TF-IDF and Cosine Similarity."),
    ("what is cosine similarity",
     "Cosine similarity measures the angle between two vectors. In NLP, it compares how similar two text documents are based on their TF-IDF representation."),
    ("what is tfidf",
     "TF-IDF stands for Term Frequency-Inverse Document Frequency. It measures how important a word is in a document relative to a collection of documents."),
    ("what is tokenization",
     "Tokenization is the process of splitting text into smaller units called tokens (words or sentences). It is a fundamental step in NLP preprocessing."),
]

# ── NLP Preprocessing ──────────────────────────────────────────────────────────
stemmer   = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

questions    = [q for q, _ in FAQ_DATA]
answers      = [a for _, a in FAQ_DATA]
proc_q       = [preprocess(q) for q in questions]

vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(proc_q)

THRESHOLD = 0.15  # minimum similarity to give an answer

def get_answer(user_input):
    proc_input = preprocess(user_input)
    vec = vectorizer.transform([proc_input])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    best_idx  = sims.argmax()
    best_score = sims[best_idx]
    if best_score < THRESHOLD:
        return ("🤔 I'm not sure about that. Try rephrasing or ask about AI, Python, "
                "Machine Learning, or the CodeAlpha internship!", best_score)
    return answers[best_idx], best_score

# ── UI ─────────────────────────────────────────────────────────────────────────
BG      = "#0d0d1a"
PANEL   = "#14142b"
ACCENT  = "#7c3aed"
BOT_BG  = "#1e1b4b"
USER_BG = "#1a3a1a"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"
WHITE   = "#ffffff"
GREEN   = "#22c55e"
PURPLE  = "#a78bfa"

root = tk.Tk()
root.title("💬 FAQ Chatbot — CodeAlpha")
root.geometry("680x620")
root.resizable(False, False)
root.configure(bg=BG)

# Title
title_f = tk.Frame(root, bg=ACCENT, pady=10)
title_f.pack(fill="x")
tk.Label(title_f, text="💬  FAQ Chatbot",
         font=("Segoe UI", 17, "bold"), bg=ACCENT, fg=WHITE).pack()
tk.Label(title_f, text="Ask me about AI, Python & CodeAlpha Internship  •  NLP Powered",
         font=("Segoe UI", 9), bg=ACCENT, fg="#c4b5fd").pack()

# Chat area
chat_frame = tk.Frame(root, bg=BG)
chat_frame.pack(fill="both", expand=True, padx=16, pady=10)

chat_box = scrolledtext.ScrolledText(
    chat_frame, font=("Segoe UI", 10), wrap="word",
    bg=PANEL, fg=TEXT, bd=0, relief="flat",
    padx=12, pady=10, state="disabled",
    insertbackground=WHITE
)
chat_box.pack(fill="both", expand=True)

# Configure text tags
chat_box.tag_config("bot_name",  foreground=PURPLE, font=("Segoe UI", 9, "bold"))
chat_box.tag_config("bot_text",  foreground=TEXT,   font=("Segoe UI", 10))
chat_box.tag_config("user_name", foreground=GREEN,  font=("Segoe UI", 9, "bold"))
chat_box.tag_config("user_text", foreground="#86efac", font=("Segoe UI", 10))
chat_box.tag_config("meta",      foreground=SUBTEXT, font=("Segoe UI", 8))
chat_box.tag_config("divider",   foreground="#2d2d5e", font=("Segoe UI", 8))

# Suggestion chips
chips_lbl = tk.Label(root,
    text="💡 Try: 'What is AI?'  |  'How many tasks?'  |  'What is cosine similarity?'  |  'What is NLP?'",
    font=("Segoe UI", 8), bg=BG, fg=SUBTEXT, wraplength=640)
chips_lbl.pack(pady=(0, 4))

# Input area
input_frame = tk.Frame(root, bg=PANEL, pady=8, padx=10)
input_frame.pack(fill="x", padx=16, pady=(0, 12))

user_entry = tk.Entry(input_frame, font=("Segoe UI", 11),
                      bg="#1e1e3f", fg=TEXT, insertbackground=WHITE,
                      relief="flat", bd=0)
user_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(6, 8))

send_btn = tk.Button(input_frame, text="Send ➤",
                     font=("Segoe UI", 10, "bold"),
                     bg=ACCENT, fg=WHITE, relief="flat",
                     padx=16, pady=6, cursor="hand2")
send_btn.pack(side="right")

# ── Chat Logic ─────────────────────────────────────────────────────────────────
def append_chat(role, message, score=None):
    chat_box.config(state="normal")
    now = datetime.datetime.now().strftime("%H:%M")

    if role == "bot":
        chat_box.insert("end", f"\n🤖 Assistant  ", "bot_name")
        chat_box.insert("end", f"[{now}]\n", "meta")
        chat_box.insert("end", f"{message}\n", "bot_text")
        if score is not None:
            chat_box.insert("end", f"   match: {score:.0%}\n", "meta")
    else:
        chat_box.insert("end", f"\n👤 You  ", "user_name")
        chat_box.insert("end", f"[{now}]\n", "meta")
        chat_box.insert("end", f"{message}\n", "user_text")

    chat_box.insert("end", "─" * 60 + "\n", "divider")
    chat_box.config(state="disabled")
    chat_box.see("end")

def send_message(event=None):
    user_input = user_entry.get().strip()
    if not user_input:
        return
    user_entry.delete(0, "end")
    append_chat("user", user_input)
    send_btn.config(state="disabled", text="…")

    def worker():
        answer, score = get_answer(user_input)
        root.after(300, lambda: (
            append_chat("bot", answer, score),
            send_btn.config(state="normal", text="Send ➤")
        ))
    threading.Thread(target=worker, daemon=True).start()

send_btn.config(command=send_message)
user_entry.bind("<Return>", send_message)

# Welcome message
append_chat("bot",
    "Hi! 👋 I'm your AI-powered FAQ Assistant.\n"
    "I can answer questions about Artificial Intelligence, Python, "
    "Machine Learning, and the CodeAlpha internship.\n"
    "What would you like to know?")

root.mainloop()
