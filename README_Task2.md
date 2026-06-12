# 💬 FAQ Chatbot — CodeAlpha AI Internship Task 2

An NLP-powered FAQ chatbot that answers questions about AI, Python, and the CodeAlpha internship using TF-IDF and Cosine Similarity.

## 📸 Features
- NLP preprocessing: tokenization, stopword removal, stemming
- TF-IDF vectorization for question representation
- Cosine Similarity for finding best matching FAQ
- 20+ FAQs on AI, Python, ML, NLP, and CodeAlpha internship
- Match confidence score displayed for each answer
- Clean dark-themed chat UI

## 🛠️ Tech Stack
- **Python 3.x**
- **Tkinter** — Chat UI
- **scikit-learn** — TF-IDF Vectorizer + Cosine Similarity
- **NLTK** — Stopwords + PorterStemmer

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install scikit-learn nltk
```

### 2. Run the app
```bash
python chatbot.py
```
> Dependencies auto-install on first run if not found.

## 📁 Project Structure
```
CodeAlpha_FAQChatbot/
│
└── chatbot.py          # Main application file
```

## 🧠 How It Works
1. FAQs are preprocessed using NLTK (lowercase → remove stopwords → stem)
2. TF-IDF matrix is built from all FAQ questions
3. User input is preprocessed the same way
4. Cosine similarity finds the closest matching question
5. Corresponding answer is returned as chatbot response

## 💬 Sample Questions to Try
- "What is artificial intelligence?"
- "What is cosine similarity?"
- "How many tasks to complete?"
- "What is Python used for?"
- "What is overfitting?"

---
Made with ❤️ for CodeAlpha AI Internship
