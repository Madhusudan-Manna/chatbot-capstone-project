"""
app.py  -  Manna ChatBot (Streamlit Cloud)
Topic  : NLP / Gen AI
Method : TF-IDF + Cosine Similarity  (retrieval-based chatbot)
Run    : streamlit run app.py
"""

import re
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manna ChatBot",
    page_icon="🤖",
    layout="centered"
)

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# ── Build data + model (cached so it runs only once per session) ───────────────
@st.cache_resource
def build_model():
    questions = [
        "hi", "hello", "hey there", "good morning", "good evening", "good night",
        "what is your name", "who are you", "who made you",
        "how are you", "how are you doing",
        "what can you do", "how can you help me",
        "what is nlp", "what is natural language processing",
        "what is tfidf", "what is cosine similarity",
        "what is a chatbot", "what is retrieval based chatbot",
        "what is machine learning", "what is artificial intelligence",
        "what is deep learning", "what is data science",
        "thank you", "thanks a lot", "that was helpful",
        "bye", "goodbye", "see you later",
    ]

    answers = [
        "Hello! How can I help you today?",
        "Hi there! How can I help you today?",
        "Hey! Good to see you.",
        "Good morning! Hope you have a great day.",
        "Good evening! How can I assist you?",
        "Good night! Sweet dreams.",
        "I am Manna, a simple NLP chatbot built by Madhusudan Manna.",
        "I am Manna, your NLP chatbot assistant!",
        "I was built by Madhusudan Manna as a learning project on NLP.",
        "I am doing great, thanks for asking!",
        "I am doing great, thanks for asking!",
        "I can answer questions using NLP techniques like TF-IDF and Cosine Similarity.",
        "Just ask me a question and I will try my best to answer it!",
        "NLP (Natural Language Processing) helps computers read and understand human language.",
        "Natural Language Processing (NLP) enables machines to interpret and respond to human text.",
        "TF-IDF stands for Term Frequency-Inverse Document Frequency. It converts text into numbers so machines can compare sentences.",
        "Cosine Similarity measures how similar two text vectors are. Score of 1 = identical, 0 = completely different.",
        "A chatbot is a program that holds a text conversation with a human using rules or AI/NLP techniques.",
        "A retrieval-based chatbot finds the best matching pre-written answer using similarity techniques.",
        "Machine Learning is the ability of a computer to learn patterns from data without being explicitly programmed.",
        "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
        "Deep Learning uses neural networks with many layers to learn from large amounts of data.",
        "Data Science is the process of extracting insights and knowledge from structured and unstructured data.",
        "You are welcome!",
        "You are most welcome!",
        "Happy to help! Let me know if you need anything else.",
        "Goodbye! Have a nice day.",
        "Bye! Take care.",
        "See you soon!",
    ]

    # Clean questions
    clean_questions = [clean_text(q) for q in questions]

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clean_questions)

    return vectorizer, X, answers

# ── Get chatbot response ───────────────────────────────────────────────────────
def get_response(user_input: str, vectorizer, X, answers) -> str:
    cleaned = clean_text(user_input)
    if not cleaned:
        return "Please type something so I can help you!"
    user_vec   = vectorizer.transform([cleaned])
    scores     = cosine_similarity(user_vec, X)
    best_idx   = int(scores.argmax())
    best_score = float(scores[0][best_idx])
    if best_score < 0.25:
        return "I am not sure I understood that. Could you rephrase?"
    return answers[best_idx]

# ── Load model ─────────────────────────────────────────────────────────────────
vectorizer, X, answers = build_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 Manna ChatBot")
st.caption("NLP Chatbot · TF-IDF + Cosine Similarity · by Madhusudan Manna")
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Try asking:")
    for tip in [
        "hi / hello",
        "what is your name?",
        "what is NLP?",
        "what is TF-IDF?",
        "what is cosine similarity?",
        "what is machine learning?",
        "what can you do?",
        "thank you",
        "bye",
    ]:
        st.markdown(f"- `{tip}`")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Manna. Ask me anything about NLP or AI!"}
    ]

# ── Display conversation ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    reply = get_response(user_input, vectorizer, X, answers)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
