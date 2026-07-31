"""
app.py  —  Manna ChatBot (Streamlit)
Topic   : NLP / Gen AI
Method  : TF-IDF + Cosine Similarity (retrieval-based chatbot)
Run     : streamlit run app.py

NOTE: No pickle file needed — data and model are built directly here.
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

# ── Build Q&A data and train model (cached — runs only once per session) ───────
@st.cache_resource
def build_model():
    questions = [
        # Greetings
        "hi", "hello", "hey there", "good morning", "good evening", "good night",
        # About the bot
        "what is your name", "who are you", "who made you",
        "how are you", "how are you doing",
        "what can you do", "how can you help me",
        # NLP concepts
        "what is nlp", "what is natural language processing",
        "what is tfidf", "what is cosine similarity",
        "what is a chatbot", "what is retrieval based chatbot",
        # ML / AI concepts
        "what is machine learning", "what is artificial intelligence",
        "what is deep learning", "what is data science",
        # Gratitude
        "thank you", "thanks a lot", "that was helpful",
        # Farewell
        "bye", "goodbye", "see you later",
    ]

    answers = [
        # Greetings
        "Hello! How can I help you today? 😊",
        "Hi there! How can I help you today? 😊",
        "Hey! Good to see you.",
        "Good morning! Hope you have a great day. 🌞",
        "Good evening! How can I assist you?",
        "Good night! Sweet dreams. 🌙",
        # About the bot
        "I am Manna 🤖, a simple NLP chatbot built by Madhusudan Manna.",
        "I am Manna, your NLP chatbot assistant!",
        "I was built by Madhusudan Manna as a learning project on NLP.",
        "I am doing great, thanks for asking! 😊",
        "I am doing great, thanks for asking!",
        "I can answer questions using NLP techniques like TF-IDF and Cosine Similarity.",
        "Just ask me a question and I will try my best to answer it!",
        # NLP concepts
        "NLP (Natural Language Processing) is the field that helps computers read and understand human language.",
        "Natural Language Processing (NLP) enables machines to interpret, understand, and respond to human text.",
        "TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text into numbers so machines can compare sentences.",
        "Cosine Similarity measures how similar two text vectors are. A score of 1 means identical, 0 means completely different.",
        "A chatbot is a software program that can hold a text conversation with a human using rules or AI/NLP techniques.",
        "A retrieval-based chatbot finds the best matching pre-written answer from a dataset using similarity techniques.",
        # ML / AI concepts
        "Machine Learning is the ability of a computer to learn patterns from data without being explicitly programmed.",
        "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
        "Deep Learning is a subset of Machine Learning that uses neural networks with many layers to learn from large data.",
        "Data Science is the process of extracting insights and knowledge from structured and unstructured data.",
        # Gratitude
        "You are welcome! 😊",
        "You are most welcome!",
        "Happy to help! Let me know if you need anything else.",
        # Farewell
        "Goodbye! Have a nice day. 👋",
        "Bye! Take care. 👋",
        "See you soon! 👋",
    ]

    # Build DataFrame
    df = pd.DataFrame({"question": questions, "answer": answers})

    # Clean questions
    df["clean_question"] = df["question"].apply(clean_text)

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_question"])

    return vectorizer, X, df


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


# ── Get chatbot response ───────────────────────────────────────────────────────
def get_response(user_input: str, vectorizer, X, df) -> str:
    cleaned = clean_text(user_input)
    if not cleaned:
        return "Please type something so I can help you! 😊"
    user_vec   = vectorizer.transform([cleaned])
    scores     = cosine_similarity(user_vec, X)
    best_idx   = int(scores.argmax())
    best_score = float(scores[0][best_idx])
    if best_score < 0.25:
        return "I am not sure I understood that. Could you rephrase? 🤔"
    return df["answer"].iloc[best_idx]


# ── Load model ─────────────────────────────────────────────────────────────────
vectorizer, X, df = build_model()

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
        {"role": "assistant", "content": "Hello! I am Manna 🤖. Ask me anything!"}
    ]

# ── Display conversation ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message here…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    reply = get_response(user_input, vectorizer, X, df)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
