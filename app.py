import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
import json
import os

# 🔐 Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase safely
if "firebase" in st.secrets:
    if not firebase_admin._apps:
        cred = credentials.Certificate(st.secrets["firebase"])
        firebase_admin.initialize_app(cred)


db = firestore.client()


# NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Resume AI", page_icon="🚀")

# ------------------ LOGIN SYSTEM (Firebase) ------------------

def register_user(username, password):
    db.collection("users").document(username).set({
        "password": password
    })

def login_user(username, password):
    user = db.collection("users").document(username).get()
    
    if user.exists:
        return user.to_dict()["password"] == password
    return False

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.sidebar.title("🔐 Login System")
choice = st.sidebar.selectbox("Login/Register", ["Login", "Register"])

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if choice == "Register":
    if st.sidebar.button("Register"):
        register_user(username, password)
        st.sidebar.success("Registered Successfully!")

if choice == "Login":
    if st.sidebar.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success("Login Successful!")
        else:
            st.sidebar.error("Invalid Credentials")

if not st.session_state.logged_in:
    st.warning("Please login to continue")
    st.stop()

# ------------------ UI ------------------

st.markdown("<h1 style='text-align: center;'>🚀 Smart Resume AI</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("📌 Paste Job Description (Optional)")

# ------------------ TEXT EXTRACTION ------------------

def extract_text(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text.lower()

# ------------------ NLP FUNCTION ------------------

def calculate_similarity(resume_text, job_desc):
    cv = CountVectorizer(stop_words='english')
    vectors = cv.fit_transform([resume_text, job_desc]).toarray()
    similarity = cosine_similarity(vectors)[0][1]
    return int(similarity * 100)

# ------------------ ROLES ------------------

role_skills = {
    "Data Analyst": ["python", "sql", "excel", "power bi", "tableau"],
    "Web Developer": ["html", "css", "javascript", "react", "node"],
    "AI Engineer": ["python", "machine learning", "tensorflow", "pandas"],
    "Cloud Engineer": ["aws", "azure", "docker", "kubernetes"],
}

# ------------------ MAIN ------------------

if uploaded_file:
    text = extract_text(uploaded_file)

    st.success("Analysis Complete ✅")

    roles_list = []
    scores_list = []
    all_missing = []

    for role, skills in role_skills.items():
        found = [s for s in skills if s in text]
        missing = [s for s in skills if s not in text]

        score = int((len(found) / len(skills)) * 100)

        roles_list.append(role)
        scores_list.append(score)
        all_missing.extend(missing)

        st.subheader(f"🎯 {role}")
        st.write(f"Score: {score}%")
        st.progress(score / 100)

        st.write("✅ Found:", found)
        st.write("❌ Missing:", missing)
        st.markdown("---")

    # Best Role
    best_score = max(scores_list)
    best_role = roles_list[scores_list.index(best_score)]

    st.subheader("🏆 Best Role")
    st.success(best_role)

    # ATS Score
    overall_score = int(sum(scores_list) / len(scores_list))

    st.subheader(f"📊 ATS Score: {overall_score}%")
    st.progress(overall_score / 100)

    # Job Match
    if job_desc:
        match_score = calculate_similarity(text, job_desc)

        st.subheader("📌 NLP Job Match")
        st.write(f"{match_score}% Match")
        st.progress(match_score / 100)

    # Suggestions
    if st.button("🤖 Run Analysis"):
        st.subheader("🤖 Smart Insights")

        for skill in set(all_missing):
            st.write(f"💡 Consider adding: {skill}")

    # Chart
    st.subheader("📊 Role Comparison")
    fig, ax = plt.subplots()
    ax.barh(roles_list, scores_list)
    ax.set_xlabel("Score (%)")
    st.pyplot(fig)

    st.text_area("Resume Text", text, height=200)
