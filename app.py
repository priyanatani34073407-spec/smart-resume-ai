import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
import json
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Resume AI", page_icon="🚀")

# ------------------ LOGIN SYSTEM ------------------
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

users = load_users()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.sidebar.title("🔐 Login System")
choice = st.sidebar.selectbox("Login/Register", ["Login", "Register"])

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if choice == "Register":
    if st.sidebar.button("Register"):
        users[username] = password
        save_users(users)
        st.sidebar.success("Registered Successfully!")

if choice == "Login":
    if st.sidebar.button("Login"):
        if users.get(username) == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success("Login Successful!")
        else:
            st.sidebar.error("Invalid Credentials")

# ✅ Logout Button
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.sidebar.success("Logged out successfully")
        st.rerun()

if not st.session_state.logged_in:
    st.warning("🔐 Welcome to Smart Resume AI Analyzer\n\nPlease login to continue")
    st.stop()

# ------------------ UI ------------------
st.markdown("""
<h1 style='text-align: center;'>🚀 Smart Resume AI Analyzer</h1>
<p style='text-align: center; color: grey;'>
Analyze your resume with AI and improve job matching
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("📌 Paste Job Description (Optional)")

# ------------------ TEXT EXTRACTION ------------------
def extract_text(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# ------------------ NLP ------------------
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
    "Backend Developer": ["django", "flask", "node", "sql"],
    "Frontend Developer": ["html", "css", "javascript", "react", "bootstrap"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "node", "sql"],
    "Software Developer": ["java", "c++", "data structures", "algorithms"],
    "Cloud Engineer": ["aws", "azure", "docker", "kubernetes"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "ci/cd"],
    "Cybersecurity Analyst": ["network security", "linux", "ethical hacking"],
    "Mobile App Developer": ["flutter", "android", "ios"],
    "UI/UX Designer": ["figma", "design", "wireframe"]
}

# ------------------ REPORT STORAGE ------------------
REPORT_FILE = "reports.json"

def load_reports():
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_reports(data):
    with open(REPORT_FILE, "w") as f:
        json.dump(data, f)

# ------------------ MAIN ------------------
if uploaded_file:
    if st.button("🔍 Analyze Resume"):

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
            st.markdown(f"**Match Score: {score}%**")
            st.progress(score / 100)

            st.write("✅ **Skills Found:**", found)
            st.write("❌ **Missing Skills:**", missing)
            st.markdown("---")

        # BEST ROLE
        best_score = max(scores_list)
        best_role = roles_list[scores_list.index(best_score)]

        st.subheader("🏆 Best Role")
        st.success(best_role)

        # ATS SCORE
        skills_score = int(sum(scores_list) / len(scores_list))
        project_score = 80 if "project" in text else 40
        experience_score = 75 if "experience" in text else 35

        overall_score = int((skills_score + project_score + experience_score) / 3)

        st.subheader("📊 Overall ATS Score")
        st.markdown(f"### {overall_score}%")
        st.progress(overall_score / 100)

        # NLP MATCH
        if job_desc:
            match_score = calculate_similarity(text, job_desc)

            st.subheader("📌 NLP Job Match")
            st.markdown(f"### 🔗 Match Score: {match_score}%")
            st.progress(match_score / 100)

        # AI INSIGHTS
        if st.button("🤖 Run Analysis"):
            st.subheader("🤖 Smart Insights")

            if "project" not in text:
                st.write("❌ Projects section missing")

            if "experience" not in text:
                st.write("⚠️ Add experience section")

            if "github" not in text:
                st.write("💡 Add GitHub link")

            if "developed" not in text:
                st.write("💡 Use action words like Developed, Built")

            for skill in set(all_missing):
                st.write(f"💡 Consider adding: {skill}")

        # SAVE REPORT
        if st.button("💾 Save Report"):
            reports = load_reports()
            user = st.session_state.username

            if user not in reports:
                reports[user] = []

            new_entry = {"score": overall_score, "role": best_role}

            if new_entry not in reports[user]:
                reports[user].append(new_entry)
                save_reports(reports)
                st.success("Report Saved Successfully ✅")
            else:
                st.warning("Report already saved ⚠️")

        # HISTORY
        reports = load_reports()
        user = st.session_state.username

        st.subheader("📁 Your Reports")
        if user in reports:
            for r in reports[user]:
                st.write(f"{r['role']} - {r['score']}%")

        # RANKING
        all_scores = []
        for u in reports:
            for r in reports[u]:
                all_scores.append(r["score"])

        if all_scores:
            rank = sum(1 for s in all_scores if s > overall_score) + 1

            st.subheader("🏆 Ranking")
            st.write(f"Rank #{rank} out of {len(all_scores)}")

        # CHART
        st.subheader("📊 Role Comparison")
        fig, ax = plt.subplots()
        ax.barh(roles_list, scores_list)
        ax.set_xlabel("Score (%)")
        st.pyplot(fig)

        # PDF DOWNLOAD
        if st.button("📥 Download Report"):
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph(f"Best Role: {best_role}", styles["Normal"]))
            story.append(Paragraph(f"ATS Score: {overall_score}%", styles["Normal"]))

            doc.build(story)

            with open("report.pdf", "rb") as f:
                st.download_button("Download", f)
