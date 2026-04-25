import streamlit as st
import PyPDF2
import json
import os
import hashlib
import re
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from io import BytesIO

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Resume AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #00C9A7, #845EC2, #00C9A7);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    text-align: center;
    margin-bottom: 0.2rem;
}

@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #00C9A755;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 20px rgba(0,201,167,0.08);
}

.role-card {
    background: linear-gradient(135deg, #12121f, #1a1a2e);
    border-left: 4px solid #00C9A7;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
}

.best-role-badge {
    background: linear-gradient(135deg, #00C9A7, #845EC2);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    padding: 1rem 2rem;
    border-radius: 50px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,201,167,0.3);
}

.suggestion-item {
    background: #1a1a2e;
    border: 1px solid #845EC255;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    color: #ccc;
    font-size: 0.9rem;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #00C9A7;
    border-bottom: 2px solid #00C9A730;
    padding-bottom: 0.3rem;
    margin: 1.5rem 0 1rem 0;
}

.rank-badge {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #FFD700;
    text-align: center;
}

.stButton > button {
    background: linear-gradient(135deg, #00C9A7, #845EC2);
    color: white;
    border: none;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    padding: 0.6rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,201,167,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,201,167,0.5);
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00C9A7, #845EC2) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
USERS_FILE = "users.json"
REPORTS_FILE = "reports.json"

# ─────────────────────────────────────────────
# HELPER: PASSWORD HASHING
# ─────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ─────────────────────────────────────────────
# JSON STORAGE HELPERS
# ─────────────────────────────────────────────
def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
# AUTH FUNCTIONS
# ─────────────────────────────────────────────
def register_user(email, password):
    users = load_json(USERS_FILE)
    email = email.strip().lower()
    if email in users:
        return False, "User already exists!"
    users[email] = hash_password(password)
    save_json(USERS_FILE, users)
    return True, "Registered successfully!"

def login_user(email, password):
    users = load_json(USERS_FILE)
    email = email.strip().lower()
    if email in users and users[email] == hash_password(password):
        return True
    # Legacy: plain text passwords for old accounts
    if email in users and users[email] == password:
        return True
    return False

# ─────────────────────────────────────────────
# REPORT HELPERS
# ─────────────────────────────────────────────
def save_report(email, role, score, ats_score):
    reports = load_json(REPORTS_FILE)
    email = email.strip().lower()
    if email not in reports:
        reports[email] = []
    reports[email].append({
        "role": role,
        "score": score,
        "ats_score": ats_score,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    save_json(REPORTS_FILE, reports)

def get_ranking(email, ats_score):
    reports = load_json(REPORTS_FILE)
    email = email.strip().lower()
    all_scores = []
    for user, user_reports in reports.items():
        if user_reports:
            best = max(r.get("ats_score", r.get("score", 0)) for r in user_reports)
            all_scores.append((user, best))
    all_scores.sort(key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, (u, _) in enumerate(all_scores) if u == email), len(all_scores))
    return rank, len(all_scores)

# ─────────────────────────────────────────────
# RESUME TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.lower()

# ─────────────────────────────────────────────
# ROLE SKILL DEFINITIONS (12 roles)
# ─────────────────────────────────────────────
ROLE_SKILLS = {
    "Data Analyst": {
        "skills": ["python", "sql", "excel", "power bi", "tableau", "statistics", "pandas", "numpy", "data visualization", "r"],
        "icon": "📊"
    },
    "Web Developer": {
        "skills": ["html", "css", "javascript", "react", "node", "bootstrap", "jquery", "typescript", "api", "git"],
        "icon": "🌐"
    },
    "AI Engineer": {
        "skills": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "nlp", "pandas", "numpy", "scikit", "neural network"],
        "icon": "🤖"
    },
    "Cloud Engineer": {
        "skills": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "devops", "linux", "ci/cd", "cloud"],
        "icon": "☁️"
    },
    "Full Stack Developer": {
        "skills": ["html", "css", "javascript", "react", "node", "mongodb", "sql", "api", "git", "python"],
        "icon": "🔧"
    },
    "ML Engineer": {
        "skills": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "scikit", "mlops", "docker", "api", "model deployment"],
        "icon": "🧠"
    },
    "DevOps Engineer": {
        "skills": ["docker", "kubernetes", "jenkins", "ci/cd", "linux", "aws", "git", "terraform", "ansible", "monitoring"],
        "icon": "⚙️"
    },
    "Cybersecurity Analyst": {
        "skills": ["network security", "penetration testing", "firewall", "linux", "python", "vulnerability", "encryption", "siem", "ethical hacking", "nmap"],
        "icon": "🔐"
    },
    "Mobile Developer": {
        "skills": ["android", "ios", "flutter", "react native", "kotlin", "swift", "java", "firebase", "api", "mobile"],
        "icon": "📱"
    },
    "UI/UX Designer": {
        "skills": ["figma", "adobe xd", "sketch", "user research", "wireframe", "prototyping", "css", "html", "design", "usability"],
        "icon": "🎨"
    },
    "Backend Developer": {
        "skills": ["python", "java", "node", "sql", "mongodb", "api", "rest", "microservices", "docker", "git"],
        "icon": "🖥️"
    },
    "Software Engineer": {
        "skills": ["java", "python", "c++", "data structures", "algorithms", "git", "sql", "oop", "system design", "api"],
        "icon": "💻"
    }
}

# ─────────────────────────────────────────────
# SECTION DETECTION
# ─────────────────────────────────────────────
SECTION_KEYWORDS = {
    "projects": ["project", "built", "developed", "created"],
    "experience": ["experience", "worked", "internship", "job", "employed"],
    "education": ["education", "university", "college", "degree", "bachelor", "master"],
    "github": ["github", "gitlab", "bitbucket", "repository"],
    "certifications": ["certification", "certified", "certificate", "course"],
    "linkedin": ["linkedin"],
    "achievements": ["achievement", "award", "winner", "hackathon"],
}

def detect_sections(text):
    found = {}
    for section, keywords in SECTION_KEYWORDS.items():
        found[section] = any(kw in text for kw in keywords)
    return found

# ─────────────────────────────────────────────
# ATS SCORE CALCULATION
# ─────────────────────────────────────────────
def calculate_ats_score(text, best_role_skills, sections):
    skill_score = sum(1 for s in best_role_skills if s in text) / len(best_role_skills) * 60
    section_score = sum(1 for s in sections.values() if s) / len(sections) * 30
    length_score = min(len(text.split()) / 300 * 10, 10)
    return int(skill_score + section_score + length_score)

# ─────────────────────────────────────────────
# NLP JOB MATCHING (Cosine Similarity)
# ─────────────────────────────────────────────
def nlp_job_match(resume_text, job_desc):
    if not job_desc.strip():
        return None
    try:
        vectorizer = CountVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return round(similarity * 100, 1)
    except:
        return None

# ─────────────────────────────────────────────
# SMART SUGGESTIONS
# ─────────────────────────────────────────────
def generate_suggestions(text, sections, missing_skills, best_role):
    suggestions = []

    if not sections["projects"]:
        suggestions.append("📁 Add a **Projects** section — recruiters love seeing real work!")
    if not sections["github"]:
        suggestions.append("🔗 Include your **GitHub** profile link to showcase your code.")
    if not sections["certifications"]:
        suggestions.append("🏅 Add relevant **certifications** to boost credibility.")
    if not sections["linkedin"]:
        suggestions.append("💼 Add your **LinkedIn** profile for professional presence.")
    if not sections["achievements"]:
        suggestions.append("🏆 Mention **hackathons, awards, or competitions** you've participated in.")

    action_verbs = ["developed", "built", "designed", "implemented", "created", "managed", "led", "optimized"]
    if not any(v in text for v in action_verbs):
        suggestions.append("✍️ Use strong **action verbs** like: Developed, Built, Optimized, Led.")

    top_missing = missing_skills[:5]
    for skill in top_missing:
        suggestions.append(f"🛠️ Learn **{skill.title()}** — it's key for a {best_role} role.")

    if len(text.split()) < 200:
        suggestions.append("📝 Your resume seems short. Aim for at least **300–500 words**.")

    return suggestions

# ─────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────
def generate_pdf_report(username, best_role, ats_score, role_scores, suggestions, job_match):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Background
    c.setFillColor(colors.HexColor("#0d1117"))
    c.rect(0, 0, width, height, fill=True, stroke=False)

    # Header
    c.setFillColor(colors.HexColor("#00C9A7"))
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(40, height - 50, "Smart Resume AI — Analysis Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, height - 68, f"User: {username}   |   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ATS Score
    c.setFillColor(colors.HexColor("#00C9A7"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 110, f"ATS Score: {ats_score}%")

    c.setFillColor(colors.HexColor("#845EC2"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 135, f"Best Role: {best_role}")

    if job_match is not None:
        c.setFillColor(colors.HexColor("#FFD700"))
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 160, f"Job Match Score: {job_match}%")

    # Role Scores
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, height - 195, "Role-wise Scores:")
    c.setFont("Helvetica", 11)
    y = height - 215
    for role, score in role_scores.items():
        c.setFillColor(colors.HexColor("#00C9A7") if score == max(role_scores.values()) else colors.HexColor("#aaaaaa"))
        c.drawString(50, y, f"• {role}: {score}%")
        y -= 18
        if y < 300:
            break

    # Suggestions
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y - 15, "Smart Suggestions:")
    c.setFont("Helvetica", 10)
    y -= 35
    for sug in suggestions[:6]:
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', sug)
        clean = re.sub(r'[^\x00-\x7F]+', '', clean)
        c.setFillColor(colors.HexColor("#cccccc"))
        c.drawString(50, y, clean[:90])
        y -= 16

    # Footer
    c.setFillColor(colors.HexColor("#333333"))
    c.rect(0, 0, width, 30, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#666666"))
    c.setFont("Helvetica", 9)
    c.drawString(40, 10, "Generated by Smart Resume AI  |  Powered by NLP & AI")

    c.save()
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
# BAR CHART
# ─────────────────────────────────────────────
def plot_role_chart(roles, scores):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#12121f')

    colors_bar = ['#00C9A7' if s == max(scores) else '#845EC2' for s in scores]
    bars = ax.barh(roles, scores, color=colors_bar, edgecolor='none', height=0.6)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{score}%', va='center', ha='left', color='white', fontsize=9, fontweight='bold')

    ax.set_xlim(0, 115)
    ax.set_xlabel('Score (%)', color='#888888')
    ax.tick_params(colors='#cccccc', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333')
    ax.spines['bottom'].set_color('#333')
    ax.xaxis.label.set_color('#888888')

    best_patch = mpatches.Patch(color='#00C9A7', label='Best Match')
    other_patch = mpatches.Patch(color='#845EC2', label='Other Roles')
    ax.legend(handles=[best_patch, other_patch], facecolor='#1a1a2e', labelcolor='white', fontsize=8)

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ─────────────────────────────────────────────
# SIDEBAR — LOGIN / REGISTER
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 Account")

    if st.session_state.logged_in:
        st.success(f"✅ Logged in as\n**{st.session_state.username}**")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    else:
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            email_in = st.text_input("Email", key="login_email")
            pass_in = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", key="btn_login"):
                if email_in and pass_in:
                    if login_user(email_in, pass_in):
                        st.session_state.logged_in = True
                        st.session_state.username = email_in.strip().lower()
                        st.success("Login Successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")
                else:
                    st.warning("Please fill in all fields.")

        with tab_register:
            reg_email = st.text_input("Email", key="reg_email")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            reg_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
            if st.button("Register", key="btn_register"):
                if reg_email and reg_pass and reg_pass2:
                    if reg_pass != reg_pass2:
                        st.error("Passwords don't match!")
                    elif len(reg_pass) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        ok, msg = register_user(reg_email, reg_pass)
                        if ok:
                            st.success(msg + " Please login.")
                        else:
                            st.error(msg)
                else:
                    st.warning("Please fill all fields.")

    st.markdown("---")
    st.markdown("### 📁 Report History")
    if st.session_state.logged_in:
        reports = load_json(REPORTS_FILE)
        user_reports = reports.get(st.session_state.username, [])
        if user_reports:
            for i, r in enumerate(reversed(user_reports[-5:]), 1):
                ats = r.get("ats_score", r.get("score", "N/A"))
                st.markdown(f"**{i}.** {r['role']} — ATS: **{ats}%**")
                if "date" in r:
                    st.caption(r["date"])
        else:
            st.info("No reports yet.")
    else:
        st.info("Login to see history.")

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🚀 Smart Resume AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze • Match • Improve • Get Hired</div>', unsafe_allow_html=True)

if not st.session_state.logged_in:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Role Matching\nMatches your resume against **12+ job roles** with detailed skill analysis.")
    with col2:
        st.markdown("### 🤖 NLP Powered\nUses **cosine similarity** to compare your resume with job descriptions.")
    with col3:
        st.markdown("### 📊 ATS Score\nCalculates your **ATS score** and suggests improvements.")
    st.warning("👈 Please **Login** or **Register** from the sidebar to get started!")
    st.stop()

# ─── Upload Section ───
st.markdown('<div class="section-header">📄 Upload Your Resume</div>', unsafe_allow_html=True)

col_upload, col_jd = st.columns([1, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], label_visibility="collapsed")
with col_jd:
    job_desc = st.text_area("📌 Paste Job Description (Optional)", height=120, placeholder="Paste the job description here for NLP matching...")

if not uploaded_file:
    st.info("👆 Upload your resume PDF to begin analysis.")
    st.stop()

# ─── ANALYSIS ───
with st.spinner("🔍 Analyzing your resume..."):
    resume_text = extract_text(uploaded_file)

if not resume_text.strip():
    st.error("Could not extract text from this PDF. Please try a text-based PDF.")
    st.stop()

st.success("✅ Resume uploaded and analyzed successfully!")

# Role Analysis
roles_list = []
scores_list = []
all_missing = []

role_results = {}
for role, data in ROLE_SKILLS.items():
    skills = data["skills"]
    icon = data["icon"]
    found = [s for s in skills if s in resume_text]
    missing = [s for s in skills if s not in resume_text]
    score = int((len(found) / len(skills)) * 100)
    role_results[role] = {"found": found, "missing": missing, "score": score, "icon": icon}
    roles_list.append(role)
    scores_list.append(score)
    all_missing.extend(missing)

best_score = max(scores_list)
best_role = roles_list[scores_list.index(best_score)]
best_role_skills = ROLE_SKILLS[best_role]["skills"]

# Sections & ATS
sections = detect_sections(resume_text)
ats_score = calculate_ats_score(resume_text, best_role_skills, sections)

# NLP Match
job_match = nlp_job_match(resume_text, job_desc) if job_desc.strip() else None

# Suggestions
missing_for_best = role_results[best_role]["missing"]
suggestions = generate_suggestions(resume_text, sections, missing_for_best, best_role)

# ─── TOP METRICS ───
st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""<div class="metric-card">
        <div style="color:#888;font-size:0.8rem">🏆 BEST ROLE</div>
        <div style="color:#00C9A7;font-size:1.1rem;font-weight:700">{best_role}</div>
    </div>""", unsafe_allow_html=True)

with m2:
    color = "#00C9A7" if ats_score >= 60 else "#FFD700" if ats_score >= 40 else "#FF6B6B"
    st.markdown(f"""<div class="metric-card">
        <div style="color:#888;font-size:0.8rem">📈 ATS SCORE</div>
        <div style="color:{color};font-size:1.5rem;font-weight:700">{ats_score}%</div>
    </div>""", unsafe_allow_html=True)

with m3:
    if job_match is not None:
        jcolor = "#00C9A7" if job_match >= 60 else "#FFD700" if job_match >= 40 else "#FF6B6B"
        st.markdown(f"""<div class="metric-card">
            <div style="color:#888;font-size:0.8rem">🔗 JOB MATCH</div>
            <div style="color:{jcolor};font-size:1.5rem;font-weight:700">{job_match}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="metric-card">
            <div style="color:#888;font-size:0.8rem">🔗 JOB MATCH</div>
            <div style="color:#555;font-size:0.9rem">Paste JD above</div>
        </div>""", unsafe_allow_html=True)

with m4:
    rank, total = get_ranking(st.session_state.username, ats_score)
    st.markdown(f"""<div class="metric-card">
        <div style="color:#888;font-size:0.8rem">🏅 YOUR RANK</div>
        <div style="color:#FFD700;font-size:1.5rem;font-weight:700">#{rank} / {total}</div>
    </div>""", unsafe_allow_html=True)

# ─── CHART ───
st.markdown('<div class="section-header">📊 Role Score Comparison</div>', unsafe_allow_html=True)
role_scores_dict = {r: s for r, s in zip(roles_list, scores_list)}
fig = plot_role_chart(roles_list, scores_list)
st.pyplot(fig)

# ─── ROLE DETAILS ───
st.markdown('<div class="section-header">🎯 Detailed Role Analysis</div>', unsafe_allow_html=True)

cols = st.columns(2)
for i, (role, data) in enumerate(role_results.items()):
    with cols[i % 2]:
        score = data["score"]
        found = data["found"]
        missing = data["missing"]
        icon = data["icon"]

        border_color = "#00C9A7" if role == best_role else "#845EC255"
        st.markdown(f"""<div class="role-card" style="border-left-color: {border_color}">
            <strong>{icon} {role}</strong> {'⭐ BEST MATCH' if role == best_role else ''}
            <br><span style="color:#00C9A7">Score: {score}%</span>
        </div>""", unsafe_allow_html=True)
        st.progress(score / 100)
        if found:
            st.caption(f"✅ Found: {', '.join(found)}")
        if missing:
            st.caption(f"❌ Missing: {', '.join(missing)}")

# ─── RESUME SECTIONS ───
st.markdown('<div class="section-header">📋 Resume Sections Detected</div>', unsafe_allow_html=True)
sec_cols = st.columns(4)
for i, (section, present) in enumerate(sections.items()):
    with sec_cols[i % 4]:
        icon = "✅" if present else "❌"
        color = "#00C9A7" if present else "#FF6B6B"
        st.markdown(f'<div style="text-align:center;color:{color};font-size:0.9rem">{icon} {section.title()}</div>', unsafe_allow_html=True)

# ─── SMART SUGGESTIONS ───
st.markdown('<div class="section-header">💡 Smart Suggestions</div>', unsafe_allow_html=True)
for sug in suggestions:
    st.markdown(f'<div class="suggestion-item">{sug}</div>', unsafe_allow_html=True)

# ─── SAVE REPORT + DOWNLOAD ───
st.markdown('<div class="section-header">💾 Save & Download</div>', unsafe_allow_html=True)
col_save, col_download = st.columns(2)

with col_save:
    if st.button("💾 Save Report"):
        save_report(st.session_state.username, best_role, best_score, ats_score)
        st.success("✅ Report saved to your history!")

with col_download:
    pdf_buffer = generate_pdf_report(
        st.session_state.username,
        best_role,
        ats_score,
        role_scores_dict,
        suggestions,
        job_match
    )
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name=f"resume_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )

# ─── EXTRACTED TEXT ───
with st.expander("📄 View Extracted Resume Text"):
    st.text_area("", resume_text, height=200, label_visibility="collapsed")