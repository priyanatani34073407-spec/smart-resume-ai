# 🚀 Smart Resume AI Analyzer

An AI-powered web application that analyzes resumes and provides job matching insights, ATS score, and improvement suggestions.

---

## 🔐 Features
- ✅ User Authentication (Register/Login with hashed passwords)
- ✅ Resume Upload (PDF)
- ✅ 12+ Job Role Analysis
- ✅ NLP Job Description Matching (Cosine Similarity)
- ✅ ATS Score Calculation
- ✅ Smart AI Suggestions
- ✅ Report Saving & History
- ✅ Global Ranking System
- ✅ PDF Report Download
- ✅ Beautiful Dark UI

---

## 🛠 Tech Stack
- Python 3.11
- Streamlit
- PyPDF2
- scikit-learn (NLP / Cosine Similarity)
- Matplotlib
- ReportLab
- JSON (local storage — no Firebase needed)

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/smart-resume-ai.git
cd smart-resume-ai
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud (FREE)

1. Push your code to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New App**
4. Select your GitHub repo
5. Set **Main file path** → `app.py`
6. Click **Deploy** ✅

> ⚠️ Make sure `users.json` and `reports.json` exist in your repo (even as empty `{}`)

---

## 📁 File Structure

```
smart-resume-ai/
│
├── app.py               # Main application
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version
├── users.json           # User credentials (hashed)
├── reports.json         # Saved reports
└── .streamlit/
    └── config.toml      # Theme settings
```

---

## 📊 Key Highlights
- 🎯 12+ job roles analyzed
- 📈 ATS Score based on skills, sections & content
- 🔗 NLP cosine similarity job matching
- 💡 Smart suggestions for resume improvement
- 🏆 Global ranking among all users
- 📥 Downloadable PDF report