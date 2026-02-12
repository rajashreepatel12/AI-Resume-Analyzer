import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🦩",
    layout="centered"
)

# ---------- FLAMINGO BEACH HD BACKGROUND ----------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1526481280691-781c9c9b1c1f");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Glass overlay for readability */
    .main {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 25px;
        border-radius: 15px;
    }

    /* Title styling */
    h1 {
        color: #ff9ecb;
        text-align: center;
        font-size: 48px;
        text-shadow: 2px 2px 6px black;
    }

    /* Subheaders */
    h3 {
        color: #00ffff;
    }

    /* File uploader styling */
    .stFileUploader {
        background-color: rgba(255,255,255,0.9);
        padding: 10px;
        border-radius: 10px;
    }

    /* Text area styling */
    .stTextArea textarea {
        background-color: rgba(255,255,255,0.9);
        color: black;
        border-radius: 10px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #ff69b4;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- TITLE ----------
st.title("🦩 AI Resume Analyzer")
st.write("Analyze your resume with job description using AI 🌴")

# ---------- LOAD SKILLS ----------
def load_skills():
    with open("skills.txt", "r") as file:
        skills = file.read().splitlines()
    return [s.lower() for s in skills]

# ---------- EXTRACT PDF TEXT ----------
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

# ---------- INPUTS ----------
uploaded_file = st.file_uploader(
    "Upload Resume (PDF)", type=["pdf"]
)

job_desc = st.text_area(
    "Paste Job Description"
)

# ---------- ANALYSIS ----------
if uploaded_file and job_desc:

    resume_text = extract_text(uploaded_file)

    # Similarity
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(
        [resume_text, job_desc.lower()]
    )

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    st.subheader("📊 Match Score")
    st.progress(similarity)
    st.success(f"{similarity*100:.2f}% Match")

    # Skills
    skills = load_skills()

    found_skills = [
        skill for skill in skills
        if skill in resume_text
    ]

    missing_skills = [
        skill for skill in skills
        if skill not in found_skills
    ]

    st.subheader("✅ Skills Found")
    st.write(found_skills)

    st.subheader("❌ Missing Skills")
    st.write(missing_skills)

else:
    st.info("Upload resume and paste job description to analyze.")

