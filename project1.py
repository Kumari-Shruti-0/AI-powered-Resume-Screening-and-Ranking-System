import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Sample data for trending skills and salary insights
trending_skills = {
    "Software Engineer": ["Python", "Java", "Cloud Computing", "Docker"],
    "Data Analyst": ["SQL", "Tableau", "Python", "Data Visualization"],
    "AI/ML Engineer": ["TensorFlow", "PyTorch", "Deep Learning", "NLP"],
    "Web Developer": ["JavaScript", "React", "Node.js", "CSS"]
}

salary_data = {
    "Software Engineer": "₹6-15 LPA",
    "Data Analyst": "₹5-12 LPA",
    "AI/ML Engineer": "₹8-20 LPA",
    "Web Developer": "₹4-10 LPA"
}

def extract_text(file):
    pdf = PdfReader(file)
    text=""
    for page in pdf.pages:
        text+=page.extract_text()
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

def recommend_skills(job_description):
    for role, skills in trending_skills.items():
        if any(skill.lower() in job_description.lower() for skill in skills):
            return role, skills
    return "General", []

def analyze_resume_for_role(resume_text):
    for role, skills in trending_skills.items():
        if any(skill.lower() in resume_text.lower() for skill in skills):
            return role
    return "Unknown"

st.title("AI Resume Screening & Career Guidance System")
st.header("Job Description")
job_description = st.text_area("Enter the job description")

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Resume Analysis & Ranking")
    resumes = []
    for file in uploaded_files:
        text = extract_text(file)
        resumes.append(text)
    
    scores = rank_resumes(job_description, resumes)
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    st.write(results)
    
    st.header("Skill Gap Analysis & Recommendations")
    best_fit_role, recommended_skills = recommend_skills(job_description)
    st.subheader(f"Best Fit Role: {best_fit_role}")
    st.write("### Recommended Skills:")
    st.write(recommended_skills)
    
    if best_fit_role in salary_data:
        st.write(f"### Estimated Salary: {salary_data[best_fit_role]}")
    
    st.header("Best Fit Role for Each Resume")
    for i, file in enumerate(uploaded_files):
        role = analyze_resume_for_role(resumes[i])
        st.write(f"- {file.name}: Best fit for {role} role based on ranking score")
    
    st.header("Diversity & Inclusion Analysis")
    unique_roles = set([analyze_resume_for_role(text) for text in resumes])
    st.write(f"The candidate pool includes professionals from {len(unique_roles)} different backgrounds: {', '.join(unique_roles)}")
