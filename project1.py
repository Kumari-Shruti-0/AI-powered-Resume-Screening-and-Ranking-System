import streamlit as st                                       # Import Streamlit for creating a web app
from PyPDF2 import PdfReader                                 # Import PdfReader to extract text from PDF resumes
import pandas as pd                                          # Import pandas for handling tabular data
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer for text processing
from sklearn.metrics.pairwise import cosine_similarity       # Import cosine similarity for ranking resumes

# Sample data for trending skills mapped to various job roles
trending_skills = {
    "Software Engineer": ["Python", "Java", "Cloud Computing", "Docker"],
    "Data Analyst": ["SQL", "Tableau", "Python", "Data Visualization"],
    "AI/ML Engineer": ["TensorFlow", "PyTorch", "Deep Learning", "NLP"],
    "Web Developer": ["JavaScript", "React", "Node.js", "CSS"]
}

# Sample data for estimated salary ranges of job roles
salary_data = {
    "Software Engineer": "₹6-15 LPA",
    "Data Analyst": "₹5-12 LPA",
    "AI/ML Engineer": "₹8-20 LPA",
    "Web Developer": "₹4-10 LPA"
}

# Function to extract text from a PDF file
def extract_text(file):
    pdf = PdfReader(file)            # Read the uploaded PDF file
    text = ""                        # Initialize an empty string to store extracted text
    for page in pdf.pages:           # Loop through each page in the PDF
        text += page.extract_text()  # Extract text from the page and append it to the string
    return text                      # Return the extracted text

# Function to rank resumes based on job description using cosine similarity
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes                  # Combine job description and resumes into a list
    vectorizer = TfidfVectorizer().fit_transform(documents)  # Convert text data into numerical vectors using TF-IDF
    vectors = vectorizer.toarray()                           # Convert the sparse matrix into an array
    
    job_description_vector = vectors[0]                      # Extract the job description vector
    resume_vectors = vectors[1:]                             # Extract resume vectors (excluding job description)
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()  # Compute cosine similarity scores
    return cosine_similarities                               # Return similarity scores

# Function to recommend skills based on job description
def recommend_skills(job_description):
    for role, skills in trending_skills.items():                               # Loop through predefined job roles and their skills
        if any(skill.lower() in job_description.lower() for skill in skills):  # Check if any skill is mentioned in the job description
            return role, skills                                                # Return the matched role and relevant skills
    return "General", []                                                       # Default to "General" if no specific role is found

# Function to determine best fit role for a resume
def analyze_resume_for_role(resume_text):
    for role, skills in trending_skills.items():                           # Loop through job roles and skills
        if any(skill.lower() in resume_text.lower() for skill in skills):  # Check if any skill matches in resume text
            return role                                                    # Return the matched role
    return "Unknown"                                                       # Return "Unknown" if no specific role is identified

# Streamlit web app setup
st.title("AI Resume Screening & Career Guidance System")     # Title of the application
st.header("Job Description")                                 # Section header for job description input
job_description = st.text_area("Enter the job description")  # Input field for job description

st.header("Upload Resumes")                                  # Section header for resume upload
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)  # Upload resumes

if uploaded_files and job_description:      # Check if resumes are uploaded and job description is provided
    st.header("Resume Analysis & Ranking")  # Section header for ranking resumes
    resumes = []                            # Initialize an empty list to store resume texts
    for file in uploaded_files:
        text = extract_text(file)           # Extract text from each uploaded resume
        resumes.append(text)                # Add extracted text to the list
    
    scores = rank_resumes(job_description, resumes)  # Compute ranking scores based on job description
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})  # Store results in a DataFrame
    results = results.sort_values(by="Score", ascending=False)  # Sort resumes based on ranking scores
    
    st.write(results)  # Display the ranked resumes
    
    st.header("Skill Gap Analysis & Recommendations")                      # Section header for skill recommendations
    best_fit_role, recommended_skills = recommend_skills(job_description)  # Get recommended skills based on job description
    st.subheader(f"Best Fit Role: {best_fit_role}")                        # Display the best fit role
    st.write("### Recommended Skills:")                                    # Display recommended skills section
    st.write(recommended_skills)                                           # Display the list of skills
    
    if best_fit_role in salary_data:                                     # Check if salary data is available for the role
        st.write(f"### Estimated Salary: {salary_data[best_fit_role]}")  # Display salary range
    
    st.header("Best Fit Role for Each Resume")      # Section header for role analysis per resume
    for i, file in enumerate(uploaded_files):
        role = analyze_resume_for_role(resumes[i])  # Determine best fit role for each resume
        st.write(f"- {file.name}: Best fit for {role} role based on ranking score")  # Display result
    
    st.header("Diversity & Inclusion Analysis")     # Section header for diversity analysis
    unique_roles = set([analyze_resume_for_role(text) for text in resumes])  # Identify unique roles among candidates
    st.write(f"The candidate pool includes professionals from {len(unique_roles)} different backgrounds: {', '.join(unique_roles)}")  # Display diversity insights
