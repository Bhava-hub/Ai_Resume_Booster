import streamlit as st
import fitz  
from transformers import pipeline
import google.generativeai as genai

# Streamlit Page Config
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Sidebar Styling
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .nav-button {
            width: 100%;
            text-align: left;
            padding: 10px;
            border: none;
            background: transparent;
            color: white;
            font-size: 16px;
        }
        .selected {
            background-color: #1abc9c;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load NER Model (Cached)
@st.cache_resource
def load_ner_model():
    return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

ner_pipeline = load_ner_model()

# Google Gemini API Key
genai.configure(api_key="AIzaSyBV-gwmXSmwppGhIFaBdBlf3-aqpOrd7DY")

# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()

# Extract Skills from Resume
def extract_skills(resume_text):
    entities = ner_pipeline(resume_text)
    skills = [entity["word"] for entity in entities if entity["entity_group"] == "MISC"]
    return list(set(skills))

# Get Relevant Job Roles
def get_job_roles(skills):
    model = genai.GenerativeModel("gemini-pro")
    skills_text = ", ".join(skills)
    prompt = f"Suggest 5 job roles for someone with these skills: {skills_text}"
    response = model.generate_content(prompt)
    if response and response.text:
        job_roles = [role.strip("-• ") for role in response.text.split("\n") if role.strip()]
        return job_roles[:5]
    return []

# Find Missing Skills for Selected Job Role
def find_missing_skills(job_role, extracted_skills):
    model = genai.GenerativeModel("gemini-pro")
    skills_text = ", ".join(extracted_skills)
    prompt = f"What skills are missing for {job_role}, given these skills: {skills_text}?"
    response = model.generate_content(prompt)
    if response and response.text:
        missing_skills = [skill.strip("-• ") for skill in response.text.split("\n") if skill.strip()]
        return missing_skills
    return ["No missing skills found."]

# Generate Interview Questions
def get_interview_questions(job_role, round_type):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Generate 5 {round_type} interview questions for {job_role}."
    response = model.generate_content(prompt)
    if response and response.text:
        questions = [q.strip("-• ") for q in response.text.split("\n") if q.strip()]
        return questions[:5]
    return ["No questions generated."]

# Evaluate Answers
def evaluate_answers(questions, answers):
    model = genai.GenerativeModel("gemini-pro")
    formatted_qa = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
    prompt = f"Evaluate the following interview answers and provide feedback:\n{formatted_qa}"
    response = model.generate_content(prompt)
    if response and response.text:
        return response.text
    return "No feedback available."

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "home"
if "extracted_skills" not in st.session_state:
    st.session_state.extracted_skills = None
if "job_roles" not in st.session_state:
    st.session_state.job_roles = None
if "selected_job_role" not in st.session_state:
    st.session_state.selected_job_role = None
if "missing_skills" not in st.session_state:
    st.session_state.missing_skills = None
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = ["" for _ in range(5)]
if "feedback" not in st.session_state:
    st.session_state.feedback = None

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Interview", "Feedback"]
for page in pages:
    if st.sidebar.button(page, key=page, use_container_width=True, help=f"Go to {page}"):
        st.session_state.page = page.lower()
        st.rerun()

# Home Page
if st.session_state.page == "home":
    st.header("Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
    if uploaded_file:
        extracted_text = extract_text_from_pdf(uploaded_file)
        with st.spinner("Extracting skills..."):
            st.session_state.extracted_skills = extract_skills(extracted_text)
            st.session_state.job_roles = get_job_roles(st.session_state.extracted_skills)
        st.success("Skills extracted successfully! Moving to Interview...")
        st.session_state.page = "interview"
        st.rerun()

# Interview Page
elif st.session_state.page == "interview":
    st.header("Interview Preparation")
    if st.session_state.job_roles:
        new_selected_job_role = st.selectbox("Choose a job role:", ["Select a job role"] + st.session_state.job_roles)
        if new_selected_job_role != "Select a job role":
            st.session_state.selected_job_role = new_selected_job_role

            # Missing Skills Section
            if st.button("Find Missing Skills"):
                st.session_state.missing_skills = find_missing_skills(new_selected_job_role, st.session_state.extracted_skills)
            if st.session_state.missing_skills:
                st.subheader("Missing Skills:")
                st.write(", ".join(st.session_state.missing_skills))

            # Generate Interview Questions
            if st.button("Generate Questions"):
                st.session_state.interview_questions = get_interview_questions(new_selected_job_role, "Technical")

            for i, question in enumerate(st.session_state.interview_questions):
                st.session_state.user_answers[i] = st.text_area(f"Q{i+1}: {question}", st.session_state.user_answers[i])

            if st.button("Submit Answers"):
                st.session_state.page = "feedback"
                st.rerun()

# Feedback Page
elif st.session_state.page == "feedback":
    st.header("Feedback on Your Answers")
    if st.session_state.user_answers:
        st.session_state.feedback = evaluate_answers(st.session_state.interview_questions, st.session_state.user_answers)
    st.write(st.session_state.feedback if st.session_state.feedback else "No feedback yet.")
