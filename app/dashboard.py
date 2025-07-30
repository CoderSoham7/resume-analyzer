import sys
import os
import streamlit as st
import pandas as pd
import subprocess
import spacy

# Add 'src' directory to system path
base_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
sys.path.append(base_dir)

import parser
import analyzer
import predictor
import visualizer

# Safe spaCy model loading
def get_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = get_spacy_model()

# Streamlit page config
st.set_page_config(page_title="📄 Smart Resume Analyzer", layout="wide")

def set_custom_css():
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border: none;
            padding: 0.5em 1em;
            border-radius: 5px;
        }
        .dark-mode {
            background-color: #1E1E1E;
            color: white;
        }
        .light-mode {
            background-color: #FFFFFF;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

set_custom_css()
mode_class = "dark-mode" if st.session_state.dark_mode else "light-mode"
st.markdown(f'<div class="{mode_class}">', unsafe_allow_html=True)

st.markdown("## 📄 Smart Resume Analyzer & Career Predictor")
st.markdown("Upload your resume to discover your top job matches and future career path! 🚀")
mode_label = "🌙 Switch to Dark Mode" if not st.session_state.dark_mode else "☀️ Switch to Light Mode"
st.button(mode_label, on_click=toggle_mode)

uploaded_file = st.file_uploader("📤 Drag and drop your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    temp_resume_path = f"temp_resume{file_ext}"
    with open(temp_resume_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        resume_text = parser.parse_resume(temp_resume_path)
        skill_list = analyzer.load_skill_list("notebooks/skills.csv")
        resume_skills = analyzer.extract_skills_from_text_spacy(resume_text, skill_list)

        st.markdown("### 🧠 Extracted Skills")
        if resume_skills:
            st.info(", ".join(resume_skills))
        else:
            st.warning("No skills were extracted from the resume.")

        job_roles_df = analyzer.load_job_roles("notebooks/job_roles.csv")
        top_matches = analyzer.match_job_roles(resume_skills, job_roles_df)

        st.markdown("### 💼 Top Matching Job Roles")
        if not top_matches.empty:
            st.dataframe(top_matches[['Job Title', 'Job Description', 'Similarity']])
            top_job_skills = top_matches.iloc[0]['Required Skills']
            visualizer.plot_skill_gaps(resume_skills, top_job_skills, save_path="skill_gaps.png")
            st.image("skill_gaps.png", caption="🧩 Skill Gaps for the Top Matched Role")

            current_role = top_matches.iloc[0]['Job Title']
            try:
                predicted_role = predictor.predict_next_role(
                    model_path="models/career_model.pkl",
                    skill_encoder_path="models/skill_encoder.pkl",
                    role_columns_path="models/role_columns.pkl",
                    current_role=current_role,
                    skills=resume_skills
                )
                st.markdown(f"### 🔮 Predicted Next Role: **{predicted_role}**")
                visualizer.plot_career_trajectory([current_role, predicted_role], save_path="career_trajectory.png")
                st.image("career_trajectory.png", caption="📈 Predicted Career Trajectory")
            except FileNotFoundError:
                st.error("Prediction model not found.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("No matching job roles found.")
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    finally:
        if os.path.exists(temp_resume_path):
            os.remove(temp_resume_path)

st.markdown("</div>", unsafe_allow_html=True)
