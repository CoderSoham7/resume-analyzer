import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import parser
import analyzer
import predictor
import visualizer

import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="📄 Smart Resume Analyzer", layout="wide")

# Custom CSS for dark/light mode and styling
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
            background: linear-gradient(to bottom, #000000, #444444);
            color: white;
        }
        .light-mode {
            background: white;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

# App state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

set_custom_css()
mode_label = "🌙 Switch to Dark Mode" if not st.session_state.dark_mode else "☀️ Switch to Light Mode"
st.button(mode_label, on_click=toggle_mode)

mode_class = "dark-mode" if st.session_state.dark_mode else "light-mode"
st.markdown(f'<div class="{mode_class}">', unsafe_allow_html=True)

# Title
st.markdown("## 📄 Smart Resume Analyzer & Career Predictor")
st.markdown("Upload your resume to discover your top job matches and future career path! 🚀")

# Upload resume
uploaded_file = st.file_uploader("📤 Drag and drop your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    # Get the file extension from the uploaded file name
    file_ext = os.path.splitext(uploaded_file.name)[1]
    temp_resume_path = f"temp_resume{file_ext}"
    with open(temp_resume_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        resume_text = parser.parse_resume(temp_resume_path)
    except Exception as e:
        st.error(f"Failed to parse resume: {e}")
    else:
        skill_list = analyzer.load_skill_list("notebooks/skills.csv")
        resume_skills = analyzer.extract_skills_from_text_spacy(resume_text, skill_list)

        st.markdown("### 🧠 Extracted Skills")
        if resume_skills:
            st.write(", ".join(resume_skills))
        else:
            st.warning("No skills extracted from the resume.")

        job_roles_df = analyzer.load_job_roles("notebooks/job_roles.csv")
        try:
            top_matches = analyzer.match_job_roles(resume_skills, job_roles_df)
        except ValueError as e:
            st.error(f"Failed to match job roles: {e}")
            top_matches = pd.DataFrame()
        except Exception as e:
            st.error(f"Unexpected error during job role matching: {e}")
            top_matches = pd.DataFrame()
        else:
            st.markdown("### 💼 Top Matching Job Roles")
            st.dataframe(top_matches)

            if not top_matches.empty:
                required_skills = top_matches.iloc[0]['Required Skills']
                visualizer.plot_skill_gaps(resume_skills, required_skills)
                st.image("skill_gaps.png", caption="🧩 Skill Gaps")

                current_role = top_matches.iloc[0]['Job Title']
                try:
                    predicted_role = predictor.predict_next_role(
                        model_path="models/career_model.pkl",
                        skill_encoder_path="models/skill_encoder.pkl",
                        role_columns_path="models/role_columns.pkl",
                        current_role=current_role,
                        skills=resume_skills
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                else:
                    st.markdown(f"### 🔮 Predicted Next Role: **{predicted_role}**")

                    visualizer.plot_career_trajectory([current_role, predicted_role])
                    st.image("career_trajectory.png", caption="📈 Career Trajectory")

st.markdown("</div>", unsafe_allow_html=True)
