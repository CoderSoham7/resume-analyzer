import sys
import os
import streamlit as st
import pandas as pd

# Add the 'src' directory to the system path to find the custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import parser
import analyzer
import predictor
import visualizer

# Set page configuration for the Streamlit app
st.set_page_config(page_title="📄 Smart Resume Analyzer", layout="wide")

# Custom CSS for styling the application
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
        /* Basic styles for light and dark mode */
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

# Initialize session state for dark mode toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_mode():
    """Toggles the dark_mode session state."""
    st.session_state.dark_mode = not st.session_state.dark_mode

# Apply custom CSS
set_custom_css()

# --- App Layout ---

# Create a container to apply the light/dark mode class
mode_class = "dark-mode" if st.session_state.dark_mode else "light-mode"
st.markdown(f'<div class="{mode_class}">', unsafe_allow_html=True)

# Header section
st.markdown("## 📄 Smart Resume Analyzer & Career Predictor")
st.markdown("Upload your resume to discover your top job matches and future career path! 🚀")

# Dark mode toggle button
mode_label = "🌙 Switch to Dark Mode" if not st.session_state.dark_mode else "☀️ Switch to Light Mode"
st.button(mode_label, on_click=toggle_mode)

# File uploader for resumes
uploaded_file = st.file_uploader("📤 Drag and drop your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    # Save the uploaded file temporarily to be processed
    file_ext = os.path.splitext(uploaded_file.name)[1]
    temp_resume_path = f"temp_resume{file_ext}"
    with open(temp_resume_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        # --- 1. Parse Resume ---
        resume_text = parser.parse_resume(temp_resume_path)

        # --- 2. Extract Skills ---
        # Ensure the paths to data files are correct
        skill_list = analyzer.load_skill_list("notebooks/skills.csv")
        resume_skills = analyzer.extract_skills_from_text_spacy(resume_text, skill_list)

        st.markdown("### 🧠 Extracted Skills")
        if resume_skills:
            st.info(", ".join(resume_skills))
        else:
            st.warning("No skills were extracted from the resume. Please ensure the resume has clear skill descriptions.")

        # --- 3. Match Job Roles ---
        job_roles_df = analyzer.load_job_roles("notebooks/job_roles.csv")
        top_matches = analyzer.match_job_roles(resume_skills, job_roles_df)

        st.markdown("### 💼 Top Matching Job Roles")
        if not top_matches.empty:
            # Display the top matching jobs as a dataframe
            st.dataframe(top_matches[['Job Title', 'Job Description', 'Similarity']])

            # --- 4. Visualize Skill Gaps ---
            # Get the required skills from the best matching job
            top_job_skills = top_matches.iloc[0]['Required Skills']
            visualizer.plot_skill_gaps(resume_skills, top_job_skills, save_path="skill_gaps.png")
            st.image("skill_gaps.png", caption="🧩 Skill Gaps for the Top Matched Role")

            # --- 5. Predict Next Career Move ---
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

                # --- 6. Visualize Career Trajectory ---
                visualizer.plot_career_trajectory([current_role, predicted_role], save_path="career_trajectory.png")
                st.image("career_trajectory.png", caption="📈 Predicted Career Trajectory")

            except FileNotFoundError:
                st.error("Prediction model not found. Please ensure the model files are in the 'models/' directory.")
            except Exception as e:
                st.error(f"Could not predict the next role. Error: {e}")

        else:
            st.warning("No matching job roles found based on the extracted skills.")

    except FileNotFoundError as e:
        st.error(f"A required file was not found. Please check the path: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_resume_path):
            os.remove(temp_resume_path)

# Close the div for the light/dark mode container
st.markdown("</div>", unsafe_allow_html=True)
