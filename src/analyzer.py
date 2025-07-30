import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess

def get_spacy_model():
    """
    Loads the spaCy English model, downloading it if necessary.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

def extract_skills_from_text_spacy(resume_text, skill_list):
    """
    Extracts skills from resume text using spaCy noun chunks and matches them to a predefined skill list.
    """
    nlp = get_spacy_model()
    doc = nlp(resume_text)
    noun_chunks = set(chunk.text.lower().strip() for chunk in doc.noun_chunks)
    processed_skill_list = {skill.lower().strip(): skill for skill in skill_list}
    matched_skills = [
        processed_skill_list[skill] for skill in processed_skill_list if skill in noun_chunks
    ]
    return matched_skills

def load_skill_list(skill_file_path):
    """
    Loads a list of skills from a CSV file with a column named 'Skill'.
    """
    df = pd.read_csv(skill_file_path)
    return df['Skill'].dropna().tolist()

def load_job_roles(job_roles_file):
    """
    Loads job roles and required skills from a CSV file.
    Assumes columns: 'Job Title', 'Required Skills', 'Job Description'
    """
    df = pd.read_csv(job_roles_file)
    df['Required Skills'] = df['Required Skills'].apply(
        lambda x: [skill.strip() for skill in x.split(',')] if isinstance(x, str) else []
    )
    return df

def match_job_roles(resume_skills, job_roles_df):
    """
    Matches resume skills to job roles using cosine similarity by building vectors manually.
    This correctly handles multi-word skills.
    """
    if not resume_skills or job_roles_df.empty:
        return pd.DataFrame(columns=['Job Title', 'Job Description', 'Required Skills', 'Similarity'])

    all_skills_set = set(s.lower() for s in resume_skills)
    for skills_list in job_roles_df['Required Skills']:
        all_skills_set.update(s.lower() for s in skills_list)

    if not all_skills_set:
        return pd.DataFrame(columns=['Job Title', 'Job Description', 'Required Skills', 'Similarity'])

    all_skills_list = sorted(list(all_skills_set))
    skill_to_idx = {skill: i for i, skill in enumerate(all_skills_list)}
    vocab_size = len(all_skills_list)

    resume_vector = np.zeros(vocab_size)
    for skill in resume_skills:
        idx = skill_to_idx.get(skill.lower())
        if idx is not None:
            resume_vector[idx] = 1

    job_vectors = np.zeros((len(job_roles_df), vocab_size))
    for i, skills_list in enumerate(job_roles_df['Required Skills']):
        for skill in skills_list:
            idx = skill_to_idx.get(skill.lower())
            if idx is not None:
                job_vectors[i, idx] = 1

    resume_vector_2d = resume_vector.reshape(1, -1)
    similarities = cosine_similarity(resume_vector_2d, job_vectors)
    job_roles_df['Similarity'] = similarities[0]

    top_matches = job_roles_df.sort_values(by='Similarity', ascending=False).head(3)
    return top_matches[['Job Title', 'Job Description', 'Required Skills', 'Similarity']]
