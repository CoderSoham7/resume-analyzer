import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def extract_skills_from_text_spacy(resume_text, skill_list):
    """
    Extracts skills from resume text using spaCy noun chunks and matches them to a predefined skill list.
    """
    doc = nlp(resume_text)
    noun_chunks = set(chunk.text.lower().strip() for chunk in doc.noun_chunks)
    matched_skills = [skill for skill in skill_list if skill.lower() in noun_chunks]
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
    df['Required Skills'] = df['Required Skills'].apply(lambda x: [skill.strip() for skill in x.split(',')])
    return df

def match_job_roles(resume_skills, job_roles_df):
    """
    Matches resume skills to job roles using cosine similarity.
    Returns top 3 matching job roles.
    """
    # Defensive: ensure resume_skills is not empty and all_skills is not empty
    if not resume_skills or not any(isinstance(sk, str) and sk.strip() for sk in resume_skills):
        return job_roles_df.head(0)[['Job Title', 'Job Description', 'Required Skills']]
    all_skills = set(resume_skills)
    for skills in job_roles_df['Required Skills']:
        all_skills.update(skills)

    all_skills = list(all_skills)
    if not all_skills:
        # No skills to match, return empty DataFrame
        return job_roles_df.head(0)[['Job Title', 'Job Description', 'Required Skills']]
    vectorizer = CountVectorizer(vocabulary=all_skills, binary=True)

    # Use np.asarray to convert to dense numpy arrays, avoiding .toarray
    resume_vector = np.asarray(vectorizer.transform([" ".join(resume_skills)]))

    job_vectors = []
    for skills in job_roles_df['Required Skills']:
        job_vector = np.asarray(vectorizer.transform([" ".join(skills)]))
        # Ensure job_vector is always 2D before indexing
        if job_vector.ndim == 2 and job_vector.shape[0] > 0:
            job_vectors.append(job_vector[0])
        else:
            # If job_vector is empty or not 2D, append a zero vector of correct length
            job_vectors.append(np.zeros(len(all_skills)))

    job_vectors = np.array(job_vectors)
    # Defensive: ensure resume_vector and job_vectors are both 2D and have the same number of features
    if resume_vector.ndim == 1:
        resume_vector = resume_vector.reshape(1, -1)
    if job_vectors.ndim == 1:
        job_vectors = job_vectors.reshape(1, -1)
    # If either is empty, return empty DataFrame
    if resume_vector.size == 0 or job_vectors.size == 0:
        return job_roles_df.head(0)[['Job Title', 'Job Description', 'Required Skills']]
    if resume_vector.shape[1] == 0 or job_vectors.shape[1] == 0:
        return job_roles_df.head(0)[['Job Title', 'Job Description', 'Required Skills']]
    if resume_vector.shape[1] != job_vectors.shape[1]:
        # Pad job_vectors or resume_vector to match shapes
        max_features = max(resume_vector.shape[1], job_vectors.shape[1])
        pad_width_resume = max_features - resume_vector.shape[1]
        pad_width_jobs = max_features - job_vectors.shape[1]
        if pad_width_resume > 0:
            resume_vector = np.pad(resume_vector, ((0,0),(0,pad_width_resume)), 'constant')
        if pad_width_jobs > 0:
            job_vectors = np.pad(job_vectors, ((0,0),(0,pad_width_jobs)), 'constant')
    # Final robust check before similarity
    if resume_vector.shape[1] != job_vectors.shape[1] or resume_vector.size == 0 or job_vectors.size == 0:
        return job_roles_df.head(0)[['Job Title', 'Job Description', 'Required Skills']]
    similarities = cosine_similarity(resume_vector, job_vectors)[0]
    job_roles_df['Similarity'] = similarities

    top_matches = job_roles_df.sort_values(by='Similarity', ascending=False).head(3)
    return top_matches[['Job Title', 'Job Description', 'Similarity']]
