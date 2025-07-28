import matplotlib.pyplot as plt
import seaborn as sns

def plot_skill_gaps(resume_skills, required_skills, save_path="skill_gaps.png"):
    missing_skills = list(set(required_skills) - set(resume_skills))
    skill_counts = {skill: 1 for skill in missing_skills}

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(skill_counts.keys()), y=list(skill_counts.values()))
    plt.title("Missing Skills for Target Job Role")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_career_trajectory(roles, save_path="career_trajectory.png"):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list(range(len(roles))), y=roles, marker="o")
    plt.title("Predicted Career Trajectory")
    plt.xlabel("Career Stage")
    plt.ylabel("Job Role")
    plt.xticks(ticks=list(range(len(roles))), labels=roles, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
