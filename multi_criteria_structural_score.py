import pandas as pd
import re
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------
# Candidate dataset (example)
# ---------------------------
data = pd.read_csv("candidates.csv")

candidates_df = pd.DataFrame(data)

# JD Extraction

def extract_job_json(job_description):
    prompt = f"""
    You are an AI assistant that extracts structured job details from unstructured job descriptions.
    Return the result as **valid JSON only**, no extra text.

    Schema (always include these fields, even if missing):
    {{
      "Role": string,
      "Required Skills": list of strings,
      "Location Preference": string or null,
      "Content Categories": list of strings,
      "Budget": string or null,
      "Special Preferences": string or null,
      "Creator Name": string or null,
      "Creator Channel": string or null
    }}

    Rules:
    - If any field is not explicitly mentioned, set its value to null (do not guess).
    - Extract skills as they appear in the text.
    - Content Categories can include multiple (Entertainment, Food & Cooking, Productivity, etc).
    - Keep budget in the same unit (e.g., "$2500/month" or "100-150 per hour").
    - Parse creator/channel name from YouTube URL if possible.

    Job Description:
    {job_description}
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)


# ---------------------------
# JD parsing utilities
# ---------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9, ]+', '', text).lower().strip()

def tokenize(text):
    return [t.strip() for t in clean_text(text).split(",") if t.strip()]

def parse_jd(jd_text):
    """
    Very simple JD parser. Later, can be replaced with LLM-based extraction.
    """
    jd_dict = {
        "skills": tokenize(jd_text.get("skills", "")),
        "domain": jd_text.get("domain", "").lower(),
        "languages": tokenize(jd_text.get("languages", "")),
        "soft_skills": tokenize(jd_text.get("soft_skills", ""))
    }
    return jd_dict

# ---------------------------
# Scoring functions
# ---------------------------
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_structural_score(candidate, jd):
    # Tokenize candidate fields
    cand_skills = set(tokenize(candidate["skills"]))
    cand_soft = set(tokenize(candidate.get("soft_skills", "")))
    cand_lang = set(tokenize(candidate.get("languages", "")))
    cand_domain = candidate.get("domain", "").lower()

    # JD fields
    jd_skills = set(jd["skills"])
    jd_soft = set(jd["soft_skills"])
    jd_lang = set(jd["languages"])
    jd_domain = jd["domain"]

    # Sub-scores
    skills_score = jaccard_similarity(cand_skills, jd_skills)
    soft_score = jaccard_similarity(cand_soft, jd_soft)
    lang_score = jaccard_similarity(cand_lang, jd_lang)
    domain_score = 1.0 if cand_domain == jd_domain else 0.0

    # Weighted score
    final_score = (
        0.3 * skills_score +
        0.2 * soft_score +
        0.1 * lang_score +
        0.2 * domain_score
    )

    return {
        "skills_score": skills_score,
        "soft_score": soft_score,
        "lang_score": lang_score,
        "domain_score": domain_score,
        "final_score": final_score
    }

# ---------------------------
# Example Usage
# ---------------------------
jd_input = {
    "skills": "Project management, Budgeting, Reconciliation",
    "domain": "Finance",
    "languages": "English",
    "soft_skills": "Leadership, Communication"
}

jd_parsed = parse_jd(jd_input)

results = []
for _, row in candidates_df.iterrows():
    score_dict = compute_structural_score(row, jd_parsed)
    results.append({
        "candidate_id": row["candidate_id"],
        "name": row["name"],
        **score_dict
    })

results_df = pd.DataFrame(results).sort_values(by="final_score", ascending=False)

print(results_df)
