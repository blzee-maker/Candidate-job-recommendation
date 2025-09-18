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

    try:
        jd_data = json.loads(response.text)
    except Exception:
        # Fallback: strip markdown/code fences
        cleaned = response.text.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
        jd_data = json.loads(cleaned)

    return jd_data

# -----------------------------
# Step 2: Extract ONLY Must-Haves
# -----------------------------
def extract_must_have_details(jd_text: str) -> dict:
    """
    Extract only the must-have candidate requirements.
    """
    prompt = f"""
    From the following Job Description, identify ONLY the must-have requirements.
    Ignore nice-to-haves or optional qualities.
    Return JSON with fields:
    - required_skills: list of absolutely required skills
    - required_languages: list of must-have languages
    - required_domain: main domain/industry experience that is mandatory
    - required_experience: minimum years of experience if specified

    you can also add fields if you find other must-have criteria.
    If a field is not explicitly mentioned, set its value to null or empty list.

    Job Description:
    {jd_text}
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    try:
        must_have_data = json.loads(response.text)
    except Exception:
        # Fallback
        cleaned = response.text.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
        must_have_data = json.loads(cleaned)

    return must_have_data

# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    jd_text = """https://www.youtube.com/@imjennim is hiring a Producer/Video Editor 
    based in New York (1st priority) or remote from the US to help her scale her channel in the Entertainment/Education/Food & Cooking vertical. They want someone with deep experience in TikTok. Her top required skills: Storyboarding, Sound Designing, Rough Cut & Sequencing, Filming Her budget: 100-150 per hour. Jenn would prefer female candidates but not a must.

    """

    # Pass 1: Full details
    jd_full = extract_job_json(jd_text)
    print("🔹 Full JD Details:", json.dumps(jd_full, indent=2))

    # Save
    with open("jd_full.json", "w") as f:
        json.dump(jd_full, f, indent=4)

    # Pass 2: Must-have details
    jd_must = extract_must_have_details(jd_text)
    print("✅ Must-Have JD Details:", json.dumps(jd_must, indent=2))

    # Save
    with open("jd_must_have.json", "w") as f:
        json.dump(jd_must, f, indent=4)
