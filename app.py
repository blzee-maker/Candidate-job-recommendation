import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessing import dataPreprocessor  # your class

# -------------------------
# Load Candidate Dataset
# -------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("candidates.csv")

    multi_hot_cols = [
        "Skills", "Software", "Content Verticals",
        "Creative Styles", "Platforms", "Past Creators"
    ]
    numeric_cols = ["Monthly Rate", "Hourly Rate", "# of Views by Creators"]

    preprocessor = dataPreprocessor(multi_hot_cols, numeric_cols)
    candidate_vectors = preprocessor.fit_transform(df)

    return df, candidate_vectors, preprocessor

df, candidate_vectors, preprocessor = load_data()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔎 Candidate Recommendation System")

st.write("Enter a Job Description (JD) and find the Top 10 matching candidates.")

jd_text = st.text_area("Paste Job Description:", height=200)

if st.button("Find Top 10 Candidates"):
    if not jd_text.strip():
        st.warning("Please enter a job description first.")
    else:
        # Process JD into vector
        jd_vector = preprocessor.process_job_description(jd_text)

        # Compute cosine similarity
        scores = cosine_similarity(candidate_vectors, jd_vector).flatten()

        # Attach scores to original df
        df["MatchScore"] = scores

        # Get Top 10
        top10 = df.sort_values("MatchScore", ascending=False).head(10)

        st.subheader("Top 10 Recommended Candidates")
        st.dataframe(
            top10[["First Name", "Last Name", "City", "Country"]],
            use_container_width=True
        )
