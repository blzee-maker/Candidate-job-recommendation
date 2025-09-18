import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from typing import Dict, List

class dataPreprocessor:
    def __init__(self, multi_hot_cols: List[str], numeric_cols: List[str]):
        """
        Args:
            multi_hot_cols: list of categorical multi-label fields (comma-separated strings).
            numeric_cols: list of numeric fields to normalize.
        """
        self.multi_hot_cols = multi_hot_cols
        self.numeric_cols = numeric_cols
        self.mlb_dict: Dict[str, MultiLabelBinarizer] = {}
        self.scaler = MinMaxScaler()

    def clean_split(self, x):
        """Split comma-separated string into clean lowercase tokens, handle lists too."""
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, list):
            return [i.strip().lower() for i in x if isinstance(i, str) and i.strip()]
        if isinstance(x, str):
            return [i.strip().lower() for i in x.split(",") if i.strip()]
        return []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders and transform candidate dataset into vectors."""
        transformed_features = []

        # --- Multi-hot encode ---
        for col in self.multi_hot_cols:
            df[col] = df[col].apply(self.clean_split)
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(df[col])
            self.mlb_dict[col] = mlb  # save for later use
            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{col}_{c}" for c in mlb.classes_],
                index=df.index
            )
            transformed_features.append(encoded_df)

        # --- Normalize numeric features ---
        df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])
        transformed_features.append(df[self.numeric_cols])

        # --- Combine all candidate vectors ---
        candidate_vectors = pd.concat(transformed_features, axis=1)

        return candidate_vectors

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data (using fitted encoders)."""
        transformed_features = []

        for col in self.multi_hot_cols:
            df[col] = df[col].apply(self.clean_split)
            mlb = self.mlb_dict[col]
            encoded = mlb.transform(df[col])
            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{col}_{c}" for c in mlb.classes_],
                index=df.index
            )
            transformed_features.append(encoded_df)

        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        transformed_features.append(df[self.numeric_cols])

        candidate_vectors = pd.concat(transformed_features, axis=1)

        return candidate_vectors
    
    def process_job_description(self, jd_text: str) -> pd.DataFrame:
        """
        Convert JD text into the same multi-hot + numeric vector format.
        """
        jd_text = jd_text.lower()

        jd_data = {}

        # --- Multi-hot features: check if vocab words appear in JD text ---
        for col, mlb in self.mlb_dict.items():
            vocab = mlb.classes_
            matched = [word for word in vocab if word in jd_text]
            jd_data[col] = [matched]

        # --- Numeric features (set to neutral = 0.5 if not specified) ---
        for col in self.numeric_cols:
            jd_data[col] = [0.5]

        # Convert to DataFrame
        jd_df = pd.DataFrame(jd_data)

        # Transform using the same pipeline
        jd_vector = self.transform(jd_df)
        return jd_vector


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    # Load candidate dataset
    df = pd.read_csv("candidates.csv")

    multi_hot_cols = [
        "Skills", "Software", "Content Verticals",
        "Creative Styles", "Platforms", "Past Creators"
    ]
    numeric_cols = ["Monthly Rate", "Hourly Rate", "# of Views by Creators"]

    preprocessor = dataPreprocessor(multi_hot_cols, numeric_cols)

    # Step 1: Candidate preprocessing
    candidate_vectors = preprocessor.fit_transform(df)
    print("Candidate vector shape:", candidate_vectors.shape)

    # Step 2: JD preprocessing
    jd_text = """
    is looking for a talented Video Editor with experience in Adobe Premiere Pro 
    who can edit content in Entertainment/Lifestyle & Vlogs categories.
    Historically, his team has preferred to hire talent in Asia for cheaper but is 
    open to all locations for this role. Content form: Any short-form and long-forms 
    Their top required skills: Splice & Dice, Rough Cut & Sequencing, 2D Animation 
    Their budget: $2500/month

    """
    jd_vector = preprocessor.process_job_description(jd_text)
    print("JD vector shape:", jd_vector.shape)

    # Step 3: Compare JD vs candidates
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(candidate_vectors, jd_vector)
    df["MatchScore"] = scores
    top50 = df.sort_values("MatchScore", ascending=False).head(50)

    print(top50[["First Name", "Last Name", "MatchScore"]])
