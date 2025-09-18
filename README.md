Candidate–Job Recommendation System
📌 Overview

This project is an AI-powered recommendation system that matches job descriptions with candidate profiles.
It uses Large Language Models (LLMs) for information extraction and vector similarity search to recommend the best candidates for a given job posting.

✨ Features

Extracts structured information from job descriptions (skills, responsibilities, requirements).

Generates embeddings for both job descriptions and candidate data.

Uses vector similarity search to find top-matching candidates.

Provides a ranking score for candidate-job matches.

Web interface for uploading job descriptions and displaying the Top10 Candidates.

🏗️ Tech Stack

Python – Core backend

Google Gemini – LLM-powered text extraction

Vector similarity search (can be replaced with Pinecone/Weaviate)

Streamlit – Web interface (depending on implementation)

pandas & numpy – Data processing

📂 Project Structure
.
├── app.py                # Main web app / API  
├── .env                  # Holds the sensitive data (Gemini API)
├── jd_extraction.py      # Extracts job description details  
├── multi_criteria_structural_score.py   # Matching and scoring   
├── requirements.txt      # Dependencies  
├── README.md             # Documentation
├── Candiates.csv         # Candidate dataset  
└── Documentation  

🚀 Installation

Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate      # On Windows


Install dependencies:

pip install -r requirements.txt

Set up environment variables (API keys for Gemini).

export GEMINI_API_KEY="your_api_key"

⚙️ Usage

There are 2 working files here:

1. app.py (Main File)

contains the frontend code based on streamlit for UX - Simple (Upload and Display Structure)

2. Data_preprocessing.py (Backend Working)

contains the actual code that is processing the data and ranking the candidates


Run the app:

streamlit run app.py

🔮 Future Enhancements

Use LLMs to extract not only skills but also personality traits.

Replace FAISS with a cloud vector DB (e.g., Pinecone, Weaviate).

Train a learning-to-rank model on historical hiring data.

Add a dashboard for recruiters to manage candidates and jobs.

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.