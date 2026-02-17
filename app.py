import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Resume Screening AI")
st.write("Match resumes with job descriptions using AI and semantic similarity")

@st.cache_data
def load_data():
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df = df.drop_duplicates(subset="Resume")
    df = df.reset_index(drop=True)
    return df

df = load_data()

resumes = df["Resume"].tolist()
categories = df["Category"].tolist()

st.write(f"📊 Total unique resumes loaded: {len(resumes)}")

@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model()

@st.cache_resource
def create_embeddings(resume_list):
    embeddings = model.encode(resume_list)
    return embeddings

resume_embeddings = create_embeddings(resumes)

job_description = st.text_area(
    "Enter Job Description",
    placeholder="Example: Looking for Python developer with machine learning and NLP experience"
)

if st.button("Match Resumes"):

    if job_description.strip() == "":
        st.warning("⚠️ Please enter a job description")
    
    else:
        job_embedding = model.encode([job_description])
        similarity = cosine_similarity(job_embedding, resume_embeddings)
        scores = similarity.flatten()
        valid_indices = np.where(scores > 0.25)[0]
        sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
        st.subheader("🎯 Top Matching Resumes")
        if len(sorted_indices) == 0:
            st.error("No strong matches found. Try a more detailed job description.")
        else:
            rank = 1
            for index in sorted_indices[:5]:
                score_percent = scores[index] * 100
                st.write(f"### Rank {rank}")
                st.write(f"**Category:** {categories[index]}")
                st.write(f"**Match Score:** {score_percent:.2f}%")
                with st.expander("View Resume"):
                    st.write(resumes[index])
                st.write("---")
                rank += 1

st.markdown("---")
st.write("✅ AI Resume Screening System using BERT Semantic Matching")
