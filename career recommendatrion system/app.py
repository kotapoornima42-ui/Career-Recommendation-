import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer, util
st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="🎯"
)

st.title("🎯 Career Recommendation System")
st.write("Enter your details to get career recommendations.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

careers = {
    "Data Scientist": "python data science statistics machine learning pandas numpy analytics",
    "Data Analyst": "excel sql python data analysis statistics visualization",
    "Machine Learning Engineer": "python machine learning deep learning algorithms",
    "Web Developer": "html css javascript react frontend backend web development",
    "Software Developer": "python java c++ programming software development algorithms",
    "AI Engineer": "artificial intelligence machine learning deep learning python neural networks",
    "Cybersecurity Analyst": "cybersecurity networking ethical hacking security linux",
    "Cloud Engineer": "cloud computing aws azure devops linux docker kubernetes",
    "UI/UX Designer": "ui ux design figma user experience creativity",
    "Business Analyst": "business analysis requirements sql excel communication problem solving"
}

st.subheader("👤 Student Information")

name = st.text_input("Your Name")

education = st.selectbox(
    "Education",
    ["B.Tech / B.E", "B.Sc", "BCA", "MCA", "M.Tech", "MBA", "Other"]
)

skills = st.text_area(
    "Your Skills",
    placeholder="Example: Python, SQL, Excel, Machine Learning"
)

interests = st.text_area(
    "Your Interests",
    placeholder="Example: Data Science, AI, Web Development"
)

if st.button("🔍 Recommend Careers"):

    if not name or not skills or not interests:
        st.warning("Please fill in your Name, Skills and Interests.")

    else:
        profile = f"""
        Education: {education}
        Skills: {skills}
        Interests: {interests}
        """

        profile_embedding = model.encode(
            profile,
            convert_to_tensor=True
        )

        results = []

        for career, description in careers.items():

            career_embedding = model.encode(
                description,
                convert_to_tensor=True
            )

            score = util.cos_sim(
                profile_embedding,
                career_embedding
            ).item()

            results.append((career, score))

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        st.success(f"Hello {name}! Here are your recommended careers.")

        st.subheader("🏆 Top Career Recommendations")

        for i, (career, score) in enumerate(results[:5], 1):

            percentage = max(
                0,
                min(100, score * 100)
            )

            st.write(f"### {i}. {career}")

            st.progress(
                int(percentage),
                text=f"Match: {percentage:.1f}%"
            )
