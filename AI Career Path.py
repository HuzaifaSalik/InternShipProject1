import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()

st.title("🎯 AI Career Path Recommender")
st.write("Get personalized AI career path recommendations based on your skills and interests.")

# API Key
api_key = st.text_input("Enter your GROQ API Key:", type="password")

if api_key:
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=api_key)

    # Input Section
    st.subheader("📄 Provide Your Profile")

    option = st.radio("Choose input method:", ["Upload Resume (PDF)", "Enter Skills & Interests"])

    user_profile = ""

    if option == "Upload Resume (PDF)":
        uploaded_resume = st.file_uploader("Upload Resume", type="pdf")
        if uploaded_resume:
            import fitz  # PyMuPDF for reading PDF
            doc = fitz.open(stream=uploaded_resume.read(), filetype="pdf")
            resume_text = ""
            for page in doc:
                resume_text += page.get_text("text")
            user_profile = resume_text

    elif option == "Enter Skills & Interests":
        skills = st.text_area("List your skills (comma separated):", 
                              "Python, Machine Learning, Data Analysis")
        interests = st.text_area("What areas of AI interest you?", 
                                 "Natural Language Processing, Computer Vision")
        education = st.text_area("Your Education Background:", 
                                 "BSc Computer Engineering")
        user_profile = f"Skills: {skills}\nInterests: {interests}\nEducation: {education}"

    # Generate Recommendations
    if st.button("🔍 Recommend Career Paths") and user_profile:
        prompt = f"""
        You are an expert AI Career Advisor. Based on the following profile, 
        recommend the best career paths in AI. Output must include:

        1. Suggested AI Career Paths
        2. Why these paths fit the user
        3. Missing Skills (if any)
        4. Suggested Roadmap (courses, tools, resources)

        User Profile:
        {user_profile}
        """

        response = llm.invoke(prompt)

        st.subheader("🚀 Career Path Recommendations")
        st.write(response.content)

else:
    st.warning("Please enter your GROQ API Key to continue.")
