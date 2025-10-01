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


# Load environment
load_dotenv()

st.title("🎤 Generative AI Mock Interviewer")
st.write("Practice interviews with AI. Choose a role and get real-time questions & feedback.")

# API Key
api_key = st.text_input("Enter your GROQ API Key:", type="password")

if api_key:
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=api_key)

    # User Input: Role & Resume
    role = st.selectbox("Select Job Role:", 
                        ["Data Scientist", "AI Engineer", "Software Developer", "ML Researcher", "Custom Role"])
    
    if role == "Custom Role":
        role = st.text_input("Enter custom role:")

    resume_text = st.text_area("Paste Resume (optional)", 
                               "BSc Computer Engineering, experience in Python, ML, and NLP projects...")

    st.subheader("🧑‍💼 AI Mock Interview")

    # Start Interview
    if st.button("Start Interview"):
        st.session_state["interview_started"] = True
        st.session_state["question_number"] = 1

    if "interview_started" in st.session_state and st.session_state["interview_started"]:
        # Generate Interview Question
        q_prompt = f"""
        You are acting as an interviewer for the role: {role}.
        Ask the candidate a challenging interview question.
        Use resume context if available: {resume_text}.
        """
        question = llm.invoke(q_prompt).content

        st.subheader(f"❓ Question {st.session_state['question_number']}")
        st.write(question)

        # Candidate Answer
        answer = st.text_area("Your Answer:", key=f"answer_{st.session_state['question_number']}")

        if st.button("Submit Answer"):
            feedback_prompt = f"""
            You are an AI Interview Evaluator. The role is: {role}.
            Interview question: {question}
            Candidate's answer: {answer}

            Provide:
            1. Score (0-10)
            2. Strengths in the answer
            3. Weaknesses
            4. How to improve
            """

            feedback = llm.invoke(feedback_prompt).content

            st.subheader("📊 Feedback")
            st.write(feedback)

            # Move to next question
            st.session_state["question_number"] += 1
            st.success("Next question will be generated when you continue.")

else:
    st.warning("Please enter your GROQ API Key to continue.")
