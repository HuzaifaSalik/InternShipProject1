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

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("🧠 AI Resume Evaluator")
st.write("Upload your resume in PDF format and paste the job description. The AI will evaluate your resume.")

# API Key
api_key = st.text_input("Enter your GROQ API Key:", type="password")

if api_key:
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=api_key)

    # Upload Resume
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")

    # Job Description
    job_description = st.text_area("Paste the Job Description here:")

    if uploaded_resume and job_description:
        # Process Resume
        file_name = uploaded_resume.name
        temp_path = os.path.join("./", file_name)
        with open(temp_path, "wb") as file:
            file.write(uploaded_resume.getvalue())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        # Create vector DB
        vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # System Prompt for Evaluation
        evaluation_prompt = ChatPromptTemplate.from_template("""
        You are an AI Resume Evaluator. Evaluate the following resume 
        against the given job description. Provide output in this format:

        **Resume Evaluation Report**
        1. Match Score (0-100): 
        2. Key Strengths: 
        3. Weaknesses / Missing Skills: 
        4. Suggestions for Improvement: 

        Resume Content:
        {context}

        Job Description:
        {job_description}
        """)

        # Create QA Chain
        qa_chain = create_stuff_documents_chain(llm, evaluation_prompt)

        # Run Chain
        response = qa_chain.invoke(
            {"input": "Evaluate my resume", 
             "context": split_docs, 
             "job_description": job_description}
        )

        # Display Results
        st.subheader("📊 Resume Evaluation Report")
        st.write(response)

else:
    st.warning("Please enter your GROQ API Key to continue.")
