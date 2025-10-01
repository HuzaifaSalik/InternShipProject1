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
st.title("🤖 Intern Query Chatbot")
st.write("Upload your intern handbook / company policies (PDF) and ask questions.")

# API Key
api_key = st.text_input("Enter your GROQ API Key:", type="password")

if api_key:
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=api_key)

    # Upload Docs
    uploaded_files = st.file_uploader("Upload Intern Documents (PDF)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            temp_path = os.path.join("./", file_name)

            with open(temp_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        # Split docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)

        # Vector store
        vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Prompt
        system_prompt = """
        You are a helpful assistant for interns. 
        Answer questions only from the provided context. 
        If you don't know, say "Sorry, I couldn’t find that in the documents."
        Keep answers short and clear.
        \n\n{context}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        # User Input
        user_input = st.text_input("Ask your intern query:")

        if user_input:
            response = qa_chain.invoke({"input": user_input, "context": split_docs,"chat_history": []})
            st.subheader("💡 Answer")
            st.write(response)

else:
    st.warning("Please enter your GROQ API Key to continue.")
