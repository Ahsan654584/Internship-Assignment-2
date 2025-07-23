import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HuggingFace API Token Here")  # ✅ Make sure it's defined in .env

# Title
st.title("Context-Aware Chatbot")

# Cache document vectorstore
@st.cache_resource
def load_knowledge():
    loader = TextLoader("data/knowledge.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

retriever = load_knowledge()

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load small model from Hugging Face
text_gen = pipeline(
    "text-generation",
    model="distilgpt2",  # Lightweight model (downloads fast and runs easily)
    max_new_tokens=150,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=text_gen)

# LangChain Retrieval QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
question = st.text_input("Ask a question:")

if question:
    answer = qa_chain.run(question)
    st.session_state.chat_history.append(("You", question))
    st.session_state.chat_history.append(("Bot", answer))

# Display chat
for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")
