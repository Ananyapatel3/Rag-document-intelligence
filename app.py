import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Custom CSS for modern, premium UI
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background with subtle pattern */
    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        background-attachment: fixed;
        color: white;
        min-height: 100vh;
    }

    /* Subtle background pattern */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            radial-gradient(circle at 25% 25%, rgba(102, 126, 234, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(118, 75, 162, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }

    /* Sidebar Styling - Modern Glassmorphism */
    .sidebar .sidebar-content {
        background: rgba(30, 41, 59, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Sidebar Header */
    .sidebar-header {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 50%, #F093FB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .sidebar-subtitle {
        font-size: 0.9rem;
        opacity: 0.7;
        margin: 0.5rem 0 0 0;
    }

    /* Hero Section - Premium Design */
    .hero-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        box-shadow:
            0 20px 40px rgba(0, 0, 0, 0.3),
            0 0 60px rgba(102, 126, 234, 0.1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        animation: rotate 20s linear infinite;
        pointer-events: none;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 50%, #FFFFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 2;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 2rem;
        position: relative;
        z-index: 2;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.4;
    }

    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 2;
    }

    .hero-badge {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    .hero-badge:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }

    /* Feature Cards - Enhanced Glassmorphism */
    .feature-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s;
    }

    .feature-card:hover::before {
        left: 100%;
    }

    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow:
            0 20px 40px rgba(102, 126, 234, 0.3),
            0 0 60px rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }

    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        color: white;
    }

    .feature-description {
        margin: 0;
        opacity: 0.8;
        line-height: 1.5;
    }

    /* Metrics Cards - Premium Design */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow:
            0 8px 32px rgba(102, 126, 234, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667EEA, #764BA2, #F093FB);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow:
            0 20px 40px rgba(102, 126, 234, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }

    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 50%, #F093FB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }

    /* Tab Styling - Modern Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.6);
        border-radius: 16px;
        padding: 0.75rem;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.7);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.15);
        color: white;
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 50%, #F093FB 100%);
        color: white;
        box-shadow:
            0 4px 20px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* Chat Bubbles - Premium Design */
    .chat-container {
        max-height: 800px;
        min-height: 500px;
        overflow-y: auto;
        padding: 2rem;
        background: rgba(30, 41, 59, 0.6);
        border-radius: 24px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin: 2rem 0;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
    }

    .chat-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, #667EEA, #764BA2, #F093FB);
        opacity: 0.5;
    }

    .user-bubble {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 24px 24px 6px 24px;
        margin: 1rem 0;
        margin-left: 2rem;
        box-shadow:
            0 6px 24px rgba(102, 126, 234, 0.4),
            0 0 40px rgba(102, 126, 234, 0.1);
        animation: slideInLeft 0.6s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.15);
        position: relative;
        max-width: 80%;
        word-wrap: break-word;
        line-height: 1.6;
        font-size: 1.05rem;
    }

    .user-bubble::before {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 30px;
        width: 0;
        height: 0;
        border-left: 10px solid transparent;
        border-right: 10px solid transparent;
        border-top: 10px solid rgba(102, 126, 234, 0.9);
    }

    .ai-bubble {
        background: rgba(30, 41, 59, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 24px 24px 24px 6px;
        margin: 1rem 0;
        margin-right: 2rem;
        box-shadow:
            0 6px 24px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: slideInRight 0.6s ease-out;
        position: relative;
        max-width: 80%;
        word-wrap: break-word;
        line-height: 1.6;
        font-size: 1.05rem;
    }

    .ai-bubble::before {
        content: '';
        position: absolute;
        bottom: -10px;
        right: 30px;
        width: 0;
        height: 0;
        border-left: 10px solid transparent;
        border-right: 10px solid transparent;
        border-top: 10px solid rgba(30, 41, 59, 0.95);
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateX(0) scale(1);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateX(0) scale(1);
        }
    }

    /* Upload Area - Premium Drag & Drop */
    .upload-area {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(20px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .upload-area::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .upload-area:hover::before {
        opacity: 1;
    }

    .upload-area:hover {
        border-color: #667EEA;
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-4px) scale(1.01);
        box-shadow:
            0 20px 40px rgba(102, 126, 234, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        opacity: 0.7;
        transition: all 0.3s ease;
    }

    .upload-area:hover .upload-icon {
        opacity: 1;
        transform: scale(1.1);
    }

    .upload-text {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: white;
    }

    .upload-subtext {
        opacity: 0.7;
        font-size: 0.9rem;
    }

    /* Button Styling - Premium Design */
    .stButton button {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 50%, #F093FB 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow:
            0 4px 20px rgba(102, 126, 234, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }

    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .stButton button:hover::before {
        left: 100%;
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow:
            0 8px 30px rgba(102, 126, 234, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }

    .stButton button:active {
        transform: translateY(-1px);
    }

    /* Input Styling - Modern Design */
    .stTextInput input {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        color: white;
        padding: 1.25rem 1.5rem;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
        font-size: 1.1rem;
        line-height: 1.5;
        min-height: 20px;
    }

    .stTextInput input:focus {
        border-color: #667EEA;
        box-shadow:
            0 0 20px rgba(102, 126, 234, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        background: rgba(30, 41, 59, 0.9);
        transform: scale(1.01);
    }

    .stTextInput input::placeholder {
        color: rgba(255, 255, 255, 0.5);
        font-size: 1rem;
    }

    /* Chat Input Container */
    .chat-input-container {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    /* Quick Suggestions */
    .suggestion-button {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        color: white;
        transition: all 0.3s ease;
        margin: 0.25rem;
        font-size: 0.9rem;
    }

    .suggestion-button:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        background: rgba(30, 41, 59, 0.9);
        border-color: rgba(102, 126, 234, 0.3);
    }

    /* Footer - Premium Design */
    .footer {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 4rem;
        border-radius: 24px 24px 0 0;
        box-shadow: 0 -8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
    }

    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, #667EEA, #764BA2, #F093FB);
    }

    .footer-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 50%, #F093FB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }

    .footer-subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    .footer-meta {
        font-size: 0.9rem;
        opacity: 0.6;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Loading Animation */
    .loading-dots {
        display: inline-block;
    }

    .loading-dots::after {
        content: '';
        animation: loading 1.5s infinite;
    }

    @keyframes loading {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667EEA, #764BA2);
        border-radius: 4px;
        transition: all 0.3s ease;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5A67D8, #6B46C1);
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }

        .hero-subtitle {
            font-size: 1.1rem;
        }

        .hero-badges {
            gap: 0.5rem;
        }

        .hero-badge {
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
        }

        .feature-card {
            padding: 1.5rem;
        }

        .metric-card {
            padding: 1.5rem;
        }

        .upload-area {
            padding: 2rem 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store_created' not in st.session_state:
        st.session_state.vector_store_created = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('GOOGLE_API_KEY', '')
    if 'documents_uploaded' not in st.session_state:
        st.session_state.documents_uploaded = 0
    if 'questions_asked' not in st.session_state:
        st.session_state.questions_asked = 0

# Utility functions
def get_pdf_text(pdf_docs):
    """Extract text from PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_file_text(text_docs):
    """Extract text from text files"""
    text = ""
    for doc in text_docs:
        text += doc.read().decode('utf-8')
    return text

def get_text_chunks(text):
    """Split text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store"""
    if not st.session_state.api_key:
        st.error("⚠️ Please set your Google API Key in the sidebar.")
        return False

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.session_state.vector_store_created = True
        st.session_state.documents_uploaded += 1
        return True
    except Exception as e:
        st.error(f"❌ Unable to create vector store: {e}")
        return False

def generate_summary(text):
    """Generate AI-powered document summary"""
    if not st.session_state.api_key:
        return "⚠️ Please set your Google API Key."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = f"Summarize the following document in 200-300 words:\n\n{text[:5000]}"
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ Error generating summary: {e}"

def user_input(user_question):
    """Process user question and return AI response"""
    if not st.session_state.vector_store_created:
        st.error("📄 No vector store found. Please upload and process documents first.")
        return None

    if not st.session_state.api_key:
        st.error("🔑 Please set your Google API Key in the sidebar.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        # Add to chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response["output_text"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.questions_asked += 1

        return response["output_text"]
    except Exception as e:
        st.error(f"❌ Unable to process the question: {e}")
        return None

def get_conversational_chain():
    """Create conversational chain for Q&A"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, just say:
    'Answer is not available in the uploaded documents.'

    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def export_chat_history():
    """Export chat history as downloadable text"""
    if not st.session_state.chat_history:
        st.warning("📝 No chat history to export.")
        return

    chat_text = "\n\n".join([f"Q: {item['question']}\nA: {item['answer']}\nTime: {item['timestamp']}" for item in st.session_state.chat_history])
    st.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

def create_metric_card(title, value, icon):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def create_feature_card(title, description, icon):
    """Create a styled feature card"""
    st.markdown(f"""
    <div class="feature-card fade-in-up">
        <div class="feature-icon">{icon}</div>
        <h3 class="feature-title">{title}</h3>
        <p class="feature-description">{description}</p>
    </div>
    """, unsafe_allow_html=True)

# Main Application
def main():
    # Load CSS
    load_css()

    # Initialize session state
    init_session_state()

    # Page configuration
    st.set_page_config(
        page_title="🤖 AI Document Intelligence",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2 class="sidebar-title">🚀 AI Assistant</h2>
            <p class="sidebar-subtitle">Intelligent Document Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # API Key Input
        st.markdown("### 🔑 API Configuration")
        api_key_input = st.text_input(
            "Google API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your Google Generative AI API Key",
            label_visibility="visible"
        )
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            os.environ['GOOGLE_API_KEY'] = api_key_input
            st.success("✅ API Key updated!")

        st.markdown("---")

        # Metrics
        st.markdown("### 📊 System Metrics")
        create_metric_card("Documents", st.session_state.documents_uploaded, "📄")
        create_metric_card("Questions", st.session_state.questions_asked, "❓")
        status_icon = "🟢" if st.session_state.vector_store_created else "🔴"
        status_text = "Ready" if st.session_state.vector_store_created else "Waiting"
        create_metric_card("Status", status_text, status_icon)

        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Quick Navigation")
        st.markdown("""
        - 🏠 **Home**: Overview & Features
        - 📤 **Upload**: Process Documents
        - 💬 **AI Chat**: Ask Questions
        - 📜 **History**: Chat Records
        - 📋 **Export**: Summaries & Downloads
        """)

    # Main Content
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🤖 AI Document Intelligence</h1>
        <p class="hero-subtitle">Transform your documents into intelligent conversational experiences powered by advanced AI and cutting-edge language models</p>
        <div class="hero-badges">
            <span class="hero-badge">📄 Multi-format Support</span>
            <span class="hero-badge">🧠 AI-Powered Q&A</span>
            <span class="hero-badge">📊 Smart Summaries</span>
            <span class="hero-badge">💾 Persistent History</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "📤 Upload & Process", "💬 AI Chat", "📜 Chat History", "📋 Summary & Export"])

    # Home Tab
    with tab1:
        st.markdown("## 🌟 Welcome to the Future of Document Intelligence")

        col1, col2 = st.columns(2)

        with col1:
            create_feature_card(
                "Smart Document Processing",
                "Upload PDFs and text files to create intelligent knowledge bases with advanced vector embeddings.",
                "📄"
            )
            create_feature_card(
                "AI-Powered Conversations",
                "Ask natural language questions and receive context-aware answers from your documents.",
                "🧠"
            )

        with col2:
            create_feature_card(
                "Intelligent Summaries",
                "Generate comprehensive document summaries using cutting-edge language models.",
                "📊"
            )
            create_feature_card(
                "Complete Chat History",
                "Maintain, search, and export your conversation history with advanced filtering.",
                "💾"
            )

        st.markdown("---")
        st.markdown("### 🚀 How It Works")
        st.markdown("""
        1. **Configure** your Google API key in the sidebar
        2. **Upload** your documents (PDFs, text files)
        3. **Process** them to create a knowledge base
        4. **Chat** with your AI assistant
        5. **Review** history and export summaries
        """)

    # Upload & Process Tab
    with tab2:
        st.markdown("## 📤 Upload & Process Documents")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 PDF Documents")
            st.markdown("""
            <div class="upload-area">
                <div class="upload-icon">📄</div>
                <div class="upload-text">Drop PDF files here</div>
                <div class="upload-subtext">or click to browse</div>
            </div>
            """, unsafe_allow_html=True)
            pdf_docs = st.file_uploader(
                "Upload PDF files",
                accept_multiple_files=True,
                type=['pdf'],
                help="Select one or more PDF files to process",
                label_visibility="collapsed"
            )

        with col2:
            st.markdown("### 📝 Text Documents")
            st.markdown("""
            <div class="upload-area">
                <div class="upload-icon">📝</div>
                <div class="upload-text">Drop text files here</div>
                <div class="upload-subtext">or click to browse</div>
            </div>
            """, unsafe_allow_html=True)
            text_docs = st.file_uploader(
                "Upload text files",
                accept_multiple_files=True,
                type=['txt'],
                help="Select one or more text files to process",
                label_visibility="collapsed"
            )

        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if not pdf_docs and not text_docs:
                st.error("⚠️ Please upload at least one document.")
            else:
                with st.spinner("🔄 Processing documents<span class='loading-dots'></span>"):
                    progress_bar = st.progress(0)
                    raw_text = ""

                    # Process PDFs
                    if pdf_docs:
                        for i, pdf in enumerate(pdf_docs):
                            st.text(f"Processing PDF: {pdf.name}")
                            raw_text += get_pdf_text([pdf])
                            progress_bar.progress((i + 1) / (len(pdf_docs) + len(text_docs)))

                    # Process text files
                    if text_docs:
                        for i, txt in enumerate(text_docs):
                            st.text(f"Processing text: {txt.name}")
                            raw_text += get_text_file_text([txt])
                            progress_bar.progress((len(pdf_docs) + i + 1) / (len(pdf_docs) + len(text_docs)))

                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if get_vector_store(text_chunks):
                            st.session_state.document_text = raw_text
                            st.success("✅ Documents processed successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to process documents.")
                    else:
                        st.error("❌ No text extracted from documents.")

                    progress_bar.empty()

    # AI Chat Tab
    with tab3:
        st.markdown("## 💬 AI Chat Assistant")

        # Chat Interface Container
        st.markdown("""
        <div style="margin: 2rem 0;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="flex: 1; height: 2px; background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);"></div>
                <span style="padding: 0 1rem; background: rgba(30, 41, 59, 0.8); border-radius: 20px; font-size: 0.9rem; opacity: 0.8;">💬 Conversation</span>
                <div style="flex: 1; height: 2px; background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chat Interface
        if st.session_state.chat_history:
            st.markdown("### Recent Conversation")
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            for chat in st.session_state.chat_history[-8:]:  # Show last 8 conversations
                st.markdown(f"""
                <div class="user-bubble">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">You:</div>
                    <div style="line-height: 1.6;">{chat['question']}</div>
                </div>
                <div class="ai-bubble">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">AI Assistant:</div>
                    <div style="line-height: 1.6;">{chat['answer']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Empty chat state
            st.markdown("""
            <div class="chat-container" style="display: flex; align-items: center; justify-content: center; min-height: 300px;">
                <div style="text-align: center; opacity: 0.7;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">💬</div>
                    <div style="font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem;">Start a conversation</div>
                    <div style="font-size: 0.9rem; opacity: 0.8;">Upload documents and ask questions to begin</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced Input Area
        st.markdown("""
        <div style="margin: 2rem 0 1rem 0;">
            <div style="display: flex; align-items: center;">
                <div style="flex: 1; height: 2px; background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);"></div>
                <span style="padding: 0 1rem; background: rgba(30, 41, 59, 0.8); border-radius: 20px; font-size: 0.9rem; opacity: 0.8;">✨ Ask Your Question</span>
                <div style="flex: 1; height: 2px; background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Input Area with improved layout
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

        col1, col2 = st.columns([6, 1])

        with col1:
            user_question = st.text_input(
                "Ask your AI assistant...",
                placeholder="Type your question here... (e.g., 'What are the main benefits?' or 'Summarize the key points')",
                label_visibility="collapsed",
                key="chat_input"
            )

        with col2:
            ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Quick question suggestions when no history
        if not st.session_state.chat_history and st.session_state.vector_store_created:
            st.markdown("### 💡 Quick Suggestions")
            suggestion_cols = st.columns(2)
            suggestions = [
                "What are the main topics covered?",
                "Summarize the key points",
                "What are the benefits mentioned?",
                "Explain the main concepts"
            ]

            for i, suggestion in enumerate(suggestions):
                col_idx = i % 2
                with suggestion_cols[col_idx]:
                    if st.button(f"💭 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                        with st.spinner("🤔 Thinking<span class='loading-dots'></span>"):
                            answer = user_input(suggestion)
                            if answer:
                                st.markdown(f"""
                                <div class="user-bubble">
                                    <div style="font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">You:</div>
                                    <div style="line-height: 1.6;">{suggestion}</div>
                                </div>
                                <div class="ai-bubble">
                                    <div style="font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">AI Assistant:</div>
                                    <div style="line-height: 1.6;">{answer}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.success("✅ Answer generated!")
                                st.rerun()

        if ask_button and user_question:
            with st.spinner("🤔 Thinking<span class='loading-dots'></span>"):
                answer = user_input(user_question)
                if answer:
                    st.markdown(f"""
                    <div class="user-bubble">
                        <div style="font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">You:</div>
                        <div style="line-height: 1.6;">{user_question}</div>
                    </div>
                    <div class="ai-bubble">
                        <div style="font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">AI Assistant:</div>
                        <div style="line-height: 1.6;">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ Answer generated!")
                    st.rerun()

    # Chat History Tab
    with tab4:
        st.markdown("## 📜 Chat History")

        if st.session_state.chat_history:
            st.markdown(f"### Total Conversations: {len(st.session_state.chat_history)}")

            # Search and Filter
            col1, col2, col3 = st.columns(3)

            with col1:
                search_term = st.text_input("🔍 Search conversations", placeholder="Enter keywords...")

            with col2:
                show_all = st.checkbox("Show all history", value=False)

            with col3:
                if st.button("🗑️ Clear History", type="secondary"):
                    st.session_state.chat_history = []
                    st.success("🗑️ Chat history cleared!")
                    st.rerun()

            # Filter history
            filtered_history = st.session_state.chat_history
            if search_term:
                filtered_history = [chat for chat in st.session_state.chat_history
                                  if search_term.lower() in chat['question'].lower()
                                  or search_term.lower() in chat['answer'].lower()]

            # Display history
            history_to_show = filtered_history if show_all else filtered_history[-10:]

            if search_term and not filtered_history:
                st.warning("🔍 No conversations found matching your search.")
            else:
                for i, chat in enumerate(reversed(history_to_show)):
                    with st.expander(f"#{len(st.session_state.chat_history) - st.session_state.chat_history.index(chat)} - {chat['timestamp']}"):
                        st.markdown(f"""
                        <div class="user-bubble" style="margin-bottom: 1rem;">
                            <strong>Question:</strong><br>{chat['question']}
                        </div>
                        <div class="ai-bubble">
                            <strong>AI Response:</strong><br>{chat['answer']}
                        </div>
                        <small style="opacity: 0.6;">{chat['timestamp']}</small>
                        """, unsafe_allow_html=True)

                if not show_all and len(filtered_history) > 10:
                    st.info(f"📄 Showing last 10 conversations. Check 'Show all history' to view all {len(filtered_history)} conversations.")
        else:
            st.info("💬 No chat history yet. Start asking questions in the AI Chat tab!")

    # Summary & Export Tab
    with tab5:
        st.markdown("## 📋 Summary & Export")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Document Summary")
            if 'document_text' in st.session_state:
                if st.button("📝 Generate AI Summary", type="primary", use_container_width=True):
                    with st.spinner("🤖 Generating summary..."):
                        summary = generate_summary(st.session_state.document_text)
                        st.markdown("#### AI-Generated Summary")
                        st.write(summary)
            else:
                st.info("📄 Upload and process documents first to generate a summary.")

        with col2:
            st.markdown("### 💾 Export Options")
            export_chat_history()

            if st.session_state.chat_history:
                st.markdown(f"**Available Exports:** {len(st.session_state.chat_history)} conversations")

    # Footer
    st.markdown("""
    <div class="footer">
        <h3 class="footer-title">🚀 Built for the Future</h3>
        <p class="footer-subtitle">Advanced AI Document Intelligence System with cutting-edge conversational AI</p>
        <p class="footer-meta">Powered by Streamlit, LangChain, Google Generative AI & FAISS • 2026</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()