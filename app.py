import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pypdf import PdfReader

load_dotenv()


# ---------- UI ----------
def load_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --panel-soft: #1f2937;
            --line: rgba(255,255,255,0.08);
            --text: #f8fafc;
            --muted: #cbd5e1;
            --primary: #7c3aed;
            --primary-2: #3b82f6;
            --success: #10b981;
        }

        .stApp {
            background: linear-gradient(180deg, #020817 0%, #0f172a 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-header {
            padding: 1.2rem 1.5rem;
            border: 1px solid var(--line);
            background: rgba(17, 24, 39, 0.9);
            border-radius: 18px;
            margin-bottom: 1.4rem;
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
        }

        .muted {
            color: var(--muted);
        }

        .card {
            border: 1px solid var(--line);
            background: rgba(17, 24, 39, 0.9);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .chat-box {
            border: 1px solid var(--line);
            background: rgba(15, 23, 42, 0.85);
            border-radius: 16px;
            padding: 1rem;
            max-height: 480px;
            overflow-y: auto;
        }

        .user-msg, .bot-msg {
            margin: 0.8rem 0;
            padding: 0.9rem 1rem;
            border-radius: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .user-msg {
            background: linear-gradient(135deg, var(--primary), var(--primary-2));
            color: white;
            margin-left: 1.5rem;
        }

        .bot-msg {
            background: rgba(31, 41, 55, 0.9);
            border: 1px solid var(--line);
            color: var(--text);
            margin-right: 1.5rem;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--primary-2));
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.7rem 1.2rem;
        }

        .stTextInput > div > div > input {
            background: rgba(31, 41, 55, 0.85);
            color: white;
            border: 1px solid var(--line);
            border-radius: 12px;
        }

        .stFileUploader > section {
            background: rgba(17, 24, 39, 0.8);
            border: 1px dashed rgba(124, 58, 237, 0.6);
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Session state ----------
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
    if "documents_uploaded" not in st.session_state:
        st.session_state.documents_uploaded = 0
    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = 0


# ---------- Document helpers ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"
    return text


def get_text_file_text(text_docs):
    text = ""
    for doc in text_docs:
        text += doc.read().decode("utf-8", errors="ignore")
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    if not st.session_state.api_key:
        st.error("Please add your Google API key in the sidebar.")
        return False

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.session_state.vector_store_created = True
        st.session_state.documents_uploaded += 1
        return True
    except Exception as e:
        st.error(f"Unable to create vector store: {e}")
        return False


# ---------- AI helpers ----------
def get_conversational_chain():
    prompt_template = """
    Answer the question using only the provided context.
    If the answer is not available in the context, say:
    "Answer is not available in the uploaded documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    if not st.session_state.vector_store_created:
        st.warning("Please upload and process documents first.")
        return None

    if not st.session_state.api_key:
        st.warning("Please set your Google API key.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]

        st.session_state.chat_history.append(
            {
                "question": user_question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        st.session_state.questions_asked += 1
        return answer
    except Exception as e:
        st.error(f"Unable to process the question: {e}")
        return None


def generate_summary(document_text):
    if not st.session_state.api_key:
        return "Please add your Google API key first."

    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = f"Summarize the following document in 150-250 words:\n\n{document_text[:5000]}"
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Summary failed: {e}"


# ---------- Render helpers ----------
def render_chat_history():
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="card" style="text-align:center; padding:2rem; color:#cbd5e1;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">💬</div>
                <div>No conversation yet. Upload documents and start asking questions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for item in st.session_state.chat_history[-8:]:
        st.markdown(f'<div class="user-msg"><strong>You:</strong><br>{item["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-msg"><strong>AI:</strong><br>{item["answer"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Main app ----------
def main():
    load_css()
    init_session_state()

    st.set_page_config(page_title="AI Document Intelligence", page_icon="📄", layout="wide")

    with st.sidebar:
        st.markdown("### 🔑 API key")
        api_key_input = st.text_input(
            "Google API Key",
            value=st.session_state.api_key,
            type="password",
            help="Add your Google Generative AI API key.",
        )
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            os.environ["GOOGLE_API_KEY"] = api_key_input

        st.markdown("---")
        st.markdown("### 📊 Stats")
        st.metric("Documents", st.session_state.documents_uploaded)
        st.metric("Questions", st.session_state.questions_asked)
        st.metric("Status", "Ready" if st.session_state.vector_store_created else "Waiting")

    st.markdown(
        """
        <div class="main-header">
            <h1>📄 AI Document Intelligence</h1>
            <div class="muted">Upload a document and ask questions in natural language.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📤 Upload", "💬 Chat", "📋 Summary"])

    with tab1:
        st.markdown("### Upload PDF or text files")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        text_docs = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

        if st.button("Process Documents", type="primary"):
            if not pdf_docs and not text_docs:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing your files..."):
                    text = ""

                    if pdf_docs:
                        for pdf in pdf_docs:
                            text += get_pdf_text([pdf])

                    if text_docs:
                        for doc in text_docs:
                            text += get_text_file_text([doc])

                    if text.strip():
                        chunks = get_text_chunks(text)
                        if get_vector_store(chunks):
                            st.session_state.document_text = text
                            st.success("Documents processed successfully.")
                    else:
                        st.error("No text could be extracted from the uploaded files.")

    with tab2:
        st.markdown("### Ask about your uploaded documents")
        render_chat_history()

        user_question = st.text_input("Your question", placeholder="Type your question here...")
        if st.button("Ask") and user_question:
            with st.spinner("Thinking..."):
                answer = user_input(user_question)
                if answer:
                    st.success("Answer generated.")
                    st.rerun()

    with tab3:
        st.markdown("### Summary")
        if "document_text" in st.session_state:
            if st.button("Generate Summary"):
                summary = generate_summary(st.session_state.document_text)
                st.write(summary)
        else:
            st.info("Upload and process documents before generating a summary.")

        st.markdown("---")

        if st.session_state.chat_history:
            st.download_button(
                "Download chat history",
                data="\n\n".join(
                    [
                        f"Q: {item['question']}\nA: {item['answer']}\nTime: {item['timestamp']}"
                        for item in st.session_state.chat_history
                    ]
                ),
                file_name="chat_history.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
