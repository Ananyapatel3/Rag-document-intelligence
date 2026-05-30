# RAG Document Intelligence System

An advanced AI-powered Document Intelligence Chatbot built with cutting-edge technologies for intelligent document analysis and question answering.

## 🚀 Features

- **Multi-format Support**: Upload and process PDF and text files
- **Intelligent Q&A**: Ask questions and get context-aware answers from your documents
- **Document Summarization**: Generate concise summaries of uploaded documents
- **Comprehensive Chat History**: Dedicated history tab with search, filtering, and export capabilities
- **Interactive Chat Interface**: Real-time Q&A with immediate responses
- **Export Functionality**: Download chat logs and conversation history
- **Advanced UI**: Modern, tabbed interface with real-time feedback and progress indicators
- **Secure API Handling**: Input and manage Google Generative AI API keys securely
- **Session Management**: Persistent state across interactions with clear history management

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, Google Generative AI (Gemini), FAISS Vector Database
- **Document Processing**: PyPDF2, Text File Parsing
- **Environment Management**: python-dotenv
- **Deployment**: Ready for Streamlit Cloud or local deployment

## 📋 Prerequisites

- Python 3.8+
- Google Generative AI API Key (Get from [Google AI Studio](https://makersuite.google.com/app/apikey))

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-document-intelligence
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your Google API Key: `GOOGLE_API_KEY=your_api_key_here`

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage

1. **Launch the app** and navigate to the local URL (usually `http://localhost:8501`)

2. **Configure API Key**: Enter your Google Generative AI API key in the sidebar

3. **Upload Documents**:
   - Go to "Upload & Process" tab
   - Upload PDF or text files
   - Click "Process Documents" to create the knowledge base

4. **Ask Questions**:
   - Switch to "Chat" tab
   - Type your question in the input field
   - View answers and chat history

5. **Review Chat History**:
   - Use "History" tab for detailed conversation review
   - Search through past conversations
   - Filter and sort history entries
   - Clear history when needed

6. **Generate Summaries & Export**:
   - Use "Summary & Export" tab for document summaries
   - Download chat history as text files

## 🎮 Demo

Try the system with the included sample document:
- Navigate to `Data/sample_document.txt`
- Upload and process it in the "Upload & Process" tab
- Ask questions like:
  - "What is RAG?"
  - "How does the system work?"
  - "What are the benefits?"

## 🎨 UI Features

Experience the premium glassmorphism interface with:
- **🌟 Gradient Hero Section**: Large AI title with feature badges
- **💎 Glassmorphism Cards**: Animated feature cards with hover effects
- **📊 Metric Dashboard**: Real-time statistics with glowing effects
- **💬 Modern Chat Bubbles**: Smooth animations and professional styling
- **🎭 Interactive Elements**: Hover effects, transitions, and micro-animations
- **📱 Responsive Design**: Optimized for all screen sizes and devices

## 📁 Project Structure

```
rag-document-intelligence/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # Project documentation
├── faiss_index/          # Vector database storage (created dynamically)
└── Data/                 # Sample data directory
```

## 🔍 How It Works

1. **Document Processing**: Uploaded files are parsed and split into manageable chunks
2. **Vector Embeddings**: Text chunks are converted to vector embeddings using Google Generative AI
3. **Vector Storage**: Embeddings are stored in FAISS for efficient similarity search
4. **Question Processing**: User queries are embedded and matched against document vectors
5. **Answer Generation**: Relevant context is fed to Gemini AI for generating accurate responses

## 🎨 Screenshots

*Add screenshots of your app here*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the powerful AI framework
- [Google Generative AI](https://ai.google.dev/) for advanced language models
- [Streamlit](https://streamlit.io/) for the intuitive web app framework
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for AI-powered document intelligence**
