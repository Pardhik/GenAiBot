<<<<<<< HEAD
# Information Retrieval System 📚 

A smart document query system that uses TF-IDF and semantic search to find relevant information from multiple PDF documents.

## 🌟 Features

- **PDF Document Processing**: Upload and process multiple PDF files simultaneously
- **Smart Search**: Uses TF-IDF vectorization with both unigrams and bigrams
- **Context-Aware Responses**: Returns relevant excerpts with surrounding context
- **Fallback Mechanisms**: Graceful degradation when external APIs are unavailable
- **Clean UI**: Interactive Streamlit interface with clear result presentation

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Pardhik/GenAiBot.git
cd GenAiBot
```

### 2. Set Up Environment
```bash
# Create and activate conda environment
conda create -n docquery python=3.12 -y
conda activate docquery

# Install requirements
pip install -r requirements.txt
```

### 3. Optional: Configure OpenAI (Optional)
Create a `.env` file in the root directory:
```ini
OPENAI_API_KEY=your_api_key_here
```
Note: The system works without OpenAI API, using local TF-IDF search by default.

### 4. Run the Application
```bash
streamlit run app.py
```
Access the web interface at http://localhost:8501

## 💡 How It Works

1. **Document Processing**:
   - PDFs are loaded and text is extracted
   - Text is split into meaningful chunks
   - TF-IDF vectors are created for semantic search

2. **Search Process**:
   - User queries are processed using TF-IDF
   - Most relevant document chunks are retrieved
   - Context around matches is extracted
   - Results are presented in a clean, readable format

3. **Intelligent Fallbacks**:
   - Local TF-IDF search by default
   - Optional OpenAI integration if configured
   - Graceful error handling and user feedback

## 🛠️ Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web interface and UI components
- **scikit-learn**: TF-IDF vectorization and similarity search
- **NLTK**: Text processing and tokenization
- **PyPDF2**: PDF document processing
- **FAISS**: Vector similarity search (optional)
- **LangChain**: Document processing pipeline

## 📋 Usage Guide

1. **Start the Application**:
   - Run `streamlit run app.py`
   - Open your browser to `http://localhost:8501`

2. **Upload Documents**:
   - Use the sidebar to upload one or more PDF files
   - Click "Submit & Process" to index the documents

3. **Ask Questions**:
   - Type your question in the input field
   - View relevant excerpts from the documents
   - Context is automatically included in responses

## 🔧 Configuration

The system can be configured through environment variables:
- `USE_OPENAI`: Set to `True` to enable OpenAI integration (default: False)
- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI)
- `HTTP_PROXY`/`HTTPS_PROXY`: Proxy settings if needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
=======

>>>>>>> 1a15db993b9e4f6b14a008bffae681307ddd8f9f
