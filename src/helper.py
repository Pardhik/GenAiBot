import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)


try:
    nltk.data.find('tokenizers/punkt')
except LookupError as e:
    print(f"Warning: Could not find punkt tokenizer. Error: {e}")
    print("Using fallback tokenization method.")


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # Optional API base URL
HTTP_PROXY = os.getenv("HTTP_PROXY")  # Optional HTTP proxy
HTTPS_PROXY = os.getenv("HTTPS_PROXY")  # Optional HTTPS proxy

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Configure proxy settings if provided
if HTTP_PROXY or HTTPS_PROXY:
    os.environ['HTTP_PROXY'] = HTTP_PROXY if HTTP_PROXY else ''
    os.environ['HTTPS_PROXY'] = HTTPS_PROXY if HTTPS_PROXY else ''

def check_internet_connection():
    """Check if we can connect to OpenAI's API"""
    import socket
    try:
        # Try to connect to OpenAI's API
        socket.create_connection(("api.openai.com", 443), timeout=5)
        return True
    except OSError:
        return False

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > 1000 and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    except Exception as e:
        chunks = []
        for i in range(0, len(text), 1000):
            chunk = text[i:i + 1000]
            if chunk:
                last_period = chunk.rfind('.')
                last_space = chunk.rfind(' ')
                break_point = last_period if last_period != -1 else last_space if last_space != -1 else len(chunk)
                chunks.append(chunk[:break_point + 1])
    
    return chunks


class TfidfEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.is_fit = False
        
    def embed_documents(self, texts):
        self.vectors = self.vectorizer.fit_transform(texts).toarray()
        self.is_fit = True
        return self.vectors.tolist()
        
    def embed_query(self, text):
        if not self.is_fit:
            raise ValueError("Must call embed_documents first")
        vector = self.vectorizer.transform([text]).toarray()
        return vector.tolist()[0]
    
    def __call__(self, text):
        # Make it work both for single texts and lists
        if isinstance(text, list):
            if not self.is_fit:
                return self.embed_documents(text)
            return [self.embed_query(t) for t in text]
        return self.embed_query(text)

def get_vector_store(text_chunks):
    try:
        # Create local search instance
        local_search = AdvancedLocalSearch()
        # Convert text chunks to document format
        from langchain.schema import Document
        from langchain.schema.retriever import BaseRetriever
        
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        local_search.add_documents(documents)
        
        class LocalRetriever(BaseRetriever):
            def __init__(self, local_search):
                super().__init__()
                self.local_search = local_search
                
            def get_relevant_documents(self, query):
                return self.local_search.search(query, k=3)
                
            def similarity_search(self, query, k=3):
                return self.local_search.search(query, k=k)
            
            async def aget_relevant_documents(self, query):
                return self.get_relevant_documents(query)
                
            def as_retriever(self, **kwargs):
                return self
            
        class LocalVectorStore:
            def __init__(self, retriever):
                self.retriever = retriever
                
            def as_retriever(self, **kwargs):
                return self.retriever
                
            def similarity_search(self, query, k=3):
                return self.retriever.similarity_search(query, k=k)
        
        retriever = LocalRetriever(local_search)
        return LocalVectorStore(retriever)
        
    except Exception as e:
        print(f"Error setting up local search: {str(e)}. Falling back to basic TF-IDF.")
        embeddings = TfidfEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store



def get_local_chain(vector_store, query):
    """Fallback to use local TF-IDF when OpenAI is not available"""
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Based on the available documents, here is the relevant information:\n\n{context}"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AdvancedLocalSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        self.doc_vectors = None
        self.documents = []
        
    def add_documents(self, documents):
        """Add documents to the search index"""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
    def search(self, query, k=3):
        """Search for most relevant documents"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_indices]

def format_local_response(docs, question):
    """Format response from local processing to be more readable"""
    from nltk.tokenize import sent_tokenize
    import re
    
    def get_context_window(text, keyword, window=2):
        """Get sentences around a keyword match"""
        sentences = sent_tokenize(text)
        for i, sent in enumerate(sentences):
            if keyword.lower() in sent.lower():
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                return " ".join(sentences[start:end])
        return None

    def summarize_content(text, max_length=100):
        """Create a brief summary of longer text"""
        sentences = sent_tokenize(text)
        if not sentences:
            return text
        if len(sentences[0]) <= max_length:
            return sentences[0]
        return sentences[0][:max_length] + "..."

    contents = []
    keywords = [w for w in question.lower().split() if len(w) > 3]  # Only use significant words
    
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        
        # Get relevant context around each keyword
        contexts = []
        for keyword in keywords:
            context = get_context_window(content, keyword)
            if context:
                contexts.append(context)
        
        if contexts:
            seen = set()
            unique_contexts = [x for x in contexts if not (x in seen or seen.add(x))]
            contents.append(f"\n📄 Document {i}:\n" + "\n".join(f"• {ctx.strip()}" for ctx in unique_contexts))
        elif content:  # If no keyword matches, include a summary
            summary = summarize_content(content)
            contents.append(f"\n📄 Document {i}:\n• {summary}")
    
    if not contents:
        return "I could not find any directly relevant information in the documents for your query."
    
    response = "� Found the following information:\n"
    response += "\n".join(contents)
    
    return response

def get_conversational_chain(vector_store):
    """Create a conversational chain using local search by default"""
    USE_OPENAI = False  # Set to False to use local search only
    
    def create_local_chain():
        return lambda x: {
            "chat_history": [
                {"role": "user", "content": x["question"]}, 
                {"role": "assistant", "content": format_local_response(
                    vector_store.similarity_search(x["question"], k=3),
                    x["question"]
                )}
            ]
        }
    
    if not USE_OPENAI:
        return create_local_chain()
        
    import time
    import random
    from openai import RateLimitError, APIError, APIConnectionError
    import httpx

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Configure API settings
            api_kwargs = {
                "openai_api_key": OPENAI_API_KEY,
                "temperature": 0,
                "model_name": "gpt-3.5-turbo",
                "request_timeout": 30,
            }
            
            if OPENAI_API_BASE:
                api_kwargs["base_url"] = OPENAI_API_BASE
            
            if HTTP_PROXY or HTTPS_PROXY:
                proxies = {
                    "http://": HTTP_PROXY if HTTP_PROXY else None,
                    "https://": HTTPS_PROXY if HTTPS_PROXY else None
                }
                api_kwargs["http_client"] = httpx.Client(proxies=proxies)
            
            llm = ChatOpenAI(**api_kwargs)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
            )
            return conversation_chain
            
        except APIError as e:
            if "insufficient_quota" in str(e):
                print("OpenAI API quota exceeded. Switching to local processing...")
                return lambda x: {"chat_history": [{"role": "user", "content": x["question"]}, 
                                                 {"role": "assistant", "content": format_local_response(
                                                     vector_store.similarity_search(x["question"], k=3),
                                                     x["question"]
                                                 )}]}
            if attempt == max_retries - 1:
                print(f"API error after {max_retries} attempts: {str(e)}. Falling back to local processing...")
                return lambda x: {"chat_history": [{"role": "user", "content": x["question"]}, 
                                                 {"role": "assistant", "content": format_local_response(
                                                     vector_store.similarity_search(x["question"], k=3),
                                                     x["question"]
                                                 )}]}
            wait = (2 ** attempt) + random.random()
            time.sleep(wait)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"API error after {max_retries} attempts: {str(e)}. Falling back to local processing...")
                return lambda x: {"chat_history": [{"role": "user", "content": x["question"]}, 
                                                 {"role": "assistant", "content": format_local_response(
                                                     vector_store.similarity_search(x["question"], k=3),
                                                     x["question"]
                                                 )}]}
            wait = (2 ** attempt) + random.random()
            time.sleep(wait)
        except (APIConnectionError, httpx.ConnectError) as e:
            print(f"Connection error: {str(e)}. Falling back to local processing...")
            return lambda x: {"chat_history": [{"role": "user", "content": x["question"]}, 
                                             {"role": "assistant", "content": format_local_response(
                                                 vector_store.similarity_search(x["question"], k=3),
                                                 x["question"]
                                             )}]}