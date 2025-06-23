import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import tempfile

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="ICEB Constitution Chatbot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global reset and base styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Force white background everywhere */
    .stApp {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Main container - pure white */
    .main .block-container {
        padding: 1.5rem 2rem;
        background: #ffffff !important;
        border-radius: 16px;
        margin: 1rem auto;
        max-width: 1200px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Override any dark backgrounds */
    .stApp > header {
        background: transparent !important;
    }
    
    /* Professional header design */
    .main-header {
        display: none;
    }
    
    .logo-container {
        display: none;
    }
    
    /* Header container styling */
    .header-container {
        background: #ffffff !important;
        padding: 2rem 0 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid #e2e8f0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    /* Force sidebar to be white */
    .css-1d391kg, .css-1d391kg .stMarkdown, .css-1d391kg > div {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Sidebar styling - force white */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Force all sidebar text to be dark */
    .css-1d391kg .stMarkdown, .css-1d391kg .stMarkdown h1, .css-1d391kg .stMarkdown h2, 
    .css-1d391kg .stMarkdown h3, .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown div,
    .css-1d391kg .stMarkdown span, .css-1d391kg .stMarkdown li {
        color: #1e293b !important;
        background: transparent !important;
    }
    
    /* Input styling - white background */
    .stTextArea > div > div > textarea {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px;
        font-size: 15px;
        padding: 1rem;
        transition: all 0.2s ease;
        color: #1e293b !important;
        line-height: 1.6;
        resize: vertical;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        outline: none;
        transform: translateY(-1px);
        background: #ffffff !important;
    }
    
    /* Professional button design */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        letter-spacing: 0.025em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.5) !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    }
    
    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
        box-shadow: 0 4px 12px rgba(100, 116, 139, 0.4) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #475569 0%, #334155 100%) !important;
        box-shadow: 0 8px 20px rgba(100, 116, 139, 0.5) !important;
    }
    
    /* Clean white answer box */
    .answer-box {
        background: #ffffff !important;
        color: #1e293b !important;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-size: 15px;
        line-height: 1.7;
        position: relative;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        border-radius: 16px 16px 0 0;
    }
    
    /* Clean info cards */
    .info-card {
        background: #ffffff !important;
        color: #1e293b !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .info-card h3 {
        color: #1e293b !important;
        margin-bottom: 1rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Sample question buttons - Clean design */
    .sample-question-btn > button {
        background: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
        width: 100% !important;
        text-align: left !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    .sample-question-btn > button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border-color: #3b82f6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }
    
    /* Fix all button styling - ensure clean white appearance */
    div[data-testid="column"] .stButton > button {
        background: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem 0 !important;
        width: 100% !important;
        text-align: left !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    div[data-testid="column"] .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border-color: #3b82f6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }
    
    /* Ensure primary buttons stay blue */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Selectbox styling - ensure white background */
    .stSelectbox > div > div > div {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Sidebar selectbox styling */
    section[data-testid="stSidebar"] .stSelectbox > div > div > div {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        color: #1e293b !important;
    }
    
    /* Sidebar button styling - ensure white */
    section[data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border-color: #3b82f6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }
    
    /* Ensure primary buttons in sidebar stay blue */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Remove old template button styling */
    .prompt-template-btn {
        display: none !important;
    }
    
    /* Professional sidebar header */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b !important;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        background: rgba(59, 130, 246, 0.1) !important;
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Clean success message */
    .success-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 500;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Question section styling */
    .question-section {
        background: #ffffff !important;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    .question-section h2 {
        color: #1e293b !important;
        margin-bottom: 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }
    
    /* Professional expandable sections */
    .stExpander {
        background: #ffffff !important;
        border-radius: 12px;
        border: 1px solid #e2e8f0 !important;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stExpander > div > div {
        background: #ffffff !important;
        color: #1e293b !important;
        border-radius: 12px;
    }
    
    /* Professional footer */
    .footer {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        color: #f8fafc !important;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 8px 20px rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer h3 {
        margin-bottom: 1rem !important;
        color: #f8fafc !important;
    }
    
    /* Professional alerts */
    .stAlert {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Force all text to be dark */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #1e293b !important;
        line-height: 1.6 !important;
    }
    
    /* Input labels */
    .stTextArea > label, .stSelectbox > label {
        color: #374151 !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    
    /* Override any remaining dark elements */
    div[data-testid="stMarkdownContainer"], 
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] div,
    div[data-testid="stMarkdownContainer"] span {
        color: #1e293b !important;
        background: transparent !important;
    }
    
    /* Force sidebar content to be visible */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown div,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1e293b !important;
        background: transparent !important;
    }
    
    /* Ensure expander text is visible */
    .streamlit-expanderHeader {
        color: #1e293b !important;
        background: #ffffff !important;
    }
    
    /* Override any Streamlit default dark styles */
    .css-1d391kg, .css-1d391kg div, .css-1d391kg p, .css-1d391kg span {
        color: #1e293b !important;
        background: #ffffff !important;
    }
    
    /* Clean section headers */
    h3 {
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        color: #1e293b !important;
    }
    
    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    }
    
    /* Professional spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Force white background on main content area */
    .main > div {
        background: #ffffff !important;
    }
    
    /* Remove any blue or dark backgrounds */
    div[style*="background-color: rgb(28, 131, 225)"],
    div[style*="background-color: #1c83e1"],
    div[style*="background: rgb(28, 131, 225)"],
    div[style*="background: #1c83e1"],
    div[style*="background-color: rgb(14, 17, 23)"],
    div[style*="background: rgb(14, 17, 23)"] {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Force ALL selectbox elements to be white - aggressive styling */
    .stSelectbox, .stSelectbox > div, .stSelectbox > div > div, 
    .stSelectbox > div > div > div, .stSelectbox > div > div > div > div {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Target the actual dropdown container */
    div[data-baseweb="select"] {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    
    /* Target dropdown options */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Target the dropdown menu when opened */
    div[role="listbox"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Target individual dropdown options */
    div[role="option"] {
        background: #ffffff !important;
        color: #1e293b !important;
        padding: 0.75rem 1rem !important;
    }
    
    div[role="option"]:hover {
        background: #f8fafc !important;
        color: #1e293b !important;
    }
    
    /* Force sidebar selectboxes to be white */
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div > div {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Override any dark theme selectbox styling */
    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        color: #1e293b !important;
    }
    
    /* Ensure selectbox text is visible */
    .stSelectbox label {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    /* SUPER AGGRESSIVE dropdown styling - override all Streamlit defaults */
    div[role="listbox"], div[role="listbox"] * {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Target the dropdown when it's opened */
    div[data-baseweb="popover"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    div[data-baseweb="popover"] * {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Force all select elements and their children to be white */
    [role="combobox"], [role="combobox"] *,
    [role="button"][aria-haspopup="listbox"], [role="button"][aria-haspopup="listbox"] * {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Override any remaining dark styling */
    .css-1wa3eu0-placeholder, .css-1uccc91-singleValue {
        color: #1e293b !important;
    }
    
    /* Target specific Streamlit select styling */
    .css-1d391kg .stSelectbox [role="combobox"] {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Code blocks and reference boxes - make them white */
    .stCodeBlock, .stCodeBlock > div, .stCodeBlock > div > div, .stCodeBlock pre {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Force all code elements to be white */
    code, pre, .stCodeBlock code, .stCodeBlock pre {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
    }
    
    /* Expander content styling */
    .streamlit-expanderContent {
        background: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None
    return text

def create_vector_store(text):
    """Create vector store from text"""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    # Create documents
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

def initialize_qa_chain(vector_store, custom_prompt=None):
    """Initialize the QA chain with custom prompt"""
    # Try to get API key from Streamlit secrets (for cloud deployment)
    try:
        openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
    except:
        # Fall back to environment variable (for local development)
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.error("⚠️ **API Key Missing:** OpenAI API key not found.")
        st.info("📋 **For Local Development:** Set your OPENAI_API_KEY in the .env file")
        st.info("☁️ **For Cloud Deployment:** Add your API key in the Streamlit Cloud secrets section")
        return None
    
    # Initialize OpenAI LLM
    llm = OpenAI(
        temperature=0.1,
        openai_api_key=openai_api_key
    )
    
    # Create custom prompt template
    if custom_prompt:
        prompt_template = f"""
        {custom_prompt}
        
        Context: {{context}}
        
        Question: {{question}}
        
        Answer: """
    else:
        prompt_template = """
        You are an AI assistant specialized in answering questions about the IQRA Constitution. 
        Please provide accurate, helpful, and detailed answers based solely on the provided context.
        If the question is not related to the IQRA Constitution or cannot be answered from the context, 
        politely redirect the user to ask questions specifically about the IQRA Constitution.
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

@st.cache_data
def load_and_process_pdf(pdf_path):
    """Load and process PDF - cached for performance"""
    text = extract_text_from_pdf(pdf_path)
    if text:
        return text
    return None

@st.cache_resource
def create_cached_vector_store(text):
    """Create vector store - cached for performance"""
    return create_vector_store(text)

def main():
    # Professional Header without Logo
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    # Create a centered layout for title only
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        # Title section only (logo removed)
        st.markdown('<div style="text-align: center; padding: 1rem 0;">', unsafe_allow_html=True)
        
        st.markdown('''
        <h1 style="
            font-size: 2.2rem;
            font-weight: 700;
            text-align: center;
            color: #1e293b;
            margin: 1rem 0 0.5rem 0;
            letter-spacing: -0.025em;
            line-height: 1.2;
        ">📖 ICEB Constitution Chatbot</h1>
        ''', unsafe_allow_html=True)
        
        # Decorative line
        st.markdown('''
        <div style="
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            border-radius: 2px;
            margin: 0 auto 1rem auto;
        "></div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar with clean professional design
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
        
        # Show current prompt status
        if 'custom_prompt' in st.session_state and st.session_state.custom_prompt:
            st.info("🎯 **Active:** Custom AI instructions are applied")
        else:
            st.info("📋 **Active:** Default AI instructions")
        
        # Custom Instructions Section
        st.markdown("### 🎯 Custom AI Instructions")
        st.markdown("*Customize how the AI responds to your questions:*")
        
        default_prompt = """You are an AI assistant specialized in answering questions about the IQRA Constitution. 
Please provide accurate, helpful, and detailed answers based solely on the provided context.
If the question is not related to the IQRA Constitution, politely redirect the user to ask questions specifically about the IQRA Constitution."""
        
        custom_prompt = st.text_area(
            "Enter your custom instructions:",
            value=default_prompt,
            height=150,
            help="Guide the AI's response style and focus. Be specific for best results."
        )
        
        # Apply prompt button
        if st.button("💾 Apply Custom Prompt", type="primary"):
            st.session_state.custom_prompt = custom_prompt
            st.session_state.qa_chain = None  # Force reinitialization
            # Clear any existing answers so user sees the change
            if 'current_answer' in st.session_state:
                del st.session_state.current_answer
            st.success("✅ Custom prompt applied successfully!")
            st.rerun()
        
        st.markdown("---")
        
        # Quick Templates Section
        st.markdown("### 📝 Quick Templates")
        
        prompt_templates = {
            "Select a template...": "",
            "📋 Strict IQRA Only": "You are an AI assistant that ONLY answers questions about the IQRA Constitution. If any question is not directly related to IQRA Constitution content, respond with: 'I can only answer questions about the IQRA Constitution. Please ask about IQRA's structure, policies, or governance.'",
            "📖 Detailed Explanations": "You are an expert on the IQRA Constitution. Provide comprehensive, detailed explanations with specific references to relevant sections. Always cite which part of the constitution your answer comes from.",
            "💡 Simple & Clear": "Answer questions about the IQRA Constitution in simple, clear language that anyone can understand. Avoid jargon and explain complex terms."
        }
        
        selected_template = st.selectbox(
            "Choose a response style:",
            list(prompt_templates.keys()),
            key="template_dropdown",
            help="Select how you want the AI to respond to questions"
        )
        
        # Handle template selection
        if selected_template and selected_template != "Select a template...":
            if st.button("✨ **Apply This Template**", key="apply_template", type="primary", use_container_width=True):
                st.session_state.custom_prompt = prompt_templates[selected_template]
                st.session_state.qa_chain = None  # Force reinitialization
                # Clear any existing answers so user sees the change
                if 'current_answer' in st.session_state:
                    del st.session_state.current_answer
                st.success(f"✅ Applied: {selected_template}")
                st.rerun()
        
        st.markdown("---")
        
        # Information Section
        st.markdown("### ℹ️ About This Tool")
        st.markdown("""
        **ICEB Constitution Chatbot** is an intelligent assistant designed to help you explore and understand the IQRA Constitution document.
        
        **Key Features:**
        - 🎯 Customizable AI responses
        - 📚 Context-aware answers
        - 🔍 Source reference tracking
        - 💡 Sample question suggestions
        """)
    
    # Check PDF availability
    pdf_path = "IQRAConstitution (1).pdf"
    
    if not os.path.exists(pdf_path):
        st.error("❌ **Error:** IQRA Constitution PDF not found. Please ensure 'IQRAConstitution (1).pdf' is in the current directory.")
        st.info("📋 **Instructions:** Place the PDF file in the same folder as this application.")
        return
    
    # Initialize document processing
    if 'vector_store' not in st.session_state:
        with st.spinner("🔄 **Processing Document** - Loading and analyzing IQRA Constitution..."):
            text = load_and_process_pdf(pdf_path)
            
            if text:
                st.session_state.vector_store = create_cached_vector_store(text)
                st.markdown('<div class="success-message">✅ IQRA Constitution loaded and ready for questions!</div>', unsafe_allow_html=True)
            else:
                st.error("❌ **Processing Error:** Failed to extract text from PDF. Please check the file format.")
                return
    
    # Initialize QA system
    if 'qa_chain' not in st.session_state or st.session_state.qa_chain is None:
        custom_prompt = st.session_state.get('custom_prompt', None)
        with st.spinner("🔧 **Initializing AI Assistant** - Setting up with your custom instructions..."):
            st.session_state.qa_chain = initialize_qa_chain(st.session_state.vector_store, custom_prompt)
        if custom_prompt:
            st.info("🎯 **AI Assistant Ready** - Using your custom instructions for responses.")
    
    # Main Content Layout
    col1, col2 = st.columns([2.2, 1], gap="large")
    
    with col1:
        # Question Input Section
        st.markdown('<div class="question-section">', unsafe_allow_html=True)
        st.markdown("## 💬 Ask Your Questions")
        st.markdown("*Enter any question about the IQRA Constitution below:*")
        
        question = st.text_area(
            "Your Question:",
            placeholder="Example: What are the main objectives of IQRA? What is the organizational structure? Who are the key stakeholders?",
            height=100,
            help="Ask specific questions about the constitution's content, structure, policies, or governance."
        )
        
        # Action Buttons
        col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])
        with col_btn1:
            ask_button = st.button("🔍 **Get Answer**", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", type="secondary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle clear button
        if clear_button:
            if 'current_answer' in st.session_state:
                del st.session_state.current_answer
            st.rerun()
        
        # Handle sample question selection FIRST
        if 'sample_question' in st.session_state:
            question = st.session_state.sample_question
            del st.session_state.sample_question
            # Set the answer to be displayed
            if st.session_state.qa_chain and question.strip():
                with st.spinner("🤔 **Processing sample question...**"):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": question})
                        st.session_state.current_answer = response
                        st.session_state.current_question = question
                    except Exception as e:
                        st.error(f"❌ **Error:** {str(e)}")
        
        # Process question from button click
        elif ask_button and question.strip():
            if st.session_state.qa_chain:
                with st.spinner("🤔 **Analyzing your question** - Searching through the constitution..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": question})
                        st.session_state.current_answer = response
                        st.session_state.current_question = question
                    except Exception as e:
                        st.error(f"❌ **Processing Error:** {str(e)}")
                        st.info("💡 **Tip:** Try rephrasing your question or check your OpenAI API key configuration.")
            else:
                st.error("❌ **System Error:** QA system not initialized. Please verify your OpenAI API key is set correctly.")
        elif ask_button and not question.strip():
            st.warning("⚠️ **Input Required:** Please enter a question to get started.")
        
        # Display answer immediately below question if available
        if 'current_answer' in st.session_state and st.session_state.current_answer:
            st.markdown("---")
            st.markdown("### 📝 **Answer**")
            st.markdown(f'<div class="answer-box">{st.session_state.current_answer["result"]}</div>', unsafe_allow_html=True)
            
            # Source references
            if st.session_state.current_answer.get('source_documents'):
                with st.expander("📚 **View Source References & Context**"):
                    st.markdown("*The following sections from the constitution were used to generate this answer:*")
                    for i, doc in enumerate(st.session_state.current_answer['source_documents']):
                        st.markdown(f"**📄 Reference {i+1}:**")
                        st.code(doc.page_content[:500] + ('...' if len(doc.page_content) > 500 else ''))
                        
                        if i < len(st.session_state.current_answer['source_documents']) - 1:
                            st.markdown("---")
    
    with col2:
        # Information Panel
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 **What You Can Ask About**")
        
        topics = [
            "📋 Organizational structure & hierarchy",
            "🎯 Mission, vision & objectives", 
            "📜 Policies & procedures",
            "👥 Membership requirements & rights",
            "⚖️ Governance & decision-making",
            "🏛️ Leadership roles & responsibilities",
            "📊 Committee structures & functions",
            "💼 Administrative processes"
        ]
        
        for topic in topics:
            st.markdown(f"• {topic}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample Questions
        st.markdown("### 💡 **Try These Sample Questions**")
        st.markdown("*Select a question from the dropdown below:*")
        
        sample_questions = [
            "Select a sample question...",
            "What are the main objectives of IQRA?",
            "Describe the organizational structure",
            "Who are the founding members?",
            "What are the membership requirements?",
            "How is the organization governed?",
            "What committees exist and their roles?",
            "What are the key policies?",
            "How are decisions made?"
        ]
        
        selected_question = st.selectbox(
            "Choose a question:",
            sample_questions,
            key="sample_question_dropdown",
            help="Select a sample question to automatically fill the question box"
        )
        
        # Handle sample question selection
        if selected_question and selected_question != "Select a sample question...":
            if st.button("🔍 **Use This Question**", key="use_sample_question", type="primary", use_container_width=True):
                st.session_state.sample_question = selected_question
                st.rerun()
    
    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>🚀 **Built with Advanced Technology**</h3>
        <p><strong>Streamlit</strong> • <strong>LangChain</strong> • <strong>OpenAI GPT</strong> • <strong>FAISS Vector Search</strong></p>
        <p><em>Professional AI-powered document analysis for the IQRA Constitution</em></p>
        <p>💡 **Features:** Custom AI Instructions • Intelligent Search • Source References • Modern Interface</p>
        <p style="margin-top: 1rem; font-size: 0.9em; opacity: 0.8;">
            Built for professional constitutional analysis and research
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 