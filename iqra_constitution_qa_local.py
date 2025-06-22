import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="IQRA Constitution Q&A Assistant (Local)",
    page_icon="📖",
    layout="wide"
)

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
    
    return vector_store, chunks

def get_relevant_chunks(question, vector_store, k=3):
    """Get relevant document chunks for a question"""
    results = vector_store.similarity_search(question, k=k)
    return [doc.page_content for doc in results]

def answer_question_local(question, relevant_chunks):
    """Answer question using local model"""
    try:
        # Use a lightweight local model for question answering
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad"
        )
        
        # Combine relevant chunks as context
        context = " ".join(relevant_chunks)
        
        # Limit context length to avoid model limits
        if len(context) > 3000:
            context = context[:3000]
        
        # Get answer
        result = qa_pipeline(question=question, context=context)
        
        return result['answer'], result['score']
    
    except Exception as e:
        st.error(f"Error with local model: {str(e)}")
        return None, 0

def simple_qa_fallback(question, relevant_chunks):
    """Simple fallback QA using text matching"""
    question_words = set(question.lower().split())
    
    best_chunk = ""
    best_score = 0
    
    for chunk in relevant_chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(question_words.intersection(chunk_words))
        score = overlap / len(question_words) if question_words else 0
        
        if score > best_score:
            best_score = score
            best_chunk = chunk
    
    if best_chunk:
        # Try to extract a relevant sentence
        sentences = best_chunk.split('.')
        best_sentence = ""
        best_sentence_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Avoid very short sentences
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
                score = overlap / len(question_words) if question_words else 0
                
                if score > best_sentence_score:
                    best_sentence_score = score
                    best_sentence = sentence.strip()
        
        if best_sentence:
            return best_sentence, best_sentence_score
        else:
            return best_chunk[:300] + "..." if len(best_chunk) > 300 else best_chunk, best_score
    
    return "I couldn't find a specific answer to your question in the IQRA Constitution.", 0

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
    st.title("📖 IQRA Constitution Q&A Assistant (Local Version)")
    st.markdown("*This version works entirely offline using local AI models*")
    st.markdown("---")
    
    # Check if PDF exists
    pdf_path = "IQRAConstitution (1).pdf"
    
    if not os.path.exists(pdf_path):
        st.error("❌ IQRA Constitution PDF not found. Please ensure 'IQRAConstitution (1).pdf' is in the current directory.")
        return
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        with st.spinner("🔄 Loading and processing IQRA Constitution (this may take a few minutes on first run)..."):
            # Extract text from PDF
            text = load_and_process_pdf(pdf_path)
            
            if text:
                # Create vector store
                vector_store, chunks = create_cached_vector_store(text)
                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                st.success("✅ IQRA Constitution loaded successfully!")
            else:
                st.error("❌ Failed to extract text from PDF.")
                return
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Questions About the IQRA Constitution")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the main objectives of IQRA? What is the organizational structure?",
            height=100
        )
        
        # QA Method selection
        qa_method = st.radio(
            "Choose QA method:",
            ["Local AI Model (DistilBERT)", "Simple Text Matching"],
            help="Local AI Model provides better answers but requires more processing time."
        )
        
        # Submit button
        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                with st.spinner("🤔 Thinking..."):
                    # Get relevant chunks
                    relevant_chunks = get_relevant_chunks(
                        question, 
                        st.session_state.vector_store, 
                        k=3
                    )
                    
                    if qa_method == "Local AI Model (DistilBERT)":
                        answer, confidence = answer_question_local(question, relevant_chunks)
                        
                        if answer:
                            # Display answer
                            st.subheader("📝 Answer:")
                            st.write(answer)
                            st.write(f"*Confidence: {confidence:.2f}*")
                        else:
                            # Fallback to simple method
                            st.warning("⚠️ Local AI model failed, using simple text matching...")
                            answer, confidence = simple_qa_fallback(question, relevant_chunks)
                            st.subheader("📝 Answer:")
                            st.write(answer)
                            st.write(f"*Match score: {confidence:.2f}*")
                    else:
                        answer, confidence = simple_qa_fallback(question, relevant_chunks)
                        st.subheader("📝 Answer:")
                        st.write(answer)
                        st.write(f"*Match score: {confidence:.2f}*")
                    
                    # Display source chunks
                    if relevant_chunks:
                        with st.expander("📚 Source References"):
                            for i, chunk in enumerate(relevant_chunks):
                                st.write(f"**Reference {i+1}:**")
                                st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                                st.write("---")
            else:
                st.warning("⚠️ Please enter a question.")
    
    with col2:
        st.subheader("ℹ️ Information")
        st.info("""
        This local version uses:
        - **HuggingFace Embeddings** for semantic search
        - **DistilBERT** for question answering
        - **No API keys required**
        - **Works completely offline**
        
        Perfect for privacy-sensitive environments!
        """)
        
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are the main objectives of IQRA?",
            "What is the organizational structure?", 
            "Who are the founding members?",
            "What are the membership requirements?",
            "How is the organization governed?",
            "What are the key policies?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                # Set the question in the text area (this is a simplified approach)
                st.session_state.sample_question = q
        
        # Performance info
        st.subheader("⚡ Performance Notes")
        st.info("""
        - First run downloads models (~500MB)
        - Local AI model: Better accuracy, slower
        - Text matching: Faster, simpler answers
        - All processing happens on your computer
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🔧 Built with Streamlit, HuggingFace Transformers, and FAISS")
    st.markdown("*No API keys or internet required after initial setup*")

if __name__ == "__main__":
    main() 