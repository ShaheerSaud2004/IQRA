import streamlit as st
import PyPDF2
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="IQRA Constitution Q&A Assistant (Local - Fixed)",
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

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk.strip()) > 50:  # Only add meaningful chunks
            chunks.append(chunk.strip())
        
        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks

def create_vector_index(chunks, model_name="all-MiniLM-L6-v2"):
    """Create FAISS vector index from text chunks"""
    # Load embedding model
    embedding_model = SentenceTransformer(model_name)
    
    # Create embeddings
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    return index, embedding_model, chunks

def search_similar_chunks(query, index, embedding_model, chunks, k=3):
    """Search for similar chunks using FAISS"""
    # Encode query
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding.astype('float32'), k)
    
    # Return relevant chunks
    relevant_chunks = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            relevant_chunks.append({
                'text': chunks[idx],
                'score': float(scores[0][i])
            })
    
    return relevant_chunks

def answer_question_with_distilbert(question, relevant_chunks):
    """Answer question using DistilBERT"""
    try:
        # Initialize QA pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad"
        )
        
        # Combine relevant chunks as context
        context_parts = [chunk['text'] for chunk in relevant_chunks]
        context = " ".join(context_parts)
        
        # Limit context length to avoid model limits
        if len(context) > 4000:
            context = context[:4000]
        
        # Get answer
        result = qa_pipeline(question=question, context=context)
        
        return result['answer'], result['score']
    
    except Exception as e:
        st.error(f"Error with DistilBERT model: {str(e)}")
        return None, 0

def simple_keyword_search(question, relevant_chunks):
    """Simple keyword-based answer extraction"""
    question_words = set(question.lower().split())
    
    best_chunk = None
    best_score = 0
    best_sentence = ""
    
    for chunk_data in relevant_chunks:
        chunk = chunk_data['text']
        
        # Score sentences within the chunk
        sentences = chunk.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 30:  # Meaningful sentences only
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
                score = overlap / len(question_words) if question_words else 0
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
                    best_chunk = chunk
    
    if best_sentence:
        return best_sentence, best_score
    elif relevant_chunks:
        # Return first chunk if no good sentence found
        chunk = relevant_chunks[0]['text']
        return chunk[:300] + "..." if len(chunk) > 300 else chunk, 0.1
    else:
        return "I couldn't find relevant information in the IQRA Constitution.", 0

@st.cache_data
def load_pdf_text(pdf_path):
    """Load PDF text with caching"""
    return extract_text_from_pdf(pdf_path)

@st.cache_resource
def build_search_index(text):
    """Build search index with caching"""
    chunks = split_text_into_chunks(text)
    index, model, chunks = create_vector_index(chunks)
    return index, model, chunks

def main():
    st.title("📖 IQRA Constitution Q&A Assistant (Local - Fixed)")
    st.markdown("*Dependency-free local version using pure HuggingFace models*")
    st.markdown("---")
    
    # Check if PDF exists
    pdf_path = "IQRAConstitution (1).pdf"
    
    if not os.path.exists(pdf_path):
        st.error("❌ IQRA Constitution PDF not found. Please ensure 'IQRAConstitution (1).pdf' is in the current directory.")
        return
    
    # Initialize session state
    if 'search_ready' not in st.session_state:
        with st.spinner("🔄 Processing IQRA Constitution (downloading models on first run - may take 2-3 minutes)..."):
            try:
                # Load PDF
                text = load_pdf_text(pdf_path)
                
                if text:
                    # Build search index
                    index, embedding_model, chunks = build_search_index(text)
                    
                    # Store in session state
                    st.session_state.index = index
                    st.session_state.embedding_model = embedding_model
                    st.session_state.chunks = chunks
                    st.session_state.search_ready = True
                    
                    st.success("✅ IQRA Constitution processed successfully!")
                    st.info(f"📊 Created {len(chunks)} searchable text chunks")
                else:
                    st.error("❌ Failed to extract text from PDF.")
                    return
            except Exception as e:
                st.error(f"❌ Error during setup: {str(e)}")
                return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Questions About the IQRA Constitution")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the main objectives of IQRA Community Services?",
            height=100
        )
        
        # Method selection
        method = st.radio(
            "Choose answer method:",
            ["🤖 DistilBERT AI Model", "🔍 Keyword Search"],
            help="AI Model provides better answers but is slower. Keyword Search is faster but simpler."
        )
        
        # Search button
        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                with st.spinner("🤔 Searching and analyzing..."):
                    try:
                        # Search for relevant chunks
                        relevant_chunks = search_similar_chunks(
                            question, 
                            st.session_state.index,
                            st.session_state.embedding_model,
                            st.session_state.chunks,
                            k=3
                        )
                        
                        if relevant_chunks:
                            if method == "🤖 DistilBERT AI Model":
                                answer, confidence = answer_question_with_distilbert(question, relevant_chunks)
                                
                                if answer:
                                    st.subheader("📝 Answer:")
                                    st.write(answer)
                                    st.write(f"*Confidence: {confidence:.3f}*")
                                else:
                                    # Fallback to keyword search
                                    st.warning("⚠️ AI model failed, using keyword search...")
                                    answer, score = simple_keyword_search(question, relevant_chunks)
                                    st.subheader("📝 Answer:")
                                    st.write(answer)
                                    st.write(f"*Keyword match score: {score:.3f}*")
                            else:
                                answer, score = simple_keyword_search(question, relevant_chunks)
                                st.subheader("📝 Answer:")
                                st.write(answer)
                                st.write(f"*Keyword match score: {score:.3f}*")
                            
                            # Show source references
                            st.subheader("📚 Source References:")
                            for i, chunk_data in enumerate(relevant_chunks):
                                with st.expander(f"Reference {i+1} (Similarity: {chunk_data['score']:.3f})"):
                                    st.write(chunk_data['text'][:800] + "..." if len(chunk_data['text']) > 800 else chunk_data['text'])
                        else:
                            st.warning("⚠️ No relevant information found.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during search: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question.")
    
    with col2:
        st.subheader("ℹ️ System Information")
        st.info("""
        **This version uses:**
        - SentenceTransformers for embeddings
        - FAISS for vector search  
        - DistilBERT for Q&A
        - No LangChain dependencies
        - 100% local processing
        """)
        
        if st.session_state.get('search_ready'):
            st.success("✅ System Ready")
            st.metric("Text Chunks", len(st.session_state.chunks))
        
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are the main objectives of IQRA?",
            "What is the organizational structure?",
            "Who are the founding members?",
            "What are the membership requirements?",
            "How is the organization governed?",
            "What committees exist?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.sample_question = q
                st.rerun()
        
        # Handle sample question selection
        if 'sample_question' in st.session_state:
            question = st.session_state.sample_question
            del st.session_state.sample_question
        
        st.subheader("⚡ Performance")
        st.info("""
        **First Run:**
        - Downloads models (~400MB)
        - Takes 2-3 minutes setup
        
        **Subsequent Runs:**
        - Uses cached models
        - Fast startup
        - Instant search
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🔧 **Built with:** Streamlit + HuggingFace + FAISS")
    st.markdown("*No external APIs • Complete privacy • Offline capable*")

if __name__ == "__main__":
    main() 