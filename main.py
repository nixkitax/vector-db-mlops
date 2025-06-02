import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
import hashlib
import io

# Page configuration
st.set_page_config(
    page_title="Vector Database MLOps Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = []

class VectorMLOpsPipeline:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.dimension = 384
        
    @st.cache_resource
    def load_model(_self):
        """Load embedding model"""
        return SentenceTransformer(_self.model_name)
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        model = self.load_model()
        embeddings = model.encode(texts)
        return embeddings.astype('float32')
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index"""
        index = faiss.IndexFlatIP(self.dimension)  # Inner Product for similarity
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        index.add(embeddings)
        return index
    
    def search_similar(self, query: str, k: int = 5) -> Tuple[List[float], List[int]]:
        """Search for similar documents"""
        if st.session_state.faiss_index is None:
            return [], []
        
        start_time = time.time()
        
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = st.session_state.faiss_index.search(query_embedding, k)
        
        retrieval_time = time.time() - start_time
        self.log_query_metrics(query, retrieval_time, len(indices[0]))
        
        return scores[0].tolist(), indices[0].tolist()
    
    def log_query_metrics(self, query: str, retrieval_time: float, num_results: int):
        """Log query metrics"""
        metric = {
            'timestamp': datetime.now(),
            'query': query,
            'retrieval_time_ms': retrieval_time * 1000,
            'num_results': num_results,
            'query_length': len(query),
            'query_hash': hashlib.md5(query.encode()).hexdigest()[:8]
        }
        st.session_state.performance_metrics.append(metric)

# Initialize pipeline
pipeline = VectorMLOpsPipeline()

# Sidebar navigation
st.sidebar.title("🔍 Vector DB MLOps")
page = st.sidebar.selectbox(
    "Select Section",
    ["Dashboard", "Data Ingestion", "Semantic Search", "MLOps Monitoring"]
)

# DASHBOARD
if page == "Dashboard":
    st.title("Vector Database MLOps Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Indexed Documents", len(st.session_state.documents))
    
    with col2:
        st.metric("Total Queries", len(st.session_state.query_history))
    
    with col3:
        if st.session_state.performance_metrics:
            avg_time = np.mean([m['retrieval_time_ms'] for m in st.session_state.performance_metrics])
            st.metric("Average Time (ms)", f"{avg_time:.2f}")
        else:
            st.metric("Average Time (ms)", "N/A")
    
    with col4:
        index_status = "Active" if st.session_state.faiss_index else "Inactive"
        st.metric("Index Status", index_status)
    
    st.subheader("Current Model")

    a, b, c = st.columns(3)
    
    with a:
        st.info(f"**Model:** {pipeline.model_name}")
    with b:
        st.info(f"**Dimensionality:** {pipeline.dimension}")
    with c:
        st.info(f"**Index Status:** {'Active' if st.session_state.faiss_index else 'Inactive'}")
    
    if st.session_state.documents:
        st.subheader("Recently Added Documents")
        df_docs = pd.DataFrame([
            {
                'ID': doc['id'][:8],
                'Title': doc['content'][:50] + "...",
                'Timestamp': doc['timestamp'],
                'Size': len(doc['content'])
            }
            for doc in st.session_state.documents[-5:]
        ])
        st.dataframe(df_docs, use_container_width=True)

# DATA INGESTION
elif page == "Data Ingestion":
    st.title("Data Ingestion Pipeline")
    
    tab1, tab2 = st.tabs(["Upload Documents", "Batch Processing"])
    
    with tab1:
        st.subheader("Add Documents to Vector Database")
        
        # Input methods
        input_method = st.radio(
            "Input Method",
            ["Manual Text", "File Upload"]
        )
        
        if input_method == "Manual Text":
            title = st.text_input("Document Title")
            content = st.text_area("Content", height=200)
            
            if st.button("➕ Add Document", type="primary"):
                if content:
                    doc_id = hashlib.md5(content.encode()).hexdigest()
                    document = {
                        'id': doc_id,
                        'title': title or f"Doc_{len(st.session_state.documents)+1}",
                        'content': content,
                        'timestamp': datetime.now(),
                        'source': 'manual_input'
                    }
                    st.session_state.documents.append(document)
                    st.success(f"Document added! ID: {doc_id[:8]}")
                    st.rerun()
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                title = uploaded_file.name
                
                if st.button("Process File"):
                    doc_id = hashlib.md5(content.encode()).hexdigest()
                    document = {
                        'id': doc_id,
                        'title': title,
                        'content': content,
                        'timestamp': datetime.now(),
                        'source': 'file_upload'
                    }
                    st.session_state.documents.append(document)
                    st.success(f"File processed! ID: {doc_id[:8]}")
                    st.rerun()
    
    with tab2:
        st.subheader("FAISS Index Reconstruction")
        
        if st.session_state.documents:
            st.info(f"{len(st.session_state.documents)} documents ready for indexing")
            
            if st.button("🔧 Build FAISS Index", type="primary"):
                with st.spinner("🔄 Creating embeddings and FAISS index..."):
                    # Extract contents
                    contents = [doc['content'] for doc in st.session_state.documents]
                    
                    # Create embeddings
                    embeddings = pipeline.create_embeddings(contents)
                    
                    # Build FAISS index
                    st.session_state.faiss_index = pipeline.build_faiss_index(embeddings)
                    
                    # Save timestamp
                    st.session_state.index_built_at = datetime.now()
                
                st.success("FAISS index built successfully!")
                st.balloons()
        else:
            st.warning("Add documents before building the index")

# SEMANTIC SEARCH
elif page == "Semantic Search":
    st.title("Semantic Search Interface")
    
    if st.session_state.faiss_index is None:
        st.error("FAISS index not available. Go to Data Ingestion section to build it.")
    else:
        st.success("FAISS index active")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your semantic query",
                placeholder="Search for similar documents..."
            )
        
        with col2:
            num_results = st.selectbox("Number of results", [1, 3, 5, 10], index=1)
        
        if st.button("🔍 Search", type="primary") and query:
            with st.spinner("Searching..."):
                scores, indices = pipeline.search_similar(query, num_results)
                
                # Save query to history
                st.session_state.query_history.append({
                    'query': query,
                    'timestamp': datetime.now(),
                    'num_results': len(indices)
                })
            
            if indices:
                st.subheader("Search Results")
                
                for i, (idx, score) in enumerate(zip(indices, scores)):
                    if idx < len(st.session_state.documents):
                        doc = st.session_state.documents[idx]
                        
                        with st.expander(f"Result {i+1} - Score: {score:.4f}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Title:** {doc['title']}")
                                st.write(f"**Content:**")
                                st.write(doc['content'])
                            
                            with col2:
                                st.metric("Similarity Score", f"{score:.4f}")
                                st.write(f"**ID:** {doc['id'][:8]}")
                                st.write(f"**Date:** {doc['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning("No results found")
        
        if st.session_state.query_history:
            st.subheader("Recent Queries")
            recent_queries = st.session_state.query_history[-5:]
            for i, q in enumerate(reversed(recent_queries)):
                st.write(f"🔸 **{q['query']}** - {q['timestamp'].strftime('%H:%M:%S')} ({q['num_results']} results)")

# MLOPS MONITORING
elif page == "MLOps Monitoring":
    st.title("MLOps Monitoring Dashboard")
    
    if not st.session_state.performance_metrics:
        st.info("No metrics available. Execute some queries to generate data.")
    else:
        # General metrics
        st.subheader("General Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_queries = len(st.session_state.performance_metrics)
            st.metric("Total Queries", total_queries)
        
        with col2:
            avg_time = np.mean([m['retrieval_time_ms'] for m in st.session_state.performance_metrics])
            st.metric("Average Time (ms)", f"{avg_time:.2f}")
        
        with col3:
            max_time = max([m['retrieval_time_ms'] for m in st.session_state.performance_metrics])
            st.metric("Max Time (ms)", f"{max_time:.2f}")

# Sidebar information
st.sidebar.markdown("---")

st.sidebar.info("""
    **🔧 Technology Stack:**
    - FAISS for vector indexing and search
    - Sentence Transformers for semantic embedding generation
    - Streamlit for UI

    **MLOps Pipeline Stages:**
    - **Data Ingestion** → Document insertion via forms, files or batch ("Data Ingestion" page)
    - **Feature Engineering** → Document embedding via SentenceTransformer
    - **Model Deployment (Indexing)** → FAISS index creation in memory
    - **Inference** → Real-time semantic search via text queries
    - **Monitoring** → Query logging, response times and performance analysis ("MLOps Monitoring" page)
    - **Model Management** → Data reset, configuration export and state management ("Model Management" page)
    """)