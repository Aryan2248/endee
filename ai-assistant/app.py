import streamlit as st
import numpy as np
from pypdf import PdfReader
import ollama
import time
from endee_db import SimpleEndeeDB

st.set_page_config(
    page_title="Your Assistant | Endee AI",
    page_icon="🧠",
    layout="wide"
)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSecondaryBlock { background-color: #1c1f26; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #4CAF50; color: white; }
    .chunk-box { padding: 10px; border-left: 4px solid #4CAF50; background: #262730; margin-bottom: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

if "db" not in st.session_state:
    st.session_state.db = SimpleEndeeDB()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

def get_embedding(text):
    try:
        res = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return np.array(res["embedding"])
    except:
        return np.zeros(768)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def split_text(text, chunk_size=150):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Your Assistant")
    st.markdown("---")
    
    st.subheader("📁 Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF Documents", type="pdf", help="Upload notes like DSA, OS, or ML")

    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        with st.status("Vectorizing Knowledge...", expanded=True) as status:
            reader = PdfReader(uploaded_file)
            raw_text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
            chunks = split_text(raw_text)
            
            for chunk in chunks:
                emb = get_embedding(chunk)
                st.session_state.db.add(chunk, emb)
            
            st.session_state.processed_files.add(uploaded_file.name)
            status.update(label="Index Complete!", state="complete", expanded=False)
        st.success(f"Added {len(chunks)} vectors to Endee DB")

    st.markdown("---")
    st.info("💡 **Tip:** Ask specific questions about your uploaded documents for better RAG accuracy.")

# Main Dashboard

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    query = st.chat_input("Ask a question...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            query_emb = get_embedding(query)
            results = st.session_state.db.search(query_emb, cosine_similarity, top_k=2)

            if results:
                context = "\n---\n".join(results)
                response = ollama.chat(
                    model="llama3",
                    messages=[
                        {"role": "system", "content": "You are 'Your Assistant'. Answer based ONLY on the provided context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ]
                )
                st.write(response["message"]["content"])
                
                with st.expander("🔍 View Semantic Source"):
                    for r in results:
                        st.markdown(f'<div class="chunk-box">{r}</div>', unsafe_allow_html=True)
            else:
                st.write("I need some data first! Please upload a PDF in the sidebar.")

with col2:
    st.subheader("📊 System Stats")
    st.metric("Vectors in Endee DB", len(st.session_state.db.data))
    st.metric("Files Indexed", len(st.session_state.processed_files))
    
    st.divider()
    st.write("**Architecture:**")
    st.code("RAG: Retrieval-Augmented Generation\nDB: Vector-based Similarity Search\nLLM: Llama 3 (Ollama)")