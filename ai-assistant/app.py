import streamlit as st
import numpy as np
from pypdf import PdfReader
import ollama
from concurrent.futures import ThreadPoolExecutor
from endee_db import SimpleEndeeDB

st.set_page_config(
    page_title="Endee AI · Knowledge Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── Base background: deep navy-slate, not pure black ── */
html, body, .main, [data-testid="stAppViewContainer"] {
    background-color: #0d1117 !important;
    color: #cdd5e0;
    font-family: 'Syne', sans-serif;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131929 0%, #0f1520 100%) !important;
    border-right: 1px solid #1f2d42 !important;
    min-width: 280px !important;
    max-width: 320px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── Brand block ── */
.brand-block {
    padding: 28px 24px 22px;
    background: linear-gradient(135deg, #161f32 0%, #1a2540 100%);
    border-bottom: 1px solid #1f2d42;
    position: relative;
    overflow: hidden;
}
.brand-block::after {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 110px; height: 110px;
    background: radial-gradient(circle, rgba(124,106,247,0.10) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.brand-name {
    font-size: 22px;
    font-weight: 800;
    color: #e8edf5;
    letter-spacing: -0.4px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.live-dot {
    width: 9px; height: 9px;
    background: #8b7cf8;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(139,124,248,0.6);
    animation: blink 2.8s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.25; }
}
.brand-sub {
    font-size: 11px;
    color: #4a5d78;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
    font-weight: 400;
}

/* ── Section wrapper ── */
.sb-section {
    padding: 20px 24px 0;
}
.sb-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a5d78;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
}
.sb-label::before {
    content: '';
    display: inline-block;
    width: 18px; height: 1px;
    background: #2a3d55;
}

/* ── File uploader override ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 24px !important;
}
[data-testid="stFileUploader"] > div {
    background: #131e30 !important;
    border: 1.5px dashed #253448 !important;
    border-radius: 14px !important;
    padding: 20px 18px !important;
    transition: border-color 0.2s, background 0.2s !important;
    text-align: center;
}
[data-testid="stFileUploader"] > div:hover {
    background: #162035 !important;
    border-color: #8b7cf8 !important;
}
[data-testid="stFileUploader"] label {
    color: #7a8fa8 !important;
    font-size: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] small {
    color: #3a4f68 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}
[data-testid="stFileUploader"] button {
    background: #1c2d45 !important;
    border: 1px solid #2d4260 !important;
    color: #8b7cf8 !important;
    border-radius: 9px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 7px 18px !important;
    transition: all 0.18s !important;
    margin-top: 8px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #223554 !important;
    border-color: #8b7cf8 !important;
    box-shadow: 0 0 14px rgba(139,124,248,0.18) !important;
}

/* ── Thin divider ── */
.sb-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1f2d42 30%, #1f2d42 70%, transparent);
    margin: 20px 0;
}

/* ── Stat cards ── */
.stat-row {
    display: flex;
    gap: 10px;
    padding: 0 24px;
}
.stat-card {
    flex: 1;
    background: linear-gradient(135deg, #131e30 0%, #162035 100%);
    border: 1px solid #1f2d42;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(139,124,248,0.35), transparent);
}
.stat-value {
    font-size: 24px;
    font-weight: 700;
    color: #dde4f0;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3a5068;
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── System info table ── */
.sys-table {
    margin: 0 24px;
    background: #111b2b;
    border: 1px solid #1f2d42;
    border-radius: 13px;
    overflow: hidden;
}
.sys-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 16px;
    border-bottom: 1px solid #1a2638;
}
.sys-row:last-child { border-bottom: none; }
.sys-key {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3a5068;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}
.sys-val {
    font-size: 12px;
    color: #5c7a96;
    font-family: 'JetBrains Mono', monospace;
}
.sys-val.hi { color: #8b7cf8; }

/* ── Tips ── */
.tip-list {
    padding: 4px 24px 24px;
}
.tip-item {
    font-size: 12.5px;
    color: #3a5068;
    font-family: 'JetBrains Mono', monospace;
    line-height: 2.2;
    display: flex;
    align-items: center;
    gap: 9px;
}
.tip-item::before {
    content: '→';
    color: #2a4060;
    font-size: 11px;
    flex-shrink: 0;
}


/* Buttons */
.stButton button {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    border-radius: 9px !important;
    padding: 8px 16px !important;
    transition: all 0.18s ease !important;
}
.stButton button[kind="secondary"] {
    background: #111b2b !important;
    border: 1px solid #1f2d42 !important;
    color: #4a6080 !important;
}
.stButton button[kind="secondary"]:hover {
    background: #162035 !important;
    border-color: #8b7cf8 !important;
    color: #cdd5e0 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35) !important;
}

/* Page heading */
.page-title {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3a5068;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 5px;
}
.page-heading {
    font-size: 28px;
    font-weight: 800;
    color: #e8edf5;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 20px;
}
.page-heading span { color: #8b7cf8; }

/* Status badge */
.sys-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #111b2b;
    border: 1px solid #1f2d42;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: #4a6080;
    margin-bottom: 22px;
}



[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}
[data-testid="stChatMessage"] > div:last-child {
    width: 100% !important;
    padding: 0 !important;
}

.msg-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin: 18px 0;
}
.msg-row.user-row { flex-direction: row-reverse; }

.avatar {
    width: 38px; height: 38px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.02em;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}
.avatar-user {
    background: linear-gradient(135deg, #112340, #163060);
    border: 1px solid #1e3a5f;
    color: #5a90c0;
}
.avatar-ai {
    background: linear-gradient(135deg, #170f2e, #1f1545);
    border: 1px solid #2d1f55;
    color: #8b7cf8;
}

.bubble {
    max-width: 76%;
    padding: 15px 20px;
    border-radius: 16px;
    font-size: 15.5px;
    line-height: 1.8;
    font-family: 'Syne', sans-serif;
    font-weight: 400;
}
.bubble-user {
    background: linear-gradient(135deg, #0f1e30 0%, #132640 100%);
    border: 1px solid #1e3450;
    color: #a8c4de;
    border-top-right-radius: 4px;
}
.bubble-ai {
    background: linear-gradient(135deg, #130e22 0%, #1a1235 100%);
    border: 1px solid #26184a;
    color: #cdc5f2;
    border-top-left-radius: 4px;
}
.stream-bubble {
    background: linear-gradient(135deg, #130e22 0%, #1a1235 100%);
    border: 1px solid #26184a;
    color: #cdc5f2;
    border-radius: 16px;
    border-top-left-radius: 4px;
    padding: 15px 20px;
    font-size: 15.5px;
    line-height: 1.8;
    font-family: 'Syne', sans-serif;
    max-width: 76%;
}

.msg-label {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
    opacity: 0.4;
    font-weight: 500;
}
.user-row .msg-label { text-align: right; color: #5a90c0; }
.msg-label.ai-label { color: #8b7cf8; }

/* Chat input */
[data-testid="stChatInput"] {
    background: #111b2b !important;
    border: 1px solid #1f2d42 !important;
    border-radius: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    color: #cdd5e0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #8b7cf8 !important;
    box-shadow: 0 0 0 3px rgba(139,124,248,0.08) !important;
}

/* Source chunks */
.chunk-box {
    background: #111b2b;
    border-left: 3px solid #8b7cf8;
    border-radius: 10px;
    padding: 15px 20px;
    margin: 8px 0;
    font-size: 13.5px;
    font-family: 'JetBrains Mono', monospace;
    color: #4a6080;
    line-height: 1.85;
}
[data-testid="stExpander"] {
    background: #111b2b !important;
    border: 1px solid #1f2d42 !important;
    border-radius: 11px !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: #3a5068 !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Empty state */
.empty-state { text-align: center; padding: 70px 20px; }
.empty-icon { font-size: 34px; margin-bottom: 16px; opacity: 0.2; }
.empty-text {
    font-size: 14.5px;
    font-family: 'JetBrains Mono', monospace;
    line-height: 2.1;
    color: #2a3d55;
}

/* Misc */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1f2d42; border-radius: 4px; }

.main p, .main li { font-size: 15.5px !important; line-height: 1.8 !important; }

</style>
""", unsafe_allow_html=True)


if "db" not in st.session_state:
    st.session_state.db = SimpleEndeeDB()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None


@st.cache_data(show_spinner=False)
def get_embedding(text):
    try:
        res = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return np.array(res["embedding"], dtype=np.float32)
    except:
        return np.zeros(768, dtype=np.float32)

def split_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def index_chunks_parallel(chunks):
    with ThreadPoolExecutor(max_workers=6) as executor:
        embeddings = list(executor.map(get_embedding, chunks))
    for chunk, emb in zip(chunks, embeddings):
        st.session_state.db.add(chunk, emb)

def render_message(role, content):
    if role == "user":
        st.markdown(f"""
        <div class="msg-row user-row">
            <div>
                <div class="bubble bubble-user">{content}</div>
                <div class="msg-label">You</div>
            </div>
            <div class="avatar avatar-user">YOU</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar avatar-ai">AI</div>
            <div>
                <div class="bubble bubble-ai">{content}</div>
                <div class="msg-label ai-label">Endee AI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


with st.sidebar:

    st.markdown("""
    <div class="brand-block">
        <div class="brand-name">
            <span class="live-dot"></span>
            Endee AI
        </div>
        <div class="brand-sub">Knowledge Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Knowledge Base</div></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        with st.status("⚙ Indexing document..."):
            reader = PdfReader(uploaded_file)
            raw_text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
            chunks = split_text(raw_text)
            st.write(f"Embedding {len(chunks)} chunks in parallel…")
            index_chunks_parallel(chunks)
            st.session_state.processed_files.add(uploaded_file.name)
        st.success(f"✓ {len(chunks)} chunks indexed — {uploaded_file.name}")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.db.data)}</div>
            <div class="stat-label">Vectors</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.processed_files)}</div>
            <div class="stat-label">Docs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.messages)}</div>
            <div class="stat-label">Msgs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sys-table">
        <div class="sys-row">
            <span class="sys-key">Model</span>
            <span class="sys-val hi">llama3</span>
        </div>
        <div class="sys-row">
            <span class="sys-key">Embeddings</span>
            <span class="sys-val">nomic-embed-text</span>
        </div>
        <div class="sys-row">
            <span class="sys-key">Vector Store</span>
            <span class="sys-val">NumPy Matrix</span>
        </div>
        <div class="sys-row">
            <span class="sys-key">Mode</span>
            <span class="sys-val hi">RAG · Streaming</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-list">
        <div class="tip-item">Upload a PDF to index it</div>
        <div class="tip-item">Ask questions naturally</div>
        <div class="tip-item">Answers stream in real-time</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="page-title">◈ Endee AI · Chat Interface</div>', unsafe_allow_html=True)
st.markdown('<div class="page-heading">Ask your <span>documents</span> anything.</div>', unsafe_allow_html=True)

status_text = "System Online" if len(st.session_state.db.data) > 0 else "Awaiting Document"
dot_color = "#8b7cf8" if len(st.session_state.db.data) > 0 else "#3a5068"
st.markdown(f"""
<div class="sys-badge">
    <span style="width:7px;height:7px;background:{dot_color};border-radius:50%;
    display:inline-block;box-shadow:0 0 7px {dot_color}99;flex-shrink:0;"></span>
    {status_text} &nbsp;·&nbsp; {len(st.session_state.db.data)} vectors indexed
</div>
""", unsafe_allow_html=True)

# Action bar
btn_col1, btn_col2, spacer = st.columns([0.11, 0.13, 0.76])
with btn_col1:
    clear = st.button("🗑 Clear", type="secondary", use_container_width=True)
with btn_col2:
    regenerate = st.button("↺ Redo", type="secondary", use_container_width=True)

if clear:
    st.session_state.messages = []
    st.session_state.last_query = None
    st.rerun()

# Chat history
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">◈</div>
        <div class="empty-text">
            Upload a PDF from the sidebar.<br>
            Then ask anything — answers stream instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"])

query = st.chat_input("Ask a question from your documents…")
if regenerate and st.session_state.last_query:
    query = st.session_state.last_query

if query:
    st.session_state.last_query = query
    render_message("user", query)
    if not regenerate:
        st.session_state.messages.append({"role": "user", "content": query})

    query_emb = get_embedding(query)
    results = st.session_state.db.search(query_emb, top_k=2)

    if results:
        context = "\n---\n".join(results)

        st.markdown("""
        <div class="msg-row">
            <div class="avatar avatar-ai">AI</div>
            <div style="max-width:76%;">
        """, unsafe_allow_html=True)

        response_placeholder = st.empty()
        full_response = ""

        stream = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a precise AI assistant. Answer ONLY using the provided context. Be concise and professional."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            stream=True
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            response_placeholder.markdown(
                f'<div class="stream-bubble">{full_response}▌</div>',
                unsafe_allow_html=True
            )

        response_placeholder.markdown(
            f'<div class="stream-bubble">{full_response}</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
            <div class="msg-label ai-label">Endee AI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        with st.expander("▸ View Source Chunks"):
            for i, r in enumerate(results, 1):
                st.markdown(
                    f'<div class="chunk-box"><strong style="color:#8b7cf8;font-size:11px;letter-spacing:0.08em;">CHUNK {i}</strong><br><br>{r}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning("⚠ No indexed documents found. Please upload a PDF from the sidebar first.")