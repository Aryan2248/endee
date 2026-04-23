Your Assistant: Local RAG Knowledge Base 🤖
Built specifically for the Endee.io Recruitment Challenge (B.Tech 2027).

Your Assistant is a high-performance, privacy-focused AI Knowledge Assistant. It allows users to upload complex academic documents (PDFs) and perform semantic queries using a custom RAG (Retrieval-Augmented Generation) pipeline powered by Endee-style Vector Storage and Ollama.

🌟 Key Features
Semantic Search: Unlike keyword search, this understands the context of your notes using 768-dimensional vector embeddings.

Privacy-First: 100% local execution. No data ever leaves the machine.

Smart Chunking: Implements recursive text splitting to maintain context within local LLM constraints.

Vector Stats: Real-time monitoring of indexed vectors in the Endee database.

🏗️ System Architecture
The system follows a production-grade RAG workflow:

Data Ingestion: Extracts raw text from PDFs using PyPDF.

Recursive Chunking: Breaks text into 150-word segments to optimize for nomic-embed-text context windows.

Vectorization: Converts segments into mathematical vectors using the Ollama embedding API.

Endee Vector Storage: Stores high-dimensional vectors in a custom SimpleEndeeDB class for low-latency retrieval.

Similarity Search: Uses Cosine Similarity to find the top-k most relevant segments for a user query.

LLM Synthesis: Injects retrieved context into a system prompt for Llama 3 to generate a precise answer.

🛠️ Tech Stack
AI Engine: Ollama (Llama 3 & Nomic-Embed-Text)

Vector Logic: Python (Inspired by Endee.io architecture)

Frontend: Streamlit (Professional Dashboard)

Mathematical Processing: NumPy (Vector similarity calculations)

🚀 Setup & Installation
1. Prerequisites
Ensure you have Ollama installed on your system. Before running the app, pull the necessary models:

Bash
ollama pull llama3
ollama pull nomic-embed-text
2. Installation
Navigate to your project directory and install the required Python libraries:

Bash
# Recommended: Create and activate a virtual environment
# python -m venv venv
# source venv/bin/activate (Mac/Linux) or venv\Scripts\activate (Windows)

pip install -r requirements.txt
3. Running the App
Launch the Streamlit interface with the following command:

Bash
streamlit run app.py
The app will automatically open in your default browser at http://localhost:8501.

<details>
<summary><b>🛠️ Technical Implementation Details (Deep Dive)</b></summary>

1. Vector Mathematical Logic
The system uses Cosine Similarity to measure the distance between the query vector and document vectors.

A score of 1.0 means an exact semantic match.

High-dimensional vectors (768D) are managed through a custom index optimized for the Endee recruitment use-case.

2. Prompt Engineering
The LLM is governed by a strict system prompt:

"You are 'Your Assistant'. Answer based ONLY on the provided context. If the answer is not in the context, say you do not know."
This prevents "AI Hallucination" and ensures academic integrity.

3. Optimized Chunking
I implemented a 150-word chunk limit specifically to handle the status code: 500 context limits found in local inference models, ensuring system stability even with dense PDF content.

</details>

<details>
<summary><b>💡 Future Roadmap</b></summary>

[ ] Integration with Endee's Cloud API for persistent cloud-based storage.

[ ] Support for multiple file formats (.docx, .txt).

[ ] Multi-document cross-referencing capabilities.

</details>

👨‍💻 Author
Aryan Yadav CSE-1 | B.Tech (CSE) 2027

Specialization: AI & Full-Stack Development