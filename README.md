<h1 align="center">🧠 CortexRAG – Intelligent Reasoning Chatbot with RAG</h1>

<p align="center">
  🔍 Contextual AI Chatbot &nbsp; | &nbsp; 🧠 Retrieval-Augmented Generation &nbsp; | &nbsp; 📚 Vector Database with FAISS
</p>

---

## 🚀 Project Overview

**CortexRAG** is an intelligent chatbot that utilizes **Retrieval-Augmented Generation (RAG)** to provide **context-aware**, real-time answers to user queries.

By combining **vector-based search** with **large language models (LLMs)**, the system enhances the relevance and depth of generated responses — ideal for **knowledge-based Q&A**, **digital assistants**, and **automated support**.

---

## 🔍 Key Features

- 💬 Conversational AI powered by LLMs  
- 📄 Contextual retrieval from custom knowledge base  
- ⚡ Fast, accurate responses via vector similarity search  
- 🧠 Natural reasoning with LangChain & LangGraph  
- 📺 Deployed via interactive Streamlit interface

---

## 🧰 Tech Stack

| Technology    | Description                           |
|---------------|---------------------------------------|
| 🐍 Python      | Primary language                      |
| 🔗 LangChain   | LLM orchestration & prompt chaining   |
| 🧠 LangGraph   | Reasoning logic + memory modeling     |
| 📦 FAISS       | Vector similarity search              |
| 🌐 Streamlit   | User interface for real-time queries  |

---

## 📂 Project Structure

├── frontent.py # Streamlit app UI
├── rag_pipeline.py # Core RAG logic pipeline
├── vector_database.py # FAISS database initialization
├── vectorstore/db_faiss # Stored embeddings
├── requirements.txt # Dependencies


---

## ⚙️ How It Works

1. **User submits a question**
2. 🧠 **FAISS** retrieves relevant document chunks based on embeddings
3. 🔗 **LangChain + LangGraph** generate coherent answers
4. 💡 Chatbot responds with human-like clarity, grounded in facts

---
Preview : https://www.linkedin.com/posts/om-jade_theaianddatastreak100-ai-rag-activity-7315377134183534592-ZYjI?utm_source=share&utm_medium=member_desktop&rcm=ACoAADagc6EBPi0pG9PwngFn40e8NkTOqF3QhdM

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Omjade/CortexRAG-Intelligent-Reasoning-Chatbot-with-Retrieval-Augmented-Generation.git
cd CortexRAG-Intelligent-Reasoning-Chatbot-with-Retrieval-Augmented-Generation

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run frontent.py

⚠️ You’ll need OpenAI or Gemini API key and sample documents for embeddings.

💡 Use Cases
📚 Educational Q&A bots

📖 Knowledge base assistants

📑 Document-specific search interfaces

🤖 AI chat agents with internal memory

🔗 Access & Deployment
🧠 GitHub Repo: CortexRAG GitHub

👨‍💻 Author
Om Jade
🔗 LinkedIn
🌐 Portfolio
📧 omjade2854@gmail.com

<p align="center"> <em>“Intelligence is not in remembering everything, but knowing where to look.”</em> </p> ```
