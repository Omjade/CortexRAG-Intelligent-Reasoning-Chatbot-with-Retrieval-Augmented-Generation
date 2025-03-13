import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# ✅ Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Define the embedding model
ollama_model_name = "deepseek-r1:1.5b"
embeddings = OllamaEmbeddings(model=ollama_model_name)

# ✅ Load FAISS with embeddings
FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.load_local(FAISS_DB_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

# ✅ Setup LLM model
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

# ✅ Retrieve documents
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# ✅ Custom prompt template
custom_prompt_template = """
Use the provided context to answer the question.
If you don't know the answer, say you don't know. Do not make up an answer.

Question: {question}
Context: {context}

Answer:
"""

# ✅ Answer query
def answer_query(documents, query, model):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context})
    return response
