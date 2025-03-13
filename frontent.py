import streamlit as st
import re
from vector_database import process_and_store_pdf  # Handles PDF processing
from rag_pipeline import answer_query, retrieve_docs, llm_model

# Step 1: Upload PDF dynamically
st.title("📄 AI Reasoning Chatbot")
uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    st.success(f"✅ Uploaded: `{uploaded_file.name}`")

    # Step 2: Process and store the new PDF in FAISS
    with st.spinner("⚙️ Processing PDF..."):
        faiss_db = process_and_store_pdf(uploaded_file)

    st.success("✅ PDF processed and indexed successfully!")

# Step 3: Chatbot interface
user_query = st.text_area("💬 Enter your question:", height=150, placeholder="Ask anything!")

if st.button("🔍 Ask AI"):
    if uploaded_file and user_query.strip():
        st.chat_message("👤 User").write(user_query)

        # Retrieve relevant docs from updated FAISS
        retrieved_docs = retrieve_docs(user_query)

        # Get AI response
        raw_response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        # ✅ Ensure response is a string before processing
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        # Format AI response
        def format_ai_response(response):
            """Formats AI response for Streamlit display with clean, structured output."""
            if not isinstance(response, str):
                return "**⚠️ Error: AI response is not valid text output.**"

            # Remove unwanted JSON-like metadata and system info
            response = re.sub(r"(\s*additional_kwargs\s*=\s*\{.*?\})", "", response, flags=re.DOTALL)
            response = re.sub(r"(\s*response_metadata\s*=\s*\{.*?\})", "", response, flags=re.DOTALL)
            response = re.sub(r"(\s*usage_metadata\s*=\s*\{.*?\})", "", response, flags=re.DOTALL)
            response = re.sub(r"(\s*id\s*=\s*'.*?')", "", response, flags=re.DOTALL)

            # Extract and highlight the <think> section
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            thinking_section = ""
            if think_match:
                thinking_section = f"🧠 **THINKING:**\n\n> {think_match.group(1).strip()}\n\n"
                response = response.replace(think_match.group(0), "")

            # Format numbered lists and bullet points
            response = response.strip()
            response = re.sub(r"(?<=\n)\d+\.", lambda m: f"\n**{m.group(0)}**", response)  # Bold numbering
            response = re.sub(r"(?<=\n)-", lambda m: f"\n  {m.group(0)}", response)  # Indent bullet points
            response = re.sub(r"\*\*(.*?)\*\*", r"**\1**", response)  # Keep bold text for Streamlit

            return f"{thinking_section}📌 **FINAL ANSWER:**\n\n{response.strip()}"

        formatted_response = format_ai_response(raw_response)

        # Display formatted response
        st.chat_message("🤖 AI").markdown(formatted_response)
    else:
        st.error("⚠️ Please upload a PDF and enter a question.")
