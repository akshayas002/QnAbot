import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📄 PDF Q&A Assistant", layout="wide")
st.title("📄 AI PDF Q&A Assistant with Memory")

# Sidebar Controls
st.sidebar.header("⚙️ Settings")
chunk_size = st.sidebar.slider("Chunk size", 200, 2000, 800, 100)
overlap = st.sidebar.slider("Chunk overlap", 0, 500, 100, 50)
model_choice = st.sidebar.selectbox("Choose Model", ["llama3", "mistral", "gemma"])
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# File Upload
# -----------------------------
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        # Save PDF temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(uploaded_file.name)
        all_docs.extend(loader.load())

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(all_docs)

    # Create persistent DB
    persist_dir = "pdf_db"
    embeddings = OllamaEmbeddings(model=model_choice)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    db.persist()

    # Initialize LLM + Retriever
    llm = OllamaLLM(model=model_choice)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=True)

    # -----------------------------
    # Chat Interface
    # -----------------------------
    st.subheader("💬 Chat with your PDFs")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask me anything about the PDFs..."):
        st.session_state.messages.append({"role": "user", "content": query})

        result = qa_chain.invoke({"question": query})
        answer = result["answer"]

        # Add sources
        sources = []
        for doc in result.get("source_documents", []):
            fname = os.path.basename(doc.metadata.get("source", ""))
            page = doc.metadata.get("page", "?")
            sources.append(f"- **{fname}** (Page {page})")
        if sources:
            answer += "\n\n**Sources:**\n" + "\n".join(sources)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

    # -----------------------------
    # Download Chat History
    # -----------------------------
    if st.sidebar.button("💾 Download Chat History"):
        chat_log = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Download Chat Log", chat_log, "chat_history.txt")
