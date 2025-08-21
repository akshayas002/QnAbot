import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ----------------- UI -----------------
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Q&A Bot (LangChain + Ollama)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # ----------------- Backend Setup -----------------
    llm = OllamaLLM(model="llama3")  # Or "mistral", "phi3", etc.
    loader = PyPDFLoader("uploaded.pdf")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Create vector DB
    embeddings = OllamaEmbeddings(model="llama3")
    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # Add memory for chat
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    # ----------------- Chat UI -----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask something about the PDF...")
    if query:
        # Display user msg
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Get answer
        result = qa.invoke({"question": query})
        answer = result["answer"]

        # Display bot msg
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
