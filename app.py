import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # use the repo_id that worked in qa.py

st.set_page_config(page_title="PDF Chat Analyzer", page_icon="📄")
st.title("📄 PDF Chat Analyzer")

# ---------- Session state (no re-loading on every rerun) ----------
if "embeddings" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
if "vs" not in st.session_state:
    st.session_state.vs = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- LLM + prompt ----------
endpoint = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.2,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
llm = ChatHuggingFace(llm=endpoint)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer ONLY using the context below. "
     "If the answer is not in the context, say \"I don't know based on the document.\"\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])
chain = prompt | llm

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Upload PDFs")
    files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Process"):
        if not files:
            st.error("Please upload at least one PDF.")
        else:
            try:
                with st.spinner("Processing PDFs..."):
                    all_docs = []
                    for f in files:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp.write(f.read())
                        tmp.close()
                        pages = PyPDFLoader(tmp.name).load()
                        os.remove(tmp.name)
                        for p in pages:
                            p.metadata["filename"] = f.name
                        all_docs.extend(pages)

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(all_docs)

                    if not chunks:
                        st.error("No text found. Your PDF may be scanned (image-only).")
                    else:
                        st.session_state.vs = FAISS.from_documents(
                            chunks, st.session_state.embeddings
                        )
                        st.session_state.messages = []
                        st.success(f"Ready! {len(files)} file(s), {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Failed to process PDFs: {e}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Show chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s['file']} (page {s['page']})**")
                    st.caption(s["text"])

# ---------- Chat input ----------
question = st.chat_input("Ask a question about your PDFs...")

if question:
    if st.session_state.vs is None:
        st.warning("Upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    retriever = st.session_state.vs.as_retriever(search_kwargs={"k": 4})
                    docs = retriever.invoke(question)

                first_page = st.session_state.vs.similarity_search(
                    "abstract introduction title authors", k=1,
                    filter={"page": 0},
                        )
                docs = first_page + docs
                context = "\n\n".join(d.page_content for d in docs)
                answer = chain.invoke({"context": context, "question": question}).content

                sources = [
                    {
                        "file": d.metadata.get("filename", "file"),
                        "page": d.metadata.get("page_label", "?"),
                        "text": d.page_content[:200] + "...",
                    }
                    for d in docs
                ]
                st.markdown(answer)
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"**{s['file']} (page {s['page']})**")
                        st.caption(s["text"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Error getting answer: {e}")