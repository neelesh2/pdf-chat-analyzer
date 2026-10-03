from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

INDEX_DIR = "faiss_index"

# Load PDF
loader = PyPDFLoader("C:\\Users\\LENOVO\\OneDrive\\Desktop\\project folder\\PDF Chat Analyzer\\pdf-chat-analyzer\\Resume_Neelesh_pahuja (1).pdf")
docs = loader.load()
print(f"Pages Loaded: {len(docs)}")

# split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)
chunks = splitter.split_documents(docs)
print(f"Chunks created: {len(chunks)}")
print("Sample chunk metadata:", chunks[0].metadata)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

# Build and save the FAISS index
vs = FAISS.from_documents(chunks , embeddings)
vs.save_local(INDEX_DIR)
print("Index Saved")

# test and search
results = vs.similarity_search("What is this document about",k=3)
for i , r in enumerate(results , 1):
    print(f"\n--Result {i} (page {r.metadata.get('page')})--")
    print(r.page_content[:300])