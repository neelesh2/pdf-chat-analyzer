import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings , HuggingFaceEndpoint , ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load the saved index
embeddings = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)
vs = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vs.as_retriever(search_kwargs={"k":4})

# connect the hugging face LLM
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    temperature = 0.2,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

LLM = ChatHuggingFace(llm = endpoint)

# Script Prompt
prompt = ChatPromptTemplate.from_messages([
    ('system',
    "You are a helpful assistant. Answer ONLY using the context below. "
    "If the answer is not in the context, say \"I don't know based on the document.\"\n\n"
    "context:\n{context}"),
    ("human", "{question}")
])

chain = prompt | LLM

# Ask a question in loop
while True:
    question = input("\n Ask a question (or type exit:)")
    if question.lower() == 'exit':
        break

    docs = retriever.invoke(question)
    context = "\n\n".join((d.page_content for d in docs))

    answer = chain.invoke({
        'context':context,
        'question':question
    })

    print("\nAnswer:", answer.content)

    print("\nSources")
    for d in docs:
        print(f"- page {d.metadata.get('page_label')} : {d.page_content[:80]}...")