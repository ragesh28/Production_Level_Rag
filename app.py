import streamlit as st
import os
import base64
import time
import pandas as pd  # <--- NEW IMPORT FOR EXCEL
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- AGENT & TOOLS IMPORTS ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# --- PAGE CONFIG ---
st.set_page_config(page_title="Production RAG", layout="wide", page_icon="🧠")
st.title("Production RAG 🧠 (Hybrid + Memory + Stream + Excel)")

# 1. SETUP GEMINI 2.5 FLASH
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    max_retries=2,
)

# 2. SETUP QDRANT CLOUD
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")
collection_name = "production_hybrid_v4" 

if not url or not api_key:
    st.error("Missing keys in .env")
    st.stop()

client = QdrantClient(url=url, api_key=api_key)

# --- HELPER: STREAMING GENERATOR ---
def stream_text(text):
    """Simulates typing effect for the AI response"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# --- HELPER: IMAGE TO TEXT ---
def summarize_image(image_file):
    image_bytes = image_file.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail for search indexing."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    )
    response = llm.invoke([message])
    return response.content

# --- HELPER: EXCEL TO TEXT (NEW) ---
def process_excel(file_path):
    """Reads Excel and converts rows to text"""
    df = pd.read_excel(file_path)
    # Convert the entire dataframe to a string format the AI can read
    return df.to_string(index=False)

# --- 3. SIDEBAR (DATA LOADING) ---
with st.sidebar:
    st.header("⚙️ Settings")
    use_web_search = st.toggle("Enable Web Search 🌍", value=False)
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.header("📂 Knowledge Base")
    
    # UPDATED: Added xlsx and xls to the accepted types
    uploaded_files = st.file_uploader(
        "Upload Data", 
        type=["pdf", "txt", "jpg", "png", "xlsx", "xls"], 
        accept_multiple_files=True
    )
    
    user_text_input = st.text_area("Paste Text:", height=100)
    process_btn = st.button("Save to Brain")

    if process_btn:
        documents = []
        with st.spinner("Processing & Vectorizing..."):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = f"./temp_{uploaded_file.name}"
                    
                    # LOGIC 1: IMAGES
                    if uploaded_file.type in ["image/jpeg", "image/png"]:
                        desc = summarize_image(uploaded_file)
                        documents.append(Document(page_content=desc, metadata={"source": uploaded_file.name}))
                    
                    # LOGIC 2: FILES (PDF/TXT/EXCEL)
                    else:
                        # Save temp file
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Handle PDF
                        if uploaded_file.name.endswith(".pdf"):
                            documents.extend(PyPDFLoader(file_path).load())
                        
                        # Handle Excel (NEW)
                        elif uploaded_file.name.endswith((".xlsx", ".xls")):
                            text_data = process_excel(file_path)
                            documents.append(Document(page_content=text_data, metadata={"source": uploaded_file.name}))
                        
                        # Handle Text
                        else:
                            documents.extend(TextLoader(file_path, encoding="utf-8").load())
            
            if user_text_input:
                documents.append(Document(page_content=user_text_input, metadata={"source": "User Paste"}))

            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                dense_embeddings = FastEmbedEmbeddings() 
                sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") 
                
                QdrantVectorStore.from_documents(
                    chunks, 
                    embedding=dense_embeddings, 
                    sparse_embedding=sparse_embeddings, 
                    url=url, 
                    api_key=api_key, 
                    collection_name=collection_name, 
                    retrieval_mode=RetrievalMode.HYBRID, 
                    prefer_grpc=True
                )
                st.success(f"Saved {len(chunks)} chunks!")

# --- 4. SETUP TOOLS & AGENT ---
dense_embeddings = FastEmbedEmbeddings()
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

try:
    # Connect to Qdrant
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=collection_name, 
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID 
    )
    
    # Hybrid Retriever + Reranking (FlashRank)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 50})
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    # Tool 1: Docs
    retriever_tool = create_retriever_tool(
        compression_retriever, 
        "knowledge_base_search",
        "Use this tool to find information in the user's uploaded documents."
    )

    tools = [retriever_tool]
    
    # Tool 2: Web (Optional)
    if use_web_search:
        tools.append(DuckDuckGoSearchRun())

    # SYSTEM PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a smart assistant. Follow this logic strictly:
        1. ALWAYS check the 'chat_history' first. If the user is referring to a previous message, answer from memory.
        2. If the answer is NOT in memory, use 'knowledge_base_search' to check the documents (including Excel data).
        3. Use web search only for current events.
        """),
        MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

except Exception as e:
    st.error(f"⚠️ Database Error: {e}")
    st.stop()

# --- 5. CHAT INTERFACE & HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. HANDLE USER INPUT ---
if prompt_input := st.chat_input("Ask anything..."):
    
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Prepare History
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # Run Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({
                "input": prompt_input,
                "chat_history": chat_history 
            })
            answer = response["output"]
        
        # Stream the output
        st.write_stream(stream_text(answer))
    
    # Update Session State
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})