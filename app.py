import streamlit as st
import os
import base64
import time
import subprocess
import pandas as pd  # <--- CRITICAL: Import Pandas
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- IMPORTS ---
from chatgpt_automation import ChatGPTBridge, ChatGPTAutomationLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- AGENT & TOOLS IMPORTS ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

PASSWORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "password.txt")

def get_stored_password():
    if not os.path.exists(PASSWORD_FILE):
        try:
            with open(PASSWORD_FILE, "w") as f:
                f.write("admin123")
        except Exception:
            pass
        return "admin123"
    try:
        with open(PASSWORD_FILE, "r") as f:
            return f.read().strip()
    except Exception:
        return "admin123"

# --- SYSTEM CONTROL TOOL (MCP) ---
APP_SHORTCUTS = {
    "chrome": "start chrome",
    "google chrome": "start chrome",
    "notepad": "start notepad",
    "calculator": "start calc",
    "calc": "start calc",
    "file explorer": "start explorer",
    "explorer": "start explorer",
    "cmd": "start cmd",
    "terminal": "start cmd",
    "command prompt": "start cmd",
    "powershell": "start powershell",
    "task manager": "start taskmgr",
    "paint": "start mspaint",
    "word": "start winword",
    "excel": "start excel",
    "vscode": "start code",
    "vs code": "start code",
    "spotify": "start spotify",
    "settings": "start ms-settings:",
    "snipping tool": "start snippingtool",
}

def is_delete_command(command: str) -> bool:
    """Check if the command contains file deletion patterns."""
    cmd_lower = command.lower().strip()
    import re
    
    # List of forbidden delete command words/patterns
    delete_patterns = [
        r'\bdel\b',
        r'\brm\b',
        r'\brmdir\b',
        r'\brd\b',
        r'\berase\b',
        r'\bshred\b',
        r'\bsdelete\b',
        r'\bremove-item\b'
    ]
    
    for pattern in delete_patterns:
        if re.search(pattern, cmd_lower):
            return True
            
    # Substring checks for explicit deletion syntax
    for kw in ["rm -", "rd /", "rmdir /", "remove-item "]:
        if kw in cmd_lower:
            return True
            
    return False

@tool
def system_control(command: str) -> str:
    """Execute a system command on the user's Windows PC.
    Use this to open applications (e.g. 'open chrome', 'open notepad'),
    run terminal commands (e.g. 'list files', 'show ip address'),
    or open websites (e.g. 'open youtube.com').
    Input should be a natural language description of what to do.
    """
    import streamlit as st
    cmd_lower = command.lower().strip()

    # Block direct access to credential files
    if "password.txt" in cmd_lower or "users.json" in cmd_lower:
        return "❌ Access Denied: You do not have permission to access security credentials."

    # 1. Check if the input specifies an existing local file or folder path
    cleaned_path = command.strip()
    for prefix in ["start", "open", "launch"]:
        if cleaned_path.lower().startswith(prefix):
            cleaned_path = cleaned_path[len(prefix):].strip()
            break
    if cleaned_path.startswith('""'):
        cleaned_path = cleaned_path[2:].strip()
    elif cleaned_path.startswith("''"):
        cleaned_path = cleaned_path[2:].strip()
    cleaned_path = cleaned_path.strip("\"'")
    
    # 1b. Check if command is to open any random image
    if "random image" in cmd_lower or "any image" in cmd_lower:
        if not st.session_state.get("password_verified", False):
            return "PASSWORD_REQUIRED_FOR_RUN_CMD:random_image"
        import random
        search_dirs = [
            os.path.join(os.path.expanduser("~"), "Downloads"),
            os.path.join(os.path.expanduser("~"), "Pictures"),
            os.getcwd()
        ]
        image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
        found_images = []
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                try:
                    for file in os.listdir(s_dir):
                        if file.lower().endswith(image_extensions):
                            found_images.append(os.path.join(s_dir, file))
                except Exception:
                    pass
        if found_images:
            selected_image = random.choice(found_images)
            try:
                os.startfile(selected_image)
                return f"✅ Opened file/folder successfully: {selected_image}"
            except Exception as e:
                return f"❌ Failed to open random image {selected_image}: {e}"
        else:
            return "❌ Could not find any images in Downloads, Pictures, or the current directory."

    if not os.path.exists(cleaned_path) and is_delete_command(command):
        return "❌ Access Denied: Deletion commands (such as del, rm, rmdir, rd, erase, remove-item) are strictly prohibited for safety."

    if os.path.exists(cleaned_path):
        if not st.session_state.get("password_verified", False):
            return f"PASSWORD_REQUIRED_FOR_RUN_CMD:{cleaned_path}"
        try:
            os.startfile(cleaned_path)
            return f"✅ Opened file/folder successfully: {cleaned_path}"
        except Exception as e:
            return f"❌ Failed to open path: {e}"

    # 2. Check for "open <app>" pattern
    for prefix in ["open ", "launch ", "start "]:
        if cmd_lower.startswith(prefix):
            target = cmd_lower[len(prefix):].strip()

            # Check known app shortcuts
            if target in APP_SHORTCUTS:
                try:
                    subprocess.Popen(APP_SHORTCUTS[target], shell=True)
                    return f"✅ Opened {target} successfully."
                except Exception as e:
                    return f"❌ Failed to open {target}: {e}"

            # Check if it looks like a URL (prevent confusing local paths with URLs)
            if "." in target and " " not in target and not "\\" in target:
                url = target if target.startswith("http") else f"https://{target}"
                try:
                    subprocess.Popen(f'start "" "{url}"', shell=True)
                    return f"✅ Opened {url} in browser."
                except Exception as e:
                    return f"❌ Failed to open URL: {e}"

            # Try opening as a generic command
            try:
                subprocess.Popen(f"start {target}", shell=True)
                return f"✅ Tried to open '{target}'."
            except Exception as e:
                return f"❌ Could not open '{target}': {e}"

    # 3. Direct shell command execution (preserving case)
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip() or result.stderr.strip() or "Command executed (no output)."
        return f"✅ Command result:\n{output}"
    except subprocess.TimeoutExpired:
        return "⏱️ Command timed out after 15 seconds."
    except Exception as e:
        return f"❌ Error running command: {e}"

@tool
def share_file_to_user(file_path: str) -> str:
    """Use this tool when the user asks you to send or share a file from the PC to their device.
    Provide the absolute path to the file. This returns a special flag to trigger a download button in the UI."""
    import os
    import streamlit as st
    clean_path = file_path.strip("\"'")
    
    # Block direct download of password.txt file
    if "password.txt" in clean_path.lower():
        return "❌ Access Denied: You do not have permission to share the password file."
        
    if not os.path.exists(clean_path):
        return f"Could not find the file at {clean_path}"
        
    if not st.session_state.get("password_verified", False):
        return f"PASSWORD_REQUIRED_FOR_FILE_SHARE:{clean_path}"
        
    return f"FILE_SHARE_REQUEST:{clean_path}"

# --- FIX: Commented out broken imports ---
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import FlashrankRerank

# --- PAGE CONFIG ---
st.set_page_config(page_title="Production RAG", layout="wide", page_icon="🧠")
st.title("Production RAG 🧠 (Hybrid + Memory + Excel Support)")

if "password_verified" not in st.session_state:
    st.session_state["password_verified"] = False

# 1. SETUP LLM
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Password lock status
    st.subheader("🔐 File Protection")
    if st.session_state["password_verified"]:
        st.success("🔓 Files Unlocked")
        if st.button("🔒 Lock Files", key="lock_files_btn"):
            st.session_state["password_verified"] = False
            try:
                import requests
                requests.post("http://localhost:5000/api/reset_password_status")
            except Exception:
                pass
            st.rerun()
    else:
        st.error("🔒 Files Locked")
        entered_pass = st.text_input("Enter password to unlock:", type="password", key="sidebar_pass_input")
        if st.button("Verify Password", key="verify_pass_btn"):
            stored_pass = get_stored_password()
            if entered_pass == stored_pass:
                st.session_state["password_verified"] = True
                try:
                    import requests
                    requests.post("http://localhost:5000/api/verify_password", json={"password": entered_pass})
                except Exception:
                    pass
                st.success("Successfully unlocked!")
                st.rerun()
            else:
                st.error("Incorrect password.")
                
    st.divider()
    model_choice = st.radio("Select AI Model 🤖", [
        "Gemini 2.5 Flash", 
        "Groq (Llama 3)", 
        "OpenRouter", 
        "Local LM Studio (Offline)",
        "Local LLM 2 (ChatGPT)"
    ])

# Initialize ChatGPT Bridge in session state
if "chatgpt_bridge" not in st.session_state:
    st.session_state.chatgpt_bridge = ChatGPTBridge(chrome_version=148)

if model_choice == "Gemini 2.5 Flash":
    google_keys = os.getenv("GOOGLE_API_KEYS", "")
    api_key = google_keys.split(",")[0].strip() if google_keys else None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
    )
elif model_choice == "Groq (Llama 3)":
    from langchain_groq import ChatGroq
    groq_keys = os.getenv("GROQ_API_KEYS", "")
    api_key = groq_keys.split(",")[0].strip() if groq_keys else None
    
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        groq_api_key=api_key,
        temperature=0
    )
elif model_choice == "OpenRouter":
    from langchain_openai import ChatOpenAI
    or_keys = os.getenv("OPENROUTER_API_KEYS", "")
    api_key = or_keys.split(",")[0].strip() if or_keys else None
    
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="deepseek/deepseek-v4-flash:free",
        temperature=0,
    )
elif model_choice == "Local LLM 2 (ChatGPT)":
    with st.sidebar:
        st.info("💡 To use ChatGPT, click 'Open Chrome' and solve any Captcha.")
        if st.button("🌐 Open Chrome (Solve Captcha)"):
            st.session_state.chatgpt_bridge.initialize()
            st.success("Browser Opened! Solve any captcha there.")
        
        # ADDED: Direct Chat Toggle
        use_agent_mode = st.checkbox("Enable Agent Mode (RAG/Tools)", value=False, help="Turn this on to use your uploaded files and system control. Turn it off for direct, clean chat with ChatGPT.")
    
    llm = ChatGPTAutomationLLM(bridge=st.session_state.chatgpt_bridge)
    
    # If not in agent mode, we skip the AgentExecutor and talk to LLM directly later
    if not use_agent_mode:
        model_is_direct = True
    else:
        model_is_direct = False
else:
    model_is_direct = False
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="local-model",
        temperature=0,
    )

# 2. SETUP CHROMA DB (LOCAL)
collection_name = "production_hybrid_v4"
persist_directory = "./chroma_db"

# --- CUSTOM CHATGPT JSON AGENT LOOP ---
def execute_tool_by_name_streamlit(action, action_input):
    try:
        if action == "run_cmd":
            return system_control.func(action_input)
        elif action == "knowledge_base_search":
            vector_store = Chroma(
                collection_name="production_hybrid_v4", 
                embedding_function=FastEmbedEmbeddings(),
                persist_directory="./chroma_db"
            )
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
            docs = base_retriever.invoke(action_input)
            if not docs:
                return "No matching documents found in knowledge base."
            return "\n\n".join([f"Source File: {doc.metadata.get('source', 'unknown')}\nContent Snippet:\n{doc.page_content}" for doc in docs])
        elif action == "share_file_to_user":
            return share_file_to_user.func(action_input)
        else:
            return f"Error: Tool '{action}' is not supported."
    except Exception as e:
        return f"Error executing tool '{action}': {str(e)}"

def run_chatgpt_agent_loop_streamlit(user_message, chat_history):
    import json
    import re
    
    project_dir = os.getcwd()
    
    system_prompt = f"""You are a Windows System Control and RAG Expert assistant.
Current Directory: {project_dir}

You have access to the following tools:
1. run_cmd:
   Description: Run a Windows shell command or open a local file/folder path.
   Input: A Windows command string OR a direct absolute path to a file/folder you want to open.
   CRITICAL: To open a file or folder, just output the path directly in `action_input` (e.g. "C:\\Users\\Ragesh.l\\Downloads\\download.jpg"). Do not use 'start ""' or extra quotes.
2. knowledge_base_search:
   Description: Search the local vector database of uploaded documents (PDF, Excel, txt, images) for relevant information.
   Input: A search query string.
3. share_file_to_user:
   Description: Share or prepare a local file for the user to download.
   Input: Absolute path of the file.

Your goal is to help the user. If you need to use any tool to answer the user's request, you must respond with a JSON object in this exact format:
{{
  "thought": "Reasoning about what to do next",
  "action": "tool_name",
  "action_input": "tool input details"
}}

If you do not need any tools (e.g. just greeting, or you have completed all tool runs), respond with a JSON object in this exact format:
{{
  "thought": "Reasoning about the final answer",
  "final_answer": "Your actual final response to the user"
}}

CRITICAL INSTRUCTIONS:
1. Do NOT write any conversation or explanations outside of the JSON block. Respond ONLY with the JSON object.
2. Always escape double quotes inside the JSON string values (e.g. write \\\" instead of \").
3. If your final answer contains any programming code (Python, JS, SQL, HTML, etc.), you MUST wrap it inside standard Markdown fenced code blocks (e.g. ```python ... ```) so it renders correctly in our code box UI.
4. When using the share_file_to_user tool, you MUST include the exact file share tag returned by the tool (e.g., FILE_SHARE_REQUEST:path) inside the "final_answer" string so the user can download the file.
"""

    prompt = f"INSTRUCTIONS:\n{system_prompt}\n\n"
    if chat_history:
        prompt += "PREVIOUS CONVERSATION HISTORY:\n"
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                prompt += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                prompt += f"Assistant: {msg.content}\n"
        prompt += "\n"
    
    prompt += f"USER REQUEST: {user_message}\n"
    
    max_iterations = 5
    current_prompt = prompt
    shared_files = []
    
    # Helper to sanitize single backslashes in Windows file paths inside JSON values
    def sanitize_json_backslashes(text_str):
        pattern = r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})'
        return re.sub(pattern, r'\\\\', text_str)
    
    for i in range(max_iterations):
        response_text = st.session_state.chatgpt_bridge.get_response(current_prompt.strip())
        
        match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if not match:
            ans = response_text
            if shared_files:
                for sf in shared_files:
                    if sf not in ans:
                        ans += f"\n\n{sf}"
            return ans
        
        raw_json_str = match.group(1).strip()
        json_str = sanitize_json_backslashes(raw_json_str)
        res_json = {}
        try:
            res_json = json.loads(json_str)
        except Exception as e:
            # Regex fallback
            act_match = re.search(r'"action"\s*:\s*"([^"]+)"', json_str)
            if act_match:
                res_json["action"] = act_match.group(1).strip()
                inp_match = re.search(r'"action_input"\s*:\s*"(.*)"\s*\}', json_str, re.DOTALL)
                if not inp_match:
                    inp_match = re.search(r'"action_input"\s*:\s*"(.*?)"\s*,', json_str, re.DOTALL)
                if inp_match:
                    res_json["action_input"] = inp_match.group(1).strip()
            else:
                ans_match = re.search(r'"final_answer"\s*:\s*"(.*)"\s*\}', json_str, re.DOTALL)
                if not ans_match:
                    ans_match = re.search(r'"final_answer"\s*:\s*"(.*?)"\s*,', json_str, re.DOTALL)
                if ans_match:
                    res_json["final_answer"] = ans_match.group(1).strip()
            
        if "action" in res_json:
            action = res_json["action"]
            action_input = res_json.get("action_input", "")
            
            observation = execute_tool_by_name_streamlit(action, action_input)
            if action == "share_file_to_user" and "FILE_SHARE_REQUEST:" in observation:
                shared_files.append(observation)
            
            current_prompt += f"\nAssistant: {json_str}\nObservation: {observation}\n"
        elif "final_answer" in res_json:
            ans = res_json["final_answer"]
            if shared_files:
                for sf in shared_files:
                    if sf not in ans:
                        ans += f"\n\n{sf}"
            return ans
        else:
            ans = response_text
            if shared_files:
                for sf in shared_files:
                    if sf not in ans:
                        ans += f"\n\n{sf}"
            return ans
            
    ans = "Agent loop reached maximum iterations without a final answer."
    if shared_files:
        for sf in shared_files:
            ans += f"\n\n{sf}"
    return ans

# --- HELPER: STREAMING GENERATOR ---
def stream_text(text):
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

# --- HELPER: EXCEL TO TEXT (THE FIX) ---
def process_excel(file_path):
    """Reads Excel and converts it to text format for the AI"""
    try:
        df = pd.read_excel(file_path)
        # Convert to text so the AI can read it like a document
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading Excel: {str(e)}"

# --- 3. SIDEBAR (DATA LOADING) ---
with st.sidebar:
    use_web_search = st.toggle("Enable Web Search 🌍", value=False)
    use_system_control = st.toggle("Enable System Control 🖥️", value=True)
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.header("📂 Knowledge Base")
    
    # --- THE FIX IS HERE: Added "xlsx" and "xls" to allowed types ---
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
            # 1. Handle Files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = f"./temp_{uploaded_file.name}"
                    
                    # Image Logic
                    if uploaded_file.type in ["image/jpeg", "image/png"]:
                        desc = summarize_image(uploaded_file)
                        documents.append(Document(page_content=desc, metadata={"source": uploaded_file.name}))
                    
                    # Document Logic
                    else:
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        if uploaded_file.name.endswith(".pdf"):
                            documents.extend(PyPDFLoader(file_path).load())
                        
                        # --- EXCEL LOGIC ---
                        elif uploaded_file.name.endswith((".xlsx", ".xls")):
                            text_data = process_excel(file_path)
                            documents.append(Document(page_content=text_data, metadata={"source": uploaded_file.name}))
                        
                        # Text Logic
                        else:
                            documents.extend(TextLoader(file_path, encoding="utf-8").load())
            
            # 2. Handle Pasted Text
            if user_text_input:
                documents.append(Document(page_content=user_text_input, metadata={"source": "User Paste"}))

            # 3. Save to Vector DB
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                dense_embeddings = FastEmbedEmbeddings() 
                
                Chroma.from_documents(
                    chunks, 
                    embedding=dense_embeddings, 
                    persist_directory=persist_directory,
                    collection_name=collection_name
                )
                st.success(f"Saved {len(chunks)} chunks! Now you can ask about the file.")
            else:
                st.warning("No data found to save!")

# --- 4. SETUP TOOLS & AGENT ---
dense_embeddings = FastEmbedEmbeddings()

try:
    vector_store = Chroma(
        collection_name=collection_name, 
        embedding_function=dense_embeddings,
        persist_directory=persist_directory
    )
    
    # --- FIX: BYPASS RERANKER TO PREVENT ERRORS ---
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 50})
    
    # compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    # compression_retriever = ContextualCompressionRetriever(
    #    base_compressor=compressor, 
    #    base_retriever=base_retriever
    # )

    retriever_tool = create_retriever_tool(
        base_retriever,  # <--- CHANGED THIS from compression_retriever to base_retriever
        "knowledge_base_search",
        "Use this tool to find information in uploaded documents and Excel files."
    )

    tools = [retriever_tool, share_file_to_user]
    
    if use_web_search:
        tools.append(DuckDuckGoSearchRun())
    
    if use_system_control:
        tools.append(system_control)

    if model_choice == "Local LLM 2 (ChatGPT)":
        from langchain.agents import create_react_agent
        # ReAct prompt format
        react_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a smart assistant. 
            To use a tool, you MUST use the following format:
            Thought: Do I need to use a tool? Yes
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            CRITICAL: DO NOT use ANY tools unless the user EXPLICITLY asks you to.
            Current Tools: {tools}

            CRITICAL SECURITY: If any tool returns 'PASSWORD_REQUIRED_FOR_FILE_SHARE:<path>' or 'PASSWORD_REQUIRED_FOR_RUN_CMD:<path>', your Final Answer MUST strictly and only be 'PASSWORD_REQUIRED:<path>' and nothing else. You do not have permission to delete files. If the user asks you to delete a file, explain that file deletion is disabled.
            """),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}\n\n{agent_scratchpad}"),
        ])
        agent = create_react_agent(llm, tools, react_prompt)
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a smart assistant. 
            CRITICAL RULE: DO NOT use ANY tools unless the user EXPLICITLY asks you to. If the user just says "hi", "hello", or asks a general chat question, reply directly without using any tools!
            
            1. FIRST check 'chat_history' for context.
            2. Use 'knowledge_base_search' ONLY if the user asks a question that requires searching the uploaded documents.
            3. If the user explicitly asks to open an application (e.g. "open chrome") or run a system command, use the 'system_control' tool.
            4. If the user asks to send, share, or download a file to their device, use the 'share_file_to_user' tool.
            5. CRITICAL SECURITY: If any tool returns a string starting with 'PASSWORD_REQUIRED_FOR_FILE_SHARE:' or 'PASSWORD_REQUIRED_FOR_RUN_CMD:', you MUST immediately stop and return exactly and only 'PASSWORD_REQUIRED:<path>' as your output. Do NOT explain anything else.
            6. You do not have permission to delete files. If the user asks you to delete any files, reply that file deletion is disabled for safety.
            """),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

except Exception as e:
    st.error(f"⚠️ Database Error: {e}")
    st.stop()

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_input_val = st.chat_input("Ask about your Excel file...")
prompt_input = chat_input_val
if "pending_prompt" in st.session_state:
    prompt_input = st.session_state.pop("pending_prompt")

if prompt_input:
    with st.chat_message("user"):
        st.markdown(prompt_input)

    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # --- NEW: Handle ChatGPT with Custom Agent Loop ---
                if model_choice == "Local LLM 2 (ChatGPT)":
                    answer = run_chatgpt_agent_loop_streamlit(prompt_input, chat_history)
                else:
                    # Standard Agent execution (for Gemini/Groq, etc.)
                    response = agent_executor.invoke({
                        "input": prompt_input,
                        "chat_history": chat_history 
                    })
                    answer = response["output"]
            except Exception as e:
                st.error(f"⚠️ **Model Provider Error:** The selected AI model is currently overloaded, rate-limited, or unavailable. Please switch to another model from the sidebar (like Gemini or Groq) and try again.\n\n*Technical Details: {str(e)}*")
                st.stop()
            
        if "PASSWORD_REQUIRED:" in answer:
            st.session_state["pending_prompt"] = prompt_input
            st.warning("🔐 Action Protected: File lock is active. Please enter the password to execute this action.")
            entered_pass_chat = st.text_input("Enter password:", type="password", key=f"chat_pass_{len(st.session_state.messages)}")
            if st.button("Verify & Execute", key=f"chat_pass_btn_{len(st.session_state.messages)}"):
                stored_pass = get_stored_password()
                if entered_pass_chat == stored_pass:
                    st.session_state["password_verified"] = True
                    try:
                        import requests
                        requests.post("http://localhost:5000/api/verify_password", json={"password": entered_pass_chat})
                    except Exception:
                        pass
                    st.success("Unlocked! Re-running command...")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        elif "FILE_SHARE_REQUEST:" in answer:
            import os
            parts = answer.split("FILE_SHARE_REQUEST:")
            path = parts[1].strip().split("\n")[0]
            clean_answer = answer.replace(f"FILE_SHARE_REQUEST:{path}", "I have prepared the file for download below.")
            st.write_stream(stream_text(clean_answer))
            if os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(label=f"📥 Download {os.path.basename(path)}", data=f, file_name=os.path.basename(path))
        else:
            st.write_stream(stream_text(answer))
    
    # Only append to session history if it wasn't a password protection block
    if "PASSWORD_REQUIRED:" not in answer:
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        st.session_state.messages.append({"role": "assistant", "content": answer})