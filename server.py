import os
import time
import base64
import subprocess
import threading
from datetime import datetime
import pandas as pd
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, send_file, session
from flask_cors import CORS

# Load environment variables
load_dotenv()

# --- LANGCHAIN IMPORTS ---
try:
    from chatgpt_automation import ChatGPTBridge, ChatGPTAutomationLLM
except ImportError:
    ChatGPTBridge = None
    ChatGPTAutomationLLM = None
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from graph_templates import generate_graph as gen_graph, get_file_columns, GRAPH_TYPES

# ============================================================
# DATA FILES DIR — persistent storage for CSV/Excel uploads
# ============================================================
DATA_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")
os.makedirs(DATA_FILES_DIR, exist_ok=True)
# LOCAL VECTOR DB CONFIG
# ============================================================
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db_data")
COLLECTION_NAME = "neuralrag_local"

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

# ============================================================
# API KEY POOL — auto-rotate on failure
# ============================================================
class KeyPool:
    """Manages multiple API keys per provider with automatic rotation."""

    def __init__(self):
        self.pools = {
            "gemini": [],
            "openrouter": [],
            "groq": [],
        }
        self.index = {"gemini": 0, "openrouter": 0, "groq": 0}
        self._load_from_env()

    def _load_from_env(self):
        """Load comma-separated keys from .env file."""
        env_map = {
            "gemini": "GOOGLE_API_KEYS",
            "openrouter": "OPENROUTER_API_KEYS",
            "groq": "GROQ_API_KEYS",
        }
        for provider, env_var in env_map.items():
            raw = os.getenv(env_var, "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            self.pools[provider] = keys
            self.index[provider] = 0

    def get_current_key(self, provider):
        """Get the current active key for a provider."""
        keys = self.pools.get(provider, [])
        if not keys:
            return None
        idx = self.index.get(provider, 0) % len(keys)
        return keys[idx]

    def rotate(self, provider):
        """Move to the next key. Returns True if a new key is available, False if all exhausted."""
        keys = self.pools.get(provider, [])
        if len(keys) <= 1:
            return False
        self.index[provider] = (self.index.get(provider, 0) + 1) % len(keys)
        return True

    def add_key(self, provider, key):
        """Add a new key at runtime."""
        key = key.strip()
        if key and key not in self.pools.get(provider, []):
            if provider not in self.pools:
                self.pools[provider] = []
            self.pools[provider].append(key)
            return True
        return False

    def get_status(self):
        """Return key counts and masked active key per provider."""
        status = {}
        for provider in self.pools:
            keys = self.pools[provider]
            current = self.get_current_key(provider)
            status[provider] = {
                "total": len(keys),
                "active_index": self.index.get(provider, 0) + 1 if keys else 0,
                "active_key_masked": f"{current[:8]}...{current[-4:]}" if current and len(current) > 12 else current or "none",
                "keys_masked": [f"{k[:8]}...{k[-4:]}" if len(k) > 12 else k for k in keys],
            }
        return status


key_pool = KeyPool()

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================
MODEL_CONFIGS = {
    "groq": {
        "name": "Groq (Llama 3.3 70B)",
        "base_url": "https://api.groq.com/openai/v1",
        "key_provider": "groq",
        "model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "name": "OpenRouter (Auto)",
        "base_url": "https://openrouter.ai/api/v1",
        "key_provider": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "type": "gemini",
        "key_provider": "gemini",
        "model": "gemini-2.5-flash",
    },
    "ollama": {
        "name": "LM Studio (Offline)",
        "base_url": "http://127.0.0.1:1234/v1",
        "key_provider": None,
        "model": "qwen2.5-14b-instruct",
    },
        "ollama": {
        "name": "Ollama (Qwen 2.5 7B)",
        "base_url": "http://127.0.0.1:11434/v1",
        "key_provider": None,
        "model": "qwen2.5:7b",
    },
}

# Persistent Bridge for Server (now handled dynamically per user session)

# ============================================================
# SYSTEM CONTROL TOOL
# ============================================================
APP_SHORTCUTS = {
    "chrome": "start chrome", "google chrome": "start chrome",
    "notepad": "start notepad", "calculator": "start calc", "calc": "start calc",
    "file explorer": "start explorer", "explorer": "start explorer",
    "cmd": "start cmd", "terminal": "start cmd", "command prompt": "start cmd",
    "powershell": "start powershell", "task manager": "start taskmgr",
    "paint": "start mspaint", "word": "start winword", "excel": "start excel",
    "vscode": "start code", "vs code": "start code", "spotify": "start spotify",
    "settings": "start ms-settings:", "snipping tool": "start snippingtool",
}

# Natural language → real Windows command mapping
COMMAND_MAP = {
    "show ip": "ipconfig", "show my ip": "ipconfig", "show ip address": "ipconfig",
    "my ip": "ipconfig", "my ip address": "ipconfig", "ip address": "ipconfig",
    "what is my ip": "ipconfig", "get ip": "ipconfig", "ipconfig": "ipconfig",
    "list files": "dir", "show files": "dir", "dir": "dir", "ls": "dir",
    "show folders": "dir", "list folders": "dir",
    "whoami": "whoami", "who am i": "whoami", "current user": "whoami", "username": "whoami",
    "hostname": "hostname", "computer name": "hostname", "show hostname": "hostname",
    "system info": "systeminfo", "show system info": "systeminfo",
    "battery": "powershell (Get-WmiObject win32_battery).EstimatedChargeRemaining",
    "battery status": "powershell (Get-WmiObject win32_battery).EstimatedChargeRemaining",
    "disk space": "wmic logicaldisk get size,freespace,caption",
    "show disk": "wmic logicaldisk get size,freespace,caption",
    "ping google": "ping -n 2 google.com", "check internet": "ping -n 2 google.com",
    "date": "date /t", "time": "time /t", "show date": "date /t", "show time": "time /t",
    "running processes": "tasklist /fo table", "show processes": "tasklist /fo table",
    "wifi": "netsh wlan show interfaces", "wifi status": "netsh wlan show interfaces",
    "show wifi": "netsh wlan show interfaces",
    # Power controls — sleep/lock/logoff only (shutdown/restart handled by timer system)
    "sleep": "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
    "sleep mode": "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
    "lock": "rundll32.exe user32.dll,LockWorkStation",
    "lock my pc": "rundll32.exe user32.dll,LockWorkStation",
    "lock screen": "rundll32.exe user32.dll,LockWorkStation",
    "log off": "shutdown /l", "logoff": "shutdown /l", "sign out": "shutdown /l",
}

# Shutdown/restart keywords that trigger the timer system instead of direct execution
SHUTDOWN_KEYWORDS = ["shutdown", "shut down", "turn off", "power off"]
RESTART_KEYWORDS = ["restart", "reboot"]
CANCEL_KEYWORDS = ["cancel shutdown", "abort shutdown", "stop shutdown", "cancel restart", "cancel timer"]

# Shutdown timer state
shutdown_timer = {
    "active": False,
    "type": None,       # "shutdown" or "restart"
    "end_time": 0,      # Unix timestamp when it will execute
    "seconds": 0,       # Original seconds set
    "timer_obj": None,  # threading.Timer object
}

def is_delete_command(command: str) -> bool:
    """Check if the command contains file deletion patterns."""
    cmd_lower = command.lower().strip()
    import re
    
    # List of forbidden delete command words/patterns
    # Matches words with word boundaries to avoid false positives (e.g. "deliver" or "model.json")
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
def run_cmd(command: str) -> str:
    import platform
    if platform.system() == "Linux":
        return "❌ System control is disabled in cloud environments."

    """Execute a system command on the user's Windows PC.
    Use this for:
    1. Opening Apps (e.g. 'start chrome', 'start notepad', 'start spotify', 'start code').
    2. Terminal commands (e.g. 'ipconfig', 'dir', 'systeminfo').
    3. Opening Websites (e.g. 'start https://google.com').
    Input must be a valid Windows shell command.
    """
    cmd_lower = command.lower().strip()
    
    # 0a. Block direct access to password.txt and users.json
    if "password.txt" in cmd_lower or "users.json" in cmd_lower:
        return "❌ Access Denied: You do not have permission to access security credentials."
    
    # 0. Check for cancel shutdown/restart
    for kw in CANCEL_KEYWORDS:
        if kw in cmd_lower:
            if shutdown_timer["timer_obj"]:
                shutdown_timer["timer_obj"].cancel()
            shutdown_timer["active"] = False
            shutdown_timer["type"] = None
            shutdown_timer["timer_obj"] = None
            subprocess.run("shutdown /a", shell=True, capture_output=True)
            return "✅ Shutdown/restart timer cancelled! Your PC is safe."

    # 0b. Check for shutdown/restart — use timer system
    is_shutdown = any(kw in cmd_lower for kw in SHUTDOWN_KEYWORDS)
    is_restart = any(kw in cmd_lower for kw in RESTART_KEYWORDS)
    if is_shutdown or is_restart:
        import re
        seconds = 60
        hours = re.search(r'(\d+)\s*hour', cmd_lower)
        mins = re.search(r'(\d+)\s*min', cmd_lower)
        secs = re.search(r'(\d+)\s*sec', cmd_lower)
        if hours or mins or secs:
            seconds = 0
            if hours: seconds += int(hours.group(1)) * 3600
            if mins: seconds += int(mins.group(1)) * 60
            if secs: seconds += int(secs.group(1))
        action = "restart" if is_restart else "shutdown"
        if shutdown_timer["timer_obj"]:
            shutdown_timer["timer_obj"].cancel()
        shutdown_timer["active"] = True
        shutdown_timer["type"] = action
        shutdown_timer["seconds"] = seconds
        shutdown_timer["end_time"] = time.time() + seconds
        shutdown_timer["timer_obj"] = threading.Timer(seconds, execute_power_action)
        shutdown_timer["timer_obj"].daemon = True
        shutdown_timer["timer_obj"].start()
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        time_str = ""
        if h: time_str += f"{h} hour{'s' if h>1 else ''} "
        if m: time_str += f"{m} minute{'s' if m>1 else ''} "
        if s and not h: time_str += f"{s} second{'s' if s>1 else ''}"
        return f"⏱️ {action.title()} scheduled in **{time_str.strip()}**. A countdown timer is shown at the top of the screen. Say **'cancel shutdown'** to abort."

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
        if not app_state.get("password_verified", False):
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
                return f"✅ Found and opened a random image: {selected_image}"
            except Exception as e:
                return f"❌ Failed to open random image {selected_image}: {e}"
        else:
            return "❌ Could not find any images in Downloads, Pictures, or the current directory."

    if not os.path.exists(cleaned_path) and is_delete_command(command):
        return "❌ Access Denied: Deletion commands (such as del, rm, rmdir, rd, erase, remove-item) are strictly prohibited for safety."

    if os.path.exists(cleaned_path):
        if not app_state.get("password_verified", False):
            return f"PASSWORD_REQUIRED_FOR_RUN_CMD:{cleaned_path}"
        try:
            os.startfile(cleaned_path)
            return f"✅ Opened file/folder successfully: {cleaned_path}"
        except Exception as e:
            return f"❌ Failed to open path: {e}"

    # 2. Simple mapping for known app shortcuts (run only if request is strictly to open/start/launch that app)
    target_cmd = cmd_lower
    for prefix in ["open ", "launch ", "start "]:
        if cmd_lower.startswith(prefix):
            target_cmd = cmd_lower[len(prefix):].strip()
            break
            
    if target_cmd in APP_SHORTCUTS:
        shell_cmd = APP_SHORTCUTS[target_cmd]
        try:
            subprocess.Popen(shell_cmd, shell=True)
            return f"✅ Executed: {shell_cmd} (opened {target_cmd})"
        except Exception as e:
            return f"❌ Failed to open {target_cmd}: {e}"

    # 3. Direct execution (preserving case of original command)
    try:
        # Use Popen to prevent blocking for GUI apps
        if cmd_lower.startswith("start"):
            subprocess.Popen(command, shell=True)
            return f"✅ Launched: {command}"
        
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip() or result.stderr.strip() or "Command executed (no output)."
        return f"✅ Command Output:\n{output}"
    except subprocess.TimeoutExpired:
        return "⏱️ Command timed out after 15 seconds."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================
# MYSQL QUERY TOOL
# ============================================================
def get_mysql_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "neuralrag_db")
    )

@tool
def mysql_query(query: str) -> str:
    """Run a SQL query on the connected MySQL database and return results.
    Use this when the user asks about customers, employees, products, orders, or any database data.
    You can run SELECT, SHOW, or DESCRIBE queries.
    Always use SELECT queries to answer questions about data.
    """
    query_stripped = query.strip().rstrip(';')
    first_word = query_stripped.split()[0].upper() if query_stripped else ""
    if first_word in ("DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT", "CREATE"):
        return "Only SELECT / SHOW / DESCRIBE queries are allowed for safety."
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(query_stripped)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return "Query returned 0 rows."

        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows[:50])
        result = f"{header}\n{sep}\n{body}"
        if len(rows) > 50:
            result += f"\n\n*...showing 50 of {len(rows)} rows*"
        return f"Query returned {len(rows)} rows:\n\n{result}"
    except Exception as e:
        return f"Database Error: {e}. IMPORTANT: The database is offline. Use 'knowledge_base_search' to look for the information in the user's uploaded files instead."

# ============================================================
# GRAPH TOOL — Separation of Concerns (pre-built templates)
# ============================================================
@tool
def make_graph(params: str) -> str:
    """Generate a graph/chart from uploaded CSV or Excel data.
    Input must be a pipe-separated string: graph_type|file_name|x_column|y_column|title
    graph_type: bar, line, pie, scatter, histogram, hbar
    file_name: name of the uploaded file (e.g. sales.csv)
    x_column: column name for X axis (or labels for pie)
    y_column: column name for Y axis (or values for pie, leave empty for histogram)
    title: chart title
    Example: bar|sales.csv|Product|Revenue|Top Products by Revenue
    Example: histogram|data.csv|Age||Age Distribution
    Example: pie|sales.csv|Category|Amount|Sales by Category
    """
    try:
        parts = [p.strip() for p in params.split("|")]
        if len(parts) < 4:
            # List available files and their columns
            available = []
            for f in os.listdir(DATA_FILES_DIR):
                cols = get_file_columns(os.path.join(DATA_FILES_DIR, f))
                available.append(f"{f}: columns={cols}")
            return f"Need format: graph_type|file_name|x_col|y_col|title. Available files: {'; '.join(available) if available else 'No files uploaded yet'}. Graph types: {list(GRAPH_TYPES.keys())}"

        graph_type = parts[0].lower()
        file_name = parts[1]
        x_col = parts[2]
        y_col = parts[3] if parts[3] else None
        title = parts[4] if len(parts) > 4 else f"{graph_type.title()} Chart"

        file_path = os.path.join(DATA_FILES_DIR, file_name)
        if not os.path.exists(file_path):
            # Try finding file with fuzzy match
            for f in os.listdir(DATA_FILES_DIR):
                if file_name.lower() in f.lower():
                    file_path = os.path.join(DATA_FILES_DIR, f)
                    break

        filename, desc = gen_graph(graph_type, file_path, x_col, y_col, title)
        if filename:
            return f"GRAPH_IMAGE:/static/graphs/{filename}|{desc}"
        else:
            return f"Error: {desc}"
    except Exception as e:
        return f"Graph error: {str(e)}"

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__, static_folder="static")
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "neuralrag-super-secret-key-12345")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# --- LOCAL EMBEDDINGS (loaded once) ---
print("[*] Loading embedding model (first time may take a moment)...")
dense_embeddings = FastEmbedEmbeddings()
print("[OK] Embedding model loaded!")

# ============================================================
# MULTI-USER STATE & SESSION MANAGEMENT
# ============================================================
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
USER_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")
os.makedirs(USER_DATA_DIR, exist_ok=True)

def load_users():
    """Load authorized users and passwords from users.json."""
    if not os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "w") as f:
                json.dump({"admin": "admin123"}, f, indent=4)
        except Exception:
            pass
        return {"admin": "admin123"}
    try:
        with open(USERS_FILE, "r") as f:
            import json
            return json.load(f)
    except Exception:
        return {"admin": "admin123"}

import threading
thread_local = threading.local()

def get_current_username():
    """Retrieve the current logged-in username, supporting thread-local context fallback."""
    if hasattr(thread_local, "username") and thread_local.username:
        return thread_local.username
    try:
        from flask import has_request_context, session
        if has_request_context():
            val = "default"
            if val:
                return val
    except Exception:
        pass
    return "default"

user_states = {}

def load_user_state_from_disk(username):
    """Load a specific user's chat history and settings from disk."""
    state_file = os.path.join(USER_DATA_DIR, f"state_{username}.json")
    default_state = {
        "model": "ollama",
        "web_search": False,
        "system_control": True,
        "mysql_enabled": True,
        "agent_mode": False,
        "sessions": {},
        "active_session_id": "",
        "group_chat_enabled": False,
        "group_chat_models": ["gemini", "groq"],
        "chat_history": [],
        "uploaded_files": [],
        "conversation_summary": "",
        "password_verified": False,
    }
    
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                import json
                saved = json.load(f)
                # Ensure all default keys exist
                for k, v in default_state.items():
                    if k not in saved:
                        saved[k] = v
                return saved
        except Exception as e:
            print(f"[USER STATE] Error loading state for {username}: {e}")
            return default_state
    return default_state

def save_user_state_to_disk(username):
    """Save a user's state to disk."""
    if username not in user_states:
        return
    state_file = os.path.join(USER_DATA_DIR, f"state_{username}.json")
    try:
        with open(state_file, "w") as f:
            import json
            json.dump(user_states[username], f, indent=4)
    except Exception as e:
        print(f"[USER STATE] Error saving state for {username}: {e}")

def get_user_state(username):
    """Retrieve or initialize state for a logged-in user."""
    if username not in user_states:
        user_states[username] = load_user_state_from_disk(username)
    return user_states[username]

def ensure_active_session(user_state):
    """Ensure that the user state has a valid sessions dict and an active session."""
    if "sessions" not in user_state or not isinstance(user_state["sessions"], dict):
        user_state["sessions"] = {}
    
    # If there is legacy history in user_state, migrate it to a session
    if "chat_history" in user_state and user_state["chat_history"] and not user_state["sessions"]:
        import uuid
        legacy_id = str(uuid.uuid4())
        user_state["sessions"][legacy_id] = {
            "title": "Migrated Chat",
            "chat_history": user_state["chat_history"],
            "conversation_summary": user_state.get("conversation_summary", ""),
            "created_at": time.time()
        }
        user_state["active_session_id"] = legacy_id
        # Clear legacy fields so we don't migrate again
        user_state["chat_history"] = []
        user_state["conversation_summary"] = ""
    
    active_id = user_state.get("active_session_id")
    if not active_id or active_id not in user_state["sessions"]:
        import uuid
        new_id = str(uuid.uuid4())
        user_state["sessions"][new_id] = {
            "title": "New Chat",
            "chat_history": [],
            "conversation_summary": "",
            "created_at": time.time()
        }
        user_state["active_session_id"] = new_id
    
    return user_state["active_session_id"]

class SessionStateWrapper:
    """Thread-local proxy to route app_state lookups to the logged-in user."""
    def __init__(self):
        self._default_state = {
            "model": "ollama",
            "web_search": False,
            "system_control": True,
            "mysql_enabled": True,
            "agent_mode": False,
            "sessions": {},
            "active_session_id": "",
            "group_chat_enabled": False,
            "group_chat_models": ["gemini", "groq"],
            "chat_history": [],
            "uploaded_files": [],
            "conversation_summary": "",
            "password_verified": False,
        }

    def _get_current_state(self):
        username = get_current_username()
        return get_user_state(username)

    def __getitem__(self, key):
        state = self._get_current_state()
        if key in ("chat_history", "conversation_summary"):
            ensure_active_session(state)
            active_id = state["active_session_id"]
            return state["sessions"][active_id][key]
        return state[key]

    def __setitem__(self, key, value):
        username = get_current_username()
        if username != "default":
            state = get_user_state(username)
            if key in ("chat_history", "conversation_summary"):
                ensure_active_session(state)
                active_id = state["active_session_id"]
                state["sessions"][active_id][key] = value
            else:
                state[key] = value
            save_user_state_to_disk(username)
            return
        if key in ("chat_history", "conversation_summary"):
            if "sessions" not in self._default_state:
                self._default_state["sessions"] = {}
            if not self._default_state.get("active_session_id"):
                self._default_state["active_session_id"] = "default"
                self._default_state["sessions"]["default"] = {
                    "title": "Default Chat",
                    "chat_history": [],
                    "conversation_summary": "",
                    "created_at": time.time()
                }
            active_id = self._default_state["active_session_id"]
            self._default_state["sessions"][active_id][key] = value
        else:
            self._default_state[key] = value

    def __contains__(self, key):
        return key in self._get_current_state()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

app_state = SessionStateWrapper()

chatgpt_bridges = {}

def get_chatgpt_bridge(username):
    """Get or create the ChatGPT Chrome automation bridge for the user."""
    if username not in chatgpt_bridges:
        chatgpt_bridges[username] = ChatGPTBridge(chrome_version=148, profile_suffix=username)
    return chatgpt_bridges[username]

# Summary Memory Config
MEMORY_WINDOW_SIZE = 6  # Keep last 6 messages (3 user + 3 assistant exchanges)

def summarize_old_messages(llm, old_messages, existing_summary=""):
    """Summarize old messages into a compact paragraph using the LLM."""
    if not old_messages:
        return existing_summary

    conversation_text = ""
    for msg in old_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n"

    summary_prompt = f"""Progressively summarize the conversation below, adding onto the previous summary.
Return a NEW summary that captures all key information in 2-3 sentences max.

Previous summary: {existing_summary if existing_summary else '(none)'}

New conversation:
{conversation_text}

New summary:"""
    try:
        result = llm.invoke(summary_prompt)
        return result.content.strip()
    except Exception as e:
        print(f"[SUMMARY ERROR] {e}")
        # Fallback: just keep the existing summary
        return existing_summary

def get_llm(model_key=None, username=None):
    """Get LLM instance using current key from the pool."""
    if model_key is None:
        model_key = app_state["model"]
    if username is None:
        from flask import has_request_context
        username = session.get("username", "default") if has_request_context() else "default"

    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["ollama"])

    # Automation (ChatGPT)
    if config.get("type") == "automation":
        return ChatGPTAutomationLLM(bridge=get_chatgpt_bridge(username))

    # Gemini uses its own LangChain class
    if config.get("type") == "gemini":
        api_key = key_pool.get_current_key("gemini")
        if not api_key:
            raise ValueError("No Gemini API keys available. Add one via the sidebar.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config["model"],
            google_api_key=api_key,
            temperature=0,
            max_retries=2,
        )

    # All others use OpenAI-compatible API (Groq, OpenRouter, LM Studio)
    provider = config.get("key_provider")
    if provider:
        api_key = key_pool.get_current_key(provider)
        if not api_key:
            raise ValueError(f"No {provider} API keys available. Add one via the sidebar.")
    else:
        api_key = "lm-studio"

    return ChatOpenAI(
        base_url=config["base_url"],
        api_key=api_key,
        model=config["model"],
        temperature=0,
    )

@tool
def share_file_to_user(file_path: str) -> str:
    """Use this tool when the user asks you to send or share a file from the PC to their device.
    It returns a download link that the user can click to download the file."""
    import urllib.parse
    clean_path = file_path.strip("\"'")
    
    # Block direct download of the password file
    if "password.txt" in clean_path.lower():
        return "❌ Access Denied: You do not have permission to share the password file."

    if not os.path.exists(clean_path):
        return f"Could not find the file at {clean_path}"

    if not app_state.get("password_verified", False):
        return f"PASSWORD_REQUIRED_FOR_FILE_SHARE:{clean_path}"

    encoded_path = urllib.parse.quote(clean_path)
    return f"Here is the file: [Click to download {os.path.basename(clean_path)}](/api/download?path={encoded_path})"

def get_vector_store(username=None):
    """Get or create the local ChromaDB vector store isolated per user session."""
    if username is None:
        from flask import has_request_context
        username = session.get("username", "default") if has_request_context() else "default"
    user_collection = f"collection_{username}"
    return Chroma(
        collection_name=user_collection,
        embedding_function=dense_embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

def get_agent(model_key=None, username=None):
    if model_key is None:
        model_key = app_state["model"]
    if username is None:
        from flask import has_request_context
        username = session.get("username", "default") if has_request_context() else "default"

    llm = get_llm(model_key, username)
    vector_store = get_vector_store(username)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    retriever_tool = create_retriever_tool(base_retriever, "knowledge_base_search",
        "Use this tool to find information in uploaded documents and Excel files.")

    user_state = get_user_state(username) if username else app_state

    tools = [retriever_tool]
    if user_state["web_search"]:
        tools.append(DuckDuckGoSearchRun())
    if user_state["system_control"]:
        tools.append(run_cmd)
    if user_state["mysql_enabled"]:
        tools.append(mysql_query)
    tools.append(make_graph)
    tools.append(share_file_to_user)

    db_info = ""
    if app_state["mysql_enabled"]:
        db_info = """\n        6. If you think the user is asking about the database, use 'mysql_query'.
           Write proper SQL SELECT queries. The database tables are:
           - customers, employees, offices, orderdetails, orders, payments, productlines, products
           CRITICAL: If 'mysql_query' returns a Database Error, use 'knowledge_base_search' to find the answer in uploaded files instead.
        """

    if app_state["model"] == "chatgpt":
        from langchain.agents import create_react_agent
        react_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Windows System Control Expert.
            You MUST respond using this exact JSON structure:

            FOR TASKS:
            {{
              "thought": "I need to open Chrome",
              "action": "run_cmd",
              "action_input": "start chrome"
            }}

            FOR CHAT/ANSWERS:
            {{
              "thought": "The user said hi",
              "final_answer": "Hello! How can I help you today?"
            }}

            CRITICAL: The ONLY tool for Windows tasks is 'run_cmd'. 
            Always use 'start <app>' to open things.

            CRITICAL SECURITY: If any tool returns 'PASSWORD_REQUIRED_FOR_FILE_SHARE:<path>' or 'PASSWORD_REQUIRED_FOR_RUN_CMD:<path>', your final_answer MUST strictly and only be 'PASSWORD_REQUIRED:<path>' and nothing else. You do not have permission to delete files. If the user asks you to delete a file, explain that file deletion is disabled.
            
            TOOLS: [{{tool_names}}]
            {db_info}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{{input}}\n\n{{agent_scratchpad}}"),
        ])
        agent = create_react_agent(llm, tools, react_prompt)
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a smart assistant with system control and database capabilities.
            1. FIRST check 'chat_history' for context. You use Summary memory to maintain context.
            2. THEN use 'knowledge_base_search' to find answers in the uploaded files.
            3. If the user asks about a specific row or data point in an Excel file, search for the keywords in that row.
            4. If the user asks to open an application or run a system command, use the 'system_control' tool.
            5. Format your responses using Markdown for readability.
            6. ALWAYS format any code (Python, JavaScript, etc.) using standard markdown fenced code blocks (e.g. ```python ) so it renders properly in the UI.
            7. The current system date and time is: {{current_time}}
            8. If the user wants to schedule a shutdown or restart at a specific time (e.g. "at 10 PM"), calculate the number of seconds from NOW until that time, and use the system_control tool with "shutdown in X seconds".
            9. If the user asks what type of memory you use, strictly and only reply with: "Summary memory" and nothing else.
            10. If the user asks to show data as a graph/chart/plot/visualization, use the 'make_graph' tool.
                Format: graph_type|file_name|x_column|y_column|title
                Graph types: bar, line, pie, scatter, histogram, hbar
                If the tool returns GRAPH_IMAGE:..., include it in your response as: ![Chart Title](image_url)
            11. If the user asks to send, share, or download a file to their device, use the 'share_file_to_user' tool.
            12. CRITICAL SECURITY: If any tool returns a string starting with 'PASSWORD_REQUIRED_FOR_FILE_SHARE:' or 'PASSWORD_REQUIRED_FOR_RUN_CMD:', you MUST immediately stop and return exactly and only 'PASSWORD_REQUIRED:<path>' as your output. Do NOT explain anything else.
            13. You do not have permission to delete files. If the user asks you to delete any files, reply that file deletion is disabled for safety.
            {db_info}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def process_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        content = f"Source File: {os.path.basename(file_path)}\n\n"
        content += df.to_string(index=False)
        return content
    except Exception as e:
        return f"Error reading Excel: {str(e)}"

# ============================================================
# RESPONSE CLEANUP UTILITY
# ============================================================

def clean_llm_response(text):
    """
    Strip any JSON wrapper from an LLM response so the user always
    receives clean, readable natural language.

    Handles these patterns:
      1. Markdown code block:  ```json { ... } ```  or  ``` { ... } ```
      2. Plain JSON object:    { "final_answer": "..." }  (entire response is JSON)
      3. JSON embedded in surrounding text (extra sentences before/after the block)
    In all cases we extract 'final_answer' when present, or fall back to the
    full original text so nothing is lost.
    """
    import json, re

    if not isinstance(text, str):
        return text

    stripped = text.strip()

    # --- Pattern 1: markdown fenced JSON block ---
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', stripped, re.DOTALL)
    if md_match:
        candidate = md_match.group(1).strip()
        try:
            data = json.loads(candidate)
            if "final_answer" in data:
                return data["final_answer"]
            if "action" in data:
                return f"✅ Executing Task: {data['action']} with input '{data.get('action_input', '')}'"
        except Exception:
            pass
        # Could not parse — remove the code block wrapper and return plain text
        return re.sub(r'```(?:json)?\s*\{.*?\}\s*```', '', stripped, flags=re.DOTALL).strip() or stripped

    # --- Pattern 2: entire response is a JSON object ---
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            if "final_answer" in data:
                return data["final_answer"]
            if "action" in data:
                return f"✅ Executing Task: {data['action']} with input '{data.get('action_input', '')}'"
        except Exception:
            pass
        # If JSON parse fails but it still looks like raw JSON, try extracting final_answer via regex
        fa_match = re.search(r'"final_answer"\s*:\s*"(.*?)"\s*[},]', stripped, re.DOTALL)
        if fa_match:
            return fa_match.group(1).replace('\\n', '\n').replace('\\"', '"')

    # --- Pattern 3: JSON object embedded somewhere inside normal text ---
    inner_match = re.search(r'(\{[^{}]*"final_answer"[^{}]*\})', stripped, re.DOTALL)
    if inner_match:
        try:
            data = json.loads(inner_match.group(1))
            if "final_answer" in data:
                answer = data["final_answer"]
                # Stitch any surrounding text back (before + after the JSON block)
                before = stripped[:inner_match.start()].strip()
                after = stripped[inner_match.end():].strip()
                parts = [p for p in [before, answer, after] if p]
                return "\n\n".join(parts)
        except Exception:
            pass

    # Nothing matched — return original text unchanged
    return text

# ============================================================
# CUSTOM CHATGPT JSON AGENT LOOP
# ============================================================

def execute_tool_by_name(action, action_input, username=None):
    """Executes a tool by name using the corresponding python/decorator function."""
    try:
        if action == "run_cmd":
            return run_cmd.func(action_input)
        elif action == "mysql_query":
            return mysql_query.func(action_input)
        elif action == "make_graph":
            return make_graph.func(action_input)
        elif action == "share_file_to_user":
            return share_file_to_user.func(action_input)
        elif action == "knowledge_base_search":
            vector_store = get_vector_store(username)
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
            docs = base_retriever.invoke(action_input)
            if not docs:
                return "No matching documents found in knowledge base."
            return "\n\n".join([f"Source File: {doc.metadata.get('source', 'unknown')}\nContent Snippet:\n{doc.page_content}" for doc in docs])
        else:
            return f"Error: Tool '{action}' is not supported."
    except Exception as e:
        return f"Error executing tool '{action}': {str(e)}"

def run_chatgpt_agent_loop(user_message, chat_history, username=None):
    """Manages the Selenium ChatGPT prompt loop to execute JSON tools step-by-step."""
    import json
    import re
    
    if username is None:
        username = get_current_username()
    chatgpt_bridge = get_chatgpt_bridge(username)
    
    user_state = get_user_state(username)
    active_id = user_state.get("active_session_id")
    active_session = user_state["sessions"].get(active_id) if active_id else None

    def save_current_url():
        try:
            if chatgpt_bridge.driver:
                time.sleep(1)
                current_url = chatgpt_bridge.driver.current_url
                if "chatgpt.com/c/" in current_url:
                    if active_session:
                        active_session["chatgpt_url"] = current_url
                        save_user_state_to_disk(username)
                        print(f"[ChatGPT Agent Loop] Saved conversation URL: {current_url}")
        except Exception as e:
            print(f"[ChatGPT Agent Loop] Failed to save conversation URL: {e}")
    chatgpt_url = active_session.get("chatgpt_url") if active_session else None

    # Start Chrome browser session if it hasn't been started
    try:
        chatgpt_bridge.initialize()
    except Exception as e:
        print(f"[ChatGPT Agent Loop] Failed to initialize Chrome: {e}")

    # Load saved conversation URL or start a new chat session
    try:
        if chatgpt_bridge.driver:
            try:
                current_url = chatgpt_bridge.driver.current_url.strip().rstrip('/')
            except Exception:
                current_url = ""
                
            if chatgpt_url and "chatgpt.com/c/" in chatgpt_url:
                if current_url != chatgpt_url.strip().rstrip('/'):
                    print(f"[ChatGPT Agent Loop] Navigating to saved conversation URL: {chatgpt_url}")
                    chatgpt_bridge.driver.get(chatgpt_url)
                    time.sleep(3)
                else:
                    print("[ChatGPT Agent Loop] Already at the correct conversation URL. Continuing thread...")
            else:
                print("[ChatGPT Agent Loop] Starting a brand new chat...")
                chatgpt_bridge.driver.get("https://chatgpt.com/")
                time.sleep(2)
    except Exception as e:
        print(f"[ChatGPT Agent Loop] Failed to navigate: {e}")
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    db_info = ""
    if app_state["mysql_enabled"]:
        db_info = """
3. mysql_query:
   Description: Run a SQL query on the connected MySQL database and return results. Allowed tables: customers, employees, offices, orderdetails, orders, payments, productlines, products. SELECT/SHOW/DESCRIBE queries only.
   Input: SQL query string.
"""
    
    system_prompt = f"""You are a Windows System Control and RAG Expert assistant.
Current Directory: {project_dir}
Current System Time: {datetime.now().strftime('%A, %Y-%m-%d %I:%M %p')}

You have access to the following tools:
1. run_cmd:
   Description: Run a Windows shell command or open a local file/folder path.
   Input: A Windows command string OR a direct absolute path to a file/folder you want to open.
   CRITICAL: To open a file or folder, just output the path directly in `action_input` (e.g. "C:\\Users\\Ragesh.l\\Downloads\\download.jpg"). Do not use 'start ""' or extra quotes.
2. knowledge_base_search:
   Description: Search the local vector database of uploaded documents (PDF, Excel, txt, images) for relevant information.
   Input: A search query string.{db_info}
4. make_graph:
   Description: Generate charts/graphs from uploaded CSV or Excel data.
   Input: pipe-separated format: graph_type|file_name|x_column|y_column|title
5. share_file_to_user:
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
4. When using the share_file_to_user tool, you MUST include the exact markdown link returned by the tool (e.g., [Click to download filename](/api/download?path=...)) inside the "final_answer" string so the user can click it to download the file.
5. If any tool returns 'PASSWORD_REQUIRED_FOR_FILE_SHARE:<path>' or 'PASSWORD_REQUIRED_FOR_RUN_CMD:<path>', your "final_answer" MUST strictly and only be 'PASSWORD_REQUIRED:<path>' and nothing else.
6. You do not have permission to delete files. If the user asks you to delete a file, explain that file deletion is strictly disabled for safety.
"""

    # Build initial prompt with conversation context
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
        if i == 0:
            prompt_to_send = current_prompt
        else:
            prompt_to_send = f"Observation: {observation}\n\nInstructions: Continue the task and return your next tool call or final answer in the exact JSON format specified."
            
        print(f"[ChatGPT Agent Loop] Sending prompt to browser automation (Iteration {i+1})...")
        response_text = chatgpt_bridge.get_response(prompt_to_send.strip())
        print(f"[ChatGPT Agent Loop] Raw response:\n{response_text}")
        
        # Extract JSON from response if there is surrounding text
        match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if not match:
            ans = clean_llm_response(response_text)
            if shared_files:
                for sf in shared_files:
                    if sf not in ans:
                        ans += f"\n\n{sf}"
            save_current_url()
            return ans
        
        raw_json_str = match.group(1).strip()
        json_str = sanitize_json_backslashes(raw_json_str)
        res_json = {}
        try:
            res_json = json.loads(json_str)
        except Exception as e:
            print(f"[ChatGPT Agent Loop] Standard JSON parsing failed: {e}. Trying regex fallback...")
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
            print(f"[ChatGPT Agent Loop] Tool call: {action}('{action_input}')")
            
            # Execute the tool
            observation = execute_tool_by_name(action, action_input, username=username)
            print(f"[ChatGPT Agent Loop] Tool observation: {observation}")
            
            if action == "share_file_to_user" and "Click to download" in observation:
                shared_files.append(observation)
                
        elif "final_answer" in res_json:
            ans = res_json["final_answer"]
            if shared_files:
                for sf in shared_files:
                    if sf not in ans:
                        ans += f"\n\n{sf}"
            save_current_url()
            return ans
        else:
            ans = clean_llm_response(response_text)
            if shared_files:
                for sf in shared_files:
                    if sf not in ans:
                        ans += f"\n\n{sf}"
            save_current_url()
            return ans
            
    ans = "Agent loop reached maximum iterations without a final answer."
    if shared_files:
        for sf in shared_files:
            ans += f"\n\n{sf}"
    save_current_url()
    return ans

# ============================================================
# ROUTES
# ============================================================

@app.before_request
def check_authentication():
    pass

@app.route("/api/check_auth", methods=["GET"])
def check_auth():
    username = "default"
    if username:
        return jsonify({"authenticated": True, "username": username})
    return jsonify({"authenticated": False})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    
    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password are required."}), 400
        
    users = load_users()
    if username in users and users[username] == password:
        session["username"] = username
        # Pre-load/create user state
        get_user_state(username)
        return jsonify({"status": "ok", "message": f"Successfully signed in as {username}", "username": username})
        
    return jsonify({"status": "error", "message": "Invalid username or password."}), 401

@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("username", None)
    return jsonify({"status": "ok", "message": "Logged out successfully."})

@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"history": app_state["chat_history"]})

@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    username = "default"
    state = get_user_state(username)
    ensure_active_session(state)
    
    sessions_list = []
    for s_id, s_data in state["sessions"].items():
        sessions_list.append({
            "id": s_id,
            "title": s_data.get("title", "New Chat"),
            "created_at": s_data.get("created_at", 0)
        })
    sessions_list.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify({
        "sessions": sessions_list,
        "active_session_id": state["active_session_id"]
    })

@app.route("/api/sessions/new", methods=["POST"])
def create_session():
    username = "default"
    state = get_user_state(username)
    
    import uuid
    new_id = str(uuid.uuid4())
    state["sessions"][new_id] = {
        "title": "New Chat",
        "chat_history": [],
        "conversation_summary": "",
        "created_at": time.time()
    }
    state["active_session_id"] = new_id
    save_user_state_to_disk(username)
    
    return jsonify({
        "status": "ok",
        "session_id": new_id,
        "title": "New Chat"
    })

@app.route("/api/sessions/new_load", methods=["POST"])
def new_load_session():
    username = "default"
    state = get_user_state(username)
    
    active_id = state.get("active_session_id")
    if active_id and active_id in state["sessions"]:
        if not state["sessions"][active_id]["chat_history"]:
            return jsonify({
                "status": "ok",
                "session_id": active_id,
                "title": state["sessions"][active_id]["title"]
            })
            
    import uuid
    new_id = str(uuid.uuid4())
    state["sessions"][new_id] = {
        "title": "New Chat",
        "chat_history": [],
        "conversation_summary": "",
        "created_at": time.time()
    }
    state["active_session_id"] = new_id
    save_user_state_to_disk(username)
    
    return jsonify({
        "status": "ok",
        "session_id": new_id,
        "title": "New Chat"
    })

@app.route("/api/sessions/select", methods=["POST"])
def select_session():
    data = request.json or {}
    session_id = data.get("session_id")
    
    username = "default"
    state = get_user_state(username)
    
    if not session_id or session_id not in state.get("sessions", {}):
        return jsonify({"error": "Invalid session ID"}), 400
        
    state["active_session_id"] = session_id
    save_user_state_to_disk(username)
    
    return jsonify({
        "status": "ok",
        "session_id": session_id,
        "history": state["sessions"][session_id]["chat_history"]
    })

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    username = "default"
    state = get_user_state(username)
    
    if "sessions" in state and session_id in state["sessions"]:
        del state["sessions"][session_id]
        if state.get("active_session_id") == session_id:
            state["active_session_id"] = ""
            ensure_active_session(state)
        save_user_state_to_disk(username)
        return jsonify({"status": "ok", "message": "Session deleted successfully"})
        
    return jsonify({"error": "Session not found"}), 404

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/download", methods=["GET"])
def download_file():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        username = session.get("username", "default")
        state = get_user_state(username)
        active_id = ensure_active_session(state)
        active_session = state["sessions"][active_id]

        # Auto-title if it's the first message or default "New Chat"
        if not active_session.get("chat_history") or active_session.get("title", "") == "New Chat":
            title = user_message[:35] + ("..." if len(user_message) > 35 else "")
            active_session["title"] = title
            save_user_state_to_disk(username)

        # --- Summary Memory: build compact chat_history ---
        chat_history = []

        # Inject the running summary as context (if any)
        if app_state["conversation_summary"]:
            chat_history.append(HumanMessage(content=f"[Previous conversation summary: {app_state['conversation_summary']}]"))
            chat_history.append(AIMessage(content="Understood, I have the context from our previous conversation."))

        # (We strictly only pass the conversation summary context above for Summary Memory)

        # --- GROUP CHAT PARALLEL MODE ---
        if state.get("group_chat_enabled", False):
            selected_models = state.get("group_chat_models", ["gemini", "groq"])
            if not selected_models:
                return jsonify({"error": "Group Chat is enabled but no models are selected. Select models in the Settings sidebar."}), 400

            from concurrent.futures import ThreadPoolExecutor

            def query_single_model(m_key):
                thread_local.username = username
                config = MODEL_CONFIGS.get(m_key, {})
                provider = config.get("key_provider")
                m_retries = len(key_pool.pools.get(provider, [])) if provider else 1
                m_retries = max(m_retries, 1)

                last_err = None
                for attempt in range(m_retries):
                    try:
                        # Local LLMs (local/chatgpt) do not use memory history
                        m_chat_history = [] if m_key in ("ollama", "chatgpt") else chat_history
                        if m_key == "chatgpt":
                            ans = run_chatgpt_agent_loop(user_message, m_chat_history, username=username)
                        else:
                            agent_executor = get_agent(model_key=m_key, username=username)
                            response = agent_executor.invoke({
                                "input": user_message,
                                "chat_history": m_chat_history,
                                "current_time": datetime.now().strftime('%A, %Y-%m-%d %I:%M %p')
                            })
                            ans = response["output"]

                        # Clean JSON / code-block wrappers from the response
                        ans = clean_llm_response(ans)

                        return m_key, ans

                    except Exception as e:
                        last_err = str(e)
                        error_lower = last_err.lower()
                        if provider and any(kw in error_lower for kw in ["429", "quota", "rate", "exhausted", "unauthorized", "invalid", "401", "403"]):
                            old_key = key_pool.get_current_key(provider)
                            rotated = key_pool.rotate(provider)
                            new_key = key_pool.get_current_key(provider)
                            if rotated and new_key != old_key:
                                continue
                        break
                return m_key, f"⚠️ Error querying {m_key.title()}: {last_err}"

            with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
                results = list(executor.map(query_single_model, selected_models))

            # Combine output nicely
            combined_answer = "### 🤖 Group Chat Responses\n\n"
            for m_key, ans in results:
                m_name = MODEL_CONFIGS.get(m_key, {}).get("name", m_key.title())
                combined_answer += f"**{m_name}**\n{ans}\n\n---\n\n"

            # Strip trailing ruler
            combined_answer = combined_answer.rstrip("\n- ")

            # Save to active session history
            app_state["chat_history"].append({"role": "user", "content": user_message})
            app_state["chat_history"].append({"role": "assistant", "content": combined_answer})

            # Update running summary on every turn
            latest_exchange = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": combined_answer}
            ]
            try:
                # Use Gemini as a fast summarizer for the multi-model logs
                sum_llm = get_llm(model_key="gemini", username=username)
                app_state["conversation_summary"] = summarize_old_messages(
                    sum_llm, latest_exchange, app_state["conversation_summary"]
                )
                print(f"[SUMMARY MEMORY] Updated running conversation summary for Group Chat.")
            except Exception as e:
                print(f"[SUMMARY ERROR] Could not update running summary in group chat: {e}")

            save_user_state_to_disk(username)

            return jsonify({"response": combined_answer})

        # --- STANDARD SINGLE MODEL MODE ---
        model_key = app_state["model"]
        config = MODEL_CONFIGS.get(model_key, {})
        provider = config.get("key_provider")
        max_retries = len(key_pool.pools.get(provider, [])) if provider else 1
        max_retries = max(max_retries, 1)

        last_error = None
        for attempt in range(max_retries):
            try:
                # Local LLMs (local/chatgpt) do not use memory history
                m_chat_history = [] if model_key in ("ollama", "chatgpt") else chat_history
                # --- NEW: Handle ChatGPT with Custom Agent Loop ---
                if model_key == "chatgpt":
                    answer = run_chatgpt_agent_loop(user_message, m_chat_history, username=username)
                else:
                    # Standard Agent execution
                    agent_executor = get_agent()
                    response = agent_executor.invoke({
                        "input": user_message,
                        "chat_history": m_chat_history,
                        "current_time": datetime.now().strftime('%A, %Y-%m-%d %I:%M %p')
                    })
                    answer = response["output"]

                # --- CLEANUP JSON / CODE-BLOCK WRAPPERS FOR UI ---
                answer = clean_llm_response(answer)

                # Save to full history
                app_state["chat_history"].append({"role": "user", "content": user_message})
                app_state["chat_history"].append({"role": "assistant", "content": answer})

                # Update running summary on every turn
                latest_exchange = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": answer}
                ]
                try:
                    # Summarize using active model, or Gemini if using ChatGPT (to avoid slow browser calls for summary tasks)
                    sum_llm = get_llm(model_key=model_key, username=username) if model_key != "chatgpt" else get_llm(model_key="gemini", username=username)
                    app_state["conversation_summary"] = summarize_old_messages(
                        sum_llm, latest_exchange, app_state["conversation_summary"]
                    )
                    print(f"[SUMMARY MEMORY] Updated running conversation summary.")
                except Exception as e:
                    print(f"[SUMMARY ERROR] Could not update running summary: {e}")

                save_user_state_to_disk(username)
                return jsonify({"response": answer})

            except Exception as e:
                last_error = str(e)
                error_lower = last_error.lower()
                # Rotate key on quota/auth errors
                if provider and any(kw in error_lower for kw in ["429", "quota", "rate", "exhausted", "unauthorized", "invalid", "401", "403"]):
                    old_key = key_pool.get_current_key(provider)
                    rotated = key_pool.rotate(provider)
                    new_key = key_pool.get_current_key(provider)
                    if rotated and new_key != old_key:
                        print(f"[KEY ROTATION] {provider}: key {attempt+1} failed, switching to key {attempt+2}")
                        continue
                # Non-rotatable error or no more keys
                break

        return jsonify({"error": f"All {provider or model_key} keys exhausted. Last error: {last_error}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/verify_password", methods=["POST"])
def verify_password():
    data = request.json or {}
    password = data.get("password", "")
    stored = get_stored_password()
    if password == stored:
        app_state["password_verified"] = True
        return jsonify({"status": "ok", "message": "Password verified successfully!"})
    return jsonify({"status": "error", "message": "Incorrect password. Access denied."}), 401

@app.route("/api/reset_password_status", methods=["POST"])
def reset_password_status():
    app_state["password_verified"] = False
    return jsonify({"status": "ok", "message": "Files locked. Password status reset to False."})

@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        files = request.files.getlist("files")
        paste_text = request.form.get("paste_text", "").strip()
        documents = []

        for f in files:
            if f.filename.lower() == "password.txt":
                continue # Skip indexing the password file completely
            app_state["uploaded_files"].append(f.filename)
            file_path = f"./temp_{f.filename}"
            f.save(file_path)
            # Also save CSV/Excel to data_files/ for graph generation
            if f.filename.endswith((".csv", ".xlsx", ".xls")):
                import shutil
                data_copy = os.path.join(DATA_FILES_DIR, f.filename)
                shutil.copy2(file_path, data_copy)
                print(f"[DATA FILES] Saved {f.filename} to data_files/ for graphing")
            if f.filename.endswith(".pdf"):
                loaded = PyPDFLoader(file_path).load()
                for doc in loaded:
                    doc.page_content = f"Source: {f.filename}\n" + doc.page_content
                documents.extend(loaded)
            elif f.filename.endswith((".xlsx", ".xls")):
                text_data = process_excel(file_path)
                documents.append(Document(page_content=text_data, metadata={"source": f.filename}))
            elif f.filename.endswith((".jpg", ".jpeg", ".png")):
                with open(file_path, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                llm = get_llm()
                message = HumanMessage(content=[
                    {"type": "text", "text": "Describe this image in detail for search indexing."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ])
                desc = llm.invoke([message]).content
                documents.append(Document(page_content=desc, metadata={"source": f.filename}))
            else:
                documents.extend(TextLoader(file_path, encoding="utf-8").load())

        if paste_text:
            documents.append(Document(page_content=paste_text, metadata={"source": "User Paste"}))

        if not documents:
            return jsonify({"error": "No data found to save"}), 400

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        return jsonify({"message": f"Saved {len(chunks)} chunks to brain! (stored locally)", "chunks": len(chunks)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        data = request.json or {}
        if "model" in data:
            app_state["model"] = data["model"]
        if "web_search" in data:
            app_state["web_search"] = data["web_search"]
        if "system_control" in data:
            app_state["system_control"] = data["system_control"]
        if "mysql_enabled" in data:
            app_state["mysql_enabled"] = data["mysql_enabled"]
        if "agent_mode" in data:
            app_state["agent_mode"] = data["agent_mode"]
        if "group_chat_enabled" in data:
            app_state["group_chat_enabled"] = data["group_chat_enabled"]
        if "group_chat_models" in data:
            app_state["group_chat_models"] = data["group_chat_models"]
    return jsonify({
        "status": "ok",
        "settings": {
            "model": app_state["model"],
            "web_search": app_state["web_search"],
            "system_control": app_state["system_control"],
            "mysql_enabled": app_state["mysql_enabled"],
            "agent_mode": app_state["agent_mode"],
            "group_chat_enabled": app_state["group_chat_enabled"],
            "group_chat_models": app_state["group_chat_models"]
        }
    })

@app.route("/api/clear", methods=["POST"])
def clear():
    app_state["chat_history"] = []
    return jsonify({"status": "ok"})

@app.route("/api/files", methods=["GET"])
def get_files():
    return jsonify({"files": app_state["uploaded_files"]})

# ============================================================
# SHUTDOWN TIMER ROUTES
# ============================================================

def execute_power_action():
    """Called when timer expires — executes the actual shutdown/restart."""
    action = shutdown_timer["type"]
    shutdown_timer["active"] = False
    shutdown_timer["timer_obj"] = None
    if action == "shutdown":
        subprocess.Popen("shutdown /s /t 5", shell=True)
    elif action == "restart":
        subprocess.Popen("shutdown /r /t 5", shell=True)

@app.route("/api/init_chatgpt", methods=["POST"])
def init_chatgpt():
    try:
        username = session.get("username", "default")
        get_chatgpt_bridge(username).initialize()
        return jsonify({"status": "ok", "message": "Browser opened! Solve captcha if needed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/timer", methods=["GET"])
def get_timer():
    if not shutdown_timer["active"]:
        return jsonify({"active": False})
    remaining = max(0, shutdown_timer["end_time"] - time.time())
    return jsonify({
        "active": True,
        "type": shutdown_timer["type"],
        "remaining": int(remaining),
        "total": shutdown_timer["seconds"],
    })

@app.route("/api/timer", methods=["POST"])
def set_timer():
    data = request.json
    seconds = int(data.get("seconds", 60))
    action = data.get("type", "shutdown")  # "shutdown" or "restart"
    # Cancel existing timer
    if shutdown_timer["timer_obj"]:
        shutdown_timer["timer_obj"].cancel()
    shutdown_timer["active"] = True
    shutdown_timer["type"] = action
    shutdown_timer["seconds"] = seconds
    shutdown_timer["end_time"] = time.time() + seconds
    shutdown_timer["timer_obj"] = threading.Timer(seconds, execute_power_action)
    shutdown_timer["timer_obj"].daemon = True
    shutdown_timer["timer_obj"].start()
    return jsonify({"message": f"{action.title()} scheduled in {seconds} seconds", "seconds": seconds})

@app.route("/api/timer", methods=["DELETE"])
def cancel_timer():
    if shutdown_timer["timer_obj"]:
        shutdown_timer["timer_obj"].cancel()
    shutdown_timer["active"] = False
    shutdown_timer["type"] = None
    shutdown_timer["timer_obj"] = None
    # Also cancel any Windows-level pending shutdown
    subprocess.run("shutdown /a", shell=True, capture_output=True)
    return jsonify({"message": "Timer cancelled!"})

# ============================================================
# API KEY MANAGEMENT ROUTES
# ============================================================

@app.route("/api/keys", methods=["GET"])
def get_keys():
    """Return masked key info for all providers."""
    return jsonify(key_pool.get_status())

@app.route("/api/keys", methods=["POST"])
def add_key():
    """Add a new API key for a provider."""
    data = request.json
    provider = data.get("provider", "").strip().lower()
    key = data.get("key", "").strip()

    if provider not in key_pool.pools:
        return jsonify({"error": f"Unknown provider: {provider}. Use: gemini, openrouter, groq"}), 400
    if not key:
        return jsonify({"error": "API key cannot be empty"}), 400

    added = key_pool.add_key(provider, key)
    if added:
        return jsonify({"message": f"Key added to {provider}! Total keys: {len(key_pool.pools[provider])}", "status": key_pool.get_status()})
    else:
        return jsonify({"error": "Key already exists or is invalid"}), 400

@app.route("/api/models", methods=["GET"])
def list_models():
    """Return available models for the frontend."""
    models = []
    for key, config in MODEL_CONFIGS.items():
        provider = config.get("key_provider")
        has_key = True
        if provider:
            has_key = bool(key_pool.get_current_key(provider))
        models.append({
            "id": key,
            "name": config["name"],
            "available": has_key,
        })
    return jsonify({"models": models, "active": app_state["model"]})

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import webbrowser, threading
    def open_browser():
        webbrowser.open("http://localhost:5000")
    threading.Timer(1.5, open_browser).start()

    print("\n========================================")
    print("  NeuralRAG Server - Local Vector DB")
    print("========================================")
    print(f"  URL:     http://localhost:5000")
    print(f"  Vectors: {CHROMA_DB_PATH}")
    print(f"  Model:   {MODEL_CONFIGS[app_state['model']]['name']}")
    print(f"  Keys:    Gemini={len(key_pool.pools['gemini'])}, Groq={len(key_pool.pools['groq'])}, OpenRouter={len(key_pool.pools['openrouter'])}")
    print("========================================\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
