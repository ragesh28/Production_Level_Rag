import re
import os

file_path = "server.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
content = content.replace(
    "import mysql.connector",
    "try:\n    import mysql.connector\n    MYSQL_AVAILABLE = True\nexcept ImportError:\n    MYSQL_AVAILABLE = False"
)

content = content.replace(
    "from chatgpt_automation import ChatGPTBridge, ChatGPTAutomationLLM",
    "try:\n    from chatgpt_automation import ChatGPTBridge, ChatGPTAutomationLLM\nexcept ImportError:\n    ChatGPTBridge = None\n    ChatGPTAutomationLLM = None"
)

# 2. MODEL_CONFIGS
ollama_config = """    "ollama": {
        "name": "Ollama (Qwen 2.5 7B)",
        "base_url": "http://127.0.0.1:11434/v1",
        "key_provider": None,
        "model": "qwen2.5:7b",
    },"""
content = re.sub(
    r'"chatgpt": \{.*?"model": "chatgpt-browser",\s*\},',
    ollama_config,
    content,
    flags=re.DOTALL
)

# 3. Replace session.get("username") with "default"
content = content.replace('session.get("username")', '"default"')

# 4. Remove check_authentication and login routes
# We just replace the before_request to do nothing
content = re.sub(
    r'@app\.before_request\s*def check_authentication\(\):.*?return jsonify\(\{"error": "Unauthorized. Please log in first."\}\), 401',
    '@app.before_request\ndef check_authentication():\n    pass',
    content,
    flags=re.DOTALL
)

# 5. Disable run_cmd on Linux
content = content.replace(
    'def run_cmd(command: str) -> str:',
    'def run_cmd(command: str) -> str:\n    import platform\n    if platform.system() == "Linux":\n        return "❌ System control is disabled in cloud environments."\n'
)

# 6. Change default model to ollama
content = content.replace('"local"', '"ollama"')

# Save
with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patched server.py successfully!")
