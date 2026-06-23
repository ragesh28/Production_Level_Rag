import re

file_path = "server.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace MODEL_CONFIGS to only have ollama
ollama_only_configs = """MODEL_CONFIGS = {
    "ollama": {
        "name": "Ollama (Qwen 2.5 7B)",
        "base_url": "http://127.0.0.1:11434/v1",
        "key_provider": None,
        "model": "qwen2.5:7b",
    }
}"""
content = re.sub(r'MODEL_CONFIGS = \{.*?\}\n', ollama_only_configs + '\n', content, flags=re.DOTALL)

# Default to ollama directly if there are any remaining references
content = content.replace('"local"', '"ollama"')

# Save
with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patched server.py successfully for ollama-only!")
