import re

file_path = r"static\index.html"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove login overlay modal
# The login overlay is usually in a div with id="loginOverlay" or similar.
# Let's just remove the entire login UI block by looking for NeuralRAG Access
login_ui_pattern = r'<div id="loginOverlay".*?<!-- PASSWORD MODAL -->'
content = re.sub(login_ui_pattern, '<!-- PASSWORD MODAL -->', content, flags=re.DOTALL)

# But wait, looking at the grep earlier:
# It starts with: `<div style="position: fixed; inset: 0; background: var(--bg-deep); z-index: 9999; display: flex;`
# So we can remove the JS logic that shows it instead.
content = content.replace('document.getElementById("loginOverlay").style.display = "flex";', '')
content = content.replace('document.getElementById("loginOverlay").style.display = "none";', '')

# 2. Replace ChatGPT with Ollama in the model dropdown
content = content.replace(
    '<div class="model-dropdown-item" data-model="chatgpt">\n                                    <span>Local LLM 2 (ChatGPT)</span>\n                                    <span class="model-tag local">🚀 Web</span>\n                                </div>',
    '<div class="model-dropdown-item" data-model="ollama">\n                                    <span>Ollama (Qwen 2.5 7B)</span>\n                                    <span class="model-tag local">🧠 Local</span>\n                                </div>'
)

# 3. Remove chatgpt from JS maps
content = content.replace(
    "'chatgpt': 'Local LLM 2 (ChatGPT)',",
    "'ollama': 'Ollama (Qwen 2.5 7B)',"
)

# 4. Remove ChatGPT Browser Control section
chatgpt_ctrl = r'<!-- ChatGPT Browser Control -->.*?</div>\s*</div>\s*</div>'
content = re.sub(chatgpt_ctrl, '<!-- ChatGPT Removed for Cloud -->', content, flags=re.DOTALL)

# 5. Remove ChatGPT from Group Chat checkboxes
content = re.sub(
    r'<input type="checkbox" class="group-model-checkbox" value="chatgpt".*?Local LLM 2 \(ChatGPT\)',
    '',
    content,
    flags=re.DOTALL
)

# 6. JS Auth Check bypass
content = content.replace(
    'const res = await fetch(\'/api/check_auth\');',
    '// Bypassed check_auth\n            const res = { ok: true, json: async () => ({ authenticated: true }) };'
)

# 7. Update the active model pill display
content = content.replace(
    '<span class="model-name" id="modelPillName">Gemini</span>',
    '<span class="model-name" id="modelPillName">Ollama</span>'
)

# Save
with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patched index.html successfully!")
