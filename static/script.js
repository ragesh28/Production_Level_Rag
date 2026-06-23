/* ============================================================
   NeuralRAG — Frontend JavaScript
   ============================================================ */

// --- DOM Elements ---
const sidebar = document.getElementById('sidebar');
const main = document.getElementById('main');
const chatArea = document.getElementById('chatArea');
const messagesContainer = document.getElementById('messages');
const welcome = document.getElementById('welcome');
const chatInput = document.getElementById('chatInput');
const btnSend = document.getElementById('btnSend');
const btnToggleSidebar = document.getElementById('btnToggleSidebar');
const btnClear = document.getElementById('btnClear');
const btnSaveBrain = document.getElementById('btnSaveBrain');
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadList = document.getElementById('uploadList');
const savedFilesList = document.getElementById('savedFilesList');
const toggleWebSearch = document.getElementById('toggleWebSearch');
const toggleSystemControl = document.getElementById('toggleSystemControl');
const toggleMySQL = document.getElementById('toggleMySQL');
const topbarModel = document.getElementById('topbarModel');
const statusDot = document.getElementById('statusDot');
const toastContainer = document.getElementById('toastContainer');

let pendingFiles = [];
let isProcessing = false;

// --- Initialize Marked.js ---
marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function (code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    }
});

// ============================================================
// SIDEBAR TOGGLE
// ============================================================
btnToggleSidebar.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
});

// ============================================================
// MODEL SELECTOR
// ============================================================
const modelButtons = document.querySelectorAll('.model-btn');
const modelNames = {
    'groq': 'Groq (Llama 3.3)',
    'openrouter': 'OpenRouter',
    'gemini': 'Gemini 2.5 Flash',
    'local': 'LM Studio (Offline)',
};

modelButtons.forEach(btn => {
    btn.addEventListener('click', () => selectModel(btn.dataset.model));
});

function selectModel(model) {
    modelButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.model === model);
    });
    topbarModel.textContent = modelNames[model] || model;
    updateSettings({ model });
}

// ============================================================
// TOGGLES
// ============================================================
toggleWebSearch.addEventListener('change', () => {
    updateSettings({ web_search: toggleWebSearch.checked });
});

toggleSystemControl.addEventListener('change', () => {
    updateSettings({ system_control: toggleSystemControl.checked });
});

toggleMySQL.addEventListener('change', () => {
    updateSettings({ mysql_enabled: toggleMySQL.checked });
});

async function updateSettings(data) {
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
    } catch (err) {
        showToast('Failed to update settings', 'error');
    }
}

// ============================================================
// CHAT INPUT
// ============================================================
chatInput.addEventListener('input', () => {
    // Auto-resize textarea
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
    // Enable/disable send button
    btnSend.disabled = !chatInput.value.trim();
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (chatInput.value.trim() && !isProcessing) sendMessage();
    }
});

btnSend.addEventListener('click', () => {
    if (chatInput.value.trim() && !isProcessing) sendMessage();
});

// ============================================================
// SUGGESTION CHIPS
// ============================================================
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        chatInput.value = chip.dataset.prompt;
        chatInput.dispatchEvent(new Event('input'));
        sendMessage();
    });
});

// ============================================================
// SEND MESSAGE
// ============================================================
async function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    isProcessing = true;
    statusDot.classList.add('busy');

    // Hide welcome, show messages
    welcome.classList.add('hidden');

    // Add user message
    addMessage('user', msg);

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    btnSend.disabled = true;

    // Add typing indicator
    const typingEl = addTypingIndicator();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg }),
        });

        const data = await res.json();
        typingEl.remove();

        if (data.error) {
            addMessage('assistant', `⚠️ Error: ${data.error}`);
            showToast(data.error, 'error');
        } else {
            addMessage('assistant', data.response);
        }
    } catch (err) {
        typingEl.remove();
        addMessage('assistant', '⚠️ Connection error. Is the server running?');
        showToast('Connection failed', 'error');
    } finally {
        isProcessing = false;
        statusDot.classList.remove('busy');
    }
}

// ============================================================
// ADD MESSAGE TO DOM
// ============================================================
function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🧠';
    const roleName = role === 'user' ? 'You' : 'NeuralRAG';

    // Render markdown for assistant messages
    let htmlContent = role === 'assistant' ? marked.parse(content) : escapeHtml(content);

    div.innerHTML = `
        <div class="message-inner">
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-role">${roleName}</div>
                <div class="message-body">${htmlContent}</div>
            </div>
        </div>
    `;

    messagesContainer.appendChild(div);
    scrollToBottom();

    // Apply syntax highlighting to any code blocks
    if (role === 'assistant') {
        div.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });

        // Style graph images — make them responsive and clickable
        div.querySelectorAll('img[src*="/static/graphs/"]').forEach(img => {
            img.style.maxWidth = '100%';
            img.style.borderRadius = '12px';
            img.style.marginTop = '10px';
            img.style.cursor = 'pointer';
            img.style.boxShadow = '0 4px 20px rgba(108, 99, 255, 0.3)';
            img.style.border = '1px solid rgba(108, 99, 255, 0.2)';
            img.addEventListener('click', () => window.open(img.src, '_blank'));
        });
    }
}

function addTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-inner">
            <div class="message-avatar">🧠</div>
            <div class="message-content">
                <div class="message-role">NeuralRAG</div>
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================================
// FILE UPLOAD
// ============================================================
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
    fileInput.value = '';
});

function handleFiles(files) {
    for (const file of files) {
        pendingFiles.push(file);
        addFileToList(file);
    }
    btnSaveBrain.disabled = pendingFiles.length === 0;
}

function addFileToList(file) {
    const iconMap = {
        'pdf': '📄', 'txt': '📝',
        'xlsx': '📊', 'xls': '📊',
        'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️',
    };
    const ext = file.name.split('.').pop().toLowerCase();
    const icon = iconMap[ext] || '📎';

    const div = document.createElement('div');
    div.className = 'upload-item';
    div.innerHTML = `
        <span class="upload-item-icon">${icon}</span>
        <span class="upload-item-name">${file.name}</span>
        <button class="upload-item-remove" title="Remove">✕</button>
    `;

    div.querySelector('.upload-item-remove').addEventListener('click', () => {
        pendingFiles = pendingFiles.filter(f => f !== file);
        div.remove();
        btnSaveBrain.disabled = pendingFiles.length === 0;
    });

    uploadList.appendChild(div);
}

async function fetchSavedFiles() {
    if (!savedFilesList) return;
    try {
        const res = await fetch('/api/files');
        const data = await res.json();
        savedFilesList.innerHTML = '';
        data.files.forEach(name => {
            const iconMap = {'pdf': '📄', 'txt': '📝', 'xlsx': '📊', 'xls': '📊', 'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️'};
            const ext = name.split('.').pop().toLowerCase();
            const icon = iconMap[ext] || '📎';
            const div = document.createElement('div');
            div.className = 'upload-item saved';
            div.innerHTML = `<span class="upload-item-icon">${icon}</span><span class="upload-item-name">${name}</span><span class="upload-badge">✓</span>`;
            savedFilesList.appendChild(div);
        });
    } catch (err) {
        console.error("Could not fetch saved files:", err);
    }
}
// Load saved files when app starts
fetchSavedFiles();

btnSaveBrain.addEventListener('click', uploadFiles);

async function uploadFiles() {
    if (pendingFiles.length === 0) return;

    btnSaveBrain.disabled = true;
    btnSaveBrain.innerHTML = '<div class="spinner"></div> Processing...';
    btnSaveBrain.classList.add('loading');

    const formData = new FormData();
    pendingFiles.forEach(f => formData.append('files', f));

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();
        if (data.error) {
            showToast(data.error, 'error');
        } else {
            showToast(data.message, 'success');
            pendingFiles = [];
            uploadList.innerHTML = '';
            fetchSavedFiles(); // Refresh the list of saved files
        }
    } catch (err) {
        showToast('Upload failed', 'error');
    } finally {
        btnSaveBrain.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a6 6 0 0 1 6 6c0 3-2 5.5-6 8.5C8 13.5 6 11 6 8a6 6 0 0 1 6-6z"/><circle cx="12" cy="8" r="2"/></svg>
            Save to Brain
        `;
        btnSaveBrain.classList.remove('loading');
        btnSaveBrain.disabled = pendingFiles.length === 0;
    }
}

// ============================================================
// CLEAR CHAT
// ============================================================
btnClear.addEventListener('click', async () => {
    try {
        await fetch('/api/clear', { method: 'POST' });
        messagesContainer.innerHTML = '';
        welcome.classList.remove('hidden');
        showToast('Chat cleared', 'info');
    } catch (err) {
        showToast('Failed to clear', 'error');
    }
});

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================
function showToast(message, type = 'info') {
    const icons = { success: '✅', error: '❌', info: '💡' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || ''}</span> ${message}`;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ============================================================
// API KEY MANAGER
// ============================================================
const keyProvider = document.getElementById('keyProvider');
const keyInput = document.getElementById('keyInput');
const btnAddKey = document.getElementById('btnAddKey');
const keyList = document.getElementById('keyList');

async function fetchKeys() {
    if (!keyList) return;
    try {
        const res = await fetch('/api/keys');
        const data = await res.json();
        keyList.innerHTML = '';
        const providerLabels = { gemini: 'Gemini', groq: 'Groq', openrouter: 'OpenRouter' };
        for (const [provider, info] of Object.entries(data)) {
            const section = document.createElement('div');
            section.className = 'key-provider-section';
            let keysHtml = '';
            info.keys_masked.forEach((k, i) => {
                const isActive = (i + 1) === info.active_index;
                keysHtml += `<div class="key-item ${isActive ? 'active' : ''}">
                    <span class="key-dot ${isActive ? 'active' : ''}"></span>
                    <span class="key-masked">${k}</span>
                    ${isActive ? '<span class="key-active-badge">ACTIVE</span>' : ''}
                </div>`;
            });
            section.innerHTML = `
                <div class="key-provider-header">
                    <span>${providerLabels[provider] || provider}</span>
                    <span class="key-count">${info.total} key${info.total !== 1 ? 's' : ''}</span>
                </div>
                ${keysHtml || '<div class="key-item empty">No keys added</div>'}
            `;
            keyList.appendChild(section);
        }
    } catch (err) {
        console.error("Could not fetch keys:", err);
    }
}

if (btnAddKey) {
    btnAddKey.addEventListener('click', async () => {
        const provider = keyProvider.value;
        const key = keyInput.value.trim();
        if (!key) { showToast('Please paste an API key', 'error'); return; }

        btnAddKey.disabled = true;
        try {
            const res = await fetch('/api/keys', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, key }),
            });
            const data = await res.json();
            if (data.error) {
                showToast(data.error, 'error');
            } else {
                showToast(data.message, 'success');
                keyInput.value = '';
                fetchKeys();
            }
        } catch (err) {
            showToast('Failed to add key', 'error');
        } finally {
            btnAddKey.disabled = false;
        }
    });
}

// Load keys on startup
fetchKeys();

// Auto-select gemini on load
selectModel('gemini');

// ============================================================
// SHUTDOWN TIMER
// ============================================================
const timerBanner = document.getElementById('timerBanner');
const timerControls = document.getElementById('timerControls');
const timerActive = document.getElementById('timerActive');
const timerLabel = document.getElementById('timerLabel');
const timerCountdown = document.getElementById('timerCountdown');
const timerBar = document.getElementById('timerBar');
const timerCancelBtn = document.getElementById('timerCancelBtn');
const timerStartBtn = document.getElementById('timerStartBtn');
const timerAction = document.getElementById('timerAction');
const timerHours = document.getElementById('timerHours');
const timerMinutes = document.getElementById('timerMinutes');

function formatTime(totalSec) {
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    if (h > 0) return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function showCountdown() {
    timerBanner.style.display = '';
    timerActive.style.display = '';
}

function hideTimer() {
    timerBanner.style.display = 'none';
    timerActive.style.display = 'none';
}

async function pollTimer() {
    try {
        const res = await fetch('/api/timer');
        const data = await res.json();
        if (data.active) {
            showCountdown();
            const actionLabel = data.type === 'restart' ? 'Restart' : 'Shutdown';
            timerLabel.textContent = `${actionLabel} in`;
            timerCountdown.textContent = formatTime(data.remaining);
            const pct = data.total > 0 ? (data.remaining / data.total) * 100 : 0;
            timerBar.style.width = pct + '%';
            if (data.remaining <= 10) {
                timerCountdown.style.color = '#ef4444';
                timerCountdown.style.animation = 'pulse 0.5s infinite';
            } else {
                timerCountdown.style.color = '';
                timerCountdown.style.animation = '';
            }
        } else {
            hideTimer();
        }
    } catch (e) { /* ignore */ }
}

// Manual start button
if (timerStartBtn) {
    timerStartBtn.addEventListener('click', async () => {
        const action = timerAction.value;
        const h = parseInt(timerHours.value) || 0;
        const m = parseInt(timerMinutes.value) || 0;
        const seconds = (h * 3600) + (m * 60);

        if (action === 'sleep') {
            // Sleep is instant — no timer needed
            try {
                await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: 'sleep' }),
                });
                showToast('Putting PC to sleep...', 'info');
            } catch (e) { showToast('Failed', 'error'); }
            return;
        }

        if (seconds < 10) {
            showToast('Set at least 10 seconds (1 minute recommended)', 'error');
            return;
        }

        try {
            const res = await fetch('/api/timer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ seconds, type: action }),
            });
            const data = await res.json();
            showToast(data.message, 'success');
            pollTimer();
        } catch (e) {
            showToast('Failed to start timer', 'error');
        }
    });
}

// Cancel button
if (timerCancelBtn) {
    timerCancelBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/timer', { method: 'DELETE' });
            hideTimer();
            showToast('Timer cancelled!', 'success');
        } catch (e) {
            showToast('Failed to cancel', 'error');
        }
    });
}

// Poll timer every second
setInterval(pollTimer, 1000);
pollTimer();
