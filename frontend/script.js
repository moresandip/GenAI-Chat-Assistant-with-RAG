// Frontend State Controller
let state = {
    token: null,
    username: null,
    activeSessionId: null,
    sessions: [],
    documents: [],
    // Map to store message indexes mapping to their RAG metadata
    messageMetadataMap: {} 
};

try {
    state.token = localStorage.getItem("accessToken") || null;
    state.username = localStorage.getItem("username") || null;
} catch (e) {
    console.warn("localStorage is not available, falling back to in-memory state.", e);
}

// API Endpoint Constants
const API_BASE = window.location.origin;

// Initialize Application on Load
window.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function initApp() {
    if (state.token) {
        showDashboard();
    } else {
        showAuthScreen();
    }
}

// --- VIEW CONTROLLERS ---

function showAuthScreen() {
    document.getElementById("auth-screen").classList.remove("hidden");
    document.getElementById("dashboard-screen").classList.add("hidden");
    switchAuthTab("login");
}

// Fixed Login / Navigation issue: after enter user & pass, navigate into dashboard
function showDashboard() {
    document.getElementById("auth-screen").classList.add("hidden");
    document.getElementById("dashboard-screen").classList.remove("hidden");
    
    // Set username display
    document.getElementById("username-display").innerText = state.username;
    document.getElementById("user-avatar-initials").innerText = state.username ? state.username.substring(0, 2).toUpperCase() : "U";
    
    // Fetch Sidebar Metadata
    loadSessions();
    loadDocuments();
    
    // Start with a clean slate
    startNewChat();
}

function switchAuthTab(tab) {
    const loginForm = document.getElementById("login-form");
    const registerForm = document.getElementById("register-form");
    const tabLogin = document.getElementById("tab-login");
    const tabRegister = document.getElementById("tab-register");

    if (tab === "login") {
        loginForm.classList.remove("hidden");
        registerForm.classList.add("hidden");
        tabLogin.classList.add("active");
        tabRegister.classList.remove("active");
    } else {
        loginForm.classList.add("hidden");
        registerForm.classList.remove("hidden");
        tabLogin.classList.remove("active");
        tabRegister.classList.add("active");
    }
}

// --- AUTHENTICATION ACTIONS ---

async function handleAuthSubmit(event, mode) {
    event.preventDefault();
    
    const usernameInput = mode === "login" ? "login-username" : "reg-username";
    const passwordInput = mode === "login" ? "login-password" : "reg-password";
    
    const username = document.getElementById(usernameInput).value.trim();
    const password = document.getElementById(passwordInput).value;

    const endpoint = mode === "login" ? "/api/auth/login" : "/api/auth/register";
    const payload = { username, password };

    try {
        const response = await fetch(API_BASE + endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail?.error || data.error || `Authentication failed: Code ${response.status}`);
        }

        if (mode === "register") {
            showToast("Registration successful! Logging you in...", "success");
            
            // Automatically log in using the newly created credentials
            const loginResponse = await fetch(API_BASE + "/api/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const loginData = await loginResponse.json();
            if (!loginResponse.ok) {
                throw new Error(loginData.detail?.error || loginData.error || "Auto-login failed");
            }
            
            try {
                localStorage.setItem("accessToken", loginData.access_token);
                localStorage.setItem("username", username);
            } catch (e) {
                console.warn("localStorage not accessible:", e);
            }
            state.token = loginData.access_token;
            state.username = username;
            
            showToast("Logged in successfully!", "success");
            showDashboard();
            
            // Clear inputs
            document.getElementById("reg-username").value = "";
            document.getElementById("reg-password").value = "";
        } else {
            // Login Successful
            try {
                localStorage.setItem("accessToken", data.access_token);
                localStorage.setItem("username", username);
            } catch (e) {
                console.warn("localStorage not accessible:", e);
            }
            state.token = data.access_token;
            state.username = username;
            
            showToast("Logged in successfully!", "success");
            showDashboard();
            
            // Clear inputs
            document.getElementById("login-username").value = "";
            document.getElementById("login-password").value = "";
        }

    } catch (err) {
        showToast(err.message, "error");
    }
}

function handleLogout() {
    try {
        localStorage.removeItem("accessToken");
        localStorage.removeItem("username");
    } catch (e) {
        console.warn("localStorage not accessible:", e);
    }
    state.token = null;
    state.username = null;
    state.activeSessionId = null;
    state.messageMetadataMap = {};
    
    showToast("Signed out successfully", "info");
    showAuthScreen();
}

// --- CHAT SESSION MANAGEMENT ---

function startNewChat() {
    state.activeSessionId = generateUUID();
    state.messageMetadataMap = {};
    
    // Reset view
    const messageContainer = document.getElementById("chat-messages");
    messageContainer.innerHTML = `
        <div class="welcome-card">
            <div class="welcome-icon">⚡</div>
            <h3>Welcome to GenAI Chat Assistant with RAG Support Portal</h3>
            <p>Ask technical questions about our GPU resources, data structures, GitLab container integrations, or Triton model serving limits. Your answers are mathematically grounded in our verified documents.</p>
            
            <div class="sample-prompts">
                <button class="sample-prompt-btn" onclick="useSamplePrompt('How do I allocate GPU resources for training?')">
                    "How do I allocate GPU resources for training?"
                </button>
                <button class="sample-prompt-btn" onclick="useSamplePrompt('What are the MFA and password rules?')">
                    "What are the MFA and password rules?"
                </button>
                <button class="sample-prompt-btn" onclick="useSamplePrompt('How to deploy models on Triton?')">
                    "How to deploy models on Triton?"
                </button>
            </div>
        </div>
    `;
    
    document.getElementById("active-chat-title").innerText = "New Conversation";
    
    // Clear sidebar active states
    document.querySelectorAll(".session-card").forEach(card => card.classList.remove("active"));
    
    // Reset Debug Panel
    resetInspectorPanel();
}

function useSamplePrompt(text) {
    document.getElementById("chat-input").value = text;
    document.getElementById("chat-input").focus();
}

async function loadSessions() {
    const listContainer = document.getElementById("sessions-list");
    
    try {
        const response = await fetch(API_BASE + "/api/chat/sessions", {
            headers: getAuthHeaders()
        });

        if (response.status === 401) {
            handleLogout();
            return;
        }

        if (!response.ok) throw new Error("Failed to load sessions");

        const sessions = await response.json();
        state.sessions = sessions;

        if (sessions.length === 0) {
            listContainer.innerHTML = `<div class="sessions-loading">No active chat sessions.</div>`;
            return;
        }

        listContainer.innerHTML = "";
        sessions.forEach(session => {
            const card = document.createElement("div");
            card.className = `session-card ${session.id === state.activeSessionId ? 'active' : ''}`;
            card.id = `session-${session.id}`;
            card.onclick = () => selectSession(session.id, session.title);
            
            card.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                <span>${escapeHtml(session.title)}</span>
            `;
            listContainer.appendChild(card);
        });

    } catch (err) {
        listContainer.innerHTML = `<div class="sessions-loading" style="color: var(--error)">Error loading history</div>`;
    }
}

async function selectSession(sessionId, title) {
    if (state.activeSessionId === sessionId) return;
    
    state.activeSessionId = sessionId;
    state.messageMetadataMap = {};
    resetInspectorPanel();

    // Mark active in list UI
    document.querySelectorAll(".session-card").forEach(card => card.classList.remove("active"));
    const activeCard = document.getElementById(`session-${sessionId}`);
    if (activeCard) activeCard.classList.add("active");

    document.getElementById("active-chat-title").innerText = title;
    
    const messageContainer = document.getElementById("chat-messages");
    messageContainer.innerHTML = `<div class="sessions-loading">Loading message logs...</div>`;

    try {
        const response = await fetch(`${API_BASE}/api/chat/history?sessionId=${sessionId}`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) throw new Error("Failed to load messages history");

        const messages = await response.json();
        messageContainer.innerHTML = "";
        
        if (messages.length === 0) {
            messageContainer.innerHTML = `<div class="sessions-loading">Empty history.</div>`;
            return;
        }

        messages.forEach((msg, idx) => {
            const isUser = msg.role === "user";
            
            const msgWrapper = document.createElement("div");
            msgWrapper.className = `message-wrapper ${isUser ? 'user' : 'assistant'}`;
            
            const msgBubble = document.createElement("div");
            msgBubble.className = "message-bubble";
            msgBubble.innerHTML = isUser ? escapeHtml(msg.content) : parseMarkdown(msg.content);
            
            msgWrapper.appendChild(msgBubble);
            
            if (!isUser) {
                const metaDiv = document.createElement("div");
                metaDiv.className = "message-meta";
                
                let metaText = `<span class="message-meta-tag">Tokens: ${msg.tokens_used}</span>`;
                if (msg.retrieved_chunks_count > 0) {
                    metaText += `<span class="message-meta-tag">Sources: ${msg.retrieved_chunks_count}</span>`;
                    
                    msgWrapper.style.cursor = "pointer";
                    msgWrapper.title = "Click to inspect vector similarity metrics";
                    
                    state.messageMetadataMap[idx] = {
                        tokensUsed: msg.tokens_used,
                        retrievedChunks: msg.retrieved_chunks_count,
                        sourceDocuments: msg.similarity_scores.map((score, sIdx) => ({
                            chunk_id: sIdx,
                            document_title: "Retrieved Source Document",
                            text: `Content is saved in vector store. Score: ${score}`,
                            similarity: score
                        }))
                    };
                    msgWrapper.onclick = () => renderInspectorData(state.messageMetadataMap[idx]);
                }
                metaDiv.innerHTML = metaText;
                msgWrapper.appendChild(metaDiv);
                
                if (idx === messages.length - 1 && msg.retrieved_chunks_count > 0) {
                    renderInspectorData(state.messageMetadataMap[idx]);
                }
            }

            messageContainer.appendChild(msgWrapper);
        });
        
        scrollChatToBottom();

    } catch (err) {
        showToast(err.message, "error");
        messageContainer.innerHTML = `<div class="sessions-loading" style="color: var(--error)">Failed to retrieve messages.</div>`;
    }
}

// --- SEND & PROCESS MESSAGE ---

async function handleSendMessage(event) {
    event.preventDefault();
    
    const inputField = document.getElementById("chat-input");
    const query = inputField.value.trim();
    if (!query) return;

    // Append User bubble to screen immediately
    appendMessage("user", query);
    inputField.value = "";
    
    // Show loading spinner state
    const typingIndicator = document.getElementById("typing-indicator");
    const sendBtn = document.getElementById("send-btn");
    typingIndicator.classList.remove("hidden");
    sendBtn.disabled = true;
    
    scrollChatToBottom();

    try {
        const payload = {
            sessionId: state.activeSessionId,
            message: query
        };

        const response = await fetch(API_BASE + "/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...getAuthHeaders()
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.status === 401) {
            handleLogout();
            return;
        }

        if (!response.ok) {
            throw new Error(data.detail?.error || data.error || "RAG query processing failed");
        }

        // Append assistant response
        appendMessage("assistant", data.reply, data);

        // Update Inspector panel
        renderInspectorData(data);

        // Refresh sidebar sessions list
        loadSessions();

    } catch (err) {
        showToast(err.message, "error");
        appendMessage("assistant", "⚠️ An error occurred while retrieving answer from services: " + err.message);
    } finally {
        typingIndicator.classList.add("hidden");
        sendBtn.disabled = false;
        scrollChatToBottom();
    }
}

function appendMessage(role, content, metadata = null) {
    const messageContainer = document.getElementById("chat-messages");
    
    // Clear welcome card if present
    const welcome = messageContainer.querySelector(".welcome-card");
    if (welcome) welcome.remove();

    const isUser = role === "user";
    const msgWrapper = document.createElement("div");
    msgWrapper.className = `message-wrapper ${isUser ? 'user' : 'assistant'}`;
    
    const msgBubble = document.createElement("div");
    msgBubble.className = "message-bubble";
    msgBubble.innerHTML = isUser ? escapeHtml(content) : parseMarkdown(content);
    msgWrapper.appendChild(msgBubble);

    if (!isUser && metadata) {
        const metaDiv = document.createElement("div");
        metaDiv.className = "message-meta";
        
        let metaText = `<span class="message-meta-tag">Tokens: ${metadata.tokensUsed}</span>`;
        if (metadata.retrievedChunks > 0) {
            metaText += `<span class="message-meta-tag">Sources: ${metadata.retrievedChunks}</span>`;
            
            msgWrapper.style.cursor = "pointer";
            msgWrapper.title = "Click to inspect vector similarity metrics";
            msgWrapper.onclick = () => renderInspectorData(metadata);
        }
        metaDiv.innerHTML = metaText;
        msgWrapper.appendChild(metaDiv);
    }

    messageContainer.appendChild(msgWrapper);
    scrollChatToBottom();
}

// --- RAG DEBUG INSPECTOR RENDERER ---

function resetInspectorPanel() {
    document.getElementById("metric-chunks-count").innerText = "0";
    document.getElementById("metric-tokens-used").innerText = "0";
    
    document.getElementById("similarity-scores-container").innerHTML = `
        <div class="inspector-placeholder-text">Ask a question to view matching scores.</div>
    `;
    document.getElementById("source-chunks-container").innerHTML = `
        <div class="inspector-placeholder-text">Retrieved text chunks will render here.</div>
    `;
}

function renderInspectorData(data) {
    document.getElementById("metric-chunks-count").innerText = data.retrievedChunks;
    document.getElementById("metric-tokens-used").innerText = data.tokensUsed;

    const scoreContainer = document.getElementById("similarity-scores-container");
    scoreContainer.innerHTML = "";

    const chunksContainer = document.getElementById("source-chunks-container");
    chunksContainer.innerHTML = "";

    if (!data.sourceDocuments || data.sourceDocuments.length === 0) {
        scoreContainer.innerHTML = `<div class="inspector-placeholder-text">No matches found (fallback response triggered).</div>`;
        chunksContainer.innerHTML = `<div class="inspector-placeholder-text">No vectors fetched.</div>`;
        return;
    }

    data.sourceDocuments.forEach((doc, idx) => {
        const pct = Math.round(doc.similarity * 100);
        
        const meterCard = document.createElement("div");
        meterCard.className = "score-meter-card";
        meterCard.innerHTML = `
            <div class="score-meter-info">
                <span class="score-meter-title" title="${escapeHtml(doc.document_title)}">Chunk #${doc.chunk_id}: ${escapeHtml(doc.document_title)}</span>
                <span class="score-meter-percentage">${pct}%</span>
            </div>
            <div class="score-progress-bar-bg">
                <div class="score-progress-bar-fill" style="width: ${pct}%"></div>
            </div>
        `;
        scoreContainer.appendChild(meterCard);

        const snippetCard = document.createElement("div");
        snippetCard.className = "chunk-snippet-card";
        snippetCard.innerHTML = `
            <div class="chunk-snippet-header">
                <span>Doc: ${escapeHtml(doc.document_title)}</span>
                <span style="color: var(--accent-violet)">Score: ${doc.similarity.toFixed(4)}</span>
            </div>
            <div class="chunk-snippet-body">
                ${escapeHtml(doc.text)}
            </div>
        `;
        chunksContainer.appendChild(snippetCard);
    });
}

// --- ACTIVE DOCUMENTS SIDEBAR LOADING ---

async function loadDocuments() {
    const list = document.getElementById("documents-list");
    list.innerHTML = `<div class="kb-loading">Querying active list...</div>`;
    
    try {
        const response = await fetch(API_BASE + "/api/documents", {
            headers: getAuthHeaders()
        });

        if (!response.ok) throw new Error("Failed to load documents catalog");
        
        const docs = await response.json();
        state.documents = docs;

        if (docs.length === 0) {
            list.innerHTML = `<div class="kb-loading">No documents indexed. Click refresh to index docs.json.</div>`;
            return;
        }

        list.innerHTML = "";
        docs.forEach(doc => {
            const item = document.createElement("div");
            item.className = "doc-item";
            item.innerHTML = `
                <div class="doc-info">
                    <div class="doc-title" title="${escapeHtml(doc.title)}">${escapeHtml(doc.title)}</div>
                    <div class="doc-meta">${doc.char_count} chars • ${doc.chunks} chunks</div>
                </div>
                <div class="doc-badge">Indexed</div>
            `;
            list.appendChild(item);
        });

    } catch (err) {
        list.innerHTML = `<div class="kb-loading" style="color: var(--error)">Failed to load.</div>`;
    }
}

async function reloadKnowledgeBase() {
    showToast("Re-indexing docs.json into vectors...", "info");
    
    try {
        const response = await fetch(API_BASE + "/api/index/reload", {
            method: "POST",
            headers: getAuthHeaders()
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail?.error || data.error || "Re-indexing action failed");
        }

        showToast(`Indexed ${data.documentsIndexed} documents into ${data.chunksCreated} chunks!`, "success");
        loadDocuments();
        startNewChat();

    } catch (err) {
        showToast(err.message, "error");
    }
}

// --- UTILITY CLIENT-SIDE HELPERS ---

function getAuthHeaders() {
    return {
        "Authorization": `Bearer ${state.token}`
    };
}

function generateUUID() {
    try {
        return crypto.randomUUID();
    } catch (e) {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
}

function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function scrollChatToBottom() {
    const el = document.getElementById("chat-messages");
    el.scrollTop = el.scrollHeight;
}

function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    toast.innerHTML = `
        <span>${escapeHtml(message)}</span>
        <span style="margin-left: 10px; cursor: pointer; opacity: 0.6;" onclick="this.parentElement.remove()">✕</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(120%)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}
