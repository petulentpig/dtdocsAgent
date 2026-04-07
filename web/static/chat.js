const chatEl = document.getElementById("chat");
const form = document.getElementById("input-form");
const input = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");

let sessionId = localStorage.getItem("dt_session_id");
if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("dt_session_id", sessionId);
}

function addMessage(role, content, sources) {
    const msg = document.createElement("div");
    msg.className = `message ${role}`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "content";

    if (role === "assistant") {
        contentDiv.innerHTML = marked.parse(content);
    } else {
        contentDiv.textContent = content;
    }

    msg.appendChild(contentDiv);

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement("div");
        sourcesDiv.className = "sources";
        sourcesDiv.innerHTML = `<div class="sources-title">Sources</div>` +
            sources.map(s => `<a href="${s.url}" target="_blank">${s.title || s.url}</a>`).join("");
        msg.appendChild(sourcesDiv);
    }

    chatEl.appendChild(msg);
    chatEl.scrollTop = chatEl.scrollHeight;
}

function addLoading() {
    const msg = document.createElement("div");
    msg.className = "message assistant";
    msg.id = "loading-msg";

    const contentDiv = document.createElement("div");
    contentDiv.className = "content";
    contentDiv.innerHTML = `<div class="loading"><span></span><span></span><span></span></div>`;

    msg.appendChild(contentDiv);
    chatEl.appendChild(msg);
    chatEl.scrollTop = chatEl.scrollHeight;
}

function removeLoading() {
    const el = document.getElementById("loading-msg");
    if (el) el.remove();
}

async function sendMessage(message) {
    addMessage("user", message);
    input.value = "";
    sendBtn.disabled = true;
    addLoading();

    try {
        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId }),
        });

        removeLoading();

        if (!resp.ok) {
            const err = await resp.text();
            addMessage("assistant", `Error: ${err}`);
            return;
        }

        const data = await resp.json();
        sessionId = data.session_id;
        localStorage.setItem("dt_session_id", sessionId);
        addMessage("assistant", data.answer, data.sources);
    } catch (err) {
        removeLoading();
        addMessage("assistant", `Connection error: ${err.message}`);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

form.addEventListener("submit", (e) => {
    e.preventDefault();
    const msg = input.value.trim();
    if (msg) sendMessage(msg);
});
