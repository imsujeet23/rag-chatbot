const componentStatus = document.querySelector("#componentStatus");
const indexStats = document.querySelector("#indexStats");
const refreshStatus = document.querySelector("#refreshStatus");
const uploadForm = document.querySelector("#uploadForm");
const documentFile = document.querySelector("#documentFile");
const fileName = document.querySelector("#fileName");
const uploadButton = document.querySelector("#uploadButton");
const uploadMessage = document.querySelector("#uploadMessage");
const chatForm = document.querySelector("#chatForm");
const chatMessages = document.querySelector("#chatMessages");
const questionInput = document.querySelector("#questionInput");
const sendButton = document.querySelector("#sendButton");
const topKInput = document.querySelector("#topK");
const maxLengthInput = document.querySelector("#maxLength");
const clearIndexButton = document.querySelector("#clearIndex");

function clampNumber(value, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) return min;
  return Math.min(Math.max(parsed, min), max);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message = typeof payload === "string" ? payload : payload.detail || "Request failed";
    throw new Error(message);
  }

  return payload;
}

function setMessage(element, text, type = "") {
  element.textContent = text;
  element.className = `form-message ${type}`.trim();
}

function renderComponents(components = {}) {
  componentStatus.innerHTML = Object.entries(components)
    .map(([name, ready]) => {
      const className = ready ? "ready" : "offline";
      const label = ready ? "Ready" : "Offline";
      return `
        <div class="status-item">
          <span>${escapeHtml(name)}</span>
          <strong class="${className}">${label}</strong>
        </div>
      `;
    })
    .join("");
}

function renderStats(stats = {}) {
  const items = [
    ["Chunks", stats.total_chunks ?? 0],
    ["Index size", stats.index_size ?? 0],
    ["Embedding dim", stats.embedding_dim ?? "-"],
  ];

  indexStats.innerHTML = items
    .map(([label, value]) => `
      <div class="stat-item">
        <span>${label}</span>
        <strong>${escapeHtml(value)}</strong>
      </div>
    `)
    .join("");
}

async function refreshDashboard() {
  refreshStatus.disabled = true;
  try {
    const [health, stats] = await Promise.all([
      requestJson("/health"),
      requestJson("/index-stats"),
    ]);
    renderComponents(health.components);
    renderStats(stats);
  } catch (error) {
    componentStatus.innerHTML = `
      <div class="status-item">
        <span>API</span>
        <strong class="offline">${escapeHtml(error.message)}</strong>
      </div>
    `;
  } finally {
    refreshStatus.disabled = false;
  }
}

function addMessage(role, content, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<p>${escapeHtml(content)}</p>`;

  if (sources.length > 0) {
    const sourceMarkup = sources
      .map((source, index) => {
        const document = source.document || "Document";
        const score = source.similarity_score ? `${Math.round(source.similarity_score * 100)}% match` : "source";
        return `
          <details class="source">
            <summary>${escapeHtml(document)} - Chunk ${escapeHtml(source.chunk_index ?? index + 1)} - ${score}</summary>
            <pre class="source-content">${escapeHtml(source.content || "")}</pre>
          </details>
        `;
      })
      .join("");
    bubble.insertAdjacentHTML("beforeend", `<div class="sources">${sourceMarkup}</div>`);
  }

  article.append(avatar, bubble);
  chatMessages.appendChild(article);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return article;
}

documentFile.addEventListener("change", () => {
  fileName.textContent = documentFile.files[0]?.name || "Choose a file to index";
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = documentFile.files[0];
  if (!file) {
    setMessage(uploadMessage, "Choose a PDF or TXT file first.", "error-text");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadButton.disabled = true;
  setMessage(uploadMessage, "Indexing document...");

  try {
    const result = await requestJson("/upload", {
      method: "POST",
      body: formData,
    });
    setMessage(uploadMessage, `${result.filename} indexed with ${result.chunks_created} chunks.`, "success-text");
    uploadForm.reset();
    fileName.textContent = "Choose a file to index";
    await refreshDashboard();
  } catch (error) {
    setMessage(uploadMessage, error.message, "error-text");
  } finally {
    uploadButton.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  const topK = clampNumber(topKInput.value, 1, 20);
  const maxLength = clampNumber(maxLengthInput.value, 64, 1024);

  addMessage("user", question);
  const pendingMessage = addMessage("assistant", "Thinking through the indexed context...");
  questionInput.value = "";
  sendButton.disabled = true;

  try {
    const result = await requestJson("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, top_k: topK, max_length: maxLength }),
    });

    pendingMessage.remove();
    addMessage("assistant", result.answer || "I could not generate an answer.", result.source_chunks || []);
  } catch (error) {
    pendingMessage.remove();
    addMessage("assistant", `Something went wrong: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

clearIndexButton.addEventListener("click", async () => {
  const confirmed = window.confirm("Clear the current vector index? Uploaded files will remain in the uploads folder.");
  if (!confirmed) return;

  clearIndexButton.disabled = true;
  try {
    await requestJson("/clear-index", { method: "POST" });
    addMessage("assistant", "The index is clear. Upload documents to start a new knowledge base.");
    await refreshDashboard();
  } catch (error) {
    addMessage("assistant", `Could not clear the index: ${error.message}`);
  } finally {
    clearIndexButton.disabled = false;
  }
});

refreshStatus.addEventListener("click", refreshDashboard);
refreshDashboard();
