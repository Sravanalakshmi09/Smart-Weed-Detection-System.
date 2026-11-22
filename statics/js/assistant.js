
document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("assistantForm");
    const chatInput = document.getElementById("assistantInput");
    const chatMessages = document.getElementById("assistantMessages");
    const sendButton = document.getElementById("assistantSend");
    const fieldContainer = document.getElementById("fieldAnalysisContainer");
    const refreshButton = document.getElementById("refreshFieldAnalysis");

    const CHAT_URL = "/api/assistant/chat";
    const FIELD_STATUS_URL = "/api/assistant/field-status";

    function escapeHTML(value) {
        return String(value ?? "").replace(/[&<>"']/g, char => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        })[char]);
    }

    function addMessage(message, sender = "assistant") {
        if (!chatMessages) return;

        const messageElement = document.createElement("div");
        messageElement.className = `assistant-message ${sender}-message`;

        const content = document.createElement("div");
        content.className = "assistant-message-content";
        content.textContent = message;

        messageElement.appendChild(content);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function setSending(isSending) {
        if (!sendButton) return;

        sendButton.disabled = isSending;
        sendButton.textContent = isSending ? "Sending..." : "Send";
    }

    async function sendMessage(message) {
        try {
            const response = await fetch(CHAT_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Unable to get a response.");
            }

            addMessage(
                data.response || data.reply || "I could not generate a response.",
                "assistant"
            );
        } catch (error) {
            addMessage(error.message || "Assistant connection failed.", "assistant");
        } finally {
            setSending(false);
        }
    }

    if (chatForm && chatInput) {
        chatForm.addEventListener("submit", async event => {
            event.preventDefault();

            const message = chatInput.value.trim();
            if (!message) return;

            addMessage(message, "user");
            chatInput.value = "";
            setSending(true);

            await sendMessage(message);
        });
    }

    function createPercentageBar(label, percentage, color) {
        const safeValue = Math.max(0, Math.min(100, Number(percentage) || 0));

        return `
            <div class="field-class-row">
                <div class="field-class-heading">
                    <span>${escapeHTML(label)}</span>
                    <strong>${safeValue.toFixed(2)}%</strong>
                </div>
                <div class="field-class-track">
                    <div class="field-class-fill"
                         style="width:${safeValue}%;background:${color};">
                    </div>
                </div>
            </div>
        `;
    }

    function createFieldCard(field) {
        const weed = Number(field.weed_percentage) || 0;
        const crop = Number(field.crop_percentage) || 0;
        const soil = Number(field.soil_percentage) || 0;

        const card = document.createElement("div");
        card.className = "field-analysis-card";

        card.innerHTML = `
            <div class="field-analysis-header">
                <div>
                    <h3>${escapeHTML(field.field_name || "Unnamed Field")}</h3>
                    <p>${escapeHTML(field.location || "Location not specified")}</p>
                </div>
                <span class="field-analysis-date">
                    ${escapeHTML(field.last_analysis || field.created_at || "")}
                </span>
            </div>

            <div class="field-analysis-total">
                <span>Weed coverage</span>
                <strong>${weed.toFixed(2)}%</strong>
            </div>

            <div class="field-class-bars">
                ${createPercentageBar("Weed", weed, "#e74c3c")}
                ${createPercentageBar("Crop", crop, "#27ae60")}
                ${createPercentageBar("Background / Soil", soil, "#9e9e9e")}
            </div>

            <div class="field-analysis-footer">
                <span>Severity: ${escapeHTML(field.severity || "Not available")}</span>
                <span>Analyses: ${Number(field.analysis_count) || 1}</span>
            </div>
        `;

        return card;
    }

    async function loadFieldAnalyses() {
        if (!fieldContainer) return;

        fieldContainer.innerHTML = "<p>Loading field analysis...</p>";

        try {
            const response = await fetch(FIELD_STATUS_URL, {
                method: "GET",
                headers: {
                    "Accept": "application/json"
                }
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Could not load field analysis.");
            }

            const fields = Array.isArray(data)
                ? data
                : (data.fields || data.analyses || []);

            fieldContainer.innerHTML = "";

            if (fields.length === 0) {
                fieldContainer.innerHTML =
                    "<p>No saved field analyses are available yet.</p>";
                return;
            }

            fields.forEach(field => {
                fieldContainer.appendChild(createFieldCard(field));
            });

        } catch (error) {
            fieldContainer.innerHTML =
                `<p class="assistant-error">${escapeHTML(error.message)}</p>`;
        }
    }

    if (refreshButton) {
        refreshButton.addEventListener("click", loadFieldAnalyses);
    }

    loadFieldAnalyses();
});
