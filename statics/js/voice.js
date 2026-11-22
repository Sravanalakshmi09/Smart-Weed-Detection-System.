
/* =========================================================
   SMART WEED DETECTION
   AI CHAT AND VOICE ASSISTANT
========================================================= */

(() => {
    "use strict";

    const CHAT_URL = "/api/assistant/chat";
    const TRANSCRIBE_URL = "/api/local-transcribe";

    const LANGUAGES = {
        en: "en-IN",
        te: "te-IN",
        ta: "ta-IN",
        kn: "kn-IN",
        ml: "ml-IN",
        hi: "hi-IN"
    };

    const get = (...ids) => {
        for (const id of ids) {
            const element = document.getElementById(id);
            if (element) return element;
        }
        return null;
    };

    // Support both the original IDs and updated base.html IDs.
    const toggle = get("assistantToggle", "aiToggle");
    const panel = get("assistantPanel", "aiPanel");
    const closeButton = get("assistantClose", "aiClose");

    const form = get("aiChatForm");
    const input = get("assistantInput", "aiChatInput");
    const messages = get("assistantMessages", "aiChatMessages");

    const status = get("assistantStatus", "aiStatus");
    const statusText = get("assistantStatusText");

    const micButton = get("assistantMic", "aiMicButton");
    const sendButton = get("assistantSend", "aiSendButton");

    const languageSelect = get("aiLanguageSelect");

    if (!toggle || !panel || !form || !input || !messages) {
        console.warn(
            "AI assistant elements were not found. Check base.html IDs."
        );
        return;
    }

    let recorder = null;
    let mediaStream = null;
    let audioChunks = [];

    let isRecording = false;
    let isSending = false;

    /* =====================================================
       STATUS
    ===================================================== */

    function setStatus(message) {
        if (status) {
            status.textContent = message;
        }

        if (statusText) {
            statusText.textContent = message;
        }
    }

    /* =====================================================
       OPEN AND CLOSE ASSISTANT
    ===================================================== */

    function openAssistant(open) {
        panel.hidden = !open;
        panel.classList.toggle("open", open);

        toggle.setAttribute(
            "aria-expanded",
            String(open)
        );

        if (open) {
            input.focus();
        } else {
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }

            if (
                isRecording &&
                recorder &&
                recorder.state !== "inactive"
            ) {
                recorder.stop();
            }
        }
    }

    toggle.addEventListener("click", () => {
        const isOpen =
            !panel.hidden &&
            panel.classList.contains("open");

        openAssistant(!isOpen);
    });

    if (closeButton) {
        closeButton.addEventListener("click", () => {
            openAssistant(false);
        });
    }

    /* =====================================================
       LANGUAGE DETECTION
    ===================================================== */

    function detectLanguage(text) {
        const value = String(text || "");

        if (/[\u0C00-\u0C7F]/.test(value)) return "te";
        if (/[\u0B80-\u0BFF]/.test(value)) return "ta";
        if (/[\u0C80-\u0CFF]/.test(value)) return "kn";
        if (/[\u0D00-\u0D7F]/.test(value)) return "ml";
        if (/[\u0900-\u097F]/.test(value)) return "hi";

        return "en";
    }

    function getSelectedLanguage(text) {
        const selected = languageSelect
            ? languageSelect.value
            : "auto";

        if (
            selected !== "auto" &&
            LANGUAGES[selected]
        ) {
            return selected;
        }

        return detectLanguage(text);
    }

    /* =====================================================
       CHAT MESSAGES
    ===================================================== */

    function addMessage(type, message) {
        const element = document.createElement("div");

        element.className =
            "ai-message " +
            (type === "user"
                ? "ai-user-message"
                : "ai-assistant-message");

        element.textContent = String(message);

        element.setAttribute("role", "listitem");

        messages.appendChild(element);

        messages.scrollTop = messages.scrollHeight;

        return element;
    }

    /* =====================================================
       SPEAK AI RESPONSE
    ===================================================== */

    function speakAnswer(text, language) {
        if (!window.speechSynthesis) {
            return;
        }

        const answer = String(text || "").trim();

        if (!answer) return;

        const languageCode =
            LANGUAGES[language]
                ? language
                : "en";

        const speech =
            new SpeechSynthesisUtterance(answer);

        speech.lang = LANGUAGES[languageCode];
        speech.rate = 0.9;
        speech.pitch = 1;
        speech.volume = 1;

        window.speechSynthesis.cancel();

        speech.onstart = () => {
            setStatus("Speaking...");
        };

        speech.onend = () => {
            if (!isSending) {
                setStatus("Ready");
            }
        };

        speech.onerror = () => {
            setStatus("Ready");
        };

        window.speechSynthesis.speak(speech);
    }

    /* =====================================================
       READ SERVER RESPONSE
    ===================================================== */

    async function readResponse(response) {
        const raw = await response.text();

        let data;

        try {
            data = JSON.parse(raw);
        } catch (error) {
            throw new Error(
                "Invalid server response. Please check Flask."
            );
        }

        if (!response.ok || data.success === false) {
            throw new Error(
                data.error ||
                data.message ||
                "Something went wrong."
            );
        }

        return data;
    }

    /* =====================================================
       SEND MESSAGE TO FLASK AI ROUTE
    ===================================================== */

    async function sendMessage(question, forcedLanguage) {
        question = String(question || "").trim();

        if (!question || isSending) {
            return;
        }

        const language =
            forcedLanguage ||
            getSelectedLanguage(question);

        isSending = true;

        addMessage("user", question);

        input.value = "";

        setStatus("Thinking...");

        if (sendButton) {
            sendButton.disabled = true;
        }

        if (micButton) {
            micButton.disabled = true;
        }

        try {
            const response = await fetch(CHAT_URL, {
                method: "POST",

                credentials: "same-origin",

                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },

                body: JSON.stringify({
                    message: question,
                    language: language
                })
            });

            const data = await readResponse(response);

            const answer =
                String(data.answer || "").trim();

            if (!answer) {
                throw new Error(
                    "The AI assistant returned an empty answer."
                );
            }

            addMessage("assistant", answer);

            setStatus("Ready");

            speakAnswer(
                answer,
                data.language || language
            );

        } catch (error) {
            console.error("AI Assistant Error:", error);

            addMessage(
                "assistant",
                "Sorry, I could not answer your question.\n\n" +
                error.message
            );

            setStatus("Assistant error");

        } finally {
            isSending = false;

            if (sendButton) {
                sendButton.disabled = false;
            }

            if (micButton) {
                micButton.disabled = false;
            }

            input.focus();
        }
    }

    /* =====================================================
       SEND TYPED MESSAGE
    ===================================================== */

    form.addEventListener("submit", event => {
        event.preventDefault();

        sendMessage(input.value);
    });

    /* =====================================================
       MICROPHONE BUTTON
    ===================================================== */

    function updateMicrophoneButton() {
        if (!micButton) return;

        micButton.textContent =
            isRecording ? "■" : "🎙️";

        micButton.classList.toggle(
            "recording",
            isRecording
        );

        micButton.setAttribute(
            "aria-label",
            isRecording
                ? "Stop recording"
                : "Start voice recording"
        );
    }

    function stopMicrophone() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => {
                track.stop();
            });

            mediaStream = null;
        }
    }

    /* =====================================================
       START RECORDING
    ===================================================== */

    async function startRecording() {
        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia ||
            !window.MediaRecorder
        ) {
            setStatus(
                "Microphone is not supported in this browser."
            );
            return;
        }

        try {
            mediaStream =
                await navigator.mediaDevices.getUserMedia({
                    audio: true
                });

            audioChunks = [];

            const options = {};

            const supportedTypes = [
                "audio/webm;codecs=opus",
                "audio/webm",
                "audio/mp4",
                "audio/ogg;codecs=opus"
            ];

            for (const mimeType of supportedTypes) {
                if (MediaRecorder.isTypeSupported(mimeType)) {
                    options.mimeType = mimeType;
                    break;
                }
            }

            recorder = new MediaRecorder(
                mediaStream,
                options
            );

            const activeRecorder = recorder;

            activeRecorder.addEventListener(
                "dataavailable",
                event => {
                    if (
                        event.data &&
                        event.data.size > 0
                    ) {
                        audioChunks.push(event.data);
                    }
                }
            );

            activeRecorder.addEventListener(
                "stop",
                async () => {
                    const audioBlob = new Blob(
                        audioChunks,
                        {
                            type:
                                activeRecorder.mimeType ||
                                "audio/webm"
                        }
                    );

                    stopMicrophone();

                    isRecording = false;

                    updateMicrophoneButton();

                    if (!audioBlob.size) {
                        setStatus("No audio recorded.");
                        return;
                    }

                    await transcribeAudio(audioBlob);
                }
            );

            activeRecorder.addEventListener(
                "error",
                error => {
                    console.error(
                        "Recording error:",
                        error
                    );

                    isRecording = false;

                    stopMicrophone();

                    updateMicrophoneButton();

                    setStatus("Recording failed.");
                }
            );

            activeRecorder.start();

            isRecording = true;

            updateMicrophoneButton();

            setStatus(
                "Listening... Speak now and press the button again to stop."
            );

        } catch (error) {
            console.error("Microphone error:", error);

            stopMicrophone();

            isRecording = false;

            updateMicrophoneButton();

            setStatus(
                "Please allow microphone permission."
            );
        }
    }

    /* =====================================================
       TRANSCRIBE RECORDED AUDIO
    ===================================================== */

    async function transcribeAudio(blob) {
        setStatus("Converting speech to text...");

        const formData = new FormData();

        let extension = "webm";

        if (blob.type.includes("mp4")) {
            extension = "m4a";
        } else if (blob.type.includes("ogg")) {
            extension = "ogg";
        } else if (blob.type.includes("wav")) {
            extension = "wav";
        }

        formData.append(
            "audio",
            blob,
            "recording." + extension
        );

        const selected =
            languageSelect
                ? languageSelect.value
                : "auto";

        formData.append("language", selected);

        try {
            const response = await fetch(
                TRANSCRIBE_URL,
                {
                    method: "POST",

                    credentials: "same-origin",

                    body: formData,

                    headers: {
                        "Accept": "application/json"
                    }
                }
            );

            const data = await readResponse(response);

            const transcript =
                String(data.text || "").trim();

            if (!transcript) {
                throw new Error(
                    "No speech was recognized."
                );
            }

            const language =
                LANGUAGES[data.language]
                    ? data.language
                    : detectLanguage(transcript);

            setStatus("Speech recognized. Sending...");

            await sendMessage(
                transcript,
                language
            );

        } catch (error) {
            console.error(
                "Transcription error:",
                error
            );

            addMessage(
                "assistant",
                "Speech recognition failed: " +
                error.message
            );

            setStatus("Transcription error");
        }
    }

    /* =====================================================
       MICROPHONE CLICK
    ===================================================== */

    if (micButton) {
        micButton.addEventListener("click", async () => {

            if (
                isRecording &&
                recorder &&
                recorder.state !== "inactive"
            ) {
                setStatus("Finishing recording...");

                recorder.stop();

            } else if (!isSending) {
                await startRecording();
            }

        });
    }

    /* =====================================================
       CLEANUP
    ===================================================== */

    window.addEventListener("beforeunload", () => {
        if (
            recorder &&
            recorder.state !== "inactive"
        ) {
            recorder.stop();
        }

        stopMicrophone();

        if (window.speechSynthesis) {
            window.speechSynthesis.cancel();
        }
    });

    console.log(
        "Smart Weed Detection AI Assistant loaded successfully."
    );

})();
