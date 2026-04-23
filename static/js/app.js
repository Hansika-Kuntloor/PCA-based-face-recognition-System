const state = {
    samples: [],
    stream: null,
};

const elements = {
    video: document.getElementById("video"),
    overlayCanvas: document.getElementById("overlayCanvas"),
    captureCanvas: document.getElementById("captureCanvas"),
    nameInput: document.getElementById("nameInput"),
    captureBtn: document.getElementById("captureBtn"),
    registerBtn: document.getElementById("registerBtn"),
    trainBtn: document.getElementById("trainBtn"),
    recognizeBtn: document.getElementById("recognizeBtn"),
    sampleCounter: document.getElementById("sampleCounter"),
    previewStrip: document.getElementById("previewStrip"),
    clearSamplesBtn: document.getElementById("clearSamplesBtn"),
    statusMessage: document.getElementById("statusMessage"),
    recognitionName: document.getElementById("recognitionName"),
    recognitionDistance: document.getElementById("recognitionDistance"),
    recognitionCorrelation: document.getElementById("recognitionCorrelation"),
    recognitionEyeGap: document.getElementById("recognitionEyeGap"),
    metricModelAvailable: document.getElementById("metricModelAvailable"),
    metricSamples: document.getElementById("metricSamples"),
    metricAccuracy: document.getElementById("metricAccuracy"),
};

async function startCamera() {
    if (!elements.video) {
        return;
    }

    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        elements.video.srcObject = state.stream;
        setStatus("Camera ready. Capture several clear face samples before training.", false);
    } catch (error) {
        setStatus(`Unable to access webcam: ${error.message}`, true);
    }
}

function setStatus(message, isError = false) {
    if (!elements.statusMessage) {
        return;
    }
    elements.statusMessage.textContent = message;
    elements.statusMessage.className = `alert ${isError ? "error" : "info"}`;
    elements.statusMessage.scrollIntoView({ behavior: "smooth", block: "center" });
}

function syncCanvasSize() {
    if (!elements.video || !elements.overlayCanvas || elements.video.videoWidth === 0) {
        return;
    }

    elements.overlayCanvas.width = elements.video.videoWidth;
    elements.overlayCanvas.height = elements.video.videoHeight;
    elements.captureCanvas.width = elements.video.videoWidth;
    elements.captureCanvas.height = elements.video.videoHeight;
}

function captureFrame() {
    syncCanvasSize();
    const context = elements.captureCanvas.getContext("2d");
    context.drawImage(elements.video, 0, 0, elements.captureCanvas.width, elements.captureCanvas.height);
    return elements.captureCanvas.toDataURL("image/png");
}

function renderSamples() {
    elements.previewStrip.innerHTML = "";
    state.samples.forEach((sample) => {
        const img = document.createElement("img");
        img.src = sample;
        img.alt = "Captured face sample";
        elements.previewStrip.appendChild(img);
    });
    elements.sampleCounter.textContent = String(state.samples.length);
}

function clearOverlay() {
    const context = elements.overlayCanvas.getContext("2d");
    context.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
}

function drawBoundingBox(box, label) {
    syncCanvasSize();
    clearOverlay();

    if (!box) {
        return;
    }

    const context = elements.overlayCanvas.getContext("2d");
    context.strokeStyle = "#22c55e";
    context.lineWidth = 4;
    context.strokeRect(box.x, box.y, box.w, box.h);

    context.fillStyle = "rgba(12, 20, 21, 0.72)";
    context.fillRect(box.x, Math.max(0, box.y - 34), 180, 28);
    context.fillStyle = "#ffffff";
    context.font = "18px Segoe UI";
    context.fillText(label, box.x + 8, Math.max(20, box.y - 14));
}

function setButtonBusy(button, busy, busyLabel) {
    if (!button) {
        return;
    }

    const originalLabel = button.dataset.originalLabel || button.textContent;
    button.dataset.originalLabel = originalLabel;
    button.disabled = busy;
    button.textContent = busy ? busyLabel : originalLabel;
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") || "";

    if (response.redirected || response.url.includes("/login")) {
        throw new Error("Your session expired. Please log in again.");
    }

    if (!contentType.includes("application/json")) {
        const text = await response.text();
        throw new Error(text || "Unexpected server response.");
    }

    const data = await response.json();
    if (!response.ok || data.success === false) {
        throw new Error(data.message || `Request failed with status ${response.status}.`);
    }

    return data;
}

async function registerUser() {
    const name = elements.nameInput.value.trim();
    if (!name) {
        setStatus("Enter a user name before registration.", true);
        return;
    }
    if (state.samples.length < 3) {
        setStatus("Capture at least 3 samples for registration.", true);
        return;
    }

    setButtonBusy(elements.registerBtn, true, "Registering...");
    setStatus("Registering user and saving dataset samples...", false);

    try {
        const data = await requestJson(window.APP_STATE.registerUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, images: state.samples }),
        });

        setStatus(`${data.name} registered with ${data.sample_count} usable samples. Train the model next.`, false);
        state.samples = [];
        renderSamples();
        elements.nameInput.value = "";
    } catch (error) {
        console.error(error);
        setStatus(error.message || "Registration failed.", true);
    } finally {
        setButtonBusy(elements.registerBtn, false, "Registering...");
    }
}

async function trainModel() {
    setButtonBusy(elements.trainBtn, true, "Training...");
    setStatus("Training PCA model from stored samples...", false);

    try {
        const data = await requestJson(window.APP_STATE.trainUrl, { method: "POST" });
        if (elements.metricModelAvailable) {
            elements.metricModelAvailable.textContent = "Yes";
        }
        elements.metricSamples.textContent = String(Math.round(data.metrics.samples || 0));
        elements.metricAccuracy.textContent = `${data.metrics.accuracy || 0}%`;
        setStatus(`Training complete. Accuracy: ${data.metrics.accuracy || 0}% across ${data.registered_users} user profiles.`, false);
    } catch (error) {
        console.error(error);
        setStatus(error.message || "Training failed.", true);
    } finally {
        setButtonBusy(elements.trainBtn, false, "Training...");
    }
}

async function recognizeCurrentFace() {
    const image = captureFrame();
    setButtonBusy(elements.recognizeBtn, true, "Recognizing...");
    setStatus("Running recognition on the current frame...", false);

    try {
        const data = await requestJson(window.APP_STATE.recognizeUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image }),
        });

        elements.recognitionName.textContent = data.name || "Unknown";
        elements.recognitionDistance.textContent = data.distance ?? "-";
        elements.recognitionCorrelation.textContent = data.correlation ?? "-";
        elements.recognitionEyeGap.textContent = data.eye_gap ?? "-";
        drawBoundingBox(data.bounding_box, data.name || "Unknown");
        setStatus(data.message || "Recognition processed.", !data.matched);
    } catch (error) {
        console.error(error);
        setStatus(error.message || "Recognition failed.", true);
        clearOverlay();
    } finally {
        setButtonBusy(elements.recognizeBtn, false, "Recognizing...");
    }
}

function bindEvents() {
    if (!elements.captureBtn) {
        return;
    }

    elements.captureBtn.addEventListener("click", () => {
        const frame = captureFrame();
        state.samples.push(frame);
        renderSamples();
        setStatus(`Sample ${state.samples.length} captured.`, false);
    });

    elements.clearSamplesBtn.addEventListener("click", () => {
        state.samples = [];
        renderSamples();
        setStatus("Captured samples cleared.", false);
    });

    elements.registerBtn.addEventListener("click", registerUser);
    elements.trainBtn.addEventListener("click", trainModel);
    elements.recognizeBtn.addEventListener("click", recognizeCurrentFace);
    elements.video.addEventListener("loadedmetadata", syncCanvasSize);
    window.addEventListener("resize", syncCanvasSize);
}

startCamera();
bindEvents();
