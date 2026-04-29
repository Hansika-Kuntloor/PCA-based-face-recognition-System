const authElements = {
    video: document.getElementById("video"),
    overlayCanvas: document.getElementById("overlayCanvas"),
    captureCanvas: document.getElementById("captureCanvas"),
    recognizeBtn: document.getElementById("recognizeBtn"),
    authMessage: document.getElementById("authMessage"),
    authStatus: document.getElementById("authStatus"),
    authDistance: document.getElementById("authDistance"),
    authEyeDifference: document.getElementById("authEyeDifference"),
    authCorrelation: document.getElementById("authCorrelation"),
    userDetails: document.getElementById("userDetails"),
    detailName: document.getElementById("detailName"),
    detailIdentifier: document.getElementById("detailIdentifier"),
    detailEmail: document.getElementById("detailEmail"),
};

let authStream = null;

async function startRecognitionCamera() {
    if (!authElements.video) {
        return;
    }

    try {
        authStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        authElements.video.srcObject = authStream;
    } catch (error) {
        console.error(error);
        authElements.authMessage.textContent = `Unable to access webcam: ${error.message}`;
        authElements.authMessage.className = "alert error";
    }
}

function syncAuthCanvasSize() {
    if (!authElements.video || authElements.video.videoWidth === 0) {
        return;
    }
    authElements.overlayCanvas.width = authElements.video.videoWidth;
    authElements.overlayCanvas.height = authElements.video.videoHeight;
    authElements.captureCanvas.width = authElements.video.videoWidth;
    authElements.captureCanvas.height = authElements.video.videoHeight;
}

function captureAuthFrame() {
    syncAuthCanvasSize();
    const context = authElements.captureCanvas.getContext("2d");
    context.drawImage(
        authElements.video,
        0,
        0,
        authElements.captureCanvas.width,
        authElements.captureCanvas.height,
    );
    return authElements.captureCanvas.toDataURL("image/png");
}

function clearOverlay() {
    const context = authElements.overlayCanvas.getContext("2d");
    context.clearRect(0, 0, authElements.overlayCanvas.width, authElements.overlayCanvas.height);
}

function drawBox(box, label, granted) {
    clearOverlay();
    if (!box) {
        return;
    }

    const context = authElements.overlayCanvas.getContext("2d");
    context.strokeStyle = granted ? "#34d399" : "#f87171";
    context.lineWidth = 4;
    context.strokeRect(box.x, box.y, box.w, box.h);
    context.fillStyle = "rgba(12, 20, 21, 0.72)";
    context.fillRect(box.x, Math.max(0, box.y - 34), 220, 28);
    context.fillStyle = "#ffffff";
    context.font = "18px Segoe UI";
    context.fillText(label, box.x + 8, Math.max(20, box.y - 14));
}

function showUserDetails(user) {
    if (!user) {
        authElements.userDetails.classList.add("hidden");
        authElements.detailName.textContent = "";
        authElements.detailIdentifier.textContent = "";
        authElements.detailEmail.textContent = "";
        return;
    }

    authElements.userDetails.classList.remove("hidden");
    authElements.detailName.textContent = user.name || "-";
    authElements.detailIdentifier.textContent = user.person_identifier || "-";
    authElements.detailEmail.textContent = user.email || "-";
}

function formatMetric(value, fallback = "Unavailable") {
    return value === null || value === undefined ? fallback : value;
}

async function authenticateFace() {
    authElements.recognizeBtn.disabled = true;
    authElements.recognizeBtn.textContent = "Authenticating...";

    try {
        const response = await fetch("/recognize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: captureAuthFrame() }),
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Authentication failed.");
        }

        authElements.authMessage.textContent = data.message;
        authElements.authMessage.className = `alert ${data.matched ? "info" : "error"}`;
        authElements.authStatus.textContent = data.status;
        authElements.authDistance.textContent = data.pca_distance ?? "-";
        authElements.authEyeDifference.textContent = formatMetric(data.eye_difference);
        authElements.authCorrelation.textContent = data.correlation ?? "-";
        showUserDetails(data.matched ? data.user : null);
        drawBox(data.bounding_box, data.matched ? data.user.name : "Unknown", data.matched);
    } catch (error) {
        console.error(error);
        authElements.authMessage.textContent = error.message;
        authElements.authMessage.className = "alert error";
        authElements.authStatus.textContent = "error";
        showUserDetails(null);
        clearOverlay();
    } finally {
        authElements.recognizeBtn.disabled = false;
        authElements.recognizeBtn.textContent = "Authenticate Face";
    }
}

authElements.recognizeBtn?.addEventListener("click", authenticateFace);
authElements.video?.addEventListener("loadedmetadata", syncAuthCanvasSize);
window.addEventListener("resize", syncAuthCanvasSize);

startRecognitionCamera();
