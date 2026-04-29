const captureState = {
    samples: [],
    stream: null,
};

const captureElements = {
    video: document.getElementById("video"),
    canvas: document.getElementById("captureCanvas"),
    captureBtn: document.getElementById("captureBtn"),
    clearBtn: document.getElementById("clearBtn"),
    counter: document.getElementById("sampleCounter"),
    previewStrip: document.getElementById("previewStrip"),
    samplesJson: document.getElementById("samplesJson"),
    form: document.getElementById("userForm"),
};

async function startCaptureCamera() {
    if (!captureElements.video) {
        return;
    }

    try {
        captureState.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        captureElements.video.srcObject = captureState.stream;
    } catch (error) {
        console.error(error);
        window.alert(`Unable to access webcam: ${error.message}`);
    }
}

function syncCaptureCanvasSize() {
    if (!captureElements.video || !captureElements.canvas || captureElements.video.videoWidth === 0) {
        return;
    }

    captureElements.canvas.width = captureElements.video.videoWidth;
    captureElements.canvas.height = captureElements.video.videoHeight;
}

function captureFrame() {
    syncCaptureCanvasSize();
    const context = captureElements.canvas.getContext("2d");
    const sourceWidth = captureElements.video.videoWidth;
    const sourceHeight = captureElements.video.videoHeight;
    const maxWidth = 640;
    const scale = sourceWidth > maxWidth ? maxWidth / sourceWidth : 1;
    const targetWidth = Math.max(1, Math.round(sourceWidth * scale));
    const targetHeight = Math.max(1, Math.round(sourceHeight * scale));

    captureElements.canvas.width = targetWidth;
    captureElements.canvas.height = targetHeight;
    context.drawImage(
        captureElements.video,
        0,
        0,
        targetWidth,
        targetHeight,
    );
    return captureElements.canvas.toDataURL("image/jpeg", 0.82);
}

function renderCapturedSamples() {
    captureElements.previewStrip.innerHTML = "";
    captureState.samples.forEach((sample) => {
        const image = document.createElement("img");
        image.src = sample;
        image.alt = "Captured face sample";
        captureElements.previewStrip.appendChild(image);
    });
    captureElements.counter.textContent = String(captureState.samples.length);
    captureElements.samplesJson.value = JSON.stringify(captureState.samples);
}

function bindCaptureEvents() {
    if (!captureElements.captureBtn || !captureElements.form) {
        return;
    }

    captureElements.captureBtn.addEventListener("click", () => {
        if (captureState.samples.length >= window.ADMIN_CAPTURE.maxSamples) {
            window.alert(`Only ${window.ADMIN_CAPTURE.maxSamples} samples are allowed for one user.`);
            return;
        }

        captureState.samples.push(captureFrame());
        renderCapturedSamples();
    });

    captureElements.clearBtn.addEventListener("click", () => {
        captureState.samples = [];
        renderCapturedSamples();
    });

    captureElements.form.addEventListener("submit", (event) => {
        captureElements.samplesJson.value = JSON.stringify(captureState.samples);
        if (
            window.ADMIN_CAPTURE.mode === "create" &&
            captureState.samples.length < window.ADMIN_CAPTURE.minSamples
        ) {
            event.preventDefault();
            window.alert(
                `Capture at least ${window.ADMIN_CAPTURE.minSamples} samples before creating a user.`,
            );
        }
        if (
            window.ADMIN_CAPTURE.mode === "edit" &&
            captureState.samples.length > 0 &&
            captureState.samples.length < window.ADMIN_CAPTURE.minSamples
        ) {
            event.preventDefault();
            window.alert(
                `If you replace samples during edit, capture at least ${window.ADMIN_CAPTURE.minSamples} new samples.`,
            );
        }
    });
}

startCaptureCamera();
bindCaptureEvents();
