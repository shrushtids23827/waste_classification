const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const classifyBtn = document.getElementById("classifyBtn");
const result = document.getElementById("result");

let previewUrl = null;

function showResult(title, detail, state) {
    result.replaceChildren();
    result.className = `result result-${state}`;

    const heading = document.createElement("h2");
    heading.textContent = title;
    result.appendChild(heading);

    if (detail) {
        const description = document.createElement("p");
        description.textContent = detail;
        result.appendChild(description);
    }
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    result.replaceChildren();
    result.className = "result";

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    if (!file) {
        preview.removeAttribute("src");
        preview.hidden = true;
        return;
    }

    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
});

classifyBtn.addEventListener("click", async () => {
    const file = imageInput.files[0];
    if (!file) {
        showResult("Choose an image first.", "", "error");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    classifyBtn.disabled = true;
    classifyBtn.textContent = "Classifying...";
    showResult("Analyzing image...", "This may take a few seconds.", "pending");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(data.error || "Classification failed. Please try again.");
        }

        const confidence = Number(data.confidence);
        if (typeof data.prediction !== "string" || !Number.isFinite(confidence)) {
            throw new Error("The server returned an invalid classification result.");
        }

        showResult(
            `Waste type: ${data.prediction}`,
            `Confidence: ${confidence.toFixed(2)}%`,
            "success",
        );
    } catch (error) {
        showResult(
            "Could not classify this image.",
            error instanceof Error ? error.message : "Please try again.",
            "error",
        );
    } finally {
        classifyBtn.disabled = false;
        classifyBtn.textContent = "Classify Waste";
    }
});
