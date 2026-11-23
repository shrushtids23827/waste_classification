import json
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "waste_classifier.onnx"
CLASS_NAMES_PATH = PROJECT_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE


def load_class_names():
    try:
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as class_file:
            class_names = json.load(class_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not load class names from {CLASS_NAMES_PATH}."
        ) from exc

    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
    ):
        raise RuntimeError(
            f"{CLASS_NAMES_PATH} must contain a non-empty JSON list of class names."
        )

    return class_names


CLASS_NAMES = load_class_names()
_session = None
_input_name = None
_output_name = None
_session_lock = threading.Lock()


def get_session():
    global _session, _input_name, _output_name

    if _session is None:
        with _session_lock:
            if _session is None:
                if not MODEL_PATH.is_file():
                    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

                import onnxruntime as ort

                session = ort.InferenceSession(
                    str(MODEL_PATH),
                    providers=["CPUExecutionProvider"],
                )
                inputs = session.get_inputs()
                outputs = session.get_outputs()

                if len(inputs) != 1 or len(outputs) != 1:
                    raise ValueError(
                        "The ONNX model must have exactly one input and one output."
                    )

                model_input = inputs[0]
                if model_input.type != "tensor(float)":
                    raise ValueError("The ONNX model input must be float32.")
                if len(model_input.shape) != 4:
                    raise ValueError(
                        "The ONNX model input must have NHWC image dimensions."
                    )
                for actual_size, expected_size in zip(
                    model_input.shape[1:], (*IMAGE_SIZE, 3)
                ):
                    if (
                        isinstance(actual_size, int)
                        and actual_size != expected_size
                    ):
                        raise ValueError(
                            "The ONNX model input must accept 224x224 RGB images."
                        )

                _session = session
                _input_name = model_input.name
                _output_name = outputs[0].name

    return _session, _input_name, _output_name


def preprocess_image(image_file):
    try:
        with Image.open(image_file.stream) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
            image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32)
    except (
        Image.DecompressionBombError,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        raise ValueError(
            "The uploaded file is not a valid or supported image."
        ) from exc

    # The model includes MobileNetV2 rescaling, so keep pixel values in [0, 255].
    return np.expand_dims(image_array, axis=0)


def prediction_probabilities(raw_prediction):
    scores = np.asarray(raw_prediction, dtype=np.float64)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]

    if scores.ndim != 1 or scores.size != len(CLASS_NAMES):
        raise ValueError("The model output does not match the configured classes.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction values.")

    total = float(np.sum(scores))
    if (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(total, 1.0, rtol=1e-3, atol=1e-3)
    ):
        return scores

    shifted_scores = scores - np.max(scores)
    exponentials = np.exp(shifted_scores)
    return exponentials / np.sum(exponentials)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify(error="The image is too large. The maximum upload size is 10 MB."), 413


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify(error="No image was uploaded. Choose an image and try again."), 400
    if not image_file.filename:
        return jsonify(error="The selected image has no filename."), 400

    try:
        image_batch = preprocess_image(image_file)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    try:
        session, input_name, output_name = get_session()
    except FileNotFoundError:
        app.logger.exception("The ONNX model file is missing.")
        return jsonify(
            error=(
                "The ONNX model file is missing. Generate waste_classifier.onnx "
                "using the provided Colab notebook and place it beside app.py."
            )
        ), 503
    except Exception:
        app.logger.exception("Could not load the ONNX waste classification model.")
        return jsonify(
            error=(
                "The ONNX model could not be loaded. Check that ONNX Runtime is "
                "installed and waste_classifier.onnx is a valid model."
            )
        ), 503

    try:
        raw_prediction = session.run(
            [output_name],
            {input_name: image_batch},
        )[0]
        probabilities = prediction_probabilities(raw_prediction)
    except Exception:
        app.logger.exception("Waste image prediction failed.")
        return jsonify(
            error="The image could not be classified. Please try another image."
        ), 500

    predicted_index = int(np.argmax(probabilities))
    return jsonify(
        prediction=CLASS_NAMES[predicted_index],
        confidence=round(float(probabilities[predicted_index]) * 100, 2),
        message="Waste classified successfully.",
    )


if __name__ == "__main__":
    app.run(debug=False)
