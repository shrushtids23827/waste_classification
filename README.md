# Waste Classification AI

A Flask website that classifies an uploaded image as cardboard, glass, metal,
paper, plastic, or trash using the included ONNX model and ONNX Runtime.

## Run locally

Use Python 3.10 or 3.11:

```powershell
cd D:\Waste_Classification_AI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000/> and upload an image. The app also uses the `PORT`
environment variable when set. The `/predict` API accepts a multipart upload
with the field name `image`.

The app loads `waste_classifier.onnx` from the project root. Uploads are
EXIF-corrected, converted to RGB, resized to 224 × 224, and passed to the model
as float32 pixel values in the 0–255 range. The model includes its own
preprocessing; the app does not apply additional normalization. Images larger
than 10 MB are rejected.

## Deploy on Render

The included [`render.yaml`](./render.yaml) configures a Python web service,
installs `requirements.txt`, and starts Gunicorn bound to Render's assigned
`PORT`.

1. Push this repository, including `waste_classifier.onnx`, to GitHub.
2. Sign in to [Render](https://render.com/) and choose **New → Blueprint**.
3. Connect `shrushtids23827/waste_classification` and select the `main` branch.
4. Review the service generated from `render.yaml` and choose **Apply**.
5. Wait for the build and deploy to finish, then open the service's `onrender.com`
   URL and test an image upload.

No API keys or extra environment variables are required. Render supplies `PORT`
for the web service. The free instance may sleep when idle and take longer to
respond on its first request.

## Models and files

- `waste_classifier.onnx` is required at runtime and is included in this
  repository.
- `waste_classifier.keras` is retained as the original conversion source; the
  Flask app does not need TensorFlow.
- `class_names.json` defines the six output classes in model order.
- `convert_model_to_onnx_colab.ipynb` documents the conversion and real-image
  parity checks.
- `dataset/` is local training/test data and is excluded from Git.

The ONNX model predicts one category for the whole image. Confidence is the
model's highest output expressed as a percentage; it is not a calibrated
measure of real-world certainty.
