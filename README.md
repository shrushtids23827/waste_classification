# Waste Classification Using AI

A Flask application that classifies an uploaded image as cardboard, glass,
metal, paper, plastic, or trash. It runs the trained model with ONNX Runtime;
TensorFlow is not needed to run the web application.

The application expects these files in the project root:

- `waste_classifier.onnx` — generated from the supplied Keras model using the
  Colab notebook below.
- `class_names.json` — the output labels in model order.

The original `waste_classifier.keras` is kept as the conversion source. The
training dataset in `dataset/` is not used by the app and is not included in
Git.

## Convert the model in Google Colab

The current Windows environment blocks TensorFlow's model-loading DLL, so the
Keras model could not be converted locally. Open
[`convert_model_to_onnx_colab.ipynb`](./convert_model_to_onnx_colab.ipynb) in
Google Colab and run its cells:

1. Install the conversion tools.
2. Upload `waste_classifier.keras` and `class_names.json`.
3. Convert the saved model to `waste_classifier.onnx`.
4. Upload a real test image. The notebook compares Keras and ONNX Runtime
   outputs on that image before downloading the ONNX file.
5. Download `waste_classifier.onnx` and copy it into this project folder beside
   `app.py`.

The converter preserves the Keras model graph, including its MobileNetV2
`Rescaling` layer. Do not apply a second `/255` normalization in the Flask app.
The notebook checks the conversion against Keras before it downloads the
artifact. The ONNX model is not generated or included until that conversion
and parity check succeed.

## Install

Use Python 3.10 or 3.11 in PowerShell:

```powershell
cd D:\Waste_Classification_AI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activating the virtual environment is blocked by your PowerShell settings,
run its Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Make sure `waste_classifier.onnx` is beside `app.py`, then run:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>, select an image, and click **Classify Waste**.
The `/predict` API accepts the uploaded image in a multipart `image` field.
Images are EXIF-corrected, converted to RGB, resized to 224 × 224, and sent to
ONNX Runtime as float32 values in the 0–255 range. The model performs its own
rescaling.

## Limitations and troubleshooting

- The app requires a successfully converted `waste_classifier.onnx`; it will
  return a clear service error if that file is missing or invalid.
- The model predicts one category for the whole image. It does not locate
  multiple objects or guarantee correct recycling decisions.
- Confidence is the highest model output expressed as a percentage; it is not
  a calibrated measure of real-world certainty.
- Uploads are limited to 10 MB.
- The ONNX Runtime package imports successfully in the inspected environment,
  but actual inference still needs to be tested after the ONNX artifact is
  generated. If local Windows policy also blocks ONNX Runtime native files,
  do not weaken Windows security settings; use an administrator-approved
  WSL2/Linux environment or test inference in Colab.

## GitHub

`.gitignore` excludes `dataset/`, Python environments, caches, and common IDE
or operating-system temporary files. It does not exclude
`waste_classifier.keras`, `waste_classifier.onnx`, or `class_names.json`.
Check applicable licensing before publishing the model or data.
