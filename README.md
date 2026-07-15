# Digit Recognition Canvas

This is my first end-to-end machine learning project — applying **multiclass classification** (digits 0–9) on the MNIST dataset, then serving that model behind a FastAPI backend with a hand-drawn HTML canvas frontend.

The goal wasn't just "train a model that gets good accuracy" — it was to actually understand the full workflow of taking a model from a training script to a real, interactive product: data flow, pipeline structure, train/serve separation, and the messy reality of connecting a live user input (mouse drawing) to a model trained on a very different, cleaner dataset.

## The Data

The dataset is **MNIST** (`mnist_784` from OpenML) — 70,000 images of handwritten digits, each a 28×28 grayscale image flattened into a 784-length vector of pixel intensities (0–255, where higher values represent darker ink). Labels are the digit 0–9 each image represents.

Critically, MNIST's images originated from **actual handwritten digits on paper, scanned/photographed, and then downsampled to 28×28**. That origin story matters a lot later — the pixel-level *texture* of these images (soft, anti-aliased stroke edges) is a direct consequence of that scanning-and-downsampling process, not something inherent to "handwriting" in general.

## Architecture

```
digit-recognition-canvas/
├── models/                  # trained model artifact (not code)
│   └── digit_pipeline.joblib
├── src/
│   ├── training/
│   │   └── train.py         # offline: load MNIST, train, persist
│   ├── api/
│   │   └── main.py          # online: load model, preprocess, serve predictions
│   └── frontend/
│       └── index.html       # canvas UI: draw, downsample, send to API
├── requirements.txt
└── README.md
```

- **`src/training/train.py`** — loads MNIST, builds an sklearn `Pipeline` (`StandardScaler` + `RandomForestClassifier`), trains it, evaluates it, and persists the *entire fitted pipeline* to `models/digit_pipeline.joblib` via `joblib`.
- **`src/api/main.py`** — a FastAPI app that loads the trained pipeline and exposes a `POST /predict` endpoint.
- **`src/frontend/index.html`** — a canvas where a user draws a digit; JavaScript downsamples the drawing to a 28×28 pixel array and sends it to `/predict`.

## Setup

1. `pip install -r requirements.txt`
2. `python -m src.training.train` — trains the model and saves it to `models/`
3. `uvicorn src.api.main:app --reload`
4. Visit `http://127.0.0.1:8000/`

## Key Design Decisions

**Bundling preprocessing and model into one Pipeline object.**
`StandardScaler` and `RandomForestClassifier` are trained and serialized together as a single `Pipeline`. This avoids a whole class of bugs where preprocessing and model state can drift apart (e.g. accidentally re-fitting a scaler on test data, or forgetting to apply it at inference time). Worth noting: since `RandomForestClassifier` is scale-invariant, the scaler isn't doing meaningful work for this particular classifier — it's kept for structural consistency and in case the classifier changes again later, not because it's strictly necessary here.

**Loading the model once, at startup — not per request.**
The pipeline is deserialized from `models/digit_pipeline.joblib` a single time when the FastAPI app starts, and kept in memory as a module-level object. Every `/predict` request reuses that same in-memory pipeline. Loading it inside the route function instead would mean re-reading and re-deserializing the file from disk on every single request — unnecessary latency for something that never changes between requests.

**Explicit error handling instead of silent failure.**
Three specific failure modes are handled deliberately rather than left to crash or silently misbehave:
- If the trained model file doesn't exist when the API starts, it fails immediately with a clear message telling you to run training first — instead of a buried `FileNotFoundError` traceback.
- If a client submits a blank canvas, the API explicitly detects this (no drawn pixels found during preprocessing) and returns a clean `400` error rather than confidently predicting a digit for an empty image.
- Incoming pixel arrays are validated for both correct length (784) and valid numeric range (no NaN, Infinity, or out-of-bounds values) before ever reaching the model.

**Gaussian blur — the fix that actually made this app work.**
This was the most important thing I solved, and it wasn't obvious at first. Early on, predictions on canvas-drawn digits were poor even after fixing color inversion and centering. The reason: **MNIST's training images come from handwritten notes that were scanned/photographed and then downsampled** — that process naturally introduces soft, anti-aliased gradients along every stroke edge (pixel values gradually fading from 0 to 255 rather than jumping sharply). A canvas drawing, by contrast, produces hard vector-edged strokes with no such gradient, and thresholding the canvas image to remove noise makes this worse by flattening any softness that did exist.

In other words, the model wasn't just learning "what a 3 looks like" — it was implicitly learning "what a scanned photograph of a 3 looks like," texture included. Feeding it a crisp, hard-edged canvas drawing was asking it a subtly different question than it was trained to answer.

The fix: apply `scipy.ndimage.gaussian_filter` to the canvas image *after* inversion and centering, artificially reintroducing soft, blurred edges similar to what MNIST's scanning process produced naturally. This single step closed most of the remaining accuracy gap between "prediction on a real MNIST test sample" and "prediction on a live canvas drawing" — more than color inversion or centering did individually.

## Limitations

- **RandomForestClassifier treats the image as an unordered list of 784 independent numbers, not as a 2D image.** Flattening a 28×28 image into a 784-length vector throws away all spatial structure — the model has no built-in notion that pixel #100 is next to pixel #101, or that a row of pixels forms a line stroke.
- **It cannot learn spatial patterns like edges, curves, or strokes directly.** Each decision tree just picks thresholds on individual pixel positions (e.g. "is pixel #350 > 128?") — it never learns a *reusable* concept like "a curve" or "a vertical stroke" that generalizes across different digits or positions.
- **A CNN (Convolutional Neural Network) learns fundamentally differently.** It uses convolutional filters that slide across the image and explicitly learn spatial features — edges, corners, curves — and stack these into increasingly complex shapes across layers. This is much closer to how the digit actually "looks" as a 2D image.
- **This is why RandomForest is more sensitive to shifts, rotations, and stroke-style variation.** A digit drawn slightly off-center, at a different angle, or with different stroke thickness than training data can confuse a RandomForest much more than a CNN, because the RandomForest is comparing raw pixel positions rather than translation-tolerant learned features.
- **This is also why our centering and blur preprocessing steps matter more here than they would for a CNN.** A CNN's convolutional filters are naturally more tolerant of small position/texture shifts; a pixel-position-based model like RandomForest has no such tolerance built in, so preprocessing has to do more of that work manually.
- **RandomForest also can't share what it learns from one part of the image with another.** A CNN's filters are reused across the entire image (the same edge-detector filter scans everywhere), so it needs less data to generalize. RandomForest treats pixel #5 and pixel #500 as completely unrelated features, learned independently.
- **In short:** RandomForest's ~96% accuracy is genuinely good for a non-neural, classic ML approach on flattened pixel data, but a CNN would typically outperform it, and would be considerably more robust to real-world canvas input (different stroke styles, thicknesses, and positions) without needing as much manual preprocessing.

## What I Learned

- A model's accuracy on paper (test-set score) doesn't guarantee good real-world performance if the *serving-time* input distribution differs from the *training-time* one — even in ways as subtle as edge texture.
- Preprocessing at serving time isn't just about matching *shape* (784 numbers) or *convention* (black background vs white) — it has to match the underlying *data-generating process* as closely as possible.
- Keeping the frontend "dumb" (capture + send raw pixels only) and centralizing all MNIST-specific assumptions in the backend's `preprocess()` function made this mismatch easy to locate, diagnose, and fix in one place.
