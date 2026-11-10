
<div align="center">
# 🔮 EmotionLens

**Real-time text emotion detection powered by a Bidirectional GRU neural network**

`sadness 😢` · `joy 😄` · `love ❤️` · `anger 😠` · `fear 😨` · `surprise 😲`

---

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

## 📖 Overview

EmotionLens is a full-stack NLP application that classifies human emotions from raw text using deep learning. A user types (or pastes) any English sentence and gets an instant emotion prediction with a full probability breakdown across six emotion classes.

The backend runs a **Bidirectional GRU (BiGRU)** model served through a **FastAPI** REST API, while the frontend is a hand-crafted glassmorphism UI with animated visualizations — all without any JavaScript framework.

---

## 🧠 Model Architecture & Training

### Dataset

The model was trained on the [**dair-ai/emotion**](https://huggingface.co/datasets/dair-ai/emotion) dataset from Hugging Face:

| Split | Samples |
|-------|---------|
| Train | 16,000 |
| Test  | 2,000 |

Each sample is a short English sentence labeled with one of six emotions: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.

### Preprocessing Pipeline

```
Raw Text → Lowercase → Remove Apostrophes → Strip Special Characters → Collapse Whitespace
                                         ↓
                               Tokenize (top 10,000 words, <unk> OOV)
                                         ↓
                               Pad / Truncate to length 50
```

### Models Compared

Four RNN architectures were benchmarked during experimentation:

| Model | Architecture | Embedding Dim | Test Accuracy | Test Loss |
|-------|-------------|:---:|:---:|:---:|
| **BiGRU** ✅ | Bidirectional GRU × 2 | 300 | **92.15%** | **0.2177** |
| LSTM | LSTM × 2 | 128 | 87.30% | 0.4801 |
| RNN | SimpleRNN × 2 | 128 | 24.95% | 1.7689 |
| GRU | GRU × 2 | 128 | 11.10% | 1.7969 |

> The BiGRU model dramatically outperforms the others — nearly **5×** the accuracy of a vanilla RNN.

### Final Model — BiGRU

```
Embedding(10000, 300)
    ↓
Bidirectional(GRU(128, return_sequences=True))
    ↓
Dropout(0.5)
    ↓
Bidirectional(GRU(64))
    ↓
Dropout(0.5)
    ↓
Dense(6, softmax)
```

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Cross-Entropy
- **Class Balancing:** Sklearn `compute_class_weight("balanced")`
- **Early Stopping:** patience = 3, monitoring `val_loss`, restoring best weights
- **Trained for:** 4 epochs (early stopped at epoch 7, best at epoch 4)

---

## 🏗️ Project Structure

```
rnn_nlp/
├── main.py                            # FastAPI server — API endpoints & model inference
├── requirements.txt                   # Python dependencies
├── emotion_classification.ipynb       # Jupyter notebook — EDA, training, evaluation
├── Artifacts/
│   ├── Bigru_model.keras              # Trained BiGRU model (~40 MB)
│   └── tokenizer.pkl                  # Fitted Keras Tokenizer
└── static/
    ├── index.html                     # Frontend — page structure
    ├── style.css                      # Styles — glassmorphism, animations
    └── app.js                         # Client logic — API calls, bar rendering
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repo
git clone https://github.com/slokrami07/rnn_nlp.git
cd rnn_nlp

# Create virtual environment
python -m venv .venv

# Activate it
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **NumPy version matters.** TensorFlow 2.17 requires `numpy<2`. The pinned version in `requirements.txt` (`numpy==1.26.4`) handles this.

### Run the Server

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## 📡 API Reference

### `GET /`

Serves the frontend UI.

---

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "server is running",
  "model_laoded": true
}
```

---

### `POST /predict`

Predict the emotion of a sentence.

**Request Body:**

```json
{
  "text": "I feel so happy today!"
}
```

| Field | Type | Constraints |
|-------|------|-------------|
| `text` | `string` | min: 1 char, max: 2000 chars |

**Response:**

```json
{
  "text": "I feel so happy today!",
  "predicted_emotion": "joy",
  "confidence": 0.9834,
  "all_probabilities": {
    "sadness": 0.0021,
    "joy": 0.9834,
    "love": 0.0058,
    "anger": 0.0012,
    "fear": 0.0034,
    "surprise": 0.0041
  }
}
```

**Error (503):**

```json
{
  "detail": "model is not loaded yet please try again later"
}
```

---

## 🎨 Frontend Features

The UI is a **zero-dependency**, framework-free frontend built with vanilla HTML, CSS, and JavaScript.

| Feature | Details |
|---------|---------|
| 🌌 Animated background blobs | Three floating gradient blobs with independent animations |
| 🪟 Glassmorphism cards | `backdrop-filter: blur(20px)` with subtle borders and hover lifts |
| 🔴 Live server status | Polls `/health` every 15s — green dot = ready, red = offline |
| 🎯 Confidence ring | Animated SVG circular progress with gradient stroke |
| 📊 Color-coded bars | Each emotion has a distinct color, sorted by probability |
| ✨ Shimmer effect | The top probability bar gets a looping light sweep |
| 🎭 Emoji pop | Result emoji enters with a spring-curve bounce animation |
| ⌨️ Keyboard shortcut | `Ctrl + Enter` triggers analysis |
| 📱 Responsive | Adapts to mobile screens |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | TensorFlow / Keras 2.17 |
| **Model** | Bidirectional GRU (BiGRU) |
| **Backend** | FastAPI 0.115 + Uvicorn 0.30 |
| **Validation** | Pydantic v2 |
| **Frontend** | Vanilla HTML / CSS / JS |
| **Data** | HuggingFace `dair-ai/emotion` |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ and a lot of epochs**

</div>

