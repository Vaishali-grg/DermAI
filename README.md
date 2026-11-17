# DermAI 🩺✨
### An AI-Powered, Evidence-Grounded Skin Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

**DermAI** bridges the gap between fast computer vision triage and trusted, evidence-grounded dermatological education. Current consumer skin-checking apps typically output shallow predictions (e.g., *"Acne 85%"*) without clinical context, while unguided web searches frequently lead to misleading self-diagnoses. 

DermAI provides a unified multimodal system that combines **EfficientNet-B2** visual feature extraction, a **Retrieval-Augmented Generation (RAG)** pipeline backed by peer-reviewed literature and American Academy of Dermatology (AAD) guidelines, and **Gemma 3 4B** to generate transparent, citation-backed educational summaries.

> **Authors:** Khushi (2410993046) & Vaishali (2410992989)

---

## 🚀 Key Features

* **Instant Photo Classification:** Accurately classifies 5 primary skin concerns (**Acne, Blackheads, Dark Spots, Enlarged Pores, Wrinkles**) with confidence metrics.
* **Textbook-Backed Answers:** Explains underlying causes, dietary factors, and home remedies using verified medical literature rather than unvetted web text.
* **Full Citation Transparency:** Displays exact document names and page numbers for retrieved clinical literature.
* **Safe & Simple Summaries:** Synthesizes complex clinical terminology into clear, accessible 3–5 sentence summaries.
* **Resource-Efficient Inference:** Engineered to execute locally on consumer GPUs within **6 GB VRAM**.

---

## 🏗️ System Architecture

DermAI is structured across three core modules:

```
[ User Input ] ---> [ Webcam / Photo Capture ] + [ User Query ]
                             │
                             ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. Vision Feature Extraction & Classification (EfficientNet-B2)       │
 │    • Resolution analysis, resize (256x256), center crop (224x224)      │
 │    • Generates predicted condition + classification confidence         │
 └───────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 2. Knowledge Retrieval Layer (ChromaDB + BGE Embeddings)               │
 │    • Search Query Assembly: Predicted Condition + User Query           │
 │    • Vector Embedding: BAAI/bge-small-en-v1.5 (384-dim, L2 normalized) │
 │    • Strict Similarity Distance Threshold (< 0.35)                     │
 │    • Top-K Evidence Chunks with Source & Page Metadata                 │
 └───────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 3. Multimodal Generation & Evidence Grounding (Gemma 3 4B)             │
 │    • Assembled Context: Visual Features + Clinical Chunks + Query      │
 │    • Structured Output: Condition, Causes, Home Care, Diet & Citations │
 └───────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
[ Final Response ] ---> [ Interactive Web UI (FastAPI + ngrok) ]
```

---

## 🔬 Core Pipeline Modules

### 1. Vision Feature Extraction (`EfficientNet-B2`)
* **Backbone:** EfficientNet-B2 fine-tuned on the **Kaggle Skin V2 Dataset** (9,770 images).
  * Train Split: 70% (6,839 images)
  * Validation Split: 15% (1,465 images)
  * Test Split: 15% (1,466 images)
* **Pre-processing:** Resize to `256×256`, Center Crop to `224×224`, ImageNet normalization.
* **Classes:** `0: Acne`, `1: Blackheads`, `2: Dark Spots`, `3: Pores`, `4: Wrinkles`.

#### Model Evaluation
| Skin Condition | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Acne** | 0.9591 | 0.9652 | 0.9621 |
| **Blackheads** | 0.9457 | 0.9667 | 0.9560 |
| **Dark Spots** | 0.9827 | 0.9714 | 0.9770 |
| **Pores** | 0.9867 | 0.9824 | 0.9845 |
| **Wrinkles** | 0.9933 | 0.9834 | 0.9884 |
| **Overall** | **97.36%** | **97.34%** | **97.34%** |
* **Overall Accuracy:** **97.34%**

---

### 2. Evidence-Grounded RAG (`src/`)
* **`pdf_loader.py`:** Cleans raw medical texts, rejoins hyphenated breaks, and filters noise (DOIs, page numbers, reference blocks).
* **`chunker.py`:** Context-aware recursive chunker (400 tokens / 50 overlap) with section hierarchy header prepending (`[Section > Subsection]`) and category tags (`dark_spots`, `acne`, `diet`, etc.).
* **`embeddings.py`:** BAAI 384-dimensional `bge-small-en-v1.5` SentenceTransformer with L2 normalization (`normalize_embeddings=True`).
* **`vector_store.py`:** Persistent storage in ChromaDB (`collection: dermatology_kb`) with inner product metric and `< 0.35` distance threshold filtering.

---

### 3. Generation & Serving
* **LLM Generator:** Gemma 3 4B executes conditioned response generation grounded exclusively on retrieved evidence chunks.
* **API & Deployment:** FastAPI exposes REST endpoints for webcam capture and chat queries, hosted via ngrok secure HTTPS tunnels for lightweight mobile/web access without cloud GPU costs.

---

## 🛠️ Engineering Challenges & Solutions

| Challenge | Solution |
| :--- | :--- |
| **6 GB GPU VRAM Limitation** | Lightweight model pairing: EfficientNet-B2 + Gemma 3 4B + compact 384-dim BGE embeddings. |
| **Noisy PDF Extraction** | Regex cleaning, hyphenation rejoining, header prepending, and reference filtering. |
| **Irrelevant RAG Retrieval** | Condition-driven query construction + category metadata filtering + `< 0.35` distance cutoff. |
| **LLM Hallucinations** | Strict grounding prompt forcing generation to cite verified literature or disclaim unknown facts. |

---

## 🛡️ Responsible AI & Safety

| DermAI IS ✅ | DermAI IS NOT ❌ |
| :--- | :--- |
| Educational and informational | A certified medical doctor or dermatologist |
| Evidence-grounded with citations | A definitive medical diagnosis |
| Clinical decision-support oriented | A replacement for in-person consultation |
| Transparent about source limitations | A medical prescription generator |

---

## 📂 Project Structure

```bash
medical_rag/
├── app.py                                # FastAPI backend service
├── main.py                               # Application entrypoint
├── final_multimodal_rag_gemma3.py        # Complete multimodal pipeline (EfficientNet + RAG + Gemma 3)
├── requirements.txt                      # Project dependencies
├── models/
│   └── efficientnet_b2_skin_issues.pth   # Fine-tuned classification weights
├── src/
│   ├── chunker.py                        # Hierarchical text chunking
│   ├── efficientnet.py                   # CNN feature extractor
│   ├── embeddings.py                     # BGE embedding generator
│   ├── pdf_loader.py                     # Medical PDF parser & cleaner
│   ├── retriever.py                      # ChromaDB retrieval & filter logic
│   └── vector_store.py                   # ChromaDB vector store wrapper
├── frontend/                             # Web interface assets
├── data/                                 # Medical literature PDFs & knowledge sources
└── test_images/                          # Sample test skin condition images
```

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
cd medical_rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Launch the FastAPI server with web UI
python app.py
```

---

## 🗺️ Roadmap
* [x] **Phase 1: Current MVP** — 5 skin conditions, EfficientNet-B2 classifier, ChromaDB RAG, Gemma 3 integration, interactive UI.
* [ ] **Phase 2: Expansion** — Expand to 10+ skin conditions, extended nutrition/lifestyle database, user history & progress tracking.
* [ ] **Phase 3: Platform** — Cross-platform mobile app, clinic SaaS, and dermatologist review dashboard.
* [ ] **Phase 4: Ecosystem** — Multilingual support, automated follow-ups, and telemedicine partner integration.
