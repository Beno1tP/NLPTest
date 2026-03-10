# Vietnamese AI Call Center - Tech Stack

## Overview

Full-stack task-oriented dialogue system with 3-way NLU model comparison.

---

## Core Stack

### Language & Runtime
| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| Package Manager | pip/uv | latest |
| Environment | venv/conda | - |

### Deep Learning
| Component | Technology | Notes |
|-----------|------------|-------|
| Framework | PyTorch | 2.0+ |
| Transformers | HuggingFace | 4.30+ |
| Pre-trained Model | PhoBERT-base-v2 | `vinai/phobert-base-v2` |

---

## NLU Models (3-way Comparison)

### 1. TF-IDF + SVM Baseline
| Component | Library | Config |
|-----------|---------|--------|
| Vectorizer | scikit-learn TfidfVectorizer | ngram_range=(1,2), max_features=10000 |
| Classifier | scikit-learn LinearSVC | C=1.0, class_weight='balanced' |
| Slot Filling | sklearn-crfsuite CRF | algorithm='lbfgs' |
| Tokenizer | underthesea | word_tokenize() |

### 2. JointBERT + PhoBERT
| Component | Library | Config |
|-----------|---------|--------|
| Encoder | vinai/phobert-base-v2 | 135M params |
| Intent Head | Linear + Softmax | CLS token |
| Slot Head | Linear + CRF | token embeddings |
| Segmenter | py_vncorenlp (VnCoreNLP) | wseg annotator |

**Training Hyperparameters:**
- Learning rate: 4e-5
- Batch size: 32 (8 if CPU)
- Epochs: 15
- Warmup ratio: 0.1
- Max seq length: 128

### 3. LLM Zero-Shot
| Component | API | Config |
|-----------|-----|--------|
| Provider | Anthropic Claude | claude-3-haiku |
| Fallback | OpenAI GPT-4o-mini | - |
| Output Format | Tool Use (structured JSON) | - |
| Temperature | 0.0 | deterministic |

---

## Speech Pipeline

### Speech-to-Text (STT)
| Provider | Model | Use Case |
|----------|-------|----------|
| Google Cloud | Chirp 3 (V2 API) | Production |
| faster-whisper | large-v3-turbo | Offline/Free |
| PhoWhisper | VinAI | Accent robustness |

**Config:**
- Language: `vi-VN`
- Sample rate: 16kHz
- Encoding: LINEAR16/FLAC

### Text-to-Speech (TTS)
| Provider | Voice | Use Case |
|----------|-------|----------|
| Google Cloud | vi-VN-Neural2-A | Production |
| edge-tts | vi-VN-HoaiMyNeural | Free alternative |
| gTTS | vi | Basic prototype |

---

## Dialogue Management

### Dialogue State Tracking (DST)
- **Method:** Rule-based state accumulation
- **State:** Intent history, slot values, turn count

### Policy
- **Method:** Rule-based action selection
- **Actions:** Confirm, RequestSlot, APICall, Response

### Natural Language Generation (NLG)
- **Method:** Template-based responses
- **Templates:** Vietnamese domain-specific

---

## Demo UI

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | Streamlit | 1.34+ |
| Audio Input | st.audio_input |
| Chat | st.chat_message + st.chat_input |
| Layout | st.sidebar, st.columns |

### Visualization
- Pipeline progress: `st.status`
- Metrics: `st.metric`
- Expandable details: `st.expander`

---

## Data

### Dataset
| Attribute | Value |
|-----------|-------|
| Name | PhoATIS |
| Source | VinAI Research |
| Samples | 5,871 |
| Intents | 28 |
| Slot Types | 82 |
| Domain | Air travel |

### Data Format
```
data/processed/
├── train/
│   ├── seq.in      # Segmented input
│   ├── seq.out     # BIO slot labels
│   └── label       # Intent labels
├── dev/
└── test/
```

---

## Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Intent Accuracy | correct/total | >90% |
| Intent F1 Macro | macro avg F1 | >85% |
| Slot F1 | seqeval micro F1 | >85% |
| Sentence Accuracy | both correct | >75% |

---

## Dependencies

### Core
```
torch>=2.0.0
transformers>=4.30.0
datasets
scikit-learn
seqeval
underthesea
py-vncorenlp
sklearn-crfsuite
pytorch-crf
```

### Speech
```
google-cloud-speech
google-cloud-texttospeech
faster-whisper
edge-tts
pydub
SpeechRecognition
```

### LLM
```
anthropic
openai
```

### UI & Utilities
```
streamlit>=1.34.0
pandas
numpy
matplotlib
seaborn
pyyaml
python-dotenv
tqdm
```

---

## Environment Variables

```env
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GOOGLE_CLOUD_API_KEY=your_key

# LLM APIs (optional for zero-shot)
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key

# App
DEBUG=false
LOG_LEVEL=INFO
```

---

## Hardware Requirements

### Development (CPU)
- RAM: 16GB minimum
- Storage: 5GB for models
- Note: JointBERT training slow on CPU

### Production (GPU)
- VRAM: 6GB+ for faster-whisper/PhoBERT
- Recommended: NVIDIA RTX 3060+

---

## Project Structure

```
NLPsubject/
├── configs/           # YAML configurations
├── data/              # PhoATIS dataset
│   ├── raw/
│   └── processed/
├── src/
│   ├── nlu/           # SVM, JointBERT, LLM classifiers
│   ├── speech/        # STT/TTS wrappers
│   ├── dst/           # Dialogue state tracker
│   ├── policy/        # Action selection
│   ├── nlg/           # Response generation
│   ├── pipeline/      # End-to-end orchestration
│   └── evaluation/    # Metrics computation
├── app/               # Streamlit demo
├── scripts/           # Training/eval scripts
├── tests/             # Unit tests
├── results/           # Model outputs
├── report/            # Academic report
└── docs/              # Documentation
```
