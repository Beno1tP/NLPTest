# Vietnamese AI Call Center — NLP Course Project

## Mục tiêu
Build một AI Call Center demo cho Vietnamese customer support (air travel domain). User nói chuyện bằng giọng nói (hoặc text), hệ thống hiểu → xử lý → trả lời bằng giọng nói (hoặc text). Đây là project môn NLP, năm 4, team 3 người, 2 tuần.

## Demo flow
```
User nói (voice) → [STT API] → text → [NLU → DST → Policy → NLG] → text → [TTS API] → Bot nói (voice)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^
                                        ★ CORE RESEARCH — phần này tự build
```

STT và TTS chỉ dùng API (không train), highlight là phần Task-Oriented Dialogue ở giữa.

## Core research highlight
So sánh 3 paradigm NLP trên task joint intent classification + slot filling:
1. Traditional ML: TF-IDF + SVM
2. Fine-tuned Transformer: JointBERT với PhoBERT (vinai/phobert-base-v2)
3. Zero-shot LLM: Claude/GPT prompting (mock results nếu không có API key)

## Dataset
PhoATIS — Vietnamese version of ATIS, từ https://github.com/VinAIResearch/JointIDSF
- 5,871 samples, 28 intents, 82 slot types, air travel domain
- Format: 3 files mỗi split (train/dev/test): seq.in (text), seq.out (BIO slot labels), label (intent)

## Full system architecture

### Layer 1: Speech-to-Text (API only — không train)
- Dùng OpenAI Whisper API hoặc Google Speech-to-Text API
- Nếu không có API key: dùng library `speech_recognition` (free Google API) hoặc fallback text input
- Input: audio (microphone hoặc file) → Output: Vietnamese text
- Cần xử lý: noise, silence detection, Vietnamese diacritics

### Layer 2: Task-Oriented Dialogue Pipeline (★ CORE — tự build + research)

**Module 2.1 — NLU (Natural Language Understanding) ★ core research**
- Input: text string
- Output: { intent, confidence, slots: { slot_name: value } }
- 3 models so sánh:
  - TF-IDF + SVM (baseline)
  - JointBERT + PhoBERT (primary — vinai/phobert-base-v2)
  - LLM zero-shot (modern comparison)
- JointBERT architecture:
  - Encoder: PhoBERT (768 hidden)
  - Intent head: Linear(768, num_intents) on CLS token
  - Slot head: Linear(768, num_slots) on all tokens
  - Loss: intent_CE + slot_CE (ignore_index=-100 cho subword padding)
  - Subword alignment: first subword giữ slot label, còn lại = -100
  - Hyperparams: lr=5e-5, batch=32 (8 nếu CPU), epochs=15, warmup=10%, AdamW, early stopping patience=3

**Module 2.2 — DST (Dialogue State Tracking) — rule-based**
- Input: previous belief state + NLU output
- Output: updated belief state dict
- Rule-based dictionary tracking: new value → update, user correction → reset slot
- Slots: fromloc.city_name, toloc.city_name, depart_date.day_name, depart_date.month_name, depart_date.day_number, depart_time.time, airline_name, flight_number, class_type, round_trip

**Module 2.3 — Policy — rule-based decision tree**
- Input: belief state + intent + confidence
- Output: action dict
- Actions: clarify (confidence < 0.5), greet, request_slot(slot), confirm(params), provide_info(info), execute(params), escalate
- Logic: if unknown → clarify, if missing slots → request first missing, if all filled → confirm, if confirmed → execute

**Module 2.4 — NLG (Natural Language Generation) — template-based**
- Input: action dict
- Output: Vietnamese response text
- ~20-30 templates:
  - request fromloc: "Bạn muốn bay từ thành phố nào ạ?"
  - request toloc: "Bạn muốn bay đến đâu ạ?"
  - request date: "Bạn muốn bay vào ngày nào ạ?"
  - request time: "Bạn muốn bay lúc mấy giờ ạ?"
  - request airline: "Bạn muốn đi hãng hàng không nào ạ?"
  - confirm: "Xác nhận: Chuyến bay {airline} từ {fromloc} đến {toloc} ngày {date}. Đúng không ạ?"
  - execute: "Đã tìm thấy chuyến bay cho bạn! Mã đặt chỗ: {booking_id}"
  - clarify: "Xin lỗi, em chưa nghe rõ. Anh/chị có thể nói lại được không ạ?"
  - greet: "Xin chào! Em là trợ lý hàng không ảo. Em có thể giúp gì cho anh/chị?"
  - escalate: "Em sẽ chuyển anh/chị đến nhân viên hỗ trợ. Vui lòng chờ trong giây lát ạ."

### Layer 3: Text-to-Speech (API only — không train)
- Dùng Google TTS (gTTS — free, dễ), hoặc ElevenLabs API, hoặc Edge TTS
- Ưu tiên gTTS vì free và hỗ trợ Vietnamese tốt: `from gtts import gTTS`
- Input: Vietnamese text → Output: audio file/stream
- Cần xử lý: tốc độ nói phù hợp, natural prosody

## Evaluation cần có (cho phần NLU — core research)
- Intent accuracy, Intent F1 macro, Slot F1 (seqeval entity-level), Sentence accuracy
- Confusion matrix (top 15 intents)
- Error analysis: 20+ misclassified examples, categorized (ambiguous intent, rare intent, slot boundary error, OOV)
- Learning curve: train với 20/40/60/80/100% data, plot cho cả 3 models
- Model comparison table: accuracy, F1, training time, inference time
- Per-intent accuracy breakdown

## Demo UI — Streamlit
Giao diện call center demo với 2 modes:

**Mode 1: Voice Call (ấn tượng hơn)**
- Nút "🎤 Nói" để record voice → STT → pipeline → TTS → phát audio response
- Hiển thị transcript realtime
- Hoặc upload audio file

**Mode 2: Text Chat (fallback)**
- Chat interface bình thường, gõ text

**Sidebar hiển thị pipeline visualization:**
- STT transcript (nếu voice mode)
- NLU output: intent + confidence + detected slots
- DST: current belief state (update realtime mỗi turn)
- Policy: action taken
- NLG: generated response text
- Conversation history

**Layout:**
```
┌────────────────────────────────────────────────────┐
│  📞 Vietnamese AI Call Center                       │
│  [🎤 Voice Mode] [💬 Text Mode]                    │
├───────────────────────┬────────────────────────────┤
│                       │ 📊 Pipeline Visualization  │
│   Conversation        │                            │
│                       │ 🎙️ STT: "đặt vé đi đà nẵng"│
│   👤 Khách: ...       │ 🧠 NLU:                    │
│   🤖 Bot: ...         │   Intent: book_flight 96%  │
│   👤 Khách: ...       │   Slots: {toloc: Đà Nẵng}  │
│   🤖 Bot: ...         │ 📋 State: {toloc: Đà Nẵng} │
│                       │ 🎯 Policy: request_slot    │
│   [🔊 Playing...]     │ 💬 NLG: "Bay từ đâu ạ?"    │
│                       │                            │
├───────────────────────┴────────────────────────────┤
│  [🎤 Hold to speak] hoặc [Type message...] [Send] │
└────────────────────────────────────────────────────┘
```

## Project structure
```
ai-call-center/
├── CLAUDE.md
├── requirements.txt
├── configs/
│   ├── svm_config.yaml
│   ├── joint_bert_config.yaml
│   └── llm_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── speech/
│   │   ├── stt.py              # Speech-to-Text wrapper (API)
│   │   └── tts.py              # Text-to-Speech wrapper (API)
│   ├── nlu/
│   │   ├── svm_baseline.py     # TF-IDF + SVM
│   │   ├── joint_bert.py       # JointBERT model
│   │   ├── phobert_nlu.py      # PhoBERT fine-tuning
│   │   ├── llm_zeroshot.py     # LLM evaluation
│   │   └── trainer.py          # Training loop
│   ├── dst/
│   │   └── rule_based_dst.py
│   ├── policy/
│   │   └── rule_based_policy.py
│   ├── nlg/
│   │   └── template_nlg.py
│   ├── pipeline/
│   │   └── dialogue_system.py  # Full pipeline orchestrator
│   └── evaluation/
│       ├── metrics.py
│       ├── confusion_matrix.py
│       ├── error_analysis.py
│       └── learning_curve.py
├── scripts/
│   ├── train_svm.py
│   ├── train_joint_bert.py
│   ├── evaluate_llm.py
│   ├── evaluate_all.py
│   └── generate_report_figures.py
├── app/
│   └── streamlit_app.py       # Call center demo UI
├── tests/
├── results/
│   ├── figures/
│   └── tables/
└── report/
    └── sections/
```

## Dependencies
```
torch>=2.0
transformers>=4.30
datasets
scikit-learn
seqeval
pandas
numpy
matplotlib
seaborn
streamlit
streamlit-webrtc          # for voice recording in browser
pyyaml
tqdm
underthesea               # Vietnamese word segmentation
gTTS                      # Google Text-to-Speech (free)
SpeechRecognition         # STT wrapper
PyAudio                   # microphone access
pydub                     # audio processing
anthropic                 # Claude API (optional)
openai                    # Whisper API / GPT (optional)
pytest
```

## Speech module details

### STT (src/speech/stt.py)
```python
class SpeechToText:
    # Ưu tiên thứ tự:
    # 1. OpenAI Whisper API (tốt nhất cho Vietnamese) — cần API key
    # 2. Google Speech Recognition (free, okay cho Vietnamese)
    # 3. Fallback: text input
    
    def transcribe(audio_file_or_stream) -> str:
        # Returns Vietnamese text
    
    def record_from_mic(duration=5) -> str:
        # Record → transcribe → return text
```

### TTS (src/speech/tts.py)
```python
class TextToSpeech:
    # Ưu tiên:
    # 1. gTTS (free, good Vietnamese) — pip install gTTS
    # 2. Edge TTS (free, better quality) — pip install edge-tts
    # 3. ElevenLabs (best quality, paid)
    
    def speak(text: str) -> audio_bytes:
        # Convert Vietnamese text → audio
    
    def save(text: str, filepath: str):
        # Save audio to file
```

## Constraints
- STT/TTS chỉ dùng API hoặc library có sẵn — KHÔNG train model speech
- Core research focus 100% vào NLU comparison (SVM vs PhoBERT vs LLM)
- Nếu không có GPU: batch_size=8, epochs=5, max_seq_length=64
- Nếu không có API key cho LLM: mock results (intent ~75%, slot ~65%)
- Nếu microphone không hoạt động trong Streamlit: fallback text mode
- Nếu VnCoreNLP không cài được: dùng underthesea thay thế
- Giữ giọng nói bot tự nhiên, lịch sự (dùng "ạ", "anh/chị")

## Deliverables
1. Trained NLU models (SVM + PhoBERT checkpoint)
2. Full pipeline: STT → NLU → DST → Policy → NLG → TTS
3. Streamlit demo với voice + text mode
4. Evaluation results: tables, figures, confusion matrix, learning curve, error analysis
5. Unit tests cho core modules
6. Report sections (markdown, academic English, 8-12 pages): introduction, related work, methodology, dataset, experiments, results, analysis, limitations, conclusion

## Bắt đầu
Tạo project structure, download PhoATIS, preprocess data, implement STT/TTS wrappers, train 3 NLU models, build full pipeline, build call center demo UI, run evaluations, generate figures, viết report. Làm từ đầu đến cuối.
