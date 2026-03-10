# JointBERT + PhoBERT for Vietnamese NLU

## 1. JointBERT Architecture

Joint intent classification + slot filling using shared BERT encoder.

```
Input: [CLS] tok1 tok2 ... tokN [SEP]
         |    |    |        |
       BERT Encoder (shared)
         |    |    |        |
      [h_CLS] [h1] [h2] ... [hN]
         |         \________/
    Intent Head    Slot Head (per-token)
    (softmax)      (CRF or softmax)
```

**Loss**: `total_loss = intent_loss + slot_coef * slot_loss`

**Key components**:
- CLS token -> intent classification (cross-entropy)
- Token embeddings -> slot labels (CRF preferred for sequence consistency)
- Optional: Intent-slot attention (JointIDSF) incorporates intent into slot prediction

## 2. PhoBERT-base-v2 Specifics

| Property | Value |
|----------|-------|
| Model ID | `vinai/phobert-base-v2` |
| Parameters | 135M |
| Max seq length | 256 tokens |
| Pre-training | 20GB Wiki+News + 120GB OSCAR |
| Architecture | RoBERTa-based |

**Critical requirement**: Input MUST be word-segmented before tokenization.

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2")

# CORRECT: pre-segmented input (underscores join multi-syllable words)
text = "Tôi muốn đặt vé máy_bay đi Hà_Nội"
inputs = tokenizer(text, return_tensors="pt")
```

## 3. Subword Alignment for Slot Filling

**Strategy**: First-token labeling - only first subword of each word gets the label.

```python
def align_labels_to_tokens(words, slot_labels, tokenizer, max_len=256):
    """Align word-level slot labels to subword tokens."""
    input_ids = [tokenizer.cls_token_id]
    slot_ids = [-100]  # ignore CLS

    for word, label in zip(words, slot_labels):
        word_tokens = tokenizer.tokenize(word)
        word_ids = tokenizer.convert_tokens_to_ids(word_tokens)

        if len(input_ids) + len(word_ids) > max_len - 1:
            break

        input_ids.extend(word_ids)
        # First token gets label, rest get -100 (ignored in loss)
        slot_ids.append(label)
        slot_ids.extend([-100] * (len(word_ids) - 1))

    input_ids.append(tokenizer.sep_token_id)
    slot_ids.append(-100)  # ignore SEP

    # Pad to max_len
    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    slot_ids += [-100] * pad_len
    attention_mask = [1] * (max_len - pad_len) + [0] * pad_len

    return input_ids, attention_mask, slot_ids
```

## 4. Training Hyperparameters

### Recommended (from JointIDSF/JointBERT repos)

| Parameter | JointBERT | JointIDSF (PhoBERT) |
|-----------|-----------|---------------------|
| Learning rate | 5e-5 | 4e-5 |
| Batch size | 32 | 16-32 |
| Epochs | 10 | 50 |
| Warmup | 0 | 0.1 ratio |
| Max seq len | 50 | 50-100 |
| Slot loss coef | 1.0 | 1.0 |
| Intent loss coef | - | 0.15 |
| Dropout | 0.1 | 0.1 |
| Optimizer | AdamW | AdamW |
| Adam epsilon | 1e-8 | 1e-8 |
| CRF | optional | recommended |

### Training loop pattern

```python
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=4e-5, eps=1e-8)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

for epoch in range(epochs):
    for batch in train_loader:
        intent_logits, slot_logits = model(batch)
        intent_loss = F.cross_entropy(intent_logits, batch["intent_labels"])
        slot_loss = compute_slot_loss(slot_logits, batch["slot_labels"])  # CRF or CE
        loss = intent_loss + slot_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

## 5. PhoATIS Dataset

First public Vietnamese intent detection + slot filling dataset (VinAI, INTERSPEECH 2021).

**Structure** (expected format):
```
data/PhoATIS/
├── train/
│   ├── seq.in      # Input sentences (word-segmented)
│   ├── seq.out     # Slot labels (BIO format)
│   └── label       # Intent labels
├── dev/
└── test/
```

**Example**:
```
# seq.in
tôi muốn đặt vé máy_bay từ Hà_Nội đến Đà_Nẵng

# seq.out
O O O O O O B-fromloc.city_name O B-toloc.city_name

# label
atis_flight
```

**Preprocessing pipeline**:
1. Word segment raw Vietnamese text (VnCoreNLP)
2. Convert to lowercase (optional)
3. Build intent/slot label vocabularies
4. Align labels to PhoBERT subwords

## 6. Word Segmentation: VnCoreNLP vs Underthesea

| Aspect | VnCoreNLP | Underthesea |
|--------|-----------|-------------|
| Accuracy | Higher (PhoBERT trained with this) | Good |
| Speed | Slower (Java) | Faster (pure Python) |
| Dependencies | Java 1.8+ | pip only |
| Memory | ~140MB models | Lighter |
| Recommendation | **Use for PhoBERT** | Quick prototyping |

### VnCoreNLP (recommended)

```python
import py_vncorenlp

# One-time setup
py_vncorenlp.download_model(save_dir='./vncorenlp')

# Usage
segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./vncorenlp')
text = "Tôi muốn đặt vé máy bay đi Hà Nội"
segmented = segmenter.word_segment(text)
# -> ['Tôi muốn đặt vé máy_bay đi Hà_Nội']
```

### Underthesea (alternative)

```python
from underthesea import word_tokenize

text = "Tôi muốn đặt vé máy bay đi Hà Nội"
segmented = word_tokenize(text, format="text")
# -> 'Tôi muốn đặt vé máy_bay đi Hà_Nội'
```

## 7. Model Architecture Code

```python
import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class JointPhoBERT(nn.Module):
    def __init__(self, num_intents, num_slots, dropout=0.1, use_crf=True):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        hidden_size = self.phobert.config.hidden_size  # 768

        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.slot_classifier = nn.Linear(hidden_size, num_slots)
        self.dropout = nn.Dropout(dropout)

        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_slots, batch_first=True)

    def forward(self, input_ids, attention_mask, slot_labels=None):
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])  # CLS

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        slot_loss = None
        if slot_labels is not None and self.use_crf:
            slot_loss = -self.crf(slot_logits, slot_labels,
                                   mask=attention_mask.bool(), reduction='mean')

        return intent_logits, slot_logits, slot_loss

    def decode_slots(self, slot_logits, attention_mask):
        if self.use_crf:
            return self.crf.decode(slot_logits, mask=attention_mask.bool())
        return slot_logits.argmax(dim=-1)
```

## References

1. Chen et al. "BERT for Joint Intent Classification and Slot Filling" (2019)
2. Dao et al. "Intent Detection and Slot Filling for Vietnamese" INTERSPEECH 2021
3. Nguyen & Nguyen "PhoBERT: Pre-trained language models for Vietnamese" EMNLP 2020
4. [JointBERT repo](https://github.com/monologg/JointBERT)
5. [JointIDSF repo](https://github.com/VinAIResearch/JointIDSF)
6. [PhoBERT HuggingFace](https://huggingface.co/vinai/phobert-base-v2)

## Unresolved Questions

- Exact PhoATIS label inventory (intent count, slot types) - need dataset access
- Optimal max_seq_len for Vietnamese airline domain (50 vs 100 vs 256)
- Fast tokenizer stability for PhoBERT v2 (PR pending in transformers)
