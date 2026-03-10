# TF-IDF + SVM Baseline for Vietnamese Intent Classification

## 1. TF-IDF Vectorization

| Approach | Use Case | Recommendation |
|----------|----------|----------------|
| Word n-grams (1,2) | Primary - captures semantics | **Use with underthesea** |
| Char n-grams (3,5) | Typo-robust, no tokenizer | Supplementary only |

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize

def vietnamese_tokenizer(text):
    return word_tokenize(text, format="text").split()

vectorizer = TfidfVectorizer(
    tokenizer=vietnamese_tokenizer,
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True  # log(tf) scaling
)
```

## 2. SVM Kernel Selection

**Linear kernel preferred** for text classification:
- Text already high-dimensional, linearly separable
- RBF adds O(n^2) cost without accuracy gain
- Single hyperparameter (C) vs RBF (C + gamma)

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

svm = CalibratedClassifierCV(
    LinearSVC(C=1.0, class_weight='balanced', max_iter=5000),
    cv=3  # enables predict_proba
)
```

## 3. Vietnamese Preprocessing

```python
from underthesea import word_tokenize
import re

FIXED_WORDS = ["hạng thương gia", "khứ hồi", "Vietnam Airlines"]

def preprocess(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.lower()).strip()
    return word_tokenize(text, fixed_words=FIXED_WORDS, format="text")
# "đặt vé máy bay" -> "đặt vé_máy_bay"
```

## 4. Feature Engineering

| Feature | Impact | Implementation |
|---------|--------|----------------|
| TF-IDF word n-grams | High | Primary vectorizer |
| Question indicators | Medium | Binary: ai/gi/nao/dau |
| Sentence length | Low | Normalized numeric |

## 5. Slot Filling with CRF

SVM for intent, **CRF for slots** (sequence labeling requires context).

```python
import sklearn_crfsuite

def word2features(sent, i):
    word = sent[i]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.istitle()': word.istitle(),
    }
    if i > 0:
        features['-1:word.lower()'] = sent[i-1].lower()
    if i < len(sent)-1:
        features['+1:word.lower()'] = sent[i+1].lower()
    return features

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
```

## 6. Expected Benchmarks

| Model | Intent Acc | Slot F1 |
|-------|------------|---------|
| TF-IDF + SVM + CRF | 90-94% | 85-90% |
| PhoBERT JointBERT | 95-97% | 93-95% |

Gap: 3-5% due to Vietnamese segmentation errors, smaller data.

## 7. Complete Pipeline

```python
from sklearn.pipeline import Pipeline

class SVMIntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                tokenizer=vietnamese_tokenizer,
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True
            )),
            ('clf', CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight='balanced'),
                cv=3
            ))
        ])

    def fit(self, texts, labels):
        return self.pipeline.fit([preprocess(t) for t in texts], labels)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba([preprocess(t) for t in texts])
```

## 8. Key References

1. Joachims (1998) - SVM for text classification
2. Lafferty et al. (2001) - CRF for sequence labeling
3. underthesea - Vietnamese CRF-based word segmentation
4. PhoATIS - VinAI Vietnamese ATIS dataset

## Unresolved Questions

1. Optimal n-gram range (1,2) vs (1,3) needs empirical testing
2. underthesea vs VnCoreNLP performance comparison
3. SMOTE vs class_weight for rare intent handling
