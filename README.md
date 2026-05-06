# 🧠 Modeling Cross-Paragraph Consistency for Few-Shot AI-Generated Text Detection: A Dual-Path Framework

## 🔥 TL;DR

We propose a **Dual-Path Detection Framework** for AI-generated text detection under **few-shot setting (200 samples/class)**.

* 🧩 Semantic Evidence Path → pretrained LM classifiers (local signals)
* 📊 Cross-Paragraph Consistency Path → discourse-level structural + stylistic consistency
* 🔗 Fusion → adaptive weighted ensemble with safety-aware calibration
* 🧪 Strong robustness under adversarial attack texts


---

# 📌 Overview

This repository implements the paper:

> **Modeling Cross-Paragraph Consistency for Few-Shot AI-Generated Text Detection: A Dual-Path Framework**

The key idea is that AI-generated text is not only detectable via **local semantic cues**, but also via **global inconsistency across discourse structure**.

---

# ⚙️ Framework

## 🧠 Architecture

```
Input Text
   │
   ├── Semantic Evidence Path
   │       └── Pretrained LM (RoBERTa / DeBERTa)
   │
   ├── Cross-Paragraph Consistency Path
   │       ├── Structural features
   │       ├── Perplexity features
   │       └── Stylistic features
   │
   └── Fusion Module
           └── S_final = α·S_text + (1−α)·S_cons
```

---

## 🔹 1. Semantic Evidence Path

We split long documents using sliding windows:

* Window size: `L = 512`
* Stride: `S = 256`

Each chunk is passed through pretrained models:

* RoBERTa-base
* RoBERTa-large
* DeBERTa-v3

Final score:

```text
S_text = mean(chunk_predictions)
```

---

## 🔹 2. Cross-Paragraph Consistency Path

We explicitly model **discourse-level inconsistency** across:

* Introduction
* Body
* Conclusion

### Feature groups:

### 📊 Structural (9-dim)

* readability gap
* Brunet index difference
* formality deviation

### 📉 Perplexity (5-dim)

* GPT-2 perplexity differences
* intra-doc asymmetry
* global variance

### 🎭 Stylistic (7-dim)

* POS distribution similarity
* NER distribution similarity
* trajectory consistency

---

### 🧩 Three-Level Stacked Fusion

We train:

* LR_struct
* LR_ppl
* LR_style

Then stack via meta-classifier:

```text
S_cons = Meta(LR_struct, LR_ppl, LR_style)
```

---

## 🔗 3. Final Fusion

```math
S_final = α·S_text + (1-α)·S_cons
```

### Adaptive weighting:

* α optimized on calibration set
* Safety rule:

  * if Consistency < Semantic − 0.003 → α = 1

---

# 📁 Repository Structure

```
├── Feature_extractions_终_(1).ipynb   # Feature extraction pipeline
├── dualpath_detection.ipynb           # Training + evaluation + experiments
├── data/
│   └── extracted_features.csv         # Precomputed features
└── README.md
```

---

# 🚀 Quick Start

## 1️⃣ Feature Extraction

Input must contain:

```text
intro_text
body_text
conclusion_text
```

Run:

```
Feature_extractions.ipynb
```

Output:

```
data/extracted_features.csv
```

---

## 2️⃣ Train + Evaluate

Run:

```
dualpath_detection.ipynb
```

Pipeline:

* Train 3 LR models
* Stack fusion
* Compute S_text / S_cons
* Optimize α
* Run ablation
* Run calibration analysis
* Generate figures
* Run attack robustness tests

---

# 🧪 Experiments

## 🔬 Ablation Study

| Setting     | Description            |
| ----------- | ---------------------- |
| Full Model  | Semantic + Consistency |
| Ablation A  | Semantic only          |
| Ablation B1 | Structural only        |
| Ablation B2 | Perplexity only        |
| Ablation B3 | Stylistic only         |

---

## 📈 Calibration Study

We vary calibration size:

```
25 → 50 → 100 → 200 → 400
```

Evaluate:

* AUROC
* Confidence interval width
* Fusion stability

---

## 🧠 Robustness (Attack Texts)

We include:

* adversarial paraphrasing
* structure perturbation
* lexical substitution

Evaluate model degradation under attack.

---

# 📊 Key Figures

Generated in `dualpath_detection.ipynb`:

* Cross-paragraph feature distributions (Fig 1–3)
* Calibration vs AUROC (Fig 5)
* Confidence interval analysis (Fig 6)
* Fusion convergence (Fig 7)

---

