# 261R0136COSE34102

# Ko-Binoculars: Zero-Shot Detection of AI-Generated Korean Text via NLL Dispersion Calibration

This repository contains the official implementation of **Ko-Binoculars**, the final project of **Team 14** for Korea University COSE461. The project introduces a zero-shot framework for detecting AI-generated Korean text without requiring any labeled training data.

---

## 📌 Key Contributions

### 1. Korean-Optimized Model Selection

- We replace the original English-centric Binoculars framework (Falcon-based) with Korean-optimized observer–performer model pairs.
- This simple replacement improves ROC-AUC from **67.54 to 95.40** on the KatFish benchmark without any additional training.

### 2. NLL Dispersion Calibration

- We observe that AI-generated text exhibits significantly lower token-level NLL (Negative Log-Likelihood) dispersion than human-written text.
- Based on this observation, we propose **Multiplicative Calibration** and **Gated Fusion**, which leverage NLL dispersion as an uncertainty-aware signal and further improve detection performance by **+2.44 ROC-AUC**.

### 3. Segment-level Detection

- We extend document-level detection to mixed-authorship scenarios using a sliding-window approach.
- The framework can localize AI-generated regions within a document and estimate the proportion of AI-generated content.
- Compared to a random baseline, Ko-Binoculars reduces AI ratio estimation error (MAE) by **45.7%**.

---

## ⚙️ Methodology

### 1. Ko-Binoculars Baseline

Ko-Binoculars follows the Binoculars framework by computing the ratio between the performer model's perplexity (PPL) and the observer–performer pair's cross-perplexity (XPPL).

$$
B(X) = \frac{PPL(X)}{XPPL(X)}
$$

where:

- **PPL(X)**: Perplexity computed by the performer model
- **XPPL(X)**: Cross-perplexity computed using the observer–performer pair

---

### 2. NLL Dispersion Calibration

For a given text $X$, we compute the standard deviation of token-level NLL values, denoted as $D(X)$:

$$
D(X) = \text{std}(NLL_1, NLL_2, ..., NLL_n)
$$

We then normalize the dispersion using human reference statistics $(\mu_H, \sigma_H)$:

$$
z(D(X)) = \frac{D(X)-\mu_H}{\sigma_H}
$$

Our analysis suggests that human-written text corresponds to a higher effective sampling temperature ($\tau \approx 1.38$), whereas AI-generated text exhibits a lower effective temperature ($\tau \approx 1.05 \sim 1.11$), resulting in more uniform token distributions.

The final calibrated score is defined as:

$$
S(X) = \text{Norm}(B(X)) \cdot \exp(-\lambda \cdot z(D(X)))
$$

where:

- **Norm(B(X))**: Min-Max normalized Binoculars score
- **z(D(X))**: Human-normalized NLL dispersion
- **λ**: Calibration strength hyperparameter

This formulation preserves the zero-shot nature of Binoculars while incorporating uncertainty-related information derived from token-level NLL statistics.

### 3. Gated Fusion

To further incorporate uncertainty information, we introduce a gated fusion mechanism:

$$
S(X)=w(X)\cdot Norm(B(X))
+(1-w(X))\cdot Norm(D(X))
$$

The gate dynamically adjusts the contribution of Binoculars and NLL Dispersion signals on a per-sample basis while preserving the zero-shot setting.

### 4. Segment-level Detection

We extend document-level classification to mixed-authorship documents using a sliding-window approach (window = 100, stride = 50). This enables localization of AI-generated regions and estimation of AI content ratio.

---

## 📊 Experimental Results

Performance is evaluated on the KatFish benchmark dataset, which consists of three genres (Essay, Abstract, and Poetry) and multiple Korean LLMs.

### 1. Observer–Performer Pair Comparison (Baseline)

All Ko-Binoculars variants operate in a fully zero-shot setting and require no additional training data.

| Method (Observer + Performer) | Training | ROC-AUC | PR-AUC | R@FPR5% | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Falcon-7B + Instruct (Original) | X | 67.54 | 41.00 | 66.48 | 55.27 |
| **Polyglot-Ko $1.3B + 3.8B$** | **X** | **96.62** | **66.65** | **75.60** | **65.27** |
| EXAONE $2.4B + 7.8B$ | X | 94.41 | 70.46 | 79.74 | 82.95 |
| Qwen2.5-3B base + inst. | X | 95.40 | 52.69 | 86.56 | 85.65 |
| *KatFishNet (Supervised)* | *✓* | *97.57* | *78.99* | *62.65* | *94.88* |

**Key Findings**

- Replacing English-centric Falcon models with Korean-optimized models dramatically improves detection performance.
- Ko-Binoculars achieves up to **96.62 ROC-AUC** without any supervised training.
- The performance gap between Ko-Binoculars and the supervised KatFishNet baseline becomes remarkably small.

### 2. NLL Dispersion Analysis

To understand why NLL dispersion is an effective detection signal, we analyze its relationship with sampling temperature. We observe a strong positive correlation between token-level NLL dispersion and generation temperature.

| Metric | Value |
| :--- | :---: |
| Pearson correlation | 0.73 |
| p-value | < 0.0001 |
| Human effective temperature | ≈ 1.38 |
| AI effective temperature | ≈ 1.05 ~ 1.11 |

**Key Findings**

- Human-written text exhibits substantially higher NLL dispersion than AI-generated text.
- Higher NLL dispersion indicates greater token-level unpredictability.
- This supports the use of NLL Dispersion as an uncertainty-aware calibration signal.
- If LLMs generate text with sufficiently high temperature, the dispersion gap may become smaller, making detection more difficult.

### 3. Calibration and Gated Fusion Performance (Qwen2.5-3B)

NLL Dispersion Calibration and Gated Fusion further improve detection performance without requiring any additional training data. Both methods preserve the zero-shot nature of Ko-Binoculars while leveraging uncertainty-related signals derived from token-level NLL dispersion.

| Method | Solar | Qwen2 | Llama3.1 | Avg |
| :--- | :---: | :---: | :---: | :---: |
| Binoculars Only | 89.46 | 82.78 | 84.29 | 85.51 |
| + Calibration | 88.87 | 88.52 | 86.77 | 88.06 |
| + Gated Fusion | 88.75 | 88.94 | 86.82 | **88.17** |

**Key Findings**

- Calibration improves average ROC-AUC from **85.51 → 88.06 (+2.55)**.
- Gated Fusion achieves the best overall performance (**88.17 ROC-AUC**).
- The largest improvement is observed on **Qwen2 (+5.74 ROC-AUC)**.
- Both approaches remain fully **training-free** and require no supervised learning.

### 4. Genre-wise Detection Performance (ROC-AUC)

Calibration and fusion provide substantial gains on Essay and Poetry, while Abstract remains challenging due to the stylistic similarity between human-written and AI-generated academic text.

| Genre | Binoculars Baseline | + Calibration | + Gated Fusion |
| :--- | :---: | :---: | :---: |
| **Essay** | 94.98 | 97.92 | **98.32** |
| **Poetry** | 86.84 | 92.82 | **93.14** |
| **Abstract** | 56.57 | 49.01 | 48.44 |

**Key Findings**

- Essay and Poetry benefit significantly from NLL-based calibration.
- Abstract remains a difficult domain due to its highly formulaic writing style.
- This highlights an important limitation of uncertainty-based signals in low-diversity domains.

### 5. Segment-level Detection

Unlike conventional document-level detectors, Ko-Binoculars can estimate the proportion of AI-generated content within a mixed document. Using a sliding-window approach (window = 100, stride = 50), the framework identifies AI-generated regions and estimates the AI content ratio.

| True AI Ratio | Ko-Binoculars MAE | Random Baseline |
| :---: | :---: | :---: |
| 0% | 0.121 | 0.500 |
| 25% | 0.090 | 0.250 |
| 50% | 0.175 | 0.000 |
| 75% | 0.205 | 0.250 |
| 100% | 0.225 | 0.500 |
| **Average** | **0.163** | **0.300** |

**Key Findings**

- Ko-Binoculars reduces AI ratio estimation error from **0.300 → 0.163**.
- This corresponds to a **45.7% reduction in MAE** compared to the random baseline.
- Performance remains stable regardless of the insertion position of AI-generated content.
- The framework can localize AI-generated segments without requiring any training data.

---

## 📂 Project Structure

```text
📁 Ko-Binoculars/
│
├── 📁 Codes/                                  # Source code and experiment notebooks
│   ├── 📄 Ko_Binoculars_Experiment.ipynb      # Baseline Ko-Binoculars experiments
│   ├── 📄 Calibration_and_Gated.ipynb         # NLL Calibration and Gated Fusion experiments
│   ├── 📄 Metrics_Experiment.ipynb            # Evaluation metrics and benchmark analysis
│   ├── 📄 NLP_score_측정.ipynb                # Linguistic and statistical score analysis
│   ├── 📄 temperture.ipynb                    # NLL Dispersion vs. Temperature analysis
│   ├── 📄 Ko_Binoculars_실험_로그.md          # Experiment log (Part 1)
│   └── 📄 Ko_Binoculars_실험_로그_Part2.md    # Experiment log (Part 2)
│
├── 📁 Report/                                 # Final report and project materials
│   ├── 📄 template.tex                        # Main LaTeX source
│   ├── 📄 references.bib                      # Bibliography and citation database
│   ├── 📄 neurips_2020.sty                    # NeurIPS conference style file
│   ├── 📄 fig1.png                            # Segment-level Detection figure
│   └── 📄 fig2.png                            # NLL Dispersion analysis figure
│
├── 📄 Ko_Binoculars.pdf                       # Final project report
├── 📄 README.md                               # Project overview, methodology, and results
└── 📄 .gitignore
```

---

## 👥 Team

**Team 14 (Korea University COSE461)**

- **SeongHyeon Mun** (Department of Computer Science, 2020320012)
- **GyeongMin Bae** (Department of Artificial Intelligent, 2025320605)

---
