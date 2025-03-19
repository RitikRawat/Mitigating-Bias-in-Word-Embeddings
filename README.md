# 🧠 Mitigating Bias in Word Embeddings: A Dual Approach

A CS596 course project focused on reducing gender bias in word embeddings using **Adversarial Debiasing** and **Hard-Debiasing** techniques.

## ✨ Overview

Word embeddings like GloVe and Word2Vec capture linguistic relationships but often carry harmful societal biases — especially gender bias. This project presents two debiasing methods to address this problem:

- **Adversarial Debiasing**: Uses a predictor and adversary network in a minimax setup to remove gender information from embeddings.
- **Hard-Debiasing**: Geometrically removes gender direction via PCA and projects embeddings to a gender-neutral subspace.

Our goal is to reduce gender bias in word embeddings **without sacrificing semantic quality**, enabling fairer downstream NLP applications.

---

## 🧪 Methods

### 🔹 Adversarial Debiasing

- **Predictor**: 4-layer linear network with residual connection.
- **Adversary**: 3-layer binary classifier to predict gender.
- **Training**:
  - Phase 1: Train predictor independently to reconstruct embeddings.
  - Phase 2: Alternate training using a Gradient Reversal Layer (GRL) so the predictor minimizes gender info.
- **Loss Functions**:
  - `MSE` for predictor (reconstruction error)
  - `Binary Cross-Entropy` for adversary

This setup encourages the predictor to learn embeddings that **retain meaning** but are **gender-neutral**.

---

### 🔹 Hard-Debiasing

- **Step 1: Bias Direction Computation**  
  - Use PCA on 10 paired gendered words (e.g., *he-she*, *man-woman*) to extract the **gender direction** in embedding space.

- **Step 2: Neutralizing Word Embeddings**  
  - Project occupation and gender-neutral words onto the subspace **orthogonal to the gender direction**.

- **Step 3: Equalization & Normalization**  
  - Normalize the debiased embeddings to maintain structure and preserve their relationships in other semantic dimensions.

This method **removes gender components geometrically** while keeping core meaning.

---

## 📊 Results

| Metric                 | Original | Hard-Debiasing | Adversarial Debiasing |
|------------------------|----------|----------------|------------------------|
| **Direct Bias**        | 0.0919   | 1.25 × 10⁻⁸    |      0.0052            |
| **RG-65 Similarity**   | 0.7317   | 0.7167         | 0.7242                 |
| **WordSim-353 Score**  | 0.6666   | 0.6521         | 0.6641                 |

### Key Observations:

- ✅ **Adversarial Debiasing** was highly effective at reducing direct gender bias.
- ✅ Both methods preserved semantic similarity well.
- ⚠️ Slight drop in semantic nuance with hard-debiasing.

---

## 📈 Evaluation Metrics

- **Direct Bias**: Mean cosine similarity between occupation words and the gender axis.
- **Semantic Quality**: Measured using:
  - [WordSim-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)
  - [RG-65](https://aclweb.org/anthology/W13-3516) dataset

---

## 📌 Insights

- Residual connections in the predictor help retain semantic structure.
- The GRL (Gradient Reversal Layer) was crucial for adversarial training.
- Hard-debiasing is simple and interpretable, but can slightly reduce embedding richness.
- Both methods successfully remove gender bias **without sacrificing NLP utility**.

---

## 🔮 Future Work

### 1. Multidimensional Bias Mitigation  
Extend methods to handle **racial, ethnic, and socio-economic biases**, not just gender.

### 2. Task-Specific Debiasing  
Tailor debiasing strength depending on downstream NLP task sensitivity.

### 3. Fine-Grained Evaluation  
Include:
- Bias across word categories (e.g., pronouns, adjectives)
- Word analogies and sentence-level bias tests
- Human evaluations of fairness and utility

---

## 📄 References

1. Bolukbasi, T. et al. (2016). *Man is to Computer Programmer as Woman is to Homemaker?*  
2. Zhao, J. et al. (2018). *Gender Bias in Contextualized Word Embeddings*  
3. Peters, M. E. et al. (2018). *Deep Contextualized Word Representations*  
4. Stanford NLP Group – *GloVe: Global Vectors for Word Representation*

## 📁 Project Structure

```bash
.
├── embedding_dev.ipynb         # Initial development and testing notebook
├── embedding_final.ipynb       # Final implementation and evaluation
├── glove.6B.300d.txt           # Pretrained GloVe embeddings (300d)
├── gender_pairs.txt            # Gender word pairs (e.g., man-woman, he-she)
├── occupation_words.txt        # Occupation words used for bias evaluation
├── README.md                   # This file
└── CS596 Final Report.pdf      # Project report with full methodology and results

