# SemDiff-QA-Semantic-Differentiation-for-Multiple-Choice-QA

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Note:** This project implements a Differentiating Choices framework inspired by the [**DCQA**](https://arxiv.org/pdf/2408.11554) methodology (ECAI 2024), optimized for resource-constrained environments.

## Overview

Standard Multiple-Choice Question Answering (MCQA) models often struggle with "semantic drift," where the model attends to common features shared among distractors rather than the distinguishing nuances of the correct answer. 

This project implements a **Discriminative Ranking Framework** using a **T5-Base** backbone. By restructuring the generation task into a list-wise ranking task, the model learns to implicitly filter out commonality and focus on the semantic differentiation required to answer complex commonsense reasoning questions.

##  Key Features & Optimizations

This implementation is optimized for resource-constrained environments (specifically free-tier Colab T4 GPUs) while maintaining high-fidelity training standards:

*   **Architecture:** T5-Base Encoder with a custom Mean-Pooling + Linear Classification head.
*   **Mixed Precision Training (FP16):** Utilizes `torch.cuda.amp` to reduce memory footprint by ~40% and accelerate computation.
*   **Gradient Accumulation:** Implements virtual batching (Micro-batch: 2, Accumulation Steps: 8) to simulate an effective batch size of 16, ensuring stable convergence without OOM errors.
*   **Gradient Clipping:** Prevents exploding gradients common in Transformer fine-tuning.

## Performance

Evaluated on the **CommonsenseQA (CSQA)** validation set:

| Model Setting | Accuracy | Notes |
| :--- | :--- | :--- |
| Random Guessing | 20.00% | 5-way classification |
| T5-Small Baseline | ~40.0% | Limited capacity |
| **DCQA (T5-Base)** | **~60.52%** | **Current Implementation** |

##  Methodology

### 1. Input Linearization
Unlike standard T5 text generation, we treat each `(Question, Choice)` pair as a distinct hypothesis.
```text
Input: "question: Where do you put your shoes... choice: foyer"
Label: 1 (if correct) / 0 (if incorrect)
```

### 2. Model Architecture
Instead of generating text, we extract the latent semantic vector from the Encoder:
1.  **Encode:** Pass input through T5 Encoder $\rightarrow$ Hidden States.
2.  **Pool:** Mean Pooling over the sequence length.
3.  **Score:** Project the pooled vector to a scalar logit via a Linear Layer.
4.  **Rank:** Apply Cross-Entropy Loss over the 5 choices.

## 📂 Project Structure

```
├── data/
│   ├── csqa/              # CommonsenseQA Dataset
│   └── dummy/             # Unit test data
├── model/
│   └── dcqa.py            # T5ForConditionalGeneration wrapper
├── main.ipynb             # Jupyter Notebook containing full pipeline
├── utils.py               # Data loading and tokenization helpers
└── README.md
```

##  Installation & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch 2.0+
*   Transformers 4.30+

### Installation
```bash
git clone https://github.com/yourusername/dcqa-implementation.git
cd dcqa-implementation
pip install transformers torch tqdm sentencepiece numpy matplotlib seaborn
```

### Training
The project is designed to run within a Jupyter Notebook or Google Colab environment. 

1.  Open `semdiffqa.ipynb`.
2.  Run the **Configuration Cell** to select hyperparameters:
    ```python
    class Config:
        model_name = "t5-base"
        batch_size = 2
        accumulation_steps = 8
        lr = 1e-4
    ```
3.  Execute the training loop.

### Inference Example
To use the trained model for prediction:

```python
from model.dcqa import DCQAModel

# Load Model
model = DCQAModel("t5-base")
# ... load weights ...

# Predict
question = "Where do you put your shoes when you enter a Japanese house?"
choices = ["roof", "fridge", "foyer", "bed", "kitchen"]

predict_answer(model, question, choices)
# Output: "foyer" (Correctly identifies cultural nuance)
```

## 📜 References

*   **Original Paper:** [Differentiating Choices via Commonality for Multiple-Choice Question Answering (ECAI 2024)](https://arxiv.org/pdf/2408.11554)
*   **Dataset:** [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa).

