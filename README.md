# LLaMA 3.2B Patent Classification using 4-bit Quantization

This project demonstrates the fine-tuning of Meta's LLaMA 3.2B Large Language Model (LLM) using 4-bit quantization for efficient and accurate patent classification on the **CCDV Patent Dataset**. The primary goal is to develop a scalable and resource-efficient NLP solution capable of understanding and classifying complex legal and technical patent documents.

## 🚀 Project Overview

Patent classification is a critical task in IP management, legal tech, and innovation tracking. Leveraging state-of-the-art transformer-based models like **LLaMA 3.2B**, this project fine-tunes the model to classify patents based on their text descriptions using reduced compute resources via **4-bit quantization**.

## ⚙️ Methodology

1. **Dataset Preparation:**
   - Utilized the **CCDV Patent Classification Dataset**, which includes patent abstracts and metadata.
   - Preprocessed the dataset for model consumption (cleaning, tokenization, and formatting for input).

2. **Model Selection:**
   - Chose **Meta's LLaMA 3.2B** model due to its high performance in downstream NLP tasks and transformer-based architecture optimized for generative and classification tasks.
   
3. **Quantization Technique:**
   - Implemented **4-bit quantization** to reduce memory footprint and speed up training/inference while maintaining competitive accuracy.
   - Applied quantization-aware training to fine-tune efficiently on consumer-grade GPUs.

4. **Fine-Tuning Pipeline:**
   - Fine-tuned using **LoRA (Low-Rank Adaptation)** and **QLoRA** techniques, enabling memory-efficient training.
   - Optimized hyperparameters (learning rate, batch size, number of epochs) for convergence.
   - Used mixed-precision training for additional speedup.

5. **Evaluation:**
   - Assessed model performance using classification metrics such as **accuracy**, **F1-score**, **precision**, and **recall**.
   - Validated on a held-out test set of patent abstracts.

6. **Deployment-Ready:**
   - The quantized model is lightweight and optimized for deployment in resource-constrained environments (e.g., edge devices or cloud functions).

## 🛠️ Core Technologies

- **LLaMA 3.2B Model** (Meta AI)
- **4-bit Quantization (bitsandbytes)**
- **QLoRA (Quantized Low-Rank Adaptation)**
- **Hugging Face Transformers**
- **PyTorch**
- **Datasets library (Hugging Face)**
- **Mixed Precision Training (FP16)**
- **Google Colab (GPU Runtime)**

## 📊 Results

- Successfully fine-tuned a quantized LLaMA model to achieve high accuracy on patent classification.
- Demonstrated significant resource efficiency while maintaining high-quality predictions.

## 📈 Future Improvements

- Integrate additional metadata (e.g., assignee info, filing dates) for multi-modal classification.
- Explore distillation for even smaller model deployment.
- Deploy the model via FastAPI + Docker for production-ready API.

---

## 💡 Why this project matters

- Combines cutting-edge **large language models (LLMs)** with **model optimization techniques** like **quantization** and **LoRA**, making advanced NLP tasks feasible even on limited hardware.
- Tackles real-world **legal-tech** challenges relevant to enterprises, law firms, and IP consultancies.

---

## 🤝 Let's Connect

If you're interested in NLP, legal-tech, or efficient AI systems, feel free to connect!

**Email:** gauravhsn8@gmail.com
