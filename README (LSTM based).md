# 🧬 DNAcodec: Hybrid Transformer-BiLSTM-CNN for Noisy DNA Sequence Decoding

This project implements a hybrid deep learning architecture for **binary decoding and error localization** from **noisy DNA sequences**, tailored for **DNA-based data storage** systems. The model combines **Transformer encoders**, **BiLSTM layers**, and **CNNs** to perform accurate multi-task learning on a synthetic dataset with 5% sequencing noise.

## 📌 Key Features

- ✅ **Transformer Encoders** to learn contextual relationships in DNA sequences  
- 🔄 **Bidirectional LSTM** for robust temporal decoding of binary data  
- 🧠 **CNN** for high-resolution error localization  
- 📉 **Focal Loss** for handling class imbalance in error detection  
- 📊 Visualizations for error predictions and training performance  
- ⚙️ Advanced scheduling: cosine learning rate decay with warm-up  
- 📥 Dataset noise simulation and sequence-level preprocessing

---
## 🧪 Dataset

The dataset consists of DNA sequences with 5% induced noise:

- `DNA`: Original reference DNA sequence  
- `Noisy_DNA`: Mutated DNA after simulated sequencing noise  
- `Binary`: Corresponding 300-bit binary representation of each sequence  

> 📁 CSV file: `dna_storage_dataset_5percent_noise.csv`  
> 📏 Sequence Length: 150 bases  
> 🔢 Binary Length: 300 bits

---

## 🏗️ Model Architecture

### 🔧 Inputs
- One-hot encoded or index-encoded DNA sequence (`shape=(1, 150)`)

### 🧬 Encoding (Shared)
- `Embedding + N Transformer Blocks` with Multi-Head Attention  
- Positional dependencies modeled using attention

### 🧩 Dual Output Heads
1. **Binary Decoder**
   - BiLSTM + Dense layers  
   - Predicts 300-bit binary output (sigmoid activation)
2. **Error Localization**
   - BiLSTM + Conv1D layers  
   - Detects mutation positions (shape: 150)

---
## 🔍 Metrics & Loss

- **Binary Output**
  - Loss: Binary Crossentropy  
  - Metric: Bit Error Rate (BER)

- **Error Localization**
  - Loss: Focal Loss (class imbalance-aware)  
  - Metrics: Accuracy, F1-Score

---

## 📈 Training Strategy

- Optimizer: `Adam` with `clipvalue=0.5`  
- Learning Rate Schedule: Cosine decay with linear warm-up  
- Early Stopping: Patience = 10 epochs on `val_loss`  
- Epochs: 50  
- Batch Size: 64  

---


## 🧪 Evaluation Results 

```bash
Bit Error Rate (BER): 0.0213
Error Localization Accuracy (ELA): 0.9731
Error Localization F1 Score: 0.8415


