# DNAcodec: Hybrid Transformer-BiLSTM-CNN for Noisy DNA Sequence Decoding

This project is a hybrid deep learning model for decoding binary data from noisy DNA sequences, built for DNA-based data storage systems. It combines a Transformer encoder, a BiLSTM, and a CNN to do two things at the same time: recover the original binary message and find where the sequencing errors happened.

## Key features

- Transformer encoder to learn context between bases in the DNA sequence
- Bidirectional LSTM for decoding the binary data
- CNN head for error localization at base-level resolution
- Focal loss to deal with class imbalance in error detection
- Cosine learning rate decay with warm-up
- Simple noise simulation and preprocessing for the synthetic dataset

## Dataset

The dataset is synthetic DNA storage data with 5% induced sequencing noise. Each row has:

- `DNA`: the original reference DNA sequence
- `Noisy_DNA`: the sequence after simulated sequencing noise
- `Binary`: the 300-bit binary message encoded in that sequence

File: `dna_storage_dataset_5percent_noise.csv`
Sequence length: 150 bases
Binary length: 300 bits

## Model architecture

**Input:** one-hot or index-encoded DNA sequence, shape `(1, 150)`

**Shared encoder:** embedding layer followed by several Transformer blocks with multi-head attention. Positional relationships between bases are learned through attention, no separate positional encoding trick needed.

The model has two output heads that share this encoder:

1. **Binary decoder head** — BiLSTM followed by dense layers, sigmoid activation, predicts the 300-bit binary output.
2. **Error localization head** — BiLSTM followed by Conv1D layers, predicts which of the 150 positions were mutated.

## Loss and metrics

- Binary output: binary crossentropy loss, evaluated with Bit Error Rate (BER)
- Error localization: focal loss (to handle the class imbalance between mutated and non-mutated positions), evaluated with accuracy and F1-score

## Training setup

- Optimizer: Adam, `clipvalue=0.5`
- Learning rate schedule: cosine decay with a linear warm-up
- Early stopping on `val_loss`, patience 10
- 50 epochs, batch size 64

## Results

```
Bit Error Rate (BER): 0.0213
Error Localization Accuracy (ELA): 0.9731
Error Localization F1 Score: 0.8415
```
