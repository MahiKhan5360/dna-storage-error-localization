import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Configuration
class Config:
    VOCAB_SIZE = 5  # A, G, C, T + PAD
    MAX_DNA_LENGTH = 150
    BINARY_LENGTH = 300
    EMBEDDING_DIM = 128
    NUM_HEADS = 8
    FF_DIM = 256
    NUM_TRANSFORMER_BLOCKS = 3
    DROPOUT_RATE = 0.2  # Increased from 0.1
    BATCH_SIZE = 64
    EPOCHS = 50
config = Config()

# Preprocessing with Noise (Reduced to Dataset Noise Only)
def preprocess_data(df):
    char_to_idx = {'PAD': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}
    
    def dna_to_int(seq):
        return [char_to_idx.get(char, 0) for char in seq[:config.MAX_DNA_LENGTH]]
    
    def pad_sequence(seq, max_len):
        return seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
    
    X_noisy = np.array([pad_sequence(dna_to_int(seq), config.MAX_DNA_LENGTH) for seq in df["Noisy_DNA"]])
    X_combined = np.expand_dims(X_noisy, axis=1)  # Shape: (samples, 1, 150)
    
    y_binary = np.array([list(map(int, b)) + [0]*(300-len(b)) if len(b)<300 else list(map(int, b[:300])) for b in df["Binary"]])
    error_true = np.array([np.array([1 if n != c else 0 for n, c in zip(n_seq.ljust(150, 'A'), c_seq.ljust(150, 'A'))]) 
                          for n_seq, c_seq in zip(df["Noisy_DNA"], df["DNA"])])
    
    return X_combined, y_binary, error_true

# Metrics
def bit_error_rate(y_true, y_pred):
    y_true_float = K.cast(y_true, 'float32')
    y_pred_binary = K.cast(K.greater(y_pred, 0.5), 'float32')
    return K.mean(K.not_equal(y_true_float, y_pred_binary))

def error_localization_accuracy(y_true, y_pred):
    y_true_float = K.cast(y_true, 'float32')
    y_pred_binary = K.cast(K.greater(y_pred, 0.5), 'float32')
    return K.mean(K.equal(y_true_float, y_pred_binary))

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.1):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    alpha_weight = alpha * y_true + (1 - alpha) * (1 - y_true)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = K.pow(1 - p_t, gamma)
    ce_loss = -K.log(p_t)
    return K.mean(alpha_weight * focal_weight * ce_loss)

