# Week 9: Recurrent Neural Networks (RNNs)

### 🎯 Learning Objectives
By the end of this week, you’ll understand:
- What Recurrent Neural Networks (RNNs) are and how they handle sequential data
- The concept of memory and recurrence in networks
- Variants like LSTM and GRU
- Applications in text, speech, and time series
- How to build and train RNNs using Keras

---

## 🧠 1. Why RNNs?
Traditional neural networks assume all inputs are **independent**, which fails for **sequential data** — like sentences, time series, or speech — where order matters.

**RNNs** solve this by maintaining a **hidden state** that carries information from previous steps.

Example use cases:
- Predicting the next word in a sentence
- Stock price prediction
- Sentiment analysis

---

## ⚙️ 2. RNN Architecture

### Basic Structure:
```
Input (x₁, x₂, ..., xₜ) → Hidden States (h₁, h₂, ..., hₜ) → Output (y₁, y₂, ..., yₜ)
```

Each hidden state depends on both the **current input** and the **previous hidden state**:
\[ h_t = f(W_x x_t + W_h h_{t-1} + b) \]

Where:
- \( W_x \), \( W_h \): weight matrices
- \( b \): bias
- \( f \): activation function (often tanh)

---

## 🔁 3. Forward Propagation in RNNs

Each time step processes input sequentially:
```python
import numpy as np

# Simplified RNN forward step
def rnn_step(x_t, h_prev, W_x, W_h, b):
    h_t = np.tanh(np.dot(W_x, x_t) + np.dot(W_h, h_prev) + b)
    return h_t
```

The output at each step depends on all previous steps — giving RNNs their “memory”.

---

## ⚡ 4. The Vanishing Gradient Problem
When training long sequences, gradients can shrink (or explode), making it hard for RNNs to learn long-term dependencies.

### Solution:
Use advanced architectures:
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

---

## 🧩 5. LSTM (Long Short-Term Memory)
LSTM adds **gates** that control information flow.

### Gates:
| Gate | Purpose |
|------|----------|
| Forget gate | Decides what to forget |
| Input gate | Decides what to store |
| Output gate | Decides what to output |

Simplified equations:
\[ f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \]
\[ i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \]
\[ \tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \]
\[ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t \]
\[ o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \]
\[ h_t = o_t * \tanh(C_t) \]

---

## ⚙️ 6. GRU (Gated Recurrent Unit)
A simplified LSTM with two gates:
- **Update gate** \( z_t \)
- **Reset gate** \( r_t \)

It’s faster and requires fewer parameters.

---

## 🧠 7. Example: Text Generation with RNN
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

# Example sequence data
model = Sequential([
    Embedding(input_dim=5000, output_dim=64),
    SimpleRNN(128, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```
This model could be used for **sentiment analysis** or **sequence classification**.

---

## 🔮 8. Example: LSTM for Text Classification
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

model = Sequential([
    Embedding(10000, 128),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## 📈 9. Applications of RNNs
| Domain | Task |
|---------|------|
| NLP | Text generation, translation, sentiment analysis |
| Finance | Time series forecasting |
| Speech | Voice recognition |
| IoT | Sensor data prediction |

---

## 🧩 10. Summary
- RNNs handle sequential data using hidden states.
- LSTMs and GRUs solve the vanishing gradient issue.
- Widely used in NLP, speech, and time-series applications.
- Can be combined with CNNs for video or hybrid models.

---

## 💻 Mini Project: Sentiment Analysis on IMDB
```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load data
max_features = 10000
maxlen = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Build LSTM model
model = Sequential([
    Embedding(max_features, 128),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))
```

Output:
```
Validation Accuracy: ~88-90%
```

---

✅ **Next Week (Week 10):** Natural Language Processing (NLP) and Word Embeddings – tokenization, word2vec, and text vectorization techniques.