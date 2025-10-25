# Week 7: Neural Networks and Deep Learning

### 🎯 Learning Objectives
By the end of this week, you’ll understand:
- The structure and intuition behind artificial neural networks (ANNs)
- How perceptrons and activation functions work
- The concept of forward and backward propagation
- Training deep networks using gradient descent
- Overfitting and regularization in deep learning

---

## 🧠 1. What is a Neural Network?
A **neural network** is a computational model inspired by the human brain. It’s made up of layers of nodes (neurons) that learn to recognize patterns through training.

### Structure:
- **Input layer:** Receives data
- **Hidden layers:** Learn features
- **Output layer:** Produces predictions

Each connection between neurons has a **weight** that determines how strong the signal is.

Example:
```
Input (x) → Hidden Layer (z) → Output (y)
```

---

## ⚙️ 2. The Perceptron
The **perceptron** is the simplest form of a neural network — a single neuron that learns a linear boundary.

### Formula:
$$ y = f(\sum w_i x_i + b) $$

Where:
- $ x_i $: inputs
- $ w_i $: weights
- $ b $: bias
- $ f $: activation function (e.g., step function)

Example (Binary classification):
```python
import numpy as np

def perceptron(x, w, b):
    z = np.dot(x, w) + b
    return 1 if z > 0 else 0
```

---

## ⚡ 3. Activation Functions
Activation functions add **non-linearity**, allowing the network to learn complex patterns.

Common examples:
| Function | Formula | Range | Use Case |
|-----------|----------|--------|-----------|
| Sigmoid | $ \frac{1}{1+e^{-x}} $ | (0,1) | Probabilities |
| Tanh | $ \tanh(x) $ | (-1,1) | Centered outputs |
| ReLU | $ \max(0, x) $ | [0,∞) | Fast convergence |
| Softmax | $ e^{x_i}/\sum e^{x_j} $ | (0,1) | Multi-class |

---

## 🔁 4. Forward Propagation
In forward propagation, data flows from input → output.

Each layer performs:
$$ z = w \cdot x + b $$
$$ a = f(z) $$

Where $ a $ is the activation output.

Example:
```python
# Simple forward propagation for one hidden layer
import numpy as np

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = np.dot(a1, w2) + b2
    y_hat = 1 / (1 + np.exp(-z2))  # Sigmoid
    return y_hat
```

---

## 🔙 5. Backpropagation and Gradient Descent
**Backpropagation** computes the gradient of the loss function with respect to each weight using the chain rule.

### Steps:
1. Perform forward propagation
2. Calculate loss (e.g., Mean Squared Error)
3. Compute gradients of loss wrt weights
4. Update weights:  
$$ w = w - \eta * \frac{\partial L}{\partial w} $$

Where $ \eta $ = learning rate.

---

## 📉 6. Overfitting and Regularization
Overfitting happens when the model learns noise instead of the pattern.

### Techniques to reduce it:
- **Dropout:** Randomly deactivate neurons during training.
- **Early stopping:** Stop training when validation loss stops improving.
- **L2 Regularization:** Penalize large weights.

Example (L2 term):
$$ L = Loss + \lambda \sum w^2 $$

---

## 🧪 7. Hands-On Mini Project
Train a simple neural network using **scikit-learn**.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Define model
model = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=500)
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```

Output:
```
Accuracy: ~97%
```

---

## 🧩 8. Summary
- Neural networks are made of interconnected neurons with weights.
- Activation functions allow non-linearity.
- Training uses forward and backward propagation.
- Deep learning extends neural networks to multiple layers.

---

## 🧠 Challenge
1. Implement a 2-layer neural network from scratch (without sklearn).
2. Experiment with ReLU, sigmoid, and tanh.
3. Visualize training accuracy and loss over epochs.

---

Next Week (Week 8): **Convolutional Neural Networks (CNNs)** – image recognition and fe