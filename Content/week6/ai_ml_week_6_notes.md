## Week 6: Neural Networks & Deep Learning Basics

### 🎯 Learning Objectives
By the end of this week, you’ll understand:
- What neural networks are and how they learn
- The structure of neurons and layers
- Activation functions
- Forward and backward propagation
- Gradient Descent optimization
- Implementation of a simple neural network in Python

---

## 🧠 1. What is a Neural Network?
A **Neural Network (NN)** is a computational model inspired by the human brain. It consists of **neurons (nodes)** arranged in layers that process data and learn patterns.

### **Basic Structure:**
1. **Input Layer** → Receives the input features (X)
2. **Hidden Layers** → Process and transform data
3. **Output Layer** → Produces the final prediction (Y)

---

### **Mathematical Representation:**
Each neuron performs the following:

$$
Z = W \cdot X + b
$$
$$
A = f(Z)
$$

Where:
- **W** = weights
- **b** = bias
- **f()** = activation function
- **A** = output (activation)

---

## ⚙️ 2. Activation Functions
Activation functions introduce **non-linearity** to help the network learn complex patterns.

| Function | Formula | Use Case |
|-----------|----------|-----------|
| **Sigmoid** | $ f(x) = \frac{1}{1 + e^{-x}} $ | Binary classification |
| **Tanh** | $ f(x) = \tanh(x) $ | Hidden layers (smooth gradient) |
| **ReLU** | $ f(x) = \max(0, x) $ | Deep networks, fast convergence |
| **Leaky ReLU** | $ f(x) = x \text{ if } x>0, 0.01x \text{ otherwise} $ | Avoids dead neurons |
| **Softmax** | Converts outputs to probabilities | Multi-class classification |

**Python Visualization:**
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,100)
plt.plot(x, 1/(1+np.exp(-x)), label='Sigmoid')
plt.plot(x, np.tanh(x), label='Tanh')
plt.plot(x, np.maximum(0,x), label='ReLU')
plt.legend(); plt.title('Activation Functions'); plt.show()
```

---

## 🔁 3. Forward Propagation
The process of moving data through the network from input to output.

Steps:
1. Multiply inputs by weights and add bias: `Z = WX + b`
2. Apply activation function: `A = f(Z)`
3. Repeat for each layer.

**Example:**
If $ X = [x_1, x_2] $ and weights $ W = [w_1, w_2] $:

$$
A = f(w_1x_1 + w_2x_2 + b)
$$

---

## 🔁 4. Backpropagation
Backpropagation is the process of updating weights to minimize the loss.

**Steps:**
1. Compute loss (difference between predicted and actual output)
2. Find gradient of loss w.r.t. each weight
3. Update weights using gradient descent

**Weight update rule:**
$$
W = W - \eta \frac{\partial L}{\partial W}
$$
Where $ \eta $ is the **learning rate**.

---

## 🧮 5. Gradient Descent
Gradient Descent is an optimization algorithm to minimize the loss function.

| Type | Description |
|-------|--------------|
| **Batch GD** | Uses entire dataset each update (slow) |
| **Stochastic GD (SGD)** | Updates weights after each sample (fast but noisy) |
| **Mini-batch GD** | Updates after a batch of samples (best balance) |

**Python Example:**
```python
# Simple gradient descent demo
y = lambda x: (x-3)**2  # function to minimize

def grad(x):
    return 2*(x-3)

x = 10
lr = 0.1
for i in range(10):
    x -= lr * grad(x)
    print(f"Iter {i+1}: x={x:.2f}, loss={y(x):.4f}")
```

---

## 🧠 6. Building a Simple Neural Network (from scratch)
```python
import numpy as np

# Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])  # XOR problem

# Initialize parameters
np.random.seed(1)
W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))
W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))

# Activation function
def sigmoid(x): return 1/(1+np.exp(-x))

def sigmoid_deriv(x): return x*(1-x)

# Training
lr = 0.1
for epoch in range(10000):
    # Forward
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Backpropagation
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2)
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_deriv(A1)
    dW1 = np.dot(X.T, dZ1)

    # Update
    W1 -= lr * dW1
    W2 -= lr * dW2

# Prediction
print("Predictions:", (A2 > 0.5).astype(int))
```

---

## 🧾 7. Loss Functions
| Task | Common Loss Function |
|------|-----------------------|
| Regression | Mean Squared Error (MSE) |
| Binary Classification | Binary Cross-Entropy |
| Multi-Class | Categorical Cross-Entropy |

---

## 🚀 8. Applications
- Handwritten digit recognition (MNIST)
- Image classification (CNNs)
- Text sentiment analysis (RNNs, LSTMs)
- Speech recognition, recommendation systems

---

## ✅ Summary
| Concept | Description |
|----------|-------------|
| Neuron | Basic processing unit |
| Activation | Adds non-linearity |
| Forward Propagation | Compute predictions |
| Backpropagation | Update weights using gradients |
| Gradient Descent | Optimization algorithm |
| Loss Function | Measures model error |

---

**Next Week → Week 7: Deep Learning Architectures (CNNs & RNNs)** — we’ll explore how neural networks process images and sequential data.