# Week 8: Convolutional Neural Networks (CNNs)

### 🎯 Learning Objectives
By the end of this week, you’ll understand:
- What convolutional neural networks are and why they work well for images
- The building blocks of CNNs: convolution, pooling, and fully connected layers
- How feature maps and filters extract image patterns
- How to build and train a CNN in Python using Keras/TensorFlow

---

## 🧠 1. What is a CNN?
A **Convolutional Neural Network (CNN)** is a deep learning model designed for visual data like images. Instead of processing raw pixels directly, it uses **filters (kernels)** that scan the image to extract features like edges, shapes, and textures.

### Key Idea:
A CNN automatically learns spatial hierarchies — simple patterns at lower layers (edges), and complex objects at higher layers (faces, cars, etc.).

---

## ⚙️ 2. Architecture Overview
Typical CNN architecture:
```
Input → Convolution → ReLU → Pooling → (Repeat) → Fully Connected → Output
```

### Example:
| Layer Type | Function |
|-------------|-----------|
| Convolution | Extract features using filters |
| ReLU | Add non-linearity |
| Pooling | Reduce spatial dimensions |
| Fully Connected | Combine features for prediction |
| Softmax | Output class probabilities |

---

## 🧩 3. Convolution Operation
Each filter (e.g., 3×3 matrix) slides over the image to compute dot products.

### Formula:
$$ Feature\ Map = Input * Filter + Bias $$

Example:
```python
import numpy as np
from scipy.signal import convolve2d

image = np.array([[1,2,0],[0,1,2],[2,1,0]])
filter = np.array([[1,0],[0,-1]])
output = convolve2d(image, filter, mode='valid')
print(output)
```

Output:
```
[[1 2]
 [1 -1]]
```

---

## ⚡ 4. Pooling Layer
Pooling reduces the size of feature maps, helping prevent overfitting.

### Common types:
- **Max Pooling:** Takes the maximum value (most common)
- **Average Pooling:** Takes the average

Example:
```python
import numpy as np

def max_pooling(feature_map, size=2):
    pooled = feature_map.reshape(feature_map.shape[0]//size, size, -1, size).max(axis=(1,3))
    return pooled
```

---

## 🧮 5. Fully Connected Layers
After several convolution and pooling layers, the output is flattened and passed to dense (fully connected) layers that perform classification.

Example:
```
Flatten → Dense(128, activation='relu') → Dense(10, activation='softmax')
```

---

## 🧠 6. CNN in Practice (Keras Example)
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load data
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")
```

---

## 🔍 7. Visualization of Feature Maps
You can visualize what each filter learns:
```python
from tensorflow.keras.models import Model
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_test[:1])
```
This helps you understand what patterns (edges, textures) are being detected.

---

## 🧩 8. Common CNN Architectures
| Model | Key Feature | Use Case |
|--------|--------------|----------|
| LeNet-5 | Early CNN | Digit recognition |
| AlexNet | Deep, ReLU, Dropout | ImageNet (2012) |
| VGGNet | Stacked 3×3 Conv | Simple & effective |
| ResNet | Skip connections | Deep networks |
| MobileNet | Lightweight | Mobile devices |

---

## 🧠 9. Summary
- CNNs automatically learn spatial hierarchies from image data.
- Convolution and pooling layers reduce the need for manual feature extraction.
- Fully connected layers classify learned features.
- Modern CNNs (VGG, ResNet) power image recognition systems today.

---

## 💻 Challenge / Mini Project
Train a CNN on **MNIST** handwritten digits:
```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

---

✅ **Next Week (Week 9):** Recurrent Neural Networks (RNNs) – Understanding sequential models for text, time series, and language.