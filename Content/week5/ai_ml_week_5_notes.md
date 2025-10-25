## Week 5: Machine Learning Basics

### 🎯 Learning Objectives
By the end of this week, you will understand:
- The core concepts of **machine learning**
- Types of learning: **supervised**, **unsupervised**, and **reinforcement**
- Key algorithms: **Linear Regression**, **Logistic Regression**, **k-NN**, **Decision Trees**
- Model training, testing, and evaluation metrics

---

## 🧠 1. Introduction to Machine Learning

**Machine Learning (ML)** is a subset of AI that allows computers to learn from data and make predictions or decisions without being explicitly programmed.

**Basic idea:**
> The machine learns a function that maps input data (X) to output labels (Y).

$$Y = f(X)$$

---

## 🧩 2. Types of Machine Learning

| Type | Description | Examples |
|------|--------------|-----------|
| **Supervised Learning** | Learns from labeled data (input-output pairs). | Spam detection, price prediction |
| **Unsupervised Learning** | Finds patterns in unlabeled data. | Clustering, dimensionality reduction |
| **Reinforcement Learning** | Learns by interacting with an environment and receiving rewards. | Game AI, robotics |

---

## 📈 3. Supervised Learning

### 🔹 Linear Regression
Predicts continuous numeric values.

**Equation:**
$$y = mx + c$$

**Python Example:**
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Model Training
model = LinearRegression()
model.fit(X, y)

# Prediction
print(model.predict([[6]]))
```

**Output:** Predicted y for x = 6

---

### 🔹 Logistic Regression
Used for binary classification (Yes/No, True/False).

**Equation:**
$$p = \frac{1}{1 + e^{-(b_0 + b_1x)}}$$

**Python Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Dummy Data
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```

---

### 🔹 k-Nearest Neighbors (k-NN)
Classifies based on the majority class of k nearest data points.

**Python Example:**
```python
from sklearn.neighbors import KNeighborsClassifier

X = [[1,1], [2,2], [3,3], [6,6], [7,7]]
y = [0,0,0,1,1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print(knn.predict([[5,5]]))
```

---

### 🔹 Decision Trees
A tree-based model that splits data based on feature conditions.

**Python Example:**
```python
from sklearn.tree import DecisionTreeClassifier

X = [[1,1], [2,2], [3,3], [6,6], [7,7]]
y = [0,0,0,1,1]

tree = DecisionTreeClassifier()
tree.fit(X, y)

print(tree.predict([[5,5]]))
```

---

## 📊 4. Model Evaluation Metrics

| Metric | Description | Use Case |
|---------|-------------|-----------|
| **Accuracy** | (TP+TN)/(TP+FP+FN+TN) | Classification |
| **Precision** | TP/(TP+FP) | Positive prediction reliability |
| **Recall** | TP/(TP+FN) | Sensitivity |
| **F1-Score** | Harmonic mean of Precision & Recall | Balanced metric |
| **MSE/RMSE** | Measures prediction error in regression | Regression |

---

## 🧩 Mini Project: House Price Prediction

**Goal:** Predict house prices based on size and location.

**Steps:**
1. Collect dataset (e.g., `housing.csv`)
2. Preprocess (handle missing values)
3. Split data (train/test)
4. Train with Linear Regression
5. Evaluate performance using RMSE

**Python Skeleton:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('housing.csv')
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split
a,b,c,d = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(a, c)

pred = model.predict(b)
print("RMSE:", mean_squared_error(d, pred, squared=False))
```

---

## 🧠 Summary
- ML models learn from data instead of explicit programming.
- Supervised learning predicts known labels.
- Evaluation metrics are essential for model quality.
- Scikit-learn makes ML implementation easier.

---

**Next Week → Week 6: Neural Networks** — you’ll learn perceptrons, activation functions, and training via gradient descent.