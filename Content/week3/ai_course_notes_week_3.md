## Week 3: Uncertainty and Probability in AI

### 🧠 Concept Overview
In real life, information is often **incomplete** or **uncertain**. Logic fails when facts are not strictly true or false — so AI uses **probability theory** to reason under uncertainty.

This week focuses on **probabilistic reasoning**, **Bayesian inference**, and **applications** like spam filtering or medical diagnosis.

---

### 🔹 Why Probability in AI?
- Logic assumes facts are 100% true or false.
- Probability allows reasoning with partial belief.

**Example:**
> It might rain tomorrow with probability 0.7.

AI uses this to make **rational decisions** even when outcomes are uncertain.

---

### 🔹 Basic Probability Concepts

#### 1. **Sample Space (S):**
All possible outcomes.
Example: Rolling a die → S = {1, 2, 3, 4, 5, 6}

#### 2. **Event (E):**
A subset of the sample space.
Example: E = {even numbers} = {2, 4, 6}

#### 3. **Probability of E:**
\( P(E) = \frac{\text{Favorable outcomes}}{\text{Total outcomes}} \)

---

### 🔹 Conditional Probability
Represents the probability of event A **given** that event B occurred.

\[ P(A|B) = \frac{P(A \cap B)}{P(B)} \]

**Example:**
- A: It’s cloudy
- B: It’s raining

Then `P(Rain | Cloudy)` is the probability it rains *given* it’s cloudy.

---

### 🔹 Bayes’ Theorem
One of the most important formulas in AI.

\[ P(H|E) = \frac{P(E|H) P(H)}{P(E)} \]

Where:
- H → Hypothesis (e.g., patient has disease)
- E → Evidence (e.g., test result is positive)

**Intuition:**
Updates the belief about a hypothesis given new evidence.

**Example:**
> If a test is 99% accurate and 1% of the population has a disease, what’s the probability that a person who tested positive actually has the disease?

**Python Simulation:**
```python
P_disease = 0.01
P_positive_given_disease = 0.99
P_positive_given_no_disease = 0.05

P_positive = P_positive_given_disease * P_disease + P_positive_given_no_disease * (1 - P_disease)

P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive
print(round(P_disease_given_positive, 4))  # ≈ 0.1667 → 16.7%
```

So even if the test is 99% accurate, the probability the person actually has the disease is only 16.7%.

---

### 🔹 Joint Probability & Independence

**Joint probability:** \( P(A, B) = P(A) * P(B|A) \)

**If A and B are independent:** \( P(A, B) = P(A) * P(B) \)

Example:
```python
P_rain = 0.3
P_traffic = 0.4

# Independent events
P_rain_and_traffic = P_rain * P_traffic
print(P_rain_and_traffic)  # 0.12
```

---

### 🔹 Bayesian Networks
A **Bayesian Network** is a directed acyclic graph (DAG) where:
- Nodes represent random variables.
- Edges represent conditional dependencies.

**Example:**
```
Rain → Wet Grass ← Sprinkler
```

Meaning: The grass can be wet because of rain **or** a sprinkler.

**Probability Table Example:**
| Rain | Sprinkler | Wet Grass | P(Wet Grass) |
|------|------------|------------|---------------|
| T | T | T | 0.99 |
| T | F | T | 0.9  |
| F | T | T | 0.8  |
| F | F | F | 0.0  |

---

### 💻 Python Simulation — Simple Bayesian Network
```python
import random

# Probability settings
P_Rain = 0.3
P_Sprinkler_given_Rain = 0.1
P_Sprinkler_given_NoRain = 0.5
P_Wet_given_Rain_Sprinkler = 0.99
P_Wet_given_Rain_NoSprinkler = 0.9
P_Wet_given_NoRain_Sprinkler = 0.8
P_Wet_given_NoRain_NoSprinkler = 0.0

def sample_event():
    rain = random.random() < P_Rain
    sprinkler = random.random() < (P_Sprinkler_given_Rain if rain else P_Sprinkler_given_NoRain)
    if rain and sprinkler:
        wet = random.random() < P_Wet_given_Rain_Sprinkler
    elif rain and not sprinkler:
        wet = random.random() < P_Wet_given_Rain_NoSprinkler
    elif not rain and sprinkler:
        wet = random.random() < P_Wet_given_NoRain_Sprinkler
    else:
        wet = random.random() < P_Wet_given_NoRain_NoSprinkler
    return rain, sprinkler, wet

# Simulation
trials = 10000
wet_count = 0
for _ in range(trials):
    _, _, wet = sample_event()
    if wet:
        wet_count += 1

print("Estimated P(Wet Grass):", wet_count / trials)
```
This simulates thousands of random events and estimates how often the grass ends up wet.

---

### 🔹 Naive Bayes Classifier (Core ML Algorithm)
Assumes features are independent and applies Bayes’ theorem to classify.

**Example:** Spam Detection.

**Formula:**
\[ P(Spam | Words) ∝ P(Words | Spam) * P(Spam) \]

**Python Example:**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Dataset
texts = [
    "win money now",
    "limited offer",
    "meet me at lunch",
    "project meeting schedule"
]
labels = [1, 1, 0, 0]  # 1=Spam, 0=Not spam

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Predict
msg = vectorizer.transform(["win free offer"])
print(model.predict(msg))  # Output: [1] → Spam
```

---

### 🧩 Mini Project: Medical Diagnosis Simulation
Build a Python program that calculates the probability of having a disease given symptoms and test results using **Bayes' theorem**.

**Hints:**
- Define prior probabilities (disease prevalence).
- Define conditional probabilities (test sensitivity & specificity).
- Calculate posterior probability `P(Disease | Positive)`.

---

### ✅ Practice Ideas
1. Implement a Naive Bayes classifier **from scratch** using counts and probabilities.
2. Visualize a simple Bayesian Network using `networkx`.
3. Simulate different conditional probability scenarios (e.g., weather, traffic, accidents).

---

Next Week → **Week 4: Optimization & Constraint Satisfaction Problems (CSP)**  
We’ll explore backtracking, local search, and solving puzzles like Sudoku or N-Queens using Python.