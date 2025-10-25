# Complete AI/ML Study Guide (Week 1–10)

This guide consolidates all weeks from the CS50 AI-like curriculum with theory, Python examples, mini-projects, and suggested roadmap.

---

## Week 1: Introduction to AI & Python Basics
- **Topics:** AI definition, applications, Python refresher, data structures
- **Mini-project:** Implement a simple chatbot using `input()` and conditional statements
- **Python Example:**
```python
name = input("What's your name?")
print(f"Hello, {name}!")
```

---

## Week 2: Probability & Statistics
- **Topics:** Probability rules, random variables, distributions (Binomial, Poisson, Normal)
- **Python Example:**
```python
import numpy as np
np.random.binomial(n=10, p=0.5, size=1000)
```
- **Mini-project:** Simulate dice rolls and plot probability distributions

---

## Week 3: Search Algorithms
- **Topics:** BFS, DFS, Uniform Cost, A*, Greedy Search
- **Python Example:** Implement BFS/DFS for graph traversal
- **Mini-project:** Maze solver using A* search

---

## Week 4: Knowledge Representation & Logic
- **Topics:** Propositional logic, first-order logic, inference, satisfiability
- **Python Example:** Using `pyDatalog` for logic queries
- **Mini-project:** Build a rule-based expert system for disease diagnosis

---

## Week 5: Constraint Satisfaction Problems (CSP)
- **Topics:** Backtracking, forward checking, heuristics
- **Python Example:** Solve Sudoku using backtracking
- **Mini-project:** Simple map coloring problem solver

---

## Week 6: Neural Networks & Deep Learning Basics
- **Topics:** Neurons, layers, activation functions, forward/backward propagation, gradient descent
- **Python Example:** Simple 2-layer NN for XOR problem
- **Mini-project:** Train NN using `scikit-learn` on digits dataset

---

## Week 7: Advanced Neural Networks
- **Topics:** Deep architectures, overfitting, regularization, dropout
- **Python Example:** 2-layer NN from scratch, activation comparisons
- **Mini-project:** Visualize hidden layer activations

---

## Week 8: Convolutional Neural Networks (CNNs)
- **Topics:** Convolution, pooling, feature maps, modern CNN architectures (VGG, ResNet)
- **Python Example:** CNN on CIFAR-10 using Keras/TensorFlow
- **Mini-project:** MNIST digit classifier with CNN

---

## Week 9: Recurrent Neural Networks (RNNs)
- **Topics:** Sequential data, hidden states, LSTM, GRU, vanishing gradient problem
- **Python Example:** IMDB sentiment analysis using LSTM
- **Mini-project:** Generate text sequences using a simple RNN

---

## Week 10: Natural Language Processing (NLP) & AI/ML Roadmap
- **Topics:** Text preprocessing, tokenization, stopwords, stemming, lemmatization, Bag-of-Words, TF-IDF, Word2Vec, embeddings, sequence modeling
- **Python Example:** Text preprocessing and embedding
- **Mini-project:** Sentiment analysis on IMDB dataset

### Suggested Roadmap Post-Week 10
1. **Strengthen Fundamentals:** Linear algebra, probability, statistics, Python
2. **Core ML Concepts:** Supervised & unsupervised learning, evaluation metrics
3. **Deep Learning:** CNNs, RNNs, LSTMs, GRUs, Transformers, GANs
4. **NLP & CV:** Word embeddings, BERT/GPT, image classification, object detection
5. **Reinforcement Learning:** Q-learning, DQN, policy gradients
6. **Projects & Deployment:** Build and deploy models with Flask/Streamlit, Kaggle competitions
7. **Advanced Learning:** Research papers, diffusion models, LLM fine-tuning

---

This guide provides theory, Python implementation, and hands-on projects, forming a **complete foundation for AI/ML mastery**.