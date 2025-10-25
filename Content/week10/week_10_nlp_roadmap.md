# Week 10: Natural Language Processing (NLP) & AI/ML Roadmap

### 🎯 Learning Objectives
By the end of this week, you’ll understand:
- Basics of Natural Language Processing (NLP)
- Text preprocessing: tokenization, stopwords, stemming, lemmatization
- Text representation: Bag-of-Words, TF-IDF, Word Embeddings (Word2Vec, GloVe)
- Sequence modeling: using RNNs/LSTMs for NLP
- Suggested roadmap for advanced AI/ML topics beyond CS50 AI

---

## 🧠 1. NLP Basics
Natural Language Processing is a branch of AI that enables computers to understand, interpret, and generate human language.

### Core tasks in NLP:
- Text Classification: Sentiment analysis, spam detection
- Named Entity Recognition (NER): Extracting names, places, dates
- Machine Translation: English → French, etc.
- Question Answering: Chatbots, virtual assistants

---

## ⚙️ 2. Text Preprocessing
Text must be cleaned and converted into numerical format before feeding into models.

| Step | Description |
|------|-------------|
| Tokenization | Split text into words or sentences |
| Lowercasing | Standardize words |
| Stopwords removal | Remove common words (the, is, and) |
| Stemming | Reduce words to root (running → run) |
| Lemmatization | Convert to base dictionary form |

**Python Example:**
```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

text = "I am learning AI and it is amazing!"
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
ps = PorterStemmer()
tokens = [ps.stem(w) for w in tokens]
print(tokens)
```

---

## 🧩 3. Text Representation
### Bag-of-Words (BoW)
Counts word frequency, ignores order.
```python
from sklearn.feature_extraction.text import CountVectorizer
texts = ["I love AI", "AI is amazing"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
print(X.toarray())
``` 

### TF-IDF
Accounts for word importance across documents.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
print(X.toarray())
```

### Word Embeddings
- Word2Vec, GloVe, FastText
- Maps words to dense vectors capturing semantic meaning
```python
from gensim.models import Word2Vec
sentences = [['i','love','ai'],['ai','is','amazing']]
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1)
print(model.wv['ai'])
```

---

## ⚡ 4. Sequence Modeling for NLP
- Use **RNNs/LSTMs/GRUs** for tasks like text generation, sentiment analysis, and translation.
- Input: sequences of word indices or embeddings
- Output: classification or next-word prediction

---

## 📈 5. Suggested Roadmap After CS50 AI / Week 10
To become an AI/ML expert, follow these steps:

### Step 1: Strengthen Fundamentals
- Linear algebra, probability, statistics
- Python programming and NumPy, Pandas

### Step 2: Core ML Concepts
- Supervised Learning: Regression, Classification
- Unsupervised Learning: Clustering, Dimensionality Reduction
- Evaluation metrics (accuracy, precision, recall, F1)

### Step 3: Deep Learning
- Neural Networks, CNNs, RNNs, LSTMs, GRUs
- Transformers (attention mechanism)
- GANs (Generative Adversarial Networks)

### Step 4: NLP & Computer Vision
- NLP: Transformers (BERT, GPT), embeddings
- Computer Vision: Image classification, object detection, segmentation
- Libraries: TensorFlow, PyTorch, Hugging Face

### Step 5: Reinforcement Learning
- Q-learning, Deep Q-Networks (DQN)
- Policy gradients
- Applications: Game AI, robotics

### Step 6: Real-world Projects
- Deploy ML models using Flask, FastAPI, or Streamlit
- Data pipelines, ETL, cloud services (AWS, GCP)
- Kaggle competitions to gain experience

### Step 7: Continuous Learning
- Research papers (arXiv, paperswithcode.com)
- Advanced topics: Self-supervised learning, diffusion models, LLM fine-tuning

---

## ✅ 6. Summary
- Week 10 covers **NLP basics**, preprocessing, text representation, and sequence modeling.
- Beyond CS50 AI, focus on deep learning, NLP, computer vision, RL, and deployment.
- Practical experience via projects and competitions is key to mastery.

---

💡 Next Step: Start building small NLP projects using datasets like IMDB, Twitter Sentiment, or News Classification and combine it with your RNN/LSTM knowledge to solidify your skills.