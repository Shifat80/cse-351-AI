**Week 4: Sampling & Estimation in AI/ML**

---

### **1. Sampling Techniques**
Sampling is the process of selecting a subset (sample) from a large population to make statistical inferences.

#### **Types of Sampling:**

**(a) Random Sampling:**
Every element has an equal chance of being selected.
- Example: Using `random.sample()` in Python.
- Advantage: Minimizes bias.

**Python Example:**
```python
import random
population = list(range(1, 101))  # numbers 1 to 100
sample = random.sample(population, 10)
print(sample)
```

**(b) Stratified Sampling:**
Population divided into groups (strata) and samples taken from each group proportionally.
- Used when population has subgroups (like gender, age group).

**(c) Systematic Sampling:**
Select every k-th item after a random start.
- Example: From 1000 records, pick every 10th.

**(d) Cluster Sampling:**
Divide population into clusters; randomly choose clusters and include all members.

**(e) Convenience Sampling:**
Based on easy availability — may introduce bias.

---

### **2. Sampling Distribution**
A sampling distribution is the probability distribution of a given statistic (like mean) based on a random sample.

#### Example:
If we repeatedly take samples from a population and compute their means, the distribution of those means is the **sampling distribution of the mean**.

- The **mean of the sampling distribution** = population mean (μ)
- The **standard deviation of the sampling distribution** = standard error (SE)

$$
SE = \frac{\sigma}{\sqrt{n}}
$$

---

### **3. Central Limit Theorem (CLT)**

> No matter the shape of the population distribution, the sampling distribution of the mean approaches a **normal distribution** as sample size increases (n > 30 usually sufficient).

**Python Simulation:**
```python
import numpy as np
import matplotlib.pyplot as plt

population = np.random.exponential(scale=2, size=10000)
sample_means = [np.mean(np.random.choice(population, 30)) for _ in range(1000)]
plt.hist(sample_means, bins=30)
plt.title('Sampling Distribution Approaches Normality')
plt.show()
```

---

### **4. Point and Interval Estimation**

#### **Point Estimation:**
A single value used to estimate a population parameter.
- Example: Sample mean $ \bar{X} $ estimates population mean $ \mu $

#### **Interval Estimation (Confidence Interval):**
A range of values that likely contains the population parameter.

$$
CI = \bar{X} \pm Z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}
$$

For 95% confidence:
$ Z_{\alpha/2} = 1.96 $

**Python Example:**
```python
import numpy as np
from scipy import stats

sample = np.random.normal(50, 10, 100)
mean = np.mean(sample)
sem = stats.sem(sample)
ci = stats.norm.interval(0.95, loc=mean, scale=sem)
print('95% Confidence Interval:', ci)
```

---

### **5. Standard Error (SE)**

- SE measures the variability of a sample statistic.
$$
SE = \frac{\sigma}{\sqrt{n}}
$$
- As sample size increases, SE decreases — meaning larger samples give more reliable estimates.

---

### **6. Bias and Variance**

| Concept | Description | Effect |
|----------|--------------|---------|
| **Bias** | Error from wrong assumptions (underfitting) | High bias → model too simple |
| **Variance** | Error from sensitivity to small fluctuations in training data (overfitting) | High variance → model too complex |

**Goal:** Minimize both bias and variance for good generalization.

---

### **7. Bootstrap Sampling (Resampling Technique)**
Used to estimate the distribution of a statistic by resampling with replacement.

**Python Example:**
```python
import numpy as np
sample = np.array([5, 10, 15, 20, 25])
bootstrap_means = [np.mean(np.random.choice(sample, size=len(sample), replace=True)) for _ in range(1000)]
print('Bootstrap Estimate:', np.mean(bootstrap_means))
```

---

### **8. Applications in AI/ML**
- Model validation using bootstrapping.
- Estimating confidence intervals for performance metrics (accuracy, F1-score).
- Feature importance estimation.
- Bayesian parameter estimation.

---

### ✅ **Summary**
| Concept | Key Idea |
|----------|-----------|
| Sampling | Selecting representative data from population |
| CLT | Sampling mean tends to be normal |
| Point Estimation | Single best guess of parameter |
| Interval Estimation | Range of plausible parameter values |
| Standard Error | Measure of sampling variability |
| Bias-Variance Tradeoff | Balance between simplicity and complexity |
| Bootstrap | Repeated resampling for better estimation |