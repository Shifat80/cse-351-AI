## Week 2: Knowledge Representation and Logic

### 🧠 Concept Overview
AI systems need a way to **represent knowledge** about the world and reason about it. This is called **Knowledge Representation (KR)**. Logic provides a formal way to represent and manipulate facts and rules.

---

### 🔹 Why Knowledge Representation?
- To allow an AI system to **store**, **reason**, and **infer** new information.
- Example: If we know that *All humans are mortal* and *Socrates is human*, we can infer that *Socrates is mortal*.

---

### 🔹 Propositional Logic (PL)
Propositional Logic uses **propositions (statements)** that can be **true or false**.

#### Example:
- P: It is raining.
- Q: The ground is wet.

**Rules:**
- Negation: ¬P → It is not raining
- Conjunction: P ∧ Q → It is raining AND the ground is wet
- Disjunction: P ∨ Q → It is raining OR the ground is wet
- Implication: P → Q → If it is raining, then the ground is wet

#### 🧩 Truth Table Example
| P | Q | P → Q |
|---|---|--------|
| T | T | T |
| T | F | F |
| F | T | T |
| F | F | T |

---

### 🔹 Inference
**Inference** means deriving new facts from known facts.

**Common inference rules:**
1. **Modus Ponens**:  
   If `P → Q` and `P` are true, then `Q` must be true.
2. **Modus Tollens**:  
   If `P → Q` and `¬Q` are true, then `¬P` must be true.
3. **Resolution**: Used in automated theorem proving.

---

### 💻 Python Implementation — Logical Inference
We can simulate simple propositional logic inference using Python sets.

```python
# Knowledge Base (KB)
KB = {
    "Human(Socrates)",
    "All x: Human(x) -> Mortal(x)"
}

# Inference Rule Simulation
def infer(kb, query):
    if query == "Mortal(Socrates)":
        if "Human(Socrates)" in kb and "All x: Human(x) -> Mortal(x)" in kb:
            return True
    return False

print(infer(KB, "Mortal(Socrates)"))  # Output: True
```

---

### 🔹 First-Order Logic (FOL)
FOL extends propositional logic with **quantifiers** and **predicates**.

#### Elements:
- **Constants:** Socrates, Paris
- **Predicates:** Human(x), Loves(x, y)
- **Variables:** x, y
- **Quantifiers:**
  - ∀ (For all): ∀x Human(x) → Mortal(x)
  - ∃ (There exists): ∃x Cat(x)

#### Example:
"All humans are mortal" → ∀x (Human(x) → Mortal(x))

"Socrates is human" → Human(Socrates)

By inference: Mortal(Socrates)

---

### 💻 Python Simulation — Rule-Based Reasoning
```python
# Simple rule-based inference engine
rules = {
    "Human": "Mortal"
}

facts = ["Socrates is Human"]

def infer_rule(facts, rules):
    new_facts = []
    for fact in facts:
        entity, _, category = fact.split()
        if category in rules:
            new_facts.append(f"{entity} is {rules[category]}")
    return new_facts

new_knowledge = infer_rule(facts, rules)
print(new_knowledge)  # ['Socrates is Mortal']
```

---

### 🔹 Logical Agents
A **logical agent** uses its knowledge base to make decisions.

#### Agent Cycle:
1. **Perceive** the environment.
2. **Update** the knowledge base.
3. **Infer** new information.
4. **Act** based on conclusions.

---

### 🧩 Mini Project: Logical Puzzle Solver
**Goal:** Build a Python program that solves a small logical puzzle.

**Puzzle:**
- A, B, C are people.
- Exactly one of them is telling the truth.
- A says: "B is lying."
- B says: "A and C are lying."
- C says nothing.

Find who is telling the truth.

**Hint:** Try all combinations of truth values for A, B, and C, and check which satisfies the statements.

---

### ✅ Practice Ideas
1. Build a small rule-based chatbot using if-then logic.
2. Implement a forward-chaining inference system.
3. Use the `sympy.logic` module in Python to automate truth table generation.

---

Next Week → **Week 3: Uncertainty and Probability**  
We’ll explore probabilistic reasoning, Bayesian networks, and Python simulations of uncertain environments.

