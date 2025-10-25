## Week 1: Search Algorithms (AI Fundamentals)

### 🧠 Concept Overview
AI search is the process of exploring possible states to reach a goal. Each state represents a situation, and actions move you between states.

#### 🔹 Types of Search
1. **Uninformed (Blind) Search** – No extra information about the goal.
   - Breadth-First Search (BFS)
   - Depth-First Search (DFS)
   - Uniform-Cost Search (UCS)

2. **Informed (Heuristic) Search** – Uses knowledge (heuristics) to guide the search.
   - Greedy Best-First Search
   - A* (A-star) Search

#### 🔹 Key Terms
- **State:** A configuration of the system.
- **Action:** Transition from one state to another.
- **Path Cost:** The total cost to reach a state.
- **Frontier:** The set of states to be explored next.
- **Goal Test:** A condition that determines if the goal is reached.

---

### 🧩 Breadth-First Search (BFS)
Explores all neighbors before going deeper.

**When to use:**
- The goal is close to the start.
- The graph/tree is shallow.

**Python Example:**
```python
from queue import Queue

def bfs(start, goal, graph):
    visited = set()
    q = Queue()
    q.put((start, [start]))

    while not q.empty():
        node, path = q.get()
        if node == goal:
            return path

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                q.put((neighbor, path + [neighbor]))

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': ['F'], 'F': []
}

print(bfs('A', 'F', graph))  # Output: ['A', 'C', 'F']
```

---

### 🧩 Depth-First Search (DFS)
Explores as far as possible along a branch before backtracking.

**When to use:**
- Memory is limited.
- The solution is deep.

**Python Example:**
```python
def dfs(start, goal, graph, path=None, visited=None):
    if visited is None:
        visited = set()
    if path is None:
        path = [start]

    visited.add(start)

    if start == goal:
        return path

    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            new_path = dfs(neighbor, goal, graph, path + [neighbor], visited)
            if new_path:
                return new_path

    return None

print(dfs('A', 'F', graph))  # Output: ['A', 'C', 'F']
```

---

### 🧩 Uniform-Cost Search (UCS)
Expands the node with the lowest cumulative cost.

**When to use:**
- Path costs vary.
- Finding the least-cost path.

**Python Example:**
```python
import heapq

def uniform_cost_search(graph, start, goal):
    pq = [(0, start, [start])]  # (cost, node, path)
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == goal:
            return path, cost
        if node in visited:
            continue
        visited.add(node)
        for neighbor, step_cost in graph.get(node, []):
            heapq.heappush(pq, (cost + step_cost, neighbor, path + [neighbor]))

# Example weighted graph
graph = {
    'A': [('B', 2), ('C', 5)],
    'B': [('D', 4), ('E', 1)],
    'C': [('F', 2)],
    'D': [], 'E': [('F', 3)], 'F': []
}

path, cost = uniform_cost_search(graph, 'A', 'F')
print(path, cost)  # Output: ['A', 'B', 'E', 'F'] 6
```

---

### ⭐ A* Search
Uses both path cost and a heuristic estimate of distance to the goal.

**Formula:**  
`f(n) = g(n) + h(n)`  
- `g(n)` = cost so far (path cost)
- `h(n)` = estimated cost to goal (heuristic)

**Python Example:**
```python
import heapq

def a_star_search(graph, start, goal, h):
    pq = [(h[start], 0, start, [start])]
    visited = set()

    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node == goal:
            return path, g
        if node in visited:
            continue
        visited.add(node)
        for neighbor, cost in graph.get(node, []):
            heapq.heappush(pq, (g + cost + h[neighbor], g + cost, neighbor, path + [neighbor]))

# Example
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 3)],
    'D': [], 'E': [('F', 1)], 'F': []
}
heuristics = {'A': 7, 'B': 6, 'C': 2, 'D': 3, 'E': 1, 'F': 0}

path, cost = a_star_search(graph, 'A', 'F', heuristics)
print(path, cost)  # Output: ['A', 'B', 'E', 'F'] 7
```

---

### 🧩 Mini Project: Maze Solver
Create a maze in a 2D grid (using lists). Implement BFS and A* to find the shortest path from start to goal.

**Hints:**
- Represent walls with `#`, open paths with `.`
- Use `(x, y)` coordinates as states.
- Heuristic for A*: Manhattan distance:  
  `h = abs(x1 - x2) + abs(y1 - y2)`

---

### ✅ Practice Ideas
1. Write BFS, DFS, and UCS from scratch without using built-in libraries.
2. Modify the A* code to visualize the search steps.
3. Implement a search to solve puzzles (e.g., 8-puzzle, N-Queens).

---

Next Week → **Week 2: Knowledge Representation & Logic**  
We’ll cover propositional logic, inference rules, and build a simple AI that solves logical puzzles using Python.

