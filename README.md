# AISC Practicals --- 20 Python Algorithm Programs

A collection of 20 Python programs for Artificial Intelligence and Soft
Computing (AISC) practicals.

## Requirements

-   Python 3
-   Python standard library only; no third-party packages are required.
-   Run one program at a time by copying its code into a `.py` file.

## Table of Contents

-   [1. A\* Search Algorithm](#1-a-search-algorithm)
-   [2. Alpha-Beta Pruning](#2-alpha-beta-pruning)
-   [3. Single-Layer Perceptron (SLP)](#3-single-layer-perceptron-slp)
-   [4. Multi-Layer Perceptron (MLP) with
    Backpropagation](#4-multi-layer-perceptron-mlp-with-backpropagation)
-   [5. K-Means Clustering](#5-k-means-clustering)
-   [6. Hierarchical Clustering](#6-hierarchical-clustering)
-   [7. Associative Memory --- Hebbian
    Learning](#7-associative-memory-hebbian-learning)
-   [8. Genetic Algorithm](#8-genetic-algorithm)
-   [9. Simulated Annealing](#9-simulated-annealing)
-   [10. Fuzzy Logic Membership
    Function](#10-fuzzy-logic-membership-function)
-   [11. Delta Learning Rule](#11-delta-learning-rule)
-   [12. Fuzzy Logic Control System](#12-fuzzy-logic-control-system)
-   [13. Resolution Using First-Order
    Logic](#13-resolution-using-first-order-logic)
-   [14. Min-Max Tree](#14-min-max-tree)
-   [15. Hopfield Network](#15-hopfield-network)
-   [16. Water Jug Problem using BFS](#16-water-jug-problem-using-bfs)
-   [17. N-Queen using DFS](#17-n-queen-using-dfs)
-   [18. Breadth-First Search (BFS)](#18-breadth-first-search-bfs)
-   [19. Depth-First Search (DFS)](#19-depth-first-search-dfs)
-   [20. Feedforward ANN and
    Backpropagation](#20-feedforward-ann-and-backpropagation)

------------------------------------------------------------------------

## 1. A\* Search Algorithm

**Aim:** To implement 1. A\* Search Algorithm.

**Formula / principle:** Formula: f(n) = g(n) + h(n). g is the path cost
so far; h estimates the remaining cost.

**Algorithm:** Choose the candidate with lowest f, expand neighbours,
update costs, stop at the goal.

### Python Code

``` python
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2)],
    'C': [('D', 1)],
    'D': []
}
h = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
open_list = [(h['A'], 0, 'A', ['A'])]
visited = {}
while open_list:
    # Find minimum f-score manually
    best = 0
    for i in range(1, len(open_list)):
        if open_list[i][0] < open_list[best][0]:
            best = i
    f, g, node, path = open_list.pop(best)
    if node == 'D':
        print("Path:", path)
        print("Cost:", g)
        break
    if node in visited and visited[node] <= g:
        continue
    visited[node] = g
    for nxt, cost in graph[node]:
        ng = g + cost
        open_list.append((ng + h[nxt], ng, nxt, path + [nxt]))
```

**Expected output / viva note:** Expected: path A → B → D, cost 3. A\*
uses the priority queue to select the lowest f-score.

------------------------------------------------------------------------

## 2. Alpha-Beta Pruning

**Aim:** To implement 2. Alpha-Beta Pruning.

**Formula / principle:** Formula: update alpha = max(alpha, best at
MAX); beta = min(beta, best at MIN). Prune when alpha \>= beta.

**Algorithm:** Recursively evaluate the game tree; MAX takes the largest
value, MIN takes the smallest, and branches are pruned at a cutoff.

### Python Code

``` python
import math
def ab(d,i,ismax,V,alpha,beta,height):
    if d==height: return V[i]
    if ismax:
        best=-math.inf
        for j in range(2):
            best=max(best,ab(d+1,2*i+j,False,V,alpha,beta,height))
            alpha=max(alpha,best)
            if alpha>=beta: break
        return best
    best=math.inf
    for j in range(2):
        best=min(best,ab(d+1,2*i+j,True,V,alpha,beta,height))
        beta=min(beta,best)
        if alpha>=beta: break
    return best
print("Best value:",ab(0,0,True,[3,5,6,9],-math.inf,math.inf,2))
```

**Expected output / viva note:** Output: Best value 6.

------------------------------------------------------------------------

## 3. Single-Layer Perceptron (SLP)

**Aim:** To implement 3. Single-Layer Perceptron (SLP).

**Formula / principle:** Formula: z = sum(w\[i\]*x\[i\]) + b. Error =
target − prediction. Update w\[i\] += learning_rate*error*x\[i\]; b +=
learning_rate*error.

**Algorithm:** Initialize weights and bias, predict with a step
function, calculate error, update weights, repeat for each example.

### Python Code

``` python
X=[[0,0],[0,1],[1,0],[1,1]]; Y=[0,0,0,1]
w=[0.0,0.0]; b=0.0; lr=0.1
for epoch in range(20):
    for x,target in zip(X,Y):
        z=w[0]*x[0]+w[1]*x[1]+b
        pred=1 if z>0 else 0
        error=target-pred
        w[0]+=lr*error*x[0]; w[1]+=lr*error*x[1]; b+=lr*error
for x in X:
    z=w[0]*x[0]+w[1]*x[1]+b
    print(x,"->",1 if z>0 else 0)
```

**Expected output / viva note:** Expected AND outputs: 0, 0, 0, 1.

------------------------------------------------------------------------

## 4. Multi-Layer Perceptron (MLP) with Backpropagation

**Aim:** To implement 4. Multi-Layer Perceptron (MLP) with
Backpropagation.

**Formula / principle:** Sigmoid: σ(z)=1/(1+e\^(-z)). Output delta =
(target-output)*output*(1-output). Hidden delta =
h*(1-h)*weight*output_delta. Update weight +=
learning_rate*delta\*input.

**Algorithm:** Forward pass; calculate output error; calculate output
and hidden deltas; update weights and biases. This network learns XOR.

### Python Code

``` python
import math, random
random.seed(1)
X=[[0,0],[0,1],[1,0],[1,1]]; Y=[0,1,1,0]; lr=0.5
w1=[[random.uniform(-1,1) for _ in range(2)] for _ in range(2)]
b1=[0.0,0.0]; w2=[random.uniform(-1,1) for _ in range(2)]; b2=0.0
def sig(z):
    z=max(-40,min(40,z))
    return 1/(1+math.exp(-z))
for epoch in range(20000):
    for x,target in zip(X,Y):
        h=[sig(w1[j][0]*x[0]+w1[j][1]*x[1]+b1[j]) for j in range(2)]
        out=sig(w2[0]*h[0]+w2[1]*h[1]+b2)
        d2=(target-out)*out*(1-out)
        d1=[h[j]*(1-h[j])*w2[j]*d2 for j in range(2)]
        for j in range(2):
            w2[j]+=lr*d2*h[j]
            w1[j][0]+=lr*d1[j]*x[0]; w1[j][1]+=lr*d1[j]*x[1]
            b1[j]+=lr*d1[j]
        b2+=lr*d2
for x in X:
    h=[sig(w1[j][0]*x[0]+w1[j][1]*x[1]+b1[j]) for j in range(2)]
    out=sig(w2[0]*h[0]+w2[1]*h[1]+b2)
    print(x,"->",int(out>=0.5))
```

**Expected output / viva note:** Expected classes: 0, 1, 1, 0. This
manually implements forward propagation, backpropagation and updates.

------------------------------------------------------------------------

## 5. K-Means Clustering

**Aim:** To implement 5. K-Means Clustering.

**Formula / principle:** Distance: d(x,c)=sqrt(sum((x\[i\]-c\[i\])²)).
Centroid: mean of all points assigned to that cluster.

**Algorithm:** Choose K centroids; assign points to the nearest
centroid; recalculate centroids; repeat until assignments/centroids stop
changing.

### Python Code

``` python
import math
points=[[1,2],[2,3],[8,9],[9,8]]
centroids=[points[0][:],points[2][:]]
for iteration in range(10):
    labels=[]
    for p in points:
        distances=[]
        for c in centroids:
            distances.append(math.sqrt(sum((p[j]-c[j])**2 for j in range(2))))
        labels.append(distances.index(min(distances)))
    new_centroids=[]
    for k in range(2):
        group=[points[i] for i in range(len(points)) if labels[i]==k]
        center=[sum(p[j] for p in group)/len(group) for j in range(2)] if group else centroids[k]
        new_centroids.append(center)
    if new_centroids==centroids: break
    centroids=new_centroids
print("Labels:",labels)
print("Centroids:",centroids)
```

**Expected output / viva note:** Manual implementation: the two nearby
points at the left form one cluster and the two at the right form
another.

------------------------------------------------------------------------

## 6. Hierarchical Clustering

**Aim:** To implement 6. Hierarchical Clustering.

**Formula / principle:** Single-link distance = minimum distance between
any point in cluster A and any point in cluster B.

**Algorithm:** Start with one point per cluster; find the closest pair
of clusters; merge them; repeat until two clusters remain.

### Python Code

``` python
import math
P=[[1,2],[2,3],[8,9],[9,8]]
clusters=[[i] for i in range(len(P))]
def dist(i,j):
    return math.sqrt(sum((P[i][k]-P[j][k])**2 for k in range(2)))
while len(clusters)>2:
    best=None; minimum=float('inf')
    for a in range(len(clusters)):
        for b in range(a+1,len(clusters)):
            d=min(dist(i,j) for i in clusters[a] for j in clusters[b])
            if d<minimum: minimum=d; best=(a,b)
    a,b=best; clusters[a]+=clusters[b]; clusters.pop(b)
print("Clusters:",clusters)
```

**Expected output / viva note:** Manual agglomerative clustering using
single-link distance.

------------------------------------------------------------------------

## 7. Associative Memory --- Hebbian Learning

**Aim:** To implement 7. Associative Memory --- Hebbian Learning.

**Formula / principle:** Hebbian formula: w\[i\]\[j\] = w\[i\]\[j\] +
x\[i\]\*x\[j\]. Diagonal self-connections are set to zero.

**Algorithm:** Start with zero weights; add each pattern's pairwise
products; recall by taking the sign of each weighted sum.

### Python Code

``` python
patterns=[[1,-1,1],[-1,1,-1]]
n=len(patterns[0]); W=[[0]*n for _ in range(n)]
for p in patterns:
    for i in range(n):
        for j in range(n):
            W[i][j]+=p[i]*p[j]
for i in range(n): W[i][i]=0
x=[1,-1,1]; output=[]
for i in range(n):
    value=sum(W[i][j]*x[j] for j in range(n))
    output.append(1 if value>=0 else -1)
print("Weights:",W); print("Output:",output)
```

**Expected output / viva note:** Uses bipolar patterns −1 and +1.
Hebbian learning stores correlations between neuron states.

------------------------------------------------------------------------

## 8. Genetic Algorithm

**Aim:** To implement 8. Genetic Algorithm.

**Formula / principle:** Fitness: f(x)=x², to be maximised. Selection
chooses fit parents; crossover combines them; mutation introduces
variation.

**Algorithm:** Create a population; calculate fitness; select parents;
make offspring with crossover and mutation; repeat generations.

### Python Code

``` python
import random
random.seed(2)
population=[random.randint(0,31) for _ in range(6)]
for generation in range(30):
    population.sort(key=lambda x:x*x,reverse=True)
    parents=population[:3]; new_population=parents[:]
    while len(new_population)<6:
        a,b=random.choices(parents,k=2)
        child=(a+b)//2                 # crossover
        if random.random()<0.2:         # mutation
            child=random.randint(0,31)
        new_population.append(child)
    population=new_population
best=max(population,key=lambda x:x*x)
print("Best:",best,"Fitness:",best*best)
```

**Expected output / viva note:** Randomized result can vary. Fitness is
explicitly evaluated as x squared.

------------------------------------------------------------------------

## 9. Simulated Annealing

**Aim:** To implement 9. Simulated Annealing.

**Formula / principle:** For maximisation, accept a worse move with
probability P=exp((new_score-old_score)/T).

**Algorithm:** Try a nearby solution; accept it if better, or
probabilistically if worse; reduce temperature and repeat.

### Python Code

``` python
import random, math
random.seed(3)
def score(x): return -(x-5)**2+25
x=random.uniform(-10,10); best=x; T=100
while T>0.01:
    new=max(-10,min(10,x+random.uniform(-1,1)))
    change=score(new)-score(x)
    if change>0 or random.random()<math.exp(change/T): x=new
    if score(x)>score(best): best=x
    T*=0.95
print("Best x:",round(best,2))
print("Best score:",round(score(best),2))
```

**Expected output / viva note:** The maximum is at x=5 with score 25;
randomized search should approach it.

------------------------------------------------------------------------

## 10. Fuzzy Logic Membership Function

**Aim:** To implement 10. Fuzzy Logic Membership Function.

**Formula / principle:** Triangular membership rises from 0 at a to 1 at
b and falls to 0 at c.

**Algorithm:** Read x and parameters a,b,c; use the appropriate
piecewise formula segment.

### Python Code

``` python
def triangular(x,a,b,c):
    if x<=a or x>=c: return 0
    elif x==b: return 1
    elif x<b: return (x-a)/(b-a)
    else: return (c-x)/(c-b)
x=float(input("Enter x: "))
a,b,c=map(float,input("Enter a b c: ").split())
print("Membership:",triangular(x,a,b,c))
```

**Expected output / viva note:** Example: x=5, a=0, b=5, c=10 gives
membership 1.

------------------------------------------------------------------------

## 11. Delta Learning Rule

**Aim:** To implement 11. Delta Learning Rule.

**Formula / principle:** Output=sum(w\[i\]*x\[i\])+b;
error=target-output. Update: w\[i\]+=learning_rate*error*x\[i\];
b+=learning_rate*error.

**Algorithm:** Calculate linear output; calculate error; update weights
and bias; repeat for all training examples.

### Python Code

``` python
X=[[0,0],[0,1],[1,0],[1,1]]; Y=[0,0,0,1]
w=[0.0,0.0]; b=0.0; lr=0.1
for epoch in range(1000):
    for x,target in zip(X,Y):
        output=w[0]*x[0]+w[1]*x[1]+b
        error=target-output
        w[0]+=lr*error*x[0]; w[1]+=lr*error*x[1]; b+=lr*error
print("Weights:",w); print("Bias:",round(b,3))
```

**Expected output / viva note:** This is the linear Delta Rule. Its
continuous outputs approximate targets; they are not automatically
binary classifications.

------------------------------------------------------------------------

## 12. Fuzzy Logic Control System

**Aim:** To implement 12. Fuzzy Logic Control System.

**Formula / principle:** Weighted output = sum(membership × rule output)
/ sum(membership).

**Algorithm:** Calculate low, medium and high temperature memberships;
apply fan-speed rules of 25%, 60%, and 100%; combine by weighted
average.

### Python Code

``` python
def tri(x,a,b,c):
    if x<=a or x>=c: return 0
    if x==b: return 1
    if x<b: return (x-a)/(b-a)
    return (c-x)/(c-b)
temp=float(input("Enter temperature: "))
low=tri(temp,0,15,30); medium=tri(temp,20,30,40); high=tri(temp,30,40,50)
total=low+medium+high
speed=(low*25+medium*60+high*100)/total if total else 0
print("Fan speed:",round(speed,2),"%")
```

**Expected output / viva note:** A simple fuzzy controller implemented
using formulas and weighted-average output.

------------------------------------------------------------------------

## 13. Resolution Using First-Order Logic

**Aim:** To implement 13. Resolution Using First-Order Logic.

**Formula / principle:** Clauses: NOT Human(x) OR Mortal(x);
Human(Socrates); negated goal NOT Mortal(Socrates).

**Algorithm:** Substitute Socrates, derive Mortal(Socrates), then
resolve it with its negation to obtain the empty clause.

### Python Code

``` python
print("Clause 1: NOT Human(x) OR Mortal(x)")
print("Clause 2: Human(Socrates)")
print("Resolvent: Mortal(Socrates)")
print("Negated goal: NOT Mortal(Socrates)")
print("Contradiction: empty clause")
print("Therefore, Socrates is mortal.")
```

**Expected output / viva note:** This shows resolution proof steps
manually; it is not a general automated theorem prover.

------------------------------------------------------------------------

## 14. Min-Max Tree

**Aim:** To implement 14. Min-Max Tree.

**Formula / principle:** MAX returns the maximum child value; MIN
returns the minimum child value.

**Algorithm:** Recursively evaluate leaf values, alternate MAX and MIN
levels, and return the root value.

### Python Code

``` python
def minimax(depth,index,ismax,values,height):
    if depth==height: return values[index]
    left=minimax(depth+1,2*index,not ismax,values,height)
    right=minimax(depth+1,2*index+1,not ismax,values,height)
    return max(left,right) if ismax else min(left,right)
print("Optimal value:",minimax(0,0,True,[3,5,6,9],2))
```

**Expected output / viva note:** Output: Optimal value 6.

------------------------------------------------------------------------

## 15. Hopfield Network

**Aim:** To implement 15. Hopfield Network.

**Formula / principle:** Weight:
w\[i\]\[j\]=(1/N)*sum(pattern\[i\]*pattern\[j\]) for i≠j; set diagonal
to zero. Update neuron state by sign of weighted sum.

**Algorithm:** Store a bipolar pattern in weights; give a noisy pattern;
update neurons using the weighted sum.

### Python Code

``` python
pattern=[1,-1,1,-1]; n=len(pattern)
W=[[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i!=j: W[i][j]=pattern[i]*pattern[j]/n
x=[1,-1,-1,-1]
for i in range(n):
    value=sum(W[i][j]*x[j] for j in range(n))
    x[i]=1 if value>=0 else -1
print("Recalled pattern:",x)
```

**Expected output / viva note:** Simple one-pattern demonstration;
recall depends on stored patterns and update dynamics.

------------------------------------------------------------------------

## 16. Water Jug Problem using BFS

**Aim:** To implement 16. Water Jug Problem using BFS.

**Formula / principle:** State=(amount in jug 1, amount in jug 2). BFS
uses a FIFO queue to explore states by number of steps.

**Algorithm:** Start at (0,0); generate fill, empty and pour states;
avoid visited states; stop when either jug reaches the target.

### Python Code

``` python
def water_jug(x, y, target):
    queue = [((0, 0), [(0, 0)])]
    visited = {(0, 0)}
    while queue:
        (a, b), path = queue.pop(0)
        if a == target or b == target:
            print("Solution:", path)
            return
        p = min(a, y - b)
        r = min(b, x - a)
        states = [
            (x, b), (a, y), (0, b), (a, 0),
            (a-p, b+p), (a+r, b-r)
        ]
        for state in states:
            if state not in visited:
                visited.add(state)
                queue.append((state, path + [state]))
water_jug(4, 3, 2)
```

**Expected output / viva note:** Prints a sequence of states reaching 2
litres.

------------------------------------------------------------------------

## 17. N-Queen using DFS

**Aim:** To implement 17. N-Queen using DFS.

**Formula / principle:** Two queens conflict if they share a column or
if abs(column difference) equals row difference.

**Algorithm:** Place one queen per row; check column and diagonal
safety; recurse; backtrack if a placement fails.

### Python Code

``` python
def solve(n):
    board=[]
    def dfs(row):
        if row==n: return True
        for col in range(n):
            safe=True
            for r in range(row):
                if board[r]==col or abs(board[r]-col)==row-r:
                    safe=False; break
            if safe:
                board.append(col)
                if dfs(row+1): return True
                board.pop()
        return False
    if dfs(0):
        for col in board: print("."*col+"Q"+"."*(n-col-1))
    else: print("No solution")
solve(4)
```

**Expected output / viva note:** Prints one valid solution to the
4-Queen problem.

------------------------------------------------------------------------

## 18. Breadth-First Search (BFS)

**Aim:** To implement 18. Breadth-First Search (BFS).

**Formula / principle:** BFS uses a FIFO queue and a visited set.

**Algorithm:** Enqueue the start node; dequeue and visit it; enqueue
unvisited neighbours; repeat until empty.

### Python Code

``` python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': [], 'F': []
}
queue = ['A']
visited = set()
while queue:
    node = queue.pop(0)
    if node not in visited:
        print(node, end=" ")
        visited.add(node)
        for neighbour in graph[node]:
            if neighbour not in visited:
                queue.append(neighbour)
```

**Expected output / viva note:** Output: A B C D E F.

------------------------------------------------------------------------

19. Depth-First Search (DFS)

Aim

To implement Depth-First Search (DFS).

Formula / Principle

DFS explores a branch deeply before backtracking. A visited set prevents cycles.

Algorithm

1. Visit the node and mark it as visited.
2. Print the node.
3. Recursively visit each unvisited neighbour.
4. Stop when all reachable nodes have been visited.

Program

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

def dfs(node, visited):
    if node in visited:
        return

    print(node, end=' ')
    visited.add(node)

    for nxt in graph[node]:
        dfs(nxt, visited)

dfs('A', set())

Expected Output

A B D E C F

Viva Note

DFS explores one branch completely before backtracking. It uses recursion or a stack and a visited set to avoid revisiting nodes.

---

20. Feedforward ANN and Backpropagation

Aim

To implement a Feedforward Artificial Neural Network (ANN) using Backpropagation.

Formula / Principle

1. Sigmoid Function

[
\sigma(z) = \frac{1}{1+e^{-z}}
]

2. Output Delta

[
\delta_o = (target-output)\times output\times(1-output)
]

3. Hidden Delta

[
\delta_h = h(1-h)\times w\times\delta_o
]

4. Weight Update

[
w_{\text{new}} = w_{\text{old}} + learning_rate\times\delta\times input
]

Algorithm

1. Initialize the weights, biases, and learning rate.
2. Perform forward propagation through the hidden and output layers.
3. Calculate the output error and output delta.
4. Calculate the hidden-layer deltas using the output delta.
5. Update the weights and biases using backpropagation.
6. Repeat the process for the specified number of epochs.
7. Predict the output for each input.

Program

import math
import random

random.seed(1)

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]
lr = 0.5

w1 = [
    [random.uniform(-1, 1) for _ in range(2)]
    for _ in range(2)
]

b1 = [0.0, 0.0]
w2 = [random.uniform(-1, 1) for _ in range(2)]
b2 = 0.0


def sigmoid(z):
    z = max(-40, min(40, z))
    return 1 / (1 + math.exp(-z))


for epoch in range(20000):
    for x, target in zip(X, Y):

        # Forward propagation
        h = [
            sigmoid(
                w1[j][0] * x[0]
                + w1[j][1] * x[1]
                + b1[j]
            )
            for j in range(2)
        ]

        out = sigmoid(
            w2[0] * h[0]
            + w2[1] * h[1]
            + b2
        )

        # Calculate output delta
        d2 = (target - out) * out * (1 - out)

        # Calculate hidden-layer deltas
        d1 = [
            h[j] * (1 - h[j]) * w2[j] * d2
            for j in range(2)
        ]

        # Update weights and biases
        for j in range(2):
            w2[j] += lr * d2 * h[j]

            w1[j][0] += lr * d1[j] * x[0]
            w1[j][1] += lr * d1[j] * x[1]

            b1[j] += lr * d1[j]

        b2 += lr * d2


# Test the trained network
for x in X:
    h = [
        sigmoid(
            w1[j][0] * x[0]
            + w1[j][1] * x[1]
            + b1[j]
        )
        for j in range(2)
    ]

    out = sigmoid(
        w2[0] * h[0]
        + w2[1] * h[1]
        + b2
    )

    print(x, "->", round(out, 2), int(out >= 0.5))

Expected Output

The predicted classes should approximate the XOR pattern:

[0, 0] -> probability 0
[0, 1] -> probability 1
[1, 0] -> probability 1
[1, 1] -> probability 0

Note: The actual program prints numerical probabilities, not the words "probability 0" or "probability 1". The predicted classes should be "0, 1, 1, 0".

Viva Note

A feedforward ANN passes input data through hidden and output layers. Backpropagation calculates errors and updates weights and biases to reduce the error during training.
