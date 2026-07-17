# Union-Find (Disjoint Set Union)

> [!TIP] Say this first
> "When the problem is about **dynamic connectivity** — edges arrive and I keep asking 'are these two in the same group?' — Union-Find answers both `union` and `connected` in near-`O(1)` amortized." With path compression *and* union by rank, operations are `O(α(N))`, where `α` (inverse Ackermann) is ≤ 4 for any conceivable input.

DSU maintains a forest where each tree is one set. Two optimizations make it effectively constant time: **path compression** flattens trees during `find`, **union by rank/size** keeps them shallow. It's the right tool when DFS/BFS would need re-running as edges are added.

## When to reach for it

<div class="proscons"><div><div class="pros-t">Cues</div>

- "Connected ⇒ same group," count of connected components.
- **Cycle detection in an undirected graph** (union fails ⇒ edge closes a cycle).
- Edges arrive **online**; connectivity queries interleave with additions.
- Kruskal's MST (add cheapest edge if it joins two components).
- Grouping by equivalence (accounts by shared email, equations by variable).

</div><div><div class="cons-t">Prefer other tools</div>

- Static graph, one-shot component count → DFS/BFS is just as good.
- Need shortest paths or the actual edges → [Graphs](#/coding/graphs-bfs-dfs).
- Edges get **removed** → DSU can't un-union; use offline tricks or link-cut trees.

</div></div>

## The template — memorize this exact class

```python
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n            # upper bound on tree height
        self.count = n                 # number of disjoint sets

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False               # already connected ⇒ this edge is redundant
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra            # attach the shorter tree under the taller
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)
```

<figure>
<svg viewBox="0 0 480 130" width="100%" role="img" aria-label="Path compression flattens the tree on find">
  <g font-family="ui-monospace, monospace" font-size="13" text-anchor="middle">
    <text x="110" y="16">before find(3)</text>
    <circle cx="110" cy="35" r="13" fill="#6366f122" stroke="#6366f1"/><text x="110" y="40">0</text>
    <circle cx="110" cy="70" r="13" fill="#6366f122" stroke="#6366f1"/><text x="110" y="75">1</text>
    <circle cx="110" cy="105" r="13" fill="#6366f122" stroke="#6366f1"/><text x="110" y="110">3</text>
    <line x1="110" y1="48" x2="110" y2="57" stroke="#6366f1"/>
    <line x1="110" y1="83" x2="110" y2="92" stroke="#6366f1"/>
    <text x="360" y="16">after: children hang off root</text>
    <circle cx="360" cy="45" r="13" fill="#12a15022" stroke="#12a150"/><text x="360" y="50">0</text>
    <circle cx="320" cy="100" r="13" fill="#12a15022" stroke="#12a150"/><text x="320" y="105">1</text>
    <circle cx="400" cy="100" r="13" fill="#12a15022" stroke="#12a150"/><text x="400" y="105">3</text>
    <line x1="350" y1="55" x2="326" y2="88" stroke="#12a150"/>
    <line x1="370" y1="55" x2="394" y2="88" stroke="#12a150"/>
  </g>
</svg>
<figcaption>Path compression during find rewires nodes directly to the root, so later finds are O(1).</figcaption>
</figure>

## Representative problems

### 1. Number of Connected Components (Medium)
Union every edge; the answer is `count`. Isolated nodes are handled by initializing `count = n`.

```python
def count_components(n: int, edges: list[list[int]]) -> int:
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.count
```
`O(N + E·α(N))`.

### 2. Redundant Connection (Medium) — undirected cycle
A tree plus one extra edge. The first edge whose endpoints are *already* connected is the culprit.

```python
def find_redundant(edges: list[list[int]]) -> list[int]:
    uf = UnionFind(len(edges) + 1)     # nodes are 1-indexed
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]              # union returns False ⇒ cycle-closing edge
    return []
```
`O(N·α(N))`. `union` returning `False` *is* the cycle detector — cleaner than DFS for undirected graphs.

### 3. Accounts Merge (Medium) — union by shared key
Two accounts belong to the same person if they share any email. Union account indices through a `email → first index seen` map.

```python
from collections import defaultdict

def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    uf = UnionFind(len(accounts))
    owner = {}                                    # email -> account index
    for i, acc in enumerate(accounts):
        for email in acc[1:]:
            if email in owner:
                uf.union(i, owner[email])
            else:
                owner[email] = i
    groups = defaultdict(list)
    for email, i in owner.items():
        groups[uf.find(i)].append(email)
    return [[accounts[r][0]] + sorted(emails) for r, emails in groups.items()]
```
`O(N·K log K)` dominated by sorting emails. Note: the *name* never drives the merge — only shared emails do.

### 4. Graph Valid Tree (Medium)
A tree has exactly `n-1` edges, is fully connected, and is acyclic. Union-Find checks all three cheaply.

```python
def valid_tree(n: int, edges: list[list[int]]) -> bool:
    if len(edges) != n - 1:
        return False                    # too few/many edges ⇒ can't be a tree
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return False                # cycle
    return uf.count == 1                 # fully connected
```
With the `n-1` edge check, "no cycle" already implies "connected," but stating all three conditions is the clean answer.

### 5. Kruskal's MST (Hard) — the flagship application
Sort edges by weight; add an edge iff it joins two different components. Union-Find makes the "would this create a cycle?" test `O(α)`.

```python
def kruskal_mst(n: int, edges: list[tuple[int, int, int]]) -> int:
    uf = UnionFind(n)
    total = 0
    for w, u, v in sorted(edges):          # cheapest first
        if uf.union(u, v):                 # joins two components ⇒ take it
            total += w
            if uf.count == 1:
                break                       # spanning tree complete
    return total
```
`O(E log E)` from the sort; the DSU work is effectively linear. Compare to Prim's (heap-based) — Kruskal wins on sparse graphs and edge-list inputs.

## Variations to name

- **Union by size** instead of rank — track subtree sizes (useful when the problem asks for the largest component).
- **Weighted / relational DSU** — store an offset to the parent for "equations divide" (LC 399) or parity problems.
- **DSU on a grid** — map `(r,c) → r*cols + c`; used in *Number of Islands II* (online island count) and *Swim in Rising Water* (activate cells by height).
- **Rollback DSU** — union by rank *without* path compression supports undo, enabling offline dynamic connectivity.

## Pitfalls

- **Off-by-one indexing:** 1-indexed nodes need size `n+1`.
- **Skipping an optimization:** path compression *or* union by rank alone is `O(log N)`; both give `O(α)`. Interviewers probe this.
- **Forgetting isolated nodes** in the component count — initialize `count = n`, not 0.
- **Trying to delete edges** — plain DSU can't; recognize and switch approaches.
- **Comparing `find(a) == find(b)` without compression** on adversarial inputs → degenerate chains.

## Q&A

<details class="qa"><summary>Why is Union-Find nearly O(1)? What is α(N)?</summary>
<div class="qa-body">

**Short:** With both path compression and union by rank, `m` operations on `n` elements cost `O(m·α(n))`, where `α` is the inverse Ackermann function — it grows so slowly it's ≤ 4 for any `n` that fits in the universe. So amortized per-op is effectively constant.

**Deep:** Union by rank bounds tree height at `O(log n)` on its own; path compression flattens paths so repeated finds amortize away. Neither alone reaches `α` — union by rank alone is `O(log n)`, compression alone is `O(log n)` amortized; the Tarjan bound needs both together. I'd state the ≤4 practical bound rather than derive Ackermann live.
</div></details>

<details class="qa"><summary>Union-Find or DFS/BFS for connectivity?</summary>
<div class="qa-body">

**Short:** DFS/BFS for a static graph queried once — it's `O(V+E)` and gives you paths too. Union-Find when edges arrive online, when you interleave many connectivity queries, or when you specifically need incremental component counts (and for Kruskal's MST).

**Deep:** The dividing line is "does the graph change?" A traversal must re-run from scratch after each edge insertion, so `q` queries cost `O(q(V+E))`; DSU absorbs each edge in `O(α)`. The catch is DSU only supports *adding* edges — deletions need rollback DSU or a link-cut tree, which is when I'd reconsider.
</div></details>

**Follow-ups you should expect**
- "Which optimization matters more?" → both together for `α`; either alone is `O(log N)`.
- "Now edges get removed." → plain DSU can't; offline + rollback, or advanced structures.
- "Largest component size?" → union by size, track a `size[]` array.
- "Directed graph cycles?" → DSU is for *undirected*; use DFS 3-coloring instead.

## Cheat-sheet

| Fact | Detail |
| --- | --- |
| Purpose | dynamic connectivity: `union`, `find`, `connected` |
| Complexity | `O(α(N))` amortized with both optimizations (≤4 in practice) |
| Path compression | flatten path to root during `find` |
| Union by rank/size | attach shorter/smaller tree under the other |
| Component count | init `count = n`, decrement on successful union |
| Undirected cycle | `union` returns `False` ⇒ edge closes a cycle |
| Valid tree | `n-1` edges + no cycle + `count == 1` |
| Kruskal MST | sort edges, `union` if it joins two components |
| Cannot | delete edges (needs rollback DSU), directed cycles |

**Related:** [Graphs (BFS/DFS)](#/coding/graphs-bfs-dfs) · [Greedy & Intervals](#/coding/greedy-intervals) · [Binary Search](#/coding/binary-search) · back to [The Core Patterns](#/coding/patterns) and [Coding Round Strategy](#/coding/strategy).
