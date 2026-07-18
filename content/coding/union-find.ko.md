# Union-Find (Disjoint Set Union)

> [!TIP] 이 말부터 시작하세요
> "문제가 **동적 연결성(dynamic connectivity)**에 관한 것일 때 — edge가 도착하고 '이 둘이 같은 그룹인가?'를 계속 묻는 상황 — Union-Find는 `union`과 `connected`를 둘 다 amortized로 거의 `O(1)`에 답합니다." path compression *과* union by rank를 함께 쓰면 연산이 `O(α(N))`인데, `α`(inverse Ackermann)는 상상 가능한 어떤 입력에서도 ≤ 4입니다.

DSU는 각 tree가 하나의 집합인 forest를 유지합니다. 두 가지 최적화가 이를 사실상 상수 시간으로 만듭니다: **path compression**은 `find` 도중 tree를 평탄화하고, **union by rank/size**는 tree를 얕게 유지합니다. edge가 추가될 때마다 DFS/BFS를 재실행해야 하는 상황에서 알맞은 도구입니다.

## 언제 꺼내 쓰나

<div class="proscons"><div><div class="pros-t">신호</div>

- "Connected ⇒ 같은 그룹," 연결 컴포넌트 개수.
- **Undirected graph에서의 사이클 감지** (union 실패 ⇒ edge가 사이클을 닫음).
- edge가 **online**으로 도착; 연결성 query가 추가와 뒤섞임.
- Kruskal's MST (두 컴포넌트를 잇는다면 가장 싼 edge를 추가).
- 동치 관계로 그룹화 (공유 email로 계정, 변수로 방정식).

</div><div><div class="cons-t">다른 도구를 선호</div>

- 정적 graph, 일회성 컴포넌트 개수 → DFS/BFS로 충분.
- 최단 경로나 실제 edge가 필요 → [Graphs](#/coding/graphs-bfs-dfs).
- edge가 **제거**됨 → DSU는 un-union할 수 없음; offline 트릭이나 link-cut tree 사용.

</div></div>

## 템플릿 — 이 클래스를 정확히 외우세요

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
<figcaption>find 도중의 path compression은 노드를 root로 직접 재연결하므로, 이후의 find는 O(1)이 됩니다.</figcaption>
</figure>

## 대표 문제

### 1. Number of Connected Components (Medium)
모든 edge를 union; 답은 `count`입니다. 고립된 노드는 `count = n`으로 초기화하면 처리됩니다.

```python
def count_components(n: int, edges: list[list[int]]) -> int:
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.count
```
`O(N + E·α(N))`.

### 2. Redundant Connection (Medium) — undirected 사이클
tree에 여분의 edge 하나. 양 끝점이 *이미* 연결된 첫 edge가 범인입니다.

```python
def find_redundant(edges: list[list[int]]) -> list[int]:
    uf = UnionFind(len(edges) + 1)     # nodes are 1-indexed
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]              # union returns False ⇒ cycle-closing edge
    return []
```
`O(N·α(N))`. `union`이 `False`를 반환하는 것이 사이클 감지기 *그 자체*입니다 — undirected graph에는 DFS보다 깔끔합니다.

### 3. Accounts Merge (Medium) — 공유 키로 union
두 계정이 email을 하나라도 공유하면 같은 사람의 것입니다. `email → 처음 본 index` map을 통해 계정 index를 union합니다.

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
email 정렬이 지배하는 `O(N·K log K)`. 주의: *이름*은 병합을 결정하지 않습니다 — 오직 공유 email만이 합니다.

### 4. Graph Valid Tree (Medium)
tree는 정확히 `n-1`개의 edge를 갖고, 완전히 연결되어 있으며, 비순환입니다. Union-Find가 이 세 가지를 값싸게 확인합니다.

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
`n-1` edge 체크가 있으면 "사이클 없음"이 이미 "연결됨"을 함의하지만, 세 조건을 모두 언급하는 것이 깔끔한 답입니다.

### 5. Kruskal's MST (Hard) — 대표적인 응용
edge를 가중치로 정렬; 서로 다른 두 컴포넌트를 잇는 경우에만 edge를 추가. Union-Find가 "이게 사이클을 만드나?" 테스트를 `O(α)`로 만듭니다.

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
정렬에서 오는 `O(E log E)`; DSU 작업은 사실상 선형입니다. Prim(heap 기반)과 비교하면 — Kruskal은 sparse graph와 edge-list 입력에서 유리합니다.

## 언급할 변형

- rank 대신 **Union by size** — subtree 크기를 추적 (문제가 가장 큰 컴포넌트를 물을 때 유용).
- **Weighted / relational DSU** — "equations divide"(LC 399)나 parity 문제를 위해 parent로의 offset을 저장.
- **Grid에서의 DSU** — `(r,c) → r*cols + c`로 매핑; *Number of Islands II*(online 섬 개수)와 *Swim in Rising Water*(높이로 셀 활성화)에 사용.
- **Rollback DSU** — path compression *없이* union by rank는 undo를 지원해 offline 동적 연결성을 가능케 함.

## 함정

- **Off-by-one 인덱싱:** 1-indexed 노드는 크기 `n+1`이 필요.
- **최적화를 건너뜀:** path compression *만* 혹은 union by rank *만*은 `O(log N)`; 둘 다면 `O(α)`. 면접관이 이걸 파고듭니다.
- 컴포넌트 개수에서 **고립된 노드를 잊음** — `count`를 0이 아니라 `n`으로 초기화.
- **edge를 지우려 시도** — 평범한 DSU는 못 함; 알아채고 접근을 바꾸세요.
- adversarial 입력에서 **compression 없이 `find(a) == find(b)` 비교** → 퇴화된 chain.

## Q&A

<details class="qa"><summary>Union-Find는 왜 거의 O(1)인가요? α(N)이 뭔가요?</summary>
<div class="qa-body">

**짧게:** path compression과 union by rank를 함께 쓰면, `n`개 원소에 대한 `m`개 연산이 `O(m·α(n))`이 듭니다. 여기서 `α`는 inverse Ackermann 함수인데, 너무 느리게 자라서 우주에 들어갈 만한 어떤 `n`에서도 ≤ 4입니다. 그래서 연산당 amortized는 사실상 상수입니다.

**깊게:** Union by rank는 그 자체로 tree 높이를 `O(log n)`으로 제한합니다; path compression은 path를 평탄화해서 반복된 find가 amortize됩니다. 어느 하나만으로는 `α`에 도달하지 못합니다 — union by rank만은 `O(log n)`, compression만은 amortized `O(log n)`; Tarjan 경계는 둘을 함께 필요로 합니다. Ackermann을 실전에서 유도하기보다 ≤4의 실용적 경계를 말하겠습니다.
</div></details>

<details class="qa"><summary>연결성에 Union-Find냐 DFS/BFS냐?</summary>
<div class="qa-body">

**짧게:** 한 번만 query하는 정적 graph에는 DFS/BFS — `O(V+E)`이고 경로도 줍니다. edge가 online으로 도착하거나, 많은 연결성 query를 뒤섞거나, 특히 점진적 컴포넌트 개수가 필요할 때(그리고 Kruskal's MST) Union-Find.

**깊게:** 분기점은 "graph가 변하는가?"입니다. traversal은 각 edge 삽입 후 처음부터 재실행해야 하므로 `q`개 query가 `O(q(V+E))`; DSU는 각 edge를 `O(α)`에 흡수합니다. 함정은 DSU가 *추가*만 지원한다는 것 — 삭제는 rollback DSU나 link-cut tree가 필요하고, 그때 재고합니다.
</div></details>

**예상되는 후속 질문**
- "어떤 최적화가 더 중요한가요?" → `α`에는 둘 다; 어느 하나만은 `O(log N)`.
- "이제 edge가 제거됩니다." → 평범한 DSU는 못 함; offline + rollback, 혹은 고급 자료구조.
- "가장 큰 컴포넌트 크기는?" → union by size, `size[]` 배열 추적.
- "Directed graph 사이클은?" → DSU는 *undirected*용; DFS 3-coloring을 쓰세요.

## Cheat-sheet

| 사실 | 세부 |
| --- | --- |
| 목적 | 동적 연결성: `union`, `find`, `connected` |
| 복잡도 | 두 최적화 모두 쓸 때 amortized `O(α(N))` (실전 ≤4) |
| Path compression | `find` 도중 path를 root로 평탄화 |
| Union by rank/size | 더 짧은/작은 tree를 다른 쪽 밑에 붙임 |
| 컴포넌트 개수 | `count = n`으로 초기화, 성공한 union마다 감소 |
| Undirected 사이클 | `union`이 `False` 반환 ⇒ edge가 사이클을 닫음 |
| Valid tree | `n-1` edge + 사이클 없음 + `count == 1` |
| Kruskal MST | edge 정렬, 두 컴포넌트를 잇는다면 `union` |
| 불가능 | edge 삭제 (rollback DSU 필요), directed 사이클 |

**관련:** [Graphs (BFS/DFS)](#/coding/graphs-bfs-dfs) · [Greedy & Intervals](#/coding/greedy-intervals) · [Binary Search](#/coding/binary-search) · 다시 [The Core Patterns](#/coding/patterns)와 [Coding Round Strategy](#/coding/strategy)로.
