# Dynamic Programming

> [!TIP] 이 말부터 시작하세요
> DP는 항상 네 박자로 소리 내어 서술하세요: **state → transition → base case → evaluation order.** 면접관은 문법이 아니라 *recurrence를 정의*할 수 있는지로 DP를 평가합니다. "`dp[i]`를 ...라고 하면 `dp[i] = ...`"라고 말할 수 있으면 코드는 저절로 써집니다.

DP는 문제가 **overlapping subproblem**과 **optimal substructure**를 가질 때 적용됩니다: 답이 더 작은 인스턴스의 답으로 분해되고, 그것들이 반복되는 경우입니다. 핵심은 transition이 국소적이고 값싸도록 state를 고르는 것입니다.

## 언제 꺼내 쓰나 (그리고 언제 greedy로 충분한가)

<div class="proscons"><div><div class="pros-t">DP 신호</div>

- "max / min / 경우의 수 / 가능한가," 구조가 있는 선택들에 대해.
- greedy 선택에 **반례**가 있음 (순서가 중요하고, 선택들이 상호작용).
- brute-force recursion에서 겹치는 재계산.
- 정석적인 형태: knapsack, LIS, edit distance, 경로 세기, interval merging.

</div><div><div class="cons-t">DP 아님</div>

- 증명 가능한 exchange argument가 local-optimal을 global-optimal로 만듦 → [Greedy](#/coding/greedy-intervals).
- Subproblem이 겹치지 않음 → 평범한 divide-and-conquer.
- state 공간이 지수적이고 압축이 안 됨 → 대신 search / heuristic.

</div></div>

## Memoization vs tabulation

```python
from functools import cache

# Top-down: write the recurrence, add @cache. Fastest to derive; risks
# recursion-depth limits and skips unreachable states (a plus).
@cache
def climb(n: int) -> int:
    if n <= 2:
        return n
    return climb(n - 1) + climb(n - 2)

# Bottom-up: same recurrence, explicit table + evaluation order.
# No recursion limit; enables O(1)-row space optimization.
def climb_tab(n: int) -> int:
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
```

> [!NOTE] 면접 무브
> recurrence를 `@cache`로 top-down으로 도출하고(빠르고 틀리기 어려움), 공간을 물어보면 tabulation으로 변환해 테이블을 한두 행으로 굴려 내립니다. 두 방향을 모두 보여주는 것이 강한 신호입니다.

## 1-D DP

### House Robber — `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
`i`를 털거나(`i-1`은 건너뜀) `i`를 건너뜁니다. rolling scalar 두 개면 충분합니다.

```python
def rob(nums: list[int]) -> int:
    skip, take = 0, 0            # best excluding / including previous house
    for x in nums:
        skip, take = max(skip, take), skip + x
    return max(skip, take)
```
`O(N)` 시간, `O(1)` 공간. 원형 변형(LC 213): 첫 집 혹은 마지막 집을 제외하고 두 번 실행.

### Coin Change — unbounded knapsack (최소 동전 수)
`dp[a]` = 금액 `a`를 만드는 최소 동전 수. 금액을 바깥으로 반복합니다.

```python
def coin_change(coins: list[int], amount: int) -> int:
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] < INF else -1
```
`O(amount · |coins|)`. **루프 순서가 중요합니다:** amount-바깥/coin-안쪽은 "경우의 수"(LC 518)에 대해 *조합*을 올바르게 셉니다; coin-바깥으로 바꾸면 순열을 세는 것을 피합니다. 무엇을 계산하는지 말하세요.

### Longest Increasing Subsequence — `O(N log N)`
Patience sorting: `tails[k]` = 길이 `k+1`인 증가 subsequence의 가능한 가장 작은 tail.

```python
import bisect

def length_of_lis(nums: list[int]) -> int:
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)     # bisect_right → longest non-decreasing
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```
naive DP는 `O(N²)`입니다 (`dp[i] = 1 + max(dp[j] for j<i if nums[j]<nums[i])`); 둘 다 알되 빠른 것을 먼저 제시하세요.

## 2-D DP

### Edit Distance (Levenshtein) — 정석적인 grid DP
`dp[i][j]` = `word1[:i]`을 `word2[:j]`로 바꾸는 데 필요한 편집 수.

<figure>
<svg viewBox="0 0 360 190" width="100%" role="img" aria-label="Edit distance DP table for HORSE to ROS">
  <g font-family="ui-monospace, monospace" font-size="13" text-anchor="middle">
    <text x="30" y="20">""</text><text x="80" y="20">R</text><text x="130" y="20">O</text><text x="180" y="20">S</text>
    <text x="18" y="55">""</text><text x="18" y="90">H</text><text x="18" y="125">O</text><text x="18" y="160">R</text>
    <g fill="none" stroke="#6366f1" opacity="0.5">
      <rect x="55" y="40" width="50" height="25"/><rect x="105" y="40" width="50" height="25"/><rect x="155" y="40" width="50" height="25"/>
      <rect x="55" y="75" width="50" height="25"/><rect x="105" y="75" width="50" height="25"/><rect x="155" y="75" width="50" height="25"/>
      <rect x="55" y="110" width="50" height="25"/><rect x="105" y="110" width="50" height="25"/><rect x="155" y="110" width="50" height="25"/>
      <rect x="55" y="145" width="50" height="25"/><rect x="105" y="145" width="50" height="25"/><rect x="155" y="145" width="50" height="25"/>
    </g>
    <text x="30" y="58">0</text><text x="80" y="58">1</text><text x="130" y="58">2</text><text x="180" y="58">3</text>
    <text x="30" y="93">1</text><text x="80" y="93">1</text><text x="130" y="93">2</text><text x="180" y="93">3</text>
    <text x="30" y="128">2</text><text x="80" y="128">2</text><text x="130" y="128" fill="#12a150">1</text><text x="180" y="128">2</text>
    <text x="30" y="163">3</text><text x="80" y="163" fill="#12a150">2</text><text x="130" y="163">2</text><text x="180" y="163" fill="#e0533f">2</text>
    <text x="300" y="128" font-size="11" fill="#12a150">match O</text>
    <text x="300" y="163" font-size="11" fill="#e0533f">answer</text>
  </g>
</svg>
<figcaption>HORSE → ROS = 3. 각 셀은 1 + min(up, left, diag), 글자가 일치하면 diag.</figcaption>
</figure>

```python
def min_distance(w1: str, w2: str) -> int:
    m, n = len(w1), len(w2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i           # delete all
    for j in range(n + 1): dp[0][j] = j           # insert all
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if w1[i-1] == w2[j-1]:
                dp[i][j] = dp[i-1][j-1]            # free match
            else:
                dp[i][j] = 1 + min(dp[i-1][j],     # delete
                                   dp[i][j-1],     # insert
                                   dp[i-1][j-1])   # replace
    return dp[m][n]
```
`O(mn)` 시간, `O(min(m,n))` 공간으로 축소 가능(두 행). LCS, min-path-sum, unique-paths가 이 grid 뼈대를 공유합니다.

### 0/1 Knapsack — 누구나 머릿속에 담고 있어야 할 DP
`dp[w]` = 용량 `w`에서의 최선 가치; 각 아이템이 최대 한 번만 쓰이도록 weight를 **아래로** 반복합니다.

```python
def knapsack(weights, values, cap: int) -> int:
    dp = [0] * (cap + 1)
    for wt, val in zip(weights, values):
        for w in range(cap, wt - 1, -1):     # reverse ⇒ 0/1 (each item once)
            dp[w] = max(dp[w], dp[w - wt] + val)
    return dp[cap]
```
정방향 반복은 아이템을 재사용합니다 → 그건 *unbounded* knapsack(coin change)입니다. 루프 방향이 유일한 차이입니다.

### Interval DP — `dp[i][j]`를 subrange에 대해, `k`에서 분할
matrix-chain, burst balloons(LC 312), optimal BST 같은 문제에서: `dp[i][j] = min/max over k of dp[i][k] + dp[k][j] + cost`. subrange가 준비되도록 **interval 길이를 늘려가며** 채웁니다.

```python
def interval_dp(n, cost):
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):            # length-first evaluation order
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = min(dp[i][k] + dp[k+1][j] + cost(i, k, j)
                           for k in range(i, j))
    return dp[0][n-1]
```

## 프레이밍 체크리스트 (실전에서 사용)

| 박자 | 스스로에게 질문 |
| --- | --- |
| **State** | `dp[...]`이 한 문장으로 무엇을 *의미*하나? transition을 국소적으로 만드는 최소 차원. |
| **Transition** | 어떤 더 작은 state들로부터, 어떤 선택으로 `dp[i]`를 만드나? |
| **Base case** | 가장 작은 입력 — 빈 문자열, 용량 0, 첫 원소. |
| **Order** | 이걸 계산하기 전에 어떤 state들이 존재해야 하나? (행/열/길이) |
| **Answer** | 어떤 셀이 결과를 담나 — `dp[n]`, `dp[m][n]`, `max(dp)`? |

## 함정

- knapsack에서 **잘못된 루프 방향** (0/1 vs unbounded) — 역방향 vs 정방향.
- **문자열에서 off-by-one:** `dp[i]`는 보통 "첫 `i`개 문자"를 뜻하므로 `s[i-1]`을 인덱싱합니다.
- **경우의 수 vs 순서:** amount-바깥은 순열을, item-바깥은 조합을 셉니다.
- **subarray(연속, Kadane)와 subsequence(LIS) 혼동.**
- 물어봤을 때 **공간 최적화를 놓침** — 두 행과 rolling-scalar 축소를 연습하세요.
- 큰 `N`에 대한 top-down의 **recursion 깊이** — tabulation으로 변환하세요.

## Q&A

<details class="qa"><summary>DP vs greedy를 어떻게 결정하나요?</summary>
<div class="qa-body">

**짧게:** 먼저 greedy를 시도하고 반례를 찾습니다. 선택들이 상호작용해서 국소적으로 최적인 선택이 뒤집힐 수 있으면, 모든 조합을 고려하기 위해 DP가 필요합니다; greedy 선택이 안전함을 *증명*할 수 있으면(exchange argument) greedy가 이기고 더 간단/빠릅니다.

**깊게:** Coin change가 교과서적 분기점입니다: greedy(가장 큰 동전 먼저)는 정규 화폐에서는 되지만 `{1,3,4}`로 6을 만들 때 실패합니다(greedy 4+1+1=3개, 최적 3+3=2개). 선택 간 상호작용이 DP의 신호입니다. 확실치 않을 때 저는 greedy를 말하고, 반례를 제시하고, DP로 전환합니다 — 그 추론 자체가 면접관이 원하는 신호입니다.
</div></details>

<details class="qa"><summary>Memoization이냐 tabulation이냐 — 중요한가요?</summary>
<div class="qa-body">

**짧게:** 같은 복잡도; 다른 trade-off. Top-down은 도출이 가장 빠르고 도달 가능한 state만 계산합니다; bottom-up은 recursion 한계를 피하고 공간 최적화를 가능케 합니다.

**깊게:** 저는 `@cache`로 도출하는데, recurrence가 코드로 1:1 매핑되고 evaluation order를 망칠 수 없기 때문입니다. state 공간이 밀집이고 공간이 중요하면 테이블로 변환해 `O(width)`로 굴립니다; sparse하면 memoization의 lazy함이 실질적 이득입니다. 한 가지 함정: `N ~ 10^5`에 대한 top-down은 Python recursion 한계에 부딪힐 수 있어서, 그걸 짚고 전환합니다.
</div></details>

**예상되는 후속 질문**
- "공간을 줄이세요." → 두 행, 그다음 한 행, 그다음 rolling scalar.
- "값만이 아니라 실제 sequence를 복원하세요." → parent 포인터 저장 / 테이블 backtrack.
- "LIS를 `O(N²)`보다 빠르게?" → patience sorting `O(N log N)`.
- "제약을 더하면(k transactions, cooldown) state는?" → 차원을 추가; 새 transition을 짚어가세요.

## Cheat-sheet

| 사실 | 세부 |
| --- | --- |
| 적용성 | overlapping subproblem + optimal substructure |
| 프레이밍 | state → transition → base → order → answer |
| 1-D | Fibonacci/rob (scalar 굴리기), coin change |
| 2-D grid | edit distance, LCS, paths — `O(mn)`, 2행으로 축소 가능 |
| 0/1 knapsack | 용량을 **아래로** 반복 |
| Unbounded | 용량을 **정방향으로** 반복 |
| LIS | `O(N log N)` patience sorting |
| Interval DP | 길이를 늘려가며 채우고, `k`에서 분할 |
| 경우의 수 vs 조합 | item-바깥 = 조합; amount-바깥 = 순열 |
| 먼저 도출 후 최적화 | `@cache` 먼저, 공간 위해 tabulate |

**관련:** [Greedy & Intervals](#/coding/greedy-intervals) · [Binary Search](#/coding/binary-search) · [Trees & BSTs](#/coding/trees-bst) · 다시 [The Core Patterns](#/coding/patterns)와 [Coding Round Strategy](#/coding/strategy)로.
