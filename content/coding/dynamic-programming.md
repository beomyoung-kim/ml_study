# Dynamic Programming

> [!TIP] Say this first
> Always narrate the DP out loud in four beats: **state → transition → base case → evaluation order.** Interviewers score DP on whether you can *define the recurrence*, not on syntax. If you can say "let `dp[i]` be ... then `dp[i] = ...`," the code writes itself.

DP applies when a problem has **overlapping subproblems** and **optimal substructure**: the answer decomposes into answers of smaller instances, and those recur. The whole game is choosing the state so transitions are local and cheap.

## When to reach for it (and when greedy is enough)

<div class="proscons"><div><div class="pros-t">DP cues</div>

- "Max / min / count of ways / is it possible," over choices with structure.
- A greedy choice has a **counterexample** (order matters, choices interact).
- Overlapping recomputation in the brute-force recursion.
- Classic shapes: knapsack, LIS, edit distance, path counting, interval merging.

</div><div><div class="cons-t">Not DP</div>

- A provable exchange argument makes local-optimal global-optimal → [Greedy](#/coding/greedy-intervals).
- Subproblems don't overlap → plain divide-and-conquer.
- State space is exponential with no compression → search / heuristics instead.

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

> [!NOTE] Interview move
> Derive the recurrence top-down with `@cache` (fast, hard to get wrong), then, if asked for space, convert to tabulation and roll the table down to one or two rows. Showing both directions is a strong signal.

## Practice — implement, run, test

> [!TIP] How to use this section
> Several problems below have a **live Python editor**. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime (~10 MB); later runs are instant. Prefer your own editor? Each problem links out to **LeetCode**.

The labs are grouped by shape below — 1-D rolling-table DPs first, then the 2-D grid family. Work them in order.

## 1-D DP

### House Robber — `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/house-robber/)
Rob `i` (skip `i-1`) or skip `i`. Two rolling scalars suffice.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"rob","starter":"def rob(nums: list[int]) -> int:\n    # rob i (skip i-1) or skip i; two rolling scalars suffice\n    pass","tests":[{"args":[[1,2,3,1]],"expect":4},{"args":[[2,7,9,3,1]],"expect":12},{"args":[[2,1,1,2]],"expect":4},{"args":[[5]],"expect":5},{"args":[[]],"expect":0}],"solution":"def rob(nums: list[int]) -> int:\n    skip, take = 0, 0\n    for x in nums:\n        skip, take = max(skip, take), skip + x\n    return max(skip, take)"}
</script>
</div>

`O(N)` time, `O(1)` space. Circular variant (LC 213): run twice, excluding first or last house.

### Coin Change — unbounded knapsack (fewest coins) <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/coin-change/)
`dp[a]` = min coins to make amount `a`. Iterate amounts outward.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"coin_change","starter":"def coin_change(coins: list[int], amount: int) -> int:\n    # dp[a] = fewest coins to make amount a; iterate amounts outward\n    pass","tests":[{"args":[[1,2,5],11],"expect":3},{"args":[[2],3],"expect":-1},{"args":[[1],0],"expect":0},{"args":[[2,5,10,1],27],"expect":4}],"solution":"def coin_change(coins: list[int], amount: int) -> int:\n    INF = amount + 1\n    dp = [0] + [INF] * amount\n    for a in range(1, amount + 1):\n        for c in coins:\n            if c <= a:\n                dp[a] = min(dp[a], dp[a - c] + 1)\n    return dp[amount] if dp[amount] < INF else -1"}
</script>
</div>

`O(amount · |coins|)`. **Loop order matters:** amount-outer/coin-inner counts *combinations* correctly for "number of ways" (LC 518); swapping to coin-outer avoids counting permutations. Say which you're computing.

### Longest Increasing Subsequence — `O(N log N)` <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/longest-increasing-subsequence/)
Patience sorting: `tails[k]` = smallest possible tail of an increasing subsequence of length `k+1`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"length_of_lis","starter":"import bisect\n\ndef length_of_lis(nums: list[int]) -> int:\n    # patience sorting: tails[k] = smallest tail of an increasing subsequence of length k+1\n    pass","tests":[{"args":[[10,9,2,5,3,7,101,18]],"expect":4},{"args":[[0,1,0,3,2,3]],"expect":4},{"args":[[7,7,7,7,7]],"expect":1},{"args":[[4,10,4,3,8,9]],"expect":3},{"args":[[]],"expect":0}],"solution":"import bisect\n\ndef length_of_lis(nums: list[int]) -> int:\n    tails = []\n    for x in nums:\n        i = bisect.bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)"}
</script>
</div>

The naive DP is `O(N²)` (`dp[i] = 1 + max(dp[j] for j<i if nums[j]<nums[i])`); know both and lead with the fast one.

## 2-D DP

### Edit Distance (Levenshtein) — the canonical grid DP <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/edit-distance/)
`dp[i][j]` = edits to turn `word1[:i]` into `word2[:j]`.

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
<figcaption>HORSE → ROS = 3. Each cell is 1 + min(up, left, diag), or diag if the letters match.</figcaption>
</figure>

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"min_distance","starter":"def min_distance(w1: str, w2: str) -> int:\n    # dp[i][j] = edits to turn w1[:i] into w2[:j]; match is free, else 1 + min(delete, insert, replace)\n    pass","tests":[{"args":["horse","ros"],"expect":3},{"args":["intention","execution"],"expect":5},{"args":["","abc"],"expect":3},{"args":["abc",""],"expect":3},{"args":["abc","abc"],"expect":0}],"solution":"def min_distance(w1: str, w2: str) -> int:\n    m, n = len(w1), len(w2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(m + 1): dp[i][0] = i\n    for j in range(n + 1): dp[0][j] = j\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if w1[i-1] == w2[j-1]:\n                dp[i][j] = dp[i-1][j-1]\n            else:\n                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n    return dp[m][n]"}
</script>
</div>

`O(mn)` time, reducible to `O(min(m,n))` space (two rows). LCS, min-path-sum, and unique-paths share this grid skeleton.

### 0/1 Knapsack — the DP everyone should hold in their head
`dp[w]` = best value at capacity `w`; iterate weights **downward** so each item is used at most once.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"knapsack","starter":"def knapsack(weights, values, cap: int) -> int:\n    # dp[w] = best value at capacity w; iterate capacity downward so each item is used at most once\n    pass","tests":[{"args":[[1,3,4,5],[1,4,5,7],7],"expect":9},{"args":[[2,3,4,5],[3,4,5,6],5],"expect":7},{"args":[[1,2,3],[6,10,12],5],"expect":22},{"args":[[4,5],[1,2],3],"expect":0}],"solution":"def knapsack(weights, values, cap: int) -> int:\n    dp = [0] * (cap + 1)\n    for wt, val in zip(weights, values):\n        for w in range(cap, wt - 1, -1):\n            dp[w] = max(dp[w], dp[w - wt] + val)\n    return dp[cap]"}
</script>
</div>

Forward iteration would reuse items → that's *unbounded* knapsack (coin change). The loop direction is the entire distinction.

### Interval DP — `dp[i][j]` over a subrange, split at `k`
For problems like matrix-chain, burst balloons (LC 312), or optimal BST: `dp[i][j] = min/max over k of dp[i][k] + dp[k][j] + cost`. Fill by **increasing interval length** so subranges are ready.

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

## The framing checklist (use it live)

| Beat | Ask yourself |
| --- | --- |
| **State** | What does `dp[...]` *mean* in one sentence? Fewest dimensions that make transitions local. |
| **Transition** | From which smaller states, and what choice, do I build `dp[i]`? |
| **Base case** | Smallest inputs — empty string, capacity 0, first element. |
| **Order** | Which states must exist before I compute this one? (row/col/length) |
| **Answer** | Which cell holds the result — `dp[n]`, `dp[m][n]`, `max(dp)`? |

## Pitfalls

- **Wrong loop direction** in knapsack (0/1 vs unbounded) — reverse vs forward.
- **Off-by-one on strings:** `dp[i]` usually means "first `i` characters," so it indexes `s[i-1]`.
- **Counting ways vs sequences:** amount-outer counts permutations, item-outer counts combinations.
- **Confusing subarray (contiguous, Kadane) with subsequence** (LIS).
- **Missing the space optimization** when asked — practice the two-row and rolling-scalar reductions.
- **Recursion depth** on top-down over large `N` — convert to tabulation.

## Q&A

<details class="qa"><summary>How do you decide DP vs greedy?</summary>
<div class="qa-body">

**Short:** Try greedy first and look for a counterexample. If a locally optimal choice can be beaten because choices interact, I need DP to consider all combinations; if I can *prove* the greedy choice is safe (exchange argument), greedy wins and it's simpler/faster.

**Deep:** Coin change is the textbook split: greedy (largest coin first) works for canonical currencies but fails for `{1,3,4}` making 6 (greedy 4+1+1=3 coins, optimal 3+3=2). The interaction between choices is the DP signal. When unsure, I state the greedy, give the counterexample, and pivot to DP — that reasoning is itself the signal interviewers want.
</div></details>

<details class="qa"><summary>Memoization or tabulation — does it matter?</summary>
<div class="qa-body">

**Short:** Same complexity; different trade-offs. Top-down is fastest to derive and only computes reachable states; bottom-up avoids recursion limits and unlocks space optimization.

**Deep:** I derive with `@cache` because the recurrence maps 1:1 to code and I can't botch the evaluation order. If the state space is dense and space matters, I convert to a table and roll it to `O(width)`; if it's sparse, memoization's laziness is a real win. The one gotcha: top-down over `N ~ 10^5` can hit Python's recursion limit, so I flag that and switch.
</div></details>

**Follow-ups you should expect**
- "Reduce the space." → two rows, then one, then rolling scalars.
- "Reconstruct the actual sequence, not just the value." → store parent pointers / backtrack the table.
- "LIS faster than `O(N²)`?" → patience sorting `O(N log N)`.
- "What's the state if we add a constraint (k transactions, cooldown)?" → add a dimension; walk the new transition.

## Cheat-sheet

| Fact | Detail |
| --- | --- |
| Applicability | overlapping subproblems + optimal substructure |
| Framing | state → transition → base → order → answer |
| 1-D | Fibonacci/rob (roll scalars), coin change |
| 2-D grid | edit distance, LCS, paths — `O(mn)`, reducible to 2 rows |
| 0/1 knapsack | iterate capacity **downward** |
| Unbounded | iterate capacity **forward** |
| LIS | `O(N log N)` patience sorting |
| Interval DP | fill by increasing length, split at `k` |
| Ways vs combos | item-outer = combinations; amount-outer = permutations |
| Derive then optimize | `@cache` first, tabulate for space |

**Related:** [Greedy & Intervals](#/coding/greedy-intervals) · [Binary Search](#/coding/binary-search) · [Trees & BSTs](#/coding/trees-bst) · back to [The Core Patterns](#/coding/patterns) and [Coding Round Strategy](#/coding/strategy).
