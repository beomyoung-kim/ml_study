# Binary Search

> [!TIP] Say this first
> "Binary search isn't about sorted arrays — it's about a **monotonic predicate**. If I can define a boolean `feasible(x)` that flips from false to true exactly once over the search space, I can find the boundary in `O(log N)`." That one sentence signals you know the general pattern, not just LeetCode 704.

Halve the search space each step. The array version is trivial; the interview value is recognizing the *hidden* search space — **binary search on the answer** (a.k.a. parametric search) — and getting the boundary conditions exactly right without an infinite loop.

## When to reach for it

<div class="proscons"><div><div class="pros-t">Cues</div>

- Input is **sorted** (or rotated-sorted) → search / insert position / bounds.
- "**Minimum X such that** ..." or "maximum X such that ...", where checking a fixed `X` is easy but searching is not (capacity, speed, distance, time).
- The predicate is **monotonic**: once feasible, always feasible for larger (or smaller) `X`.
- You need `O(log N)` and a linear scan is too slow.

</div><div><div class="cons-t">Not this</div>

- Unsorted and no monotone predicate → sort first (costs the `log`) or use a hash/heap.
- The "answer check" is itself expensive *and* non-monotonic → binary search doesn't apply.

</div></div>

<figure>
<svg viewBox="0 0 520 96" width="100%" role="img" aria-label="Monotonic predicate flips once from false to true">
  <g font-family="ui-monospace, monospace" font-size="13">
    <rect x="10"  y="30" width="55" height="34" fill="#e0533f22" stroke="#e0533f"/>
    <rect x="65"  y="30" width="55" height="34" fill="#e0533f22" stroke="#e0533f"/>
    <rect x="120" y="30" width="55" height="34" fill="#e0533f22" stroke="#e0533f"/>
    <rect x="175" y="30" width="55" height="34" fill="#12a15022" stroke="#12a150"/>
    <rect x="230" y="30" width="55" height="34" fill="#12a15022" stroke="#12a150"/>
    <rect x="285" y="30" width="55" height="34" fill="#12a15022" stroke="#12a150"/>
    <text x="37"  y="52" text-anchor="middle" fill="#e0533f">F</text>
    <text x="92"  y="52" text-anchor="middle" fill="#e0533f">F</text>
    <text x="147" y="52" text-anchor="middle" fill="#e0533f">F</text>
    <text x="202" y="52" text-anchor="middle" fill="#12a150">T</text>
    <text x="257" y="52" text-anchor="middle" fill="#12a150">T</text>
    <text x="312" y="52" text-anchor="middle" fill="#12a150">T</text>
    <line x1="175" y1="18" x2="175" y2="76" stroke="#6366f1" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="360" y="52" fill="#6366f1">← first True = answer</text>
  </g>
</svg>
<figcaption>Every binary-search-on-the-answer problem reduces to finding this boundary.</figcaption>
</figure>

## The two templates worth memorizing

```python
# 1. Exact match (classic). Loop while lo <= hi, exclusive shrink.
def search(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2        # overflow-safe habit
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# 2. Boundary / lower_bound (the workhorse). Loop while lo < hi,
#    hi never below the candidate. Returns the FIRST index where pred is True.
def lower_bound(lo: int, hi: int, pred) -> int:
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if pred(mid):
            hi = mid          # keep mid: it might be the answer
        else:
            lo = mid + 1      # discard mid: definitely not the answer
    return lo                 # == hi; smallest x with pred(x) True
```

> [!WARNING] The only two bugs
> **Infinite loop:** `while lo < hi` with `lo = mid` (not `mid + 1`) never advances when `hi = lo + 1`. Rule: whichever side *keeps* `mid` must be the one that shrinks toward it (`hi = mid`); the other adds one (`lo = mid + 1`). **Off-by-one boundary:** decide `<` vs `<=` in the predicate to pick lower vs upper bound, and test the two-element case by hand.

## Representative problems

### 1. Search Insert Position — lower bound (Easy)
Return the index where `target` belongs. This *is* `lower_bound` with `pred = nums[mid] >= target`.

```python
def search_insert(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums)                 # hi = len, not len-1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
```
`O(log N)` time, `O(1)` space. Switching `<` to `<=` gives the **upper bound** — that pair covers `bisect_left`/`bisect_right`.

### 2. Search in Rotated Sorted Array (Medium)
No duplicates. At each step, one half is sorted; decide whether `target` lies inside it.

```python
def search_rotated(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                 # left half sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                      # right half sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```
`O(log N)`. With duplicates (LC 81) the `nums[lo] == nums[mid]` tie forces a linear `lo += 1` step, degrading to `O(N)` worst case — say this out loud.

### 3. Koko Eating Bananas — binary search on the answer (Medium)
Minimum eating speed `k` to finish `piles` within `h` hours. The predicate "can finish at speed `k`" is monotonic in `k`.

```python
import math

def min_eating_speed(piles: list[int], h: int) -> int:
    def can_finish(speed: int) -> bool:
        return sum(math.ceil(p / speed) for p in piles) <= h

    lo, hi = 1, max(piles)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if can_finish(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```
`O(N log M)` where `M = max(piles)`. Same skeleton solves *Capacity to Ship Packages* (LC 1011), *Split Array Largest Sum* (LC 410), *Minimize Max Distance* — only `can_finish` changes.

### 4. Find Peak Element (Medium)
Any index larger than both neighbors, in `O(log N)`. Walk uphill: the larger neighbor's side always contains a peak (treat out-of-bounds as `-∞`).

```python
def find_peak_element(nums: list[int]) -> int:
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1        # peak strictly to the right
        else:
            hi = mid            # mid could be the peak
    return lo
```

### 5. Median of Two Sorted Arrays (Hard)
Binary-search the partition of the *shorter* array so that `left_max ≤ right_min`.

```python
def find_median_sorted_arrays(a: list[int], b: list[int]) -> float:
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    half = (m + n + 1) // 2
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2               # take i from a, j from b
        j = half - i
        l1 = a[i - 1] if i > 0 else float("-inf")
        r1 = a[i]     if i < m else float("inf")
        l2 = b[j - 1] if j > 0 else float("-inf")
        r2 = b[j]     if j < n else float("inf")
        if l1 <= r2 and l2 <= r1:
            if (m + n) % 2:
                return float(max(l1, l2))
            return (max(l1, l2) + min(r1, r2)) / 2
        if l1 > r2:
            hi = i - 1
        else:
            lo = i + 1
    raise ValueError("inputs not sorted")
```
`O(log(min(m, n)))`. The `±inf` sentinels remove all boundary special-casing — that trick is the whole trick.

## Variations you should name unprompted

| Variation | Twist | Key idea |
| --- | --- | --- |
| Real-valued search | tolerance `ε` instead of integers | `while hi - lo > eps` (or fixed ~100 iterations) |
| Search 2D matrix (LC 240) | rows & cols sorted | staircase from top-right, `O(m+n)`; or treat as flat sorted array `O(log mn)` |
| First/last occurrence | duplicates | run lower_bound and upper_bound |
| Rotated min (LC 153) | find pivot | compare `nums[mid]` to `nums[hi]` |
| Peak / valley | local extremum | move toward the higher neighbor |

## Pitfalls

- **`bisect` exists.** `bisect_left`/`bisect_right` from the standard library is production-correct — reach for it unless the interviewer wants the loop. Know both.
- **Wrong initial `hi`.** Index searches use `len(nums)-1`; insertion/boundary searches use `len(nums)`.
- **Non-integer midpoints** in answer-search: keep the space integral, or use the `ε`/fixed-iteration form for floats.
- **Predicate not actually monotonic.** If `can_finish(k)` isn't monotone, the boundary is undefined — reframe the problem.

## Q&A

<details class="qa"><summary>How do you avoid the classic off-by-one / infinite loop?</summary>
<div class="qa-body">

**Short:** Pick one template and keep an invariant. I use `while lo < hi` where the answer always lives in `[lo, hi]`; the branch that could *be* the answer sets `hi = mid`, the other sets `lo = mid + 1`. That guarantees strict shrinkage.

**Deep:** The failure mode is `lo = mid` with `mid = (lo+hi)//2` — when `hi = lo+1`, `mid == lo`, so `lo = mid` is a no-op → infinite loop. Either bias the midpoint up (`mid = (lo+hi+1)//2`) for the `lo = mid` branch, or restructure so the kept side is `hi`. I verify on the two-element array before running.
</div></details>

<details class="qa"><summary>When would you binary search on the answer instead of the array?</summary>
<div class="qa-body">

**Short:** When the answer is a number in a known range and there's a cheap monotone feasibility check — the array itself may be unsorted or irrelevant.

**Deep:** Signal phrases: "minimize the maximum," "maximum minimum," "smallest capacity/speed/time that works." I define `feasible(x)`, argue monotonicity (if `x` works, `x+1` trivially works), bound the range `[lo, hi]`, and run `lower_bound`. Cost is `O(log(range) · cost(feasible))`. This is the pattern that separates people who memorized 704 from people who understand binary search.
</div></details>

**Follow-ups you should expect**
- "Now the array has duplicates." → rotated search degrades to `O(N)`; explain why the sorted-half test breaks.
- "Do it without the library." → write `lower_bound` from scratch.
- "The feasibility check is `O(N)` — total complexity?" → `O(N log(range))`.
- "Floats instead of ints?" → fixed iteration count for guaranteed precision.

## Cheat-sheet

| Fact | Detail |
| --- | --- |
| Core invariant | answer stays in `[lo, hi]`; shrink strictly every iteration |
| Overflow-safe mid | `mid = lo + (hi - lo) // 2` |
| Keep-mid rule | branch that may be the answer sets `hi = mid`; other sets `lo = mid + 1` |
| Lower vs upper bound | `>=` vs `>` in the predicate (`bisect_left` vs `bisect_right`) |
| Index search | `hi = len-1`, loop `lo <= hi` |
| Boundary search | `hi = len`, loop `lo < hi` |
| Answer search | define monotone `feasible(x)`, cost `O(log range · check)` |
| Float search | loop on `hi - lo > eps` or fix ~100 iterations |
| Complexity | `O(log N)` time, `O(1)` space |

**Related:** [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) · [Trees & BSTs](#/coding/trees-bst) · [Greedy & Intervals](#/coding/greedy-intervals) · back to [The Core Patterns](#/coding/patterns) and [Coding Round Strategy](#/coding/strategy).
