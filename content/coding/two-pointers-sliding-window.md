# Two Pointers & Sliding Window

> [!TIP] The cue
> Two closely related patterns that both collapse an O(N²) scan into O(N). **Two pointers:** a sorted array where you converge from both ends (pairs, palindromes, containers). **Sliding window:** a contiguous subarray/substring whose window you expand and shrink to maintain a constraint (max/min length, ≤ k distinct, sum ≥ target).

The tell for **two pointers** is *sorted input + a pairing question*. The tell for **sliding window** is *contiguous* + *a constraint you can maintain incrementally as the window moves*. A third cousin — **fast & slow pointers** — handles cycles (linked-list loop, "find the duplicate number").

## When to use which

| Cue | Variant | Complexity |
| --- | --- | --- |
| Sorted array, find pair/triplet summing to target | Converging two pointers | O(N) / O(N²) |
| Palindrome check, reverse in place | Opposing two pointers | O(N), O(1) space |
| Max area / trap between two ends | Opposing, move smaller side | O(N), O(1) space |
| Longest/shortest **contiguous** window with a constraint | Sliding window | O(N) |
| "At most k …" then "exactly k" | `atMost(k) − atMost(k−1)` | O(N) |
| Cycle in linked list / find duplicate | Fast & slow (Floyd) | O(N), O(1) space |

## The two templates

```python
# Converging two pointers (sorted input)
def two_pointer(a: list[int], target: int) -> bool:
    lo, hi = 0, len(a) - 1
    while lo < hi:
        s = a[lo] + a[hi]
        if s == target: return True
        if s < target:  lo += 1     # need larger sum
        else:           hi -= 1     # need smaller sum
    return False

# Sliding window (grow right, shrink left while invariant is violated)
def sliding_window(s: str) -> int:
    left, best = 0, 0
    state = {}                       # window's running state
    for right, ch in enumerate(s):
        # 1. include s[right] into state
        while window_is_invalid(state):
            # 2. evict s[left] from state
            left += 1
        best = max(best, right - left + 1)   # 3. window [left, right] is valid
    return best
```

## Worked examples

### 1. Valid Palindrome <span class="badge badge-easy">Easy</span>

Ignore non-alphanumerics and case; is `s` a palindrome?

```python
def is_palindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left, right = left + 1, right - 1
    return True
```

*O(N) time, O(1) space.* **Pitfall:** building a filtered copy first uses O(N) space — the in-place two-pointer is the expected answer.

### 2. Two Sum II — Sorted Input <span class="badge badge-easy">Easy</span>

Return the 1-indexed pair summing to `target` in a sorted array.

```python
def two_sum(numbers: list[int], target: int) -> list[int]:
    lo, hi = 0, len(numbers) - 1
    while lo < hi:
        s = numbers[lo] + numbers[hi]
        if s == target:
            return [lo + 1, hi + 1]
        if s < target:
            lo += 1
        else:
            hi -= 1
    return []
```

*O(N) time, O(1) space.* **Pitfall:** using the hash-map [Two Sum](#/coding/hashing) here wastes the sorted guarantee and O(1) space — the interviewer gave you sortedness on purpose.

### 3. Container With Most Water <span class="badge badge-med">Medium</span>

Two lines forming the container with the most water; return max area.

```python
def max_area(height: list[int]) -> int:
    lo, hi, best = 0, len(height) - 1, 0
    while lo < hi:
        best = max(best, min(height[lo], height[hi]) * (hi - lo))
        if height[lo] < height[hi]:
            lo += 1                  # move the shorter side
        else:
            hi -= 1
    return best
```

*O(N) time, O(1) space.* **Why move the shorter side?** Width only shrinks as pointers converge, and area is bounded by the shorter wall; moving the taller side can never increase `min(...)`, so it's strictly dominated.

### 4. Longest Substring Without Repeating Characters <span class="badge badge-med">Medium</span>

Length of the longest substring with all-distinct characters.

```python
def length_of_longest_substring(s: str) -> int:
    last_seen: dict[str, int] = {}
    start = best = 0
    for i, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= start:
            start = last_seen[ch] + 1     # jump past the earlier occurrence
        last_seen[ch] = i
        best = max(best, i - start + 1)
    return best
```

*O(N) time, O(min(N, alphabet)) space.* **Pitfall:** the `last_seen[ch] >= start` guard is essential — without it, a stale index from outside the window drags `start` backward.

### 5. Longest Repeating Character Replacement <span class="badge badge-med">Medium</span>

Longest substring of one repeated character after replacing at most `k` characters.

```python
from collections import defaultdict

def character_replacement(s: str, k: int) -> int:
    count: dict[str, int] = defaultdict(int)
    left = max_freq = best = 0
    for right, ch in enumerate(s):
        count[ch] += 1
        max_freq = max(max_freq, count[ch])
        # chars to replace = window size − most frequent char count
        if (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

*O(N) time, O(alphabet) space.* **Pitfall:** don't recompute `max_freq` on shrink — keeping its historical max is correct because `best` only grows when the window grows.

### 6. Minimum Window Substring <span class="badge badge-hard">Hard</span>

Smallest substring of `s` containing all characters of `t` (with multiplicity).

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not s or not t:
        return ""
    need = Counter(t)
    missing = len(t)                  # total chars still required (with dups)
    left = best_start = 0
    best_len = float("inf")
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1                 # negative = surplus
        while missing == 0:           # window is valid; shrink from left
            if right - left + 1 < best_len:
                best_len, best_start = right - left + 1, left
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    return "" if best_len == float("inf") else s[best_start:best_start + best_len]
```

*O(|s| + |t|) time, O(alphabet) space.* **Pitfall:** track `missing` as total required chars (counting multiplicity), and treat negative `need` as surplus — dry-run to convince yourself.

## Common variations

- **"Exactly k distinct" via at-most.** `exactly(k) = atMost(k) − atMost(k−1)`. This turns a hard "exactly" window into two easy monotone windows — a very common follow-up.
- **Fixed-size window** (max sum of size k, sliding-window average): no shrink loop; add the entering element, subtract the leaving one.
- **Sliding-window maximum**: needs a **monotonic deque**, not plain pointers — see [Stacks & Queues](#/coding/stack-queue).
- **Fast & slow pointers**: Linked List Cycle, Find the Duplicate Number, and finding a list's midpoint all use a 1×/2× speed pair with O(1) space.
- **ML dressing**: "longest run of frames with confidence ≥ threshold" is a fixed-constraint window.

## Pitfalls

> [!WARNING] Window bugs to rehearse against
> - Shrinking with `if` when you need `while` (multiple evictions per step).
> - Updating `best` before the window is valid (in shrink-to-minimize problems, update *inside* the valid region).
> - For two pointers, moving both ends on a tie when only one should move.
> - Forgetting that sliding window requires a **monotone** invariant — it breaks with negative numbers for "subarray sum = k"; use [prefix sum + hashing](#/coding/hashing) there instead.

<details class="qa"><summary>When does sliding window fail, and what replaces it?</summary>
<div class="qa-body">

**Short:** When the constraint isn't monotone in the window — most notably subarray-sum problems with negative numbers.

**Deep:** Sliding window relies on being able to safely shrink from the left once the window is "too big/valid," which requires that extending the window moves the quantity in one direction. With all non-negative values, a running sum only increases as you add elements, so shrinking to restore `sum ≤ target` is valid. Introduce negatives and adding an element can *decrease* the sum, so a shrunk window might still be improvable — the monotonicity is gone. Then you switch to **prefix sum + hash map** (see [Subarray Sum Equals K](#/coding/hashing)), which counts qualifying windows without relying on monotonicity.
</div></details>

<details class="qa"><summary>Two Sum in a sorted array — two pointers or binary search?</summary>
<div class="qa-body">

**Short:** Two pointers: O(N) time, O(1) space, one pass.

**Deep:** You *could* fix each element and binary-search its complement for O(N log N). Two pointers is strictly better here because each comparison discards one candidate from one end — an O(N) linear convergence. Binary search wins only when you're querying many independent targets against a static sorted array (amortize the sort, then O(log N) per query). State the trade-off; it's a common probe.
</div></details>

<details class="qa"><summary>Follow-ups</summary>
<div class="qa-body">

- **"3Sum / 4Sum?"** → sort, fix outer indices, two-pointer the rest; skip duplicates carefully. O(N²)/O(N³).
- **"Subarrays with exactly k distinct integers?"** → the at-most-k trick.
- **"Detect the *start* of a linked-list cycle?"** → Floyd's: after meeting, reset one pointer to head, advance both by 1; they meet at the entrance.
</div></details>

## Cheat-sheet

| Problem | Technique | Time | Space |
| --- | --- | --- | --- |
| Pair sum, sorted | converging pointers | O(N) | O(1) |
| Palindrome / reverse | opposing pointers | O(N) | O(1) |
| Max container / trap | move shorter side | O(N) | O(1) |
| Longest distinct substring | window + last-seen map | O(N) | O(k) |
| Char replacement ≤ k | window + max-freq | O(N) | O(1) |
| Min window substring | window + need/missing | O(N) | O(1) |
| Exactly k distinct | atMost(k)−atMost(k−1) | O(N) | O(k) |
| Cycle detection | fast & slow (Floyd) | O(N) | O(1) |

**Related:** [Arrays & Strings](#/coding/arrays-strings) · [Hashing](#/coding/hashing) · [Binary Search](#/coding/binary-search) · [Patterns hub](#/coding/patterns)
