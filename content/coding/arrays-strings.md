# Arrays & Strings

> [!TIP] The cue
> Array/string problems are the interview warm-up and the substrate for every other pattern. The decision you make in the first minute: **is a single linear scan enough, do I need to sort, or do I need a helper structure (hash map / count array)?** Say which and why before you type.

Most "array" problems reduce to one of a few moves: a single pass with running state, a prefix/suffix aggregate, an in-place two-pointer rewrite, or a reversal trick. When the core difficulty is *frequency* or *lookup*, jump to [Hashing](#/coding/hashing); when it's *contiguous windows* or *converging ends*, jump to [Two Pointers](#/coding/two-pointers-sliding-window).

## When to use which move

| Cue in the statement | Move | Complexity |
| --- | --- | --- |
| "in-place", "O(1) extra space", overwrite | Two-index write / swap | O(N) time, O(1) space |
| "product/sum except self", running aggregate | Prefix + suffix pass | O(N) time, O(1) extra |
| "reverse / rotate" | Reversal trick (3 reverses) | O(N) time, O(1) space |
| "max profit / best so far" | Single pass with running extremum | O(N) time, O(1) space |
| group by signature (anagrams) | Hash on a canonical key | O(N·K) |
| "trap water / two walls" | Two-pointer with running maxes | O(N) time, O(1) space |

## Worked examples

### 1. Best Time to Buy and Sell Stock <span class="badge badge-easy">Easy</span>

Given daily `prices`, maximize profit from one buy followed by a later sell; 0 if none.

**Approach.** Brute force checks all (buy, sell) pairs → O(N²). Optimize: scan left to right, keep the **minimum price seen so far**, and update best profit as `price − min_so_far`.

```python
def max_profit(prices: list[int]) -> int:
    if not prices:
        return 0
    min_so_far = prices[0]
    best = 0
    for price in prices[1:]:
        best = max(best, price - min_so_far)   # compute profit BEFORE updating min
        min_so_far = min(min_so_far, price)
    return best
```

*O(N) time, O(1) space.* **Pitfall:** update the min *after* computing profit, or you'll allow a same-day buy/sell.

### 2. Product of Array Except Self <span class="badge badge-med">Medium</span>

Return `out[i] = product of all elements except nums[i]`, no division, better than O(N²).

**Approach.** `out[i] = (product of everything left of i) × (product of everything right of i)`. One left-to-right pass fills left products; one right-to-left pass multiplies in the right products.

```python
def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    out = [1] * n
    left = 1
    for i in range(n):
        out[i] = left
        left *= nums[i]
    right = 1
    for i in range(n - 1, -1, -1):
        out[i] *= right
        right *= nums[i]
    return out
```

*O(N) time, O(1) extra space* (output array excluded). **Pitfall:** the division approach (`total // nums[i]`) breaks on any zero — the prefix/suffix method handles any number of zeros safely.

### 3. Rotate Array <span class="badge badge-med">Medium</span>

Rotate `nums` right by `k`, in place with O(1) extra space.

**Approach.** The **triple reversal** trick: reverse all, reverse the first `k`, reverse the rest.

```python
def rotate(nums: list[int], k: int) -> None:
    n = len(nums)
    if n == 0:
        return
    k %= n                     # essential: k can exceed n
    def reverse(lo: int, hi: int) -> None:
        while lo < hi:
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo, hi = lo + 1, hi - 1
    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
```

*O(N) time, O(1) space.* **Pitfall:** forgetting `k %= n` (index error when `k > n`); mixing up the three reversal ranges — desk-check on `[1,2,3,4,5], k=2`.

### 4. Group Anagrams <span class="badge badge-med">Medium</span>

Group words that are anagrams of each other.

**Approach.** Two words are anagrams iff they share a canonical signature. Sorting the letters is O(K log K) per word; a **26-length count vector** (as a hashable tuple) is O(K). In an interview, mention the sort first, then optimize to the count vector.

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for ch in s:
            count[ord(ch) - ord('a')] += 1
        groups[tuple(count)].append(s)      # tuple is hashable → dict key
    return list(groups.values())
```

*O(N·K) time* (N words, K avg length), *O(N·K) space.* This is really a [Hashing](#/coding/hashing) problem in array clothing.

### 5. Trapping Rain Water <span class="badge badge-hard">Hard</span>

Total water trapped between bars of the given `height`s.

**Approach.** Water above index `i` is `min(max_left, max_right) − height[i]`. Brute force recomputes maxes → O(N²). Prefix/suffix-max arrays give O(N) time, O(N) space. **Optimal: two pointers**, O(1) space — move the side with the smaller running max, because that side's water level is then fully determined.

```python
def trap(height: list[int]) -> int:
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]
    return water
```

*O(N) time, O(1) space.* **Pitfall:** be ready to *explain* why the smaller-max side is safe to finalize — interviewers always follow up. Practice the prefix/suffix version first, then the space-optimal two-pointer.

## Common variations

- **In-place constraint added** (Remove Duplicates, Move Zeroes): keep a slow write-index and a fast read-index; overwrite from the front. This is the [two-pointer](#/coding/two-pointers-sliding-window) in-place idiom.
- **String parsing/formatting** (`atoi`, run-length compression): the code is easy; the score is in exhaustive edge cases — overflow, leading/trailing spaces, signs, empty input. List them out loud first.
- **Matrix as array** (spiral order, rotate image, set-matrix-zeroes): same moves in 2-D; watch boundary indices and whether you can mutate in place.
- **ML dressing**: "sliding max/mean over a 1-D signal (e.g., logits over frames)" is an array problem wearing a time-series costume — the pattern is unchanged.

## Pitfalls

> [!WARNING] The usual array bugs
> - Off-by-one on the last index (`range(n)` vs `range(n-1)`), especially in reversal ranges.
> - Mutating a list while iterating it — iterate over a copy or use indices.
> - Assuming a single pass suffices when order matters both ways (prefix *and* suffix).
> - Using `list.pop(0)` / `insert(0, x)` in a loop — that's O(N) each; use a [`deque`](#/coding/stack-queue).
> - Slicing (`nums[1:]`) copies — fine for readability, but say "O(N) extra" if space is scored.

<details class="qa"><summary>Product Except Self — how would you handle it if division <em>were</em> allowed?</summary>
<div class="qa-body">

**Short:** Compute the total product and divide, but special-case zeros.

**Deep:** With no zeros, `out[i] = total / nums[i]` in O(N). With exactly one zero, every `out[i]` is 0 except at the zero's index, which gets the product of all non-zero elements. With two or more zeros, every entry is 0. The prefix/suffix method sidesteps all this branching and is what interviewers prefer, so lead with it and offer division only as the "if the array is guaranteed non-zero" simplification.
</div></details>

<details class="qa"><summary>Trapping Rain Water — why is the smaller-max side safe to commit?</summary>
<div class="qa-body">

**Short:** If `left_max < right_max`, the water at `left` is bounded by `left_max` regardless of what's between the pointers, because some bar at least `right_max` (> `left_max`) exists to the right.

**Deep:** The trapped height at any index is `min(max_left, max_right) − height[i]`. When we move the pointer on the smaller-max side, we *know* the opposite side has a max at least as large, so `min(...)` on the moving side equals its own running max — no future bar can lower it. That guarantee is exactly why we can finalize the water there in O(1) space without ever seeing the far side's exact profile.
</div></details>

<details class="qa"><summary>Follow-ups</summary>
<div class="qa-body">

- **"Prices with a transaction fee, or at most k transactions?"** → this becomes a [DP](#/coding/dynamic-programming) (state = day × holding × transactions left).
- **"Rotate a 2-D matrix 90°?"** → transpose then reverse each row, in place.
- **"Group anagrams if input is a stream?"** → emit/merge groups incrementally keyed by the count-vector signature.
</div></details>

## Cheat-sheet

| Problem type | Key move | Time | Space |
| --- | --- | --- | --- |
| Best-so-far (max profit) | single pass + running extremum | O(N) | O(1) |
| Except-self aggregate | prefix + suffix pass | O(N) | O(1)* |
| Rotate / reverse | triple reversal, `k %= n` | O(N) | O(1) |
| Group by signature | hash on canonical key | O(N·K) | O(N·K) |
| Trap water / two walls | two pointers + running maxes | O(N) | O(1) |
| Remove dup / move zeroes | slow-write / fast-read indices | O(N) | O(1) |

\*output array excluded.

**Related:** [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) · [Hashing](#/coding/hashing) · [Greedy & Intervals](#/coding/greedy-intervals) · [Patterns hub](#/coding/patterns)
