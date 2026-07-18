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

## Practice — implement, run, test

> [!TIP] How to use this section
> Each problem below has a **live Python editor**. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime (~10 MB); later runs are instant. Prefer your own editor? Each problem links out to **LeetCode**.

### 1. Best Time to Buy and Sell Stock <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

Given daily `prices`, maximize profit from one buy followed by a later sell; 0 if none.

**Approach.** Brute force checks all (buy, sell) pairs → O(N²). Optimize: scan left to right, keep the **minimum price seen so far**, and update best profit as `price − min_so_far`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"max_profit","starter":"def max_profit(prices: list[int]) -> int:\n    # single pass; track the min price so far, best = price - min\n    pass","tests":[{"args":[[7,1,5,3,6,4]],"expect":5},{"args":[[7,6,4,3,1]],"expect":0},{"args":[[1,2,3,4,5]],"expect":4},{"args":[[2,4,1]],"expect":2},{"args":[[5]],"expect":0}],"solution":"def max_profit(prices: list[int]) -> int:\n    if not prices:\n        return 0\n    min_so_far = prices[0]\n    best = 0\n    for price in prices[1:]:\n        best = max(best, price - min_so_far)\n        min_so_far = min(min_so_far, price)\n    return best"}
</script>
</div>

*O(N) time, O(1) space.* **Pitfall:** update the min *after* computing profit, or you'll allow a same-day buy/sell.

### 2. Product of Array Except Self <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/product-of-array-except-self/)

Return `out[i] = product of all elements except nums[i]`, no division, better than O(N²).

**Approach.** `out[i] = (product of everything left of i) × (product of everything right of i)`. One left-to-right pass fills left products; one right-to-left pass multiplies in the right products.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"product_except_self","starter":"def product_except_self(nums: list[int]) -> list[int]:\n    # prefix pass fills left products; suffix pass multiplies in right products\n    pass","tests":[{"args":[[1,2,3,4]],"expect":[24,12,8,6]},{"args":[[-1,1,0,-3,3]],"expect":[0,0,9,0,0]},{"args":[[2,3]],"expect":[3,2]},{"args":[[0,0]],"expect":[0,0]}],"solution":"def product_except_self(nums: list[int]) -> list[int]:\n    n = len(nums)\n    out = [1] * n\n    left = 1\n    for i in range(n):\n        out[i] = left\n        left *= nums[i]\n    right = 1\n    for i in range(n - 1, -1, -1):\n        out[i] *= right\n        right *= nums[i]\n    return out"}
</script>
</div>

*O(N) time, O(1) extra space* (output array excluded). **Pitfall:** the division approach (`total // nums[i]`) breaks on any zero — the prefix/suffix method handles any number of zeros safely.

### 3. Rotate Array <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/rotate-array/)

Rotate `nums` right by `k`, in place with O(1) extra space.

**Approach.** The **triple reversal** trick: reverse all, reverse the first `k`, reverse the rest.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"rotate","starter":"def rotate(nums: list[int], k: int) -> list[int]:\n    # triple reversal in place (k %= n first); return the rotated list\n    pass","tests":[{"args":[[1,2,3,4,5,6,7],3],"expect":[5,6,7,1,2,3,4]},{"args":[[-1,-100,3,99],2],"expect":[3,99,-1,-100]},{"args":[[1,2],3],"expect":[2,1]},{"args":[[1],0],"expect":[1]}],"solution":"def rotate(nums: list[int], k: int) -> list[int]:\n    n = len(nums)\n    if n == 0:\n        return nums\n    k %= n\n    def reverse(lo: int, hi: int) -> None:\n        while lo < hi:\n            nums[lo], nums[hi] = nums[hi], nums[lo]\n            lo, hi = lo + 1, hi - 1\n    reverse(0, n - 1)\n    reverse(0, k - 1)\n    reverse(k, n - 1)\n    return nums"}
</script>
</div>

*O(N) time, O(1) space.* **Pitfall:** forgetting `k %= n` (index error when `k > n`); mixing up the three reversal ranges — desk-check on `[1,2,3,4,5], k=2`.

### 4. Group Anagrams <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/group-anagrams/)

Group words that are anagrams of each other.

**Approach.** Two words are anagrams iff they share a canonical signature. Sorting the letters is O(K log K) per word; a **26-length count vector** (as a hashable tuple) is O(K). In an interview, mention the sort first, then optimize to the count vector.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"group_anagrams","starter":"from collections import defaultdict\n\ndef group_anagrams(strs: list[str]) -> list[list[str]]:\n    # hash each word by its 26-length count vector (as a tuple)\n    pass","tests":[{"args":[["eat","tea","tan","ate","nat","bat"]],"expect":[["eat","tea","ate"],["tan","nat"],["bat"]],"unordered":true},{"args":[[""]],"expect":[[""]],"unordered":true},{"args":[["a"]],"expect":[["a"]],"unordered":true}],"solution":"from collections import defaultdict\n\ndef group_anagrams(strs: list[str]) -> list[list[str]]:\n    groups = defaultdict(list)\n    for s in strs:\n        count = [0] * 26\n        for ch in s:\n            count[ord(ch) - ord('a')] += 1\n        groups[tuple(count)].append(s)\n    return list(groups.values())"}
</script>
</div>

*O(N·K) time* (N words, K avg length), *O(N·K) space.* This is really a [Hashing](#/coding/hashing) problem in array clothing.

### 5. Trapping Rain Water <span class="badge badge-hard">Hard</span> · [LeetCode ↗](https://leetcode.com/problems/trapping-rain-water/)

Total water trapped between bars of the given `height`s.

**Approach.** Water above index `i` is `min(max_left, max_right) − height[i]`. Brute force recomputes maxes → O(N²). Prefix/suffix-max arrays give O(N) time, O(N) space. **Optimal: two pointers**, O(1) space — move the side with the smaller running max, because that side's water level is then fully determined.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"trap","starter":"def trap(height: list[int]) -> int:\n    # two pointers; move the side with the smaller running max\n    pass","tests":[{"args":[[0,1,0,2,1,0,1,3,2,1,2,1]],"expect":6},{"args":[[4,2,0,3,2,5]],"expect":9},{"args":[[]],"expect":0},{"args":[[1,2,3]],"expect":0}],"solution":"def trap(height: list[int]) -> int:\n    if not height:\n        return 0\n    left, right = 0, len(height) - 1\n    left_max, right_max = height[left], height[right]\n    water = 0\n    while left < right:\n        if left_max < right_max:\n            left += 1\n            left_max = max(left_max, height[left])\n            water += left_max - height[left]\n        else:\n            right -= 1\n            right_max = max(right_max, height[right])\n            water += right_max - height[right]\n    return water"}
</script>
</div>

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
