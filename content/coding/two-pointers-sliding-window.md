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

## Practice — implement, run, test

> [!TIP] How to use this section
> Each problem below has a **live Python editor**. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime (~10 MB); later runs are instant. Prefer your own editor? Each problem links out to **LeetCode**.

Work them roughly in order — the two-pointer warm-ups first, then the sliding-window escalation from Easy to Hard.

### 1. Valid Palindrome <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/valid-palindrome/)

Ignore non-alphanumerics and case; is `s` a palindrome?

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"is_palindrome","starter":"def is_palindrome(s: str) -> bool:\n    # two pointers from both ends; skip non-alphanumerics\n    pass","tests":[{"args":["A man, a plan, a canal: Panama"],"expect":true},{"args":["race a car"],"expect":false},{"args":[" "],"expect":true},{"args":["0P"],"expect":false},{"args":[""],"expect":true}],"solution":"def is_palindrome(s: str) -> bool:\n    left, right = 0, len(s) - 1\n    while left < right:\n        while left < right and not s[left].isalnum():\n            left += 1\n        while left < right and not s[right].isalnum():\n            right -= 1\n        if s[left].lower() != s[right].lower():\n            return False\n        left, right = left + 1, right - 1\n    return True"}
</script>
</div>

*O(N) time, O(1) space.* **Pitfall:** building a filtered copy first uses O(N) space — the in-place two-pointer is the expected answer.

### 2. Two Sum II — Sorted Input <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

Return the 1-indexed pair summing to `target` in a sorted array.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"two_sum","starter":"def two_sum(numbers: list[int], target: int) -> list[int]:\n    # converging pointers on the sorted array; return the 1-indexed pair\n    pass","tests":[{"args":[[2,7,11,15],9],"expect":[1,2]},{"args":[[2,3,4],6],"expect":[1,3]},{"args":[[-1,0],-1],"expect":[1,2]},{"args":[[5,25,75],100],"expect":[2,3]}],"solution":"def two_sum(numbers: list[int], target: int) -> list[int]:\n    lo, hi = 0, len(numbers) - 1\n    while lo < hi:\n        s = numbers[lo] + numbers[hi]\n        if s == target:\n            return [lo + 1, hi + 1]\n        if s < target:\n            lo += 1\n        else:\n            hi -= 1\n    return []"}
</script>
</div>

*O(N) time, O(1) space.* **Pitfall:** using the hash-map [Two Sum](#/coding/hashing) here wastes the sorted guarantee and O(1) space — the interviewer gave you sortedness on purpose.

### 3. Container With Most Water <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/container-with-most-water/)

Two lines forming the container with the most water; return max area.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"max_area","starter":"def max_area(height: list[int]) -> int:\n    # two pointers; always move the shorter wall inward\n    pass","tests":[{"args":[[1,8,6,2,5,4,8,3,7]],"expect":49},{"args":[[1,1]],"expect":1},{"args":[[4,3,2,1,4]],"expect":16},{"args":[[1,2,1]],"expect":2}],"solution":"def max_area(height: list[int]) -> int:\n    lo, hi, best = 0, len(height) - 1, 0\n    while lo < hi:\n        best = max(best, min(height[lo], height[hi]) * (hi - lo))\n        if height[lo] < height[hi]:\n            lo += 1\n        else:\n            hi -= 1\n    return best"}
</script>
</div>

*O(N) time, O(1) space.* **Why move the shorter side?** Width only shrinks as pointers converge, and area is bounded by the shorter wall; moving the taller side can never increase `min(...)`, so it's strictly dominated.

### 4. Longest Substring Without Repeating Characters <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

Length of the longest substring with all-distinct characters.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"length_of_longest_substring","starter":"def length_of_longest_substring(s: str) -> int:\n    # sliding window with a last-seen index map\n    pass","tests":[{"args":["abcabcbb"],"expect":3},{"args":["bbbbb"],"expect":1},{"args":["pwwkew"],"expect":3},{"args":[""],"expect":0},{"args":["dvdf"],"expect":3}],"solution":"def length_of_longest_substring(s: str) -> int:\n    last_seen = {}\n    start = best = 0\n    for i, ch in enumerate(s):\n        if ch in last_seen and last_seen[ch] >= start:\n            start = last_seen[ch] + 1\n        last_seen[ch] = i\n        best = max(best, i - start + 1)\n    return best"}
</script>
</div>

*O(N) time, O(min(N, alphabet)) space.* **Pitfall:** the `last_seen[ch] >= start` guard is essential — without it, a stale index from outside the window drags `start` backward.

### 5. Longest Repeating Character Replacement <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/longest-repeating-character-replacement/)

Longest substring of one repeated character after replacing at most `k` characters.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"character_replacement","starter":"from collections import defaultdict\n\ndef character_replacement(s: str, k: int) -> int:\n    # window stays valid while (window_size - max_freq) <= k\n    pass","tests":[{"args":["ABAB",2],"expect":4},{"args":["AABABBA",1],"expect":4},{"args":["AAAA",0],"expect":4},{"args":["ABCDE",1],"expect":2}],"solution":"from collections import defaultdict\n\ndef character_replacement(s: str, k: int) -> int:\n    count = defaultdict(int)\n    left = max_freq = best = 0\n    for right, ch in enumerate(s):\n        count[ch] += 1\n        max_freq = max(max_freq, count[ch])\n        if (right - left + 1) - max_freq > k:\n            count[s[left]] -= 1\n            left += 1\n        best = max(best, right - left + 1)\n    return best"}
</script>
</div>

*O(N) time, O(alphabet) space.* **Pitfall:** don't recompute `max_freq` on shrink — keeping its historical max is correct because `best` only grows when the window grows.

### 6. Minimum Window Substring <span class="badge badge-hard">Hard</span> · [LeetCode ↗](https://leetcode.com/problems/minimum-window-substring/)

Smallest substring of `s` containing all characters of `t` (with multiplicity).

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"min_window","starter":"from collections import Counter\n\ndef min_window(s: str, t: str) -> str:\n    # expand right; when t is fully covered, shrink left to minimize\n    pass","tests":[{"args":["ADOBECODEBANC","ABC"],"expect":"BANC"},{"args":["a","a"],"expect":"a"},{"args":["a","aa"],"expect":""},{"args":["cabwefgewcwaefgcf","cae"],"expect":"cwae"}],"solution":"from collections import Counter\n\ndef min_window(s: str, t: str) -> str:\n    if not s or not t:\n        return \"\"\n    need = Counter(t)\n    missing = len(t)\n    left = best_start = 0\n    best_len = float(\"inf\")\n    for right, ch in enumerate(s):\n        if need[ch] > 0:\n            missing -= 1\n        need[ch] -= 1\n        while missing == 0:\n            if right - left + 1 < best_len:\n                best_len, best_start = right - left + 1, left\n            need[s[left]] += 1\n            if need[s[left]] > 0:\n                missing += 1\n            left += 1\n    return \"\" if best_len == float(\"inf\") else s[best_start:best_start + best_len]"}
</script>
</div>

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
