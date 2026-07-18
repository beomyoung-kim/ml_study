# Hashing

> [!TIP] The cue
> Whenever a brute force does repeated **membership, complement, frequency, or "have I seen this before?"** lookups inside a loop, a hash map/set buys you **average O(1)** on each and collapses the nested loop to O(N). If you catch yourself writing an inner loop just to *find* something, reach for a hash.

The trade is time for space: you spend O(N) memory to remember what you've seen — values, their indices, their counts, or prefix aggregates. The classic upgrades are **complement lookup** (Two Sum), **prefix-sum + hash** (subarray counts, even with negatives), and **canonical-key grouping** (anagrams — see [Arrays & Strings](#/coding/arrays-strings)).

## When to use which

| Cue | Store in the hash | Complexity |
| --- | --- | --- |
| Pair summing to target | value → index (look up complement) | O(N) time / O(N) space |
| Duplicate exists? | set of seen values | O(N) / O(N) |
| Frequency / top-k | value → count (`Counter`) | O(N) / O(N) |
| Subarray sum/count = k (incl. negatives) | prefix-sum → how many times seen | O(N) / O(N) |
| Group by property | canonical key → list | O(N·K) |
| Longest consecutive run | set for O(1) neighbor checks | O(N) / O(N) |
| O(1) get/put with eviction | hash + ordering structure | O(1)/op |

## Practice — implement, run, test

> [!TIP] How to use this section
> Each problem below has a **live Python editor**. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime (~10 MB); later runs are instant. Prefer your own editor? Each problem links out to **LeetCode**.

### 1. Two Sum <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/two-sum/)

Indices of the two numbers adding to `target`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"two_sum","starter":"def two_sum(nums: list[int], target: int) -> list[int]:\n    # store value -> index; look up the complement before inserting\n    pass","tests":[{"args":[[2,7,11,15],9],"expect":[0,1]},{"args":[[3,2,4],6],"expect":[1,2]},{"args":[[3,3],6],"expect":[0,1]},{"args":[[-3,4,3,90],0],"expect":[0,2]}],"solution":"def two_sum(nums: list[int], target: int) -> list[int]:\n    index_of = {}\n    for i, value in enumerate(nums):\n        if target - value in index_of:\n            return [index_of[target - value], i]\n        index_of[value] = i\n    return []"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** insert *after* the complement check, or you can pair an element with itself. (If the array were sorted, prefer the O(1)-space [two-pointer](#/coding/two-pointers-sliding-window) version.)

### 2. Contains Duplicate <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/contains-duplicate/)

Return `True` if any value repeats.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"contains_duplicate","starter":"def contains_duplicate(nums: list[int]) -> bool:\n    # track seen values in a set; True on the first repeat\n    pass","tests":[{"args":[[1,2,3,1]],"expect":true},{"args":[[1,2,3,4]],"expect":false},{"args":[[1,1,1,3,3,4,3,2,4,2]],"expect":true},{"args":[[]],"expect":false}],"solution":"def contains_duplicate(nums: list[int]) -> bool:\n    seen = set()\n    for x in nums:\n        if x in seen:\n            return True\n        seen.add(x)\n    return False"}
</script>
</div>

*O(N) time, O(N) space.* Sorting + adjacent comparison is O(N log N) with O(1) extra but destroys order — mention both and the trade-off.

### 3. Subarray Sum Equals K <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/subarray-sum-equals-k/)

Count contiguous subarrays summing to `k` (values may be negative).

**Approach.** With prefix sum `S`, a subarray `(i, j]` sums to `k` iff `S[j] − S[i] = k`, i.e. `S[i] = S[j] − k`. As you scan, count how many earlier prefix sums equal `running − k`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"subarray_sum","starter":"from collections import defaultdict\n\ndef subarray_sum(nums: list[int], k: int) -> int:\n    # prefix sum + hash; count earlier prefixes equal to running - k\n    pass","tests":[{"args":[[1,1,1],2],"expect":2},{"args":[[1,2,3],3],"expect":2},{"args":[[1,-1,0],0],"expect":3},{"args":[[3,4,7,2,-3,1,4,2],7],"expect":4}],"solution":"from collections import defaultdict\n\ndef subarray_sum(nums: list[int], k: int) -> int:\n    count = defaultdict(int)\n    count[0] = 1\n    running = answer = 0\n    for x in nums:\n        running += x\n        answer += count[running - k]\n        count[running] += 1\n    return answer"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** the `count[0] = 1` seed — without it you miss subarrays starting at index 0. Sliding window can't be used here because negatives break monotonicity; hashing is required.

### 4. Top K Frequent Elements <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/top-k-frequent-elements/)

The `k` most frequent values (any order).

**Approach.** Count with a `Counter`, then **bucket sort by frequency** for O(N). A [heap](#/coding/heap-priority-queue) of size k is the O(N log k) alternative — mention it as the streaming-friendly option.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"top_k_frequent","starter":"from collections import Counter\n\ndef top_k_frequent(nums: list[int], k: int) -> list[int]:\n    # count, then bucket-sort by frequency; collect the top k\n    pass","tests":[{"args":[[1,1,1,2,2,3],2],"expect":[1,2],"unordered":true},{"args":[[1],1],"expect":[1],"unordered":true},{"args":[[4,1,-1,2,-1,2,3],2],"expect":[-1,2],"unordered":true}],"solution":"from collections import Counter\n\ndef top_k_frequent(nums: list[int], k: int) -> list[int]:\n    freq = Counter(nums)\n    buckets = [[] for _ in range(len(nums) + 1)]\n    for val, c in freq.items():\n        buckets[c].append(val)\n    out = []\n    for c in range(len(buckets) - 1, 0, -1):\n        for val in buckets[c]:\n            out.append(val)\n            if len(out) == k:\n                return out\n    return out"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** `sorted(freq, key=freq.get)` is an easy O(N log N) answer — write it if stuck, but volunteer the bucket/heap improvement.

### 5. Longest Consecutive Sequence <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/longest-consecutive-sequence/)

Length of the longest run of consecutive integers, in O(N), unsorted.

**Approach.** Put everything in a set; only start counting from a value that has no left neighbor (`num − 1` absent), so each run is walked exactly once.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"longest_consecutive","starter":"def longest_consecutive(nums: list[int]) -> int:\n    # put all in a set; only count runs from a value with no left neighbor\n    pass","tests":[{"args":[[100,4,200,1,3,2]],"expect":4},{"args":[[0,3,7,2,5,8,4,6,0,1]],"expect":9},{"args":[[]],"expect":0},{"args":[[1,2,0,1]],"expect":3}],"solution":"def longest_consecutive(nums: list[int]) -> int:\n    num_set = set(nums)\n    best = 0\n    for num in num_set:\n        if num - 1 in num_set:\n            continue\n        length, cur = 1, num\n        while cur + 1 in num_set:\n            cur, length = cur + 1, length + 1\n        best = max(best, length)\n    return best"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** running the inner `while` from *every* element is O(N²); the "start-only" guard is what keeps it linear.

### 6. LRU Cache <span class="badge badge-hard">Hard</span>

Fixed-capacity cache with O(1) `get` and `put`.

**Approach.** A hash map for O(1) lookup + an ordering structure for O(1) recency updates. Python's `OrderedDict` gives both; the "from scratch" version is a hash map + doubly linked list.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)          # mark most-recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)   # evict least-recently used
```

*O(1) average per op, O(capacity) space.* **Pitfall:** a plain `dict` can't reorder in O(1); use `OrderedDict`, and be ready to describe the hash-map-plus-DLL implementation ("evict from the tail, promote on access") if asked to avoid the stdlib.

## Common variations

- **Prefix-sum + hash family**: "subarray sum divisible by k" (store `running % k`), "contiguous array with equal 0s and 1s" (map 0→−1, store first index of each prefix sum), "longest subarray sum = k."
- **Two-pass complement**: 4Sum-count (hash all pair sums of two arrays, look up negatives from the other two) turns O(N⁴) into O(N²).
- **Design problems**: LFU Cache, Insert/Delete/GetRandom O(1) (hash + array with swap-to-end), Design HashMap.
- **Rolling hash** (Rabin–Karp): substring search / duplicate-substring detection as a hashing follow-up.
- **ML connection**: feature hashing (the "hashing trick"), embedding lookup tables, and dedup of near-identical samples all reduce to hash-map lookups — worth a one-line mention for rapport.

## Pitfalls

> [!WARNING] Hashing gotchas
> - Say **"average O(1)"** for hash operations, not worst-case — adversarial keys degrade to O(N).
> - Keys must be **hashable**: use `tuple`, not `list` (e.g., a frozen count vector for anagrams).
> - `defaultdict` creates missing keys on read — that can silently inflate your map; use `.get(k, default)` when you don't want insertion.
> - Storing objects by identity vs value — define `__hash__`/`__eq__` consistently, or key on a primitive signature.
> - Hash sets don't preserve order (pre-3.7 `dict` didn't either); don't rely on iteration order for correctness.

<details class="qa"><summary>When is sorting better than hashing for a "find pairs / dedup" problem?</summary>
<div class="qa-body">

**Short:** When you need O(1) *space*, ordering as a by-product, or worst-case guarantees — sorting's O(N log N)/O(1) beats hashing's O(N)/O(N).

**Deep:** Hashing wins on time (O(N) vs O(N log N)) and is the default. But choose sorting when: memory is tight (in-place sort → O(1) extra); you also need the elements ordered afterward (e.g., output sorted pairs); or you need a hard worst-case bound (hashing can degrade under collisions/adversarial input). For a *sorted* input you're often better off with [two pointers](#/coding/two-pointers-sliding-window) — O(1) space and no hashing at all. State which resource (time vs space vs order) you're optimizing.
</div></details>

<details class="qa"><summary>Why does prefix-sum hashing beat sliding window for "sum equals k"?</summary>
<div class="qa-body">

**Short:** It handles negative numbers, which break the monotonicity sliding window depends on.

**Deep:** Sliding window can shrink the left edge only if adding elements moves the sum monotonically (true when all values are non-negative). With negatives, extending the window can decrease the sum, so a shrunk window might still be improvable and the two-pointer logic is unsound. Prefix-sum + hash counts every qualifying window directly via `S[j] − S[i] = k` without assuming monotonicity — it's O(N) regardless of sign. If the problem guarantees positives *and* asks for a longest/shortest window (not a count), sliding window is the leaner O(1)-space choice.
</div></details>

<details class="qa"><summary>Follow-ups</summary>
<div class="qa-body">

- **"Top-k from a stream that doesn't fit in memory?"** → count-min sketch + a size-k [heap](#/coding/heap-priority-queue).
- **"Two Sum where the array is sorted?"** → drop the hash, use O(1)-space two pointers.
- **"Design a cache with frequency-based eviction (LFU)?"** → hash map + hash of frequency→ordered list; O(1) with a min-frequency pointer.
</div></details>

## Cheat-sheet

| Problem | What to hash | Time | Space |
| --- | --- | --- | --- |
| Two Sum | value → index | O(N) | O(N) |
| Contains Duplicate | set of seen | O(N) | O(N) |
| Subarray sum = k | prefix sum → count | O(N) | O(N) |
| Top-k frequent | value → count + buckets | O(N) | O(N) |
| Longest consecutive | set + start-only walk | O(N) | O(N) |
| Group anagrams | count-vector tuple → list | O(N·K) | O(N·K) |
| LRU cache | hash + OrderedDict/DLL | O(1)/op | O(cap) |

**Related:** [Arrays & Strings](#/coding/arrays-strings) · [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) · [Heaps & Priority Queues](#/coding/heap-priority-queue) · [Patterns hub](#/coding/patterns)
