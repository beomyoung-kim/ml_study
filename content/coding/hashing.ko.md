# Hashing

> [!TIP] The cue
> brute force가 loop 안에서 **membership, complement, frequency, "이거 전에 본 적 있나?"** 조회를 반복할 때마다, hash map/set은 각각에 **average O(1)**을 사주고 nested loop를 O(N)으로 무너뜨립니다. 뭔가를 *찾기* 위해 inner loop를 쓰고 있다는 걸 알아차리면, hash를 꺼내세요.

거래는 time을 space와 바꾸는 것입니다: 본 것을 기억하는 데 O(N) 메모리를 씁니다 — 값, 그 index, 그 count, 또는 prefix aggregate. 고전적인 업그레이드는 **complement lookup**(Two Sum), **prefix-sum + hash**(음수가 있어도 subarray count), **canonical-key grouping**(anagram — [Arrays & Strings](#/coding/arrays-strings) 참고)입니다.

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

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임(~10 MB)을 내려받고, 이후 실행은 즉시입니다. 본인 에디터가 편하면 각 문제의 **LeetCode** 링크로 이동하세요.

### 1. Two Sum <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/two-sum/)

`target`에 합해지는 두 수의 index.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"two_sum","starter":"def two_sum(nums: list[int], target: int) -> list[int]:\n    # store value -> index; look up the complement before inserting\n    pass","tests":[{"args":[[2,7,11,15],9],"expect":[0,1]},{"args":[[3,2,4],6],"expect":[1,2]},{"args":[[3,3],6],"expect":[0,1]},{"args":[[-3,4,3,90],0],"expect":[0,2]}],"solution":"def two_sum(nums: list[int], target: int) -> list[int]:\n    index_of = {}\n    for i, value in enumerate(nums):\n        if target - value in index_of:\n            return [index_of[target - value], i]\n        index_of[value] = i\n    return []"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** complement 확인 *뒤에* insert하세요, 안 그러면 한 원소를 자기 자신과 짝지을 수 있습니다. (array가 sorted라면 O(1)-space [two-pointer](#/coding/two-pointers-sliding-window) 버전을 선호하세요.)

### 2. Contains Duplicate <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/contains-duplicate/)

값이 하나라도 반복되면 `True`를 반환하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"contains_duplicate","starter":"def contains_duplicate(nums: list[int]) -> bool:\n    # track seen values in a set; True on the first repeat\n    pass","tests":[{"args":[[1,2,3,1]],"expect":true},{"args":[[1,2,3,4]],"expect":false},{"args":[[1,1,1,3,3,4,3,2,4,2]],"expect":true},{"args":[[]],"expect":false}],"solution":"def contains_duplicate(nums: list[int]) -> bool:\n    seen = set()\n    for x in nums:\n        if x in seen:\n            return True\n        seen.add(x)\n    return False"}
</script>
</div>

*O(N) time, O(N) space.* Sorting + 인접 비교는 O(N log N)에 O(1) extra이지만 순서를 파괴합니다 — 둘 다와 trade-off를 언급하세요.

### 3. Subarray Sum Equals K <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/subarray-sum-equals-k/)

`k`에 합해지는 contiguous subarray의 개수를 세세요(값은 음수일 수 있음).

**Approach.** prefix sum `S`에서, subarray `(i, j]`가 `k`에 합해지는 건 `S[j] − S[i] = k`, 즉 `S[i] = S[j] − k`일 때뿐입니다. 스캔하며, `running − k`와 같은 이전 prefix sum이 몇 개였는지 세세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"subarray_sum","starter":"from collections import defaultdict\n\ndef subarray_sum(nums: list[int], k: int) -> int:\n    # prefix sum + hash; count earlier prefixes equal to running - k\n    pass","tests":[{"args":[[1,1,1],2],"expect":2},{"args":[[1,2,3],3],"expect":2},{"args":[[1,-1,0],0],"expect":3},{"args":[[3,4,7,2,-3,1,4,2],7],"expect":4}],"solution":"from collections import defaultdict\n\ndef subarray_sum(nums: list[int], k: int) -> int:\n    count = defaultdict(int)\n    count[0] = 1\n    running = answer = 0\n    for x in nums:\n        running += x\n        answer += count[running - k]\n        count[running] += 1\n    return answer"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** `count[0] = 1` seed — 이게 없으면 index 0에서 시작하는 subarray를 놓칩니다. 여기서는 음수가 monotonicity를 깨므로 sliding window를 쓸 수 없습니다; hashing이 필수입니다.

### 4. Top K Frequent Elements <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/top-k-frequent-elements/)

가장 빈도 높은 `k`개의 값(순서 무관).

**Approach.** `Counter`로 세고, 그다음 **frequency로 bucket sort**하여 O(N). size-k [heap](#/coding/heap-priority-queue)이 O(N log k) 대안입니다 — streaming에 친화적인 옵션으로 언급하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"top_k_frequent","starter":"from collections import Counter\n\ndef top_k_frequent(nums: list[int], k: int) -> list[int]:\n    # count, then bucket-sort by frequency; collect the top k\n    pass","tests":[{"args":[[1,1,1,2,2,3],2],"expect":[1,2],"unordered":true},{"args":[[1],1],"expect":[1],"unordered":true},{"args":[[4,1,-1,2,-1,2,3],2],"expect":[-1,2],"unordered":true}],"solution":"from collections import Counter\n\ndef top_k_frequent(nums: list[int], k: int) -> list[int]:\n    freq = Counter(nums)\n    buckets = [[] for _ in range(len(nums) + 1)]\n    for val, c in freq.items():\n        buckets[c].append(val)\n    out = []\n    for c in range(len(buckets) - 1, 0, -1):\n        for val in buckets[c]:\n            out.append(val)\n            if len(out) == k:\n                return out\n    return out"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** `sorted(freq, key=freq.get)`은 쉬운 O(N log N) 답입니다 — 막히면 쓰되, bucket/heap 개선을 자진해서 제시하세요.

### 5. Longest Consecutive Sequence <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/longest-consecutive-sequence/)

연속 정수의 가장 긴 run의 길이를, O(N)로, unsorted에서.

**Approach.** 모든 걸 set에 넣고, 왼쪽 이웃이 없는(`num − 1`이 부재) 값에서만 세기 시작하여, 각 run을 정확히 한 번만 걷게 합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"longest_consecutive","starter":"def longest_consecutive(nums: list[int]) -> int:\n    # put all in a set; only count runs from a value with no left neighbor\n    pass","tests":[{"args":[[100,4,200,1,3,2]],"expect":4},{"args":[[0,3,7,2,5,8,4,6,0,1]],"expect":9},{"args":[[]],"expect":0},{"args":[[1,2,0,1]],"expect":3}],"solution":"def longest_consecutive(nums: list[int]) -> int:\n    num_set = set(nums)\n    best = 0\n    for num in num_set:\n        if num - 1 in num_set:\n            continue\n        length, cur = 1, num\n        while cur + 1 in num_set:\n            cur, length = cur + 1, length + 1\n        best = max(best, length)\n    return best"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** *모든* 원소에서 inner `while`을 돌리면 O(N²)입니다; "start-only" guard가 이것을 선형으로 유지하는 핵심입니다.

### 6. LRU Cache <span class="badge badge-hard">Hard</span>

O(1) `get`과 `put`을 갖는 고정 용량 cache.

**Approach.** O(1) lookup을 위한 hash map + O(1) recency 갱신을 위한 ordering 구조. Python의 `OrderedDict`가 둘 다 줍니다; "from scratch" 버전은 hash map + doubly linked list입니다.

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

*op당 average O(1), O(capacity) space.* **Pitfall:** 순수 `dict`는 O(1)로 재정렬할 수 없습니다; `OrderedDict`를 쓰고, stdlib를 피하라고 하면 hash-map-plus-DLL 구현("tail에서 evict, access 시 promote")을 설명할 준비를 하세요.

## Common variations

- **Prefix-sum + hash 계열**: "subarray sum divisible by k"(`running % k` 저장), "0과 1이 같은 개수인 contiguous array"(0→−1 매핑, 각 prefix sum의 첫 index 저장), "longest subarray sum = k".
- **Two-pass complement**: 4Sum-count(두 array의 모든 pair sum을 hash하고, 나머지 두 array에서 음수를 조회)는 O(N⁴)를 O(N²)로 바꿉니다.
- **Design 문제**: LFU Cache, Insert/Delete/GetRandom O(1)(hash + swap-to-end array), Design HashMap.
- **Rolling hash**(Rabin–Karp): substring search / duplicate-substring detection을 hashing follow-up으로.
- **ML connection**: feature hashing("hashing trick"), embedding lookup table, 그리고 거의 동일한 sample의 dedup은 모두 hash-map lookup으로 환원됩니다 — 라포를 위해 한 줄 언급할 만합니다.

## Pitfalls

> [!WARNING] Hashing gotchas
> - hash 연산은 worst-case가 아니라 **"average O(1)"**이라고 말하세요 — adversarial key는 O(N)으로 나빠집니다.
> - key는 **hashable**해야 합니다: `list`가 아니라 `tuple`을 쓰세요(예: anagram용 frozen count vector).
> - `defaultdict`는 read 시 없는 key를 생성합니다 — 그게 조용히 map을 부풀릴 수 있습니다; insertion을 원치 않으면 `.get(k, default)`를 쓰세요.
> - 객체를 identity로 저장하느냐 value로 저장하느냐 — `__hash__`/`__eq__`를 일관되게 정의하거나, primitive signature를 key로 하세요.
> - hash set은 순서를 보존하지 않습니다(3.7 이전엔 `dict`도 안 했습니다); correctness를 위해 iteration order에 의존하지 마세요.

<details class="qa"><summary>"find pairs / dedup" 문제에서 sorting이 hashing보다 나은 건 언제인가요?</summary>
<div class="qa-body">

**Short:** O(1) *space*가 필요하거나, 부산물로서의 ordering이 필요하거나, worst-case 보장이 필요할 때 — sorting의 O(N log N)/O(1)이 hashing의 O(N)/O(N)을 이깁니다.

**Deep:** Hashing은 time에서 이기고(O(N) vs O(N log N)) 기본 선택입니다. 하지만 다음일 때 sorting을 고르세요: 메모리가 빠듯할 때(in-place sort → O(1) extra); 이후에 원소가 정렬되어 있어야 할 때(예: 정렬된 pair 출력); 또는 hard worst-case bound가 필요할 때(hashing은 collision/adversarial input에서 나빠질 수 있음). *sorted* input이라면 종종 [two pointers](#/coding/two-pointers-sliding-window)가 더 낫습니다 — O(1) space에 hashing이 전혀 없죠. 어떤 자원(time vs space vs order)을 최적화하는지 말하세요.
</div></details>

<details class="qa"><summary>"sum equals k"에서 왜 prefix-sum hashing이 sliding window를 이기나요?</summary>
<div class="qa-body">

**Short:** sliding window가 의존하는 monotonicity를 깨는 음수를 처리하기 때문입니다.

**Deep:** sliding window는 원소를 더하는 게 sum을 monotone하게 움직일 때에만(모든 값이 non-negative일 때 참) 왼쪽 끝을 shrink할 수 있습니다. 음수가 있으면 window를 확장하는 게 sum을 감소시킬 수 있어, shrink한 window도 여전히 개선 가능할 수 있고 two-pointer 논리가 부적절해집니다. Prefix-sum + hash는 monotonicity를 가정하지 않고 `S[j] − S[i] = k`로 조건을 만족하는 모든 window를 직접 셉니다 — 부호와 무관하게 O(N)입니다. 문제가 양수를 보장하고 *또한* (count가 아니라) longest/shortest window를 요구하면, sliding window가 더 가벼운 O(1)-space 선택입니다.
</div></details>

<details class="qa"><summary>Follow-ups</summary>
<div class="qa-body">

- **"메모리에 안 들어가는 stream에서 top-k?"** → count-min sketch + size-k [heap](#/coding/heap-priority-queue).
- **"array가 sorted인 Two Sum?"** → hash를 버리고, O(1)-space two pointers를 쓰세요.
- **"frequency 기반 eviction(LFU) cache를 설계?"** → hash map + frequency→ordered list의 hash; min-frequency pointer로 O(1).
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
