# Binary Search

> [!TIP] 이 말부터 시작하세요
> "Binary search는 정렬된 배열에 관한 게 아니라 **monotonic predicate**에 관한 겁니다. 탐색 공간 전체에서 정확히 한 번 false에서 true로 뒤집히는 boolean `feasible(x)`를 정의할 수 있다면, 그 경계를 `O(log N)`에 찾을 수 있습니다." 이 한 문장이 LeetCode 704가 아니라 일반적인 패턴을 이해하고 있다는 신호를 줍니다.

매 단계마다 탐색 공간을 절반으로 줄입니다. 배열 버전은 사소하고, 면접에서의 가치는 *숨겨진* 탐색 공간을 알아보는 것 — **답에 대한 binary search**(일명 parametric search) — 그리고 무한 루프 없이 경계 조건을 정확히 맞추는 데 있습니다.

## 언제 꺼내 쓰나

<div class="proscons"><div><div class="pros-t">신호</div>

- 입력이 **정렬되어** 있음 (혹은 rotated-sorted) → 탐색 / 삽입 위치 / 경계.
- "**X가 ...를 만족하는 최소값**" 또는 "...를 만족하는 최대 X", 여기서 고정된 `X`를 확인하는 건 쉽지만 탐색은 어려운 경우 (용량, 속도, 거리, 시간).
- predicate가 **monotonic**함: 일단 feasible하면, 더 큰 (혹은 더 작은) `X`에 대해서도 항상 feasible.
- `O(log N)`이 필요하고 linear scan은 너무 느림.

</div><div><div class="cons-t">이건 아님</div>

- 정렬되지 않았고 monotone predicate도 없음 → 먼저 정렬하거나 (`log` 비용) hash/heap 사용.
- "답 확인" 자체가 비싸고 *동시에* non-monotonic → binary search 적용 불가.

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
<figcaption>모든 답-탐색형 binary search 문제는 이 경계를 찾는 것으로 귀결됩니다.</figcaption>
</figure>

## 외워둘 가치가 있는 두 가지 템플릿

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

# 2. Boundary / lower_bound on an inclusive integer domain [lo, hi].
#    Contract: pred is monotone False→True and pred(hi) is guaranteed True.
def lower_bound(lo: int, hi: int, pred) -> int:
    if lo > hi or not pred(hi):
        raise ValueError("need a non-empty domain with a feasible hi sentinel")
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if pred(mid):
            hi = mid          # keep mid: it might be the answer
        else:
            lo = mid + 1      # discard mid: definitely not the answer
    return lo                 # == hi; smallest feasible x
```

배열의 `bisect_left`처럼 "True가 없으면 끝 index"를 반환하고 싶다면 half-open domain `[0,n)`에 sentinel `n`을 두고, loop 안에서는 `pred(n)`을 호출하지 않는 별도 템플릿을 쓰세요. **inclusive answer search와 half-open insertion search의 계약을 섞지 않는 것**이 off-by-one 방지의 핵심입니다.

> [!WARNING] 버그는 단 두 가지
> **무한 루프:** `while lo < hi`에서 `lo = mid` (`mid + 1`이 아니라)로 하면 `hi = lo + 1`일 때 절대 전진하지 않습니다. 규칙: `mid`를 *유지하는* 쪽이 그쪽으로 줄어드는 쪽(`hi = mid`)이어야 하고, 다른 쪽은 하나를 더합니다(`lo = mid + 1`). **Off-by-one 경계:** predicate에서 `<` vs `<=`를 결정해 lower bound와 upper bound를 고르고, 원소가 두 개인 경우를 손으로 직접 테스트하세요.

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임(~10 MB)을 내려받고, 이후 실행은 즉시입니다. 본인 에디터가 편하면 각 문제의 **LeetCode** 링크로 이동하세요.

순서대로 진행하세요 — 배열 탐색을 먼저, 그다음 답에 대한 binary search, 마지막으로 Hard 파티션 문제입니다.

### 1. Search Insert Position <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/search-insert-position/)
`target`이 들어갈 인덱스를 반환합니다. 이건 `pred = nums[mid] >= target`인 `lower_bound` *그 자체*입니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"search_insert","starter":"def search_insert(nums: list[int], target: int) -> int:\n    # lower_bound: hi starts at len(nums); pred is nums[mid] >= target\n    pass","tests":[{"args":[[1,3,5,6],5],"expect":2},{"args":[[1,3,5,6],2],"expect":1},{"args":[[1,3,5,6],7],"expect":4},{"args":[[1,3,5,6],0],"expect":0},{"args":[[1],0],"expect":0}],"solution":"def search_insert(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums)\n    while lo < hi:\n        mid = lo + (hi - lo) // 2\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo"}
</script>
</div>

`O(log N)` 시간, `O(1)` 공간. `<`를 `<=`로 바꾸면 **upper bound**가 됩니다 — 이 한 쌍이 `bisect_left`/`bisect_right`를 커버합니다.

### 2. Search in Rotated Sorted Array <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/search-in-rotated-sorted-array/)
중복 없음. 각 단계에서 한쪽 절반은 정렬되어 있으니, `target`이 그 안에 있는지 판단합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"search_rotated","starter":"def search_rotated(nums: list[int], target: int) -> int:\n    # one half is always sorted; decide whether target lies inside it\n    pass","tests":[{"args":[[4,5,6,7,0,1,2],0],"expect":4},{"args":[[4,5,6,7,0,1,2],3],"expect":-1},{"args":[[4,5,6,7,0,1,2],7],"expect":3},{"args":[[5,1,3],5],"expect":0},{"args":[[1],0],"expect":-1}],"solution":"def search_rotated(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = lo + (hi - lo) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[lo] <= nums[mid]:\n            if nums[lo] <= target < nums[mid]:\n                hi = mid - 1\n            else:\n                lo = mid + 1\n        else:\n            if nums[mid] < target <= nums[hi]:\n                lo = mid + 1\n            else:\n                hi = mid - 1\n    return -1"}
</script>
</div>

`O(log N)`. 중복이 있으면 (LC 81) `nums[lo] == nums[mid]` 동점 때문에 linear한 `lo += 1` 스텝이 강제되어 최악의 경우 `O(N)`으로 나빠집니다 — 이 점을 소리 내어 말하세요.

### 3. Koko Eating Bananas <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/koko-eating-bananas/)
`piles`를 `h`시간 안에 다 먹기 위한 최소 먹는 속도 `k`. "속도 `k`로 다 먹을 수 있다"는 predicate는 `k`에 대해 monotonic합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"min_eating_speed","starter":"import math\n\ndef min_eating_speed(piles: list[int], h: int) -> int:\n    # binary-search k in [1, max(piles)]; feasible if sum(ceil(p/k)) <= h\n    pass","tests":[{"args":[[3,6,7,11],8],"expect":4},{"args":[[30,11,23,4,20],5],"expect":30},{"args":[[30,11,23,4,20],6],"expect":23},{"args":[[1,1,1,999999999],10],"expect":142857143}],"solution":"import math\n\ndef min_eating_speed(piles: list[int], h: int) -> int:\n    def can_finish(speed: int) -> bool:\n        return sum(math.ceil(p / speed) for p in piles) <= h\n    lo, hi = 1, max(piles)\n    while lo < hi:\n        mid = lo + (hi - lo) // 2\n        if can_finish(mid):\n            hi = mid\n        else:\n            lo = mid + 1\n    return lo"}
</script>
</div>

`M = max(piles)`일 때 `O(N log M)`. 같은 뼈대가 *Capacity to Ship Packages* (LC 1011), *Split Array Largest Sum* (LC 410), *Minimize Max Distance*를 풉니다 — `can_finish`만 바뀔 뿐입니다.

### 4. Find Peak Element <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/find-peak-element/)
양쪽 이웃보다 큰 임의의 인덱스를, `O(log N)`에. 오르막으로 걸어 올라가면 됩니다: 더 큰 이웃 쪽에는 항상 peak이 있습니다 (범위 밖은 `-∞`로 취급).

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"find_peak_element","starter":"def find_peak_element(nums: list[int]) -> int:\n    # climb toward the larger neighbor; out-of-bounds counts as -inf\n    pass","tests":[{"args":[[1,2,3,1]],"expect":2},{"args":[[1,2,1,3,5,6,4]],"expect":5},{"args":[[1]],"expect":0},{"args":[[1,2]],"expect":1},{"args":[[2,1]],"expect":0}],"solution":"def find_peak_element(nums: list[int]) -> int:\n    lo, hi = 0, len(nums) - 1\n    while lo < hi:\n        mid = lo + (hi - lo) // 2\n        if nums[mid] < nums[mid + 1]:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo"}
</script>
</div>

### 5. Median of Two Sorted Arrays <span class="badge badge-hard">Hard</span> · [LeetCode ↗](https://leetcode.com/problems/median-of-two-sorted-arrays/)
`left_max ≤ right_min`이 되도록 *더 짧은* 배열의 분할을 binary search합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"find_median_sorted_arrays","starter":"def find_median_sorted_arrays(a: list[int], b: list[int]) -> float:\n    # binary-search the partition of the shorter array so left_max <= right_min\n    pass","tests":[{"args":[[1,3],[2]],"expect":2.0},{"args":[[1,2],[3,4]],"expect":2.5},{"args":[[],[1]],"expect":1.0},{"args":[[1,2],[]],"expect":1.5},{"args":[[1,2,3,4,5],[6,7,8]],"expect":4.5}],"solution":"def find_median_sorted_arrays(a: list[int], b: list[int]) -> float:\n    if len(a) > len(b):\n        a, b = b, a\n    m, n = len(a), len(b)\n    half = (m + n + 1) // 2\n    lo, hi = 0, m\n    while lo <= hi:\n        i = (lo + hi) // 2\n        j = half - i\n        l1 = a[i - 1] if i > 0 else float(\"-inf\")\n        r1 = a[i] if i < m else float(\"inf\")\n        l2 = b[j - 1] if j > 0 else float(\"-inf\")\n        r2 = b[j] if j < n else float(\"inf\")\n        if l1 <= r2 and l2 <= r1:\n            if (m + n) % 2:\n                return float(max(l1, l2))\n            return (max(l1, l2) + min(r1, r2)) / 2\n        if l1 > r2:\n            hi = i - 1\n        else:\n            lo = i + 1\n    raise ValueError(\"inputs not sorted\")"}
</script>
</div>

`O(log(min(m, n)))`. `±inf` sentinel이 모든 경계 특수 케이스 처리를 없애줍니다 — 이 트릭이 사실상 전부입니다.

## 묻기 전에 먼저 언급해야 할 변형

| 변형 | 반전 | 핵심 아이디어 |
| --- | --- | --- |
| Real-valued search | 정수 대신 tolerance `ε` | `while hi - lo > eps` (혹은 고정 ~100회 반복) |
| Search 2D matrix | LC 240와 LC 74의 보장이 다름 | LC 240: 각 행·열 정렬 → 우상단 계단식 `O(m+n)`. LC 74: 행 경계까지 전역 정렬 → 평탄화해 `O(log mn)` |
| First/last occurrence | 중복 | lower_bound와 upper_bound 실행 |
| Rotated min (LC 153) | pivot 찾기 | `nums[mid]`를 `nums[hi]`와 비교 |
| Peak / valley | local extremum | 더 높은 이웃 쪽으로 이동 |

## 함정

- **`bisect`가 이미 있음.** 표준 라이브러리의 `bisect_left`/`bisect_right`는 프로덕션에서 검증된 정확한 코드입니다 — 면접관이 루프를 원하지 않는 한 이걸 꺼내 쓰세요. 둘 다 알아두세요.
- **잘못된 초기 `hi`.** 인덱스 탐색은 `len(nums)-1`을, 삽입/경계 탐색은 `len(nums)`를 씁니다.
- 답-탐색에서 **정수가 아닌 midpoint**: 공간을 정수로 유지하거나, float에는 `ε`/고정-반복 형태를 쓰세요.
- **predicate가 사실 monotonic이 아님.** `can_finish(k)`가 monotone이 아니면 경계가 정의되지 않습니다 — 문제를 다시 프레이밍하세요.

## Q&A

<details class="qa"><summary>고전적인 off-by-one / 무한 루프를 어떻게 피하나요?</summary>
<div class="qa-body">

**짧게:** 하나의 템플릿을 고르고 불변식(invariant)을 유지합니다. 저는 답이 항상 `[lo, hi]`에 있는 `while lo < hi`를 씁니다. 답이 *될 수 있는* 가지는 `hi = mid`로, 다른 가지는 `lo = mid + 1`로 설정합니다. 이렇게 하면 엄격한 축소가 보장됩니다.

**깊게:** 실패 모드는 `mid = (lo+hi)//2`에서의 `lo = mid`입니다 — `hi = lo+1`일 때 `mid == lo`이므로 `lo = mid`는 no-op이 되어 무한 루프가 됩니다. `lo = mid` 가지에 대해서는 midpoint를 위로 편향시키거나(`mid = (lo+hi+1)//2`), 유지되는 쪽이 `hi`가 되도록 재구성합니다. 실행 전에 원소가 두 개인 배열에서 검증합니다.
</div></details>

<details class="qa"><summary>언제 배열이 아니라 답에 대해 binary search를 하나요?</summary>
<div class="qa-body">

**짧게:** 답이 알려진 범위 안의 숫자이고 값싼 monotone feasibility 체크가 있을 때입니다 — 배열 자체는 정렬되지 않았거나 무관할 수 있습니다.

**깊게:** 신호가 되는 문구: "maximum을 minimize," "minimum을 maximize," "동작하는 최소 capacity/속도/시간." 저는 `feasible(x)`를 정의하고, monotonicity를 논증하고(`x`가 되면 `x+1`은 자명하게 됨), 범위 `[lo, hi]`를 잡고, `lower_bound`를 실행합니다. 비용은 `O(log(range) · cost(feasible))`입니다. 이것이 704를 외운 사람과 binary search를 이해한 사람을 가르는 패턴입니다.
</div></details>

**예상되는 후속 질문**
- "이제 배열에 중복이 있습니다." → rotated 탐색이 `O(N)`으로 나빠짐; 정렬된-절반 테스트가 왜 깨지는지 설명.
- "라이브러리 없이 해보세요." → `lower_bound`를 처음부터 작성.
- "feasibility 체크가 `O(N)`입니다 — 전체 복잡도는?" → `O(N log(range))`.
- "정수 대신 float라면?" → 보장된 정밀도를 위한 고정 반복 횟수.

## Cheat-sheet

| 사실 | 세부 |
| --- | --- |
| 핵심 불변식 | 답은 `[lo, hi]`에 유지; 매 반복마다 엄격히 축소 |
| Overflow-safe mid | `mid = lo + (hi - lo) // 2` |
| Keep-mid 규칙 | 답이 될 수 있는 가지는 `hi = mid`; 다른 쪽은 `lo = mid + 1` |
| Lower vs upper bound | predicate에서 `>=` vs `>` (`bisect_left` vs `bisect_right`) |
| Index search | `hi = len-1`, 루프 `lo <= hi` |
| Boundary search | `hi = len`, 루프 `lo < hi` |
| Answer search | monotone `feasible(x)` 정의, 비용 `O(log range · check)` |
| Float search | `hi - lo > eps`로 루프하거나 ~100회로 고정 |
| 복잡도 | `O(log N)` 시간, `O(1)` 공간 |

**관련:** [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) · [Trees & BSTs](#/coding/trees-bst) · [Greedy & Intervals](#/coding/greedy-intervals) · 다시 [The Core Patterns](#/coding/patterns)와 [Coding Round Strategy](#/coding/strategy)로.
