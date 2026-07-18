# Arrays & Strings

> [!TIP] The cue
> Array/string 문제는 면접의 워밍업이자 다른 모든 pattern의 바탕입니다. 첫 1분에 내려야 할 결정: **단 한 번의 linear scan이면 충분한가, sort가 필요한가, 아니면 helper 구조(hash map / count array)가 필요한가?** 무엇을 왜 쓰는지 타이핑 전에 말하세요.

대부분의 "array" 문제는 몇 가지 동작 중 하나로 환원됩니다: running state를 들고 하는 single pass, prefix/suffix aggregate, in-place two-pointer rewrite, 또는 reversal 트릭. 핵심 난점이 *frequency*나 *lookup*이면 [Hashing](#/coding/hashing)으로 가고, *contiguous window*나 *양 끝 수렴*이면 [Two Pointers](#/coding/two-pointers-sliding-window)로 가세요.

## When to use which move

| Cue in the statement | Move | Complexity |
| --- | --- | --- |
| "in-place", "O(1) extra space", overwrite | Two-index write / swap | O(N) time, O(1) space |
| "product/sum except self", running aggregate | Prefix + suffix pass | O(N) time, O(1) extra |
| "reverse / rotate" | Reversal trick (3 reverses) | O(N) time, O(1) space |
| "max profit / best so far" | Single pass with running extremum | O(N) time, O(1) space |
| group by signature (anagrams) | Hash on a canonical key | O(N·K) |
| "trap water / two walls" | Two-pointer with running maxes | O(N) time, O(1) space |

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임(~10 MB)을 내려받고, 이후 실행은 즉시입니다. 본인 에디터가 편하면 각 문제의 **LeetCode** 링크로 이동하세요.

### 1. Best Time to Buy and Sell Stock <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

일별 `prices`가 주어지면, 한 번 사고 나중에 한 번 파는 것으로 profit을 최대화하세요; 없으면 0.

**Approach.** brute force는 모든 (buy, sell) 쌍을 확인 → O(N²). 최적화: 왼쪽에서 오른쪽으로 스캔하며 **지금까지 본 최소 price**를 유지하고, best profit을 `price − min_so_far`로 갱신합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"max_profit","starter":"def max_profit(prices: list[int]) -> int:\n    # single pass; track the min price so far, best = price - min\n    pass","tests":[{"args":[[7,1,5,3,6,4]],"expect":5},{"args":[[7,6,4,3,1]],"expect":0},{"args":[[1,2,3,4,5]],"expect":4},{"args":[[2,4,1]],"expect":2},{"args":[[5]],"expect":0}],"solution":"def max_profit(prices: list[int]) -> int:\n    if not prices:\n        return 0\n    min_so_far = prices[0]\n    best = 0\n    for price in prices[1:]:\n        best = max(best, price - min_so_far)\n        min_so_far = min(min_so_far, price)\n    return best"}
</script>
</div>

*O(N) time, O(1) space.* **Pitfall:** min은 profit을 계산한 *뒤에* 갱신하세요, 안 그러면 같은 날 buy/sell을 허용하게 됩니다.

### 2. Product of Array Except Self <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/product-of-array-except-self/)

`out[i] = nums[i]를 제외한 모든 원소의 곱`을 반환하세요, 나눗셈 없이, O(N²)보다 나은 방법으로.

**Approach.** `out[i] = (i 왼쪽 전부의 곱) × (i 오른쪽 전부의 곱)`. 왼쪽에서 오른쪽으로 한 번 pass하여 left product를 채우고, 오른쪽에서 왼쪽으로 한 번 pass하며 right product를 곱해 넣습니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"product_except_self","starter":"def product_except_self(nums: list[int]) -> list[int]:\n    # prefix pass fills left products; suffix pass multiplies in right products\n    pass","tests":[{"args":[[1,2,3,4]],"expect":[24,12,8,6]},{"args":[[-1,1,0,-3,3]],"expect":[0,0,9,0,0]},{"args":[[2,3]],"expect":[3,2]},{"args":[[0,0]],"expect":[0,0]}],"solution":"def product_except_self(nums: list[int]) -> list[int]:\n    n = len(nums)\n    out = [1] * n\n    left = 1\n    for i in range(n):\n        out[i] = left\n        left *= nums[i]\n    right = 1\n    for i in range(n - 1, -1, -1):\n        out[i] *= right\n        right *= nums[i]\n    return out"}
</script>
</div>

*O(N) time, O(1) extra space* (output array 제외). **Pitfall:** 나눗셈 방식(`total // nums[i]`)은 zero가 하나라도 있으면 깨집니다 — prefix/suffix 방법은 zero가 몇 개든 안전하게 처리합니다.

### 3. Rotate Array <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/rotate-array/)

`nums`를 오른쪽으로 `k`만큼 회전하세요, in place로 O(1) extra space.

**Approach.** **triple reversal** 트릭: 전체를 reverse, 앞의 `k`개를 reverse, 나머지를 reverse.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"rotate","starter":"def rotate(nums: list[int], k: int) -> list[int]:\n    # triple reversal in place (k %= n first); return the rotated list\n    pass","tests":[{"args":[[1,2,3,4,5,6,7],3],"expect":[5,6,7,1,2,3,4]},{"args":[[-1,-100,3,99],2],"expect":[3,99,-1,-100]},{"args":[[1,2],3],"expect":[2,1]},{"args":[[1],0],"expect":[1]}],"solution":"def rotate(nums: list[int], k: int) -> list[int]:\n    n = len(nums)\n    if n == 0:\n        return nums\n    k %= n\n    def reverse(lo: int, hi: int) -> None:\n        while lo < hi:\n            nums[lo], nums[hi] = nums[hi], nums[lo]\n            lo, hi = lo + 1, hi - 1\n    reverse(0, n - 1)\n    reverse(0, k - 1)\n    reverse(k, n - 1)\n    return nums"}
</script>
</div>

*O(N) time, O(1) space.* **Pitfall:** `k %= n`을 잊기(`k > n`일 때 index error); 세 reversal 범위를 헷갈리기 — `[1,2,3,4,5], k=2`로 손으로 확인하세요.

### 4. Group Anagrams <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/group-anagrams/)

서로 anagram인 단어들을 묶으세요.

**Approach.** 두 단어는 canonical signature를 공유할 때에만 anagram입니다. 글자를 sort하면 단어당 O(K log K); **길이 26짜리 count vector**(hashable tuple로)는 O(K). 면접에서는 sort를 먼저 언급하고, 그다음 count vector로 최적화하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"group_anagrams","starter":"from collections import defaultdict\n\ndef group_anagrams(strs: list[str]) -> list[list[str]]:\n    # hash each word by its 26-length count vector (as a tuple)\n    pass","tests":[{"args":[["eat","tea","tan","ate","nat","bat"]],"expect":[["eat","tea","ate"],["tan","nat"],["bat"]],"unordered":true},{"args":[[""]],"expect":[[""]],"unordered":true},{"args":[["a"]],"expect":[["a"]],"unordered":true}],"solution":"from collections import defaultdict\n\ndef group_anagrams(strs: list[str]) -> list[list[str]]:\n    groups = defaultdict(list)\n    for s in strs:\n        count = [0] * 26\n        for ch in s:\n            count[ord(ch) - ord('a')] += 1\n        groups[tuple(count)].append(s)\n    return list(groups.values())"}
</script>
</div>

*O(N·K) time* (N 단어, K 평균 길이), *O(N·K) space.* 이건 사실 array의 옷을 입은 [Hashing](#/coding/hashing) 문제입니다.

### 5. Trapping Rain Water <span class="badge badge-hard">Hard</span> · [LeetCode ↗](https://leetcode.com/problems/trapping-rain-water/)

주어진 `height`들의 막대 사이에 갇히는 총 물의 양.

**Approach.** index `i` 위의 물은 `min(max_left, max_right) − height[i]`입니다. brute force는 max를 매번 재계산 → O(N²). Prefix/suffix-max array는 O(N) time, O(N) space. **최적: two pointers**, O(1) space — running max가 더 작은 쪽을 이동하세요, 그 쪽의 물 높이가 그 시점에 완전히 결정되기 때문입니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"trap","starter":"def trap(height: list[int]) -> int:\n    # two pointers; move the side with the smaller running max\n    pass","tests":[{"args":[[0,1,0,2,1,0,1,3,2,1,2,1]],"expect":6},{"args":[[4,2,0,3,2,5]],"expect":9},{"args":[[]],"expect":0},{"args":[[1,2,3]],"expect":0}],"solution":"def trap(height: list[int]) -> int:\n    if not height:\n        return 0\n    left, right = 0, len(height) - 1\n    left_max, right_max = height[left], height[right]\n    water = 0\n    while left < right:\n        if left_max < right_max:\n            left += 1\n            left_max = max(left_max, height[left])\n            water += left_max - height[left]\n        else:\n            right -= 1\n            right_max = max(right_max, height[right])\n            water += right_max - height[right]\n    return water"}
</script>
</div>

*O(N) time, O(1) space.* **Pitfall:** 왜 max가 더 작은 쪽을 확정해도 안전한지 *설명*할 준비를 하세요 — 면접관은 항상 follow-up합니다. prefix/suffix 버전을 먼저 연습하고, 그다음 space 최적인 two-pointer를 하세요.

## Common variations

- **In-place 제약이 추가됨** (Remove Duplicates, Move Zeroes): slow write-index와 fast read-index를 유지하고, 앞에서부터 overwrite하세요. 이건 [two-pointer](#/coding/two-pointers-sliding-window) in-place 관용구입니다.
- **String parsing/formatting** (`atoi`, run-length compression): 코드는 쉽고, 점수는 빠짐없는 edge case에 있습니다 — overflow, leading/trailing space, sign, empty input. 소리 내어 먼저 나열하세요.
- **Matrix as array** (spiral order, rotate image, set-matrix-zeroes): 2-D에서 같은 동작; boundary index와 in place로 변형 가능한지를 주의하세요.
- **ML dressing**: "1-D signal에 대한 sliding max/mean (예: frame별 logits)"은 시계열 옷을 입은 array 문제입니다 — pattern은 그대로입니다.

## Pitfalls

> [!WARNING] 흔한 array 버그
> - 마지막 index에서의 off-by-one(`range(n)` vs `range(n-1)`), 특히 reversal 범위에서.
> - iterate 중에 list를 mutate하기 — 복사본을 iterate하거나 index를 쓰세요.
> - 순서가 양방향으로 중요할 때 single pass면 충분하다고 가정하기(prefix *와* suffix 둘 다).
> - loop 안에서 `list.pop(0)` / `insert(0, x)` 쓰기 — 각각 O(N)입니다; [`deque`](#/coding/stack-queue)를 쓰세요.
> - Slicing(`nums[1:]`)은 복사합니다 — readability엔 괜찮지만, space가 채점되면 "O(N) extra"라고 말하세요.

<details class="qa"><summary>Product Except Self — 나눗셈이 <em>허용된다면</em> 어떻게 처리하겠어요?</summary>
<div class="qa-body">

**Short:** 전체 곱을 계산하고 나누되, zero를 특수 처리하세요.

**Deep:** zero가 없으면 `out[i] = total / nums[i]`가 O(N). zero가 정확히 하나면, 모든 `out[i]`는 0이고 zero의 index만 non-zero 원소들의 곱을 얻습니다. zero가 둘 이상이면 모든 항목이 0입니다. prefix/suffix 방법은 이 모든 분기를 피해가고 면접관이 선호하는 방식이므로, 이걸 앞세우고 나눗셈은 "array가 non-zero임이 보장될 때"의 단순화로만 제시하세요.
</div></details>

<details class="qa"><summary>Trapping Rain Water — 왜 max가 더 작은 쪽을 확정해도 안전한가요?</summary>
<div class="qa-body">

**Short:** `left_max < right_max`이면, pointer 사이에 무엇이 있든 `left`의 물은 `left_max`로 bound됩니다. 오른쪽에 적어도 `right_max`(> `left_max`)인 막대가 존재하기 때문입니다.

**Deep:** 임의의 index에서 갇히는 높이는 `min(max_left, max_right) − height[i]`입니다. max가 더 작은 쪽의 pointer를 이동할 때, 반대쪽에 최소한 그만큼 큰 max가 있음을 *알기* 때문에, 이동하는 쪽의 `min(...)`은 자기 자신의 running max와 같습니다 — 미래의 어떤 막대도 그것을 낮출 수 없습니다. 그 보장이 바로 far side의 정확한 profile을 보지 않고도 O(1) space에서 그 지점의 물을 확정할 수 있는 이유입니다.
</div></details>

<details class="qa"><summary>Follow-ups</summary>
<div class="qa-body">

- **"transaction fee가 있거나, 최대 k번 거래라면?"** → 이건 [DP](#/coding/dynamic-programming)가 됩니다(state = day × holding × 남은 transaction 수).
- **"2-D matrix를 90° 회전하려면?"** → transpose 후 각 row를 reverse, in place로.
- **"input이 stream일 때 anagram을 묶으려면?"** → count-vector signature를 key로 그룹을 점진적으로 emit/merge하세요.
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

\*output array 제외.

**Related:** [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) · [Hashing](#/coding/hashing) · [Greedy & Intervals](#/coding/greedy-intervals) · [Patterns hub](#/coding/patterns)
