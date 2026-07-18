# Two Pointers & Sliding Window

> [!TIP] The cue
> O(N²) 스캔을 O(N)으로 무너뜨리는, 밀접하게 관련된 두 pattern입니다. **Two pointers:** 양 끝에서 수렴하는 sorted array(pair, palindrome, container). **Sliding window:** 제약(max/min length, ≤ k distinct, sum ≥ target)을 유지하기 위해 window를 확장하고 축소하는 contiguous subarray/substring.

**two pointers**의 신호는 *sorted input + pairing 질문*입니다. **sliding window**의 신호는 *contiguous* + *window가 이동함에 따라 점진적으로 유지할 수 있는 제약*입니다. 세 번째 사촌 — **fast & slow pointers** — 은 cycle(linked-list loop, "find the duplicate number")을 다룹니다.

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

non-alphanumeric와 대소문자를 무시하고, `s`가 palindrome인가요?

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

*O(N) time, O(1) space.* **Pitfall:** 먼저 filter된 복사본을 만들면 O(N) space를 씁니다 — in-place two-pointer가 기대되는 답입니다.

### 2. Two Sum II — Sorted Input <span class="badge badge-easy">Easy</span>

sorted array에서 `target`에 합해지는 1-indexed 쌍을 반환하세요.

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

*O(N) time, O(1) space.* **Pitfall:** 여기서 hash-map [Two Sum](#/coding/hashing)을 쓰면 sorted 보장과 O(1) space를 낭비합니다 — 면접관은 일부러 sortedness를 준 겁니다.

### 3. Container With Most Water <span class="badge badge-med">Medium</span>

가장 많은 물을 담는 container를 이루는 두 line; 최대 area를 반환하세요.

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

*O(N) time, O(1) space.* **왜 더 짧은 쪽을 이동하나요?** pointer가 수렴하며 width는 줄기만 하고, area는 더 짧은 벽으로 bound됩니다. 더 큰 쪽을 이동해도 `min(...)`을 결코 늘릴 수 없으므로, 그건 엄격히 열등한 선택입니다.

### 4. Longest Substring Without Repeating Characters <span class="badge badge-med">Medium</span>

모든 문자가 서로 다른 가장 긴 substring의 길이.

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

*O(N) time, O(min(N, alphabet)) space.* **Pitfall:** `last_seen[ch] >= start` guard는 필수입니다 — 이게 없으면 window 밖의 낡은 index가 `start`를 뒤로 끌어당깁니다.

### 5. Longest Repeating Character Replacement <span class="badge badge-med">Medium</span>

최대 `k`개의 문자를 교체한 뒤 하나의 반복 문자로 이루어진 가장 긴 substring.

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

*O(N) time, O(alphabet) space.* **Pitfall:** shrink할 때 `max_freq`를 재계산하지 마세요 — 역대 최댓값을 유지하는 게 옳습니다. `best`는 window가 커질 때만 커지기 때문입니다.

### 6. Minimum Window Substring <span class="badge badge-hard">Hard</span>

`t`의 모든 문자를(중복 포함) 담는 `s`의 가장 작은 substring.

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

*O(|s| + |t|) time, O(alphabet) space.* **Pitfall:** `missing`을 (중복을 세는) 총 필요 문자 수로 추적하고, 음수 `need`를 surplus로 취급하세요 — 스스로 납득하도록 dry-run 하세요.

## Common variations

- **at-most로 "정확히 k distinct".** `exactly(k) = atMost(k) − atMost(k−1)`. 어려운 "exactly" window를 쉬운 monotone window 두 개로 바꿉니다 — 아주 흔한 follow-up입니다.
- **Fixed-size window** (size k의 max sum, sliding-window 평균): shrink loop 없음; 들어오는 원소를 더하고, 나가는 원소를 뺍니다.
- **Sliding-window maximum**: 단순 pointer가 아니라 **monotonic deque**가 필요합니다 — [Stacks & Queues](#/coding/stack-queue)를 보세요.
- **Fast & slow pointers**: Linked List Cycle, Find the Duplicate Number, 그리고 list의 중점 찾기 모두 1×/2× 속도 쌍을 O(1) space로 씁니다.
- **ML dressing**: "confidence ≥ threshold인 frame의 가장 긴 연속 구간"은 고정 제약 window입니다.

## Pitfalls

> [!WARNING] 대비해 연습할 window 버그
> - 여러 번의 eviction이 필요할 때(step당 여러 개) `while` 대신 `if`로 shrink하기.
> - window가 valid해지기 전에 `best` 갱신하기(shrink-to-minimize 문제에서는 valid 영역 *안에서* 갱신).
> - two pointers에서, 하나만 이동해야 할 tie에서 양 끝을 다 이동하기.
> - sliding window가 **monotone** invariant를 요구한다는 걸 잊기 — "subarray sum = k"에서 음수가 있으면 깨집니다; 거기서는 [prefix sum + hashing](#/coding/hashing)을 쓰세요.

<details class="qa"><summary>sliding window는 언제 실패하고, 무엇이 그것을 대체하나요?</summary>
<div class="qa-body">

**Short:** 제약이 window에 대해 monotone하지 않을 때 — 가장 대표적으로 음수가 있는 subarray-sum 문제.

**Deep:** sliding window는 window가 "너무 크거나/valid"해지면 왼쪽에서 안전하게 shrink할 수 있다는 데 의존하며, 이는 window를 확장하면 그 양이 한 방향으로 움직여야 함을 요구합니다. 모든 값이 non-negative면 running sum은 원소를 더할수록 커지기만 하므로 `sum ≤ target`을 회복하도록 shrink하는 게 valid합니다. 음수를 도입하면 원소를 더하는 게 sum을 *감소*시킬 수 있으므로, shrink한 window도 여전히 개선 가능할 수 있습니다 — monotonicity가 사라집니다. 그러면 **prefix sum + hash map**([Subarray Sum Equals K](#/coding/hashing) 참고)으로 전환하는데, 이는 monotonicity에 의존하지 않고 조건을 만족하는 window를 셉니다.
</div></details>

<details class="qa"><summary>sorted array에서의 Two Sum — two pointers인가 binary search인가?</summary>
<div class="qa-body">

**Short:** Two pointers: O(N) time, O(1) space, one pass.

**Deep:** 각 원소를 고정하고 그 complement를 binary-search하여 O(N log N)로 할 *수도* 있습니다. 여기서는 two pointers가 엄격히 낫습니다. 각 비교가 한쪽 끝의 후보 하나를 버리기 때문입니다 — O(N) linear 수렴이죠. binary search는 정적인 sorted array에 여러 독립적인 target을 질의할 때만 이깁니다(sort를 amortize하고, 이후 query당 O(log N)). trade-off를 말하세요 — 흔한 탐침 질문입니다.
</div></details>

<details class="qa"><summary>Follow-ups</summary>
<div class="qa-body">

- **"3Sum / 4Sum?"** → sort하고, 바깥 index를 고정하고, 나머지를 two-pointer; duplicate를 조심스럽게 skip. O(N²)/O(N³).
- **"정확히 k개의 distinct 정수를 가진 subarray?"** → at-most-k 트릭.
- **"linked-list cycle의 *시작*을 탐지?"** → Floyd's: 만난 뒤 한 pointer를 head로 reset하고 둘을 1씩 전진; 입구에서 만납니다.
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
