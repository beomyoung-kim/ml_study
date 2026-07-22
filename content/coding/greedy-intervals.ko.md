# Greedy & Intervals

> [!TIP] 이 말부터 시작하세요
> "Greedy는 **국소적으로 최적인 선택이 증명 가능하게 전역 최적으로 이어질** 때 동작합니다 — exchange argument를 스케치할 수 있어야 합니다. interval 문제에서는 특히 90%의 작업이 **정렬 키**를 고르는 것입니다: start로, end로, 아니면 길이로." 정렬 키와 *그 이유*를 언급하는 것이 신호의 전부입니다.

Greedy는 빠르지만(보통 정렬 한 번 + 순회 한 번) greedy-choice 속성이 성립할 때만 정확합니다. Interval 문제는 가장 흔한 greedy 하위 장르이고, 정렬 키가 전략 전체를 인코딩합니다. Greedy가 실패하면 거의 항상 [DP](#/coding/dynamic-programming)로의 인계입니다.

## 언제 greedy가 안전한가 (그리고 아닐 때)

<div class="proscons"><div><div class="pros-t">Greedy 신호</div>

- Interval 스케줄링 / 병합 / 최소 제거 / 최소 화살.
- "최대 non-overlapping," "최소 자원," "끝에 도달."
- 논증 가능: greedy 선택을 다른 것으로 바꿔도 결과가 절대 나아지지 않음 (**exchange argument**).
- 정렬이 스텝별 명백한 최선의 선택을 드러냄.

</div><div><div class="cons-t">Greedy 실패 → DP</div>

- 선택들이 **상호작용**함 (하나를 취하면 이후 옵션의 값이 사소하지 않게 바뀜).
- local-best가 지는 반례를 구성할 수 있음 (coin change `{1,3,4}`로 6).
- 문제가 해의 *개수*를 세거나 조합에 대해 최적화하라고 함.

</div></div>

## Interval 정렬 키 결정

| 목표 | 정렬 기준 | Greedy 규칙 |
| --- | --- | --- |
| Overlapping 병합 | **start** | 다음 start ≤ 현재 end인 동안 현재 end를 연장 |
| Max non-overlapping / min 제거 | **end** | 가장 먼저 끝나는 것을 유지, 그 전에 시작하는 건 건너뜀 |
| 모두 찌르는 min 화살 / 점 | **end** | 각 interval의 end에서 쏘아, 겹치는 것을 모두 커버 |
| Min 회의실 | start + end-heap | 가장 이른 end ≤ 다음 start면 방을 재사용 |

> [!NOTE] Activity-selection 통찰
> **가장 많은** non-overlapping interval을 담으려면, 항상 **가장 먼저 끝나는** 것을 유지하세요 — 나머지에 최대한의 여지를 남깁니다. 이 "earliest finish time" 규칙이 여러분이 말할 수 있어야 할 exchange argument이고, min-removal / max-selection 문제가 start가 아니라 *end*로 정렬하는 이유입니다.

<figure>
<svg viewBox="0 0 480 130" width="100%" role="img" aria-label="Intervals sorted by end time; greedy keeps earliest-finishing">
  <g font-family="ui-monospace, monospace" font-size="12">
    <line x1="20" y1="115" x2="460" y2="115" stroke="currentColor" opacity="0.3"/>
    <rect x="20"  y="25" width="120" height="16" rx="3" fill="#12a15033" stroke="#12a150"/><text x="24" y="20">keep (ends first)</text>
    <rect x="90"  y="50" width="150" height="16" rx="3" fill="#e0533f22" stroke="#e0533f"/><text x="94" y="45" fill="#e0533f">drop (overlaps)</text>
    <rect x="160" y="75" width="130" height="16" rx="3" fill="#12a15033" stroke="#12a150"/><text x="164" y="70">keep</text>
    <rect x="310" y="100" width="120" height="16" rx="3" fill="#12a15033" stroke="#12a150"/><text x="314" y="95">keep</text>
    <line x1="140" y1="18" x2="140" y2="120" stroke="#6366f1" stroke-dasharray="3 3"/>
  </g>
</svg>
<figcaption>end time으로 정렬한 뒤, 마지막으로 유지한 end 이후에 시작하는 각 interval을 greedy하게 유지합니다.</figcaption>
</figure>

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임(~10 MB)을 내려받고, 이후 실행은 즉시입니다. 본인 에디터가 편하면 각 문제의 **LeetCode** 링크로 이동하세요.

순서대로 풀어보세요 — 정렬 키가 승부의 전부이니, 코드를 치기 전에 무엇으로 정렬할지 먼저 정하세요.

### 1. Merge Intervals <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/merge-intervals/) — start로 정렬
<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "merge", "starter": "def merge(intervals: list[list[int]]) -> list[list[int]]:\n    # sort by start; extend the last kept interval when they overlap\n    pass", "tests": [{"args": [[[1, 3], [2, 6], [8, 10], [15, 18]]], "expect": [[1, 6], [8, 10], [15, 18]]}, {"args": [[[1, 4], [4, 5]]], "expect": [[1, 5]]}, {"args": [[[1, 4], [2, 3]]], "expect": [[1, 4]]}, {"args": [[[1, 4], [0, 4]]], "expect": [[0, 4]]}], "solution": "def merge(intervals):\n    intervals.sort(key=lambda iv: iv[0])\n    out = []\n    for start, end in intervals:\n        if out and start <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], end)\n        else:\n            out.append([start, end])\n    return out"}
</script>
</div>
`O(N log N)`. 맞닿은 끝점(`[1,2],[2,3]`)이 병합되는지는 `<` vs `<=` 결정입니다 — 문제의 관례를 소리 내어 확인하세요.

### 2. Non-overlapping Intervals <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/non-overlapping-intervals/) — end로 정렬, activity selection
최소 제거 수 = `N −` (유지한 최대 non-overlapping).

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "erase_overlap", "starter": "def erase_overlap(intervals: list[list[int]]) -> int:\n    # sort by end (earliest finish first); count how many you can keep\n    pass", "tests": [{"args": [[[1, 2], [2, 3], [3, 4], [1, 3]]], "expect": 1}, {"args": [[[1, 2], [1, 2], [1, 2]]], "expect": 2}, {"args": [[[1, 2], [2, 3]]], "expect": 0}, {"args": [[[1, 100], [11, 22], [1, 11], [2, 12]]], "expect": 2}], "solution": "def erase_overlap(intervals):\n    intervals.sort(key=lambda iv: iv[1])\n    kept, prev_end = 0, float('-inf')\n    for start, end in intervals:\n        if start >= prev_end:\n            kept += 1\n            prev_end = end\n    return len(intervals) - kept"}
</script>
</div>
`O(N log N)`. 여기서 *start*로 정렬하는 것이 고전적인 오답입니다 — exchange-argument 보장을 주지 못합니다.

### 3. Minimum Arrows to Burst Balloons <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)
같은 earliest-end greedy: 각 interval의 end에 놓인 화살 하나가 그와 겹치는 모든 풍선을 터뜨립니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "find_min_arrows", "starter": "def find_min_arrows(points: list[list[int]]) -> int:\n    # sort by end; new arrow whenever the current one can't reach\n    pass", "tests": [{"args": [[[10, 16], [2, 8], [1, 6], [7, 12]]], "expect": 2}, {"args": [[[1, 2], [3, 4], [5, 6], [7, 8]]], "expect": 4}, {"args": [[[1, 2], [2, 3], [3, 4], [4, 5]]], "expect": 2}, {"args": [[[1, 2]]], "expect": 1}], "solution": "def find_min_arrows(points):\n    points.sort(key=lambda p: p[1])\n    arrows, last = 0, float('-inf')\n    for start, end in points:\n        if start > last:\n            arrows += 1\n            last = end\n    return arrows"}
</script>
</div>
`O(N log N)`. 구조적으로 activity selection과 동일합니다 — 이를 알아채면 유도 시간을 아낍니다.

### 4. Meeting Rooms II <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/meeting-rooms-ii/) — end-time의 heap
최소 방 수 = 최대 동시 overlap. end-time의 min-heap이 사용 중인 방을 추적합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "min_meeting_rooms", "starter": "import heapq\n\ndef min_meeting_rooms(intervals: list[list[int]]) -> int:\n    # process by start; a min-heap of end-times tracks rooms in use\n    pass", "tests": [{"args": [[[0, 30], [5, 10], [15, 20]]], "expect": 2}, {"args": [[[7, 10], [2, 4]]], "expect": 1}, {"args": [[[1, 5], [8, 9], [8, 9]]], "expect": 2}, {"args": [[[13, 15], [1, 13]]], "expect": 1}], "solution": "import heapq\n\ndef min_meeting_rooms(intervals):\n    intervals.sort(key=lambda iv: iv[0])\n    ends = []\n    for start, end in intervals:\n        if ends and ends[0] <= start:\n            heapq.heappop(ends)\n        heapq.heappush(ends, end)\n    return len(ends)"}
</script>
</div>

`O(N log N)`. 대안: 정렬된 start/end 이벤트에 대한 sweep-line, `+1`/`−1`, 최대값 추적 — 같은 복잡도, heap 없음. [Heaps](#/coding/heap-priority-queue)로 연결됩니다.

### 5. Jump Game <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/jump-game/) — greedy 도달 가능성
도달 가능한 가장 먼 인덱스를 추적; 루프 인덱스가 그것을 넘어서면 막힌 것입니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "can_jump", "starter": "def can_jump(nums: list[int]) -> bool:\n    # track the farthest reachable index; fail if the loop index passes it\n    pass", "tests": [{"args": [[2, 3, 1, 1, 4]], "expect": true}, {"args": [[3, 2, 1, 0, 4]], "expect": false}, {"args": [[0]], "expect": true}, {"args": [[2, 0, 0]], "expect": true}, {"args": [[1, 0, 1, 0]], "expect": false}], "solution": "def can_jump(nums):\n    farthest = 0\n    for i, jump in enumerate(nums):\n        if i > farthest:\n            return False\n        farthest = max(farthest, i + jump)\n    return True"}
</script>
</div>
`O(N)`, `O(1)` 공간. DP 정식화는 `O(N²)`입니다 — greedy 도달 가능성 논증을 먼저 제시하는 것이 승부수입니다. *Jump Game II*(최소 점프)는 이를 BFS류 레벨 카운트로 확장하며, 여전히 `O(N)`입니다.

## Greedy가 실패할 때 — DP로의 인계

- **Coin change** `{1,3,4}`, 금액 6: greedy는 `4+1+1`(3개)를 잡지만; 최적은 `3+3`(2개)입니다. 선택이 상호작용 → [DP](#/coding/dynamic-programming).
- **0/1 knapsack:** 가치/무게 비율로 greedy하는 것은 *fractional* 버전에서만 최적입니다; 0/1 케이스는 DP가 필요합니다.
- **Longest increasing subsequence:** greedy prefix 규칙이 없음; DP 혹은 patience sorting.
- 결정적 신호: exchange argument를 한 문장으로 스케치할 수 없다면, greedy가 틀렸다고 가정하고 DP를 꺼내세요.

## 함정

- **잘못된 정렬 키** — 병합은 start로, selection/화살은 end로 정렬합니다. 가장 흔한 interval 실수입니다.
- **맞닿은 끝점에서 `<` vs `<=`** — `[1,2]`와 `[2,3]`이 겹치는지 명확히 하세요.
- **정당화 없이 greedy를 단언** — 항상 exchange argument나 반례를 준비하세요.
- **Jump Game(도달 가능성)과 Jump Game II(최소 점프) 혼동.**
- 호출자가 원래 순서를 필요로 하는데 **정렬로 입력을 변형** — 필요하면 복사하세요.

## Q&A

<details class="qa"><summary>Greedy가 그럴듯한 게 아니라 정확하다는 걸 어떻게 아나요?</summary>
<div class="qa-body">

**짧게:** **exchange argument**로 greedy-choice 속성을 찾습니다: 최적 해가 greedy 선택과 다르다고 가정한 뒤, greedy 선택으로 바꿔도 나빠지지 않음을 보입니다 — 따라서 greedy-optimal 해가 존재합니다. 그 논증을 못 하면 greedy가 틀렸다고 의심하고 작은 반례를 테스트합니다.

**깊게:** Activity selection에서 논증은 구체적입니다: 가장 먼저 끝나는 활동 `g`를 잡습니다; 어떤 최적 집합의 첫 활동 `o`도 그보다 먼저 끝나지 않으므로, `o`를 `g`로 대체하면 최소한 같은 만큼의 시간이 남고 개수도 유지됩니다. 나머지에 대해 귀납합니다. 이것이 "greedy가 샘플을 통과한다"와 "greedy가 증명 가능하게 정확하다"의 차이입니다 — 면접관이 정확히 이 점을 파고듭니다.
</div></details>

<details class="qa"><summary>Merge intervals vs meeting rooms — 왜 다른 도구인가요?</summary>
<div class="qa-body">

**짧게:** 병합은 overlap의 *union*을 요구 → start로 정렬, 진행 중인 end를 연장(`O(1)` state). 회의실은 *최대 동시성*을 요구 → 동시에 몇 개가 열려 있는지 알아야 하므로, heap(혹은 sweep line)으로 활성 end-time을 추적합니다.

**깊게:** "overlap"이 만들어내는 것이 다릅니다. 병합은 overlap을 더 적은 interval로 합칩니다 — scalar 하나(현재 end)면 충분합니다. 회의실은 최대 동시 overlap을 셉니다 — 그건 열린 interval의 multiset이고, end-time의 min-heap이 방이 비면 pop하며 이를 유지합니다. 같은 입력, 다른 질문, 다른 state.
</div></details>

**예상되는 후속 질문**
- "당신의 greedy가 최적임을 증명하세요." → earliest-finish에 대한 exchange argument.
- "Heap 없이 회의실을 하세요." → `+1`/`−1` 이벤트의 sweep-line.
- "Interval에 가중치가 있으면?" → weighted interval scheduling은 greedy가 아니라 **DP**.
- "도달 가능성이 아니라 최소 점프는?" → greedy 레벨 확장, 여전히 `O(N)`.

## Cheat-sheet

| 사실 | 세부 |
| --- | --- |
| Greedy 유효성 | exchange argument 필요; 아니면 DP 사용 |
| Overlapping 병합 | **start**로 정렬, 진행 중인 end 연장 |
| Max non-overlap / min 제거 | **end**로 정렬 (activity selection) |
| Min 화살 / 찌르기 | **end**로 정렬, 각 end에서 쏨 |
| Min 회의실 | start로 정렬 + end의 min-heap (= 최대 동시성) |
| Jump Game | 도달 가능한 가장 먼 곳 추적, `O(N)` |
| `<` vs `<=` | 맞닿은 끝점 관례를 명확히 |
| Greedy 실패 | coin change `{1,3,4}`, 0/1 knapsack, LIS, weighted scheduling |
| 전형적 비용 | 정렬 한 번 + 순회 한 번, `O(N log N)` |

**관련:** [Dynamic Programming](#/coding/dynamic-programming) · [Heaps & Priority Queues](#/coding/heap-priority-queue) · [Arrays & Strings](#/coding/arrays-strings) · 다시 [The Core Patterns](#/coding/patterns)와 [Coding Round Strategy](#/coding/strategy)로.
