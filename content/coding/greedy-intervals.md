# Greedy & Intervals

> [!TIP] Say this first
> "Greedy works when a **locally optimal choice provably leads to a global optimum** — I should be able to sketch an exchange argument. For interval problems specifically, 90% of the work is choosing the **sort key**: by start, by end, or by length." Naming the sort key and *why* is the entire signal.

Greedy is fast (usually one sort + one pass) but only correct when the greedy-choice property holds. Interval problems are the most common greedy subgenre, and the sort key encodes the whole strategy. When greedy fails, it's almost always a hand-off to [DP](#/coding/dynamic-programming).

## When greedy is safe (and when it isn't)

<div class="proscons"><div><div class="pros-t">Greedy cues</div>

- Interval scheduling / merging / min-removals / min-arrows.
- "Maximum non-overlapping," "minimum resources," "reach the end."
- You can argue: swapping the greedy pick for any other never improves the result (**exchange argument**).
- Sorting reveals an obvious per-step best choice.

</div><div><div class="cons-t">Greedy fails → DP</div>

- Choices **interact** (taking one changes the value of later options non-trivially).
- You can construct a counterexample where local-best loses (coin change `{1,3,4}` for 6).
- The problem asks to *count* solutions or optimize over combinations.

</div></div>

## The interval sort-key decision

| Goal | Sort by | Greedy rule |
| --- | --- | --- |
| Merge overlapping | **start** | extend current end while next start ≤ current end |
| Max non-overlapping / min removals | **end** | keep earliest-finishing, skip anything starting before it |
| Min arrows / points to stab all | **end** | shoot at each interval's end, cover all that overlap |
| Min meeting rooms | start + end-heap | reuse a room when the earliest end ≤ next start |

> [!NOTE] The activity-selection insight
> To pack the **most** non-overlapping intervals, always keep the one that **finishes earliest** — it leaves maximal room for the rest. This "earliest finish time" rule is the exchange argument you should be ready to state, and it's why min-removal / max-selection problems sort by *end*, not start.

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
<figcaption>Sorted by end time, greedily keep each interval that starts at or after the last kept end.</figcaption>
</figure>

## Practice — implement, run, test

> [!TIP] How to use this section
> Each problem below has a **live Python editor**. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime (~10 MB); later runs are instant. Prefer your own editor? Each problem links out to **LeetCode**.

Work them in order — the sort key is the whole game, so name it (start, end, or length) before you type.

### 1. Merge Intervals <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/merge-intervals/) — sort by start
<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "merge", "starter": "def merge(intervals: list[list[int]]) -> list[list[int]]:\n    # sort by start; extend the last kept interval when they overlap\n    pass", "tests": [{"args": [[[1, 3], [2, 6], [8, 10], [15, 18]]], "expect": [[1, 6], [8, 10], [15, 18]]}, {"args": [[[1, 4], [4, 5]]], "expect": [[1, 5]]}, {"args": [[[1, 4], [2, 3]]], "expect": [[1, 4]]}, {"args": [[[1, 4], [0, 4]]], "expect": [[0, 4]]}], "solution": "def merge(intervals):\n    intervals.sort(key=lambda iv: iv[0])\n    out = []\n    for start, end in intervals:\n        if out and start <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], end)\n        else:\n            out.append([start, end])\n    return out"}
</script>
</div>
`O(N log N)`. Whether touching endpoints (`[1,2],[2,3]`) merge is a `<` vs `<=` decision — confirm the problem's convention out loud.

### 2. Non-overlapping Intervals <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/non-overlapping-intervals/) — sort by end, activity selection
Minimum removals = `N −` (max non-overlapping kept).

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "erase_overlap", "starter": "def erase_overlap(intervals: list[list[int]]) -> int:\n    # sort by end (earliest finish first); count how many you can keep\n    pass", "tests": [{"args": [[[1, 2], [2, 3], [3, 4], [1, 3]]], "expect": 1}, {"args": [[[1, 2], [1, 2], [1, 2]]], "expect": 2}, {"args": [[[1, 2], [2, 3]]], "expect": 0}, {"args": [[[1, 100], [11, 22], [1, 11], [2, 12]]], "expect": 2}], "solution": "def erase_overlap(intervals):\n    intervals.sort(key=lambda iv: iv[1])\n    kept, prev_end = 0, float('-inf')\n    for start, end in intervals:\n        if start >= prev_end:\n            kept += 1\n            prev_end = end\n    return len(intervals) - kept"}
</script>
</div>
`O(N log N)`. Sorting by *start* here is the classic wrong turn — it doesn't give the exchange-argument guarantee.

### 3. Minimum Arrows to Burst Balloons <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)
Same earliest-end greedy: one arrow at each interval's end bursts every balloon overlapping it.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "find_min_arrows", "starter": "def find_min_arrows(points: list[list[int]]) -> int:\n    # sort by end; new arrow whenever the current one can't reach\n    pass", "tests": [{"args": [[[10, 16], [2, 8], [1, 6], [7, 12]]], "expect": 2}, {"args": [[[1, 2], [3, 4], [5, 6], [7, 8]]], "expect": 4}, {"args": [[[1, 2], [2, 3], [3, 4], [4, 5]]], "expect": 2}, {"args": [[[1, 2]]], "expect": 1}], "solution": "def find_min_arrows(points):\n    points.sort(key=lambda p: p[1])\n    arrows, last = 0, float('-inf')\n    for start, end in points:\n        if start > last:\n            arrows += 1\n            last = end\n    return arrows"}
</script>
</div>
`O(N log N)`. Structurally identical to activity selection — recognizing that saves derivation time.

### 4. Meeting Rooms II <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/meeting-rooms-ii/) — heap of end-times
Minimum rooms = max simultaneous overlap. A min-heap of end-times tracks rooms in use.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "min_meeting_rooms", "starter": "import heapq\n\ndef min_meeting_rooms(intervals: list[list[int]]) -> int:\n    # process by start; a min-heap of end-times tracks rooms in use\n    pass", "tests": [{"args": [[[0, 30], [5, 10], [15, 20]]], "expect": 2}, {"args": [[[7, 10], [2, 4]]], "expect": 1}, {"args": [[[1, 5], [8, 9], [8, 9]]], "expect": 2}, {"args": [[[13, 15], [1, 13]]], "expect": 1}], "solution": "import heapq\n\ndef min_meeting_rooms(intervals):\n    intervals.sort(key=lambda iv: iv[0])\n    ends = []\n    for start, end in intervals:\n        if ends and ends[0] <= start:\n            heapq.heappop(ends)\n        heapq.heappush(ends, end)\n    return len(ends)"}
</script>
</div>
`O(N log N)`. Alternative: sweep-line over sorted start/end events, `+1`/`−1`, track the max — same complexity, no heap. Bridges to [Heaps](#/coding/heap-priority-queue).

### 5. Jump Game <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/jump-game/) — greedy reachability
Track the farthest index reachable; if the loop index ever exceeds it, you're stuck.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func": "can_jump", "starter": "def can_jump(nums: list[int]) -> bool:\n    # track the farthest reachable index; fail if the loop index passes it\n    pass", "tests": [{"args": [[2, 3, 1, 1, 4]], "expect": true}, {"args": [[3, 2, 1, 0, 4]], "expect": false}, {"args": [[0]], "expect": true}, {"args": [[2, 0, 0]], "expect": true}, {"args": [[1, 0, 1, 0]], "expect": false}], "solution": "def can_jump(nums):\n    farthest = 0\n    for i, jump in enumerate(nums):\n        if i > farthest:\n            return False\n        farthest = max(farthest, i + jump)\n    return True"}
</script>
</div>
`O(N)`, `O(1)` space. The DP formulation is `O(N²)` — leading with the greedy reachability argument is the win. *Jump Game II* (min jumps) extends this to a BFS-like level count, still `O(N)`.

## When greedy fails — the DP hand-off

- **Coin change** `{1,3,4}`, amount 6: greedy grabs `4+1+1` (3 coins); optimal is `3+3` (2). Choices interact → [DP](#/coding/dynamic-programming).
- **0/1 knapsack:** greedy by value/weight ratio is optimal only for the *fractional* version; the 0/1 case needs DP.
- **Longest increasing subsequence:** no greedy prefix rule; DP or patience sorting.
- The tell: if you can't sketch an exchange argument in a sentence, assume greedy is wrong and reach for DP.

## Pitfalls

- **Wrong sort key** — merging sorts by start, selection/arrows sort by end. This is the single most common interval mistake.
- **`<` vs `<=` on touching endpoints** — clarify whether `[1,2]` and `[2,3]` overlap.
- **Asserting greedy without justification** — always have the exchange argument or a counterexample ready.
- **Confusing Jump Game (reachability) with Jump Game II (min jumps).**
- **Mutating input by sorting** when the caller needs original order — copy if required.

## Q&A

<details class="qa"><summary>How do you know greedy is correct and not just plausible?</summary>
<div class="qa-body">

**Short:** I look for the greedy-choice property via an **exchange argument**: assume an optimal solution differs from the greedy pick, then show you can swap in the greedy choice without making it worse — so a greedy-optimal solution exists. If I can't make that argument, I suspect greedy is wrong and test a small counterexample.

**Deep:** For activity selection, the argument is concrete: take the earliest-finishing activity `g`; any optimal set's first activity `o` finishes no earlier, so replacing `o` with `g` frees at least as much time and keeps the count. Induct on the rest. That's the difference between "greedy passes the samples" and "greedy is provably correct" — interviewers push on exactly this.
</div></details>

<details class="qa"><summary>Merge intervals vs meeting rooms — why different tools?</summary>
<div class="qa-body">

**Short:** Merging asks for the *union* of overlaps → sort by start, extend a running end (`O(1)` state). Meeting rooms asks for the *max concurrency* → I need to know how many are simultaneously open, so I track active end-times with a heap (or a sweep line).

**Deep:** They differ in what "overlap" produces. Merge collapses overlaps into fewer intervals — a single scalar (current end) is enough. Rooms count peak simultaneous overlap — that's a multiset of open intervals, which the min-heap of end-times maintains, popping rooms as they free up. Same input, different question, different state.
</div></details>

**Follow-ups you should expect**
- "Prove your greedy is optimal." → exchange argument for earliest-finish.
- "Do meeting rooms without a heap." → sweep-line of `+1`/`−1` events.
- "What if intervals are weighted?" → weighted interval scheduling is **DP**, not greedy.
- "Min jumps, not just reachability?" → greedy level expansion, still `O(N)`.

## Cheat-sheet

| Fact | Detail |
| --- | --- |
| Greedy validity | needs an exchange argument; else use DP |
| Merge overlapping | sort by **start**, extend running end |
| Max non-overlap / min removal | sort by **end** (activity selection) |
| Min arrows / stabbing | sort by **end**, shoot at each end |
| Min meeting rooms | sort by start + min-heap of ends (= peak concurrency) |
| Jump Game | track farthest reachable, `O(N)` |
| `<` vs `<=` | clarify touching-endpoint convention |
| Greedy fails | coin change `{1,3,4}`, 0/1 knapsack, LIS, weighted scheduling |
| Typical cost | one sort + one pass, `O(N log N)` |

**Related:** [Dynamic Programming](#/coding/dynamic-programming) · [Heaps & Priority Queues](#/coding/heap-priority-queue) · [Arrays & Strings](#/coding/arrays-strings) · back to [The Core Patterns](#/coding/patterns) and [Coding Round Strategy](#/coding/strategy).
