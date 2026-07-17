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

## Representative problems

### 1. Merge Intervals (Medium) — sort by start
```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda iv: iv[0])
    out = []
    for start, end in intervals:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)     # overlap ⇒ extend
        else:
            out.append([start, end])
    return out
```
`O(N log N)`. Whether touching endpoints (`[1,2],[2,3]`) merge is a `<` vs `<=` decision — confirm the problem's convention out loud.

### 2. Non-overlapping Intervals (Medium) — sort by end, activity selection
Minimum removals = `N −` (max non-overlapping kept).

```python
def erase_overlap(intervals: list[list[int]]) -> int:
    intervals.sort(key=lambda iv: iv[1])         # earliest finish first
    kept, prev_end = 0, float("-inf")
    for start, end in intervals:
        if start >= prev_end:                    # no overlap with last kept
            kept += 1
            prev_end = end
    return len(intervals) - kept
```
`O(N log N)`. Sorting by *start* here is the classic wrong turn — it doesn't give the exchange-argument guarantee.

### 3. Minimum Arrows to Burst Balloons (Medium)
Same earliest-end greedy: one arrow at each interval's end bursts every balloon overlapping it.

```python
def find_min_arrows(points: list[list[int]]) -> int:
    points.sort(key=lambda p: p[1])
    arrows, last = 0, float("-inf")
    for start, end in points:
        if start > last:                         # current arrow can't reach
            arrows += 1
            last = end                           # new arrow at this end
    return arrows
```
`O(N log N)`. Structurally identical to activity selection — recognizing that saves derivation time.

### 4. Meeting Rooms II (Medium) — heap of end-times
Minimum rooms = max simultaneous overlap. A min-heap of end-times tracks rooms in use.

```python
import heapq

def min_meeting_rooms(intervals: list[list[int]]) -> int:
    intervals.sort(key=lambda iv: iv[0])         # process by start
    ends = []                                    # min-heap of end times
    for start, end in intervals:
        if ends and ends[0] <= start:            # earliest room freed in time
            heapq.heappop(ends)
        heapq.heappush(ends, end)
    return len(ends)                             # peak concurrency
```
`O(N log N)`. Alternative: sweep-line over sorted start/end events, `+1`/`−1`, track the max — same complexity, no heap. Bridges to [Heaps](#/coding/heap-priority-queue).

### 5. Jump Game (Medium) — greedy reachability
Track the farthest index reachable; if the loop index ever exceeds it, you're stuck.

```python
def can_jump(nums: list[int]) -> bool:
    farthest = 0
    for i, jump in enumerate(nums):
        if i > farthest:
            return False                         # unreachable gap
        farthest = max(farthest, i + jump)
    return True
```
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
