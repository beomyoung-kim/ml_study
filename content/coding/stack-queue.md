# Stacks & Queues

> [!TIP] The cue
> **Stack (LIFO):** nested/matched structure (parentheses, calculators, decode), or "nearest greater/smaller element" via a **monotonic stack**. **Queue/Deque (FIFO):** level-by-level processing (BFS), or sliding-window maxima via a **monotonic deque**. When you find yourself wanting the *most recent unresolved* item, it's a stack.

The high-value idea here is the **monotonic stack/deque**: by keeping the stack sorted (increasing or decreasing) and popping violators, you answer "for each element, the nearest larger/smaller neighbor" in a single O(N) pass. It powers Daily Temperatures, Largest Rectangle, Trapping Rain Water (stack variant), and Sliding-Window Maximum.

## When to use which

| Cue | Structure | Complexity |
| --- | --- | --- |
| Matched/nested symbols, undo, expression eval | Stack | O(N) |
| Nearest greater/smaller, spans, histogram | Monotonic stack | O(N) |
| FIFO order, BFS frontier | Queue (`deque`) | O(N) |
| Sliding-window max/min | Monotonic deque | O(N) |
| O(1) push/pop/top/min | Stack of (value, running-min) | O(1)/op |
| Queue from stacks / stack from queues | Two-structure amortization | amortized O(1) |

> [!NOTE] Use `deque`, never `list`, as a queue
> `list.pop(0)` is O(N). `collections.deque` gives O(1) `append`/`appendleft`/`pop`/`popleft`. For a plain stack, a `list` with `append`/`pop` (both O(1) at the end) is idiomatic.

## Practice — implement, run, test

> [!TIP] How to use this section
> Each problem below has a **live Python editor**. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime (~10 MB); later runs are instant. Prefer your own editor? Each problem links out to **LeetCode**. (The two class-design problems below stay as static references.)

### 1. Valid Parentheses <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/valid-parentheses/)

Is a string of `()[]{}` correctly matched and nested?

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"is_valid","starter":"def is_valid(s: str) -> bool:\n    # push openers; on a closer, the popped top must be its match\n    pass","tests":[{"args":["()"],"expect":true},{"args":["()[]{}"],"expect":true},{"args":["(]"],"expect":false},{"args":["([)]"],"expect":false},{"args":["{[]}"],"expect":true}],"solution":"def is_valid(s: str) -> bool:\n    match = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n    stack = []\n    for ch in s:\n        if ch in \"([{\":\n            stack.append(ch)\n        elif not stack or stack.pop() != match[ch]:\n            return False\n    return not stack"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** two edge cases — a closer with an empty stack, and leftover openers at the end (`return not stack`).

### 2. Implement Queue using Stacks <span class="badge badge-easy">Easy</span>

FIFO queue from two LIFO stacks.

```python
class MyQueue:
    def __init__(self):
        self.in_s: list[int] = []
        self.out_s: list[int] = []

    def push(self, x: int) -> None:
        self.in_s.append(x)

    def _move(self) -> None:
        if not self.out_s:                       # only when out is empty
            while self.in_s:
                self.out_s.append(self.in_s.pop())

    def pop(self) -> int:
        self._move(); return self.out_s.pop()

    def peek(self) -> int:
        self._move(); return self.out_s[-1]

    def empty(self) -> bool:
        return not self.in_s and not self.out_s
```

*Amortized O(1) per op, O(N) space.* **Pitfall:** transfer *only when `out_s` is empty*; each element moves at most once, giving the amortized bound.

### 3. Daily Temperatures <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/daily-temperatures/)

For each day, how many days until a warmer temperature (0 if none).

**Approach.** A **monotonic decreasing stack of indices**: when today beats the top, that day's answer is resolved.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"daily_temperatures","starter":"def daily_temperatures(temps: list[int]) -> list[int]:\n    # monotonic decreasing stack of indices; resolve a day when today beats the top\n    pass","tests":[{"args":[[73,74,75,71,69,72,76,73]],"expect":[1,1,4,2,1,1,0,0]},{"args":[[30,40,50,60]],"expect":[1,1,1,0]},{"args":[[30,60,90]],"expect":[1,1,0]},{"args":[[90,80,70]],"expect":[0,0,0]}],"solution":"def daily_temperatures(temps: list[int]) -> list[int]:\n    ans = [0] * len(temps)\n    stack = []\n    for i, t in enumerate(temps):\n        while stack and temps[stack[-1]] < t:\n            j = stack.pop()\n            ans[j] = i - j\n        stack.append(i)\n    return ans"}
</script>
</div>

*O(N) time, O(N) space* (each index pushed/popped once). **Pitfall:** store **indices**, not values — you need the distance.

### 4. Evaluate Reverse Polish Notation <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

Evaluate a postfix expression.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"eval_rpn","starter":"def eval_rpn(tokens: list[str]) -> int:\n    # stack of operands; on an operator pop b then a, push a op b\n    pass","tests":[{"args":[["2","1","+","3","*"]],"expect":9},{"args":[["4","13","5","/","+"]],"expect":6},{"args":[["10","6","9","3","+","-11","*","/","*","17","+","5","+"]],"expect":22},{"args":[["3","-4","+"]],"expect":-1}],"solution":"def eval_rpn(tokens: list[str]) -> int:\n    stack = []\n    ops = {\"+\": lambda a, b: a + b, \"-\": lambda a, b: a - b,\n           \"*\": lambda a, b: a * b, \"/\": lambda a, b: int(a / b)}\n    for tok in tokens:\n        if tok in ops:\n            b, a = stack.pop(), stack.pop()\n            stack.append(ops[tok](a, b))\n        else:\n            stack.append(int(tok))\n    return stack[0]"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** operand order — pop `b` then `a`, compute `a op b`. Use `int(a / b)` (truncates toward zero); `a // b` floors and is wrong for negatives.

### 5. Min Stack <span class="badge badge-med">Medium</span>

Stack with O(1) `getMin`.

```python
class MinStack:
    def __init__(self):
        self.stack: list[tuple[int, int]] = []   # (value, min_so_far)

    def push(self, val: int) -> None:
        cur_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, cur_min))

    def pop(self) -> None:      self.stack.pop()
    def top(self) -> int:       return self.stack[-1][0]
    def getMin(self) -> int:    return self.stack[-1][1]
```

*O(1) per op, O(N) space.* **Pitfall:** a single `min` variable can't be restored after `pop`; store the running min *alongside each element*.

### 6. Sliding Window Maximum <span class="badge badge-hard">Hard</span> · [LeetCode ↗](https://leetcode.com/problems/sliding-window-maximum/)

Maximum of every window of size `k` — a **monotonic deque**, not a heap.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"max_sliding_window","starter":"from collections import deque\n\ndef max_sliding_window(nums: list[int], k: int) -> list[int]:\n    # monotonic deque of indices (values decreasing); front is the window max\n    pass","tests":[{"args":[[1,3,-1,-3,5,3,6,7],3],"expect":[3,3,5,5,6,7]},{"args":[[1],1],"expect":[1]},{"args":[[1,-1],1],"expect":[1,-1]},{"args":[[9,11],2],"expect":[11]}],"solution":"from collections import deque\n\ndef max_sliding_window(nums: list[int], k: int) -> list[int]:\n    dq = deque()\n    out = []\n    for i, x in enumerate(nums):\n        while dq and nums[dq[-1]] <= x:\n            dq.pop()\n        dq.append(i)\n        if dq[0] <= i - k:\n            dq.popleft()\n        if i >= k - 1:\n            out.append(nums[dq[0]])\n    return out"}
</script>
</div>

*O(N) time, O(k) space.* A heap gives O(N log k); the deque is O(N) because each index enters and leaves once. **Pitfall:** evict the stale front by index, and pop the tail while it's `≤` the incoming value.

### 7. Largest Rectangle in Histogram <span class="badge badge-hard">Hard</span> · [LeetCode ↗](https://leetcode.com/problems/largest-rectangle-in-histogram/)

Biggest axis-aligned rectangle under a histogram.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"largest_rectangle_area","starter":"def largest_rectangle_area(heights: list[int]) -> int:\n    # monotonic increasing stack with sentinels; width = i - stack[-1] - 1\n    pass","tests":[{"args":[[2,1,5,6,2,3]],"expect":10},{"args":[[2,4]],"expect":4},{"args":[[2,1,2]],"expect":3},{"args":[[6,2,5,4,5,1,6]],"expect":12}],"solution":"def largest_rectangle_area(heights: list[int]) -> int:\n    heights = [0] + heights + [0]\n    stack = [0]\n    best = 0\n    for i in range(1, len(heights)):\n        while heights[stack[-1]] > heights[i]:\n            h = heights[stack.pop()]\n            width = i - stack[-1] - 1\n            best = max(best, h * width)\n        stack.append(i)\n    return best"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** width is `i − stack[-1] − 1` (span between the new left boundary and `i`), *not* `i − popped_index`. Sentinels avoid special-casing the ends.

## Common variations

- **Decode String** (`3[a2[c]]`): two stacks (counts and partial strings) for nested expansion.
- **Basic Calculator I/II/III**: stack for signs, parentheses, and operator precedence.
- **Next Greater Element (circular)**: iterate `2N` times mod N over a monotonic stack.
- **BFS** (level order, shortest path in a grid): the queue *is* the pattern — see [Graphs](#/coding/graphs-bfs-dfs) and [Trees](#/coding/trees-bst).
- **Monotonic stack ↔ Trapping Rain Water**: the same stack that finds nearest-greater solves the water problem (an alternative to the [two-pointer version](#/coding/arrays-strings)).

## Pitfalls

> [!WARNING] Stack/queue traps
> - Popping from an empty stack — always guard (`if stack and …`).
> - Pushing values when you need indices (distances, widths).
> - Using a `list` as a queue → O(N) `pop(0)`; use `deque`.
> - Wrong monotonic direction: *decreasing* stack finds the next greater element; *increasing* finds the next smaller.
> - Recursion is an implicit stack — deep DFS can hit Python's ~1000 recursion limit; convert to an explicit stack for large inputs.

<details class="qa"><summary>Sliding-window maximum: why a deque instead of a heap?</summary>
<div class="qa-body">

**Short:** The deque is O(N); a heap is O(N log k) and needs lazy deletion of stale entries.

**Deep:** A max-heap gives the current maximum, but when the window slides you must remove the element leaving on the left — and a binary heap can't delete an arbitrary element in O(log k) without extra bookkeeping. The usual fix is "lazy deletion": leave stale entries and discard them when they surface at the top, which still costs O(N log k). The monotonic deque instead keeps only *candidates that could still be a future maximum* (values in decreasing order): each index is pushed and popped once, so total work is O(N), and the front is always the window max. Explaining this trade-off is the point of the question.
</div></details>

<details class="qa"><summary>How does a monotonic stack give O(N) despite the inner while loop?</summary>
<div class="qa-body">

**Short:** Amortized analysis — each element is pushed once and popped at most once, so total pushes+pops ≤ 2N.

**Deep:** The nested `while` looks like it could be O(N²), but an element can only be popped a single time across the entire run; once popped it never returns. Summing over all iterations, the loop body executes at most N times total, not per outer step. That's the same amortization behind "queue from two stacks." Always state it explicitly — interviewers probe whether you can defend the linear bound rather than eyeball the nesting.
</div></details>

<details class="qa"><summary>Follow-ups</summary>
<div class="qa-body">

- **"Evaluate an infix expression with `+ − × ÷` and parentheses?"** → two stacks (operands, operators) with precedence, or shunting-yard.
- **"Design a queue with O(1) `max`?"** → monotonic deque alongside the queue.
- **"Validate nested tags / generate valid parentheses?"** → stack validation; generation is [backtracking](#/coding/trees-bst).
</div></details>

## Cheat-sheet

| Problem | Structure | Time | Space |
| --- | --- | --- | --- |
| Valid parentheses | stack | O(N) | O(N) |
| Queue from stacks | two stacks (amortized) | O(1)* | O(N) |
| Daily temperatures | monotonic decreasing stack | O(N) | O(N) |
| Evaluate RPN | stack | O(N) | O(N) |
| Min stack | stack of (val, min) | O(1) | O(N) |
| Sliding-window max | monotonic deque | O(N) | O(k) |
| Largest rectangle | monotonic increasing stack | O(N) | O(N) |

\*amortized.

**Related:** [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) · [Graphs (BFS/DFS)](#/coding/graphs-bfs-dfs) · [Heaps & Priority Queues](#/coding/heap-priority-queue) · [Patterns hub](#/coding/patterns)
