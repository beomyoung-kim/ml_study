# Stacks & Queues

> [!TIP] The cue
> **Stack (LIFO):** nested/matched 구조(parentheses, calculator, decode), 또는 **monotonic stack**을 통한 "nearest greater/smaller element". **Queue/Deque (FIFO):** level-by-level 처리(BFS), 또는 **monotonic deque**를 통한 sliding-window maxima. *가장 최근의 아직 해결되지 않은* 항목을 원하고 있다면, 그건 stack입니다.

여기서 가장 값진 아이디어는 **monotonic stack/deque**입니다: stack을 정렬된 상태(increasing 또는 decreasing)로 유지하고 위반자를 pop하면, "각 원소에 대해 가장 가까운 larger/smaller 이웃"을 단 한 번의 O(N) pass로 답합니다. 이것이 Daily Temperatures, Largest Rectangle, Trapping Rain Water(stack 변형), Sliding-Window Maximum을 구동합니다.

## When to use which

| Cue | Structure | Complexity |
| --- | --- | --- |
| Matched/nested symbols, undo, expression eval | Stack | O(N) |
| Nearest greater/smaller, spans, histogram | Monotonic stack | O(N) |
| FIFO order, BFS frontier | Queue (`deque`) | O(N) |
| Sliding-window max/min | Monotonic deque | O(N) |
| O(1) push/pop/top/min | Stack of (value, running-min) | O(1)/op |
| Queue from stacks / stack from queues | Two-structure amortization | amortized O(1) |

> [!NOTE] queue로는 `list`가 아니라 `deque`를 쓰세요
> `list.pop(0)`은 O(N)입니다. `collections.deque`는 O(1) `append`/`appendleft`/`pop`/`popleft`를 줍니다. 순수 stack이라면 `append`/`pop`(둘 다 끝에서 O(1))을 쓰는 `list`가 관용적입니다.

## Worked examples

### 1. Valid Parentheses <span class="badge badge-easy">Easy</span>

`()[]{}` 문자열이 올바르게 matched되고 nested되었나요?

```python
def is_valid(s: str) -> bool:
    match = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif not stack or stack.pop() != match[ch]:
            return False
    return not stack                 # leftover openers → invalid
```

*O(N) time, O(N) space.* **Pitfall:** 두 edge case — 빈 stack에서의 closer, 그리고 끝에 남은 opener(`return not stack`).

### 2. Implement Queue using Stacks <span class="badge badge-easy">Easy</span>

두 개의 LIFO stack으로 만든 FIFO queue.

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

*op당 amortized O(1), O(N) space.* **Pitfall:** *`out_s`가 비었을 때만* 옮기세요; 각 원소는 최대 한 번 이동하므로 amortized bound가 나옵니다.

### 3. Daily Temperatures <span class="badge badge-med">Medium</span>

각 날에 대해, 더 따뜻한 온도까지 며칠이 걸리나요(없으면 0).

**Approach.** **monotonic decreasing stack of indices**: 오늘이 top을 넘어서면, 그 날의 답이 해결됩니다.

```python
def daily_temperatures(temps: list[int]) -> list[int]:
    ans = [0] * len(temps)
    stack: list[int] = []               # indices, temps decreasing
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            ans[j] = i - j
        stack.append(i)
    return ans
```

*O(N) time, O(N) space* (각 index는 한 번 push/pop). **Pitfall:** value가 아니라 **index**를 저장하세요 — 거리가 필요합니다.

### 4. Evaluate Reverse Polish Notation <span class="badge badge-med">Medium</span>

postfix 식을 평가하세요.

```python
def eval_rpn(tokens: list[str]) -> int:
    stack: list[int] = []
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
           "*": lambda a, b: a * b, "/": lambda a, b: int(a / b)}
    for tok in tokens:
        if tok in ops:
            b, a = stack.pop(), stack.pop()      # order matters!
            stack.append(ops[tok](a, b))
        else:
            stack.append(int(tok))
    return stack[0]
```

*O(N) time, O(N) space.* **Pitfall:** operand 순서 — `b`를 pop한 뒤 `a`를 pop하고, `a op b`를 계산하세요. `int(a / b)`(0 방향으로 truncate)를 쓰세요; `a // b`는 floor라 음수에서 틀립니다.

### 5. Min Stack <span class="badge badge-med">Medium</span>

O(1) `getMin`을 갖는 stack.

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

*op당 O(1), O(N) space.* **Pitfall:** 단일 `min` 변수는 `pop` 후 복원할 수 없습니다; running min을 *각 원소와 함께* 저장하세요.

### 6. Sliding Window Maximum <span class="badge badge-hard">Hard</span>

size `k`인 모든 window의 최댓값 — heap이 아니라 **monotonic deque**.

```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    dq: deque[int] = deque()          # indices, values decreasing
    out: list[int] = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()                  # x dominates smaller tail values
        dq.append(i)
        if dq[0] <= i - k:            # front fell out of the window
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])   # front is the window max
    return out
```

*O(N) time, O(k) space.* heap은 O(N log k)를 주지만, deque는 각 index가 한 번 들어오고 한 번 나가므로 O(N)입니다. **Pitfall:** 낡은 front를 index로 evict하고, 들어오는 value보다 `≤`인 동안 tail을 pop하세요.

### 7. Largest Rectangle in Histogram <span class="badge badge-hard">Hard</span>

histogram 아래의 가장 큰 axis-aligned rectangle.

```python
def largest_rectangle_area(heights: list[int]) -> int:
    heights = [0] + heights + [0]      # sentinels flush the stack cleanly
    stack = [0]                        # indices, heights increasing
    best = 0
    for i in range(1, len(heights)):
        while heights[stack[-1]] > heights[i]:
            h = heights[stack.pop()]
            width = i - stack[-1] - 1
            best = max(best, h * width)
        stack.append(i)
    return best
```

*O(N) time, O(N) space.* **Pitfall:** width는 `i − stack[-1] − 1`(새 left boundary와 `i` 사이의 span)이지, `i − popped_index`가 *아닙니다*. sentinel이 양 끝의 특수 처리를 피하게 해줍니다.

## Common variations

- **Decode String** (`3[a2[c]]`): nested 확장을 위한 두 stack(count와 partial string).
- **Basic Calculator I/II/III**: sign, parenthesis, operator precedence를 위한 stack.
- **Next Greater Element (circular)**: monotonic stack 위로 `2N`번 mod N으로 iterate.
- **BFS** (level order, grid에서의 shortest path): queue *가* 곧 pattern입니다 — [Graphs](#/coding/graphs-bfs-dfs)와 [Trees](#/coding/trees-bst)를 보세요.
- **Monotonic stack ↔ Trapping Rain Water**: nearest-greater를 찾는 바로 그 stack이 물 문제도 풉니다([two-pointer version](#/coding/arrays-strings)의 대안).

## Pitfalls

> [!WARNING] Stack/queue 함정
> - 빈 stack에서 pop하기 — 항상 guard하세요(`if stack and …`).
> - index가 필요할 때(거리, width) value를 push하기.
> - queue로 `list`를 쓰기 → O(N) `pop(0)`; `deque`를 쓰세요.
> - 잘못된 monotonic 방향: *decreasing* stack은 next greater element를 찾고; *increasing*은 next smaller를 찾습니다.
> - recursion은 암묵적 stack입니다 — 깊은 DFS는 Python의 ~1000 recursion limit에 걸릴 수 있습니다; 큰 입력에서는 명시적 stack으로 전환하세요.

<details class="qa"><summary>sliding-window maximum: 왜 heap이 아니라 deque인가요?</summary>
<div class="qa-body">

**Short:** deque는 O(N)이고; heap은 O(N log k)이며 낡은 항목의 lazy deletion이 필요합니다.

**Deep:** max-heap은 현재 최댓값을 주지만, window가 미끄러지면 왼쪽으로 나가는 원소를 제거해야 하는데 — binary heap은 추가 bookkeeping 없이는 임의의 원소를 O(log k)에 삭제할 수 없습니다. 통상적인 해법은 "lazy deletion": 낡은 항목을 남겨두고 top으로 떠오를 때 버리는 건데, 여전히 O(N log k)입니다. monotonic deque는 대신 *여전히 미래의 최댓값이 될 수 있는 후보*만(decreasing 순서의 값들) 유지합니다: 각 index가 한 번 push되고 한 번 pop되므로 총 작업은 O(N)이고, front가 항상 window max입니다. 이 trade-off를 설명하는 게 이 질문의 요점입니다.
</div></details>

<details class="qa"><summary>monotonic stack이 inner while loop에도 불구하고 어떻게 O(N)인가요?</summary>
<div class="qa-body">

**Short:** amortized 분석 — 각 원소는 한 번 push되고 최대 한 번 pop되므로, 총 push+pop ≤ 2N입니다.

**Deep:** nested `while`은 O(N²)처럼 보이지만, 한 원소는 전체 실행에서 딱 한 번만 pop될 수 있습니다; 한 번 pop되면 결코 돌아오지 않죠. 모든 iteration에 걸쳐 합하면, loop body는 outer step당이 아니라 총 최대 N번 실행됩니다. 이건 "queue from two stacks" 뒤의 amortization과 동일합니다. 항상 명시적으로 말하세요 — 면접관은 nesting을 눈대중하는 대신 linear bound를 방어할 수 있는지 탐침합니다.
</div></details>

<details class="qa"><summary>Follow-ups</summary>
<div class="qa-body">

- **"`+ − × ÷`와 parenthesis가 있는 infix 식을 평가?"** → precedence가 있는 두 stack(operand, operator), 또는 shunting-yard.
- **"O(1) `max`를 갖는 queue를 설계?"** → queue와 나란히 두는 monotonic deque.
- **"nested tag 검증 / valid parentheses 생성?"** → stack 검증; 생성은 [backtracking](#/coding/trees-bst)입니다.
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
