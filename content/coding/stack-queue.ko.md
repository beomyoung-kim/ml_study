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

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임(~10 MB)을 내려받고, 이후 실행은 즉시입니다. 본인 에디터가 편하면 각 문제의 **LeetCode** 링크로 이동하세요. (아래 두 개의 class-design 문제는 정적 참고용으로 남겨둡니다.)

### 1. Valid Parentheses <span class="badge badge-easy">Easy</span> · [LeetCode ↗](https://leetcode.com/problems/valid-parentheses/)

`()[]{}` 문자열이 올바르게 matched되고 nested되었나요?

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"is_valid","starter":"def is_valid(s: str) -> bool:\n    # push openers; on a closer, the popped top must be its match\n    pass","tests":[{"args":["()"],"expect":true},{"args":["()[]{}"],"expect":true},{"args":["(]"],"expect":false},{"args":["([)]"],"expect":false},{"args":["{[]}"],"expect":true}],"solution":"def is_valid(s: str) -> bool:\n    match = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n    stack = []\n    for ch in s:\n        if ch in \"([{\":\n            stack.append(ch)\n        elif not stack or stack.pop() != match[ch]:\n            return False\n    return not stack"}
</script>
</div>

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

### 3. Daily Temperatures <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/daily-temperatures/)

각 날에 대해, 더 따뜻한 온도까지 며칠이 걸리나요(없으면 0).

**Approach.** **monotonic decreasing stack of indices**: 오늘이 top을 넘어서면, 그 날의 답이 해결됩니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"daily_temperatures","starter":"def daily_temperatures(temps: list[int]) -> list[int]:\n    # monotonic decreasing stack of indices; resolve a day when today beats the top\n    pass","tests":[{"args":[[73,74,75,71,69,72,76,73]],"expect":[1,1,4,2,1,1,0,0]},{"args":[[30,40,50,60]],"expect":[1,1,1,0]},{"args":[[30,60,90]],"expect":[1,1,0]},{"args":[[90,80,70]],"expect":[0,0,0]}],"solution":"def daily_temperatures(temps: list[int]) -> list[int]:\n    ans = [0] * len(temps)\n    stack = []\n    for i, t in enumerate(temps):\n        while stack and temps[stack[-1]] < t:\n            j = stack.pop()\n            ans[j] = i - j\n        stack.append(i)\n    return ans"}
</script>
</div>

*O(N) time, O(N) space* (각 index는 한 번 push/pop). **Pitfall:** value가 아니라 **index**를 저장하세요 — 거리가 필요합니다.

### 4. Evaluate Reverse Polish Notation <span class="badge badge-med">Medium</span> · [LeetCode ↗](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

postfix 식을 평가하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"eval_rpn","starter":"def eval_rpn(tokens: list[str]) -> int:\n    # stack of operands; on an operator pop b then a, push a op b\n    pass","tests":[{"args":[["2","1","+","3","*"]],"expect":9},{"args":[["4","13","5","/","+"]],"expect":6},{"args":[["10","6","9","3","+","-11","*","/","*","17","+","5","+"]],"expect":22},{"args":[["3","-4","+"]],"expect":-1}],"solution":"def eval_rpn(tokens: list[str]) -> int:\n    stack = []\n    def trunc_div(a, b):\n        if b == 0:\n            raise ZeroDivisionError\n        q = abs(a) // abs(b)\n        return -q if (a < 0) != (b < 0) else q\n    ops = {\"+\": lambda a, b: a + b, \"-\": lambda a, b: a - b,\n           \"*\": lambda a, b: a * b, \"/\": trunc_div}\n    for tok in tokens:\n        if tok in ops:\n            b, a = stack.pop(), stack.pop()\n            stack.append(ops[tok](a, b))\n        else:\n            stack.append(int(tok))\n    return stack[0]"}
</script>
</div>

*O(N) time, O(N) space.* **Pitfall:** operand 순서 — `b`를 pop한 뒤 `a`를 pop하고 `a op b`를 계산하세요. `int(a / b)`는 큰 정수를 float로 바꿀 수 있으므로, 절댓값 정수 나눗셈에 부호를 복원해 0 방향 truncation을 구현합니다. `a // b`는 음수에서 floor라 다릅니다.

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

### 6. Sliding Window Maximum <span class="badge badge-hard">Hard</span> · [LeetCode ↗](https://leetcode.com/problems/sliding-window-maximum/)

size `k`인 모든 window의 최댓값 — heap이 아니라 **monotonic deque**.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"max_sliding_window","starter":"from collections import deque\n\ndef max_sliding_window(nums: list[int], k: int) -> list[int]:\n    # monotonic deque of indices (values decreasing); front is the window max\n    pass","tests":[{"args":[[1,3,-1,-3,5,3,6,7],3],"expect":[3,3,5,5,6,7]},{"args":[[1],1],"expect":[1]},{"args":[[1,-1],1],"expect":[1,-1]},{"args":[[9,11],2],"expect":[11]}],"solution":"from collections import deque\n\ndef max_sliding_window(nums: list[int], k: int) -> list[int]:\n    dq = deque()\n    out = []\n    for i, x in enumerate(nums):\n        while dq and nums[dq[-1]] <= x:\n            dq.pop()\n        dq.append(i)\n        if dq[0] <= i - k:\n            dq.popleft()\n        if i >= k - 1:\n            out.append(nums[dq[0]])\n    return out"}
</script>
</div>

*O(N) time, O(k) space.* heap은 O(N log k)를 주지만, deque는 각 index가 한 번 들어오고 한 번 나가므로 O(N)입니다. **Pitfall:** 낡은 front를 index로 evict하고, 들어오는 value보다 `≤`인 동안 tail을 pop하세요.

### 7. Largest Rectangle in Histogram <span class="badge badge-hard">Hard</span> · [LeetCode ↗](https://leetcode.com/problems/largest-rectangle-in-histogram/)

histogram 아래의 가장 큰 axis-aligned rectangle.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"largest_rectangle_area","starter":"def largest_rectangle_area(heights: list[int]) -> int:\n    # monotonic increasing stack with sentinels; width = i - stack[-1] - 1\n    pass","tests":[{"args":[[2,1,5,6,2,3]],"expect":10},{"args":[[2,4]],"expect":4},{"args":[[2,1,2]],"expect":3},{"args":[[6,2,5,4,5,1,6]],"expect":12}],"solution":"def largest_rectangle_area(heights: list[int]) -> int:\n    heights = [0] + heights + [0]\n    stack = [0]\n    best = 0\n    for i in range(1, len(heights)):\n        while heights[stack[-1]] > heights[i]:\n            h = heights[stack.pop()]\n            width = i - stack[-1] - 1\n            best = max(best, h * width)\n        stack.append(i)\n    return best"}
</script>
</div>

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
