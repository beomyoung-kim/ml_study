/* =========================================================================
   BOOK MANIFEST — the single source of truth for the table of contents.
   To add a chapter: drop a Markdown file in content/<part>/, then add a line
   here. To add a part: add an object to BOOK.parts. Order here == book order.
   `file` is relative to content/  (omit the .md extension).
   `titleKo` / `titleKo` on parts+chapters is the Korean-mode label; omit it and
   the English `title` is used (kept for technical/proper-noun chapter names).
   ========================================================================= */
window.BOOK = {
  title: "ML Interview Codex",
  updated: "2026-07-18",
  parts: [
    {
      title: "Start Here", titleKo: "시작하기", emoji: "🚀",
      chapters: [
        { id: "home",            title: "Introduction",                  titleKo: "소개",                    file: "index" },
        { id: "how-to-use",      title: "How to Use This Book",          titleKo: "이 책 사용법",            file: "start/how-to-use" },
        { id: "landscape-2026",  title: "The 2026 Landscape",            titleKo: "2026 지형도",             file: "start/landscape-2026", badge: "2026" },
        { id: "prep-plan",       title: "An 8-Week Prep Plan",           titleKo: "8주 준비 플랜",           file: "start/prep-plan" },
      ],
    },
    {
      title: "The Interview Process", titleKo: "인터뷰 프로세스", emoji: "🗺️",
      chapters: [
        { id: "pipeline",        title: "The Big-Tech RS/AS Pipeline",   titleKo: "빅테크 RS/AS 파이프라인", file: "process/pipeline" },
        { id: "recruiter-hm",    title: "Recruiter & Hiring-Manager Screens", titleKo: "리크루터 & HM 스크리닝", file: "process/recruiter-hm" },
        { id: "companies",       title: "Company Playbooks",             titleKo: "회사별 플레이북",          file: "process/companies" },
        { id: "negotiation",     title: "Offers, Levels & Negotiation",  titleKo: "오퍼 · 레벨 · 협상",      file: "process/negotiation" },
      ],
    },
    {
      title: "Coding", titleKo: "코딩", emoji: "⌨️",
      chapters: [
        { id: "coding-strategy", title: "Coding Round Strategy",         titleKo: "코딩 라운드 전략",         file: "coding/strategy" },
        { id: "patterns",        title: "The Core Patterns",             titleKo: "핵심 패턴",               file: "coding/patterns" },
        { id: "arrays-strings",  title: "Arrays & Strings",              file: "coding/arrays-strings" },
        { id: "two-pointers",    title: "Two Pointers & Sliding Window", file: "coding/two-pointers-sliding-window" },
        { id: "hashing",         title: "Hashing",                       file: "coding/hashing" },
        { id: "stack-queue",     title: "Stacks & Queues",               file: "coding/stack-queue" },
        { id: "binary-search",   title: "Binary Search",                 file: "coding/binary-search" },
        { id: "trees-bst",       title: "Trees & BSTs",                  file: "coding/trees-bst" },
        { id: "graphs",          title: "Graphs (BFS/DFS)",              file: "coding/graphs-bfs-dfs" },
        { id: "dp",              title: "Dynamic Programming",           file: "coding/dynamic-programming" },
        { id: "heap",            title: "Heaps & Priority Queues",       file: "coding/heap-priority-queue" },
        { id: "union-find",      title: "Union-Find",                    file: "coding/union-find" },
        { id: "greedy",          title: "Greedy & Intervals",            file: "coding/greedy-intervals" },
      ],
    },
    {
      title: "ML Coding From Scratch", titleKo: "ML 코딩 (밑바닥부터)", emoji: "🔬",
      chapters: [
        { id: "ml-coding-intro", title: "The ML Coding Round",           titleKo: "ML 코딩 라운드",          file: "ml-coding/intro" },
        { id: "nms-iou",         title: "IoU & Non-Max Suppression",     file: "ml-coding/nms-iou" },
        { id: "conv-pool",       title: "Conv & Pooling",                file: "ml-coding/conv-pooling" },
        { id: "attention",       title: "Attention From Scratch",        file: "ml-coding/attention" },
        { id: "transformer",     title: "A Transformer Block",           file: "ml-coding/transformer" },
        { id: "kmeans",          title: "K-Means",                       file: "ml-coding/kmeans" },
        { id: "dataloader",      title: "Dataloaders & Augmentation",    file: "ml-coding/dataloader-augmentation" },
        { id: "losses",          title: "Losses & Gradients",            file: "ml-coding/losses-gradients" },
        { id: "metrics",         title: "mAP & mIoU",                    file: "ml-coding/metrics-map-miou" },
      ],
    },
    {
      title: "ML & DL Foundations", titleKo: "ML & DL 기초", emoji: "📐",
      chapters: [
        { id: "math",            title: "Linear Algebra & Calculus",     titleKo: "선형대수 & 미적분",        file: "foundations/linear-algebra-calculus" },
        { id: "prob-stats",      title: "Probability & Statistics",      titleKo: "확률 & 통계",             file: "foundations/probability-statistics" },
        { id: "optimization",    title: "Optimization",                  titleKo: "Optimization",            file: "foundations/optimization" },
        { id: "regularization",  title: "Regularization & Generalization", titleKo: "Regularization & 일반화", file: "foundations/regularization-generalization" },
        { id: "metrics-eval",    title: "Evaluation Metrics",            titleKo: "평가 지표",               file: "foundations/evaluation-metrics" },
        { id: "architectures",   title: "CNNs, RNNs & Transformers",     file: "foundations/architectures" },
        { id: "normalization",   title: "Normalization & Stability",     titleKo: "Normalization & 학습 안정성", file: "foundations/normalization-stability" },
        { id: "distributed",     title: "Distributed Training",          titleKo: "분산 학습",               file: "foundations/distributed-training" },
        { id: "efficiency",      title: "Mixed Precision & Efficiency",  titleKo: "Mixed Precision & 효율화", file: "foundations/mixed-precision-efficiency" },
        { id: "debugging",       title: "Debugging & Experimentation",   titleKo: "디버깅 & 실험",           file: "foundations/debugging-experimentation" },
      ],
    },
    {
      title: "Computer Vision", titleKo: "컴퓨터 비전", emoji: "👁️",
      chapters: [
        { id: "cv-segmentation", title: "Segmentation",                  file: "cv/segmentation" },
        { id: "cv-detection",    title: "Object Detection",              file: "cv/detection" },
        { id: "cv-matting",      title: "Image Matting",                 file: "cv/matting" },
        { id: "cv-weaksup",      title: "Weak & Semi-Supervised",        file: "cv/weak-semi-supervised" },
        { id: "cv-continual",    title: "Continual Learning",            file: "cv/continual-learning" },
        { id: "cv-foundation",   title: "Vision Foundation Models",      file: "cv/foundation-models" },
      ],
    },
    {
      title: "LLMs, VLMs & Agents", titleKo: "LLM · VLM · 에이전트", emoji: "🤖",
      chapters: [
        { id: "llm-fundamentals", title: "LLM Fundamentals",             titleKo: "LLM Fundamentals",       file: "llm/fundamentals" },
        { id: "llm-alignment",    title: "Post-Training & Alignment",    titleKo: "Post-Training & Alignment", file: "llm/alignment", badge: "2026" },
        { id: "llm-reasoning",    title: "Reasoning & Test-Time Compute", file: "llm/reasoning", badge: "2026" },
        { id: "llm-agents",       title: "Agentic AI & Tool Use",        file: "llm/agents", badge: "2026" },
        { id: "vlm-pretraining",  title: "Vision-Language Pretraining",  file: "vlm/pretraining" },
        { id: "vlm-practical",    title: "VLM Implementation Details",   file: "vlm/practical" },
        { id: "vlm-instruction",  title: "Instruction Tuning & Decoding", file: "vlm/instruction-tuning" },
        { id: "vlm-grounding",    title: "Grounding & Region Reasoning", file: "vlm/grounding" },
        { id: "vlm-video",        title: "Video-Language Models",        file: "vlm/video" },
        { id: "vlm-agents",       title: "Visual Reasoning Agents",      file: "vlm/visual-agents", badge: "2026" },
      ],
    },
    {
      title: "ML System Design", titleKo: "ML 시스템 디자인", emoji: "🏗️",
      chapters: [
        { id: "mlsd-framework",  title: "The Design Framework",          titleKo: "설계 프레임워크",          file: "system-design/framework" },
        { id: "mlsd-cases",      title: "Worked Case Studies",           titleKo: "사례 연구",               file: "system-design/case-studies" },
        { id: "mlsd-llm",        title: "Designing LLM/Agent Systems",   titleKo: "LLM/에이전트 시스템 설계", file: "system-design/llm-systems", badge: "2026" },
      ],
    },
    {
      title: "Research Interviews", titleKo: "리서치 인터뷰", emoji: "📄",
      chapters: [
        { id: "research-talk",   title: "The Research Job Talk",         titleKo: "리서치 잡토크",            file: "research/job-talk" },
        { id: "research-papers", title: "Reading & Critiquing Papers",   titleKo: "논문 읽기 & 비평",         file: "research/papers" },
        { id: "research-exp",    title: "Experiment Design & Ablations", titleKo: "실험 설계 & Ablation",     file: "research/experiment-design" },
        { id: "research-fail",   title: "Failure & Negative Results",    titleKo: "실패 & 부정적 결과",       file: "research/failure" },
        { id: "research-present",title: "Presenting Research",           titleKo: "리서치 발표",             file: "research/presenting" },
      ],
    },
    {
      title: "Behavioral", titleKo: "행동 면접 (Behavioral)", emoji: "🎭",
      chapters: [
        { id: "behavioral-star", title: "STAR & The Story Bank",         titleKo: "STAR & 스토리 뱅크",       file: "behavioral/star" },
        { id: "behavioral-qs",   title: "Common Questions & Answers",    titleKo: "자주 나오는 질문 & 답변",   file: "behavioral/questions" },
      ],
    },
    {
      title: "CV Deep-Dives (Beomyoung Kim)", titleKo: "이력서 딥다이브 (Beomyoung Kim)", emoji: "🎯",
      chapters: [
        { id: "cv-overview",     title: "Your CV → Interview Map",       titleKo: "이력서 → 인터뷰 맵",       file: "resume/overview" },
        { id: "cv-questions",    title: "Predicted Questions & Answers", titleKo: "예상 질문 & 답변",         file: "resume/predicted-questions", badge: "new" },
        { id: "dd-zim",          title: "Deep-Dive: ZIM",                titleKo: "딥다이브: ZIM",           file: "resume/zim" },
        { id: "dd-eclipse",      title: "Deep-Dive: ECLIPSE",            titleKo: "딥다이브: ECLIPSE",       file: "resume/eclipse" },
        { id: "dd-pointwssis",   title: "Deep-Dive: PointWSSIS & BESTIE", titleKo: "딥다이브: PointWSSIS & BESTIE", file: "resume/pointwssis-bestie" },
        { id: "dd-facesign",     title: "Deep-Dive: FaceSign",           titleKo: "딥다이브: FaceSign",      file: "resume/facesign" },
        { id: "dd-ondevice",     title: "Deep-Dive: On-Device Seg",      titleKo: "딥다이브: On-Device Seg", file: "resume/on-device-segmentation" },
        { id: "dd-grounded",     title: "Deep-Dive: Grounded VLM/Agents", titleKo: "딥다이브: Grounded VLM/Agents", file: "resume/grounded-vlm-agents" },
        { id: "dd-nra",          title: "Deep-Dive: Spatial-Reasoning Agent (NeurIPS'26)", titleKo: "딥다이브: Spatial-Reasoning Agent (NeurIPS'26)", file: "resume/neurips26-spatial-reasoning", badge: "2026" },
      ],
    },
    {
      title: "The Playbook", titleKo: "플레이북", emoji: "🧭",
      chapters: [
        { id: "tips-communication", title: "Communication & Whiteboarding", titleKo: "커뮤니케이션 & 화이트보딩", file: "playbook/communication" },
        { id: "tips-remote",     title: "Remote Interview Setup",        titleKo: "원격 인터뷰 셋업",         file: "playbook/remote-setup" },
        { id: "tips-tactics",    title: "Day-Of Tactics & Recovery",     titleKo: "당일 전략 & 리커버리",     file: "playbook/tactics" },
        { id: "tips-questions",  title: "Questions to Ask Them",         titleKo: "그들에게 할 질문",         file: "playbook/questions-to-ask" },
        { id: "tips-mistakes",   title: "Common Mistakes & Red Flags",   titleKo: "흔한 실수 & 레드 플래그",   file: "playbook/mistakes" },
      ],
    },
    {
      title: "Resources", titleKo: "리소스", emoji: "📚",
      chapters: [
        { id: "res-opensource",  title: "Curated Open-Source Resources", titleKo: "엄선한 오픈소스 자료",     file: "resources/open-source" },
        { id: "res-glossary",    title: "Glossary",                      titleKo: "용어집",                  file: "resources/glossary" },
        { id: "res-changelog",   title: "Changelog",                     titleKo: "변경 이력",               file: "resources/changelog" },
      ],
    },
  ],
};

/* Flattened chapter list + lookup maps (built once). */
window.BOOK_FLAT = [];
window.BOOK.parts.forEach((part) => {
  part.chapters.forEach((ch) => {
    window.BOOK_FLAT.push({ ...ch, part: part.title, partKo: part.titleKo, partEmoji: part.emoji });
  });
});
window.BOOK_BY_ID = Object.fromEntries(window.BOOK_FLAT.map((c) => [c.id, c]));
window.BOOK_BY_FILE = Object.fromEntries(window.BOOK_FLAT.map((c) => [c.file, c]));
