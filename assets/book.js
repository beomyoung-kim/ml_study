/* =========================================================================
   BOOK MANIFEST — the single source of truth for the table of contents.
   To add a chapter: drop a Markdown file in content/<part>/, then add a line
   here. To add a part: add an object to BOOK.parts. Order here == book order.
   `file` is relative to content/  (omit the .md extension).
   `title` is the English label; `titleKo` is the Korean-mode label. Omit `titleKo`
   when the English title should remain unchanged for a technical/proper noun.
   NOTE: a chapter's `file` path (its content/<dir>/) is independent of the part
   it is listed under — e.g. foundations/architectures is grouped in Foundations,
   ml-coding/nms-iou is grouped in Computer Vision. Grouping follows the reader's
   learning flow, not the directory prefix.
   ========================================================================= */
window.BOOK = {
  title: "ML Interview Codex",
  updated: "2026-07-21",
  parts: [
    {
      title: "Start Here", titleKo: "시작하기", emoji: "🚀",
      chapters: [
        { id: "home",            title: "Introduction",                  titleKo: "소개",                    file: "index" },
        { id: "how-to-use",      title: "How to Use This Book",          titleKo: "이 책 사용법",            file: "start/how-to-use" },
        { id: "pipeline",        title: "The RS/AS Interview Pipeline",  titleKo: "RS/AS 인터뷰 파이프라인", file: "process/pipeline" },
        { id: "prep-plan",       title: "2-, 4-, or 8-Week Prep Plan",   titleKo: "2·4·8주 준비 플랜",       file: "start/prep-plan" },
        { id: "landscape-2026",  title: "The 2026 Landscape",            titleKo: "2026 지형도",             file: "start/landscape-2026", badge: "2026" },
      ],
    },
    {
      title: "Interview Process & Execution", titleKo: "면접 프로세스 & 실행", emoji: "🗺️",
      chapters: [
        { id: "tips-mistakes",   title: "Common Mistakes & Red Flags",   titleKo: "흔한 실수 & 레드 플래그",   file: "playbook/mistakes" },
        { id: "tips-communication", title: "Communication & Whiteboarding", titleKo: "커뮤니케이션 & 화이트보딩", file: "playbook/communication" },
        { id: "phone-screens",   title: "Phone Screen Playbook",         titleKo: "폰 스크린 플레이북",       file: "process/phone-screens", badge: "hub" },
        { id: "recruiter-hm",    title: "Recruiter & Hiring-Manager Screens", titleKo: "리크루터 & HM 스크리닝", file: "process/recruiter-hm" },
        { id: "companies",       title: "Company Playbooks",             titleKo: "회사별 플레이북",          file: "process/companies" },
      ],
    },
    {
      title: "Coding", titleKo: "코딩", emoji: "⌨️",
      chapters: [
        { id: "coding-strategy", title: "Coding Round Strategy",         titleKo: "코딩 라운드 전략",         file: "coding/strategy" },
        { id: "patterns",        title: "The Core Patterns",             titleKo: "핵심 패턴",               file: "coding/patterns" },
        { id: "arrays-strings",  title: "Arrays & Strings",              file: "coding/arrays-strings" },
        { id: "hashing",         title: "Hashing",                       file: "coding/hashing" },
        { id: "two-pointers",    title: "Two Pointers & Sliding Window", file: "coding/two-pointers-sliding-window" },
        { id: "stack-queue",     title: "Stacks & Queues",               file: "coding/stack-queue" },
        { id: "binary-search",   title: "Binary Search",                 file: "coding/binary-search" },
        { id: "heap",            title: "Heaps & Priority Queues",       file: "coding/heap-priority-queue" },
        { id: "trees-bst",       title: "Trees & BSTs",                  file: "coding/trees-bst" },
        { id: "graphs",          title: "Graphs (BFS/DFS)",              file: "coding/graphs-bfs-dfs" },
        { id: "union-find",      title: "Union-Find",                    file: "coding/union-find" },
        { id: "greedy",          title: "Greedy & Intervals",            file: "coding/greedy-intervals" },
        { id: "dp",              title: "Dynamic Programming",           file: "coding/dynamic-programming" },
      ],
    },
    {
      title: "ML & DL Foundations", titleKo: "ML · DL 기초", emoji: "🌱",
      chapters: [
        { id: "what-is-ml",      title: "What is Machine Learning?",     titleKo: "머신러닝이란?",           file: "foundations/what-is-ml", badge: "primer" },
        { id: "math",            title: "Linear Algebra & Calculus",     titleKo: "선형대수 & 미적분",        file: "foundations/linear-algebra-calculus" },
        { id: "prob-stats",      title: "Probability & Statistics",      titleKo: "확률 & 통계",             file: "foundations/probability-statistics" },
        { id: "nn-basics",       title: "Neural Networks: Neuron → MLP", titleKo: "신경망 첫걸음: 뉴런 → MLP", file: "foundations/neural-networks-basics", badge: "primer" },
        { id: "optimization",    title: "Optimization",                  titleKo: "Optimization",            file: "foundations/optimization" },
        { id: "architectures",   title: "CNNs, RNNs & Transformers",     titleKo: "CNN · RNN · Transformer", file: "foundations/architectures" },
        { id: "normalization",   title: "Normalization & Stability",     titleKo: "Normalization & 학습 안정성", file: "foundations/normalization-stability" },
        { id: "regularization",  title: "Regularization & Generalization", titleKo: "Regularization & 일반화", file: "foundations/regularization-generalization" },
        { id: "metrics-eval",    title: "Evaluation Metrics",            titleKo: "평가 지표",               file: "foundations/evaluation-metrics" },
        { id: "debugging",       title: "Debugging & Experimentation",   titleKo: "디버깅 & 실험",           file: "foundations/debugging-experimentation" },
        { id: "efficiency",      title: "Mixed Precision & Efficiency",  titleKo: "Mixed Precision & 효율화 (심화)", file: "foundations/mixed-precision-efficiency" },
        { id: "distributed",     title: "Distributed Training",          titleKo: "분산 학습 (심화)",         file: "foundations/distributed-training" },
      ],
    },
    {
      title: "From-Scratch Implementations", titleKo: "밑바닥부터 구현", emoji: "🧱",
      chapters: [
        { id: "ml-coding-intro", title: "The ML Coding Round",           titleKo: "ML 코딩 라운드",          file: "ml-coding/intro" },
        { id: "numpy-primer",    title: "NumPy & Broadcasting Primer",   titleKo: "NumPy & 브로드캐스팅 프라이머", file: "ml-coding/numpy-primer", badge: "primer" },
        { id: "losses",          title: "Losses & Gradients (from scratch)", titleKo: "손실 & gradient (밑바닥)", file: "ml-coding/losses-gradients" },
        { id: "kmeans",          title: "K-Means (from scratch)",        titleKo: "K-Means (밑바닥)",        file: "ml-coding/kmeans" },
        { id: "conv-pool",       title: "Conv & Pooling (from scratch)", titleKo: "Conv & Pooling (밑바닥)", file: "ml-coding/conv-pooling" },
        { id: "attention",       title: "Attention (from scratch)",      titleKo: "Attention (밑바닥)",      file: "ml-coding/attention" },
        { id: "pos-rope",        title: "Positional Encoding & RoPE",    titleKo: "Positional Encoding & RoPE", file: "ml-coding/positional-encoding-rope", badge: "lab" },
        { id: "transformer",     title: "A Transformer Block (from scratch)", titleKo: "Transformer Block (밑바닥)", file: "ml-coding/transformer" },
        { id: "dataloader",      title: "Dataloaders & Augmentation",    titleKo: "Dataloader & 증강",       file: "ml-coding/dataloader-augmentation" },
      ],
    },
    {
      title: "Computer Vision", titleKo: "컴퓨터 비전", emoji: "👁️",
      chapters: [
        { id: "cv-image-basics", title: "Image Basics: Pixels & Tensors", titleKo: "이미지 기초: 픽셀·채널·텐서", file: "cv/image-basics", badge: "primer" },
        { id: "cv-cnns",         title: "CNNs for Vision",               titleKo: "비전을 위한 CNN",          file: "cv/cnns-for-vision", badge: "core" },
        { id: "cv-classification", title: "Image Classification",        titleKo: "이미지 분류",             file: "cv/classification", badge: "core" },
        { id: "cv-backbones",    title: "Backbones & Transfer Learning", titleKo: "백본 & 전이학습",          file: "cv/backbones-transfer", badge: "core" },
        { id: "cv-augmentation", title: "Data Augmentation for Vision",  titleKo: "비전 데이터 증강",         file: "cv/augmentation", badge: "core" },
        { id: "cv-ssl",          title: "Self-Supervised Learning",      titleKo: "자기지도학습 입문",        file: "cv/self-supervised", badge: "core" },
        { id: "cv-detection",    title: "Object Detection",              titleKo: "객체 검출 (Detection)",    file: "cv/detection" },
        { id: "nms-iou",         title: "IoU & Non-Max Suppression",     titleKo: "IoU & NMS 구현",          file: "ml-coding/nms-iou" },
        { id: "cv-segmentation", title: "Segmentation",                  titleKo: "Segmentation",            file: "cv/segmentation" },
        { id: "cv-upsampling",   title: "Upsampling & U-Net",            titleKo: "업샘플링 & U-Net",         file: "cv/upsampling-unet", badge: "core" },
        { id: "metrics",         title: "mAP & mIoU (from scratch)",     titleKo: "mAP & mIoU (밑바닥)",     file: "ml-coding/metrics-map-miou" },
        { id: "cv-weaksup",      title: "Weak & Semi-Supervised",        titleKo: "Weak & Semi-Supervised",  file: "cv/weak-semi-supervised" },
        { id: "cv-continual",    title: "Continual Learning",            titleKo: "Continual Learning",      file: "cv/continual-learning" },
        { id: "cv-foundation",   title: "Vision Foundation Models",      titleKo: "Vision Foundation Models", file: "cv/foundation-models" },
        { id: "cv-matting",      title: "Image Matting",                 titleKo: "Image Matting",           file: "cv/matting" },
      ],
    },
    {
      title: "LLM Core", titleKo: "LLM 기초", emoji: "🤖",
      chapters: [
        { id: "llm-tokenization", title: "Tokenization & BPE",           titleKo: "토크나이제이션 & BPE",     file: "llm/tokenization", badge: "primer" },
        { id: "llm-embeddings",  title: "Embeddings",                    titleKo: "임베딩",                  file: "llm/embeddings", badge: "primer" },
        { id: "llm-next-token",  title: "Next-Token Prediction",         titleKo: "다음 토큰 예측 직관",      file: "llm/next-token", badge: "primer" },
        { id: "llm-fundamentals", title: "LLM Fundamentals",             titleKo: "LLM Fundamentals",       file: "llm/fundamentals" },
        { id: "llm-decoding",    title: "Decoding & Sampling",           titleKo: "디코딩 & 샘플링 전략",     file: "llm/decoding-sampling", badge: "core" },
        { id: "llm-prompting",   title: "Prompting & In-Context Learning", titleKo: "프롬프팅 & ICL",         file: "llm/prompting", badge: "core" },
      ],
    },
    {
      title: "LLM Alignment, Reasoning & Agents", titleKo: "LLM 정렬 · 추론 · 에이전트", emoji: "🧠",
      chapters: [
        { id: "llm-rl-primer",   title: "RL Fundamentals Primer",        titleKo: "RL 기초 프라이머",         file: "llm/rl-primer", badge: "primer" },
        { id: "llm-alignment",    title: "Post-Training & Alignment",    titleKo: "Post-Training & Alignment", file: "llm/alignment", badge: "2026" },
        { id: "llm-reasoning",    title: "Reasoning & Test-Time Compute", file: "llm/reasoning", badge: "2026" },
        { id: "llm-rag",         title: "RAG (Retrieval-Augmented Gen.)", titleKo: "RAG (검색 증강 생성)",    file: "llm/rag", badge: "core" },
        { id: "llm-agents",       title: "Agentic AI & Tool Use",        file: "llm/agents", badge: "2026" },
      ],
    },
    {
      title: "VLMs & Visual Agents", titleKo: "VLM · 비주얼 에이전트", emoji: "🖼️",
      chapters: [
        { id: "vlm-101",         title: "VLM 101: Image → Tokens",       titleKo: "VLM 101: 이미지 → 토큰",   file: "vlm/vlm-101", badge: "primer" },
        { id: "vlm-pretraining",  title: "Vision-Language Pretraining",  titleKo: "VLM Pretraining (CLIP·융합)", file: "vlm/pretraining" },
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
        { id: "research-papers", title: "Reading & Critiquing Papers",   titleKo: "논문 읽기 & 비평",         file: "research/papers" },
        { id: "research-exp",    title: "Experiment Design & Ablations", titleKo: "실험 설계 & Ablation",     file: "research/experiment-design" },
        { id: "research-fail",   title: "Failure & Negative Results",    titleKo: "실패 & 부정적 결과",       file: "research/failure" },
        { id: "research-present",title: "Presenting Research",           titleKo: "리서치 발표",             file: "research/presenting" },
        { id: "research-talk",   title: "The Research Job Talk",         titleKo: "리서치 잡토크",            file: "research/job-talk" },
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
      title: "Personal Resume Interview Packet", titleKo: "개인 이력서 인터뷰 패킷", emoji: "🎯",
      chapters: [
        { id: "cv-overview",     title: "Your CV → Interview Map",       titleKo: "개인 이력서 → 인터뷰 맵",  file: "resume/overview" },
        { id: "dd-zim",          title: "Deep-Dive: ZIM",                titleKo: "딥다이브: ZIM",           file: "resume/zim" },
        { id: "dd-pointwssis",   title: "Deep-Dive: PointWSSIS & BESTIE", titleKo: "딥다이브: PointWSSIS & BESTIE", file: "resume/pointwssis-bestie" },
        { id: "dd-eclipse",      title: "Deep-Dive: ECLIPSE",            titleKo: "딥다이브: ECLIPSE",       file: "resume/eclipse" },
        { id: "dd-phoenix",      title: "Deep-Dive: Phoenix (ECCV'26 Mask Refinement)", titleKo: "딥다이브: Phoenix (ECCV'26 마스크 정제)", file: "resume/phoenix-mask-refinement", badge: "2026" },
        { id: "dd-facesign",     title: "Deep-Dive: FaceSign",           titleKo: "딥다이브: FaceSign",      file: "resume/facesign" },
        { id: "dd-ondevice",     title: "Deep-Dive: On-Device Seg",      titleKo: "딥다이브: On-Device Seg", file: "resume/on-device-segmentation" },
        { id: "dd-grounded",     title: "Deep-Dive: Grounded VLM/Agents", titleKo: "딥다이브: Grounded VLM/Agents", file: "resume/grounded-vlm-agents" },
        { id: "dd-nra",          title: "Deep-Dive: Spatial-Reasoning Agent (under review)", titleKo: "딥다이브: Spatial-Reasoning Agent (심사 중)", file: "resume/neurips26-spatial-reasoning", badge: "2026" },
        { id: "cv-stage-answers", title: "Stage-by-Stage Resume Answers", titleKo: "이력서 기반 단계별 예시 답변", file: "resume/interview-stage-answers", badge: "practice" },
        { id: "cv-questions",    title: "Predicted Questions & Answers", titleKo: "예상 질문 & 답변",         file: "resume/predicted-questions", badge: "practice" },
      ],
    },
    {
      title: "Day-Of, Questions & Offers", titleKo: "당일 · 질문 · 오퍼", emoji: "🧭",
      chapters: [
        { id: "tips-questions",  title: "Questions to Ask Them",         titleKo: "그들에게 할 질문",         file: "playbook/questions-to-ask" },
        { id: "tips-remote",     title: "Remote Interview Setup",        titleKo: "원격 인터뷰 셋업",         file: "playbook/remote-setup" },
        { id: "tips-tactics",    title: "Day-Of Tactics & Recovery",     titleKo: "당일 전략 & 리커버리",     file: "playbook/tactics" },
        { id: "negotiation",     title: "Offers, Levels & Negotiation",  titleKo: "오퍼 · 레벨 · 협상",      file: "process/negotiation" },
      ],
    },
    {
      title: "Resources", titleKo: "리소스", emoji: "📚",
      chapters: [
        { id: "res-glossary",    title: "Glossary",                      titleKo: "용어집",                  file: "resources/glossary" },
        { id: "res-opensource",  title: "Curated Open-Source Resources", titleKo: "엄선한 공개 학습 자료",     file: "resources/open-source" },
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
