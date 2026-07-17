/* =========================================================================
   BOOK MANIFEST — the single source of truth for the table of contents.
   To add a chapter: drop a Markdown file in content/<part>/, then add a line
   here. To add a part: add an object to BOOK.parts. Order here == book order.
   `file` is relative to content/  (omit the .md extension).
   ========================================================================= */
window.BOOK = {
  title: "ML Interview Codex",
  updated: "2026-07-17",
  parts: [
    {
      title: "Start Here", emoji: "🚀",
      chapters: [
        { id: "home",            title: "Introduction",                  file: "index" },
        { id: "how-to-use",      title: "How to Use This Book",          file: "start/how-to-use" },
        { id: "landscape-2026",  title: "The 2026 Landscape",            file: "start/landscape-2026", badge: "2026" },
        { id: "prep-plan",       title: "An 8-Week Prep Plan",           file: "start/prep-plan" },
      ],
    },
    {
      title: "The Interview Process", emoji: "🗺️",
      chapters: [
        { id: "pipeline",        title: "The Big-Tech RS/AS Pipeline",   file: "process/pipeline" },
        { id: "recruiter-hm",    title: "Recruiter & Hiring-Manager Screens", file: "process/recruiter-hm" },
        { id: "companies",       title: "Company Playbooks",             file: "process/companies" },
        { id: "negotiation",     title: "Offers, Levels & Negotiation",  file: "process/negotiation" },
      ],
    },
    {
      title: "Coding", emoji: "⌨️",
      chapters: [
        { id: "coding-strategy", title: "Coding Round Strategy",         file: "coding/strategy" },
        { id: "patterns",        title: "The Core Patterns",             file: "coding/patterns" },
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
      title: "ML Coding From Scratch", emoji: "🔬",
      chapters: [
        { id: "ml-coding-intro", title: "The ML Coding Round",           file: "ml-coding/intro" },
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
      title: "ML & DL Foundations", emoji: "📐",
      chapters: [
        { id: "math",            title: "Linear Algebra & Calculus",     file: "foundations/linear-algebra-calculus" },
        { id: "prob-stats",      title: "Probability & Statistics",      file: "foundations/probability-statistics" },
        { id: "optimization",    title: "Optimization",                  file: "foundations/optimization" },
        { id: "regularization",  title: "Regularization & Generalization", file: "foundations/regularization-generalization" },
        { id: "metrics-eval",    title: "Evaluation Metrics",            file: "foundations/evaluation-metrics" },
        { id: "architectures",   title: "CNNs, RNNs & Transformers",     file: "foundations/architectures" },
        { id: "normalization",   title: "Normalization & Stability",     file: "foundations/normalization-stability" },
        { id: "distributed",     title: "Distributed Training",          file: "foundations/distributed-training" },
        { id: "efficiency",      title: "Mixed Precision & Efficiency",  file: "foundations/mixed-precision-efficiency" },
        { id: "debugging",       title: "Debugging & Experimentation",   file: "foundations/debugging-experimentation" },
      ],
    },
    {
      title: "Computer Vision", emoji: "👁️",
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
      title: "LLMs, VLMs & Agents", emoji: "🤖",
      chapters: [
        { id: "llm-fundamentals", title: "LLM Fundamentals",             file: "llm/fundamentals" },
        { id: "llm-alignment",    title: "Post-Training & Alignment",    file: "llm/alignment", badge: "2026" },
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
      title: "ML System Design", emoji: "🏗️",
      chapters: [
        { id: "mlsd-framework",  title: "The Design Framework",          file: "system-design/framework" },
        { id: "mlsd-cases",      title: "Worked Case Studies",           file: "system-design/case-studies" },
        { id: "mlsd-llm",        title: "Designing LLM/Agent Systems",   file: "system-design/llm-systems", badge: "2026" },
      ],
    },
    {
      title: "Research Interviews", emoji: "📄",
      chapters: [
        { id: "research-talk",   title: "The Research Job Talk",         file: "research/job-talk" },
        { id: "research-papers", title: "Reading & Critiquing Papers",   file: "research/papers" },
        { id: "research-exp",    title: "Experiment Design & Ablations", file: "research/experiment-design" },
        { id: "research-fail",   title: "Failure & Negative Results",    file: "research/failure" },
        { id: "research-present",title: "Presenting Research",           file: "research/presenting" },
      ],
    },
    {
      title: "Behavioral", emoji: "🎭",
      chapters: [
        { id: "behavioral-star", title: "STAR & The Story Bank",         file: "behavioral/star" },
        { id: "behavioral-qs",   title: "Common Questions & Answers",    file: "behavioral/questions" },
      ],
    },
    {
      title: "CV Deep-Dives (Beomyoung Kim)", emoji: "🎯",
      chapters: [
        { id: "cv-overview",     title: "Your CV → Interview Map",       file: "resume/overview" },
        { id: "cv-questions",    title: "Predicted Questions & Answers", file: "resume/predicted-questions", badge: "new" },
        { id: "dd-zim",          title: "Deep-Dive: ZIM",                file: "resume/zim" },
        { id: "dd-eclipse",      title: "Deep-Dive: ECLIPSE",            file: "resume/eclipse" },
        { id: "dd-pointwssis",   title: "Deep-Dive: PointWSSIS & BESTIE", file: "resume/pointwssis-bestie" },
        { id: "dd-facesign",     title: "Deep-Dive: FaceSign",           file: "resume/facesign" },
        { id: "dd-ondevice",     title: "Deep-Dive: On-Device Seg",      file: "resume/on-device-segmentation" },
        { id: "dd-grounded",     title: "Deep-Dive: Grounded VLM/Agents", file: "resume/grounded-vlm-agents" },
        { id: "dd-nra",          title: "Deep-Dive: Spatial-Reasoning Agent (NeurIPS'26)", file: "resume/neurips26-spatial-reasoning", badge: "2026" },
      ],
    },
    {
      title: "The Playbook", emoji: "🧭",
      chapters: [
        { id: "tips-communication", title: "Communication & Whiteboarding", file: "playbook/communication" },
        { id: "tips-remote",     title: "Remote Interview Setup",        file: "playbook/remote-setup" },
        { id: "tips-tactics",    title: "Day-Of Tactics & Recovery",     file: "playbook/tactics" },
        { id: "tips-questions",  title: "Questions to Ask Them",         file: "playbook/questions-to-ask" },
        { id: "tips-mistakes",   title: "Common Mistakes & Red Flags",   file: "playbook/mistakes" },
      ],
    },
    {
      title: "Resources", emoji: "📚",
      chapters: [
        { id: "res-opensource",  title: "Curated Open-Source Resources", file: "resources/open-source" },
        { id: "res-glossary",    title: "Glossary",                      file: "resources/glossary" },
        { id: "res-changelog",   title: "Changelog",                     file: "resources/changelog" },
      ],
    },
  ],
};

/* Flattened chapter list + lookup maps (built once). */
window.BOOK_FLAT = [];
window.BOOK.parts.forEach((part) => {
  part.chapters.forEach((ch) => {
    window.BOOK_FLAT.push({ ...ch, part: part.title, partEmoji: part.emoji });
  });
});
window.BOOK_BY_ID = Object.fromEntries(window.BOOK_FLAT.map((c) => [c.id, c]));
window.BOOK_BY_FILE = Object.fromEntries(window.BOOK_FLAT.map((c) => [c.file, c]));
