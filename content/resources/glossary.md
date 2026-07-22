# Glossary

<div class="tag-row"><span class="tag">A–Z</span><span class="tag">quick recall</span><span class="tag">CV</span><span class="tag">VLM</span><span class="tag">LLM</span><span class="tag">systems</span></div>

> [!TIP] How to use this
> One-line definitions for fast recall — the register you want *out loud* when an interviewer says "what's X?" Lead with the crisp version here, then reach for depth in the linked chapters. If you can't state the one-liner cleanly, you don't own the concept yet. Terms are grouped by first letter; acronyms sit under their initial.

### A

**A/B test** — Online experiment comparing randomly assigned treatment and control groups on live traffic to estimate a causal effect on a metric. It provides strong evidence for a deployment decision when the experimental unit, interference assumptions, guardrails, and statistical tests are sound.

**Ablation** — Removing or swapping one component to isolate its contribution to performance; the core evidence unit in a research deep-dive.

**Activation function** — Element-wise nonlinearity (ReLU, GELU, SwiGLU) that lets a network model non-linear functions.

**AdamW** — Adam variant with *decoupled* weight decay, separate from a gradient-based L2 penalty; a common choice for training transformers.

**AIMv2** — Family of vision encoders pretrained with an autoregressive objective; evaluated in its paper as an alternative for dense and localization features on the reported tasks.

**Alpha matte** — Per-pixel opacity map ($\alpha \in [0,1]$) separating foreground from background; the output of image matting (vs. a hard binary mask).

**AnyRes / dynamic resolution** — A design that divides an image into tiles near its native aspect ratio and processes a variable number of visual tokens. Useful for balancing detail retention against token cost in OCR, documents, and long video.

**ANN (approximate nearest neighbor)** — Sub-linear vector search (HNSW, IVF, ScaNN) trading exactness for speed; the retrieval backbone of RAG and embedding search.

**AP / mAP** — Average Precision = area under the precision–recall curve; **mean AP** averages it over classes (and IoU thresholds, e.g. COCO's mAP@[.5:.95]). The standard detection metric.

**AR (autoregressive)** — Generating a sequence one token at a time, each conditioned on the prefix; the decoding paradigm of GPT-style models.

**Attention** — Mechanism computing a weighted sum of values, weights from query–key similarity ($\text{softmax}(QK^\top/\sqrt{d})V$); the core operation of transformers.

**AUC / ROC-AUC** — Area under the ROC curve; probability a random positive outranks a random negative — a threshold-free ranking metric.

**Augmentation** — Label-preserving input transforms (crop, flip, color jitter, Mixup, RandAugment) that expand effective data and regularize.

### B

**Backpropagation** — Reverse-mode automatic differentiation computing loss gradients w.r.t. all parameters via the chain rule.

**Batch normalization (BN)** — Normalizing activations over the batch dimension to stabilize and speed training; behaves differently at train vs. inference (running stats).

**Beam search** — Decoding that keeps the top-$k$ partial sequences at each step; higher likelihood than greedy, but can be bland.

**BERT** — Bidirectional encoder pretrained with masked language modeling; a representative reference point for the encoder-only language-model family.

**Bias–variance tradeoff** — Decomposition of expected error into underfitting (bias) vs. sensitivity to data (variance); the lens for diagnosing over/underfitting.

**Bounding box** — Axis-aligned rectangle (x, y, w, h) localizing an object; the output primitive of object detection.

**BLEU / ROUGE / METEOR** — N-gram-overlap metrics for translation/summarization; cheap but weakly correlated with quality vs. human or model judges.

### C

**Calibration** — Alignment between predicted confidence and empirical accuracy; measured by ECE / reliability diagrams. Deep nets are often overconfident.

**Catastrophic forgetting** — A network losing previously learned tasks when trained on new ones; the central problem of continual learning (see ECLIPSE).

**Chain-of-thought (CoT)** — A technique—or the resulting rationale—in which prompting, training, or sampling makes a model generate intermediate reasoning tokens before a final answer. It does not imply that a model's internal reasoning is exposed verbatim, nor does it guarantee that a displayed explanation is faithful.

**CLIP** — Contrastive Language–Image Pretraining: dual encoders aligned so matching image/text pairs have high cosine similarity; enables zero-shot classification and open-vocabulary tasks.

**CNN (convolutional neural network)** — Network built from learned convolutional filters exploiting translation equivariance and locality; the classic vision backbone.

**Constitutional AI** — RLAIF variant where an AI critiques and revises outputs against written principles ("a constitution") instead of human labels.

**Continual / lifelong learning** — Learning a stream of tasks without forgetting old ones under limited access to past data.

**Contrastive learning** — Learning representations by pulling positive pairs together and pushing negatives apart (InfoNCE); powers CLIP, SimCLR, MoCo.

**ControlNet** — Adapter adding spatial conditioning (edges, depth, pose) to a frozen diffusion model.

**Cross-attention** — Attention where queries come from one stream and keys/values from another; the fusion mechanism between modalities in VLMs and encoder–decoder models.

**Cross-entropy** — Expected negative log-probability of a predicted distribution $q$ under a target distribution $p$: $H(p,q)=H(p)+D_{KL}(p\|q)$. A proper loss widely used for classification.

**Curriculum learning** — Ordering training from easy to hard examples to improve convergence or final quality.

### D

**DDP (Distributed Data Parallel)** — Data parallelism replicating the model per GPU and all-reducing gradients; the baseline multi-GPU training strategy.

**DDPM / DDIM** — Denoising Diffusion Probabilistic Models (stochastic) and their deterministic implicit-sampler variant (DDIM); the foundations of diffusion generation.

**DETR** — DEtection TRansformer: end-to-end detection as set prediction with bipartite (Hungarian) matching, removing anchors and NMS.

**Dice loss** — Segmentation loss based on the Dice/F1 overlap coefficient. Normalization by foreground size can reduce sensitivity to class imbalance, but small objects, empty masks, and calibration still require separate handling.

**Diffusion model** — Family of generative models that learn a score or denoising field for reversing a gradual noising process. Widely used for image, video, and audio generation, with a design space that increasingly overlaps flow-matching methods.

**DINO / DINOv2 / DINOv3** — Family of self-supervised vision representations that use self-distillation. DINOv3 reports backbones up to 7B parameters and Gram anchoring; claims about frozen-feature advantages apply to the tasks and protocols evaluated in the paper.

**Distillation (knowledge)** — Training a small student to mimic a larger teacher's outputs/logits/features; compresses models for deployment.

**Distribution shift** — Train and deployment data differ (covariate, label, or concept shift); a top cause of silent production degradation.

**DPO (Direct Preference Optimization)** — Preference alignment recast as a classification loss on chosen/rejected pairs, skipping the reward model and RL loop.

**Dropout** — Randomly zeroing activations during training as a regularizer / implicit ensemble.

### E

**Early stopping** — Halting training when validation loss stops improving to prevent overfitting.

**ECE (Expected Calibration Error)** — Weighted average gap between confidence and accuracy across bins; the standard calibration metric.

**Embedding** — Dense vector representation of a discrete or high-dimensional input (token, image, user) in a learned continuous space.

**EMA (exponential moving average)** — Slowly-updated running average of weights (teacher in self-distillation) or statistics; smooths and stabilizes.

**Encoder / decoder** — An encoder maps input to a context-aware representation; a decoder produces a target representation or sequence. Some decoders, such as GPT, are autoregressive, but not every decoder is. Architectures include encoder-only (BERT), decoder-only (GPT), and encoder–decoder (T5).

**Expert parallelism** — Sharding MoE experts across devices so each holds a subset; the parallelism dimension specific to MoE.

### F

**F1 score** — Harmonic mean of precision and recall; the go-to single number under class imbalance.

**Few-shot / in-context learning** — Solving a task from a handful of examples placed in the prompt, without weight updates.

**FID (Fréchet Inception Distance)** — Distance between real and generated image feature distributions; the standard image-generation quality metric (lower is better).

**Fine-tuning** — Continuing training a pretrained model on a target task/dataset; full or parameter-efficient (LoRA).

**FlashAttention** — IO-aware exact attention kernel that tiles the computation to avoid materializing the $N\times N$ matrix in HBM; FA-2/3/4 track successive GPU generations.

**Focal loss** — Cross-entropy down-weighting easy examples ($(1-p)^\gamma$) to focus training on hard ones; designed for extreme foreground–background imbalance in detection.

**FLOPs** — Floating-point operations; a hardware-agnostic proxy for compute cost.

**FP8 / FP16 / BF16 / FP4** — Reduced-precision floating-point formats trading numerical range/precision for speed and memory; FP8 is routine and 4-bit (NVFP4/MXFP4) now reaches pretraining.

**FSDP (Fully Sharded Data Parallel)** — Sharding parameters, gradients, and optimizer states across GPUs (ZeRO-3 style) to fit large models.

### G

**GAN** — Generative Adversarial Network: a generator and discriminator trained in a minimax game; prone to mode collapse and training instability.

**GELU** — Gaussian Error Linear Unit; a smooth activation used in many transformers such as BERT and ViT. Gated activations such as SwiGLU are also common in recent LLM feed-forward networks.

**GEMM** — General matrix multiply; the dominant compute kernel in deep nets.

**GQA (Grouped-Query Attention)** — Attention that maps query heads to a smaller number of key/value-head groups. It trades between MHA's expressivity and MQA's smaller KV cache; quality effects depend on the model and training recipe.

**Gradient accumulation** — Summing gradients over several micro-batches before an update, to simulate a larger batch under memory limits.

**Gradient checkpointing** — Trading compute for memory by recomputing activations in the backward pass instead of storing them.

**Gradient clipping** — Capping gradient norm to prevent exploding updates; standard in RNN/transformer training.

**Gram anchoring** — Regularizing the Gram matrix of dense features to prevent their degradation over long SSL training (DINOv3).

**GRPO (Group Relative Policy Optimization)** — Critic-free policy-optimization family that estimates advantages from a *group* of sampled completions instead of a separate value network; one of several choices for RLVR.

**GSPO (Group Sequence Policy Optimization)** — Group-relative policy-optimization variant that moves the importance ratio to the *sequence* level. Its proposal reports better MoE-RL stability than token-level alternatives.

### H

**Hallucination** — A model generating fluent but false or unsupported content; controlled via grounding, RAG, and verification.

**HNSW** — Hierarchical Navigable Small World graph; a leading ANN index for fast vector search.

**Hungarian matching** — Optimal bipartite assignment (predictions ↔ ground truth) used for set-prediction losses in DETR-style models.

**Hyperparameter** — A setting not learned by gradient descent (learning rate, batch size, depth) chosen by search/tuning.

### I

**ImageNet** — 1.28M-image, 1000-class benchmark; the historical pretraining/eval standard for vision.

**InfoNCE** — Contrastive loss treating one positive against many negatives as a classification problem; the objective behind CLIP and SimCLR.

**Instruction tuning** — Supervised fine-tuning on (instruction, response) pairs to make a base model follow directions.

**IoU (Intersection over Union)** — Overlap metric = area of intersection ÷ area of union; the basis of detection matching and segmentation quality. See [NMS & IoU](#/ml-coding/nms-iou).

### J

**Jailbreak** — Prompt engineered to bypass a model's safety guardrails; a subset of prompt injection.

**JEPA (Joint-Embedding Predictive Architecture)** — Self-supervised approach predicting representations (not pixels) of masked regions; LeCun's proposed path to world models (I-JEPA, V-JEPA).

### K

**KL divergence** — Asymmetric measure of how one distribution diverges from another; the regularizer keeping RLHF/DPO policies near the reference model.

**KV cache** — Stored key/value tensors from prior tokens so autoregressive decoding avoids recomputing them; the main inference memory cost, targeted by GQA/MQA/MLA and paging.

**k-NN / k-means** — Nearest-neighbor classification / centroid-based clustering; classic non-parametric and unsupervised baselines.

### L

**Label smoothing** — Regularization that replaces hard one-hot targets with softened targets. It can help generalization, but does not always improve calibration.

**LayerNorm / RMSNorm** — Usually normalize activation scale across the feature dimension for each sample or token. RMSNorm omits mean-centering and is common in modern LLMs.

**Learning rate schedule** — How the LR evolves (warmup, cosine decay); a top determinant of training quality. See the [lr-schedule widget](#/foundations/optimization).

**LoRA / QLoRA** — Low-Rank Adaptation: freeze base weights and train small low-rank updates. QLoRA trains adapters over a quantized base model; both are widely used parameter-efficient fine-tuning methods.

**Logits** — Pre-softmax raw scores output by a classifier or LM head.

**Loss landscape** — The geometry of loss over parameter space; flatness/sharpness relates to generalization.

### M

**Mamba / SSM** — State-space sequence models with linear-time recurrence; an efficient alternative to attention, now used in hybrid layouts.

**Masked language modeling (MLM)** — Predicting randomly masked tokens from bidirectional context; BERT's pretraining objective.

**mAP** — See **AP / mAP**.

**Matting** — Estimating a soft alpha matte (and foreground color) to composite objects; harder than segmentation because of fractional opacity (hair, edges). See ZIM.

**MHA / MQA** — Multi-Head Attention (independent K/V per head) vs. Multi-Query Attention (all heads share one K/V head, minimal KV cache).

**mIoU (mean IoU)** — IoU averaged over classes; the standard semantic-segmentation metric. See [Segmentation](#/cv/segmentation).

**Mixed precision** — Training with low-precision math (BF16/FP16) plus higher-precision master weights/accumulation for speed with stability.

**MLA (Multi-head Latent Attention)** — Compressing K/V into a shared low-rank latent to shrink the KV cache (DeepSeek).

**Mode collapse** — GAN failure where the generator produces limited variety, ignoring parts of the data distribution.

**MoE (Mixture of Experts)** — Sparse layer where a router sends each token to a few of many expert FFNs; decouples total capacity from per-token compute (report *active* vs *total* params).

**Momentum** — Accumulating past gradients to smooth and accelerate SGD.

### N

**NeRF** — Neural Radiance Fields: an MLP encoding a scene as volumetric radiance for novel-view synthesis.

**NMS (Non-Maximum Suppression)** — Post-processing that removes overlapping duplicate detections by keeping the highest-scoring box per cluster (IoU threshold). See [NMS & IoU](#/ml-coding/nms-iou).

**Normalization** — Rescaling activations (Batch/Layer/RMS/Group Norm) to stabilize optimization.

### O

**Object detection** — Localizing and classifying objects with bounding boxes; evaluated by mAP.

**OCR** — Optical Character Recognition: extracting text from images; a driver of high-resolution / AnyRes VLM design.

**Online vs offline evaluation** — Offline = held-out metrics before deployment; online = A/B / live KPIs after. Strong systems answers cover both.

**ORM (Outcome Reward Model)** — Reward model that evaluates a completed response against the final outcome; contrast with a PRM, which evaluates intermediate steps.

**Overfitting** — Model fits training noise, generalizing poorly; diagnosed by a train–val gap.

### P

**PagedAttention** — vLLM technique managing the KV cache in non-contiguous pages (like OS virtual memory) to cut fragmentation and raise throughput.

**Panoptic segmentation** — Unified task assigning every pixel a class *and* an instance id — merging semantic ("stuff") and instance ("things") segmentation. Metric: **PQ**.

**PEFT** — Parameter-Efficient Fine-Tuning: adapting a model by training a small fraction of parameters (LoRA, adapters, prefix tuning).

**Perceptual loss** — Loss in a pretrained network's feature space (e.g. VGG/LPIPS) capturing perceptual similarity better than pixel MSE.

**Perplexity** — $\exp$ of average per-token negative log-likelihood; the intrinsic LM quality metric (lower is better).

**Pipeline parallelism** — Splitting model *layers* across devices and streaming micro-batches through the stages.

**Positional encoding** — Injecting token order into the permutation-invariant transformer (sinusoidal, learned, or RoPE).

**PPO (Proximal Policy Optimization)** — Policy-gradient family using a clipped surrogate objective to limit overly large policy updates. Classic RLHF typically uses it with a value critic, whereas GRPO-family methods avoid a separate learned critic.

**PQ (Panoptic Quality)** — Panoptic metric = segmentation quality (mean IoU of matched segments) × recognition quality (F1 of segment matching).

**Precision / Recall** — Precision = TP / (TP+FP) (correctness of positives); Recall = TP / (TP+FN) (coverage of actual positives).

**PRM (Process Reward Model)** — Reward model scoring each *reasoning step*; process supervision beats outcome-only on hard reasoning (cf. *Let's Verify Step by Step*, PRM800K).

**Prompt injection** — Adversarial input that overrides intended instructions; the core LLM-application security threat.

### Q

**Quantization** — Representing weights/activations in lower precision (INT8/INT4/FP8) to cut memory and latency; post-training (PTQ) or quantization-aware (QAT).

**Query / Key / Value** — The three projections in attention: queries probe, keys are matched against, values are aggregated.

### R

**RAG (Retrieval-Augmented Generation)** — Retrieving relevant documents and conditioning generation on them to ground answers and reduce hallucination.

**RandAugment / Mixup / CutMix** — Strong data-augmentation policies mixing transforms or images to regularize vision models.

**Recall@k / Precision@k** — Ranking/retrieval metrics measuring hits within the top-$k$ results.

**Regularization** — Techniques constraining model complexity (weight decay, dropout, augmentation, early stopping) to improve generalization.

**ReLU** — Rectified Linear Unit ($\max(0,x)$); the historically dominant activation.

**Residual connection** — Adding a layer's input to its output ($x + f(x)$); enables training very deep nets by easing gradient flow.

**ResNet** — CNN built from residual blocks; the standard strong CV backbone.

**RLAIF** — RL from AI Feedback: preferences generated by an AI critic instead of humans, to scale alignment data.

**RLHF** — RL from Human Feedback: broad family of methods that derive reward or preference signals from human feedback and use them to improve a policy. The classic pipeline uses a reward model plus PPO, while deployed post-training can also use other online-RL and preference-optimization methods.

**RLVR (RL with Verifiable Rewards)** — RL that optimizes a policy using signals whose outcomes can be checked by tests, symbolic rules, or graders. The reward need not be binary or perfectly deterministic, and a verifier or harness can itself be gamed. Tülu 3 helped popularize the term.

**RMSNorm** — See **LayerNorm / RMSNorm**.

**RNN / LSTM / GRU** — Recurrent networks processing sequences step-by-step; gated variants (LSTM/GRU) mitigate vanishing gradients — largely superseded by transformers.

**RoPE (Rotary Position Embedding)** — Encoding position by rotating query/key vectors by an angle proportional to position; injects *relative* position and extrapolates to longer contexts (with scaling like YaRN/NTK).

**Reward hacking** — A policy exploiting flaws in the reward signal to score high without the intended behavior.

### S

**SAM (Segment Anything Model)** — Meta's promptable segmentation foundation model (points/boxes/masks → masks); SAM 2 adds video, SAM 3 adds promptable *concept* segmentation (text/exemplar) with a presence head.

**Self-attention** — Attention where queries, keys, and values all come from the same sequence; how transformers mix token information.

**Self-supervised learning (SSL)** — Learning from unlabeled data via pretext tasks (masking, contrastive, distillation); e.g. DINO, MAE, SimCLR.

**Semantic segmentation** — Per-pixel class labeling without distinguishing instances; metric mIoU.

**SFT (Supervised Fine-Tuning)** — Fine-tuning on curated (input, target) demonstrations; the first stage of the modern post-training stack.

**SigLIP / SigLIP 2** — CLIP-style vision–language pretraining with a *sigmoid* (pairwise) loss instead of softmax contrastive; SigLIP 2 adds self-distillation for better dense features.

**Softmax** — Function mapping logits to a probability distribution; the normalizer in classification and attention.

**Speculative decoding** — A draft model or additional heads propose multiple tokens for parallel verification by a target model. It is a common serving option that can reduce latency when acceptance rates and workloads fit; preserving the target distribution requires correct rejection or residual sampling.

**SwiGLU** — Gated feed-forward activation that multiplies a Swish branch by a linear gate; a common choice in modern transformers.

### T

**Temperature** — Softmax scaling controlling sampling randomness; higher = more diverse/creative, lower = more deterministic.

**Tensor parallelism** — Splitting individual weight matrices across devices so a single layer's matmul runs in parallel.

**Test-time compute** — Spending more inference compute (longer CoT, search, best-of-N) to raise accuracy; a scaling axis distinct from parameter count (Snell 2024).

**Tokenizer / BPE** — Splitting text into subword units (Byte-Pair Encoding, SentencePiece) mapped to embeddings.

**Top-k / top-p (nucleus) sampling** — Decoding that restricts sampling to the $k$ most-likely tokens, or the smallest set with cumulative probability $p$.

**Transformer** — Architecture built from self-attention + feed-forward blocks with residual connections and normalization; the backbone of modern NLP, vision, and multimodal models.

**Transfer learning** — Reusing knowledge from a pretrained model on a new task; the default paradigm in deep learning.

**Triplet loss** — Ranking loss on (anchor, positive, negative) enforcing that the positive is closer than the negative by a margin; used in metric/embedding learning.

### U

**U-Net** — Encoder–decoder CNN with skip connections; widely used for segmentation and early or latent-diffusion denoisers, while DiT-family backbones are also common in recent generative models.

**Underfitting** — Model too simple to capture the signal (high bias); train and val error both high.

**Unified understanding + generation** — "Any-to-any" models handling both perception and synthesis in one network (Janus-Pro, BAGEL, Show-o2).

### V

**VAE (Variational Autoencoder)** — Generative model learning a latent distribution via an encoder/decoder and the ELBO; provides the latent space for latent-diffusion models.

**Vanishing / exploding gradients** — Gradients shrinking or growing across many layers/timesteps, stalling or destabilizing training; addressed by residuals, normalization, gating, clipping.

**ViT (Vision Transformer)** — Applies a transformer to image patches as tokens; a major backbone family in modern vision.

**VLM (Vision-Language Model)** — Model that processes images or video together with text (e.g. LLaVA, Qwen-VL, InternVL). How the vision encoder, projector, and language model are connected—and how much is jointly trained—varies by model.

**VLP (Vision-Language Pretraining)** — Large-scale pretraining aligning visual and textual representations (contrastive, captioning, masked objectives) before task fine-tuning.

**vLLM** — High-throughput LLM serving engine built around PagedAttention and continuous batching.

### W

**Warmup** — Ramping the learning rate from ~0 over the first steps to stabilize early training.

**Weight decay** — Regularization that shrinks parameter magnitudes on each update. With adaptive optimizers, adding an L2 penalty to the gradient and using decoupled AdamW do not produce the same trajectory.

**World model** — A learned model of environment dynamics used to predict/plan; a proposed path to sample-efficient agents (cf. JEPA).

### X

**XLA** — Accelerated Linear Algebra: a compiler that fuses and optimizes ML graphs for GPUs/TPUs.

### Y

**YaRN / NTK scaling** — Methods for extending a model's context window by rescaling RoPE frequencies with minimal retraining.

**YOLO** — "You Only Look Once": single-stage real-time object detector family; the speed-oriented counterpart to two-stage detectors.

### Z

**Zero-shot** — Performing a task with no task-specific training examples, relying on pretrained knowledge (e.g. CLIP zero-shot classification, ZIM zero-shot matting).

**ZeRO** — DeepSpeed-family strategy that progressively shards optimizer state, gradients, and parameters to reduce per-device memory. ZeRO-3 and FSDP share the core idea of full sharding, but their implementations evolved independently.

---

> [!NOTE] Missing a term?
> This glossary grows with the book. Terms are added when a chapter introduces one worth one-line recall. See the [Changelog](#/resources/changelog) for the append convention and [Curated Resources](#/resources/open-source) for where these concepts are treated in depth.
