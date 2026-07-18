# Glossary

<div class="tag-row"><span class="tag">A–Z</span><span class="tag">quick recall</span><span class="tag">CV</span><span class="tag">VLM</span><span class="tag">LLM</span><span class="tag">systems</span></div>

> [!TIP] 사용법
> 빠른 회상을 위한 한 줄 정의 — 인터뷰어가 "X가 뭐죠?"라고 물을 때 *입 밖으로* 내고 싶은 레지스터입니다. 여기 있는 명료한 버전으로 시작하고, 깊이는 링크된 챕터에서 가져오세요. 한 줄을 깔끔하게 말하지 못한다면, 아직 그 개념을 소유하지 못한 겁니다. 용어는 첫 글자로 묶었고, acronym은 해당 이니셜 아래에 둡니다.

### A

**A/B test** — treatment를 control과 실트래픽에서 비교해 비즈니스 metric에 대한 인과 효과를 측정하는 online 실험. 배포 결정의 골드 스탠더드.

**Ablation** — 한 구성 요소를 제거하거나 교체해 성능에 대한 그 기여를 분리하는 것. research deep-dive에서 핵심 증거 단위.

**Activation function** — 네트워크가 비선형 함수를 모델링하게 해주는 element-wise 비선형성(ReLU, GELU, SwiGLU).

**AdamW** — *decoupled* weight decay를 쓰는 Adam optimizer. transformer 학습의 기본값.

**AIMv2** — Autoregressive로 pretrain된 vision encoder. 강한 dense/localization feature를 주는 CLIP/SigLIP의 대안.

**Alpha matte** — 전경을 배경에서 분리하는 픽셀별 opacity map($\alpha \in [0,1]$). image matting의 출력(하드한 binary mask와 대비).

**AnyRes / dynamic resolution** — 이미지를 native aspect ratio로 tiling해 처리하며 가변 개수의 visual token을 만드는 것. OCR, 문서, 비디오에 결정적.

**ANN (approximate nearest neighbor)** — 정확성을 속도와 맞바꾸는 sub-linear vector search(HNSW, IVF, ScaNN). RAG와 embedding search의 검색 백본.

**AP / mAP** — Average Precision = precision–recall 곡선 아래 면적. **mean AP**는 클래스(및 IoU threshold, 예: COCO의 mAP@[.5:.95])에 걸쳐 평균낸 것. 표준 detection metric.

**AR (autoregressive)** — 시퀀스를 한 번에 한 token씩, 각각을 prefix에 조건화해 생성. GPT 계열 모델의 decoding 패러다임.

**Attention** — value의 가중합을 계산하는 메커니즘으로, 가중치는 query–key 유사도에서 나옴($\text{softmax}(QK^\top/\sqrt{d})V$). transformer의 핵심 연산.

**AUC / ROC-AUC** — ROC 곡선 아래 면적. 무작위 positive가 무작위 negative보다 높게 랭크될 확률 — threshold와 무관한 ranking metric.

**Augmentation** — 유효 데이터를 늘리고 regularize하는 label-preserving 입력 변환(crop, flip, color jitter, Mixup, RandAugment).

### B

**Backpropagation** — chain rule을 통해 모든 파라미터에 대한 loss gradient를 계산하는 reverse-mode 자동 미분.

**Batch normalization (BN)** — 학습을 안정화·가속하기 위해 batch 차원에 걸쳐 activation을 normalize. 학습 시와 inference 시 동작이 다름(running stats).

**Beam search** — 각 단계에서 상위 $k$개의 부분 시퀀스를 유지하는 decoding. greedy보다 likelihood가 높지만 밋밋해질 수 있음.

**BERT** — masked language modeling으로 pretrain된 bidirectional encoder. 이해 태스크를 위한 정전의 encoder-only 모델.

**Bias–variance tradeoff** — 기대 오차를 underfitting(bias) vs 데이터에 대한 민감도(variance)로 분해. over/underfitting을 진단하는 렌즈.

**Bounding box** — 객체를 위치시키는 축 정렬 직사각형(x, y, w, h). object detection의 출력 primitive.

**BLEU / ROUGE / METEOR** — 번역/요약을 위한 N-gram 겹침 metric. 저렴하지만 human이나 model judge에 비해 품질과의 상관이 약함.

### C

**Calibration** — 예측 confidence와 경험적 accuracy 사이의 정렬. ECE / reliability diagram으로 측정. 딥넷은 종종 과신함(overconfident).

**Catastrophic forgetting** — 새 태스크로 학습할 때 이전에 배운 태스크를 잃는 현상. continual learning의 핵심 문제(ECLIPSE 참고).

**Chain-of-thought (CoT)** — 답 이전에 중간 reasoning 단계를 내놓도록 모델을 prompting하거나 학습하는 것. reasoning 모델의 기반(substrate).

**CLIP** — Contrastive Language–Image Pretraining: 매칭되는 image/text 쌍이 높은 cosine similarity를 갖도록 정렬된 dual encoder. zero-shot 분류와 open-vocabulary 태스크를 가능케 함.

**CNN (convolutional neural network)** — translation equivariance와 locality를 활용하는 학습된 convolutional filter로 구성된 네트워크. 고전적 vision 백본.

**Constitutional AI** — human label 대신 명문화된 원칙("헌법")에 비추어 AI가 출력을 비평·수정하는 RLAIF 변형.

**Continual / lifelong learning** — 과거 데이터 접근이 제한된 상황에서 옛 태스크를 잊지 않고 태스크의 스트림을 학습.

**Contrastive learning** — positive 쌍은 당기고 negative는 밀어내며 representation을 학습(InfoNCE). CLIP, SimCLR, MoCo를 뒷받침.

**ControlNet** — frozen diffusion 모델에 공간적 conditioning(edge, depth, pose)을 더하는 adapter.

**Cross-attention** — query는 한 스트림에서, key/value는 다른 스트림에서 오는 attention. VLM과 encoder–decoder 모델에서 모달리티 간 융합 메커니즘.

**Cross-entropy** — 예측 분포와 실제 분포 사이의 divergence를 재는 loss. 기본 분류 목적함수.

**Curriculum learning** — 수렴이나 최종 품질을 개선하기 위해 학습을 쉬운 것에서 어려운 예제 순으로 정렬.

### D

**DDP (Distributed Data Parallel)** — 모델을 GPU마다 복제하고 gradient를 all-reduce하는 data parallelism. 기본 멀티 GPU 학습 전략.

**DDPM / DDIM** — Denoising Diffusion Probabilistic Models(확률적)과 그 결정론적 implicit-sampler 변형(DDIM). diffusion 생성의 기초.

**DETR** — DEtection TRansformer: bipartite(Hungarian) matching을 통한 set prediction으로서의 end-to-end detection. anchor와 NMS를 제거.

**Dice loss** — Dice/F1 겹침 계수에 기반한 segmentation loss. 전경–배경 클래스 불균형에 강건.

**Diffusion model** — 점진적 noising 과정을 역행하는 법을 학습하는 생성 모델. image/video/audio 합성에서 SOTA.

**DINO / DINOv2 / DINOv3** — self-distillation을 통한 self-supervised ViT 학습. DINOv3는 *frozen* feature가 특화된 dense-task 모델을 능가하는 7B 백본(Gram anchoring이 긴 학습을 안정화).

**Distillation (knowledge)** — 작은 student가 큰 teacher의 출력/logits/feature를 모방하도록 학습. 배포용으로 모델을 압축.

**Distribution shift** — 학습 데이터와 배포 데이터가 다름(covariate, label, concept shift). silent한 프로덕션 성능 저하의 주요 원인.

**DPO (Direct Preference Optimization)** — reward model과 RL 루프를 건너뛰고, chosen/rejected 쌍에 대한 분류 loss로 재구성한 preference alignment.

**Dropout** — regularizer / implicit ensemble로서 학습 중 activation을 무작위로 0으로 만드는 것.

### E

**Early stopping** — overfitting을 막기 위해 validation loss가 개선을 멈추면 학습을 중단.

**ECE (Expected Calibration Error)** — bin에 걸친 confidence와 accuracy의 가중 평균 격차. 표준 calibration metric.

**Embedding** — 이산적이거나 고차원인 입력(token, image, user)을 학습된 연속 공간에 표현한 dense vector.

**EMA (exponential moving average)** — 천천히 갱신되는 weight의 running average(self-distillation의 teacher)나 statistics. 매끄럽게 하고 안정화.

**Encoder / decoder** — Encoder는 입력을 representation으로 매핑(이해). decoder는 출력을 autoregressive하게 생성(생성). 아키텍처: encoder-only(BERT), decoder-only(GPT), encoder–decoder(T5).

**Expert parallelism** — MoE expert를 여러 device에 sharding해 각각이 부분집합을 갖게 하는 것. MoE에 특유한 parallelism 차원.

### F

**F1 score** — precision과 recall의 조화 평균. 클래스 불균형 하에서 단일 숫자로 즐겨 쓰임.

**Few-shot / in-context learning** — weight 갱신 없이, prompt에 놓인 소수의 예제로 태스크를 푸는 것.

**FID (Fréchet Inception Distance)** — 실제와 생성된 이미지 feature 분포 사이의 거리. 표준 image-generation 품질 metric(낮을수록 좋음).

**Fine-tuning** — pretrain된 모델을 타깃 태스크/데이터셋에서 계속 학습. full 또는 parameter-efficient(LoRA).

**FlashAttention** — HBM에 $N\times N$ 행렬을 실체화하지 않도록 계산을 tiling하는 IO-aware exact attention 커널. FA-2/3/4가 연속된 GPU 세대를 따라감.

**Focal loss** — 쉬운 예제를 down-weight($(1-p)^\gamma$)해 어려운 것에 학습을 집중시키는 cross-entropy. detection의 극단적 전경–배경 불균형을 위해 설계됨.

**FLOPs** — 부동소수점 연산 수. 하드웨어에 무관한 compute 비용 proxy.

**FP8 / FP16 / BF16 / FP4** — 수치 범위/정밀도를 속도·메모리와 맞바꾸는 저정밀 부동소수점 포맷. FP8은 일상적이며 4-bit(NVFP4/MXFP4)는 이제 pretraining에 도달.

**FSDP (Fully Sharded Data Parallel)** — 큰 모델을 담기 위해 파라미터, gradient, optimizer state를 GPU에 걸쳐 sharding(ZeRO-3 스타일).

### G

**GAN** — Generative Adversarial Network: minimax 게임으로 학습되는 generator와 discriminator. mode collapse와 학습 불안정에 취약.

**GELU** — Gaussian Error Linear Unit. transformer의 표준인 매끄러운 activation.

**GEMM** — General matrix multiply. 딥넷에서 지배적인 compute 커널.

**GQA (Grouped-Query Attention)** — 여러 query head가 하나의 key/value head를 공유하는 attention. 품질 손실을 거의 없이 KV cache를 줄임(MHA와 MQA 사이).

**Gradient accumulation** — 메모리 한계 하에서 더 큰 batch를 흉내내기 위해, 갱신 전에 여러 micro-batch에 걸쳐 gradient를 합산.

**Gradient checkpointing** — activation을 저장하는 대신 backward pass에서 재계산해 compute를 메모리와 맞바꾸는 것.

**Gradient clipping** — 폭발하는 갱신을 막기 위해 gradient norm을 상한 처리. RNN/transformer 학습의 표준.

**Gram anchoring** — 긴 SSL 학습에서 dense feature의 열화를 막기 위해 그 Gram matrix를 regularize(DINOv3).

**GRPO (Group Relative Policy Optimization)** — value network 대신 샘플링된 completion의 *그룹*에서 advantage를 추정하는 critic-free RL. 사실상의 RLVR 알고리즘.

**GSPO (Group Sequence Policy Optimization)** — MoE 모델에서 안정적인 RL을 위해 importance ratio를 *sequence* 수준으로 옮긴 GRPO 변형.

### H

**Hallucination** — 유창하지만 거짓이거나 근거 없는 내용을 모델이 생성하는 것. grounding, RAG, verification으로 통제.

**HNSW** — Hierarchical Navigable Small World 그래프. 빠른 vector search를 위한 선도적 ANN 인덱스.

**Hungarian matching** — 최적 bipartite 할당(prediction ↔ ground truth). DETR 계열 모델의 set-prediction loss에 사용.

**Hyperparameter** — gradient descent로 학습되지 않고 search/tuning으로 고르는 설정(learning rate, batch size, depth).

### I

**ImageNet** — 128만 이미지, 1000 클래스 benchmark. vision의 역사적 pretraining/eval 표준.

**InfoNCE** — 하나의 positive를 다수의 negative에 대해 분류 문제로 다루는 contrastive loss. CLIP과 SimCLR 뒤의 목적함수.

**Instruction tuning** — base 모델이 지시를 따르도록 (instruction, response) 쌍에 대해 하는 supervised fine-tuning.

**IoU (Intersection over Union)** — 겹침 metric = 교집합 면적 ÷ 합집합 면적. detection matching과 segmentation 품질의 기반. [NMS & IoU](#/ml-coding/nms-iou)를 보세요.

### J

**Jailbreak** — 모델의 safety guardrail을 우회하도록 설계된 prompt. prompt injection의 부분집합.

**JEPA (Joint-Embedding Predictive Architecture)** — 마스킹된 영역의 픽셀이 아닌 representation을 예측하는 self-supervised 접근. LeCun이 제안한 world model로 가는 경로(I-JEPA, V-JEPA).

### K

**KL divergence** — 한 분포가 다른 분포로부터 얼마나 벗어나는지에 대한 비대칭 측도. RLHF/DPO policy를 reference 모델 근처에 유지하는 regularizer.

**KV cache** — autoregressive decoding이 재계산을 피하도록 저장해둔 이전 token의 key/value tensor. 주요 inference 메모리 비용이며, GQA/MQA/MLA와 paging이 이를 겨냥.

**k-NN / k-means** — 최근접 이웃 분류 / centroid 기반 클러스터링. 고전적 non-parametric 및 unsupervised baseline.

### L

**Label smoothing** — calibration과 일반화를 개선하기 위해 하드한 one-hot target을 약간 부드럽게 한 것으로 대체.

**LayerNorm / RMSNorm** — token마다 feature 차원에 걸쳐 activation을 normalize. RMSNorm은 효율을 위해 mean-centering을 빼며 현대 LLM의 표준.

**Learning rate schedule** — LR이 어떻게 변하는지(warmup, cosine decay). 학습 품질의 주요 결정 요인. [lr-schedule widget](#/foundations/optimization)을 보세요.

**LoRA / QLoRA** — Low-Rank Adaptation: base weight를 freeze하고 작은 low-rank 갱신 행렬을 학습. QLoRA는 base의 4-bit quantization을 더함 — 지배적인 parameter-efficient fine-tuning 방법.

**Logits** — 분류기나 LM head가 출력하는 pre-softmax raw score.

**Loss landscape** — 파라미터 공간에 걸친 loss의 기하. 평탄함/날카로움이 일반화와 관련.

### M

**Mamba / SSM** — linear-time recurrence를 갖는 state-space sequence 모델. attention의 효율적 대안으로, 이제 hybrid 레이아웃에 사용.

**Masked language modeling (MLM)** — bidirectional 맥락에서 무작위로 마스킹된 token을 예측. BERT의 pretraining 목적함수.

**mAP** — **AP / mAP**를 보세요.

**Matting** — 객체를 합성하기 위해 soft alpha matte(와 foreground color)를 추정. 분수 opacity(머리카락, 가장자리) 때문에 segmentation보다 어려움. ZIM을 보세요.

**MHA / MQA** — Multi-Head Attention(head마다 독립적 K/V) vs Multi-Query Attention(모든 head가 하나의 K/V head 공유, KV cache 최소).

**mIoU (mean IoU)** — 클래스에 걸쳐 평균낸 IoU. 표준 semantic-segmentation metric. [Segmentation](#/cv/segmentation)을 보세요.

**Mixed precision** — 속도와 안정성을 위해 저정밀 연산(BF16/FP16)에 고정밀 master weight/accumulation을 더해 학습.

**MLA (Multi-head Latent Attention)** — KV cache를 줄이기 위해 K/V를 공유 low-rank latent로 압축(DeepSeek).

**Mode collapse** — generator가 데이터 분포의 일부를 무시하고 제한된 다양성만 생성하는 GAN 실패.

**MoE (Mixture of Experts)** — router가 각 token을 다수의 expert FFN 중 소수로 보내는 sparse layer. 전체 capacity를 per-token compute에서 분리(*active* vs *total* 파라미터를 보고).

**Momentum** — SGD를 매끄럽게 하고 가속하기 위해 과거 gradient를 누적.

### N

**NeRF** — Neural Radiance Fields: 새로운 시점 합성을 위해 장면을 volumetric radiance로 인코딩하는 MLP.

**NMS (Non-Maximum Suppression)** — 클러스터마다 가장 높은 점수의 box를 유지(IoU threshold)해 겹치는 중복 detection을 제거하는 후처리. [NMS & IoU](#/ml-coding/nms-iou)를 보세요.

**Normalization** — 최적화를 안정화하기 위해 activation을 rescale(Batch/Layer/RMS/Group Norm).

### O

**Object detection** — bounding box로 객체를 위치·분류. mAP로 평가.

**OCR** — Optical Character Recognition: 이미지에서 텍스트를 추출. high-resolution / AnyRes VLM 설계의 동인.

**Online vs offline evaluation** — Offline = 배포 전 held-out metric. online = 배포 후 A/B / 실시간 KPI. 강한 systems 답변은 둘 다 다룸.

**ORM (Outcome Reward Model)** — *최종* 답만 점수 매기는 reward model. PRM(per-step)과 대비.

**Overfitting** — 모델이 학습 노이즈를 맞춰 일반화가 나쁨. train–val 격차로 진단.

### P

**PagedAttention** — 단편화를 줄이고 throughput을 높이기 위해 KV cache를 (OS 가상 메모리처럼) 비연속 page로 관리하는 vLLM 기법.

**Panoptic segmentation** — 모든 픽셀에 클래스 *그리고* instance id를 할당하는 통합 태스크 — semantic("stuff")과 instance("things") segmentation을 병합. Metric: **PQ**.

**PEFT** — Parameter-Efficient Fine-Tuning: 파라미터의 작은 일부만 학습해 모델을 적응(LoRA, adapters, prefix tuning).

**Perceptual loss** — pretrain된 네트워크의 feature 공간(예: VGG/LPIPS)에서의 loss. pixel MSE보다 지각적 유사성을 더 잘 포착.

**Perplexity** — 평균 per-token 음의 로그우도의 $\exp$. 내재적 LM 품질 metric(낮을수록 좋음).

**Pipeline parallelism** — 모델 *layer*를 device에 걸쳐 나누고 micro-batch를 stage로 흘려보냄.

**Positional encoding** — permutation-invariant한 transformer에 token 순서를 주입(sinusoidal, learned, RoPE).

**PPO (Proximal Policy Optimization)** — clipped-objective actor–critic RL 알고리즘. 고전적 RLHF optimizer(value network가 필요 — GRPO가 없애는 비용).

**PQ (Panoptic Quality)** — panoptic metric = segmentation quality(매칭된 segment의 mean IoU) × recognition quality(segment 매칭의 F1).

**Precision / Recall** — Precision = TP / (TP+FP)(positive의 정확성). Recall = TP / (TP+FN)(실제 positive의 커버리지).

**PRM (Process Reward Model)** — 각 *reasoning 단계*를 점수 매기는 reward model. process supervision은 어려운 reasoning에서 outcome-only를 능가(cf. *Let's Verify Step by Step*, PRM800K).

**Prompt injection** — 의도된 지시를 덮어쓰는 적대적 입력. 핵심 LLM-application 보안 위협.

### Q

**Quantization** — 메모리와 지연을 줄이기 위해 weight/activation을 저정밀(INT8/INT4/FP8)으로 표현. post-training(PTQ) 또는 quantization-aware(QAT).

**Query / Key / Value** — attention의 세 projection: query는 탐색하고, key는 대조되며, value는 집계됨.

### R

**RAG (Retrieval-Augmented Generation)** — 관련 문서를 검색하고 그것에 생성을 조건화해 답을 grounding하고 hallucination을 줄임.

**RandAugment / Mixup / CutMix** — vision 모델을 regularize하기 위해 변환이나 이미지를 섞는 강한 data-augmentation 정책.

**Recall@k / Precision@k** — 상위 $k$개 결과 안의 hit를 재는 ranking/retrieval metric.

**Regularization** — 일반화를 개선하기 위해 모델 복잡도를 제약하는 기법(weight decay, dropout, augmentation, early stopping).

**ReLU** — Rectified Linear Unit($\max(0,x)$). 역사적으로 지배적이었던 activation.

**Residual connection** — layer의 입력을 그 출력에 더하는 것($x + f(x)$). gradient 흐름을 쉽게 해 아주 깊은 네트워크의 학습을 가능케 함.

**ResNet** — residual block으로 지은 CNN. 표준의 강한 CV 백본.

**RLAIF** — RL from AI Feedback: alignment 데이터를 확장하기 위해 human 대신 AI critic이 생성한 preference.

**RLHF** — RL from Human Feedback: human preference에 reward model을 맞춘 뒤 그것에 대해 policy를 최적화(PPO). 고전적 alignment 파이프라인.

**RLVR (RL with Verifiable Rewards)** — 학습된 reward model 대신 *결정론적 verifier*(맞으면→1, 아니면 0)를 쓰는 RL. reward hacking에 강건하지만 검증 가능한 도메인(math, code)에 국한. Tülu 3가 명명.

**RMSNorm** — **LayerNorm / RMSNorm**을 보세요.

**RNN / LSTM / GRU** — 시퀀스를 단계별로 처리하는 recurrent 네트워크. gated 변형(LSTM/GRU)은 vanishing gradient를 완화 — 대체로 transformer로 대체됨.

**RoPE (Rotary Position Embedding)** — position에 비례하는 각도로 query/key vector를 회전해 position을 인코딩. *상대적* position을 주입하고 더 긴 맥락으로 extrapolate(YaRN/NTK 같은 scaling과 함께).

**Reward hacking** — 의도된 행동 없이 높은 점수를 받기 위해 policy가 reward 신호의 결함을 악용하는 것.

### S

**SAM (Segment Anything Model)** — Meta의 promptable segmentation foundation 모델(point/box/mask → mask). SAM 2는 비디오를, SAM 3는 presence head와 함께 promptable *concept* segmentation(text/exemplar)을 더함.

**Self-attention** — query, key, value가 모두 같은 시퀀스에서 오는 attention. transformer가 token 정보를 섞는 방식.

**Self-supervised learning (SSL)** — pretext 태스크(masking, contrastive, distillation)를 통해 unlabeled 데이터로 학습. 예: DINO, MAE, SimCLR.

**Semantic segmentation** — instance를 구분하지 않는 픽셀별 클래스 labeling. metric은 mIoU.

**SFT (Supervised Fine-Tuning)** — 큐레이션된 (input, target) demonstration에 대한 fine-tuning. 현대 post-training 스택의 첫 단계.

**SigLIP / SigLIP 2** — softmax contrastive 대신 *sigmoid*(pairwise) loss를 쓰는 CLIP 스타일 vision–language pretraining. SigLIP 2는 더 나은 dense feature를 위해 self-distillation을 더함.

**Softmax** — logits를 확률 분포로 매핑하는 함수. 분류와 attention의 정규화기(normalizer).

**Speculative decoding** — 작은 모델(또는 추가 head: Medusa/EAGLE/MTP)로 여러 token을 초안 작성하고 큰 모델의 한 번의 pass로 검증. 이제 기본 서빙 레이어.

**SwiGLU** — 현대 transformer FFN의 표준인 gated activation(Swish × linear gate).

### T

**Temperature** — 샘플링 무작위성을 제어하는 softmax scaling. 높을수록 더 다양/창의적, 낮을수록 더 결정론적.

**Tensor parallelism** — 하나의 layer의 matmul이 병렬로 실행되도록 개별 weight 행렬을 device에 걸쳐 나눔.

**Test-time compute** — accuracy를 높이기 위해 더 많은 inference compute(더 긴 CoT, search, best-of-N)를 쓰는 것. 파라미터 수와 구별되는 scaling 축(Snell 2024).

**Tokenizer / BPE** — 텍스트를 embedding에 매핑되는 subword 단위로 분할(Byte-Pair Encoding, SentencePiece).

**Top-k / top-p (nucleus) sampling** — 샘플링을 가장 가능성 높은 $k$개 token으로, 또는 누적 확률이 $p$가 되는 가장 작은 집합으로 제한하는 decoding.

**Transformer** — residual connection과 normalization을 갖춘 self-attention + feed-forward 블록으로 지은 아키텍처. 현대 NLP, vision, multimodal 모델의 백본.

**Transfer learning** — pretrain된 모델의 지식을 새 태스크에서 재사용. 딥러닝의 기본 패러다임.

**Triplet loss** — positive가 negative보다 margin만큼 가깝도록 강제하는 (anchor, positive, negative)에 대한 ranking loss. metric/embedding learning에 사용.

### U

**U-Net** — skip connection을 갖는 encoder–decoder CNN. segmentation과 diffusion denoiser 백본의 표준.

**Underfitting** — 신호를 포착하기엔 모델이 너무 단순함(high bias). train과 val 오차가 모두 높음.

**Unified understanding + generation** — 지각과 합성을 한 네트워크에서 다루는 "Any-to-any" 모델(Janus-Pro, BAGEL, Show-o2).

### V

**VAE (Variational Autoencoder)** — encoder/decoder와 ELBO를 통해 latent 분포를 학습하는 생성 모델. latent-diffusion 모델을 위한 latent 공간을 제공.

**Vanishing / exploding gradients** — gradient가 많은 layer/timestep에 걸쳐 줄어들거나 커져 학습이 멈추거나 불안정해지는 것. residual, normalization, gating, clipping으로 해결.

**ViT (Vision Transformer)** — 이미지 patch를 token으로 삼아 transformer를 적용. 지배적인 현대 vision 백본.

**VLM (Vision-Language Model)** — 이미지(+비디오)와 텍스트를 함께 처리하는 모델(예: LLaVA, Qwen-VL, InternVL). 프론티어 모델은 frozen LLM에 vision encoder를 붙이는 대신 모달리티를 함께 pretrain함.

**VLP (Vision-Language Pretraining)** — 태스크 fine-tuning 전에 시각과 텍스트 representation을 정렬하는 대규모 pretraining(contrastive, captioning, masked objective).

**vLLM** — PagedAttention과 continuous batching을 중심으로 지은 고throughput LLM 서빙 엔진.

### W

**Warmup** — 초기 학습을 안정화하기 위해 처음 몇 step에 걸쳐 learning rate를 ~0에서 끌어올림.

**Weight decay** — 매 step weight를 줄이는 L2 스타일 페널티. regularize하며, decoupled(AdamW) 형태가 transformer의 표준.

**World model** — 예측/계획에 쓰이는, 환경 동역학의 학습된 모델. sample-efficient한 agent로 가는 제안된 경로(cf. JEPA).

### X

**XLA** — Accelerated Linear Algebra: GPU/TPU를 위해 ML 그래프를 fuse하고 최적화하는 컴파일러.

### Y

**YaRN / NTK scaling** — 최소한의 재학습으로 RoPE frequency를 rescale해 모델의 context window를 확장하는 방법.

**YOLO** — "You Only Look Once": 단일 stage 실시간 object detector 계열. 2-stage detector의 속도 지향 대응편.

### Z

**Zero-shot** — 태스크별 학습 예제 없이, pretrain된 지식에 의존해 태스크를 수행(예: CLIP zero-shot 분류, ZIM zero-shot matting).

**ZeRO** — per-GPU 메모리를 줄이는 optimizer-state/gradient/parameter sharding(DeepSpeed). ZeRO-3가 FSDP의 기반.

---

> [!NOTE] 용어가 빠졌나요?
> 이 glossary는 책과 함께 자랍니다. 챕터가 한 줄 회상 가치가 있는 용어를 도입할 때 추가됩니다. append 관례는 [Changelog](#/resources/changelog)를, 이 개념들이 깊이 다뤄지는 곳은 [Curated Resources](#/resources/open-source)를 보세요.
