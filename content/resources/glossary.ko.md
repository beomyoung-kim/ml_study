# Glossary

<div class="tag-row"><span class="tag">A–Z</span><span class="tag">quick recall</span><span class="tag">CV</span><span class="tag">VLM</span><span class="tag">LLM</span><span class="tag">systems</span></div>

> [!TIP] 사용법
> 빠른 회상을 위한 한 줄 정의 — 인터뷰어가 "X가 뭐죠?"라고 물을 때 *입 밖으로* 내고 싶은 레지스터입니다. 여기 있는 명료한 버전으로 시작하고, 깊이는 링크된 챕터에서 가져오세요. 한 줄을 깔끔하게 말하지 못한다면, 아직 그 개념을 소유하지 못한 겁니다. 용어는 첫 글자로 묶었고, acronym은 해당 이니셜 아래에 둡니다.

### A

**A/B test** — 무작위 배정된 treatment와 control을 실트래픽에서 비교해 metric에 대한 인과 효과를 추정하는 online 실험. 실험 단위·간섭·guardrail·통계 검정이 올바를 때 강한 배포 근거가 됨.

**Ablation** — 한 구성 요소를 제거하거나 교체해 성능에 대한 그 기여를 분리하는 것. research deep-dive에서 핵심 증거 단위.

**Activation function** — 네트워크가 비선형 함수를 모델링하게 해주는 element-wise 비선형성(ReLU, GELU, SwiGLU).

**AdamW** — gradient 기반 L2 penalty와 분리된 *decoupled* weight decay를 쓰는 Adam 변형. transformer 학습에서 흔한 선택.

**AIMv2** — autoregressive objective로 pretrain된 vision encoder 계열. 논문이 평가한 task에서는 dense/localization feature용 대안으로 비교됨.

**Alpha matte** — 전경을 배경에서 분리하는 픽셀별 opacity map($\alpha \in [0,1]$). image matting의 출력(하드한 binary mask와 대비).

**AnyRes / dynamic resolution** — 이미지를 native aspect ratio에 가까운 tile로 나눠 가변 개수의 visual token으로 처리하는 설계. OCR·문서·긴 영상에서 세부 보존과 token cost를 조절하는 데 유용.

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

**BERT** — masked language modeling으로 pretrain된 bidirectional encoder. Encoder-only language-model 계열의 대표적 기준점.

**Bias–variance tradeoff** — 기대 오차를 underfitting(bias) vs 데이터에 대한 민감도(variance)로 분해. over/underfitting을 진단하는 렌즈.

**Bounding box** — 객체를 위치시키는 축 정렬 직사각형(x, y, w, h). object detection의 출력 primitive.

**BLEU / ROUGE / METEOR** — 번역/요약을 위한 N-gram 겹침 metric. 저렴하지만 human이나 model judge에 비해 품질과의 상관이 약함.

### C

**Calibration** — 예측 confidence와 경험적 accuracy 사이의 정렬. ECE / reliability diagram으로 측정. 딥넷은 종종 과신함(overconfident).

**Catastrophic forgetting** — 새 태스크로 학습할 때 이전에 배운 태스크를 잃는 현상. continual learning의 핵심 문제(ECLIPSE 참고).

**Chain-of-thought (CoT)** — 최종 답 전에 중간 reasoning token을 생성하도록 prompting·학습·sampling하는 기법 또는 그 rationale. 모델의 내부 reasoning이 사용자에게 그대로 공개된다는 뜻은 아니며, 공개된 설명이 faithful하다고 자동 보장되지도 않음.

**CLIP** — Contrastive Language–Image Pretraining: 매칭되는 image/text 쌍이 높은 cosine similarity를 갖도록 정렬된 dual encoder. zero-shot 분류와 open-vocabulary 태스크를 가능케 함.

**CNN (convolutional neural network)** — translation equivariance와 locality를 활용하는 학습된 convolutional filter로 구성된 네트워크. 고전적 vision 백본.

**Constitutional AI** — human label 대신 명문화된 원칙("헌법")에 비추어 AI가 출력을 비평·수정하는 RLAIF 변형.

**Continual / lifelong learning** — 과거 데이터 접근이 제한된 상황에서 옛 태스크를 잊지 않고 태스크의 스트림을 학습.

**Contrastive learning** — positive 쌍은 당기고 negative는 밀어내며 representation을 학습(InfoNCE). CLIP, SimCLR, MoCo를 뒷받침.

**ControlNet** — frozen diffusion 모델에 공간적 conditioning(edge, depth, pose)을 더하는 adapter.

**Cross-attention** — query는 한 스트림에서, key/value는 다른 스트림에서 오는 attention. VLM과 encoder–decoder 모델에서 모달리티 간 융합 메커니즘.

**Cross-entropy** — target 분포 $p$ 아래에서 예측 분포 $q$의 음의 로그확률 기대값 $H(p,q)=H(p)+D_{KL}(p\|q)$. 분류에 널리 쓰이는 proper loss.

**Curriculum learning** — 수렴이나 최종 품질을 개선하기 위해 학습을 쉬운 것에서 어려운 예제 순으로 정렬.

### D

**DDP (Distributed Data Parallel)** — 모델을 GPU마다 복제하고 gradient를 all-reduce하는 data parallelism. 기본 멀티 GPU 학습 전략.

**DDPM / DDIM** — Denoising Diffusion Probabilistic Models(확률적)과 그 결정론적 implicit-sampler 변형(DDIM). diffusion 생성의 기초.

**DETR** — DEtection TRansformer: bipartite(Hungarian) matching을 통한 set prediction으로서의 end-to-end detection. anchor와 NMS를 제거.

**Dice loss** — Dice/F1 overlap 계수에 기반한 segmentation loss. foreground 크기로 정규화되어 class imbalance의 영향을 줄이는 데 도움이 되지만 작은 객체·빈 mask 처리와 calibration은 별도 확인.

**Diffusion model** — 점진적 noising process를 뒤집는 score 또는 denoising field를 학습하는 생성 모델 family. image·video·audio 생성에서 널리 사용되며, 최근에는 flow-matching 계열과 설계 공간을 공유함.

**DINO / DINOv2 / DINOv3** — self-distillation을 활용하는 self-supervised vision representation family. DINOv3는 최대 7B backbone과 Gram anchoring을 보고했으며, frozen-feature 우위는 논문이 평가한 task·protocol 범위의 결과.

**Distillation (knowledge)** — 작은 student가 큰 teacher의 출력/logits/feature를 모방하도록 학습. 배포용으로 모델을 압축.

**Distribution shift** — 학습 데이터와 배포 데이터가 다름(covariate, label, concept shift). silent한 프로덕션 성능 저하의 주요 원인.

**DPO (Direct Preference Optimization)** — reward model과 RL 루프를 건너뛰고, chosen/rejected 쌍에 대한 분류 loss로 재구성한 preference alignment.

**Dropout** — regularizer / implicit ensemble로서 학습 중 activation을 무작위로 0으로 만드는 것.

### E

**Early stopping** — overfitting을 막기 위해 validation loss가 개선을 멈추면 학습을 중단.

**ECE (Expected Calibration Error)** — bin에 걸친 confidence와 accuracy의 가중 평균 격차. 표준 calibration metric.

**Embedding** — 이산적이거나 고차원인 입력(token, image, user)을 학습된 연속 공간에 표현한 dense vector.

**EMA (exponential moving average)** — 천천히 갱신되는 weight의 running average(self-distillation의 teacher)나 statistics. 매끄럽게 하고 안정화.

**Encoder / decoder** — Encoder는 입력을 context-aware representation으로 매핑하고, decoder는 target representation 또는 sequence를 생성. GPT처럼 autoregressive한 decoder도 있지만, 모든 decoder가 autoregressive인 것은 아님. 아키텍처: encoder-only(BERT), decoder-only(GPT), encoder–decoder(T5).

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

**GELU** — Gaussian Error Linear Unit. BERT·ViT 등 많은 transformer에서 쓰인 매끄러운 activation; 최근 LLM FFN에서는 SwiGLU 같은 gated activation도 흔함.

**GEMM** — General matrix multiply. 딥넷에서 지배적인 compute 커널.

**GQA (Grouped-Query Attention)** — query head들을 더 적은 수의 key/value head group에 매핑하는 attention. MHA보다 KV cache를 줄이고 MQA보다 표현력을 남기는 절충이며, 품질 영향은 모델과 학습 recipe에 따라 다름.

**Gradient accumulation** — 메모리 한계 하에서 더 큰 batch를 흉내내기 위해, 갱신 전에 여러 micro-batch에 걸쳐 gradient를 합산.

**Gradient checkpointing** — activation을 저장하는 대신 backward pass에서 재계산해 compute를 메모리와 맞바꾸는 것.

**Gradient clipping** — 폭발하는 갱신을 막기 위해 gradient norm을 상한 처리. RNN/transformer 학습의 표준.

**Gram anchoring** — 긴 SSL 학습에서 dense feature의 열화를 막기 위해 그 Gram matrix를 regularize(DINOv3).

**GRPO (Group Relative Policy Optimization)** — 별도 value network 대신 sampled completion *group*을 이용해 advantage를 추정하는 critic-free policy-optimization family. RLVR의 여러 선택지 중 하나.

**GSPO (Group Sequence Policy Optimization)** — importance ratio를 *sequence* 수준으로 옮긴 group-relative policy-optimization 변형. 제안 논문은 token-level 방식보다 MoE RL 안정성이 나아졌다고 보고.

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

**Label smoothing** — hard one-hot target을 부드러운 target으로 바꾸는 regularization. 일반화에 도움이 될 수 있지만 calibration을 항상 개선하는 것은 아님.

**LayerNorm / RMSNorm** — 보통 sample/token별 feature 차원에 걸쳐 activation scale을 normalize. RMSNorm은 mean-centering을 생략하며 현대 LLM에서 흔함.

**Learning rate schedule** — LR이 어떻게 변하는지(warmup, cosine decay). 학습 품질의 주요 결정 요인. [lr-schedule widget](#/foundations/optimization)을 보세요.

**LoRA / QLoRA** — Low-Rank Adaptation: base weight를 freeze하고 작은 low-rank update를 학습. QLoRA는 quantized base 위에서 adapter를 학습하는, 널리 쓰이는 parameter-efficient fine-tuning 방법.

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

**ORM (Outcome Reward Model)** — 완성된 전체 응답을 최종 outcome 기준으로 평가하는 reward model. 각 중간 step을 평가하는 PRM과 대비.

**Overfitting** — 모델이 학습 노이즈를 맞춰 일반화가 나쁨. train–val 격차로 진단.

### P

**PagedAttention** — 단편화를 줄이고 throughput을 높이기 위해 KV cache를 (OS 가상 메모리처럼) 비연속 page로 관리하는 vLLM 기법.

**Panoptic segmentation** — 모든 픽셀에 클래스 *그리고* instance id를 할당하는 통합 태스크 — semantic("stuff")과 instance("things") segmentation을 병합. Metric: **PQ**.

**PEFT** — Parameter-Efficient Fine-Tuning: 파라미터의 작은 일부만 학습해 모델을 적응(LoRA, adapters, prefix tuning).

**Perceptual loss** — pretrain된 네트워크의 feature 공간(예: VGG/LPIPS)에서의 loss. pixel MSE보다 지각적 유사성을 더 잘 포착.

**Perplexity** — 평균 per-token 음의 로그우도의 $\exp$. 내재적 LM 품질 metric(낮을수록 좋음).

**Pipeline parallelism** — 모델 *layer*를 device에 걸쳐 나누고 micro-batch를 stage로 흘려보냄.

**Positional encoding** — permutation-invariant한 transformer에 token 순서를 주입(sinusoidal, learned, RoPE).

**PPO (Proximal Policy Optimization)** — policy update가 지나치게 커지지 않도록 clipped surrogate objective를 쓰는 policy-gradient family. 고전적 RLHF에서는 보통 value critic과 함께 사용하며, GRPO 계열은 별도 learned critic을 두지 않음.

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

**RLHF** — RL from Human Feedback: human feedback으로 reward·preference signal을 만들고 policy를 개선하는 넓은 계열. 고전적 pipeline은 reward model + PPO지만, 실제 post-training에는 다른 online RL·preference optimization도 쓰임.

**RLVR (RL with Verifiable Rewards)** — test·symbolic rule·grader처럼 결과를 검사할 수 있는 신호로 policy를 최적화하는 RL. reward가 꼭 binary·완전 결정적일 필요는 없고, verifier/harness도 game될 수 있음. Tülu 3가 이 표현을 널리 알림.

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

**Speculative decoding** — 작은 draft model 또는 추가 head가 여러 token을 제안하고 target model이 병렬 검증하는 방식. acceptance rate와 workload가 맞을 때 latency를 줄이는 흔한 serving 선택지이며, distribution 보존에는 올바른 rejection/residual sampling이 필요.

**SwiGLU** — Swish branch와 linear gate를 곱하는 gated FFN activation. 현대 transformer에서 흔한 선택.

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

**U-Net** — skip connection을 갖는 encoder–decoder CNN. segmentation과 초기/latent diffusion denoiser에서 널리 쓰였으며, 최근 생성 모델에는 DiT 계열도 흔함.

**Underfitting** — 신호를 포착하기엔 모델이 너무 단순함(high bias). train과 val 오차가 모두 높음.

**Unified understanding + generation** — 지각과 합성을 한 네트워크에서 다루는 "Any-to-any" 모델(Janus-Pro, BAGEL, Show-o2).

### V

**VAE (Variational Autoencoder)** — encoder/decoder와 ELBO를 통해 latent 분포를 학습하는 생성 모델. latent-diffusion 모델을 위한 latent 공간을 제공.

**Vanishing / exploding gradients** — gradient가 많은 layer/timestep에 걸쳐 줄어들거나 커져 학습이 멈추거나 불안정해지는 것. residual, normalization, gating, clipping으로 해결.

**ViT (Vision Transformer)** — image patch를 token으로 삼아 transformer를 적용하는, 현대 vision의 주요 backbone family.

**VLM (Vision-Language Model)** — image/video와 text를 함께 처리하는 모델(예: LLaVA, Qwen-VL, InternVL). vision encoder·projector·language model을 잇는 방식과 joint-training 범위는 모델마다 다름.

**VLP (Vision-Language Pretraining)** — 태스크 fine-tuning 전에 시각과 텍스트 representation을 정렬하는 대규모 pretraining(contrastive, captioning, masked objective).

**vLLM** — PagedAttention과 continuous batching을 중심으로 지은 고throughput LLM 서빙 엔진.

### W

**Warmup** — 초기 학습을 안정화하기 위해 처음 몇 step에 걸쳐 learning rate를 ~0에서 끌어올림.

**Weight decay** — update마다 parameter 크기를 줄이는 regularization. adaptive optimizer에서는 gradient에 L2 penalty를 더하는 것과 decoupled AdamW가 같은 궤적이 아님.

**World model** — 예측/계획에 쓰이는, 환경 동역학의 학습된 모델. sample-efficient한 agent로 가는 제안된 경로(cf. JEPA).

### X

**XLA** — Accelerated Linear Algebra: GPU/TPU를 위해 ML 그래프를 fuse하고 최적화하는 컴파일러.

### Y

**YaRN / NTK scaling** — 최소한의 재학습으로 RoPE frequency를 rescale해 모델의 context window를 확장하는 방법.

**YOLO** — "You Only Look Once": 단일 stage 실시간 object detector 계열. 2-stage detector의 속도 지향 대응편.

### Z

**Zero-shot** — 태스크별 학습 예제 없이, pretrain된 지식에 의존해 태스크를 수행(예: CLIP zero-shot 분류, ZIM zero-shot matting).

**ZeRO** — optimizer state·gradient·parameter를 단계별로 shard해 per-device memory를 줄이는 DeepSpeed 계열 전략. ZeRO-3와 FSDP는 full-sharding이라는 핵심 아이디어가 유사하지만 구현은 독립적으로 발전함.

---

> [!NOTE] 용어가 빠졌나요?
> 이 glossary는 책과 함께 자랍니다. 챕터가 한 줄 회상 가치가 있는 용어를 도입할 때 추가됩니다. append 관례는 [Changelog](#/resources/changelog)를, 이 개념들이 깊이 다뤄지는 곳은 [Curated Resources](#/resources/open-source)를 보세요.
