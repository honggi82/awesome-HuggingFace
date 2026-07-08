# arXiv:2604.11035v1[cs.AI]13Apr2026

## Introspective Diffusion Language Models

Yifan Yu∗,1,2, Yuqing Jian∗,1, Junxiong Wang1, Zhongzhu Zhou1, Donglin Zhuang1, Xinyu Fang1, Sri Yanamandra1, Qingyang Wu1, Tri Dao1,4 Xiaoxia Wu1, Shuaiwen Leon Song1, Ben Athiwaratkun1, James Zou†,1,5, Fan Lai†,♢,2, Chenfeng Xu†,♢,1,3

1Together AI 2University of Illinois Urbana-Champaign 3The University of Texas at Austin 4Princeton University 5Stanford University

∗Equal contribution †Equal advising ♢Corresponding author

#### Abstract

Diffusion language models promise parallel generation, yet still lag behind autoregressive (AR) models in quality. We stem this gap to a failure of introspective consistency: AR models agree with their own generations, while DLMs often do not. We define the introspective acceptance rate, which measures whether a model accepts its previously generated tokens. This reveals why AR training has a structural advantage: causal masking and logit shifting implicitly enforce introspective consistency. Motivated by this observation, we introduce Introspective Diffusion Language Model (I-DLM), a paradigm that retains diffusion-style parallel decoding while inheriting the introspective consistency of AR training. I-DLM uses a novel introspective strided decoding (ISD) algorithm, which enables the model to verify previously generated tokens while advancing new ones in the same forward pass. From a systems standpoint, we build I-DLM inference engine on AR-inherited optimizations and further customize it with a stationary-batch scheduler. To the best of our knowledge, I-DLM is the first DLM to match the quality of its same-scale AR counterpart while outperforming prior DLMs in both model quality and practical serving efficiency across 15 benchmarks. It reaches 69.6 on AIME-24 and 45.7 on LiveCodeBench-v6, exceeding LLaDA-2.1-mini (16B) by more than 26 and 15 points, respectively. Beyond quality, I-DLM is designed for the growing demand of large-concurrency serving, delivering about 3× higher throughput than prior state-of-the-art DLMs.

### Code | Website | Models 1 Introduction

Diffusion language models (DLMs) (Austin et al., 2021; Sahoo et al., 2024; Nie et al., 2025a; Cheng et al., 2025) offer an appealing alternative to autoregressive (AR) language models: by iteratively refining a block of tokens, they break the sequential bottleneck of next-token decoding and enable parallel generation. Yet this promise remains largely unrealized, as shown in Figure 1. First, a substantial quality gap between DLMs and AR models remains a key barrier to adoption (§2). Second, from an efficiency standpoint, limited system support for diffusion inference prevents DLMs from translating their theoretical parallelism into practical speedups.

The historical trajectory of DLM development is highly revealing. Across a broad line of work, from early continuous formulations (Li et al., 2022) and uniform-state DLMs (Austin et al., 2023) to discrete diffusion models (Lou et al., 2024); and from fully bidirectional attention (Nie et al., 2025b), to blockwise decoding (Cheng et al., 2025; Wu et al., 2025) and causal-mask decoding (Liu et al., 2025a), each generation moves closer to discrete AR language models. Increasingly AR-like training signals have also been introduced to

[Figure 1]

- Figure 1: (a) Introspective consistency: standard DLMs generate tokens whose distributions q diverge from the model’s own next-step predictions p; I-DLM trains generation and introspection to agree (p ≈ q). (b) Quality vs. throughput on MATH-500: I-DLM-8B matches Qwen3-8B (thinking) AR performance while achieving 3.1× higher throughput and +11.8 points over LLaDA-2.1-mini (16B), and 4.0× higher throughput over SDAR (8B).

improve optimization and quality (Liu et al., 2025b; Gat et al., 2025; Ye et al., 2025; Tian et al., 2025). In other words, much of the field has implicitly converged on the same intuition: the path to stronger DLMs should progressively move closer to AR models. In this work, we argue for a different trajectory. Rather than beginning from diffusion and asking how to make it more like AR, we begin from AR and ask: what is the essential principle that makes AR models so strong, and can it be preserved in a parallel generation paradigm?

We revisit this question from two perspectives: algorithms and systems. From the algorithmic perspective, this change in viewpoint leads us to re-examine a deceptively simple design at the core of AR modeling: causal masking together with logit shifting. Beyond enabling next-token prediction, this training structure implicitly teaches the model to revisit and validate its previously generated tokens under the same predictive rule. In effect, AR models are trained to agree with what they generate, exposing a fundamental limitation of existing DLMs: although they can generate tokens in parallel, they are generally not trained to agree with their own generations (e.g., due to multi-step bidirectional denoising). We formalize this gap through the notion of introspective acceptance rate, which measures whether a model internally accepts its previously generated tokens. We find that this property serves as a useful proxy for the degree to which a DLM remains consistent with its own generations.

From the systems perspective, existing DLMs are often optimized for aggressive low-latency decoding, but this comes at the cost of substantially higher computational overhead. While such overhead can be partially hidden in memory-bound regimes, production deployments requiring large-batch inference quickly hit the compute-bound regime. Unfortunately, LLM serving stacks are poorly matched to the multiple-query, multiple-denoising patterns.

Contributions. These algorithmic-system mismatches motivate us to co-design the Introspective Diffusion Language Model (I-DLM), a new paradigm that preserves the introspective consistency of AR while retaining parallelism. I-DLM is built on introspective-consistency training, an efficient recipe for converting pretrained AR models into DLMs, and a novel introspective strided decoding (ISD) algorithm that generates N tokens per forward pass while verifying prior tokens against a causal anchor distribution. We show that explicitly enforcing introspective consistency during training is key to substantially closing the quality gap between DLMs and strong same-scale AR models. We make the following contributions:

• A key insight: introspective consistency is the missing principle in prior DLMs. We show that diffusion language models do not inherit the introspective consistency of AR models: they are not trained to agree with their own generations. This missing property fundamentally limits their ability to realize the full capability of the underlying model.

Qwen3-8B (AR)

1.000

I-DLM (Ours)

0.984

LLaDA2.1 (w/ edit)

0.949

LLaDA2.1 (w/o edit)

0.933

0.699

SDAR

0.568

LLaDA 2.0

0.4 0.6 0.8 1.0

Introspective Acceptance Rate

(a) Generation–introspection gap. Introspective acceptance rate (avg min(1, p/q)) between generation (q) and introspection (p) distributions. AR and Ours achieve near-perfect consistency.

I-DLM (var. query)

1.0 p=1.0

###### TokensPerForward(TPF)

4.0

3.5

p=0.9 0.9

I-DLM (fix. query)

- p=0.7

0.7

- p=0.8

3.0

TiDAR

2.5

SDAR

0.8

p=0.6 0.6

2.0

p=0.5 0.5

1.5

1.0

AR

2.5 5.0 7.5 10.0 12.5

Compute Overhead

(b) Compute economics at stride N=4. p denotes the per-token acceptance probability. ISD achieves high TPF at low overhead.

2000

###### Throughput(tokens/s)

TPS growth=549

SDAR

1500

I-DLM (Ours)

TPS growth=84

1000

500

0

1 2 3 4

Average TPF (per request)

(c) Batching efficiency. TPF vs. batch throughput (tokens/s) at batch size 8. SDAR’s throughput barely scales with TPF; I-DLM translates higher TPF into proportionally higher throughput.

- Figure 2: Bottleneck analysis. Key gaps between DLMs and AR models: (a) existing DLMs exhibit a generation–introspection gap—they can generate tokens but cannot reliably introspect on their own output, as measured by the introspection rate; (b) DLM parallel decoding consumes far more compute per token, collapsing throughput under concurrency; (c) higher TPF does not translate to proportionally higher throughput for existing DLMs.

- • A new training paradigm for high-quality parallel generation. We introduce introspective-consistency training, a simple yet effective recipe for converting pretrained AR models into introspective DLMs (e.g., just using 5B tokens). By explicitly enforcing model’s introspective consistency, our method enables parallel decoding without sacrificing AR-level quality. Unlike prior approaches, it requires neither distillation schedules nor masking curricula, yielding a stable and efficient path to high-quality DLMs.
- • A novel single-pass decoding algorithm that unifies generation and verification. We propose Introspective Strided Decoding (ISD), which simultaneously generates new tokens and revises prior ones within the same forward pass. At [MASK] positions, the model proposes new tokens; at introspection positions, it revisits previous tokens against its causal anchor distribution. This yields outputs that provably match the base AR distribution, without confidence heuristics or separate verification passes.
- • An AR-compatible serving stack for deployable DLMs and self-speculative decoding We design an inference stack that is directly compatible to existing AR serving systems (e.g., SGLang). Besides, we develop a gated residual adaptation mechanism where LoRA adapters are applied only at mask positions, while verification relies on the base model weights. It supports a continuum between near-lossless and strictly lossless modes.
- • Comprehensive evidence that closes the quality gap while delivering real efficiency gains. Across 15 benchmarks, we show that I-DLM is the first DLM to match strong same-scale AR quality while substantially outperforming prior DLMs in both capability and serving efficiency. Our results establish a new quality–efficiency frontier.

We will open-source models and systems to facilitate community research and deployment.

#### 2 Background and Motivation

We trace these gaps to three inherent limitations of current DLMs (Figure 2). (1) Low introspective consistency: DLMs cannot reliably agree with their generations, making coherent reasoning difficult; (2) Compute inefficiency: both training and inference require substantially more FLOPs in tokens per forward (TPF) than AR models; and (3) Inference engine incompatibility: multi-token, multi-step denoising in DLM inference is poorly aligned with modern AR serving stacks, leading to inefficient execution.

DLMs lack introspective consistency. We formalize introspective acceptance rate α as follows: for each generated token xk sampled from distribution qk, we perform a separate forward pass with all tokens revealed to obtain the corresponding causal distribution pk and compute α = L1 ∑k min(1, pk(xk)/qk(xk)). For AR models, p = q by construction, yielding α = 1

and thus perfect generation-introspection consistency. We evaluate α on IFEval using each model’s best configuration from its official release (Figure 2a). SDAR (8B) achieves only 0.699 and LLaDA 2.0-flash (8B) only 0.568, indicating substantial divergence between what these models generate and what they would endorse upon re-examination. LLaDA 2.1 (Bie et al., 2026) is philosophically close to us. Although it is motivated as improving the revision capability of DLMs, we view revision and introspective acceptance as deeply connected: improving a model’s ability to revise its own outputs naturally improves its ability to endorse them upon re-examination. LLaDA 2.1 restructures the training and data pipeline for a token-to-token supervision format, whereas our introspective-consistency training is much simpler, jointly training generation and introspection under a unified objective without redesigning the data pipeline (§3.1). Despite this simplicity, I-DLM achieves substantially higher introspective acceptance rates with far greater token efficiency: it matches its AR base model using only 4.5B training tokens, while SDAR needs 54B tokens (12× more) and still yields much worse downstream quality (10.0 vs. 69.6 on AIME-24).

Parallel decoding does not translate to compute efficiency. Existing baselines incur more flops per decoded token, which effectively dilutes the achieved speedup by pushing the kernel to a compute bound, as shown in Figure 2b. We define compute overhead as the ratio of total FLOPs to decode a given number of tokens compared to AR decoding. We provide detailed tokens per forward (TPF) pass and compute overhead analysis given different per-token acceptance rates in Appendix B. Figure 2b shows the tradeoff at stride N=4 for I-DLM, SDAR (block diffusion model), and Tidar (DLM with branched inference methods): at a TPF of ∼2.5, I-DLM incurs only ∼2.5× compute overhead, while TiDAR requires ∼7.8×. In contrast, SDAR’s TPF is capped at 2.0. This is because block diffusion (SDAR, LLaDA) requires T denoising steps plus a mandatory KV-commit forward that produces no new tokens, capping TPF at N/2 even under ideal acceptance (p=1). Appendix C details why this overhead is hard to eliminate.

To quantify this tradeoff, we define compute efficiency as TPF/OH; efficiency > 1 means the TPF gain outweighs the overhead cost. Figure 3 plots efficiency against acceptance rate p for N=4. ISD is the only method that crosses the break-even line, reaching efficiency > 1 at p ≈ 0.83 (variable-query) and p ≈ 0.86 (fixed-query). At our empirically observed rates (p ≥ 0.85), ISD achieves 1.08–2.29× efficiency. In contrast, TiDAR’s efficiency is capped at N/(N+1) = 0.80 even at p = 1, and SDAR remains around 0.64–0.72 at practical acceptance rates.

2.5

I-DLM variable query

###### ComputeEfficiency

I-DLM fixed query

2.0

TiDAR

SDAR

1.5

Break-even

1.0

- 0.4 0.6 0.8 1.0 Per-token Acceptance Rate (p)

- 0.5

Figure 3: Compute efficiency (TPF/OH) vs. acceptance rate at N=4. ISD is the only method above the break-even line.

DLM inference is incompatible with AR serving engines. The modern LLM serving stack (e.g., continuous batching, fused attention kernels, paged KV cache) is highly optimized for AR decoding. DLMs break these assumptions. In AR serving, continuous batching works because all requests advance uniformly. In block diffusion, tokens within a block converge at different rates: some positions pass the confidence threshold early, yet all requests must synchronize at the slowest block, wasting the TPF gain. Figure 2c reflects this: SDAR’s TPS growth rate (slope=84) is nearly flat with respect to TPF. In contrast, ISD uses strict causal attention (compatible with AR kernels) and adaptive stride, where each step produces at least one quality-guaranteed token via introspection, achieving a TPS growth rate of 549 (§3.3). We show other AR serving mismatch (e.g., attention kernel mismatch) in Appendix D.

We bring up more discussion on detailed related works in Appendix A.

#### 3 Introspective Diffusion Language Model

We present I-DLM to address the aforementioned bottlenecks. Causal attention with logit shift closes the generation–introspection gap and improves training efficiency (§3.1); introspective strided decoding eliminates the compute overhead of iterative denoising (§3.2); and the preserved causal structure enables direct integration into AR serving stacks (§3.3).

##### 3.1 Introspective-Consistency Training

We convert a pretrained AR model into an introspective diffusion model by combining causal attention, logit shift, and an all-masked objective. The training mask is shown in

- Appendix E. We note that our proposed introspective-consistency training method itself is simple, but it is built on a deep and fundamental insight inspired by AR: AR training unifies generation and introspection in one forward pass; DLM training should inherit the same spirit.

Causal training with logit shift. We apply strict causal masking uniformly across both the masked and clean portions: for any query position j and key position i, attention is permitted only when i ≤ j. Unlike SDAR which uses block-causal attention (bidirectional within blocks), this holds for masked tokens in xt as well—each [MASK] attends only to preceding positions. In the clean region x0, standard causal attention is used to ensure generation and introspection operate under the same attention pattern (Section 2) and enable KV cache reuse at inference. We pair this with a logit shift: the hidden state at position i is trained to predict token i+1 rather than token i. Standard masked diffusion trains hidden[Mi] → token[i], which breaks the AR model’s inherent logits[i] → token[i+1] mapping and crucially prevents clean positions from providing a meaningful verify signal. Our shifted formulation preserves this mapping: clean positions produce the causal anchor p (the verify distribution), while masked positions produce tokens q (the decode distribution).

All-masked training with auto-balanced loss. Given a clean sequence x0 = (x1, . . . , xL), we replace all tokens with a special [MASK] token to obtain the fully masked input xt = (M, . . . ,M). The training input is the concatenation [xt | x0], where xt is the all-masked sequence and x0 provides the clean reference. Unlike standard masked diffusion training, which masks a random fraction r of tokens, wasting (1−r) of the compute on unsupervised positions, our all-masked regime ensures that every position contributes a useful training signal, eliminating the supervision dilution identified in Section 2.

We apply a cross-entropy loss with shifted labels separately to the masked and clean regions:

1

1

log pθ(x0ℓ+1 | [xt, x0]≤ℓ), Lclean = −

log pθ(x0ℓ+1 | [xt, x0]≤ℓ),

|St| ∑

|S0| ∑

Lmask = −

ℓ∈St

ℓ∈S0

(1) where St and S0 are the non-padding positions in the masked and clean regions, respectively, and x0ℓ+1 is the shifted target (next token). This is the same cross-entropy objective used in AR pretraining—no separate distillation loss or teacher model is required. On clean positions, Lclean trains the introspection pathway: the model learns to produce the causal anchor distribution pθ(xi+1 | x≤i), recovering the exact AR training objective. On masked positions, Lmask trains the decode pathway: the model learns to produce tokens q from [MASK] hidden states, enabling strided generation at stride N > 1.

Although both pathways share the same cross-entropy objective, their loss magnitudes can differ substantially: masked positions face a harder prediction task and tend to produce larger losses, especially early in training. A fixed weighting risks the decode pathway dominating the gradient, undermining the introspection pathway that is critical for verification quality. We address this with an auto-balanced loss:

L = Lmask + sˆ · Lclean, sˆ = Lmask Lclean

, (2)

where sˆ is the ratio of loss magnitudes, treated as a fixed scalar at each training step (not differentiated through). This dynamically rescales the clean-position loss to match the magnitude of the masked-position loss, ensuring both pathways receive equal effective gradient magnitude without manual tuning. Because both pathways share the same objective and receive balanced supervision, the model naturally aligns q with p, maximizing the introspective acceptance rate (Figure 2a).

A note on why previous works fail to achieve introspective consistency: While the individual ingredients of our approach, such as causal-mask training (Hu et al., 2025), logit shifting (Ye et al., 2025), and full-mask training (Liu et al., 2025b), have each been explored

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

|[Figure 13]| |[Figure 14]|[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]|[Figure 19]| |
|---|---|---|---|---|---|
|[Figure 20]<br><br>[Figure 21]| | |[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]| |[Figure 25]|
|[Figure 26]<br><br>[Figure 27]| | |[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]|[Figure 31]| |
|[Figure 32]<br><br>[Figure 33]| |[Figure 34]<br><br>[Figure 35]| |[Figure 36]<br><br>[Figure 37]| |
| |[Figure 38]|[Figure 39]|[Figure 40]|[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]| |
|[Figure 44]<br><br>[Figure 45]|[Figure 46]| | |[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]| |

|[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]| |[Figure 53]| |[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]| |
|---|---|---|---|---|---|
|[Figure 57]<br><br>[Figure 58]| |[Figure 59]| |[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]| |
| |[Figure 63]<br><br>[Figure 64]| |[Figure 65]<br><br>[Figure 66]| |[Figure 67]|
| |[Figure 68]<br><br>[Figure 69]|[Figure 70]| |[Figure 71]|[Figure 72]<br><br>[Figure 73]|
| |[Figure 74]<br><br>[Figure 75]|[Figure 76]| |[Figure 77]|[Figure 78]<br><br>[Figure 79]|
|[Figure 80]<br><br>[Figure 81]| |[Figure 82]|[Figure 83]<br><br>[Figure 84]|[Figure 85]| |

|[Figure 86]| |[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]|[Figure 90]|[Figure 91]<br><br>[Figure 92]| |
|---|---|---|---|---|---|
|[Figure 93]<br><br>[Figure 94]| |[Figure 95]|[Figure 96]|[Figure 97]<br><br>[Figure 98]| |
| |[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]| |[Figure 102]|[Figure 103]<br><br>[Figure 104]| |
|[Figure 105]<br><br>[Figure 106]<br><br>| |[Figure 107]| |[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]| |
|[Figure 111]|[Figure 112]|[Figure 113]<br><br>[Figure 114]|[Figure 115]| |[Figure 116]<br><br>[Figure 117]|
|[Figure 118]<br><br>[Figure 119]|[Figure 120]| |[Figure 121]|[Figure 122]<br><br>[Figure 123]| |

|[Figure 124]<br><br>[Figure 125]| | |[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]| |[Figure 130]|
|---|---|---|---|---|---|
| |[Figure 131]|[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]|[Figure 135]| |[Figure 136]|
| |[Figure 137]<br><br>[Figure 138]| |[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]| |[Figure 142]|
| |[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]| |[Figure 146]<br><br>|[Figure 147]<br><br>[Figure 148]|[Figure 149]|
| |[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]| | |[Figure 153]<br><br>[Figure 154]|[Figure 155]|
|[Figure 156]| |[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]| |[Figure 160]<br><br>[Figure 161]| |

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

|[Figure 219]<br><br>[Figure 220]| |[Figure 221]|[Figure 222]| |[Figure 223]<br><br>[Figure 224]|
|---|---|---|---|---|---|
| |[Figure 225]<br><br>[Figure 226]|[Figure 227]<br><br>[Figure 228]| | |[Figure 229]<br><br>[Figure 230]|
| |[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]| |[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]| |[Figure 238]|
| |[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]|[Figure 242]|[Figure 243]|[Figure 244]| |
| |[Figure 245]<br><br>[Figure 246]<br><br>[Figure 247]| |[Figure 248]|[Figure 249]<br><br>[Figure 250]| |
|[Figure 251]<br><br>[Figure 252]|[Figure 253]| |[Figure 254]<br><br>[Figure 255]|[Figure 256]| |

| |[Figure 257]<br><br>[Figure 258]| |[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]| |[Figure 262]|
|---|---|---|---|---|---|
|[Figure 263]<br><br>[Figure 264]| | |[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]| |[Figure 268]|
| |[Figure 269]|[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]| |[Figure 273]<br><br>[Figure 274]| |
| |[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]| |[Figure 278]<br><br>[Figure 279]| |[Figure 280]|
|[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]| | |[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]| |[Figure 287]<br><br>[Figure 288]|
| |[Figure 289]|[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]| |[Figure 293]<br><br>[Figure 294]| |

| |[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]|[Figure 298]| |[Figure 299]<br><br>[Figure 300]| |
|---|---|---|---|---|---|
|[Figure 301]<br><br>[Figure 302]| |[Figure 303]<br><br>[Figure 304]| |[Figure 305]<br><br>[Figure 306]| |
| |[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]| |[Figure 311]|[Figure 312]<br><br>[Figure 313]| |
|[Figure 314]<br><br>[Figure 315]|[Figure 316]|[Figure 317]| | |[Figure 318]<br><br>[Figure 319]|
|[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]|[Figure 323]|[Figure 324]| |[Figure 325]<br><br>[Figure 326]| |
|[Figure 327]<br><br>[Figure 328]| |[Figure 329]<br><br>[Figure 330]| | |[Figure 331]<br><br>[Figure 332]|

| |[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]|[Figure 336]| |[Figure 337]<br><br>[Figure 338]| |
|---|---|---|---|---|---|
| | | | | | |
|[Figure 339]<br><br>[Figure 340]| | |[Figure 341]<br><br>[Figure 342]|[Figure 343]<br><br>[Figure 344]| |
| |[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]|[Figure 348]| |[Figure 349]|[Figure 350]|
|[Figure 351]<br><br>[Figure 352]| | |[Figure 353]<br><br>[Figure 354]|[Figure 355]<br><br>[Figure 356]| |
| |[Figure 357]<br><br>[Figure 358]| |[Figure 359]<br><br>[Figure 360]|[Figure 361]|[Figure 362]|
| |[Figure 363]<br><br>[Figure 364]<br><br>[Figure 365]<br><br>[Figure 366]|[Figure 367]|[Figure 368]|[Figure 369]|[Figure 370]|

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

| |[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]| | |[Figure 422]<br><br>[Figure 423]<br><br>[Figure 424]|[Figure 425]|
|---|---|---|---|---|---|
| |[Figure 426]|[Figure 427]<br><br>[Figure 428]<br><br>[Figure 429]|[Figure 430]| |[Figure 431]|
|[Figure 432]| |[Figure 433]<br><br>[Figure 434]| |[Figure 435]<br><br>[Figure 436]|[Figure 437]|
| |[Figure 438]|[Figure 439]<br><br>[Figure 440]<br><br>[Figure 441]| |[Figure 442]|[Figure 443]|
| |[Figure 444]<br><br>[Figure 445]<br><br>[Figure 446]|[Figure 447]| |[Figure 448]|[Figure 449]|
| |[Figure 450]<br><br>[Figure 451]<br><br>[Figure 452]|[Figure 453]| |[Figure 454]<br><br>[Figure 455]| |

| |[Figure 456]<br><br>[Figure 457]|[Figure 458]|[Figure 459]<br><br>[Figure 460]|[Figure 461]| |
|---|---|---|---|---|---|
| |[Figure 462]<br><br>[Figure 463]|[Figure 464]|[Figure 465]<br><br>[Figure 466]|[Figure 467]| |
|[Figure 468]<br><br>[Figure 469]| |[Figure 470]<br><br>[Figure 471]|[Figure 472]| |[Figure 473]<br><br>[Figure 474]|
| |[Figure 475]<br><br>[Figure 476]|[Figure 477]| |[Figure 478]|[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]|
|[Figure 482]<br><br>[Figure 483]| |[Figure 484]|[Figure 485]| |[Figure 486]<br><br>[Figure 487]|
|[Figure 488]<br><br>[Figure 489]| |[Figure 490]| |[Figure 491]|[Figure 492]<br><br>[Figure 493]|

|[Figure 494]<br><br>[Figure 495]<br><br>[Figure 496]<br><br>[Figure 497]<br><br>[Figure 498]|[Figure 499]<br><br>[Figure 500]|[Figure 501]<br><br>[Figure 502]<br><br>|[Figure 503]<br><br>[Figure 504]<br><br>[Figure 505]<br><br>[Figure 506]| |
|---|---|---|---|---|
| |[Figure 507]<br><br>[Figure 508]<br><br>[Figure 509]<br><br>[Figure 510]<br><br>[Figure 511]<br><br>[Figure 512]<br><br>|[Figure 513]<br><br>[Figure 514]<br><br>| |[Figure 515]<br><br>[Figure 516]<br><br>[Figure 517]<br><br>[Figure 518]<br><br>|
| |[Figure 519]<br><br>[Figure 520]<br><br>[Figure 521]<br><br>|[Figure 522]<br><br>|[Figure 523]<br><br>[Figure 524]| |
| |[Figure 525]<br><br>[Figure 526]<br><br>[Figure 527]<br><br>|[Figure 528]<br><br>|[Figure 529]<br><br>[Figure 530]| |

|[Figure 531]| |[Figure 532]<br><br>[Figure 533]|[Figure 534]|[Figure 535]|[Figure 536]|
|---|---|---|---|---|---|
|[Figure 537]<br><br>[Figure 538]| |[Figure 539]|[Figure 540]|[Figure 541]|[Figure 542]|
|[Figure 543]<br><br>[Figure 544]| |[Figure 545]|[Figure 546]|[Figure 547]|[Figure 548]|
| |[Figure 549]<br><br>[Figure 550]<br><br>[Figure 551]| | |[Figure 552]<br><br>[Figure 553]|[Figure 554]|
|[Figure 555]<br><br>[Figure 556]| |[Figure 557]|[Figure 558]|[Figure 559]|[Figure 560]<br><br>[Figure 561]|
|[Figure 562]<br><br>[Figure 563]| |[Figure 564]|[Figure 565]|[Figure 566]|[Figure 567]|

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

Figure 4: Comparison of decoding paradigms. Our I-DLM uses strict causal attention with adaptive stride (1 < stride < N) and is a drop-in replacement within AR serving infrastructure. ISD produces a quality-guaranteed token xi+1 together with draft tokens xˆi+2:i+N via introspective strided decoding, xi+1, xˆi+2:i+N = πstrided(c1:i, m1:N−1); Residual ISD (R-ISD) additionally gates a LoRA residual for bit-for-bit lossless output: xi+1:i+N = πar(c1:i, m1:N−1) + πL(m1:N−1).

[Figure 580]

in isolation, their combined role in enabling true introspective consistency has remained largely overlooked. We ablate these components in Figure 6a: removing causal attention and logit shift (reverting to block diffusion) causes sharp degradation on reasoning tasks (e.g., HumanEval: 92.7 → 60.3), confirming that introspective consistency is critical for longhorizon generation. Specifically, causal masking ensures generation-time context consistency across denoising steps, logit shifting further bridges the verification and generation with unified hidden states and ensures training robustness by respecting the autoregressive (AR) model’s inherent behavior, and the all-masked objective ensures training efficiency through dense supervision; no single component achieves consistency, robustness, and efficiency simultaneously. Prior work such as NBDiff (Tian et al., 2025) adopts causal masking for prefilling and block-causal masking for decoding, and achieves substantial gains. This confirms that even partial generation-time consistency at the prefilling stage alone delivers notable improvements, but switching between masking schemes introduces nontrivial computational overhead at inference time. Similarly, LLaDA-2.1 shares our broader goal of revision during generation, but it depends on heavy data engineering to construct multiturn revision sequences. In contrast, our method provides a unified training pathway that ensures consistency, robustness, and efficiency simultaneously by directly distilling the AR model’s inherent behavior through an efficient single-stage training regime.

##### 3.2 Introspective Strided Decoding (ISD)

Introspective Strided Decoding (ISD) is the inference counterpart to I-DLM training. Because the model uses strict causal attention, ISD begins with a standard AR prefill over the prompt, then dynamically selects the effective stride at each decode step via the p/q acceptance criterion. Unlike block diffusion which commits to a fixed block size, or speculative decoding which requires a separate draft model, the stride in ISD adapts intrinsically based on the model’s own self-consistency: easy tokens are accepted in parallel while difficult tokens fall back toward AR-quality generation.

Single-pass stride and introspection. ISD generates tokens in steps (Algorithm 1, Figure 4 bottom): 1 Step 1 (bootstrap). We append N [MASK] tokens to the prefix and run a single forward pass. Due to the logit shift, the last clean position produces a quality-guaranteed token x1—identical to an AR prediction, requiring no introspection. This token anchors the subsequent introspection chain: at the next step, x1’s hidden state produces the causal anchor p2 for evaluating xˆ2, and so on. The remaining N−1 [MASK] positions produce strided tokens xˆ2, . . . , xˆN; and 2 Step t > 1 (stride + introspection). We fill in the accepted (or

Algorithm 1 Introspective Strided Decoding (one iteration)

Require: Prefix tokens x1:L, stride N, model M with logit shift Ensure: Accepted new tokens appended to prefix

- 1: // Stride: N−1 masks produce N tokens (logit shift gives +1 free)
- 2: Construct input [x1:L, MASK1, . . . , MASKN−1] (N−1 masks)
- 3: Run M → logitsstride[1:L+N−1]
- 4: Sample xL+1 ∼ softmax(logitsstride[L]) {quality-guaranteed (AR prediction)}
- 5: for k = 2 to N do
- 6: Sample xL+k ∼ qk ≜ softmax(logitsstride[L+k−1]) {strided proposal}
- 7: end for
- 8: // Introspect: verify proposals against causal anchors
- 9: // (Only introspection shown; in practice fused with next proposal into one 2N−1 token pass)
- 10: Construct input [x1:L, xL+1, . . . , xL+N]
- 11: Run M → logitsanchor[1:L+N]
- 12: pk ← softmax(logitsanchor[L+k−1]) for k = 1, . . . , N {causal anchor distributions}
- 13: // Introspection with adaptive stride
- 14: naccepted ← 1 {xL+1 always accepted (quality-guaranteed)}
- 15: for k = 2 to N do
- 16: Draw r ∼ Uniform(0,1)
- 17: if r < min 1, pk(xL+k) / qk(xL+k) then
- 18: Accept xL+k; naccepted ← naccepted + 1
- 19: else
- 20: Resample xL+k ∼ normalize max(0, pk − qk)
- 21: naccepted ← naccepted + 1; break
- 22: end if
- 23: end for
- 24: // Bonus token if all proposals accepted
- 25: if naccepted = N then
- 26: Sample xL+N+1 ∼ softmax(logitsanchor[L+N])
- 27: naccepted ← naccepted + 1
- 28: end if
- 29:
- 30: Append xL+1:L+naccepted to prefix; commit KV cache

resampled) tokens from the previous step, append N fresh [MASK] tokens, and run a single forward pass that simultaneously:

- • Introspects on the previous tokens, the filled-in tokens are now clean, so the logits at each position produce causal anchor distributions pk (the true AR next-token distributions);
- • Generates new tokens from the appended [MASK] positions.

Crucially, introspection piggybacks on the next stride step at zero additional cost: every step after the first produces both accepted tokens and fresh tokens in one forward pass. This contrasts with SDAR (T+1 forwards per block), WeDLM (confidence-based streaming without distributional guarantees), and TiDAR (unified draft and introspection but N(N+1) token queries per step; our logit shift reduces this to N2).

Adaptive stride via p/q acceptance. We introspect on each token by comparing it against its causal anchor using the p/q acceptance criterion (Leviathan et al., 2023): token xk sampled from qk is accepted with probability min(1, pk(xk)/qk(xk)). When accepted, the token provably follows the causal anchor distribution. On rejection, the token is resampled from the corrected distribution normalize(max(0, pk − qk)) and all subsequent tokens are discarded; the next step becomes a pure stride with no tokens to introspect. When all tokens are accepted, a bonus token is sampled from the final anchor distribution, achieving stride N+1. This mechanism naturally adapts the effective stride per step without manual tuning. The formal procedure is given in Algorithm 1, with a detailed step-by-step illustration in

- Appendix F.

Lossless ISD with residual adaptation. ISD uses a single model as both proposer and introspector, eliminating the need for a separate draft model to train, maintain, or synchronize.

While it can be used as a standalone generative model, its single-forward drafting-andverification structure also makes it a natural fit for self-speculative decoding.

Inspired by the gated LoRA approach in multi-token prediction (Samragh et al., 2025), we gate the LoRA residual with a per-token binary mask: [MASK] positions use base+LoRA weights to produce high-quality strided tokens, while introspection positions use base-modelonly weights. Critically, because the entire model uses strict causal attention, introspection positions cannot attend to token positions—the causal anchor distribution p is computed from base-only weights over a base-only KV cache, identical to a pure base AR forward pass. This guarantees that ISD introspects against the exact base AR distribution, making the output bit-for-bit lossless (Appendix G).

Theoretical speedup analysis. Let pk denote the acceptance probability of the k-th strided token, and Pk = ∏kj=1 pj the cumulative acceptance probability. For ISD at stride N, the expected tokens per forward pass (TPF) is (see Appendix B for derivation):

###### 2 + P1 + P2 + · · · + PN−2 2 − PN−1

TPFN =

. (3)

At perfect acceptance (pk = 1), TPFN = N recovers the theoretical maximum. At pk = 0, TPFN = 1 degenerates to AR. In the memory-bound decode regime, forward pass latency is approximately constant regardless of stride size, so wall-clock speedup ≈ TPF. With typical acceptance rates of p ≥ 0.85, stride N=3 achieves TPF ≈ 2.3–2.4× with a compute overhead of only ∼2×—meaning ISD translates most of the parallel decoding benefit into real throughput gain with minimal wasted compute. Our evaluations show that I-DLM achieves close-to-optimal speedup empirically (Section 4).

##### 3.3 I-DLM Serving Stack: AR-Compatible Serving

Prior DLMs rely on custom inference pipelines that forgo the optimizations accumulated in AR serving systems (§2). Because I-DLM preserves strict causal attention, we integrate it directly into SGLang as a drop-in extension, inheriting paged KV cache, continuous batching, and tensor parallelism without modification. We illustrate the key system optimizations below (ablated in §4.3).

- (1) AR-inherited optimizations. Each ISD step maps to SGLang’s native extend mode, appending 2N−1 tokens with causal attention. Unlike block diffusion—where requests within a batch must synchronize at the slowest converging block, breaking continuous batching (§2)—ISD produces at least one quality-guaranteed token every step, so all requests advance uniformly and continuous batching works unmodified. This is why I-DLM’s throughput scales proportionally with concurrency while block diffusion methods plateau (Figure 2c). The causal structure further enables two key reuses. First, we capture the entire extend forward into a single CUDA graph, replaying it each step with only input ids and attention metadata updated in-place. Second, since our extended sizes are small (≤9 tokens), we replace the three-kernel attention cascade with a single paged-only kernel per layer, eliminating 2L redundant launches (Appendix D).

- (2) Stationary-batch scheduling. ISD has a strict dependency chain: forward→verify→ trim → prepare → forward. This prevents the CPU–GPU overlap used in AR serving, making CPU overhead directly additive to step latency. We mitigate this with a stationarybatch decode loop that reuses the batch object across consecutive ISD steps, bypassing the full scheduler rebuild. Within this loop, KV slots are allocated via a single batched scatter, constant metadata is cached, and the ISD-specific KV trim-and-commit cycle frees rejected and MASK positions after each verification. Non-critical I/O is deferred to an overlap window during the next GPU forward.
- (3) Kernel fusion and proposal optimization. The verification step is fused into a single Triton kernel with online softmax and Gumbel-max correction; the common accept path (∼78% of positions) returns after one streaming pass, skipping correction entirely. Since ISD’s p/q criterion guarantees output correctness regardless of proposal quality, we use

argmax for proposals to maximize acceptance rate without affecting output diversity. For lossless R-ISD (Appendix G), we implement segment-gated LoRA where a per-token binary mask gates the LoRA residual within the CUDA graph, with cuBLAS replacing segmented GEMV at small token counts and a dedicated CUDA stream overlapping the LoRA shrink with base projections.

#### 4 Experiments

##### 4.1 Evaluation Methodology

We train two I-DLM variants: I-DLM-8B and I-DLM-32B, converted from Qwen3-8B and Qwen3-32B (Yang et al., 2025), respectively, using the all-masked causal training recipe (Section 3.1). Training uses 4.5B tokens on 8 H100 GPUs. For lossless ISD, we additionally train LoRA adapters (rank 128) on the same data. Full training details are in Appendix H.

Baselines. We compare against two classes of state-of-the-art methods. (i) Diffusion LLMs: LLaDA-2.1-mini (16B)(Bie et al., 2026), LLaDA-2.0-flash (100B)(Bie et al., 2025), LLaDA-2.1flash (100B)(Bie et al., 2026), SDAR (8B, 30B-A3B)(Cheng et al., 2025), NBDiff (7B)(Tian et al.,

- 2025), DREAM (7B)Ye et al. (2025), WeDLM (8B)Liu et al. (2025a), LightningRL (8B), TiDAR (8B), Jacobi Forcing (7B)(Hu et al., 2026), Fast-dLLM (7B)(Wu et al., 2025), Mercury Coder Small(Labs et al., 2025), and Gemini Diffusion; and (ii) Speculative decoding: EAGLE-3 (Li et al., 2025). We use our I-DLM AR counterparts, Qwen3-8B and Qwen3-32B, as baselines.

Benchmarks. We evaluate on 15 benchmarks covering four domains: (i) Knowledge and Reasoning: ARC-C(Clark et al., 2018), MMLU(Hendrycks et al., 2020), MMLU-Pro(Wang et al., 2024), GPQA-D(Rein et al., 2024), GPQA(Rein et al., 2024); (ii) Math Reasoning: GSM8K(Cobbe et al., 2021), MATH-500(Hendrycks et al., 2021), MathBench(Liu et al., 2024), AIME-24(AIME), AIME-25(AIME); (iii) Code Generation: HumanEval(Chen et al., 2021), MBPP(Odena et al., 2021), LiveCodeBench-v6(Jain et al., 2024); and (iv) Instruction Following: IFEval(Zhou et al., 2023). All with thinking mode enabled.

Metrics. For quality, we report accuracy on each benchmark with average values over three runs, where baseline results are taken from their original papers where available. For efficiency, we report request-level tokens per second (latency) and server-level tokens per second (throughput) under varying concurrency levels. Details in Appendix H.

##### 4.2 End-to-End Performance

Table 1 summarizes I-DLM’s quality; ablations on different stride settings are in §4.3.

I-DLM surpasses the quality of strongest DLMs with comparable sizes. I-DLM consistently outperforms all existing DLMs, often by large margins. At 8B scale, I-DLM-8B exceeds LLaDA-2.1-mini (16B) across all benchmarks despite using half the parameters, with particularly large gains on reasoning and code tasks (e.g., +26.3 on AIME-24 and +15.3 on LiveCodeBench-v6). Compared to SDAR, which is built on the same Qwen3-8B base, I-DLM improves dramatically (e.g., 69.6 vs. 10.0 on AIME-24). At a larger scale, I-DLM-32B continues to outperform substantially larger models, surpassing LLaDA-2.1-flash (100B) by +16.7 on AIME-25 and +11.7 on LiveCodeBench-v6. Moreover, Table 2 shows that our I-DLM even outperforms proprietary DLMs on code generation, such as Mercury Coder Small (90.0 vs. 76.6) and Gemini Diffusion (89.6 vs. 76.0).

I-DLM achieves comparable quality to AR models. Beyond outperforming prior DLMs, I-DLM is the first to match the quality of same-scale AR models. As shown in Table 1, I-DLM-8B achieves near-identical performance to Qwen3-8B, matching exactly on ARC-C, IFEval, remaining within ∼1 point on MMLU, even surpassing Qwen3-8b on Math-500. This result validates that introspective consistency is a key missing ingredient in prior DLMs: by aligning generation with verification through logit-shifted causal training, I-DLM recovers AR-level capability.

I-DLM delivers superior serving efficiency than existing DLMs. We evaluate end-to-end serving performance on MBPP, MATH-500, and LMSYS-Chat under concurrency levels C ∈

- Table 1: End-to-end quality. Accuracy (%) on 15 benchmarks. I-DLM results use ISD (N=4, sampling). Underline: best non-AR result under 30B. †: best non-AR result under 100B.

LLaDA-2.1 LLaDA-2.0 LLaDA-2.1 SDAR SDAR I-DLM Qwen3 I-DLM Qwen3

-mini -flash -flash 8B 30B-A3B 8B 8B 32B 32B Params 16B 100B 100B 8B 30B 8B 8B 32B 32B Knowledge & Reasoning

ARC-C 90.2 — — 91.9 93.2 95.8 95.8 96.8† 97.2 MMLU 74.5 — — 78.6 82.8 82.4 83.5 86.8† 87.2 MMLU-Pro 64.8 74.8 76.6 56.9 61.5 73.1 75.1 79.7† 80.1 GPQA-D 46.0 — — 40.2 36.7 55.6 58.9 62.1† 64.1 GPQA 53.3 62.3 67.3† — — 54.9 55.4 58.7 65.0

Math

GSM8K 89.0 — — 91.7 91.4 95.0† 96.0 94.9 94.7 MATH-500 85.0 — — 78.6 77.8 96.8 95.8 97.6† 97.8 MathBench 84.2 — — 76.9 79.3 89.1 93.1 95.6† 95.5

- AIME-24 43.3 — — 10.0 16.7 69.6 73.1 83.3† 76.7

- AIME-25 43.3 60.0 63.3 10.0 10.8 60.8 65.4 80.0† 80.0

Code

HumanEval 86.0 — — 78.7 87.2 93.3 95.1 96.3† 96.3 MBPP 82.1 — — 72.0 71.6 92.2 93.4 94.6† 95.7 LCB-v6 30.4 42.5 45.4 16.6 21.7 45.7 50.3 57.1† 58.3

Instruction Following IFEval 83.2 82.6 83.6 61.4 60.6 84.7 84.7 84.7† 84.5

- Table 2: Extended comparison on benchmarks commonly reported across diffusion LLM methods. “—” indicates the result is not reported in the original paper.

Method GSM8K MMLU HumanEval MBPP IFEval Qwen3-8B (AR) 96.0 83.5 95.1 93.4 84.7 NBDiff (7B) 91.0 82.9 89.0 87.6 60.8 Jacobi Forcing (7B) 91.4 — 83.5 70.4 WeDLM (8B) 90.2 75.5 75.0 67.0 LightningRL (8B) 90.3 — 72.6 58.3 TiDAR (8B) 80.4 76.6 57.9 65.4 DREAM (7B) 81.0 70.6 57.9 58.8 62.5 Fast-dLLM (7B) 78.5 — 43.3 28.2 Mercury Coder Small — — 90.0 76.6 Gemini Diffusion — — 89.6 76.0 Ours (8B) 95.0 82.4 93.3 92.2 84.7

{1,2,4,8,16,32,64} (Appendix H). Figure 5 shows the throughput–latency tradeoff. Across all workloads, I-DLM consistently outperforms prior DLMs at moderate to high concurrency. Starting from C≥4, I-DLM achieves higher per-request throughput than both SDAR (8B) and LLaDA-2.1-mini (16B), with the advantage widening as concurrency increases. At typical deployment scales (C=16–32), I-DLM delivers 2.2–3.8× higher throughput than LLaDA-2.1-mini and 3.7–4.5× over SDAR. Under heavy load (C=64), I-DLM sustains stable per-request throughput (∼125 tok/s), translating to 2.9–4.1× higher throughput at C=64.

Comparison with speculative decoding for AR models. Figure 5(b) compares I-DLM against EAGLE3 (Li et al., 2025), a speculative decoding method that relies on an auxiliary draft model on top of the base AR model. I-DLM outperforms EAGLE3 in per-request throughput from C=1 through C=32 across all benchmarks (e.g., 341 vs. 238 tok/s on MATH-500, 319 vs. 221 on LMSYS-Chat, and 327 vs. 245 on MBPP at C=1). Notably, even IDLM-Lossless—which produces output bit-for-bit identical to the base AR model—exceeds EAGLE3 at most concurrencies (e.g., 310 vs. 238 tok/s at C=1 on MATH-500), despite EAGLE3 requiring a separate draft model. As concurrency increases, I-DLM maintains its advantage over baselines (199 vs. 176 tok/s on MATH-500 and 195 vs. 184 on LMSYS-Chat at C=32).

##### 4.3 Ablation Studies

Ablations of the training design. Figure 6a compares I-DLM’s introspective-consistency training (causal attention + logit shift + all-masked objective) against standard block diffusion training (block-causal attention, no logit shift), on the same data budget. The gap is substantial, especially on long-horizon reasoning tasks: code generation drops sharply

I-DLM (Ours) LLaDA 2.1-mini SDAR I-DLM (Ours) I-DLM-Lossless (Ours) AR EAGLE3

MBPP

###### MATH-500

###### LMSYS-Chat

MBPP

###### MATH-500

###### LMSYS-Chat

TotalThroughput(tok/s)

TotalThroughput(tok/s)

64

8000

|6|4| | |
|---|---|---|---|
| | |16| |
| | | | |
| | | |4|
| | | | |
| | | | |

64

64

64

64

6000

6000

16

16

4000

4000

16

16

16

2000

2000

4

4

4

4

4

1

1

1

1

1

1

0

0

150 300 450

150 300 450

100 200 300

160 240 320

160 240 320

160 240 320

Per-Request TPS (tok/s) Per-Request TPS (tok/s)

(a) Comparison with DLMs (b) Comparison with Spec. Decoding

Figure 5: Throughput–latency tradeoff across batch sizes (1, 4, 16, 64).

100

Baseline Paged Attn.

CUDA Graph Decode Loop

Argmax Prop. Fused Verify

Full System

95.0

| |
|---|

| |
|---|

| |
|---|

| |
|---|

92.2 93.3 90.1

89.1

| |
|---|

| |
|---|

| |
|---|

90

Accuracy(%)

15 96 24 27 4 5

82.4

111

80.0

80

282 (2.5×) 829 2,084 (2.5×) 2,791 267 1,286 920 582 5,882 (2.1×)

C = 1 C = 8

86 586 243 254 84

71.6

70

67.4

C = 32

I-DLM (Ours) 60.3

60

Block Diffusion training

0 1000 2000 3000 4000 5000 6000

50 GSM8KMathBench MMLU MBPPHumanEval

Total Throughput (tok/s)

(a) Training ablation. I-DLM (causal + logit shift) vs. block diffusion (block-causal, no logit shift).

(b) Systems optimization ablation. Cumulative throughput at C=1,8,32. The full system achieves 2.1–2.5× over the naive baseline.

##### Figure 6: Performance breakdown of the training design and systems optimizations.

(HumanEval: 92.7 → 60.3; MBPP: 92.8 → 67.4), and math reasoning degrades significantly (MathBench: 89.1 → 71.6). Yet, knowledge tasks are relatively unaffected (MMLU: 82.4 → 80.0). This indicates that introspective consistency significantly reduces error accumulation over long reasoning chains.

Ablation of the system design. Figure 6b reports efficiency breakdown at C=1,8,32. The largest gain comes from CUDA graph capture (+42–76%), which eliminates kernel launch overhead. Decode-loop (stationary-batch scheduling) optimizations (+11–21%) and argmax proposals (+11–15%) further improve throughput by reducing host-side scheduling. Additional gains come from paged-only attention (+10–14%) and kernel fusion (+1–4%).

- Table 3: Impact of stride size. TPF, TPS (bs=1), and accuracy at stride N on 1×H100.

N TPF TPS MATH-500 MBPP

- 2 1.80 209.6 96.8 93.4
- 3 2.48 281.9 95.8 92.8
- 4 2.96 324.5 96.8 92.2 8 4.01 445.1 94.6 88.3

Table 4: Relaxed acceptance. τ trades quality for TPF at N=4.

τ 0 0.1 0.2 0.5 1

HumanEval 93.3 93.3 92.1 91.8 91.2 TPF 2.63 2.62 2.68 2.72 2.73

Impact of stride size. The stride N in ISD controls the parallelism-quality trade-off. Starting from our I-DLM-8B, we extend to larger strides (N=8) via continued training. As shown in Table 3, TPF scales nearly linearly (1.80 → 4.01 from N=2 to N=8), while accuracy remains stable across tasks (e.g., MATH-500 within 94.6–96.8%, MBPP within 88.3–93.4%). These results show that I-DLM sustains quality under increased parallelism.

Adaptive stride via relaxed acceptance. ISD offers a control knob: a loose threshold τ that multiplies the acceptance ratio by (1+τ), boosting the effective acceptance rate. At τ=0 (strict), ISD provably matches the AR distribution. Increasing τ relaxes this guarantee for higher TPF without retraining. As shown in Table 4, quality is robust to relaxation: at τ=1.0, HumanEval drops by only 1.6 points (93.3 → 91.2) while TPF increases from 2.63 to 2.73, suggesting that I-DLM’s proposals are already well-aligned with the causal anchor.

#### 5 Conclusion

In this paper, we revisit the DLM design through the lens of autoregressive modeling and identify introspective consistency as the missing principle behind their quality gap. Building on this insight, we introduce I-DLM, a new paradigm that unifies parallel generation and selfverification via logit-shifted causal training and introspective strided decoding. Our results show that enforcing consistency enables DLMs to match AR-level quality, substantially outperforming existing DLMs, achieving 2.9–4.1× better throughput on large concurrency.

#### References

AIME. Aime problems and solutions. https://artofproblemsolving.com/wiki/index.php/ AIME Problems and Solutions.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces, 2023. URL https://arxiv. org/abs/2107.03006.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.

Tiwei Bie, Maosong Cao, Xiang Cao, Bingsen Chen, Fuyuan Chen, Kun Chen, Lun Du, Daozhuo Feng, Haibo Feng, Mingliang Gong, et al. Llada2. 1: Speeding up text diffusion via token editing. arXiv preprint arXiv:2602.08676, 2026.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Shuang Cheng, Yihan Bian, Dawei Liu, Linfeng Zhang, Qian Yao, Zhongbo Tian, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, et al. Sdar: A synergistic diffusionautoregression paradigm for scalable sequence generation. arXiv preprint arXiv:2510.06303, 2025.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via selfdistillation through time. arXiv preprint arXiv:2410.21035, 2024.

Yonggan Fu, Lexington Whalen, Zhifan Ye, Xin Dong, Shizhe Diao, Jingyu Liu, Chengyue Wu, Hao Zhang, Enze Xie, Song Han, et al. Efficient-dlm: From autoregressive to diffusion language models, and beyond in speed. arXiv preprint arXiv:2512.14067, 2025.

Itai Gat, Heli Ben-Hamu, Marton Havasi, Daniel Haziza, Jeremy Reizenstein, Gabriel Synnaeve, David Lopez-Paz, Brian Karrer, and Yaron Lipman. Set block decoding is a language model inference accelerator, 2025. URL https://arxiv.org/abs/2509.04185.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozi`ere, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Lanxiang Hu, Siqi Kou, Yichao Fu, Samyam Rajbhandari, Tajana Rosing, Yuxiong He, Zhijie Deng, and Hao Zhang. Fast and accurate causal parallel decoding using jacobi forcing. arXiv preprint arXiv:2512.14681, 2025.

Yanzhe Hu, Yijie Jin, Pengfei Liu, Kai Yu, and Zhijie Deng. Lightningrl: Breaking the accuracy-parallelism trade-off of block-wise dllms via reinforcement learning. arXiv preprint arXiv:2603.13319, 2026.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. Cllms: Consistency large language models. In Forty-first International Conference on Machine Learning, 2024.

Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, et al. Mercury: Ultra-fast language models based on diffusion. arXiv preprint arXiv:2506.17298, 2025.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pp. 19274–19286. PMLR, 2023.

Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori Hashimoto. Diffusion-LM improves controllable text generation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=3s9IrEsjLyk.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077, 2024.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. arXiv preprint arXiv:2503.01840, 2025.

Aiwei Liu, Minghua He, Shaoxun Zeng, Sijun Zhang, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Xiao Zhou, and Jie Zhou. Wedlm: Reconciling diffusion language models with standard causal attention for fast inference. arXiv preprint arXiv:2512.22737, 2025a.

Hongwei Liu, Zilong Zheng, Yuxuan Qiao, Haodong Duan, Zhiwei Fei, Fengzhe Zhou, Wenwei Zhang, Songyang Zhang, Dahua Lin, and Kai Chen. Mathbench: Evaluating the theory and application proficiency of llms with a hierarchical mathematics benchmark. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 6884–6915, 2024.

Jingyu Liu, Xin Dong, Zhifan Ye, Rishabh Mehta, Yonggan Fu, Vartika Singh, Jan Kautz, Ce Zhang, and Pavlo Molchanov. Tidar: Think in diffusion, talk in autoregression. arXiv preprint arXiv:2511.08923, 2025b.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. 2023.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution, 2024. URL https://arxiv.org/abs/2310.16834.

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. Specinfer: Accelerating generative large language model serving with tree-based speculative inference and verification. arXiv preprint arXiv:2305.09781, 2023.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025a.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models, 2025b. URL https://arxiv.org/abs/2502.09992.

Augustus Odena, Charles Sutton, David Martin Dohan, Ellen Jiang, Henryk Michalewski, Jacob Austin, Maarten Paul Bosma, Maxwell Nye, Michael Terry, and Quoc V Le. Program synthesis with large language models. n/a, page n/a, n/a. N/a, 2021.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. In First conference on language modeling, 2024.

Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Mohammad Samragh, Arnav Kundu, David Harrison, Kumari Nishu, Devang Naik, Minsik Cho, and Mehrdad Farajtabar. Your llm knows the future: Uncovering its multi-token prediction potential. arXiv preprint arXiv:2507.11851, 2025.

Yuchuan Tian, Yuchen Liang, Shuo Zhang, Yingte Shu, Guangwen Yang, Wei He, Sibo Fang, Tianyu Guo, Kai Han, Chao Xu, et al. From next-token to next-block: A principled adaptation path for diffusion llms. arXiv preprint arXiv:2512.06776, 2025.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.

Shutong Wu and Jiawei Zhang. Free draft-and-verification: Toward lossless parallel decoding for diffusion large language models. arXiv preprint arXiv:2510.00294, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

#### A Detailed Related work

Diffusion Language Models. Masked diffusion language models corrupt text by replacing tokens with [MASK] and train a model to reverse the process (Austin et al., 2021; Sahoo et al., 2024; Lou et al., 2023). LLaDA (Nie et al., 2025a) scaled this paradigm to 8B parameters, LLaDA 2.0 (Bie et al., 2025) to 100B via mixture-of-experts, and LLaDA 2.1 (Bie et al.,

- 2026) introduced token editing with confidence-based decoding. DREAM (Ye et al., 2025) introduced the logit shift technique, aligning the diffusion objective with the AR model’s logits[i] → token[i+1] mapping. Block Diffusion (Arriola et al., 2025) generates fixed-size blocks autoregressively while denoising tokens within each block.

Converting pretrained AR models into diffusion models is a more data-efficient alternative. Gong et al. (2024) showed that fine-tuning with a masked diffusion objective reduces training cost significantly. Self-Distillation Through Time (Deschenaux & Gulcehre, 2024) and Efficient-DLM (Fu et al., 2025) further streamline the pipeline. SDAR (Cheng et al., 2025) converts AR models via full-model training on ∼50B tokens for block-parallel generation, and NBDiff (Tian et al., 2025) extends this with causal prefix constraints. TiDAR (Liu et al., 2025b) proposes a sequence-level hybrid that drafts via diffusion and verifies autoregressively.

Unlike these approaches, I-DLM combines strict causal attention with logit-shifted prediction from MASK positions—by maximally respecting the pretrained AR model’s attention and prediction patterns, our conversion is far more data-efficient and closes the quality gap to the base AR model.

Speculative decoding and multi-token prediction for LLMs. Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates AR models by drafting tokens with a fast model and verifying them in parallel, provably preserving the target distribution. This guarantee relies on the target model having a well-trained verify distribution p—a property that AR models possess inherently but standard DLLMs lack due to mask-only training. Extensions include Medusa (Cai et al., 2024) (multiple prediction heads), the EAGLE family (Li et al., 2024; 2025) (feature-level drafting), and SpecInfer (Miao et al., 2023) (tree-structured verification). Multi-token prediction (MTP) trains models to predict multiple future tokens simultaneously (Gloeckle et al., 2024). Samragh et al. (2025) proposed gated sparse expert LoRA for MTP—the closest prior work to our conditional LoRA, though designed for multi-head prediction rather than diffusion verification. Consistency LLMs (Kou et al., 2024) achieve acceleration via Jacobi iteration. ISD differs from these by using a single model with training-time stride capability to both achieve quality guarantee and deliver high latency and throughput gains under high concurrency.

DLLM-specific decoding. FastDLLM (Wu et al., 2025) enables confidence-aware parallel decoding with KV cache reuse. Jacobi Forcing (Hu et al., 2025) distills diffusion models for fewer-step convergence but degrades at larger batch sizes. Free Draft-and-Verification (Wu & Zhang, 2025) explores self-speculative approaches. WeDLM (Liu et al., 2025a) reconciles diffusion with causal attention for KV cache reuse. These methods rely on confidencebased acceptance or iterative denoising, which either lacks formal quality guarantees or incurs high compute overhead; our ISD provides provable AR-quality output in a single stride–introspection cycle.

Despite these advances, fundamental gaps remain between DLLMs and AR models in training scalability, inference alignment, compute efficiency, and infrastructure compatibility. We analyze these gaps quantitatively in the next section.

#### B TPF and Compute Overhead Analysis

We analyze the theoretical tokens per forward pass (TPF) and compute overhead for three parallel decoding paradigms: ISD (ours), block diffusion (SDAR), and branched self-speculative decoding (TiDAR). Throughout, N is the block/stride size and p is the

uniform per-token acceptance probability (Pk = pk). Compute overhead (OH) is the ratio of total query tokens to output tokens; AR has OH = 1 by definition.

##### B.1 ISD (Ours)

ISD alternates between an NP step (propose only: append N masks, produce 1 free token via logit shift + N−1 speculative proposals, finalize nothing) and P steps (introspect previous proposals + propose new ones). A renewal cycle is one NP step followed by a geometric chain of P steps ending on the first rejection.

Each P step has probability pN−1 of all-pass (finalizing N tokens, continuing the chain) and 1 − pN−1 of rejection. On rejection at position k, we finalize k + 1 tokens (1 free + k−1 accepted + 1 resampled).

TPF. Expected tokens and forwards per cycle:

2 + p + · · · + pN−2 1 − pN−1

2 − pN−1 1 − pN−1

E[tokens] =

, E[forwards] =

. (4)

2 + p + p2 + · · · + pN−2 2 − pN−1

TPFISD =

. (5)

At p = 1: TPF = N; at p = 0: TPF = 1 (AR). Overhead. The P step processes 2N−1 query tokens (N−1 filled proposals + N masks). The NP step processes N queries (variable) or 2N−1 queries (fixed, padded to match P). Variable query:

3N − 1 − NpN−1 2 + p + · · · + pN−2

OHvar =

. (6)

Fixed query (SGLang/vLLM, both steps use 2N−1):

(2N−1)(2 − pN−1) 2 + p + · · · + pN−2

OHfix =

. (7)

The gap vanishes as p → 1 and is irrelevant in the memory-bound regime where wall-clock speedup ≈ TPF.

##### B.2 Block Diffusion (SDAR)

SDAR generates a block of N tokens via iterative denoising with a force schedule, then commits the KV cache in a separate forward pass. At each denoising step, H ∼ Binomial(R, p) tokens pass the confidence threshold out of R remaining; at least max(H,1) are committed. Let E[S | N] be the expected denoising steps for N tokens (computed recursively). Total forwards = E[S | N] + 1 (denoising + KV commit), each processing all N tokens.

N E[S | N] + 1

TPFSDAR =

, OHSDAR = E[S | N] + 1. (8)

Note TPF × OH = N always. Even at p = 1 (1 denoising step + 1 KV commit = 2 forwards), TPFSDAR ≤ N/2—the mandatory KV commit caps throughput at half the theoretical maximum.

##### B.3 Branched Self-Speculative Decoding (TiDAR)

TiDAR uses a single forward to both verify N draft tokens and pre-draft N branches of N masks each, covering all possible rejection outcomes. The input is N verify tokens + N2 branch masks = N(N+1) queries per forward, always exactly 1 forward per cycle.

1 − pN 1 − p

TPFTiDAR = 1 + p + p2 + · · · + pN−1 =

. (9)

N(N+1) 1 + p + · · · + pN−1

OHTiDAR =

. (10)

TiDAR achieves the highest TPF (no NP recovery step), but the N2 branch masks are structurally wasteful: only 1 of N branches is selected per forward, so (N−1) · N mask queries are always discarded. Even at p = 1, OH = N+1 and efficiency = N/(N+1) < 1TiDAR can never be FLOP-efficient.

##### B.4 Summary

Table 5: TPF and overhead formulas (uniform p, stride N).

Method TPF Overhead Queries/fwd ISD (var.) 2+p+···+p

3N−1−NpN−1 2+p+···+pN−2 N or 2N−1

N−2 2−pN−1

N−1)

ISD (fix.) (same) (2N−1)(2−p

2+p+···+pN−2 2N−1 SDAR E[S|NN]+1 E[S|N] + 1 N TiDAR 1−p

N

N(N+1)(1−p)

1−pN N(N+1)

1−p

#### C Why Block Diffusion Requires a Separate KV Commit Pass

Our SDAR inference uses SGLang’s native DLLM support1. In this section we explain why block diffusion methods (SDAR, LLaDA) require a mandatory KV commit forward pass after denoising, and why this overhead is difficult to eliminate.

After T denoising steps produce N final tokens, a separate forward pass with store kv=True must write these tokens into the KV cache so that subsequent blocks can attend to them. This pass produces zero new tokens and is mandatory even at p=1, capping TPF at N/2.

A natural optimization would fuse commit with the next block’s denoising in one forward: [t1, . . . , tN commit

, M, . . . ,M

]. However, this requires per-position mixed attention masks: committed

denoise

tokens need bidirectional attention among themselves, while MASK tokens need blockcausal attention. Current attention kernels (FlashAttention, FlashInfer) support only a single global causal flag and cannot mix bidirectional and causal attention within one forward pass. Implementing this would require custom attention kernels, doubled query size (2N), and complex batching logic for concurrent requests at different stages. SGLang’s codebase confirms this: dllm is commit and dllm needs commit flags are defined but never activated—the optimization was considered but abandoned.

Because I-DLM uses strict causal attention throughout, no commit pass is needed. Each ISD step is a standard extend operation that incrementally commits KV entries as in AR decoding—no mixed masks, no custom kernels.

1https://github.com/sgl-project/sglang/pull/19044

#### D Attention Kernel Overhead

Block diffusion uses block-causal attention, which differs from the strict token-level causal mask that AR serving kernels are optimized for. In SGLang, the standard extend path uses a cascade of three attention kernels per layer: (1) ragged attention among new tokens, (2) paged attention against the cached prefix, and (3) a merge kernel to combine the two via log-sum-exp renormalization. This cascade is optimized for large prefills but wasteful for DLLM decode steps that process only N=4–5 tokens, where the ragged kernel’s advantage vanishes and the 3× kernel launch overhead dominates.

Because I-DLM uses strict causal attention with small extended sizes, it can bypass the cascade and use a single paged attention kernel per layer, reducing kernel launches from 3L to L (where L is the number of layers). Figure 7 shows the resulting forward latency comparison across concurrency levels. At low concurrency (C=1), the cascade overhead is modest (+4%), but it grows to +20% at C=64 as the additional kernel launch overhead compounds with batching.

I-DLM (Paged) +20%

ForwardLatency(ms)

20

Block DLLM (Cascade)

15

+15%

+12%

+7%

10

+4% +4% +6%

5

0

1 2 4 8 16 32 64

Concurrency

Figure 7: Attention kernel forward latency at varying concurrency. I-DLM (Paged, single kernel) vs. Block DLLM (Cascade, three kernels). The cascade overhead grows from +4% at C=1 to +20% at C=64.

#### E Attention Mask Structure

- Figure 8 visualizes the attention mask used during I-DLM training, contrasted with SDAR’s block-causal mask. The input sequence is the concatenation [xt | x0], where xt is the allmasked (noisy) region and x0 is the clean reference. The mask is composed of three components:

- • Noisy self-attention (Mnoisy): Self-attention within the noisy region xt. In our setting (use regular causal=True), this is causal within each block: position j in block b attends only to positions i ≤ j in the same block. SDAR instead uses bidirectional attention within blocks.

- • Cross-attention (Mcross): Cross-attention from noisy tokens to clean tokens. Each noisy block b attends to clean tokens from all preceding blocks (b′ < b), providing conditional context from the clean reference.
- • Clean self-attention (Mclean): Self-attention within the clean region x0. In our setting, this is strict token-level causal (q ≥ kv), preserving the AR attention pattern exactly. SDAR instead uses block-causal attention here.

The final mask is M = Mnoisy ∪ Mcross ∪ Mclean.

xt (noisy) x0 (clean)

xt (noisy) x0 (clean)

(noisy)x(clean)xt0

(noisy)x(clean)xt0

Mnoisy: causal within xt blocks Mcross: noisy←clean cross-attn Mclean: strict causal in x0

Mnoisy: bidir. within xt blocks

Mcross: noisy←clean cross-attn

Mclean: block-causal in x0

- (a) I-DLM (Ours). Strict causal within xt blocks and strict causal in x0.

(b) SDAR (Block Diffusion). Bidirectional within xt blocks and block-causal in x0.

- Figure 8: Attention mask comparison for block size N=2, sequence length L=6. Input is

[xt | x0] (noisy ∥ clean). Rows are query positions; columns are key positions. Our I-DLM (left) uses strict causal attention everywhere, preserving AR compatibility. SDAR (right) uses bidirectional attention within noisy blocks and block-causal attention in the clean region. The three mask components—Mnoisy (noisy self-attention), Mcross (noisy←clean cross-attention), Mclean (clean self-attention)—are color-coded.

F ISD Step-by-Step Illustration

- Figure 9 provides a detailed step-by-step illustration of Introspective Strided Decoding at stride N=3, showing both the all-accept and rejection cases.

G Lossless ISD with Gated LoRA

- Figure 10 illustrates the gated LoRA mechanism used in Residual ISD (R-ISD). During each forward pass, token positions are partitioned into two types based on their input: [MASK] positions (proposals) activate the LoRA residual, while clean positions (introspection) use base-model-only weights. Because of strict causal attention, introspection positions cannot attend to any [MASK] position—their KV cache entries are computed entirely from base weights over clean tokens.

Per-position computation. Table 6 summarizes the computation at each position type. The gated LoRA mechanism applies a per-token binary mask m: each linear layer computes hj ← Wxj + 1[j∈M] · BA xj, where 1[j∈M] is 1 for [MASK] positions and 0 otherwise.

- Table 6: Per-position breakdown in R-ISD. Introspect and exact positions use base-only weights, producing the AR-identical causal anchor p. Propose positions add the LoRA residual to produce aligned proposals q.

Position Weights Output Role

xˆk (introspect) Wx (base only) pk (causal anchor) Produces p for p/q acceptance x1 (exact) Wx (base only) p (AR-identical) Quality-guaranteed token [M]k (propose) Wx + BAx (base+LoRA) qk (proposal) Strided proposal aligned with p

Clean Introspect Mask Exact (free)

- Step 1 t1 · · · tk m m m Accepted Resampled Discarded

prefix propose (N=3)

- Step 2 t1 · · · tk

+ append mask

introspect propose

exact

|x1|
|---|

xˆ2 xˆ3 xˆ4 m m m

qi =h[mi] (Step 1) pi =h[xˆi−1] (Step 2)

p2 p3 p4

accept: min 1, qpi

i single forward pass: introspect + propose unified

✓ ✓ ✓ ✓ ⋆

|x5|
|---|

+5 accepted +3 proposals

(a) all accept x1 x2 x3 x4

xˆ6 xˆ7 xˆ8

KV cached introspect propose

Step 3 t1 · · · tk x1 · · · x5 xˆ6 xˆ7 xˆ8 m m m

✓ ✓ ×

+3 accepted discard rest

(b) reject xˆ3 x1 x2 x3′ xˆ4 x5 xˆ6 xˆ7 xˆ8

KV cached propose only

no prev. proposals to introspect (restart from x3′ )

Step 3 t1 · · · tk x1 x2 x3′ m m m

- Figure 9: Detailed ISD illustration at stride N=3. Step 1: Bootstrap—append 3 [MASK] tokens, producing x1 (exact) and proposals xˆ2, xˆ3, xˆ4. Step 2: Single forward pass that introspects on previous proposals (computing causal anchors pk) while generating new proposals. (a) All accept: 4 tokens accepted + bonus x5; Step 3 introspects on new proposals.

- (b) Reject xˆ3: x1, x2 accepted, x3′ resampled, rest discarded; Step 3 is a pure propose step.

The overall R-ISD pipeline can be expressed compactly as:

, xˆi+2, . . . , xˆi+N

= πθ(c1:i, M1:N−1) (11)

xi+1 from h[ci] (base only)

from h[M1:N−1] (base+LoRA)

where πθ denotes the model with gated LoRA: W → W + m ⊙ BA. With LoRA active on [MASK] positions, all proposals become higher quality, increasing the introspective acceptance rate while the causal anchor p remains the exact base AR distribution.

#### H Training, Serving, and Evaluation Details

##### H.1 Training Setup

Base models and data. We train two I-DLM variants: I-DLM-8B and I-DLM-32B, converted from Qwen3-8B and Qwen3-32B (Yang et al., 2025), respectively, using the introspectiveconsistency training recipe described in Section 3.1. Our training codebase builds on the open-source SDAR framework(https://github.com/JetAstra/SDAR). The training data consists of 4.5B tokens of responses generated across reasoning datasets. All training is performed on 8 H100 GPUs.

prefix introspect exact propose

t1 · · · tk xˆ2 xˆ3 x1 m m

Wx Wx Wx Wx Wx Wx + ABx Wx + ABx

p p p p p q q

base-only weights ⇒ p = exact AR distribution base+LoRA weights ⇒ q aligned with p

Clean Introspect Exact Propose Base (W) LoRA (AB)

- Figure 10: Gated LoRA in Residual ISD (R-ISD). During a single forward pass, [MASK] (propose) positions compute Wx + ABx using base+LoRA weights, producing proposal distributions q. Clean and introspect positions compute Wx using base-only weights, producing the causal anchor distribution p—identical to a pure base AR forward pass. Because of causal attention, introspection positions never attend to [MASK] positions, so p is computed entirely from base-only KV entries. This makes the output bit-for-bit lossless with respect to the base AR model.

Training schedule. For I-DLM-8B, we train for 2 epochs with a stride curriculum: the first epoch trains with stride N=2, and the second epoch trains with stride N=3. During the first two stride expansions (N=1 → 2 and N=2 → 3), we use a fixed low scale of 0.2 for the cross-entropy loss on clean tokens to speed up masked token learning. Starting from the N=3 → 4 expansion, we switch to the auto-balanced loss scaling described in Eq. 2. For I-DLM-32B, we train for 1 epoch with stride N=2 with LoRA using rank 1024 and a fixed scale of 0.2. We use N=4 for evaluation without additional full training as well.

Hyperparameters. We use full fine-tuning with DeepSpeed ZeRO Stage 2. The learning rate is 1 × 10−5 with cosine decay and a warmup ratio of 0.03. We use a per-device batch size of 1 with gradient accumulation steps of 4, yielding an effective batch size of 32 across 8 GPUs. The maximum sequence length is 4096 tokens. Training uses bf16 mixed precision.

LoRA training. For lossless ISD (R-ISD), we additionally train LoRA adapters with rank 128 on the same data using the same hyperameters with a learning rate of 2e-4. The inference follows the gated residual design described in Section 3.2.

##### H.2 Hardware

All experiments are conducted on NVIDIA H100 80GB SXM GPUs with NVLink interconnect. We use CUDA 12.9 with FlashInfer for attention computation. CUDA graphs are enabled for all configurations.

##### H.3 Serving Configuration

- Table 7 summarizes the serving configurations used across all models.

ISD algorithm configuration. Table 8 lists the ISD algorithm parameters used for each stride configuration.

Speculative decoding baselines. Table 9 details the configurations for speculative decoding baselines. All baselines use Qwen3-8B as the target model and are served with SGLang.

SGLang server parameters. Common parameters across all configurations: mem-fraction-static=0.85, attention-backend=flashinfer, disable-radix-cache=true (required for DLLM KV trim). CUDA graph capture is enabled by default; the DLLM extend forward (all 2N−1 tokens) is captured into a single graph per batch size, with attention metadata updated via in-place tensor writes before each replay.

- Table 7: Serving configurations. All models are served with SGLang on H100 GPUs.

Model TP #Servers Dtype DLLM Config LoRA 8B models (1 GPU per server) Qwen3-8B (AR) 1 8 bf16 — I-DLM-8B (N=3) 1 8 bf16 blockN3 I-DLM-8B (N=5) 1 8 bf16 blockN5 I-DLM-8B (Lossless) 1 8 bf16 blockN3+LoRA r=128 EAGLE-3 1 8 bf16 — 32B models (2 GPUs per server)

Qwen3-32B (AR) 2 4 bf16 — I-DLM-32B (N=3) 2 4 bf16 blockN3 I-DLM-32B (Lossless) 2 4 bf16 blockN3+LoRA r=1024

- Table 8: ISD algorithm configurations. “blockNk” denotes stride N=k with block size = 2k−1.

Config N block size temperature top k top p spec verify

blockN3 3 5 1.0 50 0.95 true blockN5 5 9 1.0 50 0.95 true blockN3+LoRA 3 5 1.0 50 0.95 true

- H.4 Evaluation Configuration Table 10 details the evaluation settings for each benchmark.

Sampling parameters. For quality evaluation, all models use temperature t=1.0, topk=50, top-p=0.95 (matching the ISD algorithm configuration). For the AR baseline (Qwen38B/32B), the same sampling parameters are used to ensure a fair comparison. For each benchmark, results are averaged over three runs.

Answer extraction. For math benchmarks (GSM8K, MATH-500, AIME), we extract the final \boxed{...} answer after stripping the <think>...</think> block. For multiple-choice benchmarks (ARC-C, MMLU, MMLU-Pro, GPQA), we extract the answer letter using the pattern ANSWER: [A-D]. For code benchmarks (HumanEval, MBPP, LCB), we extract the last Python code block containing a function definition and execute it against the provided test cases using a sandboxed subprocess with a 10-second timeout. For IFEval, we use the official Google evaluator after stripping thinking blocks.

Baseline reproduction. Results for LLaDA-2.1-mini are reproduced using SGLang with the official configuration (block size=4, threshold=0.95, edit threshold=0.9). Results for SDAR are reproduced using SGLang’s DLLM serving mode. Results for EAGLE-3 are reproduced using its SGLang integration. All other baseline results are taken from their original publications.

- H.5 Throughput Benchmark Configuration

For latency and throughput measurements (Section 5), we use genai-bench2 with the following settings:

- H.6 Infrastructure Ablation Configuration

The ablation study in Section 6 isolates the contribution of each serving optimization by disabling them individually via environment variables. Table 12 lists the ablation toggles.

2https://github.com/sgl-project/genai-bench

- Table 9: Speculative decoding baseline configurations. “Steps” and “topk” are EAGLE-3 draft parameters.

Method Draft Model Draft Params Notes EAGLE-3 Tengyunw/qwen3 8b eagle3 steps=3, topk=1, d=4 3 specs verified per step

- Table 10: Evaluation configurations per benchmark. All benchmarks use the Qwen3 chat template with thinking mode enabled. “max tokens” controls the maximum generation length including the <think> block.

Category Benchmark #Problems max tokens Metric / Extraction

ARC-C 1,172 32,768 Accuracy; choice letter MMLU 14,042 32,768 Accuracy; ABCD extraction MMLU-Pro 12,032 32,768 Accuracy; ABCD extraction GPQA-Diamond 198 32,768 Accuracy; ABCD extraction GPQA 448 32,768 Accuracy; ABCD extraction

Knowledge

GSM8K 1,319 32,768 Accuracy; \boxed{} MATH-500 500 32,768 Accuracy; \boxed{} MathBench 3,709 32,768 Accuracy; numerical

Math

- AIME-24 30 32,768 Accuracy; \boxed{}
- AIME-25 30 32,768 Accuracy; \boxed{}

HumanEval 164 32,768 pass@1; code execution MBPP (sanitized) 257 32,768 pass@1; code execution LCB-v6 175 32,768 pass@1; code execution

Code

Instruction IFEval 541 32,768 Prompt-strict accuracy Knowledge TriviaQA 17,944 32,768 Exact match

The ablation is cumulative: starting from the naive configuration (all optimizations disabled), we add each optimization one at a time and measure throughput at C=1, C=8, and C=32 on a single H100.

#### I Additional Results

##### I.1 Peak Throughput on Different Hardware

Table 13 reports peak per-request TPS under favorable conditions (low concurrency, long generation), using different model variants and stride configurations. The base I-DLM-8B is trained at N=3; the N=4 checkpoint is obtained via stride extension training from I-DLM8B, and N=8 is further extended from N=4. The LoRA variant uses a rank-128 adapter with segment-gated conditional activation (R-ISD).

##### Table 11: Throughput benchmark configuration.

###### Parameter Value

Concurrency levels C ∈ {1,2,4,8,16,32,48,64} Request arrival Burst mode (all requests submitted simultaneously) Output length 2,048 tokens (fixed) Measurement Total tokens / wall-clock time (first to last completion) Warmup 5 requests discarded before measurement GPU 1× H100 (TP=1) for 8B; 2× H100 (TP=2) for 32B

- Table 12: Infrastructure ablation toggles. Each toggle disables one optimization to measure its isolated contribution.

Toggle Disables Effect

--disable-cuda-graph CUDA graph capture/replay Forces eager model forward NO DECODE LOOP=1 Stationary-batch decode loop Falls back to full scheduler pipeline NO ARGMAX PROPOSALS=1 Argmax for draft positions Uses full sampling for specs NO FUSED VERIFY=1 Fused Triton verify kernel Uses 7+ separate kernel launches NO DEFERRED STREAM=1 Deferred stream output Runs on critical path DLLM USE RAGGED=1 Paged-only attention Forces cascade (3 kernels/layer)

- Table 13: Peak per-request TPS at various stride, hardware, and model configurations. All models are 8B scale.

Hardware TP N Model Variant Benchmark TPS B200 2 8 I-DLM-8B (ext. training N=8) GSM8K 925 H100 2 8 I-DLM-8B (ext. training N=8) MBPP 685 H100 1 4 I-DLM-8B (ext. training N=4) MATH-500 341 H100 1 4 I-DLM-8B (ext. training N=4) AIME 314 H100 1 4 I-DLM-8B (ext. training N=4) ShareGPT 319 H100 1 3 I-DLM-8B MATH-500 272 H100 1 3 I-DLM-8B R-ISD (LoRA r=128) MATH-500 240

