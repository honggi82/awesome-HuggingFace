# arXiv:2605.11011v1[cs.LG]10May2026

## LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models

Taekyhun Park Department of Data Science Pusan National University Busan, Republic of Korea pthpark1@pusan.ac.kr

Dohee Kim Department of Artificial Intelligence Engineering Changwon National University Changwon, Republic of Korea kimdohee@changwon.ac.kr

Yongjae Lee Department of Industrial Engineering Pusan National University Busan, Republic of Korea yongzzai1102@gmail.com

#### Hyerim Bae∗

Department of Industrial Engineering Pusan National University Busan, Republic of Korea hrbae@pusan.ac.kr

### Abstract

Looped computation shows promise in improving the reasoning-oriented performance of LLMs by scaling test-time compute. However, existing approaches typically require either training recurrent models from scratch or applying disruptive retrofits, which involve substantial computational costs and may compromise pretrained capabilities. To address these limitations, we introduce Looped Depth Up-Scaling (LoopUS), a post-training framework that converts a standard pretrained LLM into a looped architecture. As a key technical contribution, LoopUS recasts the pretrained LLM into an encoder, a looped reasoning block, and a decoder. It operationalizes this latent-refinement architecture through four core components: (1) block decomposition, guided by staged representation dynamics; (2) an input-dependent selective gate to mitigate hidden-state drift; (3) random deep supervision for memory-efficient learning over long recursive horizons; and (4) a confidence head for adaptive early exiting. Collectively, these mechanisms transform a standard non-looped model into a looped form while stabilizing it against both computational bottlenecks and representation collapse. Through stable latent looping, LoopUS improves reasoning-oriented performance without extending the generated traces or requiring recurrent training from scratch. For more details, see https://thrillcrazyer.github.io/LoopUS.

### 1 Introduction

The reasoning performance of large language models (LLMs) can be improved during inference by allocating additional computation, or test-time compute (TTC), in latent space to refine hidden states before producing the next token [1–3]. By deepening internal processing rather than inflating sequence length, latent-space computation offers a complementary axis along which reasoning capacity can scale within a fixed model without increasing its parameter count [4–9]. Looped language models are one example of this paradigm: they iterate a designated block (e.g., a transformer block or a stack of layers) to increase effective computational depth without additional parameters. However, training looped architectures from scratch is expensive at modern scales [4, 6].

∗Corresponding author

Preprint.

###### Raw Cosine Distance 1 cos(ht, ht+1)

1.0

Baby you make me

Yeah you re just my type and I like your

What you what you lookin' at

0.8

404 Not Found in the system 404 The new

CosineDistance

- 0.6

0.4

0.2

0.0

0 5 10 15 20 25

Layer Transition (t t + 1)

(a) Hidden state distance across layers.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| |h19 h17<br><br>| | | | | | |
| | | | | |h25<br><br>| | |
| | | | | | | | |

- PC-2(3.8%var)

Baby you make me

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| |h19 h17<br><br>| | | | | | |
| | | | | |h25<br><br>| | |
| | | | | | | | |

PC-1 (93.6% var)

PC-2(3.8%var)

Yeah you're just my type and I like your

PC-1 (93.5% var)

- PC-2(4.1%var)

PC-1 (92.6% var)

404 Not Found in the system 404 The new era

###### What you what you lookin' at

PC-2(2.9%var)

h19 h17 h25

h19 h17

h25

PC-1 (94.9% var)

(b) Hidden state trajectory visualized via PCA.

- Figure 1: Staged representation dynamics in Qwen/Qwen3-1.7B. (a) Cosine distance between consecutive hidden states reveals three distinct regimes. (b) Hidden-state trajectories confirm that middle layers trace a gradual arc within a confined region of latent space, while the final layers project sharply toward the output vocabulary space.

As an alternative, recent studies [10, 11] have explored tuning pretrained LLMs into a looped form. However, these approaches suffer from three limitations: (i) There is no principled recipe for identifying which layers should be reused as the recurrent block because existing methods rely on heuristics rather than an analysis of the internal representation dynamics of the model [10, 11]. (ii) Naive iteration causes hidden-state drift because the layers were trained for single-pass use at a fixed depth rather than as a recurrent operator. Repeated reuse can therefore degrade representational fidelity, preventing iterative refinement of output quality [12]. (iii) Backpropagation through a long unrolled loop is both memory-intensive and prone to vanishing or exploding gradients [13–15].

We begin by analyzing the hidden-state geometry of a pretrained LLM to understand how representations evolve across depth. In our preliminary investigation (Figure 1), the representation trajectory follows a staged pattern: early layers rapidly transform token embeddings, middle layers evolve gradually within a stable plateau, and final layers make a sharp transition toward output decoding. This pattern is consistent with recent findings on hidden-state geometry [16–18]. Building on this observation, we decompose the LLM into three functionally distinct blocks.

We propose Looped Depth Up-Scaling (LoopUS), a post-training framework that recasts a pretrained LLM into a looped form through four components. (i) Block Decomposition resolves the layerselection problem by partitioning the model into encoder, reasoning, and decoder blocks, grounded in the staged representation dynamics shown in Figure 1 rather than relying on heuristic layer selection. Note that only the reasoning block is reused as the loop body. (ii) A Selective Gate addresses hiddenstate drift by interpolating each proposed update with the previous state, turning every iteration into a damped refinement step instead of an unconstrained jump. (iii) Random Deep Supervision sidesteps full Backpropagation Through Time (BPTT): at each step, only a few uniformly sampled iterations receive gradients, while the rest run detached. This keeps training manageable as the loop budget grows. (iv) A Confidence Head predicts when further refinement is unnecessary, enabling adaptive test-time compute that allocates more iterations to harder inputs and fewer to easier ones.

Empirically, LoopUS improves zero-shot accuracy by 3.0% over pretrained backbones and reduces WikiText and LAMBADA perplexities by 17.4% and 21.3%, respectively. It also demonstrates high adaptation efficiency, yielding a 14.6% relative gain on TinyLlama with 17–20× fewer training tokens than existing looped baselines. Our analyses confirm that training remains stable across extended loop depths: hidden-state trajectories contract and token distributions sharpen, indicating that gains stem from controlled, iterative latent refinement rather than uncontrolled depth expansion.

The main contributions of this paper are threefold:

- • Representation-guided looped post-training framework: We propose LoopUS, a posttraining framework that converts a pretrained LLM into a looped latent-reasoning model. LoopUS decomposes the model into encoder, reasoning, and decoder blocks using staged representation dynamics, and reuses only the middle reasoning block as the loop body.

- • Stable and efficient latent recursion: We introduce mechanisms that make latent looping stable and practical in pretrained LLMs, including a Mamba-inspired selective decay gate, random deep supervision, and a confidence head. The gate mitigates hidden-state drift, while random deep supervision avoids full BPTT over long recursive horizons.
- • Empirical analysis of loop dynamics: We show that LoopUS improves reasoning-oriented performance, remains competitive under limited training budgets, and exhibits convergent loop dynamics through loop-depth analyses, latent-trajectory visualizations, token-level prediction analyses, and component ablations.

### 2 Background

LLM Hidden State Representations. Recent LLM interpretability studies, including Anthropic’s work [19, 20], suggest that LLM hidden states quickly move into an abstract predictive space in which high-level concepts can be represented, manipulated, and refined across depth rather than being rewritten at each layer. Prior studies on representation evolution and logit-lens analyses demonstrate a progression from local, lexical processing in lower layers to increasingly abstract, prediction-oriented representations in deeper layers [21, 22]. In this context, middle layers often form a plateau that changes relatively little, encoding information needed for the final prediction [17]. This is followed by a sharper transition near the final layers, where representations are further transformed toward the vocabulary space [16]. Furthermore, Ng [18] and Upstage [23] show that duplicating or stacking pretrained blocks can improve performance. We therefore treat the middle layers as a reusable latent workspace, exploiting this region through looping rather than by adding distinct blocks.

Looped LLMs. Complementing TTC [24], which scales sequence length to elicit more explicit reasoning, looped transformers scale computation depth by repeatedly applying the same block to refine latent representations without increasing the parameter count [10, 25, 26, 15]. Building on recurrent-transformer formulations [27], retrofitted recurrence [11, 10, 5], latent refinement [25, 26, 6], and adaptive recursion [28], this work treats inference-time compute as repeated hidden-state computation. LoopUS is most similar to retrofitting-based approaches, but differs in its use of block decomposition to ground the loop, selective gating and random deep supervision to explicitly stabilize latent refinement, and a learned confidence head to enable adaptive computation.

Deep Learning Gating Mechanisms. Gating mechanisms have long been used to regulate state updates in recurrent and deep networks [29–31]. For our setting, the key distinction is between softmax-style gating, which normalizes scores across alternatives [32], and decay-style gating, which directly controls state retention [33]. Recent sequence models increasingly adopt the latter approach, ranging from simple exponential decay to Mamba-style input-dependent selective decay [34, 33, 35– 37]. LoopUS follows this Mamba-style perspective in the depth domain, using an input-dependent exponential decay gate that is well-suited for iterative refinement.

### 3 Looped Depth Up-Scaling (LoopUS)

#### 3.1 Recasting LLM as a Looped LLM

As shown in Figure 2 (a), LoopUS partitions a pretrained LLM into an encoder E, a reasoning block M, and a decoder D. Following Mi:DM [38], we choose this front-middle-back split based on cosine-similarity analysis across depth, placing the encoder-reasoning and reasoning-decoder boundaries near the layers where the similarity profile changes most abruptly.

Given an input sequence, LoopUS applies the encoder once to obtain the initial representation:

h(0) = E(x0:T), h(0) ∈ RT×h. (1) It then performs B loop iterations. For b = 0,...,B − 1, the reasoning block proposes an update, and the selective gate incorporates it into the current hidden state:

R(h(b)) = G M,h(b) ,h(b+1) = R(h(b)),h(B) = R ◦ R ◦ ··· ◦ R

##### h(0) (2)

B iterations

Here G is introduced in Section 3.1. After B iterations, the decoder maps the final refined state to vocabulary logits, ℓ(B) = D(h(B)).

Decoder Block 𝒟

|Output Pr|[Figure 1]<br><br>ojection|
|---|---|
|[Figure 2]<br><br>Transforme|r Block N|
|Transformer|[Figure 3]<br><br>Block N-1|
| | |
|[Figure 4]<br><br>Confiden|ce Head|
|Reasoning Block ℳ| |
|[Figure 5]<br><br>Transformer|Block 𝑁𝑆|
| | |
|Transformer|[Figure 6]<br><br>Block 𝑁𝐸 + 2|
|Transformer|[Figure 7]<br><br>[Figure 8]<br><br>Block 𝑁𝐸 + 1|
|Encoder Block ℰ| |
|[Figure 9]<br><br>Transformer|Block 𝑁𝐸|
| | |
|Transforme|[Figure 10]<br><br>r Block 0|
| | |
|Input Em|[Figure 11]<br><br>bedding|
| | |

[Figure 12]

Forward Flow Backward Flow

Output Projection

ℎ𝐵.𝑑𝑒𝑡𝑎𝑐ℎ()

|[Figure 13]<br><br>Reasoning Block ℳ|
|---|

[Figure 14]

Transformer Block N

𝒉𝑩 Random Selection

ℎ𝐵−1.𝑑𝑒𝑡𝑎𝑐ℎ()

[Figure 15]

|[Figure 16]<br><br>Reasoning Block ℳ|
|---|

Transformer Block N-1

[Figure 17]

Confidence Head

ℒ𝑄𝐵−2

× 𝐵

𝒉𝑩−𝟐.𝑑𝑒𝑡𝑎𝑐ℎ()

[Figure 18]

...

Transformer Block N-2

- 𝜕ℎ𝐵−2
- 𝜕ℎ𝐵−3

|[Figure 19]<br><br>Reasoning Block ℳ|
|---|

Rea

[Figure 20]

ℒ𝐿𝑀𝐵−2 ℒ𝑀𝑜𝑛𝑜𝐵−2

Decoder Block 𝒟

...

ℎ𝐵−3.𝑑𝑒𝑡𝑎𝑐ℎ()

[Figure 21]

Selective Gating

......

...

- Transformer Block 0

[Figure 22]

- Transformer Block 1

[Figure 23]

- Transformer Block 2

[Figure 24]

- Transformer Block 3

𝒢

Recasting

|[Figure 25]<br><br>Reasoning Block ℳ|
|---|

ℎ2.𝑑𝑒𝑡𝑎𝑐ℎ()

|[Figure 26]<br><br>Reasoning Block ℳ|
|---|

[Figure 27]

Confidence

Head ℒ𝑄1

[Figure 28]

𝒉𝟏.𝑑𝑒𝑡𝑎𝑐ℎ()

𝜕ℎ1 𝜕ℎ0

|[Figure 29]<br><br>Reasoning Block ℳ|
|---|

ℒ𝐿𝑀1 ℒ𝑀𝑜𝑛𝑜1

[Figure 30]

[Figure 31]

Decoder Block 𝒟

Input Embedding

ℎ0.𝑑𝑒𝑡𝑎𝑐ℎ()

|[Figure 32]<br><br>Encoder Block ℰ|
|---|

(a) Recasting a Pretrained LLM as a Looped LLM (b) Training Objectives for a Looped LLM

- Figure 2: Overview of the LoopUS architecture. (a) A pretrained LLM is recast into encoder, reasoning, and decoder blocks, using a selective gate (G) inserted between loop iterations to stabilize the loop dynamics. (b) The looped LLM is trained with random deep supervision using next-token prediction loss (LLM), monotonicity loss (LMono), and confidence loss (LQ).

Selective Gating for Stable Loop Dynamics. Naively reapplying a pretrained middle block induces hidden-state drift, as it was originally optimized for single-pass execution rather than as a recurrent operator [10]. Therefore, a stable latent workspace requires a structural condition restricting each update to a damped refinement. LoopUS realizes this via a selective gate that interpolates proposed updates with the previous state. This dampens latent-space displacement and steers the trajectory toward regions increasingly favoring the correct answer. Figure 3 visualizes this: each gated refinement preserves part of the prior representation while making a directed move toward an answer-supporting latent subspace. Consequently, LoopUS incorporates an input-dependent

ℳ(ℎ(0)) ℎ(1)

ℎ𝐵:ℎ𝑖𝑑𝑑𝑒𝑛 𝑠𝑡𝑎𝑡𝑒

ℎ(2)

ℳ ⋅ :𝑅𝑒𝑎𝑠𝑜𝑛𝑖𝑛𝑔 𝐵𝑙𝑜𝑐𝑘

ℳ(ℎ(1))

Answer𝑅𝑒𝑔𝑖𝑜𝑛

𝛼(1)

𝛼(2)

𝛼𝐵:𝑆𝑒𝑙𝑒𝑐𝑡𝑖𝑣𝑒 𝐺𝑎𝑡𝑒

ℎ(3)

𝛼(0)

𝛼(3)

ℎ(4)

ℳ(ℎ(2))

ℎ(0)

ℳ(ℎ(3))

Answer

𝛼(4)

ℎ(5)

ℎ(𝑏+1) = 𝛼(𝑏)⨀ℳ ℎ(𝑏) + 1 − 𝛼(𝑏) ℎ(𝑏)

ℳ(ℎ(4))

Figure 3: Conceptual view of latent refinement in LoopUS. As the reasoning block is looped, each proposed update is mixed with the previous hidden state by the selective gate, gradually steering the trajectory toward the answer region instead of allowing it to drift.

selective gate after each reasoning iteration. Given the current hidden state h(b), the gate first measures the residual change proposed by the reasoning block and maps it to a positive per-token, per-channel step size:

δ(b) = M(h(b)) − h(b), ∆(b) = softplus W∆δ(b) + b∆ . (3)

Since the pretrained block M is highly nonlinear and lacks strict Lipschitz bounds, guaranteeing a formal global contraction is intractable. To effectively mitigate unconstrained drift, LoopUS instead enforces a relaxed, contraction-like iteration. Using a learned channel-wise decay coefficient A ∈ R<0, it computes a discrete decay factor,

α(b) = exp ∆(b) ⊙ A , (4)

ensuring α(b) ∈ (0,1) elementwise, an approach that shares conceptual synergy with input-dependent decay mechanisms in recent sequence models like Mamba [33]. The subsequent hidden state is then obtained by interpolating between the proposed update and the prior state:

h(b+1) = G M,h(b) = α(b) ⊙ M(h(b)) + 1 − α(b) ⊙ h(b) (5)

Since α(b) ∈ (0,1), Equation 5 provides a convex interpolation between the proposed update and the previous state. Although this convex combination does not mathematically guarantee the entire composite operator is a strict contraction, it restricts the maximal stride of each update. Consequently, each iteration acts as a damped relaxation step—analogous to an Euler integration step in a bounded

vector field—rather than an extrapolative update that might amplify drift. Specifically, larger values of α(b) weight the new update more heavily, whereas smaller values preserve more of the prior state:

h(b+1) − h(b) = α(b) ⊙ M(h(b)) − h(b) , (6) or, expressed in vector form:

h(b+1) = h(b) + P(b) M(h(b)) − h(b) , P(b) = Diag(α(b)). (7)

Under a continuous-time analogy, this recursion corresponds to a forward Euler step for the statedependent ordinary differential equation:

h˙ = P(h)(M(h) − h), (8)

where P(h) acts as a diagonal preconditioner induced by the gate. Because its diagonal entries lie strictly in (0,1), the gate applies a damped step size along each coordinate. The discrete update therefore realizes a diagonally preconditioned, relaxed fixed-point iteration toward h⋆ = M(h⋆), where the data-dependent step sizes serve as an implicit per-coordinate regularizer. This design encourages contraction-like behavior across loop iterations, as empirically confirmed in Section 4.5, enabling the stable reuse of pretrained middle layers without architectural modification.

Adaptive Computation via Early Stopping Mechanism. To enable adaptive computation at inference time, LoopUS augments each reasoning step with a confidence-based stopping rule. After the b-th refinement step, the confidence head produces a raw logit and its corresponding probability:

##### q˜(b) = qϕ h(b) , q(b) = σ q ˜(b) . (9)

The model compares q(b) against a predefined threshold qth, continuing to refine the representation while q(b) < qth and halting once q(b) ≥ qth. This reflects the adaptive-computation principle of Less is More [39]: additional loop steps are allocated only when the current latent state lacks sufficient confidence. In this way, pretrained transformer depth is dynamically converted into adaptive TTC.

Random Deep Supervision for Loop Training. Backpropagating through all loop steps would tightly couple the fully unrolled graph, rendering training memory-intensive and unstable [13]. Thus, LoopUS employs random deep supervision [8]: for each training batch, the model is unrolled for B steps, but gradients are computed only for a uniformly sampled subset of steps S ⊆ {0,...,B − 1} with size |S| = K. Steps in S receive normal gradient updates, whereas the intermediate steps are executed without gradient tracking (no_grad) and detached before the subsequent iteration, effectively blocking gradient flow through unsupervised depths. Coupled with the stabilizing effect of the selective gate, this strategy trains the model to halt robustly at diverse stopping depths while circumventing the prohibitive cost of full BPTT [40].

#### 3.2 Training Objective

As illustrated in Figure 2(b), LoopUS is trained by jointly optimizing a next-token prediction loss, a monotonicity loss, and a confidence loss at each sampled depth b ∈ S.

Overall Objective. At a sampled depth b ∈ S, the total per-step loss is defined as:

L(b) = L(LMb) + L(monob) + L(Qb), (10) where L(LMb) and L(monob) optimize latent refinement, and L(Qb) trains early stopping. Refinement Losses. To optimize latent refinement, we employ an autoregressive cross-entropy loss alongside a monotonicity regularizer. The primary supervision acts on the updated logits:

##### L(LMb) = CE D(h(b)),x2:T , (11)

which directly drives the refined latent state to deliver better predictive distributions. To prevent detrimental updates, we evaluate the pre-update state and systematically penalize predictive regressions:

L(monob) = SiLU L(LMb) − L(LMb−1) . (12) This monotonicity term penalizes updates that degrade the subsequent prediction loss, while remaining negligible for updates that preserve or enhance predictive quality. We adopt the SiLU activation [41] because, unlike ReLU [42] or SELU [43], it yields small negative values for minor improvements while asymptoting to zero for large negative arguments. This softly rewards beneficial refinements, encourages the loop to progress via small, stable updates, and stabilizes training without enabling the monotonicity penalty to dominate the primary objective LLM. Effectively, the monotonicity term enforces a gradual decay in the task-aligned surrogate error across successive loop iterations.

h = Encoder(x) sampled = RandomSampler(B, K) for b in range(B):

# Supervised step if b in sampled:

h_prev = h h_prop = Reasoner(h) h = Gate(h_prev, h_prop) q_logit = ConfidenceHead(h) y_hat = Decoder(h) y_prev = Decoder(h_prev) loss = CE(y_hat, y_true) # L_LM loss += SiLU(CE(y_hat, y_true)

- CE(y_prev, y_true)) # L_Mono loss += BCEWithLogits(q_logit,

(y_hat == y_true)) # L_Q

loss.backward() opt.step() opt.zero_grad() h = h.detach()

else: # Unsupervised step with no_grad():

h = Gate(h, Reasoner(h)) h = h.detach()

Figure 4: Pseudocode of LoopUS.

Confidence Loss. To train adaptive stopping, we supervise the post-update confidence logit q˜(b+1) with per-sample token accuracy,

L(Qb) = BCEWithLogits(˜q(b+1),qtarget(b) ),

(13)

T−1

1 Tvalid

qtarget(b) =

1 x ˆ(jb+1) = xj+1 ,

j=1

This formulation yields a lightweight stopping criterion that requires only a single scalar prediction per step, avoiding the extra statistics required by convergence-based [7] or cumulative distribution function (CDF)-based adaptive rules [4]. Together, these terms train LoopUS to make each loop step predictive, avoid regressive updates, and estimate whether further computation is unnecessary.

### 4 Empirical Validation

#### 4.1 Evaluation Protocol

We evaluate LoopUS across five pretrained backbones spanning model families and scales: Qwen3-

- 1.7B, Qwen3-4B, and Qwen3-8B [44], using cloud NVIDIA L40S, RTX PRO 6000, and RTX PRO

6000 GPUs, respectively; TinyLlama [45], using NVIDIA L40S GPUs; and Phi-4 [46], using NVIDIA H200 GPUs. Unless otherwise stated, models are trained on FineWeb-Edu [47] with 3B tokens, a context length of 1024, the AdamW optimizer, a cosine learning-rate schedule, bf16 mixed precision, and the default LoopUS setting of B = 20 total loop steps with K = 5 supervised depths per batch. Models are evaluated with lm-evaluation-harness [48]. We report perplexity on WikiText [49] and Lambada [50], and accuracy on MMLU [51], HellaSwag (HS) [52], ARC-Easy (ARC-E), ARCChallenge (ARC-C) [53], PIQA [54], WinoGrande (WG) [55], and OpenBookQA (OBQA) [56]. Unless otherwise noted, inference uses a maximum recursion budget of 8 with confidence-based stopping and KV caching. Full details are provided in Appendix A.

#### 4.2 Backbone-Level Evaluation across Model Scales

- Table 1: LoopUS improves pretrained backbones across scales. Results on language modeling and downstream benchmarks. ppl denotes perplexity (lower is better), and acc denotes accuracy (higher is better). AVG is the mean over the seven acc benchmarks, and ∆ denotes the change in AVG from the original backbone (w/o LoopUS) to the adapted checkpoint (w/ LoopUS). Bold highlights the better result between the two variants of each backbone. All models are evaluated zero-shot.

Model Setting

Wiki LAMBADA MMLU HS ARC-E ARC-C PIQA WG OBQA

AVG ∆ ppl ↓ acc ↑

Qwen 1.7B

w/o LoopUS 21 12.21 55.4 46.2 72.5 40.2 72.2 61.3 28 53.7 –

w/ LoopUS 16.9 7.43 56.6 46.3 74.9 43.1 73.3 63.0 29.6 55.3 +1.6 Qwen 4B

w/o LoopUS 16.4 7.29 68.3 52.1 80.2 50.4 75.0 66.5 29.4 60.3 –

w/ LoopUS 13.9 5.33 67.7 51.4 81.3 54.0 76.8 68.9 34.4 62.1 +1.8 Qwen 8B

w/o LoopUS 12.2 4.58 72.8 57.2 81.5 55.4 76.3 67.9 31.6 63.2 –

w/ LoopUS 10.3 4.32 71.5 56.0 83.9 58.1 78.9 72.4 37.0 65.4 +2.2 Phi-4 14B

w/o LoopUS 9.59 4.03 76.9 63.1 81.3 55.8 80.7 77.0 34.0 67.0 – w/ LoopUS 7.75 3.49 77.5 60.58 83.5 57.7 81.8 77.5 41.8 68.6 +1.7

LoopUS reuses pretrained computation by partitioning the backbone into encoder, reasoning, and decoder blocks while preserving the external decoding interface. Table 1 shows that this recasting yields consistent gains across models, reducing WikiText and LAMBADA perplexities and improving average downstream accuracy by +1.6 to +2.2 points, with the clearest gains on ARC-C and OBQA.

The effect is task-dependent: MMLU and HS remain close to the base models, whereas ARC-C, PIQA, WG, and OBQA improve more consistently. This pattern suggests that LoopUS is most useful when extra latent computation can refine a decision process, and less so when performance depends more on broad knowledge retrieval or on already strong single-pass predictions. The same reasoning-oriented trend holds across model scales, indicating that architectural recasting provides a stable post-training modification rather than a task-specific patch.

4.3 Comparison with Prior Methods under Limited Training Budgets

- Table 2: LoopUS shows adaptation efficiency under a smaller training-token budget. All methods adapt a TinyLlama-based backbone; w/o and w/ LoopUS denote the checkpoint before and after adaptation, respectively. AVG is the unweighted mean over the six tasks, and ∆ reports the change in AVG from Original to Adapted. Results for prior methods are taken from the corresponding papers.

Task

Base Model

Train Tokens

Method

Setting

ARC-E ARC-C HS WG PIQA OBQA AVG ∆ acc_n ↑ acc_n ↑ acc_n ↑ acc ↑ acc_n ↑ acc_n ↑

TinyLlama 1.1B

Original 47.1 25.1 42.2 53.4 66.8 24.2 43.1 – Adapted 53.0 29.6 55.5 57.9 69.8 30.6 49.4 +6.3

Ours

3B

Original 55.7 31.0 59.1 58.9 73.0 35.0 52.1 – Adapted 58.6 35.6 45.1 57.6 66.4 32.2 49.3 -2.9

McLeish et al. (2025)

TinyLlama 1.1B-3T

52B

Bae et al. (2025)

TinyLlama 1.1B

Original 44.7 23.2 42.2 53.4 66.8 29.2 43.3 – Adapted 49.9 26.2 48.8 54.1 68.6 32.8 46.7 +3.5

60B

LoopUS is designed to keep loop training stable and adaptation-efficient through selective gating and sparse supervision across depths. Table 2 shows the practical effect of this design choice on a

shared six-task reasoning suite. Since prior results are drawn from the corresponding papers, we treat this comparison as an adaptation-efficiency reference rather than a fully controlled head-to-head benchmark. In this comparison, LoopUS achieves the largest average gain (∆= + 6.3), compared with ∆=−2.9 for McLeish et al. [10] and ∆=+3.5 for Bae et al. [11], while using fewer additional training tokens. These results suggest that LoopUS improves adaptation efficiency not simply by adding recurrence but by preserving and reusing pretrained computation through decomposition, selective gating, and random deep supervision.

#### 4.4 Inference-Time Recursion-Depth Analysis

Avg recursion=3.39

LAMBADA

ARC-E

WikiText

7.5

n=1: 842.52

n=1: 0.677

n=1: 24985.15

16.5

0.820

7.0

16.0

0.815

Accuracy

LDS Looped

LDS Looped

15.5

6.5

0.810

PPL

PPL

Best Orig Adaptive Computing

Best Orig Adaptive Computing

15.0

6.0

0.805

LDS Looped

14.5

Best Orig Adaptive Computing

0.800

5.5

14.0

0.795

5.0

12 468101214161820 40 80 100

12 468101214161820 40 80 100

12 468101214161820 40 80 100

Thinking Iteration

Thinking Iteration

Thinking Iteration

- Figure 5: Test-Time Scaling of LoopUS. On Qwen3-4B, benchmark performance is plotted against the number of latent reasoning iterations used at inference time. The dashed gray line denotes the original backbone, the dashed orange line denotes the trained LoopUS checkpoint, and the star marks the best observed depth for each task.

LoopUS uses a confidence-based stopping rule to allocate TTC adaptively. Figure 5 shows that most of the benefit is obtained within only a few iterations, after which additional recursion yields diminishing returns while remaining stable rather than diverging. This stability extends well beyond the training regime: the checkpoint continues to behave robustly even at unseen recursion depths such as 40, 80, and 100. With adaptive stopping enabled, the same checkpoint halts after 3.39 iterations on average out of a maximum budget of 8, yet remains close to the best observed performance. These results suggest that the confidence head does not merely stop early; it learns to identify an effective stopping point quickly and allocate extra refinement only when it is useful.

4.5 Dynamics of Stable Latent Refinement

0 5000 10000 15000

Training step

0.05

0.00

0.05

0.10

0.15

0.20

0.25

MONOLoss

iter 0 iter 2 iter 4 iter 8

iter 12 iter 16 iter 19

0 5000 10000 15000

Training step

2

4

6

8

LMLoss

iter 0 iter 2 iter 4 iter 8

iter 12 iter 16 iter 19

0 5000 10000 15000

Training step

0.2

0.3

0.4

0.5

0.6

0.7

QAccuracy

iter 0 iter 2 iter 4 iter 8

iter 12 iter 16 iter 19

- Figure 6: Training organizes the loop into a stable refinement process. We plot the step-wise monotonicity loss, next-token prediction loss, and confidence-head Accuracy for loop indices {0,2,4,8,12,16,19}.

LoopUS trains each loop step as a damped corrective update through selective gating and a monotonicity-aware objective. This is consistent with recent energy-based views of autoregressive modeling, where extra latent computation acts as iterative refinement toward more compatible states [57, 7]. Figure 6 shows this behavior emerging during training: the monotonicity loss decreases toward zero across loop positions, while the next-token prediction loss and confidence loss remain well-behaved across shallow and deep unrolls. Training therefore encourages each transition to be a small, stable corrective edit rather than an unstable depth expansion.

Figures 7 and 8 show the same Qwen3-4B example for the prompt “32 * 64 =” from latent- and token-space perspectives, respectively. The latent trajectory makes its largest move in the first few iterations and then contracts, indicating convergence toward a stable answer region. Consistently, the correct next token “2” rises from 2.17 × 10−5% at iteration 0 to 81.9% after one refinement

###### 32 * 64 =

h0

1.0

100

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | |cos euc<br><br>|ine lidean|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.5

mean ± std

EuclideanDistance

PC-2(15.5%var)

CosineDistance

h23456h7891119203426578

0.8

mean

0.4

- 100
- 101
- 102

0

0.6

Alpha

0.3

100

0.4

0.2

0.1

0.2

200

h1

0.0

0.0

400 200 0

1 5 9 13 17 20

1 5 9 13 17 20

PC-1 (82.2% var)

Thinking Iteration

Thinking Iteration

- Figure 7: The learned loop induces convergent trajectories. The largest latent-space movement occurs in the earliest iterations, after which the step-to-step distance contracts, indicating that the latent trajectory approaches a fixed point rather than diverging.

2lectual 1 6reeting 3ergencystructors 5 4isticallyluetooth

Token

0

20

40

60

80

100

Probability(%)

- iter 0

- iter 1

- iter 2

- iter 3

- iter 4

32 * 64 =

- Figure 8: Loop updates translate into token-level predictive refinement. Across iterations, probability mass shifts across candidate tokens, showing how latent updates refine the next-token prediction.

step and to about 89.8% by iteration 4, while the remaining candidates lose most of their mass early on. Together with Figure 6, these results suggest that LoopUS uses a large initial corrective update followed by smaller, convergent refinements that sharpen the final prediction.

4.6 Component Ablation Study

0 1000 2000 3000 4000 5000

Steps

2

4

6

8

10

LMLoss

Selective Gate: w/ vs w/o

w/ Selective Gate

w/o Selective Gate

0 1000 2000 3000 4000 5000

Steps

2

4

6

8

10

12

LMLoss

Encoder-Decoder: w/ vs w/o

w/ Encoder-Decoder

w/o Encoder-Decoder

0 1000 2000 3000 4000 5000

Steps

2

4

6

8

10

LMLoss

Random vs No Random

Random

No Random

0 1000 2000 3000 4000 5000

Steps

2

- 4

6

8

10

LMLoss

Decay vs Sigmoid

Decay

Sigmoid

0 1000 2000 3000 4000 5000

Steps

2

4

6

8

10

LMLoss

Activation Functions

SiLU

SELU ReLU SoftPlus

0 1000 2000 3000 4000 5000

Steps

2

4

6

8

10

LMLoss

DeepSupervision vs TBPTT

DeepSupervision

TBPTT

- Figure 9: Ablation study of LoopUS components. We report average LLM over 20 runs after (a) removing the selective gate, (b) removing the encoder-decoder decomposition, (c) training without random deep supervision, (d) replacing the decay gate with sigmoid gating, (e) changing the monotonicity-loss activation among ReLU, SiLU, SELU, and SoftPlus, and (f) comparing the standard LoopUS training recipe against TBPTT.

Figure 9 analyzes how LLM changes when key components of LoopUS are removed or replaced. (a) Removing the selective gate causes convergence to a higher LLM because it eliminates the damped

interpolation that preserves the previous hidden state, thereby weakening drift-controlled latent refinement. (b) Removing the encoder–decoder decomposition also leads to a higher final LLM. This demonstrates that without an explicit separation between representation extraction, latent refinement, and output decoding, the loop fails to preserve the pretrained latent workspace. Instead, it relearns a less stable recurrent trajectory that converges to a worse optimum. (c) Training without random deep supervision destabilizes optimization and slows convergence, even if the final LLM remains similar. This highlights that sparse supervision across depths is critical for efficiently training long loops in practice. (d) Replacing the decay gate with sigmoid gating also makes optimization less stable and yields a higher final LLM, indicating that the decay-style gate better supports stable long-loop training. (e) Among the tested activation functions for the monotonicity term, SiLU [41] provides the most reliable optimization behavior. ReLU [42], SELU [43], and SoftPlus [58] each lead to less stable or less favorable trajectories, which supports the design choice used in the main LoopUS recipe. (f) TBPTT [59] incurs higher computational cost while plateauing at a substantially higher LLM than the standard LoopUS training recipe, indicating lower efficiency and worse performance in this setting.

### 5 Conclusion

This paper presents Looped Depth Up-Scaling (LoopUS), a post-training framework that recasts a pretrained LLM into a looped latent-refinement model through encoder–reasoning–decoder decomposition, a selective gate, random deep supervision with stepwise detachment, and a lightweight confidence head for adaptive stopping. Across diverse model scales, LoopUS improves pretrained backbones while preserving standard interfaces, yielding enhanced reasoning performance, consistent perplexity reductions, and high adaptation efficiency under limited training budgets. Our analyses suggest that these gains arise from controlled latent refinement rather than uncontrolled depth expansion. Training remains well behaved across loop depths, with hidden-state trajectories contracting through diminishing corrections, token distributions becoming sharper, and adaptive halting allocating computation where it is most useful. Ablations further confirm that the selective gate, architectural decomposition, and random deep supervision are central to making long latent loops effective and trainable. Overall, latent looping provides a practical way to turn pretrained transformer depth into an adaptive allocation of test-time compute and stronger task-aligned inference.

### References

- [1] Aakash Lahoti, Kevin Li, Berlin Chen, Caitlin Wang, Aviv Bick, J Zico Kolter, Tri Dao, and Albert Gu. Mamba-3: Improved sequence modeling using state space principles. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=HwCvaJOiCj.
- [2] Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for LLM problem-solving. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=VNckp7JEHn.
- [3] Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM testtime compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=4FWAwZtd2n.
- [4] Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, et al. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741, 2025.
- [5] Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, and Se-Young Yun. Mixture-ofrecursions: Learning dynamic recursive depths for adaptive token-level computation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=QuqsEIVWIG.
- [6] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute

- with latent reasoning: A recurrent depth approach, 2025. URL https://arxiv.org/abs/ 2502.05171.
- [7] Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, and Tariq Iqbal. Energy-based transformers are scalable learners and thinkers. In The Fourteenth International Conference on Learning Representations,

2026. URL https://openreview.net/forum?id=ZBj3Qp1bYg.

- [8] Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model, 2025. URL https://arxiv.org/abs/ 2506.21734.
- [9] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=KnqiC0znVF.
- [10] Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, and Micah Goldblum. Teaching pretrained language models to think deeper with retrofitted recurrence, 2025. URL https: //arxiv.org/abs/2511.07384.
- [11] Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster. Relaxed recursive transformers: Effective parameter sharing with layer-wise loRA. In The Thirteenth International Conference on Learning Representations, 2025. URL https:// openreview.net/forum?id=WwpYSOkkCt.
- [12] Guanxu Chen, Dongrui Liu, and Jing Shao. Loop as a bridge: Can looped transformers truly link representation space and natural language outputs?, 2026. URL https://arxiv.org/ abs/2601.10242.
- [13] Ilya Sutskever. Training Recurrent Neural Networks. PhD thesis, University of Toronto, Canada, 2013.
- [14] Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In International conference on machine learning, pages 1310–1318. Pmlr, 2013.
- [15] Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, Tianle Cai, Taylor Kergan, Assel Kembay, Andrew Smith, Chenghua Lin, Binh Nguyen, Yuqi Pan, Yuhong Chou, Zefan Cai, Zhenhe Wu, Yongchi Zhao, Tianyu Liu, Jian Yang, Wangchunshu Zhou, Chujie Zheng, Chongxuan Li, Yuyin Zhou, Zhoujun Li, Zhaoxiang Zhang, Jiaheng Liu, Ge Zhang, Wenhao Huang, and Jason Eshraghian. A survey on latent reasoning, 2025. URL https://arxiv.org/abs/2507.06203.
- [16] Keigo Shibata, Kazuki Yano, Ryosuke Takahashi, Jaesung Lee, Wataru Ikeda, and Jun Suzuki. Suppressing final layer hidden state jumps in transformer pretraining, 2026. URL https: //arxiv.org/abs/2601.18302.
- [17] Pavel Tikhonov and Dmitry Ilvovsky. Frozen in the middle: Hidden states remain unchanged across intermediate layers of language models. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, CIKM ’25, page 5289–5293, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400720406. doi: 10.1145/3746252.3760890. URL https://doi.org/10.1145/3746252.3760890.
- [18] David Noel Ng. Llm neuroanatomy: How i topped the llm leaderboard without changing a single weight. March 2026. URL https://dnhkng.github.io/posts/rys/.
- [19] Anthropic. Mapping the mind of a large language model, 2024. URL https://www. anthropic.com/research/mapping-mind-language-model.
- [20] Anthropic. On the biology of a large language model, 2025. URL https:// transformer-circuits.pub/2025/attribution-graphs/biology.html.

- [21] nostalgebraist. Interpreting GPT: The logit lens. LessWrong, 2020. URL https://www. lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.
- [22] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. The bottom-up evolution of representations in the transformer: A study with machine translation and language modeling objectives. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 4395–4405, 2019. URL https://aclanthology.org/D19-1448/.
- [23] Sanghoon Kim, Dahyun Kim, Chanjun Park, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee, Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, and Sunghun Kim. SOLAR 10.7B: Scaling large language models with simple yet effective depth up-scaling. In Yi Yang, Aida Davani, Avi Sil, and Anoop Kumar, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pages 23–35, Mexico City, Mexico, June

2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-industry.3. URL https://aclanthology.org/2024.naacl-industry.3/.

- [24] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id=_VjQlMeSB_J.
- [25] Boyi Zeng, Shixiang Song, Siyuan Huang, Yixuan Wang, He Li, Ziwei He, Xinbing Wang, Zhiyu Li, and Zhouhan Lin. Pretraining language models to ponder in continuous space, 2025. URL https://arxiv.org/abs/2505.20674.
- [26] Yue Fu, Shruti Rijhwani, Graham Neubig, and Yonatan Bisk. Think-at-hard: Teaching small language models to think on hard problems, 2025. URL https://arxiv.org/abs/2506. 04458.
- [27] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers, 2019. URL https://arxiv.org/abs/1807.03819.
- [28] Rui-Jie Zhu et al. Ouro: A latent reasoning model with adaptive depth via gated recurrence,

2025. URL https://arxiv.org/abs/2507.07919.

- [29] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735–1780, 1997.
- [30] Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder– decoder for statistical machine translation. In Alessandro Moschitti, Bo Pang, and Walter Daelemans, editors, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724–1734, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1179. URL https://aclanthology.org/ D14-1179/.
- [31] Rupesh Kumar Srivastava, Klaus Greff, and J"urgen Schmidhuber. Training very deep networks. Advances in Neural Information Processing Systems Workshop on Deep Learning, 2015. URL https://arxiv.org/abs/1507.06228.
- [32] Sunghyun Sim, Dohee Kim, and Hyerim Bae. Correlation recurrent units: A novel neural architecture for improving the predictive performance of time-series data. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(12):14266–14283, 2023. doi: 10.1109/TPAMI.

- 2023.3319557.

[33] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces,

- 2024. URL https://arxiv.org/abs/2312.00752.

- [34] Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=uYLFoz1vlAC.
- [35] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael K Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Extended long short-term memory. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=ARAxPPIAhq.
- [36] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024.
- [37] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=r8H7xhYPwz.
- [38] Donghoon Shin, Sejung Lee, Soonmin Bae, Hwijung Ryu, Changwon Ok, Hoyoun Jung, Hyesung Ji, Jeehyun Lim, Jehoon Lee, Ji-Eun Han, et al. Mi: dm 2.0 korea-centric bilingual language models. arXiv preprint arXiv:2601.09066, 2026.
- [39] Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks, 2025. URL https://arxiv.org/abs/2510.04871.
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.
- [41] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks, 107:3–11, 2018.
- [42] Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th international conference on machine learning (ICML-10), pages 807–814, 2010.
- [43] Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing neural networks. Advances in neural information processing systems, 30, 2017.
- [44] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [45] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model, 2024.
- [46] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 technical report, 2024. URL https://arxiv.org/abs/2412.08905.
- [47] Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/ forum?id=n6SCkn2QaG.

- [48] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/ 10256836.
- [49] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017. URL https:// openreview.net/forum?id=Byj72udxe.
- [50] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Katrin Erk and Noah A. Smith, editors, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1144. URL https://aclanthology.org/P16-1144/.
- [51] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021. URL https: //arxiv.org/abs/2009.03300.
- [52] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence?, 2019. URL https://arxiv.org/abs/1905.07830.
- [53] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.
- [54] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language, 2019. URL https://arxiv.org/abs/ 1911.11641.
- [55] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale, 2019. URL https://arxiv.org/abs/1907. 10641.
- [56] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018. URL https: //arxiv.org/abs/1809.02789.
- [57] Mathieu Blondel, Michael E. Sander, Germain Vivier-Ardisson, Tianlin Liu, and Vincent Roulet. Autoregressive language models are secretly energy-based models: Insights into the lookahead capabilities of next-token prediction, 2026. URL https://arxiv.org/abs/2512.15605.
- [58] Charles Dugas, Yoshua Bengio, François Bélisle, Claude Nadeau, and René Garcia. Incorporating second-order functional knowledge for better option pricing. In T. Leen, T. Dietterich, and V. Tresp, editors, Advances in Neural Information Processing Systems, volume 13. MIT Press, 2000. URL https://proceedings.neurips.cc/paper_files/paper/2000/ file/44968aece94f667e4095002d140b5896-Paper.pdf.
- [59] F.A. Gers, J. Schmidhuber, and F. Cummins. Learning to forget: continual prediction with lstm. In 1999 Ninth International Conference on Artificial Neural Networks ICANN 99. (Conf. Publ. No. 470), volume 2, pages 850–855 vol.2, 1999. doi: 10.1049/cp:19991218.
- [60] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=mZn2Xyh9Ec.
- [61] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=St1giarCHLP.

- [62] Ruihan Xu, Yuting Gao, Lan Wang, Jianing Li, Weihao Chen, Qingpei Guo, Ming Yang, and Shiliang Zhang. Looping back to move forward: Recursive transformers for efficient and flexible large multimodal models. arXiv preprint arXiv:2602.09080, 2026.
- [63] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. URL https://qwen.ai/blog?id=qwen3.6-27b.
- [64] DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.
- [65] Eunbi Choi, Kibong Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Hyunjik Jo, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Haeju Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Heuiyeen Yeen, Hwan Chang, Stanley Jungkyu Choi, Yejin Choi, Jiwon Ham, Kijeong Jeon, Geunyeong Jeong, Gerrard Jeongwon Jo, Yonghwan Jo, Jiyeon Jung, Naeun Kang, Dohoon Kim, Euisoon Kim, Hayeon Kim, Hyosang Kim, Hyunseo Kim, Jieun Kim, Minu Kim, Myoungshin Kim, Unsol Kim, Youchul Kim, YoungJin Kim, Chaeeun Lee, Chaeyoon Lee, Changhun Lee, Dahm Lee, Edward Hwayoung Lee, Honglak Lee, Jinsang Lee, Jiyoung Lee, Sangeun Lee, Seungwon Lim, Solji Lim, Woohyung Lim, Chanwoo Moon, Jaewoo Park, Jinho Park, Yongmin Park, Hyerin Seo, Wooseok Seo, Yongwoo Song, Sejong Yang, Sihoon Yang, Chang En Yea, Sihyuk Yi, Chansik Yoon, Dongkeun Yoon, Sangyeon Yoon, and Hyeongu Yun. K-exaone technical report, 2026. URL https://arxiv. org/abs/2601.01739.
- [66] Sungrae Park, Sanghoon Kim, Jungho Cho, Gyoungjin Gim, Dawoon Jung, Mikyoung Cha, Eunhae Choo, Taekgyu Hong, Minbyul Jeong, SeHwan Joo, Minsoo Khang, Eunwon Kim, Minjeong Kim, Sujeong Kim, Yunsu Kim, Hyeonju Lee, Seunghyun Lee, Sukyung Lee, Siyoung Park, Gyungin Shin, Inseo Song, Wonho Song, Seonghoon Yang, Seungyoun Yi, Sanghoon Yoon, Jeonghyun Ko, Seyoung Song, Keunwoo Choi, Hwalsuk Lee, Sunghun Kim, Du-Seong Chang, Kyunghyun Cho, Junsuk Choe, Hwaran Lee, Jae-Gil Lee, KyungTae Lim, and Alice Oh. Solar open technical report, 2026. URL https://arxiv.org/abs/2601.07022.
- [67] NVIDIA, :, Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Kondratenko, Alexander Bukharin, Alexandre Milesi, Ali Taghibakhshi, Alisa Liu, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Amy Shen, Anahita Bhiwandiwalla, Andrew Tao, Anjulie Agrusa, Ankur Verma, Ann Guan, Anubhav Mandarwal, Arham Mehta, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asit Mishra, Asma Kuriparambil Thekkumpate, Ayush Dattagupta, Banghua Zhu, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Besmira Nushi, Bilal Kartal, Bita Darvish Rouhani, Boris Ginsburg, Brandon Norick, Brandon Soubasis, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Carlo del Mundo, Chantal Hwang, Charles Wang, Cheng-Ping Hsieh, Chenghao Zhang, Chenhan Yu, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Christopher Parisien, Collin Neale, Cyril Meurillon, Damon Mosk-Aoyama, Dan Su, Dane Corneil, Daniel Afrimi, Daniel Lo, Daniel Rohrer, Daniel Serebrenik, Daria Gitman, Daria Levy, Darko Stosic, David Mosallanezhad, Deepak Narayanan, Dhruv Nathawani, Dima Rekesh, Dina Yared, Divyanshu Kakwani, Dong Ahn, Duncan Riach, Dusan Stosic, Edgar Minasyan, Edward Lin, Eileen Long, Eileen Peters Long, Elad Segal, Elena Lantz, Ellie Evans, Elliott Ning, Eric Chung, Eric Harper, Eric Tramel, Erick Galinkin, Erik Pounds, Evan Briones, Evelina Bakhturina, Evgeny Tsykunov, Faisal Ladhak, Fay Wang, Fei Jia, Felipe Soares, Feng Chen, Ferenc Galko, Frank Sun, Frankie Siino, Gal Hubara Agam, Ganesh Ajjanagadde, Gantavya Bhatt, Gargi Prasad, George Armstrong, Gerald Shen, Gorkem Batmaz, Grigor Nalbandyan, Haifeng Qian, Harsh Sharma, Hayley Ross, Helen Ngo, Herbert Hum, Herman Sahota, Hexin Wang, Himanshu Soni, Hiren Upadhyay, Huizi Mao, Huy C Nguyen, Huy Q Nguyen, Iain Cunningham, Ido Galil, Ido Shahaf, Igor Gitman, Ilya Loshchilov, Itamar Schen, Itay Levy, Ivan Moshkov, Izik Golan, Izzy Putterman, Jan Kautz, Jane Polak Scowcroft, Jared Casper, Jatin Mitra, Jeffrey Glick, Jenny Chen, Jesse Oliver, Jian Zhang, Jiaqi Zeng, Jie Lou, Jimmy Zhang, Jinhang Choi, Jining Huang, Joey Conway, Joey Guman, John Kamalu, Johnny Greco, Jonathan Cohen, Joseph Jennings, Joyjit Daw, Julien Veron Vialard, Junkeun Yi, Jupinder Parmar, Kai Xu, Kan Zhu, Kari Briski, Katherine Cheung, Katherine Luna, Keith Wyss, Keshav Santhanam, Kevin Shih, Kezhi Kong, Khushi Bhardwaj, Kirthi Shankar, Krishna C. Puvvada, Krzysztof Pawelec, Kumar Anik, Lawrence McAfee, Laya Sleiman, Leon Derczynski, Li Ding, Lizzie Wei, Lucas Liebenwein, Luis Vega, Maanu Grover, Maarten Van Segbroeck, Maer Rodrigues

- de Melo, Mahdi Nazemi, Makesh Narsimhan Sreedhar, Manoj Kilaru, Maor Ashkenazi, Marc Romeijn, Marcin Chochowski, Mark Cai, Markus Kliegl, Maryam Moosaei, Matt Kulka, Matvei Novikov, Mehrzad Samadi, Melissa Corpuz, Mengru Wang, Meredith Price, Michael Andersch, Michael Boone, Michael Evans, Miguel Martinez, Mikail Khona, Mike Chrzanowski, Minseok Lee, Mohammad Dabbah, Mohammad Shoeybi, Mostofa Patwary, Nabin Mulepati, Najeeb Nabwani, Natalie Hereth, Nave Assaf, Negar Habibi, Neta Zmora, Netanel Haber, Nicola Sessions, Nidhi Bhatia, Nikhil Jukar, Nikki Pope, Nikolai Ludwig, Nima Tajbakhsh, Nir Ailon, Nirmal Juluru, Nishant Sharma, Oleksii Hrinchuk, Oleksii Kuchaiev, Olivier Delalleau, Oluwatobi Olabiyi, Omer Ullman Argov, Omri Puny, Oren Tropp, Ouye Xie, Parth Chadha, Pasha Shamis, Paul Gibbons, Pavlo Molchanov, Pawel Morkisz, Peter Dykas, Peter Jin, Pinky Xu, Piotr Januszewski, Pranav Prashant Thombre, Prasoon Varshney, Pritam Gundecha, Przemek Tredak, Qing Miao, Qiyu Wan, Rabeeh Karimi Mahabadi, Rachit Garg, Ran El-Yaniv, Ran Zilberstein, Rasoul Shafipour, Rich Harang, Rick Izzo, Rima Shahbazyan, Rishabh Garg, Ritika Borkar, Ritu Gala, Riyad Islam, Robert Hesse, Roger Waleffe, Rohit Watve, Roi Koren, Ruoxi Zhang, Russell Hewett, Russell J. Hewett, Ryan Prenger, Ryan Timbrook, Sadegh Mahdavi, Sahil Modi, Samuel Kriman, Sangkug Lim, Sanjay Kariyappa, Sanjeev Satheesh, Saori Kaji, Satish Pasumarthi, Saurav Muralidharan, Sean Narentharen, Sean Narenthiran, Seonmyeong Bak, Sergey Kashirsky, Seth Poulos, Shahar Mor, Shanmugam Ramasamy, Shantanu Acharya, Shaona Ghosh, Sharath Turuvekere Sreenivas, Shelby Thomas, Shiqing Fan, Shreya Gopal, Shrimai Prabhumoye, Shubham Pachori, Shubham Toshniwal, Shuoyang Ding, Siddharth Singh, Simeng Sun, Smita Ithape, Somshubra Majumdar, Soumye Singhal, Stas Sergienko, Stefania Alborghetti, Stephen Ge, Sugam Dipak Devare, Sumeet Kumar Barua, Suseella Panguluri, Suyog Gupta, Sweta Priyadarshi, Syeda Nahida Akter, Tan Bui, Teodor-Dumitru Ene, Terry Kong, Thanh Do, Tijmen Blankevoort, Tim Moon, Tom Balough, Tomer Asida, Tomer Bar Natan, Tomer Ronen, Tugrul Konuk, Twinkle Vashishth, Udi Karpas, Ushnish De, Vahid Noorozi, Vahid Noroozi, Venkat Srinivasan, Venmugil Elango, Victor Cui, Vijay Korthikanti, Vinay Rao, Vitaly Kurin, Vitaly Lavrukhin, Vladimir Anisimov, Wanli Jiang, Wasi Uddin Ahmad, Wei Du, Wei Ping, Wenfei Zhou, Will Jennings, William Zhang, Wojciech Prazuch, Xiaowei Ren, Yashaswi Karnati, Yejin Choi, Yev Meyer, Yi-Fu Wu, Yian Zhang, Yigong Qin, Ying Lin, Yonatan Geifman, Yonggan Fu, Yoshi Subara, Yoshi Suhara, Yubo Gao, Zach Moshe, Zhen Dong, Zhongbo Zhu, Zihan Liu, Zijia Chen, and Zijie Yan. Nvidia nemotron 3: Efficient and open intelligence, 2025. URL https://arxiv.org/abs/2512.20856.
- [68] Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie, Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui, Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu, Zhilin Yang, and Xinyu Zhou. Attention residuals, 2026. URL https://arxiv.org/abs/2603.15031.
- [69] Guiyao Tie, Zeli Zhao, Dingjie Song, Fuyang Wei, Rong Zhou, Yurou Dai, Wen Yin, Zhejian Yang, Jiangyue Yan, Yao Su, Zhenhan Dai, Yifeng Xie, Yihan Cao, Lichao Sun, Pan Zhou, Lifang He, Hechang Chen, Yu Zhang, Qingsong Wen, Tianming Liu, Neil Zhenqiang Gong, Jiliang Tang, Caiming Xiong, Heng Ji, Philip S. Yu, and Jianfeng Gao. A survey on post-training of large language models, 2025. URL https://arxiv.org/abs/2503.06072.
- [70] Zhanhui Zhou, Lingjie Chen, Hanghang Tong, and Dawn Song. dllm: Simple diffusion language modeling, 2026. URL https://arxiv.org/abs/2602.22661.

h = Encoder(x) sampled = RandomSampler(B, K) for b in range(B):

# Supervised step if b in sampled:

h_prev = h h_prop = Reasoner(h) h = Gate(h_prev, h_prop) q_logit = ConfidenceHead(h) y_hat = Decoder(h) y_prev = Decoder(h_prev) loss = CE(y_hat, y_true) # L_LM loss += SiLU(CE(y_hat, y_true)

- CE(y_prev, y_true)) # L_Mono loss += BCEWithLogits(q_logit,

(y_hat == y_true)) # L_Q

loss.backward() opt.step() opt.zero_grad() h = h.detach()

else: # Unsupervised step with no_grad():

h = Gate(h, Reasoner(h)) h = h.detach()

Figure 10: Pseudocode of LoopUS.

### A Experimental Details

#### A.1 Backbones and Training Data

We evaluate Qwen3-1.7B, Qwen3-4B, Qwen3-8B, TinyLlama, and Phi-4 backbones [44, 46]. Our reported main experiments use streaming training on FineWeb-Edu with the CC-MAIN-2025-26 configuration [47], a total budget of 3B tokens, and sequence length 1024. The released public reference recipes are built on the same data pipeline. When applying the LoopUS architecture, the models are unrolled by selecting specific layers as the encoder and decoder, reserving the intermediate layers as the reusable reasoning block:

- • Qwen3-1.7B: Encoder layers 0–1, Decoder layer 27.
- • Qwen3-4B: Encoder layers 0–1, Decoder layer 35.
- • Qwen3-8B: Encoder layers 0–5, Decoder layer 35.
- • Phi-4: Encoder layers 0–5, Decoder layer 39.
- • TinyLlama: Encoder layer 0, Decoder layer 21.

This formulation separates a shallow encoder from a late decoder while repurposing the entire middle transformer block as the looped latent workspace.

#### A.2 Optimization Details

Training is implemented with Accelerate and distributed data parallelism, with gradient checkpointing enabled throughout the run. The reference script uses AdamW with a learning rate of 5 × 10−5, one epoch over the token budget, bf16 mixed precision, FlashAttention-2 [60], cosine scheduling, 300 warmup steps, 8 dataloader workers, and pinned-memory data loading. Logging is performed every 50 steps, checkpoints are saved every 5000 steps, and at most 3 checkpoints are retained. The data pipeline reserves a small held-out slice with val_ratio = 10−4, although periodic validation is disabled in the released main training script by setting eval_interval = −1.

The loop-specific configuration uses 20 total reasoning steps with deep supervision on 5 loop positions per example, a training stopping threshold of 0.55, and the all stopping mode. The training code

also supports checkpoint-time lm-evaluation-harness runs; in the released script this option is enabled for WikiText with a limit of 200 samples.

#### A.3 Evaluation Details

We evaluate five pretrained backbones: Qwen3-1.7B, Qwen3-4B, Qwen3-8B, TinyLlama, and Phi-4. The reported training runs were conducted in cloud GPU environments, using NVIDIA L40S GPUs for Qwen3-1.7B and TinyLlama, NVIDIA RTX PRO 6000 GPUs for Qwen3-4B and Qwen3-8B, and NVIDIA H200 GPUs for Phi-4. Unless otherwise stated, models are trained on FineWeb-Edu with a budget of 3B tokens, context length 1024, AdamW, cosine learning-rate decay, bf16 mixed precision, and the default LoopUS setting of B=20 total loop steps with K=5 supervised depths per batch.

For evaluation, we use lm-evaluation-harness in a zero-shot setting and report perplexity on WikiText and Lambada, together with accuracy on MMLU, HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande, and OpenBookQA. Standard inference uses a maximum recursion budget of 8, a stopping threshold of 0.6, and KV caching for autoregressive decoding. The main experiments are organized to test three claims: that architectural recasting improves the pretrained backbone while preserving its language-modeling interface, that selective gating and random deep supervision improve adaptation behavior under limited budgets relative to prior retrofitting schemes, and that the resulting recurrence behaves as controlled iterative refinement rather than uncontrolled extra depth.

#### A.4 KV-Cache Implementation for Autoregressive Inference

Qwen3-1.7B

0.275 Qwen3-4B

Qwen3-8B

Secondspergeneratedtoken

Uncached

Uncached

Uncached

0.12

0.250

Cached

Cached

Cached

0.35

0.11

0.225

0.30

0.200

0.10

1.64x faster

2.31x faster

2.49x faster

0.25

0.175

0.09

0.150

0.20

0.08

0.125

0.15

4 32 256 512 1024

4 32 256 512 1024

4 32 256 512 1024

Generated tokens

Generated tokens

Generated tokens

- Figure 11: Effect of KV caching on LoopUS autoregressive decoding speed with the recursion budget fixed to B = 8. Across Qwen3-1.7B, Qwen3-4B, and Qwen3-8B, caching consistently reduces seconds per generated token, with the largest gains appearing at longer generations.

Let x1:t denote the already processed prefix at autoregressive decoding step t, and let B denote the maximum recursion budget used for that forward pass. We use t for decoding time and b for latent refinement depth. LoopUS stores the autoregressive cache state as

Ct = Ctenc,{Ctrea,b}Bb=1,Ctdec,st , (14)

where Ctenc is the encoder KV cache, Ctrea,b is the KV cache of the b-th refinement depth, Ctdec is the decoder KV cache, and st is the number of previously seen tokens. In the implementation, these objects are instantiated as separate Hugging Face DynamicCaches plus a seen_tokens counter. Conceptually, inference proceeds in two phases. In the prefill phase, the full prompt is processed once and all encoder, reasoning, and decoder caches are populated for the prompt tokens. In the subsequent decode phase, LoopUS no longer recomputes the full prefix. Instead, when generating token xt+1, each module receives only the newest token representation together with its previously accumulated cache, so the new query attends to the stored keys and values of the entire prefix while appending only one additional KV entry per layer and per loop depth. The initial forward pass processes the full prompt, whereas every later decoding step uses only the newest token,

##### xt+1, pt+1 = st, (15)

with the absolute position pt+1 supplied through cache_position. This keeps rotary position IDs and causal masks aligned with the full prefix even though the model no longer recomputes x1:t.

The per-token latent update can be written as

h(0)t+1,Ctenc+1 = E (xt+1;Ctenc,pt+1), (16) h(t+1b) ,Ctrea+1,b = Rb h(t+1b−1);Ctrea,b,pt+1 , b = 1,...,B, (17)

ℓt+1,Ctdec+1 = D h(t+1B);Ctdec,pt+1 , (18)

where ℓt+1 denotes the next-token logits. The key design choice is that LoopUS does not share a single cache across all recursive refinements; instead, each refinement depth b has its own cache

Ctrea,b. This separation is necessary because the hidden state entering loop depth b is not the same object as the hidden state entering depth b′ ̸= b: although the block parameters are shared, the token representations evolve after every refinement step, so the corresponding keys and values represent different latent trajectories. Reusing one common cache across all depths would mix KV states produced at different refinement stages and would no longer correspond to the actual recurrence being executed. Maintaining one cache per depth therefore preserves the semantics of the unrolled loop while still amortizing prefix computation over autoregressive time. This design matches the fact that the same middle block is reused as a depth-indexed dynamical operator rather than as one flat transformer pass, and it keeps the implementation compatible with recursive decoding views similar to Bae et al. [11, 5].

If the active context reaches the maximum window size M, the cached state is truncated before appending the next token,

Ct ← crop(Ct,M − 1), st ← M − 1, (19)

so all encoder, reasoning, and decoder caches remain synchronized with the same sliding window. In this form, KV caching turns LoopUS generation into an incremental state update over Ct rather than repeated recomputation of the entire prefix at every token.

To quantify the practical effect of this design, we benchmark autoregressive decoding with the recursion budget fixed to B = 8 and measure token-generation throughput over five repeated runs on an NVIDIA L40S GPU. Figure 11 summarizes the resulting cached-versus-uncached comparison across output lengths. At 1024 generated tokens, KV caching yields a 1.64× speedup for Qwen3-1.7B, a 2.31× speedup for Qwen3-4B, and a 2.49× speedup for Qwen3-8B. This pattern is consistent with the observation of Bae et al. [5] that looped transformers benefit strongly from cache reuse because recurrent decoding repeatedly reaccesses shared memory across refinement steps. Once the prefix state has been materialized, caching removes much of the repeated computation that would otherwise be spent reconstructing the same context at each autoregressive step.

#### A.5 Halting and Recursion Diagnostics

The halting comparison in Figure 18 is implemented with a fixed Qwen3-1.7B LoopUS checkpoint and compares three inference rules: threshold halting, convergence halting based on hidden-state change, and CDF-based halting. These runs use the task set {MMLU, HellaSwag, ARC-Easy, PIQA, WinoGrande, Lambada, WikiText}, maximum recursion budget B = 10, batch size 16, maximum length 1024, seed 2026, and runtime-stat logging. The convergence sweep uses ϵ ∈ {0.1,1,5,10,100}, while the CDF sweep uses thresholds {0.2,0.3,0.5,0.7,0.8}.

The separate recursion-depth study varies the inference budget on the Qwen3-4B LoopUS checkpoint while keeping the remaining evaluation settings fixed. This diagnostic isolates how far additional loop depth can improve performance before gains saturate or begin to trade off against over-refinement; the corresponding results are reported in Figure 5 in Section 4.4.

### B Dynamical and Geometric Interpretation of LoopUS

Recent work suggests that latent-space iterative reasoning can be understood through a dynamical lens. Gladstone et al. [7] argue that “thinking” can be framed as optimization with respect to a learned verifier that measures the compatibility between an input and a candidate prediction, with additional computation corresponding to iterative refinement until convergence. Complementarily, Blondel et al. [57] establish an explicit bijection between autoregressive models and EBMs in function space. They

show that a sequence-level energy decomposes into per-token rewards,

|y|

r(x ⊕ y<t,yt), (20)

R(x,y) =

t=1

and that an autoregressive model’s next-token logits q relate to these per-token rewards r through the soft Bellman equation: q(st,yt) = r(st,yt) + Vq(st ⊕ yt), where Vq is a soft value function encoding lookahead over all possible continuations. Thus even a model trained purely on next-token prediction implicitly induces a global compatibility function rather than making only isolated tokenlevel decisions. Taken together, these results suggest that an autoregressive transformer can exhibit gradual refinement trajectories similar to energy minimization processes.

This view is consistent with LoopUS because our hidden-state analysis in Figure 1 suggests that the reused middle layers already form a near-fixed-point latent workspace. LoopUS then turns this workspace into an iterative refinement process. Using the recursion-depth index b, after vectorizing the hidden state and writing P(b) = Diag(α(b)), Eq. 5 can be rewritten exactly as

##### h(b+1) = h(b) − P(b) h(b) − M(h(b)) . (21)

This is a diagonally preconditioned relaxed fixed-point iteration. Therefore, the mathematically unconditional statement is that LoopUS searches for a latent state h⋆ satisfying h⋆ = M(h⋆). Any converged point with strictly positive diagonal entries in P(b) must satisfy this fixed-point condition. In other words, the rigorous part of the argument is a stable fixed-point search in latent space; the energy interpretation becomes exact only under an additional assumption on the residual field.

Specifically, suppose there exists a scalar potential Φx(h) on the relevant latent manifold such that

∇hΦx(h) = h − M(h). (22) Then Eq. 21 becomes

h(b+1) = h(b) − P(b)∇hΦx(h(b)), (23) which is simply diagonally preconditioned gradient descent on Φx. If Φx is L-smooth and the effective step size is small enough, e.g., ∥P(b)∥2 ≤ 1/L, the descent lemma yields

L

2 ∥P(b)g(b)∥22, g(b) = ∇hΦx(h(b)), (24) so the potential decreases whenever the gated step is sufficiently small. Under this assumption, fixed points of M coincide with stationary points of Φx. This provides a geometric intuition for why LoopUS can be viewed as mimicking an energy minimization process, provided the residual field h − M(h) locally resembles a gradient field.

Φx(h(b+1)) ≤ Φx(h(b)) − ⟨g(b),P(b)g(b)⟩ +

More directly, our training objective enforces a next-token-prediction surrogate energy descent. Define

Ex(h) = CE(D(h),x2:T), (25) which is exactly the next-token prediction loss for the gold continuation under the decoder. Then the monotonicity loss from Section 3.2 can be written as

L(monob) = SiLU Ex(h(b+1)) − Ex(h(b)) . (26) Because SiLU is positive for positive arguments and produces only small negative values for negative arguments, minimizing L(monob) penalizes positive increments in Ex and mildly rewards negative increments. Therefore, even without proving that h − M(h) is the gradient of a global scalar potential, LoopUS is explicitly trained to make this decoder-induced surrogate energy approximately non-increasing along the loop. The confidence head complements this by stopping once the predicted benefit of further refinement becomes small. In this sense, the halting trends in Figure 18 and the smooth contraction-like trajectories in Figures 16 and 17 are not just qualitative analogies: they are the empirical signature of a loop trained to behave like a descent process on a task-aligned surrogate energy.

The monotonicity loss and stepwise detachment sharpen this picture at the level of individual refinement transitions. The monotonicity term does not only prefer a good final state; it explicitly

trains each local update h(b) → h(b+1) to avoid increasing the decoder surrogate energy Ex. Stepwise detachment then prevents gradients from coupling all loop positions into one long backpropagationthrough-time graph, so each supervised depth is optimized primarily as a local correction applied to the current latent state rather than as one stage in a globally entangled trajectory. Combined with the selective gate, this biases LoopUS toward sequences of small, progressively improving latent edits instead of brittle one-shot remappings. In this limited dynamical sense, LoopUS resembles deterministic iterative refinement procedures such as DDIM [61] and recent language diffusion models [9]: each step is trained to move the representation toward a more task-aligned region while preserving stable multi-step evolution.

The diffusion analogy should not be overstated. LoopUS does not inject explicit noise, does not learn a reverse diffusion process or noise schedule, and does not optimize a score-matching objective. Strictly speaking, LoopUS is also not an explicit EBM. We do not learn a standalone scalar energy over all input-output pairs, nor do we run gradient descent on that scalar in output space. A more precise statement is that LoopUS is an implicit latent optimizer: the refinement process is amortized into the recurrent hidden-state dynamics of a pretrained autoregressive transformer, approximating a steepest-descent-like process through repeated latent updates. The connection to diffusion or EBMs is therefore geometric and dynamical rather than probabilistic. This distinction matters, but the main conceptual point remains: what makes the loop useful is not merely extra depth, but the emergence of a compatibility-seeking dynamical process that keeps allocating compute until the representation settles into a more predictive state.

### C Additional Qualitative Example

- Figure 12 provides a representative example of how LoopUS refines its latent state across loop iterations during generation. Rather than making one large, brittle update, the model follows a coherent multi-step trajectory: early iterations make the largest representational corrections, while later iterations make smaller adjustments that stabilize the prediction. This qualitative behavior is consistent with the view developed in the main text that LoopUS acts as an iterative latent refinement process rather than as a one-shot remapping of the hidden state.

Figures 13, 14, and 15 further visualize these loop trajectories in a shared PCA space for multiple LoopUS backbones. Across models, the hidden states evolve along smooth, low-dimensional paths instead of exhibiting erratic jumps between iterations. The trajectories typically show larger motion in the first few refinement steps followed by progressive contraction, which supports our interpretation of LoopUS as a controlled descent-like process in latent space. Although the exact geometry varies by backbone, the common pattern is that iterative reuse of the reasoning block produces structured movement toward a more stable predictive state.

### D Hidden-State Trajectory Analysis

To test whether the staged representation dynamics used to motivate LoopUS are specific to a single backbone or reflect a broader pattern, we extend the hidden-state analysis to additional pretrained LLMs. Figure 16 plots PCA trajectories of hidden states across layers for six backbones, while Figure 17 reports the corresponding layer-to-layer distance profiles. Together, these figures show that the qualitative structure highlighted in Figure 1 is not unique to Qwen3-1.7B.

Across model families, we repeatedly observe the same three-phase organization: rapid representational motion in early layers, a relatively smooth middle-layer regime, and a sharper transition near the final layers. In the PCA plots, this appears as a gradual arc or plateau through the middle of the network followed by a more pronounced turn toward the output-facing layers. In the distance profiles, the same effect appears as reduced consecutive-layer change through the middle block bracketed by larger changes at the beginning and end of the network. This cross-model consistency supports our architectural prior that pretrained decoder-only transformers naturally admit an encoder–reasoning– decoder decomposition, with the middle layers forming the most suitable region for stable iterative reuse.

Prompt: A farmer has 17 sheep and all but 9 run away. Explain briefly how many sheep remain and why. Let's think step by step.

Response: First , the farmer has 1 7 sheep . The problem says "

all but 9 run away ." The phrase " all but " means " all

except ." So , if all but 9 run away , that means 9

sheep remain . The farmer 's sheep count is .Go to the

answer .Go 1 7 -

Prompt: A person stops watering a houseplant for several weeks. Explain step by step why the plant may wilt.

Response: Include the following terms in your explanation : x yle m , ph

lo em , and trans piration . ( 1 0 points ) When a person

stops watering a house plant for several weeks , the plant may

wilt due to a series of physiological and environmental changes .

Here is a step -by -step explanation of why this happens ,

including

Prompt: A doctor tells a patient to finish the full course of antibiotics even after symptoms improve. Explain the reasoning behind that advice. Let's think step by step.

Response: First , why do doctors prescribe antibiotics ? Antib iotics are

designed to kill bacteria . But not all bacteria are killed by

the first dose . Some may become resistant to the antibiotic .

If the patient stops the medication too early , the remaining

bacteria can become resistant . This means the antibiotic won 't

work if the infection comes back .

Prompt: A student can either study one hour every day for a week or study seven hours the night before the exam. Explain step by step which approach is likely to work better and why.

Response: ( 1 0 points ) I would say that studying one hour every day for a week is better than studying seven hours the night before

the exam . I would say that is because studying for a week

is better than studying for one night . I would say that is

because studying for a week is better than studying for one

night

step 1 step 2 step 3 step 4

Figure 12: Example generation thinking trace from LoopUS on Qwen3-4B. The figure visualizes how the model’s intermediate reasoning trajectory evolves across loop iterations for a representative sample.

###### 32 * 64 =

EuclideanDistance

h0

1.00

0.6

CosineDistance

PC-2(6.4%var)

h234567891119202345678

cosine

mean ± std

0

- 101

- 102

- 103

0.75

euclidean

mean

0.4

Alpha

0.50

500

0.2

0.25

1000

h1

0.0

0.00

4000 3000 2000 1000 0

1 5 9 13 17 20

1 5 9 13 17 20

PC-1 (92.7% var)

Thinking Iteration

Thinking Iteration

###### 82 *64 + 21 =

EuclideanDistance

h0

1.00

CosineDistance

PC-2(6.7%var)

0.6

cosine

mean ± std

h234567891119202345678

- 101

- 102

- 103

0

0.75

euclidean

mean

0.4

Alpha

0.50

500

0.2

0.25

h1

1000

0.0

0.00

4000 3000 2000 1000 0

1 5 9 13 17 20

1 5 9 13 17 20

PC-1 (92.7% var)

Thinking Iteration

Thinking Iteration

###### 404 Not Found in the system 404 The new era

EuclideanDistance

h0

1.00

0.6

CosineDistance

PC-2(6.4%var)

cosine

mean ± std

h234567891119202345678

0

- 101

- 102

- 103

0.75

euclidean

mean

0.4

Alpha

0.50

500

0.2

0.25

1000

h1

0.0

0.00

4000 3000 2000 1000 0

1 5 9 13 17 20

1 5 9 13 17 20

PC-1 (92.7% var)

Thinking Iteration

Thinking Iteration

###### What is the capital of South Korea? S

- PC-2(8.1%var)

h0

h1

- h2
- h3
- h4
- 5
- 6
- 7
- 8
- 9 0

h1119202345678

1 5 9 13 17 20

Thinking Iteration

0.00

0.25

0.50

0.75

1.00

Alpha

mean ± std

mean

1 5 9 13 17 20

Thinking Iteration

0.0

0.2

0.4

0.6

CosineDistance

cosine

euclidean

- 102

- 103

EuclideanDistance

Figure 13: LoopUS thinking PCA visualization for Qwen3-1.7B.

32 * 64 =

1000 800 600 400 200 0

PC-1 (96.5% var)

100

50

0

50

PC-2(3.0%var)

h0

- h1
- h2
- h3

h45

- 6
- 7
- 8

h911192002345678

1 5 9 13 17 20

Thinking Iteration

0.00

0.25

0.50

0.75

1.00

Alpha

mean ± std

mean

1 5 9 13 17 20

Thinking Iteration

0.0

0.2

CosineDistance

cosine

euclidean

- 101

- 102

- 103

EuclideanDistance

82 *64 + 21 =

800 600 400 200 0

PC-1 (98.8% var)

25

0

25

PC-2(1.1%var)

h0

- h1
- h2
- h3
- 4
- 5
- 6
- 7
- 8
- 9 0

11 2

h1920345678

1 5 9 13 17 20

Thinking Iteration

0.00

0.25

0.50

0.75

1.00

Alpha

mean ± std

mean

1 5 9 13 17 20

Thinking Iteration

0.0

0.2

0.4

CosineDistance

cosine

euclidean

- 101

- 102

- 103

EuclideanDistance

404 Not Found in the system 404 The new era

1000 750 500 250 0

PC-1 (97.8% var)

0

50

PC-2(1.8%var)

h0

- h1
- h2
- h3
- 4
- 5
- 6
- 7

h8911192002345678

1 5 9 13 17 20

Thinking Iteration

0.00

0.25

0.50

0.75

1.00

Alpha

mean ± std

mean

1 5 9 13 17 20

Thinking Iteration

0.0

0.1

0.2

0.3

CosineDistance

cosine

euclidean

- 101

- 102

- 103

EuclideanDistance

What is the capital of South Korea? S

1000 750 500 250 0

PC-1 (88.9% var)

100

0

100

200

- PC-2(9.6%var)

500

0

4000 3000 2000 1000 0

PC-1 (89.5% var)

EuclideanDistance

- h1
- h2
- h3
- h4
- 5
- 6
- 7
- 8
- 9

1.00

CosineDistance

- 102

- 103

cosine

mean ± std

0.75

euclidean

mean

0.4

Alpha

0.50

0.2

h11192002345678

h0

0.25

0.0

0.00

1 5 9 13 17 20

1 5 9 13 17 20

Thinking Iteration

Thinking Iteration

###### Figure 14: LoopUS thinking PCA visualization for Qwen3-8B.

### E Halting Strategies

We compare three inference-time halting rules while keeping the LoopUS checkpoint fixed. Let h¯(t+1b) ∈ Rh denote the hidden state of the newest token after the b-th latent refinement step at autoregressive decoding step t + 1, and let

q˜t(+1b) = qϕ h ¯(t+1b) , qt(+1b) = σ q ˜t(+1b) . (27) Given a maximum recursion budget B, each strategy returns an exit depth

bexit(t + 1) = min b ∈ {1,...,B} : Stop(t+1b) = 1 , (28)

with the convention that bexit(t+1) = B if no earlier stopping condition is satisfied. In our implementation, the confidence head is evaluated at every refinement step by default (q_eval_interval = 1), although the code also supports evaluating it every r steps and always at the final step. During batched evaluation, halting is applied conservatively so that all examples in the active batch remain synchronized: threshold and CDF halting use the minimum stopping score across the batch, whereas convergence halting uses the maximum hidden-state change across the batch.

Threshold-based halting. This is the main stopping rule used by LoopUS and corresponds directly to the confidence-based stopping rule introduced in Section 3.1. We stop as soon as the confidence head exceeds a fixed threshold,

qt(+1b) ≥ qth, (29) or equivalently,

bthexit(t + 1) = min b : qt(+1b) ≥ qth . (30)

Because LQ trains qϕ to approximate the post-update token accuracy, this rule interprets qt(+1b) as a direct estimate that the current latent state is already sufficiently predictive and that further refinement

is unlikely to be necessary. For a batch B, the implementation stops only when

qn,t(b)+1 ≥ qth, (31)

min

n∈B

which ensures that every sequence in the batch is ready to halt before the shared forward pass terminates.

Convergence-based halting. The second strategy is a non-parametric baseline that monitors whether the latent trajectory has numerically converged. Let

∆(t+1b) = h ¯(t+1b) − h¯(t+1b−1)

, b ≥ 1, (32)

2

where h¯(t+1b−1) and h¯(t+1b) are the last-token hidden states before and after the current refinement step. The model exits when the representation change becomes smaller than a preset tolerance ϵ,

∆(t+1b) ≤ ϵ, (33) that is,

bconvexit (t + 1) = min b : ∆(t+1b) ≤ ϵ . (34)

This rule does not use the confidence head at all. Instead, it assumes that once the hidden state of the current token changes only marginally from one loop to the next, the decoder distribution is unlikely to improve enough to justify another refinement step. In batched evaluation, we stop only when

∆(n,tb)+1 ≤ ϵ, (35) so that the entire batch has converged under the same geometric criterion.

max

n∈B

CDF-based halting. For comparison, we also implement a cumulative-distribution-based halting rule inspired by the Q-exit criterion used in Ouro [28]. In this strategy, the stepwise confidence is reinterpreted as a hazard rate,

λ(t+1b) = qt(+1b) ∈ (0,1). (36) The probability of exiting exactly at step b is then defined by

b−1

πt(+1b) = λ(t+1b)

j=1

1 − λ(t+1j) , (37)

and the cumulative probability of having exited by step b is

CDF(t+1b) =

The corresponding stopping rule is

b

b

πt(+1i) = 1 −

i=1

j=1

1 − λ(t+1j) . (38)

bcdfexit(t + 1) = min b : CDF(t+1b) ≥ qth . (39) Intuitively, this rule accumulates evidence for stopping over multiple refinement steps: even if no single step produces a confidence score above qth, several moderately large values of λ(t+1b) can combine into a large cumulative exit probability. For numerical stability, the implementation accumulates the survival probability in log space,

log St(+1b) =

b

log 1 − λ(t+1j) , CDF(t+1b) = 1 − exp log St(+1b) . (40)

j=1

Since LoopUS trains its confidence head with a direct binary target on post-update token accuracy rather than with Ouro’s full exit-distribution objective, we use this CDF rule as a comparative inference heuristic rather than as the primary training-consistent stopping criterion. In batched evaluation, stopping occurs only when

CDF(n,tb)+1 ≥ qth. (41)

min

n∈B

Discussion. These three strategies expose different inductive biases. Threshold halting is the rule most tightly matched to our training objective because qϕ is explicitly supervised to predict whether the current refinement is already good enough. Convergence halting ignores the confidence head and instead uses only latent geometry, making it a useful diagnostic for whether the loop behaves like a contractive refinement process. The CDF-based rule is more aggressive in aggregating multiple moderate confidence values across steps and therefore serves as a natural comparison point to prior looped-language-model early-exit mechanisms [28]. Figure 18 compares these strategies under matched checkpoints while varying qth or ϵ.

### F Limitations and Future Work

Extension beyond text-only language modeling. All experiments in this paper focus on text-only, decoder-only language models. Although LoopUS is formulated as a latent-space refinement method and is therefore not tied to a particular input modality in principle, we have not evaluated whether the same encoder–reasoning–decoder decomposition remains stable for multimodal models. Recent progress on recurrent and looped architectures suggests that iterative latent computation can also be useful beyond text-only settings, but it remains unclear whether a pretrained multimodal transformer exhibits the same middle-layer geometry that makes LoopUS effective in our experiments [62, 15]. Vision, audio, and cross-modal tokens may induce different hidden-state structure, different layer specialization, and different failure modes under repeated latent refinement. Extending LoopUS to vision-language and other multimodal transformers is therefore an important next step, especially for testing whether looped latent computation can improve grounding, compositional perception, and cross-modal reasoning rather than only text-domain prediction.

Heterogeneous and hybrid model architectures. Our experiments instantiate LoopUS on relatively standard transformer backbones. Modern language models increasingly combine heterogeneous layers and operators, such as gated delta networks [63], sparse attention [64], mixture-of-experts routing [65, 66], state-space modules [67], and other hybrid sequence mechanisms [68]. In such models, the middle-layer region may no longer behave as a uniform reusable block, and the optimal recursion policy may depend on the operator type, layer role, or token state. Future work should study how to select, compose, or schedule looped modules in architectures whose layers have substantially different computational semantics. A particularly useful direction is to learn architecture-aware recursion policies that decide not only when to stop, but also which operator or layer group should be reused at each refinement step.

Scaling to larger and more diverse training regimes. The present results are obtained under a moderate post-training budget rather than a full large-scale training pipeline. This setting is useful for isolating the effect of looped up-scaling, but it leaves open how LoopUS behaves under substantially larger corpora, longer contexts, curriculum schedules, or more diverse data mixtures. Scaling the training recipe is especially relevant because the confidence head and monotonicity objective may benefit from richer distributions of refinement trajectories than those observed in limited-token adaptation. Larger-scale studies could also clarify whether the observed gains remain concentrated on reasoning-oriented tasks or broaden to knowledge-heavy and long-context settings as the loop is exposed to more varied supervision.

Dedicated math and long-context reasoning coverage. Although our benchmark suite includes several reasoning-oriented evaluations, it does not include dedicated mathematical reasoning tasks or training on math-focused corpora. This is an important limitation of the current study. Because of compute constraints, our released training setup remained at a context length of 1024 and did not incorporate additional math-oriented datasets during adaptation. As a result, the paper does not yet test whether LoopUS yields similar gains on problems that require longer derivations, sustained multi-step reasoning, or explicit mathematical supervision. A useful next step is therefore to scale the training budget and context window while adding dedicated math and reasoning datasets, so that the benefits and failure modes of latent looping can be assessed under more computation-intensive reasoning workloads.

Integration with instruction tuning and preference optimization. We evaluate LoopUS as a post-training adaptation method for base models. In practical LLM development, however, model quality is shaped not only by continued pretraining but also by instruction tuning, long-context adaptation, and preference optimization methods such as reinforcement learning from feedback or GRPO-style objectives [69]. Integrating LoopUS with SFT and preference-based post-training could test whether looped latent computation improves not only benchmark accuracy but also instruction following, controllability, calibration, and user-facing reliability [4]. This direction is also important because adaptive test-time computation may interact with alignment objectives: a model should learn when extra refinement is helpful for faithfully following an instruction and when additional computation risks overthinking or drifting from the user’s intent.

LoopUS applied to diffusion LLMs. Although our current implementation focuses on standard autoregressive transformers, the core mechanism of LoopUS—iterative latent refinement controlled by dynamic gating and early halting—naturally aligns with the continuous denoising trajectories of diffusion language models [9, 70]. Diffusion LLMs fundamentally rely on iterative processes to refine continuous latent representations before decoding. However, their denoising steps are typically fixed or bound to computationally uniform schedules. Integrating the LoopUS framework into diffusion LLMs could introduce data-dependent adaptive computation to the reverse diffusion process, allowing the model to halt denoising dynamically when the latent state is sufficiently clean, guided by our confidence head. Furthermore, the selective gating mechanism and monotonicity loss could serve as structural regularizers for the score network, ensuring stable, monotonic improvement across timesteps and potentially accelerating generation without requiring explicit trajectory distillation. Exploring this intersection remains a promising direction for making continuous-space language generation more practical.

LoopUS as a pretraining-native architecture. Finally, LoopUS may also be useful earlier in the model lifecycle. This paper treats looped depth up-scaling as a minimally invasive retrofit applied

to pretrained backbones, but the same architectural prior could be incorporated during pretraining itself. Pretraining with looped latent refinement from the beginning may allow the model to allocate representational capacity differently, learn more calibrated stopping behavior, and make better use of adaptive test-time computation. A pretraining-native LoopUS model could also expose the recurrence to a much broader distribution of intermediate states, potentially reducing the gap between trainingtime refinement and inference-time recursion.

###### 32 * 64 =

EuclideanDistance

h0

1.00

100

PC-2(15.5%var)

CosineDistance

cosine

mean ± std

h23456h7891119203426578

0.75

0.4

euclidean

mean

- 100

- 101

- 102

0

Alpha

0.50

100

0.2

0.25

200

h1

0.0

0.00

400 200 0

1 5 9 13 17 20

1 5 9 13 17 20

PC-1 (82.2% var)

Thinking Iteration

Thinking Iteration

###### 82 *64 + 21 =

EuclideanDistance

h0

1.00

PC-2(14.1%var)

CosineDistance

cosine

mean ± std

2h34567981119202345678

0.75

0

euclidean

mean

- 100

- 101

- 102

0.4

Alpha

0.50

100

0.2

0.25

h1

200

0.0

0.00

500 400 300 200 100 0

1 5 9 13 17 20

1 5 9 13 17 20

PC-1 (83.3% var)

Thinking Iteration

Thinking Iteration

###### 404 Not Found in the system 404 The new era

- PC-2(14.0%var)

h0

h1

23h45678911192002356478

1 5 9 13 17 20

Thinking Iteration

0.00

0.25

0.50

0.75

1.00

Alpha

mean ± std

mean

1 5 9 13 17 20

Thinking Iteration

0.0

0.2

0.4

CosineDistance

cosine

euclidean

- 101

- 102

EuclideanDistance

What is the capital of South Korea? S

600 400 200 0

PC-1 (82.0% var)

100

0

100

200

- PC-2(14.1%var)

0

100

200

400 200 0

PC-1 (84.0% var)

EuclideanDistance

h1

1.00

CosineDistance

0.6

cosine

mean ± std

0.75

euclidean

mean

- 101

- 102

0.4

Alpha

0.50

- h2
- h3 h4567891120192453678

0.2

0.25

h0

0.0

0.00

1 5 9 13 17 20

1 5 9 13 17 20

Thinking Iteration

Thinking Iteration

###### Figure 15: LoopUS thinking PCA visualization for Qwen3-4B.

Baby you make me

Yeah you re just my type and I like your

100

| | | | | | |
|---|---|---|---|---|---|
| |H5<br><br>| | | | |
| |HH1H9H1317 H<br><br>| | |H<br><br>| |
| |21<br><br>|H25<br><br>H29<br><br>| |33| |
| | | | | | |
| | | | | | |

100

- PC-2(10.4%var)

PC-2(6.0%var)

H5

50

50

H1

H91317 H21

0

0

H25

H33

H29

50

50

100 0 100 200 300

0 200 400 PC-1 (85.9% var)

PC-1 (76.7% var)

What you what you lookin' at

404 Not Found in the system 404 The new era

H29

50

H33

H25

100

PC-2(5.9%var)

PC-2(5.0%var)

H21

H17

0

H15913

50

HH915H1317 H21

50

0

H33

H25

100

50

H29

0 200 400 PC-1 (87.5% var)

0 200 400 PC-1 (89.4% var)

(a) Qwen3-4B

- PC-2(9.4%var)

HH1H5H913H21172529H3337H414549

H53

H57

H61

Baby you make me

0 100 200 300 PC-1 (75.4% var)

100

0

100

PC-2(8.1%var)

HHH15913

HH211725H29H333741H4549

H53 H57

H61

Yeah you re just my type and I like your

0 100 200 300 PC-1 (69.0% var)

50

0

50

100

- PC-2(10.5%var)

100

0

100

0 100 200 300 PC-1 (72.8% var)

What you what you lookin' at

404 Not Found in the system 404 The new era

150

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| |H1H5H9HH211325297 HH<br><br>|33 374145| |H61<br><br>| |
| | |H49<br><br>H53<br><br>|H57<br><br>| | |
| | | | | | |

PC-2(7.6%var)

100

H61

H15H9173

50

HH21H252933 H37H4145 H49

0

H57

H53

50

0 100 200 300 PC-1 (78.2% var)

(c) Qwen3.5-27B

Baby you make me

Yeah you re just my type and I like your

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | |HH1 5 HH91317 H21<br><br>| | | | | |
| | | |H25<br><br>H<br><br>|29<br><br>|H33<br><br>| | |
| | | | | | | | |

200

200

PC-2(6.0%var)

PC-2(4.1%var)

100

100

H19135 HH1721

0

0

H25

H33

H29

100

100

200 0 200 400 600 800

0 500 1000 PC-1 (91.6% var)

PC-1 (88.1% var)

What you what you lookin' at

404 Not Found in the system 404 The new era

H29 H33

100

H25

PC-2(5.8%var)

PC-2(3.5%var)

200

H21

0

H159173

HH159H131721

100

0

H25

H33

H29

200

0 500 1000 PC-1 (88.7% var)

0 500 1000 PC-1 (92.7% var)

(b) Qwen3-8B

Baby you make me

Yeah you re just my type and I like your

H195H17253H12933H37

H195H17253H12933H37

0

0

PC-2(11.1%var)

PC-2(9.9%var)

10

10

20

20

30

30

0 20 40 60 PC-1 (71.2% var)

0 20 40 60 PC-1 (75.0% var)

What you what you lookin' at

404 Not Found in the system 404 The new era

1H951721H325H2933 H37

0

PC-2(10.6%var)

PC-2(9.9%var)

10

10

5

20

15

9 H17213

H37

30

0

HH25H2933

0 10 20 30 40 PC-1 (68.2% var)

0 10 20 30 40 PC-1 (75.9% var)

(d) Qwen3.5-35B-A3B

150

- PC-2(8.6%var)

HH211713951 H25

H29 H33

H37

Baby you make me

400 300 200 100 0 100 PC-1 (85.1% var)

0

100

PC-2(7.7%var)

HH21139751 H25

H33 H29

H37

Yeah you re just my type and I like your

400 200 0 PC-1 (83.9% var)

0

100

200

- PC-2(9.0%var)

100

50

0

50

400 300 200 100 0 100 PC-1 (81.5% var)

What you what you lookin' at

404 Not Found in the system 404 The new era

PC-2(7.5%var)

100

H171395 H21

H171395 H21

0

H37

H25 H29

H25 H29

H37

H33

H33

600 400 200 0 PC-1 (87.8% var)

(e) Phi-4

Baby you make me

Yeah you re just my type and I like your

40

H57 H61

5 PC-2(22.3%var)

H53

PC-2(13.6%var)

H49

H45

20

H61

H333741

0

295

179513

HH591725H312933 H37

H57

0

H53

H41 H45 H49

5

0 20 40 60 80 PC-1 (56.0% var)

0 20 40 PC-1 (67.1% var)

What you what you lookin' at

404 Not Found in the system 404 The new era

60

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | |H61<br><br>| | | | |
| | | |H57| |H<br><br>|HH29H 33<br><br>|
| | | |H<br><br>H53<br><br>|49 H45<br><br>|H37 H41<br><br>| |
| | | | | | | |

40

PC-2(13.4%var)

PC-2(16.0%var)

40

20

H61

20

25211793

H591725H293133H37

0

0

H57

H41

H45 H49H53

20

60 40 20 0 20 PC-1 (67.4% var)

0 20 40 PC-1 (61.0% var)

(f) EXAONE-4.0-32B

Figure 16: PCA trajectories of hidden-state refinement across model backbones.

###### Raw Cosine Distance 1 cos(ht, ht+1)

CosineDistance

1.0

Baby you make me

Yeah you re just my type and I like your

What you what you lookin' at

0.5

404 Not Found in the system 404 The new

0.0

0 5 10 15 20 25 30 35

Layer Transition (t t+1)

Euclidean Distance ht+1 ht 2

EuclideanDistance

- 101

- 102

0 5 10 15 20 25 30 35

Layer Transition (t t+1)

(a) Qwen3-4B

###### Raw Cosine Distance 1 cos(ht, ht+1)

1.0

CosineDistance

Baby you make me

Yeah you re just my type and I like your

What you what you lookin' at

0.5

404 Not Found in the system 404 The new

0.0

0 10 20 30 40 50 60

Layer Transition (t t+1)

Euclidean Distance ht+1 ht 2

EuclideanDistance

- 101

- 102

0 10 20 30 40 50 60

Layer Transition (t t+1)

(c) Qwen3.5-27B

###### Raw Cosine Distance 1 cos(ht, ht+1)

CosineDistance

Baby you make me

0.75

Yeah you re just my type and I like your

0.50

What you what you lookin' at

404 Not Found in the system 404 The new

0.25

0.00

0 5 10 15 20 25 30 35

Layer Transition (t t+1)

Euclidean Distance ht+1 ht 2

EuclideanDistance

- 101

- 102

- 103

0 5 10 15 20 25 30 35

Layer Transition (t t+1)

(b) Qwen3-8B

###### Raw Cosine Distance 1 cos(ht, ht+1)

CosineDistance

Baby you make me

0.4

Yeah you re just my type and I like your

What you what you lookin' at

404 Not Found in the system 404 The new

0.2

0.0

0 5 10 15 20 25 30 35 40

Layer Transition (t t+1)

Euclidean Distance ht+1 ht 2

EuclideanDistance

- 100

- 101

- 102

0 5 10 15 20 25 30 35 40

Layer Transition (t t+1)

(d) Qwen3.5-35B-A3B

###### Raw Cosine Distance 1 cos(ht, ht+1)

CosineDistance

0.75

Baby you make me

Yeah you re just my type and I like your

0.50

What you what you lookin' at

404 Not Found in the system 404 The new

0.25

0.00

0 5 10 15 20 25 30 35 40

Layer Transition (t t+1)

Euclidean Distance ht+1 ht 2

EuclideanDistance

- 101
- 102
- 103

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0 5 10 15 20 25 30 35 40

Layer Transition (t t+1)

(e) Phi-4

###### Raw Cosine Distance 1 cos(ht, ht+1)

CosineDistance

0.4

Baby you make me

Yeah you re just my type and I like your

What you what you lookin' at

0.2

404 Not Found in the system 404 The new

0.0

0 10 20 30 40 50 60

Layer Transition (t t+1)

Euclidean Distance ht+1 ht 2

EuclideanDistance

- 100

- 101

- 102

10 1

0 10 20 30 40 50 60

Layer Transition (t t+1)

(f) EXAONE-4.0-32B

Figure 17: Hidden-state distance profiles across model backbones.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Figure 18: Halting behavior under different stopping strategies. The learned threshold-based rule yields the sharpest halting distribution, indicating that LoopUS learns a meaningful stopping signal.

