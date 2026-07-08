# arXiv:2606.20097v1[cs.CL]18Jun2026

[Figure 1]

2026-06-19

## HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization

Zhentao Tan, Wei Chen, Jingyi Shen, Yao Liu, Xu Shen, Yue Wu, Jieping Ye Alibaba Group

### Abstract

The quadratic complexity of attention poses a critical bottleneck for long-context processing, spurring interest in hybrid attention designs. Most open-source hybrid models adopt a layerwise strategy. Yet, prior work has noted the inherent diﬀiculty of integrating Linear Attention (LA) with Full Attention (FA), suggesting that the design space of attention hybridization remains underexplored. To probe this space, we conduct interpretability analysis and observe that layers exhibit block-wise functional similarity, while individual heads within the same layer display distinct functional specialization despite sharing input features. This head-level heterogeneity suggests that the head dimension provides a natural and principled granularity for fusing heterogeneous attention signals. Building on this insight, we introduce HydraHead, a novel architecture that hybridizes FA and LA along the head axis. HydraHead features two key innovations: (1) an interpretability-driven selection strategy that identifies retrieval-critical heads and preserves FA only for them, and (2) a scale-normalized fusion module that reconciles the distributional gap between FA and LA head outputs. By leveraging a three-stage transfer pipeline with parameter reuse and distillation, we achieve high-performance hybrid models with minimal training overhead. Under a unified training setup, HydraHead outperforms other hybrid designs in long-context tasks while maintaining strong general reasoning. With interpretability-driven head selection, it matches a 3:1 layer-wise hybrid’s long-context performance at a 7:1 LA-to-FA ratio. Crucially, trained on only 15B tokens, HydraHead achieves over 69% improvement over the baseline at 512K context length, approaching Qwen3.5, a leading model of comparable size with a native context length of 256K. This highlights the significant scaling potential of head-level hybridization.

Qwen3-1.7B

Qwen3-1.7B-yarn

HydraHead

Qwen3.5-2B-Base

| |
|---|

| |
|---|

| |
|---|

###### (a) RULER Single NIAH

###### (b) RULER Multi-Key NIAH (MK1)

+16.0

+54.3

+21.2

100

+86.6

+58.4

80

+69.2

Recall(%)

60

40

20

0 0 0 0

0 0 0 0

0

16K 32K 64K 128K 256K 512K

16K 32K 64K 128K 256K 512K

Context Length

Context Length

Figure 1: HydraHead, trained on only 15B tokens, significantly outperforms the Qwen3-1.7B baseline on NIAH tasks, approaching Qwen3.5-2B-Base, a leading model of comparable scale with native 256K support.

### 1 Introduction

The landscape of Large Language Models (LLMs) has witnessed a pivotal transition from static questionanswering systems to autonomous agents capable of complex reasoning and long-horizon planning [1, 2, 3, 4]. This evolution places unprecedented demands on context window extension, yet the quadratic complexity of standard Full Attention (FA) remains a rigid computational barrier. While Linear Attention (LA) paradigms —often rooted in State Space Models (SSMs) [5, 6, 7] or kernel-based approximations [8, 9, 10, 11]—offer linear-time complexity, they frequently suffer from “expressivity collapse,” struggling to maintain the highprecision retrieval required for sophisticated reasoning tasks.

Driven by this tension, many recent LLMs adopt hybrid architectures—predominantly layer-wise designs that interleave different attention mechanisms across layers—integrating linear with softmax attention [3, 10, 12], or blending full with sliding-window attention [13]. These efforts reflect a shared direction: scaling longcontext capabilities is no longer a matter of selecting a single attention mechanism, but an exercise in architectural fusion. Despite these successes, some prior work has noted that training hybrid models that combine linear attention with full attention remains challenging, with efforts ultimately focusing on fullor sparse-attention variants [1, 14, 4], suggesting that the design space of attention hybridization remains underexplored.

###### Per-Head Logit for Correct Answer Token

###### Layer-output cosine similarity (28x28) baseline

0123456789101112131415

0123456789101112131415161718192021222324252627

[Figure 2]

60

1.00

[Figure 3]

0.75

logit(positive=promotescorrectanswer)

40

0.50

20

0.25

HeadIndex

Layer

0

0.00

- -1.00

- -0.75

- -0.50

- -0.25

- -60

- -40

- -20

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27

Layer

Layer Index

(a) Per-head logit for correct answer token.

(b) Layer-output cosine similarity.

- Figure 2: The head is a natural granularity for hybridization. (a) Individual heads within the same layer are functionally heterogeneous, with per-head contributions to the final output varying substantially. (b) Layer-level outputs vary smoothly across depth, making it diﬀicult to identify clear boundaries for assigning distinct attention mechanisms to different layers.

To navigate this underexplored design space, it is essential to look beyond surface-level performance and examine how computation is internally organized. Mechanistic interpretability provides the necessary lens, having matured from small models to frontier-scale systems. Specifically, it now spans circuit-level analyses in small transformers [15, 16, 17], scalable feature extraction in advanced models from Anthropic [18], OpenAI [19], and Google DeepMind [20], as well as comprehensive anatomical studies inside frontier LLMs [21, 22, 23]. Importantly, a recurring finding across these studies is that attention heads exhibit substantial diversity in both functional role and importance [24, 25, 26], and this understanding has been extended to diagnose and modify targeted model behaviors [27, 28], providing an analytical foundation for grounding architectural design in the internal structure of pretrained models.

We examine two candidate granularities for attention hybridization: the attention head, one of the most commonly studied functional units in interpretability research, and the layer, which mainstream hybrids predominantly adopt. At the head level, the analysis reveals sharp functional heterogeneity. Figure 2a visualizes per-head logit contributions [15] to the correct answer token on a long-context retrieval task: within a given layer, only a sparse subset of heads contributes substantially while many heads in the same layer remain nearly inactive. These retrieval-critical heads concentrate in deeper layers, consistent with the head-level functional diversity noted above and with prior findings that deeper layers carry more taskspecific signals [16, 29]. These observations are consistent with the head-level diversity noted above. At the layer level, by contrast, the output similarity matrix exhibits a smooth block-wise structure (Figure 2b),

where similarity decays gradually across adjacent layers—a pattern consistent with earlier findings [30, 31]. This smooth variation does not preclude layer-wise hybridization, but offers limited discriminative signal for deciding where to place different attention mechanisms. Together, these observations suggest a natural design principle: the head, rather than the layer, provides a functionally grounded and suﬀiciently fine-grained unit for attention hybridization, enabling different heads within the same layer to process identical input through distinct mechanisms in a shared representation space.

Building on these insights, we propose HydraHead, a fine-grained head-level attention hybridization architecture that addresses two core challenges: (1) determining which heads should adopt FA versus LA, and (2) effectively fusing the heterogeneous attention patterns produced by different mechanisms. For the first challenge, we leverage interpretability techniques to identify heads critical for precise token-level retrieval directly from a pretrained model, retaining FA exclusively for these heads while assigning LA to the remainder for eﬀicient long-context scaling. This head selection procedure is lightweight and one-shot, requiring only a few forward passes over a small calibration set. For the second challenge, we mix FA and LA at the head dimension within each layer, assigning each attention head an independently configurable attention mechanism. Owing to the paradigm gap between FA and LA in how attention features are formed [32, 33], we introduce a head-wise scale-normalized fusion to mitigate interference when these heterogeneous attention signals are mixed. To train the resulting hybrid model, we introduce a three-stage transfer pipeline: parameter inheritance from the pretrained checkpoint and layer-wise alignment; global distillation; and long-context fine-tuning.

We evaluate HydraHead against several representative hybridization strategies, all built upon Qwen3-

- 1.7B [34] under a unified training configuration, using both long-context retrieval benchmarks at varying sequence lengths, and general reasoning benchmarks spanning multiple diﬀiculty levels. These baselines span three dominant paradigms: layer-wise hybrids, token-wise hybrids, and head-wise hybrids. Our experiments yield the following findings.

- • HydraHead achieves state-of-the-art long-context performance while maintaining advanced general reasoning capabilities. Existing hybridization strategies reveal a consistent trade-off across granularity levels: compared to layer-wise hybrids, both head-wise and token-wise approaches excel at general reasoning but struggle with length extrapolation. As a fine-grained headwise approach, HydraHead retains a robust >10% gain over layer-wise baselines on hard reasoning tasks, while achieving state-of-the-art long-context performance.
- • Interpretability-based head selection enables aggressive FA compression with minimal performance degradation. By effectively identifying retrieval-critical heads, our method matches the overall performance of a 3:1 layer-wise hybrid even at substantially higher LA-to-FA mixing ratios (e.g., 7:1).

Beyond controlled comparisons, we further challenge HydraHead by directly benchmarking against state-ofthe-art models. As shown in Figure 1, through meticulous optimization of training hyperparameters and data composition, our model achieves substantial long-context capabilities with only 15B training tokens. Specifically, it delivers over 69% improvement on NIAH benchmarks at 512k context compared to the pretrained baseline, approaching the performance of Qwen3.5 [3], a leading model of comparable scale. Crucially, these gains are achieved without compromising general reasoning performance, further validating the significant potential of our head-wise hybrid method in eﬀicient long-context modeling.

2 Related Work

- 2.1 Mechanistic Interpretability and Head-Level Analysis

Mechanistic interpretability aims to reverse-engineer the internal computations of neural networks through causal and structural analysis. Foundational work establishes a mathematical framework for reasoning about transformer circuits [15] and standardizes activation patching as the primary tool for measuring componentlevel causal effects [35]. These techniques have enabled the identification of concrete circuits underlying specific behaviors, such as indirect object identification [16] and factual knowledge storage [36].

A recurring finding across this line of work is that attention heads serve as meaningful functional units. Voita et al. [24] identify three specialized head types—positional, syntactic, and rare-word heads—and show that the majority of remaining heads can be pruned with negligible quality loss. Michel et al. [25] confirm this finding broadly, demonstrating that in many tasks, only a small fraction of heads accounts for most of the

model’s capability. Clark et al. [26] further show that individual heads learn to attend to specific syntactic relations, revealing consistent functional specialization. At a deeper level, Wang et al. [16] decompose the IOI circuit in GPT-2 small into 26 heads organized into seven functional classes—the canonical worked example of head-level circuit discovery—establishing that head-level granularity is suﬀicient to capture complex multistep computations.

Collectively, these findings establish attention heads as meaningful and well-studied functional units, and the associated causal tools—particularly activation patching and path patching—provide a principled methodology for quantifying their individual contributions. This body of work directly motivates and supports our investigation: we leverage these tools to analyze the functional structure of a pretrained model and use the resulting insights to guide the design of our hybrid attention architecture.

##### 2.2 Linear Attention

Linear attention addresses the quadratic complexity of softmax attention by replacing the explicit tokento-token similarity computation with a recurrent state update, reducing complexity from O(T2) to O(T). One prominent line of work frames this through the lens of state space models, progressively introducing input-dependent selectivity to overcome the rigidity of fixed dynamics [5, 6], and more recently extending to complex-valued state spaces with higher-order discretization for richer memory representation [7]. A complementary perspective views linear attention through the lens of fast-weight memory [37], where the key-value state is treated as an associative memory updated via the delta rule for precise credit assignment [8]. Subsequent advances integrate gating mechanisms to control memory retention [9], enhance the expressiveness of the gating function itself [10], and most recently decouple the update and forget operations for finer-grained memory control [11]. Despite these advances, pure linear attention remains constrained by its fixed-dimensional recurrent state, which limits its capacity for long-range retrieval and exact token-level replication—motivating the hybrid designs that combine linear attention with full attention in a complementary fashion.

##### 2.3 Hybrid Transformers

The pursuit of balancing computational eﬀiciency and representational expressivity has catalyzed a diverse range of hybrid Transformer architectures. Existing efforts can be categorized into three primary paradigms based on the granularity of their hybridization.

Layer-wise Hybridization. This paradigm alternates between different attention kernels across Transformer layers. Many works primarily focus on interweaving full-attention layers with eﬀicient alternatives, such as windowed attention or linear approximations, to mitigate the quadratic memory footprint of long contexts [38, 39, 40]. For instance, models such as MiMo [13], MiniMax-M1 [12], and Qwen [3] systematically balance standard attention with eﬀicient attention at configurable layer-wise ratios. Furthermore, automated strategies like those explored in GLM-5 [1] utilize Neural Architecture Search (NAS) to discover optimal layer-wise configurations, demonstrating that static interleaving is often suboptimal. Despite the prevalence of layer-wise hybridization in state-of-the-art architectures, prior work has noted that training hybrid models combining linear attention with full attention remains challenging, with many efforts ultimately focusing on full or sparse attention variants [1, 14, 4], suggesting that the design space of attention hybridization remains underexplored.

Token-wise Hybridization. To achieve finer granularity, recent studies have proposed blending that operates at the token or sequence level [32, 41, 42, 43]. Among them, STILL [32] employs saliency-based gating to decide which tokens warrant the precision of FA versus the eﬀiciency of linear or sparse approximations. While these approaches refine the allocation granularity from the layer level to the token level, they face a potential challenge: maintaining stable and consistent sequence representations across long contexts is non-trivial when different tokens are processed through heterogeneous attention pathways. Few existing works have reported evaluation results beyond 32K context length for such methods. Moreover, the very pursuit of such stability and consistency may inadvertently drive different attention patterns toward convergence, thereby diminishing their ability to complement each other. Additionally, in the case of sparse attention, the set of tokens retrieved varies across queries, necessitating the retention of full KV caches and thus undermining memory eﬀiciency during inference.

Head-wise Hybridization. Recognizing the functional heterogeneity of multi-head attention, recent work has explored head-wise hybridization along two distinct paradigms. The first paradigm performs per-head

selection: methods such as DuoAttention [44] and Elastic Attention [45] treat each head as the unit of allocation, using heuristic partitioning or learned routers to designate “critical” heads for FA while assigning the remainder to sparse attention variants. However, since sparse attention and standard FA share the same quadratic dot-product computation core, their complementary benefits are inherently limitedboth remain constrained by similar attention patterns, yielding marginal gains rather than a fundamental eﬀiciency breakthrough. The second paradigm performs per-head mixing. For example, Hymba [46] processes the input through two parallel branches—a full multi-head FA branch and a Mamba SSM branchand fuses their output features via element-wise averaging. Although these fusion-based approaches enrich representational capacity, they apply a uniform mixing strategy across all heads without accounting for their inherent functional specialization. In contrast, our architecture leverages interpretability analysis to uncover the inherent functional specialization of different attention heads, and strategically assigns FA and LA to heads based on their complementary strengths. By promoting representational complementarity, it harmonizes the high-precision recall of FA with the constant-cost global context modeling of LA, achieving a synergistic fusion that fully exploits the distinct capabilities of each attention mechanism.

##### 2.4 Taming Transformer to Hybrid Architecture

The prohibitive cost of pre-training hybrid architectures from scratch catalyzes a surge in cross-architecture distillation methods, aiming to transfer the reasoning capabilities of pre-trained Transformers into more eﬀicient counterparts. Early attempts focus on distilling Transformers into pure RNN or State-Space Models (SSMs). For example, T2R [47] and RADLADS [48] demonstrate the feasibility of architectural conversion. However, MOHAWK [49] and LoLCATs [50] later highlight that naive distillation often leads to performance degradation on recall-intensive tasks, as the disparity between quadratic attention and linear-time mixers creates a significant knowledge transfer barrier. To bridge this gap, recent works shift toward multi-stage pipelines and principled initialization strategies. HedgeMamba [51] proposes a two-stage recipe that first distills softmax attention into a linearized kernel (Hedgehog) and then uses it to initialize an SSM-based Mamba mixer. Complementary approaches, such as KL-Guided [52], emphasize the criticality of selective layer conversion to maximize long-context retention. We leverage the HALO [53] as our foundation, adapting its eﬀicient distillation procedure to support our novel head-wise hybrid structures. By extending HALO to handle head-level feature distribution shifts through decoupled QKV projections and modulated fusion, our method successfully transitions from standard layer-wise conversion to a more granular, head-specific hybridization strategy.

### 3 Preliminaries

##### 3.1 Causal Patching Methods

Activation Patching. Activation patching [35] is a causal intervention technique for quantifying the contribution of individual model components to a target behavior. The procedure requires a paired input: a clean input x on which the model produces the correct behavior, and a corrupted input x′ obtained by a minimal edit that alters the target answer. For a component of interest—e.g., the output Ol,h of attention head h at layer l—the clean-run activation is replaced with its corrupted-run counterpart while all other components remain at their clean values. The resulting change in model behavior, measured by a scalar readout such as the logit difference

m(x) = z[a+] − z[a−], (1)

where z denotes the model’s output logits, and a+, a− are the correct and counterfactual answer tokens respectively, quantifies the component’s causal necessity: a large drop indicates a causally indispensable component, while a negligible change indicates a dispensable one [16].

Path Patching. While activation patching measures each component’s direct effect on the output, path patching [16] enables finer-grained attribution by restricting the intervention to a specific computational path. Concretely, to measure how much an upstream head contributes through a particular downstream head, one first runs the corrupted input and records the activations the upstream head sends to the downstream head; then runs an otherwise-clean forward pass but substitutes only those recorded activations at the downstream head’s input. The resulting behavioral change isolates the upstream head’s contribution along that specific path, revealing not only which components are important but also how they contribute to each other.

##### 3.2 Grouped-Query Attention Mechanism

Modern large language models predominantly adopt Grouped-Query Attention (GQA) [54] to balance inference eﬀiciency and modeling capacity. Thus, we consider a Transformer model with L layers, where each layer consists of a GQA and a Multi-Layer Perceptron (MLP). Let H denote the total number of query heads and G denote the number of key-value heads, with G < H. The set of query heads H = {1,...,H} is partitioned into G disjoint groups G = {1,...,G}, such that all query heads in a group share a single key-value head.

Given an input sequence represented as a matrix X ∈ RT×d, where T is the sequence length and d is the hidden dimension, we define the head dimension as dh = d/H. In standard GQA implementations, the key and value heads share the same dimension dh as the query heads.

The attention mechanism employs three global learnable projection matrices: WQ ∈ Rd×d, WK ∈ Rd×d

G, and WV ∈ Rd×d

G, where dG = d · HG is the total dimension for all key/value heads. The global projected representations are computed as:

Q = XWQ ∈ RT×d, K = XWK ∈ RT×d

, V = XWV ∈ RT×d

. (2)

G

G

These tensors are split along the feature dimension to obtain individual head representations. For each query head h ∈ H, let Qh ∈ RT×d

h denote the slice of Q corresponding to the h-th head. Similarly, for each key-value group g ∈ G, let Kg ∈ RT×d

h denote the slices of K and V corresponding to the g-th group. Note that each Kg and Vg has dimension dh, matching the query head dimension.

h and Vg ∈ RT×d

Let g(h) ∈ G be the mapping function that assigns query head h to its corresponding key-value group. The shared key and value matrices for the group associated with head h are denoted as Kg(h) and Vg(h). In the standard FA mode, the output matrix for query head h is computed as:

)Vg(h). (3)

Oh = softmax(

QhK⊤g(h) √dh

The outputs from all heads are concatenated along the feature dimension to form O = [O1,O2,..., OH] ∈ RT×d, which is then projected to produce the final layer output Y ∈ RT×d:

Y = OWO, (4) where WO ∈ Rd×d is the output projection matrix.

##### 3.3 Linear Attention with Gated DeltaNet (GDN)

Although GQA reduces the KV cache footprint, its quadratic computational complexity still hinders the deployment of models in long-context scenarios, thereby motivating the development of linear attention mechanisms. Given that GDN [9] has already been adopted in industrial-grade hybrid models, we likewise select GDN as the representative linear attention layer.

Standard Linear Attention. Standard linear attention [37] maintains a matrix-valued recurrent state St ∈ Rd

h×dh that accumulates key-value associations. Given head-wise input projections qt,kt,vt ∈ Rd

h at step t, the state update and output are defined as:

St = St−1 + ktv⊤t , (5) yt = S⊤t qt. (6)

While computationally eﬀicient, this formulation lacks a forgetting mechanism, causing the state to grow unbounded and leading to significant interference over long contexts.

Gated DeltaNet. To address these limitations, GDN interprets the state update as online gradient descent on a reconstruction loss, incorporating both a delta rule and a gating mechanism. Specifically, GDN introduces a scalar forget gate αt ∈ (0,1] and a learning rate βt to regulate memory retention. The state update rule is formulated as:

(

)

St = αt

I − βtktk⊤t

St−1 + βtktv⊤t , (7)

(

)

where I ∈ Rd

I − βtktk⊤t

h×dh is the identity matrix. The term

acts as a projection that corrects the state

based on the current key, while αt serves as a data-dependent weight decay factor, controlling the lifespan of stored memories. The output is then computed as:

yt = S⊤t qt. (8)

This formulation allows GDN to maintain a constant-size hidden state S while dynamically balancing the retention of new information and the forgetting of outdated associations, thereby facilitating stable and eﬀicient linear-time inference.

### 4 Method

We present the design of our head-wise hybrid architecture in three parts. Section 4.1 details an interpretability-based head selection algorithm that identifies critical heads to be preserved with full attention. Section 4.2 then describes how FA and LA structures decompose input features and are organically fused at the head level. Finally, Section 4.3 explains the transfer learning pipeline for eﬀiciently converting a pretrained standard FA model into the proposed hybrid.

##### 4.1 Head Importance Estimation via Causal Intervention

Deciding which heads should retain full attention—and which can be safely approximated by linear recurrence—is the decision on which our eﬀiciency–quality trade-off hinges. Our procedure operates in three steps: (1) measure each head’s direct causal effect on the target behavior via activation patching (receivers); (2) trace upstream contributions via iterative path patching (senders); and (3) fuse the per-head scores across multiple target capabilities into a single ranking. We describe each step below.

Observational diagnostics such as logit lens projections (Figure 2a) or attention-pattern visualization can reveal head-level differences, but they show what a head computes, not whether the model depends on it: a head may attend strongly to a target token yet be overridden downstream, or attend diffusely but carry the critical signal in its output projection. Only causal intervention—replacing a head’s output and observing the behavioral change—tests necessity directly, distinguishing load-bearing heads from correlated bystanders.

We therefore estimate, for each query head, its causal necessity for the model’s target capabilities, and retain FA only for the heads that prove causally indispensable. Building on the causal patching framework introduced in Section 3.1, we describe the procedure below for an abstract set of target capabilities C; its concrete instantiation (long-context retrieval and general ability) is deferred to Section 5.

##### Counterfactual construction and readout. For a capability c ∈ C we construct paired inputs (Sec-

- tion 3.1) by replacing the correct answer in each prompt with a randomly generated counterfactual of the same type and token length (e.g., substituting one multi-digit number for another), producing a structureand length-preserving edit that flips the target answer while holding the surrounding context fixed. This symmetric token replacement keeps every activation on the model’s natural data distribution, avoiding the out-of-distribution artifacts of additive-noise corruption [55]. Since our target capabilities involve multitoken answers (e.g., multi-digit retrieval results), we extend the standard single-token logit difference (Eq. 1)

to a span-level readout. Let A denote the set of answer-token positions and let a+j , a−j be the correct and counterfactual tokens at position j ∈ A. We aggregate with an exponential decay factor λ ∈ (0,1), where each position j receives weight λj, emphasizing the earlier, more informative answer tokens:

1 Z ∑

, Z = ∑

(

)

λj, (9)

zj[a+j ] − zj[a−j ]

λj

m(x) =

j∈A

j∈A

where zj denotes the output logits at position j, and we use λ = 0.9 by default. We adopt the logit difference rather than probability or cross-entropy because it is approximately linear in the residual stream and monotone in the underlying capability, avoiding the softmax-saturation and probability measurementfloor effects that distort probability-based scores [55, 56].

Necessity via activation patching (receivers). We apply activation patching (Section 3.1) at the granularity of individual heads, with a critical refinement: we freeze the downstream attention outputs to their clean values, so that the patched signal can propagate only through the residual stream (the running sum of all component outputs that accumulates across layers)—a direct-effect measurement. Without this

refinement, downstream heads could compensate for the corrupted signal, masking the true importance of the patched head. The head’s importance, which we denote IEl,h, is the resulting drop in the readout, normalized to the clean–corrupt range:

) m(x) − m(x′) ∈ [0,1], (10)

(

x; Ol,h←Ol,h(x′)

m(x) − m

IEl,h =

where m(x; Ol,h ← Ol,h(x′)) denotes the readout when head h at layer l has its output replaced with the corrupted-run value while all other components remain at their clean state. The score is averaged over a small calibration set of counterfactual pairs drawn from the target capability (see Section 5.7 for the exact configuration). A value IEl,h≈0 marks a head whose corruption leaves the behavior intact (dispensable, and safely approximated by linear recurrence), whereas IEl,h≈1 marks a head whose corruption alone collapses the capability. High-IE heads are the receivers—heads that directly write the capability-critical signal into the residual stream. We denote the receiver importance of head h as IErecvh .

Upstream attribution via path patching (senders). A head can be causally important without writing to the output directly, by feeding a receiver. We trace one step back with path patching (Section 3.1): for each candidate upstream head, we run the corrupted input and record the activations it sends to the receivers; we then run an otherwise-clean forward pass but substitute only those recorded activations at

the receivers’ inputs. The resulting normalized drop in the readout, which we denote IEsendh , measures the upstream head’s contribution through the receiver pathway. We iterate, promoting the senders found in one round to receivers in the next, until new contributions vanish. For long-context retrieval, this converges in roughly two rounds—a shallow circuit—so a small top-K head set, rather than a full multi-stage circuit, already captures almost all of the causal signal.

Per-capability score and cross-capability fusion. For capability c we combine the two roles by the larger of the receiver and sender effects, weighted by a task-consistency factor κ(hc) equal to the fraction of sub-probes in which h exceeds the importance threshold (i.e., IEl,h ≥ 0.01) in its strongest role. This factor down-weights heads that achieve a high importance score on a single sub-probe but are negligible on the rest, favoring heads whose contribution is stable across tasks:

) · κ(hc). (11) To preserve all target capabilities jointly, we min–max normalize each s(c) to [0,1] (the per-capability drops differ in scale), yielding sˆ(hc), and fuse them by a weighted mean,

(

IErecvh , IEsendh

s(hc) = max

Sh = ∑ c∈C

wc sˆ(hc), ∑ c∈C

wc = 1, (12)

with equal weights by default, reflecting an equal prior over capabilities when no task-specific preference is given. We rank heads by Sh and retain FA for the top-K heads, assigning the remainder to LA, with K set by the target FA budget.

##### 4.2 Head-wise Hybridization

FA excels at precise content retrieval through token-level interaction, yet its quadratic complexity constrains scalability to extremely long sequences. LA, by contrast, enables eﬀicient long-context processing with O(T) complexity, but its fixed-size recurrent state inevitably compresses historical information, leading to degraded recall fidelity over extended contexts. To harness the complementary strengths of both mechanisms, we propose a Head-wise Hybrid Attention mechanism in Figure 3. Instead of applying a uniform attention pattern across all heads or layers, we selectively assign each query head to either the FA branch or the LA (GDN) branch based on its functional importance. This fine-grained hybridization allows the model to retain critical reasoning capabilities in specific heads via FA, while leveraging the O(T) eﬀiciency and stable state evolution of GDN to facilitate effective context extension and reduce overall computational complexity.

Head Partitioning. Let H = {1,...,H} denote the set of all query head indices. We partition H into two disjoint subsets: HF for heads assigned to the FA, and HL for heads assigned to the GDN branch, such that HF ∪HL = H and HF ∩HL = ∅. The assignment is determined via an interpretability-driven selection process (detailed in Section 4.1), which identifies heads crucial for precise context retrieval and complex reasoning.

Hybrid Attention

𝑊

Full Attention

|0|1|2|3|4|5|6|7|
|---|---|---|---|---|---|---|---|

𝑊

|Norm & Scale|
|---|

𝑂 𝑂

Softmax attention operation

Full Attention Linear Attention

RoPE RoPE

K V

###### Q

0 2 3 0

2 𝐾

0 2 2 0 1 3

3 𝑉

1 2 3

4 5 6 7 0 1 3

𝐾

𝑉

𝑄

0 1 2 3 4 5 6 7

0 1 2 3 0 1 2 3

Heads

𝑄

| | | | | |
|---|---|---|---|---|
|0|1|2|3| |

0 1 2 3 4 5 6 7

0 1 2 3 Heads

𝑊 𝑊 𝑊

𝑊 𝑊 𝑊

(a) (b)

- Figure 3: Comparison of standard FA and our proposed head-wise hybrid attention. A subset of heads retains the original FA branch, while the remaining heads employ GDN as a representative LA variant to reduce computational overhead.

Parallel Branch Computation. For each head h ∈ H, the input projections Qh,Kg(h),Vg(h) ∈ RT×d

h

are computed as defined in Section 3.2. These projections are then routed to their respective branches: for h ∈ HF, the output OFh ∈ RT×d

h is computed via standard softmax attention; for h ∈ HL, the output OLh ∈ RT×d

h is computed via the Gated DeltaNet recurrence.

Head-wise Scale-normalized Fusion. A fundamental asymmetry exists between FA and LA at the level of attention distributions. As NaLaFormer [33] reveals, FA’s softmax exponential function naturally incorporates query vector magnitude into attention scores, producing sharp, low-entropy distributions concentrated on a few key tokens. LA’s normalization operations, by contrast, cancel out query magnitude, yielding smoother, higher-entropy distributions. This divergence in attention patterns directly propagates to the output features: FA heads produce representations strongly modulated by query norm and dominated by a small set of high-weight tokens, while LA heads generate more uniform, query-norm-agnostic representations. These distributional shifts may destabilize optimization when the two feature streams are naïvely concatenated.1 To mitigate this, we propose an independent normalization and head-wise modulation scheme that reconciles the distributional gap while preserving the structural integrity of the original attention heads.

Let OFh ∈ RT×d

h and OLh ∈ RT×d

h denote the raw outputs of the FA and GDN branches for head h, respectively. We first apply RMSNorm independently to each head’s output to unify their feature scales:

Oˆh = Norm(Oh), where Oh = {

OFh if h ∈ HF, OLh if h ∈ HL.

(13)

Crucially, we concatenate these normalized head outputs {Oˆ1,..., OˆH} along the head dimension according to their original head indices to form a tensor Oˆ ∈ RT×H×d

h. This index-preserving concatenation ensures that the positional and functional identity of each head is maintained within the global representation.

To fully utilize the characteristics of these two attention mechanisms, we introduce a learnable head-wise scaling vector γ ∈ RH. This vector assigns a scalar weight to each head based on its original index, allowing the model to adaptively recalibrate the contribution of each head regardless of its underlying attention type. The modulated tensor O˜ ∈ RT×H×d

h is computed via broadcasting multiplication along the head dimension: O˜:,h,: = γh · Oˆ:,h,:, ∀h ∈ [1,H]. (14)

1We observe that when FA and LA are unevenly distributed, directly concatenating features leads to a significant degradation in capabilities in Table 5.

Linear Attention

Full Attention

Norm

𝑍

𝛼,𝛽

Linear attention operation

Softmax attention operation

| | |
|---|---|
|RoPE| |

Scale

RoPE

𝐾

𝑉

𝑍

𝑄 𝐾

𝑉

| | |𝑄| |
|---|---|---|---|
|0|2|3|7|

| | |
|---|---|
|𝑊 , , | |

|𝑊|
|---|

| | | | |
|---|---|---|---|
|1|4|5|6|

0 1 3

0 1 3

|0|2|2|3|
|---|---|---|---|

0 2 2

3

- Figure 4: Details of FA and LA branches. We use GDN as a representative linear attention variant to illustrate the design. Specifically, we replace the original RoPE in the FA branch with a scaling mechanism, while applying RoPE to the LA (GDN) branch to enhance its positional awareness.

Finally, O˜ is reshaped into RT×d (where d = H · dh) and projected by the shared output matrix WO to produce the final attention output. This design ensures that the distributional gaps between FA and GDN features are bridged before fusion, while preserving the distinct functional roles of individual heads through learnable scalars aligned with their original positions.

Branch-specific Refinements. The internal structural designs of the two attention types—such as positional encoding, gating mechanisms, and key-value head expansion—also exert a significant influence on head-wise hybridization. While HypeNet [53] validates the effectiveness of these designs in the context of layer-wise mixing, we verify them under the setting of head-wise hybridization (Figure 4).

For the FA branch, we aim to enhance attention stability in long-context, especially extended-context scenarios. To this end, we depart from the standard Rotary Position Embedding (RoPE) protocol: following HypeNet [53] and DroPE [57], we remove RoPE and instead apply a log-scale coeﬀicient to the query features after training, which effectively stabilizes the attention distribution. In addition, motivated by gated attention [58], we introduce an auxiliary gate branch within the FA module. This design not only boosts the model’s representational capacity but also effectively alleviates the “attention sink” phenomenon commonly observed in high-precision attention heads [59].

For the GDN branch, we aim to strengthen its positional awareness within the inherent receptive field. Retaining the native short convolution and gating mechanism for local pattern recognition and stable state transitions, we explicitly integrate RoPE into the query and key projections to compensate for the limited positional sensitivity of linear recurrence, enabling precise capture of relative positional relationships within the perception window. Furthermore, to enhance representational capacity, we expand the number of key-value heads to match that of query heads—effectively transitioning from a GQA-like to an MHA-like configuration. Although described here in the context of GDN, these enhancements are directly transferable to other linear attention variants.

##### 4.3 Eﬀicient Hybrid Transfer Learning

Transforming a pre-trained Transformer into a hybrid architecture offers a practical path to leveraging the strong performance of the original model while substantially reducing training overhead. The key challenge is that the newly introduced GDN branches start with fundamentally different computational dynamics than the FA heads they replace; direct fine-tuning from this mismatched initialization leads to optimization instability. Following prior work [52, 53], we address this through a three-stage pipeline with a principled progression: first align local behavior (Stage 1: each layer’s output should match the original FA layer), then align global semantics (Stage 2: the model’s output distribution should match the teacher’s), and finally adapt to the target domain (Stage 3: standard fine-tuning). Each stage builds on the stability achieved by the previous one.

##### 4.3.1 Stage 1: Parameter Migration and Layer-wise Output Alignment

The first stage focuses on initializing the hybrid architecture and aligning the local behavior of each layer. After identifying the critical heads to remain as FA via our interpretability-driven selection mechanism, we initialize the remaining heads as GDN branches.

Parameter Migration Initialization. We employ a careful weight initialization strategy to minimize the distribution shift at the beginning of training:

- • FA Branch with Gate Modulation: For the heads retained as FA, we introduce a lightweight gate branch to allow for dynamic modulation during later stages. To encourage the gate to initially approximate an identity function (i.e., maintaining fidelity to the original FA output), we initialize the gate branch projection weights to near-zero and set the bias so that initial gate values are close to 1. This guarantees that the FA branch starts with its pre-trained functionality intact, providing a stable anchor for the hybrid model.
- • GDN Branch Weight Reuse: For the heads converted to GDN, we directly reuse the Q,K,V projection weights from the original pre-trained FA layers. Since our model adopts GQA, where the number of KV heads is smaller than the number of Query heads, we handle the dimension mismatch by applying a channel-wise repeat technique. Specifically, the key-value projection matrices for the GDN branch are initialized by repeating the channels of the original pre-trained key-value weights to align with the required GDN internal dimensions. This ensures that the LA states initially capture similar key-value interactions as the original attention heads.

Layer-wise Hidden State Alignment. With the parameters initialized, we freeze the pre-trained backbone and optimize only the newly introduced hybrid attention layers. The objective is to align the output hidden states of each hybrid attention layer with those of the original pre-trained FA layer. We employ the

Mean Squared Error (MSE) loss between the hidden state representations of the original FA layer (H(FAl) ) and the hybrid layer (H(Hybridl) ) for each layer l:

∑L

∥H(FAl) (x) − H(Hybridl) (x)∥22. (15)

Lalign =

l=1

This layer-wise MSE alignment ensures that the hybrid architecture locally mimics the feature space of the pre-trained model, preventing error accumulation in deeper layers and providing a robust foundation for subsequent global distillation.

##### 4.3.2 Stage 2: Global Logits Distillation

In the second stage, we unfreeze the entire model and perform global knowledge distillation to align the final output distribution of the hybrid student model with the original pre-trained teacher model. Unlike Stage 1, which focuses on intermediate layer consistency, this stage targets the final logits to ensure global semantic coherence. We combine the KL divergence loss on the final logits with a standard cross-entropy loss on the ground truth labels:

LKD = DKL (Pteacher(·|x) ∥ Pstudent(·|x)), (16)

where Pteacher and Pstudent are the softmax probabilities of the teacher and student models, respectively. This stage allows the hybrid model to recover any minor performance degradation caused by the linear approximation and gate modulation, ensuring that the global linguistic capabilities of the pre-trained backbone are fully preserved.

##### 4.3.3 Stage 3: Long-Context Fine-tuning

By this point, the hybrid architecture has been well-initialized and its output distribution aligned with that of the teacher model through the preceding stages. We therefore adopt standard supervised fine-tuning with the Next Token Prediction (NTP) objective in the final stage, using a longer context length than in the first two stages to consolidate long-context capabilities.

###### LNTP = −∑

logPstudent(xt+1|x1:t). (17)

t

Table 1: Default training details of three training stages, which follow the HypeNet [53]. Stage Batch Size Context Length Training Steps (Tokens) LR scheduler Learning Rate

- 1 32 512 20K (0.3B) Cosine 1e-3 → 1e-5

- 2 96 512 20K (1.0B) Cosine 1e-4 → 1e-5

- 3 128 16384 500 (1.0B) Constant 1e-5

- 5 Experiments We conduct comprehensive experiments to evaluate HydraHead along four progressively refined questions:

- (1) Does HydraHead outperform existing hybridization strategies? (Section 5.2): Controlled comparison against layer-wise, token-wise, and other head-wise alternatives under the same training pipeline.
- (2) What makes it work? (Section 5.3-5.4): Ablation of structural components and fusion mechanisms.
- (3) Does interpretability-guided selection matter? (Section 5.5-5.7): Validation of causal selection vs. naive strategies, and analysis of head importance anatomy. (4) How far can it scale? (Section 5.8-5.9): Training optimization and benchmarking against open-source models.

- 5.1 Experimental Setup 5.1.1 Model, Training, and Head Selection Configuration

Model and Data Configuration. We conduct our primary experiments on the Qwen3-1.7B model [34], which serves as the backbone for the initialization of our hybrid architecture. For training data, following the recent work [5, 53], we utilize the FineWeb-Edu dataset [60], a high-quality subset of the FineWeb corpus specifically filtered for educational content. This dataset is made available under the Open Data Commons Attribution License (ODC-By).

Model settings. By default, we retain FA computation for 25% of the heads, while the remaining 75% use the GDN structure.

Head selection configuration. We construct the calibration set for head importance estimation (Sec-

- tion 4.1) from five NIAH sub-probes in the RULER benchmark (three single-key and two multi-key retrieval tasks), each comprising eight counterfactual pairs at 4K context length. All patching scores are computed

in fp16 precision. A head is deemed critical when its normalized importance |IEl,h| ≥ 0.01; the per-head ranking is already stable from roughly six samples (validated in Section 5.7 and Appendix C.1).

Training details. We train the hybrid models with the three-stage strategy as mentioned in Section 4.3. In our ablation studies, we follow the default settings of existing work [53]. Some training configurations within each stage are summarized in Table 1.

- 5.1.2 Evaluation benchmarks

To comprehensively assess the effectiveness of our method, we evaluate the model in two primary dimensions: Long-Context Retrieval Capability (Needle In A Haystack) and General Reasoning Proficiency.

Long-Context Retrieval (RULER). We employ the RULER benchmark [61] to evaluate the model’s ability to retrieve information from extended contexts. Specifically, we focus on two categories within RULER: Single-Key (1/2/3) and Multi-Key (1/2) retrieval tasks, which test the model’s precision in locating single or multiple distinct pieces of information embedded in long sequences. We evaluate performance at five context lengths: 16K, 32K, 64K, 128K, and 256K tokens. Given that Qwen3-1.7B has a native context window of 32K, we categorize the evaluation results into two metrics:

- • Native Context Performance: The average accuracy across 16K and 32K lengths, reflecting the model’s baseline capability within its pre-trained window.
- • Extended Context Performance: The average accuracy across 64K, 128K, and 256K lengths, measuring the model’s extrapolation ability and the effectiveness of our hybrid architecture in handling sequences beyond the native limit.

- Table 2: Architecture comparison of various hybrid designs. Cache denotes the relative key-value cache under a 16K context length, where the standard FA cache is normalized to 1.0. Light. denotes the lightning attention [72]. ∗ and ‡ indicate the hybrids are similar to HypeNet [53] and Liger [43], respectively. † indicates that an FA layer is inserted after every 3 LA layers. More details are provided in Appendix A.

RULER Single RULER Multi-Key General Reasoning (Native) (Extended) (Native) (Extended) (Hard) (Easy)

Model LA Cache

Layer-wise Hybrid Transformers

FA & LA∗ Light. 0.25 89.07 85.00 24.85 24.37 19.80 59.72 FA & LA† Light. 0.25 86.03 67.76 31.70 17.80 15.94 60.29 FA & LA GDN 0.25 47.63 7.02 22.50 5.53 16.44 58.18 LA-FA & LA GDN 0.25 58.50 8.13 19.00 6.83 18.42 58.70

Token-wise & Head-wise Hybrid Transformers

Token-wise GDN 0.25 20.77 3.73 16.05 2.43 47.31 63.40 Token-wise‡ GDN 0.25 96.03 58.78 26.20 11.03 36.66 63.37 Head-wise Mixing GDN 1.00 93.20 60.42 37.35 14.53 38.07 62.77 HydraHead GDN 0.35 98.47 87.49 37.10 27.37 31.03 62.12

General Reasoning and Knowledge. To ensure that hybrid attention does not compromise the model’s general reasoning capabilities in practical scenarios, we evaluate on a broad suite of benchmarks spanning two tiers: standard academic benchmarks commonly used in prior work, and more challenging evaluations widely adopted for assessing frontier large language models.

- • Basic Competency (Easy): Following prior works [6, 42, 53], we include six widely used datasets that assess basic language understanding and common sense reasoning: ARC-Challenge (norm score) [62], ARC-Easy [62], HellaSwag (norm score) [63], Winogrande [64], PIQA [65], and LAMBADA-OpenAI [66]. These tasks serve as a sanity check for the model’s foundational capabilities.
- • Advanced Reasoning (Hard): To probe deeper cognitive abilities, we evaluate on four challenging benchmarks requiring complex logical deduction, mathematical reasoning, and code generation: MMLU [67], GSM8K [68], MBPP [69], and BBH [70]. This tier critically tests whether the eﬀiciency gains from our hybrid design come at the cost of high-level reasoning performance.

All evaluations are conducted using the lm-evaluation-harness framework [71], adhering to its default configurations for each benchmark. This includes standard zero-shot or few-shot settings and greedy decoding as specified by the original protocols, unless otherwise noted.

##### 5.2 Hybrid Architecture Comparison

We first systematically compare HydraHead against representative hybridization baselines spanning layerwise, token-wise, and head-wise paradigms under the transfer learning protocol described in Section 4.3, to elucidate its distinct advantages.

Experimental Setup. We adopt GDN as the representative linear attention mechanism and construct baselines across three hybridization paradigms: layer-wise, token-wise, and head-wise. For layer-wise hybrids, we follow [53] to selectively retain FA in 25% of the layers while employing GDN in the remainder, and additionally include a uniform variant that evenly distributes FA layers for comparison. For tokenwise hybrids, we consider two variants. The first combines 4K-context Sliding Window Attention (SWA) with GDN within every layer. The second adopts a layer-interleaved design inspired by Liger [43]: 75% of the layers use SWA (window size 128) combined with GDN, while the remaining 25% combine full FA with GDN. For head-wise mixing, we implement a fusion-based variant inspired by [46] that processes the input through parallel FA and GDN branches, each operating over the full set of heads, and fuses their outputs at the feature level. For head-wise selection, existing per-head hybridization methods have focused on pairing FA with sparse attention [44, 45]. We use our HydraHead as the representative of this paradigm, retaining 25% of heads as FA. Detailed architectural specifications are provided in Appendix A.

- Results and Analysis. Table 2 presents a comprehensive study comparing our proposed head-wise hybrid architecture, HydraHead, against representative layer-wise, token-wise, and head-wise hybrids under identical training conditions and parameter budgets. Overall, we observe distinct performance trade-offs across architectures: while token-wise and head-wise hybrids demonstrate significant advantages in challenging general reasoning tasks, layer-wise hybrids generally exhibit superior capabilities in extending context length—with the notable exception of our method. Our HydraHead achieves the best overall balance, delivering state-of-the-art performance in long-context evaluations (particularly in context extrapolation) while maintaining robust general domain capabilities.

To understand these dynamics, we first examine the strength of token-wise and head-wise designs in complex reasoning. Compared to the layer-wise baseline, the other two mixing architectures (token-wise and headwise) achieve substantially better performance in diﬀicult general reasoning benchmarks, with gains exceeding 10%. This suggests that fine-grained feature interaction within each layer is crucial for preserving the complex logical dependencies required for high-level reasoning. Layer-wise hybrid models, by uniformly substituting entire layers with LA, tend to disrupt these delicate attention patterns, leading to significant performance degradation in tasks requiring precise contextual understanding. However, this advantage in general reasoning traditionally comes at the cost of limited scalability in long-context scenarios, where naive token-wise and head-wise hybrid models struggle to maintain retrieval accuracy as the sequence length increases. In contrast, our model transcends this trade-off. By leveraging interpretability-guided head-wise selection, we preserve the critical global context awareness necessary for long-context extrapolation, thereby achieving superior performance in extended sequence tasks while maintaining a significant advantage in general-domain reasoning over layer-wise hybrids.

We note that the pure SWA+GDN token-wise variant (without FA-interleaved layers) exhibits notably inferior long-context performance compared to other variants, likely because the current transfer learning setup is insuﬀicient for it to generalize to sequence lengths beyond 4× the training window.

5.3 Structural Components Study

We conduct a systematic ablation study to dissect the contribution of each architectural component in our head-wise hybrid design, focusing on the interaction between the FA and GDN heads.

Experimental Setup: In this series of experiments, we adopt a fixed head-wise allocation strategy to isolate the impact of structural modules from the variability of dynamic routing. Specifically, for every transformer layer, we assign the first 75% of attention heads to the GDN branch and the remaining 25% to the FA branch (Figure 6 (a)). This static configuration ensures a consistent computational budget across all variants, allowing for a fair comparison of how different internal mechanisms affect model expressiveness and stability.

Progressive Module Integration. We start from a minimal hybrid baseline and progressively introduce key modules to observe their individual and combined effects. The variants are defined as follows:

- • Base Hybrid: A naive combination of FA and GDN heads without any specialized stabilization or feature alignment techniques. Note that the normalization of output (see Equation 13) is used.
- • + FA NoPE & Scale: We remove positional embeddings (NoPE) for the FA branch and use position-related scaling instead. The scale hyperparameter is set to 500.
- • + GDN RoPE: We apply location encoding to the GDN branch to enhance content-aware capabilities.
- • + FA Gate: We incorporate a gating mechanism within the FA heads to dynamically modulate their contribution based on input context.
- • + GDN MHA: Unlike FA, which uses GQA to reduce key-value cache, we use MHA in GDN to better utilize the multi-head mechanism to capture semantic associations.
- • + Query Decomposition: Finally, we decompose the query projection matrices for FA and GDN heads. This avoids potential optimization bottlenecks in practice that may arise from numerical precision limitations [73] and implicit gradient conflicts between the two distinct heads.

- Results and Analysis. Table 3 summarizes the performance variations resulting from the incremental addition of each module. We observe that replacing the positional encoding in the FA branch with scale

- Table 3: Ablation study on structural components of the hybrid architecture under the fixed head-wise allocation strategy. Each module in each row is added based on the previous row.

Model

RULER Single RULER Multi-Key General Reasoning (Native) (Extended) (Native) (Extended) (Hard) (Easy)

Base Hybrid 35.33 2.96 9.90 1.60 31.14 62.42 + FA NoPE & Scale 64.73 40.58 31.65 8.47 27.87 62.22 + GDN RoPE 72.80 56.38 24.00 14.00 26.36 62.09 + FA Gate 76.07 60.89 31.05 15.07 26.98 62.60 + GDN MHA 79.97 54.27 31.35 14.17 29.66 62.42 + Query Decomposition 85.63 62.62 27.35 13.67 28.65 62.59

- Table 4: Query decomposition yields more pronounced gains under interpretability-guided head selection than under fixed allocation.

RULER Single RULER Multi-Key General Reasoning (Native) (Extended) (Native) (Extended) (Hard) (Easy)

Model

Basic Model 98.70 81.73 38.35 25.93 31.70 62.68 w/o Query Decomposition 91.30 79.43 46.85 25.77 31.79 62.78

modulation significantly improves the hybrid model’s long-context capabilities. In both Single-key and Multi-key tests, performance within the original context length increased by over 10%, while extended context lengths improved from negligible gains to a substantial level. However, this modification reduces the positional awareness of the FA branch, leading to a decline of more than 3% in general reasoning performance. Adopting a Multi-Head Attention (MHA) configuration for the GDN branch helps recover part of this lost general capability. The full configuration further decomposes the query projection matrices, resulting in the splitting of QKV projection matrices for both branches. This configuration achieves optimal performance on the RULER Single test while maintaining robust results in Multi-Key and general benchmarks. This advantage also holds consistently under the interpretability-guided head selection2 setting (Table 4). Therefore, we denote this version model as Basic Model.

##### 5.4 Feature Fusion Designs

We systematically investigate feature fusion strategies based on a head-wise hybrid architecture. The core challenge lies in reconciling the different statistical properties of features generated using softmax and linear attention mechanisms—particularly differences in amplitude and variance. Directly concatenating these heterogeneous features often imposes significant optimization pressure on subsequent output projection layers, potentially leading to training instability or performance degradation.

Experimental Setup. We conduct experiments under the interpretability-guided head selection strategy, where heads are selected based on their causal importance scores derived from mechanistic interpretability analysis. Unlike the fixed strategy, this global selection allows for heterogeneous layer-wise distributions; specific layers may exhibit extreme configurations, such as being composed entirely of GDN/FA heads, while others retain a higher concentration of GDN/FA heads. This setup creates a more complex and realistic optimization landscape, enabling us to rigorously assess the robustness of different fusion mechanisms in handling significant distributional shifts between attention branches.

Feature Fusion Variants. To systematically evaluate the impact of feature alignment strategies, we construct three variants based on the Basic Model, progressively introducing mechanisms to handle the distributional heterogeneity between FA and GDN branches:

• Direct Concatenation (w/o Norm): This baseline variant removes all normalization layers, directly concatenating the raw output features from FA and GDN heads.

2It’s denoted as Global Interpretability Screening in Section 5.5.

- Table 5: Comparison of feature fusion variants under the interpretability-guided global selection.

RULER Single RULER Multi-Key General Reasoning (Native) (Extended) (Native) (Extended) (Hard) (Easy)

Model

Basic Model 98.70 81.73 38.35 25.93 31.70 62.68 w/o Norm 85.07 71.36 41.90 23.93 30.62 62.58 Head-wise Scale 98.47 87.49 37.10 27.37 31.03 62.12 Head-wise Gate 95.80 67.16 41.85 26.87 28.24 61.90

- • Head-wise Scale Modulation: Building upon a standard normalization step (e.g., RMSNorm) applied to each branch independently, this variant introduces learnable scalar coeﬀicients (see Equation 14). This allows the model to statically adjust the relative contribution magnitude of each head type during training.
- • Head-wise Gated Competition: This variant replaces static scaling with a dynamic gating mechanism. A lightweight gate branch takes the original input sequence representation x as input and produces head-wise modulation weights via a softmax activation, scaled by the total number of heads H to preserve the overall feature magnitude. The final fusion is performed just like headwise scale modulation. By enforcing a sum-to-one constraint across the FA and GDN components for each head position, this design introduces explicit competition, enabling context-aware selection between FA and GDN.

Results and Analysis. We present the comprehensive experimental results in Table 5. Our analysis focuses on two key aspects: the necessity of feature normalization, and the trade-offs between static scaling and dynamic gating.

Impact of Feature Normalization. The results reveal a consistent and substantial degradation across most evaluation dimensions. On RULER Single-key retrieval, removing normalization leads to a performance drop exceeding 10% at both the original and extended context lengths, indicating that the feature distributions from heterogeneous attention heads are suﬀiciently misaligned to severely impair retrieval fidelity when naively combined. On challenging general reasoning benchmarks, a moderate but consistent degradation of approximately 1% is observed. The sole exception is Multi-Key retrieval at the original context length, where direct concatenation yields a 3.55% improvement—a counterintuitive result that warrants further investigation. One plausible explanation is that in the Multi-Key setting, the model must attend to multiple dispersed positions simultaneously, and the unnormalized feature mixing may inadvertently amplify certain cross-head interactions that facilitate multi-location tracking, at the cost of degraded single-location precision. Nevertheless, given the overwhelming degradation across all other metrics, this isolated gain does not justify removing normalization. We conclude that feature normalization is a critical design component for ensuring robust and universal performance across diverse head allocation strategies and evaluation scenarios.

Scale Modulation vs. Gated Competition. We compare the two advanced fusion variants, Head-wise Scale Modulation and Head-wise Gated Competition, across the full evaluation suite. The results reveal a clear and consistent advantage for the Scale Modulation variant: it outperforms Gated Competition on nearly all metrics, with the sole exception of Multi-Key retrieval at the original context length. The most striking gain is observed in long-context extension scenarios, where Scale Modulation achieves a 20% improvement over Gated Competition on Single-key retrieval tasks, demonstrating that static, learnable coeﬀicients provide substantially more stable representations for modeling long-range dependencies under length extrapolation. Furthermore, compared to the Basic model, Scale Modulation delivers consistent improvements across all extended-context evaluations, confirming that the head-wise scaling mechanism generalizes robustly beyond the training distribution. The only setting where Gated Competition holds an edge—Multi-Key at the original context length—suggests that dynamic, context-aware gating may offer localized benefits when multiple retrieval targets are densely distributed within the training-length regime, but this advantage is not preserved under length expansion. Given the overwhelming and consistent superiority of Scale Modulation across both general reasoning and long-context benchmarks, we adopt Head-wise Scale Modulation as the Default Model for all subsequent experiments.

Feature Distribution Differences. Figure 5 reports the per-head feature RMS statistics of the FA and GDN branches prior to head-wise fusion. In shallow layers (0–17), the two branches exhibit comparable

[Figure 4]

- Figure 5: Per-head feature RMS statistics of FA and GDN branches before normalization (Default Model). (a) Layer-wise RMS comparison with min–max range. (b) Divergence measured by ∆ RMS and ratio.

magnitudes (GDN/FA ratio within [0.2,1.5]), partly because shallow layers are dominated by GDN heads (Figure 6(e)), leaving few FA heads as a contrasting reference. In deeper layers (18–27), however, a pronounced scale mismatch emerges, with GDN RMS reaching up to 6.2× that of FA at layer 27. This divergence stems from the norm-awareness asymmetry between the two paradigms: softmax in FA constrains output magnitudes via exponential normalization, whereas linear attention in GDN lacks such regularization. The observed mismatch motivates independent RMSNorm for each branch before fusion, preventing either branch from dominating the combined representation due to scale disparities alone.

5.5 Effectiveness of Interpretability Screening

To validate the hypothesis that functional heterogeneity among attention heads drives the eﬀicacy of hybrid architectures, we investigate the impact of head selection strategies. This section aims to demonstrate that interpretability-guided selection, which prioritizes heads responsible for precise retrieval and factual recall, significantly outperforms naive distribution strategies. By aligning the architectural inductive bias with the intrinsic functional specialization of pre-trained heads, we aim to maximize the synergy between the high-precision FA branch and the eﬀicient GDN branch.

Experimental Setup. We compare five distinct head allocation strategies, all maintaining a global fixed ratio of 3:1 between GDN and FA heads to ensure comparable computational budgets. The baselines include:

- • (a) Fixed Allocation, which uniformly assigns a fixed fraction of heads to FA in every layer.
- • (b) Layer-wise Random Selection, which randomly selects FA heads within each layer.
- • (c) Global Random Selection, which randomly selects FA heads across the entire network.

As our proposed methods, we evaluate two interpretability-driven variants based on causal importance scores:

- • (d) Layer-wise Interpretability Screening, which selects the top-ranked heads within each layer according to their local importance.
- • (e) Global Interpretability Screening, which selects the most critical heads across all layers regardless of their positional distribution.

- Figure 6 visualizes the resulting head distributions for these five strategies, highlighting the non-uniform, layer-skewed patterns emerging from the global interpretability-based approach.

Results and Analysis. The comparative results across the five selection strategies are summarized in Table 6. We have the following findings:

Baseline Stability and Global Random Degradation. We observe that fixed allocation and layer-wise random selection yield nearly identical performance across all evaluation dimensions. This suggests that a uniform distribution of FA heads provides a stable baseline, and random variations within layers do not significantly alter the model’s functional capacity. In contrast, global random selection leads to noticeable performance degradation, whether in single-key or multi-key retrieval tasks. This indicates that disrupting the layer-specific structural balance without functional guidance harms the model’s ability to coordinate information flow across depths.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

#### (a) (b) (c) (d) (e)

- Figure 6: Comparison of distribution results of 5 head screening strategies: (a) Fixed Allocation; (b) Layerwise Random Selection; (c) Global Random Selection; (d) Layer-wise Interpretability Screening; (e) Global Interpretability Screening.

- Table 6: Comparison of different head selection strategies. All experiments are conducted based on Basic Model.

RULER Single RULER Multi-Key General Reasoning (Native) (Extended) (Native) (Extended) (Hard) (Easy)

Model

- (a) Fixed 85.63 62.62 27.35 13.67 28.65 62.59
- (b) Layer-Rand 87.07 65.33 27.70 17.37 27.91 62.95
- (c) Global-Rand 59.40 32.96 16.55 5.33 26.19 62.40

- (d) Layer-Interp 91.13 65.22 35.10 15.97 31.93 62.15
- (e) Global-Interp 98.70 81.73 38.35 25.93 31.70 62.68

Eﬀicacy of Layer-wise Interpretability Screening. Layer-wise interpretability screening demonstrates significant gains across almost all evaluation benchmarks, validating our hypothesis that preserving functionally critical heads (e.g., retrieval heads) in the FA branch is more effective than naive uniform allocation. By prioritizing heads with higher causal importance within each layer, this strategy ensures that local computational resources are allocated to the most impactful attention patterns.

Unlocking Potential via Global Interpretability Screening. Global interpretability screening achieves the best overall performance, further unlocking the model’s intrinsic potential. By selecting the most causally important heads across the entire network, this strategy allows for a non-uniform, optimized distribution where FA resources are concentrated in layers that contribute most to reasoning and recall. As visualized in Figure 6, this global approach results in a sparse but highly effective allocation pattern, proving that functional heterogeneity is a key driver for eﬀicient hybrid attention design.

Why interpretability outperforms naive strategies. The advantage of interpretability-guided selection has a structural explanation. Prior work has established that attention heads specialize in distinct functional roles [24] and vary dramatically in importance—many can be pruned with minimal loss, while a few are indispensable [25]. Our localization results (Section 5.7) confirm this for long-context retrieval: only ≈6.5% of heads are causally critical, and they are scattered across layers rather than concentrated in a few. Fixed and random allocation strategies are blind to this structure: they either waste FA budget on heads that do not need it (fixed assigns FA uniformly, regardless of function) or risk converting away critical heads entirely (global random can leave entire layers without FA, as the performance collapse in Table 6(c) shows). Interpretability screening, by contrast, allocates every FA slot to a head that demonstrably carries retrieval signal, which is why it dominates at all ratios—and increasingly so at aggressive ratios (Table 7), where the margin for wasted budget shrinks.

##### 5.6 Impact of Higher Hybrid Ratios

We investigate the performance trends under higher GDN-to-FA mixing ratios to further explore the eﬀiciency potential of our hybrid architecture. While a 3:1 ratio provides a balanced trade-off, increasing the proportion of GDN can significantly reduce memory footprint and computational overhead.

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

3:1 7:1 9:1

- Figure 7: Comparison of head selection distributions under varying GDN-to-FA ratios. The top row illustrates the distribution obtained by direct screening based on head interpretability scores. The bottom row shows the distribution with an additional constraint that ensures at least one FA head is retained in each layer, while still prioritizing high-scoring heads.

Experimental Setup. We evaluate three mixing ratios, extending from the baseline 3:1 to more aggressive configurations of 7:1 and 9:1. For each ratio, we compare two head selection strategies:

- • Global Interpretability Screening: Heads are selected strictly based on their global causal importance scores, regardless of their layer distribution. This serves as the baseline to test the raw eﬀicacy of interpretability-guided selection.
- • Constrained Global Screening: Recognizing that strict global ranking at high sparsity levels may concentrate FA heads in only a few layers—leaving most layers with pure GDN and potentially disrupting hierarchical information flow—we introduce a structural constraint. This strategy ensures that at least one FA head is retained in every layer, while the remaining FA budget is allocated to the highest-scoring heads globally.

This comparison allows us to disentangle the benefits of functional importance from the necessity of structural coverage across network depth. Figure 7 visualizes the resulting head distributions under both strategies. As the GDN-to-FA ratio increases, the proportion of layers composed entirely of GDN heads rises steadily, reaching nearly 50% in the extreme 9:1 configuration.

Results and Analysis. Table 7 summarizes the performance results across different hybrid ratios. We can summarize the findings as follows:

Performance Degradation and Structural Resilience. As the GDN-to-FA ratio increases, we observe a consistent decline in general reasoning capabilities, with complex reasoning tasks being particularly sensitive to the reduction of FA heads. In contrast, long-context performance exhibits a fluctuating downward trend. This divergence suggests that while LA can effectively handle local dependencies and simple retrieval, the loss of global receptive fields in FA heads disproportionately impacts multi-step logical deduction. The Global-Interp-C strategy, which ensures at least one FA head per layer, demonstrates significant advantages at higher sparsity levels (7:1 and 9:1). By preventing the complete collapse of FA capacity in certain layers, this constraint maintains the hierarchical information flow essential for deep networks. At lower ratios (3:1), while its impact on long-context tasks varies, it consistently yields improvements in general reasoning, indicating that structural diversity across layers is beneficial even when the overall FA budget is suﬀicient.

- Table 7: Performance comparison under varying GDN-to-FA ratios. We evaluate two selection strategies: Global-Interp (direct screening by interpretability scores) and Global-Interp-C (with a constraint of at least one FA head per layer). All experiments are conducted based on Basic Model.

Model Ratio

RULER Single RULER Multi-Key General Reasoning (Native) (Extended) (Native) (Extended) (Hard) (Easy)

Global-Interp 3:1 98.70 81.73 38.35 25.93 31.70 62.68 Global-Interp-C 3:1 96.27 83.91 36.45 25.20 31.83 62.73

Global-Interp 7:1 81.40 58.07 17.90 18.67 27.75 62.63 Global-Interp-C 7:1 88.70 81.04 26.70 22.93 29.46 62.65

Global-Interp 9:1 89.12 60.58 18.65 14.50 26.04 62.65 Global-Interp-C 9:1 90.10 72.69 21.05 18.63 28.92 62.68

- Table 8: Comparison of layer-wise hybrids with a 3:1 LA:FA ratio (marked with ∗ in Table 2) against our Basic Model with a 7:1 ratio. Our model performs Global-Interp-C head selection strategy.

General Reasoning 16K 32K 64K 128K 256K (Hard) (Easy)

RULER Single & Multi-Key

Model Ratio

Avg

Layer-wise hybrid 3:1 60.52 53.40 60.79 53.59 49.69 55.59 19.80 59.72 Global-Interp-C 7:1 59.55 55.85 57.35 51.04 47.59 54.27 29.46 62.65

Sparsity Limits and Practical Effectiveness. A direct comparison between the constrained 9:1 configuration and the standard 3:1 baseline reveals the current limitations of our hybrid architecture under extreme sparsity. While the model retains over 80% of its performance on simple long-context tasks, it suffers from significant degradation in more complex scenarios. Specifically, accuracy drops to approximately 60% in multi-key retrieval settings, and general reasoning capabilities on challenging benchmarks decline by about 3%. These results indicate that while the layer-wise constraint effectively prevents structural collapse, the severe reduction in FA heads still compromises the model’s ability to handle intricate dependencies and complex logical patterns, suggesting that further optimization is required to bridge the performance gap in extreme sparse regimes. Despite these limitations at extreme sparsity, our model retains a substantial advantage over layer-wise hybrids in practical regimes. At a 7:1 ratio, we achieve long-context performance on RULER (16K–256K) broadly comparable to the layer-wise hybrid at 3:1, with per-length differences within ±3% (Table 8). Meanwhile, it demonstrates a substantial advantage in general reasoning (+9.66% on Hard, +2.93% on Easy), underscoring the effectiveness of head-wise hybridization in preserving strong general-domain capabilities while enabling aggressive LA:FA ratios for long-context extension.

##### 5.7 Anatomy of Interpretability-Guided Head Selection

Tables 6 and 7 establish that interpretability-guided screening works; this section explains why. We analyze the head-importance scores of Equation (12) computed on the pretrained dense Qwen3-1.7B checkpoint (28 layers × 16 heads = 448 query heads) using the long-context retrieval probe, and show that (i) the score is a stable causal measurement, (ii) retrieval is localized to a sparse set of heads that does not respect layer boundaries—the empirical foundation of head-level hybridization—and (iii) the score is causally faithful under knockout. A complementary analysis—the representational diversity of heads within a layer—is deferred to Appendix C.2.

Experimental Setup. We instantiate the procedure of Section 4.1 with long-context retrieval as the calibration capability, realized by five Needle-in-a-Haystack (NIAH) sub-probes from RULER (three singlekey, two multi-key) that combine activation patching (receivers) with iterative path patching (senders). We write |drop| for the per-head normalized importance IEl,h of Equation (10) and, following common practice, call a head critical when |drop| ≥ 0.01. Unless noted, importance is taken as the union over the five

Sample Stability Analysis: Correlation vs Sample Count

###### Top-64 Head Set Overlap vs Context Length (Reference: Length 16384)

###### Head Importance Correlation vs Context Length (Reference: Length 16384)

- 1.0

1.000

1.0

1.000

1.0

JaccardSimilaritywithLongestLength(top-64)

0.886

0.886

SpearmanCorrelationwithLongestLength

0.844

SpearmanCorrelationwithFinalResult

0.826

0.9

0.804

0.918

0.787

0.912

0.765

0.8

0.9

0.725

0.878

0.855

0.8

0.800

0.6

0.8

0.765

0.7

0.4

0.689

0.7

0.680

0.6

0.2

0.6

Jaccard to Length 16384 (top-64)

Correlation to Length 16384

Correlation to Final

High (0.7)

High (0.9)

Threshold (0.9)

Moderate (0.4)

Moderate (0.7)

Recommended (6 samples)

Reference (16384)

Reference (16384)

0.5

0.0

0.5

1 2 3 4 6 8 10 12 16 20 24 32

256 512 1024 2048 3072 4096 8192 12288 16384

256 512 1024 2048 3072 4096 8192 12288 16384

Number of Samples

Context Length

Context Length

- Figure 8: Stability and robustness of the head-importance ranking on Qwen3-1.7B. (Left) Sample stability: Spearman correlation between the rank from k calibration samples and the 32-sample reference (crosses the 0.9 threshold at k = 6, ρ ≈ 0.921, and saturates by k = 12). (Middle, Right) Robustness to the localization context length, measured against the 16k reference: top-32 Jaccard overlap of the critical-head set, and Spearman rank correlation. The head set is well preserved across both samples and lengths, while exact ranks drift at intermediate lengths—justifying localization from a few short-context samples and transfer to long context.

sub-probes, computed on a small calibration set.3 General ability is the second capability fused under Equation (12); its localization protocol is analogous and omitted for brevity.

The importance score is a stable measurement. A selection rule is only as trustworthy as the measurement under it. Figure 8 (left) reports the Spearman rank correlation between the per-head ranking computed from a growing number of calibration samples and a larger-sample reference: the ranking stabilizes after only a handful of samples and is essentially fixed thereafter, the empirical basis for the “a few forward passes” claim in Section 4.1. The selection is also robust to the context length at which it is computed (Figure 8, middle and right): over a wide range of context lengths, the top-ranked set is well preserved and converges toward the long-context reference, even though absolute ranks drift at intermediate lengths. Across the sub-probes, the critical-head sets overlap substantially, with number-needle probes agreeing most strongly and uuid-needle probes diverging; we therefore take the union across sub-probes, which is the conservative choice. The selection is thus stable at the level of the head set, which is what we rely on, while individual ranks are noisier—consistent with the redundancy structure discussed in Appendix C.

Retrieval is head-localized, not layer-localized. This is the central observation behind head-level hybridization. If long-context retrieval were a layer-level property—the implicit assumption of layer-wise hybrids—then within any critical layer all 16 heads would carry comparable importance, and a clean classification of heads into long-context vs. replaceable would make each layer pure. Neither holds. The per-layer Gini coeﬀicient of head importance averages 0.622 (ranging 0.399 to 0.915): importance is concentrated in a few heads even within the most critical layers. Classifying heads by their average importance over context lengths yields only 29 long-context-critical heads (|drop| ≥ 0.01) against 407 replaceable ones (< 0.005)—i.e., only ≈ 6.5% of the 448 heads are essential for retrieval, while ≈ 90.8% are safely convertible to GDN. Crucially, these critical heads are scattered: the head-specialization map (Figure 9a) shows that almost every layer is a mixture of critical and replaceable heads rather than a single role, with ten layers each containing both a long-context-critical head and several replaceable ones (enumerated in Table 16). Any per-layer assignment must therefore either waste FA on unimportant heads or convert away critical ones; only head-level allocation can separate them. This causal, importance-based argument—not representational fineness—is why the head is the right axis for hybridization.

Causal faithfulness under knockout. The importance score is a soft logit-difference measurement; we validate it against the hard downstream behavior. We knock out heads in decreasing order of estimated importance (ranked by −drop) by replacing their output with the corrupted-run activation during prefill, then greedily generate and score needle-retrieval accuracy; as controls we knock out the same number of randomly chosen heads over several random runs. As shown in Figure 9b, removing only the top few importance-ranked heads collapses retrieval accuracy from near-perfect to a small fraction, and it falls further toward zero as more are removed, whereas the random controls remain far higher at every budget and near-perfect over the first several knockouts. This confirms that the score identifies the heads that causally carry retrieval, not an arbitrary subset.

3In our experiments, the calibration set consists of a few samples (eight) localized at short context and scored in fp16; the per-head ranking is already stable from roughly six samples (see the stability analysis below and Appendix C.1).

###### Head specialization map (+ = increasing with length, - = decreasing) high_thr=0.01, low_thr=0.005

###### Knockout Degradation Curve (total heads = 448, merged eval N = 128)

[Figure 16]

+ + + - - +

long-context borderline non-long-context

1.0

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

- 17

- 18

- 19

- 20

- 21

- 22

- 23

- 24

- 25

- 26

- 27

| |
|---|

- - - - - - - - - -
- - - - - + - - +
- - - - - - - + - - +

+ - + + - - - + - - - - - - - - -

- - - - - + - - - - +

- - - - - - + -

- - - - - - + - + -

- - - - - + - + - -

- - - - +

- - - + + - + + + -

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.8

NeedleRetrievalAccuracy

| |
|---|

0.6

Important Heads (ranked by -drop)

+ + - + - - - + - + + + - - - + + - - + + - - - - - - - - - + - + - + -

Random Heads (mean +/- std, n=9)

Layer

- - - + - - + - + + + - - -

- - - - + - - -

- - - - + + + - - - - - - - -

+ - + - + - + - + - - - - +

- - + - + - - + + - + - - -
- - + - + + - - - + - - + +
- - - - + - - - + - - - +

+ - + + + - + - + + + - -

- - + - - + + + + + - - + +
- - + - + + - - - + - + +

0.4

0.2

+ + + + - + + - - + + + - -

1% 5% 10%

0.0

+ + - + + + - + - + + - + - +

0 10 20 30 40 50 60

+ - - - - - - + + + + - +

Number of Heads Knocked Out

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

Head

(a)

(b)

- Figure 9: Head-level localization and its causal validation on Qwen3-1.7B (448 heads). (a) Headspecialization map: each cell is a head, colored by role, with +/− markers where importance increases/decreases with context length. Retrieval importance is localized to scattered heads, not whole layers. (b) Knocking faithfulness: ablating heads by estimated importance (−drop) collapses needle-retrieval accuracy after only the top few heads (≈ 1%), whereas random controls (mean ± std) stay far higher at every budget.

Table 9: Optimized training configuration of three training stages. Stage Batch Size Context Length Training Steps (Tokens) LR scheduler Learning Rate

- 1 128 2048 3052 (0.8B) Cosine 1e-3 → 1e-5

- 2 256 2048 7630 (4.0B) Cosine 1e-4 → 1e-5

- 3 128 16384 477 (1.0B) Constant 1e-5

Together, these analyses explain the gains in Tables 6 and 7: retrieval lives in a sparse, scattered, causally verifiable set of heads that no per-layer assignment can isolate, and interpretability-guided screening is precisely the tool that recovers it.

##### 5.8 Training configuration optimization

To maximize the potential of our head-wise hybrid architecture, we conducted a systematic optimization of the training configurations across the three-stage transfer learning pipeline.

Experimental Setup. While maintaining FineWeb-Edu as the core pre-training corpus, we focused on refining the data scale and hyperparameter schedules to enhance both convergence stability and final model performance. We conduct experiments based on the Default Model as denoted before.

Results and Analysis. We optimize the training of the hybrid model transfer learning from multiple dimensions, including data scale and training configuration. The optimized configuration of three training stages is summarized in Table 9. Compared to the original setup, we observe that scaling up the training data for Stage 1 and Stage 2 significantly enhances the eﬀicacy of knowledge transfer. Furthermore, we increase both the training batch size and the sequence length to better accommodate long-context dependencies and stabilize the optimization dynamics across the heterogeneous attention branches. Table 10 presents a detailed comparison of model performance before and after the training configuration optimization. The results highlight two key observations:

Significant Performance Gains for Our Hybrid Model. Under the optimized configuration, our proposed head-wise hybrid architecture demonstrates consistent improvements across all six evaluation benchmarks. Notably, the performance gains are most pronounced in challenging scenarios, such as RULER Multi-key retrieval and complex general reasoning tasks. For instance, accuracy on RULER Multi-key increases by 13.90% at the native context length, indicating a substantial improvement in handling intricate long-context dependencies. This robust improvement underscores the eﬀicacy of our head-wise selection

- Table 10: Comparison of training configuration optimization. † denotes that the results are reproduced by ourselves. New-Config indicates the optimized configuration, which is summarized in Table 9.

RULER Single RULER Multi-Key General Reasoning (Native) (Extended) (Native) (Extended) (Hard) (Easy)

Model

HypeNet† 89.07 85.00 24.85 24.37 19.80 59.72 + New-Config 86.57 76.62 33.85 25.30 23.71 60.87

Default Model 98.47 87.49 37.10 27.37 31.03 62.12 + New-Config 99.40 89.84 51.00 35.13 34.91 63.14

strategy in capturing fine-grained semantic interactions, validating that scaling up data in the alignment and distillation stages effectively empowers the model to internalize heterogeneous attention patterns.

Divergent scaling behavior of HypeNet. In contrast, HypeNet exhibits marginal improvements under the optimized configuration. While it achieves moderate gains exceeding 4.00% on Multi-Key retrieval within the native context length and on challenging general reasoning benchmarks, these improvements are overshadowed by a pronounced degradation on Single-key retrieval tasks, where performance drops by

- 2.50% to 8.38%. This asymmetry suggests that head-wise hybridization has a higher capacity for absorbing additional training data: the finer-grained FA/GDN assignment allows the model to better exploit the representational complementarity between branches, whereas HypeNet’s layer-wise mixing offers less room for the model to redistribute learned representations across attention types.

5.9 Scaling Up with More Training Tokens

We further probe the potential of our hybrid models by scaling up the carefully filtered high-quality training data to over 15B tokens and benchmarking against a range of open-source models around 2B parametersincluding standard Transformers and other hybrid architectures.

Experimental Setup. For fair and rigorous comparison, all baseline models utilized in our evaluation are the oﬀicial releases provided by their respective authors. We conduct all experiments within a unified evaluation framework, employing identical preprocessing pipelines and hyperparameter settings to eliminate implementation biases and guarantee the reproducibility of our results. To assess long-context retrieval capabilities across varying sequence lengths, we evaluate models on the NIAH benchmark, reporting the averaged results of both single-key and multi-key retrieval tasks over context lengths spanning from 4K to 256K tokens. For general reasoning, we deliberately select the hard subset to better expose performance differences among models. We benchmark against two categories of state-of-the-art architectures: (1) Standard Transformer-based models, including Qwen3-1.7B and MiniCPM-1B-Base [74]; and (2) Hybrid Architecture models, such as Gemma-3n-E2B [75], HypeNet, Hymba-1.5B [46], and Jet-Nemotron-2B [76].

Results and Analysis. We present quantitative results on long-context retrieval (Table 11) and general reasoning (Table 12), and summarize our findings in two aspects below.

Superior Long-Context Extrapolation over Standard Transformers. Compared to Qwen3-1.7B, HydraHead demonstrates strong extrapolation capability beyond 64K tokens. At 256K context length, it surpasses the YaRN-enhanced variant by +54.23% on RULER Single and +38.50% on RULER Multi-Key. Within the native 32K range, HydraHead remains competitive with the base model and even exceeds the YaRN version on certain tasks. On the general reasoning benchmark, HydraHead trails Qwen3-1.7B by only

- 3.40 percentage points on average—a marginal gap given that our model was adapted via merely 15B tokens of transfer training rather than full-scale pretraining. Moreover, it matches the YaRN-enhanced variant, indicating that the hybrid architecture itself does not compromise general-purpose capabilities.

Robustness Advantage over Existing Hybrid Architectures. While most hybrid models degrade to near-zero performance at 256K, HydraHead sustains 94.53% (Single) and 52.70% (Multi-Key). On the general benchmark, HydraHead outperforms all competing hybrid models by at least 10 percentage points on average, with one exception: Jet-Nemotron-2B achieves a higher average score (+9.69 points). However,

4Base model result; the original paper reports 58.76% for the instruct-tuned variant. The score is lower than expected and may warrant further investigation.

- Table 11: Comparison with SOTA models on long-context retrieval benchmarks across context lengths ranging from 4K to 256K.

Model Type 4K 8K 16K 32K 64K 128K 256K RULER Single

Qwen3-1.7B FA 100.00 99.87 99.93 99.93 98.13 0.00 0.00 Qwen3-1.7B-yarn FA 100.00 99.93 98.93 97.73 98.80 83.47 40.20 MiniCPM5-1B-Base FA 99.93 99.47 98.60 97.60 90.93 93.73 59.53 Gemma-3n-E2B Hybrid 99.93 99.87 99.67 99.67 22.23 0.00 0.00 HypeNet-2B Hybrid 93.60 94.53 86.33 84.33 83.93 78.80 68.93 Hymba-1.5B Hybrid 95.60 88.20 25.60 0.07 0.00 0.00 0.00 Jet-Nemotron-2B Hybrid 100.00 99.67 99.47 92.80 27.27 12.60 1.07 HydraHead Hybrid 100.00 99.93 100.00 99.80 99.80 99.47 94.53

RULER Multi-Key

Qwen3-1.7B FA 99.50 99.40 97.80 94.10 42.20 0.00 0.00 Qwen3-1.7B-yarn FA 97.40 91.80 54.30 46.10 47.80 41.50 14.20 MiniCPM5-1B-Base FA 98.10 96.60 89.90 87.20 76.20 54.40 16.60 Gemma-3n-E2B Hybrid 98.70 98.50 89.70 76.30 11.70 0.00 0.00 HypeNet-2B Hybrid 30.50 28.90 25.00 13.90 27.40 16.00 16.90 Hymba-1.5B Hybrid 44.10 33.40 12.50 0.00 0.00 0.00 0.00 Jet-Nemotron-2B Hybrid 95.70 89.90 69.00 42.00 8.80 1.90 0.00 HydraHead Hybrid 98.50 96.40 94.30 92.50 82.00 68.80 52.70

Table 12: Comparison with SOTA models on the general benchmark (Hard). Model Type MMLU BBH MBPP GSM8k Average

Qwen3-1.7B FA 55.46 48.16 43.00 69.45 54.02 Qwen3-1.7B-yarn FA 53.45 45.65 42.40 59.97 50.37 MiniCPM5-1B-Base FA 39.30 37.81 33.20 34.12 36.11 Gemma-3n-E2B Hybrid 53.82 42.47 35.00 25.85 39.29 HypeNet-2B Hybrid 46.52 9.15 4.80 4.93 16.35 Hymba-1.5B Hybrid 49.67 8.89 16.60 0.994 19.04 Jet-Nemotron-2B Hybrid 53.59 58.16 54.20 75.28 60.31 HydraHead Hybrid 55.80 46.64 47.20 52.84 50.62

this advantage comes at a steep cost to long-context ability—Jet-Nemotron-2B falls behind HydraHead by more than 50 percentage points at context lengths ≥64K. This stark contrast validates that our fine-grained head-wise allocation strategy not only preserves strong general reasoning abilities but also ensures superior scalability and reliability in long-context scenarios, offering a more balanced solution than current state-ofthe-art hybrid models.

### 6 Conclusion

In this report, we present a head-wise hybrid attention architecture that converts a pre-trained Transformer into a hybrid model by organically integrating FA and LA at the head level. A head-wise scale-normalized mechanism is designed to effectively fuse the heterogeneous features from different attention mechanisms. Guided by interpretability analysis, we identify and allocate critical heads to the FA branch to preserve high-fidelity representations, while routing the remaining heads to the LA branch for eﬀicient long-context extension. To enable effective transfer learning, we employ a three-stage training pipeline that combines parameter reuse initialization with distillation.

In controlled experiments under identical training configurations, our head-wise hybrid architecture achieves over 10% improvement on challenging general reasoning benchmarks compared to layer-wise hybrid alternatives, while also consistently outperforming them across long-context evaluations. Scaling the training

to over 15B tokens, our model comprehensively surpasses the baseline, demonstrating strong potential for further scaling.

Several directions remain open for future work, including deeper exploration of interpretability-driven head allocation strategies, scaling studies across larger model and data sizes, and the integration of additional attention variants into the hybrid architecture. We hope these findings offer valuable insights to both academia and industry on the design and deployment of eﬀicient hybrid attention models.

### References

- [1] GLM-5-Team, :, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chengwei Hu, Chenhui Zhang, Dan Zhang, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huanpeng Chu, Jia’ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li, Jingwei Yuan, Jinhua Du, Jinxin Liu, Junkai Zhi, Junwen Duan, Kaiyue Zhou, Kangjian Wei, Ke Wang, Keyun Luo, Laiqiang Zhang, Leigang Sha, Liang Xu, Lindong Wu, Lintao Ding, Lu Chen, Minghao Li, Nianyi Lin, Pan Ta, Qiang Zou, Rongjun Song, Ruiqi Yang, Shangqing Tu, Shangtong Yang, Shaoxiang Wu, Shengyan Zhang, Shijie Li, Shuang Li, Shuyi Fan, Wei Qin, Wei Tian, Weining Zhang, Wenbo Yu, Wenjie Liang, Xiang Kuang, Xiangmeng Cheng, Xiangyang Li, Xiaoquan Yan, Xiaowei Hu, Xiaoying Ling, Xing Fan, Xingye Xia, Xinyuan Zhang, Xinze Zhang, Xirui Pan, Xu Zou, Xunkai Zhang, Yadi Liu, Yandong Wu, Yanfu Li, Yidong Wang, Yifan Zhu, Yijun Tan, Yilin Zhou, Yiming Pan, Ying Zhang, Yinpei Su, Yipeng Geng, Yong Yan, Yonglin Tan, Yuean Bi, Yuhan Shen, Yuhao Yang, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yurong Wu, Yutao Zhang, Yuxi Duan, Yuxuan Zhang, Zezhen Liu, Zhengtao Jiang, Zhenhe Yan, Zheyu Zhang, Zhixiang Wei, Zhuo Chen, Zhuoer Feng, Zijun Yao, Ziwei Chai, Ziyuan Wang, Zuzhou Zhang, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-5: from vibe coding to agentic engineering, 2026.
- [2] Kimi Team, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Chenxiao Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Qizheng Gu, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yang Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Haoyu Lu, Lijun Lu, Yashuo Luo, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Zeyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Lin Sui, Xinjie Sun, Flood Sung, Yunpeng Tai, Heyi Tang, Jiawen Tao, Qifeng Teng, Chaoran Tian, Chensi Wang, Dinglu Wang, Feng Wang, Hailong Wang, Haiming Wang, Jianzhou Wang, Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Si Wang, Xinyuan Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Haoning Wu, Wenhao Wu, Xingzhe Wu, Yuxin Wu, Chenjun Xiao, Jin Xie, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Jing Xu, Jing Xu, Junjie Yan, Yuzi Yan, Hao Yang, Xiaofei Yang, Yi Yang, Ying Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie Yuan, Siyu Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang, Yadong Zhang, Yangkun Zhang, Yichi Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang,

- Zheng Zhang, Haotian Zhao, Yikai Zhao, Zijia Zhao, Huabin Zheng, Shaojie Zheng, Longguang Zhong, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Jinguo Zhu, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi k2: Open agentic intelligence, 2026.
- [3] Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog, April 2026. Oﬀicial Blog Post.
- [4] DeepSeek-AI. Deepseek-v4: Towards highly eﬀicient million-token context intelligence. https: //huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf, April 2026. Technical Report.
- [5] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
- [6] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and eﬀicient algorithms through structured state space duality. In International Conference on Machine Learning (ICML), 2024.
- [7] Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, and Albert Gu. Mamba-3: Improved sequence modeling using state space principles, 2026.
- [8] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers, 2021.
- [9] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations, 2025.
- [10] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, Enzhe Lu, Weizhou Liu, Yanru Chen, Weixin Xu, Longhui Yu, Yejie Wang, Yu Fan, Longguang Zhong, Enming Yuan, Dehao Zhang, Yizhi Zhang, T. Y. Liu, Haiming Wang, Shengjun Fang, Weiran He, Shaowei Liu, Yiwei Li, Jianlin Su, Jiezhong Qiu, Bo Pang, Junjie Yan, Zhejun Jiang, Weixiao Huang, Bohong Yin, Jiacheng You, Chu Wei, Zhengtao Wang, Chao Hong, Yutian Chen, Guanduo Chen, Yucheng Wang, Huabin Zheng, Feng Wang, Yibo Liu, Mengnan Dong, Zheng Zhang, Siyuan Pan, Wenhao Wu, Yuhao Wu, Longyu Guan, Jiawen Tao, Guohong Fu, Xinran Xu, Yuzhi Wang, Guokun Lai, Yuxin Wu, Xinyu Zhou, Zhilin Yang, and Yulun Du. Kimi linear: An expressive, eﬀicient attention architecture, 2025.
- [11] Ali Hatamizadeh, Yejin Choi, and Jan Kautz. Gated deltanet-2: Decoupling erase and write in linear attention, 2026.
- [12] MiniMax, :, Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, Chengjun Xiao, Chengyu Du, Chi Zhang, Chu Qiao, Chunhao Zhang, Chunhui Du, Congchao Guo, Da Chen, Deming Ding, Dianjun Sun, Dong Li, Enwei Jiao, Haigang Zhou, Haimo Zhang, Han Ding, Haohai Sun, Haoyu Feng, Huaiguang Cai, Haichao Zhu, Jian Sun, Jiaqi Zhuang, Jiaren Cai, Jiayuan Song, Jin Zhu, Jingyang Li, Jinhao Tian, Jinli Liu, Junhao Xu, Junjie Yan, Junteng Liu, Junxian He, Kaiyi Feng, Ke Yang, Kecheng Xiao, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Li, Lin Zheng, Linge Du, Lingyu Yang, Lunbin Zeng, Minghui Yu, Mingliang Tao, Mingyuan Chi, Mozhi Zhang, Mujie Lin, Nan Hu, Nongyu Di, Peng Gao, Pengfei Li, Pengyu Zhao, Qibing Ren, Qidi Xu, Qile Li, Qin Wang, Rong Tian, Ruitao Leng, Shaoxiang Chen, Shaoyu Chen, Shengmin Shi, Shitong Weng, Shuchang Guan, Shuqi Yu, Sichen Li, Songquan Zhu, Tengfei Li, Tianchi Cai, Tianrun Liang, Weiyu Cheng, Weize Kong, Wenkai Li, Xiancai Chen, Xiangjun Song, Xiao Luo, Xiao Su, Xiaobo Li, Xiaodong Han, Xinzhu Hou, Xuan Lu, Xun Zou, Xuyang Shen, Yan Gong, Yan Ma, Yang Wang, Yiqi Shi, Yiran Zhong, Yonghong Duan, Yongxiang Fu, Yongyi Hu, Yu Gao, Yuanxiang Fan, Yufeng Yang, Yuhao Li, Yulin Hu, Yunan Huang, Yunji Li, Yunzhi Xu, Yuxin Mao, Yuxuan Shi, Yuze Wenren, Zehan Li, Zelin Li, Zhanxu Tian, Zhengmao Zhu, Zhenhua Fan, Zhenzhen Wu, Zhichao Xu, Zhihang Yu, Zhiheng Lyu, Zhuo Jiang, Zibo Gao, Zijia Wu, Zijian Song, and Zijun Sun. Minimax-m1: Scaling test-time compute eﬀiciently with lightning attention, 2025.
- [13] Core Team, Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, Gang Xie, Hailin Zhang, Hanglong Lv, Hanyu Li, Heyu Chen, Hongshen Xu, Houbin Zhang, Huaqiu Liu, Jiangshan Duo, Jianyu Wei, Jiebao Xiao, Jinhao Dong, Jun Shi, Junhao Hu, Kainan Bao, Kang Zhou, Lei Li, Liang Zhao, Linghao Zhang, Peidian Li, Qianli Chen, Shaohui Liu, Shihua Yu, Shijie Cao, Shimao Chen, Shouqiu Yu, Shuo Liu, Tianling Zhou,

- Weijiang Su, Weikun Wang, Wenhan Ma, Xiangwei Deng, Bohan Mao, Bowen Ye, Can Cai, Chenghua Wang, Chengxuan Zhu, Chong Ma, Chun Chen, Chunan Li, Dawei Zhu, Deshan Xiao, Dong Zhang, Duo Zhang, Fangyue Liu, Feiyu Yang, Fengyuan Shi, Guoan Wang, Hao Tian, Hao Wu, Heng Qu, Hongfei Yi, Hongxu An, Hongyi Guan, Xing Zhang, Yifan Song, Yihan Yan, Yihao Zhao, Yingchun Lai, Yizhao Gao, Yu Cheng, Yuanyuan Tian, Yudong Wang, Zhen Tang, Zhengju Tang, Zhengtao Wen, Zhichao Song, Zhixian Zheng, Zihan Jiang, Jian Wen, Jiarui Sun, Jiawei Li, Jinlong Xue, Jun Xia, Kai Fang, Menghang Zhu, Nuo Chen, Qian Tu, Qihao Zhang, Qiying Wang, Rang Li, Rui Ma, Shaolei Zhang, Shengfan Wang, Shicheng Li, Shuhao Gu, Shuhuai Ren, Sirui Deng, Tao Guo, Tianyang Lu, Weiji Zhuang, Weikang Zhang, Weimin Xiong, Wenshan Huang, Wenyu Yang, Xin Zhang, Xing Yong, Xu Wang, Xueyang Xie, Yilin Jiang, Yixin Yang, Yongzhe He, Yu Tu, Yuanliang Dong, Yuchen Liu, Yue Ma, Yue Yu, Yuxing Xiang, Zhaojun Huang, Zhenru Lin, Zhipeng Xu, Zhiyang Chen, Zhonghua Deng, Zihan Zhang, and Zihao Yue. Mimo-v2-flash technical report, 2026.
- [14] MiniMax, :, Aili Chen, Aonian Li, Baichuan Zhou, Bangwei Gong, Binyang Jiang, Boji Dan, Changqing Yu, Chao Wang, Cheng Ma, Cheng Zhong, Cheng Zhu, Chengjun Xiao, Chengyi Yang, Chengyu Du, Chenyang Zhang, Chi Zhang, Chuangyi Huang, Chunhao Zhang, Chunhui Du, Chunyu Zhao, Congchao Guo, Da Chen, Deming Ding, Dianjun Sun, Dongyu Zhang, Enhui Yang, Fei Yu, Guang Zheng, Guodong Zheng, Guohong Li, Haichao Zhu, Haigang Zhou, Haimo Zhang, Han Ding, Hao Zhang, Haohai Sun, Haolin Lyu, Haonan Lu, Haoyu Wang, Huajie Shi, Huiyang Li, Jiacheng Chen, Jian Zhang, Jiaqi Zhuang, Jiaren Cai, Jiaxin Pan, Jiayao Li, Jiayuan Song, Jichuan Zhang, Jie Wang, Jihao Gu, Jin Zhu, Jingwei Dong, Jingyang Li, Jingyu Zhang, Jingze Zhuang, Jinhao Tian, Jinli Liu, Jinyi Hu, Jun Tao, Jun Zhang, Junbin Ruan, Junhao Xu, Junjie Yan, Junteng Liu, Junxian He, Kang Xu, Ke Ji, Ke Yang, Kecheng Xiao, Keyu Duan, Keyu Li, Le Han, Letian Ruan, Li Yuan, Lianfei Yu, Liheng Feng, Lijie Mo, Lin Li, Lingye Bao, Lingyu Yang, Lingyuan Zhou, Loki, Lu Chen, Lunbin Ceng, Ming Li, Ming Zhong, Mingliang Tao, Mingyuan Chi, Mujie Lin, Nan Hu, Ningxin Chen, Peiyin Zhu, Peng Gao, Pengcheng Gao, Pengfei Li, Penglin Li, Pengyu Zhao, Qibin Ren, Qidi Xu, Qihan Ren, Qile Li, Qin Wang, Quanliang Chen, Qunhong Ceng, Rong Tian, Rui Dong, Ruitao Leng, Ruize Zhang, Shanqi Liu, Shaoyu Chen, Sheng Jia, Shun Yao, Shuoran Zhao, Shuqi Yu, Sichen Li, Sicheng Pan, Songquan Zhu, Tengfei Li, Tian Xie, Tiancheng Qin, Tianrun Liang, Wei Liu, Weiqi Xu, Weitao Li, Weixiang Chen, Weiyu Cheng, Weiyu Zhang, Wenhu Chen, Wenqian Zhao, Xiancai Chen, Xiangjun Song, Xiangyuan Wang, Xiao Luo, Xiao Su, Xiaobo Li, Xiaodong Han, Xiaojie Wu, Xihao Song, Xingyi Han, Xinyu Guan, Xuan Lu, Xun Zou, Xunhao Lai, Xutong Li, Yan Gong, Yang Wang, Yang Xu, Yangsen Wang, Ye Tang, Yicheng Chen, Yinran Qiu, Yiqi Shi, Yiting Guo, Yiwen Huang, Yixuan Wang, Yongyi Hu, Yu Gao, Yu Zhang, Yuanxiang Ying, Yuanzhen Zhang, Yubo Wang, Yuchen Song, Yufeng Yang, Yuhang Meng, Yuhang Miao, Yuhao Li, Yujie Liu, Yulin Hu, Yunan Huang, Yunji Li, Yunyi Huang, Yusen Zhang, Yusu Hong, Yutao Xie, Yutong Zhang, Yuwen Liao, Yuxuan Shi, Yuze Wenren, Zebin Li, Zehan Li, Zejian Luo, Zeyu Jin, Zeyuan Sun, Zhanpeng Zhou, Zhaochen Su, Zhendong Li, Zhengmao Zhu, Zhengyuan Peng, Zhenhua Fan, Zhi Zhang, Zhichao Xu, Zhiheng Lv, Zhikang Xu, Zhitao He, Zhiwei He, Zhongyuan Li, Zibo Gao, Zijia Wu, Zijian Song, Zijian Zhou, Zijun Sun, Zishan Huang, Ziying Chen, and Ziyue Ge. The minimax-m2 series: Mini activations unleashing max real-world intelligence, 2026.
- [15] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tristan Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Scott Laskin, Scott Lee, Duane Lindner, Matt Maskey, Parker Nicholson, Karina Nguyen, Ethan Perez, Sam Ringer, Craig Sanderson, Toby Shevlane, Miles Tran, Lily Weng, Jeff Wu, Michael Ziegler, and Chris Olah. A mathematical framework for transformer circuits. https://transformer-circuits.pub/2021/framework/ index.html, 2021. Anthropic Research.
- [16] Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small, 2022.
- [17] Leo Gao, Achyuta Rajaram, Jacob Coxon, Soham V. Govande, Bowen Baker, and Dan Mossing. Weightsparse transformers have interpretable circuits. CoRR, abs/2511.13653, 2025.
- [18] Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, et al. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. arXiv preprint arXiv:2605.29358, 2026.
- [19] Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. In The Thirteenth International Conference on Learning Representations, ICLR 2025. OpenReview.net, 2025.

- [20] Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca D. Dragan, Rohin Shah, and Neel Nanda. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. CoRR, abs/2408.05147, 2024.
- [21] Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L Turner, and Joshua Batson. On the biology of a large language model. Transformer Circuits Thread, 2025.
- [22] Jack Lindsey. Emergent introspective awareness in large language models. CoRR, abs/2601.01828, 2026.
- [23] Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, Joshua Batson, Sam Zimmerman, Kelley Rivoire, Kyle Fish, Chris Olah, and Jack Lindsey. Emotion concepts and their function in a large language model. CoRR, abs/2604.07729, 2026.
- [24] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head selfattention: Specialized heads do the heavy lifting, the rest can be pruned, 2019.
- [25] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one?, 2019.
- [26] Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. What does bert look at? an analysis of bert’s attention, 2019.
- [27] Wei Zhang, Chaoqun Wan, Yonggang Zhang, Yiu ming Cheung, Xinmei Tian, Xu Shen, and Jieping Ye. Interpreting and improving large language models in arithmetic calculation, 2024.
- [28] Wei Chen, Zhen Huang, Liang Xie, Binbin Lin, Houqiang Li, Le Lu, Xinmei Tian, Deng Cai, Yonggang Zhang, Wenxiao Wang, Xu Shen, and Jieping Ye. From yes-men to truth-tellers: Addressing sycophancy in large language models with pinpoint tuning, 2025.
- [29] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 12216–12235. Association for Computational Linguistics, 2023.
- [30] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited, 2019.
- [31] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks?, 2022.
- [32] Weikang Meng, Liangyu Huo, Yadan Luo, Jiawen Guan, Jingyi Zhang, Yingjian Li, and Zheng Zhang. Still: Selecting tokens for intra-layer hybrid attention to linearize llms, 2026.
- [33] Weikang Meng, Yadan Luo, Liangyu Huo, Yingjian Li, Yaowei Wang, Xin Li, and Zheng Zhang. Norm×direction: Restoring the missing query norm in vision linear attention, 2026.
- [34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.
- [35] Stefan Heimersheim and Neel Nanda. How to use and interpret activation patching, 2024.
- [36] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt, 2023.
- [37] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention, 2020.

- [38] Yonggan Fu, Xin Dong, Shizhe Diao, Matthijs Van Keirsbilck, Hanrong Ye, Wonmin Byeon, Yashaswi Karnati, Lucas Liebenwein, Han Zhang, Nikolaus Binder, Maksim Khadkevich, Alexander Keller, Jan Kautz, Yingyan Lin, and Pavlo Molchanov. Nemotron-flash: Towards latency-optimal hybrid small language models. ArXiv, abs/2511.18890, 2025.
- [39] NVIDIA, :, Aakshita Chandiramani, Aaron Blakeman, Abdullahi Olaoye, Abhibha Gupta, Abhilash Somasamudramath, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal, Aditya Vavre, Ahmad Kiswani, Aishwarya Padmakumar, Ajay Hotchandani, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Gronskiy, Alex Kondratenko, Alex Neefus, Alex Steiner, Alex Yang, Alexander Bukharin, Alexander Young, Ali Hatamizadeh, Ali Taghibakhshi, Alina Galiautdinova, Alisa Liu, Alok Kumar, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Anahita Bhiwandiwalla, Ananth Subramaniam, Andrew Tao, Anjaney Shrivastava, Anjulie Agrusa, Ankur Srivastava, Ankur Verma, Ann Guan, Anna Shors, Annamalai Chockalingam, Anubhav Mandarwal, Aparnaa Ramani, Arham Mehta, Arti Jain, Arun Venkatesan, Asha Anoosheh, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asit Mishra, Asli Sabanci Demiroz, Asma Kuriparambil Thekkumpate, Atefeh Sohrabizadeh, Avinash Kaur, Ayush Dattagupta, Barath Subramaniam Anandan, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Benjamin Chislett, Besmira Nushi, Bilal Kartal, Bill Thiede, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Buvaneswari Mani, Carlo del Mundo, Chankyu Lee, Chanran Kim, Chantal Hwang, Chao Ni, Charles Wang, Charlie Truong, Cheng-Ping Hsieh, Chenhan Yu, Chenjie Luo, Cherie Wang, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Chris Holguin, Chris Wing, Christian Munley, Christopher Parisien, Chuck Desai, Chunyang Sheng, Collin Neale, Cyril Meurillon, Dakshi Kumar, Dan Gil, Dan Su, Dane Corneil, Daniel Afrimi, Daniel Burkhardt Eliuth Triana, Daniel Egert, Daniel Fatade, Daniel Lo, Daniel Rohrer, Daniel Serebrenik, Daniil Sorokin, Daria Gitman, Daria Levy, Darko Stosic, David Edelsohn, David Messina, David Mosallanezhad, David Tamok, Deena Donia, Deepak Narayanan, Devin O’Kelly, Dheeraj Peri, Dhruv Nathawani, Di Wu, Dima Rekesh, Dina Yared, Divyanshu Kakwani, Dmitry Konyagin Brandon Tuttle, Dong Ahn, Dongfu Jiang, Dorrin Poorkay, Douglas O’Flaherty, Duncan Riach, Dusan Stosic, Dustin Van Stee, Edgar Minasyan, Edward Lin, Eileen Peters Long, Elad Segal, Elena Lantz, Elena Lewis, Ellie Evans, Elliott Ning, Eric Chung, Eric Harper, Eric Pham-Hung, Eric W. Tramel, Erick Galinkin, Erik Pounds, Esti Etrog, Evan Briones, Evan Wu, Evelina Bakhturina, Evgeny Tsykunov, Ewa Dobrowolska, Farshad Saberi Movahed, Farzan Memarian, Fay Wang, Fei Jia, Felipe Soares, Felipe Vieira Frujeri, Feng Chen, Fengguang Lin, Ferenc Galko, Fortuna Zhang, Frankie Siino, Frida Hou, Gantavya Bhatt, Gargi Prasad, Geethapriya Venkataramani, Geetika Gupta, George Armstrong, Gerald Shen, Giulio Borghesi, Gordana Neskovic, Gorkem Batmaz, Grace Lam, Grace Wu, Greg Pauloski, Greyson Davis, Grigor Nalbandyan, Guoming Zhang, Guy Farber, Guyue Huang, Haifeng Qian, Haran Kumar Shiv Kumar, Harry Kim, Harsh Sharma, Hayate Iso, Hayley Ross, Herbert Hum, Herman Sahota, Hexin Wang, Himanshu Soni, Hiren Upadhyay, Huy Nguyen, Iain Cunningham, Ido Galil, Ido Shahaf, Igino Padovani, Igor Gitman, Igor Shovkun, Ikroop Dhillon, Ilya Loshchilov, Ingrid Kelly, Itamar Schen, Itay Levy, Ivan Moshkov, Izik Golan, Izzy Putterman, Jain Tu, Jan Baczek, Jan Kautz, Jane Polak Scowcroft, Janica Rosenberg, Jared Casper, Jarrod Pflum, Jason Grant, Jason Sewall, Jatin Mitra, Jeffrey Glick, Jenny Chen, Jesse Oliver, Jiacheng Xu, Jiafan Zhu, Jialin Song, Jian Zhang, Jiaqi Zeng, Jie Lou, Jill Milton, Jim Chow, Jimmy Zhang, Jinhang Choi, Jining Huang, Jocelyn Huang, Joel Caruso, Joey Conway, Joey Guman, Johan Jatko, John Kamalu, Johnny Greco, Jonathan Cohen, Jonathan Raiman, Joseph Jennings, Joyjit Daw, Juan Yu, Julio Tapia, Junkeun Yi, Jupinder Parmar, Jyothi Achar, Kari Briski, Kartik Mattoo, Katherine Cheung, Katherine Luna, Keith Wyss, Kevin Shih, Kezhi Kong, Khanh Nguyen, Khushi Bhardwaj, Kirill Buryak, Kirthi Shankar Sivamani, Konstantinos Krommydas, Kris Murphy, Krishna C. Puvvada, Krzysztof Pawelec, Kumar Anik, Laikh Tewari, Laya Sleiman, Leo Du, Leon Derczynski, Li Ding, Lilach Ilan, Lingjie Wu, Lizzie Wei, Luis Vega, Lun Su, Maarten Van Segbroeck, Maer Rodrigues de Melo, Magaret Zhang, Mahan Fathi, Makesh Narsimhan Sreedhar, Makesh Sreedhar, Makesh Tarun Chandran, Manuel Reyes Gomez, Maor Ashkenazi, Marc Cuevas, Marc Romeijn, Margaret Zhang, Mark Cai, Mark Gabel, Markus Kliegl, Martyna Patelka, Maryam Moosaei, Matthew Varacalli, Matvei Novikov, Mauricio Ferrato, Mehrzad Samadi, Melissa Corpuz, Meng Xin, Mengdi Wang, Mengru Wang, Meredith Price, Micah Schaffer, Michael Andersch, Michael Boone, Michael Evans, Michael Z Wang, Miguel Martinez, Mikail Khona, Mike Chrzanowski, Mike Hollinger, Mingyuan Ma, Minseok Lee, Mohammad Dabbah, Mohammad Shoeybi, Mostofa Patwary, Nabin Mulepati, Nader Khalil, Najeeb Nabwani, Nancy Agarwal, Nanthini Balasubramaniam, Narimane Hennouni, Narsi Kodukula, Natalie Hereth, Nathaniel Pinckney, Nave Assaf, Negar Habibi, Nestor Qin, Neta Zmora, Netanel Haber, Nick Reamaroon, Nickson Quak, Nidhi Bhatia, Nikhil Jukar, Nikki Pope, Nikolai Ludwig, Nima Tajbakhsh, Nir Ailon, Nirmal

- Juluru, Nirmalya De, Nowel Pitt, Oleg Rybakov, Oleksii Hrinchuk, Oleksii Kuchaiev, Olivier Delalleau, Oluwatobi Olabiyi, Omer Ullman Argov, Omri Almog, Omri Puny, Oren Tropp, Otavio Padovani, Ouye Xie, Parth Chadha, Pasha Shamis, Paul Gibbons, Pavlo Molchanov, Peter Belcak, Peter Jin, Pinky Xu, Piotr Januszewski, Pooya Jannaty, Prachi Shevate, Pradeep Thalasta, Pranav Prashant Thombre, Prasoon Varshney, Prerana Gambhir, Pritam Gundecha, Przemek Tredak, Qing Miao, Qiyu Wan, Quan Tran Minh, Rabeeh Karimi Mahabadi, Rachel Oberman, Rachit Garg, Rahul Kandu, Raina Zhong, Ran El-Yaniv, Ran Zilberstein, Rasoul Shafipour, Renee Yao, Renjie Pi, Richard Mazzarese, Richard Wang, Rick Izzo, Ridhima Singla, Rima Shahbazyan, Rishabh Garg, Ritika Borkar, Ritu Gala, Riyad Islam, Robert Clark, Robert Hesse, Roger Waleffe, Rohit Varma Kalidindi, Rohit Watve, Roi Koren, Ron Fan, Ruchika Kharwar, Ruisi Cai, Ruoxi Zhang, Russell J. Hewett, Ryan Prenger, Ryan Timbrook, Ryota Egashira, Sadegh Mahdavi, Sagar Singh Ashutosh Joshi, Sahil Modi, Samuel Kriman, Sandeep Pombra, Sanjay Kariyappa, Sanjeev Satheesh, Santiago Pombo, Saori Kaji, Satish Pasumarthi, Saurav Mishra, Saurav Muralidharan, Scott Hara, Sean Narenthiran, Sebastian Rogawski, Seonjin Na, Seonmyeong Bak, Sepehr Sameni, Seth Poulos, Shahar Mor, Shantanu Acharya, Shaona Ghosh Adam Lord, Sharath Turuvekere Sreenivas, Shaun Kotek, Shaya Gharghabi, Shelby Thomas, Sheng-Chieh Lin, Shibani Likhite, Shiqing Fan, Shiyang Chen, Shreya Gopal, Shrimai Prabhumoye, Shubham Pachori, Shubham Toshniwal, Shuo Zhang, Shuoyang Ding, Shyam Renjith, Shyamala Prayaga, Siddhartha Jain, Simeng Sun, Sirisha Rella, Sirshak Das, Smita Ithape, Sneha Harishchandra S, Somshubra Majumdar, Soumye Singhal, Sri Harsha Singudasu, Sriharsha Niverty, Stas Sergienko, Stefana Gloginic, Stefania Alborghetti, Stephen Ge, Stephen McCullough, Sugam Dipak Devare, Suguna Varshini Velury, Sukrit Rao, Sumeet Kumar Barua, Sunny Gai, Suseella Panguluri, Sushil Koundinyan, Swathi Patnam, Sweta Priyadarshi, Swetha Bhendigeri, Syeda Nahida Akter, Sylendran Arunagiri, Tailling Yuan, Talor Abramovich, Tan Bui, Tan Yu, Terry Kong, Thanh Do, Thomas Gburek, Thorgane Marques, Tiffany Moore, Tijmen Blankevoort, Tim Moon, Timothy Ma, Tiyasa Mitra, Tomasz Grzegorzek, Tomer Asida, Tomer Bar Natan, Tomer Keren, Tomer Ronen, Traian Rebedea, Trenton Starkey, Tugrul Konuk, Twinkle Vashishth, Tyler Condensa, Udi Karpas, Ushnish De, Vahid Noorozi, Vahid Noroozi, Vanshil Atul Shah, Veena Vaidyanathan, Venkat Srinivasan, Venmugil Elango, Victor Cui, Vijay Korthikanti, Vikas Mehta, Virginia Adams, Virginia Wu, Vitaly Kurin, Vitaly Lavrukhin, Vladimir Anisimov, Wan Seo, Wanli Jiang, Wasi Uddin Ahmad, Wei Du, Wei Ping, Wei-Ming Chen, Wendy Quan, Wenliang Dai, Wenwen Gao, Will Jennings, William Zhang, Xiaowei Ren, Xiaowen Xin, Xin Li, Yang Yu, Yangyi Chen, Yaniv Galron, Yashaswi Karnati, Yejin Choi, Yev Meyer, Yi-Fu Wu, Yian Zhang, Ying Lin, Yonatan Geifman, Yonggan Fu, Yoshi Suhara, Youngeun Kwon, Yuan Zhang, Yuki Huang, Zach Moshe, Zhilin Wang, Zhiyu Cheng, Zhongbo Zhu, Zhuolin Yang, Zihan Liu, Zijia Chen, Zijie Yan, and Zuhair Ahmed. Nemotron 3 super: Open, eﬀicient mixture-of-experts hybrid mamba-transformer model for agentic reasoning, 2026.
- [40] William Merrill, Yanhong Li, Tyler Romero, Anej Svete, Caia Costello, Pradeep Dasigi, Dirk Groeneveld, David Heineman, Bailey Kuehl, Nathan Lambert, Chuan Li, Kyle Lo, Saumya Malik, DJ Matusz, Benjamin Minixhofer, Jacob Morrison, Luca Soldaini, Finbarr Timbers, Pete Walsh, Noah A. Smith, Hannaneh Hajishirzi, and Ashish Sabharwal. Olmo hybrid: From theory to practice and back, 2026.
- [41] Jusen Du, Jiaxi Hu, Tao Zhang, Weigao Sun, and Yu Cheng. Native hybrid attention for eﬀicient sequence modeling, 2026.
- [42] Yusheng Zhao, Hourun Li, Bohan Wu, Yichun Yin, Lifeng Shang, Jingyang Yuan, Meng Zhang, and Ming Zhang. Switch attention: Towards dynamic and fine-grained hybrid transformers, 2026.
- [43] Disen Lan, Weigao Sun, Jiaxi Hu, Jusen Du, and Yu Cheng. Liger: Linearizing large language models to gated recurrent structures. In Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2025.
- [44] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. Duoattention: Eﬀicient long-context llm inference with retrieval and streaming heads, 2024.
- [45] Zecheng Tang, Quantong Qiu, Yi Yang, Zhiyi Hong, Haiya Xiang, Kebin Liu, Qingqing Dang, Juntao Li, and Min Zhang. Elastic attention: Test-time adaptive sparsity ratios for eﬀicient transformers, 2026.
- [46] Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Van Keirsbilck, Min-Hung Chen, Yoshi Suhara, Yingyan Lin, Jan Kautz, and Pavlo Molchanov. Hymba: A hybrid-head architecture for small language models, 2024.

- [47] Jungo Kasai, Hao Peng, Yizhe Zhang, Dani Yogatama, Gabriel Ilharco, Nikolaos Pappas, Yi Mao, Weizhu Chen, and Noah A. Smith. Finetuning pretrained transformers into rnns, 2021.
- [48] Daniel Goldstein, Eric Alcaide, Janna Lu, and Eugene Cheah. Radlads: Rapid attention distillation to linear attention decoders at scale, 2026.
- [49] Aviv Bick, Kevin Y. Li, Eric P. Xing, J. Zico Kolter, and Albert Gu. Transformers to ssms: Distilling quadratic knowledge to subquadratic models, 2025.
- [50] Michael Zhang, Simran Arora, Rahul Chalamala, Alan Wu, Benjamin Spector, Aaryan Singhal, Krithik Ramesh, and Christopher Ré. Lolcats: On low-rank linearizing of large language models, 2025.
- [51] Abhinav Moudgil, Ningyuan Huang, Eeshan Gunesh Dhekane, Pau Rodríguez, Luca Zappella, and Federico Danieli. Attention to mamba: A recipe for cross-architecture distillation, 2026.
- [52] Yanhong Li, Songlin Yang, Shawn Tan, Mayank Mishra, Rameswar Panda, Jiawei Zhou, and Yoon Kim. Distilling to hybrid attention models via kl-guided layer selection, 2025.
- [53] Yingfa Chen, Zhen Leng Thai, Zihan Zhou, Zhu Zhang, Xingyu Shen, Shuo Wang, Chaojun Xiao, Xu Han, and Zhiyuan Liu. Hybrid linear attention done right: Eﬀicient distillation and effective architectures for extremely long contexts, 2026.
- [54] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints, 2023.
- [55] Fred Zhang and Neel Nanda. Towards best practices of activation patching in language models: Metrics and methods, 2024.
- [56] Neel Nanda. Attribution patching: Activation patching at industrial scale. https://www.neelnanda. io/mechanistic-interpretability/attribution-patching, 2023. Blog post.
- [57] Yoav Gelberg, Koshi Eguchi, Takuya Akiba, and Edoardo Cetin. Extending the context of pretrained llms by dropping their positional embeddings, 2025.
- [58] Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free, 2025.
- [59] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Eﬀicient streaming language models with attention sinks, 2024.
- [60] Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024.
- [61] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models?, 2024.
- [62] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.
- [63] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.
- [64] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. arXiv preprint arXiv:1907.10641, 2019.
- [65] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language, 2019.
- [66] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context, 2016.

- [67] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.
- [68] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.
- [69] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021.
- [70] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them, 2022.
- [71] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024.
- [72] Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Various lengths, constant speed: Eﬀicient language modeling with lightning attention, 2024.
- [73] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training, 2018.
- [74] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024.
- [75] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. ChoquetteChoo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Plucińska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gustavo

- Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard Hussenot. Gemma 3 technical report, 2025.
- [76] Yuxian Gu, Qinghao Hu, Shang Yang, Haocheng Xi, Junyu Chen, Song Han, and Han Cai. Jet-nemotron: Eﬀicient language model with post neural architecture search. arXiv preprint arXiv:2508.15884, 2025.

### A More Details of Various Hybrid Architectures

To comprehensively evaluate the eﬀicacy of different hybridization strategies, we construct and compare three representative architectures: layer-wise hybrids, token-wise hybrids, and head-wise hybrids (Figure 10). All models incorporate the structural optimizations proposed in our main paper, including the addition of a gating branch to Full Attention (FA) layers with scaling mechanisms replacing RoPE, and the integration of RoPE positional encoding into Gated DeltaNet (GDN) layers.

Layer-wise Hybrid Architectures. We explore two distinct granularities for layer-wise mixing, guided by the layer selection masks derived from HypeNet. (a) FA & LA: Each transformer layer is configured as either an FA block (FA + MLP) or a Linear block (GDN + MLP). The HypeNet-derived mask determines which layers retain FA capabilities. (b) LA-FA & LA: Each layer is configured as either a Composite block (GDN + FA + MLP) or a Linear block (GDN + MLP). In this setting, the HypeNet mask identifies layers that require the combined power of both attention mechanisms, while others rely solely on GDN for eﬀiciency.

Token-wise Hybrid Architectures. As illustrated in Figure 10, our token-wise design conceptually assigns FA (implemented as Sliding Window Attention, SWA) to tokens within the recent window and LA to those outside it. For implementation simplicity, we realize this by processing all tokens through the GDN branch for global context modeling while additionally attending to recent tokens within the sliding window via FA to capture high-precision local dependencies. We consider two variants of this design. The first applies 4K-context SWA combined with GDN uniformly across all layers. The second adopts a layerinterleaved strategy inspired by Liger [43]: 75% of the layers use SWA (window size 128) with GDN, while the remaining 25% combine full FA with GDN. Unlike Liger, which employs pure FA in its global layers, our variant retains GDN alongside FA to preserve token-wise hybridization throughout all layers.

Although this design can be categorized as token-wise mixing due to the varying receptive fields across token positions, the way features from the two branches are combined is essentially identical to the head-wise mixing approach. A more canonical form of token-wise mixing operates at the KV cache level during softmax attention computation, where different tokens attend to KV caches of varying lengths or compositions [32, 50]. However, our preliminary experiments reveal that such methods struggle to acquire long-context capabilities through transfer learning when the pretraining context window is orders of magnitude shorter than the target evaluation length. We therefore defer their comparison to future work and do not include them in this paper.

Head-wise Hybrid Architectures. We define two sub-paradigms for head-wise hybridization: headwise mixing, where each head simultaneously computes both FA and GDN branches and fuses their outputs, and head-wise selection, where each head is exclusively assigned to either the FA or GDN branch based on a learned or heuristic criterion. For head-wise mixing, we implement a variant inspired by [46] that processes the input through parallel FA and GDN branches, each operating over the full set of heads, and fuses their outputs at the feature level within every layer. For head-wise selection, existing work has explored per-head hybridization between FA and sparse attention [44, 45]. We therefore use HydraHead as the representative of this paradigm that adopts linear attention instead.

### B More Results of Hybrid Architecture Comparison

We perform a detailed comparison of various typical hybrid architectures in Table 2. Table 13 summarizes the long-context evaluation results on the RULER benchmark, covering both Single and Multi-key scenarios across context lengths ranging from 16K to 256K tokens. Our method achieves the best performance in nearly all settings, demonstrating robust and broad superiority in long-context modeling. Tables 14 and 15 present the general capability comparisons. Our approach significantly surpasses layer-wise hybrid models, with particularly notable gains in mathematical reasoning and code generation. While other head-wise and token-wise hybrid variants achieve stronger general performance than our model, they suffer from severely compromised long-context capabilities. Taken together, our hybrid design achieves the most favorable tradeoff between general competence and long-context proficiency.

- Table 13: Comparison of various hybrid architectures in the RULER benchmark. Cache denotes the relative key-value cache under a 16K context length, where the standard FA cache is normalized to 1.0. Light. denotes the lightning attention [72]. ∗ and ‡ indicate the hybrids are similar to HypeNet [53] and Liger [43], respectively. † indicates that an FA layer is inserted after every 3 LA layers. HW denotes Head-wise.

Model LA

RULER Single RULER Multi-Key 16K 32K 64K 128K 256K 16K 32K 64K 128K 256K Layer-wise Hybrid Transformers

FA & LA∗ Light. 92.13 86.00 88.87 84.47 81.67 28.90 20.80 32.70 22.70 17.70 FA & LA† Light. 87.67 84.40 80.33 69.40 53.53 36.70 26.70 26.00 14.60 12.80 FA & LA GDN 62.87 32.40 15.67 4.20 1.20 25.10 19.90 10.30 4.80 1.50 LA-FA & LA GDN 73.13 43.87 17.27 5.20 1.93 23.00 15.00 12.80 5.70 2.00

Token-wise & Head-wise Hybrid Transformers

Token-wise GDN 28.07 13.47 6.07 3.00 2.13 21.60 10.50 3.40 2.40 1.50 Token-wise‡ GDN 97.87 94.20 80.87 56.13 39.33 29.00 23.40 15.70 9.60 7.80 HW Mixing GDN 98.33 88.07 77.73 61.73 41.80 46.10 28.60 21.30 14.30 8.00 HydraHead GDN 98.60 98.33 94.60 89.00 78.87 41.40 32.80 32.30 26.60 23.20

- Table 14: Comparison of various hybrid architectures in the general benchmark (Easy). Cache denotes the relative key-value cache under a 16K context length, where the standard FA cache is normalized to 1.0. Light. denotes the lightning attention [72]. ∗ and ‡ indicate the hybrids are similar to HypeNet [53] and Liger [43], respectively. † indicates that an FA layer is inserted after every 3 LA layers.

Model LA ARC-C ARC-E HellaSwag LAMBADA PIQA Winogrande Layer-wise Hybrid Transformers

FA & LA∗ Light. 41.38 71.13 58.81 52.88 72.74 61.40 FA & LA† Light. 43.52 72.01 59.20 50.86 72.85 63.30 FA & LA GDN 39.85 70.75 58.05 47.25 71.76 61.40 LA-FA & LA GDN 41.21 70.24 58.46 49.93 72.03 60.30

Token-wise & Head-wise Hybrid Transformers

Token-wise GDN 46.50 75.38 62.58 58.49 74.05 63.38 Token-wise‡ GDN 45.65 75.34 62.43 58.28 74.27 64.25 Head-wise Mixing GDN 43.60 76.09 61.97 57.33 73.61 64.01 HydraHead GDN 42.83 76.05 61.82 56.32 73.88 61.80

- Table 15: Comparison of various hybrid architectures in the general benchmark (Hard). Cache denotes the relative key-value cache under a 16K context length, where the standard FA cache is normalized to 1.0. Light. denotes the lightning attention [72]. ∗ and ‡ indicate the hybrids are similar to HypeNet [53] and Liger [43], respectively. † indicates that an FA layer is inserted after every 3 LA layers.

Model LA MMLU BBH MBPP GSM8k Layer-wise Hybrid Transformers

FA & LA∗ Light. 48.80 13.79 10.00 6.60 FA & LA† Light. 39.70 9.77 7.60 6.67 FA & LA GDN 46.67 13.58 2.40 3.11 LA-FA & LA GDN 48.32 14.53 4.40 6.44

Token-wise & Head-wise Hybrid Transformers

Token-wise GDN 56.65 47.87 34.60 50.11 Token-wise‡ GDN 53.17 27.65 29.20 36.63 Head-wise Mixing GDN 54.46 35.94 29.20 32.68 Ours GDN 49.86 29.15 22.00 23.12

Layer-wise (FA & LA) Layer-wise (LA-FA & LA) Head-wise (Selection)

Token-wise Head-wise (Mixing)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | |Heads| |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

s

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| |
|---|

Layers

LA

FA FA+LA

- Figure 10: Visualization of some typical hybrid methods. The hybrids can be categorized in terms of layerwise, token-wise, and head-wise.

### C Head-Selection Details and Additional Interpretability Analysis

This appendix collects the implementation details of the head-importance estimator of Section 4.1, the robustness caveats referenced there, the second-capability (general-ability) localization, and the full set of interpretability figures for Qwen3-1.7B summarized in Section 5.7.

##### C.1 Estimator details and caveats

Direct-effect measurement. When scoring receivers (Equation (10)) we freeze every downstream attention output to its clean value, so the patched head can influence the logits only through the residual stream and the downstream MLPs. This isolates the head’s direct effect on the readout, in the sense of Wang et al. [16], rather than a total effect that would also route through downstream attention. The downstream MLPs are intentionally left unfrozen, so the measurement captures the head’s contribution as actually consumed by the rest of the network.

Corruption design. For the retrieval capability the counterfactual x′ is a symmetric token replacement: the needle value is replaced by a fresh distractor of the same type and the same token length, with the haystack, query, and structure held identical. This keeps x′ on the model’s natural distribution—the rec-

###### Head-vs-layer similarity distributions (wide within-layer spread = heads in same layer are diverse)

- 16

#distincttop-1tokens

Per-layer head logit-lens top-1 diversity

max (16)

0 5 10 15 20 25

Layer

0.0

2.5

5.0

7.5

10.0

12.5

15.0

- 17.5

14

within-layer head pairs (n=3360)

- 0

- 1

- 2

- 3

- 4

- 5

12

between-layer head pairs (n=96768) between-layer layer outputs (n=378)

10

8

6

4

2

0

0 5 10 15 20 25

density

Layer

Per-layer head logit divergence (higher = heads encode more distinct info)

meanpairwisesym-KL

-0.50 -0.25 0.00 0.25 0.50 0.75 1.00 1.25

cosine similarity

- Figure 11: Functional diversity of heads within a layer (Qwen3-1.7B). (Left) Distributions of cosine similarity for within-layer head pairs vs. between-layer head pairs vs. between-layer full-output pairs; within-layer head pairs span a wide range, indicating non-redundancy. (Right) Per-layer count of distinct top-1 logitlens tokens across the 16 heads (max 16) and their mean pairwise symmetric KL; both are high in mid/late layers.

ommended practice over additive activation noise, which pushes the residual stream out of distribution and propagates the artifact downstream [55]. The span decay in Equation (9) is λ = 0.9 for the retrieval probe.

Calibration configuration. We localize on a small calibration set—eight NIAH samples in our experiments—drawn at short context, and transfer the resulting selection to long context, which is justified by the length-robustness of the selected set (Figure 8). The per-head ranking is already stable from roughly six samples (Figure 8), so the exact sample count is not load-bearing. Scores are computed in fp16: this preserves the importance ranking, whereas lower-precision bf16 can collapse genuinely distinct heads to indistinguishable scores. We aggregate importance as the union of the per-head scores over the five NIAH sub-probes and mark a head as critical when |drop| ≥ 0.01.

Patching engine. We use full activation patching, implemented in two phases for tractability: a layerlevel screen (patch all heads of a layer at once) followed by a head-level refinement of the surviving layers. No gradients or attribution approximations are used in the reported numbers. The scalable alternative, attribution patching, estimates all heads’ effects from a single backward pass via a first-order Taylor expansion of Equation (10); its linearization is accurate enough for the rank-ordered top-K use we make of it (its exploratory regime), though not for absolute effect sizes [56].

Backup heads and rank noise. Single-head necessity under-estimates heads that have redundant “backup” pathways: knocking one out lets a backup compensate, so the measured drop is smaller than the head’s true role [16]. Individual head ranks are therefore noisy even when the selected set is robust, which is why we (i) take the union across sub-probes and (ii) rely on the head population rather than exact ranks (Section 5.7). The selection step is exploratory (rank order); the knockout study (Figure 9b) is the confirmatory check that the chosen set is causally load-bearing.

##### C.2 Functional diversity of heads within a layer

The head-localization result of Section 5.7 presumes that heads in a layer actually do different things, which we verify directly. Comparing the 448×448 matrix of per-head output cosine similarities against the 28×28 matrix of full layer-output similarities (Figure 12 and Figure 2b), the within-layer head-pair similarities span a wide range—down to near-orthogonal for at least one pair in most layers (Figure 12)—whereas the layer-level matrix is smooth. A head-level logit lens tells the same story: projecting each head’s contribution to vocabulary space, the 16 heads of a mid-to-late layer predict many distinct top-1 tokens and have high pairwise symmetric KL divergence (Figure 11). Heads inside a single layer are therefore not redundant copies, and the per-head granularity carries information invisible at the layer level—the representational counterpart of the causal head-localization result and the empirical claim underlying Figure 2.

- Table 16: Counter-examples to layer-level granularity on Qwen3-1.7B: the top layers that simultaneously contain long-context-critical heads (|drop| ≥ 0.01) and replaceable heads (< 0.005). Every such layer would be mis-served by a uniform per-layer mechanism assignment.

Layer 27 23 20 21 25 26 19 22 24 # long-context heads 6 4 3 3 3 3 2 2 2

# replaceable heads 9 12 11 10 13 12 13 14 13

###### Within-layer head similarity range (low min = at least one pair of heads in the layer is very different)

1.0

0.8

within-layerhead-paircosine

0.6

0.4

min..max within-layer cosine

mean

0.2

0.0

-0.2

0 5 10 15 20 25

Layer

- Figure 12: Per-layer (min, mean, max) within-layer head-pair cosine similarity on Qwen3-1.7B; the minimum dips to near zero for many layers, i.e. at least one pair of heads in such a layer encodes near-orthogonal information. The full layer-level (28 × 28) similarity matrix is shown in Figure 2b.

### D Limitations and Future Work

While HydraHead demonstrates promising results in balancing general reasoning and long-context retrieval, several limitations point to directions for future exploration.

Model scale and architecture diversity. Our experiments are primarily conducted on Qwen3-1.7B, a relatively compact dense Transformer. Despite this modest scale, our results provide compelling evidence for the potential of head-wise hybridization as a general design principle—the consistent gains observed across both retrieval-centric and reasoning benchmarks suggest that functional specialization of attention heads is a robust phenomenon worth exploiting at larger scales. Validating this principle on larger-scale models (e.g., 7B, 13B, and beyond) and across diverse architectures—including Mixture-of-Experts and multimodal models—remains an important next step. The functional specialization patterns of attention heads may shift with model scale and modality, and understanding how these shifts interact with our interpretability-guided head selection strategy would further strengthen the generality of the approach.

Training data scale and post-training. The current scaling experiments are limited to approximately 15B tokens, which is modest by contemporary standards. Scaling up the training corpus by an order of magnitude or more, combined with post-training stages such as instruction tuning and reinforcement learning from human feedback, could unlock substantially stronger hybrid models. In particular, it remains to be seen whether head-wise hybridization confers additional benefits in alignment-centric post-training regimes, where the balance between precise instruction following and long-context reasoning becomes especially critical.

Scalability and generality of interpretability-guided selection. The current head selection procedure relies on full activation patching, which requires iterating over all attention heads with separate forward passes per head—tractable at the 1.7B scale but increasingly costly at frontier scales. Nevertheless, even full activation patching remains far cheaper than the alternative of training multiple hybrid models under different head configurations to search for a good assignment. While attribution patching offers a singlebackward-pass approximation, its accuracy for this specific use case has not been systematically validated in our experiments. Beyond computational cost, the procedure also depends on carefully designed calibration inputs: the paired counterfactual construction, the choice of readout metric, and the target capability

specification (NIAH sub-probes as a proxy for general long-context retrieval) all require domain-informed design decisions that may not transfer directly to new capabilities or task families. That said, this design effort is a one-time cost amortized across all downstream hybrid models built from a given checkpoint, and the resulting selection is stable and transfers across context lengths. Developing more automated or taskagnostic selection protocols would further broaden the applicability of interpretability-guided hybridization.

Gap between interpretability-based importance and deployable FA budget. Our interpretability analysis suggests that only a small fraction of heads are highly important: roughly 6.5% of heads are identified as carrying the most critical retrieval-related signal. However, the ratio experiments indicate that when the retained FA budget is reduced to around 10%, model capability already deteriorates noticeably. This mismatch suggests that the current interpretability signal, while clearly informative, is not yet suﬀicient to fully determine the minimal FA allocation required for strong end-task performance. One possibility is that the head-importance estimation itself remains incomplete or noisy, and could be improved with better causal analysis or more robust calibration procedures. Another possibility is that the architecture and training pipeline are not yet able to preserve capability under such aggressive compression, even when the most important heads are retained. Addressing this gap may therefore require progress on both fronts: more accurate interpretability-guided selection, and stronger hybrid architectures or optimization strategies that can better exploit extremely sparse FA budgets.

Broader attention mechanism combinations. It is important to emphasize that the head-wise hybridization architecture we propose is not inherently tied to any specific attention mechanism—it is a general architectural paradigm that can accommodate arbitrary combinations of attention variants. In this work, we instantiate the architecture with two representative mechanisms, FA and GDN, as a concrete proof of concept. The architecture naturally extends to a much richer design space. On the eﬀicient attention side, beyond LA variants (e.g., Mamba, RWKV, Linear Transformer), FA can also be combined with sparse attention mechanisms (e.g., sliding window, local attention) or latent-space attention formulations, each offering distinct eﬀiciency–expressiveness trade-offs. More ambitiously, the architecture supports functional hybridization of more than two attention types—for instance, jointly mixing FA, LA, and sparse attention within a single model, where different heads adopt different mechanisms according to their functional roles. Systematically characterizing the head-level compatibility patterns across this rich design space could lead to more effective and deployment-specific hybrid recipes.

