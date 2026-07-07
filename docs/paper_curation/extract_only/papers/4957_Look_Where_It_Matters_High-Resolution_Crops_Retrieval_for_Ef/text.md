# arXiv:2603.16932v1[cs.CV]14Mar2026

## Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs

Nimrod Shabtay1,2⋆, Moshe Kimhi1,3⋆, Artem Spector1, Sivan Haray1, Ehud Rivlin3, Chaim Baskin4, Raja Giryes2, and Eli Schwartz1

- 1 IBM Research
- 2 Tel-Aviv University 3 Technion

4 Ben-Gurion University

Abstract. Vision-language models (VLMs) typically process images at a native high-resolution, forcing a trade-off between accuracy and computational efficiency: high-resolution inputs capture fine details but incur significant computational costs, while low-resolution inputs advocate for efficiency, they potentially miss critical visual information, like small text. We present AwaRes, a spatial-on-demand framework that resolves this accuracy–efficiency trade-off by operating on a low-resolution global view and using tool-calling to retrieve only high-resolution segments needed for a given query. We construct supervised data automatically: a judge compares low- vs. high-resolution answers to label whether cropping is needed, and an oracle grounding model localizes the evidence for the correct answer, which we map to a discrete crop set to form multi-turn tool-use trajectories. We train our framework with cold-start SFT followed by multi-turn GRPO with a composite reward that combines semantic answer correctness with explicit crop-cost penalties.

Project Page: https://nimrodshabtay.github.io/AwaRes/

[Figure 1]

What’s the temperature?

[Figure 2]

[Figure 3]

###### U

Low-Res Image

[Figure 4]

[Figure 5]

GET_CROPS:['center']

[Figure 6]

[Figure 7]

<tool_response>:

[Figure 8]

High-Res Crop

[Figure 9]

The temperature is 5°C ✓

Fig.1: AwaRes overview. Left: Given a low-resolution image, AwaRes uses tool-calling to request only the high-resolution crops needed to answer the query. Right: Accuracy vs. retained visual tokens across six benchmarks. AwaRes performs similarly to native high-resolution (80.3%) while using only 36% of the visual tokens.

⋆ Equal contribution.

### 1 Introduction

Vision–language models (VLMs) increasingly rely on high-resolution visual inputs to solve detail-sensitive tasks such as document question answering, chart understanding, and understanding semantics and text in dense natural images. However, high resolution is expensive: the number of visual tokens grows rapidly with image resolution, making high-resolution inference a major bottleneck in practice.

Existing approaches to reduce this cost largely fall into two camps. First, token pruning methods selectively discard visual tokens to reduce computation [4,26,27,30,32]. While effective in principle, they often introduce irregular token patterns and dynamic sequence lengths that can be difficult to translate into end-to-end serving speedups in common inference stacks, such as vLLM [13], where efficiency is tied to predictable sequence length. Second, resolution escalation methods [12,19] learn when to request a higher-resolution view, but typically treat the decision as binary: if more details are needed, the entire highresolution image is retrieved, wasting computation on regions irrelevant to the question.

A key observation is that the demand for high fidelity is usually spatially sparse, as can be seen in Fig. 3. Many questions require fine detail in only a small portion of the image: a single value on a chart axis, a specific cell in a table, or a tiny object in the corner of an image. In cases where low resolution image do not poses the fine-grained information, retrieving the full image at native high-resolution is unnecessarily expensive. We advocate that answering the question of where to look matters as much as whether to look.

We propose VLM that is spatially aware to resolution (abbreviated AwaRes), a framework that exploits this spatial sparsity via a simple tool-calling interface that targets high-resolution crop acquisition. AwaRes processes a low-resolution global view by default, and when additional detail is required, it invokes a tool-call that requests only specific high-resolution sub-regions, and then answers conditioned on both. This multi-turn structure is naturally compatible with KV-caching: computation from the initial low-resolution turn is reused and extended in the crop turn without architectural changes, making AwaRes practical for deployment.

We train AwaRes to learn a single coupled-decision policy (CDP) that jointly decides

- (i) whether additional resolution is needed and (ii) where to acquire it by selecting a subset of crops. Crucially, these decisions are fused into the model’s first-turn action: either answer directly, or emit a structured crop request that simultaneously signals escalation and specifies the target regions.

For the cold-start phase, we construct the supervision automatically, without manual spatial annotations, by (i) identifying examples where low resolution is insufficient using an LLM as a Judge (LaaJ) that compares low- vs. high-resolution model outputs, and

- (ii) localizing the evidence for the correct answer using an oracle grounding model to produce target crops.

We evaluate AwaRes on six benchmarks spanning document understanding and general visual QA. Across these tasks, AwaRes almost matches full high-resolution performance on average (80.3% vs. 80.46%) while using only 36% of the pixels/tokens, substantially reducing inference cost. On ChartQA, DocVQA and OCRBench, AwaRes even slightly improves over full-resolution baselines while remaining significantly more efficient.

Our Contributions are listed as follows:

- – We introduce a spatial-on-demand inference framework for VLMs that requests only targeted high-resolution crops through tool-calling, enabling system-friendly multi-turn KV-cache reuse.
- – We propose an automatic data curation pipeline that produces multi-turn tool-use trajectories without manual spatial annotations.
- – We refine crop usage with multi-turn GRPO using an explicit accuracy–efficiency objective that penalizes unnecessary crop acquisition while discouraging missed crop requests when detail is required.

### 2 Related Work

Several strategies have emerged to prune, compress, or dynamically reduce the number of visual tokens in Vision Language Models.

One line of research focuses on dynamic token pruning. Methods such as FastV [4], HoloV [32], PyramidDrop [26], FitPrune [28], TopV [8], SparseVILA [11], IVTP [7], LLaVolta [3], and SAINT [9] discard uninformative tokens within the LLM layers based on attention scores or learned criteria. Alternatively, VisionZip [27], FastVLM [22], and SparseVLM [30] prune tokens directly after the vision encoder. While effective, pruning-based approaches must commit to a fixed retention ratio before inference, applying the same token budget regardless of sample complexity. In contrast, our method is fully adaptive: it dynamically determines both whether additional detail is needed and which spatial regions to acquire, allowing simple images to be processed at minimal cost while allocating more resources only when the query demands fine-grained perception.

A second line of work explores resolution selection. CARES [12] uses an external lightweight model to predict the optimal input resolution before the VLM processes the image, while CROP [6] identifies contextual regions of interest via an auxiliary module. These methods rely on external components to make resolution decisions, whereas our approach enables the VLM itself to determine when and where additional detail is needed through its native capabilities, requiring no auxiliary models.

Recent frameworks like ZoomEye [20] and DeepEyes [31] enhance VLM performance through dynamic zooming and high-resolution cropping. However, these methods prioritize accuracy over efficiency: ZoomEye performs multiple inference passes through a hierarchical image tree, while DeepEyes appends zoomed crops to the context, progressively increasing the token count. In contrast, our work employs cropping specifically for efficiency—requesting only the minimal high-resolution regions needed while maintaining a compact token budget.

VisionThink [19] introduced a reinforcement learning approach where the model processes a low-resolution image and emits a tool call to request a high-resolution version when needed. While effective at determining resolution sufficiency, VisionThink retrieves the entire high-resolution image globally when escalation is triggered. Our method goes further by identifying the specific regions that matter for answering the query, requesting only targeted high-resolution sub-regions rather than the full image. This spatial-on-demand approach minimizes token overhead while preserving the accuracy benefits of high-resolution perception exactly where it matters.

### 3 Method

AwaRes implements spatial-on-demand perception via a simple multi-turn interaction: the model first observes a low-resolution global view, and only if needed issues a tool call to retrieve a set of high-resolution crops (Fig. 1). We first formalize this interaction protocol (§3.1), then describe how we automatically curate supervision for the CDP, namely whether additional resolution is needed and where it matters (Fig. 2; §3.2). Finally, we train in two stages: (i) a cold-start supervised fine-tuning (SFT) stage that teaches the tool protocol and yields a supervised reference policy πref (§3.3); and (ii) multi-turn GRPO initialized from πref and regularized toward it via a KL penalty (§3.4), explicitly optimizing the accuracy–efficiency trade-off.

#### 3.1 Problem setup

Given an image–question–answer triple (I,q,a⋆), the model is first shown a low-resolution view Ilow (obtained by downsampling I) together with the question q. The model then chooses between two actions:

- (i) Direct answer: Produces an answer aˆ conditioned only on (q,Ilow).
- (ii) Crop request + answer: Emits a tool call that requests a subset of crops from a predefined candidate set, Creq⊆C. The tool returns the corresponding high-resolution

crop images {Ichigh}c∈Creq

, which are appended to the dialogue context, and the model produces the final answer aˆ conditioned on the full multi-turn history. A fused coupled-decision policy: We parameterize a single policy over high resolution request, and localized crop selection:

πθ(C|q,Ilow), C⊆C, (1) where C =∅ corresponds to no tool call (answer directly) and C≠ ∅ corresponds to escalation with localization. Under this view, “when to crop” is the marginal event 1[C≠ ∅], while “where to crop” is the conditional distribution over C given C≠ ∅. The two are inherently coupled: the value of escalating depends on which regions will be retrieved, since inaccurate localization can waste compute without improving answer correctness. This interface targets efficiency by restricting high-resolution perception to a small number of structured regions, while preserving the low-resolution global context throughout the interaction (See Fig. 1 for a conversation example).

#### 3.2 Data curation: automatic supervision for crop requests

A key challenge is to supervise two coupled decisions: whether the low-resolution view is insufficient, and where to crop when additional detail is needed. We generate this supervision to initiate the model to learn a reference policy πθ

in an automatic fashion using the three-stage pipeline (Illustrated in Fig. 2):

ref

- Stage 1: resolution-sufficiency labeling (when to crop). For each example (I,q,a⋆) we utilize a base VLM T on both the low-resolution and full-resolution inputs:

aˆlow=T(q,Ilow), aˆfull=T(q,I). (2)

Because aˆlow and aˆfull may differ in form though semantically correct, direct string matching (exact match) with a⋆ is unreliable. Instead, we use an LaaJ (LLaMA-3.370B [5]) to compare both predictions to the ground truth a⋆. If it judges aˆlow as correct (or ties it with aˆfull), we label the example as no crop needed LR; otherwise we label it as HR.

- Stage 2: crop target construction (where to crop). For examples labeled HR, we identify the region that contains the visual evidence needed to answer (q,a⋆). We prompt an oracle grounding model G (namely, Qwen3-VL-A235B-A22B [1]) to localize the evidence and

return a bounding box b=(x1,y1,x2,y2) in the coordinate system of the original image.

We then map b to our discrete crop candidate set C, which includes four quadrants, a center crop, four merged half-image regions (top/bottom/left/right), and a full-image. We define the target crop subset as

C⋆={c∈C|IoU(b,c)≥τ}, (3) where τ =0.5 is the IoU threshold. Fig. 3 shows a representative example, and Fig. 5 (left side) summarizes the empirical distribution of selected crops in the curated training set.

No

Yes

Low-Res

Instruction LLM Judge

"Is LR enough?”

Single Turn Conversation

[Figure 10]

[Figure 11]

Base VLM PredictionLow-Res

GT

High-Res

Prediction

Multi Turn Conversation

High-Res

Base VLM

Oracle

Predict crop

Fig.2: Overview of the automatic supervision pipeline. Each sample is processed at two resolutions; an LLM judge determines resolution sufficiency by comparing predictions to ground truth. Sufficient cases yield single-turn conversations, while insufficient cases are routed to an oracle for crop localization, producing multi-turn trajectories with tool-calling.

- Stage 3: supervised tool-use trajectories. The procedure above yields two types of training transcripts:

Direct-answer trajectories (LR). The model observes (q,Ilow) and is supervised to output a⋆ in a single turn.

Tool-call-then-answer trajectories (HR). In the first turn, the model issue a tool call selecting C⋆. After the tool returns {Ichigh}c∈C⋆, the model is trained to produces a⋆ in a second turn conditioned on both the low-resolution and the retrieved high-resolution crops.

This curation pipeline produces multi-turn tool-use supervision at scale in order to learn an initial reference policy πθ

, while keeping the crop interface structured and deployment-friendly (Fig. 2). We provide additional details in the supplementary material.

ref

#### 3.3 Cold-start supervised reference policy (SFT)

We cold-start our crop-request policy by supervised fine-tuning (SFT) on the mixture of direct-answer and tool-call-then-answer trajectories produced in §3.2.

This stage serves two purposes: (i) teach the model to follow the multi-turn tool-calling protocol and learn the coupled decisions (whether additional detail is needed and where it matters), and (ii) produce a strong supervised reference policy πref that we later use for KL-regularized GRPO (§3.4).

Prompt:

What website is being advertised on the plane?

Ground Truth Answer:

Jetstar.com

Low Resolution Oracle Detection Boxes Selected Crop

[Figure 12]

[Figure 13]

[Figure 14]

Prompt:

What is the Grand Total for Net Block As of 31.3.2012?

Ground Truth Answer:

52708.86

Low Resolution Oracle Detection Boxes Selected Crop

[Figure 15]

[Figure 16]

[Figure 17]

Fig. 3: Crop annotation example. Left: low-resolution input where text is illegible. Middle: oracle-predicted bounding box localizing the answer region. Right: selected high-resolution crop enabling correct response (best viewed when zoomed in).

Let y1:T denote the assistant tokens in a supervised transcript, and let ht be the dialogue history at step t (including (q,Ilow), any previously generated tokens, and tool outputs if a crop request occurred). We minimize a weighted negative log-likelihood:

T

LSFT(θ)=−

wtlogπθ(yt|ht) (4)

t=1

The tool-call turn, despite having small numbers of tokens, fully specifies the CDP action and carry disproportionate control over both efficiency and downstream answer quality. Upweighting this turn therefore directly stabilizes learning of the fused first-turn decision. After SFT, we freeze the resulting model as the reference policy πref and initialize GRPO from it.

#### 3.4 Multi-turn GRPO

After the cold-start SFT stage, the model reliably follows the tool protocol but tends to over-request crops even when Ilow is sufficient. We therefore apply Group Relative Policy Optimization (GRPO) on full multi-turn interactions to explicitly optimize the accuracy–efficiency trade-off.

obtained from the SFT. GRPO is initialized from πref and uses a KL penalty to keep πθ close to πref while improving tool usage. Rollouts and trajectories. Given an input prompt x=(q,Ilow), the policy πθ generates first turn that may include a crop tool call. The requested crops are appended to the dialogue context, and generation continues until a final answer aˆ is produced. We treat only assistant tokens as actions; tool outputs are treated as observations. Thus, each rollout yields a multi-turn trajectory τ consisting of assistant actions interleaved with tool observations, ending with aˆ.

We denote by πref the frozen SFT policy πθ

ref

Unlike supervised training with dense per-token loss, GRPO enables optimization with task-specific rewards that directly target improved tool usage.

Reward design. We assign a single scalar reward to the completed trajectory τ, composed from two components:

R(τ)=Rans(ˆa,a⋆)−Ctool(C,y), (5) Answer reward (Rans(ˆa,a⋆)): measures semantic correctness using the cosine similarity between sentence-transformer embeddings of aˆ and a⋆. Tool-use cost: Penalize tool usage with an asymmetric cost:

 

αmiss if y=HR and C=∅ (missed tool-call) αuse+λ∥C∥ if C≠ ∅ (tool usage) 0 if y=LR and C=∅,

(6)

Ctool(C,y)=



This asymmetry biases the policy toward recall in tool invocation: missing a necessary crop request is penalized more heavily than making an unnecessary request. When the tool is used, we additionally penalize the amount of high-resolution evidence requested via ∥C∥, defined as the total fraction of image area covered by the selected crops. This encourages the policy to prefer smaller crops when they suffice. Importantly, the cost depends on how much is requested but remains agnostic to which specific region is chosen, allowing the GRPO to explore alternative policies.

GRPO optimization. For each prompt x, we sample a group of G trajectories {τ1,...,τG} from the current policy πθ. Each trajectory τi consists of a sequence of assistant tokens (actions) interleaved with tool observations, culminating in a final answer. We compute the advantage for each trajectory using the group-relative baseline:

R(τi)−µG σG+ϵ

Aˆi=

, (7)

where R(τi) is the total reward for trajectory τi, and µG, σG are the mean and standard deviation of rewards within the group.

We optimize a PPO-style clipped objective with KL regularization to the reference policy:

LGRPO(θ)=Ex∼D

|τi|

G

1 G

1 |τi|

min rt(i)Aˆi,clip(rt(i),1−ϵ,1+ϵ)Aˆi

t=1

i=1

−βDKL(πθ∥πref) , (8)

(i) t |x,a<t(i))

where rt(i)= πθ(a

is the importance sampling ratio, ϵ is the clipping threshold, and β controls the strength of the KL divergence penalty against a reference policy πref.

πold(at(i)|x,a<t(i))

- As in PPO-style updates, πold denotes a snapshot of the policy before the current

GRPO update step. The KL term is computed over assistant-token distributions along the sampled trajectories, encouraging stable improvements over πref. 3.5 Inference

- At test time, we follow the same interaction protocol used during training. The model re-

ceives (q,Ilow) and either answers directly or emits a tool call selecting a crop subset C⊆C. If a tool call occurs, the corresponding high-resolution crops are appended to the dialogue

context while retaining Ilow, and the model produces the final answer in a second turn.

This results in two possible inference paths: a single prefill pass for queries answerable from the low-resolution view, or two prefill passes when high-resolution detail is required. In the latter case, the model benefits from both the global context preserved in Ilow and the fine-grained detail in the requested crops. When a second turn is required the lowresolution view and the query are already in the KV cache saving on the required compute. Crucially, the decision of which path to take, and which regions to acquire - is made entirely by the learned policy, requiring no external heuristics or task-specific thresholds.

### 4 Experimental Results

We evaluate AwaRes on six benchmarks spanning document understanding and general visual QA, and compare against both fixed-budget token-pruning methods and adaptive resolution-escalation baselines. We report (i) the dataset metric from lmms-eval [29] and (ii) an Retain Token Ratio (RTR), defined as the fraction of visual tokens processed relative to the full-resolution baseline. RTR directly reflects the model’s first-turn coupled-decision policy (answer directly vs. request crops), while accuracy reflects the quality of the full multi-turn interaction. We first describe our evaluation protocol 4.1, evaluated datasets 4.2 and implementation details 4.3. Then, we provide a detailed discussion of the main results 4.5 and conclude by extensive ablations 4.6.

#### 4.1 Evaluation Protocol

All models are evaluated using lmms-eval [29], and we report the per-dataset metrics provided by the framework.

Retain Token Ratio (RTR): We measure efficiency via visual token usage, which dominates compute and KV-cache memory at high resolution. For a sample i, let Ti denote the total number of visual tokens processed across all turns (e.g., low-resolution pass plus any high-resolution crop pass), and let Tfull be the number of visual tokens when processing the full-resolution image once with the baseline model. We define:

Ti Tfull

, RTR=Ei[RTRi]. (9)

RTRi=

For fixed-budget efficient methods we configure the method to retain either 50% or 70% of the full-resolution visual tokens and report the resulting RTR. For adaptive methods, we compute RTR post-hoc by counting the visual tokens actually consumed per sample and averaging over the dataset.

Latency: When reporting wall-clock time (Fig. 4), we measure end-to-end per-sample latency, including both turns for methods that invoke the crop tool.

#### 4.2 Datasets

We curated a diverse training set comprising 10K samples from each of five publicly available training sets: ChartQA [17], DocVQA [18], TextVQA [21], LLaVA-Multi [10], and VisionThink-Smart [19]. We also collected 2k samples with the same distribution as a validation set. The SFT phase using a subset of the training set (5k samples from each dataset), while the GRPO uses all collected examples. This mixture spans both document understanding and natural image domains.

For evaluations, we conduct a comprehensive evaluation across six benchmarks that span diverse visual understanding capabilities.

For natural image understanding, we evaluate on RealWorldQA [25], which tests real-world spatial understanding capabilities through questions about everyday scenes, and POPE [14], which specifically measures object hallucination by probing whether models accurately identify the presence or absence of objects in images. Additionally, we include V ∗-Bench [24] for evaluating visual search capabilities, which measures the model’s ability to locate and reason about specific visual details within high-resolution images containing abundant and complex visual information.

For document understanding, we assess performance on ChartQA [17], which evaluates the ability to answer complex reasoning questions involving logical and arithmetic operations over data presented in charts and graphs; DocVQA [18], which tests comprehension of diverse document types including forms, tables, letters, memos, and handwritten text; and OCRBench [16], which provides a comprehensive assessment of text recognition and text-centric visual reasoning.

Together, this mix of benchmarks provides a holistic assessment of vision-language model capabilities.

#### 4.3 Implementation Details

We conduct experiments based on Qwen2.5-VL-7B-Instruct [2]; all compared methods are built on the same base VLM to isolate the impact of efficiency mechanisms.

Unless otherwise specified, each sample is first processed at a low-resolution setting

Ilow has hight and width devided by 2, corresponding to the “LR” baseline in Table 1 (RTR=0.25). When a crop is requested, the crop image(s) are rendered at the native high-resolution token density of the base model.

We use the discrete crop set C from §3.2 (four quadrants, four half-image regions, a center crop, and the full image). Requested crops are appended to the context together with Ilow.

For training, we use HuggingFace TRL [23] for SFT and GRPO, adapting TRL’s GRPO trainer to support multi-image and multi-turn conversations. For the SFT phase we use a total batch size of 16 with learning rate 1×10−4 and LoRA rank 8. wt=5 only for the tool-call turn and 1 otherwise. For the GRPO phase we use a total batch size of 64 with G=8 generations per sample, learning rate 1×10−5, and LoRA rank 8. αmiss=2, αuse=0.25 and λ=0.01.

Table 1: Main results across vision-language benchmarks. We compare AwaRes against fixed-ratio efficient methods (VisionZIP, SparseVLM, Holo-V) and adaptive baselines (VisionThink). Retain Token Ratios in parentheses denote the fraction of visual tokens retained. AwaRes matches the full-resolution baseline (Qwen2.5-VL-7B) while using only 36% of computational resources, outperforming all efficient alternatives. Qwen2.5-VL-7B-LR indicates the base model performance with low-res images. Best results per column in bold.

ChartQA DocVQA OCRBench POPE RealWorld V∗ Bench Average Model Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓

Qwen2.5-VL-7B 79.80 1.00 94.00 1.00 81.10 1.00 87.87 1.00 68.80 1.00 71.20 1.00 80.46 1.00 Qwen2.5-VL-7B-LR 65.00 0.25 91.00 0.25 70.70 0.25 84.41 0.25 66.00 0.25 63.20 0.25 73.39 0.25

Holo-V (50%) 62.04 0.50 70.77 0.50 68.20 0.50 86.54 0.50 67.19 0.50 64.40 0.50 69.86 0.50 Holo-V (70%) 69.32 0.70 76.40 0.70 72.40 0.70 87.37 0.70 68.89 0.70 69.11 0.70 73.92 0.70 SparseVLM (50%) 73.20 0.50 83.60 0.50 75.60 0.50 85.50 0.50 68.40 0.50 54.45 0.50 73.46 0.50 SparseVLM (70%) 75.80 0.70 87.20 0.70 79.30 0.70 85.40 0.70 68.50 0.70 54.45 0.70 75.11 0.70 VisionZIP (50%) 74.76 0.50 89.39 0.50 69.20 0.50 87.56 0.50 66.01 0.50 66.49 0.50 75.57 0.50 VisionZIP (70%) 76.72 0.70 90.75 0.70 72.70 0.70 87.86 0.70 64.84 0.70 65.97 0.70 76.47 0.70 VisionThink 79.90 1.15 90.35 0.32 80.10 0.83 86.70 0.34 66.60 0.55 71.73 0.49 79.23 0.61

AwaRes 80.64 0.32 94.43 0.28 81.30 0.42 85.73 0.27 68.50 0.43 71.20 0.42 80.30 0.36

#### 4.4 Baselines

We compare AwaRes against two classes of efficiency approaches, all models are based on Qwen-2.5-VL-7B [2].

Fixed-budget token pruning/compression: VisionZip [27], SparseVLM [30], and Holo-V [32], which are training-free inference-time methods, configured to retain either 50% or 70% of full-resolution visual tokens.

Adaptive resolution escalation: VisionThink [19], which performs a low-resolution pass and optionally escalates to the full high-resolution image. In contrast, AwaRes escalates spatially by requesting only a subset of high-resolution crops.

#### 4.5 Main Results

- Table 1 reports task performance and efficiency. AwaRes achieves an average score of 80.30, matching the full-resolution baseline (80.46) while using only 0.36× the visual tokens on average. This indicates that the learned coupled-decision policy answers directly from the low-resolution overview when possible, and invokes targeted crops only when necessary.

Compared to fixed-budget pruning methods, AwaRes consistently attains higher accuracy at comparable or lower RTR. For example, at 70% retained tokens, VisionZip trails AwaRes by over 4 percent on average (76.47 vs. 80.30), reflecting the limitation of a global, sample-agnostic token budget.

Against the adaptive escalation baseline VisionThink [19], AwaRes improves both accuracy (80.30 vs. 79.23) and efficiency (RTR 0.36 vs. 0.61). Notably, on ChartQA AwaRes slightly exceeds the full-resolution baseline (+0.84 percent) while reducing RTR to 0.32, whereas VisionThink often performs two visual passes and exceeds baseline compute (RTR=1.15).

Beyond Token Efficiency: Inference Latency Although RTR captures the dominant visual compute, end-to-end latency also depends on the number of autoregressive text

- Table 2: Agreement on resolution-selection labels. Confusion matrix comparing labels produced by LLaMA-3.370B against DeepSeek-V3.2 and ANLS. We observe high agreement with DeepSeek-V3.2, and low agreement with ANLS metric. Values are reported as percentages (%).

100

DocVQA

95

DocVQA

90

POPE POPE

85

Accuracy(%)

ChartQA ChartQA

OCRBench OCRBench

80

75

V* Bench V* Bench

70

RealWorldQA RealWorldQA

65

LLaMA LR HR

VisionThink

AwaRes

60

1 2 3 4 5

Wall Clock Time (seconds)

LR 78.01 1.39 HR 1.73 18.87

DeepSeek

Fig.4: Performance vs. Wall Clock Time. AwaRes achieves sub-second average latency across all benchmarks by encoding resolution decisions in short tool calls, whereas VisionThink’s explicit reasoning traces increase decoding time (e.g., 4.3s vs. 0.6s on ChartQA).

LR 69.04 9.19 HR 10.69 11.07

ANLS

tokens generated. VisionThink decides whether to escalate resolution via elaborated reasoning traces, which can substantially increase decoding cost even when the final answer is short. In contrast, AwaRes encodes the decision in a short, structured tool call without generating intermediate reasoning steps, and reuses the KV cache between turns.

As shown in Fig. 4, AwaRes achieves sub-second average latency across benchmarks while maintaining competitive or superior accuracy. On ChartQA, VisionThink averages 4.3 seconds per sample, whereas AwaRes averages 0.6 seconds under the same decoding setup (see supplementary for hardware and measurement details).

From Over-Calling to Looking Where It Matters Fig. 5 illustrates how the model’s first-turn crop-request policy evolves from oracle supervision (Oracle GT) through SFT to GRPO. The SFT designed to introduce the tool-calling protocol, and as so, the policy becomes conservative with respect to accuracy and therefore over-invokes the crop tool: the probability of taking the no-call action (LR) drops sharply (80.8%→46.3%), while escalation to the full-image crop (“All”) is substantially inflated (3.4%→16.6%). This behavior is consistent with imitation-style training that prioritizes adhering to the tool protocol, even in cases where the low-resolution view is sufficient.

GRPO then reshapes the same policy under the explicit accuracy–efficiency objective in Eq. 5; as a result, the policy shifts toward selective tool use: LR increases to 72.2% and “All” decreases to 4.9%, approaching the oracle distribution. Due to the tool-use cost that penalize crop size, the learned policy may shift towards efficient strategies that differ from the oracle annotations.

#### 4.6 Ablations

We perform extensive ablation to test our method, We start with ablations on the data preparation pipeline, supervised fine-tuning phase (SFT), and the GRPO stage.

[Figure 18]

Fig.5: From Over-Using to Looking Where It Matters. The flow of crop selection decisions from Oracle GT (left), SFT-tuned model predictions (middle), and GRPO predictions (right). SFT, designed to introduce the tool-calling protocol, tends to over-use the crop tool as it learns the mechanics of tool usage. The GRPO phase corrects this behavior through the tool-use cost, increasing low-resolution decisions while exploring alternative crop strategies that balance accuracy and efficiency.

Data preparation To assess whether our data-preparation pipeline is overly sensitive to the choice of LaaJ, we replace the default judge (LLaMA-3.3-70B [5]) with DeepSeek-V3.2 [15] and measure agreement on the resolution-selection label. Table 2 reports the resulting confusion matrix.

We observe strong consistency between the two LaaJ models: the agreement (sum of diagonal entries) is 96.88%, suggesting that our automatic labeling procedure is not driven by a bias of a specific model. In contrast, agreement with ANLS-based labeling is substantially lower (80.11%) which indicates systematic label shifts. Moreover, training πθ

using ANLS-based labels degrades average performance by 2.8 points across benchmarks. This divergence supports using LaaJ for semantic correctness judgments in our setting, while ANLS, being string-oriented, can over-penalize semantically correct but paraphrased answers. Further details and analyses are provided in the supplementary material.

ref

Tool introduction Table 3 studies how different SFT recipes affect the emergence of reliable crop-tool usage. The default training uses standard token-level SFT where each assistant message is optimized independently under the same objective, while Traj performs trajectory-level SFT by optimizing the entire two-turn tool-use interaction as a single sample. Phased denote a 2-phase training where the first phase uses only the tool-call turns. Joint training improves both average performance as well as efficiency compared to split training (77.90 vs. 75.15 and 0.43 vs. 0.66 respectively). Increasing the

##### Table 3: Cold-start ablations. wt follows Eq. 4. We report Accuracy ↑ and RTR ↓ across all benchmarks.

Components ChartQA DocVQA OCRBench POPE RealWorld V∗ Bench Average Traj. Phased Ups. wt Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓

x x x 1 72.9 0.35 93.1 0.42 73.7 0.32 85.8 0.29 64.1 0.33 61.3 0.42 75.15 0.36 v x x 1 69.8 0.38 93.8 0.32 76.9 0.54 87.8 0.30 70.5 0.55 68.6 0.48 77.90 0.43 v x v 1 76.9 0.30 93.2 0.27 72.9 0.41 84.7 0.25 69.0 0.43 66.0 0.55 77.12 0.37 v v x 1 70.9 0.36 93.4 0.40 76.1 0.32 86.6 0.30 68.8 0.34 64.4 0.41 76.70 0.36 v v v 1 70.8 0.33 92.7 0.37 72.7 0.31 85.7 0.28 66.8 0.33 63.9 0.43 75.43 0.33

v x x 1 69.8 0.38 93.8 0.32 76.9 0.54 87.8 0.30 70.5 0.55 68.6 0.48 77.90 0.43 v x x 5 77.0 0.42 94.0 0.35 78.8 0.61 88.0 0.32 69.7 0.64 70.7 0.60 79.70 0.49

tool-turn weight wt further improves accuracy (Joint wt=5: 79.70) at the cost of higher RTR (0.49), suggesting that stronger emphasis on the first-turn decision encourages more confident tool invocation. Among the recipe variants, Upsampled settings reduce RTR (down to 0.33 on average) but also reduce performance, reflecting the expected accuracy–efficiency trade-off. Based on this ablation, we use joint trajectory-level SFT with wt=5 as the default initialization for the subsequent GRPO stage.

Tool Optimization rewards We analyze the GRPO tool optimization phase by ablating the reward components. We optimize for CDP using a positive reward for correct answers based on textual similarity from a sentence-transformer, combined with a tool-use cost.

We evaluate our design choices by training the tool optimization step while systematically removing each reward component. Table 4 summarizes our analysis. Removing the crop area cost raises RTR from 0.36 to 0.42 (λ=0), and removing the crop cost entirely increases it further to 0.51. The limited increase in RTR when remvoing the tool cost entirely can be attributed to both the SFT model initialization and KL regularization, which keeps the model close to the reference model that already has a solid foundation for tool usage.

Additionally, we observe that correctness rewards produced by ANLS and LLM-asa-judge yield similar results. This is because we optimize on short answers rather than lengthy reasoning traces. In this setting, sentence-transformer similarity provides an effective balance between semantic understanding - which ANLS cannot capture and evaluation robustness, where LLM judges may be less reliable.

The effect of cold start initialization An alternative to our two-stage approach is to apply Reinforcement Learning with the base model as a reference policy, allowing it to learn tool usage from scratch during the GRPO phase. We compare this strategy against our approach, which first introduces the tool via supervised fine-tuning and then optimizes its usage with GRPO.

We trained the base model with GRPO alone for 3 epochs (denoted GRPO only in Tab. 4) on the same training set as AwaRes. Using just RL, the model barely uses the tool, relying on low-resolution processing and achieving lower accuracy. In contrast, AwaRes leverages SFT to establish reliable tool-use behavior, then refines it with GRPO to balance accuracy and efficiency - resulting in higher average accuracy

- Table 4: GRPO Ablations. Top: comparing training the model only for the πθref (SFT) vs only GRPO (no cold start). Middle: effect of the tool cost penalty, λ = 0 removes the area-based cost while retaining the misuse penalty, and removing the tool cost entirely leads to further over-cropping. Bottom: comparison of correctness reward functions. Text similarity (sentence-transformer) balances semantic matching and robustness, outperforming both ANLS and LLM-as-a-judge on short-answer evaluation. RTR denotes the fraction of full-resolution compute used (↓ is better).

ChartQA DocVQA OCRBench POPE RealWorldQA V∗ Bench Average Model Acc. ↑ RTR ↓ Acc. ↑ RTR ↓ Acc. ↑ RTR ↓ Acc. ↑ RTR ↓ Acc. ↑ RTR ↓ Acc. ↑ RTR ↓ Acc. ↑ RTR ↓

SFT only 77.0 0.42 94.0 0.35 78.8 0.61 88.0 0.32 69.7 0.64 70.7 0.60 79.70 0.49 GRPO only 76.48 0.25 93.98 0.25 72.20 0.31 85.91 0.25 67.06 0.32 67.02 0.54 77.11 0.31

w/o tool cost 80.60 0.44 94.60 0.37 81.50 0.64 87.80 0.34 69.40 0.65 71.20 0.62 80.85 0.51 w/o area cost (λ=0) 81.28 0.38 94.42 0.37 80.70 0.40 85.93 0.37 69.28 0.41 71.20 0.58 80.47 0.42

Text-Sim→ ANLS 79.84 0.29 94.35 0.25 80.00 0.41 85.78 0.23 69.67 0.43 70.16 0.57 79.96 0.37 Text-Sim→ LaaJ 79.00 0.29 94.20 0.25 80.30 0.42 85.80 0.24 69.40 0.44 70.70 0.58 79.90 0.37

- AwaRes 80.64 0.32 94.43 0.28 81.30 0.42 85.73 0.27 68.5 0.43 71.20 0.41 80.30 0.36

across benchmarks. Using SFT only may yields good accuracy, but high RTR due to over-usage of the tool-call (Tab. 4). Moreover, the SFT can be a double edged sword, as observed on external validation set in Fig. 5, the SFT model overfitted to the training data, and as a result have a significant shift in policy. That makes the GRPO not only exploratory, but also recover from overfitting to training data.

- 5 Conclusion

We presented AwaRes, a spatial-on-demand inference framework for VLMs that preserves a low-resolution global view and selectively retrieves only the high-resolution crops required for a given query through a simple tool-calling interface. This design directly targets the practical bottleneck of high-resolution VLM inference while remaining deployment-friendly via multi-turn KV-cache reuse.

To train AwaRes without manual spatial supervision, we introduced an automatic data curation pipeline that (i) labels whether additional resolution is necessary by comparing low- vs. full-resolution predictions with an LLM judge, and (ii) localizes the supporting evidence with an oracle grounding model to create multi-turn crop-request trajectories. We then train in two stages: a cold-start SFT phase that yields a supervised reference policy, followed by multi-turn GRPO that optimizes the accuracy–efficiency trade-off using a composite reward combining semantic answer correctness with explicit crop-usage penalties.

Across six benchmarks spanning document understanding and general visual QA, AwaRes matches full-resolution performance on average while using substantially fewer visual tokens, and improves end-to-end efficiency relative to global resolution-escalation baselines. We believe spatial-on-demand crop acquisition provides a practical path toward high-detail multimodal reasoning under tight compute and latency budgets, and opens the door to richer multi-step perception strategies that allocate resolution progressively as needed.

Future work may explore extending the crop selection from a discrete set to continuous bounding box predictions, enabling finer-grained spatial control. Additional promising direction is generalizing spatial-on-demand perception to video understanding, where temporal sparsity offers additional efficiency gains.

### References

- 1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)
- 2. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. ArXiv abs/2502.13923 (2025), https://api.semanticscholar.org/CorpusID:276449796
- 3. Chen, J., Ye, L., He, J., Wang, Z.Y., Khashabi, D., Yuille, A.: Efficient large multi-modal models via visual context compression. In: NeurIPS (2024)
- 4. Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., Chang, B.: An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In: CVPR (2024)
- 5. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)
- 6. Guo, J., Zhai, F., Jian, P., Wei, Q., Zhou, Y.: Crop: Contextual region-oriented visual token pruning (2025)
- 7. Huang, K., Zou, H., Xi, Y., Wang, B., Xie, Z., Yu, L.: Ivtp: Instruction-guided visual token pruning for large vision-language models. In: ECCV. pp. 214–230 (2024)
- 8. Huang, Z., Wang, S., Zhang, W., Zhao, B., Lin, W.: Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In: CVPR (2025)
- 9. Jeddi, A., Baghbanzadeh, N., Dolatabadi, E., Taati, B.: Similarity-aware token pruning: Your vlm but faster (2025)
- 10. Jiang, D., He, X., Zeng, H., Wei, C., Ku, M., Liu, Q., Chen, W.: Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483 (2024)
- 11. Khaki, S., Guo, J., Tang, J., Yang, S., Chen, Y., Plataniotis, K.N., Lu, Y., Han, S., Liu, Z.: Sparsevila: Decoupling visual sparsity for efficient vlm inference. In: ICCV. pp. 23784–23794 (October 2025)
- 12. Kimhi, M., Shabtay, N., Giryes, R., Baskin, C., Schwartz, E.: Cares: Context-aware resolution selector for vlms (2025)
- 13. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., Stoica, I.: Efficient memory management for large language model serving with pagedattention. In: Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles (2023)
- 14. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., Wen, J.R.: Evaluating object hallucination in large vision-language models. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 292–305. Association for Computational Linguistics (2023)
- 15. Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al.: Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556 (2025)
- 16. Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.C., Liu, C.L., Jin, L., Bai, X.: OCRBench: On the hidden mystery of OCR in large multimodal models. Science China Information Sciences 67(12) (2024)
- 17. Masry, A., Long, D.X., Tan, J.Q., Joty, S., Hoque, E.: ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In: Findings of the Association for Computational Linguistics: ACL 2022. pp. 2263–2279. Association for Computational Linguistics (2022)

- 18. Mathew, M., Karatzas, D., Jawahar, C.: DocVQA: A dataset for VQA on document images. In: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 2200–2209 (2021)
- 19. Senqiao, Y., Junyi, L., Xin, L., Jinming, W., Wei, L., Zejun, M., Bei, Y., Hengshuang, Z., Jiaya, J.: Visionthink: Smart and efficient vision language model via reinforcement learning. In: NeurIPS (2025)
- 20. Shen, H., Zhao, K., Zhao, T., Xu, R., Zhang, Z., Zhu, M., Yin, J.: Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (2025)
- 21. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards vqa models that can read. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)
- 22. Vasu, P.K.A., Faghri, F., Li, C.L., Koc, C., True, N., Santhanam, G.K., Antony, A., Gabriel, J., Grasch, P., Tuzel, O., Pouransari, H.: Fastvlm: Efficient vision encoding for vision language models. In: CVPR (June 2025)
- 23. von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., Gallouédec, Q.: TRL: Transformers Reinforcement Learning (2020), https://github.com/huggingface/trl
- 24. Wu, P., Xie, S.: V?: Guided visual search as a core mechanism in multimodal llms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13084–13094 (2024)
- 25. xAI: Realworldqa. https://x.ai/blog/grok-1.5v (2024), accessed: 2024
- 26. Xing, L., Huang, Q., Dong, X., Lu, J., Zhang, P., Zang, Y., Cao, Y., He, C., Wang, J., Wu, F., Lin, D.: Conical visual concentration for efficient large vision-language models. In: CVPR. pp. 14593–14603 (June 2025)
- 27. Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., Jia, J.: Visionzip: Longer is better but not necessary in vision language models. In: CVPR. pp. 19792–19802 (2025)
- 28. Ye, W., Wu, Q., Lin, W., Zhou, Y.: Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. Proceedings of the AAAI Conference on Artificial Intelligence (2025)
- 29. Zhang, K., Li, B., Zhang, P., Pu, F., Cahyono, J.A., Hu, K., Liu, S., Zhang, Y., Yang, J., Li, C., Liu, Z.: LMMs-Eval: Reality check on the evaluation of large multimodal models. In: Findings of the Association for Computational Linguistics: NAACL 2025. pp. 881–916. Association for Computational Linguistics (2025)
- 30. Zhang, Y., Fan, C.K., Ma, J., Zheng, W., Huang, T., Cheng, K., Gudovskiy, D., Okuno, T., Nakata, Y., Keutzer, K., Zhang, S.: Sparsevlm: Visual token sparsification for efficient vision-language model inference. In: ICML (2025)
- 31. Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., Yu, X.: Deepeyes: Incentivizing "thinking with images" via reinforcement learning. In: ICLR (2026)
- 32. Zou, X., Lu, D., Wang, Y., Yan, Y., Lyu, Y., Zheng, X., Zhang, L., Hu, X.: Don’t just chase "highlighted tokens" in mllms: Revisiting visual holistic context retention. In: NeurIPS (2025)

## Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs Supplementary Materials

Nimrod Shabtay*1,2, Moshe Kimhi*1,3, Artem Spector1, Sivan Haray1, Ehud Rivlin3, Chaim Baskin4, Raja Giryes2, and Eli Schwartz1

- 1 IBM Research
- 2 Tel-Aviv University 3 Technion

4 Ben-Gurion University

This supplementary document provides additional details, analyses, and visual examples that complement the main paper. We organize the material as follows: Section 3.2 presents supplementary visual examples from our data curation pipeline, including both successful annotations and failure cases of the automatic process. Section 3 details our latency analysis, including hardware setup, wall-clock measurements comparing AwaRes and VisionThink, and a response length analysis explaining the observed efficiency gains. Section 4 offers an in-depth analysis of the cold-start (SFT) stage, examining the coupled-decision policy (CDP) behavior, sensitivity to the tool-turn weight wt, and toolcall formatting reliability. Section 5 analyzes alternative data curation strategies using ANLS-based filtering. Section 6 provides comprehensive training hyperparameters and configuration details. Finally, Section 7 includes the complete prompts used throughout our pipeline: the LLM-as-a-Judge prompt for data curation, the oracle localization prompt for grounding, and the system prompt used during SFT, GRPO, and inference.

### 1 Data Annotation

- 1.1 Data Curation Pipeline

In this section, we provide in Figure 1 supplementary visual examples from our data curation pipeline. Additionally, in Figure 2 we illustrate failure cases where the automatic process did not produce satisfactory results.

- 1.2 Training data statistics

Figure 3 shows the resolution distribution of our training data. DocVQA [18] exhibits the highest resolutions, whereas the LLaVA-Multi [10] subset contains the lowest. VisionThink-Smart [19] and ChartQA [17] display a broad spread across resolutions, while TextVQA concentrations cluster around 1000 pixels along one axis. We cap all resolutions at 2000×2000, as prior work [12] demonstrated that truncating DocVQA resolution has negligible impact on performance.

### 2 Benchmark Qualitative Examples

Figures 4 - 9 present positive examples where AwaRes successfully identifies and crops the image region relevant to the question. Each conversation shows the question, the

Prompt:

Which tour topped the list with over 558 thousand tickets sold?

Ground Truth Answer:

Trans-Siberian Orchestra

Low Resolution Oracle Detection Boxes Selected Crop

[Figure 19]

[Figure 20]

[Figure 21]

Prompt:

What is the brand name on the screen?

Ground Truth Answer:

spesial

Low Resolution Oracle Detection Boxes Selected Crop

[Figure 22]

[Figure 23]

[Figure 24]

- Fig. 1: Visual examples from our data curation pipeline. Each row shows the low-resolution input image, the oracle-detected bounding boxes (question region in blue, answer region in red), and the selected crop used for training. Top: A Chart example where the oracle correctly localizes the relevant bar and its label. Bottom: A natural image example requiring OCR of a brand banner.

tool call selected by the model, the predicted answer (matching the ground truth), the low-resolution input, and the retrieved high-resolution crop. Across charts, documents, and natural images, the retrieved high-resolution crops consistently isolate the taskrelevant objects or text - such as axis labels, table cells, brand logos, and foreground subjects, enabling the model to extract fine-grained details that are unresolvable in the low-resolution overview alone.

#### 2.1 Failure cases

Figure 10 shows representative failure cases where the model’s either crop incorrect region, or correct crop that lead to a wrong answer. Each conversation shows the question, the tool call selected by the model, the predicted answer (in red) and the ground truth answer from the data, the low-resolution input, and the retrieved high-resolution crop.These

failures typically arise when the question-relevant detail occupies a small or ambiguous portion of the image, causing the crop to capture nearby but insufficient context.

- 3 Latency Analysis

- 3.1 Hardware and Measurement Setup

To compare VisionThink [19] and AwaRes, we evaluate both methods using their respective evaluation implementations in lmms-eval [29], with both models running in native HuggingFace configuration. We measure wall-clock time (WC) from the start of generation until the final answer is produced (encompassing both turns when applicable).All measurements done on Nvidia-H100-80GB GPU.

Prompt:

What was the market share of ZDF among 14 to 49 year olds?

Ground Truth Answer:

- 6.4

Low Resolution Oracle Detection Boxes Selected Crop

[Figure 25]

[Figure 26]

[Figure 27]

Prompt:

What country had the highest number of new HIV infections in 2015?

Ground Truth Answer:

South Africa

Low Resolution Oracle Detection Boxes Selected Crop – Full Image

[Figure 28]

[Figure 29]

[Figure 30]

- Fig. 2: Failure cases from the automatic data curation pipeline. Top: The oracle grounding model incorrectly localizes the answer region, missing the relevant bar (ZDF’s 14-49 age group value). However, the crop regions helps mitigate such localization errors—the selected crop still contains the correct answer. Bottom: The oracle correctly identifies the question and answer regions but the bounding boxes span nearly the entire image, resulting in a full-image crop selection. While this produces correct supervision, it is inefficient as no resolution savings are achieved.

count

2000

1

4007 8014

1800

1600

1400

Height(px)

1200

1000

800

chartqa docvqa

| |
|---|

llava textvqa VTsmart

600

| |
|---|

| |
|---|

400

600 800 1000 1200 1400 1600 1800 2000 Width (px)

- Fig. 3: Resolution distribution of training datasets. Each point represents an image’s width and height in pixels.

#### 3.2 Detailed Latency Results

- Table 1 presents the performance and latency (WC) comparison between AwaRes and VisionThink across six benchmarks. AwaRes achieves lower latency on all benchmarks while maintaining competitive or superior accuracy. On average, AwaRes reduces wall-clock time by 4.4× (from 2.71s to 0.61s) compared to VisionThink, while improving the average metric score from 79.23 to 80.47. The efficiency gains are most pronounced on ChartQA (7.7× faster) and OCRBench (5.3× faster), where VisionThink’s extended reasoning traces incur substantial overhead.

#### 3.3 Response Length Comparison

The latency differences stem primarily from response length. As shown in Table 2, VisionThink generates substantially longer reasoning traces, approximately 5.8× to

##### Table 1: Comparison of dynamic methods (AwaRes and VisionThink)

ChartQA DocVQA OCRBench POPE RealWorldQA V∗ Bench Average Model Acc.↑ WC↓ ANLS↑ WC↓ Acc.↑ WC↓ Metric↑ WC↓ Acc.↑ WC↓ Acc.↑ WC↓ Metric↑ WC↓ VisionThink 79.9 4.32 90.35 1.78 80.10 3.36 86.70 1.23 66.60 2.31 71.73 3.24 79.23 2.71

- AwaRes 81.3 0.56 94.40 0.51 80.70 0.64 85.9 0.50 69.30 0.66 71.20 0.81 80.47 0.61

28.8× more characters than AwaRes. This verbosity directly translates to increased generation time, as autoregressive decoding scales linearly with output length.

Beyond raw efficiency, AwaRes exhibits significantly lower variance in response length across all benchmarks. This predictability has practical implications: estimated completion times become more reliable, enabling better resource allocation and user experience. In contrast, VisionThink’s high standard deviations—often exceeding the mean—make response length and latency highly unpredictable, complicating deployment in latency-sensitive applications.

- Table 2: Response verbosity comparison between VisionThink and AwaRes across the benchmarks. We report the mean ± std number of characters in model responses. AwaRes produces substantially shorter responses than VisionThink (5.8×–28.8× reduction), reflecting its efficiency beyond visual token savings.

Benchmark samples VisionThink AwaRes Ratio ChartQA 2500 89.46±205.71 6.54±6.16 13.7× DocVQA (val) 5349 79.95±242.21 13.74±12.79 5.8× OCRBench 1000 178.67±540.12 16.23±16.39 11.0× POPE 9000 46.46±155.52 2.84±2.33 16.4× RealWorldQA 765 160.67±261.55 5.57±5.86 28.8× V*Bench 191 118.54±232.17 5.29±5.78 22.4×

### 4 Additional Cold-Start (SFT) Analysis

This supplementary section provides additional evidence that the cold-start stage learns a coupled-decision policy (CDP) whose first-turn action jointly determines when to request additional resolution (C =∅ vs. C ≠ ∅) and, when escalating, where to look via the chosen crop subset C⊆C. Beyond final-task accuracy, we therefore report policy-centric diagnostics that quantify (i) the tendency to call the crop tool, (ii) failure to escalate when detail is required, and (iii) the amount of high-resolution evidence requested.

#### 4.1 Metrics for CDP diagnostics

For each evaluated sample with resolution-sufficiency label y ∈ {LR,HR}, the model either takes a no-call action (C=∅) or requests one or more crops (C≠ ∅). We evaluate the CDP as two coupled components:

Call decision (when to crop). We treat tool invocation as a binary classifier where the positive class is y=HR and a prediction is positive iff C≠ ∅. We report:

- – Call Precision ↑: P(y=HR|C≠ ∅).
- – Call Recall ↑: P(C≠ ∅|y=HR).
- – Call F1 ↑: 2Prec×Rec/(Prec+Rec).
- – FPR (LR-call) ↓: P(C≠ ∅|y=LR).

Region decision (where to look | call). Conditioned on C≠ ∅, we measure overlap between the requested crop set and the oracle target crops C⋆ used for SFT supervision. We report:

- – Exact match (IoU=1) ↑: the predicted region matches an oracle target exactly.
- – Relaxed match (IoU≥0.25) ↑: the predicted region overlaps an oracle target by at least 0.25 IoU, accounting for the hierarchical structure of C (e.g., for a quadrant target, its two adjacent half-image regions are also considered acceptable; likewise, All and Center may be acceptable when they satisfy the IoU threshold).
- – Avg. area ↓: E[s(C)], the average fraction of image area requested when calling.

In addition, we report Accuracy (↑) and RTR (↓) across all benchmarks, consistent with the main paper.

#### 4.2 CDP behavior across cold-start variants

- Table 3 decomposes cold-start behavior into the two components of the coupled-decision policy (CDP): the call decision (when to request crops) and the region decision (where to look, conditioned on calling).

Call decision (when to crop). Trajectory-level SFT substantially improves the calibration of the first-turn decision relative to baseline SFT. While baseline SFT achieves moderate precision (62.49) it exhibits very low recall (15.34) and a high false-positive rate on LR samples (79.69), indicating that tool invocation is both unreliable on HR cases and overly frequent on LR cases. Trajectory-level SFT increases recall to 41.02 and reduces FPR to 63.33, yielding a higher overall F1 score (24.63→44.56). Further upweighting the tool-call turn (wt=5) strengthens this behavior: precision rises to 77.8, recall to 47.70, and FPR drops to 49.85, improving F1 to 59.14. Together, these results indicate that emphasizing the first-turn action stabilizes the fused CDP decision of whether to escalate.

Region decision (where to look | call). Trajectory-level SFT also improves alignment with oracle supervision, increasing exact region match (IoU=1) from 13.8 to 15.9 and relaxed overlap match (IoU≥ 0.25) from 32.6 to 48.85, while reducing the average requested area from 0.59 to 0.402. Upweighting the tool-call turn yields a large jump in localization quality (IoU= 1: 41.3; IoU≥ 0.25: 75.5), with a modest increase in requested area (0.402→0.463). This suggests that stronger supervision of the tool-call turn improves not only the decision to escalate but also the selection of informative regions once escalation occurs.

Overall, improved cold-start recipes shape both parts of the CDP: they improve the reliability of escalation (when) and the quality of evidence localization (where), while keeping the requested high-resolution area controlled. This motivates the subsequent GRPO stage, which further refines the same fused CDP under an explicit accuracy–efficiency objective.

#### 4.3 Cold-start sensitivity to data and parameters

Table 4 summarizes how common cold-start knobs shape both downstream performance and the induced coupled-decision policy (CDP). Trajectory-level SFT improves average accuracy over vanilla (77.90 vs. 75.15), but it also shifts the policy toward more frequent crop requests (call rate 22.55%→25.21%) while substantially reducing the average requested area (0.59→0.402), indicating that the model learns to localize with smaller crops rather than relying on large regions. Increasing the tool-turn weight to wt=5 further strengthens the first-turn CDP action, yielding the best accuracy (79.70) but also the highest RTR (0.49), consistent with a policy that escalates more often (call rate 29.02%) and requests slightly larger regions (Avg. area 0.463).

Beyond wt, data and schedule choices can trade off calibration and efficiency. In particular, HR upsampling is expected to reduce missed escalations by exposing the policy to more detail-critical instances, but may increase tool usage if applied aggressively; phased (tool-first) schedules can stabilize tool invocation but sometimes change how strongly the policy couples region selection to answer quality. Overall, the best cold-start configuration is the one that yields a reliable CDP (low misses and good region selection) while keeping call rate and requested area controlled, leaving GRPO to fine-tune efficiency rather than compensating for frequent cold-start failures.

#### 4.4 Tool-call formatting reliability

Finally, Table 5 quantifies tool-call formatting reliability after cold-start by measuring whether a generated tool call can be parsed into a valid crop subset C⊆C without any post-processing. This isolates protocol learning from downstream accuracy: malformed outputs can prevent escalation even when the policy intends to call the tool, effectively breaking the CDP at the interface level.

Trajectory-level SFT substantially improves tool-call validity, reducing corruption from 10.17% to 1.43%. Moreover, the remaining failures are not merely cosmetic: a large fraction of corrupted calls lead to incorrect crop requests even after simple recovery (5.48% for baseline SFT vs. 0.94% for Traj.). Upweighting the tool-call turn (wt=5) eliminates formatting corruption entirely (100% valid parse), supporting our design choice to explicitly emphasize the first-turn action during cold-start.

- Table 3: Cold-start (SFT) policy diagnostics for the coupled-decision policy (CDP). We evaluate the call decision as a binary classifier where the positive class is y=HR and a prediction is positive iff the model calls the tool (C≠ ∅). We report precision, recall, and their harmonic mean (F1), as well as the false-positive rate on LR samples (FPR=P(C≠ ∅|y=LR)). For the region decision (conditioned on calling), we report overlap-based match rates to oracle targets using exact region match (IoU=1) and relaxed overlap (IoU≥0.25), along with the average requested area. Higher is better (↑) unless noted.

CDP: call decision CDP: region decision | call Model Call Prec.↑ Call Rec.↑ Call F1↑ FPR (LR-call)↓ IoU=1↑ IoU≥0.25↑ Avg. area↓ SFT (baseline) 62.49 15.34 24.63 79.69 13.8 32.6 0.59 SFT + Traj. 48.75 41.02 44.56 63.33 15.9 48.85 0.402 SFT + Traj. + wt=5 77.8 47.70 59.14 49.85 41.3 75.5 0.463

- Table 4: Sensitivity of cold-start SFT. We vary data/objective knobs and report both downstream performance (Avg. Accuracy and Avg. RTR across all benchmarks) and induced CDP behavior (call rate and average requested area, both measured over evaluation prompts). Call rate is P(C≠ ∅) and Avg. area is E[s(C)] (fraction of image area requested when calling).

Setting HR upsample wt Traj. SFT Phased Acc↑ RTR↓ Call rate Avg. area↓ Default SFT x 1 x x 75.15 0.36 22.55% 0.59 Traj. only x 1 v x 77.90 0.43 25.21% 0.402 Traj. + high wt x 5 v x 79.70 0.49 29.02% 0.463 Traj. + HR upsample v 1 v x 77.12 0.37 23.0% 0.35 Phased (tool-first) x 1 v v 76.70 0.36 24.0% 0.44

- Table 5: Tool-call formatting reliability after cold-start. Valid parse indicates that the tool output can be parsed into a crop subset C ⊆C without post-processing. Corrupt outputs are split into recoverable cases (simple extraction of intended crop id(s) succeeds) and incorrect cases (post-processing yields a wrong crop).

Corrupt outputs (%)↓ Model Total Recoverable Incorrect

Valid parse (%)↑

SFT (baseline) 89.83 10.17 4.69 5.48 SFT + Traj. 98.57 1.43 0.49 0.94 SFT + Traj. + wt=5 100.00 0.00 0.00 0.00

### 5 ANLS-Based Data Curation Analysis

We compare annotating the crops using LaaJ vs. using ANLS in Tab. 6. ANLS is a string-oriented similarity measure designed for OCR-style exactness, and as such it is ill-suited for supervising our resolution-sufficiency labels, specifically since we observe the VLM answers with crops act as augmented input and thus perturbed the answers a bit (even when they are semantically correct).

This effect is reflected empirically. Training with ANLS-based labels for cold-start yields substantially lower average accuracy than LaaJ-based labels (75.9 vs. 79.70), with pronounced drops on text- and detail-sensitive benchmarks such as ChartQA (70.5 vs. 77.0) and OCRBench (70.0 vs. 78.8). While ANLS labeling attains a lower RTR (0.38 vs. 0.49), the accompanying accuracy degradation indicates that it often suppresses crop requests even when additional resolution is required, rather than producing a better-calibrated policy. Overall, LaaJ provides a more reliable semantic correctness signal for annotation, which is critical for learning the CDP without over-penalizing benign surface-form variation.

### 6 Training Details

We provide detailed hyperparameters for both training stages of AwaRes: Cold-Start SFT and Tool Optimization via GRPO. Table 7 conclude all the training parameters for both stages.

- Table 6: Comparing labeling strategies: LaaJ (LLaMA-3.3-70B) vs. ANLS. We report Accuracy ↑ and RTR ↓ across all benchmarks.

Label Strategy ChartQA DocVQA OCRBench POPE RealWorld V∗ Bench Average

Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓ Acc ↑ RTR ↓

ANLS 70.5 0.31 93.6 0.28 70.0 0.41 86.7 0.27 69.0 0.47 65.9 0.54 75.9 0.38 LLaMA-3.3-70B (LaaJ) 77.0 0.42 94.0 0.35 78.8 0.61 88.0 0.32 69.7 0.64 70.7 0.60 79.70 0.49

- Table 7: Training hyperparameters for Cold-Start SFT and Tool Optimization (GRPO) stages.

Parameter Cold-Start (SFT) Tool Optimization (GRPO) Model Configuration Base model Instruct HF model Cold-Start model LORA Rank 8 8 Optimization Optimizer AdamW AdamW Learning rate 1e-4 5e−5 LR schedule Cosine Linear Warmup ratio 0.05 0 Weight decay 0.001 0 Max gradient norm - 1.0 Batch Configuration Per-device batch size 16 8 Effective batch size 128 64 Number of epochs 2 1 Sequence Length

Max sequence / prompt length 8192 512 Max completion length 512 512

Stage-Specific Tool-turn weight wt 5 – Trajectory-level optimization True – Number of generations G – 8 Temperature τ – 1.0 Top-p sampling – 1.0 KL coefficient β – 0.05 Accuracy weight α – 10.0 Crop cost weight (1−α) – 0.25 Area weight (1−α) – 0.01

### 7 Prompts

#### 7.1 LLM-as-a-Judge Prompt LLM-as-Judge Prompt

You are tasked with evaluating which of two AI responses better answers the given question based on the provided ground truth. Original prompt: {prompt} Ground Truth Answer: {gt}

- Response 1: {lr_resp}
- Response 2: {hr_resp}

Compare both responses against the ground truth answer and respond with just one number, DO NOT include any other text:

- 0 - Both responses are equally good or equally bad, no significant difference
- 1 - Response 1 is better
- 2 - Response 2 is better

#### 7.2 Oracle Grounding Prompt Oracle Localization Prompt

You are an expert visual assistant. Your task is to analyze an image and determine which regions are relevant for both understanding and answering a given question. Instructions:

- 1. Analyze the question and identify what visual information is needed
- 2. Localize TWO types of regions:

- a) QUESTION region: The area containing visual elements mentioned in

or relevant to the question

- b) ANSWER region: The area containing the answer to the question

- 3. For each region type, provide a bounding box in normalized coordinates [x_min, y_min, x_max, y_max]

- Coordinates should be between 0 and 1000 (normalized to image dimensions)

- - x_min, y_min: top-left corner of the bounding box
- - x_max, y_max: bottom-right corner of the bounding box

- 4. Provide tight bounding boxes that closely encompass only the relevant text/visual elements
- 5. Output ONLY a Python dictionary with two keys: ’question’ and ’answer’

- NOTHING ELSE!! CRITICAL: Your entire response must be ONLY a Python dictionary. Do not include any explanation, reasoning, or other text. Output format:

{

’question’: [[x_min, y_min, x_max, y_max]], ’answer’: [[x_min, y_min, x_max, y_max]]

} Notes:

- - If the question and answer regions overlap significantly or are the same, you may provide the same coordinates for both
- - If multiple regions are needed for either question or answer, include multiple bounding boxes in the list
- - Keep bounding boxes as tight as possible around the relevant text or visual elements Your response must be ONLY: {’question’: [[x_min, y_min, x_max, y_max],

...], ’answer’: [[x_min, y_min, x_max, y_max], ...]} No other text, no explanation, just the dictionary. Question: ...

#### 7.3 SFT / GRPO / Inference Prompt

You are a vision-language model that analyzes images and answers questions about them. If the image resolution is too low for accurate analysis, respond with GET_CROPS: followed by a list of crop numbers in square brackets (e.g., GET_CROPS: [’3’] or GET_CROPS: [’0’, ’5’]), where the available crop numbers and their corresponding areas are {’0’: ’topleft’, ’1’: ’top-right’, ’2’: ’bottom-left’, ’3’: ’bottom-right’, ’4’: ’center’, ’5’: ’top’, ’6’: ’bottom’, ’7’: ’left’, ’8’: ’right’, ’all’: ’all’}, otherwise provide your answer.

[Figure 31]

[Figure 32]

[Figure 33]

##### Fig. 4: Positive examples of AwaRes’s adaptive cropping (1/6). The model isolates the relevant bar segments and axis labels to identify the smallest chart value, zooms into data points along a trend line to compute a median, and focuses on labeled chart categories to read a specific percentage, extracting precise numerical values that are indiscernible in the low-resolution overview alone.

[Figure 34]

[Figure 35]

[Figure 36]

##### Fig.5: Positive examples of AwaRes’s adaptive cropping (2/6). The crops target the top bar and its label to identify the subject with the highest ratio, zoom into line-chart trends and country labels to compare medians across countries, and focus on the top entries of a horizontal bar chart to read demographic rankings. In each case, the high-resolution crop brings the answer-relevant region into sharp focus.

[Figure 37]

[Figure 38]

##### Fig. 6: Positive examples of AwaRes’s adaptive cropping (3/6). The model crops into table cells of a scanned document to extract a specific weight value, and zooms into a wine bottle label to read the variety name. These examples demonstrate how targeted crops resolve fine-grained text on documents and labels that lack of OCR capabilities at low resolution.

[Figure 39]

[Figure 40]

[Figure 41]

##### Fig. 7: Positive examples of AwaRes’s adaptive cropping (4/6). The model correctly crops to count pedestrians in a rainy street scene, identify and count traffic cones in a parking lot, and determine the travel direction of a dog in an urban setting. The crops consistently center on the objects referenced by the question, enabling accurate spatial reasoning and counting from the high-resolution view of outdoor scenarios.

[Figure 42]

[Figure 43]

[Figure 44]

##### Fig.8: Positive examples of AwaRes’s adaptive cropping (5/6). The crops focuses on the church building top and detect the dove, zoom into the road next to the boats at a riverfront to resolve a van’s color, and focus on a broom among garden clutter to identify its color. These examples show how AwaRes localizes small or partially occluded objects that the question refers to, enabling fine-grained attribute recognition.

[Figure 45]

[Figure 46]

##### Fig. 9: Positive examples of AwaRes’s adaptive cropping (6/6). The model focuses on product packaging in a cluttered display to identify a brand name, and crops into the center of a snowy scene to zoom in on a person riding an ATV, resolving the green color of their scarf.

[Figure 47]

[Figure 48]

[Figure 49]

- Fig. 10: Failure cases of AwaRes’s adaptive cropping. The crop either target the wrong region due to lack of sufficient context from the question, or produce a crop that targets a plausible region but fail to produce the correct answer. In the first example, the crop isolates the correct line, yet the model skips the value at 2013 and use 2012 instead, yielding the wrong average. In the second, the crop zooms into the Nutrition Facts label, instead of the ingredient list and reads “Aspartame” instead of the “NutraSweet”. In the third, the crop focuses on the center, where the phone is located, but the granularity of the crop keep access of information around the object that makes the model misidentify its color.

