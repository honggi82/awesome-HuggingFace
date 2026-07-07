# arXiv:2605.05185v1[cs.CV]6May2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

## OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents

Shawn Chen1,2, Kaituo Feng3, Hangting Chen1, Wenxuan Huang3, Dasen Dai3, Quanxin Shou2,4 Yunlong Lin3, Xiangyu Yue3, Shenghua Gao4, Tianyu Pang1,†

1Tencent Hunyuan, 2University of California, Los Angeles 3The Chinese University of Hong Kong, 4The University of Hong Kong

Deep search has become a crucial capability for frontier multimodal agents, enabling models to solve complex questions through active search, evidence verification, and multi-step reasoning. Despite rapid progress, top-tier multimodal search agents remain difficult to reproduce, largely due to the absence of open high-quality training data, transparent trajectory synthesis pipelines, or detailed training recipes. To this end, we introduce OpenSearch-VL, a fully open-source recipe for training frontier multimodal deep search agents with agentic reinforcement learning. First, we curated a dedicated pipeline to construct high-quality training data through Wikipedia path sampling, fuzzy entity rewriting, and source-anchor visual grounding, which jointly reduce shortcuts and one-step retrieval collapse. Based on this pipeline, we curate two training datasets, SearchVLSFT-36k for SFT and SearchVL-RL-8k for RL. Besides, we design a diverse tool environment that unifies text search, image search, OCR, cropping, sharpening, super-resolution, and perspective correction, enabling agents to combine active perception with external knowledge acquisition. Finally, we propose a multi-turn fatal-aware GRPO training algorithm that handles cascading tool failures by masking post-failure tokens while preserving useful pre-failure reasoning through one-sided advantage clamping. Built on this recipe, OpenSearch-VL delivers substantial performance gains, with over 10-point average improvements across seven benchmarks, and achieves results comparable to proprietary commercial models on several tasks. We will release all data, code, and models to support open research on multimodal deep search agents.

Home: https://github.com/shawn0728/OpenSearch-VL Huggingface: https://huggingface.co/OpenSearch-VL

### 1 Introduction

Multimodal deep search has emerged as a critical direction for multimodal large language models (MLLMs), enabling them to evolve from passive visual understanding systems into agents that actively search evidence, verify facts, and reason over knowledge-intensive visual queries (Chen et al., 2026; Feng et al., 2026; Huang et al., 2026). However, frontier multimodal search agents remain difficult to reproduce, as their training data, code are often proprietary or insufficiently disclosed (Huang et al., 2026; Seed, 2026; Singh et al., 2025; Team, 2026b). As a result, the community still lacks a fully open recipe for building, analyzing, and improving strong multimodal search agents.

Among these missing components, high-quality training data is a central bottleneck. The strongest frontier systems are still largely dominated by well-funded commercial corporations (Comanici et al., 2025; Team, 2025b), where the data sources, filtering criteria, expert demonstrations, and tool-use trajectories are typically kept private. This makes it difficult to reproduce advanced multimodal search capabilities or systematically study which data properties are essential for agentic search behavior. The issue is even more pronounced in multimodal settings, where effective training data must capture image-grounded understanding, multi-hop retrieval, evidence verification, and long-horizon tool use rather than simple visual question answering. Therefore, releasing high-quality training data is crucial for making frontier multimodal search agent research more transparent, reproducible, and accessible.

Beyond data, training multimodal search agents also poses unique challenges, especially when applying agentic reinforcement learning (agentic RL) (Fan et al., 2026; Geng et al., 2025; Huang et al., 2026) to long-horizon tool-use

†Corresponding author. 1

settings. Agentic search trajectories involve multiple rounds of reasoning, tool invocation, and observation integration, where a single malformed call, timeout, irrelevant query, or repeated failure can invalidate the remaining rollout. Simply discarding such trajectories wastes useful pre-failure reasoning, while training on the full rollout introduces noisy gradients from meaningless post-failure tokens. Another practical challenge is that real-world visual inputs are often imperfect, such as blurred photos, low-resolution thumbnails, skewed documents, and crowded screenshots. In these cases, searching alone is insufficient, and the agent must first crop, enhance, rectify, or parse the visual evidence before reliable search can begin. However, most existing multimodal search agents focus mainly on retrieval and do not jointly address robust visual pre-processing and failure-aware long-horizon RL.

In this work, we introduce OpenSearch-VL, a fully open recipe for training frontier multimodal deep search agents with agentic RL. Our recipe addresses the above challenges from data, tools, and training. First, we develop a dedicated data curation pipeline to build high-quality training data. Starting from the Wikipedia hyperlink graph, we sample multi-hop entity paths and convert them into multi-hop VQA instances by rewriting intermediate entities into fuzzy descriptions, followed by a carefully designed filtering mechanism. This design avoids single-hop image lookup shortcuts and encourages the agent to learn multi-hop search and reasoning behaviors. This pipeline yields two training datasets SearchVL-SFT-36k for SFT and SearchVL-RL-8k for agentic RL. Second, we build a tool environment that goes beyond retrieval-only multimodal agent. In addition to search, the agent is equipped with OCR, cropping, sharpening, super-resolution, and perspective correction, allowing it to handle imperfect visual inputs in real-world scenarios before querying external knowledge. Finally, we develop an agentic RL algorithm based on GRPO (Guo et al., 2025) for long-horizon multimodal tool use, where multi-step interactions often lead to cascading tool failures. To address this issue, we introduce fatal-aware token masking that removes invalid post-failure suffixes from optimization, while preserving useful pre-failure reasoning through one-sided advantage clamping. This enables the model to learn from partially successful trajectories without being affected by noisy gradients from failed rollouts. Together, these designs enable OpenSearch-VL to learn robust long-horizon search behavior over multimodal evidence in real-world scenarios.

Experiments across multimodal deep search benchmarks show that OpenSearch-VL consistently improves over strong baselines. For example, compared with the Qwen3-VL-30B-A3B (Bai et al., 2025) agentic baseline, our model improves the average score from 47.8 to 61.6, with large gains on VDR (+13.3) (Zeng et al., 2026), MMSearch (+24.5) (Jiang et al.), FVQA (+10.2) (Wang et al., 2017), and InfoSeek (+16.2) (Chen et al., 2023). Moreover, OpenSearch-VL achieves comparable or even better performance than proprietary commercial models on several benchmarks. In summary, our main contributions can be summarized as follows:

- • We introduce OpenSearch-VL, a fully open recipe for training frontier multimodal deep search agents. We will release the training data, code, and models to provide an open foundation for reproducible research on multimodal agentic search.
- • We build the key components required for training advanced multimodal search agents, including high-quality image-grounded multi-hop training data, a diverse tool environment, and a multi-turn fatal-aware GRPO algorithm.
- • Extensive experiments demonstrate the effectiveness of our recipe. For example, our trained OpenSearch-VL-30B-A3B brings an average improvement of 13.8 points across 7 multimodal deep search benchmarks.

### 2 Preliminaries

Problem Formulation. Given an input image I0 and a question q, the agent answers q by interleaving reasoning with tool calls over a diverse tool set T = Tv ∪ Ts, where Tv contains visual tools that transform or parse images and Ts contains retrieval tools that query external knowledge. At step l, the model conditions on the accumulated history

hl = Il, q, a<l, o<l , (1)

where Il, a<l, and o<l denote the images, actions, and observations accumulated up to step l. The interaction unfolds as a multi-turn trajectory

τ = (h0,a0,o0), (h1,a1,o1), ..., (hL−1,aL−1,oL−1), (hL,aL) , (2) where the final step emits the answer without a subsequent observation. Following the ReAct (Yao et al., 2022) think-then-act convention, each action decomposes as al = [zl, cl], where zl is a reasoning trace, and cl denotes a tool invocation for l < L or the final response for l = L.

Multimodal Observations and Active Visual Context. Unlike text-only formulations (Jin et al., 2025), our environment

E returns multimodal observations. Given a control command cl, E deterministically routes the invocation by tool family, ol = E(cl,hl) ∈

Oimg, if cl invokes t ∈ Tv \ {OCR}, Otxt, if cl invokes t ∈ Ts ∪ {OCR},

(3)

so that O = Oimg ∪ Otxt. The active visual context grows monotonically as Il = {I0} ∪ {ok : k < l, ok ∈ Oimg}; historical visual observations are strictly preserved so that the policy can cross-reference multi-hop visual transformations (e.g. a localised Crop against its SuperResolution-enhanced counterpart). The rollout is compactly written as τ ∼ πθ(· | I0,q) ⊗ E, where ⊗ denotes the strict interleaving of policy-emitted actions and environment-returned observations.

Trajectory Likelihood. The policy models the joint trajectory probability via standard autoregressive factorisation:

L

L

Pθ(zl | hl)Pθ(cl | hl,zl). (4)

πθ(τ | I0,q) =

Pθ(al | hl) =

l=0

l=0

Observations ol are excluded from the generative probability mass since they are exogenous outputs of E; they influence the trajectory likelihood only by modulating subsequent histories hl′ for l′ > l. This factorisation is the object directly supervised by SFT (Eq. 8) and the basis of the per-token importance ratio in our RL objective (Eq. 12).

Token-level Generation Mask. Optimisation gradients must be restricted to tokens emitted by the policy itself. For textual observations (originating from Ts and OCR), we define an indicator Mgen(yt) ∈ {0,1} with Mgen(yt) = 1 iff token yt is constituent to a generated action al = [zl, cl], and Mgen(yt) = 0 if yt belongs to an observation span ol. Image-valued observations (from Tv \ {OCR}) are injected directly into the visual backbone and inherently bypass the token-level loss. This protocol, inspired by the retrieved-token masking of Search-R1 (Jin et al., 2025), underlies both the SFT objective (Eq. 8) and the fatal-aware RL mask (Eq. 10); textual serialisations of search results and OCR parses are characteristically noisy and structurally divergent from the policy’s intrinsic generative distribution, and including them in the loss destabilises training.

Search Tools. OpenSearch-VL is equipped with a suite of tools covering three complementary functions: retrieval (TextSearch, ImageSearch) for gathering external evidence, image enhancement (Sharpen, SuperResolution, PerspectiveCorrect) for remedying low-quality inputs, and attention and parsing (Crop, OCR) for localizing and decoding fine-grained content. The suite combines lightweight offline primitives with online services backed by expert models, and is summarized in Table 1. Full specifications are deferred to Appendix F.

- Table 1 | The search-oriented tool suite integrated within OpenSearch-VL. The suite spans three complementary functions—retrieval for acquiring external information, image enhancement for improving low-quality visual inputs, and attention & parsing for focusing on and extracting content from specific regions. Detailed specifications of each tool are provided in Appendix F.

Tool Description Arguments Tool Output

TextSearch Web search with page reading and LLM summarization Query + TopK Query-focused passage summaries ImageSearch Reverse image / visual entity search over the web Image + TopK Visual matches and related webpages Sharpen Unsharp-masking based deblurring / detail enhancement Image + Amount Sharpened image SuperResolution Deep super-resolution (EDSR) for low-resolution inputs Image + Scale High-resolution image PerspectiveCorrect Auto perspective rectification of skewed documents Image Fronto-parallel image Crop Extract a user-specified rectangular region Image + Coordinates Cropped image OCR Structured document parsing with text and layout labels Image + Flags Text blocks with labels and reading order

### 3 Dataset Curation

To equip the model with robust reasoning and tool-use capabilities, we design a scalable data curation pipeline (Figure 1) that synthesizes high-quality trajectories without manual human annotation. The pipeline proceeds in three stages—VQA construction, staged filtering and enhancement, and trajectory synthesis—yielding the final dataset used for the following stage training.

###### (a) High-Quality VQA Construction

[Figure 5]

[Figure 6]

[Figure 7]

Fuzzy Entity Rewriting

Wikipedia Path Sampling Canonical QA Generation

Progressively rewrite entities to obtain fuzzy question qf

[Figure 8]

Constrained random walk on Wikipedia graph

Extract short answer A from answer node vh

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

qt qf

A

[Figure 13]

Generate canonical question qt

[Figure 14]

[Figure 15]

[Figure 16]

qt

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Quality Control

[Figure 21]

Anchor Visual Grounding

Run automated and model-based quality checks

[Figure 22]

[Figure 23]

Search representative Image I for anchor v0 from Wikipedia

Masking (feed the image)

[Figure 24]

[Figure 25]

[Figure 26]

Uniqueness (answer is unique)

[Figure 27]

[Figure 28]

[Figure 29]

Accepted VQA Samples

###### Visual Relevance (image is useful)

[Figure 30]

[Figure 31]

[Figure 32]

Non-triviality (multi-hop reasoning)

Replace anchor mention with a visual referring expression in qf

Multi-hop VQA Pool (I, q, a)

###### (c) Expert Trajectory Synthesis

###### (b) Filtering and Enhancement

Rejection Sampling

Multi-turn Trajectory Synthesis

Staged Filtering

Enhanced Subset (10%)

Answer Correctness Judge

- Filter A: Tool-Free Model

Discard if answer without tools

- Filter B: One-call ImageSearch

Expert Model: Claude Opus 4.6

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Select 10% of filtered samples

[Figure 37]

Final answer must match ground truth A

[Figure 38]

Degrade Images

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Process-level Judge

Discard if a single image search already solves it

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Browse Page-Visit Search Python

[Figure 49]

Goal: retain only tool-demanding, non-trivial samples

[Figure 50]

[Figure 51]

Enhance with Tools

High-Quality Expert Trajectories

[Figure 52]

[Figure 53]

36592 Trajectories; Avg 6.3 tool turns

[Figure 54]

Raw Expert Rollouts

Filtered VQA Pool

Exercises think-with-image behavior

- Figure 1 | Overview of the data curation pipeline. (a) Starting from the English Wikipedia hyperlink graph, we construct high-quality multi-hop VQA instances by sampling constrained paths, generating canonical question–answer pairs, rewriting them into fuzzy questions, grounding anchor entities with representative images, and applying automated quality control. (b) We then perform staged filtering to retain only tool-demanding, non-trivial samples, and create an enhanced subset through image degradation and tool-based restoration to encourage think-with-image behavior. (c) Finally, we synthesize multi-turn expert trajectories in a real tool environment and apply rejection sampling with answer-correctness and process-level judges, yielding the final high-quality trajectories used for the following stage training.

##### 3.1 High-Quality VQA Construction

A central challenge for training multimodal search agents is the supply of questions that encourage non-trivial use of the diverse tool set T . Directly prompting a VLM on an image tends to yield shallow, perception-level queries that can be resolved in a single forward pass (Geng et al., 2025; Huang et al., 2026). Building on this observation, we adopt a unified construction pipeline: we sample multi-hop trajectories over the Wikipedia hyperlink graph, synthesize textual QA pairs along each trajectory, and lift them into image-grounded VQA via answer-preserving fuzzy rewriting and source-anchored visual grounding. Compared with prior QA constructions (Geng et al., 2025; Li et al., 2025b; Wu et al.,

2025a), our pipeline (i) assigns each node on the sampled path an explicit functional role within the reasoning chain, and (ii) deliberately decouples the visual anchor from the answer entity, thereby suppressing single-shot retrieval shortcuts.

Wikipedia Path Sampling. We cast the Wikipedia (wik) as a directed graph G = (V,E) with articles as nodes and in-article hyperlinks as edges. From a seed v0 ∈ V, a constrained random walk of length h ∈ {2,3,4} produces a path

P = v0 −→ρ1 v1 −→ρ2 ··· −→ρh vh , (5) where each relation ρj is induced by the hyperlink’s anchor text. The walk skips (i) disambiguation and list pages, (ii) cycles, and (iii) hub nodes whose in-degree exceeds a threshold τhub; full thresholds, resampling heuristics, and additional filters are deferred to Appendix D.2. Each node on P is assigned a functional role: v0 is the anchor (visual entry point, to be replaced by a visual referring expression), v1,...,vh−1 are bridge nodes (intermediate entities with fuzzified names), and vh is the answer node (source of the target attribute). These roles govern the rewriting and grounding stages below.

We extract a short, unambiguous answer a from vh and prompt GPT-4o (Team, 2024) to synthesize a canonical question qt that verbalizes P and references vh only through the queried attribute (extraction details in Appendix D.2). The canonical qt is not a training target but a manipulable object for rewriting.

Fuzzy Entity Rewriting. Preserving entity names in qt enables the agent to short-circuit the chain with a single retrieval (Huang et al., 2026; Li et al., 2025b). We therefore progressively rewrite qt into a fuzzy counterpart qf while fixing a. Following the iterative style of Skywork-R1V4 (Zhang et al., 2025), we rewrite one entity at a time, from the farthest bridge vh−1 toward v0: each name is replaced by a relational or attribute-based descriptor drawn from the entity’s Wikipedia context, and an LLM uniqueness evaluator verifies that the substitution still resolves to the intended entity conditional on the partially rewritten question. A rewrite is accepted only when

, (6)

###### , hj=0 aliases(vj) ∩ qf = ∅

, |R(qf)| = 1

a(qf) = a(qt) answer invariance

uniqueness

non-leakage

where R(qf) denotes the set of entities compatible with qf under the evaluator’s world knowledge. We further interleave entity rewriting with occasional answer obfuscation (Huang et al., 2026) to avoid collapsing onto a stereotyped relational template.

Anchor-aware Visual Grounding. We retrieve a representative image I of the anchor v0 from Wikimedia Commons or its Wikipedia infobox, filter candidates by CLIP similarity to a short textual description of v0, and replace v0 in qf with a visual referring expression (e.g., “the person in the image”) to yield the final question q. Unlike prior QA-to-VQA conversions (Geng et al., 2025; Zhang et al., 2025) that ground on or near the answer entity, anchoring v0 at the source of P substantially reduces single-hop shortcuts: the agent must first identify the visual anchor and then follow the intermediate textual relations before reaching a.

Each candidate triple (I,q,a) is gated by automatic checks for masking, uniqueness, and visual relevance, generalizing the selector/examiner protocol of WebWatcher (Geng et al., 2025) (full criteria in Appendix D.2); non-triviality is handled jointly with the staged filtering of Sec. 3.2. Instances passing these checks form the Wikipedia portion of our VQA pool, subsequently merged with open-source multimodal corpora before trajectory synthesis.

##### 3.2 Filtering and Enhancement

Before trajectory synthesis, we consolidate the Wikipedia-derived VQA instances from Sec. 3.1 with three open-source multimodal corpora—LiveVQA (Fu et al., 2025), FVQA (Wang et al., 2017), and WebQA (Chang et al., 2022)—to broaden coverage across live entities, commonsense fact lookup, and open-web multi-hop reasoning. We then apply a two-stage difficulty filter using a frozen Qwen3-VL-32B (Bai et al., 2025): first discarding examples answerable without tools, and then discarding examples solvable with a single ImageSearch call. This removes samples that rely only on parametric knowledge, perceptual shortcuts, answer-coincident anchors, or one-hop bridge leakage, ensuring that retained instances genuinely require the intended visual-to-text search chain.

To further expose the agent to realistic visual imperfections, we randomly select 10% of the filtered VQA pool and apply controlled degradations—blur, downsampling, and perspective distortion—paired with the corresponding enhancement tools in Tv (Sharpen, SuperResolution, and PerspectiveCorrect). This enhancement subset diversifies the training distribution and induces a think-with-image behavior: when the input image is unreliable, the policy learns to repair the visual evidence before initiating retrieval. Together, the filtered retrieval-heavy instances

[Figure 55]

- Figure 2 | Overview of the RL training pipeline. Starting from a supervised fine-tuned model, we sample a group of multi-turn trajectories against the real environment E. Each trajectory is evaluated by a composite reward combining

final-task success (racc) and process-level search quality (rquery) along with a format check (rfmt). To preserve valid reasoning in trajectories that eventually encounter fatal errors, we apply fatal-aware token masking to truncate the sequence and employ one-sided advantage clamping during policy optimization, preventing the suppression of viable early steps.

and the enhancement-required subset exercise both visual restoration and evidence acquisition within the unified tool environment.

##### 3.3 Multi-turn Trajectory Synthesis

For each instance (I,q,a) that survives the filters of Sec. 3.2, we synthesize expert trajectories by rolling out Claude Opus 4.6 (Team, 2026a) as the expert model against the real execution environment E, prompted with the agent system prompt of Appendix E and free to invoke any tool in T . We draw K = 5 independent rollouts per instance, each formatted as a multi-turn ReAct (Yao et al., 2022) trajectory aligned with Eq. 2. Then the raw rollouts are passed through a two-stage rejection cascade. The first stage discards any trajectory whose final answer disagrees with the ground truth a (adjudicated by the same GPT-4o (Team, 2024) LLM-as-judge (Gu et al., 2025) we use for racc, Sec. 4.2). The surviving trajectories are then vetted by a GPT-5.4 process-level judge on tool-use, logical consistency between reasoning and observations, and absence of ineffective repetition, sharing the four-dimension rubric of rquery (Sec. 4.2).

Applying both stages to the full rollout corpus yields 36,592 high-quality expert trajectories with an average of 6.3 tool-invocation turns per trajectory, which together constitute the SFT corpus consumed in Sec. 4.1.

### 4 Training

We train OpenSearch-VL in two sequential stages. First, we perform supervised fine-tuning (SFT) to instill fundamental reasoning and tool-use behaviors; subsequently, we apply reinforcement learning (RL) via a multi-turn, search-augmented objective (Figure 2) to discover more effective exploration strategies.

##### 4.1 Supervised Fine-Tuning

We perform SFT on a curated set of 36592 multi-turn expert trajectories (Section 3). Using the history hl (Eq. 1) and the action decomposition al = [zl, cl], by autoregressive factorisation the step-level action probability decomposes as

Pθ(al | hl) = Pθ(zl | hl) Pθ(cl | hl, zl). (7)

Summing over all trajectories i ∈ {1,...,N} and steps l ∈ {1,...,Li}, the standard SFT objective can be equivalently written as

Li

N

log Pθ zl(i) | h(li) + log Pθ c(li) | h(li), zl(i) , (8)

max

θ

i=1

l=1

where tool observations ol enter only as conditioning context and are excluded from the loss computation following the retrieved-token masking strategy of (Jin et al., 2025). This provides a structured interpretation of the training signal, showing that it jointly supervises both the reasoning trace and the subsequent tool invocation (or terminal response) at each step.

##### 4.2 Multi-Turn Search Fatal-Aware GRPO

While SFT provides a strong initialization for tool use, it remains bounded by the coverage of the demonstration trajectories and therefore cannot discover improved search strategies through exploration. We address this limitation with reinforcement learning, building on GRPO (Shao et al., 2024) and its search-augmented extension (Jin et al., 2025). Our setting, however, differs from prior search-only formulations in three important respects: we optimize over a multimodal environment E with diverse tools rather than a text-only retriever R; we use a composite reward that combines final-task success with process-level search quality; and we introduce fatal-aware masking together with one-sided advantage clamping to preserve useful supervision from partially successful trajectories.

During training, for each prompt (I0,q) we sample a group of G multi-turn rollouts τi ∼ πθ

(· | I0,q) ⊗ E, for i = 1,...,G.

old

Composite Multi-Turn Reward. Long-horizon tasks pose a sparse-reward challenge: outcome-only rewards miss credit for partially successful reasoning, while process-only rewards risk misalignment from the end goal. We therefore use a composite trajectory-level reward

r(τ) = rfmt(τ) · α racc(τ) + (1−α)rquery(τ) , (9) where α = 0.8. The composite trajectory-level reward (Eq. 9) is structured to balance algorithmic formatting, terminal accuracy, and process-level search quality. We define each component as follows:

- • Format reward rfmt ∈ [0,1]. A deterministic, algorithmic prior that enforces structural integrity. We define rfmt(τ) = L+11 Ll=0 rfmt(l), where rfmt(l) = 1 iff step l emits a contiguous <think>···</think> block immediately followed by either a <tool_call>···</tool_call> block (for l < L) or a <response>···</response> block

(for l = L); rfmt(l) = 0 for any structural violation, including steps that trigger tool-execution errors (e.g., malformed API arguments). By acting as a multiplicative gate in Eq. 9, rfmt drives the overall return of structurally degraded trajectories toward zero.

- • Accuracy reward racc ∈ {0,1}. A terminal outcome metric assessing the fidelity of the agent’s final resolution. A GPT-4o (Team, 2024) judge under an LLM-as-Judge protocol (Gu et al., 2025) verifies semantic equivalence between

the agent’s terminal <response> and the ground-truth annotation, assigning racc = 1 for a match and 0 otherwise. Convention for truncated trajectories: for trajectories aborted by the fatal-state condition (Eq. 10) before emitting a terminal <response>, we deterministically assign racc = 0. This is a structural guarantee rather than an evaluative judgment—in the absence of a terminal answer, correctness is strictly undefined and conservatively zeroed out—and it ensures the outcome signal remains well-defined for group-relative advantage estimation regardless of completion status.

- • Query-quality reward rquery ∈ [0,1]. A process-level signal that counteracts the inherent sparsity of racc in longhorizon interactions. We use GPT-5.4 (OpenAI, 2025), a proprietary frontier reasoning model, as the query-quality judge to score the cumulative sequence of search queries on a continuous [0,1] scale, providing dense feedback

for unsuccessful trajectories with racc = 0. The rubric covers four dimensions: (i) semantic relevance of issued queries to the initial prompt; (ii) logical progression and iterative refinement of queries across successive turns; (iii) signal-to-noise ratio within retrieved payloads; and (iv) cross-modal complementary use of image and text retrieval tools. For trajectories designated as fatal, the judge restricts its evaluation to the valid pre-fatal prefix (steps l < fi), so that early-stage reasoning is credited despite subsequent collapse.

Fatal-Aware Token Masking. In unconstrained multi-turn environments, agents frequently encounter “fatal” statessuch as cascading tool-execution failures or infinite loops—after which subsequent reasoning becomes meaningless.

Standard approaches either discard the entire trajectory (Huang et al., 2026) (wasting the valid early steps) or train on it blindly (injecting noise). We introduce a fatal-aware masking strategy to preserve the viable prefix.

We define the fatal step index fi for trajectory τi as the earliest step where K = 3 consecutive tool-execution errors commence, with fi = Li + 1 if no such cascade occurs. We then extend the observation-token (generation) mask Mgen(yi,t) to additionally zero out all tokens generated after the fatal step:

M(yi,t) = Mgen(yi,t) · ⊮ s(t) < fi , (10)

where s(t) maps token index t to its step index l. Crucially, the process rewards rfmt and rquery are computed exclusively over the valid prefix l < fi, ensuring the model is not penalised for structural collapse that occurs after the trajectory is deemed fatal.

All G trajectories—including fatal ones—contribute to the standard group-normalised reward ri = (r(τi)−mean)/(std+ δ) to keep group statistics unbiased. However, directly using ri for fatal trajectories is pathological: a sub-mean ri would push the policy gradient to suppress the viable prefix, discouraging the valid reasoning before the error cascade. We therefore apply one-sided advantage clamping:

ri if fi = Li + 1 (non-fatal), max( ri, 0) if fi ≤ Li (fatal).

(11)

Aˆi =

This clamping ensures that the valid prefix of a fatal trajectory is only ever reinforced if its partial reward exceeds the group mean, and otherwise receives zero gradient rather than an undeserved penalty. In this sense, it generalizes the hard-masking baseline (Huang et al., 2026) while recovering strictly more useful learning signal.

Integrating the fatal-aware mask Mi,t ≡ M(yi,t) and the clamped advantage Aˆi, our final GRPO objective over the multimodal environment E is formulated as:

  1

 . (12)

|τi|

G

1 t Mi,t

Mi,t min ρi,t(θ)Aˆi,clip1+1−ϵϵ(ρi,t(θ))Aˆi

J (θ) = E(I

0,q)∼D {τi}Gi=1∼πθold(·|I0,q; E)

G

t=1

i=1

where ρi,t(θ) = πθ(yi,t | I0,q,yi,<t;E)/πθ

(yi,t | I0,q,yi,<t;E) is the token-level importance ratio, and the standard β DKL[πθ∥πref] term is omitted from the display since it is identical to that of standard GRPO (Eq. 14). Relative to search-augmented GRPO (Jin et al., 2025), the differences (color-matched to Eq. 12) are threefold: the execution environment is generalized from R to E, the generation mask Mgen is extended to the fatal-aware mask M, and the advantage is computed from the composite reward with one-sided clamping. Additional details and derivations are provided in Appendix B.

old

### 5 Experiments

We first describe our experimental setups as follows:

Models and Benchmarks. Our OpenSearch-VL is built on three Qwen3-VL variants (Bai et al., 2025): Qwen3-VL-8BInstruct, Qwen3-VL-30B-A3B-Instruct, and Qwen3-VL-32B-Instruct. For evaluation, we use seven knowledge-intensive benchmarks from our main results: SimpleVQA (Cheng et al., 2025), VDR (Zeng et al., 2026), MMSearch (Jiang et al.), LiveVQA (Fu et al., 2025), BrowseComp-VL (Geng et al., 2025), FVQA (Wang et al., 2017), and InfoSeek (Chen et al., 2023). Together, they cover visual entity recognition, web evidence retrieval, multi-hop reasoning, and long-tail QA.

Baselines and Evaluation Metrics. We evaluate OpenSearch-VL against baseline types: Direct Reasoning, where the model answers from its parametric knowledge and visual perception alone; RAG Workflow, where external retrieval results are provided in-context but the reasoning remains single-pass; and Agentic Workflow, where the model autonomously interleaves reasoning with tool calls in a multimodal environment. We report Pass@1 on all seven benchmarks. Correctness is adjudicated by a GPT-4o judge that compares the model’s final response with the reference answer. To ensure fair comparison across heterogeneous answer styles, we adopt the same evaluation protocol as VDR-Bench (Zeng et al., 2026); the full judge prompt is provided in Figure 10.

Datasets. Our SFT data consists of the 36K multi-turn trajectories synthesized by the procedure in Sec. 3.3. For RL training, we randomly sample 8K examples from the VQA pool after the staged filtering and enhancement process in

- Sec. 3.2, ensuring that these examples are disjoint from the VQA instances used to synthesize the SFT trajectories.

- Table 2 | Performance on multimodal knowledge-intensive QA and web-search benchmarks. Bold and underline mark the best and second-best score in each column.

Model SimpleVQA VDR MMSearch LiveVQA BrowseComp-VL FVQA InfoSeek Avg Direct Reasoning

- GPT-4o (Team, 2024) 51.7 1.7 18.7 28.1 5.5 48.0 52.9 29.5
- GPT-5 (OpenAI, 2025) 61.6 9.8 35.1 44.4 48.6 54.4 61.7 45.1 Gemini-2.5-Flash (Comanici et al., 2025) 57.9 6.2 30.4 51.0 37.1 47.7 44.1 39.2 Gemini-2.5-Pro (Comanici et al., 2025) 63.0 8.0 39.8 60.3 43.1 60.7 46.9 46.0 Claude-4-Sonnet (Team, 2025b) 50.9 2.0 18.7 38.5 29.3 35.3 57.3 33.1 Claude-3.7-Sonnet (Team, 2025a) 42.7 4.6 21.1 38.0 32.3 36.7 54.8 32.9 Qwen3-VL-8B (Bai et al., 2025) 47.1 2.8 11.7 23.1 24.1 24.2 23.1 22.3 Qwen3-VL-30B-A3B (Bai et al., 2025) 53.2 3.8 18.7 42.7 29.6 34.7 26.4 29.9 Qwen3-VL-32B (Bai et al., 2025) 58.0 4.1 19.8 45.5 30.8 34.1 28.8 31.6

###### RAG Workflow

- GPT-4o (Team, 2024) 63.6 4.5 49.1 40.1 13.4 66.3 59.5 42.4

- GPT-5 (OpenAI, 2025) 55.9 22.3 52.6 56.0 54.9 62.6 70.6 53.6 Claude-3.7-Sonnet (Team, 2025a) 59.3 11.3 32.7 30.3 10.0 59.1 60.2 37.6 Qwen3-VL-8B (Bai et al., 2025) 62.3 7.3 47.3 39.3 29.3 53.6 46.1 40.7

Agentic Workflow DeepMMSearch-R1-7B (Narayan et al., 2025a) 55.8 – – – – – 47.5 – Visual-ARFT-7B (Liu et al., 2025) 42.4 3.3 34.5 25.4 16.5 41.7 37.9 28.8 MMSearch-R1-7B(Wu et al., 2025b) 57.4 2.9 53.8 48.4 20.9 58.4 55.1 42.4 DeepEyes-v2-7B (Hong et al., 2025) 59.4 7.8 63.7 – – 60.6 51.1 – WebWatcher-7B (Geng et al., 2025) 54.3 10.3 49.1 51.2 21.2 – – – Qwen3-VL-8B (Bai et al., 2025) 52.0 17.0 37.4 50.6 27.9 58.7 50.3 42.0 SenseNova-MARS-8B (Chng et al., 2025) 61.7 19.4 67.4 56.2 35.1 67.1 61.7 52.7 OpenSearch-VL-8B(Ours) 71.6 20.8 64.5 59.6 37.6 71.5 70.2 56.6

Qwen3-VL-30B-A3B (Bai et al., 2025) 55.1 20.2 44.2 62.0 34.1 63.0 56.2 47.8 OpenSearch-VL-30B-A3B(Ours) 74.9 33.5 68.7 67.4 41.1 73.2 72.4 61.6

Qwen3-VL-32B (Bai et al., 2025) 58.7 23.1 53.9 45.5 35.1 61.2 58.5 48.0 WebWatcher-32B Geng et al. (2025) 59.0 – 55.3 58.7 27.0 – – – OpenSearch-VL-32B(Ours) 76.2 33.8 72.3 70.5 43.8 74.7 74.8 63.7

Implementation Details. OpenSearch-VL extends LlamaFactory (Zheng et al., 2024) for agentic SFT and builds on rLLM (Tan et al., 2025) and VDR (Huang et al., 2026) for multi-turn tool-interleaved RL. All stages of OpenSearch-VL are trained on Nvidia H20 GPUs. Agentic SFT takes roughly 2 days for the 8B dense model and 4 days for the 30B-A3B MoE model on 256 H20s (32 nodes × 8 GPUs); the subsequent multi-turn fatal-aware GRPO stage runs for approximately 200 optimization steps over 10 days on 64 H20s (8 nodes × 8 GPUs). The complete per-stage hyperparameter configurations are listed in the following subsections (SFT in Table 4; RL in Table 5). More details are reported in Appendix C.

##### 5.1 Main Results

- Table 2 reports the results on seven multimodal knowledge-intensive QA and web-search benchmarks. OpenSearch-VL exhibits a clear advantage over both direct-reasoning and RAG baselines across all scales, underscoring the necessity of an agentic loop for complex multimodal queries. Among 8B-scale agents, OpenSearch-VL-8B achieves the best average score of 56.6, surpassing the previous strongest open 8B agent, SenseNova-MARS-8B, by 3.9 points on average. At larger scales, OpenSearch-VL-30B-A3B and OpenSearch-VL-32B further improve the average score to 61.6 and 63.7, respectively; notably, OpenSearch-VL-32B outperforms strong proprietary direct-reasoning models such as Gemini-2.5-Pro and substantially exceeds the corresponding Qwen3-VL agentic baselines. These results demonstrate that our training recipe scales effectively from 8B to 32B and yields strong gains on both search-heavy and visually grounded benchmarks.

##### 5.2 Ablation Study

We conduct ablations on the Qwen3-VL-8B model to validate the two design choices that define OpenSearch-VL: the data synthesis pipeline that produces tool-demanding multimodal trajectories, and the fatal-aware RL recipe that improves the policy beyond offline imitation.

- Table 3 | Ablation studies on the SFT data pipeline and RL training recipe. Deltas in the top panel are measured relative to the full pipeline; deltas in the bottom panel are measured relative to Vanilla GRPO (Jin et al., 2025).

(a) SFT data pipeline ablation

Settings SimpleVQA InfoSeek FVQA Avg. Full pipeline (Ours) 66.1 62.4 65.3 64.6

w/o source-anchor grounding 53.6(-12.5) 54.5(-7.9) 51.2(-14.1) 53.1(-11.5) w/o fuzzy entity rewriting 51.7(-14.4) 56.4(-6.0) 54.7(-10.6) 54.3(-10.3) w/o staged filtering 57.6(-8.5) 55.2(-7.2) 56.3(-9.0) 56.4(-8.2) w/o enhancement subset 64.9(-1.2) 61.7(-0.7) 63.2(-2.1) 63.3(-1.3)

###### (b) RL recipe ablation

Method SimpleVQA InfoSeek FVQA Avg. Qwen3-VL-8B 52.0 50.3 58.7 53.7 Qwen3-VL-8B + SFT only 66.1 62.4 65.3 64.6

+ Vanilla GRPO 68.8 66.5 67.4 67.6

+ GRPO w/ Hard Masking 68.3(-0.5) 67.9(+1.4) 66.9(-0.5) 67.7(+0.1) + GRPO w/ Fatal Masking only 69.7(+0.9) 68.3(+1.8) 69.2(+1.8) 69.1(+1.5) + Fatal Masking + One-sided Clamp (Ours) 71.6(+2.8) 72.4(+5.9) 71.5(+4.1) 71.8(+4.2)

Data Pipeline Ablation. Table 3 shows that each stage of our data synthesis pipeline contributes to the final performance. The full pipeline achieves the best average score of 64.6, while removing source-anchor grounding, fuzzy entity rewriting, or staged filtering leads to large drops of 11.5, 10.3, and 8.2 points, respectively. These results indicate that effective training data must both prevent shortcut retrieval and preserve genuinely tool-demanding queries. Removing the enhancement subset causes a smaller but consistent decline (64.6 → 63.3 Avg.), suggesting that image-restoration trajectories mainly improve robustness rather than driving the core gains.

Training Recipe Ablation. Table 3 studies the effect of our RL recipe after the same Qwen3-VL-8B SFT initialization, using Search-R1-style vanilla GRPO (Jin et al., 2025) as the RL baseline. SFT improves the base model from 53.7 to 64.6 average accuracy, and vanilla GRPO further raises it to 67.6, showing the benefit of online exploration. The way fatal trajectories are handled is crucial: the hard-masking strategy of Vision-DeepResearch (Huang et al., 2026) brings almost no gain over vanilla GRPO (Guo et al., 2025; Jin et al., 2025)(67.6 → 67.7), while fatal masking improves the average to 69.1 by preserving valid pre-failure reasoning. Our full method with one-sided advantage clamping achieves the best score on every benchmark and reaches 71.8 average accuracy, a 4.2-point gain over vanilla GRPO. The training curves in Fig. 3 are consistent with this result: fatal-aware GRPO sustains longer tool-use trajectories while achieving higher batch accuracy, indicating that it encourages productive exploration rather than prematurely suppressing difficult rollouts.

0.8

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Averagenumberofturns

12

0.7

10

Accuracy

0.6

8

0.5

Hard Mask

6

GRPO

0.4

Fatal-aware GRPO

4

0 50 100 150 200

0 50 100 150 200

Training steps

Training steps

- Figure 3 | Training dynamics over the RL phase. Left: average number of turns per rollout. Right: batch-level accuracy. Fatal-aware GRPO sustains a higher number of turns and reaches a higher accuracy than vanilla GRPO (Guo et al., 2025; Jin et al., 2025) and the Hard-Mask (Huang et al., 2026) baseline.

Empirical Visualization of the Two Cases. Figure 4 visualizes how Aˆi = max( ri,0) interacts with per-group difficulty on two representative rollout groups. In the higher-difficulty case, most fatal trajectories still emit a coherent prefix before hitting a tool error; a fraction of them beat the group mean and contribute a positive gradient to the prefix tokens only. In the lower-difficulty case, every fatal trajectory is dominated by the successful non-fatal rollouts, and

Non-fatal ¿i Fatal ¿i Fatal kept Fatal clamped Group mean r¹

(a) Higher-difficulty prompt: fatal rollouts scattered around group mean

(b) Lower-difficulty prompt: all fatal rollouts far below group mean

0/3 fatal ¿i above r¹ ) A^i = 0 (clamped)

3/5 fatal ¿i above r¹ ) A^i > 0 (preserved)

0.30

1.2

0.25

1.0

r(¿)Rewardi

r(¿)Rewardi

0.20

0.8

0.15

0.6

r ¹

0.10

0.4

r ¹

0.05

0.2

0.00

0.0

¿(1) ¿(16)

¿(1) ¿(16)

Trajectories in group, sorted by r(¿i)

Trajectories in group, sorted by r(¿i)

A^i = +2:34 example clamped fatal ¿i (any fatal bar above)

A^i = 0

example preserved fatal ¿i (top bar above)

|preserve prefix| |masked suffix|
|---|---|---|

|clamp to 0| |masked suffix|
|---|---|---|

l = 0 fatal onset l = fi l = Li

l = 0 fatal onset l = fi l = Li

- Figure 4 | Fatal-aware masking with one-sided clamping on two illustrative rollout groups. Each panel shows a group of G=16 rollouts (non-fatal in blue, fatal in brick red) together with a cartoon of the resulting token-level update on one representative fatal trajectory τi. Bar outlines mark fatal rollouts whose pre-clamp score ri exceeds the group mean r¯ (green, preserved) or falls below it (dark grey, clamped). (a) Higher-difficulty prompt. When r¯ is low, several fatal prefixes already beat it: ri > 0 and Aˆi = ri, so gradients flow only through the viable prefix (tokens up to the fatal onset fi) while the post-failure suffix is hard-masked—the partial reasoning that reached the failure point is reinforced even though the trajectory itself is unsolvable. (b) Lower-difficulty prompt.

When most rollouts succeed, every fatal reward falls far below r¯ and one-sided clamping sets Aˆi = max( ri, 0) = 0, degenerating to pure hard-masking and avoiding any suppression of the possibly-valid prefix.

clamping to zero prevents a noisy negative signal from pushing the policy away from what may actually be a valid prefix.

- Figure 5 shows that this split is not anecdotal: aggregated over 10,000 groups, the vast majority of fatal trajectories sit on the negative side of the pre-clamp score and are safely zeroed out, while the small preserved tail is distributionally close to the right mode of the non-fatal reference. One-sided clamping thus recovers a principled credit signal from failed tool-call trajectories without amplifying their inherent noise.
- 6 Related Work

##### 6.1 Multimodal Agentic Search

The integration of active search has reframed LLMs from static knowledge bases into agentic reasoners capable of dynamic information retrieval. Search-R1 (Jin et al., 2025) crystallized this shift by using RL to incentivize autonomous, multi-turn querying within the reasoning chain. This paradigm has since migrated to multimodal domains; MMSearch-R1 (Wu et al., 2025b) and Vision-DeepResearch (Huang et al., 2026; Narayan et al., 2025b) embed visual retrieval into agent pipelines, while recent efforts (Chen et al., 2026; Feng et al., 2026; Li et al., 2025a; Yao et al., 2026; Zhang et al., 2025) attempt to unify image manipulation with web search. However, a common but fragile assumption in these works is the availability of pristine visual inputs. In practice, when agents encounter degraded or text-dense real-world images, the "search-only" approach fails as retrieval cannot fix fundamentally broken visual evidence.

##### 6.2 Visual Perception and Retrieval

The performance of multimodal agents is often bottlenecked not by retrieval logic, but by the fidelity of initial perception Wei et al. (2025); Zhang et al. (2026). While RAG frameworks like VisRAG (Yu et al., 2024) emphasize preserving visual structure, they treat the model as a passive observer that must "make do" with whatever is retrieved. Even as tool-augmented agents (Geng et al., 2025; Hong et al., 2025; Song et al., 2026) introduce dynamic orchestration, their toolsets remain largely homogeneous and scenario-bound. We argue that robust reasoning requires active perception: the agent must not only search but also intervene—autonomously invoking tools like super-resolution or specialized OCR to remediate visual noise before attempting to reason over it.

##### 6.3 Reinforcement Learning for Agentic Reasoning

Training agents to operate in long-horizon, multi-tool environments poses substantial challenges for standard RL. While GRPO (Shao et al., 2024) has shown strong effectiveness in aligning reasoning trajectories for language models (Chen et al., 2025a,b; Feng et al., 2025a,b; Guo et al., 2025; Hu et al., 2026), applying it to agentic rollouts with diverse tool interactions remains non-trivial. The primary challenge lies in cascading failures: an early tool error renders the rest of the trajectory incoherent (Dong et al., 2025; Vuddanti et al., 2025; Zheng et al., 2026; Zhu et al., 2025), yet these post-failure tokens still inject noise into the policy gradient (Deng et al., 2026; Garg et al., 2026; Liu and Xiao, 2025). Furthermore, standard group-normalization tends to penalize the valid reasoning prefixes of partially successful rollouts (Huang et al., 2026). To counter this, we introduce a fatal-aware RL objective that prunes learning signals from post-failure states and uses one-sided advantage clamping to protect the gradients of constructive reasoning steps.

### 7 Conclusion

We present OpenSearch-VL, a fully open recipe for training multimodal deep search agents with agentic reinforcement learning. Our recipe combines a Wikipedia-based data curation pipeline that mitigates one-step retrieval shortcuts and produces two high-quality datasets: SearchVL-SFT-36k & SearchVL-RL-8k; a diverse tool environment spanning retrieval, image enhancement, and attention-and-parsing tools; and a multi-turn fatal-aware GRPO algorithm that preserves useful pre-failure reasoning through one-sided advantage clamping. Based on this recipe, OpenSearch-VL achieves over 10-point average gains across seven multimodal deep search benchmarks, with competitive performance on representative tasks such as VDR compared with strong proprietary reasoning models. We will release our data, code, models, and training recipe, with the aim to lower the reproducibility barrier and provide an open foundation for future research on multimodal deep search agents.

### Limitations and Future Work

A non-trivial fraction of training instability traces to the external tool environment E—including search ranking drift, fetch failures, and occasional summarization hallucinations in TextSearch and ImageSearch—which inflates reward variance and motivates future work on on-policy reliability estimation. Furthermore, our composite reward (Eq. 9) relies on proprietary GPT-4o judges, which are costly, version-dependent, and currently score only textual queries while ignoring intermediate visual operations (e.g., Crop); replacing these with open process reward models covering the full visual action space Tv remains a natural next step. Finally, exact numerical reproducibility is challenged by the reliance on these externally hosted APIs (e.g., Serper, PaddleX OCR) and the prohibitive cost of reporting multi-seed error bars for large-scale evaluations (Table 2). To mitigate these constraints and support open research, we will release our complete datasets (SearchVL-SFT-36k / SearchVL-RL-8k), model checkpoints, and training code under permissive licenses.

### References

Wikipedia. https://www.wikipedia.org/.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa, 2022. URL https://arxiv.org/abs/2109.00590.

Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. arXiv preprint arXiv:2506.04207, 2025a.

Shuang Chen, Yue Guo, Yimeng Ye, Shijue Huang, Wenbo Hu, Haoxi Li, Manyuan Zhang, Jiayu Chen, Song Guo, and Nanyun Peng. Ares: Multimodal adaptive reasoning via difficulty-aware token-level entropy shaping. arXiv preprint arXiv:2510.08457, 2025b.

Shuang Chen, Quanxin Shou, Hangting Chen, Yucheng Zhou, Kaituo Feng, Wenbo Hu, Yi-Fan Zhang, Yunlong Lin, Wenxuan Huang, Mingyang Song, et al. Unify-agent: A unified multimodal agent for world-grounded image synthesis. arXiv preprint arXiv:2603.29620, 2026.

Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can pre-trained vision and language models answer visual information-seeking questions? arXiv preprint arXiv:2302.11713, 2023.

Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, et al. Simplevqa: Multimodal factuality evaluation for multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4637–4646, 2025.

Yong Xien Chng, Tao Hu, Wenwen Tong, Xueheng Li, Jiandong Chen, Haojia Yu, Jiefan Lu, Hewei Guo, Hanming Deng, Chengjun Xie, et al. Sensenova-mars: Empowering multimodal agentic reasoning and search via reinforcement learning. arXiv preprint arXiv:2512.24330, 2025.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Wenlong Deng, Yushu Li, Boying Gong, Yi Ren, Christos Thrampoulidis, and Xiaoxiao Li. On group relative policy optimization collapse in agent search: The lazy likelihood-displacement, 2026. URL https://arxiv.org/abs/25 12.04220.

Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Agentic reinforced policy optimization, 2025. URL https://arxiv.org/abs/2507.19849.

Kaixuan Fan, Kaituo Feng, Manyuan Zhang, Tianshuo Peng, Zhixun Li, Yilei Jiang, Shuang Chen, Peng Pei, Xunliang Cai, and Xiangyu Yue. Exploring reasoning reward model for agents. arXiv preprint arXiv:2601.22154, 2026.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025a.

Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, et al. Onethinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025b.

Kaituo Feng, Manyuan Zhang, Shuang Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, and Xiangyu Yue. Gen-searcher: Reinforcing agentic search for image generation. arXiv preprint arXiv:2603.28767, 2026.

Mingyang Fu, Yuyang Peng, Benlin Liu, Yao Wan, and Dongping Chen. Livevqa: Live visual knowledge seeking. arXiv preprint arXiv:2504.05288, 2025.

Anisha Garg, Claire Zhang, Nishit Neema, David Bick, Ganesh Venkatesh, and Joel Hestness. Corpo: Adding a correctness bias to grpo improves generalization, 2026. URL https://arxiv.org/abs/2511.04439.

Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, et al. Webwatcher: Breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748, 2025.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on llm-as-a-judge, 2025. URL https://arxiv.org/abs/2411.15594.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081): 633–638, 2025.

Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, and Xing Yu. Deepeyesv2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271, 2025.

Wenbo Hu, Xin Chen, Yan Gao-Tian, Yihe Deng, Nanyun Peng, and Kai-Wei Chang. Openvlthinkerv2: A generalist multimodal reasoning model for multi-domain visual tasks, 2026. URL https://arxiv.org/abs/2604.08539.

Wenxuan Huang, Yu Zeng, Qiuchen Wang, Zhen Fang, Shaosheng Cao, Zheng Chu, Qingyu Yin, Shuang Chen, Zhenfei Yin, Lin Chen, et al. Vision-deepresearch: Incentivizing deepresearch capability in multimodal large language models. arXiv preprint arXiv:2601.22060, 2026.

Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Pengshuo Qiu, Pan Lu, Zehui Chen, Guanglu Song, Peng Gao, Yu Liu, et al. Mmsearch: Unveiling the potential of large models as multi-modal search engines. In The Thirteenth International Conference on Learning Representations.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025. URL https: //arxiv.org/abs/2503.09516.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent, 2025a. URL https://arxiv.org/abs/2507.02592.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, et al. Websailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025b.

Haoyu Liu and Le Xiao. Re-grpo: Leveraging hard negative cases through large language model guided self training. Neurocomputing, page 132543, 2025.

Ziyu Liu, Yuhang Zang, Yushan Zou, Zijian Liang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual agentic reinforcement fine-tuning. arXiv preprint arXiv:2505.14246, 2025.

Kartik Narayan, Yang Xu, Tian Cao, Kavya Nerella, Vishal M Patel, Navid Shiee, Peter Grasch, Chao Jia, Yinfei Yang, and Zhe Gan. Deepmmsearch-r1: Empowering multimodal llms in multimodal web search. arXiv preprint

- arXiv:2510.12801, 2025a.

Kartik Narayan, Yang Xu, Tian Cao, Kavya Nerella, Vishal M Patel, Navid Shiee, Peter Grasch, Chao Jia, Yinfei Yang, and Zhe Gan. Deepmmsearch-r1: Empowering multimodal llms in multimodal web search. arXiv preprint

- arXiv:2510.12801, 2025b.

OpenAI. Introducing gpt-5. https://openai.com/index/introducing-gpt-5/, 2025. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda

Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization

algorithms. arXiv preprint arXiv:1707.06347, 2017. Bytedance Seed. Seed2.0 model card, 2026. URL https://github.com/ByteDance-Seed/Seed2.0. Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li,

Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Mingyang Song, Haoyu Sun, Jiawei Gu, Linjie Li, Luxin Xu, Ranjay Krishna, and Yu Cheng. Adareasoner: Dynamic tool orchestration for iterative visual reasoning, 2026. URL https://arxiv.org/abs/2601.18631.

Sijun Tan, Michael Luo, Colin Cai, Tarun Venkat, Kyle Montgomery, Aaron Hao, Tianhao Wu, Arnav Balyan, Manan Roongta, Chenguang Wang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. rllm: A framework for post-training language agents, 2025. Notion Blog.

Anthropic Team. Claude 3.7 sonnet system card. https://www-cdn.anthropic.com/9ff93dfa8f445c932415d

335c88852ef47f1201e.pdf, 2025a.

Anthropic Team. System card: Claude opus 4 & claude sonnet 4. https://www-cdn.anthropic.com/6d8a80550

20700718b0c49369f60816ba2a7c285.pdf, 2025b.

Anthropic Team. Claude opus 4.6 system card. https://www-cdn.anthropic.com/14e4fb01875d2a69f646fa5

e574dea2b1c0ff7b5.pdf, 2026a. Kimi Team. Kimi k2.5: Visual agentic intelligence, 2026b. URL https://arxiv.org/abs/2602.02276. OpenAI Team. Gpt-4o system card, 2024. URL https://arxiv.org/abs/2410.21276. Sri Vatsa Vuddanti, Aarav Shah, Satwik Kumar Chittiprolu, Tony Song, Sunishchal Dev, Kevin Zhu, and Maheep

Chaudhary. Paladin: Self-correcting language model agents to cure tool-failure cases, 2025. URL https: //arxiv.org/abs/2509.25238.

Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton Van Den Hengel. Fvqa: Fact-based visual question answering. IEEE transactions on pattern analysis and machine intelligence, 40(10):2413–2427, 2017.

Yana Wei, Liang Zhao, Kangheng Lin, En Yu, Yuang Peng, Runpei Dong, Jianjian Sun, Haoran Wei, Zheng Ge, Xiangyu Zhang, et al. Perception in reflection. arXiv preprint arXiv:2504.07165, 2025.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, et al. Webdancer: Towards autonomous information seeking agency. arXiv preprint arXiv:2505.22648, 2025a.

Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, and Ziwei Liu. Mmsearch-r1: Incentivizing lmms to search. arXiv preprint arXiv:2506.20670, 2025b.

Huanjin Yao, Qixiang Yin, Min Yang, Ziwang Zhao, Yibo Wang, Haotian Luo, Jingyi Zhang, and Jiaxing Huang. Mm-deepresearch: A simple and effective multimodal agentic search baseline. arXiv preprint arXiv:2603.01050, 2026.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594, 2024.

Yu Zeng, Wenxuan Huang, Zhen Fang, Shuang Chen, Yufan Shen, Yishuo Cai, Xiaoman Wang, Zhenfei Yin, Lin Chen, Zehui Chen, Shiting Huang, Yiming Zhao, Yao Hu, Philip Torr, Wanli Ouyang, and Shaosheng Cao. Visiondeepresearch benchmark: Rethinking visual and textual search for multimodal large language models. preprint, 2026.

Jiankun Zhang, Shenglai Zeng, Kai Guo, Xinnan Dai, Hui Liu, Jiliang Tang, and Yi Chang. Fix before search: Benchmarking agentic query visual pre-processing in multimodal retrieval-augmented generation. arXiv preprint arXiv:2602.13179, 2026.

Yifan Zhang, Liang Hu, Haofeng Sun, Peiyu Wang, Yichen Wei, Shukang Yin, Jiangbo Pei, Wei Shen, Peng Xia, Yi Peng, et al. Skywork-r1v4: Toward agentic multimodal intelligence through interleaved thinking with images and deepresearch. arXiv preprint arXiv:2512.02395, 2025.

YaoweiZheng, RichongZhang, JunhaoZhang, YanhanYe, ZheyanLuo, ZhangchiFeng, andYongqiangMa. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing "thinking with images" via reinforcement learning, 2026. URL https://arxiv.org/abs/2505.143 62.

Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, XiaotengMa, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Zicheng Liu, Pan Lu, James Zou, and Jiaxuan You. Where llm agents fail and how they can learn from failures, 2025. URL https://arxiv.org/abs/2509.25370.

### Appendix Appendix Contents

- A Preliminary of Reinforcement Learning 17

- A.1 Standard GRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.2 GRPO with Search Engine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- B Multi-Turn Search Fatal-Aware GRPO Details 19

- B.1 Fatal Step Detection Logic . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.2 Advantage Derivation with One-Sided Clamping . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- C Implementation Details 21

- C.1 SFT Training Configuration. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.2 RL Training Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- D Data Curation Details 24

- D.1 Path Sampling Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.2 Running Example: Australia_Zoo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.3 Counterfactual Design Choices. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- E System Prompts 26
- F Tool Definition and Usage 30
- G Case Study 31

### A Preliminary of Reinforcement Learning

This section details the reinforcement learning (RL) preliminaries underpinning our multi-turn training objective. We first review the standard Group Relative Policy Optimization (GRPO) algorithm (Shao et al., 2024), which operates on single-turn generations. Subsequently, we describe its extension to search-augmented rollouts, as introduced by Search-R1 (Jin et al., 2025), wherein the policy interleaves generation with calls to an external search engine. These formulations serve as the direct precursors to the multi-turn, multi-tool objective employed by OpenSearch-VL (see

- Sec. 4.2).

##### A.1 Standard GRPO

GRPO (Shao et al., 2024) is an actor-critic variant of Proximal Policy Optimization (PPO) (Schulman et al., 2017) that eliminates the reliance on a learned value function. Instead of maintaining a separate critic model to estimate a baseline, GRPO derives its advantage estimate by comparing multiple responses sampled for the same prompt, collectively defining a group. This parameter-efficient design is particularly advantageous in settings where only a scalar outcome-level reward is available, reducing memory overhead and mitigating the instability of training a dense token-level value function.

Formally, given a prompt distribution D and a reference policy πθ

, GRPO samples a group of G candidate responses {o1,o2,...,oG} ∼ πθ

old

(· | q) for each prompt q ∼ D. Each response is evaluated via a reward function ri = r(q,oi).

old

To compute the advantages, GRPO standardizes these rewards within the local group:

ri − mean({rj}Gj=1) std({rj}Gj=1)

, for all tokens t = 1,...,|oi|. (13)

Aˆi,t = ri =

By this formulation, all constituent tokens of a given response oi are assigned a uniform advantage scalar. The policy parameters θ are subsequently optimized by maximizing the clipped surrogate objective:

|oi|

G

1 |oi|

1 G

JGRPO(θ) = Eq∼D,{o

i}Gi=1∼πθold(·|q)

t=1

i=1

(14)

min ρi,t(θ)Aˆi,t, clip ρi,t(θ), 1−ϵ, 1+ϵ A ˆi,t

− β DKL πθ ∥πref ,

where the importance sampling ratio is defined as

πθ(oi,t | q, oi,<t) πθ

. (15)

ρi,t(θ) =

(oi,t | q, oi,<t)

old

Here, ϵ represents the probability ratio clipping hyperparameter, and β modulates the Kullback–Leibler (KL) divergence penalty against a fixed reference policy πref. In contrast to standard PPO—which subsumes the KL penalty directly into the reward signal and necessitates a parameterized value network to approximate Aˆi,t (Schulman et al., 2017)—GRPO enforces the KL regularization explicitly within the loss landscape. By deriving Aˆi,t strictly from empirical group statistics (Eq. 13), GRPO naturally aligns with the comparative structure of preference-based reward modeling.

##### A.2 GRPO with Search Engine

Search-R1 (Jin et al., 2025) extends the GRPO framework from unimodal, single-turn generation to an interleaved, search-augmented generative process. Rather than sampling an isolated response oi ∼ πθ

(· | q), Search-R1 generates a multi-step rollout comprising both policy-emitted tokens and external evidence retrieved from a search engine R. This search-augmented sampling distribution is denoted as:

old

(· | q) R, (16)

oi ∼ πθ

###### (· | q;R) = πθ

old

old

where represents the interleaving operator: whenever the policy emits a search command, the retrieved documents are appended to the conditioning context, and the policy resumes generation conditioned on this expanded prefix.

To prevent the optimizer from erroneously updating parameters based on exogenous environmental tokens, Search-R1 utilizes a masking function that filters out non-generated tokens. Defining Mgen(yi,t) ∈ {0,1} as an indicator variable where Mgen(yi,t) = 1 if yi,t is explicitly generated by the policy, the corresponding Search-R1 GRPO objective becomes:

|oi|

G

1

1 G

JGRPOR (θ) = Eq∼D,{o

i}Gi=1∼πθold(·|q; R)

|oi| t=1 Mgen(yi,t)

t=1 Mgen(yi,t)=1 min ρRi,t(θ)Aˆi,t, clip ρRi,t(θ), 1−ϵ, 1+ϵ A ˆi,t

i=1

(17)

− β DKL πθ ∥πref ,

featuring the environment-conditioned importance ratio:

πθ(yi,t | q, yi,<t; R) πθ

. (18)

ρRi,t(θ) =

(yi,t | q, yi,<t; R)

old

Relative to standard GRPO (Eq. 14), this adaptation introduces two pivotal modifications. First, the expectation is calculated with respect to the interleaved distribution πθ

(· | q;R), ensuring the causal conditioning history incorporates all previously retrieved external evidence. Second, the objective is normalized strictly by the count of generated tokens,

old

t Mgen(yi,t), restricting gradient calculations to policy-authored positions. Equivalently masking the KL divergence ensures the model is not penalized for distributional divergence over environmental observations. These mechanisms adapt GRPO to tool-in-the-loop architectures and establish the theoretical foundation for our multimodal environment E formulation (Sec. 4.2).

(c) Clamped vs. preserved fatal rollouts

Clamped (rei < 0)

1.6

Preserved (rei ¸ 0)

Non-fatal reference

1.4

1.2

Density

0

1.0

0.8

0.6

0.4

0.2

0.0

−2.0 −1.5 −1.0 −0.5 0.0 0.5 1.0 1.5 2.0

Pre-clamp score rei of fatal rollouts

Figure 5 | Aggregate distribution of clamped and preserved fatal rollouts. Pre-clamp group-normalized scores ri aggregated over 10,000 groups (G=16; 47,978 fatal rollouts in total). 91.8% of fatal rollouts fall on the negative side of the clamp threshold (mean r−= − 0.68) and are zeroed out by Aˆi=max( ri,0); the remaining 8.2% are preserved (mean r+= + 0.57) and overlap with the positive mode of the non-fatal reference density (blue line: pre-clamp scores of all non-fatal rollouts from the same groups). Preservation is therefore driven by the same group-normalized score that GRPO already computes, and the preserved fatal prefixes sit in the same score regime as stronger non-fatal behaviors – not an ad-hoc heuristic.

### B Multi-Turn Search Fatal-Aware GRPO Details

Building upon the foundations established in Appendix A, this section details the mechanics of our multi-turn, searchaugmented RL objective introduced in Sec. 4.2. We articulate the detection logic for fatal execution cascades and rigorously derive the advantage estimation process incorporating one-sided clamping; the composite reward r(τ) and its three components rfmt, racc, rquery are defined directly in Sec. 4.2.

##### B.1 Fatal Step Detection Logic

Unconstrained interactions with external environments inevitably yield “fatal” states—irrecoverable error cascades rendering subsequent reasoning invalid. We articulate the procedure for computing the fatal step index fi, which drives the fatal-aware token mask (Eq. 10).

For a given trajectory τi of length Li, we instantiate a stateful error counter nerr initialized to zero. At each sequential step l ∈ {0,1,...,Li}, the counter transitions according to:

n(errl−1) + 1 if step l triggers a tool-execution error, 0 otherwise.

(19) The fatal step index is subsequently defined as the earliest step where the error threshold is breached:

n(errl) =

fi = min{l : n(errl) = K}. (20) In our implementation, we set K = 3. If the threshold is never reached, the trajectory is classified as non-fatal, denoted by setting fi = Li + 1.

This formulation embodies two critical design properties. First, the explicit reset condition (n(errl) = 0) ensures that isolated transient errors—which are ubiquitous and often recoverable in realistic web environments—do not prematurely abort the trajectory. Second, the conservative threshold (K = 3) dictates that a trajectory is only deemed fatal following three consecutive failures, affording the autoregressive policy a robust opportunity to self-correct prior to gradient masking. Tool-execution errors triggering this counter encompass timeouts, malformed API payloads, and argument-parsing failures; these are flagged deterministically by the execution sandbox, independent of any learned heuristic.

- B.2 Advantage Derivation with One-Sided Clamping We rigorously detail the advantage estimation protocol underpinning the objective function in Eq. 12.

- Step 1: Unbiased Group Statistics. For a sampled group of G rollouts originating from the identical prompt (I0,q), the empirical group mean and standard deviation are computed over the composite rewards:

µG = sg

  1

G

G

j=1

r(τj)

 , σG = sg

  1

G

G

j=1

r(τj) − µG 2

 . (21)

Here, sg(·) denotes the stop-gradient operator. Consequently, µG and σG act as static normalizing constants during backpropagation. Gradients flow exclusively through the importance sampling ratios evaluated on the unmasked, policy-generated tokens.

- Step 2: Universal Group Normalization. Each trajectory within the group is assigned a standardized reward:

ri =

r(τi) − µG σG + δ

, (22)

where δ > 0 provides numerical stability. Crucially, all G trajectories—expressly including those truncated by the fatal condition (fi ≤ Li)—are incorporated into the calculation of µG and σG. This deliberate inclusion ensures that fatal trajectories actively shape the group-level baseline, anchoring the relative performance ranking across the entire sampled cohort.

- Step 3: Fatal-Aware One-Sided Clamping. The standardized rewards are mapped to the final advantage estimates Aˆi via a piecewise clamping function:

Aˆi =

ri if fi = Li+1 (non-fatal trajectory), max ri, 0 if fi ≤ Li (fatal trajectory).

(23)

- Step 4: Gradient Dominance over Hard-Masking. We now formalize the informal claim that fatal-aware clamping strictly dominates the hard-masking baseline of (Huang et al., 2026) in gradient informativeness. Let

|τi|

1 t Mi,t

giours =

Mi,t ∇θ log πθ yi,t | hs(t) A ˆi (24)

t=1

denote the (un-clipped) per-trajectory contribution to ∇θJ (θ) under our scheme, and let gihard denote the corresponding contribution under the hard-masking baseline, which discards every fatal trajectory in full so that gihard = 0 whenever fi ≤ Li. We compare the two on the support of fatal trajectories.

Proposition 1 (Dominance over hard-masking). Fix any fatal trajectory τi (fi ≤ Li). Then:

- 1. If ri < 0, then giours = gihard = 0.
- 2. If ri ≥ 0, then gihard = 0, while giours coincides with the search-augmented GRPO gradient (Eq. 17) of τi restricted to its viable prefix {t : s(t) < fi, Mgen(yi,t) = 1}, evaluated at the non-negative advantage ri.

Consequently, giours is weakly informative-dominant over gihard: it never propagates gradient through the post-fatal suffix, never penalises the viable prefix, and strictly extracts positive reinforcement on prefixes whose group-normalised return exceeds the baseline.

Proof. By Eq. 11, ri < 0 implies Aˆi = 0, so the surrogate term and its gradient vanish identically in Eq. 24; combined with gihard = 0 by definition of hard-masking, this establishes (i). For ri ≥ 0, Aˆi = ri, while the fatal-aware mask Mi,t = Mgen(yi,t) · ⊮[s(t) < fi] retains exactly the policy-generated tokens of the viable prefix and zeros out (a) environment-emitted tokens and (b) all post-fatal tokens. On this support the per-token integrand of Eq. 24 reduces tokenby-token to that of Eq. 17 with the same importance ratio ρi,t and advantage ri ≥ 0, so the per-token surrogates—and hence their gradients—agree. Since gihard = 0 by definition, this establishes (ii).

| |
|---|

In aggregate, the one-sided clamp safely ignores invalid credit assignment while selectively harvesting positive reinforcement from prematurely truncated yet high-quality exploratory rollouts.

Bias of Group Statistics under Clamping. The asymmetric definition of Aˆi deserves an honest accounting. We compute (µG,σG) in Step 1 over all G rollouts—including fatal ones—so the standardised score ri is zero-mean within the group; Step 3 then maps any negative ri on a fatal trajectory to zero. Letting F ⊆ {1,...,G} index the fatal subset of the group,

G

G

1 G

1 G

1 G i∈F

, (25)

Aˆi =

max 0, − ri

ri

+

i=1

i=1

= 0

= bG ≥ 0

so the clamp induces a non-negative bias bG relative to the zero-mean GRPO baseline, in exchange for non-zero gradient on viable prefixes. We argue this trade-off is benign in our regime for two reasons. (a) Concentration of fatal rollouts in the lower mode. Figure 5 shows that 91.8% of fatal rollouts have ri < 0 and are clamped to zero, while the residual 8.2% that survive sit in the same score regime as competitive non-fatal rollouts; bG is therefore dominated by samples whose viable prefix is empirically high-quality, rather than by indiscriminate inflation of low-quality fatals. (b) Preservation of relative ranking. Because bG shifts only the fatal subset upward and never displaces non-fatal advantages, the within-group ordering between non-fatal trajectories—which is the actual signal GRPO exploits—remains intact. We therefore interpret bG as a controlled positive bias that buys back gradient on prematurely truncated yet promising prefixes; replacing the clamp with an unbiased estimator (e.g., a separate baseline computed only over non-fatal rollouts, or a doubly-robust correction for bG) is an interesting alternative we leave for future work.

A sanity check of the resulting clamping behaviour, including a per-group-difficulty breakdown and the aggregate score distribution over 10,000 groups, is reported in Sec. 5 (Figs. 4 and 5).

### C Implementation Details

Our agentic SFT pipeline builds on LlamaFactory (Zheng et al., 2024), which we extend with multi-turn, toolinterleaved data collators and a Qwen3-VL-aware vision/text packing scheme so that interleaved image observations produced by visual tools (Crop, Sharpen, SuperResolution, PerspectiveCorrect, OCR) can be consumed verbatim by the policy. Our RL pipeline builds jointly on rLLM (Tan et al., 2025) and Vision-DeepResearch (Huang et al., 2026): from the former we adopt the asynchronous agent rollout architecture (decoupled trajectory generation, replay, and policy-update workers), and from the latter we adopt the multimodal trajectory abstraction. On top of these two, we implement a Qwen3-VL chat-template renderer, an interleaved image-token re-alignment routine for variable-length visual observations, and an asynchronous multi-turn rollout engine that supports tool-call interruption and resumption. The resulting pipeline is what we use to train all OpenSearch-VL variants in this paper.

##### C.1 SFT Training Configuration

- Table 4 reports the full hyperparameter configuration used for the agentic supervised fine-tuning stage across all three OpenSearch-VL variants. The three model sizes (8B dense, 32B dense, 30B-A3B Mixture-of-Experts) share identical training hyperparameters—only the base checkpoint path differs—so we present them in a single consolidated table. All runs use full-parameter finetuning (including the vision tower and multi-modal projector) with DeepSpeed ZeRO-3 and Ray-based orchestration over 256 GPUs (32 nodes × 8 GPUs).

C.2 RL Training Configuration

- Table 5 reports the key hyperparameters for the multi-turn fatal-aware GRPO stage across the two OpenSearch-VL variants we RL-finetune: 8B dense (Qwen3-VL-8B-Instruct) and 30B-A3B MoE (Qwen3-VL-30B-A3B-Instruct). Both runs use the same async SGLang rollout engine and the same Megatron-parallel actor, with a shared algorithm template (RLOO leave-one-out advantage under the GRPO group-relative objective, low-variance KL controller, no critic). We list only the fields set explicitly in our launch scripts; framework defaults (e.g. optimizer betas, reward-model paths, checkpoint-resume flags) are omitted. Rows whose value is identical for both variants are written once; rows that differ are split into two cells.

Category Hyperparameter Value / Setting

Base Model Qwen3-VL-8B-Instruct (32B, 30B-A3B) Image Max Pixels 262,144 (≈ 512 × 512) Video Max Pixels 16,384 (≈ 128 × 128) Trust Remote Code True

Model

Finetuning Type Full Vision Tower Frozen False MM Projector Frozen False DeepSpeed Stage ZeRO-3 Mixed Precision bfloat16

Method

Total Samples 36592 Template qwen3_vl Cutoff Length 32,000 tokens Preprocessing Workers 16 Dataloader Workers 4

Dataset

Batch Size per Device 1 Gradient Accumulation Steps 1 Effective Batch Size 256 (= 1 × 1 × 256 GPUs) Gradient Checkpointing True Learning Rate 2.0 × 10−5 Epochs 8 LR Scheduler cosine Warmup Ratio 0.1

Training

Total GPUs 256 (32 nodes × 8) Orchestration Ray + DeepSpeed ZeRO-3 Placement Strategy PACK Resources per Worker 1 GPU

Infrastructure

Logging Steps 5 Checkpoint Save Steps 400 Plot Loss True Report Backend TensorBoard

Logging / IO

- Table 4 | Agentic SFT training configuration and hyperparameters for OpenSearch-VL. All three model sizes (8B dense, 32B dense, 30B-A3B MoE) use the identical recipe; only the Base Model row differs at launch time. Hyperparameters not exercised in our pipeline—e.g. held-out evaluation split, LoRA/adapter settings, or layer-freezing schedules—are omitted.

Category Hyperparameter 8B (dense) 30B-A3B (MoE)

Base Model Qwen3-VL-8B-Instruct Qwen3-VL-30B-A3B-Instruct Training Dtype bfloat16

Model

Train Batch Size (prompts) 256 Val Batch Size 64 512 Max Prompt Length 4,096 tokens Max Response Length 70,000 tokens Data Seed 3407

Data

Engine SGLang (async mode) Rollout Tensor Parallel 4 # Samples per Prompt (n) 8 16 GPU Mem. Utilization 0.85 0.65 Train / Val Temperature 0.7 / 0.7 Train / Val Top-p 1.0 / 0.95 Top-k −1 (disabled)

Rollout

Strategy Megatron-LM Tensor Parallel (TP) 4 Pipeline Parallel (PP) 2 Context Parallel (CP) 8 4 Expert Parallel / ETP — 8 / 1 PPO Mini-batch Size 64 128 PPO Max Token Len / GPU 74,576 74,596 Micro-batch Size / GPU 1 Dynamic Batch Size True Param / Optim / Grad Offload CPU Gradient Checkpointing Full recompute, uniform (1 layer)

Policy (Actor)

Actor LR 1 × 10−6 PPO Clip Ratio (high) 0.28 Entropy Coefficient 0.0 Use KL Loss False KL Loss / Controller Coef 1×10−3 / 1×10−3 Loss Aggregation seq-mean-token-sum

Optim. / Loss

Advantage Estimator RLOO (within GRPO objective) KL Type low-variance KL Fatal-aware Masking True (unknown + error)

Algorithm

# Parallel Tasks 256 # Parallel Tool Calls 2,048 Stepwise Advantage False

Tool-Agent

Cluster 8 nodes × 8 GPUs = 64 GPUs Save / Test Freq (steps) 50 / 10 1 / 10 Total Epochs 100 5 Critic Warmup 0 (critic-free)

Trainer

Router Dtype — fp32 Auxiliary-Loss Coef — 0.01 Router Z-loss Coef — 1×10−3 Permute Fusion — True

MoE-only (30B-A3B)

- Table 5 | Key RL training hyperparameters for the multi-turn fatal-aware GRPO stage of OpenSearch-VL. The 8B dense run and the 30B-A3B MoE run share the same algorithm template and rollout/actor stack; values that differ between the two (response budget per prompt, mini-batch size, context/expert parallelism, MoE-specific routing coefficients, save cadence and total epochs) are shown side-by-side. Entries marked “—” are not applicable (e.g. no expert parallelism in the dense 8B model). Framework-default fields and cluster-network environment variables are omitted.

### D Data Curation Details

This section complements Sec. 3.1 with the full operational specification of the multi-hop VQA construction pipeline and traces the pipeline end-to-end through a concrete seed page. All statistics below are computed on the 2025-05-01 snapshot of English Wikipedia.

##### D.1 Path Sampling Hyperparameters

We fix τhub = 10,000 on the in-degree measured against the snapshot. Any candidate node whose incoming count exceeds τhub is skipped, rejecting roughly the top 0.03% of all article nodes. This cut-off excludes continent-, country-, and century-level pages (e.g. United_States, Queensland, 21st_century, English_language) for which the uniqueness invariant in Eq. 6 is routinely violated, while preserving the long tail of entity pages that carry substantive semantic content.

Path length distribution. Path lengths are sampled as h ∼ Categorical({2,3,4}; (0.4, 0.4, 0.2)). Shorter paths are favoured in order to keep the downstream rollout horizons tractable; the upper bound h = 4 is set empirically, as walks longer than four hops rarely survive the uniqueness and non-leakage checks of Eq. 6 without heavy resampling.

Beyond the three rules stated in the main text, the walk additionally skips:

- 1. any title containing the substring (disambiguation), or beginning with List of, Outline of, Index of, or Timeline of;
- 2. all non-article namespaces, i.e. Template:, Category:, File:, User:, Help:, Portal:, Wikipedia:;
- 3. redirect pages: each outgoing link is first dereferenced to its target article, and the exclusion rules above are applied to the dereferenced target rather than the surface link.

Seeds are drawn by stratified sampling across five coarse domains—Person, Building/Place, Location (non-hub), Organism, and Artifact—to balance the representation of visually groundable categories. A node is eligible as a seed iff it (i) exposes an infobox; (ii) links to at least one Wikimedia Commons image of resolution no smaller than 512 × 512; and (iii) has in-degree in [50, τhub], so that the seed is neither a dead end nor a hub.

A walk rooted at a fixed seed is retried up to 10 times whenever it (i) hits a hub or a filtered namespace, (ii) fails to produce any descriptor for some bridge vj that survives the LLM uniqueness evaluator, or (iii) terminates at a node vh whose infobox contains no attribute meeting the six-token length bound. Seeds that exhaust all 10 attempts are dropped from the final pool.

#### D.2 Running Example: Australia_Zoo

To make the five stages of Sec. 3.1 concrete, we trace the pipeline end-to-end on the seed page Australia_Zoo with h = 2.

The outgoing links of Australia_Zoo are first partitioned by the exclusion rules of Appendix D.1. Representative entries of each partition are:

- • Retained (entitypages, in-degreebelowτhub): Bob_Irwin,Steve_Irwin, Terri_Irwin,Bindi_Irwin, Robert_Irwin, The_Crocodile_Hunter, Wildlife_Warriors, Beerwah,_Queensland,Rosedale,_Queensland,Angkor_Wat,

. . .

- • Hub-rejected (in-degree > τhub): Queensland, Brisbane, Australia, United_States, Zoo. A random walk with h = 2 then samples

###### −−−−−−−−−−→ρ1=“managed by” Steve_Irwin

###### −−−−−−−−−→ρ2=“spouse of” Terri_Irwin

Australia_Zoo

###### .

v0 (anchor)

v1 (bridge)

v2 (answer)

From the lead paragraph of v2, we extract the attribute date of Australian citizenship, yielding the short answer a = “20 November 2009” (3 tokens). GPT-4o (Team, 2024) is then prompted with the full path and its relations to

synthesize the canonical question

qt = “On what date did Terri Irwin, the wife of Steve Irwin—the man who

took over management of Australia Zoo in 1991—become an Australian citizen?”

Rewriting proceeds from the farthest bridge v1 = Steve_Irwin toward v0. Table 6 lists the descriptor candidates proposed by GPT-4o (Team, 2024) from v1’s Wikipedia context and the verdicts returned by the uniqueness evaluator. The first accepted descriptor is retained (ties broken uniformly at random). The accepted descriptor produces the fuzzy

Descriptor for v1 Uniqueness verdict

“the man who took over management of [v0] in 1991” ✓ unique “the wildlife documentary host killed by a stingray in 2006” ✓ unique “the son of the founders of [v0]” × – Bob and Lyn had three children (Joy, Steve,

Mandy)

- Table 6 | Descriptor candidates for the bridge node v1 = Steve_Irwin and the verdicts returned by the LLM uniqueness evaluator.

form

qf = “On what date did the wife of the man who took over management of

Australia Zoo in 1991 become an Australian citizen?”

[Figure 56]

All three invariants of Eq. 6 are satisfied: a is preserved, the descriptor combined with “wife of” resolves uniquely to Terri Irwin, and qf contains no surface form or alias of any node along P (e.g. neither Steve Irwin, Terri Irwin, nor The Crocodile Hunter appears in qf).

We then retrieve K = 8 candidate images from Wikimedia Commons under the query Australia_Zoo and rank them by CLIP (Radford et al., 2021) cosine similarity to the canonical description “Australia Zoo entrance”. The top-ranked image (Fig. 6, similarity 0.34) is selected as I. Replacing the anchor mention in qf with a visual referring expression yields the final VQA instance

Figure 6 | Representative image I for the anchor v0 = Australia_Zoo.

(I, q, a) = I, “On what date did the wife of the man who took over management of the zoo in

the image in 1991 become an Australian citizen?”, “20 November 2009” .

TheinstancepassesallfourchecksofSec.3.1: masking(q containsnoentitynameoraliasfrom{Australia Zoo,Steve Irwin,Terri Irwin}); uniqueness (a GPT-4o judge given only q returns exactly one consistent answer); visual relevance (CLIP similarity 0.34 exceeds our 0.28 threshold); and non-triviality, verified jointly under the filtering process of Sec. 3.2.

##### D.3 Counterfactual Design Choices

We probe the necessity of the three key design choices in Sec. 3.1 by counterfactually relaxing each one on the running example above.

Anchor = Answer. Replacing I with a photograph of Terri Irwin and asking for her citizenship date reduces the task to a single reverse-image lookup: ImageSearch on the new I returns the Wikipedia page of Terri Irwin directly, from which the citizenship date is read off in one step. The multi-hop structure collapses, independently of how the question is phrased.

No fuzzing. Retaining the canonical qt as the final question—i.e. skipping the fuzzy-rewriting stage—exposes all entity names in plain text, and a single TextSearch(“Terri Irwin Australian citizenship date”) suffices to recover a. The image I becomes decorative rather than load-bearing.

No hub avoidance. If the walk were allowed to admit the hub Queensland as a bridge, a natural descriptor such as “the zoo located in [vj]” would fail the uniqueness check—thousands of zoos satisfy this relation—forcing either unsuccessful resampling or a contrived descriptor that itself leaks the identity of Australia_Zoo. Hub avoidance (Appendix D.1) removes this failure mode at sampling time rather than relying on the downstream filters to catch it.

### E System Prompts

This section provides the complete system prompts used in OpenSearch-VL. The agent system prompt (Figure 7) is shared across inference and SFT data collection, and is paired with the machine-readable tool schema (Figure 8) that is injected into the model context as <tools>...</tools> for OpenAI-style function calling. The two reward judge prompts (Figures 8 and 11) are used during RL training to compute racc and rquery, respectively. For final benchmark reporting, we additionally adopt a GPT-4o judge prompt (Figure 10) that is aligned with the evaluation protocol of Vision-DeepResearch (Huang et al., 2026), so that our numbers remain directly comparable to prior multimodal deep-research work.

Accuracy Reward Judge Prompt (racc)

You are an impartial judge evaluating whether a deep research report contains the correct answer. [Question] {question} [Correct Answer] {reference_answer} [Deep Research Report] {assistant_answer} Task: Determine if the deep research report contains the correct answer anywhere in its content. Instructions:

- 1. Read through the entire research report carefully
- 2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)
- 3. Check if the information in the report is consistent with the correct answer
- 4. The answer does NOT need to be in a specific format or labelled as “final answer”
- 5. Provide your reasoning
- 6. Answer with “yes” if the report contains the correct answer, “no” if it doesn’t or contradicts it Output format: correct: [yes/no] reasoning: [your explanation]

- Figure 8 | The GPT-4o judge prompt used to compute the accuracy reward racc ∈ {0,1} during RL training.

###### Benchmark Evaluation Judge Prompt (GPT-4o)

You are an impartial judge evaluating whether a deep research report contains the correct answer. [Question] {question} [Correct Answer] {correct_answer} [Deep Research Report] {response} Task: Determine if the deep research report contains the correct answer anywhere in its content. Instructions:

- 1. Read through the entire research report carefully
- 2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)
- 3. Check if the information in the report is consistent with the correct answer
- 4. The answer does NOT need to be in a specific format or labelled as “final answer”
- 5. Provide your reasoning
- 6. Answer with “yes” if the report contains the correct answer, “no” if it doesn’t or contradicts it Output format: correct: [yes/no] reasoning: [your explanation]

- Figure 10 | GPT-4o judge prompt used for benchmark evaluation of OpenSearch-VL and all baselines. We deliberately keep this prompt aligned with the evaluation protocol released by Vision-DeepResearch (Huang et al., 2026), so that reported accuracies are directly comparable across systems. Although structurally similar to the RL accuracy-reward judge (Figure 8), this prompt is applied post-hoc to final agent trajectories rather than as a training signal, and uses the field names (question, correct_answer, response) consumed by our evaluation script.

###### Query-Quality Reward Judge Prompt (rquery)

You are an impartial judge evaluating the quality and utility of an agent’s search trajectory. [Original Question] {question} [Ground Truth Answer] {ground_truth} [Agent’s Final Answer] {prediction} [Search Trajectory] {trajectory_summary} [Evaluation Criteria] Evaluate the overall utility of the agent’s search queries and retrieved results:

- 1. Image search utility: Did image searches retrieve visual evidence that genuinely supports answering the question? Were the images relevant, or just noise?
- 2. Text search utility: Did text searches find relevant textual information? Were queries well-formed and targeted?
- 3. Query progression: Did the queries show logical progression—refining, narrowing, or covering different aspects? Or did they repeat / drift aimlessly?
- 4. Complementarity: Did image and text searches complement each other, providing evidence that one modality alone couldn’t supply?
- 5. Evidence vs. noise ratio: What fraction of retrieved results actually contained useful evidence versus irrelevant content? Score the overall query utility from 0.0 to 1.0: - 0.0: No useful information retrieved; all searches irrelevant or failed - 0.3: Mostly noise with occasional marginally relevant results - 0.5: Mixed—some useful evidence found but significant noise or inefficiency - 0.7: Good search strategy; majority of results were relevant - 1.0: Excellent—targeted, efficient queries that retrieved highly relevant evidence Output format (strictly follow): score: [a single float between 0.0 and 1.0] reasoning: [brief explanation in 2–3 sentences]

- Figure 11 | The GPT-4o judge prompt used to compute the query-quality reward rquery ∈ [0,1] during RL training.

###### Agent System Prompt (Inference / SFT Data Collection)

You are an advanced Visual Investigation Agent. Answer user questions with maximum precision by proactively using image-processing and retrieval tools. Core Philosophy: “Verify, Don’t Guess”

- 1. Tool-First Mindset: small text ⇒ crop; blurry ⇒ sharpen; tilted ⇒ perspective_correct. Never rely on the internal encoder when a tool gives a sharper view.
- 2. Chain Your Tools: non-trivial queries usually require a pipeline, e.g. perspective_correct → crop → layout_parsing.
- 3. External Validation: whenever the answer depends on facts not purely visible in the pixels, you must call text_search.

- 1. Tool Calling Format Function signatures are provided inside <tools>...</tools>. Emit tool calls as one JSON object inside <tool_call>{"name": <f>, "arguments": <args>}</tool_call>, one call per turn.
- 2. Your Toolbox {Tool List} For each of the seven tools the production prompt specifies its trigger (when to call), params (JSON schema), and output (how to consume the return value). The tools fall into three families: visual perception (crop, layout_parsing); image enhancement (perspective_correct, super_resolution, sharpen); and knowledge retrieval (text_search, image_search; image search must be followed by text search).
- 3. Thinking Protocol Before any action, emit a <think> block with: (i) analyse request; (ii) assess image quality—legibility, geometry, and target size, mapping any deficiency to the corresponding enhancement tool; (iii) identify information gaps—retrieval needed?; (iv) formulate plan—commit to a single next action. Critical reminders: (a) when layout_parsing returns text, treat it as ground truth—do not fall back to visual guessing; (b) after image_search, always call text_search for factual detail; (c) text_search summaries are already query-focusedtrust them and cite the returned URLs.
- 4. Workflow Recipes

- • Unreadable document: perspective_correct → sharpen → layout_parsing.
- • Dense chart: crop (region of interest) → layout_parsing.
- • Entity identification: image_search → text_search (mandatory follow-up).

- 5. Output Rules

- 1. Single action per turn; wait for its result before the next.
- 2. Think first: never emit <tool_call> without a preceding <think>.
- 3. Image refs: initial image is img_1; each tool output yields img_2, img_3, . . . ; always operate on the latest best version.
- 4. Final answer: emit <response>...</response> once evidence suffices.

- 6. Execution Example (tool-use turn) <think> Invoice img_1 is skewed; correct perspective first. </think> <tool_call>{"name": "perspective_correct", "arguments": {"image": "img_1"}}</tool_call>

- Figure 7 | Condensed agent system prompt used during both inference and SFT trajectory collection. The placeholder {Tool List} stands in for the per-tool description block of the production prompt; concrete trigger/params/output content for each of the seven tools is reproduced in Figure 8 (machine-readable schema) and Table 1 (human-readable summary). Relative to the original prompt used in our codebase, we condense the three long “critical reminder” paragraphs into a single line, drop redundant execution examples, and merge per-tool workflow sub-rules into the core philosophy; no behavioural rule is removed.

###### Tool Definitions (OpenAI-compatible JSON schema, injected as <tools>)

{

"tools": [ {

"type": "function", "function": {

"name": "crop", "description": "Crop a region from an image; trigger when target covers < 30% of the frame.", "parameters": {

"type": "object", "properties": {

"image": {"type": "string", "description": "Image reference, e.g. ’img_1’"},

- "x": {"type": "integer", "description": "Top-left X coordinate"},
- "y": {"type": "integer", "description": "Top-left Y coordinate"}, "width": {"type": "integer", "description": "Width of the crop region"}, "height": {"type": "integer", "description": "Height of the crop region"}

}, "required": ["image", "x", "y", "width", "height"]

} }

}, {

"type": "function", "function": {

"name": "text_search", "description": "Serper + JINA Reader + Qwen3-32B summarisation over top-k web pages.", "parameters": {

"type": "object", "properties": {

"q": {"type": "string", "description": "Search query keywords"}, "hl": {"type": "string", "description": "Language code", "default": "en"}, "top_k": {"type": "integer", "description": "#results to summarise", "default": 5}

}, "required": []

} }

}, // ---- Remaining tools follow the same schema; full specs in the tool-suite table ----

{ "function": { "name": "layout_parsing", "required": [] } }, { "function": { "name": "perspective_correct", "required": ["image"] } }, { "function": { "name": "super_resolution", "required": ["image"] } }, { "function": { "name": "sharpen", "required": ["image"] } }, { "function": { "name": "image_search", "required": ["url"] } }

] }

- Figure 8 | Machine-readable tool schema produced by our agent codebase and injected into the model context as an OpenAI-style <tools> block. Two representative tools (crop for visual perception and text_search for knowledge retrieval) are shown in full; the remaining five follow the same schema and are collapsed to their name/required fields—their complete specifications are listed in Table 1. At inference time the agent emits tool calls inside <tool_call>{"name": ..., "arguments": ...}</tool_call> conforming to the parameters schema.

### F Tool Definition and Usage

This section details the search-oriented tools integrated within OpenSearch-VL. For each tool, we articulate its core functionality, formalize its input–output signature, describe its backend implementation, and contextualize its operational role within visual search trajectories. The tool set is partitioned into three functional modalities: Retrieval, Image Enhancement, and Attention & Parsing. Retrieval tools interface with the open web via remote APIs; image enhancement tools operate as lightweight, deterministic local primitives; and attention & parsing tools combine local spatial priors with remote document-understanding services.

##### Retrieval Tools

###### • TextSearch

- – Functionality: A composite textual retrieval mechanism comprising three pipelined stages: (i) a Serper-backend search query to retrieve top-k candidate URLs; (ii) a JINA Reader invocation to fetch and normalize the HTML payloads into clean Markdown; and (iii) a Qwen3-32B summarization pass that distills a query-focused 2–4 sentence synopsis per document. The terminal output is a structured list of [Passage i] (title, url, summary) objects.
- – Operational Role: Functions as the primary “read the web” primitive when surface-level snippets are insufficient. It is indispensable for resolving complex, multi-hop knowledge queries and retrieving long-tail factual evidence that necessitates reasoning over entire paragraphs.

###### • ImageSearch

- – Functionality: A visual-entity and reverse-image search tool powered by the Polaris Lens API. It accepts a publicly routable image URL and returns a structured JSON payload encompassing visually similar images, recognized entities, related domain URLs, and textual captions associated with the query image.
- – Operational Role: Serves as the critical bridge transmuting purely visual queries into retrievable textual entities. When the agent determines that resolving a query hinges on identifying an unknown visual entity (e.g., a landmark, logo, or public figure), it invokes ImageSearch to extract external semantic grounding, which can subsequently be cross-referenced via TextSearch.

##### Image Enhancement Tools

###### • Sharpen

- – Functionality: A deterministic deblurring operator implemented via OpenCV Unsharp Masking. Parameterized by an input image and an optional sharpening intensity α (default 1.5), it computes the enhanced image Iout = (1+α)I − α Gσ ∗ I, where Gσ is a Gaussian blur kernel.
- – Operational Role: Deployed when the input image exhibits pervasive blur or soft edge gradients, which frequently degrade downstream optical character recognition (OCR) or object detection. As a computationally cheap, side-effect-free preprocessing step, the agent can heuristically apply Sharpen prior to reinvoking perceptual tools.

###### • SuperResolution

- – Functionality: A deep-learning-based upscaling tool employing the EDSR architecture via OpenCV’s dnn_superres module. It accepts an image and a discrete scale factor (default ×4), emitting a high-resolution reconstruction. For robustness, if the EDSR weights are unavailable in the deployment environment, it gracefully degrades by returning the original image.
- – Operational Role: Crucial for mitigating resolution bottlenecks, particularly on tightly cropped patches or inherently low-fidelity inputs (e.g., thumbnails). By synthesizing high-frequency details, it substantially elevates the reliability and extraction accuracy of subsequent OCR invocations.

###### • PerspectiveCorrect

- – Functionality: An automated perspective-rectification primitive. It executes a Canny edge detection pipeline on the grayscale projection, extracts the maximal quadrilateral contour, and computes a four-point perspective transform to warp the image into a fronto-parallel plane. If a reliable quadrilateral cannot be established, it falls back to the original image alongside a diagnostic warning.
- – Operational Role: Addresses the pervasive domain shift of real-world captures, such as skewed photographs

of documents, receipts, or screens. Rectifying the image geometry dramatically improves the bounding-box precision and text-recognition fidelity of downstream OCR parsing.

##### Attention and Parsing Tools

###### • Crop

- – Functionality: A deterministic spatial-attention primitive. Parameterized by an image and a bounding box coordinate tuple (x,y,w,h), it extracts and isolates the specified rectangular sub-region, writing the artifact to disk for subsequent reference.
- – Operational Role: Models the human cognitive mechanism of foveating onto a dense sub-region within a cluttered scene (e.g., isolating a single chart in a multi-panel figure). It is the canonical mechanism for the policy to suppress peripheral noise prior to passing a clean, localized patch to downstream tools such as OCR, SuperResolution, or ImageSearch.

###### • OCR

- – Functionality: A layout-aware optical character recognition service backed by a remote PaddleX infrastructure. It consumes a base64-encoded image alongside optional flags for chart-recognition and document-orientation classification. It emits a hierarchical list of detected text blocks (categorized into titles, body text, footnotes, etc.), yielding explicit block_label, block_content, and a reading-order-reconstructed formatted_text.
- – Operational Role: Serves as the agent’s primary “read the image” capability. Beyond plain character recognition, its preservation of the document’s logical layout is indispensable for queries dependent on structural hierarchy (e.g., distinguishing a figure caption from a section title). It is optimally invoked following upstream enhancements (PerspectiveCorrect, Sharpen, SuperResolution) or spatial isolation (Crop).

### G Case Study

- Figure 9 presents a representative end-to-end trajectory of OpenSearch-VL on a knowledge-intensive visual question (“In what year did this bridge open?”), whose answer is recoverable neither from the encoder’s parametric knowledge nor from any single retrieval call. The agent first dispatches a Crop on the roadside signage to foveate on the most diagnostic sub-region, forwards the cropped patch to ImageSearch to identify the structure as the Kessock Bridge, and then issues a targeted TextSearch that corroborates the opening year as 1982. The trace exemplifies the compositional

“verify, don’t guess” behavior that the visual–retrieval action space Tv ∪ Ts, the query-quality reward rquery, and the fatal-aware masking scheme are jointly designed to incentivize: tools are chained in an order that progressively grounds the question, and the rollout terminates as soon as cross-modal evidence converges.

[Figure 57]

- Figure 9 | Case study of OpenSearch-VL. Given an image-based question about the opening year of a bridge, the model first inspects visual evidence and crops the road sign to obtain finer-grained location cues. It then uses image search to identify the bridge as the Kessock Bridge and issues a targeted text search to verify its official opening date. The retrieved evidence confirms that the bridge opened in 1982, leading to the final answer. This example illustrates how interleaved visual inspection, image retrieval, and textual evidence acquisition can resolve knowledge-intensive visual questions.

