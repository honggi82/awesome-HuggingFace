# arXiv:2603.18558v2[cs.CV]26Jun2026

## HIMU: HIERARCHICAL MULTIMODAL FRAME SELECTION FOR LONG VIDEO QUESTION ANSWERING

A PREPRINT

Dan Ben-Ami1 Gabriele Serussi1 Kobi Cohen2 Chaim Baskin1 1INSIGHT Lab, Ben-Gurion University of the Negev, Israel 2Ben-Gurion University of the Negev, Israel

[Figure 1]

Figure 1: HiMu: compositional, multimodal frame selection under a controlled frame budget. Left: flat similarity selectors collapse multimodal, compositional queries into a single embedding and retrieve query-irrelevant frames; HiMu instead parses the query into a logic tree whose leaves (ASR, CLIP, ...) are scored by modality-specific experts and composed into a per-frame satisfaction curve T(t). Right: on Video-MME, HiMu Pareto-dominates prior controlled selectors (AKS, VSLS, BOLT, MDP3) and uniform sampling across frame budgets, matching uniform-atK=64 with only K=16 frames.

### ABSTRACT

Long-form video question answering requires reasoning over extended temporal contexts, making frame selection a critical bottleneck for multi-modal large language models (MLLMs) bound by finite context windows. Within the controlled frame-budget regime that governs practical deployment, prior selectors score frames against a single global query embedding; as a result, compositional multimodal questions that involve temporal ordering or cross-modal cues such as “what happens on screen right after the narrator mentions the reaction?” are flattened into a representation that loses sub-event ordering and modality bindings. We introduce HiMu, a training-free framework

for compositional multimodal frame selection. A single text-only LLM call decomposes the query into a hierarchical logic tree whose leaves are atomic predicates, each routed to a lightweight expert spanning vision (CLIP, open-vocabulary detection, OCR) and audio (speech recognition and non-speech sound matching). Expert signals are normalized, smoothed to align across modalities, and composed bottom-up through fuzzy-logic operators that enforce temporal sequencing and adjacency, yielding a continuous per-frame satisfaction curve. Under the standard 16-frame budget on Video-MME, LongVideoBench, and HERBench-Lite, HiMu achieves state-of-the-art accuracy among frame selection methods and improves over uniform sampling across seven diverse MLLMs as a drop-in module, matching the accuracy of uniform sampling at 4× its frame budget, without retraining and without multiple iterative MLLM calls during selection.

Project Page

[Figure 2]

### 1 Introduction

Long-form video question answering (VideoQA) requires reasoning over extended temporal horizons. Due to the finite context windows of current MLLMs [2, 6, 21], processing entire videos at native frame rates is computationally infeasible. Practical systems must therefore operate within a controlled frame budget: a small, fixed set of frames passed to the MLLM. Within this regime, frame selection becomes a critical bottleneck: regardless of its expressiveness, a model can answer correctly only when given the relevant visual evidence.

The dominant family of controlled frame selectors scores each candidate frame against the query through a frozen vision-language encoder and keeps the top-K [19, 22, 24, 25, 35]. While computationally lightweight, these methods collapse long, compositional queries into a single dense vector, forcing distinct temporal and semantic constraints to be evaluated through one global similarity score. Consider “After the narrator mentions the chemical reaction, what happens to the beaker on the left?”: answering it requires reasoning jointly over audio (the narration) and vision (the beaker’s state) and enforcing their temporal order, but a vision-only encoder is blind to audio and a global similarity score cannot enforce sub-event ordering. A second line of work injects explicit relational operators into selection [13, 32], but is restricted to a small set of predefined relations or incurs dense per-frame grounding costs. Multi-call methods that repeatedly invoke an MLLM within the selection loop [5, 9, 17, 27, 33] reason more richly but operate in a fundamentally different compute class: in the standardized estimates of Table 1, representative systems have several to over ten times higher first-query latency and are outside the controlled frame-budget setting we target.

Underlying both limitations is a single gap: a complex query rarely reduces to one signal or one moment. Its clauses span multiple modalities, including concepts and objects on screen, actions, on-screen text, spoken narration, and even non-speech sounds. Locating the right frames means evaluating each modality across the full timeline, then composing the results under the query’s logical and temporal structure. What is missing is a selector that is at once modality-complete and compositional, yet stays within the controlled frame-budget cost.

We introduce HiMu (Hierarchical Multimodal Frame Selection), a training-free framework that resolves compositional and multimodal structure within the controlled frame-budget setting, without any MLLM call during selection. The core insight is that complex natural-language queries naturally decompose into structured logic trees. HiMu uses a single text-only LLM call to parse the query into a hierarchical tree of atomic predicates; each leaf is routed to a lightweight modality-specific expert spanning visual experts (CLIP, open-vocabulary detection (OVD), and OCR) and audio experts (automatic speech recognition (ASR) and contrastive language-audio pretraining (CLAP)). The localized signals are evaluated over the timeline and composed bottom-up through continuous fuzzy-logic operators, yielding a per-frame satisfaction curve T(t) ∈ [0,1] from which the final frames are sampled. By separating reusable video evidence from question-specific reasoning, HiMu spends its per-query computation on targeted expert grounding and lightweight composition rather than repeated visual-token MLLM calls, keeping it within the latency class of competing controlled selectors. Crucially, the lone LLM call sees only the text query, sidestepping the thousands of visual tokens that dominate the cost of every MLLM-based selector.

Across Video-MME [11], LongVideoBench [29], and HERBench-Lite [3], HiMu achieves state-of-the-art accuracy among controlled frame selection methods. As a plug-and-play module, it consistently improves over uniform sampling for seven diverse MLLMs, ranging from open-source 8B models to proprietary frontier models, without any model-specific tuning. These gains are strongest on longer videos, indicating that query-aware evidence selection can recover relevant context more efficiently than simply increasing uniform coverage.

Our contributions can be summarized as follows:

- • A neuro-symbolic framework for query-aware frame selection that decomposes queries into hierarchical logic trees, routes atomic predicates to modality-specific experts, and is the first controlled frame selector

- to treat audio (ASR and CLAP) as first-class evidence, composing their signals via fuzzy-logic operators enforcing temporal sequencing and adjacency.
- • A training-free, single-shot pipeline that replaces iterative MLLM inference with one text-only LLM planning step and selectively reused lightweight expert evidence, keeping per-query latency within the class of competing controlled selectors and avoiding MLLM calls during selection.
- • State-of-the-art accuracy among controlled frame selectors on Video-MME [11], LongVideoBench [29], and HERBench-Lite [3], with plug-and-play gains across seven MLLMs and consistent advantages across frame budgets, providing evidence that compositional query representation, not larger context alone, is the missing ingredient.

### 2 Related Work

Controlled frame selection methods for long-video QA aim to identify informative frames within a fixed budget, without invoking the downstream MLLM during selection.

Similarity-based frame selection. The most efficient selectors score each frame against the query through a frozen vision-language encoder. BOLT [19] pairs query-frame similarity (e.g., CLIP [22]/ SigLIP [35]) with inversetransform sampling to prioritize relevant frames while preserving selection diversity; AKS [25] recursively splits the timeline, allocating more keyframes to high-scoring segments; and MDP3 [24] formulates selection as a determinantal point process solved via dynamic programming, capturing relevance, diversity, and sequentiality. These methods add minimal overhead, yet they collapse multi-clause queries into a single dense representation, offering limited capacity to preserve sub-event ordering or cross-modal bindings that a query may demand (e.g.distinguishing a spoken reference from a visual action and enforcing their temporal adjacency). Learning-based variants (e.g. Frame-Voyager [34], FFS [4], MLLM-FS [14], VidF4 [16]) train scoring or policy modules for richer signals, but require task-specific supervision. Most selectors in this family compute scores from visual-only features; audio, when available, is typically consumed downstream rather than used to drive selection.

Structured and logic-based selection. A second line of work injects explicit relational or logical operators into selection. T* [32] (detector-based variant) casts temporal search as spatial search, using YOLO-World [7] in an iterative loop that samples frames, detects objects, and reweights the distribution toward relevant evidence. VSLS [13] extends this idea with four predefined dependencies (spatial co-occurrence, temporal proximity, attribute dependency, causal order), using relation checks to iteratively refine the sampling distribution. These methods are meaningful steps toward compositional selection, but their reasoning remains tied to flat object/relation queries and repeated queryconditioned visual grounding, rather than a general nested temporal-logic program over multimodal evidence. HiMu instead parses the query once into a hierarchical tree and composes cached or query-conditioned expert signals in a non-iterative selection pass, avoiding MLLM calls during selection.

Table 1: Comparison of frame selection methods for long-form video QA. Latency columns report selector-side firstquery and cached per-query cost for a 10-minute video (600 frames at 1FPS, K=16), one query, on 8× NVIDIA RTX 6000 Pro GPUs, excluding the final QA call. HiMu latencies are direct measurements, with the breakdown reported in the supplementary material; all other latencies are estimated from reported measurements and normalized to the same video length, sampling rate, query count, and hardware protocol for rough order-of-magnitude comparison rather than an exact apples-to-apples benchmark. Gray rows denote agentic/multi-call methods shown for latency reference only; they are not part of the controlled frame-budget comparison.

Query Representation

Selection Evidence

First-query latency (s)

Per-query latency (s)

Interpretability (Compositional Qs)

TrainFree

Method (Venue)

FFS (CVPR’25) ✗ Learned policy Vis. 0.4 0.2 None VidF4 (NLPCC’25) ✗ Learned score fn. Vis. 1.0 0.3 None BOLT (CVPR’25) ✓ Global embedding Vis. 1.2 0.1 Per-frame score AKS (CVPR’25) ✓ Global embedding Vis. 2.0 0.2 Per-frame score MDP3 (ICCV’25) ✓ Global embedding Vis. 2.0 0.2 Per-frame score

T∗ (CVPR’25) ✓ Flat object queries Vis. (OVD) 7.5 7.5 Detection logs VSLS (NeurIPS’25) ✓ Fixed relation triplets Vis. (OVD) 8.0 7.0 Detection logs

HiMu (ours) ✓ Hierarchical logic tree Vis.+Aud. 4.6 1.9 Frame×Expert scores

Agentic / multi-call methods shown for latency reference only; not part of the controlled frame-budget comparison

VideoZoomer (ICLR’26) [8] ✗ Implicit (iterative VLM) Vis. 15 13 None LVAgent (ICCV’25) [5] ✗ Implicit (multi-agent MLLM) Vis. 34 30 None VCA (ICCV’25) [31] ✓ Implicit (iterative VLM) Vis. ∼60 ∼55 None

Agentic and multi-call VideoQA systems. A separate family of VideoQA systems trades compute for deeper reasoning by repeatedly invoking an MLLM or VLM inside the selection loop, either through agent planners that call tools iteratively [5, 8, 9, 17, 27, 28, 31] or MLLM-in-the-loop scorers [33, 36]. These systems address the broader task of finding answer-relevant evidence in long videos, but their iterative visual-language inference places them in a different compute regime from controlled frame selectors; we therefore include representative methods in Table 1 only as latency references, not as directly comparable selector baselines.

- Table 1 summarizes the gap: controlled-budget selectors either flatten queries into global similarity scores or rely on fixed-relation visual search, while multi-call systems leave the regime; HiMu instead composes visual, speech, and non-speech audio evidence through a hierarchical temporal-logic tree in a single non-iterative selection pass.

### 3 Method

Given a video V = {v1,...,vN} sampled at a fixed rate (e.g.1fps) with its audio track, a natural-language question Q (optionally with answer options), and a frame budget K, HiMu selects the K most question-relevant frames for a single downstream MLLM call. The pipeline (Fig. 2) proceeds in four stages: (i) a text-only LLM decomposes Q into a hierarchical logic tree T (Sec. 3.1); (ii) each leaf is scored by a modality-specific expert, and the resulting signals are lightly post-processed (Sec. 3.2); (iii) signals are composed bottom-up via fuzzy-logic operators into a per-frame satisfaction curve; and (iv) the top-K frames are selected via PASS (Sec. 3.3). Thus, the only LLM interaction inside selection is a text-only tree parsing call; expert scoring then proceeds from cached features or from query-conditioned detector (OVD) when the tree requires it.

##### 3.1 Neuro-Symbolic Query Decomposition

A single text-only LLM call receives Q (with answer options, if multiple-choice) and outputs a hierarchical logic tree T in structured JSON. The tree generation is a text-in, text-out forward pass. The exact system prompt and the JSON schema constraint are provided in the supplementary. The tree has two node types:

Leaf nodes. Each leaf specifies a modality-specific expert and a text query: ℓ = (expert, query), where expert ∈ {CLIP, OVD, OCR, ASR, CLAP} and query is a natural-language atomic predicate (e.g.OVD(“red car”) or ASR(“reaction”)). The LLM routes each predicate to the best-suited expert: actions, scenes, and abstract visual concepts → CLIP; physical objects and people → OVD; on-screen text → OCR; spoken content → ASR; environmental sounds → CLAP. Routing rules and worked examples are in the system prompt.

Internal nodes. Each internal node applies one of four logical or temporal operators to its children:

- • AND: co-occurrence; all children must be active simultaneously.
- • OR: disjunction; at least one child must be active.
- • SEQ: temporal sequence; children are ordered chronologically.
- • RIGHTAFTER: tight temporal adjacency; the effect immediately follows the cause.

MCQ tree pattern. For MCQs, the tree factors shared context from answer-specific branches, typically following the And(shared_context, Or(option_1, ..., option_n)) pattern. Each option branch decomposes into expertspecific predicates (Fig. 3).

##### 3.2 Multimodal Expert Signals Extraction and Processing

Leaf nodes are grouped by expert type for efficient batched inference. Each expert produces a per-frame raw relevance signal ui(t) ∈ R for leaf i at every timestamp t ∈ {1,...,N}. Five experts span two modality categories; to our knowledge, no prior frame selector leverages audio experts, and we show their inclusion is critical for key-moment discovery.

Visual experts. CLIP [22] computes cosine similarity between frame and text-query embeddings, mapped to [0,1]; frame embeddings are extracted once and shared across all CLIP leaves. OVD [7] runs open-vocabulary object detection, returning the maximum detection confidence for the queried class per frame; query variations (singular/plural, with/without adjectives) are generated for robust matching. OCR [20] performs on-screen text recognition with substring and Levenshtein-distance fuzzy matching.

Audio experts. ASR [23] transcribes the audio track once into timestamped word segments; queries are matched via exact substring matching (score = 1.0) or, failing that, semantic similarity via a sentence-embedding model, with

[Figure 3]

Figure 2: The HiMu pipeline. (1) An LLM parses the question into a logic tree of modality-specific experts. (2) Experts (CLIP, ASR, OVD, CLAP) extract raw signals, which are then normalized and smoothed. (3) Fuzzy logic operators compose signals into a temporal satisfaction curve. (4) Top frames are sampled for the MLLM using PASS.

segment scores mapped to frames by temporal-overlap weighting. CLAP [30] computes cosine similarity between frame-aligned audio chunks and the text query for non-speech sounds (environmental sounds, effects, music).

Caching and conditional execution. CLIP, ASR, CLAP, and OCR features are query-independent and cached per video; only OVD is query-conditioned and re-run per query. Unused experts are skipped entirely.

Normalization. Raw expert scores live on incomparable scales (CLIP cosine similarities, OVD confidences, binary ASR matches), each with different ranges and noise profiles. Each signal ui is mapped to (0,1) via:

ui(t) − med(ui) MAD(ui) + δ

u˜i(t) = σ γ ·

, (1)

where med(·) and MAD(·) denote the median and Median Absolute Deviation, σ(·) is the sigmoid, γ controls sharpness, and δ is a small stabilizer. Median/MAD gives robustness to the heavy-tailed distributions typical of detection and retrieval models, and the sigmoid yields a smooth, fuzzy-logic-compatible mapping. When multiple leaves share an expert, statistics are computed jointly over all their signals, preserving relative magnitudes; otherwise independent normalization would stretch a high-confidence detection (0.9, ‘Man’) and a low-confidence one (0.2, ‘Car’) both to [0,1], falsely implying equal relevance and undermining the AND operator.

Bandwidth-matched smoothing. After normalization, each signal is convolved with a modality-specific Gaussian kernel:

N

u˜i(t′) G(t − t′; σm), (2)

uˆi(t) =

t′=1

where G(∆;σ) = √21πσ exp −∆2/2σ2 and m = experti. Visual signals (CLIP, OVD, OCR) are frame-precise and receive narrow kernels; ASR and CLAP have coarser temporal resolution and receive wider kernels. This resolves cross-modal asynchrony by ensuring that peaks from different modalities overlap temporally, preventing missed conjunctions at the composition stage.

[Figure 4]

Figure 3: Logic tree examples on VideoMME questions, combining visual and audio experts.

##### 3.3 Fuzzy Logic Composition

Bottom-up tree evaluation. The logic tree T is evaluated bottom-up: leaf nodes return their processed signals uˆi(t); internal nodes apply continuous fuzzy-logic operators. We use four operators as a compact vocabulary for VideoQA evidence: co-occurrence, alternatives, ordered events, and tight “right after” transitions.

Logical operators. Logical composition follows standard fuzzy operators that keep scores in [0,1]: the product t-norm suppresses AND when any child is weak, while the probabilistic sum lets either branch satisfy OR without unbounded accumulation. Both are applied pairwise left-to-right for n>2 children:

AND(A,B)(t) = A(t) · B(t), (3) OR(A,B)(t) = A(t) + B(t) − A(t) · B(t). (4)

##### Temporal operators.

SEQ (temporal ordering). Given children (u1,...,uL) in chronological order, this operator enforces sequence through past/future evidence rather than brittle timestamp alignment, while still selecting frames from every step:

 uℓ(t) ·

 , (5)

SEQ(t) = max

Hj(t) ·

Fj(t)

ℓ∈{1,...,L}

j<ℓ

j>ℓ

where Hj(t) = maxs<t uj(s) is the has-occurred signal (running max up to t) and Fj(t) = maxs>t uj(s) is the yet-to-occur signal (running max after t). Thus, a step ℓ activates at time t only when earlier steps have already peaked and later steps still peak in the future; the outer max lets all events contribute, not just the final one.

RIGHTAFTER (tight temporal proximity). For event pairs that should happen close together in time, this operator uses exponential decay rather than a fixed window: a frame scores highly when the paired event occurred nearby, with sharpness controlled by κ:

Seffect(t) = effect(t) · s<tcause(s)e−κ(t−s), (6) Scause(t) = cause(t) · s>teffect(s)e−κ(s−t), (7)

RIGHTAFTER(t) = max Seffect(t), Scause(t) . (8)

The two terms ensure frames are selected from both the cause and the effect side: Seffect scores effect frames weighted by how recently the cause fired, while Scause does the reverse.

The root of T produces the satisfaction curve T(t) ∈ [0,1], a per-frame composite score reflecting how well the entire logic tree is satisfied at each timestamp.

PASS: Peak-And-Spread Selection. Naïvely selecting the top-K frames of T(t) over-concentrates on a single highscoring segment, missing other relevant events and short-term motion context. We therefore introduce PASS (PeakAnd-Spread Selection): we first select Np local maxima of T(t) with a minimum inter-peak distance ∆, preventing redundant peaks without forcing coverage of low-satisfaction regions. Each peak is augmented with its Nn highestscoring neighbors within a window of size w, capturing short-term motion while scaling local context with the budget. The remaining budget is filled greedily from the highest-scoring unselected frames of T(t). Each selected frame vt∗ carries its per-leaf scores {uˆi(t∗)}, an interpretable trace of which experts and predicates drove its selection. Detailed pseudocode and a visual comparison for PASS, together with per-leaf heatmap examples illustrating HiMu’s interpretability, are provided in the supplementary material.

### 4 Experiments

We evaluate HiMu along five axes: controlled accuracy against prior frame selectors, plug-and-play gains across downstream MLLMs, the contribution of the compositional design, behavior under different frame budgets, and selector latency.

- 4.1 Setup

Benchmarks. We use three complementary long-video QA benchmarks. Video-MME [11] contains 2,700 multiplechoice questions over 900 videos, split into Short (<2min), Medium (4–15min), and Long (30–60min) durations; its native audio and duration splits test both multimodal grounding and temporal scale. LongVideoBenchval [29] contains roughly 1.3K validation questions with referring queries over 17 categories, where subtitles or Whisper transcripts serve as speech evidence. HERBench-Lite [3] contains 2K purely visual questions requiring integration of at least three non-overlapping cues, stressing whether selected frames can support multi-evidence reasoning once audio is unavailable.

Implementation details. Unless stated otherwise, all methods select K=16 frames from videos sampled at 1fps and answer with the same downstream MLLM within each comparison. HiMu is training-free: the logic tree is generated by the same LLM used as the answerer. The default experts are CLIP-dfn [10], YOLO-World v2 [7] (OVD), docTR [20] (OCR), faster-whisper large-v3-turbo [23] (ASR), and LAION CLAP [30]; all reusable features except query-conditioned OVD are cached per video. All hyperparameters are fixed once across benchmarks, models, and ablations with no per-dataset tuning. The supplementary material reports the exact values, sensitivity analysis, backbone swaps, tree parser comparison, and latency breakdown.

Baselines. The controlled comparison fixes the answerer to Qwen3-VL-8B [2] and compares HiMu to uniform sampling, similarity-based selectors BOLT [19], AKS [25], and MDP3 [24], and structured selectors T∗ [32] and VSLS [13]. The selector inputs follow each benchmark’s modality regime: on Video-MME, frame selectors operate without subtitles, using the video/audio stream; on LongVideoBenchval, audio is unavailable, so selectors may use the benchmark-provided subtitles as the speech-language stream; HERBench-Lite is visual-only. At answer time, the downstream MLLM always receives the same selected frames, question, candidates, and full subtitle/transcript context for Video-MME and LongVideoBenchval, since many questions require speech evidence. This keeps the selector as the only variable within each benchmark: performance differences reflect which visual evidence the selector surfaces rather than whether text evidence is withheld.

- 4.2 Main Results

- (Q1) HiMu is the strongest controlled selector on every benchmark. Table 2 isolates frame selection by us-

ing Qwen3-VL-8B for all methods. HiMu achieves the best score on Video-MME (73.2%), LongVideoBenchval (64.2%), and HERBench-Lite (43.2%), improving over the strongest baseline by +3.2, +5.5, and +1.0pp respectively. Its visual-only configuration, HiMu-Visual, reparses each query using only CLIP, OVD, and OCR; it remains close to full HiMu on Video-MME (72.8%, −0.4pp) and matches the naturally visual HERBench-Lite setting, while LongVideoBenchval shows a larger speech-dependent gap (61.0%, −3.2pp). The Video-MME gain is consistent across all duration splits, including a +3.1pp margin on Medium videos and a +0.7pp margin on Long videos over the best prior selector. The largest benchmark-level improvement appears on LongVideoBenchval, where moment-level referring queries most directly reward query-conditioned multimodal grounding. The smaller but still positive HERBenchLite gain is also informative: even in a visual-only setting where downstream MLLMs still struggle to fuse multiple cues, better selected evidence remains helpful.

- Table 2: Controlled comparison of frame selection methods at K=16 frames with Qwen3-VL-8B on Video-MME,

LongVideoBenchval, and HERBench-Lite. HiMu outperforms all baselines on every benchmark; HiMu-Visual uses only CLIP, OVD, and OCR.

Method (Venue)

Video-MME

LVBval HERBench-Lite Short Medium Long Overall

Uniform Sampling 76.3 66.3 55.6 66.1 55.7 41.7 BOLT [19] (CVPR’25) 69.6 67.9 68.7 68.7 54.6 42.2 T∗ [32] (CVPR’25) 73.7 67.4 68.1 69.8 57.5 39.1 AKS [25] (CVPR’25) 70.1 65.1 68.7 68.0 57.1 40.3 MDP3 [24] (ICCV’25) 75.4 62.4 64.7 67.5 55.6 37.5 VSLS [13] (NeurIPS’25) 74.1 66.7 69.2 70.0 58.7 39.7

HiMu-Visual (ours) 77.8 70.9 69.4 72.8 61.0 43.2 HiMu (ours) 78.6 71.0 69.9 73.2 64.2 43.2

- Table 3: Generalization of HiMu as a plug-and-play frame selector across seven diverse MLLMs at K=16 frames. HiMu delivers broad gains over uniform sampling, with the strongest improvements on Video-MME and LongVideoBenchval.

Video-MME

Method Model

LVBval HERBench-Lite Short Medium Long Overall

Uniform Qwen3-VL-8B 76.3 66.3 55.6 66.1 55.7 41.7 HiMu (Ours) Qwen3-VL-8B 78.6 71.0 69.9 73.2 64.2 43.2

Uniform LLaVA-OV-1.5-8B 72.3 62.3 54.9 63.2 54.3 35.8 HiMu (Ours) LLaVA-OV-1.5-8B 71.9 67.0 63.9 67.6 57.9 35.9

Uniform InternVL-3.5-8B 75.4 67.4 56.6 66.6 59.2 38.3 HiMu (Ours) InternVL-3.5-8B 76.9 70.4 66.5 71.4 64.1 38.3

Uniform Qwen2.5-VL-7B 72.5 61.0 53.8 62.6 54.6 34.1 HiMu (Ours) Qwen2.5-VL-7B 73.1 65.1 62.9 67.1 57.5 35.2

Uniform Gemma-3-12B 73.3 60.3 55.8 63.0 47.6 31.2 HiMu (Ours) Gemma-3-12B 71.6 65.7 67.5 68.3 53.9 31.5

Proprietary MLLMs (evaluated on a stratified random 25% subset of each benchmark)

Uniform Gemini-2.5-Flash 78.4 67.1 61.2 69.0 56.3 37.3 HiMu (Ours) Gemini-2.5-Flash 78.9 75.0 74.5 76.1 70.1 37.7

Uniform GPT-4o 77.0 73.0 71.4 73.8 55.6 37.5 HiMu (Ours) GPT-4o 80.9 77.2 76.5 78.2 65.1 40.7

- (Q2) HiMu transfers as a plug-and-play selector. Table 3 evaluates seven MLLMs under the same HiMu configuration: five open-source models and two proprietary models. Replacing uniform sampling with HiMu improves

Video-MME Overall for every model, with gains from +4.4 to +7.1pp, and improves LongVideoBenchval by +2.9 to +13.8pp. HERBench-Lite also improves or matches uniform for every model, though margins are naturally smaller because the benchmark removes audio/subtitles and stresses the answerer’s multi-frame fusion ability. Across models, the gains concentrate on Medium and Long Video-MME splits; uniform sampling only occasionally matches or exceeds HiMu on Short videos, where dense temporal coverage is easier. This pattern supports the central claim: HiMu’s benefit comes from query-aware evidence selection, not from a special interaction with one MLLM.

4.3 Component Analysis and Frame Budget

- (Q3) The gains come from hierarchical composition, not simply more signals. Table 4 ablates HiMu on VideoMME using Qwen3-VL-8B. The largest drop comes from replacing the logic tree with Flat Fusion (−5.5pp), which pools all leaf scores without hierarchy or typed operators. This loss is larger than removing any individual expert, showing that the main bottleneck is how evidence is composed. The structural rows further separate the two ingredients: flattening the tree while preserving the LLM-chosen root operator still costs −1.4pp, while operator substitutions

- Table 4: Component ablation on Video-MME (Qwen3-VL-8B, K=16). Compositional structure isolates hierarchy from operator semantics: “Flat Fusion” pools all leaf scores with no tree or typed operators; “w/o nesting” flattens the tree but keeps the LLM-chosen root operator (isolating hierarchy); each “X→Y ” row holds the topology fixed and swaps one operator type (isolating that operator). Experts removes one expert at a time.

Video-MME Short Med. Long Overall HiMu 78.6 71.0 69.9 73.2 Compositional structure

Configuration

Flat Fusion 68.7 66.1 68.5 67.7 w/o nesting 76.9 70.6 67.6 71.8 OR→AND 77.5 70.4 68.5 72.2 SEQ→AND 76.8 69.2 68.5 71.5 RIGHTAFTER→AND 77.4 69.3 68.8 71.9 AND→OR 74.0 68.4 69.7 70.7

Experts (leave-one-out)

w/o ASR 76.3 69.1 68.1 71.2 w/o CLAP 78.0 69.4 69.1 72.2 w/o CLIP 76.5 69.6 69.2 71.8 w/o OCR 77.4 69.9 69.1 72.2 w/o OVD 77.4 70.6 69.2 72.5

cost −1.0 to −2.5pp. Conjunction is the most sensitive operator (AND→OR, −2.5pp), but temporal operators also matter (SEQ→AND, −1.7pp; RIGHTAFTER→AND, −1.3pp). Thus the improvement is not reducible to temporal logic alone; it comes from the joint hierarchy-plus-operator design.

Expert leave-one-out uses a fixed-tree intervention: the parser output is held fixed and all leaves of one expert are neutralized before composition (constant 1 under AND and temporal parents, constant 0 under OR). This isolates each expert’s marginal evidence contribution, while HiMu-Visual in Table 2 measures the adaptive visual-only setting by reparsing with the visual expert set. Among experts, ASR has the largest leave-one-out effect (−2.0pp), followed by CLIP (−1.4pp), OCR (−1.0pp), CLAP (−1.0pp), and OVD (−0.7pp). Because Video-MME selectors do not receive subtitles and every ablation still feeds the same full subtitles to the answerer, the ASR drop reflects better frame selection aligned to speech cues from the audio track, not extra text made available to the MLLM. The supplementary material strengthens this conclusion: swapping expert backbones changes accuracy by at most 0.6pp, and changing the tree parser stays within a 0.8pp range, indicating that the compositional interface is more important than a particular pretrained expert or parser.

- (Q4) HiMu keeps its advantage across frame budgets. Table 5 varies K ∈ {8,16,32,64}. HiMu improves VideoMME Overall over uniform sampling at every budget, by +6.4, +7.1, +4.9, and +4.1pp respectively. The gains are concentrated on Medium and Long videos, where uniform sampling is most likely to miss query-relevant moments; on Short videos, larger budgets make most reasonable samplers competitive because much of the clip is already covered. Most importantly, HiMu with only K=16 frames (73.2%) exceeds uniform sampling with K=64 frames (71.7%). This 4× frame-budget reduction shows that the bottleneck is not only context length: many uniformly sampled frames are simply not the evidence requested by the question.

4.4 Efficiency Analysis

- (Q5) HiMu trades modest selector cost for the best controlled accuracy. Table 1 reports standardized selectoronly latency for a 10-minute video at 1FPS on 8× NVIDIA RTX 6000 Pro GPUs, excluding the final QA call that every method pays. Lightweight learned and similarity-based selectors remain fastest (0.4–2.0s first-query latency; 0.1–0.3s per-query latency) but remain below HiMu in Table 2. HiMu’s measured selector latency is 4.6s for the first query and 1.9s for each cached per-query run. Structured selectors with heavier query-conditioned grounding, T∗ and VSLS, require 7.5–8.0s first-query latency and 7.0–7.5s per-query latency, while VideoZoomer, LVAgent, and VCA are included only as latency references and are slower by a wider margin under the standardized estimates. The supplementary material breaks HiMu’s cost down: reusable CLIP/OCR/ASR/CLAP features are cached, while the per-query cost is dominated by tree parsing and OVD. This is the intended trade-off: HiMu spends a few seconds beyond global visual retrieval to obtain compositional, visual-plus-audio evidence selection, per-frame expert traces, and the best accuracy across all controlled benchmarks.

- Table 5: Effect of frame budget K on Video-MME (Qwen3-VL-8B). HiMu consistently improves Overall and is especially strong on Medium and Long videos.

Video-MME Short Med. Long Overall 8

# Frames Method

Uniform 73.5 63.4 54.7 64.0 HiMu 73.5 68.7 68.8 70.4

Uniform 76.3 66.3 55.6 66.1 HiMu 78.6 71.0 69.9 73.2

16

Uniform 81.4 69.2 58.7 69.9 HiMu 80.8 73.3 70.0 74.8

32

Uniform 82.9 71.1 60.6 71.7 HiMu 82.4 73.5 71.1 75.8

64

### 5 Discussion

HiMu shows that compositional, multimodal frame selection is achievable within the controlled frame-budget regime without any MLLM call during selection, by decomposing each query into a hierarchical logic tree of lightweight modality-specific experts that treat audio as compositionally typed, first-class evidence. The broader implication is that controlled-budget selection is not merely a compromise between quality and cost: much of what larger uniform budgets buy is redundant coverage that a structured query representation can recover more directly.

Limitations. HiMu makes a deliberate trade-off between accuracy and cost: expert extraction is slower than globalembedding retrieval, but its measured latency remains below detector-heavy structured selectors and far below multicall agentic systems in the standardized selector-latency comparison. Its quality also depends on faithful tree parsing and on the coverage of the underlying experts, especially ASR for multilingual or noisy speech. A remaining bottleneck lies downstream: even with the right frames, the MLLM must still fuse evidence across them. By separating evidence selection from evidence fusion, HiMu provides a controlled substrate for studying this next step.

### References

- [1] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, Huajie Tan, Chunyuan Li, Jing Yang, Jie Yu, Xiyao Wang, Bin Qin, Yumeng Wang, Zizhen Yan, Ziyong Feng, Ziwei Liu, Bo Li, and Jiankang Deng. LLaVA-OneVision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025. 20
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025. 2, 7, 20
- [3] Dan Ben-Ami, Gabriele Serussi, Kobi Cohen, and Chaim Baskin. HERBench: A benchmark for multi-evidence integration in video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2026. 2, 3, 7

- [4] Shyamal Buch, Arsha Nagrani, Anurag Arnab, and Cordelia Schmid. Flexible frame selection for efficient video reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 29071–29082, 2025. 3
- [5] Boyu Chen, Zhengrong Yue, Siran Chen, Zikang Wang, Yang Liu, Peng Li, and Yali Wang. LVAgent: Long video understanding by multi-round dynamical collaboration of MLLM agents. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20237–20246, 2025. 2, 3, 4
- [6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24185–24198, 2024. 2
- [7] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. YOLO-World: Real-time openvocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16901–16911, 2024. 3, 4, 7, 19
- [8] Yang Ding, Xin Lai, Yizhen Zhang, Wei Li, Ruihang Chu, and Yujiu Yang. Videozoomer: Reinforcement-learned temporal focusing for long video reasoning. In The Fourteenth International Conference on Learning Representations, 2026. 3, 4
- [9] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In Computer Vision: ECCV 2024, 18th European Conference, Milan, Italy, September 29 to October 4, 2024, Proceedings, Part XXII, pages 75–92, Berlin, Heidelberg, 2024. Springer-Verlag. 2, 4, 21

- [10] An Fang, Andrew M Jose, Anmol Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. In International Conference on Learning Representations, 2024. 7, 19
- [11] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. In Advances in Neural Information Processing Systems, 2024. 2, 3, 7
- [12] Gemini Team, Google. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 20
- [13] Weiyu Guo, Ziyang Chen, Shaoguang Wang, Jianxiang He, Yijie Xu, Jinhui Ye, Ying Sun, and Hui Xiong. Logic-in-frames: Dynamic keyframe search via visual semantic-logical verification for long video understanding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 2, 3, 7, 8, 21
- [14] Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, and Trishul Chilimbi. M-LLM based video frame selection for efficient video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13702–13712, 2025. 3
- [15] JaidedAI. EasyOCR: Ready-to-use OCR with 80+ supported languages, 2020. 19
- [16] Jianxin Liang, Xiaojun Meng, Yueqian Wang, Chang Liu, Qun Liu, and Dongyan Zhao. End-to-end videoqa with frame scoring mechanisms and adaptive sampling. In Natural Language Processing and Chinese Computing (NLPCC 2025), pages 141–154. Springer, Singapore, 2026. 3
- [17] Runtao Liu, Ziyi Liu, Jiaqi Tang, Yue Ma, Renjie Pi, Jipeng Zhang, and Qifeng Chen. Longvideoagent: Multi-agent reasoning with long videos. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL), 2026. 2, 4
- [18] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In European Conference on Computer Vision (ECCV), 2024. 19
- [19] Shuming Liu, Chen Zhao, Tianqi Xu, and Bernard Ghanem. Bolt: Boost large vision-language model without training for long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3318–3327, 2025. 2, 3, 7, 8, 21
- [20] Mindee. docTR: Document text recognition, 2021. Open-source OCR library. 4, 7, 19
- [21] OpenAI, Aaron Hurst, Adam Lerer, et al. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2
- [22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021. 2, 3, 4
- [23] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning, pages 28492– 28518, 2023. 4, 7, 19
- [24] Hui Sun, Shiyin Lu, Huanyu Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Ming Li. Mdp3: A training-free approach for list-wise frame selection in video-llms. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 24090–24101, 2025. 2, 3, 7, 8, 21
- [25] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 29118–29128, 2025. 2, 3, 7, 8, 21
- [26] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 20
- [27] Xuan Wang, Yiming Zhang, Omer Zohar, and Sivan Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision (ECCV), pages 58–76, 2024. 2, 4
- [28] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. VideoTree: Adaptive tree-based video representation for LLM reasoning on long videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3272–3283, 2025. 4
- [29] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for long-context interleaved videolanguage understanding. In Advances in Neural Information Processing Systems, pages 28828–28857, 2024. 2, 3, 7
- [30] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023. 5, 7, 19
- [31] Zeyuan Yang, Delin Chen, Xueyang Yu, Maohao Shen, and Chuang Gan. VCA: Video curious agent for long video understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20168–20179, 2025. 3, 4

- [32] Jinhui Ye, Zihan Wang, Haosen Sun, Keshigeyan Chandrasegaran, Zane Durante, Cristobal Eyzaguirre, Yonatan Bisk, Juan Carlos Niebles, Ehsan Adeli, Li Fei-Fei, Jiajun Wu, and Manling Li. Re-thinking temporal search for long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8579–8591, 2025. 2, 3, 7, 8, 21
- [33] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. SeViLA: Self-chained video localization and answering via llm. In Advances in Neural Information Processing Systems, 2023. 2, 4, 21
- [34] Sicheng Yu, Chengkai Jin, Huanyu Wang, Zhenghao Chen, Sheng Jin, Zhongrong Zuo, Xiaolei Xu, Zhenbang Sun, Bingni Zhang, Jiawei Wu, Hao Zhang, and Qianru Sun. Frame-voyager: Learning to query frames for video large language models. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [35] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11975–11986, 2023. 2, 3, 19
- [36] Yuanhao Zou, Shengji Jin, Andong Deng, Youpeng Zhao, Jun Wang, and Chen Chen. A.i.r.: Enabling adaptive, iterative, and reasoning-based frame selection for video question answering. In The Fourteenth International Conference on Learning Representations, 2026. 4

[Figure 5]

- Figure 4: HiMu pipeline overview. Given a natural-language query, an LLM decomposes it into a hierarchical logic tree whose leaves are routed to modality-specific experts (OCR, CLIP, ASR, OVD, CLAP). Each expert produces a perframe relevance signal over time; these signals are composed bottom-up via fuzzy-logic operators into a satisfaction curve T(t). The top-scoring frames are selected and passed to the multi-modal large language model (MLLM) for answering.

Supplementary Material

- A Neuro-Symbolic Query Decomposition Details

- A.1 System Prompt for Tree Generation

The system prompt guides the LLM to decompose each question into a hierarchical logic tree. Because different benchmarks provide different audio modalities, the prompt is adapted per benchmark by enabling or disabling expert descriptions and their associated rules. We first present the common prompt shared across all benchmarks (Listing 1), then summarize the per-benchmark adaptations in Table 6.

Listing 1: Common system prompt for logic tree generation (shared across all benchmarks). Sections marked [IF_ASR] and [IF_CLAP] are included only when the corresponding expert is enabled (see Table 6).

You are a Neuro -Symbolic Logic Parser for a Multimodal

Video Understanding system. Convert Video QA questions into structured logical trees that leverage ALL relevant modalities.

### THE EXPERTS

**Visual Experts :**

- 1. **OVD** - Open -Vocabulary Object Detection (YOLO -World)

- - For: Physical objects , people , and visual attributes

- - Examples: "person", "car", "dog", "red car", "man in suit"

- - Supports attribute+noun phrases. Never include numbers , counts , or proper names.

- - OVD CANNOT detect: countries , teams , names , states , emotions , actions , text , numbers

- 2. **OCR** - On-Screen Text Recognition

- - For: Text visible on screen - signs , labels , jersey numbers , names , scoreboards

- - Examples: "Exit", "10", "Warning", "Korea"

- 3. **CLIP** - Semantic Visual Understanding

- - For: Actions , scenes , visual states , atmosphere , abstract visual concepts

- - CLIP is VISUAL ONLY - queries must describe something you can SEE in a video frame

- - Good: "person speaking", "diagram on screen", "cooking"

- - Bad: "quantum entanglement", "innovation" (invisible concepts)

[IF_ASR]

- 4. **ASR** - Speech Recognition

- - For: Spoken words , dialogue , narration , verbal references

- - Use SHORT keywords (1-3 words), never full sentences

- - CRITICAL: People TALK about what is shown. Add an ASR leaf with related spoken keywords alongside visual leaves.

[/IF_ASR] [IF_CLAP]

- 5. **CLAP** - Environmental Audio Events

- - For: Non -speech sounds , music , sound effects , ambient audio

- - Examples: "doorbell ringing", "applause", "glass breaking"

[/IF_CLAP] ### THE OPERATORS

- - **AND**: All children must co-occur (same frame).

- - **OR**: At least one child satisfied.

- - **SEQ**: Temporal sequence (earliest first , latest last). ONLY when order is EXPLICITLY STATED. IMPORTANT: If the question ASKS about order/sequence , do NOT use SEQ. Instead: build one AND per shot , wrap in a flat OR. Each shot appears exactly once.

- - **RIGHT_AFTER **: Immediate temporal succession. Exactly 2 children [cause , effect].

### KEY RULES

- 1. MULTIMODAL: Each MCQ option should combine more than one expert type , [IF_ASR: visual AND audio evidence when possible. Never make a tree

with only one expert type.] [ELSE: Use multiple visual experts when possible.]

[IF_ASR]

- 2. ASR OVERLAP: Add ASR leaves with short keywords alongside visual leaves - narrators often describe what is shown.

[/IF_ASR]

- 3. MCQ STRUCTURE: AND(shared_context , OR(opt_1 , opt_2 , ...))

- factor shared elements OUT of the OR.

- 4. DECOMPOSE RICH DESCRIPTIONS: Create separate leaves for each element: OVD for objects/people , CLIP for settings/states.

- 5. SEQ ONLY FOR KNOWN ORDER.

- 6. NAMES -> OCR [IF_ASR: + ASR (spoken)].

- 7. VISUAL STATES -> CLIP , not ASR alone.

- 8. META -OPTIONS: "Same", "All of the above", etc. -> ALWAYS skip in the OR.

- 9. ACTIONS IN OPTIONS: AND(OVD:object , CLIP:action).

- 10. TEMPORAL CAUSE: cause child = action (CLIP), not person.

- 11. OVERLAPPING EXPERTS encouraged. [IF_ASR]

- 12. VISUAL GROUNDING: Never build ASR -only options.

- 13. OVERLAPPING PREDICATES: Mix experts with overlapping terms for robust detection.

- Table 6: Per-benchmark prompt adaptations for HiMu’s logic-tree parser. The common prompt (Listing 1) is shared across all benchmarks; only the dimensions shown below differ per benchmark.

Video-MME LongVideoBench HERBench-Lite

Active experts OVD, OCR, CLIP, OVD, OCR, CLIP, OVD, OCR, CLIP ASR, CLAP ASR

[IF_ASR] included ✓ ✓ ✗ [IF_CLAP] included ✓ ✗ ✗

Rules 2, 12–14 ✓ ✓ ✗ Prompt examples 4 (audio, OCR+ASR, 3 (speech, scientific, 3 (MCQ, ordering,

multimodal, temporal) temporal) OCR+visual)

14. SUBTITLES -> ASR + OCR. [/IF_ASR]

### OUTPUT FORMAT Return a single JSON object: {"op": "AND"|"OR"|"SEQ"|" RIGHT_AFTER "|"LEAF",

"children": [...], "expert": <available experts >, "query": "string"}

The prompt also includes three to four worked examples tailored to each benchmark’s modality regime (e.g., CLAP-based examples for Video-MME, speech-triggered examples for LongVideoBench, visual-only examples for HERBench-Lite). The complete per-benchmark prompts including all examples will be released.

##### A.2 JSON Schema Constraint

The LLM output is constrained to valid JSON matching the following schema. Leaf nodes carry expert and query fields; internal nodes carry an op and a children array.

Listing 2: JSON schema for logic tree output.

{

"op": "AND" | "OR" | "SEQ" | "RIGHT_AFTER" | "LEAF", "children": [<recursive tree nodes >], // non -LEAF only "expert": "CLIP"|"OVD"|"OCR"|"ASR"|"CLAP", // LEAF only "query": "<atomic predicate string >" // LEAF only

}

The available values for expert are restricted per benchmark to match the active expert set (Table 6).

###### A.3 Expert Routing Rules The system prompt encodes the following routing logic, which the LLM applies when assigning predicates to experts:

- • Physical objects and people → OVD (e.g., “red car”, “man in suit”).
- • Actions, scenes, visual states → CLIP (e.g., “person running”, “sunset”).
- • On-screen text → OCR (e.g., “Exit”, “Warning”, jersey numbers).
- • Spoken content and dialogue → ASR (e.g., “reaction”, “careful”).
- • Environmental sounds → CLAP (e.g., “applause”, “glass breaking”).
- • Proper names → OCR + ASR (text on screen and spoken references).
- • Actions with objects → AND(OVD:object, CLIP:action) for robust detection.

Worked example. For the question “After the doorbell rings, who opens the door?” with options [“A man”, “A woman”], the parser produces a tree that grounds the doorbell with the audio expert, grounds the door-opening event visually, and keeps the multiple-choice hypotheses under an OR:

{

"op": "RIGHT_AFTER", "children": [

{

"op": "LEAF", "expert": "CLAP", "query": "doorbell ringing"

}, {

"op": "AND",

"children": [ {

"op": "LEAF", "expert": "CLIP", "query": "person opens the door"

}, {

"op": "OR", "children": [

{

"op": "LEAF", "expert": "OVD", "query": "man"

}, {

"op": "LEAF", "expert": "OVD", "query": "woman"

} ]

} ]

} ]

}

The RIGHT_AFTER root captures the local causal relation expressed by the doorbell cue: high-scoring frames are those in which the visual door-opening event follows closely after the ring. The effect branch is an AND, requiring both the door-opening action and one of the candidate people, while the nested OR represents the answer alternatives.

### B Implementation Details

Table 7 lists all hyperparameter values used throughout the paper. Parameters are grouped by the pipeline stage they belong to, with references to the corresponding equations and sections in the main paper.

### C PASS Algorithm

[Figure 6]

- Figure 5: PASS vs. naive top-K selection. Given the multimodal satisfaction curve T(t), naive top-K (left, red) concentrates all selected frames around the single highest peak, missing other relevant events. PASS (right, blue) detects multiple peaks, spreads neighbors around each, and fills the remaining budget greedily, yielding diverse temporal coverage.

Algorithm 1 provides the pseudocode for PASS (Peak-And-Spread Selection). Phase 1 identifies Np prominent peaks in the satisfaction curve while enforcing a minimum temporal separation of ∆ frames, preventing redundant selection from a single high-scoring segment without artificially requiring coverage of the full timeline. Phase 2 augments each peak with its Nn highest-scoring neighbors within a local window of w frames, capturing short-term motion context. Phase 3 fills the remaining budget greedily from the highest-scoring unselected frames.

PASS vs. vanilla top-K. As illustrated in Fig. 5, on the full Video-MME test set (Qwen3-VL-8B, K=16), PASS achieves 73.23% compared to 72.37% for vanilla top-K selection (+0.86pp). Vanilla top-K tends to concentrate all selected frames within a single high-scoring segment, missing relevant events elsewhere in the video. PASS mitigates

Table 7: Complete hyperparameter settings for HiMu.

Parameter Symbol Value Description Normalization (Eq. 1, Sec. 3.2 in the main paper)

Sigmoid sharpness γ 3.0 Controls the steepness of the sigmoid in median/MAD normalization; higher values yield sharper contrast between high- and low-scoring frames

MAD stabilizer δ 10−6 Prevents division by zero when all ex-

pert scores are identical (MAD=0) Temporal operators (Eqs. 6–8, Sec. 3.3 in the main paper)

RightAfter decay κ 2.0 Exponential decay rate for the RIGHTAFTER operator; larger values restrict causal pairing to temporally closer frames

Bandwidth-matched smoothing (Eq. 2, Sec. 3.2 in the main paper) CLIP smoothing σclip 0.5 Narrow kernel for frame-precise visual

similarity scores OVD smoothing σovd 0.5 Narrow kernel for frame-precise object detection confidences OCR smoothing σocr 0.5 Narrow kernel for frame-precise onscreen text detections ASR smoothing σasr 1.5 Wider kernel to bridge the coarser temporal resolution of speech transcripts

CLAP smoothing σclap 2.0 Widest kernel to account for the coarse temporal granularity of environmental audio events

PASS selection (Sec. 3.3 in the main paper) Number of peaks Np ⌊

√

K⌋ Number of peaks to detect; scales sublinearly with K to encourage coverage of multiple relevant events rather than concentrating on a single segment

√

Neighbors per peak Nn ⌊

K/2⌋ Neighbors added around each peak to capture fine-grained details, small movements, and short-term temporal changes within each key moment

√

Local window size w ⌊

K⌋ Total width of the neighbor search window centered at each peak (⌊w/2⌋ frames per side); equal to ∆ to prevent overlap between adjacent peak neighborhoods

√

Min inter-peak distance ∆ ⌊

K⌋ Minimum frame separation between peaks; uses the same formula as Np (⌊

√

K⌋); this equality ensures nonoverlapping temporal neighborhoods and diverse event coverage

this by explicitly detecting multiple peaks and spreading neighbors around each, ensuring coverage of distinct temporal events while preserving fine-grained context around each key moment.

### D Sensitivity Analysis

Unless otherwise noted, sensitivity experiments use Qwen3-VL-8B as the downstream MLLM with K=16 frames on the full Video-MME test set. Each experiment changes exactly one component from the default configuration, isolating the contribution of each design choice.

Algorithm 1 PASS: Peak-And-Spread Selection Require: Satisfaction curve T(t) for t ∈ {1,...,N}, frame budget K, number of peaks Np, neighbors per peak Nn,

window size w, min inter-peak distance ∆

Ensure: Selected frame indices S with |S| = K

- 1: S ← ∅
- 2:
- 3: // Phase 1: Peak detection
- 4: P ← ∅ // set of selected peak indices
- 5: C ← {t : T(t) > T(t−1) and T(t) > T(t+1)} // find all local maxima in the satisfaction curve
- 6: Sort C by T(t) in descending order // process highest-scoring peaks first
- 7: for t ∈ C do
- 8: if |t − t′| ≥ ∆ for all t′ ∈ P then
- 9: P ← P ∪ {t} // enforce min separation: skip peaks too close to existing ones
- 10: end if
- 11: if |P| = Np then
- 12: break // enough diverse peaks collected
- 13: end if
- 14: end for
- 15: S ← S ∪ P
- 16:
- 17: // Phase 2: Neighbor spread // add nearby frames around each peak to capture local temporal context
- 18: for each peak p ∈ P do
- 19: Wp ← {t : |t − p| ≤ ⌊w/2⌋, t ∈/ S} // candidate neighbors within window w around peak p
- 20: Sort Wp by T(t) in descending order
- 21: S ← S ∪ Wp[1 : Nn] // keep top-Nn: best context frames around this peak
- 22: end for
- 23:
- 24: // Phase 3: Greedy fill // use remaining budget on highest-scoring uncovered frames
- 25: R ← {1,...,N} \ S // frames not yet selected
- 26: Sort R by T(t) in descending order
- 27: S ← S ∪ R[1 : (K − |S|)] // fill to budget K
- 28: return S

##### D.1 Hyperparameter Sensitivity

We systematically vary the smoothing bandwidths, temporal decay factor, and sigmoid sharpness one at a time (Table 8). All perturbations remain within at most 0.71pp of the baseline, confirming that HiMu is not tightly coupled to any single hyperparameter setting. This stability stems from the neuro-symbolic architecture: frame relevance is determined by the compositional structure of the logic tree rather than by raw expert scores, so moderate changes to score processing have limited effect on the final frame ranking.

Smoothing bandwidths. Smoothing has a small overall effect: disabling it entirely (+0.49pp) or zeroing visual smoothing (+0.26pp) leaves accuracy essentially unchanged, while over-smoothing the visual signal (visual σ=2) is the most harmful setting (−0.71pp), blurring shot boundaries. For speech, removing smoothing has no effect (0.00pp) and over-smoothing (σasr=4) costs only −0.41pp, confirming that the default σasr=1.5 is a safe choice.

Temporal decay κ. Across a wide range of decay rates (κ ∈ {0.5,1.0,4.0,8.0}), accuracy varies by at most −0.08pp, showing that the RIGHTAFTER operator is highly robust to the exact decay setting.

Sigmoid sharpness γ. Reducing the sigmoid contrast (γ=1.0) slightly improves accuracy (+0.56pp), while increasing it (γ=5.0) costs −0.23pp; both remain within 0.56pp of the default, demonstrating that the median/MAD normalization is robust to the exact sharpness setting.

##### D.2 Expert Backbone Ablation

To evaluate whether HiMu’s gains depend on a specific set of pretrained models, we swap each expert’s underlying backbone one at a time (Table 9). All substitutions stay within 0.6pp of the default, with Grounding DINO (+0.52pp) and the music-speech CLAP (+0.07pp) even yielding marginal improvements. This confirms that HiMu’s strength lies

- Table 8: Hyperparameter sensitivity on the full Video-MME test set (Qwen3-VL-8B, K=16). Each row changes one hyperparameter from the default. ∆ is the difference from the baseline.

Configuration Accuracy ∆ HiMu (default) 73.23 0.00 Smoothing bandwidths All smoothing disabled 73.72 +0.49 Visual σ (CLIP/OVD/OCR, default: 0.5) → 0 73.49 +0.26 Visual σ (CLIP/OVD/OCR, default: 0.5) → 2 72.52 −0.71 Speech σ (ASR, default: 1.5) → 0 73.23 0.00 Speech σ (ASR, default: 1.5) → 4 72.82 −0.41 Temporal decay factor κ (default: 2.0)

- κ=0.5 73.19 −0.04
- κ=1.0 73.19 −0.04 κ=4.0 73.15 −0.08 κ=8.0 73.15 −0.08

Sigmoid sharpness γ (default: 3.0)

γ=1.0 73.79 +0.56 γ=5.0 73.00 −0.23

- Table 9: Component-level ablation on the full Video-MME test set (Qwen3-VL-8B, K=16): expert backbone swaps and selection strategy. Each row changes one component from the default configuration. ∆ is the difference from the baseline.

Configuration Accuracy ∆ HiMu (default) 73.23 0.00 Expert backbone swap

CLIP-dfn [10] → SigLIP2 [35] 73.15 −0.08 YOLO-World v2 [7] → Grounding DINO [18] 73.75 +0.52 docTR [20] → EasyOCR [15] 72.63 −0.60 faster-whisper large-v3-turbo [23] → whisper-large [23] 72.63 −0.60 LAION CLAP [30] → CLAP music-speech [30] 73.30 +0.07

Selection strategy PASS → Vanilla top-K 72.37 −0.86

in the hierarchical composition of multimodal signals rather than in any single expert backbone, allowing practitioners to substitute backbones based on deployment constraints with minimal accuracy degradation.

##### D.3 LLM Tree Parser Comparison

In the default configuration, Qwen3-VL-8B serves as both the tree parser and the downstream MLLM answerer; here we hold the answerer fixed (Qwen3-VL-8B) and vary only the LLM that generates the logic tree, evaluating on the full Video-MME test set (Table 10). All four parsers, spanning open-source and proprietary models of different scales, achieve accuracy within a 0.78pp range. This tight spread demonstrates that HiMu’s structured prompt and JSON schema constraint (Appendix A.2) effectively guide diverse LLMs toward trees that yield near-equivalent selection accuracy, making the system robust to the choice of tree parser. The proprietary Gemini-2.5-Flash is marginally best (+0.07pp), and even the weakest parser (LLaVA-OV-1.5-8B) reaches 72.52%, only 0.71pp below the default.

##### D.4 Component Ablation Protocol

The main paper reports expert, visual-only, and structural ablations on the full Video-MME test set with Qwen3-VL8B as the downstream MLLM and K=16 selected frames; the full-tree reference scores 73.23% overall.

Expert leave-one-out. The expert rows use a fixed-tree intervention: each question is parsed once with the full expert set, then all leaves of the ablated expert are neutralized before score composition. A removed leaf is replaced by the neutral constant of its parent operator: 1 under AND, SEQ, and RIGHTAFTER, and 0 under OR. This keeps the

- Table 10: Effect of LLM tree parser on accuracy over the full Video-MME test set (Qwen3-VL-8B as MLLM answerer). ∆ is the difference from the Qwen3-VL baseline.

Tree Parser LLM Accuracy ∆ Qwen3-VL-8B [2] (default) 73.23 0.00

Gemini-2.5-Flash [12] 73.30 +0.07 InternVL-3.5-8B [26] 72.82 −0.41 LLaVA-OV-1.5-8B [1] 72.52 −0.71

query decomposition, topology, remaining expert outputs, answerer, and frame budget fixed, isolating the marginal evidence carried by the removed expert.

Visual-only configuration. HiMu-Visual evaluates the adaptive visual-only setting: the tree parser prompt and JSON schema expose only CLIP, OVD, and OCR, so each question is reparsed before scoring with the visual experts. This measures the deployable behavior of HiMu when speech and non-speech audio experts are unavailable.

Structural ablations. Each structural condition perturbs the parsed logic tree before score composition while keeping the expert outputs, downstream answerer, and frame budget fixed.

Operator substitutions. The four operator substitutions leave the tree topology byte-identical (same internal nodes, same children, same leaves and DFS order; leaves are never modified); only the fusion function at each matching node changes. Per-frame signals are rescaled to [0.5,1.0] before each operator, so a substitution alters how children are combined but not the underlying evidence:

- • AND→OR (NO_AND): every conjunction becomes a disjunction (probabilistic sum), giving the loosest reading, “any child active” instead of “all children active”.
- • OR→AND (NO_OR): every disjunction becomes a conjunction (product t-norm), giving the strictest reading, “all” instead of “any”.
- • SEQ→AND (NO_SEQ): temporal ordering is dropped, retaining only product co-occurrence of the same children.
- • RIGHTAFTER→AND (NO_RIGHT_AFTER): the exponentially-decayed cause→effect coupling (κ=2) is dropped, retaining only co-occurrence.

Flattened hierarchy. The w/o nesting condition (NO_NESTING) is the only one that changes topology: all leaves are collected depth-first and wrapped under a single node that applies the tree’s original root operator. This removes hierarchy while preserving the LLM’s top-level operator choice. It is therefore milder than Flat Fusion, which additionally forces a generic flat combine (SIMPLE_OR) rather than the query-appropriate root operator.

- E Selector Latency Breakdown

- Table 11 reports HiMu’s selector-only forward-pass compute for a 10-minute video sampled at 1FPS (600 candidate frames, K=16) on 8× RTX 6000 Pro GPUs; OCR is evaluated at half rate on 300 sampled frames. We measure the selector inference path with model weights loaded and prepared inputs already placed on the assigned devices.

Preprocessing. Cacheable preprocessing runs once per video and extracts reusable evidence before question-specific composition. CLIP encodes 600 frames, CLAP scores 300 two-second audio windows, OCR processes 300 sampled frames (∼2.5k crops), and Whisper transcribes the full 600s audio track. The half-rate OCR schedule is an algorithmic design choice: on-screen text such as road signs, digital displays, and text printed on objects is typically less dynamic than objects, events, and visual concepts, so dense 1FPS OCR provides limited additional evidence relative to its cost. Under the 8-GPU schedule, the preprocessing critical path is 2.74s, governed by the slowest parallel branch; the TFLOP entry in the total row reports the busiest compute branch used for the latency accounting.

Per-query processing. Each query runs the Qwen3-8B text parser, query-conditioned YOLOv8x-worldv2 when an OVD leaf appears, lightweight CLIP-text/cosine and ASR matching, and CPU composition with PASS. The per-query selector critical path is 1.88s; the first-query selector cost is 4.63s, given by 2.74s cacheable preprocessing plus 1.88s per-query processing up to rounding. Most video evidence is therefore amortized across questions: after caching, the query-specific path is dominated by the text parser, while OVD, scoring, and composition remain sub-second.

Table 11: Per-component selector forward-pass compute and latency for HiMu on 8× RTX 6000 Pro GPUs. A 10-minute video at 1FPS yields 600 candidate frames; OCR uses 300 half-rate sampled frames. Preprocessing is cacheable once per video; per-query stages run for each question. We measure the selector inference path with model weights loaded and prepared inputs already placed on the assigned devices. Total rows report wall-clock criticalpath latency under the parallel schedule, not the sum of all row latencies. †Conditional on the corresponding expert appearing in the logic tree; when absent, the component is skipped entirely.

Component Description HW TFLOPs Latency (s) Preprocessing (cacheable, once per video) CLIP-dfn ViT-L/14 visual semantics; 600 frames @224 1 GPU 96.2 0.75 LAION CLAP† audio events; 300 win @2s 1 GPU 3.5 0.20 docTR OCR (db_resnet50+parseq)†

on-screen text; 300 frames (∼2.5k crops) 2 GPUs 46.3 1.72

faster-whisper lv3-turbo† (est.)

speech; full 600s audio 4 GPUs 11.5 2.74

Preprocessing total busiest GPU 8 GPUs 96.2 2.74 Per-query (uncacheable)

Qwen3-8B parser query→logic tree; 2348+67 tok 1 GPU 37.0 1.58 YOLOv8x-worldv2† open-vocab detection; 600 frames 6 GPUs 16.1 0.31 Per-query scoring CLIP text+cos., ASR match; 1 query 1 GPU <0.05 0.20 Composition + PASS normalize/smooth/logic/select CPU <0.05 <0.01 Per-query total 53.1 1.88

#### First-query total 149.4 4.63

Interpretation. The breakdown highlights where HiMu spends computation under a fixed RTX 6000 Pro measurement protocol. Expensive video-level evidence extraction is cacheable and parallelized across the 8 GPUs, so subsequent questions reuse CLIP, OCR, ASR, and CLAP evidence. The uncached path for a new question is primarily the text-only tree parser, followed by lightweight query-conditioned detection, matching, and composition.

### F Interpretability of Frame Selection

A key structural benefit of HiMu’s neuro-symbolic design is that every frame-selection decision is fully auditable. Because the satisfaction score T(t) is computed by a deterministic fuzzy-logic tree over named leaf scores, each frame

t carries an explicit attribution vector a(t) = (s(1)e

(t),...,s(eL)

(t)) ∈ [0,1]L, where each entry records how strongly a specific expert-predicate pair fired at that frame. This stands in sharp contrast to other controlled frame selectors: similarity-based methods (BOLT [19], AKS [25], MDP3 [24]) collapse the entire query into a single embedding and return one opaque similarity scalar per frame, leaving no record of which part of the query drove selection or which modality was activated; structured selectors (T∗ [32], VSLS [13]) expose detection logs but are restricted to flat object queries or a fixed vocabulary of relations, so their explanations cannot reflect nested temporal-logic structure. Beyond the controlled-selector class, multi-call agentic methods (VideoAgent [9], SeViLA [33]) produce readable reasoning traces but make their frame-selection decisions implicitly through opaque LLM outputs rather than an explicit scored ranking, so one cannot directly determine which predicate supported a selected frame or whether a failure arose from perception, temporal composition, or query decomposition. In HiMu, by contrast, any selected frame can be immediately explained from its leaf activations. For example, “this frame was chosen because the OVD leaf ‘man in suit’ scored 0.91 and the CLAP leaf ‘applause’ scored 0.83”; any rejected frame can be confirmed to have scored uniformly low across all predicates.

1

L

- Figure 6 illustrates this for a representative Video-MME question (“What does the black dog do in the water?”): each

column is one of the K=16 selected frames; each row is a named leaf, coloured by its normalised score s(ei)(t) ∈ [0,1]. The shared-context leaves under the root AND node (OVD: “black dog” and CLIP: “in water”) are uniformly dark across nearly all frames, confirming that the selected frames consistently depict the relevant scene. Within the OR over the four MCQ options, option-specific leaves exhibit clearly distinct activation patterns: for instance, CLIP: “submerging” fires strongly (1.00) on every frame, whereas CLIP: “searching” and OVD: “fish” are dark only in the earlier temporal cluster (about 133 to 173s) and fade to pale in the later cluster (about 459 to 586s), revealing which

[Figure 7]

Figure 6: Per-leaf activation heatmap for a sampled Video-MME question. Top-left: the MCQ question (“What does the black dog do in the water?”) with its four answer options. Centre: the logic tree with per-leaf scores for each

of the K=16 selected frames (columns, in temporal order). Cell colour encodes s(ei)(t) ∈ [0,1] (dark ≈ 1: predicate strongly satisfied; light/pale ≈ 0: not satisfied). Bottom: a filmstrip of a few of the 1fps sampled frames and enlarged thumbnails of five representatives from the selected frames. This leaf-level transparency is a structural consequence of HiMu’s neuro-symbolic design and is unavailable in similarity-based or other controlled frame selection methods.

moments support which answer candidate. The bottom of the figure shows the corresponding frame thumbnails, letting a user visually verify the heatmap activations against the actual video content.

This interpretability has direct practical value: when HiMu answers a question incorrectly, the heatmap pinpoints whether the failure stems from a specific expert (e.g., OCR missing on-screen text), the tree structure (e.g., a SEQ operator with the wrong temporal ordering), or the query decomposition (e.g., an overly generic predicate). Such finegrained, per-predicate transparency is, to our knowledge, unique among current frame-selection methods and turns HiMu into a diagnostic tool rather than merely a preprocessing step, enabling practitioners to iteratively refine expert configurations, tree templates, and smoothing parameters with targeted, evidence-based feedback.

