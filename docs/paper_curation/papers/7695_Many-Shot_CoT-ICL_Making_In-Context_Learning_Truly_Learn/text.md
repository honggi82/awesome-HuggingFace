## Many-Shot CoT-ICL: Making In-Context Learning Truly Learn

Tsz Ting Chung1 Lemao Liu2 Mo Yu3 Dit-Yan Yeung1

# arXiv:2605.13511v3[cs.CL]31May2026

### Abstract

While many-shot ICL achieves remarkable performance, prior studies of its scaling behavior have mainly focused on non-reasoning tasks. In this work, we study many-shot ICL on reasoning tasks, with a particular focus on many-shot chain-of-thought in-context learning (CoT-ICL). Analyzing across non-reasoning and reasoning tasks and across non-reasoning and reasoningoriented LLMs, we identify several distinctive properties of many-shot CoT-ICL. We further interpret these findings by viewing many-shot CoTICL as in-context test-time learning rather than scaled pattern matching, and suggest two principles: (i) demonstrations should be easy for the target model to understand, and (ii) they should be ordered to support a smooth conceptual progression. Guided by the principle, we propose Curvilinear Demonstration Selection (CDS), a simple ordering method that yields up to a 5.42 percentage-point gain on a math task with 64 demonstrations. Overall, our results reframe the long context window from a retrieval buffer into a structured curriculum for in-context test-time learning.

### 1. Introduction

In-context learning (ICL) enables large language models (LLMs) to perform tasks by conditioning on a sequence of input-output demonstrations without updating their parameters (Min et al., 2022; Von Oswald et al., 2023). Research has focused on improving ICL through strategies like selecting effective demonstrations (Sorensen et al., 2022; Liu et al., 2022; Wu et al., 2023). Recently, with the expansion of context windows, many-shot ICL has emerged, where dozens to hundreds of demonstrations can be provided,

1Hong Kong University of Science and Technology 2Fudan University 3Wechat AI, Tencent. Correspondence to: Lemao Liu <lemaoliu@fudan.edu.cn>, Dit-Yan Yeung <dyyeung@ust.hk>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Figure 1. Reframing of CoT-ICL as in-context test-time learning.

achieving performance competitive with fine-tuning (Agarwal et al., 2024; Bertsch et al., 2025).

In parallel, chain-of-thought (CoT) prompting has become a standard practice for complex reasoning (e.g., arithmetic and narrative reasoning), where models generate intermediate steps before an answer (Wei et al., 2022; Kojima et al., 2022). At the same time, test-time scaling studies how to improve model performance through additional computation during inference without parameter updates (Snell et al., 2025b; Lin et al., 2024). These threads naturally intersect: many-shot CoT-ICL is a basic form of test-time computation, where long sequences of reasoning demonstrations shape the model’s behavior at inference time.

However, there exists a critical gap. Current understanding of many-shot dynamics derives almost entirely from studies of non-reasoning tasks. For example, a consistent finding on many-shot ICL is that for non-reasoning tasks, the impact of demonstration order diminishes with scale (Bertsch et al., 2025; Baek et al., 2025). It remains unknown whether the established findings on many-shot ICL extend to many-shot CoT-ICL for reasoning. Does providing more reasoning demonstrations lead to reliable improvement, or does it introduce new instabilities? These questions are practically important for deploying reasoning-capable LLMs and theoretically fundamental: it probes whether many-shot CoTICL for reasoning is merely large-scale pattern matching or a form of genuine learning in in-context learning that follows pedagogical principles.

In this work, we demonstrate that the established rules of many-shot ICL break down for reasoning tasks. Through systematic experiments across model types (non-reasoning vs. reasoning-oriented) and tasks (non-reasoning vs. reason-

ing), our experiments uncover: (1) a setting-dependent scaling effect, where many-shot ICL scales on non-reasoning tasks but many-shot CoT-ICL on reasoning tasks is unstable for non-reasoning LLMs and improves mainly for reasoningoriented LLMs; (2) that similarity-based retrieval explains non-reasoning scaling but fails on reasoning because question similarity does not ensure procedural compatibility, pointing to in-context learning beyond surface matching; and (3) an order-scaling effect, where performance variance grows with the number of CoT demonstrations.

We explain these results by reframing many-shot CoT-ICL as in-context test-time learning rather than pattern matching. We propose that successful demonstrations must be both understandable to the model and smoothly sequenced. We examine this through two principles: (1) The Ease of Understanding: demonstrations should align with the model’s current knowledge (explaining why self-generated demonstrations work best for weaker models); and (2) The Smoothness of Knowledge Progression: the conceptual transition between consecutive demonstrations should be gradual (quantifiable via the curvature of their embedding trajectory) as illustrated in Figure 1. Building on these principles, we introduce Curvilinear Demonstration Selection (CDS), a practical method that orders demonstrations to minimize total conceptual curvature. This approach yields an average of 3.81% performance gains across both math and narrative reasoning tasks.

Our contributions are threefold: (1) We characterize the scaling dynamics of many-shot CoT-ICL; (2) We reframe effective many-shot CoT through the ease of understanding and smoothness of information flow, bridging ICL with insights from test-time learning; (3) We introduce and validate a practical, principle-driven method for demonstration ordering that advances many-shot reasoning.

### 2. Related Works

Many-shot ICL The extension of LLM context windows (Peng et al., 2024; Han et al., 2024) has enabled many-shot ICL, where models process significantly more demonstrations (Agarwal et al., 2024; Bertsch et al., 2025; Chung

- et al., 2024). Wan et al. (2025) show that many-shot gains are often driven by a subset of influential examples, and that similarity-based retrieval is ineffective compared with validation-guided optimization and generation. We instead study when many-shot CoT-ICL is effective: when scaling helps across task and model types, why semantic similarity fails for reasoning, and how demonstration order shapes in-context procedural learning. Findings in broader manyshot ICL reveal that with sufficient demonstrations, model sensitivity to ordering diminishes for standard classification tasks (Baek et al., 2025; Bertsch et al., 2025), suggesting a form of robustness with scaling. This led to a narrative

that in many-shot settings, careful demonstration engineering may be unnecessary. However, these studies focused overwhelmingly on non-reasoning tasks (e.g., classification, simple QA) (Baek et al., 2025; Bertsch et al., 2025), neglecting performance on reasoning tasks (Hendrycks et al., 2021; Chung et al., 2025; Xu et al., 2024; Yu et al., 2025a). Concurrent work on test-time scaling leverages extended computation for self-improvement without parameter updates (Snell et al., 2025b; Li et al., 2025b), suggesting that effective in-context learning can be viewed as a form of real-time optimization. Our work connects many-shot CoTICL to test-time learning, guided by two key principles that explain how learning occurs inside the context.

Chain-of-Thought CoT prompting (Wei et al., 2022) decomposes reasoning into intermediate steps, substantially improving LLM performance on complex tasks. Subsequent studies like Tree-of-Thoughts (Yao et al., 2023) and Program-of-Thoughts (Chen et al., 2023) explore structured reasoning paths, while methods like rStar-Math (Guan et al., 2025) employ search algorithms for trajectory optimization. These approaches primarily focus on enhancing the reasoning process for a single query. In the ICL setting, Dr.ICL (Luo et al., 2023) demonstrates that retrieving relevant CoT demonstrations boosts few-shot performance, and Auto-CoT (Zhang et al., 2023) proposes an automatic few-shot CoT prompting method that clusters questions to sample diverse representatives and generate reasoning chains as demonstrations. However, a critical gap remains: existing CoT-ICL work largely operates in few-shot settings. The fundamental question of how CoT demonstrations scale with context length and whether the principles of effective demonstration design change from few-shot to many-shot is largely unexplored. Our work positions many-shot CoT not merely as ”more examples”, but as a potential in-context curriculum that requires principled sequencing.

Demonstration Selection Demonstration selection has long been studied for effective few-shot ICL. The dominant paradigm is similarity-based retrieval, where demonstrations semantically closest to the test query are selected (Liu et al., 2022; Wu et al., 2023; Kapuriya et al., 2025). This approach implicitly frames ICL as a form of pattern matching (Olsson et al., 2022; Crosbie & Shutova, 2025; Yu et al., 2025b). Interestingly, this paradigm finds a direct analogy in Retrieval-Augmented Generation (RAG), where relevant context chunks are retrieved via embedding similarity (Lewis et al., 2020). Our work challenges whether this conclusion extends to reasoning tasks. We hypothesize that for CoT-ICL, effective demonstration selection is less about retrieving semantically similar examples and more about constructing a smooth learning sequence that facilitates conceptual understanding, acting as a shift from ”retrieval for matching” to ”retrieval for learning”.

### 3. Settings

We establish an experimental framework for studying manyshot In-Context Learning (ICL), with and without Chain-ofThought (CoT), under long-context constraints. Our design spans three dimensions: task type (non-reasoning vs. reasoning), model type (standard instruction-tuned vs. explicitly “reasoning” models), and ICL configuration (prompt format and number of demonstrations).

#### 3.1. Tasks Studied

Prior many-shot work has largely emphasized non-reasoning classification benchmarks (Li et al., 2025a; Bertsch et al., 2025). We extend evaluation to include both classificationstyle tasks and multi-step reasoning tasks, while using a unified open-ended generation evaluation for all datasets.

Evaluation protocol. For each test instance, the model generates a free-form text completion. We map the completion to a predicted answer using task-specific extraction and normalization, and score it by exact match against the reference. Prompt templates for evaluation are provided in Appendix E. For numerical datasets (e.g., GSM8K/MATH), we extract the final numeric value or mathematical expression from the completion and compare it to the ground truth under the same exact-match criterion.

Non-reasoning tasks. These tasks require little intermediate reasoning and primarily test semantic understanding and label mapping. We include benchmarks with different labelspace sizes: SuperGLUE (Wang et al., 2019) (small label space), NLU (nlu, 2021), TREC (Hovy et al., 2001), and BANKING77 (Casanueva et al., 2020) (large label space).

Reasoning tasks. These tasks require deduction and/or mathematical derivation. We focus on mathematical reasoning with GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021), and include DetectiveQA (Xu et al., 2024) for narrative reasoning over long contexts. For tasks that provide gold rationales, we use the dataset-provided reasoning chains Ci as the CoT component in demonstrations (Section 3.3).

#### 3.2. LLMs Studied

We evaluate a range of LLMs and group them by whether they explicitly contain extended reasoning at inference time.

(Qwen2.5-14B-Instruct).

Reasoning-oriented LLMs. These models expose an explicit reasoning segment (e.g., a <think> token). We evaluate Qwen 3 (8B) (Qwen3-8B) and Qwen 3 (14B) (Qwen3-14B) (Yang et al., 2025), QwQ (32B) (QwQ-32B) (Qwen et al., 2024), and DeepSeek-R1 (685B) (DeepSeek-R1) (DeepSeek-AI, 2025). For reasoningoriented models, we enable the model’s reasoning mode during inference to allow the generation of intermediate reasoning tokens.

Long-context configuration. To support many-shot prompts (up to 131K tokens for Qwen-family models), we apply the official RoPE scaling configurations provided by each model release. All other decoding and system-prompt settings follow the model providers’ recommended defaults unless stated otherwise.

#### 3.3. ICL Configuration

We study scaling from few-shot to many-shot under two prompting paradigms.

Traditional ICL. Prompts consist of k input–output pairs (xi,yi) followed by a query x′. The model produces an output y′ conditioned on the ordered demonstration sequence:

y′ = LLM(x′ | [(x1,y1),...,(xn,yn)]). (1)

CoT-ICL. Prompts consist of k triples (xi,Ci,yi), where Ci is a reasoning chain. Given a query x′, the model generates both an intermediate chain C′ and a final answer:

(C′,y′) = LLM(x′ | [(x1,C1,y1),...,(xn,Cn,yn)]).

(2)

Context scaling. CoT demonstrations are substantially longer than standard ICL examples (e.g., in our geometry setting, a single CoT demonstration can be ∼30× longer than a BANKING77 example). As a result, while hundreds to thousands of demonstrations may fit for traditional ICL, CoT-ICL is typically limited to at most a few hundred demonstrations by context length. We therefore focus our scaling analysis on n ≤ 128, which captures the most informative trade-offs between model type, task type, and demonstration count in our long-context regime.

Non-reasoning LLMs. These models are primarily tuned to produce direct answers given instructions, without an explicit “thinking” token. We evaluate LLaMA 3.1 (Llama-3.1-8B-Instruct), LLaMA 3.3 (Llama-3.3-70B-Instruct) (MetaAI, 2024), Qwen 2.5 (7B) (Qwen2.5-7B-Instruct), and Qwen 2.5 (14B)

### 4. Properties of CoT-ICL 4.1. Scaling with Reasoning Tasks

Prior work reports that many-shot ICL yields reliable improvements on non-reasoning tasks (Bertsch et al., 2025; Baek et al., 2025). We replicate this behavior, but find

1

NormalizedAccuracy

WSC

2

GSM8K

precalculus

COPA TREC BANKING77

c&p

number_theory

prealgebra

algebra

geometry

NLU

DetectiveQA

20 50 80

20 50 80

Number of examples in-context

###### (a) Llama-3.1-8B-Instruct

| | | | |
|---|---|---|---|
| | |WSC| |
| | |COPA TREC BANKI<br><br>NLU|NG77|

1

NormalizedAccuracy

2

GSM8K

precalculus

c&p

number_theory

prealgebra

algebra

geometry

DetectiveQA

20 50 80

20 50 80

Number of examples in-context

###### (b) Qwen2.5-7B-Instruct

1

NormalizedAccuracy

WSC

2

GSM8K

precalculus

COPA TREC BANKING77

c&p

number_theory

prealgebra

algebra

geometry

NLU

DetectiveQA

20 50 80

20 50 80

Number of examples in-context

(c) Qwen2.5-14B-Instruct

Figure 2. Scaling disparity between task types. Performance (normalized accuracy) of non-reasoning LLMs on classification tasks (warm colors) versus reasoning tasks (cool colors). The xaxis represents normalized accuracy (i.e., x−x¯

σx for accuracy x), while the y-axis indicates the number of in-context demonstrations.

it does not extend to reasoning tasks when demonstrations include CoT rationales. Figure 2 shows a clear contrast: nonreasoning tasks improve steadily as the number of demonstrations increases, whereas reasoning performance is unstable and often degrades for non-reasoning LLMs.

This failure is not explained by insufficient parameter scale. As shown in Figure 3 (left), even Llama 3.3 70B can incur negative gains from adding more CoT demonstrations. Together, these results suggest a qualitative difference between scaling traditional ICL and CoT-ICL.

#### 4.2. Scaling with Reasoning LLMs

The scaling behavior changes markedly for models with explicit reasoning capabilities. Figure 3 (right) shows that QwQ (32B) and R1 (685B) improve consistently as more CoT demonstrations are added. This trend also holds for

| | |number_<br><br>geometry|theory|
|---|---|---|---|
| | | | |
| | | | |

|numbe geome numbe<br><br>|r_theoryQwQ tryQwQ r_theoryR1| | |
|---|---|---|---|
|geome|tryR1| | |
| | | | |

1

NormalizedAccuracy

1

20 50 80

20 50 80

(a) Llama 3.3

(b) QwQ & R1 Number of examples in-context

Figure 3. Scaling disparity between model types on math reasoning tasks. Left: Llama 3.3 (non-reasoning LLM) shows negative gains. Right: QwQ (32B) and R1 (685 B) (reasoning LLM) shows clear positive scaling.

Model geometry number theory DetectiveQA

Qwen3-14B (en) 73.07 91.30 72.73 Qwen3-14B (dis) 65.76 88.15 72.73 Qwen3-8B (en) 67.01 84.63 69.48 Qwen3-8B (dis) 62.63 79.81 66.88

Table 1. Performance at n = 128 with reasoning-oriented models’ thinking mode (en)abled versus (dis)abled.

smaller reasoning-optimized models: across the Qwen3 family (Figure 4), performance increases near-monotonically with additional demonstrations.

The divergence between model classes indicates that benefiting from long CoT contexts is not a generic consequence of having more examples in context. Instead, positive scaling appears to require model mechanisms that can use demonstrations as intermediate reasoning signal (e.g., via thinking tokens and/or reasoning-oriented training), rather than relying primarily on shallow pattern matching. To directly test this interpretation, we evaluate the same n = 128 manyshot CoT contexts with thinking enabled versus disabled. As shown in Table 1, suppressing the generation of intermediate reasoning hurts performance on geometry and number theory for both Qwen3 models, and also hurts DetectiveQA for Qwen3-8B. Furthermore, when thinking is enabled on geometry, increasing n from 16 to 128 improves Qwen3-14B accuracy from 66.18% to 73.07%, while reducing the average number of generated tokens inside the <think> segment by 24.02%. This suggests that larger CoT contexts help the model internalize task procedures, reducing the need for verbose query-time deliberation.

#### 4.3. Rethinking ICL with similarity

Sections 4.1–4.2 reveal a consistent split: many-shot ICL scales reliably on non-reasoning tasks, while many-shot CoT-ICL for reasoning is unstable for non-reasoning LLMs and improves mainly for reasoning-optimized LLMs.

For positive scaling effect, a common explanation for why many-shot ICL works is the retrieval hypothesis: additional demonstrations help because the model can locate and reuse

NormalizedAccuracy

0

1

geometry

geometry

number_theory

number_theory

c&p

c&p

GSM8K

GSM8K

20 50 80

20 50 80

(a) Qwen 3 (8B)

(b) Qwen 3 (14B)

Number of examples in-context

Figure 4. Positive scaling of reasoning LLMs. The Qwen3 family (reasoning LLMs) demonstrates consistent performance improvements with more demonstrations on math reasoning tasks. Left: Qwen3 (8B) Right: Qwen3 (14B)

examples that are semantically similar to the query (Liu et al., 2022; Wu et al., 2023). If many-shot CoT-ICL for reasoning were driven by the same mechanism, then (i) retrieving question-similar demonstrations should help more as k grows, and (ii) the most-similar set should consistently outperform dissimilar or uncurated sets.

For each test query, we embed all candidate training questions (question-only) with Qwen3-Embedding-4B (Zhang

- et al., 2025) and rank candidates by cosine similarity. We then build two k-shot demonstration sets per query: (i) mostsimilar (top-k) and (ii) most-dissimilar (bottom-k), keeping the original CoT+answer paired with each selected question. We evaluate five base LLMs (Llama 3.1, Qwen 2.5 7B/14B, Qwen3 8B/14B) and report averages; details are in Appendix A.

Similarity retrieval succeeds for non-reasoning tasks, but fails for reasoning tasks. Figure 5 supports the retrieval hypothesis on a non-reasoning task BANKING77. The mostsimilar sets consistently outperform the most-dissimilar sets. However, the same heuristic breaks on reasoning tasks. Across geometry, number theory, and DetectiveQA, the most-similar sets are consistently worse than either the mostdissimilar sets or the original (unretrieved) sets. This conclusion holds when evaluating reasoning and non-reasoning LLMs separately (Appendix A.5).

Similarity optimizes matching, not learning. These results align with the paper’s central message: many-shot CoTICL for reasoning is not well explained as scaled-up pattern matching. For non-reasoning tasks, question-level similarity is often a reliable proxy for label similarity, so retrieving similar demonstrations improves performance. For reasoning tasks, in contrast, question-level similarity is a weak proxy for procedural compatibility. Two problems can look semantically similar while requiring different solution strategies, and their associated CoT-ICL may induce conflicting intermediate steps. We provide qualitative examples and additional analysis in Appendix A.4.

###### BANKING77

DetectiveQA

70

71

Sim>Ori/Dis

| |
|---|

Original

45

69

Most Similar

Sim>Ori/Dis Ori/Dis>Sim Original

Most Dissimilar

| |
|---|

| |
|---|

20

67

Most Similar

Most Dissimilar

Accuracy

20 60 100

20 60 100

geometry

number_theory

Ori/Dis>Sim

Most Similar

| |
|---|

Original

Most Dissimilar

52

65

48

60

55

44

Ori/Dis>Sim

Most Similar

| |
|---|

Original

Most Dissimilar

20 60 100

20 60 100

Number of examples in-context

Figure 5. Performance with original(ori), similarity(sim) and dissimilar(dis) sets averaged across five LLMs. The area between the two sets is filled with colors, indicating the relative performance.

This provides a mechanism-level explanation for the negative scaling observed in Section 4.1. Solving reasoning tasks depends on extracting and reusing procedures, not merely matching surface patterns. With purely surface matching, LLMs are likely to be misled by a set of “similar” but procedurally mismatched CoTs, leading to negative gains with similar retrieval.

The failure of similarity-based retrieval with reasoning LLMs also suggests that the mechanism behind positive scaling differs across settings. In particular, the rationale for why scaling works for reasoning-oriented LLMs on reasoning tasks (Section 4.2) is not the same as why scaling works for non-reasoning LLMs on non-reasoning tasks (Section 4.1). From a learning perspective, a plausible explanation is that reasoning-oriented models can better interpret the provided CoT-ICL and extract higher-level procedural structure in the thinking content beyond surface pattern matching, allowing the benefit from additional demonstrations.

#### 4.4. Ordering Stability of CoT-ICL

If CoT demonstrations act as a learning signal rather than a static reference, their order should matter, since order changes the trajectory of intermediate states induced by the context. This prediction contrasts with findings on nonreasoning tasks, where order sensitivity decreases as the number of demonstrations grows (Bertsch et al., 2025; Baek et al., 2025).

We quantify order sensitivity by sampling five random permutations of the same demonstration set and measuring the standard deviation of accuracy. For non-reasoning tasks, we reproduce the low-variance behavior (Figure 6, left). In con-

| |BANKING77<br><br>NLUQwen2.5|Qwen2.5|
|---|---|---|
| | | |
| | | |

|ntQw<br><br>geo|en3<br><br>metryQwen3<br><br>geometry<br><br>ntQwen2.5|Qwen2.5|
|---|---|---|
| | | |
| | | |

Model Setting n = 16 n = 128 Qwen3-8B Valid 57.62 67.01

- 1

- 2

StandardDeviation

Corrupted 57.62 65.76 Qwen3-14B Valid 66.17 73.07 Corrupted 67.01 70.56

Table 2. Procedural-corruption ablation on geometry. The larger drop at n = 128 indicates that models use demonstration-specific procedures, not only answer labels or long-context activation.

30 80

30 80

Number of examples in-context

(xi,Ci,yi), against a procedural-corruption condition that preserves every question and final answer but replaces all rationales with the same static chain from the first demonstration, (xi,C0,yi). This controls for format, context length, and the x → y mapping, isolating whether the aligned procedure Ci matters. Table 2 shows that at n = 16 the two settings are nearly indistinguishable, but at n = 128 corrupted procedures cause clear drops for both Qwen3-8B and Qwen3-14B. Thus, when enough informative rationales are provided, reasoning models appear to use the procedural steps in the demonstrations rather than merely memorizing answer labels or passively activating long-context priors.

Figure 6. Standard deviation of performance across five random demonstration orders on classification tasks (warm colors) versus reasoning tasks (cool colors), where nt corresponds to number theory. Results shown for Qwen2.5 (14B) (non-reasoning) and Qwen3 (14B) (reasoning).

trast, for reasoning tasks we observe the opposite trend: variance increases with more demonstrations (Figure 6, right). This holds for both non-reasoning and reasoning LLMs.

Overall, many-shot CoT-ICL exhibits strong and growing path dependence: performance depends not only on which demonstrations are provided, but also on how they are sequenced. This instability is consistent with CoT-ICL behaving as an in-context learning process whose effectiveness depends on the induced reasoning trajectory, motivating our in-context test-time learning perspective in the next section. We further validate these conclusions by computing mean and standard deviation across five random demonstrationordering seeds on an independently sampled ICL subset. The same qualitative trends persist for reasoning-oriented models, non-reasoning models, and cross-model CoT transfer, with full results in Appendix B.

This framing yields two practical principles for demonstration design. First, demonstrations must be understandable to the model (otherwise they cannot be used as supervision at test time). Second, demonstrations should be arranged to yield a smooth information flow across the prompt (otherwise the induced procedure becomes unstable), providing a direct explanation for the order sensitivity observed in Section 4.4. These principles also connect to recent evidence that scaling test-time computation improves performance by refining internal solution procedures (Snell et al., 2025a).

### 5. Rethinking ICL: From Pattern Matching to In-Context Test-Time Learning

Sections 4.4 and 4.3 suggest that many-shot CoT-ICL does not behave like a simple nearest-neighbor or patternmatching mechanism: increasing the number of demonstrations amplifies order sensitivity, and similarity-based selection is not reliably helpful on reasoning tasks. We therefore adopt a different lens: in-context test-time learning, where the prompt serves as training data and the forward pass resembles a gradient-free form of adaptation. Under this view, demonstrations do not only provide answers to copy; they can shape an internal procedure for how to solve the task.

Before deriving design principles from this view, we first provide direct evidence that models indeed absorb procedures from demonstrations rather than merely exploiting input–output associations.

Direct evidence for procedure absorption. We further test whether the model uses demonstration-specific procedures rather than only the input–output mapping. On geometry, we compare standard many-shot CoT demonstrations,

#### 5.1. Principle 1: Ease of understanding

If ICL operates as in-context test-time learning, then demonstrations must fall within the model’s current ability to parse and internalize. In educational psychology, effective instruction targets a learner’s “zone of proximal development” (Benson, 2020), the range between what they can solve unaided and what they can solve with appropriate guidance. By analogy, we posit a zone of understandable reasoning: demonstrations are most useful when the model can follow their reasoning steps and internalize the implied procedure, rather than when they are merely “higher quality” but stylistically or procedurally misaligned with the model.

5.1.1. SETTINGS

We test whether demonstration effectiveness depends more on answer correctness or on alignment with the model’s own generation distribution. For each training instance, we sample CoT demonstrations from each LLM with temperature 1.0 (10 samples per instance) and construct:

- • Correct (cr): generated CoT with correct answers.
- • Wrong (wr): generated CoT with incorrect answers.

• First (first): the first sampled CoT regardless of correctness.

We compare these sets against the dataset-provided groundtruth CoT (origin).

We use cr/wr for the Qwen 2.5 family, where incorrect generations are sufficiently frequent to construct a sizable wr set, except for the GSM8K task. For Qwen 3 family, which achieves higher accuracy and therefore rarely produces wrong answer under our sampling budget, constructing wr is difficult. We instead use the first set to evaluate the effect of self-generated (and distribution-aligned) demonstrations without conditioning on correctness. We additionally evaluate cross-model transfer by using CoTs generated by stronger models (Qwen2.5/3 14B) as demonstrations for weaker models.

- 5.1.2. RESULTS

- (1) Understanding improves with distributional alignment: self-generated CoTs perform best. Figures 7 and 8 show that prompts constructed from self-generated demonstrations (cr/wr/first) consistently outperform dataset-provided CoTs (origin) and cross-model demonstrations written by stronger LLMs when used to prompt weaker ones, even when some self-generated demonstrations have incorrect final answers. Because self-generated CoTs are drawn from the target model’s own generation distribution, they are more likely to be understandable to that model (i.e., easier to condition on and reuse as procedural supervision), consistent with Principle 1.
- (2) Understanding improves with scale: the selfgenerated advantage diminishes for stronger models. If demonstration effectiveness is limited by understanding, then as model capability increases it should become easier to extract useful procedures from less-aligned supervision (i.e., origin), reducing the relative benefit of selfgenerated CoTs. Consistent with this intuition, the gain of self-generated demonstrations over origin shrinks with model ability (e.g., Qwen3-14B exhibits a smaller gain than Qwen3-8B in Figure 8). This suggests that stronger models can understand and exploit ground-truth rationales more reliably.
- (3) Understanding improves with reasoning-oriented priors: reasoning models are less brittle to supervision mismatch. Beyond scale, Section 4.2 shows that reasoning LLMs outperform non-reasoning LLMs at comparable parameter sizes under the provision of dataset-provided CoTICL. A plausible explanation is that the reasoning within the thinking tokens acts as an additional prior that guides how demonstrations are interpreted and how procedural patterns are extracted from them. Under this view, bet-

GSM8K

###### GSM8K

| | | | |
|---|---|---|---|
| | | | |
| | |o c<br><br>|rigin r|

70

91

89

30

origin

cr

crqwen14b

number_theory

number_theory

origin

wr

cr

40

82

wr>origin

Accuracy

28

78

crqwen14b wr>origin

origin

wr

| |
|---|

cr

geometry

geometry

30

61

| |
|---|

origin

13

56

crqwen14b wr>origin

origin

wr

cr wr>origin

wr

| |
|---|

cr

| |
|---|

20 50 80

20 50 80

(a) Llama 3.1

(b) Qwen 2.5 (14B)

Number of examples in-context

Figure 7. Performance of two sets of self-generated in-context CoT, including the set filtered with only correct answer(cr) and the set filtered with only wrong answer(wr). crqwen14 is prompting the LLaMA model with the in-context CoT generated by Qwen 2.5 (14B). Left: Llama 3.1 Right: Qwen 2.5 (14B)

ter performance reflects a stronger ability to leverage the supervision signal in provided examples, even when the dataset-provided demonstrations are not perfectly aligned with the target model.

#### 5.2. Principle 2: Smoothness of information flow

Effective learning requires not just comprehensible individual examples, but a coherent progression between them. We hypothesize that smooth transitions between demonstrations facilitate the model’s construction of a coherent reasoning schema, while abrupt conceptual jumps disrupt this process.

5.2.1. SETTINGS: QUANTIFYING TRANSITION SMOOTHNESS

We measure smoothness by viewing an ordered list of demonstrations as a trajectory in embedding space. We represent each demonstration di as (question + CoT + final answer) and embed it using Qwen3-Embedding-4B to obtain ei ∈ Rd. Unlike Section 4.3 (question-only similarity), we embed the full demonstration because ordering effects should depend on procedural content. This representation is designed to capture not only topical similarity but also the logical structures and operations expressed in the CoT rationale. We compute curvature directly in the original embedding space, with details in Appendix C.

DetectiveQA

DetectiveQA

| |
|---|

| |
|---|

73

69

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

origin

65

70

first

origin

firstqwen3(14b) wr>origin

first wr>origin

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Accuracy

number_theory

number_theory

|| |
|---|
<br><br>|| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|---|---|---|
|| |
|---|
<br><br>or|igin| | |
|fi fi w<br><br>| |
|---|
|rst rstqwen3(14b) r>origin| | |

| | || |
|---|
| |
|---|---|---|---|
|| |
|---|
<br><br>| |
|---|
|| |
|---|
| | |
| | || |
|---|
|origin first wr>origin|

75

85

65

80

20 60 100

20 60 100

(a) Qwen 3 (8B)

(b) Qwen 3 (14B)

Number of examples in-context

Figure 8. Performance of the first set of self-generated in-context CoT. firstqwen3(14b) is prompting the Qwen 3 (8B) model with the in-context CoT generated by Qwen 3 (14B). Left: Qwen 3 (8B) Right: Qwen 3 (14B)

Given an ordering O = [d1,...,dn], we define local curvature at position i as the turning angle between consecutive displacement vectors:

(ei − ei−1) · (ei+1 − ei) ∥ei − ei−1∥ ∥ei+1 − ei∥

(3)

θi = arccos

Total curvature is Θ(O) = ni=2−1 θi, where smaller values indicate smoother transitions. Implementation details (in-

cluding the exact concatenation template and robustness checks) are in Appendix C.

- 5.2.2. RESULTS

Across three math reasoning tasks, ordering curvature is strongly negatively correlated with accuracy: overall r = −0.547, with task-wise correlations of −0.545 (geometry), −0.468 (number theory), and −0.628 (counting and probability). Thus, smoother orderings tend to yield better performance.

This also provides a concrete explanation for the increasing order variance observed in Section 4.4. As the number of demonstrations grows, random permutations are more likely to contain sharp “conceptual jumps” (high curvature), amplifying variability across orders. Controlling the ordering to reduce curvature yields a more stable learning trajectory, motivating our ordering method in Section 6.

Causal smoothness ablation. To separate smooth transitions from local clustering, we construct two orderings from the same demonstrations using bge-m3 embeddings. Both orderings constrain Euclidean proximity, but the highcurvature baseline inverts the curvature objective, forcing abrupt conceptual turns while preserving local neighbor-

hoods. Across number theory and geometry, CDS consistently outperforms this high-curvature ordering in Table 4, supporting transition smoothness as a causal factor rather than a by-product of grouping similar examples.

- 5.2.3. DISCUSSION: PEDAGOGICAL ANALOGY

This principle mirrors effective textbook design: concepts are introduced progressively, with each chapter building smoothly upon the previous. Abrupt topic changes or missing prerequisites hinder learning. Similarly, in many-shot CoT-ICL, demonstrations must be ordered to create a ”conceptual curriculum” that guides the model from basic to advanced reasoning steps.

- 6. Curvilinear Demonstration Selection

Motivated by the curvature–performance correlation in Section 5.2, we introduce Curvilinear Demonstration Selection (CDS), a practical method for constructing an ordering of many-shot CoT demonstrations. CDS aims to produce a smooth trajectory in embedding space, avoiding abrupt transitions between successive demonstrations.

#### 6.1. Experiment Settings

We evaluate CDS on three reasoning tasks spanning diverse domains, including geometry, number theory, and DetectiveQA. Our primary experiments use reasoning LLMs from the Qwen3 family (8B and 14B) across multiple demonstration budgets, with the motivation and experimental details provided in Appendix D.1. We additionally include gpt-5.2 in the robustness study to test whether the ordering effect transfers to a closed-source reasoning model.

Objective. Given a set of n demonstrations with original embeddings {ei}ni=1, CDS seeks a permutation O = [dπ(1),...,dπ(n)] that minimizes the total curvature

n−1

Θ(O) =

arccos

t=2

vt · vt+1 ∥vt∥ ∥vt+1∥

(4)

vt = eπ(t) − eπ(t−1). (5)

TSP-based approximation. Directly optimizing Eq. (4) is combinatorial for large n: exact minimization requires evaluating n! permutations, which is infeasible for the longest prompts studied in this paper (n ≤ 128). Moreover, optimizing angles alone can produce trajectories that are geometrically straight but make very large jumps across the embedding space. We therefore use a practical TSP-based heuristic with a pairwise transition cost that balances spatial proximity and a local curvature proxy. For each ordered pair

Task Model Method Number of demonstrations 16 32 64 128

Task Model Method Number of demonstrations 16 32 64 128

number theory gpt-5.2 CDS 88.15 91.48 88.70 91.30

number theory gpt-5.2 origin 89.63 91.11 88.56 91.48 CDS 89.26 92.59 92.04 91.85 CDSbge 88.15 91.48 88.70 91.30

high curv 85.74 89.63 86.48 88.15 Qwen3-14B CDS 85.37 87.78 87.22 91.30

high curv 79.26 84.44 84.26 90.37 geometry gpt-5.2 CDS 80.37 78.29 76.83 75.11

Qwen3-14B origin 86.67 87.96 86.30 90.93 CDS 85.56 87.85 87.78 90.74 CDSbge 85.37 87.78 87.22 91.30

high curv 72.65 73.90 76.33 74.53 Qwen3-14B CDS 66.60 67.85 70.36 74.32

geometry gpt-5.2 origin 75.99 74.74 75.37 75.78 CDS 81.21 78.08 80.79 75.99 CDSbge 80.37 78.29 76.83 75.11

high curv 66.38 66.81 66.60 71.80

Table 4. Controlled smoothness ablation with bge-m3 embeddings. The same demonstrations are used for both orderings; only the transition curvature objective is inverted.

Qwen3-14B origin 66.18 65.76 65.14 73.07 CDS 65.55 68.27 68.89 73.90 CDSbge 66.60 67.85 70.36 74.32

DetectiveQA gpt-5.2 origin 80.52 82.47 83.77 85.71 CDS 80.52 83.12 85.06 88.31 CDSbge 81.17 85.71 86.36 88.31

High-curvature control. To isolate curvature from local clustering, we compare CDS with a high-curvature ordering constructed from the same demonstrations. Both use Euclidean proximity, but the high-curvature variant inverts the curvature objective:

Qwen3-14B origin 75.97 74.03 70.78 72.73 CDS 76.62 75.32 73.38 75.32 CDSbge 75.32 77.29 75.32 76.63

Table 3. CDS robustness across tasks, embedding models, and target LLMs. CDS uses the original embedding model, while CDSbge replaces it with bge-m3.

γab − γij

Dhigh curv(i,j) = δij + max

a,b

(i,j), let

δij = ∥ej − ei∥

maxa,b ∥eb − ea∥ + ϵ be the normalized Euclidean distance, where ϵ is a small constant for numerical stability. Let

ei + ej 2

eℓ −

k(i,j) = arg min

ℓ/∈{i,j}

be the demonstration nearest to the midpoint of ei and ej, excluding the endpoints. We define the curvature proxy

γij =

1 π

arccos

(ej − ei)⊤(ek(i,j) − ej) ∥ej − ei∥ ∥ek(i,j) − ej∥

,

with the cosine clipped to [−1,1] in implementation. The CDS edge cost is

##### DCDS(i,j) = δij + γij.

The Euclidean component keeps adjacent demonstrations in related local neighborhoods, while the curvature proxy discourages sharp conceptual turns. We build a complete graph under this cost and compute a short tour using nearestneighbor initialization followed by 2-opt local search (Croes, 1958). We then break the tour at its longest edge to obtain a path, and choose the best path across up to ten starting points according to the final smoothness score 1/(1 + θ¯). Our theoretical claim only requires a sufficiently smooth pedagogical progression, not the global minimum of Eq. (4); empirically, this approximation is effective and remains inexpensive, taking under one minute on a standard CPU for n ≤ 128.

Thus, it still groups semantically related demonstrations while forcing abrupt turns between consecutive examples.

Result. We evaluate CDS on three reasoning tasks: geometry proof generation, number theory problem solving, and DetectiveQA logical reasoning. We further test robustness by replacing the original embedding model with bge-m3 and by evaluating an additional closed-source model, gpt5.2. Table 3 shows that the gains persist across embedding models and target LLMs, especially on geometry and DetectiveQA; number theory shows smaller margins because baseline accuracies are already high, leaving less room for ordering to improve performance. The performance gains also depend on the curvature of the original ordering. Our ablation against a high-curvature baseline strengthens our curvature claim.

### 7. Conclusion

Many-shot ICL has been largely understood through nonreasoning tasks, where scaling demonstrations is stable and ordering effects often fade. Our results show that these regularities do not transfer to many-shot CoT-ICL for reasoning: scaling depends on the model and task, similarity retrieval can fail due to procedural mismatch, and order variance grows with more CoT demonstrations. We therefore frame many-shot CoT-ICL as in-context test-time learning, where useful demonstrations should be understandable to the target model and arranged in a smooth knowledge progression. Guided by this view, CDS orders demonstrations by minimizing conceptual curvature and yields consistent gains across math and narrative reasoning, suggesting that reasoning trajectories can serve as reusable procedural guidance for future retrieval and prompting.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Benchmarking natural language understanding services for building conversational agents. In Increasing Naturalness and Flexibility in Spoken Dialogue Interaction, Lecture Notes in Electrical Engineering, pp. 165–183. Springer, 2021. ISBN 9789811593222. doi: 10.1007/ 978-981-15-9323-9 15. URL https://iwsds2019.

unikore.it/.

Agarwal, R., Singh, A., Zhang, L., Bohnet, B., Rosias, L., Chan, S., Zhang, B., Anand, A., Abbas, Z., Nova, A., et al. Many-shot in-context learning. volume 37, pp. 76930–76966, 2024.

Baek, J., Lee, S. J., Gupta, P., Oh, G., Dalmia, S., and Kolhar, P. Revisiting in-context learning with long context language models. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 26950–26966, 2025. URL https://aclanthology.org/2025.

findings-acl.1382/. Benson, J. B. Encyclopedia of infant and early childhood development. Elsevier, 2020.

Bertsch, A., Ivgi, M., Xiao, E., Alon, U., Berant, J., Gormley, M. R., and Neubig, G. In-context learning with long-context models: An in-depth exploration. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 12119–12149, 2025.

Casanueva, I., Temˇcinas, T., Gerz, D., Henderson, M., and Vuli´c, I. Efficient intent detection with dual sentence encoders. In Wen, T.-H., Celikyilmaz, A., Yu, Z., Papangelis, A., Eric, M., Kumar, A., Casanueva, I., and Shah, R. (eds.), Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, pp. 38–45, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.nlp4convai-1.5. URL https:// aclanthology.org/2020.nlp4convai-1.5.

Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023. ISSN 28358856. URL https://openreview.net/forum? id=YfZ4ZPt8zd.

Chung, T. T., Cui, L., Liu, L., Huang, X., Shi, S., and Yeung, D.-Y. Selection-p: Self-supervised task-agnostic prompt compression for faithfulness and transferability. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 11057–11070, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp. 646. URL https://aclanthology.org/2024.

findings-emnlp.646/.

Chung, T. T., Liu, L., Yu, M., and Yeung, D.-Y. DivLogicEval: A framework for benchmarking logical reasoning evaluation in large language models. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 901–915, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025. findings-emnlp.47. URL https://aclanthology.

org/2025.findings-emnlp.47/.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. ArXiv preprint, abs/2110.14168, 2021. URL https://arxiv.org/ abs/2110.14168.

Croes, G. A. A method for solving traveling-salesman problems. Operations research, 6(6):791–812, 1958.

Crosbie, J. and Shutova, E. Induction heads as an essential mechanism for pattern matching in in-context learning. In Chiruzzo, L., Ritter, A., and Wang, L. (eds.), Findings of the Association for Computational Linguistics: NAACL 2025, pp. 5049–5111, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-195-7. doi: 10.18653/v1/2025. findings-naacl.283. URL https://aclanthology.

org/2025.findings-naacl.283/. DeepSeek-AI. Deepseek-r1. Nature, 645:633–638, 2025. doi: 10.1038/s41586-025-09422-z.

Guan, X., Zhang, L. L., Liu, Y., Shang, N., Sun, Y., Zhu, Y., Yang, F., and Yang, M. rstar-math: Small llms can master math reasoning with self-evolved deep thinking. In Proceedings of the 42nd International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=5zwF1GizFa.

Han, C., Wang, Q., Peng, H., Xiong, W., Chen, Y., Ji, H., and Wang, S. LM-infinite: Zero-shot extreme length generalization for large language models. In Duh, K., Gomez, H., and Bethard, S. (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association

for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3991–4008, Mexico City, Mexico, 2024. Association for Computational Linguistics. URL https://aclanthology.org/ 2024.naacl-long.222.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

Hovy, E., Gerber, L., Hermjakob, U., Lin, C.-Y., and Ravichandran, D. Toward semantics-based answer pinpointing. In Proceedings of the first international conference on Human language technology research, 2001.

Kapuriya, J., Kaushik, M., Ganguly, D., and Bhatia, S. Exploring the role of diversity in example selection for in-context learning. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2962– 2966, 2025. URL https://doi.org/10.1145/ 3726302.3730194.

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. volume 35, pp. 22199–22213, 2022.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨uttler, H., Lewis, M., Yih, W.-t., Rockt¨aschel, T., et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. volume 33, pp. 9459–9474, 2020.

Li, T., Zhang, G., Do, Q. D., Yue, X., and Chen, W. Long-context llms struggle with long in-context learning. Transactions on Machine Learning Research, 2025a. URL https://openreview.net/forum? id=Cw2xlg0e46.

Li, Y., Hu, X., Qu, X., Li, L., and Cheng, Y. Test-time preference optimization: On-the-fly alignment via iterative textual feedback. In Proceedings of the 42nd International Conference on Machine Learning, 2025b. URL https:

//openreview.net/forum?id=ArifAHrEVD.

Lin, B. Y., Ravichander, A., Lu, X., Dziri, N., Sclar, M., Chandu, K. R., Bhagavatula, C., and Choi, Y. The unlocking spell on base llms: Rethinking alignment via incontext learning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=wxJ0eXwwda.

Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., and Chen, W. What makes good in-context examples for GPT-3? In Agirre, E., Apidianaki, M., and Vuli´c, I. (eds.), Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pp. 100–114, Dublin, Ireland and Online, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. deelio-1.10. URL https://aclanthology.org/ 2022.deelio-1.10.

Luo, M., Xu, X., Dai, Z., Pasupat, P., Kazemi, M., Baral, C., Imbrasaite, V., and Zhao, V. Y. Dr.icl: Demonstrationretrieved in-context learning, 2023. URL https:// arxiv.org/abs/2305.14128.

MetaAI. Introducing meta llama 3: The most capable openly available llm to date, 2024. URL https://ai.meta. com/blog/meta-llama-3/.

Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., and Zettlemoyer, L. Rethinking the role of demonstrations: What makes in-context learning work? In Goldberg, Y., Kozareva, Z., and Zhang, Y. (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11048–11064, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. emnlp-main.759. URL https://aclanthology.

org/2022.emnlp-main.759.

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. In-context learning and induction heads, 2022. URL https:// arxiv.org/abs/2209.11895.

Peng, B., Quesnelle, J., Fan, H., and Shippole, E. Yarn: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=wHBfxhZu1u.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2024. URL https:

//arxiv.org/abs/2412.15115.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.net/forum? id=4FWAwZtd2n.

Snell, C. V., Lee, J., Xu, K., and Kumar, A. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum? id=4FWAwZtd2n.

Sorensen, T., Robinson, J., Rytting, C., Shaw, A., Rogers, K., Delorey, A., Khalil, M., Fulda, N., and Wingate, D. An information-theoretic approach to prompt engineering without ground truth labels. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 819–862, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.60. URL https: //aclanthology.org/2022.acl-long.60.

Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pp. 35151–35174. PMLR, 2023.

Wan, X., Zhou, H., Sun, R., Nakhost, H., Jiang, K., ¸Serban, C¸., and Arık, S. O.¨ From few to many: Selfimproving many-shot reasoners through iterative optimization and generation. In International Conference on Learning Representations (ICLR), 2025. URL https:

//openreview.net/forum?id=JBXO05r4AV.

Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. Superglue: A stickier benchmark for general-purpose language understanding systems. volume 32, 2019.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Wu, Z., Wang, Y., Ye, J., and Kong, L. Self-adaptive in-context learning: An information compression perspective for in-context example selection and ordering. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1423–1436, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/

2023.acl-long.79. URL https://aclanthology. org/2023.acl-long.79.

Xu, Z., Ye, J., Liu, X., Liu, X., Sun, T., Liu, Z., Guo, Q., Li, L., Liu, Q., Huang, X., and Qiu, X. Detectiveqa: Evaluating long-context reasoning on detective novels, 2024. URL https://arxiv.org/abs/2409.02465.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang,

- J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang,
- K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang,

- X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan,
- Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 technical report, 2025. URL https: //arxiv.org/abs/2505.09388.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Yu, M., Chung, T. T., Zhou, C., Li, T., Lu, R., Li, J., Xu, L., Lu, H., Zhang, N., Li, J., and Zhou, J. Prelude: A benchmark designed to require global comprehension and reasoning over long contexts, 2025a. URL https:

//arxiv.org/abs/2508.09848.

Yu, M., Liu, L., Wu, J., Chung, T. T., Zhang, S., Li, J., Yeung, D.-Y., and Zhou, J. The stochastic parrot on LLM’s shoulder: A summative assessment of physical concept understanding. In Chiruzzo, L., Ritter, A., and Wang, L. (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 11416–11431, Albuquerque, New Mexico, April 2025b. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long. 569. URL https://aclanthology.org/2025.

naacl-long.569/.

- Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., and Zhou, J. Qwen3 embedding: Advancing text embedding and reranking through foundation models, 2025. URL https://arxiv.org/abs/2506.05176.
- Zhang, Z., Zhang, A., Li, M., and Smola, A. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda,

###### May 1-5, 2023. OpenReview.net, 2023. URL https: //openreview.net/pdf?id=5NTt8GFjUHkr.

### A. Details for Similarity-Based Demonstration Selection

This appendix provides implementation details for the similarity-based demonstration selection experiments in Section 4.3.

#### A.1. Candidate pool and data splits

For each task, we form a demonstration candidate pool from the task’s training split. All test queries are drawn from the task’s test split. Since candidate demonstrations and evaluated queries come from disjoint splits, there is no overlap between a test query and any candidate demonstration.

#### A.2. Embedding model and similarity

We embed each question (not the answer or rationale) using Qwen3-Embedding-4B (Zhang et al., 2025). Let e(q) ∈ Rd denote the embedding of a test question and e(x) ∈ Rd the embedding of a candidate training question. We measure semantic similarity by cosine similarity:

e(q)⊤e(x) ∥e(q)∥∥e(x)∥

s(q,x) =

.

For each test query q, we rank all candidates x by s(q,x).

#### A.3. Constructing the most-similar and most-dissimilar sets

Given a target number of demonstrations k, we construct: (i) the most-similar set by selecting the top-k candidates under s(q,x), and (ii) the most-dissimilar set by selecting the bottom-k candidates.

Unless otherwise stated, we present the selected examples to the LLM in descending order of similarity for the mostsimilar set, and ascending order of similarity for the mostdissimilar set.

#### A.4. Qualitative Example: When “Similar” Questions Provide Misleading CoT

Table 5 shows an illustrative failure case from a reasoning task. Although the retrieved demonstration question is highly similar under embedding similarity, its solution uses a different invariant/decomposition than the query. When the LLM is conditioned on this demonstration, it tends to reuse the same intermediate steps, leading to an incorrect conclusion. In contrast, a less similar (but structurally closer) demonstration encourages the correct decomposition and improves accuracy.

Test query (geometry). In the diagram, $\triangle XYZ$ is right-angled at $X,$ with

$YX=60$ and $XZ=80.$ The point $W$ is on $YZ$ so that $WX$ is perpendicular to $YZ.$ Determine the length of $WZ.$ [asy]

pair X, Y, Z, W;

- Y=(0,0); X=(36,48);
- Z=(100,0); W=(36,0); draw(X--Y--Z--X--W);

- label("Y", Y, SW); label("X", X, N); label("W", W, S);
- label("Z", Z, SE); label("60", (X+Y)/2, NW); label("80", (X+Z)/2, NE); {[/asy]}\\

###### Solution.

By the Pythagorean Theorem, \begin{align*} YZˆ2 &= YXˆ2 + XZˆ2

&= 60ˆ2+80ˆ2 &= 3600+6400 &=10000,

\end{align*} so $YZ=100.$ (We could also have found $YZ$ without using the Pythagorean

Theorem by noticing that $\triangle XYZ$ is a rightangled triangle with its right-angle at $X$ and $XY=60=3\ cdot 20$ and $XZ=80=4\cdot 20.$ This means that $\ triangle XYZ$ is similar to a 3-4-5 triangle, and so $YZ

=5\cdot 20=100.$)

Since $\triangle YXZ$ is right-angled at $X,$ its area is $$\ frac{1}{2}\cdot 60\cdot 80=2400.$$ Since $XW$ is perpendicular to $YZ,$ then the area of $\triangle YXZ$ is also equal to $$\frac{1}{2}\cdot 100\cdot XW=50XW.$$ Therefore, $50XW=2400,$ so $XW=48.$ By the Pythagorean Theorem, \begin{align*}

WZˆ2 &= 80ˆ2 - 48ˆ2 &= 6400 - 2304 &= 4096. \end{align*} Thus, $WZ = \sqrt{4096}=\boxed{64}.$

An alternative solution comes by noticing that $\triangle XZW$ and $\triangle YZX$ are similar. Therefore \[\frac{WZ}{XZ }=\frac{XZ}{YZ}\] or \[\frac{WZ}{80}=\frac{80}{100}=\ frac45.\] This tells us that \[WZ=\frac45\cdot80=\boxed {64}.\]

Similar demonstration with cosine similarity. Test query. In the diagram, $\triangle ABE$, $\triangle BCE$ and $\triangle

CDE$ are right-angled, with $\angle AEB=\angle BEC = \ angle CED = 60ˆ\circ$, and $AE=24$. [asy]

pair A, B, C, D, E;

- A=(0,20.785);
- B=(0,0);
- C=(9,-5.196);
- D=(13.5,-2.598);
- E=(12,0);

- draw(A--B--C--D--E--A);
- draw(B--E);
- draw(C--E);

- label("A", A, N);
- label("B", B, W);
- label("C", C, SW);
- label("D", D, dir(0));
- label("E", E, NE); [/asy] Find the length of $CE.$

Solution. We find $CE$ by first finding $BE$. Since $AE = 24$ and $\angle AEB = 60ˆ\circ$ and $AEB$ is a

right triangle, then we can see that $AE$ is the hypotenuse and $BE$ is the shorter leg, so $BE = \dfrac {1}{2} \cdot 24 = 12.$

Likewise, since $BE = 12$ and $\angle BEC = 60ˆ\circ$, then $CE

= \dfrac{1}{2} \cdot 12 = \boxed{6}$.

Why it is misleading. Since the retrieved example uses a 30◦−60◦−90◦ triangle with the 1 : 2 ratio and never uses an altitude-to-hypotenuse configuration, its method does not transfer.

Table 5. A qualitative example illustrating why question-level semantic similarity selects demonstrations with incompatible reasoning trajectories.

#### A.5. Analysis of Similarity in Different LLM Types

###### Figures 9 and 10 compare similarity-based selection across non-reasoning and reasoning LLMs separately.

###### BANKING77

DetectiveQA

71

70

Sim>Ori/Dis

| |
|---|

Original

45

68

Most Similar

Sim>Ori/Dis Ori/Dis>Sim Original

Most Dissimilar

| |
|---|

| |
|---|

20

65

Most Similar

Most Dissimilar

Accuracy

20 60 100

20 60 100

geometry

number_theory

|| |
|---|
<br><br>| |
|---|
|Sim>Ori/Dis Ori/Dis>Sim|Most Most<br><br>|Similar Dissimilar|
|---|---|---|---|
| |Original<br><br>| | |
| | | | |
| | | | |

63

52

48

60

44

57

Ori/Dis>Sim

Most Similar

| |
|---|

Original

Most Dissimilar

20 60 100

20 60 100

Number of examples in-context

- Figure 9. Performance with original (ori), similarity(sim) and dissimilar(dis) sets averaged across three non-reasoning LLMs. The area between the two sets is filled with colors, indicating the relative performance at each point.

20 60 100

20

45

70

BANKING77

| |
|---|

Sim>Ori/Dis

Original

Most Similar

Most Dissimilar

20 60 100

67

69

71

DetectiveQA

| |
|---|

Ori/Dis>Sim

Original

Most Similar

Most Dissimilar

20 60 100

36

45

56

geometry

| |
|---|

Ori/Dis>Sim

Original

Most Similar

Most Dissimilar

20 60 100

50

63

75

number_theory

| |
|---|

Ori/Dis>Sim

Original

Most Similar

Most Dissimilar

Number of examples in-context

Accuracy

- Figure 10. Performance with original (ori), similarity(sim) and dissimilar(dis) sets averaged across two reasoning LLMs. The area between the two sets is filled with colors, indicating the relative performance at each point.

### B. Statistical Robustness on a New ICL Subset

We compute the mean and standard deviation across five random demonstration-ordering seeds, repeating the analysis on a newly sampled ICL subset. These results strengthen the claims in Figures 7, 8, and 6: the observed trends persist beyond a single ordering or candidate pool.

Model Shots Origin (µ ± σ) First (µ ± σ) Qwen3-14B 16 86.33 ± 0.86 89.33 ± 1.28

32 88.07 ± 0.34 92.19 ± 0.50 64 88.85 ± 0.80 93.30 ± 0.73 128 91.19 ± 0.76 94.44 ± 0.57

Qwen3-8B 16 83.48 ± 0.81 83.63 ± 0.79 32 85.19 ± 0.51 87.41 ± 0.74 64 88.04 ± 0.84 91.08 ± 0.69 128 86.93 ± 1.42 88.52 ± 0.49

- Table 6. Reasoning-oriented LLMs on number theory across five random ordering seeds.

Model Shots Origin (µ ± σ) Wrong (µ ± σ)

Llama-3.1-8B 16 23.55 ± 3.58 35.57 ± 0.72 32 22.21 ± 8.08 35.91 ± 1.13 64 25.97 ± 6.42 35.61 ± 1.62 90 32.78 ± 6.29 36.12 ± 1.16

Qwen2.5-14B 16 58.37 ± 0.83 61.05 ± 1.19 32 59.00 ± 0.91 59.67 ± 0.97 64 57.70 ± 1.13 60.75 ± 1.25 90 57.41 ± 1.58 59.46 ± 0.70

- Table 7. Non-reasoning LLMs on geometry across five random ordering seeds.

Task Target model Shots Stronger LLM Self-generated

number theory Qwen3-8B 16 86.26 ± 0.61 83.63 ± 0.79 32 89.55 ± 1.00 87.41 ± 0.74 64 88.19 ± 1.63 91.08 ± 0.69 128 81.15 ± 2.01 88.52 ± 0.49

geometry Llama-3.1-8B 16 33.53 ± 2.05 35.57 ± 0.72 32 34.28 ± 2.06 35.91 ± 1.13 64 35.61 ± 1.05 35.61 ± 1.62 90 33.48 ± 0.52 36.12 ± 1.16

- Table 8. CoT-ICL generated from stronger LLMs versus selfgenerated demonstrations across five random ordering seeds. Values report µ ± σ.

### C. Curvature-based Smoothness: Details and Implementation

To quantify the relationship between demonstration ordering smoothness and ICL performance, we develop Algorithm 1. The algorithm takes as input multiple orderings of demonstrations and their corresponding performance scores, and outputs a correlation coefficient between ordering smoothness and performance.

#### C.1. Demonstration Format

For the smoothness analysis in Section 5.2, each demonstration is embedded as a single text string that concatenates Question + Chain-of-Thought + Answer. We use the same template in Appendix E for constructing demonstrations. All curvature results in the main paper use this template unless stated otherwise.

Algorithm 1 Curvature–Performance Correlation Analysis

Given a query, answer yes or no to the query. The predicted answer must come from the demonstration examples with the exact format. The examples are as follows:

Input: k orderings {E(j)}kj=1, where E(j) = [e(1j), . . . , e(Nj)]⊤; performance scores S = [S1, . . . , Sk] Output: Pearson correlation coefficient r between smoothness scores m and performance S Initialize smoothness scores m ← [0, . . . , 0] (length k) for j = 1 to k do

Question: In the sentence "{text_1}", does the pronoun "{

span2_text_1}" refer to {span1_text_1}? Answer: {answer_1}

... Question: In the sentence "{text_n}", does the pronoun "{

span2_text_n}" refer to {span1_text_n}? Answer: {answer_n} Now predict the answer for the following query: Question: In the sentence "{text_i}", does the pronoun "{

Initialize curvature list Θ ← [ ] for i = 2 to N − 1 do

span2_text_i}" refer to {span1_text_i}? Reply in the following format: Answer: [yes | no]

v1 ← e(ij) − e(i−j)1 v2 ← e(i+1j) − e(ij) if ∥v1∥ > 0 and ∥v2∥ > 0 then

Figure 11. Prompt for WSC task

v1⊤v2 ∥v1∥ ∥v2∥ c ← min(1, max(−1, c)) θ ← arccos(c) Append θ to Θ

c ←

We also control for confounding factors unrelated to ordering quality by using dataset-provided CoT rationales and answers, avoiding performance degradation due to incorrect or low-quality generated rationales (i.e., self-generated CoT in section 5.1). Under this setup, we study models that can interpret and leverage the provided CoT.

end if end for θ¯(j) ← mean(Θ)

1 1 + θ¯(j)

m[j] ←

end for r ← PearsonCorrelation(m, S) return r

Our primary evaluation uses Qwen3 (8B and 14B) across varying numbers of demonstrations and three tasks (geometry, number theory, and DetectiveQA), as these models satisfy the above criteria and provide a stable platform for many-shot experiments.

- C.2. Embedding Model

Similar to section 4.3, we encode demonstrations using Qwen3-Embedding-4B (Zhang et al., 2025). Let di denote the formatted demonstration string and Embed(·) the em-

bedding model. We obtain vectors ei = Embed(di) ∈ Rd using the embedding model’s default output representation with vLLM deployment (Kwon et al., 2023).

C.3. Correlation Protocol

To study the relationship between ordering smoothness and downstream accuracy, we use the multiple random permutations of a fixed demonstration set with n = 128 demonstrations from Section 4.4. For each ordering O, we compute the bounded smoothness score m(O) = 1/(1 + θ¯(O)) and evaluate task accuracy under the corresponding prompt. We then compute the Pearson correlation between {m(O)} and accuracy over the sampled orderings.

- D. CDS: Details and Implementation

### E. Prompt Formatting for Each Task

#### E.1. SuperGlue

We evaluate the Winograd Schema Challenge (WSC) for coreference resolution, and the Choice of Plausible Alternatives (COPA) for open-domain commonsense causal reasoning. Both are formatted as a binary-label classification task. The prompt for inference is presented in Figure 11.

Answer in A or B. The predicted answer must come from the demonstration examples with the exact format. The examples are as follows: Premise: {premise_1} Question: What is the {question_1} for this? Options:

- A. {choice1_1}
- B. {choice2_1} Answer: {answer_1}

... Premise: {premise_n} Question: What is the {question_n} for this? Options:

- A. {choice1_n}
- B. {choice2_n} Answer: {answer_n}

Now predict the answer for the following query: Premise: {premise_i} Question: What is the {question_i} for this? Options:

- D.1. Model Studies

We design our experiments to isolate the effect of demonstration ordering in many-shot CoT-ICL. To this end, we focus on reasoning-oriented LLMs that exhibit a positive scaling trend with more demonstrations, since such models demonstrate in-context learning capacity and should benefit from improved ordering.

- A. {choice1_i}
- B. {choice2_i} Reply in the following format: Answer: [A | B]

Figure 12. Prompt for COPA task

#### E.2. TREC

We evaluate the Text REtrieval Conference (TREC) Question Classification dataset with 50 fine class labels. The prompt for inference is presented in Figure 13.

Given a question, predict the label of the question. You can only

make predictions from the following categories: { LIST_OF_CATEGORIES}

Please predict the label of the FINAL question with the provided

demonstration example queries as follows: question: {question_1} label: {label_1}

... question: {question_n} label: {label_n}

Now predict the answer for the following query: question: {question_i}

Reply in the following format: label: [category_name]

Figure 13. Prompt for TREC task

#### E.3. BANKING77

We evaluate the BANKING77 dataset with 77 fine-grained intents in the banking domain. The prompt for inference is

- presented in Figure 14.

Given a question, predict the label of the question. You can only

make predictions from the following categories: { LIST_OF_CATEGORIES}

Please predict the intent category of the FINAL query with the

provided demonstration example queries as follows: service query: {question_1} intent category: {label_1}

... service query: {question_n} intent category: {label_n}

Now predict the intent category for the following query: service query: {question_i}

Reply in the following format: intent category: [category_name]

Figure 14. Prompt for BANKING77 task

- E.4. NLU

We evaluate the NLU dataset with 68 fine-grained intents in the conversational domain. The prompt for inference is presented in Figure 15.

- E.5. GSM8K

We evaluate the GSM8K dataset for grade school math word problems. The prompt for inference is presented in Figure 16.

- E.6. MATH

Given a question, predict the label of the question. You can only

make predictions from the following categories: { LIST_OF_CATEGORIES}

Please predict the intent category of the FINAL utterance with

the provided demonstration example queries as follows: utterance: {question_1} intent category: {label_1}

... utterance: {question_n} intent category: {label_n}

Now predict the intent category for the following utterance: utterance: {question_i}

Reply in the following format: intent category: [category_name]

Figure 15. Prompt for NLU task

In the end of the response, add a summary ‘The answer is [answer

].’ Q: {question_1} A: {CoT_1} {answer_1}

... Q: {question_n} A: {CoT_n} {answer_n}

### Q: {question_t} ### A: Let’s think step by step.

Figure 16. Prompt for GSM8K task

The prompt for inference is presented in Figure 17.

Write a response that appropriately completes the request and

wrap the final answer inside \boxed{}. Problem: {question_1} Solution: {CoT_with_answer_1}

... Problem: {question_n} Solution: {CoT_with_answer_n}

### Problem: {question_t} ### Solution: Let’s think step by step.

Figure 17. Unified prompt for MATH task

We evaluate the Mathematics Aptitude Test of Heuristics (MATH) dataset for mathematics competition problems, including the question types of counting and probability, prealgebra, geometry, precalculus, number theory and algebra.

#### E.7. DetectiveQA

DetectiveQA (Xu et al., 2024) is a long-context narrative reasoning benchmark. Each instance includes an evidence section as part of the input, and the goal is to answer a question by reasoning over this evidence. DetectiveQA additionally provides annotated reasoning chains for deriving the answer from the evidence. When constructing CoT demonstrations, we use the derivation labeled “-1” as the corresponding chain-of-thought.

To avoid potential information leakage, we further filter the test split. We exclude any test instances that share the same data source (i.e., novel ID) with the training split. This prevents the model from receiving extra clues through CoT-ICL demonstrations drawn from the same underlying narrative. The prompt for inference is presented in Figure 18.

Below is an instruction that describes a task.\n Select the correct option from A/B/C/D. Answer with ’The answer is {A/ B/C/D}.’ in the end of your response.\n\n"

Question: {question_1} Context: {context_1} Options:

- A. {option_1_1}
- B. {option_1_2}
- C. {option_1_3}
- D. {option_1_4} Answer: {derivation_1} The answer is {answer_1}.

... Question: {question_n} Context: {context_n} Options:

- A. {option_n_1}
- B. {option_n_2}
- C. {option_n_3}
- D. {option_n_4} Answer: {derivation_n} The answer is {answer_n}.

### Question: {question_i} ### Context: {context_i} ### Options:

- A. {option_i_1}
- B. {option_i_2}
- C. {option_i_3}
- D. {option_i_4} ### Answer:

Figure 18. Prompt for DetectiveQA task

