## arXiv:2512.17351v1[cs.CL]19Dec2025

### Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers

Zeyuan Allen-Zhu zeyuanallenzhu@meta.com FAIR at Meta

May 2, 2025

(overall version 2.0)∗

Abstract

Understanding architectural differences in language models is challenging, especially at academic-scale pretraining (e.g., 1.3B parameters, 100B tokens), where results are often dominated by noise and randomness. To overcome this, we introduce controlled synthetic pretraining tasks that isolate and evaluate core model capabilities. Within this framework, we discover Canon layers: lightweight architectural components—named after the musical term “canon”—that promote horizontal information flow across neighboring tokens. Canon layers compute weighted sums of nearby token representations and integrate seamlessly into Transformers, linear attention, state-space models, or any sequence architecture.

We present 12 key results. This includes how Canon layers enhance reasoning depth (e.g., by 2×), reasoning breadth, knowledge manipulation, etc. They lift weak architectures like NoPE to match RoPE, and linear attention to rival SOTA linear models like Mamba2/GDN—validated both through synthetic tasks and real-world academic-scale pretraining. This synthetic playground offers an economical, principled path to isolate core model capabilities often obscured at academic scales. Equipped with infinite high-quality data, it may even predict how future architectures will behave as training pipelines improve—e.g., through better data curation or RL-based post-training—unlocking deeper reasoning and hierarchical inference.

∗Physics of Language Models is a series of works, of which Part 4.1 is a standalone paper. Due to technical issues, earlier versions of Part 4.1 could not be successfully submitted to arXiv. The version history of this Part 4.1 paper is as follows: V1 appeared on SSRN on May 2, 2025; V1.1 (May 18, 2025) improves writing and adds the relu2 experiments (and is accepted by NeurIPS 2025); V2 (this version) adds GDN experiments, tightens some experiments for a stronger, fairer comparison, and re-organizes sections. Code release and future updates can be found on SSRN and the project page physics.allen-zhu.com.

ZA sincerely thanks Vahab Mirrokni for the invitation to the Yale workshop in October 2023, where this research was sparked through enlightening discussions with Vahab Mirrokni and Peilin Zhong. Canon layers build on the idea of uniform attention previously explored in [6]. ZA thanks Alberto Alfarano for introducing the papers [31, 45, 66, 82], and the PyTorch scaled dot product attention function. At Meta, we extend our heartfelt gratitude to Lin Xiao and Kristin Lauter for their insightful discussions and unwavering supports, which made this research possible. Special thanks go to Wangzhi Dai, Sam Doud, Dinesh Kannappan, Niki Kim, Junjie Qian, Ammar Rizvi, Travis Seevers, and Stephen Hartken at Meta, as well as Abraham Leal from W&B; without their invaluable technical assistance, the experiments presented in this paper would not have been feasible. We are deeply grateful to Songlin Yang and Ali Behrouz for providing detailed instructions on replicating their academic-scale pretraining experiments, and Fangcheng Sun for many helpful conversations on architecture design in general.

Contribution statement. ZA proposed all ideas, conducted all investigations, implemented all code, performed all experiments, authored the entire manuscript, and managed all necessary compliance reviews and social promotions; the term Canon Layers was jointly conceived and designed with Xiaoli Xu.

#### 1 Introduction

R

ecent advances in large language models (LLMs) have sparked transformative progress across numerous tasks, including question answering, summarization, translation, code generation [14, 16, 40, 64]. Despite rapid progress, systematic understanding of effective

neural architecture design has remained elusive, fundamentally hindered by some major challenges.

- Challenge 1: Pretraining loss as an unreliable proxy for intelligence. Architectural comparisons often rely on perplexity or cross-entropy loss, but these metrics do not reliably reflect real-world capabilities—especially since natural data is skills-mixed. For example, state-space architectures like Mamba [19, 26] frequently achieve lower perplexity early in training due to rapid memorization, yet perform poorly on complex reasoning tasks. Reliance on early stopping via perplexity is thus problematic: it may lead to comparing models that have merely internalized surface-level linguistic patterns without developing deeper reasoning or factual understanding [32].
- Challenge 2: Noise below emergence thresholds. Emergent abilities—complex skills that only arise in large-scale models (e.g., 7B parameters, 10T tokens [1])—complicate architectural comparisons at smaller, academic scales (e.g., 1.3B parameters, 100B tokens [10, 25, 73]). At these scales, small benchmark gains (e.g., 2%) often result from random initialization (and/or data shuffling)—variance that can cause 2–4% swings in accuracy (see Figure 1). More fundamentally, models fail even the simplest 2-hop reasoning tasks, performing no better than random guessing.1 This basic reasoning floor masks architectural differences in more advanced cognitive skills, making evaluation at this scale deeply unreliable. While large-scale industry training might reveal these differences, its prohibitive cost blocks systematic ablations, impeding academic contributions to rigorous architecture science—and often reducing design choices to heuristics and guesswork.
- Challenge 3: Grokking, Data Quality, and Curriculum Learning. Failures in complex reasoning tasks typically stem from deficiencies in training data, not architectural limitations. Too few challenging samples and a lack of intermediate-complexity data often force models to rely on unstable grokking behavior—where generalization only emerges after unnecessarily long pretraining [44]—and disrupt curriculum learning [11]. For instance, models lacking 2-hop reasoning data may unpredictably learn 3-hop tasks after extensive exposure to 1-hop and 3-hop examples. This makes training highly sensitive to randomness, further complicating architectural comparisons. Reinforcement learning (RL)-based post-training methods, such as GRPO [55] and PPO [54], aim to address this by delivering tailored data at optimal difficulty levels. While effective, these methods introduce new experimental confounds—it becomes unclear whether performance gains stem from pretraining, RL fine-tuning, stochastic training dynamics, or architectural strength.

Our approach: Atomic decomposition of intelligence. To overcome the noise and cost of real-world pretraining—especially at academic scales where even 2-hop reasoning fails to emerge—we decompose intelligence into core (ideally atomic!) components, such as reasoning depth and breadth, and design synthetic, controllable pretrain tasks to isolate and evaluate them independently. This framework sharply characterizes architectural strengths and scalability under clean, idealized conditions (see Figure 1), offering a principled and economical path for architecture design.

This directly addresses Challenge 1 by enabling single-skill evaluations, minimizing the confounding factors prevalent in real-world pretraining data. For example, it allows rigorous comparisons of whether architecture A outperforms architecture B in reasoning depth, while ensuring modifications do not degrade other capabilities. By isolating intrinsic architectural biases, synthetic

1In our simplest 2-hop reasoning tasks, birth years for 3 individuals are presented, followed by 3 “[name2] was born in the same year as [name1]” equivalences. The model is prompted to infer the second group’s birth years. Academic-scale pretrained models can only guess. See Result 12.

Real-life Pretraining Obscures Architecture Differences

Synthetic Pretraining Enables Reliable Comparison

At academic scale (1.3B parms, 100B tokens):

Our solution: synthetic pretraining playground

[Figure 1]

[Figure 2]

- • knowledge capacity (Capo)
- • knowledge manipulation (Mano)
- • hierarchical structures (Lano)

- • reasoning depth (Depo)
- • reasoning breadth (Brevo)

FineWeb-edu

SlimPajama

Clear & controlled outcomes

✓ Mini scaling-laws reveal model limits

✓ Sharply reveal model differences (e.g., 2x reasoning depth)

datadifficulty

architectural differences lost in noise

Llama(RoP E)

this paper

across models: 1-2%

[Figure 3]

[Figure 4]

across random seeds: ≥2-4%

insufficient to see architectural strengths models fail simplest 2-hop reasoning:

X born in 1970. Y same birth year as X. When was Y born?

model size

GPT2small size

✓ Early emergence of advanced skills ✓ Low cost supports rigorous studies

real-world data: too skill-mixed, delays “emergent” skills

|computationally infeasible|
|---|

controlled study ≈

Beyond this scale:

✓ High-quality data predicts future architectures

Figure 1: Architecture search in noisy real-life pretraining (good luck!) vs. our synthetic playground (scientific rigor). See Figure 21 (Page 43) for more benchmark variability, including fixed data and varied model random init.

pretrain tasks reveal properties often obscured by noise and mixed signals in typical real-life setups.

Challenge 2 is mitigated by lowering resource needs for rigorous comparisons. Synthetic benchmarks yield infinite high-quality data, enabling meaningful pretraining even for smaller models (e.g., GPT2-small) where complex skills might otherwise not emerge. In these controlled environments, capabilities like deep multi-hop reasoning emerge clearly and reliably, allowing rapid identification of architectural limitations, investigation of mini scaling-laws, and uncover trends that real-world pretrained models often fail to reveal due to noise or insufficient signal despite extensive training.

For Challenge 3, we manage data difficulty distributions to ensure adequate representation of intermediate-complexity samples, smoothing learning curves and enabling the early and consistent emergence of advanced skills—unlike less predictable real-world data prone to grokking-driven instability. As training pipelines improve—via better data curation or RL-based continued pretraining—synthetic pretrain benchmarks may provide predictive insight into which architectures best support scaling to more advanced tasks in the future.

We draw inspiration from physics, where idealized settings—such as frictionless planes or vacuum chambers—reveal first principles by removing confounding factors. Similarly, synthetic tasks eliminate the noise, randomness, and data contamination of real-world datasets, enabling clean, controlled, apples-to-apples architectural comparisons, much like Galileo’s Pisa tower experiment.

This paper’s key contributions are summarized below:

- Result 0: Building the Synthetic Playground (Section 2+3). We introduce five synthetic pretraining tasks—Depo (reasoning depth), Brevo (reasoning breadth), Capo (knowledge capacity), Mano (knowledge manipulation), and Lano (hierarchical language structure). This controlled setup reveals clear, commonsense capability trends at small scale: linear attention (e.g., GLA [72]) underperforms consistently; state-space model Mamba2 [19] excels at knowledge but struggles with reasoning; and GDN [73] and Transformers dominate complex reasoning.
- Result 1: Canon Layers Add Horizontal Information Flow (Section 4). Transformers lack horizontal information flow within layers, leading to inefficiencies even on simple tasks like associative recall. Drawing on the musical canon (overlapping repetition), we introduce Canon layers, horizontal “residual links” across neighboring tokens that can be flexibly inserted at multiple points — before attention (Canon-A), inside attention (Canon-B), before MLP (Canon-C), inside MLP (Canon-D). While Canon layers can be implemented in many ways—even simple random averaging is highly effective—this paper focuses on trainable 1-d linear convolutions of kernel size

4. This is lightweight and integrates seamlessly into any sequence model with minimal code. Results 2–5: When Transformer Meets Canon (Section 5).

- • Boost performance. In our playground, Canon layers improve reasoning depth (200–400%), reasoning breadth (30%), knowledge manipulation length (30%), and more. These stem from enhanced hierarchical learning dynamics and come with minimal computational overhead.
- • Reviving NoPE. Integrating Canon layers transforms NoPE models into strong performers, often matching or surpassing RoPE(+Canon). Canon layers outperform positional fixes like ALiBi [45] or H-Alibi [31], and reducing/removing RoPE usage improves length generalization.
- • Ablation study. Canon layers contribute cumulatively across sublayer positions (CanonA/B/C/D), independently of attention or MLP components. Residual Canon improve training efficiency; minimal parameter tuning is required without compromising stability.
- • MLP and MoE. Canon layers can recover some knowledge capacity lost in gated MLP or mixture-of-expert (MoE) architectures, via improved training efficiency and stability.

###### Results 6–9: When Linear Models Meet Canon (Section 6).

- • Universal boost. Across all linear architectures—GLA, Mamba2, and GDN—Canon layers consistently enhance reasoning: in-context (Depo/Brevo), knowledge (Mano), and structural (Lano), though by varying degrees.

- – For linear attention (GLA), Canon lifts reasoning depth from 1 to 4-hop, doubles reasoning breadth and knowledge length, and even surpasses Mamba2.
- – Mamba2’s built-in conv1d (partial Canon-B) drives most of its gains; removing it drops performance to GLA, while replacing it with full Canon yields further improvements.
- – GDN benefits least, as its gating and delta updates capture part of Canon-like behavior.

- • Ablation findings. Canon’s residual design ensures stability and never hurts performance. Canon-ACD alone often matches conv1d/Canon-B, showing horizontal context flow is universal—not limited to linear-attention or SSM sub-layers.
- • Architectural insight. Most linear-model performance (for Mamba2/GDN) is achievable with the simple GLA+Canon design, suggesting that many modern refinements might largely replicate Canon-like mixing rather than introduce new computation.

###### Results 10–11: Comparing Transformers and Linear Models (Section 7) .

- • Controlled comparisons. Equipping all architectures with full Canon layers enables a fair, apple-to-apples evaluation. Linear models show ∼40% higher knowledge capacity, but Transformers reach 2–4× greater reasoning depth and stronger structural reasoning.
- • Root cause of shallow reasoning. Linear models fall short not from insufficient memory—each layer’s recurrent state is vastly over-provisioned—but from cumulative compression and retrieval errors, pinpointing memory dynamics as the main bottleneck.
- • Path forward. Canon-equipped Transformer–linear hybrids can mitigate these limits, enabling deep reasoning with linear efficiency.

Result 12: Academic-Scale Real-World Pretraining (Section 8) . Pretraining 1.3Bparameter models on 100B tokens (context length 4096) shows high noise and limited resolution, making many architectural comparisons statistically unreliable. Still, several consistent patterns emerge. Canon layers markedly improve NoPE and GLA—raising them to match RoPE and Mamba2/GDN, respectively—while removing conv1d reduces Mamba2 to GLA level. Linear models lag behind full Transformers on retrieval-heavy tasks even with Canon, and all models fail 2-hop reasoning, even in short (100-token) contexts, underscoring the limits of academic-scale pretraining. Reducing or removing RoPE improves long-context generalization when Canon layers are present. These trends mirror our synthetic results (Results 3, 6.1, 7.1, 8.1, 9, 10, 11).

Design Criteria for Synthetic Pretrain Tasks

| ||…|
|---|
<br><br>|…|
|---|
<br><br>mentalreasoning<br><br>system 2 reasoning (CoT)|
|---|---|
| | |

[Figure 5]

[Figure 6]

###### Ensure real-world relevance

###### Challenge architectural depth:

mentalreasoning

avoid shallow tasks (e.g., associative recall)

avoid tasks solvable by external tools

|“452352 + 547647 = 999999”<br><br>[Figure 7]|
|---|

[Figure 8]

Test mental reasoning (system-1):

mental depth 4 × 8 CoT steps = 32 total steps.

[Figure 9]

Focus on short (e.g., 4096) context length

###### our focus for architecture design

|| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
<br><br>context length 4096<br><br>summarization (CoT)<br><br>long context (e.g., 1M tokens)|
|---|

long context often summarized to

short windows for deep reasoning

Figure 2: Our design criteria for synthetic pretrain tasks.

In summary, Canon layers fundamentally improve horizontal information flow across diverse architectures, enabling deeper reasoning and efficient scalability. Combined with synthetic benchmarks, they provide systematic insights into future opportunities in model design.

Future research. We plan to extend our study of Canon layers beyond the academic scale. Preliminary results from larger pretrains (1–8B models on 1–2T tokens) closely align with the findings reported here. Notably, several synthetic trends—such as Transformer+Canon strongly outperforming Transformer, GLA+Canon matching GDN and outperforming Mamba2—become clearly observable at these larger scales. Code is available on GitHub [2], some models on HuggingFace, and all resources are linked at physics.allen-zhu.com.

#### 2 Synthetic Tasks for Decomposing Intelligence

We design synthetic tasks to systematically evaluate specific capabilities of language model architectures under controlled conditions, minimizing confounds and enabling clean comparisons. Task selection is guided by four criteria:

- Criterion 1: Tasks must not be shallow. Shallow tasks—like associative recall or copying—are easily solvable by small and shallow models, and do not meaningfully test architectural strength. Deep learning relies on stacked layers to progressively learn abstract features [4], so tasks involving hierarchical reasoning better evaluate architectural scalability and efficiency.
- Criterion 2: Emphasis on mental thinking. Tasks should assess a model’s ability to reason internally without Chain-of-Thought (CoT). While CoT helps decompose problems, it does not reflect intrinsic “system 1” reasoning [77]. For example, a model reasoning 4 steps internally and 8 via CoT achieves 32 steps, but only internal ones reflect architectural strength. Current models like o3/R1 produce verbose reasoning traces even for trivial prompts (e.g., “Hello”)—revealing inefficiencies in system 1. To guide architectural progress, tasks must target mental reasoning.
- Criterion 3: Avoid emphasis on length generalization. Length generalization is often unstable—sensitive to random seeds and training order [82]—and thus unreliable for comparing architectures. While length generalization is important, models over-optimized for long contexts (e.g., 100k tokens) may exhibit reduced performance on standard lengths like 4096 tokens.2 In practice, long inputs are typically summarized into shorter windows before reasoning, so we prioritize evaluating architectures on dense, 4096-token contexts, where critical reasoning unfolds.
- Criterion 4: Relevance to real-world skills. Tasks should prioritize broadly applicable skills while avoiding capabilities better suited to external tools. For example, large-number arithmetic

2This is observed in methods like ALiBi [45], Halibi [31], and Mimetic initialization [66], whose performance degrades on shorter contexts, as we show in this paper.

| | |
|---|---|
|Five Synthetic Tasks Isolating Atomic Skills|❖ (MANO): Knowledge manipulation|
|❖ (DEPO): Mental reasoning depth<br><br>❖ (BREVO): Mental reasoning breadth<br><br>❖ (CAPO): Knowledge capacity<br><br>❖ (LANO): Hierarchical language structure<br><br>(directed path given in random order)<br><br>⟹ What’s the 𝑘-th successor of 𝐴?<br><br>(DAG given in random order)<br><br>⟹ What does A<br><br>depend on, list in<br><br>topological order?<br><br>[name] was born in [year], hometown is [city], works for [company]…<br><br>How many bit-per-parameter can a model store?<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>|⟹ What’s answer mod 23?|
|---|
<br><br>Structural reasoning: resolving ambiguity via global dynamic programming on CFG languages<br><br>13 20 15 2<br><br>+ −<br><br>×<br><br>multi-hop reasoning on knowledge<br><br>(i.e., 23×23 lookup tables)<br><br>|1 2 3 3 1 3 3 1 2 1 2 2 1 1 1 1 2 ...|
|---|
<br><br>parse tree 2 parse tree 1<br><br>01010 010110 1110 00110<br><br>⋯<br><br>|⋯|
|---|
<br><br>|⋯|
|---|
<br><br>⋯| |

- Figure 3: Overview of our five synthetic tasks, each isolating an atomic skill for rigorous architectural comparison.

(e.g., adding 10-digit numbers) is theoretically interesting but can be delegated to Python interpreters; failures in this area typically reflect limited data exposure rather than architectural weaknesses (e.g., Llama3-70B miscalculates 452352 + 547647). Synthetic tasks should focus on universally relevant skills, aligned with real-world applications, to ensure meaningful assessments.

##### 2.1 Our First Set of Five Synthetic Pretrain Tasks

To operationalize the criteria above, we design five synthetic tasks—each targeting a distinct dimension of language model capability. We name them Depo, Breo, Capo, Mano, and Lano.

Task Depo: Mental reasoning depth. Reasoning depth represents a fundamental capability for LLMs, requiring models to retrieve information through multi-step computation. Task Depo evaluates reasoning depth as k-hop traversal over directed permutations, where models compute the k-th successor for each query q entirely internally, without intermediate steps like Chain-of-Thought (CoT).3 Each instance is formatted as:

<bos> x1 y1 x2 y2 ... xn yn <query_k1> q1 a1 <query_k2> q2 a2 ... <eos> Here, 2n tokens encode n directed edges xi → yi, forming a random permutation of n nodes.

The dataset is controlled by two parameters: N, the maximum permutation size, and K, the maximum reasoning depth. During training, n is sampled from [3,N], while k ∈ [1,K]. Context lengths are fixed to 2048 tokens. We employ two variants of Depo:

- • Depo1: Each node spans 1–2 tokens from vocab size 50, with N = 225, 300, 375 and K = 8.
- • Depo2: Each node spans 5–7 tokens from vocab size 4, with N = 75, 100, 125 and K = 16.

Evaluation focuses on both the hardest cases (n = N, k = K) and intermediate difficulty (k = K/2). For weaker models, we utilize reduced training setups with K = 4, denoted Depo1(K = 4) and Depo2(K = 4). The full methodological details are provided in Appendix A.1.

Task Brevo: Mental reasoning breadth. This evaluates a model’s ability to process multiple dependencies simultaneously, as required in tasks involving tree-like traversal or dependency graphs. For example, solving queries like “Who are Alice’s nephews?” or GSM-like examples requires parallel reasoning across branches of a graph to process relationships bottom-up [75]. Task Brevo isolates this capability using recursive traversal of directed acyclic graphs (DAGs), abstracting away natural language or arithmetic complexities. Each task instance is formatted as:

<bos> x1 y1 x2 y2 ... xm ym <query> q <ans> a1 a2 ... ap <eos>

Here, 2m tokens define m edges xi → yi, representing dependencies where yi depends on xi. Upon receiving a query vertex q, the model outputs all vertices recursively reachable from q, sorted in topological order starting from the leaves (e.g., u → v → q yields output u followed by v).

3Using CoT would reduce the k-hop task to simpler 1-hop associative recall.

The dataset is parameterized by N, the maximum graph size, with DAGs created using n ≤ N nodes, each of degree at most 4. Pretraining data is sampled by varying graph sizes, while testing focuses on the hardest graphs (n = N). We employ two variants of Brevo:

- • Brevo1: Each vertex name spans a single token, with N = 70/90/110, fit within 1024 tokens.
- • Brevo2: Name spans 2–4 tokens of vocab size 4, with N = 30/40/50, fit within 1536 tokens.

A key discovery from [75] revealed that, due to the non-uniqueness of valid outputs, language models must preprocess the entire topological order of the DAG mentally before generating the first token a1. This insight confirms that our synthetic data rigorously evaluates reasoning breadth by requiring models to globally process the underlying graph structure before producing outputs.

Task Capo: Knowledge capacity. Task Capo evaluates a model’s efficiency in encoding factual knowledge directly within its parameters, quantified as bits per parameter, which measures reliable storage capacity. Following the framework in [8], synthetic datasets of (fake) biographies are constructed to test knowledge retention. Each biography includes several attributes (e.g., birthdate, university, employer, etc.) and is presented in diverse paraphrased formats to reduce surface-level memorization [5, 7]. Capacity is measured using the next-token prediction distribution, accounting for both exact correctness and partial accuracy.

To highlight architectural differences, we adopt an undertrained regime where each biography is exposed only 100 times during pretraining.4 The dataset includes N = 50K to 2M biographies, encoding 2 × 106 to 108 total bits of information. Models of varying sizes are tested, and results are visualized via “bit vs. model size” plots. Additional details are provided in Appendix A.3.

Task Mano: Knowledge manipulation. Task Mano evaluates a distinct form of reasoning: the ability to manipulate stored knowledge internally, contrasting with in-context reasoning tasks like Depo or Brevo. While those tasks focus on reasoning over external tokens, Mano requires models to retrieve factual knowledge embedded in their parameters and perform hierarchical computation entirely mentally. This combination of retrieval and reasoning makes knowledge manipulation uniquely challenging and a skill that must be learned during pretraining.5

To test this capability, Mano employs synthetic modular arithmetic expressions inspired by human mental computation, particularly small-number arithmetic like the 9×9 multiplication table. Models solve multi-step arithmetic problems without intermediate steps like Chain-of-Thought. For example, given: <bos> + * a b - c d <ans> the task requires evaluating ((a×b)+(c−d)) mod 23 for ℓ = 3, where operands a,b,c,d are sampled uniformly from [0,22]. Modular arithmetic provides the foundational factual knowledge (23×23 operation tables), while the task challenges hierarchical reasoning by recursively composing operations. Additional details are provided in Appendix A.4.

The dataset is parameterized by a maximum expression length L, with ℓ sampled uniformly from [1,L]. We prepare three Mano datasets across difficulty levels: L = 10, 13, and 16.

Task Lano: Hierarchical language structure. Task Lano evaluates structural reasoning over hierarchical relationships and long-range dependencies. Unlike Depo, Brevo, and Mano, which rely on explicit key-value pairs (in-context or knowledge), Lano challenges models to infer implicit recursive structures across sequences and resolve global ambiguities within them.

To test this, Lano leverages synthetic datasets built from context-free grammars (CFGs). Training sequences consist of CFG-valid sentences separated by <bos> tokens. For example:

- 4Exposing each biography 1000 times during pretraining diminishes architectural differences, as even transformers without MLP layers can achieve similar storage efficiency [8]. Uniform exposure ensures clean systematic comparisons while avoiding confounding effects tied to rare outliers and junk data [8].
- 5For instance, questions like “Was [name] born in an even or odd month?” or derived 2-hop queries such as “What is [name]’s sister’s birthdate?” demand reasoning layers over stored knowledge. These skills cannot reliably emerge through supervised fine-tuning alone [7] and require development during pretraining or continued pretraining.

<bos> 3 3 2 2 1 ... 3 3 1 2 <bos> 1 2 3 3 1 ... 1 2 2 1 <bos> ...

CFGs are designed with token-level ambiguity, where local tokens (e.g., 1, 2, 3) provide insufficient information to directly infer their mapping to CFG rules. Resolving this requires dynamic programming to globally map the entire sequence to a valid recursive application of CFG rules, which must also be learned during training. This reasoning grows in worst-case complexity (O(n3)) as sequence lengths increase. Details are in Appendix A.5.

Building upon cfg3f [6], which includes sequences of lengths 100–500, we introduce extended datasets cfg3j and cfg3k, with sequences ranging up to 200–1000 tokens to increase recursive depth and test models on more nested rules and longer dependencies. Training uses context lengths of 1536 for cfg3j and cfg3k, compared to 512 for cfg3f. Evaluation prompts models with <bos> to generate CFG-valid sentences, validated via a dynamic programming parser. KL divergence is also used to compare token distributions against ground truth.

In summary. This set of five synthetic tasks covers non-overlapping skills and distinct aspects of accuracy—token-level (Depo, Mano), generative (Brevo, Lano), and distributional (Capo, Lano). While this pool can be further enriched, it serves as a strong starting point for deriving meaningful architectural insights, as demonstrated in the following sections.

#### 3 Initial Comparison on Well-Known Base Architectures

Language model architectures have evolved significantly since Transformers [67], giving rise to three major families distinguished by their computational mechanisms.

Quadratic-time attention models include BERT [36] and GPT-2 [48]. Refinements such as Rotary Position Embeddings (RoPE) [13, 61] and gated MLPs [56] define their modern variants. We use the Huggingface implementation of Llama, denoted Llama(RoPE), which includes both refinements, and Llama(NoPE), which omits positional embeddings. When clear, we refer to them as RoPE and NoPE. Relative positional embeddings (e.g., [28]) are omitted due to limited empirical benefit but added computational cost [6].

RoPE models often generalize poorly beyond training context lengths, whereas NoPE generalizes better but achieves lower overall performance. Recent attention-score variants such as ALiBi [45] and Hard-ALiBi [31] partially mitigate this, and we shall investigate closely in this paper.

Linear-time attention reduces computation by compressing sequences into fixed-length representations. Notable architectures include Linformer [68], Performer [15], Linear Transformer [35]. We focus on more recent Gated Linear Attention (GLA) [72] for its efficiency and scalability.

Recurrent and state-space models (SSM) process long sequences via evolving hidden states rather than full attention. Mamba [19, 26] exemplifies this family; we study its 2nd generation (Mamba2). Another key model is Gated DeltaNet (GDN) [73], which we also analyze. Other notable variants include S4 [58], S5 [58], RetNet [62], RWKV [43], HGRN [46], GSA [80], and DeltaNet [74].

Exclusion of hybrid architectures. We omit hybrid models integrating attention with linear or state-space mechanisms—e.g., Griffin [20], Samba [50], GDN-H1/H2 [73], or sliding-window attention—to preserve clarity. Although such hybrids may excel on long contexts (up to 1M tokens), our focus is precision within standard windows (e.g., 4096 tokens). In practice, long contexts are often compressed (e.g., via CoTs) for detailed reasoning, making local precision the key concern.

Hybrids can obscure architectural trade-offs, as aggregated results blur the contributions of individual modules. For instance, Mamba2 performs well on memory tasks but underperforms on structured reasoning; hybridization may conceal such contrasts. To ensure transparency, we analyze isolated base architectures to reveal their intrinsic strengths and weaknesses.

Task Depo1(K=4, k=4/2) Llama(RoPE) - original

###### Task Depo1(K=4, k=4/2) GLA - original

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - original (conv1d)

###### Task Depo1(K=4, k=4/2) GDN - original (conv1d)

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- 97/100% 99/100% 96/100% 93/100%
- 98/100% 84/99% 43/99% 95/100% 79/98% 99/100% 1/24% 3/27%

7/46% 2/14% 14/55% 19/62% 1/11% 1/9% 1/22% 4/31%

12/65% 29/67% 24/74% 43/84% 4/35% 12/46% 13/62% 13/61% 1/22% 13/56% 7/33% 10/42%

85/97% 85/95% 22/86% 32/92% 47/90% 22/85% 61/93% 57/89% 28/78% 32/87% 11/51% 26/71%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/4% 0/0% 0/6% 1/16%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Depo2(K=4, k=4/2) Llama(RoPE) - original

###### Task Depo2(K=4, k=4/2) GLA - original

###### Task Depo2(K=4, k=4/2) Mamba2(mlp) - original (conv1d)

Task Depo2(K=4, k=4/2) GDN - original (conv1d)

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

99/100% 100/100% 99/100% 100/100% 99/100% 100/100%100/100%100/100% 97/100% 100/100%100/100%100/100%

3/22% 1/5% 5/39% 3/12% 6/34% 1/13% 3/25% 1/6%

18/61% 80/95% 30/86% 69/89% 17/63% 47/83% 17/55% 42/82%

94/99% 90/99% 96/99% 99/100% 87/98% 79/95% 94/98% 96/99% 69/95% 86/96% 86/97% 84/97%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

1/1% 1/3% 2/26% 4/18%

5/39% 46/85% 10/41% 24/75%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Llama(RoPE) - original

###### Task Brevo1 GLA - original

###### Task Brevo1 Mamba2(mlp) - original (conv1d)

Task Brevo1 GDN - original (conv1d)

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

45.6% 76.9% 79.8% 88.5% 32.6% 64.5% 44.5% 63.1%

33.7% 36.5% 46.1% 42.2% 1.7% 2.8% 6.2% 11.9% 1.2% 10.7% 2.9% 15.2%

3.7% 80.1% 50.1% 72.4% 0.3% 0.5% 3.8% 4.8% 0.1% 0.0% 1.1% 1.2%

92.5% 94.9% 96.2% 96.7% 78.2% 90.1% 91.7% 87.1% 63.8% 79.3% 90.6% 88.0%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

8.0% 31.2% 17.7% 27.5%

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 Llama(RoPE) - original

###### Task Brevo2 GLA - original

###### Task Brevo2 Mamba2(mlp) - original (conv1d)

Task Brevo2 GDN - original (conv1d)

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

69.3% 89.8% 83.7% 96.0% 40.3% 79.5% 60.5% 88.0% 22.4% 68.2% 40.2% 81.4%

2.8% 45.5% 21.5% 33.2% 0.7% 1.0% 1.8% 8.8% 0.1% 0.7% 1.0% 1.6%

50.8% 95.6% 68.1% 3.4% 12.5% 67.0% 14.5% 0.5%

97.3% 98.8% 98.6% 98.7% 92.7% 96.3% 96.1% 96.8% 73.1% 92.7% 87.4% 89.1%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

3.3% 12.4% 4.0% 0.5%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Capo - Llama(RoPE) - original

###### Task Capo - GLA - original

###### Task Capo - Mamba2(mlp) - original(conv1d)

Task Capo - GDN - original(conv1d)

|5-3 4-3<br><br>6-3<br><br>5-4 3-6 2-8 5-66-6<br><br>10-6 2-8 5-6<br><br>2-3<br><br>5-4<br><br>8-2<br><br>3-6<br><br>8-6 12-6 8-8<br><br>4-2<br><br>6-2<br><br>6-6<br><br>4-35-3<br><br>10-6<br><br>6-3<br><br>6-6<br><br>12-6 10-8<br><br>2-20<br><br>8-6 16-8<br><br>3-6<br><br>8-8<br><br>2-8<br><br>5-6<br><br>5-4<br><br>5-6 2-8<br><br>3-6<br><br>3-20<br><br>10-8<br><br>2-20<br><br>16-8<br><br>10-6<br><br>8-16 8-8<br><br>6-20<br><br>4-20<br><br>3-20<br><br>8-6<br><br>10-6<br><br>4-20<br><br>6-6<br><br>12-6<br><br>8-16<br><br>2-20<br><br>6-20 6-24<br><br>12-16 8-2420-16<br><br>24-1612-24<br><br>8-8<br><br>12-6<br><br>10-8<br><br>20-2024-20<br><br>2-20<br><br>12-16<br><br>16-8<br><br>6-24 4-20<br><br>8-16<br><br>6-20<br><br>8-24<br><br>3-20<br><br>20-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|5-3<br><br>6-3 5-4 3-6 2-8 5-66-6 4-3<br><br>4-2 2-3<br><br>5-4 6-6 6-3<br><br>2-8 5-6 8-6 8-8<br><br>3-6<br><br>10-612-6<br><br>6-2<br><br>8-2<br><br>8-610-612-6 5-6<br><br>6-6 8-810-8 16-8<br><br>4-3<br><br>5-4<br><br>5-3<br><br>6-3<br><br>3-6 2-8<br><br>2-20 3-6 2-8<br><br>8-8<br><br>5-6<br><br>12-6<br><br>10-8 2-20 16-83-20 4-20 8-166-20<br><br>10-6 8-6<br><br>6-6<br><br>16-8 8-8 10-8<br><br>3-20 4-20 8-166-2012-166-24 8-2420-16<br><br>12-6<br><br>10-8 8-8<br><br>16-8<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2-20<br><br>3-20<br><br>4-20 8-166-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-3 5-4 3-6 6-6<br><br>8-2<br><br>5-3 2-8 5-6<br><br>4-2<br><br>5-4<br><br>8-2 6-2<br><br>3-6 2-8 5-66-6 8-610-612-6<br><br>2-3<br><br>8-8<br><br>5-3<br><br>4-3<br><br>5-4<br><br>8-6<br><br>3-6<br><br>6-6<br><br>6-3<br><br>5-6 10-612-68-8 10-8 2-20<br><br>5-3<br><br>16-8<br><br>8-8 10-8 3-20<br><br>2-8<br><br>5-6<br><br>16-8 6-20<br><br>3-6<br><br>10-612-6 2-20 4-20 8-16<br><br>16-8<br><br>10-6<br><br>8-16<br><br>8-6<br><br>6-6<br><br>10-8 3-20 4-20 6-2012-166-24 8-2420-16 12-6<br><br>10-8 8-8 2-20<br><br>16-8<br><br>4-20 8-166-20 3-20<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|8-2 6-2<br><br>8-2 4-35-36-3 5-4 3-6 2-8 5-66-6 6-2<br><br>4-2<br><br>2-3<br><br>6-3<br><br>5-4 3-6 2-8 5-66-6 8-610-612-68-8 4-3<br><br>5-3<br><br>6-3<br><br>5-4<br><br>3-6<br><br>2-8<br><br>2-8 5-66-6 8-610-612-68-810-8 2-2016-8<br><br>3-6<br><br>5-6<br><br>8-6<br><br>10-612-6 8-810-8 2-2016-83-20 4-20 8-166-20 6-6<br><br>8-6<br><br>10-6<br><br>3-20 12-6<br><br>2-2016-8<br><br>8-8 10-8<br><br>4-20 8-166-2012-166-24 8-2420-16<br><br>12-16 24-1612-2420-2024-20 3-20<br><br>8-166-20 6-24 8-2420-16 16-8<br><br>4-20<br><br>12-6<br><br>8-8<br><br>10-8<br><br>2-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

106 107 108

106 107 108

106 107 108

106 107 108

model size (#params)

model size (#params)

model size (#params)

model size (#params)

###### Task Mano Llama(RoPE) - original

###### Task Mano GLA - original

Task Mano Mamba2(mlp) - original (conv1d)

Task Mano GDN - original (conv1d)

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

59.4% 75.5% 84.5% 85.2% 55.6% 53.8% 52.5% 46.5% 26.3% 19.7% 20.9% 41.6%

59.8% 56.0% 76.2% 56.1% 36.5% 31.1% 69.5% 44.4% 16.8% 24.7% 24.8% 22.4%

96.5% 95.1% 95.2% 95.7% 79.9% 84.8% 88.0% 91.8% 74.4% 90.1% 72.3% 87.4%

93.6% 97.9% 91.6% 89.8% 90.0% 98.4% 85.0% 75.0% 81.2% 63.6% 55.2% 52.6%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - original

###### Task Lano GLA - original

Task Lano Mamba2(mlp) - original (conv1d)

Task Lano GDN - original (conv1d)

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

91.1% 96.3% 93.4% 97.6% 74.1% 91.4% 82.3% 90.3% 64.0% 75.1% 60.0% 79.1%

37.5% 61.1% 41.8% 54.9%

83.8% 92.2% 86.8% 92.2% 45.5% 72.0% 54.2% 74.3% 32.7% 50.0% 35.3% 46.1%

83.4% 93.8% 91.5% 94.4% 54.9% 80.3% 70.9% 86.5% 38.0% 63.4% 44.6% 65.7%

cfg3f

cfg3f

cfg3f

cfg3f

2.7% 17.1% 9.1% 35.7% 13.0% 11.6% 12.9% 19.8%

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

- Figure 4: Initial comparison of base models on five synthetic tasks. GLA performs weakest; Mamba2(mlp) excels in knowledge (Capo, Mano); GDN strengthens reasoning and surpasses Llama(RoPE) on Brevo (reasoning breadth), while RoPE remains best on Depo+Lano (depth and structural reasoning). These results confirm our synthetic playground as effective for architectural comparison, but adding Canon layers (see later) will build a “Pisa tower”—enabling controlled, fair comparisons where the landscape shifts drastically and reasoning depth improves 2–4×.

Notably, Falcon-H1 [63] (May 2025, 32B) combines Mamba2 with full attention, while Qwen3Next [47] (Sep 2025, 80B) combines GDN with full attention. These recent hybrids validate our choice of Mamba2 and GDN as representative base linear models. Architecture size standardization. To ensure fair comparison, we standardize model sizes and evaluate Llama, GLA, Mamba2, and GDN as representatives of their respective families.

For all tasks except Capo, we test four sizes: Llama models with 12 or 8 layers and hidden dimensions of 768 or 512 (12 or 8 heads), denoted 12L768D, 12L512D, 8L768D, and 8L512D. (12L768D matches GPT-2-small.) These configurations are translated to GLA, Mamba2(mlp), and GDN for comparable parameter counts.6

For Capo (bit-per-parameter knowledge capacity), we vary model and data scales more broadly.

6See Appendix C for details. Briefly, with hidden size d, GLA follows the 4d2 + 8d2 design (linear attention 4d2, MLP 8d2), while Mamba2(mlp) and GDN use 6d2 + 6d2. We also test Mamba2 without MLP, reported separately in the appendix and referred to as Mamba2.

Following [8], model size is denoted ℓ-h: for Llama, ℓ layers, hidden size 64h, and h heads. This notation extends consistently to GLA, Mamba2, and GDN (see Appendix C).

Training. All architectures share identical training settings (batch size, steps, learning rate, etc.) to ensure fairness. Full details appear in Appendix A. Random seeds are fixed so that all models pre-train on identical data sequences.

##### 3.1 Initial Comparison Results

From Figure 4, linear-attention GLA performs weakest overall. Mamba2 excels on knowledge tasks (Capo, Mano) but lags in reasoning. GDN improves Mamba2’s reasoning and occasionally surpasses Llama(RoPE) on certain reasoning tasks (e.g., Brevo), though not others. These patterns align with real-world observations on natural data, supporting the validity of our synthetic playground. We defer deeper interpretation, as both Llama and GLA later prove to lack a critical architectural component—making this comparison incomplete and partially unfair.

For now, we highlight several key remarks.

3×4 mini scaling laws. Randomness can affect outcomes, especially on hard tasks where grokking emerges. In Mano, even with two seeds and four learning rates, smaller models sometimes outperform larger ones. This reflects staged reasoning: a model must learn k-hop reasoning (e.g., Mano, Depo) before advancing to k+1, and the transition often depends on random training dynamics. To reduce such variance, we test all tasks across three data scales and four model sizes (more for Capo). These “3×4” mini scaling laws yield more stable and interpretable comparisons.

Benefits of synthetic tasks. Synthetic tasks clarify architectural differences starkly (e.g., 90% vs 5%), clearly exposing strengths and weaknesses. By contrast, real-world experiments often produce modest differences (e.g., 2%) buried in noise. Thus, synthetic pretraining environments allow clean evaluations of architectures’ scalability and true capabilities.

Interpreting task failures. If a specific architecture (of a given size) fails at a certain difficulty level (e.g., large N or k), it does not imply the model cannot learn the skill given infinite training. Our comparison uses a fixed, limited training budget: all architectures train for the same number of steps with identical data and shuffling, reporting best accuracy across multiple learning rates. Thus, results should be seen as differences in the speed of skill acquisition, not absolute capability.7 Predicting future pipelines. Synthetic tasks simulate idealized, high-quality pretraining conditions targeting core skills like multi-hop reasoning (Depo). Unlike datasets such as FineWeb-edu or SlimPajama, which contain sparse reasoning examples obscured by simpler content, synthetic tasks highlight core capabilities. Currently, 100B-token pretraining fails even simplest 2-hop reasoning (Result 12). As training pipelines evolve—via improved data curation or RL-based post-trainingsynthetic tasks like Depo may better predict models’ potential and guide architectural choices.

#### 4 Canon Layers: Enhancing Horizontal Information Flow

Attention-based Transformers are widely recognized for their ability to perform associative recall—e.g., predicting ? in the sequence [A] [B] ... [A] [?] where ? = [B]. One might expect the second [A] could simply attend to the first to retrieve [B], but causal masking makes this impossible: the first occurrence of [A] sees no future tokens. Accurate recall thus “requires” two

7Faster learning is practically important—for example, a model ideally learns reasoning skills quicker than pure memorization. Similar observations arise in knowledge capacity tasks [8], where architectural differences vanish with ample training but become pronounced when training budgets are limited.

||. A B . . . . . . A ?|
|---|
<br><br>How to predict ?=B store A to B<br><br>attending to A is useless<br><br>(B is not there)<br><br>[Figure 39]<br><br>(first attention) (second attention)<br><br>use key=A to retrieve value=B<br><br>a folklore: associative recall “needs” two layers of attention<br><br>|
|---|

62% 100%100%100%100%100%100%100%100%100%100%100%100%100%

[Figure 40]

- RoPE(1L-2H-16D)

- RoPE(1L-4H-32D)

- RoPE(1L-8H-64D)

- RoPE(1L-16H-128D)

- RoPE(2L-2H-16D)

- RoPE(2L-4H-32D)

- RoPE(2L-8H-64D)

- RoPE(2L-16H-128D)

0% 91% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 91% 100%100%100%100%100%100%100%100%100%100%100%100%

- 0% 13% 96% 99% 100%100%100%100%100%100%100%100%100%100%

0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0%

0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0%

0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0%

0% 0% 0% 0% 100%100%100%100%100%100%100%100%100%100%

- 0% 14% 99% 100%100%100%100%100%100%100%100%100%100%100%

Canon(cst) - RoPE(1L-2H-16D)

0% 59% 100%100%100%100%100%100%100%100%100%100%100%100%

Canon - RoPE(1L-2H-16D)

5001000150025005000100001500020000250003000035000400004500050000

training steps

first attention only serves the purpose of “read your neighbor” most language models are missing a horizontal “residual link”

- Figure 5: A trivial token-copying experiment for 500 tokens, added for completeness. 1-layer RoPE requires d ≥ 128, while 2-layer RoPE or 1-layer RoPE + Canon achieves 100% with d = 16.

attention layers—the first copies the first [A] into its neighbor [B]; the second uses this enriched representation, querying by [A] to retrieve value = [B] (via key = [A]). Using global attention just to pass information between adjacent tokens is, in effect, shooting a bird with a cannon.

Remark 4.1. This is not a strict lower bound. A 1-layer Transformer is Turing-complete and can perform recall by blindly aggregating most (or all) context into one position, allowing MLP to do local query/key/value computations. But this is inefficient: Figure 5 shows that 1-layer Transformer needs hidden size 128 to recall length-500 sequences, while 2 layers succeed with size 16.

The importance of local context. Even simple tasks like token recall require careful mixing of local context—not to say more complex ones or when words span multiple tokens. Since MLP layers don’t mix tokens, attention must handle all communication. Rotary and relative positional encodings help by biasing attention toward nearby tokens, but they remain tied to attention and still “shoot birds with cannons.” Similar issues arise in GLA [72] and Mamba2, where recent-token information must be retrieved via compression mechanisms not optimized for local detail.

Canon layers: general form. Inspired by (vertical) residual connections, we introduce Canon layers to enhance horizontal information flow across neighboring tokens. Canon layers aggregate nearby hidden states into the current position, enabling lightweight local mixing within a fixed window (e.g., size 4), unlike attention-based global aggregation or recurrent compression.

Formally, for any hidden states ht ∈ Rm at token position t, a Canon layer computes: h′t = w0 ⊙ ht + w1 ⊙ ht−1 + w2 ⊙ ht−2 + w3 ⊙ ht−3,

where ⊙ denotes element-wise multiplication, wi ∈ Rm (i = 0,1,2,3) are weights, and padding zeros are used for boundary conditions. We call this Canon, borrowing from the musical term, as it resembles melodies played sequentially at fixed temporal delays.8

Flexible Integration. Canon layers integrate at multiple points within each Transformer block:

- • Canon-A: Before the attention block (m = d if hidden size is d), after RMSnorm.
- • Canon-B: Inside the attention block, applied after Q/K/V projections (m = 3d).
- • Canon-C: Before the MLP block (m = d), after RMSnorm.
- • Canon-D: Within MLP (m = 4d for standard, m = 163 d for gated MLP), before activation.

Combining all four points gives Canon-ABCD (full-score Canon); partial combinations (CanonA/B/ABC) can also be explored. Canon layers integrate flexibly across diverse architectures, including linear-attention and state-space models. For Mamba2 (without standard MLP layers), Canon layers appear at Canon-A and Canon-B positions (yielding Canon-AB); for Mamba2(mlp), the complete Canon-ABCD applies. Canon-B in Mamba2 scales as m = 4d + o(d).9

- 8In Pachelbel’s Canon in D, violins sequentially play the same melody with delays, creating overlapping horizontal repetition patterns analogous to Canon layers.
- 9For example, Mamba2 settings with ssm state size=64, num heads=16 result in m = 4d + 144 dimensions.

Canon Layers

delayed repetition, like in music

Explicitly add “horizontal residual” connections:

|…|
|---|

ℎ𝑡′ = 𝑤0 ⊙ ℎ𝑡 + 𝑤1 ⊙ ℎ𝑡−1 + 𝑤2 ⊙ ℎ𝑡−2 + 𝑤3 ⊙ ℎ𝑡−3 ∈ ℝ𝑚

[Figure 41]

|…|
|---|

|❑ random fixed 𝑤0,𝑤1, 𝑤2,𝑤3 ∈ ℝ𝑚 already highly effective|
|---|

❑ trainable 𝑤0, 𝑤1,𝑤2,𝑤3: ℎ′ = 𝒉 + causal_conv1d𝑤(ℎ) – used in this paper

shift=1, shift=2,…

- point A
- point B
- point C
- point D

[Figure 42]

can be: attention, linear-attn, SSM, MLP...

residual link “ℎ +” accelerates training

[Figure 43]

[Figure 44]

minimal overhead + flexible integration

𝑤1 𝑤2

❑ more complex forms (e.g., dynamic conv with 𝑤 depends on ℎ), possible but less efficient, not explored in this paper for clarity.

can add Canon anywhere (e.g., A/B/C/D sub-layers)

Figure 6: Illustration of Canon layers.

Canon layers: Implementation variants. Canon layers can be implemented in many ways. Even a simple version with fixed, random weights—aggregating past three tokens as horizontal residual links—already notably enhances performance (Figure 24 on Page 46).10 More complex variants—e.g., dynamic convolutions with input-dependent weighting—are possible but not studied here, as it remains unclear whether such additional cost is justified.

In this paper, for simplicity and efficiency, we implement Canon layers as 1-d causal convolution with kernel size 4, available through efficient CUDA kernels implemented by the open-source H3 library (pip package causal conv1d) [23]. We also incorporate explicit residual connections:

h′t = ht + conv1d [ht,ht−1,ht−2,ht−3] , (4.1) denoted as Canon(res). Without residual connections, we denote it Canon(no-res). Minimal code changes (just a few lines) are needed for integration. Even fully enabled (Canon-ABCD), Canon layers increase the parameter count minimally.11 Our emphasis is on clearly demonstrating Canon layers’ substantial performance benefits; detailed runtime optimizations remain future work.

Related Work. A precursor to Canon layers appears in [6], which studied uniform attention—i.e., averaging the past k tokens—for k ∈ {1,2,4,8,...} on CFG tasks. Surprisingly, this simple mixing outperformed GPT2 with absolute positional embeddings and closely approached GPT2(RoPE).12 Canon layers generalize this idea: we apply learned, position-specific mixing over a short window (typically 4 tokens), removing value and projection matrices for better efficiency and modularity.

Our use of causal conv1d is inspired by Mamba [19, 26] and GLA [72], which trace back to H3 [23], where the component was introduced as “shift-SSM.” After the initial release of our paper, we also became aware of Primer [59], which proposes “multi-dconv-head” attention. These models apply conv1d (often with SiLU activation) within SSM or attention modules, without residual connections. In our terminology, these roughly correspond to Canon-B(no-res).

Our work generalizes and isolates this design as the Canon layer, and systematically evaluates its effect across all types of sequential models and sublayers (A/B/C/D). By studying Canon under controlled synthetic pretraining, we can clearly attribute performance gains to the conv1d-

- 10Unlike vertical residual links (h′ = h + σ(Wh)), Canon layers aggregate multiple token vectors from different relative positions (t−1, t−2, t−3). Assigning fixed orthogonal directions effectively provides each position a unique “ID” for aggregation. Simple scalar weighting (e.g., h′t = ht +0.4ht−1 +0.2ht−2 +0.1ht−3) can degrade performance.
- 11Fewer than 0.45% parameters for GPT2-small. For a 1.3B-parameter Llama with Canon-ABCD enabled, parameters increase by 0.0063%, runtime overhead on an H100 GPU with naive implementation (PyTorch bf16, flash attention, causal conv1d kernels) is 12.4%, 14.1%, and 20.8% for forward, backward, and generation respectively. For Canon-AC, overheads reduce to 5.8%, 5.8%, and 7.0%. Further runtime efficiencies are possible (e.g., consolidating multiple Canon operations across layers), though these optimizations remain beyond this paper’s scope.
- 12One ICML reviewer rejected the paper, commenting that the results were “too surprising to be true.” We invite curious readers to try it themselves—it really works.

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - original

Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) Llama(NoPE) - original

Task Depo1(K=8, k=8/4) Llama(NoPE) - Canon-ABCD(res)

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

0/34% 1/50% 1/4% 0/1% 0/27% 0/0% 0/12% 0/0%

97/100% 92/100% 73/89% 94/100% 57/97% 54/93% 92/99% 99/99% 76/99% 53/99% 16/66% 97/100%

99/100% 97/100% 99/100% 100/100% 98/100% 92/99% 95/100% 95/100%

0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0%

99/100% 99/100% 99/100% 100/100% 96/99% 99/100% 99/100% 99/100% 99/100% 99/100% 98/100% 99/100%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/2% 0/56% 0/0% 0/0%

75/99% 97/100% 85/100% 90/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - original

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) Llama(NoPE) - original

Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-ABCD(res)

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

2/1% 2/1% 1/1% 30/99%

91/99% 97/100% 98/100% 99/100% 98/100% 98/100% 99/100% 98/100%

92/100% 100/100% 97/100% 99/100% 97/100% 99/100% 96/100% 97/100% 85/100% 99/100% 98/100% 98/100%

0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0%

96/100% 85/99% 86/100% 99/100% 94/100% 86/99% 99/100% 99/100% 90/100% 98/100% 93/100% 96/100%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/90% 1/3% 21/96%
- 1/2% 1/92% 1/3% 1/50%

71/98% 90/100% 94/99% 96/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Llama(RoPE) - original

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

###### Task Brevo1 Llama(NoPE) - original

Task Brevo1 Llama(NoPE) - Canon-ABCD(res)

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

45.6% 76.9% 79.8% 88.5% 32.6% 64.5% 44.5% 63.1%

84.6% 88.7% 88.3% 91.3% 51.3% 72.4% 69.9% 75.7% 24.8% 49.1% 41.2% 58.8%

83.7% 93.8% 91.3% 96.5% 62.9% 84.5% 81.2% 90.7% 47.9% 82.2% 69.7% 84.5%

0.2% 0.0% 0.0% 0.4% 0.1% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.1%

84.8% 94.4% 91.1% 96.2% 63.9% 85.8% 75.5% 92.2% 42.0% 75.3% 58.2% 84.9%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

8.0% 31.2% 17.7% 27.5%

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 Llama(RoPE) - original

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

###### Task Brevo2 Llama(NoPE) - original

Task Brevo2 Llama(NoPE) - Canon-ABCD(res)

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

69.3% 89.8% 83.7% 96.0% 40.3% 79.5% 60.5% 88.0% 22.4% 68.2% 40.2% 81.4%

87.5% 94.5% 92.3% 95.4% 66.0% 85.3% 79.3% 90.5% 44.6% 75.5% 68.5% 87.8%

87.1% 95.6% 92.2% 97.1% 75.4% 87.7% 80.1% 93.5% 55.1% 82.5% 69.3% 88.1%

0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0%

87.4% 93.2% 89.0% 96.1% 61.2% 84.0% 75.2% 91.7% 40.4% 56.0% 56.3% 79.9%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Capo - Llama(RoPE) - original

###### Task Capo - Llama(RoPE) - Canon-ABCD(res)

###### Task Capo - Llama(RoPE) - Canon-ABCD(res)

###### Task Capo - Llama(NoPE) - original

Task Capo - Llama(NoPE) - Canon-ABCD(res)

|5-3 4-3<br><br>6-3<br><br>5-4 3-6 2-8 5-66-6<br><br>10-6 2-8 5-6<br><br>2-3<br><br>5-4<br><br>8-2<br><br>3-6<br><br>8-6 12-6 8-8<br><br>4-2<br><br>6-2<br><br>6-6<br><br>4-35-3<br><br>10-6<br><br>6-3<br><br>6-6<br><br>12-6 10-8<br><br>2-20<br><br>8-6 16-8<br><br>3-6<br><br>8-8<br><br>2-8<br><br>5-6<br><br>5-4<br><br>5-6 2-8<br><br>3-6<br><br>3-20<br><br>10-8<br><br>2-20<br><br>16-8<br><br>10-6<br><br>8-16 8-8<br><br>6-20<br><br>4-20<br><br>3-20<br><br>8-6<br><br>10-6<br><br>4-20<br><br>6-6<br><br>12-6<br><br>8-16<br><br>2-20<br><br>6-20 6-24<br><br>12-16 8-2420-16<br><br>24-1612-24<br><br>8-8<br><br>12-6<br><br>10-8<br><br>20-2024-20<br><br>2-20<br><br>12-16<br><br>16-8<br><br>6-24 4-20<br><br>8-16<br><br>6-20<br><br>8-24<br><br>3-20<br><br>20-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|2-3<br><br>6-3 5-4 3-6 2-8 5-66-6<br><br>2-3<br><br>4-2<br><br>5-3 4-3<br><br>8-2<br><br>3-6 12-6<br><br>6-2<br><br>2-8 5-66-6 8-610-6 8-8<br><br>4-3<br><br>10-6 5-6<br><br>5-4 6-3<br><br>3-6<br><br>5-3<br><br>6-6 8-6 12-6 8-8 10-8 2-2016-8 3-6<br><br>2-20<br><br>2-8<br><br>5-6<br><br>8-8 10-8 16-83-20 4-20 8-166-20<br><br>8-6 6-6<br><br>10-6<br><br>4-20 2-20<br><br>16-8<br><br>3-20 8-166-2012-166-24 8-2420-16<br><br>8-24<br><br>12-6 8-8<br><br>20-16<br><br>10-8<br><br>12-166-24 24-1612-2420-2024-20<br><br>2-20<br><br>16-8<br><br>4-20 3-20<br><br>8-166-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-6<br><br>4-2<br><br>6-3<br><br>5-4 3-6 5-6<br><br>2-3<br><br>2-8<br><br>4-2<br><br>12-6<br><br>8-2<br><br>8-610-6<br><br>2-3<br><br>3-6<br><br>4-3<br><br>5-4<br><br>6-2<br><br>2-8 6-6<br><br>5-3<br><br>5-6 8-8<br><br>5-3 4-3<br><br>3-6<br><br>6-3<br><br>12-6 10-8 6-6<br><br>8-6 8-8 5-6<br><br>10-6<br><br>2-8<br><br>5-4<br><br>2-2016-8<br><br>4-20<br><br>3-6<br><br>5-6<br><br>2-8<br><br>10-8 2-2016-83-20 8-166-20 10-6<br><br>8-6<br><br>8-16<br><br>6-6<br><br>4-20 6-2012-166-24 8-2420-16<br><br>10-8<br><br>2-20<br><br>8-16<br><br>3-20<br><br>8-8 12-6<br><br>4-20<br><br>16-8<br><br>6-20<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-6<br><br>4-2 2-3<br><br>5-4<br><br>3-6<br><br>2-8<br><br>5-6<br><br>6-2<br><br>6-2<br><br>2-3<br><br>4-2<br><br>2-8<br><br>6-6<br><br>3-6<br><br>5-6<br><br>8-8<br><br>8-2<br><br>8-610-612-6<br><br>5-3<br><br>5-6<br><br>6-3<br><br>4-3<br><br>2-20<br><br>16-8<br><br>5-4<br><br>8-6<br><br>2-8<br><br>12-6 8-8<br><br>10-8<br><br>6-6<br><br>10-6 5-6<br><br>2-8<br><br>10-8 8-8<br><br>10-6<br><br>3-6<br><br>2-20<br><br>16-8<br><br>4-20<br><br>8-166-20<br><br>8-6<br><br>10-6<br><br>8-24<br><br>6-20 6-24<br><br>3-20<br><br>4-20<br><br>8-16<br><br>20-16<br><br>6-6 2-20<br><br>10-8<br><br>12-6<br><br>16-8<br><br>8-8<br><br>4-20<br><br>3-20<br><br>8-16<br><br>6-20<br><br>12-24<br><br>12-16<br><br>6-24 8-24 24-16 20-20<br><br>20-16<br><br>24-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|4-2<br><br>4-3<br><br>5-36-3 5-4<br><br>3-6 2-8 5-6<br><br>8-2<br><br>6-6<br><br>2-3<br><br>4-2<br><br>6-2<br><br>5-6 5-4<br><br>3-6 12-6 8-8 2-8<br><br>4-3<br><br>8-2<br><br>6-6 8-610-6<br><br>8-8 5-6<br><br>6-3<br><br>4-3<br><br>12-6 10-8 2-20<br><br>2-8<br><br>5-3<br><br>5-4<br><br>6-6 8-610-6 16-8<br><br>12-6<br><br>3-6 2-8<br><br>5-6<br><br>8-8 2-20<br><br>10-8 16-83-20 4-20 8-166-20 10-6<br><br>6-6<br><br>4-20 6-20<br><br>8-6<br><br>8-16 12-166-24 3-20<br><br>8-2420-16<br><br>8-8<br><br>2-20 10-8<br><br>12-6<br><br>4-20<br><br>16-8<br><br>8-16 3-20<br><br>6-2012-166-24 8-2420-1624-1612-2420-2024-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

106 107 108

106 107 108

106 107 108

106 107 108

106 107 108

model size (#params)

model size (#params)

model size (#params)

model size (#params)

model size (#params)

###### Task Mano Llama(RoPE) - original

Task Mano Llama(RoPE) - Canon-ABCD(res)

Task Mano Llama(RoPE) - Canon-ABCD(res)

###### Task Mano Llama(NoPE) - original

Task Mano Llama(NoPE) - Canon-ABCD(res)

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

59.4% 75.5% 84.5% 85.2% 55.6% 53.8% 52.5% 46.5% 26.3% 19.7% 20.9% 41.6%

95.1% 99.3% 99.3% 99.5% 66.0% 94.6% 97.1% 98.8% 63.7% 82.8% 91.4% 83.0%

94.2% 98.0% 99.2% 99.6% 89.8% 88.5% 98.2% 99.2% 83.7% 83.6% 88.8% 85.3%

- 7.1% 7.1% 7.1% 6.9%
- 7.1% 7.1% 7.2% 7.1%
- 7.3% 7.3% 7.4% 7.3%

97.7% 98.9% 99.3% 99.3% 83.1% 90.1% 95.9% 98.1% 53.7% 55.5% 89.4% 94.3%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - original

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

###### Task Lano Llama(NoPE) - original

Task Lano Llama(NoPE) - Canon-ABCD(res)

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

91.1% 96.3% 93.4% 97.6% 74.1% 91.4% 82.3% 90.3% 64.0% 75.1% 60.0% 79.1%

96.6% 98.0% 97.2% 98.3% 88.2% 92.0% 88.6% 94.3% 75.2% 87.1% 83.0% 86.7%

95.2% 97.5% 96.0% 98.1% 81.4% 90.1% 85.9% 92.6% 66.0% 77.9% 76.1% 78.9%

0.0% 0.0% 0.0% 38.8% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0%

87.9% 91.9% 88.5% 92.5% 55.1% 70.3% 58.6% 78.3% 33.5% 51.0% 37.2% 53.1%

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

- Figure 7: Column 1→2: Canon layers dramatically enhance RoPE, improving reasoning depth by 2–4×. Column 4→5: Canon transforms NoPE into a strong performer on par with RoPE-based models. Column 2+5→3: With Canon, RoPE usage can be reduced — RoPE + ˇ “Canon (RoPE enabled for 1/4 dimensions) outperforms both RoPE/NoPE + Canon, great news for length generalization!

Remark. This figure uses Depo1(K=8) and Depo2(K=16). Earlier results in Figure 4 were based on Depo1(K=4) and Depo2(K=4), because model performances were weaker.

based mixing mechanism, rather than to other architectural components such as attention or statespace recurrence. Moreover, we show that Canon layers are intrinsically not tied to attention or SSMs—and in fact, may not benefit from being tightly coupled to them.

Convolutions have been used in Transformers for different goals. Conformer [27] and CvT [70] integrate heavier convolutional modules for feature extraction in speech and vision. In contrast, Canon layers are lightweight and designed to enhance horizontal information flow—like horizontal “residual links.” Notably, even random-weight Canon layers yield substantial improvements.

Concurrent work on Multi-Token Attention (MTA) [25] explores more complex 2D convolutional layers within attention heads. While MTA improves associative recall, it is heavier and more attention-specific. Investigating whether such designs offer further gains when combined with Canon, or whether Canon alone suffices for most settings, is an interesting direction for future work.

#### 5 When Transformer Meets Canon

Figure 4+7 show that a 12-layer, 768-dimension Llama(RoPE) model trained on our ideal data can only handle 4-hop retrieval in contexts of length 2048. Can this be any better?

Task Depo2(K=16) | N=75 | Llama(RoPE) - original

Task Depo2(K=16) | N=125 | Llama(RoPE) - original

Task Depo2(K=16) | N=100 | Llama(RoPE) - original

100

100

100

k=16

k=16

k=16

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 8L512D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 8L512D)

Task Depo2(K=16) | N=75 | Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16) | N=125 | Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16) | N=100 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

k=16

k=16

k=16

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 8L512D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 8L512D)

- Figure 8: Training curves for RoPE models w/+w/o Canon, on Depo2(K = 16), evaluated at k = 1, 2, 4, 8, 16 and maximum size n = N, shown in two best LRs. More model sizes/data are in Figure 19 on Page 41.

##### 5.1 RoPE with Canon Layers

|Result 2 (Figure 7 — 1st vs. 2nd column). In our controlled playground, Canon layers (ABCD) introduce substantial improvements: with a 0.5% increase in trainable parameters, reasoning depth of RoPE increases by 2-4×, reasoning breadth by 30%, knowledge capacity by 10–15%, knowledge manipulation length by 30%, measurable gains in hierarchical language structure reasoning.|
|---|

Task Depo. In reasoning depth, RoPE pretrained on Depo1(K = 8)—covering (k ≤ 8)–hop instances—achieves near-zero accuracy even at k = 4, whereas RoPE+Canon-ABCD exceeds 50% at k = 8. On Depo2(K = 16)—a more challenging setup where each directed edge spans 10–14 tokens, far beyond a 4-token Canon window—RoPE completely fails, while RoPE+Canon-ABCD attains near-perfect accuracy at k = 16. This demonstrates that Canon layers are not merely for single-token recall: by enriching local representations of multi-token segments, they empower the global attention to more effectively chain information across hops.13

These gains may seem surprising. For associative recall (analogous to Depo1 with k = 1), theory suggests a single Canon + attention layer suffices (recall Figure 5), suggesting Canon could reduce required attention layers by at most one. So, why a 2–4× increase in reasoning depth?

The answer lies in learning dynamics. Deep reasoning tasks like Depo unfold through a hierarchical learning process—models first master 1-hop, then gradually progress to 2-hop, 3-hop, and beyond. This process relies heavily on two factors: (1) training data spanning a range of difficulty levels and (2) architectural support like residual connections. Without either—e.g., training only with k = 8 data or removing residuals—the model can fails entirely.14

Thus, architectures that enable faster mastery of 1- and 2-hop reasoning climb the hierarchy faster, as illustrated in Figure 8. RoPE + Canon-ABCD achieves deeper reasoning progression much faster than vanilla RoPE, leveraging the inherent easy-to-hard structure of multi-hop tasks. We emphasize again that this is not about performance under infinite training data—RoPE could eventually achieve similar accuracy on Depo2(K = 16). However, RoPE + Canon achieves comparable results with significantly fewer training steps, making it far more efficient.

Task Brevo. On reasoning breadth, we observe 30% improvement by introducing Canon-ABCD. Specifically, the accuracies of RoPE to solve Brevo1(N = 70) or Brevo2(N = 30) resemble the performance of RoPE+Canon to solve Brevo1(N = 90) or Brevo2(N = 40). Since input length scales with N, this reflects roughly 30% increase in reasoning breadth.

To understand the source of this improvement, we analyze the accuracy across tasks stratified

- 13Depo2 is designed so a 4-token window cannot resolve key–value pairs spanning 10–14 tokens, posing a substantial challenge even for Canon.
- 14The first theory foundation for why deep learning can perform deep (hierarchical) learning was established by Allen-Zhu and Li [3] (in the 3-layer case) and Allen-Zhu and Li [4] (for ω(1)-layer). They show that deep learning relies on easy-to-hard curricula and residual structures for progressively building complexity.

###### Task Brevo1 - Llama(RoPE) - original

46% 51% 47% 44% 43% 43% 77% 85% 79% 75% 74% 74% 80% 86% 79% 79% 78% 77% 88% 91% 89% 87% 88% 87% 33% 45% 36% 30% 26% 22% 64% 74% 69% 61% 57% 55% 45% 56% 46% 43% 38% 39% 63% 71% 66% 64% 56% 53%

[Figure 75]

N=70 N=90

8% 20% 9% 5% 4% 6% 31% 46% 35% 28% 26% 18% 18% 32% 21% 13% 11% 12% 28% 44% 32% 22% 19% 19%

N=110

all acc 8L512D

depth 1 8L512D

depth 2 8L512D

depth 3 8L512D

depth 4 8L512D

depth 5 8L512D

all acc 12L512D

depth 1 12L512D

depth 2 12L512D

depth 3 12L512D

depth 4 12L512D

depth 5 12L512D

all acc 8L768D

depth 1 8L768D

depth 2 8L768D

depth 3 8L768D

depth 4 8L768D

depth 5 8L768D

all acc 12L768D

depth 1 12L768D

depth 2 12L768D

depth 3 12L768D

depth 4 12L768D

depth 5 12L768D

###### Task Brevo1 - Llama(RoPE) - Canon-ABCD(res)

85% 87% 86% 84% 82% 81% 89% 91% 90% 87% 87% 84% 88% 92% 89% 87% 86% 89% 91% 95% 92% 90% 90% 91% 51% 64% 56% 48% 44% 38% 72% 79% 75% 71% 68% 63% 70% 79% 73% 68% 64% 62% 76% 83% 78% 74% 71% 70% 25% 43% 29% 19% 19% 14% 49% 66% 53% 45% 42% 36% 41% 61% 46% 37% 33% 27% 59% 73% 63% 55% 50% 50%

[Figure 76]

N=70 N=90

N=110

all acc 8L512D

depth 1 8L512D

depth 2 8L512D

depth 3 8L512D

depth 4 8L512D

depth 5 8L512D

all acc 12L512D

depth 1 12L512D

depth 2 12L512D

depth 3 12L512D

depth 4 12L512D

depth 5 12L512D

all acc 8L768D

depth 1 8L768D

depth 2 8L768D

depth 3 8L768D

depth 4 8L768D

depth 5 8L768D

all acc 12L768D

depth 1 12L768D

depth 2 12L768D

depth 3 12L768D

depth 4 12L768D

depth 5 12L768D

Task Brevo1 - Llama(RoPE) - Canon-ABCD(res)

84% 86% 85% 82% 82% 83% 94% 95% 94% 94% 95% 94% 91% 94% 92% 91% 91% 89% 97% 98% 97% 96% 97% 97% 63% 72% 65% 62% 61% 56% 84% 89% 85% 84% 83% 80% 81% 86% 82% 80% 79% 80% 91% 93% 91% 90% 89% 93% 48% 66% 51% 45% 41% 35% 82% 88% 84% 82% 79% 75% 70% 78% 72% 67% 65% 65% 84% 90% 86% 85% 82% 83%

[Figure 77]

N=70 N=90

N=110

all acc 8L512D

depth 1 8L512D

depth 2 8L512D

depth 3 8L512D

depth 4 8L512D

depth 5 8L512D

all acc 12L512D

depth 1 12L512D

depth 2 12L512D

depth 3 12L512D

depth 4 12L512D

depth 5 12L512D

all acc 8L768D

depth 1 8L768D

depth 2 8L768D

depth 3 8L768D

depth 4 8L768D

depth 5 8L768D

all acc 12L768D

depth 1 12L768D

depth 2 12L768D

depth 3 12L768D

depth 4 12L768D

depth 5 12L768D

- Figure 9: Detailed accuracies for Task Brevo1, shown overall and stratified by dependency graph depths 1, 2, 3, 4, 5.

by depth of the dependency depth. Recall each query in Brevo requires the model to identify all vertices it recursively depends on, forming a sub-DAG of varying (minimum) depth. In Figure 9, we plot model accuracy not only overall but also separately for problem instaces spanning DAG depths of 1,2,3,4,5. The results show that vanilla RoPE struggles with instances involving greater DAG depth, whereas RoPE+Canon improves reasoning performance on deeper structures. This suggests that Canon-ABCD enhances localized reasoning paths within Transformer blocks, allowing for better handling of recursive dependency, which can be challenging for standard attention alone. Task Capo. On knowledge capacity, prior work [8] found that gated MLP layers in Llama(RoPE) reduce model capacity due to slower and less stable training dynamics. One remedy proposed in that work is to revert gated MLP back to standard MLP; however, this sacrifices reasoning capability (see Section 5.4). Here, we present an alternative solution: adding Canon layers. Canon layers improve training speed and increase the effective capacity by 10–15% in the controlled 100exposure pretraining regime for Capo. On a separate note, GPT2(RoPE) models that originally employ standard MLP exhibit no capacity loss after Canon layers are introduced (Figure 11).

Task Mano. On knowledge manipulation, Canon layers increase manipulable length. RoPE+Canon

matches the performance of vanilla RoPE on Mano(L = 10) when tested on Mano(L = 13), a 30% improvement in length. This again stems from Canon layers accelerating hierarchical learning, enabling the model to scale from simpler tasks (L = 1) to more complex ones (L = 2, L = 3, and beyond) faster. For simplicity, we omit the hierarchical learning speed visualization.

Task Lano. Canon layers improve RoPE’s performance on hierarchical language structure reasoning, though interpreting the gains requires some algorithmic background. For context, dataset cfg3k adds one level of structural complexity above cfg3f, using the same CFG rule distribution (see Appendix A.5). RoPE+Canon outperforms standard RoPE on cfg3k, but still struggles to fully handle this increased complexity. This is expected, as deeper CFG structures increase sequence length n by 2–3×, and parsing these CFGs with dynamic programming involves worst-case time complexity O(n3). Consequently, cfg3k poses arguably more than 8× greater computational challenge compared to cfg3f. Our intermediate dataset cfg3j has difficulty around 4×, suggesting RoPE+Canon can handle roughly twice as challenging structure-learning tasks comparing to RoPE. Summary. Canon layers consistently improve performance across reasoning, knowledge and language tasks, all without introducing instability or accuracy trade-offs.

##### 5.2 NoPE with Canon Layers

|Result 3 (Figure 7+10a). Canon layers transform NoPE. Key findings include:<br><br>• NoPE+Canon matches RoPE+Canon and even surpasses it on Depo; a remarkable result given that without Canon, NoPE achieves essentially zero performance on all measures.<br>• NoPE+Canon significantly outperforms existing fixes for NoPE, such as ALiBi and H-Alibi.<br>• With Canon layers, RoPE usage can be greatly reduced: RoPE on only 1/4 dims (denoted RoPE+ Canon) outperforms both RoPE/NoPE+Canon, great news for length generalization.<br><br><br>a(Sub-results correspond to Figure 7 (4th vs 5th column), Figure 10, and Figure 7 (3rd column), respectively.)|
|---|

ˇ “

Canon layers skyrocket NoPE performance. Canon layers dramatically improve NoPE (No Positional Embedding) transformers, lifting them from near-zero accuracy to competitive levels, even slightly surpassing RoPE+Canon on reasoning depth. NoPE-Canon is only weaker on Task Lano, which involves hierarchical structural learning over long sequences, thus relying more heavily on relative distance between input tokens; yet even there NoPE-Canon remains competitive with alternatives such as Mamba2/GDN.

Dominance over existing fixes on NoPE. While NoPE excels at length generalization, its performance on complex reasoning tasks has historically been weak. Fixes like ALiBi [45] and HardAlibi [31] partially address this: ALiBi applies a distance-based penalty to attention weights15, while Hard-Alibi disables attention beyond distance h for the h-th head. Although these methods improve NoPE performance (partly mimicking RoPE), Canon layers clearly dominate. As shown in Figure 10 (top), NoPE+Canon significantly outperforms both alternatives.

Minimal RoPE usage with Canon layers. Canon layers eliminate the need for heavy RoPE usage, and excessive RoPE can even hurt performance. With Canons, minimal RoPE usage is sufficient—often preferable—for optimal results. For example, enabling RoPE on half of the heads at half of their dimensions (denoted ˇ “Canon) consistently outperforms full RoPE usage or NoPE, as shown in Figure 7 (3rd column). This is great news for long-context generalization: RoPE is a known bottleneck for Transformers with longer inputs. As Canon layers allow significantly reduced RoPE without performance loss, they become indispensable for length generalization tasks.16

- Remark 5.1. Despite their versatility, Canon layers alone cannot fully resolve extremely challenging tasks that require deep hierarchical reasoning over long sequences (e.g., cfg3k in Task Lano). Such tasks, requiring O(n3) dynamic programming over 1000 tokens, remain computationally demanding. Nevertheless, Canon layers consistently offer huge improvements outside these specialized scenarios.

These findings translate to real-life. To be shown in Section 8, NoPE+Canon consistently matches or surpasses RoPE+Canon in real-world pretraining; the RoPE+ˇ “Canon variants outperform RoPE+Canon on several reasoning tasks, particularly involving long-context inputs.

- Remark 5.2. This paper focuses on architectural differences within computational stages after relevant information is retrieved into manageable contexts (e.g., 4096 tokens). Techniques like DeepSeek’s NSA architecture [78], designed for retrieval and compression from extremely long inputs (e.g., 1M tokens), are complementary to Canon layers. Such techniques and Canon layers can thus jointly handle distinct processing phases in long-context models.

15Specifically, adding −|j − i| · 2−8h/H to the logits of head h of H total heads. 16Alternative reduced-RoPE configurations explored in the appendix include: ˇ “ˇ “ (1/4 of heads at full dimensions)

and ˇ “ˇ “ˇ “ (all heads at 1/4 dimensions, as in GPT-NeoX [13]). Among these, ˇ “ and ˇ “ˇ “ˇ “ are comparable, slightly better than ˇ “ˇ “ according to Figure 26 on Page 47.

###### Task Brevo1 - Ablation study - Llama(NoPE)

###### Task Mano - Ablation study - Llama(NoPE)

###### Task Depo1(K=8, k=8/4) - Ablation study - Llama(NoPE)

[Figure 78]

83/99% 5/53% 29/84% 85/100% 66/99% 100/100% 99/100% 76/99% 98/100% 93/99% 85/99% 38/43% 71/96% 0/0%

6.9% 1.2% 92.5% 92.7% 84.5% 82.0% 88.6% 69.3% 90.9% 92.2% 93.4% 83.6% 84.4% 0.4% 88.5% 95.3% 96.2% 94.6% 95.6% 61.6% 92.8% 90.5% 91.8%

- 55.9% 93.3% 39.4% 52.1% 95.8% 97.5% 94.4% 97.1% 99.2% 28.3% 11.6% 98.8% 99.2% 6.9% 90.3% 98.6% 99.3% 98.3% 99.3% 95.6% 99.2% 78.2% 61.2% 24.0% 29.5% 7.6% 25.5%
- 56.2% 87.7% 84.9% 73.0% 96.7% 33.7% 7.5% 98.4% 98.4% 7.1% 66.3% 93.1% 98.1% 97.1% 98.7% 70.8% 98.1% 30.1% 15.8% 29.7% 30.3% 7.6% 7.9%

N=225 - Act - Res N=225 - NoRes

[Figure 79]

[Figure 80]

N=70 - Act - Res N=70 - NoRes

L=10 - Act - Res L=10 - NoRes

99/100% 99/100% 100/100% 98/100% 99/100% 62/97% 99/99% 76/98% 60/96% 7/28% 0/0% 77/99% 53/97%

N=70 - Res N=90 - Act - Res

L=10 - Res L=13 - Act - Res

N=225 - Res N=300 - Act - Res

- 21.4% 2.9% 55.3% 84.7% 45.6% 49.6% 70.0% 24.6% 72.9% 75.0% 76.8% 49.3% 58.1% 0.0%
- 22.2% 89.0% 92.2% 85.6% 90.9% 51.0% 81.7% 79.4% 74.8% 1.6% 2.1% 34.1% 49.1%

N=90 - NoRes

L=13 - NoRes

68/99% 100/100% 99/100% 82/99% 96/100% 66/98% 0/68% 23/85% 47/89% 0/0%

N=300 - NoRes

N=90 - Res N=110 - Act - Res

L=13 - Res L=16 - Act - Res

94/100% 99/100% 99/100% 93/99% 86/91% 49/93% 88/99% 90/99% 60/93% 0/10% 0/0% 5/49% 1/90% 7/78% 99/100% 96/100% 78/97% 97/100% 49/97% 43/75% 0/28% 53/94% 0/0%

N=300 - Res N=375 - Act - Res

58.3% 20.4% 38.2% 14.2% 20.0% 40.6% 57.7% 31.3% 31.8% 0.1% 50.7% 77.1% 84.9% 63.6% 83.0% 2.9% 67.3% 41.1% 40.5%

8.1% 81.2% 27.5% 48.3% 93.8% 7.6% 7.5% 72.8% 97.6% 7.3% 29.0% 93.3% 94.3% 90.9% 96.8% 80.5% 72.1% 19.5% 7.8%

N=110 - NoRes N=110 - Res

L=16 - NoRes L=16 - Res

N=375 - NoRes N=375 - Res

54/98% 93/99% 99/100% 89/99% 83/99% 52/94% 95/97% 55/93% 90/99%

Canon-ACanon-ABCCanon-ABCDCanon-ACCanon-ACDCanon-BCanon-BDCanon-CCanon-D Alibi Halibi original

Canon-ACanon-ABCCanon-ABCDCanon-ACCanon-ACDCanon-BCanon-BDCanon-CCanon-D Alibi Halibi original

Canon-ACanon-ABCCanon-ABCD Canon-ACCanon-ACD Canon-B Canon-BD Canon-C Canon-D Alibi Halibi original

###### Task Brevo2 - Ablation study - Llama(NoPE)

###### Task Lano - Ablation study - Llama(NoPE)

###### Task Depo2(K=16, k=16/8) - Ablation study - Llama(NoPE)

[Figure 81]

85/99% 37/88% 51/99% 1/82%

- 93.6% 91.0% 94.0% 93.7% 89.3% 90.0% 93.6% 87.7% 92.0% 92.5% 93.8% 93.4% 88.4% 0.0%
- 94.0% 95.4% 96.1% 95.4% 96.0% 93.2% 94.1% 93.7% 94.7% 84.1% 79.8% 85.7% 84.8% 80.4% 82.9% 81.9% 70.0% 88.4% 84.7% 85.5% 84.8% 80.4% 0.0% 87.0% 88.8% 91.7% 88.7% 91.6% 81.2% 86.3% 86.7% 84.3% 75.0% 64.2% 73.0% 75.6% 66.3% 55.7% 68.7% 19.4% 72.4% 71.8% 75.0% 72.9% 63.5% 0.0% 75.9% 76.0% 79.9% 75.7% 85.2% 50.4% 68.0% 74.0% 76.9%

77.5% 77.3% 92.2% 89.1% 73.9% 68.9% 67.9% 85.1% 86.2% 79.3% 81.3% 78.3% 8.0% 38.8% 82.9% 91.1% 92.5% 90.6% 93.1% 82.7% 89.7% 90.9% 91.2% 50.7% 59.0% 71.6% 76.6% 29.8% 12.0% 17.3% 54.1% 56.3% 55.3% 69.3% 54.1% 6.8% 0.0% 64.2% 78.7% 78.3% 76.7% 82.3% 65.2% 72.7% 76.4% 70.5% 36.0% 39.7% 57.6% 50.6% 18.3% 18.1% 9.4% 28.7% 26.3% 36.0% 37.6% 23.6% 5.2% 0.0% 41.8% 54.1% 53.1% 54.0% 51.2% 41.4% 42.3% 52.7% 57.8%

N=75 - Act - Res N=75 - NoRes

[Figure 82]

[Figure 83]

N=30 - Act - Res N=30 - NoRes

cfg3f - Act - Res cfg3f - NoRes cfg3f - Res

- 34/95% 1/1% 1/1% 80/97% 30/98% 25/97% 43/93% 78/98% 46/98% 0/0%
- 35/96% 100/100% 99/100% 90/100% 99/100% 71/98% 97/100% 36/99% 75/99% 73/99% 8/82% 72/98% 41/99% 25/88% 1/1% 1/1% 1/83% 2/78% 47/93% 5/95% 39/89% 29/87% 0/0%

N=30 - Res N=40 - Act - Res

N=75 - Res N=100 - Act - Res

- cfg3j - Act - Res cfg3j - NoRes

cfg3j - Res

- cfg3k - Act - Res cfg3k - NoRes

N=40 - NoRes

N=100 - NoRes

N=40 - Res N=50 - Act - Res

94/100% 99/100% 99/100% 96/100% 99/100% 17/91% 98/100% 45/93% 17/93% 10/91% 1/52% 30/97% 2/81%

N=100 - Res N=125 - Act - Res

N=50 - NoRes N=50 - Res

1/36% 1/1% 1/1% 41/83% 2/78% 68/97% 23/94% 68/90% 23/91% 0/0% 41/96% 99/100% 96/100% 90/99% 95/100% 2/92% 63/96% 12/93% 81/99%

N=125 - NoRes N=125 - Res

cfg3k - Res

Canon-ACanon-ABCCanon-ABCDCanon-ACCanon-ACDCanon-BCanon-BDCanon-CCanon-D Alibi Halibi original

Canon-ACanon-ABCCanon-ABCDCanon-ACCanon-ACDCanon-BCanon-BDCanon-CCanon-D Alibi Halibi original

Canon-ACanon-ABCCanon-ABCD Canon-ACCanon-ACD Canon-B Canon-BD Canon-C Canon-D Alibi Halibi original

###### Task Brevo1 - Ablation study - Llama(RoPE)

###### Task Depo1(K=8, k=8/4) - Ablation study - Llama(RoPE)

[Figure 84]

85/100% 48/96% 0/80% 0/81% 97/100% 40/92% 96/100% 98/100% 97/100% 78/99% 19/94% 0/1% 99/100% 93/100% 59/81% 94/100% 94/100% 96/100% 25/43% 92/100% 0/98% 100/100% 100/100% 97/100% 99/100%

87.1% 88.9% 90.8% 87.9% 92.7% 85.0% 84.9% 90.1% 92.7% 93.0% 92.4% 88.5% 92.2% 89.8% 92.3% 91.3% 91.3% 92.4% 85.9% 93.7% 91.7% 91.4% 96.5% 96.3% 96.1% 54.7% 69.2% 72.2% 71.2% 81.4% 68.0% 65.8% 70.4% 77.7% 80.5% 79.6% 63.1% 84.8% 67.1% 67.6% 75.7% 76.8% 87.8% 57.2% 72.5% 71.2% 63.8% 90.7% 91.1% 91.5%

N=225 - Act - Res N=225 - NoRes

[Figure 85]

N=70 - Act - Res N=70 - NoRes

N=70 - Res N=90 - Act - Res

N=225 - Res N=300 - Act - Res

0/30% 0/10% 0/77% 98/100% 66/99% 34/91% 54/99% 98/100% 95/100% 84/97% 79/99% 0/0% 84/100% 51/98% 89/99% 99/99% 51/95% 98/100% 0/40% 75/94% 0/85% 48/94% 95/100% 70/97% 100/100%

N=90 - NoRes

N=300 - NoRes

N=90 - Res N=110 - Act - Res

N=300 - Res N=375 - Act - Res

- 62.9% 41.0% 49.6% 54.7%
- 63.3% 39.8% 43.5% 51.0% 66.5% 57.5% 65.9% 27.5% 68.8% 51.6% 52.6% 58.8% 48.6% 49.9% 36.2% 55.4% 47.9% 54.6% 84.5% 84.7% 79.7%

0/98% 68/100% 0/0% 0/23% 17/96% 28/92% 68/95% 48/98% 81/99% 26/99% 0/99% 0/0% 99/100% 0/13% 3/86% 97/100% 89/99% 68/100% 0/13% 51/100% 3/61% 60/99% 90/100% 6/99% 98/100%

N=110 - NoRes N=110 - Res

N=375 - NoRes N=375 - Res

Canon-ACanon-ABCCanon-ABCDCanon-ACCanon-ACDCanon-BCanon-BDCanon-CCanon-D originalCanon-ABCDCanon-BCanon-ABCDCanon-ABCD

Canon-ACanon-ABCCanon-ABCD Canon-ACCanon-ACD Canon-B Canon-BD Canon-C Canon-D originalCanon-ABCD Canon-BCanon-ABCDCanon-ABCD

###### Task Brevo2 - Ablation study - Llama(RoPE)

###### Task Depo2(K=16, k=16/8) - Ablation study - Llama(RoPE)

[Figure 86]

54/99% 87/99% 86/99% 12/96%

96.3% 95.3% 96.6% 95.1% 93.0% 84.7% 89.6% 91.0% 92.6% 95.0% 94.4% 96.0% 92.9% 96.3% 95.7% 95.4% 96.5% 97.0% 94.6% 95.9% 96.5% 95.9% 97.1% 96.3% 97.3% 91.1% 88.0% 91.4% 91.9% 89.1% 73.7% 77.0% 81.8% 83.7% 91.9% 88.0% 88.0% 85.6% 91.7% 89.8% 90.5% 91.8% 92.6% 87.0% 92.2% 90.6% 92.9% 93.5% 89.2% 95.5% 88.7% 77.3% 86.0% 84.8%

N=75 - Act - Res N=75 - NoRes

[Figure 87]

N=30 - Act - Res N=30 - NoRes

1/17% 1/1% 1/18% 9/97% 10/98% 75/99% 1/94% 30/99% 36/92% 99/100% 99/100% 99/100% 99/100% 82/98% 98/100% 99/100% 83/99% 38/96% 99/100% 93/100% 100/100% 90/100% 62/96% 1/37% 58/98%

N=30 - Res N=40 - Act - Res

N=75 - Res N=100 - Act - Res

N=40 - NoRes

1/58% 1/1% 36/93% 65/97% 19/94% 1/77% 2/99% 21/96% 61/97% 97/100% 99/100% 98/100% 99/100% 99/100% 96/100% 96/100% 93/100% 42/98% 97/100% 97/100% 99/100% 1/98% 24/97% 1/80% 1/81% 1/1% 1/1% 1/58% 1/82% 1/9% 60/98% 1/97% 1/50% 37/98% 87/99% 98/100% 96/100% 93/99% 96/100% 89/99% 93/99% 90/100% 66/99% 98/100% 96/100% 99/100%

N=100 - NoRes

N=40 - Res N=50 - Act - Res

N=100 - Res N=125 - Act - Res

- 81.7% 67.9% 62.5% 65.6% 78.3% 84.0% 80.4% 81.4% 80.0%
- 82.3% 82.5% 87.8% 87.2% 88.7% 77.8% 84.5% 83.1% 85.7% 88.1% 85.9% 91.7%

N=50 - NoRes N=50 - Res

N=125 - NoRes N=125 - Res

Canon-ACanon-ABCCanon-ABCDCanon-ACCanon-ACDCanon-BCanon-BDCanon-CCanon-D originalCanon-ABCDCanon-BCanon-ABCDCanon-ABCD

Canon-ACanon-ABCCanon-ABCD Canon-ACCanon-ACD Canon-B Canon-BD Canon-C Canon-D originalCanon-ABCD Canon-BCanon-ABCDCanon-ABCD

###### Task Mano - Ablation study - Llama(RoPE)

Task Lano - Ablation study - Llama(RoPE)

99.0% 91.8% 77.7% 83.8% 92.1% 97.9% 94.5% 85.9% 98.6% 88.9% 50.3% 85.2% 94.3% 96.7% 97.7% 99.5% 95.8% 97.6% 97.5% 98.6% 83.3% 90.3% 99.6% 99.7% 99.6% 85.3% 69.5% 80.7% 66.4% 69.7% 95.8% 53.7% 79.4% 95.7% 32.9% 45.6% 46.5% 90.1% 83.8% 97.5% 98.8% 88.6% 98.0% 90.3% 87.7% 75.9% 77.6% 99.2% 99.1% 96.3% 47.7% 35.2% 47.9% 58.3% 57.8% 50.4% 40.1% 27.4% 78.0% 18.2% 26.9% 41.6% 53.9% 88.1% 86.4% 83.0% 78.3% 83.2% 79.4% 79.8% 60.9% 53.5% 85.3% 97.9% 88.0%

97.7% 97.6% 98.1% 97.8% 93.6% 83.1% 84.7% 95.7% 94.8% 95.1% 95.9% 97.6% 94.1% 97.5% 97.7% 98.3% 98.2% 98.5% 97.5% 97.9% 97.8% 97.5% 98.1% 97.6% 97.5% 92.9% 92.4% 93.8% 91.1% 64.0% 55.0% 63.6% 80.3% 82.7% 88.4% 87.7% 90.3% 65.9% 92.4% 94.9% 94.3% 95.1% 93.4% 94.2% 93.3% 93.4% 92.5% 92.6% 91.5% 93.3% 87.0% 84.8% 87.4% 83.3% 42.4% 23.9% 43.4% 53.0% 53.1% 72.5% 73.2% 79.1% 35.8% 87.5% 89.8% 86.7% 86.3% 84.7% 89.8% 85.2% 85.1% 85.8% 78.9% 81.1% 84.5%

[Figure 88]

[Figure 89]

L=10 - Act - Res L=10 - NoRes

cfg3f - Act - Res cfg3f - NoRes cfg3f - Res

L=10 - Res L=13 - Act - Res

- cfg3j - Act - Res cfg3j - NoRes

cfg3j - Res

- cfg3k - Act - Res cfg3k - NoRes

L=13 - NoRes

L=13 - Res L=16 - Act - Res

L=16 - NoRes L=16 - Res

cfg3k - Res

Canon-ACanon-ABCCanon-ABCDCanon-ACCanon-ACDCanon-BCanon-BDCanon-CCanon-D originalCanon-ABCDCanon-BCanon-ABCDCanon-ABCD

Canon-ACanon-ABCCanon-ABCDCanon-ACCanon-ACDCanon-BCanon-BDCanon-CCanon-D originalCanon-ABCDCanon-BCanon-ABCDCanon-ABCD

- Figure 10: Ablation study on 12-layer, 768-dim Transformers—NoPE (top) and RoPE (bottom)—with Canon variants (A–D), residual links, activation functions, ALiBi, and H-Alibi. Blank entries indicate untested configs due to resource limits. Additional ablation studies (with more model sizes) are in Figure 27 (RoPE), Figure 29 (NoPE), and Figure 28 (RoPE+Primer) in Appendix G.

##### 5.3 Ablation Studies With Canon Layers

This section systematically investigates the design choices in Canon layers via ablation studies.

Component-level contributions. Each Canon component (A/B/C/D) contributes meaningfully to performance, with cumulative benefits from combinations. Adding even a single Canon layer yields notable gains, and stacking multiple Canon layers across sub-layers further amplifies these improvements, especially on weaker architectures like NoPE. Summaries appear in Figure 10 (for model size 12L768D) and additional size experiments in Appendix G.

Role of residual connections. Residual links around Canon layers — i.e., the “ht+” part of (4.1) — are critical for training stability and effective learning, preserving vertical computational pathways and allowing global representations to selectively incorporate local context. Without residual connections, training becomes slower and less stable (see rows marked “NoRes” in Figure 10).

Independence of Attention/MLP. Prior works (e.g., the GLA [72] codebase and Primer [59]) focused solely on convolution operations within attention projections — Canon-B(no-res). However, we find that Canon-ACD alone already achieves substantial performance improvements, without modifying attention mechanisms. Similarly, Canon-ABC or even Canon-AC perform strongly without adjusting MLP layers. They all strongly outperform Canon-B(no-res) and thus outperform Primer. This highlights Canon layers’ general role in enhancing horizontal information flow across architecture sub-layers, independently complementing attention or MLP mechanisms.

###### Nonlinear activations and computational simplicity. Contrary to prior works (e.g., H3/Mamba),

###### Task Capo - Llama(RoPE) - original

|5-3 4-3<br><br>6-3<br><br>5-4 3-6 2-8 5-66-6<br><br>10-6 2-8 5-6<br><br>2-3<br><br>5-4<br><br>8-2<br><br>3-6<br><br>8-6 12-6 8-8<br><br>4-2<br><br>6-2<br><br>6-6<br><br>4-35-3<br><br>10-6<br><br>6-3<br><br>6-6<br><br>12-6 10-8<br><br>2-20<br><br>8-6 16-8<br><br>3-6<br><br>8-8<br><br>2-8<br><br>5-6<br><br>5-4<br><br>5-6 2-8<br><br>3-6<br><br>3-20<br><br>10-8<br><br>2-20<br><br>16-8<br><br>10-6<br><br>8-16 8-8<br><br>6-20<br><br>4-20<br><br>3-20<br><br>8-6<br><br>10-6<br><br>4-20<br><br>6-6<br><br>12-6<br><br>8-16<br><br>2-20<br><br>6-20 6-24<br><br>12-16 8-2420-16<br><br>24-1612-24<br><br>8-8<br><br>12-6<br><br>10-8<br><br>20-2024-20<br><br>2-20<br><br>12-16<br><br>16-8<br><br>6-24 4-20<br><br>8-16<br><br>6-20<br><br>8-24<br><br>3-20<br><br>20-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

learnedknowledge(bits)

- 106
- 107
- 108

106 107 108

model size (#params)

###### Task Capo - Llama(RoPE) - Canon-ABCD(res)

|2-3<br><br>6-3 5-4 3-6 2-8 5-66-6<br><br>2-3<br><br>4-2<br><br>5-3 4-3<br><br>8-2<br><br>3-6 12-6<br><br>6-2<br><br>2-8 5-66-6 8-610-6 8-8<br><br>4-3<br><br>10-6 5-6<br><br>5-4 6-3<br><br>3-6<br><br>5-3<br><br>6-6 8-6 12-6 8-8 10-8 2-2016-8 3-6<br><br>2-20<br><br>2-8<br><br>5-6<br><br>8-8 10-8 16-83-20 4-20 8-166-20<br><br>8-6 6-6<br><br>10-6<br><br>4-20 2-20<br><br>16-8<br><br>3-20 8-166-2012-166-24 8-2420-16<br><br>8-24<br><br>12-6 8-8<br><br>20-16<br><br>10-8<br><br>12-166-24 24-1612-2420-2024-20<br><br>2-20<br><br>16-8<br><br>4-20 3-20<br><br>8-166-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

learnedknowledge(bits)

- 106
- 107
- 108

106 107 108

model size (#params)

###### Task Capo - GPT2(RoPE) - original

###### Task Capo - 32-MoE - Llama(RoPE)

|6-3 5-4 3-6 6-6 4-3<br><br>2-8 5-3<br><br>8-2<br><br>5-6<br><br>5-3<br><br>6-6 5-4<br><br>3-6 10-6<br><br>8-2<br><br>2-8 8-6<br><br>4-2<br><br>5-6<br><br>2-36-2<br><br>12-68-8 5-3<br><br>10-6 8-8<br><br>4-3<br><br>12-6 5-6<br><br>10-8 2-8<br><br>5-4<br><br>16-8<br><br>6-3<br><br>8-6 3-6<br><br>2-20 6-6<br><br>8-166-20<br><br>3-6<br><br>2-2016-83-20 4-20 12-6<br><br>5-6<br><br>10-8 10-6<br><br>2-8<br><br>8-8<br><br>12-16<br><br>6-6<br><br>3-20 4-20<br><br>8-6<br><br>6-24 8-2420-16<br><br>10-6<br><br>8-16 2-20<br><br>6-20<br><br>24-16<br><br>12-6<br><br>12-16 12-24<br><br>8-8<br><br>10-8<br><br>20-16 20-20<br><br>2-20<br><br>24-20<br><br>16-8<br><br>3-20<br><br>6-24 8-24 4-20<br><br>8-166-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|4-2<br><br>2-3<br><br>6-2 8-2<br><br>4-3 5-3<br><br>2-3<br><br>4-3<br><br>5-3 6-3<br><br>5-4<br><br>4-2<br><br>5-4 3-62-8 5-6<br><br>2-3 8-2 6-2<br><br>4-3<br><br>5-3<br><br>5-4<br><br>3-6<br><br>5-6<br><br>6-6 8-6 10-612-6<br><br>5-4<br><br>6-3<br><br>3-6<br><br>5-3<br><br>5-6<br><br>6-6 8-6<br><br>2-20<br><br>5-6<br><br>8-6 2-8<br><br>3-6<br><br>5-4<br><br>6-6<br><br>8-8<br><br>10-612-6 10-8<br><br>16-8<br><br>2-20 3-20 4-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

- 106
- 107
- 108

- 106
- 107
- 108

106 107 108

106 107 108

model size (#params)

model size (#params)

###### Task Capo - GPT2(RoPE) - Canon-ABCD(res)

Task Capo - 32-MoE - Llama(RoPE) + Canon-ABC(r)

|5-36-3 5-4 3-6 2-8 5-66-6<br><br>2-3 4-2<br><br>6-2<br><br>8-2<br><br>5-4<br><br>4-3<br><br>3-6 2-8 5-66-6 8-610-612-6 8-8<br><br>4-3<br><br>5-66-6 8-610-612-6 8-8 2-20<br><br>5-3<br><br>2-8<br><br>5-4<br><br>10-8 16-8<br><br>6-3<br><br>2-8<br><br>12-6<br><br>8-8 4-20 8-166-20<br><br>5-6<br><br>3-6<br><br>10-8 2-2016-83-20<br><br>6-20<br><br>8-6<br><br>8-24<br><br>6-6<br><br>10-6<br><br>2-20<br><br>4-20 8-16 12-166-24 20-16<br><br>12-6<br><br>16-83-20<br><br>8-8<br><br>8-16<br><br>2-20 10-8<br><br>12-6<br><br>4-20 6-20 3-20<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|2-3<br><br>6-2 8-24-3 5-3<br><br>6-2 8-2<br><br>5-4 4-3<br><br>5-3 6-3 2-3<br><br>4-2<br><br>5-4<br><br>3-6 5-3<br><br>2-8 5-6<br><br>2-3<br><br>6-2<br><br>8-2<br><br>12-6<br><br>4-3<br><br>3-6<br><br>6-6 2-8<br><br>5-6 8-6 10-6<br><br>5-3<br><br>2-20<br><br>6-3<br><br>5-4<br><br>3-6<br><br>8-6<br><br>10-612-68-8 10-8<br><br>3-20<br><br>5-4<br><br>3-6<br><br>16-8 4-20<br><br>5-6 2-8<br><br>8-6 6-6<br><br>10-6<br><br>12-68-8<br><br>10-8 2-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

- 106
- 107
- 108

- 106
- 107
- 108

106 107 108

106 107 108

model size (#params)

model size (#params)

- Figure 11: Evaluation of knowledge capacity (Capo) across architectures, measured as bits per parameter. The first row represents baseline models, while the second row shows improvements with Canon layers added. Conclusion: Canon layers enhance knowledge storage for architectures that are slower to train, such as gated MLP and MoE, mitigating the capacity gap between gated and standard MLP as identified in [8].

adding activation functions such as SiLU after the Canon layers does not yield noticeable benefits. Canon layers effectively inject local context directly into token positions, and nonlinear operations are sufficiently handled by the attention and MLP blocks (see rows marked “Act” in Figure 10).

|Result 4 (Figure 10). Canon layers are lightweight, versatile, and effective enhancements that integrate seamlessly into Transformers. Key findings:<br><br>• Canon-A/B/C/D yield meaningful, cumulative improvements when stacked, and can be flexibly applied anywhere independent of attention or MLP modifications.<br>• Residual connections in Canon design are essential for stable, efficient training.<br>• Adding nonlinear activations (e.g., SiLU) provide no measurable benefit, simplifying design.<br>|
|---|

(This differs from prior works: we show where to insert Canon layers, how to stabilize them, and why they matter.)

##### 5.4 MLP and Mixture-of-Experts

Our synthetic playground provides a valuable framework to evaluate broader architectural choices. Gated vs. standard MLPs. Gated MLPs [56], which replace standard MLP operations V σ(Wx) by V (σ(W1x) · (W2x)), improve expressiveness and parameter efficiency. Widely adopted by largescale models (e.g., PaLM [16], Llama [64, 65], Mistral [33]), gated MLPs have become standard design choices. However, [8] found that gated MLP reduces knowledge capacity by about 30% in limited-exposure scenarios (e.g., 100-exposure Task Capo) due to slower convergence.

Thus, what is the best tradeoff? Our experiments (Figure 24 on Page 46) confirm gated MLP has slight advantage over standard MLP (“GPT2-style”) on reasoning-heavy tasks, showing noticeable improvements on knowledge manipulation (Mano) and smaller gains on reasoning breadth (Brevo). Thus, replacing gated MLP with standard MLP may not be the best choice. However, keep in mind that adding Canon layers already partially mitigates gated MLP’s capacity loss (recall Result 2), due to improving training dynamics and speed, recovering about half of its lost capacity. Mixture-of-Experts. Mixture-of-Experts (MoE) [22, 57] enhances parameter efficiency by replacing dense MLPs with multiple parallel “experts,” selectively routing tokens to fewer active

experts. While MoE achieves good scalability (particularly on knowledge capacity) and competitive inference-time performance, it suffers from significantly slower knowledge acquisition speed during training. For example, a 32-expert transformer may acquire 10× less knowledge in the same 100-exposure regime (mimicking rare knowledge) compared to dense models (Figure 11). Could Canon layers mitigate this due to their improved training dynamics?

Integrating Canon layers with MoE, however, poses a challenge. Canon-D relies on neighboring tokens’ hidden states, conflicting with MoE’s independent token-wise expert dispatching. Adapting Canon-D to MoE would require complex engineering. To avoid such complexity, we test CanonABC layers alone, which already significantly accelerate MoE knowledge acquisition and improve bit-per-parameter efficiency (Figure 11), recovering at least half of the MoE-induced capacity loss. MLP with Squared ReLU. The Primer [59] paper proposes using ReLU2 as the activation function in standard MLPs, reporting improved performance over gated MLPs (e.g., SwiGLU) on real-world data. They also claim this gain exceeds that of Canon-B(no-res), which they refer to as “Multi-DConv-Head Attention.” In our synthetic playground (see Figure 25 on Page 46), we confirm that ReLU2 slightly improves standard MLPs (though not necessarily outperforming gated MLPs, consistent with recent findings [81]), while applying ReLU2 to gated MLPs degrades performance. However, these effects are negligible compared to the gains provided by Canon layers.

|Result 5 (Figure 11+24+25). Key insights for MLP and MoE architectures:<br><br>• Gated MLP slightly outperforms standard MLP (especially on Mano).<br>• Gated MLP reduces knowledge capacity (Capo); Canon layers partially recover this loss.<br>• ReLU2 activation slightly improves standard MLP but degrades performance in gated MLP.<br>• Canon-ABC substantially improves MoE knowledge acquisition and bit-per-param capacity.<br>|
|---|

#### 6 When Linear Models Meet Canon

The three base linear models we study—GLA, Mamba2(mlp), and GDN—share a block-wise structure where each block consists of a “linear attention” layer (GLA, GDN, or Mamba2) followed by an MLP. This design naturally defines four insertion points for Canon layers, analogous to standard Transformers: A before the linear attention, B inside it, C before the MLP, and D inside. In the following subsections, we analyze each architecture separately.

##### 6.1 When Linear Attention Meets Canon

Linear attention models reduce computation by maintaining a compact state instead of attending over all tokens. In Gated Linear Attention (GLA) [72], the attention map is updated recursively as Wt = αtWt−1 + vtkt⊤, where Wt ∈ Rdkey×dvalue remains fixed in size regardless of context length. This design is efficient but effectively averages over past tokens, weakening the influence of nearby ones—crucial for reasoning. Canon layers restore localized horizontal context flow, alleviating this limitation and improving reasoning fidelity.

Following the original GLA release, its authors added a conv1d-based enhancement in their GitHub repo—corresponding to our Canon-B variant but using SiLU activation and omitting residual connections. We refer to this as GLA conv1d or equivalently GLA+Canon-b. To show the strongest comparison, our Canon-AbCD(res) extends it by adding residual Canon-ACD layers while keeping their conv1d. We also explore the full Canon-ABCD design in the appendix.

###### GLA - original

###### GLA - conv1d

###### GLA - Canon-AbCD(res)

###### Mamba2(mlp) - noconv1d

###### Mamba2(mlp) - original (conv1d)

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

7/46% 2/14% 14/55% 19/62% 1/11% 1/9% 1/22% 4/31%

14/70% 37/91% 72/96% 72/97% 2/41% 22/77% 15/80% 27/86% 2/26% 13/72% 11/73% 15/77%

19/75% 37/91% 22/85% 50/96% 7/49% 15/76% 11/81% 39/83% 5/28% 10/59% 7/56% 6/55%

2/40% 11/71% 1/24% 6/43% 1/27% 9/46% 0/15% 4/35%

12/65% 29/67% 24/74% 43/84% 4/35% 12/46% 13/62% 13/61% 1/22% 13/56% 7/33% 10/42%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/4% 0/0% 0/6% 1/16%

0/7% 1/12% 0/10% 0/10%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=4, k=4/2) GLA - original

###### Task Depo2(K=4, k=4/2) GLA - conv1d

Task Depo2(K=4, k=4/2) GLA - Canon-AbCD(res)

###### Task Depo2(K=4, k=4/2) Mamba2(mlp) - noconv1d

Task Depo2(K=4, k=4/2) Mamba2(mlp) - original (conv1d)

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

3/22% 1/5% 5/39% 3/12% 6/34% 1/13% 3/25% 1/6%

28/62% 42/79% 70/91% 53/85% 13/46% 19/65% 26/70% 39/82% 17/39% 22/59% 8/35% 20/51%

67/89% 83/96% 77/98% 89/98% 42/77% 46/85% 63/96% 73/92% 35/71% 13/79% 46/93% 69/91%

6/37% 20/63% 2/60% 2/38% 1/3% 1/12% 2/37% 7/71% 1/2% 4/44% 1/7% 1/9%

18/61% 80/95% 30/86% 69/89% 17/63% 47/83% 17/55% 42/82%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

1/1% 1/3% 2/26% 4/18%

5/39% 46/85% 10/41% 24/75%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 GLA - original

###### Task Brevo1 GLA - conv1d

Task Brevo1 GLA - Canon-AbCD(res)

###### Task Brevo1 Mamba2(mlp) - noconv1d

###### Task Brevo1 Mamba2(mlp) - original (conv1d)

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

33.7% 36.5% 46.1% 42.2% 1.7% 2.8% 6.2% 11.9% 1.2% 10.7% 2.9% 15.2%

62.5% 94.8% 69.7% 96.5% 41.4% 74.8% 26.2% 78.0% 38.6% 43.6% 10.5% 16.2%

88.0% 96.3% 95.3% 97.4% 83.3% 90.6% 89.7% 93.2% 65.5% 68.2% 79.4% 81.9%

1.0% 23.5% 0.5% 5.9% 0.9% 0.5% 0.6% 1.0% 0.5% 0.4% 0.8% 0.4%

3.7% 80.1% 50.1% 72.4% 0.3% 0.5% 3.8% 4.8% 0.1% 0.0% 1.1% 1.2%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 GLA - original

Task Brevo2 GLA - conv1d

Task Brevo2 GLA - Canon-AbCD(res)

###### Task Brevo2 Mamba2(mlp) - noconv1d

###### Task Brevo2 Mamba2(mlp) - original (conv1d)

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

2.8% 45.5% 21.5% 33.2% 0.7% 1.0% 1.8% 8.8% 0.1% 0.7% 1.0% 1.6%

91.5% 94.9% 95.0% 94.4% 67.5% 71.7% 80.0% 79.5% 31.8% 39.2% 57.6% 55.7%

91.1% 96.7% 94.8% 97.3% 71.1% 89.6% 78.5% 88.1% 46.8% 67.5% 50.9% 68.0%

20.8% 46.6% 14.1% 63.8% 1.8% 36.7% 3.5% 10.1% 1.1% 1.9% 1.2% 3.8%

50.8% 95.6% 68.1% 3.4% 12.5% 67.0% 14.5% 0.5%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

3.3% 12.4% 4.0% 0.5%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Mano GLA - original

Task Mano GLA - conv1d

Task Mano GLA - Canon-AbCD(res)

###### Task Mano Mamba2(mlp) - noconv1d

Task Mano Mamba2(mlp) - original (conv1d)

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

59.8% 56.0% 76.2% 56.1% 36.5% 31.1% 69.5% 44.4% 16.8% 24.7% 24.8% 22.4%

94.3% 95.8% 92.3% 97.8% 81.6% 54.0% 86.4% 90.2% 46.1% 35.3% 35.1% 62.1%

96.2% 99.4% 79.7% 97.8% 95.5% 95.1% 59.9% 84.2% 84.2% 72.7% 55.1% 65.9%

68.0% 60.3% 70.8% 62.4% 35.9% 40.3% 49.8% 40.8% 12.7% 28.7% 21.9% 26.7%

96.5% 95.1% 95.2% 95.7% 79.9% 84.8% 88.0% 91.8% 74.4% 90.1% 72.3% 87.4%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Capo - GLA - original

###### Task Capo - GLA - conv1d

###### Task Capo - GLA - Canon-AbCD(res)

###### Task Capo - Mamba2(mlp) - noconv1d

Task Capo - Mamba2(mlp) - original(conv1d)

|5-3<br><br>6-3 5-4 3-6 2-8 5-66-6 4-3<br><br>4-2 2-3<br><br>5-4 6-6 6-3<br><br>2-8 5-6 8-6 8-8<br><br>3-6<br><br>10-612-6<br><br>6-2<br><br>8-2<br><br>8-610-612-6 5-6<br><br>6-6 8-810-8 16-8<br><br>4-3<br><br>5-4<br><br>5-3<br><br>6-3<br><br>3-6 2-8<br><br>2-20 3-6 2-8<br><br>8-8<br><br>5-6<br><br>12-6<br><br>10-8 2-20 16-83-20 4-20 8-166-20<br><br>10-6 8-6<br><br>6-6<br><br>16-8 8-8 10-8<br><br>3-20 4-20 8-166-2012-166-24 8-2420-16<br><br>12-6<br><br>10-8 8-8<br><br>16-8<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2-20<br><br>3-20<br><br>4-20 8-166-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-3 6-2<br><br>3-6 5-3<br><br>5-4 2-8 5-6 4-3<br><br>6-6 8-2<br><br>4-22-3<br><br>5-4 2-8 6-6<br><br>6-3<br><br>3-6 5-6 8-610-612-6 8-8<br><br>8-2 6-2<br><br>10-6 2-8<br><br>5-66-6 8-6 12-6 8-8 10-8 16-8<br><br>4-35-3<br><br>5-4<br><br>6-3<br><br>3-6<br><br>2-20 3-6<br><br>2-8<br><br>12-6 8-8 10-8 16-8 8-610-6<br><br>5-6<br><br>8-6 2-20 3-20 4-20 8-166-20 6-6<br><br>10-6<br><br>8-8 10-8<br><br>2-20 16-83-20 4-20 8-166-2012-166-24 8-2420-16<br><br>8-8 10-8<br><br>12-6<br><br>16-8<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2-20<br><br>3-20 4-20 8-166-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|5-36-3 3-6 2-8 5-66-6 4-3<br><br>5-4<br><br>6-2 4-2<br><br>5-4<br><br>8-2<br><br>2-3<br><br>3-6 2-8 5-66-6 8-610-612-6 8-8<br><br>6-3<br><br>5-4<br><br>4-35-3<br><br>2-8 5-66-6 8-610-6<br><br>3-6 12-6 8-810-8 2-20 16-8<br><br>2-8<br><br>12-6<br><br>5-6<br><br>10-6 8-810-8 2-20 16-83-20 4-20 8-166-20<br><br>8-8<br><br>6-6<br><br>2-20 16-8<br><br>8-6<br><br>10-6<br><br>3-20 4-20 8-166-2012-166-24 8-2420-16<br><br>12-16 24-1612-2420-2024-20<br><br>8-8 10-8<br><br>12-6<br><br>16-8 4-20<br><br>2-20<br><br>3-20 8-166-20 6-24 8-2420-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|2-3<br><br>8-2<br><br>5-3<br><br>6-3<br><br>5-4 3-6 2-8 5-66-6<br><br>4-2<br><br>2-36-2<br><br>8-2<br><br>4-3<br><br>5-4<br><br>3-6 2-8 5-66-6 8-610-612-68-8<br><br>4-35-3<br><br>6-3<br><br>5-4<br><br>5-6<br><br>8-610-612-68-8 10-8 2-2016-8 2-8<br><br>5-6 3-6<br><br>8-610-6<br><br>2-20<br><br>16-83-20 4-20 8-166-20 10-6<br><br>8-16<br><br>6-6<br><br>3-20<br><br>8-6<br><br>4-20 6-20<br><br>10-8 2-20 8-8<br><br>16-8<br><br>12-166-24 8-2420-16<br><br>6-24 8-24 24-1612-2420-2024-20<br><br>16-8<br><br>4-20<br><br>6-20<br><br>12-6<br><br>3-20<br><br>2-20<br><br>10-8 8-8<br><br>8-16 12-16 20-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-3 5-4 3-6 6-6<br><br>8-2<br><br>5-3 2-8 5-6<br><br>4-2<br><br>5-4<br><br>8-2 6-2<br><br>3-6 2-8 5-66-6 8-610-612-6<br><br>2-3<br><br>8-8<br><br>5-3<br><br>4-3<br><br>5-4<br><br>8-6<br><br>3-6<br><br>6-6<br><br>6-3<br><br>5-6 10-612-68-8 10-8 2-20<br><br>5-3<br><br>16-8<br><br>8-8 10-8 3-20<br><br>2-8<br><br>5-6<br><br>16-8 6-20<br><br>3-6<br><br>10-612-6 2-20 4-20 8-16<br><br>16-8<br><br>10-6<br><br>8-16<br><br>8-6<br><br>6-6<br><br>10-8 3-20 4-20 6-2012-166-24 8-2420-16 12-6<br><br>10-8 8-8 2-20<br><br>16-8<br><br>4-20 8-166-20 3-20<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

106 107 108

106 107 108

106 107 108

106 107 108

106 107 108

model size (#params)

model size (#params)

model size (#params)

model size (#params)

model size (#params)

###### Task Lano GLA - original

Task Lano GLA - conv1d

Task Lano GLA - Canon-AbCD(res)

###### Task Lano Mamba2(mlp) - noconv1d

Task Lano Mamba2(mlp) - original (conv1d)

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

37.5% 61.1% 41.8% 54.9%

79.0% 89.5% 85.1% 90.6% 50.8% 73.9% 63.5% 79.3% 32.0% 58.7% 35.1% 62.6%

88.5% 92.8% 90.0% 94.1% 68.7% 84.6% 78.1% 84.3% 53.5% 64.9% 54.8% 73.3%

41.2% 53.4% 29.8% 56.0% 3.8% 8.9% 6.9% 8.3% 10.4% 13.4% 9.3% 13.7%

83.8% 92.2% 86.8% 92.2% 45.5% 72.0% 54.2% 74.3% 32.7% 50.0% 35.3% 46.1%

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

2.7% 17.1% 9.1% 35.7% 13.0% 11.6% 12.9% 19.8%

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Figure 12: Columns 1, 2, 3, 5: Canon drastically improves GLA, making it better than Mamba2 (Result 6.1). Columns 1, 4, 5: Removing conv1d reduces Mamba2’s performance back to match GLA (Result 7.1). Remark. Synthetic results here predict similar trends in real-life experiments (Result 12 and Figure 16).

As shown in Figure 12, integrating Canon-AbCD substantially boosts GLA’s original (nonconv1d) performance across all benchmarks, transforming it from a weak baseline into a strong competitor. Despite its simplicity, GLA+Canon matches or surpasses Mamba2, particularly on reasoning breadth (Brevo). This upward trend persists in large-scale real-world pretraining (Section 8), improving nearly all standard evaluation metrics.

|Result 6.1 (Figure 12). Adding Canon layers:<br><br>• Dramatically improves GLA’s original performance—raising reasoning depth from 1-hop to 4-hop, doubling reasoning breadth, and more than doubling knowledge manipulation length.<br>• Brings GLA on par with or beyond Mamba2, significantly outperforming it on Brevo.<br>• Yields additional gains even over the stronger GLA conv1d baseline.<br>|
|---|

As in the Transformer case, we perform ablations to determine optimal Canon placement. GLA also supports feature-map variants like Wt = αtWt−1 + vtϕ(kt)⊤, with the popular choice ϕ(x) = 1 + elu(x) [35]. We test Canon compatibility both with and without this feature map.

|Result 6.2 (Figure 33+34 on Page 52). Ablation study on GLA:<br><br>• Residualness. Unlike in full Transformers, Canon residualness is less critical: non-residual variants work better for Mano, while residual ones suit Lano/Brevo1.<br>• Positioning. Canon design is not intrinsic to the attention layer. Canon-ACD (or even Canon-A/C/D alone) can outperform Canon-b/B on many tasks, and combining all is best.<br>• Feature maps. Canon works well with 1 + elu(x) feature map, though better without it.a aConsistent with [72], where original GLA (without Canon) also performed better without feature maps.<br><br><br>|
|---|

Overall, these ablations highlight the importance of horizontal information flow independent of the architecture sublayers. Interested readers can find our full ablation results in Figure 33+34, where we for instance carefully compared Canon-AbCD(res/no-res), Canon-ABCD(res/no-res), and many more. We recommend the Canon-AbCD(res) configuration for GLA—keeping the nonresidual conv1d from their original codebase while combining it with our residual, activation-free Canon-ACD. This achieves strong gains with minimal code changes.

##### 6.2 When Mamba Meets Canon

While Mamba2 is recognized as a state-space model (SSM), it quietly includes a non-linear conv1d operation in each SSM block.17 Originally introduced in H3 [23] as a shift-SSM, this mechanism effectively acts as a partial Canon-B layer—performing horizontal mixing on selected coordinates, applying non-linear activation, and omitting residual connections.

Surprisingly, this built-in conv1d contributes more to Mamba2’s performance than its SSM formulation itself. Disabling it sharply degrades results, reducing Mamba2 to GLA-level performance on both synthetic (Figure 12) and real-world datasets (Section 8). This raises a key question: is Mamba2’s strength primarily due to its Canon-like conv1d rather than the state-space mechanism?

To isolate this effect, we refer to Mamba2’s internal conv1d as Canon-b, and extend it by adding residual Canon-A/C/D layers—denoted Mamba2(mlp)+Canon-AbCD. We also test our own Canon-B design in later ablations and in the appendix.18 We additionally examine Mamba2 without MLP layers (which exposes Canon-A/B positions), reported in the appendix.19

As shown in Figure 13, adding Canon-AbCD further improves Mamba2(mlp) performance over the built-in conv1d (Canon-b), especially on Mano and Lano.

|Result 7.1 (Figure 12+13). Key observations on Mamba2:<br><br>• Mamba2 includes an internal non-linear conv1d (partial Canon-B) that contributes more to performance than the SSM itself. Removing it drops performance to GLA levels.<br>• Replacing this with full Canon-AbCD layers further improves, notably on Mano, Lano.<br>|
|---|

(Mamba1 [26] shows similar trends but is consistently outperformed by Mamba2 in our playground.)

To further understand Canon–Mamba interactions, we perform ablations varying Canon position, residualness, and initialization. Results mirror GLA: Canon layers remain effective even when placed outside the SSM block, showing that horizontal information flow is architecture-independent.

For initialization, we test the recent mimetic initialization [66], proposed to enhance associative

17Mamba1 also contains this component, but since Mamba2 consistently outperforms it, we report only Mamba2. 18For example, with ssm state size=64 and num heads=16, our Canon-B applies to all 4d + 144 intermediate

coordinates for hidden size d, whereas Mamba2’s original conv1d acts only on a subset (2d + o(d)) with activation.

19Such Mamba2 doubles the layer count and recurrent state size compared to Mamba2(mlp). In practice, Mamba2(mlp) is preferred, e.g., in Falcon-H1 [63].

###### Mamba2(mlp) - original (conv1d)

###### Mamba2(mlp) - Canon-AbCD(res)

###### GDN - noconv1d

###### GDN - original (conv1d)

###### GDN - Canon-AbCD(res)

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

12/65% 29/67% 24/74% 43/84% 4/35% 12/46% 13/62% 13/61% 1/22% 13/56% 7/33% 10/42%

51/87% 21/80% 67/96% 87/99% 11/45% 19/61% 15/63% 34/78% 8/46% 16/75% 6/53% 4/52%

11/81% 26/78% 23/75% 22/83% 3/24% 14/73% 7/47% 20/72% 1/19% 8/43% 3/31% 1/30%

85/97% 85/95% 22/86% 32/92% 47/90% 22/85% 61/93% 57/89% 28/78% 32/87% 11/51% 26/71%

74/96% 48/92% 69/94% 91/98% 12/58% 14/75% 17/72% 83/95% 25/73% 47/87% 21/68% 50/83%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=4, k=4/2) Mamba2(mlp) - original (conv1d)

Task Depo2(K=4, k=4/2) Mamba2(mlp) - Canon-AbCD(res)

Task Depo2(K=4, k=4/2) GDN - noconv1d

Task Depo2(K=4, k=4/2) GDN - original (conv1d)

Task Depo2(K=4, k=4/2) GDN - Canon-AbCD(res)

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

18/61% 80/95% 30/86% 69/89% 17/63% 47/83% 17/55% 42/82%

69/93% 93/98% 87/97% 87/96% 44/83% 84/95% 66/91% 69/94% 23/52% 77/93% 24/63% 67/88%

69/96% 94/99% 77/97% 87/99% 46/89% 75/95% 13/72% 79/96% 11/68% 44/86% 11/84% 67/94%

94/99% 90/99% 96/99% 99/100% 87/98% 79/95% 94/98% 96/99% 69/95% 86/96% 86/97% 84/97%

96/99% 98/100% 95/99% 96/100% 89/99% 95/98% 92/99% 94/99% 66/94% 93/99% 84/96% 89/96%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

5/39% 46/85% 10/41% 24/75%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Mamba2(mlp) - original (conv1d)

###### Task Brevo1 Mamba2(mlp) - Canon-AbCD(res)

###### Task Brevo1 GDN - noconv1d

Task Brevo1 GDN - original (conv1d)

Task Brevo1 GDN - Canon-AbCD(res)

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

3.7% 80.1% 50.1% 72.4% 0.3% 0.5% 3.8% 4.8% 0.1% 0.0% 1.1% 1.2%

13.8% 78.7% 32.4% 92.7% 0.6% 17.2% 21.8% 56.3% 0.6% 38.4% 3.6% 40.6%

87.8% 90.8% 90.2% 96.2% 64.1% 75.8% 82.6% 90.6%

92.5% 94.9% 96.2% 96.7% 78.2% 90.1% 91.7% 87.1% 63.8% 79.3% 90.6% 88.0%

95.8% 96.7% 95.7% 96.8% 91.4% 95.0% 91.2% 86.3% 83.1% 89.6% 80.3% 88.3%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

1.7% 68.3% 42.7% 81.9%

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 Mamba2(mlp) - original (conv1d)

###### Task Brevo2 Mamba2(mlp) - Canon-AbCD(res)

Task Brevo2 GDN - noconv1d

Task Brevo2 GDN - original (conv1d)

Task Brevo2 GDN - Canon-AbCD(res)

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

50.8% 95.6% 68.1% 3.4% 12.5% 67.0% 14.5% 0.5%

53.8% 87.1% 65.0% 68.1% 14.7% 27.5% 19.3% 10.7% 0.8% 20.9% 0.9% 3.0%

90.3% 93.2% 91.3% 97.5% 56.5% 84.2% 71.2% 89.4% 15.5% 32.9% 58.2% 69.1%

97.3% 98.8% 98.6% 98.7% 92.7% 96.3% 96.1% 96.8% 73.1% 92.7% 87.4% 89.1%

98.0% 97.6% 98.4% 98.6% 92.6% 94.0% 93.4% 96.3% 87.4% 86.4% 86.6% 90.6%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

3.3% 12.4% 4.0% 0.5%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Capo - Mamba2(mlp) - original(conv1d)

###### Task Capo - Mamba2(mlp) - Canon-AbCD(res)

###### Task Capo - GDN - noconv1d

###### Task Capo - GDN - original(conv1d)

Task Capo - GDN - Canon-AbCD(res)

|6-3 5-4 3-6 6-6<br><br>8-2<br><br>5-3 2-8 5-6<br><br>4-2<br><br>5-4<br><br>8-2 6-2<br><br>3-6 2-8 5-66-6 8-610-612-6<br><br>2-3<br><br>8-8<br><br>5-3<br><br>4-3<br><br>5-4<br><br>8-6<br><br>3-6<br><br>6-6<br><br>6-3<br><br>5-6 10-612-68-8 10-8 2-20<br><br>5-3<br><br>16-8<br><br>8-8 10-8 3-20<br><br>2-8<br><br>5-6<br><br>16-8 6-20<br><br>3-6<br><br>10-612-6 2-20 4-20 8-16<br><br>16-8<br><br>10-6<br><br>8-16<br><br>8-6<br><br>6-6<br><br>10-8 3-20 4-20 6-2012-166-24 8-2420-16 12-6<br><br>10-8 8-8 2-20<br><br>16-8<br><br>4-20 8-166-20 3-20<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-3 5-4 3-6 2-8 5-66-6<br><br>4-2<br><br>2-3<br><br>6-2<br><br>8-2<br><br>5-3<br><br>3-6 2-8 5-66-6 8-610-612-68-8 5-3<br><br>4-3<br><br>6-3<br><br>5-4<br><br>6-6 8-6 2-8<br><br>5-6 10-612-68-8 10-8 2-2016-8<br><br>12-68-8<br><br>3-6<br><br>2-8<br><br>5-6<br><br>10-6<br><br>10-8 2-2016-83-20 4-20 8-166-20<br><br>10-6 8-6<br><br>16-8<br><br>6-6<br><br>3-20 4-20 8-166-2012-166-24 8-2420-16<br><br>12-166-24 8-24 24-1612-2420-2024-20<br><br>2-20 12-6<br><br>16-83-20<br><br>8-16<br><br>8-8<br><br>6-20<br><br>10-8<br><br>4-20 20-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|8-2 2-3<br><br>4-35-3<br><br>5-4 6-3<br><br>8-2 3-6 2-8 5-66-6 6-2<br><br>4-2<br><br>2-3<br><br>6-3<br><br>5-4 3-6 2-8 5-66-6 8-610-612-68-8<br><br>4-35-3<br><br>6-3<br><br>5-4<br><br>3-6 2-8<br><br>5-66-6 8-610-612-68-8 10-8 2-2016-8<br><br>3-6 2-8<br><br>5-66-6<br><br>8-6<br><br>10-612-6 8-8 10-8 2-2016-83-20 4-20 8-166-20 6-6<br><br>10-6 8-6<br><br>16-8 8-8 2-20<br><br>3-20 4-20 8-166-2012-166-24 8-2420-16<br><br>8-16 24-1612-2420-2024-20 3-20<br><br>12-166-24 8-24<br><br>2-20<br><br>6-20 20-16 16-8<br><br>4-20<br><br>12-6<br><br>8-8 10-8<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|8-2 6-2<br><br>8-2 4-35-36-3 5-4 3-6 2-8 5-66-6 6-2<br><br>4-2<br><br>2-3<br><br>6-3<br><br>5-4 3-6 2-8 5-66-6 8-610-612-68-8 4-3<br><br>5-3<br><br>6-3<br><br>5-4<br><br>3-6<br><br>2-8<br><br>2-8 5-66-6 8-610-612-68-810-8 2-2016-8<br><br>3-6<br><br>5-6<br><br>8-6<br><br>10-612-6 8-810-8 2-2016-83-20 4-20 8-166-20 6-6<br><br>8-6<br><br>10-6<br><br>3-20 12-6<br><br>2-2016-8<br><br>8-8 10-8<br><br>4-20 8-166-2012-166-24 8-2420-16<br><br>12-16 24-1612-2420-2024-20 3-20<br><br>8-166-20 6-24 8-2420-16 16-8<br><br>4-20<br><br>12-6<br><br>8-8<br><br>10-8<br><br>2-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-3 5-4 3-6<br><br>4-35-3 2-8 5-66-6 6-2<br><br>8-2<br><br>4-2<br><br>5-4 3-6 2-8 5-66-6 8-610-612-6 8-8<br><br>2-3<br><br>8-8 3-6<br><br>8-610-612-6 10-8<br><br>6-3<br><br>6-6<br><br>5-4<br><br>5-6 2-8<br><br>2-20<br><br>5-3<br><br>16-8<br><br>4-3<br><br>2-8 3-6<br><br>5-6<br><br>8-610-612-68-8 10-8 2-2016-83-20 4-20 8-166-20<br><br>8-6 6-6<br><br>10-6<br><br>8-8<br><br>10-8 2-2016-83-20 4-20 8-166-2012-166-24 8-2420-16<br><br>20-2024-20<br><br>12-6<br><br>8-8<br><br>3-20 8-166-20 24-1612-24<br><br>2-20<br><br>4-20 16-8<br><br>10-8<br><br>12-166-24 8-2420-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

106 107 108

106 107 108

106 107 108

106 107 108

106 107 108

model size (#params)

model size (#params)

model size (#params)

model size (#params)

model size (#params)

Task Mano Mamba2(mlp) - original (conv1d)

Task Mano Mamba2(mlp) - Canon-AbCD(res)

###### Task Mano GDN - noconv1d

Task Mano GDN - original (conv1d)

Task Mano GDN - Canon-AbCD(res)

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

96.5% 95.1% 95.2% 95.7% 79.9% 84.8% 88.0% 91.8% 74.4% 90.1% 72.3% 87.4%

- 97.8% 99.5% 99.6% 99.4%
- 98.6% 86.6% 98.7% 98.0% 96.1% 83.1% 96.6% 99.0%

54.3% 64.1% 66.1% 72.3% 44.9% 44.7% 42.2% 36.4% 25.1% 53.3% 54.8% 18.9%

93.6% 97.9% 91.6% 89.8% 90.0% 98.4% 85.0% 75.0% 81.2% 63.6% 55.2% 52.6%

99.7% 98.3% 99.3% 98.1% 90.3% 87.1% 98.7% 95.5% 93.1% 87.8% 80.3% 89.3%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Mamba2(mlp) - original (conv1d)

Task Lano Mamba2(mlp) - Canon-AbCD(res)

###### Task Lano GDN - noconv1d

Task Lano GDN - original (conv1d)

Task Lano GDN - Canon-AbCD(res)

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

83.8% 92.2% 86.8% 92.2% 45.5% 72.0% 54.2% 74.3% 32.7% 50.0% 35.3% 46.1%

88.3% 93.1% 90.1% 93.6% 71.3% 80.0% 72.1% 80.8% 44.1% 61.1% 49.3% 64.4%

35.3% 49.7% 25.5% 69.9% 2.8% 3.6% 2.6% 26.1% 9.4% 13.4% 9.1% 12.9%

83.4% 93.8% 91.5% 94.4% 54.9% 80.3% 70.9% 86.5% 38.0% 63.4% 44.6% 65.7%

90.6% 93.6% 92.8% 95.1% 69.7% 86.2% 82.4% 87.1% 49.6% 68.3% 62.9% 75.2%

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

- Figure 13: Mamba(mlp) and GDN architectures with no conv1d, with conv1d (original), and with full Canon.

recall and length generalization. However, our experiments (Figure 30+31) find no measurable benefit—and often degradation—on other tasks, suggesting that mimetic init may have overfit length generalization at the cost of broader reasoning. These findings highlight the importance of evaluating architectural choices over a diverse synthetic playground.

|Result 7.2 (Figure 30+32 on Page 50). Ablation study on Mamba2(mlp):<br><br>• Mamba2(mlp) slightly prefers residual Canon for Lano, but non-residual for Mano.<br>• Canon layers stay effective outside the SSM block; e.g., Canon-ACD surpasses Mamba2(conv1d) on Depo2/Lano, highlighting their strength as general horizontal-mixing modules.<br>• Mimetic initialization [66], designed for length generalization, harms shorter-context performance, reinforcing the need for diverse-task evaluation.<br>|
|---|

We also evaluate Mamba2 without MLP layers (Figure 30+31); results remain consistent with those above. Interested readers can refer to Figure 30+31+32 for complete ablation results, including detailed comparisons between Canon-ABCD(res/no-res), Canon-AbCD(res/no-res) and many more. Our overall recommendation remains Canon-AbCD(res) for simplicity.

##### 6.3 When Gated DeltaNet Meets Canon

Gated DeltaNet (GDN) [73] extends GLA with a gated delta-rule update. Instead of GLA’s Wt = αtWt−1 + vtkt⊤, GDN adopts Wt = αtWt−1(I − βtktkt⊤) + βtvtkt⊤, where βt controls the balance between forgetting and writing. This formulation retains GLA’s efficiency while adaptively

suppressing redundant information, allegedly yielding better reasoning and improved gradient flow.

Each GDN block retains the linear-attention–plus–MLP structure but also includes a nonresidual, activated conv1d layer within its linear attention sublayer—referred to here as conv1d or Canon-b. This component remains important, though less critical than in GLA or Mamba2. Removing it destroys knowledge manipulation (Mano) and hierarchical reasoning (Lano), while incontext reasoning (Depo/Brevo) is largely unaffected. (Section 8 later shows such differences may vanish in academic-scale real-life pretraining, highlighting the importance of a versatile synthetic pretrain playground.)

Following prior sections, we extend GDN by adding residual Canon-A/C/D layers, forming GDN+Canon-AbCD. We also test our own Canon-B design in later ablations and the appendix. As shown in Figure 13, Canon-AbCD slightly improves GDN+conv1d across benchmarks.

|Result 8.1 (Figure 13). Key observations on GDN:<br><br>• GDN is less dependent on its internal conv1d (Canon-b) for strong performance.<br>• Replacing it with full Canon-AbCD layers still yields improvements, albeit marginal.<br>|
|---|

We further perform ablation studies on Canon positioning and residualness:

|Result 8.2 (Figure 35+36 on Page 53). Ablation studies on GDN:<br><br>• GDN slightly prefers non-residual Canon on Mano, though overall differences are minor.<br>• Canon layers remain effective even outside the GDN layer; e.g., Canon-ACD performs on par with GDN+conv1d, underscoring their generality as horizontal-mixing components.<br>|
|---|

Interested readers can refer to Figure 35+36 for full ablation results, including detailed comparisons among Canon-ABCD(res/no-res), Canon-AbCD(res/no-res), and others. For simplicity and consistency, we recommend Canon-AbCD(res) as the default configuration.

#### 7 Final Comparisons and Lessons to Architecture Design

Applying Canon uniformly across all architectures creates a controlled environment—like dropping them from the same height at the Tower of Pisa—revealing their true architectural trade-offs. We exclude hybrid models (e.g., Griffin [20], Samba [50]) to isolate behaviors of the base architectures.

##### 7.1 Summary on Linear Models vs. Canon Layers

While many more linear-time architectures remain worth exploring, this study focuses on GLA, Mamba2, and GDN.20 Despite their structural differences, several consistent insights emerge.

|Result 9 (Section 6+Figure 14). Summary of Canon effects on linear models:<br><br>• Universality. Canon-ACD already matches internal conv1d, showing that horizontal mixing is useful across all sublayers, not limited to linear attention (i.e., the recurrent / SSM layer).<br>• Robustness. Adding Canon layers never hurts; the residual design stabilizes training.<br>• Sufficiency. Most performance appears achievable with the simplest GLA+Canon-AbCD, suggesting the current direction of linear-model architecture design may warrant re-evaluation.<br>|
|---|

To elaborate more on the third bullet, modern models (Mamba2, GDN) show only marginal gains over the simple GLA+Canon-AbCD baseline. This suggests that many recent architectural

20GDN results were newly added after the NeurIPS 2025 accepted version (V1.1).

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=8, k=8/4) Llama(NoPE) - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - Canon-AbCD(res)

###### Task Depo1(K=8, k=8/4) GLA - Canon-AbCD(res)

###### Task Depo1(K=8, k=8/4) GDN - Canon-AbCD(res)

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

99/100% 97/100% 99/100% 100/100% 98/100% 92/99% 95/100% 95/100%

99/100% 99/100% 99/100% 100/100% 96/99% 99/100% 99/100% 99/100% 99/100% 99/100% 98/100% 99/100%

9/50% 12/49% 6/37% 28/76% 3/22% 1/15% 1/16% 21/56% 0/2% 0/1% 0/5% 2/25%

7/53% 8/39% 44/87% 39/86% 0/7% 3/34% 4/43% 7/61% 0/3% 0/2% 1/9% 5/49%

26/72% 47/84% 28/71% 52/85% 28/65% 22/66% 14/55% 52/83%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

75/99% 97/100% 85/100% 90/100%

1/7% 2/19% 3/30% 15/51%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - Canon-AbCD(res)

###### Task Depo1(K=4, k=4/2) GLA - Canon-AbCD(res)

###### Task Depo1(K=4, k=4/2) GDN - Canon-AbCD(res)

[Figure 155]

[Figure 156]

[Figure 157]

51/87% 21/80% 67/96% 87/99% 11/45% 19/61% 15/63% 34/78% 8/46% 16/75% 6/53% 4/52%

19/75% 37/91% 22/85% 50/96% 7/49% 15/76% 11/81% 39/83% 5/28% 10/59% 7/56% 6/55%

74/96% 48/92% 69/94% 91/98% 12/58% 14/75% 17/72% 83/95% 25/73% 47/87% 21/68% 50/83%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) Mamba2(mlp) - Canon-AbCD(res)

###### Task Depo2(K=16, k=16/8) GLA - Canon-AbCD(res)

###### Task Depo2(K=16, k=16/8) GDN - Canon-AbCD(res)

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

92/100% 100/100% 97/100% 99/100% 97/100% 99/100% 96/100% 97/100% 85/100% 99/100% 98/100% 98/100%

96/100% 85/99% 86/100% 99/100% 94/100% 86/99% 99/100% 99/100% 90/100% 98/100% 93/100% 96/100%

2/1% 1/19% 2/15% 1/1%

2/1% 1/5% 1/1% 1/14% 1/1% 1/1% 1/1% 2/15% 1/1% 1/1% 1/1% 1/1%

1/5% 30/73% 5/40% 40/83%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/15% 1/2% 1/2%
- 1/1% 2/21% 1/1% 1/1%

- 1/2% 1/3% 1/6% 1/7%
- 1/3% 1/1% 1/7% 2/22%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Depo2(K=4, k=4/2) Mamba2(mlp) - Canon-AbCD(res)

Task Depo2(K=4, k=4/2) GLA - Canon-AbCD(res)

Task Depo2(K=4, k=4/2) GDN - Canon-AbCD(res)

[Figure 163]

[Figure 164]

[Figure 165]

69/93% 93/98% 87/97% 87/96% 44/83% 84/95% 66/91% 69/94% 23/52% 77/93% 24/63% 67/88%

67/89% 83/96% 77/98% 89/98% 42/77% 46/85% 63/96% 73/92% 35/71% 13/79% 46/93% 69/91%

96/99% 98/100% 95/99% 96/100% 89/99% 95/98% 92/99% 94/99% 66/94% 93/99% 84/96% 89/96%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

Task Brevo1 Llama(NoPE) - Canon-ABCD(res)

###### Task Brevo1 Mamba2(mlp) - Canon-AbCD(res)

Task Brevo1 GLA - Canon-AbCD(res)

Task Brevo1 GDN - Canon-AbCD(res)

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

83.7% 93.8% 91.3% 96.5% 62.9% 84.5% 81.2% 90.7% 47.9% 82.2% 69.7% 84.5%

84.8% 94.4% 91.1% 96.2% 63.9% 85.8% 75.5% 92.2% 42.0% 75.3% 58.2% 84.9%

13.8% 78.7% 32.4% 92.7% 0.6% 17.2% 21.8% 56.3% 0.6% 38.4% 3.6% 40.6%

88.0% 96.3% 95.3% 97.4% 83.3% 90.6% 89.7% 93.2% 65.5% 68.2% 79.4% 81.9%

95.8% 96.7% 95.7% 96.8% 91.4% 95.0% 91.2% 86.3% 83.1% 89.6% 80.3% 88.3%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

Task Brevo2 Llama(NoPE) - Canon-ABCD(res)

###### Task Brevo2 Mamba2(mlp) - Canon-AbCD(res)

Task Brevo2 GLA - Canon-AbCD(res)

Task Brevo2 GDN - Canon-AbCD(res)

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

87.1% 95.6% 92.2% 97.1% 75.4% 87.7% 80.1% 93.5% 55.1% 82.5% 69.3% 88.1%

87.4% 93.2% 89.0% 96.1% 61.2% 84.0% 75.2% 91.7% 40.4% 56.0% 56.3% 79.9%

53.8% 87.1% 65.0% 68.1% 14.7% 27.5% 19.3% 10.7% 0.8% 20.9% 0.9% 3.0%

91.1% 96.7% 94.8% 97.3% 71.1% 89.6% 78.5% 88.1% 46.8% 67.5% 50.9% 68.0%

98.0% 97.6% 98.4% 98.6% 92.6% 94.0% 93.4% 96.3% 87.4% 86.4% 86.6% 90.6%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Capo - Llama(RoPE) - Canon-ABCD(res)

###### Task Capo - Llama(NoPE) - Canon-ABCD(res)

###### Task Capo - Mamba2(mlp) - Canon-AbCD(res)

###### Task Capo - GLA - Canon-AbCD(res)

Task Capo - GDN - Canon-AbCD(res)

|6-6<br><br>4-2<br><br>6-3<br><br>5-4 3-6 5-6<br><br>2-3<br><br>2-8<br><br>4-2<br><br>12-6<br><br>8-2<br><br>8-610-6<br><br>2-3<br><br>3-6<br><br>4-3<br><br>5-4<br><br>6-2<br><br>2-8 6-6<br><br>5-3<br><br>5-6 8-8<br><br>5-3 4-3<br><br>3-6<br><br>6-3<br><br>12-6 10-8 6-6<br><br>8-6 8-8 5-6<br><br>10-6<br><br>2-8<br><br>5-4<br><br>2-2016-8<br><br>4-20<br><br>3-6<br><br>5-6<br><br>2-8<br><br>10-8 2-2016-83-20 8-166-20 10-6<br><br>8-6<br><br>8-16<br><br>6-6<br><br>4-20 6-2012-166-24 8-2420-16<br><br>10-8<br><br>2-20<br><br>8-16<br><br>3-20<br><br>8-8 12-6<br><br>4-20<br><br>16-8<br><br>6-20<br><br>12-166-24 8-2420-1624-1612-2420-2024-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|4-2<br><br>4-3<br><br>5-36-3 5-4<br><br>3-6 2-8 5-6<br><br>8-2<br><br>6-6<br><br>2-3<br><br>4-2<br><br>6-2<br><br>5-6 5-4<br><br>3-6 12-6 8-8 2-8<br><br>4-3<br><br>8-2<br><br>6-6 8-610-6<br><br>8-8 5-6<br><br>6-3<br><br>4-3<br><br>12-6 10-8 2-20<br><br>2-8<br><br>5-3<br><br>5-4<br><br>6-6 8-610-6 16-8<br><br>12-6<br><br>3-6 2-8<br><br>5-6<br><br>8-8 2-20<br><br>10-8 16-83-20 4-20 8-166-20 10-6<br><br>6-6<br><br>4-20 6-20<br><br>8-6<br><br>8-16 12-166-24 3-20<br><br>8-2420-16<br><br>8-8<br><br>2-20 10-8<br><br>12-6<br><br>4-20<br><br>16-8<br><br>8-16 3-20<br><br>6-2012-166-24 8-2420-1624-1612-2420-2024-20<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-3 5-4 3-6 2-8 5-66-6<br><br>4-2<br><br>2-3<br><br>6-2<br><br>8-2<br><br>5-3<br><br>3-6 2-8 5-66-6 8-610-612-68-8 5-3<br><br>4-3<br><br>6-3<br><br>5-4<br><br>6-6 8-6 2-8<br><br>5-6 10-612-68-8 10-8 2-2016-8<br><br>12-68-8<br><br>3-6<br><br>2-8<br><br>5-6<br><br>10-6<br><br>10-8 2-2016-83-20 4-20 8-166-20<br><br>10-6 8-6<br><br>16-8<br><br>6-6<br><br>3-20 4-20 8-166-2012-166-24 8-2420-16<br><br>12-166-24 8-24 24-1612-2420-2024-20<br><br>2-20 12-6<br><br>16-83-20<br><br>8-16<br><br>8-8<br><br>6-20<br><br>10-8<br><br>4-20 20-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|5-36-3 3-6 2-8 5-66-6 4-3<br><br>5-4<br><br>6-2 4-2<br><br>5-4<br><br>8-2<br><br>2-3<br><br>3-6 2-8 5-66-6 8-610-612-6 8-8<br><br>6-3<br><br>5-4<br><br>4-35-3<br><br>2-8 5-66-6 8-610-6<br><br>3-6 12-6 8-810-8 2-20 16-8<br><br>2-8<br><br>12-6<br><br>5-6<br><br>10-6 8-810-8 2-20 16-83-20 4-20 8-166-20<br><br>8-8<br><br>6-6<br><br>2-20 16-8<br><br>8-6<br><br>10-6<br><br>3-20 4-20 8-166-2012-166-24 8-2420-16<br><br>12-16 24-1612-2420-2024-20<br><br>8-8 10-8<br><br>12-6<br><br>16-8 4-20<br><br>2-20<br><br>3-20 8-166-20 6-24 8-2420-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

|6-3 5-4 3-6<br><br>4-35-3 2-8 5-66-6 6-2<br><br>8-2<br><br>4-2<br><br>5-4 3-6 2-8 5-66-6 8-610-612-6 8-8<br><br>2-3<br><br>8-8 3-6<br><br>8-610-612-6 10-8<br><br>6-3<br><br>6-6<br><br>5-4<br><br>5-6 2-8<br><br>2-20<br><br>5-3<br><br>16-8<br><br>4-3<br><br>2-8 3-6<br><br>5-6<br><br>8-610-612-68-8 10-8 2-2016-83-20 4-20 8-166-20<br><br>8-6 6-6<br><br>10-6<br><br>8-8<br><br>10-8 2-2016-83-20 4-20 8-166-2012-166-24 8-2420-16<br><br>20-2024-20<br><br>12-6<br><br>8-8<br><br>3-20 8-166-20 24-1612-24<br><br>2-20<br><br>4-20 16-8<br><br>10-8<br><br>12-166-24 8-2420-16<br><br>2 bit / param 1 bit / param 0.5 bit / param<br><br>0.25 bit / param<br><br>N=2000000 N=1000000<br><br>N=500000 N=200000 N=100000 N=50000<br><br>|
|---|

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

learnedknowledge(bits)

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

- 106
- 107
- 108

106 107 108

106 107 108

106 107 108

106 107 108

106 107 108

model size (#params)

model size (#params)

model size (#params)

model size (#params)

model size (#params)

Task Mano Llama(RoPE) - Canon-ABCD(res)

Task Mano Llama(NoPE) - Canon-ABCD(res)

Task Mano Mamba2(mlp) - Canon-AbCD(res)

Task Mano GLA - Canon-AbCD(res)

Task Mano GDN - Canon-AbCD(res)

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

94.2% 98.0% 99.2% 99.6% 89.8% 88.5% 98.2% 99.2% 83.7% 83.6% 88.8% 85.3%

97.7% 98.9% 99.3% 99.3% 83.1% 90.1% 95.9% 98.1% 53.7% 55.5% 89.4% 94.3%

- 97.8% 99.5% 99.6% 99.4%
- 98.6% 86.6% 98.7% 98.0% 96.1% 83.1% 96.6% 99.0%

96.2% 99.4% 79.7% 97.8% 95.5% 95.1% 59.9% 84.2% 84.2% 72.7% 55.1% 65.9%

99.7% 98.3% 99.3% 98.1% 90.3% 87.1% 98.7% 95.5% 93.1% 87.8% 80.3% 89.3%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(NoPE) - Canon-ABCD(res)

Task Lano Mamba2(mlp) - Canon-AbCD(res)

Task Lano GLA - Canon-AbCD(res)

Task Lano GDN - Canon-AbCD(res)

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

95.2% 97.5% 96.0% 98.1% 81.4% 90.1% 85.9% 92.6% 66.0% 77.9% 76.1% 78.9%

87.9% 91.9% 88.5% 92.5% 55.1% 70.3% 58.6% 78.3% 33.5% 51.0% 37.2% 53.1%

88.3% 93.1% 90.1% 93.6% 71.3% 80.0% 72.1% 80.8% 44.1% 61.1% 49.3% 64.4%

88.5% 92.8% 90.0% 94.1% 68.7% 84.6% 78.1% 84.3% 53.5% 64.9% 54.8% 73.3%

90.6% 93.6% 92.8% 95.1% 69.7% 86.2% 82.4% 87.1% 49.6% 68.3% 62.9% 75.2%

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- Figure 14: Final comparison of base architectures equipped with full-score Canon layers: RoPE(ˇ “), NoPE, Mamba2, GLA and GDN. Most notably, with Canon layers added, Mamba2/GLA/GDN still underperform Transformers by 2× in reasoning depth, with meaningful results only for Depo(K=4).

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

innovations may largely replicate Canon-like horizontal mixing rather than introduce fundamentally new computation. While such mechanisms can reduce explicit reliance on Canon layers, their improvements remain limited—raising the question of whether increasing architectural complexity truly expands capability or merely redistributes existing functions.21

- Remark 7.1. We do not claim that “replicating” Canon is unworthy—such designs may improve efficiency and reduce GPU memory. However, it is crucial to understand what the model actually learns: complex module designs need not realize complex functions, as optimizers may often converge to simpler functions (e.g., Canon-like solutions in this case).22

- 21In our follow-up work [2], we show that Canon layers can lift GLA to match GDN (+ full Canon) even on 1Bto 8B-sized models pretrained using real-life data, further strengthening this point.
- 22The same holds broadly in deep learning: although an ℓ-layer quadratic MLP can represent a 2ℓ-degree parity function, learning it is computationally intractable. Existence rarely implies learnability via training [4].

##### 7.2 Summary on Transformer vs. Linear Models

We now compare Transformers and linear models under a controlled, apple-to-apples setting with full Canon layers added to all architectures.

ˇ “

|Result 10 (Figure 14). With full-score Canon layers added, we find:<br><br>• reasoning depth: RoPE( ) ≈ NoPE ≫ Mamba2 ≈ GLA ≈ GDN (e.g., 4× deeper reasoning);<br>• reasoning breadth: GDN ≥ RoPE( ) ≈ NoPE ≈ GLA > Mamba2;<br>• knowledge capacity: Mamba2 ≈ GLA ≈ GDN ≫ RoPE( ) ≈ NoPE (e.g., 1.4× capacity);<br>• knowledge manipulation: GDN ≈ Mamba2 ≈ RoPE( ) ≥ NoPE ≈ GLA;<br>• hierarchical structure: RoPE( ) > NoPE ≈ Mamba2 ≈ GLA ≈ GDN.<br>|
|---|

ˇ “

ˇ “ ˇ “

ˇ “

- Remark 7.2. The initial comparison (Figure 4) was not controlled: Mamba2 and GDN included internal conv1d layers, whereas GLA and Transformers did not. By adding full Canon (CanonABCD or -AbCD) layers to all, the comparison becomes scientifically meaningful.

While others may interpret the fine differences across architectures, we focus here on the most pronounced contrasts. First, linear models—regardless of design—consistently show a ∼40% gain in Capo knowledge capacity compared to full Transformers. This is intuitive: their recurrent structure better supports associative-memory representations (an existential proof), and more importantly, optimizers can learn such representations effectively in practice.

More surprising is the behavior on reasoning depth. Linear models remain systematically weaker—about 2× on Depo1 (8-hop vs. 4-hop) and up to 4× on Depo2 (16-hop vs. 4-hop)—even under identical training conditions. We next examine this phenomenon in detail.

Deep Dive into Deep Reasoning for Linear-Time Models. We find that, due to compression of in-context knowledge, linear models struggle to reach 99% accuracy even on simple 1- or 2-hop retrievals (Figure 15), despite extended training. When reasoning depth exceeds 2 hops, early-step errors compound rapidly, preventing successful deep reasoning. In contrast, Transformers—especially with Canons—achieve near-perfect 1- and 2-hop accuracy very quickly (Figure 15).

Importantly, this weakness is not due to insufficient recurrent memory. For instance, in Mamba2, each layer passes 128d parameters (expansion × ssm state size × hidden size d)—hundreds of times more than sufficient to store the full input sequence.23 Moreover, Mamba2 performs well on 1-hop tasks (K=1) even with a single layer, confirming the bottleneck is not information-theoretic (a finding also to be reinforced in Section 8).

The same pattern holds for GLA and GDN, whose per-layer recurrent states (64d–144d) also provide ample capacity to store entire contexts (see Appendix C for architecture specifications). Hence, the true limitation lies in memory dynamics—how efficiently in-context information is encoded during compression and how reliably it is retrieved for reasoning. Errors in encoding or retrieval accumulate across hops, severely degrading multi-hop reasoning.

These results expose the Achilles’ heel of current linear architectures and point to a concrete direction for future research: improving the fidelity of compressed in-context memory. Until such limitations are resolved, hybrid approaches that combine sliding-window attention (for deep reasoning) with linear or state-space components (for long-context compression) remain the most practical path forward.

23In Task Depo, representing N key-value pairs with vocabulary V requires at most 2N log2 V bits. For Depo2, with N=75 and V ≤2500, this is under 1700 bits, compared to Mamba2’s recurrent state of 12 × 128 × 768≈1.2M 32-bit floats. This occupies ∼0.001 bits per float; in contrast, long-term (factual) memory in weights can reach 2 bits per float (see [8] our Task Capo).

Task Depo2(K=4) | N=125 | Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=4) | N=100 | Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=4) | N=75 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 Train steps (model size = 12L768D)

Task Depo2(K=4) | N=75 | Mamba2(mlp) - Canon-AbCD(res)

Task Depo2(K=4) | N=125 | Mamba2(mlp) - Canon-AbCD(res)

Task Depo2(K=4) | N=100 | Mamba2(mlp) - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

Task Depo2(K=4) | N=75 | GLA - Canon-AbCD(res)

Task Depo2(K=4) | N=125 | GLA - Canon-AbCD(res)

Task Depo2(K=4) | N=100 | GLA - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

Task Depo2(K=4) | N=75 | GDN - Canon-AbCD(res)

Task Depo2(K=4) | N=125 | GDN - Canon-AbCD(res)

Task Depo2(K=4) | N=100 | GDN - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

- Figure 15: Training curves for 12L768D architectures on Depo2(K=4), evaluated at k = 1, 2, 4 and n = N, with results shown across two best LRs for each k. Results for other data are in Figure 20 on Page 42.

|Result 11 (Figure 15). Linear models such as Mamba2/GLA/GDN struggle with deep reasoning—not from lack of memory, but from accumulated errors in compression and retrieval. Hybrid models combining Transformers and linear layers, equipped with Canon, mitigate these limitations.|
|---|

#### 8 Real-Life Experiments

We conduct real-life pretraining at the academic scale: 1.3B-parameter models trained on 100B tokens from FineWeb-Edu [42] and SlimPajama [60], using a 4096 context length (details in Appendix B). This mirrors setups common in recent studies such as Titans [10], GDN [73], and MTA [25], representing the standard academic pretraining paradigm.

Evaluation suites. We first evaluate all models on two benchmark suites. The first, based on lm-evaluation-harness [24], covers discriminative tasks: PIQA [12], HellaSwag [79], WinoGrande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]. Following prior work [10, 73], we adopt the original accuracy metrics for consistency.24

The second generative-task suite uses the Just Read Twice (JRT) protocol [9], designed to reduce noise in generative testing at this scale.25 Tasks include SWDE, FDA, SQuAD(v2) [49], TriviaQA [34], NQ [38], and DROP [21], plus their JRT-enhanced variants (denoted as FDA2, SWDE2, etc.) We again follow the official JRT codebase for evaluation.

Key observations across both suites. Results show large variance across random seeds:

• Benchmark scores fluctuate with random seeds—up to 4% on LAMBADA, 3% on BoolQ, and 1–3% elsewhere. Generative tasks vary even more (9% on FDA, 8% on SWDE, 3–5% on others). The same holds even if data shuffling is fixed and model init varies (Appendix E.1).

Hence, only differences beyond these thresholds are statistically meaningful. From Figure 16:

24Following tradition [10, 72, 73], we use (acc n) for HellaSwag and ARC-c, but acc n for other tasks. 25Generative testing can be noisy at this scale, as such models often struggle with prompt comprehension. JRT

addresses this by repeating the context and question twice, allowing models to more accurately reveal their intrinsic generative capabilities.

[Figure 186]

[Figure 187]

Result 12: Real-life pretraining at academic scale “noise” from random seeds NoPE or RoPE(♩) helps length gen.

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

|[Figure 194]<br><br>GLA + Mamba2 (no conv1d)<br><br>+ NoPE perform the worst|
|---|

[Figure 195]

[Figure 196]

|[Figure 197]<br><br>GLA+Canon performs no<br><br>worse than Mamba2/GDN|
|---|

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Physics says: Beyond this is noise.

[Figure 205]

|[Figure 206]<br><br>Linear models struggle (even short-context) retrieval tasks|
|---|

simplest 2-hop reasoning fails at academic scale

We stop here to avoid over-interpretation

how to interpret this table

###### SlimPajama | 100B token pretrain | 1.3B models

- Llama(RoPE) - original - seed 20

- Llama(RoPE) - original - seed 21

- Llama(RoPE) - original - seed 22

- Llama(RoPE) - original - seed 23

- Llama(RoPE) - original - seed 24

- Llama(RoPE) - original - seed 25

- Llama(RoPE) - original - seed 26

- Llama(RoPE) - original - seed 27

[Figure 207]

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(NoPE) - original - seed 20

Llama(NoPE) - Canon-ABCD(res) - seed 20

Mamba2(mlp) - original(conv1d) - seed 20

Mamba2(mlp) - noconv1d - seed 20

Mamba2(mlp) - Canon-AbCD(res) - seed 20

Mamba2(mlp) - Canon-ABCD(res) - seed 20

Mamba2 - original(conv1d) - seed 20

Mamba2 - noconv1d - seed 20

Mamba2 - Canon-Ab(res) - seed 20

Mamba2 - Canon-AB(res) - seed 20

GLA - original(noconv1d) - seed 20

GLA - conv1d - seed 20

GLA - Canon-AbCD(res) - seed 20

GDN - original(conv1d) - seed 20

GDN - noconv1d - seed 20

- 52.1% 73.2% 57.6% 59.0% 61.3% 29.3% 40.4% 61.3% 15.4 9.9 64.9% 78.0% 50.7% 54.5% 31.1% 38.5% 39.0% 60.2% 61.7% 62.0% 25.0% 38.5% 83.8% 74.3% 56.7% 42.9% 32.8% 22.6% 16.1% 31.8% 33.2% 29.9% 28.5% 25.5% 21.2% 18.0%

- 50.6% 72.3% 56.7% 56.5% 60.5% 29.8% 42.4% 62.5% 14.4 10.7 65.0% 79.2% 48.8% 57.6% 31.0% 39.0% 39.6% 58.6% 61.1% 61.8% 23.9% 39.2% 79.0% 67.6% 52.8% 41.5% 34.4% 25.8% 19.0% 30.1% 29.0% 26.4% 26.5% 20.7% 19.1% 16.6%
- 51.1% 72.2% 55.8% 58.5% 61.0% 29.6% 41.8% 61.7% 14.8 10.7 60.2% 75.8% 52.1% 55.6% 31.0% 38.3% 39.4% 60.0% 61.8% 62.6% 23.9% 40.4% 68.8% 59.4% 47.0% 36.4% 28.9% 17.9% 12.6% 31.8% 29.0% 28.2% 28.6% 24.8% 20.5% 16.7%

53.8% 73.4% 58.0% 59.2% 61.4% 31.2% 42.1% 62.9% 15.2 9.4 66.1% 79.6% 51.5% 55.7% 33.0% 38.7% 40.0% 60.7% 62.3% 62.2% 22.7% 40.3% 96.0% 90.8% 76.1% 64.9% 50.6% 38.7% 31.3% 33.2% 32.2% 28.2% 27.5% 24.8% 22.1% 17.7%

- 50.0% 71.8% 54.1% 57.1% 58.8% 29.8% 40.6% 61.0% 15.3 10.9 61.6% 77.7% 50.6% 59.0% 30.3% 35.5% 38.7% 61.2% 59.8% 59.0% 23.1% 36.7% 84.0% 74.2% 51.3% 35.6% 24.0% 18.4% 11.8% 35.7% 33.6% 27.8% 28.5% 25.7% 22.5% 16.8%

44.2% 71.2% 47.9% 54.0% 55.2% 27.4% 40.3% 61.5% 17.1 15.8 58.5% 73.4% 42.4% 56.7% 27.2% 35.0% 35.0% 59.1% 55.7% 55.6% 22.2% 38.6% 71.6% 55.0% 30.5% 18.4% 12.1% 10.4% 9.2% 32.0% 29.6% 24.3% 23.6% 20.9% 15.6% 13.6%

- 50.6% 72.5% 55.5% 57.1% 60.0% 29.4% 40.9% 60.6% 15.7 10.7 68.2% 79.3% 50.0% 56.0% 29.8% 36.8% 38.8% 56.7% 60.1% 59.3% 22.2% 37.1% 75.0% 63.8% 42.5% 26.8% 19.0% 12.8% 8.7% 33.1% 32.5% 29.4% 27.9% 24.5% 16.1% 11.8%
- 51.6% 73.3% 57.2% 57.3% 61.0% 30.1% 41.6% 62.5% 15.5 10.3 64.9% 77.7% 50.1% 59.0% 32.1% 39.3% 37.6% 58.4% 60.9% 61.4% 23.3% 37.0% 80.5% 64.3% 38.8% 28.1% 17.3% 11.7% 8.0% 34.4% 33.1% 32.0% 29.9% 26.6% 17.9% 13.9%

42.4% 70.5% 49.2% 54.8% 54.7% 27.2% 40.7% 60.3% 18.6 16.8 48.2% 64.4% 44.0% 49.1% 27.1% 33.0% 33.6% 56.5% 55.4% 57.3% 20.9% 34.1% 49.2% 34.6% 21.3% 15.4% 12.6% 9.8% 8.3% 33.0% 29.1% 28.6% 25.2% 21.0% 14.4% 10.1%

- 49.5% 72.6% 57.0% 57.1% 60.5% 29.1% 41.0% 60.7% 15.5 11.9 56.4% 74.3% 48.2% 55.7% 30.9% 38.4% 38.3% 58.6% 60.0% 59.8% 24.3% 38.6% 75.4% 64.8% 40.4% 27.8% 16.6% 12.0% 9.4% 34.0% 33.9% 31.4% 34.3% 26.8% 23.1% 20.8%
- 50.3% 72.7% 55.9% 57.2% 60.3% 29.5% 42.0% 61.1% 15.7 10.9 69.8% 77.9% 46.8% 57.3% 30.6% 38.5% 38.5% 57.3% 61.0% 61.4% 21.5% 37.4% 72.7% 54.8% 37.0% 31.9% 25.9% 17.2% 13.5% 35.6% 31.9% 28.1% 28.8% 25.1% 17.2% 15.1%

48.8% 73.9% 56.6% 59.2% 59.8% 30.5% 41.8% 62.5% 15.7 11.7 63.6% 75.8% 46.1% 53.7% 30.4% 36.8% 38.3% 58.5% 60.4% 61.5% 22.4% 36.3% 62.3% 50.5% 34.8% 24.4% 18.3% 12.7% 8.5% 34.8% 32.8% 32.6% 31.2% 25.4% 19.5% 14.5%

- 42.8% 71.9% 50.9% 55.2% 55.3% 27.8% 41.2% 61.1% 17.7 15.6 54.3% 62.6% 38.4% 50.7% 26.8% 32.2% 34.4% 56.4% 55.5% 58.4% 21.3% 38.1% 67.9% 52.9% 37.4% 26.9% 18.2% 14.5% 9.0% 33.1% 31.5% 24.9% 24.7% 18.8% 13.3% 10.0%

48.5% 72.5% 56.6% 57.8% 59.2% 30.0% 40.9% 62.1% 15.8 11.8 56.4% 72.1% 45.2% 51.0% 31.2% 36.1% 37.7% 59.7% 59.7% 60.2% 21.4% 37.8% 75.7% 65.2% 42.8% 30.7% 23.7% 15.5% 12.8% 34.3% 32.8% 28.6% 29.1% 21.7% 15.8% 11.6%

48.8% 72.1% 55.7% 58.6% 58.1% 29.3% 41.6% 62.8% 15.6 12.9 77.7% 86.5% 56.5% 59.9% 36.1% 40.2% 49.1% 58.3% 63.7% 59.8% 26.8% 40.9% 99.3% 99.4% 98.9% 98.8% 99.4% 88.9% 10.1% 36.1% 34.8% 35.0% 33.4% 34.3% 30.5% 2.6%

- 43.8% 70.3% 49.2% 54.5% 55.1% 27.3% 39.0% 60.4% 18.1 21.3 78.6% 84.8% 55.7% 63.4% 32.9% 35.9% 45.1% 53.6% 59.1% 55.7% 23.7% 38.3% 98.8% 98.8% 99.0% 98.5% 98.6% 66.8% 1.7% 28.9% 28.5% 25.5% 29.1% 28.4% 18.8% 1.6%

52.0% 72.4% 56.5% 60.4% 58.4% 29.0% 40.9% 62.8% 15.4 11.0 78.6% 86.9% 54.8% 58.4% 34.1% 35.0% 44.6% 56.2% 63.8% 60.6% 24.2% 39.6% 98.9% 99.1% 98.0% 95.6% 91.3% 7.0% 1.2% 34.4% 34.2% 33.3% 31.4% 30.8% 6.2% 1.2%

- 51.4% 71.7% 56.5% 58.9% 60.1% 29.9% 40.8% 63.2% 15.4 10.4 78.6% 87.2% 57.3% 61.5% 35.4% 40.0% 46.2% 59.0% 65.5% 61.7% 25.8% 41.9% 99.7% 99.8% 99.1% 98.6% 95.3% 60.8% 18.6% 35.0% 32.3% 33.0% 30.6% 34.5% 29.1% 10.4%

51.7% 71.7% 57.2% 57.8% 60.9% 29.4% 41.0% 63.1% 15.2 11.1 80.1% 87.4% 56.5% 62.6% 35.2% 39.4% 46.3% 57.1% 63.9% 61.4% 26.8% 42.3% 99.2% 99.6% 99.7% 99.3% 99.6% 80.6% 35.8% 34.2% 33.0% 32.5% 29.5% 31.5% 27.9% 25.1%

50.8% 72.3% 55.8% 57.2% 59.1% 29.6% 41.1% 63.5% 15.7 11.2 79.6% 85.6% 50.7% 59.4% 32.5% 36.0% 44.4% 55.2% 62.5% 60.0% 25.2% 40.8% 98.8% 96.6% 90.7% 84.0% 73.1% 14.8% 0.4% 30.5% 31.0% 30.5% 28.5% 29.3% 14.9% 0.2%

55.0% 72.0% 56.0% 57.7% 58.9% 29.1% 41.1% 62.3% 15.7 9.8 80.3% 85.2% 52.1% 60.8% 35.1% 36.7% 45.3% 59.7% 63.4% 59.7% 23.8% 39.4% 99.2% 98.8% 94.6% 94.2% 80.5% 5.1% 0.3% 31.1% 30.1% 29.7% 24.6% 26.1% 6.7% 0.0%

52.3% 71.6% 56.7% 58.2% 59.4% 31.0% 42.1% 62.9% 15.8 10.3 81.3% 87.4% 54.8% 59.4% 32.7% 36.2% 46.3% 54.4% 63.7% 61.4% 25.1% 41.1% 98.9% 98.0% 93.5% 89.3% 78.0% 4.4% 0.4% 29.2% 29.9% 28.4% 28.5% 27.7% 8.9% 0.2%

- 52.8% 72.1% 56.0% 58.8% 58.5% 29.0% 41.0% 60.7% 15.7 10.2 76.2% 85.5% 50.9% 59.7% 34.1% 37.5% 48.9% 57.3% 63.2% 61.4% 25.6% 39.1% 99.3% 98.8% 96.2% 92.6% 79.1% 9.7% 0.4% 31.8% 31.9% 30.8% 30.1% 27.2% 7.6% 0.2%
- 53.1% 71.9% 56.5% 59.2% 60.8% 29.2% 41.3% 61.8% 15.7 9.9 81.5% 87.2% 52.6% 62.2% 34.3% 36.6% 45.5% 55.4% 62.3% 59.8% 22.3% 42.8% 97.4% 95.5% 93.0% 84.4% 80.3% 8.5% 0.3% 31.2% 30.1% 30.5% 26.5% 26.4% 10.1% 0.2%

50.7% 72.1% 56.0% 57.9% 59.5% 29.6% 40.8% 61.1% 15.7 11.3 78.7% 85.4% 53.8% 63.5% 33.3% 34.4% 45.6% 57.3% 62.9% 60.3% 24.9% 38.6% 99.2% 96.6% 96.2% 91.1% 80.5% 9.4% 0.7% 30.4% 30.1% 28.4% 25.4% 26.9% 13.5% 0.4%

- 52.3% 71.2% 56.1% 58.7% 59.3% 29.0% 40.9% 60.8% 15.8 10.4 79.4% 84.6% 54.5% 61.9% 35.3% 36.0% 45.6% 56.4% 64.2% 59.1% 27.4% 39.3% 97.8% 98.7% 98.6% 95.7% 87.0% 7.3% 1.2% 31.0% 29.5% 27.8% 23.6% 23.8% 12.1% 1.3%
- 53.3% 71.4% 56.2% 58.5% 59.1% 29.9% 40.9% 61.2% 15.7 10.1 79.0% 86.6% 52.8% 62.3% 33.5% 35.3% 43.9% 55.4% 61.4% 60.6% 25.0% 40.3% 99.3% 98.0% 95.6% 89.4% 80.8% 6.2% 0.2% 31.6% 31.4% 28.3% 26.9% 27.8% 6.1% 0.0%

GDN - Canon-AbCD(res) - seed 20

LMB PIQA Hella Wino ARC-e ARC-c SIQA BoolQWikipplLMBppl FDA FDA2 SWDESWDE2 NQ NQ2 SquadSquad2TriviaQATriviaQA2 Drop Drop2 1-hop-0k1-hop-1k1-hop-2k1-hop-3k1-hop-4k1-hop-5k1-hop-6k 2-hop-0k2-hop-1k2-hop-2k2-hop-3k2-hop-4k2-hop-5k2-hop-6k

FineWeb-Edu | 100B token pretrain | 1.3B models

48.1% 73.0% 59.3% 59.4% 72.6% 41.1% 42.1% 63.5% 16.7 13.1 71.7% 78.3% 49.7% 60.5% 29.2% 34.3% 44.0% 55.3% 64.3% 61.0% 24.3% 36.6% 98.9% 97.6% 90.5% 86.3% 65.4% 3.3% 0.7% 28.9% 29.5% 29.3% 26.3% 26.2% 3.3% 0.4%

- Llama(RoPE) - original - seed 20

- Llama(RoPE) - original - seed 21

- Llama(RoPE) - original - seed 22

- Llama(RoPE) - original - seed 23

- Llama(RoPE) - original - seed 24

- Llama(RoPE) - original - seed 25

- Llama(RoPE) - original - seed 26

- Llama(RoPE) - original - seed 27

[Figure 208]

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(NoPE) - original - seed 20

Llama(NoPE) - Canon-ABCD(res) - seed 20

Mamba2(mlp) - original(conv1d) - seed 20

Mamba2(mlp) - noconv1d - seed 20

Mamba2(mlp) - Canon-AbCD(res) - seed 20

Mamba2(mlp) - Canon-ABCD(res) - seed 20

Mamba2 - original(conv1d) - seed 20

Mamba2 - noconv1d - seed 20

Mamba2 - Canon-Ab(res) - seed 20

Mamba2 - Canon-AB(res) - seed 20

GLA - original(noconv1d) - seed 20

GLA - conv1d - seed 20

GLA - Canon-AbCD(res) - seed 20

GDN - original(conv1d) - seed 20

- 48.0% 74.1% 60.0% 58.9% 75.3% 42.4% 42.8% 64.3% 15.6 11.7 42.5% 57.3% 37.1% 48.0% 26.7% 33.2% 35.7% 61.5% 61.7% 64.4% 22.8% 38.0% 74.6% 65.3% 43.2% 29.4% 22.0% 12.9% 10.5% 33.5% 30.5% 26.2% 24.2% 20.0% 18.9% 13.1%

- 48.7% 72.6% 58.5% 59.8% 73.6% 40.0% 43.0% 63.7% 15.8 11.2 50.0% 61.1% 41.9% 46.7% 27.7% 34.0% 36.8% 60.2% 62.8% 62.7% 22.6% 39.0% 52.7% 43.0% 26.3% 15.1% 8.9% 7.0% 4.6% 32.8% 32.2% 26.7% 25.7% 19.9% 16.7% 13.2%

50.7% 74.0% 60.3% 61.7% 72.7% 40.4% 41.8% 64.7% 16.2 10.6 54.6% 71.0% 42.3% 51.2% 27.7% 33.4% 38.8% 60.4% 64.5% 62.6% 23.2% 38.4% 73.5% 62.4% 41.1% 22.0% 16.2% 10.8% 6.8% 35.0% 33.2% 26.3% 27.2% 21.7% 20.1% 14.9%

50.4% 73.3% 59.0% 60.4% 73.2% 40.7% 42.9% 63.7% 15.9 11.3 45.4% 61.8% 40.5% 50.9% 26.5% 32.4% 36.5% 60.1% 62.2% 63.3% 23.0% 37.3% 85.5% 77.3% 53.5% 35.2% 22.3% 13.9% 8.2% 31.7% 32.8% 24.1% 24.1% 16.9% 14.1% 10.8%

44.4% 72.5% 53.7% 56.4% 69.8% 37.0% 41.1% 63.3% 18.2 14.3 34.0% 49.4% 27.8% 42.1% 23.9% 28.7% 33.5% 60.4% 58.1% 62.9% 22.0% 35.3% 43.6% 30.9% 10.6% 4.9% 2.7% 2.5% 1.9% 31.2% 25.2% 19.8% 14.1% 10.9% 7.5% 4.9%

- 47.6% 73.4% 59.3% 59.3% 73.5% 39.2% 41.7% 62.6% 16.8 11.7 43.2% 51.4% 34.3% 42.8% 25.0% 31.9% 36.8% 58.9% 62.7% 62.1% 22.0% 36.9% 51.6% 35.2% 22.2% 16.9% 11.9% 7.8% 4.8% 32.6% 31.2% 23.7% 24.2% 18.1% 14.1% 10.7%

49.0% 73.4% 60.5% 58.4% 73.1% 41.1% 41.2% 63.8% 16.6 11.2 43.0% 51.3% 30.0% 40.2% 26.4% 31.5% 36.5% 56.8% 63.3% 64.0% 21.7% 35.9% 58.5% 47.0% 28.7% 20.5% 12.2% 7.5% 5.5% 30.9% 28.9% 26.4% 23.3% 16.7% 13.0% 9.3%

43.4% 72.0% 56.2% 55.0% 68.9% 36.3% 41.6% 62.0% 18.7 14.5 23.6% 30.7% 27.8% 33.5% 22.7% 28.1% 31.8% 56.2% 58.5% 61.7% 19.8% 33.0% 50.0% 32.6% 18.0% 9.1% 6.3% 4.6% 3.2% 30.3% 27.5% 22.9% 20.5% 14.2% 11.6% 9.9%

- 48.5% 73.6% 60.3% 60.3% 73.1% 41.3% 42.3% 63.5% 16.6 11.8 39.5% 49.6% 33.8% 42.7% 27.0% 31.3% 36.7% 57.0% 61.5% 62.7% 22.6% 38.7% 51.8% 38.4% 19.4% 13.1% 6.4% 5.5% 3.2% 33.4% 32.3% 24.8% 24.3% 20.8% 15.1% 13.6%

- 47.6% 73.1% 59.8% 57.9% 72.2% 39.1% 41.7% 62.4% 16.8 11.9 37.3% 50.7% 34.9% 43.3% 26.3% 32.7% 35.1% 58.7% 62.0% 64.4% 23.5% 37.1% 56.9% 33.5% 20.2% 14.2% 9.3% 6.5% 4.9% 31.6% 30.2% 23.0% 21.5% 16.3% 12.7% 10.1%

- 47.1% 74.0% 59.8% 59.9% 72.5% 39.8% 41.8% 64.7% 16.6 11.8 31.5% 44.3% 31.3% 43.2% 25.6% 30.7% 35.3% 57.3% 62.3% 63.6% 24.4% 38.6% 49.8% 35.1% 19.3% 12.5% 7.4% 5.1% 3.3% 32.7% 30.7% 25.6% 23.2% 16.6% 13.5% 8.6%

- 46.2% 73.1% 56.3% 57.3% 71.0% 37.5% 41.4% 63.5% 18.6 13.0 27.2% 33.1% 29.7% 37.4% 23.7% 28.0% 32.4% 54.1% 59.4% 63.4% 21.6% 36.8% 48.5% 28.3% 14.7% 7.0% 3.9% 3.5% 2.9% 30.5% 26.9% 21.9% 17.4% 10.9% 9.6% 5.1%
- 47.3% 74.3% 60.0% 59.6% 73.0% 40.4% 42.2% 64.7% 16.8 11.6 31.2% 42.4% 32.7% 37.7% 25.5% 31.6% 36.4% 58.9% 62.3% 63.8% 21.0% 38.8% 55.6% 37.0% 17.9% 10.3% 7.0% 6.2% 3.7% 31.9% 30.4% 24.4% 25.6% 17.9% 14.1% 10.5%

49.9% 73.9% 58.7% 58.4% 72.1% 38.1% 42.6% 64.2% 16.6 11.3 76.4% 83.7% 52.5% 57.4% 33.1% 37.5% 45.4% 58.4% 65.6% 60.6% 24.5% 38.0% 99.6% 99.1% 99.1% 98.6% 96.9% 85.5% 2.1% 34.4% 34.7% 34.2% 29.8% 35.7% 31.4% 12.8%

47.2% 71.9% 55.2% 57.2% 69.7% 38.6% 41.9% 62.8% 18.1 13.5 76.3% 84.5% 53.8% 55.2% 28.4% 32.6% 43.7% 55.5% 62.5% 57.9% 23.3% 37.8% 99.2% 99.0% 98.4% 97.6% 96.8% 75.6% 6.4% 33.8% 33.6% 31.4% 31.3% 33.9% 23.1% 1.2%

- 50.7% 73.0% 59.6% 60.2% 74.0% 41.5% 42.5% 65.1% 16.2 11.2 79.2% 82.7% 52.5% 59.2% 30.7% 36.3% 43.0% 54.8% 65.9% 63.5% 23.3% 38.8% 99.7% 99.1% 95.7% 89.2% 68.4% 4.9% 0.7% 31.1% 32.2% 31.7% 26.7% 28.3% 6.8% 0.7%
- 51.1% 72.5% 59.4% 60.3% 73.2% 40.4% 42.5% 63.9% 16.3 10.9 76.6% 83.0% 57.7% 60.6% 31.2% 37.5% 45.1% 61.4% 64.2% 63.9% 23.3% 39.4% 98.8% 98.0% 97.6% 96.6% 93.4% 81.1% 17.8% 34.4% 33.0% 33.0% 28.7% 32.0% 28.7% 9.8%

50.8% 73.2% 59.9% 60.5% 72.4% 41.5% 42.3% 64.6% 16.0 11.0 77.1% 84.7% 54.1% 57.1% 31.9% 37.3% 45.3% 56.4% 66.4% 63.9% 24.4% 38.7% 99.1% 99.1% 97.9% 97.0% 94.5% 69.8% 29.4% 33.1% 34.3% 32.1% 28.9% 29.5% 28.7% 21.7%

49.2% 73.2% 59.0% 58.8% 72.3% 39.1% 43.4% 64.6% 16.8 11.8 68.8% 81.5% 52.0% 56.8% 29.4% 31.2% 43.4% 57.2% 64.0% 61.8% 22.1% 36.3% 99.4% 98.6% 96.5% 90.9% 51.1% 5.6% 0.2% 32.0% 31.1% 28.5% 26.0% 26.0% 5.4% 0.2%

- 48.7% 72.9% 58.8% 59.0% 71.4% 37.9% 42.8% 63.6% 16.6 12.7 75.4% 82.1% 50.4% 56.9% 29.2% 34.0% 41.9% 56.1% 65.6% 62.8% 25.0% 37.6% 98.4% 97.8% 96.4% 91.3% 64.7% 1.6% 0.7% 30.8% 30.5% 28.6% 26.0% 25.5% 3.2% 0.3%

- 50.0% 73.8% 59.0% 59.7% 71.8% 38.7% 42.3% 62.5% 16.8 11.6 70.8% 77.8% 45.3% 52.3% 28.5% 33.9% 44.1% 56.3% 62.1% 61.9% 20.9% 34.5% 98.0% 93.4% 87.0% 81.6% 49.0% 3.3% 0.4% 31.2% 31.2% 33.0% 27.2% 29.3% 3.9% 0.4%

- 47.4% 73.4% 58.6% 59.6% 71.7% 40.4% 42.9% 63.2% 16.6 13.3 73.1% 73.4% 48.9% 54.7% 28.1% 31.5% 44.8% 55.3% 63.4% 61.3% 23.3% 38.2% 97.8% 93.4% 87.6% 80.9% 46.4% 7.5% 0.3% 30.2% 30.6% 30.7% 25.5% 27.4% 11.8% 0.0%
- 48.9% 72.5% 59.1% 59.0% 71.4% 41.1% 42.9% 62.3% 16.9 12.6 74.7% 79.7% 49.2% 54.7% 30.3% 33.7% 47.9% 58.2% 65.0% 61.8% 24.2% 37.4% 99.0% 97.4% 92.3% 84.7% 58.4% 5.4% 0.9% 31.2% 31.2% 30.5% 30.4% 27.7% 8.1% 0.5%

- 50.1% 72.7% 58.9% 60.5% 72.2% 41.0% 42.3% 63.5% 16.8 12.0 74.2% 76.8% 53.6% 52.6% 31.0% 34.1% 43.6% 57.4% 64.5% 63.6% 23.1% 37.3% 97.3% 96.9% 92.8% 88.7% 56.1% 2.7% 0.6% 30.8% 31.1% 28.9% 27.0% 25.9% 4.0% 0.2%

- 48.5% 72.7% 58.8% 60.7% 71.7% 39.8% 42.4% 63.4% 16.7 12.3 71.3% 82.7% 46.2% 56.5% 29.0% 33.0% 44.8% 55.1% 63.1% 62.5% 24.5% 34.9% 98.0% 96.2% 91.2% 83.2% 59.8% 7.0% 0.8% 32.5% 31.1% 28.7% 22.4% 24.7% 6.1% 0.3%

GDN - noconv1d - seed 20

50.4% 73.5% 60.0% 60.9% 73.6% 40.5% 42.3% 64.8% 16.6 10.7 50.0% 62.5% 40.3% 50.8% 26.9% 33.8% 37.8% 60.8% 63.4% 63.9% 22.3% 40.4% 48.5% 36.6% 20.6% 15.1% 9.2% 7.9% 6.2% 33.3% 28.9% 26.5% 25.4% 21.0% 17.1% 13.2%

GDN - Canon-AbCD(res) - seed 20

LMB PIQA Hella Wino ARC-e ARC-c SIQA BoolQWikipplLMBppl FDA FDA2 SWDESWDE2 NQ NQ2 SquadSquad2TriviaQATriviaQA2 Drop Drop2 1-hop-0k1-hop-1k1-hop-2k1-hop-3k1-hop-4k1-hop-5k1-hop-6k 2-hop-0k2-hop-1k2-hop-2k2-hop-3k2-hop-4k2-hop-5k2-hop-6k

- Figure 16: Performance of 1.3B models pretrained on 100B tokens across discriminative (left), generative (middle), and 1/2-hop reasoning (right) tasks. Best of 2 learning rates for Llama; 3 for GLA, Mamba, and GDN. GPT2 variants (e.g., squared ReLU) shown in Figure 22 on Page 44.

- • Linear models (Mamba2, GLA, GDN) underperform full Transformers on generative tasks, even for contexts shorter than training length.26 Retrieval-heavy tasks (FDA, SWDE) amplify this gap, consistent with Result 11.
- • NoPE, GLA, and Mamba2 (w/o conv1d) perform poorly in base form but improve markedly with full Canon. GLA+Canon surpasses Mamba2 and matches GDN (even with Canon); NoPE+Canon performs on par with RoPE. GDN is least sensitive to Canon yet not clearly stronger than GLA+Canon—consistent with Result 3, 6.1, 7.1, 8.1, and 9.
- • At this scale (1.3B/100B), RoPE, RoPE+Canon, and NoPE+Canon perform comparably, and most linear+Canon variants cluster together. Academic-scale pretraining cannot reliably distinguish finer architectural differences.

Needle, Babilong, and our Multi-Hop Reasoning Tasks. The Needle-in-a-Haystack (NIAH) task from RULER [29] tests recall of a “needle” value (e.g., a magic number) in long text. This makes it too easy: models—especially linear ones—may appear accurate while failing at most basic short-context retrieval (see later). For completeness, results are shown in Figure 23 (Page 45).

The Babilong dataset [37] embeds bAbi [69] tasks in long junk-filled passages to test multi-hop

26Generative task prompts are capped at 1024–2048 tokens (per original codebase), while training used 4096.

reasoning but proves overly difficult at this scale.27 As shown in Figure 23, Babilong results are mostly indistinguishable; only trends are clear:

- • Linear models underperform Transformers even on short contexts, confirming their weakness stems from inefficient compression and retrieval, not memory size (Result 11).
- • Transformers gain on longer contexts when RoPE is reduced (RoPEˇ “) or removed (NoPE), particularly in 4k-token junk passages (c.f. Result 3).

To balance NIAH’s simplicity and Babilong’s difficulty, we introduce our own multi-hop reasoning tasks. 1-hop-L embeds five birth-year statements within Wikipedia passages of length L, requiring direct recall of one of the birth years. 2-hop-L embeds three birth-year statements plus three equivalence links (e.g., “XXX was born the same year as YYY”), requiring inference of the linked names’ birth years. Details are in Appendix B. Results (Figure 16) show:

- • All models struggle with 2-hop-L, achieving only 30–36% (near random) even with L = 0.
- • 1-hop-L separates architectures: full Transformers outperform linear models even for L = 0 (short contexts < 100 tokens), while NoPE and RoPE(ˇ “) generalize better as L increases.

To summarize:

|Result 12 (Figure 16+23). Academic-scale pretraining (1.3B params, 100B tokens, 4k context) shows high noise and limited resolution, making most architectural differences statistically insignificant. Yet several consistent findings hold:<br><br>• Linear models (Mamba2, GLA, GDN) underperform full Transformers even on short-context retrieval tasks (FDA, SWDE, or 1-hop-L with ∼100 tokens), even with Canon (Result 11).<br>• Canon elevates NoPE to RoPE-level (Result 3), GLA to Mamba2/GDN-level (Result 6.1, 9); removing conv1d downgrades Mamba2 to GLA (Result 7.1) but hardly affects GDN (Result 8.1).<br>• All models fail 2-hop reasoning, even within 100 tokens, revealing limit of academic-scale pretrain.<br>• Reducing or removing RoPE (NoPE, RoPE ) improves long-context generalization (Result 3).<br>|
|---|

ˇ “

#### 9 Conclusion and Future Direction

Academic-scale pretraining suffers from high noise and failed multi-hop reasoning, hindering reliable architectural comparison. Our controlled synthetic playground offers a cost-effective, principled alternative: by decomposing intelligence into atomic tasks, we discover and optimize Canon layers—lightweight constructs that enhance reasoning depth and breadth, knowledge capacity and manipulation, and structural reasoning across diverse architectures.

Canon layers revive weaker models (e.g., NoPE, GLA) to match or surpass stronger baselines (e.g., RoPE, Mamba2), reduce reliance on RoPE to improve length generalization, and pinpoint that linear models’ depth limitations arise from compression/retrieval inefficiencies rather than memory. Like residual connections or LoRA—simple yet powerful—Canon layers may become a minimal yet broadly applicable architectural primitive.

While our academic-scale real-world experiments align with synthetic findings, industrial-scale validation remains crucial; we hope our systematic, economical methodology encourages future investigations at larger scales. We plan to open-source our playground and evaluation suite to support rigorous, reproducible architecture research.

27For instance, in babilong.qa2, “Charlie got a bottle ... Charlie moved to the balcony.” → “Where is the bottle?”—models score < 37% even without junk, i.e., random guessing.

Future Directions. Several interesting directions arise from this work:

- • Alternative Canon Implementations. We focused on simple linear convolutional (kernel size 3) Canon layers for their simplicity and efficient CUDA kernels. Future work should explore dynamic, adaptive convolutions—with weights conditioned on hidden states to enable gating—to assess whether performance gains justify the added computational overhead.
- • Fine-grained Canon Design. We briefly explored selective application (e.g., early layers)

and cross-layer connections—e.g., h′ℓ+1 = hℓ+1 + Canon(hℓ)—which can fuse multiple intralayer Canon operations into a single step, improving efficiency. A systematic evaluation within

our synthetic framework could identify optimal Canon configurations. We are open to exploring this direction further, especially if the community expresses significant interest.

- • Evaluating Emergent Architectures. We selected one representative per architecture family to ensure controlled comparisons and consistent inclusion of Canon layers. Without this rigor, results may misleadingly attribute Canon’s gains to inherent architectural differences (e.g., Mamba2’s built-in conv1d). With controlled comparisons in mind, future work can fairly evaluate emergent architectures, potentially discovering new components with statistically significant improvements.
- • Enriching the Synthetic Playground. Our five synthetic tasks are only a starting point. Designing additional tasks that isolate other architectural capabilities beyond those revealed here—while remaining as atomic as possible—is crucial for finer-grained characterization of model strengths and weaknesses.
- • Interpretability and Probing. We omitted interpretability and probing analyses here for clarity, despite existing frameworks for most tasks (e.g., Lano [6], Capo [5], Mano [7], Brevo [75, 76]). We have conducted preliminary probing for Depo, revealing internal model strategies such as positional parsing (even/odd positioning encoding “→ a” or “a →”) and preprocessing of permutations before the first query (analogous to Brevo [75]). We choose not to include them for clarity, as this paper focuses on architectural comparison.
- • Sparking New Architecture Designs. By pinpointing specific weaknesses (e.g., linear models’ reasoning depth limits and compression inefficiencies), our framework provides targeted signals for improved future designs. We hope synthetic benchmarking informs and inspires the next generation of architecture innovations.

# Appendix

This appendix contains full technical specifications and implementation details for all experiments presented in the main paper. It is intended to support reproduction and in-depth inspection. We provide complete training protocols and evaluation procedures for all five synthetic tasks (Depo, Brevo, Capo, Mano, Lano), real-life experiments (1-hop-L, 2-hop-L, Babilong), and 100B-token SlimPajama/FineWeb-Edu pretraining. We also document the architectural configurations for all models, including Transformers, GLA, Mamba, GDN variants, and MoEs. Additional ablation figures, KL-divergence evaluations, and variant comparisons are included for readers interested in deeper technical insights or replication of results.

#### A Details on Synthetic Pretraining Tasks

We intend to release the code for generating all synthetic pretraining datasets used in this paper, though this may require additional time. To make this paper fully self-contained, we provide detailed specifications below.

- Remark A.1. Throughout this paper, we utilize combinations of A100, H100, and H200 GPUs with bf16 mixed-precision training. While we report the total batch size used in our experiments, we do not specify the exact number of GPUs, as this does not materially affect the results.28

##### A.1 Details on Task Depo: Mental Reasoning Depth

The synthetic pretraining task Depo is designed to evaluate mental reasoning depth by requiring multi-step traversal over directed permutations. The dataset is defined by two parameters: the maximum permutation size N and the reasoning depth K. Each problem instance is generated as follows:

First, a permutation length n is sampled uniformly from {3,4,...,N}. A directed permutation of n nodes is then created, representing a cycle where each node points to its successor: x1 → x2 → ··· → xn → x1. The permutation is presented as edges in the form of ordered pairs (xi,xi+1), but these edges are shuffled randomly into a sequence of 2n tokens. This random ordering ensures that the original cycle structure is not immediately apparent, which would otherwise make the task trivial. The final data format is:

<bos> x1 y1 x2 y2 ... xn yn <query k_1> q_1 <ans> a_1 ... <query k_t> q_t <ans> a_t

Here, xi → yi represents shuffled edges of the permutation. For each query qj, a node is randomly chosen from {x1,...,xn}, and its kj-th successor in the permutation is computed based on the reasoning depth kj ∈ [K], sampled uniformly. The correct answer aj is the kj-th successor of node qj. The number of queries t is set as min(10,n) to balance computational feasibility while ensuring smaller graphs remain interpretable.

Two variants of Depo are used:

• Depo1: Each node name is encoded as 1–2 tokens, with a vocabulary size of 50.

28For instance, training with a single GPU and a batch size of 128 is equivalent to training with 64 GPUs where each GPU processes a batch size of 2. Our codebase supports dynamic GPU allocation, ensuring the total batch size is fixed across training runs while the number and type of GPUs may vary.

• Depo2: Each node name spans 5–7 tokens using a small vocabulary size of 4, introducing

ambiguity that challenges the model’s disambiguation capabilities.29 In addition to node names, special tokens are used: <bos>, <ans>, and <query k> for k ∈ {1,...,K}. The total number of special tokens is K + 2.

Sampling distribution. To ensure controlled task difficulty progression, n is sampled proportionally to √N1+n. This distribution biases training toward simpler cases early on, allowing the model to gradually build foundational reasoning skills before encountering harder examples. Although this distribution is not perfect, it is both simple and effective, enabling clean comparisons between architectural designs without introducing unnecessary hyperparameter complexity. More sophisticated curriculum-based approaches, such as scheduled difficulty [39], may provide an alternative solution but could introduce significant noise, thereby complicating controlled comparisons.

- Remark A.2. This distribution was proposed and tested thoroughly by ZA in 2023 in a number of settings, and subsequently tested (via private communication) by Alfarano in modular arithmetic pretraining [53], where it was benchmarked against other options and shown to also perform well. While synthetic data like this cannot fully replicate the intricacies of real-world distributions, it allows us to simulate an ideal training regime. This forward-looking approach anticipates future improvements in pretraining data—such as higher-quality datasets or RL-based post-training—and evaluates model architectures based on their scalability under such optimal conditions.

Training protocol. To reduce computational cost, we employ label masking: cross-entropy loss is computed only on tokens associated with <ans> and aj. This optimization halves training duration without affecting architectural comparisons. Problem instances are generated online, concatenated, and aligned into 2048-token context windows. Left alignment ensures that the first problem instance in each context is never truncated, as truncation leads to incomplete edges and unusable data.

Evaluation protocol. During evaluation, the permutation size is fixed at n = N, and reasoning depth is tested at both k = K (maximum depth) and k = K/2 (intermediate depth). The protocol mirrors training by generating and concatenating evaluation samples online into 2048-token windows. Accuracy is reported over all answer tokens in the window, ensuring that results are stable regardless of whether answers appear early or late in the sequence.

Data splits and hyperparameters. For Depo1, we use N = 375, 300, 225 and primarily

- K = 8, while testing K = 4 for weaker models. Models are trained from scratch with fresh data while using a fixed random seed to ensure data consistency across architectures. Training uses a batch size of 128, AdamW optimizer (β = 0.9,0.98 and ε = 10−6), weight decay of 0.03, learning rate warmup for the first 1000 steps, and cosine decay to 10%. Training steps are set to 112.5k, 100k, or 87.5k, adjusted for the problem lengths N = 375, 300, 225. The best accuracy is reported across four runs using learning rates {0.0003,0.0005,0.001,0.002}.

Similarly, in Depo2, we use N = 125, 100, 75 and K = 16 (or K = 4 for weaker models). Training steps are set to 150k, 125k, and 100k, respectively.

##### A.2 Details on Task Brevo: Mental Reasoning Breadth

Our pretraining synthetic task Brevo is designed to test mental reasoning breadth by requiring a subgraph topological sort from a given directed acyclic graph (DAG). The dataset is defined by a maximum graph (node) size N. For each problem instance, we first sample a graph of size

29 Multi-token names are generated such that the first ℓ − 1 tokens are chosen from [1, V ], while the final token is selected from [V +1, 2V ]. This creates implicit word boundaries similar to those handled by BPE-based tokenization strategies, such as GPT2Tokenizer.

n ∈ {3,4,...,N} using the same sampling distribution ∝ √N1+n as employed in Depo, and generate data in the following format:

<bos> x1 y1 x2 y2 ... xm ym <query> q <ans> a1 a2 ... ap <eos>

Here, the 2m tokens define m directed edges xi → yi spanning n nodes, meaning that yi depends on xi. Given a query vertex q, the model must return all vertices it recursively depends on, in topological order starting from the leaves. Specifically, if u → v → q, the model must output u before v.

DAG generation protocol. After sampling n, we generate the random DAG as follows. First, we randomly shuffle all the vertices and begin inserting edges. We select a random number L ∈ {1,...,⌈n−41⌉ + 1}, designating the first L vertices as leaves (no incoming edges). Starting from vertex L + 1, we iteratively process each vertex by selecting all preceding vertices that have an out-degree of at most 3. From this set, we randomly pick a subset of between 1 and up to 4 vertices and connect them to the current vertex. This process continues until all vertices are traversed, yielding a DAG with a maximum in-degree and out-degree of 4.30

At this point, the vertices naturally form a topological order from left to right. We then select a random query vertex from the last quarter of the vertices. Choosing vertices closer to the right increases the depth of the dependency graph while avoiding degenerate cases where all nodes are reachable (such as if the query were the last vertex). Finally, we reshuffle all the vertices and assign random names to them. Vertex names are uniquely selected, as described below.

Vertex names. In Brevo1, each vertex name consists of a single unique token, randomly selected from {1,...,N}. In Brevo2, each vertex name spans 2–4 tokens using a vocabulary of size 4, which introduces ambiguity (e.g., multiple token combinations can encode unique vertex names). See Footnote 29 for the method used to generate multi-token words. Aside from vertex names, we use 4 distinct special tokens: <bos>, <query>, <ans>, and <eos>.

Training protocol. To reduce computational costs, we enable label masking, where the crossentropy loss is computed only on <ans>, <eos>, and aj tokens. Selective testing showed that this technique saves training time without affecting architectural comparisons. Instances are generated online, concatenated, and left-aligned into context windows. By left-aligned, we mean that the first instance in each context window is never truncated. Without left alignment, truncation of the first instance would render it incomplete (e.g., missing edges in the graph), making the instance a useless training example.

Evaluation protocol. During evaluation, we fix n = N and test only the largest graph. The model is prompted with a random DAG of size n and query vertex q, and tasked to generate the answer sequence a1,...,ap. The generated sequence is then parsed and validated against the following criteria:

- • The answer sequence must contain all reachable vertices from q and no non-reachable vertices.
- • The vertices in the answer sequence must appear in a valid topological order. Since topological orderings are not unique, any valid ordering is accepted.

Invalid tokens, duplicate outputs, or missing vertices are not accepted, and no partial credit is given.

Training details. In Brevo1, we use N = 110,90,70 with vertex names consisting of one token, and each problem fits within 1024 tokens. Models are trained from scratch with fresh data but

30Constraining the maximum in-degree and out-degree to 4 prevents the dependency graph from becoming too shallow, which would make the task trivial.

a fixed seed (ensuring pretraining data consistency across model architectures). Training uses a context length of 1024, a total batch size of 256, AdamW optimizer (β = 0.9,0.98 and ε = 10−6), weight decay of 0.03, learning rate warmup over the first 1000 steps, and cosine decay to 10%. Pretraining lasts 150k, 125k, or 100k steps respectively for N = 110,90,70, accounting for the varying problem lengths. We report the best performance out of four runs using learning rates {0.0003,0.0005,0.001,0.002}.

In Brevo2, we use N = 50,40,30, with vertex names spanning 2–4 tokens, and each problem fits within 1536 tokens. Models are trained in the same manner as Brevo1, except that we use a context length of 1536, a total batch size of 192, and pretraining lasts 250k, 225k, or 200k steps respectively for N = 50,40,30.

The comparison between Brevo1 and Brevo2 demonstrates that the ambiguity introduced by multi-token vertex names does not noticeably impact architectural comparisons, which is the focus of this paper.

##### A.3 Details on Task Capo: Knowledge Capacity

The synthetic pretraining task Capo borrows directly from Allen-Zhu and Li [8], where the authors introduced the bioS(N) dataset. This dataset contains N biographies of randomly generated individuals, each described by six attributes: birth date, birth city, university, major, employer, and working city.31

To represent these biographies in natural language, each individual is described via randomly selected English sentences for every exposure to the pretraining data. Sentence templates correspond to the individual’s attributes, ensuring diverse paraphrasing across exposures. For example:

Anya Briar Forger was born on October 2, 1996. She spent her early years in Princeton, NJ. She received mentorship and guidance from faculty members at Massachusetts Institute of Technology. She completed her education with a focus on Communications. She had a professional role at Meta Platforms. She was employed in Menlo Park, CA.

The diversity in writing ensures that models learn to store explicit knowledge about an individual’s attributes, rather than merely memorizing surface-level patterns in specific templates [5, 7]. Following the recommendations of [8], we pretrain models over 100 exposures per individual, which provides a controlled environment for comparing architectural differences. Training beyond 100 exposures diminishes architectural differences, as longer training typically allows all models to converge toward similar levels of performance [8].

Knowledge format independence. Previous experimental evidence suggests that a model’s knowledge capacity does not heavily depend on the specific format in which the knowledge is stored. For example, one could consider synthetic alternatives such as longer word lengths, different vocabulary sizes, or even abstract encoding formats. Importantly, any such synthetic configuration remains a reliable discriminator for comparing model architectures. For simplicity and interpretability, however, we adhere to the more English-like biography format in bioS(N), aligned with [8].

Clean experimental comparisons. Models could alternatively be pretrained on exposures distributed according to power-law dynamics or incorporating infrequent “junk data.” While such approaches might better mimic real-life datasets, they introduce subtle stochastic effects that can depend heavily on the formatting of rare samples. To avoid confounding factors, we adopt the cleaner 100-exposure baseline for pretraining individual biographies, as it allows for clearer isolation of architectural capabilities.

31The working city is derived from the employer’s headquarters, while all other attributes are sampled uniformly and independently. Possible attribute domains include N0 = 400 × 400 × 1000 person names, 12 × 28 × 200 birth dates, 200 birth cities, 300 universities, 100 majors, 263 employers, and two pronouns.

Evaluation protocol. After pretraining on bioS(N) data, knowledge capacity is measured based on the number of bits a model reliably stores. This quantity is further normalized to bits per parameter to account for model scale. Partial correctness (e.g., recalling the year but not the full date of birth) is accounted for in the bit computation to ensure fine-grained evaluation of knowledge storage. For detailed computation, we direct readers to [8]. Unlike other tasks presented in this paper, measurement of bits per parameter requires varying both data sizes N and model sizes to compute the Pareto frontier of knowledge capacity versus parameter count. For this reason, we vary N between 50K and 2M while testing models ranging from 1M to 500M parameters.

Pretraining setup. To ensure consistency across all architectures, pretraining uses the GPT2 tokenizer and ties weights for embedding and output layers. Tying weights ensures consistent learning dynamics across model families (e.g., GPT, Llama, Mamba, GLA), while limiting the vocabulary size to 3275 tokens (from GPT-2’s original 50257 tokens), as the bioS(N) dataset does not utilize the entire vocabulary.

Batch size, learning rate decay, and other hyperparameters strictly follow the 100-exposure baseline outlined in [8], with only minor modifications. Specifically, we test two learning rates per configuration (selected from their three choices) and report the best results. As a result, our reported knowledge capacity in Figure 4 may slightly deviate from the original results, though introducing Canon layers restores capacity without adding hyperparameter choices.

Hyperparameters for dense models. The following hyperparameters were used for dense models in the 100-exposure setup:

- • For N = 50K: weight decay wd = 0.01, learning rates lr = 0.001/0.0005, batch size 12.
- • For N = 100K: wd = 0.01, lr = 0.001/0.0005, batch size 24.
- • For N = 200K: wd = 0.01, lr = 0.001/0.0005, batch size 48.
- • For N = 500K: wd = 0.01, lr = 0.001/0.0005, batch size 96.
- • For N = 1M: wd = 0.01, lr = 0.001/0.0005, batch size 192.
- • For N = 2M: wd = 0.01, lr = 0.0005/0.0003, batch size 384.

Hyperparameters for MoE models. Mixture-of-Experts (MoE) training was conducted using the tutel moe package [30], consistent with [8]. MoE training uses 32 experts with topk = 1 and cap factor = 2. Due to the larger learning rates required for MoE-based pretraining, we use the following hyperparameters:

- • For N = 50K: wd = 0.01, lr = 0.005/0.002/0.001, batch size 12.
- • For N = 100K: wd = 0.01, lr = 0.005/0.002/0.001, batch size 24.
- • For N = 200K: wd = 0.01, lr = 0.005/0.002/0.001, batch size 48.
- • For N = 500K: wd = 0.01, lr = 0.002/0.001, batch size 96.
- • For N = 1M: wd = 0.01, lr = 0.002/0.001, batch size 192.
- • For N = 2M: wd = 0.01, lr = 0.001/0.0005, batch size 384.

##### A.4 Details on Task Mano: Knowledge Manipulation

The synthetic pretraining task Mano evaluates a model’s ability to manipulate stored knowledge mentally without relying on explicit intermediate cues (e.g., Chain-of-Thought reasoning). Unlike memorization tasks, Mano requires multi-step internal computation, testing the model’s capacity for hierarchical manipulation.

Task format and setup. The dataset is defined by a maximum length L, with each instance consisting of arithmetic expressions of ℓ operations, where ℓ is sampled uniformly from [1,L]. Expressions are presented in prefix (pre-order) notation to eliminate ambiguities in parentheses and operator precedence. For example, a length-ℓ = 3 instance is:

|25|->22 23 25|->24 24 25|->24 23<br><br>22|->21 19<br>22|->21 20 20<br>22|->21 21<br><br><br>22|->19 21<br>23|->20 21 20<br><br><br>23|->20 21 21<br><br>23|->20 20<br><br>23|->19 20<br>24|->21 20<br><br><br>24|->20 21<br><br><br>24|->19 19 21<br><br><br>19|->17 18 16<br>19|->18 16<br><br><br>19|->18 16 18<br><br>19|->18 18 17<br>20|->17 17 17<br><br>20|->16 18<br><br>20|->17 17 16<br>20|->18 17 16<br><br><br>21|->18 17<br><br><br>21|->16 18 16 21|->17 16 16 21|->17 18 18<br><br><br><br><br>16|->15 14<br><br>16|->14 15<br>16|->15 14 15<br><br><br>16|->14 13 13<br>17|->14 15 14<br><br>17|->15 13 13<br><br>17|->15 15<br>18|->14 14<br><br><br>18|->15 13<br><br><br>18|->13 14 14<br><br><br><br><br>cfg3k<br><br>13|->10 12 12<br>13|->11 11 12<br>13|->12 12<br><br><br>14|->10 11<br><br>14|->10 11 11<br>14|->11 10 10<br><br><br>14|->11 11 15|->12 11<br><br><br>15|->10 10<br>15|->11 12<br><br><br>10|->7 7<br>10|->8 8<br>10|->9 9 7<br><br><br>10|->7 9 7<br>11|->8 7 9<br><br>11|->7 7 7<br>11|->7 8 7<br><br><br>12|->7 8<br><br><br>12|->8 8 7<br>12|->9 7 9 12|->8 9<br><br><br>7|->3 3<br><br>7|->1 3 1<br>7|->2 1<br><br><br>8|->2 3<br><br>8|->1 1 2<br><br>8|->3 1<br><br>8|->1 2 2<br><br>9|->1 3 3<br><br><br><br><br>9|->2 2 9|->1 1 1 9|->3 2 3<br>|
|---|

|25|->22 22<br>25|->23 22<br>25|->24 23 24<br><br><br>22|->19 20<br><br>22|->19 19 21<br>22|->20 19 19<br><br><br>23|->21 19 20<br><br>23|->21 20<br><br>23|->19 20 20<br>24|->19 19<br><br><br>24|->20 21 19<br><br><br>24|->19 21 21<br><br><br>19|->17 17 16<br><br>19|->16 18 18<br><br>19|->16 17<br>20|->17 17 17<br><br>20|->18 18 20|->18 17 18<br>21|->18 16<br><br><br>21|->16 17 17<br><br>21|->16 18 16<br><br><br><br><br><br><br>16|->14 15<br><br>16|->13 15<br><br>16|->15 15<br><br>17|->14 14<br><br>17|->15 15 15<br><br>17|->15 14<br>18|->14 13 15<br><br><br>18|->13 13 14<br><br><br>18|->13 14<br><br><br><br><br>cfg3j<br><br>13|->10 12<br><br>13|->12 10<br><br>13|->11 11<br><br>14|->10 11<br><br>14|->12 11 10<br><br>14|->10 11 11<br>15|->11 12 12<br><br><br>15|->10 10<br><br><br>15|->11 12<br><br><br><br><br>10|->7 7 9<br>10|->8 7<br>10|->9 8<br><br><br>11|->7 8 8<br><br>11|->9 9<br><br>11|->8 8<br><br>12|->7 7 8<br><br><br><br><br>12|->8 7 7<br>12|->9 8 8<br><br><br>7|->1 2<br>7|->2 2<br>7|->3 3 2<br><br><br>8|->3 1<br><br>8|->2 3<br><br>8|->1 1 2<br>9|->3 2 1<br><br><br>9|->1 3<br><br><br>9|->3 2<br>|
|---|

|22|->20 21<br><br>22|->20 19 21<br>22|->21 19 19 22|->20 20<br><br><br>19|->18 16 18<br><br>19|->17 18<br>19|->18 18<br><br><br>20|->16 16<br><br>20|->16 17<br>20|->17 16 18<br><br><br>21|->18 17 21|->17 16<br><br><br>21|->16 17 18<br>21|->16 18<br><br><br>16|->15 15<br><br>16|->13 15 13<br>16|->14 13<br><br><br>16|->14 14<br>17|->15 14 13<br><br>17|->14 15<br><br>17|->15 14<br>18|->14 15 13<br><br><br>18|->15 13 13<br><br><br>18|->13 15<br><br><br><br><br>cfg3f<br><br>13|->11 12<br>13|->12 11 12<br><br><br>13|->10 12 11<br>14|->10 12<br><br>14|->12 10 12<br>14|->12 11<br><br><br>14|->10 12 12<br>15|->10 11 11<br><br><br>15|->11 11 10 15|->10 10 15|->12 12 11<br><br><br>10|->8 9 9<br>10|->9 7 9<br><br><br>10|->7 9 9<br>11|->8 8<br><br>11|->9 7<br><br>11|->9 7 7<br>12|->7 9 7<br><br><br>12|->9 8<br><br><br>12|->8 8 9<br><br><br>7|->2 2 1<br>7|->3 2 2<br><br><br>7|->3 1 2<br>7|->3 2<br><br><br>8|->3 1 1<br><br>8|->1 2<br><br>8|->3 3 1<br>9|->1 2 1<br><br><br>9|->3 3<br><br><br>9|->1 1<br>|
|---|

- a sample from cfg3j:
- a sample from cfg3k:

2312213113121122122212221131312113332333333333231312123212121213321223313 3131313321112212131331331122113323211331221333322213323211221123332332311 1331222333131233322332333122232331212131131131211313321233321333121133112 2113321231311313121112111213121313111233323333333131212311222211112133211 3113133212121333232111133231311123311113131211222231211111222122131332111 2112213312213111213132313313122122333232313122233233313133323131131212311

2211222313123323133131323131332131122131131232113113133323131332233221113

3133111331111113313113121112331221311113321213232131331113321131312111111 2212213332321131131332321133221313231121221312133323331331222313133122131 1221221312233211122123133222233332113113322332313132311221223311213112233 3231222111131133112333231121311112112221133221313321

2223112122231112222233213223323213112232222123131321233231312223112231232 3213213232321322311233212313322232132133232132112112332221332321312333223 2311222123232233322233222233323131131121233213132332111223332121323112332 1123132112133212112313321122312332311222112321323122122231321232323223322 3323123232332323232132123112332121333211223312231112312322121123111212121

3321123111233231233322323321123131332332231121123323322222321233323232221

2231233211223332132332132221231332332232223112321313213111231313211122322 3323213211331133131233213222332321332122332123311232233232132231123112323 2223322332321312313321332222322223212322123321232321311121121222112313321 2323323321332132231122212311222232333232132122213131332113321122312321133 211121123321223121232111233213112112321122222323112

a sample from cfg3f:

3322131233121131232113223123121112132113223113113223331231211121311331121 3212133333123221213123222111121332213113113113111111323123313313331133133 3332231211311121221111211233312331121113313333331123333131111333312113211 3121211333332121111212132232233221332211132211323233131112132232232212111 33331121322221332211212133121331332212213221211213331232233312

Figure 17: Task Lano: our constructed dataset cfg3k,cfg3j against the cfg3f dataset from [6].

<bos> <len_3> + * a b - c d <ans> ans

This corresponds to the expression ((a × b) + (c − d)) mod 23, where operands a, b, c, and d are integers sampled uniformly from [0,22]. The task involves three operations (+, -, *), each represented as distinct tokens, with all computations performed modulo 23.

The factual base consists of three 23×23 arithmetic tables (addition, subtraction, and multiplication), which models learn implicitly during pretraining. Operands are encoded as single tokens from [0,22], while special tokens (<bos>, <ans>, and <query_ℓ> for ℓ ∈ [L]) structure the sequence.

Expressions are generated recursively: the first operator is sampled uniformly from the three available options, and its operands are split into sub-lengths ℓ′ and ℓ − 1 − ℓ′ (with ℓ′ chosen uniformly), recursively generating sub-expressions.

Why modular arithmetic? Modular arithmetic (mod 23) ensures manageable knowledge size while introducing sufficient diversity in intermediate and final results. Similarly, limiting operations to addition, subtraction, and multiplication simplifies task design while retaining depth, enabling models to focus on hierarchical manipulation instead of memorizing surface-level patterns.

Training protocol. Models are pretrained on three datasets corresponding to difficulty levels

- L = 16, L = 13, and L = 10. The cross-entropy loss is applied over all tokens (problem description and answer), without label masking, since hierarchical manipulation requires attention across the full sequence. Instances are generated online, concatenated, and left-aligned into context windows of length 1024.

Models are trained from scratch using fixed random seeds for consistency across architectures. Training lasts 110k, 95k, and 80k steps for L = 16, L = 13, and L = 10, respectively. Hyperparameters include a batch size of 64, AdamW optimizer (β = 0.9,0.98 and ε = 10−6), weight decay of 0.1, learning rate warmup for 1000 steps, and cosine decay to 10% of the initial learning rate. Results are reported based on eight training runs with learning rates {0.0001,0.0002,0.0003,0.0005} and two seeds.

Evaluation protocol. During evaluation, expressions are sampled from the same distribution used for training, with ℓ fixed at L (maximum difficulty). Accuracy is computed over all problem instances within 1024-token context windows, including non-first instances. Since outputs are single tokens representing exact modular arithmetic results, partial correctness is not applied.

##### A.5 Details on Task Lano: Hierarchical Language Structure

The synthetic pretraining task Lano evaluates a language model’s ability to perform structural reasoning, specifically long-range structural planning that requires dynamic programming to resolve ambiguity. Unlike in-context reasoning tasks (e.g., Depo, Brevo) or knowledge reasoning tasks (e.g., Mano), Lano challenges models to learn hierarchical structures governed by probabilistic context-free rules and process sequences that cannot be resolved locally.

Task format and setup. Sentences are generated probabilistically using context-free rules. The cfg3f dataset [8] starts with the root non-terminal (NT) symbol 22, which uniformly expands into one of four rules:

22  → 20 21, 22  → 20 19 21, 22  → 21 19 19, 22  → 20 20.

Each rule is chosen with probability 1/4, ensuring uniform randomness. Rules are applied recursively and probabilistically to NT symbols (e.g., 19, 20, 21), replacing all NT symbols with terminal (T) symbols 1, 2, or 3. The process generates sentences composed entirely of terminal symbols based on probabilistic expansions.

Pretraining involves predicting next tokens in CFG-generated sequences without access to the underlying rules, requiring models to learn structural reasoning implicitly. During evaluation, models are prompted with a single <bos> token and tasked to generate CFG-compliant sentences using temperature 1. Accuracy is assigned only for fully valid sentences, with no partial credit applied.

Parsing difficulty and ambiguity. Parsing CFG-generated sequences is uniquely challenging because resolving derivation chains requires global reasoning. For example, parsing ”221213133” requires resolving structural ambiguity between terminal symbols that cannot be inferred from local patterns alone. Instead, parsing requires an O(n3) dynamic programming algorithm to globally reconstruct relationships across the sequence, even when CFG rules (from Figure 17) are explicitly available. During pretraining, models face additional difficulty as they must learn these relationships without direct access to the probabilistic rules.

Building upon cfg3f as a baseline, we introduce two extended datasets in this paper:

- • cfg3k: Retains the probabilistic framework of cfg3f but increases depth by one level, doubling sequence length and increasing parsing complexity by eight times due to the cubic nature of dynamic programming (O(n3)).
- • cfg3j: Extends cfg3f by one level but reduces the number of rules, creating shorter sequences with intermediate difficulty between cfg3f and cfg3k.

Both datasets use the same probabilistic generation process and are detailed in Figure 17.

Training details. Pretraining uses cross-entropy loss computed over all tokens without label masking. Sentences are generated online, concatenated, and aligned into context windows. For cfg3f, we use a context length of 512 as in [8], while longer datasets cfg3j and cfg3k require extended context lengths of 1536.

Models are trained from scratch using fixed seeds for consistency across architectures. Training uses a batch size of 96, AdamW optimizer (β = 0.9,0.98 and ε = 10−6), weight decay of 0.1, no learning rate warmup, and linear decay to 0. Pretraining lasts 100k steps, and results are reported from four training runs using learning rates {0.0002,0.0003,0.0005,0.001}.

Evaluation details. During evaluation, models generate sentences from a <bos> prompt using temperature 1 and beam width 1.32 Generated sentences are validated using an O(n3m) dynamic

32This is crucial to ensure that the model is generating the genuine probabilistic distribution of sentences; if using temperature 0 for instance, the generation is always a fixed string, and accuracy would be either 0 or 100% forever.

programming parser (n: sequence length, m: CFG rules) to confirm compliance. An alternative evaluation computes KL divergence between the model’s next-token prediction distribution and the ground-truth CFG predictions. Both methods yield consistent architecture comparisons.

#### B Details on Other + Real-Life Experiments

This section provides a brief description of additional tasks used in the paper.

Full Copy. In Figure 5, we evaluated the performance of models with 1 or 2 layers on a trivial pretraining task. This task involves choosing N = 500 and generating a sequence starting with <bos>, followed by a random permutation of N tokens between 1 and N, then appending <query> and an identical copy of the sequence. The task uses label masking, where the loss is computed only on the N answer tokens. Models are pretrained with a context length of 1024, a total batch size of 32, AdamW optimizer (β = 0.9,0.98 and ε = 10−6), weight decay of 0.03, learning rate warmup for the first 1000 steps, and cosine decay to 10%. Training duration is set to 50k steps, and the best results are reported across learning rates {0.0005,0.001,0.002,0.005}.

For this task, we also assessed the models’ ability to correctly copy the first t = 1,2,4 tokens within the sequence. As shown in Figure 18, these results are nearly identical to those in Figure 5.

68% 100%100%100%100%100%100%100%100%100%100%100%100%100%

67% 100%100%100%100%100%100%100%100%100%100%100%100%100%

61% 100%100%100%100%100%100%100%100%100%100%100%100%100%

[Figure 209]

[Figure 210]

[Figure 211]

- RoPE(1L-2H-16D)

- RoPE(1L-4H-32D)

- RoPE(1L-8H-64D)

- RoPE(1L-16H-128D)

- RoPE(2L-2H-16D)

- RoPE(2L-4H-32D)

- RoPE(2L-8H-64D)

- RoPE(2L-16H-128D)

- RoPE(1L-2H-16D)

- RoPE(1L-4H-32D)

- RoPE(1L-8H-64D)

- RoPE(1L-16H-128D)

- RoPE(2L-2H-16D)

- RoPE(2L-4H-32D)

- RoPE(2L-8H-64D)

- RoPE(2L-16H-128D)

- RoPE(1L-2H-16D)

- RoPE(1L-4H-32D)

- RoPE(1L-8H-64D)

- RoPE(1L-16H-128D)

- RoPE(2L-2H-16D)

- RoPE(2L-4H-32D)

- RoPE(2L-8H-64D)

- RoPE(2L-16H-128D)

- 0% 1% 11% 57% 100%100%100%100%100%100%100%100%100%100%

- 0% 63% 100%100%100%100%100%100%100%100%100%100%100%100%
- 1% 100%100%100%100%100%100%100%100%100%100%100%100%100%

- 1% 97% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 95% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 95% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 99% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 99% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 52% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 42% 100%100%100%100%100%100%100%100%100%100%100%100%

- 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0%

- 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0%
- 0% 0% 0% 0% 1% 1% 1% 1% 1% 1% 1% 1% 1% 1%

- 0% 0% 1% 43% 100%100%100%100%100%100%100%100%100%100%

0% 0% 0% 35% 100%100%100%100%100%100%100%100%100%100%

- 0% 0% 1% 1% 1% 3% 4% 5% 6% 6% 6% 6% 6% 7%
- 0% 1% 2% 6% 9% 11% 12% 12% 12% 13% 14% 15% 15% 15%

0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0%

0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0%

- 0% 73% 97% 99% 100%100%100%100%100%100%100%100%100%100%
- 1% 1% 1% 1% 1% 1% 1% 1% 1% 1% 2% 2% 2% 2%

0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0% 0%

0% 60% 97% 99% 100%100%100%100%100%100%100%100%100%100%

0% 54% 97% 99% 100%100%100%100%100%100%100%100%100%100%

Canon(cst) - RoPE(1L-2H-16D)

Canon(cst) - RoPE(1L-2H-16D)

Canon(cst) - RoPE(1L-2H-16D)

0% 68% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 69% 100%100%100%100%100%100%100%100%100%100%100%100%

0% 67% 100%100%100%100%100%100%100%100%100%100%100%100%

Canon - RoPE(1L-2H-16D)

Canon - RoPE(1L-2H-16D)

Canon - RoPE(1L-2H-16D)

5001000150025005000100001500020000250003000035000400004500050000

5001000150025005000100001500020000250003000035000400004500050000

5001000150025005000100001500020000250003000035000400004500050000

training steps

training steps

training steps

(c) Evaluated with t = 4 Figure 18: A trivial experiment for copying 500 tokens, evaluated only on correctly copying the first t tokens.

(a) Evaluated with t = 1

(b) Evaluated with t = 2

Task 1-hop-L and 2-hop-L. In the real-life experiment (Section 8), we evaluated models’ performance on extremely simple 1-hop and 2-hop information retrieval tasks.

- For the 1-hop-L task, we prepared five random birth year statements of the form “[name] was

born in the year of [year],” where names are generated as random combinations of first, middle, and last names, and years are sampled uniformly from 1950 to 2009. The five sentences were embedded into random Wikipedia documents of length L tokens, with each statement inserted between sentences at up to five randomly chosen positions. Finally, the model was prompted with “\n\nAnswer me: name was born in the year of” to test its ability to retrieve the birth year. This setup closely replicates the needle-in-a-haystack task [29], but we intentionally made the task more “natural English” by using birth years (commonly found in pretraining datasets like Wikipedia) instead of abstract multi-digit numbers.

- For the 2-hop-L task, three random birth year statements were prepared in the same format

as above. This was followed by three equivalence statements of the form “[name1] was born in the same year as [name2],” where random names were generated such that the equivalences formed a bijection between the two sets of three random names. To simplify the task, we did not shuffle the ordering of the statements; the three equivalence statements always followed the three original ones. These six sentences were then embedded into random Wikipedia documents of length L tokens, inserted at up to six different positions between sentences, respecting their original order. At the end, the model was prompted with “\n\nAnswer me: name was born in the year of” to test its ability to infer and retrieve the correct birth year. To further assist the model, an instructional

statement was added at the beginning of the context. 33 This design represents arguably the simplest possible and most natural 2-hop in-context reasoning task, yet even with L = 0, models largely failed to perform, as demonstrated in Figure 16.

Babilong. For the Babilong experiments, we found the default few-shot prompts (qa1–qa5) slightly suboptimal and replaced them with improved ones, which are released in our GitHub repo [2].

SlimPajama and FineWeb-edu 100B. The SlimPajama dataset is taken from HuggingFace (cerebras/SlimPajama-627B), using the first 100M samples (more than 100B tokens). FineWebEdu [42] is obtained from HuggingFaceFW/fineweb-edu, using its predefined 100B split. Both datasets provide sufficient scale for our 1.3B-model pretraining experiments.

Following standard practice, all data are tokenized in order, concatenated into a continuous text stream, and sampled into random 4096-token windows for pretraining across architectures. We train with total batch size 48 using AdamW (β1=0.9, β2=0.98, ϵ=10−6, weight decay 0.03). Llama and GPT models use learning rates {0.001,0.002}, while linear models (Mamba, GLA, GDN) use {0.0005,0.001,0.002} for stronger baselines. Each model is trained for 510,000 steps, processing 4096 × 48 × 510,000≈100.2B tokens per run. For each evaluation task, we report the best accuracy across the tested learning rates.

To ensure fairness, all architectures share the same random seed, guaranteeing identical data order and content—even if runs are interrupted and resumed. This setup minimizes variability from data differences and isolates architectural effects. For Llama(RoPE), we additionally test eight random seeds to measure variance, shown in Figure 16 and detailed in Appendix E.1, including both joint (data + model init) and model-init-only random seed variations. Architecture specifications appear in Appendix C.1.

Beyond 100B-1.3B. We find that academic-scale pretraining (100B tokens, 1.3B models) is too noisy to reveal subtle architectural gaps (e.g., Llama vs. Llama+Canon). Larger-scale experiments (1–8B models pretrained on 1–2T tokens) are therefore reported in our follow-up work [2].

#### C Details on Architectures Used

Transformer Models (Llama/GPT). In this paper, “Llama(RoPE)” refers to the Huggingface (HF) implementation LlamaForCausalLM, which employs rotary embeddings across all hidden dimensions and utilizes gated MLP layers. We did not enable group-query attention, as this study focuses on smaller-scale models. The intermediate size is set to 83d, ensuring that each MLP layer contains 8d2 trainable parameters, consistent with standard MLP layers. “Llama(NoPE)” refers to the same architecture with rotary embedding completely disabled. “Llama(RoPE)ˇ “” refers to the version where rotary embeddings are applied to only a quarter of the dimensions. The variants ˇ “, ˇ “ˇ “, and ˇ “ˇ “ˇ “represent differing fractions of dimensionality on which RoPE is enabled, as described in the main paper.

For direct comparisons, “GPT2(RoPE)” refers to the Llama architecture with gated MLP layers replaced by standard MLP layers. The intermediate size in these models is set to 4d, ensuring that each MLP layer contains 8d2 trainable parameters.34

We denote “GPT2(RoPE,R2)” as the GPT2(RoPE) model with its silu activation replaced by ReLU2, following the design proposed in Primer [59]. Similarly, “Llama(RoPE,R2)” refers to

- 33“You will be asked questions about people’s birth years, and the birth year descriptions are hidden in some random text. Some people’s birth years are directly given, while others are given in the form that ‘name1’ was born in the same year as ‘name2’. ”
- 34The original GPT2 architecture differs from Llama in other minor ways, such as using GeLU activation and slightly different initialization. We do not investigate these small architectural differences in this paper.

Llama(RoPE) with ReLU2 in place of silu.

Alibi and H-Alibi. For ALiBi [45], we follow the original recommendation of using a geometric sequence 2−8/n for an n-head Transformer, which determines how each head is biased toward local context. For H-Alibi [31], we use their proposed strategy of restricting the h-th head to attend only to the nearest h tokens, and applied to half of the heads. (We briefly tested applying this to one-third of the heads instead, but observed slightly worse performance.)

Mamba Models. For “Mamba2,” we use the HF implementation Mamba2ForCausalLM, with recommended configuration parameters (2 means expansion factor):

ssm state size=64, num heads=16, and head dim=hidden size * 2 / num heads.

This setup ensures each Mamba layer has 6d2+o(d2) trainable parameters. The recurrent state size (per layer) is therefore 2d × ssm state size = 128d plus conv1d. We briefly tested num heads=8 but observed worse results, so did not include it. The model initialization follows the HF default (which uses PyTorch default init as opposed to a fixed 0.02 std init).35

For “Mamba2(mlp),” we use the same HF implementation but alternate between Mamba SSM layers and gated MLP layers. The intermediate size for gated MLP is 2d, ensuring each MLP layer contains 6d2 trainable parameters. This ensures that ℓ-layer d-dimensional Llama(RoPE) and Mamba2(mlp), as well as 2ℓ-layer d-dimensional Mamba2, have comparable parameter counts.

Mamba1. We briefly tested Mamba1 and found it consistently outperformed by Mamba2 in our pretraining playground, so we excluded it from main results. Notably, removing its conv1d layer also degrades Mamba1 to GLA-level performance.

Mimetic initialization. Following [66], we enabled A ≈ 1 (via c=8), ∆ ≈ 1 (via b∆=0.54), WC⊤WB ≈ I, and conv1d ≈ I. We also tested c=4 and c=2 but observed no improvement.

GLA Models. For Gated Linear Attention (GLA) [72], we use the official fla-org implementation [71].36 We use 4 linear attention heads (their default configuration; also suggested by their first author). With d = 512 or 768, this corresponds to headdim = 128 or 192, thus the recurrent state size (per layer) is d2 ×headdim = 64d or 96d — both smaller than the Mamba2 models we tested. We briefly tested 8 attention heads but found that these consistently degraded performance. Each linear attention layer contains about 4d2 trainable parameters; the (gated) MLP has an intermediate size of 83d, contributing roughly 8d2 parameters, matching Llama.

The default GLA implementation has disabled conv1d (the functionality was not part of the original publication [72]), although their codebase supports conv1d, which we explicitly tested in this paper. They used 0.02 as initializer std for such conv1d layers with SiLU activation.

For GLA(elu) experiments in the ablation studies, we replaced the default feature map with elu(x) + 1, and conducted evaluations with and without conv1d and Canon layers.

GDN Models. For Gated DeltaNet (GDN) [73], we use the official fla-org implementation [71].37 We use 4 or 6 heads for d = 512 or d = 768, respectively (as suggested by their first author). This

- 35We briefly tested the 0.02 init and did not observe significant difference.
- 36We use the default expand k = 0.5 and expand v = 1. From March to May 2025, the repo authors up-

dated initializer range to 0.006 (from the previously popular 0.02), which we found to negatively affect performance. We reverted it to 0.02; the authors also restored this value on May 3, 2025. We further disabled rescale prenorm residual for fair comparison. This option, inherited from GPT-2 [48], scales down the output projection (e.g., o proj) initialization by 1/

√

N, where N is the number of residual layers. The default HF implementations of Llama and Mamba2 both have this disabled, whereas the fla-org implementations enable it by default. We find that disabling this slightly improves model performance, and after communicating with Yang and Zhang [71], they also disabled it on June 24, 2025. Some of these were introduced after V2.0 of this paper, leading to small diffs in experimental results compared to V1.1.

37Similar to GLA (see Footnote 36), we adopt initializer range=0.02 and disable rescale prenorm residual. Note their default expand k = 0.75 and expand v = 1.5.

corresponds to key/value headdim of (96,192), giving a recurrent state size (per layer) of 144d, comparable to Mamba2. Each GDN layer contains about 6d2 trainable parameters, so we set the (gated) MLP intermediate size to yield another 6d2 parameters, matching Llama per layer block.

Weight tying, tokenizer. Unless otherwise stated (i.e., in Task Capo), we do not tie weights between the embedding and output layers in any of the architectures (e.g., Llama, Mamba, GLA, GDN). Additionally, no tokenizers are used during pretraining except for Task Capo.

Task Capo. The knowledge-capacity task pretrains on synthetic biographies following [8]. For consistency, we use GPT2Tokenizer and tie embedding/output weights to minimize capacity loss in small models (though the effect is minor).

Since Capo measures bit-per-parameter knowledge capacity, both model and data scales are increased to assess scaling behavior. Following [8], we adopt the ℓ-h notation for model size, where Llama(ℓ-h) has ℓ layers, hidden size 64h, and h heads, and extend this convention to GLA, Mamba2, and GDN for comparability.38

GPT2 experiments in Figure 11 use the original GPT2 architecture augmented with RoPE, as in [8]. Mixture-of-Experts (MoE) experiments employ tutel [30] with 32 standard MLP experts (topk=1, cap factor=2).

##### C.1 Real-Life Experiments

For pretraining experiments on SlimPajama and FineWeb-Edu, we use all the architectures listed above alongside the Llama2 tokenizer (with vocab size 32,000) [65]. Weight tying is disabled to maintain consistency with prior works (e.g., [10, 73] and references therein).

The architectural configurations used in the real-life experiments are summarized below. They follow the setups described in Section C, except that we increase both width and depth to yield approximately 1.35B parameters per model:

- • Llama (RoPE/NoPE): 24 layers, 32 heads, hidden size d = 2048.
- • GLA: 24 layers, 4 heads, hidden size d = 2048.
- • Mamba2: 48 layers, 16 heads, hidden size d = 2048.
- • Mamba2(mlp): 24 layers, 16 heads, hidden size d = 2048.
- • GDN: 24 layers, 12 heads, hidden size d = 2048.

For linear models (excluding conv1d), the per-layer recurrent state sizes are 256d for GLA, 128d for Mamba2 (with twice the layers), and 192d for GDN. These are of the same order of magnitude, while remaining close to the original authors’ recommended settings. Each model contains roughly 12d2 trainable parameters per layer (except Mamba2, which has 6d2 per layer but twice as many layers), ensuring a fair comparison across architectures.

##### C.2 Canon Implementations

Canon layers (i.e., type A,B,C,D) in this paper are implemented using PyTorch’s nn.Conv1D with kernel size 4, zero padding, and default initialization (i.e., kaiming uniform with a = √5). Unlike in GLA/GDN and in most linear models, this choice of the “default initialization” makes their weights initialized at O(1) instead of 0.02. Based on our testing, this, combined with Canon’s residual link, usually gives a very stable performance improvement, without ever hurting.

38GLA: ℓ layers, hidden size 64h, 4 fixed attention heads. Mamba2: 2ℓ layers, hidden size 64h (ssm state size 64 and num heads 16). Mamba2(mlp): ℓ layers, hidden size 64h (ssm state size 64 and num heads 16). GDN: ℓ layers, hidden size 64d, and max{4, 64d/128} heads. This ensures comparable parameter counts across architectures.

We use causal conv1d [23] for its fast CUDA implementation. Canon layers are applied after layer normalization (if present, e.g., Canon-A/C) and before any non-linearity (if present, e.g., Canon-B/D).

We refer to the original conv1d implementations inside GLA/GDN/Mamba2 as Canon-b, and we leave its configuration identical to what was proposed by the original authors. In particular:

- • conv1d in GLA/GDN has 0.02 initialization, with activation, without residual;
- • conv1d in Mamba2 has O(1) initialization, with activation, without residual.

We refer to cst-Canon as the constant, untrained version of Canon(res), where the convolution weights are fixed to PyTorch’s default initialization.

Our implementation of Canon-ABCD for Llama, as well as Canon-AbCD for GLA, GDN and Mamba2, have been open-sourced on GitHub [2] (up-to-date links at physics.allen-zhu.com).

#### D Extensions of Figures 8 and 15

Task Depo2(K=16) | N=75 | Llama(RoPE) - original

Task Depo2(K=16) | N=125 | Llama(RoPE) - original

Task Depo2(K=16) | N=100 | Llama(RoPE) - original

100

100

100

k=16

k=16

k=16

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

Task Depo2(K=16) | N=75 | Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16) | N=125 | Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16) | N=100 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

k=16

k=16

k=16

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

Task Depo1(K=8) | N=300 | Llama(RoPE) - original

Task Depo1(K=8) | N=375 | Llama(RoPE) - original

Task Depo1(K=8) | N=225 | Llama(RoPE) - original

100

100

100

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |k=1<br>k=2<br>| |
| | | | | | | | | | | | |
| | | | | | | | | | |k=4| |
| | | | | | | | | | | | |
| | | | | | | | | | |k=8| |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=8) | N=300 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=8) | N=375 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=8) | N=225 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=8) | N=300 | Llama(RoPE) - original

Task Depo1(K=8) | N=375 | Llama(RoPE) - original

Task Depo1(K=8) | N=225 | Llama(RoPE) - original

100

100

100

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | |k=|1| |
| | | | | | |k= k=<br><br>|2 4| |
| | | | | | |k=|8| |
| | | | | | | | | |

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

0 20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 8L512D)

Task Depo1(K=8) | N=300 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=8) | N=375 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=8) | N=225 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

0 20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 8L512D)

Task Depo1(K=4) | N=300 | Llama(RoPE) - original

Task Depo1(K=4) | N=375 | Llama(RoPE) - original

Task Depo1(K=4) | N=225 | Llama(RoPE) - original

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

k=1 k=2 k=4

- k=1

- k=2

85

85

85

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=4) | N=300 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=4) | N=375 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=4) | N=225 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 Train steps (model size = 12L768D)

Task Depo1(K=4) | N=300 | Llama(RoPE) - original

Task Depo1(K=4) | N=375 | Llama(RoPE) - original

Task Depo1(K=4) | N=225 | Llama(RoPE) - original

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

0 20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 8L512D)

Task Depo1(K=4) | N=300 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=4) | N=375 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=4) | N=225 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

0 20000 40000 60000 80000 100000 Train steps (model size = 8L512D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 8L512D)

###### Figure 19: This is an extension of Figure 8: Training curves for 12L768 and 8L512D RoPE models, with and without Canon layers, on Task Depo2(K = 16), Depo1(K = 8), Depo1(K = 4), evaluated at varied depths and maximum graph size n = N, shown in two best learning rates.

Task Depo1(K=4) | N=300 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=4) | N=375 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=4) | N=225 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 Train steps (model size = 12L768D)

Task Depo1(K=4) | N=300 | Mamba2(mlp) - Canon-AbCD(res)

Task Depo1(K=4) | N=375 | Mamba2(mlp) - Canon-AbCD(res)

Task Depo1(K=4) | N=225 | Mamba2(mlp) - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=4) | N=300 | GLA - Canon-AbCD(res)

Task Depo1(K=4) | N=375 | GLA - Canon-AbCD(res)

Task Depo1(K=4) | N=225 | GLA - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=4) | N=300 | GDN - Canon-AbCD(res)

Task Depo1(K=4) | N=375 | GDN - Canon-AbCD(res)

Task Depo1(K=4) | N=225 | GDN - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

95

95

95

90

90

90

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

85

85

85

k=4

k=4

k=4

80

80

80

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=8) | N=300 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=8) | N=375 | Llama(RoPE) - Canon-ABCD(res)

Task Depo1(K=8) | N=225 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=8) | N=300 | Mamba2(mlp) - Canon-AbCD(res)

Task Depo1(K=8) | N=375 | Mamba2(mlp) - Canon-AbCD(res)

Task Depo1(K=8) | N=225 | Mamba2(mlp) - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=8) | N=300 | GLA - Canon-AbCD(res)

Task Depo1(K=8) | N=375 | GLA - Canon-AbCD(res)

Task Depo1(K=8) | N=225 | GLA - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo1(K=8) | N=300 | GDN - Canon-AbCD(res)

Task Depo1(K=8) | N=375 | GDN - Canon-AbCD(res)

Task Depo1(K=8) | N=225 | GDN - Canon-AbCD(res)

100

100

100

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

10000 20000 30000 40000 50000 60000 70000 80000 Train steps (model size = 12L768D)

Task Depo2(K=16) | N=75 | Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16) | N=125 | Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16) | N=100 | Llama(RoPE) - Canon-ABCD(res)

100

100

100

k=16

k=16

k=16

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

Task Depo2(K=16) | N=75 | Mamba2(mlp) - Canon-AbCD(res)

Task Depo2(K=16) | N=125 | Mamba2(mlp) - Canon-AbCD(res)

Task Depo2(K=16) | N=100 | Mamba2(mlp) - Canon-AbCD(res)

100

100

100

k=16

k=16

k=16

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

Task Depo2(K=16) | N=75 | GLA - Canon-AbCD(res)

Task Depo2(K=16) | N=125 | GLA - Canon-AbCD(res)

Task Depo2(K=16) | N=100 | GLA - Canon-AbCD(res)

100

100

100

k=16

k=16

k=16

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

Task Depo2(K=16) | N=75 | GDN - Canon-AbCD(res)

Task Depo2(K=16) | N=125 | GDN - Canon-AbCD(res)

Task Depo2(K=16) | N=100 | GDN - Canon-AbCD(res)

100

100

100

k=16

k=16

k=16

Accuracyonk

Accuracyonk

Accuracyonk

75

75

75

- k=1

- k=2

- k=1

- k=2

- k=1

- k=2

50

50

50

k=4 k=8

k=4 k=8

k=4 k=8

25

25

25

0

0

0

20000 40000 60000 80000 100000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 140000 Train steps (model size = 12L768D)

0 20000 40000 60000 80000 100000 120000 Train steps (model size = 12L768D)

###### Figure 20: This is an extension of Figure 15: Training curves for 12L768D architectures on Task Depo1(K=4 or 8), Depo2(K=16), evaluated at varied k and maximum n = N; two best learning rates for each k.

#### E More Real-Life Experiments

##### E.1 Insufficiencies of Real-Life Pretraining at Academic Scale

###### As shown earlier in Figure 1, real-life pretrained models (FineWeb-Edu or SlimPajama) display large performance variance across random seeds. Here, we expand those results in Figure 21, including full experiments over eight seeds. Following feedback from an anonymous NeurIPS 2025 reviewer, we further test a controlled setup where data order is fixed and only model initialization varies—yet substantial benchmark variance remains.

###### FineWeb-Edu | 100B token pretrain | 1.3B models

48.1% 73.0% 59.3% 59.4% 72.6% 41.1% 42.1% 63.5% 16.7 13.1 71.7% 78.3% 49.7% 60.5% 29.9% 34.3% 45.2% 55.3% 64.3% 61.0% 24.9% 36.6%

- Llama - seed 20

- Llama - seed 21

- Llama - seed 22

- Llama - seed 23

- Llama - seed 24

- Llama - seed 25

- Llama - seed 26

- Llama - seed 27 48.5% 72.7% 58.8% 60.7% 71.7% 39.8% 42.4% 63.4% 16.7 12.3 71.3% 82.7% 46.2% 56.5% 29.6% 33.0% 44.8% 55.1% 63.1% 62.5% 24.5% 34.9%

[Figure 212]

- 49.5% 73.1% 58.9% 59.0% 72.6% 40.7% 42.1% 64.6% 16.7 11.6 73.6% 81.3% 51.7% 58.1% 31.3% 34.8% 45.4% 56.8% 63.9% 63.0% 25.8% 34.8%

- 49.7% 73.0% 59.4% 59.5% 72.9% 39.6% 42.3% 63.1% 16.6 11.7 73.3% 79.7% 49.5% 55.0% 31.4% 32.7% 45.0% 57.8% 64.9% 63.2% 24.7% 33.9%

- 49.2% 72.0% 58.9% 59.2% 71.3% 39.7% 42.6% 65.0% 16.7 12.2 76.6% 80.2% 51.7% 56.4% 28.6% 32.3% 44.2% 56.5% 62.0% 64.7% 24.7% 36.0%
- 49.2% 73.2% 59.0% 58.8% 72.3% 39.1% 43.4% 64.6% 16.8 11.8 68.8% 81.5% 52.0% 56.8% 29.8% 31.2% 44.6% 57.2% 64.0% 61.8% 23.8% 36.3%

48.7% 72.9% 58.8% 59.0% 71.4% 37.9% 42.8% 63.6% 16.6 12.7 75.4% 82.1% 50.4% 56.9% 30.2% 34.0% 45.0% 56.1% 65.6% 62.8% 24.9% 37.6%

- 50.0% 73.8% 59.0% 59.7% 71.8% 38.7% 42.3% 62.5% 16.8 11.6 70.8% 77.8% 45.3% 52.3% 29.5% 33.9% 44.7% 56.3% 62.1% 61.9% 20.6% 34.5%

- 47.4% 73.4% 58.6% 59.6% 71.7% 40.4% 42.9% 63.2% 16.6 13.3 73.1% 73.4% 48.9% 54.7% 28.5% 31.5% 48.5% 55.3% 63.4% 61.3% 24.3% 38.2%
- 48.9% 72.5% 59.1% 59.0% 71.4% 41.1% 42.9% 62.3% 16.9 12.6 74.7% 79.7% 49.2% 54.7% 30.4% 33.7% 49.2% 58.2% 65.0% 61.8% 24.8% 37.4%

- 50.1% 72.7% 58.9% 60.5% 72.2% 41.0% 42.3% 63.5% 16.8 12.0 74.2% 76.8% 53.6% 52.6% 31.0% 34.1% 45.5% 57.4% 64.5% 63.6% 24.3% 37.3%

- Llama - fix data - seed 20

- Llama - fix data - seed 21

- Llama - fix data - seed 22

- Llama - fix data - seed 23

- Llama - fix data - seed 24

- Llama - fix data - seed 25

- Llama - fix data - seed 26

- Llama - fix data - seed 27

- 49.4% 72.3% 59.5% 58.3% 72.0% 39.8% 42.9% 63.5% 16.9 11.7 75.3% 81.1% 52.7% 58.1% 28.8% 32.9% 45.2% 57.0% 64.3% 62.7% 24.0% 33.8%

- 49.8% 72.9% 59.5% 59.0% 73.1% 40.9% 42.4% 64.6% 16.8 11.8 72.1% 81.9% 49.4% 57.6% 31.8% 34.7% 46.2% 56.7% 63.4% 62.8% 25.0% 37.0%

51.1% 72.4% 59.1% 59.3% 72.7% 40.3% 41.8% 62.2% 16.8 11.1 69.1% 81.0% 52.5% 56.0% 31.1% 32.8% 45.1% 56.8% 65.0% 62.9% 25.8% 36.8%

49.5% 72.7% 59.4% 60.2% 71.5% 40.7% 41.8% 63.5% 16.6 11.7 65.0% 78.8% 50.2% 55.9% 30.7% 34.4% 43.5% 57.5% 63.0% 62.6% 24.7% 32.7%

- 50.8% 72.4% 59.2% 59.3% 73.4% 41.2% 42.2% 63.8% 16.7 11.3 74.5% 82.9% 51.1% 57.4% 29.5% 33.8% 45.3% 58.0% 63.0% 62.4% 22.4% 35.6%

LMB PIQA Hella WinoARC-eARC-c SIQABoolQWikipplLMBppl FDA FDA2SWDESWDE2 NQ NQ2SquadSquad2TriviaQATriviaQA2 DropDrop2

###### FineWeb-Edu | 100B token pretrain | 1.3B models

50.7% 50.8% 39.4% 32.5% 33.0% 28.6% 20.4% 13.9% 27.7% 26.9% 2.4% 8.7% 52.2% 40.5% 28.2% 19.2% 64.2% 63.1% 55.3% 50.3%

- Llama - seed 20

- Llama - seed 21

- Llama - seed 22

- Llama - seed 23

- Llama - seed 24

- Llama - seed 25

- Llama - seed 26

- Llama - seed 27

[Figure 213]

- Llama - fix data - seed 20

- Llama - fix data - seed 21

- Llama - fix data - seed 22

- Llama - fix data - seed 23

- Llama - fix data - seed 24

- Llama - fix data - seed 25

- Llama - fix data - seed 26

- Llama - fix data - seed 27

- 44.5% 53.1% 41.0% 36.0% 33.1% 26.7% 19.4% 14.1% 24.4% 27.6% 17.1% 16.2% 38.7% 40.6% 39.6% 28.9% 60.2% 58.5% 49.1% 35.6%

51.9% 48.7% 41.4% 31.0% 32.9% 25.1% 20.0% 16.6% 29.1% 28.7% 18.9% 14.8% 58.1% 38.1% 31.1% 24.5% 57.8% 53.8% 46.2% 34.0%

- 53.1% 58.2% 42.6% 32.9% 30.7% 25.3% 19.7% 13.5% 31.3% 28.7% 18.7% 17.5% 51.0% 43.3% 33.6% 21.8% 63.0% 61.9% 54.6% 37.5%

- 50.4% 54.7% 37.8% 30.1% 34.1% 22.7% 16.6% 11.7% 28.1% 27.1% 16.1% 13.0% 44.6% 38.7% 37.1% 22.2% 59.2% 60.2% 55.3% 34.4%
- 51.2% 61.1% 38.9% 32.4% 30.9% 24.9% 20.7% 12.7% 25.5% 23.5% 14.4% 14.7% 53.8% 43.7% 30.6% 21.0% 52.0% 59.8% 54.5% 35.1%

53.5% 57.0% 43.3% 41.3% 34.4% 29.7% 22.4% 15.1% 29.0% 28.1% 17.2% 18.0% 52.8% 36.5% 26.2% 25.7% 60.3% 59.6% 51.4% 50.1%

55.1% 63.5% 46.7% 30.5% 34.3% 23.2% 19.6% 15.4% 25.4% 23.9% 14.7% 14.5% 53.4% 40.1% 28.3% 28.1% 65.8% 56.7% 51.0% 37.3%

- 53.2% 51.1% 41.7% 28.5% 35.3% 26.9% 20.0% 12.3% 26.5% 27.1% 14.9% 14.1% 56.9% 38.4% 31.2% 22.6% 61.8% 61.5% 54.9% 39.3%

47.5% 54.9% 40.9% 35.2% 28.5% 26.2% 23.3% 14.8% 27.7% 24.6% 18.4% 11.3% 54.4% 39.7% 31.9% 22.8% 55.0% 57.8% 56.4% 43.3%

- 47.8% 51.2% 33.6% 38.3% 30.5% 25.4% 19.2% 16.2% 25.6% 25.6% 18.8% 14.1% 51.6% 34.5% 27.9% 25.2% 60.2% 66.9% 54.1% 43.5%
- 47.9% 44.7% 34.0% 35.5% 30.8% 24.8% 18.9% 13.4% 22.6% 28.0% 6.5% 12.2% 53.6% 37.5% 30.6% 18.6% 60.1% 55.7% 48.2% 38.0%

53.4% 50.5% 41.3% 40.1% 35.5% 28.0% 21.4% 15.8% 33.9% 28.0% 12.1% 13.0% 57.2% 37.8% 29.6% 23.1% 60.7% 64.7% 57.4% 36.2%

45.7% 61.5% 38.8% 33.1% 34.0% 31.5% 22.8% 12.4% 27.9% 29.5% 4.8% 13.0% 56.8% 41.9% 31.5% 19.3% 59.8% 61.2% 49.8% 37.5%

47.0% 56.5% 43.5% 34.8% 32.0% 22.0% 18.1% 15.3% 23.5% 22.2% 14.9% 11.6% 57.8% 33.2% 26.8% 21.9% 55.0% 62.1% 54.7% 43.9%

- 53.3% 52.7% 38.8% 31.9% 35.0% 28.3% 17.6% 14.7% 33.0% 31.8% 13.3% 12.3% 50.1% 41.3% 30.6% 22.7% 62.0% 60.5% 53.7% 39.4%

qa1-0kqa1-1kqa1-2kqa1-4k qa2-0kqa2-1kqa2-2kqa2-4k qa3-0kqa3-1kqa3-2kqa3-4k qa4-0kqa4-1kqa4-2kqa4-4k qa5-0kqa5-1kqa5-2kqa5-4k

###### SlimPajama | 100B token pretrain | 1.3B models

- 52.3% 71.2% 56.1% 58.7% 59.3% 29.0% 40.9% 60.8% 15.8 10.4 79.4% 84.6% 54.5% 61.9% 36.2% 36.0% 48.9% 56.4% 64.2% 59.1% 27.8% 39.3%
- 53.3% 71.4% 56.2% 58.5% 59.1% 29.9% 40.9% 61.2% 15.7 10.1 79.0% 86.6% 52.8% 62.3% 33.8% 35.3% 46.5% 55.4% 61.4% 60.6% 26.8% 40.3%

- Llama - seed 20

- Llama - seed 21

- Llama - seed 22

- Llama - seed 23

- Llama - seed 24

- Llama - seed 25

- Llama - seed 26

- Llama - seed 27

[Figure 214]

50.7% 72.1% 56.0% 57.9% 59.5% 29.6% 40.8% 61.1% 15.7 11.3 78.7% 85.4% 53.8% 63.5% 34.1% 34.4% 46.6% 57.3% 62.9% 60.3% 24.7% 38.6%

- 52.8% 72.1% 56.0% 58.8% 58.5% 29.0% 41.0% 60.7% 15.7 10.2 76.2% 85.5% 50.9% 59.7% 33.9% 37.5% 49.4% 57.3% 63.2% 61.4% 25.7% 39.1%
- 53.1% 71.9% 56.5% 59.2% 60.8% 29.2% 41.3% 61.8% 15.7 9.9 81.5% 87.2% 52.6% 62.2% 34.5% 36.6% 48.3% 55.4% 62.3% 59.8% 23.0% 42.8%

- 51.8% 71.6% 56.2% 57.1% 59.0% 29.8% 40.9% 62.4% 15.6 10.4 79.9% 86.2% 56.1% 65.6% 33.6% 35.3% 46.9% 55.1% 64.0% 60.0% 26.3% 41.2%

50.8% 72.3% 55.8% 57.2% 59.1% 29.6% 41.1% 63.5% 15.7 11.2 79.6% 85.6% 50.7% 59.4% 32.9% 36.0% 45.4% 55.2% 62.5% 60.0% 26.8% 40.8%

55.0% 72.0% 56.0% 57.7% 58.9% 29.1% 41.1% 62.3% 15.7 9.8 80.3% 85.2% 52.1% 60.8% 35.0% 36.7% 46.9% 59.7% 63.4% 59.7% 24.0% 39.4%

- 52.3% 71.6% 56.7% 58.2% 59.4% 31.0% 42.1% 62.9% 15.8 10.3 81.3% 87.4% 54.8% 59.4% 33.9% 36.2% 49.1% 54.4% 63.7% 61.4% 27.0% 41.1%

- Llama - fix data - seed 20

- Llama - fix data - seed 21

- Llama - fix data - seed 22

- Llama - fix data - seed 23

- Llama - fix data - seed 24

- Llama - fix data - seed 25

- Llama - fix data - seed 26

- Llama - fix data - seed 27

- 51.6% 71.3% 56.0% 58.2% 59.0% 28.4% 40.6% 61.3% 15.7 10.5 72.2% 86.4% 55.3% 62.8% 34.5% 36.5% 45.8% 56.0% 63.0% 61.8% 24.7% 38.5%
- 52.6% 71.7% 56.2% 57.3% 59.7% 30.1% 41.2% 64.1% 15.7 10.3 76.8% 84.7% 55.4% 61.3% 34.9% 36.0% 45.8% 57.0% 63.0% 60.9% 27.9% 40.8%

52.6% 71.6% 55.6% 57.4% 59.8% 29.9% 40.6% 62.3% 15.8 10.2 81.4% 86.3% 58.1% 63.0% 34.0% 35.0% 46.9% 56.6% 62.6% 60.3% 25.8% 39.2%

- 51.5% 71.6% 55.9% 57.8% 59.5% 29.7% 41.1% 62.8% 15.7 10.9 78.3% 86.5% 55.4% 62.2% 33.1% 35.7% 47.1% 56.1% 63.2% 60.4% 25.5% 41.4%

- 51.4% 73.4% 56.0% 57.9% 58.8% 29.0% 41.0% 62.8% 15.7 10.5 75.3% 86.8% 56.3% 61.9% 32.9% 36.2% 46.6% 56.0% 62.5% 59.2% 24.8% 38.9%
- 52.3% 72.0% 56.3% 57.5% 60.9% 29.9% 40.7% 62.7% 15.7 10.4 76.5% 88.3% 55.5% 60.8% 35.0% 37.4% 47.6% 54.9% 63.6% 60.1% 25.8% 37.7%

- 51.6% 71.2% 56.2% 58.8% 59.1% 29.7% 41.6% 61.9% 15.7 10.6 75.0% 85.6% 54.2% 61.0% 35.1% 35.2% 49.4% 56.1% 63.6% 60.0% 27.3% 40.2%

LMB PIQA Hella WinoARC-eARC-c SIQABoolQWikipplLMBppl FDA FDA2SWDESWDE2 NQ NQ2SquadSquad2TriviaQATriviaQA2 DropDrop2

###### SlimPajama | 100B token pretrain | 1.3B models

- 53.5% 61.7% 51.8% 36.2% 39.8% 29.0% 22.7% 17.0% 31.3% 32.4% 29.3% 12.2% 58.9% 37.3% 31.3% 28.0% 70.1% 67.3% 61.0% 49.3%

- 57.1% 65.1% 46.8% 29.9% 36.8% 31.2% 21.4% 18.6% 30.1% 30.7% 24.9% 10.1% 56.5% 34.3% 29.3% 26.8% 66.3% 68.5% 63.1% 49.5%

55.7% 57.1% 48.6% 37.6% 35.0% 30.7% 29.4% 17.7% 31.1% 31.7% 27.5% 12.9% 58.6% 38.9% 29.7% 25.3% 62.5% 62.6% 54.3% 43.8%

55.1% 60.7% 33.6% 34.3% 37.4% 31.8% 25.3% 18.9% 35.6% 34.7% 23.7% 12.4% 55.6% 40.7% 35.3% 26.0% 67.5% 63.2% 63.7% 49.7%

- 51.1% 64.2% 48.1% 35.3% 37.1% 28.7% 23.0% 18.3% 33.6% 31.2% 22.8% 10.1% 61.0% 42.5% 29.2% 24.4% 66.3% 67.4% 57.2% 37.6%

54.3% 64.7% 46.9% 38.2% 34.7% 27.6% 21.6% 17.3% 31.4% 28.0% 27.4% 13.2% 55.8% 40.7% 36.2% 25.1% 72.3% 63.8% 58.4% 46.3%

53.8% 58.6% 47.7% 35.7% 39.0% 30.7% 23.0% 17.1% 36.2% 32.4% 26.0% 8.9% 57.2% 31.6% 27.8% 23.2% 64.2% 59.7% 54.4% 39.6%

- 52.4% 59.3% 49.4% 37.2% 33.0% 26.8% 20.2% 16.3% 27.2% 26.2% 23.5% 10.6% 59.5% 36.5% 28.4% 21.4% 60.5% 57.6% 59.9% 45.9%

55.1% 63.5% 55.9% 31.6% 37.9% 27.1% 23.9% 20.8% 34.5% 30.3% 28.1% 12.8% 60.9% 35.0% 31.0% 25.7% 69.7% 64.0% 60.6% 54.0%

52.2% 59.0% 43.2% 32.3% 33.5% 28.9% 23.2% 18.4% 35.1% 32.5% 22.0% 10.1% 58.9% 44.6% 34.3% 26.9% 63.6% 71.0% 63.6% 41.7%

59.9% 66.7% 53.6% 33.8% 35.9% 30.7% 25.5% 18.7% 29.8% 29.1% 19.1% 8.8% 57.7% 46.1% 35.7% 24.6% 73.8% 70.4% 65.7% 46.8%

50.9% 54.4% 43.0% 37.7% 30.3% 27.1% 23.1% 16.4% 25.8% 30.7% 24.5% 10.8% 54.0% 39.5% 31.1% 21.6% 57.4% 62.8% 57.6% 47.7%

- 58.7% 64.7% 51.2% 36.5% 37.5% 29.8% 23.9% 14.7% 32.9% 30.2% 24.6% 11.4% 57.4% 41.6% 29.4% 22.0% 70.2% 68.8% 63.1% 47.6%

- 56.5% 65.1% 56.0% 28.6% 36.7% 32.9% 27.5% 19.1% 30.9% 29.7% 25.3% 15.4% 54.6% 41.5% 32.4% 26.1% 68.3% 73.5% 68.7% 52.9%
- 57.0% 61.8% 50.8% 33.0% 38.9% 31.1% 25.1% 17.0% 34.3% 32.8% 26.5% 15.7% 59.9% 44.0% 39.6% 21.3% 62.6% 67.6% 61.6% 53.4%
- 58.2% 58.5% 47.2% 37.0% 32.7% 25.3% 22.2% 22.0% 27.1% 27.6% 26.3% 9.8% 56.8% 38.3% 39.0% 27.5% 63.8% 66.7% 56.4% 52.6%

- Llama - seed 20

- Llama - seed 21

- Llama - seed 22

- Llama - seed 23

- Llama - seed 24

- Llama - seed 25

- Llama - seed 26

- Llama - seed 27

[Figure 215]

- Llama - fix data - seed 20

- Llama - fix data - seed 21

- Llama - fix data - seed 22

- Llama - fix data - seed 23

- Llama - fix data - seed 24

- Llama - fix data - seed 25

- Llama - fix data - seed 26

- Llama - fix data - seed 27

qa1-0kqa1-1kqa1-2kqa1-4k qa2-0kqa2-1kqa2-2kqa2-4k qa3-0kqa3-1kqa3-2kqa3-4k qa4-0kqa4-1kqa4-2kqa4-4k qa5-0kqa5-1kqa5-2kqa5-4k

- Figure 21: Extended results to Figure 1 showing strong variance in benchmark accuracies for academic-scale real-life pretraining (1.3B models trained for 100B tokens). Observations: Accuracy varies greatly across random seeds—both when changing data and initialization, and even when fixing data but varying initialization. HellaSwag [79] and wiki-ppl are relatively stable, though perplexity alone is an unreliable indicator of model capability.

##### E.2 Complete Real-Life Experiments

###### SlimPajama | 100B token pretrain | 1.3B models

- Llama(RoPE) - original - seed 20

- Llama(RoPE) - original - seed 21

- Llama(RoPE) - original - seed 22

- Llama(RoPE) - original - seed 23

- Llama(RoPE) - original - seed 24

- Llama(RoPE) - original - seed 25

- Llama(RoPE) - original - seed 26

- Llama(RoPE) - original - seed 27

[Figure 216]

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(NoPE) - original - seed 20

Llama(NoPE) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - original - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE) - original - seed 20

GPT2(RoPE) - Canon-ABCD(res) - seed 20

GPT2(RoPE) - Canon-ABCD(res) - seed 20

Mamba2(mlp) - original(conv1d) - seed 20

Mamba2(mlp) - noconv1d - seed 20

Mamba2(mlp) - Canon-AbCD(res) - seed 20

Mamba2(mlp) - Canon-ABCD(res) - seed 20

Mamba2 - original(conv1d) - seed 20

Mamba2 - noconv1d - seed 20

Mamba2 - Canon-Ab(res) - seed 20

Mamba2 - Canon-AB(res) - seed 20

GLA - original(noconv1d) - seed 20

GLA - conv1d - seed 20

GLA - Canon-AbCD(res) - seed 20

GDN - original(conv1d) - seed 20

GDN - noconv1d - seed 20

- 52.1% 73.2% 57.6% 59.0% 61.3% 29.3% 40.4% 61.3% 15.4 9.9 64.9% 78.0% 50.7% 54.5% 31.1% 38.5% 39.0% 60.2% 61.7% 62.0% 25.0% 38.5% 83.8% 74.3% 56.7% 42.9% 32.8% 22.6% 16.1% 31.8% 33.2% 29.9% 28.5% 25.5% 21.2% 18.0%

- 50.6% 72.3% 56.7% 56.5% 60.5% 29.8% 42.4% 62.5% 14.4 10.7 65.0% 79.2% 48.8% 57.6% 31.0% 39.0% 39.6% 58.6% 61.1% 61.8% 23.9% 39.2% 79.0% 67.6% 52.8% 41.5% 34.4% 25.8% 19.0% 30.1% 29.0% 26.4% 26.5% 20.7% 19.1% 16.6%
- 51.1% 72.2% 55.8% 58.5% 61.0% 29.6% 41.8% 61.7% 14.8 10.7 60.2% 75.8% 52.1% 55.6% 31.0% 38.3% 39.4% 60.0% 61.8% 62.6% 23.9% 40.4% 68.8% 59.4% 47.0% 36.4% 28.9% 17.9% 12.6% 31.8% 29.0% 28.2% 28.6% 24.8% 20.5% 16.7%

53.8% 73.4% 58.0% 59.2% 61.4% 31.2% 42.1% 62.9% 15.2 9.4 66.1% 79.6% 51.5% 55.7% 33.0% 38.7% 40.0% 60.7% 62.3% 62.2% 22.7% 40.3% 96.0% 90.8% 76.1% 64.9% 50.6% 38.7% 31.3% 33.2% 32.2% 28.2% 27.5% 24.8% 22.1% 17.7%

50.0% 71.8% 54.1% 57.1% 58.8% 29.8% 40.6% 61.0% 15.3 10.9 61.6% 77.7% 50.6% 59.0% 30.3% 35.5% 38.7% 61.2% 59.8% 59.0% 23.1% 36.7% 84.0% 74.2% 51.3% 35.6% 24.0% 18.4% 11.8% 35.7% 33.6% 27.8% 28.5% 25.7% 22.5% 16.8%

44.2% 71.2% 47.9% 54.0% 55.2% 27.4% 40.3% 61.5% 17.1 15.8 58.5% 73.4% 42.4% 56.7% 27.2% 35.0% 35.0% 59.1% 55.7% 55.6% 22.2% 38.6% 71.6% 55.0% 30.5% 18.4% 12.1% 10.4% 9.2% 32.0% 29.6% 24.3% 23.6% 20.9% 15.6% 13.6%

- 50.6% 72.5% 55.5% 57.1% 60.0% 29.4% 40.9% 60.6% 15.7 10.7 68.2% 79.3% 50.0% 56.0% 29.8% 36.8% 38.8% 56.7% 60.1% 59.3% 22.2% 37.1% 75.0% 63.8% 42.5% 26.8% 19.0% 12.8% 8.7% 33.1% 32.5% 29.4% 27.9% 24.5% 16.1% 11.8%
- 51.6% 73.3% 57.2% 57.3% 61.0% 30.1% 41.6% 62.5% 15.5 10.3 64.9% 77.7% 50.1% 59.0% 32.1% 39.3% 37.6% 58.4% 60.9% 61.4% 23.3% 37.0% 80.5% 64.3% 38.8% 28.1% 17.3% 11.7% 8.0% 34.4% 33.1% 32.0% 29.9% 26.6% 17.9% 13.9%

42.4% 70.5% 49.2% 54.8% 54.7% 27.2% 40.7% 60.3% 18.6 16.8 48.2% 64.4% 44.0% 49.1% 27.1% 33.0% 33.6% 56.5% 55.4% 57.3% 20.9% 34.1% 49.2% 34.6% 21.3% 15.4% 12.6% 9.8% 8.3% 33.0% 29.1% 28.6% 25.2% 21.0% 14.4% 10.1%

- 49.5% 72.6% 57.0% 57.1% 60.5% 29.1% 41.0% 60.7% 15.5 11.9 56.4% 74.3% 48.2% 55.7% 30.9% 38.4% 38.3% 58.6% 60.0% 59.8% 24.3% 38.6% 75.4% 64.8% 40.4% 27.8% 16.6% 12.0% 9.4% 34.0% 33.9% 31.4% 34.3% 26.8% 23.1% 20.8%
- 50.3% 72.7% 55.9% 57.2% 60.3% 29.5% 42.0% 61.1% 15.7 10.9 69.8% 77.9% 46.8% 57.3% 30.6% 38.5% 38.5% 57.3% 61.0% 61.4% 21.5% 37.4% 72.7% 54.8% 37.0% 31.9% 25.9% 17.2% 13.5% 35.6% 31.9% 28.1% 28.8% 25.1% 17.2% 15.1%

- 48.8% 73.9% 56.6% 59.2% 59.8% 30.5% 41.8% 62.5% 15.7 11.7 63.6% 75.8% 46.1% 53.7% 30.4% 36.8% 38.3% 58.5% 60.4% 61.5% 22.4% 36.3% 62.3% 50.5% 34.8% 24.4% 18.3% 12.7% 8.5% 34.8% 32.8% 32.6% 31.2% 25.4% 19.5% 14.5%

- 42.8% 71.9% 50.9% 55.2% 55.3% 27.8% 41.2% 61.1% 17.7 15.6 54.3% 62.6% 38.4% 50.7% 26.8% 32.2% 34.4% 56.4% 55.5% 58.4% 21.3% 38.1% 67.9% 52.9% 37.4% 26.9% 18.2% 14.5% 9.0% 33.1% 31.5% 24.9% 24.7% 18.8% 13.3% 10.0%

- 48.5% 72.5% 56.6% 57.8% 59.2% 30.0% 40.9% 62.1% 15.8 11.8 56.4% 72.1% 45.2% 51.0% 31.2% 36.1% 37.7% 59.7% 59.7% 60.2% 21.4% 37.8% 75.7% 65.2% 42.8% 30.7% 23.7% 15.5% 12.8% 34.3% 32.8% 28.6% 29.1% 21.7% 15.8% 11.6%
- 49.4% 71.4% 54.9% 55.7% 60.5% 30.3% 40.8% 62.6% 15.8 13.3 77.7% 86.7% 54.1% 58.5% 34.6% 40.4% 48.8% 61.4% 63.3% 60.4% 23.9% 38.4% 98.2% 98.1% 97.2% 94.7% 93.8% 26.4% 3.7% 34.2% 33.9% 34.1% 29.5% 32.4% 20.9% 3.5%

49.2% 71.8% 55.6% 57.0% 60.7% 28.8% 40.8% 62.2% 15.6 13.5 78.1% 85.6% 58.1% 62.6% 32.9% 39.9% 47.0% 61.2% 63.2% 60.8% 26.6% 40.2% 98.2% 98.6% 99.0% 98.8% 96.2% 70.5% 19.7% 33.4% 33.1% 33.0% 29.2% 32.2% 27.5% 20.4%

48.9% 70.8% 55.0% 58.1% 58.4% 28.2% 40.9% 62.7% 16.0 12.5 76.7% 86.1% 56.2% 60.5% 32.9% 36.2% 48.2% 62.2% 62.3% 60.2% 23.3% 40.9% 99.1% 98.7% 94.7% 89.2% 68.9% 3.9% 0.5% 32.8% 30.6% 29.3% 27.4% 28.5% 4.3% 0.3%

- 51.2% 71.8% 57.0% 58.6% 61.1% 29.6% 41.5% 62.1% 15.5 11.0 81.0% 84.9% 56.9% 62.4% 34.9% 37.4% 47.9% 61.4% 65.3% 61.6% 25.1% 41.1% 97.9% 97.8% 94.2% 93.9% 82.2% 2.7% 0.9% 34.2% 33.2% 31.9% 28.4% 30.5% 7.2% 1.1%
- 52.2% 72.2% 56.8% 57.9% 59.6% 30.8% 42.1% 61.9% 15.3 10.4 80.5% 87.5% 58.8% 60.1% 34.9% 42.2% 48.7% 61.1% 64.6% 61.6% 24.8% 41.2% 99.0% 98.8% 98.3% 97.0% 95.2% 84.0% 5.0% 34.1% 30.6% 32.5% 31.0% 32.6% 25.4% 5.4%

50.9% 71.8% 57.8% 57.4% 60.8% 31.2% 41.1% 61.6% 15.2 11.2 77.8% 87.7% 55.4% 60.9% 35.2% 41.3% 48.8% 60.4% 64.4% 61.4% 26.4% 40.3% 99.8% 99.7% 99.8% 99.6% 99.1% 85.5% 44.6% 32.7% 32.5% 33.0% 29.7% 30.5% 27.4% 19.7%

- 50.3% 72.1% 56.1% 56.0% 60.9% 28.8% 40.9% 62.1% 15.7 12.7 73.7% 86.7% 57.8% 61.9% 35.0% 37.2% 47.9% 61.8% 65.0% 61.4% 26.7% 42.8% 96.0% 93.4% 92.8% 86.1% 64.7% 3.6% 0.2% 33.6% 33.7% 32.1% 29.9% 28.7% 7.2% 0.0%

48.8% 72.1% 55.7% 58.6% 58.1% 29.3% 41.6% 62.8% 15.6 12.9 77.7% 86.5% 56.5% 59.9% 36.1% 40.2% 49.1% 58.3% 63.7% 59.8% 26.8% 40.9% 99.3% 99.4% 98.9% 98.8% 99.4% 88.9% 10.1% 36.1% 34.8% 35.0% 33.4% 34.3% 30.5% 2.6%

43.8% 70.3% 49.2% 54.5% 55.1% 27.3% 39.0% 60.4% 18.1 21.3 78.6% 84.8% 55.7% 63.4% 32.9% 35.9% 45.1% 53.6% 59.1% 55.7% 23.7% 38.3% 98.8% 98.8% 99.0% 98.5% 98.6% 66.8% 1.7% 28.9% 28.5% 25.5% 29.1% 28.4% 18.8% 1.6%

52.0% 72.4% 56.5% 60.4% 58.4% 29.0% 40.9% 62.8% 15.4 11.0 78.6% 86.9% 54.8% 58.4% 34.1% 35.0% 44.6% 56.2% 63.8% 60.6% 24.2% 39.6% 98.9% 99.1% 98.0% 95.6% 91.3% 7.0% 1.2% 34.4% 34.2% 33.3% 31.4% 30.8% 6.2% 1.2%

- 51.4% 71.7% 56.5% 58.9% 60.1% 29.9% 40.8% 63.2% 15.4 10.4 78.6% 87.2% 57.3% 61.5% 35.4% 40.0% 46.2% 59.0% 65.5% 61.7% 25.8% 41.9% 99.7% 99.8% 99.1% 98.6% 95.3% 60.8% 18.6% 35.0% 32.3% 33.0% 30.6% 34.5% 29.1% 10.4%

51.7% 71.7% 57.2% 57.8% 60.9% 29.4% 41.0% 63.1% 15.2 11.1 80.1% 87.4% 56.5% 62.6% 35.2% 39.4% 46.3% 57.1% 63.9% 61.4% 26.8% 42.3% 99.2% 99.6% 99.7% 99.3% 99.6% 80.6% 35.8% 34.2% 33.0% 32.5% 29.5% 31.5% 27.9% 25.1%

50.8% 72.3% 55.8% 57.2% 59.1% 29.6% 41.1% 63.5% 15.7 11.2 79.6% 85.6% 50.7% 59.4% 32.5% 36.0% 44.4% 55.2% 62.5% 60.0% 25.2% 40.8% 98.8% 96.6% 90.7% 84.0% 73.1% 14.8% 0.4% 30.5% 31.0% 30.5% 28.5% 29.3% 14.9% 0.2%

55.0% 72.0% 56.0% 57.7% 58.9% 29.1% 41.1% 62.3% 15.7 9.8 80.3% 85.2% 52.1% 60.8% 35.1% 36.7% 45.3% 59.7% 63.4% 59.7% 23.8% 39.4% 99.2% 98.8% 94.6% 94.2% 80.5% 5.1% 0.3% 31.1% 30.1% 29.7% 24.6% 26.1% 6.7% 0.0%

52.3% 71.6% 56.7% 58.2% 59.4% 31.0% 42.1% 62.9% 15.8 10.3 81.3% 87.4% 54.8% 59.4% 32.7% 36.2% 46.3% 54.4% 63.7% 61.4% 25.1% 41.1% 98.9% 98.0% 93.5% 89.3% 78.0% 4.4% 0.4% 29.2% 29.9% 28.4% 28.5% 27.7% 8.9% 0.2%

- 52.8% 72.1% 56.0% 58.8% 58.5% 29.0% 41.0% 60.7% 15.7 10.2 76.2% 85.5% 50.9% 59.7% 34.1% 37.5% 48.9% 57.3% 63.2% 61.4% 25.6% 39.1% 99.3% 98.8% 96.2% 92.6% 79.1% 9.7% 0.4% 31.8% 31.9% 30.8% 30.1% 27.2% 7.6% 0.2%
- 53.1% 71.9% 56.5% 59.2% 60.8% 29.2% 41.3% 61.8% 15.7 9.9 81.5% 87.2% 52.6% 62.2% 34.3% 36.6% 45.5% 55.4% 62.3% 59.8% 22.3% 42.8% 97.4% 95.5% 93.0% 84.4% 80.3% 8.5% 0.3% 31.2% 30.1% 30.5% 26.5% 26.4% 10.1% 0.2%

50.7% 72.1% 56.0% 57.9% 59.5% 29.6% 40.8% 61.1% 15.7 11.3 78.7% 85.4% 53.8% 63.5% 33.3% 34.4% 45.6% 57.3% 62.9% 60.3% 24.9% 38.6% 99.2% 96.6% 96.2% 91.1% 80.5% 9.4% 0.7% 30.4% 30.1% 28.4% 25.4% 26.9% 13.5% 0.4%

- 52.3% 71.2% 56.1% 58.7% 59.3% 29.0% 40.9% 60.8% 15.8 10.4 79.4% 84.6% 54.5% 61.9% 35.3% 36.0% 45.6% 56.4% 64.2% 59.1% 27.4% 39.3% 97.8% 98.7% 98.6% 95.7% 87.0% 7.3% 1.2% 31.0% 29.5% 27.8% 23.6% 23.8% 12.1% 1.3%
- 53.3% 71.4% 56.2% 58.5% 59.1% 29.9% 40.9% 61.2% 15.7 10.1 79.0% 86.6% 52.8% 62.3% 33.5% 35.3% 43.9% 55.4% 61.4% 60.6% 25.0% 40.3% 99.3% 98.0% 95.6% 89.4% 80.8% 6.2% 0.2% 31.6% 31.4% 28.3% 26.9% 27.8% 6.1% 0.0%

GDN - Canon-AbCD(res) - seed 20

LMB PIQA Hella Wino ARC-e ARC-c SIQA BoolQWikipplLMBppl FDA FDA2 SWDESWDE2 NQ NQ2 SquadSquad2TriviaQATriviaQA2 Drop Drop2 1-hop-0k1-hop-1k1-hop-2k1-hop-3k1-hop-4k1-hop-5k1-hop-6k 2-hop-0k2-hop-1k2-hop-2k2-hop-3k2-hop-4k2-hop-5k2-hop-6k

FineWeb-Edu | 100B token pretrain | 1.3B models

48.1% 73.0% 59.3% 59.4% 72.6% 41.1% 42.1% 63.5% 16.7 13.1 71.7% 78.3% 49.7% 60.5% 29.2% 34.3% 44.0% 55.3% 64.3% 61.0% 24.3% 36.6% 98.9% 97.6% 90.5% 86.3% 65.4% 3.3% 0.7% 28.9% 29.5% 29.3% 26.3% 26.2% 3.3% 0.4%

- Llama(RoPE) - original - seed 20

- Llama(RoPE) - original - seed 21

- Llama(RoPE) - original - seed 22

- Llama(RoPE) - original - seed 23

- Llama(RoPE) - original - seed 24

- Llama(RoPE) - original - seed 25

- Llama(RoPE) - original - seed 26

- Llama(RoPE) - original - seed 27

[Figure 217]

48.5% 72.7% 58.8% 60.7% 71.7% 39.8% 42.4% 63.4% 16.7 12.3 71.3% 82.7% 46.2% 56.5% 29.0% 33.0% 44.8% 55.1% 63.1% 62.5% 24.5% 34.9% 98.0% 96.2% 91.2% 83.2% 59.8% 7.0% 0.8% 32.5% 31.1% 28.7% 22.4% 24.7% 6.1% 0.3%

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(NoPE) - original - seed 20

Llama(NoPE) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - original - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE) - original - seed 20

GPT2(RoPE) - Canon-ABCD(res) - seed 20

GPT2(RoPE) - Canon-ABCD(res) - seed 20

Mamba2(mlp) - original(conv1d) - seed 20

Mamba2(mlp) - noconv1d - seed 20

Mamba2(mlp) - Canon-AbCD(res) - seed 20

Mamba2(mlp) - Canon-ABCD(res) - seed 20

Mamba2 - original(conv1d) - seed 20

Mamba2 - noconv1d - seed 20

Mamba2 - Canon-Ab(res) - seed 20

Mamba2 - Canon-AB(res) - seed 20

GLA - original(noconv1d) - seed 20

GLA - conv1d - seed 20

GLA - Canon-AbCD(res) - seed 20

GDN - original(conv1d) - seed 20

- 48.0% 74.1% 60.0% 58.9% 75.3% 42.4% 42.8% 64.3% 15.6 11.7 42.5% 57.3% 37.1% 48.0% 26.7% 33.2% 35.7% 61.5% 61.7% 64.4% 22.8% 38.0% 74.6% 65.3% 43.2% 29.4% 22.0% 12.9% 10.5% 33.5% 30.5% 26.2% 24.2% 20.0% 18.9% 13.1%

- 48.7% 72.6% 58.5% 59.8% 73.6% 40.0% 43.0% 63.7% 15.8 11.2 50.0% 61.1% 41.9% 46.7% 27.7% 34.0% 36.8% 60.2% 62.8% 62.7% 22.6% 39.0% 52.7% 43.0% 26.3% 15.1% 8.9% 7.0% 4.6% 32.8% 32.2% 26.7% 25.7% 19.9% 16.7% 13.2%

50.7% 74.0% 60.3% 61.7% 72.7% 40.4% 41.8% 64.7% 16.2 10.6 54.6% 71.0% 42.3% 51.2% 27.7% 33.4% 38.8% 60.4% 64.5% 62.6% 23.2% 38.4% 73.5% 62.4% 41.1% 22.0% 16.2% 10.8% 6.8% 35.0% 33.2% 26.3% 27.2% 21.7% 20.1% 14.9%

50.4% 73.3% 59.0% 60.4% 73.2% 40.7% 42.9% 63.7% 15.9 11.3 45.4% 61.8% 40.5% 50.9% 26.5% 32.4% 36.5% 60.1% 62.2% 63.3% 23.0% 37.3% 85.5% 77.3% 53.5% 35.2% 22.3% 13.9% 8.2% 31.7% 32.8% 24.1% 24.1% 16.9% 14.1% 10.8%

44.4% 72.5% 53.7% 56.4% 69.8% 37.0% 41.1% 63.3% 18.2 14.3 34.0% 49.4% 27.8% 42.1% 23.9% 28.7% 33.5% 60.4% 58.1% 62.9% 22.0% 35.3% 43.6% 30.9% 10.6% 4.9% 2.7% 2.5% 1.9% 31.2% 25.2% 19.8% 14.1% 10.9% 7.5% 4.9%

- 47.6% 73.4% 59.3% 59.3% 73.5% 39.2% 41.7% 62.6% 16.8 11.7 43.2% 51.4% 34.3% 42.8% 25.0% 31.9% 36.8% 58.9% 62.7% 62.1% 22.0% 36.9% 51.6% 35.2% 22.2% 16.9% 11.9% 7.8% 4.8% 32.6% 31.2% 23.7% 24.2% 18.1% 14.1% 10.7%

49.0% 73.4% 60.5% 58.4% 73.1% 41.1% 41.2% 63.8% 16.6 11.2 43.0% 51.3% 30.0% 40.2% 26.4% 31.5% 36.5% 56.8% 63.3% 64.0% 21.7% 35.9% 58.5% 47.0% 28.7% 20.5% 12.2% 7.5% 5.5% 30.9% 28.9% 26.4% 23.3% 16.7% 13.0% 9.3%

43.4% 72.0% 56.2% 55.0% 68.9% 36.3% 41.6% 62.0% 18.7 14.5 23.6% 30.7% 27.8% 33.5% 22.7% 28.1% 31.8% 56.2% 58.5% 61.7% 19.8% 33.0% 50.0% 32.6% 18.0% 9.1% 6.3% 4.6% 3.2% 30.3% 27.5% 22.9% 20.5% 14.2% 11.6% 9.9%

- 48.5% 73.6% 60.3% 60.3% 73.1% 41.3% 42.3% 63.5% 16.6 11.8 39.5% 49.6% 33.8% 42.7% 27.0% 31.3% 36.7% 57.0% 61.5% 62.7% 22.6% 38.7% 51.8% 38.4% 19.4% 13.1% 6.4% 5.5% 3.2% 33.4% 32.3% 24.8% 24.3% 20.8% 15.1% 13.6%

- 47.6% 73.1% 59.8% 57.9% 72.2% 39.1% 41.7% 62.4% 16.8 11.9 37.3% 50.7% 34.9% 43.3% 26.3% 32.7% 35.1% 58.7% 62.0% 64.4% 23.5% 37.1% 56.9% 33.5% 20.2% 14.2% 9.3% 6.5% 4.9% 31.6% 30.2% 23.0% 21.5% 16.3% 12.7% 10.1%

47.1% 74.0% 59.8% 59.9% 72.5% 39.8% 41.8% 64.7% 16.6 11.8 31.5% 44.3% 31.3% 43.2% 25.6% 30.7% 35.3% 57.3% 62.3% 63.6% 24.4% 38.6% 49.8% 35.1% 19.3% 12.5% 7.4% 5.1% 3.3% 32.7% 30.7% 25.6% 23.2% 16.6% 13.5% 8.6%

- 46.2% 73.1% 56.3% 57.3% 71.0% 37.5% 41.4% 63.5% 18.6 13.0 27.2% 33.1% 29.7% 37.4% 23.7% 28.0% 32.4% 54.1% 59.4% 63.4% 21.6% 36.8% 48.5% 28.3% 14.7% 7.0% 3.9% 3.5% 2.9% 30.5% 26.9% 21.9% 17.4% 10.9% 9.6% 5.1%
- 47.3% 74.3% 60.0% 59.6% 73.0% 40.4% 42.2% 64.7% 16.8 11.6 31.2% 42.4% 32.7% 37.7% 25.5% 31.6% 36.4% 58.9% 62.3% 63.8% 21.0% 38.8% 55.6% 37.0% 17.9% 10.3% 7.0% 6.2% 3.7% 31.9% 30.4% 24.4% 25.6% 17.9% 14.1% 10.5%

50.4% 71.9% 57.6% 58.1% 72.3% 38.9% 42.0% 64.0% 16.7 11.8 75.5% 83.8% 53.6% 60.1% 30.9% 37.4% 45.8% 61.6% 63.9% 61.7% 23.9% 36.7% 96.8% 96.5% 93.6% 89.5% 85.6% 47.5% 13.0% 33.7% 32.1% 34.0% 29.1% 31.6% 19.9% 12.6%

49.2% 72.4% 58.4% 58.4% 71.4% 39.0% 41.8% 65.0% 16.6 11.9 75.5% 82.7% 52.7% 58.4% 31.4% 36.9% 46.0% 59.2% 65.6% 62.1% 26.7% 39.7% 99.3% 99.5% 98.5% 98.4% 93.3% 57.8% 9.6% 32.6% 32.0% 33.0% 30.8% 29.6% 26.8% 27.0%

- 48.8% 71.9% 57.9% 58.4% 71.8% 38.7% 41.6% 63.5% 17.1 12.2 69.8% 79.1% 48.1% 55.3% 29.7% 34.3% 47.0% 58.8% 64.0% 62.6% 22.4% 37.1% 97.0% 94.9% 81.8% 79.2% 42.9% 6.3% 0.4% 30.6% 29.0% 27.3% 23.7% 22.0% 7.9% 0.1%
- 49.9% 73.1% 59.7% 61.3% 74.0% 40.5% 42.9% 63.0% 16.1 11.2 77.2% 83.0% 51.4% 61.3% 29.8% 36.7% 45.8% 59.4% 62.9% 62.0% 21.7% 40.0% 97.0% 95.5% 86.9% 83.5% 63.3% 1.0% 0.7% 31.4% 32.0% 29.8% 24.0% 22.2% 1.0% 0.5%

- 51.8% 72.9% 60.2% 59.2% 72.6% 40.9% 42.1% 63.7% 16.2 10.4 71.3% 84.1% 52.3% 56.5% 32.1% 39.1% 46.4% 59.3% 63.9% 62.5% 24.2% 37.4% 98.1% 97.7% 95.6% 93.3% 89.9% 45.0% 9.0% 34.8% 34.4% 34.5% 31.7% 32.6% 23.4% 6.2%
- 52.0% 72.7% 59.8% 61.8% 73.7% 41.4% 42.3% 65.0% 16.0 10.3 76.2% 82.3% 55.2% 57.0% 32.0% 37.7% 45.6% 57.9% 63.6% 62.1% 23.5% 38.6% 99.6% 99.4% 99.0% 99.1% 97.0% 68.8% 31.6% 33.1% 34.6% 33.1% 28.4% 30.7% 28.7% 25.5%

- 49.4% 72.9% 59.2% 60.5% 73.5% 40.9% 42.6% 62.9% 16.7 11.9 70.7% 81.2% 52.1% 56.2% 29.3% 32.1% 47.0% 58.4% 62.9% 63.0% 22.8% 34.5% 98.4% 97.6% 95.3% 91.1% 57.9% 2.8% 0.3% 29.4% 28.2% 28.4% 24.9% 24.4% 2.7% 0.1%

- 49.9% 73.9% 58.7% 58.4% 72.1% 38.1% 42.6% 64.2% 16.6 11.3 76.4% 83.7% 52.5% 57.4% 33.1% 37.5% 45.4% 58.4% 65.6% 60.6% 24.5% 38.0% 99.6% 99.1% 99.1% 98.6% 96.9% 85.5% 2.1% 34.4% 34.7% 34.2% 29.8% 35.7% 31.4% 12.8%

47.2% 71.9% 55.2% 57.2% 69.7% 38.6% 41.9% 62.8% 18.1 13.5 76.3% 84.5% 53.8% 55.2% 28.4% 32.6% 43.7% 55.5% 62.5% 57.9% 23.3% 37.8% 99.2% 99.0% 98.4% 97.6% 96.8% 75.6% 6.4% 33.8% 33.6% 31.4% 31.3% 33.9% 23.1% 1.2%

- 50.7% 73.0% 59.6% 60.2% 74.0% 41.5% 42.5% 65.1% 16.2 11.2 79.2% 82.7% 52.5% 59.2% 30.7% 36.3% 43.0% 54.8% 65.9% 63.5% 23.3% 38.8% 99.7% 99.1% 95.7% 89.2% 68.4% 4.9% 0.7% 31.1% 32.2% 31.7% 26.7% 28.3% 6.8% 0.7%
- 51.1% 72.5% 59.4% 60.3% 73.2% 40.4% 42.5% 63.9% 16.3 10.9 76.6% 83.0% 57.7% 60.6% 31.2% 37.5% 45.1% 61.4% 64.2% 63.9% 23.3% 39.4% 98.8% 98.0% 97.6% 96.6% 93.4% 81.1% 17.8% 34.4% 33.0% 33.0% 28.7% 32.0% 28.7% 9.8%

- 50.8% 73.2% 59.9% 60.5% 72.4% 41.5% 42.3% 64.6% 16.0 11.0 77.1% 84.7% 54.1% 57.1% 31.9% 37.3% 45.3% 56.4% 66.4% 63.9% 24.4% 38.7% 99.1% 99.1% 97.9% 97.0% 94.5% 69.8% 29.4% 33.1% 34.3% 32.1% 28.9% 29.5% 28.7% 21.7%

- 49.2% 73.2% 59.0% 58.8% 72.3% 39.1% 43.4% 64.6% 16.8 11.8 68.8% 81.5% 52.0% 56.8% 29.4% 31.2% 43.4% 57.2% 64.0% 61.8% 22.1% 36.3% 99.4% 98.6% 96.5% 90.9% 51.1% 5.6% 0.2% 32.0% 31.1% 28.5% 26.0% 26.0% 5.4% 0.2%

48.7% 72.9% 58.8% 59.0% 71.4% 37.9% 42.8% 63.6% 16.6 12.7 75.4% 82.1% 50.4% 56.9% 29.2% 34.0% 41.9% 56.1% 65.6% 62.8% 25.0% 37.6% 98.4% 97.8% 96.4% 91.3% 64.7% 1.6% 0.7% 30.8% 30.5% 28.6% 26.0% 25.5% 3.2% 0.3%

- 50.0% 73.8% 59.0% 59.7% 71.8% 38.7% 42.3% 62.5% 16.8 11.6 70.8% 77.8% 45.3% 52.3% 28.5% 33.9% 44.1% 56.3% 62.1% 61.9% 20.9% 34.5% 98.0% 93.4% 87.0% 81.6% 49.0% 3.3% 0.4% 31.2% 31.2% 33.0% 27.2% 29.3% 3.9% 0.4%

- 47.4% 73.4% 58.6% 59.6% 71.7% 40.4% 42.9% 63.2% 16.6 13.3 73.1% 73.4% 48.9% 54.7% 28.1% 31.5% 44.8% 55.3% 63.4% 61.3% 23.3% 38.2% 97.8% 93.4% 87.6% 80.9% 46.4% 7.5% 0.3% 30.2% 30.6% 30.7% 25.5% 27.4% 11.8% 0.0%
- 48.9% 72.5% 59.1% 59.0% 71.4% 41.1% 42.9% 62.3% 16.9 12.6 74.7% 79.7% 49.2% 54.7% 30.3% 33.7% 47.9% 58.2% 65.0% 61.8% 24.2% 37.4% 99.0% 97.4% 92.3% 84.7% 58.4% 5.4% 0.9% 31.2% 31.2% 30.5% 30.4% 27.7% 8.1% 0.5%

- 50.1% 72.7% 58.9% 60.5% 72.2% 41.0% 42.3% 63.5% 16.8 12.0 74.2% 76.8% 53.6% 52.6% 31.0% 34.1% 43.6% 57.4% 64.5% 63.6% 23.1% 37.3% 97.3% 96.9% 92.8% 88.7% 56.1% 2.7% 0.6% 30.8% 31.1% 28.9% 27.0% 25.9% 4.0% 0.2%

GDN - noconv1d - seed 20

50.4% 73.5% 60.0% 60.9% 73.6% 40.5% 42.3% 64.8% 16.6 10.7 50.0% 62.5% 40.3% 50.8% 26.9% 33.8% 37.8% 60.8% 63.4% 63.9% 22.3% 40.4% 48.5% 36.6% 20.6% 15.1% 9.2% 7.9% 6.2% 33.3% 28.9% 26.5% 25.4% 21.0% 17.1% 13.2%

GDN - Canon-AbCD(res) - seed 20

LMB PIQA Hella Wino ARC-e ARC-c SIQA BoolQWikipplLMBppl FDA FDA2 SWDESWDE2 NQ NQ2 SquadSquad2TriviaQATriviaQA2 Drop Drop2 1-hop-0k1-hop-1k1-hop-2k1-hop-3k1-hop-4k1-hop-5k1-hop-6k 2-hop-0k2-hop-1k2-hop-2k2-hop-3k2-hop-4k2-hop-5k2-hop-6k

Figure 22: This is identical to Figure 16 but additionally includes GPT2(RoPE) models—identical to Llama(RoPE) but using standard MLPs—and GPT2(RoPE,R2), which uses ReLU2 activation [59]. Key conclusions remain unchanged: reducing RoPE improves length generalization, and many architectural differences (e.g., standard vs. gated MLP, SiLU vs. ReLU2) are buried in noise.

###### SlimPajama | 100B token pretrain | 1.3B models

- 55.1% 63.5% 55.9% 31.6% 37.9% 27.1% 23.9% 20.8% 34.5% 30.3% 28.1% 12.8% 60.9% 35.0% 31.0% 25.7% 69.7% 64.0% 60.6% 54.0% 100.0%100.0%100.0% 85.6% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 71.8% 0.0% 0.0% 0.0% 97.4% 91.4% 89.6% 10.6% 0.0% 0.0% 0.0%

52.2% 59.0% 43.2% 32.3% 33.5% 28.9% 23.2% 18.4% 35.1% 32.5% 22.0% 10.1% 58.9% 44.6% 34.3% 26.9% 63.6% 71.0% 63.6% 41.7% 100.0%100.0%100.0% 40.6% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 69.8% 0.0% 0.0% 0.0% 99.6% 99.6% 98.0% 3.4% 0.0% 0.0% 0.0%

59.9% 66.7% 53.6% 33.8% 35.9% 30.7% 25.5% 18.7% 29.8% 29.1% 19.1% 8.8% 57.7% 46.1% 35.7% 24.6% 73.8% 70.4% 65.7% 46.8% 100.0%100.0%100.0% 94.4% 0.0% 0.0% 0.0% 100.0%100.0% 98.6% 78.4% 0.0% 0.0% 0.0% 100.0% 98.8% 93.2% 17.0% 0.0% 0.0% 0.0%

50.9% 54.4% 43.0% 37.7% 30.3% 27.1% 23.1% 16.4% 25.8% 30.7% 24.5% 10.8% 54.0% 39.5% 31.1% 21.6% 57.4% 62.8% 57.6% 47.7% 100.0%100.0% 99.6% 56.8% 0.0% 0.0% 0.0% 100.0%100.0% 99.8% 67.2% 0.0% 0.0% 0.0% 89.0% 99.4% 90.2% 18.0% 0.0% 0.0% 0.0%

58.7% 64.7% 51.2% 36.5% 37.5% 29.8% 23.9% 14.7% 32.9% 30.2% 24.6% 11.4% 57.4% 41.6% 29.4% 22.0% 70.2% 68.8% 63.1% 47.6% 100.0%100.0%100.0% 96.0% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 77.4% 0.0% 0.0% 0.0% 99.6% 97.6% 96.6% 9.0% 0.0% 0.0% 0.0%

- 56.5% 65.1% 56.0% 28.6% 36.7% 32.9% 27.5% 19.1% 30.9% 29.7% 25.3% 15.4% 54.6% 41.5% 32.4% 26.1% 68.3% 73.5% 68.7% 52.9% 100.0%100.0%100.0% 83.4% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 80.0% 0.0% 0.0% 0.0% 99.8% 93.0% 84.8% 39.0% 0.0% 0.0% 0.0%
- 57.0% 61.8% 50.8% 33.0% 38.9% 31.1% 25.1% 17.0% 34.3% 32.8% 26.5% 15.7% 59.9% 44.0% 39.6% 21.3% 62.6% 67.6% 61.6% 53.4% 100.0% 98.8% 99.8% 85.6% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 77.6% 0.0% 0.0% 0.0% 99.6% 97.4% 97.8% 19.2% 0.0% 0.0% 0.0%
- 58.2% 58.5% 47.2% 37.0% 32.7% 25.3% 22.2% 22.0% 27.1% 27.6% 26.3% 9.8% 56.8% 38.3% 39.0% 27.5% 63.8% 66.7% 56.4% 52.6% 100.0%100.0%100.0% 93.8% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 83.0% 0.0% 0.0% 0.0% 99.4% 99.8% 95.2% 7.4% 0.0% 0.0% 0.0%

- Llama(RoPE) - original - seed 20

- Llama(RoPE) - original - seed 21

- Llama(RoPE) - original - seed 22

- Llama(RoPE) - original - seed 23

- Llama(RoPE) - original - seed 24

- Llama(RoPE) - original - seed 25

- Llama(RoPE) - original - seed 26

- Llama(RoPE) - original - seed 27

[Figure 218]

- 59.4% 70.9% 57.8% 35.9% 38.2% 30.1% 25.6% 20.6% 31.7% 31.6% 27.2% 10.8% 59.4% 34.8% 31.9% 23.7% 68.4% 64.9% 63.6% 52.1% 100.0%100.0%100.0% 94.4% 4.2% 2.2% 0.0% 100.0%100.0%100.0% 54.2% 3.0% 0.0% 0.0% 99.6% 99.8% 88.8% 6.8% 0.0% 0.0% 0.0%

62.1% 71.5% 65.0% 47.3% 36.5% 29.5% 25.6% 24.5% 29.9% 29.5% 27.4% 24.6% 58.7% 42.2% 34.1% 36.5% 63.1% 65.2% 63.9% 56.9% 100.0%100.0%100.0%100.0% 8.8% 14.6% 24.6% 100.0%100.0%100.0%100.0% 3.0% 0.2% 0.0% 99.6% 99.8% 100.0% 92.2% 2.4% 0.0% 0.0%

- 60.0% 70.1% 63.1% 55.8% 39.7% 33.3% 31.1% 29.4% 27.7% 29.1% 24.8% 27.9% 55.8% 42.6% 37.4% 33.3% 67.6% 63.3% 56.7% 60.7% 100.0%100.0%100.0%100.0%100.0% 99.8% 99.6% 100.0%100.0%100.0%100.0%100.0% 95.8% 87.4% 100.0% 99.0% 99.2% 97.0% 93.8% 49.8% 34.6%

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

52.7% 61.6% 58.9% 48.2% 32.4% 30.7% 26.8% 21.0% 26.8% 25.2% 23.9% 20.1% 60.1% 39.9% 32.1% 29.0% 60.1% 64.8% 62.2% 60.2% 100.0%100.0%100.0%100.0% 0.0% 0.0% 0.0% 100.0%100.0%100.0%100.0% 58.0% 0.0% 0.0% 98.4% 99.6% 99.8% 78.4% 2.8% 0.0% 0.0%

Llama(NoPE) - original - seed 20

- 59.2% 71.1% 66.7% 51.2% 42.7% 34.2% 29.4% 28.8% 28.8% 28.5% 28.7% 23.7% 54.3% 43.2% 36.8% 31.9% 58.6% 67.2% 67.6% 61.4% 100.0%100.0%100.0%100.0% 1.0% 7.0% 0.0% 100.0%100.0%100.0%100.0% 0.0% 0.0% 0.0% 91.4% 95.4% 87.4% 81.4% 0.0% 0.0% 0.0%

52.4% 64.7% 58.8% 46.4% 38.4% 33.8% 31.9% 24.6% 29.4% 25.4% 28.2% 23.1% 52.6% 44.4% 42.8% 39.8% 67.4% 70.1% 66.8% 63.2% 100.0%100.0%100.0%100.0%100.0%100.0% 64.0% 100.0%100.0%100.0%100.0%100.0% 89.8% 53.2% 99.8% 97.8% 98.2% 99.4% 91.4% 50.6% 8.4%

50.9% 61.6% 50.5% 33.9% 34.4% 31.4% 27.4% 20.3% 26.3% 28.5% 25.9% 12.3% 60.9% 36.9% 25.8% 27.4% 59.8% 64.8% 61.6% 42.6% 100.0%100.0%100.0% 95.0% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 61.0% 0.0% 0.0% 0.0% 96.4% 99.2% 95.0% 2.2% 0.0% 0.0% 0.0%

- 60.9% 63.6% 56.1% 48.9% 36.9% 33.3% 27.8% 26.1% 24.0% 24.0% 24.5% 22.4% 54.0% 40.0% 35.0% 33.5% 53.1% 57.4% 56.2% 52.9% 100.0%100.0%100.0%100.0%100.0%100.0% 38.2% 100.0%100.0%100.0%100.0% 44.0% 0.0% 0.0% 98.0% 96.0% 97.8% 93.6% 15.6% 0.0% 0.0%

Llama(NoPE) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - original - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

50.2% 62.4% 42.9% 30.3% 30.9% 28.5% 22.2% 16.8% 33.5% 32.0% 27.3% 6.4% 56.9% 46.4% 34.8% 22.1% 66.3% 69.1% 61.5% 28.7% 100.0%100.0%100.0% 83.0% 1.8% 0.0% 0.0% 100.0%100.0%100.0% 3.2% 0.0% 0.0% 0.0% 98.6% 99.2% 93.6% 1.0% 0.0% 0.0% 0.0%

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

53.8% 62.6% 49.9% 33.9% 35.2% 31.0% 23.7% 18.8% 29.5% 29.1% 24.0% 13.6% 61.7% 37.3% 30.2% 21.1% 63.6% 61.9% 55.1% 40.3% 100.0%100.0%100.0% 83.8% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 40.8% 0.0% 0.0% 0.0% 99.8% 98.8% 98.0% 0.0% 0.0% 0.0% 0.0%

GPT2(RoPE) - original - seed 20

59.2% 72.4% 63.6% 47.7% 37.0% 30.2% 28.8% 26.4% 30.3% 30.0% 27.4% 22.2% 57.9% 40.8% 34.7% 31.8% 67.3% 71.2% 68.5% 60.8% 100.0%100.0%100.0%100.0% 11.8% 13.4% 2.0% 100.0%100.0%100.0%100.0% 98.2% 0.2% 1.2% 100.0% 99.6% 97.4% 93.8% 0.4% 0.0% 0.0%

GPT2(RoPE) - Canon-ABCD(res) - seed 20

55.4% 63.5% 58.2% 46.4% 31.0% 29.4% 27.7% 26.0% 29.3% 27.7% 25.1% 23.0% 54.1% 40.0% 36.9% 30.2% 69.0% 68.8% 66.4% 61.9% 100.0%100.0%100.0%100.0% 0.0% 0.0% 0.0% 100.0%100.0%100.0%100.0% 0.0% 0.0% 0.0% 100.0% 99.8% 99.6% 92.2% 0.0% 0.0% 0.0%

GPT2(RoPE) - Canon-ABCD(res) - seed 20

Mamba2(mlp) - original(conv1d) - seed 20

Mamba2(mlp) - noconv1d - seed 20

Mamba2(mlp) - Canon-AbCD(res) - seed 20

Mamba2(mlp) - Canon-ABCD(res) - seed 20

Mamba2 - original(conv1d) - seed 20

Mamba2 - noconv1d - seed 20

Mamba2 - Canon-Ab(res) - seed 20

Mamba2 - Canon-AB(res) - seed 20

GLA - original(noconv1d) - seed 20

GLA - conv1d - seed 20

- 46.2% 42.8% 33.9% 25.8% 35.2% 23.9% 19.4% 15.7% 30.5% 29.0% 27.0% 19.9% 56.7% 40.3% 31.5% 24.9% 56.3% 56.0% 54.4% 46.3% 100.0%100.0% 99.4% 99.4% 93.0% 91.8% 83.2% 100.0%100.0% 99.4% 97.0% 86.4% 71.6% 62.8% 99.4% 98.0% 87.0% 59.2% 47.0% 38.6% 31.4%

- 43.7% 35.1% 25.6% 24.1% 32.5% 23.8% 19.8% 16.9% 27.1% 25.1% 21.8% 11.6% 54.2% 32.5% 25.3% 21.9% 50.7% 53.2% 52.5% 44.6% 100.0%100.0% 82.6% 73.4% 48.8% 39.0% 30.0% 100.0%100.0% 99.6% 88.8% 81.4% 59.8% 58.6% 99.8% 92.6% 58.2% 39.8% 25.0% 21.6% 20.2%

- 49.0% 29.4% 26.7% 21.3% 32.5% 19.4% 15.6% 16.0% 22.2% 20.6% 8.3% 6.8% 49.5% 21.7% 21.7% 16.2% 48.0% 42.3% 28.3% 21.5% 100.0% 99.8% 72.8% 54.8% 27.2% 21.2% 13.2% 100.0% 99.6% 91.2% 74.6% 67.0% 42.0% 35.4% 99.0% 94.8% 37.6% 28.8% 20.2% 18.0% 12.8%

- 45.4% 41.6% 30.1% 28.4% 26.8% 23.6% 20.5% 16.9% 23.4% 25.1% 13.4% 10.2% 51.3% 34.8% 27.7% 17.0% 57.2% 54.0% 50.6% 40.6% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 98.4% 95.2% 55.2% 37.4% 26.6% 99.8% 95.8% 70.8% 43.4% 19.4% 18.6% 11.2%

47.6% 41.5% 34.0% 24.8% 34.1% 26.9% 20.5% 14.9% 23.7% 21.9% 17.8% 11.9% 54.6% 42.3% 31.9% 22.2% 59.4% 58.6% 53.5% 43.5% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 99.0% 90.4% 66.6% 41.2% 36.4% 99.8% 95.8% 59.6% 41.6% 23.4% 16.8% 15.6%

44.6% 40.3% 23.2% 16.3% 24.1% 17.7% 14.1% 11.2% 21.6% 21.6% 6.3% 7.4% 48.0% 27.2% 22.3% 10.4% 43.8% 44.0% 42.0% 33.9% 100.0% 99.8% 100.0%100.0% 99.6% 99.8% 98.8% 100.0%100.0% 85.8% 50.8% 19.8% 17.2% 18.6% 99.6% 87.2% 51.4% 31.4% 13.4% 18.6% 12.2%

47.7% 39.7% 28.6% 23.0% 35.0% 22.4% 18.4% 13.6% 23.5% 21.8% 18.0% 12.8% 48.5% 35.9% 28.5% 18.2% 57.5% 55.5% 53.9% 39.5% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 99.0% 88.2% 68.8% 49.2% 41.4% 100.0% 98.6% 72.2% 39.6% 20.6% 19.6% 14.4%

- 45.2% 41.7% 26.4% 22.6% 35.5% 22.9% 17.1% 17.2% 26.0% 20.8% 15.1% 12.8% 59.6% 33.2% 25.1% 20.5% 56.7% 52.7% 53.9% 37.7% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 93.8% 79.2% 51.4% 53.6% 29.8% 99.6% 98.8% 60.2% 35.6% 24.2% 20.0% 11.2%
- 46.0% 39.4% 35.5% 28.8% 32.6% 22.3% 20.6% 19.8% 24.7% 22.9% 14.4% 14.0% 54.4% 35.8% 31.0% 23.4% 51.9% 54.1% 47.5% 35.5% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 99.4% 96.4% 66.0% 53.0% 55.6% 100.0% 90.2% 69.0% 42.6% 16.2% 17.8% 16.0%

- 46.5% 41.6% 32.2% 18.6% 30.3% 20.0% 19.1% 13.7% 27.9% 15.4% 8.1% 7.3% 43.8% 24.9% 21.8% 13.5% 43.5% 53.2% 45.8% 38.0% 100.0%100.0%100.0% 99.8% 100.0% 99.8% 99.6% 100.0% 99.8% 90.0% 68.0% 21.2% 29.0% 25.4% 99.2% 92.8% 36.8% 23.6% 6.4% 9.6% 6.4%

- 49.1% 36.9% 27.7% 19.3% 33.1% 25.4% 16.7% 12.1% 24.9% 16.0% 14.7% 11.8% 55.6% 31.4% 25.5% 18.1% 53.9% 55.0% 52.7% 37.1% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 98.0% 86.6% 25.8% 31.2% 27.0% 99.8% 98.6% 65.4% 50.4% 30.6% 23.8% 15.6%

GLA - Canon-AbCD(res) - seed 20

55.4% 46.5% 37.6% 26.2% 32.8% 25.5% 21.5% 18.1% 26.3% 25.9% 27.1% 18.9% 60.8% 39.8% 30.2% 22.3% 62.6% 59.3% 54.3% 42.0% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 99.6% 95.6% 79.8% 59.8% 54.2% 96.0% 95.0% 68.8% 46.4% 25.0% 24.6% 23.8%

GDN - original(conv1d) - seed 20

43.8% 43.6% 33.7% 24.9% 35.6% 27.5% 21.9% 17.8% 28.2% 25.5% 22.5% 13.1% 62.4% 38.8% 29.6% 22.5% 50.4% 56.3% 54.0% 42.7% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 97.8% 95.0% 69.2% 52.2% 44.6% 100.0% 97.6% 71.6% 50.0% 17.8% 22.2% 18.8%

GDN - noconv1d - seed 20

47.1% 41.7% 39.3% 29.0% 38.2% 25.8% 23.4% 19.6% 27.2% 27.5% 26.1% 18.5% 59.7% 38.4% 31.7% 24.8% 63.1% 56.9% 48.4% 44.4% 100.0%100.0%100.0%100.0%100.0%100.0%100.0% 100.0%100.0% 99.2% 95.4% 87.8% 47.6% 41.0% 99.8% 98.8% 76.4% 68.6% 53.2% 39.4% 25.2%

GDN - Canon-AbCD(res) - seed 20

qa1-0kqa1-1kqa1-2kqa1-4k qa2-0kqa2-1kqa2-2kqa2-4k qa3-0kqa3-1kqa3-2kqa3-4k qa4-0kqa4-1kqa4-2kqa4-4k qa5-0kqa5-1kqa5-2kqa5-4k niah-S1-0.5kniah-S1-2kniah-S1-4kniah-S1-5kniah-S1-6kniah-S1-7kniah-S1-8k niah-S2-0.5kniah-S2-2kniah-S2-4kniah-S2-5kniah-S2-6kniah-S2-7kniah-S2-8k niah-S3-0.5kniah-S3-2kniah-S3-4kniah-S3-5kniah-S3-6kniah-S3-7kniah-S3-8k

FineWeb-Edu | 100B token pretrain | 1.3B models

50.7% 50.8% 39.4% 32.5% 33.0% 28.6% 20.4% 13.9% 27.7% 26.9% 2.4% 8.7% 52.2% 40.5% 28.2% 19.2% 64.2% 63.1% 55.3% 50.3% 100.0% 99.8% 64.2% 2.2% 0.0% 0.0% 0.0% 100.0%100.0% 95.8% 43.8% 0.0% 0.0% 0.0% 97.0% 98.8% 70.8% 6.2% 0.0% 0.0% 0.0%

- Llama(RoPE) - original - seed 20

- Llama(RoPE) - original - seed 21

- Llama(RoPE) - original - seed 22

- Llama(RoPE) - original - seed 23

- Llama(RoPE) - original - seed 24

- Llama(RoPE) - original - seed 25

- Llama(RoPE) - original - seed 26

- Llama(RoPE) - original - seed 27

[Figure 219]

53.3% 52.7% 38.8% 31.9% 35.0% 28.3% 17.6% 14.7% 33.0% 31.8% 13.3% 12.3% 50.1% 41.3% 30.6% 22.7% 62.0% 60.5% 53.7% 39.4% 100.0% 95.6% 93.4% 40.6% 0.0% 0.0% 0.0% 100.0%100.0%100.0% 62.2% 0.0% 0.0% 0.0% 97.0% 98.0% 79.0% 8.6% 0.0% 0.0% 0.0%

47.0% 56.5% 43.5% 34.8% 32.0% 22.0% 18.1% 15.3% 23.5% 22.2% 14.9% 11.6% 57.8% 33.2% 26.8% 21.9% 55.0% 62.1% 54.7% 43.9% 100.0%100.0% 95.4% 4.6% 0.0% 0.0% 0.0% 100.0%100.0% 99.8% 44.2% 0.0% 0.0% 0.0% 99.0% 99.0% 82.8% 9.2% 0.0% 0.0% 0.0%

45.7% 61.5% 38.8% 33.1% 34.0% 31.5% 22.8% 12.4% 27.9% 29.5% 4.8% 13.0% 56.8% 41.9% 31.5% 19.3% 59.8% 61.2% 49.8% 37.5% 100.0% 97.0% 95.6% 0.8% 0.0% 0.0% 0.0% 100.0%100.0% 99.4% 62.8% 0.2% 0.0% 0.0% 99.6% 99.2% 92.4% 4.8% 0.0% 0.0% 0.0%

53.4% 50.5% 41.3% 40.1% 35.5% 28.0% 21.4% 15.8% 33.9% 28.0% 12.1% 13.0% 57.2% 37.8% 29.6% 23.1% 60.7% 64.7% 57.4% 36.2% 100.0% 92.4% 92.4% 22.0% 0.0% 0.0% 0.0% 100.0%100.0% 99.2% 74.2% 0.0% 0.0% 0.0% 97.8% 88.2% 62.2% 25.4% 0.0% 0.0% 0.0%

47.9% 44.7% 34.0% 35.5% 30.8% 24.8% 18.9% 13.4% 22.6% 28.0% 6.5% 12.2% 53.6% 37.5% 30.6% 18.6% 60.1% 55.7% 48.2% 38.0% 100.0% 99.0% 89.6% 1.0% 0.0% 0.0% 0.0% 100.0%100.0% 99.4% 48.4% 0.0% 0.0% 0.0% 89.0% 96.2% 80.8% 0.2% 0.0% 0.0% 0.0%

- 43.3% 41.2% 37.6% 28.0% 27.4% 22.3% 19.3% 18.3% 24.1% 24.7% 18.3% 12.2% 54.3% 40.4% 35.4% 26.0% 53.2% 52.6% 46.4% 35.6% 100.0%100.0%100.0%100.0% 99.8% 100.0%100.0% 100.0%100.0% 94.4% 88.6% 59.8% 50.8% 38.6% 99.2% 82.8% 44.4% 29.2% 15.0% 19.6% 20.0%

39.4% 42.4% 36.0% 30.9% 24.0% 23.1% 21.7% 18.2% 26.0% 18.8% 17.0% 16.9% 49.5% 42.3% 35.6% 26.9% 51.8% 55.5% 50.4% 38.6% 100.0%100.0% 99.2% 99.0% 95.8% 94.2% 94.0% 100.0% 99.8% 86.2% 75.6% 32.4% 41.0% 36.6% 97.0% 86.0% 33.6% 23.0% 10.0% 14.6% 4.8%

41.5% 37.1% 32.3% 27.3% 26.6% 21.7% 20.2% 16.9% 24.5% 23.1% 14.4% 16.1% 48.4% 37.1% 33.5% 27.0% 48.5% 50.6% 41.3% 27.0% 100.0%100.0%100.0%100.0% 99.8% 100.0% 99.6% 100.0%100.0% 95.8% 83.8% 60.4% 49.8% 51.2% 98.6% 64.0% 33.2% 15.0% 13.2% 12.2% 10.4%

- 44.5% 34.5% 29.1% 23.6% 28.9% 20.2% 17.3% 12.4% 23.9% 13.7% 10.1% 12.0% 55.6% 37.5% 33.7% 26.1% 45.1% 47.1% 40.0% 31.1% 100.0% 70.6% 24.8% 22.2% 14.8% 12.6% 10.8% 100.0%100.0% 91.6% 77.8% 69.2% 56.0% 46.8% 97.6% 87.6% 32.2% 23.0% 18.8% 21.4% 16.0%
- 45.7% 40.9% 30.8% 26.1% 22.9% 21.0% 18.8% 16.5% 25.3% 24.0% 12.3% 11.4% 53.7% 36.8% 33.5% 25.7% 45.8% 49.7% 46.9% 40.3% 100.0% 70.6% 25.6% 19.2% 12.6% 10.2% 9.0% 100.0% 99.6% 83.2% 63.6% 47.4% 39.2% 34.4% 94.6% 73.2% 30.6% 21.6% 13.4% 14.0% 6.8%

- 39.7% 37.1% 27.8% 22.7% 23.4% 20.1% 18.8% 15.8% 11.7% 9.2% 13.8% 13.6% 38.9% 35.3% 31.3% 14.6% 47.3% 51.4% 43.5% 26.6% 99.6% 49.0% 21.0% 14.2% 10.6% 9.4% 7.0% 100.0% 96.2% 51.4% 31.4% 17.2% 17.8% 16.6% 79.2% 26.6% 8.2% 3.4% 2.8% 1.2% 1.0%
- 40.6% 33.7% 29.9% 24.3% 27.9% 20.1% 16.7% 12.9% 23.2% 14.0% 13.8% 12.1% 49.0% 34.1% 31.3% 17.2% 49.2% 51.3% 41.4% 22.1% 100.0%100.0%100.0%100.0% 95.2% 83.2% 63.2% 100.0%100.0% 86.0% 75.8% 44.2% 37.4% 29.6% 94.2% 60.2% 30.8% 21.0% 11.8% 12.4% 9.2%

39.3% 38.2% 30.7% 22.2% 27.7% 23.0% 18.1% 13.1% 25.4% 11.4% 13.1% 12.4% 46.6% 33.4% 29.2% 17.7% 56.2% 54.2% 45.5% 28.1% 100.0%100.0%100.0%100.0% 98.8% 96.2% 91.8% 100.0% 99.4% 69.8% 61.0% 28.4% 37.0% 31.4% 93.4% 68.6% 18.8% 18.0% 8.4% 11.8% 9.4%

32.1% 33.9% 26.6% 21.9% 20.7% 20.7% 19.1% 10.9% 20.4% 8.5% 11.9% 12.2% 43.4% 33.8% 24.9% 11.3% 48.8% 46.6% 41.4% 25.5% 100.0% 98.8% 95.4% 95.2% 95.2% 91.0% 90.0% 100.0% 98.6% 57.2% 52.8% 36.6% 32.6% 28.8% 68.2% 28.8% 14.8% 5.8% 2.8% 1.4% 2.8%

- 41.3% 40.6% 33.6% 26.8% 31.9% 25.8% 19.5% 13.3% 22.4% 13.8% 17.6% 17.3% 46.5% 34.0% 27.0% 17.5% 53.4% 52.5% 44.7% 28.0% 100.0%100.0% 98.8% 97.6% 93.8% 88.2% 67.6% 100.0% 99.8% 80.2% 63.8% 43.8% 45.0% 36.0% 88.2% 68.0% 27.2% 14.0% 7.6% 9.0% 6.6%

- 46.5% 36.5% 30.1% 25.3% 28.3% 22.4% 21.1% 15.8% 21.4% 19.6% 13.5% 12.3% 50.4% 34.3% 27.5% 18.3% 55.7% 49.6% 40.4% 29.3% 100.0%100.0% 99.8% 99.8% 99.8% 99.8% 100.0% 100.0% 99.8% 71.8% 67.8% 41.2% 44.4% 41.6% 94.4% 53.6% 25.2% 14.4% 9.2% 11.2% 10.2%

36.5% 38.8% 30.5% 22.0% 24.9% 20.5% 18.3% 13.3% 19.0% 16.7% 12.1% 12.1% 42.7% 32.0% 27.6% 16.1% 50.2% 49.9% 40.6% 27.7% 100.0%100.0% 99.8% 98.0% 86.0% 77.8% 52.0% 100.0% 99.8% 75.0% 63.8% 14.8% 30.4% 26.0% 92.2% 64.2% 17.8% 15.2% 5.6% 9.6% 9.2%

43.4% 39.2% 28.4% 25.6% 30.7% 23.3% 19.6% 17.0% 18.6% 16.9% 17.3% 14.3% 44.0% 31.8% 24.1% 15.3% 44.0% 49.3% 42.4% 31.2% 100.0% 99.8% 93.8% 90.4% 64.8% 54.8% 39.8% 100.0% 94.4% 76.2% 57.6% 13.0% 33.0% 29.0% 73.8% 25.6% 8.6% 2.8% 2.4% 1.6% 1.2%

39.7% 37.9% 31.2% 24.8% 23.1% 19.1% 16.9% 13.0% 20.2% 16.4% 13.7% 16.0% 52.3% 34.6% 28.4% 19.1% 56.4% 54.2% 47.3% 30.9% 100.0%100.0% 99.2% 96.4% 84.8% 73.8% 63.8% 100.0% 99.8% 69.2% 63.6% 39.2% 46.2% 36.8% 91.6% 46.0% 21.2% 15.8% 9.0% 7.8% 8.0%

50.4% 55.9% 45.2% 35.3% 29.4% 24.7% 23.0% 21.2% 28.6% 28.2% 25.3% 20.4% 53.3% 35.3% 30.6% 28.5% 60.9% 58.2% 52.2% 52.4% 100.0%100.0%100.0%100.0% 4.2% 7.4% 0.0% 100.0%100.0%100.0%100.0% 1.2% 0.0% 0.0% 87.0% 96.4% 68.2% 32.6% 0.0% 0.0% 0.0%

- 52.4% 62.6% 45.9% 36.9% 31.6% 23.6% 22.0% 21.1% 27.5% 24.4% 23.9% 21.8% 56.4% 38.2% 34.3% 36.4% 57.2% 57.3% 47.8% 46.0% 100.0%100.0%100.0% 99.4% 18.2% 38.8% 6.8% 100.0%100.0%100.0%100.0% 97.2% 2.0% 0.0% 99.2% 93.8% 90.2% 77.6% 8.4% 0.8% 0.0%

48.2% 54.1% 40.9% 37.2% 36.4% 26.6% 18.2% 14.0% 26.8% 26.8% 22.0% 13.0% 53.0% 36.9% 32.0% 19.6% 55.5% 55.1% 47.6% 45.5% 100.0%100.0% 99.4% 40.6% 0.0% 0.0% 0.0% 100.0%100.0% 99.8% 4.0% 0.0% 0.0% 0.0% 99.2% 94.8% 76.4% 0.0% 0.0% 0.0% 0.0%

- 46.8% 52.1% 41.0% 22.5% 31.5% 25.4% 20.5% 10.5% 26.6% 23.9% 14.6% 11.6% 54.7% 36.3% 34.4% 19.5% 55.5% 69.1% 59.7% 29.8% 100.0%100.0% 99.6% 53.4% 0.0% 0.0% 0.0% 100.0%100.0% 99.8% 2.8% 0.0% 0.0% 0.0% 99.2% 99.2% 66.6% 0.2% 0.0% 0.0% 0.0%

56.1% 65.1% 53.2% 44.7% 37.6% 32.0% 31.2% 26.9% 27.8% 32.4% 26.9% 26.2% 48.1% 32.4% 31.7% 30.3% 61.2% 55.2% 54.3% 50.6% 100.0%100.0%100.0%100.0% 6.0% 6.8% 0.0% 100.0%100.0%100.0% 98.6% 1.2% 0.0% 0.0% 97.0% 86.6% 93.8% 69.4% 0.0% 0.0% 0.0%

50.8% 62.6% 56.3% 44.2% 34.6% 32.3% 25.5% 26.2% 28.1% 29.6% 27.3% 22.2% 54.8% 40.6% 30.5% 30.2% 60.7% 66.7% 65.0% 53.9% 100.0%100.0%100.0%100.0% 99.4% 98.4% 94.0% 100.0%100.0%100.0%100.0% 94.0% 62.8% 37.0% 97.4% 92.8% 95.4% 74.6% 43.6% 6.8% 0.0%

- 54.2% 54.6% 33.7% 34.9% 32.4% 26.9% 21.9% 17.6% 27.6% 28.1% 4.3% 16.8% 55.5% 36.3% 30.9% 21.4% 59.5% 61.9% 55.6% 35.8% 100.0%100.0% 94.8% 2.2% 0.0% 0.0% 0.0% 100.0%100.0% 98.6% 3.4% 0.0% 0.0% 0.0% 97.6% 90.8% 67.8% 0.0% 0.0% 0.0% 0.0%
- 55.5% 64.4% 61.5% 53.3% 34.1% 32.5% 27.0% 21.5% 25.0% 27.5% 24.2% 18.7% 50.2% 39.5% 32.8% 30.6% 56.2% 64.1% 61.0% 57.1% 100.0%100.0%100.0%100.0% 92.8% 68.4% 0.0% 100.0%100.0%100.0%100.0% 31.6% 0.0% 0.0% 98.0% 96.6% 94.4% 77.4% 0.0% 0.0% 0.0%

48.7% 59.5% 52.3% 41.6% 31.2% 27.8% 25.3% 21.2% 24.7% 24.0% 24.7% 21.5% 52.6% 34.7% 32.1% 31.2% 50.7% 56.4% 54.5% 55.0% 100.0%100.0% 99.8% 99.6% 96.0% 90.6% 47.2% 100.0%100.0%100.0% 99.4% 1.8% 0.0% 0.0% 98.4% 97.6% 96.4% 26.6% 0.4% 0.0% 0.0%

- 53.6% 59.7% 45.1% 37.5% 35.4% 25.1% 22.0% 17.3% 27.0% 28.0% 10.1% 7.9% 57.1% 37.1% 29.2% 24.8% 62.0% 65.5% 56.1% 44.5% 100.0%100.0% 99.6% 93.8% 0.2% 0.0% 0.0% 100.0%100.0%100.0% 65.6% 0.0% 0.0% 0.0% 99.4% 98.0% 72.2% 6.2% 0.0% 0.0% 0.0%
- 54.4% 56.8% 49.9% 42.6% 35.7% 29.6% 27.6% 23.2% 30.0% 29.1% 26.5% 22.1% 46.4% 38.0% 33.6% 31.4% 72.0% 65.7% 57.6% 59.4% 100.0%100.0%100.0%100.0% 10.8% 14.6% 11.2% 100.0%100.0%100.0%100.0% 1.8% 0.0% 0.0% 98.2% 96.2% 94.2% 74.0% 0.6% 0.0% 0.0%

60.3% 60.6% 52.9% 45.6% 32.1% 26.1% 24.3% 21.5% 21.2% 24.0% 22.3% 19.6% 45.3% 40.5% 33.1% 40.2% 57.7% 56.9% 52.5% 56.2% 100.0% 99.8% 100.0% 99.8% 99.4% 97.8% 82.4% 100.0%100.0%100.0%100.0% 92.8% 68.0% 49.2% 93.4% 94.0% 93.0% 90.2% 74.2% 29.8% 9.4%

- 47.5% 54.9% 40.9% 35.2% 28.5% 26.2% 23.3% 14.8% 27.7% 24.6% 18.4% 11.3% 54.4% 39.7% 31.9% 22.8% 55.0% 57.8% 56.4% 43.3% 100.0% 99.4% 97.8% 16.8% 0.0% 0.0% 0.0% 100.0%100.0% 99.8% 66.4% 0.0% 0.0% 0.0% 100.0% 99.0% 77.2% 7.4% 0.0% 0.0% 0.0%

- 47.8% 51.2% 33.6% 38.3% 30.5% 25.4% 19.2% 16.2% 25.6% 25.6% 18.8% 14.1% 51.6% 34.5% 27.9% 25.2% 60.2% 66.9% 54.1% 43.5% 100.0%100.0% 99.0% 4.6% 0.0% 0.0% 0.0% 100.0%100.0% 99.6% 37.4% 0.0% 0.0% 0.0% 95.4% 95.8% 83.4% 1.6% 0.0% 0.0% 0.0%

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(RoPE) - Canon-ABCD(res) - seed 20

Llama(NoPE) - original - seed 20

Llama(NoPE) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - original - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE,R2) - Canon-ABCD(res) - seed 20

GPT2(RoPE) - original - seed 20

GPT2(RoPE) - Canon-ABCD(res) - seed 20

GPT2(RoPE) - Canon-ABCD(res) - seed 20

Mamba2(mlp) - original(conv1d) - seed 20

Mamba2(mlp) - noconv1d - seed 20

Mamba2(mlp) - Canon-AbCD(res) - seed 20

Mamba2(mlp) - Canon-ABCD(res) - seed 20

Mamba2 - original(conv1d) - seed 20

Mamba2 - noconv1d - seed 20

Mamba2 - Canon-Ab(res) - seed 20

Mamba2 - Canon-AB(res) - seed 20

GLA - original(noconv1d) - seed 20

GLA - conv1d - seed 20

GLA - Canon-AbCD(res) - seed 20

GDN - original(conv1d) - seed 20

GDN - noconv1d - seed 20

GDN - Canon-AbCD(res) - seed 20

qa1-0kqa1-1kqa1-2kqa1-4k qa2-0kqa2-1kqa2-2kqa2-4k qa3-0kqa3-1kqa3-2kqa3-4k qa4-0kqa4-1kqa4-2kqa4-4k qa5-0kqa5-1kqa5-2kqa5-4k niah-S1-0.5kniah-S1-2kniah-S1-4kniah-S1-5kniah-S1-6kniah-S1-7kniah-S1-8k niah-S2-0.5kniah-S2-2kniah-S2-4kniah-S2-5kniah-S2-6kniah-S2-7kniah-S2-8k niah-S3-0.5kniah-S3-2kniah-S3-4kniah-S3-5kniah-S3-6kniah-S3-7kniah-S3-8k

- Figure 23: Results on the Babilong + S-NIAH dataset evaluating multi-hop reasoning across varied junk context lengths. Most architectural comparisons are statistically insignificant. Key findings include:

- 1. Linear models consistently underperform Transformers, even in short contexts without junk.
- 2. Models with reduced RoPE (NoPE, RoPEˇ “) achieve notable improvements in long-context accuracy.
- 3. S-NIAH is too easy: linear models appear accurate but fail at short-context 1-hop retrieval (Figure 22).

#### F More Synthetic Experiments

We present missing figures that were intentionally omitted from the main body of the paper for the sake of clarity and conciseness.

###### Llama(RoPE) - original

###### Llama(RoPE) - Canon-ABCD(res)

###### Llama(RoPE) - cst-Canon-ABCD(res)

###### GPT2(RoPE) - original

###### GPT2(RoPE) - Canon-ABCD(res)

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

0/34% 1/50% 1/4% 0/1% 0/27% 0/0% 0/12% 0/0%

99/100% 97/100% 99/100% 100/100% 98/100% 92/99% 95/100% 95/100%

30/77% 8/94% 61/98% 98/100% 56/92% 28/82% 0/84% 3/75%

0/1% 19/89% 0/14% 10/87% 0/1% 1/26% 0/3% 0/0% 0/0% 0/22% 0/0% 0/0%

98/100% 98/100% 99/100% 97/99% 100/100% 95/100% 97/100% 93/100%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/2% 0/56% 0/0% 0/0%

75/99% 97/100% 85/100% 90/100%

2/49% 11/51% 0/77% 0/26%

61/100% 97/100% 99/100% 97/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - original

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16, k=16/8) Llama(RoPE) - cst-Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) GPT2(RoPE) - original

Task Depo2(K=16, k=16/8) GPT2(RoPE) - Canon-ABCD(res)

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

2/1% 2/1% 1/1% 30/99%

92/100% 100/100% 97/100% 99/100% 97/100% 99/100% 96/100% 97/100% 85/100% 99/100% 98/100% 98/100%

67/99% 99/100% 99/100% 99/100% 86/100% 97/100% 84/99% 96/100% 89/100% 96/100% 96/100% 96/99%

1/1% 1/2% 1/7% 2/2% 1/46% 1/22% 1/1% 1/1%

97/100% 97/100% 99/100% 98/100% 86/99% 96/100% 98/100% 98/100% 84/99% 96/100% 91/100% 95/100%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/90% 1/3% 21/96%
- 1/2% 1/92% 1/3% 1/50%

1/1% 1/1% 1/1% 1/1%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Llama(RoPE) - original

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

###### Task Brevo1 Llama(RoPE) - cst-Canon-ABCD(res)

###### Task Brevo1 GPT2(RoPE) - original

Task Brevo1 GPT2(RoPE) - Canon-ABCD(res)

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

45.6% 76.9% 79.8% 88.5% 32.6% 64.5% 44.5% 63.1%

83.7% 93.8% 91.3% 96.5% 62.9% 84.5% 81.2% 90.7% 47.9% 82.2% 69.7% 84.5%

75.4% 86.0% 83.4% 91.7% 36.0% 64.7% 57.9% 79.4% 21.1% 44.5% 34.7% 65.1%

59.3% 82.8% 64.2% 85.8% 6.9% 50.4% 26.0% 67.5% 5.8% 25.0% 9.7% 37.3%

84.9% 91.2% 88.6% 93.3% 66.3% 78.5% 76.8% 82.5% 34.1% 51.1% 53.4% 70.4%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

8.0% 31.2% 17.7% 27.5%

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 Llama(RoPE) - original

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

Task Brevo2 Llama(RoPE) - cst-Canon-ABCD(res)

Task Brevo2 GPT2(RoPE) - original

Task Brevo2 GPT2(RoPE) - Canon-ABCD(res)

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

69.3% 89.8% 83.7% 96.0% 40.3% 79.5% 60.5% 88.0% 22.4% 68.2% 40.2% 81.4%

87.1% 95.6% 92.2% 97.1% 75.4% 87.7% 80.1% 93.5% 55.1% 82.5% 69.3% 88.1%

82.3% 91.3% 87.4% 95.1% 58.7% 81.8% 69.3% 91.1% 40.1% 68.3% 57.5% 79.8%

70.9% 85.0% 80.3% 93.7% 44.8% 66.3% 63.4% 81.4% 24.7% 55.9% 43.4% 76.2%

84.2% 94.2% 89.8% 95.9% 62.0% 86.0% 81.2% 91.4% 49.2% 75.5% 63.2% 84.4%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Mano Llama(RoPE) - original

Task Mano Llama(RoPE) - Canon-ABCD(res)

Task Mano Llama(RoPE) - cst-Canon-ABCD(res)

###### Task Mano GPT2(RoPE) - original

Task Mano GPT2(RoPE) - Canon-ABCD(res)

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

59.4% 75.5% 84.5% 85.2% 55.6% 53.8% 52.5% 46.5% 26.3% 19.7% 20.9% 41.6%

94.2% 98.0% 99.2% 99.6% 89.8% 88.5% 98.2% 99.2% 83.7% 83.6% 88.8% 85.3%

97.4% 99.2% 99.5% 99.8% 92.4% 94.0% 96.9% 98.9% 94.5% 93.2% 94.1% 98.1%

33.1% 22.3% 49.2% 40.7% 20.9% 23.1% 21.8% 12.2% 11.8% 7.4% 20.5% 11.4%

81.4% 87.1% 92.8% 98.0% 76.9% 86.7% 96.2% 92.1% 15.7% 44.5% 69.2% 72.5%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - original

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - cst-Canon-ABCD(res)

Task Lano GPT2(RoPE) - original

Task Lano GPT2(RoPE) - Canon-ABCD(res)

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

91.1% 96.3% 93.4% 97.6% 74.1% 91.4% 82.3% 90.3% 64.0% 75.1% 60.0% 79.1%

95.2% 97.5% 96.0% 98.1% 81.4% 90.1% 85.9% 92.6% 66.0% 77.9% 76.1% 78.9%

95.0% 96.4% 96.5% 98.1% 79.6% 89.7% 85.6% 92.1% 67.4% 79.4% 77.1% 84.4%

78.1% 93.0% 86.5% 94.8% 56.6% 82.4% 71.1% 82.9% 46.1% 69.6% 47.3% 71.8%

91.1% 95.0% 94.8% 96.4% 72.5% 87.2% 78.2% 90.7% 64.3% 74.9% 68.5% 73.9%

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - original

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - cst-Canon-ABCD(res)

###### Task Lano GPT2(RoPE) - original

Task Lano GPT2(RoPE) - Canon-ABCD(res)

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

0.00095 0.00050 0.00073 0.00038 0.00135 0.00056 0.00095 0.00057 0.00232 0.00156 0.00235 0.00131

0.00067 0.00041 0.00053 0.00035 0.00104 0.00060 0.00080 0.00050 0.00217 0.00145 0.00156 0.00129

0.00068 0.00050 0.00050 0.00034 0.00113 0.00066 0.00080 0.00054 0.00208 0.00134 0.00146 0.00106

0.00196 0.00089 0.00134 0.00072 0.00235 0.00101 0.00156 0.00092 0.00362 0.00184 0.00343 0.00173

0.00098 0.00068 0.00066 0.00053 0.00144 0.00076 0.00114 0.00060 0.00229 0.00165 0.00197 0.00164

cfg3f

cfg3f cfg3j

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

- Figure 24: Columns 1,2,3: Constant Canon implementation (random, non-trained average of the past 3 tokens, denoted cst-Canon) already achieves strong performance, clearly outperforming vanilla Llama. Columns 2,4,5: Canon layers also perform strongly on GPT2 models (with standard MLP). Our playground reveals standard MLP is slightly weaker than gated MLP, especially in knowledge manipulation (cf. Result 5).

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - original

###### Task Depo1(K=8, k=8/4) GPT2(RoPE) - original

###### Task Depo1(K=8, k=8/4) Llama(RoPE,R2) - original

###### Task Depo1(K=8, k=8/4) GPT2(RoPE,R2) - original

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) GPT2(RoPE) - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) Llama(RoPE,R2) - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) GPT2(RoPE,R2) - Canon-ABCD(res)

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

0/34% 1/50% 1/4% 0/1% 0/27% 0/0% 0/12% 0/0%

0/1% 19/89% 0/14% 10/87% 0/1% 1/26% 0/3% 0/0% 0/0% 0/22% 0/0% 0/0%

0/3% 0/18% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/90% 0/0% 0/0%

0/42% 0/24% 1/0% 0/0% 0/0% 0/22% 0/0% 0/0% 0/4% 0/0% 0/0% 0/0%

99/100% 97/100% 99/100% 100/100% 98/100% 92/99% 95/100% 95/100%

98/100% 98/100% 99/100% 97/99% 100/100% 95/100% 97/100% 93/100%

99/100% 66/100% 99/100% 97/100% 59/100% 91/100% 95/99% 99/100%

72/100% 80/100% 98/100% 97/100% 100/100% 74/100% 82/100% 91/100%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/2% 0/56% 0/0% 0/0%

75/99% 97/100% 85/100% 90/100%

61/100% 97/100% 99/100% 97/100%

20/99% 23/100% 0/91% 69/100%

41/99% 96/100% 88/100% 92/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - original

###### Task Depo2(K=16, k=16/8) GPT2(RoPE) - original

###### Task Depo2(K=16, k=16/8) Llama(RoPE,R2) - original

###### Task Depo2(K=16, k=16/8) GPT2(RoPE,R2) - original

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16, k=16/8) GPT2(RoPE) - Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) Llama(RoPE,R2) - Canon-ABCD(res)

Task Depo2(K=16, k=16/8) GPT2(RoPE,R2) - Canon-ABCD(res)

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

2/1% 2/1% 1/1% 30/99%

1/1% 1/2% 1/7% 2/2% 1/46% 1/22% 1/1% 1/1%

1/3% 1/5% 1/28% 21/92% 1/1% 1/24% 1/1% 1/99% 1/1% 1/2% 1/1% 1/84%

2/3% 1/54% 2/38% 1/85% 1/1% 1/1% 1/1% 1/6% 1/1% 1/37% 1/1% 97/100%

92/100% 100/100% 97/100% 99/100% 97/100% 99/100% 96/100% 97/100% 85/100% 99/100% 98/100% 98/100%

97/100% 97/100% 99/100% 98/100% 86/99% 96/100% 98/100% 98/100% 84/99% 96/100% 91/100% 95/100%

10/90% 94/100% 82/100% 99/100% 72/100% 92/100% 77/99% 100/100%

- 94/99% 99/100% 99/100% 99/100%
- 95/100% 95/100% 81/100% 99/100% 87/100% 98/100% 96/100% 99/100%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/90% 1/3% 21/96%
- 1/2% 1/92% 1/3% 1/50%

1/1% 1/1% 1/1% 1/1%

64/98% 77/99% 96/100% 100/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Llama(RoPE) - original

###### Task Brevo1 GPT2(RoPE) - original

###### Task Brevo1 Llama(RoPE,R2) - original

###### Task Brevo1 GPT2(RoPE,R2) - original

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

Task Brevo1 GPT2(RoPE) - Canon-ABCD(res)

Task Brevo1 Llama(RoPE,R2) - Canon-ABCD(res)

Task Brevo1 GPT2(RoPE,R2) - Canon-ABCD(res)

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

45.6% 76.9% 79.8% 88.5% 32.6% 64.5% 44.5% 63.1%

59.3% 82.8% 64.2% 85.8% 6.9% 50.4% 26.0% 67.5% 5.8% 25.0% 9.7% 37.3%

32.7% 87.5% 67.2% 88.4% 38.0% 36.1% 15.5% 64.0% 16.9% 25.4% 7.2% 20.7%

37.7% 72.5% 71.8% 82.6% 9.2% 45.7% 16.1% 62.5% 5.4% 20.8% 5.5% 24.2%

83.7% 93.8% 91.3% 96.5% 62.9% 84.5% 81.2% 90.7% 47.9% 82.2% 69.7% 84.5%

84.9% 91.2% 88.6% 93.3% 66.3% 78.5% 76.8% 82.5% 34.1% 51.1% 53.4% 70.4%

82.8% 91.7% 90.1% 95.6% 69.8% 82.3% 72.0% 86.4% 35.5% 68.4% 57.5% 78.3%

84.4% 89.1% 85.1% 93.7% 56.2% 76.2% 75.9% 83.8% 33.8% 57.2% 51.7% 72.2%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

8.0% 31.2% 17.7% 27.5%

N=110

N=110

N=110

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Brevo2 Llama(RoPE) - original

Task Brevo2 GPT2(RoPE) - original

Task Brevo2 Llama(RoPE,R2) - original

###### Task Brevo2 GPT2(RoPE,R2) - original

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

Task Brevo2 GPT2(RoPE) - Canon-ABCD(res)

Task Brevo2 Llama(RoPE,R2) - Canon-ABCD(res)

Task Brevo2 GPT2(RoPE,R2) - Canon-ABCD(res)

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

69.3% 89.8% 83.7% 96.0% 40.3% 79.5% 60.5% 88.0% 22.4% 68.2% 40.2% 81.4%

70.9% 85.0% 80.3% 93.7% 44.8% 66.3% 63.4% 81.4% 24.7% 55.9% 43.4% 76.2%

67.0% 90.8% 80.8% 94.7% 33.8% 87.4% 54.3% 88.8% 23.9% 70.7% 37.4% 77.1%

60.4% 90.4% 77.4% 93.7% 45.7% 82.3% 66.4% 88.8% 25.1% 63.6% 39.2% 75.5%

87.1% 95.6% 92.2% 97.1% 75.4% 87.7% 80.1% 93.5% 55.1% 82.5% 69.3% 88.1%

84.2% 94.2% 89.8% 95.9% 62.0% 86.0% 81.2% 91.4% 49.2% 75.5% 63.2% 84.4%

88.4% 94.0% 91.9% 96.7% 69.4% 87.0% 83.4% 92.5% 56.1% 75.2% 65.7% 87.3%

87.3% 92.9% 90.7% 95.6% 71.6% 86.6% 78.9% 92.0% 54.6% 78.7% 64.5% 83.9%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Mano Llama(RoPE) - original

###### Task Mano GPT2(RoPE) - original

###### Task Mano Llama(RoPE,R2) - original

###### Task Mano GPT2(RoPE,R2) - original

Task Mano Llama(RoPE) - Canon-ABCD(res)

Task Mano GPT2(RoPE) - Canon-ABCD(res)

Task Mano Llama(RoPE,R2) - Canon-ABCD(res)

Task Mano GPT2(RoPE,R2) - Canon-ABCD(res)

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

59.4% 75.5% 84.5% 85.2% 55.6% 53.8% 52.5% 46.5% 26.3% 19.7% 20.9% 41.6%

33.1% 22.3% 49.2% 40.7% 20.9% 23.1% 21.8% 12.2% 11.8% 7.4% 20.5% 11.4%

40.0% 38.4% 63.9% 64.7% 15.1% 26.9% 52.1% 33.2%

63.4% 70.2% 77.9% 83.4%

94.2% 98.0% 99.2% 99.6% 89.8% 88.5% 98.2% 99.2% 83.7% 83.6% 88.8% 85.3%

81.4% 87.1% 92.8% 98.0% 76.9% 86.7% 96.2% 92.1% 15.7% 44.5% 69.2% 72.5%

97.6% 98.2% 98.4% 99.1% 87.3% 87.2% 93.8% 96.1% 55.2% 76.6% 81.7% 90.6%

97.9% 98.8% 98.9% 98.8% 94.4% 97.1% 95.3% 94.6% 65.6% 95.0% 95.0% 96.9%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

7.5% 37.5% 72.4% 77.0% 18.5% 26.6% 25.5% 67.3%

7.6% 14.4% 31.3% 15.0%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - original

Task Lano GPT2(RoPE) - original

Task Lano Llama(RoPE,R2) - original

Task Lano GPT2(RoPE,R2) - original

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano GPT2(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE,R2) - Canon-ABCD(res)

Task Lano GPT2(RoPE,R2) - Canon-ABCD(res)

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

91.1% 96.3% 93.4% 97.6% 74.1% 91.4% 82.3% 90.3% 64.0% 75.1% 60.0% 79.1%

78.1% 93.0% 86.5% 94.8% 56.6% 82.4% 71.1% 82.9% 46.1% 69.6% 47.3% 71.8%

93.4% 96.8% 95.7% 96.6% 80.2% 87.9% 79.4% 87.3% 51.3% 68.8% 58.1% 74.5%

94.5% 96.5% 94.8% 97.4% 80.8% 90.3% 85.3% 90.6% 62.8% 80.3% 70.2% 84.7%

95.2% 97.5% 96.0% 98.1% 81.4% 90.1% 85.9% 92.6% 66.0% 77.9% 76.1% 78.9%

91.1% 95.0% 94.8% 96.4% 72.5% 87.2% 78.2% 90.7% 64.3% 74.9% 68.5% 73.9%

95.4% 97.6% 96.7% 97.8% 78.4% 91.6% 83.6% 91.2% 60.3% 74.6% 70.7% 81.9%

95.5% 97.5% 96.7% 97.4% 81.6% 89.9% 86.7% 92.1% 67.6% 76.3% 72.6% 82.5%

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - original

###### Task Lano GPT2(RoPE) - original

Task Lano Llama(RoPE,R2) - original

Task Lano GPT2(RoPE,R2) - original

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano GPT2(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE,R2) - Canon-ABCD(res)

Task Lano GPT2(RoPE,R2) - Canon-ABCD(res)

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

0.00095 0.00050 0.00073 0.00038 0.00135 0.00056 0.00095 0.00057 0.00232 0.00156 0.00235 0.00131

0.00196 0.00089 0.00134 0.00072 0.00235 0.00101 0.00156 0.00092 0.00362 0.00184 0.00343 0.00173

0.00076 0.00042 0.00054 0.00047 0.00105 0.00074 0.00105 0.00076 0.00321 0.00195 0.00265 0.00150

0.00066 0.00045 0.00063 0.00041 0.00103 0.00060 0.00081 0.00055 0.00231 0.00127 0.00184 0.00099

0.00067 0.00041 0.00053 0.00035 0.00104 0.00060 0.00080 0.00050 0.00217 0.00145 0.00156 0.00129

0.00098 0.00068 0.00066 0.00053 0.00144 0.00076 0.00114 0.00060 0.00229 0.00165 0.00197 0.00164

0.00060 0.00037 0.00044 0.00037 0.00116 0.00053 0.00089 0.00055 0.00253 0.00157 0.00184 0.00120

0.00060 0.00043 0.00047 0.00038 0.00099 0.00062 0.00081 0.00057 0.00208 0.00156 0.00175 0.00119

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

gated MLP (silu)

standard MLP (silu)

gated MLP (relu2)

standard MLP (relu2)

gated MLP (silu) + Canon

standard MLP (silu) + Canon

gated MLP (relu2) + Canon

standard MLP (relu2) + Canon

- Figure 25: Effect of ReLU2 activation on standard vs. gated MLP. Columns 1→2, 5→6: gated MLP outperforms standard MLP with silu. Columns 2→4, 6→8: adding ReLU2 to standard MLP yields slight gains. Columns 1→3, 5→7: adding ReLU2 to gated MLP hurts performance.

###### Llama(RoPE) - Canon-ABCD(res)

###### Llama(RoPE) - Canon-ABCD(res)

###### Llama(RoPE) - Canon-ABCD(res)

###### Llama(RoPE) - Canon-ABCD(res)

###### Llama(NoPE) - Canon-ABCD(res)

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

97/100% 92/100% 73/89% 94/100% 57/97% 54/93% 92/99% 99/99% 76/99% 53/99% 16/66% 97/100%

99/100% 97/100% 99/100% 100/100% 98/100% 92/99% 95/100% 95/100%

65/99% 92/100% 78/99% 97/100% 69/100% 94/100% 74/99% 70/97% 94/100% 74/100% 92/99% 6/99%

97/100% 98/100% 98/100% 99/100% 94/100% 71/100% 97/100% 100/100% 93/100% 70/99% 71/100% 98/100%

99/100% 99/100% 99/100% 100/100% 96/99% 99/100% 99/100% 99/100% 99/100% 99/100% 98/100% 99/100%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

75/99% 97/100% 85/100% 90/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-ABCD(res)

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

91/99% 97/100% 98/100% 99/100% 98/100% 98/100% 99/100% 98/100%

92/100% 100/100% 97/100% 99/100% 97/100% 99/100% 96/100% 97/100% 85/100% 99/100% 98/100% 98/100%

94/99% 99/100% 99/100% 93/100% 66/99% 99/100% 98/100% 97/100%

99/100% 99/100% 98/100% 100/100% 98/100% 99/100% 99/100% 99/100%

96/100% 85/99% 86/100% 99/100% 94/100% 86/99% 99/100% 99/100% 90/100% 98/100% 93/100% 96/100%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

71/98% 90/100% 94/99% 96/100%

84/100% 96/100% 96/100% 96/100%

85/99% 99/100% 96/100% 99/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

Task Brevo1 Llama(NoPE) - Canon-ABCD(res)

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

84.6% 88.7% 88.3% 91.3% 51.3% 72.4% 69.9% 75.7% 24.8% 49.1% 41.2% 58.8%

83.7% 93.8% 91.3% 96.5% 62.9% 84.5% 81.2% 90.7% 47.9% 82.2% 69.7% 84.5%

90.4% 94.6% 91.7% 96.3% 72.7% 84.0% 83.2% 91.1% 58.3% 77.6% 61.2% 84.7%

90.0% 93.7% 91.8% 96.1% 69.0% 83.6% 79.5% 91.5% 42.3% 63.8% 72.5% 79.7%

84.8% 94.4% 91.1% 96.2% 63.9% 85.8% 75.5% 92.2% 42.0% 75.3% 58.2% 84.9%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

Task Brevo2 Llama(NoPE) - Canon-ABCD(res)

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

87.5% 94.5% 92.3% 95.4% 66.0% 85.3% 79.3% 90.5% 44.6% 75.5% 68.5% 87.8%

87.1% 95.6% 92.2% 97.1% 75.4% 87.7% 80.1% 93.5% 55.1% 82.5% 69.3% 88.1%

85.2% 94.1% 92.7% 96.3% 68.8% 87.9% 82.0% 89.2% 51.2% 82.1% 74.2% 85.9%

89.6% 96.4% 94.7% 97.3% 74.5% 91.2% 84.9% 95.5% 56.0% 80.3% 73.0% 91.7%

87.4% 93.2% 89.0% 96.1% 61.2% 84.0% 75.2% 91.7% 40.4% 56.0% 56.3% 79.9%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Mano Llama(RoPE) - Canon-ABCD(res)

Task Mano Llama(RoPE) - Canon-ABCD(res)

Task Mano Llama(RoPE) - Canon-ABCD(res)

Task Mano Llama(RoPE) - Canon-ABCD(res)

Task Mano Llama(NoPE) - Canon-ABCD(res)

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

95.1% 99.3% 99.3% 99.5% 66.0% 94.6% 97.1% 98.8% 63.7% 82.8% 91.4% 83.0%

94.2% 98.0% 99.2% 99.6% 89.8% 88.5% 98.2% 99.2% 83.7% 83.6% 88.8% 85.3%

98.9% 98.3% 97.6% 99.7% 80.9% 97.1% 97.5% 99.1% 69.1% 96.7% 93.6% 97.9%

98.3% 99.4% 99.0% 99.6% 79.2% 96.8% 98.0% 96.3% 85.1% 77.9% 72.4% 88.0%

97.7% 98.9% 99.3% 99.3% 83.1% 90.1% 95.9% 98.1% 53.7% 55.5% 89.4% 94.3%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

###### Task Lano Llama(NoPE) - Canon-ABCD(res)

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

96.6% 98.0% 97.2% 98.3% 88.2% 92.0% 88.6% 94.3% 75.2% 87.1% 83.0% 86.7%

95.2% 97.5% 96.0% 98.1% 81.4% 90.1% 85.9% 92.6% 66.0% 77.9% 76.1% 78.9%

93.8% 97.0% 96.2% 97.6% 77.7% 88.6% 86.0% 91.5% 58.4% 76.6% 72.8% 81.1%

96.0% 97.3% 96.6% 97.5% 82.0% 88.9% 87.8% 93.3% 71.5% 81.9% 74.1% 84.5%

87.9% 91.9% 88.5% 92.5% 55.1% 70.3% 58.6% 78.3% 33.5% 51.0% 37.2% 53.1%

cfg3f

cfg3f

cfg3f cfg3j

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

Task Lano Llama(NoPE) - Canon-ABCD(res)

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

0.00048 0.00034 0.00042 0.00028 0.00071 0.00052 0.00065 0.00040 0.00155 0.00090 0.00113 0.00091

0.00067 0.00041 0.00053 0.00035 0.00104 0.00060 0.00080 0.00050 0.00217 0.00145 0.00156 0.00129

0.00078 0.00047 0.00058 0.00042 0.00116 0.00071 0.00080 0.00055 0.00270 0.00157 0.00173 0.00119

0.00060 0.00045 0.00048 0.00037 0.00099 0.00070 0.00071 0.00046 0.00179 0.00122 0.00162 0.00106

0.00129 0.00090 0.00117 0.00084 0.00253 0.00161 0.00223 0.00116 0.00509 0.00318 0.00452 0.00307

cfg3f

cfg3f

cfg3f

cfg3f cfg3j

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- Figure 26: Transformer+Canon with varying RoPE configurations. From left to right: (1) RoPE; (2) RoPEˇ “: half of heads each with half RoPE dimensions; (3) RoPEˇ “ˇ “: a quarter of heads with full RoPE dimensions;

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

(4) RoPEˇ “ˇ “ˇ “: all heads each with quarter RoPE dimensions; (5) NoPE.

Conclusion: Canon layers eliminate the need for extensive RoPE usage, and reducing RoPE usage to 1/4 is even preferable, outperforming both full RoPE and NoPE setups. Among these reduced RoPE variants, RoPEˇ “ achieves slightly better overall performance.

#### G Complete Ablation Studies

This section presents full ablation results, including KL-divergence evaluations for Task Mano. These details were omitted from the main text for clarity but are included here for completeness and for readers seeking deeper experimental insight.

##### G.1 Llama(RoPE) family

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - original

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-B(res)

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-AC(res)

###### Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-BD(res)

Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ACD(res)

Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ABC(res)

Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ABCD(res)

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

0/34% 1/50% 1/4% 0/1% 0/27% 0/0% 0/12% 0/0%

51/99% 51/97% 13/86% 25/43% 11/98% 20/91% 2/61% 0/40%

98/100% 95/100% 97/99% 94/100% 52/98% 85/100% 5/99% 51/95%

93/98% 96/99% 85/100% 92/100% 50/83% 69/100% 86/99% 75/94%

100/100% 92/100% 57/99% 96/100% 68/98% 32/97% 90/99% 98/100% 0/85% 40/96% 93/95% 68/100%

96/100% 87/99% 93/97% 59/81% 19/100% 65/97% 12/88% 89/99% 6/92% 78/99% 0/99% 3/86%

97/100% 92/100% 73/89% 94/100% 57/97% 54/93% 92/99% 99/99% 76/99% 53/99% 16/66% 97/100%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/2% 0/56% 0/0% 0/0%

0/21% 55/92% 0/77% 0/13%

2/21% 44/97% 0/91% 89/99%

0/73% 19/98% 30/98% 51/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - original

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-B(res)

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-AC(res)

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-BD(res)

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ACD(res)

###### Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABC(res)

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

2/1% 2/1% 1/1% 30/99%

1/76% 43/89% 2/86% 98/100% 3/88% 35/96% 24/95% 96/100% 1/67% 20/91% 1/53% 89/99%

63/99% 89/100% 98/100% 99/100% 34/86% 75/100% 93/99% 99/100% 30/95% 90/99% 99/100% 93/99%

14/97% 83/99% 86/100% 99/100%

91/99% 86/98% 52/99% 82/98% 89/99% 54/94% 15/92% 99/100% 73/99% 94/100% 9/85% 96/100%

97/100% 96/100% 96/100% 99/100% 90/100% 97/100% 76/99% 99/100%

91/99% 97/100% 98/100% 99/100% 98/100% 98/100% 99/100% 98/100%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/90% 1/3% 21/96%
- 1/2% 1/92% 1/3% 1/50%

1/76% 96/100% 59/99% 96/100% 1/3% 85/99% 1/23% 93/99%

79/98% 97/100% 96/100% 98/100%

71/98% 90/100% 94/99% 96/100%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Llama(RoPE) - original

###### Task Brevo1 Llama(RoPE) - Canon-B(res)

Task Brevo1 Llama(RoPE) - Canon-AC(res)

###### Task Brevo1 Llama(RoPE) - Canon-BD(res)

Task Brevo1 Llama(RoPE) - Canon-ACD(res)

###### Task Brevo1 Llama(RoPE) - Canon-ABC(res)

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

45.6% 76.9% 79.8% 88.5% 32.6% 64.5% 44.5% 63.1%

42.3% 84.1% 86.7% 85.9% 26.1% 57.8% 50.7% 57.2% 15.3% 31.8% 11.0% 36.2%

71.7% 85.1% 80.4% 91.3% 53.6% 72.5% 58.8% 76.8% 23.4% 42.5% 37.2% 48.6%

78.5% 90.7% 80.9% 93.7% 56.0% 68.1% 64.5% 72.5% 25.5% 45.8% 33.4% 55.4%

85.3% 90.6% 86.6% 92.4% 62.7% 78.6% 68.4% 87.8% 38.2% 54.0% 46.0% 49.9%

76.8% 93.1% 79.3% 92.3% 44.0% 69.9% 49.6% 67.6% 27.0% 35.6% 34.3% 52.6%

84.6% 88.7% 88.3% 91.3% 51.3% 72.4% 69.9% 75.7% 24.8% 49.1% 41.2% 58.8%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

8.0% 31.2% 17.7% 27.5%

N=110

N=110

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 Llama(RoPE) - original

Task Brevo2 Llama(RoPE) - Canon-B(res)

Task Brevo2 Llama(RoPE) - Canon-AC(res)

Task Brevo2 Llama(RoPE) - Canon-BD(res)

Task Brevo2 Llama(RoPE) - Canon-ACD(res)

Task Brevo2 Llama(RoPE) - Canon-ABC(res)

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

69.3% 89.8% 83.7% 96.0% 40.3% 79.5% 60.5% 88.0% 22.4% 68.2% 40.2% 81.4%

74.1% 91.2% 84.0% 94.6% 49.5% 80.4% 58.2% 87.0% 33.0% 75.1% 42.5% 77.8%

82.9% 94.3% 91.5% 96.5% 65.3% 86.4% 79.3% 91.8% 39.2% 77.5% 66.2% 87.2%

85.7% 94.8% 89.0% 95.9% 62.6% 82.7% 77.1% 92.2% 38.1% 74.5% 55.0% 84.5%

86.4% 94.3% 92.1% 97.0% 71.7% 88.4% 83.2% 92.6% 48.5% 78.3% 67.0% 88.7%

85.1% 92.7% 89.2% 95.7% 66.9% 83.9% 74.0% 89.8% 46.8% 74.3% 59.1% 82.5%

87.5% 94.5% 92.3% 95.4% 66.0% 85.3% 79.3% 90.5% 44.6% 75.5% 68.5% 87.8%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Mano Llama(RoPE) - original

Task Mano Llama(RoPE) - Canon-B(res)

Task Mano Llama(RoPE) - Canon-AC(res)

Task Mano Llama(RoPE) - Canon-BD(res)

Task Mano Llama(RoPE) - Canon-ACD(res)

Task Mano Llama(RoPE) - Canon-ABC(res)

Task Mano Llama(RoPE) - Canon-ABCD(res)

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

59.4% 75.5% 84.5% 85.2% 55.6% 53.8% 52.5% 46.5% 26.3% 19.7% 20.9% 41.6%

95.2% 86.6% 97.0% 97.5% 47.1% 44.0% 91.0% 90.3% 43.2% 59.4% 93.1% 79.4%

94.2% 98.0% 94.5% 95.8% 76.5% 87.6% 91.6% 88.6% 49.7% 53.0% 64.6% 78.3%

76.7% 84.2% 97.7% 98.6% 81.9% 86.1% 83.6% 87.7% 41.2% 53.9% 81.2% 79.8%

95.6% 98.6% 98.9% 97.6% 95.8% 89.7% 96.8% 98.0% 78.5% 42.4% 76.9% 83.2%

97.8% 97.8% 94.1% 97.7% 86.9% 97.7% 97.1% 97.5% 59.5% 59.9% 95.0% 86.4%

95.1% 99.3% 99.3% 99.5% 66.0% 94.6% 97.1% 98.8% 63.7% 82.8% 91.4% 83.0%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - original

Task Lano Llama(RoPE) - Canon-B(res)

Task Lano Llama(RoPE) - Canon-AC(res)

Task Lano Llama(RoPE) - Canon-BD(res)

Task Lano Llama(RoPE) - Canon-ACD(res)

Task Lano Llama(RoPE) - Canon-ABC(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

91.1% 96.3% 93.4% 97.6% 74.1% 91.4% 82.3% 90.3% 64.0% 75.1% 60.0% 79.1%

94.1% 97.7% 95.7% 97.5% 86.8% 93.7% 87.9% 94.2% 65.9% 84.0% 78.0% 89.8%

96.7% 97.5% 97.5% 98.2% 87.8% 93.1% 88.9% 95.1% 73.5% 86.3% 80.6% 86.3%

96.2% 97.9% 96.4% 97.9% 82.5% 93.2% 88.3% 93.3% 69.3% 82.6% 77.8% 85.2%

97.1% 98.1% 97.0% 98.5% 86.0% 92.1% 90.2% 93.4% 73.4% 84.6% 77.5% 84.7%

96.3% 97.8% 97.1% 97.7% 86.4% 92.0% 91.0% 94.9% 79.1% 85.7% 82.1% 89.8%

96.6% 98.0% 97.2% 98.3% 88.2% 92.0% 88.6% 94.3% 75.2% 87.1% 83.0% 86.7%

cfg3f

cfg3f cfg3j

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Llama(RoPE) - original

Task Lano Llama(RoPE) - Canon-B(res)

Task Lano Llama(RoPE) - Canon-AC(res)

Task Lano Llama(RoPE) - Canon-BD(res)

Task Lano Llama(RoPE) - Canon-ACD(res)

Task Lano Llama(RoPE) - Canon-ABC(res)

Task Lano Llama(RoPE) - Canon-ABCD(res)

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

0.00095 0.00050 0.00073 0.00038 0.00135 0.00056 0.00095 0.00057 0.00232 0.00156 0.00235 0.00131

0.00069 0.00038 0.00056 0.00039 0.00082 0.00046 0.00071 0.00042 0.00214 0.00109 0.00140 0.00073

0.00046 0.00039 0.00039 0.00032 0.00073 0.00046 0.00063 0.00039 0.00167 0.00095 0.00122 0.00094

0.00052 0.00036 0.00052 0.00035 0.00096 0.00047 0.00068 0.00044 0.00186 0.00115 0.00135 0.00098

0.00044 0.00034 0.00040 0.00030 0.00079 0.00051 0.00059 0.00043 0.00170 0.00101 0.00135 0.00101

0.00056 0.00036 0.00043 0.00035 0.00077 0.00051 0.00055 0.00036 0.00134 0.00094 0.00117 0.00069

0.00048 0.00034 0.00042 0.00028 0.00071 0.00052 0.00065 0.00040 0.00155 0.00090 0.00113 0.00091

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

- Figure 27: Llama(RoPE) family: (from left to right) original, Canon-B, -AC, -BD, -ACD, -ABC, -ABCD. This figure complements Figure 10 and gives more technical details.

[Figure 395]

8L512D 12L512D 8L768D 12L768D

N=225 N=300 N=375

0/34% 1/50% 1/4% 0/1% 0/27% 0/0% 0/12% 0/0%

0/2% 0/56% 0/0% 0/0%

Task Depo1(K=8, k=8/4) Llama(RoPE) - original

[Figure 396]

8L512D 12L512D 8L768D 12L768D

N=225 N=300 N=375

63/96% 93/100% 61/97% 98/100% 12/54% 42/100% 16/97% 98/100% 0/44% 32/95% 20/76% 48/98%

Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-B(no-res)

[Figure 397]

8L512D 12L512D 8L768D 12L768D

N=225 N=300 N=375

97/100% 92/100% 73/89% 94/100% 57/97% 54/93% 92/99% 99/99% 76/99% 53/99% 16/66% 97/100%

Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ABCD(res)

[Figure 398]

8L512D 12L512D 8L768D 12L768D

N=225 N=300 N=375

45/99% 95/99% 2/92% 99/100% 4/80% 97/100% 0/98% 84/100% 0/59% 75/100% 97/100% 99/100%

Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-B(no-res)

[Figure 399]

8L512D 12L512D 8L768D 12L768D

N=225 N=300 N=375

99/100% 97/100% 99/100% 100/100% 98/100% 92/99% 95/100% 95/100%

75/99% 97/100% 85/100% 90/100%

Task Depo1(K=8, k=8/4) Llama(RoPE) - Canon-ABCD(res)

[Figure 400]

8L512D 12L512D 8L768D 12L768D

N=75 N=100 N=125

2/1% 2/1% 1/1% 30/99%

- 1/1% 1/90% 1/3% 21/96%
- 1/2% 1/92% 1/3% 1/50%

Task Depo2(K=16, k=16/8) Llama(RoPE) - original

[Figure 401]

8L512D 12L512D 8L768D 12L768D

N=75 N=100 N=125

1/91% 1/34% 1/50% 9/97%

1/1% 2/54% 2/62% 65/97% 1/18% 1/15% 54/93% 1/82%

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-B(no-res)

[Figure 402]

8L512D 12L512D 8L768D 12L768D

N=75 N=100 N=125

91/99% 97/100% 98/100% 99/100% 98/100% 98/100% 99/100% 98/100%

71/98% 90/100% 94/99% 96/100%

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

[Figure 403]

8L512D 12L512D 8L768D 12L768D

N=75 N=100 N=125

2/77% 1/16% 1/29% 36/92% 1/33% 1/52% 5/85% 61/97% 1/56% 26/94% 37/97% 37/98%

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-B(no-res)

[Figure 404]

8L512D 12L512D 8L768D 12L768D

N=75 N=100 N=125

92/100% 100/100% 97/100% 99/100% 97/100% 99/100% 96/100% 97/100% 85/100% 99/100% 98/100% 98/100%

Task Depo2(K=16, k=16/8) Llama(RoPE) - Canon-ABCD(res)

[Figure 405]

8L512D 12L512D 8L768D 12L768D

N=70 N=90

N=110

45.6% 76.9% 79.8% 88.5% 32.6% 64.5% 44.5% 63.1%

8.0% 31.2% 17.7% 27.5%

Task Brevo1 Llama(RoPE) - original

[Figure 406]

8L512D 12L512D 8L768D 12L768D

N=70 N=90

N=110

42.2% 84.6% 75.2% 90.1% 32.8% 61.6% 32.9% 70.4%

5.3% 35.1% 17.6% 51.0%

Task Brevo1 Llama(RoPE) - Canon-B(no-res)

[Figure 407]

8L512D 12L512D 8L768D 12L768D

N=70 N=90

N=110

84.6% 88.7% 88.3% 91.3% 51.3% 72.4% 69.9% 75.7% 24.8% 49.1% 41.2% 58.8%

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

[Figure 408]

8L512D 12L512D 8L768D 12L768D

N=70 N=90

N=110

64.4% 88.5% 80.0% 92.2% 35.1% 70.0% 54.1% 84.8% 15.6% 47.2% 43.3% 68.8%

Task Brevo1 Llama(RoPE) - Canon-B(no-res)

[Figure 409]

8L512D 12L512D 8L768D 12L768D

N=70 N=90

N=110

83.7% 93.8% 91.3% 96.5% 62.9% 84.5% 81.2% 90.7% 47.9% 82.2% 69.7% 84.5%

Task Brevo1 Llama(RoPE) - Canon-ABCD(res)

[Figure 410]

8L512D 12L512D 8L768D 12L768D

N=30 N=40 N=50

69.3% 89.8% 83.7% 96.0% 40.3% 79.5% 60.5% 88.0% 22.4% 68.2% 40.2% 81.4%

Task Brevo2 Llama(RoPE) - original

[Figure 411]

8L512D 12L512D 8L768D 12L768D

N=30 N=40 N=50

74.5% 87.1% 84.2% 91.0% 44.5% 71.8% 65.2% 81.8% 21.1% 60.2% 40.2% 65.6%

Task Brevo2 Llama(RoPE) - Canon-B(no-res)

[Figure 412]

8L512D 12L512D 8L768D 12L768D

N=30 N=40 N=50

87.5% 94.5% 92.3% 95.4% 66.0% 85.3% 79.3% 90.5% 44.6% 75.5% 68.5% 87.8%

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

[Figure 413]

8L512D 12L512D 8L768D 12L768D

N=30 N=40 N=50

75.0% 91.5% 82.9% 92.9% 59.0% 76.8% 67.6% 85.6% 40.2% 67.6% 44.0% 80.0%

Task Brevo2 Llama(RoPE) - Canon-B(no-res)

[Figure 414]

8L512D 12L512D 8L768D 12L768D

N=30 N=40 N=50

87.1% 95.6% 92.2% 97.1% 75.4% 87.7% 80.1% 93.5% 55.1% 82.5% 69.3% 88.1%

Task Brevo2 Llama(RoPE) - Canon-ABCD(res)

[Figure 415]

8L512D 12L512D 8L768D 12L768D

L=10 L=13 L=16

59.4% 75.5% 84.5% 85.2% 55.6% 53.8% 52.5% 46.5% 26.3% 19.7% 20.9% 41.6%

Task Mano Llama(RoPE) - original

[Figure 416]

8L512D 12L512D 8L768D 12L768D

L=10 L=13 L=16

71.2% 70.7% 83.8% 85.9% 55.8% 87.8% 65.0% 79.4%

9.5% 13.7% 66.9% 27.4%

Task Mano Llama(RoPE) - Canon-B(no-res)

[Figure 417]

8L512D 12L512D 8L768D 12L768D

L=10 L=13 L=16

95.1% 99.3% 99.3% 99.5% 66.0% 94.6% 97.1% 98.8% 63.7% 82.8% 91.4% 83.0%

Task Mano Llama(RoPE) - Canon-ABCD(res)

[Figure 418]

8L512D 12L512D 8L768D 12L768D

L=10 L=13 L=16

92.3% 90.5% 93.2% 94.3% 48.8% 11.7% 72.2% 90.1% 14.5% 35.0% 55.6% 53.9%

Task Mano Llama(RoPE) - Canon-B(no-res)

[Figure 419]

8L512D 12L512D 8L768D 12L768D

L=10 L=13 L=16

94.2% 98.0% 99.2% 99.6% 89.8% 88.5% 98.2% 99.2% 83.7% 83.6% 88.8% 85.3%

Task Mano Llama(RoPE) - Canon-ABCD(res)

[Figure 420]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

91.1% 96.3% 93.4% 97.6% 74.1% 91.4% 82.3% 90.3% 64.0% 75.1% 60.0% 79.1%

Task Lano Llama(RoPE) - original

[Figure 421]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

92.1% 96.0% 94.5% 95.7% 55.0% 69.3% 68.1% 80.3% 40.4% 50.5% 40.0% 53.0%

Task Lano Llama(RoPE) - Canon-B(no-res)

[Figure 422]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

96.6% 98.0% 97.2% 98.3% 88.2% 92.0% 88.6% 94.3% 75.2% 87.1% 83.0% 86.7%

Task Lano Llama(RoPE) - Canon-ABCD(res)

[Figure 423]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

89.1% 92.5% 90.7% 94.1% 53.2% 65.0% 52.1% 65.9% 28.9% 40.1% 34.3% 35.8%

Task Lano Llama(RoPE) - Canon-B(no-res)

[Figure 424]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

95.2% 97.5% 96.0% 98.1% 81.4% 90.1% 85.9% 92.6% 66.0% 77.9% 76.1% 78.9%

Task Lano Llama(RoPE) - Canon-ABCD(res)

[Figure 425]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

0.00095 0.00050 0.00073 0.00038 0.00135 0.00056 0.00095 0.00057 0.00232 0.00156 0.00235 0.00131

Task Lano Llama(RoPE) - original

[Figure 426]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

0.00092 0.00050 0.00066 0.00056 0.00247 0.00160 0.00170 0.00108 0.00428 0.00326 0.00432 0.00312

Task Lano Llama(RoPE) - Canon-B(no-res)

[Figure 427]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

0.00048 0.00034 0.00042 0.00028 0.00071 0.00052 0.00065 0.00040 0.00155 0.00090 0.00113 0.00091

Task Lano Llama(RoPE) - Canon-ABCD(res)

[Figure 428]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

0.00120 0.00091 0.00107 0.00070 0.00260 0.00187 0.00262 0.00177 0.00571 0.00439 0.00503 0.00456

Task Lano Llama(RoPE) - Canon-B(no-res)

[Figure 429]

8L512D 12L512D 8L768D 12L768D

cfg3f

- cfg3j

- cfg3k

0.00067 0.00041 0.00053 0.00035 0.00104 0.00060 0.00080 0.00050 0.00217 0.00145 0.00156 0.00129

Task Lano Llama(RoPE) - Canon-ABCD(res)

- Figure 28: Llama(RoPE) family: (left to right) original, Canon-B(no-res), Canon-ABCD(res), ˇ “Canon-B(no-res), ˇ “Canon-ABCD(res). This figure complements Figure 10 and directly compares to Primer [59] (i.e., Canon-B(no-res)), showing its inefficiency and highlighting: (1) Canon layers are not tied to Attention;

(2) Canon(res) at multiple points is safe and more effective.

##### G.2 Llama(NoPE) family

###### Task Depo1(K=8, k=8/4) Llama(NoPE) - original

###### Task Depo1(K=8, k=8/4) Llama(NoPE) - Canon-B(res)

###### Task Depo1(K=8, k=8/4) Llama(NoPE) - Canon-AC(res)

###### Task Depo1(K=8, k=8/4) Llama(NoPE) - Canon-BD(res)

Task Depo1(K=8, k=8/4) Llama(NoPE) - Canon-ACD(res)

Task Depo1(K=8, k=8/4) Llama(NoPE) - Canon-ABC(res)

Task Depo1(K=8, k=8/4) Llama(NoPE) - Canon-ABCD(res)

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0%

53/95% 0/13% 74/98% 62/97% 25/93% 0/12% 26/74% 49/93%

86/99% 97/100% 98/100% 98/100%

99/100% 94/100% 97/100% 99/99%

95/100% 99/100% 96/100% 99/100% 82/98% 95/99% 95/100% 86/91% 87/98% 71/96% 87/100% 83/99%

88/100% 99/100% 88/100% 99/100% 73/99% 93/100% 93/99% 99/100% 90/99% 97/99% 76/99% 93/99%

99/100% 99/100% 99/100% 100/100% 96/99% 99/100% 99/100% 99/100% 99/100% 99/100% 98/100% 99/100%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

- 80/100% 95/100% 93/100% 93/99%
- 81/98% 81/98% 76/95% 89/99%

- 94/100% 98/100% 86/99% 88/99%
- 95/100% 91/99% 57/98% 95/97%

0/23% 1/76% 16/71% 52/94%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) Llama(NoPE) - original

###### Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-B(res)

Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-AC(res)

###### Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-BD(res)

###### Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-ACD(res)

Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-ABC(res)

Task Depo2(K=16, k=16/8) Llama(NoPE) - Canon-ABCD(res)

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0%

1/1% 47/93% 2/15% 71/98%

59/98% 98/100% 95/100% 90/100% 19/95% 87/99% 79/99% 96/100% 75/98% 91/99% 87/100% 90/99%

8/91% 72/100% 13/96% 97/100% 2/75% 55/98% 40/96% 98/100%

92/100% 95/100% 96/100% 99/100%

90/100% 99/100% 91/100% 100/100% 75/99% 95/100% 87/99% 99/100% 65/99% 98/100% 93/100% 99/100%

96/100% 85/99% 86/100% 99/100% 94/100% 86/99% 99/100% 99/100% 90/100% 98/100% 93/100% 96/100%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/60% 1/1% 17/91%
- 1/2% 84/98% 1/48% 2/92%

73/99% 97/100% 30/97% 99/100% 92/100% 97/100% 76/100% 95/100%

28/97% 31/96% 6/87% 63/96%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Llama(NoPE) - original

###### Task Brevo1 Llama(NoPE) - Canon-B(res)

###### Task Brevo1 Llama(NoPE) - Canon-AC(res)

Task Brevo1 Llama(NoPE) - Canon-BD(res)

Task Brevo1 Llama(NoPE) - Canon-ACD(res)

Task Brevo1 Llama(NoPE) - Canon-ABC(res)

Task Brevo1 Llama(NoPE) - Canon-ABCD(res)

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

0.2% 0.0% 0.0% 0.4% 0.1% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.1%

15.6% 82.1% 14.5% 61.6% 12.6% 17.6% 3.3% 51.0% 1.0% 2.1% 1.3% 2.9%

69.2% 92.2% 85.1% 94.6% 43.7% 80.9% 60.9% 85.6% 33.7% 60.5% 43.8% 63.6%

76.3% 93.6% 83.2% 92.8% 35.2% 76.6% 66.6% 81.7% 14.7% 67.3% 44.9% 67.3%

80.2% 93.6% 89.3% 95.6% 60.4% 87.3% 70.5% 90.9% 46.3% 75.5% 54.3% 83.0%

78.8% 92.1% 88.3% 95.3% 57.2% 82.9% 72.9% 89.0% 27.2% 64.7% 37.7% 77.1%

84.8% 94.4% 91.1% 96.2% 63.9% 85.8% 75.5% 92.2% 42.0% 75.3% 58.2% 84.9%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=110

N=110

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 Llama(NoPE) - original

###### Task Brevo2 Llama(NoPE) - Canon-B(res)

Task Brevo2 Llama(NoPE) - Canon-AC(res)

Task Brevo2 Llama(NoPE) - Canon-BD(res)

Task Brevo2 Llama(NoPE) - Canon-ACD(res)

Task Brevo2 Llama(NoPE) - Canon-ABC(res)

Task Brevo2 Llama(NoPE) - Canon-ABCD(res)

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0%

62.0% 84.0% 75.2% 93.2% 21.9% 64.2% 46.8% 81.2%

76.9% 92.7% 86.3% 95.4% 49.5% 81.6% 63.3% 88.7% 30.3% 72.6% 48.4% 75.7%

77.5% 93.1% 83.5% 94.1% 53.2% 80.0% 64.6% 86.3% 33.9% 53.9% 31.4% 68.0%

82.9% 92.5% 86.0% 96.0% 63.5% 84.3% 68.6% 91.6% 36.1% 75.4% 54.2% 85.2%

80.5% 91.5% 87.2% 95.4% 57.9% 78.8% 68.0% 88.8% 37.1% 66.5% 42.8% 76.0%

87.4% 93.2% 89.0% 96.1% 61.2% 84.0% 75.2% 91.7% 40.4% 56.0% 56.3% 79.9%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

7.0% 48.2% 25.2% 50.4%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Mano Llama(NoPE) - original

Task Mano Llama(NoPE) - Canon-B(res)

Task Mano Llama(NoPE) - Canon-BD(res)

Task Mano Llama(NoPE) - Canon-AC(res)

Task Mano Llama(NoPE) - Canon-ACD(res)

Task Mano Llama(NoPE) - Canon-ABC(res)

Task Mano Llama(NoPE) - Canon-ABCD(res)

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

- 7.1% 7.1% 7.1% 6.9%
- 7.1% 7.1% 7.2% 7.1%
- 7.3% 7.3% 7.4% 7.3%

80.9% 69.2% 90.9% 95.6% 71.9% 65.4% 82.6% 70.8% 22.3% 9.4% 60.3% 80.5%

89.3% 93.8% 97.6% 99.2% 79.9% 93.9% 92.8% 98.1% 58.5% 84.3% 83.6% 72.1%

93.6% 68.1% 97.3% 98.3% 75.1% 78.2% 88.5% 97.1% 57.4% 87.1% 81.0% 90.9%

94.0% 96.2% 98.4% 99.3% 92.9% 92.4% 95.4% 98.7% 71.2% 87.3% 90.2% 96.8%

92.3% 98.8% 98.4% 98.6% 87.7% 84.2% 94.7% 93.1% 39.0% 67.7% 87.9% 93.3%

97.7% 98.9% 99.3% 99.3% 83.1% 90.1% 95.9% 98.1% 53.7% 55.5% 89.4% 94.3%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Lano Llama(NoPE) - original

###### Task Lano Llama(NoPE) - Canon-B(res)

Task Lano Llama(NoPE) - Canon-AC(res)

Task Lano Llama(NoPE) - Canon-BD(res)

###### Task Lano Llama(NoPE) - Canon-ACD(res)

###### Task Lano Llama(NoPE) - Canon-ABC(res)

Task Lano Llama(NoPE) - Canon-ABCD(res)

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

0.0% 0.0% 0.0% 38.8% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0%

46.5% 83.4% 57.3% 82.7%

76.9% 88.0% 82.8% 90.6% 40.2% 74.2% 59.9% 76.7% 27.2% 32.8% 39.1% 54.0%

80.0% 90.9% 84.6% 89.7% 27.5% 75.4% 46.0% 72.7% 23.8% 41.7% 29.4% 42.3%

84.0% 90.1% 87.5% 93.1% 44.8% 69.2% 65.5% 82.3% 28.7% 51.0% 44.7% 51.2%

83.9% 89.3% 87.6% 91.1% 52.4% 65.3% 60.4% 78.7% 27.0% 49.7% 37.6% 54.1%

87.9% 91.9% 88.5% 92.5% 55.1% 70.3% 58.6% 78.3% 33.5% 51.0% 37.2% 53.1%

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

- 18.3% 58.3% 24.9% 65.2%
- 19.7% 28.7% 26.5% 41.4%

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Lano Llama(NoPE) - original

###### Task Lano Llama(NoPE) - Canon-B(res)

###### Task Lano Llama(NoPE) - Canon-BD(res)

###### Task Lano Llama(NoPE) - Canon-AC(res)

###### Task Lano Llama(NoPE) - Canon-ACD(res)

###### Task Lano Llama(NoPE) - Canon-ABC(res)

Task Lano Llama(NoPE) - Canon-ABCD(res)

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

0.33576 0.13860 0.33685 0.00655 0.30068 0.27650 0.29977 0.27394 0.26508 0.26093 0.26957 0.25573

0.00569 0.00177 0.00418 0.00172 0.00612 0.00230 0.00506 0.00189 0.00749 0.00559 0.00616 0.00419

0.00196 0.00104 0.00160 0.00109 0.00476 0.00134 0.00310 0.00146 0.00660 0.00405 0.00553 0.00397

0.00230 0.00126 0.00171 0.00101 0.00360 0.00131 0.00220 0.00126 0.00590 0.00502 0.00441 0.00302

0.00156 0.00105 0.00130 0.00080 0.00317 0.00161 0.00189 0.00095 0.00558 0.00334 0.00383 0.00320

0.00167 0.00115 0.00134 0.00097 0.00258 0.00186 0.00213 0.00119 0.00589 0.00342 0.00461 0.00298

0.00129 0.00090 0.00117 0.00084 0.00253 0.00161 0.00223 0.00116 0.00509 0.00318 0.00452 0.00307

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

- Figure 29: Llama(NoPE) family: (from left to right) original, Canon-B, -AC, -BD, -ACD, -ABC, -ABCD. This figure complements Figure 10 and gives more technical details.

##### G.3 Mamba2 family

###### Task Brevo1 - Ablation study - Mamba2

###### Task Mano - Ablation study - Mamba2

###### Task Depo1(K=4, k=4/2) - Ablation study - Mamba2

[Figure 479]

16/82% 59/95% 81/97% 14/72% 23/76% 20/57% 92/98% 47/90% 95/99%

[Figure 480]

[Figure 481]

89.5% 89.7% 94.1% 62.7% 1.2% 56.5% 96.2% 92.1% 95.0% 85.3% 70.0% 87.9% 30.8% 1.9% 24.2% 92.2% 69.3% 87.1% 61.6% 10.2% 42.3% 23.0% 0.4% 26.2% 84.5% 35.4% 23.7%

99.7% 99.8% 99.9% 60.7% 70.5% 98.8% 92.0% 99.9% 94.0% 99.6% 99.5% 99.8% 14.5% 39.1% 88.0% 77.6% 99.7% 59.9% 99.2% 99.0% 99.7% 11.9% 33.1% 94.0% 73.2% 98.9% 52.4%

N=225 - NoRes

N=70 - NoRes N=70 - Res N=90 - NoRes

L=10 - NoRes L=10 - Res L=13 - NoRes L=13 - Res L=16 - NoRes L=16 - Res

N=225 - Res

5/38% 14/77% 38/87% 37/81% 15/72% 3/33% 71/92% 21/77% 58/98%

N=300 - NoRes

N=90 - Res N=110 - NoRes

N=300 - Res

3/26% 8/64% 45/86% 9/37% 13/62% 2/33% 32/80% 1/25% 80/98%

N=375 - NoRes

N=110 - Res

N=375 - Res

Canon-ABCanon-AbCanon-Bmimeticnoconv1doriginal(conv1d)

Canon-ABCanon-AbCanon-Bmimeticnoconv1doriginal(conv1d)

Canon-AB Canon-Ab Canon-B mimetic noconv1doriginal(conv1d)

###### Task Brevo2 - Ablation study - Mamba2

###### Task Lano - Ablation study - Mamba2

###### Task Depo2(K=4, k=4/2) - Ablation study - Mamba2

[Figure 482]

- 94/99% 92/99% 94/100% 29/71% 60/94% 90/98% 97/99% 94/99% 93/99% 88/98% 94/98% 93/99% 19/60% 53/95% 86/97%
- 95/99% 83/96% 90/97% 57/92% 89/98% 69/96% 6/38% 74/96% 67/91% 88/98% 82/96% 90/97%

[Figure 483]

[Figure 484]

88.1% 88.4% 90.2% 41.5% 67.6% 50.4% 88.7% 92.9% 97.1% 68.9% 48.0% 77.1% 3.6% 9.4% 4.8% 52.7% 54.9% 5.2% 41.3% 35.2% 64.1% 1.5% 10.8% 2.7%

91.2% 91.2% 90.2% 74.6% 41.9% 94.1% 93.3% 93.8% 93.3% 76.0% 75.3% 83.9% 33.8% 4.1% 84.2% 86.1% 84.7% 81.1% 64.3% 65.3% 69.1% 24.1% 14.6% 70.3% 68.9% 64.6% 61.7%

N=75 - NoRes

N=30 - NoRes N=30 - Res N=40 - NoRes N=40 - Res N=50 - NoRes N=50 - Res

cfg3f - NoRes cfg3f - Res

N=75 - Res

- cfg3j - NoRes

- cfg3j - Res

cfg3k - NoRes

- cfg3k - Res

N=100 - NoRes

N=100 - Res

N=125 - NoRes

0.6% 52.2% 2.8%

N=125 - Res

Canon-ABCanon-AbCanon-Bmimeticnoconv1doriginal(conv1d)

Canon-ABCanon-AbCanon-Bmimeticnoconv1doriginal(conv1d)

Canon-AB Canon-Ab Canon-B mimetic noconv1doriginal(conv1d)

###### Task Brevo1 - Ablation study - Mamba2(mlp)

###### Task Depo1(K=4, k=4/2) - Ablation study - Mamba2(mlp)

[Figure 485]

76/96% 68/97% 82/97% 43/87% 74/97% 89/97% 61/98% 77/99% 16/80% 6/43% 43/84% 52/95% 87/99% 87/97% 87/99% 87/98% 96/100% 30/90% 53/92% 41/80% 61/87% 77/97% 16/74% 76/95% 60/94% 26/79% 71/97% 7/40% 4/35% 13/61% 31/85% 64/96% 57/92% 34/78% 60/91% 84/97% 18/69% 33/85% 25/76% 9/55% 20/78% 9/59% 19/63% 17/73% 12/60% 48/88% 2/29% 0/10% 10/42% 11/60% 74/93% 49/83% 4/52% 19/73% 66/93% 5/59% 10/78%

[Figure 486]

87.4% 90.6% 78.6% 70.2% 90.4% 55.4% 72.7% 54.2% 32.7% 5.9% 72.4% 14.6% 95.1% 88.8% 92.7% 94.2% 93.9% 0.3% 1.3% 48.1% 83.7% 59.7% 30.8% 65.7% 63.9% 1.4% 1.0% 5.4% 1.0% 4.8%

N=225 - NoRes

N=70 - NoRes N=70 - Res N=90 - NoRes

N=225 - Res

N=300 - NoRes

1.5% 81.7% 40.9% 56.3% 11.5% 61.0% 9.4% 0.4% 27.5% 63.7% 16.5% 35.5% 25.8% 40.1% 0.9% 3.0% 2.0% 0.4% 1.2% 1.2% 63.8% 23.8% 40.6% 2.2% 41.7% 1.9% 7.4%

N=90 - Res N=110 - NoRes

N=300 - Res

N=375 - NoRes

N=110 - Res

N=375 - Res

Canon-ACanon-ABCDCanon-ACDCanon-AbCDCanon-BCanon-BDCanon-CCanon-Dmimeticnoconv1doriginal(conv1d)

Canon-ACanon-ABCD Canon-ACDCanon-AbCD Canon-B Canon-BD Canon-C Canon-D mimetic noconv1doriginal(conv1d)

###### Task Brevo2 - Ablation study - Mamba2(mlp)

###### Task Depo2(K=4, k=4/2) - Ablation study - Mamba2(mlp)

[Figure 487]

93/99% 95/99% 98/100% 89/99% 94/99% 97/100% 83/99% 77/98% 13/74% 2/38% 69/89%

[Figure 488]

94.8% 92.2% 86.8% 72.0% 93.9% 4.7% 5.4% 11.5% 15.4% 63.8% 3.4% 30.2% 91.1% 60.4% 68.1% 90.9% 34.5% 15.8% 14.2% 72.8% 74.1% 26.1% 35.3% 84.1% 5.3% 1.0% 1.3% 0.6% 10.1% 0.5% 21.0% 55.0% 7.3% 10.7% 80.5% 7.1% 2.4% 1.8% 33.6% 55.0% 1.2% 6.7% 8.7% 0.8% 1.1% 0.6% 1.1% 3.8% 0.5%

N=75 - NoRes

N=30 - NoRes N=30 - Res N=40 - NoRes N=40 - Res N=50 - NoRes N=50 - Res

- 85/97% 98/100% 94/99% 87/96% 89/98% 88/98% 89/99% 90/99%
- 86/98% 90/99% 93/99% 51/95% 84/98% 86/98% 66/97% 60/94% 20/77% 7/71% 42/82% 77/95% 90/98% 85/96% 69/94% 58/88% 75/95% 74/98% 67/97% 73/96% 89/98% 84/99% 49/79% 83/97% 87/97% 35/88% 38/93% 10/37% 1/9% 24/75% 65/89% 91/97% 76/97% 67/88% 34/75% 67/89% 38/66% 32/94%

N=75 - Res

N=100 - NoRes

N=100 - Res

N=125 - NoRes

1.2% 32.0% 3.6% 3.0% 5.3% 5.5% 1.2% 1.4%

N=125 - Res

Canon-ACanon-ABCDCanon-ACDCanon-AbCDCanon-BCanon-BDCanon-CCanon-Dmimeticnoconv1doriginal(conv1d)

Canon-ACanon-ABCD Canon-ACDCanon-AbCD Canon-B Canon-BD Canon-C Canon-D mimetic noconv1doriginal(conv1d)

###### Task Mano - Ablation study - Mamba2(mlp)

Task Lano - Ablation study - Mamba2(mlp)

[Figure 489]

[Figure 490]

99.7% 99.2% 99.3% 99.6% 99.8% 99.8% 71.2% 89.2% 44.3% 62.4% 95.7% 84.8% 96.7% 98.5% 99.4% 95.6% 93.8% 83.3% 92.1% 99.4% 99.2% 99.1% 99.0% 99.6% 99.4% 57.3% 92.5% 26.0% 40.8% 91.8% 95.1% 94.2% 94.3% 98.0% 78.1% 88.6% 47.1% 52.1% 99.1% 98.4% 98.6% 98.3% 99.1% 99.0% 38.7% 70.1% 14.3% 26.7% 87.4% 64.2% 77.2% 79.1% 99.0% 54.5% 83.4% 50.0% 60.2%

92.1% 86.9% 89.7% 85.5% 90.4% 90.0% 84.5% 86.9% 64.9% 56.0% 92.2% 88.8% 93.6% 94.5% 93.6% 89.6% 92.3% 89.5% 88.3% 78.7% 65.4% 64.6% 77.8% 79.5% 72.2% 49.5% 61.5% 43.6% 8.3% 74.3% 75.0% 85.0% 83.8% 80.8% 69.4% 75.2% 73.6% 72.5% 57.7% 45.0% 54.1% 44.3% 56.0% 49.4% 37.9% 43.2% 21.9% 13.7% 46.1% 50.3% 63.2% 65.6% 64.4% 46.9% 53.7% 39.3% 44.9%

L=10 - NoRes L=10 - Res L=13 - NoRes L=13 - Res L=16 - NoRes L=16 - Res

cfg3f - NoRes cfg3f - Res

- cfg3j - NoRes

- cfg3j - Res

cfg3k - NoRes

- cfg3k - Res

Canon-ACanon-ABCDCanon-ACDCanon-AbCDCanon-BCanon-BDCanon-CCanon-Dmimeticnoconv1doriginal(conv1d)

Canon-ACanon-ABCDCanon-ACDCanon-AbCDCanon-BCanon-BDCanon-CCanon-Dmimeticnoconv1doriginal(conv1d)

- Figure 30: Ablation study of Mamba2 models of 12L768D size with Canon layers, Canon residuals, original non-linear conv1d, mimetic initialization. Full ablation studies (with additional model sizes, such as the effectiveness of Canon-ACD) are in Figure 31-32.

###### Task Depo1(K=4, k=4/2) Mamba2 - original (conv1d)

###### Task Depo1(K=4, k=4/2) Mamba2 - mimetic

###### Task Depo1(K=4, k=4/2) Mamba2 - noconv1d

###### Task Depo1(K=4, k=4/2) Mamba2 - Canon-AB(no-res)

Task Depo1(K=4, k=4/2) Mamba2 - Canon-AB(res)

###### Task Depo1(K=4, k=4/2) Mamba2 - Canon-Ab(no-res)

###### Task Depo1(K=4, k=4/2) Mamba2 - Canon-Ab(res)

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

20/70% 47/90% 9/76% 20/57% 9/42% 5/41% 3/50% 3/33% 2/16% 12/41% 3/28% 2/33%

30/74% 17/64% 3/36% 14/72% 26/64% 10/52% 3/30% 37/81% 6/30% 14/70% 3/27% 9/37%

14/65% 14/69% 9/60% 23/76% 4/40% 5/41% 6/50% 15/72% 1/16% 5/37% 2/21% 13/62%

16/70% 20/80% 32/83% 16/82% 14/70% 1/38% 14/58% 5/38%

90/99% 94/98% 83/98% 92/98% 79/97% 85/95% 74/91% 71/92% 46/83% 58/91% 42/87% 32/80%

27/70% 31/77% 33/82% 59/95% 0/51% 20/75% 5/48% 14/77% 6/49% 2/55% 4/35% 8/64%

43/85% 16/56% 17/67% 47/90% 17/69% 6/40% 8/47% 21/77% 15/56% 2/15% 2/19% 1/25%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/19% 1/20% 2/25% 3/26%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo1(K=8, k=8/4) Mamba2 - original (conv1d)

###### Task Depo1(K=8, k=8/4) Mamba2 - mimetic

###### Task Depo1(K=8, k=8/4) Mamba2 - noconv1d

###### Task Depo1(K=8, k=8/4) Mamba2 - Canon-AB(no-res)

###### Task Depo1(K=8, k=8/4) Mamba2 - Canon-AB(res)

###### Task Depo1(K=8, k=8/4) Mamba2 - Canon-Ab(no-res)

###### Task Depo1(K=8, k=8/4) Mamba2 - Canon-Ab(res)

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

2/19% 11/50% 1/15% 1/11% 0/5% 1/15% 0/8% 0/4% 0/1% 0/1% 0/1% 0/1%

1/13% 0/7% 1/14% 0/2% 1/10% 0/8% 0/1% 0/2%

0/2% 0/13% 0/8% 0/16% 0/2% 0/5% 0/3% 0/2% 0/0% 0/1% 0/0% 0/1%

0/10% 0/6% 1/15% 1/22% 0/1% 0/3% 0/6% 0/1% 0/1% 0/0% 0/1% 0/6%

43/86% 70/93% 27/84% 60/92% 1/35% 18/70% 2/36% 34/82% 0/19% 6/43% 3/43% 17/72%

0/9% 0/15% 1/14% 0/23% 0/1% 0/4% 0/3% 0/6% 0/1% 0/1% 0/2% 0/1%

1/14% 1/15% 1/14% 2/28%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

- 0/5% 0/3% 0/6% 2/29%
- 0/6% 0/2% 0/2% 0/2%

0/5% 0/5% 0/1% 0/1%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Depo2(K=4, k=4/2) Mamba2 - original (conv1d)

###### Task Depo2(K=4, k=4/2) Mamba2 - mimetic

###### Task Depo2(K=4, k=4/2) Mamba2 - noconv1d

Task Depo2(K=4, k=4/2) Mamba2 - Canon-AB(no-res)

Task Depo2(K=4, k=4/2) Mamba2 - Canon-AB(res)

Task Depo2(K=4, k=4/2) Mamba2 - Canon-Ab(no-res)

Task Depo2(K=4, k=4/2) Mamba2 - Canon-Ab(res)

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

75/95% 91/98% 74/95% 90/98% 55/85% 86/97% 57/90% 86/97% 46/83% 69/91% 41/81% 67/91%

22/74% 45/86% 14/46% 29/71% 4/22% 48/83% 1/20% 19/60% 2/13% 29/75% 19/55% 6/38%

5/47% 16/76% 78/98% 60/94%

89/99% 88/99% 92/99% 94/99% 86/98% 73/97% 82/97% 88/98% 74/97% 49/78% 77/98% 57/92%

94/99% 97/99% 88/99% 97/99% 89/98% 96/99% 87/98% 95/99% 79/93% 92/98% 76/94% 88/98%

81/96% 91/99% 89/98% 92/99% 72/97% 74/97% 72/95% 94/98% 58/85% 31/81% 68/95% 89/98%

93/99% 92/99% 89/98% 94/99% 75/96% 78/96% 85/97% 83/96% 64/91% 60/90% 64/91% 82/96%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 2/19% 13/82% 12/61% 53/95%
- 3/24% 4/47% 10/52% 74/96%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) Mamba2 - original (conv1d)

###### Task Depo2(K=16, k=16/8) Mamba2 - mimetic

###### Task Depo2(K=16, k=16/8) Mamba2 - noconv1d

###### Task Depo2(K=16, k=16/8) Mamba2 - Canon-AB(no-res)

###### Task Depo2(K=16, k=16/8) Mamba2 - Canon-AB(res)

###### Task Depo2(K=16, k=16/8) Mamba2 - Canon-Ab(no-res)

Task Depo2(K=16, k=16/8) Mamba2 - Canon-Ab(res)

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

1/2% 31/72% 1/1% 1/1% 1/1% 1/1% 1/2% 1/4% 1/1% 1/13% 1/1% 1/1%

1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/19% 1/1% 1/12% 33/81% 1/2% 1/1% 1/3% 8/64% 1/1% 1/1% 1/18% 3/17%

1/6% 1/13% 2/18% 5/48% 1/4% 3/35% 1/10% 19/65% 1/1% 1/2% 1/4% 3/23%

1/1% 1/1% 1/2% 1/9%

3/41% 1/3% 2/2% 1/12% 1/15% 1/1% 1/1% 1/1%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/6% 2/24% 1/5%
- 1/1% 1/7% 1/1% 1/1%

1/1% 1/5% 1/1% 1/2%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Mamba2 - original (conv1d)

###### Task Brevo1 Mamba2 - mimetic

###### Task Brevo1 Mamba2 - noconv1d

###### Task Brevo1 Mamba2 - Canon-AB(no-res)

Task Brevo1 Mamba2 - Canon-AB(res)

###### Task Brevo1 Mamba2 - Canon-Ab(no-res)

###### Task Brevo1 Mamba2 - Canon-Ab(res)

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

17.0% 40.1% 71.4% 56.5% 11.5% 87.2% 48.1% 24.2%

53.5% 64.6% 13.8% 62.7% 9.2% 76.9% 61.2% 30.8% 1.3% 55.5% 13.5% 23.0%

1.7% 1.1% 1.7% 1.2% 0.8% 0.4% 1.0% 1.9% 0.5% 1.5% 0.4% 0.4%

68.4% 90.0% 90.2% 89.5% 68.3% 85.0% 71.2% 85.3% 29.7% 38.0% 46.8% 61.6%

91.2% 95.1% 94.3% 96.2% 66.7% 90.5% 86.1% 92.2% 33.7% 71.8% 74.2% 84.5%

55.7% 87.4% 75.3% 89.7% 25.6% 72.9% 62.0% 70.0% 11.4% 34.4% 55.4% 10.2%

77.2% 70.6% 91.0% 92.1% 2.7% 85.5% 63.7% 69.3% 0.1% 52.9% 12.9% 35.4%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

2.9% 8.5% 10.8% 26.2%

N=110

N=110

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 Mamba2 - original (conv1d)

###### Task Brevo2 Mamba2 - mimetic

###### Task Brevo2 Mamba2 - noconv1d

Task Brevo2 Mamba2 - Canon-AB(no-res)

###### Task Brevo2 Mamba2 - Canon-AB(res)

###### Task Brevo2 Mamba2 - Canon-Ab(no-res)

###### Task Brevo2 Mamba2 - Canon-Ab(res)

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

93.5% 92.8% 86.3% 50.4%

0.8% 1.7% 8.5% 41.5% 0.6% 0.8% 6.0% 3.6% 0.4% 1.3% 2.4% 1.5%

29.5% 28.4% 54.9% 67.6% 6.4% 4.9% 34.8% 9.4% 4.6% 6.8% 5.7% 10.8%

83.3% 91.3% 92.2% 88.1% 74.8% 85.7% 64.3% 68.9% 31.5% 14.6% 13.2% 41.3%

90.1% 81.5% 90.5% 88.7% 48.1% 38.0% 52.0% 52.7% 21.7% 20.5% 48.2% 0.6%

40.0% 89.4% 89.7% 88.4%

22.4% 37.5% 96.4% 92.9% 1.3% 0.8% 51.7% 54.9% 0.2% 0.4% 64.2% 52.2%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

66.6% 80.9% 46.4% 4.8% 17.8% 0.6% 22.3% 2.7%

1.5% 74.8% 24.1% 48.0% 23.0% 21.7% 5.8% 35.2%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Mano Mamba2 - original (conv1d)

###### Task Mano Mamba2 - mimetic

###### Task Mano Mamba2 - noconv1d

Task Mano Mamba2 - Canon-AB(no-res)

Task Mano Mamba2 - Canon-AB(res)

Task Mano Mamba2 - Canon-Ab(no-res)

Task Mano Mamba2 - Canon-Ab(res)

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

91.4% 97.1% 96.3% 98.8% 81.5% 95.3% 87.9% 88.0% 52.8% 64.5% 69.9% 94.0%

27.6% 28.5% 32.5% 60.7% 24.4% 17.0% 24.9% 14.5% 12.4% 14.2% 10.6% 11.9%

68.6% 65.0% 62.1% 70.5% 55.8% 49.1% 35.6% 39.1% 26.8% 35.8% 29.2% 33.1%

99.4% 99.6% 99.4% 99.7% 99.1% 99.5% 99.1% 99.6% 98.4% 99.0% 97.6% 99.2%

97.4% 95.9% 92.6% 92.0% 76.1% 74.0% 79.5% 77.6% 52.6% 80.5% 50.8% 73.2%

99.4% 99.8% 99.7% 99.8% 99.1% 99.5% 99.1% 99.5% 98.9% 98.1% 99.1% 99.0%

99.7% 98.4% 99.6% 99.9% 99.5% 99.7% 94.1% 99.7% 97.3% 98.1% 77.1% 98.9%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Mamba2 - original (conv1d)

###### Task Lano Mamba2 - mimetic

###### Task Lano Mamba2 - noconv1d

Task Lano Mamba2 - Canon-AB(no-res)

Task Lano Mamba2 - Canon-AB(res)

Task Lano Mamba2 - Canon-Ab(no-res)

Task Lano Mamba2 - Canon-Ab(res)

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

88.2% 93.5% 91.1% 94.1% 69.1% 82.2% 78.2% 84.2% 43.3% 65.3% 50.9% 70.3%

52.2% 82.6% 39.5% 74.6%

37.1% 31.9% 17.5% 41.9% 4.4% 2.7% 1.4% 4.1% 7.9% 17.2% 7.6% 14.6%

87.7% 91.5% 89.8% 91.2% 65.7% 80.0% 68.2% 76.0% 46.4% 62.2% 52.7% 64.3%

87.6% 92.2% 90.8% 93.3% 67.9% 87.1% 66.9% 86.1% 53.2% 66.8% 54.0% 68.9%

87.4% 90.7% 84.2% 91.2% 70.7% 76.8% 68.9% 75.3% 47.4% 60.7% 43.4% 65.3%

89.9% 92.2% 90.3% 93.8% 73.7% 86.4% 77.0% 84.7% 51.8% 70.1% 56.0% 64.6%

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

4.7% 52.6% 18.8% 33.8% 19.5% 29.2% 22.9% 24.1%

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Mamba2 - original (conv1d)

###### Task Lano Mamba2 - mimetic

###### Task Lano Mamba2 - noconv1d

Task Lano Mamba2 - Canon-AB(no-res)

Task Lano Mamba2 - Canon-AB(res)

###### Task Lano Mamba2 - Canon-Ab(no-res)

Task Lano Mamba2 - Canon-Ab(res)

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

0.00113 0.00075 0.00090 0.00062 0.00163 0.00090 0.00111 0.00084 0.00420 0.00215 0.00338 0.00185

0.00483 0.00182 0.00671 0.00262 0.00979 0.00265 0.00600 0.00416 0.00750 0.00563 0.00654 0.00662

0.00694 0.00790 0.01156 0.00626 0.00986 0.01099 0.01291 0.00992 0.01105 0.00781 0.01102 0.00848

0.00132 0.00090 0.00105 0.00098 0.00180 0.00100 0.00162 0.00128 0.00366 0.00233 0.00294 0.00221

0.00122 0.00082 0.00096 0.00067 0.00169 0.00073 0.00169 0.00078 0.00294 0.00202 0.00303 0.00200

0.00126 0.00100 0.00158 0.00093 0.00155 0.00123 0.00169 0.00131 0.00347 0.00258 0.00390 0.00213

0.00106 0.00082 0.00100 0.00067 0.00139 0.00074 0.00117 0.00081 0.00310 0.00189 0.00278 0.00213

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Figure 31: Mamba2 variants (left to right): original (conv1d), mimetic (w/ conv1d), no conv1d, Canon-AB(nores), Canon-AB(res), Canon-Ab(no-res), Canon-Ab(res).

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - original (conv1d)

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - mimetic

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - noconv1d

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - Canon-ABCD(no-res)

Task Depo1(K=4, k=4/2) Mamba2(mlp) - Canon-ABCD(res)

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - Canon-AbCD(no-res)

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - Canon-AbCD(res)

###### Task Depo1(K=4, k=4/2) Mamba2(mlp) - Canon-ACD(res)

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

12/65% 29/67% 24/74% 43/84% 4/35% 12/46% 13/62% 13/61% 1/22% 13/56% 7/33% 10/42%

5/34% 16/55% 6/50% 16/80% 3/31% 12/47% 4/28% 7/40% 1/18% 18/55% 5/32% 2/29%

2/40% 11/71% 1/24% 6/43% 1/27% 9/46% 0/15% 4/35%

37/88% 55/89% 27/86% 68/97% 21/68% 19/78% 17/64% 61/87% 5/43% 17/58% 4/29% 9/55%

88/97% 92/98% 96/98% 87/99% 65/96% 84/95% 69/94% 64/96% 41/82% 59/90% 62/96% 74/93%

49/90% 51/86% 52/92% 43/87% 8/62% 18/58% 14/70% 16/74% 6/42% 5/39% 13/69% 9/59%

51/87% 21/80% 67/96% 87/99% 11/45% 19/61% 15/63% 34/78% 8/46% 16/75% 6/53% 4/52%

43/90% 55/91% 66/92% 87/97% 22/77% 25/69% 50/87% 57/92% 16/68% 62/92% 21/72% 49/83%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/7% 1/12% 0/10% 0/10%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - original (conv1d)

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - mimetic

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - noconv1d

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - Canon-ABCD(no-res)

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - Canon-AbCD(no-res)

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - Canon-AbCD(res)

###### Task Depo1(K=8, k=8/4) Mamba2(mlp) - Canon-ACD(res)

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

0/9% 2/28% 1/21% 25/71% 0/3% 1/14% 0/7% 5/40% 0/0% 0/4% 0/2% 0/8%

1/1% 3/34% 0/1% 1/19% 0/1% 0/1% 0/0% 0/7% 0/0% 0/0% 0/0% 0/2%

0/0% 1/2% 0/0% 0/1% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0%

1/29% 1/21% 0/7% 18/66% 0/6% 0/9% 0/7% 0/9% 0/1% 0/1% 0/1% 0/6%

- 25/76% 65/93% 42/87% 52/91%
- 26/77% 40/75% 40/84% 34/81% 1/23% 5/42% 5/49% 31/78%

1/10% 1/21% 1/21% 1/21% 0/2% 0/4% 0/4% 0/9% 0/2% 0/2% 0/2% 0/2%

9/50% 12/49% 6/37% 28/76% 3/22% 1/15% 1/16% 21/56% 0/2% 0/1% 0/5% 2/25%

10/58% 14/57% 34/81% 38/82%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

2/28% 3/27% 26/81% 15/58% 0/7% 1/17% 10/58% 2/49%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=4, k=4/2) Mamba2(mlp) - original (conv1d)

###### Task Depo2(K=4, k=4/2) Mamba2(mlp) - mimetic

###### Task Depo2(K=4, k=4/2) Mamba2(mlp) - noconv1d

Task Depo2(K=4, k=4/2) Mamba2(mlp) - Canon-ABCD(no-res)

Task Depo2(K=4, k=4/2) Mamba2(mlp) - Canon-ABCD(res)

Task Depo2(K=4, k=4/2) Mamba2(mlp) - Canon-AbCD(no-res)

Task Depo2(K=4, k=4/2) Mamba2(mlp) - Canon-AbCD(res)

Task Depo2(K=4, k=4/2) Mamba2(mlp) - Canon-ACD(res)

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

18/61% 80/95% 30/86% 69/89% 17/63% 47/83% 17/55% 42/82%

- 6/31% 51/82% 1/9% 13/74% 4/30% 20/73% 9/38% 20/77%
- 7/37% 29/72% 25/53% 10/37%

6/37% 20/63% 2/60% 2/38% 1/3% 1/12% 2/37% 7/71% 1/2% 4/44% 1/7% 1/9%

85/97% 93/99% 93/99% 95/99% 80/97% 88/97% 82/99% 90/99% 46/86% 41/92% 63/90% 89/98%

94/98% 93/99% 91/97% 98/100% 74/96% 75/94% 79/97% 90/98% 62/95% 76/92% 80/94% 91/97%

84/97% 87/98% 91/99% 89/99% 68/97% 79/94% 82/97% 51/95% 40/90% 57/90% 54/92% 49/79%

69/93% 93/98% 87/97% 87/96% 44/83% 84/95% 66/91% 69/94% 23/52% 77/93% 24/63% 67/88%

87/98% 93/98% 84/97% 94/99% 69/94% 81/96% 68/93% 85/96% 57/90% 76/98% 61/91% 76/97%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

5/39% 46/85% 10/41% 24/75%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) Mamba2(mlp) - original (conv1d)

###### Task Depo2(K=16, k=16/8) Mamba2(mlp) - mimetic

###### Task Depo2(K=16, k=16/8) Mamba2(mlp) - noconv1d

###### Task Depo2(K=16, k=16/8) Mamba2(mlp) - Canon-ABCD(no-res)

###### Task Depo2(K=16, k=16/8) Mamba2(mlp) - Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) Mamba2(mlp) - Canon-AbCD(no-res)

###### Task Depo2(K=16, k=16/8) Mamba2(mlp) - Canon-AbCD(res)

Task Depo2(K=16, k=16/8) Mamba2(mlp) - Canon-ACD(res)

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

1/1% 1/1% 1/1% 1/1% 1/1% 1/7% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/3% 2/24% 1/13% 12/57% 1/1% 1/2% 1/1% 1/4% 1/1% 1/1% 1/3% 1/3%

1/1% 1/2% 1/6% 2/20% 1/3% 1/3% 1/1% 6/50% 1/1% 1/5% 1/2% 3/28%

- 1/1% 1/1% 1/4% 2/1%
- 1/1% 1/2% 1/3% 1/1%
- 1/1% 1/3% 1/4% 1/1%

2/1% 1/19% 2/15% 1/1%

2/5% 1/4% 1/8% 1/3% 1/1% 1/1% 1/3% 1/10% 1/1% 1/1% 1/1% 1/1%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/15% 1/2% 1/2%
- 1/1% 2/21% 1/1% 1/1%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 Mamba2(mlp) - original (conv1d)

###### Task Brevo1 Mamba2(mlp) - mimetic

###### Task Brevo1 Mamba2(mlp) - noconv1d

###### Task Brevo1 Mamba2(mlp) - Canon-ABCD(no-res)

Task Brevo1 Mamba2(mlp) - Canon-ABCD(res)

###### Task Brevo1 Mamba2(mlp) - Canon-AbCD(no-res)

###### Task Brevo1 Mamba2(mlp) - Canon-AbCD(res)

###### Task Brevo1 Mamba2(mlp) - Canon-ACD(res)

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

3.7% 80.1% 50.1% 72.4% 0.3% 0.5% 3.8% 4.8% 0.1% 0.0% 1.1% 1.2%

4.8% 10.0% 1.0% 32.7%

1.0% 23.5% 0.5% 5.9% 0.9% 0.5% 0.6% 1.0% 0.5% 0.4% 0.8% 0.4%

83.7% 79.8% 88.0% 90.6% 15.2% 74.4% 31.6% 83.7%

89.9% 94.8% 91.7% 95.1% 59.2% 89.3% 53.3% 81.7% 50.0% 51.1% 26.8% 63.8%

51.3% 69.9% 57.6% 70.2% 10.7% 37.6% 14.9% 30.8%

13.8% 78.7% 32.4% 92.7% 0.6% 17.2% 21.8% 56.3% 0.6% 38.4% 3.6% 40.6%

83.6% 85.6% 88.1% 88.8% 8.6% 82.8% 13.5% 40.9% 0.1% 17.5% 10.2% 23.8%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

- 2.2% 2.5% 4.0% 5.4%
- 3.3% 3.3% 1.5% 2.0%

8.5% 9.1% 31.6% 63.7%

0.3% 8.8% 21.3% 35.5%

N=110

N=110

N=110

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 Mamba2(mlp) - original (conv1d)

###### Task Brevo2 Mamba2(mlp) - mimetic

###### Task Brevo2 Mamba2(mlp) - noconv1d

###### Task Brevo2 Mamba2(mlp) - Canon-ABCD(no-res)

###### Task Brevo2 Mamba2(mlp) - Canon-ABCD(res)

###### Task Brevo2 Mamba2(mlp) - Canon-AbCD(no-res)

###### Task Brevo2 Mamba2(mlp) - Canon-AbCD(res)

###### Task Brevo2 Mamba2(mlp) - Canon-ACD(res)

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

50.8% 95.6% 68.1% 3.4% 12.5% 67.0% 14.5% 0.5%

11.0% 7.7% 9.0% 15.4% 0.8% 3.0% 6.6% 0.6% 0.5% 0.3% 1.1% 1.1%

20.8% 46.6% 14.1% 63.8% 1.8% 36.7% 3.5% 10.1% 1.1% 1.9% 1.2% 3.8%

81.5% 80.3% 68.0% 92.2% 40.7% 41.0% 26.8% 74.1% 24.0% 27.1% 9.5% 55.0%

71.3% 83.7% 65.9% 91.1% 39.5% 23.2% 24.4% 55.0% 14.2% 3.7% 24.4% 32.0%

78.9% 90.1% 62.7% 72.0% 58.0% 77.9% 32.6% 35.3% 8.8% 55.5% 4.1% 6.7%

53.8% 87.1% 65.0% 68.1% 14.7% 27.5% 19.3% 10.7% 0.8% 20.9% 0.9% 3.0%

9.6% 66.0% 31.3% 60.4% 0.6% 0.8% 1.6% 7.3% 0.3% 1.6% 3.2% 3.6%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

3.3% 12.4% 4.0% 0.5%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Mano Mamba2(mlp) - original (conv1d)

###### Task Mano Mamba2(mlp) - mimetic

###### Task Mano Mamba2(mlp) - noconv1d

Task Mano Mamba2(mlp) - Canon-ABCD(no-res)

Task Mano Mamba2(mlp) - Canon-ABCD(res)

Task Mano Mamba2(mlp) - Canon-AbCD(no-res)

Task Mano Mamba2(mlp) - Canon-AbCD(res)

Task Mano Mamba2(mlp) - Canon-ACD(res)

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

96.5% 95.1% 95.2% 95.7% 79.9% 84.8% 88.0% 91.8% 74.4% 90.1% 72.3% 87.4%

38.8% 42.5% 55.6% 44.3% 18.2% 48.7% 21.8% 26.0% 14.5% 22.4% 16.3% 14.3%

68.0% 60.3% 70.8% 62.4% 35.9% 40.3% 49.8% 40.8% 12.7% 28.7% 21.9% 26.7%

99.1% 99.1% 99.0% 99.2% 98.0% 96.5% 98.0% 99.2% 95.3% 97.7% 97.6% 98.4%

98.6% 90.2% 99.1% 96.7% 93.6% 96.8% 91.9% 94.2% 95.6% 73.6% 78.3% 77.2%

99.1% 99.0% 99.3% 99.6% 98.3% 99.1% 98.5% 99.0% 96.8% 98.0% 95.8% 98.3%

- 97.8% 99.5% 99.6% 99.4%
- 98.6% 86.6% 98.7% 98.0% 96.1% 83.1% 96.6% 99.0%

93.8% 91.9% 96.0% 98.5% 73.1% 77.8% 81.4% 94.3% 56.0% 64.7% 53.0% 79.1%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Lano Mamba2(mlp) - original (conv1d)

###### Task Lano Mamba2(mlp) - mimetic

###### Task Lano Mamba2(mlp) - noconv1d

###### Task Lano Mamba2(mlp) - Canon-ABCD(no-res)

Task Lano Mamba2(mlp) - Canon-ABCD(res)

###### Task Lano Mamba2(mlp) - Canon-AbCD(no-res)

Task Lano Mamba2(mlp) - Canon-AbCD(res)

Task Lano Mamba2(mlp) - Canon-ACD(res)

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

83.8% 92.2% 86.8% 92.2% 45.5% 72.0% 54.2% 74.3% 32.7% 50.0% 35.3% 46.1%

41.6% 77.8% 52.3% 64.9%

41.2% 53.4% 29.8% 56.0% 3.8% 8.9% 6.9% 8.3% 10.4% 13.4% 9.3% 13.7%

79.3% 85.7% 79.5% 86.9% 39.2% 68.4% 46.4% 65.4% 30.9% 47.9% 33.0% 45.0%

89.8% 93.7% 92.9% 93.6% 70.6% 76.7% 70.8% 85.0% 48.4% 64.8% 40.6% 63.2%

80.8% 85.4% 79.4% 85.5% 43.3% 63.0% 48.6% 77.8% 34.0% 46.5% 38.1% 44.3%

88.3% 93.1% 90.1% 93.6% 71.3% 80.0% 72.1% 80.8% 44.1% 61.1% 49.3% 64.4%

89.2% 93.1% 88.9% 94.5% 56.1% 81.6% 70.9% 83.8% 42.7% 64.9% 46.0% 65.6%

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- 14.8% 29.5% 25.6% 43.6%
- 15.5% 17.7% 21.9% 21.9%

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Lano Mamba2(mlp) - original (conv1d)

###### Task Lano Mamba2(mlp) - mimetic

###### Task Lano Mamba2(mlp) - noconv1d

###### Task Lano Mamba2(mlp) - Canon-ABCD(no-res)

Task Lano Mamba2(mlp) - Canon-ABCD(res)

###### Task Lano Mamba2(mlp) - Canon-AbCD(no-res)

Task Lano Mamba2(mlp) - Canon-AbCD(res)

Task Lano Mamba2(mlp) - Canon-ACD(res)

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

0.00151 0.00088 0.00133 0.00084 0.00308 0.00144 0.00247 0.00138 0.00509 0.00332 0.00466 0.00366

0.00628 0.00223 0.00490 0.00327 0.00690 0.00463 0.00497 0.00329 0.00838 0.00791 0.00686 0.00681

0.00627 0.00459 0.00824 0.00413 0.01066 0.00803 0.00855 0.00822 0.01019 0.00904 0.01067 0.00913

0.00207 0.00144 0.00198 0.00136 0.00362 0.00167 0.00303 0.00186 0.00537 0.00346 0.00512 0.00376

0.00103 0.00072 0.00076 0.00066 0.00150 0.00119 0.00151 0.00079 0.00336 0.00220 0.00417 0.00230

0.00181 0.00144 0.00199 0.00140 0.00326 0.00197 0.00284 0.00116 0.00497 0.00357 0.00441 0.00380

0.00120 0.00074 0.00102 0.00071 0.00147 0.00108 0.00146 0.00102 0.00380 0.00244 0.00332 0.00223

0.00110 0.00073 0.00110 0.00066 0.00232 0.00096 0.00150 0.00088 0.00389 0.00215 0.00373 0.00212

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

- Figure 32: Mamba2(mlp) variants (left to right): original (conv1d), mimetic (w/ conv1d), no conv1d, CanonABCD(no-res), Canon-ABCD(res), Canon-AbCD(no-res), Canon-AbCD(res), Canon-ACD(res).

##### G.4 GLA family

###### Task Brevo1 - Ablation study - GLA

###### Task Depo1(K=4, k=4/2) - Ablation study - GLA

[Figure 626]

11/47% 12/72% 5/42% 41/89% 1/12% 1/13% 25/65% 79/95% 44/82% 78/97% 26/68% 63/93% 61/77% 72/97% 19/62% 4/26% 13/61% 1/18% 7/51% 21/78% 62/92% 62/96% 50/96% 29/74% 32/72% 31/92% 1/27% 5/39% 1/20% 32/81% 0/6% 0/6% 6/29% 35/75% 22/69% 49/87% 1/11% 20/75% 37/82% 27/86% 4/31% 3/20% 1/16% 0/2% 1/18% 12/45% 7/55% 14/78% 39/83% 5/39% 12/64% 16/69%

42.9% 31.0% 2.6% 59.1% 28.7% 1.4% 75.1% 95.2% 89.2% 95.5% 95.5% 75.0% 89.2% 96.5% 42.2% 92.7% 4.4% 2.4% 81.0% 95.9% 96.9% 97.6% 97.4% 96.2% 42.6% 66.5% 33.5% 5.8% 0.7% 62.7% 0.3% 0.7% 23.5% 83.6% 85.0% 87.3% 75.3% 14.5% 52.2% 78.0% 11.9% 49.8% 51.2% 0.9% 63.3% 86.3% 89.4% 95.0% 93.2% 90.3% 42.5% 42.5%

[Figure 627]

N=225 - NoRes N=225 - Res N=300 - NoRes N=300 - Res N=375 - NoRes N=375 - Res

N=70 - NoRes N=70 - Res N=90 - NoRes

N=90 - Res N=110 - NoRes

6.8% 1.7% 0.1% 43.2% 0.7% 0.6% 4.4% 82.1% 56.5% 71.2% 29.8% 15.0% 14.0% 16.2% 15.2% 19.0% 3.2% 0.2% 25.7% 37.7% 89.3% 86.9% 81.9% 67.5% 31.5% 36.9%

- 0/14% 2/31% 0/9% 15/53% 0/3% 0/4% 1/7% 7/59% 11/48% 15/63% 1/7% 14/56% 13/68% 15/77% 1/16%
- 1/10% 1/10% 1/2% 0/8% 3/26% 10/48% 35/83% 6/55% 3/33% 2/25% 4/24%

N=110 - Res

Elu+Canon-ABCDElu+Canon-ACDElu+Canon-AbCDElu+Canon-BElu+conv1dElu+originalId+Canon-AId+Canon-ABCDId+Canon-ACDId+Canon-AbCDId+Canon-BId+Canon-CId+Canon-DId+conv1dId+original

Elu+Canon-ABCDElu+Canon-ACDElu+Canon-AbCDElu+Canon-BElu+conv1dElu+originalId+Canon-AId+Canon-ABCDId+Canon-ACDId+Canon-AbCDId+Canon-BId+Canon-CId+Canon-DId+conv1dId+original

###### Task Brevo2 - Ablation study - GLA

###### Task Depo2(K=4, k=4/2) - Ablation study - GLA

[Figure 628]

74/95% 68/93% 15/47% 22/46% 4/21% 4/27% 71/89% 91/98% 69/91% 93/99% 78/96% 43/75% 62/90% 53/85% 3/12% 63/85% 54/84% 45/72% 11/36% 38/64% 79/93% 85/95% 89/98% 45/81% 4/18% 38/72% 58/90% 46/85% 4/23% 26/54% 3/25% 1/6% 28/61% 79/96% 59/89% 84/96% 38/81% 16/46% 35/80% 39/82% 1/6% 47/81% 42/72% 22/50% 9/30% 19/58% 67/90% 34/69% 73/92% 33/69% 3/21% 16/63% 50/85% 35/75% 2/14% 5/19% 1/6% 1/2% 13/30% 65/92% 43/82% 49/88% 33/74% 5/41% 13/43% 20/51% 4/18% 30/71% 11/36% 16/38% 4/13% 22/62% 50/82% 27/61% 69/91% 25/46% 3/16% 5/34%

80.5% 27.5% 60.8% 94.2% 29.8% 73.3% 94.3% 92.1% 87.3% 96.5% 97.2% 72.0% 68.9% 94.4% 33.2% 71.5% 44.3% 76.5% 92.1% 93.5% 93.3% 80.6% 97.3% 95.0% 64.0% 95.8% 29.0% 10.8% 4.5% 78.1% 1.3% 18.8% 83.0% 78.0% 40.1% 82.8% 90.2% 21.2% 14.1% 79.5% 8.8% 36.0% 5.2% 29.0% 17.0% 83.0% 86.9% 16.7% 88.1% 84.3% 23.6% 48.5%

[Figure 629]

N=75 - NoRes

N=30 - NoRes N=30 - Res N=40 - NoRes N=40 - Res N=50 - NoRes N=50 - Res

N=75 - Res N=100 - NoRes

N=100 - Res N=125 - NoRes N=125 - Res

6.1% 1.8% 2.5% 64.0% 0.9% 5.2% 61.6% 34.0% 14.6% 67.1% 70.0% 1.9% 7.8% 55.7% 1.6% 13.0% 0.8% 1.9% 16.6% 55.3% 12.3% 6.2% 68.0% 21.1% 2.9% 3.8%

Elu+Canon-ABCDElu+Canon-ACDElu+Canon-AbCDElu+Canon-BElu+conv1dElu+originalId+Canon-AId+Canon-ABCDId+Canon-ACDId+Canon-AbCDId+Canon-BId+Canon-CId+Canon-DId+conv1dId+original

Elu+Canon-ABCDElu+Canon-ACDElu+Canon-AbCDElu+Canon-BElu+conv1dElu+originalId+Canon-AId+Canon-ABCDId+Canon-ACDId+Canon-AbCDId+Canon-BId+Canon-CId+Canon-DId+conv1dId+original

###### Task Mano - Ablation study - GLA

Task Lano - Ablation study - GLA

99.4% 99.1% 99.4% 69.3% 91.1% 50.5% 94.1% 99.1% 99.2% 99.3% 95.4% 92.5% 88.5% 97.8% 56.1% 99.6% 99.1% 99.1% 61.9% 74.7% 99.5% 98.0% 97.8% 60.3% 86.5% 88.4% 98.1% 98.9% 97.3% 44.3% 76.0% 20.4% 85.1% 99.3% 98.0% 98.7% 73.8% 85.8% 89.7% 90.2% 44.4%

86.1% 84.9% 86.9% 91.6% 90.6% 79.5% 91.8% 90.3% 87.2% 89.7% 89.9% 86.2% 89.7% 90.6% 54.9% 92.8% 92.5% 93.1% 89.1% 87.1% 93.9% 93.3% 94.1% 85.3% 91.8% 92.5% 67.9% 59.0% 68.6% 75.1% 69.3% 32.7% 75.1% 75.0% 78.6% 78.6% 82.9% 66.3% 74.5% 79.3% 35.7% 81.9% 82.0% 77.3% 59.6% 76.0% 87.9% 86.0% 84.3% 73.8% 73.2% 80.2% 43.8% 33.7% 46.7% 52.8% 44.8% 21.4% 55.4% 62.4% 59.8% 59.7% 54.6% 53.9% 58.4% 62.6% 19.8% 63.9% 60.1% 57.0% 54.1% 51.6% 64.6% 70.4% 73.3% 37.3% 60.5% 53.9%

[Figure 630]

[Figure 631]

L=10 - NoRes L=10 - Res L=13 - NoRes L=13 - Res L=16 - NoRes L=16 - Res

cfg3f - NoRes cfg3f - Res

- cfg3j - NoRes

- cfg3j - Res

cfg3k - NoRes

- cfg3k - Res

- 96.8% 93.0% 96.4% 40.5% 61.5% 84.0% 77.8% 84.2% 21.7% 79.9% 58.3%
- 96.9% 97.7% 97.9% 20.4% 42.0% 19.7% 81.1% 98.4% 96.7% 96.0% 48.2% 74.8% 75.9% 62.1% 22.4% 75.7% 93.0% 98.0% 12.9% 25.8% 96.0% 80.5% 65.9% 29.4% 40.4% 38.6%

Elu+Canon-ABCDElu+Canon-ACDElu+Canon-AbCDElu+Canon-BElu+conv1dElu+originalId+Canon-AId+Canon-ABCDId+Canon-ACDId+Canon-AbCDId+Canon-BId+Canon-CId+Canon-DId+conv1dId+original

Elu+Canon-ABCDElu+Canon-ACDElu+Canon-AbCDElu+Canon-BElu+conv1dElu+originalId+Canon-AId+Canon-ABCDId+Canon-ACDId+Canon-AbCDId+Canon-BId+Canon-CId+Canon-DId+conv1dId+original

- Figure 33: Ablation study on 12L768D GLA with Canon/conv1d layers, residual vs. non-residual, identity feature map vs non-linear (ϕ(x) = elu(x) + 1) feature map.

###### Task Depo1(K=4, k=4/2) GLA - original

###### Task Depo1(K=4, k=4/2) GLA - conv1d

###### Task Depo1(K=4, k=4/2) GLA - Canon-ABCD(no-res)

###### Task Depo1(K=4, k=4/2) GLA - Canon-ABCD(res)

###### Task Depo1(K=4, k=4/2) GLA - Canon-AbCD(no-res)

###### Task Depo1(K=4, k=4/2) GLA - Canon-AbCD(res)

###### Task Depo1(K=4, k=4/2) GLA - Canon-ACD(res)

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

7/46% 2/14% 14/55% 19/62% 1/11% 1/9% 1/22% 4/31%

14/70% 37/91% 72/96% 72/97% 2/41% 22/77% 15/80% 27/86% 2/26% 13/72% 11/73% 15/77%

30/58% 12/69% 34/90% 79/95% 12/55% 12/67% 21/71% 35/75% 8/43% 2/23% 21/69% 7/59%

8/76% 24/70% 23/79% 62/92% 9/51% 3/44% 8/54% 7/55% 2/22% 3/24% 5/45% 10/48%

76/96% 62/97% 62/91% 78/97% 27/77% 35/87% 56/91% 49/87%

19/75% 37/91% 22/85% 50/96% 7/49% 15/76% 11/81% 39/83% 5/28% 10/59% 7/56% 6/55%

22/80% 23/67% 36/86% 62/96% 3/31% 2/43% 26/74% 14/78% 3/27% 1/20% 9/45% 35/83%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/4% 0/0% 0/6% 1/16%

6/39% 12/73% 28/72% 15/63%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo1(K=8, k=8/4) GLA - original

###### Task Depo1(K=8, k=8/4) GLA - conv1d

###### Task Depo1(K=8, k=8/4) GLA - Canon-ABCD(no-res)

###### Task Depo1(K=8, k=8/4) GLA - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) GLA - Canon-AbCD(no-res)

###### Task Depo1(K=8, k=8/4) GLA - Canon-AbCD(res)

###### Task Depo1(K=8, k=8/4) GLA - Canon-ACD(res)

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

0/0% 1/1% 0/2% 0/1% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0% 0/0%

1/13% 6/41% 11/64% 35/84% 0/7% 2/20% 1/31% 8/54% 0/9% 0/7% 0/6% 3/30%

4/36% 4/40% 11/51% 54/87% 0/14% 0/2% 2/25% 1/22% 0/6% 0/5% 0/8% 0/8%

0/8% 1/28% 1/23% 12/62% 0/8% 0/6% 3/24% 1/25% 0/0% 0/1% 0/1% 0/5%

5/39% 21/66% 19/69% 30/79% 1/15% 3/41% 1/21% 6/35%

7/53% 8/39% 44/87% 39/86% 0/7% 3/34% 4/43% 7/61% 0/3% 0/2% 1/9% 5/49%

0/10% 2/32% 1/23% 4/42% 0/6% 1/15% 1/9% 1/19% 0/2% 1/12% 0/4% 1/22%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

0/9% 1/13% 1/24% 1/16%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=4, k=4/2) GLA - original

###### Task Depo2(K=4, k=4/2) GLA - conv1d

Task Depo2(K=4, k=4/2) GLA - Canon-ABCD(no-res)

Task Depo2(K=4, k=4/2) GLA - Canon-ABCD(res)

Task Depo2(K=4, k=4/2) GLA - Canon-AbCD(no-res)

Task Depo2(K=4, k=4/2) GLA - Canon-AbCD(res)

###### Task Depo2(K=4, k=4/2) GLA - Canon-ACD(res)

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

3/22% 1/5% 5/39% 3/12% 6/34% 1/13% 3/25% 1/6%

28/62% 42/79% 70/91% 53/85% 13/46% 19/65% 26/70% 39/82% 17/39% 22/59% 8/35% 20/51%

58/91% 80/95% 88/98% 91/98% 40/75% 49/81% 76/94% 79/96% 33/69% 47/84% 50/88% 65/92%

55/88% 83/98% 74/94% 79/93% 36/66% 42/82% 70/88% 67/90% 20/50% 41/73% 56/88% 50/82%

64/91% 77/95% 75/94% 93/99% 54/87% 35/59% 41/80% 84/96% 14/69% 43/86% 56/87% 49/88%

67/89% 83/96% 77/98% 89/98% 42/77% 46/85% 63/96% 73/92% 35/71% 13/79% 46/93% 69/91%

42/76% 78/94% 74/92% 85/95% 15/49% 28/64% 60/84% 34/69%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

1/1% 1/3% 2/26% 4/18%

5/25% 23/47% 24/54% 27/61%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) GLA - original

###### Task Depo2(K=16, k=16/8) GLA - conv1d

###### Task Depo2(K=16, k=16/8) GLA - Canon-ABCD(no-res)

###### Task Depo2(K=16, k=16/8) GLA - Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) GLA - Canon-AbCD(no-res)

###### Task Depo2(K=16, k=16/8) GLA - Canon-AbCD(res)

Task Depo2(K=16, k=16/8) GLA - Canon-ACD(res)

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/1% 1/1% 2/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/1% 1/6% 1/1% 2/14%

1/1% 1/1% 1/1% 1/8% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

2/1% 1/5% 1/1% 1/14% 1/1% 1/1% 1/1% 2/15% 1/1% 1/1% 1/1% 1/1%

1/1% 1/1% 1/2% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

- 1/1% 1/1% 1/3% 1/8%
- 1/1% 1/2% 1/1% 1/1%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 GLA - original

###### Task Brevo1 GLA - conv1d

Task Brevo1 GLA - Canon-ABCD(no-res)

Task Brevo1 GLA - Canon-ABCD(res)

Task Brevo1 GLA - Canon-AbCD(no-res)

Task Brevo1 GLA - Canon-AbCD(res)

Task Brevo1 GLA - Canon-ACD(res)

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

33.7% 36.5% 46.1% 42.2% 1.7% 2.8% 6.2% 11.9% 1.2% 10.7% 2.9% 15.2%

62.5% 94.8% 69.7% 96.5% 41.4% 74.8% 26.2% 78.0% 38.6% 43.6% 10.5% 16.2%

85.4% 92.4% 90.6% 95.2% 46.9% 84.8% 70.8% 83.6% 10.7% 50.2% 45.0% 82.1%

93.6% 93.0% 96.9% 96.9% 79.1% 88.2% 80.7% 89.4% 47.4% 76.0% 70.2% 89.3%

83.7% 93.0% 87.9% 95.5% 57.6% 89.0% 78.7% 87.3% 38.4% 68.1% 37.4% 71.2%

88.0% 96.3% 95.3% 97.4% 83.3% 90.6% 89.7% 93.2% 65.5% 68.2% 79.4% 81.9%

91.3% 96.6% 97.0% 97.6% 59.6% 90.1% 88.7% 95.0% 52.0% 78.4% 72.4% 86.9%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=110

N=110

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo2 GLA - original

Task Brevo2 GLA - conv1d

###### Task Brevo2 GLA - Canon-ABCD(no-res)

###### Task Brevo2 GLA - Canon-ABCD(res)

Task Brevo2 GLA - Canon-AbCD(no-res)

Task Brevo2 GLA - Canon-AbCD(res)

###### Task Brevo2 GLA - Canon-ACD(res)

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

2.8% 45.5% 21.5% 33.2% 0.7% 1.0% 1.8% 8.8% 0.1% 0.7% 1.0% 1.6%

91.5% 94.9% 95.0% 94.4% 67.5% 71.7% 80.0% 79.5% 31.8% 39.2% 57.6% 55.7%

86.0% 83.3% 90.1% 92.1% 36.5% 63.9% 74.1% 78.0% 27.0% 22.9% 55.4% 34.0%

67.5% 92.7% 93.1% 93.3% 34.4% 49.1% 80.4% 86.9%

90.0% 93.5% 93.6% 96.5% 60.9% 78.2% 81.6% 82.8% 25.2% 20.6% 63.2% 67.1%

91.1% 96.7% 94.8% 97.3% 71.1% 89.6% 78.5% 88.1% 46.8% 67.5% 50.9% 68.0%

69.2% 89.2% 86.1% 80.6% 26.5% 31.2% 47.0% 16.7% 2.1% 3.8% 10.2% 6.2%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

4.3% 1.8% 53.5% 12.3%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Mano GLA - original

Task Mano GLA - conv1d

Task Mano GLA - Canon-ABCD(no-res)

Task Mano GLA - Canon-ABCD(res)

Task Mano GLA - Canon-AbCD(no-res)

Task Mano GLA - Canon-AbCD(res)

Task Mano GLA - Canon-ACD(res)

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

59.8% 56.0% 76.2% 56.1% 36.5% 31.1% 69.5% 44.4% 16.8% 24.7% 24.8% 22.4%

94.3% 95.8% 92.3% 97.8% 81.6% 54.0% 86.4% 90.2% 46.1% 35.3% 35.1% 62.1%

95.1% 95.2% 98.6% 99.1% 97.9% 90.7% 98.0% 99.3% 95.4% 91.1% 95.5% 98.4%

95.8% 99.3% 97.3% 99.5% 93.4% 95.7% 96.8% 84.0% 56.8% 87.5% 85.4% 96.0%

- 96.3% 98.0% 99.0% 99.3%
- 97.4% 90.2% 98.4% 98.7% 87.3% 94.7% 96.3% 96.0%

96.2% 99.4% 79.7% 97.8% 95.5% 95.1% 59.9% 84.2% 84.2% 72.7% 55.1% 65.9%

96.3% 97.8% 96.4% 98.0% 83.1% 91.9% 96.0% 77.8% 69.5% 74.7% 52.7% 80.5%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Lano GLA - original

Task Lano GLA - conv1d

###### Task Lano GLA - Canon-ABCD(no-res)

Task Lano GLA - Canon-ABCD(res)

Task Lano GLA - Canon-AbCD(no-res)

Task Lano GLA - Canon-AbCD(res)

Task Lano GLA - Canon-ACD(res)

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

37.5% 61.1% 41.8% 54.9%

79.0% 89.5% 85.1% 90.6% 50.8% 73.9% 63.5% 79.3% 32.0% 58.7% 35.1% 62.6%

85.0% 88.6% 85.0% 90.3% 58.8% 71.1% 68.0% 75.0% 44.3% 53.5% 50.0% 62.4%

88.9% 92.6% 91.2% 93.9% 73.6% 82.9% 76.9% 87.9% 51.8% 64.8% 58.7% 64.6%

81.4% 87.6% 88.4% 89.7% 54.5% 70.2% 67.9% 78.6% 41.7% 56.0% 51.5% 59.7%

88.5% 92.8% 90.0% 94.1% 68.7% 84.6% 78.1% 84.3% 53.5% 64.9% 54.8% 73.3%

86.2% 92.1% 89.3% 93.3% 73.2% 83.2% 77.1% 86.0% 49.1% 66.8% 55.8% 70.4%

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

2.7% 17.1% 9.1% 35.7% 13.0% 11.6% 12.9% 19.8%

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Lano GLA - original

###### Task Lano GLA - conv1d

###### Task Lano GLA - Canon-ABCD(no-res)

Task Lano GLA - Canon-ABCD(res)

###### Task Lano GLA - Canon-AbCD(no-res)

Task Lano GLA - Canon-AbCD(res)

Task Lano GLA - Canon-ACD(res)

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

0.00702 0.00364 0.00618 0.00436 0.01115 0.00615 0.00792 0.00378 0.00932 0.00958 0.00924 0.00718

0.00202 0.00103 0.00142 0.00095 0.00270 0.00133 0.00190 0.00107 0.00530 0.00269 0.00488 0.00239

0.00154 0.00119 0.00157 0.00103 0.00235 0.00150 0.00169 0.00128 0.00401 0.00314 0.00329 0.00239

0.00113 0.00075 0.00094 0.00064 0.00140 0.00089 0.00118 0.00068 0.00322 0.00225 0.00267 0.00218

0.00179 0.00122 0.00114 0.00101 0.00249 0.00162 0.00169 0.00110 0.00401 0.00279 0.00316 0.00251

0.00115 0.00072 0.00097 0.00064 0.00165 0.00083 0.00111 0.00084 0.00304 0.00218 0.00292 0.00166

0.00138 0.00084 0.00103 0.00070 0.00141 0.00089 0.00117 0.00074 0.00342 0.00200 0.00293 0.00188

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

cfg3f

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

- cfg3j

- cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Figure 34: GLA variants (left to right): original, original + conv1d, original + Canon-ABCD(no-res), CanonABCD(res), Canon-AbCD(no-res), Canon-AbCD(res), Canon-ACD(res).

##### G.5 GDN family

###### Task Depo1(K=8, k=8/4) - Ablation study - GDN

###### Task Depo2(K=16, k=16/8) - Ablation study - GDN

[Figure 695]

[Figure 696]

23/67% 50/91% 40/78% 80/96% 42/79% 1/35% 5/59% 2/26% 26/73% 76/96% 40/86% 32/73% 52/85% 58/89% 34/78% 45/85% 16/59% 62/90% 16/57% 49/84% 1/22% 4/40% 17/62% 0/5% 0/1% 23/67% 41/80% 10/42% 52/83% 16/64% 9/53% 17/74%

- 1/2% 7/60% 28/75% 51/87% 23/73% 19/70% 16/71% 1/1% 50/79%

7/43% 58/90% 2/19% 40/83% 40/78% 14/68% 2/1%

- 1/3% 1/1% 4/38% 18/61% 4/29% 1/1% 1/8% 1/1% 2/30%
- 1/4% 1/28% 1/1% 1/7% 1/1% 5/41% 1/1% 1/1% 1/1% 2/22% 11/54% 1/2% 1/3% 1/1% 1/1% 1/1% 1/9% 2/20% 1/1% 2/22% 1/8% 1/1% 1/1%

N=225 - NoRes

N=75 - NoRes

N=225 - Res

N=75 - Res

N=300 - NoRes

N=100 - NoRes

N=300 - Res

N=100 - Res

6/35% 7/48% 3/23% 14/62% 0/0% 0/3% 0/5% 0/0% 0/0% 12/53% 6/31% 0/7% 15/51% 1/29% 5/37% 2/25%

N=375 - NoRes

N=125 - NoRes

N=375 - Res

N=125 - Res

Canon-ACanon-ABCD Canon-ACDCanon-AbCD Canon-B Canon-C Canon-D noconv1doriginal(conv1d)

Canon-ACanon-ABCD Canon-ACDCanon-AbCD Canon-B Canon-C Canon-D noconv1doriginal(conv1d)

###### Task Brevo1 - Ablation study - GDN

###### Task Depo1(K=4, k=4/2) - Ablation study - GDN

[Figure 697]

85/98% 89/98% 72/97% 85/98% 69/93% 70/96% 50/88% 22/83% 32/92% 63/95% 65/93% 74/94% 91/98% 51/97% 74/94% 72/98% 80/96% 67/93% 62/93% 77/96% 52/83% 21/79% 26/90% 20/72% 57/89% 24/72% 21/64% 42/87% 83/95% 18/67% 60/94% 23/68% 66/92% 65/86% 27/88% 50/85% 32/84% 35/88% 15/65% 1/30% 26/71% 17/51% 30/76% 19/48% 50/83% 9/50% 2/41% 4/48%

[Figure 698]

95.4% 94.7% 97.4% 95.5% 96.0% 96.6% 95.6% 96.2% 96.7% 97.3% 89.8% 98.7% 96.8% 96.6% 95.0% 95.6%

N=225 - NoRes

N=70 - NoRes N=70 - Res N=90 - NoRes

N=225 - Res

- 84.0% 92.0% 94.9% 87.7% 89.8% 92.3% 91.9% 90.6% 87.1% 94.8% 88.4% 96.3% 86.3% 80.3% 85.5% 89.8% 78.6% 74.5% 91.2% 85.0% 82.5% 82.5% 80.6% 81.9% 88.0%
- 85.0% 93.9% 91.5% 88.3% 38.5% 86.6% 82.1%

N=300 - NoRes

N=90 - Res N=110 - NoRes

N=300 - Res

N=375 - NoRes

N=110 - Res

N=375 - Res

Canon-ACanon-ABCDCanon-ACDCanon-AbCDCanon-BCanon-CCanon-Dnoconv1doriginal(conv1d)

Canon-ACanon-ABCD Canon-ACDCanon-AbCD Canon-B Canon-C Canon-D noconv1doriginal(conv1d)

###### Task Brevo2 - Ablation study - GDN

###### Task Depo2(K=4, k=4/2) - Ablation study - GDN

[Figure 699]

98/100% 99/100% 97/100% 98/100% 97/100% 99/100% 91/98% 87/99% 99/100% 98/100% 95/99% 96/99% 96/100% 99/100% 95/99% 87/98%

[Figure 700]

99.5% 96.7% 96.1% 98.1% 98.4% 96.5% 96.8% 97.5% 98.7% 97.8% 97.4% 98.0% 98.6% 98.0% 98.7% 97.8% 98.0% 92.3% 89.6% 94.5% 91.0% 90.9% 83.0% 89.4% 96.8% 97.8% 93.4% 86.1% 96.3% 96.4% 92.7% 90.1% 93.2% 67.2% 68.7% 87.8% 84.0% 86.7% 75.0% 69.1% 89.1% 89.0% 63.4% 76.5% 90.6% 89.7% 85.8% 60.2%

N=75 - NoRes

N=30 - NoRes N=30 - Res N=40 - NoRes N=40 - Res N=50 - NoRes N=50 - Res

N=75 - Res

94/99% 95/99% 94/99% 94/99% 90/98% 82/98% 81/97% 79/96% 96/99%

N=100 - NoRes

- 88/98% 96/99% 91/99% 94/99% 96/99% 84/97% 84/98%
- 89/97% 94/99% 77/95% 88/96% 81/94% 81/95% 39/94% 67/94% 84/97% 82/98% 85/97% 88/98% 89/96% 83/97% 78/96% 70/94%

N=100 - Res

N=125 - NoRes

N=125 - Res

Canon-ACanon-ABCDCanon-ACDCanon-AbCDCanon-BCanon-CCanon-Dnoconv1doriginal(conv1d)

Canon-ACanon-ABCD Canon-ACDCanon-AbCD Canon-B Canon-C Canon-D noconv1doriginal(conv1d)

###### Task Mano - Ablation study - GDN

Task Lano - Ablation study - GDN

[Figure 701]

[Figure 702]

93.5% 99.8% 99.8% 99.8% 97.3% 78.2% 77.9% 72.3% 89.8% 87.4% 96.4% 98.6% 98.1% 86.3% 86.8% 96.3% 95.9% 99.3% 98.2% 99.1% 96.0% 72.9% 76.1% 36.4% 75.0% 86.4% 96.7% 93.4% 95.5% 92.0% 39.7% 57.8% 96.4% 99.3% 97.8% 98.0% 73.0% 33.6% 38.8% 18.9% 52.6% 73.3% 94.0% 72.8% 89.3% 59.0% 40.5% 63.0%

94.4% 95.1% 93.3% 94.6% 95.1% 90.8% 92.2% 69.9% 94.4% 92.5% 95.3% 94.7% 95.1% 91.5% 93.0% 93.1% 83.6% 84.8% 80.4% 85.1% 83.6% 63.5% 63.6% 26.1% 86.5% 80.7% 88.1% 87.5% 87.1% 79.0% 76.5% 83.2% 63.2% 76.7% 65.5% 74.2% 69.8% 40.8% 39.5% 12.9% 65.7% 60.0% 68.8% 67.6% 75.2% 52.0% 46.5% 53.7%

L=10 - NoRes L=10 - Res L=13 - NoRes L=13 - Res L=16 - NoRes L=16 - Res

cfg3f - NoRes cfg3f - Res

- cfg3j - NoRes

- cfg3j - Res

cfg3k - NoRes

- cfg3k - Res

Canon-ACanon-ABCDCanon-ACDCanon-AbCDCanon-BCanon-CCanon-Dnoconv1doriginal(conv1d)

Canon-ACanon-ABCDCanon-ACDCanon-AbCDCanon-BCanon-CCanon-Dnoconv1doriginal(conv1d)

###### Figure 35: Ablation study on 12L768D GDN with Canon/conv1d layers, residual vs. non-residual.

###### Task Depo1(K=4, k=4/2) GDN - noconv1d

###### Task Depo1(K=4, k=4/2) GDN - original (conv1d)

Task Depo1(K=4, k=4/2) GDN - Canon-ABCD(no-res)

###### Task Depo1(K=4, k=4/2) GDN - Canon-ABCD(res)

Task Depo1(K=4, k=4/2) GDN - Canon-AbCD(no-res)

###### Task Depo1(K=4, k=4/2) GDN - Canon-AbCD(res)

###### Task Depo1(K=4, k=4/2) GDN - Canon-ACD(res)

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

11/81% 26/78% 23/75% 22/83% 3/24% 14/73% 7/47% 20/72% 1/19% 8/43% 3/31% 1/30%

85/97% 85/95% 22/86% 32/92% 47/90% 22/85% 61/93% 57/89% 28/78% 32/87% 11/51% 26/71%

68/96% 67/95% 85/98% 89/98% 41/87% 62/97% 69/96% 67/93% 22/71% 12/71% 21/82% 65/86%

38/92% 48/93% 42/81% 65/93% 33/78% 52/90% 56/89% 21/64% 15/61% 14/63% 12/44% 30/76%

87/98% 91/99% 92/99% 85/98% 24/90% 81/97% 93/97% 77/96% 34/83% 57/90% 68/87% 50/85%

74/96% 48/92% 69/94% 91/98% 12/58% 14/75% 17/72% 83/95% 25/73% 47/87% 21/68% 50/83%

24/76% 47/78% 26/76% 74/94% 33/69% 9/56% 23/50% 42/87%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

7/39% 11/52% 23/70% 19/48%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo1(K=8, k=8/4) GDN - noconv1d

###### Task Depo1(K=8, k=8/4) GDN - original (conv1d)

###### Task Depo1(K=8, k=8/4) GDN - Canon-ABCD(no-res)

###### Task Depo1(K=8, k=8/4) GDN - Canon-ABCD(res)

###### Task Depo1(K=8, k=8/4) GDN - Canon-AbCD(no-res)

###### Task Depo1(K=8, k=8/4) GDN - Canon-AbCD(res)

###### Task Depo1(K=8, k=8/4) GDN - Canon-ACD(res)

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

0/2% 1/36% 0/4% 2/26% 0/3% 0/5% 0/5% 0/5% 0/1% 0/0% 0/0% 0/0%

43/87% 47/87% 16/69% 26/73% 20/68% 32/76% 1/3% 0/1% 12/52% 12/50% 0/0% 0/0%

21/70% 43/82% 22/78% 50/91%

1/18% 39/80% 27/71% 40/86% 0/1% 12/45% 2/18% 41/80% 0/1% 8/46% 0/14% 6/31%

49/89% 51/89% 75/95% 80/96% 6/59% 25/74% 49/82% 49/84% 6/45% 8/51% 12/63% 14/62%

26/72% 47/84% 28/71% 52/85% 28/65% 22/66% 14/55% 52/83%

4/30% 31/74% 5/33% 32/73% 1/16% 4/27% 1/15% 10/42% 1/14% 6/37% 0/4% 0/7%

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

N=225 N=300 N=375

2/20% 3/35% 13/65% 62/90% 0/8% 1/18% 6/34% 7/48%

1/7% 2/19% 3/30% 15/51%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Depo2(K=4, k=4/2) GDN - noconv1d

Task Depo2(K=4, k=4/2) GDN - original (conv1d)

Task Depo2(K=4, k=4/2) GDN - Canon-ABCD(no-res)

Task Depo2(K=4, k=4/2) GDN - Canon-ABCD(res)

Task Depo2(K=4, k=4/2) GDN - Canon-AbCD(no-res)

Task Depo2(K=4, k=4/2) GDN - Canon-AbCD(res)

Task Depo2(K=4, k=4/2) GDN - Canon-ACD(res)

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

69/96% 94/99% 77/97% 87/99% 46/89% 75/95% 13/72% 79/96% 11/68% 44/86% 11/84% 67/94%

94/99% 90/99% 96/99% 99/100% 87/98% 79/95% 94/98% 96/99% 69/95% 86/96% 86/97% 84/97%

98/100% 98/99% 95/99% 99/100% 92/98% 95/99% 82/98% 95/99% 88/99% 90/99% 76/96% 94/99%

92/99% 94/99% 94/99% 95/99% 85/97% 94/99% 92/98% 96/99% 80/95% 81/96% 85/98% 85/97%

95/99% 98/99% 95/100% 98/100% 89/98% 94/99% 90/98% 94/99% 76/96% 85/97% 83/96% 88/96%

96/99% 98/100% 95/99% 96/100% 89/99% 95/98% 92/99% 94/99% 66/94% 93/99% 84/96% 89/96%

91/99% 99/100% 86/99% 96/99% 79/96% 95/99% 79/97% 91/99% 62/92% 89/99% 75/93% 88/98%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Depo2(K=16, k=16/8) GDN - noconv1d

###### Task Depo2(K=16, k=16/8) GDN - original (conv1d)

###### Task Depo2(K=16, k=16/8) GDN - Canon-ABCD(no-res)

###### Task Depo2(K=16, k=16/8) GDN - Canon-ABCD(res)

###### Task Depo2(K=16, k=16/8) GDN - Canon-AbCD(no-res)

###### Task Depo2(K=16, k=16/8) GDN - Canon-AbCD(res)

Task Depo2(K=16, k=16/8) GDN - Canon-ACD(res)

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1% 1/1%

1/6% 19/61% 2/18% 50/79% 1/5% 1/3% 2/23% 2/30% 1/2% 1/1% 1/2% 1/1%

1/3% 1/1% 1/6% 7/60% 1/9% 1/2% 1/2% 1/1% 1/1% 1/3% 1/1% 1/1%

3/31% 49/87% 10/57% 58/90% 1/8% 4/32% 1/10% 1/28% 1/7% 1/1% 1/3% 2/20%

13/57% 37/82% 9/63% 51/87% 1/2% 1/1% 1/1% 18/61% 1/1% 3/21% 1/5% 11/54%

1/5% 30/73% 5/40% 40/83% 1/2% 1/3% 1/6% 1/7% 1/3% 1/1% 1/7% 2/22%

1/1% 9/59% 2/14% 2/19% 1/1% 1/7% 1/1% 1/1% 1/1% 1/4% 1/1% 1/1%

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

N=75 N=100 N=125

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Brevo1 GDN - noconv1d

Task Brevo1 GDN - original (conv1d)

Task Brevo1 GDN - Canon-ABCD(no-res)

Task Brevo1 GDN - Canon-ABCD(res)

Task Brevo1 GDN - Canon-AbCD(no-res)

Task Brevo1 GDN - Canon-AbCD(res)

Task Brevo1 GDN - Canon-ACD(res)

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

87.8% 90.8% 90.2% 96.2% 64.1% 75.8% 82.6% 90.6%

92.5% 94.9% 96.2% 96.7% 78.2% 90.1% 91.7% 87.1% 63.8% 79.3% 90.6% 88.0%

92.9% 93.9% 88.3% 94.7% 71.3% 87.5% 47.4% 92.0% 53.8% 74.1% 60.5% 74.5%

96.8% 93.6% 97.1% 89.8% 92.9% 87.9% 89.9% 88.4% 73.0% 89.6% 70.0% 93.9%

92.9% 94.6% 93.4% 95.5% 83.4% 88.1% 88.5% 87.7% 59.7% 85.1% 78.8% 85.0%

95.8% 96.7% 95.7% 96.8% 91.4% 95.0% 91.2% 86.3% 83.1% 89.6% 80.3% 88.3%

93.7% 97.4% 97.0% 98.7% 86.2% 95.5% 92.5% 96.3% 86.2% 87.3% 87.3% 91.5%

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

N=70 N=90

1.7% 68.3% 42.7% 81.9%

N=110

N=110

N=110

N=110

N=110

N=110

N=110

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Task Brevo2 GDN - noconv1d

Task Brevo2 GDN - original (conv1d)

Task Brevo2 GDN - Canon-ABCD(no-res)

Task Brevo2 GDN - Canon-ABCD(res)

Task Brevo2 GDN - Canon-AbCD(no-res)

Task Brevo2 GDN - Canon-AbCD(res)

Task Brevo2 GDN - Canon-ACD(res)

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

90.3% 93.2% 91.3% 97.5% 56.5% 84.2% 71.2% 89.4% 15.5% 32.9% 58.2% 69.1%

97.3% 98.8% 98.6% 98.7% 92.7% 96.3% 96.1% 96.8% 73.1% 92.7% 87.4% 89.1%

94.9% 98.2% 96.5% 96.7% 88.1% 95.2% 81.1% 92.3% 79.1% 82.0% 70.1% 67.2%

95.9% 97.9% 95.3% 97.4% 75.2% 94.4% 84.6% 93.4% 62.6% 87.9% 77.5% 63.4%

96.8% 97.3% 93.4% 98.1% 85.3% 91.7% 82.2% 94.5% 80.4% 81.3% 70.2% 87.8%

98.0% 97.6% 98.4% 98.6% 92.6% 94.0% 93.4% 96.3% 87.4% 86.4% 86.6% 90.6%

96.2% 98.5% 95.7% 98.0% 85.6% 91.6% 87.8% 86.1% 53.3% 89.5% 76.8% 76.5%

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

N=30 N=40 N=50

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Mano GDN - noconv1d

Task Mano GDN - original (conv1d)

Task Mano GDN - Canon-ABCD(no-res)

###### Task Mano GDN - Canon-ABCD(res)

Task Mano GDN - Canon-AbCD(no-res)

Task Mano GDN - Canon-AbCD(res)

Task Mano GDN - Canon-ACD(res)

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

54.3% 64.1% 66.1% 72.3% 44.9% 44.7% 42.2% 36.4% 25.1% 53.3% 54.8% 18.9%

93.6% 97.9% 91.6% 89.8% 90.0% 98.4% 85.0% 75.0% 81.2% 63.6% 55.2% 52.6%

99.0% 99.1% 99.5% 99.8% 98.8% 98.9% 97.0% 99.3% 94.1% 96.4% 98.3% 99.3%

92.1% 99.4% 99.4% 96.4%

- 98.9% 99.5% 99.6% 99.8%
- 99.3% 97.7% 98.8% 99.1% 93.2% 79.6% 99.0% 98.0%

99.7% 98.3% 99.3% 98.1% 90.3% 87.1% 98.7% 95.5% 93.1% 87.8% 80.3% 89.3%

89.6% 90.9% 96.9% 98.6% 82.8% 93.0% 76.9% 93.4% 53.8% 31.5% 83.7% 72.8%

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

L=10 L=13 L=16

- 81.6% 82.8% 94.7% 96.7%
- 82.8% 75.0% 84.3% 94.0%

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Lano GDN - noconv1d

Task Lano GDN - original (conv1d)

Task Lano GDN - Canon-ABCD(no-res)

Task Lano GDN - Canon-ABCD(res)

Task Lano GDN - Canon-AbCD(no-res)

Task Lano GDN - Canon-AbCD(res)

Task Lano GDN - Canon-ACD(res)

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

35.3% 49.7% 25.5% 69.9% 2.8% 3.6% 2.6% 26.1% 9.4% 13.4% 9.1% 12.9%

83.4% 93.8% 91.5% 94.4% 54.9% 80.3% 70.9% 86.5% 38.0% 63.4% 44.6% 65.7%

89.7% 94.0% 91.8% 95.1% 63.7% 79.2% 68.4% 84.8% 51.9% 63.6% 53.3% 76.7%

91.9% 95.1% 92.0% 95.3% 70.5% 84.3% 78.9% 88.1% 49.4% 65.2% 57.4% 68.8%

87.8% 93.8% 90.8% 94.6% 67.1% 82.0% 73.7% 85.1% 57.4% 76.3% 63.5% 74.2%

90.6% 93.6% 92.8% 95.1% 69.7% 86.2% 82.4% 87.1% 49.6% 68.3% 62.9% 75.2%

90.2% 93.1% 92.0% 94.7% 56.2% 82.1% 74.0% 87.5% 46.9% 61.0% 53.3% 67.6%

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

###### Task Lano GDN - noconv1d

###### Task Lano GDN - original (conv1d)

###### Task Lano GDN - Canon-ABCD(no-res)

Task Lano GDN - Canon-ABCD(res)

Task Lano GDN - Canon-AbCD(no-res)

Task Lano GDN - Canon-AbCD(res)

Task Lano GDN - Canon-ACD(res)

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

0.00711 0.00490 0.00912 0.00278 0.01114 0.01038 0.01172 0.00475 0.01041 0.00898 0.01020 0.00919

0.00157 0.00067 0.00087 0.00063 0.00247 0.00102 0.00151 0.00074 0.00435 0.00231 0.00366 0.00214

0.00108 0.00068 0.00086 0.00058 0.00198 0.00111 0.00170 0.00083 0.00309 0.00228 0.00302 0.00151

0.00088 0.00058 0.00084 0.00056 0.00160 0.00090 0.00111 0.00066 0.00339 0.00218 0.00279 0.00193

0.00124 0.00074 0.00091 0.00061 0.00180 0.00099 0.00138 0.00081 0.00284 0.00148 0.00233 0.00169

0.00102 0.00071 0.00076 0.00054 0.00157 0.00078 0.00098 0.00068 0.00333 0.00197 0.00237 0.00162

0.00100 0.00076 0.00086 0.00062 0.00237 0.00098 0.00140 0.00071 0.00380 0.00245 0.00299 0.00204

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3f cfg3j

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

cfg3k

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

8L512D 12L512D 8L768D 12L768D

Figure 36: GDN variants (left to right): no conv1d, original (w/ conv1d), Canon-ABCD(no-res), CanonABCD(res), Canon-AbCD(no-res), Canon-AbCD(res), Canon-ACD(res).

#### References

- [1] Marah Abdin, Jyoti Aneja, Harkirat Behl, S´ebastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.
- [2] Zeyuan Allen-Zhu. Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality, 2025. URL https://physics.allen-zhu. com/part-4-architecture-design/part-4-2. Code released at https://github.com/ facebookresearch/PhysicsLM4.
- [3] Zeyuan Allen-Zhu and Yuanzhi Li. Can SGD Learn Recurrent Neural Networks with Provable Generalization? In NeurIPS, 2019. Full version available at http://arxiv.org/abs/1902.01028.
- [4] Zeyuan Allen-Zhu and Yuanzhi Li. Backward Feature Correction: How Deep Learning Performs Deep (Hierarchical) Learning. In Conference on Learning Theory, COLT ’23, 2023. Full version available at http://arxiv.org/abs/2001.04413.
- [5] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. In Proceedings of the 41st International Conference on Machine Learning, ICML 2024,

2024. Full version available at http://arxiv.org/abs/2309.14316.

- [6] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 1, Learning Hierarchical Language Structures. Transactions on Machine Learning Research, 2025. Full version available at http://arxiv. org/abs/2305.13673.
- [7] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.2, Knowledge Manipulation. In Proceedings of the 13th International Conference on Learning Representations, ICLR 2025, 2025. Full version available at http://arxiv.org/abs/2309.14402.
- [8] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. In Proceedings of the 13th International Conference on Learning Representations, ICLR 2025,

2025. Full version available at http://arxiv.org/abs/2404.05405.

- [9] Simran Arora, Aman Timalsina, Aaryan Singhal, Benjamin Spector, Sabri Eyuboglu, Xinyi Zhao, Ashish Rao, Atri Rudra, and Christopher R´e. Just read twice: closing the recall gap for recurrent language models. arXiv preprint arXiv:2407.05483, 2024.
- [10] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024.
- [11] Yoshua Bengio, J´erˆme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.
- [12] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.
- [13] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPT-NeoX-20B: An open-source autoregressive language model. In Proceedings of the ACL Workshop on Challenges & Perspectives in Creating Large Language Models, 2022. URL https://arxiv.org/abs/2204.06745.
- [14] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [15] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.
- [16] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.
- [17] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2924–2936, 2019. doi: 10.18653/v1/N19-1300.
- [18] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.
- [19] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024. URL https://arxiv.org/abs/ 2405.21060.
- [20] Soham De, Samuel L Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, et al. Griffin: Mixing gated linear recurrences with local attention for efficient language models. arXiv preprint arXiv:2402.19427, 2024.
- [21] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1246. URL https: //aclanthology.org/N19-1246/.
- [22] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research, 23(1):5232–5270, 2022.
- [23] Daniel Y Fu, Tri Dao, Khaled Kamal Saab, Armin W Thomas, Atri Rudra, and Christopher R´e. Hungry hungry hippos: Towards language modeling with state space models. arXiv preprint arXiv:2212.14052,

2022. URL https://arxiv.org/abs/2212.14052.

- [24] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. URL https://zenodo.org/records/12608602.
- [25] Olga Golovneva, Tianlu Wang, Jason Weston, and Sainbayar Sukhbaatar. Multi-token attention. arXiv preprint arXiv:2504.00927, 2025.
- [26] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. URL https://arxiv.org/abs/2312.00752.
- [27] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, et al. Conformer: Convolution-augmented transformer for speech recognition. arXiv preprint arXiv:2005.08100, 2020.
- [28] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654, 2020.
- [29] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.
- [30] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, Joe Chau, Peng Cheng, Fan Yang, Mao Yang, and Yongqiang Xiong. Tutel: Adaptive mixture-of-experts at scale. CoRR, abs/2206.03382, June 2022. URL https://arxiv.org/ pdf/2206.03382.pdf.
- [31] Samy Jelassi, David Brandfonbrener, Sham M Kakade, and Eran Malach. Repeat after me: Transformers are better than state space models at copying. arXiv preprint arXiv:2402.01032, 2024.
- [32] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023. doi: 10.1145/3571730. URL https://doi.org/10.1145/3571730.
- [33] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [34] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.
- [35] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Fran¸cois Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020.
- [36] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186, 2019.
- [37] Yury Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and Mikhail Burtsev. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack. Advances in Neural Information Processing Systems, 37:106519–106554, 2024.
- [38] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl a 00276. URL https://aclanthology.org/Q19-1026/.

- [39] Nayoung Lee, Ziyang Cai, Avi Schwarzschild, Kangwook Lee, and Dimitris Papailiopoulos. Selfimproving transformers overcome easy-to-hard and length generalization challenges. arXiv preprint arXiv:2502.01612, 2025. URL https://arxiv.org/abs/2502.01612.
- [40] OpenAI. Gpt-4 technical report, 2023.

- [41] Denis Paperno, Germ´n Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fern´ndez. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, 2016. doi: 10.18653/v1/P16-1144.
- [42] Guilherme Penedo, Hynek Kydlı´ˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://arxiv.org/abs/2406.17557.
- [43] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.
- [44] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022. URL https://arxiv.org/abs/2201.02177.
- [45] Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.
- [46] Zhen Qin, Songlin Yang, and Yiran Zhong. Hierarchically gated recurrent neural network for sequence modeling. Advances in Neural Information Processing Systems, 36:33202–33221, 2023.
- [47] QwenTeam. Qwen3-Next: Towards Ultimate Training & Inference Efficiency, Sep 2025. URL https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research. latest-advancements-list.
- [48] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [49] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia, 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-2124. URL https://aclanthology.org/P18-2124/.
- [50] Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling. arXiv preprint arXiv:2406.07522, 2024.
- [51] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial winograd schema challenge at scale. arXiv preprint arXiv:1907.10641, 2019.
- [52] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463–4473, 2019. doi: 10.18653/v1/D19-1454.
- [53] Eshika Saxena, Alberto Alfarano, Emily Wenger, and Kristin Lauter. Teaching transformers modular arithmetic at scale. arXiv preprint arXiv:2410.03569, 2024. URL https://arxiv.org/abs/2410. 03569.
- [54] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [55] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [56] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.
- [57] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2016.
- [58] Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. Simplified state space layers for

- sequence modeling. arXiv preprint arXiv:2208.04933, 2022.
- [59] DR So, W Manke, H Liu, Z Dai, N Shazeer, and QV Le. Primer: Searching for efficient transformers for language modeling. arxiv 2021. arXiv preprint arXiv:2109.08668, 2021.
- [60] Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras. net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, June

2023. URL https://huggingface.co/datasets/cerebras/SlimPajama-627B.

- [61] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2021.
- [62] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.
- [63] Falcon-LLM Team. Falcon-h1: A family of hybrid-head language models redefining efficiency and performance, May 2025. URL https://falcon-lm.github.io/blog/falcon-h1.
- [64] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [65] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [66] Asher Trockman, Hrayr Harutyunyan, J Zico Kolter, Sanjiv Kumar, and Srinadh Bhojanapalli. Mimetic initialization helps state space models learn to recall. arXiv preprint arXiv:2410.11135, 2024.
- [67] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,  Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [68] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.
- [69] Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M Rush, Bart Van Merri¨enboer, Armand Joulin, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks. arXiv preprint arXiv:1502.05698, 2015.
- [70] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22–31, 2021.
- [71] Songlin Yang and Yu Zhang. Fla: A triton-based library for hardware-efficient implementations of linear attention mechanism, January 2024. URL https://github.com/fla-org/flash-linear-attention.
- [72] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2023.
- [73] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024.
- [74] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. arXiv preprint arXiv:2406.06484, 2024.
- [75] Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of Language Models: Part 2.1, GradeSchool Math and the Hidden Reasoning Process. In Proceedings of the 13th International Conference on Learning Representations, ICLR 2025, 2025. Full version available at https://arxiv.org/abs/2407. 20311.
- [76] Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems. In Proceedings of the 13th International Conference on Learning Representations, ICLR 2025, 2025. Full version available at http://arxiv. org/abs/2408.16293.

- [77] Ping Yu, Jing Xu, Jason Weston, and Ilia Kulikov. Distilling system 2 into system 1. arXiv preprint arXiv:2407.06023, 2024.
- [78] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025.
- [79] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, 2019. doi: 10.18653/v1/P19-1472.
- [80] Yu Zhang, Songlin Yang, Rui-Jie Zhu, Yue Zhang, Leyang Cui, Yiqiao Wang, Bolun Wang, Freda Shi, Bailin Wang, Wei Bi, et al. Gated slot attention for efficient linear-time sequence modeling. Advances in Neural Information Processing Systems, 37:116870–116898, 2024.
- [81] Zhengyan Zhang, Yixin Song, Guanghui Yu, Xu Han, Yankai Lin, Chaojun Xiao, Chenyang Song, Zhiyuan Liu, Zeyu Mi, and Maosong Sun. Relu2 wins: Discovering efficient activation functions for sparse llms. arXiv preprint arXiv:2402.03804, 2024.
- [82] Yongchao Zhou, Uri Alon, Xinyun Chen, Xuezhi Wang, Rishabh Agarwal, and Denny Zhou. Transformers can achieve length generalization but not robustly. arXiv preprint arXiv:2402.09371, 2024.

