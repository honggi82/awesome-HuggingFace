# arXiv:2509.06806v6[cs.CL]11Apr2026

MACHINELEARNINGLM†: SCALING MANY-SHOT

IN-CONTEXT LEARNING VIA CONTINUED PRETRAINING

Haoyu Dong*, Pengkun Zhang, Mingzhe Lu, Yanzhen Shen, Guolin Ke UCAS, Microsoft, SCUT, Stanford

ABSTRACT

Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine-learning (ML) tasks—i.e., to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MACHINELEARNINGLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust many-shot ICL capability while preserving its general knowledge and reasoning for broader chat workflows.

Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a randomforest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3–6× more examples per context window and delivering up to 50× amortized throughput via batch inference.

Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MACHINELEARNINGLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of ∼15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest–level accuracy across hundreds of shots. General chat capabilities—including knowledge and reasoning—are preserved: it achieves 75.4% on MMLU.

[Figure 2]

Model & Data: https://huggingface.co/MachineLearningLM Code: https://github.com/HaoAreYuDong/MachineLearningLM

[Figure 3]

[Figure 4]

[Figure 5]

- (b) Accuray boosting of MachineLearningLM across tabular tasks.
- (c) Many-shot scaling of MachineLearningLM, averaged across tasks.

[Figure 6]

[Figure 7]

(a) Prompt template of MachineLearningLM.

Figure 1: MACHINELEARNINGLM on in-context ML tasks: (a) prompt template; (b) 512-shot accuracy across domains vs. Qwen-2.5-7B-Instruct; (c) many-shot scaling (23–210 shots) vs. LLMs.

†At present, MACHINELEARNINGLM targets tabular ML; broader ML is future work.

*Corresponding author; details in Appendix 7.

- 1 INTRODUCTION

LLMs possess broad world knowledge, general perceptual as well as reasoning abilities, making

- them promising few-shot learners (Brown et al., 2020). However, they often fail to learn new tasks despite being given many-shot demonstrations on standard ML benchmarks (Agarwal et al., 2024; Gardner et al., 2024), or to fully exploit rich experiences stored in agent memory in interactive, open-ended workflows (Wang et al., 2023; Shinn et al., 2023; Wang et al., b). In addition, accuracy gains typically plateau —often after just a handful of demonstrations— and are sensitive to the label biases and choice/order of examples (Chen et al.; Liu et al., 2024a; Zhao et al., 2021; Fei et al., 2023; Liu et al., 2022). In practice, LLMs are largely guided by surface-level signals, such

- as distributional, formatting cues (Min et al., 2022) and nearest-neighbor imitation (Agarwal et al., 2024), and rarely uncover new causal mechanisms or statistical dependencies that are required to yield accurate predictions.

Orthogonally, pioneering tabular models (Hollmann et al., 2025; Qu et al., 2025) have demonstrated ML tasks can be solved purely by ICL—without gradient descent. However, these tabular-specific model architectures cannot leverage the broad prior world knowledge and general-purpose multimodal perception that LLMs acquire during pretraining; consequently, they depend heavily on welldesigned featurization (Shi et al., 2021; Mráz et al., 2025) and abundant labeled data for training. At this intersection, we ask:

## Can we teach an LLM to “do ML in context” while preserving general abilities?

To this end, we propose a pretraining-plus-prompting framework, MACHINELEARNINGLM, that equips LLMs with in-context ML capabilities to fully exploit many-shot in-context examples. MACHINELEARNINGLM performs LoRA-based (Hu et al., 2022) continued pretraining with a standard next-token objective on millions of synthetic tabular prediction tasks drawn from SCM-based priors (Peters et al., 2017; Pearl, 2009): a task generator samples arbitrarily large numbers of binary/multiclass tasks under SCM-based priors spanning diverse feature types, marginal distributions, and label mechanisms following (Qu et al., 2025). This approach ensures strict non-overlap between our pretraining data and evaluation datasets. After pretraining, the model can directly leverage incontext examples of a new task to generate predictions for unseen instances—without parameter update.

As summarized in Table 1, unlike previous instruction-tuning methods (Wang et al., 2022b; Chung

- et al., 2024) that rely on a limited set of real tasks (typically ∼ 103), we train on O(106) synthetic tasks with diverse causal mechanisms and varied shot counts. Our approach also differs from specialized tabular learners as we preserve the versatility of LLMs, which enables them to continue to leverage contextual task descriptions, draw on external knowledge, and interact directly with multimodal, heterogeneous inputs. Through large-scale pretraining, our method is able to equip LLMs with striking many-shot scaling and robust numerical modeling. We hope MACHINELEARNINGLM is a promising paradigm to inspire new research that leverages LLMs’ general perception and reasoning capabilities, and also possibly extending to more modalities beyond text, as detailed in Section 6.

Table 1: Comparison across ML paradigms.

Method In-context learning

Robust numerical modeling

General knowledge priors

Native multimodal input

RF, Boosted trees ✗ ✓ ✗ ✗ TabPFN, TabICL ✓ ✓ ✗ ✗ TabLLM ✗ ✗ ✓ ✓ GPT-5, Qwen ✓ ✗ ✓ ✓ MACHINELEARNINGLM ✓ ✓ ✓ ✓ Key: ✓ Yes, ✗ No. Note: “Native multimodal” primarily means textual + numerical + tabular and can be

naturally extended to modalities like images by building on multimodal LLM backbones.

Warm-up training with a Random Forest teacher. Directly training on synthetic tasks can lead to model collapse or underperformance, particularly when tasks are too complex to be learn from only a few (e.g., 64) examples. This makes strong ML methods no better than random guessing or always predicting the majority class. We stabilize the onset by mimicking a random-forest (RF) teacher on each task—first matching example predictions—before transitioning to self-reliant incontext prediction. This leverages knowledge distillation for improved optimization (Hinton et al., 2015). We choose a random forest teacher for its balance of robustness and interpretability. Its decision process can be transparently decomposed into rule paths and feature attributions and directly serialized into interpretable “reasoning steps”—–then rule chains and faithful local explanations (Deng, 2014; Friedman & Popescu, 2008; Lundberg & Lee, 2017), which naturally align with the chain-of-thought (CoT) reasoning. In this work, we distill the predicted labels only, but future directions could leverage its “reasoning steps” as rationales for reasoning-augmented training (Guo

- et al., 2025) and enhancing the interpretability of model predictions.

Token-efficient prompting for in-context ML. LLMs are fundamentally bottlenecked by their context length. Our design enables 3-6× more examples within a context window and yields 50× amortization via batch inference. We achieve this with three composable design choices.

- (i) A tabular encoding to organize many-shot examples rather than as scattered and lengthy NL descriptions (Hegselmann et al., 2023; Gardner et al., 2024). Recent works (Dong et al., 2024; Sui et al., 2024) have shown that regular tabular structures can be seamlessly understood by LLMs without requiring special row/column attention (Hollmann et al.). In addition, tabular encoding 1 naturally combines heterogeneous with no mandatory categorization for text.
- (ii) A compact integer-based encoding in which we normalize all numbers for each numerical feature to integers in [0,999], eliminating the tokenization fragmentation caused by decimal points (“.”) and leading signs (“+”/“–”). This not only reduces the token count—e.g., cl100k_base treats any integer in [0,999] as a single token, but also avoids a common pitfall where LLMs compare decimals

- as strings (e.g., “1.11” (1|.|11) vs. “1.9” (1|.|9)) (Hugging Face, 2024).

- (iii) A sequence-level batch-prediction mechanism that packs dozens of test examples into each sequence and predicts all of them in a single forward pass, stabilizing gradients for continued pretraining and amortizing instruction/context overhead (Lin et al.; Cheng et al., 2023).

Order-robust, confidence-aware self-consistency. LLM predictions can be sensitive to longcontext position effects (Liu et al., 2024a). At inference, we marginalize over permutations of demonstration and feature order, combining outputs via confidence-weighted self-consistency—i.e., a calibrated, probability-weighted vote over samples (Zhao et al., 2021; Wang et al., a).

MACHINELEARNINGLM continues pretraining on millions of synthetic tasks with diverse causal mechanisms and varied shot counts; our key experimental highlights are:

- • Many-shot scaling. MACHINELEARNINGLM exhibits striking many-shot scaling across diverse tasks from the TALENT benchmark (Liu et al., 2024b), surpassing both leading open-source (e.g. Qwen-2.5-7B) and closed-source (e.g. GPT-5-mini) LLMs by an average of ∼15% under high-shot settings. Moreover, it exhibits clear out-of-distribution generalization in context length, from a 32k-token pretraining budget to a 131k-token inference budget.
- • Competitive many-shot performance vs. state-of-the-art tabular methods. Without any task-specific training, MACHINELEARNINGLM reaches random-forest–level accuracy across shot counts from 8 to 512 (within 2% relative on average) and clearly surpasses instance-based kNN, demonstrating robust numerical modeling.
- • Heterogeneous (multimodal)-input generalization. By combining LLM versatility with classical ML’s robust numerical fitting, MACHINELEARNINGLM natively consumes natural-language (NL) features alongside numerical values—without relying on text bucketing or embeddings—and achieves competitive results on the Talent benchmarks.

1It is extensible and can support images, hyperlinks, and embedded objects, as in HTML tables.

• General abilities preserved (chat workflows). MACHINELEARNINGLM retains general capabilities in chat-style workflows; for example, on MMLU it attains 73.2% micro accuracy in 0-shot and 75.4% in 50-shot, comparable to strong general-purpose LLMs.

- 2 METHOD

- 2.1 PRETRAINING CORPUS SYNTHESIS

To generate pretraining data, we build a task generator following TabICL2 (Qu et al., 2025) that samples diverse binary/multiclass problems under SCM-based priors determined by choices of graph structure, mechanisms, feature types, and class formation.

SCM graph and mechanisms. We first sample a DAG G following the structure of a fully connected MLP, where each neuron corresponds to a variable, then assign structural equations to each node v in G:

### xv ← fv xparents(v) + εv, εv i.i.d. noise.

where fv is drawn from a rich pool of functions, following (Qu et al., 2025), including tanh, leakyReLU, ReLU, ReLU6, SELU, SiLU, Softplus, Hardtanh, sign, sine, RBF, exp, gaussian-processsampled random activations,

Tree-based SCMs. To inject tree inductive biases beyond neural layers, 30% fvs are replaced by multi-output gradient-boosted regressors (fℓ) fitted on fake targets drawn from Gaussian noise and parent inputs. For each layer:

nestimators ∼ min 4, 1 + Exponential(λ = 0.5) , max_depth ∼ min 4, 2 + Exponential(λ = 0.5) ,

We also fit fℓ : xparents(ℓ)  → y with y ∼ N(0,I), and set xℓ ← fℓ xparents(ℓ) .

Feature and label typing. We sample a fraction of categorical features, discretize dense columns into bins, and (with some probability) shuffle category IDs to avoid spurious order. Continuous features remain numeric. As for a scalar y⋆. We create K-way labels by sampling K ∈{2,...,10}, drawing ordered bounds τ1<···<τK−1 from the empirical distribution of y⋆, and mapping

⊮ τk−1 ≤ y⋆ < τk ,

y = arg max

k∈{1,...,K}

Followed by a random permutation of class IDs to destroy ordinal hints. Class imbalance arises naturally from the sampled bounds.

Task sampling policy and corpus scale. We repeatedly instantiate fresh SCMs, yielding a corpus of ∼ 3 × 106 distinct SCMs. For each SCM, we propagate noise through the structural equations to generate i.i.d. samples (X,y). For each task, we sample the number of demonstrations M (capped

- at 1,024), fix the number of queries at N = 50, and draw the number of features d ∈ {5,...,50} and classes K ∈ {2,...,10}. Each task is serialized into a prompt following Section 2.4 and comprises the following components: a compact tabular many-shot block (features with labels), a tabular test block (IDs with features), and an instruction specifying a JSON array output {"id" : ·,"label" : ·},... , as illustrated in Figure 1a. Given a token budget of 32k Qwen 2.5-Instruct in our framework, we truncate the number of many-shot samples per task to fit the length limitation for continued pre-training.

2We directly use the source code of TabICL in https://github.com/soda-inria/tabicl.

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

- Figure 2: Heatmap of task density across feature and shot counts. We sample 100k tasks from the synthetic generator and aggregate by feature_num and shot_num bins. Color encodes the sum of num per bin. Training token budget capped at 32k tokens.

- 2.2 OBJECTIVE OF CONTINUED PRE-TRAINING

As illustrated in Figure 1a, we cast each in-context tabular prediction task as a conditional sequence modeling problem. For a given task, the prompt x concatenates: (i) an instruction header H describing the task and specifying a JSON output format, (ii) a many-shot training block with a demonstration block S = {si}Mi=1 of M-shot labeled demonstrations (demonstration order immaterial), and (iii) a query block Q = {x(i)}Ni=1 of N unlabeled queries (each with an id). (ii) and (iii) follow the token-efficient prompting in Sec. 2.4.

H

instruction + schema

∥ s1 ∥ ··· ∥si ∥ ··· ∥sM−1

in-context demonstrations

∥ x1 ∥ ··· ∥xN−1

batched queries

The model is required to generate a single JSON array.

y = {"id" : 0, "label" : ℓˆ1},...,{"id" : N − 1, "label" : ℓˆN−1} .

Loss. We introduce no auxiliary heads or bespoke objectives, and use the standard left-to-right loglikelihood language modeling objective over the JSON target for continued pre-training. Formally, let tok(·) denote the tokenizer and

x = tok(x), y = tok(y) = (y1,...,yT). The training loss for parameters θ is the negative log-likelihood

L(θ) = −

T

t=1

log pθ(yt | x,y<t), (1)

applied to the entire JSON string (brackets, keys, colons, commas, IDs, and label tokens). This encourages (i) learning the mapping from features to labels, (ii) preserving test-row order via aligned "id" fields, and (iii) strict adherence to the required JSON format.

- 2.3 RANDOM-FOREST MIMIC WARM START

Directly training on synthetic tasks can cause training collapse as low-signal tasks, limited few-shot examples, severe class imbalance, and randomly sampled label regimes may yield poor local optima,

bend the loss curve, and occasionally destabilize training (Ochal et al., 2023). We mitigate this risk by first mimicking a Random Forest (RF) teacher on each task, where we match per-example predictions in a short warm-up—before transitioning to standalone in-context prediction. This provides more informative targets and smoother gradients in the early phase, similar to knowledge distillation (Hinton et al., 2015). We choose RF as the teacher because its decision process decomposes transparently into rule paths and feature attributions. These can be serialized as explicit “reasoning steps” (Deng, 2014; Friedman & Popescu, 2008; Lundberg & Lee, 2017), which aligns naturally with the stepwise reasoning patterns large language models can emulate.

For each pretraining task, we (i) utilize a better-than-random guard at the task level to skip lowsignal or degenerate tasks, and (ii) apply an example-level consensus filter that retains only test examples where the RF prediction matches the ground truth; we filter examples at this stage but do not alter the original train/test labels. After the warm-up ends, we continue to apply (i) but disable (ii), enabling the model to transition from relying on random-forest–induced priors to developing its own in-context ML behavior. This design prunes low-signal tasks before they corrupt optimization and, by supplying high-precision targets, reduces gradient noise, prevents collapse-to-majority under imbalance, and yields a smoother warm-up that mimics robust numerical modeling—ultimately easing the shift from teacher imitation to independent in-context ML.

- (i) Task-level guard against chance and collapse. To rule out degenerate tasks where the RF performs no better than chance, we compare its accuracy to a conservative random baseline

p0 = max

K

k=1

p2k, max

k

pk ,

where K is the number of classes, and pk denotes the class prior from ytrue. This baseline corresponds to the larger of “sampling by prior” or “always predicting the majority class.” Let

Ncorrect = #{i : yi = yˆi} be the number of correct predictions, we then perform a one-sided binomial tail test p-value = Pr[X ≥Ncorrect], X ∼Bin(N,p0) and require p-value < α.3 To avoid trivial solutions and collapsed behavior under class imbalance, we enforce the following criteria4—each targets a distinct failure mode:

- • Chance-corrected agreement: Cohen’s κ > 0.
- • Imbalance-robust accuracy: Balanced accuracy > 1/K + δbacc.
- • Macro-F1 dominance: F1macro ≥ F1(macromaj) + δF1.
- • Non-collapse checks: At least two predicted classes; dominant predicted class fraction ≤ τdom.
- • Evaluation setup: N = 50 and K ≥ 2.

Tasks failing to meet all criteria are skipped, and only tasks passing all criteria are used in pretraining.

- (ii) Example-level label-prediction consensus filter (warm-up only). For admitted tasks, we retain in the prompt only those evaluation examples where the RF prediction matches the ground truth (RF = GT), discarding mismatched cases while keeping original labels and splits unchanged. This enforces teacher–student alignment without relabeling, yielding cleaner supervision in the earliest updates. After the warm-up stage, consensus filtering is discontinued, allowing the model to move beyond imitation and develop standalone in-context policies.

After filtering, we reduce the evaluation set size from N = 50 to Nconse = 20. For the remaining usable examples, we resample within each label class to restore Nconse = 20, aiming to match the proportions of many-shot demonstrations. This prevents collapse-to-majority under imbalance and avoids sub-optimal supervision.

- 3For N = 50 the test has relatively low power; we use a slightly relaxed α (default 0.2) to avoid discarding borderline-but-promising tasks.
- 4We use κ > 0.01, δbacc = 0.03, δF1 = 0.00, and τdom = 0.95, where maj denotes the always-majority classifier.

NL Description Style

- Example 1: The income is 29370. The occupation is Doctorate. The annual growth rate is -12.34%. Label: 1.
- Example 2: The income is 41020. The occupation is Bachelors. The annual growth rate is 20.25%. Label: 0.

[Figure 27]

Tabular Style

29370,Doctorate,-12.34%|1 41020,Bachelors,20.25%|0

- Figure 3: A comparison between the NL description style and tabular style of many-shot examples.

- 2.4 TOKEN-EFFICIENT PROMPTING FOR IN-CONTEXT ML

The application of many-shot in-context ML is often constrained by the fixed context length and high computational cost of long contexts (Gardner et al., 2024). We alleviate these limitations through three design choices: (i) compact structuring of in-context examples, (ii) compression of numeric tokens, and (iii) sequence-level batching.

- 2.4.1 TABULAR ENCODING

Instead of presenting demonstrations S as scattered NL descriptions (Hegselmann et al., 2023; Gardner et al., 2024), we place many-shot examples in a concise table-style format. Building on recent findings that normal tabular structures can be seamlessly understood by popular LLMs (Dong et al., 2024; Sui et al., 2024), our approach preserves column/field meaning while dramatically reducing tokens relative to full-sentence templates.

The example in 3 contrasts an encoding table expressed as scattered NL descriptions, as in TabLLM (Hegselmann et al., 2023) and TabuLa (Gardner et al., 2024). with our compact tabular encoding, where commas separate features and “|” separates the label (the header is optional).

Notably, while feature names and task descriptions may contain contextual semantics that could be readily exploited by LLMs, our approach avoids using such information. This ensures a fair comparison with traditional ML methods and avoids data contamination and memorization (Bordt et al.). Moreover, the current tabular encoding jointly represents heterogeneous textual and numeric features, without requiring mandatory categorical bucketing or embedding extraction for text. Toward future extensions of current tabular encoding to multimodal features, HTML tables can be leveraged to support images, hyperlinks, and embedded objects.

- 2.4.2 COMPACT INTEGRAL-BASED NUMBER ENCODING

We normalize all numbers of each numerical feature to non-negative integers in [0,999] to eliminate the tokenization fragmentation caused by decimal points (“.”) and leading signs (“+”/“–”). The resulting values are then represented as plain text. This preserves ordinal structure yet maps each number from a sequence of scattered tokens to an integral single token under GPT’s cl100k_base. Beyond token savings, integer rendering avoids a frequent pitfall where decimals are compared textwise (e.g., “1.11” (1|.|11) vs. “1.9” (1|.|9 where 11 is bigger than 9 while 1.11 is smaller than

- 1.9)) rather than numerically (Spithourakis & Riedel, 2018; Wallace et al., 2019; Singh & Strouse, 2024; Hugging Face, 2024).

Connection to z-norm. Recent work in tabular modeling, including TabICL (Qu et al., 2025), has adopted z-score normalization (mean–variance standardization) for each feature, which yields realvalued decimals. We also adopt this approach as the initial step before discretizing the normalized values into bounded integers in [0,999].

Integer mapping. We map each normalized value z to an integer i ∈ [0,999] via i = clip round(120z + 500), 0, 999 , where round(x) = ⌊x + 0.5⌋ and clip(x,a,b) = min{max(x,a), b}.

Under this mapping, z = 0 is mapped to i = 500, and z = ±4.166–which covers 99.997% of values under the standard normal distribution– maps approximately to i = 0 − 999, with out-ofband values clipped. This transformation yields [0,999]-norm, which preserves numeric order while ensuring that most values are tokenized as a single token under cl100k_base. Specifically, GPT (cl100k_base) and LLaMA-3 vocabularies merge consecutive digits, commonly treating a sequence of three digits as a single token. In contrast, Qwen and LLaMA-2 vocabularies tokenize numbers

- at the level of individual digits, leading to much more fragmented representations. Below illustrates typical fragmentations of popular tokenizers for z-norm decimals and [0-999] non-negative integers.

Z-norm (decimal form) GPT (cl100k_base), LLaMA-3: −0.1234 → - | 0 | . | 123 | 4 Qwen, LLaMA-2: −0.1234 → - | 0 | . | 1 | 2 | 3 | 4 [0,999]-norm (integer form) GPT (cl100k_base), LLaMA-3: 486 → 486 Qwen, LLaMA-2: 486 → 4 | 8 | 6

- 2.4.3 SEQUENCE-LEVEL BATCH-PREDICTION

As shown in Figure 1, we pack N query rows into a single sequence and decode all predictions in one forward pass, increasing the effective batch size at the sequence level (Lin et al.; Cheng et al., 2023). Let the sequence consist of a shared header H (task instruction, schema) and in-

context demonstrations S, followed by N test queries {x(i)}Ni=1 and an output section for predictions {y(i)}Ni=1:

### =⇒ y(1) ∥ ··· ∥y(N)

### ∥ x(1) ∥ ··· ∥x(N)

### H ∥S

### .

batched queries

batched predictions

instruction + schema + M shots

During training, MACHINELEARNINGLM remains purely autoregressive and is optimized with the standard next-token objective (Section 2.2). Increasing N accelerates and stabilizes continued pretraining. At inference, the model predicts N samples in a single pass, amortizing instruction and context overhead (Section 2.4.4).

Stability and reliability. A big N not only consumes the token budget that would otherwise be allocated to demonstrations, but also increases instruction-following errors (e.g., cross-item interference and run-on/non-terminated generations), particularly for smaller open-source models. We therefore set N = 50 by default. In addition, we randomize the within-sequence order of {x(i)} to reduce position bias and improve exchangeability of the batched items.5 Importantly, id should be 0-indexed; 1-based indexing often destabilizes instruction following and leads to incorrect IDs.

- 2.4.4 COMPRESSION AND AMORTIZATION RATIO Token-cost model. We model the amortized token cost per predicted label as follows.

Let M denote the number of many-shot demonstrations (with labels) and N be the number of query examples to be inferred (with IDs). Each row is dominated by its d features, so we represent the perrow token cost by R, without distinguishing labels and IDs. The instruction/schema header incurs a token cost of H, which is negligible compared to R(M + N) and can be omitted. The resulting amortized token cost per predicted label is therefore

H + R (M + N) N ≈

R (M + N) N

C =

.

5This simple permutation also helps mitigate long-context position effects observed in batch-prediction.

Compression and amortization ratio. The overall compression ratio is defined as the product of three saving factors: (A) tabular structure, (B) number normalization, and (C) batch inference

- (A) TABULAR STRUCTURE (NL–DECIMAL → TABULAR–DECIMAL). We replace scattered NL sentences (e.g., “the first feature is 0.1234) ” with compact, comma-delimited rows, and we apply this format throughout Section 2.4.4. This will substantially reduce the per-row cost R. Counting words, spaces (absorbed into word tokens), numbers, and delimiters, we obtain:

Model / Tokenizer RNL,dec RTab,dec Ratio GPT(cl100k_base), LLaMA-3 10 5 2.0× Qwen, LLaMA-2 12 7 1.71×

- (B) NUMBER NORMALIZATION. Mapping normalized decimals to the integer range [0,999] preserves the relative rank of decimals while collapsing numbers to a single token under cl100k_base (and 2–3 tokens under Qwen’s tokenizer).

Without delimiters. For example, “0.1234” requires 4 tokens, whereas “234” requires only 1;similarly, “–0.1234” requires 5 tokens, compared to 1 for “234”. Assuming a balanced distribution of positive/negative numbers, the expected token length is (0.5 × 4 + 0.5 × 5) = 4.5 before normalization and 1 after. Thus, the expected per-number reduction under cl100k_base is 4.5×. In Qwen’s tokenizer, positives use 6 tokens vs. 3 after, negatives 7 vs. 3 after; the expected length is (0.5 × 6 + 0.5 × 7) = 6.5 before and 3 after, giving 6.5/3 ≈ 2.17×.

With delimiters. In practice, each number will be followed by a delimiter (a single-token comma), which increases numerator and denominator lengths by +1. Hence for cl100k_base, this yields an expected length of 5.5 before normalization versus 2 after, giving 5.5/2 = 2.75×. For Qwen’s tokenizer, the corresponding expectation is 7.5 before versus 4 after, resulting in 7.5/4 ≈ 1.88×.

These expected compression ratios propagate to corpus-level savings:

Model / Tokenizer w/o delimiters w/ delimiters

GPT(cl100k_base), LLaMA-3 4.5× 2.75× Qwen, LLaMA-2 2.17× 1.88×

- (C) BATCH INFERENCE. Given C(N) = R(MN+N), increasing N amortizes the many-shot block. General form:

C(1) C(N)

N(M + 1) M + N

(independent of tokenizer since R cancels). Derivation for M=1024, from N=1 to N=50:

=

50 × (1024 + 1) (1024 + 50)

C(N = 1) C(N = 50)

= 47.7 × .

=

Overall compression and amortization. Since Stages (A), (B), and (C) are independent, their effects compound multiplicatively. From a compression perspective, (A) structural formatting and (B) number normalization compound multiplicatively to shrink the per-row token cost R, which

- then will allow the LLM context to accommodate more many-shot demonstrations. Hence Compression (a×b):

|5.5×|
|---|

|3.2×|
|---|

GPT(cl100k_base), LLaMA 3: 2.0×2.75 ≈

, Qwen, LLaMA 2: 1.71×1.88 ≈

. Amortization (c, n=50):

|47.7×|
|---|

Notably, the above estimate assumes that the prompt header H contributes negligibly to the overall computation. Beyond this estimation, we leverage our SCM-based synthetic pretraining corpus to

evaluate the amortized token cost per inference sample using both our encoding and TabuLa (Gardner et al., 2024)’s. Under this setup, we observe a 136× improvement using Qwen’s tokenizer.

- 2.5 ORDER-ROBUST, CONFIDENCE-AWARE SELF-CONSISTENCY AT INFERENCE TIME

Our approach is similar to self-consistency proposed by (Wang et al., 2022a), but modifies both the source of diversity and the aggregation method. Instead of generating diverse reasoning paths from identical prompts via stochastic sampling, we create diversity by shuffling in-context demonstrations or features across prompt variants. We then aggregate the model’s responses using weighted majority voting (Littlestone & Warmuth, 1994) to select the most consistent prediction.

More specifically, given a prompt with M in-context demonstrations, we generate V shuffled variants {P0,P1,...,PV −1} and obtain the model’s next-token probabilities for each variant via parsing responses of LLMs. For each query, we extract the corresponding probability of the next token for each possible label yj ∈ {y1,y2,...,yk} from the model’s vocabulary distribution. For prompt variant Pi, let pi(yj) denote the probability assigned by the LLM to the token representing label yj as the next token in the sequence. We compute the aggregate probability for each label by summing across all prompt variants:

p˜(yj) =

V −1

pi(yj) (2)

i=0

The final prediction is then determined by selecting the label with the highest aggregate probability: yˆ = arg maxy

p˜(yj). We set V = 5 in the following experiments—far fewer than the 32 used by TabICL and TabPFN.

j

- 3 EXPERIMENTS

- 3.1 PRELIMINARY

- 3.1.1 EVALUATION DATASETS

We evaluate on the TALENT benchmark (Ye et al., 2024), which comprises 200 classification tasks, several of which include both tabular and textual features. For the main study, we select 32 datasets via domain clustering and sampling, and add 86 more as an extended set. We exclude one dataset whose combined numerical and textual fields exceeded our context-length budget. All tasks are listed in the Appendix. Each dataset is split 80/20 into train/test, and we report accuracy (ACC).

- 3.1.2 BASELINES

Tree-based learners. We include classical tree ensembles as strong tabular baselines: Random Forest (RF) (Breiman, 2001), LightGBM (Ke et al., 2017), XGBoost (Chen & Guestrin, 2016), and CatBoost (Prokhorenkova et al., 2018). In our main comparisons, we report Random Forest because it is stable and robust from few-shot to many-shot regimes and easy to reproduce at scale. We use the same hyperparameters as the RF teacher in the LLM experiments. For Random Forest, we set n_estimators=30, fix random_state for determinism, and use n_jobs=8 (we already parallelize across processes).

Instance-based learner. Recent work by Agarwal et al. (2024) suggests that LLMs can behave similarly to local neural networks when learning linear high-dimensional functions with numerical inputs. Motivated by this observation, we include k-nearest neighbors (kNN) (Cover & Hart, 1967) as a simple instance-based baseline. Specifically, we use n_neighbors=8, weights= “distance”, and the Minkowski distance with p=2.

Tabular ICL models. We compare to TabICL (Qu et al., 2025), a recent in-context learners for tabular data. We run their public ICL-style inference (no gradient updates), which is faster to evaluate and aligns with our in-context setup. We also compare against TabuLa-8B (Gardner et al., 2024), an 8B LLM fine-tuned specifically for tabular prediction.

General-purpose LLMs. For closed-source LLMs, we evaluate GPT-5-mini (latest public snapshot) and the reasoning-oriented o3-mini (OpenAI, 2024; 2025). For open-source LLMs, we use Qwen-2.5-7B-Instruct (Bai et al., 2023) as a strong lightweight baseline. MACHINELEARNINGLM is also built on Qwen-2.5-7B-Instruct via continued pretraining, enabling a direct apples-to-apples comparison under identical prompting.

- 3.1.3 CONTINUED PRETRAINING SETTING INTRODUCTION

We employ the task generator described in Section 2.1 to construct our synthetic pretraining corpus. Our dataset comprises 3 million synthetic tasks, each containing no more than 1,024 samples with feature dimensions ranging from 5 to 50 and up to 10 class labels. Owing to the 32k-token context limit during training, we truncate the number of in-context examples per task during continued pretraining.

We build upon Qwen-7B-Instruct (Bai et al., 2023) as our backbone model, which utilizes rotary positional embeddings. We adopt a two-stage continual pretraining approach, implemented using LoRA (Hu et al., 2022) with rank 8 over attn+MLP and the Adam optimizer (Kingma & Ba, 2014), within the LLaMAFactory framework (Zheng et al., 2024). Training was distributed across 5 nodes with 40 A100 GPUs, achieving a throughput of 300 tokens per second (100s per iteration, with each iteration processing ∼30k tokens). We set lr_scheduler_type as cosine to use the cosine learning rate scheduler throughout training. In the first (warm-up) stage, we set the learning rate to 1 × 10−5 and trained on approximately 1 million tasks over 100 hours. The second stage resumed from the Stage 1 checkpoint with a reduced learning rate of 1×10−6, continuing on about 2 million tasks for 200 hours. We set the batch size per GPU to 1, and updated model parameters every 8 steps. With 40 GPUs, this corresponds to one parameter update per 320 samples.

- 3.1.4 TEST TIME SETTING

For ICL evaluation, we partition test data into chunks of size N, with each chunk serving as a test batch. For each test chunk, we randomly sample M training examples as in-context demonstrations, and we use different sets of M examples for each batch to ensure the results are not biased by a particular fixed set. When generating prompt variants as described in Section 2.5, we shuffle only the order of the M in-context examples while keeping the test examples and their order fixed. For fair comparison with traditional ML models, we randomly sample M training examples for model training and evaluate on the complete test set, as detailed in our public code.

Importantly, although the header row in tabular data could be used to incorporate feature names/descriptions that LLMs could easily leverage (Bordt et al.), our method intentionally excludes such information during training and inference. This design choice promotes a fair comparison with conventional machine learning approaches and mitigates risks of data leakage and memorization (Bordt et al.), while still delivering competitive performance. Additionally, our current tabular encoding strategy jointly models both textual and numerical features, without relying on categorical bucketing or explicit text embedding extraction. The number of S samples for voting is 5 due to computational constraints, while TabICL and TabPFN use 32.

- 3.2 EXPERIMENT RESULTS

- 3.2.1 MACHINELEARNINGLM VS. VANILLA LLMS

As Table 2 and Table 3 show, pretraining solely on synthetic tabular tasks yields large, consistent gains over vanilla LLMs. Averaged across tasks, MACHINELEARNINGLM improves absolute accuracy of the backbone Qwen-2.5-7B-Instruct model by ∼15% (Table 2), with about ∼50% of tasks seeing 15% improvements across finance, biology, vision, speech, robotics, statistics, and healthcare domains (Table 3); virtually no task exhibits a marked degradation (Table 3). At high shot counts (128–1,024), MACHINELEARNINGLM surpasses GPT-5-mini by ∼12% and o3-mini by ∼16% on average (cf. Figure 1c and Table 2). More results on extended tasks are presented in Appendix C.

- Table 2: Test accuracy (%) across number of shots M=8–512. Notably, as shown in Figure 2, a large fraction of training tasks exceed the 32k-token limit when both the number of shots M and number of features K are big; nevertheless, the model exhibits out-ofdistribution generalization to 131k-token contexts. Abbreviations: Qwen-7B=Qwen-2.5-7BInstruct; Ours=MACHINELEARNINGLM (base: Qwen-2.5-7B-Instruct).

Not in-context learning In-context learning Number of shots KNN Random Forest TabICL GPT-5 mini o3-mini Qwen-7B Ours

8 55.3 59.1 57.4 57.8 57.0 51.8 58.4 16 60.1 63.5 64.0 60.0 58.6 53.9 63.1 32 63.4 68.0 68.6 60.2 59.0 55.6 66.7 64 65.9 71.4 72.6 60.4 59.6 57.2 70.0 128 68.1 74.3 76.0 61.0 58.4 58.6 72.0 256 69.5 76.1 79.1 61.7 58.8 59.3 74.3 512 71.1 77.7 80.9 62.5 58.8 60.1 75.3

- 3.2.2 MANY-SHOT SCALING

MACHINELEARNINGLM displays a striking many-shot scaling law: accuracy increases monotonically as we raise the in-prompt demonstrations from 23 to 210 (within a 131k-token inference budget, 4× the 32k-token budget in the pretraining phase), with steady gains across domains (Figure 1c; Table 2). While our model can improve accuracy over 15% from 8 to 512 shots, vanilla LLMs exhibit limited scaling: o3-mini improves by only ∼1.8% (and even declines from 64 to 512 shots), while GPT-5-mini gains merely ∼4.7%. This indicates substantially higher sample efficiency of our approach.

- 3.2.3 MACHINELEARNINGLM VS. TABULA-8B

Thanks to the token-efficient prompting, MACHINELEARNINGLM supports an order of magnitude more demonstrations per prompt than TabuLa-8B (Table 3), which is typically capped by an 8k context (often ≤20–32 shots). In contrast, we scale stably to 1,024 shots for most tasks while maintaining significant many-shot gains, as shown in Appendix B. In the few-shot regime (e.g., 32-shot), MACHINELEARNINGLM also outperforms TabuLa-8B on average—even though TabuLa explicitly exploits feature names— with particularly large gaps on certain datasets (e.g., a ∼14.5% shortfall for vehicle reported by TabuLa).

- 3.2.4 COMPETITIVE MANY-SHOT PERFORMANCE VS. STATE-OF-THE-ART TABULAR-SPECIFIC METHODS

Without any task-specific training, MACHINELEARNINGLM reaches random-forest–level accuracy across M=8–512 shots—typically within 2% relative (Table 2)—and, even at 512 shots, outperforms random forest on roughly 30% of tasks (Table 3). It also clearly surpasses simple instancebased learners (e.g., kNN) by >4% relative on average, indicating robust numerical modeling (Table 2; Table 3).

- 3.2.5 MACHINELEARNINGLM VS. TABULAR ICL MODELS (TABICL / TABPFN)

As Table 2 shows, compared to state-of-the-art tabular ICL models like TabICL, MACHINELEARNINGLM can be modestly behind at very high shot counts—largely due to LLM compute requirements. Unlike tabular architectures with row/column attention and specialized tokenizer (Hollmann et al.; Qu et al., 2025; Su et al., 2024; Wang et al., 2021), MACHINELEARNINGLM uses a general-purpose LLM backbone yet remains order-robust: permuting in-prompt demonstrations leaves performance unchanged on average. Importantly, our method is LLM-compatible by design and thus uniquely positioned to exploit external knowledge, heterogeneous/multimodal inputs, and reasoning-style alignment (e.g., CoT), offering a practical path to close the remaining gap while retaining broad capabilities.

- Table 3: Per-task test accuracy (%) with M ∈ {32,512} shots. Abbreviations: Tabula = TabuLa-8B; Qwen = Qwen-2.5-7B-Instruct; Ours = MACHINELEARNINGLM (base: Qwen-2.5-7B-Instruct). To reduce memorization risks (Bordt et al.), all methods drop feature names/descriptions and encode only values except for TabuLa-8B, because we adopt the paper’s results on the overlapping datasets, using the setting that keeps the original header names. “EL” indicates the prompt exceeds TabuLa-8B’s token limit; “NA” indicates the dataset is not included in TabuLa-8B’s evaluation set.

Task # Shots KNN RF TabICL Tabula GPT-5-mini o3-mini Qwen Ours

Bank 32 87.3 87.6 88.3 84.4 85.4 85.3 87.2 87.3 Bank 512 87.6 89.0 89.4 EL 87.8 86.7 88.1 88.7 BLE_RSSI_Indoor_Loc. 32 33.7 61.5 61.7 NA 59.4 61.2 44.7 63.7 BLE_RSSI_Indoor_Loc. 512 35.1 69.2 73.5 EL 61.4 58.4 52.7 70.3 Churn 32 84.6 86.5 86.6 91.4 82.1 79.1 85.4 86.7 Churn 512 86.6 91.1 92.0 EL 86.1 83.6 86.2 89.5 CMC 32 45.1 45.8 46.1 35.2 42.7 41.4 37.6 44.1 CMC 512 52.2 50.8 55.6 EL 41.0 42.7 50.2 50.2 Contaminant_10_0GHz 32 69.4 72.5 73.3 NA 60.8 53.3 55.6 71.7 Contaminant_10_0GHz 512 78.8 85.6 92.1 EL 60.4 49.4 62.7 82.3 Contaminant_9_5GHz 32 71.9 71.0 71.0 NA 55.0 53.8 53.5 73.8 Contaminant_9_5GHz 512 79.4 84.6 89.2 EL 53.1 48.5 59.4 80.4 Credit_g 32 64.0 69.0 72.0 70.3 70.0 63.5 58.5 71.5 Credit_g 512 70.0 76.0 76.5 EL 71.5 68.0 71.0 71.5 FICO_HELOC_Cleaned 32 59.0 64.7 68.3 NA 54.6 57.6 54.0 63.1 FICO_HELOC_Cleaned 512 64.3 71.5 71.3 EL 54.7 52.6 52.6 69.0 FOREX_AUDCHF_Day 32 50.7 52.3 51.2 EL 48.8 47.9 49.9 51.2 FOREX_AUDCHF_Day 512 44.4 52.6 71.1 EL 51.2 49.0 54.5 49.6 FOREX_AUDJPY_Day 32 45.5 46.6 49.0 NA 46.6 46.3 45.8 46.9 FOREX_AUDJPY_Day 512 57.8 61.3 72.8 EL 54.2 47.4 49.3 62.4 GAMETES_Heterog. 32 47.5 51.3 46.3 NA 52.2 48.8 48.1 49.1 GAMETES_Heterog. 512 57.8 59.7 71.3 EL 52.2 49.1 49.7 52.2 HELOC 32 59.8 66.0 58.7 EL 49.5 51.8 52.1 62.5 HELOC 512 65.6 71.3 72.3 EL 51.4 51.0 53.0 70.8 KC1 32 81.8 79.9 78.4 82.0 78.9 75.8 81.8 78.4 KC1 512 82.5 82.7 84.4 EL 81.9 78.7 83.4 83.9 LED24 32 30.8 41.9 46.4 NA 8.3 14.5 13.1 15.3 LED24 512 55.8 69.4 71.7 EL 9.1 12.3 11.7 48.4 LED7 32 55.9 57.7 58.8 EL 55.9 55.5 32.8 58.4 LED7 512 68.8 68.8 70.3 EL 66.6 57.3 48.4 66.9 Maternal_Health_Risk 32 45.8 52.7 60.1 NA 41.4 52.7 45.3 50.7 Maternal_Health_Risk 512 74.4 78.3 80.0 EL 73.9 63.0 59.6 78.3 PC1 32 92.3 92.3 90.1 89.8 88.7 88.3 85.1 88.7 PC1 512 91.9 93.7 94.1 EL 93.3 89.2 93.2 93.2 Phoneme 32 74.4 75.3 79.9 73.4 71.5 71.9 67.0 73.5 Phoneme 512 83.0 84.9 84.6 EL 69.6 64.3 68.7 82.0 Pima_Indians_Diabetes 32 64.9 74.7 66.9 70.3 69.5 61.0 65.6 71.4 Pima_Indians_Diabetes 512 68.8 75.3 74.7 EL 62.3 68.2 70.8 75.3 RingNorm 32 53.8 79.6 84.2 NA 66.2 75.9 69.2 93.2 RingNorm 512 58.6 93.5 97.0 EL 74.5 75.3 69.3 96.0 RL 32 56.6 58.5 55.8 NA 53.5 51.8 53.9 57.1 RL 512 60.7 66.7 68.5 EL 52.7 54.8 51.8 64.1 Segment 32 63.6 68.2 61.0 EL 60.4 51.5 32.0 64.7 Segment 512 88.1 90.9 93.5 EL 68.6 52.0 52.2 87.2 Seismic_Bumps 32 93.2 92.7 93.0 NA 91.3 86.3 92.5 90.3 Seismic_Bumps 512 92.8 92.8 93.0 EL 90.7 86.5 93.0 90.7 Statlog 32 61.0 65.0 64.0 NA 60.0 61.0 57.5 63.0 Statlog 512 67.0 74.0 72.5 EL 58.0 63.0 63.0 67.0 Thyroid 32 92.6 92.8 93.3 NA 92.3 91.2 92.6 92.8 Thyroid 512 94.0 98.0 98.8 EL 92.6 92.5 92.6 95.0 TwoNorm 32 94.4 88.8 95.9 NA 88.2 96.0 62.9 95.0 TwoNorm 512 97.0 95.9 97.6 EL 87.6 93.1 74.9 97.5 Vehicle 32 52.3 60.0 70.0 48.4 47.1 24.7 32.9 62.9 Vehicle 512 64.7 74.7 84.7 EL 41.2 27.1 32.4 72.9 WallRobot_Navigation 32 51.4 74.2 70.1 NA 43.0 35.8 42.0 60.1 WallRobot_Navigation 512 73.4 96.7 94.1 EL 48.9 35.8 45.2 83.3 Waveform 32 74.1 72.7 76.0 NA 44.2 40.5 38.6 72.5 Waveform 512 81.6 82.7 85.1 EL 32.4 33.0 43.5 84.0 Wine 32 63.8 64.6 68.7 NA 61.6 64.6 52.0 64.8 Wine 512 65.6 69.7 75.3 EL 56.2 56.4 56.2 71.0 Yeast 32 43.1 41.8 41.8 NA 36.7 40.4 33.7 41.8 Yeast 512 55.6 58.2 59.9 EL 42.8 36.7 32.3 59.3

- Table 4: MMLU results for MACHINELEARNINGLM and baselines. (a) Macro accuracy across M-shot; (b) per-subject accuracies at 50-shot.

Number of shots Qwen-2.5-7B-Instruct Qwen-2.5-7B TabuLa-8B MACHINELEARNINGLM

0 73.8 73.5 61.6 73.2 10 75.9 75.5 65.2 75.1 50 75.8 75.4 N/A 75.4

(a) Macro accuracy on the full MMLU benchmark across different M-shot settings at temperature = 0. TabuLa-8B (Gardner et al., 2024) uses a maximum 8k-token length.

(b) Per-subject accuracies at 50-shot. Subject Qwen-2.5-7B-Inst. Qwen-2.5-7B TabuLa-8B MACHINELEARNINGLM

high_school_statistics 73.6 69.9 59.3 74.1 high_school_physics 62.9 55.6 47.7 61.6 astronomy 86.8 86.2 68.4 85.5 college_math 47.0 54.0 32.0 49.0 college_physics 53.9 61.8 53.9 57.8 conceptual_physics 74.0 74.0 62.1 76.2 econometrics 69.3 64.9 49.1 65.8 elementary_math 72.8 74.1 42.9 71.4 high_school_math 55.6 56.3 32.6 55.2

All accuracies (%) are reported using 50 shots, except for TabuLa-8B, which suggests a maximum sequence length of 8k and therefore uses a 20-shot setting. Models are evaluated with temperature=0.05, votes=3.

- 3.2.6 MACHINELEARNINGLM IN GENERAL CHAT WORKFLOWS

As shown in Table 4, general abilities are well preserved. On MMLU, our model achieves ∼73.2% (0-shot) and ∼75.4% (50-shot), comparable to strong general-purpose LLMs. Notably, we observe consistent gains in numeracy-heavy subjects (e.g., high-school statistics/conceptual physics). Although training LLMs on real tables, TabuLa-8B underperforms on MMLU (by ∼10%), underscoring that our training strategy of reusing the model architecture and tokenizer integrates smoothly with general LLM capabilities and remains compatible with future multimodal/heterogeneous extensions.

- 3.3 EMPIRICAL STUDY

Numeric-dominant tables. On pure numeric tables (e.g., churn, maternal_health_risk, waveform, twonorm, vehicle, yeast, heloc), our model remains competitive, indicating that the same prompt design handles both modalities without architecture changes.

Heterogeneous (text+numeric) tables. Our tabular encoding natively mixes free-text/categorical fields with numbers—without mandatory text bucketing or learned text embeddings—yielding reliable gains on heterogeneous schemas. On mixed-feature datasets such as bank, adult, credit-g, online_shoppers, Bank_Customer_Churn_Dataset, and okcupid_stem, MACHINELEARNINGLM consistently outperforms vanilla LLMs. By contrast, for highly symbolic/abstract textual fields that behave more like token sequences than NL (e.g., DNA base strings in the UCI splice dataset), MACHINELEARNINGLM falls short relative to numerical modeling methods such as Random Forests.

High-cardinality labels. On tasks with many classes (e.g., kropt, letter (26-way), walking_activity), MACHINELEARNINGLM underperforms strong tabular baselines (RF/TabICL). We attribute this to pretraining that sampled tasks with K ≤ 10 classes, biasing the decoder toward small label vocabularies; when K > 10, labels may become multi-token (e.g., “12”). Expanding K in synthesis is a promising mitigation.

Resilience to class imbalance. Across the task suite, MACHINELEARNINGLM remains stable under skewed label distributions: its many-shot accuracy tracks Random-Forest–level perfor-

mance while avoiding collapse-to-majority. On notably imbalanced datasets—e.g., bank, pc1, and kc1—it sustains competitive accuracy.

Limits on forecasting-style tasks. On time-series/forecasting tasks—particularly the FOREX variants (FOREX task series)—we observe significant declines relative to tabular methods. This gap is expected: our pretraining targets i.i.d. tabular prediction, whereas forecasting requires temporal inductive bias (lagged context, trend/seasonality priors) and higher numeric precision. We view this as an opportunity: future work will incorporate a time-aware strategy to better align MACHINELEARNINGLM with forecasting workloads. More limitations are detailed in Section 5.

- 4 RELATED WORK

Many-shot ICL and scaling laws. A growing literature observes that simply adding more demonstrations often yields diminishing or even negative returns in LLMs—e.g., one or a few high-quality demonstrations capture most gains (Chen et al.; Min et al., 2022; Liu et al., 2022; Pan, 2023), long-context usage is position-sensitive such as “lost in the middle” (Liu et al., 2024a; An et al.; Xiong et al., 2024), and many-shot gains can plateau in multimodal VLMs such as Flamingo beyond ∼32 shots (Alayrac et al., 2022; Tai et al., 2024). Recent efforts improve many-shot ICL via specialized training or mechanisms: DeepMind’s study shows task-dependent trends and proposes reinforced/unsupervised ICL variants (Agarwal et al., 2024) to mitigate many-shot scaling; DrICL reweights demonstrations and modifies objectives to mitigate plateauing (Zhang et al., 2025); semisupervised many-shot ICL mitigates many-shot scaling when using self-generated annotations (Gu et al.); multimodal works report mixed but often dataset-specific monotonic gains with longer contexts (Jiang et al.). Moreover, long-context evaluation disentangles retrieval-like tasks from global context understanding, with many models degrading on the latter at 16k tokens (Zou et al., 2025). TabDPT reports scaling laws with respect to both model size and pretraining-corpus size Ma et al. (2024). Unlike these approaches, MACHINELEARNINGLM induces many-shot ICL through pretraining on millions of synthetic tabular tasks, yielding monotonic example-count scaling while preserving the general reasoning of the base LLM.

LM-based ML learner. Compared with TabLLM (Hegselmann et al., 2023)—an approach that demands computationally intensive, task-specific fine-tuning and hyper-parameter tuning—MACHINELEARNINGLM adapts to new tabular tasks purely through ICL at inference time, with no gradient updates. Crucially, it delivers markedly stronger reasoning over large volumes of numerical data, a long-standing weakness of TabLLM 6. Most relevant to us, (Gardner et al., 2024) proposes TabuLa-8B—finetuning Llama 3-8B on a web-scale corpus (T4) distilled from TabLib, and introducing a row-causal tabular masking (RCTM) scheme that packs samples by table and encourages few-shot behavior (Gardner et al., 2024). Their main evaluations focus on k∈[0,32] shots. Our synthetic pretraining complements real-world data pretraining and differs in three ways: (i) we continue pretraining a general-purpose LLM without architectural modifications without losing the backbone’s general capabilities, (ii) we pretrain on synthetic, SCM-driven tabular tasks with explicitly controlled diversity, ensuring no dataset leakage from downstream evaluations, and (iii) we target many-shot ICL behavior on tabular tasks (1,024 demonstrations under a fixed context budget) using our proposed token-efficient method.

In-context tabular ML learners. TabPFN frames tabular classification as ICL with a pre-trained transformer hypernetwork and achieves strong few-shot accuracy on small tables (Hollmann et al.). TabICL scales PFN-style ICL to much larger tables via a two-stage, column-then-row architecture and synthetic pretraining (Qu et al., 2025). However, these tabular ICL models are trained independently of language models—incompatible with LM architectures or checkpoints—so they lack general text reasoning and open-domain knowledge, which limits application when textual fields carry meaning and constrains performance on multimodal (text–numeric) tasks. In contrast, MACHINELEARNINGLM is built upon a pre-trained LLM backbone and therefore combines robust numeric processing with the ability to interpret textual headers, free-text cells, and world knowledge priors, offering a unified foundation model for heterogeneous, real-world in-context ML tasks.

6https://gael-varoquaux.info/science/carte-toward-table-foundation-models. html

Tool-using agents for ML vs. in-context learners. A complementary line of work evaluates or builds LLM agents that call external ML toolchains (e.g., LightGBM/XGBoost/CatBoost, AutoML) to solve end-to-end ML engineering tasks, as in MLE-Bench (Chan et al., 2024), MLAgentBench (Huang et al., 2024), R&D-Agent (Yang et al., 2025), and ML-Master (Anonymous, 2025). However, their performance is bound by the invoked learners and pipelines. In contrast, MACHINELEARNINGLM explores a way to internalize the learning procedure: the model performs in-context prediction, requiring neither per-task fine-tuning nor calls to external ML models, and MACHINELEARNINGLM can also be called by the MLE agent as a tool.

- 5 LIMITATIONS

While MACHINELEARNINGLM shows strong in-context ML performance, several limitations remain.

- • Task scope. Our pretraining corpus focuses on tabular classification synthesized from SCMs. We do not yet cover regression, ranking, time–series forecasting, or structured prediction, and we cap the number of classes (K ≤10). Extending beyond IID rows (e.g., temporal or relational dependencies) is future work.
- • Context length and compute. Continued pretraining used a 32k-token context; although inference generalizes beyond this (tested up to ∼131k tokens), scaling to many thousands of shots like TabPFN/TabICL remains challenging due to compute and memory costs.
- • Numerical encoding trade-offs. The [0,999] integer mapping preserves order but coarsens magnitudes and can obscure semantically meaningful constants (e.g., age 18). As context windows grow, more expressive number encodings can be explored to retain such cues while remaining token-efficient.
- • Warm-start bias. The RF-mimic warm-up and example-level consensus filtering may bias early training toward tree-like decision boundaries and easy examples. Although we disable consensus after warm-up, some inductive bias may persist; we currently distill labels only (not rationales). How to best leverage external teachers is still an open question.
- • Model scale and adaptation. Results are with a 7B backbone and low-rank adaptation (LoRA rank 8). Larger backbones, alternative adapters, or optimizer/regularization choices may further improve many-shot numeracy but were not explored here.

- 6 FUTURE WORK

We hope MACHINELEARNINGLM will ignite new lines of research, as exemplified below. Feel free

- to contact us for further discussions .

- • Beyond text and numbers: synthetic tasks with multimodal features. Starting from our synthetic tabular tasks, map numerical/categorical features to weakly labeled multimodal surrogates—short texts, speech snippets, images, geospatial signals, or time series—using generative back-ends (LLM/VLM/TTS) conditioned on latent factors to preserve coarse class semantics. This produces controllable, cross-modal pairs aligned with the tabular schema 7 and enables MACHINELEARNINGLM to practice multimodal in-context prediction on heterogeneous signals.
- • Longer contexts via parallelism and system optimizations. Extend context length using tensor/pipeline parallelism and memory-efficient attention/KV caching to support substantially more in-prompt examples. Recent work on cache-augmented architectures demonstrates that principled KV-cache management can significantly improve efficiency and scalability for long-context reasoning (Bhaskar et al., 2025), suggesting promising directions to integrate such mechanisms into MACHINELEARNINGLM.
- • Toward interpretability: narrative distillation and reasoning-augmented learning. Each task in our pretraining corpus is generated from a structural causal model (SCM),

7Table formats (e.g., HTML tables) naturally organize images, hyperlinks, and embedded objects.

which we treat as an intrinsic source of explanation. We train the LM to narrate the underlying SCM—variables, relations, and mechanisms—as an auxiliary objective aligned with the prediction task, thereby yielding more faithful rationales and stronger reasoning. Moreover, couple feature attribution signals (e.g., SHAP (Lundberg & Lee, 2017) / LIME (Ribeiro et al., 2016)) with MACHINELEARNINGLM’s generation to produce faithful instance- and cohort-level narratives, while distilling rules from ensembles (e.g., inTrees (Deng, 2014) / RuleFit (Friedman & Popescu, 2008)) into compact, human-readable reasoning traces. Recent advances such as DeepSeek-R1 (Guo et al., 2025) demonstrate the promise of incentivizing reasoning capability in LLMs through RL. This integration promises not only accurate predictions but also transparent rationales, enhancing compliance and trust.

- • Uncertainty-aware responses. Recent analyses from OpenAI argue that training/evaluation schemes which penalize uncertainty and implicitly reward guessing are a root cause of hallucinations 8. We suspect a similar incentive mismatch in our setting. Future work may (i) train on teacher predictive distributions—e.g., randomized/ensemble teachers that provide calibrated confidences—instead of hard labels, optimized with proper scoring rules (NLL/Brier) and selective-prediction losses that allow abstention; (ii) explicitly represent and reward well-calibrated uncertainty in the output schema (e.g., an UNCERTAIN/defer option with risk–coverage evaluation); and (iii) directly optimize task-level metrics (AUROC, RMSE, F1) via reinforcement learning or differentiable surrogates, with robust reward shaping (e.g., clipped/Huberized or quantile-based rewards) to reduce outlier sensitivity and improve end-to-end utility.
- • Combined with Retrieval-Augmented Methods. Retrieval-augmented method has shown that attaching a retrieval module to LLMs enables scalable any-shot learning and yields power-law error reductions as data grows (Wen et al., 2025). Integrating such retrieval into MACHINELEARNINGLM can boost many-shot example limits by dynamically injecting the most relevant examples during pretraining and inference.
- • Real-data alignment via continued fine-tuning. Beyond purely synthetic pretraining, we will perform lightweight continued fine-tuning on a curated suite of real tabular prediction tasks to better align with practical distributions.
- • Extending MACHINELEARNINGLM to fully leverage Agent Memory. Recent advances leverage memory mechanisms so that agents can recall and reuse successful or similar past experiences to inform their next action (Wang et al., 2023; Lin et al., 2025; Wang et al., b). Such experiences may be textual traces, numerical features, or temporal signals. By pretraining the policy model with many-shot in-context experiences, the agent can improve decision-making and robustness in dynamic environments.
- • Integration with MLE agents. Fuse MACHINELEARNINGLM with MLE-agent–style planners to orchestrate end-to-end MLE workflows. The agent treats MACHINELEARNINGLM both as a solver and as an internal tool, enabling a flexible interface with heterogeneous real-world MLE tasks.

- 7 CONCLUSION

We introduced MACHINELEARNINGLM, a portable continued-pretraining recipe that equips a general-purpose LLM with robust in-context tabular prediction without architectural or tokenizer changes. By synthesizing millions of SCM-driven tasks and using a Random-Forest warm start, the model acquires strong numerical modeling while preserving general knowledge and reasoning. A token-efficient tabular encoding, integer-based number normalization, and sequence-level batchprediction together expand the effective context budget and amortize prompt overhead. Across diverse out-of-distribution classification tasks, MACHINELEARNINGLM remarkably improves over its backbone and approaches random-forest–level accuracy from 8 to 512 shots; accuracy further increases up to 1,024 shots and is robust to exemplar-order permutations. These results indicate that targeted continued pretraining on synthetic prediction tasks is a practical path to scaling many-shot ICL within general-purpose LLMs. Finally, we discuss the current limitations and chart several promising avenues for future work.

8https://openai.com/index/why-language-models-hallucinate/

AUTHOR CONTRIBUTIONS

Conceptualization (LLM continued pretraining with SCM-based synthesis, RF-mimic warm-start, token-efficient prompting, and order-robust self-consistency): Haoyu Dong9.

Methodology: Pretraining recipe and objectives — Haoyu Dong, Pengkun Zhang10; SCM task synthesis — Pengkun Zhang, Haoyu Dong; RF-mimic warm-start (gating and consensus filtering) — Haoyu Dong; token-efficient prompting — Haoyu Dong; self-consistency — Yanzhen Shen11, Mingzhe Lu12, Haoyu Dong.

Software & Engineering: Pretraining pipeline and training scripts — Pengkun Zhang, Haoyu Dong; evaluation harness — Yanzhen Shen, Mingzhe Lu, Haoyu Dong; SCM generator and data pipelines — Pengkun Zhang, Haoyu Dong; token-efficient prompting utilities — Haoyu Dong; code optimization, cross-platform support, and parallelization — Mingzhe Lu, Pengkun Zhang.

Investigation & Validation: Experiments and statistical aggregation of results — Mingzhe Lu, Pengkun Zhang, Haoyu Dong.

Writing: Original draft (main text, tables, and figures) — Haoyu Dong; review and editing Yanzhen Shen, Pengkun Zhang, Mingzhe Lu, Guolin Ke13.

Limitations & Future Work: Haoyu Dong, Guolin Ke. ML Expertise: High-level guidance on ML practice, interpretability, and online/continual learning directions — Guolin Ke. Moreover, we thank Hui Xue for suggestions on method and evaluation, including adding the MMLU benchmark and future directions that leverage random-forest predictive probabilities.

REFERENCES

Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al. Many-shot in-context learning. Advances in Neural Information Processing Systems, 37:76930–76966, 2024.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716– 23736, 2022.

Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, Jingjing Xu, and Lingpeng Kong. Why does the effective context length of llms fall short? In The Thirteenth International Conference on Learning Representations.

Anonymous. Ml-master: Towards ai-for-ai via integration of exploration and reasoning. arXiv preprint arXiv:2506.16499, 2025. URL https://arxiv.org/abs/2506.16499.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Adithya Bhaskar, Alexander Wettig, Tianyu Gao, Yihe Dong, and Danqi Chen. Cache me if you can: How many kvs do you need for effective long-context lms? arXiv preprint arXiv:2506.17121, 2025.

Sebastian Bordt, Harsha Nori, Vanessa Cristiny Rodrigues Vasconcelos, Besmira Nushi, and Rich Caruana. Elephants never forget: Memorization and learning of tabular data in large language models. In First Conference on Language Modeling.

9University of Chinese Academy of Sciences, Microsoft, donghaoyu82@gmail.com

- 10South China University of Technology, sezhangpengkun@mail.scut.edu.cn
- 11Stanford University, yanzhen4@stanford.edu 12University of Chinese Academy of Sciences, lumingzhe23@mails.ucas.edu.cn 13Individual, guolin.ke@outlook.com

Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,

Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Jun Shern Chan et al. Mle-bench: Evaluating machine learning agents on real-world competitions. arXiv preprint arXiv:2410.07095, 2024. URL https://arxiv.org/abs/2410.07095.

Jiuhai Chen, Lichang Chen, Chen Zhu, and Tianyi Zhou. How many demonstrations do you need for in-context learning? In The 2023 Conference on Empirical Methods in Natural Language Processing.

Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In KDD, pp. 785–794, 2016.

Zhoujun Cheng, Jungo Kasai, and Tao Yu. Batch prompting: Efficient inference with large language model apis. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 792–810, 2023.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Thomas M. Cover and Peter E. Hart. Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1):21–27, 1967.

Houtao Deng. Interpreting tree ensembles with intrees. arXiv preprint arXiv:1408.5456, 2014.

Haoyu Dong, Jianbo Zhao, Yuzhang Tian, Junyu Xiong, Shiyu Xia, Mengyu Zhou, Yun Lin, José Cambronero, Yeye He, Shi Han, et al. Spreadsheetllm: encoding spreadsheets for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 20728–20748, 2024.

Yu Fei, Yifan Hou, Zeming Chen, and Antoine Bosselut. Mitigating label biases for in-context learning. In The 61st Annual Meeting Of The Association For Computational Linguistics, 2023.

Jerome H. Friedman and Bogdan E. Popescu. Predictive learning via rule ensembles. The Annals of Applied Statistics, 2(3):916–954, 2008. doi: 10.1214/07-AOAS148.

Josh Gardner, Juan C Perdomo, and Ludwig Schmidt. Large scale transfer learning for tabular data via language modeling. Advances in Neural Information Processing Systems, 37:45155–45205, 2024.

Zhengyao Gu, Henry Peng Zou, Aiwei Liu, Yankai Chen, Weizhi Zhang, and Philip S Yu. Scaling laws for many-shot in-context learning with self-generated annotations. In ICML 2025 Workshop on Long-Context Foundation Models.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. Tabllm: Few-shot classification of tabular data with large language models. In International conference on artificial intelligence and statistics, pp. 5549–5581. PMLR, 2023.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, 2025.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. Mlagentbench: evaluating language agents on machine learning experimentation. In Proceedings of the 41st International Conference on Machine Learning, pp. 20271–20309, 2024.

Hugging Face. Number tokenization blog, 2024. URL https://huggingface.co/spaces/ huggingface/number-tokenization-blog. Hugging Face Spaces; last updated Nov 29, 2024.

Yixing Jiang, Jeremy Andrew Irvin, Ji Hun Wang, Muhammad Ahmed Chaudhry, Jonathan H Chen, and Andrew Y Ng. Many-shot in-context learning in multimodal foundation models. In ICML 2024 Workshop on In-Context Learning.

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and TieYan Liu. Lightgbm: A highly efficient gradient boosting decision tree. In NeurIPS, 2017.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Jianzhe Lin, Maurice Diesendruck, Liang Du, and Robin Abraham. Batchprompt: Accomplish more with less. In The Twelfth International Conference on Learning Representations.

Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Daxin Jiang, Binxing Jiao, Chen Hu, et al. Se-agent: Self-evolution trajectory optimization in multi-step reasoning with llm-based agents. arXiv preprint arXiv:2508.02085, 2025.

Nick Littlestone and Manfred K Warmuth. The weighted majority algorithm. Information and computation, 108(2):212–261, 1994.

Jiachang Liu, Dinghan Shen, Yizhe Zhang, William B Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pp. 100–114, 2022.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 11:157–173, 2024a.

Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, and Han-Jia Ye. Talent: A tabular analytics and learning toolbox. arXiv preprint arXiv:2407.04057, 2024b.

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Hamidreza Kamkari, Alex Labach, Jesse C Cresswell, Keyvan Golestan, Guangwei Yu, Maksims Volkovs, and Anthony L Caterini. Tabdpt: Scaling tabular foundation models. arXiv preprint arXiv:2410.18164, 2024.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11048–11064, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. emnlp-main.759. URL https://aclanthology.org/2022.emnlp-main.759/.

Martin Mráz, Breenda Das, Anshul Gupta, Lennart Purucker, and Frank Hutter. Towards benchmarking foundation models for tabular data with text. arXiv preprint arXiv:2507.07829, 2025.

Mateusz Ochal, Massimiliano Patacchiola, Jose Vazquez, Amos Storkey, and Sen Wang. Few-shot learning with class imbalance. IEEE Transactions on Artificial Intelligence, 4(5):1348–1358, 2023.

OpenAI. Openai changelog: o3-mini released. https://platform.openai.com/docs/ changelog, 2024. Lists the o3-mini reasoning model.

OpenAI. Openai o3 and o4-mini system card. https://openai.com/systems/o3/, 2025. System card covering the o3 family and o4-mini.

Jane Pan. What in-context learning “learns” in-context: Disentangling task recognition and task

learning. Master’s thesis, Princeton University, 2023. Judea Pearl. Causality. Cambridge university press, 2009. Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Elements of causal inference: foundations

and learning algorithms. The MIT press, 2017.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. Catboost: Unbiased boosting with categorical features. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, 2018.

Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. Tabicl: A tabular foundation model for in-context learning on large data. In ICML 2025-Forty-Second International Conference on Machine Learning, 2025.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "why should i trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pp. 1135–1144, 2016.

Xingjian Shi, Jonas Mueller, Nick Erickson, Mu Li, and Alexander J Smola. Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705, 2021.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Aaditya K Singh and DJ Strouse. Tokenization counts: the impact of tokenization on arithmetic in frontier llms. arXiv preprint arXiv:2402.14903, 2024.

Georgios Spithourakis and Sebastian Riedel. Numeracy for language models: Evaluating and improving their ability to predict numbers. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2104–2115, 2018.

Aofeng Su, Aowen Wang, Chao Ye, Chen Zhou, Ga Zhang, Gang Chen, Guangcheng Zhu, Haobo Wang, Haokai Xu, Hao Chen, et al. Tablegpt2: A large multimodal model with tabular data integration. arXiv preprint arXiv:2411.02059, 2024.

Yuan Sui, Mengyu Zhou, Mingjie Zhou, Shi Han, and Dongmei Zhang. Table meets llm: Can large language models understand structured table data? a benchmark and empirical study. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM), 2024. doi: 10.1145/3616855.3635752.

Yan Tai, Weichen Fan, Zhao Zhang, Feng Zhu, Rui Zhao, and Ziwei Liu. Link-context learning for multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. URL https://openaccess.thecvf.com/content/ CVPR2024/papers/Tai_Link-Context_Learning_for_Multimodal_LLMs_ CVPR_2024_paper.pdf.

Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, and Matt Gardner. Do NLP models know numbers? probing numeracy in embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 5307–5315, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1534. URL https://aclanthology.org/D19-1534/.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, a.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022a.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705, 2022b.

Zhiruo Wang, Haoyu Dong, Ran Jia, Jia Li, Zhiyi Fu, Shi Han, and Dongmei Zhang. Tuta: Treebased transformers for generally structured table pre-training. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 1780–1790, 2021.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In Forty-second International Conference on Machine Learning, b.

Xumeng Wen, Shun Zheng, Zhen Xu, Yiming Sun, and Jiang Bian. Scalable in-context learning on tabular data via retrieval-augmented large language models. arXiv preprint arXiv:2502.03147, 2025.

Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. Effective long-context scaling of foundation models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4643–4663, 2024.

Xu Yang, Xiao Yang, Shikai Fang, Bowen Xian, Yuante Li, Jian Wang, Minrui Xu, Haoran Pan, Xinpeng Hong, Weiqing Liu, et al. R&d-agent: Automating data-driven ai solution building through llm-powered automated research, development, and evolution. arXiv preprint arXiv:2505.14738, 2025.

Han-Jia Ye, Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, and De-Chuan Zhan. A closer look at deep learning methods on tabular datasets. arXiv preprint arXiv:2407.00956, 2024.

Xiaoqing Zhang, Ang Lv, Yuhan Liu, Flood Sung, Wei Liu, Jian Luan, Shuo Shang, Xiuying Chen, and Rui Yan. More is not always better? enhancing many-shot in-context learning with differentiated and reweighting objectives. In Proceedings of ACL 2025 (Long Papers), 2025. URL https://aclanthology.org/2025.acl-long.1475.pdf.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In International conference on machine learning, pp. 12697–12706. PMLR, 2021.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models, 2024. URL https: //arxiv.org/abs/2403.13372.

Kaijian Zou, Muhammad Khalifa, and Lu Wang. On many-shot in-context learning for long-context evaluation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 25605–25639, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1245. URL https://aclanthology.org/2025.acl-long.1245/.

- A LIST OF SYMBOLS

Symbol Meaning Range/Default

G DAG of the structural causal model (SCM) v Node index in G —

xv Variable value at node v fv Structural function at node v εv i.i.d. noise term —

- M # in-context demonstrations (shots) ≤ 1024
- N # query/test rows predicted per batched sequence 50 d Feature dimension (columns) 5 ∼ 50

K # classes 2 ∼ 10 S Demonstration set {si}Mi=1 Q Query set {x(i)}Ni=1 y Target JSON array of predictions ℓˆi Predicted label for query i —

tok(·) Tokenizer T Target token length θ Model parameters —

L(θ) Training loss (NLL) H Token cost of the instruction/schema header R Per-row token cost (demo or query) C Amortized token cost per predicted label z z-score normalized numeric value i Integer after [0, 999]-norm mapping —

p0 Conservative random baseline accuracy —

Ncorrect # correct predictions by RF per batched sequence α One-sided binomial test threshold 0.2 κ Cohen’s κ (chance-corrected agreement) > 0.01

δbacc Balanced-accuracy margin 0.03

δF1 Macro-F1 margin 0.00 τdom Max dominant-class fraction 0.95

V # shuffled prompt variants 5

pi(yj) Next-token prob. for label yj under variant Pi p˜(yj) Aggregated prob.: Vi=0−1 pi(yj) —

- B MANY-SHOT SCALING PER-TASK EVALUATION

- Table 6: Per-dataset test accuracy from 8 to 1,024 shots on more sampled domain-representative datasets in Talent. Abbreviations: Qwen-2.5-7B = Qwen-2.5-7B-Instruct; Our = MACHINELEARNINGLM (base: Qwen-2.5-7B-Instruct).

Dataset Shots KNN RF Qwen-2.5-7B Our

bank 8 0.8657 0.8695 0.8526 0.8704 bank 16 0.8723 0.8759 0.8666 0.8707 bank 32 0.8725 0.8764 0.8722 0.8727 bank 64 0.8764 0.8828 0.8780 0.8771 bank 128 0.8778 0.8847 0.8798 0.8830 bank 256 0.8757 0.8873 0.8803 0.8867 bank 512 0.8762 0.8897 0.8808 0.8870 bank 1,024 0.8801 0.8947 0.8795 0.8903 BLE_RSSI_dataset_for_Indoor_localization 8 0.3385 0.5167 0.3956 0.5328 BLE_RSSI_dataset_for_Indoor_localization 16 0.3525 0.5884 0.4382 0.5889 BLE_RSSI_dataset_for_Indoor_localization 32 0.3370 0.6154 0.4467 0.6375 BLE_RSSI_dataset_for_Indoor_localization 64 0.3360 0.6560 0.4567 0.6715 BLE_RSSI_dataset_for_Indoor_localization 128 0.3400 0.6960 0.4852 0.6950 BLE_RSSI_dataset_for_Indoor_localization 256 0.3460 0.6865 0.5153 0.7020 BLE_RSSI_dataset_for_Indoor_localization 512 0.3515 0.6915 0.5268 0.7031 BLE_RSSI_dataset_for_Indoor_localization 1,024 0.3375 0.6955 0.5238 0.7126 churn 8 0.8260 0.8460 0.7970 0.8430 churn 16 0.8530 0.8660 0.8550 0.8620 churn 32 0.8460 0.8650 0.8540 0.8670 churn 64 0.8660 0.8710 0.8630 0.8650 churn 128 0.8650 0.8760 0.8640 0.8680 churn 256 0.8660 0.8940 0.8640 0.8870 churn 512 0.8660 0.9110 0.8620 0.8950 churn 1,024 0.8680 0.9330 0.8670 0.8960 cmc 8 0.3661 0.3492 0.3559 0.3356 cmc 16 0.4000 0.3492 0.3492 0.3492 cmc 32 0.4508 0.4576 0.3763 0.4407 cmc 64 0.4577 0.4847 0.4034 0.4780 cmc 128 0.4441 0.4881 0.4237 0.4746 cmc 256 0.4848 0.5119 0.3966 0.4915 cmc 512 0.5220 0.5085 0.4271 0.5017 cmc 1,024 0.4882 0.5017 0.4508 0.4949 Contaminant_10_0GHz 8 0.5479 0.5750 0.5062 0.5708 Contaminant_10_0GHz 16 0.6792 0.6625 0.5479 0.6833 Contaminant_10_0GHz 32 0.6937 0.7250 0.5563 0.7167 Contaminant_10_0GHz 64 0.7084 0.7375 0.5833 0.7604 Contaminant_10_0GHz 128 0.7354 0.8083 0.5979 0.7688 Contaminant_10_0GHz 256 0.7333 0.8292 0.6146 0.8125 Contaminant_10_0GHz 512 0.7875 0.8562 0.6271 0.8230 Contaminant_10_0GHz 1,024 0.8042 0.8750 0.5708 0.8230 Contaminant_9_5GHz 8 0.5354 0.6313 0.5042 0.6250 Contaminant_9_5GHz 16 0.6542 0.6646 0.5396 0.6771 Contaminant_9_5GHz 32 0.7188 0.7104 0.5354 0.7375 Contaminant_9_5GHz 64 0.7188 0.7500 0.5750 0.7375 Contaminant_9_5GHz 128 0.7354 0.7834 0.5875 0.7563 Contaminant_9_5GHz 256 0.7584 0.8208 0.6208 0.7875 Contaminant_9_5GHz 512 0.7937 0.8458 0.5938 0.8042 Contaminant_9_5GHz 1,024 0.8375 0.8667 0.5583 0.8042 credit_g 8 0.6050 0.6900 0.6300 0.7050 credit_g 16 0.6750 0.6850 0.6650 0.7100 credit_g 32 0.6400 0.6900 0.5850 0.7150 credit_g 64 0.6450 0.6600 0.6100 0.6850 credit_g 128 0.6750 0.7400 0.6150 0.7200 credit_g 256 0.6700 0.7650 0.6750 0.7500 credit_g 512 0.7000 0.7600 0.7100 0.7150

Table 6 – continued from previous page

##### Dataset Shots KNN RF Qwen-2.5-7B Our

credit_g 1,024 0.6700 0.7900 0.6650 0.7600 FICO_HELOC_cleaned 8 0.5458 0.5757 0.5175 0.5544 FICO_HELOC_cleaned 16 0.5620 0.6192 0.5094 0.6253 FICO_HELOC_cleaned 32 0.5899 0.6466 0.5403 0.6309 FICO_HELOC_cleaned 64 0.6147 0.6795 0.5129 0.6785 FICO_HELOC_cleaned 128 0.6258 0.7018 0.5215 0.6906 FICO_HELOC_cleaned 256 0.6390 0.7043 0.5276 0.7089 FICO_HELOC_cleaned 512 0.6430 0.7149 0.5256 0.6901 FICO_HELOC_cleaned 1,024 0.6512 0.7271 0.5362 0.6112 FOREX_audchf_day_High 8 0.5150 0.5123 0.5150 0.5013 FOREX_audchf_day_High 16 0.5095 0.5150 0.5422 0.5014 FOREX_audchf_day_High 32 0.5068 0.5232 0.4986 0.5123 FOREX_audchf_day_High 64 0.5341 0.5286 0.5422 0.4932 FOREX_audchf_day_High 128 0.4932 0.5259 0.5450 0.5232 FOREX_audchf_day_High 256 0.4768 0.5041 0.5613 0.5422 FOREX_audchf_day_High 512 0.4441 0.5259 0.5450 0.4959 FOREX_audchf_day_High 1,024 0.4795 0.6240 0.5395 0.5150 FOREX_audjpy_day_High 8 0.4986 0.4387 0.4496 0.4604 FOREX_audjpy_day_High 16 0.4632 0.4850 0.4550 0.4796 FOREX_audjpy_day_High 32 0.4550 0.4659 0.4578 0.4687 FOREX_audjpy_day_High 64 0.4796 0.5068 0.4796 0.4823 FOREX_audjpy_day_High 128 0.5041 0.5259 0.4959 0.5368 FOREX_audjpy_day_High 256 0.5341 0.5395 0.4659 0.5340 FOREX_audjpy_day_High 512 0.5777 0.6131 0.4932 0.6240 FOREX_audjpy_day_High 1,024 0.5531 0.6185 0.4877 0.5422 GAMETES_Heterogeneity 8 0.5281 0.5156 0.4875 0.5281 GAMETES_Heterogeneity 16 0.4875 0.5219 0.4625 0.5531 GAMETES_Heterogeneity 32 0.4750 0.5125 0.4813 0.4906 GAMETES_Heterogeneity 64 0.4656 0.5250 0.4656 0.4687 GAMETES_Heterogeneity 128 0.5125 0.4781 0.4906 0.4750 GAMETES_Heterogeneity 256 0.5437 0.5812 0.4844 0.5281 GAMETES_Heterogeneity 512 0.5781 0.5969 0.4969 0.5093 GAMETES_Heterogeneity 1,024 0.5906 0.6343 0.4938 0.5969 heloc 8 0.5315 0.5880 0.5310 0.5325 heloc 16 0.5710 0.6190 0.5085 0.6060 heloc 32 0.5975 0.6605 0.5210 0.6245 heloc 64 0.6180 0.6695 0.5250 0.6640 heloc 128 0.6375 0.6870 0.5105 0.6855 heloc 256 0.6455 0.7100 0.5105 0.6955 heloc 512 0.6555 0.7125 0.5300 0.7075 heloc 1,024 0.6575 0.7075 0.5155 0.7115 kc1 8 0.8033 0.8081 0.8104 0.7441 kc1 16 0.8152 0.8199 0.8104 0.8223 kc1 32 0.8175 0.7986 0.8175 0.7844 kc1 64 0.8176 0.8080 0.8270 0.8247 kc1 128 0.8199 0.8294 0.8341 0.8246 kc1 256 0.8246 0.8223 0.8365 0.8341 kc1 512 0.8247 0.8270 0.8341 0.8389 kc1 1,024 0.8317 0.8484 0.8365 0.8413 led24 8 0.1484 0.2203 0.0969 0.1125 led24 16 0.2094 0.2938 0.0969 0.1406 led24 32 0.3078 0.4187 0.1313 0.1531 led24 64 0.3906 0.5625 0.1266 0.2469 led24 128 0.4703 0.6187 0.1172 0.3766 led24 256 0.5375 0.6656 0.1250 0.4766 led24 512 0.5578 0.6938 0.1172 0.4844 led24 1,024 0.5922 0.7000 0.1375 0.5125 led7 8 0.2734 0.3625 0.2281 0.3219 led7 16 0.4390 0.5141 0.2609 0.4828 led7 32 0.5594 0.5766 0.3281 0.5844 led7 64 0.6562 0.6578 0.3875 0.6344

Table 6 – continued from previous page

##### Dataset Shots KNN RF Qwen-2.5-7B Our

led7 128 0.6875 0.6688 0.4437 0.6609 led7 256 0.6609 0.6453 0.4625 0.6422 led7 512 0.6875 0.6875 0.4844 0.6688 led7 1,024 0.7156 0.7110 0.5250 0.6438 maternal_health_risk 8 0.4089 0.4237 0.3744 0.4384 maternal_health_risk 16 0.4384 0.4877 0.4039 0.4975 maternal_health_risk 32 0.4582 0.5271 0.4532 0.5074 maternal_health_risk 64 0.5616 0.5961 0.4729 0.6108 maternal_health_risk 128 0.6503 0.6995 0.5123 0.6700 maternal_health_risk 256 0.6896 0.7439 0.5468 0.7438 maternal_health_risk 512 0.7439 0.7833 0.5961 0.7832 maternal_health_risk 1,024 0.8030 0.8276 0.5271 0.8424 pc1 8 0.9280 0.9369 0.8964 0.9189 pc1 16 0.9189 0.9189 0.8649 0.8874 pc1 32 0.9234 0.9234 0.8514 0.8874 pc1 64 0.9234 0.9324 0.9234 0.9279 pc1 128 0.9325 0.9279 0.9234 0.9279 pc1 256 0.9279 0.9459 0.9324 0.9369 pc1 512 0.9189 0.9369 0.9324 0.9324 pc1 1,024 0.9234 0.9369 0.9324 0.9280 phoneme 8 0.6337 0.6734 0.6281 0.6540 phoneme 16 0.6965 0.7280 0.6549 0.6920 phoneme 32 0.7437 0.7530 0.6698 0.7345 phoneme 64 0.7817 0.7835 0.6864 0.7484 phoneme 128 0.7909 0.7947 0.6864 0.7715 phoneme 256 0.8113 0.8252 0.6855 0.8021 phoneme 512 0.8298 0.8492 0.6873 0.8196 phoneme 1,024 0.8612 0.8650 0.6883 0.8390 Pima_Indians_Diabetes_Database 8 0.6623 0.6948 0.6429 0.6624 Pima_Indians_Diabetes_Database 16 0.6623 0.6818 0.6234 0.6883 Pima_Indians_Diabetes_Database 32 0.6493 0.7467 0.6558 0.7143 Pima_Indians_Diabetes_Database 64 0.6233 0.7272 0.6234 0.7338 Pima_Indians_Diabetes_Database 128 0.6818 0.7857 0.6818 0.7208 Pima_Indians_Diabetes_Database 256 0.7143 0.7533 0.6883 0.7338 Pima_Indians_Diabetes_Database 512 0.6883 0.7533 0.7078 0.7532 Pima_Indians_Diabetes_Database 1,024 0.7532 0.7468 0.7013 0.7402 ringnorm 8 0.5284 0.5581 0.6034 0.7230 ringnorm 16 0.5385 0.6912 0.6466 0.8987 ringnorm 32 0.5378 0.7959 0.6919 0.9324 ringnorm 64 0.5426 0.8615 0.7108 0.9520 ringnorm 128 0.5540 0.8939 0.7358 0.9541 ringnorm 256 0.5716 0.9169 0.7155 0.9628 ringnorm 512 0.5858 0.9345 0.6932 0.9595 ringnorm 1,024 0.6216 0.9358 0.7250 0.9696 rl 8 0.5181 0.5393 0.5070 0.5201 rl 16 0.5413 0.5523 0.5181 0.5332 rl 32 0.5664 0.5845 0.5392 0.5714 rl 64 0.5754 0.5905 0.5050 0.5804 rl 128 0.5946 0.5936 0.5060 0.5805 rl 256 0.5876 0.6328 0.4869 0.5966 rl 512 0.6067 0.6670 0.5181 0.6408 rl 1,024 0.6097 0.6861 0.5121 0.6278 segment 8 0.4481 0.4632 0.1970 0.4394 segment 16 0.5650 0.5390 0.2706 0.5779 segment 32 0.6364 0.6818 0.3203 0.6472 segment 64 0.7446 0.8095 0.3961 0.7684 segment 128 0.8247 0.8745 0.4113 0.8160 segment 256 0.8442 0.8961 0.4654 0.8571 segment 512 0.8810 0.9091 0.5216 0.8723 segment 1,024 0.8961 0.9156 0.4437 0.8853 seismic_bumps 8 0.9226 0.9265 0.9110 0.9246

Table 6 – continued from previous page

##### Dataset Shots KNN RF Qwen-2.5-7B Our

seismic_bumps 16 0.9226 0.9304 0.9284 0.9265 seismic_bumps 32 0.9323 0.9265 0.9246 0.9033 seismic_bumps 64 0.9265 0.9323 0.9304 0.9168 seismic_bumps 128 0.9303 0.9303 0.9304 0.9284 seismic_bumps 256 0.9246 0.9246 0.9304 0.9284 seismic_bumps 512 0.9284 0.9284 0.9304 0.9303 seismic_bumps 1,024 0.9265 0.9285 0.9304 0.9323 statlog 8 0.5100 0.6150 0.5350 0.5400 statlog 16 0.5400 0.6450 0.6000 0.5950 statlog 32 0.6100 0.6500 0.5750 0.6300 statlog 64 0.6350 0.6850 0.5750 0.6550 statlog 128 0.6600 0.7050 0.6250 0.6700 statlog 256 0.6850 0.7050 0.5900 0.6650 statlog 512 0.6700 0.7400 0.6300 0.6700 statlog 1,024 0.6800 0.7400 0.6300 0.7050 thyroid 8 0.9250 0.9132 0.9125 0.9139 thyroid 16 0.9257 0.9250 0.9215 0.9160 thyroid 32 0.9264 0.9277 0.9257 0.9277 thyroid 64 0.9278 0.9340 0.9243 0.9347 thyroid 128 0.9292 0.9444 0.9257 0.9396 thyroid 256 0.9361 0.9632 0.9250 0.9458 thyroid 512 0.9403 0.9798 0.9257 0.9500 thyroid 1,024 0.9438 0.9916 0.9257 0.9562 twonorm 8 0.5845 0.7155 0.4595 0.8858 twonorm 16 0.8979 0.8331 0.5230 0.9439 twonorm 32 0.9439 0.8885 0.6291 0.9500 twonorm 64 0.9547 0.9297 0.6912 0.9568 twonorm 128 0.9595 0.9446 0.7095 0.9635 twonorm 256 0.9649 0.9547 0.7318 0.9716 twonorm 512 0.9696 0.9588 0.7493 0.9750 twonorm 1,024 0.9723 0.9635 0.8000 0.9730 vehicle 8 0.3059 0.3353 0.2235 0.3647 vehicle 16 0.3824 0.4353 0.2412 0.4294 vehicle 32 0.5235 0.6000 0.3294 0.6294 vehicle 64 0.5824 0.6471 0.3706 0.6882 vehicle 128 0.5941 0.7059 0.4118 0.6588 vehicle 256 0.5706 0.7529 0.3941 0.7118 vehicle 512 0.6470 0.7471 0.3235 0.7294 vehicle 1,024 0.6529 0.7529 0.3882 0.6883 wall_robot_navigation 8 0.4332 0.5220 0.3727 0.4139 wall_robot_navigation 16 0.4716 0.5980 0.4057 0.4606 wall_robot_navigation 32 0.5138 0.7417 0.4203 0.6007 wall_robot_navigation 64 0.5604 0.8599 0.4258 0.6859 wall_robot_navigation 128 0.6181 0.9130 0.4560 0.7463 wall_robot_navigation 256 0.6722 0.9368 0.4451 0.8077 wall_robot_navigation 512 0.7344 0.9670 0.4524 0.8333 wall_robot_navigation 1,024 0.7985 0.9799 0.4762 0.8572 waveform_database_generator_version_1 8 0.4840 0.5540 0.3480 0.5480 waveform_database_generator_version_1 16 0.6330 0.6490 0.3890 0.6100 waveform_database_generator_version_1 32 0.7410 0.7270 0.3860 0.7250 waveform_database_generator_version_1 64 0.7910 0.7690 0.4140 0.7800 waveform_database_generator_version_1 128 0.7940 0.8190 0.4160 0.8060 waveform_database_generator_version_1 256 0.8120 0.7980 0.4050 0.8160 waveform_database_generator_version_1 512 0.8160 0.8270 0.4350 0.8400 waveform_database_generator_version_1 1,024 0.8220 0.8360 0.4210 0.8520 wine 8 0.6066 0.6184 0.5538 0.6281 wine 16 0.6066 0.6360 0.5499 0.6086 wine 32 0.6379 0.6458 0.5205 0.6478 wine 64 0.6477 0.6184 0.5225 0.6888 wine 128 0.6575 0.6791 0.5362 0.7006 wine 256 0.6947 0.7006 0.5714 0.7201

Table 6 – continued from previous page

##### Dataset Shots KNN RF Qwen-2.5-7B Our

wine 512 0.6556 0.6967 0.5616 0.7103 wine 1,024 0.6810 0.7123 0.5675 0.7260 yeast 8 0.3030 0.3401 0.2256 0.2997 yeast 16 0.3502 0.3603 0.2761 0.3434 yeast 32 0.4310 0.4175 0.3367 0.4175 yeast 64 0.4579 0.4848 0.3232 0.4949 yeast 128 0.5152 0.5185 0.2963 0.5252 yeast 256 0.5522 0.5690 0.3333 0.5522 yeast 512 0.5556 0.5825 0.3232 0.5926 yeast 1,024 0.5993 0.5993 0.3266 0.5960

Continued on next page

- C EXTENDED PER-TASK EVALUATION

- Table 7: Per-dataset test accuracy from 8 to 512 shots on extended datasets in Talent. Evaluation on all 8-1024 shots and full datasets is very computation intensive, and we leave it for the next revision. Abbreviations: Qwen = Qwen-2.5-7B-Instruct; Our = MACHINELEARNINGLM (base: Qwen-2.5-7B-Instruct).

Dataset Shots KNN RF Qwen Our

Amazon_employee_access 8 0.9254 0.9245 0.8264 0.9227 Amazon_employee_access 16 0.9221 0.9274 0.813 0.9257 Amazon_employee_access 32 0.928 0.9311 0.8407 0.9279 Amazon_employee_access 64 0.928 0.9295 0.8895 0.9266 Amazon_employee_access 128 0.9305 0.9293 0.91 0.9343 Amazon_employee_access 256 0.9276 0.9313 0.9215 0.9364 Amazon_employee_access 512 0.9291 0.9305 0.9291 0.939 BNG_breast_w 8 0.7732 0.8881 0.7915 0.921 BNG_breast_w 16 0.8894 0.9294 0.8027 0.9496 BNG_breast_w 32 0.9424 0.9409 0.8387 0.9548 BNG_breast_w 64 0.9596 0.9587 0.8538 0.959 BNG_breast_w 128 0.9666 0.9672 0.8405 0.9621 BNG_breast_w 256 0.97 0.9717 0.8455 0.9511 BNG_breast_w 512 0.9724 0.9733 0.849 0.7156 BNG_cmc 8 0.3871 0.4004 0.3691 0.3797 BNG_cmc 16 0.4029 0.4164 0.3851 0.392 BNG_cmc 32 0.4189 0.4349 0.3884 0.4199 BNG_cmc 64 0.4366 0.4586 0.3883 0.4415 BNG_cmc 128 0.4538 0.4818 0.4009 0.4769 BNG_cmc 256 0.473 0.4952 0.4062 0.4948 BNG_cmc 512 0.4855 0.5085 0.4023 0.4442 BNG_tic_tac_toe 8 0.6153 0.6018 0.529 0.5723 BNG_tic_tac_toe 16 0.6068 0.6118 0.5649 0.6012 BNG_tic_tac_toe 32 0.6152 0.6391 0.5567 0.6171 BNG_tic_tac_toe 64 0.6429 0.6645 0.5763 0.6237 BNG_tic_tac_toe 128 0.6635 0.6917 0.5837 0.6421 BNG_tic_tac_toe 256 0.6806 0.7127 0.6086 0.6545 BNG_tic_tac_toe 512 0.6971 0.7239 0.5982 0.6615 Bank_Customer_Churn_Dataset 8 0.7435 0.776 0.7105 0.731 Bank_Customer_Churn_Dataset 16 0.721 0.7765 0.7275 0.75 Bank_Customer_Churn_Dataset 32 0.725 0.79 0.744 0.767 Bank_Customer_Churn_Dataset 64 0.732 0.8045 0.757 0.787 Bank_Customer_Churn_Dataset 128 0.7485 0.816 0.7675 0.7915 Bank_Customer_Churn_Dataset 256 0.7435 0.83 0.776 0.806 Bank_Customer_Churn_Dataset 512 0.7565 0.8405 0.7935 0.8095 California_Housing_Classification 8 0.5206 0.5748 0.5301 0.6008 California_Housing_Classification 16 0.5378 0.6345 0.5434 0.6432 California_Housing_Classification 32 0.556 0.6977 0.5293 0.6953 California_Housing_Classification 64 0.5816 0.7253 0.5371 0.734

Continued on next page

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

California_Housing_Classification 128 0.5919 0.7529 0.5196 0.7512 California_Housing_Classification 256 0.6085 0.7851 0.5339 0.7766 California_Housing_Classification 512 0.6107 0.8071 0.5194 0.7822 Cardiovascular_Disease_dataset 8 0.5215 0.5707 0.5624 0.5547 Cardiovascular_Disease_dataset 16 0.5338 0.6139 0.5636 0.5892 Cardiovascular_Disease_dataset 32 0.5438 0.6445 0.5681 0.6179 Cardiovascular_Disease_dataset 64 0.5458 0.6642 0.5719 0.6488 Cardiovascular_Disease_dataset 128 0.5617 0.6773 0.5705 0.6718 Cardiovascular_Disease_dataset 256 0.5694 0.6888 0.5573 0.6878 Cardiovascular_Disease_dataset 512 0.5978 0.6929 0.5559 0.6918 Click_prediction_small 8 0.7866 0.7689 0.7016 0.7029 Click_prediction_small 16 0.7883 0.7677 0.7247 0.7068 Click_prediction_small 32 0.8095 0.7964 0.7438 0.735 Click_prediction_small 64 0.816 0.8105 0.7768 0.7803 Click_prediction_small 128 0.8151 0.8172 0.7827 0.8058 Click_prediction_small 256 0.8174 0.8201 0.7863 0.8148 Click_prediction_small 512 0.8229 0.8228 0.8054 0.8239 Credit_c 8 0.4335 0.4757 0.418 0.45 Credit_c 16 0.4487 0.5232 0.4502 0.5124 Credit_c 32 0.4576 0.5602 0.4722 0.5422 Credit_c 64 0.4726 0.5895 0.5 0.5688 Credit_c 128 0.4928 0.6121 0.5013 0.5957 Credit_c 256 0.5054 0.6354 0.5246 0.6069 Credit_c 512 0.5218 0.6514 0.5288 0.6127 E_CommereShippingData 8 0.5696 0.5486 0.4991 0.5454 E_CommereShippingData 16 0.5878 0.5823 0.4995 0.5496 E_CommereShippingData 32 0.6041 0.6055 0.4932 0.5805 E_CommereShippingData 64 0.6141 0.6205 0.5159 0.6132 E_CommereShippingData 128 0.6259 0.6368 0.5054 0.6182 E_CommereShippingData 256 0.6391 0.6532 0.5127 0.6318 E_CommereShippingData 512 0.6454 0.6428 0.5204 0.6304 Employee 8 0.5715 0.5886 0.6036 0.6101 Employee 16 0.6294 0.6348 0.6079 0.6251 Employee 32 0.6208 0.6638 0.6219 0.6391 Employee 64 0.6294 0.7024 0.6402 0.6746 Employee 128 0.6702 0.7605 0.6487 0.7197 Employee 256 0.6799 0.7852 0.6477 0.7486 Employee 512 0.7025 0.8163 0.6466 0.7551 FOREX_audcad_hour_High 8 0.4965 0.501 0.5013 0.5029 FOREX_audcad_hour_High 16 0.496 0.5074 0.4943 0.506 FOREX_audcad_hour_High 32 0.4971 0.5067 0.5033 0.4968 FOREX_audcad_hour_High 64 0.5084 0.5024 0.5075 0.5171 FOREX_audcad_hour_High 128 0.5009 0.5092 0.5108 0.501 FOREX_audcad_hour_High 256 0.5077 0.5183 0.5128 0.5218 FOREX_audcad_hour_High 512 0.5101 0.5378 0.5082 0.5212 FOREX_audjpy_hour_High 8 0.5004 0.5028 0.4962 0.5006 FOREX_audjpy_hour_High 16 0.4985 0.4945 0.5061 0.4995 FOREX_audjpy_hour_High 32 0.4975 0.4909 0.5006 0.4977 FOREX_audjpy_hour_High 64 0.5013 0.4947 0.5135 0.4975 FOREX_audjpy_hour_High 128 0.507 0.504 0.4998 0.5108 FOREX_audjpy_hour_High 256 0.5044 0.5101 0.5083 0.5118 FOREX_audjpy_hour_High 512 0.5154 0.5179 0.5081 0.515 FOREX_audsgd_hour_High 8 0.5041 0.4992 0.4975 0.5033 FOREX_audsgd_hour_High 16 0.5029 0.5101 0.4995 0.5101 FOREX_audsgd_hour_High 32 0.5123 0.504 0.4941 0.5056 FOREX_audsgd_hour_High 64 0.5081 0.5081 0.5084 0.5059 FOREX_audsgd_hour_High 128 0.505 0.5024 0.5112 0.4987 FOREX_audsgd_hour_High 256 0.5178 0.5081 0.5124 0.5168 FOREX_audsgd_hour_High 512 0.5113 0.5149 0.5186 0.5103 FOREX_audusd_hour_High 8 0.506 0.4967 0.5059 0.5012 FOREX_audusd_hour_High 16 0.502 0.4934 0.5113 0.498

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

FOREX_audusd_hour_High 32 0.5033 0.498 0.5117 0.4962 FOREX_audusd_hour_High 64 0.5073 0.5091 0.5108 0.5097 FOREX_audusd_hour_High 128 0.5146 0.5071 0.5232 0.5112 FOREX_audusd_hour_High 256 0.5175 0.5125 0.5291 0.5111 FOREX_audusd_hour_High 512 0.5209 0.5212 0.5283 0.5159 FOREX_cadjpy_hour_High 8 0.5069 0.5018 0.5078 0.5066 FOREX_cadjpy_hour_High 16 0.5023 0.5034 0.498 0.5074 FOREX_cadjpy_hour_High 32 0.5021 0.508 0.4989 0.5068 FOREX_cadjpy_hour_High 64 0.5123 0.5058 0.5084 0.514 FOREX_cadjpy_hour_High 128 0.5042 0.5018 0.5101 0.5072 FOREX_cadjpy_hour_High 256 0.4973 0.5109 0.5099 0.5163 FOREX_cadjpy_hour_High 512 0.494 0.5128 0.5149 0.5064 Firm_Teacher_Clave_Direction_Classification 8 0.4102 0.5111 0.3694 0.3907 Firm_Teacher_Clave_Direction_Classification 16 0.5542 0.5514 0.3833 0.4829 Firm_Teacher_Clave_Direction_Classification 32 0.644 0.6143 0.4269 0.556 Firm_Teacher_Clave_Direction_Classification 64 0.6833 0.6639 0.4236 0.5986 Firm_Teacher_Clave_Direction_Classification 128 0.7125 0.7028 0.4366 0.6731 Firm_Teacher_Clave_Direction_Classification 256 0.7301 0.7273 0.4379 0.681 Firm_Teacher_Clave_Direction_Classification 512 0.7412 0.7556 0.4264 0.6648 Gender_Gap_in_Spanish_WP 8 0.4958 0.4684 0.4368 0.52 Gender_Gap_in_Spanish_WP 16 0.5095 0.5221 0.4884 0.5832 Gender_Gap_in_Spanish_WP 32 0.5316 0.5389 0.4621 0.5664 Gender_Gap_in_Spanish_WP 64 0.5053 0.5263 0.4548 0.5884 Gender_Gap_in_Spanish_WP 128 0.5358 0.5284 0.4347 0.5779 Gender_Gap_in_Spanish_WP 256 0.5442 0.5495 0.4842 0.5937 Gender_Gap_in_Spanish_WP 512 0.5558 0.5716 0.5221 0.5895 GesturePhaseSegmentationProcessed 8 0.3104 0.3104 0.2744 0.2891 GesturePhaseSegmentationProcessed 16 0.3479 0.3514 0.2532 0.3438 GesturePhaseSegmentationProcessed 32 0.4005 0.3904 0.2927 0.3489 GesturePhaseSegmentationProcessed 64 0.4299 0.438 0.2921 0.4051 GesturePhaseSegmentationProcessed 128 0.4349 0.4486 0.3078 0.4253 GesturePhaseSegmentationProcessed 256 0.439 0.4663 0.3235 0.3899 GesturePhaseSegmentationProcessed 512 0.4714 0.5073 0.3367 0.4385 HR_Analytics_Job_Change_of_Data_Scientists 8 0.6788 0.7289 0.6498 0.6738 HR_Analytics_Job_Change_of_Data_Scientists 16 0.6733 0.7369 0.6485 0.6843 HR_Analytics_Job_Change_of_Data_Scientists 32 0.6715 0.7416 0.6816 0.7098 HR_Analytics_Job_Change_of_Data_Scientists 64 0.6746 0.7484 0.6837 0.7252 HR_Analytics_Job_Change_of_Data_Scientists 128 0.6855 0.7578 0.6801 0.7333 HR_Analytics_Job_Change_of_Data_Scientists 256 0.7035 0.7724 0.6996 0.75 HR_Analytics_Job_Change_of_Data_Scientists 512 0.7103 0.7782 0.7075 0.7388 INNHotelsGroup 8 0.667 0.6286 0.5247 0.5745 INNHotelsGroup 16 0.7024 0.6713 0.5443 0.6168 INNHotelsGroup 32 0.7159 0.7027 0.5446 0.6471 INNHotelsGroup 64 0.7206 0.7344 0.5474 0.6906 INNHotelsGroup 128 0.7311 0.7591 0.5509 0.7134 INNHotelsGroup 256 0.7432 0.7796 0.5789 0.7294 INNHotelsGroup 512 0.7458 0.7976 0.5904 0.7127 Insurance 8 0.6885 0.691 0.6355 0.6609 Insurance 16 0.6794 0.6884 0.6242 0.6654 Insurance 32 0.6813 0.7097 0.6463 0.7061 Insurance 64 0.6855 0.7182 0.6419 0.7234 Insurance 128 0.6875 0.7229 0.63 0.7346 Insurance 256 0.6964 0.7244 0.6736 0.7437 Insurance 512 0.7013 0.7286 0.6605 0.748 Intersectional_Bias_Assessment 8 0.5559 0.6477 0.5091 0.5077 Intersectional_Bias_Assessment 16 0.8109 0.765 0.5177 0.5541 Intersectional_Bias_Assessment 32 0.8691 0.8236 0.5304 0.6695 Intersectional_Bias_Assessment 64 0.8941 0.8787 0.5359 0.7418 Intersectional_Bias_Assessment 128 0.9096 0.8977 0.5182 0.8145 Intersectional_Bias_Assessment 256 0.9141 0.9127 0.519 0.8314 Intersectional_Bias_Assessment 512 0.9209 0.9264 0.5318 0.8341

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

JapaneseVowels 8 0.1445 0.2629 0.1239 0.2223 JapaneseVowels 16 0.1751 0.3658 0.1244 0.3236 JapaneseVowels 32 0.1872 0.4947 0.133 0.4646 JapaneseVowels 64 0.2027 0.6342 0.131 0.6106 JapaneseVowels 128 0.2173 0.7486 0.1485 0.7501 JapaneseVowels 256 0.2474 0.8334 0.128 0.7963 JapaneseVowels 512 0.274 0.8786 0.137 0.842 Long 8 0.6474 0.7478 0.6496 0.6551 Long 16 0.6417 0.8359 0.6674 0.7121 Long 32 0.6306 0.8627 0.6551 0.7857 Long 64 0.6294 0.8862 0.6763 0.808 Long 128 0.6462 0.9085 0.673 0.8806 Long 256 0.6529 0.9308 0.6774 0.8303 Long 512 0.6607 0.9352 0.6897 0.865 MagicTelescope 8 0.6078 0.6104 0.5949 0.6078 MagicTelescope 16 0.6454 0.6488 0.5886 0.6861 MagicTelescope 32 0.6687 0.7119 0.6199 0.7235 MagicTelescope 64 0.6961 0.7594 0.6383 0.7603 MagicTelescope 128 0.719 0.7968 0.6364 0.7713 MagicTelescope 256 0.7403 0.8089 0.6441 0.79 MagicTelescope 512 0.75 0.8207 0.658 0.6643 PhishingWebsites 8 0.5577 0.6929 0.5658 0.6377 PhishingWebsites 16 0.7277 0.7973 0.5803 0.711 PhishingWebsites 32 0.7834 0.858 0.5889 0.7838 PhishingWebsites 64 0.8268 0.9 0.6192 0.8223 PhishingWebsites 128 0.8471 0.9068 0.6133 0.8621 PhishingWebsites 256 0.8748 0.9145 0.6432 0.8707 PhishingWebsites 512 0.8869 0.9222 0.6105 0.891 Rain_in_Australia 8 0.7417 0.7263 0.6996 0.6702 Rain_in_Australia 16 0.7583 0.7519 0.7135 0.703 Rain_in_Australia 32 0.767 0.7668 0.7209 0.7242 Rain_in_Australia 64 0.7732 0.782 0.7178 0.7486 Rain_in_Australia 128 0.775 0.79 0.717 0.7748 Rain_in_Australia 256 0.7799 0.803 0.7219 0.786 Rain_in_Australia 512 0.7828 0.8087 0.7058 0.7926 SDSS17 8 0.4949 0.6039 0.468 0.533 SDSS17 16 0.5023 0.7126 0.5077 0.6122 SDSS17 32 0.5132 0.8072 0.5296 0.6808 SDSS17 64 0.5235 0.883 0.5457 0.7328 SDSS17 128 0.5299 0.9278 0.5591 0.7722 SDSS17 256 0.5431 0.9473 0.5653 0.7976 SDSS17 512 0.5544 0.9556 0.548 0.8195 Shipping 8 0.5437 0.5423 0.4928 0.5564 Shipping 16 0.5546 0.5582 0.5045 0.5514 Shipping 32 0.5763 0.5818 0.4977 0.5827 Shipping 64 0.5827 0.6036 0.4996 0.5986 Shipping 128 0.6146 0.6218 0.4845 0.6041 Shipping 256 0.625 0.6327 0.5077 0.6036 Shipping 512 0.6255 0.6241 0.4946 0.6268 Telecom_Churn_Dataset 8 0.8531 0.859 0.8006 0.862 Telecom_Churn_Dataset 16 0.8515 0.8576 0.7781 0.8591 Telecom_Churn_Dataset 32 0.8575 0.8621 0.7886 0.8606 Telecom_Churn_Dataset 64 0.8651 0.8636 0.8051 0.8591 Telecom_Churn_Dataset 128 0.8635 0.8726 0.8081 0.8636 Telecom_Churn_Dataset 256 0.8755 0.8861 0.8156 0.8681 Telecom_Churn_Dataset 512 0.8816 0.907 0.8426 0.8696 VulNoneVul 8 0.9886 0.9868 0.9631 0.9763 VulNoneVul 16 0.9886 0.9868 0.9859 0.9798 VulNoneVul 32 0.9886 0.9886 0.9789 0.9859 VulNoneVul 64 0.9886 0.9886 0.9868 0.9886 VulNoneVul 128 0.9886 0.9886 0.9833 0.9886

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

VulNoneVul 256 0.9886 0.9886 0.9886 0.9886 VulNoneVul 512 0.9886 0.9886 0.9868 0.9886 Water_Quality_and_Potability 8 0.5411 0.5335 0.5671 0.4969 Water_Quality_and_Potability 16 0.5259 0.5732 0.5519 0.5686 Water_Quality_and_Potability 32 0.5442 0.5701 0.5549 0.5533 Water_Quality_and_Potability 64 0.5503 0.5747 0.5701 0.5351 Water_Quality_and_Potability 128 0.5442 0.593 0.5869 0.5869 Water_Quality_and_Potability 256 0.5366 0.5762 0.5686 0.59 Water_Quality_and_Potability 512 0.5122 0.6067 0.5945 0.5915 Wilt 8 0.9523 0.9482 0.9016 0.9451 Wilt 16 0.9472 0.9347 0.8829 0.9461 Wilt 32 0.9513 0.944 0.912 0.9462 Wilt 64 0.9544 0.9544 0.9254 0.9388 Wilt 128 0.9523 0.9565 0.9295 0.9617 Wilt 256 0.9544 0.9627 0.941 0.9689 Wilt 512 0.9596 0.97 0.9482 0.9741 abalone 8 0.4294 0.4809 0.3947 0.4629 abalone 16 0.4785 0.5095 0.3971 0.5096 abalone 32 0.5167 0.5239 0.3708 0.5371 abalone 64 0.5455 0.5383 0.3768 0.5299 abalone 128 0.5634 0.555 0.4174 0.5706 abalone 256 0.5933 0.5694 0.3888 0.5718 abalone 512 0.6088 0.5993 0.4653 0.6017 ada_prior 8 0.6528 0.7043 0.7262 0.7306 ada_prior 16 0.6594 0.7327 0.7426 0.7415 ada_prior 32 0.6408 0.7491 0.7591 0.7426 ada_prior 64 0.6835 0.7765 0.7524 0.7656 ada_prior 128 0.6879 0.7962 0.7601 0.7821 ada_prior 256 0.7031 0.793 0.7568 0.7908 ada_prior 512 0.6999 0.8247 0.7371 0.805 adult 8 0.6705 0.7319 0.7313 0.7344 adult 16 0.6785 0.7531 0.7485 0.7579 adult 32 0.6908 0.7734 0.7737 0.766 adult 64 0.6946 0.7959 0.7674 0.7761 adult 128 0.708 0.8098 0.7736 0.791 adult 256 0.7095 0.8197 0.7713 0.7985 adult 512 0.7162 0.8337 0.7651 0.8174 airlines_seed_0_nrows_2000_nclasses_10_ncols_100 8 0.51 0.54 0.9285 0.5375 airlines_seed_0_nrows_2000_nclasses_10_ncols_100 16 0.5075 0.52 0.9325 0.515 airlines_seed_0_nrows_2000_nclasses_10_ncols_100 32 0.4725 0.5125 0.9471 0.51 airlines_seed_0_nrows_2000_nclasses_10_ncols_100 64 0.5 0.5075 0.9364 0.495 airlines_seed_0_nrows_2000_nclasses_10_ncols_100 128 0.535 0.56 0.9471 0.5275 airlines_seed_0_nrows_2000_nclasses_10_ncols_100 256 0.535 0.59 0.9483 0.54 airlines_seed_0_nrows_2000_nclasses_10_ncols_100 512 0.5175 0.5575 0.57 0.5475 allbp 8 0.9431 0.943 0.9484 0.9496 allbp 16 0.9536 0.9523 0.9351 0.951 allbp 32 0.9536 0.955 0.8954 0.9523 allbp 64 0.9536 0.9563 0.943 0.9536 allbp 128 0.9536 0.9576 0.9563 0.9563 allbp 256 0.9576 0.9563 0.955 0.9563 allbp 512 0.9603 0.9669 0.951 0.9523 artificial_characters 8 0.1595 0.16 0.1135 0.1477 artificial_characters 16 0.184 0.1854 0.1154 0.1786 artificial_characters 32 0.227 0.2446 0.1389 0.2148 artificial_characters 64 0.2994 0.3068 0.1453 0.2794 artificial_characters 128 0.3591 0.4095 0.1771 0.3576 artificial_characters 256 0.4193 0.4731 0.205 0.4095 artificial_characters 512 0.4594 0.5259 0.1937 0.4618 compass 8 0.532 0.5554 0.5173 0.5389 compass 16 0.5614 0.5678 0.5296 0.5551 compass 32 0.5816 0.6176 0.5425 0.5918

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

compass 64 0.5843 0.6254 0.5458 0.5978 compass 128 0.5788 0.6416 0.541 0.6326 compass 256 0.6056 0.6575 0.5455 0.6435 compass 512 0.6098 0.6606 0.5293 0.6419 connect_4 8 0.6032 0.5793 0.5593 0.6055 connect_4 16 0.6122 0.6011 0.6038 0.618 connect_4 32 0.6147 0.6152 0.6171 0.6315 connect_4 64 0.6139 0.6334 0.6246 0.6417 connect_4 128 0.6181 0.6429 0.64 0.6322 connect_4 256 0.621 0.6622 0.6483 0.643 connect_4 512 0.6282 0.6762 0.653 0.65 credit 8 0.5073 0.5375 0.4986 0.571 credit 16 0.5121 0.5558 0.5178 0.6156 credit 32 0.5214 0.5881 0.5186 0.6539 credit 64 0.5139 0.6174 0.5238 0.6826 credit 128 0.5166 0.6357 0.5283 0.7152 credit 256 0.5178 0.6428 0.5328 0.7197 credit 512 0.5059 0.6533 0.5151 0.7248 customer_satisfaction_in_airline 8 0.5158 0.6521 0.5669 0.5982 customer_satisfaction_in_airline 16 0.5242 0.7118 0.5964 0.6621 customer_satisfaction_in_airline 32 0.5368 0.7682 0.6067 0.74 customer_satisfaction_in_airline 64 0.5451 0.8106 0.5961 0.7777 customer_satisfaction_in_airline 128 0.5536 0.8412 0.6107 0.7994 customer_satisfaction_in_airline 256 0.559 0.8646 0.6193 0.8113 customer_satisfaction_in_airline 512 0.565 0.883 0.5726 0.8294 dabetes_130_us_hospitals 8 0.4986 0.5214 0.5231 0.497 dabetes_130_us_hospitals 16 0.5083 0.5215 0.5298 0.5032 dabetes_130_us_hospitals 32 0.5099 0.5397 0.5323 0.5119 dabetes_130_us_hospitals 64 0.5139 0.5449 0.5353 0.5246 dabetes_130_us_hospitals 128 0.5166 0.557 0.5283 0.5419 dabetes_130_us_hospitals 256 0.5187 0.5742 0.5346 0.5474 dabetes_130_us_hospitals 512 0.5221 0.5868 0.5288 0.5518 default_of_credit_card_clients 8 0.7355 0.7335 0.6715 0.6913 default_of_credit_card_clients 16 0.7432 0.7535 0.6756 0.7298 default_of_credit_card_clients 32 0.7354 0.7582 0.7177 0.7244 default_of_credit_card_clients 64 0.7437 0.7756 0.7456 0.7723 default_of_credit_card_clients 128 0.7414 0.7919 0.7478 0.7849 default_of_credit_card_clients 256 0.7463 0.798 0.7588 0.7995 default_of_credit_card_clients 512 0.7486 0.802 0.764 0.7819 delta_ailerons 8 0.6851 0.7686 0.5301 0.758 delta_ailerons 16 0.744 0.866 0.5533 0.7763 delta_ailerons 32 0.8233 0.9088 0.5406 0.8654 delta_ailerons 64 0.8843 0.9166 0.5617 0.8885 delta_ailerons 128 0.9096 0.9243 0.5652 0.9005 delta_ailerons 256 0.9187 0.9257 0.6052 0.9004 delta_ailerons 512 0.9242 0.9327 0.5743 0.92 dry_bean_dataset 8 0.3992 0.5237 0.227 0.455 dry_bean_dataset 16 0.4741 0.6739 0.267 0.6287 dry_bean_dataset 32 0.5207 0.765 0.3092 0.7664 dry_bean_dataset 64 0.5476 0.8281 0.3606 0.8061 dry_bean_dataset 128 0.5762 0.8652 0.426 0.8358 dry_bean_dataset 256 0.5832 0.8818 0.4815 0.8656 dry_bean_dataset 512 0.6049 0.8931 0.5094 0.8329 eeg_eye_state 8 0.511 0.5391 0.525 0.5157 eeg_eye_state 16 0.5487 0.5481 0.528 0.5261 eeg_eye_state 32 0.5721 0.5718 0.5254 0.5277 eeg_eye_state 64 0.6015 0.6202 0.5451 0.5801 eeg_eye_state 128 0.6752 0.6562 0.5384 0.6155 eeg_eye_state 256 0.7447 0.6852 0.5464 0.6812 eeg_eye_state 512 0.8021 0.747 0.548 0.716 electricity 8 0.5282 0.6019 0.5937 0.5696

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

electricity 16 0.53 0.6489 0.5952 0.6126 electricity 32 0.5396 0.6945 0.603 0.6654 electricity 64 0.5553 0.7169 0.6158 0.6925 electricity 128 0.5795 0.7411 0.6225 0.7056 electricity 256 0.6172 0.7622 0.6142 0.7304 electricity 512 0.6376 0.7728 0.6152 0.7458 eye_movements 8 0.3548 0.3633 0.3436 0.337 eye_movements 16 0.3634 0.3771 0.3535 0.3522 eye_movements 32 0.3653 0.3817 0.3568 0.366 eye_movements 64 0.3693 0.4021 0.3594 0.4034 eye_movements 128 0.3817 0.431 0.3436 0.4192 eye_movements 256 0.4061 0.4573 0.3555 0.41 eye_movements 512 0.3956 0.4718 0.343 0.4409 eye_movements_bin 8 0.4973 0.4993 0.4816 0.4974 eye_movements_bin 16 0.4869 0.4921 0.5072 0.4796 eye_movements_bin 32 0.5046 0.5 0.4796 0.4757 eye_movements_bin 64 0.5125 0.5197 0.477 0.4869 eye_movements_bin 128 0.5453 0.5243 0.4882 0.5328 eye_movements_bin 256 0.5303 0.523 0.4764 0.5138 eye_movements_bin 512 0.5309 0.546 0.4915 0.5152 house_16H 8 0.6064 0.6334 0.4922 0.5771 house_16H 16 0.6486 0.7187 0.5015 0.6442 house_16H 32 0.6646 0.7517 0.4967 0.6839 house_16H 64 0.6567 0.7865 0.513 0.7135 house_16H 128 0.6708 0.8065 0.5048 0.7231 house_16H 256 0.6686 0.8184 0.5204 0.7717 house_16H 512 0.676 0.8262 0.5152 0.7413 in_vehicle_coupon_recommendation 8 0.5207 0.5205 0.4927 0.5089 in_vehicle_coupon_recommendation 16 0.5353 0.5345 0.4888 0.5277 in_vehicle_coupon_recommendation 32 0.5199 0.5412 0.5002 0.5483 in_vehicle_coupon_recommendation 64 0.5412 0.5664 0.4982 0.5309 in_vehicle_coupon_recommendation 128 0.5408 0.5605 0.4841 0.5306 in_vehicle_coupon_recommendation 256 0.5613 0.6042 0.5211 0.5577 in_vehicle_coupon_recommendation 512 0.5712 0.6437 0.5124 0.5692 internet_firewall 8 0.7062 0.7704 0.5475 0.7163 internet_firewall 16 0.8121 0.8463 0.6043 0.7863 internet_firewall 32 0.8669 0.8898 0.6278 0.7993 internet_firewall 64 0.8908 0.9105 0.6365 0.8101 internet_firewall 128 0.9099 0.9242 0.6446 0.8031 internet_firewall 256 0.921 0.9323 0.6133 0.7941 internet_firewall 512 0.9245 0.9327 0.5876 0.7995 jm1 8 0.7781 0.7607 0.7423 0.7217 jm1 16 0.7804 0.7772 0.7658 0.7478 jm1 32 0.7754 0.7827 0.7818 0.7606 jm1 64 0.7731 0.7915 0.7869 0.7754 jm1 128 0.779 0.7979 0.7997 0.8002 jm1 256 0.7851 0.7869 0.813 0.8062 jm1 512 0.7998 0.8011 0.8089 0.8209 jungle_chess_2pcs_raw_endgame_complete 8 0.5064 0.5282 0.4732 0.4986 jungle_chess_2pcs_raw_endgame_complete 16 0.5851 0.5868 0.5069 0.5723 jungle_chess_2pcs_raw_endgame_complete 32 0.6188 0.629 0.5116 0.6074 jungle_chess_2pcs_raw_endgame_complete 64 0.6523 0.6713 0.5227 0.6405 jungle_chess_2pcs_raw_endgame_complete 128 0.672 0.7033 0.5422 0.6785 jungle_chess_2pcs_raw_endgame_complete 256 0.6961 0.7392 0.5462 0.7001 jungle_chess_2pcs_raw_endgame_complete 512 0.7317 0.7615 0.5543 0.6779 kdd_ipums_la_97_small 8 0.5963 0.6686 0.5617 0.63 kdd_ipums_la_97_small 16 0.6715 0.7755 0.5915 0.7052 kdd_ipums_la_97_small 32 0.7042 0.8199 0.605 0.763 kdd_ipums_la_97_small 64 0.7746 0.8275 0.6435 0.7987 kdd_ipums_la_97_small 128 0.8025 0.8343 0.5896 0.7688 kdd_ipums_la_97_small 256 0.8064 0.8565 0.5839 0.7871

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

kdd_ipums_la_97_small 512 0.8218 0.8468 0.5829 0.7033 kr_vs_kp 8 0.5 0.5531 0.4734 0.5219 kr_vs_kp 16 0.5703 0.6141 0.4672 0.5344 kr_vs_kp 32 0.7219 0.7422 0.5109 0.6109 kr_vs_kp 64 0.7281 0.8422 0.4875 0.6547 kr_vs_kp 128 0.7797 0.925 0.5234 0.725 kr_vs_kp 256 0.8359 0.9578 0.5203 0.75 kr_vs_kp 512 0.8562 0.9656 0.5359 0.7641 kropt 8 0.1354 0.1431 0.0882 0.0699 kropt 16 0.1454 0.1623 0.087 0.0661 kropt 32 0.1725 0.1855 0.0775 0.0665 kropt 64 0.2015 0.2201 0.077 0.0624 kropt 128 0.2363 0.2651 0.0823 0.0764 kropt 256 0.2753 0.3282 0.083 0.1005 kropt 512 0.3175 0.379 0.0752 0.1162 law_school_admission_bianry 8 0.6377 0.7245 0.6048 0.6738 law_school_admission_bianry 16 0.6538 0.8238 0.6365 0.7413 law_school_admission_bianry 32 0.6447 0.9279 0.6565 0.7996 law_school_admission_bianry 64 0.6483 0.982 0.7058 0.8642 law_school_admission_bianry 128 0.6714 0.9971 0.717 0.9029 law_school_admission_bianry 256 0.6822 0.9998 0.7414 0.925 law_school_admission_bianry 512 0.693 1 0.7685 0.9437 letter 8 0.0825 0.109 0.0398 0.0383 letter 16 0.132 0.156 0.0387 0.0375 letter 32 0.184 0.228 0.0375 0.0462 letter 64 0.2665 0.3377 0.0425 0.068 letter 128 0.3688 0.4742 0.044 0.127 letter 256 0.488 0.616 0.042 0.1777 letter 512 0.605 0.718 0.039 0.215 mammography 8 0.9772 0.9772 0.9472 0.9567 mammography 16 0.9772 0.9763 0.9625 0.9597 mammography 32 0.9772 0.9759 0.9566 0.9665 mammography 64 0.9767 0.9763 0.9687 0.9687 mammography 128 0.9781 0.9808 0.9705 0.9759 mammography 256 0.9803 0.9826 0.9678 0.9812 mammography 512 0.9821 0.9848 0.9723 0.983 microaggregation2 8 0.4627 0.436 0.417 0.3815 microaggregation2 16 0.4627 0.468 0.4475 0.436 microaggregation2 32 0.4895 0.4748 0.4733 0.4945 microaggregation2 64 0.5033 0.5102 0.491 0.5238 microaggregation2 128 0.519 0.5188 0.503 0.5375 microaggregation2 256 0.518 0.535 0.523 0.5385 microaggregation2 512 0.5218 0.5465 0.5393 0.551 mobile_c36_oversampling 8 0.6599 0.6687 0.5412 0.6654 mobile_c36_oversampling 16 0.7128 0.7639 0.5549 0.7429 mobile_c36_oversampling 32 0.7902 0.8164 0.5759 0.7937 mobile_c36_oversampling 64 0.8269 0.8513 0.6104 0.825 mobile_c36_oversampling 128 0.8485 0.8735 0.6309 0.8474 mobile_c36_oversampling 256 0.8669 0.8899 0.6525 0.8667 mobile_c36_oversampling 512 0.8854 0.9011 0.6751 0.8838 mozilla4 8 0.6822 0.7533 0.6057 0.7089 mozilla4 16 0.7507 0.8636 0.6224 0.7668 mozilla4 32 0.7932 0.8945 0.6494 0.7951 mozilla4 64 0.8205 0.9051 0.6825 0.834 mozilla4 128 0.8546 0.9218 0.6687 0.8523 mozilla4 256 0.8749 0.9267 0.6822 0.8575 mozilla4 512 0.8852 0.9306 0.6536 0.6208 national_longitudinal_survey_binary 8 0.5855 0.7271 0.6059 0.5377 national_longitudinal_survey_binary 16 0.5896 0.8584 0.6029 0.5968 national_longitudinal_survey_binary 32 0.6222 0.9399 0.6283 0.6446 national_longitudinal_survey_binary 64 0.6273 0.9827 0.6456 0.6843

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

national_longitudinal_survey_binary 128 0.6283 0.9939 0.6456 0.7465 national_longitudinal_survey_binary 256 0.6436 0.998 0.6467 0.7597 national_longitudinal_survey_binary 512 0.6426 0.998 0.6558 0.8065 okcupid_stem 8 0.6824 0.6793 0.6374 0.6079 okcupid_stem 16 0.6887 0.6769 0.6419 0.657 okcupid_stem 32 0.6884 0.6807 0.6488 0.6752 okcupid_stem 64 0.6902 0.696 0.6477 0.6968 okcupid_stem 128 0.6964 0.6996 0.6778 0.711 okcupid_stem 256 0.6955 0.706 0.6726 0.7155 okcupid_stem 512 0.688 0.7135 0.6747 0.7135 online_shoppers 8 0.854 0.8447 0.7137 0.7891 online_shoppers 16 0.8573 0.852 0.6833 0.8269 online_shoppers 32 0.8686 0.858 0.8086 0.8362 online_shoppers 64 0.8763 0.8718 0.7977 0.854 online_shoppers 128 0.8824 0.8791 0.8025 0.8654 online_shoppers 256 0.8873 0.884 0.8285 0.8763 online_shoppers 512 0.8917 0.8917 0.8321 0.8844 page_blocks 8 0.8822 0.8904 0.821 0.8356 page_blocks 16 0.8868 0.8959 0.8156 0.8521 page_blocks 32 0.8868 0.8996 0.8648 0.8886 page_blocks 64 0.8922 0.9178 0.8639 0.9114 page_blocks 128 0.8959 0.9297 0.8803 0.9133 page_blocks 256 0.9087 0.9407 0.8777 0.9306 page_blocks 512 0.9251 0.9535 0.8822 0.9398 pendigits 8 0.2865 0.326 0.1073 0.2628 pendigits 16 0.4279 0.4529 0.1323 0.4443 pendigits 32 0.5925 0.6362 0.1514 0.6367 pendigits 64 0.7408 0.7776 0.171 0.7781 pendigits 128 0.8436 0.8467 0.1792 0.8276 pendigits 256 0.8963 0.9086 0.2278 0.8872 pendigits 512 0.9359 0.9418 0.2178 0.8608 pol 8 0.5548 0.5627 0.5052 0.5305 pol 16 0.6113 0.6411 0.5101 0.5791 pol 32 0.6743 0.7888 0.5251 0.7025 pol 64 0.7506 0.8533 0.5196 0.7764 pol 128 0.7928 0.9063 0.5448 0.8547 pol 256 0.8414 0.9281 0.5543 0.8904 pol 512 0.8686 0.9395 0.5538 0.8394 rice_cammeo_and_osmancik 8 0.769 0.8491 0.5407 0.7939 rice_cammeo_and_osmancik 16 0.8491 0.8858 0.6457 0.8504 rice_cammeo_and_osmancik 32 0.8583 0.891 0.6706 0.8478 rice_cammeo_and_osmancik 64 0.8662 0.9134 0.7612 0.8937 rice_cammeo_and_osmancik 128 0.8635 0.916 0.7703 0.8976 rice_cammeo_and_osmancik 256 0.8675 0.9081 0.7743 0.9029 rice_cammeo_and_osmancik 512 0.8806 0.9081 0.811 0.8058 shill_bidding 8 0.8822 0.8656 0.845 0.8205 shill_bidding 16 0.8759 0.8814 0.8466 0.838 shill_bidding 32 0.8704 0.8799 0.8348 0.8553 shill_bidding 64 0.8885 0.8775 0.8553 0.8775 shill_bidding 128 0.8798 0.879 0.8727 0.8838 shill_bidding 256 0.8814 0.8735 0.8672 0.8877 shill_bidding 512 0.8861 0.8751 0.8783 0.8973 shrutime 8 0.7465 0.747 0.6965 0.732 shrutime 16 0.75 0.775 0.7425 0.7665 shrutime 32 0.7445 0.7935 0.756 0.7835 shrutime 64 0.741 0.803 0.7695 0.789 shrutime 128 0.7365 0.8155 0.774 0.7965 shrutime 256 0.743 0.825 0.7875 0.8125 shrutime 512 0.753 0.84 0.792 0.824 shuttle 8 0.7998 0.8302 0.6834 0.7713 shuttle 16 0.8209 0.8661 0.7357 0.8181

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

shuttle 32 0.8523 0.9063 0.7641 0.876 shuttle 64 0.8956 0.958 0.7905 0.9319 shuttle 128 0.9443 0.9873 0.8018 0.9541 shuttle 256 0.9706 0.9941 0.7899 0.9626 shuttle 512 0.9818 0.9952 0.8112 0.8215 splice 8 0.4624 0.4608 0.4107 0.4671 splice 16 0.4624 0.5204 0.3887 0.4467 splice 32 0.5094 0.6269 0.4828 0.5 splice 64 0.5344 0.7053 0.4294 0.5047 splice 128 0.5643 0.79 0.4749 0.5203 splice 256 0.6191 0.8511 0.4546 0.5345 splice 512 0.6724 0.906 0.4874 0.5298 sylvine 8 0.5073 0.599 0.6068 0.5571 sylvine 16 0.5444 0.7434 0.6215 0.6693 sylvine 32 0.5697 0.8263 0.601 0.7395 sylvine 64 0.5707 0.882 0.6098 0.8078 sylvine 128 0.6107 0.8858 0.6644 0.8283 sylvine 256 0.6683 0.9063 0.6547 0.8556 sylvine 512 0.7073 0.9073 0.5991 0.84 telco_customer_churn 8 0.7161 0.6834 0.6452 0.6387 telco_customer_churn 16 0.7218 0.7303 0.6899 0.6742 telco_customer_churn 32 0.7303 0.7303 0.6707 0.7189 telco_customer_churn 64 0.7608 0.7814 0.7069 0.7587 telco_customer_churn 128 0.7616 0.7814 0.6991 0.7637 telco_customer_churn 256 0.7573 0.7722 0.7005 0.7714 telco_customer_churn 512 0.7665 0.7843 0.6934 0.7842 texture 8 0.0375 0.0375 0.1273 0.2118 texture 16 0.0625 0.05 0.1709 0.2664 texture 32 0.0844 0.0625 0.1964 0.3927 texture 64 0.1 0.0969 0.21 0.53 texture 128 0.1906 0.2437 0.2427 0.6464 texture 256 0.3781 0.3563 0.2436 0.6455 texture 512 0.4844 0.5312 0.2365 0.6506 thyroid_ann 8 0.9033 0.9298 0.8781 0.9272 thyroid_ann 16 0.9205 0.9245 0.9179 0.9245 thyroid_ann 32 0.9178 0.9391 0.9192 0.9324 thyroid_ann 64 0.9179 0.947 0.9086 0.9245 thyroid_ann 128 0.9179 0.9537 0.9179 0.9325 thyroid_ann 256 0.9206 0.9589 0.9166 0.9073 thyroid_ann 512 0.9232 0.9762 0.9231 0.9232 thyroid_dis 8 0.5464 0.5607 0.4268 0.4553 thyroid_dis 16 0.55 0.5893 0.4179 0.5625 thyroid_dis 32 0.5322 0.5947 0.5018 0.5875 thyroid_dis 64 0.5268 0.6464 0.4947 0.65 thyroid_dis 128 0.5464 0.6661 0.4304 0.6411 thyroid_dis 256 0.5947 0.6964 0.4929 0.6768 thyroid_dis 512 0.5768 0.7036 0.3929 0.6572 turiye_student_evaluation 8 0.2646 0.2466 0.2294 0.238 turiye_student_evaluation 16 0.2655 0.2835 0.2388 0.25 turiye_student_evaluation 32 0.3127 0.3359 0.2543 0.2809 turiye_student_evaluation 64 0.3497 0.3858 0.2809 0.3127 turiye_student_evaluation 128 0.3823 0.4304 0.2843 0.3634 turiye_student_evaluation 256 0.4004 0.4399 0.3016 0.3832 turiye_student_evaluation 512 0.4098 0.4587 0.3033 0.4175 walking_activity 8 0.1241 0.1355 0.0192 0.0138 walking_activity 16 0.14 0.1702 0.0175 0.0092 walking_activity 32 0.1502 0.2148 0.0164 0.0089 walking_activity 64 0.166 0.2593 0.0154 0.0135 walking_activity 128 0.1848 0.3094 0.0173 0.0267 walking_activity 256 0.2169 0.3573 0.0167 0.0508 walking_activity 512 0.2577 0.4078 0.0155 0.0629

Table 7 – continued from previous page

##### Dataset Shots KNN RF Qwen Our

water_quality 8 0.8856 0.8763 0.8275 0.8619 water_quality 16 0.8857 0.8694 0.8294 0.8644 water_quality 32 0.8819 0.8781 0.8656 0.8688 water_quality 64 0.8825 0.8782 0.8469 0.8807 water_quality 128 0.8819 0.8775 0.8531 0.885 water_quality 256 0.8813 0.8769 0.8737 0.8893 water_quality 512 0.8819 0.8825 0.8737 0.8907 wine_quality_white 8 0.3694 0.4051 0.3429 0.3296 wine_quality_white 16 0.3908 0.4163 0.3888 0.3929 wine_quality_white 32 0.4031 0.4561 0.3939 0.4459 wine_quality_white 64 0.4194 0.4663 0.4306 0.4337 wine_quality_white 128 0.4072 0.4888 0.4245 0.4439 wine_quality_white 256 0.4439 0.4867 0.4234 0.4806 wine_quality_white 512 0.4541 0.5224 0.402 0.5071

