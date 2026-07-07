# arXiv:2601.02151v1[cs.LG]5Jan2026

## Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting

Muxi Diao1,2*, Lele Yang1*, Wuxuan Gong1*, Yutong Zhang1, Zhonghao Yan1, Yufei Han1, Kongming Liang1, Weiran Xu1, Zhanyu Ma1† 1Beijing University of Posts and Telecommunications, 2Zhongguancun Academy {dmx, yang_happy, mazhanyu}@bupt.edu.cn

GitHub: https://github.com/PRIS-CV/EAFT

### Abstract

Supervised Fine-Tuning (SFT) is the standard paradigm for domain adaptation, yet it frequently incurs the cost of catastrophic forgetting. In sharp contrast, on-policy Reinforcement Learning (RL) effectively preserves general capabilities. We investigate this discrepancy and identify a fundamental distributional gap: while RL aligns with the model’s internal belief, SFT forces the model to fit external supervision. This mismatch often manifests as "Confident Conflicts"—tokens characterized by low probability but low entropy. In these instances, the model is highly confident in its own prediction but is forced to learn a divergent ground truth, triggering destructive gradient updates. To address this, we propose Entropy-Adaptive Fine-Tuning (EAFT). Unlike methods relying solely on prediction probability, EAFT utilizes token-level entropy as a gating mechanism to distinguish between epistemic uncertainty and knowledge conflict. This allows the model to learn from uncertain samples while suppressing gradients on conflicting data. Extensive experiments on Qwen and GLM series (ranging from 4B to 32B parameters) across mathematical, medical, and agentic domains confirm our hypothesis. EAFT consistently matches the downstream performance of standard SFT while significantly mitigating the degradation of general capabilities.

### 1 Introduction

Supervised Fine-Tuning (SFT) is the standard method for adapting Large Language Models (LLMs) to specific domains (e.g., mathematics or agentic tool-use) (Yang et al., 2024; Shao et al., 2024; Team et al., 2025). However, this paradigm often comes with a significant cost known as catastrophic forgetting (Kirkpatrick et al., 2017; Ouyang et al., 2022). Previous studies have extensively doc-

*Equal contribution. †Corresponding author.

umented that while fitting specific target distributions, models frequently suffer from degradation in general capabilities (Ouyang et al., 2022; Luo et al., 2023). In contrast, on-policy Reinforcement Learning (RL) has shown a remarkable ability to improve domain-specific performance while effectively preserving the robustness of the base model (Chen et al., 2025; Shenfeld et al., 2025). This sharp contrast raises a fundamental question:

Why does SFT frequently degrade general abilities, while on-policy RL preserves them?

To investigate the mechanisms behind this phenomenon, we systematically analyze the tokenlevel probability and entropy of the training data. As visualized in Fig. 1, this analysis reveals a distinct distributional gap arising from different data sources. In on-policy RL, training sequences are generated via self-rollout; consequently, the tokens inherently align with the model’s current probability landscape, falling into either high-probability confidence zones or high-entropy exploration regions. Conversely, SFT relies on external supervision (e.g., humans or strong teacher models), introducing a mismatch manifested as low-probability, low-entropy tokens. Crucially, this mismatch manifests as tokens characterized by low probability yet low entropy. This specific region corresponds to scenarios where the model is highly confident in its own prediction (low entropy) but is forced to fit a divergent ground-truth label (low probability). We term these instances "Confident Conflicts". See App. A for representative word clouds.

To verify whether these conflicts are indeed the drivers of forgetting, we conducted a pilot experiment. By simply masking out these "Confident Conflict" tokens during training (Fig. 2). We observed that catastrophic forgetting was significantly mitigated compared to standard SFT. This confirms that enforcing updates on these conflicting samples is the primary driver of capability degradation.

[Figure 1]

- Figure 1: (a) Conceptual illustration. When SFT forces the model to override its strong priors (e.g., labeling a “ball” as a “truncated icosahedron”), it creates a Confident Conflict. Fitting these conflicts distorts the model’s existing representations, leading to catastrophic forgetting. (b) Token-level entropy–probability landscape. Compared to on-policy rollouts (right), the SFT data (left) exhibits a prominent cluster of Low Entropy, Low Probability tokens.

[Figure 2]

- Figure 2: (a) Masking “Confident Conflict” tokens (the bottom 15% in both entropy and probability) effectively mitigates the general capability degradation observed in standard SFT. (b) Across Math, Medical, and Agent domains, EAFT matches SFT in target task improvements (upper bars) while significantly minimizing performance drops on general benchmarks (lower bars).

and agent domains. Our comprehensive evaluation covers diverse model families (Qwen, GLM) and scales ranging from 4B to 32B parameters. The results are presented in Fig. 2 and Tab. 1.

Quantitative results (Sec. 4.2) demonstrate that EAFT consistently outperforms standard SFT and existing mitigation strategies. It achieves a Pareto improvement: matching or exceeding baselines on target tasks while significantly mitigating catastrophic forgetting on general benchmarks.

Beyond performance, we provide an in-depth analysis of the method’s intrinsic properties. We empirically verify that the entropy-adaptive mechanism successfully targets "Confident Conflicts" (Sec. 4.3), and further demonstrate that EAFT is both robust to hyperparameter variations (Sec. 5.1) and computationally efficient (Sec. 5.2).

Building on this insight, we propose EntropyAdaptive Fine-Tuning (EAFT). Instead of using discrete thresholds, EAFT employs a soft gating mechanism that dynamically modulates the training loss based on token-level entropy.

In summary, our contributions are as follows:

- • We uncover the distinct distributional gap between SFT and on-policy RL data. Through visualization and pilot experiments, we pinpoint "Confident Conflicts" (low-entropy, lowprobability tokens) as the primary cause of catastrophic forgetting.
- • We propose Entropy-Adaptive Fine-Tuning (EAFT), a novel objective that utilizes tokenlevel entropy to modulate the training loss. This mechanism automatically down-weights destructive updates from conflicting data.
- • We validate our approach through extensive experiments on Math, Agent, and Medical domains. The results establish EAFT as an effective and universal solution that successfully mitigates catastrophic forgetting across diverse model families and scales (4B–32B).

Crucially, this approach differentiates EAFT from standard Cross-Entropy or probability-based re-weighting strategies (Wu et al., 2025; Lin et al., 2025; Sanyal et al., 2025; Shenfeld et al., 2025). These methods rely solely on prediction probability, and thus risk amplifying destructive gradients on "Confident Conflicts." In contrast, EAFT leverages entropy to distinguish rigidity from uncertainty. By down-weighting low-entropy tokens to suppress conflicting gradients, while concentrating supervision on high-entropy ones to facilitate adaptation, EAFT effectively balances domain proficiency with the preservation of general capabilities.

To verify the effectiveness and universality of our approach, we validate EAFT through extensive experiments on mathematical, extending to medical

### 2 Related Work

Post-training Paradigms: SFT vs. RL. Posttraining methods, primarily Supervised FineTuning (SFT) and Reinforcement Learning (RL), are widely used to align pre-trained LMs (Qwen et al., 2024; GLM et al., 2024; Yang et al., 2025). SFT optimizes the model to maximize the likelihood of ground-truth demonstrations (Off-policy). By contrast, RL optimizes the model based on its own generated responses guided by reward signals (On-policy). These signals typically originate from parameterized reward models or verifiable signals (Schulman et al., 2017; Ouyang et al., 2022; Shao et al., 2024; Zeng et al., 2025).

Emerging research highlights a fundamental dichotomy in their learning behaviors. While SFT is efficient, it is inherently prone to memorization, often fits specific training samples at the expense of generalization (Chu et al., 2025). RL demonstrates superior robustness: it can benefit from single training examples without severe overfitting (Wang et al., 2025), and it updates a smaller, more effective subspace of parameters compared to SFT (Mukherjee et al., 2025). A common thread connecting these results is that RL parameter updates are more local and targeted (Razin et al., 2023).

Our work investigates the root cause of SFT’s instability. We argue that unlike on-policy methods which naturally operate within the model’s distribution, standard SFT indiscriminately forces the fitting of "confident conflicts"—low-entropy samples that contradict the model’s pre-trained knowledge.

Catastrophic forgetting. Catastrophic forgetting remains a foundational challenge in neural networks (McCloskey and Cohen, 1989; Kirkpatrick et al., 2017). Initial efforts to mitigate forgetting focused on preventing parameters from drastically changing (Kirkpatrick et al., 2017; Li and Hoiem, 2017; Lopez-Paz and Ranzato, 2017).

In LLM post-training, this manifests as the "Alignment Tax" (Askell et al., 2021; Ouyang et al., 2022): fine-tuning for domain-specific capabilities (e.g., mathematical problem solving, tool utilization, or biomedical adaptation) often significantly degrades the model’s general capabilities (Ouyang et al., 2022; Luo et al., 2023; Shi et al., 2025).

To overcome these limitations, recent works have explored dynamic training strategies that adjust optimization based on token-level metrics. TALR (Lin et al., 2025) dynamically scales learning rates based on token confidence to accelerate

convergence. DFT (Wu et al., 2025) re-weights the SFT loss according to prediction probability. Others like RL’s Razor (Shenfeld et al., 2025) employ KL divergence as a regularization term to constrain the model’s drift from its base distribution.

However, existing dynamic methods predominantly rely on probability or KL divergence as proxies for difficulty or drift. We argue that probability alone is an insufficient statistic: a low-probability token can represent either epistemic uncertainty (valid knowledge to be learned) or a "Confident Conflict" (a destructive sample that contradicts the model’s strong priors). By forcing the model to fit these conflicts based on probability, prior methods risk accelerating forgetting. Our work advances this by introducing Entropy as a gating signal.

### 3 Empirical Analysis & Methodology

In this section, we systematically investigate the causes of catastrophic forgetting in SFT and propose a targeted solution. We begin by defining the problem setup and key metrics in Sec. 3.1. We then present an empirical analysis identifying ‘Confident Conflicts’ as the primary source of destructive gradients in Sec. 3.2. Finally, based on these insights, we introduce our method, EntropyAdaptive Fine-Tuning (EAFT), in Sec. 3.3.

#### 3.1 Preliminaries

SFT is the standard process of adapting a base model θ, denoted by its probability distribution Pθ, to a target dataset D = {(x,y)i}Ni=1. For each sample, the response is a sequence of tokens y = (y1,...,yT), where T denotes the sequence length. The adaptation is typically achieved by minimizing the Cross-Entropy (CE) loss, which maximizes the likelihood of the target sequences:

T

log Pθ(yt|x,y<t) (1)

LCE(θ) = −

t=1

A key limitation of this objective is its uniform treatment of all tokens. It aggressively updates model parameters to fit every token yt regardless of the model’s prior knowledge or uncertainty.

To investigate the dynamics of how this uniform objective interacts with the model’s internal state, we introduce two token-level metrics that serve as the foundation for our analysis and method:

1. Probability. pt = Pθ(yt|x,y<t), represents model’s confidence in the ground-truth token.

- 2. Predictive Entropy. Let Pt(v) ≜ Pθ(v|x,y<t) denote the distribution at step t. The entropy is

defined as: Ht = − v∈V Pt(v)log Pt(v) This measures the model’s predictive uncertainty over

the vocabulary V.

- 3.2 Analysis: The Origins of Forgetting

To understand why SFT leads to forgetting while on-policy RL does not, we compare the tokenlevel statistics of standard SFT data against modelgenerated rollouts (the data source for on-policy RL). We visualize the distribution of probability pt and entropy Ht for both datasets in Fig. 1.

Distributional Gap: Confident Conflicts. The visualization reveals a critical distributional shift. On-policy data falls into either high-probability (model is correct) or high-entropy (model is exploring). In sharp contrast, SFT data contains a significant cluster of tokens with both Low Entropy (Ht ↓) and Low Probability (pt ↓). We term these samples ‘Confident Conflicts’. They represent cases where the model holds a strong, stubborn prior belief (low entropy) that directly contradicts the ground-truth label (low probability).

Pilot Study: Masking Confident Conflicts. We hypothesize that these ‘Confident Conflicts’ are the primary drivers of forgetting. To verify this, we conducted a pilot experiment where we masked the loss for tokens falling within the bottom 15% of both entropy and probability rankings. As shown in Fig. 2, this simple intervention significantly mitigates the general capability degradation observed in standard SFT.

Notably, masking these specific tokens nearly eliminated catastrophic forgetting on our benchmarks. This finding confirms that the degradation of general capabilities stems primarily from forcing the model to accommodate these conflicting samples, rather than from the SFT process itself.

Theoretical Insight. We analyze the optimization dynamics to understand this damage. Consider the CE loss (Eq. 1). When the model is highly confident in a prediction that contradicts the target (low entropy, low probability), the CE loss induces a very large gradient. Because the model strongly favors another token, fitting the target requires substantial parameter updates, which can overwrite general representations in the base model. By contrast, when the model is uncertain (high entropy), the gradients are smaller and updates are gentler, helping preserve the model’s original capabilities.

#### 3.3 Entropy-Adaptive Fine-Tuning (EAFT)

While the pilot study validates our hypothesis, the hard masking strategy has two limitations: it discards training data, leads to ineffective learning on the target domain, and it relies on sensitive hyperparameters (τ,δ). To address this, we propose Entropy-Adaptive Fine-Tuning (EAFT), a soft gating mechanism that dynamically adjusts the learning signal based on the model’s uncertainty.

The EAFT Objective. We formulate the EAFT loss by scaling the standard supervision with the normalized entropy. This mechanism prioritizes learning from samples where the model is exploring, while effectively suppressing the gradients when the model is confident but conflicting. The objective is decomposed as:

T

H˜t Adaptive Gating Signal

LEAFT(θ) = −

##### ·log Pθ(yt|x,y<t)

t=1

Standard Supervision

(2)

Here, the gating term H˜t is derived from the entropy of the Top-K tokens. This approximation greatly reduces computation compared with using the full vocabulary (analysis in Sec. 5.2). Normalized to the range [0,1] with K = 20, we calculate:

Httop-K ln(K) ≈

Httop-20 3.0

H˜t =

(3)

where Httop-K denotes the entropy calculated over the top-K probability distribution, and ln(K) serves as the normalization factor (the maximum entropy for K outcomes). This normalization creates a self-regulating mechanism:

- • Conflict Suppression (H˜t → 0): When the model is stubborn (low entropy), the weight drops, effectively masking the destructive gradient from the conflicting label.
- • Knowledge Acquisition (H˜t → 1): When the model is uncertain (high entropy) or exploring, the weight remains high, recovering the standard SFT objective to learn new patterns.

### 4 Experiments

In this section, we empirically validate the effectiveness of EAFT. Our experiments are designed to answer the following three key research questions:

• RQ1 (Performance): Can EAFT mitigate catastrophic forgetting without compromising performance on the target task?

Math Domain

General Domain

Method

General Avg. AIME24 AIME25 GSM8K MMLU IFEval CLUEWSC

Math Avg.

Qwen3-4B-Instruct 63.3 47.4 94.3 68.3 77.1 81.0 85.2 81.1

+ SFT 63.3 50.0 94.8 69.4 76.5 79.5 74.5 76.5 (-4.6) + SFTKL 63.3 50.0 93.6 69.0 74.5 74.9 89.4 79.6 (-1.5) + FLOW 66.7 46.7 94.3 69.2 76.2 78.3 82.8 79.1 (-2.0) + DFT 56.7 40.0 93.9 63.5 75.9 77.0 81.4 78.1 (-3.0) + TALR 50.0 50.0 93.3 64.4 76.2 78.1 74.5 76.2 (-4.9) + EAFT 60.0 53.3 94.5 69.3 76.6 80.1 83.7 80.1 (-1.0)

Qwen2.5-32B-Instruct 22.2 13.3 96.0 43.8 84.1 78.3 91.9 84.8

+ SFT 53.3 50.0 96.3 66.5 76.9 74.2 93.8 81.6 (-3.2) + SFTKL 33.3 33.3 94.1 53.6 81.4 68.1 93.2 80.9 (-3.9) + FLOW 50.0 50.0 96.3 65.4 78.6 75.1 93.6 82.4 (-2.4) + DFT 33.3 36.7 95.9 55.3 77.8 70.0 94.4 80.7 (-4.1) + TALR 40.0 43.3 95.3 59.5 73.1 72.5 94.1 79.9 (-4.9) + EAFT 53.3 46.7 96.5 65.5 79.0 78.4 93.9 83.7 (-1.1)

GLM4-9B-0414 6.7 6.7 90.1 34.5 70.2 74.4 85.1 76.6

+ SFT 20.0 10.0 90.3 40.1 57.3 69.8 84.8 70.6 (-6.0) + SFTKL 13.3 6.7 90.1 36.7 60.0 66.4 85.3 70.5 (-6.1) + FLOW 16.7 13.3 91.1 40.4 57.5 71.5 85.2 71.4 (-5.2) + DFT 13.3 6.7 89.0 36.4 48.9 69.7 86.0 68.2 (-8.4) + TALR 15.6 13.3 91.2 40.0 57.4 71.3 84.5 71.5 (-5.1) + EAFT 13.3 13.3 91.5 39.4 60.8 72.0 85.3 72.7 (-3.9)

- Table 1: Main results on the target domain (Math) and general domain benchmarks. We evaluate performance on AIME24, AIME25 and GSM8K as the training target, alongside MMLU, IFEval, and CLUEWSC for general capabilities. The top two outcomes are bolded and underlined. All results are averaged over three independent runs. The “Avg.” represents the average performance of the datasets in the corresponding domain.

- • RQ2 (Mechanism): Does the entropy-adaptive gating mechanism work as intended in filtering “Confident Conflict” samples?
- • RQ3 (Generalization): Is the efficacy of EAFT inherently domain-agnostic?

#### 4.1 Experimental Settings

Datasets. We utilize prompts from NuminaMath (Li et al., 2024), BigMathVerified (Albalak et al., 2025), and Nemotron-CrossThink (Akter et al., 2025), synthesizing responses via Qwen3235B-A22B-Instruct (Yang et al., 2025). We randomly select 19k correctly answered data pairs as our math training data. Details in App. B

Benchmarks. Our evaluation covers the target domain (Math: AIME 24/25 (AI-MO, 2025), GSM8K (Cobbe et al., 2021)) and general domain (MMLU (Hendrycks et al., 2020), IFEval (Zhou et al., 2023), CLUEWSC (Xu et al., 2020)). We report the average score (“Avg”) across benchmarks, with full details available in App. C.

Models. To verify the scalability and generalizability of our method, we conduct experiments across a diverse model zoo spanning multiple families and parameter scales. Specifically, our evaluation includes models ranging from 4B to 32B parameters: Qwen3-4B-Instruct (Yang et al., 2025), GLM4-

9B-0414 (GLM et al., 2024), and Qwen2.5-32BInstruct (Qwen et al., 2024). Details in App. D

Baselines. We compare our proposed method against standard Supervised Fine-Tuning (SFT) and a regularized variant, denoted as SFTKL, which incorporates a Kullback-Leibler (KL) divergence constraint into the loss function to prevent model drift. Additionally, we include several advanced alignment baselines, including FLOW (Sanyal et al., 2025), DFT (Wu et al., 2025), and TALR (Lin et al., 2025). Details and implementation settings are provided in App. E and App. F

4.2 Main Results (Answering RQ1)

Tab. 1 presents the performance comparison across diverse model families and scales. The results provide a compelling answer to RQ1: EAFT maintains competitive performance on target domain (Math) while significantly mitigating catastrophic forgetting on general capabilities.

General Capabilities. The main advantage of EAFT lies in stability. Standard SFT causes a sharp drop in general benchmarks (e.g., a 10.7 point drop on CLUEWSC for Qwen3-4B). In contrast, EAFT effectively preserves the model’s original knowledge, achieving the highest average score across general tasks among all baselines. This demon-

[Figure 3]

- Figure 3: Gradient Magnitude Landscape. Left: SFT exerts strong optimization pressure (dark purple) on Confident Conflicts in the bottom-left. Right: EAFT effectively suppresses these gradients (light yellow), protecting the model’s existing representations.

strates that EAFT offers the most robust capability retention compared to other methods.

Target Domain Performance. EAFT achieves highly competitive results on the target domain compared to other baselines. Specifically, the gap between EAFT and the best-performing method on math score is consistently less than 1 point. Notably, EAFT achieves the best performance on several sub-benchmarks. This demonstrates that EAFT effectively adapts to target domain, maintaining the same level of learning capability as standard SFT.

#### 4.3 Mechanism Analysis (Answering RQ2)

To understand the inner workings of EAFT, we analyze the training dynamics from both spatial and temporal perspectives. The results provide a clear affirmative answer to RQ2: EAFT effectively filters out "Confident Conflict" samples, preventing destructive gradient updates.

Gradient Landscape. Fig. 3 visualizes the optimization strength for each token, where color intensity represents the gradient magnitude. The gradient distribution exhibits a distinct skew rather than uniformity. Due to the nature of Cross-Entropy loss, low-probability tokens are subjected to the strongest optimization pressure (indicated by the dark density in the left region). Crucially, in the "Confident Conflict" zone (bottom-left), the model is forcefully updated to fit labels that contradict its confident priors. These aggressive updates destabilize the model’s established representation space. In contrast (right), EAFT dramatically alters this landscape. The "Confident Conflict" region becomes pale, indicating near-zero gradients. Here, the low entropy term directly suppresses the high gradients generated by the Cross-Entropy loss. This protects the model’s established knowledge from being over-optimized by conflicting data.

[Figure 4]

Figure 4: Training dynamics of token subgroups. EAFT matches SFT on high-entropy tokens while keeping losses stable on low-entropy conflicts, preventing overoptimization of conflicting priors. High and low entropy correspond to values ≥ 2.0 and ≤ 0.5, respectively.

Training Dynamics. Fig. 4 tracks the loss of cross-entropy of different token types throughout the training process, comparing EAFT (Blue) with Standard SFT (Red). We categorize tokens into high-entropy and low-entropy groups. HighEntropy Tokens (marked with dots •): EAFT exhibits a rapid loss reduction comparable to SFT. This confirms that our entropy-based gating effectively optimizes high-entropy tokens, ensuring the model adapts to the target domain without hindrance. Low-Entropy Tokens (marked with triangles ▲): SFT (Red) aggressively drives this loss toward zero, indicating that the model is being forced to memorize data that conflicts with its priors. In contrast, EAFT (Blue) maintains a stable loss throughout training. This demonstrates that the mechanism successfully prevents over-optimization on "Confident Conflict" tokens.

#### 4.4 Universality (Answering RQ3)

To verify whether the efficacy of EAFT is domainagnostic, we extend our evaluation to two distinct specialized domains: Biomedical (knowledgeintensive) and Agent Tool-Use (syntax-intensive). The results clearly answer RQ3: EAFT is a domain-agnostic solution that consistently mitigates catastrophic forgetting across tasks.

Medical Domain. We first examine the medical domain, a field demanding domain-specific knowledge application. For this experiment, we finetune the Qwen3-4B-Thinking (Yang et al., 2025) model using the Huatuo-O1 (Chen et al., 2024) dataset and conduct evaluations on the MedMCQA (Pal et al., 2022),PubMedQA (Jin et al., 2019)and MedQA (Jin et al., 2021) benchmarks (see App.

Medical Domain

General Domain

Method

General Avg. MedMCQA MedQA PubMedQA MMLU IFEval CLUEWSC

Medical Avg.

Qwen3-4B-Thinking 63.5 78.2 76.0 72.6 79.3 85.0 94.1 86.1

+ SFT 63.3 79.5 78.0 73.6 78.3 75.3 90.4 81.3 (-4.8) + EAFT 63.9 80.0 77.2 73.7 80.1 81.7 91.8 84.5 (-1.6)

- Table 2: Results on target (Medical) and General domain benchmarks. We evaluate performance on MedMCQA, MedQA, and PubMedQA for the medical domain, alongside MMLU, IFEval, and CLUEWSC for general capabilities. The top two outcomes are bolded and underlined. All results are averaged over three independent runs.

Method

Agent Domain General Domain

General Avg. BFCL v3 MMLU IFEval CLUEWSC

Qwen3-4B-Instruct 60.5 77.1 81.0 85.2 81.1

+ SFT 61.4 74.5 77.8 72.2 74.8 (-6.3) + EAFT 60.8 76.1 78.6 77.7 77.5 (-3.6)

- Table 3: Results on target (Agent Toolcall) and General domain benchmarks. We evaluate performance on BFCL for the agent toolcall domain, alongside MMLU, IFEval, and CLUEWSC for general capabilities. The top two outcomes are bolded and underlined. All results are averaged over three independent runs.

C for more details). In Tab. 2, standard SFT triggers severe catastrophic forgetting while adapting to the target domain, causing average performance on general benchmarks to drop significantly from 86.1 to 81.3. EAFT effectively mitigates this issue. It not only preserves general capabilities (maintaining an average of 84.5) but also marginally outperforms standard SFT on the target medical tasks (73.7 vs. 73.6). This result suggests that EAFT can inject specific knowledge without destructively overwriting the model’s core representation.

Agent Tool-Use. We further evaluate EAFT on agentic tool-use tasks, which require strict adherence to syntactic constraints and formats. Specifically, we utilize the subset of Nemotron-AgenticTool-Use-v1 (NVIDIA, 2025) to train the Qwen34B-Instruct (Yang et al., 2025) and assess its performance on the Berkeley Function Calling Leaderboard (BFCL) (Patil et al.) (see App. C). The results in Tab. 3 reveal a similar pattern regarding robustness. Although SFT fits the target distribution aggressively to achieve a slightly higher score (61.4), it does so at the cost of catastrophic forgetting, resulting in a sharp decline in general capabilities (81.1 → 74.8). In contrast, EAFT achieves a far superior balance: it remains highly competitive on the target task (60.8, within 1% of SFT) while maintaining general performance (77.5). These experiments confirm that EAFT is a domain-agnostic solution that consistently alleviates catastrophic forgetting across varying data distributions.

### 5 Analysis and Discussion

Having demonstrated the superior performance of EAFT, this section investigates the intrinsic robustness and computational efficiency of the proposed mechanism. Specifically, we conduct ablation studies to answer two fundamental questions:

- • Robustness: Does performance rely on linear entropy scaling, or is the "entropy-aware" mechanism the primary driver? (Sec. 5.1)
- • Efficiency: Can we approximate the token-level entropy efficiently without incurring significant computational overhead? (Sec. 5.2)

5.1 Robustness to Gating Function Variations In Sec. 3, we adopted a linear gating function L = H˜t · LCE. To verify that our gains stem from the intrinsic mechanism of entropy awareness rather than a specific hyperparameter choice, we generalize the gating signal to f(H˜t) and evaluate three distinct categories of variants:

- • Polynomial Scaling: We test stricter suppression of confident samples using power functions

f(H˜t) = (H˜t)p with p ∈ {2,3}. We denote these variants as EAFT2 and EAFT3.

- • Non-linear Gating (Sigmoid): We employ a

Sigmoid activation f(H˜t) = σ α(H˜t − β) where α controls the steepness and β determines the centering threshold. In our experiments, we set β = 0.17 (aligning with the bottom 15% entropy percentile used in the Masked SFT baseline) and α = 30. This variant is denoted as EAFTsig. See App. G for further comparisons.

[Figure 5]

Figure 5: Pareto trade-off analysis. Unlike Masked SFT (drop in target score) or Standard SFT (severe forgetting), EAFT variants consistently occupy the optimal top-right frontier. This confirms that soft entropygating effectively preserves general capabilities without compromising target domain adaptation.

• Piecewise Hard Thresholding (Hard Mask): We implement a binary filter that strictly discards the bottom 15% lowest entropy tokens:

##### f(H˜t) = I(H˜t > τ0.15) (4)

where I(·) is the indicator function and τ0.15 denotes the threshold value corresponding to the 15th percentile of the entropy distribution. This sets the loss of "confident" tokens to zero, retaining only uncertain samples for training. We refer to this baseline as Masked SFT.

Results and Discussion. Fig. 5 visualizes the trade-off between target domain performance (math average score) and general average score across all variants. We observe two pivotal insights:

Universality of Entropy Awareness. All entropy-aware variants (EAFT, EAFT2, EAFT3, EAFTsig) consistently outperform SFT in general capabilities. This confirms that the reduction in forgetting stems from the core mechanism of entropy monitoring rather than the specific mathematical form. Simply down-weighting high-conflict tokens is sufficient to preserve general abilities.

The Necessity of Soft Gating. We observe a critical pitfall in the Mask baseline: while strictly discarding confident tokens prevents forgetting, it significantly harms target task performance (Math Score drops to 65.60 vs. EAFT’s 69.27). This implies that "confident conflicts" carry essential adaptation signals that a hard cutoff destroys. In contrast, EAFT’s soft gating reduces their impact without removing them, successfully occupying the Pareto frontier of learning versus retaining.

[Figure 6]

Figure 6: Trade-off between Approximation Accuracy and Memory Cost. The Red solid line (Left Axis) shows the Pearson correlation between Top-K and exact entropy, which saturates rapidly at 0.999. The Blue dashed line (Right Axis) tracks the additional memory overhead. The green shaded region highlights K = 20 as the optimal operating point, achieving near-perfect fidelity with negligible computational cost.

#### 5.2 Efficiency of Top-K Approximation

Computing exact entropy over the entire vocabulary (often |V| > 100k) creates unnecessary computational bottlenecks. In Sec. 3, we proposed a Top-K approximation strategy (H˜ ≈ Htop-K), premised on the property that the probability mass of LLMs is highly sparse and concentrated in the leading tokens (Fan et al., 2018; Radford et al., 2019; Holtzman et al., 2019).

Fig. 6 quantitatively validates this design by plotting approximation fidelity (Left Axis, Red) against memory cost (Right Axis, Blue). We observe a decisive trade-off: as K increases, the Pearson correlation with the exact entropy rises sharply and rapidly plateaus. Specifically, at K = 20 (green zone), the correlation reaches 0.999, confirming that the long-tail distribution contributes negligibly to entropy. Conversely, the memory cost for the required operations (sort and log-sum-exp) remains virtually zero (< 0.4 KB) for K ≤ 20. Based on these findings, we adopt the Top-20 approximation to estimate entropy. This setting provides an accurate estimate of the full-vocabulary entropy, while introducing no noticeable computational overhead compared to standard SFT.

### 6 Conclusion

In this work, we identify "Confident Conflicts" as the primary driver of catastrophic forgetting in SFT. We introduce Entropy-Adaptive Fine-Tuning (EAFT), a method that dynamically modulates training loss based on token-level entropy. By suppressing gradients from conflicting data, EAFT

effectively prevents destructive updates while maintaining learning efficiency. Extensive experiments across diverse domains and model scales validate our approach. Ultimately, EAFT provides a simple yet robust solution for balancing domain adaptation with the preservation of general capabilities.

### 7 Limitations

Scope of Applicability (Counterfactual Scenarios). It is important to clarify that EAFT is designed primarily for domain adaptation and continual learning, where the goal is to extend the model’s capabilities without erasing existing knowledge. It is not a universal replacement for standard SFT, particularly in scenarios requiring knowledge editing or counterfactual training (e.g., teaching the model that “the sky is green” or correcting outdated facts). In such cases, the model’s resistance to "Confident Conflicts" is undesirable, as the objective is precisely to override the prior belief. EAFT would interpret these necessary updates as conflicts and suppress them, hindering the intended learning.

Target Performance Trade-off. While EAFT successfully mitigates catastrophic forgetting, it does not aim to surpass standard SFT on the target domain metrics. As observed in our experiments, EAFT achieves a Pareto improvement—maintaining high general capabilities while closely approaching, but not necessarily exceeding, the peak specialization performance of SFT. For applications where maximizing target domain performance is the sole priority, regardless of the degradation in other areas, standard SFT may still be the preferred choice.

Dependence on Base Model Quality. EAFT operates on the assumption that the model’s highconfidence priors represent valuable general knowledge worth preserving. However, if the base model exhibits miscalibrated confidence (e.g., being confidently wrong or hallucinating), EAFT inadvertently protect these erroneous behaviors. Future work could explore incorporating uncertainty calibration techniques to distinguish between true knowledge and confident hallucinations.

### References

AI-MO. 2025. Ai-mo/aimo-validation-aime · datasets at hugging face. https://huggingface.co/ datasets/AI-MO/aimo-validation-aime. Online; accessed 2025-12-28.

Syeda Nahida Akter, Shrimai Prabhumoye, Matvei

Novikov, Seungju Han, Ying Lin, Evelina Bakhturina, Eric Nyberg, Yejin Choi, Mostofa Patwary, Mohammad Shoeybi, and 1 others. 2025. Nemotroncrossthink: Scaling self-learning beyond math reasoning. arXiv preprint arXiv:2504.13941.

Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, and 1 others. 2025. Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models. arXiv preprint arXiv:2502.17387.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, and 1 others. 2021. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861.

Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen. 2025. Retaining by doing: The role of on-policy data in mitigating forgetting. arXiv preprint arXiv:2510.18874.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. 2024. Huatuogpt-o1, towards medical complex reasoning with llms. arXiv preprint arXiv:2412.18925.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. 2025. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. arXiv preprint arXiv:1805.04833.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, and 37 others. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. Preprint, arXiv:2406.12793.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseekr1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. 2019. Pubmedqa: A dataset for biomedical research question answering. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 2567–2577.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, and 1 others. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, and 1 others. 2024. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9.

Zhizhong Li and Derek Hoiem. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947.

Jiacheng Lin, Zhongruo Wang, Kun Qian, Tian Wang, Arvind Srinivasan, Hansi Zeng, Ruochen Jiao, Xie Zhou, Jiri Gesi, Dakuo Wang, and 1 others. 2025. Sft doesn’t always hurt general capabilities: Revisiting domain-specific fine-tuning in llms. arXiv preprint arXiv:2509.20758.

David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2023. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. arXiv e-prints, pages arXiv– 2308.

Michael McCloskey and Neal J Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation, volume 24, pages 109–165. Elsevier.

Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. 2025. Reinforcement learning finetunes small subnetworks in large language models. arXiv preprint arXiv:2505.11711.

NVIDIA. 2025. Nemotron-agentic-v1 dataset. Hugging Face. Accessed: 2026-01-04.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. 2022. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, pages 248–260. PMLR.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 24 others. 2024. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, and 1 others. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Noam Razin, Hattie Zhou, Omid Saremi, Vimal Thilak, Arwen Bradley, Preetum Nakkiran, Joshua Susskind, and Etai Littwin. 2023. Vanishing gradients in reinforcement finetuning of language models. arXiv preprint arXiv:2310.20703.

Sunny Sanyal, Hayden Prairie, Rudrajit Das, Ali Kavis, and Sujay Sanghavi. 2025. Upweighting easy samples in fine-tuning mitigates forgetting. arXiv preprint arXiv:2502.02797.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. 2025. Rl’s razor: Why online reinforcement learning forgets less. arXiv preprint arXiv:2509.04259.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible

and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna Ebrahimi, and Hao Wang. 2025. Continual learning of large language models: A comprehensive survey. ACM Computing Surveys, 58(5):1–42.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, and 1 others. 2025. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Bo Wang, Qinyuan Cheng, Runyu Peng, Rong Bao, Peiji Li, Qipeng Guo, Linyang Li, Zhiyuan Zeng, Yunhua Zhou, and Xipeng Qiu. 2025. Implicit reward as the bridge: A unified view of sft and dpo connections. arXiv preprint arXiv:2507.00018.

Yongliang Wu, Yizhou Zhou, Zhou Ziheng, Yingzhe Peng, Xinyu Ye, Xinting Hu, Wenbo Zhu, Lu Qi, Ming-Hsuan Yang, and Xu Yang. 2025. On the generalization of sft: A reinforcement learning perspective with reward rectification. arXiv preprint arXiv:2508.05629.

Liang Xu, Hai Hu, Xuanwei Zhang, Lu Li, Chenjie Cao, Yudong Li, Yechen Yxn, Shushan Bai, Man Shu, and Xiangang Xi. 2020. CLUE: A Chinese Language Understanding Evaluation Benchmark. In Proceedings of the 28th International Conference on Computational Linguistics, pages 4762–4772.

Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, and 1 others. 2025. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. arXiv preprint arXiv:2506.07044.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, and 1 others. 2024. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. 2025. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892.

Yifan Zhang and Team Math-AI. 2025. American invitational mathematics examination (aime) 2025.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. Association for Computational Linguistics.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

### Appendix

- A Qualitative Analysis: What are "Confident Conflicts"? 12
- B Open-source Datasets 13

- B.1 Mathematical Datasets . . . . . . 13
- B.2 Medical Datasets . . . . . . . . . 13
- B.3 Agent Datasets . . . . . . . . . . 13

- C Benchmarks 13

- C.1 General Benchmarks . . . . . . . 13
- C.2 Mathematical Benchmarks . . . . 14
- C.3 Medical Benchmarks . . . . . . . 14
- C.4 Agent Benchmarks . . . . . . . . 15

- D Models 15
- E Baselines 15
- F Implementation Details 16

- F.1 Training Setup . . . . . . . . . . . 16
- F.2 Hyperparameter Settings . . . . . 16
- F.3 Evaluation Protocol . . . . . . . . 16

- G Extended Discussion 16

- G.1 Comparison with EAFTsig . . . . 16
- G.2 Comparison with RL and RFT . . 17

- H LLM Usage 17

### A Qualitative Analysis: What are "Confident Conflicts"?

To better understand the motivation behind our Entropy-Adaptive Fine-Tuning loss, we visualize the vocabulary distribution across different entropy and probability regimes in Figure 7. We categorize tokens into three distinct groups based on the model’s predictive uncertainty:

- • (a) Branching Points (High Entropy): As shown in the top panel, high-entropy tokens are predominantly abstract verbs (e.g., vary, depends, reconstruct), reasoning connectors, and general nouns. These tokens typically represent semantic branching points where the model faces multiple plausible continuation paths. They often correspond to complex reasoning steps or logical transitions, which are critical for the model to learn during SFT.
- • (b) Confident (Low Entropy): The middle panel displays tokens where the model exhibits

[Figure 7]

Figure 7: Word clouds visualizing tokens categorized by entropy and probability. (a) High entropy tokens often correspond to reasoning steps. (b) Low entropy tokens correspond to syntax and fixed patterns. (c) Confident Conflicts tokens often involve specific entities or data noise. In these visualizations, word size represents frequency, while color represents entropy.

low entropy and high probability. This category is dominated by mathematical symbols (e.g., Convert, 4y, 6d), LATEX syntax (e.g., \frac, \sin), and functional words. These tokens represent deterministic syntactic structures or rote memorization, which are generally easier for the model to master.

• (c) Confident Conflicts (Low Entropy and Low probability): The bottom panel illustrates tokens where the model is confident but potentially incorrect. This group consists largely of specific entities (e.g., Jayden, modpacks), rare domain terms, or noise. These often indicate a mismatch between the model distribution and the SFT data, or specific long-tail knowledge.

Insight: This qualitative distinction justifies our entropy-based adaptive strategy. The visualiza-

tion highlights “Confident Conflicts” (Low Entropy, Low Probability) where the model’s prior strongly mismatches the SFT data—a primary source of catastrophic forgetting. By differentiating these tokens from critical reasoning steps (High Entropy), our method mitigates the risk of destroying the model’s original distribution while effectively adapting to new reasoning patterns.

### B Open-source Datasets

This section details the open-source datasets utilized for training and fine-tuning models, categorized into mathematical reasoning, medical knowledge, and agent capabilities.

#### B.1 Mathematical Datasets

The datasets listed below serve as prompt sources for our data distillation process. We utilized the Qwen3-235B-A22B-Instruct-2507 (Yang et al., 2025) model to generate responses to these prompts. For the final training set, we randomly sampled 19k instances where the model’s outputs were verified as correct.

NuminaMath (Li et al., 2024): is a massive dataset dedicated to enhancing the mathematical reasoning capabilities of large language models. Representing the largest public collection in the field, it consists of approximately 860,000 pairs of competition-level math problems and detailed solutions. The dataset covers a broad spectrum of mathematical disciplines and difficulty levels.

BigMathVerified (Albalak et al., 2025): is a largescale, high-quality dataset designed specifically for applying reinforcement learning to LMs in the domain of mathematics. It aggregates a vast collection of mathematical problems and solutions from diverse sources, prioritizing data quality and reasoning rigor. By providing a rich environment for training and fine-tuning, Big-Math aims to advance the ability of models to perform reliable multistep mathematical reasoning and self-correction through reinforcement learning techniques.

Nemotron-CrossThink (Akter et al., 2025): is a large-scale dataset originally designed to extend self-learning mechanisms across various domains. While the full dataset covers a broad spectrum of reasoning tasks, we exclusively select the mathematics subset to align with our specific research focus. This subset contains high-quality synthetic problems with rigorous logical verification. By isolating these instances, we leverage the dataset’s

inherent self-correction patterns to specifically enhance the model’s precision in math derivation.

#### B.2 Medical Datasets

Huatuo-o1 (Chen et al., 2024): is a specialized supervised fine-tuning (SFT) dataset generated through knowledge distillation from the DeepSeekR1 (Guo et al., 2025) model. Built upon the set of verifiable medical problems from HuatuoGPTo1 (Chen et al., 2024), this dataset leverages DeepSeek-R1 to generate comprehensive, native reasoning chains. The primary objective is to transfer the advanced reasoning capabilities of the R1 model to target models, enabling them to initialize and internalize deep clinical thinking patterns and logical deductions essential for medical tasks.

#### B.3 Agent Datasets

Nemotron-Agentic-Tool-Use-v1 (NVIDIA, 2025) is a synthetic dataset designed to strengthen models’ capabilities as interactive, tool-using agents. It focuses on multi-turn conversations where models decompose user goals and execute tool calls to complete tasks. For our experiments, we randomly sampled 20,000 trajectories from the non-thinking subset of this dataset to serve as the training set.

### C Benchmarks

In this section, we provide a detailed introduction to the benchmarks used in experiments.

#### C.1 General Benchmarks

- • MMLU (Hendrycks et al., 2020): is a benchmark in evaluating the massive multitask language understanding and general knowledge capabilities of models. It consists of 57 distinct tasks. All of them are derived from various domains including STEM, the humanities, and the social sciences, ranging from elementary to professional levels. The problems in the MMLU dataset cover a wide variety of subjects such as elementary mathematics, US history, computer science, law, and medicine.
- • IFEval (Zhou et al., 2023): is a benchmark designed to assess the instruction-following capabilities of large language models through objective and verifiable means. It consists of approximately 500 prompts containing various verifiable constraints, such as formatting requirements, word count limits, and keyword restrictions. Unlike benchmarks that rely on subjective

human or model-based evaluation, IFEval employs a rule-based verification method to determine accuracy. This objective approach allows for a deterministic and reproducible evaluation of a model’s ability to strictly adhere to complex user instructions.

• CLUEWSC (Xu et al., 2020): is a benchmark task designed to evaluate coreference resolution and common sense reasoning within the Chinese Language Understanding Evaluation (CLUE) framework. Adapted from the classic Winograd Schema Challenge, the dataset consists of difficult ambiguity resolution problems where a model must identify the correct antecedent of a pronoun in a sentence. The questions are constructed such that changing a single word in the sentence alters the correct reference. Therefore, solving CLUEWSC requires the model to utilize deep contextual understanding and general world knowledge, rather than relying on simple statistical associations or surface-level patterns.

#### C.2 Mathematical Benchmarks

- • AIME24 (AI-MO, 2025): is a dataset in evaluating the mathematical reasoning ability of models. It consists of 30 challenging math problems. All of them are from the American Invitational Mathematics Examination. The problems in the AIME24 dataset cover a wide variety of mathematical fields such as algebraic equations and geometric puzzles. Due to the difficulty characteristics and the richness of question types, it has become a popular benchmark for evaluating the reasoning performance of models, and is widely used in multiple related research experiments.
- • AIME25 (Zhang and Math-AI, 2025): consists of 30 challenging math problems. It is directly composed of the real questions from the American Invitational Mathematics Examination (AIME I & II) newly released in February

2025. AIME25’s knowledge areas are extremely wide. It deeply covers core mathematical sections such as algebra, geometry, number theory, and combinatorial mathematics. This characteristic enables the AIME25 dataset to effectively distinguish the mathematical reasoning abilities of different models.

- • GSM8K (Cobbe et al., 2021): is an elementary school math problem dataset released by OpenAI. These problems require 2 to 8 steps to solve,

mainly through a series of basic calculations to obtain the final answer. This dataset is primarily used to test the logical and mathematical abilities of models and has been applied in multiple benchmark tests.

#### C.3 Medical Benchmarks

- • MedMCQA (Pal et al., 2022): is a large-scale multi-subject dataset designed to assess the medical knowledge and reasoning capabilities of models. It comprises over 194,000 multiple-choice questions collected from prestigious Indian medical entrance examinations, such as AIIMS and NEET-PG. The dataset covers a vast spectrum of 21 medical subjects, ranging from basic biomedical sciences to advanced clinical disciplines like surgery and internal medicine. Given its high difficulty and broad professional coverage, MedMCQA serves as a crucial benchmark for evaluating how well large language models can handle complex healthcare scenarios.
- • PubMedQA (Jin et al., 2019): is a biomedical question answering dataset designed to evaluate reasoning over scientific literature. It is derived from the titles and abstracts of research papers found in the PubMed database. The task requires models to answer research questions with "yes", "no", or "maybe" using the corresponding abstract as context. By focusing on evidence-based reasoning, PubMedQA effectively assesses a model’s ability to interpret complex biomedical texts and draw accurate conclusions directly from scientific data.
- • MedQA (Jin et al., 2021): is a large-scale dataset specifically designed to evaluate the clinical reasoning and professional medical knowledge of models. The dataset is derived from professional medical board examinations, with its English subset collected from the United States Medical Licensing Examination (USMLE). It consists of complex multiple-choice questions that simulate real-world clinical scenarios, covering diverse topics such as pathology, pharmacology, and patient management. Solving these problems requires deep domain knowledge and the ability to interpret patient case histories. Consequently, MedQA serves as a rigorous benchmark for determining whether models have achieved human-level competency in medicine.

#### C.4 Agent Benchmarks

• BFCL v3 (Patil et al.): is a comprehensive benchmark designed to evaluate the functioncalling and tool-use capabilities of large language models. It comprises a diverse set of roughly 2K entries across multiple programming languages, including Python, Java, JavaScript, and REST APIs. The dataset assesses models on complex scenarios ranging from simple function calls to parallel, multiple, and nested calls, as well as multi-turn interactions. By employing an Abstract Syntax Tree (AST) evaluation method, BFCL v3 accurately measures a model’s ability to generate syntactically correct and executable API calls, providing a more robust assessment than traditional string-matching metrics.

### D Models

Qwen3-4B-Instruct-2507 (Yang et al., 2025): is a lightweight large language model with 4 billion parameters released by Alibaba Cloud in July 2025. As the instruction-tuned variant of the Qwen3 series, it is engineered for efficient dialogue and general task execution, supporting a native context window of 256K tokens. Unlike its "Thinking" counterpart, this model prioritizes direct response generation without explicit chain-of-thought reasoning, optimizing it for low-latency performance and deployment on consumer-grade hardware.

Qwen3-4B-Thinking-2507 (Yang et al., 2025): is a reasoning lightweight model within the Qwen3 family, featuring 4 billion parameters. Unlike the standard instruction-tuned variant, this model is engineered to perform explicit chain-of-thought (CoT) reasoning, generating intermediate logical steps before producing a final answer. It supports a native context window of 256K tokens and is optimized for complex analytical tasks such as mathematical problem-solving and logical deduction.

Qwen2.5-32B-Instruct (Qwen et al., 2024): is a 32-billion parameter instruction-tuned large language model released by Alibaba Cloud in September 2024. Positioned as a mid-sized model within the Qwen2.5 series, it is designed to offer an optimal balance between computational efficiency and task performance. The model features a 64-layer Transformer architecture with Grouped Query Attention (GQA) and supports a context window of up to 128K tokens. It demonstrates significant improvements over its predecessors in instruction following, structured data understanding, and logical

reasoning, making it highly suitable for deployment in resource-constrained environments that require high-quality generation.

GLM-9B-0414 (GLM et al., 2024): is a lightweight 9-billion parameter language model released by Zhipu AI (THUDM) in April 2025. As an iteration of the GLM-4 open-source series, this model version (0414) is specifically optimized for on-device deployment and agentic tasks. It features a native context window of 128K tokens and demonstrates state-of-the-art performance in tool use (function calling) and long-context understanding among models of similar size. The model is designed to provide a balance between inference latency and task complexity, making it a competitive alternative to larger 32B models for specific downstream applications.

Qwen3-235B-A22B-Instruct-2507 (Yang et al., 2025): is a flagship Mixture-of-Experts (MoE) model released by Alibaba Cloud. While it boasts a massive total parameter count of 235 billion, it utilizes a sparse architecture that activates only 22 billion parameters during inference. This design allows the model to achieve performance comparable to dense state-of-the-art models while maintaining the inference speed and computational efficiency of a much smaller 22B model.

### E Baselines

- • SFT (Supervised Fine-Tuning): The standard fine-tuning approach that maximizes the likelihood of target tokens using a uniform crossentropy loss. It treats all tokens equally, making it prone to overfitting specific data patterns and "catastrophic forgetting" of general capabilities.
- • SFTKL (SFT with KL Regularization): A prevalent robust baseline that adds a KullbackLeibler (KL) divergence penalty to the loss function. It explicitly constrains the policy from deviating too far from the base model (reference model). While effective at preventing forgetting, it incurs significant memory overhead due to the need to maintain a frozen reference model.
- • FLOW (Sanyal et al., 2025): A dynamic reweighting method that adjusts the importance of training samples based on their learning dynamics. FLOW monitors the loss trends to identify samples that are likely to cause forgetting, down-weighting them to maintain a smoother optimization trajectory compared to SFT.

- • DFT (Wu et al., 2025): Dynamic Fine-Tuning. This method reinterprets SFT through a Reinforcement Learning lens, identifying that standard cross-entropy implicitly applies an unstable

"inverse-probability weighting" (1/πθ) to gradients. DFT rectifies this by actively scaling the loss with the model’s current prediction probability πθ(y|x). This effectively dampens the gradients for low-probability target tokens (where the model is "surprised" or wrong), stabilizing optimization by preventing aggressive fitting of hard or noisy samples.

- • TALR (Lin et al., 2025): Token-Adaptive Loss Reweighting. A granular approach that assigns varying weights to individual tokens based on their training difficulty (often measured by loss magnitude or gradient norms). TALR aims to focus the model’s capacity on "hard" tokens while reducing the impact of easy or noisy tokens, though it typically lacks the uncertaintyawareness of our entropy-based method.

### F Implementation Details

In this section, we provide a comprehensive overview of our training infrastructure, hyperparameter configurations, and evaluation protocols to ensure the reproducibility of our results.

- F.1 Training Setup All methods were implemented using LLaMA-

Factory (Zheng et al., 2024), except for SFTKL, which was trained using the Verl (Sheng et al., 2024) framework. All experiments were conducted on 8 NVIDIA A100 GPUs.

- F.2 Hyperparameter Settings

To ensure a fair comparison, we adopted a unified set of hyperparameters across all baseline models and our approach, unless specified otherwise. The specific values for the shared hyperparameters are detailed in Tab. 4. Regarding checkpoint selection, we prioritized the checkpoint that demonstrated the best mean performance averaged over all the evaluated benchmarks.

Method-Specific Hyperparameters. The KLdivergence coefficient β in SFTKL was set to 0.5.

- F.3 Evaluation Protocol

For all evaluations, we conducted three independent runs and reported the average performance.

#### Hyperparameter Value

Learning Rate 1 × 10−5 LR Scheduler Cosine Warm-up Steps 0.03 ratio Optimizer AdamW Batch Size 64 Num of Epochs 10 Max Sequence Length 16384

Table 4: Hyperparameter settings used for all models.

For mathematical reasoning benchmarks, we strictly followed the evaluation implementation of Qwen2.5-Math (Yang et al., 2024). For medical domain tasks, we aligned our evaluation pipeline with the MedEvalKit framework (Xu et al., 2025).

### G Extended Discussion

In this section, we elaborate on the design philosophy of the gating mechanism and position EAFT within the broader landscape of alignment techniques (e.g., RL (Shao et al., 2024) and RFT (Touvron et al., 2023)).

#### G.1 Comparison with EAFTsig

While Section 5.1 demonstrates that Sigmoid-based gating (EAFTsig) achieves competitive performance, we advocate for the linear formulation primarily due to its hyperparameter robustness and ease of deployment.

Sensitivity of Non-Linear Gating. The Sigmoid function introduces two sensitive hyperparameters: steepness (α) and the centering threshold (β).

- • High Sensitivity: Performance fluctuates significantly with β. If β is too high, the method degenerates into standard SFT; if too low, it mimics a Hard Mask, discarding valuable training signals. This necessitates expensive grid searches for every new dataset or model scale.
- • Binary Bias: High steepness (α) forces a nearbinary decision boundary, ignoring uncertainty and treating moderately confident tokens the same as extremely confident ones.

The "Parameter-Free" Advantage. The linear formulation acts as a structural prior: loss weight is directly proportional to uncertainty. It effectively removes the need for hyperparameter tuning. This

"out-of-the-box" robustness ensures EAFT generalizes across domains without requiring the delicate calibration needed for Sigmoid-based variants.

#### G.2 Comparison with RL and RFT

Existing alignment methods like On-policy RL (Schulman et al., 2017; Shao et al., 2024) or Rejection Sampling Fine-Tuning (RFT) (Touvron et al., 2023) mitigate the "alignment tax" but incur high computational or operational costs (Chen et al., 2025; Shenfeld et al., 2025). EAFT provides a distinct trade-off favoring efficiency.

Lightweight Efficiency (vs. RL). RL methods typically triple memory requirements by maintaining multiple models simultaneously (Policy, Reference, and Value/Reward models). EAFT retains the nearly identical memory footprint and computational graph of standard SFT. Crucially, it requires no Reference Model (using the model’s own entropy as a proxy for "trustworthiness") and avoids the complex rollout generation phase, significantly accelerating training throughput.

Dynamic Adaptation (vs. RFT). Methods like RFT rely on static data curation—filtering or generating data based on the model’s capability before training begins.

- • Static vs. Dynamic: RFT assumes a fixed knowledge boundary. However, as the model learns, its uncertainty shifts.
- • On-the-fly Correction: EAFT operates dynamically. If the model becomes overconfident about a hallucination mid-training, EAFT automatically down-weights the loss for that specific token. This allows the gating mechanism to evolve in real-time alongside the model parameters, offering an adaptive advantage that static data filtering cannot achieve.

### H LLM Usage

We used large language models (LLMs) to improve the clarity and grammatical correctness of the manuscript. After using these tools, the authors carefully reviewed and edited all generated content, and take full responsibility for the final version of the paper.

