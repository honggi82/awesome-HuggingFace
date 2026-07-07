## The Invisible Leash? Why RLVR May or May Not Escape Its Origin

# arXiv:2507.14843v4[cs.LG]4Feb2026

Fang Wu*1 Weihao Xuan*23 Ximing Lu45 Mingjie Liu5 Yi Dong5 Zaid Harchaoui4 Yejin Choi15

Math Reasoning Tasks 𝑝𝑎𝑠𝑠@8192

[Figure 1]

### Abstract

AIME2024 AIME2025 AMC Minerva Olympiad

Non-math Reasoning Tasks 𝑝𝑎𝑠𝑠@2048

Recent advances highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising method for enhancing LLMs’ capabilities. However, it remains unclear whether the current practice of RLVR truly expands a model’s reasoning boundary or mainly amplifies high-reward outputs that the base model already knows, thereby improving precision. This study presents an empirical investigation that provides fresh insights into the limits of RLVR. We examine how RLVR can operate as a support-constrained optimization mechanism that may restrict the discovery of entirely original solutions, remaining constrained by the base model’s initial distribution. We also identify an entropy-reward trade-off: while RLVR reliably enhances precision, it may progressively narrow exploration and potentially overlook correct yet underrepresented solutions. Extensive empirical experiments validate that while RLVR consistently improves pass@1, the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets, failing to recover correct answers that were previously accessible to the base model. Interestingly, while RLVR sometimes increases tokenlevel entropy, it results in greater uncertainty at each generation step and declining answer-level entropy. This indicates that these seemingly more uncertain paths ultimately converge onto a smaller set of distinct answers. Taken together, we reveal potential limits of RLVR in extending reasoning horizons. Breaking this invisible leash requires future innovations that seed probability mass into underrepresented solution regions.

SimpleQA LiveBench-R LiveBench-C LiveBench-L SciBench

Figure 1. Conceptual illustration of empirical support. We define four regions based on whether a correct completion y∗ ∈ C is assigned non-negligible probability mass by the base and RLVR models, q and πθ Support Preservation covers completions with q(y∗|x) > ϵ and πθ(y∗|x) > ϵ; Support Shrinkage includes correct completions downweighted by RLVR below ϵ; Support Expansion includes completions that RLVR newly upweights above ϵ despite negligible base model mass; and Out of Support refers to completions missed by both.

### 1. Introduction

The rise of large reasoning models, such as DeepSeekR1 (Guo et al., 2025) and OpenAI-o1 (Jaech et al., 2024), marks a breakthrough in AI capabilities, particularly in solving mathematics (Luo et al., 2025c; Zeng et al., 2025) and programming (Luo et al., 2025b; Liu & Zhang, 2025). The key ingredient behind this remarkable progress is large-scale Reinforcement Learning with Verifiable Rewards (RLVR), where a pretrained base model is optimized via RL using simple, automatically computed rewards. Prior work has also explored stronger notions of verifiability, such as models that can generate interactive proofs of their own correctness (Amit et al., 2025). Despite empirical success, a fundamental question remains under active debate within the research community: Does the current practice of RLVR expand base models’ reasoning capabilities, or simply reinforce patterns base models already know, sometimes at the expense of exploring alternative correct solutions?

1Stanford University, CA, USA 2The University of Tokyo, Japan 3RIKEN AIP, Tokyo, Japan 4University of Washington, Seattle, WA, USA 5NVIDIA, CA, USA. Correspondence to: Yejin Choi <yejinc@stanford.edu>.

Preprint. February 5, 2026.

Recent studies have revealed a puzzling pattern that hints at this limitation. While models trained with RLVR consistently outperform base models on a single attempt, base models often perform better with multiple attempts. Shao et al. (2025) also suggests that RLVR can even benefit from noisy or spurious reward signals, further calling into question whether observed gains reflect deeper reasoning improvements or merely sharper exploitation of existing heuristics.

While pass@k may have limitations as a comprehensive measure of reasoning boundaries, as it primarily captures solution retrieval rather than novel reasoning capacity (Wen et al., 2025), we adopt it here as a proxy metric following Chen et al. (2021); Shao et al. (2024); Chen et al. (2025b). pass@k provides a useful lens for examining how RLVR affects solution accessibility, though future work should explore more nuanced measures of reasoning capability expansion. Besides, prior work examines RLVR only through before/after snapshots, leaving unexplored how RLVR reshapes the model’s effective reasoning support throughout training. Some studies interpret pass@k as evidence that RLVR primarily performs conservative optimization within the base model’s existing capabilities (Yue et al., 2025a; Zhao et al., 2025; Shah et al., 2025; Ma et al., 2025; He et al., 2025). Others argue that this pattern only appears in specialized domains where base models were already well-trained, and that RLVR can substantially expand reasoning in other domains (Liu et al., 2025a).

Seeking a definitive answer to this debate remains an open challenge. In the extreme case, it seems unlikely that RLVR can unlock advanced reasoning capabilities for any model out of the box, such as GPT-2 (Radford et al., 2019). We are curious if there may exist inherent limitations in the current RLVR practice. This paper provides a systematic empirical investigation into the fundamental capabilities and potential limitations of RLVR. We introduce the concept of empirical support: the set of correct solutions that a model can realistically discover under finite sampling. Using this framework, we show that:

- 1. The current RLVR recipe primarily preserves rather than expands the base model’s solution coverage. Across diverse reasoning benchmarks, RLVR consistently loses access to more correct solutions than it gains, even while improving single-sample accuracy. We also analyze the temporal evolution of support dynamics during RLVR training, revealing how RLVR progressively narrows the accessible solution space.
- 2. The precision-diversity trade-off is fundamental, not domain-specific. This pattern appears across mathematics, logical reasoning, factual QA, and code generation, suggesting it reflects inherent properties of current RLVR rather than domain-specific quirks.

3. Local uncertainty and global diversity can diverge. RLVR sometimes increases token-level entropy (appearing more “uncertain” during generation) while simultaneously reducing answer-level entropy (converging to fewer final solutions).

These findings show that RLVR may face an “invisible leash”. They remain fundamentally constrained by their initialization and cannot discover reasoning patterns beyond the base model’s effective reach. To break this invisible leash, RLVR may need to be augmented with explicit exploration or hybrid strategies that seed probability mass into underrepresented regions of the solution space. We hope this work offers novel insights into the strengths and limitations of the current RLVR recipe, guiding future efforts to improve RLVR and build LLM systems that unlock genuinely new reasoning capacity.

### 2. Related Works

RLVR has emerged as a scalable alternative to RLHF by replacing human preference labels with automated, verifiable signals. Unlike RLHF, which relies on learned reward models for subjective alignment, RLVR optimizes objective, algorithmically computable rewards that directly reflect ground-truth success criteria in domains with clear verifiability. It has achieved strong gains on real-world applications, including math (Guo et al., 2025) and medical multiplechoice QA (Zhang et al., 2025; Lai et al., 2025; Feng et al., 2025), often outperforming SFT with limited training data.

Recent analyses have examined both the benefits and limitations of RLVR. Studies (Wen et al., 2025) argue that standard evaluation metrics, such as Pass@K, can overstate RLVR gains by ignoring the correctness of intermediate reasoning, motivating alternatives (e.g., CoT-Pass@K) that jointly assess reasoning chains and final answers. Under such metrics, RLVR exhibits stronger incentives toward correct reasoning. Complementarily, a recent work by Amit et al. (2025) reports a very interesting theoretical perspective, in which verifier-based training can provide per-instance correctness guarantees by conditioning acceptance on externally verifiable proofs, shifting evaluation from average-case accuracy to worst-case soundness. Concurrent position highlights hidden costs in measurement protocols, data contamination, and evaluation bias, advocating tax-aware evaluation that jointly optimizes accuracy, grounding, and calibrated abstention (Tu et al., 2025).

At the algorithmic level, recent RLVR variants address challenges in reward sparsity, verifier reliability, and exploration. Cai et al. (2025) explicitly models asymmetric noise in automated verifiers to improve robustness under imperfect rewards. Self-verification approaches (Liu et al., 2025b) integrate generation and critique within a single RL loop

to produce denser and more informative feedback. Other work explores efficiency-oriented objectives and supervised surrogates to stabilize optimization under sparse verifiable signals (Sun et al., 2025). Finally, emerging theoretical perspectives investigate semantic hidden-state dynamics beyond token-level entropy, proposing structural incentives such as VERL to better decouple exploration and exploitation in reasoning models (Huang et al., 2025).

### 3. Numerical Metrics

#### 3.1. Formalizing Solution Accessibility

Effective Support of Correct Completions. Let X denote the space of natural language prompts, and Y denote the space of token sequences (e.g., reasoning traces or completions). For a fixed prompt x ∈ X, q(y | x) is the output distribution of the base model, and R(x,y) ∈ {0,1} is a verifiable reward function indicating whether y is a correct solution. Various RLVR algorithms, including PPO (Schulman et al., 2017), RLOO (Kool et al., 2019), GRPO (Guo et al., 2025), DAPO (Yu et al., 2025), or REINFORCE++ (Hu, 2025), learn a new distribution πθ(y | x) to optimize different variants of the following regularized objective: maxθ Ey∼π

θ(y|x)

θ(·|x),x∼D R(x,y) − β−1 log π

q(y|x) , where D is the distribution of prompts. An optional log ratio corresponds to a regularized policy update that penalizes divergence from q controlled by a hyperparameter β > 0.

Definition 3.1 (Support of Correct Completions). Let C = {y ∈ Y | R(x,y) = 1} denote the set of correct completions under the reward function R. Then the effective support on correct completions of a distribution p(y | x) is defined as supp(p) := {y ∈ C | p(y | x) > 0}.

We define empirical support over answer-level correct completions, where multiple reasoning traces that yield the same verified outcome are treated as a single equivalence class.

Empirical Support Relaxation. The effective support assumes q = 0, which rarely holds. Softmax layers yield strictly positive probabilities across all tokens, making the nominal support of q span the entire space Y. This factor, along with sampling noise or temperature scaling, contributes to what we refer to as empirical support diffusion: over time, the model may assign growing probability mass to completions that initially had negligible but still nonzero probability under the base model.

While q(y | x) is technically positive for all y due to the softmax, many completions lie so deep in the tail that they are effectively invisible to the training algorithm under finite sampling. To formalize this, we develop a relaxation and define the empirical support under ϵ as

suppϵ(q) := {y ∈ C | q(y | x) > ϵ},

where ϵ > 0, with ϵ → 0, denotes a minimal cutoff that separates completions with practically observable likelihood from those that are statistically negligible. Completions outside this threshold are unlikely to be sampled in typical on-policy RL settings with finite rollouts. The choice of ϵ is thus crucial for assessing which completions are empirically reachable. Intuitively, ϵ should correspond to the minimum probability required for a correct completion to appear within finite samples. We derive a principled estimate for this threshold based on sampling confidence bounds in Appx. C.4.

#### 3.2. Characterizing Solution Access Change

With empirical support defined, we categorize what happens to correct solutions under RLVR:

- Definition 3.2 (Empirical Support Dynamics). For a given threshold ϵ > 0,

- • We say RLVR achieves empirical support expansion un-

der threshold ϵ if suppϵ(πθ) \ suppϵ(q) ̸= ∅, i.e. there exists at least one completion y∗ ∈ C such that

q(y∗ | x) ≤ ϵ but πθ(y∗ | x) > ϵ.

That is, the RLVR-trained model assigns non-negligible probability mass to correct completions that were effectively negligible under the base model.

- • We say RLVR exhibits empirical support shrinkage under

threshold ϵ if suppϵ(q) \ suppϵ(πθ) ̸= ∅, i.e. there exists at least one completion y∗ ∈ C such that

q(y∗ | x) > ϵ but πθ(y∗ | x) ≤ ϵ.

This formalizes the phenomenon in which RLVR concentrates probability mass on a narrower subset of outputs, effectively excluding correct solutions that were previously accessible under the base model.

Support Dynamics Metrics. To quantify RLVR’s impact on solution accessibility, we introduce the precision and recall-inspired metrics based on the four support categories.

- Definition 3.3 (Support Dynamics Metrics). Let P, E, S, and O denote the number of correct completions in preservation, expansion, shrinkage, and out-of-support, respectively.

- • Support Retention Rate (SRR) measures how well RLVR preserves the base model’s accessible correct solutions:

SRR =

P P + S

- • Net Discovery Rate (NDR) measures the fraction of RLVR’s accessible solutions that represent genuine discoveries:

E P + E

NDR =

- • Support Dynamic Score (SDS) provides a balanced measure combining retention and discovery:

SDS =

2 · SRR · NDR SRR + NDR

=

2PE P2 + 2PE + ES

- • Net Support Change Rate (NSCR) captures the net expansion or shrinkage of empirical support:

E − S P + E + S

NSCR =

These metrics provide complementary perspectives on RLVR’s behavior:

- • SRR ∈ [0,1]: Higher values indicate better preservation of base model solutions. SRR = 1 means no shrinkage occurred.
- • NDR ∈ [0,1]: Higher values indicate more genuine discovery. NDR = 0 means no new solutions were found; NDR = 1 means all accessible solutions are discoveries.
- • SDS ∈ [0,1]: Harmonic mean balancing retention and discovery. High SDS requires both good retention and meaningful expansion.
- • NSCR ∈ [−1,1]: Positive values indicate net expansion, negative values indicate net shrinkage.

These metrics enable us to distinguish between different RLVR behaviors: support-constrained optimization (high SRR, low NDR), genuine capability expansion (high SRR, high NDR), inefficient redistribution (low SRR, low NDR), and aggressive exploration (low SRR, high NDR). We also provide theoretical foundations to understand RLVR’s support-bounded behavior in Appx. C.

### 4. Evidence of Hidden-Support Dynamics

#### 4.1. Experimental Setup

We adopt ProRL-1.5B-v1 (Liu et al., 2025a) as our main RLVR method due to its robust long-horizon training framework. Starting from DeepSeek-R1-Distill-Qwen-1.5B as the base model, ProRL’s Nemotron-Research-Reasoning series leverages GRPO enhanced with decoupled clipping, dynamic sampling, KL divergence regularization, and periodic reference resets to sustain exploration and prevent entropy collapse during extended RL training. In addition, we evaluate other RLVR variants at multiple scales (7B–14B parameters), including Skywork (Wei et al., 2023), AceReason-Nemotron (Chen et al., 2025a), and Phi4-Reason (Abdin et al., 2025), alongside a visual LLM (Kangheng-OVR-7B (Wei et al., 2025)).

Table 1. Aggregate support dynamics across diverse models and domains. Each completion is categorized by correctness and support status: Preservation indicates both base and RLVR find the solution; Shrinkage indicates only the base model found it; Expansion indicates only RLVR found it; and Out of Support denotes solutions found by neither. Higher SRR, NDR, and SDS reflect stronger preservation, genuine discovery, and balanced optimization, respectively. NSCR values closer to zero indicate more balanced changes in support. Kangheng-OVR-7B is a visionlanguage model (VLM). Full detailed statistics for each model are in Appx. A.

Model Domain SRR NDR SDS NSCR P E S O

Math 0.96 0.00 0.01 -0.04 1355 5 56 131 Non-Math 0.91 0.03 0.06 -0.06 1045 31 107 674 Overall 0.94 0.02 0.03 -0.05 2400 36 163 805

PRORL-1.5B-V1

Math 0.96 0.01 0.01 -0.04 1349 9 62 127 Non-Math 0.90 0.04 0.07 -0.06 1039 39 113 666 Overall 0.93 0.02 0.04 -0.05 2388 48 175 793

PRORL-1.5B-V2

Math 0.99 0.00 0.01 -0.00 1431 5 9 102 Non-Math 0.97 0.02 0.04 -0.02 1284 23 47 503 Overall 0.98 0.01 0.02 -0.01 2715 28 56 605

NEMOTRON-1-7B

Math 0.98 0.00 0.00 -0.02 1406 2 34 105 Non-Math 0.96 0.02 0.04 -0.02 1279 24 52 502 Overall 0.97 0.01 0.02 -0.02 2685 26 86 607

SKYWORK-OR1-7B

Math 0.99 0.00 0.01 -0.01 1425 5 15 102 Non-Math 0.99 0.00 0.01 -0.01 993 3 8 399 Overall 0.99 0.00 0.01 -0.01 2418 8 23 501

NEMOTRON-1-14B

Math 0.99 0.01 0.01 -0.00 1407 8 12 120 Non-Math 0.99 0.01 0.01 -0.00 1067 8 11 317

PHI4-REASON-PLUS-14B

Overall 0.99 0.01 0.01 -0.00 2474 16 23 437 OLMO-2-0425-1B Math 0.88 0.10 0.18 -0.02 761 83 104 599

KANGHENG-OVR-7B Math 1.00 0.00 0.01 -0.00 781 3 4 516

We additionally include ProRL-1.5B-v2 as a separate checkpoint for comparison (Tab. 1).

Performance is measured across two categories. (1) Math reasoning tasks: MATH500 (Hendrycks et al., 2021), Minerva (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), AIME 2024, AIME 2025, and AMC 2023. (2) Non-math reasoning tasks: SimpleQA (Wei et al., 2024) (factuality), LiveBench (White et al., 2025) (logic, coding, and language comprehension), SciBench (Wang et al., 2023) (multi-domain scientific problem-solving), and Reasoning Gym (Stojanovski et al., 2025) (cognition, geometry, graph theory, games). In Reasoning Gym, we especially focus on tasks that ProRL explicitly highlighted as challenging for the base model. For SimpleQA, we employ GPT-4.1 (Achiam et al., 2023) as the judge. Sampling budgets are k ∈ {4096,8192} for math, k ∈ {1024,2048,4096,8192,16384} for Reasoning Gym, and k ∈ {1024,2048} for other non-math datasets, ensuring that any unreachable solution y∗ ∈ C remains below the empirical support threshold of the base model. More implementation details appear in Appx. B.

4.2. Results: Predominant Preservation with Limited Expansion

Support preservation dominates across all domains. Tab. 1 shows that across diverse model scales and families, RLVR predominantly acts as a support-constrained optimization mechanism. All 1.5B–14B models achieve very

dice pass@k

###### prime_factorization_hard pass@k

palindrome_generation pass@k

0.9801.0001.0001.0001.0001.0001.0001.0001.0001.000

1.000

0.981 1.000 1.000 1.000 1.000 0.923

0.980 0.980 0.980 0.980 1.000 1.000 1.000 1.000 1.000 1.000

1.0

1.0

0.920

- 0.832

0.941

0.970

0.980 0.980

- 1.000 1.000 1.000 1.000 1.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000

0.9

1.0

0.9

0.860

0.884

0.820

0.825

0.8

0.8

0.7

0.688

0.7

0.6

Pass@k

Pass@k

Pass@k

0.9

0.540

0.6

0.5

0.4600.480

0.460

0.5

0.4

0.360

0.320

0.413

0.3

0.4

0.8

0.200

0.2

0.180

0.3

0.235

0.100

Base

Base

Base

0.1

0.060

0.2

0.0000.0000.0000.020

0.154

ProRL

ProRL

ProRL

0.0

0.7

0.1

1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192

1 2 4 8 16 32 64 128 256 512 1024

1 2 4 8 16 32 64 128 256 512 1024

k

k

k

###### advanced_geometry pass@k

advanced_geometry_hard pass@k

###### prime_factorization pass@k

0.961 0.980 0.980 0.980 0.980 0.980 0.980 1.000 1.000

1.000

1.0

1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000

1.000

1.0

0.941

0.981 0.981

0.922

0.962

0.9

0.943 0.943

0.845

0.804 0.804 0.823 0.842 0.842

0.924

0.923

0.905

0.904

0.8

0.9

0.886

- 0.980 0.980 0.980
- 1.000 1.000 1.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000

0.865

0.728

1.0

0.686 0.705

0.7

0.828

Pass@k

Pass@k

Pass@k

0.8

0.6

0.551

0.511

0.5

0.7

0.690

0.414

0.4

0.611

0.3

0.275

0.6

Base

Base

Base

0.2

0.157

ProRL

ProRL

ProRL

0.5

0.1

0.9

1 2 4 8 16 32 64 128 256 512 1024 2048

1 2 4 8 16 32 64 128 256 512 1024 2048 4096

1 2 4 8 16 32 64 128 256 512 1024

k

k

k

###### graph_color pass@k

graph_color_hard pass@k

###### graph_color_vertex15 pass@k

- 0.140

0.241

0.421

0.742

- 1.000 1.000 1.000 1.000 0.9601.000 0.9801.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000 1.0001.000

1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000

0.980 1.000

- 0.100

0.161

0.222

0.382

0.602

0.742

0.841 0.861

0.941

- 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.0001.000 1.0001.000

1.0

1.0

1.0

0.960

0.941

0.881 0.901

0.9

0.9

0.9

0.802 0.822

0.8

0.8

0.8

0.7

0.7

0.7

Pass@k

Pass@k

Pass@k

0.6

0.6

0.6

0.582

0.5

0.5

0.5

0.462

0.4

0.4

0.4

0.3

0.3

0.3

0.221

0.2

0.2

0.2

Base

Base

Base

0.120 0.121

0.1

0.1

0.1

ProRL

ProRL

ProRL

0.0

0.0

0.0

1 2 4 8 16 32 64 128 256 512 1024

1 2 4 8 16 32 64 128 256 512 1024 2048

1 2 4 8 16 32 64 128 256 512 1024

k

k

k

Figure 2. Empirical support preservation in Reasoning Gym tasks, like Graph Coloring, Palindrome Generation, and Advanced Geometry.

###### arc_1d pass@k

###### boxnet pass@k

graph_color_vertex20 pass@k

0.8

0.6

0.980 0.980 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000

1.0

0.7

0.881 0.901

0.680

0.9

0.492

0.5

0.475

0.8

0.6

0.4370.443

0.762

0.560 0.571 0.577 0.578

0.570

0.526

0.7

0.4

0.499

0.5

0.366

0.447 0.460

0.3320.345

0.343

0.6

Pass@k

Pass@k

Pass@k

0.325

0.308

0.397

0.4

0.2830.294

0.3

0.289

0.375

0.5

0.348

0.258

0.253

0.239

0.406

0.297

0.295

0.4

0.3

0.2090.216

0.366

0.210

0.2

0.185

0.179

0.287

0.3

0.197

0.151

0.2

0.148

0.247

0.131

0.158

0.187

0.2

0.128

0.102

0.097

0.1

0.109

0.083

0.097

0.126

0.1

0.063

Base

Base

0.064

0.1

Base

0.048

0.040

0.037

0.000 0.000 0.001 0.023 0.044

0.024

0.010

ProRL

ProRL

ProRL

0.0

0.0

0.0

1 2 4 8 16 32 64 128 256 512 1024 2048

1 2 4 8 16 32 64 128 256 512 1024 2048 4096

1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384

k

k

k

Figure 3. Instances of empirical support expansion, as seen in Boxnet, Dice, and Arc 1D tasks.

high support retention (overall SRR ≈ 0.93–0.99) while genuine discovery remains rare (NDR ≤ 0.04). For example, Nemotron-7B and Nemotron-14B retain nearly all base-model solutions (SRR ≥ 0.98) with negligible expansion. Even smaller-scale ProRL-1.5B-v2 achieves SRR = 0.93 with only modest gains (NDR = 0.02). These patterns persist across math (SRR = 0.96–0.99, NDR ≈ 0.00–0.01) and non-math domains (SRR = 0.90–0.99, NDR ≤ 0.04). Preservation-dominated behavior is especially clear in Reasoning Gym tasks such as graph color and palindrome, where ProRL accelerates convergence toward near-perfect pass@k with large budgets (Fig. 2). Support counts confirm this: most correct completions remain shared between RLVR and base models.

Selective but limited empirical support expansion. Despite strong conservation, RLVR occasionally recovers solutions that are negligible relative to the base model. Expansion is consistently small: ProRL-1.5B-v2 discovers 48

new completions across 11 benchmarks, while larger models (e.g., Phi4-14B, Nemotron-14B) add fewer than 10 per domain. Non-math datasets exhibit the highest relative discovery (NDR ≤ 0.04), whereas math datasets are virtually stagnant (NDR ≤ 0.01). Some Reasoning Gym tasks, such as graph color vertex20 and arc 1d, show genuine expansion (Fig. 3), but remain isolated exceptions rather than the dominant trend. These suggest that while RLVR can occasionally redistribute mass into underexplored solution modes, such expansion remains the exception rather than the rule, challenging assumptions about RLVR’s capacity for genuine reasoning horizon extension.

Empirical support shrinkage outweighs expansion. Across all models and domains, shrinkage consistently exceeds expansion. ProRL-1.5B-v2 loses 175 completions while gaining only 48 (ratio ≈ 3.6:1), while Nemotron-7B and Skywork-OR1-7B display similar patterns (ratios ≈ 2:1–3:1). Even large models

###### leg_counting pass@k

###### family_relationships pass@k

###### power_function pass@k

0.5

1.000

1.0

1.000 1.000 1.000 1.000 1.000 1.000

0.960

1.0

0.900 0.900

0.9

0.840 0.860 0.860 0.860 0.860 0.860 0.860 0.860 0.860 0.860 0.880

0.941

0.860

0.400

0.4

0.901 0.901 0.901 0.901 0.901 0.901 0.901 0.901

0.9

0.8

0.780

0.340

0.842

0.842

0.7

0.680

0.320

0.802

0.300

0.300 0.300 0.300 0.300 0.300 0.300 0.300 0.300 0.300 0.300

0.8

0.3

0.280

0.280

Pass@k

Pass@k

Pass@k

0.6

0.560

0.723

0.241

0.5

0.7

0.2

0.4

0.380

0.624

0.160

0.604

0.140

0.6

0.3

0.120

0.220

0.1

0.2

0.5

0.140

Base

Base

Base

0.465

0.040 0.040

0.1

ProRL

ProRL

ProRL

0.0

0.0

0.4

1 2 4 8 16 32 64 128 256 512 1024

1 2 4 8 16 32 64 128 256 512 1024

1 2 4 8 16 32 64 128 256 512 1024

k

k

k

Figure 4. Examples of empirical support shrinkage on Reasoning Gym, like Leg Counting and Family Relationships, and Power Function.

(Nemotron-14B, Phi4-14B) show net shrinkage despite near-perfect preservation. Overall NSCR values remain slightly negative (−0.01 to −0.06), showing that RLVR systematically narrows the accessible solution set. This explains paradoxical outcomes: while RLVR models outperform bases at low k, base models dominate at high k due to broader solution coverage (e.g., AIME2024 base pass@8192 = 93.3% vs. ProRL-1.5B’s 83.3%). Reasoning Gym tasks like leg counting, family relationships, and power function illustrate this vividly (Fig. 4).

Support dynamic score confirms imbalance. SDS values remain consistently low in the domain-level aggregates for our 1.5B–14B models (≤ 0.07), reflecting poor balance between preservation and discovery. TFor example, LiveBench-L reaches SDS = 0.278 (2 expansions vs. 2 shrinkages). Math benchmarks are particularly imbalanced (SDS ≈ 0.00–0.01), while non-math domains fare only marginally better. Thus, RLVR’s improvements arise primarily from mass concentration rather than meaningful solution expansion.

Perplexity analysis on support constraints. Tab. 2 reports perplexity where base and ProRL models are evaluated against external reasoning traces from DeepSeek-R1 and Claude Sonnet 4. ProRL consistently shows higher perplexity. For instance, on AIME2024, perplexity on Claude Sonnet 4 increases from 8.76 (Base) to 14.91 (ProRL), indicating that RLVR reduces the model’s ability to assign probabilities to diverse external reasoning styles. While format differences contribute, the dominant effect is structural: RLVR concentrates probability around narrower solution trajectories. Correctness patterns further reveal the precision-coverage trade-off. In shrinkage cases, ProRL shows higher perplexity even when the base model succeeds, confirming RLVR collapses mass away from viable solution pathways. Conversely, modest perplexity improvements in rare expansion cases suggest ProRL’s new successful trajectories originate from the base model’s low-density tails rather than genuinely novel reasoning structures.

Comparison between SFT and RLVR. We fix the base model (Qwen2.5-Math-7B), dataset (DeepMath-103K), sampling protocol, and optimization hyperparameters, varying only the training objective (SFT vs. DAPO). As shown

Table 2. Perplexity of reasoning tokens from base and RLVR across math benchmarks, segmented by correctness patterns and reference types. For different problem categories (e.g., shrinkage and expansion), perplexity is computed against external references (DeepSeek-R1 and Claude Sonnet 4), reflecting each model’s compatibility with diverse and broad solution modes.

Category Correctness Reference Target AIME24 AIME25 Olympiad

Base 1.24 1.39 1.17

DeepSeek-R1

- ProRL 1.39 1.70 1.25

Claude Sonnet 4

Base 1.70 1.54 1.51

- ProRL 2.12 1.98 1.83

Shrinkage ✓ Base, ✗ ProRL

Base - - 1.41

DeepSeek-R1

ProRL - - 1.28 Claude Sonnet 4

Expansion ✗ Base, ✓ ProRL

Base - - 1.65 ProRL - - 1.38

Base 1.82 1.75 1.62

DeepSeek-R1

ProRL 2.20 2.15 1.94 Claude Sonnet 4

– – ✗ Base, ✗ ProRL

Base 8.76 6.05 5.98 ProRL 14.91 9.76 9.55

in Tab. 3 and 14, SFT consistently produces moderate support expansion with positive NSCR values across benchmarks, whereas DAPO produces sharply concentrated distributions with mixed or negative support change. While DAPO improves precision at low sampling budgets, its support dynamics reflect a high-SRR but low-NDR regime: it preserves known correct solutions but discovers few new ones. On several benchmarks, such as MATH500, Minerva, and Olympiad, DAPO reduces the number of accessible correct solutions, resulting in net shrinkage despite identical training conditions. These results confirm that support-constrained behavior is not an artifact of scaling, data mixture, or procedure, but emerges from the objective itself. By contrast, SFT expands the reachable solution set without collapsing existing modes, underscoring that shrinkage is intrinsic to RLVR-style multiplicative updates rather than SFT.

Support dynamics during RLVR training. Tab. 4 summarizes how support-dynamics metrics evolve during RLVR training. Across benchmarks, SRR remains consistently high, showing that RLVR reliably preserves previously correct reasoning paths, while NSCR gradually decreases, indicating a steady contraction of the model’s solution support. On harder datasets, mild fluctuations in NDR and SDS reflect transient exploration that is subsequently pruned. Overall, these statistics reveal that RLVR reshapes rather than monotonically improves the reasoning distribution. It rein-

- Table 3. Comparison of support dynamics between SFT and DAPO training on Qwen2.5-Math-7B. SFT moderately expands support, while DAPO sharpens support but often reduces stability and increases shrinkage (lower NSCR).

pass@k Support Dynamics Metrics Support Counts Base Target SRR NDR SDS NSCR P E S O Math Reasoning Benchmarks (pass@256)

Dataset Training

- AIME24

SFT 63.33% 63.33% 0.789 0.211 0.332 0.000 15 4 4 7 DAPO 63.33% 66.67% 0.895 0.150 0.257 0.045 17 3 2 8

- AIME25

SFT 50.00% 63.33% 0.933 0.263 0.411 0.200 14 5 1 10

DAPO 50.00% 56.67% 1.000 0.118 0.211 0.118 15 2 0 13 AMC23

SFT 100.00% 100.00% 1.000 0.000 0.000 0.000 40 0 0 0

DAPO 100.00% 92.50% 0.925 0.000 0.000 -0.075 37 0 3 0 MATH500

SFT 96.40% 99.00% 0.996 0.030 0.059 0.026 480 15 2 3 DAPO 96.40% 95.20% 0.975 0.013 0.025 -0.012 470 6 12 12 Minerva

SFT 63.24% 66.91% 0.895 0.154 0.263 0.050 154 28 18 72

DAPO 63.24% 55.15% 0.802 0.080 0.145 -0.120 138 12 34 88 Olympiad

SFT 78.22% 81.78% 0.953 0.089 0.162 0.042 503 49 25 98 DAPO 78.22% 72.89% 0.884 0.051 0.096 -0.065 467 25 61 122

forces a narrow set of stable trajectories over time, which explains both the early-stage gains and the late-stage degradation observed on more diverse math tasks.

Overall takeaway: RLVR as precision enhancer, not capability expander. Across model scales (1.5B–14B) and domains (math, non-math, multimodal), RLVR consistently behaves as a support-bounded optimizer. With SRR near one but NDR near zero, and uniformly negative NSCR, RLVR enhances precision by concentrating mass on known high-reward solutions but rarely discovers new reasoning paths. This aligns with the Temporal Forgetting effect (Li et al., 2025). Breaking the invisible leash may thus require explicit exploration strategies that deliberately seed probability mass into underrepresented solution regions.

#### 4.3. When Empirical Support Expansion Occurs

Although rare, empirical support expansion follows clear patterns that align with the base model’s latent capabilities. Across Reasoning Gym tasks, we identify two primary mechanisms that explain why expansion arises in a small set of cases such as dice, arc 1d, boxnet, and graph color vertex20.

(1) RLVR Recomposes Subskills the Base Model Already Possesses. These expansion tasks exhibit modular or compositional structure, such as local moves in graph coloring, element-wise updates in arc 1d, or JSON-style keyvalue fragments in boxnet. The base model assigns nonnegligible mass to many of these fragments individually, but not to their correct global combination. RLVR magnifies these weakly represented components and helps the model assemble them coherently. This matches our empirical patterns: all observed expansions occur in tasks where the base model’s pass@1 is low, but its pass@k curve rises steadily (Fig. 3), indicating that all fragments needed for correct reasoning already lie in the base model’s long tail. RLVR amplifies these long-tail valid completions, elevating

a few above the empirical support threshold ϵ.

(2) RLVR Corrects Prompt-Format Misalignment and Extracts Latent Ability. A second driver is the sensitivity to the prompt format. In several expanding tasks, especially dice and boxnet, the base model demonstrates partial competence but fails to follow the response format required by the reward function. RLVR reshapes the distribution to follow instructions more effectively, unlocking capabilities that were previously present but suppressed. Consistent with this interpretation, the perplexity gaps in Tab. 2 remain modest in expansion cases, showing that the “new” completions are stylistic or formatting variants of reasoning patterns already accessible to the base model, not fundamentally novel solutions beyond its support.

However, expansion remains sharply limited. Across the 1.5B–14B models in Tab. 1, NDR never exceeds 0.04, and NSCR remains non-positive. Thus, RLVR’s gains arise from amplifying low-probability but existing solution fragments or format-correct variants, but never from discovering solutions truly absent from the base distribution.

### 5. Entropy Reduction and pass@k Trade-off

#### 5.1. Experimental Setup

To study how RLVR reshapes the sampling distribution, we examine the base model and RLVR with a medium sampling budget k = 32 on the math reasoning benchmarks. We quantify two entropy metrics:

- • Token-Level Entropy: Let V denote the vocabulary and

y(i) = (y1(i),y2(i),...,yT(i)

(i)

) denote the i-th generated sequence of length T(i) for 1 ≤ i ≤ N. At each timestep t, the model outputs a probability distribution p(ti)(v) over vocabulary tokens v ∈ V. The entropy of this distribution is given by: Ht(i) = − v∈V p(ti)(v)log p(ti)(v). The average token-level entropy over all N sequences and their timesteps is computed as: TokenEntropy =

1 N

N i=1

1 T(i)

T(i) t=1 Ht(i) , capturing the local uncer-

tainty at each generation step.

- • Answer-level Entropy: Let {o(1),...,o(N)} denote the answers extracted from each generated sequence y(i) (us-

ing NA for incomplete outputs), and let {o∗1,...,o∗M} be the M unique answers. Let fj be the frequency of

answer o∗j, with empirical probability pj = fNj . Then: AnswerEntropy = − Mj=1 pj log pj. This captures global diversity over output completions, with lower values indicating increased mode collapse.

- Table 4. Effect of RLVR (DAPO) training on DeepSeek 1.5B across math benchmarks. Pass@256 values are percentages. Step = 0 corresponds to the non-RLVR baseline.

Dataset Step

Pass@256 Support Dynamics Metrics Support Counts

Dataset Step

Pass@256 Support Dynamics Metrics Support Counts RLVR SRR NDR SDS NSCR P E S O RLVR SRR NDR SDS NSCR P E S O

AIME24

0 83.33 - - - - - - - -

AIME25

0 70.00 - - - - - - - 30 80.00 0.960 0.000 0.000 -0.040 24 0 1 5 30 70.00 0.905 0.095 0.172 0.000 19 2 2 7 60 80.00 0.960 0.000 0.000 -0.040 24 0 1 5 60 73.33 0.905 0.136 0.237 0.042 19 3 2 6 90 83.33 1.000 0.000 0.000 0.000 25 0 0 5 90 70.00 0.905 0.095 0.172 0.000 19 2 2 7

120 80.00 0.960 0.000 0.000 -0.040 24 0 1 5 120 73.33 0.905 0.136 0.237 0.042 19 3 2 6 150 80.00 0.960 0.000 0.000 -0.040 24 0 1 5 150 73.33 0.905 0.136 0.237 0.042 19 3 2 6 180 83.33 1.000 0.000 0.000 0.000 25 0 0 5 180 63.33 0.905 0.000 0.000 -0.095 19 0 2 9 210 76.67 0.920 0.000 0.000 -0.080 23 0 2 5 210 66.67 0.905 0.050 0.095 -0.045 19 1 2 8 240 73.33 0.880 0.000 0.000 -0.120 22 0 3 5 240 66.67 0.905 0.050 0.095 -0.045 19 1 2 8 270 73.33 0.880 0.000 0.000 -0.120 22 0 3 5 270 63.33 0.905 0.000 0.000 -0.095 19 0 2 9 300 73.33 0.880 0.000 0.000 -0.120 22 0 3 5 300 66.67 0.864 0.095 0.172 -0.042 18 2 3 7

AMC23

0 97.50 - - - - - - - -

MATH500

0 99.20 - - - - - - - 30 97.50 1.000 0.000 0.000 0.000 39 0 0 1 30 98.80 0.996 0.000 0.000 -0.004 494 0 2 4 60 97.50 1.000 0.000 0.000 0.000 39 0 0 1 60 98.80 0.996 0.000 0.000 -0.004 494 0 2 4 90 97.50 1.000 0.000 0.000 0.000 39 0 0 1 90 99.20 0.998 0.002 0.004 0.000 495 1 1 3

120 97.50 1.000 0.000 0.000 0.000 39 0 0 1 120 99.00 0.996 0.002 0.004 -0.002 494 1 2 3 150 97.50 1.000 0.000 0.000 0.000 39 0 0 1 150 98.80 0.996 0.000 0.000 -0.004 494 0 2 4 180 97.50 1.000 0.000 0.000 0.000 39 0 0 1 180 99.00 0.998 0.000 0.000 -0.002 495 0 1 4 210 97.50 1.000 0.000 0.000 0.000 39 0 0 1 210 98.80 0.994 0.002 0.004 -0.004 493 1 3 3 240 97.50 1.000 0.000 0.000 0.000 39 0 0 1 240 98.60 0.992 0.002 0.004 -0.006 492 1 4 3 270 97.50 1.000 0.000 0.000 0.000 39 0 0 1 270 98.40 0.992 0.000 0.000 -0.008 492 0 4 4 300 97.50 1.000 0.000 0.000 0.000 39 0 0 1 300 97.80 0.984 0.002 0.004 -0.014 488 1 8 3

Minerva

0 62.50 - - - - - - - -

Olympiad

0 88.00 - - - - - - - -

30 62.13 0.935 0.059 0.111 -0.006 159 10 11 92 30 86.67 0.963 0.022 0.043 -0.015 572 13 22 68 60 59.93 0.929 0.031 0.059 -0.040 158 5 12 97 60 86.81 0.960 0.027 0.053 -0.013 570 16 24 65 90 61.76 0.924 0.065 0.122 -0.011 157 11 13 91 90 86.07 0.961 0.017 0.034 -0.022 571 10 23 71

120 62.87 0.953 0.053 0.100 0.006 162 9 8 93 120 86.52 0.961 0.022 0.044 -0.016 571 13 23 68 150 62.50 0.947 0.053 0.100 0.000 161 9 9 93 150 85.48 0.955 0.017 0.034 -0.028 567 10 27 71 180 59.56 0.906 0.049 0.094 -0.045 154 8 16 94 180 84.00 0.944 0.011 0.021 -0.045 561 6 33 75 210 60.29 0.924 0.043 0.082 -0.034 157 7 13 95 210 82.81 0.926 0.016 0.032 -0.058 550 9 44 72 240 61.40 0.941 0.042 0.080 -0.017 160 7 10 95 240 82.81 0.931 0.011 0.021 -0.058 553 6 41 75 270 59.19 0.906 0.043 0.083 -0.051 154 7 16 95 270 81.33 0.912 0.013 0.025 -0.075 542 7 52 74 300 58.46 0.912 0.025 0.049 -0.063 155 4 15 98 300 79.70 0.897 0.009 0.018 -0.093 533 5 61 76

- 5.2. Results: Precision Gains, Entropy Dynamics, and Trade-offs

like AceReason and Skywork display similar or even lower token-level entropy relative to their base counterparts, and prior work has documented sharp entropy collapse in early training phases (Cui et al., 2025).

Consistent gains in precision, but sharper global distributions. Tab. 5 shows that RLVR consistently improves avg@32 across benchmarks, raising average performance from 54.5% to 65.4% for ProRL and from 43.0% to 61.3% for DAPO (Yu et al., 2025). However, this increased precision comes at a cost: RLVR systematically reduces answerlevel entropy, indicating a collapse onto fewer distinct solutions and empirically validating our theoretical prediction that reward optimization sharpens output distributions around known modes, thereby reducing effective support coverage. Notably, intrinsically harder tasks, like AIME or Minerva, still exhibit higher absolute answer-level entropy for both the base and RLVR models, suggesting that challenging problems inherently foster broader solution spaces that require exploration over more diverse completions.

More importantly, increased token-level entropy does not imply greater exploration of the output space. Despite appearing more stochastic at the step level, RLVR models frequently converge onto a smaller set of final answers—reflected in lower answer-level entropy. Notably, even between two models built on the same base (DeepSeek7B), Skywork-OR1-7B shows lower token-level entropy than AceReason-7B, yet exhibits higher answer-level entropy. This contrast highlights that local uncertainty does not reliably predict the diversity of final solutions, revealing a critical decoupling between local uncertainty and global diversity. We refer to this phenomenon as local stochasticity without global exploration: the model exhibits variability in generation but ultimately collapses to a narrow set of solutions. Thus, token-level entropy should not be conflated with genuine exploratory behavior, and interpreting entropy dynamics in RLVR requires distinguishing between stepwise uncertainty and overall support expansion.

Decoupled local uncertainty and global diversity. While answer-level entropy consistently declines, tokenlevel entropy exhibits more varied behavior. In models like ProRL and DAPO, it increases, suggesting greater local uncertainty during generation, possibly due to longer or more elaborated reasoning chains that introduce additional decision points or “forking” tokens (Wang et al., 2025). However, this pattern is far from universal: other RLVR models

Implications. Our empirical analysis reveals a trade-off in RLVR: it improves precision by amplifying high-reward outputs, but simultaneously narrows the diversity of global

- Table 5. Summary of avg@32 accuracy, response length, and entropy metrics across math reasoning benchmarks (row colors: base models, RLVR models). RLVR consistently improves accuracy and alters distributional properties. While answer-level entropy consistently decreases, token-level entropy shows more varied behavior across models.

Metric Model AIME24 AMC23 MATH500 Minerva Olympiad Avg. DeepSeek-1.5B 31.15 72.81 85.01 32.18 51.55 54.54 ProRL-1.5B 45.62 85.70 92.01 39.27 64.56 65.43

DeepSeek-7B 53.23 89.30 93.95 43.07 66.67 69.24 avg@32 AceReason-7B 65.83 95.08 95.81 45.35 73.92 75.20 Acc. (%) Skywork-OR1-7B 67.40 93.59 95.73 43.81 73.05 74.71

DeepSeek-14B 67.81 95.39 95.28 46.43 72.06 75.39 AceReason-14B 77.29 98.67 97.01 47.20 77.74 79.58

Qwen2.5-32B 18.12 55.23 75.84 24.55 41.40 43.03 DAPO-32B 51.25 92.81 80.75 32.50 49.15 61.29

DeepSeek-1.5B 16363 9979 5700 8194 11873 10422 ProRL-1.5B 7786 6294 5070 6569 6678 6479 DeepSeek-7B 13613 6402 4125 5595 8988 7745

Response AceReason-7B 10740 5961 4313 6261 7703 6995 Length Skywork-OR1-7B 15628 8282 5735 8742 12094 10096

DeepSeek-14B 11295 5735 3781 4919 8042 6755 AceReason-14B 13871 7239 4622 7720 10033 8697

Qwen2.5-32B 1247 874 585 3544 881 1426 DAPO-32B 6908 3157 3386 5665 5827 4989

DeepSeek-1.5B 0.45 0.40 0.42 0.49 0.44 0.44 ProRL-1.5B 0.47▲ 0.51▲ 0.54▲ 0.55▲ 0.52▲ 0.52▲ DeepSeek-7B 0.38 0.34 0.35 0.39 0.38 0.37

Token-Level AceReason-7B 0.18▼ 0.23▼ 0.27▼ 0.24▼ 0.23▼ 0.23▼ Entropy Skywork-OR1-7B 0.14▼ 0.16▼ 0.19▼ 0.17▼ 0.16▼ 0.16▼

DeepSeek-14B 0.33 0.30 0.32 0.35 0.33 0.33 AceReason-14B 0.12▼ 0.13▼ 0.15▼ 0.15▼ 0.14▼ 0.14▼ Qwen2.5-32B 0.17 0.16 0.15 0.28 0.15 0.18

DAPO-32B 0.26▲ 0.19▲ 0.27▲ 0.44▲ 0.30▲ 0.29▲

DeepSeek-1.5B 2.15 0.91 0.46 1.65 1.33 1.30 ProRL-1.5B 1.24 0.35 0.18 0.90 0.63 0.66 DeepSeek-7B 1.47 0.36 0.18 0.96 0.80 0.75

Answer-Level AceReason-7B 0.96 0.14 0.11 0.77 0.53 0.50 Entropy Skywork-OR1-7B 0.97 0.20 0.12 0.80 0.58 0.54

DeepSeek-14B 1.01 0.14 0.13 0.83 0.59 0.54 AceReason-14B 0.66 0.06 0.07 0.67 0.44 0.38

Qwen2.5-32B 2.37 1.32 0.68 2.27 1.41 1.61 DAPO-32B 1.12 0.09 0.26 0.96 0.63 0.61

solutions. This limitation is especially consequential in domains that admit multiple valid answers or benefit from creative reasoning, underscoring the need for explicit exploration mechanisms or diversity-promoting strategies to complement standard RLVR. Moreover, the observed divergence between token- and answer-level entropy highlights the need for a more nuanced interpretation of stochasticity in reward-optimized models, showing that precision gains often come at the expense of global diversity, and that maintaining controlled variability is critical for sustaining effective exploration.

### 6. Conclusion

We show that current RLVR increases precision by sharpening probability mass around known high-reward trajectories while largely preserving the base model’s support. Crucially, this sharpening goes beyond pruning incorrect outputs: it concentrates mass on a narrower subset of correct solutions, sometimes excluding valid alternatives that the more diverse base model could recover. Moreover, the gap between tokenlevel uncertainty and answer-level diversity suggests that stepwise stochasticity alone is insufficient for global exploration. To extend reasoning beyond the base model’s scope, RLVR must be paired with explicit exploration strategies

or off-policy mechanisms that allocate probability mass to underrepresented regions of the solution space.

### Impact Statements

This work examines the strengths and limitations of reinforcement learning with verifiable rewards (RLVR) in large language models, focusing on how such training reshapes the accessibility of correct solutions rather than introducing new capabilities. By providing empirical evidence that RLVR primarily improves precision while potentially narrowing solution diversity, our findings aim to help researchers and practitioners better understand the trade-offs involved in deploying RLVR-based systems. We do not anticipate direct negative societal impacts from this analysis; instead, we hope it encourages more transparent evaluation practices and motivates the development of training methods that balance reliability with broader reasoning coverage.

### References

Abdin, M., Agarwal, S., Awadallah, A., Balachandran, V., Behl, H., Chen, L., de Rosa, G., Gunasekar, S., Javaheripi, M., Joshi, N., et al. Phi-4-reasoning technical report. arXiv preprint arXiv:2504.21318, 2025.

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Amit, N., Goldwasser, S., Paradise, O., and Rothblum, G. N. A theory for worst-case vs. average-case guarantees for LLMs. In Advances in Neural Information Processing Systems 39: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, 2025.

Cai, X.-Q., Wang, W., Liu, F., Liu, T., Niu, G., and Sugiyama, M. Reinforcement learning with verifiable yet noisy rewards under imperfect verifiers. arXiv preprint arXiv:2510.00915, 2025.

Chang, H.-F. and Li, T. A framework for collaborating a large language model tool in brainstorming for triggering creative thoughts. Thinking Skills and Creativity, pp. 101755, 2025.

Chen, G., Dong, S., Shu, Y., Zhang, G., Sesay, J., Karlsson, B. F., Fu, J., and Shi, Y. Autoagents: A framework for automatic agent generation. arXiv preprint arXiv:2309.17288, 2023.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

- Chen, Y., Yang, Z., Liu, Z., Lee, C., Xu, P., Shoeybi, M., Catanzaro, B., and Ping, W. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. arXiv preprint arXiv:2505.16400, 2025a.
- Chen, Z., Qin, X., Wu, Y., Ling, Y., Ye, Q., Zhao, W. X., and Shi, G. Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025b.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Ding, S., Liu, Z., Dong, X., Zhang, P., Qian, R., He, C., Lin, D., and Wang, J. Songcomposer: A large language model for lyric and melody composition in song generation. arXiv preprint arXiv:2402.17645, 2024.

Feizi, S., Hajiaghayi, M., Rezaei, K., and Shin, S. Online advertisements with llms: Opportunities and challenges. arXiv preprint arXiv:2311.07601, 2023.

Feng, Y., Wang, J., Zhou, L., Lei, Z., and Li, Y. Doctoragentrl: A multi-agent collaborative reinforcement learning system for multi-turn clinical dialogue. arXiv preprint arXiv:2505.19630, 2025.

Guo, D., Yang, D., Zhang, H., and Song, J. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.07570, 2025. URL https://arxiv.org/abs/2501.07570.

Habib, N., Fourrier, C., Kydl´ıˇcek, H., Wolf, T., and Tunstall, L. Lighteval: A lightweight framework for llm evaluation, 2023. URL https://github.com/ huggingface/lighteval.

He, A., Fried, D., and Welleck, S. Rewarding the unlikely: Lifting grpo beyond distribution sharpening. arXiv preprint arXiv:2506.02355, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Hu, J. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.

Huang, F., Huang, G., Fan, X., He, Y., Liang, X., Chen, X., Jiang, Q., Khan, F. N., Jiang, J., and Wang, Z. Beyond the exploration-exploitation trade-off: A hidden state approach for llm reasoning in rlvr. arXiv preprint arXiv:2509.23808, 2025.

Jaech, A. et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Kool, W., van Hoof, H., and Welling, M. Buy 4 reinforce samples, get a baseline for free! ICLR 2019 Workshop drlStructPred, 2019.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Lai, Y., Zhong, J., Li, M., Zhao, S., Li, Y., Psounis, K., and Yang, X. Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models. arXiv preprint arXiv:2503.13939, 2025.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

Li, Y., Xu, Z., Jiang, F., Ramasubramanian, B., Niu, L., Lin, B. Y., Yue, X., and Poovendran, R. Temporal sampling for forgotten reasoning in llms. arXiv preprint arXiv:2505.20196, 2025.

Liu, J. and Zhang, L. Code-r1: Reproducing r1 for code with reliable rewards. 2025.

Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025a.

Liu, X., Liang, T., He, Z., Xu, J., Wang, W., He, P., Tu, Z., Mi, H., and Yu, D. Trust, but verify: A self-verification approach to reinforcement learning with verifiable rewards. arXiv preprint arXiv:2505.13445, 2025b.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of

foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Luo, M., Tan, S., Huang, R., Patel, A., Ariyak, A., Wu, Q., Shi, X., Xin, R., Cai, C., Weber, M., et al. Deepcoder: A fully open-source 14b coder at o3-mini level. Notion Blog, 2025a.

Luo, M., Tan, S., Huang, R., Shi, X., Xin, R., Cai, C., Patel, A., Ariyak, A., Wu, Q., Zhang, C., Li, L. E., Popa, R. A., and Stoica, I. Deepcoder: A fully open-source 14b coder at o3-mini level. https://pretty-radio-b75.notion.site/DeepCoder-AFully-Open-Source-14B-Coder-at-O3-mini-Level1cf81902c14680b3bee5eb349a512a51, 2025b. Notion Blog.

Luo, M., Tan, S., Wong, J., Shi, X., Tang, W. Y., Roongta, M., Cai, C., Luo, J., Li, L. E., Popa, R. A., and Stoica, I. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025c. Notion Blog.

Ma, L., Liang, H., Qiang, M., Tang, L., Ma, X., Wong, Z. H., Niu, J., Shen, C., He, R., Cui, B., et al. Learning what reinforcement learning can’t: Interleaved online fine-tuning for hardest questions. arXiv preprint arXiv:2506.07527, 2025.

Peng, S., Kalliamvakou, E., Cihon, P., and Demirer, M. The impact of ai on developer productivity: Evidence from github copilot. arXiv preprint arXiv:2302.06590, 2023.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shah, D. J. et al. Rethinking reflection in pre-training. arXiv preprint arXiv:2504.04022, 2025.

Shao, R., Li, S. S., Xin, R., Geng, S., Wang, Y., Oh, S., Du, S. S., Lambert, N., Min, S., Krishna, R., Tsvetkov, Y., Hajishirzi, H., Koh, P. W., and Zettlemoyer, L. Spurious rewards: Rethinking training signals in rlvr, 2025. Notion Blog.

Shao, Z. et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Stojanovski, Z., Stanley, O., Sharratt, J., Jones, R., Adefioye,

- A., Kaddour, J., and K¨opf, A. Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards, 2025. URL https://arxiv.org/abs/ 2505.24760.

Sun, Y., Guo, J., Kok, S., Wang, Z., Wen, Z., and Zhang, Z. Efficient reinforcement learning for large language models with intrinsic exploration. arXiv preprint arXiv:2511.00794, 2025.

Tu, A., Xuan, W., Qi, H., Huang, X., Zeng, Q., Talaei, S., Xiao, Y., Xia, P., Tang, X., Zhuang, Y., et al. Position: The hidden costs and measurement gaps of reinforcement learning with verifiable rewards. arXiv preprint arXiv:2509.21882, 2025.

Wang, K., Pan, J., Shi, W., Lu, Z., Ren, H., Zhou, A., Zhan, M., and Li, H. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.

Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X., Yang, J., Zhang, Z., et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025.

Wang, X., Hu, Z., Lu, P., Zhu, Y., Zhang, J., Subramaniam, S., Loomba, A. R., Zhang, S., Sun, Y., and Wang, W. Scibench: Evaluating college-level scientific problemsolving abilities of large language models. arXiv preprint arXiv:2307.10635, 2023.

Wei, J., Karina, N., Chung, H. W., Jiao, Y. J., Papay, S., Glaese, A., Schulman, J., and Fedus, W. Measuring shortform factuality in large language models. arXiv preprint arXiv:2411.04368, 2024.

Wei, T., Zhao, L., Zhang, L., Zhu, B., Wang, L., Yang, H., Li, B., Cheng, C., L¨u, W., Hu, R., et al. Skywork: A more open bilingual foundation model. arXiv preprint arXiv:2310.19341, 2023.

Wei, Y., Zhao, L., Sun, J., Lin, K., Yin, J., Hu, J., Zhang, Y., Yu, E., Lv, H., Weng, Z., et al. Open vision reasoner: Transferring linguistic cognitive behavior for visual reasoning. arXiv preprint arXiv:2507.05255, 2025.

Wen, X., Liu, Z., Zheng, S., Xu, Z., Ye, S., Wu, Z., Liang, X., Wang, Y., Li, J., Miao, Z., et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

White, C., Dooley, S., Roberts, M., et al. Livebench: A challenging, contamination-limited llm benchmark. In International Conference on Learning Representations (ICLR), 2025.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Fan, T., Liu, G., Liu, L., Liu, X., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., and Huang, G. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025a. URL https:// arxiv.org/abs/2504.13837.

Yue, Y., Yuan, Y., Yu, Q., Zuo, X., Zhu, R., Xu, W., Chen, J., Wang, C., Fan, T., Du, Z., et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025b.

Zeng, W., Huang, Y., Liu, Q., Liu, W., He, K., Ma, Z., and He, J. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025. URL https://arxiv.org/abs/ 2503.18892.

Zhang, S., Liu, Q., Qin, G., Naumann, T., and Poon, H. Med-rlvr: Emerging medical reasoning from a 3b base model via reinforcement learning. arXiv preprint arXiv:2502.19655, 2025.

Zhao, H., Ye, C., Gu, Q., and Zhang, T. Sharp analysis for kl-regularized contextual bandits and rlhf. arXiv preprint arXiv:2411.04625, 2024.

Zhao, R. et al. Echo chamber: Rl post-training amplifies behaviors learned in pretraining. arXiv preprint arXiv:2504.07912, 2025.

Zhu, X., Xia, M., Wei, Z., Chen, W.-L., Chen, D., and Meng, Y. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.

### A. Detailed Statistics for Support Dynamics

We provide full per-model statistics that underlie the aggregate values in Tab. 1. For each model and domain (Math, Non-Math, and Overall), we report the raw counts of correct completions across the four empirical support categories: Preservation (P), Expansion (E), Shrinkage (S), and Out-of-Support (O). From these counts, we compute the derived metrics: Support Retention Rate (SRR), Net Discovery Rate (NDR), Support Dynamic Score (SDS), and Net Support Change Rate (NSCR).

These expanded tables enable a fine-grained comparison of how different RLVR variants and model scales redistribute probability mass across correct solutions. In particular, they clarify whether improvements in single-sample accuracy stem from strong preservation of the base model’s support, from genuine discovery of new solutions, or from trade-offs between expansion and shrinkage.

We include results for all evaluated models: ProRL-1.5B-V2, Nemotron-1-7B, Skywork-OR1-7B, AceReason-Nemotron-1-14B, Phi4-Reason-Plus-14B, and the visual reasoning model Kangheng-OVR-7B (VLM). These tables serve as the ground truth for the aggregate summaries in the main text and substantiate the claims about predominant preservation, limited expansion, and consistent shrinkage observed across both math and non-math domains.

Table 6. Support dynamics metrics and pass@k performance of ProRL-1.5B-v1, compared with its base model, DeepSeek-R1-DistillQwen-1.5B.

pass@k Performance Support Dynamics Metrics Support Counts Base RLVR SRR NDR SDS NSCR P E S O Math Reasoning Benchmarks (pass@8192)

Dataset

- AIME2024 93.3% 83.3% 0.893 0.000 0.000 -0.107 25 0 3 2

- AIME2025 80.0% 73.3% 0.833 0.091 0.164 -0.077 20 2 4 4 AMC23 100.0% 100.0% 1.000 0.000 0.000 0.000 40 0 0 0

MATH500 99.6% 99.4% 0.998 0.000 0.000 -0.002 497 0 1 2

Minerva 71.7% 63.6% 0.887 0.000 0.000 -0.113 173 0 22 77 Olympiad 92.7% 89.3% 0.958 0.005 0.010 -0.037 600 3 26 46

Non-Math Reasoning Benchmarks (pass@2048)

SimpleQA 23.3% 18.0% 0.743 0.038 0.073 -0.221 75 3 26 329 LiveBench-R 100.0% 94.0% 0.940 0.000 0.000 -0.060 94 0 6 0 LiveBench-C 62.5% 56.2% 0.838 0.069 0.128 -0.094 67 5 13 43 LiveBench-L 26.0% 24.0% 0.769 0.167 0.274 -0.067 10 2 3 35

SciBench 94.1% 90.5% 0.946 0.016 0.031 -0.038 616 10 35 31 LiveCodeBench v5 46.4% 43.0% 0.860 0.072 0.133 -0.069 129 10 21 163 LiveCodeBench v6 43.5% 42.0% 0.947 0.018 0.036 -0.034 54 1 3 73

Aggregate Statistics

Math – – 0.960 0.0037 0.0073 -0.036 1355 5 56 131 Non-Math – – 0.907 0.0288 0.0558 -0.064 1045 31 107 674

Overall – – 0.936 0.0148 0.0291 -0.049 2400 36 163 805

### B. Experimental Details

We provide comprehensive details of the experimental setup, including dataset descriptions and evaluation methodologies. A key aspect of our evaluation approach is the answer processing enhancement framework for Reasoning Gym, which addresses format compatibility challenges between base and ProRL models to ensure fair evaluation.

#### B.1. Evaluation Settings

We employed vLLM (Kwon et al., 2023) as the inference backend. For all models, we utilized a sampling temperature of 0.6, a top p value of 0.95, and a maximum response length of 32768.

- B.2. Datasets Math benchmarks. We utilized the complete datasets from MATH500 (Hendrycks et al., 2021), Minerva (Lewkowycz

- et al., 2022), OlympiadBench (He et al., 2024), AIME 2024, AIME 2025, and AMC 2023 for evaluating LLMs. For

- Table 7. Support dynamics metrics and pass@k performance of AceReason-Nemotron-1-7B, compared with its base model, DeepSeek-

- 7B.

pass@k Performance Support Dynamics Metrics Support Counts Base RLVR SRR NDR SDS NSCR P E S O Math Reasoning Benchmarks (pass@8192)

Dataset

- AIME2024 93.3% 93.3% 1.000 0.000 0.000 0.000 28 0 0 2

- AIME2025 100.0% 100.0% 1.000 0.000 0.000 0.000 30 0 0 0 AMC23 100.0% 100.0% 1.000 0.000 0.000 0.000 40 0 0 0

MATH500 99.8% 99.8% 1.000 0.000 0.000 0.000 499 0 0 1

Minerva 71.7% 71.0% 0.985 0.005 0.010 -0.010 192 1 3 76 Olympiad 96.0% 95.7% 0.991 0.006 0.012 -0.003 642 4 6 23

###### Non-Math Reasoning Benchmarks (pass@2048)

SimpleQA 38.6% 35.6% 0.862 0.065 0.121 -0.073 144 10 23 256 LiveBench-R 100.0% 99.0% 0.990 0.000 0.000 -0.010 99 0 1 0 LiveBench-C 85.9% 85.9% 0.991 0.009 0.018 0.000 109 1 1 17 LiveBench-L 24.0% 24.0% 0.833 0.167 0.278 0.000 10 2 2 36

SciBench 94.7% 93.5% 0.982 0.006 0.012 -0.012 643 4 12 33 LiveCodeBench v5 62.8% 62.5% 0.970 0.025 0.048 -0.005 197 5 6 115 LiveCodeBench v6 64.1% 63.4% 0.976 0.012 0.024 -0.012 82 1 2 46

###### Aggregate Statistics

Math – – 0.994 0.003 0.007 -0.003 1431 5 9 102 Non-Math – – 0.965 0.018 0.035 -0.018 1284 23 47 503

Overall – – 0.981 0.010 0.020 -0.010 2715 28 56 605

vision-language models, we evaluated on the testmini sets of MathVision (Wang et al., 2024) and MathVista (Lu et al., 2023).

Non-math benchmarks. For SimpleQA (Wei et al., 2024), we uniformly sampled 10% of the original dataset (433 samples) to enable efficient large-scale evaluation under high-pass conditions. For LiveBench (White et al., 2025), we used the 2024-11-25 version available on HuggingFace. To ensure unambiguous evaluation, we focused exclusively on tasks with binary correct/incorrect judgments and excluded tasks involving intermediate floating-point judgments, as these lack clear correctness criteria. Based on this selection criterion, we evaluated the following subsets: web of lies v2 and spatial subsets for Reasoning tasks (LiveBench-R), the typos subset for Language tasks (LiveBench-L), and all available data for Coding tasks (LiveBench-C). For SciBench (Wang et al., 2023), we evaluated on the complete dataset. For LiveCodeBench (Jain et al., 2024), we evaluated the dataset on both v5 and v6 versions. Due to computational efficiency considerations, we conducted LiveCodeBench evaluation exclusively on 1.5B and 7B models, excluding the 14B variants from this particular benchmark.

Reasoning Gym. For Reasoning Gym (Stojanovski et al., 2025), we employ the easy set from the version updated after commit 17a8431 in its repository as our default task configuration. This choice ensures consistency with the default task configuration used in prior evaluations, maintaining comparable experimental conditions. Additionally, we use the hard set as our challenging evaluation benchmark.

#### B.3. Answer Processing Enhancement in Reasoning Gym

We identified significant evaluation challenges when testing the base model on Reasoning Gym. The ProRL model, trained on Reasoning Gym data, predominantly produces responses that conform to the expected format, resulting in much higher accuracy. In contrast, the base model struggled with adherence to the format due to insufficiently detailed prompts, and its limited 1.5B parameter capacity made it particularly susceptible to evaluation inconsistencies. To address these issues, we enhanced both the answer extraction protocol and prompt design to ensure fair and objective accuracy assessments across both models. This causes the differences of ProRL’s reported performance and our evaluation results in Reasoning Gym.

- Table 8. Support dynamics metrics and pass@k performance of Skywork-OR1-7B, compared with its base model, DeepSeek-R1-DistillQwen-7B.

pass@k Performance Support Dynamics Metrics Support Counts Base RLVR SRR NDR SDS NSCR P E S O Math Reasoning Benchmarks (pass@8192)

Dataset

AIME2024 93.3% 93.3% 1.000 0.000 0.000 0.000 28 0 0 2 AIME2025 100.0% 100.0% 1.000 0.000 0.000 0.000 30 0 0 0

AMC23 100.0% 100.0% 1.000 0.000 0.000 0.000 40 0 0 0 MATH500 99.8% 99.8% 1.000 0.000 0.000 0.000 499 0 0 1

Minerva 71.7% 71.3% 0.985 0.010 0.020 -0.005 192 2 3 75 Olympiad 96.0% 91.4% 0.952 0.000 0.000 -0.048 617 0 31 27

###### Non-Math Reasoning Benchmarks (pass@2048)

SimpleQA 38.6% 37.0% 0.880 0.081 0.149 -0.039 147 13 20 253 LiveBench-R 100.0% 98.0% 0.980 0.000 0.000 -0.020 98 0 2 0 LiveBench-C 85.9% 85.2% 0.991 0.000 0.000 -0.009 109 0 1 18 LiveBench-L 24.0% 22.0% 0.917 0.000 0.000 -0.083 11 0 1 38

SciBench 94.7% 92.8% 0.974 0.006 0.012 -0.020 638 4 17 33 LiveCodeBench v5 62.8% 62.2% 0.966 0.025 0.049 -0.010 196 5 7 115 LiveCodeBench v6 64.1% 62.6% 0.952 0.024 0.048 -0.023 80 2 4 45

###### Aggregate Statistics

Math – – 0.976 0.001 0.003 -0.023 1406 2 34 105 Non-Math – – 0.961 0.018 0.036 -0.021 1279 24 52 502

###### Overall – – 0.969 0.010 0.020 -0.022 2685 26 86 607

- B.3.1. GENERAL ANSWER EXTRACTION PROTOCOL

First, we enhanced the answer extraction protocol with a hierarchical, priority-based extraction mechanism that processes responses through multiple fallback levels. Each level attempts to capture the model’s intended answer, and successful extraction at any level bypasses subsequent processing steps.

The strategy first attempts to extract content using the Reasoning Gym’s extract answer() function, which captures answers within <answer></answer> tags. This approach is given the highest priority because these tags are Reasoning Gym’s default format. When this method fails, the system searches for content within the final \boxed{} formatting.

For dice tasks using the base model, failed extract answer() attempts trigger additional processing through Lighteval (Habib et al., 2023)’s math normalizer() function. This function handles \boxed{} capture and converts a/b fractions to LATEX format \frac{a}{b}. When extract answer() successfully captures a/b fraction answers, the system applies Lighteval’s fix a slash b() function to achieve the same LATEX conversion.

For non-dice tasks or when using ProRL models, failed extract answer() attempts utilize Lighteval’s last boxed only string() and remove boxed() functions. These functions locate content within the final \boxed{}, primarily addressing cases where base model prompt modifications shifted from answer tags to boxed formatting.

As a final fallback, the system extracts content following </think> tags when all previous methods fail, and the response contains these markers. This safety mechanism captures base model responses that ignore formatting requirements in lengthy tasks.

- B.3.2. TASK-SPECIFIC PROCESSING MODIFICATIONS

Our core answer-processing pipeline applies to both models, with additional steps designed primarily to address format compatibility issues commonly encountered in base-model responses. Specifically, the processing logic for each task is enhanced as follows:

dice The ground truth for dice tasks uses a/b fraction format. Base models frequently express fractions in LATEX format, requiring format standardization for accurate evaluation. For base models only, we convert ground truth fractions from

a/b format to LATEX format \frac{a}{b} to ensure both model answers and ground truth use consistent LATEX formatting.

- Table 9. Support dynamics metrics and pass@k performance of Nemotron-1-14B, compared with its base model, DeepSeek-R1-DistillQwen-14B.

pass@k Performance Support Dynamics Metrics Support Counts Base RLVR SRR NDR SDS NSCR P E S O Math Reasoning Benchmarks (pass@4096)

Dataset

AIME2024 96.7% 93.3% 0.966 0.000 0.000 -0.034 28 0 1 1 AIME2025 100.0% 96.7% 0.967 0.000 0.000 -0.033 29 0 1 0

AMC23 100.0% 100.0% 1.000 0.000 0.000 0.000 40 0 0 0 MATH500 99.8% 99.8% 1.000 0.000 0.000 0.000 499 0 0 1

Minerva 71.7% 69.5% 0.959 0.011 0.021 -0.030 187 2 8 75 Olympiad 95.9% 95.6% 0.992 0.005 0.009 -0.003 642 3 5 25

###### Non-Math Reasoning Benchmarks (pass@1024)

SimpleQA 27.0% 26.8% 0.983 0.009 0.017 -0.008 115 1 2 315 LiveBench-R 99.0% 99.0% 1.000 0.000 0.000 0.000 99 0 0 1 LiveBench-C 92.2% 92.2% 1.000 0.000 0.000 0.000 118 0 0 10 LiveBench-L 46.0% 44.0% 0.957 0.000 0.000 -0.043 22 0 1 27

SciBench 93.1% 92.6% 0.992 0.003 0.006 -0.005 639 2 5 46

###### Aggregate Statistics

Math – – 0.990 0.0035 0.0070 -0.0069 1425 5 15 102 Non-Math – – 0.992 0.0030 0.0060 -0.0050 993 3 8 399

Overall – – 0.991 0.0033 0.0066 -0.0061 2418 8 23 501

ProRL dice processing maintains a/b formatting for both model answers and ground truth, leveraging the dice samples present in its training data.

prime factorization The ground truth format requires answers to be combinations of numbers and multiplication symbol (i.e., ×) only. We implement three key modifications to ensure compatibility with this requirement. First, we standardize LATEX multiplication symbols by replacing \times with × to meet the evaluation requirements, as base models frequently use LATEX multiplication symbols instead of standard multiplication signs. Second, we expand LATEX exponentiation by converting formats like aˆb into repeated multiplication (a × a × ... × a for b iterations), preventing errors when base models consolidate repeated factors into exponential notation. Third, we process response equations by retaining only right-side content when answers contain equals signs, transforming responses like “561 = 3 × 11 × 17” to “3 × 11 × 17” to eliminate question restatement that base models commonly include.

palindrome generation The ground-truth format expects palindromic character strings (sequences that read the same forward and backward). We remove excess whitespace to address frequent spacing issues in base model responses. This transformation converts spaced responses like “k h g a g h k” to “khgaghk”, preventing failures in string reversibility judgment that occur when spaces interfere with palindrome verification.

advanced geometry The ground truth format requires floating-point numbers. Our processing includes three main steps to address LATEX formatting issues commonly encountered in base-model output. First, we remove redundant LATEX expressions by eliminating \left and \right markers while converting ˆ\circ to ◦ symbol, addressing base models’ tendency to use LATEX for brackets and degree symbols. Second, we convert LATEX numerical expressions, transforming fractions \frac{a}{b} and other LATEX formats (\sqrt{a}, \sin{a}, \log{a}, \pi, etc.) into three-decimal floatingpoint numbers using the latex2sympy2 extended library’s latex2sympy() function. Third, we evaluate arithmetic expressions containing radicals (such as 2√16+5√4−3) by converting them into three-decimal floating-point numbers using Python’s built-in mathematical functions, handling cases where base models output final results as arithmetic expressions rather than computed values.

power function The ground-truth format uses scientific notation. We convert mixed LATEX and arithmetic symbol scientific notation to ensure format consistency. The system transforms patterns like “−2.36 × 10−16” or “1.5 × 105” to e-notation format (“-2.36e-16”, “1.5e5”), preventing numerically correct but format-incompatible evaluation errors when base models use mixed LATEX and arithmetic symbols for scientific notation.

- Table 10. Support dynamics metrics and pass@k performance of Phi4-Reason-Plus-14B, compared with its base model – Phi4-Reason14B.

Dataset

pass@k Performance Support Dynamics Metrics Support Counts Base RLVR SRR NDR SDS NSCR P E S O Math Reasoning Benchmarks (pass@4096)

AIME2024 96.7% 96.7% 1.000 0.000 0.000 0.000 29 0 0 1 AIME2025 100.0% 100.0% 1.000 0.000 0.000 0.000 30 0 0 0

AMC23 100.0% 100.0% 1.000 0.000 0.000 0.000 40 0 0 0 MATH500 100.0% 99.8% 0.998 0.000 0.000 -0.002 499 0 1 0

Minerva 66.2% 65.4% 0.972 0.017 0.033 -0.011 175 3 5 89 Olympiad 94.8% 94.7% 0.991 0.008 0.016 -0.002 634 5 6 30

Non-Math Reasoning Benchmarks (pass@1024)

SimpleQA 37.9% 37.4% 0.970 0.019 0.036 -0.012 159 3 5 266

LiveBench-R 100.0% 100.0% 1.000 0.000 0.000 0.000 100 0 0 0 LiveBench-C 97.7% 96.9% 0.992 0.000 0.000 -0.008 124 0 1 3 LiveBench-L 74.0% 74.0% 1.000 0.000 0.000 0.000 37 0 0 13

SciBench 94.2% 94.2% 0.992 0.008 0.015 0.000 647 5 5 35

Aggregate Statistics

Math – – 0.992 0.0057 0.0112 -0.0028 1407 8 12 120 Non-Math – – 0.990 0.0074 0.0148 -0.0028 1067 8 11 317

Overall – – 0.991 0.0064 0.0128 -0.0028 2474 16 23 437

- Table 11. Support dynamics metrics and pass@k performance of vision-language model OVR-7B-RL, compared with its base model, OVR-7B-ColdStart, across visual math reasoning benchmarks.

pass@k Performance Support Dynamics Metrics Support Counts Base RLVR SRR NDR SDS NSCR P E S O Math Reasoning Benchmarks (pass@8192)

Dataset

MathVista 49.1% 49.1% 0.998 0.002 0.004 0.000 490 1 1 508 MathVision 96.7% 96.4% 0.990 0.007 0.014 -0.003 291 2 3 8

Aggregate Statistics Math – – 0.995 0.0038 0.0076 -0.0013 781 3 4 516

arc 1d The ground truth format requires space-separated digit sequences. We handle two types of responses to meet this grid format requirement. For pure numerical responses, we insert spaces between consecutive digits, converting sequences like “22220000000000000000111” to “2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1”. For mixed numerical and textual responses, we extract digits and insert spaces, transforming LATEX grid formats like \begin{array}{cccc} 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 7 & 3 & 0 & 0 & 4 & 6 \\\end{array} to “0 0 0 0 0 0 0 0 7 3 0 0 4 6”, addressing base models’ tendency to output correct answers in LATEX grid format.

boxnet The ground truth format requires dictionary list formatting [{key: value}, ...]. We implement comprehensive cleaning of the JSON format to meet these evaluation requirements. Our processing pipeline includes several steps: rejecting pure numerical responses to prevent non-JSON format interference; removing JSON markdown wrappers that eliminate ‘‘‘json {content} ‘‘‘ markers; converting single dictionaries to single-element dictionary lists (dict → [dict]); and filtering illegal elements by removing non-dictionary components from JSON lists. Additionally, we clean nested structure values within individual dictionary entries. For nested lists, we extract the first element as the value ([{key1: [value1, value2, ...]}, ...] → [{key1: value1}, ...]). For nested dictionaries, we select matching key values when available ([{key1: {key1: value1, key2: value2, ...}}, ...] → [{key1: value1}, ...]) or default to the first element value when keys don’t match ([{key1: {key2:

value2, key3: value3}}, ...] → [{key1: value2}, ...]). These modifications preserve the model response content to the maximum extent while ensuring compliance with the ground-truth format.

Table 12. Support dynamics metrics and pass@256 performance of OLMo-2-0425-1B-DPO vs RLVR.

pass@k Performance Support Dynamics Metrics Support Counts Base RLVR SRR NDR SDS NSCR P E S O Math Reasoning Benchmarks (pass@256)

Dataset

- AIME2024 20.00% 16.67% 0.833 0.000 0.000 -0.167 5 0 1 24

- AIME2025 23.33% 20.00% 0.571 0.333 0.421 -0.111 4 2 3 21 AMC23 85.00% 77.50% 0.912 0.000 0.000 -0.088 31 0 3 6

MATH500 77.80% 76.20% 0.920 0.060 0.113 -0.019 358 23 31 88

Minerva-Math 33.82% 32.35% 0.837 0.125 0.218 -0.039 77 11 15 169 Olympiad 49.93% 49.33% 0.849 0.141 0.242 -0.010 286 47 51 291

Aggregate Statistics Math – – 0.880 0.098 0.177 -0.022 761 83 104 599

#### B.4. Entropy Analysis

Setup. In entropy analysis, we configure the models with a sampling temperature of 0.6, a top p value of 0.95, and a maximum response length of 32768 tokens to balance response diversity and quality. Each model generates 32 completions per problem following the avg@32 evaluation protocol, and all reported metrics (accuracy, response length, token-level entropy, and answer-level entropy) are averaged across these 32 completions and across all test problems in each benchmark.

Models. We evaluate a diverse set of reasoning models to understand entropy characteristics across different training paradigms and parameter scales, as summarized in the table below.

Table 13. Models evaluated in the entropy analysis.

Name Full Model Name Type Parameters

DeepSeek-1.5B DeepSeek-R1-Distill-Qwen-1.5B Base 1.5B ProRL-1.5B Nemotron-Research-Reasoning-Qwen-1.5B RLVR 1.5B

DeepSeek-7B DeepSeek-R1-Distill-Qwen-7B Base 7B AceReason-7B AceReason-Nemotron-7B RLVR 7B Skywork-OR1-7B Skywork-OR1-7B RLVR 7B

DeepSeek-14B DeepSeek-R1-Distill-Qwen-14B Base 14B AceReason-14B AceReason-Nemotron-14B RLVR 14B

Qwen2.5-32B Qwen2.5-32B Base 32B DAPO-32B DAPO-Qwen-32B RLVR 32B

Entropy Computation. For token-level entropy computation, we use teacher forcing to obtain probability estimates. Specifically, after generating the 32 completions with the specified sampling parameters, we feed each generated sequence back to the model and perform a single forward pass to compute the probability distribution over the vocabulary at each token position. Answer-level entropy is computed by first extracting the final answer from each completion using Lighteval (Habib

- et al., 2023), then calculating the entropy over the distribution of unique answers across the 32 completions. This approach allows us to compute both token-level and answer-level entropy directly from the model’s probability distributions without introducing additional sampling variance.

### C. Theoretical Limits of RLVR

#### C.1. Support Preservation: Why RLVR Rarely Discovers New Modes

We begin with a core limitation of RLVR: it is inherently constrained to operate within the support of the base model’s distribution. Since RLVR relies on gradient signals derived from samples generated by the base model, it cannot assign a nonzero probability to any solution that can never be sampled from q(· | x). As a result, any correct output y∗ with q(y∗ | x) = 0 remains inaccessible to policy gradient updates, regardless of reward. We formalize this intuition with Theorem C.1, which makes precise how RLVR’s reliance on the base model’s sampling prevents discovering truly new solutions.

Table 14. Pass@k comparison of Qwen2.5-Math-7B (SFT vs DAPO) across five math benchmarks. Higher values indicate better performance.

- AIME2024 @1 @2 @4 @8 @16 @32 @64 @128 @256

SFT 20.00 30.00 30.00 33.33 43.33 50.00 63.33 63.33 63.33 DAPO 20.00 26.67 26.67 40.00 46.67 56.67 60.00 63.33 66.67

- AIME2025 @1 @2 @4 @8 @16 @32 @64 @128 @256

SFT 16.67 16.67 26.67 30.00 33.33 40.00 43.33 50.00 63.33 DAPO 16.67 20.00 30.00 43.33 46.67 50.00 50.00 50.00 56.67

AMC23 @1 @2 @4 @8 @16 @32 @64 @128 @256 SFT 60.00 72.50 77.50 85.00 90.00 95.00 97.50 97.50 100.00 DAPO 62.50 72.50 85.00 87.50 87.50 92.50 92.50 92.50 92.50 MATH500 @1 @2 @4 @8 @16 @32 @64 @128 @256 SFT 78.20 85.20 90.00 92.80 95.00 96.20 98.00 98.80 99.00 DAPO 85.00 89.80 91.40 93.00 93.20 93.80 94.40 94.80 95.20 Minerva-Math @1 @2 @4 @8 @16 @32 @64 @128 @256 SFT 36.03 39.71 43.75 47.79 51.47 54.04 63.60 65.44 66.91 DAPO 40.81 45.96 48.16 51.10 51.10 52.57 53.68 54.41 55.15

Theorem C.1 (Support Preservation under RLVR). Let πθ(y | x) be the RLVR-trained distribution obtained via standard on-policy gradient updates on verifiable rewards R. Then for all x ∈ X,

##### supp(πθ(· | x)) ⊆ supp(q(· | x)).

In particular, if q(y∗ | x) = 0 for some correct solution y∗, then RLVR cannot discover y∗.

Proof. By construction, we initialize the RLVR policy to the base model as πθ

##### (y | x) = q(y | x). Hence supp πθ

0

##### (· | x) = supp q(· | x) .

0

Inductive step. Assume that at some iteration θ we have

##### πθ(y∗ | x) = 0 for a particular y∗.

All standard policy-gradient updates (e.g. REINFORCE, PPO, GRPO) take the form

θ(y|x)

θ(·|x) R(x,y) − β−1 log π

θ′ = θ + η ∇θEy∼π

q(y|x) ,

where η is the learning rate. Since the outer expectation is over y ∼ πθ, any y∗ ∈ C with πθ(y∗ | x) = 0 is never sampled and thus contributes no gradient component. Therefore

##### πθ′(y∗ | x) = 0,

and the support of πθ′ remains a subset of that of q. Conclusion. By induction, none of the updates can introduce positive probability mass on any y∗ ∈ C for which q(y∗ | x) = 0. Equivalently,

##### supp πθ(· | x) ⊆ supp q(· | x) ,

indicating that any correct solution y∗ with q(y∗ | x) = 0 remains unreachable by the RLVR policy.

| |
|---|

Corollary C.2 (Asymptotic Sampling Upper Bound). Let pass@kp(x) be the probability that at least one out of k samples yi ∼ p(· | x) is correct, i.e. pass@kp(x) = 1 − Pry∼p[R(x,y) = 0] k. Under the conditions of Thm. C.1 and the sampling independence, we have

pass@kπ

pass@kq(x).

(x) ≤ limsup

limsup

θ

k→∞

k→∞

Proof. From Thm. C.1, support preservation implies supp(πθ(·|x)) ⊆ supp(q(·|x)). Thus, for any y ∈ C, πθ(y|x) >

- 0 =⇒ q(y|x) > 0. Define the total mass on correct completions by

πθ(C) = Pr

[R(x,y) = 1], q(C) = Pr

[R(x,y) = 1].

y∼πθ

y∼q

Here, samples are assumed independent across the different draws of LLMs; otherwise, we can only assert an upper bound using union bounds. As k → ∞, the pass@k success probability becomes

(x) = 1 − (1 − πθ(C))k −→

pass@kπ

θ

1, πθ(C) > 0, 0, πθ(C) = 0,

and similarly for q. Because support preservation ensures that any correct completion reachable under πθ must also be reachable under q,

πθ(C) > 0 =⇒ q(C) > 0. Hence, the asymptotic success probability satisfies

pass@kπ

pass@kq(x).

(x) ≤ lim

lim

θ

k→∞

k→∞

| |
|---|

Theorem C.1 and Corollary C.2 prove that RLVR optimization cannot expand the search space beyond the initial support of the base model. This limitation arises because on-policy sampling means the model updates only from what it already samples — lacking representational coverage means no gradient can ever push probability mass toward truly unseen solutions. Even when rewards provide a clear training signal, RLVR cannot access or discover solutions that the base model assigns zero probability.

This manifests as a trade-off between sharpness and diversity: RLVR can improve pass@1 by concentrating mass on known high-reward modes but tends to reduce pass@k performance for larger k, where broader coverage is beneficial. By contrast, the base model may occasionally sample correct answers from its long-tail distribution, giving it a statistical edge under high-k evaluations (Yue et al., 2025a; Liu et al., 2025a). This asymptotic upper bound captures a ceiling: no matter how many samples are drawn, the RLVR-trained model cannot exceed the base model’s pass@k in the limit.

Theorem C.3 (Empirical Support Preservation). Assume ϵ is below the finite-sample detectability threshold used in rollouts. Then, under standard sampling and update procedures with a finite sample budget, we have

##### suppϵ πθ(· | x) ⊆ suppϵ q(· | x) .

Proof. Following Zhu et al. (2025), the total update in RLVR training decomposes into

##### ∇Ltotal = ∇LPSR + ∇LNSR,

where PSR (positive sample reinforcement) promotes correct completions and NSR (negative sample reinforcement) demotes incorrect ones while redistributing mass proportionally to the current policy. If y ∈/ suppϵ(q), then q(y | x) ≤ ϵ, so y is not ϵ-detectable under the base model and will not be sampled as a positive example. Thus ∇LPSR has no contribution to y, and its probability can only change via ∇LNSR.

NSR gradient structure. We first analyze a single decoding position. At any position with logits z and probabilities πv, for a sampled wrong token yt and learning rate η, the NSR gradient satisfies

∂LNSR ∂zv ∝

−

−πy

##### (1 − πy

##### ), v = yt, πv πy

t

t

, v ̸= yt,

t

∂LNSR ∂zv

∆zv = η −

.

The softmax policy update under a small NSR step ∆z has the multiplicative form

π′(v) =

π(v) exp(∆zv) u π(u) exp(∆zu)

.

For a correct token a ̸= yt, this gives

= −ηπ(yt)(1 − π(yt)), ∆zu = ηπ(u)π(yt) ≥ 0 (u ∈/ {a,yt}). Therefore,

∆za = ηπ(a)π(yt), ∆zy

t

π′(a) π(a)

exp(∆za) u π(u)exp(∆zu) ≤

exp(ηπ(a)π(yt)) 1 − ηπ(yt)2

.

=

Using exp(ηπ(a)π(yt)) ≤ exp(ηπ(yt)) and 1/(1 − x) ≤ e2x for x ∈ [0,1/2], we obtain

π′(a) π(a) ≤ exp 2ηπ(yt) .

Iterating for K steps yields the token-level bound

π(K)(a | x,y<t) ≤ π(0)(a | x,y<t) exp 2η

K−1

π(k)(yt | x,y<t) ≤ π(0)(a | x,y<t)e2ηK.

k=0

Extension to sequences. For a full sequence y⋆ = (a1,...,aT), the autoregressive factorization gives

π(K)(y⋆ | x) =

T

π(K)(at | x,a<t).

t=1

Applying the token-level bound at each position t,

π(K)(at | x,a<t) ≤ π(0)(at | x,a<t)e2ηK. Multiplying across all T positions yields

π(K)(y⋆ | x) ≤ π(0)(y⋆ | x) exp(2ηTK).

Conclusion. Thus, if a sequence y lies outside suppϵ(q) so that π(0)(y | x) ≤ ϵ, then even after K NSR updates we have π(K)(y | x) ≤ ϵe2ηTK. As ϵ → 0, multiplying it by any finite constant still yields a vanishingly small quantity; thus, any finite multiple of ϵ is statistically negligible (i.e., undetectable in practice). Therefore, ϵe2ηTK remains negligible for any finite K and T, and

suppϵ(πθ) ⊆ suppϵ(q).

| |
|---|

In this sense, RLVR inherits both the inductive biases and structural limitations of its initialization. Without deliberate intervention or scaling, it remains confined to the functional expressivity of the base model. Our framework formalizes why RLVR often improves sampling efficiency but rarely produces qualitatively new reasoning capabilities.

#### C.2. A Variational and Support-bounded Policy Update

We now present a unified view of the RLVR objective through the lens of variational inference. This reveals why RLVR is inherently support-bounded: it makes minimal updates to the base distribution while ensuring improved performance.

Proposition C.4 (KL Projection onto Reward-Consistent Distributions). Let ∆(Y) be the probability simplex over the finite output space Y. Define the set of feasible policies that achieve at least a target expected reward ρ:

Pρ := {p(y | x) ∈ ∆(Y) | Ep[R(x,y)] ≥ ρ}.

Then the solution to the variational problem, minπ∈P

KL(π ∥q), is the distribution within Pρ that is closest in KL divergence to the base model. The optimal policy takes the form:

ρ

π∗(y | x) ∝ q(y | x) · exp(βR(x,y)), where β ∈ R≥0 is the dual variable associated with the reward constraint and β = 0 degenerates to the base policy q. Proof. We provide two closely related derivations to illuminate the same optimal solution from both a hard-constrained and a soft-regularized perspective. Convexity of Feasible Set Pρ. We first prove the convexity of Pρ. Recall Pρ = p ∈ ∆(Y) : y p(y)R(x,y) ≥ ρ , where ∆(Y) denotes the probability simplex over Y. Take any two distributions p1,p2 ∈ Pρ and let λ ∈ [0,1]. Consider the convex combination

pλ := λp1 + (1 − λ)p2.

Since ∆(Y) is convex, we have pλ ∈ ∆(Y). Next, because p1,p2 ∈ Pρ, it follows that

p1(y)R(x,y) ≥ ρ and

p2(y)R(x,y) ≥ ρ.

y

y

Thus,

p1(y)R(x,y) + (1 − λ)

p2(y)R(x,y) ≥ λρ + (1 − λ)ρ = ρ.

pλ(y)R(x,y) = λ

y

y

y

Hence pλ ∈ Pρ. This shows that Pρ is convex. Convexity, existence, and strong duality. We then verify the foundational properties of the optimization problem. Recall we wish to solve

KL(π∥q), where Pρ = π ∈ ∆(Y) :

π(y)R(x,y) ≥ ρ .

min

π∈Pρ

y

The objective function KL(π∥q) is convex in π over the probability simplex ∆(Y), since relative entropy is jointly convex and thus convex in π for fixed q. The feasible set Pρ is also convex.

Moreover, if there exists a strictly feasible distribution π such that y π(y)R(x,y) > ρ, then by Slater’s condition, strong duality holds. This guarantees that the optimal value of the primal problem equals the optimal value of its Lagrangian

dual, and the Karush-Kuhn-Tucker (KKT) conditions characterize the optimal solution. In typical applications—where q arises from softmax-based models with full support—such strictly feasible distributions exist, ensuring that our subsequent Lagrangian approach is valid.

#### 1) Hard-constrained formulation (projection perspective). Consider the optimization problem:

KL(π∥q) s.t. Eπ[R(x,y)] ≥ ρ,

min

π

Using the method of Lagrange multipliers, the Lagrangian is:

y

π(y | x) = 1, π(y | x) ≥ 0.

π(y | x) q(y | x) − β

π(y | x)R(x,y) − ρ + λ

π(y | x) − 1 .

L(π,β,λ) =

π(y | x)log

y

y

y

Here, we compute the derivative concerning π(y | x) for fixed multipliers, thereby finding the stationary points of the Lagrangian. Specifically, we take derivative with respect to π(y | x) and set it to zero:

π(y | x) q(y | x)

+ 1 − βR(x,y) + λ = 0. Solving for π yields:

log

π(y | x) ∝ q(y | x) · exp(βR(x,y)).

- 2) Soft-regularized formulation (dual perspective). Alternatively, assume RLVR solves the entropy-regularized objective

Ey∼π[R(x,y)] − β−1KL(π ∥q),

πθ = arg max π≪q

for some inverse temperature parameter β > 0. Here, the constraint π ≪ q denotes that π is absolutely continuous with respect to q, meaning π(y | x) > 0 only if q(y | x) > 0.1The objective is equivalent to the following minimization:

KL(π ∥q) − β Ey∼π[R(x,y)].

πθ = arg min

π∈∆(Y)

The Lagrangian becomes

 

 ,

π(y) q(y) − β

L(π,λ) =

π(y) − 1

π(y)log

π(y)R(x,y) + λ

y∈Y

y∈Y

y∈Y

where λ ∈ R is the Lagrange multiplier enforcing the normalization constraint. Taking the derivative with respect to π(y) and setting it to zero:

∂L ∂π(y)

π(y) q(y)

+ 1 − βR(x,y) + λ = 0.

= log

Solving for π(y) gives:

π(y) = q(y) · exp(βR(x,y) − λ − 1). Letting the normalization constant be:

q(y′) · exp(βR(x,y′)),

##### Z =

y′∈Y

we absorb constants into Z and write:

q(y | x) · exp(βR(x,y)) Z

πθ(y | x) =

.

Both derivations recover the same exponentially tilted distribution that emphasizes high-reward completions relative to the base model. In the hard-constrained view, β is a Lagrange multiplier tuned to meet the target reward ρ; in the soft-regularized view, β sets the strength of the trade-off between reward and divergence. This completes the constructive proof of Prop. C.4.

| |
|---|

Notably, by standard convex duality, this solution also arises as the optimizer of the entropy-regularized problem maxπ≪q Eπ R(x,y) − β1 KL π ∥q , which softens the constraint into a penalty. Thus, RLVR can be interpreted either as a hard projection onto the closest distribution achieving the reward target, or as a soft trade-off that balances expected reward with closeness to the base model. Similar exponential tilting policy improvement oracles have been analyzed in the context of KL-regularized contextual bandits and RLHF (Zhao et al., 2024), though their focus is on sample complexity under coverage.

KL-Free Limit. A relevant special case is the KL-free limit, where explicit KL regularization is removed (β → ∞) (Wei et al., 2023; Yu et al., 2025; Luo et al., 2025a; Yue et al., 2025b). In this regime, RLVR simplifies to a hard-filtered projection onto reward-maximizing completions.

Corollary C.5 (KL-Free Projection). In the limit β → ∞, the RLVR update converges to the renormalized restriction of the base model to the correct completion set:

q(y | x)1{y ∈ C} y′∈C q(y′ | x)

πβ(y | x) =

lim

.

β→∞

1Formally, absolute continuity π ≪ q ensures that the KL divergence KL(π ∥ q) is finite. If π assigns positive mass to any output that q assigns zero probability, the divergence becomes infinite. This condition also enforces support preservation: supp(π) ⊆ supp(q).

Proof. Since R(x,y) ∈ {0,1}, we have

exp βR(x,y) =

eβ if R(x,y) = 1, 1 if R(x,y) = 0.

Thus, the RLVR distribution becomes

q(y | x) exp βR(x,y) Zβ(x)

πβ(y | x) =

ni =

q(y | x) eβ1{R(x,y) = 1} + 1{R(x,y) = 0} Zβ(x)

,

where

q(y′ | x) +

q(y′ | x).

Zβ(x) = eβ

y′:R(x,y′)=1

y′:R(x,y′)=0

As β → ∞, the term with eβ dominates whenever there exists at least one y with R(x,y) = 1. Thus

Similarly, in the numerator, we have

q(y′ | x).

Zβ(x) ≈ eβ

y′∈C

q(y | x) exp βR(x,y) =

q(y | x)eβ if y ∈ C, q(y | x) otherwise.

Dividing by Zβ(x) and taking β → ∞, the probabilities assigned to y with R(x,y) = 0 vanish:

 

q(y | x)eβ eβ y′∈C q(y′ | x)

q(y | x) y′∈C q(y′ | x)

if y ∈ C,

=

πβ(y | x) ≈



0 otherwise.

Thus we obtain

q(y | x)1{y ∈ C} y′∈C q(y′ | x)

πβ(y | x) =

lim

,

β→∞

| |
|---|

Together, Prop. C.4 and Cor. C.5 illustrate a continuum of RLVR behaviors—from softly regularized reweighting (small β) to sharply constrained filtering (large β). Even in the KL-free limit, updates remain fundamentally anchored to the base model’s distribution, preserving relative probabilities within the reward-consistent subset. Consequently, while this projection ensures stable, efficient updates, it inherently limits RLVR’s exploratory capacity. As established in Thm. C.1, RLVR remains confined to the initial support of the base model unless explicit mechanisms introduce meaningful probability mass to new regions. Thus, the variational interpretation clarifies RLVR’s strengths in improving precision and efficiency within existing capabilities, alongside its limitations in fundamentally expanding model reasoning.

#### C.3. Entropy–Reward Trade-off: Precision at the Cost of Answer Diversity

Another structural property of RLVR is its tendency to systematically reduce the entropy of the answer distribution. This behavior arises naturally from reward optimization, which statistically favors sharper distributions concentrated on highreward completions. While such entropy reduction is beneficial in domains like board games or math—where precision is paramount—it may also suppress valuable diversity in contexts that benefit from broader coverage or multiple valid outputs, such as story or dialogue generation (Chen et al., 2023) and coding copilots (Peng et al., 2023).

Theorem C.6 (Entropy Reduction and Precision–Coverage Trade-off). Assume a finite output space Y and define the Shannon entropy of a distribution as H[p] := − y∈Y p(y | x)log p(y | x). Then the following statements hold:

- (a) Entropy reduction. Any RLVR update πθ satisfies H[πθ] ≤ H[q],

with equality only if the reward is constant on the support of q.

- (b) Trade-off with coverage. Lower entropy increases sampling precision for small budgets, but for large k, reduces the diversity of explored outputs—potentially missing alternative correct completions.

Proof. (a) Entropy reduction. Consider the exponentially tilted distribution

q(y | x)exp(βR(x,y)) Z

, with Z =

πθ(y | x) =

By standard properties of KL divergence,

q(y | x)exp(βR(x,y)).

y∈Y

KL(πθ∥q) =

y

πθ(y | x) q(y | x) ≥ 0.

πθ(y | x)log

Rearranging gives

H[πθ] = H[q] − KL(πθ∥q) ≤ H[q]. Thus, any such RLVR update decreases entropy relative to the base distribution, unless the reward is constant (in which case πθ = q). (b) Trade-off with diversity at different sampling budgets. The RLVR-trained policy sharpens the probability mass around high-reward completions. Explicitly,

##### πθ(y | x) ∝ q(y | x)exp(βR(x,y)), where β > 0 controls concentration.

- • Small sampling budgets (k = 1): The increased probability on high-reward outputs generally improves single-shot success rates. Formally,

pass@1π

θ

(x) =

y:R(x,y)=1

πθ(y | x) >

y:R(x,y)=1

q(y | x) = pass@1q(x),

provided the reweighting boosts correct completions relative to incorrect ones.

- • Large sampling budgets (k ≫ 1): However, reduced entropy leads to concentration on fewer modes. As β grows, πθ may collapse onto a narrow subset of correct completions, neglecting other valid solutions accessible under the more dispersed q. Thus,

limsup

k→∞

pass@kπ

θ

(x) < limsup

k→∞

pass@kq(x),

under typical conditions of entropy reduction and selective mass shifting.

- • Loss of tail coverage: In particular, if there exist rare but correct completions that have small mass under q but are further downweighted (or eliminated) by the tilting, then the total mass on correct completions can decrease:

##### πθ(C) < q(C), C = {y : R(x,y) = 1}.

This restricts the long-run probability of recovering diverse solutions via large k sampling.

Conclusion. This establishes a trade-off: RLVR improves sampling efficiency by concentrating probability on high-reward outputs (increasing pass@1), but this comes at the cost of reduced entropy and narrower exploration of the solution space (potentially lowering pass@k for large k). Empirical studies confirm this phenomenon in settings like code generation and symbolic reasoning, where many semantically distinct correct completions exist.

| |
|---|

This trade-off underpins RLVR’s empirical strengths in tasks with narrowly defined optimal solutions, such as mathematical proofs or tactical game endgames (where precision is paramount), while also emphasizing the need for explicit diversity mechanisms in more open-ended domains, such as code generation, creative writing (Feizi et al., 2023; Ding et al., 2024), or brainstorming (Chang & Li, 2025). Importantly, entropy reduction is not inherently undesirable: when a task admits a unique correct solution, lower answer-level entropy simply reflects desirable convergence. Importantly, even in multi-solution domains, concentrating mass on a narrower set may still be desirable under constrained compute budgets. However, our results show that entropy reduction can still lead to empirical support shrinkage even in predominantly single-solution domains like math, where RLVR sometimes fails to recover valid completions still accessible to the more diverse base model. This highlights that entropy-induced narrowing is a general phenomenon, not limited to multi-solution tasks, underscoring the broader need for explicit exploration or diversity-promoting strategies.

#### C.4. Estimating the Sampling Threshold ϵ from pass@k

We provide a statistical analysis of the threshold ϵ in the pass@k sampling. Suppose we sample k times from a model π(· | x), and let y∗ ∈ C be a correct completion with unknown probability p = π(y∗ | x). If y∗ is not observed in any of those k samples, we can upper bound p using the following argument.

The probability of not sampling y∗ in a single trial is 1 − p, so the probability of missing it in all k independent trials is (1 − p)k. To ensure this event occurs with probability at most ζ, we solve:

(1 − p)k ≤ ζ. Taking logarithms of both sides:

k · log(1 − p) ≤ log ζ. Using the inequality log(1 − p) ≤ −p for p ∈ (0,1), we get:

−log ζ k

k · (−p) ≥ log ζ ⇒ p ≤

.

Consequently, if the correct completion y∗ is not observed in k samples, then with confidence 1 − ζ, its probability satisfies:

|π(y∗ | x) ≤<br><br>−log ζ k<br><br>.|
|---|

Example. If k = 8192 in the math reasoning tasks and we desire 95% confidence (i.e., ζ = 0.05), then

−log(0.05) 8192 ≈

2.996 8192 ≈ 3.66 × 10−4.

π(y∗ | x) ≤

