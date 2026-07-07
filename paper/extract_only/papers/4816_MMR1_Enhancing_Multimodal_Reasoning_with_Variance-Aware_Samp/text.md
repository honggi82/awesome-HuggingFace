# arXiv:2509.21268v1[cs.CV]25Sep2025

## MMR1: ENHANCING MULTIMODAL REASONING WITH VARIANCE-AWARE SAMPLING AND OPEN RESOURCES

Sicong Leng1,2,∗ Jing Wang1,∗ Jiaxi Li3,∗ Hao Zhang2,∗ Zhiqiang Hu2 Boqiang Zhang2 Yuming Jiang2 Hang Zhang2 Xin Li2 Lidong Bing2 Deli Zhao2 Wei Lu1 Yu Rong2 Aixin Sun1,† Shijian Lu1,†

1Nanyang Technological University 2DAMO Academy, Alibaba Group 3Singapore University of Technology and Design

∗Equal Contributions †Correspondence

ABSTRACT

Large multimodal reasoning models have achieved rapid progress, but their advancement is constrained by two major limitations: the absence of open, large-scale, high-quality long chain-of-thought (CoT) data, and the instability of reinforcement learning (RL) algorithms in post-training. Group Relative Policy Optimization (GRPO), the standard framework for RL fine-tuning, is prone to gradient vanishing when reward variance is low, which weakens optimization signals and impairs convergence. This work makes three contributions: (1) We propose Variance-Aware Sampling (VAS), a data selection strategy guided by Variance Promotion Score (VPS) that combines outcome variance and trajectory diversity to promote reward variance and stabilize policy optimization. (2) We release large-scale, carefully curated resources containing ∼1.6M long CoT cold-start data and ∼15k RL QA pairs, designed to ensure quality, difficulty, and diversity, along with a fully reproducible end-to-end training codebase. (3) We open-source a family of multimodal reasoning models in multiple scales, establishing standardized baselines for the community. Experiments across mathematical reasoning benchmarks demonstrate the effectiveness of both the curated data and the proposed VAS. Comprehensive ablation studies and analyses provide further insight into the contributions of each component. In addition, we theoretically establish that reward variance lowerbounds the expected policy gradient magnitude, with VAS serving as a practical mechanism to realize this guarantee. Our code, data, and checkpoints are available at https://github.com/LengSicong/MMR1.

1 INTRODUCTION

Recent advances in large language and multimodal reasoning models have markedly improved performance on complex tasks such as mathematics, science, and open-domain problem solving. Reinforcement learning (RL) plays a central role in these developments by optimizing models with process- or outcome-based rewards (Lightman et al., 2024; Wang et al., 2024c; Li et al., 2025b). Group Relative Policy Optimization (GRPO; Shao et al. (2024)) has emerged as a widely adopted RL framework due to its efficiency and scalability, and has been successfully applied to both language models (Guo et al., 2025) and multimodal models (Meng et al., 2025a; Wang et al., 2025b; Tan et al., 2025; Leng et al., 2025). However, GRPO is inherently susceptible to gradient vanishing: when sampled rewards have low variance, relative advantages collapse toward zero, weakening optimization signals and destabilizing training (Razin et al., 2024; 2025). This issue persists across both unimodal and multimodal contexts, posing a fundamental challenge to effective RL optimization.

In parallel, the progress of multimodal reasoning research is influenced by the limited availability of open, large-scale, high-quality long chain-of-thought (CoT) data. Compared with text-only reasoning, where multiple datasets are publicly accessible (Guha et al., 2025; Muennighoff et al., 2025), multimodal training often relies on more restricted resources, which may constrain reproducibility and

random sampler

training batch mix

rewards calculation

policy model

rollouts

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

update policy

weighted sampler

training data

[Figure 10]

[Figure 11]

sync policy & recalculate weights

[Figure 12]

GRPO Training

Data Sampling

rollouts VPS calculation

policy model

Figure 1: Overview of the Variance-Aware Sampling (VAS) framework.

further development. Recent studies have made progress by exploring heuristic data curation (Meng et al., 2025a; Huang et al., 2025c; Chen et al., 2025a), reward design modifications (Tan et al., 2025; Shen et al., 2025), and training adjustments (Deng et al., 2025; Zhang et al., 2025b). While these approaches improve downstream performance, challenges related to stable GRPO optimization and the broader availability of curated multimodal reasoning data remain underexplored.

In this work, we introduce Variance-Aware Sampling (VAS), a dynamic data selection strategy designed to mitigate gradient vanishing in GRPO-based training for multimodal reasoning models. Our approach is grounded in the theoretical insight that reward variance provides a lower bound on the expected policy gradient magnitude. Increasing reward variance, therefore, offers a principled means to stabilize training and strengthen policy optimization. Specifically, VAS employs the Variance Promotion Score (VPS), which evaluates each prompt’s potential to induce reward variance. VPS consists of two complementary components: the Outcome Variance Score (OVS), which favors prompts yielding a balanced mix of correct and incorrect responses to maximize expected reward variance, and the Trajectory Diversity Score (TDS), which encourages diversity among reasoning trajectories, thereby raising the lower bound of variance and sustaining informative gradient signals even under sparse or noisy correctness feedback. Depicted in Figure 1, VAS constructs each training batch from two subsets: one sampled with probabilities proportional to VPS, emphasizing prompts with higher potential to induce reward variance, and another drawn uniformly at random to maintain broad data coverage. This design aims to balance targeted promotion of reward variance with general exposure to the training distribution. By introducing outcome- and trajectory-level variability into GRPO’s group-based comparisons, VAS reduces the risk of weak gradients and contributes to more stable optimization. A theoretical analysis is presented in §4.

We validate VAS on a range of multimodal mathematical and logical reasoning benchmarks. Experiments show that VAS improves convergence, stability, and downstream performance. Ablation studies further demonstrate that OVS and TDS contribute complementary benefits: OVS enhances expected reward variance by balancing outcomes, while TDS increases trajectory diversity to support more consistent gradient updates. Beyond methodology, we curate and release large-scale datasets for both supervised fine-tuning and RL. The supervised dataset (∼1.6M) emphasizes long chain-of-thought reasoning paired with verified short answers, while the RL dataset (∼15k) is constructed to capture diverse levels of difficulty and domain coverage. Both datasets are curated with explicit attention to quality, difficulty, and diversity, ensuring their value for multimodal reasoning research. Together with these resources, we provide a reproducible codebase and open models at multiple scales, offering standardized baselines for future research.

- 2 RELATED WORK

Building on the success of rule-based RL (Guo et al., 2025; Kimi, 2025), recent multimodal work explores RL with verifiable rewards, typically following a pipeline that conducts optional SFT activation then applies RL (Schulman et al., 2017; Ahmadian et al., 2024), such as GRPO (Shao

- et al., 2024), with fine-grained training recipes. In the multi-modal domain, various approaches

refine this pipeline through specific reward design (Tan et al., 2025; Shen et al., 2025), sample diversification (Meng et al., 2025a; Wang et al., 2025b), hyperparameter tuning (Huang et al., 2025c; Tan et al., 2025; Yang et al., 2025a), and advanced RL strategies (Peng et al., 2025; Deng et al., 2025; Zhang et al., 2025b). Nevertheless, they often overlook the gradient vanishing problem inherent in GRPO-based training, resulting in unstable optimization and slow convergence (Razin et al., 2025). Some studies attempt to alleviate it by filtering samples with moderate pass rates (Wang et al., 2025b; Meng et al., 2025a), yet they remain largely heuristic and lack comprehensive experimental validation or theoretical grounding.

Recent studies have examined gradient vanishing in a principled manner, analyzing how training objectives degrade under reward sparsity, variance reduction, and optimization dynamics (Liu et al., 2025; Hu et al., 2025; Vassoyan et al., 2025; Zhou et al., 2025). A range of remedies has been proposed, including reward rescaling (Li et al., 2024b; Huang et al., 2025a), entropy regularization (Liu et al.,

- 2024), and improved sample selection (Wang et al., 2024d; Zhang et al., 2025c; Li et al., 2025a), which primarily operate by adjusting RL algorithms or reward mechanisms. In contrast, our work takes an orthogonal perspective by mitigating gradient vanishing through data sampling during training. Supported by both theoretical grounding and extensive empirical validation, our approach complements existing GRPO variants (Wang et al., 2025c; Liu et al., 2024; Hu et al., 2025; Yue et al.,
- 2025) and can be naturally combined with them to further improve training stability and effectiveness in reasoning.

- 3 VARIANCE-AWARE SAMPLING FRAMEWORK

This section details the Variance-Aware Sampling (VAS) framework. §3.1 defines the Variance Promotion Score (VPS), comprising Outcome Variance Score (OVS) and Trajectory Diversity Score (TDS), which guides dynamic data sampling. §3.2 then outlines the sampler implementation, including VPS updates and sample selection during training.

- 3.1 VARIANCE PROMOTION SCORE

Let x be a prompt with ground-truth answer y¯ from the training set. In GRPO framework, the model generates N responses {yi}Ni=1 for each x. A task-specific verifier V(x,yi,y¯) ∈ {0,1} evaluates each response, returning 1 if yi matches y¯ and 0 otherwise. The pass rate for x is then defined as:

P(x) =

1 N

N

i=1

V(x,yi,y¯).

Inspired by Foster & Foerster (2025), we directly calculate the Outcome Variance Score (OVS) as: OVS(x) = P(x)(1 − P(x)),

which corresponds to the Bernoulli variance of correctness across responses. It is maximized at P(x) = 0.5, where correct and incorrect outputs are balanced. We further define the Trajectory Diversity Score (TDS) to characterize variability in reasoning processes. Let Diversity({yi}Ni=1) be a diversity function over sequences (e.g., inverse self-BLEU or distinct-n):

TDS(x) = Diversity({yi}Ni=1),

where a higher value reflects greater diversity among sampled trajectories. The overall Variance Promotion Score (VPS) is computed as a weighted combination of OVS and TDS:

VPS(x) = α · OVS(x) + β · TDS(x),

where α,β > 0 balance their contributions. OVS increases the expected reward variance, while TDS provides a lower bound by encouraging trajectory diversity. Together, they are intended to strengthen the magnitude and consistency of gradient signals in GRPO training.

- 3.2 DYNAMIC SAMPLER

The dynamic sampler prioritizes prompts with higher VPS, which are expected to induce greater reward variance during training. At each sampling step, the training batch is constructed from two

subsets: one from a weighted sampler based on VPS and another from a uniform random sampler. A mix ratio hyperparameter λ ∈ [0,1] controls the proportion of samples drawn from each source, with λ specifying the fraction of the batch selected from the weighted sampler. VPS scores are periodically updated to reflect changes in the policy. After every Tupdate steps, N responses are resampled for each prompt, and OVS and TDS are recomputed. The update interval Tupdate trades off computational cost against adaptation speed. Algorithm 1 summarizes the VAS procedure.

Algorithm 1 Variance-Aware Sampling (VAS) for GRPO Training

Require: Dataset D; batch size B; rollouts per prompt N; VPS update interval Tupdate; mix ratio λ∈[0, 1]

- 1: Initialize policy parameters θ
- 2: for each prompt x ∈ D do ▷ Initial VPS estimation
- 3: Sample N rollouts {yi}Ni=1 from πθ(· | x)
- 4: Compute pass rate P(x), OVS(x), TDS(x), and VPS(x)
- 5: end for
- 6: for training step t = 1, . . . , T do
- 7: if t mod Tupdate = 0 then ▷ Periodic VPS refresh
- 8: for each prompt x ∈ D do
- 9: Sample N rollouts {yi}Ni=1; update P(x), OVS(x), TDS(x), VPS(x)
- 10: end for
- 11: end if
- 12: Bw ←⌊λB⌋, Br←B − Bw
- 13: Sample Bw prompts from D with replacement proportional to VPS(·)
- 14: Sample Br prompts from D uniformly at random
- 15: B← union of the two sets ▷ Construct training batch
- 16: for each x ∈ B do
- 17: Sample N rollouts {yi}Ni=1 from πθ(· | x)
- 18: Compute GRPO loss LGRPO(x, {yi})
- 19: end for
- 20: Update θ ← θ − η∇θ x∈B LGRPO(x, {yi})
- 21: end for

- 4 THEORY

This section formalizes the intuition that prompts with higher reward variance yield more informative policy-gradient updates. We first establish a Variance–Progress Theorem for the vanilla REINFORCE algorithm (Williams, 1992), showing that expected improvement is linearly lower-bounded by the reward variance of the prompt. We then present a two-level decomposition of reward variance that directly motivates the Outcome Variance Score and Trajectory Diversity Score in our method. Finally, we extend the analysis to GRPO, with complete proofs and derivations provided in Appendix A.

- 4.1 PRELIMINARIES

Let x ∈ X be a prompt and y ∈ Y a response drawn from the policy πθ(y | x). A learned reward model R : X × Y → [−1,1] assigns a scalar reward. The optimization objective is defined as:

θ(·|x) [R(x,y)], (1)

J(θ) = Ex∼D Ey∼π

where D is the prompt distribution. Here, we omit the KL regularization, as it only rescales constants and does not affect the variance argument.

Score-function identity. For a fixed prompt x, the gradient can be expressed as:

R(x,y)g(x,y) , g(x,y) = ∇θ log πθ(y | x), (2) where g(x,y) is the score function and Ey[g(x,y)] = 0. Baselines and variance. Subtracting a prompt-dependent baseline b(x) keeps the estimator unbiased:

##### ∇θJx(θ) = Ey∼π

θ

##### G(x) = Ey[g(x,y)(R(x,y) − b(x))], E[G(x)] = ∇θJx(θ). (3)

The variance is minimized when b⋆(x) = Ey[R(x,y)]. Under this choice, the variance factorizes as (Lemma A.1):

##### Var[G(x)] = Vary[R(x,y)] Γθ(x), (4)

where Γθ(x) = Ey∥g(x,y)∥2 is a Fisher-information term depending on the policy but not on the rewards. Proposition A.2 shows Γθ(x) is bounded above and below by model-dependent constants. Thus, the reward variance is the only prompt-dependent factor controlling the dispersion of stochastic gradients.

- 4.2 A VARIANCE–PROGRESS THEOREM FOR REINFORCE

Consider a single gradient step θ+ = θ + ηG(x) on prompt x, with learning rate η > 0.

Assumption 1 (Smoothness and gradient lower bound). For each prompt x: (i) Jx(θ) is twice differentiable with ∥∇2θJx(θ)∥ ≤ L for some L > 0; (ii) There exists cmin > 0 such that ∥∇θJx(θ)∥2 ≥ cmin Vary[R(x,y)].

The second condition links reward variance to the squared gradient norm. It follows from the positive definiteness of the Fisher information matrix under mild regularity conditions.

Theorem 1 (Variance–Progress). Let Assumption 1 hold and use the optimal baseline b⋆(x). Then for any step size 0 < η ≤ c

min 4LΓθ(x), the expected one-step gain satisfies E Jx(θ+) − Jx(θ) ≥

η cmin 4

Vary[R(x,y)]. (5)

Intuition

Prompts with higher reward variance provide stronger gradient signals. If all rollouts have similar rewards, gradients vanish; if outcomes or reasoning paths vary, the variance ensures progress per update.

Sketch of proof. A second-order Taylor expansion yields Jx(θ+) = Jx(θ) + ⟨∇Jx,G(x)⟩ +

- 1

- 2G⊤HG. Taking expectations and using unbiasedness gives a linear term η∥∇Jx∥2. The quadratic

remainder is bounded by 12η2LE∥G∥2. Substituting the variance factorization and the gradient lower bound, and restricting η as stated, ensures the remainder is at most half the linear term, giving the

result. The full details of the proof are provided in Appendix A.2.

- 4.3 VARIANCE DECOMPOSITION AND CONNECTION TO OVS/TDS

For binary rewards, the variance admits a natural two-level decomposition. Write each rollout y = (ycot,yans) as a reasoning chain and its final answer, and let the verifier assign R(x,y) = 1correct(y

ans). Define Z = φ(ycot) as a representation of the reasoning chain and pZ(x) = Pr(R = 1 | Z). The law of total variance gives

Vary[R(x,y)] = EZ[pZ(x)(1 − pZ(x))]

intra-trajectory

+VarZ[pZ(x)]

inter-trajectory

. (6)

The first term corresponds to variability in correctness given a fixed reasoning path, and is estimated by pˆ(x)(1 − pˆ(x)). This motivates the Outcome Variance Score (OVS). The second term measures variation across reasoning paths, which can be lower-bounded by diversity metrics such as normalized edit distance or self-BLEU dispersion (detailed in Appendix A.3). This motivates the Trajectory Diversity Score (TDS). Together, OVS and TDS provide complementary mechanisms to raise reward variance, directly connecting the theory to our method in §3.

- 4.4 FROM REINFORCE TO GRPO

GRPO extends REINFORCE by normalizing rewards within each group of rollouts. For prompt x, the group mean reward r¯(x) serves as a baseline, and the sample standard deviation s(x) whitens centered rewards:

R(x,yi) − r¯(x) s(x) + δ

R˜(x,yi) =

.

Table 1: Cold-start Data Statistics. Domain # Samples Datasets

Math 1.6M MM-MathInstruct (Wang et al., 2025d), MathV360K (Shi et al., 2024), MultiMath (Peng et al., 2024), MATH (Hendrycks et al., 2021), MathVision (Wang et al., 2024a), CLEVR-Math (Lindström & Abraham, 2022), IconQA (Lu et al., 2021b), MAVIS-Instruct (Zhang et al., 2024b), RCoT (Deng et al., 2024), GeoQA+ (Cao & Xiao, 2022), Geometry3K (Lu et al., 2021a), GeomVerse (Kazemi et al., 2024), Self-collected data

Science 13K ScienceQA (Lu et al., 2022a), AI2D (Kembhavi et al., 2016), CLEVR (Johnson et al., 2017), Self-collected data Chart-Figure 8K ChartQA (Masry et al., 2022b), SPIQA (Pramanick et al., 2024), DVQA (Kafle et al., 2018), PlotQA (Methani et al., 2020)

Doc-Table 8K TableMWP (Lu et al., 2022b), InfoVQA (Mathew et al., 2022), DocVQA (Mathew et al., 2021), WikiTableQuestions (Pasupat & Liang, 2015), VisualMRC (Tanaka et al., 2021)

General 8K Sherlock (*Hessel et al., 2022), A-OKVQA (Schwenk et al., 2022), PISC (Li et al., 2017), GQA (Hudson & Manning, 2019)

The gradient estimator then multiplies R˜ by the importance ratio rθ(yi | x), optionally clipped to 1 ± ε to control KL divergence.

Without clipping, the estimator remains unbiased, and Theorem 1 applies with a rescaled learning rate bound reflecting the whitening factor. With clipping, the estimator acquires a bias of order O(ε), which reduces but does not eliminate the lower bound. Thus, under both settings, prompts with higher reward variance continue to guarantee larger provable minimum improvements per update. Details and finite-sample corrections are provided in Appendix A.4.

- 5 DATA CURATION

Following prior work (Tan et al., 2025; Wang et al., 2025b; Meng et al., 2025a), our training pipeline includes two stages: a supervised cold-start stage followed by a reinforcement learning stage using GRPO, with each stage supported by meticulously curated datasets.

- 5.1 COLD-START DATA

This stage utilizes long chain-of-thought (CoT) data for supervised fine-tuning. As summarized in Table 1, we collect question-answer pairs from diverse instruction-tuning corpora, which primarily feature short answers or short CoTs. To address the limited coverage, particularly in science, in existing multimodal datasets, we supplement with publicly available practice problems and exams from biology, chemistry, geography, and physics. The final dataset spans five domain categories: Math, General, Chart-Figure, Doc-Table, and Science. To ensure balanced quality and difficulty, we apply filtering based on response correctness. For each question, multiple responses are generated via Qwen2.5-VL-72B (Bai et al., 2025) and verified by GPT-4o (OpenAI, 2024). Samples are categorized by pass rate: easy (≥ 0.8), hard (≤ 0.2), or medium (otherwise). Only medium and hard cases are retained for cold-start. Long CoT annotations are then produced using Gemini 2.5 Pro/Flash (DeepMind, 2025), which generates multi-step rationales followed by final answers. Annotations are preserved only when the final answer matches ground truth as validated by GPT-4o.

- 5.2 RL DATA

This stage employs prompts annotated with concise, verifiable short answers suitable for reward modeling. Specifically, we adopt GPT-4o to extract and rephrase open-form short answers from the original CoT annotations. The RL dataset integrates two complementary components. First, we select 8k math problems from the cold-start stage, retaining only hard-level items with low pass rates to ensure challenging supervision. These problems are further categorized into fine-grained

- Table 2: Comparison of MMR1 with other MLLMs on mathematics-related benchmarks. All models are reevaluated under identical conditions for fairness; values in parentheses are taken from the original papers. The bold and underline highlight the best and second-best scores, respectively.

Model Size MathVerse MathVista MathVision LogicVista ChartQA Avg General-Purpose Models

Qwen2.5-VL 7B 50.4 (49.2) 69.3 (68.2) 28.7 (25.1) 44.0 82.4 55.0 InternVL2.5 8B 40.0 (39.5) 61.4 (64.4) 19.9 (19.7) 37.7 (36.0) 73.4 46.5 InternVL3 8B 49.4 (39.8) 68.5 (71.6) 30.0 (29.3) 41.3 (44.1) 81.3 54.1 LLaVA-OV 7B 33.6 (26.2) 56.4 (63.2) 15.9 30.6 65.0 40.3

Reasoning-Oriented Models

MM-Eureka 8B 52.3 (50.3) 73.4 (73.0) 29.4 (26.9) 47.1 82.7 57.0 R1-VL 7B 41.3 (40.0) 61.5 (63.5) 23.0 (24.7) 36.3 76.3 47.7 R1-OneVision 7B 44.0 (46.4) 60.3 (64.1) 22.0 (29.9) 40.0 72.5 47.8 OpenVLThinker 7B 48.1 (47.9) 70.6 (70.2) 22.0 (25.3) 41.0 81.0 52.5 VL-Rethinker 7B 54.6 (54.2) 73.7 (74.9) 30.1 (32.3) 45.7 83.5 57.5 Vision-R1 7B 51.9 (52.4) 72.1 (73.5) – 44.7 82.7 – ThinkLite-VL 7B 51.3 (50.7) 72.5 (75.1) 27.5 44.3 83.1 55.7 VL-Cogito 7B 54.3 74.8 30.7 48.9 83.4 58.2

MMR1 3B 47.9 67.1 25.4 42.0 81.2 52.7 MMR1 7B 55.4 72.0 31.8 48.9 83.7 58.4

types with GPT-4o (detailed in Appendix B) and uniformly sampled to ensure balanced coverage across categories. Second, we add 7k logical reasoning problems from Raven (Zhang et al., 2019), MM-IQ (Cai et al., 2025), and EasyArc (Unsal & Akkus, 2025), curated to incentivize general reasoning ability beyond math. Together, these components form a 15k RL dataset emphasizing difficulty, diversity, and balanced coverage across mathematical and logical reasoning tasks.

- 6 EXPERIMENTS

- 6.1 EXPERIMENTAL SETUPS

Implementations. In the cold-start stage, we fine-tune Qwen2.5-VL-Instruct (Bai et al., 2025) with curated long CoT data (§5.1) using the LLaMA-Factory framework (Zheng et al., 2024). Training runs for 5 epochs with AdamW, a cosine schedule, an initial learning rate of 1 × 10−5, and a 0.1 warm-up ratio. The checkpoint with the best validation score is retained. This checkpoint initializes the policy for RL training, implemented with the EasyR1 codebase (Zheng et al., 2025). VAS (Algorithm 1) is set to N = 32, λ = 0.5, α = 0.8, β = 0.2, and Tupdate = 351. Additional hyper-parameters for both cold-start and RL training are detailed in Appendix C.

Benchmarks. To evaluate MMR1, we adopt five widely used and challenging benchmarks focusing on mathematical and logical reasoning: MathVerse (Zhang et al., 2024a), MathVista (Lu et al., 2024), MathVision (Wang et al., 2024b), LogicVista (Xiao et al., 2024), and ChartQA (Masry et al., 2022a). These benchmarks collectively assess diverse aspects of problem-solving, including complex multistep mathematics, visual reasoning, logical deduction, and chart-based understanding.

Baselines. In this work, we compare MMR1 against a broad set of MLLMs, covering both generalpurpose and reasoning-oriented designs of comparable model size. General-purpose MLLMs: Qwen2.5-VL-Instruct-7B (Bai et al., 2025), InternVL2.5-8B (Chen et al., 2025b), InternVL3-8B (Zhu

- et al., 2025), and LLaVA-OneVision-7B (LLaVA-OV; Li et al. (2024a)), representing the recent

1Due to the resource constraint, experiments in ablation studies and analysis adopt N = 8, λ = 0.5, α = 0.5, β = 0.5, and Tupdate = 28 unless otherwise specified.

Table 3: The effect of Cold-start SFT and Variance-Aware Sampling (VAS).

Model MathVerse MathVista MathVision LogicVista ChartQA Avg Qwen2.5-VL-3B 40.4 63.5 24.3 38.4 76.8 48.7 +Cold-start 42.1 58.0 25.2 39.5 78.5 48.7 +GRPO 46.2 65.6 24.7 42.4 79.9 51.8 +VAS (MMR1) 47.9 67.1 25.4 43.1 81.2 52.9

state-of-the-art general-purpose MLLMs. Reasoning-oriented MLLMs: VL-Cogito-7B (Yuan et al., 2025), MM-Eureka-8B (Meng et al., 2025b), R1-VL-7B (Zhang et al., 2025a), R1-OneVision-

- 7B (Yang et al., 2025b), OpenVLThinker-7B (Deng et al., 2025), Vision-R1-7B (Huang et al., 2025b), VL-Rethinker (Wang et al., 2025a), and ThinkLite-VL-7B (Wang et al., 2025e).

Evaluation. We adopt a unified prompt across all evaluations, requiring models to enclose final answers in “\box{}” (full prompt in Appendix D). Inference is performed using vLLM (Kwon et al.,

- 2023) for efficient generation. For benchmarks with official protocols (e.g., MathVision, MMMU), we strictly follow the original procedures. For others, mathematical questions are assessed with Math-Verify (Kydlíˇcek, 2025) and MathRuler (hiyouga, 2025), while non-mathematical ones use exact matching. To ensure robustness, we further (1) select the most semantically similar option when multiple-choice answers do not exactly match any candidate, and (2) employ GPT-4o (OpenAI,
- 2024) as an auxiliary judge for open-ended questions where exact matching or extraction fails.

- 6.2 MAIN RESULTS

As shown in Table 2, our 7B model achieves state-of-the-art performance among reasoning-oriented MLLMs, reaching an average score of 58.4, the highest across all evaluated models. It ranks first on most benchmarks, including MathVerse (55.4), MathVision (31.8), LogicVista (48.9), and ChartQA (83.7), while also delivering competitive results on MathVista (72.0). Compared to other general models with similar scales, our approach consistently yields superior results.

In addition, the 3B variant of our model demonstrates strong competitiveness with an average of 52.7. Despite its smaller scale, it matches or surpasses several 7B models (e.g., R1-VL at 47.7 and R1-OneVision at 47.8), underscoring the efficiency of our framework in resource-constrained settings.

These results highlight the complementary contributions of our pipeline: carefully curated long CoT supervision for cold-start initialization, reinforcement learning to incentivize deeper reasoning, and Variance-Aware Sampling (VAS) to stabilize optimization. Together, they enable consistent improvements in reasoning performance across both small- and large-scale models.

- 6.3 EFFECT OF COLD-START AND VAS

- Table 3 reports the effect of cold-start supervision and the proposed VAS strategy on Qwen2.5-VL-3B across several mathematics-related benchmarks. Beginning with the base model, cold-start finetuning on curated long-form CoT data yields consistent improvements, particularly on MathVerse and ChartQA. Building upon this, GRPO further enhances performance, surpassing the cold-start baseline and demonstrating that reinforcement learning effectively incentivizes exploration and strengthens reasoning. The introduction of VAS (MMR1) achieves the highest overall scores, delivering notable gains on MathVerse, MathVista, and LogicVista. Collectively, these results highlight the complementary roles of different components: (1) cold-start supervision provides a strong initialization by imitating high-quality reasoning trajectories; (2) reinforcement learning emphasizes exploratory behavior to further incentivize reasoning; and (3) VAS ensures stable, variance-aware training, thereby leading to more robust and effective learning outcomes.

Table 4: The effect of Variance-Aware Sampling hyper-parameters.

Ablated Param. Value MathVerse MathVista MathVision LogicVista ChartQA Avg Mixture ratio 0.2 46.3 67.9 23.2 40.2 79.9 51.5

0.5 46.1 66.4 24.8 41.7 79.4 51.7

- 0.8 46.9 64.8 24.3 43.8 79.9 52.0

- 1.0 44.6 65.3 24.9 43.8 79.9 51.7

Update freq. 4 47.4 65.8 23.8 39.3 79.6 51.2

7 46.6 65.7 24.6 41.5 78.9 51.5 14 46.7 66.9 24.2 44.6 79.2 52.3 28 46.1 66.4 24.8 41.7 79.4 51.7 35 47.6 66.1 24.5 42.4 80.1 52.2 56 44.5 65.5 23.9 40.2 80.2 50.9

# rollout 8 46.1 66.4 24.8 41.7 79.4 51.7 16 45.9 65.0 24.5 41.1 79.8 51.3 32 46.7 65.1 25.0 42.2 78.9 51.6

VPS ratio (0.0, 1.0) 46.8 64.4 24.3 40.9 78.8 51.0 (OVS, TDS) (0.2, 0.8) 46.4 65.6 25.0 40.6 79.7 51.5

(0.5, 0.5) 46.1 66.4 24.8 41.7 79.4 51.7

- (0.8, 0.2) 46.9 66.8 23.7 45.1 79.0 52.3
- (1.0, 0.0) 46.5 65.6 24.5 39.7 79.6 51.2

- 6.4 EFFECT OF VAS HYPER-PARAMETERS

We further investigate the sensitivity of VAS to its key hyperparameters, i.e., the mixture ratio λ, VPS update frequency Tupdate, number of rollouts N, and the weighting between OVS and TDS in VPS computation. The results are summarized in Table 4.

Mixture ratio. The mixture ratio λ controls the balance between VPS-weighted sampling and uniform random sampling in Algorithm 1 (Lines 12–14). Performance is generally robust across settings, with λ = 0.5 yielding competitive and stable results. Extremely large ratios (e.g., λ = 1.0) reduce coverage of the overall dataset and lead to degraded performance, confirming the necessity of maintaining a balance between variance promotion and coverage.

Update frequency. The update interval Tupdate specifies the frequency at which VPS scores are refreshed. Short intervals (e.g., 4 or 7 steps) enhance adaptability to model dynamics but incur higher computational overhead. Moderate intervals (i.e., 14–35 steps) strike a favorable balance, consistently yielding robust performance. In contrast, excessively long intervals (e.g., 56 steps) result in outdated VPS estimates, thereby weakening gradient signals and degrading overall performance.

Number of rollouts. The rollout number N affects the accuracy of variance estimation for OVS and TDS. Increasing N from 8 to 16 provides marginal improvements by reducing sampling noise. However, further increases (e.g., 32) offer limited gains while introducing higher computational costs.

VPS weighting. The VPS ratio (α,β) balances outcome variance (OVS) and trajectory diversity (TDS). A balanced combination (e.g., α = 0.8, β = 0.2) consistently delivers strong results, whereas relying solely on one component leads to instability. Consistent with the analysis in §4, this shows that outcome variance and trajectory diversity play complementary roles: OVS captures correctness variability, while TDS safeguards a lower bound on variance when correctness signals are sparse.

Overall, the results demonstrate that VAS maintains stability across a wide range of hyperparameter settings, with moderate mixture ratios, appropriate update frequencies, and balanced VPS weighting yielding the most consistent performance improvements.

gradient norm

policy gradient clip fraction

validation accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
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
| | | | | |
| | | | | |

0.70

0.80

0.005

0.65

0.004

0.60

0.60

0.55

0.003

0.50

0.40

0.002

0.45

0.20

0.40

0.001

0 20 40 60

0 20 40 60

0 20 40 60

training step

training step

training step

mixture =1.0 mixture =0.5 baseline

- Figure 2: Training efficiency of Variance-Aware Sampling (VAS). The plots compare three settings: full VAS sampling (λ = 1.0, orange), mixed sampling with half VAS and half random (λ = 0.5, blue), and the vanilla baseline (purple). Left: Actor gradient norm, reflecting the magnitude of gradient signals during training. Middle: Policy gradient clip fraction, indicating the proportion of updates reaching the clipping boundary. Right: Validation accuracy, showing convergence speed and final performance.

- 6.5 EFFICIENCY OF VAS

- Figure 2 presents a comparative analysis of the training efficiency of VAS under different mixture ratios (orange line: λ = 1.0, blue line: λ = 0.5) against the vanilla random-shuffle baseline (purple line). The evaluation considers three key indicators: actor gradient norm, policy-gradient clipping fraction, and validation accuracy.

Gradient norm. The gradient norm reflects the overall magnitude of parameter updates. Models trained with VAS consistently exhibit higher gradient norms compared to the shuffle baseline, indicating more substantial and informative updates. This empirical finding is consistent with our theoretical analysis in §4, which demonstrates that prompts with higher reward variance produce gradients characterized by greater magnitude and more reliable signal strength.

Policy-gradient clip fraction. The clip fraction quantifies the frequency with which the policy update magnitude reaches the clipping threshold in GRPO. Within stable ranges, a higher clip fraction indicates more effective learning: the model performs substantial yet constrained updates, exploits the trust region more fully, and extracts stronger signals from each batch. As illustrated in Figure 2, VAS configurations achieve higher and more stable clip fractions compared to the baseline, highlighting their improved sample efficiency and more effective exploration of the policy space.

Validation accuracy. On the held-out validation set, VAS demonstrates consistent improvements in both convergence speed and final accuracy compared to the shuffle baseline. Employing full VAS sampling (λ = 1.0) yields the most rapid and stable performance gain, while the mixed configuration (λ = 0.5) also surpasses uniform sampling in convergence efficiency. Although the difference between λ = 1.0 and λ = 0.5 is not pronounced on this mathematics-focused set, it is worth noting that incorporating partial random sampling can, in principle, encourage broader data coverage and reduce the risk of oversampling a limited subset of prompts. This trade-off is expected to be more beneficial in domains characterized by greater content heterogeneity or less structured reward signals.

6.6 DYNAMICS OF VARIANCE PROMOTION SCORE

- Figure 3 summarizes how Variance Promotion Scores (VPS) evolve across update intervals (t → t+14), where the histograms show the marginal VPS distribution and transition matrices of VPS assignments between step t (rows) and step t+14 (columns)2.

Convergence of rankings. As training progresses, the mass in the transition matrices progressively concentrates along the diagonal, while the off–diagonal dispersion diminishes. This behavior reflects

2Tupdate is set to 14.

2000VPS Distribution (Step 0)

2000VPS Distribution (Step 14)

2000VPS Distribution (Step 28)

2000VPS Distribution (Step 42)

2000VPS Distribution (Step 56)

Freq

1000

1000

1000

1000

1000

VPS

VPS

VPS

VPS

VPS

Weight Transition: Step 0 14

Weight Transition: Step 14 28

Weight Transition: Step 28 42

Weight Transition: Step 42 56

Weight Transition: Step 56 70

600

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

600

1200

1000

1000

500

500

1000

800

FromBin(Step0)

800

400

400

800

600

600

300

300

600

400

400

200

200

400

200

200

100

100

200

0

0

0

0

0

To Bin (Step 14)

To Bin (Step 28)

To Bin (Step 42)

To Bin (Step 56)

To Bin (Step 70)

- Figure 3: Dynamics of Variance Promotion Score (VPS) during training. The top row illustrates the distribution of VPS values across data points at different training steps. The bottom row shows transition matrices of VPS assignments between consecutive update intervals, where each cell indicates the number of data points moving from a source bin (vertical axis) at the earlier step to a target bin (horizontal axis) at the later step. The arrows indicate the direction from lower to higher VPS bins, facilitating interpretation of upward or downward transitions.

the progressive stabilization of per-prompt VPS rankings, converging toward a consistent ordering. Such convergence aligns with the intended steady state of VAS, in which high-variance “frontier” prompts persist only as a comparatively small yet stable subset.

Bidirectional movement (adaptation). Non-negligible amount of off–diagonal mass persists in both directions (low→high and high→low), reflecting the adaptive nature of VPS under the evolving policy. Certain prompts gain informativeness as their outcome become more balanced or their trajectory patterns diversity increase, whereas others gradually lose informativeness once they are either consistently solved with ease or repeatedly lead to failure.

High→high persistence fades. In the early stages, a visible block appears in the bottom-right region (high→high), but this gradually weakens as training progresses. This pattern indicates that many prompts are initially challenging, remaining near the maximum OVS/TDS levels across updates. As the policy improves, however, these prompts either become polarized in correctness—lowering OVS—or converge toward more uniform trajectories—lowering TDS. Consequently, they gradually exit the high-VPS subset.

Distributional shift toward mid-VPS. The marginal distribution of VPS progressively evolves from a relatively flat shape with a pronounced upper tail to a more compact form centered around medium scores. Since VPS = α OVS + β TDS with α(0.8) > β(0.2), this shift reflects two key dynamics: (1) as training proceeds, fewer prompts achieve near-maximum OVS (with pass rates ≈ 0.5), and (2) the contribution of TDS establishes a residual floor, anchoring many prompts at moderate VPS even when correctness becomes less stable. This emergent clustering around mid-VPS characterizes a signature of convergence: VAS increasingly narrows its focus to a frontier of prompts where reward variance remains informative and useful for further optimization.

Asymmetry of flows. In later intervals, transitions from higher to mid VPS bins occur more frequently than movements in the opposite direction, implying a gradual reduction in reward variance as competence improves. This observation aligns with the Variance–Progress principle (ref. §4): once learning progress has been achieved on a given prompt, both its reward variance and VPS tend to diminish, thereby encouraging the sampler to redistribute probability mass toward other frontier items that remain more uncertain or informative.

- 6.7 QUALITATIVE DEMONSTRATION

As shown in Figure 4, this MathVerse question highlights the reasoning capability of MMR1. The solution it generates follows a clear and logical structure: the model begins by restating the given conditions, then applies the angle-sum property of a triangle to determine the missing angle, and finally uses parallel-line properties to compute the target angle. This step-by-step organization reflects a coherent “analyze–plan–execute” process. The response also demonstrates reflective reasoning.

###### Thinking Process (Cont’d)

Question

[Figure 18]

- * Points A, E, C are collinear, forming a straight line. Angles AED and CED form a linear pair.
- * Angles in a linear pair are supplementary.
- * Therefore, Angle AED + Angle CED = 180°.
- * 40.0° + Angle CED = 180°.
- * Angle CED = 180° - 40.0° = 140.0°.

- As shown in the figure, in triangle ABC, it is known that ∠A = 80.0, ∠B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is (). Choices: A: 40° B: 60° C: 120° D: 140°

- 5.

|Verify the result:|
|---|

- * Both methods yield Angle CED = 140.0°.
- * The steps are logically sound and use correct geometric properties.

- 6. Compare with choices:

- * The calculated value is 140.0°.
- * The choices are A: 40°, B: 60°, C: 120°, D: 140°.
- * The result matches choice D.

- 7. Final Answer Formulation:

Thinking Process

- 1.

|Analyze|
|---|

the given information:

* We have a triangle ABC. Angle A = 80.0 degrees. Angle B = 60.0 degrees. Point D is on side AB. Point E is on side AC. Line segment DE is parallel to line segment BC (DE || BC).

- 2. Identify the goal:

* Find the measure of angle CED.

- 3.

|Plan|
|---|

the solution:

- * Use the property that the sum of angles in a triangle is 180 degrees to find angle C in triangle ABC.
- * Use the property of parallel lines (DE || BC) and transversals to relate angle CED to other angles in the

figure.

- 4.

* The reasoning involves calculating angle C and then using the property of consecutive interior angles formed by parallel lines DE and BC with transversal AC.

- * Angle C = 180° - (80° + 60°) = 40°.
- * Angle CED + Angle C = 180° (consecutive interior angles).
- * Angle CED = 180° - 40° = 140°.
- * The correct option is D.

|Execute|
|---|

the plan:

- * Step 1: Find angle C in triangle ABC.
- * In triangle ABC, the sum of angles is 180°.
- * Angle A + Angle B + Angle C = 180°
- * 80.0° + 60.0° + Angle C = 180°
- * 140.0° + Angle C = 180°
- * Angle C = 180° - 140.0°
- * Angle C = 40.0°
- * Step 2: Use the parallel lines property.
- * We are given that DE || BC.
- * Consider AC as a transversal line intersecting the parallel lines DE and BC.
- * The angles CED and ACB (which is angle C) are consecutive interior angles (also called same-side

Final Answer

- 1. Find the measure of angle C in triangle ABC. The sum of angles in a triangle is 180°. Angle C = 180° - (Angle A + Angle B) = 180° - (80.0° + 60.0°) = 180° - 140.0° = 40.0°
- 2. Use the property of parallel lines. Since DE || BC, and AC is a transversal line intersecting these parallel lines, the consecutive interior angles (also known as same-side interior angles) are supplementary.

Angle CED + Angle ACB = 180° Angle CED + Angle C = 180° Angle CED + 40.0° = 180° Angle CED = 180° - 40.0° = 140.0°

interior angles).

* Consecutive interior angles formed by parallel lines and a transversal are supplementary (their sum is 180°).

Alternatively:

- * Therefore, Angle CED + Angle ACB = 180°.
- * Angle CED + Angle C = 180°.
- * Substitute the value of Angle C found in Step 1:
- * Angle CED + 40.0° = 180°.
- * Angle CED = 180° - 40.0°.
- * Angle CED = 140.0°.

- 1. Find the measure of angle C in triangle ABC: Angle C = 40.0°.
- 2. Since DE || BC, and AC is a transversal, the corresponding angles are equal. Angle AED = Angle ACB = Angle C = 40.0°.
- 3. Angles AED and CED form a linear pair on the straight line AC. Therefore, they are supplementary. Angle AED + Angle CED = 180° 40.0° + Angle CED = 180° Angle CED = 180° - 40.0° = 140.0°

|Alternative Step|
|---|

*

2 (Using corresponding angles):

- * Since DE || BC, consider AC as a transversal.
- * Angle AED and Angle ACB (Angle C) are corresponding angles.
- * Therefore, Angle AED = Angle C = 40.0°.

The size of angle CED is 140.0°. Comparing this with the choices: A: 40°, B: 60°, C: 120°, D: 140°. The correct option is \boxed{D}.

- Figure 4: Qualitative demonstration of MMR1’s reasoning process on a MathVerse problem. The figure illustrates the input question, the model’s step-by-step thinking process, and the final answer. The reasoning is logically structured, including problem analysis, solution planning, execution, verification, and alternative approaches, ultimately arriving at the correct answer (140◦).

After deriving the result, the model verifies consistency with geometric rules and cross-checks against the answer choices. Furthermore, it provides an alternative method based on corresponding and supplementary angles, which strengthens confidence in the correctness of the conclusion.

Overall, the model not only produces the correct answer but also exhibits robust reasoning behaviors, including systematic decomposition, verification, and multiple-solution perspectives, which illustrate strong problem-solving ability beyond direct computation.

7 CONCLUSION AND LIMITATION

In this work, we investigate the challenge of gradient vanishing in reinforcement learning for multimodal reasoning. We introduce Variance-Aware Sampling (VAS), a sampling strategy that exploits outcome variance and trajectory diversity to prioritize informative prompts while maintaining broad data coverage. Grounded in theoretical analysis and supported by extensive empirical evaluation, VAS enhances training stability and improves the effectiveness of reinforcement learning in multimodal reasoning tasks. Beyond the methodological contribution, a central outcome of this work is the release of large-scale, carefully curated cold-start datasets and well-tuned models, which we hope will provide valuable resources for benchmarking and advancing future research in this area.

Despite these contributions, our work has several limitations. First, although VAS mitigates gradient vanishing, it does not fully resolve all training instabilities inherent to multimodal reinforcement learning. Second, the computation of variance-based prompt scores (VPS) incurs additional overhead, though this can be mitigated by increasing update intervals or selectively updating a subset of samples. Finally, our method primarily focuses on data sampling; while it is expected to complement algorithmic advances in reinforcement learning, a systematic investigation into their integration is left to future work.

Looking ahead, we believe this study opens several promising avenues. Future research may explore extending VAS to broader domains, examining its interaction with diverse reward designs, and integrating it with more advanced reinforcement learning algorithms to further improve sample efficiency and robustness. We hope that our methodological innovations, together with the released resources, will provide a solid foundation for the community to advance the development of more stable and capable multimodal reasoning models.

REFERENCES

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12248–12267, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.

org/2024.acl-long.662/.

Shun’ichi. Amari and Hiroshi Nagaoka. Methods of information geometry. Translations of mathematical monographs, v. 191. American Mathematical Society, Providence, R.I, 2000. ISBN 0821805312. URL https://primo.lib.umn.edu/permalink/01UMN_INST/1sro6u2/ alma9918523770001701.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. ArXiv, abs/2502.13923, 2025. URL https://arxiv.org/abs/2502.13923.

Huanqia Cai, Yijun Yang, and Winston Hu. Mm-iq: Benchmarking human-like abstraction and reasoning in multimodal models. ArXiv, abs/2502.00698, 2025. URL https://arxiv.org/abs/ 2502.00698.

Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th International Conference on Computational Linguistics, pp. 1511–1520, Gyeongju, Republic of Korea, October 2022. International Committee on Computational Linguistics. URL https://aclanthology.org/2022.coling-1.130/.

Guizhen Chen, Weiwen Xu, Hao Zhang, Hou Pong Chan, Deli Zhao, Anh Tuan Luu, and Yu Rong. Geopqa: Bridging the visual perception gap in mllms for geometric reasoning. ArXiv, abs/2509.17437, 2025a. URL https://arxiv.org/abs/2509.17437.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. ArXiv, abs/2412.05271, 2025b. URL https://arxiv.org/abs/2412.05271.

Google DeepMind. Gemini 2.5: Our most intelligent ai model, Mar 2025. URL https://blog. google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/ #gemini-2-5-thinking.

Linger Deng, Linghao Zhu, Yuliang Liu, Yu Wang, Qunyi Xie, Jingjing Wu, Gang Zhang, Yingying Zhu, and Xiang Bai. Theorem-validated reverse chain-of-thought problem generation for geometric reasoning. ArXiv, abs/2410.17885, 2024. URL https://arxiv.org/abs/2410.17885.

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. ArXiv, abs/2503.17352, 2025. URL https://arxiv.org/abs/2503.17352.

Thomas Foster and Jakob Foerster. Learning to reason at the frontier of learnability. ArXiv, abs/2502.12272, 2025. URL https://arxiv.org/abs/2502.12272.

Evan Greensmith, Peter L. Bartlett, and Jonathan Baxter. Variance reduction techniques for gradient estimates in reinforcement learning. Journal of machine learning research, 5:1471–1530, Nov 2004. URL https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt. Openthoughts: Data recipes for reasoning models. ArXiv, abs/2506.04178, 2025. URL https://arxiv.org/abs/2506.04178.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645:633–638, September 2025. doi: 10.1038/s41586-025-09422-z.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=7Bywt2mQsCe.

Jack *Hessel, Jena D *Hwang, Jae Sung Park, Rowan Zellers, Chandra Bhagavatula, Anna Rohrbach, Kate Saenko, and Yejin Choi. The Abduction of Sherlock Holmes: A Dataset for Visual Abductive Reasoning. In ECCV, 2022. URL https://www.ecva.net/papers/eccv_2022/papers_ECCV/ papers/136960549.pdf.

hiyouga. Mathruler. https://github.com/hiyouga/MathRuler, 2025. Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: An efficient rlhf algorithm

with robustness to both prompt and reward models. ArXiv, abs/2501.03262, 2025. URL https: //arxiv.org/abs/2501.03262.

Chenghua Huang, Lu Wang, Fangkai Yang, Pu Zhao, Zhixu Li, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Lean and mean: Decoupled value policy optimization with global value guidance. ArXiv, abs/2502.16944, 2025a. URL https://arxiv.org/abs/2502. 16944.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models.

- ArXiv, abs/2503.06749, 2025b. URL https://arxiv.org/abs/2503.06749.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models.

- ArXiv, abs/2503.06749, 2025c. URL https://arxiv.org/abs/2503.06749.

Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. URL https: //openaccess.thecvf.com/content_CVPR_2019/papers/Hudson_GQA_A_New_Dataset_ for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.pdf.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017. URL https://openaccess.thecvf.com/content_cvpr_2017/papers/ Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.pdf.

Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. URL https://openaccess.thecvf.com/content_ cvpr_2018/papers/Kafle_DVQA_Understanding_Data_CVPR_2018_paper.pdf.

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. In AI for Math Workshop @ ICML 2024, 2024. URL https://openreview.net/forum?id=1AUbiBrOF1.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision, pp. 235–251, Cham, 2016. Springer International Publishing. URL https://doi.org/10.1007/ 978-3-319-46493-0_15.

Kimi. Kimi k1.5: Scaling reinforcement learning with llms. ArXiv, abs/2501.12599, 2025. URL https://arxiv.org/abs/2501.12599.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, pp. 611–626, New York, NY, USA, 2023. Association for Computing Machinery. URL https://doi.org/10.1145/3600006.3613165.

Hynek Kydlíˇcek. Math-Verify: Math Verification Library, 2025. URL https://github.com/ huggingface/math-verify.

Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Yuming Jiang Hang Zhang, Xin Li, Deli Zhao, Fan Wang, Yu Rong, Aixin Sun, and Shijian Lu. Mmr1: Advancing the frontiers of multimodal reasoning. https://github.com/LengSicong/MMR1, 2025.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. ArXiv, abs/2408.03326, 2024a. URL https://arxiv.org/abs/2408.03326.

Junnan Li, Yongkang Wong, Qi Zhao, and Mohan S. Kankanhalli. Dual-glance model for deciphering social relationships. In 2017 IEEE International Conference on Computer Vision (ICCV), pp. 2669–2678, 2017. URL https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_ Dual-Glance_Model_for_ICCV_2017_paper.pdf.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. Limr: Less is more for rl scaling. ArXiv, abs/2502.11886, 2025a. URL https://arxiv.org/abs/2502.11886.

Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, Yingying Zhang, Fei Yin, Jiahua Dong, Zhijiang Guo, Le Song, and Cheng-Lin Liu. From system 1 to system 2: A survey of reasoning large language models. ArXiv, abs/2502.17419, 2025b. URL https://arxiv.org/abs/2502.17419.

Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. ReMax: A simple, effective, and efficient reinforcement learning method for aligning large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 29128–29163. PMLR, 21–27 Jul 2024b. URL https://proceedings.mlr.press/v235/li24cd.html.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=v8L0pN6EOi.

Adam Dahlgren Lindström and Savitha Sam Abraham. Clevr-math: A dataset for compositional language, visual and mathematical reasoning. ArXiv, abs/2208.05358, 2022. URL https:// arxiv.org/abs/2208.05358.

Jiacai Liu, Chaojie Wang, Chris Yuhao Liu, Liang Zeng, Rui Yan, Yiwen Sun, Yang Liu, and Yahui Zhou. Improving multi-step reasoning abilities of large language models with direct advantage policy optimization. ArXiv, abs/2412.18279, 2024. URL https://arxiv.org/abs/2412.18279.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. ArXiv, abs/2503.20783, 2025. URL https://arxiv.org/abs/2503.20783.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-GPS: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 6774–6786, Online, August 2021a. Association for Computational Linguistics. URL https://aclanthology.org/2021.acl-long.528/.

Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. IconQA: A new benchmark for abstract diagram understanding and visual language reasoning. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021b. URL https://openreview.net/forum?id= uXa9oBDZ9V1.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35, pp. 2507–2521. Curran Associates, Inc., 2022a. URL https://proceedings.neurips.cc/paper_ files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf.

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and A. Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. ArXiv, abs/2209.14610, 2022b. URL https://arxiv.org/abs/2209.14610.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=KUNzEQMWU7.

Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2263–2279, Dublin, Ireland, May 2022a. Association for Computational Linguistics. URL https://aclanthology.org/2022.

findings-acl.177/.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2263–2279, Dublin, Ireland, May 2022b. Association for Computational Linguistics. URL https://aclanthology.org/2022.

findings-acl.177/.

Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2200–2209, January 2021. URL https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_ Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1697–1706, January 2022. URL https://openaccess.thecvf.com/ content/WACV2022/papers/Mathew_InfographicVQA_WACV_2022_paper.pdf.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. ArXiv, abs/2503.07365, 2025a. URL https://arxiv.org/abs/2503.

07365.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. ArXiv, abs/2503.07365, 2025b. URL https: //arxiv.org/abs/2503.07365.

Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), March 2020. URL https://openaccess.thecvf.com/content_WACV_2020/ papers/Methani_PlotQA_Reasoning_over_Scientific_Plots_WACV_2020_paper.pdf.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. s1: Simple test-time scaling. In Workshop on Reasoning and Planning for Large Language Models, 2025. URL https://openreview.net/forum?id=LdH0vrgAHm.

OpenAI. Gpt-4o system card. AvrXiv, abs/2410.21276, 2024. URL https://arxiv.org/abs/2410. 21276.

Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1470–1480, Beijing, China, July 2015. Association for Computational Linguistics. URL https://aclanthology.org/P15-1142/.

Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. ArXiv, abs/2409.00147, 2024. URL https://arxiv.org/abs/2409.00147.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. ArXiv, abs/2503.07536, 2025. URL https://arxiv. org/abs/2503.07536.

Shraman Pramanick, Rama Chellappa, and Subhashini Venugopalan. SPIQA: A dataset for multimodal question answering on scientific papers. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https: //openreview.net/forum?id=h3lddsY5nf.

Noam Razin, Hattie Zhou, Omid Saremi, Vimal Thilak, Arwen Bradley, Preetum Nakkiran, Joshua M. Susskind, and Etai Littwin. Vanishing gradients in reinforcement finetuning of language models. In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=IcVNBR7qZi.

Noam Razin, Zixuan Wang, Hubert Strauss, Stanley Wei, Jason D. Lee, and Sanjeev Arora. What makes a reward model a good teacher? an optimization perspective. ArXiv, abs/2503.15477, 2025. URL https://arxiv.org/abs/2503.15477.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv, abs/1707.06347, 2017. URL https://arxiv.org/abs/1707. 06347.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, 2022. URL https://www.ecva.net/papers/eccv_2022/papers_ ECCV/papers/136680141.pdf.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. ArXiv, abs/2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model. ArXiv, abs/2504.07615, 2025. URL https://arxiv.org/abs/2504.07615.

Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-LLaVA: Bootstrapping mathematical reasoning for multimodal large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 4663– 4680, Miami, Florida, USA, November 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.findings-emnlp.268/.

Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. ArXiv, abs/2503.20752,

2025. URL https://arxiv.org/abs/2503.20752. Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. ArXiv, abs/2101.11272, 2021. URL https://arxiv.org/abs/2101.11272. Mert Unsal and Aylin Akkus. Easyarc: Evaluating vision language models on true visual reasoning. ArXiv, abs/2506.11595, 2025. URL https://arxiv.org/abs/2506.11595.

Jean Vassoyan, Nathanaël Beau, and Roman Plaud. Ignore the KL penalty! boosting exploration on critical tokens to enhance RL fine-tuning. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 6108–6118, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025.findings-naacl.340/.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. ArXiv,

- abs/2504.08837, 2025a. URL https://arxiv.org/abs/2504.08837.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. ArXiv,

- abs/2504.08837, 2025b. URL https://arxiv.org/abs/2504.08837.

Huaijie Wang, Shibo Hao, Hanze Dong, Shenao Zhang, Yilin Bao, Ziran Yang, and Yi Wu. Offline reinforcement learning for LLM multi-step reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 8881–8893, Vienna, Austria, July 2025c. Association for Computational Linguistics. URL https://aclanthology.org/2025.findings-acl.464/.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with MATH-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks

- Track, 2024a. URL https://openreview.net/forum?id=QWTCcxMpPA.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with MATH-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks

- Track, 2024b. URL https://openreview.net/forum?id=QWTCcxMpPA.

Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, and Hongsheng Li. MathCoder-VL: Bridging vision and code for enhanced multimodal mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 2505–2534, Vienna, Austria, July 2025d. Association for Computational Linguistics. ISBN 979-8-89176-256-5. URL https://aclanthology.org/2025.

findings-acl.128/.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439, Bangkok, Thailand, August 2024c. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.510/.

Tianlong Wang, Junzhe Chen, Xueting Han, and Jing Bai. Cpl: Critical plan step learning boosts llm generalization in reasoning tasks. ArXiv, abs/2409.08642, 2024d. URL https://arxiv.org/ abs/2409.08642.

Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. ArXiv, abs/2504.07934, 2025e. URL https://arxiv.org/ abs/2504.07934.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4):229–256, May 1992. ISSN 0885-6125. URL https: //doi.org/10.1007/BF00992696.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. ArXiv, abs/2407.04973, 2024. URL https://arxiv.org/abs/ 2407.04973.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, Bo Zhang, and Wei Chen. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. ArXiv, abs/2503.10615, 2025a. URL https://arxiv.org/abs/2503.10615.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. ArXiv, abs/2503.10615, 2025b. URL https://arxiv.org/ abs/2503.10615.

Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, Hao Zhang, and Yu Rong. Vl-cogito: Progressive curriculum reinforcement learning for advanced multimodal reasoning. ArXiv, abs/2507.22607, 2025. URL https://arxiv.org/abs/2507.22607.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, Tiantian Fan, Zhengyin Du, Xiang Wei, Xiangyu Yu, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan Wang, Yong-Xu Wu, and Lin Yan. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. ArXiv, abs/2504.05118, 2025. URL https://arxiv.org/ abs/2504.05118.

Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. URL https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_RAVEN_ A_Dataset_for_Relational_and_Analogical_Visual_REasoNing_CVPR_2019_paper.pdf.

Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. ArXiv, abs/2503.12937, 2025a. URL https://arxiv.org/abs/2503.12937.

Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. ArXiv, abs/2503.12937, 2025b. URL https://arxiv.org/abs/ 2503.12937.

Kechi Zhang, Ge Li, Jia Li, Yihong Dong, Jia Li, and Zhi Jin. Focused-DPO: Enhancing code generation through focused preference optimization on error-prone points. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 9578–9591, Vienna, Austria, July 2025c. Association for Computational Linguistics. ISBN 979-8-89176-256-5. URL https:

//aclanthology.org/2025.findings-acl.498/.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186, Berlin, Heidelberg, 2024a. Springer-Verlag. URL https://doi.org/10.1007/978-3-031-73242-3_ 10.

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Shicheng Li, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, Peng Gao, Chunyuan Li, and Hongsheng Li. Mavis: Mathematical visual instruction tuning with an automatic data engine. ArXiv, abs/2407.08739, 2024b. URL https://arxiv.org/abs/2407.08739.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 400– 410, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https: //aclanthology.org/2024.acl-demos.38/.

Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. https://github.com/ hiyouga/EasyR1, 2025.

Jin Peng Zhou, Kaiwen Wang, Jonathan Chang, Zhaolin Gao, Nathan Kallus, Kilian Q. Weinberger, Kianté Brantley, and Wen Sun. q♯: Provably optimal distributional rl for llm post-training. ArXiv, abs/2502.20548, 2025. URL https://arxiv.org/abs/2502.20548.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. ArXiv, abs/2504.10479, 2025. URL https://arxiv.org/abs/2504.10479.

### A VARIANCE–PROGRESS THEORY

Intuition. Gradient updates are informative only if sampled rollouts produce different rewards. When rewards are nearly identical, advantages collapse and gradients vanish. Our theorem shows that (under standard smoothness and non-degeneracy conditions) the expected improvement after a single update is linearly lower-bounded by the reward variance of the prompt. Thus, selecting prompts that induce mixed outcomes (OVS) and diverse reasoning paths (TDS) provably strengthens learning.

- At a glance (notation & assumptions). We write g(x,y) = ∇θ log πθ(y | x), R¯(x) = Ey[R(x,y)], Rres = R − R¯, and Γθ(x) = Ey[g g⊤]. Assumptions: (i) L-smoothness of Jx; (ii) bounded score function ∥g∥≤Gmax; (iii) uniform positive definiteness of Γθ(x) ⪰ λminI; (iv) gradient lower bound ∥∇Jx∥2≥cmin Var[R] (a mild consequence of (iii)).

- A.1 TECHNICAL PRELIMINARIES

Throughout we fix a prompt x ∈ X and write expectations over y as Ey[·] = Ey∼π

θ(·|x)[·]. The policy is differentiable and strictly positive on Y. We denote R¯(x) = Ey[R(x,y)] and Rres(x,y) = R(x,y) − R¯(x).

- A.1.1 ACTION-INDEPENDENT BASELINES

- Proposition 1 (Optimal action-independent baseline). For any square-integrable reward R, the baseline b⋆(x) = R¯(x) minimizes the total variance of the REINFORCE gradient estimator over all baselines b(x) that depend on x but not on y.

Proof. Let Gb(x) = Ey[g(x,y)(R(x,y) − b(x))], where g(x,y) = ∇θ log πθ(y | x). Using Ey[g] = 0, the covariance term vanishes and Var[Gb(x)] = Ey[∥g∥2(R − b)2]. The right-hand side is a convex quadratic in b. Differentiating and setting to zero yields b(x) = R¯(x) (Williams, 1992).

| |
|---|

Remark. An action-dependent baseline such as b∥g∥

2

(x) = Ey[∥g∥2R]/Ey[∥g∥2] can further reduce scalar variance (Greensmith et al., 2004), but requires inner Monte-Carlo estimates. The bounds below hold for any y-independent baseline.

- A.1.2 GRADIENT-VARIANCE BOUNDS With b⋆(x) = R¯(x),

G(x) = Ey[g(x,y)Rres(x,y)], E[G(x)] = ∇θJx(θ). Its covariance is

Var[G(x)] = Ey[Rres2 g g⊤] − ∇θJx(θ)∇θJx(θ)⊤. For bounding we drop the nonnegative outer product, which only reduces the variance, yielding a valid but looser inequality.

Assumption 2 (Bounded score-function gradient). There exists Gmax < ∞ such that ∥g(x,y)∥ ≤ Gmax for all (x,y).

Lemma 1 (Variance sandwich bound). Under Assumption 2, let Γθ(x) = Ey[g(x,y)g(x,y)⊤]. If λmin > 0 is the smallest eigenvalue of Γθ(x) uniformly over x, then

##### λmin Vary[R(x,y)]Id ⪯ Var[G(x)] ⪯ G2max Vary[R(x,y)]Id.

Thus reward variance is the only prompt-dependent factor; Fisher terms contribute bounded, modeldependent constants.

Proof. For any unit vector v, v⊤ Var[G(x)]v = Ey[(v⊤g)2Rres2 ]. Upper bound: (v⊤g)2 ≤ ∥g∥2 ≤ G2max. Lower bound: Γθ(x) ⪰ λminI implies v⊤ Var[G(x)]v ≥ λminEy[Rres2 ] = λmin Vary[R].

| |
|---|

- A.1.3 BOUNDS ON THE FISHER TERM

- Proposition 2 (Uniform Fisher bounds). Under Assumption 2, there exist constants 0 < λmin ≤ λmax = G2max such that

λmin Id ⪯ Γθ(x) ⪯ λmax Id, ∀x ∈ X.

Proof. The upper bound follows from ∥g∥ ≤ Gmax. The lower bound holds under the standard non-degeneracy assumption that πθ(· | x) defines a full-dimensional exponential family and θ ranges over a compact set, ensuring Γθ(x) ≻ 0 uniformly (Amari & Nagaoka, 2000).

| |
|---|

- A.2 PROOF OF THE VARIANCE–PROGRESS THEOREM Let the update be θ+ = θ + ηG(x) with η > 0. A second-order Taylor expansion yields

Jx(θ+) − Jx(θ) = η⟨∇Jx,G(x)⟩ + 12η2G(x)⊤Hx(θ˜)G(x), for some θ˜on the segment [θ,θ+]. Taking expectations and L-smoothness,

E[Jx(θ+) − Jx(θ)] ≥ η∥∇Jx∥2 − 21η2LE[∥G(x)∥2]. Decomposing E[∥G(x)∥2] into bias and variance and using Lemma 1,

E[Jx(θ+) − Jx(θ)] ≥ η∥∇Jx∥2 − 12η2L ∥∇Jx∥2 + dG2max Var[R] . Assuming ∥∇Jx∥2 ≥ cmin Var[R],

E[Jx(θ+) − Jx(θ)] ≥ ηcmin Var[R] − 12η2L(cmin + dG2max)Var[R]. Choosing η ≤ c

min

2L(cmin+dG2max) gives E[Jx(θ+) − Jx(θ)] ≥ ηc

min

2 Var[R]. Restricting further to η ≤ cmin/(4L) yields the simplified bound used in the main text:

E[Jx(θ+) − Jx(θ)] ≥ ηc

min

4 Vary∼π

θ

[R(x,y)].

Discussion. The bound depends only on constants cmin and L tied to the model family. All prompt dependence enters through Var[R], establishing reward variance as the decisive quantity.

- A.3 TWO-LEVEL DECOMPOSITION OF REWARD VARIANCE

Let y = (ycot,yans) and define R(x,y) = 1verifier(y

ans). Let Z = φ(ycot) represent the chain. By the law of total variance,

Vary[R] = EZ[pZ(1 − pZ)] + VarZ[pZ], where pZ(x) = Pr(R = 1 | Z). Intra/inter-trajectory terms. The first term is intra-trajectory Bernoulli variance; the second is inter-trajectory variation of success probabilities. Efron–Stein lower bound. If |pz − pz′| ≤ Ld(ycot(z),ycot(z′)) for a bounded distance d ∈ [0,1], then

2

4 EZ,Z′[d2(ycot(Z),ycot(Z′))].

VarZ[pZ] ≥ L

#### OVS and TDS estimators. With K rollouts, pˆ = K1 R,

d2(ycot(i) ,ycot(j)).

OVS(x) = pˆ(1 − pˆ), TDS(x) = K(K1−1)

i̸=j

By the strong law and U-statistic convergence, both estimators are strongly consistent for their respective population terms.

Variance Promotion Score (VPS). Define VPS = α OVS + βTDS with α,β > 0. Then VPS converges almost surely to a positive affine transform of a lower bound on Var[R]. Hence, VPS is a strongly consistent monotone surrogate for reward variance (it does not require equality to hold).

- A.4 EXTENSION FROM REINFORCE TO GRPO

GRPO replaces the scalar baseline by the in-batch mean reward and whitens with the sample standard deviation:

R(x,yi) − r¯(x) s(x) + δ

R˜(x,yi) =

.

This yields a centered, variance-controlled REINFORCE estimator. Multiplying by importance ratios (and optionally clipping) preserves the core dependence on reward variance: the Variance–Progress lower bound continues to hold after a rescaling of constants. When clipping is enabled, an O(ε) bias arises; the bound is reduced by the same order but remains strictly positive whenever Var[R] > 0. Thus prompts that induce higher reward variance guarantee larger expected improvements under GRPO, paralleling vanilla REINFORCE.

- B FINE-GRAINED MATH TYPE DEFINITIONS

For the construction of our math dataset, each problem is assigned to one of thirteen fine-grained categories. These categories provide balanced coverage across fundamental and advanced domains of mathematics, and ensure that the RL dataset spans diverse reasoning skills. The formal definitions, explanations, and illustrative examples for each category are presented below.

- 1. Arithmetic Definition: Arithmetic covers basic numerical operations, including addition, subtraction, multiplication, and division. Explanation: It forms the foundation of mathematics by establishing rules for manipulating numbers. Example: Compute 15 × 12.
- 2. Counting Definition: Counting addresses enumeration of objects or elements within a collection. Explanation: It includes both simple enumeration and principles such as permutations and combinations. Example: Determine how many integers between 1 and 100 are multiples of 5.
- 3. Combinatorics Definition: Combinatorics studies arrangements, selections, and combinations of discrete objects. Explanation: It extends counting principles to complex scenarios involving structured sets. Example: How many ways can 5 distinct books be arranged in a row?
- 4. Algebra Definition: Algebra represents relationships using symbols and equations. Explanation: It provides systematic tools for solving equations and reasoning about unknowns. Example: Solve 2x + 3 = 9 for x.
- 5. Functions Definition: A function maps each input to exactly one output. Explanation: Functions formalize dependencies between quantities, described via formulas, graphs, or rules. Example: Given f(x) = x2 + 3, find f(2).
- 6. Plane Geometry Definition: Plane Geometry studies figures such as lines, angles, and polygons in two dimensions. Explanation: It addresses lengths, angles, and areas of flat figures. Example: Find the area of a triangle with base 10 and height 5.
- 7. Solid Geometry Definition: Solid Geometry extends geometric reasoning to three-dimensional figures. Explanation: It concerns volumes, surface areas, and properties of 3D objects. Example: Find the volume of a cube with side length 4.

- 8. Combinatorial Geometry Definition: Combinatorial Geometry analyzes discrete configurations of geometric objects. Explanation: It merges counting with geometry, such as enumerating diagonals or intersections. Example: How many diagonals does a convex octagon have?
- 9. Descriptive Geometry Definition: Descriptive Geometry represents 3D objects using 2D projections. Explanation: It enables precise measurement of spatial relationships via orthographic or perspective drawings. Example: Sketch the top and front views of a cube resting on a horizontal plane.
- 10. Graph Theory Definition: Graph Theory studies structures composed of vertices and edges. Explanation: It focuses on connectivity, paths, and cycles in discrete networks. Example: Given a graph with vertices A, B, C, D and edges AB, BC, CD, and DA, does the graph contain a cycle?
- 11. Logic Definition: Logic studies formal reasoning and inference. Explanation: It examines propositions, truth values, and the validity of conclusions. Example: If "All cats are mammals" and "Fluffy is a cat," deduce whether "Fluffy is a mammal."
- 12. Statistics Definition: Statistics concerns the collection, analysis, and interpretation of data. Explanation: It uses measures such as mean, median, and variance to summarize data. Example: For {2,4,4,6,8}, compute the mean and median.
- 13. Topology Definition: Topology studies properties of spaces invariant under continuous deformations. Explanation: It focuses on qualitative properties such as connectivity and the number of holes. Example: Explain why a doughnut (torus) cannot be deformed into a sphere without removing the hole.

These categories are used to uniformly sample math problems in the RL stage, ensuring both difficulty balance and coverage across diverse mathematical skills.

- C DETAILED HYPER-PARAMETERS

The main hyperparameters used in the cold-start supervised fine-tuning stage are summarized in Table 5.

Table 5: Cold-start stage hyperparameters.

#### Setting Value

Training epochs 3 Gradient accumulation steps 4 Effective batch size 32 Sequence length cutoff 16,384 Optimizer AdamW Learning rate 1 × 10−5 Learning rate schedule Cosine decay Warm-up ratio 0.1 Weight decay 0 Max gradient norm 1.0 Precision bfloat16 Deepspeed config ZeRO-3 (32 shards)

The main hyperparameters used in the RL stage (GRPO with VAS) are summarized in Table 6.

Table 6: RL stage hyperparameters.

#### Setting Value

Initialization (policy) Cold-start checkpoint (Qwen2.5-VL) Objective GRPO (adv_estimator=grpo) Reward 1 ∗ format reward + 1 ∗ accuracy reward KL regularization Enabled, penalty=low_var_kl, coef = 0.01 (fixed) Precision bfloat16 System prompt As in Appx. D

Data & VAS sampling Max prompt / response length 2048 / 4096 Image resolution limits [7,056, 1,048,576] pixels Sampling strategy VAS (curriculum) VAS metrics / weights learnability (OVS) 0.8, self_bleu_123 (TDS) 0.2 VAS update frequency 56 steps VAS mixture ratio 0.5 (weighted vs. uniform) VPS rollout for scoring n = 16, batch size = 4096

Rollout & inference

- Sampler vLLM
- Samples per prompt n = 8 Temperature / top-p / top-k 0.6 / 1.0 / −1 Validation override temp = 0.5, n = 1 GPU memory util (vLLM) 0.75

Optimization (actor / critic) Global batch size (actor / critic) 512 / 256 Micro-batch (update) (actor / critic) 8 / 4 Micro-batch (experience) (actor / critic) 32 / 16 PPO epochs 20 Clipping policy clip_low=0.2, clip_high=0.2; value cliprange=0.5 Max grad norm 1.0 (both) Optimizer AdamW (lr = 1 × 10−6, betas = (0.9,0.999)) Weight decay 0.01 LR warmup ratio 0.0 (constant warmup style)

Parallelism & hardware FSDP Full shard (policy & critic), fsdp_size=8 Episodes total_episodes = 10

- D SYSTEM PROMPT

Training Prompt. The following system prompt is used during RL training to enforce a structured reasoning format and to require the final answer to be enclosed in “\boxed{}”.

Table 7: System prompt used in RL training.

A conversation between User and Assistant. The User provides an image and asks a question. The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here, with potential reflections and corrections </think><answer> final answer here, with the key result enclosed in \boxed{} </answer>.

Evaluation Prompt. To ensure fairness and generalizability, we use a simplified prompt for evaluation instead of the training prompt.

Table 8: System prompt used for evaluation.

Please solve the problem step by step and put your answer in one \boxed{}. If it is a multiple-choice question, only one letter should appear inside the \boxed{}.

