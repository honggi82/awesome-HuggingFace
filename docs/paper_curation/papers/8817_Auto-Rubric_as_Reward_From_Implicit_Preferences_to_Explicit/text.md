# arXiv:2605.08354v1[cs.AI]8May2026

## Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria

Juanxi Tian1,2∗ Fengyuan Liu1∗ Jiaming Han3 Yilei Jiang3 Yongliang Wu4 Yesheng Liu1 Haodong Li1 Furong Xu2

Wanhua Li1† 1Nanyang Technological University 2Ant Group 3MMLab, The Chinese University of Hong Kong 4UIUC

### Abstract

Aligning multimodal generative models with human preferences demands reward signals that respect the compositional, multi-dimensional structure of human judgment. Prevailing RLHF approaches reduce this structure to scalar or pairwise labels, collapsing nuanced preferences into opaque parametric proxies and exposing vulnerabilities to reward hacking. While recent Rubrics-as-Reward (RaR) methods attempt to recover this structure through explicit criteria, generating rubrics that are simultaneously reliable, scalable, and data-efficient remains an open problem. We introduce Auto-Rubric as Reward (ARR), a framework that reframes reward modeling from implicit weight optimization to explicit, criteria-based decomposition. Before any pairwise comparison, ARR externalizes a VLM’s internalized preference knowledge as prompt-specific rubrics, translating holistic intent into independently verifiable quality dimensions. This conversion of implicit preference structure into inspectable, interpretable constraints substantially suppresses evaluation biases including positional bias, enabling both zero-shot deployment and few-shot conditioning on minimal supervision. To extend these gains into generative training, we propose Rubric Policy Optimization (RPO), which distills ARR’s structured multi-dimensional evaluation into a robust binary reward, replacing opaque scalar regression with rubric-conditioned preference decisions that stabilize policy gradients. On text-to-image generation and image editing benchmarks, ARR-RPO outperforms pairwise reward models and VLM judges, demonstrating that explicitly externalizing implicit preference knowledge into structured rubrics achieves more reliable, data-efficient multimodal alignment, revealing that the bottleneck is the absence of a factorized interface, not a deficit of knowledge. Code is publicly available at https://github.com/OpenEnvision/AutoRubric-as-Reward.

### 1 Introduction

Human preferences are not arbitrary signals but structured, multidimensional judgments encompassing aesthetic value, semantic fidelity, and contextual appropriateness [19, 28, 47]. Aligning generative multimodal models with such preferences therefore demands more than calibration: it requires models to internalize and operationalize the explicit criteria that underpin human evaluation. Prevailing RLHF paradigms contravene this requirement. By collapsing composite preference structures into scalar scores [28, 47] or pairwise labels [19], they encode rich human judgment into opaque, entangled representations, discarding the very dimensions that confer interpretability and stability, and exposing the learning process to reward hacking [4, 10].

Despite their extensive world knowledge and perceptual capabilities, contemporary VLMs exhibit systematic unreliability in modeling human preferences [16, 35]. Pointwise scoring reduces evaluation to a single scalar, providing no constraint on how improvement is achieved and allowing degenerate optimization strategies. Pairwise comparison, while more balanced, still operates on a latent decision boundary, leading to persistent positional biases that resist standard mitigations such as positional labeling or chain-of-thought prompting [25, 35]. Recent Rubrics as Reward (RaR) approaches attempt

∗Equal contribution. †Corresponding author. Correspondence to: Wanhua Li <wanhua.li@ntu.edu.sg>.

to recover structure through explicit criteria; however, their reliance on fixed or supervised rubric construction limits scalability, prompt specificity, and data efficiency, with these limitations becoming more pronounced when extended to multimodal generation settings.

The reframing recasts multimodal alignment as a representation problem: the bottleneck is not a deficit of preference knowledge, but the absence of a stable, factorized interface for applying it. Building on training-free rubric extraction from preference pairs [46], we propose Auto-Rubric as Reward (ARR). ARR synthesizes instance-conditioned rubrics through a generate-verify-refine pipeline that induces discriminative criteria grounded in observable evidence, producing a compact set of verifiable, decision-relevant constraints spanning semantic fidelity, spatial consistency, compositional aesthetics, and edit faithfulness [11, 21, 32, 51]. These criteria compose a structured evaluation protocol for criterion-level comparison, supplanting holistic scoring. Unlike handcrafted rubrics or learned scalar rewards, ARR derives prompt-specific decision structures from minimal preference data with no parameter updates, yielding a highly data-efficient and interpretable interface. By externalizing preference structure into explicit, verifiable criteria, ARR replaces unstable latent comparisons with grounded discrimination, helping to reduce positional bias and mitigating reward hacking. Crucially, rubric quality scales with the underlying VLM’s alignment with human preferences: stronger judges produce more precise criteria without additional supervision.

This formulation extends from evaluation to optimization. If preference is inherently factorized, reward should preserve that structure rather than collapse it. We therefore introduce Rubric Policy Optimization (RPO), which uses ARR-generated criteria to produce binary preference decisions for policy optimization. Unlike prior rubric-based methods that apply criteria as auxiliary filters, RPO integrates rubric-conditioned judgments directly into the optimization objective, aligning gradient updates with interpretable dimensions of quality. This eliminates a separate reward model and mitigates reward hacking by grounding supervision in explicit criteria rather than learned proxies [4, 26]. Evaluation and generation are unified through a shared preference representation, where better understanding of human preferences in evaluation directly strengthens generative alignment.

Empirically, ARR improves preference accuracy over trained reward models and direct VLM judges by 1.7 to 6.3 points, while reducing positional bias and retaining strong zero-shot and few-shot generalization. When used for training, ARR-RPO yields further gains on text-to-image generation and image editing benchmarks [11, 15, 16, 24, 28, 37, 40, 43, 49] (e.g., GenEval: 0.66 to 0.80; DPG-Bench: 83.84 to 85.76). These improvements require no judge fine-tuning or large-scale reward annotation. The core insight is that the bottleneck in multimodal alignment lies not in acquiring more preference knowledge, but in providing a stable, factorized interface to apply it, precisely what explicit rubrics supply.

Our key contributions can be summarized as follows:

- • Auto-Rubric as Reward (ARR). We propose a training-free framework that externalizes implicit human preferences into instance-conditioned, interpretable rubrics. It enables scalable multimodal evaluation with extremely high data efficiency, requiring only a few annotated samples.
- • Rubric Policy Optimization (RPO). We introduce RPO, a policy optimization framework for contrastive preference learning. By conditioning on ARR-derived rubrics, RPO replaces scalar reward signals with structured, criterion-grounded comparisons.
- • Diagnosing the Interface Bottleneck. Ablations reveal the core bottleneck is a missing factorized interface, not a knowledge deficit. ARR-RPO resolves this via explicit rubrics; cross-model and cardinality analyses confirm that deeper comprehension of intrinsic criteria, rather than scale or data volume, drives both evaluation robustness and generative improvement.

### 2 Related Work

Multimodal Reward Modeling. RLHF underpins alignment across text-to-image generation, editing, and video synthesis. Early reward models such as PickScore, ImageReward, and HPS compress rich human preferences into scalar signals [19, 28, 47]. While effective for coarse ranking, such compression obscures preference structure and is prone to reward hacking and overfitting [4, 53]. Direct optimization methods eliminate explicit reward modeling but still rely on scalar or pairwise objectives, inheriting similar limitations in expressivity and robustness [10, 34]. Recent VLM-as-ajudge approaches leverage stronger multimodal priors, yet exhibit persistent biases, such as positional and symmetry bias, that are difficult to eliminate through prompting alone [16, 25, 35, 52]. Taken

together, these methods suggest that the core limitation is not a lack of preference knowledge, but the absence of a structured interface for expressing and applying it. We address this by externalizing implicit preferences into explicit, prompt-conditioned rubrics, enabling factorized and verifiable evaluation in place of opaque scalar scoring.

Rubric as Reward. To overcome the limitations of scalar evaluation, recent work has explored rubric-based formulations that decompose judgments into interpretable criteria. In language tasks, analytic rubric frameworks [30, 48] and LLM-Rubric [13] show that criterion-level assessment yields more stable and calibrated signals than holistic scoring [1, 18, 29]. AutoRubric [46] extends this idea by distilling generalizable criteria from preference data, yet remains confined to text-only evaluation. In multimodal settings, AutoRubric-R1V [17] compiles consistent reasoning steps from successful trajectories into problem-specific rubrics for process-level supervision, but it is designed for vision-language reasoning, not generative policy optimization. Despite these advances, no prior method in multimodal generation adopts auto-generated rubrics as the reward for both evaluation and training[22, 52]. We address this gap by treating rubrics as the direct preference interface, instantiating them as explicit, prompt-conditioned criteria that govern evaluation and provide the reward signal for optimization. This reframes alignment from implicit scalar optimization to structured discrimination over verifiable criteria, yielding a more interpretable and robust reward.

### 3 Methodology

###### Verifiable Auto-Rubric Generation Pipeline

###### 1

Verification & Refinement ri

Hierarchical Rubric Structuring

###### Final Rubrics

Rubric Generation (x, yi+, yi-)

Architectural & Structural

Lighting & Realism

Rubric1:

rubric1 rubric2 rubric3 rubric4 rubric5 rubric6

- - Tip1
- - Tip2 Rubric1:
- - Tip1
- - Tip2 Rubric1:
- - Tip1
- - Tip2

[Figure 1]

[Figure 2]

Texture & Material

[Figure 3]

[Figure 4]

Anatomical & Plausibility

VLMs

Input Image Pairs (x, y1, y2)

[Figure 5]

Visual Coherence

AI Artifact Detection

[Figure 6]

Generated Rubric ri

Detail Legibility

Background Realism

[Figure 7]

Prompt x

Composition r1 Color r2 ......

Failed refine

Pass verified

Color, Aesthetic Harmony

Micro-Texture Rendering

...

rubric n

“a 10 year old farmgirl sitting on a fancy wooden chair.”

...

[Figure 8]

[Figure 9]

###### 2 Auto-Rubric as Reward (ARR) 3

###### Rubric Policy Optimization (RPO)

From Preference to Reward r(x, y) = +λ

Rubric Judge (Conditioned VLM)

“Current Policy” Frozen ARR Judger

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

y1 (preferred)

y2 (dispreferred)

[Figure 15]

if y is preferred

[y+ y-] (Binary Reward) RPO Optimization Objective

Gradient

[Figure 16]

[Figure 17]

###### r(x, y) = -λ

Rstructured

r

if y is dispreferred

- y1 = preferred (y＋)
- y2 = dispreferred (y－)

y′＋ y′－

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Per-step reward construction For prompt A, text condition and rubric R, are sampled

Advantage

Gradient Update Step:

###### Preference

[Figure 23]

[Figure 24]

New Test Pair (xnew, y′1, y′2)

Preferened uniformly

“Updated Policy”

[Figure 25]

distributed cross timesteps t

[Figure 26]

Figure 1: Overview of the ARR-RPO framework.

- 3.1 Problem Formulation

We formulate preference learning as estimating the optimal parameters of a probabilistic model Pθ that, given a prompt x and candidate outputs y+,y−, assigns higher likelihood to the response better

satisfying human intent. Preference alignment thus optimizes Pθ to capture and generalize human preferences, raising the central design question: how should the parameters θ be specified? We address this by decomposing the problem into ARR for evaluation and RPO for training (Figure 1).

Implicit Preference Modeling. For implicit preference modeling, given a pair of outputs (y+,y−) conditioned on the same input x, the human preference probability is typically defined using the Bradley-Terry (BT) model as follows:

exp(r∗(x,y+)) exp(r∗(x,y+)) + exp(r∗(x,y−))

P∗(y+ ≻ y− | x) =

(1)

where ∗ denotes the parameters corresponding to the true underlying human preference distribution. Here, r∗ represents the ideal scalar reward model that perfectly reflects human preferences. In practice, since the true human preference distribution is inaccessible, we typically work with a pairwise preference dataset D that approximately captures human judgments. We can then parameterize a reward model rϕ and estimate the true parameters ϕ∗ by solving the following optimization problem:

LR(rϕ,D) = −E(x,y+,y−)∼D log σ(rϕ(x,y+) − rϕ(x,y−)) (2) where σ is the logistic function.

Explicit Preference Modeling. In explicit preference modeling, we define the preference distribution by employing a VLM as a judge. Given a paired input (x,y+,y−), the LLM judge processes the prompt x along with the two candidate outputs and produces a binary preference decision that approximates the underlying human preference distribution Pθ:

Pθ(y+ ≻ y−|x) = Mθ(y+ ≻ y− | x,y+,y−,R), (3)

where R is a carefully pre-defined natural language rubric designed to enhance the VLM’s ability to discern subtle differences in response quality. Here, Mθ denotes the VLM enhanced by R, which serves as the judge and outputs a binary preference decision between the two candidates.

###### 3.2 Auto-Rubric as Reward

Let S be the space of all possible rubrics. We aim to find the optimal rubric R∗ that best approximates the underlying human preference distribution. Given an ideal preference model P∗ instantiated by a highly capable LLM judge, the optimal rubric can be formulated as:

R∗ = arg max R⊂S

N

log P∗(yi+ ≻ yi−|xi,R) (4)

i=1

Since the space of all possible rubric sets S is vast and discrete, directly optimizing the ideal objective is intractable. We therefore simplify the optimization target as selecting the best rubric subset:

R∗ ≈ arg max

R⊂DR

N

I[Mθ(yi+ ≻ yi− | xi,yi+,yi−,R) = correct], (5)

i=1

where DR is a finite set of candidate rubrics. In the remainder of this section, we detail our approach for automatically constructing high-quality rubrics from data and demonstrate how these auto-generated rubrics can serve as an interpretable and effective reward signal when applied to reinforcement learning tasks.

Verifiable Rubric Generation. Given a pairwise preference dataset D = {(xi,yi+,yi−)}Ni=1, we first generate a candidate rubric for each individual pair. For every pair (xi,yi+,yi−), an VLM is prompted to produce a detailed natural language rubric ri that explains why yi+ is preferred over yi−:

ri = Mgen(xi,yi+,yi−). (6)

To ensure quality, each generated rubric ri is then verified by a separate judgment step. The verifier checks whether the rubric consistently supports the original preference:

vi = Mverify(xi,yi+,yi−,ri). (7)

Because the verifier independently checks whether the generated rubric consistently recovers the original preference label, it acts as a weak safeguard against self-reinforcing errors: rubrics that fail this consistency test are refined or discarded, reducing the chance of amplifying idiosyncratic model biases that survive the initial generation step.

If verification fails (vi = false), we iteratively refine the rubric up to a predefined maximum number of attempts Tmax:

ri(t+1) = Mrefine(xi,yi+,yi−,ri(t)), t = 0,1,...,Tmax − 1. (8)

If the rubric still fails verification after Tmax refinement attempts, it is discarded. After processing all pairs in D, we obtain a set of verified rubrics:

DR = {ri | vi = true}. (9)

This verifiable generation process yields a high-quality, instance-specific rubric collection DR directly grounded in the preference dataset.

Hierarchical Rubric Structuring. After verification, the rubric set DR captures fine-grained, perinstance criteria but lacks the coherence required for consistent conditioning across arbitrary prompts.

We therefore prompt an LLM to consolidate DR into a single, hierarchically organized rubric. The LLM groups related criteria by semantic granularity and preference dimension, producing a compact

evaluation protocol. The resulting structured rubric Rstructured is directly reused as a system-prompt component for the judge and as a reward conditioning signal during optimization, removing the need for per-instance rubric regeneration at deployment. Formally,

Rstructured = Mstruct(DR), (10)

where Mstruct denotes the LLM prompted to perform hierarchical organization and prompt synthesis. See Appendix I for final rubric examples.

From Rubric to Reward. To successfully apply the auto-rubric method to reinforcement learning tasks, we need to convert the generated rubrics into a usable reward signal. Since the VLM judge produces binary preference decisions, we assign a positive constant reward to the preferred response y+ and a negative constant reward to the dispreferred response y−. Formally, given a prompt x and a pair of outputs (y+,y−), the reward for a candidate y is defined with respect to the other output y′ as:

+λ if Mθ(x,y,y′,R) prefers y, −γ otherwise,

r(x,y;y′) =

(11) where λ,γ > 0 are constant reward magnitudes and R denotes the learned rubric set.

###### 3.3 Rubric Policy Optimization

Having established a mechanism for generating high-quality rubrics and converting them into verifiable reward signals, we now introduce Rubric Policy Optimization (RPO), an online policy optimization algorithm that directly utilizes the rubric judge to guide the generative policy πθ.

Unlike conventional RLHF and prior rubric-based methods in multimodal generation that reduce criteria to scalar composites or auxiliary filters, RPO directly leverages the VLM judge’s binary preferences conditioned on explicit rubrics as the reward signal. For each generated sample, the preferred output y+ receives a positive constant reward +λ, while the dispreferred output y− receives −γ. This yields a dense per-step training objective that preserves the advantages of rubric-based evaluation while remaining compatible with standard policy gradient methods.

The resulting RPO objective is defined as:

LRPO(θ) = Eh∼D,{xi

0:T }2i=1∼πθ

2

- 1

- 2

i=1

T−1

1 T

min rti(θ)Ai,

t=0

(12)

clip(rti(θ),1 − ϵ,1 + ϵ)Ai − β DKL(πθ∥πref) .

where the importance ratio at each timestep is

πθ(xit−1 | xit,h) πθ

rti(θ) =

. (13)

(xit−1 | xit,h)

old

Per-step reward construction. For a given prompt h (which may include both text condition c and the current rubric R), we sample two trajectories from the current policy πθ. The VLM judge, conditioned on the learned rubric, produces a binary preference decision between the two trajectories. The winning trajectory is assigned advantage Aw = +λ and the losing one Al = −γ. This pertrajectory advantage is then uniformly distributed across all denoising (or generation) timesteps, providing a dense training signal that directly reflects rubric-guided human preference.

- Table 1: Evaluator performance across four preference benchmarks. Accuracy (%) denotes agreement with human preference labels. The best result in each column is bold. Blue-shaded rows indicate ARR; green values indicate absolute gains over the corresponding baseline VLM judge.

HPDv3 MM-RewardBench2 (T2I) MM-RewardBench2 (Edit) EditReward-Bench Acc. Acc. Acc. Acc.

Method

###### Trained Reward Model

PickScore 65.6 58.6 — ImageReward 58.6 54.0 — UnifiedReward 66.0 59.8 — UnifiedReward-Thinking 68.1 66.0 — HPSv3 76.9 60.2 — EditReward — — 67.2 56.45

###### VLM-as-Judge w/o ARR

Qwen3-VL-8B 67.2 57.6 59.2 54.01 w/ ARR 70.2 (± 0.2)↑3.0 62.7 (± 0.2)↑5.1 65.5 (± 0.3)↑6.3 57.22 (± 0.1)↑3.21

GPT-5 72.4 70.5 73.8 57.53 w/ ARR 76.1 (± 0.2)↑3.7 74.7 (± 0.4)↑4.2 77.5 (± 0.3)↑3.7 61.01 (± 0.1)↑3.48

Gemini 3.1 Pro 76.6 75.1 77.4 61.23 w/ ARR 78.3 (± 0.1)↑1.7 78.9 (± 0.2)↑3.8 79.2 (± 0.2)↑1.8 63.27 (± 0.2)↑2.04

Online optimization and robustness. RPO is fully online: each iteration samples prompts from D, generates two candidates from πθ, evaluates them via the rubric judge, and applies the gradient of LRPO(θ). Because rewards come from a frozen VLM judge conditioned on explicit rubrics rather than a trainable scalar model, RPO helps mitigate reward hacking. Rubrics are regenerated per prompt–output pair, so the optimization target adapts naturally to the evolving distribution of πθ, conferring robustness against distributional shift. PPO-style clipping and KL regularization further stabilize training and enable exploration aligned with the multi-dimensional criteria in the rubrics.

### 4 Experiments

We evaluate ARR as a preference evaluator and as a structured reward for generative policy optimization. Experiments on multimodal understanding, text-to-image generation, and image editing benchmarks compare against trained reward models and direct VLM judges to assess gains in evaluative reliability and downstream performance.

###### 4.1 Experimental Setup

Evaluation Benchmarks. Evaluator fidelity is measured on three established testbeds: MMRewardBench2 [16], which provides fine-grained diagnostic splits across multimodal reward scenarios; HPDv3 (test set) [28], a large-scale text-to-image preference corpus comprising 14,400 pairwise human judgments; and EditReward-Bench [43], specifically curated to probe instruction adherence in image editing. For generative quality assessment, we adopt GenEval [11], DPG-Bench[15], TIIF(test-mini-short)[40], and UniGenBench++[37] for text-to-image synthesis, complemented by GEdit-Bench[24] and ImgEdit[49] for editing tasks.

Baselines and Implementation. For human preference evaluation, we compare against a suite of state-of-the-art trained reward models, including HPSv3 [28], PickScore [19], ImageReward [47], UnifiedReward[39] and UnifiedReward-Thinking [38], and EditReward [43], alongside representative VLM judges such as Qwen3-VL [2], GPT-5 [33], and Gemini 3.1 Pro [12].

Following the common practice in recent multimodal alignment and generation research [16, 22, 34], we adopt FLUX.1-dev [20] and Qwen-Image-Edit-2509 [41] as base models for image generation and editing, respectively. We perform post-training with RPO on LoRA-adapted versions of these models. Training prompts are drawn from ShareGPT-4o-Image [7]. Unless otherwise specified, ARR instantiates five prompt-conditioned rubrics per input using a frozen VLM, which are used to score candidate images. We further contextualize results against leading contemporary generative models.

- Table 2: Generative performance across T2I and Image Editing benchmarks. Blue-shaded rows mark ARR-RPO variants; green values indicate absolute gains over the corresponding baseline.

Text-to-Image Image Editing GenEval DPG-Bench TIIF UniGenBench++ GEdit-Bench ImgEdit

Method

Short Long Specialist Model (T2I)

Emu3 0.54 80.60 — 45.42 50.59 — JanusFlow 0.63 79.68 — 47.10 54.80 — FLUX.1-Dev 0.66 83.84 71.09 60.97 69.42 — DALLE-3 0.67 83.50 74.96 68.85 70.82 — Show-o2 0.76 86.14 — 61.90 70.33 — OmniGen2 0.80 83.57 — 63.09 71.39 — BAGEL 0.82 85.07 71.50 59.91 71.26 — —

###### ARR-RPO / T2I (OURS)

w/ RPO-Qwen3vl-8B-ARR 0.74↑0.08 85.03↑1.19 74.92↑3.83 64.17↑3.20 71.82↑2.40 — w/ RPO-GPT-5-ARR 0.78↑0.12 85.41↑1.57 76.18↑5.09 65.36↑4.39 72.41↑2.99 — w/ RPO-Gemini 3.1 Pro-ARR 0.80↑0.14 85.76↑1.92 76.85↑5.76 65.89↑4.92 72.93↑3.51 — —

###### Specialist Model (Image Editing)

Instruct-Pix2Pix — — — — — 3.68 1.88 AnyEdit — — — — — 3.21 2.45 Step1X-Edit — — — — — 6.97 3.06 Qwen-Image-Edit-2509 — — — — — 7.54 4.35 UniWorldv2 — — — — — 7.76 4.48

###### ARR-RPO / Image Editing (OURS)

w/ RPO-Qwen3vl-8B-ARR — — — — — 7.66↑0.12 4.38↑0.03 w/ RPO-GPT-5-ARR — — — — — 7.72↑0.18 4.40↑0.05 w/ RPO-Gemini 3.1 Pro-ARR — — — — — 7.85↑0.31 4.43↑0.08

###### 4.2 Human Preference Quality

We evaluate ARR as a preference evaluator on three standard benchmarks: HPDV3[28], which provides 1.17M human pairwise comparisons for text-to-image; MM-RewardBench2[16], with 4,000 expert-annotated preference pairs spanning four tasks; and EditReward-Bench, covering 13 subtasks of instruction-guided editing. For each benchmark, we report pairwise preference accuracy, defined as the fraction of test pairs where the model’s predicted preference matches the human judgment.

Results. Table 1 reports preference accuracy. Pairwise reward models specialize narrowly (e.g., HPSv3 drops from 76.9% on HPDv3 to 60.2% on MM-RewardBench2 T2I; EditReward falls from 67.2% to 56.5% on the broader EditReward-Bench), while direct VLM judges generalize better yet still struggle on challenging splits (Gemini 3.1 Pro: 75.1–77.4% on the first three columns but only 61.2% on EditReward-Bench). ARR conditioning consistently improves all judges by 1.7–6.3 points, with Gemini 3.1 Pro + ARR reaching state-of-the-art on three of four benchmarks. Critically, base VLMs exhibit severe positional bias (∆ = 30.2–34.6; Table 5); ARR reduces this gap to 27.8–31.6 (zero-shot) and to 8.9–10.3 with guidance. Gains persist across model families (Table 6), confirming that rubric quality, not generator-judge co-adaptation, drives results. Full results are in Appendices 10.

Takeaway: Rubric conditioning does not merely improve accuracy; it reframes evaluation from preference matching to criteria-aligned verification. By externalizing standards prior to comparison, ARR mitigates latent biases in implicit preference models, yielding structured, task-adaptive decisions that directly support robust reward modeling.

###### 4.3 Image Generation and Editing Performance

We evaluate ARR-RPO on six benchmarks: GenEval [11], DPG-Bench [15], TIIF [40], and UniGenBench++ [37] for text-to-image generation; GEdit-Bench [24] and ImgEdit [49] for instruction-guided image editing. ARR-RPO fine-tunes FLUX.1.dev [20] and Qwen-Image-Edit-2509 [41] using ARR-

###### ARR-RPO Performance Gains over Specialist Models

Comparing against FLUX.1-dev (T2l) and Qwen-lmage-Edit-2509 (Image Editing)

Specialist Models Baseline Model ARR-RPO Variants

Best Model

TEXT-TO-IMAGE GENERATION (T2I)

###### Geneval DPG-Bench TIIF UniGenBench++ (short) UniGenBench++ (long)

100

100

100

1.00

100

86.14 85.07 83.84 85.03 85.41 85.76

0.78 0.80

0.82

0.81

0.74

74.92 76.18 76.85

74.96 71.50 71.09

71.26 71.39 69.42 71.82 72.41 72.93

68.85

75

75

75

75

0.75

0.66

63.09 64.17 65.36 65.89

60.97

50 50

50

0.50

50

25

25

25

25

0.25

0

0

0

0

0

Gemini 3.1 ProARR

Bagel Qwen 3VL8BARR

GPT5 -ARR

Gemini 3.1 ProARR

Qwen 3VL8BARR

Omni Gen2

GPT5 -ARR

FLUX. 1.dev

Omni Gen2

Gemini 3.1 ProARR

DALL E-3

Bagel Qwen 3VL8BARR

GPT5 -ARR

FLUX. 1.dev

Gemini 3.1 ProARR

Bagel Qwen 3VL8BARR

DALL E-3

FLUX. 1.dev

GPT5 -ARR

Gemin i3.1 ProARR

Qwen 3VL8BARR

BLIP 3o-4B

GPT5 -ARR

Show

Bagel FLUX. 1.dev

FLUX. 1.dev

-o2

Image Editing

###### GEdit-Bench ImgEdit

6.0

10.0

7.76 7.66 7.72 7.85

7.54

4.48

4.38 4.40 4.43

6.97

4.35

4.5

7.5

3.06

2.45

3.0

5.0

1.88

3.68 3.21

- 0

- 1.5

2.5

0

AnyEd it

Step1 XEdit

QwenImage -Edit

UniW orldv2

Qwen 3VL8BARR

Gemini 3.1 ProARR

InstructPix2Pix

GPT5 -ARR

QwenImage -Edit

InstructPix2Pix

AnyEd it

Step1 XEdit

Qwen 3VL8BARR

Gemini 3.1 ProARR

GPT5 -ARR

UniW orldv2

Figure 2: Performance comparison of ARR-RPO variants against specialist models across text-toimage generation (top) and image editing (bottom) benchmarks.

Text-to-Image Generation

Ours (w/ ARR-RPO)

###### Auto-Rubric Theme

Baseline (w/o ARR-RPO)

[Figure 27]

[Figure 28]

[Figure 29]

Architecture Fidelity

###### Input Prompt:

Geometry, no distortion Consistent structural details

A squirrel lowering its head while eating a banana, highly realistic style, detailed fur texture, natural lighting, sharp focus, lifelike scene.

[Figure 30]

Lighting & Shadows

Consistent light intensity Realistic reflections

·

· ·

·

w/ Gemini 3.1 Pro

w/ FLUX.1.dev

Image Editing

Input Prompt: Baseline (w/o ARR-RPO)

Auto-Rubric Theme

Ours (w/ ARR-RPO)

[Figure 31]

[Figure 32]

Put a pond next to the cows.

[Figure 33]

Edit Accuracy

Matches instruction precisely

Source Image

No missing or extra edits

[Figure 34]

[Figure 35]

Content Presevation

Unedited areas remain intact Structure preserved

·

·

w/ Qwen-Image-Edit w/ Gemini 3.1 Pro

###### Figure 3: Text-to-Image and Image Editing Examples (ARR-RPO Gemini 3.1 Pro).

generated rubrics as binary reward signals. We instantiate ARR with three VLMs, Qwen3-VL-8B [2], GPT-5 [33], and Gemini 3.1 Pro [12], to examine how rubric quality scales with judge capability.

Results. Figure 2 and Table 2 report generative performance. Two patterns emerge. First, ARR-RPO consistently outperforms specialist baselines. For T2I, optimizing FLUX.1.dev with ARR rubrics lifts GenEval (0.66→0.80), DPG-Bench (83.84→85.76), TIIF (71.09→76.85), and UniGenBench++ Short (60.97→65.89). In editing, ARR-RPO elevates Qwen-Image-Edit-2509 on GEdit-Bench (7.54→7.85) and ImgEdit (4.35→4.43). Second, generated samples (Figure 3) exhibit marked improvements in visual quality and edit fidelity, aligning more closely with the multidimensional nature of human preferences. See Appendix 8 for full results.

Takeaway: Rubrics externalize preference structure in a way that makes evaluation directly usable as reward. By grounding policy gradients in rubric-conditioned decisions rather than scalar proxies, ARR-RPO replaces opaque optimization with criteria-aligned improvement, enabling a direct transfer of evaluator fidelity into generative quality without reward model training or architectural change.

###### 4.4 Ablation Analysis

###### Forward Reverse Preference Gap

|[Figure 36]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

34.6

Qwen3-VL-8B

###### Cross-Model Rubric Transfer

31.6

+ ARR

30

[Figure 37]

3.0

10.3

+ ARR w/ guide

(Forward-Reverse)

75.9

None (direct)

25

32.6

2.5

GPT-5

vsBaseline

77.5

+1.6

Qwen3-VL-8B

28.2

+ ARR

2.0

20

9.3

+ ARR w/ guide

1.5

78.3

+2.4

GPT-5

30.2

Gemini 3.1 Pro

1.0

15

27.8

+ ARR

0.5

Forward Reverse

79.2

+3.3

Gemini 3.1 Pro

8.9

+ ARR w/ guide

10

0.0

76 77 78 79

50 60 70 80 90

Accuracy (%)

Score

(a)

(b)

Figure 4: Ablation studies on ARR. (a) Forward–Reverse preference gaps across evaluators. (b) Cross-model rubric transfer with a fixed judge.

As shown in Figure 4(a), substantial positional bias (∆ = 30.2 to 34.6) remains consistent across model scales in the absence of rubric conditioning. This suggests that the bias is not primarily due to insufficient model capacity, but is instead rooted in how preferences are implicitly encoded. Zero-shot ARR provides a modest reduction in bias (∆ decreases by 3.0 to 4.8), while human-guided rubrics lead to a much more pronounced improvement (∆ reduced to 8.9 to 10.3). These results indicate that making evaluation criteria explicit can significantly improve stability.

Figure 4(b) further shows that rubrics generalize across different model families (see Appendix C). Even when applied to weaker generators, transferred rubrics recover more than half of the performance gap compared to same-family settings. This observation suggests that the effectiveness of ARR is closely related to the quality and structure of the rubric itself, rather than reliance on tight coupling between the generator and the evaluator.

Furthermore, this interpretation is supported by the rubric cardinality ablation (Appendix D), where increasing rubric dimensionality consistently improves accuracy, indicating that ARR’s gains arise from both finer-grained factorization of preference structure and the quality of the resulting rubric content, rather than model capacity or evaluator–generator coupling.

### 5 Conclusion

We present a unified ARR and RPO framework bridging multimodal preference evaluation and generative alignment. While prevailing approaches rely on implicit, entangled scalar signals that obscure underlying criteria and introduce systematic biases, ARR automatically generates instanceconditioned rubrics by prompting VLMs to externalize latent human preferences into explicit, interpretable criteria. These rubrics provide structured, factorized reward signals for RPO, enabling contrastive preference learning with fine-grained supervision across independent quality dimensions. Together, ARR and RPO replace opaque scalar rewards with explicit, composable criteria, consistently improving both evaluation reliability and generation quality without additional supervision or architectural modifications. This externalization of preference structure offers a principled, scalable pathway toward compositional alignment with nuanced, multidimensional human intent.

### References

- [1] Zachary Ankner, Mansheej Paul, Brandon Cui, Jonathan D Chang, and Prithviraj Ammanabrolu. Critique-out-loud reward models. arXiv preprint arXiv:2408.11791, 2024.
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. arXiv preprint arXiv:2310.07685, 2023.
- [4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [6] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.
- [7] Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025.
- [8] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [9] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining, 2025.
- [10] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023.
- [11] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [12] Google DeepMind. Gemini 3.1 Pro - Model Card. https://deepmind.google/models/ model-cards/gemini-3-1-pro/, feb 2026.
- [13] Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, and Chris Kedzie. Llmrubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13806–13834, 2024.
- [14] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022.
- [15] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.
- [16] Yushi Hu, Reyhane Askari-Hemmat, Melissa Hall, Emily Dinan, Luke Zettlemoyer, and Marjan Ghazvininejad. Multimodal rewardbench 2: Evaluating omni reward models for interleaved text and image. arXiv preprint arXiv:2512.16899, 2025.

- [17] Mengzhao Jia, Zhihan Zhang, Ignacio Cases, Zheyuan Liu, Meng Jiang, and Peng Qi. Autorubric: Rubric-based generative rewards for faithful multimodal reasoning, 2026.
- [18] Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al. Prometheus: Inducing fine-grained evaluation capability in language models. In The Twelfth International Conference on Learning Representations, 2023.
- [19] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.
- [20] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.
- [21] Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, et al. Holistic evaluation of text-to-image models. Advances in Neural Information Processing Systems, 36:69981–70011, 2023.
- [22] Fan Li, Chonghuinan Wang, Lina Lei, Yuping Qiu, Jiaqi Xu, Jiaxiu Jiang, Xinran Qin, Zhikai Chen, Fenglong Song, Zhixin Wang, Renjing Pei, and Wangmeng Zuo. Hp-edit: A humanpreference post-training framework for image editing, 2026.
- [23] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Feize Wu, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, et al. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025.
- [24] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [25] Yixin Liu, Yue Yu, DiJia Su, Sid Wang, Xuewei Wang, Song Jiang, Bo Liu, Arman Cohan, Yuandong Tian, and Zhengxing Chen. Examining reasoning llms-as-judges in non-verifiable llm post-training. arXiv preprint arXiv:2603.12246, 2026.
- [26] Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Liu, et al. Editscore: Unlocking online rl for image editing via high-fidelity reward modeling. arXiv preprint arXiv:2509.23909, 2025.
- [27] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7739–7751, 2025.
- [28] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095, 2025.
- [29] Tianjun Pan, Xuan Lin, Wenyan Yang, Qianyu He, Shisong Chen, Licai Qi, Wanqing Xu, Hongwei Feng, Bo Xu, and Yanghua Xiao. Rubriceval: A rubric-level meta-evaluation benchmark for llm judges in instruction following. arXiv preprint arXiv:2603.25133, 2026.
- [30] Aditya Pathak, Rachit Gandhi, Vaibhav Uttam, Arnav Ramamoorthy, Pratyush Ghosh, Aaryan Raj Jindal, Shreyash Verma, Aditya Mittal, Aashna Ased, Chirag Khatri, et al. Rubric is all you need: Improving llm-based code evaluation with question-specific rubrics. In Proceedings of the 2025 ACM Conference on International Computing Education Research V. 1, pages 181–195, 2025.
- [31] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023.

- [32] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.
- [33] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.
- [34] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization, 2023.
- [35] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, et al. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, 2024.
- [36] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [37] Yibin Wang, Zhimin Li, Yuhang Zang, Jiazi Bu, Yujie Zhou, Yi Xin, Junjun He, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Unigenbench++: A unified semantic evaluation benchmark for text-to-image generation, 2026.
- [38] Yibin Wang, Zhimin Li, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Unified multimodal chain-of-thought reward model through reinforcement fine-tuning, 2025.
- [39] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.
- [40] Xinyu Wei, Jinrui Zhang, Zeqing Wang, Hongyang Wei, Zhen Guo, and Lei Zhang. Tiif-bench: How does your t2i model follow your instructions?, 2025.
- [41] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [42] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [43] Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, and Wenhu Chen. Editreward: A human-aligned reward model for instruction-guided image editing. arXiv preprint arXiv:2509.26346, 2025.
- [44] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [45] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.
- [46] Lipeng Xie, Sen Huang, Zhuo Zhang, Anni Zou, Yunpeng Zhai, Dingchao Ren, Kezun Zhang, Haoyuan Hu, Boyin Liu, Haoran Chen, et al. Auto-rubric: Learning from implicit weights to explicit rubrics for reward modeling. arXiv preprint arXiv:2510.17314, 2025.
- [47] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.
- [48] Seonghyeon Ye, Doyoung Kim, Sungdong Kim, Hyeonbin Hwang, Seungone Kim, Yongrae Jo, James Thorne, Juho Kim, and Minjoon Seo. Flask: Fine-grained language model evaluation based on alignment skill sets. arXiv preprint arXiv:2307.10928, 2023.

- [49] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark, 2025.
- [50] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea, 2025.
- [51] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.
- [52] Xiangyu Zhao, Peiyuan Zhang, Junming Lin, Tianhao Liang, Yuchen Duan, Shengyuan Ding, Changyao Tian, Yuhang Zang, Junchi Yan, and Xue Yang. Trust your critic: Robust reward modeling and reinforcement learning for faithful image editing and generation. arXiv preprint arXiv:2603.12247, 2026.
- [53] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Appendix

- A Experimental Setup Details

This section provides a comprehensive account of the datasets, evaluation protocols, model configurations, training hyperparameters, and computational resources employed throughout the paper. All experiments were conducted on a cluster of 8 NVIDIA H100 (80GB SXM5) GPUs.

###### A.1 Datasets

We evaluate on two families of benchmarks: those designed for assessing preference evaluation fidelity, and those measuring generative quality in text-to-image synthesis and instruction-guided image editing.

Preference Evaluation Benchmarks.

- • HPDv3: A large-scale human preference dataset for text-to-image generation comprising 1.17 million pairwise comparisons collected from diverse user prompts. Each pair presents two images generated from the same prompt, with one image annotated as preferred. We use the official test split and report pairwise preference accuracy.
- • MM-RewardBench2: A diagnostic benchmark with 4,000 expert-annotated pairwise instances spanning four tasks: text-to-image alignment (T2I), image editing (Edit), visual question answering, and compositional understanding. We report accuracy on the T2I and Edit subtasks separately.
- • EditReward-Bench: A fine-grained benchmark assessing instruction adherence in image editing, encompassing 13 subtasks with expert human annotations. Each subtask targets a distinct editing operation (e.g., object addition, texture transfer, style modification).

Generative Benchmarks.

- • GenEval: Assesses compositional object accuracy in T2I synthesis by verifying whether generated images contain correct objects and attributes as specified in the prompt. Accuracy is computed via object detection against structured prompt decompositions.
- • DPG-Bench: Measures alignment with dense, paragraph-length prompts through structured question answering. We report the overall alignment score averaged across all test prompts.
- • TIIF: Evaluates instruction fidelity across three difficulty tiers (simple, complex, compositional), providing a graded measure of instruction-following capacity. We report the macro-average across tiers.
- • UniGenBench++: Probes semantic consistency with both Short and Long prompt variants, measuring coherence with brief versus detailed textual descriptions.
- • GEdit-Bench: A real-world image editing benchmark comprising naturalistic user instructions. Outputs are evaluated by GPT-5 on a 1–10 scale covering instruction adherence, image quality, and preservation of non-targeted regions.
- • ImgEdit: Evaluates single-turn and multi-turn instruction-driven editing quality using automated metrics and human assessments. We report the composite score averaged across categories and turn depths.

RL Training Datasets

• ShareGPT-4o-Image: A large-scale multimodal corpus for text-to-image generation and editing, containing around 92K high-quality GPT-4o-synthesized samples, including both text-to-image and text-guided image editing pairs, which we use to construct training and evaluation prompts.

###### A.2 Evaluation Protocols

Preference Accuracy. For all pairwise preference evaluators, we report preference accuracy: the proportion of test pairs for which the model assigns a higher reward (or preference) to the humanpreferred image. To probe positional robustness, each test pair is evaluated in both its original

(forward) and permuted (reverse) order. The gap between forward and reverse accuracy quantifies the degree of position bias.

Generative Evaluation. For text-to-image generation, FLUX.1-dev uses sampling with 30 sampling steps and a guidance scale of 3.5. For image editing, Qwen-Image-Edit-2509 performs inference with 50 sampling steps and a classifier-free guidance (CFG) scale of 4.0. All benchmark evaluations are conducted using the official evaluation scripts without modification.

###### A.3 Model Configurations

ARR Instantiation. Unless otherwise specified, ARR employs a frozen VLM to synthesize five prompt-conditioned rubrics per input instance. The generation meta-prompt instructs the VLM to decompose the given text prompt into independent evaluative dimensions (e.g., object presence, attribute accuracy, spatial layout, aesthetic quality, instruction adherence), formulating each dimension as a verifiable binary criterion. Rubric synthesis, verification, and refinement are all conducted at inference time without gradient updates to the judge VLM.

For the ARR w/ guide variant, the meta-prompt is augmented with a fixed set of human-curated preference exemplars drawn from a held-out subset of the training split of each benchmark. These exemplars consist of (prompt, preferred image, dispreferred image, preference rationale) tuples and are embedded verbatim as in-context demonstrations. No fine-tuning of the VLM is performed; the exemplars serve solely as semantic anchors.

Rubric Verification. Each candidate rubric ri generated for a preference pair (xi,yi+,yi−) is passed to a separate frozen verifier call, which checks whether applying ri as a scoring criterion yields the correct preference decision on the generating pair. If verification fails, we invoke a refinement pass (up to Tmax = 5 iterations) that presents the verifier’s critique alongside the original rubric and requests a revised formulation. Rubrics that remain unverified after Tmax attempts are discarded. In our experiments, approximately 87% of initial rubrics pass verification without refinement, and fewer than 4% are ultimately discarded.

Hierarchical Structuring. The verified rubric set DR is organized into a hierarchical prompt structure by a final synthesis call. This call groups criteria by semantic level (coarse: overall alignment; mid: compositional attributes; fine: local details) and orders them by estimated diagnostic value. The resulting structured rubric Rstructured is formatted as a numbered list of axis definitions, each accompanied by a brief operationalization clause. This structure is passed verbatim to the judge VLM as the evaluation conditioning context.

###### A.4 Generative Training: RPO Hyperparameters

RPO is a reinforcement learning algorithm adapted for denoising diffusion policies in both text-toimage generation (T2I) and image editing. Key hyperparameters are reported in Table 3.

- Table 3: RPO training hyperparameters for T2I (FLUX.1.dev) and image editing (Qwen-Image-Edit2509).

Hyperparameter T2I (FLUX.1.dev) Editing (Qwen-Image-Edit)

Base learning rate 5 × 10−5 1 × 10−5 Batch size 32 16 Candidates per prompt 2 2 Clip ϵ 0.2 0.2 KL coefficient β 0.01 0.02 Positive reward λ 1.0 1.0 Negative reward γ 0.1 0.1 Denoising steps 8 10 Optimizer AdamW AdamW Gradient clipping 1.0 1.0 Trained Parameters LoRA (rank=16) LoRA (rank=32)

Training prompts are sampled uniformly from ShareGPT4o-Image, with no data augmentation applied. At each online iteration, two candidate outputs are generated from the current policy πθ,

evaluated by the frozen ARR judge conditioned on the structured rubric, and the resulting binary advantage A ∈ {+λ,−γ} is uniformly distributed across all generation timesteps.

### B Auto-Rubric as Reward (ARR) Details

This section elaborates on the technical instantiation of Auto-Rubric as Reward (ARR), complementing the concise description provided in Section 3.2 of the main text. We provide a granular account of the rubric generation pipeline, the verification and refinement protocol, the hierarchical structuring mechanism, and a comparative characterization of ARR within the broader landscape of reward modeling approaches.

###### B.1 Rubric Generation Pipeline

ARR synthesizes prompt-conditioned rubrics through a three-stage process: generation, verification, and structuring. Each stage is implemented as a frozen (multimodal) large language model call, ensuring that the judge VLM remains unmodified throughout.

###### B.1.1 Per-Instance Rubric Generation

Given a preference pair (x,y+,y−) drawn from a pairwise dataset D, we prompt the generator model Mgen to produce a natural language explanation of why y+ is preferred over y−. The meta-prompt explicitly instructs the model to:

- • Decompose the preference into independent, verifiable quality axes (e.g., semantic fidelity, attribute accuracy, spatial coherence).
- • Formulate each axis as a binary criterion that can be evaluated without reference to the paired candidate.
- • Avoid holistic or comparative language that presupposes knowledge of both outputs.

The resulting rubric ri is a structured, axis-wise decomposition of the preference rationale.

###### B.1.2 Verification and Refinement

Each candidate rubric ri is validated by a separate verifier call Mverify. The verifier receives the original preference pair (x,y+,y−) and the generated rubric ri, and is tasked with determining whether applying ri as an evaluation protocol correctly identifies y+ as the preferred output. The verification outcome is binary:

vi =

true if Mverify(x,y+,y−,ri) confirms the original preference, false otherwise.

If verification fails, we invoke a refinement pass that supplies the verifier’s critique alongside ri to a refinement model Mrefine, which produces a revised rubric ri(t+1). Refinement iterates up to Tmax = 5 times; rubrics that remain unverified after this budget are discarded. Empirically, 87% of initial rubrics pass verification without refinement, and fewer than 4% are ultimately discarded, attesting to the stability of the generation process.

###### B.1.3 Hierarchical Structuring

The verified rubric collection DR = {ri | vi = true} is subsequently aggregated into a single, hierarchically structured prompt Rstructured. The structuring model Mstruct organizes the rubrics into different evaluation dimensions, including:

- • Overall alignment: Measures the global consistency between the generated output and the prompt intent.
- • Compositional structure: Evaluates the presence and relationships of key elements, such as object presence and spatial relations.
- • Fine-grained fidelity: Focuses on local details and editing-specific accuracy.
- • Other dimensions: ...

- Table 4: Paradigm shift from implicit to explicit reward parameterization. Comparison of multimodal reward modeling approaches along key operational axes. Pointwise and pairwise reward models require extensive preference data and yield opaque scalar signals. Mix refers to models that support both pairwise and pointwise outputs. ARR uniquely combines zero-shot rubric generation with binary scoring, eliminating training overhead entirely.

Method Mode Reward Form Hacking Risk Interpretability Data Requirement PickScore Pointwise Scalar High Low Large HPSv3 Pointwise Scalar High Low Large ImageReward Pointwise Scalar Medium Low Large UnifiedReward Mix Scalar Medium Low Large VLM-as-Judge Pairwise (Mix) Binary Medium Medium Zero-shot ARR (Ours) Pairwise (Mix) Binary Low High Zero-shot

The final structured rubric is formatted as a numbered list of evaluation dimensions, where each dimension groups a set of corresponding rubrics. For each dimension, a brief operationalization clause is provided to clarify its focus. This structured rubric is directly used as the conditioning context for the judge VLM during both evaluation and RPO training.

###### B.2 Comparative Characterization of Reward Modeling Paradigms

- Table 4 situates ARR within the landscape of contemporary multimodal reward modeling. We contrast ARR against representative pointwise reward models, pairwise reward models, and direct VLM judges. The comparison spans five dimensions: evaluation mode, reward representation, susceptibility to reward hacking, interpretability of the reward signal, and data requirements for training or deployment.

Key Distinctions. ARR differs from prior approaches in four critical respects:

- 1. Zero-shot rubric generation: ARR synthesizes rubrics on-the-fly from frozen VLMs, enabling immediate deployment in new domains without additional data collection or task-specific supervision.
- 2. Holistic, rubric-conditioned decision interface: Rather than aggregating independently scored criteria post hoc, ARR formulates evaluation as a single rubric-conditioned judgment, where all dimensions are jointly considered in a pairwise comparison. This preserves intercriterion dependencies and avoids inconsistencies introduced by independent scoring and aggregation.
- 3. Training-free reward interface: ARR operates without any parameter updates to the judge model, eliminating the computational and data overhead associated with training pointwise or pairwise reward models, while retaining strong generalization through the underlying VLM.
- 4. Data-efficient rubric induction: Across all experiments, high-quality rubrics are constructed from as few as 100 preference pairs drawn from ShareGPT-4o-Image. This demonstrates that ARR can recover structured, task-relevant evaluation criteria with minimal supervision, achieving competitive performance with substantially lower data requirements than existing methods.

These properties collectively establish ARR as a lightweight, interpretable, and bias-resilient alternative to both implicit reward models and manually curated rubric systems. Importantly, since ARR builds on a VLM-as-a-judge paradigm, the rubric-conditioned interface is inherently flexible and can be extended beyond pairwise comparison to pointwise scoring or listwise ranking settings. In this work, we focus on the pairwise formulation to isolate and evaluate the robustness of rubric-based decision making under minimal reward hacking risk, providing a controlled setting for studying structured, generative reward modeling in multimodal alignment.

### C Ablations on Position Bias in ARR

###### C.1 Setup

Position bias refers to the systematic tendency of a pairwise preference evaluator to favor whichever candidate appears in a fixed ordinal position (e.g., always preferring Image A when presented first), irrespective of actual quality. This constitutes a critical failure mode: an evaluator that achieves high accuracy in the standard presentation order but collapses under permutation produces a spurious reward signal entangled with input ordering rather than genuine quality.

To isolate this effect, we evaluate each pairwise evaluator on the HPDv3 test set under two conditions: (i) forward order, where images appear in the original benchmark order; and (ii) reverse order, where the two images are swapped. We report forward accuracy (%), reverse accuracy (%), and their arithmetic mean (Avg). An ideal, unbiased evaluator would achieve identical accuracy under both conditions. The quantity ∆ = Accfwd − Accrev serves as our primary measure of positional instability; larger ∆ indicates stronger position bias.

###### C.2 Results

Table 5 reports the position bias ablation across three base VLMs and their ARR-augmented variants. All experiments are conducted on the HPDv3 test set.

Table 5: Position bias ablation on HPDv3. Forward and reverse accuracy (%) are measured by swapping the order of the two images in each preference pair. ∆ = Fwd − Rev quantifies positional instability. ARR variants reduce ∆ consistently; ARR w/ guide provides the strongest stabilization. Rows are grouped by base model.

Method Forward Reverse Avg ∆

Qwen3-VL-8B 84.5 49.9 67.2 34.6 + ARR 86.0 54.4 70.2 31.6 + ARR w/ guide 90.1 79.8 85.0 10.3

GPT-5 88.7 56.1 72.4 32.6 + ARR 90.2 62.0 76.1 28.2

+ ARR w/ guide 93.4 84.1 88.8 9.3

Gemini 3.1 Pro 91.7 61.5 76.6 30.2 + ARR 92.2 64.4 78.3 27.8

+ ARR w/ guide 95.2 86.3 90.8 8.9

###### C.3 Analysis

- Table 5 reveals four consistent patterns:

Base VLMs exhibit severe and scale-invariant position bias. Across all three base models, the gap between forward and reverse accuracy is extreme: ∆ = 34.6 for Qwen3-VL-8B, 32.6 for GPT-5, and 30.2 for Gemini 3.1 Pro. Crucially, this gap does not diminish with model capability: the most capable model (Gemini 3.1 Pro) yields a marginally smaller but still operationally severe ∆ of 30.2. This confirms that positional instability is a structural deficiency tied to the implicit parameterization of preference knowledge, not a capacity limitation that resolves with scale.

Zero-shot ARR yields consistent but moderate debiasing. Conditioning the VLM on autogenerated rubrics reduces ∆ by 3.0–4.8 points across all three models (e.g., 34.6 → 31.6 for Qwen3-VL-8B). The mechanism is interpretable: by requiring the model to commit to explicit evaluation criteria before inspecting the candidates, ARR partially anchors the judgment in criterionlevel evidence rather than holistic gestalt impressions susceptible to ordering heuristics. However, a substantial gap persists, indicating that self-generated rubrics alone do not fully overcome the structural mismatch between latent preference encoding and stable pairwise judgment.

Preference-conditioned ARR provides qualitatively stronger stabilization. ARR w/ guide reduces ∆ dramatically, to 10.3, 9.3, and 8.9 for the three base models respectively, corresponding

to reductions of 24.3, 23.3, and 21.3 points relative to the unaugmented baseline. The effect on reverse accuracy is particularly striking: Qwen3-VL-8B improves from 49.9% (near-random on reversed pairs) to 79.8%, indicating that human preference exemplars substantially enhance the model’s capacity to identify quality differences in an order-agnostic manner. This suggests that the key failure mode in unaugmented VLM judges is not an inability to perceive relevant features, but rather an inability to stably weight them independently of presentation order, a failure that explicit, human-grounded rubrics can partially correct.

Residual bias remains non-trivial. Even under ARR w/ guide with Gemini 3.1 Pro (∆ = 8.9), meaningful positional instability persists. A perfectly unbiased evaluator would achieve ∆ = 0. This residual gap underscores that current VLMs do not yet fully ground preference evaluation in stable criteria, and that stronger human preference guidance amplifies the effect of ARR rather than eliminating the need for it.

###### C.4 Cross-Model Rubric Transfer

- Table 6: Cross-model rubric transfer on HPDv3. The judge is fixed to Gemini 3.1 Pro; only the rubric generator varies. The direct baseline uses no rubric. Accuracy (%) denotes agreement with human preference labels.

Rubric Generator Judge Accuracy (%)

None (direct) Gemini 3.1 Pro 75.9 Qwen3-VL-8B Gemini 3.1 Pro 77.5 GPT-5 Gemini 3.1 Pro 78.3 Gemini 3.1 Pro Gemini 3.1 Pro 79.2

Cross-model rubric transfer. To further verify that ARR does not rely on same-family co-adaptation, we fix Gemini 3.1 Pro as the judge and generate rubrics using Qwen3-VL-8B, GPT-5, and Gemini 3.1 Pro. Table 6 reports accuracy on HPDv3. Even rubrics from the weakest generator, Qwen3-VL8B, improve accuracy from 75.9% (direct) to 77.5%, closing more than half of the gap to same-family rubrics (79.2%). This demonstrates that the rubric structure itself, rather than shared model biases, is the primary contributor to evaluation robustness.

D Ablations on Rubric Cardinality

- D.1 Setup

The number of rubric dimensions (cardinality) generated per preference instance represents a key design choice in ARR. Too few rubrics may underspecify the relevant evaluation space, failing to capture important axes of quality; too many may introduce redundant, conflicting, or noisy criteria that degrade the signal-to-noise ratio of the resulting reward. To systematically study this trade-off, we vary the number of rubrics generated per item (K ∈ {1,5,10,20}) while applying the same hierarchical structuring step to all settings, and measure preference accuracy on the HPDv3 test set. All experiments employ Qwen3-VL-8B-Instruct as the base judge and utilize zero-shot rubric generation (i.e., without human preference exemplars), thereby isolating the effect of cardinality from that of guidance quality.

- D.2 Results

Table 7 reports preference accuracy as a function of rubric cardinality. Accuracy is reported as the average of forward and reverse evaluation conditions to ensure that gains are not confounded by positional bias.

- D.3 Analysis

Accuracy improves monotonically with rubric cardinality. Increasing K from 1 to 20 yields a consistent improvement from 69.8% to 74.4%, a net gain of 4.6 percentage points. This monotonic trend indicates that additional rubric dimensions provide genuinely complementary information rather

- Table 7: Rubric cardinality vs. preference accuracy (HPDv3, Qwen3-VL-8B-Instruct, ARR zero-shot). Accuracy (%) reported as average of forward and reverse conditions. K = 5 is used as the default in all main experiments.

Num. Rubrics per Item (K) HPDv3 Accuracy (%)

1 69.8 5 70.2

10 72.1 20 74.4

than merely redundant coverage: each additional axis captures aspects of quality that are not fully addressed by fewer criteria, leading to more discriminative and robust evaluations.

The gains from K = 1 to K = 5 are modest but non-negligible. The increment from a single rubric to five rubrics yields only 0.4 percentage points, suggesting that a well-formed single rubric already captures the primary quality axis relevant to a given preference pair. However, the subsequent gains from K = 5 to K = 10 (+1.9 points) and from K = 10 to K = 20 (+2.3 points) demonstrate that finer-grained decomposition becomes increasingly consequential for difficult pairs where the quality differential is subtle or multidimensional.

Practical trade-offs and the choice of K = 5. While larger K yields higher accuracy, it also incurs a linear increase in inference cost: each rubric requires a separate generation call, a verification call, and evaluation against both images. We find that K = 5 provides a favorable accuracy–efficiency trade-off, achieving 70.2% accuracy with modest computational overhead. For deployment contexts where inference budget is constrained, K = 5 constitutes a well-calibrated default. For high-stakes evaluation scenarios, K = 20 delivers the strongest performance at approximately 2× the inference cost relative to K = 5.

Noise considerations at high cardinality. We note that despite the accuracy gains, larger K also elevates the probability of including noisy or redundant criteria: the marginal rubric at K = 20 is necessarily less discriminative than the most salient rubric at K = 1. In RPO training, noisy rubric axes contribute low-magnitude gradient signals whose impact is diluted through averaging across axes during reward aggregation, without harming overall convergence.

### E Rubric Policy Optimization Details

###### E.1 Algorithm Overview

RPO is an online policy gradient algorithm that leverages ARR-generated rubrics as binary reward signals to align a generative policy πθ with multidimensional human preferences. As described in Section 3.3 of the main text, RPO operates in a fully online fashion: at each iteration, it (i) samples prompts from a training distribution D, (ii) generates two candidate outputs from the current policy, (iii) evaluates the candidates using the frozen ARR judge conditioned on a dynamically synthesized rubric, and (iv) updates πθ via a PPO-style policy gradient with KL regularization. The full training procedure is formalized in Algorithm 1.

###### E.2 KL Regularization and Training Stability

The KL penalty βDKL(πθ∥πref) in the RPO objective (Equation 12 in the main text) serves two purposes. First, it prevents excessive policy drift away from the pretrained reference distribution πref, preserving the generative priors learned during pretraining. Second, it stabilizes training by bounding the entropy reduction induced by reward maximization, thereby mitigating mode collapse. We set β = 0.01 for T2I and β = 0.02 for image editing; the higher editing coefficient reflects the narrower action space and greater susceptibility to distributional collapse.

We observe that RPO training exhibits substantially lower variance in reward trajectories compared to reward-model-based RL baselines. We attribute this stability to two factors: (i) the frozen nature of the ARR judge eliminates reward model drift as a source of instability, and (ii) the rubric-conditioned

Algorithm 1 Rubric Policy Optimization (RPO) Require: Pretrained policy πθ

, reference policy πref, frozen ARR judge Mθ, training prompt distribution D, number of iterations N, batch size B, positive reward magnitude λ, negative reward magnitude γ, Clip threshold ϵ, KL coefficient β

0

Ensure: Optimized policy πθ

N

- 1: for iteration k = 1,...,N do
- 2: Sample a batch of prompts {hj}Bj=1 from D
- 3: for each prompt hj do
- 4: Generate two candidate outputs: yj1,yj2 ∼ πθ

k−1

(· | hj)

- 5: Synthesize or retrieve prompt-conditioned structured rubric: Rj = Rstructured(hj) {via ARR (Section 3.2)}
- 6: Obtain binary preference: pj = Mθ(yj1 ≻ yj2 | hj,Rj)
- 7: Assign advantages: Awinj ← +λ, Alossj ← −γ
- 8: Distribute advantage uniformly across all generation timesteps t = 0,...,T − 1
- 9: end for
- 10: Compute PPO-clipped objective LRPO(θk−1) according to Equation (12)
- 11: Update policy: θk ← θk−1 − η∇θLRPO(θk−1)
- 12: end for
- 13: return πθ

N

binary signal provides a more consistent gradient direction than scalar reward models, which collapse multi-dimensional quality into a single value subject to distributional shift.

### F Limitation

Frozen-model focus and fine-tuning potential. This work deliberately concentrates on frozen multimodal foundation models to isolate the effect of externalizing latent preference knowledge through auto-generated rubrics, rather than through parameter updates. By converting implicit, entangled preference representations into explicit, independently verifiable criteria, these rubrics serve as a structured interface that systematically suppresses evaluation biases, enhances interpretability, and directly translates into more reliable and hack-resistant reward signals for generative alignment. The demonstration that this training-free rubric conditioning alone can outperform dedicated pairwise reward models underscores the overlooked significance of the preference interface itself: the primary bottleneck in multimodal alignment is not a deficit of model capacity, but the absence of a stable, factorized criterion space for applying it. While further fine-tuning of the underlying VLMs would likely improve rubric fidelity and downstream generative quality, our results establish that the core value of the rubric paradigm lies in making preference evaluation structurally transparent, biasresilient, and scalable across tasks without requiring any modification to the judge model.

Pairwise formulation as a robustness measure. We adopt pairwise comparison as the core evaluation protocol because its comparative nature offers stronger structural resistance to reward hacking than pointwise scoring or differentiable reward models. Conditioning these judgments on explicit, prompt specific rubrics amplifies this resilience by grounding evaluation in inspectable, independently verifiable criteria that leave little room for opaque manipulation. This rubric driven interface also endows ARR with considerable extensibility: because the criteria are expressed in natural language, they can be dynamically expanded, refined, or adapted to new domains without any retraining of the underlying judge. The rubric thus functions as a transparent, stable scaffold that decouples evaluation logic from model parameters, preserving interpretability even as task requirements evolve. Although generative reward models and end to end VLM judges offer alternative paradigms, we prioritize the pairwise rubric interface as the most defensible, interpretable, and hack resistant configuration within the current alignment landscape, precisely because it externalizes the preference structure that other approaches leave implicit.

Human supervision and self-improvement. While the ARR framework readily accommodates human supervision to further refine rubric quality and specificity, the present work deliberately emphasizes what can be achieved with no additional annotation. Our experiments demonstrate that multimodal foundation models can substantially self-improve their comprehension and reasoning over human preferences purely through the auto-rubric process, using only self-generated criteria to guide

evaluation and optimization. This finding is significant because it reveals that the rubric mechanism itself, even without curated exemplars, provides a sufficient and scalable structure for preference alignment, transforming latent knowledge into actionable, verifiable constraints. Nevertheless, we acknowledge that fully automated rubric generation may not yet reach the precision or domain-specific nuance that curated human guidance could provide, and we therefore treat the deeper integration of human-in-the-loop rubric curation as a natural and valuable extension. The present results establish a lower bound: even in the absence of human intervention, externalizing preference structure through auto-rubrics proves remarkably effective, while the upper bound accessible through human refinement remains an open and promising direction.

### G Image Generation and Editing Examples

A rhyno playing basketball, cartoon style.

Dog, parrot, girl and cat playing with ball.

Two dogs and a lie dog caught in the scene of humans.

A photo of a family visiting a candy store.

Boy in a garden behind the ghost.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

FLUX.1.dev

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

FLUX. 1. dev w/ ARR

-RPO-Gemini3.1 pro

###### Figure 5: Examples of text-to-image generation.

Give her a baseball cap and made them colorful

Let's add a dog next to the cows.

Make the stop sing an animal warning sign.

Make the doll wear a hat.

Give her a skirt to wear.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Origin

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Qwen-Image-Edit

- -2509

Qwen- Image- Edit

- - 2508 w/ ARR-RPO
- -Gemini3.1 pro

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

###### Figure 6: Examples of image editing.

### H Full Results

###### H.1 Image Generation and Editing

- Table 8: Generative performance across T2I and Image Editing benchmarks. Blue-shaded rows denote ARR-RPO. Green arrows indicate absolute gains over the baseline.

Method

Text-to-Image Image Editing GenEval DPG-Bench TIIF UniGenBench++ GEdit-Bench ImgEdit

Short Long Specialist Model (T2I)

SDXL[31] 0.55 74.65 54.96 40.22 41.48 — Emu3[36] 0.54 80.60 – 45.42 50.59 — JanusFlow[27] 0.63 79.68 – 47.10 54.80 — FLUX.1-Dev 0.66 83.84 71.09 60.97 69.42 — DALLE-3[3] 0.67 83.50 74.96 68.85 70.82 — BLIP3o-4B[6] 0.81 79.36 – 59.57 61.01 — Janus-Pro-7B[8] 0.80 84.19 66.50 61.36 71.11 — Show-o2[45] 0.76 86.14 – 61.90 70.33 — OmniGen2[42] 0.80 83.57 – 63.09 71.39 — BAGEL[9] 0.82 85.07 71.50 59.91 71.26 — —

ARR-RPO / T2I (OURS)

w/ RPO-Qwen3vl-8B 0.72↑0.06 84.67↑0.83 73.81↑2.72 63.28↑2.31 71.05↑1.63 — w/ RPO-Qwen3vl-8B-ARR 0.74↑0.08 85.03↑1.19 74.92↑3.83 64.17↑3.20 71.82↑2.40 — w/ RPO-GPT-5 0.76↑0.10 84.97↑1.13 74.84↑3.75 64.22↑3.25 71.78↑2.36 — w/ RPO-GPT-5-ARR 0.78↑0.12 85.41↑1.57 76.18↑5.09 65.36↑4.39 72.41↑2.99 — w/ RPO-Gemini 3.1 Pro 0.77↑0.11 85.02↑1.18 75.69↑4.60 64.76↑3.79 72.13↑2.71 — w/ RPO-Gemini 3.1 Pro-ARR 0.80↑0.14 85.76↑1.92 76.85↑5.76 65.89↑4.92 72.93↑3.51 — —

Specialist Model (Editing)

Instruct-Pix2Pix[5] — — — — — 3.68 1.88 AnyEdit[50] — — — — — 3.21 2.45 Step1X-Edit[24] — — — — — 6.97 3.06 Qwen-Image-Edit-2509[41] — — — — — 7.54 4.35 UniWorldv2[23] — — — — — 7.76 4.48

ARR-RPO / Image Editing (OURS)

w/ RPO-Qwen3vl-8B — — — — — 7.63↑0.09 4.37↑0.02 w/ RPO-Qwen3vl-8B-ARR — — — — — 7.66↑0.12 4.38↑0.03 w/ RPO-GPT-5 — — — — — 7.65↑0.11 4.38↑0.03 w/ RPO-GPT-5-ARR — — — — — 7.72↑0.18 4.40↑0.05 w/ RPO-Gemini 3.1 Pro — — — — — 7.79↑0.25 4.39↑0.04 w/ RPO-Gemini 3.1 Pro-ARR — — — — — 7.85↑0.31 4.43↑0.08

- Table 9: Additional experimental results: Post-training performance of BAGEL using ARRRPO across T2I benchmarks. Best in bold. Blue-shaded rows denote ARR-RPO variants. Green arrows indicate absolute gains over the BAGEL baseline.

Method GenEval DPG-Bench TIIF UniGenBench++ Short Long

BAGEL (Baseline) BAGEL[9] 0.82 85.07 71.50 59.91 71.26

###### ARR-RPO / T2I (OURS)

w/ RPO-Qwen3vl-8B 0.85↑0.03 85.45↑0.38 73.12↑1.62 62.34↑2.43 71.89↑0.63 w/ RPO-Qwen3vl-8B-ARR 0.88↑0.06 85.82↑0.75 74.05↑2.55 63.05↑3.14 72.35↑1.09 w/ RPO-GPT-5 0.86↑0.04 85.91↑0.84 74.48↑2.98 63.72↑3.81 72.58↑1.32 w/ RPO-GPT-5-ARR 0.90↑0.08 86.28↑1.21 75.62↑4.12 64.81↑4.90 73.15↑1.89 w/ RPO-Gemini 3.1 Pro 0.87↑0.05 86.15↑1.08 75.41↑3.91 64.62↑4.71 72.97↑1.71 w/ RPO-Gemini 3.1 Pro-ARR 0.92↑0.10 86.74↑1.67 76.85↑5.35 65.92↑6.01 73.82↑2.56

###### H.2 Human Preference

- Table 10: Evaluator performance across four preference benchmarks. Accuracy (%) denotes agreement with human preference labels. The best result in each column is bold. Blue-shaded rows indicate ARR. Green arrows indicate absolute gains over the baseline.

HPDv3 MM-RewardBench2 MM-RewardBench2 EditReward-Bench (T2I) (Edit)

Method

Acc. Acc. Acc. Acc. Trained Reward Model

CLIPScore[14] 48.6 51.0 — PickScore[19] 65.6 58.6 — ImageReward[47] 58.6 54.0 — UnifiedReward[39] 66.0 59.8 — UnifiedReward-Thinking[38] 68.1 66.0 — HPSv2[44] 65.3 54.7 — HPSv3[28] 76.9 60.2 — EditReward[43] — — 67.2 56.45

###### VLM-as-Judge (Direct)

Qwen3-VL-8B 67.2 57.6 59.2 54.01 GPT-5 72.4 70.5 73.8 57.53 Gemini 3.1 Pro 76.6 75.1 77.4 61.23

###### ARR (Ours)

Qwen3vl-8B + ARR 70.2 (± 0.2)↑3.0 62.7 (± 0.2)↑5.1 65.5 (± 0.3)↑6.3 57.22 (± 0.1)↑3.21 GPT-5 + ARR 76.1 (± 0.2)↑3.7 74.7 (± 0.4)↑4.2 77.5 (± 0.3)↑3.7 61.01 (± 0.1)↑3.48 Gemini 3.1 Pro + ARR 78.3 (± 0.1)↑1.7 78.9 (± 0.2)↑3.8 79.2 (± 0.2)↑1.8 63.27 (± 0.2)↑2.04

### I Prompts and Rubrics

###### T2I Rubrics w/ Gemini3.1 Pro

- Rubric 1: Theme: Architectural and Structural Fidelity

- - Tip1: Assess clarity and precision of building facades, windows, railings, and structural elements for geometric accuracy and absence of distortion.
- - Tip2: Evaluate consistency of architectural details (e.g., columns, arches, textures) across the scene without illogical or impossible geometry.
- - Tip3: Check for realistic material rendering (wood grain, stone patina, metal sheen) and absence of overly smooth or artificial surfaces.
- - Tip4: Ensure environmental context is coherent, with buildings, paths, and landscapes logically integrated and proportionally accurate.

- Rubric 2: Theme: Lighting, Shadow, and Reflection Realism

- - Tip1: Evaluate how naturally light interacts with surfaces, including direction, intensity, and shadow gradients that match the scene’s source.
- - Tip2: Assess reflections on water, glass, and metallic surfaces for physical plausibility and consistency with ambient lighting.
- - Tip3: Check for believable depth and dimensionality through consistent, directional shadows and highlights across all objects and subjects.
- - Tip4: Identify flat, unnatural, or inconsistent lighting that flattens the scene or creates artificial glow and highlights.

- Rubric 3: Theme: Texture and Material Fidelity

- - Tip1: Judge the clarity and detail of fine textures (fabric weave, fur, wood grain, skin pores) without smearing or over-smoothing.
- - Tip2: Assess realism of material properties (metallic sheen, fabric wear, plastic gloss) and absence of uniform, plastic-like surfaces.
- - Tip3: Look for natural imperfections (dust, scratches, frayed edges) that enhance authenticity and avoid overly pristine or artificial finishes.
- - Tip4: Evaluate surface variations and non-repetitive patterns to distinguish between realistic textures and AI-generated uniformity.

- Rubric 4: Theme: Anatomical and Object Plausibility

- - Tip1: Assess anatomical accuracy of human and animal forms, including proportions, posture, and facial features for naturalism.
- - Tip2: Evaluate mechanical and structural plausibility of objects (tools, vehicles, clothing) for logical construction and physical consistency.
- - Tip3: Check for distorted limbs, unnatural hand shapes, or illogical object boundaries that break immersion or suggest AI artifacts.
- - Tip4: Ensure subjects interact with their environment in believable ways (e.g., shadows, occlusion, posture) without surreal or impossible elements.

- Rubric 5: Theme: Visual Coherence and Composition

- - Tip1: Assess overall scene balance, depth, and arrangement of elements for aesthetic harmony and intentional design.
- - Tip2: Evaluate framing, perspective, and depth of field for natural spatial relationships and absence of awkward or cluttered compositions.
- - Tip3: Judge the integration of foreground, midground, and background for visual flow and environmental immersion.
- - Tip4: Look for compositional clarity, with focal points and visual hierarchy that guide the viewer’s eye without distraction.

- Rubric 6: Theme: AI Artifact Detection and Elimination

- - Tip1: Identify telltale signs of AI generation: unnatural edges, repeated patterns, inconsistent perspective, or surreal details.
- - Tip2: Look for artifacts such as smearing, halos, pixelation, or warped geometry that disrupt visual authenticity.
- - Tip3: Assess for illogical object placement, duplicated elements, or inconsistent lighting that break scene coherence.
- - Tip4: Prioritize images with minimal or no AI artifacts, especially in critical areas like anatomy, textures, and object boundaries.

- Rubric 7: Theme: Environmental and Background Realism

- - Tip1: Evaluate the realism of natural elements (trees, water, clouds, terrain) for accurate physics, textures, and atmospheric perspective.
- - Tip2: Assess background depth and context for logical scale, perspective, and integration with foreground subjects.
- - Tip3: Check for believable environmental details (e.g., foliage, ground texture, sky gradients) without AI-style repetition or oversaturation.
- - Tip4: Ensure background elements (buildings, landscapes, props) are contextually plausible and not visually disconnected from the scene.

- Rubric 8: Theme: Text, Typography, and Detail Legibility

- - Tip1: Assess clarity and legibility of text, including font consistency, spacing, and absence of smudging or distortion.
- - Tip2: Evaluate spelling accuracy, grammar, and semantic coherence of written elements (e.g., signs, captions, product labels).
- - Tip3: Check for clean, well-defined typography with appropriate contrast and alignment against background elements.
- - Tip4: Prioritize images with crisp, readable text and no garbled, pixelated, or AI-generated text artifacts.

- Rubric 9: Theme: Color, Gradient, and Aesthetic Harmony

- - Tip1: Assess color palette richness, saturation, and harmony for contextually appropriate and visually pleasing transitions.
- - Tip2: Evaluate natural lighting and color grading for believable gradients, shadows, and reflections without artificial neon or oversaturation.
- - Tip3: Judge overall aesthetic appeal, including balance, contrast, and intentional visual design that enhances mood or narrative.
- - Tip4: Look for unified color schemes that support the scene’s theme without jarring or mismatched tones.

- Rubric 10: Theme: Detail Fidelity and Micro-Texture Rendering

- - Tip1: Assess sharpness and clarity of fine details (hair strands, stitching, buttonholes, tool engravings) without AI-style blurring.
- - Tip2: Evaluate preservation of intricate textures (fur, fabric, metal, skin) that convey realism and depth without over-smoothing.
- - Tip3: Check for consistent rendering of small elements (e.g., droplets, dust, scratches) that add authenticity without artificial repetition.
- - Tip4: Prioritize images where micro-textures are naturally varied and physically plausible, not uniformly or algorithmically generated.

- Figure 7: Auto-generated T2I rubrics (Gemini 3.1 Pro). Example prompt-conditioned rubrics automatically synthesized by ARR for text-to-image evaluation, spanning dimensions such as architectural fidelity, lighting consistency, texture realism, and AI artifact detection.

#### T2I System Prompt

Based on the given evaluation rubrics and the caption, please perform a **pairwise comparison** between the two input images (Image 1 and Image 2) to determine which one has higher quality. {task_description_section} Evaluation Rubrics: {rubrics}

**Caption: {sample_content}** ## Task Requirements

- - You are evaluating two images: Image 1 and Image 2.
- - Using the provided evaluation rubrics, determine which image has better overall quality.
- - Assign a rank value to each image:
- - rank = 1 → higher quality (better image)
- - rank = 2 → lower quality (worse image)
- - The output format should be: [rank_of_Image1, rank_of_Image2]
- - Examples:
- - If Image 1 is better than Image 2 → output: [1, 2]
- - If Image 2 is better than Image 1 → output: [2, 1]
- - Important: The two rank values **must be different** — ties are not allowed.
- - Focus solely on the image content and quality per the rubrics; do **not** let image order or position bias your judgment.

## Output Format Please strictly follow the JSON format below: {{

"rank": [rank_of_Image1, rank_of_Image2], "reason": "Detailed explanation of the reasoning and quality assessment for both images"

}} Key reminders:

- - The `rank` array must contain exactly 2 values, one for each image.
- - rank = 1 means higher quality; rank = 2 means lower quality.

- Figure 8: T2I evaluation system prompt. The prompt template used to instruct the VLM judge to perform pairwise comparison for text-to-image generation, including task description, output format requirements, and anti-position-bias reminders.

###### Edit Rubrics w/ Gemini3.1 Pro

- Rubric 1: Theme: Fidelity of Built Structures After Editing

- - Tip 1: Verify that edited building facades, windows, railings, roofs, and other structural elements preserve precise geometric accuracy, with straight lines, correct proportions, and no visible distortions such as bending, warping, or duplicated edges.
- - Tip 2: Ensure that architectural details, including columns, arches, balconies, decorative patterns, and surface textures, remain internally consistent and logically constructed after editing, without introducing impossible or unstable structures.
- - Tip 3: Check that material properties of architectural components, such as wood grain, stone roughness, brick seams, glass transparency, and metallic reflections, are faithfully preserved or enhanced without becoming unnaturally smooth, plastic-like, or visually artificial.
- - Tip 4: Confirm that edits maintain coherent spatial relationships among buildings, pathways, roads, vegetation, and surrounding environmental elements, ensuring perspective consistency and physically plausible scene layout.

- Rubric 2: Theme: Physically Plausible Illumination Adjustments

- - Tip 1: Ensure that edited lighting direction, intensity, exposure, and color temperature remain physically plausible and align with the original illumination setup or the newly introduced light source.
- - Tip 2: Validate that shadows added, removed, or modified remain globally consistent in direction, softness, scale, and occlusion behavior across all scene objects and surfaces.
- - Tip 3: Check that reflections and specular highlights on water, glass, polished metal, or glossy surfaces are updated consistently to match the revised lighting conditions and viewing angle.
- - Tip 4: Avoid introducing flat lighting, artificial glow, or inconsistent highlights during editing.

- Rubric 3: Theme: Material Integrity During Editing

- - Tip 1: Confirm that fine textures (fabric, wood grain, skin, fur) are preserved and not blurred or over-smoothed.
- - Tip 2: Ensure that material-specific physical properties, including glossiness, roughness, transparency, translucency, and reflectivity, remain realistic and visually consistent after editing.
- - Tip 3: Introduce or maintain natural imperfections (scratches, dust, wear) where appropriate to avoid artificial perfection.
- - Tip 4: Avoid repetitive, tiled, or uniform texture artifacts that may be introduced during the editing process, particularly in large homogeneous regions.

- Rubric 4: Theme: Realism of Subjects and Objects Post-Edit

- - Tip 1: Ensure human and animal anatomy remains proportionally accurate after editing (no distorted limbs or features).
- - Tip 2: Validate that edited objects, such as tools, clothing, furniture, vehicles, or accessories, preserve logical structure, correct functionality, and physical plausibility.
- - Tip 3: Detect and correct warped shapes, broken boundaries, or blending artifacts.
- - Tip 4: Confirm that subjects interact naturally with their environment (contact, shadows, occlusion).

- Rubric 5: Theme: Cohesion and Aesthetic Integrity

- - Tip 1: Ensure edits preserve or improve scene balance and visual hierarchy.
- - Tip 2: Validate perspective, framing, and depth relationships after modifications.
- - Tip 3: Maintain smooth integration between foreground, midground, and background elements, avoiding abrupt transitions, cutout artifacts, or mismatched blur levels.
- - Tip 4: Avoid clutter, awkward cropping, or disruption of focal points.

- Rubric 6: Theme: Elimination of AI and Editing Artifacts

- - Tip 1: Identify and remove common artifacts such as smearing, halos, pixelation, jagged edges, or unnatural blur introduced during editing or generation.
- - Tip 2: Avoid introducing repeated patterns, duplicated elements, or inconsistent geometry.
- - Tip 3: Ensure seamless blending between edited and original regions, with smooth transitions in texture, lighting, and color.
- - Tip 4: Prioritize clean, polished outputs with minimal visible traces of manipulation, ensuring the image appears naturally captured rather than artificially altered.

- Rubric 7: Theme: Background and Context Integrity

- - Tip 1: Ensure environmental elements (e.g., trees, sky, terrain, buildings) remain physically plausible and visually consistent after editing.
- - Tip 2: Maintain consistent scale, depth, and perspective between foreground and background.
- - Tip 3: Avoid repetitive, tiled, or overly uniform patterns in natural elements, which may indicate synthetic generation.
- - Tip 4: Ensure all background details align contextually with the subject, supporting a coherent scene narrative and avoiding mismatched environments.

- Rubric 8: Theme: Clarity of Edited Text Elements

- - Tip 1: Ensure all text remains clearly legible after editing, with crisp edges and consistent, recognizable font styles.
- - Tip 2: Preserve correct spelling, grammar, and semantic meaning, avoiding nonsensical or AI-generated text artifacts.
- - Tip 3: Maintain proper alignment, spacing, and contrast to ensure readability against varying backgrounds.
- - Tip 4: Avoid distortions such as warping, stretching, or irregular baselines that degrade text quality or professionalism.

- Rubric 9: Theme: Harmonized Color Adjustments

- - Tip 1: Ensure color palettes remain balanced and contextually appropriate to the scene, avoiding abrupt or unrealistic shifts.
- - Tip 2: Maintain smooth and natural gradients, preventing banding, clipping, or harsh transitions between tones.
- - Tip 3: Align color grading with the scene’s lighting conditions, time of day, and intended mood or atmosphere.
- - Tip 4: Avoid oversaturation, color mismatches, or unnatural hues that reduce realism or visual comfort.

- Rubric 10: Theme: Micro-Detail Integrity in Editing

- - Tip 1: Preserve sharpness and fidelity in fine details such as hair strands, fabric stitching, surface engravings, or textures.
- - Tip 2: Ensure micro-textures remain natural and are not overly smoothed or artificially sharpened during processing.
- - Tip 3: Maintain variation and randomness in small-scale details to avoid repetitive or algorithmically generated patterns.
- - Tip 4: Prevent loss of high-frequency information, ensuring that intricate regions retain clarity and do not degrade during editing.

- Figure 9: Auto-generated image editing rubrics (Gemini 3.1 Pro). Example prompt-conditioned rubrics automatically synthesized by ARR for image editing evaluation, covering fidelity preservation, material integrity, lighting consistency, and artifact elimination.

##### Edit System Prompt

Please perform a pairwise comparison between the two edited versions (**Image 1** and **Image 2**) based on the given rubrics and caption, using the **original image (Image BASE)** as the reference benchmark. Determine which edited result is superior. {task_description_section}

**Evaluation Rubrics:** {rubrics}

**caption: {sample_content}** ## Task Requirements

- - You are evaluating two image editing results (Image 1 and Image 2) with **Image BASE (the original image)** as the ground-truth reference.
- - Compare them across the following aspects:

- - Fidelity to the original content
- - Accuracy and naturalness of instruction following

- - Assign rank values to the two edited images:

- - rank=1 means better quality (higher fidelity to Image BASE + better editing)
- - rank=2 means worse quality

- - Output format must be: [rank_of_Image1, rank_of_Image2]
- - Examples:

- - If Image 1 is better → [1, 2]
- - If Image 2 is better → [2, 1]

- - Important: The two ranks must be different (strictly 1 and 2). All judgments must explicitly reference Image BASE.
- - Judge solely based on image content, the provided rubrics, and comparison with the original Image BASE. Do not be influenced by image position or order. ## Output Format Please strictly follow this JSON format: {

"rank": [rank of Image 1, rank of Image 2], "reason": "Detailed explanation of the judgment, including comparison of each edited image

with the original Image BASE" }

**Key Reminder:**

- - The "rank" array must contain exactly two values corresponding to Image 1 and Image 2.
- - rank=1 indicates the better edit (higher fidelity + better execution), rank=2 indicates the worse one.

- Figure 10: Image editing evaluation system prompt. The prompt template used to instruct the VLM judge to perform pairwise comparison for image editing, where Image BASE serves as the ground-truth reference for fidelity assessment.

