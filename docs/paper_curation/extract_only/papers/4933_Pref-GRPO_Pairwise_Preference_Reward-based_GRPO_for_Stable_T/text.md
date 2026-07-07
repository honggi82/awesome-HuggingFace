## Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning

Yibin Wang1,2,3*, Zhimin Li3*, Yuhang Zang4*, Yujie Zhou4,5, Jiazi Bu4,5 Chunyu Wang3†, Qinglin Lu3†, Cheng Jin1,2†, Jiaqi Wang2† 1Fudan University, 2Shanghai Innovation Institute 3Hunyuan, Tencent, 4Shanghai AI Lab, 5Shanghai Jiao Tong University Project Page: codegoat24.github.io/UnifiedReward/Pref-GRPO

### arXiv:2508.20751v2[cs.CV]20Apr2026

[Figure 1]

###### Stable Optimization

T2I Model

Generate an animation: a fox with a body made of blue and white porcelain and a rabbit with a body made of terracotta warriors are racing on the Great Wall.

Fitting preferences instead of high scores.

[Figure 2]

Rollout

#### Pairwise Preference Reward-based GRPO

SDE Sampling (Tsteps)

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Win Rate Rewards

which image is better?

Advantages

mean = 0.50 std = 0.37

Image #1

Image #2

Image #4

[Figure 7]

[Figure 8]

- -1.34,

- 0.46,
- 1.34,

- -0.46

0.00,

[Figure 9]

- 0.67,
- 1.00, 0.33

Group Advantage Computation

All Pairwise Groups

Policy Optimization

Image#1 Image#2 Image#3 Image#4

Pairwise Preference Reward Model

Point Score Reward Function

Action FLUX.1-dev Attribute w/HPSv2 w/HPSv2+CLIP w/UnifiedReward w/Pref-GRPO(Ours)

[Figure 10]

Illusory Advantage

[Figure 11]

max reward diff. : 0.0073 max advantage diff. : 2.83

World Knowledge

Relationship

(a) Reward std over Training Steps

HPSv2 Rewards: [0.3457, 0.3530, 0.3496, 0.3494]

[Figure 12]

Pref-GRPO(Ours)

Group Advantage Computation

Logical Reasoning

0.5

UnifiedReward(Point)

Style

mean = 0.3494 std = 0.0026

CLIP(Point)

HPSv2(Point)

0.4

Advantages: [-1.44, 1.39, 0.07, -0.01]

std

0.3

[Figure 13]

Text

Grammar

0.2

Policy Optimization

Excessive optimization for minor reward gains

score maximization

0.1

Compbound Layout

0.0

[Figure 14]

Reward Hack

20 40 60 80 100 120 140

(b) Effectiveness Comparison on UniGenBench

Training steps

Fig. 1. PREF-GRPO: from illusory advantage to pairwise preference fitting. (a) Pointwise reward models assign tightly clustered scores to images within a group, producing an illusory advantage after normalization and causing reward hacking. (b) PREF-GRPO reformulates the GRPO objective from absolute score maximization to pairwise preference fitting, enabling stable T2I optimization.

Abstract—Recent progress has made GRPO-based reinforcement learning central to advancing text-to-image (T2I) generation. Current GRPO methods score a group of generated images with pointwise reward models and normalize the scores within each group to compute advantages. Although effective in early training, this reward-score-maximization approach is susceptible to reward hacking: scores keep rising while image quality deteriorates, manifesting as oversaturation or unnaturally dark artifacts. We trace this failure to an illusory advantage: tightly clustered scores within a group are disproportionately amplified after dividing by the small group standard deviation, driving the policy to over-optimize. To address this, we propose PREF-GRPO, which reformulates the GRPO objective to pairwise preference fitting: image pairs within a group are compared by a Pairwise Preference Reward Model (PPRM), and each image’s win rate serves as the reward. Because win rates reflect relative rankings rather than absolute scalar scores, PREF-GRPO yields larger within-group variance and is intrinsically robust to small biases. Experiments show that PREF-GRPO produces more stable advantages than pointwise scoring, substantially alleviates reward hacking, and improves semantic alignment.

offer only coarse, primary-dimension-only evaluation, reporting aggregate scores that hide where models actually succeed or fail. We propose UNIGENBENCH, a fine-grained T2I evaluation benchmark with 600 prompts across 5 primary themes and 20 sub-themes, evaluating 10 primary dimensions and 27 subdimensions with 1–5 explicit testpoints per prompt. Each testpoint is paired with a structured description, enabling precise pertestpoint assessment. Leveraging an MLLM (Gemini-2.5-Pro), we build an automated pipeline for benchmark construction and evaluation. Benchmarking representative open- and closed-source T2I models on UNIGENBENCH, we find that Style and World Knowledge are largely saturated across strong models, while finegrained compositional capabilities such as Logical Reasoning, Grammar, and Compound remain the primary bottlenecks—a diagnostic gap that aggregate scores cannot surface.

Together, PREF-GRPO and UNIGENBENCH contribute a more stable optimization objective and a more diagnostic evaluation framework, pointing to fine-grained compositional reasoning as the next frontier for T2I reinforcement learning.

Index Terms—text-to-image generation, reinforcement learning, benchmarking

Alongside the optimization side, existing T2I benchmarks

[Figure 15]

###### JOURNAL OF LATEX CLASS FILES 2

(a) Preference Score over Training Steps

ExpressionQuantity

World Knowledge

Material

[Figure 16]

Size

[Figure 17]

27

y

Shape

Sculptural Other

h

w/Pref-GRPO (Ours)

w.

p

Sub Testpoints

Color

hotogra

FLUX w/UnifiedReward (UR)

Attrib ute

(b) Subject Categories

Hand

Kno

Graphic

(score only for evaluation)

Full body

P

Other

[Figure 18]

Copywriting

Animal

World

Style

[Figure 19]

[Figure 20]

Non-Contact

Action

3.45

Imaginative

10

Art

Style

Contact

Primary Testpoints

Creative Divergence

(c) TestPoints

Illustration

State

Content

Design Assets

UR Score

Fashion

3.40

Film & Story

###### Text

Design

Text Generation

Relationship

Intelectual Property

Logo/Icon

Composition

User

Layout

Realism

Poster

Similarity

Interface

Ad./E-co m

mpound

3.35

- 2D
- 3D

Logic

Science

Gr a m m

Game

Spatial

A nim

Fiction

Inclusion

Comparison

merce

ation

Co

[Figure 21]

[Figure 22]

Animal Human Anth. Scene Others

Hack Stable

ar

Logical Reasoning

3.30

Imagination

[Figure 23]

(a) Prompt Themes

Feature Matching

Negation

Pronoun Consistency Reference

(5 Primary Categories & 20 Second Categories )

3.25

Action Attribute

Action Attribute

[Figure 24]

[Figure 25]

SDXL Playgournd-2.5 Emu-3 CogView-4 FLUX.1-dev BLIP-3o Bagel Janus-Pro SD3.5-Large Show-o2 HiDream-I1-Full

160 240 320 80 160 240 320

80

World Knowledge

World Knowledge

(b) Qualitative Comparison

Relationship

Relationship

Dalle-3

Design a cat with butterfly wings, its whole body is made of translucent glass.

FLUX-Pro-Ultra

FLUX-Kontext-Max

Nano Banana Seedream-4.0 Imagen-4.0-Ultra GPT-4o

[Figure 26]

[Figure 27]

[Figure 28]

Logical Reasoning

Qwen-Image

Logical Reasoning

20 40 60 Style

20 40 60

Style

(d) Open-Source Model Benchmarking

(e) Closed-Source Model Benchmarking

Grammar

Text

Grammar

Text

Compbound Layout

Compbound Layout

[Figure 29]

Baseline w/UnifiedReward w/Pref-GRPO

Fig. 3. UNIGENBENCH overview. (a) Prompt themes: 5 primary categories and 20 sub-categories. (b) Subject categories. (c) Evaluation testpoints: 10 primary dimensions and 27 sub-dimensions. (d,e) Benchmarking results of representative open-source and closed-source T2I models on UNIGENBENCH.

Fig. 2. Reward hacking under UnifiedReward. (a) UnifiedReward scores rise monotonically during training. (b) Generated images gradually drift toward an unnaturally dark style, despite the rising reward, which is a clear rewardhacking signature.

and 2). Moreover, small biases in the reward model are similarly amplified, encouraging the policy to exploit reward-model flaws rather than align with human preferences. (2) Current T2I models already perform well on most primary dimensions (e.g., object attributes and actions), which makes it necessary to decompose these broad dimensions into finer-grained subdimensions for more rigorous, diagnostic evaluation.

I. INTRODUCTION

# R

ECENT progress in text-to-image (T2I) generation has made reinforcement learning [1–4] and comprehensive

benchmarking [5–7] two pivotal pillars for advancing model capabilities and for evaluating them reliably. Specifically, several GRPO-based approaches [1, 3] employ pointwise reward models [8–10] to score a group of generated images at each training step and normalize these scores into advantages following the GRPO objective [11]. This pipeline has proven highly effective in aligning T2I generation with human preferences.

To this end, this work proposes PREF-GRPO, the first pairwise-preference-reward-based GRPO method for stable T2I reinforcement learning, and UNIGENBENCH, a fine-grained T2I evaluation benchmark for semantic consistency.

(1) PREF-GRPO incorporates a Pairwise Preference Reward Model (PPRM) [13], reformulating the GRPO objective from absolute reward score maximization to pairwise preference fitting. As illustrated in Fig. 1, at each training step, given a group of generated images, we use the PPRM on each image pair to identify the preferred image. Each image’s win rate (the proportion of pairwise comparisons in which it is preferred) is then used as the reward signal for policy optimization. This design offers three key advantages: (a) Amplified reward variance: driving high-quality images toward win rates near 1 and low-quality ones toward 0 yields more separable reward distributions and more informative advantage estimates. (b) Robustness to reward noise: relying on relative rankings rather than absolute scores reduces over-optimization for marginal score gains and mitigates reward hacking. (c) Alignment with human preference: pairwise comparisons mirror the way humans compare images, producing reward signals that better capture nuanced preferences. Extensive experiments show that PREF-GRPO discerns subtle quality variations, produces more stable and directional advantages than pointwise scoring, and alleviates reward hacking. (2) UNIGENBENCH is built for fine-grained T2I evaluation, covering 10 primary dimensions and 27 sub-dimensions, alongside 5 prompt themes (20 subthemes) and diverse subject categories (Fig. 3). Unlike existing benchmarks that provide only coarse, primary-dimension-only evaluation, most primary dimensions in UNIGENBENCH are further subdivided into fine-grained sub-dimensions, each

In parallel, evaluating T2I models, particularly their instruction-following capability, has become increasingly important. Widely adopted benchmarks [5, 6] probe compositional aspects using CLIP [12]-based or detection-based metrics, and TIIF-Bench [7] further incorporates additional dimensions such as text rendering.

Despite these advances, existing approaches still face two key limitations: (1) Existing GRPO-based methods use pointwise reward models to maximize reward scores, which provides early gains but often leads to reward hacking: reward scores keep rising while image quality deteriorates over prolonged training (Fig. 2), a failure mode also noted in prior work [1, 3]. (2) Current T2I benchmarks provide only coarse, primarydimension-only evaluation, covering a limited range of subdimensions without reporting fine-grained sub-dimension scores (Tab. I).

In light of these issues, we argue that (1) reward hacking in GRPO-based methods stems from an illusory advantage that arises when pointwise reward models assign tightly clustered scores to images within the same group. Normalizing these scores into advantages disproportionately amplifies small gaps. Under a reward-maximization objective, such inflated advantages drive the policy to over-optimize for trivial reward cues, eventually steering it toward reward-hacking behaviors that rapidly increase scores while destabilizing generation (Figs. 1

TABLE I BENCHMARK COMPARISON WITH EXISTING T2I BENCHMARKS. WE SUMMARIZE BENCHMARK COVERAGE ACROSS THE DIMENSIONS IN UNIGENBENCH. EXISTING BENCHMARKS MAINLY REPORT PRIMARY-DIMENSION SCORES OR COVER ONLY A SPARSE SUBSET OF SUB-DIMENSIONS, WHEREAS UNIGENBENCH PROVIDES 10 PRIMARY DIMENSIONS AND 27 SUB-DIMENSIONS WITH FINE-GRAINED PER-SUB-DIMENSION EVALUATION.

Score Mode

World Know.

Logical Reason.

Benchmark

Style

Attribute Action Relationship Compound Grammar Layout

Text

Full Body

Non Contact

Feature Matching

Pronoun Ref.

— — Quant. Expn. Material Size Shape Color Hand

Animal

Contact State Compo. Similarity Inclusion Comparison Imagin.

Consistency Negation 2D 3D — —

GenEval ✓ ✓ ✓ ✓ T2I-Comp

Primary Dimension

✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ TIIF-Bench ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

Primary & Sub Dimension

UniGenBench (Ours)

✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

associated with explicit testpoints (Tab. I). We further build an automated evaluation pipeline powered by a strong MLLM (Gemini-2.5-Pro [14]) for both benchmark construction and T2I model evaluation, illustrated in Fig. 7. We benchmark popular closed-source models, including GPT-4o [15], Nano Banana, Seedream-4.0 [16], and Imagen-4.0-Ultra [17], as well as leading open-source models such as Qwen-Image [18], HiDream-I1-Full [19], and Bagel [20]. As shown in Fig. 3(d, e), both open- and closed-source models perform relatively well on Style and World Knowledge prompts, but consistently underperform on prompts requiring Logical Reasoning, such

- as those involving causal or contrastive relationships. Contributions: (1) We identify a root cause of reward hack-

ing in pointwise-reward GRPO: the illusory advantage induced by within-group score normalization. (2) We propose PREFGRPO, the first pairwise-preference-reward-based GRPO method for stable T2I reinforcement learning, which reformulates the optimization objective from absolute reward score maximization to pairwise preference fitting. (3) Experiments on UNIGENBENCH, GenEval, and T2I-CompBench show that PREF-GRPO alleviates reward hacking and improves semantic alignment (+5.84% overall on UNIGENBENCH, with +12.69% on Text and +12.04% on Logical Reasoning), without sacrificing perceptual quality. (4) We introduce UNIGENBENCH, a finegrained T2I benchmark with 10 primary dimensions and 27 subdimensions across 5 prompt themes (20 sub-themes), together with an MLLM-based pipeline for benchmark construction and evaluation. (5) We benchmark representative open- and closedsource T2I models on UNIGENBENCH, revealing that Style and World Knowledge are saturated across strong models, while finegrained compositional capabilities such as Logical Reasoning, Grammar, and Compound remain the primary bottlenecks.

II. RELATED WORK

Reinforcement Learning for T2I Generation is gaining rapid momentum. Early efforts pursued preference-driven objectives [21–23]. For example, DiffusionDPO [23], adapted from Direct Preference Optimization (DPO) [24], optimizes preference consistency via a classification-style objective as a simpler alternative to RLHF. DyMO [21] offers a plugand-play, training-free alignment method that steers generated images toward human preferences at inference. More recently, group relative policy optimization (GRPO) has advanced online RL-enhanced image generation. Flow-GRPO [25] and DanceGRPO [3] instantiate GRPO on flow-matching models, introducing stochasticity by recasting the original deterministic ODE as an equivalent SDE. While effective, these reward

score-maximization methods are prone to reward hacking due to illusory advantage. To this end, we propose PREFGRPO, which shifts training from reward-score maximization to pairwise preference fitting, yielding more stable advantages, and thereby mitigates reward hacking.

Existing Benchmarks for T2I Evaluation have expanded the evaluation of T2I models beyond simple visual fidelity, incorporating compositional reasoning [5, 6] and world knowledge [26]. Recently, [7] introduces TIIF-Bench, containing 5k prompts spanning multiple dimensions, i.e., text rendering and style control, rigorously evaluating model robustness to variations in prompt length. However, existing benchmarks largely focus on primary dimension–level coarse assessment, covering a limited set of sub-tasks and lacking fine-grained assessment of these sub-tasks. To address this gap, we propose a unified image generation benchmark, UNIGENBENCH, consisting of 600 prompts spanning diverse themes and categories, assessing T2I models across 10 primary dimensions and 27 sub-dimensions.

III. MOTIVATION FOR PAIRWISE PREFERENCE REWARD

Existing GRPO-based text-to-image methods optimize the policy by maximizing pointwise reward scores [1–4]. This design implicitly assumes that the absolute score assigned by the reward model is both sufficiently discriminative within a rollout group and reliable enough to support gradient scaling. In practice, however, these assumptions are often violated. For visually similar images sampled from the same prompt, pointwise reward models such as HPSv2 [9] and UnifiedReward [8] frequently produce highly compressed scores (see Fig. 1). GRPO then normalizes these near-identical scores by a very small group standard deviation, turning negligible score gaps into disproportionately large advantages. The optimization is therefore driven not by substantial quality differences, but by amplified numerical fluctuations, which leads to the illusory advantage phenomenon, as analyzed in Sec. IV-B.

The consequence is not merely optimization instability, but a deeper objective mismatch. Once the training target is defined as absolute reward maximization, the policy is encouraged to chase any reward increase, even when that increase comes from reward-model bias rather than genuine improvement in text-image alignment or visual quality. This helps explain the reward hacking behaviors reported in prior work [1, 3] and observed in our experiments: HPSv2 encourages over-saturated generations (see Fig. 9), while UnifiedReward tends to favor an unnatural dark style (see Fig. 2). Importantly, simply shrinking or clipping the normalized advantage can only dampen the

update magnitude; it cannot remove the underlying tendency to optimize toward unreliable absolute-score increments.

These observations motivate replacing reward score maximization with pairwise preference fitting. Relative comparison is fundamentally better suited to this setting for two reasons. First, when candidate images are close in quality, determining which one is better is typically more reliable than assigning well-calibrated absolute scores to each image independently. Second, pairwise comparison discards uninformative score magnitude and focuses optimization on stable ordering information, making the reward signal less sensitive to small fluctuations and common calibration bias. This change also better matches human evaluation, which is usually comparative rather than absolute for similar generations. Motivated by this, we reformulate traditional GRPO with pairwise preference rewards and pairwise win rates, as detailed next.

IV. PREF-GRPO

This section presents the technical formulation of PREFGRPO. We first introduce GRPO on flow matching models in Sec. IV-A, then analyze the illusory advantage phenomenon in Sec. IV-B, and finally describe our PREF-GRPO objective in Sec. IV-C.

A. Flow Matching GRPO

a) Flow Matching: Let x0 ∼ pdata be a data sample and x1 ∼ N(0,I) a noise sample. Rectified flow [27] defines intermediate samples as

##### xt = (1 − t)x0 + tx1, t ∈ [0,1], (1)

and trains a velocity field vθ(xt,t) via the flow matching [28] objective:

0,x1 ∥v − vθ(xt,t)∥22 , v = x1 − x0. (2)

LFM(θ) = Et,x

Beyond training, the iterative denoising process at inference time can be naturally formalized as a Markov Decision Process [29]. We discretize t ∈ [0,1] into T steps. At each step, the state is st = (c,t,xt), where c denotes the prompt. The action

- at corresponds to producing the denoised sample xt−1 ∼ πθ(xt−1|xt,c), yielding a deterministic transition to st+1 = (c,t − 1,xt−1). The initial state samples a prompt c ∼ p(c) with t = T and xT ∼ N(0,I). A reward is only provided at the terminal state (t=0): R(x0,c), and zero otherwise.

b) GRPO on Flow Matching: GRPO [11] introduces a group-relative advantage to stabilize policy updates. On flow matching models, for a group of G generated images {xi0}Gi=1, the advantage of the i-th image is

R(xi0,c) − mean({R(xj0,c)}Gj=1) std({R(xj0,c)}Gj=1)

Aˆit =

. (3)

The policy is updated by maximizing the clipped, KLregularized objective

JFlow(θ) = Ec,{xi} f(r,A,θ,η,βˆ ) , (4)

where

T−1

G

1 G

1 T

f(r,A,θ,η,βˆ ) =

min rti(θ)Aˆit,

(5)

t=0

i=1

clip rti(θ),1 − η,1 + η A ˆit − βDKL(πθ ∥πref).

i t−1|xit,c)

with rti(θ) = pθ(x

pθold(xit−1|xit,c).

To enable the stochastic exploration required by GRPO, Flow-GRPO [1] converts the deterministic flow ODE dxt = vθ(xt,t)dt to an equivalent SDE:

σt2 2t

(xt +(1−t)vθ(xt,t)) dt+σtdwt, (6)

dxt = vθ(xt,t)+

where {wt} is a standard Wiener process and σt is a timedependent noise scale. Euler-Maruyama discretization gives the update rule:

σt2 2t

(xt + (1 − t)vθ(xt,t)) ∆t

xt+∆t = xt + vθ(xt,t) +

√

∆tϵ, ϵ ∼ N(0,I).

+σt

(7) where σt = a t/(1 − t) and a is a scalar hyperparameter.

B. Illusory Advantage in Pointwise-Reward GRPO

Existing flow-matching-based GRPO methods [1–4] use pointwise reward models [8, 9, 12] to score a group of generated images at each update step. The advantage of each image is then obtained by normalizing its reward within the group, as in Eq. 3. This normalization is intended to standardize updates across samples. In practice, however, pointwise reward models often assign tightly clustered scores R(xi0,c) to visually similar images from the same prompt, yielding a small within-group standard deviation σr. As a consequence, even minor score gaps are disproportionately magnified after normalization (see Fig. 1(a)).

To make this effect explicit, let µr and σr denote the mean and standard deviation of the group’s rewards. Define the centered reward of sample i as ∆ri = (xi0,c) − µr. The normalized advantage then becomes:

∆ri σr

Aˆit =

. (8)

For clarity, we drop the clip and KL terms and isolate the effect of reward normalization. The gradient of the GRPO objective with respect to θ is then:

∇θJFlow(θ) = Ec,{xi}

T−1

G

1 G

1 T

∇θ rti(θ)Aˆit . (9)

t=0

i=1

Substituting Aˆit = ∆ri/σr and treating σr as a per-group constant, we obtain:

T−1

G

1 σr

1 G

1 T

∇θ rti(θ)∆ri .

Ec,{xi}

∇θJFlow(θ) =

t=0

i=1

(10) The gradient norm therefore contains an explicit factor of

1/σr:

1 σr

. (11)

∥∇θJFlow∥ ∝

Therefore, when a pointwise reward model assigns nearly identical scores to images within the same group (i.e., R(xi0,c) ≈ R(xj0,c)), the normalization becomes highly sensitive to small perturbations. Even when ∆ri is small, dividing by a small σr produces disproportionately large normalized advantages Aˆit, making the update direction overly dependent on noisy reward differences.

This amplification causes the model to over-optimize for spurious reward differences, manifesting as reward hacking. We term this failure mode the illusory advantage phenomenon. The illusory advantage has two detrimental effects. (1) Excessive optimization: minimal score variations are exaggerated, pushing the policy toward extreme, reward-hacked behaviors (Figs. 2 and 9). (2) Sensitivity to reward noise: the optimization becomes highly susceptible to biases in the reward model, encouraging the policy to exploit model flaws rather than pursue genuine quality improvements.

C. Pairwise Preference Reward-based GRPO

To mitigate the illusory advantage, we propose PREFGRPO, which uses a Pairwise Preference Reward Model (PPRM) [13] to reformulate the optimization objective as pairwise preference fitting. Instead of relying on absolute reward scores, PREF-GRPO evaluates relative preferences among generated images, mirroring the way humans compare images pairwise. This yields a reward signal that better captures nuanced differences in image quality and prompt alignment, producing more stable and informative advantages for policy optimization while reducing susceptibility to reward hacking. Specifically, given a set of G images {xi0}Gi=1 sampled from πθ(· | c) for prompt c, for each ordered pair (i,j) with i ̸= j we use the PPRM to determine whether image i is preferred over image j. The win rate of image i is defined as

1 G − 1 j̸=i

⊮ xi0 ≻ xj0 , (12)

wi =

where ⊮(·) is the indicator function, and xi0 ≻ xj0 indicates that image i is preferred over image j according to the PPRM. The win rates are then used as rewards, replacing the scalar rewards R(xi0,c) in the GRPO advantage:

wi − mean({wj}Gj=1) std({wj}Gj=1)

Aˆit =

. (13)

Compared to reward score maximization, PREF-GRPO offers several advantages: (1) Amplified reward variance: replacing scalar rewards with pairwise win rates spreads win rates across [0,1], yielding a larger within-group variance. High-quality samples approach 1, and lower-quality samples approach 0, producing a more discriminative and robust reward distribution for advantage estimation, thereby mitigating reward hacking. (2) Robustness to reward noise: because optimization relies on relative rankings rather than raw scores, PREF-GRPO reduces sensitivity to small fluctuations and biases in the reward model, lowering the likelihood of reward hacking and improving training stability. (3) Alignment with human preference: human preference judgments over images are inherently relative rather than absolute. Mirroring this process

lets PREF-GRPO capture fine-grained quality distinctions often missed by pointwise scoring, providing a more faithful signal for policy improvement.

V. UNIGENBENCH

Rigorous evaluation is equally important for discerning whether a T2I model faithfully follows complex semantics. This requires a benchmark that goes beyond coarse aggregate scores and probes fine-grained semantic capabilities. However, existing benchmarks [5–7] still exhibit the following limitations: (1) Sparse sub-dimension coverage: existing benchmarks typically include only a few sub-dimensions within each primary dimension, missing many practical capabilities. For example, as shown in Tab. I, current benchmarks include only a single sub-dimension for Relationship and Grammar, leading to incomplete and potentially misleading assessments in these aspects. (2) No sub-dimension-level reporting: existing benchmarks report scores only at the primary-dimension level, without breaking down performance across sub-dimensions. This coarse reporting limits interpretability and prevents diagnosing a T2I model’s strengths and weaknesses.

Motivated by this gap, we propose UNIGENBENCH, a T2I evaluation benchmark designed not only to compare models, but also to reveal where their semantic understanding actually breaks down. UNIGENBENCH offers the following advantages:

- • Comprehensive fine-grained evaluation: 10 primary dimensions and 27 sub-dimensions, enabling substantially finer-grained diagnosis than benchmarks that report only coarse dimension-level scores.
- • Broad prompt coverage: 5 major prompt themes and 20 sub-themes, spanning scenarios from realistic rendering to open-ended creative composition.
- • Efficient benchmark design: 600 prompts, each with 1 to 5 explicit testpoints, achieving broad semantic coverage without the thousands of prompts used by prior benchmarks.
- • Reliable MLLM-based assessment: each prompt is paired with structured testpoint descriptions, enabling precise, interpretable per-testpoint assessment of whether the intended semantic targets are satisfied.

We will first introduce our design of prompt themes and evaluation criteria in the benchmark (Sec. V-A), and then elaborate on our MLLM-based automated pipeline for prompt generation and T2I evaluation (Sec. V-B).

A. Prompt Themes and Evaluation Dimensions

As shown in Fig. 3, UNIGENBENCH covers five major prompt themes: Art, Illustration, Creative Divergence, Design, and Film&Storytelling, further divided into 20 subcategories. It also spans diverse subject categories including animals, objects, anthropomorphic characters, scenes, and an Other category for special entities (e.g., robots in science-fiction themes). This design ensures the benchmark spans both practical generation scenarios and open-ended creative settings, rather than skewing toward a narrow prompt distribution. Representative examples of the prompt themes are shown in Fig. 4.

Prompt Themes of UniGenBench

Evaluation Dimensions of UniGenBench

A row of penguins stood neatly on the ice sheet. They all held their heads high and looked at the aurora...

leather/mechanical/glass... cunning smile/looks sadly at... tiny/small/huge...

An Art Deco sculpture...

triangular/square/cubic... bright pink/dark blue/yellow...

[Figure 30]

Dark fantasy art style, an ancient magic book with several exquisite brass hands sticking out from the cover and slowly turning the pages...

pointed into the distance/ pinching a small leaf/fixing a pin on the skirt...

In an (Impressionist/surrealist...) oil painting...

leaning forward/kicking the glass...

[Figure 31]

pigeons were pecking at wheat grains/ golden-feathered owl on his shoulder...

A modern-style music player App interface/a trendy play-blind box style/ Gothic style/classical ink painting/Ukiyo-e style/retro illustration style...

A picture is generated: a person made of ice, carefully holding a burning flame, his fingers beginning to melt and drop water.

A ceramic piggy bank had a hole in the bottom, and gold coins in it... The words ""Financial Crisis"" were scrawled on a small wooden sign...

gliding through the air/about to pounce on a rabbit... brushing sand off/use its paws to open the latch...

An octopus wearing a kimono is gracefully holding a brush with its tentacles and creating calligraphy on a scroll...

pipe was dripping with water/ hovering/emitting soft purple light...

made of exotic metal crystals/made of countless colorful butterfly wings/composed of translucent jellyfish...

The ticket read "The world is a book and those who do not travel read only one page"/ "The ancient magic awakening from its long slumber now" appeared on the page...

In a steampunk library, Detective Holmes is using a magnifying glass to observe a precision brass gear...

like a huge lightsaber/similar to the outline of Mount Fuji/ two similar transparent glass bottles...

A furry cartoon red panda..., vector the wind flat.

a boy in a red jacket in the center of the frame/ an cat wearing a translation collar was on the left side of the suspension cabin...

contains a quiet town/a library is wrapped in bubbles....

In the futuristic fashion show, the model stood quietly in the center...

taller than/smoother than/smaller than/ flies higher than...

Far behind it, another hamster was watching/a clay sculpture of a boy standing directly above a wooden cube with a yellow star floating behind him...

An environmentally themed logo consists of a leaf, a drop of water...

Instead of being afraid, it curiously stretched out its claws/ all rides are dry, except for a pool of water/the rust is consistent with the color of his coat/Because of the knowledge gained, its tungsten wire becomes bright...

Design an IP image: a Shiba Inu wearing pilot goggles...

its eyes are two bright searchlights, is swimming in the sea of stars/ a surprised robot was feeding a huge pigeon with a French stick...

Please generate a retro poster...

An ancient Roman legion phalanx, soldiers dressed in uniform red armor, holding shields closely connected to form a wall...

no fungus cover on his head/They are not made of glass/ It didn't play any sound/there are no wool balls next to it...

Design the UI interface of a pet health App with a cat

A blue glass cube with a rock texture, a smooth white ceramic disc/ The skin texture of a chameleon perfectly matches the retro-patterned sofa...

The game character design shows a mechanical wolf...

both wore blue collars/ All icons in the interface adopt a round yellow cartoon style...

In the center of ...living room, there is a ...sofa, and the background wall is....

Due to its high temperature/a graffiti on the wall behind him...

A car ad image of Cleopatra...standing next to a black luxury SUV...

a whale's body consists of glowing nebulae and galaxies, swimming in Saturn's rings, with an epic science fiction movie texture...

A young witch stood in front of her magic bookshelf, looking curiously at an ancient book floating in the air...

- Fig. 5. Evaluation dimensions of UNIGENBENCH. Representative prompt examples for each of the 27 sub-dimensions under the 10 primary evaluation dimensions.

[Figure 32]

Distribution of the number of testpoints per prompt

64

155 159 147

75

Number of Testpoints

Number of Prompts

1 2 3 4 5

- Fig. 6. Distribution of testpoint counts per prompt in UNIGENBENCH. Each prompt is associated with 1 to 5 testpoints, balancing coverage and evaluation interpretability.

Fig. 4. Prompt themes of UNIGENBENCH. Representative examples for each of the 5 primary themes and 20 sub-themes.

Unlike coarse benchmarks that only report a few highlevel metrics, UNIGENBENCH defines 10 primary evaluation dimensions and 27 sub-dimensions. As illustrated in Fig. 5, we explicitly include several challenging yet practically important capabilities that are often overlooked, such as logical reasoning, facial expressions, pronoun reference, hand actions, and finegrained relationship understanding (e.g., composition, similarity, and inclusion relations), as well as grammatical consistency across multiple entities. This finer decomposition enables the benchmark to diagnose a model’s strengths and weaknesses more precisely, providing substantially richer feedback than dimension-level averages alone.

At a high level, these ten dimensions cover complementary aspects of semantic faithfulness: Style evaluates adherence to the requested visual form; World Knowledge evaluates consistency with commonsense and real-world facts; Attribute evaluates whether entities exhibit the correct properties; Action evaluates whether entities perform the intended behaviors; Relationship evaluates whether inter-entity relations are grounded correctly; Layout evaluates whether spatial arrangements are rendered accurately; Compound evaluates whether multiple concepts are integrated coherently; Grammar evaluates languagesensitive grounding such as reference, consistency, and negation; Logical Reasoning evaluates causal, contrastive, and inferential understanding; and Text evaluates whether explicit textual content is generated correctly.

a whole still spans a broad set of semantic capabilities.

B. Benchmark Construction and Evaluation Pipeline

Having established diverse prompt themes, subject categories, and evaluation dimensions, we further construct an MLLMbased automated pipeline to operationalize the benchmark shown in Fig. 7. This pipeline serves two complementary purposes: (1) generating large-scale, diverse, and high-quality prompts in a systematic and controllable manner (Sec. V-B1), and (2) enabling scalable, reliable, and fine-grained evaluation of T2I models (Sec. V-B2). By leveraging the reasoning and perception capabilities of MLLMs, the pipeline eliminates the need for costly human annotation, while ensuring both efficiency and reliability in benchmark construction and model assessment.

To make these dimensions operational, we further instantiate them with explicit sub-dimension testpoints. Specifically, Attribute is decomposed into quantity, expression, material, size, shape, and color; Action is decomposed into hand actions, full-body motions, animal behaviors, contact interactions, noncontact interactions, and object states; Relationship is decomposed into composition, similarity, inclusion, and comparison; and Layout is decomposed into 2D placement and 3D spatial arrangement. This design allows UNIGENBENCH to reveal which aspect of semantic generation actually fails, rather than only reporting a coarse dimension-level average.

1) Prompt and Testpoint Description Generation: Let T

denote the set of prompt themes, S the set of subject categories, and C the set of evaluation dimensions. For each prompt, we sample a theme t ∼ T and a subject category s ∼ S uniformly at random. Subsequently, a subset of k testpoints {c1,...,ck} ⊂ C, with k ∈ [1,5], is sampled to target specific fine-grained evaluation aspects.

We further control the complexity of each prompt through the number of associated testpoints. Instead of relying on thousands of prompts with diffuse evaluation targets, UNIGENBENCH uses 600 prompts, each focusing on 1 to 5 specific testpoints. As shown in Fig. 6, this design achieves a practical balance between coverage and efficiency: each prompt remains interpretable and evaluation-friendly, while the benchmark as

The selected tuple (t,s,{c1,...,ck}) is input into the MLLM, which generates two outputs: (i) a natural language prompt p that conforms to the semantic constraints of the selected theme t and subject category s, and (ii) a structured description set {d1,...,dk}, where each di specifies how the

UniGenBench Construction and Evaluation Pipeline

each of HPSv2 [9], CLIP [12], and UnifiedReward (UR) [8] as the pointwise reward.

[Figure 33]

Content

(a) Prompt & TestPoint Description Generation

Creative Divergence Imagination

[Figure 34]

Subject a huge pandaOn theis GreatwearingWalla spacesuit,,

[Figure 35]

Training. Using the pipeline in Fig. 7(a), we generate 5k training prompts, disjoint from the 600 UNIGENBENCH evaluation prompts. The same 5k prompts are used to train PREF-GRPO and all reward-maximization baselines. Training is conducted on 64 H20 GPUs with 25 sampling steps, 8 rollouts per prompt from the same initial noise xT, 4 gradient accumulation steps, and a learning rate of 1×10−5. Following Flow-GRPO [1], the SDE noise-scale hyperparameter is set to a = 0.7. We host the PPRM as an inference server using vLLM [31].

###### Animal

with a small butterfly made of a nebula lying on its helmet.

1. World Knowledge 2. Grammar - Pronoun Reference

[Figure 36]

Testpoint

Gemini2.5 Pro

- 3. Compound - Imagination

- 4. Relationship - Composition

5. Attribute - Size

T2I Model

- 1. World Knowledge: [The image correctly depicts a structure resembling the Great Wall...]
- 2. Grammar - Pronoun Reference: [The image shows a butterfly resting on top of the helmet that the panda is wearing, correctly interpreting the pronoun reference "its helmet".]
- 3. Compound - Imagination: [The image successfully depicts a panda wearing a spacesuit. However, the butterfly is not made of a nebula but appears as a normal, solid insect.]

- 4. Relationship - Composition: [The butterfly shown in the image is not made of a nebula, lacking the gaseous, starry texture of a nebula.]
- 5. Attribute - Size: [The panda is depicted as a large, central figure, which can be interpreted as "huge." The butterfly is depicted as being very small, especially in contrast to the panda and its helmet.]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Prompt

[Figure 41]

TestPoint Description

[Figure 42]

(b) T2I Model Evaluation

[Figure 43]

- Fig. 7. UNIGENBENCH construction and evaluation pipeline. An MLLM (Gemini-2.5-Pro) is used for (a) prompt and testpoint-description generation and (b) fine-grained per-testpoint evaluation.

Inference. At inference, we generate 1024×1024 images with 30 sampling steps and a classifier-free guidance scale of 3.5, following the official FLUX.1-dev configuration. To ensure a fair comparison, all methods share identical prompts and initial noise at evaluation.

corresponding testpoint ci is realized in the prompt. Formally, this process can be expressed as:

Evaluation. We evaluate semantic consistency on UNIGENBENCH, GenEval [5], and T2I-CompBench [6]. For each test prompt, we generate four images and report the average score. Image quality is further evaluated with UnifiedReward [8], ImageReward [32], PickScore [10], and Aesthetic [33]. ImageReward, PickScore, and Aesthetic are disjoint from the training rewards and serve as the independent quality metrics.

(p,{d1,...,dk}) ∼ MLLM p,{di} t,s,{c1,...,ck} .

(14)

2) T2I Model Evaluation: Given the generated images {xi} for benchmark prompts {pi}, we evaluate each image using an MLLM. Specifically, the image xi, its corresponding prompt pi, and its testpoint descriptions {di,1,...,di,k} are provided as input. The MLLM evaluates each testpoint di,j in the context of xi, producing a binary score ri,j ∈ {0,1} and a textual rationale ei,j justifying the assessment. This can be formally represented as:

B. Main Results: PREF-GRPO vs. Reward-Maximization Baselines

Quantitative Results. Tab. II shows that PREF-GRPO delivers clear semantic gains on UNIGENBENCH. Relative to the strongest score-maximization baseline (UR), it improves the overall score by 5.84%, with especially large gains on challenging dimensions such as Text (+12.69%) and Logical Reasoning (+12.04%). This pattern is consistent with our motivation in Sec. IV-B: pairwise preferences provide a cleaner gradient signal than pointwise scores on dimensions that require fine-grained semantic grounding and compositional understanding.

(ri,1,...,ri,k,ei,1,...,ei,k) ∼ MLLM {ri,j,ei,j} xi,pi,{di,1,...,di,k} .

(15)

This process ensures that the evaluation captures both the quantitative performance on each testpoint and the qualitative reasoning behind the assessment.

After obtaining the scores ri,j for each testpoint di,j in all generated images, we aggregate them to compute scores of sub- and primary evaluation dimensions. Specifically, for each sub-dimension c, we define its score as the ratio of the number of times the model successfully satisfies the corresponding testpoint description to the total number of occurrences of that testpoint across the benchmark:

The gains also generalize beyond our held-out evaluation on UNIGENBENCH. As shown in Tabs. VI and IV, PREFGRPO consistently outperforms reward-maximization baselines on both GenEval and T2I-CompBench, indicating improved out-of-domain compositional reasoning, attribute binding, and spatial understanding. Moreover, Tab. III shows that these semantic gains do not come at the expense of perceptual quality: PREF-GRPO achieves the best score on all four image-quality metrics. Overall, pairwise preference fitting improves semantic alignment without sacrificing image quality.

1{di,j ∈ c and ri,j = 1}

Rc = i,j

, (16)

i,j 1{di,j ∈ c}

where 1{·} is the indicator function. The overall score for a primary dimension C is then obtained by averaging the scores of all its sub-dimensions. This procedure ensures that both fine-grained performance on sub-dimensions and broader performance on primary dimensions are captured.

Qualitative Results. Examples are shown in Fig. 8. Existing methods exhibit varying degrees of reward hacking: HPSv2optimized images tend to be oversaturated, while UR-optimized images drift toward an unnaturally dark style. We also explore mitigating reward hacking by combining multiple reward scores, i.e., summing HPSv2 and CLIP scores during training (third row in Fig. 8). While this partially reduces reward hacking, it does not eliminate it. In contrast, PREF-GRPO avoids these artifacts and faithfully renders the prompt semantics. A detailed analysis of reward hacking under UnifiedReward and HPSv2 is provided in Sec. VII-A.

VI. EXPERIMENTS A. Experimental Setup

Baselines. We use FLUX.1-dev [30] as the base T2I model and UnifiedReward-Think [13] as the pairwise preference reward model (PPRM) in PREF-GRPO. As reward-maximization baselines, we train FLUX.1-dev with Flow-GRPO [1] using

TABLE II IN-DOMAIN SEMANTIC CONSISTENCY COMPARISON ON UNIGENBENCH. FLUX.1-DEV FINE-TUNED WITH FLOW-GRPO USING DIFFERENT REWARD MODELS, VERSUS OUR PREF-GRPO. GEMINI-2.5-PRO IS USED AS THE EVALUATOR. BEST SCORES ARE IN BOLD, SECOND-BEST ARE UNDERLINED.

World Know.

Logic Reason.

Model Overall Style

Attr. Action Rel.

Grammar Compound Layout Text

FLUX.1-dev 61.30 83.90 88.92 67.84 62.17 67.26 30.91 60.96 47.04 71.83 32.18 w/ HPSv2 58.77 75.20 88.77 66.56 58.94 66.88 28.18 58.02 45.88 67.91 31.32 w/ HPSv2&CLIP 61.81 84.92 88.98 68.44 62.54 68.10 31.01 59.36 50.60 71.07 33.07 w/ UnifiedReward 63.62 86.10 89.72 71.55 63.69 70.42 32.05 62.43 52.32 73.51 34.44

###### w/ Pref-GRPO 69.46 88.40 90.35 75.00 69.77 76.52 44.09 63.27 62.43 77.61 47.13

Tsaurus rex riding a shark on the sea surface, its forelimbs holding a rocket launcher, realistic style.

A forest of giant crayons growing like trees.

A surfboard shaped like a lightning bolt, balanced on a volcanic rock.

A giraffe, a lion, and an elephant stacked like a tower.

A desert scene with sand that transitions from gold to sapphire blue.

A beige pastry sitting in a white ball next to a spoon.

Kung Fu Panda pointing to a wooden sign, which reads"The Dragon Warrior is always hungry for more dumplings."

A dragon made entirely of swirling mist and lightning.

An enormous hand rising out of the ocean, holding a city.

In the laboratory, an orange cat is floating, its paws touching an ball.

A person and a airplane, the person is bigger than the airplane.

A dragon-shaped teapot.

McDonalds Church.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Baseline

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

HPSv2

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

HPSv2+CLIP

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

UnifiedReward

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Pref-GRPO (Ours)

- Fig. 8. Qualitative comparison. Rows show FLUX.1-dev trained with HPSv2, HPSv2+CLIP, UnifiedReward, and PREF-GRPO. Pointwise-reward baselines exhibit reward-hacked artifacts (oversaturation for HPSv2, dark style for UnifiedReward), while PREF-GRPO preserves semantic faithfulness and the original FLUX.1-dev visual style.

TABLE III IMAGE QUALITY COMPARISON. FLUX.1-DEV FINE-TUNED WITH FLOW-GRPO USING DIFFERENT REWARD MODELS, VERSUS OUR PREF-GRPO. IMAGEREWARD, PICKSCORE, AND AESTHETIC ARE NOT USED AS TRAINING REWARDS AND SERVE AS INDEPENDENT QUALITY METRICS. BEST RESULTS ARE IN BOLD, SECOND-BEST ARE UNDERLINED.

Unified Reward

Pick Score

Image Reward

Model

Aes.

FLUX.1-dev 3.04 22.42 1.27 6.13 w/ HPSv2 3.09 22.62 1.34 6.20 w/ HPSv2+CLIP 3.08 22.61 1.30 6.25 w/ UnifiedReward 3.14 22.88 1.38 6.31

w/ Pref-GRPO 3.26 23.02 1.44 6.52

C. Benchmarked T2I Models on UNIGENBENCH

We further benchmark representative closed-source and opensource T2I models on UNIGENBENCH, spanning diffusion, MMDiT, autoregressive, and unified architectures. Closedsource models are accessed via their official APIs, and opensource models via their released checkpoints, both using the providers’ default inference settings.

Closed-source Models. We benchmark GPT-4o [15] (OpenAI), Imagen-4.0-Ultra [17] (Google), Seedream-3.0 and Seedream-4.0 [16] (ByteDance), Nano Banana (Google), DALL-E-3 (OpenAI), FLUX-Pro-Ultra and Kontext-Max [30] (Black Forest Labs), and Keling-Ketu (Kuaishou).

Open-source Models. We benchmark Qwen-Image [18], HiDream-I1-Full [19], Show-o2 [34], SD-3.5-Large [35], SDXL [35], FLUX.1-dev [30], CogView4 [36], Hunyuan-

DiT [37], Playground 2.5 [38], Janus [39], Janus-Pro [40], Janus-Flow [41], Emu3 [42], Bagel [20], and BLIP3-o [43].

D. Benchmarking Results on UNIGENBENCH

As shown in Tab. V, UNIGENBENCH reveals a clear yet nuanced performance hierarchy. Closed-source models remain the strongest overall, with GPT-4o [15] and Imagen-4.0-Ultra [17] leading on 7 of 10 primary dimensions. Their advantage is most pronounced on dimensions that require deeper semantic grounding (e.g., Logical Reasoning, Relationship, Compound, and Text) rather than low-level visual quality. By contrast, dimensions such as Style and World Knowledge are already saturated across many models (most score above 0.8), indicating they are no longer the main bottlenecks for current T2I systems. This contrast illustrates why coarse overall scores are insufficient: the real gap between strong and weak models lies in higher-order semantic composition, not in basic visual plausibility.

Open-source models are closing the gap but remain less balanced across dimensions. Qwen-Image [18] is the strongest open-source model overall, and HiDream-I1-Full [19] is competitive on Action, Layout, and Attribute. Compared with leading closed-source systems, however, open-source models show larger variance across the 10 primary dimensions, with noticeably weaker performance on Logical Reasoning, Grammar, and Compound. This indicates the remaining gap is not primarily one of image quality, but of fine-grained instruction grounding and coherently satisfying multiple constraints within a single prompt. UNIGENBENCH thus serves not only as a

TABLE IV OUT-OF-DOMAIN PERFORMANCE COMPARISON ON T2I-COMPBENCH. FLUX.1-DEV FINE-TUNED WITH FLOW-GRPO USING DIFFERENT REWARD MODELS, VERSUS OUR PREF-GRPO. BEST RESULTS ARE IN BOLD, SECOND-BEST ARE UNDERLINED.

2D Spatial

3D Spatial

NonSpatial

Model Overall Color Shape Texture

Numeracy

Complex

FLUX.1-dev 48.17 77.34 48.32 62.66 28.01 40.04 61.88 30.67 36.49 w/ HPSv2 46.77 78.17 51.55 66.13 22.06 33.75 56.34 30.20 35.96 w/ HPSv2+CLIP 49.18 78.44 53.22 64.24 26.90 40.83 61.58 30.56 37.69 w/ UnifiedReward 50.20 78.32 55.13 67.44 28.91 40.04 62.47 30.88 38.39

###### w/ Pref-GRPO 51.85 80.27 56.01 69.12 28.93 43.95 65.92 31.05 39.58

TABLE V BENCHMARKING RESULTS ON UNIGENBENCH. WE REPORT THE OVERALL SCORE AND PRIMARY-DIMENSION BREAKDOWN FOR THE EVALUATED T2I MODELS. BEST AND SECOND-BEST SCORES ARE COMPUTED SEPARATELY WITHIN THE CLOSED- AND OPEN-SOURCE GROUPS; THEY ARE MARKED IN BOLD AND UNDERLINED, WITH TOP-3 PER GROUP ANNOTATED BY THE MEDAL ICONS.

World Know.

Logic Reason.

Model Overall Style

Attr. Action Rel.

Grammar Compound Layout Text

Closed-source Models

Keling-Ketu 65.93 92.27 86.62 71.66 68.73 70.94 43.75 71.26 60.81 77.23 16.03 DALL-E-3 68.85 94.43 92.64 75.76 70.78 78.31 46.22 69.22 71.08 65.65 24.43 FLUX-Pro-Ultra 70.46 90.99 91.30 76.79 71.39 78.05 41.46 68.18 68.17 80.60 37.64

- Seedream-3.0 78.41 98.19 94.90 84.62 83.14 80.18 51.83 60.30 72.32 88.74 69.86 FLUX-Kontext-Max 80.00 96.59 94.19 80.93 77.38 85.08 61.36 78.53 78.99 85.04 61.92
- Seedream-4.0 87.35 98.80 95.41 88.57 85.65 87.69 67.73 78.88 86.08 90.67 93.97 Nano Banana 87.29 98.59 96.20 87.99 87.36 92.47 73.41 83.82 88.34 91.42 73.28 Imagen-4.0-Ultra 91.65 99.10 97.78 92.09 92.10 93.53 80.45 87.83 91.37 92.91 89.37 GPT-4o 92.48 98.98 98.22 94.01 90.78 94.33 83.79 91.21 92.89 91.35 89.24

[Figure 109]

[Figure 110]

[Figure 111]

###### Open-source Models

SDXL 40.22 87.45 72.28 44.66 35.10 46.37 10.34 48.48 26.68 30.80 0.00 Playground 2.5 45.61 89.50 76.11 52.78 42.68 51.52 16.59 53.21 35.44 37.13 1.15 Emu3 45.42 87.50 76.42 50.11 40.40 48.60 19.32 50.67 36.21 43.84 1.15 Janus-flow 47.10 86.34 62.98 49.20 43.57 51.45 22.41 62.80 46.49 45.76 0.00 Janus 51.60 90.08 73.56 55.34 50.92 56.54 28.74 61.74 47.10 52.01 0.00 Hunyuan-DiT 51.38 94.10 80.70 62.71 49.05 59.64 24.55 55.48 41.62 44.78 1.15 CogView4 56.00 80.80 81.96 63.14 59.51 60.91 27.95 54.81 44.97 69.03 16.95 BLIP3-o 59.57 92.81 79.97 64.77 64.59 65.99 36.78 69.05 54.57 67.19 0.00 FLUX.1-dev 60.97 85.00 87.50 67.20 62.26 66.88 29.77 62.30 45.75 70.90 32.18 Bagel 59.91 90.08 85.42 67.73 62.14 70.64 23.85 65.85 56.86 76.56 0.00 Janus-Pro 61.36 90.40 86.55 68.59 63.88 69.54 35.68 64.04 60.18 72.76 2.01 Show-o2 61.90 87.40 85.44 69.87 69.01 68.78 39.55 60.83 63.79 73.13 1.15 SD-3.5-Large 62.89 88.60 89.72 68.80 61.98 67.51 32.05 59.89 58.38 67.72 34.20

Pref-GRPO 69.46 88.40 90.35 75.00 69.77 76.52 44.09 63.27 62.43 77.61 47.13 HiDream-I1-Full 71.36 92.30 93.67 73.40 72.53 74.24 40.45 62.43 60.31 77.61 66.67 Qwen-Image 78.36 94.70 94.15 87.93 82.60 80.08 51.59 60.96 72.94 86.57 72.13

[Figure 112]

[Figure 113]

[Figure 114]

TABLE VI OUT-OF-DOMAIN PERFORMANCE COMPARISON ON GENEVAL. FLUX.1-DEV FINE-TUNED WITH FLOW-GRPO USING DIFFERENT REWARD MODELS, VERSUS OUR PREF-GRPO. BEST RESULTS ARE IN BOLD, SECOND-BEST ARE UNDERLINED.

Single Obj.

Two Obj.

Attr. Bind.

Count. Color Pos.

Model Overall

FLUX.1-dev 62.92 97.81 79.55 71.56 77.66 18.50 42.25 w/ HPSv2 59.31 97.43 75.00 62.81 73.67 21.00 34.75 w/ HPSv2+CLIP 64.85 98.12 81.00 71.81 78.44 19.00 40.75 w/ UnifiedReward 67.28 98.43 82.57 72.25 79.72 21.25 49.50

w/ Pref-GRPO 70.53 99.38 86.36 74.06 81.12 26.00 57.25

leaderboard, but also as a diagnostic benchmark that exposes where current T2I models still fail.

Tab. VII further sharpens this diagnosis by exposing the subdimension breakdown hidden beneath the primary-dimension averages. Among closed-source models, GPT-4o leads on semantically fine-grained sub-dimensions such as expression,

non-contact interaction, Grammar, and Logical Reasoning, whereas Imagen-4.0-Ultra leads on structure-sensitive subdimensions including quantity, contact interaction, composition, and 3D spatial arrangement (Layout). Even among top-tier systems, strengths are not uniform: some models are stronger on abstract semantic interpretation, others on preserving spatial and compositional structure. The fine-grained breakdown is even more informative for open-source models. Qwen-Image shows the most balanced profile and leads in a majority of subdimensions among open-source systems, indicating broad rather than narrow improvements. HiDream-I1-Full is competitive on several geometry- and action-related sub-dimensions, but remains visibly weaker on reasoning-intensive dimensions (Logical Reasoning, Grammar, and Compound). Crucially, models with similar primary-dimension averages can still differ substantially in the specific sub-dimensions they master, which UNIGENBENCH makes explicit.

TABLE VII FINE-GRAINED BENCHMARKING RESULTS OF THE MODELS REPORTED IN TAB. V. WE REPORT SUB-DIMENSION BREAKDOWNS FOR OVERLAPPING MODELS WITH AVAILABLE FINE-GRAINED ANNOTATIONS. BEST SCORES ARE IN BOLD, SECOND-BEST IN UNDERLINED, COMPUTED SEPARATELY FOR CLOSED- AND OPEN-SOURCE MODELS.

World Know.

Logic Reason.

Model Style

Attribute Action Relationship Compound Grammar Layout

Text

Full Body

Non Contact

Feat. Match.

Pron. Ref.

Quant. Express. Materi. Size Shape Color Hand

Contact State Compos. Sim. Inclus. Compare. Imagin.

Animal

Consist. Neg. 2D 3D

Closed-source Models

Keling-Ketu 92.25 87.08 74.29 56.77 78.67 74.83 53.75 89.66 53.85 72.28 71.32 70.77 59.28 75.94 68.14 69.27 72.13 69.29 66.15 53.03 74.91 64.19 66.80 77.61 71.43 45.60 16.03 DALL-E-3 94.43 92.64 60.14 63.16 87.20 84.72 66.25 91.60 60.78 76.67 77.94 68.72 63.19 76.19 82.99 71.51 85.47 66.93 78.01 63.95 76.34 72.09 59.45 54.78 77.25 46.22 24.43 FLUX-Pro-Ultra 90.99 91.30 72.92 60.65 79.25 75.00 78.12 98.33 58.97 69.02 76.47 78.06 65.48 77.83 81.08 74.44 80.98 71.88 77.30 58.85 83.46 65.74 54.23 81.25 79.92 41.46 37.64

- Seedream-3.0 98.19 94.90 79.02 81.94 89.62 83.80 77.22 96.67 75.97 89.56 86.03 75.38 81.93 89.10 81.57 74.16 83.61 80.47 76.92 67.62 77.94 68.40 35.14 88.15 89.35 51.83 69.86 FLUX-Kontext-Max 96.59 94.19 75.69 74.32 82.55 86.81 74.38 94.17 67.95 83.15 77.94 77.04 70.83 84.43 87.50 78.89 90.00 81.25 83.93 73.96 84.23 78.70 72.69 86.74 83.33 61.36 61.92

- Seedream-4.0 98.80 95.41 86.81 85.90 97.17 84.03 76.88 100.00 77.56 87.50 88.24 80.10 83.93 94.81 88.18 80.56 94.02 87.50 88.27 83.85 84.93 79.17 72.31 90.81 90.53 67.73 93.97 Nano Banana 98.59 96.20 86.43 80.77 88.46 95.83 80.77 98.33 80.13 93.48 88.24 83.67 80.95 95.28 93.49 86.67 94.02 96.09 90.21 86.46 90.44 83.33 77.31 93.01 89.77 73.41 73.28 Imagen-4.0-Ultra 99.10 97.78 94.44 80.77 95.28 94.44 88.75 100.00 89.74 93.41 93.38 88.78 87.50 98.58 96.28 87.78 96.20 91.41 92.86 89.84 91.91 90.28 81.54 93.75 92.05 80.45 89.37 GPT-4o 98.98 98.22 89.29 96.00 94.66 92.96 92.50 99.17 88.46 93.33 87.88 92.02 89.16 92.31 96.58 91.11 94.89 92.97 94.07 91.67 91.04 93.06 89.75 92.16 90.53 83.79 89.24

[Figure 115]

[Figure 116]

[Figure 117]

###### Open-source Models

SDXL 87.45 72.28 41.67 25.00 54.90 44.85 36.11 68.52 19.74 38.10 45.31 26.74 24.34 52.40 55.38 41.22 38.75 43.33 33.75 19.94 54.58 41.67 47.46 25.00 36.40 10.34 0.00 Playground 2.5 89.78 75.80 60.61 43.59 58.33 45.59 39.58 81.48 29.61 54.17 54.69 37.21 28.29 57.21 63.46 51.35 48.75 40.00 44.06 28.27 62.92 51.11 49.58 33.18 39.47 16.09 0.00 Emu3 87.50 76.42 42.36 45.51 52.83 40.28 46.25 77.50 23.08 49.46 54.41 34.69 29.17 50.47 55.41 44.44 46.74 41.41 41.33 30.99 58.09 49.07 44.23 42.28 45.45 19.32 1.15 Janus-flow 86.34 62.98 43.18 30.77 55.39 57.35 33.33 82.41 22.37 48.81 57.81 38.95 36.84 54.81 62.69 36.49 53.75 42.50 60.00 33.63 70.00 51.11 64.41 46.82 44.74 22.41 0.00 Janus 90.08 73.56 35.61 37.82 60.29 66.18 48.61 90.74 31.58 52.38 62.50 50.00 39.47 65.87 58.85 52.70 61.25 50.00 59.38 35.42 70.00 52.22 60.59 51.82 52.19 28.74 0.00 Hunyuan-DiT 94.10 80.70 67.36 44.23 71.70 61.81 47.50 86.67 35.90 54.89 54.41 46.94 35.71 62.74 60.14 64.44 60.33 50.78 46.68 36.46 62.87 57.87 45.77 39.34 50.38 24.55 1.15 CogView4 80.80 81.96 70.83 46.79 55.66 68.75 58.75 87.50 57.69 59.78 69.85 52.55 53.57 65.09 58.11 60.00 66.30 60.94 49.23 40.62 69.49 54.17 40.00 76.84 60.98 27.95 16.95 BLIP3-o 92.81 79.97 48.48 60.26 66.67 76.47 56.94 83.33 57.24 71.43 71.09 63.95 50.66 71.15 70.77 57.43 66.25 65.83 64.06 45.54 81.67 61.11 62.29 69.55 64.91 36.78 0.00 FLUX.1-dev 85.00 87.50 71.53 51.92 58.96 74.31 65.62 90.00 50.00 69.02 69.12 60.20 61.90 63.21 66.89 65.56 72.83 60.16 46.17 45.31 76.47 61.57 48.08 74.63 67.05 29.77 32.18 Bagel 90.08 85.42 56.82 50.00 73.53 77.94 59.03 94.44 51.32 64.88 67.19 64.53 56.58 66.83 77.31 68.92 70.00 59.17 67.50 46.73 74.17 64.44 58.47 77.73 75.44 23.85 0.00 Janus-Pro 90.40 86.55 56.25 57.69 74.06 73.61 61.88 90.83 47.44 65.22 72.79 60.71 59.52 75.47 76.01 58.33 73.91 64.06 67.35 52.86 76.10 64.81 50.77 74.63 70.83 35.68 2.01 Show-o2 87.40 85.44 59.03 64.10 70.75 74.31 61.25 95.00 54.49 75.00 75.00 72.45 50.60 82.08 76.35 60.56 71.20 59.38 66.84 60.68 77.57 63.43 41.15 75.37 70.83 39.55 1.15

SD-3.5-Large 88.60 89.72 69.44 51.28 70.28 70.83 64.38 91.67 57.69 63.04 62.50 59.69 58.93 68.40 73.99 65.00 66.30 57.81 68.37 48.18 77.21 60.19 41.54 70.96 64.39 32.05 34.20 HiDream-I1-Full 92.30 93.67 73.61 61.54 72.17 79.17 62.50 98.33 60.90 76.09 74.26 73.98 68.45 78.77 76.69 67.78 78.26 71.88 61.99 58.59 81.62 63.89 41.15 82.72 72.35 40.45 66.67 Qwen-Image 94.70 94.15 84.03 85.26 91.98 86.11 81.88 99.17 78.21 86.96 86.76 77.55 76.79 88.68 82.09 71.11 86.96 78.12 73.21 72.66 84.93 70.37 28.08 87.13 85.98 51.59 72.13

[Figure 118]

[Figure 119]

[Figure 120]

###### VII. ANALYSES AND ABLATIONS

reward hacking earlier than UnifiedReward, consistent with its smaller intra-group score variance in Fig. 1(a), which intensifies the amplification effect. By contrast, PREF-GRPO optimizes relative preferences rather than absolute-score increments, leading to substantially more stable training behavior.

In the style of classical ink painting, against the background of Egyptian pyramids, two pandas wearing spacesuits are leisurely tasting bamboo.

FLUX w/HPSv2

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

0.35

Several complementary observations support this mitigation. First, Fig. 1(a) shows that PREF-GRPO maintains a larger intra-group reward variance, directly suppressing the illusoryadvantage amplification mechanism. Second, Figs. 2 and 9 show that, unlike pointwise-score GRPO, PREF-GRPO does not drift toward visually degraded solutions during prolonged training. Third, the qualitative comparison in Fig. 8 shows that PREF-GRPO preserves the original FLUX.1-dev visual style and does not collapse to reward-hacked artifacts. Finally, as we show next in Sec. VII-B, a controlled point-score-derived win-rate variant further confirms that replacing absolute-score maximization with a relative-comparison objective is itself the key stabilizing factor.

HPSv2 Score

[Figure 125]

0.34

Design a cat with butterfly wings, its whole body is made of translucent glass.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Hack

0.33

[Figure 130]

80 160 240 320

Training Steps

Baseline w/HPSv2 w/Pref-GRPO

- Fig. 9. Reward hacking under HPSv2. HPSv2 scores continue to rise, but image quality collapses around step 160, manifesting as severe over-saturation.

A. Reward Hacking in Pointwise GRPO

We analyze reward hacking under two representative pointwise reward models, UnifiedReward [8] and HPSv2 [9]. For UnifiedReward, Fig. 2 shows that reward scores rise rapidly during training, yet the generated images gradually drift toward an unnatural dark style and perceptual quality degrades. For HPSv2, the same failure mode emerges even more abruptly in Fig. 9: the reward keeps rising, while image quality starts to collapse at around step 160, accompanied by severe oversaturation. Taken together, these results reveal a consistent mismatch between reward growth and real generation quality under reward score maximization.

B. Point-Score vs. Pairwise-Preference Win Rates

To disentangle the effect of the optimization objective from that of the reward model itself, we introduce a controlled variant based on point score-derived win rates. Specifically, we first compute UnifiedReward scores for all images in a rollout group, then convert these scores into pairwise win rates by comparing every image pair, and finally use the resulting win rates as rewards for training. This keeps the underlying reward model unchanged while replacing reward score maximization with a relative-comparison objective.

This mismatch is consistent with the illusory advantage analysis in Sec. IV-B. Once pointwise reward scores within a rollout group become highly compressed, GRPO amplifies tiny score gaps into large normalized advantages, causing the policy to over-optimize for spurious reward cues rather than genuine improvements in alignment or aesthetics. HPSv2 exhibits

This ablation isolates the source of the stability gain. If the improvement of PREF-GRPO came solely from using a stronger pairwise reward model, then converting UnifiedReward scores into win rates should offer little benefit. However,

w/UR Score-based winrate as Reward

TABLE IX WALL-CLOCK REWARD COMPUTING TIME PER TRAINING STEP. “REQUESTS/STEP” COUNTS THE NUMBER OF REWARD-MODEL FORWARD COMPUTATIONS INVOKED BY EACH SCHEME IN A TRAINING STEP. Reward type

FLUX.1-dev w/UR Score as Reward

[Figure 131]

[Figure 132]

[Figure 133]

Rollouts / group

Requests / step

Time / step

8 32 3s 16 64 5s

Point score (UnifiedReward)

Please create a clay sculpture: two anthropomorphic foxes. The fox on the left side of the picture is dancing happily, and the fox on the right side is sitting and applauding it.

[Figure 134]

[Figure 135]

[Figure 136]

8 112 7s 12 264 14s 16 480 22s

Pairwise winrate (Ours)

Pref-GRPO w/ CLIP

[Figure 137]

[Figure 138]

A dragon made entirely of swirling mist and lightning.

- Fig. 10. Point-score-derived win-rate training alleviates reward hacking. Converting UnifiedReward scores into pairwise win rates substantially mitigates the dark-style drift observed under direct score maximization, supporting relative preferences (not the reward model itself) as the key stabilizing factor.

On the Great Wall, a huge panda is wearing a spacesuit, with a small butterfly made of a nebula lying on its helmet.

TABLE VIII POINT-SCORE-DERIVED WIN RATE VS. PAIRWISE PREFERENCE WIN RATE. COMPARED TO DIRECT SCORE MAXIMIZATION (“W/ UR (SCORE)”), CONVERTING UNIFIEDREWARD SCORES INTO PAIRWISE WIN RATES (“W/ UR (SCORE WIN RATE)”) ALREADY IMPROVES STABILITY; PREF-GRPO FURTHER BENEFITS FROM A NATIVE PAIRWISE PREFERENCE REWARD. BEST RESULTS ARE IN BOLD, SECOND-BEST ARE UNDERLINED.

[Figure 139]

[Figure 140]

In the trendy blind box style, an anthropomorphic steamed bun immortal stands on one leg on a huge abacus, and the small dumpling next to him is looking up at it in surprise.

Unified Reward

Pick Score

Image Reward

Model UniGen GenEval

Aes.

FLUX.1-dev 61.30 62.92 3.04 22.42 1.27 6.13 w/ UR (score) 63.62 67.28 3.14 22.88 1.38 6.31 w/ UR (score winrate) 64.32 68.13 3.20 22.91 1.39 6.35 w/ Pref-GRPO 69.46 70.53 3.26 23.02 1.44 6.52

Fig. 11. Joint optimization of PREF-GRPO with a CLIP score reward. Adding CLIP improves semantic consistency while slightly reducing perceptual quality, but does not trigger reward hacking. The pairwise preference fitting acts as a stabilizing regularizer for the auxiliary score signal.

Fig. 10 shows the opposite: once the optimization target is changed from reward score maximization to win-rate fitting, the dark-style reward hacking observed under UnifiedReward is already substantially alleviated. Quantitative results in Tab. VIII further confirm this trend: training with point-score-derived win rates consistently outperforms reward score maximization, and PREF-GRPO, which uses native pairwise preference rewards, achieves the best overall performance.

4× H800 GPUs, with the reward inference service deployed on another 4 H800 GPUs via vLLM [31]; this isolates perstep cost from the multi-node overhead of the 64-GPU main setup. We compare two settings across different group sizes: (i) point-score rewards from UnifiedReward, and (ii) pairwise preference win-rate rewards used in PREF-GRPO.

Tab. IX shows that the added cost is moderate. Under the default setting G=8, pairwise win-rate rewards increase the per-step reward time from 3s to 7s, an additional 4s of overhead. At G=16, the number of reward-model requests grows from 64 to 480 (7.5×), yet wall-clock reward time grows only from 5s to 22s (3.1×), far sublinear in the number of pairwise comparisons. Pairwise supervision is therefore effectively amortized by parallel reward serving.

These results support two conclusions. First, a significant part of the instability in prior GRPO methods arises from the objective formulation, not merely from imperfections in the reward model. Second, pairwise preference rewards outperform score-derived win rates, indicating that native pairwise supervision provides cleaner and more faithful relative signals than rankings derived from compressed pointwise scores. In short, PREF-GRPO benefits from both components: relative-preference optimization is the key stabilizing factor, strengthened further by a dedicated pairwise reward model. A practical corollary is that practitioners with only scalar rewards can adopt win-rate fitting as a drop-in stability improvement.

The main concern with pairwise rewards is theoretical complexity, not practical infeasibility. In our setting, G=8 already yields strong empirical gains (Tab. III, Figs. 2, 8), so the added reward latency remains modest relative to the training benefit. Overall, PREF-GRPO achieves a favorable computation–performance trade-off: substantially improved generation quality and mitigated reward hacking, with only limited wall-clock overhead.

C. Wall-Clock Overhead of Pairwise Rewards

Although pairwise rewards require O(G2) comparisons within each rollout group of size G, the practical question is whether this additional computation becomes a prohibitive bottleneck during training. To answer this, we measure wallclock reward computation time when training FLUX.1-dev on

D. Joint Optimization with Auxiliary Score Rewards

To examine whether score-based rewards can still provide useful auxiliary supervision, we add a CLIP [12] reward on top

TABLE X JOINT OPTIMIZATION OF PREF-GRPO AND CLIP-BASED REWARD SCORE MAXIMIZATION. BEST RESULTS ARE IN BOLD.

Unified Reward

Pick Score

Image Reward

Model UniGen GenEval

Aes.

Pref-GRPO 69.46 70.53 3.26 23.02 1.44 6.52 w/ CLIP 70.02 71.26 3.18 22.86 1.41 6.44

TABLE XI ROBUSTNESS OF PREF-GRPO UNDER 10% PREFERENCE NOISE. THE NOISY VARIANT FLIPS EACH PAIRWISE PREFERENCE OUTCOME WITH PROBABILITY p=0.1 DURING TRAINING.

T2IComp

Unified Reward

Pick Score

Method UniGen

GenEval

FLUX.1.dev 61.30 48.17 62.92 3.04 22.42 w/ HPSv2+CLIP 58.77 46.77 59.31 3.09 22.62 w/ UnifiedReward 63.62 50.20 67.28 3.14 22.88

w/ Pref-GRPO (10% noise)

67.92 51.03 70.12 3.22 22.90 w/ Pref-GRPO (Ours) 69.46 51.85 70.53 3.26 23.02

of PREF-GRPO and perform joint optimization. Here CLIP serves as a score-based reward signal that focuses more on semantic consistency between the prompt and the generated image. This experiment tests whether pairwise preference fitting can serve as a stabilizing objective while score-based rewards provide complementary semantic guidance.

As shown in Tab. X, joint optimization improves semantic consistency on UNIGENBENCH and GenEval, but slightly reduces image-quality-oriented metrics relative to vanilla PREFGRPO. The qualitative results in Fig. 11 show the same trade-off: semantics improve, while perceptual quality drops mildly. Importantly, unlike pure score maximization, this setting does not exhibit obvious reward hacking. This suggests that pairwise preference fitting acts as a stabilizing regularizer in joint optimization, making it possible to incorporate auxiliary score rewards without collapsing to reward-hacked solutions.

- E. Robustness to Noisy Preferences

To test robustness to imperfect preference supervision, we inject synthetic noise into the pairwise reward during training by independently flipping each pairwise outcome with probability p=0.1, roughly matching typical human-disagreement rates on image preferences.

As shown in Tab. XI, PREF-GRPO remains strong under noisy preferences and consistently outperforms all point scorebased baselines. Although the injected noise causes a small drop relative to the clean PREF-GRPO model, the degradation is limited across all metrics. This result suggests that pairwise preference optimization remains robust under moderate label corruption, which is important in practical settings where preference annotations are often noisy, inconsistent, or partially unreliable.

- F. Human Validation of the UNIGENBENCH Evaluator

To verify the reliability of the automatic judge used in UNIGENBENCH, we conduct a testpoint-level agreement study between Gemini-2.5-Pro [14] and human annotators. We sample 400 held-out prompt-image-testpoint triplets from benchmark

TABLE XII HUMAN VALIDATION OF GEMINI-2.5-PRO AS THE UNIGENBENCH EVALUATOR. WE REPORT INTER-ANN. AGR., GEMINI-VS.-HUMAN-MAJORITY ACC., MACRO-F1, AND COHEN’S κ FOR EACH PRIMARY DIMENSION.

Inter-Ann. Agr.

MacroF1 κ

Acc.

Dimension

Overall 0.92 0.83 0.82 0.66 Style 0.96 0.90 0.89 0.76 World Know. 0.91 0.82 0.81 0.64 Attribute 0.93 0.84 0.83 0.67 Action 0.91 0.82 0.81 0.64 Relationship 0.89 0.80 0.79 0.60 Compound 0.88 0.79 0.78 0.58 Grammar 0.86 0.81 0.77 0.60 Layout 0.92 0.83 0.82 0.66 Logical Reason. 0.85 0.79 0.77 0.60 Text 0.88 0.84 0.83 0.66

predictions generated by models of varying strengths, with 40 triplets for each of the 10 primary dimensions. Each triplet contains a prompt, a generated image, and one explicit testpoint description. Five annotators independently judge whether the image satisfies the specified testpoint using the same binary protocol as in benchmark evaluation, and the human majority vote is taken as the reference label.

We report inter-annotator agreement, Gemini-vs.-humanmajority accuracy, macro-F1, and Cohen’s κ in Tab. XII. These metrics provide complementary views of evaluator reliability: inter-annotator agreement reflects the intrinsic ambiguity of each dimension and serves as a practical human reference ceiling, Gemini-vs.-human-majority accuracy measures direct agreement with human judgment, macro-F1 prevents class imbalance from dominating the result, and κ discounts agreement that may occur by chance. Gemini-2.5-Pro achieves strong overall consistency with human judgments. Agreement is highest on more direct dimensions such as Style, Attribute, and Layout, while more abstract dimensions such as Relationship, Compound, and Logical Reasoning remain relatively more challenging. This tracking behavior supports Gemini-2.5-Pro’s use as the UNIGENBENCH evaluator.

VIII. CONCLUSION

This paper addresses two limitations in current T2I research: unstable GRPO optimization under pointwise reward score maximization, and the lack of fine-grained benchmarks for diagnosing semantic consistency. To this end, we propose PREF-GRPO, which replaces absolute score maximization with pairwise preference fitting, and UNIGENBENCH, which evaluates semantic consistency through fine-grained testpoints spanning diverse prompt themes and evaluation dimensions. Extensive experiments show that PREF-GRPO yields more stable training, alleviates reward hacking, and achieves consistently stronger semantic consistency on both in-domain and out-ofdomain benchmarks. Besides, UNIGENBENCH reveals where strong T2I models still fail, especially on more demanding dimensions such as relationship understanding, compositional instruction following, logical reasoning, and text rendering. We hope these two contributions together provide a more reliable training objective and a more informative evaluation protocol for future T2I reinforcement learning research.

REFERENCES

- [1] J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang, “Flow-grpo: Training flow matching models via online rl,” arXiv preprint arXiv:2505.05470, 2025.
- [2] J. Li, Y. Cui, T. Huang, Y. Ma, C. Fan, M. Yang, and Z. Zhong, “Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde,” arXiv preprint arXiv:2507.21802, 2025.
- [3] Z. Xue, J. Wu, Y. Gao, F. Kong, L. Zhu, M. Chen, Z. Liu, W. Liu, Q. Guo, W. Huang et al., “Dancegrpo: Unleashing grpo on visual generation,” arXiv preprint arXiv:2505.07818, 2025.
- [4] X. He, S. Fu, Y. Zhao, W. Li, J. Yang, D. Yin, F. Rao, and B. Zhang, “Tempflow-grpo: When timing matters for grpo in flow mdels,” arXiv preprint arXiv:2508.04324, 2025.
- [5] D. Ghosh, H. Hajishirzi, and L. Schmidt, “Geneval: An object-focused framework for evaluating text-to-image alignment,” NIPS, vol. 36, pp. 52132–52152, 2023.
- [6] K. Huang, K. Sun, E. Xie, Z. Li, and X. Liu, “T2icompbench: A comprehensive benchmark for open-world compositional text-to-image generation,” NIPS, vol. 36, pp. 78723–78747, 2023.
- [7] X. Wei, J. Zhang, Z. Wang, H. Wei, Z. Guo, and L. Zhang, “Tiif-bench: How does your t2i model follow your instructions?” arXiv preprint arXiv:2506.02161, 2025.
- [8] Y. Wang, Y. Zang, H. Li, C. Jin, and J. Wang, “Unified reward model for multimodal understanding and generation,” arXiv preprint arXiv:2503.05236, 2025.
- [9] X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li, “Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis,” arXiv preprint arXiv:2306.09341, 2023.
- [10] Y. Kirstain, A. Polyak, U. Singer, S. Matiana, J. Penna, and O. Levy, “Pick-a-pic: An open dataset of user preferences for text-to-image generation,” NIPS, vol. 36, pp. 36652–36663, 2023.
- [11] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.
- [12] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021, pp. 8748–8763.
- [13] Y. Wang, Z. Li, Y. Zang, C. Wang, Q. Lu, C. Jin, and J. Wang, “Unified multimodal chain-of-thought reward model through reinforcement fine-tuning,” arXiv preprint arXiv:2505.03318, 2025.
- [14] Y. Huang and L. F. Yang, “Gemini 2.5 pro capable of winning gold at imo 2025,” arXiv preprint arXiv:2507.15855, 2025.
- [15] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv preprint

- arXiv:2410.21276, 2024.
- [16] Y. Gao, L. Gong, Q. Guo, X. Hou, Z. Lai, F. Li, L. Li, X. Lian, C. Liao, L. Liu et al., “Seedream 3.0 technical report,” arXiv preprint arXiv:2504.11346, 2025.
- [17] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic textto-image diffusion models with deep language understanding,” NIPS, vol. 35, pp. 36479–36494, 2022.
- [18] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen et al., “Qwen-image technical report,” arXiv preprint arXiv:2508.02324, 2025.
- [19] Q. Cai, J. Chen, Y. Chen, Y. Li, F. Long, Y. Pan, Z. Qiu, Y. Zhang, F. Gao, P. Xu et al., “Hidream-i1: A highefficient image generative foundation model with sparse diffusion transformer,” arXiv preprint arXiv:2505.22705, 2025.
- [20] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song et al., “Emerging properties in unified multimodal pretraining,” arXiv preprint arXiv:2505.14683, 2025.
- [21] X. Xie and D. Gong, “Dymo: Training-free diffusion model alignment with dynamic multi-objective scheduling,” in CVPR, 2025, pp. 13220–13230.
- [22] K. Yang, J. Tao, J. Lyu, C. Ge, J. Chen, W. Shen, X. Zhu, and X. Li, “Using human feedback to fine-tune diffusion models without any reward model,” in CVPR, 2024, pp. 8941–8951.
- [23] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik, “Diffusion model alignment using direct preference optimization,” in CVPR, 2024, pp. 8228–8238.
- [24] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” NIPS, vol. 36, pp. 53728–53741, 2023.
- [25] C. Tong, Z. Guo, R. Zhang, W. Shan, X. Wei, Z. Xing, H. Li, and P.-A. Heng, “Delving into rl for image generation with cot: A study on dpo vs. grpo,” arXiv preprint arXiv:2505.17017, 2025.
- [26] Y. Niu, M. Ning, M. Zheng, W. Jin, B. Lin, P. Jin, J. Liao, C. Feng, K. Ning, B. Zhu et al., “Wise: A world knowledge-informed semantic evaluation for textto-image generation,” arXiv preprint arXiv:2503.07265, 2025.
- [27] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” arXiv preprint arXiv:2209.03003, 2022.
- [28] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.
- [29] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine, “Training diffusion models with reinforcement learning,” arXiv preprint arXiv:2305.13301, 2023.
- [30] B. F. Labs., “Flux,” https://github.com/black-forestlabs/flux, 2024.
- [31] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient

- memory management for large language model serving with pagedattention,” in SOSP, 2023, pp. 611–626.
- [32] J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong, “Imagereward: Learning and evaluating human preferences for text-to-image generation,” NIPS, vol. 36, pp. 15903–15935, 2023.
- [33] C. Schuhmann., “Laion aesthetics,” https://github.com/LAION-AI/aesthetic-predictor, 2022.
- [34] J. Xie, Z. Yang, and M. Z. Shou, “Show-o2: Improved native unified multimodal models,” arXiv preprint arXiv:2506.15564, 2025.
- [35] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” 2021.
- [36] M. Ding, Z. Yang, W. Hong, W. Zheng, C. Zhou, D. Yin, J. Lin, X. Zou, Z. Shao, H. Yang, and J. Tang, “Cogview: Mastering text-to-image generation via transformers,” arXiv preprint arXiv:2105.13290, 2021.
- [37] Z. Li, J. Zhang, Q. Lin, J. Xiong, Y. Long et al., “Hunyuandit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding,” 2024.
- [38] D. Li, A. Kamko, E. Akhgari, A. Sabet, L. Xu, and S. Doshi, “Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation,” arXiv preprint arXiv:2402.17245, 2024.
- [39] C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan et al., “Janus: Decoupling visual encoding for unified multimodal understanding and generation,” in CVPR, 2025, pp. 12966–12977.
- [40] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan, “Janus-pro: Unified multimodal understanding and generation with data and model scaling,” arXiv preprint arXiv:2501.17811, 2025.
- [41] Y. Ma, X. Liu, X. Chen, W. Liu, C. Wu, Z. Wu, Z. Pan, Z. Xie, H. Zhang, X. yu, L. Zhao, Y. Wang, J. Liu, and C. Ruan, “Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation,” 2024.
- [42] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu et al., “Emu3: Next-token prediction is all you need,” arXiv preprint arXiv:2409.18869, 2024.
- [43] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese et al., “Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset,” arXiv preprint arXiv:2505.09568, 2025.

