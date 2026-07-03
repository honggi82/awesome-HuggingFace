# arXiv:2605.21467v1[cs.LG]20May2026

## DELTA: DISCRIMINATIVE TOKEN CREDIT ASSIGNMENT FOR REINFORCEMENT LEARNING FROM VERIFIABLE REWARDS

Kaiyi Zhang1,2 , Wei Wu2†, Yankai Lin1† 1Gaoling School of Artificial Intelligence, Renmin University of China 2Ant International

{kyzhang, yankailin}@ruc.edu.cn wuwei19850318@gmail.com

Code: https://github.com/RUCBM/DelTA

ABSTRACT

Reinforcement learning from verifiable rewards (RLVR) has emerged as a central technique for improving the reasoning capabilities of large language models. Despite its effectiveness, how response-level rewards translate into token-level probability changes remains poorly understood. We introduce a discriminator view of RLVR updates, showing that the policy-gradient update direction implicitly acts as a linear discriminator over token-gradient vectors and thereby determines which token probabilities are increased or decreased during learning. Under standard sequence-level RLVR, this discriminator is constructed from positive- and negativeside centroids formed by advantage-weighted averaging of token-gradient vectors. However, such centroid construction can be dominated by shared high-frequency patterns, such as formatting tokens, diluting sparse yet discriminative directions that better distinguish high-reward responses from low-reward ones. To address this limitation, we propose DelTA, a discriminative token credit assignment method that estimates token coefficients to amplify side-specific token-gradient directions and downweight shared or weakly discriminative ones. These coefficients reweight a self-normalized RLVR surrogate, making the effective side-wise centroids more contrastive and thereby reshaping the RLVR update direction. On seven mathematical benchmarks, DelTA outperforms the strongest same-scale baselines by 3.26 and 2.62 average points on Qwen3-8B-Base and Qwen3-14B-Base, respectively. Additional results on code generation, a different backbone, and out-of-domain evaluations further demonstrate the generalization ability of DelTA.

1 INTRODUCTION

Reinforcement learning from verifiable rewards (RLVR) has become a key paradigm for improving the reasoning ability of large language models (LLMs), with strong gains in mathematics (Shao et al., 2024; Yang et al., 2024), code generation (Hui et al., 2024; Shojaee et al., 2023; Le et al., 2022), and formal problem solving (Guo et al., 2025; Team et al., 2025). RLVR optimizes response-level verifiable rewards, such as answer correctness, without requiring dense process-level annotations. This response-level supervision creates a granularity mismatch: each response provides a single scalar advantage, while the policy update is accumulated through token-level terms. Recent studies show that RLVR induces sparse token-level distributional shifts, where substantial probability changes concentrate on a small subset of tokens while most token distributions change little (Meng et al., 2026; Ma et al., 2026). This contrast suggests that sequence-level RLVR contains an implicit token-level selection mechanism that is not directly specified by the reward signal. Hence, an essential question arises: which token probabilities are increased or decreased by an RLVR update, and what determines these changes?

†Corresponding author: Wei Wu and Yankai Lin.

We introduce a discriminator view of RLVR to explain this implicit token selection. Although an RLVR update is usually viewed as a parameter-space movement, the same update also defines a tokenlevel decision rule: it determines whether a candidate-token probability is increased or decreased by the update. The rule works by comparing token-gradient directions. For a sequence-level RLVR objective, the update direction contrasts token-gradient aggregates from positive-advantage responses and negative-advantage responses. After normalization, these aggregates define positive- and negativeside reference directions. A candidate-token probability is increased when its token-gradient vector aligns more with the positive-side reference direction than with the negative-side reference direction, and is decreased otherwise. In this sense, the RLVR update acts as an implicit linear discriminator over candidate token-gradient vectors. This view suggests that RLVR update directions can be understood and improved by analyzing and shaping the discriminator induced by the update.

Following the insights, further investigation indicates that standard sequence-level RLVR updates form the two-side directions by averaging token-gradient vectors from positive- and negative-advantage responses, yielding two centroids. Such centroids are natural summaries of each side, but a good within-side summary is not necessarily a good between-side discriminator. In reasoning tasks, higher- and lower-reward responses often share substantial common structure, such as formatting tokens or problem-specific entities. Because these shared patterns appear on both sides and occur frequently, their token-gradient directions can pull both centroids toward common background structure. Consequently, the induced discriminator may overemphasize task-agnostic commonalities and undermine sparse directions that better distinguish higher- from lower-reward responses.

To address this limitation, we propose Discriminative signal-guided Token Credit Assignment (DelTA). DelTA reshapes the induced RLVR discriminator by reweighting token-gradient terms in the RLVR surrogate. It estimates token coefficients that assign larger weights to token-gradient vectors more characteristic of their own advantage side than of the opposite side, while assigning smaller weights to shared or weakly discriminative directions. These coefficients change the effective aggregates that define the discriminator, making its positive and negative reference directions more contrastive and thereby reshaping the RLVR update direction. Empirically, DelTA consistently improves strong RLVR baselines. On seven mathematical benchmarks, it surpasses the strongest same-scale baseline by 3.26 average points on Qwen3-8B-Base and 2.62 points on Qwen3-14B-Base. It also improves code generation and generalizes to another backbone and out-of-domain evaluations.

In summary, our contributions are threefold. First, we introduce a local discriminator view of sequence-level RLVR, showing that policy-gradient updates induce an implicit linear discriminator over token-gradient vectors and thereby determine local token-probability changes. Second, using this view, we trace a limitation of standard sequence-level RLVR to the construction of the update direction: the side-wise centroids that form the induced discriminator can be pulled toward shared, high-frequency token-gradient directions, weakening its ability to separate token-gradient directions from higher- and lower-reward responses. Third, we propose DelTA, which reweights token-gradient terms by their positive-negative discriminative signal in a self-normalized RLVR surrogate, making the effective side-wise centroids more contrastive and consistently improving strong RLVR baselines across mathematical reasoning, code generation, different backbones, and out-of-domain evaluations.

- 2 PRELIMINARIES

We review the critic-free group-relative RLVR framework, taking DAPO as the main concrete example. For a prompt q, let {oi}Gi=1 be a group of sampled responses, where oi = (oi,1,...,oi,|o

i|). Each response receives a sequence-level reward Ri. Let µR and σR be the mean and standard deviation of rewards within the sampled group, and let ϵA > 0 be a small numerical constant. The group-normalized advantage and token-level importance ratio are given by

Ri − µR σR + ϵA

πθ(oi,t | q,oi,<t) πθ

Aˆi =

, ri,t(θ) =

.

(oi,t | q,oi,<t)

old

DAPO-style Surrogate. DAPO (Yu et al., 2025) is a state-of-the-art critic-free group-relative RLVR method that optimizes a clipped surrogate objective with two key designs relevant to this work: asymmetric clipping and token-level normalization over all response tokens. The expected objective is defined as

4 Weighted centroids

###### 3 Assign DelTA weights

###### 2 Initialize side-wise centroids

###### 1 Token Gradient

- (a) Intuition
- (b) Distribution

... ... ...

near 𝝁 ( ) : high

A > 0 A > 0 A < 0

Let’s think step by

𝝁 ( )

Refine Centroids

𝝁 ( )

Now I to

need

far from 𝝁 ( ) : low

𝝁

𝜇( )=𝜇̅

Sorry I don’t know

𝝁

𝝁 (   ) 𝝁 (   )

Update Weights

Gradient

𝝁 ( )

𝝁 ( )

focus on discriminative tokens

high low high

shared features

- Figure 1: Overview of DelTA. DelTA estimates token coefficients from the contrast between positiveand negative-advantage token-gradient aggregates, and uses the coefficients to reweight the sequencelevel RLVR objective.

  1

 .

|oi|

G

min ri,t(θ)Aˆi, clip ri,t(θ),1 − ϵlow,1 + ϵhigh A ˆi

JDAPO(θ) = E

G i=1 |oi|

t=1

i=1

Here Aˆi is defined at the response level and is therefore shared by all tokens in the same response. The per-token contribution to the surrogate is nevertheless accumulated through the token-level ratio

ri,t(θ), which provides the basic object for the token-gradient analysis in the next section.

The original DAPO algorithm also includes dynamic sampling to encourage each sampled group to contain both correct and incorrect responses. This component affects rollout filtering rather than the form of the per-token surrogate above. We disable dynamic sampling for all methods in our experiments, and focus our analysis on the update rule induced by the surrogate objective.

- 3 METHOD

Recent studies suggest that RLVR induces sparse and targeted changes at the token-distribution level: only a small fraction of token distributions undergo substantial shifts, while most remain nearly unchanged (Meng et al., 2026; Ma et al., 2026). For sequence-level RLVR, this sparsity is not directly explained by the reward signal itself, since all tokens in the same response share the same scalar advantage. This suggests that the token-level selection effect is induced not by explicit token rewards, but by how token-gradient vectors are aggregated in the policy-gradient update.

We therefore analyze this induced selection effect by examining how sequence-level RLVR updates determine which candidate-token probabilities are increased or suppressed. Section 3.1 investigates this question using DAPO as a concrete instance and derives the discriminator view induced by RLVR updates. Although DAPO serves as the primary showcase, the resulting conclusions rely only on the update being expressible as an advantage-weighted aggregation of token-gradient vectors, and therefore naturally extend to a broader class of sequence-level RLVR objectives, as discussed in Appendix E. Building on this analysis, Section 3.2 introduces DelTA, a new critic-free group-relative RLVR method that reweights token-gradient terms to reshape the induced discriminator and the corresponding RLVR update direction.

- 3.1 A LOCAL DISCRIMINATOR VIEW OF RLVR UPDATES

To understand how sequence-level RLVR implicitly selects tokens, we view the local policy update not only as a parameter update, but also as an implicit discriminator in token-gradient space. For sequencelevel RLVR objectives, the update direction contrasts token-gradient aggregates from positive- and negative-advantage responses. After normalization, these aggregates define two side-wise reference directions. A candidate token log-probability is locally increased when its token-gradient vector aligns more with the positive reference direction than with the negative one, and is decreased otherwise.

Formally, let c denote an arbitrary generation context, x a candidate next token under this context, and πθ(x | c) the policy model. For a local parameter update ∆θ around θold, a first-order Taylor

expansion gives

∆log π(x | c) := log πθ

old+∆θ(x | c) − log πθ

old

(x | c) ≈ ∇θ log πθ(x | c)

θ=θold

⊤

∆θ. (1)

Thus, once x, c, and θold are fixed, the local increase or decrease of the candidate token probability is determined by the inner product between its token-gradient vector and the update direction ∆θ. In the following analysis, we focus on the local update direction (i.e., ∆θ).

For concreteness, consider the DAPO-style sequence-level surrogate used in our analysis. Let {oi}Gi=1 be a rollout group, and let Aˆi be the group-normalized advantage of response oi. Around θold, clipping is locally inactive because ri,t(θold) = 1. The local policy-gradient update can therefore be written as an advantage-weighted aggregation of sampled-token gradients. Separating this aggregation by the sign of the response-level advantage gives

|oi|

|oi|

Aˆi vi,t −

|Aˆi|vi,t, vi,t := ∇θ log πθ(oi,t | q,oi,<t)

∆θRLVR ∝

##### .

θ=θold

t=1

t=1

i:Aˆi>0

i:Aˆi<0

(2) We refer to token-gradient vectors from responses with Aˆi > 0 as the positive side, and those from responses with Aˆi < 0 as the negative side. Throughout this analysis, vi,t denotes the exact full-parameter token-gradient vector. A detailed derivation of Eq. (2) from the DAPO surrogate is provided in Appendix C. We use this local characterization as an analysis and design principle for shaping the policy-update direction, rather than as an exact description of the full nonlinear clipped RLVR training trajectory.

Eq. (2) contains two components: the total mass of each side and the reference direction of each side. Let M+ = i:Aˆ

|oi| t=1

|oi| t=1 |Aˆi| denote the total positive and negative

Aˆi and M− = i:Aˆ

i>0

i<0

advantage masses. Then Eq. (2) can be rewritten as

|oi| t=1

|oi| t=1 |Aˆi|vi,t

Aˆi vi,t M+

∆θRLVR ∝ M+µ¯+−M−µ¯−, µ¯+ = i:Aˆ

, µ¯− = i:Aˆ

i>0

i<0

.

M−

(3)

Here, M+ and M− determine the total strength of the two advantage sides, while µ¯+ and µ¯− are their normalized aggregate directions. Substituting Eq. (3) into Eq. (1) yields

∆log π(x | c) ∝ M+ ∇θ log πθ(x | c)

θ=θold

⊤

µ¯+−M− ∇θ log πθ(x | c)

θ=θold

⊤

µ¯−. (4)

The two terms in Eq. (4) define the positive-side and negative-side scores assigned to the candidate token-gradient vector. The candidate token probability is locally increased when its positive-side score exceeds its negative-side score, and decreased otherwise. In this sense, the update direction has a dual role: in parameter space, it is a policy-update direction; in token-gradient space, it acts as an implicit linear discriminator over candidate token-gradient vectors. This discriminator is not explicitly parameterized or separately trained; it is induced by the policy-gradient update itself. This duality suggests a reverse design perspective: since the update direction induces a discriminator in token-gradient space, we can instead ask how to shape this induced discriminator and adjust the update direction accordingly. Thus, RLVR update directions can be understood and improved by studying and shaping the local discriminator induced by the update.

For the induced discriminator, the central objects are the side-wise reference directions µ¯+ and µ¯−. Under the standard sequence-level RLVR update, these directions are simply the advantageweighted centroids of the token-gradient vectors on the positive and negative sides. Equivalently, they are weighted least-squares summaries that minimize within-side squared distances, as shown in Appendix D. Such centroids are natural if the goal is to summarize each side independently. However, the induced discriminator uses them for a different purpose: distinguishing positive-advantage token gradients from negative-advantage token gradients.

This creates a mismatch between within-side summarization and between-side discrimination. In RLVR, positive- and negative-advantage responses often share frequent token patterns, such as common formatting tokens or problem-specific entities. The corresponding token-gradient directions

can dominate both side-wise centroids, making the positive and negative reference directions less discriminative and diluting rarer directions that better separate higher-reward responses from lowerreward responses. From a classical discriminative perspective, good within-side summaries are not necessarily good between-side discriminators (Cohen et al., 2013; Zhao et al., 2024; Khosla et al., 2020).

This motivates a centroid-level design principle: to obtain a better local update direction, we can reshape the side-wise centroids that define the induced discriminator. Changing these centroids changes the scores assigned to candidate token-gradient vectors, and therefore changes which token probabilities are locally increased or decreased. This principle motivates DelTA: we reshape the effective side-wise centroids by assigning larger weights to token-gradient directions that better distinguish the two advantage sides.

- 3.2 DELTA: DISCRIMINATIVE SIGNAL-GUIDED TOKEN CREDIT ASSIGNMENT

DelTA implements the centroid-level principle above by reweighting token terms in the RLVR surrogate. Since the side-wise centroids are induced by weighted token-gradient aggregation rather than separately parameterized, changing token weights directly reshapes these centroids, and hence the induced discriminator and local update direction. At a high level, DelTA has three steps. First, it initializes the positive and negative reference directions from the original advantage-weighted centroids. Second, it refines these directions through a small number of alternating steps: with the current centroids fixed, DelTA estimates discriminative token scores; with the scores fixed, it recomputes each side-wise centroid as a score-weighted average of token-gradient vectors from that side. Third, it maps the final scores to bounded coefficients and uses them to reweight the sequence-level RLVR surrogate.

The formulation below is written in terms of token-gradient vectors vi,t. In the exact version, these vectors are the full-parameter gradients defined in Section 3.1. In practice, explicitly materializing full-parameter gradients for all rollout tokens is computationally prohibitive at LLM scale, so we use a layer-restricted LM-head gradient representation to compute the stop-gradient token coefficients. This proxy affects only the coefficient computation; the analysis remains formulated with full-parameter token gradients, and the resulting weighted RLVR objective is still optimized over the full policy parameters. Further details and proxy ablations are provided in Appendix F.

We initialize the refinement from the original advantage-weighted centroids, µ(0)+ = µ¯+ and µ(0)− = µ¯−. DelTA then runs K stop-gradient refinement iterations. At iteration k = 0,...,K − 1, DelTA

first estimates a soft discriminative score αi,t(k) for each token-gradient vector. We describe the positive side; the negative side is obtained symmetrically by swapping µ(+k) and µ(−k), and by replacing γ+(k) with γ−(k). For a positive-advantage token, DelTA assigns a larger score when vi,t is closer to the positive centroid than to the negative centroid. Specifically, αi,t(k) is defined by the entropy-regularized assignment problem

α ∥vi,t − µ(−k)∥22 − ∥vi,t − µ(+k)∥22 + γ+(k)h(α), Aˆi > 0, (5)

αi,t(k) = arg max

α∈[0,1]

where h(α) = −α log α − (1 − α)log(1 − α) is the binary entropy regularizer, and γ+(k) > 0 is a side-specific temperature for the positive-side assignment. The distance-margin term is positive

when vi,t is closer to the positive centroid than to the negative centroid, so maximizing Eq. (5) assigns a larger score to tokens that are more characteristic of their own side. The entropy regularizer and the temperature jointly control the softness of this assignment: smaller temperatures make the score closer to a hard decision, while larger temperatures produce smoother scores. We use squared Euclidean distances to stay consistent with the centroid construction, as detailed in Appendix D.

For fixed centroids and temperatures, the closed-form solution is

 

σ ∥vi,t − µ(−k)∥22 − ∥vi,t − µ(+k)∥22 γ+(k)

, Aˆi > 0,

αi,t(k) =

(6)

σ ∥vi,t − µ(+k)∥22 − ∥vi,t − µ(−k)∥22 γ−(k)

, Aˆi < 0,



where σ(·) is the sigmoid function. The side-specific temperatures γ+(k) and γ−(k) adapt the assignment scale for the two advantage sides; their computation is detailed in Appendix H. A derivation of Eq. (6) is provided in Appendix G.

Thus, αi,t(k) is large when the token-gradient vector is more representative of its own advantage side than of the opposite side, and small for shared or weakly discriminative directions.

Given these scores, DelTA updates the centroids as score-weighted within-side averages:

|oi| t=1

|oi| t=1 |Aˆi|αi,t(k) vi,t

Aˆi αi,t(k) vi,t

µ(+k+1) = i:Aˆ

, µ(−k+1) = i:Aˆ

i>0

i<0

. (7)

|oi| t=1

|oi| t=1 |Aˆi|αi,t(k)

Aˆi αi,t(k)

i:Aˆi>0

i:Aˆi<0

This refinement gives larger influence to token-gradient vectors that are more characteristic of their own side, while downweighting shared or weakly discriminative directions. It is used only to compute stop-gradient token coefficients; no gradients are propagated through the refinement, and no additional loss term is added.

After the final refinement step, DelTA recomputes raw scores αi,t⋆ with the refined centroids and maps them to bounded coefficients λi,t = λmin + (λmax − λmin)αi,t⋆ . The bounded range prevents extreme reweighting while preserving the ranking of the discriminative scores. DelTA then replaces the uniform token average in DAPO with the following self-normalized weighted surrogate:

  1

 .

|oi|

G

λi,t min ri,t(θ)Aˆi, clip ri,t(θ),1 − ϵlow,1 + ϵhigh A ˆi

JDelTA(θ) = E

|oi| t=1 λi,t

G i=1

t=1

i=1

(8)

Around θold, Eq. (8) changes each sampled-token contribution from Aˆivi,t to λi,tAˆivi,t. This reweighting reshapes the effective side-wise centroids, and hence the induced discriminator and local RLVR update direction, by amplifying side-specific token-gradient directions and downweighting shared or weakly discriminative ones. The coefficients are stop-gradient quantities computed once per rollout batch, fixed across optimization epochs, and recomputed after new trajectories are sampled. Full details are provided in Appendix H.

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETUP

We train on two backbones, Qwen3-8B-Base (Yang et al., 2025) and Qwen3-14B-Base, using DeepMath-103K (He et al., 2025) with VeRL (Sheng et al., 2024). For DelTA, we set [λmin,λmax] = [0.8,1.2] and use one refinement iteration (K = 1). We compare against DAPO (Yu et al., 2025), DAPO w/ Forking Tokens (DAPO w/ FT) (Wang et al., 2025), SAPO (Gao et al., 2025), and FIPO (Ma et al., 2026), training all methods with the same hyperparameters. We disable dynamic sampling for all methods to isolate the effect of the policy-update objective. Detailed training settings and baseline descriptions are provided in Appendix I and Appendix J, respectively.

We evaluate our models on seven mathematical benchmarks: AIME24 (Zhang & Math-AI, 2024), AIME25 (Zhang & Math-AI, 2025), AIME26 (Zhang & Math-AI, 2026), HMMT25 (February) (Balunovi´c et al., 2025), HMMT25 (November), HMMT26 (February) and Brumo 25. To better reveal each model’s long-reasoning capability, we set the maximum generation length during evaluation to 30,000 tokens. We sample 16 responses for each problem. We report the average performance over all samples. Unless otherwise specified, Avg. denotes the question-count weighted average across benchmarks. Detailed hyperparameters are provided in Appendix I.

- 4.2 MAIN RESULTS

Table 1 reports the main results on seven mathematical reasoning benchmarks. DelTA consistently outperforms all same-scale RL baselines on both Qwen3-8B-Base and Qwen3-14B-Base, achieving the best result on every benchmark and the highest average score at both scales. Compared with the strongest same-scale baseline, DelTA improves the average score from 25.14 to 28.40 on the

Table 1: Main results on seven mathematical reasoning benchmarks for Qwen3-8B-Base and Qwen314B-Base. DelTA consistently outperforms all compared same-scale RL baselines. The best results are in bold, and the second-best results are underlined.

HMMT25 (Feb.)

HMMT25 (Nov.)

HMMT26 (Feb.)

Method AIME24 AIME25 AIME26

Brumo25 Avg. Qwen3-8B-Base

DAPO 34.79 23.33 24.17 13.54 12.08 16.86 36.46 22.95 DAPO w/ FT 36.67 23.96 26.46 15.62 15.42 17.05 39.17 24.80 SAPO 38.75 24.37 26.25 14.58 16.04 17.42 39.37 25.14 FIPO 37.50 23.13 23.96 14.58 12.92 17.99 37.71 23.89 DelTA 43.13 26.46 28.12 18.33 18.54 20.27 44.79 28.40

###### Qwen3-14B-Base

DAPO 51.25 32.29 39.79 19.79 30.00 25.38 48.13 35.09 DAPO w/ FT 54.37 33.75 41.46 20.42 31.67 24.81 52.08 36.77 SAPO 53.96 34.17 41.46 20.62 28.33 24.05 50.21 35.94 FIPO 54.58 35.00 42.50 21.46 32.29 24.43 52.08 37.29 DelTA 56.87 37.92 45.21 26.04 32.92 26.89 54.79 39.91

8B backbone and from 37.29 to 39.91 on the 14B backbone. As detailed in Appendix K, under repeated-generation evaluation, DelTA significantly outperforms the strongest same-scale baselines.

The consistent gains across benchmarks and model scales suggest that DelTA improves the policyupdate mechanism beyond a single benchmark instance. The computational overhead of DelTA is discussed in Appendix L.1. We further evaluate DelTA beyond the main math setting: Appendix L.3 shows that DelTA also improves DAPO on code generation benchmarks, and Appendix L.2 shows that DelTA remains effective on Olmo3-7B-Base (Olmo et al., 2025). These results indicate that the benefit of DelTA is not limited to a specific benchmark family or backbone architecture.

- 4.3 TRAINING DYNAMICS

- Figure 2 compares the training dynamics of DelTA and DAPO on Qwen3-8B-Base. The two methods show similar reward trajectories in the early stage, but diverge afterwards: DAPO plateaus and slightly degrades, whereas DelTA continues to improve and reaches a higher final reward. The response-length and entropy curves suggest that this divergence is not merely due to shorter answers. DAPO shifts toward shorter responses with rising entropy, whereas DelTA maintains longer responses with lower entropy and higher reward, indicating more stable and confident long-reasoning behavior.

This behavior is consistent with our discriminator view. In standard sequence-level aggregation, shared background directions can dominate the side-wise centroids and weaken the contrast of the induced update direction. DelTA counteracts this by upweighting side-specific token-gradient directions and downweighting shared or weakly discriminative ones. As a result, the effective reference directions become more contrastive, helping the update sustain useful long-reasoning trajectories without an explicit length incentive.

- 5 ANALYSIS

In this section, we provide further analysis of DelTA beyond the main results. Unless stated otherwise, we analyze DelTA on Qwen3-8B-Base using four representative math benchmarks: AIME25, AIME26, HMMT25-Nov, and HMMT26-Feb, abbreviated as HMMT25 and HMMT26.

We organize the analysis around five diagnostic questions: Q1: Is the opposite-side comparison necessary? Section 5.1 tests whether own-side centrality alone can explain DelTA’s gains. Q2: Does λi,t capture useful token-level learning signals? Section 5.2 examines this by using λi,t only for token selection. Q3: Are the design components of DelTA necessary? Section 5.3 answers this through component ablations. Q4: Is DelTA sensitive to its hyperparameters? Appendix L.4 studies

| |DAPO DelTA<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.8

0.7

###### Reward

0.6

0.5

0.4

0 50 100 150 200

Step

6000

| |DAPO DelTA<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

5000

###### ResponseLength

4000

3000

2000

1000

0 50 100 150 200

Step

| |DAPO DelTA<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

1.0

0.8

Entropy

0.6

0.4

0.2

0 50 100 150 200

Step

- Figure 2: Training dynamics of DelTA compared with DAPO. Left: Reward; Middle: Response Length; Right: Entropy.

the robustness of DelTA under different hyperparameter choices. Q5: Does DelTA generalize to out-of-domain evaluation? Appendix L.5 evaluates DelTA on additional OOD benchmarks.

- 5.1 Q1: IS THE OPPOSITE-SIDE COMPARISON NECESSARY?

To test whether DelTA’s gain can be explained by own-side centrality alone, we construct a withinside-only variant. This variant keeps the same coefficient normalization and weighted DAPO objective as DelTA, but removes the opposite-side distance from the assignment score. For to-

kens from positive-advantage responses, we use αi,t = σ −∥vi,t−µ+∥

2 2

γ+ , and define the negative side symmetrically with µ− and γ−. Thus, this variant assigns larger coefficients to tokens closer to their own-side centroid, without considering their distance to the opposite-side centroid.

Table 2: Effect of using only within-side information. The best results are in bold.

Method AIME25 AIME26 HMMT25 HMMT26 Avg. DelTA 26.46 28.12 18.54 20.27 23.27 DAPO 23.33 24.17 12.08 16.86 19.05 Within-side only 21.67 22.08 11.04 17.05 17.94

- Table 2 shows that the within-

side-only variant performs worse than both DelTA and the DAPO baseline. This result indicates that DelTA’s gains cannot be explained by simply assigning larger weights to tokens close to their own-side centroid. In fact, own-side centrality alone can be misleading: tokens near a sidewise centroid may correspond to shared patterns rather than directions that distinguish positive- from negative-advantage responses. The opposite-side comparison is therefore essential, because it assigns high coefficients only to directions that are relatively more representative of their own side than of the opposite side.

- 5.2 Q2: DOES λi,t CAPTURE USEFUL TOKEN-LEVEL LEARNING SIGNALS?

We next test whether the DelTA coefficient λi,t identifies token-gradient directions with useful learning value. To isolate the ranking effect from the continuous reweighting objective, we use λi,t only for hard token selection. Specifically, we compute DelTA coefficients for all valid response tokens and train DAPO using only the top 50% tokens ranked by λi,t. As controls, we train on a random 50% subset and on the bottom 50% tokens, while keeping the standard DAPO loss unchanged.

- Figure 3 shows a sharp separation between these selections. Training on the top-λ tokens consistently outperforms both full-token DAPO and random 50% selection, even though it uses only half of the token-gradient terms. In contrast, training on the bottom-λ tokens quickly collapses. The evaluation results in Figure 4 show the same trend: top-λ training improves accuracy across benchmarks, whereas random selection remains close to DAPO. These results indicate that DelTA’s coefficients capture more than a benign sparsification signal. If the gain came merely from reducing the number of optimized tokens, random 50% selection should provide a similar benefit, which it does not. Instead, the fact that the top half outperforms full-token DAPO while the bottom half collapses suggests that low-λ tokens are not simply uninformative; their gradient directions can actively harm the RLVR

30

| |DAPO<br><br>Random 50%<br><br>Top 50%<br><br>Bottom 50%| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |DAPO<br><br>Random 50%<br><br>Top 50%| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.8

0.7

25

### Reward

#### Accuracy

0.6

20

0.5

0.4

15

0.3

0 50 100 150 200

10

Step

AIME25 AIME26 HMMT25 HMMT26 Avg.

- Figure 3: Training reward under different tokenselection strategies.

Figure 4: Evaluation accuracy under different token-selection strategies.

update. Thus, λi,t separates token-gradient directions with high effective learning value from shared or misleading directions that weaken policy improvement.

- 5.3 Q3: ARE THE DESIGN COMPONENTS OF DELTA NECESSARY?

We ablate five design choices in DelTA while keeping all other training settings unchanged: w/o adaptive γ fixes the assignment temperatures to the initial distance scale; w/o h(α) removes the entropy regularizer and turns the soft assignment into a hard 0/1 decision; w/o λ-norm keeps the token coefficients in the numerator, but replaces DelTA’s coefficient-mass normalizer 1/ i,t λi,t with the standard DAPO token-count normalizer 1/ i |oi|; w/o range map removes the linear mapping from assignment scores to [λmin,λmax], and directly uses the raw scores αi,t ∈ (0,1) as token weights; and w/o refinement estimates coefficients only from the initial side-wise centroids.

- Table 3 shows that each component contributes to DelTA. Removing any design choice reduces the average score. The largest drop comes from w/o refinement, indicating that oneshot coefficients from the initial centroids are insufficient. The drops from w/o range map and w/o h(α) suggest that bounded soft coefficients are more stable than raw scores or hard assignments. Finally, the degradation from w/o adaptive γ and w/o λnorm shows that scale adaptation and coefficient-mass normalization are both useful.

Table 3: Ablation study of DelTA. The best results are in bold, and the second-best results are underlined.

Method AIME25 AIME26 HMMT25 HMMT26 Avg. Full DelTA 26.46 28.12 18.54 20.27 23.27 w/o adaptive γ 25.00 26.04 16.04 17.99 21.19 w/o h(α) 24.37 26.87 15.42 17.42 20.93 w/o λ-norm 24.37 26.25 15.83 19.32 21.39 w/o range map 24.79 25.83 15.83 17.05 20.78 w/o refinement 23.13 25.42 15.42 16.29 19.97

- 6 RELATED WORK

Reinforcement learning has become an effective paradigm for improving LLM reasoning, especially in domains with verifiable feedback such as mathematics. Representative methods include PPO-style optimization and recent critic-free group-relative objectives such as GRPO and DAPO (Schulman et al., 2017; Shao et al., 2024; Yu et al., 2025). Recent work further studies the mechanisms of RLVR (Yue et al., 2025; Huan et al., 2025; Meng et al., 2026), improves training stability and efficiency (Zheng et al., 2025; Gao et al., 2025; Liu et al., 2025), and explores off-policy or semi-offpolicy training (Yan et al., 2025; Zhang et al., 2025a).

A central challenge in RLVR is that rewards are usually provided at the response level, while policy updates are applied at the token level. Prior work addresses this mismatch through token-level or

step-level reweighting (Kazemnejad et al., 2025; Xie et al., 2025), process reward models or learned value functions (Cui et al., 2025; Zhang et al., 2025b), and token-selection signals such as entropy or future influence (Wang et al., 2025; Ma et al., 2026).

Our work is complementary. Instead of relying on external dense rewards, value estimates, or auxiliary token-selection rules, DelTA reweights the RLVR surrogate using token coefficients derived from the positive-negative discriminator induced by the update itself.

- 7 CONCLUSION

We studied token-level learning in RLVR from a local discriminative perspective. Using DAPO as a representative sequence-level objective, we showed that its policy-gradient update induces an implicit linear discriminator over candidate token-gradient vectors. This discriminator is defined by positive and negative side-wise centroids, which can be dominated by shared, weakly discriminative directions rather than directions that distinguish higher- from lower-reward responses. Motivated by the mismatch between within-side summarization and between-side discrimination, we proposed DelTA, which estimates token coefficients from refined centroid contrasts and uses them to reweight a self-normalized clipped RLVR surrogate. Experiments on mathematical reasoning, code generation, another model family, and out-of-domain benchmarks show consistent improvements over strong RLVR baselines. Overall, our results suggest that shaping the discriminator induced by RLVR updates is a useful route for improving token-level credit assignment under sequence-level supervision.

REFERENCES

Mislav Balunovi´c, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´c, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, February 2025. URL https: //matharena.ai/.

Jacob Cohen, Patricia Cohen, Stephen G West, and Leona S Aiken. Applied multiple regression/correlation analysis for the behavioral sciences. Routledge, 2013.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, and Junyang Lin. Soft adaptive policy optimization. arXiv preprint arXiv:2511.20347, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.

Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, and Xiang Yue. Does math reasoning improve general llm capabilities? understanding transferability of llm reasoning. arXiv preprint arXiv:2507.00432, 2025.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Refining credit assignment in rl training of llms,

2025. URL https://arxiv.org/abs/2410.01679.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673, 2020.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314–21328, 2022.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in neural information processing systems, 36:21558–21572, 2023.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Chiyu Ma, Shuo Yang, Kexin Huang, Jinda Lu, Haoming Meng, Shangshang Wang, Bolin Ding, Soroush Vosoughi, Guoyin Wang, and Jingren Zhou. Fipo: Eliciting deep reasoning with future-kl influenced policy optimization. arXiv preprint arXiv:2603.19835, 2026.

Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, Bolin Ding, and Jingren Zhou. Sparse but critical: A token-level analysis of distributional shifts in rlvr fine-tuning of llms. arXiv preprint arXiv:2603.22446, 2026.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, et al. Olmo 3. arXiv preprint arXiv:2512.13961, 2025.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems, 33: 19920–19930, 2020.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Parshin Shojaee, Aneesh Jain, Sindhu Tipirneni, and Chandan K Reddy. Execution-based code generation using deep reinforcement learning. arXiv preprint arXiv:2301.13816, 2023.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025.

Guofu Xie, Yunsheng Shi, Hongtao Tian, Ting Yao, and Xiao Zhang. Capo: Towards enhancing llm reasoning through generative credit assignment. arXiv preprint arXiv:2508.02298, 2025.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Chih-Kuan Yeh, Ankur Taly, Mukund Sundararajan, Frederick Liu, and Pradeep Ravikumar. First is better than last for language data influence. Advances in Neural Information Processing Systems, 35:32285–32298, 2022.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, and Rui Yan. Stephint: Multi-level stepwise hints enhance reinforcement learning to reason. arXiv preprint arXiv:2507.02841, 2025a.

- Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2024, 2024.
- Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025.
- Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2026, 2026.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 10495– 10516, 2025b.

Shuping Zhao, Bob Zhang, Jian Yang, Jianhang Zhou, and Yong Xu. Linear discriminant analysis. Nature Reviews Methods Primers, 4(1):70, 2024.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

- A LIMITATIONS

DelTA provides a lightweight way to incorporate discriminative token-level credit assignment into sequence-level RLVR. While our experiments show consistent gains across several settings, there remain several directions for further improvement and broader validation.

First, DelTA estimates token coefficients using a layer-restricted token-gradient proxy rather than fullparameter token gradients, since computing full gradients for all sampled tokens is computationally expensive at RLVR scale. This proxy is used only for stop-gradient coefficient estimation, and the weighted RLVR objective still updates the full policy parameters. Our proxy ablations show that DelTA is robust to different layer-restricted proxy choices, but exploring richer and more efficient token-gradient approximations is a promising direction for future work.

Second, our empirical evaluation focuses primarily on mathematical reasoning, with additional validation on code generation, different backbone architectures, and out-of-domain benchmarks. Future work could further evaluate DelTA on broader RLVR settings, including multi-turn interaction, tool-use tasks, and domains with more diverse verifiable signals.

Finally, DelTA introduces additional computation for coefficient estimation. As discussed in Appendix L.1, the measured overhead is modest in our setting, and future engineering improvements such as more efficient caching or lower-cost proxy computation could further reduce this cost.

- B BROADER IMPACTS

This work studies token-level credit assignment for reinforcement learning from verifiable rewards. Its potential positive impact is to improve the effectiveness and efficiency of training reasoning-capable language models, especially in domains where correctness can be automatically verified, such as mathematics and code generation. By providing a more interpretable view of how sequence-level RLVR updates allocate token-level credit, the work may also help researchers better diagnose and improve RL training dynamics.

At the same time, stronger reasoning models may also be misused in settings such as automated generation of misleading content, scalable code generation for harmful purposes, or other dual-use applications. DelTA does not introduce new user-facing deployment mechanisms, new datasets containing sensitive information, or new privacy-invasive capabilities, but it may improve the capabilities of models trained with RLVR. Responsible deployment of models trained with such methods should therefore follow standard safety practices, including appropriate evaluation, monitoring, access control when needed, and task-specific safeguards for high-risk applications.

- C DERIVATION OF THE LOCAL DAPO UPDATE

In this appendix, we derive the local first-order update form used in Section 3.1. We focus on the update rule induced by the DAPO surrogate. Since dynamic sampling is disabled in our training recipe, it is not included in the derivation.

For a fixed prompt q, let {oi}Gi=1 denote a sampled group of responses drawn from the old policy πθ

##### , where oi = (oi,1,...,oi,|o

i|). Let

old

G

|oi|

N :=

i=1

be the total number of valid response tokens in the group. Conditioning on this rollout batch, the DAPO surrogate is

where

1 N

JDAPO(θ) =

|oi|

G

min ri,t(θ)Aˆi, clip ri,t(θ),1 − ϵlow,1 + ϵhigh A ˆi ,

t=1

i=1

πθ(oi,t | q,oi,<t) πθ

ri,t(θ) =

.

(oi,t | q,oi,<t)

old

We analyze a local update around θold. At this point, ri,t(θold) = 1 for every sampled token. Since 1 lies inside the clipping interval, clipping is locally inactive, and the local gradient of JDAPO agrees with that of the unclipped surrogate:

Therefore,

G

1 N

i=1

∇θJDAPO(θ)

=

θ=θold

|oi|

ri,t(θ)Aˆi.

t=1

|oi|

G

1 N

Aˆi∇θri,t(θ)

##### .

θ=θold

t=1

i=1

Because the denominator of ri,t(θ) is fixed with respect to θ,

∇θπθ(oi,t | q,oi,<t) θ=θ

∇θri,t(θ)

= ∇θ log πθ(oi,t | q,oi,<t)

old

=

. Define the token-gradient vector

(oi,t | q,oi,<t)

##### πθ

θ=θold

θ=θold

old

vi,t := ∇θ log πθ(oi,t | q,oi,<t)

. Then

θ=θold

|oi|

G

1 N

Aˆi vi,t.

∇θJDAPO(θ)

=

θ=θold

t=1

i=1

For a first-order gradient ascent step with learning rate η > 0,

|oi|

G

η N

Aˆi vi,t.

∆θ = η∇θJDAPO(θ)

=

θ=θold

t=1

i=1

Thus, up to a positive proportionality constant,

|oi|

G

Aˆi vi,t.

∆θ ∝

t=1

i=1

Separating this sum by the sign of the sequence-level advantage gives the positive-negative decomposition used in Section 3.1:

|oi|

|oi|

Aˆi vi,t −

|Aˆi|vi,t.

∆θ ∝

t=1

t=1

i:Aˆi>0

i:Aˆi<0

Finally, we derive the induced local change in next-token log-probabilities. Let c denote an arbitrary generation context, and let x be a candidate next token under this context. Define

(x | c). A first-order Taylor expansion around θold gives

∆log π(x | c) := log πθ

old+∆θ(x | c) − log πθ

old

⊤

∆log π(x | c) ≈ ∇θ log πθ(x | c)

∆θ. Substituting the local DAPO update yields

θ=θold

|oi|

⊤ G

Aˆi vi,t.

∆log π(x | c) ∝ ∇θ log πθ(x | c)

θ=θold

t=1

i=1

Equivalently, using the positive-negative decomposition,

 

 .

|oi|

|oi|

⊤

Aˆi vi,t −

|Aˆi|vi,t

∆log π(x | c) ∝ ∇θ log πθ(x | c)

θ=θold

t=1

t=1

i:Aˆi>0

i:Aˆi<0

This is the first-order next-token log-probability change analyzed in Section 3.1.

This expression describes the raw surrogate-gradient direction before optimizer preconditioning, weight decay, or gradient clipping. In practical training with Adam-type optimizers, the actual parameter step may apply an additional preconditioner to this gradient. Our analysis focuses on the update direction induced by the RLVR surrogate itself.

A degenerate case occurs when one advantage side is empty. Under group-normalized advantages, if there is no positive-advantage response in a group, then there is also no negative-advantage response: all advantages in the group are zero. Such a group contributes zero to the local policy-gradient update and is therefore skipped in the positive-negative decomposition. Equivalently, the sums over empty sides are treated as zero, and centroid quantities that require nonzero side mass are computed only for rollout groups with both positive and negative advantage sides.

- D MEAN DIRECTIONS AS WEIGHTED LEAST-SQUARES CENTROIDS

In this appendix, we show that the side-wise mean directions used in the local discriminator view are exactly the weighted least-squares centroids of the positive and negative sides.

Recall the local token-gradient vectors

vi,t := ∇θ log πθ(oi,t | q,oi,<t)

,

θ=θold

where i indexes responses in the rollout group and t indexes valid response tokens. We split tokens according to the sign of the response-level advantage Aˆi. Define the total positive and negative advantage masses as

|oi|

|oi|

|Aˆi|.

Aˆi, M− :=

##### M+ :=

t=1

t=1

i:Aˆi<0

i:Aˆi>0

When M+ > 0 and M− > 0, the corresponding normalized side-wise mean directions are

|oi| t=1

|oi| t=1 |Aˆi|vi,t

Aˆivi,t M+

µ¯+ = i:Aˆ

, µ¯− = i:Aˆ

i>0

i<0

. These are the positive- and negative-side reference directions used by the local discriminator. We now show that µ¯+ and µ¯− are weighted least-squares centroids. Consider first the positive side and define

M−

|oi|

Aˆi ∥vi,t − µ∥22. Differentiating with respect to µ gives

F+(µ) :=

t=1

i:Aˆi>0

∇µF+(µ) = 2

i:Aˆi>0

Setting the gradient to zero yields

|oi|

|oi|

Aˆi(µ − vi,t) = 2M+µ − 2

t=1

i:Aˆi>0

Aˆivi,t.

t=1

|oi| t=1

Aˆivi,t M+

µ = i:Aˆ

i>0

= µ¯+. Moreover,

∇2µF+(µ) = 2M+I, which is positive definite whenever M+ > 0. Thus, µ¯+ is the unique minimizer of F+(µ). The negative side follows analogously. Define

|oi|

F−(µ) :=

i:Aˆi<0

|Aˆi|∥vi,t − µ∥22.

t=1

Then

∇µF−(µ) = 2M−µ − 2

i:Aˆi<0

The first-order optimality condition gives

|oi|

|Aˆi|vi,t.

t=1

|oi| t=1 |Aˆi|vi,t

µ = i:Aˆ

i<0

= µ¯−. Since

M−

∇2µF−(µ) = 2M−I ≻ 0 whenever M− > 0, µ¯− is also the unique minimizer.

Therefore, the side-wise mean directions µ¯+ and µ¯− are exactly the weighted least-squares centroids of the positive and negative token-gradient vectors, respectively. If one side has zero total mass, the corresponding centroid is undefined; such degenerate groups contribute no positive-negative centroid contrast and are skipped in the centroid construction.

- E REPRESENTATIVE RLVR VARIANTS AS TOKEN-WEIGHTED CENTROID ESTIMATORS

In this appendix, we show that several representative RLVR variants can be interpreted through the same local centroid view. The goal is not to claim that these methods are identical, but to clarify how their local update rules modify the effective token weights used to estimate the positive and negative directions.

Consider a generic RLVR surrogate whose unclipped local update around the old policy can be written as

|oi|

G

ρi,tAˆivi,t, (9) where

∆θ ∝

t=1

i=1

vi,t = ∇θ log πθ(oi,t | q,oi,<t)

,

θ=θold

and ρi,t ≥ 0 denotes the effective token weight induced by the objective. Global normalizers shared by all tokens are omitted when they do not affect the update direction.

Throughout this appendix, the effective token weights ρi,t are treated as fixed stop-gradient quantities. Thus, the local update keeps only the gradient through the policy ratio ri,t(θ).

Separating positive- and negative-advantage responses gives

∆θ ∝ M+(ρ)µ+(ρ) − M−(ρ)µ−(ρ), (10) where

|oi| t=1 ρi,tAˆivi,t

|oi| t=1 ρi,t|Aˆi|vi,t

µ+(ρ) = i:Aˆ

, µ−(ρ) = i:Aˆ

i>0

i<0

. (11)

|oi| t=1 ρi,tAˆi

|oi| t=1 ρi,t|Aˆi|

i:Aˆi>0

i:Aˆi<0

Here M+(ρ) and M−(ρ) are the corresponding total positive and negative masses, i.e., the denominators of the two centroids above. Therefore, under the local view, changing ρi,t changes both the side-wise centroids and their relative masses.

GRPO. GRPO averages token losses within each response and then averages over responses. Around the old policy, where clipping is locally inactive, its update direction is

G

1 G

∆θGRPO ∝

i=1

|oi|

Aˆi |oi|

vi,t. (12)

t=1

Thus, up to a global constant,

1 |oi|

ρGRPOi,t =

.

Equivalently, GRPO first forms a response-level average token-gradient vector,

|oi|

1 |oi|

v¯i =

vi,t,

t=1

and then aggregates these response-level vectors with group-relative advantages:

G

Aˆiv¯i.

∆θGRPO ∝

i=1

Therefore, each response contributes one averaged direction. Longer responses have smaller per-token weights, while shorter responses have larger per-token weights. In the centroid view, GRPO estimates side-wise directions from response-balanced token-gradient averages.

DAPO. DAPO replaces response-level averaging with token-level normalization over all valid response tokens. Locally, its update direction is

|oi|

G

Aˆivi,t, (13) where the global token-count normalizer is omitted. Hence

∆θDAPO ∝

t=1

i=1

ρDAPOi,t = 1. Compared with GRPO, DAPO gives equal weight to every valid token. Consequently, the total contribution of a response is proportional to its length. In the centroid view, DAPO changes the side-wise estimator from a response-balanced average to a token-balanced average.

DAPO with forking tokens. DAPO with forking tokens keeps only high-entropy tokens in the policy-gradient loss. Let

##### mFTi,t = I Hi,t ≥ τρB

denote the high-entropy token mask in the current batch. The local update becomes

|oi|

G

mFTi,t Aˆivi,t. (14) Thus,

∆θFT ∝

t=1

i=1

ρFTi,t = mFTi,t . This objective changes the support of the centroid estimator: low-entropy tokens are removed, and the positive and negative centroids are estimated only from high-entropy tokens. In this view, forking-token filtering emphasizes high-entropy tokens as the main contributors to the side-wise directions.

FIPO. FIPO introduces a future-influence weight for each token. In its objective, the standard advantage is multiplied by a Future-KL importance weight fi,t, and this weighted advantage is used inside a DAPO-style token-level surrogate. FIPO defines this weight from a discounted future probability-shift signal and clips it to control variance.

Ignoring clipping locally, the update direction is

|oi|

G

fi,tAˆivi,t. (15) Therefore,

∆θFIPO ∝

t=1

i=1

ρFIPOi,t = fi,t. In the centroid view, FIPO estimates the positive and negative centroids from future-influenceweighted token-gradient vectors. Tokens with larger future-influence weights receive larger centroid mass. This differs from entropy filtering: FIPO does not simply select uncertain tokens, but weights tokens according to a forward-looking estimate of their influence on later trajectory behavior.

Summary. Under a local first-order view, several representative RLVR variants can be interpreted as modifying the effective token weights ρi,t in an advantage-weighted token-gradient aggregation. These weights determine the side-wise centroids µ+(ρ) and µ−(ρ), as well as their total masses. Existing methods adjust this estimator through length normalization, entropy filtering, or futureinfluence weighting. DelTA differs by assigning token weights according to whether a token gradient is more representative of its own side than of the opposite side, thereby aligning the centroid estimator with the discriminative structure of the local DAPO update.

- F LAST-LAYER TOKEN-GRADIENT PROXY

For scalable coefficient estimation, we use a layer-restricted token-gradient proxy based on the standard fixed-representation view of the output layer. Let the LM head produce logits zt = Wht, where ht ∈ Rd is the final hidden state at step t, W ∈ R|V|×d is the LM-head matrix, and pt = softmax(zt). For the realized token yt, the token log-probability is

##### log pt(yt) = Wy⊤

ht − log

t

j

exp(Wj⊤ht).

Differentiating with respect to the corresponding LM-head row gives ∇Wyt

##### log pt(yt) = 1 − pt(yt) ht.

Thus, under a frozen-representation approximation, we use 1 − pt(yt) ht as a layer-restricted proxy for the token-gradient vector.

We do not use this proxy as an exact reconstruction of the full-parameter gradient. Instead, we use it as a scalable layer-restricted representation for estimating relative token coefficients. It is a pragmatic reduction that makes token-level analysis and reweighting feasible in large autoregressive language models. Prior work on gradient-based influence has similarly relied on first-order approximations and layer-restricted computations for scalability (Pruthi et al., 2020), and in NLP it is common to restrict such analysis to the last layer when full-parameter computations are impractical at scale (Yeh et al., 2022). This restriction should therefore be understood as an approximation, not an exact reconstruction of the full gradient signal (Yeh et al., 2022).

Proxy ablations. In addition to the default proxy above, we ablate the choice of layer-restricted proxy used for DelTA coefficient estimation. These ablations do not change the DelTA objective or the policy-gradient update; they only replace the stop-gradient proxy vectors used to compute centroids, distances, and token coefficients.

Let Kt denote the top-K vocabulary indices under the current logits, and let pt be the softmax distribution renormalized within Kt. We consider a top-K hidden-gradient proxy

vthid = Wy

t −

pt(j)Wj,

j∈Kt

which is a top-K approximation to ∇ht

log pt(yt). Compared with the default output-row proxy 1 − pt(yt) ht, this proxy explicitly incorporates the competing high-probability tokens under the

current policy.

We also include a random-coefficient baseline to test whether DelTA benefits merely from perturbing token weights. In this baseline, λi,t is sampled randomly from the same bounded range [λmin,λmax], and then used with the same coefficient normalization and weighted DAPO objective as DelTA. All other training settings are kept unchanged.

Table 4 shows that DelTA is robust to the choice of last-layer proxy. The top-K hidden-gradient proxy achieves the best average performance, improving over the default proxy from 23.27 to 24.29. This suggests that the contrast-sensitive reweighting mechanism does not rely on a single proxy choice, and that incorporating local competition among top vocabulary candidates can provide a stronger signal for coefficient estimation.

In contrast, the random-λ baseline substantially underperforms both DelTA variants, dropping to an average score of 18.34. This confirms that the benefit of DelTA is not due to arbitrary

Table 4: Proxy ablation study of DelTA. The best results are in bold.

Method AIME25 AIME26 HMMT25 HMMT26 Avg. Base DelTA 26.46 28.12 18.54 20.27 23.27 Top-K hidden-gradient proxy 27.08 27.71 20.83 21.78 24.29 Random λ 22.50 22.50 11.87 16.67 18.34

token reweighting or stochastic perturbation of the loss. Rather, the coefficients need to preserve meaningful discriminative structure: tokens should receive larger weights when their gradients are more representative of their own advantage side than of the opposite side.

Therefore, the gains reported in our main experiments are obtained with a conservative and computationally minimal proxy. The stronger performance of the top-K hidden-gradient variant further suggests that DelTA is not tied to this specific proxy choice and may benefit from more informative gradient proxies. We keep the output-row proxy as the default in the main experiments because it is the minimal and most direct layer-restricted instantiation of the token-gradient view; the stronger top-K proxy shows that DelTA can further benefit from richer proxy choices.

- G DERIVATION OF THE DELTA SOFT ASSIGNMENT SCORE

In this appendix, we derive the closed-form solution of the soft assignment score used in DelTA. We follow the notation in Section 3.2. The centroids µ+ and µ− are fixed in this derivation, and αi,t ∈ [0,1] denotes the raw assignment score before the final remapping and normalization step.

Side-specific squared-distance margin. For a token from a positive-advantage response, DelTA assigns a larger score when its token-gradient vector is closer to the positive centroid than to the negative centroid. We define the positive-side squared-distance margin as

##### ∆+i,t := ∥vi,t − µ−∥22 − ∥vi,t − µ+∥22, Aˆi > 0.

Thus, ∆+i,t > 0 means that vi,t is closer to µ+ than to µ−. For a token from a negative-advantage response, the two centroids are swapped, and we define

∆−i,t := ∥vi,t − µ+∥22 − ∥vi,t − µ−∥22, Aˆi < 0. Hence, ∆−i,t > 0 means that vi,t is closer to µ− than to µ+. Positive-side assignment. For a token from a positive-advantage response, Eq. (5) can be written compactly as

α∆+i,t + γ+h(α), where

αi,t = arg max

α∈[0,1]

h(α) = −α log α − (1 − α)log(1 − α) is the binary entropy, with the standard convention 0log 0 = 0. Since the centroids and temperature are fixed, this is a one-dimensional optimization problem. To simplify notation, write ∆ = ∆+i,t and γ = γ+. We solve

max

f(α) := α∆ + γh(α).

α∈[0,1]

For α ∈ (0,1), the derivative is

1 − α α

f′(α) = ∆ + γh′(α) = ∆ + γ log

. Setting f′(α) = 0 gives

1 − α α

∆ γ

= −

log

,

or equivalently

α 1 − α

∆ γ

. Solving for α yields

= exp

1 1 + exp(−∆/γ)

∆ γ

α =

= σ

,

where σ(z) = 1/(1 + exp(−z)). The solution is the unique maximizer. Indeed,

1 α

1 1 − α

γ α(1 − α)

f′′(α) = γh′′(α) = −γ

= −

+

< 0

for all α ∈ (0,1), since γ > 0. Therefore, f is strictly concave on (0,1), and the stationary point above is the unique optimum. Substituting back ∆ = ∆+i,t and γ = γ+, we obtain

αi,t = σ ∥vi,t − µ−∥22 − ∥vi,t − µ+∥22 γ+

, Aˆi > 0.

Negative-side assignment. For tokens from negative-advantage responses, DelTA uses the symmetric objective obtained by swapping the positive and negative centroids:

α∆−i,t + γ−h(α), Aˆi < 0.

αi,t = arg max

α∈[0,1]

Repeating the same one-dimensional derivation with ∆ = ∆−i,t and γ = γ− gives

αi,t = σ ∥vi,t − µ+∥22 − ∥vi,t − µ−∥22 γ−

, Aˆi < 0. Combining the two sides, the DelTA soft assignment score is

 

σ ∥vi,t − µ−∥22 − ∥vi,t − µ+∥22 γ+

, Aˆi > 0, σ ∥vi,t − µ+∥22 − ∥vi,t − µ−∥22

αi,t =



, Aˆi < 0, which matches Eq. (6).

γ−

The score αi,t is a raw stop-gradient assignment score. As described in Appendix H, DelTA subsequently remaps these scores to a bounded coefficient range and normalizes them before using them as loss coefficients in the weighted DAPO surrogate.

Connection to the inner-product discriminator. For fixed centroids µ+ and µ−, the positive-side squared-distance margin used by DelTA can be rewritten as

∥v − µ−∥22 − ∥v − µ+∥22 = 2v⊤(µ+ − µ−) + ∥µ−∥22 − ∥µ+∥22. Thus, up to a centroid-dependent offset, the distance margin scores v by its alignment with the centroid contrast direction µ+ −µ−. The negative-side margin is obtained symmetrically by swapping µ+ and µ−. This shows that the squared-distance comparison used for coefficient estimation is consistent with the positive-negative inner-product discriminator view in Section 3.1, while matching the weighted least-squares geometry of the side-wise centroids.

- H DELTA IMPLEMENTATION DETAILS

In this appendix, we describe how DelTA is computed in practice on each rollout batch. The raw assignment scores depend on the side-wise centroids, while the centroids are themselves estimated using these scores. We therefore use a small fixed number of lagged alternating refinement steps rather than solving the coupled problem to convergence.

Let {vi,t} denote the token-gradient vectors in the rollout batch, and let {Aˆi} denote the corresponding sequence-level advantages. Throughout this appendix, i ∈ {1,...,G} indexes sampled responses

and t ∈ {1,...,|oi|} indexes valid response tokens. We use αi,t ∈ [0,1] for raw assignment scores, and reserve λi,t for the final training coefficient used in the weighted DAPO surrogate. All distances below are standard squared Euclidean distances.

Initialization of side-wise centroids. We initialize the positive and negative centroids from the original advantage-weighted token-gradient means. Define

|oi|

|oi|

|Aˆi|.

Aˆi, M− =

##### M+ =

t=1

t=1

i:Aˆi<0

i:Aˆi>0

The initial centroids are

|oi| t=1

|oi| t=1 |Aˆi|vi,t

Aˆi vi,t M+

µ(0)+ = i:Aˆ

, µ(0)− = i:Aˆ

i>0

i<0

.

M−

In implementation, the denominators are clamped by a small numerical constant ε > 0 for stability.

Squared-distance differences and lagged adaptive temperatures. For centroids µ(+k) and µ(−k), define the positive-side and negative-side squared-distance differences as

∆+i,t,(k) = ∥vi,t − µ(−k)∥22 − ∥vi,t − µ(+k)∥22, Aˆi > 0, and

∆−i,t,(k) = ∥vi,t − µ(+k)∥22 − ∥vi,t − µ(−k)∥22, Aˆi < 0. A larger value means that the token-gradient vector is closer to its own-side centroid than to the opposite-side centroid.

The side-specific temperatures are set from the empirical standard deviations of these squared-distance differences. For the initial centroids, we compute

γ+(0) = max Var {∆+i,t,(0) : Aˆi > 0, 1 ≤ t ≤ |oi|} ,εγ ,

γ−(0) = max Var {∆−i,t,(0) : Aˆi < 0, 1 ≤ t ≤ |oi|} ,εγ , where εγ > 0 is a small numerical constant.

The superscript of γ±(k) denotes the cached temperature used when computing αi,t(k). In implementation, the temperatures are updated in a lagged manner. During the pass that computes αi,t(k), we also accumulate the empirical variances of ∆+i,t,(k) and ∆−i,t,(k), and store the resulting temperatures as γ+(k+1) and γ−(k+1) for the next score-computation step. Thus, for k > 0, γ±(k) is the cached temperature produced by the previous margin-statistics pass. This avoids an additional proxy forward pass solely for temperature estimation.

Alternating refinement. Starting from µ(0)+ , µ(0)− , γ+(0), and γ−(0), DelTA runs K stop-gradient refinement iterations. At iteration k = 0,...,K − 1, the raw assignment scores are

 

σ ∆+i,t,(k)/γ+(k) , Aˆi > 0, σ ∆−i,t,(k)/γ−(k) , Aˆi < 0,

αi,t(k) =



where σ(·) is the sigmoid function. Equivalently, this is the closed-form solution of the entropyregularized assignment objective in Section 3.2.

The centroids are then updated as score-weighted within-side averages:

|oi| t=1

Aˆi αi,t(k)vi,t

µ(+k+1) = i:Aˆ

i>0

,

|oi| t=1

Aˆi αi,t(k)

i:Aˆi>0

|oi| t=1 |Aˆi|αi,t(k)vi,t

µ(−k+1) = i:Aˆ

i<0

.

|oi| t=1 |Aˆi|αi,t(k)

i:Aˆi<0

Again, denominators are clamped by a small numerical constant for stability. The same pass also accumulates the statistics of ∆+i,t,(k) and ∆−i,t,(k), producing γ+(k+1) and γ−(k+1) for the next refinement iteration.

Final coefficient computation. After K refinement iterations, we obtain refined centroids µ(+K) and µ(−K). A final pass over the rollout batch recomputes the squared-distance differences using these refined centroids:

##### ∆+i,t,(⋆) = ∥vi,t − µ(−K)∥22 − ∥vi,t − µ(+K)∥22, Aˆi > 0,

∆−i,t,(⋆) = ∥vi,t − µ(+K)∥22 − ∥vi,t − µ(−K)∥22, Aˆi < 0. The final raw scores are computed using the latest cached temperatures:

 

σ ∆+i,t,(⋆)/γ+(K) , Aˆi > 0, σ ∆−i,t,(⋆)/γ−(K) , Aˆi < 0.

αi,t⋆ =



When K = 0, this reduces to using the initial centroids µ(0)+ , µ(0)− and initial temperatures γ+(0), γ−(0). The raw scores are then mapped to bounded loss coefficients:

##### λi,t = λmin + (λmax − λmin)αi,t⋆ , Aˆi ̸= 0.

For zero-advantage responses, which do not belong to either side, we assign

##### λi,t = λmin.

Self-normalized implementation with the standard DAPO token average. The self-normalized DelTA objective in Eq. (8) can be implemented while keeping the standard DAPO token-count normalizer. Let

|oi|

G

G

|oi|, Z =

N =

λi,t.

t=1

i=1

i=1

We define the implementation-facing coefficient

N Z

λ¯i,t = λi,t

. By construction,

|oi|

G

1 N

λ¯i,t = 1.

t=1

i=1

Therefore, for the clipped DAPO token loss

ℓi,t(θ) = min ri,t(θ)Aˆi, clip ri,t(θ),1 − ϵlow,1 + ϵhigh A ˆi , we have

|oi|

|oi|

G

G

1 Z

1 N

λ¯i,tℓi,t(θ) =

λi,tℓi,t(θ).

t=1

t=1

i=1

i=1

Thus, using λ¯i,t under the standard DAPO token average is exactly equivalent to the self-normalized DelTA objective. The normalization changes only the global scale of the coefficients and preserves their relative token reweighting.

Use in training. The coefficients λi,t and λ¯i,t are treated as stop-gradient quantities. They are computed once from the rollout batch, held fixed across repeated optimization passes over that batch,

and recomputed only when new trajectories are sampled. In the actual loss computation, λ¯i,t is used as the per-token multiplier, and the rest of the DAPO training pipeline remains unchanged.

Summary. For each rollout batch, DelTA proceeds as follows:

- 1. Compute layer-restricted token-gradient proxies vi,t for all valid rollout tokens.
- 2. Initialize µ(0)+ and µ(0)− from the original advantage-weighted side-wise centroids.
- 3. Compute initial temperatures γ+(0) and γ−(0) from the squared-distance differences under the initial centroids.
- 4. For k = 0,...,K − 1:

- (a) compute raw assignments αi,t(k) using µ(+k), µ(−k), and the cached temperatures γ+(k), γ−(k);
- (b) update µ(+k+1) and µ(−k+1) as score-weighted within-side centroids;
- (c) store γ+(k+1) and γ−(k+1) from the same squared-distance-difference statistics for the next assignment computation.

- 5. Compute final raw assignments αi,t⋆ using the refined centroids µ(+K), µ(−K), and latest cached temperatures γ+(K), γ−(K).
- 6. Map αi,t⋆ to bounded coefficients λi,t, normalize them into λ¯i,t, and use λ¯i,t inside the standard DAPO token average.

- I DETAILED SETTINGS

The training settings are presented in Table 5. We train the Qwen3-8B-Base model for 220 steps and the Qwen3-14B-Base model for 300 steps. For checkpoint selection, we evaluate checkpoints during training and select the checkpoint with the highest AIME25 avg@8 score. This selection rule is fixed in advance and applied uniformly to all methods and backbone sizes. Thus, our main comparison reflects the best checkpoint performance achieved by each method under the same training budget and the same model-selection protocol.

All existing assets used in this work are publicly available research assets. We cite their original sources and follow their corresponding licenses and terms of use.

All experiments were conducted on 8× NVIDIA B200 GPUs. For the main DelTA training experiments, the wall-clock time per training step was approximately 5 minutes at the beginning of Qwen3-8B-Base training and increased to approximately 10 minutes as response lengths grew. For Qwen3-14B-Base, the wall-clock time per training step was approximately 8 minutes at the beginning of training and increased to approximately 18 minutes as response lengths grew.

We use a simple binary verifiable reward: a response receives reward 1 if its final answer is correct, and 0 otherwise. Answer correctness is determined by math-verify1. Evaluation hyperparameters are presented in Table 6.

- J BASELINE DETAILS In this section, we briefly introduce the baseline methods used in our experiments.

DAPO. DAPO (Yu et al., 2025) extends GRPO by using asymmetric clipping, and token-level loss normalization to improve long-CoT RL training. We use DAPO as our main strong baseline.

DAPO w/ Forking Tokens. DAPO w/ Forking Tokens (Wang et al., 2025) (DAPO w/ FT for short) is a token-filtered variant of DAPO that keeps only the top 20% high-entropy tokens, referred to as forking tokens, in the policy-gradient loss. The remaining low-entropy tokens are masked out, based on the observation that high-entropy tokens often correspond to critical reasoning forks.

1https://github.com/huggingface/Math-Verify

Table 5: Training settings.

Hyper-parameter Value Train Batch Size 128 Micro Batch Size 32 Rollout n 16 Maximum Prompt Length 2048 Maximum Response Length 20,480 Clip Ratio Low 0.2 Clip Ratio High 0.28 Rollout Engine SGLang Temperature 1.0 Top-p 1.0 LR 1 × 10−6 KL Coefficient 0.0

Table 6: Evaluation hyper-parameters.

Hyper-parameter Value Max Length 30,000 Temperature 1.0 Top-p 0.7 Inference Engine SGLang

Table 7: Unique training hyper-parameters of SAPO.

Table 8: Unique training hyper-parameters of FIPO.

Hyper-parameter Value Gae Gamma 1.0 Gae Lam 0.95 Tau Pos 1.0 Tau Neg 1.05

Hyper-parameter Value Decay Rate 32.0 Chunk Size 128 Future KL Start include current Future KL Window -1 Future KL Average False Future KL Clip Ratio 0.2 Future KL Clip High Only True Safety Thresh 10.0

SAPO. SAPO (Gao et al., 2025) replaces hard clipping with a smooth, temperature-controlled soft gate to attenuate off-policy updates. It further uses different temperatures for positive- and negative-advantage tokens to improve training stability.

FIPO. FIPO (Ma et al., 2026) reweights token-level advantages using discounted Future-KL, which estimates how much each token influences the subsequent trajectory. The resulting influence weights are clipped to control variance during policy optimization.

The method-specific hyperparameters of SAPO and FIPO are provided in Table 7 and Table 8, respectively.

- K SIGNIFICANCE TEST DETAILS

We describe the statistical significance tests used in our main comparison. Due to the high cost of RLVR training, we do not repeat full training runs with multiple random seeds. Instead, we perform an evaluation-run-level significance test that captures stochasticity from repeated generation.

For each method, we repeat the full evaluation S = 16 times. Each repetition runs the model over the evaluation suite and produces one question-count-weighted aggregate score. Since different methods are evaluated with independently sampled runs, we treat the 16 scores from DelTA and the 16 scores from the baseline as two independent samples.

We use a one-sided Mann–Whitney U test as the primary non-parametric test, with the pre-specified alternative hypothesis that DelTA outperforms the baseline. We regard the improvement as statistically significant when the one-sided p-value is below 0.05.

Table 9: Main results on seven mathematical reasoning benchmarks for Olmo3-7B-Base. The best results are in bold.

HMMT25 (Feb.)

HMMT25 (Nov.)

HMMT26 (Feb.)

- Method AIME24 AIME25 AIME26

Brumo25 Avg. Olmo3-7B-Base

DAPO 25.62 21.88 22.08 14.58 8.75 14.02 26.67 19.01 DelTA 30.83 24.79 27.08 16.67 13.96 16.29 30.63 22.80

For the main significance claim, we compare DelTA with the strongest same-scale baseline for each backbone size, namely SAPO for the 8B backbone and FIPO for the 14B backbone. Under this evaluation-run-level testing protocol, DelTA significantly outperforms the strongest same-scale baseline at both model scales.

- L SUPPLEMENTARY EXPERIMENTS

- L.1 COMPUTATIONAL OVERHEAD

In this subsection, we discuss the computational overhead introduced by DelTA.

The DelTA proxy is computed from final-layer hidden states, which are already produced when evaluating old log-probabilities. Therefore, if all hidden states for the rollout batch could be cached, the centroid refinement itself would not require additional actor forward passes: the same fixed token-gradient proxies could be reused across refinement iterations.

In practice, caching all hidden states for long-response RLVR rollouts is memory intensive. We therefore recompute the proxy whenever it is needed. Relative to the standard old-log-probability computation, the additional actor forward passes for DelTA with K refinement iterations are:

- 1. one pass to estimate the initial temperature scale γ;
- 2. one pass for each refinement iteration k = 1,...,K, which computes token scores and updates the centroids;
- 3. one final pass to compute the final token coefficients used in the weighted DAPO objective.

Thus, DelTA requires K + 2 additional actor forward passes for coefficient estimation. In our main experiments, we use K = 1, which already gives strong performance. These additional passes are only used to compute stop-gradient coefficients; the resulting weighted DAPO objective is optimized in the same way as the baseline objective.

To empirically measure the overhead, we compare the execution time of the first training step between DelTA and DAPO. We focus on the first step because response lengths can diverge during training, as shown in Figure 2. Since rollout generation dominates wall-clock time in long-response RLVR, later-step end-to-end comparisons would be confounded by differences in generated response length rather than isolating the overhead of DelTA’s coefficient computation.

On 8 NVIDIA B200 GPUs, the first step of DelTA takes 37 seconds longer than DAPO. Since rollout generation dominates long-response RLVR, this corresponds to approximately 10.2% of the total first-step time of DelTA. These results indicate that, when isolating the optimization phase from length-induced rollout variation, the overhead of DelTA’s centroid refinement is modest for RLVR workloads.

- L.2 OTHER MODEL ARCHITECTURES

To further examine the generality of DelTA beyond the Qwen3 backbones, we also conduct experiments on Olmo3-7B-Base. We train DelTA and the DAPO baseline with the same hyperparameter settings as in the main experiments, without architecture-specific tuning. This setting allows us to test whether the proposed token-level credit assignment remains effective on a different model family.

Table 11: Hyperparameter sensitivity study of DelTA. The best results are in bold, and the second-best results are underlined.

- Method AIME25 AIME26 HMMT25 HMMT26 Avg. Base DelTA 26.46 28.12 18.54 20.27 23.27

λmin = 0.5 25.42 28.96 19.17 19.70 23.22 λmax = 1.5 26.46 26.25 18.75 19.89 22.77 λmin = 0.5, λmax = 1.5 25.62 27.92 18.96 20.08 23.07

- K = 2 25.00 27.08 18.33 18.56 22.15
- K = 3 26.04 26.67 18.12 18.37 22.20

- As shown in Table 9, DelTA consistently outperforms DAPO on all seven mathematical reasoning benchmarks. The average score improves from 19.01 to 22.80, corresponding to a gain of 3.79 points. The improvement is particularly large on AIME24, AIME26, HMMT25-Nov, and Brumo25, while DelTA also maintains clear gains on the remaining benchmarks. These results suggest that DelTA is not tied to a specific Qwen3 backbone, and that discriminative token-level reweighting can provide consistent benefits across different base model families.

L.3 CODE GENERATION

Table 10: Code generation results. DelTA consistently improves performance across all benchmarks.

Method HumanEval+ MBPP+ LCB Avg.

DAPO 83.0 72.1 33.4 47.7 DelTA 84.6 73.2 35.6 49.5

To further validate the effectiveness of DelTA beyond mathematical reasoning, we conduct experiments on code generation tasks. We use Eurus2RL-Code (Cui et al., 2025) as the training dataset and adopt DAPO as the baseline. All models are trained for two epochs under the same training recipe. We evaluate the trained models on HumanEval+, MBPP+ (Liu et al., 2023), and LiveCodeBench (Jain et al., 2024). For each problem, we sample 5 rollouts and report the average accuracy. We also report a weighted average across benchmarks, where each benchmark is weighted by its number of evaluation problems.

- As shown in Table 10, DelTA consistently improves over DAPO on all three code generation benchmarks, increasing the weighted average score from 47.7 to 49.5. These results suggest that the benefit of DelTA is not limited to mathematical reasoning, but also transfers to code generation tasks, where effective token-level credit assignment remains important under sequence-level supervision.

L.4 Q4: IS DELTA SENSITIVE TO ITS HYPERPARAMETERS?

We further study the sensitivity of DelTA to its method-specific hyperparameters. We focus on two factors: the coefficient range [λmin,λmax], which controls the strength of token reweighting, and the number of centroid refinement iterations K, which controls how many times the side-wise centroids are updated before computing the final coefficients. For the coefficient range, we evaluate several configurations, including [0.5,1.2], [0.8,1.5], and [0.5,1.5]. For the refinement step, we vary K over different values, including K = 2 and K = 3.

- As shown in Table 11, DelTA is relatively robust to the coefficient range: changing λmin or λmax only leads to small average-score variations, and λmin = 0.5 achieves a comparable average score to the base setting. In contrast, increasing the refinement depth to K = 2 or K = 3 consistently reduces performance, suggesting that a single refinement step is sufficient and that excessive refinement may overfit the batch-level token-gradient geometry. Overall, the base configuration achieves the

best average performance, indicating that [λmin,λmax] = [0.8,1.2] with K = 1 provides a stable trade-off between effective token reweighting and optimization robustness.

- L.5 Q5: DOES DELTA GENERALIZE TO OUT-OF-DOMAIN EVALUATION?

To examine whether DelTA generalizes beyond the indomain mathematical reasoning benchmarks, we further evaluate DAPO and DelTA on two out-of-domain benchmarks: GPQA-Diamond and MMLU-Pro. Since MMLU-Pro contains a large number of questions, we randomly sample 600 questions for evaluation. For each question, we sample 5 responses and report Avg@5. We also report a question-count weighted average across the two benchmarks.

Table 12: OOD evaluation on GPQA-D and MMLU-Pro. The best results are in bold.

Method GPQA-D MMLU-Pro Avg. Qwen3-8B-Base

DAPO 43.43 63.97 58.87 DelTA 50.00 66.47 62.38

Qwen3-14B-Base

DAPO 54.55 70.80 66.77 DelTA 56.67 72.27 68.40

- As shown in Table 12, DelTA consistently improves over DAPO on both OOD benchmarks and both backbone sizes. On Qwen3-8B-Base, DelTA improves the weighted average from 58.87 to 62.38. On Qwen314B-Base, DelTA further improves the weighted average from 66.77 to 68.40. These results suggest that DelTA does not merely overfit to the main mathematical evaluation suite, but also transfers to broader scientific and general knowledge reasoning tasks.

- L.6 TOKEN WEIGHT ANALYSIS

To understand what DelTA emphasizes, we visualize high- and lowweight token clouds in Figure 5. The visualization is based on selected generated tokens collected during training at a generation scale of roughly 108 tokens. For each token type, we compute its average DelTA coefficient over its occurrences; larger tokens indicate higher average coefficients in the high-weight cloud and lower average coefficients in the low-weight cloud. This contextagnostic visualization is qualitative, since DelTA assigns coefficients to tokens based on their gradient vectors in specific rollout contexts rather than to token strings in isolation. The high-weight cloud contains several tokens related to transformations and verification, such as scaffold, prime, =y, forward, and backward. In contrast, the low-weight cloud is dominated by more backgroundlike or entity-specific tokens, such as Seat, domain, players, Vander, and Hamilton.

[Figure 1]

- (a) High-weight tokens

[Figure 2]

- (b) Low-weight tokens

Figure 5: Token clouds of high-weight and low-weight tokens.

This pattern is consistent with DelTA’s design. Rather than reflecting semantic importance alone, λi,t reflects whether the gradient vector of a token is more representative of its own advantage side than of the opposite side. DelTA therefore tends to assign larger coefficients to more discriminative reasoning-related tokens and smaller coefficients to shared or less informative background tokens.

