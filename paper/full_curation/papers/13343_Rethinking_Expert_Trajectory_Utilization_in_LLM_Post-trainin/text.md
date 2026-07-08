# arXiv:2512.11470v2[cs.LG]11May2026

## Rethinking Expert Trajectory Utilization in LLM Post-training for Mathematical Reasoning

Bowen Ding1,2 , Yuhan Chen2* , Jiayang Lyu2* , Jiyao Yuan4 , Qi Zhu4, Shuangshuang Tian2*, Dantong Zhu2* , Futing Wang1,2 , Heyuan Deng4 , Fei Mi4† , Lifeng Shang4 , Tao Lin2,3†, 1 Zhejiang University 2 School of Engineering, Westlake University 3 Institute of Advanced Technology, Westlake Institute for Advanced Study 4 Huawei Noah’s Ark Lab 2{dingbowen, wangfuting, lintao}@westlake.edu.cn 4{yuanjiyao1, zhuqi41, dengheyuan, mifei2, Shang.Lifeng}@huawei.com

### Abstract

Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) dominate the post-training landscape for mathematical reasoning, yet differ fundamentally in their reliance on expert trajectories. To understand the optimal way to harness these trajectories for maximizing performance, we propose the Plasticity-Ceiling Framework. This framework empirically grounds the post-training landscape by decomposing the final performance ceiling into the foundational SFT performance and the subsequent RL plasticity (i.e., the maximum improvement via RL). Through extensive benchmarking, we establish the Sequential SFTthen-RL pipeline as the superior standard, overcoming the stability and premature convergence deficits inherent in synchronized approaches. Furthermore, we derive precise scaling guidelines: (1) Transitioning to RL at the Stable or Mild Overfitting Regime of SFT maximizes the final ceiling by securing a robust SFT foundation with substantial RL plasticity; (2) Refuting the “Less is More” hypothesis in SFT-then-RL scaling, we demonstrate that Data Scale determines the primary post-training potential, while Trajectory Difficulty acts as a performance multiplier; and (3) The Minimum Validation Loss of SFT serves as a reliable indicator for selecting the expert trajectories that maximize the ultimate performance ceiling. Our findings provide actionable guidelines for extracting maximum value from expert trajectories. Code: https:

//github.com/LINs-lab/RETU.

### 1 Introduction

Mathematical reasoning demands strict logic and complex planning, making it a critical benchmark for AI intelligence. Evolving generic Large Language Models (LLMs) into specialized Large Reasoning Models (LRMs) capable of tackling these

* Work done at Westlake University as an intern. † Corresponding authors.

[Figure 1]

Figure 1: The post-training overview. The sequential SFT-then-RL pipeline (blue→orange) achieves the superior performance ceiling Apost, outperforming both Pure-RL (orange) and Synchronized SFT-RL (striped) pathways. Insets highlight that: (i) larger and more challenging datasets enhance RL plasticity (top-left); and (ii) RL yields optimal results when initiated from the Stable Regime of SFT (bottom-right).

tasks relies on effective post-training, often combining Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) (Shao et al., 2024; Yang et al., 2024).

These paradigms diverge fundamentally in their reliance on expert trajectories. SFT fully relies on expert trajectories (prompt-solution pairs) to instill reasoning priors via imitation, while RL methods such as GRPO (Shao et al., 2024) allow models to leverage prompt-answer pairs to self-explore reasoning paths through reward incentives, without the expert trajectory utilization. Despite the consensus on the necessity of both, a critical question remains unresolved: What is the optimal mechanism to utilize expert trajectories (i.e., SFT data) to maximize the post-training performance ceiling in mathematical reasoning?

The methodology for effective expert trajectory utilization currently faces an unresolved paradigm dilemma. Recent works propose Synchronized SFT-RL (Syn-SFT-RL) algorithms, such as UPT (Lv et al., 2025), SRFT (Fu et al., 2025), and LUFFY (Yan et al., 2025), which integrate imita-

tion loss directly into the RL optimization loop. While these methods often claim superior efficiency over sequential approaches, this advantage is critically constrained by their reliance on limited SFT data (only about 46K). This raises a fundamental question: whether Syn-SFT-RL maintains its claimed superiority when measured against the SFT and sequential SFT-then-RL baselines thoroughly optimized on significantly larger expert trajectory datasets.

Conversely, some LLM practitioners (Yang et al., 2025; GLM et al., 2025; DeepSeek-AI, 2025) typically favor the straightforward sequential SFT-thenRL pipeline. However, the principles governing this successful approach remain largely empirical and lack systematic definition in two critical areas. First, concerning the Optimal Timing for switching from SFT to RL, the criteria lack a formal guidance. Second, regarding Data Properties, although the “Less is More” (Ye et al., 2025; Muennighoff et al., 2025) approach achieves comparable SFT accuracy with minimal data, it is unclear whether this compromises the subsequent RL scaling potential or leads to premature convergence. Similarly, while harder data push SFT boundaries (Tong et al., 2024; Zhang et al., 2025a), its precise influence on the overall post-training ceiling remains unclarified. Consequently, these tensions highlight the urgent need for a unified framework to understand how the characteristics of SFT data dictate the entire post-training performance.

To rigorously address these systemic gaps, we propose a Plasticity-Ceiling analytical framework in § 4. This framework provides a unified view of all paradigms and enables the quantitative decomposition of the theoretical performance ceiling (Apost) into two measurable components: the SFT Performance (Psft) achieved under SFT compute xsft, and the remaining RL Plasticity (PLrl), which represents the maximum potential for subsequent RL improvement.

By conducting extensive experiments with the large-scale (i.e., 889K unique samples in maximum) SFT data on Qwen2.5-7B (Qwen et al., 2025) and validating on Llama3.2-3B (Meta AI, 2024) on the test set including six popular mathematical benchmarks, we demystify expert trajectory utilization and establish a rigorous standard for post-training scaling in mathematical reasoning: ➊ Sequential Paradigm Dominance (§ 6.1). We empirically establish the superiority of the Sequential SFT-then-RL pipeline over the unstable, sensitive

Syn-SFT-RL approaches, as well as pure SFT and RL. A robust SFT phase is necessary to establish the foundational SFT performance (Psft) and unlock the maximum plasticity (PLrl) of subsequent RL. ➋ Switch RL until SFT Saturation (§ 6.2.1). We identify the Stable or Mild-Overfitting Regime of validation loss saturation as the optimal SFT-toRL transition window, both for the in-domain and the out-of-domain tasks. ➌ Scale and Difficulty Extend Ceiling (§ 6.2.2). We refute the “Less is More” hypothesis in the context of SFT-then-RL scaling. While minimal data yields SFT efficiency, the SFT data scale remains the primary determinant of the final ceiling, while the trajectory difficulty acts as a multiplier. Furthermore, the Minimum Validation Loss of SFT serves as a robust predictor of the final post-training ceiling.

Our contributions are summarized as follows: ➊ We propose the Plasticity-Ceiling Framework, a theoretical mechanism that decomposes posttraining performance into realized SFT performance and the subsequent RL plasticity to guide paradigm selection. ➋ We systematically benchmark diverse training strategies, identifying the Sequential SFT-then-RL pipeline as the rigorous standard for stability and performance over other approaches. ➌ We formulate actionable insights in SFT-then-RL pipeline, linking data properties and training dynamics to the final reasoning ceiling to enable more effective post-training development for mathematical reasoning.

### 2 Related Works

Post-Training Paradigms. Post-training primarily relies on SFT and RL. While theoretical works attempt to unify them (Swamy et al., 2025; Wang et al., 2025), they exhibit distinct empirical behaviors regarding generalization and distribution shifts (Huan et al., 2025; Shenfeld et al., 2025). The sequential SFT-then-RL strategy is the industrial standard (Yoshihara et al., 2025; Vattikonda et al., 2025), though optimizing the transition is non-trivial; Kang et al. (2025) caution that high SFT scores can be misleading, as over-fitted models may fail to improve during RL. Alternatively, Synchronized SFT-RL methods like LUFFY (Yan et al., 2025), UPT (Lv et al., 2025), SRFT (Fu et al., 2025) and Prefix-RFT (Huang et al., 2025) integrate imitation directly into RL to boost efficiency. Our work systematically compares these paradigms to identify the optimal mechanism for

maximizing the performance ceiling.

Expert Trajectories Utilization. The properties of SFT data critically influence the post-training. Regarding scale, a “Less is More” philosophy suggests that minimal, high-quality data suffices for SFT (Ye et al., 2025; Muennighoff et al., 2025). However, others argue that scale remains essential for complex reasoning (Sun et al., 2025). Regarding difficulty, methods like MetaMath (Yu et al., 2023) and D3 (Zhang et al., 2025a) demonstrate that harder, difficulty-aware data selection improves SFT outcomes. Crucially, prior works often evaluate SFT in isolation. We extend this inquiry to the RL phase, investigating how SFT data scale and difficulty dictate the model’s plasticity (its headroom for subsequent RL scaling) rather than just immediate accuracy after SFT.

### 3 Preliminary

In this section, we summarize algorithms of various post-training paradigms, each characterized by distinct approaches to utilizing expert trajectories. Details are in Appx. B.

SFT optimizes the policy πθ via imitation learning on expert trajectory pairs (q,τ) from DSFT (Abdulhai et al., 2023). While it reliably instills expert knowledge, its performance is strictly bounded by the quality of thetraining data and lacks selfexploration capabilities (Ouyang et al., 2022).

RL (GRPO & DAPO) enhances the model’s reasoning capability via reward-guided exploration. In contrast to SFT, standard RL algorithms such as GRPO (Shao et al., 2024) and DAPO (Yu et al., 2025) typically employ outcome reward modeling and operate without the utilization of expert trajectories. We adopt GRPO, which eliminates the critic value function by employing groupnormalized advantages Ai,t derived from a group of G self-generated trajectories. The objective maximizes the expected advantage subject to a KL-divergence constraint against the reference policy πref. To further enhance stability, we utilize DAPO as our primary RL algorithm. DAPO builds upon GRPO by introducing asymmetric clipping ratios (ϵlow,ϵhigh) and a dynamic difficulty sampling mechanism to filter extreme easy or hard prompts. Synchronized SFT-RL (Syn-SFT-RL) paradigm attempts to merge the benefits of both SFT and RL by explicitly injecting expert trajectories into the RL optimization loop. We consider four representative baselines: (1) LUFFY (Yan et al.,

2025), which jointly optimizes on-policy rollouts and off-policy expert trajectories using regularized importance shaping; (2) SRFT (Fu et al., 2025), which balances SFT, off-policy RL, and likelihood-shaping losses via entropy-guided dynamic weights; and (3) UPT (Lv et al., 2025), which employs a hard gating mechanism to switch between SFT (stabilization) and GRPO (exploration) based on prompt-level reward thresholds. (4) Prefix-RFT (Huang et al., 2025), which samples prefixes from expert trajectories to guide model exploration, treating the composite sequence of an off-policy prefix and its on-policy continuation as a unified trajectory for reinforcement updates.

### 4 The Plasticity-Ceiling Framework

To systematically evaluate the trade-offs among post-training methods, we propose the PlasticityCeiling analytical framework. Unlike prior works that study SFT or RL scaling in isolation (Chen et al., 2025; Khatri et al., 2025), our framework treats the SFT-then-RL pipeline as a unified continuum. This allows us to quantify the respective contributions of the SFT and RL phases to the overall post-training performance ceiling (Apost) in Def. 1.

- 4.1 Decompose the Post-training Performance Formally, we decompose the post-training perfor-

mance Ppost of the typical SFT-then-RL pipeline into three distinct components based on the training stages:

Ppost(xsft, xrl) = P0 + (Psft(xsft) − P0)

SFT gain, ∆Psft(xsft)

+ (Prl(xrl) − Psft (xsft))

RL gain, ∆Prl(xrl)

Psft (xsft = 0) = P0, Prl(xrl = 0) = Psft (xsft) ,

(1)

where P0 denotes the base model’s initial performance, and xsft, xrl denote the compute cost in FLOPs (see Appx. E) during the SFT and RL phases, respectively. This decomposition explic-

itly isolates the performance contributors: ∆Psft represents the gain realized from SFT given cost xsft, while ∆Prl represents the gain from RL given cost xrl. Notably, Eq. 1 reduces to Pure-SFT when xrl = 0, and to Pure-RL (including Syn-SFT-RL variants) when xsft = 0.

- 4.2 Ceiling and Plasticity

Since RL is the final stage of the post-training pipeline, its scaling behavior determines the ulti-

mate performance ceiling. To quantify this, we model the post-training performance Ppost as a function of RL compute xrl using a sigmoidal law (Khatri et al., 2025). This law is initialized from Psft(xsft), which is the realized SFT outcome given the SFT compute xsft:

Ppost(xsft, xrl) = (ϵ + Psft(xsft))

Apost − (ϵ + Psft(xsft)) 1 + (xrl/Cmidrl)−Brl

+

(2)

In the equation, the alignment offset ϵ ≥ 0 captures potential discontinuities during the phase transition, providing the flexibility to accommodate empirical misalignments between the SFT endpoint and the global RL scaling trend (see non-zero ϵ fitting cases in Figure 8c). The midpoint parameter Cmidrl measures the RL compute required to realize half of the gain term, Apost − (ϵ + Psft(xsft)), while the scaling steepness Brl controls the rate of convergence. On top of these parameters, this framework allows us to formalize two key metrics as follows:

- Definition 1 (Asymptotic Ceiling). For a given SFT compute allocation xsft, the post-training ceiling Apost defines the theoretical upper bound of performance achievable in the limit of infinite RL computation. Formally, it is expressed as:

lim

xrl→∞

Ppost(xsft,xrl) = Apost(xsft). (3)

- Definition 2 (RL Plasticity). The RL plasticity quantifies the maximum effective headroom available for performance improvement relative to the realized SFT baseline. Formally, it is defined as the difference between the asymptotic post-training ceiling and the corresponding SFT performance:

PLrl(xsft) = Apost(xsft) − Psft(xsft). (4)

These definitions imply that both Apost and PLrl are xsft-dependent quantities, whose values vary with the allocated SFT compute. In practice, Apost is estimated using a robust estimator detailed in Appx. F, with comprehensive fitting results provided in Table 4.

Theoretical Implication. Such framework reveals a fundamental insight: optimizing the SFT solely improves the Psft, but it may shrink the PLrl and thereby constrain Apost if the SFT data is suboptimal (e.g., limited in scale). Thus, it is crucial to understand how Psft, PLrl influence Apost in the post-training.

### 5 Experimental Setup

To determine the optimal mechanism for utilizing expert trajectories, we organize experiments progressively to address three core research questions:

- RQ1: Paradigm Selection. Among Pure-RL, Pure-SFT, Syn-SFT-RL, and Sequential SFT-thenRL, which paradigm establishes the most effective post-training baseline, and what are their characterizations?
- RQ2: Optimal SFT-to-RL Transition. Building upon the optimal paradigm identified in RQ1, what is the optimal time to transit to RL from SFT for a maximum final ceiling?
- RQ3: Data Properties (Scale & Difficulty). With the paradigm (RQ1) and optimal timing strategy (RQ2) established, what roles do data scale and difficulty play in maximizing the performance ceiling, and do they support or refute the “Less is More” hypothesis? 5.1 Models and Data

Models. We primarily use Qwen2.5-7B (Qwen et al., 2025) in § 6.1 and 6.2, and Llama3.23B (Meta AI, 2024) in § 6.3 for cross-validation. In § 6.1, we apply Syn-SFT-RL algorithms to Qwen2.5-Math-7B (Yang et al., 2024) to further examine the influence of model priors.

Training Data. We construct SFT datasets of varying scales and difficulties by curating mathematical trajectories from distilled DeepSeekR1 (DeepSeek-AI, 2025) outputs (Zhao et al., 2025; Tian et al., 2025). The resulting datasets include the large-scale SFT889K with around 889K unique samples, three medium-scale variants controlled for difficulty (Uniform/Easy/Hard102K, refer to Table 3 for the difficulty classification), and a heldout validation set Val-199 with 199 prompt and trajectory pairs. To test data efficiency, we also include S1K-1.1 (Muennighoff et al., 2025) (S1K for short), containing 1K high-quality R1-style trajectories.

For RL in the SFT-then-RL pipeline, we use RL62K, a filtered prompt set from Skywork-OR1RL (He et al., 2025). For Syn-SFT-RL methods, we augment RL62K with expert trajectories in SFT889K to create MIX37K. Refer to Appx. G.1 for details.

Benchmarks. To prevent data leakage, we filter out the benchmark prompts with over 0.8 cosine similarity against our training set using Qwen3-8BEmbedding (Zhang et al., 2025c). We evaluate on the resulting 2,157 unique problems from the fol-

###### Qwen2.5-7B

###### Qwen2.5-7B (SFT 10.8k steps)

###### Qwen2.5-Math-7B

80

80

80

DAPOd Peak: 78.1

SFT Peak: 77.4

SFT Peak: 76.9

78

- 71.5

SFT Peak: 76.9

| |
|---|

70

| |
|---|

Performance

69.8

76

| |
|---|

60

60

Best Pure-RL (DAPOd) Apost: 74.3

74

| |
|---|

Best Sync (LUFFY) Apost: 72.7

50

50

72

0 25 40 60 80 100

1000 1100 1200

0 20 40 60 80 100

Compute (exaFLOPS)

Compute (exaFLOPS)

Compute (exaFLOPS)

SFT DAPOd GRPO UPT LUFFY SRFT

Prefix-RFT

Policy Readapt

| |
|---|

| | |
|---|---|
| | |

- Figure 2: The training dynamic comparision of post-training paradigms under different initializations. (Left) Initializing from Qwen2.5-7B. Early RL-like runs converge quickly (except unstable instances), while early SFT shows a mild performance disruption due to policy readaptation (Zhang et al., 2025b). (Middle) Initializing from a

saturated SFT checkpoint (10,800 steps on Qwen2.5-7B), where SFT-then-DAPOd outperforms other paradigms. (Right) Initializing from Qwen2.5-Math-7B, where UPT and LUFFY demonstrate notable efficiency advantages.

lowing cleaned benchmarks (counts denote original to cleaned): GSM8K (Cobbe et al., 2021) (1319 to 1317), OlympiadBench (He et al., 2024) (675 to 291), Minerva (Lewkowycz et al., 2022) (272 to 262), MATH (Lightman et al., 2023) (500 to 237), and AIME24/25 (LI et al., 2024) (30 to 25). We report the average performance on these unique problems unless otherwise specified. Furthermore, we collect GPQA-Diamond (GPQA-D) (Rein et al., 2023) (198 prompts) for graduate-level science and ARC-C (Clark et al., 2018) (1,172 prompts) for robust open-domain reasoning to evaluate out-ofdomain (OOD) generalization.

#### 5.2 Training and Evaluation

Training. Our experiments include two primary paradigms: (1) Syn-SFT-RL: we implement UPT (Lv et al., 2025), SRFT (Fu et al., 2025) and LUFFY (Yan et al., 2025) using the UPT codebase (Lv et al., 2025). Prefix-RFT (Huang et al., 2025) is implemented using its official codebase. (2) Sequential SFT-then-RL: we first fine-tune the base model on SFT data, then apply RL on the finetuned checkpoints. For comparison in § 6.1, we adopt GRPO and DAPOd (GRPO with dynamic difficulty sampling (Yu et al., 2025)) as the PureRL baseline. When implementing the SFT-then-RL pipeline in 6.2, we use the enhanced DAPOdc in the RL phase, which further incorporates asymmetric ratio clipping into DAPOd. See Appx. C for implementation details.

Evaluation. We report pass@1 accuracy sampled with a temperature of 0.7 and top-p 1.0 to ensure generation diversity. For the smaller AIME24/25 datasets, we use Avg@16 for robust estimation. All

responses are generated with a maximum length of 8,192 tokens.

### 6 Experimental Results

#### 6.1 Paradigms Comparison

To determine the optimal paradigm (RQ1), we systematically benchmark three approaches: PureSFT, Pure-RL (GRPO, DAPOd), Syn-SFT-RL (LUFFY, SRFT, UPT, Prefix-RFT), and the SFTthen-RL pipeline. Following the data filtering setting of Yan et al. (2025), all RL (or Syn-SFT-RL) runs utilize MIX37K, a subset of SFT889K. Our preliminary experiments (Appx. H.1) confirm that training with these methods on the SFT889K leads to the suboptimal convergence. Thus, MIX37K serves as a robust proxy to capture their performance upper bounds. As shown in Figure 2, most of RL or Syn-SFT-RL methods typically saturate or destabilize within a single epoch, demonstrating that MIX37K suffices to capture their performance limits.

Limitations of Syn-SFT-RL. Contrasting prior claims (Yan et al., 2025; Lv et al., 2025; Fu et al., 2025), our experiments reveal severe practical limitations in Syn-SFT-RL methods, which exhibit training instability. For instance, SRFT shows performance fluctuations with a standard deviation 2.6× higher than the stable DAPOd baseline in Figure 2 (Left) and fails to converge stably from a saturated SFT checkpoint (Figure 2 Middle). Furthermore, they are highly sensitive to model priors. For instance, UPT achieves superior convergence on Qwen2.5-Math-7B, outperforming other RL variants (Figure 2 Right). However, this advantage

vanishes on general-purpose Qwen2.5-7B, where it even underperforms vanilla GRPO (Figure 2 Left). RL Variants Trade Ceiling for Efficiency. Pure-RL and stable Syn-SFT-RL methods exhibit a common trade-off: high initial efficiency but a limited ceiling. As shown in the left panel of Figure 2, GRPO, DAPOd, LUFFY, and Prefix-RFT rapidly reach around 71.5 points, outperforming Pure-SFT (69.8) within 25 exaFLOPs. However, their performance quickly plateaus thereafter. The middle panel of Figure 2 shows that even with infinity RL compute, DAPOd improves only by an additional 2.8 points to reach a ceiling of 74.3, while LUFFY gains only 1.5 points to reach

- 72.7. This suggests that without a dedicated SFT phase, the headroom for improvement remains structurally constrained in post-training. SFT Foundation and Sequential RL Maximization. In contrast, Pure-SFT demonstrates “Slow but High” scaling in the middle panel of Figure 2. SFT achieves the continuous improvement through the extensive training and reach a peak of 76.9 points. It significantly surpasses the highest PureRL and Syn-SFT-RL methods’ ceilings, which are

74.3 of DAPOd and 72.7 of LUFFY, respectively. Crucially, transitioning to RL after SFT saturation successfully unlocks further gains. Specifically, SFT-then-DAPOd achieves the best performance with 78.1 points, extending the post-training performance frontier by synergizing the SFT performance with further RL improvement.

Answer to RQ1: Sequential SFT-then-RL is the superior paradigm. Large-scale SFT provides the requisite robust foundation, and subsequent RL leverages this foundation to achieve the final performance frontier.

#### 6.2 SFT-then-RL Pipeline

Given SFT-then-RL’s superiority (§ 6.1), we now examine the two determinants of its final ceiling (Apost): SFT compute allocation (RQ2) and data properties (RQ3). We first derive the optimal SFTto-RL transition timing, then analyze the impact of data scale and difficulty.

6.2.1 The Impact of SFT Compute Allocation Balancing realized SFT performance (Psft) against preserving RL plasticity (PLrl) is crucial for determining the optimal SFT-to-RL transition.

SFT Regimes. To rigorously identify the optimal transition point, we temporally partition the SFT process based on the SFT validation loss L(xsft).

The entire trajectory is segmented based on the following mathematically defined regions:

Tstable = { xsft | L(xsft) ≤ (1 + δ)Lmin }, Tmild = {xsft | (1 + δ)Lmin < L(xsft) < (1 + δ2) · Lmin}, Tsevere = { xsft | L(xsft) ≥ (1 + δ2) · Lmin }.

(5)

where Lmin is the global minimum validation loss observed during training, (δ,δ2) are tolerance thresholds being set as (0.02, 0.1) empirically. Therefore, we have four separable regimes characterizing the SFT convergence:

- • Adaptive Regime (Padapt), where SFT is underfitting in the region.

Padapt = {xsft | xsft < min Tstable} (6)

- • Stable Regime (Pstable), where the validation loss saturates within a small tolerance threshold of 2% (i.e., δ = 0.02).

Pstable = Tstable (7)

- • Mild Overfitting Regime (Pmild), where the region where loss rises slightly but remains below the 10% tolerance, representing the “risky sweet spot.”

Pmild = {xsft | xsft > max Tstable and xsft ∈ Tmild}

(8)

- • Severe Overfitting Regime (Psevere), where loss significantly diverges (≥ 10% rise when δ2 = 0.1), leading to rapid plasticity collapse (see Easy102K in Figure 3).

Psevere = {xsft | xsft > max Tstable and xsft ∈ Tsevere} (9)

The Dynamics of Post-training Ceiling. The blue solid line in Figure 3 illustrates how Apost evolves across these regimes. We observe that initiating RL prematurely during the Adaptive Regime is consistently suboptimal because the model lacks foundational competence that subsequent RL cannot fully recover. For instance, on SFT889K, switching early at 69.8 exaFLOPs yields a ceiling of only 80.7 points, whereas extending training to the Stable Regime (698.4 exaFLOPs) boosts the ceiling to its peak of 84.0 points. Ideally, for most of SFT datasets (e.g., SFT889K, S1K, Uniform102K, Hard102K), the Stable Regime aligns perfectly with peak performance. However, on Easy102K, the peak ceiling shifts into the Mild Overfitting Regime, indicating that a slightly delayed transition is acceptable and can even be beneficial due to Psft improvement.

###### SFT889K

###### S1K

###### Avg. Psft=75.0, PLrl=7.3, Apost=82.3

###### Avg. Psft=71.9, PLrl=4.9, Apost=76.8

100

100

0.80

0.75

90

90

84.0

###### 84.0

83.4

83.1

83.1

83.0

0.70

82.8

82.5

81.0

Performance

80.7

80.3

80.1

###### 77.3

77.0

76.9

76.5

76.4

0.65

80

80

ValLoss

72.0

72.0

0.60

77.7

77.5

77.3

77.1

76.7

76.5

74.1

76.2

70

70

74.1

73.3

73.1

73.0

72.9

0.55

71.9

69.2

70.9

68.6

###### 0.37

0.50

60

60

###### 0.51

0.45

50

50

46.7

46.7

0.40

0 0.6 1.1 1.7 2.3 2.9

0 34.9 69.8104.8139.7174.6349.2523.8698.4873.01047.61222.21365.6

SFT Compute xsft (exaFLOPs)

SFT Compute xsft (exaFLOPs)

###### Easy102K

###### Uniform102K

###### Hard102K

###### Avg. Psft=75.4, PLrl=4.1, Apost=79.5

###### Avg. Psft=73.5, PLrl=4.2, Apost=77.7

###### Avg. Psft=74.5, PLrl=3.7, Apost=78.2

100

100

100

0.9

90

90

90

0.8

###### 82.8

Performance

80.2

###### 79.7

79.3

78.7

###### 78.8

78.5

78.1

77.8

77.8

77.6

77.5

76.9

76.8

76.7

80

80

80

ValLoss

74.4

0.7

72.0

72.0

72.0

76.9

76.2

76.1

70

70

70

75.7

75.5

74.8

0.6

74.7

74.8

74.5

74.4

74.2

73.3

73.3

73.2

70.7

72.2

0.5

60

60

60

###### 0.42

###### 0.40

0.4

50

50

50

###### 0.44

46.7

46.7

46.7

019.839.559.379.098.8335.9

0 34.7 69.3104.0138.7173.4

0 89.3178.5267.8357.0446.3

SFT Compute xsft (exaFLOPs)

SFT Compute xsft (exaFLOPs)

SFT Compute xsft (exaFLOPs)

Apost Psft Val Loss

Psft Adaptive Stable Mild Overfit Severe Overfit

PLrl

| |
|---|

| |
|---|

- Figure 3: SFT-then-RL scaling dynamics across data properties. The post-training ceiling (Apost) decomposes into SFT performance (Psft) and RL plasticity (PLrl), plotted against SFT compute xsft (exaFLOPs). Background colors mark the four SFT regimes (§ 6.2.1), segmented by the SFT validation loss curves. Note: Pure-RL (xsft = 0) and the Easy102K severe-overfitting checkpoint (xsft = 335.9) are excluded from the averaging statistics.

Conversely, aggressively continuing SFT into the Severe Overfitting Regime is detrimental. As demonstrated on Easy102K in Figure 3, training SFT to 335.9 exaFLOPs leads to a rapid decline in the final ceiling due to a collapse in RL plasticity.

Evaluation on OOD Generalization. Figure 4 illustrates the OOD performance of SFT889K checkpoints and their subsequent RL progress. The left panel demonstrates substantial SFT-driven gains: ARC-C improves by 10 points to surpass 90.0 with 104.8 exaFLOPs (1,080 steps), while GPQA-D increases by 30 points to peak at 57.0 after 1047.6 exaFLOPs (10,800 steps). Contrary to claims that intensive SFT degrades generalization (Rajani et al., 2025; Shenfeld et al., 2025), our results prove it establishes a robust OOD reasoning foundation. We attribute this discrepancy to initialization, as prior studies evaluated instruction-tuned models rather than pre-trained base models.

The middle and right panels depict end-to-end performance as RL compute evolves. Transi-

tioning to RL prematurely (SFT-360-RL, Adaptive Regime) yields significant RL-phase gains but suffers from the early convergence, restricting optimal scores to 92.0 on ARC-C and 51.0 on GPQA-D. Conversely, transitioning from a saturated SFT foundation (SFT-10800-RL, Stable Regime) provides marginal RL improvements but unlocks higher ultimate ceilings of 94.0 and 57.0, respectively. Thus, training SFT to saturation remains vital for maximizing OOD generalization.

Answer to RQ2: Train SFT to Saturation. Across both In-Domain (ID) and OOD tasks, the optimal strategy targets the Stable Regime, avoiding Severe Overfitting to preserve RL plasticity. While Mild Overfitting may be tolerable for select datasets, the Stable Regime provides a robust standard for maximizing the performance ceiling with scalable data.

#### 6.2.2 The Impact of SFT Data Properties

Data scale and difficulty are critical determinants of the quality of the SFT prior. We now focus on

###### SFT Scaling

###### 100 RL on ARC-C

###### RL on GPQA-D

100

100

90

94

80

Performance

SFT-0-step

92

SFT-360-step

57

57

SFT-7200-step

51

80

SFT-10800-step

| |
|---|

| |
|---|

| |
|---|

27

25

70

0

0

104.8 698.4 1047.6 SFT Compute (exaFLOPs)

0 10 21.9 30 40 RL Compute (exaFLOPs)

3.0 10 20 30 40 RL Compute (exaFLOPs)

ARC-C GPQA-D SFT-0-RL SFT-360-RL SFT-7200-RL SFT-10800-RL

- Figure 4: The OOD generalization analysis for SFT-then-RL on SFT889K. (Left) The performance dynamics on ARC-C and GPQA-D during SFT. (Middle&Right) The performance dynamics on ARC-C and GPQA-D during RL initiating from different SFT checkpoints.

investigating how these data properties influence the asymptotic post-training performance ceiling. While we focus on representative scales here, extended results across additional data granularities are provided in Appx. H.2.

Larger Scale Begets Higher Ceiling. Comparing datasets of varying scales (S1K, Uniform102K, and SFT889K in Figure 3) reveals that while minimal data can achieve rapid initial SFT gains, extensive data scale is indispensable for reaching a higher post-training ceiling. Initially, small-scale data exhibits deceptive efficiency: S1K achieves an SFT performance of 74.1 points using only 2.3 exaFLOPs, matching the similar performance level that requires 69.3 exaFLOPs on Uniform102K (Psft=74.5) and 174.6 exaFLOPs on SFT889K (Psft=74.1).

However, this efficiency proves to be unsustainable. The realized SFT performance Psft of S1K saturates prematurely at this level. In contrast, Uniform102K and SFT889K continue to improve with additional compute. Uniform102K reaches SFT performances of 74.8 at 104.0 exaFLOPs while SFT889K reaches 77.3 at 698.4 exaFLOPs, which thereby establish the superior foundation for the subsequent RL phase. Crucially, the large-scale SFT also preserves greater RL plasticity. SFT889K maintains an average PLrl of 7.3, exceeding both S1K and Uniform102K by 2.4 and 3.7 points. Consequently, by leveraging both higher realized SFT performance Psft and enhanced RL plasticity PLrl, the large-scale SFT unlocks a significantly higher post-training ceiling.

Harder Data Elevates the Ceiling. Controlling the dataset size as 102K, we investigate how trajectory difficulty affects performance by comparing Easy102K, Uniform102K, and Hard102K. As shown in the bottom row of Figure 3, training on harder trajectories leads to higher returns. Specif-

ically, Hard102K achieves a higher SFT performance (Psft) of 75.4 points in average, outperforming Easy102K (73.5) and Uniform102K (74.6) by 1.9 and 0.8 percentage points, respectively. At the same time, Hard102K maintains a similar RL plasticity (PLrl) to its 102K counterparts, with a score of 4.1 points. This combination, which includes higher SFT performance and decent RL plasticity, makes harder data the superior choice for maximizing the post-training ceiling.

Minimum Validation Loss as an Indicator. Across diverse SFT data configurations, the minimum validation loss shows a strong negative correlation with Apost (Pearson r=-0.92, Figure 8a). Hence, it serves as an efficient a priori indicator: a lower loss reliably predicts a higher post-training ceiling in the SFT-then-RL pipeline, bypassing expensive RL runs.

Answer to RQ3: Scale Dominates, Difficulty Optimizes. Refuting “Less is More”, we establish Data Scale as the primary factor to improve the post-training ceiling, while Difficulty acts as a multiplier. Harder trajectories are helpful when the data scale is limited. Thus, scaling must prioritize volume before difficulty, with the final ceiling reliably indicated by the minimum SFT validation loss.

#### 6.3 The Validation on Llama3.2-3B

We validate our findings on Llama3.2-3B (Meta AI, 2024) to for cross-model generalization. To prioritize practical relevance, we report the maximum achieved post-training performance (maxPpost) instead of the theoretical ceiling (Apost), within 200step RL training.

Llama3.2-3B results strongly corroborate our Qwen2.5-7B findings: (1) Paradigm: As shown in Table 1, SFT-then-RL pipeline remains strictly superior, achieving 68.7 points. It far surpasses Pure-RL or Syn-SFT-RL baselines. (2) Compute Allocation: Consistent with Figure 3, Figure 5 con-

[Figure 2]

##### Figure 5: The analysis of the max post-training per-

formance maxPpost when performing SFT-then-RL on Llama3.2-3B. SFT is performed on SFT889K. The Stable Regime begets a higher maxPpost result.

firms that for smaller models, transitioning to RL from the Stable Regime is critical. Specifically, the Stable Regime achieves a peak maxPpost of 70.4 at xsft=532.5 exaFLOPs, outperforming the Adaptive Regime’s best (65.5 points at xsft=177.5) by 5.9 points. (3) Data Properties: Table 1 confirms that data scale remains the primary determinant of the ceiling (maxPpost ranges from 25.0 on S1K to 70.4 on SFT889K), while difficulty acts as a secondary multiplier (Hard102K leads at 56.4 among 102K variants). Crucially, the minimum SFT validation loss retains its strong relevance with the final reasoning ceiling (Pearson r=-0.96).

Table 1: Llama3.2-3B validation results. We report the maximum post-training performance (maxPpost) and minimum SFT validation loss (Min. Val Loss). Best results are bolded. For all SFT-then-RL experiments, we transition to RL at the checkpoint with Min. Val Loss (details in Appx. C.1).

Methods Paradigm maxPpost Min. Val Loss Llama3.2-3B - 2.1 -

DAPOd Pure RL 3.0 UPT

12.2 LUFFY 8.5 -

Syn-SFT-RL

24.2 0.71 Easy102K 52.4 0.59

S1K

Uniform102K 53.6 0.54 Hard102K 55.6 0.50 SFT889K 67.7 0.44

SFT

SFT889K→ DAPOd SFT-then-RL 68.7 S1K → DAPOdc

25.0 Easy102K → DAPOdc 53.9 -

Uniform102K → DAPOdc 55.3 Hard102K → DAPOdc 56.4 SFT889K → DAPOdc 70.4 -

SFT-then-RL

### 7 Conclusions

This work presents the Plasticity-Ceiling Framework for optimizing expert trajectory utilization, formalizing the trade-off between supervised finetuning performance (Psft) and reinforcement learning plasticity (PLrl). We derive three core principles for effective scaling: (1) The sequential SFT-then-RL pipeline outperforms alternative paradigms in approaching the post-training performance ceiling. (2) Within this pipeline, RL should be initiated at SFT saturation, a point reliably predicted by validation loss minimization. (3) SFT data scale primarily determines the performance ceiling, and trajectory difficulty further optimizes the ceiling when data is limited. Together, these findings transform expert trajectory optimization from empirical guesswork into a systematic and predictable process, establishing a rigorous standard for maximizing mathematical reasoning model performance.

### Limitations

Our findings are limited to mathematical reasoning with R1-style expert trajectories. Whether the Plasticity-Ceiling Framework generalizes to other verifiable tasks (e.g., logic, coding), non-verifiable tasks (e.g., creative writing), or alternative trajectory styles (e.g., Gemini-style, human-annotated) remains an open question for future work.

### Ethical Considerations

All artifacts utilized in this work, including training datasets, evaluation benchmarks, and models, are open-sourced and governed by their respective licenses, such as MIT, Apache-2.0, CC-BY-NC-4.0, CC-BY-SA-4.0, and the Llama 3.2 Community License.

### Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (NSFC) under No. 62576285, Research Center for Industries of the Future (RCIF) at Westlake University, and Westlake Education Foundation.

### References

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, and 21

others. 2016. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. Preprint, arXiv:1603.04467.

Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, and Sergey Levine. 2023. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models. Preprint, arXiv:2311.18232.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. Preprint, arXiv:2305.13245.

Yangyi Chen, Binxuan Huang, Yifan Gao, Zhengyang Wang, Jingfeng Yang, and Heng Ji. 2025. Scaling laws for predicting downstream performance in llms. Preprint, arXiv:2410.08527.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. Preprint, arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, and 181 others. 2025a. Deepseek-v3 technical report. Preprint, arXiv:2412.19437.

DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, and 245 others. 2025b. Deepseek-v3.2: Pushing the frontier of open large language models. Preprint, arXiv:2512.02556.

Volcano Engine. 2023. VERL utils: FLOPs counter (line 149). https://github.com/volcengine/ve rl/blob/59049a66/verl/utils/flops_counter. py#L149. Version 59049a6; Accessed: 2024-12-01.

Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. 2025. SRFT: A single-stage method with supervised and reinforcement fine-tuning for reasoning. arXiv preprint arXiv:2506.19767.

Team GLM, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, and 152 others. 2025. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. Preprint, arXiv:2508.06471.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. Preprint, arXiv:2402.14008.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. 2025. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312.

Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, and Xiang Yue. 2025. Does math reasoning improve general llm capabilities? understanding transferability of llm reasoning. Preprint, arXiv:2507.00432.

Zeyu Huang, Tianhao Cheng, Zihan Qiu, Zili Wang, Yinghui Xu, Edoardo M. Ponti, and Ivan Titov. 2025. Blending supervised and reinforcement fine-tuning with prefix sampling. arXiv preprint arXiv:2507.01679.

P.J. Huber and E.M. Ronchetti. 2011. Robust Statistics.

Wiley Series in Probability and Statistics. Wiley. Hugging Face. 2024. Math-verify. https://github.c

om/huggingface/Math-Verify.

Boris Iglewicz and David C Hoaglin. 1993. How to detect and handle outliers, volume 16. Asqc Quality Press Milwaukee, WI.

Feiyang Kang, Michael Kuchnik, Karthik Padthe, Marin Vlastelica, Ruoxi Jia, Carole-Jean Wu, and Newsha Ardalani. 2025. Quagmires in sft-rl post-training: When high sft scores mislead and what to use instead. Preprint, arXiv:2510.01624.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. Preprint, arXiv:2001.08361.

Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S. Dhillon, David Brandfonbrener, and Rishabh Agarwal. 2025. The art of scaling reinforcement learning compute for llms. Preprint, arXiv:2510.13786.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models. Preprint, arXiv:2206.14858.

Christophe Leys, Christophe Ley, Olivier Klein, Philippe Bernard, and Laurent Licata. 2013. Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. Journal of Experimental Social Psychology, 49(4):764–766.

Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. 2024. Numinamath. [https://github.com/project-numina/aimo

-progress-prize](https://github.com/proje ct-numina/aimo-progress-prize/blob/main/r eport/numina_dataset.pdf).

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. Preprint, arXiv:2305.20050.

Xingtai Lv, Yuxin Zuo, Youbang Sun, Hongyi Liu, Yuntian Wei, Zhekai Chen, Lixuan He, Xuekai Zhu, Kaiyan Zhang, Bingning Wang, Ning Ding, and Bowen Zhou. 2025. Towards a unified view of large language model post-training. Preprint, arXiv:2509.04419.

Meta AI. 2024. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. https:

//ai.meta.com/blog/llama-3-2-connect-202 4-vision-edge-mobile-devices/. Meta AI blog; accessed 2025-04-13; 15 minute read.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. Preprint, arXiv:2501.19393.

Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. 2021. Efficient large-scale language model training on gpu clusters using megatronlm. Preprint, arXiv:2104.04473.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with

human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Neel Rajani, Aryo Pradipta Gema, Seraphina GoldfarbTarrant, and Ivan Titov. 2025. Scalpel vs. hammer: Grpo amplifies existing capabilities, sft replaces them. Preprint, arXiv:2507.10616.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2023. Gpqa: A graduate-level google-proof q&a benchmark. Preprint, arXiv:2311.12022.

Peter J Rousseeuw. 1984. Least median of squares regression. Journal of the American statistical association, 79(388):871–880.

Peter J. Rousseeuw and Katrien Driessen. 2006. Computing lts regression for large data sets. Data Min. Knowl. Discov., 12(1):29–45.

Peter J Rousseeuw and Annick M Leroy. 1987. Robust regression and outlier detection. John Wiley & Sons.

Yangjun Ruan, Chris J. Maddison, and Tatsunori Hashimoto. 2024. Observational scaling laws and the predictability of language model performance. Preprint, arXiv:2405.10938.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. 2025. Rl’s razor: Why online reinforcement learning forgets less. Preprint, arXiv:2509.04259.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Yiyou Sun, Georgia Zhou, Hao Wang, Dacheng Li, Nouha Dziri, and Dawn Song. 2025. Climbing the ladder of reasoning: What llms can-and still can’tsolve after sft? Preprint, arXiv:2504.11741.

Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, and J. Andrew Bagnell. 2025. All roads lead to likelihood: The value of reinforcement learning in fine-tuning. Preprint, arXiv:2503.01067.

Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Yunjie Ji, Han Zhao, and Xiangang Li. 2025. Deepdistill: Enhancing llm reasoning capabilities via large-scale difficulty-graded data training. Preprint, arXiv:2504.17565.

Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. 2024. Dart-math: Difficultyaware rejection tuning for mathematical problemsolving. https://arxiv.org/abs/2407.13690. ArXiv:2407.13690, cs.CL.

Dheeraj Vattikonda, Santhoshi Ravichandran, Emiliano Peñaloza, Hadi Nekoei, Megh Thakkar, Thibault Le Sellier de Chezelles, Nicolas Gontier, Miguel Muñoz-Mármol, Sahar Omidi Shayegan, Stefania Raimondo, Xue Liu, Alexandre Drouin, Laurent Charlin, Alexandre Piché, Alexandre Lacoste, and Massimo Caccia. 2025. How to train your LLM web agent: A statistical diagnosis. arXiv preprint arXiv:2507.04103.

Bo Wang, Qinyuan Cheng, Runyu Peng, Rong Bao, Peiji Li, Qipeng Guo, Linyang Li, Zhiyuan Zeng, Yunhua Zhou, and Xipeng Qiu. 2025. Implicit reward as the bridge: A unified view of sft and dpo connections. Preprint, arXiv:2507.00018.

Hao Wang, Hao Gu, Hongming Piao, Kaixiong Gong, Yuxiao Ye, Xiangyu Yue, Sirui Han, Yike Guo, and Dapeng Wu. 2026. Learning while staying curious: Entropy-preserving supervised fine-tuning via adaptive self-distillation for large reasoning models. Preprint, arXiv:2602.02244.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. 2025. Learning to reason under off-policy guidance. Preprint,

- arXiv:2504.14945.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint,

- arXiv:2505.09388.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. 2024. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. 2025. Limo: Less is more for reasoning. Preprint, arXiv:2502.03387.

Hiroshi Yoshihara, Taiki Yamaguchi, and Yuichi Inoue. 2025. A practical two-stage recipe for mathematical llms: Maximizing accuracy with sft and efficiency with reinforcement learning. Preprint, arXiv:2507.08267.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025. DAPO: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Jia Zhang, Chen-Xi Zhang, Yao Liu, Yi-Xuan Jin, XiaoWen Yang, Bo Zheng, Yi Liu, and Lan-Zhe Guo. 2025a. D3: Diversity, difficulty, and dependabilityaware data selection for sample-efficient llm instruction tuning. Preprint, arXiv:2503.11441.

Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and Jingren Zhou. 2025b. On-policy rl meets off-policy experts: Harmonizing supervised fine-tuning and reinforcement learning via dynamic weighting. Preprint, arXiv:2508.11408.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025c. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Han Zhao, Haotian Wang, Yiping Peng, Sitong Zhao, Xiaoyu Tian, Shuaiting Chen, Yunjie Ji, and Xiangang Li. 2025. 1.4 million open-source distilled reasoning dataset to empower large language model training. Preprint, arXiv:2503.19633.

### A Experimental Platforms

All SFT experiments in § 6.1 run on 16 GPUs; RL and Syn-SFT-RL experiments in § 6.1 are implemented on 8 GPUs, and RL experiments in § 6.2 are conducted on 128 Ascend 910B NPUs.

### B Post-training Paradigms

We summarize the details of algorithmic foundations used in our study: (1) standard supervised fine-tuning (SFT), (2) reinforcement learning (GRPO and DAPO), and (3) synchronized SFT-RL (Syn-SFT-RL) fusion methods used as single-stage baselines.

#### B.1 SFT

SFT tunes the policy πθ via imitation learning using the answer and expert trajectory pair (q,τ) in the SFT dataset DSFT:

JSFT(θ)= − E(q,τ)∼DSFT

|τ| t=1log πθ(τt|q,τ<t)

(10)

This paradigm reliably instills instructionfollowing and basic reasoning capabilities in the model (Ouyang et al., 2022; DeepSeek-AI, 2025).

#### B.2 RL

RL extends the model beyond imitation by optimizing reward-guided exploration. GRPO (Shao

- et al., 2024) and DAPO (Yu et al., 2025) are two widely-used RL algorithms. GRPO replaces a critic with a group-normalized advantage (Ai,t). For each query-answer pair (q,a) in dataset DRL, GRPO samples G response trajec-

tories {τi}Gi=1 based on the old policy πθold. Each trajectory receives a rule-derived reward score Ri. The group-normalized advantage is computed as:

Ri − mean({Rj}Gj=1) std({Rj}Gj=1)

(11)

Ai,t =

With the advantage, GRPO aims to maximize the expected advantage while regularizing the policy towards a reference policy πref via the KL divergence term β · DKL [πθ||πref]. The policy loss JGRPO(θ) is:

JGRPO(θ) = E(q,a)∼D

RL,{τi}Gi=1∼πθold

|τi| t=1 min ritAi,t, CitAi,t

G i=1

(12)

G i=1 |τi|

− β · DKL [πθ||πref]

where rit = ππθ(τi,t|q,τi,<t)

θold(τi,t|q,τi,<t) represents the importance ratio between the new and old policies for token τi,t. Its clipped counterpart, Cit = clip(rit, 1 − ϵ, 1 + ϵ), confines the policy update within a trust region, preventing excessively large and destabilizing policy updates.

DAPO (Yu et al., 2025) further stabilizes training via asymmetric clipping (ϵlow,ϵhigh) and dynamically filter the prompts with all correct or wrong onpolicy generations. We adopt DAPO as our primary RL algorithm due to its robustness on mathematical reasoning tasks.

#### B.3 Syn-SFT-RL

The Syn-SFT-RL paradigm merges SFT and RL by injecting expert trajectories into the optimization loop. We introduce four typical algorithms: LUFFY (Yan et al., 2025), SRFT (Fu et al., 2025), UPT (Lv et al., 2025) and Prefix-RFT (Huang et al., 2025).

LUFFY modifies the JGRPO(θ) in Eq. 12 by jointly optimizing on-policy trajectories and offpolicy ones. It removes both the KL regularization and importance-ratio clipping, and aggregates token-level advantages over a mixture of SFT and RL data. The mixture dataset DMIX contains triplets (q,{τj}Nj=1,a) with the prompt q, N expert trajectories {τj}Nj=1 (N=1 as the official setup), and answer a. Hence, LUFFY’s loss is formalized as:

|τj|

N

1 Z

rˆjt Aˆj,t

JLUFFY(θ) = E(q,{τ

j}Nj=1,a)∼DMIX {τi}Gi=1∼πθold

t=1

j=1

|τi|

G

1 Z

rit Aˆi,t ,

+

t=1

i=1

(13)

where Z = Nj=1 |τj|+ Gi=1 |τi| normalizes over all tokens, and the mixed advantages are computed

without normalization:

Aˆi,t = Ri − mean {Rj}Nj=1 ∪ {Ri}Gi=1 , (14) To avoid entropy collapse on off-policy data, LUFFY further applies regularized importance shaping, which transforms the importance ratio rjt to rˆjt = rjt/(rjt + γ) with a small constant γ = 0.1. SRFT combines four components: (i) the standard SFT loss JSFT in Eq. 10, (ii) the off-policy loss Joff from LUFFY (the first term in Eq. 13), and (iii) on-policy objectives for positive and negative trajectories in Eq. 15. For M on-policy positive

rollouts {τi+}Mi=1 and G − M on-policy negative ones {τj−}Gj=1−M, SRFT maximizes the likelihood of positive trajectories while suppressing that of negative ones:

|τi+|

log πθ τi,t+ | qi,τi,<t+ ,

Jpos(θ) = −E

t=1

|τj−|

log πθ τj,t− | qj,τj,<t− .

Jneg(θ) = E

t=1

(15) The final SRFT objective uses entropy-guided dynamic weights:

JSRFT = w1 JSFT + Joff + w2 Jpos + Jneg,

- w1 = 0.5 · stop_grad e−H(πθ) ,
- w2 = 0.1 · stop_grad eH(πθ) .

(16)

where H(πθ) denotes the policy entropy and stop_grad prevents gradients from flowing through the weights.

UPT employs a hard gating mechanism to mix SFT and RL. Let p denote the average reward over the trajectories sampled for the current prompt q, and γ be a threshold. UPT defines a mixed loss

JUPT = fp JSFT + gp JGRPO, (17) where (fp,gp) are determined by p:

(fp,gp) =

(1,0), p ≤ γ, (0,1), p > γ.

(18)

When the model performs poorly on a prompt (p ≤ γ), the gate prioritizes SFT-style imitation. Once the reward exceeds the threshold (p > γ), the gate switches to pure GRPO optimization to focus on exploration.

Prefix-RFT blends SFT and RL by utilizing sampled expert prefixes to guide online exploration. For a given prompt q and an expert trajectory τ∗, a prefix sequence of length l is sampled. To mitigate position bias and introduce curriculum learning, the prefix length ratio is governed by a cosine decay scheduler whose lower bound gradually decays toward zero. The current policy πθold then generates a continuation τcont to complete the response, yielding a composite hybrid trajectory τhybrid. The model optimizes over a mixture of these hybrid sequences and standard on-policy rollouts using the

GRPO objective:

JPrefix-RFT(θ) = JGRPO θ | {τhybrid} ∪ {τon-policy}

(19)

To prevent off-policy demonstration gradients from dominating the updates, Prefix-RFT incorporates an entropy-based clipping mechanism. This strategy zeroes out the advantages for all but the top-k% high-entropy demonstration tokens, thereby concentrating optimization on critical uncertain junctures and ensuring stable training.

### C Training Configuration

#### C.1 SFT

We train SFT889K and all 102K variants with batch size of 512 and learning rate 1e-5 for 8 and 9 epochs, respectively. To study severe overfitting, we continue training Easy102K up to 6,120 steps (335.9 exaFLOPs). Checkpoints are saved every 360 steps (0.2 epochs for SFT889K, 1.8 epochs for 102K variants). For S1K, we follow the official setup: batch size 16, learning rate 1e-5, weight decay 1e-4, and train for 5 epochs, with checkpoints saved every 62 steps (1 epoch).

Specifically, for the SFT-then-RL experiments in Table 1, we initialize RL from the checkpoint with minimum SFT validation loss, as these checkpoints typically reside in the Stable Regime and are more likely to achieve the post-training ceiling for the given SFT data. Concretely, we use step 124 (2 epochs) for S1K, step 1800 (9 epochs) for all three 102K variants (Easy102K, Uniform102K, Hard102K), and step 14080 (7.8 epochs) for SFT889K.

#### C.2 RL Practice

All RL and Syn-SFT-RL runs optimize a binary correctness reward (1 if correct, 0 otherwise), This correctness is verified using a script powered by Math-Verify (Hugging Face, 2024). Furthermore, token-level loss aggregation is uniformly applied across all runs.

#### C.2.1 RL in Paradigms Comparison

We summarize the RL configuration for the training in § 6.1, including the Pure-RL (GRPO and DAPOd), and Syn-SFT-RL (LUFFY, SRFT, UPT, Prefix-RFT). The shared training hyperparameters for Pure-RL and Syn-SFT-RL methods are summarized in Table 2. Unless specified, all algorithms use this default configuration.

Pure-RL. GRPO and DAPOd serve as Pure-RL baselines. DAPOd adds the dynamic difficulty sampling strategy (Yu et al., 2025) on GRPO. For DAPOd, the dynamic difficulty sampling strategy employs a batch size of 128 responses per inference round, and the asymmetric clipping ratio strategy is not applied.

- Table 2: Shared training hyperparameters for GRPO, DAPOd, LUFFY, SRFT, UPT and Prefix-RFT in § 6.1.

Hyperparameter Default value

Batch size 128 Update batch size 64 Rollout number 8 Max prompt length 1,024

Max response length 8,192

Learning rate 1e−6 Entropy coefficient 0.001

KL loss term removed Std. in group advantage norm removed

Syn-SFT-RL. We adopt training configurations from the Unify-Post-Training codebase (Lv et al., 2025), with UPT utilizing a specific learning rate of 5e-6. For rollout generation, UPT adaptively allocates up to 8 trajectories between on-policy and off-policy samples. Conversely, LUFFY (Yan

- et al., 2025) and SRFT (Fu et al., 2025) maintain a fixed 7:1 ratio of on-policy to expert trajectories per prompt.

Following the similar setting, PrefixRFT (Huang et al., 2025) generates 8 trajectories per prompt, comprising 7 standard on-policy rollouts (empty prefixes) and 1 hybrid rollout guided by a sampled expert prefix. Specifically, the expert prefix length is dynamically sampled via a Beta distribution, where the lower bound follows a cosine decay schedule from 95% to 5% of the demonstration length throughout training, while the upper bound remains constant at 95%. Additionally, Prefix-RFT incorporates an entropy-based clipping mechanism that zeroes out advantages for the 80% lowest-entropy demonstration tokens, concentrating optimization on high-uncertainty regions.

The maximum trajectory length for all SynSFT-RL algorithms is set to 8,192, as consistently adopted in prior works (Yan et al., 2025; Fu et al., 2025; Lv et al., 2025; Huang et al., 2025). Other shared hyperparameters are configured as detailed in Table 2.

#### C.2.2 RL in SFT-then-RL pipeline

Recognizing the superiority of DAPOd when starting from an SFT checkpoint in § 6.1, we further

improve DAPOd to DAPOdc as the RL method in the following SFT-then-RL experiments in § 6.2

and § 6.3. DAPOdc adds the asymmetric clipping ratio strategy to DAPOd, setting (ϵhigh,ϵlow) = (0.28,0.2). In DAPOdc, each inference round uses 128 responses for dynamic difficulty sampling. The batch size and update batch size are 64, the learning rate is 1e-6, and the rollout number is 8. Maximum lengths for prompt and response are 1,024 and 8,192, respectively. Furthermore, the entropy and KL term coefficients are 0, and the group advantage normalization is enabled.

### D Evaluation Configuration

ID Evaluation During evaluation, we find that Math-Verify may fail to parse some semantically correct answers, leading to missed positives. To reduce this bias, we use two complementary evaluators: a strict evaluator based on Math-Verify alone, and a lenient evaluator that additionally allows a fuzzy string match when symbolic verification fails. For instances on which the two evaluators disagree, we further use an LLM judge with the prompt in Table 7 for adjudication. The final reported results are based on this two-stage verification procedure.

OOD Evaluation Manual inspection of model outputs on GPQA-D and ARC-C reveals that models often generate correct answers but fail to adhere to a standardized format. Taking the Student Answer in Table 7 as an example, the answer is semantically correct but not exactly matches the ground truth. This hinders automatic answer parsing and evaluation.

To address this, we use DeepSeekV3 (DeepSeek-AI et al., 2025a) as a judge to verify correctness. We validate this approach by manually auditing 50 random samples and observe that the LLM judgments are perfectly consistent with human evaluation. This confirms the judge’s reliability for these tasks. For the DeepSeek-V3 content generation, we use a temperature of 0.6 and a top-p of 0.95, with a maximum response length of 8,192 tokens. An evaluation example is detailed in Table 7.

### E Compute Estimation

We adopt FLoating-point OPerations (FLOPs) as our computational metric because it is hardware-

- Table 3: Statistical summary of the constructed SFT datasets. It details average prompt/response lengths, logical step counts (derived via semantic analysis in Appx. G.1), and Win Rates (WR) across different DeepSeek model sizes. These metrics confirm the intended difficulty stratification, distinguishing the complexity levels of Easy, Uniform, and Hard subsets. Note: Avg. logical steps are computed on 10K randomly sampled instances, while other metrics are computed on the full dataset.

Dataset Avg. prompt length Avg. response length Avg. logical steps WR (1.5B) WR (7B) WR (671B) Easy102K 64 2253 10 1 0.98 0.99

Uniform102K 74 3673 - 0.79 0.84 0.87 Hard102K 101 8532 13 0.09 0.30 0.41 SFT889K 78 3693 - - - -

S1K 127 9884 - - - Uniform220K 74 3674 - - - Uniform440K 74 3671 - - - -

agnostic and parallelization-agnostic, depending only on model architecture and sequence lengths during training. We employ the FlopsCounter code (Engine, 2023) of the Verl framework (Sheng et al., 2024) for estimation. For SFT, FLOPs are estimated based on the sequence length of the SFT dataset; For RL and Syn-SFT-RL, we dynamically compute FLOPs using real-time prompt and response lengths recorded in TensorBoard (Abadi et al., 2016) logs. During training, both forward and backward cost the computation.

Forward FLOPs Per-Token Estimation. The the theoretical forward FLOPs per token is denoted as Fforward_token, based on the model configuration and average sequence length S. Let L be the number of layers, H the hidden size, Hff the intermediate size of the feed-forward network, and V the vocabulary size. For the attention mechanism, we define DKV as the total dimension of the Key and Value heads, accounting for Grouped Query Attention (GQA) (Ainslie et al., 2023).

First, we define the parameter counts for the constituent dense components. The MLP block, which utilizes a SwiGLU activation function with three linear projections (gate, up, and down), has a parameter count PMLP. The linear projections in the attention layer (comprising WQ,WK,WV ,WO) contribute Pattn_linear. The embedding layer and the language model head share the vocabulary-dim parameters, denoted as Pvocab. These are formulated as:

PMLP = 3HHff Pattn_linear = H(H + 2DKV + H) = 2H(H + DKV ) Pvocab = 2V H

(20)

The total FLOPs consists of the dense computation part (Fdense) and the attention score computation part (Fattn_core). The dense part aggregates

the parameters from all L layers and the vocabulary projections, multiplied by a factor of 2 (for multiply-accumulate operations). The attention core part depends linearly on the sequence length S. The final estimation is given by:

Fdense =2 · [L · (PMLP + Pattn_linear) + Pvocab] Fattn_core = 4 · S · L · H Fforward_token = Fdense + Fattn_core

(21)

Backward FLOPs Per-Token Estimation. According to Narayanan et al. (2021) and Kaplan et al. (2020), the theoretical backward FLOPs per token is approximately two times that of forward. Let Fforward_token be the theoretical backward FLOPs per token:

Fbackward_token = 2 · Fforward_token (22)

#### E.1 SFT Per-step Estimation

Per-step SFT accounts for one forward and one backward pass per step. Let B denote the batch size (number of sequences), S the average sequence length used for fine-tuning.

The total number of tokens processed during SFT per step is given by Ttotal = B · S. Since the backward pass requires approximately twice the FLOPs of the forward pass, the total FLOPs per token during training is 3·Fforward_token. Therefore, the total computational cost for SFT, denoted as FSFT, is calculated as:

Ftrain_token = Fforward_token + Fbackward_token

= 3 · Fforward_token FSFT = B · S · Ftrain_token

(23)

= 3 · B · S · Fforward_token

#### E.2 RL Per-step Estimation

DAPO. For DAPO, the computational cost per step is divided into a Generation Phase (dynamic

sampling) and a Training Phase (actor update). Let Bgen denote the generation batch size, K the number of dynamic sampling iterations, and G the number of responses per prompt (i.e., group size). In the generation phase, the model explores a large solution space by generating K·Bgen·G sequences. Since this phase involves only inference, the cost is purely forward FLOPs.

In the training phase, a subset of data (removing all correct and wrong trajectories) is selected, denoted by the training batch size Btrain (where Btrain < K · Bgen). The update step involves one forward pass to compute new log-probs and one backward pass. Following standard estimation, the combined update cost (forward + backward) is approximately 3 times the forward cost per token (Kaplan et al., 2020).

Given the total sequence length S = Sprompt + Sresponse, the FLOPs for one DAPO step are estimated as:

Fgen = (K · Bgen · G) · S · Fforward_token Ftrain = (Btrain · G) · S · 3 · Fforward_token

(24)

FDAPO = Fgen + Ftrain

=(K · Bgen + 3 · Btrain) · G · S · Fforward_token

GRPO. The algorithm serves as the baseline where no dynamic difficulty sampling is performed. In this setting, the generation batch size equals the training batch size (Bgen = Btrain = B) and sampling is performed once (K = 1). The model generates responses for all prompts in the batch and updates on all of them. Thus, the FLOPs estimation simplifies to:

Fgen = (1 · B · G) · S · Fforward_token Ftrain = (B · G) · S · 3 · Fforward_token

(25)

FGRPO = Fgen + Ftrain

= 4 · B · G · S · Fforward_token

#### E.3 Syn-SFT-RL Per-step Estimation

LUFFY and SRFT. Both LUFFY and SRFT integrate expert demonstrations into the RL optimization loop. Let G denote the number of on-policy sampled trajectories (group size) and N denote the number of expert trajectories per prompt. In the Generation Phase, the model generates G responses for each prompt in the batch B. In the Training Phase, the model updates parameters using both the on-policy generated data and the off-policy expert data. Thus, the effective training batch size per prompt becomes G+N (G = 7,N = 1 in § C.2.1).

Given the real-time recorded average on/offpolicy sequence length Son and Soff, the FLOPs

for LUFFY and SRFT are calculated as the sum of inference cost on G samples and update cost on G + N samples:

Fgen =[(B · G) · Son + (B · N) · Soff] · Fforward_token Ftrain =3[(B · G) · Son + (B · N) · Soff] · Fforward_token

(26)

FHybrid = Fgen + Ftrain

=4[(B · G) · Son + (B · N) · Soff] · Fforward_token

Note that for SRFT, although it computes multiple loss terms (Eq. 16), the dominant computational overhead remains the forward and backward passes through the transformer backbone on the combined data tokens (G + N), making this estimation applicable to both algorithms.

UPT. The per-step FLOPs of UPT are estimated dynamically based on the actual composition of the training batch, which consists of Non on-policy samples processed via GRPO and Noff expert samples processed via SFT.

Let Son be the average on-policy sequence length. The algorithm processes G · Son tokens during the Generation Phase. In this phase, the computational cost is given by:

Fgen = G · Son · Fforward_token (27) Subsequently, the algorithm filters samples based on difficulty, retaining Non on-policy samples and Noff off-policy samples per batch. Consequently, the FLOPs consumption during the Training Phase is:

Ftrain = 3·(Non · Son + Noff · Soff)·Fforward_token (28)

Therefore, the total computational cost for a single UPT step is formulated as:

FUPT = Fgen + Ftrain = [G · Son + 3 · (Non · Son + Noff · Soff)] · Fforward_token

(29)

Prefix-RFT. Since Prefix-RFT only borrows expert trajectory prefixes as a curriculum during RL training, its computational cost is otherwise identical to GRPO. We thus adopt the same FLOPs estimation method as in Eq. 25.

### F Robust Curve Fitting

In § 4, we primarily model the RL scaling dynamics using sigmoidal curves (Ruan et al., 2024; Khatri et al., 2025), which effectively characterize the behavior of most RL and Syn-SFT-RL runs. However, for unstable training instances that defy

sigmoidal fitting, such as the SRFT case shown in Figure 2 (Left), we adopt a robust fallback strategy. Specifically, we define the post-training ceiling for these instances as the peak performance observed either within the allocated RL compute budget or prior to performance collapse.

For the valid scaling instances, simply applying standard regression is often insufficient due to inherent training noise. To accurately model the relationship between computational investment (FLOPs) and model performance, we employ a robust curve-fitting pipeline. This pipeline integrates an iterative outlier detection mechanism based on Modified Z-scores with a Least Trimmed Squares (LTS) regression optimization.

#### F.1 Data Formulation

Let D = {(xi,yi)}Ni=1 denote the dataset, where xi represents cumulative FLOPs and yi represents the evaluation metric. The data is partitioned into a training set Dtrain (first Nfit points) and a held-out validation set Dval. The train-validation split may vary due to differences in training convergence. Across all experimental runs, the training set accounts for an average of 71% of the data, with the remaining 29% used for validation.

#### F.2 Robust Estimation Algorithm

Standard least-squares estimation is highly sensitive to anomalies. To derive a scaling law that reflects the consistent signal rather than transient noise, we employ a hierarchical robust optimization framework that integrates iterative statistical filtering (Modified Z-score) with subset-based optimization (Least Trimmed Squares) to isolate the true performance signal, ensuring that the derived scaling laws are predictive and generalizable across different compute regimes.

Stage-1: Coarse Outlier Rejection (Modified ZScore). First, we filter gross statistical anomalies. In each iteration, we compute residuals ri = yi − f(xi;θ) and the median residual r˜ = median(r). To quantify deviation robustly, we calculate the Median Absolute Deviation (MAD) (Huber and Ronchetti, 2011; Leys et al., 2013):

MAD = median(|ri − r˜|) (30)

Subsequently, the Modified Z-score Mi is computed as (Iglewicz and Hoaglin, 1993):

0.6745 · (ri − median(r)) MAD

Mi =

(31)

Points with |Mi| > τ are removed from the active training split, where τ is the outlier threshold. The factor 0.6745 normalizes the score such that it is consistent with the standard deviation under a normal distribution, while the use of MAD ensures resilience against extreme values that would skew a standard variance calculation.

Stage-2: Least Trimmed Squares Regression. To further refine the model against subsets of data that may distort the global trend, we employ Least Trimmed Squares (LTS) (Rousseeuw, 1984; Rousseeuw and Leroy, 1987) regression. Instead of minimizing the sum of all residuals, LTS regression minimizes only the smallest h squared residuals:

θˆLTS = arg min

θ

h

(r2)(j)(θ) (32)

j=1

where (r2)(1) ≤ ··· ≤ (r2)(Nfit) are the ordered squared residuals over the training set, and h = ⌊Nfit · α⌋ is determined by the parameter α. We define H(k+1) as the set of indices corresponding to the h smallest squared residuals, i.e., H(k+1) = {i | ri2 ≤ (r2)(h)}.

We optimize this objective using the Concentration Step (C-step) algorithm (Rousseeuw and Driessen, 2006), which proceeds iteratively as follows:

- • Estimation: Compute squared residuals ri2 = (yi − f(xi;θ(k)))2 for all Nfit training points using the current parameters θ(k).
- • Selection: Identify the index set H(k+1) corresponding to the h smallest squared residuals.
- • Update: Update parameters to θ(k+1) by fitting the model strictly to the data points indexed by H(k+1).
- • Convergence: Repeat the process until the parameter estimate θ stabilizes.

- F.3 Fitting Results and Analysis.

- F.3.1 Fitting Results

Across all fitting instances, the pipeline achieves an average Root Mean Square Error (RMSE) of approximately 0.86 on the validation split and an average R2 of 0.90 on the training split, confirming the overall robustness of the fits.

- F.3.2 Fitting Analysis

We visualize the resulting SFT-then-RL scaling curves in Figure 8 and provide the exhaustive parameter estimates in Table 4. This table systematically reports the evolution of key scaling metrics across various SFT data configurations relative to the progress of SFT compute xsft. Specifically, we characterize the realized SFT performance (Psft), the RL plasticity (PLrl), the RL convergence midpoint (Cmidrl), the RL scaling steepness (Brl), and the ultimate post-training ceiling (Apost). These results facilitate a detailed analysis of how the SFTed initial point impacts the subsequent RL trend and the final performance limits of the model.

Impact of SFT Compute on RL Plasticity. As illustrated in the PLrl column of Table 4, RL plasticity exhibits an overall downward trend, albeit with minor oscillations, as SFT compute xsft increases. Consequently, allocating minimal compute to SFT generally preserves greater plasticity and yields a larger headroom for subsequent RL-driven improvements. Specifically, across the S1K, Easy102K, Uniform102K, and SFT889K configurations, the lowest SFT compute consistently achieves the maximum PLrl. Although Hard102K presents a minor deviation, its earliest SFT checkpoint still retains the second-highest plasticity in that category.

We attribute this general decline to the inherent imitation nature of SFT. The SFT progress deeply entrenches the policy within the demonstration distribution, prematurely exhausting the exploratory capacity required for effective reward-driven optimization (Wang et al., 2026). This hypothesis is empirically corroborated by the Pure-RL baseline. By entirely bypassing the SFT phase, the Pure-RL instance retains maximum exploratory freedom and achieves the highest absolute plasticity of 25.3, outperforming all SFT-then-RL configurations. However, this plasticity advantage is ultimately offset by its low initial performance (Psft = 46.7), which severely restricts the final post-training ceiling (Apost).

Impact of SFT Compute on RL Training Efficiency. We quantify RL training efficiency using the convergence midpoint Cmidrl, which measures the amount of compute required to achieve half of the total potential gain. Lower values of Cmidrl indicate faster convergence. However, Cmidrl does not monotonically decrease with increasing xsft, instead exhibiting substantial fluctuations.

As shown in Table 4, the most efficient RL runs, those with the lowest Cmidrl, typically correspond to relative less allocated SFT compute xsft. Notably, the Pure-RL baseline achieves the lowest Cmidrl of 1 exaFLOPs, representing the fastest convergence among all evaluated scenarios. This phenomenon is broadly consistent with prior observations regarding the interplay between SFT and subsequent RLHF training (Ouyang et al., 2022). It may be attributed to the fact that relatively lower SFT compute preserves more diverse behaviors in the action space, thereby facilitating exploration and accelerating RL convergence.

### G Dataset Curation

We summarize the key characteristics of our SFT data in Table 3.

#### G.1 Expert Trajectory Collection

We curate high-quality reasoning trajectories from two large-scale datasets: AM-DeepSeekR1-Distilled-1.4M (Zhao et al., 2025) and AMDeepSeek-Distilled-40M (Tian et al., 2025). To ensure data distribution consistency and quality, we retain only mathematics-domain data, select trajectories distilled from DeepSeek-R1671B (DeepSeek-AI, 2025) to unify trajectory style, and perform deduplication based on prompt matching. The resulting filtered datasets are denoted as amthink-1.4m and amthink-40m. From these sources, we construct multiple datasets for our experiments: SFT889K, Uniform102K, Easy102K, and Hard102K for SFT training, and Val-199 for SFT validation. A prompt-trajectory example is shown in Table 8.

Difficulty Classification. To understand the influence of SFT data difficulty on post-training outcomes, we extract data of varying difficulty levels from amthink-40m. We use the Win Rate (WR), defined as the ratio of successful attempts S to total attempts N, as a proxy for problem difficulty. We derive WRs with N=4 attempts across three models: DeepSeek-R1-Distill-1.5B, 7B, and DeepSeek-

- Table 4: Scaling dynamics across SFT-then-RL configurations. The first row represents the Pure-RL baseline. xsft denotes cumulative SFT compute (exaFLOPs). Outlier Threshold is the Modified Z-score cutoff τ for Stage-1 outlier rejection (points with |Mi| > τ are removed; Appx. F.2). Use-LTS denotes whether the Least Trimmed Squares (LTS) regression in Appx. F.2 is used during curve fitting. In § 4.2 we define the alignment offset ϵ, PLrl,

in the RL scaling law (Eq. 2). For PLrl, Brl, Psft, and Apost, higher values are better; for Cmid

Apost, and the midpoint parameter Cmid

rl

, lower values are better. The best values within each SFT configuration are bolded.

rl

SFT data SFT Step xsft Outlier Threshold Use-LTS ϵ PLrl Cmidrl Brl Psft Apost - 0 0 2.5 False 0 25.3 1 1.2 46.7 77.3

62 0.6 2.5 False 4.4 8.7 107 8.5 68.6 77.0 124 1.1 3 True (α =0.85) 0 6.0 44 0.9 70.9 76.9 186 1.7 2.5 False 0 4.0 99 1.1 72.9 76.5 248 2.3 2.5 True (α =0.75) 0 2.5 93 2.8 74.1 76.4 310 2.9 3 True (α =0.75) 1.5 3.3 134 11.2 73.1 75.9

S1K

360 19.8 2.5 False 0 6.0 22 1.0 70.7 76.7 720 39.5 2.5 False 0 5.3 66 0.8 73.3 78.5

1080 59.3 2.5 False 0 4.6 135 0.4 74.2 78.8 1440 79.0 2.5 True (α =0.75) 0 2.0 41 2.0 74.8 76.8 1800 98.8 2.5 False 0 3.4 43 0.3 74.4 77.8

Easy102K

360 34.7 2.5 False 0 4.8 32 1.8 72.2 76.9 720 69.3 3 True (α =0.75) 0 3.0 28 1.4 74.5 77.6

1080 104.0 2.5 False 1.5 4.8 118 6.6 74.8 79.7 1440 138.7 2.5 False 0 1.8 46 2.2 75.7 77.5 1800 173.4 2.5 True (α =0.75) 0 3.8 161 0.7 75.5 79.3

Uniform102K

360 89.3 4 False 0 4.9 40 0.9 73.2 78.1 720 178.5 4 True (α =0.85) 0 3.1 32 1.7 74.7 77.8

1080 267.8 3 True (α =0.85) 0 6.7 61 3.5 76.2 82.8 1440 357.0 2.5 True (α =0.85) 0 3.4 48 5.5 76.9 80.3 1800 446.3 2.5 False 0 2.7 73 1.0 76.1 78.7

Hard102K

360 34.9 2.5 False 0 13.4 10 0.7 69.2 82.5 720 69.8 3 True (α =0.75) 0 8.8 8 1.0 71.9 80.7

1080 104.8 4 True (α =0.75) 0 7.1 10 1.5 73.0 80.1 1440 139.7 4 True (α =0.75) 0 7.0 9 1.7 73.3 80.3 1800 174.6 3 True (α =0.75) 0 6.9 9 1.6 74.1 81.0 3600 349.2 4 False 0 6.6 13 1.6 76.5 83.1 5400 523.8 2.5 False 0 5.3 13 2.2 77.5 82.8 7200 698.4 2.5 True (α =0.75) 0 6.7 16 1.6 77.3 84.0 9000 873.0 2.5 False 0 6.7 13 1.8 76.7 83.4

SFT889K

10800 1047.6 2.5 True (α =0.85) 0 5.4 12 2.5 77.7 83.1 12600 1222.2 3 True (α =0.75) 0 5.9 10 2.2 77.1 83.0 14080 1365.8 2.5 True (α =0.75) 0 7.8 11 1.1 76.2 84.0

R1-671B (DeepSeek-AI, 2025). Based on WRs from the 1.5B model, problems are classified as Easy (WR=1.0) or Hard (WR ≤ 0.25). Consequently, we construct Easy102K and Hard102K, each containing 102K samples from their respective pools, alongside Uniform102K, a randomly sampled neutral-difficulty baseline.

To verify that our WR-based classification captures genuine reasoning complexity rather than mere obscurity, we conduct a supplementary semantic analysis. We randomly sample 10,000 prompts from both Easy102K and Hard102K, employing DeepSeek-V3.2 (DeepSeek-AI et al., 2025b) as an expert evaluator to decompose solution trajectories into distinct logical steps while filtering out conversational redundancy. The evaluator follows the fixed instruction prompt in Table 9. Manual verification of 100 sampled outputs per dataset confirms that this automated step-counting aligns closely with human judgment.

As shown in Table 8, trajectories in Hard102K contain an average of 13 logical steps, representing a 30% increase over the 10 steps averaged in Easy102K. This reasoning depth is directly mirrored in trajectory length: Hard102K responses average 8,532 tokens, nearly four times the 2,253 tokens typical of Easy102K. This concurrent increase in both logical steps and trajectory length empirically validates WR as a robust and reliable proxy for problem difficulty in our study.

#### G.2 RL Data

We curate RL62K (62.3K prompts) from SkyworkOR1-RL (He et al., 2025) by filtering out extreme difficulty levels and prompts containing Chinese characters. For Syn-SFT-RL, we construct MIX37K (36.7K samples) by augmenting these prompts with matched expert trajectories from SFT889K, excluding sequences exceeding 8,192 tokens as suggested by (Yan et al., 2025) to ensure the complete trajectory utilization in each update step. Crucially, this data scale is sufficient, as our experiments show that RL variants typically reach saturation or instability before exhausting a single epoch over MIX37K.

H Complemetary Results

#### H.1 Paradigm Comparsion on SFT889K

We compare various expert trajectory utiliztion paradigms by training on SFT889K. In this setting, we set the batch size to 64 and maximum response

length to 12,000, with RL capped at 400 steps (see Appx. C for details).

As shown in Figure 6 (Left), when starting from the base model, both RL and Syn-SFT-RL methods exhibit premature convergence within 40 exaFLOPs. DAPOd and LUFFY reach a ceiling (Apost) of 71.0 approximately, slightly lower than their MIX37K counterparts (74.3 and 72.7 in Figure 2). However, UPT shows significant scaling gains on this larger dataset (68.6 vs. 60.0), and SRFT demonstrates improved stability compared to MIX37K.

When initialized from a saturated SFT checkpoint (Figure 6, Right), GRPO collapses rapidly, whereas DAPOd scales steadily to 78.1. This not only outperforms the SFT peak (76.9) by 1.2 points but also surpasses all other baselines. These results highlight the necessity of DAPOd’s dynamic difficulty sampling for stability and emphasize the SFT-then-RL pipeline as the superior paradigm for pushing post-training frontiers. H.2 SFT Data Scale Impact

In § 6.2.2, we observe a distinct stratification in the SFT-then-RL post-training performance ceilings across three data scales: S1K, Uniform102K, and SFT889K, suggesting that a larger scale begets a higher post-training ceiling.

To further investigate how the SFT volume dictates the ceiling, we introduce Uniform220K and Uniform440K, SFT datasets comprising 220K and 440K prompt-trajectory pairs uniformly drawn from amthink-40m. We maintain hyperparameter consistency with the Uniform102K setup (see § C.1), and we save checkpoints every 215 steps (0.5 epochs) for Uniform220K and 360 steps (0.4 epochs) for Uniform440K. Using validation loss as a guide, we select three checkpoints across distinct SFT regime for each scale to initialize the subsequent RL stage.

Table 5: The relation between the highest Apost and the minimum validation loss across different SFT data scales. Increasing the training data scale of SFT leads to a lower minimal validation loss, which indicates a higher post-training ceiling Apost.

SFT data Highest Apost Min. Val Loss

S1K 77.3 0.51 Uniform102K 79.7 0.42 Uniform220K 82.6 0.41 Uniform440K 82.8 0.39

SFT889K 84.0 0.37

###### Qwen2.5-7B

###### Qwen2.5-7B (SFT 10.8K Steps)

80

DAPOd Peak: 78.1

80

SFT Peak: 76.9

78

SFT Peak: 76.9

76

71

Performance

| |
|---|

74

| |
|---|

60

Best Sync (LUFFY) Apost: 71.0 Best Pure-RL (DAPOd) Apost: 70.9

71

| |
|---|

50

| |
|---|

| |
|---|

0 20 40 60 80 100

1000 1100 1200

| |
|---|

Compute (exaFLOPS)

Compute (exaFLOPS)

| |
|---|

SFT DAPOd GRPO UPT LUFFY SRFT

Policy Readapt

| |
|---|

- Figure 6: The training dynamic comparision of post-training paradigms on the SFT889K dataset, covering different initializations. (Left) Initializing from Qwen2.5-7B, where the all RL and Syn-SFT-RL methods suffer from the prematurely convergence. (Right) Initializing from a saturated SFT checkpoint (10,800 steps on Qwen2.57B), where DAPOd outperforms other methods, and GRPO suffers from the performance collapse.

0 20.7 145.0 330.6

SFT Compute xsft (exaFLOPs)

50

60

70

80

90

100

Performance

72.0

76.5

71.6

79.0

75.6

82.6

76.8

Uniform220K

0.41

0 34.7 104.0 519.8

SFT Compute xsft (exaFLOPs)

50

60

70

80

90

100

72.0

76.6

70.6

77.4

73.8

82.8

77.0

Uniform440K

0.35

0.40

0.45

0.50

0.55

0.60

0.65

0.70

0.75

ValLoss

0.39

Apost Psft Val Loss

| |
|---|

PLrl

| |
|---|

Psft Adaptive Stable

- Figure 7: SFT compute scaling dynamics of the SFT-then-RL pipeline across different scale of the SFT data. (Left) Performaning SFT on Uniform220K, selecting the 215-step (20.7 exaFLOPs), 1505-step (145.0 exaFLOPs) and 3432-step (330.6 exaFLOPs) SFTed checkpoints to initialize the RL training. (Right) Performing SFT on Uniform440K, selecting the 360-step (34.7 exaFLOPs), 1080-step (104.0 exaFLOPs) and 5400-step (519.8 exaFLOPs) SFTed checkpoints to initialize the RL training.

As Figure 7 illustrates, initializing RL from the Stable Regime consistently yields the optimal performance ceiling on both datasets. This success stems from a synergy between superior SFT performance and preserved RL plasticity, validating the significance of training SFT to saturation. As shown in Table 5, the highest Apost continues to trend upward as data scale increases, confirming that the larger scale, higher ceiling principle remains robust. Furthermore, we confirm that the minimum validation loss still serves as a robust indicator for pursuing a higher Apost. Even as the SFT data scale expands, the minimum validation loss remains strongly correlated with the highest Apost of each specified SFT dataset, maintaining a Pearson correlation coefficient of -0.94, and a

Spearman correlation coefficient of -1.00. H.3 The Optimal SFT-to-RL Transition

Timing on Larger Model

In this section, we validate whether the principle of training the SFT phase to saturation still holds for larger models. We construct a training set of 55K uniformly sampled instances from the amthink40m dataset, denoted as Uniform55K and perform SFT-then-RL experiments on Qwen2.5-14B (Qwen et al., 2025). We follow the SFT-then-RL configuration used for Qwen2.5-7B + SFT889K (detailed in Appx. C.1 and C.2.1), and fine-tune the Qwen2.514B for four epochs. To evaluate the transition timing, we selected SFT checkpoints from various convergence regimes to initialize the subsequent RL phase. The subsequent RL is trained for 200

- Table 6: Qwen2.5-14B on Uniform55K: SFT-to-RL transition points. Columns report cumulative SFT compute xsft (exaFLOPs), validation loss, SFT regime, realized SFT performance (Psft), maximum RL improvement over the SFT checkpoint, and peak post-training performance (maxPpost). Boldface marks the strongest outcome within each column where a single optimum is intended.

SFT–RL switch xsft Val loss SFT regime Psft Max RL improve. max Ppost

0-step (zero-RL) 0 0.67 Adaptive 59.8 14.7 74.5 172-step 32.7 0.44 Adaptive 75.7 3.1 78.8 1720-step 326.6 0.39 Stable 78.6 1.2 79.8

steps.

The established SFT foundation (Psft) and the peak post-training performance (maxPpost) are summarized in Table 6. Consistent with our observations on Qwen2.5-7B in Figure 3 and Llama3.2-

- 3B in Figure 5, the principle of training the SFT foundation to saturation holds firmly at the 14B scale. We observe that transitioning to RL from the 1720-step checkpoint, which is located in the stable regime, achieve the best overall performance

(maxPpost= 79.8). This optimal ceiling is primarily attributed to the robust realized SFT performance of 78.6. In contrast, while initiating RL from a zero-shot foundation results in the highest absolute maximum RL improvement of 14.7 points, its weak initial foundation of 59.8 prevented it from reaching a comparable end-to-end performance.

[Figure 3]

Correlation between the highest Apost and Min. Val Loss

S1K

Pearson r: -0.92 Spearman : -1.00

0.50

0.45

Easy102K

Min.ValLoss

Uniform102K Hard102K

0.40

SFT889K

0.35

0.30

0.25

77 78 79 80 81 82 83 84

Highest Apost

(a) Correlation Analysis (b) SFT889K

[Figure 4]

[Figure 5]

(c) S1K (d) Easy102K

[Figure 6]

[Figure 7]

(e) Uniform102K (f) Hard102K

##### Figure 8: Visualization of SFT-then-RL fitting on Qwen2.5-7B across different SFT data configurations. (a)

Correlation analysis between the highest Apost and the minimum validation loss of each specified SFT dataset. (b)-(f) The SFT-then-RL scaling dynamics under various data configurations. The SFT trajectory is depicted by a black dashed line. RL scaling curves initiated from different SFT steps are distinguished by a color gradient, where lighter shades indicate a higher number of SFT steps. The specific starting SFT step count for each RL curve is annotated in bold black text. Data points from the training split used for fitting the RL scaling curves are marked with solid circles, while those from the validation split used for assessing curve goodness-of-fit are marked with crosses. Magnified views are provided for the low-compute regions of SFT889K and S1K.

System Prompt: You are a teacher grading student assignments. I will provide you with the question, the student’s answer, and the ground truth for the question. Please carefully review each question and grade the student’s answer based on the ground truth. If the student’s answer matches the ground truth, reply with True; otherwise, reply with False. You need not provide any additional information—only reply with True or False. Your grading response must strictly follow the format: ’Grade: ’. Example: Question: 1+1 = _. Please choose the correct option from the following options. A.2, B.3, C.4 Student answer: I think it is A.2 Ground truth: A Grade: True User Prompt: Question: Which change is the best example of a physical change?

1) a cookie baking 2) paper burning 3) ice cream melting 4) a nail rusting Student answer: Okay, so I need to figure out which of these options is a physical change. Let me start by recalling what a physical change is. From what I learned, a physical change is when the form or state of a substance changes, but the chemical composition remains the same. That means no new substances are formed. In contrast, a chemical change involves a chemical reaction where new substances are created. Now let’s look at each option one by one.

- 1) A cookie baking. Hmm, when you bake a cookie, the dough is transformed. The ingredients like flour, eggs, sugar, etc., are mixed and then heated. Baking usually involves chemical reactions, like the Maillard reaction, where new compounds are formed. So this is probably a chemical change.
- 2) Paper burning. Burning paper is definitely a chemical change. Fire requires fuel and oxygen, and when paper burns, it combines with oxygen to form ash, carbon dioxide, and other products. The original paper is no longer there; it’s changed into different substances. So this is chemical.
- 3) Ice cream melting. Ice cream is frozen, and when it melts, it goes from solid to liquid. No chemical reactions are happening here. The molecules are just spreading out because the temperature increased, but the actual molecules of the ice cream (like water, fat, sugar) are the same. So this should be a physical change.
- 4) A nail rusting. Rusting is a chemical process. The iron in the nail reacts with oxygen and water in the air to form iron oxide. This creates a new substance, which is rust. So that’s a chemical change. So the answer should be option 3, ice cream melting. It’s a physical change because the state changes without forming new chemicals. Answer: 3) ice cream melting Ground truth: 3 Grade: DeepSeek-V3 Response: True

- Table 7: The example of using DeepSeek-V3 as a judge when evaluating the case of ARC-C. The student answer is given by the 10800-step SFTed checkpoint from Qwen-2.5-7B.

Question: Let the set A = {1, 2, 3}, and the set B = {0, 1, 2}. Then, the number of elements in A ∪ B is: A: 2; B: 3; C: 4; D: 5 Expert Trajectory: <think> Okay, so I have this problem here about sets A and B. Let me read it again to make sure I understand. It says, “Let the set A be {1, 2, 3}, and the set B be {0, 1, 2}. Then, the number of elements in A ∪ B is ( ). The options are A: 2, B: 3, C: 4, D: 5.” Hmm, union of A and B. Right, union means all the elements that are in either set A or set B or in both. So I need to combine them and remove any duplicates. Let me write down the sets first. Set A is {1, 2, 3}. So those are the numbers 1, 2, and 3. Set B is {0, 1, 2}. That has 0, 1, and 2. Now, if I combine these two sets, I should list all the elements from both, but make sure I don’t count the same element more than once. Let me list them out. Starting with set A: 1, 2, 3. Then set B has 0, 1, 2. So adding those to the union, we already have 1, 2, 3 from A, and from B we have 0 which isn’t in A. The 1 and 2 from B are already in A, so they don’t need to be added again. So combining them, the union should have elements 0, 1, 2, 3. That’s four elements. Wait, but let me check again. Let me list them all out: A ∪ B = {0, 1, 2, 3}. So the elements are 0, 1, 2, 3. Let’s count them: 0 is one, 1 is two, 2 is three, 3 is four. So that’s four elements. So the number of elements is 4. The options are C: 4. So the answer should be C? Wait, but let me make sure I didn’t miss anything. Let me recall the formula for the number of elements in a union: |A ∪ B| = |A| + |B| − |A ∩ B|. So the size of the union is the size of A plus the size of B minusthe size of their intersection. Let’s apply that. Set A has 3 elements: 1, 2, 3. Set B has 3 elements: 0, 1, 2. The intersection of A and B would be the elements common to both, which are 1 and 2. So |A ∩ B| is

2. Then |A ∪ B| = 3 + 3 − 2 = 4. Yes, that matches what I got before. So that’s confirming it. Therefore, the number of elements in A ∪ B is 4. So the answer is C. But wait, just to double-check, maybe I made a mistake in the intersection. Let me verify the intersection. A ∩ B is the set of elements that are in both A and B. A is {1, 2, 3}, B is {0, 1, 2}. So elements common to both are 1 and 2. So yes, two elements. So the formula gives 3 + 3 − 2 = 4. That seems correct. And listing them out gives four elements. So I think that’s right. Alternatively, maybe the answer is D: 5? Wait, maybe I miscounted. Let me list the elements again. If we take all elements from A and B: A has 1,2,3; B has 0,1,2. So combining all of them: 0,1,2,3. So that’s four distinct elements. There’s no 4 or anything else. So four elements. So answer is C:4. So yeah, I think that’s the correct answer. The options are A:2, B:3, C:4, D:5. So the correct one is C. </think> <answer> Given the sets A = {1, 2, 3} and B = {0, 1, 2}, we need to find the number of elements in A ∪ B. First, we recall that the union of two sets A and B includes all elements that are in A, in B, or in both. Listing the elements of both sets and combining them while removing duplicates, we get: - Elements of A: 1, 2, 3 - Elements of B: 0, 1, 2 Combining these elements and removing duplicates, the union A ∪ B is {0, 1, 2, 3}. To verify, we use the formula for the number of elements in the union of two sets:

###### |A ∪ B| = |A| + |B| − |A ∩ B|

where |A| = 3, |B| = 3, and |A ∩ B| = 2 (since the intersection A ∩ B is {1, 2}). Plugging in these values:

###### |A ∪ B| = 3 + 3 − 2 = 4

Thus, the number of elements in A ∪ B is 4, which corresponds to option C. </answer>

|C|
|---|

- Table 8: An example of a R1-style prompt-trajectory pair from SFT889K. The expert trajectory features a reflective thinking process within <think> and </think> tags, followed by a structured solution within <answer> and </answer> tags. The final outcome is boxed.

You are an expert mathematics evaluator. Your task is to analyze the reasoning complexity of a solution trajectory by breaking it down into distinct logical steps. Problem: {question} Thinking Trajectory: {think} Task:

- 1. Analyze the trajectory above. Break down the reasoning process into a numbered list of distinct logical steps.

- • A valid step includes: a mathematical deduction, a calculation, a logical transition, setting up an equation, or verifying a condition.
- • Exclude: Simple restatements of the question, pure filler phrases (e.g., “Let me see”, “Hmm”), or redundant selfcorrections that do not advance the logic.

- 2. Count the total number of valid steps derived in your list.
- 3. Output the final count inside a box format. Output Format: [Your step-by-step analysis here...] Total Logical Steps:

|number|
|---|

- Table 9: Instruction prompt for logical-step segmentation. Placeholders {question} and {think} are filled at inference time with the problem text and the model’s thinking trajectory, respectively.

