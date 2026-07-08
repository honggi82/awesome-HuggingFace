# arXiv:2606.08035v1[cs.CV]6Jun2026

[Figure 1]

## DyCo-RL: Dynamic Cross-Modal Coordination for Visual Reasoning

Hangui Lin2†, Yan Shu1†, Zhengyang Liang3, Chi Liu4, Xiangrui Liu2, Minghao Qin2, Teng Long1, Zheng Liu2, Nicu Sebe1 1University of Trento 2BAAI 3Singapore Management University 4IQuest Research

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a leading paradigm for enhancing visual reasoning in Multimodal Large Language Models (MLLMs). However, existing RLVR methods optimize primarily for the reasoning outcome, fundamentally overlooking the fine-grained cross-modal coordination required during the generation process. Through token-level analyses and controlled interventions, we reveal that during Chain-of-Thought (CoT) reasoning, MLLMs frequently fail to dynamically alternate between extracting visual evidence and synthesizing textual context—a coordination breakdown that is causally linked to reasoning failures. Motivated by these findings, we propose DyCo-RL, which integrates dynamic cross-modal coordination into RLVR optimization. Specifically, DyCo-RL uses the Fisher–Rao geodesic distance to measure within-modality attention shifts, assigning tokens to either visually-oriented or text-oriented functional roles. It then evaluates the alignment between a token’s actual attention allocation and its assigned role, leveraging this score for alignment-guided advantage reweighting during policy optimization. Extensive experiments demonstrate that the algorithm-agnostic DyCo-RL, when applied to Qwen2.5-VL-3B/7B, consistently improves four representative RLVR algorithms across seven benchmarks spanning visual-centric and mathematical reasoning. Our repository is available at https://github.com/Sammy20207109/DyCo-RL

† Equal Contribution

#### 1 Introduction

Visual reasoning [8, 12] requires progressively acquiring, interpreting, and integrating information across visual and textual modalities. Reinforcement Learning with Verifiable Rewards (RLVR) [4, 19, 31, 37], originally developed for LLMs, has recently been extended to MLLMs as a unified framework that jointly improves visual perception and multi-step reasoning through end-to-end reward-driven training [10, 18, 28, 33].

However, visual reasoning is inherently a dynamic process. As the Chain-of-Thought (CoT) unfolds, the model must continuously alternate between faithfully extracting visual evidence at certain steps and strictly anchoring its logic to the preceding textual context at others. When this cross-modal coordination breaks down, distinct failure modes emerge. As illustrated in Fig. 1(a), the model first hallucinates visual features by mistakenly asserting that angle ADE is 80°, and subsequently suffers from logical incoherence by baselessly claiming

[Figure 2]

- Figure 1: An illustrative example of reasoning failures in Qwen2.5-VL-3B optimized via GRPO on ThinkLitehard-11K [26]. (a) The model first hallucinates visual features (mistakenly asserting ∠ADE = 80◦), and subsequently generates logically incoherent text (baselessly claiming ∠CED = ∠ADE, contradicting its own reasoning history). (b) Token-level attention trajectory analysis reveals the root cause: visually-oriented tokens underattend to the image, while text-oriented tokens fail to sufficiently anchor to the preceding textual context.

that angle CED equals angle ADE without valid justification from its own reasoning history. Despite recent advances, existing RLVR methods leave this dynamic coordination problem unaddressed. They either apply uniform policy optimization across all tokens without distinction [19, 37], or focus exclusively on strengthening visual perception [10, 28, 30], thereby neglecting the intricate interplay required for robust multi-step reasoning.

To uncover the underlying mechanism behind these failures, we investigate the rationale generation process through the lens of token-level attention dynamics—specifically, how a token’s attention weights are distributed between visual patches and textual context [6, 7, 35]. As illustrated in Fig. 1(b), we first observe a correlation: in erroneous samples, visuallyoriented tokens systematically under-attend to the image, while text-oriented tokens fail to sufficiently anchor to the preceding textual context. Crucially, we further establish causality through controlled intervention experiments. We demonstrate that dynamically amplifying the appropriate modality attention based on a token’s functional role consistently rectifies these reasoning failures and recovers the correct answers.

Building on these findings, we propose Dynamic Coordination Reinforcement Learning (DyCo-RL) to integrate token-level cross-modal coordination directly into the RLVR framework. DyCo-RL operates in two stages. First, by computing the Fisher–Rao geodesic distance between consecutive within-modality attention distributions, it assigns each token a functional role (e.g., visually- or text-oriented); a larger distance reflects significant attention restructuring, indicating active information extraction from that modality. Second, it evaluates the alignment between the token’s actual attention allocation and its assigned role. This alignment score is then leveraged for alignment-guided advantage reweighting during policy optimization, effectively amplifying the learning signals for well-coordinated tokens while attenuating the influence of misaligned ones. Designed as an algorithm-agnostic plugin, DyCo-RL can be seamlessly incorporated into mainstream RL algorithms. In summary, our main contributions are as follows:

- • We identify cross-modal coordination breakdowns as a critical bottleneck in visual reasoning for MLLMs. Through token-level correlational analysis and causal intervention experiments, we reveal that tokens frequently fail to dynamically shift their focus between modalities during CoT generation.

- • We propose DyCo-RL, a novel approach that embeds dynamic, token-level cross-modal coordination into the RLVR framework. By leveraging the Fisher–Rao geodesic distance of attention shifts to assign functional roles, DyCo-RL dynamically reweights tokenlevel advantages based on modality-specific alignment signals.
- • We demonstrate that DyCo-RL is a highly versatile plug-in. Extensive evaluations on Qwen2.5-VL-3B/7B show that it consistently improves four representative RLVR algorithms (GRPO, DAPO, SAPO, and GSPO) across seven diverse benchmarks spanning both visual-centric and mathematical reasoning. Crucially, ablation studies confirm that these gains stem from improved dynamic coordination rather than naive modality bias.

#### 2 Related Work

RLVR for Visual Reasoning. Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a dominant paradigm for enhancing visual reasoning in MLLMs. Foundational algorithms such as GRPO [19], DAPO [31], GSPO [37], and SAPO [4] optimize policy gradients primarily at the trajectory level, applying uniform credit assignment across all tokens. To explicitly enhance visual perception, PAPO [28] introduces an implicit perception loss via KL divergence between outputs under original and masked visual inputs, encouraging visually grounded reasoning at the loss level. Progressing toward finer granularity, StepGRPO [13] incorporates dense step-level rewards to evaluate the accuracy and logical consistency of intermediate reasoning steps. Finally, at the token level, methods like TokenDPO [32] and Critical Tokens [15] have demonstrated the efficacy of fine-grained optimization for preference alignment in LLMs. In the multimodal domain, VPPO [10] filters policy gradients to emphasize highly visually-dependent tokens, and AT-RL [11] selectively reinforces anchor tokens exhibiting strong cross-modal connectivity via graph-based attention clustering. Despite these advances, existing token-level MLLM approaches assign credit based on a single-modality prior. They overlook the dynamic nature of CoT reasoning, where different tokens must alternate their focus between visual and textual contexts. Our work bridges this critical gap by jointly coordinating both visual and textual modality demands during token-level policy optimization.

Cross-Modal Attention Analysis in MLLMs. Attention patterns serve as a crucial window into the cross-modal interactions of MLLMs. One line of research leverages attention for inference-time interventions; for instance, attention refocusing and visual-aware decoding methods [24, 35] dynamically reweight attention distributions during generation to mitigate hallucinations and improve visual faithfulness. Another line of research investigates attention from an interpretability perspective. Recent studies [1, 14, 23, 27] demonstrate that distinct attention dynamics correlate strongly with specific functional behaviors, suggesting that the evolution of attention states intrinsically reflects the model’s underlying reasoning process. Crucially, however, these approaches utilize attention solely as an inference-time heuristic or a post-hoc diagnostic tool. In contrast, DyCo-RL is the first to fundamentally integrate dynamic attention coordination directly into the RLVR training objective.

#### 3 Preliminaries

##### 3.1 Modality-Specific Attention Decomposition

In autoregressive MLLMs, the language model decoder attends to both visual and textual tokens at each generation step. Let the context be partitioned into a set of visual token indices Ivis and a set of textual token indices Itxt (comprising the input prompt and all previously generated tokens). At the last decoder layer, the attention weight from a generated token at position i to a preceding context token at position j, averaged over all attention heads, is computed as:

exp(eijh) ∑k exp(eikh )

qih · (khj )⊤ √d

1

|H| ∑

, eijh =

(1)

ai,j =

h∈H

where qih and khj are query and key vectors for head h, and d is the head dimension. For a generated token at step t, we define its modality-specific attention score to modality m ∈ {vis,txt} as:

rtm = ∑

at,j (2)

j∈Im

which quantifies the total attention density allocated to each modality during generation.

##### 3.2 Group Relative Policy Optimization (GRPO)

Given an input prompt x, the reference policy πθold generates a group of G candidate responses {o1, . . . , oG}. Under the RLVR framework, each response receives an outcome-based reward Ri determined by whether its final answer matches the ground truth. To reduce variance and eliminate the need for a separate value network, GRPO computes a groupnormalized advantage for each response:

Ri − µ(R1:G) σ(R1:G)

Aˆi =

(3)

where µ(R1:G) and σ(R1:G) are the mean and standard deviation of the rewards within the group. The policy πθ is then optimized via a clipped surrogate objective, where the scalar advantage Aˆi is uniformly applied across all token positions within a response:

|oi|

G

1 |oi|

1 G

min ρi,t Aˆi,clip ρi,t,1−ϵ,1+ϵ A ˆi (4)

### ∑

### ∑

LGRPO(θ) = E

t=1

i=1

where ρi,t = πθ(oi,t|x, oi,<t) / πθold(oi,t|x, oi,<t) denotes the importance sampling ratio at step t.

This formulation introduces a fundamental limitation for MLLMs: a single scalar advantage is broadcast uniformly to all tokens. Consequently, it provides identical learning signals to tokens whether they function to extract visual evidence or synthesize textual logic, fundamentally ignoring the fine-grained, dynamic modality coordination required for robust visual reasoning.

[Figure 3]

- Figure 2: (Left) Correlation analysis of modality-specific attention. P/R denote visually/text-oriented tokens, and superscripts r/e indicate correct/erroneous states. (Right) Causal intervention on erroneous tokens. The green (yellow) curve shows the effect of amplifying visual (textual) attention for Pe (Re). Both targeted interventions yield a consistent recovery rate under moderate enhancement.

#### 4 Method

We first analyze the cause of cross-modal coordination breakdowns in visual reasoning (Section 4.1). Building on these findings, we propose Dynamic Coordination Reinforcement Learning (DyCo-RL), which operates in two stages: (i) assigning tokens to visually- or textoriented roles via the Fisher–Rao geodesic distance between consecutive within-modality attention distributions (Section 4.2.1); and (ii) evaluating the alignment between actual attention allocation and assigned roles to perform alignment-guided advantage reweighting during policy optimization (Section 4.2.2).

##### 4.1 Diagnosing Cross-Modal Coordination Breakdowns

We examine cross-modal coordination breakdowns at token-level granularity from two complementary perspectives: correlation analysis, which examines whether attention patterns systematically differ between correct and erroneous tokens, and causal intervention, which tests whether correcting attention misalignment recovers model performance.

Correlation Analysis. We collect 200 erroneous rollouts from MathVerse [36] and MathVision [25] which generated by Qwen2.5-VL-3B optimized via GRPO on ThinkLite-hard-11K. Each generated token span is categorized according to whether it primarily performs imagegrounded perception or text-based reasoning within the local decoding context. Specifically, we distinguish visually-oriented tokens (P), which extract or describe information from the image, from text-oriented tokens (R), which conduct logical inference based on the preceding textual context. Each category is further divided by correctness, yielding four disjoint groups: Pr (correct), Pe (erroneous), Rr (correct), and Re (erroneous). Labels are assigned by annotators at the semantic-span level and then projected back to tokens. Full annotation details are provided in Appendix F.

For each group S ∈ {Pr, Pe, Rr, Re}, we average the modality-specific attention score rtm (Eq. 2) over all tokens in the group. As shown in Fig. 2, correct visually-oriented tokens (Pr)

allocate significantly more attention to the image than erroneous ones (Pe). Symmetrically, correct text-oriented tokens (Rr) attend more to the preceding context than erroneous ones

(Re). These results reveal a clear association between modality-specific attention allocation and token-level correctness.

Causal Intervention. To test whether the observed attention misalignment causally contributes to reasoning errors, rather than merely correlating with them, we design a controlled attention enhancement experiment. For tokens in Pe or Re, we selectively amplify their attention toward the modality their functional role demands: visual inputs for visuallyoriented erroneous tokens (M = Ivis) and textual context for text-oriented erroneous tokens (M = Itxt). Specifically, we dynamically modify the aggregated attention weight from the generated token at step t to a preceding context token at position j as follows:

at,j · 1 + λ · 1j∈M ∑k at,k · 1 + λ · 1k∈M

a˜t,j =

(5)

where at,j is the original attention weight averaged across all heads (as defined in Eq. 1), 1j∈M is the indicator function for the target modality, and λ > 0 controls the enhancement strength.

As shown in Fig. 2, moderate enhancement consistently recovers performance on previously erroneous samples for both functional roles, with visually-oriented tokens showing higher sensitivity to correction. Excessively large λ, however, degrades performance, indicating that over-amplification disrupts the original attention balance. Together with the correlation analysis, these results provide convergent evidence that cross-modal attention misalignment is a significant contributing factor to visual reasoning errors.

- 4.2 Dynamic Coordination Reinforcement Learning Based on the findings above, we propose DyCo-RL, whose pipeline is illustrated in Fig. 3.

###### 4.2.1 Token Role Assignment

To identify each token’s functional role, we characterize the transition of within-modality attention between consecutive generation steps. A token actively engaging a modality will exhibit substantial redistribution of attention within that modality compared to the previous step.

First, we normalize the attention weights within each modality to obtain a conditional distribution:

at,j rtm

pmt,j =

###### , ∀j ∈ Im (6)

which describes how attention is distributed across context tokens within modality m, independent of the total amount allocated to that modality.

Next, we measure the structural change between consecutive within-modality distributions using the Fisher–Rao geodesic distance:

vmt = 2arccos ∑

j∈Im

pmt−1,j pmt,j (7)

[Figure 4]

- Figure 3: Overall Pipeline of DyCo-RL. Instead of directly broadcasting a standard sequence-level advantage (Aˆi) to all tokens, our framework transforms it into a fine-grained Dynamic Cross-Modal Coordination Advan-

tage (A˜i,t) for policy updates. Specifically, our dynamic coordination plugin computes the Fisher-Rao attention distance to assign functional roles to individual tokens. By evaluating the alignment between a token’s actual attention allocation and its assigned role, we dynamically reweight the standard advantage at the token level.

A high vmt indicates substantial structural reallocation of attention within modality m from step t−1 to t, reflecting active information acquisition, whereas vmt ≈ 0 indicates inertial behavior. Unlike KL divergence, the Fisher–Rao distance is symmetric and bounded, providing a stable measure for noisy attention dynamics.

Finally, we assign each token to the modality whose attention distribution undergoes stronger structural reconfiguration:

Dm = t | vmt − vmt¯ > τ , {m, m¯ } = {vis,txt} (8)

where τ is a stability margin that suppresses ambiguous transitions, and m¯ denotes the complementary modality. Tokens falling outside both sets (Dvis and Dtxt) are treated as neutral tokens where neither modality exhibits dominant behavior.

###### 4.2.2 Alignment-Guided Advantage Reweighting

Building on the token role assignment in Section 4.2.1, we translate modality alignment into fine-grained RL optimization guidance.

To derive token-level operations, we temporarily drop the candidate response index i. For the t-th token in a generated sequence, we first measure how well its actual attention matches its assigned functional role using the relatively normalized modality attention score r˜tm = rtm/(rtvis + rttxt).

st = r˜tvis · 1t∈Dvis + r˜ttxt · 1t∈Dtxt (9)

Consequently, visually- and text-oriented tokens are rewarded for attending to the image and textual context, respectively, while neutral tokens remain unbiased.

Next, instead of broadcasting a single scalar advantage to every token, we reweight it dynamically using the computed alignment score:

A˜t = (1 + α · wt) · Aˆ, wt = |o| · exp(st) ∑|ko=|1 exp(sk)

, (10)

where Aˆ is the standard sequence-level advantage, α > 0 controls the reweighting strength, and the scaling factor |o| preserves the overall advantage magnitude after softmax normalization.

Finally, generalizing this token-level formulation to the entire candidate group, we integrate the dynamically reweighted advantage A˜i,t (for the t-th token in the i-th response) into the standard policy gradient objective. Using GRPO as the base algorithm, the final DyCo-RL objective is:

LDyCo-RL(θ) = E

|oi|

G

1 G

1 |oi|

min ρi,t A˜i,t,clip ρi,t,1−ϵ,1+ϵ A ˜i,t (11)

### ∑

### ∑

t=1

i=1

#### 5 Experiments

##### 5.1 Experimental Settings

Models, Data, and Baselines. We integrate DyCo-RL into Qwen2.5-VL-3B and Qwen2.5-VL7B, utilizing the ThinkLite-hard-11K dataset [26] for training. To demonstrate its versatility, we evaluate DyCo-RL on top of four representative RLVR algorithms spanning diverse optimization paradigms: GRPO [19], DAPO [31], GSPO [37], and SAPO [4]. Brief descriptions of each baseline algorithm are provided in Appendix D.

Training Details. Models are trained with a learning rate of 1e-6, a rollout batch size of 4, a maximum response length of 2048 tokens, and a global batch size of 64. For our DyCoRL plugin, we set the alignment-guided reweighting strength α = 0.2 and the token role assignment stability margin τ = 0.05. For fair comparison, we strictly adhere to the official hyperparameters for all RLVR baselines. Comprehensive training configurations are detailed in Appendix E.

Evaluation Benchmarks. We evaluate on seven benchmarks spanning two primary domains: (1) Mathematical Reasoning: WeMath [17], MathVision [25], and MathVerse [36]; and (2) VisualCentric Reasoning: LogicVista [29], HallusionBench [5], MME [3], and MMBench [16]. We follow the standard evaluation protocol from VLMEvalKit [2]. All methods are evaluated using top-1 accuracy (Acc@1) at a decoding temperature of 0.1.

##### 5.2 Main Results

Table 1 presents the evaluation results on seven benchmarks across two model scales (Qwen2.5-VL-3B and 7B) and four RLVR algorithms. DyCo-RL consistently enhances baseline performance across the vast majority of benchmarks, confirming that our alignment-

Table 1: Main Results across four RLVR algorithms and two model scales. Bold formatting indicates the better performance between each baseline algorithm and its DyCo-RL enhanced counterpart.

Mathematical Reasoning Visual-Centric Reasoning WeMath MathVision MathVerse LogicVista HallusionBench MME MMBench Avg. Qwen2.5-VL-3B

Model

Zero-Shot 20.0 23.0 33.8 34.5 61.3 76.9 62.8 44.6 GRPO 23.0 21.4 34.4 39.8 60.8 79.9 54.8 44.9 GRPO + DyCo-RL 25.9 22.4 36.0 41.0 62.4 80.7 58.2 46.7 DAPO 27.4 17.1 34.0 39.8 60.7 81.0 51.7 44.5 DAPO + DyCo-RL 26.9 22.8 36.6 41.4 62.7 82.1 54.8 46.8 SAPO 31.5 20.1 35.0 39.4 59.7 82.1 53.0 45.8 SAPO + DyCo-RL 32.0 19.7 35.3 44.5 63.9 81.1 54.8 47.3 GSPO 27.2 19.7 34.0 35.8 62.4 79.6 54.9 44.8 GSPO + DyCo-RL 26.0 24.0 34.1 37.4 62.0 81.9 56.3 46.0

Qwen2.5-VL-7B

Zero-Shot 33.2 23.7 43.8 44.5 65.0 82.4 82.9 53.6 GRPO 38.7 26.6 47.4 46.2 69.2 83.3 78.0 55.6 GRPO + DyCo-RL 40.5 29.3 49.5 48.5 71.6 84.9 83.8 58.3 DAPO 36.4 23.0 44.5 42.3 70.0 86.3 62.7 52.2 DAPO + DyCo-RL 37.5 24.4 45.2 43.8 71.0 86.0 75.8 54.8 SAPO 38.5 26.6 47.0 42.3 71.0 85.6 74.1 55.0 SAPO + DyCo-RL 40.7 24.6 47.8 47.7 69.2 86.0 74.6 55.8 GSPO 39.4 26.6 42.4 46.5 71.4 81.6 65.8 53.4 GSPO + DyCo-RL 39.9 27.6 46.5 48.3 73.2 84.8 75.5 56.5

guided advantage reweighting serves as a highly effective, plug-and-play module for existing RLVR pipelines.

Algorithm-Agnostic Generality. DyCo-RL yields consistent gains across four structurally diverse optimization algorithms without requiring algorithm-specific hyperparameter tuning. Rather than marginal fluctuations, it improves the average performance across all baselines (GRPO, DAPO, SAPO, and GSPO), with individual benchmark gains reaching up to +5.7. This seamless transferability across clip-based, sequence-level, and soft-gated objectives confirms that DyCo-RL addresses a fundamental limitation shared by all representative RLVR paradigms.

Consistent Gains Across Domains. The performance boosts span both mathematical and visual-centric reasoning. Crucially, DyCo-RL consistently strengthens evidence grounding on visual tasks (e.g., HallusionBench, MME) while preserving and often enhancing chainof-thought coherence on complex mathematical benchmarks (e.g., MathVision, MathVerse). This dual improvement firmly supports our central claim: dynamically reweighting advantages based on token roles sharpens visual perception without sacrificing reasoning logic.

Scalability to Larger Models. Beyond algorithmic and task-level generality, the improvements persist on the 7B model, where baselines are inherently stronger and the performance headroom is narrower. DyCo-RL continues to elevate average performance across the board, yielding striking individual surges (e.g., up to +13.1 on MMBench). This robust scaling behavior emphatically indicates that even larger models with advanced capacities still suffer from cross-modal coordination breakdowns, which our method effectively mitigates.

Table 2: Effectiveness of Fisher–Rao Role Assignment compared to alternative assignment strategies.

Method WeMath MathVision MathVerse LogicVista HallusionBench MME MMBench Avg. GRPO (Baseline) 23.0 21.4 34.4 39.8 60.8 79.9 54.8 44.9

+ Random 23.5 20.1 33.1 36.7 59.7 79.3 54.4 43.8 + Reverse 24.2 19.4 34.0 36.0 58.6 78.0 52.6 43.3 + Entropy 25.0 21.0 34.4 40.2 61.2 81.1 56.5 45.7 + KL Divergence 24.7 21.3 35.4 40.6 62.5 80.4 57.3 46.1 + DyCo-RL (Ours) 25.9 22.4 36.0 41.0 62.4 80.7 58.2 46.7

Table 3: Effectiveness of Alignment-Guided Reweighting.

Method WeMath MathVision MathVerse LogicVista HallusionBench MME MMBench Avg. GRPO (Baseline) 23.0 21.4 34.4 39.8 60.8 79.9 54.8 44.9

+ Visual Attn Score (r˜tvis) 24.5 23.0 32.1 35.4 62.0 78.1 56.3 44.5 + Text Attn Score (r˜ttxt) 26.0 22.4 35.4 37.8 59.4 78.4 53.0 44.6 + DyCo-RL (Ours) 25.9 22.4 36.0 41.0 62.4 80.7 58.2 46.7

##### 5.3 Ablation Studies

To systematically isolate the performance gains introduced by DyCo-RL, we independently ablate its two core modules: the token role assignment strategy and the alignment-guided reweighting mechanism. All ablation experiments are conducted on the Qwen2.5-VL-3B model using the GRPO baseline. Extended ablation studies and details are detailed in Appendix G.

Effectiveness of Fisher–Rao Role Assignment. We compare DyCo-RL against four alternative token role assignment strategies, structured across progressive levels of complexity (Table 2). First, for basic sanity and directional checks, we evaluate Random (arbitrary reweighting) and Reverse (inverting our assignment logic). Random yields only marginal, unstable gains over GRPO, confirming that unprincipled reweighting merely injects noise. Conversely, Reverse causes the most severe degradation across all variants. This inverse penalty provides strong directional proof that our Fisher–Rao metric accurately identifies the correct functional roles.

Next, we examine two competitive heuristic baselines: Entropy (a static measure) and KL (a dynamic measure). Entropy measures the sharpness of the attention distribution at a single step, but it falls short because it inherently conflates predictive uncertainty with cross-modal roles, capturing only a static snapshot. To capture temporal shifts, KL uses the KL divergence between consecutive attention distributions. While more competitive, KL divergence is mathematically asymmetric and magnitude-sensitive, failing to capture the true geometric structure of the probability simplex during cross-modal transitions.

Ultimately, the full DyCo-RL method consistently achieves the best performance. By utilizing the Fisher–Rao distance, it provides a symmetric, geometrically principled measure that successfully overcomes the limitations of both static snapshots (Entropy) and asymmetric divergence (KL), correctly aligning each token’s learning signal with its true modality role.

Effectiveness of Alignment-Guided Reweighting. To validate the conditional formulation of our alignment score st in Eq. 9, we replace it with two modality-biased alternatives while keeping all other components identical. Specifically, rather than dynamically switching the

[Figure 5]

- Figure 4: Effectiveness of DyCo-RL in dynamic cross-modal coordination. (a) Alignment between assigned token roles and actual modality attention ratios; (b) Distribution of visually-oriented tokens across the generation process; (c) Distribution of text-oriented tokens across the generation process.

scoring function based on a token’s assigned role (Dvis or Dtxt), we forcefully apply a single modality’s attention ratio to all tokens: (i) + visual attention score (setting st = r˜tvis universally), and (ii) + text attention score (setting st = r˜ttxt universally). As shown in Table 3, this unidimensional reweighting exhibits a clear trade-off: universally up-weighting visual attention improves visually intensive tasks but severely degrades textual reasoning, while the textual variant shows the exact opposite pattern. This confirms that merely encouraging a model to attend to a specific modality is insufficient; coordination is key. In contrast, while singlemodality scores occasionally peak on biased tasks, DyCo-RL achieves the most robust and highest average performance across all benchmarks by dynamically aligning each token’s advantage with its assigned functional role.

#### 6 Discussion and Analysis

Beyond empirical gains, understanding why DyCo-RL succeeds requires a deeper look into the model’s internal representations. To this end, we sample 200 generated instances from both the vanilla GRPO baseline and GRPO+DyCo-RL in MathVerse and MathVision. Because the two models generate distinct reasoning trajectories, we independently annotate their tokens into visually-oriented and text-oriented roles following the diagnostic protocol detailed in Section 4.1. Based on this curated evaluation set, we investigate how our alignment-guided reweighting fundamentally reshapes cross-modal reasoning dynamics.

Strengthening Role-Aligned Attention. We first evaluate the internal attention allocations across the annotated tokens. As illustrated in Fig. 4(a), DyCo-RL successfully strengthens the alignment between a token’s functional role and its actual modality attention. Specifically, for visually-oriented tokens, the model noticeably increases its attention toward images while suppressing text attention. Conversely, for text-oriented tokens, it actively amplifies text attention at the expense of visual focus. This consistent trend confirms that our advantage reweighting directly encourages tokens to actively extract information from their designated functional modalities.

Reshaping Temporal Dynamics. Furthermore, Fig. 4(b-c) reveals how DyCo-RL reshapes temporal reasoning dynamics. The baseline is rigidly phase-locked: visual perception is front-loaded, while textual reasoning dominates later. DyCo-RL relaxes this boundary. Visually-oriented tokens maintain a sustained presence through the middle stages (e.g., 0.4– 0.6) for continuous visual re-grounding, while text-oriented tokens activate earlier and distribute more evenly. This interleaved distribution overcomes the strict “perceive first, reason later” bottleneck, enabling dynamic cross-modal coordination throughout CoT generation.

#### 7 Conclusion

We identify cross-modal coordination breakdowns as a primary bottleneck in visual RLVR, where visually- and text-oriented tokens systematically under-attend to their designated modalities. To address this, we propose DyCo-RL, an algorithm-agnostic plugin that assigns functional token roles via Fisher–Rao attention distance and dynamically reweights advantages based on role-attention alignment. Extensive evaluations show that DyCo-RL consistently improves four representative baselines across seven diverse benchmarks, establishing explicit token-level coordination as an effective path toward faithful multimodal reasoning.

#### Limitations

Our work has two main limitations. First, although DyCo-RL introduces no additional trainable parameters, computing per-token attention statistics and Fisher–Rao distances during rollout incurs extra training time and GPU memory overhead compared to vanilla RLVR algorithms. Further engineering optimization is needed to reduce this cost, particularly for longer rollouts. Second, due to computational resource constraints, all experiments are conducted on Qwen2.5-VL-3B and 7B models. While we observe consistent improvements across both scales, it remains to be verified whether the proposed mechanism generalizes effectively to substantially larger MLLMs, where attention dynamics may exhibit qualitatively different behavior.

#### References

- [1] Enrique Queipo de Llano, Álvaro Arroyo, Federico Barbero, Xiaowen Dong, Michael Bronstein, Yann LeCun, and Ravid Shwartz-Ziv. Attention sinks and compression valleys in llms are two sides of the same coin, 2026. URL https://arxiv.org/abs/2510.06477.
- [2] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024.
- [3] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [4] Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, and Junyang Lin. Soft adaptive policy optimization, 2025. URL https://arxiv.org/abs/2511.20347.
- [5] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, June 2024.
- [6] Meng-Hao Guo, Tian-Xing Xu, Jiang-Jiang Liu, Zheng-Ning Liu, Peng-Tao Jiang, Tai-Jiang Mu, Song-Hai Zhang, Ralph R. Martin, Ming-Ming Cheng, and Shi-Min Hu. Attention mechanisms in computer vision: A survey. Computational Visual Media, 8(3):331–368, 2022. doi: 10.1007/s41095-022-0271-y.
- [7] Mohammed Hassanin, Saeed Anwar, Ibrahim Radwan, Fahad Shahbaz Khan, and Ajmal Mian. Visual attention methods in deep learning: An in-depth survey. Information Fusion, 108:102417, 2024. ISSN 1566-

2535. doi: https://doi.org/10.1016/j.inffus.2024.102417. URL https://www.sciencedirect.com/science/ article/pii/S1566253524001957.

- [8] Feijuan He, Yaxian Wang, Xianglin Miao, and Xia Sun. Interpretable visual reasoning: A survey. Image and Vision Computing, 112:104194, 2021. ISSN 0262-8856. doi: https://doi.org/10.1016/j.imavis.2021.104194. URL https://www.sciencedirect.com/science/article/pii/S0262885621000998.
- [9] Wei He, Zhiheng Xi, Wanxu Zhao, Xiaoran Fan, Yiwen Ding, Zifei Shan, Tao Gui, Qi Zhang, and XuanJing Huang. Distill visual chart reasoning ability from llms to mllms. In Findings of the Association for Computational Linguistics: EMNLP 2025, 2025.

- [10] Siyuan Huang, Xiaoye Qu, Yafu Li, Yun Luo, Zefeng He, Daizong Liu, and Yu Cheng. Spotlight on token perception for multimodal reinforcement learning. arXiv preprint arXiv:2510.09285, 2025.
- [11] Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Wei Wang, Bing Zhao, Hu Wei, and Linfeng Zhang. Credit where it is due: Cross-modality connectivity drives precise reinforcement learning for mllm reasoning, 2026. URL https://arxiv.org/abs/2602.11455.
- [12] Fucai Ke, Joy Hsu, Zhixi Cai, Zixian Ma, Xin Zheng, Xindi Wu, Sukai Huang, Weiqing Wang, Pari Delir Haghighi, Gholamreza Haffari, Ranjay Krishna, Jiajun Wu, and Hamid Rezatofighi. Explain before you answer: A survey on compositional visual reasoning, 2025. URL https://arxiv.org/abs/2508.17298.
- [13] Weijie Li, Jin Wang, Liang-Chih Yu, and Xuejie Zhang. Step-grpo: Enhancing reasoning quality and efficiency via structured prm-based reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence, 40(37):31734–31742, Mar. 2026. doi: 10.1609/aaai.v40i37.40441. URL https://ojs.aaai.org/ index.php/AAAI/article/view/40441.
- [14] Xirui Li, Ming Li, and Tianyi Zhou. What does rl improve for visual reasoning? a frankenstein-style analysis,

2026. URL https://arxiv.org/abs/2602.12395.

- [15] Zicheng Lin, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xing Wang, Ruilin Luo, Chufan Shi, Siheng Li, Yujiu Yang, and Zhaopeng Tu. Critical tokens matter: Token-level contrastive estimation enhances LLM’s reasoning capability. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 37906–37918. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/lin25j.html.
- [16] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [17] Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma GongQue, Shanglin Lei, Yifan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20023–20070, 2025.
- [18] Ahmad Rezaei, Mohsen Gholami, Saeed Ranjbar Alvar, Kevin Cannons, Mohammad Asiful Hossain, Zhou Weimin, Shunbo Zhou, Yong Zhang, and Mohammad Akbari. Cppo: Contrastive perception for vision language policy optimization. arXiv preprint arXiv:XXXX.XXXXX, 2026.
- [19] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.
- [20] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.
- [21] Yan Shu, Chi Liu, Robin Chen, Derek Li, and Bryan Dai. Fleming-vl: Towards universal medical visual reasoning with multimodal llms, 2025. URL https://arxiv.org/abs/2511.00916.
- [22] Yan Shu, Bin Ren, Zhitong Xiong, Xiao Xiang Zhu, Begüm Demir, Nicu Sebe, and Paolo Rota. Terrascope: Pixel-grounded visual reasoning for earth observation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16712–16722, 2026.

- [23] Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie, Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui, Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu, Zhilin Yang, and Xinyu Zhou. Attention residuals, 2026. URL https://arxiv.org/ abs/2603.15031.
- [24] Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, and Huajun Chen. Mllm can see? dynamic correction decoding for hallucination mitigation. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 13712–13736, 2025. URL https://proceedings.iclr.cc/paper_files/paper/2025/file/ 24079b91da7257cb78805262996152b8-Paper-Conference.pdf.
- [25] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/ forum?id=QWTCcxMpPA.
- [26] Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025.
- [27] Yiming Wang, Pei Zhang, Baosong Yang, Derek F. Wong, and Rui Wang. Latent space chain-of-embedding enables output-free llm self-evaluation, 2025. URL https://arxiv.org/abs/2410.13640.
- [28] Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025.
- [29] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, Wenhai Wang, Jifeng Dai, and Jinguo Zhu. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025. URL https://arxiv.org/abs/2504.15279.
- [30] Shuo Yang, Yuwei Niu, Yuyang Liu, Yang Ye, Bin Lin, and Li Yuan. Look-back:implicit visual re-focusing in mllm reasoning, 2025. URL https://arxiv.org/pdf/2505.07889.
- [31] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.
- [32] Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. Token-level direct preference optimization. arXiv preprint arXiv:2404.11999, 2024.
- [33] Chi Zhang, Haibo Qiu, Qiming Zhang, Yufei Xu, Zhixiong Zeng, Siqi Yang, Peng Shi, Lin Ma, and Jing Zhang. Perceptual-evidence anchored reinforced learning for multimodal reasoning. arXiv preprint arXiv:2511.18437, 2025.
- [34] Haoji Zhang, Xin Gu, Jiawen Li, Chixiang Ma, Sule Bai, Chubin Zhang, Bowen Zhang, Zhichao Zhou, Dongliang He, and Yansong Tang. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416, 2025.

- [35] Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. MLLMs know where to look: Training-free perception of small visual details with multimodal LLMs. In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2502.17422.
- [36] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.
- [37] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/2507.18071.

#### A Overview of Appendix

- • B: LLM Usage Statement.
- • C: Broader Impact.
- • D: Details of Experimental Setup
- • E: Implementation Details.
- • F: Diagnostic Annotation Protocol.
- • G: More Ablation Studies.
- • H: Analysis of Computational Overhead.
- • I: Analysis of Training Dynamics.
- • J: Extended Generalization Analysis.
- • L: More Visualization Results.

#### B LLM Usage Statement

An LLM was used as an assistive tool strictly limited to proofreading and rephrasing for clarity. It was not used for generating core ideas, data analysis, or writing the main content.

#### C Broader Impact

The core mechanism of DyCo-RL, which leverages within-modality attention restructuring to assign functional roles and perform alignment-guided advantage reweighting, extends naturally beyond visual mathematical reasoning. Cross-modal coordination failures, where a model struggles to dynamically alternate between visual grounding and textual deduction, are a pervasive bottleneck across long-chain multimodal tasks. Whenever an objective requires interleaving perceptual evidence with contextual or symbolic inference, our roleaware token-level optimization becomes directly applicable.

Representative scenarios include visual document and chart understanding [9], long-video reasoning [34], remote sensing reasoning [22] , and medical image analysis [21]. All these domains require continuous dynamic coordination between perceptual grounding and higherlevel reasoning. In such settings, coordination breakdowns frequently manifest as hallucinated details, temporally inconsistent claims, or derivation steps that drift away from the underlying visual evidence.

More broadly, standard RLVR paradigms treat cross-modal coordination as a mere byproduct of sequence-level reward broadcasting. By elevating it to an explicit token-level optimization signal, DyCo-RL provides a robust framework for building faithful, hallucinationresilient multimodal reasoners. Since the Fisher–Rao assignment derives entirely from intrinsic attention geometry and the reweighting strategy remains algorithm-agnostic, this principle integrates seamlessly with existing reward designs and modern policy optimization baselines such as GRPO.

We envision role-aware token-level optimization as a core component for trustworthy multimodal AI systems deployed in high-stakes domains (e.g., healthcare, autonomous systems, and scientific discovery). In these critical fields, the step-by-step fidelity of the reasoning process is just as paramount as the final-answer correctness.

#### D Details of Experimental Setup

To provide a comprehensive overview of our evaluation framework, we detail the representative baseline algorithms and the selected benchmarks below.

Baseline Algorithms. We evaluate DyCo-RL against four reinforcement learning baselines that span the recent evolution of policy optimization for reasoning-oriented large language models. GRPO [19] relies on intra-group reward normalization to eliminate the auxiliary critic model. DAPO [31] extends this by introducing asymmetric clipping and dynamic rollout strategies to stabilize long-chain reasoning. SAPO [4] utilizes a smooth, sigmoidbased soft gating mechanism with asymmetric temperatures to address optimization discontinuities. Finally, GSPO [37] reformulates the objective at the trajectory level, computing a cumulative likelihood ratio across the entire generated sequence for coherent long-chain optimization.

Evaluation Benchmarks. We evaluate DyCo-RL on seven benchmarks that jointly stress the two failure modes our method targets: faithful visual grounding for visually-oriented tokens and consistent textual anchoring for text-oriented tokens. These benchmarks are categorized into mathematical reasoning and visual-centric reasoning domains.

###### Mathematical Reasoning.

- • WeMath [17]: A hierarchical visual mathematics benchmark that decomposes problemsolving into atomic knowledge concepts and structured reasoning steps. It provides a diagnostic taxonomy to disentangle knowledge gaps from genuine reasoning failures.
- • MathVision [25]: A dataset of 3,040 competition-level math problems with visual contexts, curated from real competitions across 16 disciplines. It is designed to stress longchain multi-step visual mathematical reasoning.
- • MathVerse [36]: A diagram-centric visual math benchmark where problems are rendered in progressively text-reduced versions. This design isolates the contribution of genuine visual perception by preventing models from relying on textual shortcuts.
- • LogicVista [29]: An integrated logical reasoning benchmark requiring models to perform multi-step symbolic inference strictly grounded in visual evidence across spatial, numerical, and mechanical domains.

###### Visual-centric Reasoning.

- • HallusionBench [5]: A diagnostic benchmark explicitly designed to probe visual illusions and language-prior-induced hallucinations in multimodal models through carefully paired image–question controls.

- • MME [3]: A comprehensive evaluation suite measuring both perception (e.g., existence, count, position, OCR) and cognition (e.g., commonsense reasoning) across 14 distinct sub-tasks.
- • MMBench [16]: An all-around multimodal benchmark comprising multiple-choice questions that span 20 fine-grained ability dimensions, evaluated under a robust circular protocol to mitigate option-order bias.

#### E Implementation Details

All experiments are conducted within the VLM-R1 training framework [20], which serves as the base training paradigm for all models in this work. And as summarized in Table 4, all models are trained for a single epoch on the ThinkLite-VL-Hard-11k dataset, which comprises 11,031 complex reasoning instances. During rollout generation, we sample 4 responses per prompt using a temperature of 0.9. The training objective is guided by a composite reward signal, integrating a binary accuracy reward to verify mathematical correctness and a format reward to enforce structural compliance.

Unlike standard parameter-efficient fine-tuning, we leave the vision tower entirely unfrozen to enable full-parameter cross-modal optimization. Across all evaluated baseline algorithms, we employ the AdamW optimizer with a peak learning rate of 1 × 10−6 and a linear decay schedule. To stabilize the training dynamics, we apply gradient checkpointing and cap the global gradient norm at 1.0. For the specific hyperparameter configurations of each RL method (e.g., clipping bounds or KL coefficients), we strictly adhere to their official implementations and recommended settings.

All experiments are executed in bfloat16 precision, heavily relying on DeepSpeed ZeRO-3 optimization (without CPU offloading) to manage memory efficiency. We configure the perdevice batch size to 4 and accumulate gradients over 2 steps, yielding a global batch size of 64. The entire training pipeline is deployed on a single server equipped with 8 × NVIDIA A100 (80GB) GPUs. The training requires approximately 48 GPU-hours for the 3B model and 72 GPU-hours for the 7B model , averaged per algorithm.

#### F Diagnostic Annotation Protocol

To ground our analysis of visual reasoning breakdowns in concrete model behavior, we construct a token-level annotated error set using the Qwen2.5-VL-3B model optimized with our standard GRPO baseline. We uniformly sample 200 incorrect rollouts from MathVerse and MathVision, two benchmarks covering diverse forms of visual mathematical reasoning, including diagram understanding, geometric reasoning, symbolic derivation, and chart interpretation. Annotators are asked to identify, for each generated token span, both the source of failure (visual grounding versus textual reasoning) and whether the corresponding operation is executed correctly. This dual-axis evaluation yields a comprehensive four-way taxonomy.

Annotation Principle. Annotations are assigned according to a token’s functional role within the local reasoning trajectory rather than its isolated lexical meaning. Specifically, anno-

Table 4: Key hyperparameters.

###### Hyperparameter Value

Optimizer AdamW Learning Rate 1 × 10−6 LR Schedule linear Epochs 1 Freeze Vision Tower No Global Batch Size 64 Per-Device Batch Size 4 Gradient Accumulation 2 GPUs per Node 8 Rollouts per Prompt (G) 4 Rollout Temperature 0.9 Rollout Top-p 1.0 Max Response Length 2048 Reward Signal accuracy + format Precision bfloat16 Gradient Clipping 1.0 DeepSpeed Stage ZeRO-3

tators determine whether a token primarily contributes to image-grounded perception or text-based reasoning under the current decoding context. To improve annotation consistency, labels are first assigned at the semantic-span level and subsequently projected back to individual tokens. Connective tokens are categorized according to the functional role of their subsequent semantic span. For example, the prefix “since angle ADE is 80◦” is labeled as visually-oriented because the span performs image-grounded perception, whereas “since Eq. (3) implies” is categorized as text-oriented reasoning.

Token Taxonomy. Each generated token within the reasoning chain is classified into exactly one of the following four categories:

- • Correct Visually-oriented Token (Pr). Tokens that accurately extract or describe imagegrounded information. Examples include correctly reading numerical values from a chart, identifying existing geometric relations, or accurately recognizing objects from the visual input.
- • Incorrect Visually-oriented Token (Pe). Tokens that attempt to describe visual content but directly contradict the provided image. This includes misread numerical values, hallucinated visual elements, or erroneous spatial descriptions.
- • Correct Text-oriented Token (Rr). Tokens that perform logically valid inference operations conditioned strictly on the preceding context. This encompasses correct algebraic manipulations, valid theorem applications, or intermediate symbolic deductions without requiring new visual evidence.

- Table 5: Ablation Study on Reweight Strength α. Bold formatting indicates the highest performance in each column.

Method WeMath MathVision MathVerse LogicVista HallusionBench MME MMBench Avg. GRPO Baseline (α = 0) 23.0 21.4 34.4 39.8 60.8 79.9 54.8 44.9 α = 0.1 25.7 23.3 34.5 39.8 63.4 80.1 56.5 46.2 DyCo-RL (α = 0.2) 26.9 22.4 36.0 41.0 62.4 80.7 58.2 46.8

- α = 0.3 25.0 24.8 36.4 40.2 59.4 81.7 57.8 46.5
- α = 0.4 23.1 19.5 32.4 37.4 59.9 79.0 54.8 43.7
- α = 0.5 21.0 17.8 31.5 35.3 58.6 78.2 51.0 41.9

- Table 6: Comparison between Advantage Reweighting and Reward Shaping. Bold formatting indicates the highest performance in each column.

Method WeMath MathVision MathVerse LogicVista HallusionBench MME MMBench Avg. GRPO Baseline 23.0 21.4 34.4 38.9 60.8 79.9 54.8 44.7 DyCo-RL (Reward Shaping) 24.4 19.7 34.5 37.6 62.4 80.3 61.3 45.7 DyCo-RL (Adv. Reweighting) 26.9 22.4 36.0 41.0 62.4 80.7 58.2 46.8

• Incorrect Text-oriented Token (Re). Tokens corresponding to logically inconsistent or unsupported reasoning steps. Examples include self-contradictory mathematical conclusions, invalid symbolic derivations, or abrupt claims lacking traceable antecedents in the reasoning history.

Annotation Procedure and Agreement. The labeling process follows a strict double-blind protocol. Each sample is independently evaluated by two annotators possessing graduate-level mathematical training. While annotators are provided with the input image, question, full model response, and reference answer, they are explicitly instructed to label tokens based on the internal consistency and visual faithfulness of the generated trajectory, rather than retrofitting labels based on final-answer correctness. Any labeling discrepancies between the two primary annotators are resolved by a third senior referee.

To quantitatively evaluate annotation reliability, we compute Cohen’s κ at the token-span level before the final adjudication. The resulting agreement score (κ = 0.85) demonstrates substantial inter-annotator consistency. The majority of the initial disagreements (approximately 3% of all annotated tokens) are heavily concentrated around the highly ambiguous transition regions between perception and reasoning spans. Crucially, this rigorous diagnostic protocol serves as the standardized foundation for the comparative mechanistic analysis later presented in Section 6.

#### G Additional Ablation Studies

Ablation on Reweight Strength (α). Table 5 presents the ablation study on the reweight strength α, which controls the intensity of modality-aware advantage adjustment during RL training. Overall, introducing a moderate reweighting penalty consistently improves visual reasoning performance over the unweighted baseline (α = 0). Specifically, α = 0.2 achieves the optimal balance, yielding the strongest or near-strongest results across most datasets. This suggests that moderate role assignment effectively encourages visually-oriented tokens to acquire richer visual evidence while preserving stable reasoning dynamics.

Table 7: Ablation Study on the Number of Training Rollouts (R).

Method WeMath MathVision MathVerse LogicVista HallusionBench MME MMBench Avg. DyCo-RL (R = 4, Default) 25.9 22.4 36.0 41.0 62.4 80.7 58.3 46.7 DyCo-RL (R = 8) 27.1 23.0 35.3 41.4 62.7 82.0 59.6 47.3 DyCo-RL (R = 16) 28.0 23.5 36.0 42.3 63.6 82.5 61.3 48.2

Conversely, when α is too small (e.g., α = 0.1), the signal is insufficient to meaningfully reshape cross-modal attention behaviors. At the other extreme, excessively large values (α ≥ 0.4) precipitously degrade performance. We hypothesize that overly aggressive reweighting disrupts the model’s intrinsic coordination, forcing attention patterns into overly constrained sub-optima. The consistent trend across benchmarks confirms that balanced modulation is strictly more beneficial than aggressive intervention.

Advantage Reweighting vs. Reward Shaping. As shown in Table 6, we further contrast our token-level advantage reweighting with a straightforward trajectory-level reward shaping baseline. In the latter, the training signal is modified by directly augmenting the binary accuracy reward (Racc) with the aggregated alignment score (s¯t):

R′ = Racc + αs¯t. (12)

While this shaping approach occasionally outperforms the vanilla GRPO baseline, its gains are highly inconsistent and overall inferior to advantage reweighting. We attribute this discrepancy to the optimization dynamics of GRPO. Reward shaping perturbs the trajectorylevel reward prior to group normalization, which artificially alters the scale and variance of the resulting advantage estimates. This amplification of noise in the group-relative baseline leads to unstable policy updates. Advantage reweighting, by contrast, injects fine-grained guidance after the baseline computation, successfully avoiding these destabilizing effects.

Scaling the Number of Training Rollouts. The number of rollouts per prompt, R, dictates a fundamental trade-off in group-relative RL: a larger R yields lower-variance advantage estimates but linearly inflates training costs. To identify where this trade-off saturates for DyCo-RL, we evaluate R ∈ {4,8,16} while holding all other hyperparameters constant. As detailed in Table 7, performance scales monotonically with R, but exhibits sharply diminishing returns. Moving from R = 4 to R = 8 provides a noticeable lift across nearly all benchmarks, whereas R = 16 yields only marginal further increments. Given that the computational overhead scales linearly, doubling the compute budget from R = 8 to R = 16 is rarely justifiable for fractional score improvements. Consequently, we adopt R = 4 as the default configuration to maintain high efficiency, while noting that practitioners with abundant compute budgets can straightforwardly scale R for maximum performance.

Table 8: Analysis of training throughput and computational overhead.

Method Throughput (samples/sec) ∆ Throughput GRPO Baseline 0.447 – DyCo-RL 0.327 -26.8%

#### H Analysis of Computational Overhead

As detailed in Table 8, DyCo-RL introduces a moderate computational overhead compared to the standard GRPO baseline, reducing overall training throughput by approximately 27%. This cost primarily stems from the token-level Fisher–Rao distance computation and the subsequent role assignment operations at each generation step. Because these mechanisms require fine-grained extraction and alignment of cross-modal attention weights throughout the decoding trajectory, they inherently increase training-time memory bandwidth utilization and computation.

Crucially, this overhead is strictly confined to the optimization phase. At inference time, the advantage reweighting pipeline is completely detached, allowing the optimized model to retain the exact generation speed and memory footprint of the vanilla backbone. Given the substantial performance surges across diverse visual and mathematical reasoning benchmarks, this one-time training cost represents a highly favorable trade-off for mitigating cross-modal coordination breakdowns.

[Figure 6]

Figure 5: Training dynamics comparing DyCo-RL against the standard GRPO baseline. The curves illustrate the smoothed average accuracy reward across training steps, demonstrating the accelerated convergence and sustained learning signal of our method.

#### I Analysis of Training Dynamics

To elucidate how DyCo-RL reshapes the optimization landscape, we compare its training trajectory against the standard GRPO baseline with respect to the accuracy reward. For visual clarity, the plotted curves are smoothed using a rolling mean with a window size of 10.

As illustrated in Fig. 5, DyCo-RL demonstrates accelerated convergence during the early stages of training and sustains a consistent performance margin over GRPO. Throughout the entire optimization process, our method maintains a strictly higher average reward. This robust trajectory confirms that role-aware token-level advantage reweighting delivers a more stable and effective learning signal than conventional sequence-level reward broadcasting.

Table 9: Out-of-Domain Generalization Results. Bold formatting indicates the highest performance on each benchmark. DyCo-RL consistently outperforms the baseline across diverse non-mathematical tasks, demonstrating robust cross-domain generalization.

Method A-OKVQA RealWorldQA MMStar SEED-Bench (IMG) ChartQA (Test)

GRPO Baseline 82.3 47.0 51.2 68.2 81.0 DyCo-RL 82.7 49.5 53.2 69.0 83.2

#### J Extended Generalization Analysis

A critical question regarding token-level advantage reweighting is whether the learned alignment signal overfits to the specific visual-mathematical distribution seen during training. To rigorously assess its out-of-domain robustness, we evaluate DyCo-RL on several additional benchmarks that encompass diverse multimodal reasoning scenarios. These include commonsense visual understanding, real-world image reasoning, chart interpretation, and general multimodal question answering.

As reported in Table 9, DyCo-RL consistently outperforms the standard GRPO baseline across all evaluated zero-shot benchmarks. These results confirm that our proposed dynamic alignment mechanism is not spuriously correlated with the training domain. Rather, by fundamentally addressing the cross-modal coordination bottleneck, the role-aware reweighting strategy generalizes seamlessly across vastly different visual formats, reasoning styles, and task structures.

#### K Prompt Template

We adopt a GRPO-style reasoning format that requires the model to generate intermediate reasoning within <think> tags and final responses within <answer> tags:

Prompt Template SYSTEM: You are a helpful assistant.

USER:A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.

Question: {question}

#### L More Visualization Results

We provide additional qualitative examples in Fig. 6–7, comparing the standard GRPO baseline against our DyCo-RL augmented version on the Qwen2.5-VL-3B architecture. DyCo-RL consistently exhibits role-aligned attention and dynamic cross-modal alternation across diverse visual reasoning cases, further corroborating the findings in the main paper.

[Figure 7]

###### Figure 6: Visualization Cases on Mathematical and Logical Reasoning.

[Figure 8]

###### Figure 7: Visualization Cases on General Task and Hallucination Reasoning.

