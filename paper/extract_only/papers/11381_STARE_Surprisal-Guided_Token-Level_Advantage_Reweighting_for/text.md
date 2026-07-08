[Figure 1]

### STARE: Surprisal-Guided Token-Level Advantage Reweighting for Policy Entropy Stability

# arXiv:2606.19236v1[cs.LG]17Jun2026

Haipeng Luo Qingfeng Sun Songli Wu Can Xu† Wenfeng Deng Han Hu Yansong Tang† Shenzhen International Graduate School, Tsinghua University Tencent Hunyuan {luohp24@mails., tang.yansong@sz.}tsinghua.edu.cn {victorqsun,leocaxu}@tencent.com

###### Abstract

Reinforcement Learning with Verifiable Rewards algorithms like GRPO have emerged as the dominant post-training paradigm for complex reasoning in LLMs, yet commonly suffer from policy entropy collapse during training. We conduct a first-order gradient analysis of token-level entropy dynamics under GRPO and identify a token-level credit assignment mismatch: the per-token entropy variation decomposes into the product of the trajectory-level advantage and an entropy sensitivity function over the next-token distribution, yielding an advantage-surprisal four-quadrant structure and a near-criticality property. Motivated by it, we propose STARE (Surprisal-guided Token-level Advantage Reweighting for policy Entropy stability), which identifies entropy-critical token subsets via batch-internal surprisal quantiles, selectively reweights their effective advantages, and incorporates a target-entropy closed-loop gate for stable entropy regulation. Across model scales from 1.5B to 32B and three task families (Short CoT, Long CoT, and Multi-Turn Tool Use), STARE sustains stable RL training over thousands of steps while maintaining policy entropy within the target band. On AIME24 and AIME25, STARE outperforms DAPO and other competitive baselines by 4%–8% in average accuracy, with reflection tokens and response length growing in tandem, indicating sustained exploration–exploitation balance that further unlocks RL training potential. Code is available at https://github.com/hp-luo/STARE

0.4

0.3

Entropy

0.2

0.1

GRPO-ds

STARE

0 400 800 1200 1600 2000

Training Steps

(a) Training Entropy

0.5

AIME24Acc

0.4

0.3

0.2

GRPO-ds

STARE

0.1

0 400 800 1200 1600 2000

Training Steps

(b) AIME24 Acc

0.3

AIME25Acc

0.2

GRPO-ds

0.1

STARE

0 400 800 1200 1600 2000

Training Steps

(c) AIME25 Acc

Figure 1: Training dynamics of STARE vs. GRPO-ds on Qwen2.5-7B-Base (cold-start SFT from Retool 2K) in the tool-use agent scenario: policy entropy, AIME24 accuracy, and AIME25 accuracy.

###### 1 Introduction

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as the dominant post-training paradigm for eliciting complex reasoning in LLMs, as exemplified by DeepSeek-R1, Qwen3, and Kimi K1.5(Guo et al., 2025; Achiam et al., 2023; Team et al., 2025a; Team, 2025; Anthropic, 2025). Among RLVR algorithms, Group Relative Policy Optimization (GRPO) dispenses with the value network and instead employs group-normalized rewards as the baseline for advantage estimation; and it has been widely adopted in mathematical reasoning, code generation, and is effective at inducing emergent behaviors like long-cot reasoning and self-reflection(Shao et al., 2024; Schulman et al., 2017).

As RL training extends over more optimization steps, however, GRPO-style algorithms commonly suffer from policy entropy collapse: entropy decays rapidly, output diversity vanishes, the policy converges prematurely, and within-group rollouts homogenize, degrading relative advantage estimation and

† Corresponding authors. This work was done during Luo’s internship at Tencent and was supported by the CIE-Tencent Ph.D. Student Research Incentive Program (Tencent Hunyuan Special Fund).

ultimately capping trainable steps, a critical bottleneck for long-horizon post-training(Yue et al., 2025a; Farquhar et al., 2024). Existing mitigations fall into three directions. (i) Adjusting the clipping thresholds for the importance-sampling ratio (i.e., DAPO’s clip-higher) protects low-probability exploratory tokens, but exerts an asymmetric and uncontrollable effect on entropy and is largely inactive in the on-policy regime where the ratios stay near one(Yu et al., 2025; Chen et al., 2026; Xi et al., 2025; Fu et al., 2026). (ii) Asymmetric trajectory-level weighting between positive and negative rollouts: Upweighting rare correct rollouts or biasing updates toward negative samples, provides coarse-grained control(Zhu et al., 2025; Tang et al., 2025; Wang et al., 2025b; 2026; Yang et al., 2025b; He et al., 2025a; 2026). (iii) Entropy-aware advantage reshaping or entropy regularization couples token-level entropy into the advantage, but tends to overamplify high-entropy tokens, induce oscillations, and remain hyperparameter-sensitive(Cheng et al., 2025; Cui et al., 2025; Xu et al., 2025; He et al., 2025c; Yang et al., 2025a; Huang et al., 2025). These approaches slow entropy decay to varying degrees, yet operate at trajectory or sample granularity, lacking a principled account of the collapse mechanism. Two questions remain open: which tokens drive entropy decay under GRPO, and how strong an intervention suffices to reverse it.

We conduct a first-order gradient analysis of token-level entropy dynamics under GRPO and identify a token-level credit assignment mismatch: although the trajectory-level advantage Aˆi is shared across all tokens within a rollout, the per-token entropy contribution decomposes into the product of Aˆi and a local entropy sensitivity function Φ determined by the next-token distribution (Section 3). This decomposition yields an advantage–surprisal four-quadrant view: within positive-advantage trajectories, low-surprisal tokens dominate the sampling frequency and drive most entropy-decreasing updates, whereas the rare high-surprisal tokens that could raise entropy are diluted (a mirror-image asymmetry holds for negativeadvantage trajectories). We establish a near-criticality property: a mild token-level weight perturbation suffices to flip the sign of entropy evolution and is robust to the specific weight value and beyond the critical threshold, the weight modulates the magnitude rather than the direction of the entropy shift.

Motivated by these, we propose STARE (Surprisal-guided Token-level Advantage Reweighting for policy Entropy stability), a minimally invasive token-level credit-rebalancing mechanism that operates inside the clipped surrogate of GRPO. STARE selects an entropy-critical token subset via batch-internal surprisal quantiles, thereby selectively amplifying the effective advantage of positive-advantage high-surprisal tokens and attenuating that of negative-advantage high-surprisal tokens. A target-entropy closed-loop gate further governs the intervention: the reweighting is activated to restore exploration when the batch-averaged entropy H¯k drops below a target level Htgt, and reverts to GRPO once entropy recovers, yielding closed-loop, stable, and low-intrusion entropy regulation.

We validate STARE across multiple model scales and task regimes. On 7B models, STARE stably sustains over 5k RL training steps; on 14B and 32B models, it sustains over 1.5k steps; throughout training, the policy entropy is held within the target band. Across three task families spanning Short CoT, Long CoT, and multi-turn tool-use agents, STARE consistently outperforms DAPO on AIME24 and AIME25 by 4%-8% in average accuracy, with reflection-related tokens and response length growing in tandem, indicating an improved exploration–exploitation balance.

The main contributions of this work are as follows: (i) From a first-order entropy-dynamics analysis, we expose the token-level credit assignment mismatch in GRPO and establish a near-criticality property: a mild weight perturbation suffices to flip the direction of entropy evolution while remaining robust to the weight value. (ii) We propose a surprisal-based advantage reweighting mechanism coupled with a target-entropy closed-loop constraint, achieving stable policy-entropy regulation through a minimal modification to the GRPO objective and sustaining RL training over thousands of steps. (iii) We validate STARE across model scales from 1.5B to 32B and across Short CoT, Long CoT, or multi-turn tool-use regimes, where it maintains stable policy entropy and substantially outperforms DAPO and other baselines by a consistent margin. We defer the discussion of related work to Appendix A.

###### 2 Preliminaries

GRPO. Given prompt x, the old policy πθold samples G responses with rewards {ri}iG=1; the groupnormalized advantage is Aˆi = (ri − mean({rj}))/ std({rj}).The clipped surrogate objective is:

Ti

B

1 N

min ρi,t(θ) Aˆi, clip(ρi,t(θ), 1−ϵ, 1+ϵ) Aˆi ,

##### ∑

##### ∑

JGRPO(θ) =

t=1

i=1

where ρi,t(θ) ≜ πθ(oi,t | xi, oi,<t)/πθold(oi,t | xi, oi,<t) is the per-token importance ratio, B is the number of responses, N = ∑iB=1 Ti is the total token count, and β = 0 throughout (no KL penalty). Let c = (x, o<t) denote context and V the vocabulary. The next-token distribution is parameterized as πv ≜ πθ(v | c) = exp(zv)/ ∑v′ exp(zv′), with softmax Jacobian ∂πv′/∂zv = πv′(δv′v − πv). Token Surprisal,

Entropy, and Logit updates. The token surprisal is sv ≜ − ln πv and the position-level policy entropy is H ≜ − ∑v πv ln πv = Eπ[s]Shannon (1948); Oh et al. (2024); Zeng et al. (2026); Oh & Schuler (2023); Smith & Levy (2013); the batch mean H¯ ≜ N−1 ∑i,t Hi,t typically decreases during RL fine-tuning (entropy collapse). In the unclipped regime, the GRPO gradient at a position where token a was sampled yields the logit update ∆zv = η Aˆ (δva − πv) for all v ∈ V, where η > 0 is an infinitesimal step size (Appendix D.2).

Lemma 2.1 (Entropy gradient w.r.t. logits: surprisal-deviation form). For any v ∈ V, ∂∂zH

= πv(sv − H). Appendix D.3 provides the derivation.

v

Raising zv increases entropy when token v is rarer than average (sv > H) and decreases it otherwise. It relates entropy to individual logits and underlies the token-level entropy dynamics derived below.

###### 3 Theoretical Analysis

We develop a theoretical framework for analyzing entropy evolution during GRPO training, proceeding from token-level gradient analysis (Section 3.1) through an advantage–surprisal decomposition (Section 3.2) and a batch-level near-criticality result (Section 3.3) to cross-step feedback dynamics (Section 3.4). Complete proofs are in Appendices D–F.

###### 3.1 First-Order Gradient Analysis of Token-Level Policy Entropy

Consider the next-token distribution π(· | c) with entropy H. Let a denote the sampled token, with probability p = π(a | c) and surprisal sa = − ln p. Define S2 ≜ ∑v∈V πv2( ln πv + H) and Φ(p) ≜ p( ln p + H) − S2. We call Φ the entropy sensitivity function: it measures the signed excess of the sampled token’s probability-weighted surprisal deviation over the distributional baseline S2.

Theorem 3.1 (Token-level entropy variation). In the unclipped regime of GRPO, let Aˆ denote the advantage at this position, and let η be the step size along the GRPO policy-gradient direction. Then dHdη

= −Aˆ Φ(p).

η=0

The proof (Appendix D.4) follows by taking the inner product of ∂H/∂zv (Lemma 2.1) with the GRPO logit update. The result decomposes the instantaneous entropy effect into the advantage Aˆ and Φ(p), governing whether probability redistribution concentrates or disperses the distribution. And their product determines the sign and magnitude of entropy variation at each position.

###### 3.2 Advantage–Surprisal Four-Quadrant Decomposition

Determining sign(dH/dη) reduces to analyzing sign(Φ(p)). Two properties, proved in Appendix E, underpin the analysis: S2 > 0 for any non-uniform distribution (Lemma E.1), and H > S2 for any non-degenerate distribution (Lemma E.2). These imply Φ(0+) = −S2 < 0 and Φ(1) = H − S2 > 0. Since Φ′(p) = ln p + H + 1, the function is strictly increasing on (e−(H+1),1], yielding:

Proposition 3.2 (Uniqueness of the Critical Surprisal Threshold). For any non-uniform, non-degenerate π, there exists a unique p∗ ∈ (e−H,1) with s∗ ≜ − ln p∗ ∈ (0, H) such that Φ(p∗) = 0 and when Φ(p) > 0 ⇐⇒ p > p∗ ⇐⇒ sa < s∗. The proof is in Appendix E.3.

The threshold s∗ partitions the vocabulary into a low-surprisal region (sa < s∗) and a high-surprisal region (sa > s∗). Substituting into Theorem 3.1 yields the four-quadrant structure.

Corollary 3.3 (Four-Quadrant Decomposition). The sign of dH/dη is determined by (sign Aˆ, 1[sa < s∗]): (i) reinforcing low-surprisal tokens (Aˆ > 0,sa < s∗) reduces entropy; (ii) reinforcing high-surprisal tokens (Aˆ > 0,sa > s∗) increases it; (iii) suppressing low-surprisal tokens (Aˆ < 0,sa < s∗) increases it; (iv) suppressing high-surprisal tokens (Aˆ < 0,sa > s∗) reduces it. Each GRPO step thus propagates four token-level entropy signals with opposing signs. The proof is in Appendix E.4.

Asymmetric entropy contributions. Since rollouts are sampled from πθ, low-surprisal tokens are drawn more frequently than high-surprisal ones at each decoding position. Within positive-advantage trajectories, entropy-decreasing tokens (low surprisal, Φ > 0) therefore constitute the statistical majority. Because

GRPO assigns a single trajectory-level Aˆi to all tokens, it cannot distinguish these opposing entropy effects: the reinforced low-surprisal majority systematically drives the distribution toward concentration, while the high-surprisal minority that could preserve diversity contributes limited entropy-increasing effects. An analogous asymmetry governs the negative-advantage subset, revealing a fundamental gradient-level mechanism underlying entropy collapse in GRPO.

###### Token-Level Entropy Analysis

###### Surprisal-Guided Partitioning

Sort tokens by surprisal s (descending), then take top-P%.

|H↓(majority)<br><br>|H↑(minority) L+ (entropy-critical)|
|---|---|
|H↑(majority)|L- (entropy-critical)<br><br>H↓(minority)|

Aˆ

>0

top-P%

high s low s 4 5 6 7 8 9 10 11 12 L+ ω=W amplify

Aˆ

>0

positiveadvantage okens

1 2 3

Aˆ

Advantage

Aˆ

<0

top-P%

high s low s 4 5 6 7 8 9 10 11 12 L+ ω=W attenuate

Aˆ <0

low s* high

negativeadvantage tokens

1 2 3

Surprisal s

Four-Quadrant Decomposition

Batch-internal Surprisal Quantile Proxy

###### Closed-Loop Token-Level Reweighting

###### Entropy Dynamics

D. Token Weight Assignment E. Policy Update

A. Rollouts

B. Compute

PolicyEntropyHk

Yes

STARE

amplify

if(i,t)∈L+

W

C. Decision

(stable)

HK <Htgt

ωi,t=

attenuate

if(i,t)∈L-

M

Htgt

…

Aˆ

advantage-reweighted

group-normalized

1 otherwise

gradient step

GRPO (collapse)

{Oi,t}~πθold

No

batch entropy

HK

ω = 1 (bypass reweighting)

training step k

- Figure 2: Overview of STARE. Guided by a four-quadrant decomposition of token-level entropy dynamics (top-left) and a batch-internal surprisal-quantile proxy that identifies entropy-critical tokens (top-right), STARE applies target-entropy-gated advantage reweighting in GRPO (bottom-left), stabilizing policy entropy where vanilla GRPO collapses (bottom-right).

###### 3.3 Batch-Level Entropy Decomposition and Near-Criticality

The preceding analysis establishes that the entropy-contribution asymmetry persists within both advantage subsets. A natural quantitative question arises: what reweighting of the entropy-increasing minority suffices to reverse the batch-level net entropy gradient?

Theorem 3.4 (Entropy Neutrality Identity). For any conditional distribution π, Ea∼π[Φ(a)] = ∑v∈V πv Φ(πv) =

0. The proof is in Appendix E.6.

Token-level advantage reweighting. Define L+ = {(i, t) : Aˆi > 0, si,t > si∗,t}. Scaling the effective advantage of each token in L+ by a multiplicative factor W ≥ 1, while retaining unit weight at all remaining positions, yields:

Proposition 3.5 (Entropy Gradient under Token-Level Reweighting). ddHη¯

= −N1 [Λ − (W−1)Γ], where Λ ≜ ∑i,t Aˆi Φi,t, Γ ≜ ∑(i,t)∈L+ Aˆi |Φi,t| > 0, and the critical weight is W∗ = 1 + Λ/Γ.

W

The proof is in Appendix E.7. Two structural properties jointly ensure that W∗ remains near unity (nearcritical regime). First, the entropy neutrality identity guarantees that Λ is a weak residual arising solely from the statistical dependence between trajectory-level advantages and token-level entropy sensitivities;

under the assumptions formalized in Appendix E.9, |Λ|/Σabs = O(T−1) where Σabs = ∑i,t |Aˆi| |Φi,t|. Second, high-surprisal tokens carry amplified entropy sensitivity (Appendix E.8): s ≥ H ⇒ |Φ(p)| ≥ S2, whereas s < s∗ ⇒ |Φ(p)| ≤ H − S2. Although L+ is a sampling minority, the amplified per-token |Φ| values ensure that Γ remains appreciable.

Corollary 3.6 (Near-Criticality). When the sequence length T and the batch size are both sufficiently large, W∗ − 1 = Λ/Γ = O(T−1) (Appendix E.11). So beyond the critical threshold, the specific value of W principally controls the magnitude rather than the sign of the per-step entropy shift.

###### 3.4 Cross-Step Entropy Dynamics

Let ∆H¯k = −N−1[Λk − (W−1) Γk] denote the batch-level entropy variation at step k. Under W = 1, the sampling asymmetry implies Λk > 0 in expectation, so ∆H¯k < 0. The resulting entropy reduction further concentrates πθ, lowering the sampling frequency of high-surprisal tokens in subsequent batches, shrinking Γk+1, and raising Wk∗+1, forming a self-reinforcing loop of entropy collapse. Conversely, when

W > Wk∗, entropy increases disperse the distribution, enlarge Γk+1, and lower Wk∗+1, forming a symmetric loop of entropy recovery (formalized in Appendix F). The near-criticality result (corollary 3.6) establishes that a modest token-level weight adjustment suffices to alter the macroscopic entropy trajectory. This theoretical insight directly motivates the algorithmic design proposed in the next section: by selectively modulating the effective advantage weights of a targeted subset of tokens within the GRPO policy gradient, one can restore a sustainable dynamic equilibrium between the entropy-increasing and entropydecreasing gradient forces.

###### 4 Method

The theoretical analysis in Section 3 reveals that GRPO’s shared trajectory-level advantages induce a tokenlevel credit assignment mismatch: high-frequency low-surprisal tokens dominate gradient aggregation while sparse high-surprisal tokens with critical entropy effects are under-represented. Motivated by it, we propose Surprisal-Guided Token-Level Advantage Reweighting For Policy Entropy Stability (STARE): a surprisal-based credit rebalancing mechanism that operates within the clipped surrogate of GRPO, assigns differentiated weights to entropy-critical tokens, and incorporates target-entropy closed-loop feedback for stable entropy regulation on training, as shown in Figure 2.

###### 4.1 Entropy-Critical Token Partitioning via High-Surprisal Quantiles

Let T + = {(i, t) : Aˆi > 0} and T − = {(i, t) : Aˆi < 0} denote the positive- and negative-advantage token sets. Computing the exact critical threshold si∗,t (Proposition 3.2) at every position requires the full conditional distribution, incurring prohibitive overhead. Instead, STARE employs a simple, stable, and theoretically motivated batch-internal surprisal-quantile proxy. Concretely, within T + and T − separately, tokens are ranked in descending order of surprisal si,t = − ln πθ(oi,t | xi, oi,<t), and the top P% tokens are selected to form the entropy-critical sets below:

L± = {(i, t) ∈ T ± : si,t ≥ QP({sj,s}(j,s)∈T ±)}.

By Corollary 3.3, L+ approximately denotes the entropy-increasing tokens among positive-advantage responses, and L− indicates the entropy-decreasing tokens in negative-advantage responses. The fixed proportion P directly controls the intervention scale, obviating per-position threshold computation.

###### 4.2 Advantage-Conditioned Token-Level Credit Rebalancing We augment the GRPO objective with positive token-level weights ωi,t > 0:

1

ωi,t min ρi,t(θ) Aˆi, clip(ρi,t(θ), 1−ϵ, 1+ϵ) Aˆi .

##### N ∑

JSTARE(θ) =

i,t

Setting ωi,t ≡ 1 recovers STARE to standard GRPO. Because ωi,t > 0, STARE preserves all tokenlevel gradient directions: tokens with positive advantage remain reinforced, while those with negative advantage remain suppressed. STARE therefore acts as an advantage-conditioned credit-rebalancing mechanism, selectively rescaling relative magnitudes along the surprisal dimension.

###### Variant I: One-Sided Entropy Amplification (Aˆi > 0, High-Surprisal tokens, denoted as O1).

ωi(,V1t ) =

W, (i, t) ∈ L+, 1, otherwise,

W > 1.

Tokens in L+ simultaneously carry positive advantage and entropy-increasing effect; amplifying their weights directly strengthens the minority that GRPO systematically underweights. The resulting batch-

level net entropy shift is ΛV1 = Λ − (W−1) Γ+, where Γ+ = ∑(i,t)∈L+ |Aˆi Φi,t| > 0. And under the near-criticality condition (Corollary 3.6), a moderate W > 1 usually suffices to reverse the sign of the batch net entropy shift. We provide the Algorithmic 1 pseudocode in the Appendix C.

###### Variant II: Dual-Sided Entropy Regulation. Extending Variant I, this variant additionally attenuates the weights of tokens in L− (Aˆi < 0, High-Surprisal tokens, denoted as C2):

 

W, (i, t) ∈ L+, M, (i, t) ∈ L−, 1, otherwise,

ωi(,V2t ) =

W > 1, 0 < M < 1.



Tokens in L− reside in the high-surprisal tail of negative-advantage responses; and large negativeadvantage updates on these tokens redistribute mass from the tail toward the peak, exacerbating concentration. Attenuating their weights alleviates this entropy-decreasing pressure. Letting Γ− = ∑(i,t)∈L− |Aˆi Φi,t|, the batch net entropy shift becomes ΛV2 = Λ − (W−1) Γ+ − (1−M) Γ−.

Two-sided regulation simultaneously amplifies the entropy-increasing signal and attenuates the entropydecreasing signal to adjust policy entropy. Definitions and analysis for all four single-polarity (O1, O2, O3, O4) and four combined operations (C1, C2, C3, C4) are deferred to Appendices G and H.

- 4.3 Closed-Loop Regulation via Target-Entropy Gating

A purely open-loop reweighting strategy risks overshooting from entropy collapse into uncontrolled divergence. To achieve stable regulation, we introduce a batch-level target entropy Htgt and employ the current batch mean entropy H¯k as a closed-loop feedback signal via a binary gate gk = 1[H¯k < Htgt] and express the weights in unified form below. When H¯k < Htgt, the gate activates and

ωi,t =

 



1 + gk (W−1), (i, t) ∈ L+, 1 − gk (1−M), (i, t) ∈ L− (two-sided only), 1, otherwise.

STARE strengthens entropy-increasing signals; when H¯k ≥ Htgt, all weights revert to unity, automatically recovering standard GRPO. This drives entropy toward Htgt via bounded oscillation. Finer-grained sample-level and token-level closed-loop variants are presented in Appendix G.6.

- 4.4 Static and Adaptive Weighting Schedules

Fixed weights. Near-criticality (Corollary 3.6) implies that the required reweight perturbation is typically modest: beyond the critical point, the specific value principally controls the magnitude rather than the direction of the per-step entropy shift, reducing sensitivity to hyperparameter choices. Fixed W and M therefore suffice in most settings.

Adaptive weights. As training progresses, intervention strength may vary across phases with distinct distributional. STARE also supports adaptive weight updates driven by the target-entropy signal:

Wk+1 = clip Wk + α sgn Htgt − H¯k , [1, Wmax] , Mk+1 = clip Mk − α sgn Htgt − H¯k , [Mmin, 1] .

When H¯k < Htgt, W increases and M decreases, intensifying the intervention; otherwise both relax toward GRPO. The constraints W ≥ 1 and M ≤ 1 ensure graceful degradation. Setting α = 0 recovers

the fixed-weight. Default values α=0.01, Wmax=1.5, Mmin=0.5 yield robust performance.

Default configuration. All main experiments adopt Variant I (O1) with batch-level target-entropy gating and fixed weights. This minimal configuration suffices to stabilize entropy and improve performance. Ablations on two-sided regulation and adaptive weights are provided in Appendix H.

- 5 Experiments

- 5.1 Experimental Setup

Models and scenarios. We systematically evaluate STARE across three scenarios. In the Short CoT scenario, we use Qwen2.5-Math-7B-Base with a maximum decoding length of 4k, Qwen2.5-14B-Instruct with 8k, and Qwen2.5-32B-Base with 8k(Yang et al., 2024; Qwen et al., 2025). In the Long CoT scenario, we employ DeepSeek-R1-Distill-Qwen-1.5B and Qwen3-8B-Base with 16k to elicit deep reasoning and self-reflection(Guo et al., 2025; Team, 2025). In the tool-use scenario, we first perform cold-start SFT on Qwen2.5-7B-Base using Retool 2K data(Feng et al., 2025), then conduct RL training with 8k length.

Training. We use a learning rate of 1 × 10−6 and a batch size of 64 samples with 8 rollouts per sample, and on-policy updates with a single gradient step per batch. Decoding adopts Top-p=1.0 and temperature T = 1.0. The STARE hyperparameters are set to W = 1.1, M = 0.9, Htgt = 0.3, and P% = 10%, with batch-level gating and fixed weights as the default configuration. The training corpus consists of 100k samples deduplicated and sampled from open-source RL datasets including DeepScaler, Skywork-o1, Polaris, and DAPO(Tan et al., 2026; He et al., 2025c; An et al., 2025; Yu et al., 2025). We construct an

enhanced GRPO baseline, denoted GRPOds, which removes the KL penalty and incorporates dynamic sampling with token-level loss(Yu et al., 2025). We report two vari-

Table 1: Performance comparison of STARE and competitive RL algorithms on six math benchmarks across 1.5B–32B scales and three scenarios (Acc avg@N); † denotes results from prior works.

ants: STARE-O1 amplifies only L+q , whereas STARE-C2 additionally attenuates L−q over O1. Ablation studies adopt STARE-O1 as the default configuration. We append the instruction “Please reason step by step, and put your final answer within \boxed{}” to each question, and then extract the content enclosed in \boxed{} as the final answer for correctness evaluation.

Method AIME24 AIME25 AMC MATH Minerva Olympiad Avg. In the Short CoT Scenario RL from the Qwen2.5-Math-7B-Base W-REINF†(Zhu et al., 2025) 31.2 10.1 58.1 76.2 34.2 38.9 41.6 GRPO†(Shao et al., 2024) 34.0 9.3 58.6 79.9 38.1 42.8 43.8 EntroReg†(Cheng et al., 2025) 34.6 12.6 61.1 82.7 43.1 42.8 46.2 DAPO†(Yu et al., 2025) 35.7 17.1 61.0 82.6 42.8 43.9 47.2 80/20 Rule†(Wang et al., 2025b) 35.8 15.1 63.7 84.1 43.0 45.5 47.9 EntroAdv†(Cheng et al., 2025) 36.8 16.3 63.8 83.5 42.7 44.5 47.9 STEER†(Hao et al., 2025) 36.9 16.2 72.2 82.4 41.7 43.3 49.1

- GRPO-ds 37.1 17.7 75.3 82.7 39.4 42.6 49.1 KL-Cov†(Cui et al., 2025) 38.9 13.8 59.2 81.5 40.9 44.2 46.4 EAPO†(He et al., 2026) 39.8 17.2 62.1 83.7 43.9 45.1 48.6 STARE-O1 44.2 23.8 83.4 86.1 44.2 44.7 54.4 STARE-C2 42.9 24.2 84.1 85.8 44.7 45.3 54.5

RL from the Qwen2.5-14B-Instruct Base-14B†(Qwen et al., 2025) 12.1 11.7 - - - - GRPO†(Shao et al., 2024) 22.5 17.6 - - - - GRPO-Clipν†(Wang et al., 2026) 23.4 21.4 - - - GRPO-ds 24.2 21.9 69.1 80.6 37.2 43.4 46.1 STARE-O1 30.8 27.1 77.5 85.4 40.5 50.9 52.0 STARE-C2 31.5 28.3 76.3 86.1 40.2 51.4 52.3 RL from the Qwen2.5-32B-Base

GRPO†(Shao et al., 2024) 28.5 22.5 - 86.6 44.9 60.3 80/20 Rule†(Wang et al., 2025b) 32.5 28.5 - 89.4 45.6 57.6 GSPO†(Zheng et al., 2025) 33.3 22.3 - 87.6 48.5 55.6 Lp-Reg†(Huang et al., 2025) 38.1 27.1 - 90.0 46.32 61.2 DAPO†(Yu et al., 2025) 38.3 29.8 - 87.6 48.5 55.6 -

- GRPO-ds 38.5 28.8 85.3 85.6 44.6 54.0 56.1 STARE-O1 43.3 34.1 87.3 90.4 48.8 60.1 60.7 STARE-C2 42.9 35.7 88.8 90.6 49.3 60.9 61.4

In the Long CoT Scenario RL from the DeepSeek-R1-Distill-Qwen-1.5B CE-GPPO†(Su et al., 2026) 29.6 23.5 73.5 76.3 26.6 43.9 45.6 Base-1.5B†(Guo et al., 2025) 32.5 24.3 69.4 85.7 37.5 55.2 50.7 EntroReg†(Cheng et al., 2025) 34.6 26.4 70.2 86.8 37.6 56.2 51.9 CISPO†(Chen et al., 2025a) 34.8 25.8 76.9 76.8 26.5 45.8 48.4 W-REINF†(Zhu et al., 2025) 35.0 25.0 69.8 87.5 37.8 55.9 51.8 CE-GPPO†(Su et al., 2026) 35.1 27.7 82.5 76.7 27.8 45.6 49.2 ASPO†(Wang et al., 2025a) 36.4 28.3 83.1 74.6 26.0 44.9 48.9 GRPO†(Shao et al., 2024) 38.1 27.1 71.4 88.9 39.4 58.7 53.9 DAPO†(Yu et al., 2025) 40.9 28.6 73.6 89.9 39.1 58.4 55.1 DGPO†(Fu et al., 2026) 43.3 32.8 86.0 77.9 28.2 48.0 52.7 KL-Cov†(Cui et al., 2025) 43.9 30.1 75.0 90.0 40.5 59.9 56.5 EntroAdv†(Cheng et al., 2025) 44.0 30.4 73.9 90.2 40.4 59.1 56.3 EAPO†(He et al., 2026) 45.1 30.1 75.5 91.1 39.7 60.8 57.0 GRPO-ds 50.4 37.4 85.9 88.3 49.2 64.1 62.5 JustRL†(He et al., 2025b) 52.6 38.8 91.0 91.7 51.5 68.0 65.6 STARE-O1 53.8 41.5 89.5 90.4 52.6 67.8 65.9 STARE-C2 53.1 40.5 91.3 92.1 51.8 68.8 66.3

RL from the Qwen3-8B-Base GRPO†(Shao et al., 2024) 31.3 24.7 75.2 88.9 55.9 61.5 56.2 80/20 Rule†(Wang et al., 2025b) 31.3 27.5 79.9 89.9 54.8 62.5 57.6 STAPO†(Liu et al., 2026) 33.4 28.7 79.9 90.4 57.2 63.0 58.8 DAPO†(Yu et al., 2025) 34.2 26.1 - 84.5 - - Lp-Reg†(Huang et al., 2025) 35.9 25.8 - 87.4 - - A3PO†(Tang et al., 2025) 37.8 30.4 - 91.3 - - -

- GRPO-ds 39.5 30.8 80.8 88.6 52.3 59.0 58.5 STARE-O1 43.9 34.7 85.3 90.6 55.6 61.8 62.0 STARE-C2 44.3 32.6 86.1 91.2 56.9 62.3 62.2

Evaluation. We evaluate on six mathematical benchmarks: AIME24, AIME25, AMC23, MATH-500, Minerva Math, and OlympiadBench(Yang et al., 2023; Hendrycks et al., 2021; Lewkowycz et al., 2022; He et al., 2024), with Top-p = 0.95 and T = 0.7 at inference. AIME24/25 and AMC23 are evaluated 32 times and the other benchmarks 4 times, and all results report average accuracy.

###### 5.2 Main Results

Table 1 presents the full performance comparison, where results marked with † are cited from prior work such as EAPO, STEER, GRPOClipν, Lp-Reg, DGPO, JustRL, STAPO, A3PO, and SimpleTIR(He et al., 2026; Hao et al., 2025; Wang et al., 2026; Huang et al., 2025; Fu et al., 2026; He et al., 2025b; Liu et al., 2026; Tang et al., 2025; Xue et al., 2025). STARE consistently delivers substantial gains across the six math benchmarks at scales ranging from 1.5B to 32B and three scenarios.

Short-CoT Scenario. At the 7B scale, STAREO1 attains an average accuracy of 54.4%, outperforming STEER (49.1%, +5.3%) and GRPOds (49.1%, +5.3%), while reaching 44.2% and 23.8% on AIME24 and AIME25, corresponding to improvements of roughly 10% and 7% over DAPO. At the 14B scale, STARE-O1 (52.0%) surpasses GRPO-ds (46.1%) by 5.9%. At the 32B scale, STARE-O1 (60.7%) exceeds GRPO-ds (56.1%) by 4.6%.

###### In the Tool-Use Agent Scenario

RL from the Qwen2.5-7B-Base Base-7B-TIR†(Qwen et al., 2025) 1.7 0.6 10.8 18.0 - 6.2 ToRL†(Wang et al., 2023) 40.2 27.9 75.0 82.2 - 49.9 Effective TIR†(Bai et al., 2025) 42.3 29.2 74.2 86.4 - - ZeroTIR†(Mai et al., 2025) 46.7 30.0 - 85.2 - GRPO-ds 46.8 32.4 75.9 81.4 38.3 48.8 53.9 SimpleTIR†(Xue et al., 2025) 50.5 30.9 79.1 88.4 - 54.8 STARE-O1 53.2 37.5 84.9 86.8 41.9 52.3 59.4 STARE-C2 52.8 38.1 86.9 87.2 43.7 53.6 60.4

Long-CoT Scenario. At the 1.5B scale, STARE-O1 reaches an average of 65.9%, substantially exceeding EAPO (57.0%, +8.9%) and DAPO (55.1%, +10.8%), while also outperforming JustRL on AIME24 and AIME25. At the 8B scale, STARE-O1 (62.0%) surpasses STAPO (58.8%) and GRPO-ds (59.0%), and STARE-C2 further raises the average to 62.2%.

Tool-Use Scenario. STARE-O1 achieves an average of 59.4%, improving over GRPO-ds (53.9%) by 5.5% and outperforms SimpleTIR, while reaching 53.2% and 37.5% on AIME24 and AIME25. STARE-C2 further lifts the average to 60.4%.

Key findings. Three principal observations emerge. (i) On the challenging AIME24 and AIME25 benchmarks, STARE improves over GRPO-ds by 4%–8% in average accuracy, with gains of 3%–6% across all six benchmarks. (ii) Across different thinking scenarios and model scales from 1.5B to 32B, STARE consistently outperforms the majority of competitive RL improvement methods, confirming its effectiveness and robustness. (iii) The additional gains of STARE-C2 further indicate that dualsided regulation, which simultaneously strengthens entropy-increasing signals and attenuates entropydecreasing ones, can yield an even more favorable exploration-exploitation balance.

GRPO-ds

STARE

0.4

Entropy

0.2

0.0

0 1000 2000 3000 4000 5000

Training Steps

(a) Training Entropy

0.45

AIME24Acc

0.30

0.15

GRPO-ds

STARE

0.00

0 1000 2000 3000 4000 5000

Training Steps

(e) AIME24 Acc

| | |
|---|---|
| | |
| |GRPO-ds<br><br>STARE|

0.8

TrainingReward

0.6

0.4

0.2

0.0

0 1000 2000 3000 4000 5000

Training Steps

(b) Training Reward

0.25

0.20

AIME25Acc

0.15

0.10

GRPO-ds

0.05

STARE

0 1000 2000 3000 4000 5000

Training Steps

(f) AIME25 Acc

0.8

TrainingFull-SolveRatio

0.6

0.4

0.2

GRPO-ds

STARE

0.0

0 1000 2000 3000 4000 5000

Training Steps

(c) Train Full-solve Ratio

0.7

AIME24Pass@32

0.6

0.5

GRPO-ds

0.4

STARE

0 1000 2000 3000 4000 5000

Training Steps

(g) AIME24 Pass@32

2000

|GRPO-ds| |
|---|---|
|STARE| |
| | |

TrainingResponse

1800

1600

1400

1200

1000

800

0 1000 2000 3000 4000 5000

Training Steps

(d) Train Response Length

0.45

AIME25Pass@32

0.40

0.35

0.30

0.25

GRPO-ds

STARE

0.20

0 1000 2000 3000 4000 5000

Training Steps

(h) AIME25 Pass@32

- Figure 3: Comparison of key training metrics between STARE and GRPO-ds on Qwen2.5-Math-7B-Base model in the Short CoT scenario over 5k RL steps.

0 200 400 600 800 1000

Training Steps

- 0

2

- 4

6

8

Entropy

GRPO-ds

STARE: W=1.01 STARE: W=1.05

- STARE: W=1.1

- STARE: W=1.2

STARE: W=1.5

- STARE: W=2.0

- STARE: W=3.0

- STARE: W=4.0

(a) Entropy evolution under varying W without target-entropy gating

0 200 400 600 800 1000

Training Steps

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

Entropy

GRPO-ds

STARE: W=1.01 STARE: W=1.05

- STARE: W=1.1

- STARE: W=1.2

STARE: W=1.5

- STARE: W=2.0

- STARE: W=3.0

- STARE: W=4.0

(b) Entropy evolution under varying W with target-entropy gating

0 200 400 600 800 1000

Training Steps

0.0

0.1

0.2

0.3

0.4

0.5

0.6

Entropy

GRPO-ds

STARE: Top 5%

STARE: Top 10%

STARE: Top 15% STARE: Top 20% STARE: Top 25%

STARE: Top 30% STARE: Top 35% STARE: Top 40%

(c) Entropy evolution under varying Top-P ratio (W : 1.1, Htgt : 0.3)

Figure 4: STARE policy entropy evolution under ablation on the varying reweighting factor W, targetentropy gating, and the high-surprisal selection ratio P on Qwen2.5-Math-7B-Base.

- 5.3 Cognitive Analysis

STARE vs. GRPO-ds: Training Dynamics across Scales and Scenarios. To validate STARE in longhorizon RL, we run 5000 training steps on Qwen2.5-Math-7B-Base under the Short CoT scenario. Figure 3 compares STARE with GRPO-ds on key metrics, and Figures 5–1 further verify its effectiveness across

- 1.5B–32B scales and diverse task scenarios. Entropy stability and performance evolution. GRPO-ds exhibits entropy collapse over steps 0–1000, with policy entropy approaching zero (Figure 3(a)), consistent with Section 3; correspondingly, its AIME24/25 accuracy peaks around step 1000 and saturates thereafter

(Figure 3(e)-(f)), indicating premature convergence. In contrast, STARE stabilizes entropy near Htgt = 0.3 via token-level reweighting and closed-loop gating, with accuracy continuing to rise beyond step 1000 and peaking at 5000, thereby unlocking long-horizon RL potential. Exploration-exploitation balance. STARE’s Pass@32 consistently exceeds GRPO-ds throughout training (Figure 3(g)-(h)), preserving output diversity and mitigating mode-seeking; whereas GRPO-ds’s reward and Full-Solve Ratio plateau early (Figure 3(b)-(c)), STARE keeps growing, and its sustained response-length increase (Figure 3(d)) reflects deeper reasoning. Cross-scale and cross-scenario generalization. Figures 5-1 reveal a consistent pattern across Short CoT (14B, 32B), Long CoT (R1-Distill-Qwen-1.5B, Qwen3-8B-Base), and Tool-Use (7B): GRPOds suffers collapse with performance saturation, while STARE maintains stable entropy and continuous accuracy gains. Notably, on DeepSeek-R1-Distill-Qwen-1.5B, GRPO-ds’s entropy decays below 0.2 by step 5000, whereas STARE rapidly restores entropy to the target band starting around step 3500, with AIME24/25 accuracy improving in tandem (Figure 7), showing its intervention-recovery capability and robustness across scales and scenarios. Further details are provided in Appendix B.1

Ablation on Key Hyperparameters and Target-Entropy Gating. We ablate W, P, and the target-entropy gate on Qwen2.5-Math-7B-Base (Figure 4). Under open-loop reweighting ( without the target-entropy gate; Figure 4(a)), W = 1.01 already mitigates the entropy decay of GRPO-ds, W ≥ 1.05 yields steady growth, and W ≥ 2.0 triggers divergenc, corroborating the near-criticality property (Corollary 3.6) that beyond the critical threshold W controls magnitude rather than direction. Open-loop reweighting, however, stabilizes entropy at an excessively high level, inducing over-exploration that hampers overall training(Appendix B.5). With the closed-loop gate (Htgt = 0.3), Figure 4(b) shows that all W ∈ [1.05,1.5] steer entropy into the target band with bounded oscillation, confirming closed-loop stability and

- 0.0

- 0.1

- 0.2

- 0.3

- 0.4

- 0.5

GRPO-ds

STARE

Entropy

0 500 1000 1500

Training Steps

(a) Training Entropy

0.30

AIME24Acc

0.25

0.20

0.15

GRPO-ds

STARE

0.10

0 500 1000 1500

Training Steps

(b) AIME24 Acc

0.24

AIME25Acc

0.20

0.16

0.12

GRPO-ds

STARE

0.08

0 500 1000 1500

Training Steps

(c) AIME25 Acc

- Figure 5: Training dynamics of STARE vs. GRPO-ds on Qwen2.5-14B-Instruct in the Short CoT scenario: policy entropy, AIME24 accuracy, and AIME25 accuracy.

0 500 1000 1500

Training Steps

0.0

0.2

0.4

0.6

0.8

Entropy

GRPO-ds

STARE

(a) Training Entropy

0 500 1000 1500

Training Steps

0.1

0.2

0.3

0.4

AIME24Acc

GRPO-ds

STARE

(b) AIME24 Acc

0 500 1000 1500

Training Steps

0.1

0.2

0.3

AIME25Acc

GRPO-ds

STARE

(c) AIME25 Acc

- Figure 6: Training dynamics of STARE vs. GRPO-ds on Qwen2.5-32B-Base in the Short CoT scenario: policy entropy, AIME24 accuracy, and AIME25 accuracy.

0 1000 2000 3000 4000 5000

Training Steps

0.2

0.4

0.6

0.8

1.0

Entropy

GRPO-ds

STARE

(a) Training Entropy

0 1000 2000 3000 4000 5000

Training Steps

0.3

0.4

0.5

AIME24Acc

GRPO-ds

STARE

(b) AIME24 Acc

0 1000 2000 3000 4000 5000

Training Steps

0.20

0.25

0.30

0.35

0.40

AIME25Acc

GRPO-ds

STARE

(c) AIME25 Acc

- Figure 7: Training dynamics of STARE vs. GRPO-ds on DeepSeek-R1-Distill-Qwen-1.5B in the Long CoT scenario: policy entropy, AIME24 accuracy, and AIME25 accuracy.

0 200 400 600 800 1000

Training Steps

0.0

0.2

0.4

0.6

0.8

1.0

Entropy

GRPO-ds

STARE

(a) Training Entropy

0 200 400 600 800 1000

Training Steps

0.1

0.2

0.3

0.4

AIME24Acc

GRPO-ds

STARE

(b) AIME24 Acc

0 200 400 600 800 1000

Training Steps

0.10

0.15

0.20

0.25

0.30

0.35

AIME25Acc

GRPO-ds

STARE

(c) AIME25 Acc

- Figure 8: Training dynamics of STARE vs. GRPO-ds on Qwen3-8B-Base in the Long-CoT scenario: policy entropy, AIME24 accuracy, and AIME25 accuracy.

substantially reducing sensitivity to W. Fixing W = 1.1 and Htgt = 0.3, Figure 4 (c) further shows that entropy stays within the target band for P ∈ [5%,20%] and remains confined to [0.1,0.2] even at P = 40%, effectively preventing collapse and indicating a broad operating range. Overall, STARE is robust to W and P, and the target-entropy gate stably constrains the policy entropy. More detailed entropy evolution and performance comparisons are reported in Appendix B.5 ( Table 3, Figure 12), with additional analysis provided in Appendix B.4.

0.6

| | | |
|---|---|---|
| |GRPO-ds<br><br>STARE-O1<br><br>STAR STAR<br><br>|E-O2<br>E-O3<br><br><br>STARE-O4|
| | | |
| | | |

0.4

Entropy

0.2

0.0

0 1000 2000 3000 4000

Training Steps

(a) Entropy evolution under single-polarity operations O1-O4, each targeting one entropy-critical quadrant.

0.6

| | |
|---|---|
|GRPO-ds STARE-C1<br><br>ST ST|ARE-C2 ARE-C3<br><br>STARE-C4|
| | |
| | |
| | |

0.4

Entropy

0.2

0.0

0 1000 2000 3000 4000

Training Steps

(b) Entropy curves under combined operations C1-C4, each jointly regulating two entropy-critical quadrants.

- Figure 9: Policy entropy trajectories of all eight STARE variants versus GRPO-ds on Qwen2.5-Math-7BBase over 4000 RL training steps, covering four single-polarity operations (O1-O4) and four combined operations (C1-C4) derived from the advantage-surprisal four-quadrant decomposition.

Validation of the High-Surprisal Quantile Proxy. Figure 11 validates the batch-internal top-P% surprisalquantile proxy. Panel (a) shows that the fraction of selected tokens falling within the theoretical entropy-

increasing region (si,t > si∗,t) rises steadily from ∼60% to ∼95% during training, indicating strong consistency with the critical threshold s∗ (Proposition 3.2); panel (b) further confirms that the cumulative

net entropy contribution of L+q remains positive and monotonically increasing, consistent with Corollary 3.3. Thus the proxy reliably identifies entropy-critical tokens while obviating per-position solutions of Φ(p∗) = 0; see Appendix B.2 for further details.

Table 2: The results of single-polarity and combined STARE operations on AIME24 and AIME25.

Effects of Single-Polarity and Combined Operations in STARE. To validate the token-level reweighting mechanism under the four-quadrant decomposition, we ablate four singlepolarity (O1–O4) and four combined (C1–C4) operations, where O1/O3 amplify entropy-increasing signals, O2/O4 attenuate entropy-decreasing ones, and C1–C4 jointly intervene on two quadrants. As shown in Table 2 and Figure 9, while GRPOds suffers rapid entropy decay, all eight variants effectively mitigate this collapse and substantially outperform GRPO-ds AIME24/25 across all four entropy-critical quadrants. Among

Model AIME24 AIME25 GRPO-ds 37.1 17.7

- STARE-O1 44.2 23.8
- STARE-O2 40.5 20.3
- STARE-O3 39.6 21.6
- STARE-O4 42.1 19.9

them, O1 (amplifying L+q ) and C2 (amplifying L+q while attenuating L−q ) perform best at 44.2%/23.8% and 42.5%/24.2%, so we adopt STARE-O1 as the default configuration and STAREC2 as the dual-sided variant. Further details are provided in Appendix B.3.

STARE-C1 43.1 23.5 STARE-C2 42.5 24.2 STARE-C3 39.9 20.8 STARE-C4 41.7 22.6

Emergent Reflection Behaviors. To examine how STARE elicits deep reasoning, we inspect the tokens selected for advantage amplification during RL training. As exemplified on Qwen2.5-32B-Base, the word cloud in Figure 10 shows reweighted tokens concentrating on uncertainty and selfcorrection vocabulary, such as should be, but, instead, and verification, confirming that the batch-internal surprisal-quantile proxy effectively identifies rare forking tokens (Wang et al., 2025b) with exploratory semantics. A complementary count across six reflection categories (Figure 13) further shows STARE markedly surpasses GRPO-ds with the largest margins on reflection and self-correction, jointly demonstrating that STARE activates deep exploration and delivers consistent gains through token-level credit rebalancing. The complete analysis is deferred to Appendix B.6.

[Figure 2]

Figure 10: Word cloud of tokens selected by STARE for advantage reweighting.

Further ablations are deferred to the appendices: fixed vs. adaptive weighting( B.7), target-entropy threshold( B.8), gating granularity( B.9), off-policy training( B.10), and fixed-threshold reweighting( B.11).

###### 6 Conclusion

We present STARE, a surprisal-guided token-level advantage reweighting mechanism that mitigates policy entropy collapse in GRPO-style RLVR. A first-order analysis of entropy dynamics reveals an advantage–surprisal four-quadrant structure and a near-criticality property. Guided by this insight, STARE reweights entropy-critical tokens via batch-internal surprisal quantiles, coupled with a targetentropy closed-loop gate for stable, minimally intrusive regulation. Across 1.5B–32B models and three task scenarios, STARE sustains thousands of stable RL steps and outperforms DAPO and other competitive baselines by 4%–8% on AIME24/25, offering a principled foundation for entropy-aware credit assignment in long-horizon RL post-training of LLMs.

###### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, et al. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https://hkunlp. github. io/blog/2025/Polaris, 2025.

AI Anthropic. System card: Claude opus 4 & claude sonnet 4. Claude-4 Model Card, 2025. Fei Bai, Yingqian Min, Beichen Zhang, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Zheng Liu, Zhongyuan

Wang, and Ji-Rong Wen. Towards effective code-integrated reasoning, 2025. URL https://arxiv.org/ abs/2505.24480.

Edward Y. Chang, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. CoRR, abs/2502.03373, 2025. doi: 10.48550/ARXIV.2502.03373. URL https://doi.org/10.48550/arXiv.2502.03373.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025a.

Kun Chen, Peng Shi, Fanfan Liu, Haibo Qiu, Zhixiong Zeng, Siqi Yang, and Wenji Mao. Flexible entropy control in rlvr with gradient-preserving perspective. arXiv preprint arXiv:2602.09782, 2026.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do NOT think that much for 2+3=? on the overthinking of long reasoning models. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net, 2025b. URL https://openreview.net/forum?id=MSbU3L7V00.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. CoRR, abs/2506.14758, 2025. doi: 10.48550/ ARXIV.2506.14758. URL https://doi.org/10.48550/arXiv.2506.14758.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models. CoRR, abs/2505.22617, 2025. doi: 10.48550/ARXIV.2505.22617. URL https://doi.org/10.48550/arXiv. 2505.22617.

Wenlong Deng, Yi Ren, Muchen Li, Danica J. Sutherland, Xiaoxiao Li, and Christos Thrampoulidis. On the effect of negative gradient in group relative deep reinforcement optimization. CoRR, abs/2505.18830,

2025. doi: 10.48550/ARXIV.2505.18830. URL https://doi.org/10.48550/arXiv.2505.18830.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nat., 630(8017):625–630, 2024. doi: 10.1038/S41586-024-07421-0. URL https://doi.org/10.1038/s41586-024-07421-0.

Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms, 2025. URL https://arxiv.org/abs/2504.11536.

Xiaoliang Fu, Jiaye Lin, Yangyi Fang, Chaowen Hu, Cong Qin, Zekai Shao, Binbin Zheng, Lu Pan, and Ke Zeng. From log π to π: Taming divergence in soft clipping via bilateral decoupled decay of probability gradient weight, 2026. URL https://arxiv.org/abs/2603.14389.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, pp. 10835–10866, 2023. URL https://proceedings.mlr.press/v202/gao23h.html.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, pp. 1856–1865, 2018. URL http://proceedings.mlr.press/v80/haarnoja18b.html.

Zhezheng Hao, Hong Wang, Haoyang Liu, Jian Luo, Jiarui Yu, Hande Dong, Qiang Lin, Can Wang, and Jiawei Chen. Rethinking entropy interventions in rlvr: An entropy change perspective. arXiv preprint arXiv:2510.10150, 2025.

Andre Wang He, Daniel Fried, and Sean Welleck. Rewarding the unlikely: Lifting grpo beyond distribution sharpening. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 25559–25571, 2025a.

Bingxiang He, Zekai Qu, Zeyuan Liu, Yinghao Chen, Yuxin Zuo, Cheng Qian, Kaiyan Zhang, Weize Chen, Chaojun Xiao, Ganqu Cui, Ning Ding, and Zhiyuan Liu. Justrl: Scaling a 1.5b llm with a simple rl recipe, 2025b. URL https://arxiv.org/abs/2512.16649.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 3828–3850, 2024. doi: 10.18653/V1/2024.ACL-LONG.211. URL https://doi.org/10.18653/v1/2024.acl-long.211.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025c.

Yuhang He, Haodong Wu, Siyi Liu, Hongyu Ge, Hange Zhou, Keyi Wu, Zhuo Zheng, Qihong Lin, Zixin Zhong, and Yongqi Zhang. Rethinking token-level credit assignment in rlvr: A polarity-entropy analysis. arXiv preprint arXiv:2604.11056, 2026.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Guanhua Huang, Tingqiang Xu, Mingze Wang, Qi Yi, Xue Gong, Siheng Li, Ruibin Xiong, Kejiao Li, Yuhao Jiang, and Bo Zhou. Low-probability tokens sustain exploration in reinforcement learning with verifiable reward. arXiv preprint arXiv:2510.03222, 2025.

Renren Jin, Pengzhi Gao, Yuqi Ren, Zhuowen Han, Tongxuan Zhang, Wuwei Huang, Wei Liu, Jian Luan, and Deyi Xiong. Revisiting entropy in reinforcement learning for large reasoning models. arXiv preprint arXiv:2511.05993, 2025.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron C. Courville, and Nicolas Le Roux. Vineppo: Refining credit assignment in RL training of llms. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, 2025. URL https://openreview.net/forum?id=Myx2kJFzAn.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html.

Shiqi Liu, Zeyu He, Guojian Zhan, Letian Tao, Zhilong Zheng, Jiang Wu, Yinuo Wang, Yang Guan, Kehua Sheng, Bo Zhang, et al. Stapo: Stabilizing reinforcement learning for llms by silencing rare spurious tokens. arXiv preprint arXiv:2602.15620, 2026.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Qingwei Lin, Jianguang Lou, Shifeng Chen, Yansong Tang, and Weizhu Chen. Arena learning: Build data flywheel for llms post-training via simulated chatbot arena. arXiv preprint arXiv:2407.10627, 2024a.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Qingwei Lin, Jianguang Lou, Shifeng Chen, Yansong Tang, and Weizhu Chen. Wizardarena: Post-training large language models via simulated offline chatbot arena. Advances in Neural Information Processing Systems, 37:111544–111570, 2024b.

Haipeng Luo, Huawen Feng, Qingfeng Sun, Can Xu, Kai Zheng, Yufei Wang, Tao Yang, Han Hu, and Yansong Tang. Agentmath: Empowering mathematical reasoning for large language models via tool-augmented agent. arXiv preprint arXiv:2512.20745, 2025.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evolinstruct. In International Conference on Learning Representations, volume 2024, pp. 27168–27188, 2024c.

Xinji Mai, Haotian Xu, Zhong-Zhi Li, Weinong Wang, Jian Hu, Yingying Zhang, Wenqiang Zhang, et al. Agent rl scaling law: Agent rl with spontaneous code execution for mathematical problem solving. arXiv preprint arXiv:2505.07773, 2025.

Byung-Doh Oh and William Schuler. Transformer-based language model surprisal predicts human reading times best with about two billion training tokens. In Findings of the association for computational linguistics: EMNLP 2023, pp. 1915–1921, 2023.

Byung-Doh Oh, Shisen Yue, and William Schuler. Frequency explains the inverse correlation of large language models’ size, training data amount, and surprisal’s fit to reading times, 2024. URL https: //arxiv.org/abs/2402.02255.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27

(3):379–423, 1948.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Qiannan Shen and Jing Zhang. Ai-enhanced disaster risk prediction with explainable shap analysis: A multi-class classification approach using xgboost. In 2026 5th International Symposium on Computer Applications and Information Technology (ISCAIT), pp. 692–698. IEEE, 2026.

Nathaniel J Smith and Roger Levy. The effect of word predictability on reading time is logarithmic. Cognition, 128(3):302–319, 2013.

Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, and Guorui Zhou. Ce-gppo: Coordinating entropy via gradient-preserving clipping policy optimization in reinforcement learning, 2026. URL https://arxiv.org/abs/2509.20712.

Sijun Tan, Michael Luo, Justin Wong, Colin Cai, Xiaoxiang Shi, William Yuan Tang, Manan Roongta, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Effective RL scaling of reasoning models via iterative context lengthening, 2026. URL https://openreview.net/forum?id= I6GzDCne7U.

Xinyu Tang, Yuliang Zhan, Zhixun Li, Wayne Xin Zhao, Zhenduo Zhang, Zujie Wen, Zhiqiang Zhang, and Jun Zhou. Rethinking sample polarity in reinforcement learning with verifiable rewards. arXiv preprint arXiv:2512.21625, 2025.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms, 2025a. URL https://arxiv.org/abs/2501.12599.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Tencent Hunyuan Team, Ao Liu, Botong Zhou, Can Xu, Chayse Zhou, ChenChen Zhang, Chengcheng Xu, Chenhao Wang, Decheng Wu, Dengpeng Wu, et al. Hunyuan-turbos: Advancing large language models through mamba-transformer synergy and adaptive chain-of-thought. arXiv preprint arXiv:2505.15431, 2025b.

Jiakang Wang, Runze Liu, Lei Lin, Wenping Hu, Xiu Li, Fuzheng Zhang, Guorui Zhou, and Kun Gai. Aspo: Asymmetric importance sampling policy optimization, 2025a. URL https://arxiv.org/abs/ 2510.06062.

Kezhou Wang, Ruijie Wu, Qinlin Zeng, Huao Lu, Hanye Wu, Qingfeng Cui, Haichao Lin, Yujia Liu, Xiaoyan Huang, Qingpeng Guo, Songtao Jian, Kaiyuan Lu, Shiyu Li, Hao Tian, Yongqin Sun, Xue Yang, Libin Song, Zejun Ou, and Guoqing Wang. ToRL: Scaling tool-integrated RL for LLMs. arXiv preprint arXiv:2312.10372, 2023.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. CoRR, abs/2506.01939, 2025b. doi: 10.48550/ARXIV.2506. 01939. URL https://doi.org/10.48550/arXiv.2506.01939.

Shumin Wang, Yuexiang Xie, Wenhao Zhang, Yuchang Sun, Yanxi Chen, Yaliang Li, and Yanyong Zhang. On the entropy dynamics in reinforcement fine-tuning of large language models. arXiv preprint arXiv:2602.03392, 2026.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, et al. Bapo: Stabilizing off-policy reinforcement learning for llms via balanced policy optimization with adaptive clipping. arXiv preprint arXiv:2510.18927, 2025.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In International Conference on Learning Representations, volume 2024, pp. 30745–30766, 2024.

Wujiang Xu, Wentian Zhao, Zhenting Wang, Yu-Jhe Li, Can Jin, Mingyu Jin, Kai Mei, Kun Wang, and Dimitris N. Metaxas. EPO: entropy-regularized policy optimization for LLM agents reinforcement learning. CoRR, abs/2509.22576, 2025. doi: 10.48550/ARXIV.2509.22576. URL https://doi.org/10. 48550/arXiv.2509.22576.

Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, and Bo An. Simpletir: Endto-end reinforcement learning for multi-turn tool-integrated reasoning. arXiv preprint arXiv:2509.02479, 2025.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via selfimprovement, 2024. URL https://arxiv.org/abs/2409.12122.

Kai Yang, Xin Xu, Yangkun Chen, Weijie Liu, Jiafei Lyu, Zichuan Lin, Deheng Ye, and Saiyong Yang. Entropic: Towards stable long-term training of llms via entropy stabilization with proportional-integral control. arXiv preprint arXiv:2511.15248, 2025a.

Tao Yang, Xiaopu Zhang, Junjie Xiao, Jianzhun Qian, Ning Bian, Jingwei Jia, Boyuan Xie, Huy Manh, Tianyi Zhou, Hongchang Zheng, Zihang Song, Yongqi Yang, Kaihang Liu, Wenhu Huang, Jinxi Guo, Zhilin Xu, Jie Liu, Hao Du, Zhou Zhang, Yicheng Zheng, Zheng Haotian, Xintao Wei, Hua Wu, Qi Liu, and Dong Zhou. AIMO-2 winning solution: Building state-of-the-art mathematical reasoning models with OpenMathReasoning dataset. arXiv preprint arXiv:2307.14047, 2023.

Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, and Yunjian Xu. Do not let low-probability tokens over-dominate in RL for llms. CoRR, abs/2505.12929, 2025b. doi: 10.48550/ARXIV.2505.12929. URL https://doi.org/10.48550/arXiv.2505.12929.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? CoRR, abs/2504.13837, 2025a. doi: 10.48550/ARXIV.2504.13837. URL https://doi.org/10.48550/ arXiv.2504.13837.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025b.

Wenhao Zeng, Yaoning Wang, Chao Hu, Yuling Shi, Chengcheng Wan, Hongyu Zhang, and Xiaodong Gu. Pruning the unsurprising: Efficient llm reasoning via first-token surprisal, 2026. URL https: //arxiv.org/abs/2508.05988.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Yuhang Zhou, Kai Zheng, Qiguang Chen, Mengkang Hu, Qingfeng Sun, Can Xu, and Jingjing Chen. Offseeker: Online reinforcement learning is not all you need for deep research agents. arXiv preprint arXiv:2601.18467, 2026.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in llm reasoning. ArXiv, abs/2506.01347, 2025. URL https: //api.semanticscholar.org/CorpusID:279075301.

###### Appendix Appendix Contents

- A Related Work 18
- B Additional Experiments and Analysis on STARE 19

- B.1 STARE vs. GRPO-ds: Detailed Comparison across Diverse Scenarios and Model Scales in RL Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.2 Validation of the High-Surprisal Quantile Proxy. . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.3 Detailed Effects of Single-Polarity and Combined Operations in STARE. . . . . . . . . . . 20
- B.4 Detailed Ablation on Key Hyperparameters and Target-Entropy Gating . . . . . . . . . . 21
- B.5 Effectiveness of Target-Entropy Closed-Loop Gating . . . . . . . . . . . . . . . . . . . . . . 21
- B.6 Details about Emergent Reflection Behaviors. . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- B.7 Fixed vs. Adaptive Weights . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.8 Ablation on Target Entropy Threshold . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.9 Ablation on Target-Entropy Gate Granularity . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.10 Validation of STARE under Off-Policy Training . . . . . . . . . . . . . . . . . . . . . . . . . 24
- B.11 STARE vs. Fixed-Threshold Low-Probability Token Reweighting . . . . . . . . . . . . . . 24

- C Algorithm: Main STARE Procedure 26
- D Basic Derivations for Sections 2 and 3.1 27

- D.1 Softmax Jacobian Derivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- D.2 Token-Level Logit Update in the Unclipped GRPO Regime . . . . . . . . . . . . . . . . . . 27
- D.3 Lemma 2.1 (Entropy Gradient with Respect to Logits: Surprisal-Deviation Form) . . . . . 28
- D.4 Theorem 3.1 (Token-Level Entropy Variation) . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- E Complete Proofs for Sections 3.2 and 3.3 and Near-Criticality Analysis 30

- E.1 Positivity of S2 under Non-Uniform Distributions . . . . . . . . . . . . . . . . . . . . . . . 30
- E.2 H > S2 under Non-Degenerate Distributions . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- E.3 Proposition 3.2 (Uniqueness of the Critical Surprisal Threshold) . . . . . . . . . . . . . . . 31
- E.4 Corollary 3.3 (Four-Quadrant Decomposition) . . . . . . . . . . . . . . . . . . . . . . . . . 32
- E.5 Asymmetric Entropy Contributions under Shared Trajectory-Level Advantages . . . . . . 33
- E.6 Theorem 3.4 (Entropy Neutrality Identity) . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- E.7 Proposition 3.5 (Entropy Gradient under Token-Level Reweighting) . . . . . . . . . . . . . 34
- E.8 Two Auxiliary Bounds: Lower Bound in the High-Surprisal Region and Upper Bound in the Low-Surprisal Region . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- E.9 Statistical Assumptions for the Near-Criticality Analysis . . . . . . . . . . . . . . . . . . . 36
- E.10 Λ Is a Weak Residual . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- E.11 Theorem 3.6 (Near-Criticality) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38

- F Formalization of the Cross-Step Entropy Dynamics in Section 3.4 39

- F.1 Mean-Field Approximation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- F.2 Sign of the One-Step Batch Entropy Change under a Fixed Weight . . . . . . . . . . . . . . 39

- F.3 Open-Loop Self-Reinforcing Entropy Collapse and Recovery . . . . . . . . . . . . . . . . . 40
- F.4 Local Stability of Batch-Level Target-Entropy Gating . . . . . . . . . . . . . . . . . . . . . . 40

###### G Single-Polarity Operations and Finer-Grained Closed-Loop Extensions 41

- G.1 Definition of Surprisal-Quantile Proxy Sets . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- G.2 Position-Level Logit Update under STARE Reweighting . . . . . . . . . . . . . . . . . . . . 42
- G.3 Exact Batch-Level Entropy Variation under Reweighting an Arbitrary Token Subset . . . 42
- G.4 Four Single-Polarity Operations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- G.5 Exact Entropy Variation under Unified Closed-Loop Gating . . . . . . . . . . . . . . . . . 44
- G.6 Unified Instantiations of Three Closed-Loop Granularities . . . . . . . . . . . . . . . . . . 45

###### H Combined Reweighting Operations and Adaptive Weights 45

- H.1 Exact Entropy Variation for Arbitrary Two-Subset Reweighting . . . . . . . . . . . . . . . 45
- H.2 Four Representative Combined Reweighting Operations . . . . . . . . . . . . . . . . . . . 45
- H.3 Summary of Single-Polarity and Combined Operations . . . . . . . . . . . . . . . . . . . . 47

###### I Limitations and Broader Impacts 48

###### Appendix Overview

The appendices are organized as follows. Appendix A reviews related work on reinforcement learning with verifiable rewards, existing entropy-collapse mitigation strategies, and emerging token-level perspectives, situating STARE within these literatures and clarifying its conceptual and methodological distinctions from prior approaches. Appendix B reports additional empirical results that complement the main paper, including a detailed comparison between STARE and GRPO-ds across diverse scenarios and model scales from 1.5B to 32B (Appendix B.1), an empirical validation of the batch-internal high-surprisal quantile proxy against the theoretical critical threshold (Appendix B.2), ablations of all four single-polarity and four combined token-level reweighting operations (Appendix B.3), ablations on the key hyperparameters W and P together with the target-entropy gate (Appendices B.4 and B.5), an analysis of emergent reflection behaviors elicited by STARE (Appendix B.6), a comparison between fixed and adaptive reweighting schedules (Appendix B.7), ablations on the target-entropy threshold and on the gate granularity (Appendices B.8 and B.9), a validation of STARE under off-policy training (Appendix B.10), and a comparison with fixed-threshold low-probability token reweighting (Appendix B.11). Appendix C presents the complete pseudocode of the default STARE-O1 procedure, integrating surprisalguided entropy-critical token selection, fixed-weight token-level advantage reweighting, and batch-level closed-loop target-entropy gating into a unified algorithm. Appendix D establishes the basic differential identities used in Sections 2 and 3.1 and proves Theorem 3.1. Appendix E proves Proposition 3.2, Corollary 3.3, Theorem 3.4, Proposition 3.5, and Theorem 3.6, provides the detailed asymmetric entropy contribution analysis, and states explicitly the statistical assumptions underlying the near-criticality result. Appendix F formalizes the cross-step entropy dynamics discussed in Section 3.4. Appendix G presents all single-polarity operations referenced in the main text, together with sample-level and token-level closed-loop extensions. Appendix H presents all combined operations and the adaptive weighting scheme. Finally, Appendix I discusses the limitations of our study and its broader societal impacts.

To keep the notation unambiguous, Appendices D–F use the exact theoretical sets from Section 3,

###### L+ ≜ {(i, t) : Aˆi > 0, si,t > si∗,t}, L− ≜ {(i, t) : Aˆi < 0, si,t > si∗,t},

namely, the sets of token positions whose trajectories have positive (resp. negative) advantage and whose token surprisal exceeds the position-dependent critical threshold. By contrast, Appendices G and H use

the quantile-based proxy sets employed in the practical implementation of Section 4, namely L+q and L−q .

###### A Related Work

Reinforcement Learning with Verifiable Rewards(RLVR). In recent years, large language models (LLMs) have advanced rapidly(Achiam et al., 2023; Anthropic, 2025; Guo et al., 2025; Team et al., 2025b; Xu et al., 2024; Luo et al., 2024c). RLVR has emerged as the dominant post-training paradigm for enhancing the reasoning capability of LLMs, as it leverages verifiable signals to provide precise outcome-level rewards while avoiding the overfitting risks inherent in learned reward models(Schulman et al., 2017; Kazemnejad et al., 2025; Gao et al., 2023). GRPO removes the value network and estimates the advantage through group-normalized rewards, demonstrating strong effectiveness in mathematical reasoning, code generation, and tool-use tasks, and further eliciting emergent behaviors such as long chain-ofthought reasoning and self-reflection(Guo et al., 2025; Shao et al., 2024; Luo et al., 2023; Team et al., 2025b; Wei et al., 2022; Luo et al., 2024a; Chang et al., 2025; Luo et al., 2024b; Shen & Zhang, 2026; Luo

- et al., 2025). Subsequent studies extend the GRPO framework along several dimensions, including advantage estimation, loss aggregation, and sampling strategies, among which DAPO has become a representative baseline through a combination of asymmetric clipping, dynamic sampling, and tokenlevel loss normalization(Yu et al., 2025; Yue et al., 2025b; Cheng et al., 2025; He et al., 2025c; Chen
- et al., 2026). As training proceeds over more optimization steps, however, GRPO-style algorithms commonly suffer from policy entropy collapse, in which the entropy decays rapidly, the output diversity vanishes, within-group rollouts become homogeneous, and the number of trainable steps is ultimately capped(Farquhar et al., 2024; Jin et al., 2025; Chen et al., 2025b; Yue et al., 2025a). Existing studies have confirmed the prevalence of this phenomenon and have established empirical correlations between policy entropy and downstream performance; nevertheless, a fine-grained theoretical characterization of the underlying token-level gradient causes of entropy collapse remains absent.

Entropy Collapse Mitigation Methods.Existing mitigation strategies fall into three categories. The first category protects low-probability tokens by adjusting the clipping thresholds of the importancesampling ratio, as exemplified by the clip-higher mechanism in DAPO and by subsequent variants such as differentiated clipping and smooth gating; the influence of these mechanisms on entropy is asymmetric and difficult to control precisely, and in the on-policy regime where the sampling ratio remains close to one, clipping is rarely activated, leaving the actual regulatory capacity limited(Yu et al.,

2025; Yue et al., 2025b; Haarnoja et al., 2018; Chen et al., 2026; Xi et al., 2025; Zhou et al., 2026). The second category applies trajectory-level differentiated weighting between positive and negative samples, including asymmetric importance sampling and the upweighting of rare correct rollouts; since these methods operate at the trajectory granularity, they cannot distinguish the opposing entropy effects of different tokens within the same trajectory(Zhu et al., 2025; Tang et al., 2025; Wang et al., 2025b; Deng et al., 2025; Yang et al., 2025a). The third category couples token-level entropy information into the advantage or loss through entropy-induced advantages(Cheng et al., 2025; Huang et al., 2025; He et al., 2025c; Cui et al., 2025), explicit entropy regularization, or token filtering based on entropy variation; entropy rewards tend to overamplify the signal of high-entropy tokens and induce oscillations, regularization remains highly sensitive to the choice of its coefficient, and entropy-variation-based methods either rely on hard-to-estimate information about unsampled tokens or impose oversimplified binary partitions. In addition, raising the sampling temperature only delays rather than prevents entropy collapse. Overall, existing methods either operate at an excessively coarse granularity or lack a principled understanding of the underlying collapse mechanism.

Token-Level Perspectives and STARE. Recent studies have begun to focus on the differentiated contributions of individual tokens(Wang et al., 2025b; Tang et al., 2025; Wang et al., 2026; Chen et al., 2026; Zhu et al., 2025). Some works demonstrate that a small subset of high-entropy tokens dominates the effective learning signal in RLVR and that performing gradient updates only on this subset already yields efficient performance gains(Wang et al., 2025b); another line of work identifies critical tokens at decision branching points along reasoning chains and encourages exploration at these positions(Chen et al., 2026; Tang et al., 2025; Cheng et al., 2025; Yang et al., 2025a). These findings echo the analysis presented in this paper. Most existing methods, however, remain heuristic in nature and either fail to keep the policy entropy stable and controllable or lack comprehensive validation across multiple scenarios, model scales, and long-horizon training settings. Starting from a first-order analysis of token-level entropy dynamics, STARE establishes an advantage-surprisal four-quadrant decomposition that exposes the credit assignment mismatch under shared trajectory-level advantages, together with a near-criticality property; building on this analysis, STARE identifies entropy-critical tokens through a batch-internal surprisal-quantile proxy, selectively reweights their effective advantages, and incorporates a target-entropy closed-loop gate, thereby achieving principled token-level regulation of the policy entropy through a minimally invasive modification to the GRPO objective. STARE is comprehensively validated across model scales ranging from 1.5B to 32B and across three task families covering Short-CoT, Long-CoT, and tool-use scenarios, where it consistently delivers stable performance improvements and further unlocks the optimization potential of long-horizon RL training.

###### B Additional Experiments and Analysis on STARE

- B.1 STARE vs. GRPO-ds: Detailed Comparison across Diverse Scenarios and Model Scales in RL Training

To validate the effectiveness of STARE in long-horizon RL training, we conduct RL training of 5000 steps on Qwen2.5-Math-7B-Base under the Short CoT scenario. Figure 3 compares STARE against GRPO-ds on key metrics, while Figures 5–1 further verify the effectiveness of STARE across model scales ranging from 1.5B to 32B and across diverse task scenarios.

Entropy stability and performance evolution. In Figure 3(a), GRPO-ds exhibits entropy collapse during the early training phase (0–1000 steps), in which the policy entropy decreases sharply and approaches zero, which is consistent with the theoretical analysis presented in Section 3. Meanwhile, the accuracy of GRPO-ds on AIME24 and AIME25 peaks around step 1000 and subsequently saturates, fluctuating without further improvement (Figure 3(e)-(f)), thereby indicating premature convergence of the policy distribution. In contrast, STARE stabilizes the policy entropy near Htgt = 0.3 through token-level advantage reweighting and closed-loop entropy gating, and the accuracy of STARE continues to improve beyond step 1000, reaching the optimum at step 5000, which extends the number of trainable steps and unlocks the optimization potential of long-horizon RL training.

Exploration-exploitation balance. In Figure 3(g)-(h), the Pass@32 of STARE consistently surpasses that of GRPO-ds throughout training, which indicates that the policy retains sufficient output diversity and effectively mitigates mode-seeking behavior. In Figure 3(b)-(c), the training reward and Full-Solve Ratio of GRPO-ds rise rapidly during the early phase and subsequently plateau, whereas STARE maintains a steady upward trajectory throughout training. Meanwhile, the sustained growth of response length under STARE (Figure 3(d)) suggests that the model addresses complex problems by extending the reasoning depth.

Cross-scale and cross-scenario generalization. Figures 5-1 systematically present the behavior of STARE under broader configurations. Across the Short CoT scenario(14B, 32B), the Long CoT scenario(R1-Distill-

(a) Fraction of top-P% high-surprisal tokens in L+q that fall within the theoretical entropy-increasing region

(si,t > si∗,t) across training.

- 0

- 1

- 2

- 3

- 4

GRPO-ds

CumulativeNetEntropyChange

STARE

0 200 400 600 800 1000

Training Steps

(b) Cumulative net entropy contribution of L+q under STARE versus GRPO-ds.

- Figure 11: Validation of the batch-internal surprisal-quantile proxy on Qwen2.5-Math-7B-Base over 1000 RL training steps, examining its alignment with the theoretical critical threshold and the cumulative net entropy contribution of the selected token subset.

Qwen-1.5B, Qwen3-8B-Base), and the tool-use scenario(7B), the results exhibit a consistent pattern: GRPOds suffers from entropy collapse accompanied by performance saturation, whereas STARE maintains entropy stability and delivers continuous accuracy improvements. Notably, on DeepSeek-R1-DistillQwen-1.5B, GRPO-ds undergoes gradual entropy decay that drops below 0.2 by step 5000, whereas under STARE the policy entropy rapidly recovers to the target band and remains stable starting around step 3500, with the accuracy on AIME24 and AIME25 improving in tandem (Figure 7), which validates the intervention-recovery capability of the proposed method.

In summary, STARE maintains stable policy entropy and delivers consistent performance gains across model scales ranging from 1.5B to 32B and across the three task scenarios, which confirms the effectiveness and robustness of the proposed method.

###### B.2 Validation of the High-Surprisal Quantile Proxy.

To verify the effectiveness of the batch-internal top-P% surprisal-quantile proxy in STARE, Figure 11 evaluates this proxy from two complementary perspectives. Figure 11(a) reports the fraction of tokens selected by the top-P% high-surprisal criterion that fall within the theoretical entropy-increasing region

(si,t > si∗,t). This fraction rises steadily from approximately 60% in the early training phase to around 95%, indicating strong consistency between the surprisal-quantile proxy and the theoretical critical threshold s∗ (Proposition 3.2). Figure 11(b) further shows that the cumulative net entropy contribution of this subset remains positive and monotonically increasing throughout training, thereby confirming

that L+q consistently produces a net entropy-increasing effect, as predicted by Corollary 3.3. These results demonstrate that the batch-internal surprisal-quantile mechanism effectively approximates the theoretical threshold s∗ and accurately identifies entropy-critical tokens while eliminating the need to solve Φ(p∗) = 0 at each position, thereby validating both the feasibility and the effectiveness of the proposed proxy strategy.

###### B.3 Detailed Effects of Single-Polarity and Combined Operations in STARE.

To validate the effectiveness of the token-level reweighting mechanism under the four-quadrant decomposition framework, we ablate all four single-polarity operations (O1, O2, O3, O4) and all four combined operations (C1, C2, C3, C4). Table 2 reports the accuracy of each operation on AIME24 and AIME25, and Figure 9 presents the corresponding policy entropy evolution curves. As shown in Figure 9, GRPO-ds suffers from rapid entropy decay toward zero in the early training phase, whereas all eight STARE variants effectively mitigate entropy collapse. Specifically, O1 and O3 strengthen entropy-increasing signals through weight amplification, O2 and O4 suppress entropy-decreasing signals through weight attenuation, and C1–C4 combine interventions from two quadrants simultaneously. Table 2 further shows that all STARE variants substantially outperform GRPO-ds, confirming that token-level reweighting is effective

across all four entropy-critical quadrants. Among all variants, O1 (amplifying L+q ) and C2 (amplifying L+q while attenuating L−q ) achieve the strongest performance, reaching 44.2%/23.8% and 42.5%/24.2% on AIME24/AIME25 respectively. We therefore adopt STARE-O1 as the default configuration and STARE-C2

4.0

GRPO-ds

STARE: W=1.1, No Target Entropy

3.5

STARE: W=1.1, Target entropy=0.3

3.0

2.5

Entropy

2.0

1.5

1.0

0.5

0.0

0 200 400 600 800 1000

Training Steps

- Figure 12: Effect of the target-entropy closed-loop gate on policy entropy regulation for STARE (W=1.1, P=10%) on Qwen2.5-Math-7B-Base over 1000 RL training steps. The closed-loop gate effectively confines

the policy entropy to bounded oscillation around the target level Htgt=0.3, preventing both entropy collapse and over-exploration observed under open-loop reweighting and standard GRPO-ds, respectively.

as the dual-sided variant in our work.

###### B.4 Detailed Ablation on Key Hyperparameters and Target-Entropy Gating

To validate the mechanisms underlying the key hyperparameters in STARE (W and P) and the targetentropy gating, we conduct ablation experiments on Qwen2.5-Math-7B-Base (Figure 4). Figure 4(a) examines the reweighting factor W under an open-loop setting (without the target-entropy gate). A minimal perturbation of W = 1.01 is sufficient to mitigate the entropy decay observed in GRPO-ds; W ≥ 1.05 yields steady entropy growth, whereas W ≥ 2.0 drives the entropy upward too aggressively and triggers divergence. This behavior corroborates the near-criticality property (Corollary 3.6): beyond the critical threshold, W governs the magnitude rather than the direction of the entropy shift. The openloop regime, however, tends to stabilize the entropy at an excessively high level and thereby induces over-exploration. Once the target-entropy closed-loop gate is introduced with Htgt = 0.3, Figure 4(b) shows that values of W across the range [1.05,1.5] steer the policy entropy into the target band and maintain bounded oscillation around it, which confirms the stability of the closed-loop mechanism and substantially reduces the sensitivity to W. Figure 4(c) further investigates the high-surprisal token selection ratio P. With W = 1.1 and Htgt = 0.3 held fixed, the policy entropy remains within the target band for P ∈ [5%,20%], and even at P = 40% it is confined to [0.1,0.2], effectively preventing collapse and indicating that the surprisal-quantile selection admits a broad operating range. Overall, STARE exhibits strong robustness to the configurations of W and P, and the target-entropy gate stably constrains the policy entropy. More detailed entropy evolution and performance comparisons are reported in Appendix B.5, Table 3, and Figure 12

- Table 3: Ablation of the target-entropy gate for STARE with W = 1.1 and P=10 on AIME24 and AIME25.

Model AIME24 AIME25

GRPO-ds 35.2 17.3 STARE without target-entropy gate 36.7 18.9 STARE with target-entropy gate 38.0 20.0

###### B.5 Effectiveness of Target-Entropy Closed-Loop Gating

To validate the necessity of the closed-loop target-entropy gate, we fix W = 1.1 and P = 10% on Qwen2.5Math-7B-Base, and compare three configurations over 1000 RL training steps: GRPO-ds, open-loop STARE, and closed-loop STARE with Htgt = 0.3. As shown in Figure 12, open-loop STARE alleviates entropy collapse but drives the policy entropy upward to an excessively high level, which induces overexploration; in contrast, closed-loop STARE confines the policy entropy to bounded oscillation around the target band. Table 3 further reports that open-loop STARE attains 36.7%/18.9% on AIME24/AIME25,

GRPO-ds STARE

2004

| |
|---|

2000

Countper1kSamples

| |
|---|

1446

1500

1000

889

669

655

518

500

335

307

284

262

85

54

0

ContrastReflectionSelf-CorrectionHesitationBacktrackingSummaryRevision

Category

- Figure 13: Reflection-related token counts per 1k samples for STARE vs. GRPO-ds on Qwen2.5-32B-Base.

already surpassing GRPO-ds, while closed-loop STARE further improves the accuracy to 38.0%/20.0%, with both variants substantially outperforming the GRPO-ds baseline at 35.2%/17.3%. These results confirm that token-level advantage reweighting in STARE mitigates entropy collapse and delivers consistent gains, and that the closed-loop gate stabilizes the policy entropy within a reasonable band to balance exploration and exploitation, thereby yielding additional performance improvements.

###### B.6 Details about Emergent Reflection Behaviors.

To investigate how STARE elicits deep reasoning, we analyze the reflection behaviors that emerge during RL training. Taking Qwen2.5-32B-Base as a representative example, we randomly sample 50 training steps shared by both STARE and GRPO-ds. This analysis is intended as a coarse, qualitative diagnostic rather than a principled measurement of reasoning depth: we apply heuristic regular-expression matching based on manually-observed patterns to obtain an approximate count of reflection-related tokens, grouped into six categories: contrast (but, however), reflection (wait, reconsider), self-correction (made a mistake, correct), hesitation (perhaps, possibly, maybe), backtracking (recalculate, redo, go back), and summary revision (summary, ultimately). Figure 13 reports reflection-related token counts per 1k samples; STARE substantially exceeds GRPO-ds across all six categories, with markedly larger margins on reflection and self-correction, indicating that the resulting policy retains stronger exploration and greater output diversity. Figure 10 further reports the frequency distribution of tokens selected by STARE for advantage amplification; the word cloud reveals that the reweighted tokens concentrate on vocabulary expressing uncertainty and self-correction, such as should be, but, instead, and verification, confirming that the batch-internal surprisal-quantile selection effectively identifies rare forking tokens carrying exploratory semantics. Taken together, the emergent growth of reflection tokens and the semantic bias of the reweighted token distribution demonstrate that STARE activates the deep exploration capability of the model and delivers consistent performance gains through token-level credit rebalancing.

- Table 4: Ablation of fixed and adaptive reweighting coefficients for STARE in 1000 RL training steps on AIME24 and AIME25.

###### Model AIME24 AIME25

GRPO-ds 35.2 17.3 STARE: Fixed W = 1.1 38.0 20.0

- STARE: Adaptive W, Wmax = 1.5, α = 0.01 37.7 19.5
- STARE: Adaptive W, Wmax = 1.5, α = 0.02 37.4 19.1

- STARE: Adaptive W, Wmax = 2.0, α = 0.01 36.9 18.6
- STARE: Adaptive W, Wmax = 2.0, α = 0.02 37.1 18.1

| | |
|---|---|
| | |
| | |
| | |

0.5

0.4

###### Entropy

0.3

0.2

0 1000 2000 3000 4000 5000

Training Steps

(a) Policy entropy evolution of STARE under off-policy training (four gradient updates per batch).

0.4

0.3

Acc

0.2

- AIME24

- AIME25

0.1

0 1000 2000 3000 4000 5000

Training Steps

(b) AIME24 and AIME25 accuracy of STARE under offpolicy training.

- Figure 14: Off-policy training dynamics of STARE on Qwen2.5-Math-7B-Base with four gradient updates per batch over 5,000 RL training steps, demonstrating that STARE preserves entropy stability and delivers consistent accuracy improvements beyond the on-policy setting.

###### B.7 Fixed vs. Adaptive Weights

- Table 4 compares fixed and adaptive weighting schemes on Qwen2.5-Math-7B-Base after 1000 RL training steps. All STARE configurations substantially outperform the GRPO-ds baseline, confirming the robustness of the token-level reweighting mechanism. The fixed weight W = 1.1 yields the strongest performance, reaching 38.0% on AIME24 and 20.0% on AIME25. This outcome aligns with the nearcriticality property (Corollary 3.6): once the reweighting factor exceeds the critical threshold, its specific value primarily controls the magnitude rather than the direction of the per-step entropy shift, so that a moderate fixed value suffices for stable regulation. Among the adaptive variants, the configuration with Wmax = 1.5 and α = 0.01 performs best and nearly matches the fixed-weight result, whereas enlarging either Wmax or α leads to a slight degradation. We therefore adopt the fixed W = 1.1 as the default configuration, with Wmax = 1.5 and α = 0.01 recommended when an adaptive schedule is preferred.
- Table 5: Ablation of target entropy threshlod for STARE in 4000 RL training steps on AIME24 and AIME25.

Model AIME24 AIME25 GRPO-ds 37.1 17.7

- STARE: Htarget = 0.1 40.4 20.5
- STARE: Htarget = 0.2 43.2 23.1
- STARE: Htarget = 0.3 44.2 23.8
- STARE: Htarget = 0.4 42.8 21.6

###### B.8 Ablation on Target Entropy Threshold

- Table 5 reports the performance of STARE under different target entropy values Htgt ∈ {0.1,0.2,0.3,0.4} on Qwen2.5-Math-7B-Base after 4000 RL training steps. All configurations substantially outperform the GRPO-ds baseline, confirming the general effectiveness of the closed-loop entropy regulation mechanism.

Performance attains its optimum at Htgt = 0.3, reaching 44.2% on AIME24 and 23.8% on AIME25, with gains of 7.1% and 6.1% over the baseline respectively. The low target entropy restricts the exploration space, whereas the high value easily drives the policy distribution toward over-exploration, and both regimes yield suboptimal outcomes. These results indicate that Htgt effectively governs the exploration-exploitation balance, and we therefore adopt Htgt = 0.3 as the default configuration in the main experiments.

###### B.9 Ablation on Target-Entropy Gate Granularity

On Qwen2.5-Math-7B-Base, we compare three closed-loop gating granularities: a token-level gate that evaluates Hi,t < Htgt at each position, a sample-level gate that conditions activation on the per-sample average entropy H¯i, and a batch-level gate that makes a unified decision based on the batch-average entropy H¯k. As reported in Table 6, all three granularities substantially outperform the GRPO-ds baseline

| | | | |
|---|---|---|---|
| | | | |
| | | | |

0.6

Entropy

0.4

0.2

0 1000 2000 3000 4000

Training Steps

(a) Policy entropy evolution under fixed-threshold lowprobability token reweighting.

0.4

0.3

###### Acc

0.2

0.1

AIME24 AIME25

0.0

0 1000 2000 3000 4000

Training Steps

(b) AIME24 and AIME25 accuracy under fixed-threshold low-probability token reweighting.

- Figure 15: Comparison between fixed-threshold low-probability token reweighting (p < 0.1) and GRPOds on Qwen2.5-Math-7B-Base over 4000 RL training steps, demonstrating the inferior entropy regulation and benchmark performance of static probability-based selection relative to STARE’s batch-internal surprisal-quantile proxy.

- Table 6: Ablation of target-entropy gate granularity for STARE in 1000 RL training steps on AIME24 and AIME25.

Model AIME24 AIME25

GRPO-ds 35.2 17.3 STARE (token-level target-entropy gate) 36.9 19.1 STARE (sample-level target-entropy gate) 37.6 19.3 STARE (batch-level target-entropy gate) 38.0 20.0

(35.2%/17.3%), confirming the general effectiveness of the closed-loop mechanism. The batch-level gate yields the best performance (38.0%/20.0%), surpassing the token-level (36.9%/19.1%) and sample-level (37.6%/19.3%) variants. Since local entropy estimates at the token and sample levels exhibit higher variance, they tend to induce frequent switching of the gating signal and undermine the stability of the regulation. We therefore adopt the batch-level gate as the default setting in our main experiments.

- Table 7: Comparison of off-policy and on-policy STARE in 4000 RL training steps on AIME24 and AIME25.

###### Model AIME24 AIME25

GRPO-ds 37.1 17.7 STARE with off-policy 43.8 22.1 STARE with on-policy 44.2 23.8

###### B.10 Validation of STARE under Off-Policy Training

To assess the applicability of STARE in the off-policy setting, we conduct 4000 steps of RL training on Qwen2.5-Math-7B-Base with four gradient updates per batch. Figure 14 shows that STARE still maintains the policy entropy within the target band, while the accuracy improves steadily throughout training. As reported in Table 7, off-policy STARE attains 43.8%/22.1% on AIME24/AIME25, exceeding GRPO-ds (37.1%/17.7%) by 6.7% and 4.4% respectively, and falling only marginally short of on-policy STARE (44.2%/23.8%). These results confirm that STARE remains effective under off-policy training, demonstrating the robustness and generality of the proposed method.

###### B.11 STARE vs. Fixed-Threshold Low-Probability Token Reweighting

STARE identifies entropy-critical tokens through a batch-internal surprisal-quantile proxy. A more direct alternative is to apply uniform weight amplification to all tokens whose sampling probability falls below a fixed threshold such as p < 0.1 (Figure 15). Under an identical configuration on Qwen2.5-Math-7B-Base with 4000 RL training steps, W = 1.1, and Htgt = 0.3, we compare these two token-selection strategies. As reported in Table 8, fixed-threshold reweighting yields only marginal improvements over GRPO-ds,

- Table 8: Comparison with Fixed-Threshold Low-Probability Token Reweighting on AIME24 and AIME25.

###### Method AIME24 AIME25

GRPO-ds 37.1 17.7 Fixed-Threshold Reweighting (p < 0.1) 38.9 19.7 STARE 44.2 23.8

with gains of 1.8% on AIME24 and 2.0% on AIME25, whereas STARE achieves substantially larger gains of 7.1% and 6.1% respectively, confirming the superiority of the batch-internal top-P% surprisal-quantile proxy in identifying entropy-critical tokens. Moreover, the quantile proxy of STARE adaptively tracks the current policy distribution at every training step, and as shown in Figure 11(a), its agreement with the theoretical threshold s∗ rises from approximately 60% to over 95% throughout training, ensuring high-precision identification of entropy-critical tokens. These results jointly demonstrate the effectiveness and reliability of the batch-internal surprisal-quantile proxy adopted by STARE.

###### C Algorithm: Main STARE Procedure

Algorithm 1 STARE-O1: Surprisal-Guided Token-Level Advantage Reweighting with Fixed Weights and Batch-Level Closed-Loop Target-Entropy Gating

Require: Initial policy πθ0; prompt distribution D; batch size B; group size G; PPO clip range ϵ; reweighting factor W > 1; top-surprisal ratio P ∈ (0,1); target entropy Htgt; learning rate η; total steps K

Ensure: Trained policy πθ

- 1: Initialize θ ← θ0, θold ← θ0
- 2: for k = 1,2, . . . , K do # Stage 1: Rollout and group-normalized advantage estimation
- 3: Sample a batch of prompts {xi}iB=1 ∼ D
- 4: for i = 1, . . . , B do
- 5: Roll out G responses {oi,j}Gj=1 ∼ πθold(· | xi) and obtain verifier rewards {ri,j}Gj=1
- 6: Aˆi,j ← ri,j − mean({ri,·}) std({ri,·}) ▷ shared trajectory-level advantage
- 7: end for
- 8: Flatten all rollouts into a token-level batch with total length N = ∑i,j Ti,j; broadcast Aˆi to every token (i, t)

- # Stage 2: Batch-level closed-loop entropy gating

9: Hi,t ← −∑v∈V πθ(v | ci,t) ln πθ(v | ci,t) for every position (i, t) 10: H¯k ← N1 ∑(i,t) Hi,t 11: gk ← 1 H ¯k < Htgt ▷ activate intervention only when entropy is below target

- # Stage 3: Surprisal-guided entropy-critical token selection (only when gated on)

12: if gk = 1 then 13: si,t ← − ln πθ(oi,t | xi, oi,<t) for every token (i, t) 14: T + ← {(i, t) : Aˆi > 0} ▷ positive-advantage token set 15: Sort T + in descending order of si,t; let K+ ← ⌈P · |T +|⌉ 16: L+q ← the first K+ entries of the sorted list ▷ top-P% high-surprisal positive-advantage tokens 17: else 18: L+q ← ∅ 19: end if

- # Stage 4: Token-level advantage reweighting (Variant I, one-sided amplification)

- 20: for each token (i, t) in the batch do
- 21: ωi,t ←

1 + gk (W − 1), (i, t) ∈ L+q 1, otherwise

- 22: end for # Stage 5: STARE clipped-surrogate policy update
- 23: ρi,t(θ) ←

πθ(oi,t | xi, oi,<t) πθold(oi,t | xi, oi,<t)

- 24: JSTARE(θ) ←

1 N

∑(i,t)ωi,t min ρi,t(θ) Aˆi, clip ρi,t(θ), 1 − ϵ, 1 + ϵ A ˆi

- 25: θ ← θ + η ∇θ JSTARE(θ) ▷ single on-policy gradient step
- 26: θold ← θ
- 27: end for
- 28: return πθ

###### D Basic Derivations for Sections 2 and 3.1

###### D.1 Softmax Jacobian Derivation

- Proposition D.1 (Softmax Jacobian). Let

πθ(v | c) =

exp(zv) ∑u∈V exp(zu)

.

Then, for any v, v′ ∈ V,

∂πv′ ∂zv

= πv′(δv′v − πv).

Proof. Let

Z ≜ ∑

u∈V

exp(zu), πv′ =

exp(zv′) Z

.

Differentiating with respect to zv yields

∂πv′ ∂zv

=

δv′v exp(zv′) Z − exp(zv′) exp(zv) Z2

.

Factoring out exp(zv′) from the numerator gives

∂πv′ ∂zv

=

exp(zv′) Z

δv′v −

exp(zv) Z

. Using

exp(zv′) Z

= πv′,

exp(zv) Z

= πv, we obtain

∂πv′ ∂zv

= πv′(δv′v − πv). This proves the result.

| |
|---|

D.2 Token-Level Logit Update in the Unclipped GRPO Regime

- Proposition D.2 (Token-level logit update in the unclipped GRPO regime). In the unclipped regime of GRPO, at a given decoding position, the gradient of the local surrogate objective with respect to the logit vector is aligned with

Aˆ ∇z log πθ(a | c). Absorbing the positive proportionality constant into an infinitesimal step size η > 0 gives the equivalent logit update

∆zv = η Aˆ (δva − πv), v ∈ V.

Proof. When clipping is inactive, the gradient direction of the local surrogate objective with respect to the model parameters is proportional to

Aˆ ∇θ log πθ(a | c).

By the chain rule, an equivalent ascent direction in logit space is

∂ ∂zv

∂ ∂zv

A ˆ log πθ(a | c) = Aˆ

log πa. Moreover,

1 πa

∂πa ∂zv

∂ ∂zv

. Substituting Proposition D.1 gives

log πa =

∂πa ∂zv

= πa(δva − πv), and therefore

1

∂ ∂zv

log πa =

πa · πa(δva − πv) = δva − πv.

Hence

∂ ∂zv

A ˆ log πa = Aˆ(δva − πv). Taking an infinitesimal gradient ascent step yields

∆zv = η Aˆ(δva − πv). This proves the result.

| |
|---|

###### D.3 Lemma 2.1 (Entropy Gradient with Respect to Logits: Surprisal-Deviation Form)

For any v ∈ V,

∂H ∂zv

= πv(sv − H), sv ≜ − ln πv.

Proof. Fix the context c and suppress it in the notation. By the definition of the Shannon entropy of the conditional next-token distribution,

H = − ∑

πu ln πu. Differentiating with respect to zv gives

u∈V

∂H ∂zv

∂ ∂zv

= − ∑

πu ln πu . Applying the product rule to each term,

u∈V

1 πu

∂ ∂zv

∂πu ∂zv

∂πu ∂zv

∂πu ∂zv

(ln πu + 1). Thus

(πu ln πu) =

ln πu + πu ·

=

∂H ∂zv

∂πu ∂zv

= − ∑

(ln πu + 1). Using Proposition D.1 yields

u∈V

∂H ∂zv

= − ∑

πu(δuv − πv)(ln πu + 1). Separating the two terms gives

u∈V

∂H ∂zv

= − ∑

πuδuv(lnπu +1) + ∑

πuπv(ln πu + 1). The first term reduces to

u∈V

u∈V

##### − ∑

πuδuv(ln πu + 1) = −πv(ln πv + 1), and the second term becomes

u∈V

##### ∑

πuπv(lnπu +1) = πv ∑

πu(ln πu + 1). Since

u∈V

u∈V

##### ∑

πulnπu = −H, ∑

πu = 1, we have

u∈V

u∈V

##### ∑

πu(ln πu + 1) = −H + 1. Substituting back yields

u∈V

∂H ∂zv

= −πv(ln πv + 1) + πv(−H + 1) = −πv ln πv − πvH. Using sv = − ln πv, we obtain

−πv ln πv = πvsv, and therefore

∂H ∂zv

= πv(sv − H). This proves the result.

| |
|---|

###### D.4 Theorem 3.1 (Token-Level Entropy Variation)

In the unclipped regime of GRPO, let Aˆ denote the trajectory-level normalized advantage assigned to the current token position, and let a denote the token sampled at the current decoding position, with conditional probability

p = π(a | c), sa = − ln p. Define

πv2(ln πv + H), and

##### S2 ≜ ∑

v∈V

Φ(p) ≜ p(ln p + H) − S2. Then the first-order directional derivative of the conditional policy entropy along the GRPO policygradient direction satisfies

dH dη η=0

= −Aˆ Φ(p).

Proof. By Proposition D.2, the logit velocity along the GRPO policy-gradient direction is

dzv dη η=0

= Aˆ(δva − πv).

Therefore, by the definition of the directional derivative, dH dη η=0

∂H ∂zv

dzv dη η=0

##### = ∑

.

v∈V

Substituting Lemma 2.1 gives

dH dη η=0

πv(sv − H) Aˆ(δva − πv).

##### = ∑

v∈V

Factoring out Aˆ yields

dH dη η=0

##### = Aˆ ∑

πv(sv − H)(δva − πv). Splitting the sum gives

v∈V

dH dη η=0

##### = Aˆ ∑

πv2(sv − H) . Only the term v = a contributes to the first sum, so

πv(sv − H)δva − ∑

v∈V

v∈V

##### ∑

πv(sv − H)δva = πa(sa − H) = p(sa − H).

v∈V

Using sv = − ln πv, the second term becomes

πv2(ln πv + H) = S2.

πv2(−sv + H) = ∑

πv2(sv − H) = ∑

##### − ∑

v∈V

v∈V

v∈V

Hence

dH dη η=0

= Aˆ p(sa − H) + S2 . Substituting sa = − ln p yields

p(sa − H) + S2 = p(− ln p − H) + S2 = − p(ln p + H) − S2 = −Φ(p). Therefore,

dH dη η=0

= −Aˆ Φ(p). This proves the result.

| |
|---|

###### E Complete Proofs for Sections 3.2 and 3.3 and Near-Criticality Analysis

###### E.1 Positivity of S2 under Non-Uniform Distributions

- Lemma E.1 (Positivity of S2 under non-uniform distributions). If the conditional distribution π is nonuniform, then

S2 = ∑

v∈V

πv2(ln πv + H) > 0.

If π is uniform, then S2 = 0. Proof. Let V ∼ π and define

X ≜ πV. Then

E[lnX + H] = ∑

v∈V

πv(ln πv + H)

= ∑

v

πvlnπv + H∑

v

πv

= −H + H

= 0. Moreover,

S2 = ∑

v

πv2(ln πv + H)

= E X(ln X + H) . Since E[ln X + H] = 0, we obtain

S2 = E X(ln X + H)

− E[X]E[ln X + H]

= Cov(X,ln X + H)

= Cov(X,ln X).

Using the symmetric form of covariance, and letting X′ be an independent copy of X, we have

Cov(X,ln X) =

- 1

- 2

E (X − X′)(ln X − ln X′) . Therefore,

S2 =

- 1

- 2 ∑

u,v∈V

πuπv(πu − πv)(ln πu − ln πv). Because the logarithm is strictly increasing, for any a, b > 0,

(a − b)(ln a − ln b) ≥ 0,

with equality if and only if a = b. Hence every term in the summation above is nonnegative.

If π is non-uniform, there exist u, v such that πu ̸= πv. In the softmax setting considered in this paper, all token probabilities are strictly positive, so the corresponding weight πuπv is also strictly positive, and the associated summand is strictly positive. Thus the full sum is strictly positive, which gives S2 > 0.

If π is uniform, then πu = πv for all u, v, so every summand is zero and S2 = 0. E.2 H > S2 under Non-Degenerate Distributions

| |
|---|

- Lemma E.2 (H > S2 under non-degenerate distributions). If the distribution π is non-degenerate, namely it is not a point mass on a single token, then

H > S2. Proof. By definition,

πv2(ln πv + H)

##### S2 = ∑

v∈V

πv2lnπv + H∑

πv2.

##### = ∑

v

v

Hence

πv2. Collecting the terms that contain H yields

πv2lnπv − H∑

H −S2 = H −∑

v

v

πv2 ln πv. Equivalently,

##### πv2 −∑

H −S2 = H 1−∑

v

v

##### πv2 +∑

πv2(− ln πv).

H −S2 = H 1−∑

v

v

We now inspect the two terms. Since π is non-degenerate, at least two tokens have positive probability. Therefore,

2

##### πv2 < ∑

##### ∑

= 1, which implies

πv

v

v

πv2 > 0.

##### 1−∑

v

The Shannon entropy of a non-degenerate distribution is strictly positive, so

πv2 > 0. For the second term, 0 < πv ≤ 1 in the softmax setting, and hence

H 1−∑

v

− ln πv ≥ 0. Since πv2 ≥ 0, each term satisfies

πv2(− ln πv) ≥ 0. Thus

πv2(− ln πv) ≥ 0.

##### ∑

v

Combining a strictly positive term with a nonnegative term gives

H − S2 > 0, and therefore

H > S2.

| |
|---|

- E.3 Proposition 3.2 (Uniqueness of the Critical Surprisal Threshold) For any non-uniform and non-degenerate distribution π, there exists a unique

p∗ ∈ (e−H,1), s∗ ≜ − ln p∗ ∈ (0, H), such that

Φ(p∗) = 0, and

Φ(p) > 0 ⇐⇒ p > p∗ ⇐⇒ sa < s∗. Proof. By definition,

Φ(p) = p(ln p + H) − S2, p ∈ (0,1]. We first determine the endpoint signs. As p → 0+, the standard limit p ln p → 0 gives

###### Φ(0+) = −S2 < 0,

- where the inequality follows from Lemma E.1. At p = 1,

Φ(1) = H − S2 > 0,

- where the inequality follows from Lemma E.2.

Next, we show that Φ is strictly increasing on [e−H,1]. Differentiating gives Φ′(p) =

d dp

p(ln p + H) − S2

= ln p + H + 1. For any p ∈ [e−H,1], we have ln p ≥ −H, and hence

Φ′(p) ≥ −H + H + 1 = 1 > 0. Therefore, Φ is strictly increasing on [e−H,1]. Moreover,

###### Φ(e−H) = e−H(−H + H) − S2

= −S2 < 0, whereas

Φ(1) = H − S2 > 0.

By continuity and the intermediate value theorem, there exists at least one

p∗ ∈ (e−H,1) such that Φ(p∗) = 0. Since Φ is strictly increasing on this interval, the zero is unique. Define

s∗ ≜ − ln p∗.

Because p∗ ∈ (e−H,1), applying the negative logarithm gives

0 < − ln p∗ < H, or equivalently,

s∗ ∈ (0, H). Finally, the strict monotonicity of Φ implies

p > p∗ ⇐⇒ Φ(p) > 0. Since − ln(·) is strictly decreasing on (0,1],

p > p∗ ⇐⇒ − ln p < − ln p∗ ⇐⇒ sa < s∗. Combining the two equivalences yields

Φ(p) > 0 ⇐⇒ p > p∗ ⇐⇒ sa < s∗.

| |
|---|

- E.4 Corollary 3.3 (Four-Quadrant Decomposition)

The sign of the first-order entropy variation at a single token position is jointly determined by (sign Aˆ,1[sa < s∗]). When Aˆ > 0 and sa < s∗, entropy decreases. When Aˆ > 0 and sa > s∗, entropy increases. When Aˆ < 0 and sa < s∗, entropy increases. When Aˆ < 0 and sa > s∗, entropy decreases. Proof. By Theorem 3.1,

dH dη η=0

= −AˆΦ(p).

Thus the sign of the first-order entropy variation is exactly the sign of −AˆΦ(p). Proposition 3.2 gives

###### sa < s∗ ⇐⇒ Φ(p) > 0, sa > s∗ ⇐⇒ Φ(p) < 0.

If Aˆ > 0 and sa < s∗, then Φ(p) > 0, so −AˆΦ(p) < 0 and entropy decreases. If Aˆ > 0 and sa > s∗, then Φ(p) < 0, so −AˆΦ(p) > 0 and entropy increases. If Aˆ < 0 and sa < s∗, then Φ(p) > 0, so −AˆΦ(p) > 0 and entropy increases. If Aˆ < 0 and sa > s∗, then Φ(p) < 0, so −AˆΦ(p) < 0 and entropy decreases. At the boundary sa = s∗, the first-order entropy variation is zero.

| |
|---|

- E.5 Asymmetric Entropy Contributions under Shared Trajectory-Level Advantages

This subsection provides the detailed quantitative analysis of the asymmetric entropy contributions summarized in Section 3.2.

Consider the subset of trajectories assigned positive advantages (Aˆi > 0). Because rollouts are sampled from the current policy πθ, the probability of drawing a low-surprisal token at each decoding position is higher than that of drawing a high-surprisal token. Within positive-advantage trajectories, entropy-decreasing tokens (low surprisal, Φ > 0) therefore constitute the statistical majority by sampling frequency, while entropy-increasing tokens (high surprisal, Φ < 0) remain a statistical minority. Denoting pi,t = πθ(oi,t | xi, oi,<t) and si,t = − ln pi,t, the net first-order entropy contribution from positive-advantage samples decomposes as

dH¯ + dη η=0

=

1

N − ∑

i,t: Aˆi>0 si,t<si∗,t

Aˆi Φi,t

entropy-decreasing majority (<0)

+ ∑

i,t: Aˆi>0 si,t>si∗,t

Aˆi |Φi,t|

entropy-increasing minority (>0)

.

During GRPO training, all tokens within a trajectory share a single trajectory-level advantage Aˆi, leaving the algorithm unable to distinguish the opposing entropy effects of these two token categories. This observation reveals a fundamental gradient-level mechanism underlying entropy collapse in GRPO. Under the shared trajectory-level advantage, the reinforced low-surprisal majority systematically drives the distribution toward concentration, with entropy-decreasing contributions dominating in expectation. In contrast, the high-surprisal minority that could preserve diversity contributes limited entropy-increasing effects. Upon aggregation over the batch, the net outcome is a systematic reduction in entropy. An analogous but mirror-image asymmetry governs the negative-advantage subset.

- E.6 Theorem 3.4 (Entropy Neutrality Identity) For any conditional distribution π,

Ea∼π[Φ(a)] = ∑

πvΦ(πv) = 0.

v∈V

Proof. Here Φ(a) denotes the value obtained by first sampling a ∼ π and then evaluating Φ(p) at p = π(a). Therefore,

Ea∼π[Φ(a)] = ∑

πvΦ(πv). Substituting the definition of Φ gives

v∈V

##### ∑

πvΦ(πv) = ∑

πv πv(ln πv + H) − S2 .

v

v∈V

Expanding the right-hand side yields

πv2(lnπv + H)−S2∑

##### ∑

πv.

v

v

By the definition of S2,

πv2(ln πv + H) = S2, and by normalization,

##### ∑

v

##### ∑

πv = 1. Thus

v

##### ∑

πvΦ(πv) = S2 − S2 = 0, which proves

v∈V

Ea∼π[Φ(a)] = 0.

| |
|---|

###### E.7 Proposition 3.5 (Entropy Gradient under Token-Level Reweighting)

Let

L+ = {(i, t) : Aˆi > 0, si,t > si∗,t}. Suppose the effective advantage of every token in L+ is multiplied by a factor W ≥ 1, while all other token positions keep unit weight. Then the first-order variation of the batch-averaged entropy satisfies

dH¯ dη W

1 N

Λ − (W − 1)Γ ,

= −

where

Aˆi|Φi,t| > 0. The critical weight is

AˆiΦi,t, Γ ≜ ∑

##### Λ ≜ ∑

i,t

(i,t)∈L+

Λ Γ

W∗ = 1 +

.

The batch-level net entropy variation is positive when W > W∗ and negative when W < W∗.

Proof. Under token-level reweighting, the policy-gradient contribution at position (i, t) is linearly scaled by a positive weight ωi,t. Since the first-order entropy derivative in Theorem 3.1 is linear in the policygradient direction, the corresponding entropy contribution becomes

−ωi,tAˆiΦi,t.

Hence the batch-averaged first-order entropy variation is dH¯ dη ω

1

ωi,tAˆiΦi,t.

##### N ∑

= −

i,t

In the present reweighting scheme,

ωi,t =

W, (i, t) ∈ L+, 1, otherwise.

Therefore,

  .



dH¯ dη W

1 N

AˆiΦi,t

AˆiΦi,t +W ∑

#####  ∑

= −

(i,t)∈L+

(i,t)∈L/ +

Equivalently,



  .

dH¯ dη W

1 N

AˆiΦi,t + (W −1) ∑

AˆiΦi,t

##### ∑

= −

i,t

(i,t)∈L+

Define

AˆiΦi,t.

##### Λ ≜ ∑

i,t

For any (i, t) ∈ L+, we have Aˆi > 0 and si,t > si∗,t. By Proposition 3.2, Φi,t < 0. Hence

AˆiΦi,t = −Aˆi|Φi,t|. It follows that

AˆiΦi,t = − ∑

Aˆi|Φi,t|. Define

##### ∑

(i,t)∈L+

(i,t)∈L+

Aˆi|Φi,t|.

##### Γ ≜ ∑

(i,t)∈L+

When L+ is nonempty, Γ > 0 because Aˆi > 0 and |Φi,t| > 0 on this set. The expression above becomes dH¯ dη W

1 N

Λ − (W − 1)Γ . Let

= −

Λ Γ

W∗ ≜ 1 +

.

Then

Λ − (W − 1)Γ = Γ(W∗ − W). Since Γ > 0,

dH¯ dη W

= − sign(W∗ − W)

sign

= sign(W − W∗).

Thus dH¯ /dη|W > 0 when W > W∗, dH¯ /dη|W < 0 when W < W∗, and the first-order entropy variation is zero when W = W∗.

| |
|---|

- E.8 Two Auxiliary Bounds: Lower Bound in the High-Surprisal Region and Upper Bound in the Low-Surprisal Region

- E.8.1 Pointwise Lower Bound in the High-Surprisal Region

Lemma E.3 (Pointwise lower bound in the high-surprisal region). If a token satisfies

s ≥ H, then

|Φ(p)| ≥ S2. Proof. The condition s ≥ H is equivalent to

− ln p ≥ H ⇐⇒ ln p + H ≤ 0. Since p > 0, multiplying both sides by p gives

p(ln p + H) ≤ 0. Therefore,

Φ(p) = p(ln p + H) − S2 ≤ −S2 < 0. Consequently,

|Φ(p)| = −Φ(p) = S2 − p(ln p + H) ≥ S2.

| |
|---|

- E.8.2 Pointwise Upper Bound in the Low-Surprisal Region

- Lemma E.4 (Pointwise upper bound in the low-surprisal region). If a token satisfies s < s∗,

then

|Φ(p)| ≤ H − S2. Proof. By Proposition 3.2,

s < s∗ ⇐⇒ Φ(p) > 0. Therefore,

|Φ(p)| = Φ(p) = p(ln p + H) − S2. Since 0 < p ≤ 1, we have ln p ≤ 0, and hence

p ln p ≤ 0. Also, pH ≤ H. Thus

p(ln p + H) = p ln p + pH ≤ H. It follows that

|Φ(p)| ≤ H − S2.

| |
|---|

###### E.9 Statistical Assumptions for the Near-Criticality Analysis

- E.9.1 Single-Token Credit Dilution

- Assumption E.1 (Single-token credit dilution). For long sequences and sufficiently large batches, define the conditional advantage function at any position (i, t) as

gi,t(v) ≜ E[Aˆi | ci,t, oi,t = v].

There exists a constant Cg > 0, independent of the sequence length T, such that for every position and every u, v ∈ V,

|gi,t(u) − gi,t(v)| ≤

Cg T

.

This assumption states that, along a trajectory of length T, the marginal effect of any single token on the trajectorylevel advantage is diluted at rate 1/T.

E.9.2 Non-Degenerate Absolute Scale of the Advantage

- Assumption E.2 (Non-degenerate absolute scale of the advantage). There exists a constant a− > 0 such that, for every position (i, t) and every candidate token v,

E |Aˆi| | ci,t, oi,t = v ≥ a−.

This assumption states that the conditional absolute scale of the normalized advantage remains O(1) rather than degenerating to zero.

E.9.3 Non-Vanishing Mass of a Strong High-Surprisal Positive-Advantage Subset

- Assumption E.3 (Non-vanishing mass of a strong high-surprisal positive-advantage subset). Define

H+ ≜ {(i, t) : Aˆi > 0, si,t ≥ Hi,t}.

There exist constants ρH > 0 and cH > 0 such that, for sufficiently large batches, with high probability,

|H+| ≥ ρHN, and

1

|H+| ∑

(i,t)∈H+

AˆiS2,i,t ≥ cH.

E.9.4 Linear Scale of Total Absolute Entropy Sensitivity

- Assumption E.4 (Linear scale of total absolute entropy sensitivity). Let

|Aˆi| |Φi,t|.

Σabs ≜ ∑

i,t

There exists a constant cΣ < ∞ such that, for sufficiently large batches, with high probability, Σabs ≤ cΣN.

###### E.10 Λ Is a Weak Residual

- Lemma E.5 (Λ is a weak residual). Under Assumptions E.1 and E.2, and under standard large-batch concentration with uniformly bounded second moments,

|Λ| Σabs

AˆiΦi,t.

= O(T−1), Λ ≜ ∑

i,t

Proof. The proof first establishes the O(T−1) scaling at the population level and then transfers it to empirical batch quantities by concentration. Fix a position (i, t) and condition on ci,t = c. Define

Φc(v) ≜ πθ(v | c) ln πθ(v | c) + H(c) − S2(c). When the sampled token is oi,t,

Φi,t = Φc(oi,t).

By the law of total expectation,

E[AˆiΦi,t] = E E[AˆiΦi,t | ci,t] . Given ci,t = c, the token oi,t is sampled from πθ(· | c). Thus

E[AˆiΦi,t | ci,t = c] = ∑

πθ(v | c)E[Aˆi | ci,t = c, oi,t = v]Φc(v).

v∈V

Let

gc(v) ≜ E[Aˆi | ci,t = c, oi,t = v]. Then

E[AˆiΦi,t | ci,t = c] = ∑

πθ(v | c)gc(v)Φc(v).

v

We now use the Entropy Neutrality Identity to remove the mean component of gc. Define

##### g¯c ≜ ∑

πθ(v | c)gc(v). By Theorem 3.4,

v

##### ∑

πθ(v | c)Φc(v) = 0. Therefore,

v

##### ∑

πθ(v | c)gc(v)Φc(v) = ∑

πθ(v | c) gc(v) − g¯c Φc(v), which implies

v

v

E[AˆiΦi,t | ci,t = c] = ∑

πθ(v | c) gc(v) − g¯c Φc(v). Since g¯c is a πθ(· | c)-weighted average of {gc(v)}v∈V, for every v,

v

|gc(u) − gc(v′)|. Assumption E.1 gives

|gc(v) − g¯c| ≤ max

u,v′

Cg T

|gc(v) − g¯c| ≤

. Consequently,

Cg

E[AˆiΦi,t | ci,t = c] ≤

##### T ∑

πθ(v | c)|Φc(v)|. Define

v

Ψ(c) ≜ ∑

πθ(v | c)|Φc(v)|. Then

v

Cg T

E[AˆiΦi,t | ci,t = c] ≤

Ψ(c). Summing over all token positions yields

E[AˆiΦi,t] ≤ ∑

#### |E[Λ]| = ∑

i,t

i,t

E[AˆiΦi,t] ≤

Cg

##### T ∑

E[Ψ(ci,t)].

i,t

We next lower bound E[Σabs]. Conditional on ci,t = c,

E |Aˆi| |Φi,t| | ci,t = c = ∑

πθ(v | c)E |Aˆi| | ci,t = c, oi,t = v |Φc(v)|.

v

- By Assumption E.2, E |Aˆi| | ci,t = c, oi,t = v ≥ a−.

Therefore,

E |Aˆi| |Φi,t| | ci,t = c ≥ a− ∑

πθ(v | c)|Φc(v)| = a−Ψ(c). Summing over positions and taking expectations gives

v

E |Aˆi| |Φi,t| ≥ a− ∑

E[Σabs] = ∑

E[Ψ(ci,t)].

i,t

i,t

Combining the upper bound on |E[Λ]| and the lower bound on E[Σabs] gives

Cg a−T

###### |E[Λ]|

. Thus, at the population level,

E[Σabs] ≤

|E[Λ]| E[Σabs]

###### = O(T−1).

For sufficiently large batches, and assuming uniformly bounded second moments of the corresponding summands, Λ and Σabs concentrate around their population values at a scale that does not change the dependence on T. Hence, with high probability,

|Λ| Σabs

###### = O(T−1).

| |
|---|

- E.11 Theorem 3.6 (Near-Criticality) If Assumptions E.1–E.4 hold and both the sequence length T and the batch size are sufficiently large, then

Λ Γ

###### = O(T−1).

W∗ − 1 =

Proof. By Proposition 3.5,

Λ Γ

W∗ − 1 =

.

It remains to control the numerator and denominator. For the numerator, Lemma E.5 gives

|Λ| Σabs

###### = O(T−1).

For the denominator, we show that Γ is lower bounded by a constant-order fraction of the total absolute entropy sensitivity. First, H+ ⊂ L+. Indeed, Proposition 3.2 gives, at every position,

si∗,t < Hi,t. Therefore, any token satisfying si,t ≥ Hi,t also satisfies

si,t > si∗,t. Together with Aˆi > 0, this implies

H+ ⊂ L+. By the definition of Γ and the inclusion above,

Aˆi|Φi,t| ≥ ∑

Aˆi|Φi,t|.

##### Γ = ∑

(i,t)∈L+

(i,t)∈H+

On H+, the condition si,t ≥ Hi,t holds. Lemma E.3 therefore gives

|Φi,t| ≥ S2,i,t. Hence

AˆiS2,i,t.

##### Γ ≥ ∑

(i,t)∈H+

- By Assumption E.3, 1

AˆiS2,i,t ≥ cH, |H+| ≥ ρHN. Therefore,

#### |H+| ∑

(i,t)∈H+

Γ ≥ cH|H+| ≥ cHρHN. Assumption E.4 gives

Σabs ≤ cΣN. Consequently,

Σabs

cΣN cHρHN

cΣ cHρH

###### = O(1).

Γ ≤

=

We now decompose the critical offset as

Λ Γ

Λ

Σabs Γ

. Taking absolute values gives

Σabs ·

=

Λ Γ

###### = |Λ|

Σabs Γ

.

Σabs ·

The first factor is O(T−1) by Lemma E.5, and the second factor is O(1) by the bound above. Hence

Λ Γ

= O(T−1).

Equivalently,

Λ Γ

W∗ − 1 =

= O(T−1).

| |
|---|

In the entropy-collapse regime studied in the main text, baseline GRPO corresponds to Λ > 0. Hence W∗ > 1, and the critical weight exceeds one only by an O(T−1) offset. The proof lower bounds Γ using the stronger subset H+ rather than the full entropy-increasing positive-advantage set L+, so the argument is conservative. In actual training, additional contributions from L+ \ H+ further enlarge Γ and make the critical offset closer to zero.

###### F Formalization of the Cross-Step Entropy Dynamics in Section 3.4

This appendix casts the dynamical discussion in Section 3.4 as a discrete-time system under a mean-field closure. The purpose is to formalize the qualitative mechanism described in the main text. It is not intended as an unconditional global convergence theorem for general deep neural network training.

###### F.1 Mean-Field Approximation

Assume that, within the training interval of interest, the batch-level quantities Λk and Γk are primarily determined by the current batch-averaged policy entropy H¯k. More precisely, suppose that there exist continuous functions Λ(h) and Γ(h) such that

Λk ≈ Λ(H¯k), Γk ≈ Γ(H¯k), Γ(h) > 0. The associated critical-weight curve is defined as

Λ(h) Γ(h)

W∗(h) ≜ 1 +

.

The statement in the main text that flatter policy distributions require a lower critical weight to sustain entropy growth is formalized through the following monotonicity condition:

W∗(h) is strictly decreasing in h on the relevant interval.

###### F.2 Sign of the One-Step Batch Entropy Change under a Fixed Weight

- Proposition F.1 (Sign of the one-step batch entropy change under a fixed weight). For a fixed token-level reweighting factor W, the one-step change in batch-averaged policy entropy satisfies

Γ(H¯k) N

1 N

W − W∗(H¯k) . Consequently,

∆H¯k = −

Λ(H¯k) − (W − 1)Γ(H¯k) =

sign(∆H¯k) = sign W − W∗(H¯k) . Proof. By the mean-field form of the batch-level entropy dynamics,

1 N

∆H¯k = −

Λ(H¯k) − (W − 1)Γ(H¯k) .

The bracketed term can be rewritten as

Λ(H¯k) Γ(H¯k) − (W − 1) .

Λ(H¯k) − (W − 1)Γ(H¯k) = Γ(H¯k)

Since

Λ(H¯k) Γ(H¯k)

W∗(H¯k) = 1 +

, we have

Λ(H¯k) Γ(H¯k) − (W − 1) = W∗(H¯k) − W.

Substituting this identity gives

Γ(H¯k) N

Γ(H¯k) N

∆H¯k = −

W∗(H¯k) − W =

W − W∗(H¯k) .

Because N > 0 and Γ(H¯k) > 0, the sign of ∆H¯k is entirely determined by W − W∗(H¯k). This proves the proposition.

| |
|---|

###### F.3 Open-Loop Self-Reinforcing Entropy Collapse and Recovery

Corollary F.2 (Open-loop self-reinforcing entropy collapse and recovery). Under the mean-field approximation above, the open-loop dynamics exhibit two symmetric feedback regimes. If W = 1 and the current entropy level satisfies W∗(H¯k) > 1, then Proposition F.1 gives ∆H¯k < 0. Since W∗(h) is strictly decreasing in h, the next step satisfies

H¯k+1 < H¯k =⇒ W∗(H¯k+1) ≥ W∗(H¯k). Thus, the unit-weight GRPO baseline moves farther below the critical-weight curve, which strengthens the entropy-decreasing pressure and reinforces entropy collapse.

Conversely, if at some step W > W∗(H¯k), then Proposition F.1 gives ∆H¯k > 0. By the same monotonicity condition,

H¯k+1 > H¯k =⇒ W∗(H¯k+1) ≤ W∗(H¯k). Therefore, the margin by which the fixed weight exceeds the critical-weight curve increases after the entropy rises. The entropy-increasing pressure is consequently strengthened, which reinforces entropy recovery.

Proof. Both claims follow from Proposition F.1 and the monotonicity of W∗(h). For the collapse regime, if W = 1 and W∗(H¯k) > 1, then ∆H¯k < 0, so H¯k+1 < H¯k. Since the critical-weight curve decreases with entropy, evaluating it at the smaller entropy value H¯k+1 yields a critical weight that is no smaller than before. The unit-weight baseline is therefore even farther below the critical threshold at the next step. The recovery regime follows by the same argument with the inequalities reversed. This proves the corollary.

| |
|---|

###### F.4 Local Stability of Batch-Level Target-Entropy Gating

- Proposition F.3 (Local stability of batch-level target-entropy gating). Consider the batch-level target-entropy gating rule introduced in Section 4.3,

gk = 1[H¯k < Htgt].

For the one-sided STARE variant, the one-step change in batch-averaged policy entropy is

1 N

Λ(H¯k) − gk(W − 1)Γ(H¯k) . Define the gated and ungated update maps as

∆H¯k = −

1 N

Fon(h) ≜ −

Λ(h) − (W − 1)Γ(h) ,

1 N

Foff(h) ≜ −

Λ(h). Assume that there exists a target-entropy neighborhood

I = [Htgt − δ, Htgt + δ]

such that the gated update points toward higher entropy below the target, while the ungated update points toward lower entropy at and above the target:

Fon(h) > 0, h ∈ [Htgt − δ, Htgt),

Foff(h) < 0, h ∈ [Htgt, Htgt + δ]. Assume in addition that the one-step displacement inside this neighborhood is bounded by the neighborhood half-width:

∆max ≜ max sup

Fon(h), sup

|Foff(h)| ≤ δ.

h∈[Htgt−δ,Htgt)

h∈[Htgt,Htgt+δ]

Then I is forward invariant for the induced discrete-time entropy dynamics. Equivalently, once H¯k enters I, all subsequent batch-averaged entropy values remain inside this neighborhood. The closed-loop system therefore exhibits bounded oscillation around the target entropy, rather than continuing to drift monotonically away from it.

Proof. Within I, the assumed sign conditions imply

Fon(h) > 0 (h < Htgt), Foff(h) < 0 (h ≥ Htgt). The bounded-step condition gives ∆max ≤ δ. If H¯k ∈ [Htgt − δ, Htgt), then gk = 1, and hence

H¯k+1 = H¯k + Fon(H¯k). Since Fon(H¯k) > 0,

H¯k+1 > H¯k ≥ Htgt − δ. Moreover, Fon(H¯k) ≤ ∆max ≤ δ, so

H¯k+1 ≤ H¯k + δ < Htgt + δ. Therefore,

H¯k+1 ∈ I. If H¯k ∈ [Htgt, Htgt + δ], then gk = 0, and hence

H¯k+1 = H¯k + Foff(H¯k). Since Foff(H¯k) < 0,

H¯k+1 < H¯k ≤ Htgt + δ. At the same time, |Foff(H¯k)| ≤ ∆max ≤ δ, which gives

H¯k+1 ≥ H¯k − δ ≥ Htgt − δ. Thus,

H¯k+1 ∈ I. In both cases, one step of the dynamics maps I into itself. Hence I is forward invariant, and any trajectory that enters the target-entropy neighborhood remains there thereafter. This proves the proposition.

| |
|---|

###### G Single-Polarity Operations and Finer-Grained Closed-Loop Extensions

- G.1 Definition of Surprisal-Quantile Proxy Sets In Section 4.1 of the main text, we define

T + = {(i, t) : Aˆi > 0}, T − = {(i, t) : Aˆi < 0}.

Within T + and T −, token positions are sorted separately in descending order of surprisal,

si,t = − ln πθ(oi,t | xi, oi,<t), and the top P% positions are selected to form

L+q ⊂ T +, L−q ⊂ T −. To cover all four single-polarity operations, we also define the corresponding low-surprisal proxy sets. Let Uq+ denote the bottom P% positions with the lowest surprisal in T +, and let Uq− denote the bottom P% positions with the lowest surprisal in T −.

These four proxy sets approximate the four theoretical quadrants in the advantage-surprisal decomposition. The set L+q corresponds to positive-advantage high-surprisal positions and targets the entropyincreasing quadrant. The set Uq+ corresponds to positive-advantage low-surprisal positions and targets the entropy-decreasing quadrant. The set Uq− corresponds to negative-advantage low-surprisal positions and targets the entropy-increasing quadrant. The set L−q corresponds to negative-advantage high-surprisal positions and targets the entropy-decreasing quadrant.

###### G.2 Position-Level Logit Update under STARE Reweighting

Proposition G.1 (Position-level logit update under STARE reweighting). Under the weighted clipped surrogate objective

1

JSTARE(θ) =

ωi,t min ρi,t(θ)Aˆi, clip(ρi,t(θ),1 − ϵ,1 + ϵ)Aˆi ,

N ∑

i,t

if position (i, t) lies in the unclipped regime, its effective logit update is ∆zv = η ωi,tAˆi(δva − πv), v ∈ V.

Since ωi,t > 0 always holds, STARE never reverses any token-level policy-gradient direction. It only selectively rescales the magnitude of the original GRPO learning signal.

Proof. The proof follows the same argument as Proposition D.2. The only change is that the local surrogate term is multiplied by the positive weight ωi,t. Therefore, the corresponding logit gradient is scaled by ωi,t:

∂ ∂zv

ωi,tAˆi log πa = ωi,tAˆi(δva − πv).

Taking one infinitesimal gradient ascent step gives the stated update. Because ωi,t > 0, the policy-gradient direction is preserved. This proves the proposition.

| |
|---|

###### G.3 Exact Batch-Level Entropy Variation under Reweighting an Arbitrary Token Subset

- Proposition G.2 (Exact batch-level entropy variation under reweighting an arbitrary token subset). Let S be an arbitrary subset of token positions. Apply a uniform weight r > 0 to positions in S and unit weight to all remaining positions:

r, (i, t) ∈ S, 1, otherwise.

ωi,t =

Then, in the unclipped regime,

dH¯ dη S,r

1 N

Λ + (r − 1)Ξ(S) , where

= −

AˆiΦi,t, Ξ(S) ≜ ∑

AˆiΦi,t.

##### Λ ≜ ∑

i,t

(i,t)∈S

Proof. By Proposition G.1 and Theorem 3.1, the first-order entropy contribution at each position is linearly scaled by its token-level weight:

−ωi,tAˆiΦi,t. Thus,

dH¯ dη ω

1

ωi,tAˆiΦi,t. Substituting the single-subset reweighting form gives

##### N ∑

= −

i,t

ωi,tAˆiΦi,t = ∑

AˆiΦi,t +r ∑

AˆiΦi,t.

##### ∑

i,t

(i,t)∈S/

(i,t)∈S

Rearranging the expression into the unweighted batch sum plus the reweighting correction yields

ωi,tAˆiΦi,t = ∑

AˆiΦi,t + (r−1) ∑

AˆiΦi,t = Λ + (r − 1)Ξ(S).

##### ∑

i,t

i,t

(i,t)∈S

Substitution completes the proof:

dH¯ dη S,r

1 N

###### Λ + (r − 1)Ξ(S) .

= −

| |
|---|

###### G.4 Four Single-Polarity Operations

- Operation O1 amplifies L+q . Define

ωi(,O1t ) =

W+, (i, t) ∈ L+q , 1, otherwise,

W+ > 1.

- By Proposition G.2, dH¯

dη O1

= −

1 N

Λ + (W+ − 1)Ξ(L+q ) .

When L+q is dominated by positive-advantage high-surprisal entropy-increasing positions, we have Ξ(L+q ) < 0. Define

ΓL+ ≜ −Ξ(L+q ) = ∑

(i,t)∈L+q

|AˆiΦi,t|.

Then,

dH¯ dη O1

= −

1 N

Λ − (W+ − 1)ΓL+ .

- Operation O2 attenuates Uq+. Define

ωi(,O2t ) =

M+, (i, t) ∈ Uq+, 1, otherwise,

0 < M+ < 1.

- By Proposition G.2, dH¯

dη O2

= −

1 N

Λ + (M+ − 1)Ξ(Uq+) .

When Uq+ is dominated by positive-advantage low-surprisal entropy-decreasing positions, we have Ξ(Uq+) > 0. Define

ΓU+ ≜ Ξ(Uq+) = ∑

(i,t)∈Uq+

|AˆiΦi,t|.

Then,

dH¯ dη O2

= −

1 N

Λ − (1 − M+)ΓU+ .

- Operation O3 amplifies Uq−. Define

ωi(,O3t ) =

W−, (i, t) ∈ Uq−, 1, otherwise,

W− > 1.

- By Proposition G.2, dH¯

dη O3

= −

1 N

Λ + (W− − 1)Ξ(Uq−) .

When Uq− is dominated by negative-advantage low-surprisal entropy-increasing positions, we have Ξ(Uq−) < 0. Define

ΓU− ≜ −Ξ(Uq−) = ∑

(i,t)∈Uq−

|AˆiΦi,t|.

Then,

dH¯ dη O3

= −

1 N

Λ − (W− − 1)ΓU− .

- Operation O4 attenuates L−q . Define

ωi(,O4t ) =

- By Proposition G.2, dH¯

dη O4

M−, (i, t) ∈ L−q , 1, otherwise,

###### 0 < M− < 1.

1 N

= −

###### Λ + (M− − 1)Ξ(L−q ) .

When L−q is dominated by negative-advantage high-surprisal entropy-decreasing positions, we have Ξ(L−q ) > 0. Define

|AˆiΦi,t|.

ΓL− ≜ Ξ(L−q ) = ∑

(i,t)∈L−q

Then,

dH¯ dη O4

1 N

Λ − (1 − M−)ΓL− .

= −

###### G.5 Exact Entropy Variation under Unified Closed-Loop Gating

- Proposition G.3 (Exact entropy variation under unified closed-loop gating). Let gi,t ∈ {0, 1} be an arbitrary binary gate. For the one-sided variant, define

ωi(,V1,t g) = 1 + gi,t(W − 1)1[(i, t) ∈ L+q ], W > 1. Then,

dH¯ dη V1,g

1 N

gi,t1[(i, t) ∈ L+q ]AˆiΦi,t .

Λ+ (W −1)∑

= −

i,t

If the activated proxy set is dominated by the corresponding entropy-increasing quadrant, the expression reduces to dH¯ dη V1,g

1 N

Λ − (W − 1)Γ+g ,

= −

where

gi,t1[(i, t) ∈ L+q ]|AˆiΦi,t|.

##### Γ+g ≜ ∑

i,t

For the two-sided variant, define

###### ωi(,V2,t g) = 1 + gi,t(W − 1)1[(i, t) ∈ L+q ] − gi,t(1 − M)1[(i, t) ∈ L−q ], W > 1, 0 < M < 1.

Then,

dH¯ dη V2,g

1 N

= −

## gi,t1[(i,t) ∈ L+q ]AˆiΦi,t +(M−1)∑

gi,t1[(i, t) ∈ L−q ]AˆiΦi,t .

Λ+(W −1)∑

i,t

i,t

If the two activated proxy sets are respectively dominated by the corresponding entropy-increasing and entropydecreasing quadrants, the expression reduces to

dH¯ dη V2,g

1 N

Λ − (W − 1)Γ+g − (1 − M)Γ−g ,

= −

where

gi,t1[(i, t) ∈ L−q ]|AˆiΦi,t|.

##### Γ−g ≜ ∑

i,t

Proof. The result follows by substituting the corresponding gated weights into the general weighted entropy identity derived in the proof of Proposition G.2. In the one-sided case,

ωi(,V1,t g) − 1 = gi,t(W − 1)1[(i, t) ∈ L+q ], which gives

dH¯ dη V1,g

1 N

gi,t1[(i, t) ∈ L+q ]AˆiΦi,t .

Λ+ (W −1)∑

= −

i,t

The two-sided case follows by the same linearity argument. When the gated proxy sets preserve the dominant signs of their target quadrants, the corresponding signed sums can be rewritten as sums of absolute contributions. This proves the proposition.

| |
|---|

- G.6 Unified Instantiations of Three Closed-Loop Granularities The gate gi,t in Proposition G.3 admits three closed-loop instantiations. For batch-level gating, define

gi(,batcht ) = 1[H¯k < Htgt]. For sample-level gating, define

Ti

1 Ti

gi(,samplet ) = 1[H¯i < Htgt], H¯i ≜

##### ∑

Hi,t. For token-level gating, define

t=1

gi(,tokent ) = 1[Hi,t < Htgt]. Because ωi,t > 0 holds in all three cases, every closed-loop variant preserves the direction of the original GRPO learning signal. The variants differ only in the granularity at which they modulate the local strength of the token-level update.

###### H Combined Reweighting Operations and Adaptive Weights

###### H.1 Exact Entropy Variation for Arbitrary Two-Subset Reweighting

- Proposition H.1 (Exact entropy variation for arbitrary two-subset reweighting). Let S1 and S2 be two

disjoint token subsets. We assign multiplicative token weights r1 > 0 and r2 > 0 to them, respectively, while all remaining token positions receive unit weight:

 

- r1, (i, t) ∈ S1,
- r2, (i, t) ∈ S2, 1, otherwise.

ωi,t =



Then, in the unclipped regime of the weighted GRPO surrogate, dH¯ dη S

1 N

###### Λ + (r1 − 1)Ξ(S1) + (r2 − 1)Ξ(S2) .

= −

1,r1;S2,r2

Proof. The proof follows directly from the same argument as Proposition G.2. By the linearity of multiplicative token weighting,

AˆiΦi,t. Using the definitions of Λ and Ξ(·), this becomes

ωi,tAˆiΦi,t = ∑

AˆiΦi,t + (r1 −1) ∑

AˆiΦi,t + (r2 −1) ∑

##### ∑

i,t

i,t

(i,t)∈S1

(i,t)∈S2

ωi,tAˆiΦi,t = Λ + (r1 − 1)Ξ(S1) + (r2 − 1)Ξ(S2).

##### ∑

i,t

Substituting this identity into the first-order batch-averaged entropy derivative yields the result.

| |
|---|

###### H.2 Four Representative Combined Reweighting Operations

We next instantiate Proposition H.1 for four representative two-subset reweighting schemes. Each scheme combines two single-polarity interventions from the four-quadrant entropy decomposition.

- (1) Combination C1: amplifying L+q and amplifying Uq−. C1 amplifies the two dominant entropyincreasing token categories on both advantage sides. Define

 

W+, (i, t) ∈ L+q , W−, (i, t) ∈ Uq−, 1, otherwise,

ωi(,C1t ) =

W+ > 1, W− > 1.



- By Proposition H.1, dH¯

1 N

###### Λ + (W+ − 1)Ξ(L+q ) + (W− − 1)Ξ(Uq−) .

= −

dη C1

When the two proxy sets capture the dominant mass of their corresponding entropy-increasing quadrants, we have

dH¯ dη C1

1 N

Λ − (W+ − 1)ΓL+ − (W− − 1)ΓU− .

= −

- (2) Combination C2: amplifying L+q and attenuating L−q . C2 strengthens high-surprisal entropyincreasing tokens under positive advantages and weakens high-surprisal entropy-decreasing tokens under negative advantages. This is the combined operation used by Variant II in the main text. Define

ωi(,C2t ) =

 



W, (i, t) ∈ L+q , M, (i, t) ∈ L−q , 1, otherwise,

W > 1, 0 < M < 1.

By Proposition H.1,

dH¯ dη C2

= −

1 N

Λ + (W − 1)Ξ(L+q ) + (M − 1)Ξ(L−q ) .

When the two proxy sets capture the dominant mass of the corresponding entropy-increasing and entropy-decreasing quadrants, respectively, we obtain

dH¯ dη C2

= −

1 N

Λ − (W − 1)ΓL+ − (1 − M)ΓL− . This expression matches the Variant II batch entropy shift in the main text:

ΛV2 = Λ − (W − 1)Γ+ − (1 − M)Γ−, where

Γ+ = ΓL+, Γ− = ΓL−.

- (3) Combination C3: attenuating Uq+ and amplifying Uq−. C3 regulates the low-surprisal region from both advantage sides. It weakens the entropy-decreasing signal from positive-advantage low-surprisal tokens and strengthens the entropy-increasing signal from negative-advantage low-surprisal tokens. Define

ωi(,C3t ) =

 



M+, (i, t) ∈ Uq+, W−, (i, t) ∈ Uq−, 1, otherwise,

0 < M+ < 1, W− > 1.

By Proposition H.1,

dH¯ dη C3

= −

1 N

Λ + (M+ − 1)Ξ(Uq+) + (W− − 1)Ξ(Uq−) .

When the proxy sets align with their target quadrants and capture the dominant contributions, we have dH¯ dη C3

= −

1 N

Λ − (1 − M+)ΓU+ − (W− − 1)ΓU− .

- (4) Combination C4: attenuating Uq+ and attenuating L−q . C4 attenuates the two dominant entropydecreasing token categories. Define

 

M+, (i, t) ∈ Uq+, M−, (i, t) ∈ L−q , 1, otherwise,

ωi(,C4t ) =

###### 0 < M+ < 1, 0 < M− < 1.



- By Proposition H.1, dH¯

1 N

###### Λ + (M+ − 1)Ξ(Uq+) + (M− − 1)Ξ(L−q ) .

= −

dη C4

When the two proxy sets capture the dominant mass of their corresponding entropy-decreasing quadrants, we obtain

dH¯ dη C4

1 N

###### Λ − (1 − M+)ΓU+ − (1 − M−)ΓL− .

= −

Table 9: Summary of O1 to O4 and C1 to C4. The Target subset column specifies the entropy quadrant or pair of quadrants selected by each operation.

Notation Type Selected Subset Weight Adjustment Mechanistic Interpretation

- O1 Single-polarity L+q Amplification, W+ > 1 Strengthens high-surprisal tokens in positive-advantage samples
- O2 Single-polarity Uq+ Attenuation, 0 < M+ < 1

Weakens low-surprisal tokens in positiveadvantage samples

- O3 Single-polarity Uq− Amplification, W− > 1 Strengthens the suppression of lowsurprisal tokens under negative advantages
- O4 Single-polarity L−q Attenuation, 0 < M− < 1

Weakens the suppression of high-surprisal tokens under negative advantages

C1 Combined L+q + Uq− Joint amplification Strengthens entropy-increasing token cate-

gories on both advantage sides

C2 Combined L+q + L−q Amplifies the former and attenuates the latter

Strengthens positive-advantage highsurprisal tokens while weakening negative-advantage high-surprisal entropydecreasing tokens

C3 Combined Uq+ + Uq− Attenuates the former and amplifies the latter

Jointly regulates the low-surprisal region on both advantage sides

C4 Combined Uq+ + L−q Joint attenuation Suppresses entropy-decreasing token cate-

gories on both advantage sides

###### H.3 Summary of Single-Polarity and Combined Operations

For clarity, we summarize the single-polarity operations O1 to O4 and the combined operations C1 to C4 below.

Single-polarity operations act on one entropy quadrant:

- • O1 amplifies L+q , the positive-advantage high-surprisal set. It strengthens the entropy-increasing minority under positive advantages.
- • O2 attenuates Uq+, the positive-advantage low-surprisal set. It weakens the dominant entropydecreasing signal under positive advantages.
- • O3 amplifies Uq−, the negative-advantage low-surprisal set. It strengthens the entropy-increasing effect induced by suppressing low-surprisal tokens.
- • O4 attenuates L−q , the negative-advantage high-surprisal set. It weakens the entropy-decreasing effect induced by suppressing high-surprisal tokens.

Under the four-quadrant analysis in Section 3.2, O1 and O3 amplify entropy-increasing signals, whereas O2 and O4 attenuate entropy-decreasing signals. Variant I in the main text is a direct instance of O1, since

it only amplifies token weights in L+q . Combined operations act on two entropy quadrants:

- • C1 jointly amplifies L+q and Uq−. It strengthens entropy-increasing signals on both the positiveand negative-advantage sides.
- • C2 amplifies L+q and attenuates L−q . It simultaneously strengthens positive-advantage highsurprisal entropy-increasing tokens and weakens negative-advantage high-surprisal entropydecreasing tokens.
- • C3 attenuates Uq+ and amplifies Uq−. It regulates the low-surprisal region by weakening positiveadvantage entropy-decreasing tokens and strengthening negative-advantage entropy-increasing tokens.
- • C4 jointly attenuates Uq+ and L−q . It suppresses the two dominant entropy-decreasing token categories.

Variant II in the main text is a representative instance of C2. It combines amplification of L+q with attenuation of L−q . Since all weights remain positive, the sign of each original GRPO policy-gradient update is preserved. The method only rescales token-level contribution magnitudes, thereby regulating

batch-level entropy dynamics through two complementary mechanisms: amplifying entropy-increasing contributions and attenuating entropy-decreasing contributions.

Single-polarity operations implement targeted interventions on individual entropy quadrants, while combined operations perform joint regulation across two quadrants.

Here, L+q and L−q denote high-surprisal proxy subsets within positive- and negative-advantage tokens, respectively, while Uq+ and Uq− denote low-surprisal proxy subsets within positive- and negative-advantage tokens. Amplification refers to assigning weights larger than one, whereas attenuation refers to assigning weights in (0,1).

###### I Limitations and Broader Impacts

Our empirical evaluation may not exhaustively cover all possible task distributions and optimization settings. Moreover, similar to other LLMs, our trained model could also generate potentially unethical or misleading information sometimes. We hope our work contributes positively to the development of more reliable post-training paradigms for large language models.

