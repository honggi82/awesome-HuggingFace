# arXiv:2512.01715v1[cs.RO]1Dec2025

∝ ∝

∝

∝BeingBeyond

BeingBeyond

智在⽆界

智在⽆界

## DiG-Flow: Discrepancy-Guided Flow Matching for Robust VLA Models

#### Wanpeng Zhang1,2 Ye Wang2,3 Hao Luo1,2 Haoqi Yuan1,2 Yicheng Feng1,2 Sipeng Zheng2 Qin Jin3 Zongqing Lu1,2,†

1Peking University 2BeingBeyond 3Renmin University of China https://beingbeyond.github.io/DiG-Flow

#### Abstract

Vision-Language-Action (VLA) models trained with flow matching have demonstrated impressive capabilities on robotic manipulation tasks. However, their performance often degrades under distribution shift and on complex multi-step tasks, suggesting that the learned representations may not robustly capture task-relevant semantics. We introduce DiG-Flow, a principled framework that enhances VLA robustness through geometric regularization. Our key insight is that the distributional discrepancy between observation and action embeddings provides a meaningful geometric signal: lower transport cost indicates compatible representations, while higher cost suggests potential misalignment. DiG-Flow computes a discrepancy measure between empirical distributions of observation and action embeddings, maps it to a modulation weight via a monotone function, and applies residual updates to the observation embeddings before flow matching. Crucially, this intervention operates at the representation level without modifying the flow matching path or target vector field. We provide theoretical guarantees showing that discrepancy-guided training provably decreases the training objective, and that guided inference refinement converges with contraction. Empirically, DiG-Flow integrates into existing VLA architectures with negligible overhead and consistently improves performance, with particularly pronounced gains on complex multi-step tasks and under limited training data.

Date: Dec 1, 2025

#### 1 Introduction

Vision-Language-Action models represent a paradigm shift in robotic learning by leveraging pretrained vision-language representations to enable flexible, instruction-following manipulation policies [1–3]. By combining powerful vision-language backbones with flow matching or diffusion-based action generation, these models have achieved impressive success across diverse tasks. However, recent studies reveal concerning fragility: performance degrades substantially under modest distribution shifts such as lighting changes, texture variations, or camera angle perturbations [2, 4]. This problem is particularly pronounced on complex tasks where sequential decisions compound, and errors in early steps cascade into failure.

The fundamental question is whether the learned representations robustly capture task-relevant semantics

†Correspondence to Zongqing Lu <lu@beingbeyond.com>.

or inadvertently encode spurious patterns. Flow matching provides an elegant framework for learning action distributions by regressing neural vector fields [3, 5], yet the regression objective alone may not sufficiently incentivize semantically grounded representations. We propose a complementary perspective: the geometric relationship between observation features and action embeddings reveals representational quality. When observation features from the vision-language backbone and action embeddings from the policy head exhibit low distributional discrepancy, their representations are geometrically compatible, suggesting semantic coherence. Conversely, high discrepancy signals potential misalignment that may indicate spurious patterns or out-of-distribution observations.

To solve this problem, we introduce the DiG-Flow framework with 3 main components: a discrepancy function that quantifies distributional distance, a monotone weight mapping that transforms discrepancy into a modulation factor, and a lightweight residual operator that adjusts observation features. The framework defaults to Wasserstein distance for its geometric interpretability but accommodates alternative discrepancies. Importantly, DiG-Flow intervenes at the representation level before flow matching, leaving the probability path and target vector field unchanged. This design enables seamless integration into existing architectures while providing theoretical guarantees on training dynamics and inference convergence.

Our contributions are threefold. First, we establish a discrepancy-guided framework for robust VLA learning. Second, we discuss theoretical motivation for the general problem and provide theoretical guarantee for our method design. Third, we validate that DiG-Flow consistently improves performance across both simulation and real-world benchmarks, with pronounced gains on complex multi-step tasks and in data-limited scenarios, while maintaining negligible computational overhead.

#### 2 Related Work

Vision-Language-Action Models. Vision Language Models (VLMs) [6, 7] typically combine LLM reasoning [8–10] with modal-specialized encoders [11–14] for unified multimodal understanding [15, 16]. Pioneering works [17–20] demonstrated strong few-shot instruction-following capability. With these visual backbones, VLA models extend them to generate robot actions from visual observations and language instructions, representing a paradigm shift in robot learning [1–3, 21, 22]. Early work like RT-1 [1] demonstrated transformer-based architectures could scale to diverse tasks, while RT-2 [21] showed web-scale vision-language pretraining improves robotic control. Recent advances follow two primary approaches: (1) Direct action prediction: OpenVLA [2] scales to 7B parameters with open-source datasets, Pi0 [3] and its enhanced version Pi0.5 [4] use flow matching for continuous control, while models like Octo [23] provide generalist policies across embodiments; (2) Visual prediction: PaLM-E [22] embeds sensors into language models, GR-3 [24, 25] uses video pretraining for action generation. Specialized variants address specific challenges: SpatialVLA [26] incorporates 3D spatial representations, Otter builds fine-grained visual feature extraction [27], Pi0-Fast [28] improves inference efficiency through action tokenization, gr00t-N1 [29] targets humanoid control, and CoT-VLA [30] adds visual chain-of-thought reasoning. Being-H0 [31] incorporates physical instruction tuning for better motion understanding. OneTwoVLA proposes a unified architecture for better adaptive reasoning [32]. Despite these advances, long-horizon performance and robustness to distributional drift remain open challenges, which we address through geometric regularization.

Flow Matching and Optimal Transport. Flow Matching [5] trains neural ODEs without simulation, directly learning vector fields that transform distributions. This paradigm improves on Continuous Normalizing Flows [33–35] which require expensive ODE solving. Recent developments include Conditional Flow Matching [36] for simplified training with conditional distributions, Rectified Flow [37] for straighter transport paths, and Stochastic Interpolants [38] unifying flows and diffusions. OT-CFM [39, 40] uses optimal transport for trajectory optimization, finding better paths between distributions at the level of flow matching dynamics.

Optimal transport theory [41, 42] provides principled tools for comparing distributions geometrically. Computational advances have made OT practical for machine learning: Sinkhorn distances [43] add entropic regularization for efficient approximation, while Sliced Wasserstein [44, 45] projects to 1D for linear complexity in the number of samples. Our key innovation differs from prior OT applications in VLA: rather than using optimal transport to modify the flow matching trajectory (as in OT-CFM), we use Wasserstein distance as an

Modulated Output input

|Foundation Model<br><br>(Pretrained VLM Backbone)<br><br>|Layer n| |
|---|---|
| | |
<br><br>|Layer n-1| |
|---|---|
| | |
<br><br>|Layer n-2| |
|---|---|
| | |
<br><br>|DiG block|
|---|
<br><br>| | |
|---|---|
|obs proj.| |
<br><br>|act proj.| |
|---|---|
| | |
<br><br>|Flow Matching<br><br>(Action Expert Head)<br><br>|DiG-Refine|
|---|
<br><br>action<br><br>noise|
|---|---|

### →

| |act/obs proj.|
|---|---|
| | |

| | |
|---|---|
|Embedding| |
| | |

| | |
|---|---|
|FFN| |

| | |
|---|---|
|| | |
|---|---|
|Discrepancy: D(µH,µZ)| |
<br><br>DiG Block<br><br>|Gate: g = ω(D)|
|---|
| |

|LayerNorm| |
|---|---|
| | |

|MHA(Q, K, V)| |
|---|---|
| | |

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

|Prompt: Turn on the stove and put the pot. Pick the cup and put it on the plate.|
|---|

### →

|LayerNorm|
|---|

Proprio state

###### (a) (b)

- Figure 1: Overview of DiG-Flow and integration of the DiG-Block. (a) We attach the DiG-Block in the pretrained VLM backbone. Observations are projected into a shared discrepancy space (obs proj.), while ground-truth (for training) or predicted actions (for inference) are mapped into the same space (act proj.). From these two streams, the block estimates a transport-aware discrepancy D and converts it into a gate g = ϕ(D), which softly modulates the backbone features before they are passed to the flow-matching action head. The flow-matching head produces action trajectories from noise as in standard VLA models, and the DiG-Refine module can apply a small number of refinement steps entirely within the action head to further polish the predicted actions. (b) Given input embeddings, the backbone first applies standard attention. The post-attention features are then normalized and fed to DiG-Block, which uses the act/obs

projections to compute D(µH,µZ) and a gate g, and performs a gated residual update. This design allows DiG-Flow to modulate the high-level representation using discrepancy signals from actions, while keeping the pretrained backbone architecture and attention blocks intact.

auxiliary signal to modulate representation learning. This representation-side intervention complements the flow matching objective and enables robustness to distributional drift without altering the fundamental action generation dynamics.

Robustness in Robot Learning. Robustness to distribution shift and spurious correlations has been studied across robot learning contexts [46]. Domain randomization techniques [47] attempt to improve generalization by training under diverse conditions, while domain adaptation methods [48] explicitly align distributions between source and target environments. Recent work has explored ways to identify and mitigate reliance on spurious features through causal reasoning [49] or by learning invariant representations [50]. However, these approaches typically operate during training data collection [51] or through architectural constraints, rather than providing adaptive mechanisms that respond to distributional drift during deployment. Our geometric modulation approach offers a complementary perspective: by dynamically measuring and responding to observation-action alignment through Wasserstein distance, we enable the model to adjust its behavior as it encounters distributional shift in long-horizon tasks.

- 3 Preliminaries We establish notation and background in this section.

Flow Matching. Flow matching learns a time-dependent vector field vθ(x,t) for t ∈ [0,1] that transports a base distribution p0 to a data distribution p1 via the ODE: dxt

dt = vθ(xt,t),x0 ∼ p0. Training minimizes the squared regression loss against a target vector field v⋆:

1,x0 ∥vθ(xt,t) − v⋆(xt,t)∥2 , (1) where xt interpolates between x0 ∼ p0 and x1 ∼ p1 along a chosen probability path.

L(θ) = Et,x

VLA Architecture and Action Chunks. A VLA model processes multimodal observations o = (ovis,olang,oprop) comprising visual images, language instructions, and proprioceptive states. A vision-language backbone produces observation features: H = (h1,...,hT) ∈ RT×d, where T is the context length and d is the feature dimension. Current VLA implementations typically adopt an action chunk formulation: at decision time t, the policy predicts a sequence of future actions:

, (2)

at:t+K−1 = (at,at+1,...,at+K−1) ∈ RK×d

a

where K is the action horizon and da is the raw action dimension. After executing this chunk, the model re-observes the environment at time t+K and generates a new chunk, making the policy effectively memoryless between chunks. Flow matching generates these action chunks by conditioning the vector field on observation features: vθ(x,t|H).

Action Embeddings. To enable geometric comparison between observations and actions, we map raw actions to a latent space via an encoder f : Rd

→ Rd. For a raw action at, we obtain an action embedding: zt = f(at) ∈ Rd. During training, ground-truth actions from the dataset are encoded to form ground-truth action embeddings ztgt = f(agtt ). During inference, predicted actions are encoded as zˆt = f(ˆat). Collecting embeddings over the chunk horizon yields: Z = (z1,...,zK) ∈ RK×d. The encoder f is implemented as a lightweight linear projection and trained jointly with the policy. This design allows observations H and action embeddings Z to reside in a common feature space where geometric comparisons are meaningful.

a

Given feature sequences H ∈ RT×d and Z ∈ RK×d, we form empirical distributions:

1 K

1 T

T

K

, µZ =

, (3)

µH =

δh

δz

i

j

i=1

j=1

where δx denotes a Dirac mass at x. These distributions enable us to quantify the geometric relationship between observations and actions via distributional discrepancy measures.

#### 4 Method

We present DiG-Flow, a framework for enhancing VLA robustness through geometric regularization of representations. Figure 1(a) shows the overall structure: a pretrained vision-language backbone produces multimodal observation tokens, a lightweight DiG-Block sits at the interface between the backbone and the flow-matching head, and an optional DiG-Refine module operates purely inside the action expert head.

###### 4.1 Discrepancy-guided architecture

Discrepancy Function. We define a discrepancy function D : P(Rd) × P(Rd) → R+ that quantifies the geometric distance between probability distributions. Our default choice is the squared 2-Wasserstein distance:

∥x − y∥2 dγ(x,y), (4)

D(µH,µZ) = inf

γ∈Γ(µH,µZ)

where Γ(µH,µZ) denotes the set of couplings between µH and µZ. The Wasserstein distance provides a geometrically meaningful measure of distributional alignment: lower transport cost indicates that observation and action representations are compatible, while higher cost suggests potential misalignment.

For computational efficiency, we approximate W22 using the sliced Wasserstein distance. We sample M random unit directions ω1,...,ωM ∼ Uniform(Sd−1) and compute:

1 M

M

W22 (ωm⊤)#µH,(ωm⊤)#µZ , (5)

D(µH,µZ) ≈

m=1

where (ωm⊤)#µ denotes the pushforward of µ under projection onto ωm. Each 1D Wasserstein distance is computed via the quantile formula:

1

|FH−1(u) − FZ−1(u)|2du, (6)

W22 (ωm⊤)#µH,(ωm⊤)#µZ =

0

Training Inference

hobs x0 hobs x0

Gate Gate

##### D D

|g = ω(D)|
|---|

|g = ω(D)|
|---|

D(µH,µZ)

zgt zˆ z0 z1 zˆ

···

- Figure 2: Discrepancy-guided gating at training and inference time. Left (training). Observation embeddings hobs and ground-truth action embeddings zgt define empirical distributions (µH,µZ). The discrepancy D(µH,µZ) is mapped to a gate g = ϕ(D), which modulates a residual update on hobs and reweights the flow-matching loss via the gated objective J(θ) = E[g ℓ(θ)]. Right (inference). The same

mechanism is applied using encoded model predictions z1,...,zˆ instead of ground-truth actions. The gate is computed from the discrepancy between observation embeddings and predicted action embeddings and is then used either once or multiple steps inside the flow-matching head.

which reduces to sorting projected features and computing mean squared differences. Alternative discrepancies such as Sinkhorn divergence, maximum mean discrepancy, or cosine distance can be substituted within the same framework.

Weight Mapping. We map the discrepancy to a modulation weight via a monotone decreasing function ϕ : R+ → [gmin,1]. Concretely, we use an exponentially decaying map with a lower clip:

g = ϕ(D) = max{gmin,exp(−τD)}, (7)

where τ > 0 is a temperature parameter and gmin ∈ (0,1) prevents the gate from vanishing even for large discrepancies. Low-discrepancy (well-aligned) pairs thus receive weights close to 1, while high-discrepancy pairs are down-weighted but still retain a non-zero contribution. The map ϕ is monotone decreasing and Lipschitz continuous, matching Assumption 2.

Residual Operator. We define a lightweight residual operator R : RT×d → RT×d that transforms observation features. In our implementation, R is a single linear layer with spectral norm regularization:

R(H) = WRH + bR, (8)

where WR ∈ Rd×d and bR ∈ Rd are learned parameters, and ∥WR∥2 ≤ BR is controlled via spectral normalization.

The enhanced observation features are computed via a residual update modulated by the gate: H˜ = H + λ · g · R(H), (9)

where λ > 0 is the residual strength parameter. This design ensures that when g is large (low discrepancy), the residual adjustment is fully applied, while when g is small (high discrepancy), the adjustment is suppressed. The enhanced features H˜ are then fed to the flow matching action head in place of the original features H.

We integrate these components as a module that can be dropped into existing transformer layers without changing their backbone structure. Figure 1(b) illustrates how the DiG-Block is integrated between multi-head attention and the feedforward network.

- Algorithm 1 DiG-Flow Training

Require: Observation o, ground-truth actions agt = {agt1 ,...,agtK}, flow-matching network vθ, residual

operator R, M (sliced projections), τ (temperature), λ (residual strength), gmin (gate floor)

- 1: Extract observation features: H ← VLM(o) ∈ RT×d
- 2: Encode ground-truth actions: zkgt ← f(agtk ) for k = 1,...,K
- 3: Mean-pool and broadcast: z¯ ← K1 Kk=1 zkgt, replicate z¯ to length T to form Zgt ∈ RT×d

- 4: Compute discrepancy via sliced Wasserstein:
- 5: for m = 1 to M do
- 6: Sample unit direction ωm ∼ Uniform(Sd−1)
- 7: Project: πH(m) ← Hωm, πZ(m) ← Zgtωm
- 8: Sort: πH(m),↑ ← sort(πH(m)), πZ(m),↑ ← sort(πZ(m))
- 9: Accumulate: Dm ← T1 Ti=1(πH,i(m),↑ − πZ,i(m),↑)2

- 10: end for
- 11: D ← M1 Mm=1 Dm

- 12: Compute gate (stop gradient): g ← stop_grad(max{gmin,exp(−τD)})
- 13: Compute enhanced features: H˜ ← H + λ · g · R(H)
- 14: Compute flow-matching loss (conditional on H˜): ℓ ← Et,x

t|x0[∥vθ(xt,t | H˜) − v⋆(xt,t)∥2]

- 15: Compute gated objective: J ← g · ℓ
- 16: Backpropagate ∇θJ and update parameters

###### 4.2 Training with Transport-aware Gating

During training, DiG-Flow uses ground-truth action chunks to construct the geometric signal; the flowmatching objective itself remains unchanged. For a training sample (o,agt), the vision-language backbone produces observation features H = VLM(o) ∈ RT×d, and the action encoder maps ground-truth actions to embeddings zkgt = f(agtk ). We form an action semantic centroid by mean-pooling and broadcasting,

1 K

K

zkgt, Zgt = (¯z,...,z¯) ∈ RT×d, (10)

z¯ =

k=1

which induces empirical measures µH and µZgt. This design implies that we effectively model the target action representation as a degenerate empirical distribution concentrated at the semantic centroid z¯. This serves two purposes: (i) it summarizes the “semantic intent” of the action chunk, and (ii) it makes the discrepancy computation invariant to the temporal ordering of actions, focusing instead on global compatibility.

Crucially, we explicitly compute the discrepancy D(µH,µZgt) using the standard Sliced Wasserstein formulation (Algorithm 1). This approach interprets the observation features H as a distribution and measures the transport cost required to condense them into the action centroid, implicitly penalizing variance and ensuring geometric alignment. While the sorting of Zgt becomes an identity operation in this degenerate case, we retain the general sorting step in Algorithm 1 to maintain the standard formulation, allowing extensions to non-degenerate action distributions.

The gate is obtained by a monotone map g = max{gmin,exp(−τD)}, with gradients stopped through g. The observation features are then modulated by a lightweight residual operator,

H˜ = H + λg R(H), (11) and H˜ is fed into the flow-matching head. The final training objective is

J(θ) = E sg(g)ℓ(θ;H,t˜ ) , (12)

where ℓ is the per-sample flow-matching loss and sg(·) denotes the stop-gradient operator. Intuitively, g acts purely as a data-dependent confidence weight: it suppresses shortcut-like examples with large discrepancy, while avoiding trivial solutions where the backbone collapses features solely to minimize D. Figure 2 summarizes this mechanism. All parameters of the backbone, action encoder, residual operator, and flow-matching head are optimized jointly by differentiating J(θ).

###### 4.3 Inference-time Refinement

- Algorithm 2 DiG-Flow Inference with optional refinement Require: Observation o, number of refinement iterations Nrefine

- 1: Extract observation features: H ← VLM(o)
- 2: Generate initial action chunk: a(0) ← FlowModel(H)
- 3: for i = 1 to Nrefine do
- 4: Encode previous prediction: Z(i−1) ← {f(a(ki−1))}Kk=1, aligned to length T
- 5: Compute discrepancy: D(i−1) ← D(µH,µZ(i−1))
- 6: Compute gate: g(i−1) ← max{gmin,exp(−τD(i−1))}
- 7: Enhance features: H˜(i−1) ← H + λ · g(i−1) · R(H)
- 8: Generate refined action: a(i) ← FlowModel(H˜(i−1))
- 9: end for
- 10: return a(N

refine)

At inference time, our default configuration mirrors the training-time modulation but avoids extra iterations. Given an observation o, the model first computes H = VLM(o) and predicts an action chunk from the flow-matching head. The predicted actions from the previous decision step are encoded and summarized in the same way as during training, yielding an action embedding sequence Z and empirical measure µZ. The discrepancy is used to form enhanced features H˜ = H + λg R(H), from which the next action chunk is generated. This single-pass, training-style enhancement is used for all main results.

When additional compute is available, DiG-Flow also supports an optional iterative refinement scheme. Starting from an initial prediction a(0) produced using H, we can repeatedly encode a(i), recompute D(i) and g(i), form H˜(i) = H + λg(i) R(H), and obtain a refined action a(i+1) from the flow-matching head. In practice, performance saturates within 2–3 refinement steps. A fixed-gate variant of this refinement admits a contraction guarantee (Theorem 3). Algorithm 2 describes inference-time usage. We first run the base policy once to obtain an initial action chunk a(0). Refinement iterations then reuse the same discrepancy function D(µH,µZ(i−1)), but now Z(i−1) is computed from the previous prediction instead of the ground truth. The gate g(i−1) controls how much residual correction is applied to H before feeding it into the flow model again. We call this procedure DiG-Refine.

#### 5 Theoretical Analysis

###### 5.1 Optimization and refinement guarantees

We now establish theoretical properties of DiG-Flow that clarify how discrepancy-guided modulation affects optimization and inference. Our analysis addresses three questions that correspond directly to the design choices in DiG-Flow: (i) does training with discrepancy-guided gates define a well-behaved optimization objective? (ii) do the residual feature updates help rather than hurt? (iii) can a simple refinement scheme converge? We first state the main results and discuss their implications; formal assumptions and proofs appear in Appendix A.

Theorem 1 Gated descent on the weighted objective

Consider the gated flow-matching objective J(θ) = E g ℓ(θ;H,t) , where ℓ(θ;H,t) is the per-sample flowmatching loss and g = ϕ D(µH,µZ) ∈ [gmin,1] is a data-dependent but parameter-independent weight (i.e., gradients are stopped through g when differentiating with respect to θ). Assume that J is LJ-smooth in θ. Then for any step size 0 < α < 2/LJ, the gradient descent update θ+ = θ − α ∇θJ(θ) satisfies

J(θ+) ≤ J(θ) − c1 ∥∇θJ(θ)∥2, (13) where c1 = α 1 − αL

2 > 0. Moreover, under Assumption 2, for all θ,

J

gmin L(θ) ≤ J(θ) ≤ L(θ), (14) so minimizing J controls the original flow-matching loss L up to the fixed factor 1/gmin.

- Theorem 1 shows that training with discrepancy-guided gates yields a standard smooth optimization problem: gradient descent on J enjoys the usual descent guarantee (13), and J uniformly brackets the original loss L via (14). In other words, the gates reweight contributions of different samples while keeping J and L uniformly equivalent as objective functions. Treating g as parameter-independent (via stop-gradient) avoids second-order cross terms and preserves this standard descent behaviour. In practice, the observation features H and action embeddings Z are updated jointly with θ, so the gates g evolve over training. Theorem 1 therefore

characterizes the dynamics of the pseudo-gradient E[g ∇θℓ] under a fixed-gate idealization, which matches the implemented stop-gradient update and is empirically observed to lead to stable and robust training.

Theorem 2 Residual update improvement

Suppose Assumption 3 holds, and assume there exists α0 > 0 such that the residual operator satisfies the gated descent condition

E g ⟨∇Hℓ(θ;H,t), R(H)⟩ ≤ −α0. (15)

Then there exists a threshold λmax > 0 such that for all residual strengths 0 < λ ≤ λmax, the gated residual update H˜ = H + λg R(H) satisfies

E ℓ(θ;H,t˜ ) ≤ E ℓ(θ;H,t) − β λ, (16) with β = α0/2 > 0.

- Theorem 2 addresses the effect of the residual operator on the loss. The gated descent condition (15) states that, on average, the direction R(H) aligns with the negative feature gradient once weighted by the gate g. Under smoothness of ℓ in H, sufficiently small gated residual steps strictly decrease the expected loss: the linear improvement from moving in a descent direction dominates the quadratic penalty for small enough λ. In practice, R is a small linear network trained jointly with the policy and implemented with spectral normalization. Because its parameters are optimized to minimize the gated objective J(θ), the network is encouraged to produce residual directions that correlate with loss reduction: if R(H) were uncorrelated with

∇Hℓ, it would contribute noise rather than useful signal and be effectively regularized away. We therefore view Eq. (15) as a structural assumption capturing the typical behaviour of a jointly trained residual operator, rather than an externally imposed constraint.

Theorem 3 Fixed-gate refinement convergence Consider an iterative refinement update in action-embedding space:

Z(k+1) = Z(k) − αE(Z(k);g), (17)

where the gate g is fixed and E(·;g) represents an error-correction field. If E(·;g) is LE-Lipschitz and µstrongly monotone (see Appendix A.1), then for step sizes 0 < α < 2µ/L2E, the update defines a contraction mapping with rate ρ ∈ (0,1):

∥Z(k) − Z⋆∥F ≤ ρk∥Z(0) − Z⋆∥F, (18) where Z⋆ is the unique fixed point.

- Theorem 3 isolates a fixed-gate refinement scheme and shows that it admits a contraction guarantee under standard conditions. When the refinement field is Lipschitz and strongly monotone (e.g., the gradient of a strongly convex functional), the refinement mapping contracts toward a unique fixed point, with the error decaying at rate ρk. Our implementation uses a more practical variant that recomputes g from the current prediction rather than keeping it fixed. Empirically, as the predicted actions improve, the discrepancy

D(µH,µZ(i)) decreases and the sequence of gates {g(i)} quickly stabilizes; the local dynamics then behave like a contraction with a nearly fixed gate, and performance saturates within 3 refinement steps (Section 6.3.4). We therefore view iterative refinement as an optional, well-behaved post-processing step whose behaviour is well captured by this idealized fixed-g analysis.

Shortcut Dist.

Dhigh

[Figure 5]

Dmid

[Figure 6]

Dlow

|Gate<br><br>g = ω(D)|
|---|

Noise Dist. Action Dist.

- Figure 3: DiG-Flow detects and suppresses shortcut learning through transport-guided gating. Flow matching transforms noise into actions, and multiple solutions can achieve low training loss. Some of these solutions correspond to semantically meaningful alignments between observations and actions (blue distribution), while others exploit spurious correlations or shortcuts (red distribution). DiG-Flow measures the Wasserstein distance D between observation and action features to distinguish these pathways. The gate g = ϕ(D) selectively modulates learning: Green path (Dlow): semantically aligned features with low transport distance receive strong gates, reinforcing robust behavior; Yellow path (Dmid): intermediate features receive moderate gating; Red path (Dhigh): shortcut-like patterns with high transport distance are down-weighted, discouraging spurious solutions. This geometric reweighting complements the flow-matching loss and helps steer optimization toward representations that remain stable under distribution shift. In this way, DiG-Flow pushes more intelligence toward the foundation model, making VLAs generate more robust actions for general manipulation.

###### 5.2 Theoretical Intuition

The previous subsection establishes that the gated objective J(θ) is smooth, that the residual update yields improvement for λ in a nontrivial interval, and that the refinement field defines a contraction. We now give a more explicit intuition in terms of how the discrepancy-controlled gate with above properties reshapes the loss landscape.

Dataset decomposition by discrepancy. Let D(H,Z) denote the discrepancy between observation features and action embeddings for a particular sample, and let g = ϕ(D) with ϕ monotone decreasing. Consider partitioning the dataset into two regions:

A = {(H,Z) : D(H,Z) ≤ δ} (geometrically aligned), (19) S = {(H,Z) : D(H,Z) > δ} (geometrically misaligned / shortcut-like), (20)

for some threshold δ > 0. Denote the corresponding probabilities by

pA = Pr[(H,Z) ∈ A], pS = Pr[(H,Z) ∈ S], pA + pS = 1. (21) Using this decomposition, the gated objective can be written as

J(θ) = E g ℓ(θ;H,t) (22)

= pA E g ℓ(θ;H,t) | A + pS E g ℓ(θ;H,t) | S , (23) and its gradient decomposes as

∇θJ(θ) = pA E g ∇θℓ(θ;H,t) | A + pS E g ∇θℓ(θ;H,t) | S . (24) Because ϕ is decreasing in D, we typically have

E[g | A] ≈ 1, E[g | S] ≪ 1 (25)

whenever aligned pairs exhibit small discrepancy and shortcut-like pairs exhibit large discrepancy. Thus, even if the raw gradients E[∇θℓ | A] and E[∇θℓ | S] have comparable magnitudes, the effective contributions to the gated gradient in (24) are reweighted by the expected gates. As optimization proceeds, J(θ) therefore behaves as if it were more strongly influenced by the aligned region A and less by the shortcut region S.

Connection to the residual update condition. The residual update analysis for Theorem 2 uses the alignment condition

E g ⟨∇Hℓ(θ;H,t),R(H)⟩ ≤ −α0. (26) Using the same decomposition A ∪ S, this can be rewritten as

pA E g ⟨∇Hℓ(θ;H,t),R(H)⟩ | A + pS E g ⟨∇Hℓ(θ;H,t),R(H)⟩ | S ≤ −α0. (27)

If the residual operator R is designed to follow the descent direction of the flow-matching loss on aligned samples (so that E[⟨∇Hℓ,R(H)⟩ | A] < 0), then the gate g further amplifies these contributions relative to those from S. Intuitively, the residual update “trusts” the directions suggested by geometrically consistent pairs and downweights those arising from high-discrepancy configurations. This is exactly the intuition formalized in Theorem 2: for small enough λ, the first-order improvement from aligned regions dominates the second-order smoothness penalty.

Refinement as gated fixed-point iteration. Finally, the refinement operator T(Z) = Z − α E(Z;g) can be viewed as a fixed-point iteration on the space of action embeddings. Under strong monotonicity and Lipschitz regularity, Theorem 3 shows that T is a contraction. In practice, this means that once the gate g has identified “reliable” observation–action configurations, the refinement map can iteratively pull predicted actions toward a geometry-consistent fixed point Z⋆ without exploding or oscillating. The contraction rate ρ < 1 quantifies how quickly refinement concentrates probability mass around Z⋆.

- Figure 3 summarizes the overall picture: discrepancy-guided gating reshapes the effective gradient contributions of different regions of the data space, the residual update exploits this reweighting to improve features in a controlled range of strengths, and the refinement operator provides a stable way to further align actions with observation geometry at inference time.

###### 5.3 Approximation stability of transport-based discrepancy

Finally, we discuss how the theoretical guarantees behave when the exact 2-Wasserstein distance is replaced by practical approximations such as sliced Wasserstein or Sinkhorn divergence.

Sliced Wasserstein concentration. Assume that observation and action embeddings lie in a bounded ball of radius R in Rd. For each random projection direction ωm ∈ Sd−1, the corresponding one-dimensional Wasserstein distance between the projected empirical measures is bounded by (2R)2. The sliced Wasserstein estimator averages M such terms, so by Hoeffding’s inequality, for any ϵ > 0,

Mϵ2 2 · (2R)4

. (28)

Pr | D − E[ D]| > ϵ ≤ 2exp −

Thus the estimation error scales as O(M−1/2) with high probability. Since the gate is obtained via a Lipschitz weight map g = ϕ(D) with constant Lϕ (Assumption 2), we have

|g − E[g]| = |ϕ( D) − ϕ(E[ D])| ≤ Lϕ | D − E[ D]|, (29)

so the gate error inherits the same O(M−1/2) concentration rate. In turn, the descent and contraction inequalities in Theorems 1–3 remain valid up to multiplicative constants that depend smoothly on Lϕ.

Sinkhorn divergence. For the Sinkhorn divergence Sε with entropic regularization ε > 0, standard results on entropic optimal transport show that

Sε − W22 ≤ b(ε), b(ε) → 0 as ε → 0, (30)

and empirical estimators of Sε concentrate around their population values at rate O(n−1/2) in the number of samples n, with constants depending on ε. Propagating this perturbation through the Lipschitz map ϕ again yields a bounded perturbation of the gate:

ϕ( Sε) − ϕ(W22) ≤ Lϕ b(ε) + C(ε)n−1/2 , (31)

for some constant C(ε). Consequently, using Sinkhorn divergence instead of exact Wasserstein distance only affects the numerical constants in our bounds; the qualitative guarantees (existence of a positive descent region in λ, and contraction of the fixed-gate refinement) remain unchanged.

#### 6 Experiments

We evaluate DiG-Flow on challenging robotic manipulation benchmarks to assess its ability to improve robustness and generalization. Our experiments are designed to answer three questions:

- 1. Does discrepancy-guided flow matching improve VLA performance?
- 2. How does DiG-Flow compare to state-of-the-art VLA baselines?
- 3. Which design choices of DiG-Flow are most important?

To demonstrate that DiG-Flow is backbone-agnostic, we select two of the most widely used flow-matching based VLA models, i.e., π0.5 [4] and GR00T-N1 [29], as backbone architectures to evaluate the effectiveness of DiG-Flow across different model designs. For π0.5, we follow the official default configuration. We then insert DiG-Block before the final transformer layer of the VLM backbone, sharing the same observation features as the original policy head. For GR00T-N1, we use the standard tabletop configuration and insert DiG-Block at the final token representation used by the action head. In both models, DiG-Block operates purely at the representation level and does not modify the underlying flow-matching architecture or training objective. We refer to the two variants as π0.5-DiG and GR00T-N1-DiG, respectively.

- 6.1 Simulation Experiments

- 6.1.1 Experimental Setup

LIBERO Benchmark. LIBERO [52] is a standardized benchmark for tabletop robotic manipulation with language-conditioned tasks. It consists of four 10-task suites: LIBERO-Spatial (spatial reasoning and object placement), LIBERO-Object (diverse object properties and appearances), LIBERO-Goal (goal-conditioned instructions), and LIBERO-Long (long-horizon, multi-step tasks). Models are evaluated by success rate over 50 rollouts per task, following the official protocol.

RoboCasa Benchmark. RoboCasa [53] is a scalable household robotics simulation platform built on NVIDIA Isaac Sim, providing photorealistic kitchen scenes and diverse manipulation tasks such as pick-andplace, opening cabinets, and interacting with appliances. We follow the 24 atomic manipulation tasks used in prior work on RoboCasa tabletop evaluation, which group naturally into three categories: (i) pick-and-place, (ii) opening and closing doors or drawers, and (iii) other manipulation skills such as pressing buttons or sliding objects. Each task comes with 50 human demonstrations and thousands of additional synthetic rollouts; to stress generalization in a low-data regime, we train all models using only 50 demonstrations per task (one trajectory per demonstration), leaving the synthetic data unused.

Training details. Unless otherwise specified, we use the same optimizer and data pipeline as the corresponding backbone. For π0.5-DiG on LIBERO, we train for 30K steps using AdamW with a global batch size of 256 across 8 A100 GPUs, a peak learning rate of 5 × 10−5, and cosine decay. The flow-matching head follows the standard squared regression loss and training-time enhancement. By default, our DiG-Block modules use 32 sliced projections for the discrepancy, a temperature of τ = 1.0, residual strength λ = 0.4, and a spectral norm bound of BR = 2.0. For GR00T-N1-DiG, we follow the official GR00T-N1 training and evaluation protocol and similarly enable DiG-Block during fine-tuning.

###### 6.1.2 Main Results on LIBERO

Table 1 shows that DiG-Flow improves both backbones and achieves state-of-the-art results on LIBERO. On top of the strong π0.5 baseline, π0.5-DiG increases the average success rate from 96.9% to 98.3%, with the largest gain on LIBERO-Long (92.4%→96.4%, +4.0 points). This suite contains long-horizon, multi-step tasks, so the improvement supports our hypothesis that discrepancy-guided modulation reduces error accumulation by encouraging semantically aligned representations.

- Table 1: Success rates (%) on LIBERO. We report mean success over 50 evaluation episodes per task.

DiG-Flow consistently improves both the π0.5 and GR00T-N1 backbones. The largest gains appear on LIBERO-Long, which contains complex multi-step tasks.

Method LIBERO-Spatial LIBERO-Object LIBERO-Goal LIBERO-Long Average

Diffusion Policy [54] 78.5 87.5 73.5 64.8 76.1 SpatialVLA [26] 88.2 89.9 78.6 55.5 78.1 CoT-VLA [30] 87.5 91.6 87.6 69.0 83.9 OpenVLA [2] 84.7 88.4 79.2 53.7 76.5 OpenVLA-OFT [55] 97.6 98.4 97.9 94.5 97.1 GR00T-N1 [29] 94.4 97.6 93.0 90.6 93.9 π0 [3] 98.0 96.8 94.4 88.4 94.4 π0-Fast [28] 96.4 96.8 88.6 60.2 85.5 π0.5 [4] 98.8 98.2 98.0 92.4 96.9 GR00T-N1-DiG (Ours) 96.0 98.4 94.8 92.1 95.3 π0.5-DiG (Ours) 99.2 99.0 98.6 96.4 98.3

GR00T-N1-DiG exhibits similar behavior, improving the average success rate from 93.9% to 95.3%. Again, the gain is most pronounced on LIBERO-Long (90.6%→92.1%), suggesting that DiG-Flow consistently helps different architectures handle long-horizon reasoning. Importantly, the relative ordering of backbones is preserved: π0.5-DiG remains stronger than GR00T-N1-DiG, indicating that DiG-Flow acts as a plug-and-play robustness layer rather than fundamentally changing the underlying policy capacity.

###### 6.1.3 Main Results on RoboCasa (Few-Shot)

RoboCasa poses a substantially harder generalization challenge than LIBERO: scenes are more diverse, the observation space is more visually complex, and most tasks involve contact-rich manipulation in cluttered kitchens. Moreover, our training regime is intentionally low-data, using only 50 demonstrations per task, which is roughly the number of human demonstrations provided per atomic task in the official benchmark.

- As shown in Table 2, DiG-Flow yields large gains in this setting. For π0.5, integrating DiG-Flow improves the average success rate from 41.4% to 52.6% (+11.2 points). Improvements are visible in all three categories, but are especially pronounced for door/drawer manipulation (+15.6 points), where precise contact and long-horizon geometry are critical. GR00T-N1 also benefits: GR00T-N1-DiG improves the average success rate from 36.0% to 43.2% (+7.2 points). The fact that both backbones gain substantially from the same DiG-Flow module, under severely limited data, reinforces our interpretation of discrepancy as a useful geometric regularizer that reduces overfitting to spurious correlations.

- Table 2: RoboCasa 24-task benchmark with 50 demonstrations per task. We report success rates (%) averaged within categories and across all tasks. All methods are trained with only 50 demonstrations per task. DiG-Flow substantially improves performance in this challenging few-shot regime.

Method Pick & Place Doors / Drawers Others 24-Task Avg

π0.5 21.5 57.8 44.9 41.4 π0.5-DiG (Ours) 27.2 73.4 57.2 52.6 GR00T-N1 18.6 50.2 39.1 36.0 GR00T-N1-DiG (Ours) 22.4 60.3 47.0 43.2

6.1.4 Robustness Analysis (Simulation)

To further probe robustness in a controlled setting, we adopt non-stationary perturbations following previous standards in non-stationarity studies [49, 56]. These perturbations apply time-varying sinusoidal noise to both visual observations and proprioceptive states: c1 cos(c2t),c3 sin(c4t), with coefficients ci ∼ N(0.01,0.5). Such perturbations are designed to disrupt brittle correlations that depend on static visual patterns or stereotyped trajectories, while leaving the underlying task semantics intact.

- Table 3: Robustness to non-stationary perturbations. Success rates (%) under different time-varying noise patterns that disrupt memorized shortcuts. DiG-Flow shows consistent improvements across perturbation types, with the most significant gains on complex long-horizon tasks.

Perturbation Method LIBERO-Spatial LIBERO-Object LIBERO-Goal LIBERO-Long Avg Cosine only

w/o DiG-Flow 87.6 89.8 86.4 70.2 83.5 w/ DiG-Flow 87.4 -0.2% 89.8 0% 91.6 +6.0% 80.0 +14.0% 87.2 +4.4%

w/o DiG-Flow 88.2 88.4 87.2 68.6 83.1 w/ DiG-Flow 90.0 +2.0% 90.6 +2.5% 91.2 +4.6% 80.4 +17.2% 88.1 +6.0%

Sine only

w/o DiG-Flow 87.8 87.2 86.0 67.8 82.2 w/ DiG-Flow 86.4 -1.6% 88.2 +1.1% 90.8 +5.6% 79.2 +16.8% 86.2 +4.0%

Cosine + sine

- As shown in Table 3, DiG-Flow consistently improves robustness across all perturbation types, with average gains between roughly 4–6 points. The improvements are most dramatic on LIBERO-Long, where long-horizon execution is particularly sensitive to small systematic biases. Spatial suites show smaller changes, and in a few cases performance is slightly reduced, reflecting that not all perturbations exclusively target shortcuts. Overall, these results support the view that discrepancy-guided gating makes policies less reliant on brittle, non-stationary correlations and more grounded in the geometric relationship between observations and actions.

- 6.2 Real Robot Experiments

- 6.2.1 Real Robot Setup

We build a real robot setup similar to LIBERO and RoboCasa to validate our experiments in a more challenging real-world physical environment. We use a 7-DoF Franka Research 3 (FR3) arm equipped with a 6-DoF Inspire dexterous hand. Perception is provided by two Intel RealSense cameras mounted at complementary viewpoints, forming a stereo-like arrangement that reduces single-view occlusions. Both cameras stream synchronized RGB frames. The VLM backbone receives a fused representation of these views together with proprioceptive inputs (joint positions, velocities, and gripper states). Commands are sent to the low-level controller at 20Hz. Following the chunked-action protocol, the policy outputs action chunks over a short horizon, which are then executed in open-loop without additional model-predictive control or trajectory optimization. The overall hardware layout is illustrated in Figure 4(a).

Compared to a simple two-finger gripper, the dexterous hand substantially increases the control dimensionality and contact complexity: the policy must not only reach the correct pose, but also coordinate multiple finger joints for stable power and precision grasps, tool operation (e.g., spraying, wiping), and in-hand adjustments. This task setting further exposes policy errors: inaccurate predictions cannot be easily corrected by low-level planners, making robustness and alignment particularly important.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

###### (a) (b)

- Figure 4: (a) Real robot hardware setup (single-arm + dexterous hand. (b) Examples of real robot tasks. The two rows correspond to the two camera views.

To assess real-world applicability of DiG-Flow, we design four distinct tasks covering different manipulation challenges. These tasks impose realistic 3D geometry, contact dynamics, and visual complexity, making

- them ideal for testing the hypothesis that discrepancy-guided modulation improves semantic grounding and robustness. We now detail the four real-robot tasks. These tasks are designed to mirror the difficulty patterns observed in LIBERO and RoboCasa while exposing additional challenges from real-world physics, sensor noise, and human interference. The tasks’ example settings are shown in Figure 4(b).

- • Stack-Bowls: stack bowls from scattered initial positions into a tower; tests precise multi-object grasping and placement, as well as collision avoidance between the hand and previously stacked bowls.
- • Spray-Plant: pick up a spray bottle and apply water mist to a potted plant along its foliage; tests tool-use behavior, fine motion control, and spatial anticipation to cover the plant surface.
- • Wipe-Whiteboard: use a cloth to erase pen marks from a whiteboard; tests contact-rich surface interaction, maintaining stable pressure, and long-horizon sweeping motions.
- • Sort-Into-Drawer: pick three varied objects and place them into a drawer one by one; tests sequential multi-step reasoning, diverse grasping strategies, and error accumulation over multiple object transfers.

For each task we only collect 50 human-teleoperated demonstrations for training, and evaluate for both whole-task success and sub-task success. Stack-Bowls has 5 subtasks, while the other tasks have 3 subtasks each. We report overall success rates and detailed analysis on perturbation and robustness tests.

- Table 4: Success rates (%) of our models vs baseline on real robot tasks. We show sub-task / whole-task success rates.

Task Stack-Bowls Spray-Plant Wipe-Whiteboard Sort-Into-Drawer

π0.5 (baseline) 58 / 40 66 / 45 62 / 38 52 / 33 π0.5-DiG (Ours) 62 / 48 72 / 52 70 / 48 60 / 41

- Table 4 shows that DiG-Flow consistently improves performance across all four tasks. On the long-horizon Sort-Into-Drawer task, whole-task success increases from 33% to 41% (+8 pts), illustrating that DiG-Flow helps mitigate error accumulation in sequential decision chains. The more moderate gains on Spray-Plant and Wipe-Whiteboard reflect shorter horizons but still show benefit, particularly in sub-task success.

###### 6.2.2 Robustness Analysis (Real Robot)

We next analyze how DiG-Flow behaves under real-world perturbations that were not in the training demonstrations. The perturbed setups are illustrated in Figure 6, and the corresponding success rates are reported in Table 5.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- (a)
- (b)
- (c)
- (d)

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

###### Figure 5: Real robot rollouts on four manipulation tasks. Each row shows a successful execution by

π0.5-DiG from left to right. (a) Stack-Bowls: the robot sequentially grasps and stacks three bowls into a stable tower. (b) Spray-Plant: the robot picks up a spray bottle and waters the plant along its foliage. (c) Wipe-Whiteboard: the robot wipes away pen marks on the board using a cloth. (d) Sort-Into-Drawer: the robot picks and places three objects into a drawer, a long-horizon task that compounds errors across steps. These qualitative rollouts illustrate that DiG-Flow enables stable multi-step behaviours across diverse contact-rich settings.

- Table 5: Success rates (%) under unseen / perturbed conditions. We show sub-task / whole-task success rates.

Method Stack-Bowls (BG) Sort-Into-Drawer (BG) Spray-Plant (Human) Wipe-Whiteboard (Human)

π0.5 42 / 15 48 / 20 35 / 10 44 / 20 π0.5-DiG (Ours) 65 / 40 60 / 35 62 / 30 58 / 30

Background Shifts. For Stack-Bowls and Sort-Into-Drawer, we change the table cloth color and pattern, add additional objects near the workspace, and vary global lighting conditions. Despite these shifts, our policy with DiG-Flow maintains high success rates that are close to the unperturbed setting: most failures are due to rare extreme occlusions rather than systematic mistakes. Qualitatively, we observe that the robot continues to focus on the geometry of task-relevant objects (bowls, drawer, target items) even when the background appearance changes substantially, suggesting that the transport-based gating helps deprioritize accidental correlations with background textures.

Human Interference. For Spray-Plant and Wipe-Whiteboard, we introduce a human hand that moves the plant or writes new strokes on the board while the robot is executing the policy. These interventions create both visual distractions and true dynamical changes (targets moving during execution). We find that the policy often adapts by slightly adjusting its trajectory to follow the plant motion or to cover newly written strokes, and overall success remains high relative to the no-interference baseline. Failures typically arise when the human intervention is adversarially timed (e.g., moving the plant exactly as the robot closes its grasp), which is challenging for any open-loop controller.

- Table 5 shows that all four perturbed variants are challenging: both policies see noticeable drops compared to the clean setting, but π0.5-DiG remains consistently more robust. On Stack-Bowls (BG) and Spray-Plant (Human), the baseline whole-task success falls to 15% and 10%, while π0.5-DiG still achieves 40% and 30%, respectively. Even on the harder long-horizon Sort-Into-Drawer (BG) scenario, π0.5-DiG improves whole-task success from 20% to 35%. A similar advantage appears for Wipe-Whiteboard (Human), where π0.5-DiG reaches 30% vs. 20% for the baseline. Overall, the relative gap between the two models is preserved or enlarged under

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- (a)
- (b)

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

- Figure 6: Unseen and interfered real-world settings. We evaluate robustness under (a) Background Shifts, where table textures, cloths, and distractor objects are changed for Stack-Bowls and Sort-Into-Drawer, and (b) Human Interference, where a human hand moves the plant or writes on the whiteboard during wiping. These perturbations introduce visual and dynamic distribution shifts beyond the training data and correspond to the robustness results in Table 5.

background and human interference, indicating that discrepancy-guided modulation helps maintain stable observation-action alignment rather than overfitting to specific training scenes.

Overall, these results support the view that geometry-aware gating improves robustness in the real world: the policy is less sensitive to superficial background statistics and more focused on the observation–action alignment relevant for the task.

###### 6.2.3 High-DoF Humanoid with Active View Control

To further stress-test DiG-Flow on more complex embodiments, we deploy π0.5 and π0.5-DiG on a 31-DoF upper-body humanoid platform equipped with dual dexterous hands and an actively controlled head camera.

- As illustrated in Figure 7, the robot has independently actuated joints for the torso, neck, arms, and multi-finger hands, resulting in substantially higher dimensionality and contact complexity than traditional real-robot experiments. Unlike prior VLA settings that operate from a fixed third-person view and control only a low-DoF end-effector, our policy must simultaneously coordinate head, body, and both hands, making both state estimation and control more challenging.

The head houses a pair of RGB cameras that provide an egocentric, movable viewpoint. During teleoperation, the operator naturally moves the head to obtain task-relevant views. The policy is trained to imitate this behavior and thus learns to actively adjust the viewpoint while manipulating objects. Figure 8 shows an example rollout: the robot first orients the head to obtain a clear view of the workspace, then uses dexterous hands to pick up objects and place them into a box while continuously refining its viewing angle. This active-view setup is particularly sensitive to representation misalignment.

We collect 1000 teleoperated demonstrations of general tabletop clean-up tasks, where the robot is instructed to place 1-3 objects on the table into a box from diverse initial layouts. Both π0.5 and π0.5-DiG are trained on this dataset with a batch size of 256 for 20K training steps, using the same flow-matching configuration as in our other real-robot experiments. We evaluate each model on 20 rollouts per condition, and report success rates in Table 6. The “Seen” setting uses the same object categories and background as training, while the “Unseen” setting replaces both objects and background textures to induce distribution shift.

[Figure 60]

- Figure 7: High-DoF humanoid setup. Our 31-DoF humanoid platform with dual dexterous hands and an actively controlled head. The head mounts dual RGB cameras, while the torso and arms provide upper-body motion. The task is to clear 1-3 objects from the table into the box. This setting requires the policy to jointly coordinate head, body, and hands, rather than controlling only a low-DoF gripper from a fixed camera.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

- Figure 8: Egocentric active-view rollouts. Example sequence from the head-mounted camera during a clean-up episode. The humanoid first adjusts its viewpoint to obtain a clear observation of the workspace,

- then successively grasps objects with its dexterous hands and places them into the box. DiG-Flow helps maintain robust observation–action alignment under camera motion and self-occlusions.

On this challenging real-robot setting, the standard π0.5 baseline achieves only moderate performance, particularly when the scene contains multiple objects or unseen visual conditions. In contrast, π0.5-DiG consistently improves success rates by 5-10 points across seen objects and backgrounds. As for unseen objects and background shift, the gains are even more pronounced. The advantages brought by DiG-Flow are also more obvious in complex multi-object tasks.

These results indicate that discrepancy-guided modulation remains effective even when the policy coordinates a high-DoF humanoid with active view control, providing both stronger performance and more robust behavior under visual and dynamical perturbations.

###### 6.3 Further Discussion on Method Design

In this section, we further explain the method design of DiG-Flow through ablation studies and mechanism analyses on the standard LIBERO benchmark.

###### 6.3.1 Discrepancy and Gating Ablations

Our method has two key design choices: the discrepancy D(µH,µZ) that measures observation–action mismatch, and the scalar gate g = ϕ(D) that maps this signal back to the backbone. We therefore ablate these two components separately.

- Table 7 examines how different discrepancy functions affect performance when applied to π0.5-DiG on LIBERO. All variants use the same scalar mapping g = exp(−τD) and residual operator; only D(µH,µZ) is changed.

- Table 6: Success rates (%) on high-DoF humanoid clean-up tasks. Each entry reports the success rate over 20 evaluation rollouts for clearing 1-3 objects into the box. “Seen” uses the same object categories and background as the demonstrations, while “Unseen” uses novel objects and background textures.

Seen Unseen & Perturbed Method 1 obj 2 objs 3 objs 1 obj 2 objs 3 objs

π0.5 75 60 35 55 30 25 π0.5-DiG 75 65 45 65 45 40

- Table 7: Effect of discrepancy choice on LIBERO (success %). All rows use the same π0.5 backbone, scalar mapping g = exp(−τD), and residual operator; only the discrepancy D(µH,µZ) is varied. Wassersteinbased discrepancies provide the strongest signal, especially on LIBERO-Long.

Discrepancy LIBERO-Spatial LIBERO-Object LIBERO-Goal LIBERO-Long Avg

Cosine distance 98.2 -1.0% 97.8 -1.2% 97.2 -1.4% 92.6 -3.8% 96.5 -1.8% MMD (RBF kernel) 98.4 -0.8% 98.2 -0.8% 97.6 -1.0% 93.4 -3.0% 96.8 -1.5% Sinkhorn (ε = 0.1) 98.8 -0.4% 98.6 -0.4% 98.2 -0.4% 95.8 -0.6% 97.9 -0.4%

Sliced Wasserstein (Ours) 99.2 99.0 98.6 96.4 98.3

Non-transport discrepancies such as cosine distance and MMD already provide moderate gains over the backbone (average 96.5% and 96.8%), confirming that even coarse measures of observation–action similarity can be beneficial. However, Wasserstein-based discrepancies consistently perform better. Sinkhorn divergence (entropic OT) achieves 97.9% average success, while our default sliced 2-Wasserstein distance attains 98.3% and the best results on LIBERO-Long. This matches our theoretical intuition: Wasserstein distances respect the underlying geometry of the feature space, yielding a more faithful measure of distributional misalignment and a better-behaved supervision signal for the gate.

- Table 8: Comparison of gating mechanisms. Success rates (%) for different gating strategies with the discrepancy fixed to sliced 2-Wasserstein. In contrast to fixed or random modulation, our transport gate converts the geometric discrepancy into a meaningful confidence signal, yielding the best performance, especially on LIBERO-Long.

Gating Strategy LIBERO-Spatial LIBERO-Object LIBERO-Goal LIBERO-Long Avg

Fixed Gate (g = 0.5) 96.8 -2.4% 95.4 -3.6% 92.2 -6.5% 84.6 -12.2% 92.3 -6.1% Random Gate (g ∼ U(0,1)) 95.2 -4.0% 93.8 -5.3% 89.4 -9.3% 80.8 -16.2% 89.8 -8.7%

Transport Gate (Ours) 99.2 99.0 98.6 96.4 98.3

- Table 8 complements this study by fixing the sliced 2-Wasserstein discrepancy and varying only the gating mechanism. Replacing our transport gate with a fixed scalar (g = 0.5) reduces performance to 92.3% average success (-6.1%), indicating that uniform feature modulation cannot distinguish between semantically useful and spurious patterns. Random gating performs even worse at 89.8% (-8.7%), showing that arbitrary feature suppression actively harms learning by disrupting both beneficial and harmful correlations.

The degradation is particularly severe on LIBERO-Long (up to -16.2%), where long-horizon reasoning amplifies the effect of shortcut solutions. In contrast, the discrepancy-guided transport gate preserves performance on simpler suites and yields the largest gains on LIBERO-Long, consistent with our interpretation that geometric alignment is most critical in sequential tasks. Taken together, Tables 7 and 8 show that both the choice of discrepancy and the shape of the gate ϕ(D) matter for robust optimization.

###### 6.3.2 Refinement Ablation

In Section 5.1, we state that performance saturates within 3 refinement steps. Here we provide a more detailed analysis of this behavior on the standard LIBERO benchmark.

99.0

LIBEROsuccessrate(%)

| |
|---|

98.5

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

98.0

| |
|---|

| |
|---|

97.5

| |
|---|

97.0

0 1 2 3 4 5 6 7 8 9 10 Refinement steps Nrefine

Figure 9: Effect of refinement steps Nrefine on LIBERO. We vary the number of refinement iterations at inference time while keeping the training setup fixed. Even without refinement (Nrefine=0), the discrepancyguided flow model already surpasses the backbone policy. Adding a small number of refinement steps further improves performance and quickly saturates around Nrefine=3.

- Figure 9 reports the average LIBERO success rate when we vary the number of refinement iterations Nrefine at test time. We observe three consistent trends:

- • Performance gains even with Nrefine=0. The curve shows that the Nrefine=0 variant already achieves a strong success rate on LIBERO. This indicates that the discrepancy-guided residual update improves the learned policy itself, rather than only acting as a test-time heuristic.
- • Few refinement steps bring stable improvements. Increasing Nrefine from 0 to 3 yields consistent performance gains and quickly brings the curve close to saturation. This supports our choice of

using Nrefine=3 by default: it captures most of the benefit of refinement while keeping the additional computation per control step modest.

- • Beyond 3 steps, performance saturates with small fluctuations. For Nrefine > 3, the success rate fluctuates slightly around the same plateau level, without systematic degradation. We attribute the small variations to evaluation noise and the fact that once the flow model has already produced a near-optimal action chunk, further refinement only provides marginal adjustments.

Overall, these results confirm that (i) DiG-Flow is already useful without refinement, and (ii) a small number of refinement steps provides a good trade-off between robustness/performance and inference cost.

6.3.3 Hyperparameter Analysis

- Figure 10 summarizes the sensitivity of DiG-Flow to its main hyperparameters. Panel 10(a) varies the number of sliced projections K used to approximate the Wasserstein discrepancy. We observe a smooth, monotone improvement as K increases from 4 to 20, after which performance quickly saturates around K ≈ 28–32. This matches the theoretical trade-off: more projections reduce Monte Carlo variance in the sliced Wasserstein estimate, but beyond a moderate budget the marginal benefit becomes negligible relative to the dominant transformer cost. In all reported experiments we therefore fix K = 32, which lies in the flat regime and offers a good balance between accuracy and overhead.

Panel 10(b) studies the joint effect of the residual strength λ (in Eq.(11)) and the temperature τ in the gate g = exp(−τD). The heatmap reveals two important properties. First, there is a broad plateau around (λ,τ) ≈ (0.4,1.0), indicating that DiG-Flow is not overly sensitive to small mis-tuning and that our default choice lies in a robust operating region rather than at a sharp optimum. Second, the high-performance region forms an elongated ridge approximately following λτ ≈ const, which is consistent with the interpretation that λ and τ jointly control an effective gating strength λg(τ): increasing λ can be partially compensated by

(a) Sensitivity to K

(b) Joint effect of and

98.5

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

[Figure 65]

[Figure 66]

98.25

1.50

SuccessRate(%)

SuccessRate(%)

98.00

98.0

1.25

Temperature

97.75

1.00

97.5

97.50

97.25

0.75

97.0

97.00

0.50

96.75

96.5

4 8 12 16 20 24 28 32

0.20 0.30 0.40 0.50 0.60

Number of Sliced Projections K

Residual Strength

- Figure 10: Hyperparameter sensitivity of DiG-Flow. (a) Success rate as a function of the number of sliced projections K used for approximating the Wasserstein discrepancy. Performance improves steadily from very small K and saturates once K ≈ 28–32, indicating that a moderate number of projections is sufficient for a stable transport estimate. (b) Joint effect of the residual strength λ (in Eq. 11) and the temperature τ

in g = max{gmin,exp(−τD)}. We observe a broad plateau around (λ,τ) ≈ (0.4,1.0) and an elongated ridge roughly following λτ ≈ 0.4, showing that the two hyperparameters jointly control an effective gating strength while still providing some flexibility: increasing λ can be partially compensated by decreasing τ, and vice versa, but moving too far from the ridge degrades performance.

reducing τ (which softens the gate), and vice versa. The slight diagonal elongation reflects the fact that the effective step size in feature space scales like

∥H˜ − H∥F ≈ λg ∥R(H)∥F ≈ λ max{gmin,exp(−τD)}∥R(H)∥F, (32)

- At the same time, the ridge is not perfectly flat, so λ and τ are not fully redundant. In particular, large λ with aggressive τ can lead to over-amplified residuals that violate the small-step assumption in Theorem 2, while very small λ with extremely soft gates under-utilizes the residual pathway and reduces the advantage over the baseline. The heatmap therefore supports the theoretical picture: there exists a nontrivial band of (λ,τ) values where the transport-guided residual update yields consistent gains, and our defaults (λ = 0.4,τ = 1.0) sit well inside this stable band rather than being a knife-edge configuration.

###### 6.3.4 Transport-Cost Dynamics and Gating Behaviour

In addition to task-level success rates, we also monitor the behaviour of the transport discrepancy D(µH,µZ) during training. Figure 11 reports the evolution of the average sliced 2-Wasserstein cost between the empirical observation and action embedding distributions over training iterations for DiG-Flow on LIBERO.

Three observations are worth highlighting.

- (i) Training phases. At the beginning of training the transport cost is relatively high, reflecting the fact that observation features H and action embeddings Z live in poorly aligned regions of the feature space. As

optimisation proceeds, D(µH,µZ) decreases monotonically and then enters a steady-state regime where it fluctuates around a stable value. This behaviour is consistent with the interpretation of D as measuring semantic compatibility between observation and action representations: the model gradually learns to place both in a geometrically coherent region.

- (ii) D as a signal, not a loss term. Importantly, the curve in Figure 11 does not keep decreasing towards zero. Instead, it converges to a medium range where D is neither vanishing nor exploding. This is exactly the

regime in which the gate g = max gmin,exp −τ D(µH,µZ) retains discriminative power: samples with relatively low discrepancy still receive stronger gates, while samples with larger discrepancy are down-weighted, but no regime dominates completely. Driving D to (near-)zero would make g ≈ 1 for almost all samples and therefore remove the benefit of discrepancy-guided modulation; conversely, extremely large D would force g ≈ 0 and effectively deactivate the residual path.

Transport cost during training

Transport cost value

Smoothed curve

1.0

0.8

Transportcost

0.6

0.4

0.2

0.0

0 5000 10000 15000 20000 25000 30000

Training step

###### Figure 11: Dynamics of the transport discrepancy D(µH,µZ) during training. We plot the average

sliced W22 between observation and action embeddings over training iterations (shaded region indicates variability across batches). The cost decreases steadily in the early phase and then stabilizes in a medium range, instead of collapsing to zero, matching the intended role of D as a discriminative geometric signal rather than a pure minimization target.

- (iii) Connection to gated residual theory and hyperparameters. The observed range of D provides empirical support for the theoretical assumptions used in Section 5. Theorem 2 requires that the gated residual update H˜ = H + λg R(H) remains in a regime where the first-order improvement term dominates

the quadratic penalty controlled by LH and BR (Assumption 3). The fact that D stabilizes in a moderate range implies that the effective step size (Eq. (32)) naturally stays within such a “small but non-negligible” band throughout training, matching the regime in which Theorem 2 guarantees expected loss reduction. This also explains the shape of the hyperparameter landscape in Figure 10. Since D(µH,µZ) concentrates in a relatively narrow interval once training has converged, there exists a family of (λ,τ) pairs that produce similar effective magnitudes of λg on typical samples. The broad plateau in Figure 10(b) and the smooth one-dimensional curves over λ and τ are consistent with this picture: DiG-Flow is robust to moderate changes in either parameter, as long as the resulting transport cost keeps the gate in the informative middle range illustrated in Figure 11.

#### 7 Conclusions and Limitations

We presented DiG-Flow, a discrepancy-guided flow matching framework that regularizes VLA representations through a simple geometric signal between observation and action embeddings. By mapping this distributional discrepancy to a scalar gate and applying residual updates in feature space, DiG-Flow leaves the flow-matching path and objective intact while shaping the representations used by the action head. Our analysis shows that the gated objective admits standard descent guarantees, that suitably small residual updates provably reduce the loss under an alignment condition, and that a fixed-gate refinement scheme forms a contraction.

While DiG-Flow demonstrates strong performance in both simulation and real-world experiments, several limitations warrant discussion. First, our theoretical analysis assumes that features have bounded norms (e.g., ∥H∥F ≤ R). This is reasonable for normalized representations but may require additional scaling or normalization mechanisms when applied to architectures that do not enforce such bounds. Second, the transport distance computation currently relies on batch-level statistics, which can be influenced by outliers or small batch sizes. In principle, this could make the gate slightly sensitive to the composition of a mini-batch. Exploring alternatives such as running statistics, more robust discrepancy measures, or fully instance-wise variants would be an interesting direction for future work. Finally, DiG-Flow uses ground-truth actions during supervised training to compute meaningful observation–action discrepancies. Extending the same principle to self-supervised or reinforcement learning regimes would require alternative alignment signals (e.g., consistency across rollouts, value-based or advantage-based weighting), which we leave for future research.

#### References

- [1] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, KuangHuei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale, 2023.
- [2] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Baljekar, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.
- [3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.
- [4] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.
- [5] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2023.
- [6] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [7] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [8] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [9] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [10] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [11] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [12] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [13] Yicheng Feng, Yijiang Li, Wanpeng Zhang, Sipeng Zheng, Hao Luo, Zihao Yue, and Zongqing Lu. Videoorion: Tokenizing object dynamics in videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20401–20412, October 2025.
- [14] Hao Luo, Zihao Yue, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Deheng Ye, and Zongqing Lu. OpenMMEgo: Enhancing egocentric understanding for LMMs with open weights and data. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [15] Wanpeng Zhang, Zilong Xie, Yicheng Feng, Yijiang Li, Xingrun Xing, Sipeng Zheng, and Zongqing Lu. From pixels to tokens: Byte-pair encoding on quantized visual modalities. In The Thirteenth International Conference on Learning Representations, 2025.

- [16] Wanpeng Zhang, Yicheng Feng, Hao Luo, Yijiang Li, Zihao Yue, Sipeng Zheng, and Zongqing Lu. Unified multimodal understanding via byte-pair visual encoding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12976–12986, October 2025.
- [17] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.
- [18] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.
- [19] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [20] Yadong Lu, Chunyuan Li, Haotian Liu, Jianwei Yang, Jianfeng Gao, and Yelong Shen. An empirical study of scaling instruct-tuned large multimodal models. arXiv preprint arXiv:2309.09958, 2023.
- [21] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023.
- [22] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, pages 8469–8488. PMLR, 2023.
- [23] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy, 2024.
- [24] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.
- [25] Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025.
- [26] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model, 2025.
- [27] Huang Huang, Fangchen Liu, Letian Fu, Tingfan Wu, Mustafa Mukadam, Jitendra Malik, Ken Goldberg, and Pieter Abbeel. Otter: A vision-language-action model with text-aware visual feature extraction. arXiv preprint arXiv:2503.03734, 2025.
- [28] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models, 2025.
- [29] Johan Bjorck, Fernando Castañeda, Nikita Cherniaiev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.
- [30] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, and Tsung-Yi Lin. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models, 2025.
- [31] Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: vision-language-action pretraining from large-scale human videos. arXiv preprint arXiv:2507.15597, 2025.

- [32] Fanqi Lin, Ruiqian Nai, Yingdong Hu, Jiacheng You, Junming Zhao, and Yang Gao. Onetwovla: A unified vision-language-action model with adaptive reasoning. arXiv preprint arXiv:2505.11917, 2025.
- [33] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in neural information processing systems, volume 31, 2018.
- [34] Will Grathwohl, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form continuous dynamics for scalable reversible generative models. arXiv preprint arXiv:1810.01367, 2018.
- [35] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64, 2021.
- [36] Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Conditional flow matching: Simulation-free training of continuous normalizing flows. arXiv preprint arXiv:2302.00482, 2023.
- [37] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [38] Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.
- [39] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky TQ Chen. Multisample flow matching: Straightening flows with minibatch couplings. arXiv preprint arXiv:2304.14772, 2023.
- [40] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482, 2024.
- [41] Cédric Villani. Optimal transport: old and new, volume 338. Springer, 2009.
- [42] Gabriel Peyré and Marco Cuturi. Computational optimal transport. Foundations and Trends in Machine Learning, 11(5-6):355–607, 2019.
- [43] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.
- [44] Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and radon wasserstein barycenters of measures. In Journal of Mathematical Imaging and Vision, volume 51, pages 22–45, 2015.
- [45] Soheil Kolouri, Kimia Nadjahi, Umut Simsekli, Roland Badeau, and Gustavo Rohde. Generalized sliced wasserstein distances. Advances in neural information processing systems, 32, 2019.
- [46] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.
- [47] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 23–30. IEEE, 2017.
- [48] Eric Tzeng, Coline Devin, Judy Hoffman, Chelsea Finn, Pieter Abbeel, Sergey Levine, Kate Saenko, and Trevor Darrell. Adapting deep visuomotor representations with weak pairwise constraints. In Algorithmic Foundations of Robotics XII: Proceedings of the Twelfth Workshop on the Algorithmic Foundations of Robotics, pages 688–703. Springer, 2020.
- [49] Wanpeng Zhang, Yilin Li, Boyu Yang, and Zongqing Lu. Tackling non-stationarity in reinforcement learning via causal-origin representation. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 59264–59288. PMLR, 2024.
- [50] David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville. Out-of-distribution generalization via risk extrapolation (rex). In International conference on machine learning, pages 5815–5826. PMLR, 2021.
- [51] Youguang Xing, Xu Luo, Junlin Xie, Lianli Gao, Hengtao Shen, and Jingkuan Song. Shortcut learning in generalist robot policies: The role of dataset diversity and fragmentation. arXiv preprint arXiv:2508.06426, 2025.

- [52] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.
- [53] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.
- [54] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. arXiv preprint arXiv:2303.04137, 2023.
- [55] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.
- [56] Fan Feng, Biwei Huang, Kun Zhang, and Sara Magliacane. Factored adaptation for non-stationary reinforcement learning. Advances in Neural Information Processing Systems, 35:31957–31971, 2022.

## Appendix

#### A Full Proofs and Further Discussions

This appendix provides formal assumptions and complete proofs of the theoretical results stated in Section 5. We keep the notation from the main text and make explicit the regularity conditions under which the guarantees hold.

###### A.1 Standing Assumptions

Recall that for a single training example with observation features H and time t, the per-sample flow-matching loss is

ℓ(θ;H,t) = vθ(H,t) − v⋆(H,t) 2, (33) and the dataset objective is

L(θ) = E ℓ(θ;H,t) . (34) The discrepancy-guided objective used by DiG-Flow is

J(θ) = E g ℓ(θ;H,t) , g = ϕ D(µH,µZ) , (35)

where µH and µZ are the empirical measures defined in the main text, and gradients are stopped through g when differentiating with respect to θ.

We use ∥ · ∥ for the Euclidean norm on vectors and ∥ · ∥F for the Frobenius norm on matrices or tensor-valued sequences (such as H ∈ RT×d or Z ∈ RK×d). The inner product between two tensors of the same shape is written as ⟨·,·⟩.

We now state the regularity assumptions used in the proofs.

- Assumption 1 Parameter smoothness in θ There exists Lθ > 0 such that L(θ) is Lθ-smooth in θ, i.e., for all θ,θ′,

L(θ′) ≤ L(θ) + ∇θL(θ),θ′ − θ +

Lθ 2 ∥θ′ − θ∥2. (36)

- Assumption 2 Gate properties The weight mapping ϕ : R+ → [gmin,1] satisfies:

- • Monotone decreasing: if s1 ≤ s2 then ϕ(s1) ≥ ϕ(s2).
- • Lipschitz: there exists Lϕ such that |ϕ(s) − ϕ(s′)| ≤ Lϕ|s − s′| for all s,s′ ≥ 0.
- • Bounded away from zero: g = ϕ(D) ∈ [gmin,1] with gmin > 0. In particular, for all θ we have the bracketing inequality in the main text,

gmin L(θ) ≤ J(θ) ≤ L(θ), (37) which is Eq. (14).

- Assumption 3 Feature smoothness and residual boundedness

For fixed (θ,t), the loss as a function of the observation features H is LH-smooth in Frobenius norm: for all H,H′,

LH

2 ∥H′ − H∥2F. (38) Moreover, the residual operator R has bounded operator norm: there exists BR > 0 such that

ℓ(θ;H′,t) ≤ ℓ(θ;H,t) + ∇Hℓ(θ;H,t),H′ − H +

∥R(H)∥F ≤ BR ∥H∥F, ∀H. (39)

- Assumption 4 Bounded feature second moment There exists a constant CH > 0 such that

E ∥H∥2F ≤ CH2 (40) over the training distribution.

- Assumption 5 Refinement field regularity For each fixed gate g, the refinement error field

E(·;g) : RK×d → RK×d (41) used in the optional DiG-Refine procedure is LE-Lipschitz and µ-strongly monotone, i.e. for all Z1,Z2,

E(Z1;g) − E(Z2;g) F ≤ LE ∥Z1 − Z2∥F, (42) Z1 − Z2, E(Z1;g) − E(Z2;g) ≥ µ∥Z1 − Z2∥2F, µ > 0. (43)

Assumptions 1–4 are standard in analyses of smooth neural-network objectives: they hold when vθ is implemented by a network with Lipschitz activations and bounded weights, and ℓ is the squared error between vθ and a target field. Assumption 2 is satisfied by the clipped exponential mapping ϕ(D) = max{gmin,exp(−τD)} used in our implementation, which is monotone and Lipschitz on R+. Assumption 5 captures the usual regularity of gradient-like refinement fields appearing in fixed-point analyses. In addition to these regularity conditions, the residual-update result in Theorem 2 also uses a gated descent condition (Eq. (15)) on the learned residual field; although we state this condition directly in the theorem rather than as a separate standing assumption, it can be interpreted as a description of the average behaviour of the jointly trained residual operator, as discussed below.

Remark (On the gated descent condition). The gated descent condition in Eq. (15), used in Theorem 2, is not intended as an arbitrary extra constraint, but as a structural description of the learned residual field. When optimizing a composite objective of the form E[g ℓ(θ;H,t˜ )] (as in our implementation), the parameters of R receive gradients that encourage them to produce directions aligned with loss reduction. If R(H) were orthogonal to ∇Hℓ(θ;H,t) on average, it would not consistently decrease the loss and would be effectively regularized away. Thus, under successful training, it is natural to expect E[g ⟨∇Hℓ(θ;H,t), R(H)⟩] < 0, so Eq. (15) can be viewed as codifying the typical behaviour of a jointly trained residual operator rather than imposing a qualitatively new regularity assumption.

###### A.2 Smoothness of the gated objective

We first record a simple lemma that links the smoothness of L to that of J, clarifying why the gated objective remains well-behaved.

Lemma 1 Smoothness of the gated objective Under Assumptions 1 and 2, the gated objective

J(θ) = E g ℓ(θ;H,t) (44) is LJ-smooth in θ with LJ ≤ Lθ.

Proof. For each sample, treating g as constant w.r.t. θ (gradients are stopped through g), we have

∇θ g ℓ(θ;H,t) = g ∇θℓ(θ;H,t). (45) By Lθ-smoothness of ℓ(θ;H,t) in θ,

∥∇θℓ(θ′;H,t) − ∇θℓ(θ;H,t)∥ ≤ Lθ ∥θ′ − θ∥. (46)

Therefore

∇θ g ℓ(θ′;H,t) − ∇θ g ℓ(θ;H,t) = g ∥∇θℓ(θ′;H,t) − ∇θℓ(θ;H,t)∥ (47) ≤ Lθ ∥θ′ − θ∥, (48)

because g ≤ 1. Thus each term g ℓ(θ;H,t) is Lθ-smooth in θ, and J(θ), being their expectation, is LJ-smooth with LJ ≤ Lθ.

| |
|---|

- A.3 Proof of Theorem 1 (Gated descent) We now prove the gated-descent guarantee stated in the main text.

- Proof of Theorem 1. By Lemma 1, J is LJ-smooth in θ. For any LJ-smooth function F, the standard descent inequality holds: for a gradient descent step θ+ = θ − α∇F(θ) with 0 < α < 2/LJ,

F(θ+) ≤ F(θ) − α 1 −

αLJ

2 ∥∇F(θ)∥2. (49) Applying this to F = J gives

J(θ+) ≤ J(θ) − α 1 −

αLJ

2 ∥∇θJ(θ)∥2, (50) which matches Eq. (13) in the main text with

c1 = α 1 −

αLJ 2

> 0. (51)

The bracketing relation gminL(θ) ≤ J(θ) ≤ L(θ) (Eq. (14)) follows directly from Assumption 2: since g ∈ [gmin,1] almost surely,

gmin ℓ(θ;H,t) ≤ g ℓ(θ;H,t) ≤ ℓ(θ;H,t), (52) and taking expectations over the data distribution gives

gmin L(θ) ≤ J(θ) ≤ L(θ). (53) This proves Theorem 1.

| |
|---|

A.4 Proof of Theorem 2 (Residual update improvement)

We next analyze the effect of the gated residual update

H˜ = H + λg R(H) (54) on the expected loss, under the alignment condition in Eq. (15) of the main text.

- Proof of Theorem 2. Fix (θ,t) and consider ℓ(θ;H,t) as a function of H. By Assumption 3, for any perturbation ∆H we have

LH

2 ∥∆H∥2F. (55) We apply this with the gated residual update ∆H = λg R(H), so that H˜ = H + ∆H. Substituting gives

ℓ(θ;H + ∆H,t) ≤ ℓ(θ;H,t) + ∇Hℓ(θ;H,t),∆H +

LH

ℓ(θ;H,t˜ ) ≤ ℓ(θ;H,t) + λg ∇Hℓ(θ;H,t),R(H) +

2 ∥λg R(H)∥2F ≤ ℓ(θ;H,t) + λg ∇Hℓ(θ;H,t),R(H) +

LH 2

λ2g2BR2 ∥H∥2F, where we used ∥R(H)∥F ≤ BR∥H∥F.

Taking expectations over the training distribution and using the gated descent condition (Eq. (15)),

E g ⟨∇Hℓ(θ;H,t),R(H)⟩ ≤ −α0, (56) we obtain

LHBR2 2

E ℓ(θ;H,t˜ ) ≤ E ℓ(θ;H,t) + λE g ⟨∇Hℓ(θ;H,t),R(H)⟩ +

λ2E g2∥H∥2F

LHBR2 2

λ2E g2∥H∥2F . Using g ≤ 1 and Assumption 4, we may further bound

≤ E ℓ(θ;H,t) − α0λ +

E g2∥H∥2F ≤ E ∥H∥2F ≤ CH2 , (57) so

LHBR2 CH2 2

E ℓ(θ;H,t˜ ) ≤ E ℓ(θ;H,t) − α0λ +

λ2. (58) We would like the linear improvement −α0λ to dominate the quadratic term. A sufficient condition is

LHBR2 CH2 2

2α0 LHBR2 CH2

λ2 ⇐⇒ 0 < λ ≤ λ˜max :=

. (59)

α0λ ≥

For any 0 < λ ≤ λ˜max we then have

α0 2

E ℓ(θ;H,t˜ ) ≤ E ℓ(θ;H,t) −

λ, (60)

which is of the form stated in Theorem 2 with β = α0/2. This shows that there exists a positive threshold (for example λ˜max) such that the gated residual update strictly decreases the expected loss for all 0 < λ ≤ λ˜max.

| |
|---|

Remark (On the constant λmax). The proof above makes explicit one convenient sufficient choice

2α0 LHBR2 CH2

λ˜max =

, (61)

which depends on the data-dependent second-moment constant CH. In the main text, Eq. (16) only requires the existence of a nontrivial interval (0,λmax] for which the residual update improves the loss; the specific closed-form expression reported there can be viewed as an equivalent parameterization after absorbing bounded data constants (such as CH2 ) into α0 and using the fact that g ∈ [gmin,1]. Importantly, none of our qualitative conclusions depend on the exact numerical value of λmax: what matters is that there exists a strictly positive range of λ where the discrepancy-guided residual update provably reduces the objective.

###### A.5 Proof of Theorem 3 (Fixed-gate refinement convergence) We now prove the contraction result for the fixed-gate refinement scheme.

- Proof of Theorem 3. Consider the refinement map T(Z) = Z − α E(Z;g), (62)

where the gate g is fixed and E(·;g) satisfies Assumption 5. For any Z1,Z2 we have

T(Z1) − T(Z2) 2F = (Z1 − Z2) − α E(Z1;g) − E(Z2;g) 2F

= ∥Z1 − Z2∥2F − 2α Z1 − Z2,E(Z1;g) − E(Z2;g) + α2 E(Z1;g) − E(Z2;g) 2F ≤ ∥Z1 − Z2∥2F − 2αµ∥Z1 − Z2∥2F + α2L2E ∥Z1 − Z2∥2F

= 1 − 2αµ + α2L2E ∥Z1 − Z2∥2F,

where we used strong monotonicity and Lipschitz continuity of E(·;g). Define

ρ2 := 1 − 2αµ + α2L2E. (63) If 0 < α < 2µ/L2E, then ρ2 < 1 and hence ρ ∈ (0,1), and we obtain

T(Z1) − T(Z2) F ≤ ρ∥Z1 − Z2∥F. (64) Thus T is a contraction mapping with rate ρ. By Banach’s fixed-point theorem, T admits a unique fixed point Z⋆ and, for any initialization Z(0),

Z(k) − Z⋆ F = Tk(Z(0)) − Z⋆ F ≤ ρk Z(0) − Z⋆ F. (65) This is exactly the convergence guarantee stated in Theorem 3.

| |
|---|

