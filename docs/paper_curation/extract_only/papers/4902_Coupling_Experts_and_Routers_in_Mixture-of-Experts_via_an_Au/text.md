# arXiv:2512.23447v2[cs.CL]24Feb2026

[Figure 1]

## Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss

Ang Lv1,2, Jin Ma1, Yiyuan Ma1, Siyuan Qiao1 1ByteDance Seed 2Renmin University of China, GSAI

### Abstract

Mixture-of-Experts (MoE) models lack explicit constraints to ensure the router’s decisions align well with the experts’ capabilities, which ultimately limits model performance. To address this, we propose expert-router coupling (ERC) loss, a lightweight auxiliary loss that tightly couples the router’s decisions with expert capabilities. Our approach treats each expert’s router embedding as a proxy token for the tokens assigned to that expert, and feeds perturbed router embeddings through the experts to obtain intermediate activations. The ERC loss enforces two constraints on these activations: (1) Each expert must exhibit higher activation for its own proxy token than for the proxy tokens of any other expert. (2) Each proxy token must elicit stronger activation from its corresponding expert than from any other expert. These constraints jointly ensure that each router embedding faithfully represents its corresponding expert’s capability, while each expert specializes in processing the tokens actually routed to it. The ERC loss is computationally efficient, operating only on n2 activations, where n is the number of experts. This represents a fixed cost independent of batch size, unlike prior coupling methods that scale with the number of tokens (often millions per batch). Through pre-training MoE-LLMs ranging from 3B to 15B parameters and extensive analysis on trillions of tokens, we demonstrate the effectiveness of the ERC loss. Moreover, the ERC loss offers flexible control and quantitative tracking of expert specialization levels during training, providing valuable insights into MoEs.

Correspondence: Ang Lv at anglv@ruc.edu.cn, Yiyuan Ma at mayiyuan.unicorn@bytedance.com

### 1 Introduction

Mixture-of-Experts (MoE, 8, 19, 36, 50) is a core architecture in modern large language models (LLMs). In MoE models, the feed-forward layer is split into multiple small, specialized “experts.” A linear classifier, known as the “router,” selects which experts process each input token. By activating a few experts per token, MoE balances efficiency with scaled parameter counts, enabling the training of trillion-parameter models.

Ideally, a router should possess an accurate representation of each expert’s capabilities to enable effective token routing. However, traditional MoEs offer no explicit constraints to guarantee this. Without direct access to expert parameters (and therefore their true capabilities), routers resort to trial-and-error learning of routing strategies, often resulting in misrouted tokens whose gradients interfere with expert specialization. While some methods [25, 30] incorporated all experts’ activations for routing guidance, they incur substantial computational and memory costs due to denser activation. A lightweight and effective solution to better couple routing decisions with true expert capabilities remains an open challenge.

M[𝑖, 𝑛]

M[𝑖, 𝑗]

###### M 𝑖, 𝑖

M[𝑖, 𝑖]

R[𝑗]

Router parameters are cluster centers.

M[𝑖, 0]

Norm

R[𝑖]

ℒ𝐄𝐑𝐂

Compute𝐿

ℒ𝐄𝐑𝐂

𝑾 𝑾

𝑾 𝑾

…

⋱

𝛼M[𝑖, 𝑖]

… …

⋱

… i-th row

⋱

M ∈ ℝ𝒏×𝒏

(1) Perturb R as proxy tokens.

R[𝑖]

i-th column

R[𝑖]

𝜹 𝜹

(2) Input all R 𝑖 to all experts. Get intermediate activation norms M.

(3) The ERC loss enforces

R[𝑖] is a proxy for tokens routed to expert 𝑖.

R[𝑗]

M 𝑖, 𝑗 < 𝛼 M 𝑖, 𝑖 , M 𝑗, 𝑖 < 𝛼 M 𝑖, 𝑖 .

∀𝑖, 𝑗 ≠ 𝑖

(𝑾 denotes expert 𝑖's input layer.)

Figure 1 Three steps for computing the expert-router coupling loss.

We propose expert-router coupling loss (ERC loss), a novel auxiliary loss for MoE models that tightly couples routers and experts with negligible overhead. The loss is based on interpreting the router parameter matrix R ∈ n×d as cluster centers, where each row R[i] serves as the center for the token set Xi routed to expert i. The ERC loss comprises three key steps:

- (1) Each R[i] is augmented with bounded random noise δi to obtain R˜[i], serving as a proxy for tokens in Xi. Here, δi is bounded by half the minimum distance between adjacent cluster centers, ensuring that the noise simulates input variations within Xi while preventing the crossing of cluster boundaries.
- (2) Inspired by prior works [9, 22, 25], the intermediate activation norm serves as an indicator of how well its capabilities align with the token. We measure the intermediate activation norms of all experts that take

- R˜[i] as input. This step produces a matrix M ∈ n×n, with M[i,j] being the activation norm from expert j given input R˜[i].

- (3) For all i ≠ j, the ERC loss imposes a penalty wherever the off-diagonal elements M[i,j] or M[j,i] exceed αM[i,i], where α is a scalar hyperparameter:

n

n

1 n2

(max(M[i,j] − αM[i,i],0) + max(M[j,i] − αM[i,i],0)).

LERC =

i=1

j̸=i

Minimizing it tightly couples experts and routers through two effects:

- • Expert specialization: The proxy token R˜[i] elicits the strongest activation from expert i versus all other experts. This indicates that expert i is optimized to best match the features of its assigned token cluster Xi.
- • Precise token routing: Expert i is most activated by its designated vector R˜[i] than to any other R˜[j] for j ≠ i. This demonstrates that R[i] aligns well with the capabilities of expert i, ensuring that the router assigns to this expert the tokens that need it most.

We conducted large-scale pre-training experiments on models from 3B to 15B parameters, using a total of several trillion tokens. The ERC loss not only significantly enhances model performance and narrows the performance gap with a competitive yet more computationally expensive MoE variant [25] but also retains the efficiency of vanilla MoEs.

Furthermore, building on the first effect, we establish that the ERC loss serves as a powerful tool for studying expert specialization. This property arises from two key features of the ERC loss: (1) the specialization level is explicitly controlled by α, and (2) the bound of noise δi provides a quantitative measure for this level. Through this lens, we reveal a trade-off between specialization and model performance. Our findings challenge some beliefs about expert specialization that were derived from small-scale experiments. These quantitative and qualitative analysis methods offer new pathways to advance the understanding of MoE models.

### 2 Background

Mixture-of-Experts Our description follows the prevailing SwiGLU structure used by advanced LLMs [7, 28, 32]. An MoE layer consists of n experts, where each expert i is parameterized by three matrices: Wgi ∈ d×D, Wpi ∈ d×D, and Woi ∈ D×d. The layer also includes a router with the weight matrix R ∈ n×d, which takes a token x ∈ d as input and outputs an expert weight1 vector:

w = softmax(xR⊤) ∈ n.

Typically, the top-K experts with the highest expert weights are selected to process the token. The processing of x by expert i is given by:

Ei(x) = SiLU(xWgi) ⊙ (xWpi) Woi, where ⊙ denotes element-wise multiplication. The final output of the entire MoE layer is the weighted sum of the outputs of the selected experts:

K

w[k]Ek(x), where k ∈ Top-K(w).

k

Expert-router coupling via denser activation Autonomy-of-Experts (AoE; 25) encodes the routing function into expert parameters. AoE factorizes Wg into two r-rank matrices Wdowni ∈ d×r and Wupi ∈ r×D. Each expert processes a token up to the point after the Wdowni projection. The expert weight vector is computed using the activation norm at this stage:

w = softmax {∥xWdowni ∥ for i = 1,...,n} .

The top-K experts exhibiting the highest activation norms are selected to continue their forward computation, and the others are terminated early. This norm-based selection is justified by the fact that the activation norm of MLPs represents how well their capabilities match their inputs [9, 22]. The computational overhead of AoE scales with the number of tokens during both training and inference. Moreover, this inefficiency worsens as the number of experts n increases or the selection count K decreases. Wu et al. [46] found that using only a small, representative subset of neurons per expert is sufficient for “autonomous expert selection,” reducing but not eliminating AoE’s token-dependent cost.

Used Param.

Compute w 𝑳𝟐 Norm & Softmax

𝑬𝟏 𝑬𝟐 𝑬𝟑

Unused Param.

w

𝑬𝟏 𝑬𝟐 𝑬𝟑

Router x

x

(a) Mixture-of-Experts (b) Autonomy-of-Experts

Figure 2 The overview of MoE and AoE models.

Pham et al. [30] use experts’ final output norms to supervise router logits. There is no inference overhead but the model is fully dense-activated during training, contradicting the core sparsity principle of MoE. Therefore, we include it only for background discussion, not as a baseline.

### 3 Method

After analyzing the strengths and limitations of prior work, we distill three design principles to ensure a lightweight, effective, and practically applicable enhancement for expert-router coupling in MoE-LLMs:

- (1) Routers must be retained in MoE architectures to preserve routing efficiency.
- (2) An auxiliary loss that enables interaction between experts and routers can strengthen their coupling.
- (3) The loss must have complexity independent of the number of input tokens and must not introduce activation density beyond that of a vanilla MoE. Below, we introduce expert-router coupling loss, which fulfills all these principles.

1In this paper, “weight” refers to the relative contribution of each expert’s output or the strength of the loss function. Please carefully distinguish between “weight” and “parameter.”

#### 3.1 Expert-router coupling loss

The expert-router coupling (ERC) loss is motivated by a clustering-based interpretation of MoE routing: The routing mechanism in traditional MoE models can be interpreted as a clustering process, where router parameters R ∈ n×d are viewed as n cluster centers. For any input token x ∈ d, the router computes an n-dimensional logit vector representing the weight assigned to each expert. Specifically, the weight for expert i is derived from the inner product between x and the cluster center R[i]. When x belongs to the cluster centered at R[i], this inner product is maximized2, making expert i the top choice.

A key advantage of this clustering view is that it enables probing an expert’s responsiveness to a set of tokens without feeding every token to all experts, unlike prior methods (See §2). Instead, we leverage each cluster center R[i] as a proxy for tokens routed to expert i (denoted as Xi), enabling us to derive intermediate activations and evaluate how well the expert aligns with a proxy token.

Our ERC loss is computed in three key steps:

- (1) For each cluster center R[i], we create a perturbed proxy token R˜[i] = R[i] ⊙ δi. δi ∈ d is bounded multiplicative random noise, which we elaborate in §3.2. This noise ensures the proxy generalizes to tokens in

Xi. Notably, the perturbed R˜ is used only for loss computation; routing still uses the clean R to compute router logits, as in standard MoEs.

- (2) Each proxy token is processed by the Wg parameter of all n experts, yielding a total of n2 intermediate activations. The L2 norm of each activation is computed to form a matrix M ∈ n×n, where M[i,j] corresponds to the norm from expert j given input R˜[i]:

M[i,j] = R ˜[i] · Wgj .

- (3) To enforce expert-router coupling, for all i and j ≠ i, the ERC loss imposes two constraints, where a scalar α ∈ [0,1] determines their strength:

- M[i,j] < αM[i,i], (1)
- M[j,i] < αM[i,i]. (2)

- Constraint 1 ensures the proxy token R˜[i] activates its corresponding expert i more than any other expert j. Since tokens similar to R[i] are routed to expert i, and given their similarity to R˜[i], they also elicit a stronger activation in expert i than in other experts. This strongest activation indicates that expert i is optimized to develop capabilities best suited to Xi [25].
- Constraint 2 requires that expert i responds more strongly to its own proxy token R˜[i] than by any other

- R˜[j]. This ensures each R[i] accurately represents expert i, guaranteeing that tokens most needing expert i are correctly routed to it.

As α decreases, the two constraints become stricter, thereby enforcing stronger expert-router coupling. Additionally, α enables flexible regulation of specialization: a smaller α increases the gap between M[i,i] and M[i,j], reflecting greater expert specialization as experts exhibit more differentiated responses to the same inputs. This feature makes the ERC loss a useful tool for investigating expert specialization and provides deeper insight into MoE behavior, as demonstrated in §4.2.

We translate these two constraints into expert-router coupling loss, formally defined as:

n

n

1 n2

(max(M[i,j] − αM[i,i],0) + max(M[j,i] − αM[i,i],0)). (3)

LERC =

i=1

j̸=i

The three steps for computing expert-router coupling loss are illustrated in Figure 1. For implementation details, we provide PyTorch-style pseudocode in Figure 8.

2This assumes all R[i]s have comparable norms. We confirm that the models used in our experiments adhere to this assumption.

- 3.2 Bounded random noise for generating proxy tokens

The perturbed proxy token R˜[i] = R[i] ⊙ δi makes expert i’s coupling generalize effectively from R[i] alone to Xi. To ensure the perturbed point R˜[i] remains within its original cluster, we require a bounded perturbation. We therefore model the noise δi as a multivariate uniform distribution, δi ∼ U(1 − ϵi,1 + ϵi)d. Let j = arg minj∗̸=i ∥R[i] − R[j∗]∥ be the nearest cluster center. For the noise level ϵ to be sufficient to avoid perturbing the cluster, it must satisfy:

ϵi ≤

∥R[i] − R[j]∥ 2∥R[i]∥

. (4)

The derivation of this bound is provided in Appendix B. We set ϵi to its maximum value, i.e., the right-hand side of this inequality. Notably, the value of ϵi is dynamically computed at each layer and every training step.

- 3.3 Efficiency analysis

Training efficiency In a standard MoE layer, T tokens are processed by K experts, resulting in a total computational cost of 6TKDd FLOPs. expert-router coupling loss introduces only 2n2Dd additional FLOPs, a cost that is negligible in practical pre-training setups where K is often in the millions. In contrast, AoE introduces an additional overhead of 2T(n − K)dr FLOPs (recall that r is AoE’s factorization rank; see §2). Given that typical MoE-LLMs operate at sparsity levels far below 25% (i.e., n > 4K), this overhead ratio exceeds r/D, making it prohibitive. A detailed breakdown of the FLOP calculations supporting the above theoretical analysis is provided in Appendix C.1.

The efficiency of our method is confirmed in practice. The ERC loss maintains low overhead during LLM pretraining under multiple parallelism strategies, adding only 0.2–0.8% overhead in our experiments. We provide a complete analysis of these real-world distributed conditions and measured throughputs in Appendix C.2.

Overhead-free inference Our method incurs no additional inference overhead as the auxiliary loss is not applied. However, AoE retains the same forward computation, carrying over the associated overhead.

- 4 Experiments

- 4.1 Experimental settings

We compare the ERC-loss-augmented MoE against both the vanilla MoE and AoE baselines. All models are trained from scratch with 3B parameters. This parameter size is chosen because it represents the largest scale at which we could successfully train the AoE model under our available resources. Our implementation is based on OLMoE [27]. The models comprise 12 layers with d = 1536 and D = 768. Each Transformer [42] layer has 16 attention heads and n = 64 experts, where K = 8 experts are selected per token. For the AoE model, we set r = 512 to ensure consistent total parameter count. The number of activated parameters is 500M. Each model is trained on 500B tokens from the open-source dataset dolmap-v1.5-sample [37], using a batch size of 3 million tokens. We use the AdamW optimizer [23] with (β1,β2) = (0.9,0.95), a weight decay of 0.1, and a learning rate of 4e-4 with a cosine schedule decaying to 4e-5. A load balancing loss [8] with a weight of 0.01 is applied consistently in all experiments.

For simplicity, the loss weight of the ERC loss is fixed at 1, and we use α = 1 by default if not specified. We evaluate LLMs on the following tasks: ARC-Challenge [4], CommonsenseQA [39], COPA [33], BoolQ [3], HellaSwag [48], OpenbookQA [26], SciQ [45], Social IQa [35], WinoGrande [34], and MMLU [14].

- 4.2 Performance, efficiency, and load balancing

Figure 3(a) reports the average accuracy across all tasks and task-specific results are presented in Figure 9. It shows that the ERC-loss-augmented MoE achieves stable performance gains, which significantly outperforms the vanilla MoE and narrows the gap between AoE and MoE.

(a) Average Downstream Task Accuracy (b) Load Balance Loss

[Figure 2]

[Figure 3]

MoE +ℒ

0.0110

AoE

56.5

MoE

56.0

0.0108

55.5

0.0106

55.0 54.5

0.0104

0.0102

54.0

100 200 300 400 500

300 400 500

Tokens (B)

Tokens (B)

- Figure 3 The 3B-parameter MoE model augmented with ERC loss achieves substantial and stable performance gains, while maintaining comparable load balancing to the baseline. Figure 9 shows task-specific details.

In terms of efficiency, MoE models with and without ERC loss have nearly identical throughput and memory costs. By contrast, AoE requires 1.6× more training hours and 1.3× higher memory usage, limiting further scaling due to impractical training times and out-of-memory issues.

Expert-router coupling loss is compatible with the load balancing loss. As shown in Figure 3(b), the difference in load balancing loss between MoE combined with LERC and the vanilla MoE is on the order of 10−5. This difference is negligible given that the overall load balancing loss magnitude remains around 10−2. By comparison, the loss difference between AoE and vanilla MoE is approximately 4 × 10−4. Although this difference is still small, it is notably larger than the difference exhibited by ours.

#### 4.3 Validating ERC Loss in 15B-Parameter MoEs

We scale models to 15 billion parameters by increasing n to 256 (keeping K=8) and doubling the model depth. This configuration results in a total of 15B parameters with approximately 700M activated. Other training hyper-parameters largely follow the setup in Section 4.1. As a large-scale, high-sparsity model, the AoE method failed to train due to being overly costly and is thus omitted from comparison. Table 1 shows that the benefits of the ERC loss persist across various public benchmarks more challenging than those used for 3B models, including MMLU [14], C-Eval [16], MMLU-Pro [44], AGI-Eval [49], BBH [38], MATH [15], GSM8K [5], and TriviaQA [17]. The consistent performance improvements demonstrate that our method effectively addresses the expert-router decoupling problem even at scale. Throughout this large-scale training, we observed no loss spikes or abnormal gradients.

Table 1 Scaling to 15B parameters: ERC loss improves performance on more challenging benchmarks.

MMLU C-Eval MMLU-Pro AGI-Eval BBH MATH GSM8K TriviaQA

MoE 63.2 67.5 31.0 42.0 44.3 25.7 45.2 47.2 MoE + LERC 64.6 69.0 31.9 44.2 45.6 26.1 45.8 49.1

#### 4.4 The ERC loss is an effective tool for exploring expert specialization

With the ERC loss, experts are more specialized, as they exhibit greater discrimination between tokens they process and those they do not, compared to vanilla MoE (without the ERC loss). An intuitive demonstration of this specialization comes from visualizing expert parameters. Following [47], we use t-SNE [41] to project each row of Wgi (where i mod 8 = 0) from layer 6 (the middle depth) onto a 2D point. As shown in Figure 4, experts in vanilla MoE lack specialization, as their parameter features do not form meaningful clusters. By contrast, MoE enhanced with the ERC loss exhibits more distinct clusters, indicating specialized capabilities.

[Figure 4]

[Figure 5]

[Figure 6]

Expert ID

0

8 16 24 32 40 48 56

(b) MoE + ℒ

(a) MoE

- Figure 4 t-SNE projections of Wg in MoE experts trained without and with the ERC loss. Our ERC loss provides greater expert specialization.

Beyond merely promoting specialization, the ERC loss can also serve as a powerful tool for exploring it. We show this capability through two features below.

- Feature 1: α enables a controllable investigation into optimal specialization In the ERC loss, α governs the coupling strength between experts and the router. When α = 0, the ERC loss encourages R[i] to be orthogonal to the parameters of other experts, thereby maximizing specialization. Conversely, when α → 1, the loss permits smaller differences in how all experts’ responsiveness to R[i], thus reducing specialization. Notably, α = 1 only weakens the ERC loss’s constraints to their maximum extent; it still retains a degree of specialization stronger than the spontaneously emerged specialization in a vanilla MoE model.
- Feature 2: ϵ provides a quantitative measure for specialization The noise level ϵ exhibits a strong correlation with α, and it can reflect changes in expert specialization throughout the training process. This correlation exists because as α increases, experts are allowed to be more homogeneous. This growing homogeneity among experts, in turn, reduces the separation between the cluster centers in the router as they are tightly coupled. A smaller separation between cluster centers ultimately derives a smaller ϵ. Thus, ϵ is a quantitative metric tracking expert specialization.

Experiments The following experiments support these two features. In Figure 5(a), we plot ϵ at each training step across a parameter search over α ∈ {0.4,0.6,0.8,1.0}. Consistent with our analysis, increasing α, which reduces expert specialization, indeed leads to a corresponding decrease in ϵ. Note that measuring router cluster distance is uninformative in vanilla MoE training without the ERC loss, because the router and experts are uncoupled and cluster distances do not reflect expert capability dynamics. We further compared downstream task performance across different values of α. Figure 5(b) shows that all tested α values outperform the vanilla MoE model. This not only confirms the robust effectiveness of the ERC loss but also demonstrates that the specialization spontaneously formed by vanilla MoE models is inadequate.

The optimal specialization degree Figure 5(b) shows that pursuing extreme specialization is not advisable, as model performance degrades with overly strict α. This highlights a trade-off between promoting expert specialization and maintaining effective collaboration, which is under-discussed in previous work.

The optimal specialization degree is influenced by several factors. The core consideration is whether, among all K n possible expert combinations, an effective K-expert set can be assembled for any given input. In general, smaller values of n favor more generalist experts, while larger n can support a higher degree of specialization. However, we currently lack quantitative metrics to characterize “large” or “small” n and K across different models; as a result, determining the optimal trade-off remains largely empirical. For example, in our experiments with a fixed K = 8: When n = 64, the optimal α = 1, suggesting n = 64 is not “large” for our 3B-parameter models. In contrast, with n = 256, we searched for an optimal α = 0.5, indicating n = 256

(b)

(a)

[Figure 7]

[Figure 8]

0.625

56.5

0.600

56.0

𝜖Thevalueof

0.575

Accuracy(%)

0.550

55.5

0.525

55.0

0.500

0.475 54.5

0.450

54.0

1 100 200 300 400 500

Tokens (B) 300 400 500

𝜶 0.4 0.6 0.8 1.0 Vanilla MoE

- Figure 5 (a) Since routers are deeply coupled with experts, the distance between neighboring cluster centers (i.e., the maximum noise level ϵ) quantitatively reflects changes in expert specialization during training, which is controlled by α. (b) Downstream performance across different values of α.

is “large” for our 15B-parameter models. This trade-off is also shaped by other architectural choices, such as the use of shared experts3. A deeper investigation into these interacting factors, reliable quantitative metrics for specialization, and an automated evaluation of the optimal specialization degree for a given model are left as important problems for future works. For practitioners implementing the ERC loss, we recommend starting with α = 1, which eliminates expert decoupling and should provide some gains. Further improvement may be achieved by experimenting with lower α values, depending on the specific configuration of your model.

Several studies [11, 13, 21] have promoted specialization via expert output orthogonality. We argue, however, that orthogonalizing expert outputs does not equate to achieving extreme specialization, as the magnitude (norm) of an expert’s response to a token remains unconstrained. Moreover, finding a set of orthogonalized high-dimensional vectors is not difficult, making it unclear whether such orthogonality sufficiently leads to effectively discriminative representations. Consequently, one should not interpret these fine-tuning experiments as supporting a broad claim that “more specialization is always better.” On a separate note, orthogonality among router embeddings [1] is only weakly correlated with specialization, since the router and experts are typically decoupled. As demonstrated in ablation studies, enforcing router orthogonality might not be a critical factor for pre-training MoE models.

#### 4.5 Ablation studies

Choice of activations for computing M We considered five candidates for calculating M: using the norms of (a) RW˜ g, (b) RW˜ p, (c) SiLU(RW˜ g), (d) the post-SwiGLU activations (i.e., SiLU(RW˜ g) ⊙ RW˜ p), and (e) experts’ final outputs (i.e., (SiLU(RW˜ g) ⊙ RW˜ p)Wo). As shown in Figure 6(a), RW˜ g is the most effective among all alternatives. While using the final output achieves comparable performance, it incurs a higher cost. We therefore adopt RW˜ g as our default choice.

Random noise δ enables the generalization of coupling The random noise δ allows R˜[i] to better capture the samples within Xi. To validate its importance, we conducted an ablation study where we trained an MoE with the ERC loss but removed δ. Specifically, we computed M directly using the original R instead of the noise-augmented R˜. As shown in Figure 6(b), removing δ greatly degrades performance. This is because the coupling between routers and experts becomes overfitted to R, failing to generalize to the real inputs that R[i]s represent.

##### Comparison with contrastive regularization solely on routers The router orthogonalization loss [1] requires

3A shared expert satisfies general input requirements, allowing the remaining experts to be more specialized even with the same n.

(a) Using various activations to compute M (b) The impact of 𝜹

(c) Comparison with orthogonalized routers

[Figure 9]

[Figure 10]

[Figure 11]

MoE + ℒ

MoE + ℒ

Vanilla MoE

MoE + ℒ

56.5

MoE + ℒ - 𝜹

56.5

56.5

𝑹𝑾 𝑹𝑾

Vanilla MoE

Vanilla MoE

SiLU( 𝑹𝑾 ) Post-SwiGLU Output

56.0

56.0

56.0

55.5

55.5

Accuracy(%)

55.5

55.0

55.0

55.0

54.5

54.5

54.5

54.0

54.0

54.0

300 400 500

300 400 500

300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

Figure 6 Results of ablation studies. For detailed task-specific results, please refer to Figure 9.

Rˆ (the row-wise normalization of R) to satisfy:

RˆRˆ⊤ = I.

As shown in Figure 6(c), the orthogonalization loss yields only limited gains. We attribute this to our finding that the router embeddings in our baseline MoE model are already nearly orthogonal, with an average absolute cosine similarity of 0.15. This value corresponds to angles between router embeddings mostly ranging from arccos(0.15) = 81° to arccos(-0.15) = 99°. Notably, we do not imply that all MoEs always have nearly orthogonal router embeddings, as this may depend on the data or specific architecture; we report this only as a characteristic of our models, which explains the limited gains from the orthogonalization loss.

This result further demonstrates that weak coupling between routers and experts is a more critical issue than imperfect orthogonality in router embeddings. The significant gains from ERC, even when applied to a baseline with already near-orthogonal routers, provide clear evidence.

Furthermore, it is important to note that even if both routers and experts are orthogonalized, there is no guarantee that each R[i] will be aligned with Wgi. Therefore, the ERC loss cannot be reduced to contrastive techniques applied individually to routers or experts, such as orthogonalization loss.

Additional analyses Appendix A provides analyses addressing several frequently asked questions, including the effect of α > 1, and verifying that the model decreases the ERC loss by learning meaningful coupling rather than by manipulating parameter norms.

### 5 Related works

Auxiliary loss for MoEs Auxiliary losses are crucial for training large-scale MoE models. Most existing work in this area focuses primarily on enhancing training stability. For instance, many studies have proposed auxiliary losses to address load balancing challenges [8, 31, 43]; Zoph et al. [50] introduced the z-loss, which penalizes excessively large logits in the gating network to enable stable training. MoE concepts have also inspired mixtures of attention heads or entire layers [10, 20], where auxiliary losses play a critical role in effective optimization. Our ERC loss is the first tailored to strengthen the expert-router coupling. Other related auxiliary losses enhancing expert specialization or orthogonality are discussed below.

Expert specialization Dai et al. [6] introduced a shared expert to handle general capabilities, encouraging the others to be more specialized. Guo et al. [11] proposes an auxiliary loss to minimize the pairwise projections of the selected top-K experts’ outputs for each token, reducing expert overlap but incurring high cost due to K2 cosine similarity calculations per token. Other methods scale the number of tiny experts to millions, making each expert more atomic and thus more specialized [12, 29, 47], but are memory-bounded. Beyond efficiency, these methods face three major limitations: (1) no quantitative control over specialization degree;

(2) no exploration of the specialized-generalized ability trade-off; and (3) failure to strengthen expert-router coupling. Our method addresses all three, both efficiently and effectively.

Some works [11, 13, 21] maximize specialization by training orthogonal experts, but their evaluations are based on fine-tuning (or reinforcement learning) experiments. We contend that orthogonalizing expert outputs is not equivalent to achieving extreme specialization, and further, that the optimal degree of specialization is a complex problem affected by various factors and requires further exploration.

Contrastive learning Constraints 1 and 2 bear similarity to contrastive learning [2, 18, 40]. Some MoE research [11, 24] applied contrastive learning to expert outputs, encouraging specialization. However, naively applying contrastive learning to either routers or experts leaves the weak expert-router coupling unaddressed.

### 6 Conclusions

The weak coupling between router decisions and expert capabilities limits MoE models in multiple important aspects. We propose expert-router coupling loss that tightly couples router parameters with their corresponding experts. The proposed ERC loss improves MoE-based LLMs on downstream tasks while incurring negligible training overhead. In addition, it exhibits several desirable properties that not only provide deeper insight into the behavior of MoE models but also offer a promising tool for future research on expert specialization.

### Acknowledgments

We thank Songhao Wu and Ziteng Wang for their insightful discussions. We are grateful to Ruobing Xie, Yining Qian, and Kaiyi Zhang for proofreading and writing suggestions. Additionally, we sincerely acknowledge the anonymous ICLR reviewers for their constructive comments and questions, which have greatly improved this work.

### References

- [1] Baidu-ERNIE-Team. Ernie 4.5 technical report, 2025.
- [2] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations, 2020. URL https://arxiv.org/abs/2002.05709.
- [3] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924– 2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1300. URL https://aclanthology.org/N19-1300/.

- [4] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https: //arxiv.org/abs/1803.05457.
- [5] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.
- [6] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2401.06066.
- [7] DeepSeek-AI. Deepseek-v3 technical report, 2025. URL https://arxiv.org/abs/2412.19437.
- [8] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022. URL http: //jmlr.org/papers/v23/21-0998.html.

- [9] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/ 2021.emnlp-main.446. URL https://aclanthology.org/2021.emnlp-main.446.

- [10] Zhuocheng Gong, Ang Lv, Jian Guan, Wei Wu, Huishuai Zhang, Minlie Huang, Dongyan Zhao, and Rui Yan. Mixture-of-modules: Reinventing transformers as dynamic assemblies of modules. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 20924–20938, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.1164. URL https://aclanthology.org/2024.emnlp-main. 1164/.

- [11] Hongcan Guo, Haolang Lu, Guoshun Nan, Bolun Chu, Jialin Zhuang, Yuan Yang, Wenhao Che, Sicong Leng, Qimei Cui, and Xudong Jiang. Advancing expert specialization for better moe, 2025. URL https://arxiv.org/ abs/2505.22323.
- [12] Xu Owen He. Mixture of a million experts, 2024. URL https://arxiv.org/abs/2407.04153.
- [13] Ahmed Hendawy, Jan Peters, and Carlo D’Eramo. Multi-task reinforcement learning with mixture of orthogonal experts. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=aZH1dM3GOX.

- [14] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

- [15] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

- [16] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems, 2023.

- [17] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL https://aclanthology.org/P17-1147/.

- [18] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 18661–

18673. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/ d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf.

- [19] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. {GS}hard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=qrwe7XHTmYb.

- [20] Hongzhan Lin, Ang Lv, Yuhan Chen, Chen Zhu, Yang Song, Hengshu Zhu, and Rui Yan. Mixture of in-context experts enhance llms'long context awareness. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 79573–79596. Curran Associates, Inc., 2024. doi: 10.52202/079017-2526. URL https://proceedings.neurips. cc/paper_files/paper/2024/file/91315fbb83ce353ae5538cba395f70d1-Paper-Conference.pdf.

- [21] Boan Liu, Liang Ding, Li Shen, Keqin Peng, Yu Cao, Dazhao Cheng, and Dacheng Tao. Diversifying the mixture-of-experts representation for language models with orthogonal optimizer, 2024. URL https://arxiv. org/abs/2310.09762.
- [22] Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, and Beidi Chen. Deja vu: contextual sparsity for efficient llms at inference time. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

- [23] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

- [24] Tongxu Luo, Jiahe Lei, Fangyu Lei, Weihao Liu, Shizhu He, Jun Zhao, and Kang Liu. Moelora: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models, 2024. URL https://arxiv.org/abs/2402.12851.
- [25] Ang Lv, Ruobing Xie, Yining Qian, Songhao Wu, Xingwu Sun, Zhanhui Kang, Di Wang, and Rui Yan. Autonomy-ofexperts models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview. net/forum?id=8BIDrYWCeg.

- [26] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018. URL https://arxiv.org/abs/1809.02789.
- [27] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2025. URL https://arxiv.org/abs/2409.02060.
- [28] OpenAI. Gpt-oss series, 8 2025. URL https://openai.com/index/introducing-gpt-oss/.
- [29] Jungwoo Park, Ahn Young Jin, Kee-Eung Kim, and Jaewoo Kang. Monet: Mixture of monosemantic experts for transformers. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=1Ogw1SHY3p.

- [30] Quang Pham, Giang Do, Huy Nguyen, TrungTin Nguyen, Chenghao Liu, Mina Sartipi, Binh T. Nguyen, Savitha Ramasamy, Xiaoli Li, Steven Hoi, and Nhat Ho. Competesmoe – effective training of sparse mixture of experts via competition, 2024.
- [31] Zihan Qiu, Zeyu Huang, Bo Zheng, Kaiyue Wen, Zekun Wang, Rui Men, Ivan Titov, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5005–5018, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.249. URL https://aclanthology.org/2025.acl-long.249/.

- [32] Qwen. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

- [33] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S. Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI Spring Symposium Series, 2011.

- [34] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

- [35] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463–4473, Hong Kong, China, November

2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1454. URL https://aclanthology.org/ D19-1454.

- [36] Noam Shazeer, *Azalia Mirhoseini, *Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=B1ckMDqlg.

- [37] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. arXiv preprint, 2024.

- [38] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging BIG-bench tasks and whether chainof-thought can solve them. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.824. URL https://aclanthology. org/2023.findings-acl.824/.

- [39] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1421. URL https://aclanthology.org/N19-1421/.

- [40] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding,

2019. URL https://arxiv.org/abs/1807.03748.

- [41] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008. URL http://jmlr.org/papers/v9/vandermaaten08a.html.

- [42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

- [43] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts, 2024. URL https://arxiv.org/abs/2408.15664.
- [44] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.

- [45] Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Leon Derczynski, Wei Xu, Alan Ritter, and Tim Baldwin, editors, Proceedings of the 3rd Workshop on Noisy User-generated Text, pages 94–106, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4413. URL https://aclanthology.org/W17-4413/.

- [46] Songhao Wu, Ang Lv, Ruobing Xie, Xingwu Sun, Di Wang, Rui Yan, and Yankai Lin. Union-of-experts: Experts in mixture-of-experts are secretly routers, 2025. URL https://openreview.net/forum?id=Ksgiup7ZNZ.
- [47] Xingyi Yang, Constantin Venhoff, Ashkan Khakzar, Christian Schroeder de Witt, Puneet K. Dokania, Adel Bibi, and Philip Torr. Mixture of experts made intrinsically interpretable. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=6QERrXMLP2.

- [48] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Lluís Màrquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1472. URL https://aclanthology.org/P19-1472/.

- [49] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. AGIEval: A human-centric benchmark for evaluating foundation models. In Kevin Duh, Helena Gomez, and Steven Bethard, editors, Findings of the Association for Computational Linguistics: NAACL 2024, pages 2299–2314, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-naacl.149. URL https://aclanthology.org/2024.findings-naacl.149/.

- [50] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models, 2022. URL https://arxiv.org/abs/2202.08906.

## Appendix

### A Frequently Asked Questions

#### A.1 What happens if α > 1?

Some readers might be interested in the value of α at which the ERC loss degenerates to no effective constraints, and the trained model consequently degenerates to a vanilla MoE. For our baseline MoE, we seek the minimum α that zeros the ERC loss computed from the M matrices of the last checkpoint. Table 2 shows that achieving zero ERC loss across all layers requires α = 5 in our pre-trained vanilla MoE baseline. This provides direct evidence that the router-expert coupling in the vanilla MoE is very weak.

- Table 2 Post-hoc ERC loss evaluation of the vanilla MoE across α values. For clarity, loss values are computed using the original R rather than R˜, making the results deterministic.

[Figure 12]

- +ℒ (α = 1)

Vanilla MoE

- +ℒ (α = 2)
- +ℒ (α = 3)

56.5

Value of α 1 2 3 4 5

Layer

56.0

Accuracy(%)

- 0 0.87 0.69 0.26 0.00 0.00

- 1 0.42 0.28 0.10 0.00 0.00

- 2 0.45 0.19 0.00 0.00 0.00

- 3 0.25 0.15 0.00 0.00 0.00

- 4 0.28 0.08 0.00 0.00 0.00

- 5 0.24 0.22 0.00 0.00 0.00

- 6 0.22 0.15 0.00 0.00 0.00

- 7 0.21 0.13 0.00 0.00 0.00

- 8 0.15 0.05 0.00 0.00 0.00

- 9 0.16 0.00 0.00 0.00 0.00

- 10 0.21 0.09 0.00 0.00 0.00

- 11 0.50 0.44 0.20 0.20 0.00

55.5

55.0

54.5

54.0

300 400 500

Tokens (B)

Figure 7 α > 1 leads to degeneration into vanilla MoEs

We further pre-trained 3B MoE models with the ERC loss at α values of 2 and 3. It is important to note that this experiment is to only demonstrate the effects of loosening the ERC constraints. We do not recommend using α > 1 in practice, as it contradicts our core motivation: the router and experts will shift from a state of no mismatch toward looser coupling constraints, ultimately causing the model to degenerate into a vanilla MoE. As shown in Figure 7, the model with α = 2 yields only limited improvement, while the model with α = 3 shows almost no improvement over the vanilla MoE.

#### A.2 Do models reduce ERC loss through manipulating parameter norms?

Some readers might assume that simply increasing or decreasing the norms of certain router embeddings or experts will increase the diagonal entries of M, thereby reducing the ERC loss. Below, we (1) explain that any attempt to reduce one term of the ERC loss by manipulating norms will simultaneously increase other terms, and (2) present detailed parameter norms as evidence.

Note that M[i,j] can be written as ∥R[i]∥∥Wgj∥c˜osi,j, where c˜osi,j denotes the averaged cosine similarity between R[i] and Wgj.4

j g [:,k]∥

4Formally, c˜osi,j = k≤D ∥W

cos θi,j,k, where θi,j,k is the angle between R[i] and the k-th column of Wgj. For clarity, we use this compact form.

∥Wgj∥

Increasing ∥R[i]∥ decreases the following loss term (as the second term increases): ∥R[j]∥∥Wgi∥c˜osj,i − ∥R[i]∥∥Wgi∥c˜osi,i. However, it simultaneously increases the following term (as the first term grows): ∥R[i]∥∥Wgj∥c˜osi,j − ∥R[j]∥∥Wgj∥c˜osj,j.

Similarly, any attempt to manipulate the norms of Wg to reduce one part of the loss necessarily increases others, assuming that ∥Wgj[:,k]∥/∥Wgj∥ remains fixed. This property ensures that the overall ERC loss is minimized only when the router embedding norms are kept comparable and a meaningful coupling is established between routers and their experts.

As shown in the first four columns of Table 3, the average parameter norms for models trained with and without the ERC loss are comparable. Meanwhile, the lower standard deviation under the ERC loss reflects more consistent norms across both router embeddings and experts. In the last two columns of the table, we present the ERC loss for each model. The ERC loss is significantly higher in the baseline model despite its similar average parameter norms.

- Table 3 The first four columns show parameter norms for models trained with and without ERC loss, while the last two show the corresponding layer-wise ERC loss. These results show that MoE + LERC learns a meaningful coupling, rather than trivially minimizing the loss through norm manipulation. All values are evaluated on the last checkpoint.

|Layer<br><br>|∥R[i]∥ ∥Wgi∥ LERC Values| | |
|---|---|---|---|
| |Baseline +LERC<br><br>|Baseline +LERC|Baseline +LERC|
|0<br><br>1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br><br>|1.85±0.39 1.67±0.31 1.25±0.13 1.13±0.12 1.17±0.12 1.07±0.09 1.10±0.08 1.01±0.07 1.03±0.08 0.89±0.05 0.93±0.08 0.87±0.06 0.86±0.08 0.83±0.07 0.82±0.07 0.75±0.06 0.77±0.06 0.76±0.06 0.80±0.07 0.74±0.06 0.74±0.08 0.69±0.06 0.80±0.14 0.73±0.10|25.46±3.93 24.14±3.02 30.14±0.68 29.42±0.69 30.63±0.77 29.88±0.76 30.18±0.77 29.42±0.78 30.59±1.21 29.88±1.09 30.33±1.13 29.86±1.06 30.65±1.15 29.82±1.11 30.56±1.20 29.96±1.16 30.46±1.02 29.82±0.88 30.58±0.88 29.86±0.79 30.80±1.03 30.16±0.89 32.03±1.46 31.50±1.26<br><br>|0.87 0.00 0.42 0.00 0.45 0.00 0.25 0.00 0.28 0.00 0.24 0.00 0.22 0.00 0.21 0.00<br><br>0.15 0.00<br><br>0.16 0.00<br><br><br>0.21 0.00 0.50 0.00<br><br>|

### B Determining the maximum multiplicative noise level

In what follows, we write Ri for R[i] and Ri,k for the k-th element of R[i] to avoid excessive brackets. δi is a random vector where each component δi,k follows a uniform distribution U(1 − ϵ,1 + ϵ), and all components are mutually independent. The perturbed point is given by:

R˜i = (δi,1(Ri,1),δi,2(Ri,2),...,δi,d(Ri,d)).

To ensure that R˜i remains in the same cluster as Ri, it must satisfy:

∥R˜i − Ri∥2 < ∥R˜i − Rj∥2, where j = arg minj∗̸=i ∥R[i] − R[j]∥.

Expanding the squared norms on both sides of the inequality yields:

- ∥R˜i − Ri∥2 =

d

k=1

(δi,k − 1)2(Ri,k)2,

- ∥R˜i − Rj∥2 =

d

(δi,kRi,k − Rj,k)2.

k=1

Substituting into the inequality and simplifying gives:

d

[2δi,k(Ri,k(Rj,k − Ri,k) + (Ri,k2 − Rj,k2 )] < 0.

k=1

To ensure this inequality holds for all realizations of δi, we consider the worst-case scenario that maximizes the left-hand side. Define:

so the inequality becomes:

Ak = 2Ri,k(Rj,k − Ri,k), B =

d

(Ri,k2 − Rj,k2 ),

k=1

d

Akδi,k + B < 0. (5)

k=1

The worst-case δi,k is chosen to maximize Akδi,k:

Substituting these values gives:

Now simplify Ak + B:

δi,k =

1 + ϵ if Ak > 0, 1 − ϵ if Ak < 0.

d

d

|Ak| + B < 0. (6)

Ak + ϵ

k=1

k=1

Ak + B = 2 Ri,kRj,k − 2 Ri,k2 + Ri,k2 − Rj,k2

= 2 Ri,kRj,k − Ri,k2 − Rj,k2

= − Ri,k2 − 2 Ri,kRj,k + Rj,k2

= −∥Ri − Rj∥2

Substituting equation 7 into equation 6 yields:

−∥Ri − Rj∥2 + 2ϵ

d

|Ri,k(Rj,k − Ri,k)| < 0.

k=1

(7)

Solving for ϵ gives:

ϵmax < ∥Rj − Ri∥2 2 dk=1 |Ri,k(Rj,k − Ri,k)|

.

However, computing the denominator of this expression is relatively complex. To balance the efficiency of loss calculation, we instead adopt a tighter upper bound for ϵ.

By the Cauchy-Schwarz Inequality, the following relationship holds:

Thus, we have:

d

|Ri,k(Rj,k − Ri,k)| ≤ ∥Ri∥∥Rj − Ri∥.

k=1

ϵmax = ∥Rj − Ri∥2 2 dk=1 |Ri,k(Rj,k − Ri,k)|

≥

∥Rj − Ri∥2 2∥Ri∥∥Rj − Ri∥

= ∥Rj − Ri∥ 2∥Ri∥

.

The term on the right-hand side of the final inequality is the value of ϵ we used in the main text. This choice ensures that the perturbed R˜[i] always remains closer in Euclidean distance to R[i] than to any other R[j ̸= i].

### C Efficiency analysis

Appendix C.1 analyzes the ideal FLOPs cost breakdown of the vanilla MoE, as well as the overhead introduced by AoE and ERC loss. Appendix C.2 discusses efficiency with consideration of the multiple parallelism strategies used in real-world MoE pre-training. Both analyses demonstrate the practicality of our method.

#### C.1 FLOPs cost breakdown of three methods

MoE forward Each expert in a MoE layer involves the following operations, with their respective FLOP counts:

- • Two matrix multiplications of dimension T × d with d × D, accounting for 4TdD FLOPs. These correspond to the linear transformations parameterized by Wg and Wp.
- • One element-wise multiplication of T × D tensors and one SiLU activation applied to a T × D tensor. The computational cost of these operations is negligible compared to the matrix multiplications.
- • One matrix multiplication of dimension T ×D with D×d, contributing 2TDd FLOPs. This corresponds to the linear transformation parameterized by Wo.

Summing these components gives a total of 6TdD FLOPs per expert. For K experts processing T tokens, the total computational cost is therefore 6KTdD FLOPs.

Computational overhead of AoE AoE factorizes the expert matrix Wg ∈ D×d into two low-rank matrices of rank r. To maintain the same number of parameters as the original matrix, we require dr + Dr = Dd,

which gives r = dDd+D. The change in FLOPs compared to an MoE is:

  2ndr

 ,

− 2KDd

T

+ 2KDr

All experts use Wdown

Top-K experts use Wup

Top-K experts use original Wg

where T is the number of tokens. Substituting the value of r and simplifying leads to an extra computational cost of:

2T(n − K)dr.

Computational overhead of expert-router coupling loss It introduces n2 matrix multiplications, each operating on tensors of shape 1 × d and d × D. In total, this results in 2n2Dd extra FLOPs.

#### C.2 Throughputs under multiple parallelism strategies

We now assess the overhead of the ERC loss in a realistic large-scale pre-training setup that employs both data parallelism (DP) and expert parallelism (EP). As derived in our previous analysis, the computational cost of the ERC loss is equivalent to a forward pass on n2/3 tokens. When distributed across devices, the costs are:

- • Base MoE forward: K · T / dp_size
- • ERC overhead: n · (n / ep_size) / 3

Consider training our 15B-parameter model with the configuration: K = 8, T = 3 × 106, n = 256, dp_size = 64, and ep_size = 8. In this scenario, the ERC overhead constitutes a mere 0.72% of the base model’s forward pass cost. This theoretical estimate is consistent with our empirical measurements: we observe a throughput of 62.03B tokens/day for the baseline versus 61.52B tokens/day for our model, representing only a 0.82% reduction. With a smaller n = 64, as in our 3B models trained with dp_size=32 and ep_size=1 (i.e., EP disabled), the overhead ratio drops further to 0.18%. This analysis confirms the practical efficiency of our method.

- 1 import torch
- 2 import torch.nn as nn
- 3 import PseudoExpertClass
- 4
- 5 class MoE(nn.Module):
- 6
- 7 def __init__(self , args):
- 8 super().__init__()
- 9
- 10 self.experts = PseudoExpertClass(args)
- 11 # Shape of experts.Wg: (n, D, d)
- 12 self.R = torch.nn.Parameter(torch.empty(
- 13 args.n, args.d))
- 14
- 15 self.alpha = args.alpha
- 16
- 17 def erc_loss(self , M):
- 18 row_diff = (M - self.alpha * torch.diag(M).unsqueeze (1))
- 19 row_diff_clamped = torch.clamp(row_diff , min=0.0)
- 20
- 21 col_diff = (M - self.alpha * torch.diag(M).unsqueeze (0))
- 22 col_diff_clamped = torch.clamp(col_diff , min=0.0)
- 23
- 24 mask = torch.ones_like(M) - torch.eye(M.size(0), device=M.device)
- 25 total_diff = (row_diff_clamped + col_diff_clamped) * mask
- 26
- 27 return total_diff.mean()
- 28
- 29 def get_noisy_router(self , R):
- 30 with torch.no_grad():
- 31 norm_R = torch.norm(R, dim=1)
- 32 distances = torch.cdist(R, R, p=2)
- 33 distances.fill_diagonal_(float(’inf’))
- 34 min_dist , _ = torch.min(distances , dim=1)
- 35 eps = min_dist / 2 / norm_R
- 36
- 37 low = (1 - eps).unsqueeze (1)
- 38 high = (1 + eps).unsqueeze (1)
- 39 noise = torch.rand_like(R)
- 40 return (low + noise * (high - low)) * R
- 41
- 42 def forward(self , x):
- 43
- 44 erc_loss = 0.0
- 45 if self.training:
- 46 R = self.get_noisy_router(self.R)
- 47 M = torch.norm(torch.einsum(’jDd ,id->ijD’, self.experts.Wg, R), dim=-1)
- 48 erc_loss = self.erc_loss(M)
- 49
- 50 logits = x.view(-1, x.shape[-1]) @ self.R.T
- 51 scores = logits.softmax(dim=-1)
- 52 expert_weights , expert_indices = torch.topk(scores , dim=-1)
- 53
- 54 return self.experts(x, expert_weights , expert_indices), erc_loss

###### Figure 8 Pseudo code for expert-router coupling loss in PyTorch.

ARC-Challenge Commonsense QA COPA BoolQ HellaSwag

[Figure 13]

Figure 3

MoE +ℒ

OpenBook QA SciQ Social IQA MMLU (Avg.) WinoGrande

AoE

MoE

ARC-Challenge Commonsense QA COPA BoolQ HellaSwag

[Figure 14]

- Figure 6(a)

MoE +ℒ

MoE

MoE +ℒ - 𝛿

- Figure 6(b)

MoE +ℒ

MoE

MoE + ℒ

- Figure 6(c)

MoE 𝑹𝑾 𝑹𝑾 SiLU( 𝑹𝑾 ) Post-SwiGLU Output

OpenBook QA SciQ Social IQA MMLU (Avg.) WinoGrande

ARC-Challenge Commonsense QA COPA BoolQ HellaSwag

[Figure 15]

OpenBook QA SciQ Social IQA MMLU (Avg.) WinoGrande

ARC-Challenge Commonsense QA COPA BoolQ HellaSwag

[Figure 16]

OpenBook QA SciQ Social IQA MMLU (Avg.) WinoGrande

ARC-Challenge Commonsense QA COPA BoolQ HellaSwag

[Figure 17]

Figure 7

- α = 1

MoE

- α = 2
- α = 3

OpenBook QA SciQ Social IQA MMLU (Avg.) WinoGrande

###### Figure 9 Task-specific downstream results for previous experiments.

