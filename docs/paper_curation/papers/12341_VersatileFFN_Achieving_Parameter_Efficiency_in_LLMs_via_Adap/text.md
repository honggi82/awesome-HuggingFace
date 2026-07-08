## VersatileFFN: Achieving Parameter Efficiency in LLMs via Adaptive Wide-and-Deep Reuse

Ying Nie1 Kai Han1 Hongguang Li1 Hang Zhou1 Tianyu Guo1 Enhua Wu23 Xinghao Chen1 Yunhe Wang1

# arXiv:2512.14531v2[cs.CL]1Feb2026

### Abstract

The rapid scaling of Large Language Models (LLMs) has achieved remarkable performance, but it also leads to prohibitive memory costs. Existing parameter-efficient approaches such as pruning and quantization mainly compress pretrained models without enhancing architectural capacity, thereby hitting the representational ceiling of the base model. In this work, we propose VersatileFFN, a novel feed-forward network (FFN) that enables flexible reuse of parameters in both width and depth dimensions within a fixed parameter budget. Inspired by the dual-process theory of cognition, VersatileFFN comprises two adaptive pathways: a width-versatile path that generates a mixture of sub-experts from a single shared FFN, mimicking sparse expert routing without increasing parameters, and a depth-versatile path that recursively applies the same FFN to emulate deeper processing for complex tokens. A difficulty-aware gating dynamically balances the two pathways, steering “easy” tokens through the efficient width-wise route and allocating deeper iterative refinement to “hard” tokens. Crucially, both pathways reuse the same parameters, so all additional capacity comes from computation rather than memory. Experiments across diverse benchmarks and model scales demonstrate the effectiveness of the method. The code is available at https://github.com/ huawei-noah/noah-research/tree/ master/VersatileFFN.

### 1. Introduction

The remarkable success of Large Language Models (LLMs) is largely governed by scaling laws, which posit a power-

1Huawei Noah’s Ark Lab 2ISCAS 3University of Macau.Correspondence to: Kai Han <kai.han@huawei.com>, Yunhe Wang <yunhe.wang@huawei.com>.

Technical Report.

law relationship between model performance and parameter count (Kaplan et al., 2020; Hoffmann et al., 2022). Following this trajectory, the pursuit of state-of-the-art accuracy has led to the proliferation of models with hundreds of billions of parameters. This trend encompasses both dense architectures (Touvron et al., 2023; Chen et al., 2025; Zuo et al., 2025) and sparse Mixture-of-Experts (MoE) variants (Jiang

- et al., 2024; Dai et al., 2024; Comanici et al., 2025; Yang
- et al., 2025; Team et al., 2025), reflecting a consistent drive toward increasing model scale.

However, the massive parameter counts of these models incur a prohibitive memory footprint that severely limits their practical deployment. High-end accelerators with large memory capacity are expensive and scarce, and distributing model weights across multiple devices introduces significant communication overhead and engineering complexity. As a result, scaling models purely by increasing parameter counts faces growing infrastructural and economic constraints.

In recent years, a line of research has explored parameterefficient LLM architectures that maintain high performance under limited memory. Such methods include weight pruning (Frantar & Alistarh, 2023; Chen et al., 2025), quantization (Xiao et al., 2023; Fu et al., 2025), low-rank adaptations (Lu et al., 2025), and other compression techniques that reduce the storage footprint of pretrained models. However, these approaches primarily aim to approximate the capabilities of the original large model, and thus remain fundamentally constrained by its architectural capacity. They do not augment the model’s representational power, nevertheless, they trade off precision or connectivity for deployability, often hitting an upper bound determined by the base model’s design. There remains a clear need for architectures that are parameter-efficient by design, capable of attaining stronger performance under the same parameter budget, rather than merely compressing existing large models.

In this work, we introduce VersatileFFN, a parameterefficient architecture that versatilely reuse parameters and computation in feed-forward networks (FFNs), serving as an efficient replacement for the standard FFN in Transformer blocks while leaving the self-attention component unchanged. As illustrated in Figure 1, VersatileFFN inte-

- System 1 (Width-Versatile)
- System 2 (Depth-Versatile)

[Figure 1]

Simple tokens

[Figure 2]

Multiple lightweight experts. Rapid execution.

Input Tokens “artificial”、“world”、“peace”、 “cat”、“quantum” …

Weighted Fusion

Shared Base FFN

Output

One high-capacity expert. Deep reasoning.

[Figure 3]

Complex tokens

Figure 1. VersatileFFN integrates a width-versatile pathway for rapid execution and a depth-versatile pathway for deep reasoning, both derived from a shared base FFN. This design enables a flexible trade-off between the two computational dimensions.

grates two complementary mechanisms: a width-versatile pathway and a depth-versatile pathway, which together enable flexible trade-offs between computational width and depth using a tightly shared parameter set.

- • In the width-versatile FFN pathway, we introduce the concept of a versatile base expert, a single and multicapable FFN, from which we derive a mixture of subexperts through structured parameter reuse. Rather than instantiating separate experts, each sub-expert is formed by selectively activating and composing different, non-overlapping hidden subspaces of the same underlying versatile FFN. This approach enables the emulation of diverse expert behaviors with minimal memory overhead, preserving the adaptive token routing benefits of classical MoE architectures while avoiding the parameter proliferation typical of such designs.
- • The depth-versatile FFN pathway achieves depth-wise versatility by recursively reusing the same base FFN multiple times, effectively emulating the function of multiple layers. Rather than instantiating separate parameters for each processing step, this design enables tokens to undergo iterative refinement through cyclical application of the shared MLP. A GumbelSoftmax-based controller predicts a token-specific iteration count, allowing the model to dynamically allocate more processing steps to harder tokens. This approach maintains full differentiability during training while preserving parameter efficiency, as the same weights are repurposed depth-wise to create a versatile, adaptive computation structure.

Inspired by the dual-process theory of human cognition (Kahneman, 2011), we design an architecture that dy-

namically allocates computation: “easy” tokens are processed rapidly via a lightweight path, while “hard” tokens receive deeper, iterative reasoning, all under a fixed parameter budget. A difficulty-aware gating mechanism unifies the two pathways, using the expected iteration count from the depth-versatile controller as a proxy for token difficulty. This signal modulates the fusion weights between the outputs of width-versatile FFN and depth-versatile FFN, favoring efficient processing on width-versatile for easy tokens and shifting toward depth-versatile for harder ones. Crucially, both pathways share the same MLP parameters, ensuring added capacity comes from computation, not memory. We conduct extensive experiments across multiple benchmarks and model scales. Models equipped with VersatileFFN consistently outperform other parameters-matched or FLOPs-matched methods, demonstrating the effectiveness of our dual-process design.

### 2. Related Work

#### 2.1. Mixture-of-Experts

The Mixture-of-Experts (MoE) architecture serves as a foundational framework for scaling model capacity without a proportional increase in computational cost. Originally introduced by (Jacobs et al., 1991; Jordan & Jacobs, 1994), the concept was later popularized in the large-scale networks by (Shazeer et al., 2017). Subsequent advancements, such as GShard (Lepikhin et al., 2020) and Switch Transformer (Fedus et al., 2022), have demonstrated that replacing standard feed-forward layers in Transformers with MoE layers facilitates efficient pre-training at the trillion-parameter scale, yielding substantial performance improvements. Recent advancements have focused on refining expert granularity and routing strategies (Jiang et al., 2024; Dai et al., 2024; Jin

et al., 2024; Huang et al., 2024a; Wang et al., 2024), enabling the successful deployment of large-scale MoE models in industrial applications (Yang et al., 2025; Comanici et al., 2025; Team et al., 2025; Zhao et al., 2025).

Despite their inference efficiency, the massive storage requirements of standard MoE models present significant deployment hurdles. This has catalyzed research into parameter-efficient MoE, utilizing techniques such as expert merging (Li et al., 2022; Zhao et al., 2024), pruning (Sarkar et al., 2024; Lu et al., 2024), and quantization (Nie et al., 2022; Dong et al., 2024; Huang et al., 2024b; Zhou et al., 2025). Another emerging direction involves composing lightweight, task-specific modules (e.g. LoRA adapters) into mixture-based systems, as seen in LoRAHub (Huang

- et al., 2023), MoA (Feng et al., 2024), MoLE (Wu et al., 2024), and MoRAgent (Han et al., 2025). While these approaches mitigate some memory constraints, they typically rely on instantiating distinct, physically separate expert parameters, which limits the extent of parameter efficiency compared to architectures designed for intrinsic reuse.

- 2.2. Recursive Computation and Adaptive Inference

A promising avenue for maximizing parameter efficiency is the decoupling of model depth from parameter count through recursive computation. Early works such as Universal Transformers (Dehghani et al., 2018) and ALBERT (Lan et al., 2019) demonstrate that cross-layer parameter sharing can induce beneficial inductive biases and improve parameter efficiency for language modeling. Recent theoretical analysis further establishes that such recursive architectures can emulate complex algorithms, acting as universal computers (Giannou et al., 2023; Gao et al., 2024), and generalize to sequence lengths far beyond those encountered during training (Fan et al., 2024; Gong et al., 2025).

Beyond the paradigm of static parameter sharing, recursive structures facilitate dynamic computation, where the computational budget is adapted to the complexity of the input. Recent advancements such as Mixture-of-Depths (Raposo

- et al., 2024), Mixture-of-Recurrence (Bae et al., 2025) and Dynamic resolution network (Zhu et al., 2021) dynamically allocate inference FLOPs, allowing models to expend more ”thinking time” on harder tokens or images. This iterative refinement is particularly potent for reasoning tasks, offering significant advantages over static counterparts. Several studies (Gatmiry et al., 2024; Saunshi et al., 2025; Merrill & Sabharwal, 2025; Zhu et al., 2025) indicate that increasing computational depth often yields greater performance gains than merely increasing width. In addition, HRM (Wang
- et al., 2025) and TRM (Jolicoeur-Martineau, 2025) show that iteratively reasoning models can outperform larger static counterparts. While these approaches successfully exploit the depth dimension for efficiency, they typically treat the

recurrent layer as a monolithic block, leaving the potential for fine-grained, width-wise parameter reuse largely unexplored. Ours aims to bridge this gap by introducing a multidimensional sharing mechanism that exploits redundancy across both depth and width.

### 3. Method

#### 3.1. Architectural Overview

Let X ∈ RB×T×d denote the input tensor to the Transformer block, where B represents the batch size, T the sequence length and d the feature dimension. In a standard Transformer layer, the input first undergoes a Self-Attention mechanism followed by residual connection and normalization:

H = X + Attention(LayerNorm(X)), (1)

Subsequently, the hidden states are processed by a FeedForward Network (FFN). We define the FFN transformation function F(·) as:

Y = F(H)

(2)

= H + Wout ϕ(Wproj LayerNorm(H)).

where Wproj ∈ Rd×d

hidden×d denote the projection and output weights, respectively, and ϕ is a distinct non-linear activation function.

hidden, Wout ∈ Rd

We introduce VersatileFFN, a parameter-efficient alternative to the standard FFN. While retaining the canonical selfattention architecture, VersatileFFN reconfigures the feedforward computation into two complementary pathways that share the underlying FFN weights (i.e., Wproj and Wout):

- • Width-Versatile (Ywidth): This pathway functions as a virtual MoE module. It routes tokens to specialized sub-experts instantiated via structured subsets of the shared weights, facilitating rapid, domain-specialized response without increasing parameter count.
- • Depth-Versatile (Ydepth): This pathway implements a recursive computation mechanism. It iteratively refines token representations by reusing the full capacity of

Wproj and Wout, thereby allocating increased computational depth to semantically complex tokens.

The final output Y is synthesized through a dynamic, difficulty-aware fusion of these two pathways:

Y = λ · Ywidth + (1 − λ) · Ydepth, (3)

where λ ∈ [0,1) acts as a gating coefficient modulated by the token difficulty. Specifically, λ is derived from the predicted iteration count of the depth-versatile controller. This mechanism ensures a flexible computational trade-off:

ensuring broad semantic coverage for simple tokens via the width pathway, while reserving deep, iterative reasoning for harder tokens via the depth pathway, all within a fixed parameter budget.

#### 3.2. Width-Versatile Mechanism

The Width-Versatile pathway, implemented as a virtual MoE, augments the model’s representational capacity while circumventing the prohibitive memory overhead typically associated with physical expert instantiation.

Construction of Virtual Experts. Rather than allocating discrete weight matrices for each expert, we leverage the shared weights Wproj and Wout as a contiguous parameter substrate. We define a set of N virtual experts by extracting structured, non-overlapping subspaces from the hidden dimension dhidden as illustrated in Figure 2. Specifically, let dexpert denote the hidden dimension of a single virtual expert. To strictly satisfy the non-overlapping requirement, we impose the constraint N × dexpert ≤ dhidden, ensuring that the total capacity of the virtual experts does not exceed the shared parameter space. We employ a strided slicing strategy to map each expert k ∈ {0,...,N − 1} to a specific view of the shared parameters. This design adheres to two principles: (1) Parameter Efficiency, achieved through the dense reuse of the backbone weights. (2) Functional Orthogonality, ensured by the non-overlapping allocation, which prevents interference between experts.

The stride S is calculated to uniformly distribute expert views across the hidden dimension:

dhidden − dexpert N − 1

. (4)

S =

The indices Iproj(k) corresponding to the k-th expert is:

Iproj(k) = {j ∈ Z | ⌊k · S⌋ ≤ j < ⌊k · S⌋ + dexpert}. (5) Leveraging the pre-trained weights of the dense model, we assign Iout(k) = Iproj(k) to maintain the structural alignment between the projection and output. The effective weight matrices for the virtual expert k are thus derived as:

Wproj(k) = Wproj[:,Iproj(k) ], Wout(k) = Wout[Iout(k),:]. (6)

Sparse Token Routing. We employ a learnable gating to orchestrate token assignment. Given the post-attention token representation H ∈ RB×T×d, a router Wg ∈ Rd×N computes the gating logits G(H) = WgH. We adopt a standard Top-K strategy, activating only the experts with the highest routing scores. The output Ywidth is computed as the probability-weighted sum of the selected experts:

gk · Yk, (7)

Ywidth =

k∈Top-K(G(H))

###### One large and real expert

|| | |…| |
|---|---|---|---|
<br><br>| |
|---|
| |
|…|
| |
<br><br>…<br><br>Slice<br><br>Slice<br><br>𝐖𝑜𝑢𝑡<br><br>𝐖𝑝𝑟𝑜𝑗|
|---|

…

###### Multiple lightweight and virtual experts

||𝐖𝑝𝑟𝑜𝑗0<br><br>|
|---|
<br><br>|𝐖𝑜𝑢𝑡0<br><br>|
|---|
|
|---|

||𝐖𝑝𝑟𝑜𝑗1<br><br>|
|---|
<br><br>|𝐖𝑜𝑢𝑡1<br><br>|
|---|
|
|---|

||𝐖𝑝𝑟𝑜𝑗𝑁−1<br><br>|
|---|
<br><br>|𝐖𝑜𝑢𝑡𝑁−1<br><br>|
|---|
|
|---|

…

Figure 2. Illustration of decoupling one large and real expert into multiple lightweight and virtual experts.

where gk represents the Softmax-normalized gating probability, and the expert-specific transformation Yk(H) is:

##### Yk = H + Wout(k) ϕ(Wproj(k) LayerNorm(H)). (8)

This sparse activation mechanism decouples computational cost from model capacity. Following previous experience (Fedus et al., 2022), we incorporate an auxiliary loadbalancing loss during training to prevent expert collapse.

#### 3.3. Depth-Versatile Mechanism

The Depth-Versatile pathway introduces token-wise adaptive computation by applying the shared MLP block recursively, enabling dynamic depth allocation.

Recursive Weight Application. In contrast to the WidthVersatile pathway, The Depth-Versatile pathway utilizes the entire shared backbone weights (Wproj,Wout) without slicing. This ensures that the recursive refinement process benefits from the full expressivity of the model parameters. Let F(·) denote the standard FFN transformation defined in Eq. 2. We generate a sequence of intermediate representations by recursively applying F:

H(ℓ) = F H(ℓ−1) , ℓ = 1,...,Lmax, (9)

where H(0) = H, and Lmax denotes the maximum allowable iterations. This iterative process allows the model to progressively refine token representations, capturing complex dependencies through repeated non-linear transformations without increasing the parameter budget.

Differentiable Loop Prediction. The optimal number of iterations per token is inherently discrete and nondifferentiable. To facilitate end-to-end training, we introduce a prediction head Wloop ∈ Rd×L

max to estimate the

required computational depth. The logits for the loop count are computed as Ploop(H) = WloopH. We then employ the Gumbel-Softmax relaxation (Jang et al., 2016) to sample a differentiable probability vector p ∈ ∆L

max−1: p = Softmax

Ploop(H) + g τ

, (10) where g ∼ Gumbel(0,1) and τ is the temperature.

We utilize the Straight-Through Estimator (STE) during training: the forward pass executes a discrete selection (via argmax) to simulate the inference-time decision, while gradients are propagated through the continuous relaxation p during the backward pass. To stabilize convergence, τ is annealed exponentially from an initial value (e.g. 5.0) to a lower bound (e.g. 0.1). The output of the Depth-Versatile pathway is computed as the soft-weighted combination of intermediate states:

Lmax

pℓ · H(ℓ). (11)

Ydepth =

ℓ=1

During inference, the probabilistic sampling is replaced by a deterministic decision ℓˆ = arg max(p). The loop is executed exactly ℓˆtimes, and the final state is taken as Ydepth = H(ℓˆ).

#### 3.4. Difficulty-Aware Fusion

The Width-Versatile and Depth-Versatile pathways represent distinct yet complementary computational paradigms: the former offers broad, parallelizable semantic capacity via virtual experts, while the latter facilitates intensive, sequential reasoning through iterative refinement. VersatileFFN synergizes these mechanisms via a difficulty-aware fusion scheme, which leverages the predicted computational depth as a proxy for token complexity.

Expected Loop Count as Difficulty Proxy. We postulate that the requisite depth of recursive processing serves as an intrinsic measure of semantic difficulty. Linguistically trivial tokens (e.g. stopwords, high-frequency bigrams) typically necessitate minimal transformation (loop count → 1), whereas semantically ambiguous or logically complex tokens benefit from deep, recurrent refinement (loop count → Lmax).

We quantify this difficulty by computing the expected loop count, E[L], derived from the soft probability distribution p output by the loop predictor:

Lmax

ℓ · pℓ. (12)

E[L] =

ℓ=1

During training, the differentiability of p allows E[L] ∈ [1,Lmax] to serve as a continuous signal for gradient propagation. During inference, although the loop execution

Algorithm 1 Training Computation of VersatileFFN Require: Input tensor X, Shared Weights Wproj,Wout,

Max loops Lmax Ensure: Output tensor Y

- 1: Compute post-attention representation H via Eq. 1
- 2: Width-Versatile Pathway:
- 3: Construct virtual experts by slicing shared weights via Eq. 5 and Eq. 6
- 4: Compute output Ywidth with Top-k routing via Eq. 7
- 5: Depth-Versatile Pathway:
- 6: Predict loop count probabilities p via Eq. 10
- 7: Initialize H(0) ← H
- 8: for ℓ = 1 to Lmax do
- 9: Update recursive hidden state H(ℓ) using full shared weights via Eq. 9
- 10: end for
- 11: Aggregate states to obtain Ydepth via Eq. 11
- 12: Difficulty-Aware Fusion:
- 13: Calculate expected loop count E[L] via Eq. 12
- 14: Compute difficulty-aware fusion scalar λ via Eq. 13
- 15: Compute output Y by fusing pathways via Eq. 3
- 16: Return Y

becomes discrete, E[L] remains a robust indicator of the model’s uncertainty and the token’s processing demand.

Dynamic Fusion Modulation. To unify the two pathways, we define a dynamic gating scalar, λ, which modulates the fusion balance based on the estimated difficulty. We formulate λ to be inversely proportional to the expected computational cost:

Lmax − E[L] Lmax

. (13)

λ =

By construction, λ ∈ [0,1). For “easy” tokens with low expected loops, λ → 1, biasing the output toward the efficient width. Conversely, for “hard” tokens, λ → 0 , shifting the focus toward the depth. The final output Y is synthesized according to Eq. 3. This mechanism automatically allocates computational resources: simple patterns are resolved rapidly via the lightweight virtual MoE, while complex reasoning tasks command the full depth of the recursive loop. Inference Optimization. The complete training procedure is summarized in Algorithm 1. To minimize latency during inference, we introduce two inference-time optimizations:

- • Discrete Early-Exit: The Depth-Versatile pathway transitions from soft aggregation to a hard cutoff. Recursion terminates immediately at the predicted step ℓˆ = arg max(p), avoiding unnecessary computation beyond the required depth.
- • Conditional Parallelism: We employ a threshold-based

[Figure 4]

[Figure 5]

0 10 20 30 40

- (a) Various total experts and active experts.

[Figure 6]

[Figure 7]

- (b) Various maximum number of loops.

Figure 3. The curves of training loss and average accuracy for various configurations of width-versatile and depth-versatile mechanisms.

execution strategy. If the contribution of the WidthVersatile pathway is negligible (i.e., λ = 0) the branch is pruned entirely. Otherwise, both pathways are executed in parallel to maximize hardware utilization and computational throughput.

The inference procedure is summarized in Algorithm 2.

### 4. Experiments

#### 4.1. Implementation Details

Experiments Setup. To validate our approach, we conducted pre-training experiments on the FineWeb-Edu dataset (Penedo et al., 2024), our implementation is based on OLMo2 (OLMo et al., 2024). Specifically, we train a 354M parameter model on 40B tokens, a 720M parameter model on 70B tokens and a 1.21B parameter model on 100B tokens. The 40B and 70B token sets are subsets derived from the 100B token corpus (HuggingFaceFW, 2024). Following the architectural settings of the open-source OLMo2 models, we set the hidden dimensions to 1,024 for 354M model, 1,536 for the 720M model and 2,048 for 1.21B model. All models consist of 15 layers and utilize SwiGLU for activation and RMSNorm (Zhang & Sennrich, 2019) for normalization. Besides, The OLMo2 tokenizer with a vocabulary size of 50,280 is employed in our models. Further training setting and evaluation benchmarks are provided in Appendix B.

Versatile Setup of Width and Depth. To determine the optimal configuration for the VersatileFFN architecture, specifically the number of active experts and total experts in the width (i.e., k and N in Eq. 7) and the maximum loops in the depth (i.e., Lmax in Eq. 11), we conduct ablation

studies using the 354M model trained on 40B tokens. In these experiments, we maintained one shared expert for the width-versatile pathway, while disabling virtual experts within the depth-versatile pathway. The training loss and average accuracy across eight academic benchmarks during training is illustrated in Figure 3. For width-versatile, the losses (and average accuracies) for varying expert configurations, specifically 4-choose-1, 8-choose-2, 8-choose-4, and 16-choose-4 are 2.655 (50.16%), 2.633 (50.98%), 2.633 (50.96%), and 2.634 (50.22%), respectively. We select the 8choose-2 setting as our primary experimental configuration due to its higher performance. For depth-versatile, when the number of loops is 1, 2, 4, and 6, the final training losses are 2.654, 2.631, 2.625, and 2.623, respectively. The average accuracies are 50.09%, 51.47%, 51.98%, and 51.94%. It can be seen that 4 loops can achieve the best balance between accuracy and computation. Therefore, we set the maximum number of loops to 4 in our main experiment.

#### 4.2. Evaluation Results

To rigorously evaluate the efficacy of our proposed method, we conduct continued pre-training on a pre-trained base model for one additional epoch, utilizing the identical corpus. We benchmark our approach against the following methods: 1) Mixture-of-Experts (MoE): A sparse architecture that inherits the configuration of the dense base model but incorporates additional 8 small experts with activating top-2 experts during the forward pass. 2) k-Loop: A variant of the dense architecture that retains the original architecture but introduces recurrence by iteratively applying the FFN k times at each layer. To ensure a fair comparison, both methods also share the experimental setting of continued pre-training on the same pre-trained base model.

Efficiency Comparison. We first provide an analysis of the efficiency in Table 2, including the average accuracy, model parameters, and FLOPs of the FFN part. We calculate the FLOPs as 1 token input and 1 token output. Since the same architecture is maintained, the number of parameters for methods k-Loop is consistent with that of the base model. The proposed VersatileFFN method, however, designs a router and loop predictor, which introduces negligible parameters. The MoE method, on the other hand, introduces real small experts on top of the base model, leading to a significant increase in parameters. For FFN FLOPs, the computational cost of k-Loop method is clearly k times that of base. We calculate the FLOPs of VersatileFFN based on the inference statistics of the ARC-c dataset. Specifically, we calculate the average loops of all layers Nmean and the proportion P of loops not equal to the maximum loop, and then compute the FLOPs by Base × Nmean + (MoE − Base) × P. Benchmark Accuracy. We report the detailed zeroshot performance of VersatileFFN against other methods

Table 1. Comparison of zero-shot performance on standard NLP benchmarks. Results are reported for two model scales. ”Avg.” indicates the average accuracy across all eight tasks.

METHOD LOSS PIQA HELLASWAG OBQA SCIQ ARC-E ARC-C COMM WINO AVG. Base Model Params: 354.71M

BASE 2.779 65.13 37.73 32.40 78.60 56.32 29.77 32.76 51.14 47.98 MOE 2.585 68.93 45.33 33.20 85.40 63.16 29.77 34.64 51.38 51.48

2-LOOP 2.631 67.90 44.06 34.60 84.60 64.56 28.76 35.46 51.78 51.47 4-LOOP 2.625 68.44 44.09 34.60 84.80 64.56 30.77 36.53 52.01 51.98 6-LOOP 2.623 67.16 44.06 36.00 84.30 64.04 30.77 35.95 53.20 51.94

VERSATILEFFN 2.617 69.10 43.95 35.00 85.70 64.56 30.10 36.69 53.51 52.33 Base Model Params: 720.81M

BASE 2.519 69.48 47.69 37.20 86.40 65.61 34.11 35.14 55.01 53.83 MOE 2.411 72.31 54.47 38.00 89.00 67.72 34.45 37.10 53.91 55.87

2-LOOP 2.448 71.33 52.99 37.00 88.20 68.42 35.45 37.84 55.41 55.83 4-LOOP 2.441 71.87 53.43 37.00 88.60 68.25 36.45 39.39 55.64 56.33 6-LOOP 2.430 71.98 53.96 38.80 89.30 69.47 34.45 38.98 55.49 56.55

VERSATILEFFN 2.430 72.03 53.44 38.00 88.90 71.05 36.12 40.79 55.88 57.03 Base Model Params: 1.21B

BASE 2.385 73.07 54.18 38.00 88.90 69.12 35.12 40.05 55.49 56.74 MOE 2.278 74.21 59.37 41.00 91.10 70.35 41.14 41.20 58.80 59.65

2-LOOP 2.328 74.59 59.08 39.60 91.10 71.93 41.47 40.70 57.38 59.48 4-LOOP 2.322 73.97 59.41 40.70 91.50 72.86 41.51 42.86 57.87 60.09 6-LOOP 2.320 74.76 59.86 41.20 91.40 71.58 40.47 41.93 59.19 60.05

VERSATILEFFN 2.319 74.65 60.17 40.80 91.90 73.16 41.14 43.73 58.17 60.47

Table 2. Efficiency comparison for various methods. METHOD AVG. PARAMS (B) FLOPS (G)

BASE 47.98 0.35 0.38 MOE 51.48 0.54 0.47

2-LOOP 51.47 0.35 0.75 4-LOOP 51.98 0.35 1.50 6-LOOP 51.94 0.35 2.25

VERSATILEFFN 52.33 0.35 1.24

BASE 53.83 0.72 0.85 MOE 55.87 1.15 1.06

2-LOOP 55.83 0.72 1.70 4-LOOP 56.33 0.72 3.40 6-LOOP 56.55 0.72 5.10

VERSATILEFFN 57.03 0.72 2.59

BASE 56.74 1.21 1.51 MOE 59.65 1.97 1.89

2-LOOP 59.48 1.21 3.02 4-LOOP 60.09 1.21 6.04 6-LOOP 60.05 1.21 9.06

VERSATILEFFN 60.47 1.21 4.69

in Table 1. From the results, VersatileFFN consistently achieves the highest average accuracy across both model scales. Specifically, on the 1.21B scale, our method attains an average accuracy of 60.47%, outperforming the strong MoE baseline (59.65%) and the 6-Loop method (60.05%). It is worth noting that while MoE achieves the lowest loss (2.278 vs. 2.319 for VersatileFFN), this advantage does not translate into accuracy. VersatileFFN demonstrates more robust generalization capabilities, particularly on reasoningintensive tasks such as ARC-e and COMM, where it leads by a significant margin (+2.81% over MoE on ARC-e).

As shown in Table 2, a key advantage of VersatileFFN is its parameter efficiency. Unlike MoE, which significantly increase the total parameters, for example, from 1.21B to 1.97B (+63%), VersatileFFN introduces negligible parameter overhead. Regarding computational complexity, VersatileFFN is significantly more efficient than the 4-Loop and 6-Loop methods. For instance, at the 350M scale, VersatileFFN requires approximately 45% fewer FLOPs than the 6-Loop method while achieving superior accuracy. This indicates that VersatileFFN allocates computational resources more effectively than simply stacking recurrent loops, offering a compelling balance between model size, training cost, and downstream performance.

#### 4.3. Visualized Analysis

Actual loops. We first visualize the average predicted loop counts per layer on the ARC-c dataset in Figure 4. The smaller 354M model exhibits a late-stage allocation strategy, dedicating the highest computational budget to the final layers (specifically layers 11–14). The medium-sized 720M model transitions to a middle-heavy distribution, concentrating recurrence primarily within the intermediate layers while tapering off at the boundaries. In contrast, the 1.21B model displays a front-loaded profile: the recurrence intensity peaks sharply at Layer 2 and subsequently stabilizes into a consistent plateau for the remainder of the network.

Word cloud. We then analyze the word cloud of 354M model at Layer 0 on ARC-c in Figure 5. A lower gating coefficients λ indicates more loops. The words in the left is dominated by specific action verbs and similar words such

[Figure 8]

[Figure 9]

[Figure 10]

Figure 4. The average predicted loops per layer on ARC-c dataset. From left to right: 354M, 720M, 1.21B model.

[Figure 11]

[Figure 12]

λ=0.25 λ=0.75

Figure 5. Word cloud for various gating coefficients on ARC-c.

as clean, remove, cut, cup. Conversely, the right consists primarily of high-frequency, generic terms like make, use, water, will. These tokens typically rely on shallow surface statistics or syntactic cues, allowing the model to capture their representations with minimal recurrence.

#### 4.4. Ablation Studies

We conduct a series of ablation studies using the 354M base model on 40B tokens.

Table 3. Analysis on each components in VersatileFFN. METHOD WIDTH DEPTH GATING LOSS ACC.

BASE 2.779 47.98

✓ 2.633 50.98

✓ 2.629 51.35 ✓ ✓ 2.618 52.10 ✓ ✓ ✓ 2.617 52.33

OURS

Analysis on Each Components. From Table 3, we observe that both width-versatile and depth-versatile individually outperform the baseline. When the fusion gating is removed , we adopt simple averaging to fuse the outputs of the width and depth pathways. The combination of all modules yields the lowest loss and the highest average accuracy. This result confirms that the two branches are complementary, functioning synergistically to enhance model expressivity.

Various MoE experts and k-Loop. We further investigate the impact of loop depth and expert configuration in Table 4. As shown in (A), while increasing the loop count to 6 minimizes loss, the zero-shot average accuracy peaks at 4 loops,

suggesting that excessive recurrence may lead to overfitting. From (B), we observe that the (8, 2) expert configuration achieves the optimal accuracy.

Table 4. Analysis of the effect of: (A) Various Loops under MoE8-2; (B) Various experts under 4-Loop.

(A) EFFECT OF LOOPS LOOPS LOSS ACC.

2 2.625 51.54 4 2.617 52.33 6 2.615 51.61

(B) EFFECT OF MOE EXPERTS EXPERTS LOSS ACC.

8-2 2.617 52.33 8-4 2.617 52.22

16-4 2.617 51.72

Performance without Continued Pre-training. Table 5 shows the performance trained from scratch. VersatileFFN attains an accuracy of 51.14%, yielding a 3.16% improvement over the Base and outperforming other methods.

Table 5. Performance without continued pre-training. METHOD LOSS ACC.

BASE 2.779 47.98 MOE 2.639 50.90

4-LOOP 2.641 50.81 VERSATILEFFN 2.654 51.14

### 5. Conclusion

In this work, we present VersatileFFN, a novel architectural paradigm that decouples model performance from parameter scaling by prioritizing computational flexibility over memory expansion. By integrating a width-versatile path for efficient computing (System 1) and a depth-versatile path for iterative reasoning (System 2), our method effectively synthesizes width and depth within a strictly constrained parameter budget. Extensive experiments demonstrate that VersatileFFN consistently outperforms other methods, validating that intelligent weight reuse and adaptive computation are viable alternatives to simply scaling model size. We hope this work encourages further exploration into ”compute-heavy, memory-light” architectures, paving the way for deploying sophisticated reasoning capabilities in resource-constrained environments where memory is the primary bottleneck.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Bae, S., Kim, Y., Bayat, R., Kim, S., Ha, J., Schuster, T., Fisch, A., Harutyunyan, H., Ji, Z., Courville, A., et al. Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation. arXiv preprint arXiv:2507.10524, 2025.

Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.

Chen, H., Qin, J., Guo, J., Yuan, T., Yin, Y., Zhen, H., Wang, Y., Li, J., Meng, X., Zhang, M., et al. Pangu light: Weight re-initialization for pruning and accelerating llms. arXiv preprint arXiv:2505.20155, 2025.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Burstein, J., Doran, C., and Solorio, T. (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/ N19-1300. URL https://aclanthology.org/ N19-1300/.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Dai, D., Deng, C., Zhao, C., Xu, R., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, Ł. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.

Dong, P., Li, L., Zhong, Y., Du, D., Fan, R., Chen, Y., Tang, Z., Wang, Q., Xue, W., Guo, Y., et al. Stbllm: Breaking the 1-bit barrier with structured binary llms. arXiv preprint arXiv:2408.01803, 2024.

Fan, Y., Du, Y., Ramchandran, K., and Lee, K. Looped transformers for length generalization. arXiv preprint arXiv:2409.15647, 2024.

Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Feng, W., Hao, C., Zhang, Y., Han, Y., and Wang, H. Mixture-of-loras: An efficient multitask tuning for large language models. arXiv preprint arXiv:2403.03432, 2024.

Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International conference on machine learning, pp. 10323–10337. PMLR, 2023.

Fu, Z., Ding, N., Han, K., Yu, X., Li, X., Chen, X., Tang, Y., and Wang, Y. Eaquant: Enhancing post-training quantization for moe models via expert-aware optimization. arXiv preprint arXiv:2506.13329, 2025.

Gao, Y., Zheng, C., Xie, E., Shi, H., Hu, T., Li, Y., Ng, M. K., Li, Z., and Liu, Z. On the expressive power of a variant of the looped transformer. CoRR, 2024.

Gatmiry, K., Saunshi, N., Reddi, S. J., Jegelka, S., and Kumar, S. Can looped transformers learn to implement multi-step gradient descent for in-context learning? arXiv preprint arXiv:2410.08292, 2024.

Giannou, A., Rajput, S., Sohn, J.-y., Lee, K., Lee, J. D., and Papailiopoulos, D. Looped transformers as programmable computers. In International Conference on Machine Learning, pp. 11398–11442. PMLR, 2023.

Gong, Z., Teng, J., and Liu, Y. What makes looped transformers perform better than non-recursive ones (provably). arXiv preprint arXiv:2510.10089, 2025.

Han, J., Yan, B., Guo, T., Bai, Z., Zheng, M., Chen, H., and Nie, Y. Moragent: Parameter efficient agent tuning with mixture-of-roles. arXiv preprint arXiv:2512.21708, 2025.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Huang, C., Liu, Q., Lin, B. Y., Pang, T., Du, C., and Lin, M. Lorahub: Efficient cross-task generalization via dynamic lora composition. arXiv preprint arXiv:2307.13269, 2023.

Huang, Q., An, Z., Zhuang, N., Tao, M., Zhang, C., Jin, Y., Xu, K., Chen, L., Huang, S., and Feng, Y. Harder tasks need more experts: Dynamic routing in moe models. arXiv preprint arXiv:2403.07652, 2024a.

Huang, W., Liao, Y., Liu, J., He, R., Tan, H., Zhang, S., Li, H., Liu, S., and Qi, X. Mixture compressor for mixture-of-experts llms gains more. arXiv preprint arXiv:2410.06270, 2024b.

HuggingFaceFW. fineweb-edu (revision 22b0aca), 2024. URL https://huggingface.co/datasets/ HuggingFaceFW/fineweb-edu.

Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.

Jang, E., Gu, S., and Poole, B. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Jin, P., Zhu, B., Yuan, L., and Yan, S. Moe++: Accelerating mixture-of-experts methods with zero-computation experts. arXiv preprint arXiv:2410.07348, 2024.

Jolicoeur-Martineau, A. Less is more: Recursive reasoning with tiny networks. arXiv preprint arXiv:2510.04871, 2025.

Jordan, M. I. and Jacobs, R. A. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2): 181–214, 1994.

Kahneman, D. Thinking, fast and slow. macmillan, 2011.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., and Soricut, R. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942, 2019.

Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.

Li, M., Gururangan, S., Dettmers, T., Lewis, M., Althoff, T., Smith, N. A., and Zettlemoyer, L. Branch-train-merge: Embarrassingly parallel training of expert language models. arXiv preprint arXiv:2208.03306, 2022.

- Lu, X., Liu, Q., Xu, Y., Zhou, A., Huang, S., Zhang, B., Yan, J., and Li, H. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. arXiv preprint arXiv:2402.14800, 2024.
- Lu, Y.-C., Chen, C.-Y., Chang, C.-C., Hu, Y.-F., and Wu, K.C. Flrc: Fine-grained low-rank compressor for efficient llm inference. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 14956–14966, 2025.

Merrill, W. and Sabharwal, A. A little depth goes a long way: The expressive power of log-depth transformers. arXiv preprint arXiv:2503.03961, 2025.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

Nie, Y., Han, K., Diao, H., Liu, C., Wu, E., and Wang, Y. Redistribution of weights and activations for addernet quantization. Advances in Neural Information Processing Systems, 35:22739–22751, 2022.

OLMo, T., Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., et al. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656, 2024.

Penedo, G., Kydl´ıˇcek, H., Lozhkov, A., Mitchell, M., Raffel, C. A., Von Werra, L., Wolf, T., et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37: 30811–30849, 2024.

Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P. C., and Santoro, A. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. arXiv preprint arXiv:2404.02258, 2024.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. arXiv preprint arXiv:1907.10641, 2019.

Sarkar, S., Lausen, L., Cevher, V., Zha, S., Brox, T., and Karypis, G. Revisiting smoe language models by evaluating inefficiencies with task specific expert pruning. arXiv preprint arXiv:2409.01483, 2024.

Saunshi, N., Dikkala, N., Li, Z., Kumar, S., and Reddi, S. J. Reasoning with latent thoughts: On the power of looped transformers. arXiv preprint arXiv:2502.17416, 2025.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., and Yadkori, Y. A. Hierarchical reasoning model. arXiv preprint arXiv:2506.21734, 2025.

Wang, Z., Zhu, J., and Chen, J. Remoe: Fully differentiable mixture-of-experts with relu routing. arXiv preprint arXiv:2412.14711, 2024.

Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions. In NUT@EMNLP, 2017.

Wu, X., Huang, S., and Wei, F. Mixture of lora experts. arXiv preprint arXiv:2404.13628, 2024.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models. In International conference on machine learning, pp. 38087–38099. PMLR, 2023.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. HellaSwag: Can a machine really finish your sentence? In Korhonen, A., Traum, D., and M`arquez, L. s. (eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1472. URL https://aclanthology.org/P19-1472/.

Zhang, B. and Sennrich, R. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

Zhao, G., Fu, Y., Li, S., Sun, X., Xie, R., Wang, A., Han, W., Yang, Z., Sun, W., Zhang, Y., et al. Towards a comprehensive scaling law of mixture-of-experts. arXiv preprint arXiv:2509.23678, 2025.

Zhao, H., Qiu, Z., Wu, H., Wang, Z., He, Z., and Fu, J. Hypermoe: Towards better mixture of experts via transferring among experts. arXiv preprint arXiv:2402.12656, 2024.

Zhou, Y., Li, Z., Zhang, J., Wang, J., Wang, Y., Xie, Z., Chen, K., and Shou, L. Floe: On-the-fly moe inference on memory-constrained gpu. arXiv preprint arXiv:2505.05950, 2025.

Zhu, M., Han, K., Wu, E., Zhang, Q., Nie, Y., Lan, Z., and Wang, Y. Dynamic resolution network. Advances in Neural Information Processing Systems, 34:27319– 27330, 2021.

Zhu, R.-J., Wang, Z., Hua, K., Zhang, T., Li, Z., Que, H., Wei, B., Wen, Z., Yin, F., Xing, H., et al. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741, 2025.

Zuo, J., Nie, Y., Guo, T., Zhang, H., Hong, J., Sang, N., Gao, C., and Han, K. L-man: A large multi-modal model unifying human-centric tasks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 11095–11103, 2025.

### A. Inference Optimization

To minimize latency during inference, we introduce two inference-time optimizations: Discrete Early-Exit and Conditional Parallelism. The complete inference procedure is illustrated in Algorithm 2.

Algorithm 2 Inference Computation of VersatileFFN Require: Input tensor X, Shared Weights Wproj,Wout, Max loops Lmax Ensure: Output tensor Y

- 1: Controller & Gating:
- 2: Compute post-attention representation H via Eq. 1
- 3: Compute loop probabilities p = Softmax(WloopH)
- 4: Determine discrete loop count ℓˆ= arg max(p)
- 5: Compute difficulty-aware fusion scalar λ via Eq. 13
- 6: Width-Versatile Pathway (Conditional):
- 7: if λ > 0 then
- 8: Construct virtual experts by slicing shared weights Eq. 5 and Eq. 6
- 9: Compute Ywidth with Top-k routing via Eq. 7
- 10: else
- 11: Ywidth ← 0 {Prune branch for efficiency}
- 12: end if
- 13: Depth-Versatile Pathway (Discrete Early-Exit):
- 14: Initialize H(0) ← H
- 15: for ℓ = 1 to ℓˆ
- 16: Update recursive hidden state H(ℓ) using full shared weights via Eq. 9
- 17: end for
- 18: Ydepth ← H(ℓˆ) {Hard selection of final state}
- 19: Difficulty-Aware Fusion:
- 20: Compute output Y by fusing pathways via Eq. 3
- 21: Return Y

### B. Implementation Details

Experiments Setup. We primarily conduct experiments under a continued pre-training setting. Specifically, we initialize the model using the pre-trained checkpoint and continue training for one epoch on the original corpus. For all models, we set the training sequence length to 4,096 tokens. AdamW optimizer is employed with a weight decay of 0.1 and a gradient clipping threshold of 1.0. The learning rate follows a cosine decay schedule, beginning with a warm-up phase for the first 5% of total steps and subsequently decaying to 10% of the peak value. Regarding model-specific hyperparameters, we use a global batch size of 1.5M tokens with a peak learning rate of 5×10−4 for the 354M model, a global batch size of 1M tokens with a peak learning rate of 4 × 10−4 for the 720M model and a global batch size of 0.5M tokens with a peak learning rate of 3 × 10−4 for the 1.21B model. Furthermore, for our method and traditional MoE, we incorporate an auxiliary load balancing loss with a weighting coefficient of 1 × 10−5.

Evaluation Benchmark Setup. We evaluate our method on a comprehensive set of benchmarks: PIQA (Bisk et al., 2020), HellaSwag (Zellers et al., 2019), OpenBookQA (OBQA) (Mihaylov et al., 2018), SciQ (Welbl et al., 2017), ARC-easy (ARC-e) and ARC-challenge (ARC-c) (Clark et al., 2018), CommonsenseQA (COMM) (Clark et al., 2019), and Winogrande (WINO) (Sakaguchi et al., 2019). We utilize OLMES (OLMo et al., 2024) to assess the performance under zero-shot setting.

### C. Features of two branches

We analyze the features of the width-versatile and depth-versatile at Layer 0 using a random sample from the ARC-c, as visualized in Figure 6. Although both branches inherit the same base parameters, the output features of the two branches are not exact replicas. This demonstrates that they learn complementary representations within a globally aligned semantic space.

1.0

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

0.8

0.6

Dim

Dim

Dim

0.4

0.2

0.0

Seq Seq

Seq

Figure 6. Heatmap of cosine similarity at Layer 0. From left to right: 354M, 720M, 1.21B model.

### D. Evolution of Downstream Accuracy During Training

We illustrate the zero-shot accuracy trajectories across eight benchmarks for three distinct model scales equipped with VersatileFFN in Figure 7, Figure 8 and Figure 9. We observe a consistent positive correlation between compute budget and downstream performance. As the number of training tokens increases, accuracy generally trends upward across all tasks and model sizes. Comparing the three figures reveals clear scaling benefits: the 1.21B parameter model achieves higher absolute accuracy peaks and steeper learning curves compared to its smaller counterparts. Notably, while the global trend is monotonic, the learning curves exhibit high-frequency oscillations, particularly evident in the 720M and 1.21B models on tasks such as ARC-Easy and OpenBookQA.

[Figure 17]

- Figure 7. Accuracy of the 354M model on eight benchmarks as training progresses.

[Figure 18]

###### Figure 8. Accuracy of the 720M model on eight benchmarks as training progresses.

[Figure 19]

###### Figure 9. Accuracy of the 1.21B model on eight benchmarks as training progresses.

