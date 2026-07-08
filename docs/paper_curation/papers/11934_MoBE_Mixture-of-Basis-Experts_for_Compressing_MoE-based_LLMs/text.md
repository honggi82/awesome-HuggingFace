# arXiv:2508.05257v1[cs.LG]7Aug2025

[Figure 1]

[Figure 2]

## MoBE: Mixture-of-Basis-Experts for Compressing MoE-based LLMs

Xiaodong Chen2,1∗, Mingming Ha1, Zhenzhong Lan1,3, Jing Zhang2†, Jianguo Li1†

1Inclusion AI 2Renmin University of China 3Westlake University

##### Abstract

The Mixture-of-Experts (MoE) architecture has become a predominant paradigm for scaling large language models (LLMs). Despite offering strong performance and computational efficiency, large MoE-based LLMs like DeepSeek-V3-0324 and Kimi-K2-Instruct present serious challenges due to substantial memory requirements in deployment. While recent works have explored MoE compression to address this issue, existing methods often suffer from considerable accuracy drops (e.g., 7-14% relatively) even at modest compression rates. This paper introduces a novel Mixture-of-Basis-Experts (MoBE) method that achieves model compression while incurring minimal accuracy drops. Specifically, each up/gate matrix in an expert is decomposed via a rank decomposition as W = AB, where matrix A is unique to each expert. The relatively larger matrix B is further re-parameterized as a linear combination of basis matrices {Bi} shared across all experts within a given MoE layer. The factorization is learned by minimizing the reconstruction error relative to the original weight matrices. Experiments demonstrate that MoBE achieves notably lower accuracy drops compared to prior works. For instance, MoBE can reduce the parameter counts of Qwen3-235B-A22B-2507, DeepSeek-V3-0324 (671B) and Kimi-K2-Instruct (1T) by 24%-30% with only 1%-2% accuracy drop (about 2% drops when measured relatively).

##### 1 introduction

Transformer-based large language models (LLMs) (Vaswani et al., 2017) have revolutionized natural language processing, achieving state-of-the-art performance in domains such as creative writing, code generation, and mathematical reasoning. This progress has been largely guided by scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022), which posit that model performance improves with increases in parameter count and training data size. However, scaling dense architectures beyond a certain threshold—typically hundreds of billions of parameters (>100B)—has proven challenging and prohibitive. Therefore, the Mixture-of-Experts (MoE) (Jacobs et al., 1991; Jordan & Jacobs, 1994; Cai et al., 2024) architecture has become popular since the sparse activation makes MoEs much easier and more efficient to scale to more than several hundreds of billions of parameters (Liu et al., 2024; Yang et al., 2025; Team et al., 2025a) since last year.

Despite the computational advantages of sparse activation, the large total parameter counts of MoE-based LLMs present a significant bottleneck for practical deployment. For instance, leading open-source LLMs such as DeepSeek-V3-0324 (671B parameters) (Liu et al., 2024) exhibit performance comparable to top closed-source models. However, their scale imposes prohibitive demands on GPU memory; even high-end infrastructure, such as a machine with 8x H100 GPUs, may be insufficient for efficient inference.

To address this challenge, much research have been proposed for MoE-based LLM compression, which could be generally categorized into two major categories. First, pruning techniques reduce total parameter counts by either removing entire experts (Xie et al., 2024; Lu et al., 2024; Yang et al., 2024) or merging similar ones (hao

∗Work done at Ant Group. †Corresponding Authors. ‡https://github.com/inclusionAI/MoBE

MoBE

MoLAE

D²-MoE

| |
|---|

| |
|---|

| |
|---|

100

###### 98.2

###### 98.1

###### 97.8

###### 96.8

###### 96.3

95.8

93.0

RelativeAccuracy(%)

92.6

92.2

91.191.3

89.8

88.9

85.9

84.5

80

DeepSeek-V2-Lite-Chat Ling-Lite-Chat Qwen3-30B-A3B-2507 Qwen3-235B-A22B-2507 DeepSeek-V3-0324 Kimi-K2-Instruct

- Figure 1: Relative performance comparison of different MoE compression methods. Relative accuracy is the ratio of the compressed model’s performance to that of the original model. The accuracy are averaged over 15 benchmarks as shown in Table 3. Applying D2-MoE to large models like Qwen3-235B-A22B-2507, DeepSeekV3-0324 and Kimi-K2-Instruct is computationally prohibitive on an 8x H100 GPU machine; therefore, it is excluded from these comparisons. MoBE is evaluated at compression rates similar to or higher than the baseline methods (MoLAE, D2-MoE). Absolute performance is detailed in Appendix A (Figure 8).

Liu et al., 2024; Li et al., 2023b; Chen et al., 2024). However, this approach often leads to a permanent loss of specialized knowledge and significant performance degradation (Gu et al., 2025). Second, decomposition techniques employ matrix factorization to compress each expert’s weight matrices (Gu et al., 2025; Liu et al., 2025; Li et al., 2025). Typical works include D2-MoE (Gu et al., 2025), which extracts shared weights and applies singular value decomposition (SVD) to the residual delta weights, and MoLAE (Liu et al., 2025), which uses SVD to represent each expert weight as a product of its unique transformation matrix and a shared latent matrix. Although these SVD-based methods generally outperform expert pruning, they can still incur substantial information loss. This is evidenced by the high Mean Squared Error (MSE) between the original and reconstructed matrices, as shown in our reconstruction error analysis (Figures 2-4).

In this paper, we introduce the Mixture-of-Basis-Experts (MoBE), a novel method for efficient, performancepreserving parameter compression for MoE-based LLMs. MoBE factorizes weight matrix W in an expert with rank decomposition W = AB, where A is unique for each expert and B is re-parameterized as a linear combination of a set of basis matrices {Bi} that are shared across all experts within each MoE layer. This formulation achieves parameter reduction for two reasons. First, the number of basis matrices m is much smaller than the number of experts n, i.e. m ≪ n, and basis {Bi} is shared across all experts within each layer so that we could save considerable parameters for B. Second, the unique transformation matrix A is smaller than W, so that the whole MoBE factorization achieves parameter savings. The MoBE factorization is optimized by minimizing the reconstruction error between the factorized representation and the original pretrained weight matrices, typically using the gradient descent method.

We conduct comprehensive experiments on a diverse set of MoE-based LLMs, including Ling-Lite-Chat (Team et al., 2025b), DeepSeek-V2-Lite-Chat (Shao et al., 2024), DeepSeek-V3-0324 (Liu et al., 2024), Qwen3-30BA3B-2507, Qwen3-235B-A22B-2507 (Yang et al., 2025) and Kimi-K2-Instruct (Team et al., 2025a). A direct comparison of reconstruction error on Ling-Lite-Chat, DeepSeek-V2-Lite-Chat, and Qwen3-30B-A3B-2507 demonstrates that MoBE achieves a consistently lower MSE than both MoLAE and D2-MoE, often with reductions of over 50%, across all layers (Figures 2-4). Similar results for Qwen3-235B-A22B-2507, DeepSeekV3-0324 and Kimi-K2-Instruct are presented in Appendix C. To assess downstream task performance, we evaluate the compressed models on a wide range of benchmarks. As shown in Figure 1, MoBE exhibits a

- 0

- 1

- 2

- 3

- 0

- 1

- 2

- 3

MoBE

MoBE

MoLAE D²-MoE

MoLAE D²-MoE

MSELoss

MSELoss

0 4 8 12 16 20 24

0 4 8 12 16 20 24

Layer

Layer

(a) Gate matrices

(b) Up matrices

- Figure 2: Comparison of pre-layer MSE for compressing the gate (a) and up (b) matrices of Ling-Lite-Chat using MoBE, D2-MoE and MoLAE.

1 5 9 13 17 21 25

Layer

- 0

- 1

- 2

- 3

MSELoss

1e 4

MoBE

MoLAE D²-MoE

(a) Gate matrices

1 5 9 13 17 21 25

Layer

- 0

- 1

- 2

- 3

MSELoss

1e 4

MoBE

MoLAE D²-MoE

(b) Up matrices

- Figure 3: Comparison of pre-layer MSE for compressing the gate (a) and up (b) matrices of DeepSeek-V2Lite-Chat using MoBE, D2-MoE and MoLAE.

superior performance advance compared to MoLAE and D2-MoE at similar or even higher compression rates.

In summary, our contributions can be summarized as follows:

- • We introduce the Mixture-of-Basis-Experts (MoBE), a parameter-efficient architecture for MoE model compression. Our analysis shows that this design yields significantly lower reconstruction error compared to existing decomposition techniques.
- • We demonstrate through extensive experiments on leading MoE models, including Qwen3-235BA22B-2507, DeepSeek-V3-0324 and Kimi-K2-Instruct, that MoBE can reduce total parameter counts by 24%-30% while retaining up to 98% of the original performance, outperforming state-of-the-art MoE counterparts by a large margin.
- • We open-source our code to facilitate further research and development in efficient MoE architectures.

##### 2 Related Works

Research on MoE compression can be categorized into expert pruning-based (Xie et al., 2024; Lu et al., 2024; Yang et al., 2024) and decomposition-based (Li et al., 2025; Liu et al., 2025; Gu et al., 2025). Below we elaborate on related works under these two categories.

###### 2.1 Expert Pruning-based MoE Compression Methods

Expert pruning-based methods aim to reduce the total parameter counts of MoE-based LLMs by either directly removing entire experts or merging them. For instance, NAEE (Lu et al., 2024) removes unimportant experts by evaluating expert combinations on a calibration dataset to minimize model loss, while STUN (Lee

- et al., 2024) groups experts based on co-activation frequency and routing weight similarity, retaining only one expert per group.

2.0

1.5

1.5

MSELoss

MSELoss

MoBE

MoBE

1.0

MoLAE D²-MoE

MoLAE D²-MoE

1.0

0.5

0.5

0 4 8 12 16 20 24 28 32 36 40 44

0 4 8 12 16 20 24 28 32 36 40 44

Layer

Layer

(a) Gate matrices

(b) Up matrices

- Figure 4: Comparison of per-layer MSE loss for compressing the gate (a) and up (b) matrices of Qwen3-30BA3B-2507 using MoBE, D2-MoE and MoLAE.

Other approaches focus on merging similar experts. DEK (Zhang et al., 2024), for example, identifies and groups similar experts in the feature space and then merges them in the weight space to reduce redundancy. MC-SMoE (Li et al., 2023b) organizes experts into distinct groups according to routing strategies and merges each group into a single expert. Because these methods remove entire expert modules, they risk a permanent loss of specialized knowledge, often leading to notable accuracy degradation on certain tasks.

- 2.2 Expert Matrix Decomposition-based MoE Compression Methods In contrast to expert pruning, expert matrix decomposition-based methods compress MoE-based LLMs by factorizing each expert’s weight matrices into relatively smaller representations. D2-MoE (Gu et al., 2025) and MoLAE (Liu et al., 2025) are two state-of-the-art examples of this category. D2-MoE approximates each expert matrix with a shared matrix and a residual delta matrix, in which the shared weight is obtained via a Fisher-weighted average of the original weights, and the residual delta weights (the difference between original and shared weights) are decomposed into low-rank matrices using SVD.

MoLAE first groups a set of up/gate matrices in each MoE layer, and then approximates each matrix in a group by an expert-specific transformation matrix and the product of a group-shared latent matrix. The approximation is achieved using SVD on the stacked up/gate matrices within the group.

Although these methods are effective in reducing parameter counts, their reliance on low-rank assumptions can be a limitation. The resulting matrix factorization does not always capture the full information of the original weights, which can introduce substantial reconstruction errors and lead to notable performance drops in downstream tasks.

In Appendix B, we analyze the effective rank of expert weight matrices in several leading open-source MoE models. Our results show that this rank consistently exceeds the compression threshold of SVD—meaning that to achieve parameter reduction, the number of retained singular values must fall below this threshold. Eliminating this excess rank reduces the matrix’s expressive power, likely explaining the performance degradation observed in these SVD-based compression methods.

- 3 Methodology

In this section, we first briefly review the standard Mixture-of-Experts (MoE) architecture (Section 3.1). Then, we elaborate our proposed Mixture-of-Basis-Experts (MoBE) architecture and detail the algorithm for converting a pretrained MoE model to MoBE architecture (Section 3.2). Finally, we describe the activation functions in MoBE (Section 3.3) and a specific Z-score normalization technique applied to the expert weight matrices during the conversion process (Section 3.4).

###### 3.1 Standard Mixture-of-Experts Architecture

A standard MoE layer replaces the dense Feed-Forward Network (FFN) in the Transformer with a sparsely activated structure comprising a router and multiple experts. For each input token, the router dynamically selects a small subset of these experts for processing, which yields significant computation cost reduction. In a typical MoE layer with n experts, the i-th expert (Ei) often employs a SwiGLU formulation (Shazeer, 2020)

Output Hidden States

Output Hidden States

Widown

### ......

Expert 1 Expert 2 Expert n

Aiup Aigate

α1upBmup ... αmupBmup α1gateB1gate ... αmgateBmgate

Router

Input Hidden States

Input Hidden States

- Figure 5: The Mixture-of-Basis-Experts (MoBE) architecture. For clarity of explanation, we omit the activation function following the gate matrix.

to process an input token embedding x ∈ Rd as Ei(x) = Wdowni · (Wupi x ⊙ SiLU(Wgatei x)), (1)

where Wupi /gate ∈ Rp×d and Wdowni ∈ Rd×p denote the up, gate, and down projection matrices of Ei, p is the intermediate dimension of MoE experts. It is observed in most open-source MoE models that p < 12d. The router G calculates a gating score for each expert and selects the top-K experts for the token:

G(x) = TopK(Softmax(Wgx)) (2)

where Wg ∈ Rn×d denotes the weight matrix of the router G. The final output y of the MoE layer is a weighted sum of the outputs from the selected experts:

K

Gi(x)Ei(x), (3)

∑

y =

i=1

where Gi(x) denotes the gating value (i.e., the router score) of the i-th expert Ei. This operation is applied independently to every token in the input sequence.

###### 3.2 Mixture-of-Basis-Experts Architecture

While large MoE models are much more efficient in inference than dense models of a similar size, they are also constrained by higher memory and storage requirements during deployment. To alleviate this, we introduce the Mixture-of-Basis-Experts (MoBE) architecture, as illustrated in Figure 5. The MoBE formulation begins by factorizing the up/gate matrix Wi ∈ Rp×d of the i-th expert from the perspective of rank decomposition (Golub & Van Loan, 2013) as

Wi = AiBi,

where Ai ∈ Rp×r, Bi ∈ Rr×d, and r is the rank of Wi with r ≤ min{p, d} = p. MoBE further considers re-parameterizing Bi with a set of shared basis matrices as

m

αi,jBj,

Bi =

∑

j=1

m

with αi,j ≥ 0,

αi,j = 1,

∑

j=1

Algorithm 1 Converting standard MoE into MoBE

- 1: Require: L-layers model MMoE with n experts per layer; target basis count m ≪ n; activation function f.
- 2: Ensure: Parameter-efficient MoBE model MMoBE.
- 3: Initialize non-MoE parts in MMoBE with parameters directly from MMoE.
- 4: for each MoE layer l ≤ L in MMoE do
- 5: // Decompose up and gate projection matrices
- 6: for type t ∈ {gate, up} do
- 7: Let {Wti}in=1 be the expert matrices of the l-th layer
- 8: Solve Eq(5) with Adam optimizer
- 9: Obtain the factorized components {Ait}, {Btj}, {αit,j}
- 10: end for
- 11: // Keep down projection matrices unchanged
- 12: Copy the l-th layer down projection matrices {Wdowni }in=1 from MMoBE
- 13: Assemble the l-th MoBE layer with {At, Bt, αt} and {Wdown}.
- 14: end for
- 15: return MMoBE

where {Bj ∈ Rr×d}mj=1 is a set of basis matrices shared in one MoE layer, and {αi,j}mj=1 are learnable, expertspecific weighted coefficients. Combining these components and introducing a non-linear activation function f (e.g., SiLU (Ramachandran et al., 2018)) to enhance representational power, we define the final MoBE factorization as:

m

Wˆ i = Ai f(

αi,jBj), (4)

#### ∑

j=1

where Wˆ i is the reconstructed version of Wi.

This factorization allows the shared basis matrices {Bj} to capture common information across all experts in one layer, while the expert-specific transformation matrices Ai encode specialized information. We apply this factorization to both the gate and up projection matrices. However, we do not decompose the down projection matrices, as prior research indicates they store critical knowledge (Geva et al., 2020; Meng et al., 2022) and are less amenable to effective compression (Liu et al., 2025).

We convert a pretrained MoE-based LLM into our proposed MoBE formulation by learning the factorized components. This is achieved by minimizing the reconstruction error between the original expert weight matrix Wi and the reconstruction matrix Wˆ i as

n

n

m

2

2

Wi − Wˆ i

αi,jBj)

Wi − Ai f(

#### ∑

#### ∑

#### ∑

min

(5)

=

Ai,Bj,αi,j

i=1

i=1

j=1

This optimization problem can be solved using various algorithms, such as gradient-based optimizers like Adam (Kingma & Ba, 2014) or the Alternating Optimization (AO) method (Wu & Lange, 2008). In our practice, we find that the Adam optimizer performs sufficiently well across layers and various models, while AO suffers from unstable behavior during its alternating optimization steps. Algorithm 1 details the full procedure for converting a standard MoE model to the MoBE formulation.

We further analyze the parameter complexity of MoBE compared to standard MoE as illustrated in Table 1. Note that this analysis considers only the total and activation parameter count for a single MoE layer, excluding other components such as the embedding and attention layers. The total parameter counts for one MoBE layer is ndp + 2npr + 2mrd, where the first term is for the down matrices Wdown, the second term is for the transformation matrices A in the up and gate projection, and the third term is for the basis matrices {Bj}. The parameter count ratio (γ) from MoE to MoBE can be computed as

ndp + 2npr + 2mrd 3ndp

1 3

2r 3d

2mr 3np

.

γ =

=

+

+

Since r ≤ p < 12d, the second term 32dr < 31. For the last term, m ≪ n, for an MoE with n = 128 experts, even if we set m = 16, we could have the last term 23mrnp < 121 . Therefore, γ < 31 + 13 + 121 < 1. When using MoBE to

Standard MoE MoBE MoBE†

#Total Parameters 3ndp ndp + 2npr + 2mrd ndp + 2npr + 2mrd #Activation Parameters 3kdp kdp + 2kpr + 2krd k′dp + 2k′pr + 2k′rd

Table 1: Comparison of total and activation parameter count for one standard MoE and MoBE layer. MoBE† is a MoBE variant with further activation expert number reduction.

Ling-Lite-Chat DeepSeek-V2-Lite-Chat DeepSeek-V3-0324 Qwen3-30B-A3B-2507 Qwen3-235B-A22B-2507 Kimi-K2-Instruct Gate Matrices Mean 2.2e-5 1.0e-6 -4.2e-6 -2.8e-5 -1.4e-5 -1.3e-6

- Std 2.8e-2 2.9e-2 1.2e-2 2.3e-2 1.6e-2 2.6e-2

Up Matrices Mean 2.3e-7 -1.6e-7 -5.3e-9 5.3e-7 1.8e-8 4.2e-8

- Std 2.8e-2 3.0e-2 1.2e-2 2.3e-2 1.6e-2 2.6e-2

Table 2: Means and stds of the gate matrices and up matrices in various MoE-based LLMs.

replace MoE, the compression ratio by MoBE is 1 − γ. From the analysis, we can draw the conclusion that the MoBE architecture could substantially compress the standard MoE models.

Notably, while MoBE reduces the total parameters quite a lot, its activation parameter count requires closer examination. The matrices B and the down matrices Wdown contribute 2krd + kdp ≤ 3kdp (since r ≤ p) to the activation parameter count, while the transformation matrices A introduce an additional 2kpr. This may lead to an increase in the number of activation parameters. To compensate for this increase, inspired by previous work (Chaudhari et al., 2025), we propose a variant MoBE†, which reduces the number of activated experts during inference from k to a smaller value k′. In many modern MoE models, the number of activated experts k is typically set to 8. In MoBE†, we reduce this to 6 (i.e., k′ = 6).

###### 3.3 Activation Function in MoBE

In Eq(4), we employ an activation function f to enhance representational power. However, not all activation functions are equally suitable. For instance, we posit that the commonly used ReLU (Glorot et al., 2011) activation function is suboptimal for this task. ReLU can induce excessive sparsity in the matrix Bi = f(∑mj=1 wi,jBj), which may cause notable information loss. As the transformation matrix Ai ∈ Rp×r is smaller than Bi ∈ Rr×d, it may struggle to compensate for this loss with such a limited representation capacity. Therefore, a bipolar activation function (i.e., one that outputs both positive and negative values like tanh) is highly desirable.

Consequently, activation functions such as Tanh (LeCun et al., 1989), SiLU (Ramachandran et al., 2018), and GeLU (Hendrycks & Gimpel, 2016) are more suited for this task, while Sigmoid (Rumelhart et al., 1986) and ReLU are expected to yield inferior results. Our ablation study in Section 4.4 provide evidence supporting this hypothesis.

###### 3.4 Z-score Normalization in MoBE

To address the impact of a wide range of weight values and obtain stable results in seeking the basis, we consider normalizing all expert weight matrices in each MoE layer. We introduce a Z-score normalization by subtracting the mean and dividing by the standard deviation (std) across all experts’ weights:

µW = mean(W1,W2,...,Wn), (6) σW = std(W1,W2,...,Wn), (7)

Wi − µW σW

WZi =

. (8)

This normalization introduces additional inference overhead. After factorization, the σW term can be folded into the transformation matrix Ai, and the µW term will require an extra bias operation during inference

The method (Chaudhari et al., 2025) reduces only activation parameters, not total parameters. Therefore, we consider it a complementary approach and did not include it in our experimental comparisons.

compared to the original form Eq(4).

m

Wˆ i = σWWˆ Zi + µW = (σWAi)f(

αi,jBj) + µW. (9)

#### ∑

j=1

However, we empirically study different off-the-shelf MoE models and find that µW is typically negligibly small as shown in Table 2. We can therefore omit the term µW in Eq(9). That means, we only require absorbing σW into Ai without introducing extra parameters and computing overhead during inference.

##### 4 Experiments

In this section, we extensively evaluate the proposed MoBE approach on a suite of popular open-source MoE models and compare to state-of-the-art MoE compression methods (Section 4.3). We then conduct a set of ablation studies on activation functions (Section 4.4) and normalization schemes (Section 4.5).

###### 4.1 Setup

Models. We evaluate our method, MoBE, on a suite of popular open-source MoE-based LLMs: Ling-LiteChat (Team et al., 2025b), DeepSeek-V2-Lite-Chat (Shao et al., 2024), DeepSeek-V3-0324 (Liu et al., 2024), Qwen3-30B-A3B-2507, Qwen3-235B-A22B-2507 (Yang et al., 2025) and Kimi-K2-Instruct (Team et al., 2025a).

Baseline. We compare our approach against two state-of-the-art MoE compression baselines, D2-MoE (Gu et al., 2025) and MoLAE (Liu et al., 2025). Both MoBE and MoLAE are data-free compression methods, whereas D2-MoE requires a calibration dataset, for which we use tulu-v3-sft-mixture (Lambert et al., 2024). Due to the high computational cost of its backward pass, applying D2-MoE to very large models like Qwen3235B-A22B-2507, DeepSeek-V3-0324 and Kimi-K2-Instruct is infeasible on a single 8xH100 GPU machine. Therefore, comparisons involving D2-MoE are excluded from these three larger models.

Hyper-parameters. Hyper-parameters are configured per case (models or methods).

- • For Ling-Lite-Chat and DeepSeek-V2-Lite-Chat, MoBE uses m = 4 basis matrices, and MoLAE uses 8 latent matrices. To compensate extra computing cost introduced by extra activation parameters in MoBE (Section 3.2), we reduce its number of activated experts from k = 6 to k′ = 4 in MoBE†.
- • For Qwen3-30B-A3B-2507 and Qwen3-235B-A22B-2507, both MoBE and MoLAE use 32 basis/latent matrices. For MoBE, the number of activated experts is reduced from k = 8 to k′ = 6 in MoBE†.
- • For DeepSeek-V3-0324, both MoBE and MoLAE use 64 basis/latent matrices. The number of activated experts is reduced from k = 8 to k′ = 6 in MoBE†.
- • For Kimi-K2-Instruct, both MoBE and MoLAE use 128 basis/latent matrices. The number of activated experts is reduced from k = 8 to k′ = 6 in MoBE†. Because jointly optimizing all 384 expert matrices within a single layer of Kimi-K2-Instruct is challenging, we instead split the experts into two sequential groups and train each group with its own set of 64 basis matrices.
- • For D2-MoE, the rank of the delta weights is set to 700 for Ling-Lite-Chat and DeepSeek-V2-Lite-Chat, and 420 for Qwen3-30B-A3B-2507. We do not apply compression to the shared weights created by D2-MoE.
- • For simplicity, we set the rank r = p in all our studies. It gets more compression ratio when setting r < p while may increasing the accuracy drops.

Implementation Details. All experiments are run on H100 GPUs with the Adam optimizer (Loshchilov & Hutter, 2017) and a 0.07 learning rate. We will make the training and inference code open-sourced.

###### 4.2 Evaluation Benchmark

We perform a comprehensive evaluation of all compressed MoE-based LLMs across a wide spectrum of benchmark. The evaluation suite covers four primary domains:

- • General Knowledge: BBH (Srivastava et al., 2022), MMLU (Hendrycks et al., 2020), CEval (Huang et al., 2023), CMMLU (Li et al., 2023a).
- • General Reasoning: ARC-Challenge (Clark et al., 2018), IFEval (Zhou et al., 2023), GPQA (Rein et al., 2023).

General Reasoning General Knowledge Mathematics Coding

LLM Method Ratio

Avg

ARC-C IFEval GPQA BBH MMLU CEval CMMLU Math GSM8k AIME24 AIME25 MBPP HumanEval Multipl-E LCB

MoE 0% 89.2 81.5 33.0 58.7 72.6 65.4 70.6 72.6 88.1 8.3 10.0 77.3 81.2 65.0 21.6 59.7 D2-MoE 14% 82.4 78.3 31.2 51.3 64.5 56.5 56.0 64.9 85.7 8.3 10.0 70.3 72.6 50.2 14.4 53.1 MoLAE 12% 85.4 75.1 29.7 51.9 69.5 61.9 62.3 66.3 83.9 10.0 4.2 71.4 82.9 60.3 15.0 55.3

Ling-Lite-Chat

MoBE 16% 87.1 79.2 29.4 61.5 71.5 66.6 66.2 70.4 88.0 11.7 9.2 77.5 82.9 64.0 14.4 58.6 MoBE† 16% 85.8 79.2 29.9 53.8 70.3 64.3 66.9 69.1 83.6 11.7 12.5 77.3 82.6 62.4 17.4 57.8

MoE 0% 65.1 49.7 25.9 36.0 53.7 55.4 58.6 27.6 61.4 0 0 59.0 40.2 34.6 2.4 38.0 D2-MoE 13% 62.7 49.0 29.7 30.1 50.4 48.1 51.0 23.2 60.0 0 0 50.1 39.2 25.8 1.8 34.7 MoLAE 11% 65.8 43.9 26.9 34.0 53.0 47.9 52.8 18.6 59.2 0.8 0 46.6 41.5 26.9 1.8 34.6

DeepSeek-V2-Lite-Chat

MoBE 15% 67.5 46.0 30.3 33.9 53.7 53.0 56.3 23.5 58.6 0.8 0 51.5 43.3 31.7 3.6 36.9 MoBE† 15% 63.1 45.1 26.6 32.5 50.9 53.0 55.3 23.0 60.0 2.5 0 51.5 50.6 29.0 3.0 36.4

MoE 0% 95.6 86.6 56.8 85.4 87.6 88.2 86.6 93.3 96.4 59.4 51.3 86.4 93.1 70.6 41.5 78.6 D2-MoE 24% 93.1 83.5 45.2 69.9 83.3 71.2 68.6 86.1 93.0 38.3 29.1 79.5 84.0 44.0 26.9 66.4 MoLAE 24% 92.5 79.2 46.3 76.5 80.3 76.0 74.9 85.4 91.4 35.2 33.1 81.7 82.9 50.8 25.6 67.5

Qwen3-30B-A3B-2507

MoBE 24% 96.6 86.9 52.1 83.5 85.6 85.1 83.5 92.5 95.2 55.0 45.2 87.4 91.8 61.9 35.6 75.8 MoBE† 24% 95.9 85.1 51.0 83.3 86.0 85.9 83.9 92.6 96.1 54.0 45.6 85.3 92.2 61.2 38.0 75.7

MoE 0% 97.0 84.8 66.7 85.4 90.3 90.4 88.6 92.0 94.9 56.9 47.3 89.7 93.4 68.2 44.6 79.3 MoLAE 30% 97.3 83.2 54.0 82.9 87.3 84.4 83.2 87.6 95.5 38.5 29.6 87.4 89.5 61.0 34.4 73.1 MoBE 30% 98.0 84.5 63.6 85.2 89.5 87.8 87.2 90.3 93.7 52.3 40.6 89.9 93.6 73.1 40.9 78.0 MoBE† 30% 96.6 84.3 62.6 85.4 87.2 87.9 89.4 91.0 94.8 49.8 41.9 89.0 93.8 73.0 42.1 77.9

DeepSeek-V3-0324

MoE 0% 97.0 90.0 60.7 89.5 90.9 90.9 90.0 94.4 96.7 61.9 51.7 93.0 96.3 70.5 48.4 81.5 MoLAE 24% 95.6 85.5 66.2 87.5 88.9 87.3 86.9 90.5 95.5 54.2 44.6 70.7 81.0 30.0 33.5 73.2 MoBE 24% 96.3 89.9 58.6 89.0 90.4 90.6 89.7 94.2 96.3 64.8 54.8 89.2 93.8 71.9 43.7 80.9

Qwen3-235B-A22B-2507

- MoBE† 24% 95.6 88.7 58.1 88.8 90.3 90.4 89.6 93.6 96.0 62.9 50.8 87.4 93.1 65.5 45.2 79.7

Kimi-K2-Instruct

MoE 0% 95.9 90.8 77.4 88.8 90.8 92.4 89.9 95.7 96.7 64.8 50.2 90.9 95.4 66.7 50.3 82.4 MoLAE 24% 96.6 88.2 66.4 86.0 89.2 89.4 87.8 90.9 93.0 44.8 35.0 88.8 91.9 60.6 40.3 76.6 MoBE 24% 97.0 91.4 73.2 87.2 90.3 90.2 89.2 94.9 96.3 62.5 44.4 89.9 94.0 68.8 47.2 81.1

- MoBE† 24% 96.3 91.7 74.6 88.1 90.2 90.3 89.3 95.1 96.6 61.7 44.2 90.4 94.1 65.0 44.6 80.8

Table 3: Performance comparison of different compression methods on various MoE-based LLMs, where “†” indicates that this model activates fewer experts than the original model to compensate for the increase in activation parameters. The column Ratio refers to the proportion of compressed parameters to the total parameters in the LLMs.

- • Mathematics: Math (Hendrycks et al., 2021), GSM8k (Cobbe et al., 2021), AIME24, AIME25.
- • Coding: MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021), LCB(LiveCodeBench-v5) (Jain et al., 2024), MultiPL-E (Cassano et al., 2022).

The evaluation includes several specialized testing protocols:

- • AIME Evaluation: On AIME24 and AIME25, we run inference 16 times per question for each model and report the average accuracy.
- • IFEval Scoring: The final score for IFEval is the average of the strict accuracies at both the prompt and instruction levels.

###### 4.3 Main Results

All the compared results of the origin model (MoE) and different compression methods (MoBE, MoBE†, D2MoE, and MoLAE) are shown in Tables 3. It shows that our proposed MoBE method generally outperforms all the compared compression methods across various benchmarks. For instance, for the Ling-Lite-Chat and DeepSeek-V2-Lite-Chat models, MoBE improves performance by 2-3% accuracy over the baseline. The performance gains are even more notable for Qwen3-30B-A3B-2507, Qwen3-235B-A22B-2507, DeepSeek-V30324 and Kimi-K2-Instruct, reaching 4-8% accuracy advantages over compared compression methods.

We note that converting a standard MoE model into our MoBE architecture results in an average performance degradation of approximately 1.4% accuracy compared to the original MoE models. For comparison, a simpler variant MoBE† that only reduces the number of activated experts from k to k′, leads to a smaller degradation of around 0.5% accuracy. This comparison suggests that it is more challenging to compress the total parameters than activation parameters for an MoE model. As the sparsity ratio (#activated-parameters/#total-parameters) of recent MoE models becomes larger and larger so that the total parameter counts reach trillion-level (≥1T), it is more useful and practical to compression the total parameters.

###### 4.4 Ablation Study on Activation Functions

In Eq(4), we apply a non-linear activation function to the linear combination of basis matrices to enhance representational capacity. To determine the most suitable activation function, we conduct experiments on the gate matrices of the Qwen3-30B-A3B model. As shown in Figure 6, among those activation functions, Sigmoid demonstrates inferior performance to the case without activation (i.e., a purely linear combination of basis matrices) in terms of the reconstruction MSE, while ReLU has an order-of-magnitude higher MSE loss. This result is consistent with our analysis in Section 3.3. GELU, SiLU, and Tanh activations achieve similar results and outperform the case without activation significantly, while we finally choose SiLU and Tanh as

- 4

1e 4

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

| |
|---|

| |
|---|

0 10 20 30 40

Layer

- 0

5

- 1e 5

2

MSELoss

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

| |
|---|

| |
|---|

| |
|---|

none

silu

tanh gelu

sigmoid

relu

- Figure 6: Comparison of per-layer MSE loss for compressing the gate matrices of Qwen3-30B-A3B when using different activation functions.

1e 5

6

MSELoss

4

w/ Z-Score Normalization

2

w/o Z-Score Normalization

0 10 20 30 40

Layer

Figure 7: Comparison of per-layer MSE loss for compressing the gate matrices of Qwen3-30B-A3B with/without Z-score normalization.

our activation function as they offer a more favorable trade-off between performance and computational efficiency.

###### 4.5 Ablation Study on Z-score Normalization

To evaluate the impact of the Z-score normalization technique introduced in Section 3.4, we conduct an ablation study using the gate matrices of the Qwen3-30B-A3B model. All experiments are conducted under identical hyperparameter and optimization settings, varying only the application of normalization. The results, as shown in Figure 7, demonstrate a notable reduction in MSE loss when Z-score normalization is applied. We hypothesize that the normalization can rescale the weight values from wide and wild ranges to a normal distribution with a mean of 0 and a std of 1, so that the optimization becomes more stable and effective.

##### 5 Conclusion

In this paper, we propose the Mixture-of-Basis-Experts (MoBE), a parameter-efficient architecture designed to address the memory requirement challenges during deployment for large-scale MoE-based LLMs. MoBE effectively combines shared basis matrices with expert-specific transformation matrices in a rank decomposition manner to mitigate key limitations of prior work. Extensive experiments demonstrate that MoBE outperforms existing counterpart methods like MoLAE and D2-MoE with a large margin in preserving higher performance and a better model compression rate. MoBE can compress leading models such as Qwen3-235B-A22B-2507, DeepSeek-V3-0324 and Kimi-K2-Instruct by up to 24%-30% while retaining up to 98% of their original performance across diverse benchmarks. Such a practical and effective method may help enable large MoE models for more scalable and efficient applications.

Limitations: While our method performs well in compressing MoE models, it still causes a slight drop in accuracy compared to the original model. To fix this gap, one potential direction is to employ full network knowledge distillation (KD) between the original and our compressed models. This requires modifying existing training frameworks to support KD training for large LLMs. Another limitation is that MoBE requires multiple times calling of current optimized kernel fused-MoE to mimic the factorization, which is relatively inefficient. Hence, it requires implementing a specific mega-kernel for the whole factorization to unleash the power of the MoBE architecture. Future work will address these two limitations.

##### References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. ArXiv, abs/2108.07732, 2021. URL https://api.semanticscholar.org/CorpusID:237142385.

Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on mixture of experts. arXiv preprint arXiv:2407.06204, 2024.

Federico Cassano, John Gouwar, Daniel Nguyen, Sy Duy Nguyen, Luna Phipps-Costin, Donald Pinckney,

Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q. Feldman, Arjun Guha, Michael Greenberg, and Abhinav Jangda. Multipl-e: A scalable and extensible approach to benchmarking neural code generation. 2022. URL https://api.semanticscholar.org/CorpusID:254854172.

Marmik Chaudhari, Idhant Gulati, Nishkal Hundia, Pranav Karra, and Shivam Raval. Moe lens - an expert is all you need. In Sparsity in LLMs (SLLM): Deep Dive into Mixture of Experts, Quantization, Hardware, and Inference, 2025. URL https://openreview.net/forum?id=GS4WXncwSF.

I-Chun Chen, Hsu-Shen Liu, Wei-Fang Sun, Chen-Hao Chao, Yen-Chang Hsu, and Chun-Yi Lee. Retrainingfree merging of sparse moe via hierarchical clustering. 2024. URL https://api.semanticscholar.org/ CorpusID:273323490.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457, 2018. URL https://api.semanticscholar.org/CorpusID:3922816.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Mor Geva, R. Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. ArXiv, abs/2012.14913, 2020. URL https://api.semanticscholar.org/CorpusID:229923720.

Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In International Conference on Artificial Intelligence and Statistics, 2011. URL https://api.semanticscholar.org/CorpusID: 2239473.

Gene H Golub and Charles F Van Loan. Matrix computations. JHU press, 2013. Hao Gu, Wei Li, Lujun Li, Qi Zhu, Mark Lee, Shengjie Sun, Wei Xue, and Yi-Ting Guo. Delta decompression

for moe-based llms compression. ArXiv, abs/2502.17298, 2025. URL https://api.semanticscholar.org/ CorpusID:276575054.

En hao Liu, Junyi Zhu, Zinan Lin, Xuefei Ning, Matthew B. Blaschko, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang. Efficient expert pruning for sparse mixture-of-experts language models: Enhancing performance and reducing inference costs. ArXiv, abs/2407.00945, 2024. URL https://api.semanticscholar.org/CorpusID:270869609.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv: Learning, 2016. URL https://api.semanticscholar.org/CorpusID:125617073.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring massive multitask language understanding. ArXiv, abs/2009.03300, 2020. URL https://api.semanticscholar.org/CorpusID:221516475.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. ArXiv, abs/2103.03874, 2021. URL https://api.semanticscholar.org/CorpusID:232134851.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego De Las Casas, Lisa Anne Hendricks, Johannes Welbl, and Aidan Clark. Training compute-optimal large language models. 2022.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322, 2023.

Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando SolarLezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of

large language models for code. ArXiv, abs/2403.07974, 2024. URL https://api.semanticscholar.org/ CorpusID:268379413.

Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181–214, 1994.

Jared Kaplan, Sam Mccandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. 2020.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\” ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Yann LeCun, Bernhard E. Boser, John S. Denker, Donnie Henderson, Richard E. Howard, Wayne E. Hubbard, and Lawrence D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1:541–551, 1989. URL https://api.semanticscholar.org/CorpusID:41312633.

Jaeseong Lee, Seung won Hwang, Aurick Qiao, Daniel F. Campos, Zhewei Yao, and Yuxiong He. Stun: Structured-then-unstructured pruning for scalable moe pruning. ArXiv, abs/2409.06211, 2024. URL https://api.semanticscholar.org/CorpusID:272550518.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212,

- 2023a.

Pingzhi Li, Zhenyu (Allen) Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen. Merge, then compress: Demystify efficient smoe with hints from its routing policy. ArXiv, abs/2310.01334,

- 2023b. URL https://api.semanticscholar.org/CorpusID:263605809.

Wei Li, Lujun Li, You-Liang Huang, Mark G. Lee, Shengjie Sun, Wei Xue, and Yike Guo. Structured mixtureof-experts LLMs compression via singular value decomposition, 2025. URL https://openreview.net/ forum?id=ho7ZUS1z8A.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Zehua Liu, Han Wu, Ruifeng She, Xiaojin Fu, Xiongwei Han, Tao Zhong, and Mingxuan Yuan. Molae: Mixture of latent experts for parameter-efficient language models. 2025. URL https://api.semanticscholar.org/ CorpusID:277451683.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2017. URL https://api.semanticscholar.org/CorpusID:53592270.

Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. In Annual Meeting of the Association for Computational Linguistics, 2024. URL https://api.semanticscholar. org/CorpusID:267782440.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. In Neural Information Processing Systems, 2022. URL https://api.semanticscholar.org/CorpusID: 255825985.

Prajit Ramachandran, Barret Zoph, and Quoc V. Le. Searching for activation functions. ArXiv, abs/1710.05941,

2018. URL https://api.semanticscholar.org/CorpusID:10919244.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. ArXiv, abs/2311.12022, 2023. URL https://api.semanticscholar.org/CorpusID:265295009.

David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by backpropagating errors. Nature, 323:533–536, 1986. URL https://api.semanticscholar.org/CorpusID: 205001834.

Zhihong Shao, Damai Dai, Daya Guo, Bo Liu (Benjamin Liu), Zihan Wang, and Huajian Xin. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. ArXiv, abs/2405.04434, 2024. URL https://api.semanticscholar.org/CorpusID:269613809.

Noam M. Shazeer. Glu variants improve transformer. ArXiv, abs/2002.05202, 2020. URL https://api. semanticscholar.org/CorpusID:211096588.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adri`a Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, , et al. Kimi k2: Open agentic intelligence, 2025a. URL https://arxiv.org/abs/2507.20534.

Ling Team, Binwei Zeng, Chao Huang, Chao Zhang, Changxin Tian, Cong Chen, Dingnan Jin, Feng Yu, Feng Zhu, Feng Yuan, et al. Every flop counts: Scaling a 300b mixture-of-experts ling llm without premium gpus. arXiv preprint arXiv:2503.05139, 2025b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,

and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. Tong Tong Wu and Kenneth Lange. Coordinate descent algorithms for lasso penalized regression. 2008. Yanyue Xie, Zhi Zhang, Ding Zhou, Cong Xie, Ziang Song, Xin Liu, Yanzhi Wang, Xue Lin, and An Xu.

Moe-pruner: Pruning mixture-of-experts large language model using the hints from its router. ArXiv, abs/2410.12013, 2024. URL https://api.semanticscholar.org/CorpusID:273375561.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Yuanlin Duan, Wenqi Jia, Miao Yin, Yu Cheng, and Bo Yuan. Moe-i²: Compressing mixture of experts models through inter-expert pruning and intra-expert low-rank decomposition. ArXiv, abs/2411.01016, 2024. URL https://api.semanticscholar.org/CorpusID: 273811289.

Zeliang Zhang, Xiaodong Liu, Hao Cheng, Chenliang Xu, and Jianfeng Gao. Diversifying the expert knowledge for task-agnostic pruning in sparse mixture-of-experts. ArXiv, abs/2407.09590, 2024. URL https://api.semanticscholar.org/CorpusID:271212712.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. ArXiv, abs/2311.07911, 2023. URL https://api.semanticscholar.org/CorpusID:265157752.

Appendix

##### A Absolute performance comparison of MoE compression methods

MoE

MoBE

MoLAE

D²-MoE

| |
|---|

| |
|---|

| |
|---|

| |
|---|

100

82.4 80.8

81.5

79.7

79.3 77.9

78.6

80

76.6

75.7

73.2

73.1

67.5 66.4

Accuracy(%)

59.7

60

57.8

55.3

53.1

40

38.0 36.4

34.6 34.7

20

0

DeepSeek-V2-Lite-Chat Ling-Lite-Chat Qwen3-30B-A3B-2507 Qwen3-235B-A22B-2507 DeepSeek-V3-0324 Kimi-K2-Instruct

- Figure 8: Absolute performance comparison of different MoE compression methods. The accuracy are averaged over 15 benchmarks as shown in Table 3. Applying D2-MoE to large models like Qwen3-235BA22B-2507, DeepSeek-V3-0324 and Kimi-K2-Instruct is computationally prohibitive on an 8x H100 GPU machine; therefore, it is excluded from these comparisons. MoBE is evaluated at compression rates similar to or higher than the baseline methods (MoLAE, D2-MoE). We present the absolute performance comparison of MoE compression methods in Figure 8.

##### B Analysis of the Effective Rank of Expert Weight Matrices

We evaluate the effective rank of expert weight matrices in Qwen3-235B-A22B-2507, DeepSeek-V3-0324, and Kimi-K2-Instruct. The effective rank re is defined as:

∑ik=1 σi2 ∑ri=1 σi2

re = min k ∈ N+

> 0.95

where σi is the i-th largest singular value (sorted in descending order) and r is the matrix rank. The expert weight matrices in Qwen3-235B-A22B-2507 have dimensions 4096×1536, while those in DeepSeek-V3-0324

and Kimi-K2-Instruct are 7168×2048. Figures 9–11 illustrate the per-layer average effective rank re and its range for each model. Taking the expert weight matrices of Kimi-K2-Instruct as an example, rank decomposition could realize parameter compression only if the intermediate rank satisfies

7168 · 2048 7168 + 2048 ≈ 1593.

rt ≤

However, according to Figure 11, the average effective rank re is larger than 1593 in most layers. This discrepancy implies that the pure rank-decomposition-based method can’t produce model compression without performance loss. An interesting finding can be drawn from the analysis: Qwen3-235B-A22B-2507 shows much broader effective rank range than the other two, which may indicate that its experts are far from being well-balanced during the training phase.

##### C Additional MSE Comparisons

We present a comparison of reconstruction errors on Qwen3-235B-A22B-2507, DeepSeek-V3-0324 and KimiK2-Instruct in the Figure 12-14.

Mean Rank

Rank Range (Min-Max)

| |
|---|

1200

ExpertMatrixRank

1000

800

600

400

200

0

0 20 40 60 80

Layer

(a) Gate matrices

Mean Rank

Rank Range (Min-Max)

| |
|---|

1200

ExpertMatrixRank

1000

800

600

400

200

0

0 20 40 60 80

Layer

(b) Up matrices

Mean Rank

Rank Range (Min-Max)

| |
|---|

1200

ExpertMatrixRank

1000

800

600

400

200

0

0 20 40 60 80

Layer

(c) Down matrices

- Figure 9: Average effective rank and effective rank range of the (a) gate, (b) up, and (c) down matrices at each layer in Qwen3-235B-A22B-2507.

10 20 30 40 50 60

Layer

0

400

800

1200

1600

ExpertMatrixRank

Mean Rank

| |
|---|

Rank Range (Min-Max)

(a) Gate matrices

10 20 30 40 50 60

Layer

0

400

800

1200

1600

ExpertMatrixRank

Mean Rank

| |
|---|

Rank Range (Min-Max)

(b) Up matrices

10 20 30 40 50 60

Layer

0

400

800

1200

1600

ExpertMatrixRank

Mean Rank

| |
|---|

Rank Range (Min-Max)

(c) Down matrices

- Figure 10: Average effective rank and effective rank range of the (a) gate, (b) up, and (c) down matrices at each layer in DeepSeek-V3-0324.

0 10 20 30 40 50 60

Layer

0

400

800

1200

1600

ExpertMatrixRank

Mean Rank

| |
|---|

Rank Range (Min-Max)

(a) Gate matrices

0 10 20 30 40 50 60

Layer

0

400

800

1200

1600

ExpertMatrixRank

Mean Rank

| |
|---|

Rank Range (Min-Max)

(b) Up matrices

0 10 20 30 40 50 60

Layer

0

400

800

1200

1600

ExpertMatrixRank

Mean Rank

| |
|---|

Rank Range (Min-Max)

(c) Down matrices

- Figure 11: Average effective rank and effective rank range of the (a) gate, (b) up, and (c) down matrices at each layer in Kimi-K2-Instruct.

1e 5

MoBE

7.5

MoLAE

MSELoss

5.0

2.5

0.0

0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90

Layer

(a) Gate matrices

1e 4

1.0

MoBE

MoLAE

MSELoss

0.5

0.0

0 5 1015202530354045505560657075808590

Layer

(b) Up matrices

- Figure 12: Comparison of pre-layer MSE for compressing the gate (a) and up (b) matrices of Qwen3-235BA22B-2507 using MoBE, D2-MoE and MoLAE.

3 8 13 18 23 28 33 38 43 48 53 58

Layer

0.0

2.5

5.0

7.5

MSELoss

1e 5

MoBE

MoLAE

(a) Gate matrices

3 8 13 18 23 28 33 38 43 48 53 58

Layer

0.0

2.5

5.0

7.5

MSELoss

1e 5

MoBE

MoLAE

(b) Up matrices

- Figure 13: Comparison of pre-layer MSE for compressing the gate (a) and up (b) matrices of DeepSeek-V30324 using MoBE, D2-MoE and MoLAE.

1 6 11 16 21 26 31 36 41 46 51 56

Layer

- 0

- 1

- 2

MSELoss

1e 4

MoBE

MoLAE

(a) Gate matrices

1 6 11 16 21 26 31 36 41 46 51 56

Layer

- 0

- 1

- 2

MSELoss

1e 4

MoBE

MoLAE

(b) Up matrices

- Figure 14: Comparison of pre-layer MSE for compressing the gate (a) and up (b) matrices of Kimi-K2-Instruct using MoBE, D2-MoE and MoLAE.

