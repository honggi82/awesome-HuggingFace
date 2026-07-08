[Figure 1]

## Frac-Connections: Fractional Extension of Hyper-Connections

#### Defa Zhu1,†, Hongzhi Huang1, Jundong Zhou1, Zihao Huang1, Yutao Zeng1, Banggu Wu1, Qiyang Min1,†, Xun Zhou1

# arXiv:2503.14125v1[cs.LG]18Mar2025

1ByteDance Seed †Corresponding authors

#### Abstract

Residual connections are central to modern deep learning architectures, enabling the training of very deep networks by mitigating gradient vanishing. Hyper-Connections recently generalized residual connections by introducing multiple connection strengths at different depths, thereby addressing the seesaw effect between gradient vanishing and representation collapse. However, Hyper-Connections increase memory access costs by expanding the width of hidden states. In this paper, we propose Frac-Connections, a novel approach that divides hidden states into multiple parts rather than expanding their width. Frac-Connections retain partial benefits of Hyper-Connections while reducing memory consumption. To validate their effectiveness, we conduct large-scale experiments on language tasks, with the largest being a 7B MoE model trained on up to 3T tokens, demonstrating that Frac-Connections significantly outperform residual connections.

Date: March 19, 2025 Correspondence: Defa Zhu at zhudefa@bytedance.com, Qiyang Min at qiyangming@bytedance.com

#### 1 Introduction

|| | | | | | | | |
|---|---|---|---|---|---|---|---|
|Frac-Connections| | | |Hyper-Connections| | | |
| | | | | | | | |
<br><br>1<br><br>|Expansion rate|
|---|
<br><br>1/8 1/4 1/2 2 4 8|
|---|

Figure 1 Comparison of Frac-Connections and Hyper-Connections based on their expansion rates. Frac-Connections correspond to n ≤ 1, while Hyper-Connections are defined by n ≥ 1. The two connection types become identical when the expansion rate is n = 1.

Residual connections [7] have revolutionized deep learning by facilitating the effective training of very deep networks. These connections mitigate gradient vanishing and are fundamental to architectures such as transformers and convolutional neural networks (CNNs). However, residual connections suffer from a tradeoff between gradient vanishing and representation collapse, where the features of adjacent layers become excessively similar, particularly in very deep models [14, 25, 28].

Zhu et al. [28] introduce Hyper-Connections, an expansion of the dimension of hidden state and learnable depth and width connections, to address this issue. While effective, Hyper-Connections increase memory access by expanding the hidden states’ width. This raises the question: Can we enjoy the benefits of Hyper-Connections without increasing memory access?

1.0

0.8

0.6

0.4

i+1icos(h,h)00

0.2

Baseline

Hyper-Connections

0.0

Frac-Connections

0.2

0 5 10 15 20 25 30

Layer Index i

- Figure 2 Cosine similarity between the input of the current and the previous layers for the OLMoE-7B models. The curve represents the median of similarity, while the shaded area indicates the range between the 5th and 95th percentiles.

To this end, we propose Frac-Connections (FC), a novel method that partitions the hidden states into multiple fractions rather than duplicating them and increasing their width. This approach extends the expansion rate n of Hyper-Connections (HC) to the fractional domain. In particular, when n = 1, Frac-Connections and Hyper-Connections are equivalent, as illustrated in Fig. 1. This reduces memory usage while preserving the ability to model multiple connection strengths. As shown in Fig.2, the similarity between adjacent hidden states in FC lies between that of HC and baseline (Pre-Norm), indicating that their representational capacity follows the order: HC > FC > Pre-Norm.

To further validate the effectiveness of Frac-Connections, we conduct extensive experiments on large language models (LLMs), including both dense and Mixture-of-Experts (MoE) [21] architectures. Our results demonstrate that Frac-Connections significantly improve training stability and enhance downstream task performance across a wide range of natural language processing benchmarks. We believe that the simplicity, scalability, and efficiency of Frac-Connections will enable their widespread adoption across various domains in machine learning, providing a robust foundation for building the next generation of dense and sparse deep learning models.

#### 2 Related Work

Transformers [6, 9, 10, 23, 24] have revolutionized deep learning, particularly in natural language processing and computer vision. They rely on self-attention mechanisms to capture long-range dependencies and have become the foundation of large-scale models such as BERT [5] and GPT [2]. A key component of Transformers is residual connections [7], which aid training but may also limit model expressiveness [28]. Our work focuses on replacing these residual connections to further enhance Transformer performance.

Residual Connections and Their Limitations. Residual connections [7] have been a key component in modern deep networks, enabling the training of very deep architectures by mitigating the gradient vanishing problem.

||Attention|
|---|
<br><br>+ +<br><br>+<br><br>(b) Transformer with Hyper-Connections<br><br>|ℎ|
|---|
<br><br>Repeat<br><br>|Attention|
|---|
<br><br>|FFN|
|---|
<br><br>+<br><br>+<br><br>(a) Transformer with Residual Connections<br><br>|FFN|
|---|
<br><br>+ +<br><br>+<br><br>+<br><br>𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>𝛼 ,  𝛼 ,  𝛼 ,  𝛼 ,  𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>𝛼 ,  𝛼 ,  𝛼 ,  𝛼 ,  𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>|h|
|---|
<br><br>|h|
|---|
<br><br>|ℎ|
|---|
<br><br>|ℎ|
|---|
<br><br>|ℎ|
|---|
<br><br>(c) Transformer with Frac-Connections<br><br>|ℎ|
|---|
<br><br>Split<br><br>|h|
|---|
<br><br>|h|
|---|
<br><br>|h|
|---|
<br><br>|h|
|---|
<br><br>|Attention|
|---|
<br><br>+ +<br><br>+<br><br>𝛽<br><br>𝛼 ,  𝛼 ,  𝛼 ,  𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>|h|
|---|
<br><br>𝛾 ,  𝛾 ,  𝛾 ,  𝛾 , <br><br>+<br><br>Cat<br><br>|FFN|
|---|
<br><br>+ +<br><br>+<br><br>𝛽<br><br>𝛼 ,  𝛼 ,  𝛼 ,  𝛼 , <br><br>𝛽<br><br>𝛾 ,  𝛾 ,  𝛾 ,  𝛾 , <br><br>+<br><br>Cat<br><br>|ℎ|
|---|
<br><br>Cat<br><br>|: scalar, multiplied with the hidden vector : hidden vectors<br><br>| |
|---|
|
|---|
|
|---|

- Figure 3 Figure 2. Frac-connections (FC) with an expansion rate of n = 1/2. (a) Residual connections. (b)

Hyper-connections: β1, β2, α0,0, α0,1, α1,0, α1,1, α2,1, and α2,2 are learnable scalars or scalars predicted by the network, depending on the specific HC version. (c) Frac-connections: Frac-connections split the hidden representations into smaller fractions and process each fraction independently. The scalars γ1,2, γ2,1, and γ2,2 are either learnable or predicted by the network, similar to hyper-connections. These fractions are concatenated (denoted as Cat) after processing, followed by integration into the main network pipeline.

They are widely used in networks such as CNNs[12] and Transformers [23]. However, despite their effectiveness, residual connections introduce a fundamental trade-off between gradient propagation and representation collapse [28], which can degrade performance in extremely deep models. ResiDual [25] addresses this issue by adopting a dual-stream design with parallel PreNorm and PostNorm structures, while Hyper-Connections use a weighted multi-stream design to significantly improve performance. While this improves performance, the multi-stream approach increases memory consumption. Our Frac-Connections build upon this design by reducing the hidden size of each stream, retaining the benefits of Hyper-Connections without increasing memory usage.

Fractal Design. FractalNet [13] proposes partitioning the hidden states into multiple segments, each processed by networks of varying depths, enabling the training of extremely deep neural networks. Frac-Connections share a similar design principle; however, instead of assigning each partition to a different depth, we associate them with different connection weights.

#### 3 Preliminaries

Hyper-Connections (HC) enhance the representation of hidden states in neural networks by introducing a hyper hidden matrix. Given the initial input h0 ∈ Rd, it is replicated n times to construct the initial hyper hidden matrix:

H0 = h0 h0 ... h0 ⊺ ∈ Rn×d, (1)

where n is the expansion rate. At the k-th layer, the input is the hyper hidden matrix from the previous layer, denoted as Hk−1:

Hk−1 = hk1−1 hk2−1 ... hkn−1 ⊺ ∈ Rn×d. (2)

The final hidden vectors are aggregated using sum pooling, which reduces the hyper hidden matrix back to a single vector.

The hyper-connections are modeled by a matrix HC ∈ R(n+1)×(n+1), which defines the connection weights across different components:

HCk =

01×1 Bk Amk Ark

, (3)

where Bk, Amk, and Ark are submatrices that define the connections within and between layers. For a given network layer T k, which integrates components such as self-attention and feed-forward networks, the output Hk+1 of the hyper-connections can be expressed as:

Hk = Bk⊺(T k(hk0−1))⊺ + Ark⊺Hk−1, (4) where hk0 is computed as the weighted sum of the hyper hidden matrix using Amk:

h0k−1⊺ = Am⊺Hk−1. (5)

These matrices capture the relationships across both the depth and width dimensions of the network and are visualized in Fig. 3.

To further improve flexibility of the connections, Dynamic Hyper-Connections (DHC) extend this framework by making the weights input-dependent. Instead of using fixed parameters, the connection weights are dynamically predicted based on the input hidden vector Hk. This adaptive mechanism improves its ability to represent complex relationships. The advantages of DHC are particularly evident in tasks such as language modeling.

#### 4 Method

##### 4.1 Overview of Frac-Connections

The purpose of introducing frac-connections is to address the seesaw problem in residual connections while retaining the flexibility of constructing connection strengths, without incurring the additional memory overhead of splitting hidden states into n parts as in hyper-connections. This is achieved by generalizing the expansion rate to fractional values. When n = 1, frac-connections are equivalent to hyper-connections. For 0 < n < 1, frac-connections can be viewed as a fractional variant of hyper-connections that divides the hidden states into m = 1/n parts instead of replicating them n times, where m (referred to as the frac-rate) represents the number of partitions.

Let h ∈ Rd represent the hidden state of a layer. Instead of replicating h into n copies as in hyper-connections, frac-connections split h into m = 1/n parts:

H = h1 h2 ... hm ⊺ = Reshape(h,(m,d/m)), (6)

where hi ∈ Rd/m for i = 1,2,...,m. The Frac-Connections (FC) can be represented by a matrix FC, where each element defines the connection weight. The matrix is structured as follows:

01×m B Y A ∈ R(m+1)×(2×m)

FC =





0 ··· 0 β1 ··· βm

- γ1,1 ··· γ1,m α1,1 ··· α1,m
- γ2,1 ··· γ2,m α2,1 ··· α2,m

. (7)

=

 

 

... . γm,1 ··· γm,m αm,1 ··· αm,m

... . .

.

Consider the k-th network layer T k, it integrates self-attention layers or feed-forward networks within transformers. The output of the FC, denoted by Hk, can be simply formulated as follows:

Hk = FCk(T k,Hk−1)

= Bk⊺T k Yk⊺Hk−1 + Ak⊺Hk−1. (8)

##### 4.2 Dynamic and Static Frac-Connections

Since Frac-Connections is the fractional variant of Hyper-Connections, Frac-Connections can be implemented in two forms likewise:

- 1. Static Frac-Connections: The weights are learnable, but static during testing.
- 2. Dynamic Frac-Connections: The weights are dynamically computed based on the input, allowing greater flexibility.

The matrix representation of dynamic frac-connections (DFC) is defined as follows:

01×m B(H) Y(H) A(H)

(9)

FC(H) =

Similarly, given a layer T and input H, we obtain the output of the DFC as follows:

Hˆ = FC(H)(T ,H). (10)

In practice, we follow that of DHC [28], combining the dynamic and static matrices to achieve DFC. The dynamic parameters are obtained through a linear transformation. To stabilize the training process, we introduce normalization before the linear transformation and apply the tanh activation function after it, scaling it by a small initial learnable factor. The following equations detail how these dynamic parameters are computed:

H = norm(H) (11) B(H) = sβ ◦ tanh(HWβ)⊺ + B ∈ R1×m (12) Y(H) = sα ◦ tanh(HWγ) + Y ∈ Rm×m (13) A(H) = sα ◦ tanh(HWα) + A ∈ Rm×m (14)

##### 4.3 Initialization and Implementation

In order to make the initialization of the frac-connections equivalent to the Pre-Norm residual connections, we adopt the following initialization strategy. The dynamic parameters Wβ, Wγ, and Wα in Eqs. 12, 13, and 14 are initialized to 0, while the static matrices are initialized as follows:

###### 01×1 B Y A

=

01×1 11×m em×m em×m

. (15)

The static components B, Y, and A in Eqs. 7, 14, 12, 13 do not utilize weight decay, whereas the dynamic component does.

Frac-Connections for transformer is illuminated in Algorithm 1 and Pytorch-style pseudocode is shown in Algorithm 2, 3.

- Algorithm 1 Frac-Connections for Transformers

Require: Initial hidden vector h0 ∈ Rd Require: Fraction rate m Ensure: Final output y

- 1: Initialize:
- 2: H0 ← Reshape h0,(m,d/m) ⊺ ∈ Rm×(d/m)
- 3: for k = 1 to L do
- 4: h0k−1 ← Reshape(Yk⊺Hk−1,(d,))
- 5: Hk ← Bk⊺Reshape T k(h0k−1),(m,d/m) + Ak⊺Hk−1
- 6: end for
- 7: hL ← Reshape HL,(m,d/m)
- 8: y ← Umembedding Norm(hL)
- 9: return y

##### 4.4 Parameters and Computation

Static Frac-Connections. All learnable parameters are included in the frac-connection matrix FC in Eq. 7. The number of parameters in one FC is given by:

|θSHC| = |θB| + |θY| + |θA| = m + m · m + m · m = m · (2m + 1). (16) Thus, the number of extra parameters is:

Pextra = |θSHC| × 2 × L, (17) where L is the number of layers. For example, in OLMo-1B-7B-SFC×4, Pextra = 1152. Dynamic Frac-Connections. The parameters of DHC are defined in Eqs. 11, 12, 13, and 14, and the number of parameters is given by:

α| + |θA| (18) = |θnorm| + dmodel/m × (2m + 1) + m · (2m + 1) + 2, (19)

|θDFC| = |θnorm| + |sβ| + |θW

β| + |θB| + |sα| + |θW

γ| + |θY| + |θW

where dmodel is the dimension of the hidden states in the transformer, and |θnorm| depends on the type of normalization module. For RMSNorm [27], |θnorm| = dmodel/m. Similar to the static hyper-connections, the number of extra parameters is:

Pextra = |θDFC| × 2 × L, (20)

For example, for OLMo-1B-7B-DFC×4, Pextra == 165,056. The number of parameters for DFC used in the experiments is detailed in Table 1.

Computational Analysis. The primary computational cost of both SFC and DFC occurs in line 5 of Algorithm 1, with a complexity of O(dmodel × 4m). For comparison, the computational cost of the Feed-Forward Network (FFN) is O(2 × dmodel × dffn), while the projection component of attention requires O(4 × dmodel × dmodel) operations.

Since O(dmodel × 4m) ≪ O(4 × dmodel × dmodel) < O(2 × dmodel × dffn), the computational overhead of FC implementations is negligible compared to the costs of both the FFN and attention projection operations. Here, dffn represents the inner dimension of the FFN. Our analysis confirms that regardless of whether SFC or DFC is implemented, both the additional parameters and computational overhead introduced remain minimal and can be considered negligible in the overall system performance. Detailed computational cost statistics of DFC are presented in Table 2.

- Table 1 Comparison of number of parameters.

Method FC Params(B) Total Params(B) Total Params ∆ rate (%)

OLMo-1B2 - 1.17676442 OLMo-1B2-DFC×4 0.000165 1.17715846 +0.014%

OLMoE-1B-7B - 6.91909427 OLMoE-1B-7B-DFC×4 0.000165 6.91948832 +0.0024%

- Table 2 FLOPs per token in forward pass.

Method FC FLOPs (G) Total FLOPs (G) Total FLOPs ∆ rate (%)

OLMo-1B - 2.5587 OLMo-1B-DFC×4 0.0013 2.5598 +0.044%

OLMoE-1B-7B - 2.3580 OLMoE-1B-7B-DFC×4 0.0013 2.3629 +0.056%

#### 5 Experiments

We evaluate Frac-Connections on the pre-training of large language models, including sparse and dense models. Specifically, for sparse models we study Sparse Mixture-of-Experts (MoE) models [21] and follow the experimental setup described by OLMoE [16], conducting ablation studies on OLMoE-1.3B, which has 1.3B total parameters with 260M activated parameters. We further validate the effectiveness of our approach on a larger sparse model, OLMoE-7B, which has 7B total parameters with 1.3B activated parameters. For dense models, we follow the OLMo2 [17] training setup to pre-train a 1B2 parameter model. Importantly, all experiments were conducted without hyperparameter tuning, and the training hyperparameters were strictly aligned across comparative baselines. Through these experiments across different model scales and architectures, we aim to comprehensively demonstrate the applicability and benefits of our proposed Frac-Connections approach.

##### 5.1 Ablation Study

###### Ablation Frac-rate m.

###### SFC v.s. DFC.

###### Ablation on DFC.

0.006

2.675

2.675

OLMoE-1.3B

OLMoE-1.3B

OLMoE-1.3B-DFCx2 OLMoE-1.3B-DFCx4

OLMoE-1.3B-SFCx4 OLMoE-1.3B-DFCx4

0.005

TrainingLossDiff

2.650

2.650

TrainingLoss

TrainingLoss

0.004

OLMoE-1.3B-DFCx2

2.625

2.625

W/O Norm

0.003

W/O Tanh

2.600

2.600

0.002

W/O Rescale

2.575

2.575

0.001

2.550

2.550

x1.4 0.014

0.000

100 200 300 400 500

100 200 300 400 500

100 200 300 400

Tokens (Billions)

Tokens (Billions)

Tokens (Billions)

Figure 4 Training loss (0.999 EMA smoothed) loss for OLMoE-1.3B models.

We conduct extensive ablation studies on the OLMoE-1.3B model to evaluate different configurations of Frac-Connections, as shown in Figure 4.

Effect of different frac-rates. The leftmost of Figure 4 compares the baseline model against versions with Dynamic Frac-Connections (DFC) at different frac-rates (DFC×2 and DFC×4). The results show that DFC×2 demonstrates significant improvement over the baseline, while DFC×4 offers only marginal additional gains compared to DFC×2. The OLMoE-1.3B-DFC×4 model exhibits a training loss reduction of approximately 0.014 compared to the baseline.

Static Frac-Connections (SFC) v.s. Dynamic Frac-Connections (DFC) . In the middle of Figure 4, both -SFC×4 and -DFC×4 outperform the baseline. Additionally, -DFC×4 achieves better results than -SFC×4, suggesting that the dynamic parameter prediction mechanism provides additional modeling capacity.

Ablation study on the components of DFC. The rightmost of Figure 4 evaluates the impact of normalization, tanh activation, and rescaling by measuring their loss differences relative to the -DFC×2 baseline. From the training loss perspective, removing rescaling (purple line, without sβ and sα in Eq. 12, 13, 14) causes the most severe performance degradation, followed by the removal of tanh activation (green line), while the absence of normalization (blue line) results in the least detrimental effect, though still negatively impacting performance. These findings demonstrate the hierarchical importance of each component in the DFC implementation, with rescaling being particularly crucial for maintaining optimal training dynamics. Given that the original DHC design components exhibit either substantial or modest improvements when implemented in DFC, we opt to preserve the complete original DHC architecture to maintain optimal performance characteristics.

These findings underscore the effectiveness of Frac-Connections as a lightweight yet impactful enhancement for transformer-based models, offering improved performance with minimal parameter overhead.

##### 5.2 MoE Models

###### Training Loss

###### HellaSwag Acc.

###### C4-en Loss

###### SciQ Acc.

75.0

2.45

OLMoE-7B

OLMoE-7B

2.75

72.5

94

OLMoE-7B-DFCx2 OLMoE-7B-DFCx4 OLMoE-7B-DHCx4

OLMoE-7B-DFCx4

Accuracy(%)

Accuracy(%)

2.40

2.70

2.35

70.0

Loss

Loss

92

2.30

2.65

67.5

2.25

OLMoE-7B

OLMoE-7B

2.60

65.0

90

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

2.20

0.012

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (Billions)

Tokens (Billions)

Tokens (Billions)

Tokens (Billions)

###### Commonsense QA Acc.

###### SciQ Acc.

###### Social Iqa Acc.

###### WinoGrande Acc.

54

70.0

- 45

- 46

- 47

- 48

- 49

52

94

67.5

Accuracy(%)

Accuracy(%)

Accuracy(%)

Accuracy(%)

50

65.0

48

92

62.5

46

60.0

OLMoE-7B

OLMoE-7B

OLMoE-7B

OLMoE-7B

44

90

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

57.5

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (Billions)

Tokens (Billions)

Tokens (Billions)

Tokens (Billions)

- Figure 5 Training and evaluation performance of OLMoE-7B models. The plots show the training loss, C4-en loss, and accuracy on HellaSwag, SciQ, Commonsense QA, Social IQA, and WinoGrande over the course of training. The results are EMA-smoothed for clarity. The OLMoE-7B-DFCx4 variant demonstrates improved loss reduction and higher accuracy across multiple benchmarks compared to the baseline OLMoE-7B model, indicating enhanced optimization efficiency and generalization.

Converge curves. As shown in Figure 5, from the training loss and C4-en loss curves, we observe that OLMoE-7B-DFC×4 achieves a faster convergence, with a reduction of 0.012 in training loss compared to the baseline. Furthermore, we observe that Hyper-Connections (OLMoE-7B-DHC×4) converge significantly faster than Frac-Connections (OLMoE-7B-DFC×4), suggesting that when applying HC or FC, a trade-off between memory consumption and performance needs to be considered.

Downstream performance. Throughout training, the OLMoE-7B-DFC×4 variant maintains a consistent advantage on most benchmarks, including Commonsense QA and WinoGrande QA. For HellaSwag, the OLMoE-7B-DFC×4 variant maintains an early advantage over the baseline; however, as training progresses, the gap narrows, and the baseline model nearly catches up toward the end. Table 3 shows the performance of models trained with 3T tokens, and the OLMoE-7B-DFC×4 variant demonstrates higher accuracy across most benchmarks. Specifically, it outperforms the baseline by +0.95% on WinoGrande (67.64% → 68.59%), +0.50%

- Table 3 Downstream evaluations for OLMoE-7B models with training 3T tokens. MMLU Var is a modified version of MMLU that includes varying few-shot examples, providing stable feedback during early training.

Method

HellaSwag

BoolQ

WinoGrande

MMLU Var

PIQA SciQ

Commonsense QA

AVG

OLMoE-7B 74.28 72.87 67.64 41.83 78.73 93.60 49.14 68.30 OLMoE-7B-DFC×4 74.48 72.11 68.59 42.33 79.16 94.10 49.80 68.65

on MMLU Var (41.83% → 42.33%), and +0.66% on Commonsense QA (49.14% → 49.80%), indicating that Frac-Connections enhance knowledge retention and generalization.

These results indicate that Frac-Connections not only improve training efficiency but also lead to better model generalization across diverse NLP tasks.

5.3 Dense Models

500 1000 1500 2000

Tokens (Billions)

2.55

2.60

2.65

2.70

2.75

Loss

Training Loss

OLMo2-1.2B

OLMo2-1.2B-DFCx4

500 1000 1500 2000

Tokens (Billions)

2.80

2.85

2.90

2.95

3.00

Loss

C4-en Loss

OLMo2-1B2

OLMo2-1B2-DFCx4

500 1000 1500 2000

Tokens (Billions)

55.0

57.5

60.0

62.5

65.0

Accuracy(%)

HellaSwag Acc.

OLMo2-1B2

OLMo2-1B2-DFCx4

500 1000 1500 2000

Tokens (Billions)

86

88

90

92

Accuracy(%)

SciQ Acc.

OLMo2-1B2

OLMo2-1B2-DFCx4

Figure 6 Training and evaluation performance of OLMo2-1B2 models. The plots show the training loss, C4-en loss, and accuracy on HellaSwag and SciQ over the course of training. The results are EMA-smoothed for clarity. The OLMo2-1B2-DFCx4 variant demonstrates improved loss reduction and higher accuracy compared to the baseline OLMo2-1B2 model.

We evaluate Frac-Connections through experiments on the OLMo2-1B2 model, as illustrated in Figure 6 and

- Table 4. OLMo2-1B2-DFC×4 variant exhibits consistently lower training loss and C4-en loss compared to the baseline OLMo2-1B2 model. Furthermore, the OLMo2-1B2-DFC×4 variant consistently outperforms the baseline on HellaSwag and SciQ throughout training. This suggests that Frac-Connections facilitate more efficient optimization and improving parameter utilization.

Table 4 Downstream evaluations for OLMo2 models with training 2T tokens. MMLU Var is a modified version of MMLU that includes varying few-shot examples, providing stable feedback during early training.

HellaSwag

WinoGrande

MMLU Var

Commonsense QA

Methods

BoolQ

PIQA SciQ

AVG

OLMo2-1B2 64.7 63.0 61.6 36.4 75.6 91.8 44.6 62.5 OLMo2-1B2-DFC×4 65.4 65.1 62.7 37.1 75.2 92.2 44.7 63.2

The downstream evaluation results in Table 4 demonstrate that OLMo2-1B2-DFC×4 achieves superior performance across multiple tasks, particularly on BoolQ (+2.1%), WinoGrande (+1.1%), and SciQ (+0.4%), while maintaining comparable performance on PIQA. The average accuracy improvement of +0.7% confirms that Frac-Connections enhance generalization across diverse benchmarks. Notably, the improvements on reasoning-intensive tasks such as BoolQ and WinoGrande highlight the ability of Frac-Connections to enhance model expressiveness without increasing computational overhead.

#### 6 Conclusion

We introduced Frac-Connections, an efficient alternative to Hyper-Connections that divides the hidden states into fractions rather than expanding their width. Frac-Connections address the seesaw effect between gradient vanishing and representation collapse while reducing memory usage and computational costs. Our experimental results demonstrate that Frac-Connections are a practical and scalable solution for large language models.

#### References

- [1] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 2020.

- [2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- [3] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

- [4] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

- [5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

- [6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

- [7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.

- [8] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

- [9] Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, and Xun Zhou. Over-tokenized transformer: Vocabulary is generally worth scaling. arXiv preprint arXiv:2501.16975, 2025.

- [10] Zihao Huang, Qiyang Min, Hongzhi Huang, Defa Zhu, Yutao Zeng, Ran Guo, and Xun Zhou. Ultra-sparse memory network. arXiv preprint arXiv:2411.12364, 2024.

- [11] Matt Gardner Johannes Welbl, Nelson F. Liu. Crowdsourcing multiple choice science questions. 2017.
- [12] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

- [13] Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Fractalnet: Ultra-deep neural networks without residuals. arXiv preprint arXiv:1605.07648, 2016.

- [14] Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers. arXiv preprint arXiv:2004.08249, 2020.

- [15] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

- [16] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2409.02060.

- [17] Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. 2 olmo 2 furious, 2024. URL https://arxiv.org/abs/2501.00656.
- [18] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI spring symposium series, 2011.

- [19] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9), 2021.

- [20] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

- [21] N Shazeer, A Mirhoseini, K Maziarz, A Davis, Q Le, G Hinton, and J Dean. The sparsely-gated mixture-of-experts layer. Outrageously large neural networks, 2017.

- [22] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.

- [23] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, 2017.

- [24] Ya Wang, Zhijian Zhuo, Yutao Zeng, Xun Zhou, Jian Yang, and Xiaoqing Li. Scale-distribution decoupling: Enabling stable and effective training of large language models. arXiv preprint arXiv:2502.15499, 2025.

- [25] Shufang Xie, Huishuai Zhang, Junliang Guo, Xu Tan, Jiang Bian, Hany Hassan Awadalla, Arul Menezes, Tao Qin, and Rui Yan. Residual: Transformer with dual residual connections. arXiv preprint arXiv:2304.14802, 2023.

- [26] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

- [27] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

- [28] Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, and Xun Zhou. Hyper-connections. arXiv preprint arXiv:2409.19606, 2024.

## Appendix

#### A PyTorch Implementation of Frac-connections

###### Algorithm 2 Pseudocode of frac-connections in a PyTorch-like style.

# h: hidden vector (BxLxD) class FracConnection(nn.Module):

def __init__(self, dim, rate, config, dynamic_alpha, dynamic_beta, device=None): super(FracConnection, self).__init__() self.rate = rate self.dynamic_alpha = dynamic_alpha self.dynamic_beta = dynamic_beta self.use_tanh = config.use_tanh self.use_hc_norm = config.use_hc_norm self.use_scale = config.use_scale self.static_beta = nn.Parameter(torch.ones((rate,), device=device)) self.static_alpha = nn.Parameter(torch.cat([torch.eye((rate), device=device), torch.eye((rate), device=device)], dim=1)

) if self.dynamic_alpha:

self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim // self.rate, rate*2), device=device)) if self.dynamic_beta:

self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim // self.rate, ), device=device))

if self.use_scale: self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01) self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)

if self.use_hc_norm: self.layer_norm = LayerNorm(dim // self.rate)

def width_connection(self, h): # get alpha and beta h_shape = h.shape h_reshape = h.reshape(h_shape[:-1] + (self.rate, h_shape[-1] // self.rate) ) if self.use_hc_norm:

norm_h = self.layer_norm(h_reshape) else:

norm_h = h_reshape if self.use_tanh:

dynamic_alpha = F.tanh(norm_h @ self.dynamic_alpha_fn) else:

dynamic_alpha = norm_h @ self.dynamic_alpha_fn if self.use_scale:

dynamic_alpha = dynamic_alpha * self.dynamic_alpha_scale alpha = dynamic_alpha + self.static_alpha[None, None, ...] if self.use_tanh:

dynamic_beta = F.tanh(norm_h @ self.dynamic_beta_fn) else:

dynamic_beta = norm_h @ self.dynamic_beta_fn if self.use_scale:

dynamic_beta = dynamic_beta * self.dynamic_beta_scale

beta = dynamic_beta + self.static_beta[None, None, ...] mix_h = (alpha.transpose(-1, -2).contiguous().float() @ h_reshape.float()).bfloat16()

return mix_h, beta

def depth_connection(self, mix_h, h_o, beta): h_o_shape = h_o.shape h = beta[..., None] * h_o.reshape(h_o_shape[:-1] + (self.rate, h_o_shape[-1]//self.rate)) + mix_h[..., self.rate:, :] h_shape = h.shape

return h.reshape(h_shape[:-2] + (h_shape[-2] * h_shape[-1], ))

###### Algorithm 3 Pseudocode of transformer with frac-connections in a PyTorch-like style.

# h: hidden vector (BxLxD) # atten_frac_connection, ffn_frac_connection: frac-connection modules # attn_norm, ffn_norm: normalization modules

# Attention Block mix_h, beta = atten_frac_connection.width_connection(h) mix_h_shape = mix_h.shape h = mix_h[...,:self.rate,:].reshape(mix_h_shape[:-2] + (mix_h_shape[-2] // 2 * mix_h_shape[-1], )) h = attn_norm(h) h = self_attention(h) h = atten_frac_connection.depth_connection(mix_h, dropout(h), beta)

# FFN Block mix_h, beta = ffn_frac_connection.width_connection(h) mix_h_shape = mix_h.shape h = mix_h[...,:self.rate,:].reshape(mix_h_shape[:-2] + (mix_h_shape[-2] // 2 * mix_h_shape[-1], )) h = ffn_norm(h) h = ffn(h) h = ffn_frac_connection.depth_connection(mix_h, dropout(h), beta)

#### B OLMo2 Model Results

###### training loss

###### C4 en val. loss

###### dolma books val. loss

###### dolma cc val. loss

3.00

3.10

2.9

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

2.95

2.7

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

3.05

2.8

2.90

3.00

2.6

2.85

2.95

2.7

2.80

2.90

2.5

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### dolma pes2o val. loss

###### dolma reddit val. loss

###### dolma stack val. loss

###### dolma wiki val. loss

2.30

1.45

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

2.5

3.20

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

2.25

1.40

3.15

2.20

1.35

2.4

3.10

2.15

1.30

2.10

3.05

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### ice val. loss

###### m2d2-s2orc val. loss

###### pile val. loss

###### wikitext 103 val. loss

2.35

3.25

2.95

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

2.30

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

2.90

3.20

2.5

2.25

2.85

3.15

2.20

2.80

2.4

3.10

2.15

2.75

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### MMLU stem Var Acc. (%)

###### MMLU hum. Var Acc. (%)

###### MMLU soc. sci. Var Acc. (%)

###### MMLU other Var Acc. (%)

34

48

34

38

46

32

32

36

44

30

30

34

42

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

28

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

40

28

32

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### MMLU avg. Acc. (%)

###### HellaSwag Acc. (%)

###### ARC Challenge Acc. (%)

###### SciQ Acc. (%)

65

92

40

36

90

60

35

34

88

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

55

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

30

86

32

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### ARC easy Acc. (%)

###### PIQA Acc. (%)

###### WinoGrande Acc. (%)

###### Openbook QA Acc. (%)

64

76

40

62

70

38

74

60

36

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

58

65

72

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

34

56

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### BoolQ Acc. (%)

###### COPA Acc. (%)

###### Commonsense QA Acc. (%)

###### Social Iqa Acc. (%)

- 44

- 45

- 46

- 47

- 48

85

46

65

44

80

60

42

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

OLMo2-1B2

55

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

OLMo2-1B2-DFCx4

75

40

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### Figure 7 Loss and accuracy curves for OLMo2-1B2 and OLMo2-1B2-DFC×4 models.

#### C OLMoE-7B Model Results

###### training loss

###### C4 en val. loss

###### dolma books val. loss

###### dolma cc val. loss

2.8

2.5

OLMoE-7B

OLMoE-7B

OLMoE-7B

OLMoE-7B

2.9

2.7

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

2.8

2.4

2.8

2.6

2.7

2.3

2.5

2.7

2.2

2.6

2.4

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### dolma pes2o val. loss

###### dolma reddit val. loss

###### dolma stack val. loss

###### dolma wiki val. loss

2.3

1.10

OLMoE-7B

OLMoE-7B

OLMoE-7B

OLMoE-7B

2.4

3.0

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

1.05

2.2

2.3

1.00

2.9

2.1

0.95

2.2

2.8

0.90

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### ice val. loss

###### m2d2-s2orc val. loss

###### pile val. loss

###### wikitext 103 val. loss

2.2

3.2

OLMoE-7B

OLMoE-7B

OLMoE-7B

OLMoE-7B

2.8

2.5

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

2.1

3.1

2.4

2.6

2.3

2.0

3.0

2.2

2.4

1.9

2.9

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### MMLU stem Var Acc. (%)

###### MMLU hum. Var Acc. (%)

###### MMLU soc. sci. Var Acc. (%)

###### MMLU other Var Acc. (%)

55

37.5

38

45

36

35.0

50

34

32.5

40

45

32

30.0

OLMoE-7B

OLMoE-7B

OLMoE-7B

OLMoE-7B

35

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

30

27.5

40

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### MMLU avg. Acc. (%)

###### HellaSwag Acc. (%)

###### ARC Challenge Acc. (%)

###### SciQ Acc. (%)

75

50

42.5

94

70

40.0

45

92

37.5

65

40

90

OLMoE-7B

OLMoE-7B

OLMoE-7B

OLMoE-7B

35.0

60

35

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

88

32.5

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### ARC easy Acc. (%)

###### PIQA Acc. (%)

###### WinoGrande Acc. (%)

###### Openbook QA Acc. (%)

70

80

80

45

75

65

78

40

76

70

60

OLMoE-7B

OLMoE-7B

OLMoE-7B

OLMoE-7B

74

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

65

35

55

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### BoolQ Acc. (%)

###### COPA Acc. (%)

###### Commonsense QA Acc. (%)

###### Social Iqa Acc. (%)

75

90

70

48

50

85

65

46

45

60

OLMoE-7B

OLMoE-7B

OLMoE-7B

OLMoE-7B

80

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

OLMoE-7B-DFCx4

55

44

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### Figure 8 Loss and accuracy curves for OLMoE-7B and OLMoE-7B-DFC×4 models.

#### D Downstream Benchmarks

Table 5 Downstream Benchmarks.

|Downstream Benchmarks|
|---|
|piqa [1] hellaswag [26] winogrande [19] openbook_qa [15] sciq [11] arc_easy [4] arc_challenage [4] copa [18] boolq [3] commonsense_qa [22] social_iqa [20] mmlu [8]<br><br>|

