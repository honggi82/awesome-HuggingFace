# arXiv:2409.19606v3[cs.LG]18Mar2025

## HYPER-CONNECTIONS

Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou Seed-Foundation-Model Team, ByteDance

{zhudefa,huanghongzhi.51,huangzihao.notabot,yutao.zeng, maoyunyao.myy,wubanggu,minqiyang,zhouxun}@bytedance.com

ABSTRACT

We present hyper-connections, a simple yet effective method that can serve as an alternative to residual connections. This approach specifically addresses common drawbacks observed in residual connection variants, such as the seesaw effect between gradient vanishing and representation collapse. Theoretically, hyperconnections allow the network to adjust the strength of connections between features at different depths and dynamically rearrange layers. We conduct experiments focusing on the pre-training of large language models, including dense and sparse models, where hyper-connections show significant performance improvements over residual connections. Additional experiments conducted on vision tasks also demonstrate similar improvements. We anticipate that this method will be broadly applicable and beneficial across a wide range of AI problems.

1 INTRODUCTION

###### Training Loss

###### HellaSwag Acc.

###### ARC Challenge Acc.

###### C4 en val. loss

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

47.5

70.0

2.85

2.400

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

45.0

67.5

Accuracy(%)

Accuracy(%)

2.375

2.80

42.5

65.0

Loss

Loss

2.350

40.0

2.75

62.5

2.325

37.5

2.70

60.0

2.300

x1.8 0.027

OLMoE-1B-7B

35.0

x1.8 0.028

OLMoE-1B-7B-DHCx4

2.65

57.5

2.275

32.5

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (Billions)

Tokens (Billions)

Tokens (Billions)

Tokens (Billions)

Figure 1: The performance of the baseline model OLMoE-1B-7B and the model with hyperconnections, OLMoE-1B-7B-DHC×4. (1) and (2) show the training loss (0.99 EMA smoothed) and the C4-en validation loss, respectively. Our method converges 1.8 times faster compared to the baseline and maintains a significant advantage at the 500B tokens. (3) and (4) show the accuracy curves on HellaSwag and ARC-Challenge, demonstrating the superior performance of the OLMoE-1B-7B-DHC×4 model.

Deep learning has achieved tremendous success across various domains, where residual connections (He et al., 2016) have been instrumental in contemporary neural network architectures, including transformers and CNNs. Residual connections help mitigate the problem of gradient vanishing, enabling the effective training of very deep networks. However, it is important to acknowledge that residual connections are not infallible solutions and still present limitations that remain unresolved.

The two main variants of residual connections, Pre-Norm and Post-Norm, each make distinct trade-offs between gradient vanishing and representation collapse. Pre-Norm applies normalization operations to the input before each residual block, effectively addressing the problem of gradient vanishing (Bengio et al., 1994; Glorot & Bengio, 2010). However, it can also lead to the issue of collapse in deep representations (Liu et al., 2020), where hidden features in deeper layers become highly similar, diminishing the contribution of additional layers as their number increases. In contrast, Post-Norm applies normalization after the output of each residual block, reducing the influence of a hidden state on subsequent layers. This approach can alleviate the issue of representation collapse but

|: scalar, weights of hyper-connections : hidden vectors<br><br>|layer|
|---|
<br><br>: attention or FFN|
|---|

𝛽 + 𝛽 +

+

+ +

+ +

𝛽

𝛽

|layer|
|---|

|layer| |
|---|---|
| | |

|layer| |
|---|---|
| | |

|layer|
|---|

+

+

𝛼 ,  𝛼 ,  𝛼 ,  𝛼 ,  𝛼 ,  𝛼 , 

𝛼 ,  𝛼 ,  𝛼 ,  𝛼 ,  𝛼 ,  𝛼 , 

𝛼 ,  𝛼 , 

|ℎ|
|---|

|h|
|---|

|h|
|---|

|h|
|---|

|h|
|---|

|h|
|---|

|h|
|---|

- (a) Residual connections

Figure 2: Hyper-connections (HC) with an expansion rate of n = 2. (a) Residual connections.

- (b) Hyper-connections: β1, β2, α0,0, α0,1, α1,0, α1,1, α2,1, and α2,2 are learnable scalars or scalars predicted by the network , depending on the specific HC version. These connections enable lateral information exchange and vertical integration of features across depths. The Transformer with HC is shown in Fig. 17. They can be decoupled into depth-connections and width-connections. (c) Depth-connections perform a weighted sum between the layer output and the hidden vector h1. (d) Width-connections allow information exchange between the hidden vectors h1 and h2.

(b) Hyper-connections (c) Depth-connections

(d) Width-connections

also reintroduces the problem of vanishing gradients. The vanishing gradient and the representation collapse are like two ends of a seesaw, with these two variants making respective trade-offs between these issues. The key issue is that residual connections, including both Pre-Norm and Post-Norm variants, predefine the strength of connections between the output and input within a layer.

Driven by the limitations of residual connections, an important question arises: Can neural networks autonomously learn the optimal strength of connections to improve performance? To address this, we propose hyper-connections (HC), which lead to significantly improved performance with a negligible increase in computation and parameters. We will show that both PostNorm and Pre-Norm variants can be expressed as specific nontrainable forms of hyper-connections, as discussed in § 3.1.

1.0

0.8

###### i+1icos(h,h)00

0.6

0.4

The core idea of hyper-connections (HC) is to propose learnable depth-connections and width-connections, as depicted in Fig.2 (b). These connections flexibly integrate features vertically across depths, compared to the residual connections shown in Fig.2 (a). Depth-connections can be considered as a generalized residual connections, assigning weights to the connections between the inputs and outputs of each layer. To enable the network to model different depth-connections simultaneously, we expand the network’s input into n copies, each having its own depth connection, as shown in Fig. 2 (b). This design allows multiple hidden vectors to reserve multiple patterns connecting preceding layers, as shown in § 4.5. Moreover, we establish width connections between the n hidden vectors, allowing information exchange between hidden vectors within the same layer, as shown in Fig. 2 (b). We argue that n (> 1) hidden states are necessary. As analyzed in Appendix F, the seesaw effect persists when n = 1, and experiments show that it does not improve performance, as shown in Fig. 5. In contrast, when n > 1, hyper-connections can not only learn to adjust the strength of residuals but also rearrange layers, either sequentially or in parallel, as discussed in § 3.2. To further enhance flexibility, we introduce dynamic hyper-connections (DHC), enabling the network to adjust connection weights according to the input. Notably, although HC seem to increase the network’s width by n times, the additional parameters and computational cost are almost negligible, as analyzed in Appendix B. The Transformer with HC is shown in Fig. 17.

Pre-Norm

0.2

Hyper-Connection

0.0

0 5 10 15 20 25 30

Layer Index i

Figure 3: Cosine similarity between the input of the current and the previous layers for the OLMo-1B models (Groeneveld et al., 2024). The curve represents the median of similarity, while the shaded area indicates the range between the 5th and 95th percentiles. The red curve shows the model with Pre-Norm, and the blue curve shows that with hyper-connections.

Our research, primarily centered on large language models (LLMs) pre-training, also extends to visual generation and classification tasks. Using Pre-Norm as a baseline, we demonstrate the significant benefits of hyper-connections, including 1B and 7B dense models as well as 7B MoE models, as

detailed in § 4. The benefits are particularly prominent for OLMoE (Muennighoff et al., 2024) as presented in Fig.1. The model utilizing DHC converges 1.8 times faster and shows an improvement of 6 points on ARC-Challenge compared to the baseline trained with 500 B tokens. According to our visualization analysis, as shown in Fig.3, the baseline model tends toward representation collapse, characterized by high similarity between features of adjacent layers. In contrast, models with HC exhibit significantly lower similarity between features across adjacent layers and a wider range of similarities. This suggests that HC enhance the impact of each layer. Further discussion is provided in §4.5 and in Appendix F. These compelling pieces of evidence demonstrate the generality of the hyper-connections principle, and we anticipate their applicability in numerous other AI challenges.

- 2 METHOD

- 2.1 STATIC HYPER-CONNECTIONS

Consider the hidden vector hk−1 ∈ Rd (or hk−1 ∈ Rd×1) as the input to the k-th layer, with the initial input h0 to the network. Initially, h0 ∈ Rd is replicated n times to form the initial hyper hidden matrix H0 = h0 h0 ... h0 ⊺ ∈ Rn×d. Here, n is the expansion rate. For the k-th layer, the input consists of the hyper hidden matrix from the previous layer Hk−1 =

h1k−1 hk2−1 ... hkn−1 ⊺ ∈ Rn×d. Finally, we sum the last hyper hidden matrix row-wise to obtain the required hidden vector, which is then passed through a final projector to produce the final output of the network (i.e., a normalization layer and an unembedding layer in transformers). To simplify the notation in subsequent analysis, we omit the layer index and simply denote the hyper-hidden matrix as H = (h1 h2 ... hn)⊺.

The hyper-connections (HC) can be represented by a matrix HC, where each element defines the connection weight. The matrix is structured as follows:





0 β1 β2 ··· βn

- α1,0 α1,1 α1,2 ··· α1,n
- α2,0 α2,1 α2,2 ··· α2,n

01×1 B Am Ar

∈ R(n+1)×(n+1). (1)

HC =

=

 

 

... .

. . .

αn,0 αn,1 αn,2 ··· αn,n

Consider a network layer T , it integrates self-attention layers and feed-forward networks within transformers. The output of the HC, denoted by Hˆ, can be simply formulated as follows:

Hˆ = HC(T ,H) = B⊺T (H⊺Am)⊺ + Ar⊺H. (2)

We use Am as weights to perform a weighted sum on the input H = (h1 h2 ... hn)⊺ to obtain the input h0 of the current layer T , which is given by:

h⊺0 = Am⊺H, (3) While Ar is used to connect H and map it to a hyper hidden matrix H′, as shown below:

H′ = Ar⊺H. (4) Subsequently, the output is given by:

Hˆ = B⊺(T h0)⊺ + H′. (5) The depth-connections can be decoupled as the following matrix, which is shown at Fig 2 (a):

β1 β2 ··· βn α1,1 α2,2 ··· αn,n ∈ R2×n, (6)

B diag(Ar)

DC =

=

where the first row B represents the weights of the output of the current layer T , and the last row diag(Ar) represents the weights of the input. We use diag(Ar) to represent the flatten vector of the diagonal entries of Ar.

The width-connections matrix can be defined as follows, which is shown at Fig 2 (b):

WC = (Am Ar) ∈ Rn×(n+1). (7) The algorithm that employs hyper-connections is presented in Algorithm 1.

- 2.2 DYNAMIC HYPER-CONNECTIONS

The entries of HC can dynamically depend on the input H, which the matrix representation of dynamic hyper-connections (DHC) is defined as follows:

HC(H) =

01×1 B(H) Am(H) Ar(H)

(8)

Similarly, given a layer T and input H, we obtain the output of the DHC as follows:

Hˆ = HC(H)(T ,H). (9)

In practice, we combine the dynamic and static matrices to achieve DHC. The dynamic parameters are obtained through a linear transformation. To stabilize the training process, we introduce normalization before the linear transformation and apply the tanh activation function after it, scaling it by a small initial learnable factor. The following equations detail how these dynamic parameters are computed:

H = norm(H) (10) B(H) = sβ ◦ tanh(HWβ)⊺ + B ∈ R1×n (11)

Am(H) = sα ◦ tanh(HWm) + Am ∈ Rn×1 (12) Ar(H) = sα ◦ tanh(HWr) + Ar ∈ Rn×n (13)

Our experiments in § 4 demonstrate that dynamic hyper-connections outperform static hyperconnections in language modeling tasks. The PyTorch implementations for both the static and dynamic variants of hyper-connections are detailed in Algorithm 2 and 3.

- 2.3 INITIALIZATION

In order to make the initialization of the hyper-connections equivalent to the Pre-Norm residual connections, we adopt the following initialization strategy. The dynamic parameters Wβ, Wm, and Wr in Eqs. 11, 12, and 13 are initialized to 0, while the static matrices are initialized as follows:

01×1 Bk Amk Ark

01×1 11×n ek mod n en×n

, (14) where k is the index of the layer. mod denotes the modulo operation.

=

- 3 WHY HYPER-CONNECTIONS

In this section, we elucidate the rationale behind hyper-connections. We explore how variants of residual connections, namely Pre-Norm and Post-Norm, can be viewed as non-trainable hyperconnections, and introduce the concept of sequential-parallel duality, demonstrating how hyperconnections can dynamically optimize layer arrangements to enhance network performance. A visulize analysis of hyper-connections through an unfolded view is discussed in § 4.5.

- 3.1 RESIDUAL CONNECTIONS AS NON-TRAINABLE HYPER-CONNECTIONS

The Pre-Norm and Post-Norm residual connections can be represented as the following hyperconnections matrices with an expansion rate n = 1:

 , (16)

 

- 0 √ 1 σi2+σo2+2σio

- 1 √ 1 σi2+σo2+2σio

, (15) HCPostNorm =

- 0 1
- 1 1

HCPreNorm =

where σi and σo denote the standard deviations of the input and output of the neural network layer, respectively, and σio is the covariance between them.

For Pre-Norm, its hyper-connection matrix is a 2 × 2 matrix where the bottom right triangular part is filled with 1 and the rest is a placeholder 0. For Post-Norm, the weights depend on the variances and covariance of the input and output, forming a 2 × 2 matrix. Therefore, their hyper-connection matrices are non-trainable. In this work, we propose hyper-connections that can be (n + 1) × (n + 1) matrices, with weights that are trainable or even predicted based on the input. The complete derivation is provided in Appendix G.

- 3.2 SEQUENTIAL-PARALLEL DUALITY

Given a series of neural network modules, we have the option to arrange them either sequentially or in parallel. However, hyper-connections offer an approach that learns to rearrange these layers in a configuration blending both sequential and parallel arrangements.

+

=

| | |
|---|---|
| | |

+ +

+

|layer 2|
|---|

||layer 2|
|---|
| |
|---|---|
| | |

| | |
|---|---|
| | |

|layer 2|
|---|

2

| | |
|---|---|
| | |

=

+ +

+

|layer 1|
|---|

||layer 1|
|---|
| |
|---|---|
| | |

| | |
|---|---|
| | |

|layer 1|
|---|

1

+

=

=

(a) Sequential Arrangement

(b) Parallel Arrangement

Figure 4: Sequential and parallel arrangements of hyper-connections with n = 2.

Without loss of generality, we set the expansion rate to n = 2. If the hyper-connections are learned as the following matrix, the neural network will be arranged sequentially:

- 0 1 1
- 1 1 0 0 0 1

. (17)

HC =

In this case, the depth connection degenerates into a residual connection, as shown in Fig. 4 (a).

When the hyper-connections for odd and even layers (with layer numbering starting from 1) are defined by the following matrices, the neural network will be arranged in parallel every two consecutive layers, similar to the arrangement of parallel transformer blocks in transformers (Wang, 2021), as shown in Fig. 4 (b). The general and complete derivation is provided in Appendix H.

- 0 1 0
- 1 1 1 1 1 1

- 0 0 1
- 0 1 0 1 0 1

, (18) HCeven =

. (19)

HCodd =

Thus, learning the hyper-connection matrix in various forms can create layer arrangements that surpass traditional sequential and parallel configurations, resulting in a soft-mixture or even dynamic

arrangement. For static hyper-connections, the layer arrangement within the network remains fixed after training. In contrast, dynamic hyper-connections allow the arrangement to adapt dynamically for each token.

- 4 RESULTS

Training Loss vs Tokens

OLMo-1B-baseline

2.60

OLMo-1B-DHCx1 OLMo-1B-DHCx2 OLMo-1B-DHCx4 OLMo-1B-DHCx8

2.55

TrainingLoss

2.50

2.45

2.40

100 150 200 250 300 350 400 450 500

Tokens (Billions)

Training Loss vs Tokens

OLMo-1B-baseline

2.60

- OLMo-1B-DHCx1 W/O tanh

- OLMo-1B-DHCx2 W/O tanh OLMo-1B-DHCx4 W/O tanh OLMo-1B-DHCx8 W/O tanh

2.55

TrainingLoss

2.50

2.45

2.40

100 150 200 250 300 350 400 450 500

Tokens (Billions)

- Figure 5: Comparison of training loss curves for different expansion rate. The left subfigure includes models with dynamic hyper-connections (DHC) at various expansion rates, while the right subfigure shows the effect of omitting the tanh function. Both subfigures illustrate how increasing the expansion rate leads to improved training loss performance over 500B tokens. Results are smoothed using an exponential moving average with a coefficient of 0.99.

Table 1: Ablation study on expansion rates n with training on 500 B tokens.

Down Stream Avg, Acc. ↑ OLMo-1B 2.811 18.023 2.544 14.229 62.5

V2 Eval Loss ↓

V2 Eval PPL ↓

V3 Eval Loss ↓

V3 Eval PPL ↓

Methods

- OLMo-1B-DHC×1 W/O tanh 2.822 18.270 2.556 14.428 62.3
- OLMo-1B-DHC×2 W/O tanh 2.792 17.663 2.537 14.033 63.8 OLMo-1B-DHC×4 W/O tanh 2.779 17.451 2.516 13.844 64.4 OLMo-1B-DHC×8 W/O tanh 2.777 17.425 2.514 13.819 63.8

- OLMo-1B-DHC×1 2.819 18.125 2.556 14.418 62.3
- OLMo-1B-DHC×2 2.802 17.950 2.534 14.114 63.0 OLMo-1B-DHC×4 2.781 17.509 2.514 13.826 63.8 OLMo-1B-DHC×8 2.778 17.445 2.516 13.843 62.8

We primarily conduct experiments on pre-training of large language model, including dense and Mixture-of-Experts (MoE) (Shazeer et al., 2017) models, and extend to visual generation and classification tasks. Due to space constraints, we include the vision experiments in the Appendix E.

Experiment Settings. We employ the experimental setup outlined by OLMo (Groeneveld et al., 2024) for dense models and by OLMoE (Muennighoff et al., 2024) for MoE models. For dense models, we use dolmap-v1.5-sample (Soldaini et al., 2024) as our training dataset. We conduct ablation studies on 1B models and assess the effectiveness of our method at the 7B model scale. For MoE models, we train the OLMoE-1B-7B model, both with and without hyper-connections, on the OLMOE-MIX dataset. These models activate 1.3B out of a total of 7B parameters. All experiments are trained on 500B tokens.

Implementation. We maintain the training configuration of the baseline model, replacing the residual connections with hyper-connections. The static component in Eqs. 1, 11, 12, 13 does not utilize weight decay, whereas the dynamic component does. Since the hyper hidden vectors of the final transformer block are ultimately summed, we ensure that the standard deviation (std) of the output (before the final layernorm and unembedding layers) remains consistent with the original. At initialization, we scale the std of the weights of the output module at all layers, including those of the second linear layer of the feedforward network and the output projector of the attention module, by a factor of √n,

where n represents the expansion rate. The parameters and computational overhead introduced by hyper-connections is negligible, see Table. 7 and 8.

Metrics. In accordance with the methodology of OLMo (Groeneveld et al., 2024), we report the average perplexities (PPL) and losses on both the V2 and V3 validation sets, along with the average metrics for zero-shot evaluation on downstream benchmarks (refer to Table 13). We observe significant volatility in the zero-shot performance indicators for the datasets (highlighted in grey in Table 13), with fluctuations exceeding 20% across neighboring checkpoints. For more reliable and consistent results, we excludes these volatile datasets from our analysis. For the MoE models, in line with OLMoE, we also present losses on V3 validation sets, and accuracies on downstream benchmarks (refer to Table 14).

- 4.1 ABLATION STUDY

We use the dynamic hyperconnections with an expansion rate of n = 4 and include the tanh function as the default method, marked with the suffix -DHC, while -SHC denotes static hyper-connections.

The evaluation results are presented in Table 1, and the training loss curves are depicted in Fig. 5. We observe that with an expansion rate of n = 1, the performance of DHC is inferior to the baseline. However, for n > 1, DHC significantly outperforms the baseline, achieving superior results at n = 4, with the increase to n = 8 providing minimal additional benefits. Notably, OLMo-1B-DHC×8 W/O tanh excels on both V2 and V3 validation sets, with a reduction in V2 Eval Loss by 0.034 and V3 Eval Loss by 0.029 compared to the baseline. Furthermore, the decline rate of training losses for DHC (n ≥ 2) is steeper than that of the baseline, and DHC demonstrates greater stability, with no spikes observed in any DHC experiments.

Static and dynamic hyper-connections. Table 2 presents an ablation study comparing SHC and DHC. All hyper-connection (HC) variants significantly outperform the baseline. At an expansion rate of 2, the improvements of DHC and SHC are similar. However, at an expansion rate of 4, DHC performs notably better than SHC.

- Table 2: Ablation study on static and dynamic hyper-connections with training on 500 B tokens.

Methods

V2 Eval Loss ↓

V2 Eval PPL ↓

V3 Eval Loss ↓

V3 Eval PPL ↓

Down Stream Avg, Acc. ↑

OLMo-1B 2.811 18.023 2.544 14.229 62.5 OLMo-1B-SHC×2 2.799 17.778 2.538 14.152 63.4 OLMo-1B-DHC×2 2.802 17.950 2.534 14.114 63.0 OLMo-1B-DHC×2 W/O tanh 2.792 17.663 2.529 14.033 63.8

OLMo-1B-SHC×4 2.791 17.671 2.528 14.025 63.6 OLMo-1B-DHC×4 2.781 17.509 2.515 13.826 63.8 OLMo-1B-DHC×4 W/O tanh 2.779 17.451 2.516 13.844 64.4

The importance of B and WC. As shown in Table 3, not training WC leads to significant performance declines, with the V2 loss increasing by 0.021 and the V3 loss by 0.017, as seen when comparing the 4th and 6th lines of Table 3. In contrast, the impact is less pronounced when B is not trained. Therefore, ensuring the trainability of both WC and B is crucial.

- 4.2 COMPARISON WITH RELATED WORKS

We implemented the Altup (Baykal et al., 2024) and ResiDual (Xie et al., 2023) methods in OLMo. Altup is motivated to widen the hidden dimension while maintaining low computation cost by passing only a part of hidden state to transformer blocks. By contrast, ResiDual is proposed to combine both Pre- and Post-Norm in a two-stream style. Both methods expand the hidden size by n times with negligible computational overhead, with ResiDual expanding it exactly 2 times. For a fair comparison, we set n = 2 in our experiments. Unfortunately, although these methods show gains in the early stages of training, they are gradually surpassed by the baseline, as demonstrated by the results in

- Table 4 and the training loss curves in Fig. 15.

- Table 3: Ablation study on OLMo-1B-DHC×4. In the B or WC column, the symbol "✗" denotes parameters that are not trainable from initialization.

V2 Eval Loss ↓

V2 Eval PPL ↓

V3 Eval Loss ↓

V3 Eval PPL ↓

Down Stream Avg, Acc. ↑

WC B Tanh

- ✗ ✓ ✗ 2.804 17.912 2.537 14.145 62.5

✓ ✗ ✗ 2.781 17.493 2.518 13.874 63.6 ✓ ✓ ✗ 2.779 17.773 2.516 13.823 64.4

- ✗ ✓ ✓ 2.802 17.914 2.532 14.072 63.4

✓ ✗ ✓ 2.783 17.504 2.520 13.906 63.4 ✓ ✓ ✓ 2.781 17.835 2.515 13.807 63.8

Table 4: Performance of related methods on OLMo-1B models.

V2 Eval Loss ↓

V2 Eval PPL ↓

V3 Eval Loss ↓

V3 Eval PPL ↓

Down Stream Avg, Acc. ↑

Methods

OLMo-1B 2.811 18.023 2.544 14.229 62.5 OLMo-1B-ResiDual 2.825 18.375 2.551 14.346 62.0 OLMo-1B-Altup×2 2.827 18.268 2.558 14.454 62.4

OLMo-1B-DHC×2 2.802 17.950 2.534 14.114 63.0 OLMo-1B-DHC×2 W/O tanh 2.792 17.663 2.529 14.033 63.8

- 4.3 7B MODELS

100 200 300 400 500

Tokens (Billions)

2.2

2.3

2.4

Loss

Training Loss

OLMo-7B

OLMo-7B-DHCx4

100 200 300 400 500

Tokens (Billions)

2.5

2.6

2.7

Loss

C4-en Loss

OLMo-7B

OLMo-7B-DHCx4

100 200 300 400 500

Tokens (Billions)

55

60

65

70

Accuracy(%)

HellaSwag Acc.

OLMo-7B

OLMo-7B-DHCx4

100 200 300 400 500

Tokens (Billions)

82

84

86

88

90

92

Accuracy(%)

SciQ Acc.

OLMo-7B

OLMo-7B-DHCx4

Figure 6: (1) and (2) Training loss (0.99 EMA smoothed) and C4-en validation loss for OLMo-7B and OLMo-7B-DHC×4 models. (3) and (4) Accuracy curves on hellaswag and sciq, demonstrating the superior performance of the OLMo-7B-DHC×4 model.

We evaluate the effectiveness of hyper-connections on the 7B model, training a model with DHCs with an expansion rate of 4, denoted as OLMo-7B-DHC×4. According to Table 5, OLMo-7B-DHC×4 significantly outperforms the baseline OLMo-7B model in all average metrics. In the V2 evaluation, OLMo-7B-DHC×4 shows improvements of 0.022 for loss and 0.293 for PPL. Furthermore, the average score of downstream benchmarks 0.710 surpasses the baseline 0.701, with the results of specific tasks shown in Fig. 10.

Based on Fig 6, the OLMo-7B-DHC×4 model consistently shows better metrics compared to baseline, including training and validation loss and accuracy in downstream benchmarks. Notably, after 400 B tokens, the model maintains its improvement without the gains diminishing. This indicates that the OLMo-7B-DHC×4 model continues to provide consistent benefits in reducing loss, even at higher token counts. Furthermore, according to Fig. 6, the baseline model exhibits frequent spikes, while our model with DHCs shows no spikes throughout the training. This shows that our approach not only achieves better loss but also ensures more stable training.

- 4.4 MOE MODELS

We evaluate the effectiveness of hyper-connections on the Mixture-of-Experts (MoE) model. We retrain the original OLMoE-1B-7B model as the baseline and train a model that applies Dynamic

- Table 5: Performance of 7B models. FLOPs refers to the computation per token in the forward pass.

Methods

Params (B)

FLOPs (G)

V2 Loss ↓

V2 PPL ↓

V3 Loss ↓

V3 PPL ↓

Tasks Avg. Acc. ↑

OLMo-7B 6.9 13.36 2.581 14.316 2.322 11.324 70.1 OLMo-7B-DHC×4 6.9 13.38 2.559 14.023 2.304 11.120 71.0

Hyper-Connections (DHC) with n = 4, replacing the residual connections. The full results are shown in Fig. 9, which illustrates that hyper-connections outperform residual connections in almost all metrics. In many metrics, our method requires only half of the training tokens to achieve the same performance as the baseline. Fig. 1 and Table 6 highlight some of the results, such as a reduction in training loss of approximately 0.027, a reduction in loss on the C4-en validation set of 0.028, an improvement of 6 points on the ARC-Challengeand an improvement of 1.2 points on MMLU Var.

- Table 6: Downstream evaluations for MoE models training with 500B tokens under the OLMoE evaluation setting. ARC-C stands for ARC-Challenge, and ARC-E for ARC-Easy. MMLU Var is a modified version of MMLU that includes varying few-shot examples, providing stable feedback during early training, as outlined in the OLMoE setting (Muennighoff et al., 2024).

MMLU Var

HellaSwag

WinoGrande

ARC-C ARC-E PIQA

BoolQ

Methods

OLMoE-1B-7B 38.5 69.5 41.8 72.8 77.6 64.4 65.4 OLMoE-1B-7B-DHC×4 39.7 70.2 47.8 76.7 78.2 64.6 68.5

- 4.5 VISUALIZATION ANALYSIS

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

1.0

0

0 4 8

0 4 8

0 4 8

0 4 8

|[Figure 1]<br><br>← PTB|
|---|

|[Figure 2]<br><br>|
|---|

|[Figure 3]<br><br>|
|---|

|[Figure 4]<br><br>|
|---|

|[Figure 5]<br><br>|
|---|

[Figure 6]

- 4 8

0.5

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32

0.0

0.5

1.0

Post-Norm

Pre-Norm

Pre-Norm PTB

Two-hop Residual

Hyper-Connection

- Figure 7: Visualization of connection matrices for hyper-connections and various related baseline methods. The attention layers, which have odd ids, are marked with green tick marks.

In this section, we investigate the learned hyper-connection weights and show how the output of the former layer contributes to the latter ones. To this end, we convert hyper-connections to dense connections cross layers. Consider the input hidden vectors hk0 in k-th layer, it can be unfolded as a weighted summation over previous layer outputs:

k−1

c(0)kj T j(hj0), (20)

hk0 =

j=0

where c(0)kj describes how much layer-j (T j) contributes to layer-k’s input hk0. Then, C(0) denotes a dense connection weight matrix. In particular, let layer-0 be the word embedding and T 0 be an identity mapping, layer-L+1 be the hidden state before the unembedding layer, which is a summation

over the last hidden vectors, i.e., hL0+1 = j hLj .

OLMo-1B-DHC×4 model is adopted for visualization. We take the checkpoint at 500B tokens and forward random validation text to obtain dynamic hyper-connection weights. In addition, we show connection patterns for some related baseline methods. Finally, the visualization is illustrated in Fig. 13. We present the following findings, with more detailed discussions provided in Appendix F.

Connection patterns for baseline methods. For Pre-Norm baseline, the connection matrix is simply a lower triangular matrix with diagonal elements erased, because each transformer layer joins the residual equally. In the Pre-Norm parallel transformer block (PTB) baseline, the connection matrix appears jagged because the input to the FFN layer does not depend on the output of the previous attention layer. For Post-Norm baseline, the connection only holds for adjacent layers, as the weight for bottom layers decays every time the residual passes a post-norm layer. For the two-hop residual baseline (Ma et al., 2024), the outputs of attention layers are not added to residual and only contributes to the next one FFN layer, resulting in a vertical strip pattern in the connection matrix.

Λ-shaped connection pattern. In the connection matrix for hyper-connections, a long-term decay pattern can be observed, where layers are generally preferred to rely on a few adjacent layer outputs. Moreover, the bottom layers (e.g. layer 0,2) are observed frequently used in most of subsequent layers. Therefore, the two patterns together form a Λ-shaped connection pattern. Note that the long-term decay pattern is a Post-Norm style pattern, while the frequently accessed pattern is Pre-Norm style, indicating that the hyper-connection introduces a free mixture of Pre- and Post-Norm architecture.

Input word embedding is eliminated from model output. As per the first column in the connection matrix for layer inputs, the input word embedding contributes to most of the layers except for the final one. This last layer, which products the model’s output, is used for next token prediction. In most cases, keeping a component of input embedding in model output is harmful to next token prediction, especially when using a tied word embedding such as that employed by OLMo-1B. Similar results are found in previous works (Ma et al., 2023).

Parallel transformer blocks are observed. As discussed in § 3.2, parallel transformer block, which performs attention and FFN in parallel, is a special case for hyper-connection. In practice, PTB-like patterns, which can be identified by the local jagged pattern, are surprisingly observed to be learned by hyper-connections. For instance, layer 11 has a minimal contribution to the input of layer 12 (refer to row 12 in the hyper-connection connection matrix). This suggests that layers 11 and 12 can operate in parallel, thereby forming a PTB module.

Attention layers tend to have fewer long-term connections. It is observed that attention layers at the bottom barely have long-term contribution, a trend that persists until layer 17. Upon examining the connection matrix for hyper hiddens (refer to Fig. 13 in the appendix), it’s evident that the outputs of the FFN layers have significantly greater magnitudes than those of the attention layers. This pattern resembles a two-hop residual connection design, wherein the attention output contributes to the input of the following FFN layer, but doesn’t join the main residual path.

- 5 RELATED WORK

Transformers (Vaswani et al., 2017) have revolutionized various fields, particularly natural language processing and computer vision. They rely heavily on residual connections to facilitate the training of deep models. Our hyper-connections approach can replace residual connections, providing stable training and consistent improvements in both natural language processing and computer vision.

The issues of gradient vanishing and representation collapse (Bengio et al., 1994; Glorot & Bengio, 2010; Liu et al., 2020) have been extensively studied. The combinations of normalization techniques (Ioffe & Szegedy, 2015; Ba et al., 2016) and residual connections (He et al., 2016), like Pre-Norm and Post-Norm, actually reflects different emphases in solving these two issues. However, despite these advancements, the fundamental trade-off between gradient vanishing and representation collapse in deep networks remains a critical challenge. Building on these findings, our work introduces a novel approach that enables neural networks to autonomously learn the optimal strength of connections, potentially improving both gradient stability and representation quality.

- 6 CONCLUSION

In conclusion, we have introduced hyper-connections as an effective alternative to residual connections in transformers. Our analysis reveals that hyper-connections not only overcome the limitations of residuals but also enable dynamic adjustments in network architecture. Experimental results confirm their promising benefits across various tasks, including pre-training of large language model, image generation, and image classification.

ACKNOWLEDGEMENTS

This research was conducted at ByteDance Inc. We are grateful for the suggestions and assistance provided by Yaowei Zheng, Yuyu Zhang, Yunshui Li, Xiang Li, Bairen Yi, Zhenyi Lu and Xintian Han.

REFERENCES

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. In arXiv preprint arXiv:1607.06450, 2016.

Cenk Baykal, Dylan Cutler, Nishanth Dikkala, Nikhil Ghosh, Rina Panigrahy, and Xin Wang. Alternating updates for efficient transformers. Advances in Neural Information Processing Systems, 36, 2024.

Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE transactions on neural networks, 5(2), 1994.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 2020.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In Machine learning challenges workshop. Springer, 2005.

Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. In proceedings of Sinn und Bedeutung, volume 23, 2019.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.

Bill Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Third international workshop on paraphrasing (IWP2005), 2005.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning. PMLR, 2015.

Matt Gardner Johannes Welbl, Nelson F. Liu. Crowdsourcing multiple choice science questions. 2017.

Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. arXiv preprint arXiv:2205.05198, 2022.

Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers. arXiv preprint arXiv:2004.08249, 2020.

Haoyan Ma, Xiang Li, Xia Yuan, and Chunxia Zhao. Denseformer: A dense transformer framework for person re-identification. IET Computer Vision, 17(5), 2023.

Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, and Chunting Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length. arXiv preprint arXiv:2404.08801, 2024.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2409.02060.

William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.

Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI spring symposium series, 2011.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9), 2021.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

N Shazeer, A Mirhoseini, K Maziarz, A Davis, Q Le, G Hinton, and J Dean. The sparsely-gated mixture-of-experts layer. Outrageously large neural networks, 2017.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, 2013.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159, 2024.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, 2017.

Ben Wang. Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.

Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, et al. Small-scale proxies for large-scale transformer training instabilities. arXiv preprint arXiv:2309.14322, 2023.

Shufang Xie, Huishuai Zhang, Junliang Guo, Xu Tan, Jiang Bian, Hany Hassan Awadalla, Arul Menezes, Tao Qin, and Rui Yan. Residual: Transformer with dual residual connections. arXiv preprint arXiv:2304.14802, 2023.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

- A TRANSFORMER WITH HYPER-CONNECTIONS

||Attention|
|---|
<br><br>+ +<br><br>+<br><br>(b) Transformer with Hyper-Connections<br><br>|ℎ|
|---|
<br><br>Repeat<br><br>|Attention|
|---|
<br><br>|FFN|
|---|
<br><br>|Attention|
|---|
<br><br>|FFN|
|---|
<br><br>+<br><br>+<br><br>+<br><br>+<br><br>(b) Transformer with Residual Connections<br><br>|FFN|
|---|
<br><br>+ +<br><br>+<br><br>|Attention|
|---|
<br><br>+ +<br><br>+<br><br>|FFN|
|---|
<br><br>+ +<br><br>+<br><br>+<br><br>𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>𝛼 ,  𝛼 ,  𝛼 ,  𝛼 ,  𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>𝛼 ,  𝛼 ,  𝛼 ,  𝛼 ,  𝛼 , <br><br>𝛽<br><br>|h|
|---|
<br><br>𝛼 , <br><br>𝛽<br><br>|h|
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
|
|---|

- Figure 8: Comparison between transformers with hyper-connections and that with residual connections.

- B PARAMETERS, COMPUTATION AND MEMORY FOOTPRINT ANALYSIS

Static Hyper-Connections. All learnable parameters are included in the hyper-connection matrix HC in Eq. 1. The number of parameters in one HC is given by:

|θSHC| = |θB| + |θA| = n + n · (n + 1) = n · (n + 2), (21)

where n is the expansion rate, |θB| is the number of parameters in B in SHC, and |θA| is the number of parameters in A. Each layer contains two hyper-connection modules (one for the self attention and one for the feedforward network). Thus, the number of extra parameters is:

Pextra = |θSHC| × 2 × L, (22)

where L is the number of layers. For example, in OLMo-1B-SHC×4, Pextra = 4×(4+2)×2×16 = 768.

Dynamic Hyper-Connections. The parameters of DHC are defined in Eqs. 10, 11, 12, and 13, and the number of parameters is given by:

r| (23) = |θnorm| + 1 + dmodel + n + 1 + dmodel + n + dmodel × n + n × n (24) = |θnorm| + dmodel × (n + 2) + n × (n + 2) + 2, (25)

|θDHC| = |θnorm| + |sβ| + |θW

β| + |θB| + |sα| + |θW

m| + |θA

m| + |θW

r| + |θA

where dmodel is the dimension of the hidden states in the transformer, and |θnorm| depends on the type of normalization module. In OLMo models, there are no parameters for normalization, so

|θnorm| = 0. In OLMoE, |θnorm| = dmodel. Similar to the static hyper-connections, the number of extra parameters is:

Pextra = |θDHC| × 2 × L, (26)

For example, for OLMo-1B-DHC×4, Pextra = (0+2048×(4+2)+4×(4+2)+2)×2×16 = 394,048.

The number of parameters for DHC and SHC used in the experiments is detailed in Table 7, while their corresponding FLOPs comparisons are provided in Table 8. Regardless of whether SHC or DHC is used, the additional parameters and computational overhead introduced are minimal and can be considered negligible.

Table 7: Comparison of number of parameters.

#### Method HC Params(B) Total Params(B) Total Params ∆ rate (%)

OLMo-1B - 1.17676442 OLMo-1B-SHC×2 0.0000026 1.17676467 +0.00002% OLMo-1B-SHC×4 0.0000077 1.17676518 +0.00007% OLMo-1B-DHC×2 0.0002625 1.17702688 +0.02230% OLMo-1B-DHC×4 0.0003940 1.17715846 +0.03349%

OLMo-7B - 6.88809574 OLMo-7B-DHC×4 0.0013124 6.88967027 +0.02286%

OLMoE-1B-7B - 6.91909427 OLMoE-1B-7B-DHC×4 0.0003940 6.91948832 +0.00570%

Computation Analysis. The main computational cost of SHC and DHC lies in line 5 of Algorithm 1, where the complexity is O(dmodel × n × (n + 1)). The computational cost of the FFN is O(2 × dmodel × dffn), and that of the projection part of attention is O(4 × dmodel × dmodel). Since O(dmodel × n × (n + 1)) ≪ O(4 × dmodel × dmodel) < O(2 × dmodel × dffn), the computational cost of HC is negligible compared to the cost of both FFN and the attention projection part. Here, dffn is the inner dimension of the FFN. The detailed computation cost statistics are presented in Table 8.

Table 8: FLOPs per token in forward pass.

#### Method HC FLOPs (G) Total FLOPs (G) Total FLOPs ∆ rate (%)

OLMo-1B - 2.3536 OLMo-1B-SHC×2 0.0010 2.3545 +0.038% OLMo-1B-SHC×4 0.0031 2.3566 +0.127% OLMo-1B-DHC×2 0.0020 2.3554 +0.076% OLMo-1B-DHC×4 0.0049 2.3583 +0.200%

OLMo-7B - 13.3647 OLMo-7B-DHC×4 0.0197 13.3844 +0.147%

OLMoE-1B-7B - 2.3580 OLMoE-1B-7B-DHC×4 0.0049 2.3629 +0.208%

Memory Footprint. The introduction of HC results in a minor increase in activation memory usage during training. For a transformer model with L layers, a model dimension of dmodel, batch size b, sequence length s, and number of attention heads a, the activation memory is calculated

- as sbdmodelL(34 + 5as/dmodel), as outlined in Korthikanti et al. (2022). Incorporating HC with an expansion rate of n adds an extra memory overhead of 2nsbdmodelL. For n = 2, this contributes less than 15% to the total memory usage of a standard transformer. Notably, the memory consumption is mostly driven by the weight parameters, which experience only a slight increase with HC. Additionally, given HC’s low computational cost, the hidden states generated by HC can be discarded post forward pass and recomputed during backpropagation to further optimize memory usage. With this approach,

the additional memory requirement is reduced to nsbdmodel. During inference, the memory usage for activations is largely determined by the Key-Value cache, which is not impacted by the extra activations brought by HC. Moreover, the hidden states from earlier layers can be released as soon as the next layer’s computations start, significantly lowering memory requirements. The actual memory footprint is empirically measured on 8 GPUs, as shown in Table 9.

Table 9: Measured Memory Footprint on 8 GPUs.

Micro Batch Size (tokens per GPU)

Method Memory (GB) Memory ∆ Rate (%)

OLMo-1B 41.11 - 16,384 OLMo-1B-SHC×2 47.55 +15.7% 16,384 OLMo-1B-SHC×4 51.85 +26.0% 16,384 OLMo-1B-DHC×2 47.56 +15.7% 16,384 OLMo-1B-DHC×4 51.86 +26.1% 16,384

OLMo-7B 26.27 - 2,048 OLMo-7B-DHC×4 33.70 +28.28% 2,048

OLMoE-1B-7B 31.59 - 4,096 OLMoE-1B-7B-DHC×4 34.65 +9.7% 4,096

### C MOE 1B/7B MODEL EXPERIMENTS

###### training loss

###### C4 en val. loss

###### dolma books val. loss

###### dolma cc val. loss

2.5

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

2.90

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

2.7

2.8

2.85

2.4

2.6

2.80

2.7

2.3

2.75

2.5

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### dolma pes2o val. loss

###### dolma reddit val. loss

###### dolma stack val. loss

###### dolma wiki val. loss

2.30

3.05

1.10

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

2.25

2.4

3.00

1.05

2.20

2.95

2.3

2.15

1.00

2.90

2.10

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### ice val. loss

###### m2d2-s2orc val. loss

###### pile val. loss

###### wikitext 103 val. loss

2.20

3.20

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

2.8

2.5

2.15

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

3.15

2.7

2.10

3.10

2.4

2.6

2.05

3.05

2.5

2.3

2.00

3.00

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### MMLU stem Var Acc. (%)

###### MMLU hum. Var Acc. (%)

###### MMLU soc. sci. Var Acc. (%)

###### MMLU other Var Acc. (%)

42.5

34

50

34

40.0

32

45

37.5

30

32

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

35.0

28

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

40

30

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### MMLU avg. Acc. (%)

###### HellaSwag Acc. (%)

###### ARC Challenge Acc. (%)

###### SciQ Acc. (%)

94

40

70

OLMoE-1B-7B

45

OLMoE-1B-7B-DHCx4

38

92

65

40

36

90

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

60

34

35

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

88

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### ARC easy Acc. (%)

###### PIQA Acc. (%)

###### WinoGrande Acc. (%)

###### Openbook QA Acc. (%)

44

78

65.0

42

75

62.5

40

76

70

60.0

38

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

74

36

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

57.5

65

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

###### BoolQ Acc. (%)

###### COPA Acc. (%)

###### Commonsense QA Acc. (%)

- 44

- 45

- 46

- 47

- 48 Social Iqa Acc. (%)

87.5

50.0

OLMoE-1B-7B

OLMoE-1B-7B-DHCx4

85.0

47.5

65

82.5

45.0

80.0

OLMoE-1B-7B

OLMoE-1B-7B

OLMoE-1B-7B

60

42.5

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

OLMoE-1B-7B-DHCx4

77.5

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

Tokens (B)

- Figure 9: Loss curves in V3 validation sets and accuracy curves on downstream tasks for OLMoE-1B7B and OLMoE-1B7B-DHC×4 models.

### D 7B MODEL EXPERIMENTS

###### c4 en val. loss

###### dolma books val. loss

###### dolma cc val. loss

OLMo-7B-

OLMo-7B-

OLMo-7B-

2.7

2.9

2.7

OLMo-7B-DHCx4

OLMo-7B-DHCx4

OLMo-7B-DHCx4

2.8

2.6

2.6

2.7

2.5

2.5

2.6

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

###### dolma pes2o val. loss

###### dolma reddit val. loss

###### dolma stack val. loss

3.0

OLMo-7B-

OLMo-7B-

OLMo-7B-

1.05

OLMo-7B-DHCx4

OLMo-7B-DHCx4

OLMo-7B-DHCx4

2.3

1.00

2.9

2.2

0.95

2.8

2.1

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

###### dolma wiki val. loss

###### ice val. loss

###### m2d2-s2orc val. loss

2.5

OLMo-7B-

OLMo-7B-

OLMo-7B-

3.2

2.8

OLMo-7B-DHCx4

OLMo-7B-DHCx4

OLMo-7B-DHCx4

2.4

2.7

3.1

2.6

2.3

3.0

2.5

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

###### HellaSwag Acc. (%)

###### pile val. loss

###### wikitext 103 val. loss

70

2.7

OLMo-7B-

OLMo-7B-

OLMo-7B-DHCx4

OLMo-7B-DHCx4

2.2

2.6

65

2.5

60

2.1

OLMo-7B-

2.4

OLMo-7B-DHCx4

55

2.0

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

###### SciQ Acc. (%)

###### COPA Acc. (%)

###### Openbook QA Acc. (%)

92.5

85

40.0

90.0

37.5

80

87.5

35.0

75

85.0

OLMo-7B-

OLMo-7B-

OLMo-7B-

OLMo-7B-DHCx4

OLMo-7B-DHCx4

OLMo-7B-DHCx4

32.5

82.5

70

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

###### ARC-Easy Acc. (%)

###### PIQA Acc. (%)

###### WinoGrande Acc. (%)

65

70

78

60

65

76

74

60

OLMo-7B-

OLMo-7B-

OLMo-7B-

55

OLMo-7B-DHCx4

OLMo-7B-DHCx4

OLMo-7B-DHCx4

72

55

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

Tokens (B)

Tokens (B)

Tokens (B)

- Figure 10: Loss curves in V3 validation set and accuracy curves on downstream tasks for OLMo-7B and OLMo-7B-DHC×4 models.

- E VISION EXPERIMENTS

Datasets. We use the ILSVRC-2012 ImageNet dataset (Deng et al., 2009) with 1k classes and 1.3M images (see ImageNet in the following) for image generation and classification.

- E.1 IMAGE GENERATION

To investigate the generalizability of hyper-connections in image generation, our experiments are conducted using the DiT framework (Peebles & Xie, 2022) training the models for 1400 epochs. In order to save experimental costs, we use FP16 precision, introduce flash-attention to speed up training, and introduce QK-Norm (Wortsman et al., 2023) to stabilize training.

- Table 10: Benchmarking class-conditional image generation on ImageNet 256×256, with cfg=1.50. NP, P, and R are short for Numerical Precision, Precision, and Recall, respectively.

Method NP QK-Norm Size (M) FID↓ sFID↓ IS↑ P↑ R↑

DiT-XL/2 FP32 ✗ 675 2.27 4.60 278.24 0.83 0.57 DiT-XL/2 FP16 ✓ 675 2.36 4.54 269.46 0.83 0.58 DiT-1B/2 FP16 ✓ 983 2.13 4.50 288.69 0.82 0.59 DiT-XL/2-SHC×2 FP16 ✓ 675 2.18 4.52 287.24 0.82 0.60

Our experimental results demonstrate that DiT models incorporating hyper-connections exhibit comparable performance metrics to DiT models with 50% more parameters. This finding underscores the efficiency and efficacy of hyper-connections in enhancing model performance without increasing model size.

E.2 IMAGE CLASSIFICATION

For the image classification experiments, we train ViT/16-Base and ViT/16-Large models with images at a resolution of 224 × 224 for 300 epochs, following the experimental setup used by (Dosovitskiy et al., 2020).To speed up the training process, we use bfloat16 numerical precision. The training configuration is detailed in Table 12. Within this configuration, we replace the residual connections with static and dynamic hyper-connections, referred to as SHC and DHC, respectively, using an expansion rate of n = 2. The top-1 accuracy results are presented in Table 11, and the training loss curves for ViT/16-Large and ViT/16-Large with DHC×2 are shown in Fig. 11.

For the Base model (85M), our re-implemented ViT/16 achieves 76.38% accuracy on 224 × 224 images. The SHC and DHC enhance performance to 77.60% and 77.26%, respectively. representing relative increases of 1.22% and 0.88%. For the Large model (307M parameters), ViT/16 achieves 77.25% accuracy. The SHC and DHC configurations further enhance accuracy to 78.38% and 79.94%, respectively. This corresponds to relative improvements of 1.13% and 2.69%, with DHC showing the highest performance. These results demonstrate that hyper-connections (SHC and DHC) significantly improve accuracy, especially in the Large model scale.

- Table 11: Accuracy on ImageNet. ViT*/16 refers to the results reported by (Dosovitskiy et al., 2020), whereas ViT/16 denotes our re-implemented baseline. SHC and DHC indicate that residual connections are replaced with static and dynamic hyper-connections, respectively.

ViT*/16 ViT/16 ViT/16-SHC×2 ViT/16-DHC×2 384 × 384 224 × 224

Model Scales Params (M)

Base 85 77.91 76.38 77.60 77.26 Large 307 76.53 77.25 78.38 79.94

###### Training Loss

1.8

ViT/16-Lagre

ViT/16-Lagre-DHCx2

1.6

1.4

1.2

Loss

1.0

0.8

0.6

0.4

0.2

30000 40000 50000 60000 70000 80000 90000

Steps

- Figure 11: Training loss curves of ViT/16-Large and ViT/16-Large-DHC×2, smoothed using an Exponential Moving Average (EMA) with a decay rate of 0.999. The gain from Hyper-Connections decreases as training progresses, likely due to pass over the same dataset across many epochs, resulting in diminishing returns from the additional capacity provided by Hyper-Connections.

- E.3 VISULIZATION OF DHC

We randomly select three categories from the ImageNet dataset and sample the corresponding examples from the validation set. These samples are fed into the ViT-Base/16-DHC×2 model to compute the dynamic connection weights of the DHC in the final layer. As shown in Fig. 12, we visualize the distribution of these weights. We observe that the intra-class distribution of beta is highly concentrated, indicating that samples within the same category tend to have similar beta values. In contrast, the distribution of alpha is less concentrated, but the differences between the distributions of different categories are more pronounced, as exemplified by α2,0.

Table 12: Training hyperparameters for ViT.

Hyperparameter Value Learning Rate (lr) 0.003 Batch Size 4096 Scheduler Cosine Annealing with Linear Warmup (10k steps) Data Augmentation Mixup (α = 0.2) Epochs 300 Optimizer AdamW (β1 = 0.9, β2 = 0.999, ϵ = 1e − 8) Gradient Clipping 1.0 Weight Decay 0.3 Dropout 0.1 Precision bf16

###### 33:loggerhead turtle

###### 998:capitulum

###### 779:school bus

40

30

40

30

Frequency

Frequency

Frequency

30

20

20

20

10

10

10

0

0

0

0.95 1.00 1.05 1.10 1.15 1.20

0.95 1.00 1.05 1.10 1.15 1.20

0.95 1.00 1.05 1.10 1.15 1.20

1

1

1

###### 33:loggerhead turtle

###### 998:capitulum

###### 779:school bus

30

50

25

25

40

20

20

Frequency

Frequency

Frequency

30

15

15

20

10

- 10

10

5

5

0

0

0

0.95 1.00 1.05 1.10 1.15 1.20

0.95 1.00 1.05 1.10 1.15 1.20

0.95 1.00 1.05 1.10 1.15 1.20

2

2

2

###### 33:loggerhead turtle

###### 998:capitulum

###### 779:school bus

- 0

- 1

- 2

- 3

- 4

- 5

30

8

25

6

Frequency

Frequency

Frequency

20

15

4

10

2

5

0

0

0.65 0.60 0.55 0.50 0.45 0.40 0.35

0.65 0.60 0.55 0.50 0.45 0.40 0.35

0.65 0.60 0.55 0.50 0.45 0.40 0.35

1, 0

1, 0

1, 0

###### 33:loggerhead turtle

###### 998:capitulum

###### 779:school bus

- 0

- 1

- 2

- 3

- 4

- 5

- 0

- 1

- 2

- 3

- 4

50

40

Frequency

Frequency

Frequency

30

20

10

0

0.9 1.0 1.1 1.2 1.3

0.9 1.0 1.1 1.2 1.3

0.9 1.0 1.1 1.2 1.3

1, 1

1, 1

1, 1

###### 33:loggerhead turtle

###### 998:capitulum

###### 779:school bus

- 0

- 1

- 2

- 3

- 4

- 5

- 0

- 1

- 2

- 3

- 4

50 Frequency

40

Frequency

Frequency

30

20

10

0

0.1 0.0 0.1 0.2 0.3

0.1 0.0 0.1 0.2 0.3

0.1 0.0 0.1 0.2 0.3

1, 2

1, 2

1, 2

###### 33:loggerhead turtle

###### 998:capitulum

###### 779:school bus

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 0

- 1

- 2

- 3

- 4

15

Frequency

Frequency

Frequency

10

5

0

2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40

2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40

2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40

2, 0

2, 0

2, 0

###### 33:loggerhead turtle

###### 998:capitulum

779:school bus

- 0

- 1

- 2

- 3

- 4

- 5

50

6

40

Frequency

Frequency

Frequency

30

4

20

2

10

0

0

0.2 0.1 0.0 0.1 0.2

0.2 0.1 0.0 0.1 0.2

-0.20 -0.10 0.00 0.10 0.20

2, 1

2, 1

2, 1

Figure 12: Distribution of weights of last DHC in ViT-Base/16-DHC×2 model.

- F MORE VISUALIZATION AND ANALYSIS

Unfolding hyper-connections. We first introduce how to determine the connection matrix C(0) for hyper-connections. To simplify writing, the layer output T k(hk0) is denoted by T k for short. The

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

1.0

0 4 8

0 4 8

0 4 8

0 4 8

0 4 8

|[Figure 7]<br><br>C(0)|
|---|

|[Figure 8]<br><br>C(1)|
|---|

|[Figure 9]<br><br>C(2)|
|---|

|[Figure 10]<br><br>C(3)|
|---|

|[Figure 11]<br><br>C(4)|
|---|

[Figure 12]

0.5

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32 1.0

0.0

0.5

(a) Connection matrix for DHC model.

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

1.0

0 4 8

0 4 8

0 4 8

0 4 8

0 4 8

|[Figure 13]<br><br>C(0)|
|---|

|[Figure 14]<br><br>C(1)|
|---|

|[Figure 15]<br><br>C(2)|
|---|

|[Figure 16]<br><br>C(3)|
|---|

|[Figure 17]<br><br>C(4)|
|---|

[Figure 18]

0.5

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32 1.0

0.0

0.5

(b) Connection matrix for SHC model.

#### Figure 13: Visualization of unfolded connection matrix. Matrices from left to right are

C(0)(Connections for {hj0}Lj=0+1), C(i) (Connections for {h′ji}Lj=0+1) for i ∈ {1,2,3,4}. The attention layers, which have odd ids, are marked with green tick marks.

recurrent form of hyper connection in Eq. 2 is expanded as follows:

h0k =Hk⊺Amk = (T k−1Bk−1 + Hk−1⊺Ark−1)Amk

k−1

T jBj(Arj+1Arj+2...Ark−1)Amk

=

j=0

k−1

k−1

#### Art)Amk. (27)

T jBj(

=

j=0

t=j+1

Therefore, we obtain connection matrix c(0)kj = Bj( kt=−j1+1 Art)Amk. Similarly, the connection matrix C(i) for the i-th hyper hidden from k-th layer can be computed by substituting the last Amk with Ark in Eq. 27, i.e.,

k−1

k

H′k = Ark⊺Hk =

Art)⊺Bj⊺T j⊺ (28)

(

j=0

t=j+1

 (

 

⊺

k

Bj⊺

c(kji) =

Art)

. (29)

t=j+1

i

Visualization for hyper hidden. We visualize connection matrices for hyper hiddens in Fig. 13 to reveal how hyper-connection maintains intermediate layer outputs. First of all, the four hyper hiddens are dissimilar and show completely different connection patterns. Then, we can see outputs from FFN layers are preserved long-termly in hyper hiddens, while attention layers are reserved less. It is also observed that the long-term connections are usually stored in pairs of hyper hiddens, where the connection is positive in one hyper hidden but negative in the other, for example, column 0 and 2 in C(1),C(3). With such strategy, these connections can be easily eliminated in the sum-pooling operation before the unembedding layer.

SHC shares similar connection pattern with DHC. We show the connection matrices for OLMo-1B-SHC×4 model in Fig. 13b. Comparing to DHC, as shown in Fig. 13a, SHC shares exactly the same connection patterns. Moreover, we observe many more PTB-like blocks in SHC, e.g., layers from 13 to 18. Note that the connection relation for SHC is token independent, and such PTB-like blocks can be physically reorganized to be parallelly computed.

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

0 4 8 12 16 20 24 28 32

1.00

1.00

1.00

0 4 8

0 4 8

0 4 8

|[Figure 19]<br><br>↓ wasted|
|---|

|[Figure 20]<br><br>|
|---|

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

0.75

0.75

0.75

0.50

0.50

0.50

12 16 20 24 28 32

12 16 20 24 28 32

12 16 20 24 28 32

0.25

0.25

0.25

0.00

0.00

0.00

0.25

0.25

0.25

0.50

0.50

0.50

0.75

0.75

0.75

1.00

1.00

1.00

(a) OLMo-1B-DHC×1

(b) OLMo-1B-DHC×2

(c) OLMo-1B-DHC×4

- Figure 14: Comparison of unfolded connection matrices for OLMo-1B-DHC×1, OLMo-1B-DHC×2 and OLMo-1B-DHC×4 model.

How HC×1 fails. The OLMo-1B×1 model is observed to perform worse than baseline in our experiments. Its connection matrix is visualized in Fig. 14 to show how it fails. Above all, we observe that layer 17 is wasted, who has no connection to subsequent layers at all. Secondly, compared to HC×2 and HC×4 models, the Λ shaped pattern does not appear. Note that HC×1 does not support the pattern of Λ in its mathematical formulation, where the connections to previous layers must be weakened or strengthened simultaneously. Thus, the lack of connection from the early layers to the final layers may suffer from gradient vanishing, like post-norm style transformers, which leads to performance degeneration.

- G DERIVATION OF NON-TRAINABLE HYPER-CONNECTION MATRIX FOR RESIDUAL CONNECTIONS

- G.1 PRE-NORM RESIDUAL CONNECTION

In the Pre-Norm residual connection, the input to a layer is first normalized before being passed through the layer. The output of the layer is then added to the original input. This can be represented as:

hˆ = T (Norm(h)) + h. (30)

By incorporating the normalization operator into the layer, T := T ◦Norm, we can express the entire process as:

hˆ = T (h) + h. (31) To express this using hyper-connections, the matrix for Pre-Norm can be structured as follows:

HCPreNorm =

- 0 1
- 1 1

(32)

Given hyper hidden matrix H = h⊺, we prove that the output of HCPreNorm Hˆ = hˆ⊺. Proof.

Hˆ = HC(T ,H)

= B⊺T (H⊺Am)⊺ + Ar⊺H

= T (h)⊺ + h⊺

= hˆ⊺.

(33)

| |
|---|

- G.2 POST-NORM RESIDUAL CONNECTION

In the Post-Norm residual connection, the input to a layer is passed through the layer first, and then the output is normalized after being added to the original input. In matrix form, this can be represented as:

h′ = T (h) (34) The summation of the input and the normalized output of the layer is:

hˆ = Norm(h + h′) (35)

We consider Norm to be LayerNorm (Zhang & Sennrich, 2019). The analysis process for RMSNorm is almost identical. In fact, the affine transformation can be incorporated into the subsequent layer, while the mean subtraction operation can be integrated into the current layer.

T = C ◦ T ◦ A, (36) where A is the affine transformation, and C is the re-centering operator. Thus, the mean of the output of T is 0.

To express this using hyper-connections with an expansion rate n = 1, we need a hyper-connection matrix HC that encapsulates this operation:

 

  =

- 0 √ 1 σh2+σh2′+2σhh′

- 1 √ 1 σh2+σh2′+2σhh′

0 B Am Ar

. (37)

HCPostNorm =

Similar to the previous proof, we prove that the output of HCPostNorm is equivalent to the transpose of the output of the Post-Norm residual connection:

Hˆ = hˆ⊺. (38) Proof. Note that

σh+h′ = σh2 + σh2′ + 2σhh′. (39) Given this fact, we can derive the Post-Norm:

hˆ = Norm(h′ + h)

h′ + h − µh′+h σh+h′

=

1 σh′+h

(h′ + h)

=

1 σh2 + σh2′ + 2σhh′

(h′ + h)

=

For hyper-connections side, we have:

#### Hˆ = B⊺h′⊺ + H′

= B⊺h′⊺ + ArH = B⊺h′⊺ + Arh⊺ =

1 σh2 + σh2′ + 2σhh′

h′⊺ +

1 σh2 + σh2′ + 2σhh′

h⊺ = hˆ⊺.

(40)

(41)

| |
|---|

- H SEQUENTIAL-PARALLEL DUALITY

- H.1 HYPER-CONNECTION MATRIX OF SEQUENTIAL ARRANGEMENT

In this section, we demonstrate that the following hyper-connection matrix will produce n identical networks arranged sequentially with residual connections between them:

01×1 11×n e1 en×n

, (42)

HC =

where en×n denotes an n × n identity matrix, ei ∈ Rn×1 represents the i-th column of en×n, and

- 11×n signifies a 1 × n matrix of ones.

We will use mathematical induction to prove that hki = hkj and hki+1 = T k(hki ) + hki , ∀i,j ∈ {0,1,...,n}, ∀k ∈ {0,1,...,L}, where L is the number of layers.

Proof. BASE CASE For k = 0, we have the initial condition h0i = h0j, ∀i,j ∈ {0,1,...,n}, as we define H0 =

##### h0 h0 ... h0 ⊺ ∈ Rn×d.

INDUCTION HYPOTHESIS

##### Assume that for some k ∈ {1,...,L − 1}, we have hki = hkj and hki = T k(hki−1) + hki−1, ∀i,j ∈ {0,1,...,n}.

INDUCTION STEP We have

Hk+1 = HC(T k,Hk) (43) = B⊺(h′0k)⊺ + H′k (44) = B⊺Am⊺Hk + Ar⊺Hk (45) = 1n×1T k(e⊺1Hk) + en×nHk (46) = T k(hk1) T k(hk1) ... T k(hk1) ⊺ + hk1 hk2 ... hkn ⊺ (47) = T k(hk1) + hk1 T k(hk1) + hk2 ... T k(hk1) + hkn ⊺ (48) = hk1+1 hk2+1 ... hkn+1 ⊺ (49)

Since hki = hkj, ∀i,j ∈ {0,1,...,n}, it follows that T k(hk1) + hki = T k(hk1) + hkj. Thus, we have

hki+1 = hkj+1 (50)

Since hki = hkj, ∀i,j ∈ {0,1,...,n}, it follows that hk1 = hki , ∀i ∈ {0,1,...,n}. Thus, we have

hki+1 = T k(hk1) + hki (51) = T k(hki ) + hki (52)

| |
|---|

- H.2 HYPER-CONNECTION MATRIX OF PARALLEL ARRANGEMENT

In this section, we demonstrate that the following hyper-connection matrix will produce a network where every n adjacent layers are arranged in parallel, with each layer incorporating residual connections. We define a parallel-arranged network such that n adjacent layers form a group, with layers within a group being parallel and groups arranged sequentially. The output of k-th group is given by:

hk+1 =

n

(T k×n+i(hk) + hk). (53)

i=1

It can be proved that this arrangement can be described by the following hyper-connection matrices. First, for k where k − 1 ≡ 0 (mod n):

01×1 e⊺1 1n×1 1n×n,

HC{k|k−1≡0 (mod n)} =

(54)

where the HC matrix can be decomposed into two operations: 1) sum up all the outputs of the previous group and use it as the input of the current layer and as the residual of the subsequent layers; 2) sum up the output and input saving to the first hidden vector slot.

##### Next, for k where k − 1 ≡ i (mod n) and i ̸= 0:

01×1 e⊺i ei en×n,

HC{k|k−1≡i (mod n),i̸=0} =

. (55)

where the HC matrix selects the i-th hidden vector as the input of the current layer, and sums up the output and input, saving to the i-th hidden vector slot.

This means:

hk+1 =HC(k+1)×n(T (k+1)×n, (56) HC(k+1)×n−1(T (k+1)×n−1, (57) ··· (58) HCk×n+1(T k×n+1,hk))) (59)

This can also be proved by mathematical induction; however, the conclusion is quite obvious through drawing, and the proof process is very tedious. Therefore, we don’t repeat the similar proof here.

- I PSEUDOCODE OF HYPER-CONNECTIONS

- Algorithm 1 Network with Hyper-Connections

Require: Initial hidden vector h0 ∈ Rd Require: Expansion rate n Ensure: Final output y

- 1: Initialize:
- 2: H0 ← h0 h0 ... h0 ⊺ ∈ Rn×d
- 3: for k = 1 to L do ▷ For each layer
- 4: H ← Hk−1
- 5: (h0 H′) ← WCk⊺H ▷ Width Connections
- 6: h′0 ← T k(h0) ▷ Layer Computation
- 7: Hˆ ← Bk⊺h′0 + H′ ▷ Depth Connections
- 8: Hk ← Hˆ
- 9: end for
- 10: Final Output:
- 11: hL ← sum rows of HL
- 12: hL ← Normalization Layer(hL)
- 13: y ← Output Layer(hL)
- 14: return y

- J PYTORCH IMPLEMENTATION OF HYPER-CONNECTIONS

- Algorithm 2 Pseudocode of hyper-connections in a PyTorch-like style. # h: hyper hidden matrix (BxLxNxD) class HyperConnection(nn.Module):

def __init__(self, dim, rate, layer_id, dynamic, device=None): super(HyperConnection, self).__init__() self.rate = rate self.layer_id = layer_id self.dynamic = dynamic self.static_beta = nn.Parameter(torch.ones((rate,), device=device)) init_alpha0 = torch.zeros((rate, 1), device=device) init_alpha0[layer_id % rate, 0] = 1. self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye((rate), device=

device)], dim=1))

if self.dynamic: self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate+1), device=device)) self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01) self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim, ), device=device)) self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01) self.layer_norm = LayerNorm(dim)

def width_connection(self, h): # get alpha and beta if self.dynamic:

norm_h = self.layer_norm(h)

if self.dynamic: wc_weight = norm_h @ self.dynamic_alpha_fn wc_weight = F.tanh(wc_weight) dynamic_alpha = wc_weight * self.dynamic_alpha_scale alpha = dynamic_alpha + self.static_alpha[None, None, ...]

else: alpha = self.static_alpha[None, None, ...]

if self.dynamic: dc_weight = norm_h @ self.dynamic_beta_fn dc_weight = F.tanh(dc_weight) dynamic_beta = dc_weight * self.dynamic_beta_scale beta = dynamic_beta + self.static_beta[None, None, ...]

else:

beta = self.static_beta[None, None, ...] # width connection mix_h = alpha.transpose(-1, -2) @ h return mix_h, beta

def depth_connection(self, mix_h, h_o, beta): h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :] return h

- Algorithm 3 Pseudocode of transformer with hyper-connections in a PyTorch-like style.

# h: hyper hidden matrix (BxLxNxD) # atten_hyper_connection, ffn_hyper_connection: hyper-connection modules # attn_norm, ffn_norm: normalization modules

# Attention Block mix_h, beta = atten_hyper_connection.width_connection(h) h = attn_norm(mix_h[...,0,:]) h = self_attention(h) h = atten_hyper_connection.depth_connection(mix_h, dropout(h), beta)

# FFN Block mix_h, beta = ffn_hyper_connection.width_connection(h) h = ffn_norm(mix_h[...,0,:]) h = ffn(h) h = ffn_hyper_connection.depth_connection(mix_h, dropout(h), beta)

- K VALIDATION SETS AND DOWNSTREAM TASKS

Table 13: OLMo’s default configuration was evaluated using multiple metrics. Perplexity (PPL) and loss were used for the V2 and V3 Validation Sets, while zero-shot testing was applied to the Downstream Benchmarks. However, the grey benchmarks were excluded from our analysis due to the instability of their performance indicators.

|V2 Validation Sets|
|---|
|v2-small-4chan-validation v2-small-c4_100_domains-validation v2-small-c4_en-validation v2-small-gab-validation v2-small-ice-validation v2-small-m2d2_s2orc-validation v2-small-m2d2_wiki-validation v2-small-manosphere-validation v2-small-mc4_en-validation v2-small-pile-validation v2-small-ptb-validation v2-small-twitterAEE-validation v2-small-wikitext_103-validation<br><br>|
|V3 Validation Sets|
|v3-small-c4_en-validation v3-small-dolma_books-validation v3-small-dolma_common-crawl-validation v3-small-dolma_pes2o-validation v3-small-dolma_reddit-validation v3-small-dolma_stack-validation v3-small-dolma_wiki-validation v3-small-ice-validation v3-small-m2d2_s2orc-validation v3-small-pile-validation v3-small-wikitext_103-validation<br><br>|
|Downstream Benchmarks|
|piqa (Bisk et al., 2020) hellaswag (Zellers et al., 2019) winogrande (Sakaguchi et al., 2021) openbook_qa (Mihaylov et al., 2018) sciq (Johannes Welbl, 2017) arc_easy (Clark et al., 2018) copa (Roemmele et al., 2011) commitment_bank (De Marneffe et al., 2019) mrpc (Dolan & Brockett, 2005) rte (Dagan et al., 2005) sst2 (Socher et al., 2013)<br><br>|

Table 14: Downstream Benchmarks for OLMoE.

|Downstream Benchmarks for OLMoE|
|---|
|piqa (Bisk et al., 2020) hellaswag (Zellers et al., 2019) winogrande (Sakaguchi et al., 2021) openbook_qa (Mihaylov et al., 2018) sciq (Johannes Welbl, 2017) arc_easy (Clark et al., 2018) arc_challenage (Clark et al., 2018) copa (Roemmele et al., 2011) boolq (Clark et al., 2019) commonsense_qa (Talmor et al., 2018) social_iqa (Sap et al., 2019) mmlu (Hendrycks et al., 2021)<br><br>|

- L 1B MODEL EXPERIMENTS

###### Training Loss

2.9

OLMo-1B

OLMo-1B-ResiDual

2.8

OLMo-1B-Altupx2

OLMo-1B-DHCx2

OLMo-1B-DHCx2 W/O tanh

2.7

Loss

2.6

2.5

2.4

100 200 300 400 500

Tokens (Billions)

- Figure 15: Training loss curves of related works, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99.

###### Training Loss

2.60

OLMo-1B

OLMo-1B-DHCx1 OLMo-1B-DHCx2 OLMo-1B-DHCx4 OLMo-1B-DHCx8

2.55

2.50

Loss

2.45

2.40

2.35

200 400 600 800 1000

Tokens (Billions)

- Figure 16: Training loss curves of DHC with tanh over 500 billion tokens, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99.

200 400 600 800 1000

Tokens (Billions)

2.35

2.40

2.45

2.50

2.55

2.60

Loss

Training Loss

OLMo-1B

- OLMo-1B-DHCx1 W/O tanh

- OLMo-1B-DHCx2 W/O tanh OLMo-1B-DHCx4 W/O tanh OLMo-1B-DHCx8 W/O tanh

- Figure 17: Training loss curves of DHC without tanh over 500 billion tokens, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99.

###### Training Loss

2.9

OLMo-1B

OLMo-1B-PTB

2.8

OLMo-1B-DHCx4 W/O tanh

OLMo-1B-DHCx4

2.7

Loss

2.6

2.5

2.4

100 200 300 400 500

Tokens (Billions)

- Figure 18: Training loss curves comparied with parallel transformer blocks (PTB), smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99.

Table 15: Results on downstream benchmarks for 1B models.

Method arc_easy copa hellaswag openbook_qa piqa sciq winogrande avg. OLMo-1B 56.8 76.0 56.1 33.8 74.4 85.1 55.6 62.5 Scaling n in DHC W/O tanh

OLMo-1B-DHCx1 W/O tanh 56.8 75.0 55.3 33.4 72.9 85.4 57.1 62.3 OLMo-1B-DHCx2 W/O tanh 63.0 74.0 57.1 34.6 73.5 86.0 58.2 63.8 OLMo-1B-DHCx4 W/O tanh 61.2 80.0 57.5 33.6 75.5 85.8 56.9 64.4 OLMo-1B-DHCx8 W/O tanh 61.1 75.0 57.6 35.4 73.8 85.2 58.5 63.8

Scaling n in DHC

OLMo-1B-DHCx1 59.7 74.0 55.5 33.6 73.5 85.4 54.5 62.3 OLMo-1B-DHCx2 59.7 73.0 56.7 34.0 74.7 85.2 57.9 63.0

- OLMo-1B-DHCx4 59.8 79.0 58.1 32.4 74.3 86.1 57.1 63.8 OLMo-1B-DHCx8 56.8 75.0 58.0 34.4 73.8 84.2 57.3 62.8

Scaling n in SHC

OLMo-1B-SHCx2 59.1 77.0 56.6 35.4 74.2 85.3 56.4 63.4 OLMo-1B-SHCx4 59.3 77.0 56.7 34.0 74.3 86.6 57.1 63.6

Non-trainable WC

- OLMo-1B-DHCx4 60.5 78.0 56.2 34.0 73.5 86.0 55.8 63.4

- OLMo-1B-DHCx4 W/O tanh 59.1 72.0 56.8 35.0 73.3 86.0 55.5 62.5 Non-trainable B

OLMo-1B-DHCx4 59.5 77.0 57.9 33.8 73.3 85.6 56.6 63.4

- OLMo-1B-DHCx4 W/O tanh 60.4 74.0 57.6 34.0 74.9 86.7 57.5 63.6

OLMo-1B-DHCx4W/Otanh2.2952.5922.7393.3472.6893.0662.5673.0052.4962.2222.8873.6382.6062.781

Non-trainableBeta OLMo-1B-DHCx42.2962.5942.7423.3482.6843.0512.5693.0082.4972.2212.9173.6272.6222.783

OLMo-1B-DHCx42.3122.6082.7523.3572.7003.0772.5833.0242.5082.2382.9593.6782.6362.802 OLMo-1B-DHCx4W/Otanh2.3082.6092.7553.3572.7103.1002.5853.0252.5102.2402.9453.6632.6442.804

OLMo-1B-SHCx42.3002.6032.7513.3572.6923.0622.5803.0182.5042.2322.8993.6532.6272.791 Non-trainableWC

ScalingninSHC OLMo-1B-SHCx22.3072.6102.7573.3602.7033.0632.5873.0232.5112.2382.9333.6432.6432.799

OLMo-1B-DHCx42.2902.5912.7383.3542.6833.0642.5643.0052.4922.2182.8903.6412.6112.781 OLMo-1B-DHCx82.2952.5912.7393.3532.6843.0542.5673.0082.4932.2192.8763.6312.6082.778

OLMo-1B-DHCx12.3232.6252.7753.3762.7283.0902.6063.0372.5332.2622.9613.6522.6782.819 OLMo-1B-DHCx22.3092.6082.7543.3672.7033.0612.5873.0222.5092.2372.9303.7042.6362.802

OLMo-1B-DHCx8W/Otanh2.2922.5892.7343.3502.6853.0602.5623.0062.4922.2182.8783.6282.6092.777 ScalingninDHC

OLMo-1B-DHCx2W/Otanh2.3112.6002.7493.3622.7003.0692.5833.0152.5032.2312.9083.6352.6252.792 OLMo-1B-DHCx4W/Otanh2.2952.5912.7353.3442.6863.0562.5623.0052.4922.2212.8983.6322.6102.779

ScalingninDHCW/Otanh OLMo-1B-DHCx1W/Otanh2.3202.6262.7733.3792.7253.1022.6093.0362.5312.2642.9483.7032.6722.822

Method4chanc4_100_domainsc4_engabicem2d2_s2orcm2d2_wikimanospheremc4_enpileptbtwitterAAEwikitext_103avg OLMo-1B2.3192.6152.7623.3642.7193.0852.5943.0282.5222.2502.9533.6722.6572.811

Table16:LossesofV2validationsetsfor1BModel.

OLMo-1B-DHCx4W/Otanh9.93213.38615.51028.43614.64121.13013.05120.25312.1429.22018.47837.61013.76617.504

Non-trainableBeta OLMo-1B-DHCx49.92713.35415.47528.41714.72221.45413.02120.18512.1359.22817.93238.00513.55317.493

OLMo-1B-DHCx410.05413.58715.72128.68915.02322.18613.26320.59412.3109.39019.01638.95914.07017.912 OLMo-1B-DHCx4W/Otanh10.09213.56615.66628.70414.87321.69613.24220.57912.2769.37719.27239.57013.96317.914

OLMo-1B-SHCx49.97713.50715.65528.69114.76621.37213.19420.45712.2349.31518.14938.56913.83617.671 Non-trainableWC

ScalingninSHC OLMo-1B-SHCx210.04613.60115.75328.78214.93121.39113.29420.56212.3199.37418.79138.21214.06017.778

OLMo-1B-DHCx49.87713.34415.43028.62414.63321.41013.00620.18612.0809.18918.10238.13613.60617.509 OLMo-1B-DHCx89.92213.34615.46728.59114.64021.19813.02520.24012.0979.19617.74937.74313.57017.445

OLMo-1B-DHCx110.21013.81016.03129.26515.30221.98613.53920.84712.5849.60619.32638.56414.55518.125 OLMo-1B-DHCx210.06113.56815.71029.00214.92521.34913.28420.52412.2949.36218.72740.59213.95717.950

OLMo-1B-DHCx8W/Otanh9.89713.31315.38728.48814.65821.33712.96020.20012.0849.18517.78237.65013.59217.425 ScalingninDHC

OLMo-1B-DHCx2W/Otanh9.92013.34015.41228.34014.67621.24312.96520.18112.0799.21918.12937.76813.59417.451 OLMo-1B-DHCx4W/Otanh10.08213.47015.62528.84814.88221.52113.23420.39212.2179.31218.32137.90513.80617.663

ScalingninDHCW/Otanh OLMo-1B-DHCx1W/Otanh10.17413.81516.00429.32815.25922.23113.58720.82312.5629.62019.07140.58014.46218.270

Method4chanc4_100_domainsc4_engabicem2d2_s2orcm2d2_wikimanospheremc4_enpileptbtwitterAAEwikitext_103avg OLMo-1B10.16713.66615.82928.90115.16621.86013.37720.65112.4539.48819.16139.32814.25118.023

Table17:PerplexitiesofV2validationsetsfor1Bmodels.

OLMo-1B-DHCx4W/Otanh2.6812.8862.7022.3062.9661.0242.4622.6803.1832.2042.6282.520

Non-trainableBeta OLMo-1B-DHCx42.6792.8802.6972.3062.9611.0252.4582.6843.1882.2042.6122.518

OLMo-1B-DHCx42.6952.9032.7162.3242.9781.0352.4772.7053.2012.2212.6492.537 OLMo-1B-DHCx4W/Otanh2.6922.8992.7142.3212.9761.0322.4742.6953.1892.2192.6412.532

OLMo-1B-SHCx42.6892.8922.7112.3152.9731.0282.4722.6883.1952.2142.6332.528 Non-trainableWC

ScalingninSHC OLMo-1B-SHCx22.6982.9072.7182.3252.9801.0322.4792.7003.1982.2212.6502.537

OLMo-1B-DHCx42.6752.8762.6972.3012.9621.0212.4552.6793.1762.2002.6172.515 OLMo-1B-DHCx82.6772.8802.7012.3042.9641.0222.4562.6803.1772.2012.6142.516

OLMo-1B-DHCx12.7142.9272.7322.3462.9911.0452.4992.7233.2112.2452.6832.556 OLMo-1B-DHCx22.6942.9012.7122.3212.9761.0322.4782.6993.2022.2182.6422.534

OLMo-1B-DHCx8W/Otanh2.6742.8762.6952.3032.9601.0222.4542.6803.1762.2002.6162.514 ScalingninDHC

OLMo-1B-DHCx2W/Otanh2.6762.8802.6982.3062.9611.0242.4562.6823.1742.2042.6172.516 OLMo-1B-DHCx4W/Otanh2.6892.8902.7062.3172.9691.0302.4712.6973.2002.2132.6332.529

ScalingninDHCW/Otanh OLMo-1B-DHCx1W/Otanh2.7122.9282.7322.3492.9911.0452.4992.7213.2192.2462.6772.556

Methodc4_endolma_booksdolma_common-crawldolma_pes2odolma_redditdolma_stackdolma_wikiicem2d2_s2orcpilewikitext_103avg OLMo-1B2.7022.9062.7222.3332.9801.0412.4872.7153.1992.2322.6632.544

Table18:LossesofV3validationsetsfor1Bmodel.

OLMo-1B-DHCx4W/Otanh14.59317.92614.90410.03219.4052.78511.72414.58824.1089.06013.83913.906

Non-trainableBeta OLMo-1B-DHCx414.57417.82014.84010.03819.3202.78711.67714.64724.2339.05913.62113.874

OLMo-1B-DHCx414.81018.22415.12010.21519.6502.81611.90214.95424.5529.22014.13514.145 OLMo-1B-DHCx4W/Otanh14.75618.16015.09510.19119.6132.80611.86814.80724.2739.20314.02114.072

OLMo-1B-SHCx414.71718.02815.04910.12119.5502.79611.84614.69924.4079.15513.91214.025 Non-trainableWC

ScalingninSHC OLMo-1B-SHCx214.85418.29315.15010.23019.6892.80711.93414.87624.4789.21414.15014.152

OLMo-1B-DHCx414.51417.74314.8299.98919.3432.77611.65014.57323.9489.02813.68913.826 OLMo-1B-DHCx814.54617.80714.88910.01119.3662.77911.65314.57923.9649.03013.65313.843

OLMo-1B-DHCx115.09318.67515.36010.44219.9092.84512.17415.22524.8109.43614.63214.418 OLMo-1B-DHCx214.79418.19015.06110.19119.6122.80611.91514.87024.5899.18714.04314.114

OLMo-1B-DHCx8W/Otanh14.49417.74914.81310.00019.3062.77911.63014.58723.9489.02113.68413.819 ScalingninDHC

OLMo-1B-DHCx2W/Otanh14.53117.81714.85710.03819.3232.78311.66214.60823.9069.06113.69413.844 OLMo-1B-DHCx4W/Otanh14.71117.99614.97510.14619.4792.80011.83014.83924.5249.14613.91714.033

ScalingninDHCW/Otanh OLMo-1B-DHCx1W/Otanh15.06418.69915.35610.47319.9092.84312.16715.19125.0139.45114.54014.428

Methodc4_endolma_booksdolma_common-crawldolma_pes2odolma_redditdolma_stackdolma_wikiicem2d2_s2orcpilewikitext_103avg OLMo-1B14.90818.28915.21610.30519.6862.83212.02615.09824.5039.31914.33414.229

Table19:PerplexitiesofV3validationsetsfor1Bmodels.

