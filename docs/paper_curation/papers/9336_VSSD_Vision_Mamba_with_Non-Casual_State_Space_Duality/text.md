# arXiv:2407.18559v2[cs.CV]4Aug2024

## VSSD: Vision Mamba with Non-Causal State Space Duality

Yuheng Shi City University of Hong Kong yuhengshi99@gmail.com

Mingjia Li Tianjin University mingjiali@tju.edu.cn

Minjing Dong City University of Hong Kong minjdong@cityu.edu.hk

Chang Xu University of Sydney c.xu@sydney.edu.au

### Abstract

Vision transformers have significantly advanced the field of computer vision, offering robust modeling capabilities and global receptive field. However, their high computational demands limit their applicability in processing long sequences. To tackle this issue, State Space Models (SSMs) have gained prominence in vision tasks as they offer linear computational complexity. Recently, State Space Duality (SSD), an improved variant of SSMs, was introduced in Mamba2 to enhance model performance and efficiency. However, the inherent causal nature of SSD/SSMs restricts their applications in non-causal vision tasks. To address this limitation, we introduce Visual State Space Duality (VSSD) model, which has a non-causal format of SSD. Specifically, we propose to discard the magnitude of interactions between the hidden state and tokens while preserving their relative weights, which relieves the dependencies of token contribution on previous tokens. Together with the involvement of multi-scan strategies, we show that the scanning results can be integrated to achieve non-causality, which not only improves the performance of SSD in vision tasks but also enhances its efficiency. We conduct extensive experiments on various benchmarks including image classification, detection, and segmentation, where VSSD surpasses existing state-of-the-art SSM-based models. Code and weights are available at https://github.com/YuHengsss/VSSD.

### 1 Introduction

In recent years, vision transformers [9, 37, 36, 19, 55, 22, 8, 51], pioneered by the Vision Transformer (ViT) [9], have achieved tremendous success in the field of computer vision. Thanks to the global receptive field and the robust information modeling capabilities of the attention mechanism, models [37, 66, 3] based on vision transformers have advanced significantly in various tasks such as classification [7], detection [33], and segmentation [67], surpassing classic CNN-based models [48, 24, 26, 27]. However, the quadratic computational complexity of the attention mechanism makes it resource-intensive for tasks involving long sequences, which limits its broader application.

Recently, State Space Models (SSMs) [15, 17, 16, 49], exemplified by Mamba [14], have garnered considerable attention from researchers. The S6 block, in particular, offers a global receptive field and exhibits linear complexity with respect to sequence length, presenting an efficient alternative. Pioneering vision mamba models such as Vim [69] and VMamba [35] have been developed to apply SSMs to vision tasks. Afterward, many variants were proposed [29, 50, 61, 47], which flatten 2D feature maps into 1D sequences using different scanning routes, model them with the S6 block, and subsequently integrate the results in multiple scanning routes. These multi-scan approaches improve

Preprint. Under review.

|81<br>82<br>83<br>84<br>85<br><br><br>0 10 20 30 40<br><br>| | | | | |
|---|---|---|---|---|
| | | | | |
| | | |Conv VMa<br><br>|NeXt mba|
| | | |VSS<br><br>Swin|D|
| | | | | |
<br><br>FLOPs(G)<br><br>(b)|
|---|

|81<br>82<br>83<br>84<br>85<br><br><br>0.4 0.7 1 1.3 1.6 1.9<br><br>ConvNeXt<br><br>VMamba<br><br>VSSD<br><br>Swin<br><br>Latency(ms)<br><br>(c)|
|---|

[Figure 1]

[Figure 2]

[Figure 3]

Top-1Accuracy(%)

Top-1Accuracy(%)

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]<br><br>|
|---|

|[Figure 8]<br><br>|
|---|

[Figure 9]

(a)

- Figure 1: (a) Two challenges when applying SSM/SSD to image data. (b) and (c) are comparisons on ImageNet. Our VSSD model achieves leading accuracy and efficiency compared to CNN-based ConvNeXt [38], ViT-based Swin Transformer [37], and SSM-based VMamba [35]. The latency of all models is measured on an A100 GPU using a batch size of 128 and FP16 precision.

the performance of SSMs in vision tasks, achieving results competitive with those of both CNNbased and ViT-based methods. More recently, Mamba2 [6] has introduced further enhancements to the S6 block, proposing the concept of State Space Duality (SSD). Mamba2 treats the state space transition matrix A as a scalar and expands the state space dimension, thereby enhancing model performance as well as training and inference efficiency. However, there exists a major concern regarding the application of SSD/SSMs in vision tasks, where the image data is naturally non-causal while SSD/SSMs have inherent causal properties. While another concern is flattening 2D feature maps into 1D sequences disrupts the inherent structural relationships among patches. We provide an illustration in Fig. 1 (a) to facilitate a more intuitive understanding of these two concerns. In this example, the central token within the flattened 1D sequences is restricted to accessing only previous tokens, unable to integrate information from subsequent tokens. Additionally, the token 1, which is adjacent to the central token in the 2D space, becomes distantly positioned in the 1D sequence, disrupting the natural structural relationships. A common practice in previous solutions [35, 29] is to increase the scanning routes on non-causal visual features, which alleviates these two concerns to some extent. Given these observations, an important question emerges: Is there a more effective and efficient way to apply SSD to non-causal vision data compared to multi-scan methods?

To address this question, our analysis of SSD reveals that treating the matrix A as a scalar facilitates a straightforward transformation of the SSD into a non-causal and position-independent manner, which we denote as Non-Causal SSD (NC-SSD). Specifically, rather than using A to determine the proportion of the hidden state to be retained, we employ it to dictate the extent of the current token’s contribution to the hidden states. In this scenario, each token’s contribution becomes self-referential. Based on this property, we demonstrate that the causal mask in SSD can be naturally removed without the need for specific scanning routes. This observation motivates us to develop a non-causal format of SSD, in which a single global hidden state can be derived to replace the previous token-wise hidden states, resulting in not only improved accuracy but also enhanced training and inference speeds. Unlike previous multi-scan methods [69, 35] that primarily alleviate the causal limitations of SSMs, our proposed NC-SSD also resolves the issue where flattening 2D feature maps into 1D sequences disrupts the continuity of adjacent tokens. Besides the NC-SSD, other techniques including hybrid with standard self-attention and overlapped downsampling are also explored. Building on these techniques, we introduce our Visual State Space Duality (VSSD) model and demonstrate its superior effectiveness and efficiency relative to methods based on CNNs, ViTs, and SSMs, as illustrated in Fig. 1 (b) and (c). Concretely, compared to the recent proposed SSM-based VMamba [35], our VSSD model outperforms it by approximately 1% in top-1 accuracy on ImageNet-1K dataset [7] while keeping a similar computational cost. Additionally, our model also consistently leads in the accuracy-latency curve. Besides the better trade-offs between performance and efficiency, another highlight of VSSD lies in the training speed. For example, compared to the vanilla SSD or multi-scan SSD (e.g., Bi-SSD with bidirectional scanning), our proposed model accelerates training speed by nearly 20% and 50% respectively.

In summary, our contributions are twofold. First, we analyze the state space duality and demonstrate that it can be seamlessly converted to a non-causal mode. Based on this insight, we introduce the NC-SSD, which retains the global receptive field and linear complexity benefits of the original SSD while incorporating an inherent non-causal property and achieving improved training and inference

efficiency. Second, utilizing NC-SSD as the foundational component, we propose the VSSD model and conduct extensive experiments to validate its effectiveness. With similar parameters and computational costs, our VSSD model outperforms other State-Of-The-Art(SOTA) SSM-based models across several widely recognized benchmarks in classification, object detection, and segmentation.

### 2 Related Work

Vision Transformers. The introduction of Vision Transformers (ViTs) [9, 37, 55, 8, 51] has revitalized the field of computer vision, which was previously dominated by Convolutional Neural Networks (CNNs) [31, 48, 24, 60, 27, 25, 50, 38]. However, the quadratic computational complexity of the self-attention mechanism in ViTs poses significant challenges when processing highresolution images, demanding significant computational resources. To address this issue, different solutions have been proposed, including hierarchical architectures [37, 36, 8, 55, 56, 21], windowed attention [37, 22, 52, 68], and variants of self-attention [54, 58, 64]. Meanwhile, linear attentions [30, 4, 41, 20] have been successful in reducing the computational complexity to a linear scale by changing the computation order of query, key and value in self-attention. Despite this advancement, the performance of linear attention remains inferior to that of the quadratic self-attention [53] and its variants [22, 12, 68].

State Space Models. State Space Models (SSMs) [15, 17, 16, 49, 13, 14] have increasingly captured the attention of researchers due to their global receptive field and linear computational complexity. Mamba [14], a prominent example of SSMs, introduced the S6 block, achieving performance on par with or better than transformers in Nature Language Processing (NLP) benchmarks. Subsequent efforts [40, 29, 10, 61, 2, 32, 63, 44] have explored the adaptation of the S6 block to vision tasks, yielding competitive results compared to both CNNs and ViT-based models. A central challenge in developing Mamba-based vision models is adapting the inherently causal properties of the Mamba block for non-causal image data. The most direct approach involves using different scanning routes to flatten 2D feature maps into 1D sequences which are then modeled with the S6 block and integrated. Inspired by these considerations, various scanning routes have been employed and proven effective, as evidenced by multiple studies [69, 35, 29, 40, 47]. More recently, Mamba2 [6] has highlighted the significant overlap between state space models and structured masked attention, identifying them as duals of each other, and introduced the concept of State Space Duality (SSD). Building on this foundation, we demonstrate that the SSD can be transformed into a non-causal mode through a straightforward transformation, without the need of specific scanning route.

### 3 Method

#### 3.1 Preliminaries

State Space Models. Classic State Space Models (SSMs) are used to describe the dynamics of a continuous system, transforming an input sequence x(t) ∈ R to a latent space representation h(t) ∈ RN. This representation is then utilized to generate an output sequence y(t) ∈ R. The mathematical formulation of an SSM is structured as follows:

◦

◦

Bx(t), y(t) = Ch(t), (1) where

h′(t) =

Ah(t) +

◦

◦

B ∈ RN×1 and C ∈ R1×N are parameters. To effectively integrate continuous SSMs into deep learning architectures, discretization is essential. This process involves introducing a timescale parameter ∆ ∈ R and applying the zero-order hold (ZOH) technique for discretization.

A ∈ RN×N,

◦

◦

Through this approach, the continuous matrices

B are transformed into their discrete counterparts, A and B. Consequently, Eq. 1 is redefined in a discrete format in Eq. 2, facilitating its application within modern computational frameworks:

A and

h(t) = Ah(t − 1) + Bx(t), y(t) = Ch(t), where A = e∆

(2)

◦

◦

◦

◦

◦

A)−1(e∆

A, B = (∆

A − I)∆

B ≈ ∆

B,

where I denotes the identity matrix. Furthermore, the process of Eq. 2 could be implemented in a global convolution manner as:

#### y = x ⊙ K, K = CB,CAB,...,CAL−1B , (3)

where K ∈ RL represents the convolution kernel. Recently, Mamba [14] makes the parameters B,C, and ∆ input-dependent. This modification addresses the limitations of the Linear Time Invariant (LTI) characteristics inherent in previous SSM models [16, 13], thus enhancing the adaptability and performance of SSMs.

#### 3.2 Non-Causal State Space Duality

More recently, Mamba2 [6] introduced the State Space Duality (SSD) and simplified the matrix A into a scalar. This special case of selective State Space Models (SSMs) can be implemented in both linear and quadratic forms. Without loss of generality, the matrix transformation form of selective state space models is expressed as follows:

y(t) = ti=1 CTt At:i+1Bix(i),where At:i = ti=2 Ai, y = SSM(A,B,C)(x) = Fx,where Fji = CTj Aj:iBi.

When Ai is reduced to a scalar, the quadratic form of Eq. 4 can be reformulated as:

(4)

y = Fx = M · (CTB)x, where Mij =

Ai+1 × ··· × Aj i > j 1 i = j 0 i < j,

(5)

while its linear form is denoted as:

h(t) = Ath(t − 1) + Btx(t),y(t) = Cth(t). (6)

To adapt SSMs for image data, the 2D feature maps should first be flattened into a 1D sequence of tokens which are then sequentially processed. Due to the causal nature of SSMs, where each token can only access previous tokens, information propagation is inherently unidirectional. This causal property leads to suboptimal performance when handling non-causal image data, a finding that has been corroborated by previous works [69, 35, 65]. Furthermore, flattening the 2D feature maps into 1D sequences disrupts their intrinsic structural information. For instance, tokens that are adjacent in the 2D map might end up being far apart in the 1D sequence, resulting in a loss of performance in vision tasks [18]. Since SSD is a variant of SSMs, adopting SSD for vision tasks presents similar challenges to those observed with SSMs:

- • Challenge 1: The causal property of the model restricts the flow of information, preventing later tokens from influencing earlier ones.
- • Challenge 2: Flattening 2D feature maps into 1D sequences disrupts the inherent structural relationships among patches during processing.

In the context of applying causal SSD to non-causal image data, it is instructive to revisit the linear formulation of SSD. In Eq. 6, the scalar At modulates the influence of the previous hidden state h(t − 1) and the information in current time step. In other words, the current hidden state h(t) could be viewed as a linear combination of the previous hidden state and the current input, weighted by At and 1, respectively. Therefore, if we discard the magnitude of these two terms and only preserve their relative weight, Eq. 6 could be rewritten as:

t

1 At

1 Ai

Bix(i). (7)

h(t) = h(t − 1) +

Btx(t) =

i=1

In this scenario, the contribution of a specific token to the current hidden state can be directly determined by itself as A1

, rather than through the cumulative multiplication of multiple coefficients. With each token’s contribution becoming self-referential, Challenge 2 is only partially addressed, as the current token can access only a subset of tokens due to the issue discussed in Challenge 1.

i

To address Challenge 1, previous SSM-based vision models frequently employed multi-scanning routes. Specifically, in the case of ViM [69], the token sequence was subjected to both forward and reverse scanning, enabling each token to access global information. Although these multi-scan approaches mitigate the causal property of SSMs, they do not address Challenge 2, as the long-range decay characteristic of SSMs remains confined to a 1D format and does not extend into 2D. To enable the acquisition of global information and thus suit non-causal image data, we also begin with a bidirectional scanning strategy. And we demonstrate that the results of forward and reverse scanning

###### SSD NC-SSD

𝑿𝟐

𝑿𝑳 𝑩𝑳

𝑿𝟏

𝑿𝟐

𝑿𝑳 𝑩𝑳

𝑿𝟏 𝑩𝟏

|𝑩 ∈ ℝ𝑳×𝑵|
|---|

𝑩𝟐 …

𝑩𝟏

𝑩𝟐 …

|𝑿 ∈ ℝ𝑳×𝑫|
|---|

|𝒁 ∈ ℝ𝑳×𝑵×𝑫|
|---|

…

𝒁𝟐

𝒁𝑳

… 𝒁𝑳

𝒁𝟏

𝒁𝟐

𝒁𝟏

|𝑯𝟏|
|---|
|𝑯𝟐|

|𝟏|
|---|
|𝑨𝟐:𝟏|

|𝟎|
|---|
|𝟏|

|𝟎|
|---|
|𝟎|

|𝟏<br><br>𝑨𝟏|
|---|
|𝟏 𝑨𝟏|

|𝟏<br><br>𝑨𝟐|
|---|
|𝟏 𝑨𝟐|

|𝟏 𝑨𝑳|
|---|
|𝟏 𝑨𝑳|

…

…

…

…

|𝑯|
|---|

…

…

…

…

…

…

…

|𝑯 ∈ ℝ𝑵×𝑫|
|---|

|𝑯𝑳|
|---|

|𝑨𝑳:𝟐|
|---|

|𝟏|
|---|

|𝑨𝑳:𝟏|
|---|

|𝟏<br><br>𝑨𝟐|
|---|

|𝟏<br><br>𝑨𝑳|
|---|

|𝟏<br><br>𝑨𝟏|
|---|

…

…

|𝑯 ∈ ℝ𝑳×𝑵×𝑫|
|---|

𝑴 ∈ ℝ𝑳×𝑳

𝑴 ∈ ℝ𝑳×𝑳

- Figure 2: Illustration of the Hidden State Generation Process for SSD and NC-SSD. During the hidden state update process, NC-SSD utilizes the scalar A to determine the extent of information increment for the current token, in contrast to SSD where A dictates the proportion of the hidden state to be retained. Unlike the SSD, which generates token-wise hidden states, the NC-SSD produces only a global hidden state to accommodate non-causal image data.

of Eq. 7 can be integrated to effectively address the aforementioned two challenges simultaneously. Let Hi denote the hidden state of the ith token in the bidirectional scanning approach, from which we can easily derive:

−i

- i
- j=1

L

1 Aj

1 A−j

1 Aj

1 Ai

Zi,where Zj = Bjx(j). (8)

Hi =

Zj +

Z−j =

Zj +

j=−L

j=1

If we consider A1

Zi in this equation as a bias and omit it, Eq. 8 can be further simplified, resulting in all tokens sharing the same hidden state H = Lj=1 A1

i

Zj. In such cases, the results from forward and reverse scanning can be seamlessly combined to establish a global context, effectively equivalent to removing the causal mask and transitioning to a non-causal format. Consequently, the first challenge associated with the causal property is resolved. Although the above results are derived from a bidirectional scanning approach, it is evident that in this non-causal format, different scanning routes yield consistent outcomes. In other words, designing specific scanning routes to capture global information becomes unnecessary. Furthermore, as demonstrated in Eq. 8, the contribution of different tokens to the current hidden state is no longer related to their spatial distance. Therefore, processing a flattened 2D feature map into a 1D sequence no longer compromises the original structural relationships. Thus, the second challenge is also resolved. Additionally, as the entire computation process can be conducted in parallel, rather than relying on the recurrent computational methods which are previously necessary for SSMs, there exists an improvement in training and inference speeds. After revising the iteration rules for the hidden state space, we update the corresponding tensor contraction algorithm or einsum notation in the linear form, following the Mamba2 framework [6]:

j

Z = contract(LD,LN → LND)(X,B) H = contract(LL,LDN → ND)(M,Z) Y = contract(LN,ND → LD)(C,H).

(9)

This algorithm involves three steps: the first step expands the input X using B, the second step unrolls scalar SSM recurrences to create a global hidden state H, and the final step contracts the hidden state H with C. For clarity, the initial two steps of SSD and NC-SSD are depicted in Fig. 2. Compared to vallina SSD, while the operation in the first step remains unchanged, the sequence length dimension in the hidden state H is eliminated in the non-causal mode, as all tokens share the same hidden state. In the final step, the output Y is produced through the matrix multiplication of C and H. Given that

, the matrix M can be reduced to a vector m ∈ RL by eliminating its first dimension. In this case, integrating m with either X or B could further simplify the transformation of Eq. 9 to:

Mi,j = A1

j

#### Y = C(BT(X · m)), (10)

Stage1

Stage2

Stage3

Stage4

|𝑯 × 𝑾 × 𝟑|
|---|

|𝑯 𝟖<br><br>×<br><br>𝑾 𝟖<br><br>× 𝑪𝟐|
|---|

|𝑯 𝟏𝟔<br><br>×<br><br>𝑾 𝟏𝟔<br><br>× 𝑪𝟑|
|---|

|𝑯 𝟑𝟐<br><br>×<br><br>𝑾 𝟑𝟐<br><br>× 𝑪𝟒|
|---|

|𝑯 𝟒<br><br>×<br><br>𝑾 𝟒<br><br>× 𝑪𝟏|
|---|

[Figure 10]

| | | |
|---|---|---|
| | | |

Down

Down

Down

Stem

VSSD Block

VSSD Block

VSSD Block

MSA Block

|× 𝑵𝟏|
|---|

|× 𝑵𝟐|
|---|

|× 𝑵𝟑|
|---|

|× 𝑵𝟒|
|---|

𝐴

NC-SSD Block

LinearLinear

| |𝑋|
|---|---|
| | |

NC-SSD

Linear

𝑌

LN

Conv

|𝜎|
|---|

FFN

𝜎

𝐵 𝐶

LN

LN

Linear

|𝑍|
|---|

- Figure 4: Overall Architecture of the Proposed VSSD Model. The VSSD model initiates with a series of overlapping convolutions serving as the stem, followed by four progressive stages of processing. First three stages are equipped with VSSD Block, which is elaborated in the lower part of the figure, comprising a NC-SSD block and a FFN. Local Perception Units (LPU) are omitted in this visualization for brevity.

[Figure 11]

which can be regarded as a variant of linear attention. However, it is worth noting that just as A plays a distinguished role in Mamba2, the vector m is also crucial, as demonstrated in our ablation studies. In practice, we directly use the learned A instead of

A since they share the same range of values. To gain a more intuitive under-

1

standing of the role of m in Eq. 10, we visualize the average of m across different heads as shown in Fig. 3. Predominantly, m focuses on foreground features, enabling the model to prioritize elements that are crucial for the task at hand.

Figure 3: Visualization of input images alongside their corresponding heat maps, which are derived by averaging the vector m across various heads in the NC-SSD.

#### 3.3 Vision State Space Duality Model

Block Design. To enhance the SSD block in Mamba2 for vision applications, several modifications have been implemented beyond merely substituting the SSD with an NC-SSD to develop our Visual State Space Duality (VSSD) block. When constructing the NC-SSD block, the causal convolution 1D is replaced by a Depth-Wise Convolution (DWConv) with a kernel size of three, in line with previous vision mamba works [35, 29]. Additionally, a Feed-Forward Network (FFN) is integrated subsequent to the NC-SSD block to facilitate enhanced information exchange across channels and to maintain alignment with the established practices of classical vision transformers [9, 37, 51]. Moreover, a Local Perception Unit (LPU) [19] is incorporated prior to the NC-SSD block and the FFN, augmenting the model’s capability for local feature perception. Skip connections [24] are also implemented among different blocks. The architecture of the VSSD block is depicted in the lower part of Fig. 4.

Hybrid with Self-Attention. Mamba2 demonstrates that integrating SSD with standard Multi-head Self Attention (MSA) yields additional improvements. In a similar vein, our model incorporates self-attention. However, unlike Mamba2, which uniformly intersperses self-attention throughout the

network, we strategically replace the NC-SSD block with self-attention module exclusively in the last stage. This modification leverages the robust capabilities of self-attention in processing high-level features, as evidenced by prior works [34, 43, 12] in vision tasks.

Overlapped downsampling layers. As hierarchical vision transformers [37] and vision state space models [35] predominantly employ non-overlapped convolution for downsampling, recent studies [22, 57] have demonstrated that overlapped downsampling convolutions can introduce beneficial inductive biases. Consequently, we adopt overlapped convolutions in our model, following the manner used in MLLA [20]. To maintain the parameter count and computational FLOPs remain comparable, we have accordingly adjusted the depth of our model.

Overall Architecture. We develop our VSSD model in accordance with the methods discussed above, and its architecture is depicted in Fig. 4. Mirroring the design principles of established vision backbones in previous works [37, 38, 35], our VSSD model is structured into four hierarchical stages. The first three stages employ the VSSD block, while the final stage incorporates the MSA block. Detailed architectures of VSSD variants are shown in the Tab 1.

Table 1: Model Specifications of VSSD varints. Model Blocks Channels Heads #Param FLOPs

VSSD-Micro [2, 2, 8, 4] [48, 96, 192, 384] [2, 4, 8, 16] 14 2.3 VSSD-Tiny [2, 4, 8, 4] [64, 128, 256, 512] [2, 4, 8, 16] 24 4.5 VSSD-Small [3, 4, 18, 5] [64, 128, 256, 512] [2, 4, 8, 16] 40 7.4 VSSD-Base [3, 4, 18, 5] [96, 192, 384, 768] [3, 6, 12, 24] 89 16.1

### 4 Experiment

#### 4.1 Classification

- Table 2: Accuracy Comparison across Various Models on ImageNet-1K. The † indicates results are obtained with MESA [11]. The LAttn is the abbreviation of linear attention.

Top-1 Acc(%) Micro Models

Top-1 Acc(%) Small Models

Method Type #Param. FLOPs

Method Type #Param. FLOPs

RegNetY-1.6G [42] Conv 11M 1.6G 78.0

ConvNeXt-S [38] Conv 50M 8.7G 83.1 EffNet-B5 [50] Conv 30M 9.9G 83.6 MambaOut-S [65] Conv 48M 9.0G 84.1 Swin-S [37] Attn 50M 8.7G 83.0 PVTv2-B3 [56] Attn 45M 6.9G 83.2 Focal-S [62] Attn 50M 8.7G 83.5 CSwin-S [8] Attn 35M 6.9G 83.6 NAT-S [22] Attn 51M 7.8G 83.7 VMamba-S [35] SSM 44M 11.2G 83.5 PMamba-L2 [61] SSM 25M 8.1G 81.6 VMambaV9-S [35] SSM 50M 8.7G 83.6 LVMamba-S [29] SSM 50M 11.4G 83.7 VSSD-S SSD 40M 7.4G 84.1 MLLA-S [20] LAttn 43M 7.3G 84.4† VSSD-S SSD 40M 7.4G 84.5†

- EffNet-B3 [50] Conv 12M 1.8G 81.6 PVTv2-b1 [56] Attn 13M 2.1G 78.7 BiFormer [68] Attn 13M 2.2G 81.4 NAT-M [22] Attn 20M 2.7G 81.8 CMT-XS [19] Attn 15M 1.5G 81.8 SMT-T [34] Attn 12M 2.4G 82.2 Vim-T [69] SSM 7M 1.5G 76.1 LVim-T [29] SSM 8M 1.5G 76.2 VSSD-M SSD 14M 2.3G 82.5

Tiny Models

RegNetY-4G [42] Conv 21M 4.0G 80.0 ConvNeXt-T [38] Conv 29M 4.5G 82.1 MambaOut-T [65] Conv 27M 4.5G 82.7

- EffNet-B4 [50] Conv 19M 4.2G 82.9 DeiT-S [51] Attn 22M 4.6G 79.8 Swin-T [37] Attn 29M 4.5G 81.3 PVTv2-B2 [56] Attn 25M 4.0G 82.0 Focal-T [62] Attn 29M 4.9G 82.2 CSwin-T [8] Attn 23M 4.3G 82.7 NAT-T [22] Attn 28M 4.3G 83.2 VMambaV9-T [35] SSM 31M 4.9G 82.5 LVMamba-T [29] SSM 26M 5.7G 82.7 MSVMamba-T [47] SSM 33M 4.6G 82.8

###### Base Models

ConvNeXt-B [38] Conv 89M 15.4G 83.8 MambaOut-B [65] Conv 85M 15.8G 84.2 DeiT-B [51] Attn 86M 17.5G 81.8 Swin-B [37] Attn 88M 15.4G 83.5 CSwin-S [8] Attn 78M 15.0G 84.2 NAT-B [22] Attn 90M 13.7G 84.3 PMamba-L3 [61] SSM 50M 14.4G 82.3 VMambaV9-B [35] SSM 89M 15.4G 83.9

- VSSD-T SSD 24M 4.5G 83.7 MLLA-T [20] LAttn 25M 4.2G 83.5†

- VSSD-T SSD 24M 4.5G 84.1†

- VSSD-B SSD 89M 16.1G 84.7 MLLA-B [20] LAttn 96M 16.2G 85.3†

- VSSD-B SSD 89M 16.1G 85.4†

[Figure 12]

- Figure 5: Comparison of the Effective Receptive Field (ERF) among our VSSD, CNN-based models (ResNet [24] and ConvNeXt [38]), attention-based models (Swin [37] and DeiT [51]), and the SSM-based VMamba [35]. Our VSSD effectively eliminates the impact of token spacing on the contribution of information compared to SSM-based VMamba.

Configurations. Our experiments are conducted using the ImageNet-1K dataset [7], consistent with methodologies from prior studies [37, 35]. Each model is subjected to a training regimen spanning 300 epochs, which includes a 20-epoch warm-up phase. Optimization is performed with AdamW, where the betas are set to (0.9, 0.999) and momentum at 0.9. A cosine decay scheduler manages the learning rate and is combined with a weight decay rate of 0.05. To further refine model accuracy and generalization, we incorporate exponential moving average (EMA) techniques and apply label smoothing with a coefficient of 0.1. More detailed configurations could be found in Appendix A. For the testing phase, images are center-cropped to dimensions of 224×224.

- Performance Evaluation. Tab. 2 presents a comparative comparison of our VSSD models against CNNs, ViTs, and other SSM-based frameworks on the ImageNet-1K dataset [7]. The VSSD-M model, equipped with 14M parameters and 2.3G FLOPs, secures a top-1 accuracy of 82.5%, surpassing the similarly priced NAT-M [22] by 0.7%. In comparisons with models categorized as tiny and small, VSSD consistently outperforms its counterparts. Specifically, the VSSD-T model, which has 24M parameters and 4.5G FLOPs, achieves an accuracy of 83.7%, outdoing VMambaV9-T [35] by 1.2%. For the small-sized model variant, VSSD-S, which comprises 40M parameters and 7.4G FLOPs, achieves an accuracy of 84.1%, surpassing the LocalVMamba-S [29] by 0.4%. In the base-sized variants, our VSSD-B, with 89M parameters and 16.1G FLOPs, records an accuracy of 84.7%, which is 0.8% higher than that of VMambaV9-B. When introducing MESA [11] for further refinement, the results for our tiny, small and base size model are improved to 84.1%, 84.5% and 85.4% respectively.

In addition to quantitative comparisons, we conducted a comparative analysis of the Effective Receptive Field (ERF) before and after training across various models, including CNN-based ResNet50 [24] and ConvNeXt-Tiny [38], attention-based Swin-Tiny [37] and DeiT-Small [51], and SSM-based VMamba-Tiny [35], along with our VSSD-Tiny. The ERF of the central pixel is plotted using the method proposed in [39], utilizing 50 randomly selected images with a resolution of 1024x1024 from the ImageNet-1K validation set. To demonstrate the effectiveness of the proposed NC-SSD, the techniques such as hybrid self-attention and overlapped downsampling layers discussed in Sec. 3.3 are not employed in our VSSD model for this analysis. Notably, only our VSSD and DeiT consistently exhibit a global receptive field both before and after training. Under after-training, a distinct cross-shaped attenuation is observed in VMamba, whereas our approach effectively eliminates the impact of token spacing on the contribution of information.

#### 4.2 Object Detection and Instance Segmentation

Configurations. Our evaluation of the VSSD model utilizes the MS COCO dataset [33] within the Mask R-CNN framework [23] for tasks related to object detection and instance segmentation. All experiments are facilitated using the MMDetection library [1]. Consistent with prior studies [37, 35], during the training phase, images are adjusted such that the shorter side measures 800 pixels, while the longer side does not exceed 1333 pixels. Optimization is carried out using the AdamW optimizer, with a set learning rate of 0.0001 and a batch size of 16. When adopting the standard "1×" training

- Table 3: Object Detection and Instance Segmentation with Mask R-CNN Framework [23] on MS COCO Dataset [33]. The FLOPs are tested with an input size of 1280 × 800.

Mask R-CNN 1x Method APb APb50 APb75 APm APm50 APm75 #Param. FLOPs

PVT-T [55] 36.7 59.2 39.3 35.1 56.7 37.3 33M 208G EffVMamba-S [40] 39.3 61.8 42.8 36.7 58.9 39.2 31M 197G MSVMamba-M [47] 43.8 65.8 47.7 39.9 62.9 42.9 32M 201G VSSD-M 45.4 67.5 49.8 41.3 64.5 44.6 33M 220G

Swin-T [37] 42.7 65.2 46.8 39.3 62.2 42.2 48M 267G ConvNeXt-T [38] 44.2 66.6 48.3 40.1 63.3 42.8 48M 262G VMamba-T [35] 46.5 68.5 50.7 42.1 65.5 45.3 42M 286G LocalVMamba-T [29] 46.7 68.7 50.8 42.2 65.7 45.5 45M 291G MSVMamba-T [47] 46.9 68.8 51.4 42.2 65.6 45.4 53M 252G VSSD-T 46.9 69.4 51.4 42.6 66.4 45.9 44M 265G

- Swin-S [37] 44.8 66.6 48.9 40.9 63.2 44.2 69M 354G

- ConvNeXt-S [38] 45.4 67.9 50.0 41.8 65.2 45.1 70M 348G

- VMamba-S [35] 48.2 69.7 52.5 43.0 66.6 46.4 64M 400G

- LocalVMamba-S [29] 48.4 69.9 52.7 43.2 66.7 46.5 69M 414G

- VSSD-S 48.4 70.1 53.1 43.5 67.2 47.1 59M 325G Mask R-CNN 3x + MS

Method APb APb50 APb75 APm APm50 APm75 #Param. FLOPs

PVT-T [55] 39.8 62.2 43.0 37.4 59.3 39.9 33M 208G LightViT-T [28] 41.5 64.4 45.1 38.4 61.2 40.8 28M 187G EffVMamba-S [40] 41.6 63.9 45.6 38.2 60.8 40.7 31M 197G MSVMamba-M [47] 46.3 68.1 50.8 41.8 65.1 44.9 32M 201G VSSD-M 47.7 69.7 52.1 42.8 66.5 46.0 33M 220G

Swin-T [37] 46.0 68.1 50.3 41.6 65.1 44.9 48M 267G ConvNeXt-T [38] 46.2 67.9 50.8 41.7 65.0 44.9 48M 262G VMamba-T [35] 48.5 69.9 52.9 43.2 66.8 46.3 42M 286G LocalVMamba-T [29] 48.7 70.1 53.0 43.4 67.0 46.4 45M 291G

- VSSD-T 48.8 70.4 53.4 43.6 67.6 46.9 44M 265G

schedule, the learning rate is reduced by a factor of 0.1 at epochs 8 and 11 while the extended "3× + MS" schedule sees a reduction in the learning rate by the same factor at epochs 27 and 33.

- Performance Evaluation. Tab. 3 details the comparative performance of our model against wellestablished CNNs, ViTs, and other SSM-based models. Our VSSD model demonstrates superior performance in various configurations. Remarkably, our VSSD-T model demonstrates a significant advantage, outperforming Swin-T [37] by margins of +4.2 in box AP and +3.3 in mask AP. Under the extended "3×" training schedule, VSSD-T still consistently outperforms various competitors.

#### 4.3 Semantic Segmentation

Configurations. In alignment with the approaches described in Swin [37] and VMamba [35], our experiments leverage the UperHead [59] framework, utilizing an ImageNet pre-trained backbone for initialization. The training regimen spans 160K iterations with a batch size of 16, executed using the MMSegmentation library [5]. The primary experiments are performed with a standard input resolution of 512 × 512. To further evaluate the robustness of our model, Multi-Scale (MS) testing is implemented. Optimization is carried out using the AdamW optimizer, with the learning rate established at 6 × 10−5.

Performance Evaluation. Detailed performance metrics for our model and its competitors are displayed in Tab. 4, encompassing both single-scale and multi-scale testing scenarios. Specifically, in the context of the Tiny model category and single-scale testing, our VSSD model demonstrates superior performance, exceeding the results of Swin, ConNeXt, and VMamba models in the tiny variant by margins of +3.5, +1.9, and +0.6 mIoU, respectively.

- Table 4: Results of Semantic Segmentation on the ADE20K Dataset [67] using the UperNet Framework [59]. FLOPs for all models are computed using input dimensions of 512 × 2048. In the table, "SS" represents single-scale testing, while "MS" indicates multi-scale testing.

Method

mIoU SS

mIoU MS

#Param. FLOPs

EffVMamba-S [40] 41.5 42.1 29M 505G MSVMamba-M [47] 45.1 45.4 42M 875G VSSD-M 45.6 46.0 42M 893G

Swin-T [37] 44.4 45.8 60M 945G ConvNeXt-T [38] 46.0 46.7 60M 939G VMamba-T [35] 47.3 48.3 55M 964G LocalVMamba-T [29] 47.9 49.1 57M 970G EffVMamba-B [40] 46.5 47.3 65M 930G MSVMamba-T [47] 47.6 48.5 65M 942G VSSD-T 47.9 48.7 53M 941G

4.4 Ablations

To validate the effectiveness of the proposed modules, we conducted detailed ablation experiments on the VSSD-Micro model. Using the SSD block as the token mixer and patchified downsamplers (e.g. convolution with 4×4 kernel and stride of 4 in stem) following Swin [37] and vallina VMamaba [35], we established the baseline configuration, detailed in the first row of Tab. 5. For throughput testing, we utilized an A100-PCIE-40G GPU with a batch size of 128 and FP16 precision.

- Table 5: Ablation study of VSSD-Micro on ImageNet-1K. Our NC-SSD consistently outperforms vallina SSD and Bi-SSD in terms of accuracy and efficiency. Other techniques further enhance the performance.

Op. Type Downsampler Layers Top-1 #Params FLOPs Thru. Train Thru. Acc(%) (G) (imgs/sec) (imgs/sec)

SSD Patch 2, 4, 8, 4 81.0 14.8 M 2.1 1818 523 Bi-SSD Patch 2, 4, 8, 4 81.4 15.2 M 2.2 1741 399 NC-SSD Patch 2, 4, 8, 4 81.6 14.8 M 2.1 1843 606 Hybrid Patch 2, 4, 8, 4 81.8 13.4 M 2.1 1890 622 Hybrid Conv 2, 2, 8, 4 82.5 13.5 M 2.3 1918 597

Different SSD Mechanisms. In our ablation study for the token mixer, we explored different scanning routes for SSD. Specifically, the Bi-SSD is introduced, where we split the channels into two parts and reverse one part to create backward scanning sequences. These sequences with opposite scanning routes are then concatenated after the SSD block. As shown in Tab. 5, our NC-SSD model outperforms both the vanilla SSD and Bi-SSD by 0.6% and 0.2% in top-1 accuracy, respectively. Moreover, both training and inference throughput are enhanced, with NC-SSD improving training throughput by nearly 50% compared to the Bi-SSD approach.

Hybrid Architecture and Overlapped Downsampler. The effectiveness of incorporating standard attention in the last stage and using overlapped downsampler is demonstrated in the last two rows of Tab. 5. Specifically, replacing NC-SSD with standard attention in the last stage results in a 0.2% improvement in accuracy while slightly reducing the parameters. Replacing the patchified downsampler with the overlapped convolutional manner improves accuracy by 0.7% while increasing the FLOPs by 0.2G. To maintain approximate parameters, we adjusted the layer configuration from [2,4,8,4] to [2,2,8,4].

Effect of m. Eq. 10 conceptualizes NC-SSD as a variant of linear attention that incorporates an additional weight vector, m. Fig. 3 visually demonstrates how m selectively emphasizes foreground features. To quantitatively assess the impact of m, we conducted experiments with and without this

component in the NC-SSD block under 100 training epochs with 5 epochs for warming up. The results, presented in Tab. 6, reveal a significant influence of m on model performance.

Without m, our experiments show that the model experiences unstable training, leading to crashes. This instability is particularly pronounced in larger models. We report the highest accuracy achieved before training crashed, marked with a †. For the tiny-sized model, the best accuracy was only 32.6%. For the smallsized model, the training crashed in the very first epoch. We hypothesize that this instability arises because, in the absence of normalization techniques typically used in linear attention approaches [4, 41, 45], the magnitude of the features escalates dramatically, leading to a crash. In contrast, with m, the model achieves a robust top-1 accuracy of 81.8%, while maintaining the same number of parameters and computational complexity.

Table 6: Ablation Study on the Effect of m in NC-SSD. The symbol † indicates the best accuracy achieved prior to encountering N.A.

#### Operation Size m Top-1 #Params FLOPs

Acc(%) (G)

✗ 32.6† 24.3M 4.5 ✓ 81.8 24.3M 4.5

Tiny

NC-SSD

Small ✗ N.A 40.0M 7.4

### 5 Limitations

Although the proposed VSSD model outperforms other SSM-based models in ImageNet-1K, the incremental performance gains of VSSD on downstream tasks [33, 67], when compared to other SSMbased models, are marginal. When evaluated against SOTA vision transformer variants [12, 58, 46], there remains a significant gap in performance on downstream tasks. Furthermore, this paper lacks experiments involving larger models and more extensive datasets, such as those using the ImageNet22K benchmark [7]. Consequently, the scalability of the proposed VSSD model remains an area ripe for further exploration.

### 6 Conclusion

In conclusion, our study introduces the NC-SSD, which redefines SSD by modifying the role of matrix A and eliminating the causal mask. These adaptations facilitate a transition to a non-causal mode, significantly enhancing both accuracy and efficiency. Extensive experiments demonstrate its superiority over the vanilla SSD and its multi-scan based variants. Furthermore, by integrating techniques such as hybrid standard attention and overlapped downsampling, our VSSD model achieves comparable or superior performance compared to well-established CNNs, ViTs, and Vision SSMs across several widely-used benchmarks.

### References

- [1] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019.
- [2] Tianxiang Chen, Zhentao Tan, Tao Gong, Qi Chu, Yue Wu, Bin Liu, Jieping Ye, and Nenghai Yu. Mim-istd: Mamba-in-mamba for efficient infrared small target detection. arXiv preprint arXiv:2403.02148, 2024.
- [3] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR, 2022.
- [4] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.
- [5] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open-mmlab/mmsegmentation, 2020.
- [6] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.
- [7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.
- [8] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining Guo. Cswin transformer: A general vision transformer backbone with cross-shaped windows. In CVPR, 2022.
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020.
- [10] Chengbin Du, Yanxi Li, and Chang Xu. Understanding robustness of visual state space models for image classification. arXiv preprint arXiv:2403.10935, 2024.
- [11] Jiawei Du, Daquan Zhou, Jiashi Feng, Vincent Tan, and Joey Tianyi Zhou. Sharpness-aware training for free. Advances in Neural Information Processing Systems, 2022.
- [12] Qihang Fan, Huaibo Huang, Mingrui Chen, Hongmin Liu, and Ran He. Rmt: Retentive networks meet vision transformers. In CVPR, 2024.
- [13] Daniel Y Fu, Tri Dao, Khaled K Saab, Armin W Thomas, Atri Rudra, and Christopher Ré. Hungry hungry hippos: Towards language modeling with state space models. arXiv preprint arXiv:2212.14052, 2022.
- [14] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
- [15] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. Hippo: Recurrent memory with optimal polynomial projections. NeurIPS, 2020.
- [16] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.
- [17] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré. Combining recurrent, convolutional, and continuous-time models with linear state space layers. NeurIPS, 2021.
- [18] Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, and Shu-Tao Xia. Mambair: A simple baseline for image restoration with state-space model. arXiv preprint arXiv:2402.15648, 2024.

- [19] Jianyuan Guo, Kai Han, Han Wu, Yehui Tang, Xinghao Chen, Yunhe Wang, and Chang Xu. Cmt: Convolutional neural networks meet vision transformers. In CVPR, pages 12175–12185, 2022.
- [20] Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, and Gao Huang. Demystify mamba in vision: A linear attention perspective. arXiv preprint arXiv:2405.16605, 2024.
- [21] Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, and Yunhe Wang. Transformer in transformer. In NeurIPS, 2021.
- [22] Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi. Neighborhood attention transformer. In CVPR, 2023.
- [23] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In ICCV, 2017.
- [24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.
- [25] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.
- [26] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In CVPR, 2018.
- [27] Gao Huang, Zhuang Liu, Geoff Pleiss, Laurens Van Der Maaten, and Kilian Weinberger. Convolutional networks with dense connectivity. IEEE TPAMI, 2019.
- [28] Tao Huang, Lang Huang, Shan You, Fei Wang, Chen Qian, and Chang Xu. Lightvit: Towards light-weight convolution-free vision transformers. arXiv preprint arXiv:2207.05557, 2022.
- [29] Tao Huang, Xiaohuan Pei, Shan You, Fei Wang, Chen Qian, and Chang Xu. Localmamba: Visual state space model with windowed selective scan. arXiv preprint arXiv:2403.09338, 2024.
- [30] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In ICML, 2020.
- [31] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. NeurIPS, 2012.
- [32] Kunchang Li, Xinhao Li, Yi Wang, Yinan He, Yali Wang, Limin Wang, and Yu Qiao. Videomamba: State space model for efficient video understanding. arXiv preprint arXiv:2403.06977, 2024.
- [33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.
- [34] Weifeng Lin, Ziheng Wu, Jiayu Chen, Jun Huang, and Lianwen Jin. Scale-aware modulation meet transformer. In ICCV, 2023.
- [35] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, and Yunfan Liu. Vmamba: Visual state space model. arXiv preprint arXiv:2401.10166, 2024.
- [36] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In CVPR, 2022.
- [37] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021.
- [38] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, 2022.

- [39] Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard S. Zemel. Understanding the effective receptive field in deep convolutional neural networks. In NeurIPS, 2016.
- [40] Xiaohuan Pei, Tao Huang, and Chang Xu. Efficientvmamba: Atrous selective scan for light weight visual mamba. arXiv preprint arXiv:2403.09977, 2024.
- [41] Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, and Yiran Zhong. cosformer: Rethinking softmax in attention. arXiv preprint arXiv:2202.08791, 2022.
- [42] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces. In CVPR, 2020.
- [43] Sucheng Ren, Xingyi Yang, Songhua Liu, and Xinchao Wang. Sg-former: Self-guided transformer with evolving token reallocation. In ICCV, 2023.
- [44] Jiacheng Ruan and Suncheng Xiang. Vm-unet: Vision mamba unet for medical image segmentation. arXiv preprint arXiv:2402.02491, 2024.
- [45] Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Efficient attention: Attention with linear complexities. In WACV, 2021.
- [46] Dai Shi. Transnext: Robust foveal visual perception for vision transformers. In CVPR, 2024.
- [47] Yuheng Shi, Minjing Dong, and Chang Xu. Multi-scale vmamba: Hierarchy in hierarchy visual state space model. arXiv preprint arXiv:2405.14174, 2024.
- [48] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
- [49] Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933, 2022.
- [50] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In ICML, 2019.
- [51] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In ICML, 2021.
- [52] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer. In ECCV, 2022.
- [53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.
- [54] Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. arXiv preprint arXiv:2211.05778, 2022.
- [55] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In ICCV, 2021.
- [56] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pvt v2: Improved baselines with pyramid vision transformer. Computational Visual Media, 2022.
- [57] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers. arXiv preprint arXiv:2103.15808, 2021.
- [58] Zhuofan Xia, Xuran Pan, Shiji Song, Li Erran Li, and Gao Huang. Dat++: Spatially dynamic vision transformer with deformable attention. arXiv preprint arXiv:2309.01430, 2023.

- [59] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In ECCV, 2018.
- [60] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In CVPR, 2017.
- [61] Chenhongyi Yang, Zehui Chen, Miguel Espinosa, Linus Ericsson, Zhenyu Wang, Jiaming Liu, and Elliot J Crowley. Plainmamba: Improving non-hierarchical mamba in visual recognition. arXiv preprint arXiv:2403.17695, 2024.
- [62] Jianwei Yang, Chunyuan Li, Xiyang Dai, and Jianfeng Gao. Focal modulation networks. NeurIPS, 2022.
- [63] Yuhuan Yang, Chaofan Ma, Jiangchao Yao, Zhun Zhong, Ya Zhang, and Yanfeng Wang. Remamber: Referring image segmentation with mamba twister. arXiv preprint arXiv:2403.17839, 2024.
- [64] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. In CVPR, 2022.
- [65] Weihao Yu and Xinchao Wang. Mambaout: Do we really need mamba for vision? arXiv preprint arXiv:2405.07992, 2024.
- [66] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and HeungYeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022.
- [67] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017.
- [68] Lei Zhu, Xinjiang Wang, Zhanghan Ke, Wayne Zhang, and Rynson Lau. Biformer: Vision transformer with bi-level routing attention. In CVPR, 2023.
- [69] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417, 2024.

### A More Detailed information of VSSD

Our experiments are conducted using the ImageNet-1K dataset [7]. Each model undergoes training for 300 epochs, which includes a 20-epoch warm-up phase. We employ the AdamW optimizer, setting the betas to (0.9, 0.999) and the momentum to 0.9. A cosine decay scheduler manages the learning rate, complemented by a weight decay rate of 0.05. The batch sizes and peak learning rates are set to 1024/1e-3 for the Micro and Tiny models, and 2048/1.2e-3 for the Small and Base models, respectively. To enhance model accuracy and generalization, we incorporate exponential moving average (EMA) techniques and apply label smoothing with a coefficient of 0.1. The stochastic depth drop rates for our Micro, Tiny, Small, and Base models are set at 0.2, 0.2, 0.4, and 0.6, respectively. Further details are provided in Tab. 7.

Table 7: Detailed Configuration Parameters for ImageNet-1K Training. Settings Micro Tiny Small Base

Input resolution 2242 Epochs 300 Batch size 1024 1024 2048 2048 Optimizer AdamW Adam ϵ 1e-8 Adam (β1,β2) (0.9, 0.999) Learning rate 1e-3 1e-3 1.2e-3 1.2e-3 Learning rate decay Cosine Warmup epochs 20 Weight decay 0.05 Rand Augment rand-m9-mstd0.5-inc1 Cutmix 1.0 Mixup 0.8 Cutmix-Mixup switch prob 0.5 Random erasing prob 0.25 Label smoothing 0.1 Stochastic depth rate 0.2 0.2 0.4 0.6 Random erasing prob 0.25 EMA decay rate 0.9999

