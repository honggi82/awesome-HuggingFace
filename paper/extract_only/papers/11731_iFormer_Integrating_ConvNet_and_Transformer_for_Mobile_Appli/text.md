# arXiv:2501.15369v2[cs.CV]17Feb2025

## IFORMER: INTEGRATING CONVNET AND TRANSFORMER FOR MOBILE APPLICATION

Chuanyang Zheng Independent Researcher chuanyang zheng@sjtu.edu.cn

ABSTRACT

We present a new family of mobile hybrid vision networks, called iFormer, with a focus on optimizing latency and accuracy on mobile applications. iFormer effectively integrates the fast local representation capacity of convolution with the efficient global modeling ability of self-attention. The local interactions are derived from transforming a standard convolutional network, i.e., ConvNeXt, to design a more lightweight mobile network. Our newly introduced mobile modulation attention removes memory-intensive operations in MHA and employs an efficient modulation mechanism to boost dynamic global representational capacity. We conduct comprehensive experiments demonstrating that iFormer outperforms existing lightweight networks across various tasks. Notably, iFormer achieves an impressive Top-1 accuracy of 80.4% on ImageNet-1k with a latency of only 1.10 ms on an iPhone 13, surpassing the recently proposed MobileNetV4 under similar latency constraints. Additionally, our method shows significant improvements in downstream tasks, including COCO object detection, instance segmentation, and ADE20k semantic segmentation, while still maintaining low latency on mobile devices for high-resolution inputs in these scenarios. Code and models are available at: https://github.com/ChuanyangZheng/iFormer.

1 INTRODUCTION

Building lightweight neural networks facilitates real-time analysis of images and videos captured by mobile applications such as smartphones. This not only enhances privacy protection and security by processing data locally on the device but also improves overall user experience.

Through the decades, convolutional neural networks (CNNs) (Krizhevsky et al., 2012; Szegedy et al., 2015; He et al., 2016) have emerged as the primary choice for balancing latency and performance on resource-constrained mobile devices. However, a significant limitation of CNNs is their reliance on a local sliding window mechanism, which imposes crucial inductive biases that may hinder modeling flexibility. Recently, the soaring development of vision transformers (ViTs) (Dosovitskiy et al., 2020) has begun to dominate various computer vision tasks, including image classification (Zhai et al., 2022), object detection (Liu et al., 2021a), and semantic segmentation (Xie et al., 2021). The core mechanism underlying ViTs is self-attention, which dynamically learns interactions between all image patches. This enables the model to focus on important regions adaptively and capture more global features. Nevertheless, deploying ViTs on mobile devices with limited resources poses significant challenges. On the one hand, the quadratic computational complexity of attention renders them unsuitable for large feature maps, which are common in the early stages of vi-

82

80

EfficientNet

| |
|---|

EfficientFormerV2

MobileNetV3

78

Top-1Accuracy(%)

MobileOne

FastViT

76

SwiftFormer

SHViT

| |
|---|

EfficientMod

74

RepViT

GhostNetV3

MobileViT

72

MobileNetV4

iFormer

70

0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 Latency (ms)

Figure 1: Comparison of latency and accuracy between our iFormer and other existing methods on ImageNet-1k. The latency is measured on an iPhone 13. Our iFormer is Pareto-optimal.

sion networks. On the other hand, the multi-head mechanism requires reshaping operations, leading to increased memory usage.

Many research efforts are devoted to combining the advantages of both CNNs and ViTs in designing lightweight networks while mitigating inefficient operations in mobile applications. Some studies (Zhang et al., 2023; Wang et al., 2024; Ma et al., 2024) revisit the architectural designs of lightweight CNNs from a ViT perspective and incorporate key components that contribute to the performance of ViTs into CNNs. Although these pure lightweight CNNs show improved performance compared to previous mobile networks (Howard et al., 2017; Zhang et al., 2018; Sandler

- et al., 2018), they still lag behind the powerful self-attention in ViTs. Another line of works (Mehta

- & Rastegari, 2021; Chen et al., 2022b; Li et al., 2023; Cai et al., 2023; Shaker et al., 2023; Vasu et al., 2023a; Qin et al., 2024) proposes innovative attention mechanisms to address the limitation of standard attention (Vaswani, 2017) and blend convolutions to achieve a better balance between latency and performance. These attention mechanisms either reduce the number of queries and keys (Shaker et al., 2023; Qin et al., 2024), limit the attention span (Wan et al., 2023), or adopt linear attention (Cai et al., 2023), which may compromise performance to some extent.

In this work, we present the iFormer, a herd of lightweight models that integrates the strengths of both CNNs and ViTs, achieving a state-of-the-art balance between latency and accuracy. Specifically, we employ a hierarchical architecture consisting of four stages. In the earlier, high-resolution stages, we utilize fast convolution to extract local representations. To construct the convolutional block, we start with a “modern” ConvNeXt (Liu et al., 2022), which incorporates a series of design decisions inspired by ViTs. Then we progressively “lighten” the ConvNeXt to create a streamlined lightweight network, optimizing it for real-time mobile latency on an iPhone 13, in contrast to the FLOPs and parameters used in prior works (Mehta & Rastegari, 2021; Chen et al., 2022b). This results in a fast convolutional architecture with strong performance. To further enhance the dynamic properties and its ability to model long-range contexts, we incorporate self-attention in the later low-resolution stages. However, direct implementation of standard multi-head self-attention (MHA) brings notable memory overheads and slows down inference speed on mobile devices. We identify that the increased latency stems primarily from the reshaping operations in MHA. More analyses reveal that multiple attention heads behave similarly. Therefore, we propose a simple yet effective single-head modulation self-attention (SHMA), which significantly minimizes memory costs while preserving strong performance. Fig. 4 provides an illustration of SHMA. In detail, SHMA learns spatial context interactions through optimized self-attention. Concurrently, a parallel feature extraction branch is employed to capture informative features. Finally, we fuse the outputs of these two branches to facilitate a more flexible and dynamic exchange of information, compensating for the slight performance degradation of the single-head attention when compared to MHA.

Benefiting from the fast local representation capacity of convolution and the efficient global modeling proficiency of the proposed SHMA, iFormer outperforms existing pure lightweight CNNs and hybrid networks across multiple visual recognition tasks, including image classification, object detection, instance segmentation, and semantic segmentation. For instance, in the context of image classification as shown in Fig. 1, iFormer-M achieves a Top-1 accuracy of 80.4% with only 1.10 ms on an iPhone 13 without advanced training strategies such as knowledge distillation (Touvron et al., 2021a) or reparameterization (Ding et al., 2021). Notably, our model obtains a 0.5% improvement in Top-1 accuracy compared to the recent MNV4-Conv-M (Qin et al., 2024), while being 1.4× faster than FastViT-SA12 (Vasu et al., 2023a) with similar accuracy. These results demonstrate the effectiveness of the proposed network in capturing both local and global feature representations.

2 RELATED WORK

- 2.1 EFFICIENT CONVOLUTIONAL NETWORKS

In the past 2010s, computer vision was dominated by CNNs, and so were efficient networks. The first remarkable breakthrough in mobile CNNs is MobileNets (Howard et al., 2017), which hatches the concept of decomposing standard convolution into depthwise and pointwise counterparts. Subsequently, MobileNetV2 (Sandler et al., 2018) introduces an inverted residual bottleneck block to push the state-of-the-art for mobile models. Numerous studies have aimed to accelerate CNNs via various approaches, such as channel shuffle in ShuffleNet (Zhang et al., 2018; Ma et al., 2018) and cheap linear transformations in GhostNet (Han et al., 2020). Meanwhile, Neural architecture search

(NAS) has emerged as a method for automating the design of neural networks, optimizing for performance under resource constraints. EfficientNet (Tan & Le, 2019), MobileNetV3 (Howard et al., 2019), and FBNet (Wu et al., 2019) all achieve rather good performance. Besides, MobileOne (Vasu

- et al., 2023b) proposes to train a model using reparameterizable branches, which are merged during inference. Recently, following the revolution of ViTs, several methods reexamine the design spaces and training strategies (Liu et al., 2024) for mobile CNNs. For instance, RepViT (Wang et al.,

2024) integrates efficient architectural designs from ViTs into MobileNetV3, outperforming existing lightweight CNNs. Other approaches, such as FocalNet (Yang et al., 2022a), Conv2Former (Hou

- et al., 2024), and EfficientMod (Ma et al., 2024), fuse features from context modeling and feature projection branches, also known as modulation mechanism, to enhance the model with dynamic properties analogous to attention. However, pure CNNs remain inherently spatially localized and their reliance on stationary weights restricts their flexibility. Although modulation can partially mitigate this limitation by enhancing dynamic capacity, they still exhibit deficiencies in building global interactions.

- 2.2 EFFICIENT VISION TRANSFORMERS

The success of Vision Transformer (Dosovitskiy et al., 2020) offers a compelling demonstration of the potential to apply transformer to computer vision tasks. Following this, ViT and its numerous variants (Liu et al., 2021a; Dong et al., 2022; Li et al., 2022a) sweep across various scenarios. However, the quadratic complexity of self-attention behind ViTs poses significant challenges for efficiency. The following researches seek to boost ViT efficiency through efficient attention mechanisms (Wang et al., 2021; Zhu et al., 2023; Hatamizadeh et al., 2023), model compression (Liu et al., 2021b; Zheng et al., 2022), knowledge distillation (Hao et al., 2021), and token reduction (Rao et al., 2021; Bolya et al., 2022). Recent studies further introduce ViTs into mobile applications. One mainstream of work combines efficient convolution and ViT to create lightweight hybrid networks (Mehta & Rastegari, 2022; Vasu et al., 2023a). MobileViT (Mehta & Rastegari, 2021) directly integrates MobileNetv2 blocks and ViT blocks, while Mobile-Former (Chen et al., 2022b) features a parallel design of MobileNet and ViT with a two-way bridge connecting the two. To further accelerate inference, some approaches replace the standard attention (Vaswani, 2017) with efficient variants within the hybrid networks. These include reducing the number of delegate tokens for computing attention (Pan et al., 2022), employing channel attention (Maaz et al., 2022), substituting projection in attention with efficient ghost modules (Ma et al., 2022), and utilizing linear attention mechanisms (Zhao et al., 2022). Besides manual designs, EfficientFormer (Li et al., 2022b; 2023) and MobileNetV4 (Qin et al., 2024) search for efficient architectures in a unified space encompassing both convolution operators and transformer operators. Another stream of work focuses on efficient attention mechanisms and directly employs them throughout the entire network (Shaker et al., 2023; Cai et al., 2023). For example, CMT (Guo et al., 2022) takes advantage of depth-wise convolution to downsample key and value to reduce computation. GhostNetV2 (Tang et al., 2022) applies two fully connected layers along the horizontal and vertical directions to compute attention, a decoupled version of MLP-Mixer (Tolstikhin et al., 2021). Recently, SHViT observes computational redundancy in the multi-head attention module and proposes to apply sing-head attention. In contrast to these existing approaches, we introduce a novel efficient attention module without sacrificing informative interactions, thereby maintaining strong representational capacity. Regarding attention design, ours is a bit similar to SHViT but is considerably superior as shown in Table 18 in the supplementary material. The key difference lies in the novel modulation attention. In addition, we explore efficient attention mechanisms in an on-device environment while SHViT focuses on general-purpose GPUs, fundamentally different hardware.

- 3 METHOD

We present the overall architecture of our iFormer in Fig. 4, which offers a Pareto-optimal accuracylatency trade-off on mobile applications. Our exploration towards a streamlined lightweight network unfolds as follows: 1) establishing the baseline and measure metric in Sec. 3.1. 2) exploring acceleration techniques consisting of macro and micro designs in Sec. 3.2. 3) injecting global attention in Sec. 3.3. Finally, we create a new family of efficient hybrid vision transformers tailored for mobile applications in Sec. 3.3. A detailed trajectory illustrating the evolution from a general hierarchical CNN to a fast hybrid vision transformer is depicted in Fig. 2.

Latency(ms)

0 0.5 1.0

| | | | | | |
|---|---|---|---|---|---|
| |1.10<br><br>1.10<br><br>1.15<br><br>1.12<br><br>1.00<br><br>0.99<br><br>1.05<br><br>0.77<br><br>1.08<br><br>1.07<br><br>80.40<br><br>80.20<br><br>80.40<br><br>79.80<br><br>79.40<br><br>79.10<br><br>78.80<br><br>77.10<br><br>76.70<br><br>74.90| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Baseline

conv stem

bn

Lightening

deeper

stage ratio

kernel sz.

SHA

SHMA

Attention

rd. width

PE

iFormer

0 75 80

Top-1 Acc.(%)

Figure 2: Illustration of the evolution from the ConvNeXt baseline towards the lightweight iFormer. The orange bars are model accuracies and the light blue bars are model latencies. We also include a red latency outline for better visualization.

- 3.1 PREPARING CONVNEXT

Our goal is to create an efficient multiscale network, where spatial dimensions of intermediate representations shrink as inference proceeds. In this hierarchical architecture, early network layers have larger spatial dimensions and fewer channels (e.g. 56×56×48), which renders them memorybound. Highly optimized convolution is more appropriate for these layers. Guided by this principle, we choose a pure convolutional network as our base architecture, specifically ConvNeXt (Liu et al.,

- 2022) which absorbed several key components from ViTs and competes favorably against ViTs. We gradually “lighten” the network to achieve a more favorable balance between latency and accuracy. For speed metric, we utilize on-device latency, measured on an actual iPhone 13 and compiled by Core ML Tools (CoreML), rather than FLOPs and parameter counts in previous methods (Mehta & Rastegari, 2021; Chen et al., 2022b; Zhang et al., 2022), which are not well correlated with latency. Regarding performance, we follow the training recipe in ConvNeXt while removing the layer scale to align prior methods (Li et al., 2022b; Wang et al., 2024) for a fair comparison. Please refer to Sec. B in the supplementary material for more details. To initiate our study, we systematically scale down the ConvNeXt by reducing the number of blocks and the width. This results in a lightweight model with a latency of 1.07 ms and a Top-1 accuracy of 74.9%, serving as our initial baseline.

- 3.2 LIGHTENING BASELINE

75

AverageCosineSimilarity(%)

70

65

60

55

50

45

14 16 18 20 22 24 Layer

Figure 3: The distribution of average cosine similarity among multiple heads within the MHA mechanism. As the layer depth increases, the similarity goes higher.

Table 1: Latency comparison between multi-head and single-head baseline.

Models Latency (ms) Top-1 Acc. (%)

MHA Baseline 1.40 79.9 SHA Baseline 1.12 (1.25×) 79.8

Seeing Better with Early Convolutions Following ViTs, ConvNeXt adopts an aggressive “patchify” strategy as the stem cell, specifically by splitting the input image into a series of nonoverlapping patches via a 4x4 non-overlapping convolutional layer. However, some studies (Xiao et al., 2021; Chen et al., 2022a) indicate that an early convolutional stem can increase optimization stability and facilitate faster model convergence. Moreover, compared to general models, lightweight models typically have fewer parameters and a reduced capacity. An aggressive nonoverlapping layer may lead to the premature loss of rich information. Consequently, we opt to replace the non-overlapping “patchify” stem with a stack of overlapping convolutional layers, as shown in Fig. 4. This modification elevates the top-1 accuracy to 76.7% with a neglectable increase in latency of 0.1 ms.

Normalization An obvious difference between ConvNeXt and previous CNNs is the normalization layer. ConvNeXt utilizes Layer Normalization (LN) (Ba et al., 2016), commonly used in Natural Language Processing (NLP), whereas the latter uses Batch Normalization (BN) (Ioffe, 2015). Albeit its superior performance, LN requires on-the-fly statistics calculation in inference along with

Stage 1 × 𝑁 Stage 2 × 𝑁 Stage 3 × 𝑁 Stage 3 × 𝑁 Stage 4 × 𝑁

[Figure 1]

[Figure 2]

[Figure 3]

Conv Block

Conv Stem

Conv Block

Conv Block

Attn Block

Attn Block 𝐵×3×𝐻×𝑊

Input

Head

𝐻 4

𝑊 4

𝐻 8

𝑊 8

𝐻 16

𝑊 16 𝐵×𝐶 ×

𝐻 32

𝑊 32

𝐵×𝐶 ×

×

𝐵×𝐶 ×

×

𝐵×𝐶 ×

×

×

|SHMA<br><br>Activation<br><br>Project<br><br>X<br><br>Project<br><br>Flatten<br><br>C R<br><br>×𝐿<br><br>Transpose<br><br>𝐿×<br><br>𝐶 𝑅<br><br>ProjectProject<br><br>Project<br><br>FlattenFlatten<br><br>ReshapeReshapeReshape<br><br>𝑁 ×𝐶 ×𝐿<br><br>Activation<br><br>𝑄<br><br>𝐾<br><br>𝑉<br><br>Reshape<br><br>Transpose<br><br>Reshape<br><br>𝐶 𝑅<br><br>×𝐻×𝑊<br><br>𝑆( )<br><br>𝐿×𝐿<br><br>𝐶×𝐿<br><br>𝐶×𝐻×𝑊<br><br>C×𝐿 / ∕ <br><br>𝑁 ×𝐶 ×𝐿<br><br>𝐶×𝐿<br><br>𝐶×𝐻×𝑊<br><br>𝐶×𝐿<br><br>𝐶 𝑅<br><br>×𝐿|
|---|

|Project<br><br>Conv Stem<br><br>Conv (k3, s2, )<br><br>3×𝐻×𝑊<br><br>𝐶 ×<br><br>𝐻 4<br><br>×<br><br>𝑊 4<br><br>Conv (k3, s2, 2𝐶 )|
|---|

|FFN<br><br>Attn Block<br><br>CPE<br><br>SHMA<br><br>𝐶×𝐻×𝑊|
|---|

|Conv Block<br><br>Conv (k3, s2)<br><br>Project 4C<br><br>Project C<br><br>𝐶×𝐻×𝑊<br><br>4𝐶×𝐻×𝑊<br><br>𝐶×𝐻×𝑊|
|---|

ReshapeReshapeReshape

Transpose

Project

Flatten

𝑁×𝐶×𝐿

Transpose

Activation

𝐶×𝐿

Reshape

Reshape

𝑁×𝐶×𝐿

C×𝐿/ ∕

ProjectProject

FlattenFlatten

Project

Figure 4: Overview of iFormer architecture, detailed convolutional stem, block design, and SHMA. The hatched area in SHMA indicates extra memory-intensive reshaping operations that are eliminated by SHMA. S(·) denotes the softmax function. R is the ratio for reducing channels of query and key. It is set to 2 in iFormer. We omit BN following project or convolution for simplicity.

division and square root operations, leading to inefficiency on mobile hardware (Yang et al., 2022b). On the contrary, BN operates with fixed statistics during inference as an offline method and can be seamlessly fused with other linear operations, providing a “free lunch”. This significantly reduces computational demands and memory overheads on mobile devices. Therefore, we substitute LN with BN throughout the network and merge it during inference. Additionally, we also substitute non-overlapping downsample layers with overlapping counterparts. These adjustments result in a reduction of overall latency to 0.77 ms while enhancing the Top-1 accuracy slightly to 77.10%.

Going Deeper There is considerable evidence indicating that increasing the depth of a model can enhance its capacity and yield performance benefits (Touvron et al., 2021b; Yang et al., 2022a). Most lightweight models typically stack more blocks to boost performance within constrained resources, as exemplified by the MobileNet series (Howard et al., 2019; Qin et al., 2024). In this study, we explore the potential of deepening ConvNeXt by increasing the number of blocks in each stage from (2,2,6,2) to (3,3,9,3). This increase in depth leads to a substantial improvement, raising the accuracy from 77.1% to 78.8%, although causing a temporary increase in latency to 1.05 ms.

Stage Ratio The stage ratio in ConNeXt is not optimized for lightweight models. A substantial number of depthwise convolutions in the early stages incurs significant memory transfer costs. Meanwhile, the presence of many blocks with a channel expansion ratio of 4 in the Feed-Forward Network (FFN) in the last stage, which already has a high channel dimension, imposes substantial computational demands. These factors lead to a sub-optimal allocation of computational resources. To address these issues, we propose reallocating more computational resources to the third stage while reducing memory access costs in the early stage. Specifically, the blocks in each stage are adjusted from (3,3,9,3) to (2,2,18,2). As expected, this achieves a better balance between latency and performance, with Top-1 accuracy increasing to 79.1% while enjoying a lower latency of 1.01ms.

Kernel Size Here we examine the effects of different kernel sizes in mobile settings and observe that utilizing a larger kernel size introduces nearly no latency burden, as shown in

### Table 2: Latency under different convolutional kernel sizes.

- Table 2. So we maintain the convolutional kernel size at 7×7 in each basic block, consistent with ConvNeXt. Furthermore, previous approaches use a kernel size of 3×3 in the convolutional stem. This small receptive field may hinder feature representation during the early downsampling process. As previously noted, the early layers are memory-bound, allowing for opportunities to employ computeintensive operations (i.e., dense convolution). Therefore, we enlarge the kernel size of the dense

Kernel Size Latency (ms)

3×3 1.00 7×7 1.01

convolutional layer in the stem cell to 5×5. As illustrated in Fig. 2, this change has no impact on inference latency while enhancing Top-1 accuracy by 0.3%.

- 3.3 SINGLE-HEAD MODULATION ATTENTION

Single-Head vs. Multi-Head ViTs typically apply MHA, which projects the queries, keys, and values multiple times with different learnable linear projections and performs multiple attention functions simultaneously. In practice, the multi-head mechanism requires the reshaping of feature maps first, causing large memory access and transfer costs. This can seriously impact inference latency, especially on resource-constrained mobile devices. To investigate this issue, we substitute the last half of the convolutional blocks in the third stage and all blocks in the last stage with standard ViT blocks, as depicted in Fig. 4. We refer to this hybrid network as the MHA baseline. Next, we build another network by substituting the MHA with Single-Head self-Attention (SHA), referring to it as the SHA baseline. The comparison is shown in Table 1. The SHA baseline shows a 1.25× acceleration over its MHA counterpart on the iPhone 13. This verifies that additional reshaping operations in MHA incur significant memory access costs, leading to a considerable decline in inference speed.

This naturally calls for optimizing MHA. Recent methods (Pan et al., 2022; Qin et al., 2024) primarily focus on downsampling the query or the key, which may hurt global attention capacity. Instead, we aim to reduce the redundant reshaping of MHA while preserving all token-to-token interactions. Previous works (Michel et al., 2019; Yun & Ro, 2024) indicate that a single attention head can approach the performance of multiple heads in general plain transformer models, such as DeiT. To investigate this on the mobile application, we analyze the average cosine similarity of multiple heads within the same layer of the aforementioned MHA baseline, which is a hierarchical lightweight network, and present our findings in Fig. 3. We clearly see that the average cosine similarity reaches 50% and even 75% in the final layer. Furthermore, the SHA baseline, as shown in Table 1, exhibits only a negligible accuracy drop of 0.1%. These suggest that SHA achieves a more favorable balance between accuracy and latency, obtaining an accuracy of 79.8% with a latency of 1.12 ms.

Modulation Attention We further introduce a novel modulation attention to boost performance and strengthen flexibility in modeling, as illustrated in Fig. 4. Formally, we start from the abstracted modulation mechanism (Ma et al., 2024), similar to the gate mechanism Shazeer (2020). Assume we are given an input feature map x ∈ RC×H×W where C, H, and W denote the channels, height, and width of the feature map. The modulated output can be written as follows:

xo = f(x) ⊙ ctx(x), (1)

where f(·) denotes the feature mapping branch and ctx(·) is the context modeling branch. The output xo is the fused features from both branches via efficient element-wise multiplication. The key idea of our approach is to modulate the feature using SHA instead of relying on convolutional layers, as seen in previous works (Yang et al., 2022a; Ma et al., 2024). Since SHA captures global interactions through self-attention, it excels in extracting rich contextual information and better controlling the flow of information. This process can be expressed as follows:

### ctx(x) = SHA(WQx,WKx,WVx), (2)

where WQ, WK, WV are the project weights for query, key, and value, respectively. For simplicity, we omit the bias term. To minimize inference costs, we utilize a single projection layer in the feature mapping branch. To enhance expressivity and improve optimization stability, we apply individual nonlinear activation functions to both branches, as follows:

xo = σ(WMx) ⊙ σ(ctx(x)), (3)

where σ is the sigmoid function and WM denotes the feature projection weight. We also experiment with various activation functions for modulation in Sec. 5 and observe that the sigmoid works rather well. Finally, the output from the modulation attention is projected in a manner as standard attention.

Equipped with Single-Head Modulation Attention (SHMA), our model improves the accuracy to 80.4% with an intermediate latency of 1.15 ms. This performance notably surpasses that of the recent MobileNetV4, which achieves an accuracy of 79.9%.

Reducing Width Until now, we have developed a lightweight network that performs pretty well, but at a bit slow speed. To push the trade-off toward the state-of-the-art, we revise the width configuration in the SHMA. The modulation mechanism enriches the output by enabling more dynamic modeling in both spatial and channel dimensions, making it possible to use a weaker SHA and FFN. In light of this, we reduce the head dimension in the SHMA (i.e., WQ,WK) to a small factor of the feature dimension, further details can be found in Table 15 in the supplementary material. Simultaneously, we shrink the expansion ratio in FFN following SHMA from 4 to 3. This process obtains a lower latency of 1.10 ms, although a slight drop of 0.2% in accuracy.

Positional Embedding Last but not least, positional information plays a crucial role in selfattention as it regards input as a set of tokens. Adding positional embedding will help the attention learn permutation-variant features. We apply conditional positional encodings (CPE) (Chu et al., 2021) that are dynamically generated and conditioned on the local neighborhood of the input tokens, as illustrated in Fig. 4. The integration of CPE further enhances our model’s performance, achieving a Top-1 accuracy of 80.4% with only 1.10 ms, establishing a state-of-the-art trade-off.

iFormer The result of these modifications is an extremely fast and efficient hybrid network, which we denote iFormer. The overall architecture is depicted inFig. 4. It integrates fast local convolutional layers in the early stages that operate on higher resolution and global SHMA in later stages which processes lower resolution. Besides, we create a series of iFormer models tailored to various hardware resource constraints. For detailed architectural hyperparameters of these model variants, please refer to Table 15 in the supplementary material.

- 4 EXPERIMENTS

- 4.1 IMAGE CLASSIFICATION

Settings. We first evaluate our models on classification on ImageNet-1K (Deng et al., 2009). To ensure a fair comparison with prior studies, we follow the previous training recipe (Touvron et al., 2021a; Liu et al., 2022) and train all models for 300 epochs with a standard image size of 224x224. Please refer to Sec. B in the supplementary material for details. Besides Top-1 validation accuracy, we also report the latency measured on an iPhone 13 with models compiled by Core ML Tools (CoreML) under a batch size of 1, as done in (Li et al., 2023; Wang et al., 2024; Vasu et al.,

- 2023b). It’s worth highlighting that we do not apply any advanced strategies such as distillation (Li et al., 2023) and reparameterization (Ding et al., 2021).

- Table 3 summarizes a comparison between our iFormer and state-of-the-art lightweight models, organized by latency. iFormer demonstrates a Paretooptimal trade-off between accuracy and latency. For example, iFormer-M obtains 80.4% top-1 accuracy with a latency of only 1.1 ms, surpassing recent MobileNetV4-Conv-M and RepViT-M1 by 0.5% and 1.0%, respectively. This is noteworthy considering that MobileNetV4 requires a longer training schedule (500 vs. 300) and takes a larger input resolution (256 vs. 224). When compared to other recent models using reparameterization, including FastViT-T12, GhostNetV3-1.3×, and MobileOne-S3, iFormer-M achieves superior accuracy while maintaining lower latency. Moreover, iFormer outperforms various hybrid networks. Thanks to the efficient SHMA, iFormer-L achieves more outstanding performance than other attention variants, such as multi-query attention in MNV4-Hybrid-M, additive attention in SwiftFormer-L1, and linear attention in EfficientVIT-B1-r288.

Table 4: Results with distillation on ImageNet-1K. * indicates the model is trained with a strong training strategy (i.e., reparameterization).

Model Latency (ms) Reso. Epochs Top-1 (%)

- EfficientFormerV2-S1 (2023) 1.02 224 300 79.0

- EfficientFormerV2-S1 (2023) 1.02 224 450 79.7 MobileViGv2-S*(2024) 1.24 224 300 79.8

FastViT-T12* (2023a) 1.12 256 300 80.3 RepViT-M1.1* (2024) 1.04 224 300 80.7

iFormer-M 1.10 224 300 81.1 SHViT-S4 (2024) 1.48 224 300 80.2

- EfficientFormerV2-S2 (2023) 1.60 224 300 81.6 MobileViGv2-M(2024) 1.70 224 300 81.7 FastViT-SA12* (2023a) 1.50 256 300 81.9

- EfficientFormerV2-S2 (2023) 1.60 224 450 82.0 RepViT-M1.5* (2024) 1.54 224 300 82.3

iFormer-L 1.60 224 300 82.7

Table 3: Classification results on ImageNet-1K. † indicates models that are trained with a variety of advanced training strategies including complex reparameterization, distillation, optimizer, and so on. We provide a more comprehensive comparison in Sec. G in the supplementary material.

Latency ↓ (ms)

Reso. Epochs Top-1 (%) MobileNetV2 1.0x (2018) 3.4 0.30 0.73 224 500 72.0

Model Params (M) GMACs

- MobileNetV3-Large 0.75x (2019) 4.0 0.16 0.67 224 600 73.3 MNV4-Conv-S (2024) 3.8 0.20 0.60 224 500 73.8

iFormer-T 2.9 0.53 0.60 224 300 74.1 MobileNetV2 1.4x (2018) 6.9 0.59 1.02 224 500 74.7

- MobileNetV3-Large 1.0x (2019) 5.4 0.22 0.76 224 600 75.2 SwiftFormer-XS (2023) 3.5 0.60 0.95 224 300 75.7 SBCFormer-XS (2024) 5.6 0.70 0.79 224 300 75.8

GhostNetV3 1.0x† (2024) 6.1 0.17 0.99 224 600 77.1

- MobileOne-S2 (2023b) 7.8 1.30 0.92 224 300 77.4 RepViT-M1.0 (2024) 6.8 1.10 0.85 224 300 78.6

iFormer-S 6.5 1.09 0.85 224 300 78.8 EfficientMod-xxs (2024) 4.7 0.60 1.29 224 300 76.0

SBCFormer-S (2024) 8.5 0.90 1.02 224 300 77.7

- MobileOne-S3 (2023b) 10.1 1.90 1.16 224 300 78.1 SwiftFormer-S (2023) 6.1 1.00 1.12 224 300 78.5

GhostNetV3 1.3x† (2024) 8.9 0.27 1.24 224 600 79.1 FastViT-T12 (2023a) 6.8 1.40 1.12 256 300 79.1 RepViT-M1.1 (2024) 8.2 1.30 1.04 224 300 79.4

MNV4-Conv-M (2024) 9.2 1.00 1.08 256 500 79.9 iFormer-M 8.9 1.64 1.10 224 300 80.4

Mobile-Former-294M (2022b) 11.4 0.29 2.66 224 450 77.9

MobileViT-S (2021) 5.6 2.00 3.55 256 300 78.4 MobileOne-S4 (2023b) 14.8 2.98 1.74 224 300 79.4

SBCFormer-B (2024) 13.8 1.60 1.44 224 300 80.0

GhostNetV3 1.6x† (2024) 12.3 0.40 1.49 224 600 80.4 EfficientViT-B1-r288 (2023) 9.1 0.86 3.87 288 450 80.4

FastViT-SA12 (2023a) 10.9 1.90 1.50 256 300 80.6 MNV4-Hybrid-M (2024) 10.5 1.20 1.75 256 500 80.7 SwiftFormer-L1 (2023) 12.1 1.60 1.60 224 300 80.9 EfficientMod-s (2024) 12.9 1.40 2.57 224 300 81.0 RepViT-M1.5 (2024) 14.0 2.30 1.54 224 300 81.2 iFormer-L 14.7 2.63 1.60 224 300 81.9

Results with distillation on ImageNet-1K. We conducted rigorously fair training as the previous methods above. Recently, some works report enhanced performance leveraging more advanced training strategies. We investigate whether these training recipes can also improve iFormer. Following previous works (Li et al., 2023; Wang et al., 2024), we employ the RegNetY-16GF (Radosavovic et al., 2020) model with a top-1 accuracy of 82.9% as the teacher model for distillation. Our findings reveal that iFormer improves obviously over its counterpart without distillation. For example, iFormer-L shows a 1.0% increase under the same latency. iFormer also outperforms EfficientFormerV2-S2, despite the latter being trained with a 1.5× longer schedule.

- 4.2 OBJECT DETECTION AND INSTANCE SEGMENTATION

To validate the effectiveness of iFormer on downstream tasks, we train Mask R-CNN (He et al., 2017) with iFormer as the backbone for 12 epochs (1×), using the MMDetection toolkit (Chen

- et al., 2019). We also report backbone latency measured at a resolution of 512×512 on an iPhone

13. The results are presented in Table 5. In comparison to lightweight models, iFormer-M surpasses FastViT-SA12 by +1.9%/+2.0% in APbox /APmask while running 1.32× faster. iFormer-L also obtains +0.1%/+0.6% in APbox /APmask than EfficientMod-S, which utilizes a convolutional modulation mechanism to learn dynamics similar to self-attention. Notably, EfficientMod-S operates

- 3.7× slower when processing high-resolution input, underscoring that the proposed novel attention mechanism is more suitable for mobile networks. Meanwhile, when compared to general networks that are not optimized for mobile applications, iFormer demonstrates significant advantages. For instance, iFormer-L exceeds the performance of ConvNeXt-T with improvements of +1.2%/+1.4% in

Table 5: Object detection & instance segmentation results on MS COCO 2017 using Mask RCNN. Semantic segmentation results on ADE20K using the Semantic FPN framework. We measure all backbone latencies with image crops of 512×512 on iPhone 13 by Core ML Tools. Failed indicated that the model runs too long to report latency by the Core ML.

Param (M)

Latency ↓ (ms)

Object Detection Instance Segmentation Semantic APbox APbox50 APbox75 APmask APmask50 APmask75 mIoU

Backbone

EfficientNet-B0 (2019) 5.3 4.55 31.9 51.0 34.5 29.4 47.9 31.2 -

ResNet18 (2016) 11.7 2.85 34.0 54.0 36.7 31.2 51.0 32.7 32.9 PoolFormer-S12 (2022) 11.9 5.70 37.3 59.0 40.1 34.6 55.8 36.9 37.2

EfficientFormer-L1 (2022b) 12.3 3.50 37.9 60.3 41.0 35.4 57.3 37.3 38.9 FastViT-SA12 (2023a) 10.9 5.27 38.9 60.5 42.2 35.9 57.6 38.1 38.0 RepViT-M1.1 (2024) 8.2 3.18 39.8 61.9 43.5 37.2 58.8 40.1 40.6

iFormer-M 8.9 4.00 40.8 62.5 44.8 37.9 59.7 40.7 42.4 ResNet50 (2016) 25.5 7.20 38.0 58.6 41.4 34.4 55.1 36.7 36.7

PoolFormer-S24 (2022) 21.4 10.0 40.1 62.2 43.4 37.0 59.1 39.6 40.3 ConvNeXt-T (Liu et al., 2022) 29.0 13.6 41.0 62.1 45.3 37.7 59.3 40.4 41.4

EfficientFormer-L3 (2022b) 31.3 8.40 41.4 63.9 44.7 38.1 61.0 40.4 43.5 RepViT-M1.5 (2024) 14.0 5.00 41.6 63.2 45.3 38.6 60.5 41.5 43.6

PVTv2-B1 (2022) 14.0 27.00 41.8 64.3 45.9 38.8 61.2 41.6 42.5 FastViT-SA24 (2023a) 20.6 8.97 42.0 63.5 45.8 38.0 60.5 40.5 41.0 EfficientMod-S (2024) 32.6 24.30 42.1 63.6 45.9 38.5 60.8 41.2 43.5

Swin-T (2021a) 28.3 Failed 42.2 64.4 46.2 39.1 61.6 42.0 41.5 iFormer-L 14.7 6.60 42.2 64.2 46.0 39.1 61.4 41.9 44.5

APbox /APmask, while requiring fewer parameters and only 50% mobile latency, suggesting iFormer’s efficient design in feature extraction and strong potential for mobile applications.

- 4.3 SEMANTIC SEGMENTATION

We conduct experiments on the ADE20K (Zhou et al., 2017) using the Semantic FPN (Kirillov et al., 2019), based on the MMSegmentation toolkit (Contributors, 2020). Thanks to its efficient attention design, iFormer outperforms all competing methods in mIoU with similar and much lower latency. For example, iFormer-L surpasses FastViT-SA24 by +3.5% in mIoU with a 1.36× faster inference speed. In addition, iFormer-M demonstrates superior mIoU compared to general networks, which typically exhibit substantially greater latency when processing higher-resolution inputs on mobile devices. Although PVTv2-B utilizes downsampled attention, it still requires 27 ms for latency. Similarly, Swin-T involves intensive operations in window partitioning, making it less suitable for mobile applications. Running at 6.6 ms, iFormer-L achieves +2.0% better mIoU than PVTv2-B1 and +3.0% better than Swin-T. These results suggest that the proposed attention mechanism offers significant benefits for tasks requiring the perception of fine-grained details.

- 5 ABLATION STUDIES

Activation Function Here we explore whether an activation function without an upper bound can enhance the SHMA by allowing neurons to express arbitrarily large values. We compare the widely used Sigmoid Linear Unit (SiLU) (Shazeer, 2020) with the sigmoid function and present the results in Table 6. Directly replacing the activation function in SHMA with SiLU will encounter diverging loss during training. The underlying cause is primarily attributed to the element-wise multiplication of the unbounded context branch. To address this, we replace Post-BN in SHMA with Pre-LN, as LN adaptively normalizes each token feature. The modified model experiences a slight decrease in accuracy but incurs an additional 0.07 ms latency, primarily brought by LN. The results suggest that the sigmoid function not only mitigates training instability but also facilitates better convergence.

Table 6: Activation function comparison in SHMA. PostBN indicates that BN is applied after projection. Pre-LN means that LN is implemented before the projection, as in standard MHA (Vaswani, 2017).

SHMA Setting Params (M) GMACs Latency (ms) Top-1 Acc. (%) SiLU + Post-BN 8.9 1.60 1.10ms Diverged

SiLU + Pre-LN 8.9 1.64 1.17ms 80.3 Sigmoid + Post-BN 8.9 1.60 1.10ms 80.4

Choice of Conv v.s. ViT Blcoks In Section 3.3, we replace the convolutional blocks in Stages 3 and 4 with the proposed SHMA block. We provide further ablation studies on the choice of ratio for the ViT blocks. Specifically, We choose the model after enlarging the kernel size as a starting point, then we progressively replace the convolutional blocks in Stages 3 and 4. We do not modify Stages 1 and 2 as their larger spatial dimensions would considerably increase the memory requirements for the self-attention mechanism.

### Table 7: Different ratio of ViT Block.

Ratio Setting Params (M) GMACs Latency (ms) Top-1 Acc. (%)

Baseline 9.4M 1760M 1.0ms 79.4 Replacing 22% Conv Blocks in Stage 3 as SHA 9.1M 1724M 1.02ms 79.5 Replacing 22% Conv Blocks in Stage 3 as SHMA 9.2M 1739M 1.04ms 79.6 Replacing 50% Conv Blocks in Stage 3 as SHA 8.8M 1689M 1.04ms 79.5 Replacing 50% Conv Blocks in Stage 3 as SHMA 8.9M 1712M 1.07ms 79.8 Replacing 78% Conv Blocks in Stage 3 as SHA 8.3M 1635M 1.12ms 79.3 Replacing 78% Conv Blocks in Stage 3 as SHMA 8.5M 1685M 1.17ms 79.6 Replacing 100% Conv Blocks in Stage 3 as SHA 7.9M 1599M 1.17ms 78.1 Replacing 100% Conv Blocks in Stage 3 as SHMA 8.3M 1665M 1.25ms 79.0

Replacing 100% Conv Blocks in Stage 3 as SHMA and 100% in Stage 4 10.0M 1792M 1.15ms 80.4

We present our findings in Table 7. Given that Stage 4 contains only two blocks, we do not conduct further splitting for the ratio. As illustrated in Table 7, although the ViT block has lower FLOPs, it still incurs increased runtime. Substituting all the convolutional blocks in Stage 3 results in the worst performance and the highest latency. Instead, by replacing half of the convolutional blocks in the third stage and all blocks in the final stage, we can better integrate these two operators, thus achieving a favorable trade-off between accuracy and latency.

Scaling to Larger Model Although iFormer is designed for mobile-device applications, the combination of fast local representation capacity of convolution and the efficient global modeling proficiency of the proposed SHMA enables its scalability for a broader range of applications. To demonstrate the scalability of iFormer, we developed a larger model named iFormer-H with 99M parameters and trained it for 300 epochs following the same strategy outlined in Section B. It is important to note that we add drop path and layer scale, which are commonly used in the training of larger models (Liu et al., 2022; Tu et al., 2022; Shi, 2024).

We summarize the results in Table 8. A highlight from the results is that iFormer is not specifically designed or trained for this scale. Despite this, iFormer-H outperforms ConvNeXt, achieving a 1.0% increase in accuracy while maintaining a similar number of FLOPs. Additionally, it demonstrates comparable performance to TransNeXt-Base, despite utilizing fewer FLOPs. These findings indicate the potential for broader applications of iFormer. We plan to explore larger models suitable for mobile devices in future work. Further ablation studies can be found in Sec. C in the supplementary material.

### Table 8: Scaling to the larger model with 99M parameters.

Model Params (M) GMACs (G) Top-1 Acc. (%)

ConvNeXt-Base (2022) 89 15.4 83.8 TransNeXt-Base (2024) 90 18.4 84.8

iFormer-H 99 15.5 84.8 MaxViT-Base (2022) 120 24.0 84.9

- 6 CONCLUSION

This work proposes iFormer, which integrates highly optimized convolutional operations for the early layers alongside a novel and efficient single-head modulation attention for the later layers. iFormer achieves SOTA Pareto-front in terms of Top-1 accuracy and mobile latency. We also validate the effectiveness of iFormer on downstream dense prediction tasks, including COCO object detection, instance segmentation, and ADE20K semantic segmentation. These inspiring results highlight the potential for mobile applications. We hope iFormer can facilitate the application of artificial intelligence on more mobile devices. In future work, we will seek to alleviate inference bottlenecks sociated with high-resolution images. Meanwhile, we plan to optimize iFormer for more hardware platforms, such as Android devices and NVIDIA Jetson Nano.

REFERENCES

William Avery, Mustafa Munir, and Radu Marculescu. Scaling graph convolutions for mobile vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5857–5865, 2024.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.

Han Cai, Junyan Li, Muyan Hu, Chuang Gan, and Song Han. Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17302–17313, 2023.

Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019.

Qiang Chen, Qiman Wu, Jian Wang, Qinghao Hu, Tao Hu, Errui Ding, Jian Cheng, and Jingdong Wang. Mixformer: Mixing features across windows and dimensions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5249–5259, 2022a.

Yinpeng Chen, Xiyang Dai, Dongdong Chen, Mengchen Liu, Xiaoyi Dong, Lu Yuan, and Zicheng Liu. Mobile-former: Bridging mobilenet and transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5270–5279, 2022b.

Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, and Chunhua Shen. Conditional positional encodings for vision transformers. arXiv preprint arXiv:2102.10882, 2021.

MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox

and benchmark. https://github.com/open-mmlab/mmsegmentation, 2020. CoreML. In https://github.com/apple/coremltools. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hi-

erarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13733–13742, 2021.

Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining Guo. Cswin transformer: A general vision transformer backbone with cross-shaped windows. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12124–12134, 2022.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Jianyuan Guo, Kai Han, Han Wu, Yehui Tang, Xinghao Chen, Yunhe Wang, and Chang Xu. Cmt: Convolutional neural networks meet vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12175–12185, 2022.

Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, and Chang Xu. Ghostnet: More features from cheap operations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1580–1589, 2020.

Zhiwei Hao, Jianyuan Guo, Ding Jia, Kai Han, Yehui Tang, Chao Zhang, Han Hu, and Yunhe Wang. Learning efficient vision transformers via fine-grained manifold distillation. arXiv preprint arXiv:2107.01378, 2021.

Ali Hatamizadeh, Greg Heinrich, Hongxu Yin, Andrew Tao, Jose M Alvarez, Jan Kautz, and Pavlo Molchanov. Fastervit: Fast vision transformers with hierarchical attention. arXiv preprint arXiv:2306.06189, 2023.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pp. 2961–2969, 2017.

Qibin Hou, Cheng-Ze Lu, Ming-Ming Cheng, and Jiashi Feng. Conv2former: A simple transformerstyle convnet for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1314–1324, 2019.

Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Sergey Ioffe. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

Alexander Kirillov, Ross Girshick, Kaiming He, and Piotr Doll´ar. Panoptic feature pyramid networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6399–6408, 2019.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Mvitv2: Improved multiscale vision transformers for classification and detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4804–4814, 2022a.

Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, and Jian Ren. Efficientformer: Vision transformers at mobilenet speed. Advances in Neural Information Processing Systems, 35:12934–12949, 2022b.

Yanyu Li, Ju Hu, Yang Wen, Georgios Evangelidis, Kamyar Salahi, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Rethinking vision transformers for mobilenet size and speed. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16889–16900, 2023.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021a.

Zhenhua Liu, Yunhe Wang, Kai Han, Wei Zhang, Siwei Ma, and Wen Gao. Post-training quantization for vision transformer. Advances in Neural Information Processing Systems, 34:28092– 28103, 2021b.

Zhenhua Liu, Zhiwei Hao, Kai Han, Yehui Tang, and Yunhe Wang. Ghostnetv3: Exploring the training strategies for compact models. arXiv preprint arXiv:2404.11202, 2024.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11976–11986, 2022.

Xiangyong Lu, Masanori Suganuma, and Takayuki Okatani. Sbcformer: Lightweight network capable of full-size imagenet classification at 1 fps on single board computers. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1123–1133, 2024.

Hailong Ma, Xin Xia, Xing Wang, Xuefeng Xiao, Jiashi Li, and Min Zheng. Mocovit: Mobile convolutional vision transformer. arXiv preprint arXiv:2205.12635, 2022.

Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. Shufflenet v2: Practical guidelines for efficient cnn architecture design. In Proceedings of the European conference on computer vision (ECCV), pp. 116–131, 2018.

Xu Ma, Xiyang Dai, Jianwei Yang, Bin Xiao, Yinpeng Chen, Yun Fu, and Lu Yuan. Efficient modulation for vision networks. arXiv preprint arXiv:2403.19963, 2024.

Muhammad Maaz, Abdelrahman Shaker, Hisham Cholakkal, Salman Khan, Syed Waqas Zamir, Rao Muhammad Anwer, and Fahad Shahbaz Khan. Edgenext: efficiently amalgamated cnntransformer architecture for mobile vision applications. In European conference on computer vision, pp. 3–20. Springer, 2022.

Sachin Mehta and Mohammad Rastegari. Mobilevit: light-weight, general-purpose, and mobilefriendly vision transformer. arXiv preprint arXiv:2110.02178, 2021.

Sachin Mehta and Mohammad Rastegari. Separable self-attention for mobile vision transformers. arXiv preprint arXiv:2206.02680, 2022.

Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? Advances in neural information processing systems, 32, 2019.

Moritz Nottebaum, Matteo Dunnhofer, and Christian Micheloni. Lowformer: Hardware efficient design for convolutional transformer backbones. arXiv preprint arXiv:2409.03460, 2024.

Junting Pan, Adrian Bulat, Fuwen Tan, Xiatian Zhu, Lukasz Dudziak, Hongsheng Li, Georgios Tzimiropoulos, and Brais Martinez. Edgevits: Competing light-weight cnns on mobile devices with vision transformers. In European Conference on Computer Vision, pp. 294–311. Springer, 2022.

Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, et al. Mobilenetv4-universal models for the mobile ecosystem. arXiv preprint arXiv:2404.10518, 2024.

Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Doll´ar. Designing network design spaces. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10428–10436, 2020.

Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34:13937–13949, 2021.

Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4510–4520, 2018.

Abdelrahman Shaker, Muhammad Maaz, Hanoona Rasheed, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Swiftformer: Efficient additive attention for transformer-based real-time mobile vision applications. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17425–17436, 2023.

Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Dai Shi. Transnext: Robust foveal visual perception for vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17773–17783, 2024.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1–9, 2015.

Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pp. 6105–6114. PMLR, 2019.

Yehui Tang, Kai Han, Jianyuan Guo, Chang Xu, Chao Xu, and Yunhe Wang. Ghostnetv2: Enhance cheap operation with long-range attention. Advances in Neural Information Processing Systems, 35:9969–9982, 2022.

Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, et al. Mlp-mixer: An all-mlp architecture for vision. Advances in neural information processing systems, 34:24261– 24272, 2021.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pp. 10347–10357. PMLR, 2021a.

Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Herv´e J´egou. Going deeper with image transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 32–42, 2021b.

Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer. In European conference on computer vision, pp. 459–

479. Springer, 2022.

Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan. Fastvit: A fast hybrid vision transformer using structural reparameterization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5785–5795, 2023a.

Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan. Mobileone: An improved one millisecond mobile backbone. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7907–7917, 2023b.

Ashish Vaswani. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017. Qiang Wan, Zilong Huang, Jiachen Lu, Gang Yu, and Li Zhang. Seaformer: Squeeze-enhanced

axial transformer for mobile semantic segmentation. arXiv preprint arXiv:2301.13156, 2023.

Ao Wang, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding. Repvit: Revisiting mobile cnn from vit perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15909–15920, 2024.

Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 568–578, 2021.

Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pvt v2: Improved baselines with pyramid vision transformer. Computational Visual Media, 8(3):415–424, 2022.

Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, and Kurt Keutzer. Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10734–10742, 2019.

Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Doll´ar, and Ross Girshick. Early convolutions help transformers see better. Advances in neural information processing systems, 34:30392–30400, 2021.

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34:12077–12090, 2021.

Jianwei Yang, Chunyuan Li, Xiyang Dai, and Jianfeng Gao. Focal modulation networks. Advances in Neural Information Processing Systems, 35:4203–4217, 2022a.

Qiming Yang, Kai Zhang, Chaoxiang Lan, Zhi Yang, Zheyang Li, Wenming Tan, Jun Xiao, and Shiliang Pu. Unified normalization for accelerating and stabilizing transformers. In Proceedings of the 30th ACM International Conference on Multimedia, pp. 4445–4455, 2022b.

Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10819–10829, 2022.

Seokju Yun and Youngmin Ro. Shvit: Single-head vision transformer with memory efficient macro design. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5756–5767, 2024.

Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12104–12113, 2022.

Haokui Zhang, Wenze Hu, and Xiaoyu Wang. Edgeformer: Improving light-weight convnets by learning from vision transformers. arXiv preprint arXiv:2203.03952, 2, 2022.

Jiangning Zhang, Xiangtai Li, Jian Li, Liang Liu, Zhucun Xue, Boshen Zhang, Zhengkai Jiang, Tianxin Huang, Yabiao Wang, and Chengjie Wang. Rethinking mobile block for efficient attention-based models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1389–1400. IEEE Computer Society, 2023.

Tianfang Zhang, Lei Li, Yang Zhou, Wentao Liu, Chen Qian, and Xiangyang Ji. Cas-vit: Convolutional additive self-attention vision transformers for efficient mobile applications. arXiv preprint arXiv:2408.03703, 2024.

Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6848–6856, 2018.

Mingshu Zhao, Yi Luo, and Yong Ouyang. Repnext: A fast multi-scale cnn using structural reparameterization. arXiv preprint arXiv:2406.16004, 2024.

Youpeng Zhao, Huadong Tang, Yingying Jiang, Qiang Wu, et al. Lightweight vision transformer with cross feature attention. arXiv preprint arXiv:2207.07268, 2022.

Chuanyang Zheng, Kai Zhang, Zhi Yang, Wenming Tan, Jun Xiao, Ye Ren, Shiliang Pu, et al. Savit: Structure-aware vision transformer pruning via collaborative optimization. Advances in Neural Information Processing Systems, 35:9010–9023, 2022.

Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 633–641, 2017.

Lei Zhu, Xinjiang Wang, Zhanghan Ke, Wayne Zhang, and Rynson WH Lau. Biformer: Vision transformer with bi-level routing attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10323–10333, 2023.

- A APPENDIX
- B EXPERIMENTAL SETTINGS

- B.1 IMAGE CLASSIFICATION

Table 9: ImageNet-1K training settings. training config iFormer-T/S/M/L/H resolution 2242 weight init trunc. normal (0.2) optimizer AdamW base learning rate 4e-3 (T/S/M/L) 8e-3 weight decay 0.05 optimizer momentum β1,β2=0.9,0.999 batch size 4096 [T/S/M/L] 8192 [H] training epochs 300 learning rate schedule cosine decay warmup epochs 20 warmup schedule linear layer-wise lr decay None randaugment (9, 0.5) mixup 0.8 cutmix 1.0 random erasing 0.25 label smoothing 0.1 stochastic depth 0.0 [T/S/M] 0.1 [L] 0.6 [H] layer scale None [T/S/M/L] 1e-6 [H] head init scale None gradient clip None exp. mov. avg. (EMA) None

We mainly follow the training recipe of ConvNeXt, while removing stochastic depth, layer scale, and exponential moving average to ensure a fair comparison with prior works. The models are trained for 300 epochs on 8 NVIDIA GPUs with a total batch size of 4096. We employ the same learning rate across all models. It is possible to further improve performance by adjusting the learning rates for different model variants, which we will explore in the future.

For distillation, we use the RegNetY-16GF model as the teacher model and apply a hard distillation loss, following the approach of DeiT (Touvron et al., 2021a). During inference, the average output of the classification head and the distillation head is used as the final output.

- B.2 OBJECT DETECTION AND SEMANTIC SEGMENTATION

For object detection experiments, we train MaskR-CNN models on the COCO 2017 dataset for 12 epochs using standard training settings from the MMDetection toolkit.

For semantic segmentation experiments, we train Semantic FPN models on the ADE20K dataset for 40,000 iterations using standard training settings from the MMSegmentation toolkit. The input images are cropped to a resolution of 512×512 during training.

For backbone latency, we keep the same input size as training (i.e., 512×512) and measure the mobile latency on an iPhone 13 compiled by Core ML Tools.

- C MORE ABLATION STUDIES

Different Ways for Reducing Latency Here we provide a comparison of different methods for reducing latency, contrasting them with the approach discussed in Sec. 3.3. Specifically, we reduce the baseline latency to similar latency by directly removing blocks, cutting down FFN expansion width,

and reducing both attention head dimension and FFN expansion dimension simultaneously. From the results in Table 10, we observe that the removal of a single block in the final stage can lead to a severe drop in accuracy (-0.7%), indicating that greater depth enhances the model’s capacity. Concurrently reducing all FFN expansion widths causes a non-trivial performance degradation (-0.6%).

In contrast, we observe that an orchestrated reduction in both attention head and FFN expansion dimensions yields a milder accuracy decline (-0.2%). These results demonstrate that a comprehensive reduction across different components offers better flexibility and performance.

### Table 10: Different ways for reducing latency.

Reducing Setting Params (M) GMACs Latency (ms) Top-1 Acc. (%)

Baseline 10.0 1.79 1.15 80.4 Number of Blocks 8.4 1.70 1.07 79.7

FFN Width 8.6 1.62 1.07 79.8 Attn. Head and FFN Width 8.9 1.64 1.10 80.2

Depthwise Convlution in FFN Recent works (Cai et al., 2023; Qin et al., 2024) attempt to insert a depthwise convolution (DW Conv) within the FFN to perform spatial mixing on the expanded features activations. We hypothesize that implementing more effective spatial mixing before the FFN diminishes its significance. In our iFormer, depthwise convolution with a kernel size of 7 is employed for spatial modeling in the early layers, while a powerful SHMA is utilized in the later layers. This approach provides a significantly enhanced spatial mixing capacity than previous methods.

As shown in Table 11, enhancing all FFN with depthwise convolution, including those within the convolutional blocks, results in a +14% increase in FLOPs and an additional latency cost of 0.33 ms. This increase is expected since the intermediate layers in the FFN possess an expanded feature dimension. However, the Top-1 accuracy only exhibits a marginal improvement of +0.1%.

Table 11: Comparison of FFN with and without depthwise convolution.

DW Conv in FFN Params (M) GMACs Latency (ms) Top-1 Acc. (%)

with 9.6 1.83 1.43 80.5 w/o. 8.9 1.60 1.10 80.4

Training for Longer Schedule Another commonly used advanced training is an extended schedule (450 vs. 300). Here we provide additional experiments for both image classification and downstream tasks where we train iFormer with distillation for 450 epochs. To ensure a fair comparison with previous methods, we develop a larger model dubbed as iFormer-L2. We report the image

### Table 12: Training with distillation for 450 epochs on ImageNet-1K.

Model Params (M) Latency (ms) Reso. Epochs Top-1 (%) ConvNeXt-B (2022) 89.0 7.54 224 300 83.8

EfficientFormerV2-L (2023) 26.1 2.40 224 450 83.5 iFormer-L2 24.5 2.30 224 450 83.9

classification results on the ImageNet-1k dataset in Table 12. It shows that training iFormer-L2 for 450 epochs yields improved performance, obtaining a Top-1 accuracy of 83.9%, even surpassing the ConvNeXt-Base model.

- Table 13: Object detection & Semantic segmentation results using backbone pretrained for 450 epochs.

Param (M)

Latency ↓ (ms)

Object Detection Instance Segmentation Semantic APbox APbox50 APbox75 APmask APmask50 APmask75 mIoU

Backbone

Pretrain Epochs

ResNet50 (2016) 25.5 7.20 300 38.0 58.6 41.4 34.4 55.1 36.7 36.7 PoolFormer-S24 (2022) 21.4 12.30 300 40.1 62.2 43.4 37.0 59.1 39.6 40.3

ConvNeXt-T (Liu et al., 2022) 29.0 12.6 300 41.0 62.1 45.3 37.7 59.3 40.4 41.4 EfficientFormer-L3 (2022b) 31.3 8.40 300 41.4 63.9 44.7 38.1 61.0 40.4 43.5 RepViT-M1.5 (2024) 14.0 5.00 300 41.6 63.2 45.3 38.6 60.5 41.5 43.6

PVTv2-B1 (2022) 14.0 27.00 300 41.8 64.3 45.9 38.8 61.2 41.6 42.5 FastViT-SA24 (2023a) 20.6 8.97 300 42.0 63.5 45.8 38.0 60.5 40.5 41.0 EfficientMod-S (2024) 32.6 24.30 300 42.1 63.6 45.9 38.5 60.8 41.2 43.5

Swin-T (2021a) 28.3 Failed 300 42.2 64.4 46.2 39.1 61.6 42.0 41.5

iFormer-L 14.7 6.60 300 42.2 64.2 46.0 39.1 61.4 41.9 44.5 EfficientFormerV2-L (2023) 26.1 12.5 450 44.7 66.3 48.8 40.4 63.5 43.2 45.2

iFormer-L2 24.5 9.06 450 44.6 66.7 49.1 41.1 64.0 44.1 46.2

|𝐶×𝐻×𝑊<br><br>ChannelSplit<br><br>Project<br><br>ChannelConcat<br><br>𝑄<br><br>𝐾<br><br>𝑉<br><br>𝑟𝐶×𝐻×𝑊<br><br>Self-Attention<br><br>Project<br><br>Project<br><br>Project 𝑋<br><br>𝑋<br><br>𝑋<br><br>𝑟𝐶×𝐻×𝑊<br><br>(1 − 𝑟)𝐶×𝐻×𝑊<br><br>𝑟𝐶×𝐻×𝑊<br><br>𝐶×𝐻×𝑊<br><br>SHA in SHViT|SHMA (ours)<br><br>Activation<br><br>Project<br><br>Project Activation<br><br>𝑄<br><br>𝐾<br><br>𝑉<br><br>C 𝑅<br><br>×𝐻×𝑊<br><br>𝐶×𝐻×𝑊<br><br>𝐶×𝐻×𝑊<br><br>Self-Attention<br><br>Project<br><br>Project<br><br>Project<br><br>𝑋<br><br>𝐶×𝐻×𝑊|
|---|---|

𝑟𝐶×𝐻×𝑊

𝐶×𝐻×𝑊

Self-Attention

ChannelConcat

Self-Attention

Activation

𝐶×𝐻×𝑊

ChannelSplit

Project

Project

Figure 5: Comparison of SHMA and SHA in SHViT. In SHViT, rC channels are utilized for spatial attention, where r is set to 4.167. SHMA projects the input into a higher dimension of 12C (i.e., R=2) and avoids split and concatenation operations.

Furthermore, we integrate iFormer-L2 into the Mask-RCNN and Semantic FPN framework for downstream tasks. As anticipated, the model with the more powerful iFormer-L2 backbone achieves SOTA performance, obtaining a significant enhancement over models pretrained for 300 epochs. It also outperforms its EfficientFormerV2-L counterpart by +0.7% in APmask and +1.0% in mIoU, while being 1.4× faster. These experiments collectively show that advanced training strategies can be easily employed to improve the performance of iFormers.

- D RELATION TO SHVIT

We clarify the difference between SHA in iFormer and its counterpart in SHViT (Yun & Ro, 2024) from the following two aspects: First, in terms of motivation, iFormer explores efficient attention mechanisms specifically tailored for the on-device environment, whereas SHViT is geared towards general-purpose GPUs, which may exhibit different hardware characteristics. Second, in terms of methodology, as shown in Fig. 5, we utilize single-head attention with more channels (R is set to 2.), while SHViT employs fewer than 1/4 of channels for attention. The reduced number of channels can result in a lower rank of the attention matrix, potentially degrading its expressiveness. Additionally, the split and concatenate operations in SHViT introduce extra runtime.

- Table 14: Process of converting SHA in iFormer towards SHViT. Intermediate models are only measured by latency.

Modification Params(M) GMACs Latency (ms) Top-1(%) SHA Baseline without Modulation 9.9M 1758M 1.12ms 79.4

+ split 9.9M 1758M 1.18ms + attention on 1/4 channels 8.3M 1547M 1.02ms + concat 8.7M 1579M 1.11ms 79.5

We also conduct a more fair comparison with SHViT. We start from the SHA baseline referenced in Table 1, specifically denoted as ’SHA’ in Figure 2. The transition to SHViT involves the following steps: 1) splitting the input into two smaller tensors, X1 and X2, along the channel dimension; 2) applying single-head attention to the tensor X1, which contains fewer than 1/4 of channels present in the original input tensor; and 3) concatenating the attention output with the residual input X2. As summarized in Table 14, split and concatenate operations introduce additional runtime. Furthermore, the performance of the SHA in the SHViT exhibits a decline compared to its counterpart in iFormer under similar latency conditions (79.8 v.s. 79.5). This degraded performance may be attributed to the reduced number of channels in the attention mechanism.

- E ARCHITECTURE DETAILS In Table 15, we show the different architecture configurations of the iFormer model variants.
- F IFORMER FOR HIGHER RESOLUTION

Self-attention exhibits quadratic complexity with respect to the number of tokens, i.e., the resolution of the input image. This issue is exacerbated in dense prediction tasks, which usually require high-

- Table 15: iFormer architecture configurations. BN stands for Batch Normalization. SHMA stands for Singe-Head Modulation Attention. DW stands for Depthwise convolution. s and d means the stride and output dimension in convolution. hd denotes the head dimension in SHMA and the number of attention heads in all variants is 1. r means the expansion ratio in FFN.

Output Size (Downs. Rate)

iFormer-T iFormer-S iFormer-M iFormer-L

Stem

56×56 (4×)

Conv-BN-GELU 5×5 s2 d16 × 1 Conv-BN-GELU 5×5 s2 d16 × 1 Conv-BN-GELU 5×5 s2 d24 × 1 Conv-BN-GELU 5×5 s2 d24 × 1 Conv-BN-GELU 5×5 s2 d64

Conv-BN 1×1 s1 d32 × 1

Conv-BN-GELU 5×5 s2 d64 Conv-BN 1×1 s1 d32 × 1

Conv-BN-GELU 5×5 s2 d96 Conv-BN 1×1 s1 d48 × 1

Conv-BN-GELU 5×5 s2 d96 Conv-BN 1×1 s1 d48 × 1

- Stage 1 56×56

(4×)

 

Conv-BN 7×7 s1 d32 Conv-BN-GELU 1×1 s1 d96 Conv-BN 1x1 s1 d32

  × 2

 

Conv-BN 7×7 s1 d32 Conv-BN-GELU 1×1 s1 d128 Conv-BN 1x1 s1 d32

  × 2

 

Conv-BN 7×7 s1d48 Conv-BN-GELU 1×1 s1 d192 Conv-BN 1x1 s1 d48

  × 2

 

Conv-BN 7×7 s1 d48 Conv-BN-GELU 1×1 s1 d192 Conv-BN 1x1 s1 d48

  × 2

- Stage 2 28×28

(8×)

Conv-BN 3×3 s2 d64 × 1 Conv-BN 3×3 s2 d64 × 1 Conv-BN 3×3 s2 d96 × 1 Conv-BN 3×3 s2 d96 × 1

 

Conv-BN 7×7 s1 d64 Conv-BN-GELU 1×1 s1 d192 Conv-BN 1x1 s1 d64

  × 2

 

Conv-BN 7×7 s1 d64 Conv-BN-GELU 1×1 s1 d256 Conv-BN 1x1 s1 d64

  × 2

 

Conv-BN 7×7 s1 d96 Conv-BN-GELU 1×1 s1 d384 Conv-BN 1x1 s1 d96

  × 2

 

Conv-BN 7×7 s1 d96 Conv-BN-GELU 1×1 s1 d384 Conv-BN 1x1 s1 d96

  × 2

- Stage 3

14×14 (16×)

Conv-BN 3×3 s2 d128 × 1 Conv-BN 3×3 s2 d176 × 1 Conv-BN 3×3 s2 d192 × 1 Conv-BN 3×3 s2 d256 × 1

 

Conv-BN 7×7 s1 d128 Conv-BN-GELU 1×1 s1 d384 Conv-BN 1×1 s1 d128

  × 6

 

Conv-BN 7×7 s1 d176 Conv-BN-GELU 1×1 s1 d704 Conv-BN 1x1 s1 d176

  × 9

 

Conv-BN 7×7 s1 d192 Conv-BN-GELU 1×1 s1 d768 Conv-BN 1x1 s1 d192

  × 9

 

Conv-BN 7×7 s1 d256 Conv-BN-GELU 1×1 s1 d1024 Conv-BN 1x1 s1 d256

  × 8

 

CPE 3×3 SHMA hd64 FFN r2

  × 3

 

CPE 3×3 SHMA hd88 FFN r3

  × 3

 

CPE 3×3 SHMA hd96 FFN r3

  × 4

 

CPE 3×3 SHMA hd128 FFN r3

  × 8

 

Conv-BN 7×7 s1 d128 Conv-BN-GELU 1×1 s1 d384 Conv-BN 1x1 s1 d128

  × 1

 

Conv-BN 7×7 s1 d176 Conv-BN-GELU 1×1 s1 d704 Conv-BN 1×1 s1 d176

  × 1

 

Conv-BN 7×7 s1 d192 Conv-BN-GELU 1×1 s1 d768 Conv-BN 1×1 s1 d192

  × 1

 

Conv-BN 7×7 s1 d256 Conv-BN-GELU 1×1 s1 d1024 Conv-BN 1×1 s1 d256

  × 1

- Stage 4

7×7 (32×)

Conv-BN 3×3 s2 d256 × 1 Conv-BN 3×3 s2 d320 × 1 Conv-BN 3×3 s2 d384 × 1 Conv-BN 3×3 s2 d384 × 1

 

CPE 3×3 SHMA hd64 FFN r2

  × 2

 

CPE 3×3 SHMA hd80 FFN r3

  × 2

 

CPE 3×3 SHMA hd96 FFN r3

  × 2

 

CPE 3×3 SHMA hd96 FFN r3

  × 2

Params (M) 2.9 6.5 8.9 14.7 GMacs 0.53 1.09 1.64 2.63

- Table 16: Comparison of different attention designs in iFormer-M. For the sake of simplicity, we exclude other blocks that are not related to attention. ws is the window size for window attention.

Attention SHMA Hybrid SHMA Chunk Hybrid SHMA

- Stage 3

14×14 (16×)

  

CPE 3×3 Window Partitioning, ws16 Window SHMA hd96, ws16 FFN r3

   × 1

  

CPE 3×3 Chunk Window Partitioning, ws16 Window SHMA hd96, ws16 FFN r3

   × 1

 

CPE 3×3 SHMA hd96 FFN r3

  × 4

 

CPE 3×3 Window SHMA hd96, ws16 FFN r3

  × 2

 

CPE 3×3 Window SHMA hd96, ws16 FFN r3

  × 2

  

CPE 3×3 Window Reversing, ws16 SHMA hd96 FFN r3

   × 1

  

CPE 3×3 Chunk Window Reversing, ws16 SHMA hd96 FFN r3

   × 1

- Stage 4

  

  

   × 1

   × 1

CPE 3×3 Window Partitioning, ws16 Window SHMA hd96 FFN r3

CPE 3×3 Chunk Window Partitioning, ws16 Window SHMA hd96 FFN r3

7×7 (32×)

  

   × 1

  

   × 1

  × 2

 

CPE 3×3 Window Reversing, ws16 SHMA hd64 FFN r3

CPE 3×3 Chunk Window Reversing, ws16 SHMA hd64 FFN r3

CPE 3×3 SHMA hd64 FFN r2

resolution input such as 512×512 in semantic segmentation and generate a large amount of 1024 image tokens even in the third stage. Consequently, this will cause huge memory and computation costs in mobile devices.

To mitigate these issues, we resort to window attention as proposed in Swin (Liu et al., 2021a). However, default window attention only performs local self-attention within windows, thus lacking interactions between tokens from different windows which will impair modeling capacity. Swin introduces shifted window attention to alleviate this limitation. Unfortunately, the shifting operation inevitably incurs additional memory costs. In contrast to Swin, we implement

### Table 17: Latency comparison of different attention mechanisms.

Attention Resolution Latency (ms)

SHMA 224 1.10 SHMA 512 Failed

Hybrid SHMA 512 11.46 CC Hybrid SHMA 512 4.0

a hybrid attention design. Specifically, we compute window attention within windows, except for the last attention block in each stage. This approach enables iFormer to capture more global features essential for dense prediction tasks. At the same time, since window partitioning and reversing also incur memory access costs, we minimize the usage of them to once per stage. We replace the standard SHMA in iFormer with a hybrid window SHMA, as shown in Table 16.

From the latency comparison in Table 17, we see that simply applying SHMA will encounter a memory bottleneck on mobile devices. Instead, our hybrid SHMA can significantly reduce memory access costs, achieving a mobile latency of 11.46 ms.

However, hybrid SHMA still lags much behind the recent FastViT-SA12, which has a latency of

- 5.27 ms. We identify the speed bottleneck as stemming from the window partitioning and reversing operations, even though we only implement them once in each stage. As the feature map size increases, the reshaping involved in these operations demands considerable memory, thereby slowing inference in resource-constrained mobile devices. To address this issue, we propose a method called “Channel Chunking” (CC). Formally, given a

- 2D input feature map x ∈ RC×H×W, the standard window partitioning divides the feature map into HP × WP non-overlapped regions, each corresponding to a window that contains P × P feature

vectors. This step is accomplished by reshaping x as xP ∈ R

HW

P2 ×C×P×P. Then we apply SHMA

within each window. To reduce the memory requirements associated with reshaping, we propose to split the feature map x along the channel dimension into a series of smaller chunks as follows:

xS1,...,xSn = Chunking(x), (4)

where K is the chunk size, n = KC is the number of chunks. We set n=16 for the input image of 512×512 in our object detection and semantic segmentation experiments. Then we apply win-

dow partitioning sequentially to these smaller chunks and concatenate them. This process can be mathematically expressed as follows:

xP = Concat(xPi ,...,xPn), where xPi = WindowPartitioning(xSi ),

(5)

These smaller chunks can be processed rapidly. As shown in Table 17, the chunking strategy allows the model to achieve 2.9× speed up in inference speed. Correspondingly, the window reversing operation is performed by reshaping multiple windows xP ∈ R

HW

P2 ×C×P×P into a 2D feature map x ∈ RC×H×W. These results demonstrate that our proposed Channel Chunking Hybrid SHMA significantly enhances the iFormer’s ability to process high-resolution images efficiently.

Computation Complexity Given an input x ∈ RC×H×W and a window size of P × P, as detailed in Section E, the computational complexity of iFormer is as follows:

Ω(SHMA) =4HWC2(QKV and output projection)+ HWC(element-wise product of modulation)+ 2P2HWC(self-attention),

(6)

Ω(FFN) = 8HWC2. (7) In image classification, we do not utilize window attention since the feature size is 14× 14 in stage

- 3 (it equals to the window attention when P=14). In downstream tasks, we adopt a window size of P=16.

- G COMPREHENSIVE COMPARISON

In Table 18, we provide a more comprehensive comparison between iFormer and other lightweight models on ImageNet-1k classification.

Table 18: Comprehensive comparison between iFormer and the previously proposed models on ImageNet-1K. Failed indicated that the model runs too long to report latency by the Core ML, often caused by excessive memory access.

Latency ↓ (ms)

Reso. Epochs Top-1 (%) MobileNetV2 1.0x (2018) 3.4 0.30 0.73 224 500 72.0

Model Params (M) GMACs

SHViT-S1 (2024) 6.3 0.24 0.74 224 300 72.8

- MobileNetV3-Large 0.75x (2019) 4.0 0.16 0.67 224 600 73.3 MNV4-Conv-S (2024) 3.8 0.20 0.60 224 500 73.8

iFormer-T 2.9 0.53 0.60 224 300 74.1

ShuffleNetV2 1.0× (2018) 2.3 0.15 0.74 224 300 69.4 MobileNetV2 1.4x (2018) 6.9 0.59 1.02 224 500 74.7

- MobileNetV3-Large 1.0x (2019) 5.4 0.22 0.76 224 600 75.2 SwiftFormer-XS (2023) 3.5 0.60 0.95 224 300 75.7 SBCFormer-XS (2024) 5.6 0.70 0.79 224 300 75.8

GhostNetV3 1.0x† (2024) 6.1 0.17 0.99 224 600 77.1 EfficientNet-B0 (2019) 5.3 0.39 0.89 224 350 77.1 MobileOne-S2 (2023b) 7.8 1.30 0.92 224 300 77.4 LowFormer-B0 (2024) 14.1 0.94 1.45 224 300 78.4

CAS-ViT-XS (2024) 3.2 0.56 0.85 224 300 77.5 EMO-5M (2023) 5.1 0.90 Failed 224 300 78.4 RepViT-M1.0 (2024) 6.8 1.10 0.85 224 300 78.6

iFormer-S 6.5 1.09 0.85 224 300 78.8 ShuffleNetV2 1.5× (2018) 3.5 0.30 1.16 224 300 72.6

EdgeViT-XXS (2022) 4.1 0.60 1.41 224 300 74.4 SHViT-S2 (2024) 11.4 0.37 1.10 224 300 75.2 EfficientMod-xxs (2024) 4.7 0.60 1.29 224 300 76.0 SBCFormer-S (2024) 8.5 0.90 1.02 224 300 77.7 MobileOne-S3 (2023b) 10.1 1.90 1.16 224 300 78.1 SwiftFormer-S (2023) 6.1 1.00 1.12 224 300 78.5 GhostNetV3 1.3x† (2024) 8.9 0.27 1.24 224 600 79.1

EfficientNet-B1 (2019) 7.8 0.70 1.29 240 350 79.1 FastViT-T12 (2023a) 6.8 1.40 1.12 256 300 79.1 RepViT-M1.1 (2024) 8.2 1.30 1.04 224 300 79.4 RepNeXt-M3 (2024) 7.8 1.30 1.04 224 300 79.4 FastViT-S12 (2023a) 8.8 1.80 1.26 256 300 79.8

MNV4-Conv-M (2024) 9.2 1.00 1.08 256 500 79.9 iFormer-M 8.9 1.64 1.10 224 300 80.4

MobileViT-XXS (2021) 1.3 0.40 2.12 256 300 69.0 MobileViTV2-0.5 (2022) 1.4 0.50 9.47 256 300 70.2 ShuffleNet v2 2.0× (2018) 7.4 0.59 1.94 224 300 74.9

EdgeViT-XS (2022) 6.7 1.10 1.79 224 300 77.5 Mobile-Former-294M (2022b) 11.4 0.29 2.66 224 450 77.9

MobileViTV2-1.0 (2022) 4.9 1.80 Failed 256 300 78.1 EfficientMod-xs (2024) 6.6 0.80 2.13 224 300 78.3 MobileViT-S (2021) 5.6 2.00 3.55 256 300 78.4 CMT-Ti (2022) 11.3 687 Failed 160 300 79.2 Mobile-Former-508M (2022b) 14 0.51 3.33 224 450 79.3 SHViT-S4 (2024) 16.5 0.99 1.48 224 300 79.4

EfficientViT-B1-r224 (2023) 9.1 0.52 2.38 224 350 79.4 MobileOne-S4 (2023b) 14.8 2.98 1.74 224 300 79.4 LowFormer-B1 (2024) 17.9 1.41 1.90 224 300 79.9

SBCFormer-B (2024) 13.8 1.60 1.44 224 300 80.0 EfficientNet-B2 (2019) 9.2 1.00 1.69 260 350 80.1 CAS-ViT-S (2024) 5.8 0.93 1.82 224 300 80.2

GhostNetV3 1.6x† (2024) 12.3 0.40 1.49 224 600 80.4 EfficientViT-B1-r288 (2023) 9.1 0.86 3.87 288 450 80.4

FastViT-SA12 (2023a) 10.9 1.90 1.50 256 300 80.6 MNV4-Hybrid-M (2024) 10.5 1.20 1.75 256 500 80.7

SwiftFormer-L1 (2023) 12.1 1.60 1.60 224 300 80.9 EfficientMod-s (2024) 12.9 1.40 2.57 224 300 81.0 SBCFormer-L (2024) 18.5 2.70 1.89 224 300 81.1 RepViT-M1.5 (2024) 14.0 2.30 1.64 224 300 81.2

LowFormer-B1.5 (2024) 33.9 2.57 3.02 224 300 81.2 RepNeXt-M4 (2024) 13.3 2.30 1.47 224 300 81.2 CAS-ViT-M (2024) 12.4 1.89 2.46 224 300 81.4 iFormer-L 14.7 2.63 1.60 224 300 81.7

