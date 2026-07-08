# arXiv:2403.09977v1[cs.CV]15Mar2024

## EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba

##### Xiaohuan Pei⋆, Tao Huang∗, and Chang Xu

School of Computer Science, Faculty of Engineering, The University of Sydney {xpei8318,thua7590}@uni.sydney.edu.au, c.xu@sydney.edu.au

Abstract. Prior efforts in light-weight model development mainly centered on CNN and Transformer-based designs yet faced persistent challenges. CNNs adept at local feature extraction compromise resolution while Transformers offer global reach but escalate computational demands O(N2). This ongoing trade-off between accuracy and efficiency remains a significant hurdle. Recently, state space models (SSMs), such as Mamba, have shown outstanding performance and competitiveness in various tasks such as language modeling and computer vision, while reducing the time complexity of global information extraction to O(N). Inspired by this, this work proposes to explore the potential of visual state space models in light-weight model design and introduce a novel efficient model variant dubbed EfficientVMamba. Concretely, our EfficientVMamba integrates a atrous-based selective scan approach by efficient skip sampling, constituting building blocks designed to harness both global and local representational features. Additionally, we investigate the integration between SSM blocks and convolutions, and introduce an efficient visual state space block combined with an additional convolution branch, which further elevate the model performance. Experimental results show that, EfficientVMamba scales down the computational complexity while yields competitive results across a variety of vision tasks. For example, our EfficientVMamba-S with 1.3G FLOPs improves VimTi with 1.5G FLOPs by a large margin of 5.6% accuracy on ImageNet. Code is available at: https://github.com/TerryPei/EfficientVMamba.

Keywords: Light-weight Architecture · Efficient Network · State Space Model

### 1 Introduction

Convolutional networks, exemplified by models such as ResNet [14], Inception [38, 39], and EfficientNet [41], and Transformer-based networks, such as SwinTransformer [25], Beit [1], and Resformer [59] have been extensively applied to visual tasks including image classification, detection, and segmentation, achieving remarkable results. Recently, Mamba [7], a network based on state-space models (SSMs) [9–11,20,31], has demonstrated competitive performance to Transformers [47] in sequence modeling tasks such as language modeling. Inspired by this,

⋆ Equal contributions.

some works [22, 24, 33, 55, 61] are pioneering in introducing SSMs into vision tasks. Among these methods, Vmamba [24] stands out by introducing an SS2D method to preserve 2D spatial dependencies by scanning images from multiple directions.

However, the impressive performance achieved by these various architectures usually comes from the scaling up of model sizes, making a critical challenge in applying them on resource-constrained devices. In pursue of light-weight models, many studies have been conducted to reduce the resource consumption of vision models while keeping a competitive performance. Early works on efficient CNNs mainly focus on narrowing the original convolutional block with efficient group convolutions [4, 15,34], light skipping connections [12, 27], e.t.c. While recently, due to the remarkable success of including the global representation ability of Transformers into vision tasks, some works are proposed to reduce the computation complexity of ViTs [17,25,46,49] and fuse ViTs with CNNs in light-weight models [19,28,46]. However, the lightening of ViTs are usually obtained with the lost of global capture capability in self-attention. Due to the O(N) time complexity of global self-attention, its computation and memory costs increase dramastically at large resolutions. As a result, existing efficient ViT methods have to perform local self-attentions within partitioned windows [17,25,46], or only conduct global self-attentions in deeper stages with low resolutions [19, 28]. The embarrassing trade-off and rollback of ViTs to CNNs hinders the ability of improving the light-weight models further.

[Figure 1]

Fig. 1: Lightweight Model Performance Comparison on ImageNet. EfficientVMamba outperforms previous work across various model variants in terms of both accuracy and computational complexity.

In this paper, recalling the previously metioned linear scaling complexity in SSMs, we are inspired to obtain efficient global capture ability in light-weight vision models by involving SSMs into model design. Its outstanding performance is demonstrated in Figure 1. We achieve this by first introducing a skip-sampling mechanism, which reduces the number of tokens that need to be scanned in the spatial dimension, and saves multiple times of computation cost in sequence modeling of SSMs while keeping the global receptive field among tokens, as illustrated in Figure 2. On the other hand, acknowledging that the convolutions provide a more efficient way for feature extraction in the case when only local representations suffice, we introduce a convolution branch in supplement of the original global SSM branch, and perform feature fusion of them through the channel attention module, SE [16]. Finally, for an optimal allocation of capabilities of various block types, we construct our network with global SSM blocks in

[Figure 2]

- Fig. 2: Illustration of efficient 2D scan methods (ES2D). (a.) Vmamba [24] employs SS2D method in vision tasks, traversing entire row or column axes, which incurs heavy computational resources. (b.) We present an efficient 2D scanning method, ES2D, which organizes patches by omitting sampling steps, and then proceeds with an intra-group traversal (with a skipping step of 2 in the Figure). The proposed scan approach reduces computational demands (4N → N) while preserving global feature maps (e.g.Each group contains eye-related patches.)

the shallow and high-resolution layers, while adopting efficient convolution blocks (MobileNetV2 blocks [34]) in the deeper layers. The final network, achieving efficient SSM computation and efficient integration of convolutions, has showcased significant improvements compared to previous CNN and ViT based light-weight models through our experiments on image classification, object detection, and semantic segmentation tasks.

In summary, the contributions of this paper are as follows.

- 1. We propose an atrous-based selective scanning strategy, which is realized through a novel skip sampling and regrouping patched in the spatial respective field. The strategy refines the building blocks to efficiently extract global dependencies while reducing computation complexity (O(N) → O(N/p2)) with step p).
- 2. We introduce a dual-pathway module that combines our efficient scanning strategy for global feature capture and a convolution branch for efficient local feature extraction, along with a channel attention module to balance the integration of both global and local features. Besides, we propose a better allocation of SSM and CNN blocks by promoting SSMs in early stages with high resolutions for better global capture, while adopting CNNs in low resolutions for better efficiency.
- 3. We conduct extensive experiments on image classification, object detection, and semantic segmentation tasks. The results and illustration shown in Figure 1 demonstrate that, our EfficientVMamba effectively reduces the FLOPs of the models while achieving significant performance improvements compared to existing light-weight models.

### 2 Related Work

#### 2.1 Light-weight Vision Models

In recent years, the realm of vision tasks has been predominantly governed by Convolutional Neural Networks (CNNs) and Visual Transformer (ViT) architectures. The focus on making these architectures lightweight to enhance efficiency has emerged as a pragmatic and promising direction in research. For CNNs, notable advancements have been made in improving image classification accuracy, as evidenced by the development of influential architectures like ResNet [14], RegNet [35], and DenseNet [18]. These advancements have set new benchmarks in accuracy but also introduced a need for lightweight architectures [51,52]. This need has been addressed through various factorization-based methods, making CNNs more mobile-friendly. For instance, separable convolutions introduced by Xception have been instrumental in this regard, leading to the development of state-of-the-art lightweight CNNs, such as MobileNets [15], ShuffleNetv2 [27], ESPNetv2 [29], MixConv [42], MNASNet [40], and GhostNets [13]. These models are not only versatile but also relatively simpler to train. Following CNNs, Transformers have gained significant traction in various vision tasks, such as image classification, object detection, and autonomous driving, rapidly becoming the mainstream approach. The lightweight versions of Transformers have been achieved through diverse methods. On the training front, sophisticated data augmentation strategies and techniques like Mixup [60], CutMix [58], and RandAugment [6] have been employed, as seen in models like CaiT [45] and DeiTIII [44], which demonstrate exceptional performance without the need for large proprietary datasets. From the architectural design perspective, efforts have been concentrated on optimizing self-attention input resolution and devising attention mechanisms that incur lower computational costs. Innovations like PVT-v1 [49]’s emulation of CNN’s feature map pyramid, Swin-T [25] and LightViT [17]’s hierarchical feature map and shifted-window mechanisms, and the introduction of (multi-scale) deformable attention modules in Deformable DETR [62] exemplify these advancements. There is also NAS for ViTs [37].

#### 2.2 State Space Models

The State Space Model (SSM) [9–11,20,31] is a family architecture encapsulates a sequence-to-sequence transformation has the potential to handle tokens with long dependencies, but it is challenging to train due to its high computational and memory usage. Nevertheless, recent works [7–9, 11, 36] have enabled deep State Space Models to become progressively more competitive with CNN and Transformer. In particular, S4 [9] employs a Normal Plus Low-Rank (NPLR) representation to efficiently compute the convolution kernel by leveraging the Woodbury identity for matrix inversion. And then Mamba [7] enhances SSMs with input-specific parameterization and a scalable, hardware-optimized algorithm, achieving simpler design and superior efficiency in processing long sequences for language and genomics. Following success of SSM, there has been a surge in

applying the framework to computer vision tasks. S4ND [30] first introduce the SSM blocks into vision tasks, facilitating the modeling of visual data across 1D, 2D, and 3D as continuous signals. Vmamba [24] pioneers a mamba-based vision backbone a cross-scan module to address the direction-sensitivity issue arising from the differences between 1D sequences and multi-channels images. Similarly, Vim [61] introduces an efficient state space model for vision tasks by leveraging bidirectional state space modeling for data-dependent global visual context without image-specific biases. The impressive performance of the Mamba backbone in various vision tasks has inspired a wave of research [2,22,33,33,48] focusing on adapting Mamba-based models for specialized vision applications. Recent works like Vm-unet [33], U-Mamba [22], and SegMamba [55] have adapted Mambabased backbones for medical image segmentation, integrating unique features such as a U-shaped architecture in Vm-unet, an encoder-decoder framework in U-Mamba, and whole volume feature modeling in SegMamba. In the domain of graph representation, GraphMamba [48] integrates Graph Guided Message Passing (GMB) with Message Passing Neural Networks (MPNN) within the Graph GPS architecture, which enhances the training and contextual filtration for graph embeddings. Furthermore, GMNs [2] present a comprehensive framework that encompasses tokenization, optional positional or structural encoding, localized encoding, sequencing of tokens, and utilizes a series of bidirectional Mamba layers for processing graphs.

### 3 Preliminaries

#### 3.1 State Space Models (S4)

State Space Models (SSMs) are a general family of sequence model used in deep learning that are influenced by systems capable of mapping one-dimensional sequences in a continuous manner. These models transform input D-dimensional sequence x(t) ∈ RL×D into output sequence y(t) ∈ RL×D by utilizing a learnable latent state h(t) ∈ RN×D that is not directly observable. The mapping process could be denoted as:

h′(t) = Ah(t) + Bx(t), y(t) = Ch(t),

(1)

where A ∈ RN×N, B ∈ RD×N and C ∈ RD×N.

Discretization. Discretization aims to convert the continuous differential equations into discrete functions, aligning the model to the input signal’s sampling frequency for more efficient computation [10]. Following the work [11], the continuous parameters (A, B) can be discretized by zero-order hold rule with a given sample timescale ∆ ∈ RD:

- A¯ = e∆A,
- B¯ = (e∆A − I)A−1B,
- C¯ = C, B¯ ≈ (∆A)(∆A)−1AB = ∆B, h(t) = A¯h(t − 1) + B¯x(t), y(t) = C¯h(t),

(2)

where A¯ ∈ RN×N, B¯ ∈ RD×N and C¯ ∈ RD×N.

To simplify calculations, the repeated application of Equation 2 can be efficiently performed simultaneously using a global convolution approach.

y = x ⊛ K

with K = (CB,CAB,...,CAL−1B),

(3)

where ⊛ denotes convolution operation, and K ∈ RL is the SSM kernel.

#### 3.2 Selective State Space Models (S6)

Mamba [7] improves the performance of SSM by introducing Selective State Space Models (S6), allowing the continuous parameters to vary with the input enhances selective information processing across sequences, which extend the discretization process by selection mechanism:

- B¯ = sB(x),
- C¯ = sC(x), ∆ = τA(Parameter + sA(x)),

(4)

where sB(x) and sC(x) are linear functions that project input x into an Ndimensional space, while sA(x) broadens a D-dimensional linear projection to the necessary dimensions. In terms of visual tasks, VMamba proposed the 2D Selective Scan (SS2D) [24], which maintains the integrity of 2D image structures by scanning four directed feature sequences. Each sequence is processed independently within an S6 block and then being combined to form a comprehensive 2D feature map.

### 4 Method

In order to design light-weight models that are friendly to resource-limited devices, we propose EfficientVMamaba, which is summarized in Figure 3. We introduce an efficient selective scan approach to reduce the computational complexity

- in Section 4.1, and build a block considering both global and local feature extraction with the integration of SSMs and CNNs in Section 4.2. Regarding the design of architecture, Section 4.4 then offers an in-depth look at various architectural variations tailored to different model sizes.

#### 4.1 Efficient 2D Scanning (ES2D)

In deep neural networks, downsampling via pooling or strided convolution is employed to broaden the receptive field with a lower computational cost; however, this comes at the expense of spatial resolution. Previous work [46, 57] demonstrate apply atrous-based strategy benefits broadening the receptive field without sacrificing resolution. Inspired by this observation and aiming to alleviate and light the computational complexity of selective scanning, we propose an efficient 2D scanning (ES2D) method to scale down the visual selective scan block (SS2D) via skipping sampling for each patches on the feature map. Given a input feature map X ∈ RC×H×W, instead of cross-scan whole patches, we skip scan patches with a step size p and partition into selected spatial dimensional features {Oi}4i=1:

Oi ←−−−scan X[:,m :: p,n :: p],

{O˜i}4i=1 ← SS2D({Oi}4i=1), Y [:,m :: p,n :: p] ←−−−−merge O˜i,

(5)

- 1

- 2

- 1

- 2

π 2

- 1

- 2

- 1

- 2

π 2

with (m,n) = (

(i − 2) ,

(i − 2) ),

+

sin

+

cos

H p ×Wp and the operation [:,m :: p,n :: p] represents slicing the matrix for each channel, starting at m on height (H) and n on width (W), skipping every p steps. The process decompose the fully scanning method into both local and global sparse forms. Skip sampling for local receptive fields reduces computational complexity by selectively scanning smaller patches of the feature map. With a step size p, we sample the (C,H/p,W/p) patches at intervals of p, compared to (C,H,W) in the SS2D, decreasing the number of tokens processed from N to pN2 for each scan and merge operation, which improves feature extraction efficiency. Re-grouping for global spatial feature maps in ES2D involves combining the processed patches to reconstruct the global structure of the feature map. This integration captures broader contextual information, balancing local detail and global context in feature extraction. Accordingly, our design is intended to streamline the scanning and merging modules while maintaining the essential benefit of global integration in the state-space architecture, with the aim of ensuring that the feature extraction remains comprehensive on the spatial axis.

where Oi,O˜i ∈ RC×

#### 4.2 Efficient Visual State Space Block (EVSS)

Based on the efficient selected scan approach, we introduce the Efficient Visual State Space (EVSS) block, which is designed to synergistically merge global and local feature representations while maintaining computational efficiency. It leverages a SqueezeEdit-modified ES2D for global information capture and a convolutional branch tailored to extract critical local features, with both branches undergoing a subsequent Squeeze-Excitation (SE) block [16]. The ES2D module aims to efficiently abstract global contextual information by implementing

[Figure 3]

- Fig. 3: Architecture overview of EfficientVMamba. We hightlight our contributions with corresponding colors in the Figure. (1) ES2D 4.1: Atrous-based selective scanning strategy via skip sampling and regrouping in the spatial space. (2) EVSS 4.2: The EVSS block merges global and local feature extraction with modified ES2D and convolutional approaches enhanced by Squeeze-Excitation blocks for refined dual-pathway feature representation. Inverted Fusion 4.3: Inverted Fusion places local-representation modules in deep layers, deviating from traditional designs by utilizing EVSS blocks early for global representation and inverted residual blocks later for local feature extraction.

an intelligent skipping mechanism presented in 4.1. It selectively scans the map with a step size p, reducing redundancy without sacrificing the representational quality of the global context in the resultant spatial dimensional features. Parallel to this, empirical evidence concurs that convolutional operations offer a more proficient approach to feature extraction, particularly in scenarios where local representations are adequate. We add the convolutional branch concentrates on discerning fine-grained local details through a 3 × 3 convolution of stride 1. The subsequent SE block adaptively recalibrates the features, allowing the network to auto re-balanced the local and global respective field on the feature map.

The outputs of the respective SE blocks are combined via element-wise summation to construct the EVSS’s output and the dual pathway could be denoted as:

Xl+1 = SE(ES2D(Xl)) + SE(Conv(Xl)), (6)

where Xl represent the feature map of the l-layer and SE(·) is the SqueezeExcitation operation. With each pathway utilizing a SE block, the EVSS ensures that the respective features of global and local information are dynamically rebalanced to emphasize the most salient features. This fusion aims to preserve

the integrity of both the expansive global perspective and the intricate local specifics, facilitating a comprehensive feature representation.

#### 4.3 Inverted Insertion of EfficientNet Blocks

As a well-established consensus, the computational efficiency of convolutional operations is more efficient than that of the global-based block such as Transformer. Prior light-weight work efforts have predominantly employed computation-efficient convolutions in the former stages to scale down the token numbers to reduce computational complexity, subsequently integrating global-based blocks (e.g., Transformer with the computational complexity of O(N2)) to capture global context in the latter stages. For example, MobileViT [28] adopts pure MobileNetV2 blocks in the first two downsampling stages, while only integrating self-attention operations in the latter stages at low resolutions. EfficientFormer [19] introduces two types of base blocks, the convolution-based blocks with local pooling are used in the first three stages, and the transformer-like self-attention blocks are only leveraged in the last stage.

However, the observation is contrast on the Mamba-based block. In the SSM framework, the computational complexity for global representation is O(N), indicating that placing local representation modules at either the front or the back of the stage could be reasonable. Through empirical observation in Table 6, we found positioning these local-representation modules towards the latter layers of the stage yields better results. This discovery significantly deviates from the design principles of previous CNN-based and Transformer-based lightweight models, thereby we call it inverted insertion. Consequently, our designed L stages architecture is an inverted insertion of EfficientNet Blocks (MobileNetV2 blocks with SE modules), which utilizes EVSS blocks 4.2 in the former two stages to capture global-representation and Inverted Residual blocks InRes(·) [34] in the subsequent stages to extract local feature maps:

Xl+1 =

EVSS(Xl) if Xl ∈ {stage1,stage2}; InRes(Xl) otherwise,

(7)

where Xl is the feature map in the l-layer. The inverted insertion design of using the shortcuts directly between the bottlenecks is considerably more memory efficient [34].

#### 4.4 Model Variants

To sufficiently demonstrate the effectiveness of our proposed model, we detail architectural variants rooted in plain structures as referenced in [61]. These variants are designated as EfficientVMamba-T, EfficientVMamba-S, and EfficientVMambaB, shown as Table 1, corresponding to different scales of the model. EfficientVMambaT is the most lightweight with 6M parameters, followed by EfficientVMamba-S with 11M, and EfficientVMamba-B being the most complex with 33M. In terms

Table 1: Model variants of EfficientVMamba.

|Layer|Feature Size|EfficientVMamba-T<br><br>|EfficientVMamba-S<br><br>|EfficientVMamba-B|
|---|---|---|---|---|
|Stem|112 × 112<br><br>|#Dim=[48, 96, 192, 384]<br><br>|#Dim=[96, 192, 384, 768]|#Dim=[96, 192, 384, 768]|
|Stage 1<br><br>Stage 3<br><br>Stage 4<br><br><br>|56 × 56 14 × 14 7 × 7|EVSS × 2 InRes × 4 InRes × 2<br><br>|EVSS × 2 InRes × 4 InRes × 2<br><br>|EVSS × 2 InRes × 9 InRes × 2|
| |1 × 1|Average pool, Fc, Softmax<br><br>| | |
|Params. (M) FLOPs (G)<br><br>| |6 0.8|11 1.3<br><br>|33 4.0|

S

of computational load, measured in FLOPs, the models exhibit a parallel increase with 0.8G for EfficientVMamba-T, 1.3G for EfficientVMamba-S, and 4.0G for EfficientVMamba-B, correlating directly with their complexity and feature size.

### 5 Experiments

To rigorously evaluate the performance of our diverse model variants, we demonstrate the results of image classification task in Section 5.1, investigate object detection performance in Section 5.2 and explore the image semantic segmentation

- in Section 5.3. In section 5.4 We further pursued ablation study to comprehensively examine the effects of atrous selective scanning , the impact of SSM-Conv fusion blocks, and the implications of incorporating convolution blocks at different stages of the models.

#### 5.1 ImageNet Classification

Training strategies. Following previous works [24, 25, 43, 61], we train our models for 300 epochs with a base batch size of 1024 and an AdamW optimizer, a cosine annealing learning rate schedule is adopted with initial value 10−3 and 20-epoch warmup. For training data augmentation, we use random cropping, AutoAugment [5] with policy rand-m9-mstd0.5, and random erasing of pixels with a probability of 0.25 on each image, then a MixUp [60] strategy with ratio

- 0.2 is adopted in each batch. An exponential moving average on model is adopted with decay rate 0.9999.

Tiny Models (FLOPs(G) ∈ [0,1]). In the pursuit of efficiency, the results of tiny models are shown in Table 2. EfficientVMamba-T achieves state-of-art performance with a Top-1 accuracy of 76.5%, rivalling its counterparts that demand higher computational costs. With a modest expenditure of only 0.8 GFLOPs, our model surpasses the PVTv2-B0 by a 6% margin in accuracy and outperforms the MobileViT-XS by 1.7%, all with less computational demand.

Small Models (FLOPs(G) ∈ [1,2]). Our model, EfficientVMamba-S, exhibits a significant improvement in accuracy, achieving a Top-1 accuracy of 78.7%. This represents a substantial increase over DeiT-Ti and MobileViTS, which achieve 72.2% and 78.4% respectively. Notably, EfficientVMamba-S

Table 2: Comparison of different backbones on ImageNet-1K classification.

|Method<br><br>|Image size Params (M) FLOPs (G)<br><br>|Top-1 ACC (%)|
|---|---|---|
|RegNetY-800M [32] PVTv2-B0 [50] MobileViT-XS [28] LVT [56] EfficientVMamba-T<br><br>|2242 6 0.8<br><br>2242 3 0.6 2242 2 1.0<br><br>2242 6 0.9 2242 6 0.8<br><br><br>|76.3 70.5 74.8 74.8 76.5<br><br>|
|RegNetY-1.6G [32] DeiT-Ti [43] MobileViT-S [28] Vim-Ti [61] EfficientVMamba-S<br><br>|2242 11 1.6<br><br>2242 6 1.3<br><br>2242 6 2.0 2242 7 1.5 2242 11 1.3<br><br><br>|78.0 72.2 78.4 73.1 78.7<br><br>|
|RegNetY-4G [32] DeiT-S [43] Swin-T [25] Vim-S [61] VMamba-T [24] EfficientVMamba-B<br><br>|2242 21 4.0<br>2242 22 4.6 2242 29 4.5 2242 26 5.1 2242 22 5.6 2242 33 4.0<br><br><br>|80.0 79.8 81.3 80.3 82.2 81.8<br><br>|

maintains this high accuracy level with computational efficiency, requiring only

- 1.3 GFLOPs, which is on par with DeiT-Ti and lower than MobileViT-S’s 2.0 GFLOPs.

Base Models (FLOPs(G) ∈ [4,5]). EfficientVMamba-B achieves an impressive Top-1 accuracy of 81.8%, surpassing DeiT-S by 2% and Vim-S by 1.5%,

- as indicated in the third group of the Table 2. This base model demonstrates the feasibility of coupling a substantial parameter count of 33 M with a modest computational demand of 4.0 GFLOPs. In comparison, VMamba-T, with a similar parameter count of 22 M requires a higher 5.6 GFLOPs.

#### 5.2 Object Detection

Training strategies. We evaluate the efficacy of our EfficientVMamba model for object detection tasks on the MSCOCO 2017 [21] dataset. Our evaluation framework relies on the mmdetection library [3]. For comparisons with lightweight backbones, we follow PvT [49] to use RetinaNet as the detector and adopt 1× training schedule. While for comparisons with larger backbones, our experiment follows the hyperparameter settings detailed in Swin [25] We use the AdamW optimization method to refine the weights of our pre-trained networks on ImageNet-1K for durations of 12 and 36 epochs. We apply drop path rates of 0.2% across the board for EfficientVMamba-T/S/B variants. The learning rate begins at 1e−5 and is decreased tenfold at epochs 9 and 11. Multi-scale training and random flipping are implemented during training with a batch size of 16, adhering to standard procedures for evaluating object detection systems.

Table 3: COCO detection results on RetinaNet.

|Model<br><br>|Params (M)<br><br>|AP AP50 AP75 APS APM APL|
|---|---|---|
|ResNet18 [14] PVTv1-Tiny [49] EfficientViT-M4 [23] PVTv2-b0 [50] EfficientVMamba-T<br><br>|21.3 23.0 8.8 13.0 13.0<br><br>|31.8 49.6 33.6 16.3 34.3 43.2<br><br>36.7 56.9 38.9 22.6 38.8 50.0<br><br>32.7 52.2 34.1 17.6 35.3 46.0<br><br>37.2 57.2 39.5 23.1 40.4 49.7 37.5 57.8 39.6 22.6 40.7 49.1<br><br><br><br><br>|
|ResNet50 [14] PVTv1-Small [49] PVTv2-b1 [50] EfficientVMamba-S<br><br>|37.7 34.2 23.8 19.0<br><br>|36.3 55.3 38.6 19.3 40.0 48.8<br><br>40.4 61.3 43.0 25.0 42.9 55.7<br>41.2 61.9 43.9 25.4 44.5 54.3 39.1 60.3 41.2 23.9 43.0 51.3<br><br><br>|
|ResNet101 [14] ResNeXt101-32x4d [54] PVTv1-Medium [49] EfficientVMamba-B<br><br>|56.7 56.4 53.9 44.0<br><br>|38.5 57.8 41.2 21.4 42.6 51.1<br><br>39.9 59.6 42.7 22.3 44.2 52.5<br><br><br>41.9 63.1 44.3 25.0 44.9 57.6<br><br>42.8 63.9 45.8 27.3 46.9 55.1<br><br><br>|

Results. We summarize the results of RetinaNet detector in Table 3. Remarkably, each variants competitively reducing the sizes while simultaneously exhibits a performance enhancement. The EfficientVMamba-T model stands out with 13M parameters and an AP of 37.5%, slightly higher by 5.7% compared to the ResNet-18, which has 21.3M parameters. The performance of EfficientVMamba-T also surpasses PVTv1-Tiny by 0.8% while matching it in terms of parameter count. EfficientVMamba-S, with only 19M parameters, achieves a commendable AP of 39.1%, outstripping the larger ResNet50 model, which shows a lower AP of 36.3% despite having 37.7M parameters. In the higher echelons, EfficientVMamba-B, which boasts 44M parameters, secures an AP of 42.8%, signifying a significant lead over both ResNet101 and ResNeXt101-32x4d, highlighting the efficiency of our models even with a smaller parameter footprint. Notably, PVTv2-b0 with 13M parameters achieves an AP of 37.2%, which EfficientVMamba-T closely follows, indicating competitive performance with a similar parameter budget. For the comparisons with other backbones on Mask R-CNN, see Appendix.

#### 5.3 Semantic Segmentation

Training strategies. Aligning with Vmamba [24] settings, we integrate an UperHead into the pre-trained model structure. Utilizing the AdamW optimizer, we initiate the learning rate at 6 × 10−5. The fine-tuning stage consists of 160k iterations, using a batch size of 16. While the standard input resolution stands

- at 512 × 512, we also conduct experiments with 640 × 640 inputs and apply multi-scale (MS) testing to broaden our evaluation.

Results. The EfficientVMamba-T model yields mIoUs of 38.9% (SS) and 39.3% (MS), surpassing the ResNet-50’s 42.1% mIoU with far fewer parameters. EfficientVMamba-S achieves 41.5% (SS) and 42.1 (MS) mIoUs, better-

- Table 4: Results of semantic segmentation on ADE20K using UperNet [53]. We measure the mIoU with single-scale (SS) and multi-scale (MS) testings on the val set. The FLOPs are measured with an input size of 512 × 2048. MLN: multi-level neck.

|Backbone|Image size Params (M) FLOPs (G)|mIoU (SS) mIoU (MS)<br><br>|
|---|---|---|
|ResNet-50 [14] ResNet-101 [14] Swin-T [25] Swin-S [25] DeiT-S + MLN [44] DeiT-B + MLN [44] VMamba-T [24] VMamba-S [24] EfficientVMamba-T EfficientVMamba-S EfficientVMamba-B<br><br>|5122 67 953 5122 85 1030 5122 60 945 5122 81 1039 5122 58 1217 5122 144 2007 5122 55 964 5122 76 1095 5122 14 230 5122 29 505 5122 65 930<br><br>|42.1 42.8<br><br>42.9 44.0<br><br>44.4 45.8<br><br>47.6 49.5 43.8 45.1<br><br>45.5 47.2<br><br>47.3 48.3 49.5 50.5 38.9 39.3 41.5 42.1<br><br>46.5 47.3<br><br><br><br><br>|

- Table 5: Ablation Analysis: Evaluating the Efficacy of Enhanced Spatially Selective Dilatation (ES2D), Assessing the Synergistic Effect of Convolutional Branch Fusion Enhanced with Squeeze-and-Excitation (SE) Techniques, and Investigating the Role of Inverted Residual Blocks in Model Performance. For comparison with the baseline VMamba, we adjust the dimensions and number of layers of it to match the FLOPs.

|Model<br><br>|ES2D Fusion InRes<br><br>|Param (M) FLOPs (G) ACC (%)|
|---|---|---|
|VMamba-T| |2.4 0.9 72.1|
|EfficientVMamba-T|✓ ✗ ✗ ✓ ✓ ✗ ✓ ✓ ✓<br><br>|5 0.8 73.6<br><br>5 0.8 75.1<br>6 0.8 76.5<br>|
|VMamba-B| |16 4.2 80.2<br><br>|
|EfficientVMamba-B<br><br>|✓ ✗ ✗ ✓ ✓ ✗ ✓ ✓ ✓|25 4.0 80.9<br><br>26 4.0 81.2<br><br><br>33 4.0 81.8|

ing the DeiT-S + MLN despite having a lower computational footprint. The EfficientVMamba-B reaches 46.5% (SS) and 47.3% (MS), outperforming the heavier VMamba-S. These findings attest to the EfficientVMamba series’ balance of accuracy and computational efficiency in semantic segmentation.

#### 5.4 Ablation Study

Effect of atrous selective scan. We implement experiment to validate the efficacy of atrous selective scan in Table 5. The upgrade from SS2D to ES2D significantly reduces the computational complexity from 0.8 GFLOPs while retains competitive accuracy at 73.6%, a 1.5% improvement on the tiny variant. Similarly, In the case of base variant, the model utilizing ES2D not only reduces the GFLOPs to 4.0 from VMamba-B’s 4.2 but also exhibits an increase in accuracy from 80.2% to 80.9%. The results suggest that the incorporation of ES2D in

- Table 6: Comparisons of injecting convolution blocks at different stages on ImageNet dataset. We use EfficientVMamba-T in the experiments.

|Layers<br><br>|Stage 1 Stage 2 Stage 3 Stage 4<br><br>|Params (M) FLOPs (G) ACC (%)|
|---|---|---|
|EVSS only Previous Ours|EVSS EVSS EVSS EVSS InRes InRes EVSS EVSS EVSS EVSS InRes InRes<br><br>|5 0.8 75.1<br><br>5 0.8 75.6<br>6 0.8 76.5<br>|

P O

our EfficientVMamba models is one of the key factor in achieving the reduction of computational complexity by skip sampling while preserve the global respective field to keep competitive performance. The reduction of GLOPs also reveals the potency of ES2D in maintaining, and even enhancing, model accuracy while significantly reducing computational overhead, demonstrating its viability for resource-constrained scenarios.

Effect of SSM-Conv fusion block. The integration of a convolutional branch following with a SE block enhances the performance of our model. For tiny variance, adding the local fusion feature extraction improves accuracy from 73.6% to 75.1%. In the case of EfficientVMamba-B, introducing fusion mechanism increases accuracy from 80.9% to 81.2%. The observed performance gains reveals the additional convolutional branch enhance the local feature extraction. By integrating Fusion, the models likely benefit from a more diversified feature set that captures a wider range of spatial details, improving the model’s ability to generalize and thus boosting accuracy. This suggests that the strategic addition of such branches can effectively enhance the model’s performance by providing a comprehensive and more nuanced respective field of the input feature map.

Comparisons of injecting convolution block in different stages. In this paper, we obtain an interesting observation that our SSM based block, EVSS, is more beneficial in the early stages of the network. In contrast, previous works on light-weight ViTs usually inject the convolution blocks in the early stages and adopt Transformer blocks in the deep stages. As shown in Table 6, we compare the performance of injecting convolution blocks in different stages of EfficientVMamba-T, and the results indicate that, adopting Inverted Residual blocks in the deep stages with performs better than that in early stages. A explanation to the opposite phenomenons between our light-weight VSSMs and ViTs is that, the self-attention in Transformers has higher computation complexity and thus its computation at high resolutions is inefficient; while the SSMs, tailored for efficient modeling of long sequences, is more efficient and beneficial on capturing information globally at high resolutions.

### 6 Conclusion

This paper proposed EfficientVMamba, a lightweight state-space network architecture that adeptly combines the strengths of global and local information extraction, addressing the trade-off between model accuracy and computational efficiency. By incorporating an atrous-based selective scan with efficient skip sampling, EfficientVMamba ensures comprehensive global receptive field coverage

while minimizing computational load. The integration of this scanning approach with a convolutional branch, followed by optimization through a Squeeze-andExcitation module, allows for a robust re-balancing of global and local features. Additionally, the innovative use of inverted residual insertion further refines the model’s multi-layer stages, enhancing its depth and effectiveness. Experimental results affirm that EfficientVMamba not only scales down computational complexity to O(N) but also delivers competitive performance across various vision tasks. The achievements of EfficientVMamba highlight its potential as a formidable framework in the evolution of lightweight, efficient, and generalpurpose visual models.

### References

- 1. Bao, H., Dong, L., Piao, S., Wei, F.: Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254 (2021) 1
- 2. Behrouz, A., Hashemi, F.: Graph mamba: Towards learning on graphs with state space models. arXiv preprint arXiv:2402.08678 (2024) 5
- 3. Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., et al.: Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155 (2019) 11
- 4. Chollet, F.: Xception: Deep learning with depthwise separable convolutions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1251–1258 (2017) 2
- 5. Cubuk, E.D., Zoph, B., Mane, D., Vasudevan, V., Le, Q.V.: Autoaugment: Learning augmentation strategies from data. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 113–123 (2019) 10
- 6. Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: Randaugment: Practical automated data augmentation with a reduced search space. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. pp. 702–703

(2020) 4

- 7. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023) 1, 4, 6
- 8. Gu, A., Goel, K., Gupta, A., Ré, C.: On the parameterization and initialization of diagonal state space models. Advances in Neural Information Processing Systems 35, 35971–35983 (2022) 4
- 9. Gu, A., Goel, K., Ré, C.: Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396 (2021) 1, 4
- 10. Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., Ré, C.: Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems 34, 572–585 (2021) 1, 4, 5
- 11. Gupta, A., Gu, A., Berant, J.: Diagonal state spaces are as effective as structured state spaces. Advances in Neural Information Processing Systems 35, 22982–22994

(2022) 1, 4, 5

- 12. Han, K., Wang, Y., Tian, Q., Guo, J., Xu, C., Xu, C.: Ghostnet: More features from cheap operations. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1580–1589 (2020) 2
- 13. Han, K., Wang, Y., Xu, C., Guo, J., Xu, C., Wu, E., Tian, Q.: Ghostnets on heterogeneous devices via cheap operations. International Journal of Computer Vision 130(4), 1050–1069 (2022) 4

- 14. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016) 1, 4, 12, 13, 19
- 15. Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H.: Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017) 2, 4
- 16. Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7132–7141 (2018) 2, 7
- 17. Huang, T., Huang, L., You, S., Wang, F., Qian, C., Xu, C.: Lightvit: Towards light-weight convolution-free vision transformers. arXiv preprint arXiv:2207.05557

(2022) 2, 4, 19

- 18. Iandola, F., Moskewicz, M., Karayev, S., Girshick, R., Darrell, T., Keutzer, K.: Densenet: Implementing efficient convnet descriptor pyramids. arXiv preprint arXiv:1404.1869 (2014) 4
- 19. Li, Y., Yuan, G., Wen, Y., Hu, J., Evangelidis, G., Tulyakov, S., Wang, Y., Ren, J.: Efficientformer: Vision transformers at mobilenet speed. Advances in Neural Information Processing Systems 35, 12934–12949 (2022) 2, 9
- 20. Li, Y., Cai, T., Zhang, Y., Chen, D., Dey, D.: What makes convolutional models great on long sequence modeling? arXiv preprint arXiv:2210.09298 (2022) 1, 4
- 21. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014) 11
- 22. Liu, J., Yang, H., Zhou, H.Y., Xi, Y., Yu, L., Yu, Y., Liang, Y., Shi, G., Zhang, S., Zheng, H., et al.: Swin-umamba: Mamba-based unet with imagenet-based pretraining. arXiv preprint arXiv:2402.03302 (2024) 2, 5
- 23. Liu, X., Peng, H., Zheng, N., Yang, Y., Hu, H., Yuan, Y.: Efficientvit: Memory efficient vision transformer with cascaded group attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14420– 14430 (2023) 12
- 24. Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Liu, Y.: Vmamba: Visual state space model. arXiv preprint arXiv:2401.10166 (2024) 2, 3, 5, 6, 10, 11, 12, 13, 19
- 25. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10012–10022

(2021) 1, 2, 4, 10, 11, 13, 19

- 26. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11976–11986 (2022) 19
- 27. Ma, N., Zhang, X., Zheng, H.T., Sun, J.: Shufflenet v2: Practical guidelines for efficient cnn architecture design. In: Proceedings of the European conference on computer vision (ECCV). pp. 116–131 (2018) 2, 4
- 28. Mehta, S., Rastegari, M.: Mobilevit: light-weight, general-purpose, and mobilefriendly vision transformer. arXiv preprint arXiv:2110.02178 (2021) 2, 9, 11
- 29. Mehta, S., Rastegari, M., Shapiro, L., Hajishirzi, H.: Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9190–9200 (2019) 4

- 30. Nguyen, E., Goel, K., Gu, A., Downs, G., Shah, P., Dao, T., Baccus, S., Ré, C.: S4nd: Modeling images and videos as multidimensional signals with state spaces. Advances in neural information processing systems 35, 2846–2861 (2022) 5
- 31. Orvieto, A., Smith, S.L., Gu, A., Fernando, A., Gulcehre, C., Pascanu, R., De, S.: Resurrecting recurrent neural networks for long sequences. arXiv preprint arXiv:2303.06349 (2023) 1, 4
- 32. Radosavovic, I., Kosaraju, R.P., Girshick, R., He, K., Dollár, P.: Designing network design spaces. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10428–10436 (2020) 11
- 33. Ruan, J., Xiang, S.: Vm-unet: Vision mamba unet for medical image segmentation. arXiv preprint arXiv:2402.02491 (2024) 2, 5
- 34. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4510–4520 (2018) 2, 3, 9, 20
- 35. Schneider, N., Piewak, F., Stiller, C., Franke, U.: Regnet: Multimodal sensor registration using deep neural networks. In: 2017 IEEE intelligent vehicles symposium (IV). pp. 1803–1810. IEEE (2017) 4
- 36. Smith, J.T., Warrington, A., Linderman, S.W.: Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933 (2022) 4
- 37. Su, X., You, S., Xie, J., Zheng, M., Wang, F., Qian, C., Zhang, C., Wang, X., Xu, C.: Vitas: Vision transformer architecture search. In: European Conference on Computer Vision. pp. 139–157. Springer Nature Switzerland Cham (2022) 4
- 38. Szegedy, C., Ioffe, S., Vanhoucke, V., Alemi, A.: Inception-v4, inception-resnet and the impact of residual connections on learning. In: Proceedings of the AAAI conference on artificial intelligence. vol. 31 (2017) 1
- 39. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2818–2826 (2016) 1
- 40. Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., Le, Q.V.: Mnasnet: Platform-aware neural architecture search for mobile. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2820– 2828 (2019) 4
- 41. Tan, M., Le, Q.: Efficientnet: Rethinking model scaling for convolutional neural networks. In: International conference on machine learning. pp. 6105–6114. PMLR

(2019) 1

- 42. Tan, M., Le, Q.V.: Mixconv: Mixed depthwise convolutional kernels. arXiv preprint arXiv:1907.09595 (2019) 4
- 43. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jégou, H.: Training data-efficient image transformers & distillation through attention. In: International conference on machine learning. pp. 10347–10357. PMLR (2021) 10, 11
- 44. Touvron, H., Cord, M., Jégou, H.: Deit iii: Revenge of the vit. In: European Conference on Computer Vision. pp. 516–533. Springer (2022) 4, 13
- 45. Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., Jégou, H.: Going deeper with image transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 32–42 (2021) 4
- 46. Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., Li, Y.: Maxvit: Multi-axis vision transformer. In: European conference on computer vision. pp. 459–479. Springer (2022) 2, 7
- 47. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017) 1

- 48. Wang, C., Tsepa, O., Ma, J., Wang, B.: Graph-mamba: Towards long-range graph sequence modeling with selective state spaces. arXiv preprint arXiv:2402.00789

(2024) 5

- 49. Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 568–578 (2021) 2, 4, 11, 12
- 50. Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pvt v2: Improved baselines with pyramid vision transformer. Computational Visual Media 8(3), 415–424 (2022) 11, 12, 19
- 51. Wang, Y., Xu, C., Qiu, J., Xu, C., Tao, D.: Towards evolutionary compression. In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. pp. 2476–2485 (2018) 4
- 52. Wang, Y., Xu, C., Xu, C., Xu, C., Tao, D.: Learning versatile filters for efficient convolutional neural networks. Advances in Neural Information Processing Systems 31 (2018) 4
- 53. Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. In: Proceedings of the European conference on computer vision (ECCV). pp. 418–434 (2018) 13
- 54. Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K.: Aggregated residual transformations for deep neural networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1492–1500 (2017) 12
- 55. Xing, Z., Ye, T., Yang, Y., Liu, G., Zhu, L.: Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation. arXiv preprint arXiv:2401.13560

(2024) 2, 5

- 56. Yang, C., Wang, Y., Zhang, J., Zhang, H., Wei, Z., Lin, Z., Yuille, A.: Lite vision transformer with enhanced self-attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11998–12008 (2022) 11
- 57. Yu, F., Koltun, V.: Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122 (2015) 7
- 58. Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization strategy to train strong classifiers with localizable features. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 6023–6032 (2019) 4
- 59. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Efficient transformer for high-resolution image restoration. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5728–5739

(2022) 1

- 60. Zhang, H., Cisse, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412 (2017) 4, 10
- 61. Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., Wang, X.: Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417 (2024) 2, 5, 9, 10, 11
- 62. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159

(2020) 4

Table 7: Object detection and instance segmentation results on COCO val set.

Mask R-CNN 1× schedule

|Backbone|Params FLOPs|APb APb50 APb75|APm APm50 APm75|
|---|---|---|---|
|EfficientVMamba-T LightViT-T [17] ResNet-18 [14] PVT-T [50] EfficientVMamba-S<br><br>|11M 60G 28M 187G 31M 207G 33M 208G 31M 197G<br><br>|35.6 57.7 38.0<br><br>37.8 60.7 40.4 34.0 54.0 36.7<br><br>36.7 59.2 39.3 39.3 61.8 42.6<br><br><br>|33.2 54.4 35.1 35.9 57.8 38.0 31.2 51.0 32.7<br><br>35.1 56.7 37.3<br><br>36.7 58.9 39.2<br><br><br>|
|ResNet-50 [14] Swin-T [25] ConvNeXt-T [26] VMamba-T [24] EfficientVMamba-B<br><br>|44M 260G 48M 267G 48M 262G 42M 286G 53M 252G<br><br>|38.2 58.8 41.4<br><br>42.7 65.2 46.8 44.2 66.6 48.3 46.5 68.5 50.7<br><br>43.7 66.2 47.9<br><br><br>|34.7 55.7 37.2<br><br>39.3 62.2 42.2<br>40.1 63.3 42.8 42.1 65.5 45.3 40.2 63.3 42.9<br><br><br>|

Mask R-CNN 3× MS schedule

|EfficientVMamba-T ResNet-18 [14] PVT-T [50] LightViT-T [17] EfficientVMamba-S<br><br>|11M 60G 31M 207G 33M 208G 28M 187G 31M 197G<br><br>|38.3 60.3 41.6 36.9 57.1 40.0<br><br>39.8 62.2 43.0<br><br><br>41.5 64.4 45.1<br><br>41.6 63.9 45.6<br><br><br>|35.3 57.2 37.6 33.6 53.9 35.7<br><br>37.4 59.3 39.9<br><br>38.4 61.2 40.8 38.2 60.8 40.7<br><br><br>|
|---|---|---|---|
|ResNet-50 [14] Swin-T [25] ConvNeXt-T [26] VMamba-T [24] EfficientVMamba-B<br><br>|44M 260G 48M 267G 48M 262G 42M 286G 53M 252G<br><br>|41.0 61.7 44.9 46.0 68.1 50.3 46.2 67.9 50.8 48.5 69.9 52.9 45.0 66.9 49.2<br><br>|37.1 58.4 40.1<br><br>41.6 65.1 44.9<br><br>41.7 65.0 44.9<br><br><br>43.2 66.8 46.3 40.8 64.1 43.7<br><br>|

### Appendix

#### Comparisons with Other Backbones on Mask R-CNN.

We also investigate the performance dynamics of our EfficientVMamba as a lightweight backbone within the Mask R-CNN schedule, as shown in Table 7. For the Mask R-CNN 1× schedule, our EfficientVMamba-T model, with 11M parameters and 60G FLOPs, achieves an Average Precision (AP) of 35.6%. This is 1.6% higher than ResNet-18, which has 31M parameters and 207G FLOPs. EfficientVMamba-S, with a greater number of parameters at 31M and 197G FLOPs, reaches an AP of 39.3%, which is 0.5% above the ResNet-50 model with 44M parameters and 260G FLOPs. Our largest model, EfficientVMamba-B, shows a superior AP of 43.7% with 53M parameters and a reduced computational requirement of 252G FLOPs, outperforming VMamba-T by 2.8%. In terms of Mask R-CNN 3× MS schedule, EfficientVMamba-T maintains an AP of 38.3%, surpassing ResNet-18’s performance by 1.4%. The small variant records an AP of 41.5%, which is a 0.5% improvement over PVT-T with a similar parameter count. Finally, EfficientVMamba-B achieves an AP of 45.0%, indicating a notable advancement of 2.2% over VMamba-T.

#### Comparisons with MobileNetV2 Backbone

We compare variant architectures and reveal a significant performance difference based on the integration of our innovative block, EVSS, versus Inverted Residual (InRes) blocks at specific stages. This results in Table 8 shows that using

Table 8: Comparisons of MobileNetV2 (All stages composed with EVSS.) on ImageNet dataset. We assess both tiny and base models on the ImageNet.

|Variants<br><br>|Stage 1 Stage 2 Stage 3 Stage 4|Params (M) FLOPs (G) ACC (%)|
|---|---|---|
|Tiny|InRes InRes InRes InRes EVSS EVSS EVSS EVSS EVSS EVSS InRes InRes<br><br>|6 0.7 76.0<br><br>5 0.8 75.1<br><br>6 0.8 76.5<br><br><br>|
|Base|InRes InRes InRes InRes EVSS EVSS EVSS EVSS EVSS EVSS InRes InRes<br><br>|34 3.8 81.4 26 4.0 81.2 33 4.0 81.8<br><br>|

InRes consistently across all stages in both tiny and base variants achieves a good performance, with the base variant notably reaching an accuracy of 81.4%. When EVSS is applied across all stages (the strategy of MobileNetV2 [34]), we observe a slight decrease in accuracy for both variants, suggesting a nuanced balance between architectural consistency and computational efficiency. Our fusion approach that combines EVSS in the initial stages with InRes in the later stages enhances accuracy to 76.5% and 81.8% for the tiny and base variants, respectively. This strategy benefits from the early-stage efficiency of EVSS and the advanced-stage convolutional capabilities of InRes, thus optimizing network performance by leveraging the strengths of both block types with limit computational resources.

#### Limitations

Visual state space models that operate with a linear-time complexity O(N) relative to sequence length demonstrate marked enhancements, particularly in high-resolution downstream tasks, which contrasted with prior CNN-based and Transformer-based models. However, the computational design of SSMs inherently exhibits increased computational sophistication than both convolutional and self-attention mechanisms, which complicates the performance of efficient parallel processing. There remains promising potential for future investigation on optimizing the computational efficiency and scalability of visual state space models (SSMs).

