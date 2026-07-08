# arXiv:2408.01031v1[cs.CV]2Aug2024

## POA: Pre-training Once for Models of All Sizes

Yingying Zhang , Xin Guo, Jiangwei Lao, Lei Yu, Lixiang Ru, Jian Wang, Guo Ye, Huimei He, Jingdong Chen, and Ming Yang

Ant Group. qichu.zyy@antgroup.com

Abstract. Large-scale self-supervised pre-training has paved the way for one foundation model to handle many different vision tasks. Most pre-training methodologies train a single model of a certain size at one time. Nevertheless, various computation or storage constraints in realworld scenarios require substantial efforts to develop a series of models with different sizes to deploy. Thus, in this study, we propose a novel tribranch self-supervised training framework, termed as POA (Pre-training Once for All), to tackle this aforementioned issue. Our approach introduces an innovative elastic student branch into a modern self-distillation paradigm. At each pre-training step, we randomly sample a sub-network from the original student to form the elastic student and train all branches in a self-distilling fashion. Once pre-trained, POA allows the extraction of pre-trained models of diverse sizes for downstream tasks. Remarkably, the elastic student facilitates the simultaneous pre-training of multiple models with different sizes, which also acts as an additional ensemble of models of various sizes to enhance representation learning. Extensive experiments, including k-nearest neighbors, linear probing evaluation and assessments on multiple downstream tasks demonstrate the effectiveness and advantages of our POA. It achieves state-of-the-art performance using ViT, Swin Transformer and ResNet backbones, producing around a hundred models with different sizes through a single pre-training session. The code is available at: https://github.com/Qichuzyy/POA.

Keywords: Self-supervised Learning · Pre-training Once for All

### 1 Introduction

Learning generalizable visual representations in a large model by self-supervised learning has delivered superior performance across a wide range of visual tasks [16,19,28,65,68,72] in recent years. Nevertheless, when deployed to real-world applications, a large model has to be adapted to diverse resource limitations in terms of computation, storage, or power consumption, etc. For example, a wellengineered AI product typically comprises a suite of models tailored for varying scenarios, such as Gemini Nano, Pro and Ultra [55]. Given a large pre-trained model, common solutions to deploy it to multiple application scenarios with different resource constraints include additional weight pruning [40,61,70], knowledge distillation [30,32], or even re-training a small network from scratch, which

all require substantial development efforts. Consequently, this issue prompts a critical question: is it possible to pre-train once to produce multiple models with different sizes simultaneously, each delivering sufficiently good representations?

To address this challenge, we introduce a new paradigm of self-supervised learning, called POA (Pre-training Once for All). POA is built upon the prevalent teacher-student self-distillation framework [9,45,74], with an additional innovative elastic student branch. The elastic student branch embeds a series of sub-networks through parameter sharing, upon the observation that smallersized models are sub-networks of a larger-sized one for modern network structures [20,29,42]. Moreover, the parameters of this branch are shared with the original, or intact student. At each pre-training step, we randomly sample a subset of parameters from the intact student to form the corresponding elastic student. Both the original intact student and the elastic student are trained to emulate the output of the teacher network. The teacher itself is continually refined via an exponential moving average (EMA) of the student’s parameters, including the sampled elastic student. The elastic student facilitates effective and efficient pre-training on different subsets of parameters, leading to the successful extraction of high-performance sub-networks from the pre-trained teacher for subsequent downstream scenarios. It also acts as a form of training regularization by enforcing the outputs to match between the teacher and various sub-networks, contributing to a stable training process. Additionally, by serving as an ensemble of different sub-networks across different pre-training steps, the elastic student improves the representation learning [63].

To the best of our knowledge, POA represents the first self-supervised learning method capable of training multiple-sized models concurrently, each obtaining high-quality representations for different resource constraints without further pre-training. Figure 1 displays the k-nearest neighbors (k-NN) evaluation results for 143 sub-networks extracted from a ViTL model [20] pre-trained by POA. By choosing different elastic widths and depths, which will be explained in Sec.

|[Figure 1]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

- 78
- 79
- 80
- 81
- 82

POA ViT-L/16: 82.3%

- 76

- 77

- 78

- 79

- 80

- 81

- 82

k-NN accuracy(%)

SOTA ViT-L/16: 82.0%

POA ViT-B/16: 80.9%

24

POA ViT-S/16: 76.8%

SOTA ViT-B/16 : 78.0%

22

Elasic Deph

20

SOTA ViT-S/16: 75.6%

18

16

400

500

14

600

700

800

12

ElasicWidh

900

1000

- 3.1, the pre-trained teacher model can generate a sufficient number of candidate sub-networks for the selection of the suitable model tailored for downstream applications according to the computational resources available. Notably, each sub-network is well-trained and exhibits superior performance, thanks to our meticulous design on same-view distillation, as validated in Sec. 4.4. In particular, the ViT-S, ViT-B and ViT-

Fig. 1: The k-NN evaluation accuracy of 143 elastic ViTs derived from the ViT-L/16 teacher model pre-trained with POA.

L models set new benchmarks, achieving the state-of-the-art (SOTA) results compared with those pre-trained by existing methods [45,48,74].

To rigorously evaluate the efficacy of our approach, we conduct extensive experiments using three widely-used backbone architectures, i.e., ViT [20], Swin Transformer [42], and ResNet [29]. Each backbone is pre-trained on ImageNet-1K dataset and assessed using k-NN and linear probing classification evaluation, as well as downstream dense prediction tasks such as object detection and semantic segmentation. Our method achieves SOTA accuracy across various model sizes with a single pre-training session. The technical contributions of this work are summarized as follows:

- – To the best of our knowledge, POA is the first pre-training paradigm that integrates unsupervised representation learning and once-for-all model generation within a single pre-training session. It tackles the Pre-training-Oncefor-All challenge which has been seldomly explored by the community but is of great practical significance for real-world deployment that usually requires a suite of models.
- – We propose a novel and elegant component called Elastic Student, featuring a range of elastic operators that enable POA to be compatible with popular backbone structures including ViT, Swin Transformer and ResNet. It provides the capability to generate models of diverse sizes. Furthermore, it serves as a model ensemble to smooth training process and improve learned representations.
- – Through thorough assessments using k-NN, linear probing and downstream dense task evaluation, our approach exhibits superior performance over existing state-of-the-art pre-training methods across multiple metrics. Moreover, we compare our POA with Self-Supervised Distillation (SEED) [21], a knowledge distillation method designed especially for self-supervised learning, further validating POA’s effectiveness.

### 2 Related Work

#### 2.1 Self-supervised Learning

Self-supervised learning (SSL) frameworks commonly fall into two categories, generative and discriminative SSL. Most generative SSL approaches [12,15,26, 31, 34, 37, 56, 67, 71] focus on learning image representations directly in pixel space. On the other hand, discriminative SSL methods [8,10,27,39,53,60,62,66] aim to learn representations by pulling those of different views of the same image closer while separating the representations of views from different images.

Contrastive learning (CL) with the InfoNCE loss [44] has emerged as a popular approach for discriminative SSL methods, attracting significant research interests in recent years. Although CL methods prevent the collapse of network representations through the use of negative samples, they still suffer from the dimensional collapse, where representations tend to collapse into a lowdimensional manifold. Grill et al. introduced a distillation-based asymmetric

framework known as BYOL [24], which circumvents collapse without self-labeling or contrastive loss relying on negative samples. Following this work, distillationbased frameworks have become a prevailing trend in self-supervised learning. These frameworks [13,23,66] often merge with others to enhance overall performance. DINO [9] presented a simple self-distillation framework and has demonstrated impressive results in ViT pre-training. Subsequent works [45,74] further improved the pre-training performance via masked token or the novel clustering design. Given the substantial benefits of distillation-based methods over other SSL techniques, we have developed our POA SSL framework upon these successful methodologies.

#### 2.2 Dynamic Architecture.

Recently, Chen et al. proposed AutoFormer [64], which trained a supernet to support the effective search of optimal sub-network under some specific parameter number constraints. On the basis of [64], MaskTAS [69] introduced a self-supervised transformer architecture search method. Cai et al. [6] trained a network that accommodates various architectural configurations to reduce the training expense. Their methodologies enable the extraction of a specialized subnetwork from the trained supernet. The design of the elastic student in our POA SSL is inspired by the weight-sharing strategy employed in these neural architecture search (NAS) methods. However, our implementation differs significantly due to the distinct purpose from NAS. Specifically, NAS aims to discover the optimal architecture within certain parameter constraints, which typically involves a huge number of sub-networks (more than 1016) in the search space. Given the limitations on the number of training iterations and the network parameter capacity, it is challenging to ensure high performance across all sub-networks. After training the supernet, NAS requires a subsequent phase of searching and re-training to finalize the output model. In contrast, our POA SSL defines a sufficiently large yet compact set of sub-networks with elastic depths and widths for the purpose of pre-training models of various sizes via a single training session. Additional design on the same-view distillation guarantees that all elastic sub-networks within our framework are adequately and efficiently trained. As a result, our POA can readily extract a range of sub-networks from the teacher model without the need for extra pre-training, facilitating an easy selection of a suitable sub-network for different computational contexts. The design of elastic student is somewhat akin to the supervised learning method, Cosub [59]. The main difference is that Cosub simply skips blocks, making only the depth elastic.

### 3 POA Self-supervised Learning Framework

Our primary goal is to pre-train models of multiple sizes via a single selfsupervised pre-training session. To this end, we propose a novel SSL framework named POA, inspired by the latest progress in self-distillation techniques [9, 45, 74]. The architecture of POA is illustrated in Figure 2, encompassing a

|Softmax Centering<br><br>+|
|---|

|Teacher| |
|---|---|
| | |

|Head| |
|---|---|
| | |

[Figure 2]

Softmax Centering

𝑝

𝑍

sg

[Figure 3]

ℒ

EMA

𝑥

|Intact Student| |
|---|---|
| | |

EMA

𝑝

ℒ

𝑍

|Softmax|
|---|

| |Head| |
|---|---|---|
| | | |
| | | |

[Figure 4]

Softmax

𝑥

ℒ

Share

|Elastic Student| |
|---|---|
| | |

𝑥

𝑝

𝑍

- Fig. 2: Overview of the POA SSL: Given an image x, two augmented views xa and xb are generated. These views are input into three branches: a teacher, an intact student, and an elastic student, the latter being derived from the intact student. POA optimizes distillation losses in a twofold manner: the intact and the elastic students are distilled from the teacher using the cross-view data respectively, and additionally, the elastic student is distilled from the intact student using the same-view data.

teacher, an intact student, an elastic student and their corresponding heads. The teacher is updated with an EMA of the students. The elastic student is a derivative of the intact one, with both the backbones and heads’ parameters shared. We leverage distillation in two aspects: both the intact and elastic students are distilled from the teacher using different views of the same image, while the elastic student additionally learns from the intact student using the identical views. The cross-view distillation works as a form of representation learning, as introduced in [9,45,74]. Notably, in addition to the regular EMA update with only the intact student as [9,45,74], the elastic student provides a randomly-sampled sub-network at each pre-training step to participate in the teacher’s EMA refinement. This procedure actually simulates an ensemble of multiple sub-networks, which is also proven to be beneficial in the realm of supervised learning [63]. The same-view distillation is a standard knowledge distillation between the intact and elastic students, promoting the quality of the elastic one. Experiments in Sec. 4.4 validate our design in details.

#### 3.1 Design of Elastic Student

The elastic student is a sub-network whose parameters are extracted from the intact student. In the context of a transformer backbone, the width refers to the dimensionality of the tokens, whereas for a convolutional backbone, the width indicates the number of channels. We denote the depth as the number of basic blocks in either a transformer or a convolutional network. Given the value of the width and depth, it yields a certain network structure. For simplicity, we focus on detailing the elastic design of the ViT in this section. For the similar elastic design of the Swin Transformer and ResNet, please refer to Appendix A.

A basic block of ViT mainly consists of a multi-head self-attention (MSA) module and a multi-layer perceptron (MLP) module. Layer Normalization (LN) [3] is applied before each module, with residual connections after each module.

##### As shown in the left part of Figure 3, an elastic block refers to a stack of elastic MSA, MLP, and LN after adjusting the width in the original basic block in ViT. In our approach, the elastic student branch is constructed by assembling a specific number of these elastic blocks at each training iteration.

output token

Elastic MSA

+

Elastic Block

𝐿 ×

Elastic MLP

𝑤 𝑤

𝐷 (𝑖 = 2)

Elastic LN

Attention & Concatenation

head 1 head 𝑁 − 1 head 𝑁

+

𝑉

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
|𝐷|

𝐾

…

Elastic MSA

𝑄

Elastic LN

𝑤

𝑤

𝐷 (𝑖 = 2) 𝐷

Embedded Patches

𝐷 (𝐷 )

input token

- Fig. 3: Illustration of the elastic MSA in an elastic ViT block. To be concise, we simply exclude the projection layers that correspond to K and V in each head.

Elastic MSA An original, or intact MSA module consists of three major components, i.e., input projection layers, an operator contains attention & concatenation, and an output projection layer. We define a projection layer as (w∗,b∗), where w∗ is the linear transformation weight, b∗ denotes the corresponding bias, and ∗ indicates the name of the layer. As shown in the right part of Figure 3, given a token dimension of Dmax = Nh · Dh, where Nh is the number of attention heads and Dh is the head dimension, an input sequence z ∈ RT×D

max with length T is initially projected to form the queries Q ∈ RT×D

h, and values V ∈ RT×D

h, keys K ∈ RT×D

h by the intact MSA. To generate elastic MSA, we define M + 1 elastic widths, including Dmax, spaced at intervals of Dh as follows:

Di = (Nh − i) · Dh, ∀i ∈ {0,1,...,M}, M < Nh. (1) For each elastic width Di, the weight wia1 ∈ RD

h that generate Q, K, and V of each head are extracted from the corresponding input projection layer (wa1, ba1) in the intact MSA, as wia1 = wa1[:,: Di] · αi and bai1 = ba1. Here, αi represents the scaling factor to cope with the reduction of the input dimension, calculated as αi = Dmax/Di. As the reduction of width, the number of attention heads in the elastic MAS is naturally reduced to Nh −i. Similarly, for the output projection layer (wa2, ba2), the weight wia2 ∈ RD

h×Di and bias bai1 ∈ RD

i×Di and bias bai2 ∈ RD

i are extracted as:

wia2 = wa2[: Di,: Di] · αi bai2 = ba2[: Di]. (2)

Elastic MLP An original, or intact MLP module in the ViT block contains two projection layers. The first layer (wm1,bm1) expands the dimension of embedding by a factor of s, which is generally set to 4 in the ViT structure. The second layer (wm2,bm2) then projects it back to the original dimension. The parameters for both layers of the elastic MLP are extracted in a manner analogous to that

described in Equation 2 as follows:

wim1 = wm1[: Di · s,: Di] · αi bmi 1 = bm1[: Di · s] wim2 = wm2[: Di,: Di · s] · αi bmi 2 = bm2[: Di].

(3)

Elastic LN For elastic LN, we directly use the first Di elements of the parameter inside the original LN, akin to the bias extraction in Equation 2.

To create a sub-network with Li elastic blocks from the intact ViT comprising Lmax blocks, we introduce a set of N + 1 elastic depths, defined as Li = Lmax −i, ∀i ∈ {0,1,...,N}, N < Lmax. For a specific depth Li, we select the corresponding blocks based on their block IDs at equal intervals. Each block ID BIDL

- i
- j that is activated for depth Li can be formulated as:

(Lmax − 1) · j Li − 1

##### BIDL

, ∀j ∈ {0,1,...,Li − 1}. (4)

- i
- j =

Consequently, by combining elastic widths and depths, we can generate a total of (N + 1) · (M + 1) distinct sub-networks. For instance, by setting the elastic width to 384 and the elastic depth to 12, we can directly extract a ViT-S from the intact network such as ViT-L. During each iteration of the pre-training, we randomly select one of these sub-networks to serve as the elastic student branch.

#### 3.2 Distillation between Views

POA performs distillation across its three branches accordingly. Given a pair of globally augmented views of an input image x, denoted as xa and xb, the teacher encoder ET extracts the feature Za = ET(xa) using xa as input. Simultaneously, xb is fed into both the intact student encoder EIS and the elastic student encoder EES, producing the features Zb1 = EIS(xb) and Zb2 = EES(xb), respectively. The feature output from the teacher encoder, Za, is then processed by the teacher head HT, followed by centering with the Sinkhorn-Knopp (SK) [17] algorithm and normalization using a temperature-scaled softmax to generate the probability pa as follows:

exp(lai /τ) P−1 k=0 exp(lak/τ)

la = SK(HT(Za)), la ∈ RP pia =

, ∀i ∈ {0,...,P − 1}, (5)

where P is the number of prototypes and τ > 0 is a temperature parameter. In a similar fashion, we compute the probabilities pib1 and pib2 for the intact and elastic student encoders, respectively, by processing the outputs through the student heads HIS and HES. These outputs are then passed through a temperature-scaled softmax function using a temperature parameter τ′ tailored for the student. It should be noted that HIS and HES share the identical parameters, except that the first projection layer of HES is adapted similarly as in Equation 2 to conform to align with the corresponding dimensionality. For simplicity, we omit the explicit expressions for pib1 and pib2, as they follow a similar

calculation to Equation 5. For the intact student branch, we perform distillation from the teacher using the cross-view data as follows:

LgIS = −pa log(pb1). (6) The elastic student branch plays a pivotal role in our POA framework. To ensure adequate training of this branch, we employ a dual distillation from the teacher and intact student branch. The first distillation involves the teacher model, which utilizes the cross-view data to guide the learning of representations. The second is a distillation process with the intact student model, which uses the same-view data. This same-view distillation is responsible for transferring the representations learned by the intact student to the elastic student branch. The loss functions for this dual distillation process are formulated as follows:

LgES1 = −pa log(pb2), LgES2 = −pb1 log(pb2). (7) Note that in both loss functions, we sum over all prototypes to compute the cross-entropy loss between the respective probability distributions.

#### 3.3 Overall Loss of POA

Following the SSL methods such as [9,45,74], we employ a multi-crop strategy [8] to create various distorted views from a single image. Apart from the two global views previously mentioned, we also generate v local views with lower resolution xl

. These local views are processed by both students to promote local-to-global correspondence. The local distillation losses for the intact and elastic students are computed as follows:

,xl

,...,xl

1

2

v

v

1 v

LlIS = −

i1), (8)

pa log(pl

i=1

v

v

1 v

1 v

LlES1 = −

i2), LlES2 = −

i2), (9)

pa log(pl

pl

i1 log(pl

i=1

i=1

where pl

and pl

are the probability produced by the intact and elastic student branches for the local view li, respectively. The total distillation loss of the intact and elastic student is calculated by summing them with the factor λ:

i1

i2

LS = λ(LgIS + LlIS) + (1 − λ)((LgES1 + LlES1) + (LgES2 + LlES2))

(10)

= λLIS + (1 − λ)(LES1 + LES2).

To ensure sufficient training of each sub-network from the elastic student, we introduce multiple projection heads (MPH) positioned after the backbone network. Each projection head has exactly the same structure, except for a different number of prototypes. For each head, the distillation loss LSi for both the intact and elastic student is calculated with Equation. 10. Finally, the overall loss function in the POA framework with H projection heads is formulated as: L = H1 Hi=1 LSi.

### 4 Experiments

#### 4.1 Implementation Details

Backbones. We have trained our POA using ViT, Swin Transformer and ResNet backbones, respectively. For the ViT, we configure the patch size to 16 and the dimension of each head in the MSA to 64. This aligns with the configurations typically used in existing SSL methods [9,48,51,74]. For the smallest and largest elastic networks of ViT, we choose the ViT-S and ViT-L, respectively. This leads to 11 elastic widths and 13 elastic depths, yielding a total of 143 ViT subnetworks. In the case of the Swin Transformer, we set the smallest and largest elastic networks as Swin-T and Swin-B, respectively. This configuration yields a total of 39 Swin sub-networks by multiplying the number of widths and depths as 3 × 13. For the ResNet, we establish the smallest and largest elastic network configurations as ResNet-50 and ResNet-152. Consequently, this setup accounts for a total of 465 ResNet sub-networks, which is the product of 3 × 155.

Pre-Training Setup. We pre-train all models on the ImageNet-1K dataset [18] without the labels. The process employs the AdamW optimizer [43] with a batch size of 1600, which is distributed across 32 A100 GPUs when employing a ViT backbone. We adopt a learning rate schedule that begins with a linear warm-up during the first 10 epochs, reaching a base value that is scaled proportionally to the total batch size: lr = 0.004 × batch size/1024, in line with [45]. Following this warm-up period, the learning rate is decayed with a cosine schedule. Similarly, the weight decay follows a cosine schedule, starting at 0.04 and increasing to 0.4. The student network’s temperature τ′ is fixed at 0.1, whereas the teacher temperature τ changes with a linear warm-up from 0.04 to 0.07 over the first 30 epochs. For further details, please refer to Appendix C.

#### 4.2 Experiments on ImageNet-1K

After unsupervised pre-training, we assess the model’s performance using two widely-recognized evaluation protocols in SSL domain on ImageNet-1K dataset, i.e., k-NN and linear probing. To ensure a fair comparison between SSL methods that employ different numbers of crop views for data augmentation, Zhou et al. [74] introduced the effective training epoch as a measure to quantify the extent of a method’s pre-training. We report the effective training epochs of the SSL methods for comparison. For additional evaluations including fine-tuning, semi-supervised and unsupervised learning, please refer to Appendix D.

k-NN and Linear Probing. To assess the quality of pre-trained features, we employ a k-NN classifier and a linear classifier on the frozen representations. For both the k-NN and linear probing (LP) evaluation, we follow the evaluation protocols established in [9,45,74]. The performance of our method when being trained using ResNet, Swin Transformer and ViT backbones is reported in Table 1. Our POA SSL achieves the SOTA k-NN accuracy of 82.3% and LP accuracy

- Table 1: Comparison results of k-NN and linear probing classification accuracy (%) on the ImageNet-1K dataset. "Param." refers to the number of parameters. "Epo." represents the number of effective training epochs following [74]. "/16" denotes patch size of 16. "/W7" means the window size of 7. "∗" indicates our implementation based on official codebase. "†" denotes reproduced results using the released code.

Method Epo. k-NN LP ViT-S/16(Param. 21M) DINO 3200 74.5 77.0 iBOT 3200 75.2 77.9 ENT 3200 75.3 77.7 POA 0 76.8 77.6

Method Epo. k-NN LP ResNet-50(Param. 23M) VICReg 2000 - 73.2 SwAV 2400 65.7 75.3 DINO 3200 67.5 75.3 UniGrad 2400 - 75.5 SCFS 3200 68.5 75.7

Method Epo. k-NN LP Swin-T/W7(Param. 28M) SMoG 1200 - 77.7 iBOT 1200 75.3 78.6

- DINOv2∗ 1200 75.4 78.0 EsViT 1200 75.7 78.1 POA 0 77.5 78.9 Swin-S/W7(Param. 49M)

- DINOv2∗ 1200 76.1 79.8

ViT-B/16(Param. 85M) DINO 1600 76.1 78.2 ENT 1600 77.1 79.1 iBOT 1600 77.1 79.5 POA 0 80.9 81.7

- ReLICv2 4000 - 77.1 POA 0 73.4 76.9 ResNet-101(Param. 41M)

- ReLICv2 4000 - 78.7 POA 0 75.7 79.1 ResNet-152(Param. 56M)

- ReLICv2 4000 - 79.3 POA 2400 76.4 79.9

- EsViT 1200 77.7 79.5 POA 0 79.3 81.3

Swin-B/W7(Param. 87M) DINOv2∗ 1200 76.9 80.9

- EsViT 1200 78.9 80.4 POA 1200 79.6 82.0

ViT-L/16(Param. 307M) iBOT 1200 78.0 81.0 DINOv2† 1200 82.0 83.3 POA 1200 82.3 83.6

(a) ResNet backbone.

(b) Swin backbone.

(c) ViT backbone.

of 83.6% when using the ViT-L/16 backbone. By employing the sub-network extraction approach outlined in Section 3.1, we derive the sub-networks ViT-S/16 and ViT-B/16 from the teacher ViT-L/16 without any additional pre-training. Thus, the number of effective training epochs of them is 0. Notably, the extracted ViT-S/16 achieves the SOTA k-NN accuracy of 76.8% and LP accuracy on par with the previous SOTA of 77.9% reported by iBOT [74]. Our derived ViT-B/16 model sets new benchmarks for both k-NN and LP, with accuracy of 80.9% and 81.7%, respectively. For the models using Swin Transformer and ResNet backbones, POA also reaches SOTA performance in k-NN and LP accuracy. The only exception is the LP accuracy of ResNet-50, which is competitive with that of ReLICv2 [58], despite the latter being trained over a much longer period (4000 epochs). The superior performance achieved across a range of backbone architectures confirms the effectiveness and versatility of our method. For additional detailed comparison with 27 existing methods, please see Appendix D.1.

#### 4.3 Evaluation on Downstream Visual Tasks

Object Detection and Instance Segmentation on COCO Dataset. For a fair comparison, we utilize the Cascade Mask R-CNN framework [7,28], which generates both bounding boxes and instance masks, in line with previous approach [74], on the COCO dataset [41]. We benchmark our results against existing SSL methods that generate pre-trained ViT-S/16 and ViT-B/16 backbones. As shown in Table 2, POA boosts the bounding box average precision (APb) for

ViT-S/16 from 49.4 to 50.6 and the mask average precision (APm) from 42.6 to 43.8. When applied to ViT-B/16, POA attains an APb of 52.4 and an APm of 45.4, which represents a remarkable step forward over previous SOTA.

Semantic Segmentation on ADE20K Dataset. For the semantic segmentation task, we primarily focus on two settings on the ADE20K dataset [73], following [74]. First, akin to the linear evaluation protocol in classification, we evaluate the quality of representations by keeping the patch features fixed and only fine-tuning one linear layer. This approach offers a clear comparison of representation quality. Second, we employ the UPerNet [65] as the task head and fine-tune the entire network. As depicted in Table 2, our POA significantly outperforms the supervised baseline using the ViT-S/16 backbone, achieving a substantial increase of 2.2 in mean Intersection over Union (mIoU), and surpassing iBOT by 1.3 mIoU. With the ViT-B/16 backbone, POA exceeds the previously best-performing method, iBOT, by 0.4 mIoU when utilizing UPerNet. Furthermore, under the assessment using solely a linear head, POA obtains an impressive improvement of 2.0 mIoU over iBOT as the performance is largely determiend by the quality of the pre-trained representation.

- Table 2: Evaluation results on downstream detection and segmentation tasks. Seg.† indicates the use of a linear head for semantic segmentation.

Det. ISeg. Seg.† Seg. APb APm mIoU APb APm mIoU mIoU

Det. ISeg. Seg.

Method Arch.

Arch.

Sup. ViT-S/16 46.2 40.1 44.5 ViT-B/16 49.8 43.2 35.4 46.6 BEiT ViT-S/16 - - - ViT-B/16 50.1 43.5 27.4 45.8 DINO ViT-S/16 - - - ViT-B/16 50.1 43.4 34.5 46.8 iBOT ViT-S/16 49.4 42.6 45.4 ViT-B/16 51.2 44.2 38.3 50.0 POA ViT-S/16 50.6 43.8 46.7 ViT-B/16 52.4 45.4 40.3 50.4

#### 4.4 Ablations and Discussions

In this section, we conduct an empirical analysis of POA using ViT as backbone. Our investigation includes the impact of the loss functions LES1 and LES2, in addition with the effectiveness of multiple projection heads. Moreover, we compare our POA with knowledge distillation techniques for self-supervised learning to demonstrate the advantages of combining once-for-all model generation with self-distillation in a unified framework. For more ablation studies, we direct readers to the Appendix D.8. In addition, we further contrast our POA with three variants tailored for elastic pre-training to showcase POA’s superiority. Finally, we discuss how the elastic student facilitates the pre-training.

Importance of Each Component. We evaluate the contributions of the components on a ViT backbone. Table 3 shows the performance of different component combinations. First, we note that employing multiple projection heads (MPH) enhances the learned representations for each elastic sub-network, particularly for smaller ones. The design consideration behind MPH is that for each

- Table 3: Contributions of each component in POA framework. We conduct assessments with the k-NN and linear probing (LP) evaluations. MPH denotes the multiple projection heads with different numbers of prototypes.

MPH LES1 LES2

k-NN LP

ViT-S/16 ViT-B/16 ViT-L/16 ViT-S/16 ViT-B/16 ViT-L/16

✓ ✓ ✓ 76.8 80.9 82.3 77.6 81.7 83.6 ✓ ✓ 76.2 80.7 82.2 77.3 81.6 83.4 ✓ 75.1 80.2 82.2 75.8 81.0 83.4 ✓ 72.8 79.1 82.1 75.3 80.8 83.3

pre-training iteration, the sub-network is chosen randomly, leading to a relatively insufficient optimization. MPH introduces different sets of prototypes, which act as multiple semantic spaces for representation learning, enabling the teacher to distill various aspects of learned knowledge into the sub-network. Furthermore, we ascertain that the same-view distillation loss LES2 is crucial for the representation quality of elastic sub-networks. Omitting LES2 causes a significant drop in k-NN accuracy, by 3.4% for ViT-S/16 and 1.6% for ViT-B/16. In addition, LES2 is more important than cross-view distillation LES1 in terms of sub-networks’ representation quality. The underlying reason is that the cross-view distillation is for the unsupervised representation learning, while the same-view distillation improves the sub-networks by distilling previously learned representations from the intact student. Table 4 confirms that distillation from already-good representations is more effective than representation learning, especially for smaller networks. It also explains why employing three different views in our POA is unnecessary.

- Table 4: Comparison with knowledge distillation for self-supervised learning. The teacher name "ViT-L/16-600" denotes a teacher model (ViT-L/16) that has been pretrained with 600 effective epochs. The student name "ViT-S/16-600" refers to a student model (ViT-S/16) that has undergone distillation from the pre-trained teacher with 600 effective epochs. The number of the left side of "→" indicates the performance of the teacher model when pre-trained individually. The number of the right side denotes the performance achieved after distillation using the SEED [21].

Method Teacher Student Total Epochs k-NN LP DINOv2 ViT-S/16 1200 72.2 73.1 DINOv2+SEED ViT-L/16-600 ViT-S/16-600 1200 81.3 → 74.0 82.4 → 75.2 DINOv2+SEED ViT-L/16-1200 ViT-S/16-1200 2400 82.0 → 75.5 83.3 → 76.2

POA ViT-S/16 1200 76.8 77.6 DINOv2 ViT-B/16 1200 77.4 78.5 DINOv2+SEED ViT-L/16-600 ViT-B/16-600 1200 81.3 → 78.8 82.4 → 80.0 DINOv2+SEED ViT-L/16-1200 ViT-B/16-1200 2400 82.0 → 79.7 83.3 → 80.9

POA ViT-B/16 1200 80.9 81.7

##### Comparison with Knowledge Distillation. Knowledge distillation (KD) is a proven strategy for enhancing small networks by leveraging the knowledge of

a well-trained, larger network. We compare our method with a self-supervised KD method SEED [21] that employs a pre-trained network from the previous SOTA DINOv2 as the teacher for distilling knowledge into smaller networks. We observe that SEED significantly boosts the performance of ViT-S/16 and ViTB/16 when a pre-trained ViT-L/16 serves as the teacher. Specifically, with the same number of training epochs, SEED yields k-NN accuracy improvements of 1.8% and 1.4% for ViT-S/16 and ViT-B/16 respectively, compared with a learnfrom-scratch setup. Our POA further outperforms SEED, delivering superior kNN accuracy gains of 2.8% (76.8% vs. 74.0%) and 2.1% (80.9% vs. 78.8%) on ViT-S/16 and ViT-B/16. Moreover, the ViT-S/16 and ViT-B/16 derived directly from the pre-trained teacher from our POA, without additional pre-training, perform better than those enhanced by SEED, despite the latter undergoing two times of training epochs. These results demonstrate the effectiveness of our unified design, which integrates pre-training with once-for-all model generation. Additional comparison on training cost can be found in Appendix D.11.

Alternative Designs to Elastic Pre-training. Upon the proposed POA, we investigate three alternatives for concurrently pre-training multi-sized models within a single pre-training session. As illustrated in Figure 4, the first variant, POA-V1, eliminates the intact student and modifies the teacher to be elastic which aligns with the architecture of the elastic student. The second variant, POA-V2, only removes the intact student. However, in this model, the intact teacher is directly updated via an EMA of the elastic students. These two variants bear resemblance to current teacher-student self-distillation paradigm, with the primary differences in the structure of the teacher or student network. For the POA-V3, we introduce an additional elastic teacher branch that mirrors the architecture of the elastic student, enabling further cross-view distillation. For more details of the three variants, please refer to the Appendix D.9.

|Teacher|
|---|

|Elastic Teacher<br><br>|
|---|
|Intact Teacher|

|Teacher|
|---|

|Elastic Teacher<br><br>|
|---|

|Head|
|---|

|Head|
|---|

| |Head|
|---|---|
| | |

| |Head|
|---|---|
| | |

EMA

|Intact Student|
|---|

EMA

EMA

EMA

EMA

EMA

###### EMA

|Elastic Student|
|---|

EMA

|Elastic Student<br><br>|
|---|

|Intact Student|
|---|

| |Head|
|---|---|
| | |

| |Head|
|---|---|
| | |

| |Head|
|---|---|
| | |

| |Head|
|---|---|
| | |

|Elastic Student|
|---|

|Elastic Student<br><br>|
|---|

POA

POA-V2

POA-V3

POA-V1

Fig. 4: Illustration of different variants of POA.

We compare the k-NN and linear probing accuracy of each variant in Table.

- 5. The results indicate that the performance drops dramatically for the variants with two branches. This decline can be attributed to two reasons: 1) in each training iteration, only a subset of the student’s parameters are updated, leading to relatively insufficient overall pre-training; and 2) each elastic network undergoes training exclusively with the cross-view distillation, missing out the standard distillation guidance from a larger network, which our ablation study suggests is crucial for the efficacy of elastic networks. Although the POA-V3,

featuring an additional elastic teacher branch, performs slightly better than our POA, it does so at the expense of increased model complexity and computational costs, making it a less appealing scheme than POA.

- Table 5: k-NN and linear probing (LP) comparison of different variants of POA.

k-NN LP ViT-S/16 ViT-B/16 ViT-L/16 ViT-S/16 ViT-B/16 ViT-L/16

Method

POA 76.8 80.9 82.3 77.6 81.7 83.6 POA-V1 73.7 78.5 79.2 74.5 79.4 80.9 POA-V2 74.3 79.1 80.9 75.1 80.0 82.3 POA-V3 76.9 81.0 82.3 77.8 81.8 83.6

How Does Elastic Student Facilitate Pre-training? The elastic student branch enables the derivation of diverse-sized models directly from a pre-trained teacher, while simultaneously enhancing the learned representation. The elastic branch plays a dual role. First, it acts as a training regularization to stabilize the training progress. In our preliminary experiments, we observe that pre-training loss of a ResNet backbone easily diverges to NaN without the elastic branch. Conversely, the inclusion of the elastic student yields a highly stable pre-training. Furthermore, the elastic student provides hundreds of sub-network candidates that will be assembled to the teacher during pre-training. Unlike existing self-distillation methods, the teacher in the POA SSL integrates a series of sub-networks through an EMA update. Wortsman et al. [63] have shown that averaging the weights of multiple models typically improves accuracy and robustness. Similarly, the integration of different sub-networks for the teacher is effective in improving the representation quality. At the same time, an improved teacher promotes the learning of students in return, fostering a positive feedback loop for pre-training. For the visualization of representations, please refer to Appendix E.2.

### 5 Conclusion

In this study, we tackle the challenge of efficiently and effectively pre-training models of various sizes within a single self-supervised learning session, facilitating model deployment given different resource constraints. We propose a novel selfsupervised learning paradigm, termed POA, which features an integrated design combining self-distillation and once-for-all model generation. It allows for the simultaneous pre-training of models of multiple sizes through an elastic branch design. POA enables the direct generation of varied-sized models from a pretrained teacher, which are ready for downstream tasks without additional pretraining. This advantage significantly improves the deployment flexibility and facilitates our pre-trained model to achieve SOTA results across various vision tasks. Looking forward, we plan to extend POA to Multimodal Large Language Models, tapping into its vast potential for real-world AI product deployment.

### A Design of Elastic Student

#### A.1 Elastic Swin Transformer

The Swin Transformer’s basic block [42] closely resembles that of the ViT, and we apply the same parameter extraction method to its Shifted Window based SelfAttention (SW-MSA) and MLP modules. In the Swin Transformer architecture, the final linear layer of the MLP in the last block of each stage quadruples the dimension of the tokens. Subsequently, a Patch Merging (PM) layer is employed to halve the token dimension. We adjust the parameters of these linear layers in the MLP and PM to match their respective expansion and reduction ratios as follows:

wimlp = wmlp[: Dis · 4,: Dis] · αi, wipm = wpm[: Dis · 2,: Dis · 4] · αi, (11)

where Dis represents the i-th elastic width of stage s, wmlp are the weight of the last linear layer of the MLP, and wpm are the weight of the PM module. For the Swin Transformer, we apply elastic depth exclusively to the third stage, where the number of blocks is larger. The approach for selecting the activated block IDs follows the same method as used for the ViT, ensuring consistency across different transformer architectures in the process of creating sub-networks with varying depths.

#### A.2 Elastic ResNet

ResNet is composed of many basic units known as bottleneck building blocks [29]. In ResNet, we primarily focus on constructing sub-networks with varying numbers of blocks. We make the feature dimension(channel) of the middle layer in each building block elastic, while keeping the output dimension from each block unchanged. The weights of the three convolutional layers within a block are extracted as follows:

wi1 = w1[: Dis,:,:,:], wi2 = w2[: Dis,: Dis,:,:] · αi, wi3 = w3[:,: Dis,:,:] · αi, (12)

where Dis denotes the number of mid-layer channel for stage s. We implement elastic depth in both the second and third stages of ResNet. The method for selecting block IDs in each stage is consistent with that used for the ViT and Swin Transformer.

### B Pseudocode

Algorithm 1: POA PyTorch-like Pseudocode without multi-crop augmentation and MPH.

Input : gIS,gT // intact student and teacher network M,N // number of elastic width and depths Dmax,Lmax // max width, max depth Dh // head dimension τs,τt,µ // temperatures and momentum for EMA λ,γ // loss weight

Initialize: gT.params = gIS.params, cand_ids = [[i,j] for i in

range(M + 1) for j in range(N + 1)], idx=0

for x in dataloader do xa, xb=augment(x), augment(x) // random views gES, idx = ExtractElastic(cand_ids, idx, Dmax, Dh, Lmax, gIS) fa = gT(xa), fb1 = gIS(xb), fb2 = gES(xb) LIS = H(fb1,fa,τs,τt) LES = LES1 + LES2 = H(fb2,fa,τs,τt) + H(fb2,fb1,τs,τs, False) L = λLIS + (1 − λ)LES + γLkoleo // total loss with Koleo

regularization

L.backward(), update(gIS) // Note that gES and gIS share parameters, and the gradient from gES is accumulated onto the gradient of gIS.

gT.params = µ · gT.params + (1 − µ)·gIS.params // update

teacher with momentum EMA end

def ExtractElastic(cand_ids, idx, Dmax, Dh, Lmax, gIS): if idx == len(cand_ids) − 1 then

random.shuffle(cand_ids) idx = 0

i,j = cand_ids[idx] Di = Dmax − i · Dh Lj = Lmax − j gES = Net(gIS,Di,Lj) // sub-network extracted from gIS with

width Di and depth Lj idx+ = 1 return gES,idx

def H(s,t,τs,τt,centering = True):

t = t.detach() // stop gradient

- s = softmax(s/τs,dim = 1) if centering then

t = SK(t) // SK centering

- t = softmax(t/τt,dim = 1) return −(t · log(s)).sum(dim = −1)

### C Implementation Details

#### C.1 Multiple Projection Heads

Following [9,45,74], we employ 3-layer MLP with L2-normalized bottlenecks to serve as projection heads. To ensure effective training of each network within these elastic frameworks, we introduce multiple projection heads, each with a varying number of prototypes, positioned subsequent to the backbone network. Our MPH design is distinct from the multiple heads utilized in the ENT [48]. In the ENT, each head contains an identical number of prototypes and employs an averaging of cross-entropy loss which is weighted by the predictive entropies of each head, to ensemble the learning of each head. In contrast, our MPH design features heads with varying numbers of prototypes, which acts as multiple semantic spaces for representation. This design is intended to improve the distillation process between the intact network and the elastic network. In our experiments, we designate the output dimensions for these projection heads as: K1 = 8192,K2 = 16384,K3 = 32768,K4 = 65536.

#### C.2 KoLeo Regularizer

To mitigate the issue of feature collapse, we incorporate the KoLeo regularizer into our training process for the global views, as described in [45]. The regularizer’s loss function is expressed as:

B

1 B

KoLeo(z) = −

||zi − zj||,

log(dB,i), dB,i = min j̸=i

(13)

i=1

Zb1 ||Zb1||

Zb2 ||Zb2||

Lkoleo = KoLeo(

) + KoLeo(

),

where B is the batch size, and dB,i represents the Euclidean distance between the i-th feature zi and its nearest feature zj within the batch. The KoLeo loss is scaled with a modest loss weight γ set to 0.1.

#### C.3 Masked Patch Tokens Prediction

iBOT [74] has demonstrated the effectiveness of predicting the masked patch tokens of student networks according to the tokens of teacher network. We also incorporate this mechanism into the training of our POA SSL. In our experiments, we observed that early implementation of masked patch token prediction can result in unstable training. To address this issue, we delay the activation of the masked patch token prediction until the model has completed 30 epochs of training.

#### C.4 Probabilistic Sampling for Elastic Student

In POA, there are an array of candidate elastic networks that vary in width and depth, each offering a different level of diversity when compared to the intact student model. For example, considering the intact student structure as ViT-L characterized by a width of 1024 and depth of 24, an elastic network such as ViT-S, with a width of 384 and depth of 12, exhibits a higher degree of diversity relative to the intact student than an elastic network with dimensions closer to the intact student, such as a width of 960 and depth of 23. Intuitively, the elastic networks with greater diversity should be sampled more frequently to ensure sufficient training. To facilitate this, we implement a probabilistic sampling method influenced by the width and depth of the elastic networks in our experiments. For N + 1 available widths and M + 1 available depths, we calculate the sampling probability for the i-th width and j-th depth elastic network as follows:

((Sw − 1) · NN−i + 1) · ((Sh − 1) · MM−j + 1) N,M k=0,l=0((Sw − 1) · NN−k + 1) · ((Sh − 1) · MM−l + 1)

. (14)

pi,j =

It is important to note that when i = 0 and j = 0, the elastic network is at its smallest width and depth, and the sampling probability pi,j achieves the largest value. In our experiment, we set: Sw = Sd = 3.

#### C.5 Data Augmentation Setting

We adopt the same data augmentation techniques as DINOv2 [45], which include color jittering, Gaussian blur, solarization, flipping, and multi-crop strategies as described in [8]. The specific parameters for these augmentations are detailed in Table 6.

- Table 6: Hyper-parameters of different data augmentations. The parameters prob_g1, prob_g2, and prob_l refer to the activation probabilities for the first global crop, the second global crop, and the local crops, respectively. The parameter min_gcs represents the minimum global crop scale, while max_gcs indicates the maximum global crop scale. Similarly, min_lcs and max_lcs specifies the minimum and maximum local crop scale, respectively.

Color jittering Gaussian blur Solarization Multi-crop Flipping brightness: 0.4 radius_min: 0.1 thresh: 128 global crops: 2 direction:

contrast: 0.4 radius_max: 2.0 prob_g1: 0.0 local crops: 8 horizontal saturation: 0.2 prob_g1: 1.0 prob_g2: 0.2 global size: 224 prob: 0.5

hue: 0.1 prob_g2: 0.1 prob_l: 0.0 local size: 96 prob: 0.8 prob_l: 0.5 min_gcs: 0.32 max_gcs: 1.0 min_lcs: 0.05 max_lcs: 0.32

- C.6 Hyper-parameters of POA We provide following hyper-parameters setting in our method:

- – projection heads: bottleneck dim: 256 hidden layer dim: 2048
- – drop path rate: ViT: 0.2 Swin: 0.2 ResNet: 0.0
- – loss weights: λ = 0.8 γ = 0.1

- C.7 Optimizing Setting

In our training, we utilize the AdamW optimizer with parameters β1 = 0.9 and β2 = 0.999. The total training batch sizes for Vit, Swin, and ResNet are 1600, 2048, and 1280 respectively. We apply a learning rate decay from top to bottom across the network blocks, scaling down by a factor of 0.9. For transformer-based backbones, the patch embedding module’s learning rate is further reduced by a factor of 0.2. Within each projection head, we keep the parameters of the final layer fixed during the initial epoch of training. Additionally, to maintain stable training, the gradient is clipped at an L2 norm of 1.5 for all parameters. The momentum in EMA updating for teacher network is initialized as 0.992 and decay to 0.9999 with cosine schedule.

### D Experiments

#### D.1 k-NN and Linear Probing Evaluation on ImageNet-1K Dataset

We provide a more detailed comparison of k-NN and linear probing evaluations against existing methods in Table 7.

Table 7: Comprehensive comparison of k-NN and linear probing (LP) accuracy (%) on the ImageNet-1K dataset. "Param." indicates the quantity of parameters within the backbone network, measured in megabytes. "Epoch" refers to the adjusted number of effective training epochs, corrected for the number of views processed by the models as described in [74]. "∗" denotes our implementation based on official code. "†" denotes results reproduced using the official code.

Method Publication Arch. Param. Epoch k-NN LP SwAV [8] NeurIPS 20

RN-50 23 2400 65.7 75.3

RN-200 250 2000 73.9 79.6 BYOL [24] NeurIPS 20 RN-50 23 2000 64.8 74.4 MoCov3 [14] ICCV 21 RN-50 23 1600 - 74.6 DINO [9] ICCV 21 RN-50 23 3200 67.5 75.3 UniGrad [53] CVPR 22 RN-50 23 2400 - 75.5

RN-50 23 4000 - 77.1 RN-101 41 4000 - 78.7 RN-152 56 4000 - 79.3

ReLICv2 [58] arXiv 22

Univip [39] CVPR 22 RN-50 23 1200 - 74.2 Caco [60] arXiv 22 RN-50 23 3200 - 75.7 SMoG [46] ECCV 22 RN-50 23 1200 - 76.4 VICReg [5] ICLR 22 RN-50 23 2000 - 73.2 HCSC [25] CVPR 22 RN-50 23 800 66.6 73.3 SDMP-MoCov3 [47] CVPR 22 RN-50 23 600 - 73.5 SimSiam+GSG [35] NeurIPS 23 RN-50 23 400 58.4 69.4 BYOL+GSG [35] NeurIPS 23 RN-50 23 400 62.2 71.1 GroCo [50] ICCV 23 RN-50 23 400 64.8 71.3 MOKD [51] CVPR 23 RN-50 23 400 70.6 75.6 SCFS [52] CVPR 23 RN-50 23 3200 68.5 75.7 BYOL+LDReg [33] ICLR 24 RN-50 23 400 - 68.5 AUC-CL [49] ICLR 24 RN-50 23 1400 - 73.5 SimCLR+WNW [38] ICLR 24 RN-50 23 1600 - 66.3 SimSiam+WNW [38] ICLR 24 RN-50 23 1600 - 71.3

RN-50 23 0 73.4 76.9 POA(Ours) RN-101 41 0 75.7 79.1

RN-152 56 2400 76.4 79.9 MoBY [66] arXiv 21 Swin-T/W7 28 600 - 75.0 iBOT [74] ICLR 22

Swin-T/W7 28 1200 75.3 78.6 Swin-T/W14 28 1200 76.2 79.3

SMoG [46] ECCV 22 Swin-T/W7 28 1200 - 77.7

Swin-T/W7 28 1200 75.7 78.1

- Swin-S/W7 49 1200 77.7 79.5 Swin-B/W7 87 1200 78.9 80.4
- Swin-T/W14 28 1200 77.0 78.7 Swin-S/W14 49 1200 79.1 80.8 Swin-B/W14 87 1200 79.3 81.3

EsViT [36] ICLR 22

Swin-T/W7 28 1200 75.4 78.0

DINOv2∗ [45] TMLR 24

- Swin-S/W7 49 1200 76.1 79.8 Swin-B/W7 87 1200 76.9 80.9

- Swin-T/W7 28 0 77.5 78.9

#### POA(Ours) Swin-S/W7 49 0 79.3 81.3

Swin-B/W7 87 1200 79.6 82.0 SwAV [8] NeurIPS 20 ViT-S/16 21 2400 66.3 73.5 MoCov3 [14] ICCV 21

ViT-S/16 21 1200 - 73.4 ViT-B/16 85 1200 - 76.7 ViT-L/16 307 1200 - 77.6

ViT-S/16 21 3200 74.5 77.0

DINO [9] ICCV 21

- ViT-B/16 85 1600 76.1 78.2

iBOT [74] ICLR 22

ViT-S/16 21 3200 75.2 77.9

- ViT-B/16 85 1600 77.1 79.5 ViT-L/16 307 1200 78.0 81.0

ViT-S/16 21 600 - 73.8 ViT-B/16 85 600 - 77.2

SDMP-MoCov3 [47] CVPR 22

SDMP-DINO [47] CVPR 22 ViT-S/16 21 1200 - 76.4 Mugs [75] arXiv 22

ViT-S/16 21 3200 75.6 78.9 ViT-B/16 85 1600 78.0 80.6 ViT-L/16 307 1000 80.3 82.1

ViT-S/16 21 800 73.3 76.6 ViT-B/16 85 400 74.7 78.1

MSN [2] ECCV 22

ViT-B/8 85 300 75.7 80.3 MOKD [51] CVPR 23

ViT-S/16 21 800 73.1 76.3 ViT-B/16 85 400 76.0 78.4

ViT-B/16 85 600 - 72.9 ViT-L/16 307 600 - 77.5

I-JEPA [51] CVPR 23

SiameseIM [54] CVPR 23 ViT-B/16 85 1600 - 78.0 ENT-DINO [48] ICLR 23

ViT-S/16 21 3200 75.2 77.4 ViT-B/16 85 1600 77.1 79.1

ViT-S/16 21 800 75.2 77.4 ViT-B/16 85 400 77.2 78.9

ENT-MSN [48] ICLR 23

ViT-B/8 85 300 78.9 80.8 SimCLR+LDReg [33] ICLR 24 ViT-B/16 85 800 - 73.0 DINOv2† [45] TMLR 24

ViT-S/16 21 1200 72.2 73.1 ViT-B/16 85 1200 77.4 78.5 ViT-L/16 307 1200 82.0 83.3

AUC-CL [49] ICLR 24 ViT-S/16 21 1400 70.7 73.7 ViT-S/16 21 0 76.8 77.6 POA(Ours) ViT-B/16 85 0 80.9 81.7 ViT-L/16 307 1200 82.3 83.6

#### D.2 Fine-Tuning Evaluation

Due to the fine-tuning process adjusting the pretrained parameters of the backbone network, the differences between pretrained features are diminished. This may result in comparisons that may not fully reflect the distinct qualities of leaned representation in each method. Consequently, only a handful of studies report this metric. In our experiments, we conduct fine-tuning on the ImageNet1K dataset and draw comparisons to self-supervised methods utilizing ViT backbone. We adhered to the fine-tuning methodology delineated in [4, 74], which incorporates layer-wise learning rate decay, weight decay, and the AdamW optimizer. The training durations for ViT and Swin variants are set at 200, 100, and 50 epochs for the large, base, and small models, respectively. Due to differences in convergence between convalutional network and transformer, all ResNet variants (ResNet-152, ResNet-101, and ResNet-50) undergo a uniform training period of 100 epochs. We apply a layer-wise decay rate of 0.55 for ViT-S, Swin-T, and ResNet-50; a decay rate of 0.4 for ViT-B, Swin-S, and ResNet-101; and a

decay rate of 0.6 for ViT-L, Swin-B, and ResNet-152. The initial learning rates for fine-tuning are configured as follows: 0.002 for ViT-S, Swin-T, and ResNet50; 0.0007 for ViT-B, Swin-S, and ResNet-101; and 0.0018 for ViT-L, Swin-B, and ResNet-152.

As is shown in Table. 8, our POA achieves a SOTA accuracy of 85.3% on the ViT-L/16 backbone, and it demonstrates comparable accuracy when utilizing ViT-S/16 and ViT-B/16 backbones. We also report the fine-tuning results for the Swin and ResNet backbones in Table 10.

Table 9: Results of semi-supervised learning on ImageNet-1K. The 1% and 10% indicate the fractions of labeled data used. SD denotes self-distillation.

Table 8: Fine-tuning results on ImageNet-1K dataset.

Method Arch. Epo. Acc.

DINO ViT-S/16 3200 82.0 iBOT ViT-S/16 3200 82.3 POA ViT-S/16 0 82.1

Method Arch. 1%. 10%.

SimCLRv2 RN50 57.9 68.1 BYOL RN50 53.2 68.8 SwAV RN50 53.9 70.2

BEiT ViT-B/16 800 83.4 DINO ViT-B/16 1600 83.6 iBOT ViT-B/16 1600 84.0 POA ViT-B/16 0 83.9

SimCLRv2

RN50 60.0 70.5

+SD POA RN50 61.8 73.1

iBOT ViT-L/16 1200 84.8 BEiT ViT-L/16 800 85.2 POA ViT-L/16 1200 85.3

DINO ViT-S/16 60.3 74.3 iBOT ViT-S/16 61.9 75.1 POA ViT-S/16 68.2 75.9

- Table 10: Fine-tuning results of Swin and ResNet backbone on ImageNet-1K dataset.

Method Arac. Epo. Acc POA ResNet-50 0 77.8 POA ResNet-101 0 80.0 POA ResNet-152 2400 81.1

(a) ResNet backbone.

Method Arac. Epo. Acc POA Swin-T 0 81.0 POA Swin-S 0 82.9 POA Swin-B 1200 83.7

(b) Swin backbone.

#### D.3 Semi-Supervised Learning Evaluation

For semi-supervised learning, we concentrate our comparison on methods that adopt the unsupervised pre-training followed by supervised fine-tuning paradigm with partial labeled data. As shown in Table 9, our method significantly outperforms iBOT when using only 1% of labeled data, with an improvement of 6.3%. These results demonstrate our method’s superior label efficiency. We attribute this performance enhancement primarily to the distillation loss LES2 , which facilitates knowledge transfer from the intact model to its elastic counterpart. This effect mirrors the improvement observed in SimCLRv2, where self-distillation from a larger model contributes to performance gains.

#### D.4 Unsupervised Learning Evaluation

For evaluating the pre-trained model on unsupervised learning, we employ standard metrics such as accuracy (ACC), adjusted rand index (ARI), normalized mutual information (NMI), and the Fowlkes-Mallows index (FMI), following [74]. We benchmark our POA with a ResNet-50 backbone against established methods like SimCLRv2 [11], Self-label [1], InfoMin [57], and SCAN [22]. Additionally, we compare POA with a ViT-S/16 backbone to DINO and iBOT. According to the results presented in Table. 11, our POA method attains accuracies of 61.8% with ViT-S/16 and 55.7% with ResNet-50, respectively. These results indicate that the POA approach in self-supervised learning enables models to learn stronger visual semantical representation.

- Table 11: Unsupervised learning on ImageNet-1K dataset. "†" denotes k-means clustering on frozen features extracted by backbones.

Method Arch. ACC ARI NMI FMI Self-label† ResNet-50 30.5 16.2 75.4 -

InfoMin† ResNet-50 33.2 14.7 68.8 SCAN ResNet-50 39.9 27.5 72.0 -

POA† ResNet-50 55.7 38.2 79.9 38.9 DINO ViT-S/16 41.4 29.8 76.8 32.8 iBOT ViT-S/16 43.4 32.8 78.6 35.6

POA† ViT-S/16 61.8 47.7 82.5 47.9

- Table 12: Transfer learning experiments by fine-tuning models pre-trained on various datasets. The Top-1 accuracy for the ViT-S/16 is presented on the left, and for the ViT-B/16 on the right.

Method Cif10 Cif100 iNa18 iNa19 Flwrs Cars

Method Cif10 Cif100 iNa18 iNa19 Flwrs Cars

BEiT 98.6 87.4 68.5 76.5 96.4 92.1 DINO 99.0 90.5 72.0 78.2 98.5 93.0 iBOT 99.1 90.7 73.7 78.5 98.6 94.0 POA 99.1 90.7 74.2 79.1 98.4 94.2

BEiT 99.0 90.1 72.3 79.2 98.0 94.2 DINO 99.1 91.7 72.6 78.6 98.8 93.0 iBOT 99.2 92.2 74.6 79.6 98.9 94.3 POA 99.4 92.6 76.2 81.7 98.8 94.6

#### D.5 Transfer Learning

We evaluate transfer learning by pre-training models on ImageNet-1K and subsequently fine-tuning them on a variety of smaller datasets, adhering to the protocol established in [20]. The results are detailed in Table 12. Our method achieves SOTA transfer performance compared to other self-supervised learning (SSL) approaches, with the exception of the Flowers dataset. Notably, we observe a more pronounced performance improvement over iBOT on larger datasets such as iNaturalist18 and iNaturalist19. This suggests that the results have not yet reached their peak, thereby providing a more effective measure for evaluating the quality of pre-trained features.

#### D.6 k-NN Accuracies of Elastic Networks

Elastic Networks of ViT We present the k-NN evaluation accuracy of each elastic network derived from the pre-trained ViT trained by POA, as detailed in Table 13, here Li and Di are the depths and widths of each elastic network, repectively.

Table 13: k-NN accuracy of elastic networks derived from pretrained ViT-L/16.

Li/Di 384 448 512 576 640 704 768 832 896 960 1024

- 12 76.78 78.42 79.17 79.78 80.27 80.68 80.86 81.02 81.10 81.14 81.09
- 13 76.84 78.37 79.17 79.85 80.83 80.66 80.93 81.23 81.24 81.12 81.15
- 14 77.59 78.95 79.69 80.26 80.69 81.04 81.25 81.36 81.52 81.60 81.59
- 15 77.89 79.16 79.86 80.35 80.75 81.16 81.40 81.62 81.62 81.63 81.58
- 16 78.15 79.30 80.13 80.75 81.06 81.36 81.55 81.78 81.92 81.77 81.85
- 17 78.34 79.65 80.26 80.76 81.11 81.54 81.63 81.76 82.02 81.92 81.88 17 78.58 79.75 80.43 80.92 81.28 81.69 81.72 81.99 82.04 82.02 82.05

- 19 78.62 79.90 80.52 80.98 81.32 81.57 81.83 82.13 82.09 82.14 82.12
- 20 78.93 79.97 80.61 81.17 81.45 81.79 81.98 82.17 82.22 82.15 82.17
- 21 78.99 80.21 80.73 81.31 81.49 81.86 82.05 82.18 82.32 82.22 82.20
- 22 79.25 80.25 80.89 81.36 81.63 81.87 82.10 82.31 82.34 82.36 82.41
- 23 79.37 80.30 80.89 81.35 81.73 81.93 82.19 82.37 82.41 82.41 82.39
- 24 79.33 80.28 80.90 81.33 81.65 81.90 82.15 82.35 82.42 82.28 82.27

Elastic Networks of Swin For the Swin Transformer architecture, we designate Swin-T as the smallest elastic network and Swin-B as the largest. We explore elastic widths of 96, 112, and 128, with elastic depths varying from 12 to 24. The k-NN accuracies of total 39 elastic networks configuration are presented in Table. 14.

Table 14: k-NN accuracy of elastic networks derived from pretrained Swin-B.

Di/Li 12 13 14 15 16 17 18 19 20 21 22 23 24

96 77.48 77.80 78.07 78.18 78.17 78.67 78.81 78.83 79.05 79.12 79.20 79.34 79.31 112 77.84 78.16 78.33 78.47 78.47 78.94 79.03 79.09 79.22 79.22 79.32 79.43 79.43 128 77.90 78.23 78.47 78.52 78.52 79.00 79.07 79.19 79.45 79.45 79.53 79.52 79.63

Elastic Network of ResNet In the case of the ResNet architecture, we designate ResNet-50 as the smallest and ResNet-152 with wider middle layer in each building block as the largest elastic network configurations. It yields a total number of 465 distinct ResNet sub-networks with the combination of 3 widths and 155 depths configurations. For the sake of clarity, we present a subset of the k-NN accuracies of these elastic networks in Table 15. Here, N2 and N3 refer to the count of bottleneck building blocks in the second and third stages, respectively, while W denotes the bottleneck dimension of middle layer in building block at the first stage.

Table 15: k-NN accuracy of elastic networks derived from pretrained ResNet-152.

W/N2 − N3 4-6 4-8 4-10 4-12 4-14 4-16 4-16 4-18

64 73.44 74.11 74.32 74.52 74.92 75.17 75.19 75.49 96 74.41 75.00 75.18 75.63 75.95 76.01 76.11 76.42

128 74.73 75.43 75.43 75.81 75.91 76.20 76.44 76.47 W/N2 − N3 4-20 4-24 8-26 8-28 8-30 8-32 8-34 8-36

64 75.49 75.77 76.05 76.20 76.20 76.33 76.31 76.38 96 76.34 76.63 76.91 76.95 77.07 77.13 77.30 77.14

128 76.53 76.90 77.13 77.34 77.31 77.54 77.59 77.72

#### D.7 Robustness Evaluation.

Robustness to Occlusion and Shuffling. We evaluate the pre-trained model’s

robustness to occlusion and alterations in spatial structure by applying masking and shuffling to the input image patches. Detailed results for various occlusion ratios are depicted in Figure 5a. Additionally, we present the effects of different shuffling grid sizes in Figure 5b.

OFA ViT-S/16

OFA ViT-S/16

80

80

70

60

60

Accuracy(%)

Accuracy(%)

50

40

40

30

20

20

10

0

0

0 20 40 60 80 100

1 4 8 16 32 64 196256

Occlusion Ratio(%)

Shuffle Grid Size

(a) Robustness to occlusion with different ratio.

(b) Robustness to shuffling with varying grid sizes.

Fig. 5: Robustness to Occlusion and Shuffling.

#### D.8 Additional Ablation Studies

Influence of Input Dimension Scaling Factor αi. To adapt the reduction of the input dimension, we apply a scaling factor αi to weight parameters. We conduct comparative experiments to assess the impact of employing this scaling factor, with results presented in Table 16. The results indicate that the scaling factor αi enhances the performance of an elastic network, particularly in cases where there is a significant reduction in width compared to the intact network, such as with ViT-S.

Influence of Loss Weight λ. Within our POA framework, the parameter λ regulates the balance between the loss contributions from the intact student and the elastic student. We assessed the performance impact of varying λ during

Table 16: Importance of Scaling Factor in POA SSL pre-training.

k-NN LP ViT-S ViT-B ViT-L ViT-S ViT-B ViT-L

Scaling Factor

✓ 76.8 80.9 82.3 77.6 81.7 83.6 75.3 80.8 82.3 75.9 81.7 83.6

pre-training. The results presented in Table 17 suggest that a larger value of λ, representing a greater loss weight for intact branch, enhances the performance of larger models like ViT-L. However, this same increase in λ adversely affects the performance of the extracted smaller sub-networks such as ViT-S and ViT-B. Conversely, a smaller λ value improves the performance of these smaller subnetworks while potentially diminishing the effectiveness of larger models. To achieve a more balanced outcome, we have chosen λ = 0.8 for our OPA approach.

Table 17: Influence of Loss Weight λ in POA SSL pre-training.

k-NN LP ViT-S ViT-B ViT-L ViT-S ViT-B ViT-L

λ

- 0.6 77.4 81.0 82.0 78.3 81.9 83.1
- 0.7 77.0 80.9 82.1 77.9 81.8 83.4
- 0.8 76.8 80.9 82.3 77.6 81.7 83.6
- 0.9 75.3 80.0 82.6 76.5 81.0 84.0

Influence of Probabilistic Sampling for Elastic Student. We provide the ablation study about probabilistic sampling for elastic student in Table. 18. The result confirms our intuitive assumption that elastic networks with greater diversity should be sampled more frequently.

Table 18: Influence of sampling for elastic student in POA SSL pre-training.

k-NN LP ViT-S ViT-B ViT-L ViT-S ViT-B ViT-L

Sw = Sd

- 1 75.8 80.6 82.3 76.5 81.4 83.6
- 2 76.5 80.7 82.3 77.4 81.6 83.6
- 3 76.8 80.9 82.3 77.6 81.7 83.6

Influence of Number of Elastic Students. We investigate the impact of varying the number of candidate elastic networks in our POA. We manipulate the number of candidates by adjusting the sampling intervals of network widths and depths. Except for the number of candidate elastic networks, all hyperparameters and training settings remain constant. The comparative results are presented in Table 19. From these results, we observe that the increase of the sampling interval, which effectively decreases the number of candidate networks, improves the k-NN accuracy for the derived ViT-S model. The primary reason is that with a constant number of iterations, a reduction in the total count of networks results in a higher proportion of smaller networks being sampled. This

leads to more training iteration for these networks. However, this adjustment appears to have a negligible effect on the performance of ViT-L models, due to the existing of the intact student branch which is trained at each iteration.

Table 19: Influence of number of candidate elastic students in POA SSL pre-training.

Number of Candidates Interval of Elastic Sampling k-NN (#Widths × #Depths) (Width/Depth) ViT-S ViT-B ViT-L

143(11 × 13) 64/1 76.8 80.9 82.3 42(6 × 7) 128/2 77.3 80.9 82.3 20(4 × 5) 192/3 77.7 80.8 82.4 16(4 × 4) 192/4 77.8 81.1 82.4

#### D.9 Alternative Designs to Elastic Pre-training

We provide a detailed illustration of the three variants mentioned in Sec.4.4 of our paper. In the first variant, illustrated in Figure 6a, the intact student is removed from the POA framework and the teacher is adapted to be elastic, aligning with the architecture of the elastic student. The second variant, which also discards the intact student from POA, is depicted in Figure 6b. The third variant introduces an additional elastic teacher branch that shares the architecture of the elastic student, facilitating cross-view distillation, shown in Figure 6c.

#### D.10 Convergence Comparison with Single Pre-training Method

By integrating same-view and cross-view distillation, POA achieves faster convergence speed, particularly for smaller-sized sub-networks. For instance, in the case of ViT-S, many existing self-supervised learning methods [9,48,74,75] necessitate a substantial number of effective training epochs (3200 epochs) reach good performance. In contrast, the ViT-S model extracted from our POA framework, pre-trained for just 1200 epochs, outperforms those methods trained separately for 3200 epochs. Figure 7 illustrates the k-NN evaluation accuracy progression throughout the 1200 effective epochs of training. It is evident that POA consistently delivers superior performance at each stage of the training process.

#### D.11 Computational Resources Required

Table. 20 provides a detailed account of the computational resources required for training with a ViT backbone on 4 machines, each equipped with 8 A100 GPUs. We compare the time and GPU memory demands of our method to those of DINOv2, which incorporates self-supervised knowledge distillation [21] and yields superior performance compared to training the models independently. Notably, our approach can generate numerous elastic networks beyond the three primary structures: small, base, and large.

|Elastic Teacher<br><br>| |
|---|---|
| | |

|Softmax Centering<br><br>+|
|---|

[Figure 5]

| |Head|
|---|---|
| | |

Softmax Centering

𝑝

𝑍

[Figure 6]

sg

ℒ : −𝑝 log 𝑝

𝑥 EMA

EMA

|Elastic Student| |
|---|---|
| | |

|Softmax| |
|---|---|
| | |

| |Head| |
|---|---|---|
| | | |
| | | |

[Figure 7]

Softmax

𝑥 𝑍

𝑝

𝑥

- (a) POA-V1: This variant of POA features both an elastic teacher and an elastic student, streamlining the architecture by ensuring both components are adaptable in size.

[Figure 8]

[Figure 9]

[Figure 10]

|Elastic Student| |
|---|---|
| | |

𝑥 EMA

𝑥

𝑥 𝑍

𝑍

|Intact Teacher<br><br>| |
|---|---|
| | |

EMA

|Softmax Centering<br><br>+|
|---|

Softmax Centering

|Softmax| |
|---|---|
| | |

Softmax

| |Head| |
|---|---|---|
| | | |
| | | |

𝑝

𝑝

ℒ : −𝑝 log 𝑝

sg

|Head|
|---|

- (b) POA-V2: In this variant, POA includes an intact, intact teacher paired with an elastic student, allowing the smaller student model to learn from the larger, fully-trained teacher.

[Figure 11]

[Figure 12]

[Figure 13]

|Intact Teacher| |
|---|---|
| | |

|Intact Student| |
|---|---|
| | |

EMA

Share

𝑥

𝑥

𝑥

𝑍

𝑍

|Elastic Student| |
|---|---|
| | |

EMA

|Softmax Centering<br><br>+|
|---|

Softmax Centering

|Softmax|
|---|

Softmax

| |Head| |
|---|---|---|
| | | |
| | | |

𝑝

𝑝

𝑝

ℒ : −𝑝 log 𝑝

ℒ : −𝑝 log 𝑝

ℒ : −𝑝 log 𝑝

|Elastic Teacher| |
|---|---|
| | |

Share

𝑍

𝑍

| |Head| |
|---|---|---|
| | | |
| | | |

𝑝

ℒ : −𝑝 log 𝑝

sg

- (c) POA-V3: This variant of POA introduces an extra elastic teacher alongside the standard configuration, providing another potential for cross-view distillation within the framework.

Fig. 6: Three alternative variants of POA.

### E Visualization

#### E.1 Visualization of Self-attention Map

We visualize the self-attention maps generated by the ViT-S/16 model, which is pre-trained using DINOv2 and our POA. For the visualizations, we select the class token as the query and represent attention maps from different heads of the final layer using distinct colors, as depicted in Figure 8. The results indicate

2)A Vi7-6/16

2)A Vi7-%/16

DI12v2 Vi7-6/16

76

DI12v2 Vi7-%/16

80

74

78

N-11AFFurDFy(%)

N-11AFFurDFy(%)

72

76

70

74

68

72

66

64

70

62

68

200 400 600 800 1000 1200

200 400 600 800 1000 1200

(IIeFtive 7rDining (poFh

(IIeFtive 7rDining (SoFh

Fig. 7: Comparison of k-NN accuracy during training.

Table 20: Comparison of computational resources required.

Method Arch. Epoch Mem. Batch Size k-NN GPU hours

DINOv2 ViT-L 1200 41G 2048 82.0 1152 DINOv2 ViT-L→ViT-B 1200 46G 4096 79.7 1024 DINOv2 ViT-L→ViT-S 1200 35G 4096 75.5 928

Total 3104 POA ViT-L 1200 77G 1600 82.3 2752 POA ViT-B 0 - - 80.9 0 POA ViT-S 0 - - 76.8 0

that POA’s self-attention focuses more concentratedly on the foreground objects compared to DINOv2. For instance, in Figure 8a, POA distinctly highlights the regions of interest associated with foreground elements (such as the human, fish, trumpet, and snake). In contrast, the DINOv2 generates attention maps exhibit a more dispersed focus, often including areas of the background. In Figure 9, we showcase more self-attention map visualizations, comparing the outputs from multiple heads in the final layer of our method with those from DINOv2.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

(a) Self-attention map of POA. (b) Self-attention map of DINOv2.

- Fig. 8: Visualization of Self-Attention Maps: we display the self-attention maps from multiple heads using distinct colors for differentiation. For both POA and DINOv2, we set the visualization threshold to 0.6, retaining top 60% of the attention mass.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

- (a) DIONv2

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

- (b) POA

###### Fig. 9: Visualization for self-attention map from multiple heads of the last layer in ViT-S/16. The results indicate that POA concentrates its attention more accurately on foreground objects than DINOv2 does.

#### E.2 Visualization of Correspondence

We conduct a correspondence task that involves matching overlapping patches from two different augmentations of the same image or patches from two distinct images labeled as the same class. We present visualizations of the these patches with the highest self-attention scores obtained from a ViT-S/16 model pre-trained by POA, averaging the scores across multiple heads in the final layer. Figure 10 illustrates a selection of these image pair samples. The results indicate that POA excels in identifying correspondences both within varied views of a single image and across different segments of separate instances within the same class.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

- (a) Correspondence between two different images of the same category

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- (b) Correspondence between two different views of the same image

- Fig. 10: Visualization of Correspondences: The top panel displays pairs of images sampled from two different views of a single image. The bottom panel shows pairs of images taken from two distinct images belonging to the same class.

- E.3 Visualization of Pattern Layout for Class Token
- Fig. 11: Visualization for pattern layout of class token of POA. We indicate that the prototypes effectively cluster images based on similar semantic features, even when they may span different categories.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Figure 11 presents a visualization of the pattern layout associated with the class token in ViT-S/16 trained by POA. We display images that have the top64 similarity scores with each prototype in ImageNet validation set, arranged in an 8×8 grid. The results indicates the high-quality semantic structure achieved through the self-distillation process applied to cross-view images within our POA framework. Furthermore, we observe that the prototypes effectively cluster images based on similar semantic features, even when they may span different categories. For instance, in the top-left image of the grid, while the primary category is ’parachute’, there are also images of related but distinct categories such as ’radio reflector’ and ’umbrella’, which are outlined in red boxes within these prototypes. Similarly, in the top-right image, the main category featured is ’odometer’, but it also includes images of semantically similar objects like ’clock’.

### References

- 1. Asano, Y.M., Rupprecht, C., Vedaldi, A.: Self-labelling via simultaneous clustering and representation learning. ArXiv abs/1911.05371 (2019)
- 2. Assran, M., Caron, M., Misra, I., Bojanowski, P., Bordes, F., Vincent, P., Joulin, A., Rabbat, M.G., Ballas, N.: Masked siamese networks for label-efficient learning. In: European Conference on Computer Vision (2022)
- 3. Ba, J., Kiros, J.R., Hinton, G.E.: Layer normalization. ArXiv abs/1607.06450

(2016)

- 4. Bao, H., Dong, L., Piao, S., Wei, F.: BEit: BERT pre-training of image transformers. In: International Conference on Learning Representations (ICLR) (2022)
- 5. Bardes, A., Ponce, J., LeCun, Y.: Vicreg: Variance-invariance-covariance regularization for self-supervised learning. ArXiv abs/2105.04906 (2021)
- 6. Cai, H., Gan, C., Wang, T., Zhang, Z., Han, S.: Once for all: Train one network and specialize it for efficient deployment. In: International Conference on Learning Representations (2020)
- 7. Cai, Z., Vasconcelos, N.: Cascade r-cnn: High quality object detection and instance segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 1483–1498 (2019)
- 8. Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A.: Unsupervised learning of visual features by contrasting cluster assignments. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 9912–9924. Curran Associates, Inc. (2020)
- 9. Caron, M., Touvron, H., Misra, I., J’egou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9630–9640 (2021)
- 10. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.E.: A simple framework for contrastive learning of visual representations. International Conference on Machine Learning (2021)
- 11. Chen, T., Kornblith, S., Swersky, K., Norouzi, M., Hinton, G.E.: Big self-supervised models are strong semi-supervised learners. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 22243–22255. Curran Associates, Inc. (2020)
- 12. Chen, T., Zhai, X., Ritter, M., Lucic, M., Houlsby, N.: Self-supervised gans via auxiliary rotation loss. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 12146–12155 (2018)
- 13. Chen, X., Fan, H., Girshick, R., He, K.: Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297 (2020)
- 14. Chen, X., Xie, S., He, K.: An empirical study of training self-supervised vision transformers. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9620–9629 (2021)
- 15. Chen, Y., Liu, Y., Jiang, D., Zhang, X., Dai, W., Xiong, H., Tian, Q.: Sdae: Self-distillated masked autoencoder. In: European Conference on Computer Vision

- (2022)

16. Cheng, X., Chen, J., Wang, R.: Modified dual attention triplet-supervised hashing network for image retrieval. Signal, Image and Video Processing 18, 1939–1948

- (2023)

- 17. Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. In: Neural Information Processing Systems (2013)

- 18. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A LargeScale Hierarchical Image Database. In: CVPR 2009 (2009)
- 19. Dong, C., Loy, C.C., He, K., Tang, X.: Image super-resolution using deep convolutional networks. IEEE Transactions on Pattern Analysis and Machine Intelligence 38, 295–307 (2014)
- 20. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations (ICLR) (2021)
- 21. Fang, Z., Wang, J., Wang, L., Zhang, L., Yang, Y., Liu, Z.: Seed: Self-supervised distillation for visual representation. International Conference on Learning Representations (2021)
- 22. Gansbeke, W.V., Vandenhende, S., Georgoulis, S., Proesmans, M., Gool, L.V.: Learning to classify images without labels. ArXiv abs/2005.12320 (2020)
- 23. Gao, Y., Zhuang, J.X., Lin, S., Cheng, H., Sun, X., Li, K., Shen, C.: Disco: Remedying self-supervised learning on lightweight models with distilled contrastive learning. In: European Conference on Computer Vision (ECCV) (2022)
- 24. Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al.: Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems 33, 21271–21284 (2020)
- 25. Guo, Y., Xu, M., Li, J., Ni, B., Zhu, X., Sun, Z., Xu, Y.: Hcsc: Hierarchical contrastive selective coding. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 9696–9705 (2022)
- 26. He, K., Chen, X., Xie, S., Li, Y., Doll’ar, P., Girshick, R.B.: Masked autoencoders are scalable vision learners. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 15979–15988 (2021)
- 27. He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.B.: Momentum contrast for unsupervised visual representation learning. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 9726–9735 (2019)
- 28. He, K., Gkioxari, G., Dollár, P., Girshick, R.B.: Mask r-cnn. 2017 IEEE International Conference on Computer Vision (ICCV) pp. 2980–2988 (2017)
- 29. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) pp. 770–778 (2015)
- 30. Hinton, G.E., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. ArXiv abs/1503.02531 (2015)
- 31. Hou, L., Shen, H., Cao, Q., Cheng, X.: Self-supervised gans with label augmentation. ArXiv abs/2106.08601 (2021)
- 32. Hu, C., Li, X., Liu, D., Wu, H., Chen, X., Wang, J., Liu, X.: Teacher-student architecture for knowledge distillation: A survey. ArXiv abs/2308.04268 (2023)
- 33. Huang, H., Campello, R.J.G.B., Erfani, S.M., Ma, X., Houle, M.E., Bailey, J.: LDReg: Local dimensionality regularized self-supervised learning. In: The Twelfth International Conference on Learning Representations (2024), https: //openreview.net/forum?id=oZyAqjAjJW
- 34. Huang, Z., Jin, X., Lu, C., Hou, Q., Cheng, M.M., Fu, D., Shen, X., Feng, J.: Contrastive masked autoencoders are stronger vision learners. IEEE transactions on pattern analysis and machine intelligence PP (2022)
- 35. Lee, B., Lee, S.: Implicit contrastive representation learning with guided stopgradient. In: Thirty-seventh Conference on Neural Information Processing Systems

(2023)

- 36. Li, C., Yang, J., Zhang, P., Gao, M., Xiao, B., Dai, X., Yuan, L., Gao, J.: Efficient self-supervised vision transformers for representation learning. International Conference on Learning Representations (ICLR) (2022)
- 37. Li, J., Wang, Y., ZHANG, X., Chen, Y., Jiang, D., Dai, W., Li, C., Xiong, H., Tian, Q.: Progressively compressed auto-encoder for self-supervised representation learning. In: The Eleventh International Conference on Learning Representations

(2023)

- 38. Li, S., Wu, C., Li, A., Wang, Y., Tang, X., Yuan, G.: Waxing-and-waning: a generic similarity-based framework for efficient self-supervised learning. In: The Twelfth International Conference on Learning Representations (2024), https: //openreview.net/forum?id=TilcG5C8bN
- 39. Li, Z., Zhu, Y., Yang, F., Li, W., Zhao, C., Chen, Y., Chen, Z., Xie, J., Wu, L., Zhao, R., Tang, M., Wang, J.: Univip: A unified framework for self-supervised visual pre-training. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 14607–14616 (2022)
- 40. Lin, J., Rao, Y., Lu, J., Zhou, J.: Runtime neural pruning. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017)
- 41. Lin, T.Y., Maire, M., Belongie, S.J., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European Conference on Computer Vision (2014)
- 42. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9992–10002 (2021)
- 43. Loshchilov, I., Hutter, F.: Fixing weight decay regularization in adam. ArXiv abs/1711.05101 (2017)
- 44. van den Oord, A., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. ArXiv abs/1807.03748 (2018)
- 45. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024)
- 46. Pang, B., Zhang, Y., Li, Y., Cai, J., Lu, C.: Unsupervised visual representation learning by synchronous momentum grouping. In: European Conference on Computer Vision (2022)
- 47. Ren, S., Wang, H., Gao, Z., He, S., Yuille, A.L., Zhou, Y., Xie, C.: A simple data mixing prior for improving self-supervised learning. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 14575–14584 (2022)
- 48. Ruan, Y., Singh, S., Morningstar, W.R., Alemi, A.A., Ioffe, S., Fischer, I., Dillon, J.V.: Weighted ensemble self-supervised learning. In: The Eleventh International Conference on Learning Representations (2023)
- 49. Sharma, R., Ji, K., zhiqiang xu, Chen, C.: AUC-CL: A batchsize-robust framework for self-supervised contrastive representation learning. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/ forum?id=YgMdDQB09U
- 50. Shvetsova, N., Petersen, F., Kukleva, A., Schiele, B., Kuehne, H.: Learning by sorting: Self-supervised learning with group ordering constraints. In: Proceedings of

- the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 16453– 16463 (October 2023)
- 51. Song, K., Xie, J., Zhang, S., Luo, Z.: Multi-mode online knowledge distillation for self-supervised visual representation learning. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 11848–11857 (2023)
- 52. Song, K., Zhang, S., Luo, Z., Wang, T., Xie, J.: Semantics-consistent feature search for self-supervised visual representation learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 16099–16108 (October 2023)
- 53. Tao, C., Wang, H., Zhu, X., Dong, J., Song, S., Huang, G., Dai, J.: Exploring the equivalence of siamese self-supervised learning via a unified gradient framework. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 14411–14420 (2021)
- 54. Tao, C., Zhu, X., Huang, G., Qiao, Y., Wang, X., Dai, J.: Siamese image modeling for self-supervised vision representation learning. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 2132–2141 (2022)
- 55. Team, G.G.: Introducing gemini: our largest and most capable ai model. Google

(2023)

- 56. Tian, K., Jiang, Y., qishuai diao, Lin, C., Wang, L., Yuan, Z.: Designing BERT for convolutional networks: Sparse and hierarchical masked modeling. In: The Eleventh International Conference on Learning Representations (2023)
- 57. Tian, Y., Sun, C., Poole, B., Krishnan, D., Schmid, C., Isola, P.: What makes for good views for contrastive learning. ArXiv abs/2005.10243 (2020)
- 58. Tomasev, N., Bica, I., McWilliams, B., Buesing, L., Pascanu, R., Blundell, C., Mitrovic, J.: Pushing the limits of self-supervised resnets: Can we outperform supervised learning without labels on imagenet? arXiv preprint arXiv:2201.05119

(2022)

- 59. Touvron, H., Cord, M., Oquab, M., Bojanowski, P., Verbeek, J., Jégou, H.: Cotraining 2l submodels for visual recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 11701– 11710 (June 2023)
- 60. Wang, X., Huang, Y., Zeng, D., Qi, G.J.: Caco: Both positive and negative samples are directly learnable via cooperative-adversarial contrastive learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 10718–10730 (2022)
- 61. Wang, X., Yu, F., Dou, Z.Y., Gonzalez, J.: Skipnet: Learning dynamic routing in convolutional networks. ArXiv abs/1711.09485 (2017)
- 62. Wang, X., Zhang, R., Shen, C., Kong, T., Li, L.: Dense contrastive learning for selfsupervised visual pre-training. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 3023–3032 (2020)
- 63. Wortsman, M., Ilharco, G., Gadre, S.Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A.S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., Schmidt, L.: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In: International Conference on Machine Learning (ICML) (2022)
- 64. Wu, H., Xu, J., Wang, J., Long, M.: Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In: Neural Information Processing Systems (2021)
- 65. Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. ArXiv abs/1807.10221 (2018)
- 66. Xie, Z., Lin, Y., Yao, Z., Zhang, Z., Dai, Q., Cao, Y., Hu, H.: Self-supervised learning with swin transformers. ArXiv abs/2105.04553 (2021)

- 67. Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., Hu, H.: Simmim: a simple framework for masked image modeling. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 9643–9653 (2021)
- 68. Yan, B., Jiang, Y., Sun, P., Wang, D., Yuan, Z., Luo, P., Lu, H.: Towards grand unification of object tracking. In: European Conference on Computer Vision (2022)
- 69. Yan, C., Chang, X., Li, Z., Yao, L., Luo, M., Zheng, Q.: Masked distillation advances self-supervised transformer architecture search. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/ forum?id=LUpC8KTvdV
- 70. Yu, F., Huang, K., Wang, M., Cheng, Y., Chu, W., Cui, L.: Width & depth pruning for vision transformers. In: AAAI Conference on Artificial Intelligence (2022)
- 71. Yu, J., Li, X., Koh, J.Y., Zhang, H., Pang, R., Qin, J., Ku, A., Xu, Y., Baldridge, J., Wu, Y.: Vector-quantized image modeling with improved VQGAN. In: International Conference on Learning Representations (2022)
- 72. Zhang, Y., Zhong, Q., Ma, L., Xie, D., Pu, S.: Learning incremental triplet margin for person re-identification. AAAI Conference on Artificial Intelligence (2019)
- 73. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ade20k dataset. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) pp. 5122–5130 (2017)
- 74. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T.: ibot: Image bert pre-training with online tokenizer. International Conference on Learning Representations (ICLR) (2022)
- 75. Zhou, P., Zhou, Y., Si, C., Yu, W., Ng, T.K., Yan, S.: Mugs: A multi-granular self-supervised learning framework. ArXiv abs/2203.14415 (2022)

