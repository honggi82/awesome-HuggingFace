# arXiv:2601.01425v1[cs.CV]4Jan2026

[Figure 1]

[Figure 2]

## DreamID-V: Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer

#### Xu Guo1,⋆, Fulong Ye2,⋆, Xinghui Li2,⋆, Pengqi Tu2, Pengze Zhang2, Qichao Sun2, Songtao Zhao2,†, Xiangwang Hou1,†, Qian He2

1Tsinghua University, 2Intelligent Creation Lab, ByteDance ⋆Equal contribution, †Corresponding author

#### Abstract

Video Face Swapping (VFS) requires seamlessly injecting a source identity into a target video while meticulously preserving the original pose, expression, lighting, background, and dynamic information. Existing methods struggle to maintain identity similarity and attribute preservation while preserving temporal consistency. To address the challenge, we propose a comprehensive framework to seamlessly transfer the superiority of Image Face Swapping (IFS) to the video domain. We first introduce a novel data pipeline SyncID-Pipe that pre-trains an Identity-Anchored Video Synthesizer and combines it with IFS models to construct bidirectional ID quadruplets for explicit supervision. Building upon paired data, we propose the first Diffusion Transformer-based framework DreamID-V, employing a core Modality-Aware Conditioning module to discriminatively inject multi-model conditions. Meanwhile, we propose a Synthetic-to-Real Curriculum mechanism and an Identity-Coherence Reinforcement Learning strategy to enhance visual realism and identity consistency under challenging scenarios. To address the issue of limited benchmarks, we introduce IDBench-V, a comprehensive benchmark encompassing diverse scenes. Extensive experiments demonstrate DreamID-V outperforms state-of-the-art methods and further exhibits exceptional versatility, which can be seamlessly adapted to various swap-related tasks.

Date: January 6, 2026 Project Page (Demo, Codes, Models): https://guoxu1233.github.io/DreamID-V/

#### 1 Introduction

Face swapping aims to generate an image or video that combines the identity of a source face with the attributes (such as background, pose, expression, lighting) from a target image or video. This technique has sparked considerable research interest, due to its significant potential for practical applications in film production, creative design and privacy protection.

Unlike Image Face Swapping (IFS), Video Face Swapping (VFS) presents more challenges, as it introduces additional critical constraints on temporal identity continuity, pose consistency, and environment preservation.

Existing studies on Image Face Swapping (IFS), such as [12, 47], have achieved remarkable success in maintaining identity similarity and preserving attributes. However, directly applying these IFS methods frame-by-frame to video sequences often leads to significant challenges in temporal consistency, resulting in noticeable flickering and jittering artifacts. Recently, the rapid advancement of diffusion-based video

|[Figure 3]|[Figure 4]|[Figure 5]|[Figure 6]|[Figure 7]|[Figure 8]|
|---|---|---|---|---|---|

[Figure 9]

[Figure 10]

[Figure 11]

Ref Identity

|[Figure 12]|[Figure 13]|[Figure 14]|[Figure 15]|[Figure 16]|[Figure 17]|
|---|---|---|---|---|---|

[Figure 18]

[Figure 19]

[Figure 20]

Ref Identity

|[Figure 21]|[Figure 22]|[Figure 23]|[Figure 24]|[Figure 25]|[Figure 26]|
|---|---|---|---|---|---|

[Figure 27]

[Figure 28]

[Figure 29]

Ref Identity

|[Figure 30]|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|
|---|---|---|---|---|

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

Ref Identity

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Ref Identity

- Figure 1 Showcase of DreamID-V. DreamID-V robustly handles challenging scenarios, e.g., complex expressions, animation, large angles, occlusions, and small faces.

generation models has greatly propelled the development of Video Face Swapping (VFS). While methods like VividFace [37], DynamicFace [41], HiFiVFS [5], and CanonSwap [26] have improved the coherence and generation quality of VFS, their capabilities in terms of identity similarity and attribute preservation still lag behind those of state-of-the-art IFS models. The fundamental difference between IFS and VFS lies in the dynamic nature of video, which requires consistent preservation of motion and expression across frames. This observation inspires us to explore whether we can bridge the gap between image and video domains by supplementing these dynamic signals, thereby harnessing the strengths of IFS to significantly boost VFS performance.

Building on these insights, we propose a comprehensive framework comprising a novel data pipeline and a customized architecture to enhance VFS performance significantly. Our proposed data curation pipeline, SyncID-Pipe, seamlessly transfers the superiority of IFS to the video domain. Specifically, the pipeline pretrains a pose-driven First-Last-Frame video generation model, which we term the Identity-Anchored Video Synthesizer (IVS). The IVS employs an adaptive pose attention mechanism to inject pose information into First-Last-Frame video foundation models. This enables the model to generate a video consistent with the

content of the given start and end frames and the actions specified by the pose sequence. Subsequently, we combine it with IFS models to construct bidirectional ID quadruplets for explicit supervision. Furthermore, to enhance data reliability, we propose an expression adaptation strategy to achieve effective expression transfer, incorporating an enhanced background recomposition mechanism to ensure strict environment alignment in the paired videos.

Building upon paired data, we develop DreamID-V, the first video face swapping framework based on Diffusion Transformer (DiT) models [31], achieving high-similarity and superior-coherence results. We first introduce a Modality-Aware Conditioning (MC) mechanism that discriminatively injects conditions from multiple modalities, enabling condition decoupling and feature fusion. Furthermore, we design a novel Synthetic-toReal Curriculum learning strategy to strengthen visual realism while maintaining identity similarity. To further enhance the preservation of facial dynamics under challenging scenarios, we develop an Identity-Coherence Reinforcement Learning (IRL) mechanism, which significantly improves model robustness in complex motions.

Through the aforementioned improvements, our approach enables effective video face swapping across diverse scenarios, as illustrated in Fig. 1. Due to the limited video-face-swapping benchmarks, we introduce IDBench-

V, a comprehensive benchmark encompassing a wide spectrum of videos with varying head poses, facial expressions, and lighting conditions. We conduct extensive evaluations on IDBench-V, demonstrating that DreamID-V achieves clear advantages over state-of-the-art methods, both quantitatively and qualitatively. Notably, our proposed framework exhibits exceptional versatility and can be seamlessly adapted to various swap-related tasks. Overall, our contributions are summarized as follows.

Technology. 1) We develop SyncID-Pipe, which seamlessly transfers the superiority of IFS to VFS, effectively boosting the video face swapping. 2) We propose DreamID-V, the first video face swapping framework based on DiT. 3) We introduce IDBench-V, a comprehensive benchmark tailored for the video face swapping task.

Significance. 1) DreamID-V demonstrates superior generation performance compared to state-of-the-art methods. 2) We present a comprehensive study of the VFS task—including data, model, and benchmark. 3) Our proposed framework shows remarkable versatility and can be flexibly adapted to various swap-related tasks.

#### 2 Related Work

Video Foundation Model. The development of diffusion models [14] has significantly advanced video foundation model research. Early latent diffusion methods [3, 11, 15] extended Text-to-Image models with U-Net architectures to the video domain by incorporating temporal modules such as 3D convolutions and temporal attention. Rencently, the emergence of Diffusion Transformer(DiT) [32]-based methods for video generation has exhibited superior performance in quality and consistency. These methods [19, 25, 27, 40, 46] employ powerful scaling transformers to generate longer and higher-quality videos. In addition to common Text-to-Video and Image-to-Video models, a growing number of keyframe interpolation models [9] have emerged (e.g., First-Last-Frame models), paving the way for various downstream video generation tasks.

Face Swapping. Early image face swapping primarily focused on GAN-based models, such as FSGAN [29, 30], FaceShifter [21], HifiFace [42], and SimSwap [4]. More recently, with the rapid development of diffusion models, several diffusion-based image face swapping models have emerged, including DiffFace [18], DiffSwap [49], FaceAdapter [12], ReFace [1], and DreamID [47], all of which have achieved promising results. Compared to image face swapping, video face swapping is still in its nascent stages. VividFace [37] models video face swapping as a conditional inpainting task and proposes the first diffusion-based framework. DynamicFace [41] incorporates precise and disentangled facial conditions for flexible and accurate control. HiFiVFS [42] introduces an additional attribute extraction module to capture fine-grained attribute features. CanonSwap [26] proposes a canonical space, performing face swapping within this space before projecting back to the real domain. While these works have improved the temporal consistency and quality of generated results, their performance in terms of identity similarity and attribute preservation remains largely unsatisfactory due to a lack of explicit supervision. In contrast, we significantly advance video face swapping by leveraging the superiority of image face swapping to construct explicit supervision.

[Figure 60]

[Figure 61]

###### BidirectionalQuadrupletPair

###### Enhanced BackgroundRecomposition

###### Image 𝐼 Identity-Anchored Video Synthesizer Image 𝐼

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

...

... ...

...

...

... ...

...

... ...

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

+

Adaptive Pose

N × MMDiT Block (FLF2V)

Pose Extractor

Pose Guider

Attention

...

...

...

...

...

Real Video 𝑉 Enhanced Video 𝑉

Pose Sequence 𝑝

Generated Video 𝑉

[Figure 84]

GroundingSAM2

Expression Adaption

[Figure 85]

... ...

[Figure 86]

... ... ...

...

[Figure 87]

... ...

[Figure 88]

[Figure 89]

Inference Path

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Loss

[Figure 96]

Training Path

Image Face Swapping

...

...

Shared Path

...

...

𝑉

+ Add

Minimax-Remover

- Figure 2 Overview of SyncID-Pipe. We pre-train the Identity-Anchored Video Synthesizer and combine it with the Image Face Swapping model to construct Bidirectional Quadruplet Pair data.
- 3 Methodology

We propose a comprehensive framework to boost Video Face Swapping (VFS) by harnessing the prowess of Image Face Swapping (IFS). We first introduce a novel data curation pipeline SyncID-Pipe, which constructs bidirectional ID quadruplets to bridge the gap between VFS and IFS (Sec. 3.1). Building upon paired data, we develop DreamID-V, the first Diffusion Transformer (DiT)-based video face swapping framework employing a core Modality-Aware Conditioning module to discriminatively inject conditions from multiple modalities (Sec. 3.2). To further enhance visual realism and identity consistency under challenging scenarios, we design a Synthetic-to-Real Curriculum mechanism and an Identity-Coherence Reinforcement Learning strategy during training (Sec. 3.3).

Moreover, our proposed framework exhibits exceptional versatility (Sec. 3.4).

##### 3.1 SyncID-Pipe

IFS has demonstrated better performance in identity and attribute preservation compared to VFS. The fundamental difference between IFS and VFS lies in the dynamic nature of video, which requires consistent preservation of motion and expression across frames. This observation inspires us to explore bridging the gap between image and video domains by supplementing dynamic signals, thereby leveraging the strengths of IFS to significantly boost VFS performance.

###### 3.1.1 Identity-Anchored Video Synthesizer

Building on this insight, we introduce a simple yet effective Identity-Anchored Video Synthesizer (IVS) to generate pair data for explicitly supervised training. As shown in Fig. 2, the IVS is trained to reconstruct a portrait video Vr by leveraging its extracted pose sequence p. This is achieved by conditioning a First-LastFrame video foundation model (FLF2V) [9] on the initial and final frames of Vr, along with its pose sequence p. Such reconstruction-based training enables large-scale video pre-training. To minimize modifications to the foundation model and facilitate seamless integration of its pre-trained motion priors with the face swapping framework (detailed in Sec. 3.2), we introduce an Adaptive Pose-Attention mechanism to inject motion information.

Adaptive Pose-Attention. We employ a lightweight Pose Guider composed of several simple convolutional layers to extract pose features and align them with the dimension of the latent feature. To ensure precise spatiotemporal alignment between the pose sequence and the noisy latent video, we reuse the Rotary Position Embedding (RoPE) [38] indices from the noisy latents for the pose condition, which can help maintain the

alignment. In each DiT block, we introduce two trainable linear layers Wk′ and Wv′ to align pose features P with latent feature Z. Formally, the output of the Pose-Attention Znew is:

Znew = Softmax

###### QK⊤

√

d

V + λ · Softmax

###### Q(K′)⊤

√

d

V′, (1)

where Q = ZWq, K = ZWk, and V = ZWv are derived from the frozen DiT layers, while K′ = PWk′ and V′ = PWv′ are from our trainable pose-adapter layers. The hyperparameter λ controls the strength and flexibility of incorporating pose features. We train IVS by collecting a large-scale portrait dataset and extracting pose sequences. The model is optimized using Flow Matching [23]. Through the aforementioned training, IVS can generate a video consistent with the content of the given keyframes and the actions specified by the pose sequence. This video is subsequently utilized for constructing bidirectional ID quadruplets.

###### 3.1.2 Bidirectional ID Quadruplets Construction

Leveraging the identity preservation and dynamic attribute controllability of our designed IVS model, we effectively bridge the gap between IFS and VFS. Building upon this capability, we construct bidirectional ID quadruplet training data, anchored by IFS, to enable explicit supervision. As illustrated in Fig. 2, for a source image-video pair (Ir,Vr) with identity ID A and a target image (Ig) with identity ID B, we first utilize a state-of-the-art IFS model [47] to transfer ID B onto the first and last frames of Vr. This process yields high-quality reference frames (Iref1, Iref2). Subsequently, these reference frames, along with the retargeted pose sequence, are fed into the pre-trained IVS module to synthesize Vg, a video of ID B. The resulting bidirectional ID quadruplet is formatted as {Ir,Vr,Ig,Vg}, where {Ir,Vg,Vr} constitutes the forward-generated paired data, and {Ir,Vr,Vg} represents the backward-real paired data. To further enhance the diversity and practicality of the training data, we implement the following strategies:

Source Data Curation. To ensure the robustness of DreamID-V across diverse and challenging scenarios, we carefully curate source videos encompassing varied makeup styles, extreme lighting conditions, and other adversarial settings. Furthermore, we incorporate talking-head datasets to enhance the ability of the model to preserve subtle facial expressions and accurate lip-synchronization.

Expression Adaptation. Considering that simply using the pose sequence of the source video to drive the generation of the target video leads to identity-expression entanglement and thus causes identity leakage, we use an expression adaptation module to decouple identity and expression, allowing for high-quality expression transfer. During inference, a 3D face reconstruction model [43] extracts identity coefficient from Ig, and the expression and pose coefficients from each Vr frame. We then recombine identity of Ig with expression and pose of each Vr frame to reconstruct a new 3D face model. Projecting this model yields retargeted facial landmarks, which replace the original ones in the pose sequence for final inference.

Enhanced Background Recomposition. Static keyframe-driven IVS often produces videos with background inconsistencies relative to the source videos, particularly with significant background motion. To enhance the applicability of the model in real-world dynamic scenarios, we design an Enhanced Background Recomposition module. As shown in Fig. 2, for forward-generated paired data {Ir,Vg,Vr}, we first extract foreground masks from Vr and Vg using SAM2 [33]. We then utilize MinimaxRemover [33] to remove the foreground from Vr, yielding a clean background video Vbg. Subsequently, the foreground from Vg is pasted back into Vbg to form enhanced video Vg′. A feathering operation is then applied at the foreground edges to achieve a smooth and natural blending. Crucially, in the final paired data obtained this way, the supervision video remains the real video Vr. This prevents the model from learning artifacts introduced during the background pasting process. By specifically augmenting these data, we aim to improve the capabilities of the model in background preservation.

##### 3.2 DreamID-V Framework

VFS aims to seamlessly transfer identity information from a source face to a target video, while preserving attributes of the target video, such as pose, expression, and background. A core challenge in designing such a model involves effectively injecting and simultaneously disentangling and fusing different types of information,

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Pose Extractor

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Structure Guidance

[Figure 114]

ReinforcementLearning

[Figure 115]

Unpatchify&&VAEDecoder

Pose Guider

Identity-Coherence

Source Video

Target Video

Mask Video Adaptive Pose Attention

[Figure 116]

...

| | |
|---|---|
|VAE En|coder|
| | |

Loss

[Figure 117]

[Figure 118]

N × MMDiT Block

Patchify

Channel

[Figure 119]

Spatio-Temporal Context

[Figure 120]

[Figure 121]

ID Encoder

Temporal

Identity Information

Figure 3 Overview of DreamID-V framework. We design customized injection mechanisms for Spatio-Temporal Context, Structural Guidance, and Identity Information, respectively.

including identity and attributes. To address this, we develop DreamID-V, the first video face swapping framework based on Diffusion Transformer (DiT) models, as illustrated in Fig. 3. Central to this framework is the Modality-Aware Conditioning (MC) mechanism, designed to efficiently inject multiple conditions, tailored to their respective needs.

###### 3.2.1 Modality-Aware Conditioning.

To make conditions disentangled from each other, the MC mechanism decomposes the conditions for video face swapping into three distinct types. The details are as follows:

Spatio-Temporal Context Module. To provide the contextual reference information to be retained (e.g., background, lighting), we inject both the reference video and the dilated face mask into the model. Since these conditions must align precisely with the latent noise in both spatial and temporal dimensions, we concatenate them with latent of source video along the channel dimension.

Structural Guidance Module. To achieve fine-grained preservation of dynamic attributes, we incorporate the pose condition as structural guidance. To this end, we employ the Pose-Attention mechanism and initialize corresponding parameters with a pre-trained model described in Sec. 3.1.1, which fully leverages the prior from the IVS model. The strategy enables efficient structural control while avoiding disruption to the high-level features being processed by the DiT.

Identity Information Module. In contrast to the context and structural information, identity condition represents high-level semantic features and requires comprehensive spatiotemporal interaction. We first employ a dedicated ID encoder to encode the reference identity into ID embeddings. And then we concatenate them with the latent noise after patchification along the token dimension, which enables full interaction with latent features through the DiT’s inherent attention mechanism.

In summary, the spatio-temporal context module provides required attribute information and leverages the mask to indicate facial regions to the model. The structural guidance module further enhances motion attributes, improving the preservation of exaggerated movements and fine-grained expressions. Finally, the identity information module efficiently injects identity-related features.

##### 3.3 DreamID-V Training Pipeline

A significant challenge in training video face swapping models involves balancing identity similarity, attribute preservation, visual realism, and ensuring temporal consistency of identity. To mitigate the issue, we introduce a Sync-to-Real Curriculum mechanism along with an Identity-Coherence Reinforcement Learning strategy. The training process is divided into the following stages:

Synthetic Training. In our constructed bidirectional ID quadruplet data {Ir,Vr,Ig,Vg}, {Ir,Vg,Vr} represents the forward-generated paired data, while {Ig,Vr,Vg} represents the backward-real paired data. In this initial stage, we train our model using the forward-generated paired data. Since Vg is synthesized by the IVS module, it remains distributionally consistent with our underlying video foundation model. (as further elaborated in the supplementary material). This alignment significantly accelerates model convergence and enables the attainment of higher identity similarity, yielding superior similarity compared to direct training with backward-real paired data.

Real Augmentation Training. The previous stage yields a VFS model with high identity similarity, however, its realism and background preservation remain limited due to the fact that the forward-generated paired data is model-generated and inherently lacks fidelity in background and realism. Therefore, we introduce a real augmentation training stage to fine-tune the model, which uses backward-real paired data {Ig,Vr,Vg′} augmented by the Enhanced Background Recomposition strategy introduced in Sec. 3.1.2. Experiments show that this stage enables the model to maintain strong identity similarity while achieving excellent background. Identity-Coherence Reinforcement Learning. While the aforementioned mechanisms establish a robust baseline, temporal identity consistency remains a challenge in complex scenarios, particularly in videos with significant motions. We observe that identity similarity fluctuates, remaining high in frontal views and mild movements but degrading considerably during profile views or intense actions. To specifically address this final-mile problem, as shown in Fig. 3, we introduce the Identity-Coherence Reinforcement Learning (IRL) mechanism, inspired by [8, 24]. The core insight behind IRL is to incentivize the model to focus its learning capacity on difficult frames. We formalize this as a policy optimization problem, where the model’s generative process is treated as a policy, πθ, that aims to produce video frames with maximal identity fidelity. Consequently, frames with low identity fidelity are inherently more valuable as learning signals, as they represent the greatest potential for policy improvement. Instead of learning a complex Q-function via temporal difference updates in some methods [22, 36], we leverage the specific nature of the video face-swapping task to define an efficient and explicit Q-value. Given a state y representing the conditional inputs and an action xˆ0 corresponding to the generated video frame, we define the Q-value as:

1 cos(E(ˆx0),E(It)) + δ

, (2)

Q(y,xˆ0) =

where E(·) denotes the feature embedding extracted by [6], cos(·,·) is the cosine similarity, It is the target identity image, and δ is a small constant for numerical stability.

As implemented in our pipeline, we first perform a full sampling pass without backpropagation to generate a video and compute the Q-value for each frame. These frame-wise Q-values are then averaged to obtain Qc for each VAE-encoded chunk. These Qc are then used as weights for the flow matching loss [23] during the IRL training step:

C

Et,ϵ Qc · ∥(zc − ϵ) − vθ ((1 − t)zc + tϵ,t,y)∥2 , (3)

LIRL(θ) =

c=1

where the loss is summed over all C VAE-encoded chunks in the video. By dynamically re-weighting the loss, our IRL mechanism steers the model to refine identity preservation in difficult segments, significantly reducing temporal flickering.

##### 3.4 DreamID-V Versatility

Notably, our proposed DreamID-V extends beyond face swapping task. By constructing explicit paired data for various human-centric swapping tasks—such as outfit, accessory, and hairstyle swapping—simply through

|[Figure 122]|[Figure 123]|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]|[Figure 128]|[Figure 129]|
|---|---|---|---|---|---|---|---|

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Face-Adapter

Face-Adapter

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Stand-In

Stand-In

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

DreamID

DreamID

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

CanonSwap

CanonSwap

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

###### Ours

Ours

|[Figure 172]|[Figure 173]|[Figure 174]|[Figure 175]|[Figure 176]|[Figure 177]|[Figure 178]|[Figure 179]|
|---|---|---|---|---|---|---|---|

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Face-Adapter

Face-Adapter

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Stand-In

Stand-In

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

DreamID

DreamID

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

CanonSwap

CanonSwap

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Ours

Ours

Figure 4 Qualitative comparisons with state-of-the-art methods. Please zoom in for more details.

replacing the Image Face Swapping (IFS) model with a general-purpose image editing model (e.g., Nano banana [2]) within the SyncID-Pipe, DreamID-V can be extended to a wider range of swap category tasks. We will detail this extensibility in Sec. 4.5.

#### 4 Experiments

##### 4.1 Setup

IDBench-V. We introduce IDBench-V, a new comprehensive benchmark for video face swapping. The benchmark comprises 200 real-world source video-target image pairs, covering a diverse range of challenging scenarios. These include small faces, extreme head poses, severe occlusions, complex and dynamic expressions, and cluttered multi-person scenes. IDench-V provides a rigorous and holistic platform for evaluation in real-world usage scenarios.

Implementation Details. We choose OpenHumanVid [20] as the training set and subsequently filter it based on ID similarity to create paired videos of the same identity. Refer to Sec. A.2.2 for more details.

Baselines. We conduct comparisons against existing face swapping state-of-the-art (SOTA) models on IDBench-V. For image face swapping, we compare with FSGAN [30], REFace [1], Face-Adapter [12], and DreamID [47], noting that these image-based methods achieve video face swapping by processing frames individually. For video face swapping, we evaluate against Stand-In [45] and CanonSwap [26]. Due to the unavailability of open-source code for VividFace [37] and DynamicFace [41], we perform a qualitative comparison using videos from their respective demos in Sec. A.5.

Evaluation Metrics We evaluate the performance of various video face swapping methods across three key

|Identity Consistency<br><br>|Attribute Preservation<br><br>|Video Quality|
|---|---|---|
|ID-Arc ↑ ID-Ins ↑ ID-Cur ↑ Variance ↓<br><br>|Pose ↓ Expression ↓ Background ↑ Subject ↑|FVD ↓ Smoothness ↑|

Method

FSGAN [30] 0.435 0.466 0.441 0.0069 7.415 3.463 0.904 0.919 6.582 0.982 REFace [1] 0.472 0.471 0.474 0.0191 5.102 2.785 0.909 0.913 7.084 0.988 Face-Adapter [12] 0.440 0.496 0.450 0.0081 5.156 3.037 0.942 0.945 3.460 0.988 DreamID [47] 0.616 0.702 0.664 0.0058 3.013 2.930 0.940 0.951 3.108 0.989 Stand-In [45] 0.403 0.403 0.367 0.0057 19.819 2.995 0.931 0.951 3.368 0.982 CanonSwap [26] 0.397 0.431 0.407 0.0030 2.430 2.477 0.950 0.954 2.176 0.991 Ours 0.659 0.713 0.688 0.0029 2.446 2.430 0.951 0.966 2.243 0.992

###### Table 1 Quantitative comparisons with baseline methods.

dimensions: Identity Consistency, Attribute Preservation, and Video Quality. For Identity Consistency, we employ ArcFace [6], InsightFace [39], and CurricularFace [16] to compute ID similarity. To quantify temporal stability, we additionally calculate the variance of these frame-wise similarities. Attribute Preservation is assessed by evaluating the fidelity of pose and expression transferred from the driving video.(details are provided in the supplementary material) Furthermore, we incorporate three metrics from VBench [17]: background consistency, subject consistency, and motion smoothness, which respectively evaluate the consistency of the background, the primary subject, and the overall motion. Finally, for Video Quality, we evaluate perceptual video quality in unpaired scenarios using the Fréchet Video Distance (FVD) [10] with a ResNext [44] feature extractor.

##### 4.2 Quantitative Comparisons

Metric Evaluation. As shown in Tab. 1, DreamID-V comprehensively outperforms state-of-the-art models in terms of identity similarity metrics. Regarding attribute preservation, DreamID-V is optimal across almost all metrics, with only a slight inferiority to CanonSwap in terms of pose. It is worth noting that CanonSwap, due to its very low identity similarity, results in minimal alteration to the original video, thereby exhibiting good attribute preservation and video quality. Our proximity to CanonSwap in attribute preservation demonstrates our model’s excellent capability in this regard, while its identity similarity is significantly higher than CanonSwap. This superiority is attributed to our ID quadruplet effectively transferring the high identity similarity from DreamID to VFS. Interestingly, owing to the effectiveness of our training strategy, our identity similarity even slightly surpasses that of DreamID. In terms of video quality, our model also achieves outstanding results, showing substantial improvements compared to IFS models such as REFace, Face-Adapter, and DreamID. This collectively demonstrates that our model not only achieves high identity similarity and robust attribute preservation but also generates high-quality videos.

User Study. We invited 19 volunteers to conduct a human evaluation of the models on IDBench. Each sample was rated across three dimensions: Identity Similarity, Attribute Preservation, and Video Quality, with scores ranging from 1-5. As presented in Tab. 2, show that our model achieved the best performance across all metrics, thereby demonstrating the superior capabilities of our model.

##### 4.3 Qualitative Analysis

Method ID Sim ↑Attr ↑Quality ↑

REFace [1] 1.45 2.15 1.11 Face-Adapter [12] 2.17 2.93 1.14 DreamID [47] 3.78 3.89 3.06 Stand-In [45] 2.45 1.60 2.91 Canonswap [26] 1.99 3.91 3.42

Ours 3.85 4.22 4.15 Table 2 User study of DreamID-V.

We conduct a qualitative comparison with VFS methods CanonSwap and Stand-In, as well as IFS methods Face-Adapter and DreamID. As illustrated in Fig. 4, DreamID-V demonstrates excellent performance across identity similarity, expression preservation, background preservation, and occlusion. In the first two cases presented in the first row, our method demonstrates superiority in identity similarity compared to other approaches, consistently across both male and female subjects. Specifically, our identity similarity is significantly better than that of Face-Adapter, Stand-In, and CanonSwap. Perceptually, the identity similarity is close to DreamID, however, we observe that our IVS module, by incorporating dynamic expression information, leads to superior expression performance compared to DreamID. Stand-In, which utilizes an inpainting approach for face swapping, introduces substantial

Synthetic-to-Real Training Source Video Source Video w/o. IRL w IRL

Only Real Training (w/o ST)

Only Synthetic Training (w/o RAT)

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

a). Ablation results on Synthetic-to-Real Curriculum mechanism b). Ablation results on the IRL mechanism

Figure 5 Ablation studies of DreamID-V.

alterations to the original video. In the second row, the left case highlights the robust performance of our model under occlusion, outperforming all other models. The right case showcases the superiority of our model in handling complex expressions.

##### 4.4 Ablation Studies

To demonstrate the effectiveness of our proposed method, we conduct the following ablation studies: a). Directly training using the self-reconstruction inpainting-based approach (w/o Quadruplet); b). Training exclusively with backward-real paired data (w/o ST); c). Training solely with forward-generated paired data (w/o RAT); d). Training without the Identity-Coherence Reinforcement Learning stage (w/o IRL). As shown in Tab. 3 a), Traditional inpainting-based method yields significantly lower identity similarity, demonstrating the effectiveness of SyncID-Pipe to construct explicitly supervised data to bridge the gap between VFS and IFS. b). As demonstrated in Fig. 5 a), the w/o ST setting yields higher realism but lower identity similarity (i.e., a good FVD score but poor ID-Arc). Conversely, w/o RAT achieves superior identity similarity but at the cost of realism (i.e., a good ID-Arc score but poor FVD). In contrast, our Sync-to-real training strategy(line w/o IRL) ultimately strikes a good balance, maintaining high identity similarity while preserving realism. Furthermore, the IRL mechanism significantly enhances identity similarity under complex motions, leading to a noticeable improvement. A comparison between lines (d) and (e) reveals that IRL not only improves ID-Arc to some extent but also substantially reduces variance, which represents the consistency of inter-frame similarity. As depicted in Fig. 5 b), the top and bottom frames present profile views, while the middle frame shows a frontal view. Without IRL, the model performs well on frontal views but poorly on profile views. However, IRL substantially boosts identity similarity in profile views, as exemplified by the topmost frame.

Method ID-Arc ↑ Variance ↓ Pose ↓ Expression ↓ FVD ↓

- a) w/o Quadruplet 0.510 0.0036 2.468 2.432 2.242

- b) w/o ST 0.604 0.0035 2.742 2.445 2.145

- c) w/o RAT 0.657 0.0042 2.557 2.443 3.845

- d) w/o IRL 0.631 0.0041 2.687 2.488 2.206

- e) Ours 0.659 0.0029 2.446 2.430 2.243 Table 3 Ablation study of DreamID-V.

##### 4.5 Versatility

As shown in Fig. 6, by expanding training data, our model can be extended to a wider range of human-centric swapping tasks, including accessory, outfit, headphone, and hairstyle swapping.

Source Video a). Swap Accessory b). Swap Clothes c). Swap Earphone d). Swap Hair

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Figure 6 Versatility of our methods. Please zoom in for more details.

#### 5 Conclusion

This work presents a comprehensive framework for Video Face Swapping (VFS). Our SyncID-Pipe data pipeline effectively transfers the superiority of image face swapping to video, enabling DreamID-V—the first DiT-based model for VFS—to achieve superior performance on our proposed comprehensive benchmark IDBench-V. The method demonstrates strong versatility and provides a systematic solution for high-fidelity VFS.

#### References

- [1] Sanoojan Baliah, Qinliang Lin, Shengcai Liao, Xiaodan Liang, and Muhammad Haris Khan. Realistic and efficient face swapping: A unified approach with diffusion models. arXiv preprint arXiv:2409.07269, 2024.

- [2] Nano Banana. Gemini 2.5 flash image. https://aistudio.google.com/models/gemini-2-5-flash-image, 2025.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

- [4] Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao Ge. Simswap: An efficient framework for high fidelity face swapping. In MM ’20: The 28th ACM International Conference on Multimedia, 2020.

- [5] Xu Chen, Keke He, Junwei Zhu, Yanhao Ge, Wei Li, and Chengjie Wang. Hifivfs: High fidelity video face swapping. arXiv preprint arXiv:2411.18293, 2024.

- [6] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019.

- [7] Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set. In IEEE Computer Vision and Pattern Recognition Workshops, 2019.

- [8] Shutong Ding, Ke Hu, Zhenhao Zhang, Kan Ren, Weinan Zhang, Jingyi Yu, Jingya Wang, and Ye Shi. Diffusionbased reinforcement learning via q-weighted variational policy optimization. Advances in Neural Information Processing Systems, 37:53945–53968, 2024.

- [9] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [10] Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, Jun-Yan Zhu, and Jia-Bin Huang. On the content bias in fréchet video distance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7277–7288, 2024.

- [11] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

- [12] Yue Han, Junwei Zhu, Keke He, Xu Chen, Yanhao Ge, Wei Li, Xiangtai Li, Jiangning Zhang, Chengjie Wang, and Yong Liu. Face adapter for pre-trained diffusion models with fine-grained id and attribute control. arXiv preprint arXiv:2405.12970, 2024.

- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [15] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

- [16] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5901–5910, 2020.

- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

- [18] K. Kim, Y. Kim, S. Cho, J. Seo, J. Nam, K. Lee, S. Kim, and K. Lee. Diffface: Diffusion-based face swapping with facial guidance. 2022.

- [19] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [20] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, et al. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. arXiv preprint arXiv:2412.00115, 2024.

- [21] Lingzhi Li, Jianmin Bao, Hao Yang, Dong Chen, and Fang Wen. Advancing high fidelity identity swapping for forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5074–5083, 2020.

- [22] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

- [23] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

- [24] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

- [25] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024.

- [26] Xiangyang Luo, Ye Zhu, Yunfei Liu, Lijian Lin, Cong Wan, Zijian Cai, Shao-Lun Huang, and Yu Li. Canonswap: High-fidelity and consistent video face swapping via canonical space modulation. arXiv preprint arXiv:2507.02691, 2025.

- [27] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.

- [28] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579–2605, 2008.

- [29] Yuval Nirkin, Yosi Keller, and Tal Hassner. FSGAN: Subject agnostic face swapping and reenactment. In Proceedings of the IEEE International Conference on Computer Vision, pages 7184–7193, 2019.

- [30] Yuval Nirkin, Yosi Keller, and Tal Hassner. FSGANv2: Improved subject agnostic face swapping and reenactment. IEEE, 2022.
- [31] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [32] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [33] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. URL https://arxiv.org/abs/2408.00714.

- [34] Nataniel Ruiz, Eunji Chong, and James M. Rehg. Fine-grained head pose estimation without keypoints. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2018.

- [35] Seyedmorteza Sadat, Otmar Hilliges, and Romann M Weber. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In The Thirteenth International Conference on Learning Representations, 2024.

- [36] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

- [37] Hao Shao, Shulun Wang, Yang Zhou, Guanglu Song, Dailan He, Shuo Qin, Zhuofan Zong, Bingqi Ma, Yu Liu, and Hongsheng Li. Vividface: A diffusion-based hybrid framework for high-fidelity video face swapping. arXiv preprint arXiv:2412.11279, 2024.

- [38] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [39] The InsightFace Team. Insightface products. https://insightface.ai/products.
- [40] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [41] Runqi Wang, Sijie Xu, Tianyao He, Yang Chen, Wei Zhu, Dejia Song, Nemo Chen, Xu Tang, and Yao Hu. Dynamicface: High-quality and consistent video face swapping using composable 3d facial priors. arXiv preprint arXiv:2501.08553, 2025.

- [42] Yuhan Wang, Xu Chen, Junwei Zhu, Wenqing Chu, Ying Tai, Chengjie Wang, Jilin Li, Yongjian Wu, Feiyue Huang, and Rongrong Ji. Hififace: 3d shape and semantic prior guided high fidelity face swapping. arXiv preprint arXiv:2106.09965, 2021.

- [43] Zidu Wang, Xiangyu Zhu, Tianshuo Zhang, Baiqin Wang, and Zhen Lei. 3d face reconstruction with the geometric guidance of facial part segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1672–1682, 2024.

- [44] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. arXiv preprint arXiv:1611.05431, 2016.

- [45] Bowen Xue, Qixin Yan, Wenjing Wang, Hao Liu, and Chen Li. Stand-in: A lightweight and plug-and-play identity control for video generation. arXiv preprint arXiv:2508.07901, 2025.

- [46] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

- [47] Fulong Ye, Miao Hua, Pengze Zhang, Xinghui Li, Qichao Sun, Songtao Zhao, Qian He, and Xinglong Wu. Dreamid: High-fidelity and fast diffusion-based face swapping via triplet id group learning. arXiv preprint arXiv:2504.14509, 2025.

- [48] Yifu Zhang, Hao Yang, Yuqi Zhang, Yifei Hu, Fengda Zhu, Chuang Lin, Xiaofeng Mei, Yi Jiang, Zehuan Yuan, and Bingyue Peng. Waver: Wave your way to lifelike video generation. arXiv preprint arXiv:2508.15761, 2025.

- [49] Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan Liu, Jie Zhou, and Jiwen Lu. Diffswap: High-fidelity and controllable face swapping via 3d-aware masked diffusion. CVPR, 2023.

#### A Appendix

In the supplementary material, the sections are organized as follows:

- • We provide the details of Flow Matching in Sec. A.1.
- • We provide more details regarding parameters, datasets, inference, evaluation metrics and user study in Sec. A.2.
- • We provide the details of our proposed benchmark IDBench-V in Sec A.3
- • We provide the distribution visualization of synthetic data and real data in Sec. A.4.
- • We provide more comparisons with baselines, more qualitative results in Sec. A.5.
- • We provide Ethical Considerations in Sec. A.6

##### A.1 Preliminary

The Diffusion Transformer (DiT) [32] model employs a transformer as the denoising network to refine the diffusion latent. Our method inherits the video diffusion transformers trained using Flow Matching [23], which conducts the forward process by linearly interpolating between noise and data in a straight line. At the time step t, latent zt is defined as: zt = (1 − t)z0 + tϵ, where z0 is the clean video, and ϵ ∼ N(0,1) is the Gaussian noise. The model is trained to directly regress the target velocity:

0,ϵ ∥(z0 − ϵ) − vθ(zt,t,y)∥2 , (4) where vθ refers to the diffusion model output and y denotes condition.

LFM = Et,z

##### A.2 Implementation Details

###### A.2.1 Inference Details

To fully leverage the model’s enhanced capabilities and maximize identity fidelity, strong Classifier-Free Guidance (CFG) [13] is required. In the context of video face swapping, it is natural to leverage the guidance vector to steer the generation towards a high degree of identity similarity with the target. This guidance vector, d, is defined as the difference between the velocity predictions with and without the target identity condition, Cid:

d = vθ(zt,Cpose,Cref,Cid) − vθ(zt,Cpose,Cref,∅), (5)

where Cpose is the pose sequence, Cref is the concatenation of the source video and its mask, and ∅ denotes the null identity condition. However, we observe that naively applying conventional CFG often introduces a pernicious trade-off: while identity similarity increases, it frequently leads to oversaturation and unrealistic artifacts. Motivated by prior work on guidance modification [35, 48], we propose ID Guidance Purification (IDGP). Our method decomposes the guidance vector d into parallel and orthogonal components relative to the normalized conditional prediction vˆcond:

d∥ = ⟨d,vˆcond⟩vˆcond, (6) d⊥ = d − d∥, (7)

where vcond = vθ(zt,Cpose,Cref,Cid) and vˆcond = vcond/∥vcond∥. We empirically find that the orthogonal component, d⊥, is the primary contributor to the aforementioned artifacts. IDGP, therefore, adjusts the composition of the guidance by differentially re-weighting these components to create a purified guidance vector, dIDGP:

1

α · d⊥, (8) where α > 1 is a hyperparameter that simultaneously amplifies the identity-preserving parallel signal and suppresses the artifact-inducing orthogonal signal. Finally, the purified guidance dIDGP is applied to the conditional prediction with an overall guidance scale s:

dIDGP = α · d∥ +

voutput = vcond + s · dIDGP, (9)

As an evaluator for video face swap tasks, your role is to assess the quality of generated videos. Based on the original video and its corresponding reference image, please evaluate the generated video across three key metrics:

- - Identity Similarity : How well does the generated face in the video preserve the identity from the reference image?
- - Attribute Preservation : How well does the generated video preserve the attributes of the original video, including actions, expressions, and background?
- - Video Quality : How well does the generated video exhibit temporal continuity and high visual quality? Scoring rules: Identity Similarity score: 1 for no resemblance, 2-3 for some resemblance, 4-5 for high similarity

Attribute preservation score: 1 for significant alteration of original video attributes, 2-3 for partial alteration, 4 for largely preserved attributes with accurate expression/action following, and 5 for perfect preservation of all non-face attributes.

Quality score: 1 for severe flickering and implausibility, 2-3 for some flickering with acceptable quality, 4 for no flickering and good quality, and 5 for a perfect video.

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Identity Similarity score:

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Attribute Preservation score:

Video Quality score:

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Figure 7 Demo of User study.

This process effectively purifies the guidance signal, enabling strong identity preservation without sacrificing realism.

###### A.2.2 Detailed Parameters

We use the AdamW optimizer with a constant learning rate of 1.0 × 10−5 for all training stages. The IVS module is trained on 1000 hours of video data. The Synthetic Training stage utilizes 100 hours of IVS-generated video as GT, followed by the Real Augmentation Training stage with 150 hours of real and synthetic data. Finally, the IRL stage is conducted on 10 hours of data selected for high ID variance.

The training process for DreamID-V commences with 50k iterations with a global batch size of 16 on exclusively synthetic ground truth (GT) data Vg to rapidly establish a strong baseline for ID similarity. Subsequently, the model is fine-tuned for an additional 80k iterations with a global batch size of 32 on a hybrid dataset of real GT Vr and synthetic GT Vg to enhance photorealism. The training regimen culminates in the application of our IRL, where the model is further refined on a curated subset of data from the previous stages—specifically, samples exhibiting high variance in ID similarity.

###### A.2.3 Evaluation Metrics

We evaluate the performance of various video face swapping methods across three key dimensions: Identity Consistency, Attribute Preservation, and Video Quality. We employ ArcFace [6], InsightFace [39], and CurricularFace [16] to extract face embeddings and compute the cosine similarity with the target identity image. Additionally, we calculate the variance of these frame-wise similarities to quantify temporal stability. Regarding Attribute Preservation, we assess the fidelity of pose and expression transferred from the driving video. This is achieved by computing the L2 distance between the generated frames and the driving frames in terms of head pose, estimated by HopeNet [34], and expression coefficients, extracted via Deep3DFaceRecon [7].

###### A.2.4 User Study

Fig. 7 illustrates the interface of our user study. Each evaluator was presented with a source video, a reference image, and six anonymized videos generated by different models. Evaluators were instructed to rate each generated video across three dimensions—identity similarity, attribute preservation, and video quality—using a 1-to-5-point scale for each dimension. We recruited 19 evaluators, and their final scores were averaged to obtain the overall evaluation.

Target Img Source Video

|[Figure 284]|[Figure 285]|[Figure 286]|[Figure 287]|
|---|---|---|---|

[Figure 288]

|[Figure 289]|[Figure 290]|[Figure 291]|[Figure 292]|
|---|---|---|---|

[Figure 293]

|[Figure 294]|[Figure 295]|[Figure 296]|[Figure 297]|
|---|---|---|---|

[Figure 298]

|[Figure 299]|[Figure 300]|[Figure 301]|[Figure 302]|
|---|---|---|---|

[Figure 303]

Figure 8 Examples of IDBench-V.

##### A.3 IDBench-V Details

To address the lack of a benchmark for Video Face Swapping (VFS), we introduce a comprehensive benchmark, IDBench-V, which consists of 200 videos paired with meticulously selected ID images. As shown in Fig. 8, our collected videos span diverse categories, including small faces, extreme head poses, severe occlusions, complex and dynamic expressions, and cluttered multi-person scenes.

##### A.4 T-SNE Visualization

To validate our claim that synthetic videos (Vg) are inherently distribution-aligned with our DiT-based architecture, we analyze their latent space representations. We sampled 300 videos respectively from three domains: real source videos (Vr), synthetic videos (Vg), and outputs from the base DiT model. Each video was encoded into a latent vector using a pre-trained VAE, and the resulting representations are visualized in 2D using t-SNE [28]. As shown in Figure 9, the visualization reveals a clear structural relationship. The distribution of synthetic videos (Vg, pink) demonstrates a high degree of overlap with that of the base model’s output (blue), forming a cohesive cluster. In contrast, the real source videos (Vr, green) occupy a more distinct and dispersed region of the latent space. This provides strong visual evidence that our IVS module generates data that is already well-aligned with the model’s target distribution. This inherent alignment explains the accelerated convergence and improved performance observed when training with synthetic data, as it effectively narrows the domain gap from the outset.

##### A.5 More Visual Results

In this section, we provide more visual results of DreamID-V. Fig. 10 and Fig. 13 present our inference results, Fig. 12 and Fig. 13 show comparisons with baselines. As DynamicFace [41] and ViVidFace [37] do not have publicly available source codes, we extract some cases from their websites for comparison. The design of our

Base Model's Output

4

Synthetic Video (Vg)

Source Video (Vr)

2

0

TSNEDimension2

2

4

6

8

10

5.0 2.5 0.0 2.5 5.0 7.5 10.0 12.5

TSNE Dimension 1

Figure 9 T-SNE Visualization of Latent Space Distributions.

SyncID-Pipe seamlessly inherits the capabilities of image face swapping, enabling high-fidelity results in diverse scenarios such as cartoon styles and under complex lighting. Moreover, the Synthetic-to-Real Curriculum learning strategy allows DreamID-V to maintain high photorealism while preserving exceptional identity similarity. Furthermore, our Identity-Coherence Reinforcement Learning (IRL) ensures that DreamID-V maintains high similarity even under challenging conditions, including large poses and extreme expressions. A comparison with closed-source models further underscores the superior identity similarity achieved by DreamID-V.

##### A.6 Ethical Considerations

DreamID-V produces high-fidelity, temporally consistent face-swapped videos that could be mis-used to create non-consensual deepfakes or disinformation. To mitigate these risks we release the model under a click-through license that explicitly prohibits malicious, privacy-violating or misleading applications and require users to obtain explicit consent from any identifiable individual before publication.

|[Figure 304]|[Figure 305]|[Figure 306]|[Figure 307]|
|---|---|---|---|

Source Video

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

|[Figure 318]|[Figure 319]|[Figure 320]|[Figure 321]|
|---|---|---|---|

Source Video

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

|[Figure 332]|[Figure 333]|[Figure 334]|[Figure 335]|
|---|---|---|---|

Source Video

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

|[Figure 341]|[Figure 342]|[Figure 343]|[Figure 344]|
|---|---|---|---|

Source Video

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

###### Figure 10 More qualitative results I.

|[Figure 350]|[Figure 351]|[Figure 352]|[Figure 353]|
|---|---|---|---|

Source Video

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

|[Figure 359]|[Figure 360]|[Figure 361]|[Figure 362]|
|---|---|---|---|

Source Video

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

|[Figure 368]|[Figure 369]|[Figure 370]|[Figure 371]|
|---|---|---|---|

Source Video

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

|[Figure 377]|[Figure 378]|[Figure 379]|[Figure 380]|
|---|---|---|---|

Source Video

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

|[Figure 386]|[Figure 387]|[Figure 388]|[Figure 389]|
|---|---|---|---|

Source Video

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

###### Figure 11 More qualitative results II.

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

DynamicFace

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

### Ours

- Figure 12 More qualitative comparisons I.

[Figure 408]

- Figure 13 More qualitative comparisons II.

