# arXiv:2507.05963v2[cs.CV]9Jul2025

## Tora2: Motion and Appearance Customized Diffusion Transformer for Multi-Entity Video Generation

Zhenghao Zhang

Junchao Liao

Xiangyu Meng

zhangzhenghao.zzh@alibabainc.com Alibaba Group China

liaojunchao.ljc@alibaba-inc.com Alibaba Group China

hzmengxiangyu@gmail.com Alibaba Group China

Long Qin

Weizhi Wang

ql362507@alibaba-inc.com Alibaba Group China

wangweizhi.wwz@alibaba-inc.com Alibaba Group China

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

On the sandy ocean floor, a cup is moving slowly.

|[Figure 6]|
|---|

|[Figure 7]|
|---|

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

A bird flies across the sky, while a duck glides through the water.

|[Figure 12]|
|---|

|[Figure 13]|
|---|

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

In a brightly lit classroom, a man and a doll gradually moving toward the center.

|[Figure 18]|
|---|

|[Figure 19]|
|---|

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

|[Figure 24]|
|---|

A man walks on the grass, with a cat leaping and running beside him, and a balloon drifting with the wind.

Reference & Trajectory

#### Figure 1: Given text prompts, motion trajectories, and reference images for human entities (e.g., man, woman) and non-human entities (e.g., cup, cat, duck), Tora2 generates natural motions following the specified trajectories and textual descriptions for each entity while preserving their unique identities. We highly recommend viewing the videos in the appendix.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MM ’25, Dublin, Ireland

### Abstract

Recent advances in diffusion transformer models for motion-guided video generation, such as Tora, have shown significant progress. In

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

this paper, we present Tora2, an enhanced version of Tora, which introduces several design improvements to expand its capabilities in both appearance and motion customization. Specifically, we introduce a decoupled personalization extractor that generates comprehensive personalization embeddings for multiple open-set entities, better preserving fine-grained visual details compared to previous methods. Building on this, we design a gated self-attention mechanism to integrate trajectory, textual description, and visual information for each entity. This innovation significantly reduces misalignment in multimodal conditioning during training. Moreover, we introduce a contrastive loss that jointly optimizes trajectory dynamics and entity consistency through explicit mapping between motion and personalization embeddings. Tora2 is, to our best knowledge, the first method to achieve simultaneous multientity customization of appearance and motion for video generation. Experimental results demonstrate that Tora2 achieves competitive performance with state-of-the-art customization methods while providing advanced motion control capabilities, which marks a critical advancement in multi-condition video generation.

### CCS Concepts

• Computing methodologies → Computer vision problems; Computer vision tasks.

### Keywords

Controllable Video Generation, Diffusion Transformer, Multimodal Fusion

ACM Reference Format:

Zhenghao Zhang, Junchao Liao, Xiangyu Meng, Long Qin, and Weizhi Wang. 2025. Tora2: Motion and Appearance Customized Diffusion Transformer for Multi-Entity Video Generation. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25). , 10 pages.

### 1 Introduction

Recent advancements in large-scale video Diffusion Transformer (DiT) models [3, 27] have notably transformed the landscape of text-to-video generation. This innovation facilitates a broad array of downstream controllable scenes, making a significant impact on the customization of entity appearance or motion pattern. For instance, Tora [51] integrates trajectory encoder with motion fusion techniques to implement motion controls within DiT frameworks. Meanwhile, ConsisID [49] utilizes a frequency-aware identity preservation DiT model tailored for videos of individual person. Additionally, Video Alchemist [5] combines multiple conditional reference images to facilitate the personalization of videos featuring multiple subjects in open-set contexts. Collectively, these innovations significantly enhance the practical applications of video generation across a variety of scenarios.

Recent research efforts [13, 39–41] have investigated the generation of customized video content tailored to both appearance and motion. These approaches are classified into tuning-based methods [39, 41] and tuning-free methods [40]. Tuning-based methods optimize model parameters and manipulate attention maps for appearance adaptation and motion dynamics, while tuning-free methods learn subject appearance and motion patterns during training, enabling integration without additional fine-tuning. Despite

notable progress, challenges remain. Primarily, the reliance on UNet architectures [30] limits application to emerging DiT models, reducing generation capabilities. Furthermore, current methods focus on single-entity customization, complicating the simultaneous manipulation of multiple entities with diverse motion control.

To address these challenges, we propose Tora2, a tuning-free DiT method specifically designed to maintain the fidelity of multiple entities while adhering to predefined motion trajectories. Existing open-set personalization approaches, such as Video Alchemist, predominantly utilize low-frequency global features extracted by DINOv2 [25] for appearance injection. However, due to the human eye’s sensitivity to high-frequency details, reliance on lowfrequency information often leads to insufficient content similarity, particularly in multi-entity scenarios. To enhance this, we introduce a decoupled personalization extractor (DPE) that integrates both low- and high-frequency details to create comprehensive personalization embeddings, thereby preserving fine-grained visual information more effectively. Specifically, high-frequency features are first extracted using facial analysis networks [7] and ReID networks [21], enabling human-specific and object-specific customization. Our experiments demonstrate that this decoupled approach improves consistency in human face representations. These features are transformed by appearance adapters, combined with global features, and queried using a Q-Former network [19] to generate personalization embeddings. For multi-entity control, a gated self-attention mechanism is utilized to fuse the personalization embeddings with corresponding motion embeddings and textual entity tokens. This mechanism enables the integration of appearance, motion, and textual information, ensuring effective learning across multiple entities. Additionally, we introduce a contrastive learning framework to strengthen cross-modal alignment between the entities and the motion patterns extracted by Tora.

To realize concurrent customization of motion and appearance, we integrate motion embeddings and personalization embeddings at distinct stages within the diffusion transformer. Specifically, motion embeddings are infused through an adaptive layer normalization [11] mechanism that modulates video latents prior to the 3D full-attention computation. Meanwhile, personalization embeddings are introduced via an independent cross-attention module positioned post the primary attention layer. Through systematic exploration of multiple embedding injection sequences, we validate that this hierarchical injection strategy achieves optimal generative performance in terms of both motion fidelity and identity preservation.

In this work, we merge the data collection pipelines from Video Alchemist and Tora, creating a dataset of 1.1 million video clips featuring diverse entities with fluid motion patterns, specifically curatedfortrainingTora2. Evaluationon the MSRVTT-Personalization benchmark [5] reveals that Tora2 excels in multi-entity video customization with precise trajectory control.

Our contributions can be summarized as follows:

• We propose Tora2, the first multi-entity customized video diffusion transformer framework. As shown in Figure 1, Tora2 allows for both appearance and motion trajectory control.

- • We introduce a decoupled personalization extractor designed to learn fine-grained personalization embeddings for openset entities, enhancing both subject and face similarity in multi-entity generation.
- • We design a novel binding strategy and a contrastive learning approach to ensure consistent and aligned representation of entities, motion patterns, and textual descriptions.
- • Experimental evaluations show that Tora2 achieves competitive performance with state-of-the-art customization methods, while also introducing advanced motion trajectory control.

- 2 Related work

- 2.1 Video Diffusion Model

The evolution of video generation models [35, 37, 50] has been driven by progressive architectural innovations. Early approaches like Stable Video diffusion [2], VideoCraft [4], and Animate Anything [6] primarily used U-Net-based architectures with 3D convolutional layers for temporal modeling. However, these frameworks struggled to maintain motion coherence in extended sequences and had limitations in dynamic scene understanding. Recent advancements have shifted towards transformer-based architectures, as seen in models such as Sora [3], MovieGen [27], Vidu [1], and HunyuanVideo [16]. These models utilize a 3D causal VAE [48] to handle the encoding and decoding of raw video data, significantly increasing the compression ratio. Next, they utilize the DiT model architecture, renowned for its exceptional scalability, and integrate a 3D full-attention mechanism, substantially enhancing the potential for controllable video generation.

- 2.2 Trajectory Control in Video Generation

Video generation through trajectory control [6, 24, 36, 52] has become increasingly popular for its precise motion control capabilities. Early methods like DragNUWA [47] and MotionCtrl [38] use flow maps and trajectory coordinates transformed into dense vector maps as guidance signals. TrailBlazer [23] utilizes bounding boxes to steer subject motion during video generation, while DragAnything [42] combines entity representation extraction with a segmentation model to enable entity-level control. LeViTor [34] utilizes keypoint trajectory maps that are augmented with depth information to guide 3D motion, effectively capturing and representing object movements. Tora [51] uses a motion variational autoencoder [48] to integrate trajectory vectors into DiT architectures, ensuring motion information is retained across frames. Despite advances in motion control, these methods can only manipulate the given reference frame, lacking the ability for appearance customization.

- 2.3 Appearance Control in Video Generation

Recent advancements have expanded model personalization techniques [12, 31, 39, 40] for video synthesis. DreamVideo [39] utilizes a tuning-based approach that simultaneously personalizes both identity and motion. VideoBooth [13] leverages foreground segmentation via Grounded-SAM [29] to construct personalized embeddings through a coarse-to-fine strategy. StoryDiffusion [53]

uses a stable self-attention mechanism alongside a semantic motion predictor to guarantee smooth transitions and maintain consistent identity throughout its process. ID-Animator [9] combines IP-Adapter [46] with AnimateDiff [8], employing joint optimization for facial personalization. Meanwhile, ConsisID [49] separates facial features into low-frequency global features and high-frequency intrinsic features, integrating them within a DiT-based framework to maintain identity integrity. Unlike these single-concept methodologies, Video Alchemist [5] achieves multi-subject personalization by employing cross-attention fusion of reference images and textual prompts within a Diffusion Transformer. ConceptMaster [12] integrates visual and textual embeddings through a Decoupled Attention Module, subsequently incorporating these embeddings into the diffusion model using a multi-concept injector. In our research, we enhance appearance control using a decoupled personalization extractor and implement a novel binding strategy, achieving advanced multi-entity video customization with trajectory control.

### 3 Methodology

Our goal is to achieve multi-entity controllable video generation with appearance and trajectory conditioning. In this section, we first outline the diffusion transformer model and Tora framework. Then, we introduce Tora2, explaining how personalization embeddings for open-set entities are obtained, linked to entity words and trajectories, and injected as conditions. Finally, we describe the construction of the training dataset.

### 3.1 Preliminary

Diffusion Transformer (DiT). The DiT [26] represents a breakthrough by integrating the robustness of diffusion models with the advanced capabilities of transformer architectures [32]. This novel architecture is designed to overcome the inherent limitations of conventional U-Net-based latent diffusion models (LDMs), thereby enhancing their performance, versatility, and scalability. While maintaining the overall structure consistent with existing LDMs, the significant innovation involves substituting the U-Net with a transformer architecture to learn the denoising objective function:

L𝜖 = ||𝜖 − 𝜖𝜃 (𝑧𝑡,𝑡,𝑐)||22, (1)

Here, 𝜖𝜃 (·) signifies the output of DiT. The condition 𝑐 is guided into the DiT blocks using cross-attention for adjustment and 𝑧𝑡 denotes the noisy hidden state.

Tora. Tora is the first trajectory-oriented Diffusion Transformer framework that unifies textual, visual, and trajectory conditions for scalable video generation with precise motion guidance. Its core innovation lies in encoding arbitrary trajectories into motion patches aligned with DiT’s input tokens. Given a trajectory

𝑡𝑟𝑎𝑗 = {(𝑥𝑖,𝑦𝑖)}𝑖𝐿=−01 where (𝑥𝑖,𝑦𝑖) denotes spatial coordinates at frame 𝑖, Tora employs a 3D motion VAE to extract compact mo-

tion embeddings𝑚 ∈ R𝑙×ℎ×𝑤×𝑐 matching video latent dimensions, where 𝑙, ℎ, 𝑤, 𝑐 denote compressed video length, height, width, and channels. A motion-guidance fuser then dynamically injects the motion embeddings into DiT blocks, which can be formulated as:

𝑧𝑡 = 𝛾 · 𝑧𝑡 + 𝛽, (2)

Trajectory

Prompt

[Figure 25]

[Figure 26]

…

|[Figure 27]|
|---|

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

A woman enjoying a leisurely walk with a dog in a picturesque park setting.

Motion-Guidance Fuser

DPE

Video Encoder

DINOv2 Face Encoder

3D Full An

Object Encoder

DiT Block

Trajectory Extractor

Text Encoder

Appearance Adapter

Appearance Adapter

Video Decoder

caption embeddings

Cross An

motion embeddings

entity textual embeddings

C C

[Figure 38]

[Figure 39]

[Figure 40]

personalization embeddings

motion condition

Gated Self An

…

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

Q-Former

C

personalization condition

- Figure 2: An overview of Tora2, which consists of a decoupled personalization extractor (DPE), a trajectory extractor, a video diffusion transformer, and a gated self-attention mechanism for entity binding. The DPE generates personalization embeddings by combining high-frequency detail information extracted via a decoupled strategy for both human and non-human objects with the low-frequency semantic features obtained from DINOv2. The trajectory extractor encodes provided trajectories into motion embeddings, which are bound to visual entities using a gated self-attention mechanism. These bound motion and personalization conditions are then fed into the video diffusion transformer, employing a motion-guidance fuser and an additional cross-attention layer to achieve both motion and appearance customization for multi entities.

where the scale 𝛾 and shift 𝛽 are converted from 𝑚 using two convolution layers that are initialized to zero. By training on a carefully curated large-scale video dataset, Tora achieves state-ofthe-art performance, significantly outperforming existing motioncontrollable methods.

### 3.2 Tora2

As shown in Figure 2, Tora2 is a latent Diffusion Transformer that integrates extraction and injection processes for entity and trajectory inputs. Retaining Tora’s trajectory conditioning architecture, we emphasize its novel personalized representation extraction method and binding strategy, which associates entities with corresponding text and trajectory conditions.

Open-set personalization embeddings extraction. To ensure the model can process multiple entities with high fidelity, deriving accurate visual representations from concept images {𝐼𝑖|𝑖 = 1 . . . 𝑁} is crucial. Previous research often employs the output from the last layer of the CLIP [28] or DINOv2 image encoder as personalization embeddings 𝑝𝑖|𝑝𝑖 = 𝜀img(𝐼𝑖),𝑖 = 1 . . . 𝑁 . Although these features encapsulate robust semantic information, they lack sufficient intrinsic identity representation, which often leads to unsatisfactory fidelity in multi-entity generation. Additionally, these features usually yield inadequate alignment with the representation space of diffusion models.

To address these limitations, we propose a decoupled personalization extractor. More specifically, we first utilize facial recognition and ReID backbones to extract identity-strong features for humanspecific and generic objects, respectively. Decoupling is justified by the sensitivity of our eyes to facial details, with experiments demonstrating that this method enhances facial similarity in our scenario. Two additional appearance adapters are then employed to project these high-frequency features into a unified feature space. These

identity-focused features are then concatenated with the features captured by the DINOv2 image encoder, which are semantically robust, thereby creating a comprehensive visual representation:

𝑣𝑖|𝑣𝑖 = Concat( 𝑓𝜃 (𝜀prior(𝐼𝑖)), 𝜀img(𝐼𝑖) ) 𝑖 𝑁=1 , (3)

where 𝜀𝑝𝑟𝑖𝑜𝑟 denotes the corresponding pretrained class-specific recognition encoder and 𝑓𝜃 represents the appearance adapter. To facilitate better alignment with the diffusion transformer, we integrate a learnable Q-Former architecture, which consists of stacked cross-attention layers and feedforward networks. The comprehensive visual representation is used as a key-value corpus, and the Q-Former is leveraged to generate personalization embeddings for each concept:

{𝑝𝑖|𝑝𝑖 = 𝑄(𝑣𝑖),𝑖 = 1 . . . 𝑁}, (4)

This approach significantly enhances the model’s ability to maintain high fidelity when processing multiple concepts simultaneously.

Binding of entity with the word and trajectory. Empirical evidence [5] has shown that unbinding visual concepts from their corresponding textual descriptions leads to misalignment in multientity personalization. Therefore, linear projection is used to extend the personalization embeddings with textual information. Our framework extends this requirement to trajectory-conditioned generation, necessitating precise trimodal alignment between entity, text, and motion trajectory. We address this through a gated selfattention that ensures spatially consistent allocation of customized entities along designated trajectories.

For a given personalization embedding 𝑝𝑖 ∈ 𝑅𝑙𝑝×𝑑, it is associated with a trajectory characterized by motion embedding 𝑚𝑖 ∈ 𝑅𝑙𝑚×𝑑 and word tokens 𝑐𝑖 ∈ 𝑅𝑙𝑐×𝑑. In this context, 𝑙𝑝, 𝑙𝑚, 𝑙𝑐, and 𝑑 denote the number of tokens per reference image, trajectory tokens, text tokens, and token dimensions, respectively. We initiate

the process by concatenating these three modal tokens and subsequently employ a self-attention mechanism integrated with two gated mechanisms to facilitate cross-modal interaction:

𝑝ˆ𝑖,𝑚ˆ𝑖,𝑐ˆ𝑖 = SelfAttention(Concat(𝑝𝑖,𝑚𝑖,𝑐𝑖)), 𝑝𝑖 = 𝑝𝑖 + 𝐺𝑝(𝑝ˆ𝑖), 𝑚𝑖 = 𝑚𝑖 + 𝐺𝑚(𝑚ˆ𝑖), (5)

where 𝐺𝑚 and 𝐺𝑝 denote the gate mechanisms for motion embeddings and personalization embeddings, respectively. Through gated cross-modal interaction, it dynamically modulates the fusion intensity, achieving an optimal balance among appearance preservation, textual description, and motion alignment for each entity.

Personalization and motion injection. As shown in Figure 2, the motion conditions are introduced by the motion-guidance fuser, which employs an adaptive layer normalization to modulate the visual tokens. The concatenation of motion-normalized visual tokens and text tokens then undergoes 3D full attention to capture fine-grained relationships between textual descriptions and visual elements. To preserve the original foundational model knowledge, we apply additional cross-attention to facilitate interaction between the personalization embeddings and the visual tokens. In this manner, we inject the three modal conditions at different locations within the DiT blocks, achieving joint controllable video generation for multiple entities.

Contrastive Loss. We also introduce a dual-branch contrastive learning framework to strengthen cross-modal alignment between entities and motion patterns. Our contrastive objective enforces three properties: (1) semantic-motion correspondence through positive entity-motion pairing, (2) inter-concept discriminability via negative entity-entity pairs, and (3) motion trajectory distinction through negative motion-motion pairs. Formally, the loss can be described as:

Lcont = −∑︁

− ∑︁

exp(m⊤𝑗 p𝑗/𝜏) 𝑖 exp(m⊤𝑗 p𝑖/𝜏)

exp(p𝑖⊤m𝑖/𝜏) 𝑗 exp(p𝑖⊤m𝑗/𝜏)

log

log

,

𝑖

𝑗

(6) where 𝜏 denotes the temperature hyperparameter. This symmetric formulation simultaneously ensures entity-specific motion binding while expanding separation between different personalization embeddings in the joint latent space. Finally, the specific diffusion process can be formulated as:

Ltotal = L𝜖 + 𝜆Lcont, (7)

L𝜖 = E𝑡,𝑧0,𝜖 ∥𝜖 − 𝜖𝜃 (𝑧𝑡,𝑡,𝑐,𝑝,𝑚)∥22 , (8) where 𝜆 denotes the weights of the contrastive loss.

### 3.3 Data collection

Building upon Video Alchemist and Tora, we develop a two-stage data curation framework for multi-entity motion learning. The first stage rigorously filters raw videos that might negatively affect object motion training or cause ambiguity in personalization, while the second constructs precise entity-text-trajectory triplets, yielding 1.1M high-fidelity training samples through systematic annotation. Video Filtering. Our video filtering process comprises multiple meticulous steps to ensure high-quality content suitable for personalized applications. Initially, videos exhibiting encoding errors,

a resolution below 720p, or containing an excessive number of words are discarded. Subsequently, videos undergo a quality assessment based on their aesthetic score1 and optical flow score [43], with only those surpassing an aesthetic score of 5 and a flow score of 2 being retained. To further refine the selection, a camera motion detector2 is employed to exclude videos with significant camera movement, maintaining a zoom detection threshold between 0.4 and 0.6 and acceptable camera movement angles within [0◦, 30◦], [150◦, 200◦], [330◦, 360◦]. Finally, videos that either lack subject entity words or contain subject entity words in plural form are removed to prevent ambiguity in personalization. This comprehensive filtering protocol ensures that the remaining videos are of high quality and content clarity, providing a robust foundation for subsequent analytical processes.

Video Annotation. We utilize Qwen2.5-Max [44] to extract nouns from the captions. Subsequently, we select three frames from the beginning, middle, and end of the video and use LISA [18] to extract entity masks. LISA provides highly accurate segmentation results, even for similar visual appearances and textual semantics. Masks that are too large, too small, or highly fragmented are removed. We employ the center points of the three frame masks and use CoTracker3 [14] twice to obtain more precise tracklets.

### 4 Experiments 4.1 Setup

Implementation details. We select the open-source Tora-T2V version, based on CogVideoX-5B [45], to initialize Tora2. In the training phase, video captions, reference conditions, and trajectory conditions—which consist of paired images, text descriptions, and trajectory points—are dropped with probabilities of 50%, 33%, and 33%, respectively, in line with the classifier-free guidance [10]. The original parameters of the CogVideoX-5B model remain fixed, while the other parameters undergo fine-tuning to facilitate effective motion and appearance control. The trajectory extractor, motion-guidance fuser, Q-Former, appearance adapters, and gated self-attention layer are all subject to joint optimization. The batch size is set to 32, the learning rate to 3 × 10−6 with AdamW [15] used as the optimizer, and the total number of training steps to 15k. Furthermore, the contrastive loss weight is set to 0.2. In the inference phase, we employ DPM [22] with a sampling step of 50 and a text-guidance ratio of 6.0.

Benchmark and Metrics. We use the MSRVTT-Personalization benchmark to evaluate our method with current approaches, which contains 2,130 video clips that are manually annotated and cover both single and multiple subjects. Additionally, for our ablation study, we manually collect and annotate 200 videos that encompass multiple concepts, from online sources. Our evaluation metrics include: 1) Text Similarity (Text-S): Cosine similarity between CLIP [28] text embeddings and generated frame features; 2) Video Similarity (Vid-S): Average CLIP feature similarity between ground truth and generated videos; 3) Subject Similarity (Subj-S): DINObased feature alignment between reference images and generated

- 1https://github.com/christophschuhmann/improved-aesthetic-predictor
- 2https://github.com/antiboredom/camera-motion-detector

Table 1: Quantitative comparison with different multi-entity video customization methods on the MSRVTT-Personalization benchmark, where bold text indicates the best result and underlined text denotes the second-best result. Tora2 demonstrates competitive customization capabilities comparable to Video Alchemist. Moreover, its customization and motion control abilities significantly surpass those of the two-stage pipeline comprising Tora with Flux.1.

Non-human object Human

Method Text-S ↑ Vid-S ↑ Subj-S ↑ TrajError ↓ Text-S ↑ Vid-S ↑ Face-S ↑ TrajError ↓ Tora + Flux.1 [17, 51] 0.254 0.719 0.587 19.72 0.265 0.671 0.363 17.41

Video Alchemist [5] 0.268 0.743 0.626 - 0.272 0.694 0.411 -

#### Tora2 (ours) 0.273 0.741 0.615 17.43 0.274 0.702 0.419 13.52

subject segments using Grounding-DINO [20]; 4) Face Similarity (Face-S): ArcFace-R100 [7] feature consistency between reference face crops and YOLOv9-C [33] detected facial regions; 5) Trajectory Error (TrajError): Mean L1 distance between the predicted motion trajectories by CoTracker3 [14] and ground truth.

### 4.2 Qualitative and Quantitative Analysis

Table 1 shows the quantitative evaluation results. To weaken unfair comparisons caused by different fundamental models, we build another baseline that first employs the tuning-based Flux.1 [17] to generate the customized image and employs the Tora-I2V model to generate the videos.

For personalized conditions, Tora2 achieves a mean subject similarity score 1.1% lower than Video Alchemist but demonstrates a 0.8% improvement in facial attribute preservation. Despite sharing the foundational video DiT model architecture, Tora2’s motionconditioned training paradigm introduces additional complexity compared to the personalized feature learning approach of Video Alchemist. The comparable personalization performance indicates that our DPE effectively captures discriminative identity features under joint optimization with motion conditioning. Under motiondriven generation scenarios, while Video Alchemist does not provide implementation for trajectory accuracy evaluation, this capability is outside the scope of their method’s design. The Tora + Flux.1 pipeline encounters substantial degradation in subject and identity fidelity, especially in facial areas, owing to the architectural limitations of separation control. While the fine-tuned Flux.1 model successfully initializes the first frame with strong reference alignment, the subsequent frames devolve because the framework lacks continuous personalization conditioning to anchor the generative process. Notably, Tora2’s joint learning strategy for fusing motion dynamics and conceptual representations consistently achieves superior generation quality metrics across both identity and motion-conditioned tasks.

Figure 3 illustrates a comparative analysis of video generations produced by contrasting approaches. Due to its closed-source status, Video Alchemist results are not included in this evaluation. Tora + Flux.1 employs a basic concatenation strategy for appearance and motion control. However, this approach results in suboptimal integration of modalities, leading to diminished entity consistency with the reference image over extended temporal sequences and excessive positional drifts in trajectory-specific regions. Such limitations compromise both spatiotemporal coherence and trajectory

#### Table 2: Effect of different designs for personalization embedding. We exclude the motion module and the binding operator to eliminate the effects from other conditions.

Personalization encoder Text-S ↑ Vid-S ↑ Subj-S ↑ Face-S ↑

DINOv2 0.262 0.717 0.602 0.389 DINOv2 + ReID 0.253 0.698 0.599 0.362

DINOv2 + ReID & Face 0.257 0.725 0.612 0.404 DPE (ours) 0.266 0.733 0.621 0.413

fidelity. In contrast, Tora2’s synergistic training framework produces videos with superior photorealism, temporal smoothness, and identity preservation, achieved through effective latent space alignment between appearance and motion conditions.

### 4.3 Ablation study

For our ablation study, we utilize the annotated set of 200 videos. The Metrics for Text Similarity, Video Similarity, and Trajectory Error are averaged across both object and human entities to provide a unified illustration.

The different design for personalization embedding extraction. To evaluate the effectiveness of our proposed decoupled personalization encoder, we conduct ablation experiments using different methods to extract personalization embeddings as control signals. These experiments include: (a) utilizing low-frequency global features extracted by the DINOv2 image encoder, as seen in Video Alchemist; (b) merging global features with high-frequency features from the ReID model through linear projection; (c) combining global features with decoupled high-frequency features extracted from ReID and facial recognition models via linear projection; and (d) employing our DPE, which integrates these features using a Q-Former architecture to query concatenated embeddings through learned cross-modal interactions. Figure 4 and Table 2 present the qualitative and quantitative results.

We observe that the semantic features extracted by DINOv2 can generate entities that follow text descriptions, but they struggle to transfer high-frequency details such as facial expressions. Injecting only the high-frequency discriminative features extracted by the ReID model causes significant training instability and convergence challenges, which substantially degrade visual fidelity and adherence to textual instructions when paired with the DINOv2 baseline. The decoupled strategy significantly boosts subject fidelity, making generated content visually resemble the reference entity. However,

| |
|---|

|[Figure 46]|
|---|

[Figure 47]

[Figure 48]

Tora + Flux.1

[Figure 49]

Tora2

| |
|---|

A woman standing in a lush, sunlit park, walking while holding a cuddly teddy bear.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Tora + Flux.1

Tora2

Tora + Flux.1

Tora2

|[Figure 58]|
|---|

|[Figure 59]|
|---|

A man in a white blazer confidently walks alongside a beautiful white horse on a sunlit path, amid autumn's warm atmosphere.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

|[Figure 70]|
|---|

|[Figure 71]|
|---|

On the moon, a man and a corgi walk forward together.

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

- Figure 3: Qualitative comparison of appearance and motion customization for multiple entities. The depth map from the first frame of Tora2 is used as a condition for Flux.1 to ensure consistent entity positioning. The joint control of Tora2 distinctly illustrates superior performance in entity fidelity and motion smoothing.

insufficient token-level interaction via MLP projection hampers the model’s ability to accurately follow textual instructions, as evident in the (c) in Figure 4 where the entities fail to look at each other as instructed. Our proposed method achieves the best results across all metrics. This comparison confirms the necessity of combining different types of features for open-set personalization. It also demonstrates that explicit query-attend mechanisms substantially outperform naive feature concatenation in achieving better alignment within the diffusion transformer context.

The effect of binding different modal features. We validate our entity-binding mechanism, which associates visual elements with corresponding motion trajectories and textual entities, by conducting comparative experiments across three architectural variants: (a)

#### Table 3: Quantitative comparison of binding strategy design choices. Our gated self-attention mechanism delivers superior performance.

Binding strategy Text-S ↑ Vid-S ↑ Subj-S ↑ Face-S ↑ TrajError ↓ No Binding 0.258 0.699 0.589 0.387 29.95

Linear Projection 0.261 0.708 0.599 0.391 20.74 Gated self-attention 0.259 0.718 0.604 0.395 17.31

direct injection of personalization embeddings into the DiT blocks; (b) concatenation of personalization embeddings, motion embeddings, and textual features along the channel dimension followed by linear projection; (c) our gated self-attention mechanism that

|[Figure 82]|
|---|

|[Figure 83]|
|---|

A man and a woman exchange smiles as they face each other, before turning to look warmly at the camera.

[Figure 84]

[Figure 85]

[Figure 86]

(a) DINOv2

[Figure 87]

[Figure 88]

(b) DINOv2 + ReID

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

(c) DINOv2 + ReID & Face

[Figure 93]

[Figure 94]

(d) DPE (ours)

[Figure 95]

#### Figure 4: The ablation of personalization embeddings extraction. The proposed DPE adheres to the actions described in the prompts and achieves stable entity fidelity.

dynamically regulates feature interactions across three modalities through learnable attention gates.

Table 3 illustrates that the baseline architecture lacking explicit entity binding exhibits significant limitations, showing an additional 12.6-pixel offset in trajectory alignment and a 1.5% reduction in subject similarity compared to our proposed method. This performance disparity manifests visually through erroneous associations between specified motion paths and irrelevant background elements, especially when processing prompts containing multiple entities. Figure 5 provides qualitative comparison results, clearly indicating that in the absence of a binding strategy, the trajectory intended for the bird incorrectly results in moving the camera to the right, failing to control the bird as intended. In the linear projection variant, although it successfully associates trajectories, entities, and words, this relatively simple interaction does not represent the optimal choice for multi-entity association. Compared to our method, there is a decrease of 1.0% in video similarity, as evidenced by deformation observed in both the cat and bird in the latter part of the video. Our gated attention architecture emerges as the optimal solution, effectively binding the entity with the corresponding trajectory while maintaining fidelity.

The effect of contrastive loss. Table 4 quantitatively validates our contrastive learning strategy. By optimizing the relative distances between personalization embeddings and motion patterns in the latent space, it ensures precise disentanglement of multiple entities’

A tabby cat walks across the soft golden sands of a tranquil beach. As it moves, a vibrant black and orange bird effortlessly flies through the clear blue sky.

|[Figure 96]|
|---|

|[Figure 97]|
|---|

| |
|---|

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

[Figure 108]

[Figure 109]

No binding Linear Projection Gated Self Aention

Figure 5: Ablation study of cross-model binding strategies. The binding mechanisms facilitate precise motion and appearance control in scenarios involving multiple entities.

Table 4: Ablation study on contrastive learning. We incorporate the DPE and binding mechanism as the baseline.

Text-S ↑ Vid-S ↑ Subj-S ↑ Face-S ↑ TrajError ↓ baseline 0.259 0.718 0.604 0.395 17.31

baseline + CL 0.261 0.723 0.613 0.404 14.16

attributes while facilitating their harmonious integration within provided motion conditions, boosting trajectory adherence accuracy by reducing the distance by approximately 3.2 pixels offset while increasing about 1.0% identity preservation fidelity.

The ablation of the injection order for motion and personalization embeddings. The injection method for motion and personalization embeddings via adaptive layer normalization and crossattention has proven to be the most effective. We adhere to these advanced designs while exploring the injection order of these features. The findings, detailed in Table 5, suggest that injecting personalization embeddings early slightly reduces motion tracking fidelity. This occurs because early activation of the cross-attention mechanism disproportionately amplifies stylistic identity cues, thereby dampening the propagation of motion trajectory semantics throughout the diffusion process. As a result, we have selected the motiontext-personalization injection order as our final architecture. This arrangement ensures a nuanced signal flow: motion embeddings initially establish temporal dynamics, followed by text-based conditioning to maintain semantic integrity, and finally, personalization embeddings enhance appearance traits without sacrificing trajectory precision.

#### Table 5: Ablation of the injection order. The "P" denotes the appearance conditions, and "M" represents the motion conditions.

Order Text-S ↑ Vid-S ↑ Subj-S ↑ Face-S ↑ TrajError ↓

P → M 0.253 0.714 0.611 0.409 19.23 M → P 0.261 0.723 0.613 0.404 14.16

### 5 Conclusion

In this paper, we introduce Tora2, a unified controllable video generation framework that effectively customizes multiple entities with motion trajectory control. Tora2 utilizes a decoupled personalization extractor to achieve open-set personalization embedding extraction by merging semantic features and high-frequency intrinsic features through a learned Q-Former framework. Additionally, it employs a novel binding strategy to associate visual entities with corresponding motion trajectories and textual words, ensuring coherent control across different entities. Extensive experiments demonstrate that Tora2 achieves competitive performance with state-of-the-art customization methods while introducing advanced motion trajectory control. These findings significantly enhance the existing capabilities of controllable video generation processes.

### References

- [1] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. 2024. Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator with Diffusion Models. arXiv:2405.04233
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. 2023. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv:2311.15127
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. https: //openai.com/research/video-generation-models-as-world-simulators.
- [4] Haoxin Chen, Menghan Xia, Yin-Yin He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao-Liang Weng, and Ying Shan. 2023. VideoCrafter1: Open Diffusion Models for High-Quality Video Generation. arXiv:2310.19512
- [5] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. 2025. Multi-subject Open-set Personalization in Video Generation. arXiv:2501.06187 [cs.CV]
- [6] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. 2023. Fine-Grained Open Domain Image Animation with Motion Guidance. arXiv:2311.12886
- [7] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 2019. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In IEEE Conf. Comput. Vis. Pattern Recog. Computer Vision Foundation / IEEE, 4690–4699. doi:10.1109/ CVPR.2019.00482
- [8] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv:2307.04725
- [9] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. 2024. ID-Animator: Zero-Shot Identity-Preserving Human Video Generation. arXiv:2404.15275 [cs.CV]
- [10] Jonathan Ho and Tim Salimans. 2022. Classifier-free diffusion guidance. arXiv:2207.12598
- [11] Xun Huang and Serge J. Belongie. 2017. Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization. In Int. Conf. Comput. Vis. IEEE Computer Society, 1510–1519. doi:10.1109/ICCV.2017.167
- [12] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. 2025. ConceptMaster: MultiConcept Video Customization on Diffusion Transformer Models Without TestTime Tuning. arXiv:2501.04698 [cs.CV]

- [13] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. 2024. VideoBooth: Diffusion-based Video Generation with Image Prompts. In IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 6689–6700. doi:10.1109/CVPR52733.2024.00639
- [14] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. 2024. CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos. arXiv:2410.11831 [cs.CV]
- [15] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In Int. Conf. Learn. Represent., Yoshua Bengio and Yann LeCun (Eds.).
- [16] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. 2025. HunyuanVideo: A Systematic Framework For Large Video Generative Models. arXiv:2412.03603 [cs.CV]
- [17] Shakker Labs. [n.d.]. Flux.1-dev-controlnet-union-pro. Accessed 2024 [Online]. https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro
- [18] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. 2024. LISA: Reasoning Segmentation via Large Language Model. In IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 9579–9589. doi:10.1109/CVPR52733.2024.00915
- [19] Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. 2022. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. In Int. Conf. Mach. Learn. (Proceedings of Machine Learning Research, Vol. 162), Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (Eds.). PMLR, 12888–12900. https://proceedings.mlr.press/v162/li22n.html
- [20] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. 2024. Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection. In Eur. Conf. Comput. Vis. (Lecture Notes in Computer Science, Vol. 15105), Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer, 38–55. doi:10.1007/978-3-031-72970-6_3
- [21] Yang Liu, Idil Esen Zulfikar, Jonathon Luiten, Achal Dave, Deva Ramanan, Bastian Leibe, Aljosa Osep, and Laura Leal-Taixé. 2022. Opening up Open World Tracking. In IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 19023–19033. doi:10.1109/CVPR52688.2022.01846
- [22] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu.

2022. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. In Adv. Neural Inform. Process. Syst., Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. 5775–5787. http://papers.nips.cc/paper_files/paper/2022/hash/ 260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html

- [23] Wan-Duo Kurt Ma, John P. Lewis, and W. Bastiaan Kleijn. 2024. TrailBlazer: Trajectory Control for Diffusion-Based Video Generation. In SIGGRAPH Asia, Takeo Igarashi, Ariel Shamir, and Hao (Richard) Zhang (Eds.). ACM, 97:1–97:11. doi:10.1145/3680528.3687652
- [24] Arun Mallya, Ting-Chun Wang, and Ming-Yu Liu. 2022. Implicit Warping for Animation with Image Sets. In Adv. Neural Inform. Process. Syst., Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.). 22438–22450. http://papers.nips.cc/paper_files/paper/2022/hash/ 8cb31912235561112339f04903657f72-Abstract-Conference.html
- [25] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, PoYao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2024. DINOv2: Learning Robust Visual Features without Supervision. Trans. Mach. Learn. Res. 2024 (2024). https://openreview.net/forum? id=a68SUt6zFt
- [26] William Peebles and Saining Xie. 2023. Scalable Diffusion Models with Transformers. In Int. Conf. Comput. Vis. IEEE, 4172–4182. doi:10.1109/ICCV51070.2023.00387
- [27] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam S. Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali K. Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dmitry Vengertsev, Edgar Schönfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie

- Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. 2024. Movie Gen: A Cast of Media Foundation Models. arXiv:2410.13720
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Int. Conf. Mach. Learn. (Proceedings of Machine Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 8748–8763. http://proceedings.mlr.press/v139/radford21a.html
- [29] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. 2024. Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks. arXiv:2401.14159 [cs.CV]
- [30] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention (Lecture Notes in Computer Science, Vol. 9351), Nassir Navab, Joachim Hornegger, William M. Wells III, and Alejandro F. Frangi (Eds.). Springer, 234–241. doi:10.1007/978-3-319-24574-4_28
- [31] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. In IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 22500–22510. doi:10.1109/CVPR52729.2023.02155
- [32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Adv. Neural Inform. Process. Syst., Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (Eds.). 5998–6008. https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
- [33] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. 2024. YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information. In Eur. Conf. Comput. Vis. (Lecture Notes in Computer Science, Vol. 15089), Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer, 1–21. doi:10.1007/978-3-031-72751-1_1
- [34] Hanlin Wang, Hao Ouyang, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Qifeng Chen, Yujun Shen, and Limin Wang. 2024. LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis. arXiv:2412.15214 [cs.CV]
- [35] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023. Modelscope text-to-video technical report. arXiv:2308.06571
- [36] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. 2023. VideoComposer: Compositional Video Synthesis with Motion Controllability. In Adv. Neural Inform. Process. Syst., Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.), Vol. 36. 7594–7611. http://papers.nips.cc/paper_files/paper/2023/hash/ 180f6184a3458fa19c28c5483bc61877-Abstract-Conference.html
- [37] Xiang Wang, Shiwei Zhang, Hangjie Yuan, Zhiwu Qing, Biao Gong, Yingya Zhang, Yujun Shen, Changxin Gao, and Nong Sang. 2024. A Recipe for Scaling up Text-to-Video Generation with Text-free Videos. In IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 6572–6582. doi:10.1109/CVPR52733.2024.00628
- [38] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Yin Shan. 2023. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. arXiv:2312.03641
- [39] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. 2024. Dream Video: Composing Your Dream Videos with Customized Subject and Motion. In IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 6537–6549. doi:10.1109/CVPR52733.2024.00625
- [40] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Feng Liu, Zhizhong Huang, Jiaxin Ye, Yingya Zhang, and Hongming Shan. 2024. DreamVideo-2: Zero-Shot Subject-Driven Video Customization with Precise Motion Control. arXiv:2410.13830 [cs.CV]
- [41] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. 2024. MotionBooth: Motion-Aware Customized Text-to-Video Generation. arXiv:2406.17758 [cs.CV]
- [42] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. 2024. DragAnything: Motion Control for Anything Using Entity Representation. In Eur. Conf. Comput. Vis. (Lecture Notes in Computer Science, Vol. 15080), Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer, 331–348. doi:10.1007/978-3-031-72670-5_19
- [43] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. 2023. Unifying Flow, Stereo and Depth Estimation. IEEE Trans. Pattern Anal. Mach. Intell. 45, 11 (2023), 13941–13958. doi:10.1109/TPAMI. 2023.3298645
- [44] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2.5 technical report. arXiv:2412.15115

- [45] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv:2408.06072
- [46] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv:2308.06721 [cs.CV]
- [47] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. 2023. DragNUWA: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv:2308.08089
- [48] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang.

2023. MAGVIT: Masked Generative Video Transformer. In IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 10459–10469. doi:10.1109/CVPR52729.2023.01008

- [49] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. 2024. Identity-Preserving Text-to-Video Generation by Frequency Decomposition. arXiv:2411.17440 [cs.CV]
- [50] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. 2023. I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models. arXiv:2311.04145
- [51] Zhenghao Zhang, Junchao Liao, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. 2024. Tora: Trajectory-oriented Diffusion Transformer for Video Generation. arXiv:2407.21705 [cs.CV]
- [52] Jian Zhao and Hui Zhang. 2022. Thin-Plate Spline Motion Model for Image Animation. In IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 3647–3656. doi:10. 1109/CVPR52688.2022.00364
- [53] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou.

2024. StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation. In Adv. Neural Inform. Process. Syst., Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (Eds.), Vol. 37. 110315–110340. http://papers.nips.cc/paper_files/paper/ 2024/hash/c7138635035501eb71b0adf6ddc319d6-Abstract-Conference.html

