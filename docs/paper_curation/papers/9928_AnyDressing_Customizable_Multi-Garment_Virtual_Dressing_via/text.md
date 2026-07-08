## AnyDressing: Customizable Multi-Garment Virtual Dressing via Latent Diffusion Models

Xinghui Li1 Qichao Sun1 Pengze Zhang1 Fulong Ye1 Zhichao Liao2 Wanquan Feng1† Songtao Zhao1† Qian He1 1ByteDance 2Tsinghua University https://crayon-shinchan.github.io/AnyDressing/

# arXiv:2412.04146v2[cs.CV]6Jan2025

Realistic Scenes

Stylized Scenes

Complex Garment

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

”A man with sunglasses, in the mountains”

”A girl, red hair, wearing a hat, by a fence”

”A girl, yellow hair, in the city”

”A girl, sitting in the hallway”

”A girl in the forest”

###### ReliabilityCompatibility

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

”A girl, in front of the Eiffel Tower”

”A girl, in the playground, holding the balloons”

”A girl, holding flowers”

”A girl, in the snow”

”A girl, in the gym”

with ControlNet & IP-Adapter-FaceID

with LoRA

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

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

[Figure 45]

[Figure 46]

[Figure 47]

”Saiyans, sitting, in the city”

”Crayon Shin-chan, doll”

”A girl with a bag in the classroom”

”A man, sky, in a field of ”A man, on the moon” flowers”

Figure 1. Customizable virtual dressing results of our AnyDressing. Reliability: AnyDressing is well-suited for a variety of scenes and complex garments. Compatibility: AnyDressing is compatible with LoRA [15] and plugins such as ControlNet [55] and FaceID [54].

#### Abstract

ios. In this paper, we focus on a new task, i.e., MultiGarment Virtual Dressing, and we propose a novel AnyDressing method for customizing characters conditioned on any combination of garments and any personalized text prompts. AnyDressing comprises two primary networks named GarmentsNet and DressingNet, which are respectively dedicated to extracting detailed clothing features and generating customized images. Specifically, we propose an efficient and scalable module called Garment-Specific Fea-

Recent advances in garment-centric image generation from text and image prompts based on diffusion models are impressive. However, existing methods lack support for various combinations of attire, and struggle to preserve the garment details while maintaining faithfulness to the text prompts, limiting their performance across diverse scenar-

†Corresponding author.

ture Extractor in GarmentsNet to individually encode garment textures in parallel. This design prevents garment confusion while ensuring network efficiency. Meanwhile, we design an adaptive Dressing-Attention mechanism and a novel Instance-Level Garment Localization Learning strategy in DressingNet to accurately inject multi-garment features into their corresponding regions. This approach efficiently integrates multi-garment texture cues into generated images and further enhances text-image consistency. Additionally, we introduce a Garment-Enhanced Texture Learning strategy to improve the fine-grained texture details of garments. Thanks to our well-craft design, AnyDressing can serve as a plug-in module to easily integrate with any community control extensions for diffusion models, improving the diversity and controllability of synthesized images. Extensive experiments show that AnyDressing achieves state-of-the-art results.

#### 1. Introduction

In recent years, the field of image generation has experienced transformative advancements [3, 9, 22], particularly with methods based on Latent Diffusion Models (LDMs) achieving remarkable success in text-to-image generation tasks [14, 37, 38, 42, 44]. Considering only textual information is inadequate for image customization, numerous approaches incorporate reference images with textual descriptions for image generation [25, 39, 49]. Specially, the Virtual Dressing (VD) task of generating garment-centric human images based on the reference garments has sparked considerable research interest [4, 40, 47], due to its significant potential for practical applications in e-commerce and creative design.

VD is used to be regarded as a subtask of traditional subject-driven image customization, prior approaches [10, 18, 25, 31, 39, 41, 45, 57] simply integrate the features of reference image into the text embeddings without fully exploiting the information from the reference image. Several subsequent works [33, 54] more comprehensively utilize the features of the reference image by training additional cross-attention layers to integrate reference image features into the diffusion model. However, these methods struggle to preserve the intricate textures of the garment. Recently, some methods [4, 40, 47] focus on garment-centric image generation. Most of them leverage a full copy of diffusion U-Net as the garment encoder named ReferenceNet to maintain fine-grained garment information. DreamFit [30] proposes a lightweight garment encoder, which utilizes trainable LoRA layers to extract garment features instead of finetuning a full copy of the UNet. Nevertheless, these methods are tailored exclusively to single items of clothing and lack support for multiple conditions, thus hindering the ability to freely dress in any combination of various gar-

ments.

In this work, our focus is on a new task Multi-Garment Virtual Dressing, personalizing a character wearing any combination of target garments according to the customized text prompt or other controls. The task poses several challenges, including: 1) Garment fidelity: preventing confusion among multiple garments while preserving the intricate textures of each; 2) Text-Image consistency: minimizing the influence of multiple garments on irrelevant regions to ensure the faithfulness of the generated images to the text prompts; 3) Plugin compatibility: enabling seamless integration with community control plugins for LDMs.

To address the aforementioned issues, we propose AnyDressing, a novel approach that customizes characters conditioned on any combination of garments and any personalized text prompts. AnyDressing primarily comprises two primary networks named GarmentsNet and DressingNet. The GarmentsNet leverages a core Garment-Specific Feature Extractor (GFE) module to extract multi-garment detailed features, which utilizes parallelized self-attention layers within a shared U-Net architecture to individually encode garment textures. And we employ LoRA mechanism within the self-attention layers to further reduce the parameter increase associated with the added garments. The GFE module not only avoids clothing blending but also ensures network efficiency, allowing for easy scalability to any number of garments. The DressingNet employs a DressingAttention (DA) mechanism to seamlessly integrate multigarment features into the denoising process. To ensure that each garment instance focuses specifically on its corresponding region, we further introduce a novel InstanceLevel Garment Localization (IGL) learning strategy in DA. This avoids influencing other irrelevant regions in the synthetic image, thus improving fidelity to arbitrary customized text prompts. Additionally, to enhance texture details, we design a Garment-Enhanced Texture Learning (GTL) strategy that strengthens the supervision of attire by imposing constraints from perceptual features and high-frequency information.

Extensive experiments show that AnyDressing has certain advantages in the quantitative and qualitative results compared to state-of-the-art methods. Especially, AnyDressing can serve as a plugin compatible with various finetuned LDMs, customized LoRAs [15], and other extensions such as ControlNet [55] and IP-Adapter [54], enhancing the diversity and controllability of synthetic images. In summary, our contributions are as follows:

- • We propose a novel GarmentsNet to efficiently capture multi-garment textures in parallel by employing a core Garment-Specific Feature Extractor.
- • We design a novel DressingNet incorporating a DressingAttention mechanism and an Instance-Level Garment Localization Learning strategy to accurately inject multi-

- garment features into their corresponding regions.
- • We introduce a Garment-Enhanced Texture Learning strategy to effectively enhance the fine-grained texture details in synthetic images.
- • Our framework can seamlessly integrate with any community control plugins for diffusion models. Both quantitative and qualitative experimental results demonstrate the superiority of our AnyDressing.

#### 2. Related Work

Latent Diffusion Models. Latent Diffusion Models (LDMs) [38] have become widely used in text-to-image generation tasks. Recent advancements have focused on making generated content more stable and controllable. For instance, ControlNet [55] and T2I Adapter [36] introduced additional conditioning modules injecting control into the denoising U-net via extra branches, such as edges and pose. Additionally, large model fine-tuning methods like LoRA [15] have significantly enhanced LDMs’ generative capabilities in specific scenarios. In this work, we can integrate with various fine-tuned LDMs and customized LoRAs to enhance the diversity of generated images.

Subject-Driven Image Generation. Subject-driven generation aims to produce content that aligns with the visual features of a reference image. Methods for this task can be categorized into Tuning-based methods [10, 12, 25, 39] and Tuning-free methods [17, 23, 27, 29, 33, 50, 51, 54, 56]. Tuning-based methods, such as DreamBooth [39] and Custom-Diffusion [25] require optimizing specific text tokens to represent target concepts using a limited set of subject images. On the other hand, Tuning-free methods generally encode the reference image into feature embeddings. FastComposer [51] integrates image features into text embeddings, while IP-Adapter[54] and SSR-Encoder[56] integrate image features into the denoising U-net through a decoupled cross-attention mechanism. However, these methods struggle to preserve the fine-grained texture.

Virtual Try-On. Virtual Try-On (VTON) aims to synthesize an image of a specific person wearing a desired garment. Early methods [5, 13, 19, 26, 28, 34, 46, 52] utilize generative adversarial networks (GANs) with two-stage strategy, which rely on an explicit warping module and struggle to handle complex backgrounds. Recent studies [6, 11, 24, 35, 53] have used pre-trained LDMs as priors for VTON tasks. LADI-VTON [35] and DCI-VTON [11] explicitly deform the clothes and then use diffusion models to fuse and refine them. Rencent works [6, 24, 53] employ parallel U-Nets for clothing feature extraction and inject features into a denoising U-Net. However, VTON is essentially a localized image editing task and requires an existing model image, lacking flexibility in application scenarios.

Virtual Dressing. Virtual Dressing (VD) [4, 40, 47] aims to generate freely editable human images with reference

garments and optional conditions. StableGarment [47] and IMAGDressing [40] leverage a garment U-Net for extracting fine-grained clothing features and a denoising U-Net with a hybrid attention module to incorporate garment features into denoising process. Magic Clothing [4] additionally proposes a joint classifier-free guidance to balance the control of garment features and text prompts. DreamFit [30] proposes a lightweight garment encoder based on trainable LoRA layers to streamline model complexity and memory usage. However, existing approaches are limited to processing single items of clothing, and difficult to maintain fidelity to text prompts. In contrast, our method allows for freely dressing multiple garments and produces coherent and attractive images following customized text prompts.

#### 3. Preliminaries

Stable Diffusion. The Diffusion Model belongs to a class of generative models that generate data through iterative denoising from random noise. In this paper, we specifically employ Stable Diffusion [38]. Stable Diffusion is a latent diffusion model that operates in the latent space of an autoencoder D(E(·)), where E and D represent the encoder and decoder, respectively. For a given image x0 with its corresponding latent feature z0 = E(x0), the diffusion forward process is defined as:

zt = √αtz0 + √1 − αtϵ, (1)

where αt = ts=1(1 − βs), ϵ ∼ N(0,1), and βs is the pre-defined variance schedule at timestep s.

In the diffusion backward process, a U-Net ϵθ is trained to predict the noise. Given the textual condition C, the training objective LLDM is defined as follows:

0,ϵ,C,t∥ϵ − ϵθ(zt,C,t)∥2. (2)

LLDM = Ez

#### 4. Methodology

Given N target garments, the proposed AnyDressing aims to generate a new image xdr, showcasing a customized character dressed in multiple target garments across various scenes, styles and actions based on the text prompt. As illustrated in Fig. 2, AnyDressing comprises two primary networks: GarmentsNet and DressingNet. The GarmentsNet leverages the Garment-Specific Feature Extractor module to extract detailed features from multiple garments (Sec. 4.1). Meanwhile, the DressingNet integrates these features for virtual dressing using a Dressing-Attention module and an Instance-Level Garment Localization Learning mechanism (Sec. 4.2). Additionally, a Garment-Enhanced Texture Learning strategy is designed further to enhance crucial texture details in the synthesis images (Sec. 4.3). Next, we will introduce the aforementioned modules, along with training and inference processes (Sec. 4.4), in detail.

Garment-Enhanced Texture Learning (GTL)

AnyDressing Pipeline

Garment-Specific Feature Extractor (GFE)

[Figure 48]

Train

[Figure 49]

[Figure 50]

[Figure 51]

Frozen

[Figure 52]

[Figure 53]

[Figure 54]

LoRA

LoRA

Q K V

[Figure 55]

[Figure 56]

[Figure 57]

Garment 1

[Figure 58]

[Figure 59]

[Figure 60]

Self Attn

Proj

...

Edge Detector

...

c

###### G F E

G F E

[Figure 61]

[Figure 62]

Garment 1

Garments Net

...

...

G F E

G F E

LoRA LoRA

...

Q K V

...

...

...

...

...

...

...

...

Garment N

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Proj Self Attn

[Figure 68]

[Figure 69]

Segmentation

[Figure 70]

Garment N

Dressing Attention (DA)

[Figure 71]

Garment Attention Maps Garment 1

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

D A

D A

Dressing Net

K

Add noise

D A

D A

...

...

Cross Attn

###### ...

Proj

c

V

Garment N

[Figure 81]

Instance-Level Garment Localization Learning

+

[Figure 82]

[Figure 83]

INPUT

[Figure 84]

OUTPUT

Q

Latent Noise

Self Attn

[Figure 85]

[Figure 86]

K

Proj

[Figure 87]

“A woman is wearing a [black T-shirt] and a [short jeans], against a white background.”

V

Garment Masks

... Garment-Specific G...F E Feature Extractor

Cross Attention c Concat + Add

DA Dressing Attention

Linear Layer

Figure 2. Overview of AnyDressing. Given N target garments, AnyDressing customizes a character dressed in multiple target garments. The GarmentsNet leverages the Garment-Specific Feature Extractor (GFE) module to extract detailed features from multiple garments. The DressingNet integrates these features for virtual dressing using a Dressing-Attention (DA) module and an Instance-Level Garment Localization Learning mechanism. Moreover, the Garment-Enhanced Texture Learning (GTL) strategy further enhances texture details.

###### 4.1. GarmentsNet

Previous methods [4, 40, 47] leverage a full copy of diffusion U-Net [2, 16] as garment encoding network, ensuring precise preservation of clothing details. However, these methods are limited to handling a single garment and face significant garment confusion issues when applied to multigarment virtual dressing, as shown in Fig. 3. A straightforward approach to dress multiple garments is to simply duplicate several garment encoding networks to manage different conditions. However, this method would result in a substantial increase in the number of parameters, making it computationally impractical.

Drawing inspiration from the successful practice of the aforementioned reference mechanisms, we observe that self-attention layers are crucial for the implicit warping of features, enabling the effective matching of input garments to the appropriate body parts. Meanwhile, other layers are typically responsible for general feature extraction and can be shared across different garments without compromising the model’s performance. Building on this insight, we innovatively design a simple yet effective architecture named GarmentsNet, which employs a core Garment-Specific Feature Extractor (GFE) module to encode features for each garment utilizing individual self-attention layers within a shared U-Net framework. Inspired by [30], we integrate LoRA [15] mechanism into self-attention layers, minimizing the increase in parameters associated with the added

garments. As a result, this design significantly avoids garment blending while maintaining network efficiency. As illustrated in Fig. 2, the GFE module employs a parallelized self-attention mechanism to extract detailed features of multiple garments. Specifically, for each garment condition Fi, we define the proprietary self-attention LoRA matrix △Wˆ i as follows:

△Wˆ i = {△Wˆ qi,△Wˆ ki ,△Wˆ vi }, (3)

where △Wˆ qi, △Wˆ ki and △Wˆ vi represent LoRA layers for the query, key and value projections of self-attention layers.

We then concatenate self-attention results of each garment condition to obtain the aggregated garment features Fnew:

Qi(Ki)⊤

Finew = Softmax(

)Vi, (4)

√

d

Fnew = Concat(F1new,F2new,··· ,FNnew), (5)

where Qi = Fi(Wˆ q + △Wˆ qi), Ki = Fi(Wˆ k + △Wˆ ki ), Vi = Fi(Wˆ v + △Wˆ vi ), only △Wˆ is trainable and N represents the number of reference garments.

Thanks to the multi-garment parallel processing design of our GFE module, GarmentsNet can seamlessly scale to any number of garments. Notably, this expansion requires only some newly added LoRA matrix △Wˆ in self-attention layers, and significantly minimizes both training and inference time compared with duplicating the entire garment

encoding network. Considering the capability of the GFE module to individually encode each garment, we excise the cross-attention modules in GarmentsNet to further reduce redundancy.

###### 4.2. DressingNet

To incorporate multi-garment features during the diffusion process, we meticulously design the DressingNet, which serves as the main denoising net and primarily includes an adaptive Dressing-Attention mechanism and an InstanceLevel Garment Localization Learning strategy.

###### 4.2.1 Adaptive Dressing-Attention

In the VD task, the main denoising network is typically kept frozen during training [4, 40] to preserve its original editing and generation capabilities as much as possible. To incorporate reference garment features into latent features, we design an adaptive Dressing-Attention (DA) mechanism to efficiently integrate multi-garment texture cues into synthetic images, inspired by [54]. As shown in Fig. 2, the Dressing-Attention module includes a frozen self-attention module and a learnable cross-attention module. Let {F1,F2,··· ,FN} denote N garment features output by the GarmentsNet at corresponding positions, we first concatenate these features along the spatial dimension to obtain the final garment features: Fall = Concat(F1,F2,··· ,FN). We then introduce two trainable linear projection layers Wk′ and Wv′ to align garment features with latent feature Z. Formally, the output of Dressing-Attention Znew is:

QK⊤

Q(K′)⊤

)V′

√

√

)V+λ∗Softmax(

Znew = Softmax(

d

d

(6) where λ is a hyperparameter ensuring the flexibility of incorporating garment features, and Q = ZWq, K = ZWk, V = ZWv, K′ = FallWk′ , V′ = FallWv′ . Here, Wq, Wk and Wv are frozen self-attention layers. To accelerate the coverage, we initialize the Wk′ , Wv′ with Wk, Wv.

###### 4.2.2 Instance-Level Garment Localization Learning

Although the above Dressing-Attention (DA) mechanism facilitates the integration of multi-garment features, it may result in text-image inconsistency. We argue that this results from the garment’s attention map covering the entire image in the DA module, thereby injecting garment cues into the other irrelevant regions incorrectly. To tackle this issue, we introduce an Instance-Level Garment Localization (IGL) learning strategy, ensuring that each garment instance focuses solely on its corresponding region. Specifically, for each garment feature, we obtain its attention map A with

the latent noise in each layer of the DA module:

√

P = Softmax(Q(K′)⊤/

d), (7)

A =

L

Pj, (8)

j=1

where L denotes the length of corresponding garment features. Then, a regularization term Lloc is applied to explicitly learn attention localization for each garment instance:

N

1 N

∥Ak − Mk∥2, (9)

Lloc =

k=1

where N is the number of garments in the reference image, and Mk represents the reference garment’s segmentation mask. It is worth noting that the proposed IGL learning strategy is applied exclusively during the training phase and does not introduce any additional cost during inference.

###### 4.3. Garment-Enhanced Texture Learning

Generally, diffusion models are merely optimized relying on the mean-squared loss defined in Eqn. 2, which treats all regions of the synthetic image equally, resulting in a struggle to maintain garment consistency, especially in cases of small text and intricate patterns. To synthesize fine-grained textures, we design a Garment-Enhanced Texture Learning (GTL) strategy to strengthen the supervision of attire details in image space, incorporating a perceptual loss Lperc and a high-frequency loss Lhigh−freq.

Before introducing the proposed two losses, we define the generated image as: Iˆ = D(zˆ0), where D denotes the VAE decoder, and zˆ0 is estimated through a single step of inference from the latent zt:

√1 − αtϵθ √αt

zt −

. (10)

zˆ0 =

Considering the one-step inference may produce noisy and flawed images, the proposed losses are only applied at less noisy timestep (t ≤ η). To sum up, GTL can be defined as:

Ltexture = Lperc + Lhigh−freq, t ≤ η 0, t > η

. (11)

Perception Loss To simultaneously enhance structural consistency and pattern similarity with reference garments, we employ a perceptual loss based on the Deep Image Structure and Texture Similarity (DISTS) metric [7]. Specifically, we use the reference garment’s segmentation mask to isolate the attire in both the generated and ground truth images, averaging their structural and textural inconsistencies within a perceptual feature space, defined as:

1 N

Lprec =

N

DIST S(Iˆ⊙ Mk,I ⊙ Mk), (12)

k=1

|Method<br><br>|Single Grament<br><br>VITON-HD [5] Proprietary Dataset| |Multiple Graments Dressing-Pair|
|---|---|---|---|
| | | | |
| |CLIP-T ↑ CLIP-I ↑ CLIP-AS ↑<br><br>|CLIP-T ↑ CLIP-I ↑ CLIP-AS ↑|CLIP-T ↑ CLIP-I∗ ↑ CLIP-AS ↑|

IP-Adapter [54] 0.268 0.644 5.674 0.272 0.632 5.678 0.277 0.523 5.795 StableGarment [47] 0.285 0.583 5.781 0.281 0.587 5.648 0.284 0.556 5.735 MagicClothing [4] 0.288 0.640 5.703 0.298 0.619 5.784 0.266 0.583 5.540 IMAGDressing [40] 0.202 0.734 5.077 0.230 0.684 5.133 0.242 0.614 5.291 Ours 0.289 0.741 5.881 0.296 0.710 5.931 0.296 0.734 5.874

Table 1. Quantitative comparisons with baseline methods for both single-garment and multi-garment evaluation.

where ⊙ signifies element-wise multiplication.

High-Frequency Loss As intricate details in the dressing garments typically appear as high-frequency components with rich edge information, we use edge detection to extract this high-frequency information, aiming to strengthen the constraints on detailed patterns. Specifically, we utilize Canny edge detection operator [8] to capture these rich-texture regions, and define the high-frequency loss Lhigh−freq as:

N

1 N

∥Iˆ⊙ Mk′ − I ⊙ Mk′∥2, (13)

Lhigh−freq =

k=1

where Mk′ = Mk ⊙ P, P is the extracted edge map of I.

- 4.4. Training and Inference

In training, we average Lloc across all m layers and define overall loss L as follows:

LLDM = Ez

0,ϵ,Ct,Cg,t∥ϵ − ϵθ(zt,Ct,Cg,t)∥2, (14) L = LLDM +

λ1 m Lloc + λ2Ltexture, (15)

where Ct and Cg represent text condition and clothing condition respectively. In the inference stage, we apply classifier-free guidance during the denoising process:

ϵˆθ(zt,Ct,Cg,t) = ωϵθ(zt,Ct,Cg,t) + (1 − ω)ϵθ(zt,t).

(16)

- 5. Experiments

- 5.1. Setup

Dataset. Notably, a dataset comprising image triplets that include model images paired with multiple laid-out garments is currently lacking. Therefore, we utilize a HumanParsing model to extract clothing items from DressCode [34] and an additional proprietary dataset collected from the internet, forming triplet data pairs (upper garment, lower garment, person image). In these triplets, one garment is an original laid-out image, while the other is a segmented image from the person’s image. Finally, we construct 26,114 public triplets from Dresscode and 37,065

Method ConsistencyTexture ↑ AlignPromptwith↑ QualityImage↑ ComprehensiveEvaluation ↑

IP-Adapter [54] 0.45% 6.65% 11.95% 2.20% StableGarment [47] 1.60% 4.85% 2.65% 2.05% MagicClothing [4] 2.05% 9.00% 9.70% 3.75% IMAGDressing [40] 2.10% 2.50% 3.90% 1.70% Ours 93.80% 77.00% 71.80% 90.30%

Table 2. User study with baseline methods.

triplets from proprietary dataset to train AnyDressing. For model evaluation, we introduce two benchmarks to evaluate the model on single-garment and multi-garment dressing respectively. Specifically, for single-garment evaluation, we select 300 reference garments from VITON-HD [5] encompassing various styles and colors, and additionally collect 300 diverse garments with intricate textures from the internet. For multi-garment evaluation, we meticulously gather 25 lowers from the internet and pair each with 10 different uppers, resulting in a total of 250 pairs, called DressingPair. We generate images for each test garment with the provided 7 text prompts.

Implementation Details. In our experiments, we initialize the weights of GarmentsNet and DressingNet with the weights of the U-Net in Stable Diffusion v1.5 [38]. Our model is trained on paired images at the resolution of 768 × 576. The trainable parameters are GarmentsNet and the cross-attention layers in Dressing-Attention module. During training, We adopt AdamW [32] optimizer with a fixed learning rate of 5e-5. The model is trained for 100k steps on 8 NVIDIA A100 GPUs with a batch size of 4. During inference, we use DDIM [43] sampler with 30 steps and set guidance scale ω to 6.0. Please refer to the supplementary materials for more details.

Baselines. We compare our method against the following state-of-the-art image synthesis method: IP-Adapter [54], MagicClothing [4], StableGarment [47] and IMAGDressing [40]. We use the official model parameters from their official implementations. For a fair comparison, all experiments are conducted with the resolution of 768 × 576.

Evaluation Metrics. We follow previous methods to adopt three metrics for evaluation: CLIP-T for text-image similarity, CLIP-I for garment consistency, and CLIP Aesthetic Score (CLIP-AS) for overall generation quality. Especially,

[Figure 88]

[Figure 89]

Single-Garment Dressing Multi-Garment Dressing

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

”A girl, red hair, wearing a hat, by a fence”

”A girl, red hair, wearing a hat, by a fence”

”A girl, winter, night, snow”

”A girl, sky, in a field of flowers”

”A girl sitting on the grass”

”A girl with a bag in the classroom”

”A girl sitting on the grass”

”A girl, on the stage”

”A man, in the gym”

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

IMAGDressingMagicClothingStableGarmentIP-Adapter

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

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

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Figure 3. Qualitative comparisons with state-of-the-art methods. Please zoom in for more details.

to better evaluate multi-garment dressing, we introduce a new metric CLIP-I∗ to assess texture consistency by leveraging OpenPose [1] to obtain the matching partitions of the reference garments in the synthesized image and averaging their CLIP-I metrics.

###### 5.2. Qualitative Analysis

Since the compared methods lack multi-garment support, we obtain baseline results by concatenating multiple garments along the spatial dimension as input. Fig. 3 presents visual comparisons between our method and baseline ap-

proaches. AnyDressing maintains superior consistency in clothing style and texture, and exhibits better text fidelity, while other methods struggle to balance garment preservation and prompt faithfulness. In particular, baselines encounter significant background contamination and garment confusion in multi-garment dressing results, whereas our method demonstrates exceptional reliability, which is attributed to our designed GarmentsNet and DressingNet architectures. And Fig. 4 presents the results of AnyDressing as a plug-in module combined with other extensions and customized LoRAs, demonstrating its powerful compatibil-

Base+GFE Base+GFE+IGL

with IP-Adapter-FaceID

Base

”A girl, in the gym” ”A girl, on the beach”

”A girl, in the library”

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

”A girl, winter, sunshine”

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

with ControlNet & IP-Adapter-FaceID

”A girl, in the gym”

[Figure 170]

”A woman, in the theater”

”A man, on Mars”

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

”A girl sitting on the grass”

[Figure 185]

with LoRA

”Crayon Shin-chan, doll” ”Blmpony, on the street” ”Saiyans, Nike shoes, sea”

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Figure 5. Ablation results on GFE and IGL modules.

[Figure 192]

[Figure 193]

[Figure 194]

w/o GTL w GTL

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Figure 4. Examples of plug-in results of AnyDressing.

[Figure 199]

ity. Please refer to the supplementary for more results.

###### 5.3. Quantitative Comparisons

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Metric Evaluation. Tab. 1 shows the quantitative results of our methods against baselines. For single-garment evaluation, extensive experiments conducted on VITON-HD [5] and proprietary dataset prove the superiority of AnyDressing compared with all baselines. And our method significantly surpasses all baselines across all metrics in multigarment virtual dressing results, fully demonstrating AnyDressing’s reliability in handling both single-garment and multi-garment virtual dressing tasks.

[Figure 204]

Figure 6. Ablation results on GTL module. GFE IGL GTL CLIP-T ↑ CLIP-I∗ ↑ CLIP-AS ↑

User Study. We conduct a user study to evaluate the generation quality of our model. We use all test garments and prompts in our dataset and randomly show the users 25 single-garment results and 25 multi-garment results from the baselines and our method. Each participant is asked to select the most preferred result under four criteria: texture consistency, alignment with the text prompt, image quality and comprehensive evaluation. In the end, we receive valid responses from 40 users. The collected preferences are reported in Tab. 2. In terms of four criteria, our method is preferred by most participants, with percentages reaching 93.80%, 77.00%, 71.80% and 90.30% respectively.

0.260 0.625 5.572 0.265 0.718 5.627 0.289 0.722 5.790 0.296 0.734 5.874

Table 3. Ablation study of AnyDressing.

encode multiple garments concurrently and then incorporate them into the denoising U-Net similar to [40] as our base model. As illustrated in Fig. 5, Base+GFE significantly reduces garment confusion and improves garment consistency compared to Base, which is attributed to the multi-garment parallel processing design of the GFE module. Base+GFE+IGL shows better fidelity to the text prompts and further mitigates background contamination, which demonstrates IGL mechanism effectively constrains garment features to attend to the correct regions. The quan-

###### 5.4. Ablation Studies

GFE & IGL. To validate the effectiveness of our proposed architecture, we employ traditional ReferenceNet [16] to

titative comparison in Tab. 3 further proves the effectiveness of each module, with GFE primarily improving the CLIP-I∗ and IGL enhancing both CLIP-T and CLIP-AS.

GTL. Fig. 6 intuitively demonstrates the effectiveness of our proposed GTL strategy, encouraging the model to enhance detail preservation, particularly in small text and intricate patterns. And quantitative result in Tab. 3 also verifies that our designed GTL improves texture consistency.

#### 6. Conclusion

This paper presents AnyDressing comprising two core networks named GarmentsNet and DressingNet to focus on a new task, i.e., Multi-Garment Virtual Dressing. The GarmentsNet employs a Garment-Specific Feature Extractor module to efficiently encode multi-garment features in parallel. The DressingNet integrates these features for virtual dressing using a Dressing-Attention module and an Instance-Level Garment Localization Learning mechanism. Additionally, we design a GarmentEnhanced Texture Learning strategy to further enhance texture details. Our approach can seamlessly integrate with any community control plugins. Extensive experiments show that AnyDressing achieves state-of-the-art results.

#### References

- [1] Z Cao, G Hidalgo, T Simon, SE Wei, and Y Sheikh. Openpose: Realtime multi-person 2d pose estimation using part affinity fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(1):172–186, 2020. 7
- [2] Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Xiao Yang, and Mohammad Soleymani. Magicdance: Realistic human dance video generation with motions & facial expressions transfer. arXiv preprint arXiv:2311.12052, 2023. 4
- [3] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023. 2
- [4] Weifeng Chen, Tao Gu, Yuhao Xu, and Chengcai Chen. Magic clothing: Controllable garment-driven image synthesis. arXiv preprint arXiv:2404.09512, 2024. 2, 3, 4, 5, 6
- [5] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14131–14140, 2021. 3, 6, 8, 1
- [6] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. arXiv preprint arXiv:2403.05139, 2024. 3
- [7] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5):2567–2581, 2020. 5

- [8] Lijun Ding and Ardeshir Goshtasby. On the canny edge detector. Pattern recognition, 34(3):721–725, 2001. 6
- [9] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34:19822–19835, 2021. 2
- [10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2, 3
- [11] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7599–7607, 2023. 3
- [12] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 3
- [13] Sen He, Yi-Zhe Song, and Tao Xiang. Style-based global appearance flow for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3470–3479, 2022. 3
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 1, 2, 3, 4, 12
- [16] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 4, 8, 2
- [17] Qihan Huang, Siming Fu, Jinlong Liu, Hao Jiang, Yipeng Yu, and Jie Song. Resolving multi-condition confusion for finetuning-free personalized image generation. arXiv preprint arXiv:2409.17920, 2024. 3
- [18] Ziqi Huang, Tianxing Wu, Yuming Jiang, Kelvin CK Chan, and Ziwei Liu. Reversion: Diffusion-based relation inversion from images. arXiv preprint arXiv:2303.13495, 2023. 2
- [19] Thibaut Issenhuth, J´er´emie Mary, and Cl´ement Calauzenes. Do not mask what you do not need to mask: a parser-free virtual try-on. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XX 16, pages 619–635. Springer, 2020. 3
- [20] Zhenchao Jin. Sssegmenation: An open source supervised semantic segmentation toolbox based on pytorch. arXiv preprint arXiv:2305.17091, 2023. 1
- [21] Zhenchao Jin, Xiaowei Hu, Lingting Zhu, Luchuan Song, Li Yuan, and Lequan Yu. Idrnet: Intervention-driven relation network for semantic segmentation. Advances in Neural Information Processing Systems, 36, 2024. 1

- [22] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10124–10134, 2023. 2
- [23] Chanran Kim, Jeongin Lee, Shichang Joung, Bongmo Kim, and Yeul-Min Baek. Instantfamily: Masked attention for zero-shot multi-id image generation. arXiv preprint arXiv:2404.19427, 2024. 3
- [24] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8176–8185, 2024. 3
- [25] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 2, 3
- [26] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In European Conference on Computer Vision, pages 204–219. Springer, 2022. 3
- [27] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024. 3
- [28] Kedan Li, Min Jin Chong, Jeffrey Zhang, and Jingen Liu. Toward accurate and realistic outfits visualization with attention to details. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15546– 15555, 2021. 3
- [29] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024. 3
- [30] Ente Lin, Xujie Zhang, Fuwei Zhao, Yuxuan Luo, Xin Dong, Long Zeng, and Xiaodan Liang. Dreamfit: Garment-centric human generation via a lightweight anything-dressing encoder. arXiv preprint arXiv:2412.17644, 2024. 2, 3, 4
- [31] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple subjects. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 57500–57519, 2023. 2
- [32] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [33] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subjectdiffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2, 3
- [34] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: Highresolution multi-category virtual try-on. In Proceedings of

- the IEEE/CVF conference on computer vision and pattern recognition, pages 2231–2235, 2022. 3, 6, 1
- [35] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8580–8589, 2023. 3
- [36] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4296–4304, 2024. 3
- [37] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 6
- [39] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 3
- [40] Fei Shen, Xin Jiang, Xin He, Hu Ye, Cong Wang, Xiaoyu Du, Zechao Li, and Jinghui Tang. Imagdressing-v1: Customizable virtual dressing. arXiv preprint arXiv:2407.12705,

2024. 2, 3, 4, 5, 6, 8

- [41] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8543–8552, 2024. 2
- [42] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2
- [43] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 6
- [44] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [45] Yael Vinker, Andrey Voynov, Daniel Cohen-Or, and Ariel Shamir. Concept decomposition for visual exploration and inspiration. ACM Transactions on Graphics (TOG), 42(6): 1–13, 2023. 2
- [46] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristicpreserving image-based virtual try-on network. In Proceedings of the European conference on computer vision (ECCV), pages 589–604, 2018. 3

- [47] Rui Wang, Hailong Guo, Jiaming Liu, Huaxia Li, Haibo Zhao, Xu Tang, Yao Hu, Hao Tang, and Peipei Li. Stablegarment: Garment-centric generation via stable diffusion. arXiv preprint arXiv:2403.10783, 2024. 2, 3, 4, 6
- [48] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 1
- [49] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023. 2
- [50] Zhichao Wei, Qingkun Su, Long Qin, and Weizhi Wang. Mm-diff: High-fidelity image personalization via multi-modal condition integration. arXiv preprint arXiv:2403.15059, 2024. 3
- [51] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. International Journal of Computer Vision, pages 1–20, 2024. 3
- [52] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gpvton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23550–23559, 2023. 3
- [53] Y Xu, T Gu, W Chen, and C Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arxiv 2024. arXiv preprint arXiv:2403.01779, 2024. 3
- [54] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 1, 2, 3, 5, 6, 11

- [55] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 1, 2, 3, 11
- [56] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024. 3
- [57] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single image editing with textto-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6027–6037, 2023. 2

## AnyDressing: Customizable Multi-Garment Virtual Dressing via Latent Diffusion Models

### Supplementary Material

In the supplementary material, the sections are organized as follows:

- • We provide more details regarding parameters, datasets and user study in Sec. 7.
- • We further prove the scalability of AnyDressing in Sec. 8.
- • We provide more ablation results in Sec. 9.
- • We provide more comparisons with baselines, more qualitative results in the wild and more applications in Sec. 10.

#### 7. Implementation Details

###### 7.1. Detailed Parameters

In our experiments, we use SOTA large multi-modal model CogVLM [48] to caption the image. GarmentsNet requires only one step forward process before the multiple denoising steps in DressingNet, causing a minimal amount of extra computational cost. The hyper-parameters used in our experiments are set as follows:

- • For the Dressing-Attention mechanism, we set the hyperparameter λ = 0.7 during inference to get customized results.
- • For the noisy timestep threshold discussed in the Garment-Enhanced Texture Learning (GTL) strategy, we set η = 350.
- • The other hyper-parameters used in the experiment are as follows: λ1 = 0.01, λ2 = 0.001.

###### 7.2. Datasets

To facilitate research on multi-garment virtual dressing, a dataset consisting of image triplets is necessary, with each triplet containing an upper garment image, a lower garment image, and a model image wearing the corresponding garments. However, existing in-shop garment to model pairs [5, 34] only contain a single reference garment. We leverage the public DressCode dataset along with a proprietary dataset to construct triplets, as illustrated in Fig. 7. Assuming we begin with the upper garment data, where we already have an in-shop upper garment and a model image wearing it, we employ human parsing techniques [20, 21] to roughly segment and extract the lower garment portion from the model image, using it as the corresponding lower garment image. At this stage, the triplet comprises an inshop upper garment image, a cropped lower garment image, and a model image. Similarly, triplets derived from the lower garment data consist of a cropped upper garment image, an in-shop lower garment image, and a model image. Finally, we constructed 26,114 public triplets from Dress-

code and 37,065 triplets from the proprietary dataset to train AnyDressing.

It is worth noting that our model has not encountered garment pairs in the form of (in-shop upper garment, in-shop lower garment) or (cropped upper garment, cropped lower garment) during training. Nevertheless, it exhibits strong robustness during inference, indicating that the model has effectively learned the proper way to combine upper and lower garments through training.

###### 7.3. User Study

To compare with the baseline methods, we conduct a user study as part of the evaluation. The survey randomly presented 50 sets of generated results to each participant. A screenshot of the survey for a set of generated results is displayed in Fig. 8, which includes five images and four questions:

- 1. Which result appears to have the highest consistency with reference garments?
- 2. Which result best matches the prompt ‘[prompt]’?
- 3. Which result appears to have the highest image quality?
- 4. Which result matches your best choice based on comprehensive considerations? For each set of results displayed in the survey, we en-

sured that their order was randomly shuffled to prevent bias. Responses where all answers had the same selection and responses with completely identical answers were considered invalid. Finally, we obtained a total of 40 valid surveys to evaluate the model.

#### 8. Scalability of AnyDressing

To further validate the scalability of our designed GarmentsNet structure, we introduce more combinations of clothing items (hat, upper garment and lower garment), as illustrated in Fig. 10. As shown in Fig. 9, to train the model, we construct datasets using the same idea as introduced in Sec. 7.2. Specifically, we select 18,059 pairs from the proprietary dataset that satisfies the model image containing the hat, and use the human parsing techniques to obtain the cropped hat image from the model image.

Notably, each additional garment condition requires only some newly added LoRA matrix △Wˆ in the GarmentSpecific Feature Extractor (GFE) module. And it requires only a single forward pass (timestep t = 0) to encode the clothing before injecting features into the DressingNet, minimizing the additional computational time during both

the training and inference process. This experiment effectively demonstrates that our GarmentsNet can be extended to accommodate any number of clothing items. Additionally, thanks to our proposed Instance-Level Garment Localization (IGL) learning mechanism, AnyDressing can further prevent garment blending and enhance fidelity to customized text prompts.

#### 9. More Ablation Study

In Fig. 11, we present additional visual results to validate the effectiveness of the Garment-Specific Feature Extractor (GFE) module and the Instance-Level Garment Localization (IGL) learning mechanism. We employ traditional ReferenceNet [16] to encode multiple garments concurrently and then incorporate them into the denoising U-Net similar to [4, 40] as our base model. As shown in Fig. 11, Base model encounters severe clothing confusion issues, resulting in the colors and patterns of multiple garments blending. In contrast, Base+GFE significantly reduces garment confusion and improves garment consistency, which is attributed to the multi-garment parallel processing design of our designed GFE module. Base+GFE+IGL shows better fidelity to the text prompts and further mitigates background contamination, which demonstrates IGL mechanism effectively constrains garment features to attend to the correct regions and avoid influencing other irrelevant regions in the synthetic images.

#### 10. More Results

###### 10.1. More Comparisons

As shown in Fig. 12-13, We provide more visual comparisons between our method and state-of-the-art baselines [4, 40, 47, 54]. It is clear from these comparisons that our method maintains superior consistency in clothing style and texture, and exhibits better text fidelity.

###### 10.2. More Visual Results

As shown in Fig. 14-16, we provide more multi-garment virtual dressing results of AnyDressing in the wild. It can be observed that our method produces high-quality customized virtual dressing results for various types of garment combinations, while faithfully adhering to personalized text prompts. Experiments in complex scenarios demonstrate that AnyDressing significantly enhances the practical application of Virtual Dressing in e-commerce and creative design.

###### 10.3. More Applications

Combined with ControlNet. Leveraging the capabilities of ControlNet [55], our model can generate personalized models guided by specific conditions. We present the OpenPose-guided generation results in Fig. 17.

Combined with IP-Adapter. Our model enables the generation of target individuals wearing specified garments integrated with the IP-Adapter. We utilize the ID preservation capability of FaceID [54] to provide an authentic virtual dressing experience. The visual results, as shown in Fig. 17.

Stylized Customization. Furthermore, by utilizing stylized base models or customized LoRAs [15], we can generate creative and stylized outputs while preserving the intricate details of the garments, as shown in Fig. 16 and Fig. 18.

##### Triplets from Upper-Garment Data Triplets from Lower-Garment Data

[Figure 205]

Model Image In-shop Upper Cropped Lower Model Image Cropped Upper In-shop Lower

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Figure 7. Examples of the training dataset I.

[Figure 217]

[Figure 218]

Figure 8. Screenshot of user study.

###### Triplets from Upper-Garment Data Triplets from Lower-Garment Data

Model Image In-shop Upper Cropped Lower Model Image Cropped Upper In-shop Lower

Cropped Hat Cropped Hat

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Figure 9. Examples of the training dataset II.

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

”A girl, red hair, wearing a hat, by a fence”

”A girl, sky, in a field of flowers”

”A girl sitting on the grass”

”A girl, in the gym”

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

###### Figure 10. Qualitative results of more combinations of clothing items.

Base+GFE Base+GFE+IGL

Base

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

”A girl, winter, night, snow”

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

”A girl with a bag in the classroom”

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

”A girl, on the stage”

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

”A girl, red hair, wearing a hat, by a fence”

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

”A girl, winter, night, snow”

###### Figure 11. More ablation results on GFE and IGL modules.

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

”A girl, red hair, wearing a hat, by a fence”

”A girl, winter, night, snow”

”A girl, sky, in a field of flowers”

”A girl with a bag in the classroom”

”A girl, in the gym”

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

IP-AdapterIMAGDressingMagicClothingStableGarment

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

###### Figure 12. More qualitative comparisons I.

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

”A girl, red hair, wearing a hat, by a fence”

”A girl with a bag in the classroom”

”A girl, sky, in a field of flowers”

”A girl, on the stage”

”A girl, in the snow”

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

IP-AdapterIMAGDressingMagicClothingStableGarment

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

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

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

###### Figure 13. More qualitative comparisons II.

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

”A girl, golden hair, simple background”

”A girl, in a garden full of flowers”

”A girl, in front of the Eiffel Tower”

”A girl, At the outdoor stadium”

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

###### Figure 14. More qualitative results I.

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

”A man, running in the park, morning”

”A man, in a high-rise office building”

”A man, on the street”

”A man, by the sea, sandbeach”

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

###### Figure 15. More qualitative results II.

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

”A girl, on the balcony of a grand medieval castle”

”A girl, standing at the helm of a pirate ship”

”A girl, under the full moon on a grassy field”

”A girl, in a magical forest”

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

###### Figure 16. More qualitative results III.

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

###### Figure 17. More results of combining ControlNet [55] and FaceID [54].

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

###### Figure 18. More results of combining LoRAs [15].

