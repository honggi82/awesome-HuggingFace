# arXiv:2502.05979v4[cs.CV]1Apr2025

## VFX Creator: Animated Visual Effect Generation with Controllable Diffusion Transformer

XINYU LIU, Hong Kong University of Science and Technology, China AILING ZENG, Tencent AI Lab, China WEI XUE, Hong Kong University of Science and Technology, China HARRY YANG, Hong Kong University of Science and Technology, China WENHAN LUO, Hong Kong University of Science and Technology, China QIFENG LIU, Hong Kong University of Science and Technology, China YIKE GUO, Hong Kong University of Science and Technology, China

Mask Reference Image

[Figure 1]

[Figure 2]

[Figure 3]

Start frame：2 End frame：24

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Ta-daitLevitateit

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Crumble it

Start frame：13 End frame：49

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Squish it

Start frame：3 Start frame：40

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

Explode it

[Figure 45]

4f 00:06

00:00 4f 00:01 4f 00:02 4f 00:03 4f 00:04 4f 00:05

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Start frame：6 Start frame：44

Start frame：4 End frame：

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Ta-daitLevitateit

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Decapitate it

Start frame：23 End frame：49

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Melt it

Start frame：1 End frame：49

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Inflate it

Temporal Control

Spatial Control

Fig. 1. VFX Creator is an efficient framework based on a Video Diffusion Transformer, enabling spatial and temporal control for visual effect (VFX) video generation. With minimal training data, a plug-and-play mask control module allows precise instance-level manipulation, while the integration of tokenized start-end motion timestamps with text space provides fine-grained temporal control over the VFX rhythm.

Crafting magic and illusions stands as one of the most thrilling facets of filmmaking, with visual effects (VFX) serving as the powerhouse behind unforgettable cinematic experiences. While recent advances in generative artificial intelligence have catalyzed progress in generic image and video synthesis, the domain of controllable VFX generation remains comparatively underexplored. More importantly, fine-grained spatial-temporal controllability in VFX generation is critical, but challenging due to data scarcity, complex dynamics, and precision in spatial manipulation. In this work, we propose a novel paradigm for animated VFX generation as image animation, where dynamic effects are generated from user-friendly textual descriptions and static reference images. Our work makes two primary contributions: i)

Authors’ addresses: Xinyu Liu, Hong Kong University of Science and Technology, China, xliugd@connect.ust.hk; Ailing Zeng, Tencent AI Lab, China, ailingzengzzz@gmail.com; Wei Xue, Hong Kong University of Science and Technology, China, weixue@ust.hk; Harry Yang, Hong Kong University of Science and Technology, China, yangharry@ust. hk; Wenhan Luo, Hong Kong University of Science and Technology, China, whluo@ust. hk; Qifeng Liu, Hong Kong University of Science and Technology, China, liuqifeng@ ust.hk; Yike Guo, Hong Kong University of Science and Technology, China, yikeguo@ ust.hk.

Open-VFX, the first high-quality VFX video dataset spanning 15 diverse effect categories, annotated with textual descriptions, instance segmentation masks for spatial conditioning, and start–end timestamps for temporal control; This dataset features a wide range of subjects for the reference images, including characters, animals, products, and scenes. ii) VFX Creator, a simple yet effective controllable VFX generation framework based on a Video Diffusion Transformer. The model incorporates a spatial and temporal controllable LoRA adapter, requiring minimal training videos. Specifically, a plug-and-play mask control module enables instance-level spatial manipulation, while tokenized start-end motion timestamps are embedded in the diffusion process accompanied by the text encoder, allowing precise temporal control over effect timing and pace. Extensive experiments on the Open-VFX test set with unseen reference images demonstrate the superiority of the proposed system to generate realistic and dynamic effects, achieving state-of-the-art performance and generalization ability in both spatial and temporal controllability. Furthermore, we introduce a specialized metric to evaluate the precision of temporal control. By bridging traditional VFX techniques with generative techniques, the proposed VFX Creator unlocks

new possibilities for efficient, user-friendly, and high-quality video effect generation, making advanced VFX accessible to a broader audience.

1 INTRODUCTION

Visual effects (VFX) video generation is paramount in video production, particularly in cinema, gaming, and virtual reality, where it enhances visual impact and improves creative efficiency [Adobe nd]. Visual effects combine live action footage with generated imagery to create realistic environments, objects, animals, and creatures that would be dangerous, expensive, impractical, or impossible to capture on film. While early visual effects involved experimentation with film stock, modern techniques include animation, computergenerated imagery (CGI), and other post-production methods [Chabanova 2022]. However, these approaches often involve high computational costs, long production cycles, and significant manual intervention. With the rapid development of diffusion models [Blattmann et al. 2023; Ho et al. 2020], visual effects generation is progressively transitioning from traditional techniques to generative models.

Recent emerging models [Lin et al. 2024; Polyak et al. 2024] have impressive video generation capabilities, showcasing strong temporal consistency and visually appealing effects. These advancements offer great potential for artists, enabling the creation of stunning videos with minimal input, such as images or text prompts. However, the field of VFX generation for video remains underexplored, with existing open-source models struggling to produce complex effects and effectively control motion generation from text prompts.

In contrast, closed-source products such as Pika [Pika 2023] and PixVerse [Pixverse 2023] have proven their ability to generate a wide array of striking visual effects using diffusion-based generative models. These platforms can create effects such as realistic explosions, anti-gravity phenomena, and cinematic character transformations without manual modeling or lengthy production timelines. Despite their power, these proprietary platforms are limited by restricted access to their visual effects resources, which hinders broader development and exploration within this domain.

In this work, we propose a novel paradigm for animated VFX generation, such as image animation, where dynamic effects are generated from user-friendly textual descriptions and static reference images. We introduce two primary contributions to address the challenges and limitations of VFX video generation. First, we present Open-VFX, the first high-quality generated VFX video dataset comprising 675 videos across 15 distinct effect categories, sourced from two commercial platforms: Pika and PixVerse. The dataset includes a diverse range of subjects—characters, animals, products, and scenes—with a minimum resolution of 700x1000 pixels. Additionally, it contains 245 static images from Pexels [Pexels 2024], annotated with textual descriptions, frame-level masks, and normalized start-end timestamps. This wide-ranging dataset provides extensive reference material for the generation of VFX across various domains. Second, we propose VFX Creator, a simple yet effective controllable VFX generation framework based on a Video Diffusion Transformer [Yang et al. 2024], as shown in Fig. 1. The model incorporates a spatial and temporal controllable LoRA adapter, enabling high-quality video generation with minimal training data. For spatial control, we integrate video mask sequences as

conditions with the latent video noise, facilitating instance-level spatial manipulation. For temporal control, we tokenize the start-end motion timestamps and integrate them into the diffusion process alongside the text encoder, allowing for precise control over the timing and pacing of the effects. Finally, we conduct comprehensive experiments on the Open-VFX dataset to demonstrate the model’s generation capabilities. We also introduce a specialized evaluation metric to assess the precision of temporal control, showcasing the model’s ability to generate dynamic, temporally consistent VFX. Due to the data efficiency of VFX Creator, our framework can be easily adapted for fine-tuning across different categories of visual effects. This flexibility enables the rapid generation of a broad array of visual effects videos, significantly reducing the cost and time typically associated with traditional VFX production in the film industry. In summary, our contributions are as follows:

- (1) We present the Open-VFX, the first high-quality VFX video dataset spanning 15 diverse effect categories, annotated with text prompt, instance segmentation masks for spatial conditioning, and start–end timestamps for temporal control.
- (2) We propose the VFX Creator, a simple yet effective controllable VFX generation framework based on a Video Diffusion Transformer. The model incorporates a spatial and temporal controllable LoRA adapter, enabling precise manipulation.
- (3) We perform a comprehensive evaluation on the Open-VFX dataset, showcasing that the proposed system surpasses existing methods in generating visual effects. Additionally, we introduce a novel metric specifically designed to assess the precision of temporal control in the generated effects.

2 RELATED WORK 2.1 General Video Generation

The rapid advancement of video generative models is driven by diffusion models (DMs) [Ho et al. 2020; Nichol et al. 2021; Rombach et al. 2022; Sohl-Dickstein et al. 2015; Song et al. 2020], which enable innovative approaches in video generation. A key architecture is the Diffusion Transformer [Peebles and Xie 2023], leveraging transformer designs to capture long-range dependencies, enhancing temporal consistency and dynamics, and multi-resolution synthesis [Brooks et al. 2024; Chen et al. 2023; Kuaishou 2024; Ma et al. 2024b; Shao et al. 2024; Team 2024; Yang et al. 2024]. For example, CogVideoX [Yang et al. 2024] uses a 3D full attention mechanism for spatial and temporal coherence, while Hunyuan-DiT [Li et al. 2024] introduces large pre-trained models for rich contextual detail.

Furthermore, controllable video generation has garnered considerable attention due to its promising applications in video editing and content creation. For instance, LAMP [Wu et al. 2023a] focuses on transferring information from the first frame to subsequent frames, ensuring the consistency of the initial image throughout the video sequence; however, it is constrained by fixed motion patterns in a few-shot setting. Recent efforts have sought to enhance control over generative models by integrating additional neural networks into diffusion frameworks. ControlNet [Zhang et al. 2023a] directs image generation based on control signals by replicating specific

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

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

(a) Overview of the subjects in the Open-VFX Dataset (b) Word cloud of VFX text prompts

Explode it

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Reference Image

(c) An example of the exploding visual effect

Fig. 2. Overview of our proposed Open-VFX Dataset. (a) demonstrates diverse input inference images in the dataset, including humans, animals, objects, and various scenes across single and multiple components. (b) shows the text descriptions of the proposed 15 VFXs, and (c) presents an example (Explode it) VFX.

layers from pre-trained models and connecting them with zero convolutions [Wang et al. 2024a]. However, the field of controllable visual effect video generation has yet to be explored.

- 2.2 LoRA-Based Video Generation

Recent advancements in fine-tuning methods [Guo et al. 2023; Ouyang et al. 2024; Wang et al. 2024b,c; Wu et al. 2024a,b] for video generation are often influenced by image customization techniques, especially LoRA [Hu et al. 2021]. For example, Tune-A-Video [Wu

- et al. 2023b] extends a text-to-image model by introducing spatialtemporal attention and selectively training specific parts of the attention layers. Similarly, [Materzynska et al. 2023] focus on finetuning only certain model components, with an emphasis on training earlier denoising steps to capture general motion rather than intricate appearance details. MotionDirector[Zhao et al. 2025] proposes a dual-path LoRA architecture and an appearance-debiased temporal loss to decouple appearance and motion. Likewise, methods like [Ren et al. 2024; Wei et al. 2024; Zhang et al. 2023b] employ separate branches for appearance and motion. VMC [Jeong et al. 2024] adapts temporal attention layers by utilizing a motion distillation strategy, employing residual vectors between consecutive noisy latent frames to serve as motion references. Nevertheless, they have not yet investigated the combination of temporal and spatial controllability with LoRA to enhance the model’s controllability.

3 DATASET

- 3.1 Definition of Visual Effects

Visual effects (VFX) can create realistic environments and characters that are difficult or impossible to capture during filming. For example, VFX involves compositing techniques to combine different visual elements into a single scene, often through green screen or

digital background replacement. Additionally, digital effects such as explosions, smoke, and weather simulations help create dynamic and immersive environments. Motion capture technology is used to animate characters or creatures, while digital makeup effects transform actors into fantastical characters. In light of these various considerations, we have curated a set of 15 distinct visual effects, including Cake-ify, Crumble, Crush, Decapitate, Deflate, Dissolve, Explode, Eye-pop, Inflate, Levitate, Melt, Squish, Ta-da, Transformer into Venom, and Transformer into Harley Quinn. As shown in Fig. 2, we demonstrate the overview of our Open-VFX Dataset, and these effects are designed to deliver a more immersive and visually compelling experience for the audience. Detailed descriptions of these effects can be found in the supplementary materials.

- 3.2 Source Videos

The Open-VFX dataset comprises 675 high-quality VFX videos sourced from two commercial platforms, Pika 1.5 [Pika 2023] and PixVerse 3.0 [Pixverse 2023], with each video having an average duration of 5 seconds. Spanning 15 distinct categories of visual effects, the dataset also includes 245 reference images collected from the Pexels [Pexels 2024] community, featuring both single and multiple objects. As shown in Fig. 3, all videos have a resolution of at least 700 × 1000 pixels and are synthesized at 24 frames per second (fps).

- 3.3 Data Annotation

To accomplish the spatial-temporal controlled VFX video generation task, we adhered to the following process to acquire timestamp and mask annotations for our Open-VFX dataset:

Start-end motion timestamps. To accurately capture the start and end timestamps of motion, traditional optical flow methods fall short

Cake-ify it Crumble it

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Crush it Decapitate it

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Levitate it Melt it

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Squish it Ta-da it

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Transform into Harley Quinn, mastering allure and chaos Transform into a black Venom

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Fig. 3. More examples of our Open-VFX dataset, including 10 VFXs and diverse reference images.

in tracking the dynamic movement of visual effects due to their inherent limitations. Consequently, we employ Co-Tracker [Karaev

- et al. 2024] for timestamp labeling. By monitoring the displacement of tracked points, motion is deemed to begin when the coordinates start to shift, and it concludes when the displacement ceases, providing precise temporal boundaries for the motion.

Instance-level mask sequences. To enable the model to generate visual effects specifically tailored to the selected object, we utilize SAM2 [Ravi et al. 2024] to semantically annotate the motion of the chosen object, producing corresponding mask sequences. We generate several times to obtain diverse generated videos with different animated objects, and we annotate them via masks. During inference, SAM2 is employed to generate binary region masks based on the user-defined area, which serve as spatial conditions to guide precise control over the spatial manipulation of the object.

4 NETWORK 4.1 Preliminary

We introduce the preliminary of CogVideoX architecture(Rombach et al., 2022), the baseline diffusion transformer Network used in our work, and Low-Rank Adaptation (LoRA) (Hu et al., 2021), which helps understand the spatial and temporal controllable LoRA adapter.

4.1.1 Baseline Diffusion Transformer Network. VFX Creator builds upon the CogVideoX architecture [Yang et al. 2024] and leverages a causal 3D Variational Autoencoder (VAE) [Kingma 2013] for video compression, achieving temporal and spatial factors of 4 and 8, respectively. Latent variables are structured as sequential inputs, while textual information is encoded into embeddings using the T5 Encoder [Raffel et al. 2020]. These inputs are processed jointly within a stacked Expert Transformer network, which integrates Adaptive Layer Normalization for better alignment and 3D Rotary Positional

Spatial Control Module

Trainable Modules

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Frozen Modules

Spatial ControlNet

+ Sum 𝑐 Concatenate

+

Mask Condition

Training Video

DiT Block 0

DiT Blocks 1,… N

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

3DFullAttention

3DFullAttention

Scale&Shift

FeedForward

Scale&Shift

FeedForward

−𝑁×1

Patchify

Gate

Gate

Gate

Gate

3D VAE

| | |
|---|---|
|Text Embedding| |

[Figure 172]

###### LoRA Adapter

First Frame

𝑊𝑞 𝑊𝑘 𝑊𝑣 𝑊𝑜𝑢𝑡

|𝐶+<br><br>| |
|---|---|
| |Timestep<br><br>Embedding|

Text Prompt EncoderText

Timestep

𝐶𝑐

Timestamps

|Temporal<br><br>Mask Embedding|
|---|

Temporal

|Timestamp Embedding|
|---|

Timestamp Encoder

Timestamp

Mask

Encoder

Temporal Control Module I

Temporal Control Module II

- Fig. 4. Pipeline of VFX Creator. We introduce two novel modules: (a) Spatial Controlled LoRA Adapter. This module integrates a mask-conditioned ControlNet with LoRA, injecting mask sequences into the model to enable instance-level spatial manipulation. (b) Temporal Controlled LoRA Adapter. We explore two strategies for incorporating temporal control: module I involves tokenizing start-end motion timestamps and embedding them into the diffusion process alongside the text space, while module II integrates temporal mask with timestep embeddings to guide the diffusion process.

Embeddings (RoPE) [Narvekar and Karam 2011] to enhance the model’s ability to capture temporal dynamics and long-range dependencies in video frames.

- 4.1.2 Low-rank Adaptation(LoRA). LoRA [Hu et al. 2021] targets the residual component of the model, denoted as Δ𝑊 , which is added to the original weight matrix, yielding the updated weights as: 𝑊 ′ = 𝑊 + Δ𝑊 . In this formulation, Δ𝑊 is expressed as the product of two low-rank matrices: Δ𝑊 = 𝐴𝐵𝑇 , where 𝐴 ∈ R𝑛×𝑑, 𝐵 ∈ R𝑚×𝑑, and 𝑑 < 𝑛, 𝑑 < 𝑚. By focusing on the smaller lowrank matrices 𝐴 and 𝐵, rather than the full-weight matrix𝑊 , LoRA effectively reduces both computational and memory costs during the training process.

- 4.2 VFX Creator

As shown in Fig. 4, VFX Creator is a controllable VFX generation framework based on a Video Diffusion Transformer. The model incorporates a spatial and temporal controllable LoRA adapter, requiring minimal training videos. Specifically, the mask control module enables instance-level spatial manipulation, while tokenized start-end motion timestamps are embedded in the diffusion process accompanied by the text encoder, allowing precise temporal control over effect timing and pace.

- 4.2.1 Temporal Controlled LoRA Adapter. To enable rhythmcontrolled visual effect generation, we first employ temporal video augmentation by utilizing the start and end timestamps of the effect’s motion to maximize data utilization. Specifically, the moving

video clip is subject to a random shift, with the start and end timestamps constrained within the ranges of [0,𝑇 −𝐷] and [𝐷,𝑇], where 𝑇 represents the maximum frame count in the training video, and 𝐷 denotes the duration of the visual effect’s motion.

In this section, we explore two strategies for integrating timestamp control signals and conduct experiments on three different effects to analyze their effectiveness and accuracy.

Strategy I: Integrate Temporal Mask with Timestep. Current temporal condition representation typically takes two forms: digitized start-end timestamps and temporal masks. The former normalizes timestamps directly, while the latter applies a temporal mask to the frame sequence, designating moving frames as 1 and static frames as 0. Existing methods for temporal condition injection can be categorized into two strategies: one integrates temporal mask with Timestep embeddings and injects them into the diffusion blocks; the other utilizes a timestamp encoder to interact with the text space and computes cross-attention with the noisy latent representation. In the following, we first focus on the former approach.

We project the temporal mask of motion in videos into a timestep embedding and add it to each frame to ensure uniform application of the motion’s timing and pace to every frame. As illustrated in Module I of Fig. 4, given the normalized timestamp, we introduce a timestamp encoder network that projects the temporal masks into the timestep embedding space, which is then incorporated into the DiT blocks [Peebles and Xie 2023].

Strategy II: Integrate Timestamps with Text Space. Inspired by [Fang et al. 2024], we first map the start and end timestamps of the visual effects to the prompt space, converting them into timestamp tokens. These tokens are then concatenated with the original text prompt tokens. By leveraging the cross-attention mechanism in the DiT block, we generate VFX videos conditioned on the temporal control signals. As shown in Module II of Fig. 4, given the normalized timestamps 𝑦timestamp, the text prompt 𝑦text, and textdomain specific encoder 𝜏𝜃, we introduce a timestamps encoder network 𝜏𝜙, that projects timestamps 𝑦timestamp to the intermediate text representation space R𝑀×𝑑𝜏 such that

𝜏𝜙 (𝑦𝑡𝑖𝑚𝑒𝑠𝑡𝑎𝑚𝑝) ∈ R𝑀×𝑑𝜏 (1) We concatenate 𝜏𝜙 (𝑦timestamp) and𝜏𝜃 (𝑦text) and input the crossattention layer Attention (𝑄,𝐾,𝑉) = softmax 𝑄𝐾

𝑇

·𝑉,with

√

𝑑

Q = 𝑊𝑄(𝑖) · 𝜑𝑖(𝑧𝑡),

(2)

K = 𝑊𝐾(𝑖) · (𝜏𝜙 (𝑦timestamp) ⊕ 𝜏𝜃 (𝑦text)), V = 𝑊𝑉(𝑖) · (𝜏𝜙 (𝑦timestamp) ⊕ 𝜏𝜃 (𝑦text)).

where 𝑧𝑡 denotes the latent representation 𝑧 at the 𝑡-th diffusion time step. 𝜑𝑖(𝑧𝑡) ∈ R𝑁×𝑑𝜖𝑖 denotes a flattened intermediate representation of the Transformer implementing 𝜖𝜃.𝑊𝑉(𝑖) ∈ R𝑑×𝑑𝜖𝑖 and 𝑊𝑄(𝑖) ∈ R𝑑×𝑑𝜏 and𝑊𝐾(𝑖) ∈ R𝑑×𝑑𝜏 are learnable projection matrices. ⊕ is the operator for tensor concatenation.

4.2.2 Spatial Controllable LoRA Adapter. Currently, we commonly employ three methods to incorporate spatial conditions into

the diffusion model: i) concatenating the reference image with the first frame mask before inputting them into the diffusion model [Ma et al. 2024a]; ii) combining the latent mask and noisy latent along the channel dimension for spatial control [Lei et al. 2024]; iii) introducing a spatial ControlNet [Zhang et al. 2023a] to extract a mask sequence that guides the generation process.

Our experimental observations reveal that the first two methods do not successfully facilitate spatial condition-based video generation tasks. While effective in U-Net-based diffusion models, they are not suitable injection techniques for transformer-based diffusion models. Therefore, to enable instance-level spatial manipulation, we introduce a plug-and- play mask control module, leveraging the mask guidance to precisely control the desiring instance.

During training, we utilize SAM2 to obtain the mask sequences of the moving instances. We retain the mask sequences preceding the start timestamp, while padding the remaining frames with zeros, and combine them to form the spatial condition. These spatial conditions are then injected into a learnable spatial ControlNet [Zhang et al. 2023a] for fine-tuning. As shown in Fig. 4, we extract mask conditions and integrate them into the main network. This branch shares trainable parameters initialized as a copy of the original half branch and operates in parallel, using a zero convolution as a bridge to integrate the conditional controls. Specifically:

𝒚𝑐 = F𝑚(𝒙) + Z(F𝑐𝑛(𝒙, 𝒄;Θ𝑐𝑛);Θ𝑧), (3) where F (·;Θ) denotes a neural model with learnable parameters Θ, Z(·;Θ𝑧) indicatesthezero convolutionlayer, and𝑥,𝑦𝑐 ∈ Rℎ×𝑤×𝑐

and 𝑐 are the 2D feature maps and conditional controls, respectively. This trainable spatial ControlNet branch is connected to the partly frozen main branch with a zero-initialized convolution layer, ensuring the integration of spatial conditions while minimizing interference with the base model.

- 5 EXPERIMENT

To assess the effectiveness of our Open-VFX dataset and VFX Creator model, we conduct experiments across all of the visual effects, demonstrating VFX Creator’s versatility in generating controllable VFX videos. To assess the performance of VFX video generation, we compare VFX Creator with the following state-of-the-art methods: (i) DynamiCrafter [Xing et al. 2025], (ii) CogVideoX [Yang et al. 2024], (iii) LTX-Video [HaCohen et al. 2024], and (iv) the pseudo ground-truth Pika or PixVerse. All of these approaches allow text prompts and reference images as input. Furthermore, we select several visual effects to conduct a focused evaluation of both temporal and spatial control accuracy.

- 5.1 Implementation Details

During the training phase, we incorporate low-rank matrices with a rank of 128 into the 3D Transformer module of the baseline network. We randomly sample 49 frames with a resolution of 480 × 720. We employ the AdamW [Loshchilov 2017] optimizer with a constant learning rate of 1e-4 for training all models. All experiments are conducted on the NVIDIA H800 GPU. We froze the gradients of most weights in the original base network and trained for 3000 steps with a learning rate of 1e-4, implementing both learning rate warm-up and decay mechanisms. Our VFX dataset is partitioned into training

and testing sets in a 9:1 ratio, both containing 15 distinct types of visual effects. The training set includes an average of 40 videos per effect, while the testing set contains 5 videos per effect.

- 5.2 Evaluation Metrics

In this experiment, we adopt three metrics following prior works: FID-VID [Unterthiner et al. 2018], FVD [Balaji et al. 2019] and Dynamic Degree [Huang et al. 2024] to evaluate the general quality and degree of dynamics of the synthesized videos. As shown in Table 2, we aim to focus on the generative performance of the motion in the video. We design three metrics to evaluate the accuracy of temporal control: frame-level errors Ef and second-level errors Es, as well as Temporal Intersection over Union (𝑇IoU). Specifically, temporal error quantifies the difference between the start and end timestamps of the predicted and ground truth segments. The frame-level temporal error and second-level temporal error are related through the frames per second (FPS):

Ef =

1 𝑁

∑︁𝑁

𝑖=1

𝑡 ˆstart,𝑖 − 𝑡start,𝑖 + 𝑡 ˆend,𝑖 − 𝑡end,𝑖 (4)

Es =

1 𝑁

∑︁𝑁

𝑖=1

𝑡 ˆstart,𝑖 FPS −

𝑡start,𝑖 FPS +

𝑡 ˆend,𝑖 FPS −

𝑡end,𝑖 FPS

(5)

where 𝑁 is the number of video segments, and 𝑡start,𝑖 and 𝑡end,𝑖 are the normalized start and end timestamps of the 𝑖-th video clip.

To evaluate the accuracy of the predicted timestamps, we randomly sample five pairs of start and end ground truth timestamps for a reference sample. The start ground truth is constrained within the range of [0, 2/𝑇], while the end ground truth falls within the range of [2/𝑇,𝑇], where𝑇 represents the total number of frames.

- 5.3 Quantitative Comparison Results

For quantitative evaluation, we conducted experiments on 15 types of visual effects from our dataset and compared VFX Creator with three state-of-the-art open-source methods, with detailed results shown in Table 1. VFX Creator outperforms the other methods across all metrics, particularly in generating visual effects with large motion patterns, indicating superior video quality and more accurate motion generation. As seen in Table 1, DynamiCrafter [Xing et al. 2025] and CogVideoX [Yang et al. 2024] are less responsive to visual effect prompts, producing videos with minimal or no motion, as reflected by their lower dynamic degrees. While LTX-Video [HaCohen et al. 2024] exhibits a higher dynamic degree, this is due to the generation of large, incorrect motions, which does not correspond to the semantic accuracy of the motion. These results align with the limitations observed in the quantitative evaluation. To further validate these findings, we conducted extensive user studies to assess the correspondence accuracy between generated visual effects and text prompts. VFX Creator consistently outperforms other state-of-the-art methods, confirming its exceptional ability to generate high-quality, semantically accurate visual effects, even when dealing with large and complex abstract motions.

###### (a) Deflate it (b) Dissolve it

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

DynamiCrafter

DynamiCrafter

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

LTX-Video

LTX-Video

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

CogVideoX

CogVideoX

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

VFX Creator (Ours)

VFX Creator (Ours)

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

Pika

Pika

###### (c) Eye-pop it (d) Explode it

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

DynamiCrafter

DynamiCrafter

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

[Figure 245]

[Figure 246]

LTX-Video

LTX-Video

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

CogVideoX

CogVideoX

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

VFX Creator (Ours)

VFX Creator (Ours)

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

Pika

Pika

- Fig. 5. Qualitative comparisons of VFX video generation on two different visual effects between our method, DynamiCrafter, LTX-Video, CogVideoX-5B, and Pika.

Table 1. Quantitative comparisons of VFX video generation for 15 visual effects in our dataset.

Method FID-VID ↓ FVD ↓ Dynamic Degree ↑

DynamiCrafter 119.78 1515.28 0.27 LTX-Video 82.93 1563.73 0.51 CogVideoX 90.82 1624.91 0.14

VFX Creator (Ours) 29.92 752.95 0.63

- 5.4 Qualitative Comparison Results

We present qualitative comparison results of VFX Creator with four representative models on three effects: "Deflate it," "Dissolve it," and "Eye-pop it," as shown in Fig. 5. For comparison, results from Pika are used as the benchmark. We found that VFX Creator consistently generates more reliable videos, even when Pika struggles to produce satisfactory outputs. For example, Pika fails to generate

correct effects in "Defalte it" and "Eye-pop it", while VFX Creator successfully generates the abstract effect video. Pika’s failure may stem from the substantial increase in complexity associated with accurately locating and generating the eye in smaller, more distant contexts. In contrast, DynamicCrafter exhibits significant challenges in generating visual effect motions that correspond to text prompts. LTX-Video and CogVideoX-5B, on the other hand, generate either small-scale motions or large, erroneous ones, which leads to a lack of temporal consistency and semantic accuracy. These limitations are not present in VFX Creator, which maintains better alignment with the intended motion and effect characteristics. This ensures improved temporal consistency and visual fidelity. These results demonstrate that VFX Creator outperforms other models, delivering visually accurate and temporally consistent visual effects, even in cases where ground truth struggles with reliability.

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

Reference Image

Mask

LevitateTa-da(b)(a)itit

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

Reference Image

Mask

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Mask

Reference Image

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Mask

Reference Image

- Fig. 6. Qualitative results of spatial controllable VFX video generation of our method on two different visual effects. Users can precisely specify the animated instance by clicking points or dropping boxes to obtain the mask.

- 5.5 Ablation Study

The proposed VFX Creator is pivotal for generating high-quality controllable VFX videos. To assess the contributions of the spatial and temporal control modules, we conduct a series of ablation studies. First, we compare the results of two temporal control injection strategies. The impact of start and end timestamp guidance on animation results is presented in Table 3. Our observation reveals that integrating timestamps with the text space yields superior results. This approach integrates timestamps with textual prompts to perform a cross-attention mechanism with video latent representations, enhancing the precise alignment of visual effects with specified temporal cues. Additionally, the transformer-based diffusion model is more effective in handling implicit representation injections compared to explicit conditions. Additionally, we present the quantitative results of integrating the spatial control module. As demonstrated in Fig. 6, VFX Creator effectively achieves accurate object manipulation, generating photorealistic videos with strong consistency between the visual effects and user interactions. Lastly, we analyze the impact of different sample sizes during training by comparing the model’s performance across varying shot numbers: 1-shot, 10-shot, and 40-shot. As shown in Table 3, we observe that the number of shots plays a significant role in the model’s performance. The results indicate that increasing the number of shots generally improves performance, with the 10-shot configuration often yielding balanced results. This highlights the model’s data efficiency, demonstrating its ability to learn abstract and complex visual effect motions without the need for large amounts of data.

- Table 2. Ablation results of two temporal control integration strategies.

Visual Effect

Temporal Strategy I Temporal Strategy II 𝑇IoU ↑ Ef ↓ Es ↓ 𝑇IoU ↑ Ef ↓ Es ↓

Ta-da it 0.69 12.52 1.56 0.85 5.04 0.63 Explode it 0.68 11.30 1.49 0.88 3.76 0.47 Levitate it 0.69 12.88 1.61 0.80 5.36 0.67

Average 0.69 12.23 1.56 0.84 4.72 0.59

- Table 3. Ablation results of different sample sizes during training across varying shot numbers.

Effect Shots FID-VID ↓ FVD ↓ Dynamic Degree ↑

1-shot 52.31 1432.40 0.6 10-shot 47.91 2861.18 1.0 40-shot 54.73 726.83 1.0

Ta-da it

1-shot 96.48 2667.72 1.0 10-shot 57.71 2829.00 1.0 40-shot 50.97 2394.20 1.0

Explode it

1-shot 140.42 3297.11 1.0 10-shot 44.62 1409.98 1.0 40-shot 44.35 1644.69 1.0

Squish it

5.6 User Study

To further validate the effectiveness of our method, we conducted a human evaluation, comparing our approach with four existing approaches, without using additional data for guidance. We invited 20 users to assess 30 sets of generated comparing results. We evaluated the quality of the generated videos across four dimensions: Text

Alignment, Subject Fidelity, Motion Fluency, and Overall Quality. Text Alignment measures how well the generated video aligns with the text prompt; Subject Fidelity evaluates how closely the generated object matches the reference image; Motion Fluency assesses the smoothness and quality of the motions in the generated video; and Overall Quality reflects whether the overall quality of the generated video meets user expectations. As shown in Fig. 7, both our method, VFX Creator, and Pika& PixVerse achieved superior user preference across all metrics, with VFX Creator slightly outperforming Pika& PixVerse, demonstrating the effectiveness of our approach.

[Figure 307]

- Fig. 7. User Study. Our VF Creator demonstrates superior human preference compared to other methods

### 6 LIMITATION

Despite the introduction of our pioneering visual effects video dataset, Open-VFX, alongside the development of VFX Creator for spatial-temporal controllable effect generation, several limitations remain. Firstly, while we have curated 15 types of effects and designed the system for data efficiency to facilitate easy VFX personalization, there is still considerable scope for expanding the dataset in both breadth and depth to address the diverse visual effects requirements across various scenes. Furthermore, we explored training a unified model for visual effect generation. However, experimental results indicate that the quality of unified training falls short compared to category-specific training, as direct unification tends to confuse multiple visual effects. In the future, implementing strategies such as Mixture of Experts (MOE) [Chen et al. 2022] may improve the performance of mixed training approaches.

### 7 CONCLUSION

In conclusion, this work presents significant advancements in the field of controllable visual effects (VFX) generation, addressing critical challenges associated with fine-grained spatial and temporal manipulation. First, we propose the Open-VFX dataset establishes a foundational resource for future explorations in VFX generation, offering a diverse array of effect categories and detailed annotations that enhance the training of VFX generation models. Furthermore, we introduce VFX Creator, a simple yet effective controllable VFX generation framework based on a Video Diffusion Transformer.

Specifically, we leverage minimal training videos and enabling finegrained spatial and temporal control, our system bridges the gap between traditional VFX techniques and modern generative models. The extensive experiments demonstrate its ability to produce realistic, dynamic effects with state-of-the-art performance in both spatial and temporal controllability. With the innovative integration of instance-level spatial manipulation and precise temporal control, VFX Creator paves the way for efficient and user-friendly VFX generation, making advanced visual effects more accessible to a broader audience and expanding creative possibilities in filmmaking.

REFERENCES

Adobe. n.d.. From pyrotechnics to prosthetics: A guide to special effects in movies. https://www.adobe.com/creativecloud/video/discover/a-guide-to-specialeffects-in-movies.html Accessed: 2023-01-17.

Yogesh Balaji, Martin Renqiang Min, Bing Bai, Rama Chellappa, and Hans Peter Graf.

2019. Conditional GAN with Discriminative Filter Generation for Text-to-Video Synthesis.. In IJCAI, Vol. 1. 2.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. (2024). https: //openai.com/research/video-generation-models-as-world-simulators

Anastasia Chabanova. 2022. VFX–A New Frontier: The Impact of Innovative Technology on Visual Effects. Ph.D. Dissertation. University of Westminster.

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. 2023. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023).

Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, and Yuanzhi Li. 2022. Towards Understanding Mixture of Experts in Deep Learning. arXiv:2208.02813 [cs.LG] https://arxiv.org/abs/2208.02813

I-Sheng Fang, Yue-Hua Han, and Jun-Cheng Chen. 2024. Camera Settings as Tokens: Modeling Photography on Latent Diffusion Models. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2023. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023).

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. 2024. LTX-Video: Realtime Video Latent Diffusion. arXiv preprint arXiv:2501.00103 (2024).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic

models. Advances in neural information processing systems 33 (2020), 6840–6851. Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21807–21818. Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. 2024. Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9212–9221.

Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. 2024. CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos. In Proc. arXiv:2410.11831.

Diederik P Kingma. 2013. Auto-encoding variational bayes. arXiv preprint

arXiv:1312.6114 (2013). Kuaishou. 2024. Keling. https://kling.kuaishou.com/z Accessed: 2025-01-19. Guojun Lei, Chi Wang, Hong Li, Rong Zhang, Yikai Wang, and Weiwei Xu. 2024.

Animateanything: Consistent and controllable animation for video generation. arXiv preprint arXiv:2411.10836 (2024).

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, Jianchen Zhu, Kai

Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, and Qinglin Lu. 2024. Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding. arXiv:2405.08748 [cs.CV] https://arxiv.org/abs/2405.08748

Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. 2024. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131 (2024).

I Loshchilov. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. 2024b. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048 (2024).

Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Chenyang Qi, Chengfei Cai, Xiu Li, Zhifeng Li, Heung-Yeung Shum, Wei Liu, et al. 2024a. Follow-your-click: Opendomain regional image animation via short prompts. arXiv preprint arXiv:2403.08268 (2024).

Joanna Materzynska, Josef Sivic, Eli Shechtman, Antonio Torralba, Richard Zhang, and Bryan Russell. 2023. Customizing motion in text-to-video diffusion models. arXiv preprint arXiv:2312.04966 (2023).

Niranjan D Narvekar and Lina J Karam. 2011. A no-reference image blur metric based on the cumulative probability of blur detection (CPBD). IEEE Transactions on Image Processing 20, 9 (2011), 2678–2683.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021).

Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. 2024. I2VEdit: FirstFrame-Guided Video Editing via Image-to-Video Diffusion Models. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In

Proceedings of the IEEE/CVF International Conference on Computer Vision. 4195–4205. Pexels. 2024. Free Stock Photos. https://www.pexels.com/ Accessed: 2024-01-19. Pika. 2023. Pika: A platform for creative AI art. https://pika.art/ Accessed: 2025-01-11. Pixverse. 2023. Pixverse: AI-powered Image and Video Editing Platform. https://app.

pixverse.ai/ Accessed: 2025-01-11.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. 2024. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720 (2024).

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2024. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714 [cs.CV] https://arxiv.org/abs/2408.00714 Yixuan Ren, Yang Zhou, Jimei Yang, Jing Shi, Difan Liu, Feng Liu, Mingi Kwon, and Abhinav Shrivastava. 2024. Customize-a-video: One-shot motion customization of text-to-video diffusion models. arXiv preprint arXiv:2402.14780 (2024).

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models. arXiv:2112.10752 [cs.CV] https://arxiv.org/abs/2112.10752

Ruizhi Shao, Youxin Pang, Zerong Zheng, Jingxiang Sun, and Yebin Liu. 2024. Human4DiT: Free-view Human Video Generation with 4D Diffusion Transformer. arXiv preprint arXiv:2405.17405 (2024).

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. PMLR, 2256–2265.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020).

Genmo Team. 2024. Mochi 1. https://github.com/genmoai/models. Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin

Michalski, and Sylvain Gelly. 2018. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717 (2018).

Tan Wang, Linjie Li, Kevin Lin, Yuanhao Zhai, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. 2024a. Disco: Disentangled control for realistic human dance generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9326–9336.

Zhao Wang, Aoxue Li, Lingting Zhu, Yong Guo, Qi Dou, and Zhenguo Li. 2024b. Customvideo: Customizing text-to-video generation with multiple subjects. arXiv preprint arXiv:2401.09962 (2024).

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024c. Motionctrl: A unified and flexible motion controller

for video generation. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. 2024. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6537–6549.

Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. 2024a. Motionbooth: Motion-aware customized text-to-video generation. arXiv preprint arXiv:2406.17758 (2024).

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023b. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7623–7633.

Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, and Xiangyu Zhang. 2023a. Lamp: Learn a motion pattern for few-shot-based video generation. arXiv preprint arXiv:2310.10769 (2023).

Tao Wu, Yong Zhang, Xintao Wang, Xianpan Zhou, Guangcong Zheng, Zhongang Qi, Ying Shan, and Xi Li. 2024b. Customcrafter: Customized video generation with preserving motion and concept composition abilities. arXiv preprint arXiv:2408.13239 (2024).

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. 2025. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision. Springer, 399–417.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. Cogvideox: Textto-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023a. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Yuxin Zhang, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. 2023b. Motioncrafter: One-shot motion customization of diffusion models. arXiv preprint arXiv:2312.05288 (2023).

Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jia-Wei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. 2025. Motiondirector: Motion customization of text-to-video diffusion models. In European Conference on Computer Vision. Springer, 273–290.

### A DEFINATIONS OF VISUAL EFFECTS

The Open VFX dataset encompasses 15 distinct categories of visual effects, featuring a wide array of subjects, including characters, animals, products, and scenes. As illustrated in Table 4, we offer a comprehensive explanation of the specific meanings associated with each visual effect, facilitating a deeper understanding for users.

Table 4. Visual Effect Types and Corresponding Definitions in the Open VFX Dataset.

Types of VFX Definition Cake-ify it

Transform the subject into hyper-realistic prop cakes.

Crumble it Break apart the subjects into fragments. Crush it Apply a hydraulic press to flatten the subject. Decapitate it Simulate the decapitation of subjects.

Similar to a balloon losing air, cause subjects to shrink and flatten.

Deflate it

Cause the object to disintegrate into nothingness.

Dissolve it

Explode it Burst the subject into fragments. Eye-pop it Make the eyes of subjects bulge or pop out. Inflate it Puff up the still subject like a balloon.

Make static objects or subjects appear to float or hover.

Levitate it

Melt it Turn objects into fluid, gooey forms. Squish it

Compress the subject as though under immense pressure.

With a flourish, subjects disappear behind a cloth.

Ta-da it

Transform into a black Venom

Characterize the static subject, transforming it into a black Venom.

Transform into Harley Quinn

Characterize the static subject, transform it into Harley Quinn.

### B QUANTITATIVE RESULTS OF MIXED TRAINING

As a data-efficient system capable of achieving visual effect personalization, we attempted to explore a unified VFX generation model. Specifically, we tuned 15 effects using 600 effect videos in a combined manner. We then evaluated the results of the unified model, calculating their FID-VID [Unterthiner et al. 2018], [Balaji et al. 2019], and Dynamic Degree [Huang et al. 2024]. Additionally, we compared the results of the 15 effects from both category-specific training and unified and mixed training. As shown in Table 5, experimental results indicate that the quality of unified training falls short compared to category-specific training, as direct unification tends to confuse multiple visual effects. Most of the effect results indicate that mixed training leads to a decline in overall video quality, although about one-third of the effects exhibit slight improvements

after mixed training, such as "Ta-da it." In the future, implementing strategies such as Mixture of Experts (MOE) [Chen et al. 2022] may improve the performance of mixed training approaches.

C MORE CONTROLLED QUALITATIVE RESULTS

In this section, we present additional results on spatial and temporal controlled VFX video generation using VFX Creator. Fig. 8 illustrates qualitative findings for two distinct effects across four cases, showcasing VFX Creator’s ability to perform instance-level object manipulation with precision. Additionally, the temporal results depicted in Fig. 10 demonstrate VFX Creator’s capacity to control the rhythm of VFX video generation over time. This underscores the accuracy and effectiveness of the control mechanisms in our method.

D MORE QUALITATIVE COMPARISON RESULTS

In this section, we present additional qualitative comparison results of VFX Creator with other video generation models, including LTXVideo [HaCohen et al. 2024], and CogVideoX [Yang et al. 2024]. As illustrated in Fig. 9, LTX-Video and CogVideoX demonstrate a failure to accurately interpret the visual effect prompt, leading to the generation of only minimal actions. In contrast, VFX Creator and Pika (ground truth) exhibit a significantly better comprehension and generation of the corresponding effect videos. Notably, for the "Crumble it" effect, our methodology yields results that are not only more complete but also superior in quality when compared to the ground truth.

Table 5. Comparison results of mix and single training for visual effect generation.

Decapitate

Crumble

Cake-ify

Dissolve

Eye-pop

Explode

Levitate

Method

Deflate

Venom

Squish

Metric

Inflate

harley

Crush

Ta-da

Melt

Single 54.48 65.11 46.71 43.76 103.90 76.14 50.97 34.87 94.62 86.14 35.12 63.37 44.35 54.73 117.90

FID-VID↓

Mix 67.22 65.06 44.52 44.52 111.28 87.00 84.19 54.43 117.34 77.35 68.32 70.38 52.36 34.65 108.99 FVD↓

#### Single 1140 1690 1000 1263 2034 1463 2394 1547 3566 1946 665 1794 1644 726 3668

Mix 1503 1738 1015 1054 2133 1794 2612 1641 3811 2184 1018 2774 1543 979 3911 Dynamic Degree↑

Single 0.8 0.8 0.0 0.6 0.0 0.8 1.0 0.0 1.0 0.8 0.0 0.6 1.0 1.0 1.0 Mix 0.8 0.8 0.0 0.6 0.0 0.8 1.0 0.0 1.0 0.8 0.0 0.6 1.0 1.0 1.0

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Reference Image Mask

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Mask

(a)Ta-dait

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

Reference Image

Mask

Mask

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

Reference Image

Mask

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

Mask

Reference Image

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

Reference Image Mask

(b)Levitateit

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

Mask

Reference Image

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

Mask

Reference Image

Fig. 8. More spatial controlled VFX generation results of our method on two different visual effects.

LTX-Video

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

CogVideoX

[Figure 372]

Reference Image

###### (a)Crumbleit

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

VFX Creator (Ours)

Pika

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

LTX-Video

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

CogVideoX

###### (b)Squishit

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

VFX Creator (Ours)

[Figure 402]

Pika

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

Fig. 9. More qualitative comparison results of VFX video generation on two different visual effects between our method, DynamiCrafter, LTX-Video, CogVideoX5B, and Pika.

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

Start Frame: 8 End Frame: 36

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

Start Frame: 16 End Frame: 40

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

Start Frame: 32 End Frame: 48

Ta-dait

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

Start Frame: 8 End Frame: 36

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

Start Frame: 16 End Frame: 40

[Figure 449]

Start Frame: 16 End Frame: 48

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

Fig. 10. More temporal controlled VFX generation results of our method.

