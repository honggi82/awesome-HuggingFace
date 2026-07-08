# arXiv:2502.03621v2[cs.CV]20Oct2025

## DynVFX: Augmenting Real Videos with Dynamic Content

DANAH YATIM∗, Weizmann Institute of Science, Israel RAFAIL FRIDMAN∗, Weizmann Institute of Science, Israel OMER BAR-TAL, Runway ML, United States of America TALI DEKEL, Weizmann Institute of Science, Israel

Input video

Input video

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Add a majestic whale in the background

Add a puppy peaking head out of the box

Output video

Output video

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Fig. 1. DynVFX augments real-world videos with new dynamic content described by a simple user-provided text instruction (shown in the top row under each input video). The framework automatically infers where the synthesized content should appear, how it should move, and how it should harmonize at the pixel level with the scene, without requiring any additional user input. The key idea is to selectively extend the attention mechanism in a pre-trained text-to-video diffusion model, enforcing the generation to be content-aware of existing scene elements (anchors) from the original video. This allows the model to generate content that naturally interacts with the environment, producing complex and realistic video edits in a fully automated way (bottom row).

We present a method for augmenting real-world videos with newly generated dynamic content. Given an input video and a simple user-provided text instruction describing the desired content, our method synthesizes dynamic objects or complex scene effects that naturally interact with the existing scene over time. The position, appearance, and motion of the new content are seamlessly integrated into the original footage while accounting for camera motion, occlusions, and interactions with other dynamic objects in the scene, resulting in a cohesive and realistic output video. We achieve this via a zero-shot, training-free framework that harnesses a pre-trained text-to-video diffusion transformer to synthesize the new content and a pretrained vision-language model to envision the augmented scene in detail. Specifically, we introduce a novel inference-based method that manipulates features within the attention mechanism, enabling accurate localization and seamless integration of the new content while preserving the integrity of the original scene. Our method is fully automated, requiring only a simple user instruction. We demonstrate its effectiveness on a wide range of edits applied

∗Both authors contributed equally to this research.

Authors’ Contact Information: Danah Yatim, Weizmann Institute of Science, Israel; Rafail Fridman, Weizmann Institute of Science, Israel; Omer Bar-Tal, Runway ML, United States of America; Tali Dekel, Weizmann Institute of Science, Israel.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SA Conference Papers ’25, Hong Kong, Hong Kong © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2137-3/2025/12 https://doi.org/10.1145/3757377.3764008

to real-world videos, encompassing diverse objects and scenarios involving both camera and object motion1. Project page: https://dynvfx.github.io/

CCS Concepts: • Computing methodologies → Computer vision. Additional Key Words and Phrases: Text-to-Video Editing, Diffusion Models ACM Reference Format:

Danah Yatim, Rafail Fridman, Omer Bar-Tal, and Tali Dekel. 2025. DynVFX: Augmenting Real Videos with Dynamic Content. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 19 pages. https://doi.org/10. 1145/3757377.3764008

1 Introduction

Incorporating computer-generated imagery (CGI) into real-world footage has been a transformative capability in film production, enabling the creation of visual effects that would be difficult or impossible to achieve otherwise. For instance, the seamless integration of CGI characters, such as Gollum in The Lord of the Rings or T-Rex in Jurassic Park, has empowered filmmakers to blend fantastical elements with real-world environments, resulting in immersive storytelling. Professional video editing software tools provide artists with precise control over visual effects integration into video footage [Adobe 2024; Autodesk 2024; Blender 2024; Unreal Engine 2024]. However, they require significant expertise, manual effort, pre-made assets, and financial resources. To disentangle these inspiring storytelling capabilities from professional and economic status, we

1Code will be made publicly available.

propose a new creative task: free-vocabulary, text-driven integration of newly generated dynamic content into real-world videos. Specifically, given an input video along with a short user instruction describing the new content (e.g., “add a massive whale”), our goal is to synthesize new dynamic objects or complex scene effects that naturally interact with the existing scene across the entire video (Fig. 1). Importantly, our focus is to achieve this without additional user input, allowing anyone to insert dynamic content into videos just by describing it.

We approach this fully automated task by inferring where the synthesized content should appear, how it should move, and how it should harmonize at the pixel level with the scene without requiring any additional user input. This is achieved by harnessing dynamic generative priors learned by a pre-trained text-to-video diffusion transformer (DiT) model and comprehensive scene understanding learned by a vision-language model without any fine-tuning or additional training. This contrasts with previous methods that rely on additional user input, such as VFX assets [Hsu et al. 2025] or perframe references to specify the placement of the new content over time. These methods struggle with integrating complex dynamic content (e.g., synthesizing multiple articulated objects or global effects, as shown in the dinosaurs and tsunami examples in Fig. 5).

Our task poses several new fundamental challenges. First, the generation process of the fixed pre-trained DiT model must be content-aware such that the position, appearance, and motion of the synthesized dynamic content integrates naturally with the original scene. This entails synthesizing objects that respect occlusions, maintain appropriate relative size and perspective relative to the camera position, and realistically interact with other dynamic objects. All of this must be achieved while maintaining the integrity of the original video, ensuring that new content enhances the scene without compromising its authenticity.

Existing text-based methods focus on controlling diffusion generation to preserve original scene aspects. However, this imposes a trade-off between preserving aspects of the original scene and adding new content to the scene: either over-fitting or under-fitting the original video in terms of appearance, motion, positioning, or scale. Attention manipulation within a UNet-based architecture has been widely used for content preservation in video editing tasks. Extending attention across frames in these models allows for global appearance coherence. Yet, spatiotemporal information is preserved only implicitly. Hence, in the context of UNet-based models, this has been used mainly for appearance transfer. We revisit extended attention in the context of DiT-based text-to-video models, revealing new insights into their internal representation. Unlike UNetbased models, we observe that keys/values in DiTs locally encode corresponding video patches across space and time. In particular, applying the same positional embedding to both source and target keys/values during the extended attention operation enables alignment between the generated and original content at corresponding spatiotemporal locations.

We take advantage of this property to enforce content-aware generation for new content placement. Specifically, we steer the localization of the edit through Anchor Extended Attention: we incorporate a specific set of keys/values extracted from the original video as additional context to the DiT-based model during sampling,

enabling the model to focus on existing scene elements essential for correctly placing and integrating new content naturally.

Despite accurate placement from Anchor Extended Attention, it does not guarantee pixel-level alignment with the original scene. To achieve better harmonization, we iteratively refine the edit by repeatedly sampling with Anchor Extended Attention while updating only the edited regions in each iteration.

To allow a fully automated framework, a vision-language model is utilized to interpret the interaction and reason about integration. We guide the model to: (1) translate the user’s instructions into detailed scene descriptions and (2) identify both the prominent visual elements of the existing scene and the new content to be introduced. These identified objects are used in our framework for the application of a text-based segmentation model to aid the localization and harmonization of the new content.

We demonstrate the effectiveness of our approach on a variety of edits applied to real-world videos. Our method supports a wide range of scene augmentations across various scenarios while maintaining realistic interaction, occlusion, lighting, and camera motion. It is worth noting that our method utilizes a publicly available textto-video model, which exhibits a significant gap in video generation quality compared to recent state-of-the-art video models. Nevertheless, we observe that within our problem formulation and objectives, we can distill surprisingly powerful generative capabilities from this model.

To summarize, our work makes the following contributions:

- • We introduce a new task of integrating newly generated dynamic content into real-world videos without relying on the user to provide complex references for the effect.
- • We propose a tuning-free, zero-shot, fully automated method that enables harmonized content integration while maintaining high fidelity to the original scene.
- • We demonstrate state-of-the-art results, achieving the best tradeoff between synthesizing new dynamic elements and maintaining high fidelity to the original content.

2 Related Work

Leveraging Text-to-Image Models. With the advancement of textto-image (T2I) models, techniques for image manipulation based on such models have evolved rapidly. Among these advancements, notable progress has been made in the task of instruction-based image editing. Several works [Brooks et al. 2023; Sheynin et al. 2024; Zhang et al. 2023a, 2024d] have proposed directly fine-tuning generative models on pairs of original and edited images coupled with user-provided instructions.

Object insertion in images falls under the broader umbrella of instruction-based editing techniques. Recent object insertion methods [Canberk et al. 2024; Wasserman et al. 2025; Winter et al. 2024] leverage inpainting models to create paired image datasets, which are then used to fine-tune image editing models. However, extending these approaches to videos presents significant challenges. In particular, generating large-scale instruction-paired video datasets can be expensive in both time and computational resources, as it requires manual effort to annotate frames. This cost and complexity

make it challenging to adapt existing image-based methodologies directly to the video domain.

Several works explore the manipulation of internal feature representation in T2I models for image and video editing tasks. In U-Net–based architectures, attention keys and values were shown to encode global appearance information [Cao et al. 2023], which several video editing methods [Khachatryan et al. 2023; Qi et al. 2023; Wu et al. 2023] exploit by extending self-attention across frames to enforce global appearance consistency. For localized appearance coherency, MasaCtrl [Cao et al. 2023] and Consistory [Tewel et al. 2024b] mask and selectively replace or extend attention keys and values. Add-It [Tewel et al. 2024a] employs global weighted extended attention for object insertion in images. FlowEdit [Kulikov et al.

- 2024] uses pre-trained flow models for inversion-free local edits. In contrast to the methods mentioned above that may fail to preserve the original content, we propose applying extended attention only to specific regions of the source scene, allowing generation to focus on essential elements.

Controllable Video Generation and Editing. Recently, numerous methods have been developed to incorporate various forms of control signals into video generation pipelines. Several video-tovideo methods propose to condition the generation on per-frame spatial maps such as depth maps and edge maps [Wang et al. 2023; Zhang et al. 2024b]. A common approach in video editing is to enforce the preservation of the original scene layout and motion for various tasks. Several consistent video editing works [Bar-Tal et al. 2022; Cong et al. 2023; Geyer et al. 2023; Lee et al. 2023] restrict the preservation of scene structure and motion, and rely on a 2D generative prior. Therefore, they cannot support large structural deviations or motion adaptation. [Jeong et al. 2024; Materzyńska et al. 2024; Park et al. 2025; Yatim et al. 2024; Zhao et al. 2024] proposed to utilize a text-to-video (T2V) model for the task of motion transfer by controlling diffusion video generation to preserve original scene motion.

[Ku et al. 2024; Ouyang et al. 2024] adopt a two-stage pipeline: first, apply an off-the-shelf image editing method to the video’s first frame, then use an image-to-video generation model to propagate that edit through time via temporal feature injection. All of these methods fail to insert new dynamic content. Our approach focuses on integrating additional dynamic elements into the video.

Reference-Based Video Content Insertion. Recent methods [BarTal et al. 2024; Bian et al. 2025; Ma et al. 2024; Mou et al. 2024; Tu et al. 2025; Zhang et al. 2024c; Zhou et al. 2023; Zi et al. 2025] have explored the adaptation of video models for video inpainting by conditioning on a masked video and a corresponding binary mask. This setup encourages the model to preserve unmasked information while generating new content in the masked region. Some methods, such as ReVideo [Mou et al. 2024] and VideoAnydoor [Tu et al.

- 2025], additionally guide the motion of the new content based on user-provided motion trajectories and bounding boxes.

VideoDoodles [Yu et al. 2023] combines hand-drawn animations with video footage in a scene-aware manner by tracking a userprovided planar canvas in 3D. However, it requires key-frame edit placement. Our task fundamentally differs from these methods,

which all require explicit per-frame control over where (via perframe masks [Bian et al. 2025] or bounding boxes [Mou et al. 2024]) or how the new content will move (via motion trajectories [Mou et al. 2024]). In contrast, our method does not constrain either aspect and requires only a simple text instruction as input. The locations and dynamics in our framework emerge from the generative prior of the T2V model. Furthermore, the reliance of these methods on user-provided per-frame annotations is impractical for integrating complex dynamics (e.g., Fig. 5 tsunami and dinosaurs). While a static object can be masked with a simple bounding box, manually defining per-frame masks for complex motion or interactions is extremely difficult. Other approaches [KlingAI 2024; Pika 2024] reference the asset itself by requiring the user to additionally input an image of the object. This approach limits their applicability whenever a suitable example cannot be found for the given video. For instance, these methods struggle to generate global effects, making them less preferable than text-based insertion methods.

Language Models for Video Content Creation. Advancements in Vision-Language Models (VLMs) have enabled methods to utilize such models in various video-related tasks. Some methods [Chen et al. 2024; Yang et al. 2024] use VLMs to produce detailed video captions from a series of frames, which are then utilized to train T2V generative models. Other methods utilize such models for achieving better generation controllability. For instance, VideoDirectorGPT [Lin et al. 2023] utilizes a VLM for multi-scene video generation by training diffusion adapters to incorporate additional conditioning inputs, while LVD [Lian et al. 2023] incorporates layout guidance from the VLM during the sampling process. AutoVFX [Hsu et al. 2025] uses an LLM to generate a video editing program pipeline based on the user’s instructions. In our work, we employ a VLM as a “VFX assistant”. Based on a short user instruction, it yields a comprehensive description of the edited video along with the prominent objects present in the scene.

3 Preliminaries

Diffusion probabilistic models [Ho et al. 2020; Sohl-Dickstein et al. 2015] are a class of generative models that aim to learn a mapping from noise 𝒙𝑇 ∼ N(0,𝐼) to a data distribution 𝑞. Starting from a Gaussian i.i.d. noise sample 𝒙𝑇 ∼ N(0,𝐼), the diffusion model Φ is applied iteratively through a sequence of denoising steps, ultimately producing a clean output sample 𝒙0.

Recently, a new class of latent T2V models, built on Diffusion Transformers (DiTs) [Peebles and Xie 2023], has gained significant popularity, as DiTs enhance spatial coherence and enable training for arbitrary aspect ratios and video lengths. Unlike previous U-Net–based architectures, which utilize interleaving 2D spatial attention and 1D temporal attention layers, DiT treats the entire video as a long token sequence, where both text and spatiotemporal tokens are processed with full self-attention layers, a mechanism often referred to as 3D attention. In each DiT block, text tokens and spatiotemporal tokens are projected into queries, keys, and values using separate sets of weights for each modality, and the sequences of the two modalities are concatenated as a joint input for the attention operation. The attention [Vaswani et al. 2017] operation

[Figure 13]

: [“A knight in shinig armor...”]

| | |
|---|---|
| | |

Vision Language Model

“Add a knight” EVF-SAM

[Figure 14]

: [“horse”] : [“knight”]

[Figure 15]

###### Input instruction

[Figure 16]

T2V Diffusion Model

[Figure 17]

Masked Keys and Values Extraction

[Figure 18]

| |[Figure 19]<br><br>[Figure 20]| | |[Figure 21]|
|---|---|---|---|---|

[Figure 22]

DiTBlock

DiTBlock

DiTBlock

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

...

...

DDIM inversion

| | | |
|---|---|---|
| |AnchorE|xtAttn|
| | | |

[Figure 29]

[Figure 30]

Input video

masks

[Figure 31]

[Figure 32]

##### ×T

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

###### VAE Decoding

DiTBlock

DiTBlock

DiTBlock

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

+noise

[Figure 44]

[Figure 45]

New content segmentation

+

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Edited video

[Figure 51]

Update

[Figure 52]

- Fig. 2. Pipeline. Top (pre-processing). Given an input video Vorig and instruction PVFX, we (i) Apply our VLM protocol for instruction interpretation, to yield a comprehensive scene description Pcomp, original objects Oorig and target object Oedit descriptions. (ii) DDIM invert Vorig to extract spatiotemporal keys/values [Korig, Vorig]. Bottom (editing). We initialize the composed latent with 𝒙comp = 𝒙orig and iterate over a list of descending noise levels 𝑡 =𝑇˜ →𝑇min used for noising 𝒙comp. At each iteration 𝑡 we: (i) noise 𝒙comp to noise level 𝑡, and sample with Anchored Extended Attention, to output 𝒙ˆcomp. (ii) Update 𝒙comp within the new contents masked regions 𝑴VFX by adding the residual 𝒙res = 𝑴VFX · (𝒙ˆcomp − 𝒙orig) to 𝒙orig. Repeating this loop gradually integrates the new content, yielding the edited video Vcomp.

computes the affinities between the d-dimensional projections Q and K to yield the output of the layer:

assistant” via a system prompt containing guidelines in the context of our tasks.

- (2) LocalizationviaAnchorExtendedAttention.Our approachachieves precise placement of new content by guiding the T2V generation model during sampling to be content-aware. We steer the localization of the edit by applying Extended Attention with a sparse set of spatiotemporal locations extracted from the original scene, allowing the generation to focus on essential elements. Our approach is based on our observation that in the pre-trained T2V DiT model, keys and values locally encode corresponding video patches in both space and time.
- (3) Content Harmonization via Iterative Refinement. To guarantee precise pixel-level alignment of the generated content with the input video and achieve better harmonization, we iteratively refine the estimated edit by repeating the sampling process with AnchorExtAttn multiple times, while preserving all unedited regions.

###### QK⊤

. (1)

A · V where A = Attention(Q, K) = softmax

√

𝑑

To capture inter-token relationships, Rotary Position Embeddings (RoPE) [Su et al. 2024] are applied to the input queries and keys in the attention operation. In this work, we utilize a publicly available DiT-based model [Yang et al. 2024], CogVideoX, for augmenting realworld videos with newly generated dynamic content in a zero-shot manner.

- 4 Method

Given an input video Vorig and a textual instruction PVFX, our goal is to synthesize a new video VVFX in which new dynamic elements are seamlessly integrated into the existing scene. Tackling this task requires ensuring that the location and size of the new content align with the camera motion and the environment, while its actions and movements must respond appropriately to other dynamic objects present in the scene. Our framework (Fig. 2) addresses these challenges by incorporating the following key components:

Our methoddoes notrequire user-provided masks to pre-determine

the location of the new content, but rather infers it automatically. We leverage a text-based segmentation model [Zhang et al. 2024a] for two purposes: (i) Localization, by segmenting prominent elements from the original scene to apply AnchorExtAttn, which allows the T2V model to infer the natural location and dynamics of the new content based on the original scene context. (ii) Harmonization, by segmenting the newly added content to ensure pixel-level alignment with the original scene.

(1) VLM as a VFX Assistant.We utilize a pre-trained VLM to interpret the user’s instructions, reason about the interactions with the scene’s dynamics, and identify both the prominent existing elements in the scene and the new content to be added. The VLM provides a descriptive prompt of the desired scene augmentation, a list of prominent existing elements, and the new content to be added. We achieve this by guiding the VLM to act as a “VFX

###### Sampling Baseline Full Method

###### Sampling with Extended Attention

[Figure 53]

[Figure 54]

[Figure 55]

T=0.9 T=0.6 Full Ext. Att. Masked Ext. Att. Anchor Ext. Att.

Ours

Input

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

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

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

“Add a knight riding the horse”

(a)

###### (b) (d) (e) (f)

(c)

- Fig. 3. Controlling Fidelity to the Original Scene Using Different Extended Attention Mechanisms. (a-b) SDEdit suffers from the original scene preservation/edit fidelity trade-off. (c-e) Three Extended Attention variants during sampling demonstrate different control levels: Full Extended Attention closely reconstructs the input scene, Masked Extended Attention proves too constrained in overlapping regions despite allowing new content emergence, and our Anchor Extended Attention. achieves optimal results by applying dropout – extending attention only at sparse points within selected regions.
- 4.1 VLM as a VFX Assistant

timestep fails to retain the original scene, resulting in misaligned new content, whereas a low noising timestep limits deviations from the original video.

To create a fully automatic framework that requires only a simple user instruction, we incorporate a VLM into our framework. Specifically, given a user instruction PVFX, along with keyframes from Vorig, we instruct the VLM [Achiam et al. 2023] to provide a composition prompt Pcomp describing the newly augmented scene. We use PVFX as the edit prompt for the T2V generation process. Hence, the edit prompt must include the following details: (i) original scene elements and actions, (ii) new content elements and actions, and

To tackle the localization challenge, we extend the attention module during sampling to include the input video’s corresponding attention features. Specifically, we apply DDIM inversion [Song et al. 2020] to the original video Vorig and extract the spatiotemporal keys and values 𝐾orig,𝑉orig from the attention module of every block in the network and generation timestep 𝑡. These keys and values are then used to extend the attention mechanism during sampling with P𝑐𝑜𝑚𝑝 to control the localization of the edit.

- (iii) the interaction between them. When simply asking the VLM to

yield PVFX, it often fails to produce captions that are both true to the original video and descriptive of the new content well. Therefore, using these for T2V generation can result in edits with low fidelity to the instruction (see examples in Sec. E.2 and Fig. 15 in the SM).

When using all keys and values, the extended attention operation can be expressed as:

Attn(𝑄VFX, [𝐾VFX,𝐾orig], [𝑉VFX,𝑉orig]) . (2)

To resolve this issue, we guide the VLM in an in-context manner by instructing it via a system prompt to imagine a conversation with a VFX artist to obtain a caption that would describe the composited scene correctly. We observe that this helps the VLM to achieve better scene understanding. Specifically, the system prompt includes guidelines to focus on: (1) spatial and dynamic awareness of existing scene elements, (2) preservation of original scene behaviors, and (3) atmospheric coherence between new and existing content.

We observe that extending attention to the full set of keys and values approximately reconstructs the original video (Fig. 3(c)). This demonstrates that the keys and values of attention layers in the DiT-based model determine more than just appearance information. In contrast to UNet-based architectures, keys and values locally encode corresponding video patches. We hypothesize that this occurs because the same positional embedding is applied to 𝐾orig as to 𝐾VFX. Remarkably, applying the same positional embedding to both source and target keys and values enables alignment between the generated and original content at corresponding spatiotemporal locations.

Additionally, we utilize the VLM to yield a list of prominent foreground objects in the original video Oorig and the object that will be added according to the edit prompt Oedit. See additional details in Sec. A.3, Sec. A.4, and Sec. E in the SM. In addition, Figs. 16-19 in the SM include an overview of the protocol, system prompts, and output examples.

Selective Attention. Based on our observation, we propose restricting the extended attention only to specific positions in the source scene. Specifically, we use a selection function F to determine which keys and values are retained.

- 4.2 Localization via Anchor Extended Attention A pivotal aspect of our method is the accurate placement of the

Masked Extended Attention. A straightforward choice is regionbased masking: identifying the most critical regions for preserving scene coherence and extending attention with keys and values within these masked regions 𝑀orig, i.e., F (𝑥) = 𝑀orig ◦ 𝑥.

new content. While Pcomp can describe the desired location, naive noising-denoising with this prompt introduces a trade-off: As shown in Figs. 3 (a)-3(b), using SDEdit [Meng et al. 2022] with a high noising

(b) w/o AnchrExtAttn. and Iter. Refin.

(a) Input (e) Our method

(c) w/o AnchorExtAttn. (d) w/o Iter Refin.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

FrameiFramej

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

- Fig. 4. Ablations. (b) Excluding both AnchorExtAttn and the Iterative refinement process results in significant misalignment with the original scene and poor harmonization (e.g., the size of the puppy relative to the scene and boundary artifacts). (c) Omitting AnchorExtAttn leads to incorrect positioning of the new content. (d) Removing iterative refinement results in poor harmonization. Our full method (e) exhibits good localization and harmonization of the edit.

To get 𝑀orig, we ask a VLM for Oorig, a list of foreground elements in Vorig (typically the most spatially prominent elements within a scene), and obtain the corresponding masks using a text-based segmentation model (see Sec. A.3 and Sec. A.4 in the SM for more details and examples of 𝑀orig).

As shown in Fig. 3(d), this strategy successfully preserves high fidelity to the original scene within𝑀orig while allowing new content to emerge in unmasked regions. However, Fig. 3(d) reveals two issues: (i) due to the hard constraint in overlapping regions, it is not suitable for handling natural occlusions, and (ii) the lack of guidance in unmasked regions can lead to inaccurate scale or positioning relative to the input scene.

Anchor Extended Attention. To achieve robust and spatially coherent integration of new content, we propose to selectively extend keys/values in a sparse set of foreground anchors and a sparser set of background anchors. Introducing Anchor Extended Attention with its formulation:

AnchorExtAtt :=Attn(𝑄VFX, [𝐾VFX,𝐾𝐸 ], [𝑉VFX,𝑉𝐸 ]) , s.t. 𝐾𝐸 := F(𝐾orig) 𝑎𝑛𝑑 𝑉𝐸 := F(𝑉orig) , F(𝑥) := DropFG(𝑀orig) ∪ DropBG(∼ 𝑀orig) ◦ 𝑥 .

(3)

For each layer, a selection of anchor points is randomly sampled via dropout, in both masked and unmasked regions. To steer the localization of the edit, we extend the attention with anchors that correspond to spatiotemporal locations in Vorig, allowing the generation to focus on essential elements (anchors) to achieve robust and spatially coherent integration of new content. (e.g., the knight’s motion is aligned with the horse in Fig. 3(e)). This balanced approach offers flexibility for creative edits while preserving key spatial cues from the original scene.

4.3 Content Harmonization

Our anchor extended attention steers the placement of the new content to align with the original scene. However, it does not guarantee precise pixel-level alignment. As can be seen in Fig. 3(e), the legs of the horse are not perfectly aligned with the original video. To guarantee pixel-level alignment, a straightforward approach is to

extract a mask of the new content 𝑴VFXrgb from the sampling with AnchorExtAttn (Eq. 3) output Vˆcomp. More concretely, we obtain this mask by applying a text-based segmentation model using the added object description provided by the VLM. The mask can then be used to replace the pixels outside it with the corresponding pix-

els from the original video: Vcomp[∼ 𝑴VFXrgb ] = Vorig[∼ 𝑴VFXrgb ]. While this preserves the unaffected regions, it often results in poor

harmonization with the input video.

To improve content harmonization, we propose repeating the sampling process with AnchorExtAttn (Eq. 3) multiple times, progressively reducing the level of noise added at each step. This iterative approach gradually refines the new content’s interaction with the original scene.

As shown in Fig. 2, we update 𝒙comp by adding a residual latent 𝒙res = 𝑴VFX · (𝒙ˆcomp −𝒙orig) to 𝒙orig, where 𝑴VFX is the latent representation of 𝑴rgbVFX (see Sec. A.4 in SM for details on computing the latent 𝑴VFX). Notably, the addition of 𝒙res is equivalent to directly replacing values of 𝒙orig with those of 𝒙ˆcomp within 𝑴VFX. The final edited video is obtained by decoding their sum with the T2V model’s VAE decoder. This allows each iteration to adjust the generated content’s high-frequency details to better match the original video. We provide implementation details and summarize our method in Sec. A and Alg. 1 in the SM.

Input video Input video

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Add a golden retriever follows the joggers

Add a parrot on the other side of the window

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Input video Input video

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Add Luminous angelic wings on the woman's back

Add a white duck walking in front of the boots

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Input video Input video

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Add a huge gummy bear running next to the dog

Add a gira e walking

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

###### Input video

###### Input video

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Add dinosaurs reaching to bite the leaves

Add a massive tsunami ﬂooding the city, apocalyptic

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

- Fig. 5. Sample Results of DynVFX. Our method supports a wide range of scene augmentations across diverse scenarios while maintaining realistic interaction, occlusion, lighting, and camera motion, for example: a golden retriever consistent with camera movement, transparent wings revealing the woman’s silhouette at sunset, and a tsunami flooding the city yet realistically respecting the car dashboard. See SM for full videos.

###### Input SDEdit DDIM Inversion Finetuning Gen-3 FlowEdit Ours

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Add a bear dancing with the woman

[Figure 169]

[Figure 170]

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

Add a ﬁre breathing dragon chasing the dog

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

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Add a pair of deer drinking water from the creek

- Fig. 6. Qualitative Comparison of Text-Based Methods. Sample results comparing our method to SDEdit [Meng et al. 2022], DDIM inversion [Song et al. 2020], Lora fine-tuning [Hu et al. 2022], Gen-3 [Runway 2024] and FlowEdit [Kulikov et al. 2024]. As can be seen, our method better augments the original scene with new dynamic content that interacts naturally with existing elements in the scene. See SM for full video comparison.

- 5 Results

We evaluated our method over a set of 57 video-text edit pairs containing 36 unique publicly available videos from the web and the DAVIS dataset [Pont-Tuset et al. 2017]. These videos feature a wide range of complex scenes in terms of camera and object motion, lighting conditions, and physical environments. Our videos and implementation details are available in the SM.

Figs. 1 and 5 show sample results of our method. As seen, our method facilitates natural integration of a broad range of visual effects, ranging from environmental effects (a tsunami in Fig. 4 and an explosion in Fig. 5) to new object insertion (knight riding a horse in Fig. 5 and dancing bear in Fig. 6). In all examples, the new content is naturally localized in the scene, even in challenging scenarios involving multiple objects (dinosaurs or workers in Fig. 5) and partial occlusions (a puppy in Fig. 1 and a giraffe in Fig. 5).

- 5.1 Qualitative Evaluation

We compare our method to the following baselines: (i) SDEdit [Meng et al. 2022] using the same T2V model as ours, (ii) DDIM inversion

[Song et al. 2020] and sampling with the target prompt, (iii) LORA fine-tuning [Hu et al. 2022] of the T2V model and sampling with the target prompt, (iv) Gen-3 [Runway 2024] video-to-video model, designed for video stylization, (v) FlowEdit [Kulikov et al. 2024] and two-stage editing approaches: (vi) AnyV2V [Ku et al. 2024] and (vii) I2VEdit [Ouyang et al. 2024].

Figs. 6 and 7 show a qualitative comparison with the baselines. As can be seen, all baselines exhibit different limitations in maintaining scene fidelity while introducing new content. SDEdit [Meng et al. 2022] and FlowEdit [Kulikov et al. 2024] both manage to fulfill the edit prompt, yet the scene may significantly deviate from the original scene in terms of appearance, motion, positioning, or scale (e.g., deer in the creek). DDIM inversion is not suitable for editing. LORA fine-tuning suffers from the trade-off between preserving aspects of the original scene and adding new content to the scene: either overfitting original scene appearance (e.g., adding a dragon-dog hybrid), or underfitting the original layout (e.g., incorrect scale of deer). Gen-3 is conditioned on structure and content representation extracted from the input video, hence it tends to significantly alter

###### Input video Ours Input video First Fame Editing AnyV2V I2VEdit

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Gemini I2I

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

+

Additional Input

|[Figure 209]|
|---|

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

Add a puppy playing with the woman

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Gemini I2I

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

+

Additional Input

|[Figure 227]|
|---|

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

Add a massive explosion from the mountain

Fig. 7. Comparison to Two-Stage Editing Methods. Left pane: our results. Right pane: AnyV2V and I2VEdit. See SM for video comparisons.

scene appearance and does not allow the insertion of new objects that change the scene layout. Both AnyV2V and I2VEdit take the first edited frame as additional input, but still struggle to maintain the coherent motion of the generated objects. In particular, AnyV2V often fails to accurately propagate the edits through the video, leading to distorted geometries. In each case, these limitations affect the overall scene coherence and realism of the added elements. Our method successfully adds new content to the scene, achieving high fidelity to the user instructions while allowing for natural interactions between original and added elements (e.g., natural interaction between woman and bear). We provide details about the baselines comparison in Sec. B in the SM. Additionally, we provide qualitative comparisons with VFX reference-based methods [KlingAI 2024; Pika 2024] in Sec. D and Fig. 12 in the SM.

- 5.2 Quantitative Evaluation

We numerically evaluate our results using the following metrics:

(i) Edit fidelity. Following previous works (e.g., [Hsu et al. 2025; Tewel et al. 2024a]), we measure per-frame Directional CLIP Similarity [Gal et al. 2022; Radford et al. 2021] to assess the alignment between the change in the source and the target prompt, and the change between the source and edited frames.

- (ii) Original content preservation. We evaluate how well the edited video preserves the original content outside the modified region. To this end, we segment the new content in the edited video using [Zhang et al. 2024a], and compute the masked Structural Similarity Index (SSIM) over the regions complementary to the edited ones.
- (iii) VLM quality evaluation. Inspired by [Hsu et al. 2025; Huang et al. 2024], we employ a VLM to expand the per-frame metrics above as follows. We input several frames from the edited videos into the VLM and instruct it to evaluate four key aspects: how well the edit follows the text prompt (Text Alignment), the overall visual quality of the edited frames (Visual Quality), how well the new content is harmonized with the source frames (Edit Harmonization), and the realism of the added object’s dynamics relative to the scene (Dynamics Score). For each aspect, the VLM outputs a score between 0 and 1, with higher scores indicating better performance. In the SM, we include our evaluation protocol and the justification of its use as an evaluation metric, showing its alignment with human preferences and consistent outcomes with an alternative VLM backbone. We include this in Sec. F and Table 2 in the SM.

Table 1 and Fig. 9 in the SM present the results of the described metrics on a set of 57 video-text edit pairs comprising 34 unique videos. As shown, our method outperforms the baselines in both

Table 1. Quantitative Comparison. We compare our method with other video editing methods and ablations using metrics for edit fidelity (CLIP Directional) and original content preservation (masked SSIM), a VLM-based evaluation of text alignment, visual quality, harmonization, and motion realism, and a user study on content integration and edit harmonization. User study values indicate the percentage of participants who preferred our method over each baseline.

Method Metrics VLM-based evaluation User Study

CLIP Directional

Text Alignment

Visual Quality

Edit Harmonization

Dynamics Score

Content Integration

Edit Harmonization

SSIM

Gen-3 0.142 0.283 0.410 0.586 0.363 0.374 98.687 94.691 LORA finetuning 0.292 0.349 0.781 0.755 0.726 0.714 92.221 71.968 DDIM inv. sampling 0.193 0.440 0.465 0.609 0.448 0.448 98.951 93.145 SDEdit (0.9) 0.290 0.321 0.778 0.750 0.717 0.730 99.154 75.645 SDEdit (0.6) 0.105 0.558 0.381 0.703 0.406 0.390 92.655 92.470 FlowEdit 0.288 0.414 0.771 0.765 0.712 0.736 97.346 65.494 AnyV2V 0.300 0.468 0.678 0.669 0.615 0.619 91.142 77.747 I2VEdit 0.288 0.524 0.769 0.773 0.724 0.719 91.030 82.563

w/o AnchorExtAttn 0.325 0.691 0.765 0.713 0.670 0.687 81.417 81.109 w/o Iterative Refinement 0.291 0.736 0.749 0.741 0.697 0.696 80.122 80.492 w/o VLM Protocol 0.254 0.741 0.705 0.755 0.669 0.667 71.360 73.708 Ours 0.307 0.749 0.843 0.784 0.778 0.775 - -

SSIM and Directional CLIP metrics, demonstrating superior edit fidelity while maintaining higher structural similarity in the unedited regions. The VLM-based evaluation aligns with this assessment and further shows that our method produces videos that achieve better content integration and greater motion realism.

- (iv) User study. We conducted a user study to evaluate the ability to integrate new content while preserving the original video. Participants were shown the input video, a text description of the new content, our result, and a baseline output. They were asked two questions: “Which video better preserves the original footage while adding new content?” and “Which video better integrates the new content in a realistic and seamless way?”. In total, we collected 17,160 user judgments from 330 users. As seen in Table 1, our method is consistently preferred over all baselines.

5.3 Ablations

We ablate key design choices of our method: Anchor Extended Attention, iterative updates of the edit, and our VLM as a VFX Assistant protocol, by excluding each component from our framework. The ablation of our VLM protocol can be found in Sec. E.2 and Fig. 15 in the SM.

As seen in Fig. 4(c), omitting AnchorExtAttn leads to new content being misaligned relative to the original scene, with the added content poorly integrated into the original scene. Applying only the first iteration of our method (w/o iterative refinement, Fig. 4(d)) results in a better alignment with the input video, but the composition is still unstable, as evident, for example, in the puppy’s body hovering over the box in the first scene. Our full method achieves better composition with proper spatial relationships, demonstrating the importance of both components for realistic scene editing (Fig. 4(f)). We numerically evaluate each ablation with the same set of metrics described in Sec 5.2 and report them in Table 1. An additional ablation example can be found in Fig. 14 in the SM.

###### Input video

[Figure 233]

[Figure 234]

[Figure 235]

Add a ﬁsh bowl encircling the boy's head,

[Figure 236]

[Figure 237]

[Figure 238]

Fig. 8. Limitations. In some cases, the T2V model struggles to precisely follow the edit prompt.

6 Discussion and Conclusions

We introduced the task of augmenting real videos with new dynamic content based on a user-provided instruction. We presented a zeroshot method utilizing the T2V model in a feature manipulation framework, enabling correct localization and natural blending of new content with existing video elements.

As our method is built upon the pre-trained T2V model, the quality of the generated edits is inherently tied to the performance and capabilities of the underlying model. As shown in Fig. 8, the T2V model sometimes struggles to generate videos that precisely follow the edit prompt; additional failure cases and limitations are detailed in Sec. C and Fig. 11 in the SM. While the quality of the edits is bound by the capabilities of the T2V model, our method is model-agnostic and can be re-implemented with future transformer-based T2V models. We anticipate its performance will improve as more powerful models become available. Even under these constraints, our method significantly outperforms baselines, expanding the capabilities of pre-trained text-to-video diffusion models.

References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 Technical Report. arXiv preprint arXiv:2303.08774 (2023).

Adobe. 2024. After Effects. https://www.adobe.com/products/aftereffects. Anthropic. 2024. Claude Haiku 3.5 [Large language model]. https://www.anthropic.

com. Autodesk. 2024. Maya. https://autodesk.com/maya. Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. 2024. Lumiere: A Space-Time Diffusion Model for Video Generation. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. 2022. Text2live: Text-driven layered image and video editing. In European conference on computer vision. Springer, 707–723.

Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. 2025. VideoPainter: Any-length Video Inpainting and Editing with Plugand-Play Context Control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–12.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Blender. 2024. VFX (Visual Effects). https://www.blender.org/features/vfx. Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. InstructPix2Pix: Learning

to Follow Image Editing Instructions. In CVPR. Alper Canberk, Maksym Bondarenko, Ege Ozguroglu, Ruoshi Liu, and Carl Vondrick.

2024. EraseDraw: Learning to Insert Objects by Erasing Them from Images. In European Conference on Computer Vision. Springer, 144–160.

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing. In Proceedings of the IEEE/CVF international conference on computer vision. 22560–22570.

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. 2024. Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13320–13331.

Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. 2023. FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing. arXiv preprint arXiv:2310.05922 (2023).

Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators. ACM Transactions on Graphics (TOG) 41, 4 (2022), 1–13.

Gemini 2.0. 2024. Google AI Studio. https://aistudio.google.com. Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2023. TokenFlow: Consistent

Diffusion Features for Consistent Video Editing. arXiv preprint arxiv:2307.10373

(2023).

Jiaqi Guo, Lianli Gao, Junchen Zhu, Jiaxin Zhang, Siyang Li, and Jingkuan Song. 2024. MagicVFX: Visual Effects Synthesis in Just Minutes. In Proceedings of the 32nd ACM International Conference on Multimedia. 8238–8246.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic

models. Advances in neural information processing systems 33 (2020), 6840–6851. Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. 2022. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868 (2022).

Hao-Yu Hsu, Chih-Hao Lin, Albert J Zhai, Hongchi Xia, and Shenlong Wang. 2025. AutoVFX: Physically Realistic Video Editing from Natural Language Instructions. In 2025 International Conference on 3D Vision (3DV). IEEE, 769–780.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 1, 2 (2022), 3.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2024. VBench: Comprehensive Benchmark Suite for Video Generative Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21807–21818.

Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. 2024. Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9212–9221.

Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15954–15964.

Yoonjeon Kim, Soohyun Ryu, Yeonsung Jung, Hyunkoo Lee, Joowon Kim, June Yong Yang, Jaeryong Hwang, and Eunho Yang. 2025. Preserve or Modify? Context-Aware Evaluation for Balancing Preservation and Modification in Text-Guided Image Editing. In Proceedings of the Computer Vision and Pattern Recognition Conference. 23474–23483.

Diederik P. Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Optimization.

arXiv preprint arXiv:1412.6980 1412, 6 (2014). KlingAI. 2024. KlingAI Platform. https://www.klingai.com. Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong,

Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. 2024. AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks. arXiv preprint arXiv:2403.14468 (2024).

Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli.

2024. FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models. arXiv preprint arXiv:2412.08629 (2024).

Yao-Chih Lee, Ji-Ze Genevieve Jang Jang, Yi-Ting Chen, Elizabeth Qiu, and Jia-Bin Huang. 2023. Shape-aware Text-driven Layered Video Editing Demo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14317–14326.

Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. 2023. LLM-grounded Video Diffusion Models. arXiv preprint arXiv:2309.17444 (2023).

Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. 2023. VideoDirectorGPT: Consistent Multi-Scene Video Generation via LLM-Guided Planning. arXiv preprint arXiv:2309.15091 (2023).

Jingwei Ma, Erika Lu, Roni Paiss, Shiran Zada, Aleksander Holynski, Tali Dekel, Brian Curless, Michael Rubinstein, and Forrester Cole. 2024. VidPanos: Generative Panoramic Videos from Casual Panning Videos. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Joanna Materzyńska, Josef Sivic, Eli Shechtman, Antonio Torralba, Richard Zhang, and Bryan Russell. 2024. NewMove: Customizing text-to-video models with novel motions. In Proceedings of the Asian Conference on Computer Vision. 1634–1651. Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations. Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang.

2024. ReVideo: Remake a Video with Motion and Content Control. Advances in Neural Information Processing Systems 37 (2024), 18481–18505.

Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. 2024. I2VEdit: FirstFrame-Guided Video Editing via Image-to-Video Diffusion Models. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Geon Yeong Park, Hyeonho Jeong, Sang Wan Lee, and Jong Chul Ye. 2025. Spectral motion alignment for video motion transfer using diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 6398–6405.

William Peebles and Saining Xie. 2023. Scalable Diffusion Models with Transformers. In

Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205. Pika. 2024. Pika AI Art. https://pika.art. Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung,

and Luc Van Gool. 2017. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675 (2017).

Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15932–15942.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In International Conference on Machine Learning. PmLR, 8748–8763.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2024. SAM 2: Segment Anything in Images and Videos. arXiv preprint arXiv:2408.00714 (2024).

Runway. 2024. RunwayML. https://runwayml.com. Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. 2024. Emu Edit: Precise Image Editing via Recognition and Generation Tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8871–8879.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. pmlr, 2256–2265.

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising Diffusion Implicit Models. In International Conference on Learning Representations.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568 (2024), 127063.

Yoad Tewel, Rinon Gal, Dvir Samuel, Yuval Atzmon, Lior Wolf, and Gal Chechik. 2024a. Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models.

arXiv preprint arXiv:2411.07232 (2024).

Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. 2024b. Training-Free Consistent Text-to-Image Generation. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–18.

Yuanpeng Tu, Hao Luo, Xi Chen, Sihui Ji, Xiang Bai, and Hengshuang Zhao. 2025. VideoAnydoor: High-fidelity Video Object Insertion with Precise Motion Control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–11.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2023. Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 1921– 1930.

Unreal Engine. 2024. Unreal Engine. https://www.unrealengine.com. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N

Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. Advances in neural information processing systems 30 (2017).

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. 2023. VideoComposer: Compositional Video Synthesis with Motion Controllability. Advances in Neural Information Processing Systems 36 (2023), 7594–7611.

Navve Wasserman, Noam Rotstein, Roy Ganz, and Ron Kimmel. 2025. Paint by Inpaint: Learning to Add Image Objects by Removing Them First. In Proceedings of the Computer Vision and Pattern Recognition Conference. 18313–18324.

Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen. 2024. ObjectDrop: Bootstrapping Counterfactuals for Photorealistic Object Removal and Insertion. In Computer Vision – ECCV 2024. Springer-Verlag, 112–129.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023. Tune-a-video: Oneshot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF international conference on computer vision. 7623–7633.

Tianhe Wu, Kede Ma, Jie Liang, Yujiu Yang, and Lei Zhang. 2024. A Comprehensive Study of Multimodal Large Language Models for Image Quality Assessment. In European Conference on Computer Vision. Springer, 143–160.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv preprint arXiv:2408.06072 (2024).

Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. 2024. Spacetime diffusion features for zero-shot text-driven motion transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8466–8476.

Emilie Yu, Kevin Blackburn-Matzen, Cuong Nguyen, Oliver Wang, Rubaiat Habib Kazi, and Adrien Bousseau. 2023. VideoDoodles: Hand-Drawn Animations on Videos with Scene-Aware Canvases. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–12.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. 2023a. MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing. Advances in Neural Information Processing Systems 36 (2023), 31428–31449.

Shiwei* Zhang, Jiayu* Wang, Yingya* Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qing, Xiang Wang, Deli Zhao, and Jingren Zhou. 2023b. I2VGen-XL: High-Quality Imageto-Video Synthesis via Cascaded Diffusion Models. (2023).

Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. 2024d. HIVE: Harnessing Human Feedback for Instructional Visual Editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9026–9036.

Yuxuan Zhang, Tianheng Cheng, Rui Hu, Lei Liu, Heng Liu, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. 2024a. EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model. arXiv preprint arXiv:2406.20076 (2024).

Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, XIAOPENG ZHANG, Wangmeng Zuo, and Qi Tian. 2024b. ControlVideo: Training-free Controllable Text-to-video Generation. In The Twelfth International Conference on Learning Representations. https: //openreview.net/forum?id=5a79AqFr0c

Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris Metaxas, and Licheng Yu. 2024c. AVID: Any-Length Video Inpainting with Diffusion Model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 7162–7172.

Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jia-Wei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. 2024. Motiondirector: Motion customization of text-to- video diffusion models. In European Conference on Computer Vision. Springer, 273–290.

Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. 2023. ProPainter: Improving Propagation and Transformer for Video Inpainting. In Proceedings of the IEEE/CVF international conference on computer vision. 10477–10486.

Bojia Zi, Shihao Zhao, Xianbiao Qi, Jianan Wang, Yukai Shi, Qianyu Chen, Bin Liang, Rong Xiao, Kam-Fai Wong, and Lei Zhang. 2025. CoCoCo: Improving Text-Guided

Video Inpainting for Better Consistency, Controllability and Compatibility. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 11067–11076.

#### Appendix

Table of Contents

- A Implementation Details 13

- A.1 Models 13
- A.2 Keys and Values Extraction 13
- A.3 Obtaining 𝑀𝑜𝑟𝑖𝑔 13
- A.4 Latent Mask Extraction 14
- A.5 Runtime and Memory Usage 14

- B Baselines Comparison Details 14
- C Limitations 14
- D Additional Qualitative Comparisons 15
- E VLM Prompting 16

- E.1 Our VLM as a VFX Assistant Protocol 16
- E.2 Ablation of the VLM as a VFX Assistant Protocol 16

- F VLM as Evaluation Metric Validation 17

- A Implementation Details A.1 Models

Text-to-Video Model. We use a publicly available CogVideoX-5B [Hong et al. 2022; Yang et al. 2024] text-to-video model, which can generate videos with up to 480 × 720 pixels resolution, 6 seconds in length, 49 frames at 8 fps. This model is a transformer-based model that processes both text and video modalities together.

Segmentation Model. To segment the prominent objects in the videoandthenewlygenerated content, we utilize EVF-SAM2 [Zhang et al. 2024a] - a text-based video segmentation model based on SAM2 [Ravi et al. 2024].

Visual Language Model. Our vision-language model of choice is GPT-4o [Achiam et al. 2023], which we use through the official OpenAI API.

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| |[Figure 239]<br><br>| | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

- Fig. 9. Metrics Visualization. We measure CLIP Directional score (higher is better) and masked SSIM (higher is better). Our method demonstrates a better balance between these two metrics. See Sec. 5.2 for full discussion.

###### Algorithm 1 DynVFX Algorithm

###### Input:

Vorig, PVFX ⊲ Input video & instruction prompt 𝜏A ⊲ Extended Attention threshold Ψ ⊲ Video segmentation model VLM ⊲ Vision Language model

###### Preprocess:

Pcomp ← VLM[Vorig, PVFX] ⊲ Composition Prompt Oorig, Oedit ← VLM[Vorig, PVFX] ⊲ Original objects and VFX object 𝑀𝑜𝑟𝑖𝑔 ← Get-Latent-Mask(Ψ; Vorig, Oorig) ⊲ Extract source masks 𝑥𝑜𝑟𝑖𝑔 ← Encode[Vorig] ⊲ Encode video into latent space Korig, Vorig ← DDIM-Inv[𝑥𝑜𝑟𝑖𝑔] ∀𝑡 ∈ [𝑇 ]

For 𝑡 =𝑇, . . .,𝑇˜ 𝑚𝑖𝑛 do 𝒙𝑟𝑒𝑠 = 0 ⊲ initialize the residual latent 𝒙𝑐𝑜𝑚𝑝 = 𝑥𝑜𝑟𝑖𝑔 + 𝑥𝑟𝑒𝑠

if 𝑡 > 𝜏𝐴 then 𝐾𝐸,𝑉𝐸 ← F(𝐾orig,𝑀orig), F(𝑉orig,𝑀orig) else 𝐾𝐸,𝑉𝐸 ← ∅ 𝒙ˆ𝑐𝑜𝑚𝑝 ← Sampling[𝑥𝑐𝑜𝑚𝑝, Pcomp,𝑡;AnchorExtAttn[𝐾𝐸,𝑉𝐸 ]]

Vˆ𝑐𝑜𝑚𝑝 ← Decode(𝒙ˆ𝑐𝑜𝑚𝑝) ⊲ Decode latent 𝑴𝑉 𝐹𝑋 ← Get-Latent-Mask(Ψ; Vˆ𝑐𝑜𝑚𝑝, Oedit) ⊲ Extract VFX masks 𝒙𝑟𝑒𝑠 = 𝑴𝑉 𝐹𝑋 · (𝑥ˆ𝑐𝑜𝑚𝑝 − 𝑥𝑜𝑟𝑖𝑔)

𝒙𝑐𝑜𝑚𝑝 = 𝑥𝑜𝑟𝑖𝑔 + 𝑥𝑟𝑒𝑠 Vcomp ← Decode[𝑥𝑐𝑜𝑚𝑝 ] ⊲ Output video Output: Vcomp

- A.2 Keys and Values Extraction

Following [Tumanyan et al. 2023; Yatim et al. 2024], to obtain the T2V diffusion model intermediate latents, we apply DDIM inversion (applying DDIM sampling in reverse order) on the input video. We perform inversion solely to extract the keys and values for AnchorExtAttn, while the sampling itself is initialized from a noisy 𝑥𝑐𝑜𝑚𝑝. For the inversion process, we use 250 steps, with an empty string as the text prompt. We derive keys and values from the noisy latents produced by the inversion process (not from the reconstruction). During the sampling iterations in our method, we use these extracted keys and values for AnchorExtAttn. (Eq. 3) randomly samples a selection of anchor points, which are re-selected in each layer of the model. For all our results, we fix the hyperparameters and only adjust the dropout percentage in the AnchorExtAttn: 5% dropout for background and 70% for foreground in regular edits, while global edits use 10% for the foreground.

- A.3 Obtaining 𝑀𝑜𝑟𝑖𝑔

To get 𝑀orig, we ask a VLM to provide a list of foreground objects in the original video Oorig (typically the most spatially prominent elements within a scene) and then we obtain corresponding masks using EVF-SAM [Zhang et al. 2024a] - a text-based segmentation model.

Specifically, given the user instruction, the input video’s key frames and Pcomp, our VLM protocol identifies the prominent elements from the original scene using the system prompt in Fig. 18. Next, each element (represented by words) is fed to EVF-SAM to produce masks, and their union forms 𝑀𝑜𝑟𝑖𝑔, which is used in our AnchorExtAttn. Examples of such masks can be seen in Fig. 10.

Input video Input video

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Add a ﬂeet of pirate ships sailing in the ocean Add a whale in the background

Original Masks Original Masks

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Result Result

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Input video Input video

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Add a puppy running beside the woman Add a bear dancing with the woman

Original Masks Original Masks

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Result Result

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

- Fig. 10. Qualitative Examples of 𝑀𝑜𝑟𝑖𝑔. Each example includes the input video, user instruction, original masks, and corresponding result. In the topright whale edit, our VLM protocol identifies the prominent elements to be coral and divers. In the top left pirate ships edit, the prominent elements are the cliffs. In the bottom left puppy edit, the woman and the ocean are identified. And in the bottom right bear edit, the woman and the couch are chosen.

- A.4 Latent Mask Extraction As discussed in Sec. 4.3, we iteratively update the residual latent

𝒙𝑟𝑒𝑠 in the regions where the new content appears. This requires calculating the mask of the new content in the latent space. To do this, we first apply the segmentation model [Zhang et al. 2024a] to the current output of SDEdit and get the mask of the new content in RGB space. However, the VAE in the T2V diffusion model involves both spatial and temporal downsampling, making it challenging to directly map RGB pixels to their corresponding latent regions. To address this, we encode the RGB masks through the VAE-Encoder and apply clustering to partition the resulting latents into two groups, effectively producing downsampled masks that align with the latent space representation.

- A.5 Runtime and Memory Usage

Our method’s most computationally intensive parts are - DDIM inversion, which takes 15 minutes, and iterative updates of the edit residual, which takes 20 minutes, while a single iteration of sampling with AnchoExtAttn takes between 2-5 minutes (depending on the noise level). Importantly, DDIM inversion needs to be performed only once per video and can support multiple subsequent edits, making the process more efficient when applying various modifications to the same video content. Querying the VLM and EVF-SAM adds negligible runtime ( 5 and 20 seconds correspondingly). We run our method on a NVIDIA A100, with 72GB memory in usage.

- B Baselines Comparison Details

The Baseline runtimes are (i) SDEdit - 5 minutes (ii) DDIM inversion + Sampling - 20 minutes (iii) LORA fine-tuning - 2 hours,

(iv) Gen-3 - 30 seconds, (v) FlowEdit - 3 minutes, (vi) AnyV2V 3 minutes and (vii) I2VEdit - 30 minutes.

In terms of utilized Video Models, baselines (i) SDEdit, (ii) DDIMinversion+Sampling, and (iii) LORA fine-tuning utilized the same text-to-video model [Yang et al. 2024] as our method, (iv) Gen3 utilizes a proprietary video-to-video model by Runway (Alpha model via the publicly accessible web-based API), (v) FlowEdit is applied with Hunyuan Video model [Kong et al. 2024], (vi) AnyV2V uses I2VGen-XL model [Zhang et al. 2023b], supporting 15 frames, and (vii) I2VEdit utilizes Stable Video Diffusion model [Blattmann et al. 2023] supporting 27 frames. For AnyV2V and I2VEdit, the first frames were edited with Gemini image editing [Gemini 2.0 2024].

For LORA fine-tuning baseline, we use the following default hyperparameters: Adam optimizer [Kingma and Ba 2014], 1𝑒 − 4 learning rank, LORA rank 128, 800 fine-tuning steps. For comparison with Gen-3 [Runway 2024], we set the "Structure Transformation" hyperparameter to 5.

For FlowEdit, we use source and target prompts produced from our VLM-VFX protocol, as this method heavily relies on source and target prompting.

C Limitations

[Figure 276]

[Figure 277]

[Figure 278]

###### Inputvideo

Add a Poodle running next to the white Havanese dog

[Figure 279]

[Figure 280]

[Figure 281]

Add a German Shephard running next to the white Havanese dog (a)FailureCase(b)SuccessCase

[Figure 282]

[Figure 283]

[Figure 284]

Fig. 11. Failure Case Example. When trying to add content that already exists in the scene (e.g., an additional dog), it is best to specify the differences between them in the user instructions. As can be seen, when trying to add a Poodle (similar appearance and size to a Havanese), EVF-SAM fails to distinguish between the two. However, when trying to add a German Shepherd, the segmentation model manages to make the distinction.

Our method relies on the target segmentation provided by a textbased segmentation model [Zhang et al. 2024a]. While this solution often succeeds for most edits, it can sometimes produce inaccurate masks and fail to account for effects like shadows and reflections if not specified in the text prompt. To handle such cases, users can add a specification in text to include these secondary effects. Furthermore, the segmentation model may struggle in scenarios where multiple entities with similar visual characteristics are present in the scene. As shown in Fig. 11, segmentation issues can occur when adding content of similar appearance and size. However, when the

###### Input video Ours Input video Refference image Pika Kling

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

###### +

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

First Frame Editing

An enthusiastic audience, clapping and snapping photos, as the dancer takes a bow

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Additional Input

Pika

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

###### +

Kling

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Add dinosaurs eating leaves from trees, taking big bites

- Fig. 12. Comparison to Reference-Based Insertion Methods. The left plane shows the results of our method, while the right plane shows the results of Pika and Kling. As can be seen, these reference-based methods struggle to generate global effects.

Original Frames MagicVFX Ours

“colorful bubbles”

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

“car on ﬁre”

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

- Fig. 13. Comparison to MagicVFX. MagicVFX (second column) deviates significantly from the original video. As seen in the third column, our method successfully adds new content while maintaining high fidelity to the original scene.

entities are distinguishable enough, the new content is successfully integrated into the scene.

Furthermore, our framework performs a single-shot edit and does not support interactive, in-context refinements - once an edit is applied, users cannot request follow-up adjustments (e.g., tweaking object placement or appearance) without restarting the process. While VFX pipelines typically involve several feedback-driven iterations to perfect a scene, our current design does not support such incremental revisions. Future work could integrate an interactive editing loop, potentially via an autoregressive T2V model, to allow users to make successive edits without reinitializing the entire workflow.

D Additional Qualitative Comparisons

We perform additional qualitative comparisons to MagicVFX [Guo et al. 2024] and two reference-based methods: Pikadditions [Pika 2024] and Kling [KlingAI 2024]. As can be seen in Fig. 13, MagicVFX struggles to remain faithful to the original scene and has lower visual quality compared to our method. Furthermore, as illustrated in Fig. 12, both Pikadditions and Kling struggle to generate global effects. As can be seen, the audience members are incorrectly overlaid in front of the dancer. Additionally, the dinosaur occludes one of the

(b) w/o AnchrExtAttn. and Iter. Refin.

(a) Input (e) Our method

(c) w/o AnchorExtAttn. (d) w/o Iter Refin.

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

FrameiFramej

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

- Fig. 14. Additional Example for Ablations. (b) Excluding both AnchorExtAttn and the Iterative refinement process results in significant misalignment with the original scene and poor harmonization (e.g., the size of the puppy relative to the scene and boundary artifacts). (c) Omitting AnchorExtAttn leads to incorrect positioning of the new content. (d) Removing iterative refinement results in poor harmonization. Our full method (e) exhibits good localization and harmonization of the edit.

bikers, and only one dinosaur is generated, although we prompt for a global edit. This demonstrates scenarios where text-based insertion is preferable over reference-based insertion. Even though such methods do not explicitly require the control of location and dynamics of the new content (as our method), they rely on the reference assets, which limits their applicability whenever a suitable example cannot be found for the given video.

- E VLM Prompting

- E.1 Our VLM as a VFX Assistant Protocol

Although the model is capable of providing descriptive captions of the source scene, in some cases, we observed that it fails to provide captions suitable for compositing VFX with the scene. To overcome this, we ask the model to imagine a conversation with a visual effects (VFX) artist to obtain a caption that would describe the composited scene correctly. In this conversation, GPT-4o will "consult" with a VFX artist about how the new content should be integrated into the scene. Based on their discussion, it will be asked to provide a caption that describes how the added content fits into the scene. This results in text prompts that encourage the generated output video to include a natural interaction between the new content and the original environment. In this prompt, we also ask the VLM to provide a list of prominent foreground objects in the original video: Oorig and the object that will be added according to the edit prompt: Oedit. The full prompt for the VLM is shown in Fig. 17.

In addition, as discussed in Sec. 5.2, we utilize the VLM for interpretable quality assessment. The full set of instructions for the VLM can be seen in Fig. 19.

- E.2 Ablation of the VLM as a VFX Assistant Protocol

To validate the importance of our VLM protocol, we performed an ablation by prompting the VLM with a simplified system prompt,

###### Input video w/o VLM protocol Ours

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

Add a group of jellyﬁsh ﬂoating

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

Add a parrot on the other side of the window

Fig. 15. Ablating VLM Protocol. Without the protocol (middle), the VLM often yields an unsuitable prompt for adding the desired content, while our protocol (right), gives Pcomp which allows for scene-aware additions to Oorig (see jellyfish and parrot behind the window). Additionally, as can be seen in Table 1 in the main paper, Table 2 and Fig. 9 from the SM, the metrics for this ablation are worse than those for our full method.

asking it to caption the original video and provide the edit prompt. As seen in Fig. 15, simplifying the system prompt results in misinterpretation of the instruction with respect to the scene or fails to add new content in general.

- F VLM as Evaluation Metric Validation

Table 2. VLM Quality Evaluation with an alternative backbone (Claude Haiku 3.5). We report text alignment, visual quality, edit harmonization, and dynamics score. The results closely match those obtained with GPT-4o in the main paper, showing consistent outcomes across all metrics and methods.

Text Align.

Visual Quality

Edit Harm.

Dynamics Score

Method

Gen-3 0.517 0.499 0.412 0.445 LORA finetuning 0.807 0.755 0.722 0.719 DDIM inv. sampling 0.423 0.487 0.384 0.385 SDEdit (0.9) 0.769 0.734 0.697 0.682 SDEdit (0.6) 0.420 0.561 0.425 0.405 FlowEdit 0.670 0.651 0.591 0.605 AnyV2V 0.693 0.683 0.625 0.625 I2VEdit 0.769 0.747 0.697 0.698

w/o AnchorExtAttn 0.656 0.620 0.547 0.556 w/o Iterative Refinement 0.704 0.672 0.631 0.634 w/o VLM Protocol 0.697 0.700 0.631 0.644 Ours 0.820 0.760 0.721 0.721

To show that our VLM-based evaluation aligns with human preferences, we performed the following assessments: Measured Pearson’s correlation coefficients between our VLM scores (Edit Harmonization and Dynamics) and the user study scores for Edit Harmonization, as both assess content integration quality. We observe a strong correlation 𝑟 = 0.82 (𝑝 = 0.025) and 𝑟 = 0.89 (𝑝 = 0.007). We compute Pearson’s correlation between the Text Alignment score and the CLIP Directional score - a metric commonly used to assess text alignment and shown to correlate with human preference [Kim et al. 2025] - and found a strong correlation: r=0.81 (p=0.029). These strong correlations suggest that our metric aligns with human judgments in the context of our task.

We utilize the same VLM (GPT-4o [Achiam et al. 2023]) in our pipeline and evaluation. However, this does not introduce bias in our evaluation due to the separation between the internal representations of the T2V model and the VLM’s reasoning. During evaluation, the VLM does not “recognize” the video generated by the T2V model as its own, making the assessment unbiased and independent of its earlier role. We further validate this by evaluating the results with a different VLM (Claude Haiku 3.5 [Anthropic 2024]) in Table 2 and demonstrate that it provides similar scoring to that reported in the paper.

Our VLM-based evaluation metrics are inspired by recent work such as AutoVFX [Hsu et al. 2025] and [Wu et al. 2024], which demonstrate the effectiveness of VLMs for automatic quality assessment tasks.

Input video

Key frames

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

User prompt

[Figure 367]

[Figure 368]

### System prompt

“Add a majestic elephant, with large tusks and ﬂapping ears”

Vision Language Model

[Figure 369]

[Figure 370]

VFX reasoning: “To integrate the majestic elephant into the forest scene,we'll position it slightly to the left, ensuring it appears naturally among the trees. The elephant's large tusks and flapping ears should catch the sunlight, creating subtle highlights and shadows on its body. We'll add soft shadows on the ground to match the existing light direction. The elephant's movement should gently rustle nearby foliage, adding realism. Its presence should not disrupt the tranquil atmosphere but rather enhance the scene's natural beauty, blending seamlessly with the existing environment.”

:

[’Tree trunks’, ‘sunlight rays’]

[Figure 371]

[Figure 372]

[Figure 373]

: [’Elephant’]

[Figure 374]

: “A majestic elephant stands gracefully among the tall trees, its large tusks gleaming in the filtered sunlight. The gentle flapping of its ears creates a soft rustling sound, harmonizing with the serene forest ambiance. Sunlight dances across its massive body, casting intricate shadows on the lush undergrowth. The elephant's presence adds a sense of grandeur to the tranquil setting, as it moves gently, causing nearby leaves to sway. The forest remains peaceful, with the elephant seamlessly blending into the natural beauty of the scene, enhancing the overall sense of wonder and tranquility.”

Output video

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

- Fig. 16. VLM as a VFX Assistant Overview. Given instructions and an input video, the VLM (grey) interprets the interaction, reasons about the integration,

and composes Pcomp to guide the T2V generation. Inputs include: (1) a system prompt (red) guiding the VLM to produce VFX reasoning (purple), (2) a user prompt describing the content to integrate into the scene (green), and (3) key frames from the input video. The VLM produces VFX reasoning (purple) describing spatial placement, lighting, and motion for seamless integration, followed by Pcomp (blue) captioning the target scene. Next, given the user prompt, the input video’s key frames and Pcomp, the VLM identifies prominent elements from the original scene Oorig (yellow) and the new content to be added Oedit (brown). These outputs are used in our framework for the application of SAM, used to aid localization and harmonization of the new content. As can be seen in the output video, in this example, a majestic elephant is seamlessly integrated into the scene.

[Figure 380]

You will receive a few images of the source scene and a description of new content to be added to the scene. It is possible that you will receive a source prompt as well.

Your task is to provide two captions based on the following steps: Source Scene Caption:

**Note**: If a source scene prompt is provided, use it as is! Provide a detailed description of the source scene without considering the added content. Focus on the existing objects, environment, and actions in the scene. Ensure the description maintains the original mood and setting. VFX Conversation: Imagine a conversation with a Visual E ects (VFX) artist about how the new content should be integrated into the scene. Remember, the new content can be objects or multiple objects or e ect or really anything the user provides. so be clear to explain this to the VFX artist. The new content should interact naturally with the environment (e.g., shadows, lighting, or physical elements like grass, water, or other objects) but without altering the dynamics of the source scene. The object must ﬁt into the scene without a ecting the original characters' behavior or actions. The interaction between new content and foreground object must be included (e.g. object A is interacting with object B). in terms of dynamics and motion as well. Describe how the object interacts and how it blends into the scene. Composited Scene Caption: Based on the conversation with the VFX artist, provide a caption that describes how the added content ﬁts into the scene. The caption must reﬂect natural interaction between the new content and the environment (e.g., lighting, shadows, physical e ects), while ensuring the original dynamics remain unchanged. The content should be aware of the surroundings, but the behavior, and ﬂow of the original scene should remain consistent.

The overall atmosphere might change of course due to this addition to scene.

**Output format*** - a dictionary with keys: "source_scene_caption", "vfx_conversation", and "composited_scene_caption".

- - **source_scene_caption**: source_scene_caption will be - A detailed caption of the source scene. If provided, use the given caption.
- - **vfx_conversation**: A simulated conversation about how the new content should be integrated into the scene.
- - **composited_scene_caption**: will be - A detailed caption of the composited scene, integrating the new content.

**Note**:The composited_scene_caption and source_scene caption must each have between 90-95 words. Extra words will be ignored.

**Note**:The vfx_conversation could be as long as required in order to succeed.

**Note**: Don't start the composited_scene_caption with - "The scene now.." or "Added to the scene" "Scene has transformed", the composited_scene_caption should be understandable to anyone that does not have access to the source_scene_caption. And you should not simply concatenate between the source and composition. You should have an entirely new caption that describes the essence of the integrated scene with both the source content and new content. Don't use anything similar to "now the scene"

###### Fig. 17. System Prompt We Input the Pre-trained VLM (VFX Assistant). The system prompt contains guidelines to supply the VLM with the context of our task in order for it to interpret the user instruction, reason about the interaction, and accordingly, output a scene description that will be used for editing.

[Figure 381]

You are provided with frames from a source video, a description of this video (Source scene caption), and a description of a scene after adding a new content (Composited scene caption). What are the foreground object/objects in the source scene (people, animals, etc..). What is the object that is added to the scene after editing it according to the composited scene caption? Output a list of foreground objects from the source scene (source_objects_list) and a string of the added object (target_object). A foreground object can also be part of an element/body part (tree trunk, The right <object>)

###### Fig. 18. System Prompt For Obtaining Oorig and Oedit. Given the user prompt, the input video’s key frames and Pcomp, the VLM identifies prominent elements from the original scene Oorig and the new content to be added Oedit. These outputs are used in our framework for the application of SAM, used to aid localization and harmonization of the new content.

[Figure 382]

You are a helpful assistant that pays attention to context and estimates the perceptual quality of provided videos, speciﬁcally for the task of integrating new content into a given video. I would like you to help me estimate the quality of an edited videos based on the original frames along with text descriptions. You will be shown four grids. Each grid will be of the following type: left column will contain three frames from the original video. The next 2 columns will each contain three frames from di erent video editing methods. Above each column there will be a caption (original video, 1, 2, ...). Each method's task is to integrate the new content into the source video according to the edit prompt. The prompt describing the original video is "{original_prompt}". The edit prompt for all of the methods is "{edit_prompt}". Now, please conduct a perceptual quality comparison in terms of 1) alignment with the edit prompt; 2) visual quality, 3) new content harmonization and 4) dynamics For each method provide a score from 0 to 1 for each of the ﬁve criteria with higher scores indicating better results. Your response must include a concise description regarding the perceptual quality of each method and a score to summarize quality for each criterion while well aligning with the given description.

- 1) When assessing the alignment with the edit prompt, consider how well the method follows the edit prompt and if the frames contain the desired content. If the method fails to follow the edit prompt, the score should be low.
- 2) For visual quality, consider how realistic the frames look - are there any visual artifacts, are the lighting and colors realistic, are the objects in the image recognizable.
- 3) For content harmonization - how well the content is harmonized with the scene, are the proportions of the added content correct, is the depth and perspective of the added content consistent with the scene. Is placement of the added object physically realistic - does it look like it belongs in the scene or does it look out of place. Are the occlusions of the added content consistent with the scene.
- 4) For dynamics assessment - how realistically the added object is moving relatively to the scene. Is its motion aligned with the camera motion of the original video? If the object, for example ﬂoats unrealistically or ﬂickers, the score should be low.

###### Fig. 19. System Prompt for VLM-Based Evaluation. containing the guidelines for the evaluation setup, where original and edited frames are compared against an edit prompt across four criteria: prompt alignment, visual quality, content harmonization, and dynamics.

