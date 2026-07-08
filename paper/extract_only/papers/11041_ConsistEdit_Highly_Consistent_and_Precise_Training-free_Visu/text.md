# arXiv:2510.17803v1[cs.CV]20Oct2025

## ConsistEdit: Highly Consistent and Precise Training-free Visual Editing

ZIXIN YIN, Hong Kong University of Science and Technology LING-HAO CHEN, Tsinghua University and International Digital Economy Academy LIONEL NI, Hong Kong University of Science and Technology, Guangzhou and Hong Kong University of Science and Technology XILI DAI, Hong Kong University of Science and Technology, Guangzhou

###### (a) Multi-round Editing

(b) Multi-reigon Editing

Source Image

1st edit 2nd edit 3rd edit

###### Existing method

single edit

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

Failed

“red T-shirt” ↓ “yellow T-shirt”

“beard … red T-Shirt” ↓ “orange beard … hair … green T-Shirt”

“red T-shirt” ↓ “yellow T-shirt”

“young man” ↓ “fat young man”

“classroom” ↓ “dark classroom”

(c) Fine-grained Editing Control (d) Video Editing

Existingmothod

SourceVideo

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Source Image

…

[Figure 31]

consistency strength

[Figure 32]

[Figure 33]

[Figure 34]

jacket”black“

jacket”blue“

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

“girl” ↓ “black cat”

↓

…

Ours

Fig. 1. (a) ConsistEdit enables multi-round editing by allowing users to specify both the target region and the nature of the editing through prompts. Unlike existing methods, it can perform structure-preserving (hair, clothing folds) and shape-changing with identity-preserving edits in edited regions while keeping non-edited regions intact. (b) ConsistEdit handles multi-region edits in one pass and preserves both the edited structure and unedited content. (c) Our method enables smooth control over consistency strength in the edited region. In contrast, existing approaches lack smooth transitions and often alter non-edited areas. (d) Beyond image editing and rectified flow models, ConsistEdit generalizes well to all MM-DiT variants, including diffusion and video models.

Recent advances in training-free attention control methods have enabled flexible and efficient text-guided editing capabilities for existing image and video generation models. However, current approaches struggle to simultaneously deliver strong editing strength while preserving consistency with the source. For instance, in color-editing tasks, they struggle to maintain structural consistency in edited regions while preserving the rest intact. This limitation becomes particularly critical in multi-round and video editing, where visual errors can accumulate over time. Moreover, most existing methods enforce global consistency, which limits their ability to modify individual attributes such as texture while preserving others, thereby hindering fine-grained editing. Recently, the architectural shift from U-Net to Multi-Modal Diffusion Transformers (MM-DiT) has brought significant improvements in generative performance and introduced a novel mechanism for integrating text and vision modalities. These advancements pave the way for overcoming challenges that previous methods failed to resolve. Through an in-depth analysis of MM-DiT, we identify three key insights into its attention mechanisms. Building on these, we propose ConsistEdit, a novel attention control method specifically tailored for MM-DiT. ConsistEdit incorporates vision-only attention control, mask-guided pre-attention fusion, and differentiated manipulation of the query, key, and value tokens

to produce consistent, prompt-aligned edits. Extensive experiments demonstrate that ConsistEdit achieves state-of-the-art performance across a wide range of image and video editing tasks, including both structure-consistent and structure-inconsistent scenarios. Unlike prior methods, it is the first approach to perform editing across all inference steps and attention layers without handcraft, significantly enhancing reliability and consistency, which enables robust multi-round and multi-region editing. Furthermore, it supports progressive adjustment of structural consistency, enabling finer control. ConsistEdit represents a significant advancement in generative model editing and unlocks the full editing potential of MM-DiT architectures. Here is the project website.

CCS Concepts: • Computing methodologies → Computer vision; Image manipulation.

Additional Key Words and Phrases: Image Editing, Video Editing, Diffusion Transformer, Latent Diffusion, Rectified Flow

1 Introduction

Attention control techniques, which manipulate the query (Q), key (K), and value (V) tokens in the attention mechanism, have been widely adopted because their training-free nature enables flexible

and efficient extensions of generative models to image and video editing tasks. For example, in image editing, Prompt-to-Prompt (P2P) [Hertz et al. 2023] introduced a method to control crossattention layers, enabling text-guided editing without the need for additional data or retraining. This inspired follow-up work in video editing, such as Video-P2P [Liu et al. 2024b], while Masactrl [Cao et al. 2023] extended P2P from rigid to non-rigid editing.

Despite these advancements, there are still two fundamental challenges in text-guided editing: (1) the method must not only modify content in according to the input text but also preserve consistency in both editing and non-editing regions. In editing regions, for example, structure should remain stable when changing color, and the character identity must stay recognizable when adjusting shape. In non-editing regions, all visual elements should remain identical to the original. However, these two requirements are often not simultaneously satisfied in existing training-free methods [Cao et al.

- 2023; Hertz et al. 2023; Jiao et al. 2025], leading to unacceptable results in tasks such as color and material modifications. As shown in part Fig. 1 (a), existing methods tend to introduce noticeable changes but compromise consistency in both edited and non-edited regions. Therefore, maintaining editing strength and consistency is essential for multi-round and video editing, where both iterative accumulation and additional temporal dimension can exacerbate visual errors. (2) Beyond the inability to satisfy both requirements, existing methods typically enforce overall (e.g., texture and structure) consistency, which severely limits performance in fine-grained editing. When a task requires preserving texture while altering structure, or vice versa, these methods often fail, see Fig. 1 (c). Allowing more targeted control, such as focusing consistency on structure alone, would enable more flexible and effective editing.

Amid these unresolved issues, the field of image and video generation has undergone astonishing advancements due to the transition in architectures from U-Net [Rombach et al. 2022] to Multi-Modal Diffusion Transformer (MM-DiT) [Esser et al. 2024], shedding a new light on solving these problems mentioned above. Since trainingfree attention control methods heavily depend on the underlying architecture of the generative model, it is crucial to study how to tailor it to MM-DiT. To date, only one work, DiTCtrl [Cai et al. 2025], has investigated attention control in MM-DiT, and even that does not target editing tasks. Instead, it targeted multi-prompt long video generation. As a result, the lack of investigation into the attention mechanisms of MM-DiT in editing tasks significantly limits existing approaches [Deng et al. 2025; Wang et al. 2025].

To address this gap, we conduct a detailed study of MM-DiT’s attention architecture, starting by contrasting it with the more commonly studied U-Net. In U-Net, cross-attention governs text guidance, while self-attention drives visual generation, resulting in a two-stage separation of modalities. In contrast, MM-DiT merges textual and visual information, applying self-attention to jointly process modalities. Through in-depth analysis and experimental exploration, we derive three key insights of MM-DiT models:

• Vision-only is crucial: Editing effectiveness relies on modifying only the vision parts, since interfering with text tokens often leads to generation instability (Fig. 4).

- • Homogeneous for all layers: Visualizations of the vision parts of Q, K, and V across attention layers (Fig. 2) show that, unlike U-Net, each layer in MM-DiT retains rich semantic content. Thus, attention control must be applied to all layers.
- • Strong structure controllability from Q and K: Applying attention control solely on the vision parts of Q, K results in strong controllable structural preservation (Fig. 9).

By grounding these insights, we introduce ConsistEdit, a novel attention control method specifically adapted to MM-DiT to address the challenges through three core operations: (1) Vision-only attention control: attention control is applied solely to the vision parts across all layers; (2) Pre-attention mask fusion: editing and non-editing regions are fused before the attention calculation; (3) Differentiated control for Q, K, and V: we apply distinct control mechanisms to Q and K for structure, and V for content.

Through extensive experiments, we show that ConsistEdit enables structurally consistent at finer levels such as lighting and shading in edited regions, while keeping non-edited regions unchanged. As a result, ConsistEdit can address the two fundamental challenges in text-guided editing mentioned before: (1) ConsistEdit achieves state-of-the-art (SOTA) performance across diverse editing tasks including structure-consistency and -inconsistency tasks, enabling iterative multi-round editing, as well as single-pass multi-region editing, see Fig. 1 (a) (b). Additionally, it demonstrates strong generalization across diverse generation models and editing tasks, including video editing, showcasing its versatility and practical potential, as shown in Fig. 1 (d) and 13. (2) Instead of enforcing overall consistency, ConsistEdit supports progressive adjustment of structural consistency, allowing fine-grained control in various downstream tasks, as shown in Fig. 1 (c).

To our best knowledge, ConsistEdit is the first approach that enables editing across all steps and layers without manual parameter adjustment, significantly improving reliability and consistency. Overall, we list our contributions as follows.

- • We identify three key insights from MM-DiT foundation generation models that enable effective training-free attention control for editing tasks.
- • We propose ConsistEdit, a novel attention control approach designed to extend the editing capabilities of MM-DiT-based models.
- • Our method supports both structure-consistent and -inconsistent edits while maintaining fidelity in non-edited regions. Extensive experiments demonstrate that ConsistEdit sets new SOTA results in both image and video editing tasks, and enables reliable multiround and multi-region editing.

2 Related Work 2.1 Text-to-image/video Generation

Earlyvisualgenerationmethods were primarily based on GANs [Reed et al. 2016; Tao et al. 2022; Wang et al. 2023; Yu et al. 2023] due to their ability to synthesize high-fidelity content. However, diffusion models [Guo et al. 2024; Ho et al. 2020; Reed et al. 2016; Rombach et al. 2022] have demonstrated superior generative performance, with U-Net-based architectures [Rombach et al. 2022] becoming the dominant paradigm. As U-Net designs encounter scalability

limitations in data and model parameter size, the field has progressively shifted toward transformer-based architectures, particularly diffusion transformers (DiT) [Peebles and Xie 2023]. Among these, MM-DiT [Esser et al. 2024] has emerged as a widely adopted backbone for text-conditional generation. Many recent state-of-the-art models [AI 2024; Esser et al. 2024; Kong et al. 2024; Labs 2024; Liu et al. 2025; Yang et al. 2024] build upon MM-DiT, achieving remarkable performance, such as SD3 [Esser et al. 2024] and FLUX [Labs 2024] for image generation, as well as CogVideoX [Yang et al. 2024] for video generation. In this work, we tailor a new attention control method for MM-DiT-based models.

- 2.2 Text-guided Editing

Early work focused on training-based approaches that leveraged generative models to achieve controllable image editing [Karras et al. 2019]. With the rapid advancement of generative models, attention has shifted toward training-free editing methods, which offer greater flexibility and efficiency. These training-free approaches generally fall into two categories: sampling-based and attention-based methods. Sampling-based approaches introduce controlled randomness into the generation process to enable flexible editing [HubermanSpiegelglas et al. 2024; Jiao et al. 2025; Kulikov et al. 2024], while attention-based methods achieve editing ability by directly altering attention tokens. Prompt-to-Prompt [Hertz et al. 2023] was the first to introduce attention manipulation on the cross-attention layers of U-Net, adopted in many subsequent editing methods [Chen et al. 2024; Yang et al.2023]. Video-P2P[Liu et al.2024b] extends this crossattention control to video editing. FateZero [Qi et al. 2023] combines self-attention with a blending mask derived from cross-attention features of the source prompt. Methods such as MasaCtrl [Cao et al.

- 2023] and DiTCtrl [Cai et al. 2025] employ similar attention control strategies, applied to U-Net and MM-DiT architectures respectively. Despite their differences, all existing attention control methods can be understood as multi-branch frameworks [Cai et al. 2025; Cao et al. 2023; Ju et al. 2024; Rout et al. 2025; Wang et al. 2025], and can be uniformly expressed as special cases of Eq. 3. Notably, all above methods rely on selectively manipulating specific inference steps or attention layers, which limits their robustness and consistency with respect to the source. In contrast, our approach is the first one requires no manual selection of steps or layers.

3 Method

- 3.1 Preliminary

- 3.1.1 Generation procedure. The current visual generation procedure is a systematical method which includes generation algorithm and foundation network architecture. 𝒛𝑇 −→ 𝒛𝑇−1 −→ . . . −→ 𝒛𝑡 −→

. . . −→ 𝒛0 shows the procedure for generating the final image or video from random noise 𝒛𝑇 in 𝑇 steps. The generation algorithm could be latent diffusion, flow matching, or rectified flow.

Beyond the generation algorithm, the foundation network architecture plays a crucial role in affecting the final generation results. In each step, the network 𝑓 (·) integrates the text prompt tokens P and the result of previous step 𝒛𝑡 to generate the result of the

−−−−−−−−−→𝑡 |P) 𝒛𝑡−1. The network 𝑓 (·) can be U-Net or MM-DiT. It takes the pair of (𝒛𝑡, P) as input, which present the

next step 𝒛𝑡−1: 𝒛𝑡 𝑓 (𝒛

vision 𝒛𝑡 and text P tokens respectively. It goes through each layer of the network, and Eq. 1 shows how each attention layer works.

### {𝒛𝑡 (𝑙), P(𝑙)} −−−−−−→𝒈(·) 𝑸𝑙, 𝑲𝑙, 𝑽𝑙,

(1)

𝑸𝑙 (𝑲𝑙)⊤

𝒛𝑡 (𝑙 + 1) = Attention(𝑸𝑙, 𝑲𝑙, 𝑽𝑙) = 𝑽𝑙 · Softmax

### √

.

𝑑

We unify the formulation of cross-attention and self-attention in Eq. 1. The function 𝒈(·) denotes a pre-attention operation that plays different roles in cross-attention and self-attention layers. Specifically, in the 𝑙-th layer of a U-Net, if it is a cross-attention layer, 𝒈(·) maps the text tokens P(𝑙) to 𝑲𝑙 and 𝑽𝑙, and maps the vision tokens 𝒛𝑡 (𝑙) to 𝑸𝑙. In contrast, for self-attention layers, 𝒈(·) ignores the text tokens and maps 𝒛𝑡 (𝑙) to all of 𝑸𝑙, 𝑲𝑙, and 𝑽𝑙.

In contrast, MM-DiT is a self-attention-only architecture, without cross-attention. Before computing attention, each MM-DiT block applies a pre-attention transformation 𝒈(·), which includes operations such as MLP modulation, residual connections, normalization, and other components. In each block, the pre-attention 𝒈(·) maps the vision 𝒛𝑡 (𝑙) and text P(𝑙) tokens respectively and concatenate every vision and text pair to get 𝑸𝑙, 𝑲𝑙, 𝑽𝑙. In other words, 𝑸𝑙, 𝑲𝑙, 𝑽𝑙 all contain text and vision parts.

- 3.1.2 Inversion. The inversion procedure aims to accurately reverse the generation process to recover the initial noise 𝒛𝑇 that can reconstruct the real image or video tokens 𝒛0.
- 3.1.3 Editing. The original editing method can trace back to the image processing era [Jähne 2005] and the task was formulated as follows:

𝑰𝑡𝑔 = (1 − 𝑴) ⊙ 𝑰𝑠 + 𝑴 ⊙ 𝑰𝑒, (2) where the user offers source image 𝑰𝑠 and editing regions (mask 𝑴). The goal of the editing task was to generate the edited content 𝑰𝑒 and then blend it back to the source image while preserving the non-edited regions of the source image. ⊙ denotes the element-wise Hadamard product of two matrices.

- 3.1.4 Attention control approach for training-free editing. The current visual editing methods in the background of generation models [Cao et al. 2023; Hertz et al. 2023], leverage the attention control approach to extend the editing capability of the foundation generation model in a training-free manner. In concrete, they employ a dual-network architecture: one network is dedicated to reconstruct-

ing the original source given the prompt tokens P𝑠 and random noise 𝒛𝑇, while the other is focused on editing. The dual-network shares the same network parameters.

We formulate the procedure of the editing in a way of dualnetwork architecture. By applying the generation process to the source 𝑰𝑠, we obtain the full generation trajectory of the source tokens: 𝒛𝑇 → 𝒛𝑇𝑠 −1 → · · · → 𝒛𝑠𝑡 → · · · → 𝒛𝑠0. At each step, the update

𝑡

−−−−−−−−−−→𝑠|P𝑠) 𝒛𝑠𝑡−1, and each attention layer is computed as 𝒛𝑠𝑡 (𝑙 + 1) = Attention(𝑸𝑠𝑙, 𝑲𝑠𝑙, 𝑽𝑠𝑙).

follows 𝒛𝑠𝑡 𝑓 (𝒛

The generation procedure for the target 𝑰𝑡𝑔 starting from the same noise, 𝒛𝑇 → 𝒛𝑇𝑡𝑔−1 → · · · → 𝒛𝑡𝑔𝑡 → · · · → 𝒛𝑡𝑔0 , and each step 𝒛𝑡𝑔𝑡

𝑓 (𝒛𝑡𝑔𝑡 |P𝑡 )

−−−−−−−−−−−→ 𝒛𝑡𝑔𝑡−1, are very similar to that of the source, but with different attention operation which we call it attention control.

while vision-only edits better preserve the source content. At lower consistency strength, both approaches perform similarly. These results clearly highlight that restricting attention control to the vision parts is critical for robust editing. The detailed implementations are provided in Appendix A.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Query

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Key

Hence, we would execute the attention control on the vision parts across all blocks. 𝑸ˆ𝑠𝑙, 𝑲ˆ𝑠𝑙 and 𝑽ˆ𝑠𝑙 have the same vision parts as 𝑸𝑠𝑙, 𝑲𝑠𝑙 and 𝑽𝑠𝑙 but the text parts are from 𝑸𝑡𝑔𝑙 , 𝑲𝑡𝑔𝑙 and 𝑽𝑡𝑔𝑙 .

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Value

###### MM-DiT Block Index

- Fig. 2. Visualization of projected Q, K, V vision tokens in attention layers of the MM-DiT blocks at 15th sampling step of prompt “A standing horse.”

- 3.2.2 Structural consistency in edited region. Besides the visual parts only replaced from source components, we also move the blending procedure before the attention operation. Then, after extensive exploration of all potential combinations of 𝑸, 𝑲, and 𝑽 from the source and target generation procedure, we find the combination shown in Eq. 4 best preserves structural consistency. The mask 𝑴, representing the editing region, is extracted from the source attention maps similar to Cai et al. [2025] and is applied only to the vision parts. We refer to this method as Structure Fusion. Furthermore, the spatially-resolved visualizations in Fig. 2 enable us to perform mask blending directly based on spatial regions. To enable controllable editing strength, we define the consistency strength 𝛼 as a ratio of steps for applying attention control, which determines the level of structural preservation during editing. Technically, in structure consist editing, the attention calculation can be formulated as:

𝑸˜𝑡𝑔𝑙 =

𝑴 ⊙ 𝑸ˆ𝑠𝑙 + (1 − 𝑴) ⊙ 𝑸𝑡𝑔𝑙 , if 𝑡 > (1 − 𝛼)𝑇 𝑴 ⊙ 𝑸𝑡𝑔𝑙 + (1 − 𝑴) ⊙ 𝑸𝑡𝑔𝑙 , otherwise

,

𝑲˜𝑡𝑔𝑙 =

𝑴 ⊙ 𝑲ˆ𝑠𝑙 + (1 − 𝑴) ⊙ 𝑲𝑡𝑔𝑙 , if 𝑡 > (1 − 𝛼)𝑇 𝑴 ⊙ 𝑲𝑡𝑔𝑙 + (1 − 𝑴) ⊙ 𝑲𝑡𝑔𝑙 , otherwise

,

𝒛𝑡𝑔𝑡 (𝑙 + 1) = Attention(𝑸˜𝑡𝑔𝑙 , 𝑲˜𝑡𝑔𝑙 , 𝑽𝑡𝑔𝑙 ).

(4)

This operation enforces structural consistency while enabling precise text control to adjust appearance and texture.

- 3.2.3 Content preservation in non-edited region. We find that using

𝑸ˆ𝑠𝑙 and 𝑲ˆ𝑠𝑙 in the non-editing regions can maintain structural consistency, but often leads to color shifts. To achieve high-fidelity content

preservation, we further use 𝑽ˆ𝑠𝑙 in non-editing regions, which yields the best results. We refer to following strategy as Content Fusion.

As a result, Eq. 5 defines the final formulation of ConsistEdit:

𝑸˜𝑡𝑔𝑙 =

𝑴 ⊙ 𝑸ˆ𝑠𝑙 + (1 − 𝑴) ⊙ 𝑸ˆ𝑠𝑙, if 𝑡 > (1 − 𝛼)𝑇 𝑴 ⊙ 𝑸𝑡𝑔𝑙 + (1 − 𝑴) ⊙ 𝑸ˆ𝑠𝑙, otherwise

,

𝑲˜𝑡𝑔𝑙 =

𝑴 ⊙ 𝑲ˆ𝑠𝑙 + (1 − 𝑴) ⊙ 𝑲ˆ𝑠𝑙, if 𝑡 > (1 − 𝛼)𝑇 𝑴 ⊙ 𝑲𝑡𝑔𝑙 + (1 − 𝑴) ⊙ 𝑲ˆ𝑠𝑙, otherwise

,

𝑽˜𝑡𝑔𝑙 = 𝑴 ⊙ 𝑽𝑡𝑔𝑙 + (1 − 𝑴) ⊙ 𝑽ˆ𝑠𝑙, 𝒛𝑡𝑔𝑡 (𝑙 + 1) = Attention(𝑸˜𝑡𝑔𝑙 , 𝑲˜𝑡𝑔𝑙 , 𝑽˜𝑡𝑔𝑙 ).

(5)

- 4 Experiments 4.1 Setup

When we get the 𝑸𝑡𝑔𝑙 , 𝑲𝑡𝑔𝑙 , 𝑽𝑡𝑔𝑙 from the 𝑙-th attention layer of 𝑡-th step in the target generation procedure, the attention control no longer apply them directly to attention operation, but replace some of them from the generation procedure of the source.

a standing horse seed 1234

{𝒛𝑡𝑔𝑡 (𝑙), P𝑡𝑔} −−−−−−→𝒈(·) 𝑸𝑡𝑔𝑙 , 𝑲𝑡𝑔𝑙 , 𝑽𝑡𝑔𝑙 , 𝑓𝑜𝑡 = Attention(𝑸𝑡𝑔𝑙 , 𝑲𝑠𝑙, 𝑽𝑠𝑙|𝑴𝑐), 𝑓𝑏𝑡 = Attention(𝑸𝑡𝑔𝑙 , 𝑲𝑠𝑙, 𝑽𝑠𝑙|1 − 𝑴𝑐),

2 5 8 11 14 16 19 23

(3)

𝒛𝑡𝑔𝑡 (𝑙 + 1) = (1 − 𝑴) ⊙ 𝑓𝑏𝑡 + 𝑴 ⊙ 𝑓𝑜𝑡.

Eq. 3 is a example of attention control formulation of MasaCtrl [Cao et al. 2023] and DiTCtrl [Cai et al. 2025], where they replace the 𝑲, 𝑽 from the source in self-attention layers. Here, 𝑴𝑐 and 𝑴 donate masks extracted from attention maps1 of the source and target generation procedure, respectively. 𝑴 is used as the attention mask.

- 3.2 ConsistEdit: A New Attention Control for MM-DiT

- 3.2.1 The analysis of the attention mechanism in MM-DiT. MMDiT [Esser et al. 2024] fundamentally differs from U-Net [Rombach et al. 2022] in its attention mechanism. In U-Net, cross-attention handles text guidance, while self-attention focuses on visual content generation, creating a two-stage process. In contrast, MM-DiT merges text and visual information and employs self-attention to process both modalities simultaneously. This architecture shift renders existing U-Net-based attention control methods ineffective.

For instance, DiTCtrl [Cai et al. 2025] adopts the strategy of MasaCtrl [Cao et al. 2023] by applying attention control to the latter blocks of the model. This design originates from the downsampling–upsampling structure of the U-Net encoder–decoder architecture, where MasaCtrl performs edits in the decoder stages. However, MM-DiT does not exhibit such stage separation, as it lacks a distinct decoder stage on which editing can be focused, as illustrated by the PCA decomposition visualization in Fig. 2. Consequently, directly transferring this strategy leads to structural artifacts, as shown in Fig. 10, Fig. 7, and Fig. 8. Further experiments (Fig. 11) confirm that editing across all blocks yields superior results.

Moreover, Fig. 4 compares FireFlow [Deng et al. 2025] and RFSolver [Wang et al. 2025], each using their original attention control method on either all parts (original) or vision-only parts of tokens. Under higher consistency strength, the original approach often fails,

4.1.1 Baselines. We compare our method against several recent state-of-the-art approaches built upon MM-DiT, including UniEditFlow [Jiao et al. 2025], DiTCtrl [Cai et al. 2025], FireFlow [Deng et al. 2025], RF-Solver [Wang et al. 2025], and SDEdit [Meng et al.

1DiTCtrl adopts all-one masks for both 𝑴𝑐 and 𝑴 in editing tasks, despite using extracted masks for long video generation.

###### Vision Parts Only

|“…𝑟𝑖𝑑𝑖𝑛𝑔 𝑎<br><br>𝑤ℎ𝑖𝑡𝑒 ℎ𝑜𝑟𝑠𝑒 …”|
|---|

[Figure 65]

MM-DiT Blocks Attention

𝑃𝑠 𝑧𝑇

Inv.

Source

× 𝑇

|𝑄෠𝑠𝑙|
|---|

|𝐾෡𝑠𝑙|
|---|

|𝑉෠𝑠𝑙|
|---|

|𝑧𝑠0| |
|---|---|
| | |

Input

[Figure 66]

| | | |
|---|---|---|
| | | |

Content

Structure Fusion

Fusion

Mask Extraction

“𝑎 𝑤ℎ𝑖𝑡𝑒

Mask Guided Attention Fusion

|𝐼𝑠∗|
|---|

Target

ℎ𝑜𝑟𝑠𝑒”

𝑄𝑡𝑔𝑙 𝐾𝑡𝑔𝑙 𝑉𝑡𝑔𝑙

Input

#### Copy

Structure Fusion

[Figure 67]

[Figure 68]

Content Fusion

𝐼𝑠

𝑀

|𝑄෨𝑡𝑔𝑙|
|---|

|𝐾෩𝑡𝑔𝑙|
|---|

|𝑉෨𝑡𝑔𝑙|
|---|

𝑧𝑇

𝑧𝑡𝑔0

Target × 𝑇 Output

MM-DiT Blocks Attention

“…𝑟𝑖𝑑𝑖𝑛𝑔 𝑎 𝑟𝑒𝑑 ℎ𝑜𝑟𝑠𝑒 …”

|𝐼𝑡𝑔|
|---|

𝑃𝑡𝑔

(a) Pipeline of ConsistEdit (b) Mask Guided Attention Fusion

- Fig. 3. (a) shows the ConsistEdit pipeline. Given a real image or video 𝑰𝑠 and source text tokens P𝑠, we first invert the source to obtain the vision tokens 𝒛𝑇 , which is concatenated with the target prompt tokens P𝑡𝑔 and passed into the generation process to produce the edited image or video 𝑰𝑡𝑔. During inference, a mask 𝑴 generated by our extraction method delineates editing and non-editing regions. We apply structure and content fusion to enable prompt-aligned edits while preserving structural consistency within edited regions and maintaining content integrity elsewhere. (b) illustrates the mask-guided attention fusion for timesteps where 𝑡 > (1 − 𝛼)𝑇, which is applied exclusively to the vision parts, while the text parts remain unchanged.

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

Ours

Source Image

End Steps++

…

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

…

Existing Editing

Source Image

“classroom” ↓ “dark classroom”

头发[100:190, 455:635]≈ dilate + 8 ⾐服[520:650, 190:450]

“beard … red T-Shirt” ↓ “orange beard … hair … green T-Shirt”

“young man” ↓ “fat young man”

“red T-shirt” ↓ “yellow T-shirt”

(b) Fine-grained Control Editing

“girl” ↓ “black cat”

(c) Video Editing

“red T-shirt” ↓ “yellow T-shirt”

[Figure 86]

Source Video “red SUV” → “blue SUV”

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

FireFlow

“white bunny” ↓ “pink bunny”

a watercolor painting of a white bunny in a cup of flowers seed 42

[Figure 91]

RF-Solver

“white bunny” ↓ “pink bunny”

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Inversion

[Figure 99]

“grey shirt” Mask

[Figure 100]

“grey shirt” ↓ “dark brown shirt”

Real Input Image Reconstruct Image First Edit (clothes)

“Yann LeCun” Mask

“is” ↓ “is shouting and”

Second Edit (motion)

“Yann LeCun” Mask

“is” ↓ “is with white color hair”

Third Edit (hair)

Yann LeCun is facing to the front in a grey shirt, seed42

Yann LeCun is shouting and facing to the front in a dark brown shirt, qk 6, mask .3

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Yann LeCun with white color hair is shouting and facing to the front in a dark brown shirt, qk 20mask .2

[Figure 105]

[Figure 106]

[Figure 107]

(a) Multi-round Multi-region Editing

First Edit Second Edit Third Edit

Directly Edit

Existing Editing

[Figure 108]

|[Figure 109]|
|---|

[Figure 110]

[Figure 111]

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

|[Figure 121]|
|---|

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Source Prompt

|Low Consistency Strength| |High Consistency Strength| |
|---|---|---|---|
|Original|Vision-only|Original|Vision-only|

- Fig. 4. Comparison of V token swapping strategies for content consistency. Swapping vision-only V tokens leads to superior content consistency under high consistency strength settings, while maintaining comparable editing capability to original methods when the consistency strength is low.

Table 1. Quantitative results of structural consistency comparison with RF-Solver and FireFlow using Canny SSIM ↑.

Sampler RF-Solver FireFlow

Edit Method

Fix seed 0.5507 0.5557 RF-Solver [Wang et al. 2025] 0.6225 FireFlow [Deng et al. 2025] — 0.5136 Ours 0.8714 0.8776

we use a fixed sampler and identical random seeds for each method within a comparison group, so that source images are consistent across all methods. For structure-consistent image editing, we adopt prompts on two tasks that require preserving the original structure: change color and change material, covering 80 image pairs in total. For structure-inconsistency image editing, we use the remaining cateogries including Change Object, Add Object, Delete Object, Change Content, Change Style, etc.

2022]. We focus exclusively on MM-DiT-based baselines, as previous works [Deng et al. 2025; Jiao et al. 2025] and our preliminary experiments (Fig. 6) show that U-Net-based methods perfom significantly worse. Methods (SDEdit) that can be adapted to MM-DiT are included in comparisons, while those cannot be transferred are excluded.

- 4.1.4 Metrics and settings. Unlike the original PIE-Bench, which uses structural distance [Tumanyan et al. 2022] to evaluate structural similarity, we employ the Structural Similarity Index (SSIM) [Wang et al. 2004] computed on Canny edge maps [Canny 1986] borrowed from Zhao et al. [2023] for a more accurate assessment. To evaluate the preservation of non-edited regions (a.k.a. BG preservation), we compute PSNR and SSIM exclusively on those regions, which are manually annotated by human annotators. The semantic alignment of the edits is assessed using CLIP similarity [Radford et al. 2021], applied to both the entire image and the edited regions.
- 4.2 Quantitative Evaluation

- 4.1.2 Implementation. We primarily conduct experiments using Stable Diffusion 3 Medium (a.k.a. SD3) [Esser et al. 2024] for image generation and CogVideoX-2B [Yang et al. 2024] for video generation, both of which employ a pure MM-DiT architecture. Unless otherwise specified, we use the Euler sampler and adopt UniEditFlow [Jiao et al. 2025] for inversion. For all baseline methods, we carefully tune the hyperparameters to ensure a fair comparison. Implementation details are provided in Appendix A.

- 4.1.3 Benchmark. We adopt prompts from PIE-Bench [Ju et al.

- 2024] which comprises 700 editing pairs across 10 types of edits. Although our method is fully compatible with inversion methods, we adopt a noise-to-image setting to better isolate and highlight the editing capabilities, minimizing the influence of reconstruction and inversion quality. To ensure fair comparison across baselines,

While prior editing methods [Cai et al. 2025; Cao et al. 2023; Hertz et al. 2023] typically lack quantitative evaluation, we incorporate evaluation metrics inspired by related tasks (i.e. PIE-Bench) to more effectively showcase the capabilities of our method.

###### (a) Multi-round Multi-region Editing

Directly Edit

First Edit Second Edit Third Edit

Existing Editing

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

||
|---|

“red T-shirt” ↓ “yellow T-shirt”

“beard … red T-Shirt” ↓ “orange beard … hair … green T-Shirt”

“red T-shirt” ↓ “yellow T-shirt”

“young man” ↓ “fat young man”

“classroom” ↓ “dark classroom”

Source Image

###### (b) Fine-grained Control Editing

(c) Video Editing

End Steps++

Source Image

…

Existing Editing

“girl” ↓ “black cat”

…

Ours

Source Video “red SUV” → “blue SUV”

头发[100:190, 455:635]≈ dilate + 8 ⾐服[520:650, 190:450]

|Low Consistency Strength| |High Consistency Strength| |
|---|---|---|---|
|Original|Vision-only|Original|Vision-only|

###### Source Prompt

“white bunny” ↓ “pink bunny”

FireFlow

consistency strength

4

“square” ↓ “small square”

8

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

11

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

“red square” ↓

“white bunny” ↓ “pink bunny”

“yellow star” an old style compute monitor on the desk shows a red square in the center of the screen with black background, seed 1234

RF-Solver

an old style compute monitor on the desk shows a small yellow star in the center of the screen with black background, seed 1234

Source

[Figure 184]

[Figure 185]

[Figure 186]

“red square” ↓ “white circle”

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

6 • Zixin Yin, et al.

a watercolor painting of a white bunny in a cup of flowers seed 42

###### Real Input Image Reconstruct Image First Edit (clothes)

###### Second Edit (motion)

###### Third Edit (hair)

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

“grey shirt” Mask

“Yann LeCun” Mask

“Yann LeCun” Mask

[Figure 198]

[Figure 199]

[Figure 200]

Real Input Image

DiTCtrl RF-Solver

UniEdit-Flow FireFlow

SDEdit

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

“grey shirt” ↓ “dark brown shirt”

“is” ↓ “is shouting and”

“is” ↓ “is with white color hair”

Inversion

a girl with a red hat and red t-shirt is sitting in a park, best quality”,

- Fig. 5. Real image multi-round editing results. Starting from a real image, we first perform inversion to project it into the latent space. We then sequentially edit the clothing color, motion, and hair.

Ours EditFriendly

InfEdit PnP-Inv. PnP P2P

"a girl with a yellow hat and red t-shirt is sitting in a park, best quality",

0, 2 2, 3; 20, 24

0, 2;13, 24

2, 3; 20, 24

Yann LeCun with white color hair is shouting and facing to the front in a dark brown shirt, qk 20mask .2

Yann LeCun is facing to the front in a grey shirt, seed42

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

4.3 Qualitative Evaluation

[Figure 211]

In this section, the evaluation begins with structure-consistent editing, highlighting the method’s ability to preserve structural consistency. This is followed by demonstrations on real images to validate practical effectiveness. Performance on structure-inconsistent editing is then presented, showcasing adaptability across varied scenarios. Finally, multi-round editing examples are provided, combining both structure-consistent and -inconsistent editing to demonstrate the method’s robustness and flexibility.

Yann LeCun is shouting and facing to the front in a dark brown shirt, qk 6, mask .3

Real Input Image Ours

DiTCtrl

UniEdit-Flow FireFlow

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

RF-Solver InfEdit

FreePromptEditing

EditFriendly

“a red hat” ↓ “a yellow hat”

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

- 4.3.1 Structure-consistent editing. Fig. 7 presents a qualitative comparison across all methods on structurally consistent editing tasks. Our approach accurately changes the color or material according to the target prompt while preserving the structure of the edited region same to that of the source image. Notably, beyond merely replacing colors, the edited outputs are also well adapted to the surrounding lighting conditions. In contrast, other methods often produce incorrect or insufficient edits and fail to maintain structural consistency. Additionally, our method faithfully preserves the non-edited regions, whereas others introduce undesirable changes. More results are shown in Appendix B.1.
- 4.3.2 Structure-consistent editing on real images. We compare our real image editing results with several existing methods, including UNet-based approaches such as FreePromptEditing [Liu et al. 2024a], EditFriendly [Huberman-Spiegelglas et al. 2024], InfEdit [Xu et al. 2023], PnP-Inv. [Ju et al. 2024], PnP [Tumanyan et al. 2023], and P2P [Hertz et al. 2023]. As shown in Fig. 6, conventional U-Net-based and MM-DiT-based methods all struggle to preserve the non-edited regions and often fail to accurately modify the hat color. In contrast, our method achieves the best performance, delivering precise edits in the target region while preserving the consistency of non-edited areas. Please refer to Appendix B.1 for further examples.
- 4.3.3 Structure-inconsistent editing. We compare various methods on structure-inconsistent editing tasks in Fig. 8. In these experiments, the consistency strength (𝛼) is set to 0.3, allowing the model to moderately edit structures for improved prompt alignment, while still preserving the overall layout. As shown, our method achieves better results in the edited regions, producing more precise editing with fewer artifacts. Moreover, it more effectively preserves the non-edited areas compared to other approaches, maintaining high content fidelity with respect to the source image. More results are shown in Appendix B.1.
- 4.3.4 Multi-round interactive editing on real images. Fig. 5 presents an example of multi-round editing on a real input image. The image

[Figure 221]

[Figure 222]

[Figure 223]

“white sofa” ↓ “brown sofa”

PnP-Inv. PnP P2P

SDEdit

Fig. 6. Qualitative comparison of methods on real image editing tasks.

Source

Depth Map Ours

"A modern living room with large windows, natural sunlight streaming in, a brown sectional sofa, a wooden coﬀee table, indoor plants near the corner

A modern living room with large windows, natural sunlight streaming in, a white sectional sofa, a wooden coﬀee table, indoor plants near the corner",

Table 2. Quantitative results of Change Color and Change Material tasks.

|Canny<br><br>|BG Preservation|Clip Similarity ↑|
|---|---|---|
|SSIM ↑<br><br>|PSNR ↑ SSIM ↑<br><br>|Whole Edited|

Method

SDEdit [2022] 0.6795 23.99 0.8697 26.59 22.80 UniEdit-Flow [2025] 0.8029 30.56 0.9554 26.55 22.59 DiTCtrl [2025] 0.8235 29.54 0.9632 26.63 22.97 Ours 0.8811 36.76 0.9869 27.19 23.73

4.2.1 Evaluation results. Tab. 1 reports a structural consistency comparison with RF-Solver [Wang et al. 2025] and FireFlow [Deng et al. 2025]. We evaluate each baseline using its native sampler and editing strategy, while applying our method under the same sampler but with our own attention control strategy. This setup ensures a fair comparison and demonstrates our method’s robustness across varying samplers. As shown in the table, the evaluation metrics of RF-Solver and FireFlow closely match those from fixed-seed generation, suggesting an inability to preserve structural consistency. Therefore, we exclude these two methods from subsequent comparisons on structure-consistent editing tasks. In contrast, our method consistently produces the best structure-preserving results.

Tab. 2 presents the whole benchmark with other baselines. Our method delivers superior results in both preserving source content and executing accurate edits, achieving state-of-the-art performance across the board.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

“bird” ↓ “crochet bird”

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

“black rocks” ↓ “white rocks”

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

“purple lilas” ↓ “orange lilas”

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

“orange circle” ↓ “cyan circle”

Source Ours DitCtrl UniEdit-Flow FireFlow RF-Solver SDEdit

Fig. 7. Qualitative comparison of methods on structure-consistent editing tasks.

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

is first inverted into the latent space. Then, we perform a series of edits on it, including modifying the clothing color, motion, and hair. These flexible operations and the reliability of the results open up new possibilities for interactive or iterative editing tasks.

Table 3. Evaluation of content preservation in non-edited regions. DiffEdit [2023] 𝑽 𝑸 & 𝑲 𝑸 & 𝑲 & 𝑽 (Ours)

“A pixel art of…” ↓ “…”

PSNR ↑ 51.49 37.98 24.32 38.85 SSIM ↑ 0.9972 0.9905 0.9286 0.9917

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

- 4.4 Fine-grained Editing Control

Fig. 9 shows the controllability of structural consistency during editing by varying the consistency strength (𝛼). A high value enforces strict structure preservation, even when the prompt includes shape-altering instructions. At the same time, it still enables accurate texture editing, e.g., color changes, as specified by the prompt. In contrast, a low consistency strength permits structural editing with the prompt. Additionally, the similar color appearance under varying consistency strengths demonstrates the effectiveness of our disentangled structure-preserving control mechanism, enabling precise and independent editing of structure and texture.

Furthermore, thanks to this disentanglement, our method enables smooth and controllable adjustment of consistency strength. In contrast, other methods struggle to maintain stable editing performance across varying strength levels, often relying on specific parameter values, as illustrated in Fig. 10. This property further highlights the potential for integrating a controllable consistency strength slider into interactive editing interfaces.

- 4.5 Ablation

- 4.5.1 Structural consistency in edited regions. As shown in Fig. 11, we conduct an ablation study to investigate the effects of different 𝑸 and 𝑲 tokens swapping strategies. Starting from the same seed ensures a well-initialized structural layout for subsequent editing. Swapping all (text and vision) 𝑸 and 𝑲 tokens preserves structural consistency to a certain extent but significantly impairs text-driven editability, as it discarding the text tokens of target. In contrast, selectively swapping only the vision part of 𝑸 and 𝑲 tokens across all blocks maintains the structural layout of the source image while preserving strong editing capabilities. To verify the necessary of swapping in all layers, we find that only swap the latter half of the model’s blocks will substantially weaken structural control and can lead to corrupted generation results. Finally, by incorporating our content fusion method on top of the full-block vision-only 𝑸 and 𝑲 swapping, we further enforce preservation in non-edited regions, achieving the best quality. These findings emphasize the importance of applying editing across all blocks while restricting editing to the vision parts of the attention mechanism.
- 4.5.2 Content preservation in non-edited regions. Tab. 3 reports PSNR and SSIM scores on the non-edited regions of 80 image pairs from the Change Color and Change Material tasks in PIE-Bench [Ju

“…” ↓ “… girl”

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

“cat” ↓ “tiger”

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

“laughing happily” ↓ “crying sadly”

Source Ours DitCtrl UniEdit-Flow FireFlow RF-Solver SDEdit

We conduct two ablation studies to validate the effectiveness of our approach.

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

“…” ↓ “A pixel art of…”

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

“…” ↓ “… girl”

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

“cat” ↓ “tiger”

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

###### b) Fine-Grained Control Editing

End Steps++

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

“laughing happily” ↓ “crying sadly”

Source Image

…

[Figure 315]

Existing Editing

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

Source Ours DiTCtrl UniEdit-Flow FireFlow RF-Solver SDEdit

“girl” ↓ “black cat”

…

Fig. 8. Qualitative comparison of methods on structure-inconsistent editing tasks.

边缘需要⼀致 边缘不需要⼀致

masactrl

Ours

###### consistency strength

consistency strength

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

[Figure 338]

4

“square” ↓ “small square”

8

SDEdit

SDEdit

[Figure 339]

[Figure 340]

[Figure 341]

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

11

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

“red square” ↓

“yellow star” an old style compute monitor on the desk shows a red square in the center of the screen with black background, seed 1234

RF-Solver

RF-Solver

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

[Figure 367]

[Figure 368]

an old style compute monitor on the desk shows a small yellow star in the center of the screen with black background, seed 1234

[Figure 369]

Source

[Figure 370]

[Figure 371]

[Figure 372]

Source

[Figure 373]

“red square” ↓ “white circle”

FireFlow

FireFlow

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

[Figure 386]

Fig. 9. Effect of consistency strength on structural consistency. High strength strictly enforces structural preservation, while low strength permits prompt-driven shape changes. Texture editing remains consistent, highlighting effective disentanglement.

“asian man” ↓ “african man”

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

UniEdit-Flow

UniEdit-Flow

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

DiTCtrl

et al. 2024], evaluating how well different methods preserve content consistency in non-edited regions. All methods use the same binary mask, extracted using our mask extraction method. According to the results in Fig. 12, we can see a hard replacement strategy described in DiffEdit [Couairon et al. 2023] introducing visible artifacts at transition boundaries. Secondly, swapping only the vision tokens of 𝑸 and 𝑲 maintains structural consistency but introduces slight color shifts, which degrade metric scores in Tab. 3. In contrast, swapping only the vision part of the 𝑽 tokens yields a more stable preservation.

DiTCtrl

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

Real Input Image Ours InfEdit PnP-Inv. EditFriendly MasaCtrl PnP P2P

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

Ours

Ours

Fig. 10. Qualitative comparison on consistency strength adjustment.

a girl with a red hat and red t-shirt is sitting in a park, best quality”,

a close-up shot of an asian man wearing white T-Shirt in a park, best quality seed 42 a close-up shot of an African man wearing white T-Shirt in a park, best quality seed 42

"a girl with a yellow hat and red t-shirt is sitting in a park, best quality",

碾压，solve 最佳参数的性能类似

Finally, Tab. 3 and Fig. 3 shows that combining vision-token swaps

Real Input Image

DiTCtrl RF-Solver

UniEdit-Flow FireFlow

SDEdit

调参简单strength，过渡丝滑 ⾮edit部分，可以完全⼀致

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

qk对应ﬁne-grained geometry

v对应coarse geometry和贴图

Ours EditFriendly

InfEdit PnP-Inv. PnP P2P

prompt产⽣贴图

0, 2 2, 3; 20, 24

0, 2;13, 24

[Figure 433]

[Figure 434]

[Figure 435]

“white sofa” ↓ “brown sofa”

Source

Depth Map Ours

"A modern living room with large windows, natural sunlight streaming in, a brown sectional sofa, a wooden coﬀee table, indoor plants near the corner

A modern living room with large windows, natural sunlight streaming in, a white sectional sofa, a wooden coﬀee table, indoor plants near the corner",

a cute yellow dog is playing with a purple ball, cartoon style a cute red dog is playing with a grey ball, cartoon style a cute yellow dog, a purple ball, 42

Source

close look up of a man in a shirt with alternating of blue and white stripes, cartoon style close look up of a man in a shirt with alternating of green and yellow stripes, cartoon style a shirt, 1000

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

a man in green T-shirt is standing next to a single white car, best quality" a man in blue T-shirt is standing next to a single red car, best quality a man in green T-shirt, a single white car, 4321

Target

cartoon style of two hands holding a red pill and a blue pill cartoon style of two hands holding a green pill and a yellow pill pill, 123

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

“green T-shirt … white car” ↓ “blue T-shirt … red car”

“yellow dog … purple ball” ↓ “red dog … grey ball”

“blue and white stripes” ↓ “green and yellow stripes”

“red sweater … black pants” ↓ “yellow sweater … blue pants”

“red and blue pill” ↓ “green and yellow pill”

a boy is wearing a red sweater and black pants a boy is wearing a yellow sweater and blue pants A red sweater, black pants, 123

ConsistEdit: Highly Consistent and Precise Training-free Visual Editing • 9

More multi region

Source

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Source Fix seed All Parts Full (Ours)

Prompt All Vision Parts

Half of Vision Parts

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

“white camel” ↓

“blue camel”

Ours

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

Source

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

“red ball”

↓ “green ball”

A young boy, dressed in a white shirt and blue shorts, strolls leisurely through a lush, green park. The sunlight ﬁlters through the dense canopy of oak trees, casting dappled shadows on his path. He carries a small, red backpack slung over one shoulder, and his brown hair is tousled by a gentle breeze. As he walks, he pauses to observe a playful squirrel darting up a tree, a look of curiosity lighting up his face. The park is alive with the sounds of birds chirping and children's laughter in the distance, creating a serene and vibrant atmosphere around him. Seed 0

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

A young boy, dressed in a white shirt and green shorts, strolls leisurely through a lush, green park. The sunlight ﬁlters through the dense canopy of oak trees, casting dappled shadows on his path. He carries a small, red backpack slung over one shoulder, and his brown hair is tousled by a gentle breeze. As he walks, he pauses to observe a playful squirrel darting up a tree, a look of curiosity lighting up his face. The park is alive with the sounds of birds chirping and children's laughter in the distance, creating a serene and vibrant atmosphere around him.

DiTCtrl

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

“a young woman” ↓

Ours

“an old woman”

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

- Fig. 11. Ablation study on attention control for structure consistency. We compare (1) fixed seed results, (2) swapping all Q and K tokens across all blocks, (3) swapping only the vision part of Q and K tokens in the last half of the blocks, (4) swapping only the vision part of Q and K tokens in all blocks, and (5) adding our non-editing region consistency module.

[Figure 493]

[Figure 494]

[Figure 495]

Source DiffEdit Sliding Mask (V) Consist Mask (V) Consist Mask (QK) Ours

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

Mona Lisa with a head, Smile",

"Mona Lisa with a dog head, Smile seed 1000, alpha=0.13

Source DiffEdit Value Query &Key Ours

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

- Fig. 12. Ablation study of non-edited region preservation. The edit prompt is “a head” → “a dog head”.

UniEdit-Flow

A detailed green toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.

0.7 strength

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

A detailed dark red toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.

DiTCtrl

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

42

FireFlow

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

UniEdit-Flow

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

RF-Solver

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

A rabbit is playing with a red ball in a gym cartoon style seed 4321

A young girl, is riding a white camel in a park, best qualit seed 1000

FireFlow

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

a photo of a head shot of the paitrait of a young woman, best quality seed 4321

SDEdit

1 17 33 49

a boy is wearing a red t-shirt sitting in a park, best quality a boy is wearing a white t-shirt sitting in a park, best quality

RF-Solver

A red plush ball on a pure white background, best quality

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

a white rabbit on a blue background magazine cover"

Source

The bright morning of the wonderful elf town

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

SDEdit

Fig. 14. Qualitative comparison of methods on video editing tasks. The edit prompt is “green toy ship” → “dark red toy ship”.

Target

More real image multi-round More video

“red T-shirt” ↓ “white T-shirt”

“red plush ball” ↓ “blue plush ball”

“blue background” ↓ “yellow background”

“blue rabbit” ↓ “white rabbit”

“bright morning” ↓ “dark night”

Recoloring

###### Relighting Animation Shape Deformation Material Change

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

Fig. 13. Examples of editing results with FLUX.

Source

of 𝑸, 𝑲, and 𝑽 achieves the best results in both quantitatively and qualitatively, as it preserves more details.

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

Target

4.6 Compatibility and Application

- 4.6.1 Generalization to MM-DiT variants. Our method not only works effectively with SD3 but also generalizes well to other MMDiT variants such as FLUX.1-dev [Labs 2024]. In Fig. 13, the consistent preservation of fine-grained details and the accurate adaptation of lighting-related reflections further highlight the potential of our approach when applied to more powerful future models.
- 4.6.2 Generalization to video editing. While our method has already been demonstrated to be agnostic to specific samplers, we further showcase its broad applicability across generation methods (e.g., diffusion models) and domains (e.g., video) by applying it to CogVideoX-2B [Yang et al. 2024], a diffusion-based video generation model. As shown in Fig. 14, our approach enables consistent and controllable editing in both the spatial and temporal domains. Importantly, small inconsistencies that may go unnoticed in static images

“bright morning” ↓ “dark night”

“black veil” ↓ “golden veil”

###### “smiling face” ↓ “laughing face”

“made of plastic” ↓ “made of bronze”

“man” ↓ “fat man”

Fig. 15. Examples of applications.

a pig made of plastic a pig made of bronze

a pig, 42

a transparent black veil covered on a plate of apples

a portrait of a handsome young man with smiling face in a room, best quality a portrait of a handsome young man with laughing face in a room, best quality

a portrait of a young man in a room best quality a portrait of a young fat man in a room best quality

a transparent golden veil covered on a plate of apples ,1234

often become amplified and distracting in videos. Our method effectively highlights its robustness and generalizability. Additional results are provided in Appendix B.1.

The bright morning of the wonderful modern city

a handsome young man, 42, 10

custom, 42, 8

The dark night of the wonderful modern city, 1001

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

veil, 41

###### Source

4.6.3 Application. Fig. 15 showcases our method’s versatility across several challenging editing tasks, including recoloring, relighting, animation, shape deformation, and material change. Extending these capabilities to video further amplifies creative possibilities by enabling temporally consistent and detailed edits. The strong editing

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

Rec.

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

Target

“blue paints” ↓ “green paints”

“red golf ball” ↓ “white golf ball”

“yellow rose” ↓ “green rose”

“light blue” ↓ “dark blue”

“blue spoon” ↓ “green spoon”

a red golf ball

a pair of light blue shoes

a blue spoon and a mug

a man wearing a light blue shirt and dark blue pants, best quality

a yellow rose

More real image multi-round

a green spoon and a mug, 42, 8

a white golf ball, 42, 4

a green rose, 42, 8

a pair of dark blue shoes, 42, 14

a man wearing a light blue shirt and dark green pants, best quality, 42, 4

power and ease of use highlight the broad potential of our approach for practical and scalable content creation.

5 Conclusion

In this work, we identify key limitations of existing training-free editing methods, including their inability to achieve both strong and consistent text-guided editing, as well as their lack of fine-grained control, where most prior approaches were designed for U-Net or naively applied to MM-DiT without architectural adaptation. To address this, we conduct a detailed analysis of the attention mechanism in MM-DiT and uncover three critical insights that reveal why existing methods fall short. Building on these findings, we propose ConsistEdit, a novel attention control method that operates exclusively on vision tokens. By separating editing and non-editing regions and applying differentiated attention manipulation, ConsistEdit achieves precise, structural consistent edits in edited regions while preserving content in non-edited regions.

Extensive experiments demonstrate that ConsistEdit achieves state-of-the-art performance across diverse image and video editing tasks, without requiring manual tuning. It delivers reliable performance out of the box while offering users fine-grained control over structural consistency. These findings highlight the potential of MM-DiT when paired with our attention control strategies.

Acknowledgments

We are deeply grateful to Morph Studio and its CEO, Huaizhe Xu, for their generous provision of the computational resources that made this work possible. We would also like to thank Heung-Yeung Shum, Baoyuan Wang, Lei Zhang, Xuan Ju, Guanlong Jiao, Yukai Shi, Shihao Zhao, and Bojia Zi for their valuable discussions and insightful feedback throughout the course of this research.

References

Stability AI. 2024. Stable Diffusion 3.5. https://github.com/Stability-AI/sd3.5. Accessed: May 2025.

Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. 2025. Ditctrl: Exploring attention control in multimodal diffusion transformer for tuning-free multi-prompt longer video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 7763–7772.

John Canny. 1986. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence 6 (1986), 679–698.

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision. 22560–22570.

Minghao Chen, Iro Laina, and Andrea Vedaldi. 2024. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF winter conference on applications of computer vision. 5343–5353.

Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. 2023. DiffEdit: Diffusion-based semantic image editing with mask guidance. In The Eleventh International Conference on Learning Representations.

Yingying Deng, Xiangyu He, Changwang Mei, Peisong Wang, and Fan Tang. 2025. FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing. In Forty-second International Conference on Machine Learning.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2024. ANIMATEDIFF: ANIMATE YOUR PERSONALIZED TEXT-TO-IMAGE DIFFUSION MODELS WITHOUT SPECIFIC TUNING. In 12th International Conference on Learning Representations, ICLR 2024.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohenor. 2023. Prompt-to-Prompt Image Editing with Cross-Attention Control. In The Eleventh International Conference on Learning Representations.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851. Jonathan Ho and Tim Salimans. 2022. Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. 2024. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12469–12478.

Bernd Jähne. 2005. Digital image processing. Springer Science & Business Media. Guanlong Jiao, Biqing Huang, Kuan-Chieh Wang, and Renjie Liao. 2025. UniEdit-

Flow: Unleashing Inversion and Editing in the Era of Flow Models. arXiv preprint arXiv:2504.13109 (2025).

Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. 2024. PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code. In The Twelfth International Conference on Learning Representations.

Tero Karras, Samuli Laine, and Timo Aila. 2019. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4401–4410.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. HunyuanVideo: A Systematic Framework For Large Video Generative Models. CoRR (2024).

Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli.

2024. Flowedit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629 (2024).

Black Forest Labs. 2024. Flux. https://github.com/black-forest-labs/flux. Accessed: May 2025.

Bingyan Liu, Chengyu Wang, Tingfeng Cao, Kui Jia, and Jun Huang. 2024a. Towards understanding cross and self-attention in stable diffusion for text-guided image editing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 7817–7826.

Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, et al. 2025. Generative video propagation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 17712–17722.

Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. 2024b. Video-p2p: Video editing with cross-attention control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8599–8608.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations. William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In

Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205.

Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15932–15942.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. 2016. Generative adversarial text to image synthesis. In International conference on machine learning. PMLR, 1060–1069.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. 2025. Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations. In The Thirteenth International Conference on Learning Representations.

Ming Tao, Hao Tang, Fei Wu, Xiao-Yuan Jing, Bing-Kun Bao, and Changsheng Xu. 2022. Df-gan: A simple and effective baseline for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 16515–16525.

Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2022. Splicing vit features for semantic appearance transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10748–10757.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2023. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1921–1930.

Duomin Wang, Yu Deng, Zixin Yin, Heung-Yeung Shum, and Baoyuan Wang. 2023. Progressive disentangled representation learning for fine-grained controllable talking head synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17979–17989.

Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. 2025. Taming Rectified Flow for Inversion and Editing. In

Forty-second International Conference on Machine Learning.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.

Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. 2023. Inversion-Free Image Editing with Natural Language. CoRR (2023).

Fei Yang, Shiqi Yang, Muhammad Atif Butt, Joost van de Weijer, et al. 2023. Dynamic prompt learning: Addressing cross-attention leakage for text-based image editing. Advances in Neural Information Processing Systems 36 (2023), 26291–26303.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. CoRR (2024).

Zhentao Yu, Zixin Yin, Deyu Zhou, Duomin Wang, Finn Wong, and Baoyuan Wang.

2023. Talking head generation with probabilistic audio-to-visual diffusion priors. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7645–7655.

Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. 2023. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems 36 (2023), 11127–11150.

ConsistEdit: Highly Consistent and Precise Training-free Visual Editing • 11

12 • Zixin Yin, et al.

A Implementation Details

- A.1 Inference Settings

We use 28inferencestepsforSD3[Esser et al.2024],50 forCogVideoX2B [Yang et al. 2024], and 20 for FLUX.1-dev [Labs 2024]. The classifier-free guidance (CFG) scale [Ho and Salimans 2022] is set to 7.5 for editing generated sources and 2.0 for real-image editing. All images are generated at 1024 × 1024, and all videos at 720 × 480. The inference device is 1× RTX-4090 GPU.

Our method is compatible with various samplers, with the Euler sampler adopted as default unless otherwise specified. It also supports various inversion techniques; in all experiments, we use the latest inversion method from UniEdit-Flow [Jiao et al. 2025]. We fix the consistency strength to 𝛼 = 1 for tasks requiring structural preservation, and set it to 𝛼 = 0.3 for other tasks.

We adopt a default mask threshold of 0.1, which consistently performs well across our experiments. This relatively coarse masking suffices thanks to the generation models’ strong global adaptation, allowing it to propagate edits from partial color cues to semantically aligned regions. The target object for mask extraction is identified either using “blended_word” keywords from PIE-Bench [Ju et al. 2024], or simply by extracting the noun of the object to be edited. Furthermore, our method supports externally provided masks, enabling users to integrate masks generated from other pipelines for more flexible control.

- A.2 Sampling Details

To accelerate inference and reduce the number of function evaluations (NFE) during sampling, similar to the approach in Wang et al. [2025], we first run the source prompt branch and cache the Q, K, and V tokens at each step and block for later use. During this stage, we also compute and store the final averaged editing mask. When editing with the target prompt, we load the stored Q, K, and V tokens from the source and apply the editing mask through MaskGuided Attention Fusion as needed. This strategy ensures that the mask extraction and editing process introduces no additional NFE, maintaining the same efficiency as standard sampling methods.

- A.3 Implementation of Compared Methods

Since some compared methods do not provide implementations for SD3 or CogVideoX-2B, or compatible sampling code, we reimplement them within the SD3 and CogVideoX-2B framework by faithfully following their original logic and carefully tuning hyperparameters to match the reported performance. Implementation details are as follows:

- • DiTCtrl [Cai et al. 2025]: For image editing, we set the start timestep to 2 and the end timestep to 17, applying edits to the last 5 blocks. For video editing, we use the official implementation. During the editing steps, K and V tokens are copied from the source branch to the target branch in the attention layers. For this method, consistency strength is controlled by increasing the number of end step during which K and V tokens are shared.
- • UniEdit-Flow [Jiao et al. 2025]: The official implementation is based on SD3, but only provides the parameter 𝜔 for CFG = 1. Following the similarity transformation described in the paper, we adopt 𝜔 = 5÷7.5 ≈ 0.6 and set 𝛼 = 0.6, which yields performance

“a horse” ↓ “a bronze horse”

“a house” ↓ “a stone house”

“green lizard” ↓ “brown lizard”

“white christmas tree” ↓ “green christmas tree”

“kids crayon drawing of…” ↓ “…”

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

“bee…” ↓ “…”

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

“mountain bicycle” ↓ “rusty mountain bicycle”

Source Ours DitCtrl UniEdit-Flow FireFlow RF-Solver SDEdit

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

Source

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

Target

Fig. 16. Examples of video editing.. The prompt is “red SUV” → “blue SUV”.

comparable to the original. The same settings are used for video generation. Under this setting, consistency strength is controlled by decreasing the value of 𝛼.

- • FireFlow [Deng et al. 2025]: We observe a significant drop in source–target consistency as CFG increases. Due to performance degradation when the number of edited timesteps increases (as shown in Fig. 4), we limit editing to timesteps from 0 to 3 across all blocks. For video generation, the end timestep is set to 9. During editing, V tokens are copied from source to target. In this approach, consistency strength is modulated by increasing the number of final step in which the V features are shared.
- • RF-Solver [Wang et al. 2025]: Similar to FireFlow, we set the editing range from timestep 0 to 7 for the latter half of the blocks. During editing, V tokens are copied from the source to the target branch. For video generation, the end timestep is set to 9. The end step of sharing of V tokens serves as the control mechanism for consistency strength in this method.
- • SDEdit [Meng et al. 2022]: We set 𝑡0 = 0.6 and apply editing to either generated source content or real input content, for both image and video generation tasks. For this method, consistency strength is controlled by decreasing the value of 𝑡0.

- A.4 Implementation of FLUX

FLUX [Labs 2024] is composed of several double blocks and single blocks. As noted by Wang et al. [2025], single blocks primarily encode general information relevant to generation. Therefore, we apply our editing methods specifically to the single blocks.

B Results and Analysis

- B.1 More Results

Additional image editing comparisons are presented in Fig. 21, covering both structure-consistent and structure-inconsistent editing tasks. The results demonstrate that our method achieves superior structural consistency, better preservation of non-edited regions, and enhanced editability compared to existing approaches.

We present additional results on video editing tasks in Fig. 16 and 18. Fig. 16 showcases examples generated by our method alone, while Fig. 18 provides comparisons with existing approaches, demonstrating our superior performance, particularly in scenarios with complex motion.

Fig. 17 presents additional cases of multi-region editing, demonstrating that our method can handle multi-object editing even in the presence of occlusion or complex geometric relationships. Notably, even when multiple regions exhibit intertwined textures, our method accurately identifies the target color for each region and

[Figure 630]

a transparent black veil covered on a plate of apples

ConsistEdit: Highly Consistent and Precise Training-free Visual Editing • 13

a transparent golden veil covered on a plate of apples

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

a cute yellow dog is playing with a purple ball, cartoon style a cute red dog is playing with a grey ball, cartoon style a cute yellow dog, a purple ball, 42

veil, 41

###### Source

Source

a bear with brown and orange texture is in the jungle, best quality a bear with black and white texture is in the jungle, best quality a bear, 52

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

a man in green T-shirt is standing next to a single white car, best quality" a man in blue T-shirt is standing next to a single red car, best quality a man in green T-shirt, a single white car, 4321

Rec.

Target

cartoon style of two hands holding a red pill and a blue pill cartoon style of two hands holding a green pill and a yellow pill pill, 123

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

“green T-shirt … white car” ↓ “blue T-shirt … red car”

“yellow dog … purple ball” ↓ “red dog … grey ball”

“blue and white stripes” ↓ “green and yellow stripes”

“red sweater … black pants” ↓ “yellow sweater … blue pants”

“red and blue pill” ↓ “green and yellow pill”

a boy is wearing a red sweater and black pants a boy is wearing a yellow sweater and blue pants A red sweater, black pants, 123

##### Fig. 17. Examples of multi-region editing.

Target

More multi region

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

“blue paints” ↓ “green paints”

“red golf ball” ↓ “white golf ball”

“yellow rose” ↓ “green rose”

“light blue” ↓ “dark blue”

“blue spoon” ↓ “green spoon”

close look up of a man in a shirt with alternating of green and yellow stripes, cartoon style

a shirt with alternating of green and yellow stripes, 1000

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

- Fig. 19. Examples of real input image editing. The first row shows the source images, the second row presents the reconstructed images via inversion, and the third row displays the editing results based on the target prompts.

Method Preference (%)

RF-Solver [Wang et al. 2025] 0.74 SDEdit [Meng et al. 2022] 5.19 FireFlow [Deng et al. 2025] 5.93 UniEdit-Flow [Jiao et al. 2025] 6.67 DiTCtrl [Cai et al. 2025] 10.37 Ours 71.11

Table 4. User study preferences over different methods.

- B.2 User Study

We conducted a user study involving 18 participants to evaluate editing quality across different methods. Each participant was presented with 30 randomly selected pairs of structure-consistent and structure-inconsistent edits, and was asked to choose the preferred result in each pair. As summarized in Tab. 4, Ours achieved a dominant preference rate of 71.11%, substantially outperforming all competing approaches.

- B.3 Consistency Strength

碾压，solve 最佳参数的性能类似

调参简单strength，过渡丝滑 ⾮edit部分，可以完全⼀致

qk对应ﬁne-grained geometry

prompt产⽣贴图

v对应coarse geometry和贴图

a close-up shot of an asian man wearing white T-Shirt in a park, best quality seed 42 a close-up shot of an African man wearing white T-Shirt in a park, best quality seed 42

Ours

FireFlow

DiTCtrl

UniEdit-Flow

SDEdit

RF-Solver

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

Ours

FireFlow

DiTCtrl

UniEdit-Flow

SDEdit

RF-Solver

Source

“asian man” ↓ “african man”

consistency strength

[Figure 745]

Ours

Ours

Ours*

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

Source

“asian man” ↓ “african man”

consistency strength

- Fig. 20. Qualitative comparison of different consistency strength settings. “Ours” denotes the method proposed in the main paper, while “Ours*” refers to a modified version of our method.

[Figure 761]

Source

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

Source

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

a red golf ball

a pair of light blue shoes

a blue spoon and a mug

a man wearing a light blue shirt and dark blue pants, best quality

a yellow rose

Ours

More real image multi-round

a green spoon and a mug, 42, 8

“brown and orange texture” ↓ “black and white texture”

a white golf ball, 42, 4

a green rose, 42, 8

a pair of dark blue shoes, 42, 14

a man wearing a light blue shirt and dark green pants, best quality, 42, 4

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

A detailed green toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.

Ours

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

A detailed dark red toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.

DiTCtrl

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

A young boy, dressed in a white shirt and blue shorts, strolls leisurely through a lush, green park. The sunlight ﬁlters through the dense canopy of oak trees, casting dappled shadows on his path. He carries a small, red backpack slung over one shoulder, and his brown hair is tousled by a gentle breeze. As he walks, he pauses to observe a playful squirrel darting up a tree, a look of curiosity lighting up his face. The park is alive with the sounds of birds chirping and children's laughter in the distance, creating a serene and vibrant atmosphere around him. Seed 0

42

A young boy, dressed in a white shirt and green shorts, strolls leisurely through a lush, green park. The sunlight ﬁlters through the dense canopy of oak trees, casting dappled shadows on his path. He carries a small, red backpack slung over one shoulder, and his brown hair is tousled by a gentle breeze. As he walks, he pauses to observe a playful squirrel darting up a tree, a look of curiosity lighting up his face. The park is alive with the sounds of birds chirping and children's laughter in the distance, creating a serene and vibrant atmosphere around him.

DiTCtrl

[Figure 782]

UniEdit-Flow

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

UniEdit-Flow

FireFlow

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

RF-Solver

FireFlow

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

SDEdit

RF-Solver

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

More video

SDEdit

Fig. 18. Additional qualitative comparison of methods on video editing tasks. The edit prompt is “blue shorts” → “green shorts”.

1 17 33 49

performs the corresponding edits. These results highlight the precise text-driven control of our method, fine-grained understanding of visual structure, and strong structure preservation capabilities.

Additional real-image editing examples are shown in Fig. 19. Our method preserves the structural integrity within the edited regions while maintaining the original content in the non-edited regions, achieving performance on par with editing generated images.

The main text demonstrates that our method offers fine-grained control over structural alignment with the source image through the consistency strength, while preserving the ability to edit texture

14 • Zixin Yin, et al.

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

“a horse” ↓ “a bronze horse”

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

“a house” ↓ “a stone house”

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

“green lizard” ↓ “brown lizard”

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

“white christmas tree” ↓ “green christmas tree”

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

“…” ↓ “kids crayon

drawing of…”

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

“bee…” ↓ “…”

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

“mountain bicycle” ↓ “rusty mountain bicycle”

Source Ours DiTCtrl UniEdit-Flow FireFlow RF-Solver SDEdit

Fig. 21. Additional qualitative comparison of methods on structure-consistent and structure-inconsistent editing tasks.

according to the prompt. However, in certain downstream applications, users may prefer a binary behavior in which a consistency strength of 1 results in an output identical to the source image, and a strength of 0 produces results fully aligned with the edited prompt. Although such scenarios are beyond the primary focus of this work, we provide a simple mechanism to enable this behavior, which may serve as a basis for future research in this direction.

As shown in Fig. 20, this simple adjustment successfully achieves the desired behavior between unedited and fully edited results.

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

B.4 Limitation

The generation quality and the precision of text-guided localization in our method are ultimately constrained by the capabilities of the base generative models. Two representative failure modes are illustrated in Fig. 22:

Source

To support this behavior, we apply a small modification to our method: within the editing region, in addition to transferring the vision part of Q and K tokens, we also transfer that of V tokens.

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

• Localization Failure: Small or abstract objects may not be edited when the attention map lacks a clear activation, leading to no

Target

ConsistEdit: Highly Consistent and Precise Training-free Visual Editing • 15

###### Source Prompt Result

the model struggles to distinguish between intertwined hair and veil.

[Figure 867]

[Figure 868]

• Semantic Ambiguity: Given a prompt to change lipstick color, the model may instead edit the lipstick case rather than the lipstick itself.

“a black veil” ↓ “a golden veil”

In a different aspect, compared with image models, current video models still lag considerably in generation fidelity. Nevertheless, as foundation models continue to improve, we expect our method to benefit correspondingly and expand its applicability.

[Figure 869]

[Figure 870]

“a red lipstick” ↓ “a green lipstick”

Furthermore, our ability to edit real images and videos is inherently constrained by the limitations of current inversion and reconstruction techniques. Although our method performs reliably on data within the distribution of the generative model, editing realworld inputs requires accurately mapping them into the latent space of the model, a task that remains challenging and highly dependent on the quality of the inversion process.

Fig. 22. Examples of typical failure cases.

visible change. For example, in the top case of Fig. 22, although the overall color, including some very small holes, is edited correctly,

