# arXiv:2601.14250v1[cs.CV]20Jan2026

[Figure 1]

[Figure 2]

## OmniTransfer: All-in-one Framework for Spatio-temporal Video Transfer

##### Pengze Zhang Yanze Wu† Mengtian Li Xu Bai Songtao Zhao‡ Fulong Ye Chong Mou Xinghui Li Zhuowei Chen Qian He Mingyuan Gao

Intelligent Creation Lab, ByteDance

##### Abstract

Videos convey richer information than images or text, capturing both spatial and temporal dynamics. However, most existing video customization methods rely on reference images or task-specific temporal priors, failing to fully exploit the rich spatio-temporal information inherent in videos, thereby limiting flexibility and generalization in video generation. To address these limitations, we propose OmniTransfer, a unified framework for spatio-temporal video transfer. It leverages multi-view information across frames to enhance appearance consistency and exploits temporal cues to enable fine-grained temporal control. To unify various video transfer tasks, OmniTransfer incorporates three key designs: Task-aware Positional Bias that adaptively leverages reference video information to improve temporal alignment or appearance consistency; Reference-decoupled Causal Learning separating reference and target branches to enable precise reference transfer while improving efficiency; and Task-adaptive Multimodal Alignment using multimodal semantic guidance to dynamically distinguish and tackle different tasks. Extensive experiments show that OmniTransfer outperforms existing methods in appearance (ID and style) and temporal transfer (camera movement and video effects), while matching pose-guided methods in motion transfer without using pose, establishing a new paradigm for flexible, high-fidelity video generation.

Date: January 20, 2026 Project Page: https://pangzecheung.github.io/OmniTransfer/

##### 1 Introduction

We have all heard the old adage: “A picture is worth a thousand words”—and if we follow that logic a little further, a video might be worth a million. After all, a static image freezes just one moment of light, texture, and form; video weaves those moments into something dynamic, carrying not just how things look, but how they move and change. These are details that neither words nor a single image can fully convey.

The same insight applies to diffusion-based video generation. A reference video offers greater flexibility and capability than using an image or text alone. Not only can the model reference identity (ID) and style in spatial appearance aspects, but it can also exploit temporal information, including camera movement, motion, and visual effects, enabling more coherent and expressive video synthesis.

† Corresponding author, ‡ Project lead. This work is purely academic and non-commercial. Demo reference images/videos are from public domains or AI-generated. For copyright concerns, please contact us for the removal of relevant content.

Motion

ID

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Image

RefV

OursRefV

[Figure 11]

(Motion)

“Playing the piano”

(Style)(ID)

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Ours

Effect

Style

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Image

OursRefV

RefV

(Effect)

[Figure 28]

“reading book under the tree”

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Ours

Temporal Video Transfer

Spatial Appearance Video Transfer

ID

ID

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

RefV1OursRefV2

RefV1RefV2

（ID+Effect)

Effect

(ID+Style)

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Style

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

(ID+Effect)

“playing guitar”

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Ours

Video Transfer Combination

- Figure 1 OmniTransfer seamlessly unifies spatial appearance (ID and style) and temporal video transfer tasks (effect, motion and camera movement) within a single framework, and exhibits strong generalization across unseen task combinations.

Despite this potential, existing methods have yet to fully exploit their spatio-temporal information. 1) For spatial information, most approaches [38, 67, 71, 72] still mainly rely on pixel-level cues extracted from reference images. However, cues such as ID and style are inherently spatio-temporal, extending across multiple frames and views. This makes it difficult for a single image to capture their full details, thereby limiting their effectiveness. 2) For temporal information, current approaches remain in its early stages: some depend on pre-defined physical priors (e.g., pose or camera position) [9, 19, 75, 78], some employ inversion-based reconstruction [22, 36], while others require additional fine-tuning for specific temporal dynamics during test time [47, 77]. Recently, CamCloneMaster [40] made an initial reference-based attempt on camera motion through temporal context concatenation, yet it struggles to generalize to in-the-wild conditions, and fails to address general temporal video reference tasks. These limitations highlight the need for a unified framework that directly leverages temporal information from reference videos, enabling flexible and robust video customization in real-world scenarios.

To tackle the aforementioned challenges, this paper introduces a novel exploration in the domain of video reference, proposing an all-in-one framework for spatio-temporal video transfer, OmniTransfer. This framework not only integrates multi-frame reference information from the reference video at the spatial level, enhancing the consistency of reference video ID and style, but also effectively models temporal aspects such as motion, camera movement, and video effects, providing multi-dimensional control and unification over the video generation process. First, to unify video transfer tasks, we introduce Task-aware Positional Bias. For temporal transfer tasks, based on our assumption that video diffusion models inherently maintain temporal consistency through spatial context, we add spatial offsets to positional embeddings to preserve temporal alignment. For appearance transfer tasks, temporal offsets are applied to propagate appearance information across frames. Second, we introduce Reference-decoupled Causal Learning, which employs unidirectional transfer from reference to target, preventing simple copy-pasting. By separating the reference and target branches, the reference branch requires only a single forward pass, reducing computational time by 20% compared to full-attention models. Finally, to improve semantic guidance and avoid cross-task confusion, we introduce a Multimodal Large Language Model (MLLM) via a Task-adaptive Multimodal Alignment module. This module leverages multiple sets of task-specific MetaQueries [46] to dynamically integrate and align semantic information across tasks, effectively meeting the requirements of each task.

In sum, our contribution can be summarized as follows.

- • We propose OmniTransfer, a unified framework for the new task of spatio-temporal video transfer that supports diverse tasks such as identity, style, effect, camera movement, and motion transfer, while generalizing seamlessly to their compositional combinations (Fig. 1).
- • We propose Task-aware Positional Bias, Reference-decoupled Causal Learning, and Task-adaptive Multimodal Alignment, each designed to unify various video customization tasks, enabling efficient and flexible spatio-temporal information transfer.
- • Experiments show that OmniTransfer outperforms existing methods in appearance (ID, style) and temporal (camera movement, effects) transfer, while matching pose-guided methods in motion transfer without using pose. Moreover, benefiting from our model design, it achieves these improvements with a 20% reduction in runtime compared to the basemodel architecture.

##### 2 Related Work

###### 2.1 Appearance reference task

The two main tasks in appearance reference are ID and style transfer. For ID transfer in images, approaches have evolved from adapter-based tuning [15, 29, 31, 57, 69, 70] to in-context learning [5, 16, 34, 45, 63]. In video ID transfer, most approaches still rely on single reference images for ID preservation [14, 18, 67, 72], while some works further explore broader subject-to-video generation tasks [7, 8, 23, 26, 35, 38, 44]. For instance, Phantom [38] concatenates ID features along the temporal dimension to maintain appearance consistency. Similarly, style transfer has been widely explored in image diffusion models [33, 54, 55, 64]. Early video stylization extends these models to the temporal domain via Image-to-Video paradigms [4, 12, 30], while recent Text-to-Video methods [6, 37, 59, 71] offer more flexible and controllable stylization. Despite these advances, these methods rely solely on single-image references, overlooking the rich multi-frame and multi-view cues inherent in videos. Our approach exploits these cues to achieve consistent appearance and stable temporal coherence.

###### 2.2 Temporal reference task

Temporal video customization covers motion generation, camera movement, and effect synthesis. Pose motion transfer initially relies on GAN-based warping [49, 50, 73, 76], while diffusion-based models [3, 9, 21, 41, 52, 60, 66, 75, 79, 80] improve temporal smoothness. However, they are limited by pose priors, as skeleton alignment may cause loss of appearance cues and hinder their extension to multi-person scenarios. Some studies explore more general motion control [28, 42, 47, 56, 68, 74, 77], but most rely on diffusion inversion or test-time finetuning. Camera motion generation typically uses explicit parameterization [17, 19, 32, 58, 62, 65, 78] and parameter-free attention inversion [22, 36], while CamCloneMaster [40] explores temporal context concatenation, yet resolution and generalization remain limited. Effect generation in industry often relies on LoRA-based [20] finetuning, whereas academic works [13, 39, 43] focus on precise spatial and temporal control of effects, but cannot generate new effects conditioned on a reference video. In contrast, our OmniTransfer achieves unified motion, camera, and effect transfer without explicit priors, demonstrating strong generalization to in-the-wild scenarios.

##### 3 Preliminary

Our framework is built upon Wan2.1 I2V 14B [53] as the underlying diffusion model. The input latent lt ∈ Rf×h×w×(2n+4) = [c,m,zt] concatenates three components along the channel dimension: latent noise zt ∈ Rf×h×w×n obtained by adding timestep t noise to VAE-compressed video features z; condition latent c ∈ Rf×h×w×n encoded from the condition image I concatenated with zero-filled frames; binary mask latent m ∈ Rf×h×w×4 with values of 1 for preserved and 0 for generated frames. [·,·] denotes feature concatenation along the channel dimension, and f, h, w, n represent frame number, height, width, and channel dimension, respectively.

###### Reference Latent Construction

Task-adaptive Multimodal Alignment

Reference Video

Task Template Input

Prompt (Optional)

MLLM Vision Encoder

MLLM Text Encoder

[Figure 61]

First Frame (Optional)

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Target Video

Task-specific Meta Queries

First Frame Tokens (Optional)

Reference Video Tokens

Task Template Tokens

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

###### ...

###### ...

... ...

[Figure 77]

[Figure 78]

[Figure 79]

MLLM

LoRA

VAE Encoder

...

Reference-Decoupled Causal Learning

time = t

0 0 0

ref

tgt

[Figure 80]

+

Noise

ref tgt

TaskFlag TaskFlag TaskFlag TaskFlag 1 0 0 0

Connector

ref

tgt

Reference Tokens Target Tokens

[Figure 81]

[Figure 82]

0

tgt

ref

DiT Block DiT Block

Frame

Frame

QKV Project

QKV Project

Channel-wise Concatation

ref tgt

Targer Video

TPB RoPE

[Figure 83]

[Figure 84]

Task-aware Positional Bias (TPB)

[Figure 85]

[Figure 86]

re f tgt ref tgt

re f

ref

ref

tgt

Width

Frame

Width

Frame

time=0

time=t

Self Attention Self Attention Cross Attention Cross Attention

Target Tokens

Target Tokens

Reference Tokens

Height

MSE Loss

[Figure 87]

[Figure 88]

Reference Tokens

Height

DiT Block

DiT Block

...

...

...

...

[Figure 89]

[Figure 90]

RoPE for Appearance Reference Task

RoPE for Temproal Reference Task

Target Output

DiT Block

DiT Block

- Figure 2 OmniTransfer comprises three key components: 1) Task-aware Positional Bias: exploits the model’s inherent spatial and temporal context capabilities for diverse tasks. 2) Reference-decoupled Causal Learning: separates reference and target branches for causal and efficient transfer. 3) Task-adaptive Multimodal Alignment: leverages an MLLM to unify and enhance semantic understanding across tasks.

Each Diffusion Transformer (DiT) block in Wan2.1 includes self-attention and cross-attention layers. The self-attention adopts 3D Rotary Positional Embedding (RoPE):

Attn(Rθ(Q),Rθ(K),V ) = softmax

Rθ(Q)Rθ(K)⊤ √

d

V, (1)

where Q = WQlt, K = WKlt, V = WV lt. Rθ(·) denotes the RoPE rotation applied to queries and keys, WQ, WK, WV are learnable projections, and d is the feature dimension. Cross-attention integrates textual features as Attn(Q,Kp,Vp) with Kp and Vp derived from prompt p.

##### 4 Method

In this work, we study both appearance and temporal video reference, across several representative tasks summarized in Table 1. To better unify these tasks, we propose OmniTransfer, as illustrated in Fig. 2. It consists of four components- Reference Latent Construction, Task-aware Positional Bias, Reference-decoupled Causal Learning and Task-adaptive Multimodal Alignment—described in detail below.

###### 4.1 Reference Latent Construction

To handle practical scenarios where reference and target videos may have different resolutions, we construct separate latents for the two inputs (Fig. 2). For the target video, the latent is constructed in the same manner as in Section 3, i.e., ltgt ∈ Rf×h

tgt×wtgt×(2n+4) = [ctgt,mtgt,ztgtt ]. For the reference video, the latent representation is defined as: lref ∈ Rf×h

ref×wref×(2n+4) = [cref,mref,zref0 ]. The conditional latent cref corresponds to the feature encoded by the VAE. The mask latent mref is assigned task-specific flags: −1 for temporal tasks, −2 for ID transfer, and −3 for style transfer. Notably, the reference latent zref0 is kept noise-free to maximally preserve its information.

Task Input Output-Vtgt

Appearance Transfer (T2V)

ID Transfer Vref,p ID from Vref, following prompt p Style Transfer Vref,p Style from Vref, following prompt p

Temporal Transfer (I2V)

Motion Transfer Vref,I Motion from Vref, starting from I Camera Movement Vref,I Camera movement from Vref, starting from I Effect Transfer Vref,I Effect from Vref, starting from I

Table 1 Overview of video reference tasks. Vref: reference video; I: first-frame image; p: prompt; Vtgt: generated video.

###### Spatial Incontext

[Figure 91]

“Generate a two-column video. The left panel shows a boy performing a dance. The right panel should show a girl performing the same dance, synchronized in time and rhythm.”

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Seedance1.0Wan2.1T2V

Pose Consistent

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

= 0 =  ’ =  //2 =  //2 +  '

Temproal Incontext

[Figure 100]

“Generate a two-shot video. [Scene 1] shows a boy performing a dance. [Scene 2] should show a girl performing the same dance, synchronized in time and rhythm.”

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

###### Wan 2.1 fails to generate multi-shot video

Seedance1.0Wan2.1T2V

Pose Inconsistent

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Shot1 Shot1

Shot2 Shot2

= 0 =  ’ =  //2 =  //2 +  '

- Figure 3 Video diffusion models are inherently capable of handling temporal consistency through spatial context.

###### 4.2 Task-aware Positional Bias

IC-Lora [24] proposes the assumption that “text-to-image models inherently possess in-context generation capabilities”. This hypothesis has been well validated in image customization methods [5, 45, 63], and is also supported in appearance-consistent video customization [38, 67, 72], where reference appearances are leveraged through temporal in-context learning. However, it remains unclear whether current video diffusion models exhibit comparable in-context capabilities for temporal consistency tasks. To investigate this, we directly evaluate the video consistency of two representative models, Wan 2.1 [53] and Seedance [11], under the T2V setting, as shown in Fig. 3. We observe that when generating side-by-side videos, both models easily maintain motion consistency across columns. In contrast, when generating two temporally consecutive shots, the model fails to keep actions consistent across two shots. This observation motivates us to propose a new assumption for video models: video diffusion models are inherently capable of handling temporal consistency

through spatial context.

Based on the proposed assumption, we introduce Task-aware Positional Bias. Specifically, for temporal reference tasks, we add an offset to the RoPE of the reference video along the spatial (width) dimension, aiming to leverage spatial in-context cues to enhance temporal consistency. The offset is set to the width of the target video, wtgt. In contrast, for appearance reference tasks, to exploit the temporal propagation of appearance information in the video model, we add an offset along the temporal dimension, equal to the number of frames in the target video, f. In summary, we define the RoPE rotation of the reference latents as Rθ∗:

Rθ(·,∆=(0,wtgt,0)),for temporal ref. Rθ(·,∆=(f,0,0)), for appearance. ref.,

Rθ∗(·)=

(2)

where ∆ = (∆T,∆W,∆H) represents the offsets applied along the temporal, width, and height dimensions.

###### 4.3 Reference-decoupled Causal Learning

A straightforward way to enable interaction between reference and target videos is through joint self-attention. However, our experiments reveal that this bidirectional attention mechanism may lead to two main issues in video transfer tasks: 1) The generated videos often exhibit a simple “copy-paste” effect from the reference video. We attribute this to the reference branch’s full access to the target video context, which encourages it to adopt a target-like representation, resulting in direct copying of simple patterns. 2) Concatenating the reference and target videos for joint self-attention doubles the number of tokens, leading to a fourfold increase in computational complexity, which is often impractical.

To address the aforementioned issues, we propose fully decoupling the reference and target branches, as illustrated in Fig. 2. Formally, both reference and target tokens are first projected into the queries, keys, and values, yielding Qref, Kref, Vref, Qtgt, Ktgt and Vtgt. Next, task-aware positional bias is applied to Qref and Kref, while Ktgt and Vtgt use the standard RoPE positional encoding. Subsequently, attention interacts between the two branches in a causal manner. The reference branch performs intra-branch self-attention to capture internal contextual dependencies:

Attnref = Attn(Rθ∗(Qref),Rθ∗(Kref),Vref), (3)

while the target branch integrates information from both its own features and the reference features by concatenating the keys and values:

Attntgt=Attn(Rθ(Qtgt),[Rθ(Ktgt);Rθ∗(Kref)],[Vtgt;Vref]), (4) where [·;·] denotes token-wise concatenation.

We further decouple the time embeddings of the two branches. Specifically, the reference branch adopts a fixed t = 0, making it independent of the target video’s noise level. Thanks to this design, the reference branch becomes time-invariant during inference, effectively reducing computational overhead and shortening the overall generation time by 20% compared to the standard architecture.

###### 4.4 Task-adaptive Multimodal Alignment

In multi-task video transfer, different tasks demand reference information with distinct semantic focuses. However, conventional in-context learning in diffusion models primarily captures shallow visual correspondences rather than semantic intent, limiting their adaptability across tasks. To overcome this, we replace the original T5 [48] features in Wan with representations from MLLM, i.e., Qwen-2.5-VL [2], introducing richer visualsemantic cues that enhance reference understanding and task-level alignment.

The MLLM takes as input the first-frame tokens of the target video, the reference video tokens, template tokens, and prompt tokens. To extract task-specific representations, we draw inspiration from MetaQuery [46] and introduce a set of learnable tokens dedicated to each task. For temporal tasks, the MetaQuery aggregates temporal cues from the reference video together with the target’s first-frame content, effectively capturing

“a blonde-haired woman in a black top, gently touches and bends down to

“a person leaning over a large sketchpad, drawing slow, deliberate lines. The sound of pencil scratching blends with faint ambient noise from an open window”

Prompt smell the flowers—picturesque background of more greenery and white flowers”

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Ref I/V

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

ConsisID (Ref I)

IDTransferStyleTransfer

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Phantom (Ref I)

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Stand-in (Ref I)

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Ours (Ref V)

Prompt

“The video depicts a person folding a large piece of fabric on a table, smoothing out wrinkles carefully with their hands.”

“The video shows a person watering a row of plants on a balcony, tilting the watering can slowly over each pot.”

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Ref I/V

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

StyleMaster (Ref I)

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Ours (Ref V)

- Figure 4 Qualitative comparison for appearance video transfer. The yellow box highlights how OmniTransfer captures richer appearance details from multiple cross-view video frames. The red box denotes the input image for image-reference methods.

cross-frame dynamics in the generated sequence. For appearance tasks, it instead fuses identity or style information from the reference with the semantic context derived from the prompt tokens.

To preserve the multimodal reasoning capability while enabling parameter-efficient adaptation, the MLLM is fine-tuned using LoRA [20]. For seamless integration with the diffusion model, its outputs are passed through a three-layer MultiLayer Perceptron (MLP) and injected solely into the target branch, thereby preventing interference with the reference branch. Consequently, the cross-attention in the target branch is defined as Attn(Qtgt,KMLLM,VMLLM), where KMLLM and VMLLM denote the keys and values derived from the aligned MLLM features.

- 5 Experiment

###### 5.1 Implementation Details

Our training process is divided into three sequential stages with distinct optimization objectives. In the first stage, we train the DiT blocks via in-context learning. Subsequently, we freeze the DiT blocks and focus on training the connector to align the MLLM with diffusion models. Finally, we unfreeze all components to conduct joint fine-tuning. In terms of training hyperparameters, we set the learning rate to 1e − 5, with a batch size of 16. The three stages are trained for 10,000, 2,000, and 5,000 training steps, respectively.

Due to the lack of public data sets containing reference video pairs at present, we collected our own data sets from the Internet to support spatio-temporal video transfer.

###### 5.2 Evaluation Details

To evaluate spatio-temporal video transfer, we curated dedicated test sets for each subtask. For ID transfer, 50 videos of diverse individuals are paired with two prompts each to test identity consistency. Style transfer includes 20 unseen visual styles, each with two prompts to assess stylistic variation. Effect transfer contains

First Frame

First Frame

Reference Video

Reference Vdieo

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

EffectTransfer

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Wan2.1 I2V

Wan2.1 I2V

[Figure 201]

[Figure 202]

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

Seedance I2V

Seedance I2V

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

[Figure 223]

[Figure 224]

Ours

Ours

###### CameraMovementTransfer

First Frame First Frame

Reference Vdieo Reference Vdieo

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

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

MotionClone

MotionClone

CamClone Master

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

CamClone Master

[Figure 263]

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

Ours

Ours

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

First Frame Reference Vdieo

First Frame Reference Vdieo

[Figure 281]

[Figure 282]

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

MotionTransfer

Mimic Motion

Mimic Motion

[Figure 301]

[Figure 302]

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

Wan Animate (Wan 2.2)

Wan Animate (Wan 2.2)

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Ours Ours

Figure 5 Qualitative comparison for appearance video transfer.

50 unseen effects from visual-effects websites. Camera movement transfer uses 50 professionally shot videos with complex trajectories. Motion transfer comprises 50 popular dance videos covering diverse dynamic and fine-grained body motions.

###### 5.3 Comparison

###### 5.3.1 ID Transfer

We compared our method with State-Of-The-Art (SOTA) video ID preservation approaches, including ConsisID [72], Phantom [38] and Stand-in [67]. Following Phantom, we measure text–video alignment using the CLIP-T [61] score, and assess identity consistency with several face recognition models, including

Method VSim-Arc ↑ VSim-Cur ↑ VSim-Glint ↑ CLIP-T ↑

ConsisID [72] 0.34 0.32 0.36 21.54 Phantom [38] 0.45 0.41 0.47 20.34 Stand-in [67] 0.30 0.21 0.26 20.38

Ours 0.48 0.43 0.51 20.35

Table 2 Quantitative comparison for ID Transfer.

Method VCSD ↑ CLIP-T ↑ Aesthetics ↑

StyleCrafter [37] 0.44 24.72 0.47 StyleMaster [71] 0.29 26.82 0.59

Ours 0.51 27.16 0.61

Table 3 Quantitative comparison for Style Transfer.

Sim-Arc [10], Sim-Cur [25], and Sim-Glint [1]. We further extend these metrics to video-level similarity by matching four evenly sampled reference frames with all generated frames, denoted as VSim-Arc, VSim-Cur, and VSim-Glint.

Leveraging multi-view facial information, our method achieves high facial similarity with natural, fluid motions. As shown in the yellow box of Fig. 4, it preserves fine details like acne across frames, which is difficult for image reference methods. In Fig. 4 (right), our method generates diverse poses—including frontal, profile, and tilted views—while maintaining high facial similarity, highlighting the advantage of using video as the reference.

###### 5.3.2 Style Transfer

We compared our method with SOTA text-to-video stylization approaches, StyleCrafter [37] and StyleMaster [71]. Following StyleMaster, we evaluated text–video alignment with CLIP-T, while aesthetics with the Aesthetics Score [27], and style consistency with the video CSD Score [51] (VCSD) using four sampled reference frames.

Table 3 shows that our method outperforms others on all three metrics. For qualitative comparison in

- Fig. 4, we only include StyleMaster, as StyleCrafter yields lower visual quality due to its earlier UNet-based design. The results demonstrate that our method effectively captures style from the video, surpassing previous image-based methods.

###### 5.3.3 Effect Transfer

Since our test set comes from effect websites with effects that existing commercial LoRA models cannot reproduce, we instead compare our method with SOTA image-to-video models, Wan 2.1 I2V [53] and Seedance [11], using prompts generated by Qwen-2.5-VL [2] from the reference effect videos. With no standard metrics for effect consistency, we conduct a user study with 20 volunteers, who rate effect fidelity, first-frame consistency, and overall visual quality on a five-point scale.

- As shown in Table 4, our method achieves the highest scores on all three metrics. Qualitative results in Fig. 4 further show that only our method accurately reproduces the effects of the reference videos, outperforming Seedance and Wan I2V. This demonstrates that text alone is insufficient, emphasizing the value of temporal video references.

###### 5.3.4 Camera Movement Transfer

We compare our method with SOTA camera movement models, MotionClone [36] and CamCloneMaster [40]. Since the estimation of complex camera trajectory remains a challenging problem, we conducted a user study using the same setup as in Section 5.3.3, evaluating camera fidelity, image consistency, and overall quality.

Method Effect Fidelity ↑ Image Consistency ↑ Quality ↑ Wan2.1 I2V [53] 1.81 2.89 2.03

Seedance I2V [11] 1.95 3.20 2.42

Ours 3.45 3.49 3.27

Table 4 User study on Effect Transfer.

Method Camera Fidelity ↑ Image Consistency ↑ Quality ↑ MotionClone [36] 1.75 1.23 1.29

CamCloneMaster [40] 1.79 1.45 1.29

Ours 4.19 3.89 3.85

Table 5 User study on Camera Movement Transfer.

RefV Baseline +TPB +RCL +TMA RefV Baseline + TPB +RCL +TMA

RefV Baseline + TPB

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

Image

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

[Figure 352]

[Figure 353]

Image

Image

[Figure 354]

[Figure 355]

[Figure 356]

(a) Effect Transfer

(b) Effect Transfer

RefV Baseline + TPB +RCL +TMA

RefV Baseline + TPB +RCL +TMA

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

“a man with a rugged beard, wearing a leather jacket, riding a vintage motorcycle along a desert highway.”“The video shows a person arranging books on a shelf, stepping back occasionally to adjust their alignment.”

(c) ID Transfer (d) Style Transfer (e) Pose Transfer

Figure 6 Qualitative ablation study comparison. Best viewed in zoom.

- As shown in Table 5, our method outperforms all others across the three metrics. The qualitative results in

- Fig. 5 show that competing methods produce only fixed-resolution outputs, resulting in unavoidable resizing or cropping. Moreover, our approach is the only one that successfully replicates camera movements from cinematic scenes (left) and complex professional tracking shots (right), demonstrating strong generalization capability.

###### 5.3.5 Motion Transfer

We compare our method with state-of-the-art motion transfer approaches, MimicMotion [75] and WanAnimate [9]. As the test set contains in-the-wild videos without paired ground truth, we conducted a user study following Section 5.3.3, evaluating motion fidelity, image consistency, and overall quality. As shown in Table 5, despite using the smaller 14B Wan 2.1 model without additional pose input, our method achieves the highest image consistency, with motion consistency and quality comparable to WanAnimate, which relies on the larger 28B Wan 2.2 model.

Qualitative comparisons in Fig. 6 show that baseline methods need to align the first frame to a target pose, often causing appearance loss (bottom-right of Fig. 6) and unnatural motion. In contrast, our method requires no pose guidance, preserves appearance, produces natural motions, and easily extends to multi-person scenarios (Fig. 1).

Method Motion Fidelity ↑ Image Consistency ↑ Quality ↑ MimicMotion [75] 2.67 1.84 2.02

WanAnimate [9] 3.71 3.53 3.48 Ours 3.62 3.88 3.45

Table 6 User study on Motion Transfer.

|Method<br><br>|Components|Appearance Temporal|Time<br><br>|
|---|---|---|---|
| |TPB RCL TMA<br><br>|Consistency/Quality Consistency/Quality| |

Baseline 2.36 / 2.53 2.69 / 2.70 180s +TPB ✓ 2.82 / 2.86 2.95 / 2.94 180s +RCL ✓ ✓ 3.10 / 3.16 3.13 / 3.10 142s

+TMA (Full) ✓ ✓ ✓ 3.27 / 3.56 3.36 / 3.51 145s

Table 7 Ablation study. “Time” indicates inference time per sample (480p, 81 frames, 8×NVIDIA A100).

###### 5.4 Ablation Study

Ablation experiments are conducted incrementally from a Baseline by successively adding TPB, RCL and TMA:

Baseline. We use a vanilla in-context learning setup, where reference features are concatenated along the temporal dimension [38, 40], and full attention is applied.

+TPB. This model incorporates the Task-aware Positional Bias (TPB) defined in Eq. (2).

+RCL. Full attention is replaced by Reference-decoupled Causal Learning (RCL).

+TMA (Full Model). The Task-adaptive Multimodal Alignment (TMA) is added to form the complete model. The ablation study is conducted on 20 cases for each of the appearance and temporal transfer tasks, with a user study rating overall quality and reference consistency on a five-point scale. The results are presented in Table 7, with a qualitative comparison in Fig. 6 for a more comprehensive analysis of each module’s contributions. (1) Baseline: Without TPB, subtle motion cannot be effectively transferred (Fig. 6-e), and task confusion occurs where appearance cues leak into temporal transfer (Fig. 6-a). (2) +TPB: Fine-grained motion transfer is achieved by leveraging spatial context (Fig. 6-e), while task confusion is alleviated by different RoPE biases (Fig. 6-a). (3) +RCL: The copy–paste issue is alleviated. For example, in Fig. 6-a and d, the identity is not fully copied; in Fig. 6-c the face appears more natural, highlighting the effectiveness of causal attention. Additionally, RCL also improves inference speed by 20%. (4) +TMA (Full Model): The TMA module substantially enhances semantic understanding. For instance, in Fig. 6-a, the model understands that the scene remains unchanged; in Fig. 6-b, it correctly understands and generates the money rather than simply copying reference patterns; and in Fig. 6-c, richer semantic control enables realistic details such as a leather jacket, beard, and side-face angle. These results demonstrate the effectiveness of semantic guidance in improving scene controllability.

##### 6 Conclusion

In this work, we introduced OmniTransfer, a unified framework for spatio-temporal video transfer. By integrating Task-aware Positional Bias, Reference-decoupled Causal Learning, and Task-adaptive Multimodal Alignment, our approach effectively leverages multi-view and temporal information from reference videos, enabling fine-grained and consistent video generation across diverse tasks. Extensive experiments verify that OmniTransfer not only achieves superior performance in both appearance and temporal transfer but also establishes a new paradigm for flexible, high-fidelity video generation.

##### 7 Acknowledgments

We would like to express our sincere gratitude to Junjie Luo, Pengqi Tu, Qi Chen, Qichao Sun and Wanquan Feng for their insightful discussions and valuable data contributions.

##### References

- [1] Xiang An, Xuhan Zhu, Yuan Gao, Yang Xiao, Yongle Zhao, Ziyong Feng, Lan Wu, Bin Qin, Ming Zhang, Debing Zhang, and Ying Fu. Partial fc: Training 10 million identities on a single machine. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), 2021.

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [3] Di Chang, Yichun Shi, Quankai Gao, Hongyi Xu, Jessica Fu, Guoxian Song, Qing Yan, Yizhe Zhu, Xiao Yang, and Mohammad Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identity-aware diffusion. In International Conference on Machine Learning (ICML), 2023.

- [4] Hila Chefer, Shiran Zada, Roni Paiss, Ariel Ephrat, Omer Tov, Michael Rubinstein, Lior Wolf, Tali Dekel, Tomer Michaeli, and Inbar Mosseri. Still-moving: Customized video generation without customized video data. ACM Transactions on Graphics (ToG), 43(6), 2024.

- [5] Bowen Chen, Mengyi Zhao, Haomiao Sun, Li Chen, Xu Wang, Kang Du, and Xinglong Wu. Xverse: Consistent multi-subject control of identity and semantic attributes via dit modulation. arXiv preprint arXiv:2506.21416, 2025.

- [6] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.

- [7] Jinshu Chen, Xinghui Li, Xu Bai, Tianxiang Ma, Pengze Zhang, Zhuowei Chen, Gen Li, Lijie Liu, Songtao Zhao, Bingchuan Li, and Qian He. Omniinsert: Mask-free video insertion of any reference via diffusion transformer models. https://arxiv.org/abs/2509.17627, 2025.

- [8] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multi-subject open-set personalization in video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6099–6110, 2025.

- [9] Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, et al. Wan-animate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055, 2025.

- [10] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [11] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [12] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In International Conference on Learning Representations (ICLR), 2024.

- [13] Jiaqi Guo, Lianli Gao, Junchen Zhu, Jiaxin Zhang, Siyang Li, and Jingkuan Song. Magicvfx: Visual effects synthesis in just minutes. In Proceedings of the ACM International Conference on Multimedia (ACMM), page 8238–8246, 2024.

- [14] Xu Guo, Fulong Ye, Xinghui Li, Pengqi Tu, Pengze Zhang, Qichao Sun, Songtao Zhao, Xiangwang Hou, and Qian He. Dreamid-v:bridging the image-to-video gap for high-fidelity face swapping via diffusion transformer. https://arxiv.org/abs/2601.01425, 2026.

- [15] Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. Pulid: Pure and lightning id customization via contrastive alignment. Advances in Neural Information Processing Systems (NeurIPS), 37:36777–36804, 2024.

- [16] Zinan Guo, Pengze Zhang, Yanze Wu, Chong Mou, Songtao Zhao, and Qian He. Musar: Exploring multi-subject customization from single-subject dataset via attention routing. arXiv preprint arXiv:2505.02823, 2025.

- [17] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2025.

- [18] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. ArXiv preprint, 2024.

- [19] Chen Hou and Zhibo Chen. Training-free camera control for video generation. In International Conference on Learning Representations (ICLR), 2025.

- [20] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

- [21] Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8153– 8163, 2024.

- [22] Teng Hu, Jiangning Zhang, Ran Yi, Yating Wang, Hongrui Huang, Jieyu Weng, Yabiao Wang, and Lizhuang Ma. Motionmaster: Training-free camera motion transfer for video generation. arXiv preprint arXiv:2404.15789, 2024.

- [23] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. ArXiv preprint, 2025.

- [24] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arxiv:2410.23775, 2024.

- [25] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: Adaptive curriculum learning loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

- [26] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025.

- [27] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

- [28] Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9212–9221, 2024.

- [29] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. Infiniteyou: Flexible photo recrafting while preserving your identity. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10898–10907, 2025.

- [30] Max Ku, Cong Wei, Weiming Ren, Huan Yang, and Wenhu Chen. Anyv2v: A plug-and-play framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468, 2024.

- [31] Mengtian Li, Jinshu Chen, Wanquan Feng, Bingchuan Li, Fei Dai, Songtao Zhao, and Qian He. Hyperlora: Parameter-efficient adaptive generation for portrait synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13114–13123, 2025.

- [32] Teng Li, Guangcong Zheng, Rui Jiang, Shuigen Zhan, Tao Wu, Yehao Lu, Yining Lin, Chuanyun Deng, Yepan Xiong, Min Chen, Lin Cheng, and Xi Li. Realcam-i2v: Real-world image-to-video generation with interactive complex camera control. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 28785–28796, 2025.

- [33] Wen Li, Muyuan Fang, Cheng Zou, Biao Gong, Ruobing Zheng, Meng Wang, Jingdong Chen, and Ming Yang. Styletokenizer: Defining image style by a single instance for controlling diffusion models. In European Conference on Computer Vision (ECCV), page 110–126, 2024.

- [34] Xinghui Li, Qichao Sun, Pengze Zhang, Fulong Ye, Zhichao Liao, Wanquan Feng, Songtao Zhao, and Qian He. Anydressing: Customizable multi-garment virtual dressing via latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23723–23733, 2025.

- [35] Feng Liang, Haoyu Ma, Zecheng He, Tingbo Hou, Ji Hou, Kunpeng Li, Xiaoliang Dai, Felix Juefei-Xu, Samaneh Azadi, Animesh Sinha, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Movie weaver: Tuning-free multiconcept video personalization with anchored prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13146–13156, 2025.

- [36] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin. Motionclone: Training-free motion cloning for controllable video generation. In International Conference on Learning Representations (ICLR), 2025.

- [37] Gongye Liu, Menghan Xia, Yong Zhang, Haoxin Chen, Jinbo Xing, Yibo Wang, Xintao Wang, Ying Shan, and Yujiu Yang. Stylecrafter: Taming artistic video diffusion with reference-augmented adapter learning. ACM Transactions on Graphics (ToG), 43(6), 2024.

- [38] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14951–14961, 2025.

- [39] Xinyu Liu, Ailing Zeng, Wei Xue, Harry Yang, Wenhan Luo, Qifeng Liu, and Yike Guo. Vfx creator: Animated visual effect generation with controllable diffusion transformer. arXiv preprint arXiv:2502.05979, 2025.

- [40] Yawen Luo, Jianhong Bai, Xiaoyu Shi, Menghan Xia, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Tianfan Xue. Camclonemaster: Enabling reference-based camera control for video generation. arXiv preprint arXiv:2506.03140, 2025.

- [41] Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao Zhang, Tianshu Hu, and Yongming Zhu. Dreamactor-m1: Holistic, expressive and robust human image animation with hybrid guidance. arXiv preprint arXiv:2504.01724, 2025.

- [42] Yue Ma, Yulong Liu, Qiyuan Zhu, Ayden Yang, Kunyu Feng, Xinhua Zhang, Zhifeng Li, Sirui Han, Chenyang Qi, and Qifeng Chen. Follow-your-motion: Video motion transfer via efficient spatial-temporal decoupled finetuning. arXiv preprint arXiv:2506.05207, 2025.

- [43] Fangyuan Mao, Aiming Hao, Jintao Chen, Dongxia Liu, Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Jiahong Wu, and Xiangxiang Chu. Omni-effects: Unified and spatially-controllable visual effects generation. arXiv preprint arXiv:2508.07981, 2025.

- [44] Chong Mou, Qichao Sun, Yanze Wu, Pengze Zhang, Xinghui Li, Fulong Ye, Songtao Zhao, and Qian He. Instructx: Towards unified visual editing with mllm guidance. https://arxiv.org/abs/2510.08485, 2025.

- [45] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915, 2025.

- [46] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

- [47] Alexander Pondaven, Aliaksandr Siarohin, Sergey Tulyakov, Philip Torr, and Fabio Pizzati. Video motion transfer with diffusion transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22911–22921, 2025.

- [48] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research (JMLR), 21(140):1–67, 2020.

- [49] Aliaksandr Siarohin, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

- [50] Aliaksandr Siarohin, Oliver J Woodford, Jian Ren, Menglei Chai, and Sergey Tulyakov. Motion representations for articulated animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13653–13662, 2021.

- [51] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024.

- [52] Shuyuan Tu, Zhen Xing, Xintong Han, Zhi-Qi Cheng, Qi Dai, Chong Luo, and Zuxuan Wu. Stableanimator: Highquality identity-preserving human image animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21096–21106, 2025.

- [53] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [54] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024.

- [55] Haofan Wang, Peng Xing, Renyuan Huang, Hao Ai, Qixun Wang, and Xu Bai. Instantstyle-plus: Style transfer with content-preserving in text-to-image generation. arXiv preprint arXiv:2407.00788, 2024.

- [56] Luozhou Wang, Ziyang Mai, Guibao Shen, Yixun Liang, Xin Tao, Pengfei Wan, Di Zhang, Yijun Li, and Ying-Cong Chen. Motion inversion for video customization. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH), 2025.

- [57] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.

- [58] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. arXiv preprint arXiv:2502.08639, 2025.

- [59] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems (NeurIPS), 36, 2024.

- [60] Xiang Wang, Shiwei Zhang, Longxiang Tang, Yingya Zhang, Changxin Gao, Yuehuan Wang, and Nong Sang. Unianimate-dit: Human image animation with large-scale video diffusion transformer. arXiv preprint arXiv:2504.11289, 2025.

- [61] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.

- [62] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. arXiv preprint arXiv:2312.03641, 2024.

- [63] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13294–13304, 2025.

- [64] Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766, 2024.

- [65] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024.

- [66] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In

- Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1481– 1490, 2024.
- [67] Bowen Xue, Qixin Yan, Wenjing Wang, Hao Liu, and Chen Li. Stand-in: A lightweight and plug-and-play identity control for video generation. arXiv preprint arXiv:2508.07901, 2025.

- [68] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8466–8476, 2024.

- [69] Fulong Ye, Miao Hua, Pengze Zhang, Xinghui Li, Qichao Sun, Songtao Zhao, Qian He, and Xinglong Wu. Dreamid: High-fidelity and fast diffusion-based face swapping via triplet id group learning. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, 2025.

- [70] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

- [71] Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, and Wenhan Luo. Stylemaster: Stylize your video with artistic generation and translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2630–2640, 2025.

- [72] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12978–12988, 2025.

- [73] Pengze Zhang, Lingxiao Yang, Jian-Huang Lai, and Xiaohua Xie. Exploring dual-task correlation for pose guided person image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7713–7722, 2022.

- [74] Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, and Yansong Tang. Flexiact: Towards flexible action control in heterogeneous scenarios. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH), 2025.

- [75] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, JunqiCheng, Yuefeng Zhu, and FangYuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. In International Conference on Machine Learning (ICML), 2025.

- [76] Jian Zhao and Hui Zhang. Thin-plate spline motion model for image animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3657–3666, 2022.

- [77] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jia-Wei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-to-video diffusion models. In European Conference on Computer Vision (ECCV), page 273–290, 2024.

- [78] Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. Cami2v: Camera-controlled image-to-video diffusion model. arXiv preprint arXiv:2410.15957, 2024.

- [79] Jingkai Zhou, Yifan Wu, Shikai Li, Min Wei, Chao Fan, Weihua Chen, Wei Jiang, and Fan Wang. Realisdance-dit: Simple yet strong baseline towards controllable character animation in the wild. arXiv preprint arXiv:2504.14977, 2025.

- [80] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision (ECCV), pages 145–162, 2025.

##### A Additional Comparison Results

We present additional comparisons for the ID transfer task in Fig. 7 and Fig. 8. Fig. 9 and Fig. 10 show further comparisons for the style transfer task. Fig. 11 and Fig. 12 provide more comparisons for effect transfer. Fig. 13 and Fig. 14 present additional results for camera motion transfer, and Fig. 15 and Fig. 16 show further comparisons for motion transfer.

##### B Video Transfer Combination

By concatenating, respectively, the reference video tokens and the MLLM tokens across different tasks, our approach enables seamless combination of multiple video transfer operations. Fig. 17 and Fig. 18 demonstrate the superiority of our method in handling entirely unseen task combinations, highlighting its strong generalization capability.

Sitting on a comfortable beige upholstered sofa in a room with a gray-blue background wall, a figure wearing a green plaid shirt has white round table in front of them, holding a piece of paper in one hand and supporting their head with the other, showing a slightly distressed expression, the tables are adorned with notebooks, pens, and small black objects, the figure makes slight movements throughout the scene, occasionally flipping through the paper and fidgeting with items on the table, with a green potted plant standing quietly beside the sofa.

PhantomStand-inRefVConsisIDOursPromptPhantomStand-inRefVConsisIDOursPrompt

Ref I

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

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

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

[Figure 408]

[Figure 409]

At a desk, a figure sits with an open book in front of them, surrounded by notebooks, colored pencils, and a pencil holder filled with vibrant colors. The figure's hands move gently as they read, shifting the pencils or flipping through pages. In the background, a figure wearing an apron moves casually through the kitchen, completing the warm, homey scene with a quiet energy.

Ref I

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

###### Figure 7 Additional ID transfer comparison with ConsisID [72], Phantom [38], and Stand-in [67] (Set 1).

###### PhantomStand-inRefVConsisIDOursPhantomStand-inPromptRefVConsisIDOursPrompt

A person wearing a light brown leather jacket and dark jeans sat on a park bench, playing guitar and singing a soft folk song. There are two boxes next to me with some coins, and some people stop to listen.

Ref I

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

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

A person in a light blue denim jacket and white pants sits on a beach chair at sunset, reading a book and occasionally looking up to watch the sun dip below the horizon, with the sky turning shades of orange, and the sound of waves crashing on the shore.

Ref I

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

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

###### Figure 8 Additional ID transfer comparison with ConsisID [72], Phantom [38], and Stand-in [67] (Set 2).

A curly - haired person in a white short - sleeved shirt stands in front of a pink background. At first, the eyes are closed and the expression looks a bit sleepy. Then the mouth slowly opens wide as if yawning, and the hand raises to cover the mouth. During the yawn, the body posture remains basically stable.

Master OursRefVPrompt

Ref I

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

RefV Style

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

Crafter Style

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

The picture shows a metallic arched passage. The setting - sun shines through the grids on the side of the passage, creating a warm - yellow halo. Several pedestrians are walking inside the passage, and their figures are slightly blurred due to the light and movement. The ground of the passage is flat, and there is a railing on one side. As the picture progresses, a person wearing red clothes and carrying a backpack enters the passage. The overall atmosphere is tranquil and a bit warm.

Master OursPrompt

Ref I

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

Crafter Style

Style

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

Figure 9 Additional style transfer comparison with StyleCrafter [37] and StyleMaster [71] (Set 1).

A woman sits on the stairs, wearing a white patterned - top and dark pants. The staircase handrail is faintly visible in the background. There is a lit candle behind, giving off a soft light. At first, the woman looks sad with her eyes slightly closed. Then she slowly raises her hand to touch her face and hold her head, seemingly immersed in painful emotions. The overall atmosphere is somewhat depressing.

Master OursRefVPrompt

Ref I

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

RefV Style

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

Crafter Style

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

Indoors, a black Mercedes - Benz sedan is parked on a metal platform. Behind the car are several motorcycles in colors like red and blue. A man wearing a black top and jeans, holding a piece of paper. Then the trunk of the car slowly opens. The man stands still and looks away. The whole scene seems to be a vehicle display or inspection.

Master OursPrompt

Ref I

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

Crafter Style

Style

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

Figure 10 Additional style transfer comparison with StyleCrafter [37] and StyleMaster [71] (Set 2).

Image Ref V

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

Wan I2V

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

Seedance I2V

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

Ours

Image Ref V

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

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

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

Wan I2V

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

Seedance I2V

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

Ours

###### Figure 11 Additional effect transfer comparison with Wan2.1 I2V [53] and Seedance 1.0 I2V [11] (Set 1).

Image Ref V

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

Wan I2V

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

Seedance I2V

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

Ours

Image Ref V

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

Wan I2V

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

Seedance I2V

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

Ours

###### Figure 12 Additional effect transfer comparison with Wan2.1 I2V [53] and Seedance 1.0 I2V [11] (Set 2).

Image Ref V

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

MotioClone

CamClone Master

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

Ours

Image Ref V

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

MotionClone

CamClone Master

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

Ours

###### Figure 13 Additional camera movement transfer comparison with MotionClone [36] and CamCloneMaster [40] (Set 1).

Image Ref V

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

MotioClone CamClone Master

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

Ours

Image Ref V

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

MotioClone

CamClone Master

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

Ours

###### Figure 14 Additional camera movement transfer comparison with MotionClone [36] and CamCloneMaster [40] (Set 2).

Image Ref V

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

Mimic Motion

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

Wan Animate (Wan 2.2)

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

Ours

Image Ref V

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

Mimic Motion

Unavaliable

Wan Animate (Wan 2.2)

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

Ours

- Figure 15 Additional motion transfer comparison with MimicMotion [75] and WanAnimate [9] (Set 1). Unavailable indicates that pose-based methods fail to generate results due to errors in pose extraction or pose alignment.

Image Ref V

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

Mimic Motion

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

Wan Animate (Wan 2.2)

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

Ours

###### Image Ref V

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

Mimic Motion Wan Animate (Wan 2.2)

Unavaliable

Unavaliable

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

Ours

- Figure 16 Additional motion transfer comparison with MimicMotion [75] and WanAnimate [9] (Set 2). Unavailable indicates that pose-based methods fail to generate results due to errors in pose extraction or pose alignment.

Ref V (ID)

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

Ref V (Effect)

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

#### ID+EffectStyle+Camera

Ours

Ref V (ID)

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

[Figure 1097]

Ref V (Effect)

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

Ours

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

Ref V (Style)

#### Style+Motion

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

Ref V (Motion)

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

Ours

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

Ref V (Style)

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

Ref V (Camera)

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

Ours

Figure 17 Video transfer combination results(Set 1).

Prompt a man walking on the beach

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

Ref V (ID)

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

Ref V (Style)

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

Ours

### ID+Style

Prompt A woman picks up a cup from the table and drinks water.

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

Ref V (ID)

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

Ref V (Style)

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

Ours

###### Figure 18 Video transfer combination results (Set 2).

