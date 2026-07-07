### MultiShotMaster: A Controllable Multi-Shot Video Generation Framework

Qinghe Wang1† Xiaoyu Shi2 Baolu Li1 Weikang Bian3 Quande Liu2 Huchuan Lu1 Xintao Wang2 Pengfei Wan2 Kun Gai2 Xu Jia1

1Dalian University of Technology 2Kling Team, Kuaishou Technology 3The Chinese University of Hong Kong

https://qinghew.github.io/MultiShotMaster

## arXiv:2512.03041v1[cs.CV]2Dec2025

###### Text-driven only Global Caption: Subject 1: A woman with brown hair … Subject 2: A fluffy orange tabby cat …The whole scene takes place in a retro-style kitchen with patterned...

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Shot 1: Subject 1 servers dishes in the kitchen. Shot 2: Subject 1 washing dishes in the kitchen. Shot 3: Subject 1 looks focused and smiled. Shot 4: Subject 2 sits on a kitchen floor. Shot 5: Subject 1 smiling and petting Subject 2.

[Figure 10]

[Figure 11]

Backgrounds only

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- Shot 1: Subject 1 and Subject 2 talks. Shot 2: Subject 2 smiles at Subject 1. Shot 3: Subject 1 is speaking animatedly at Subject 2, gesturing with his hands.

- Shot 1: Subject 2 lies on the Subject 3. Shot 2: Subject2 lies on the Subject3, close-up. Shot 3: Subject 1 is walking toward Subject 3. Shot 4: Subject 1 is holding Subject 2 in arms.

[Figure 19]

[Figure 20]

[Figure 21]

Subjects only

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

[Figure 32]

[Figure 33]

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

[Figure 34]

[Figure 35]

[Figure 36]

Backgrounds + Subjects

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

###### Shot 1: Subject 1 and Subject 2 talks. Shot 2: Subject 1 is listening and laughing. Shot 3: Subject 1 and Subject 2 talks. Shot 4: Subject 1 raises a glass in a toast.

Figure 1. We propose MultiShotMaster, the first controllable multi-shot video generation framework that supports text-driven inter-shot consistency, customized subject with motion control, and background-driven customized scene. Both shot counts and shot durations are variable. Only the global caption of the first case is shown for brevity.

##### Abstract

challenges, we propose MultiShotMaster, a framework for highly controllable multi-shot video generation. We extend a pretrained single-shot model by integrating two novel variants of RoPE. First, we introduce Multi-Shot Narrative RoPE, which applies explicit phase shift at shot transitions, enabling flexible shot arrangement while preserving the temporal narrative order. Second, we design Spatiotemporal Position-Aware RoPE to incorporate reference

Current video generation techniques excel at single-shot clips but struggle to produce narrative multi-shot videos, which require flexible shot arrangement, coherent narrative, and controllability beyond text prompts. To tackle these

† Work done during an internship at Kling Team, Kuaishou Tech.

Corresponding authors.

tokens and grounding signals, enabling spatiotemporalgrounded reference injection. In addition, to overcome data scarcity, we establish an automated data annotation pipeline to extract multi-shot videos, captions, cross-shot grounding signals and reference images. Our framework leverages the intrinsic architectural properties to support multi-shot video generation, featuring text-driven intershot consistency, customized subject with motion control, and background-driven customized scene. Both shot count and duration are flexibly configurable. Extensive experiments demonstrate the superior performance and outstanding controllability of our framework.

##### 1. Introduction

Recently, riding the wave of success in video generation [6], content creators have produced many interesting videos and attracted massive traffic. Powered by diffusion transformers (DiTs) [27, 36, 45, 60, 69], the semantic information encapsulated in text prompts can be presented in a singleshot high-quality generated video. Furthermore, advances in controllability [32, 61, 65] have endowed video generation with increased versatility and creative capabilities by incorporating diverse condition signals such as reference image [21, 29] and object motion [13, 49]. However, realworld films and television series consist of multi-shot video clips with narrative structures that convey coherent stories to audiences. Such storytelling relies on cinematic language, encompassing holistic scenes, character interactions, and microexpressions. Therefore, there is a significant gap between current video generation techniques and practical video content creation.

Ideally, the AI-powered video generation system should deliver director-level controllable multi-shot functionality to users, encompassing four critical aspects: (1) variable shot counts and flexible shot durations, (2) dedicated text descriptions for each shot, (3) character appearance and scene definition, (4) character movement control. The development of this vision is still in its infancy. Current multi-shot video generation methods typically fall into two paradigms: text-to-keyframe generation followed by imageto-video (I2V) generation [9, 58, 62, 70], and direct endto-end generation [7, 15, 22, 37, 57]. The keyframe-based paradigm first generates a set of keyframes with visual consistency and then uses the I2V model to generate each shot with the corresponding keyframe. Although it has decent applicability, the limited conditional capability of sparse keyframes cannot cover the briefly-appearing characters and scene consistency outside the keyframes. The end-toend paradigm can generate multi-shot videos directly, and keep the consistency by using full attention along the temporal dimension. Yet, it is still constrained by the fixed shot duration or limited shot counts. Moreover, both paradigms

can only be driven by text prompts. As a result, there is an urgent need to investigate comprehensive controllability in multi-shot video generation, covering flexible shot arrangements and more condition signals.

Under the DiT architecture, all image patches are projected into token embeddings and concatenated together, typically requiring positional encoding (e.g., Rotary Position Embedding (RoPE) [44]) to preserve their spatiotemporal order. The RoPE-based attention mechanism has a crucial property: tokens with closer spatiotemporal distance receive higher attention weights, enabling the model to capture local spatiotemporal correlations. This property inspires our two insights: (1) using continuous RoPE to all frames of multi-shot videos in temporal order will cause the model to confuse intra-shot consecutive frames with intershot frames across shot boundaries. (2) applying the RoPE of a specified region to reference features will bridge the connection with the corresponding video tokens.

In this work, we present MultiShotMaster, the first framework for highly controllable multi-shot video generation. Specifically, we extend a pretrained single-shot text-to-video model to controllable multi-shot generation primarily through improvements in RoPE. First, we introduce Multi-Shot Narrative RoPE that breaks RoPE’s continuity at shot transitions and helps the model recognize shot boundaries, enabling controllable shot transitions. Furthermore, we design Spatiotemporal Position-Aware RoPE, which incorporates spatiotemporal-grounded control signals into RoPE for reference tokens (subjects and backgrounds). It establishes strong correlations between reference tokens and the corresponding video tokens during attention, allowing reference injection into specified spatiotemporal regions. By specifying multiple frames, users can further control subject motion. To manage the incontext information flows between reference and video tokens, we develop a Multi-Shot & Multi-Reference Attention Mask. In addition, we propose an automatic MultiShot & Multi-Reference Data Curation pipeline to provide essential training data. Comprehensive evaluations demonstrate its superior performance and outstanding controllability. We integrate diverse conditions, including text prompts, subjects, grounding signals, and backgrounds for multi-shot video generation with flexible shot arrangement, as shown in Fig. 1. We anticipate that this work could inspire future research in controllable multi-shot video generation.

##### 2. Related Works

###### 2.1. Text-to-Video Generation

Early methods inflated the pretrained text-to-image generation models [39] with temporal layers [14, 56] for video generation, achieving preliminary short video animation. Recent approaches have employed the diffusion

Multi-Shot Videos

Reference Images

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Subjects Backgrounds

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

3D VAE 3D VAE 3D VAE

3D VAE

3D VAE

+noise(video)

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

Average

SubjectLatentsVideoLatents

[Figure 74]

Subject Latents

Self Attention

⋅⋅⋅

###### 3D Attention

[Figure 75]

[Figure 76]

BG, Video Latents

[Figure 77]

𝐾 𝐾 𝐾 𝐾 𝐾 𝐾

Temporal Attention

Height

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

[Figure 78]

+𝜙

+𝜙

𝑄𝑄𝑄𝑄𝑄𝑄

Shot-Level Cross Attention

Time

Width

Multi-Shot Narrative RoPE

[Figure 79]

FFN

Subject_1_Time_1

Subject_2_Time_1

|[Figure 80]|
|---|

|[Figure 81]|
|---|

[Figure 82]

|1|
|---|

DiT Block × 𝑁

Copy

Shot_1_Time_

[Figure 83]

[Figure 84]

[Figure 85]

Text Encoder

⋅⋅⋅

Subject_1_Time_18 Subject_2_Time_18

Caption 3 Caption 2

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]<br><br>Shot_3_T|im|e_18|
|---|---|---|

Sh

Caption 1

Copy

Global: [Subjects] [Environment]; Per-Shot: [Behavior] [Background][Camera]

⋅⋅⋅

Spatiotemporal Position-Aware RoPE

Multi-Shot & Multi-Reference Attention Mask

- Figure 2. Overview of MultiShotMaster. We extend a pretrained single-shot T2V model by two key RoPE variants: Multi-Shot Narrative RoPE for flexible shot arrangement with temporal narrative order, and Spatiotemporal Position-Aware RoPE for grounded reference injection. To manage in-context information flows, we design a Multi-Shot & Multi-Reference Attention Mask. We finetune temporal attention, cross attention and FFN, leveraging the intrinsic architectural properties to achieve flexible and controllable multi-shot video generation.

transformer (DiT) [36] architectures, which could generate longer, high-quality videos with detailed text description [6, 12, 27, 45, 60, 69]. However, scaling video generation beyond short clips remains an open problem. Existing research focuses on two distinct tasks: single-shot and multi-shot long video generation. The former typically encounters issues of error accumulation and memory loss [8, 19, 20, 63, 64], while the latter focuses on narrative coherence and inter-shot consistency [7, 15, 37, 58].

might contain “boring” frames. For instance, when a director wants to capture a close-up of an actor picking up a drink, such an insert shot requires only a few frames for audiences to understand the content. To tackle this problem, ShotAdapter [22] incorporates learnable transition tokens that interact only with shot-boundary frames to indicate transitions. CineTrans [57] constructs an attention mask to weaken the inter-shot correlations, enabling transitions at predefined positions. In contrast, our work conveys the transition signals by manipulating the RoPE embeddings, which prevents interference with token interactions in pretrained attention and explicitly achieves shot transitions.

###### 2.2. Multi-Shot Video Generation

Multi-shot videos should preserve narrative logic and ensure spatiotemporal consistency of character positioning and scene layout across all shots [17, 34]. The existing paradigms mainly comprise text-to-keyframe generation & image-to-video generation [58, 67, 70] and end-toend holistic generation [7, 15, 22, 37, 51, 57]. The former depends on the generation quality of keyframes and cannot cover the character and scene consistency outside the keyframes. The latter exhibits better consistency, benefiting from the full attention along the temporal dimension. We observe that the multi-shot videos with fixed shot duration

###### 2.3. Controllable Video Generation

Providing explicit and precise user control is essential for practical content creation [47, 48, 53, 55]. The controllable video generation field supports diverse control signal types such as camera motion [1–3, 16], object motion [31, 33, 41, 59], reference video [4, 5, 26, 30]. VACE [21] and Phantom [29] support multi-reference video generation and achieve realistic composition. Tora [68] and Motion Prompting [13] control the object motion through point tra-

jectories. However, existing methods typically focus on the single-shot setting and adopt separate adapters for reference injection and motion control. If following the traditional paradigm [29, 49], controllability in multi-shot settings would require larger networks and incur higher computational costs. To address this limitation, we propose the first controllable multi-shot framework that supports reference injection and motion control jointly, requiring no additional adapters.

##### 3. Method 3.1. Evolving from Single-Shot to Multi-Shot T2V

Preliminary: Our model is developed upon a pretrained single-shot text-to-video (T2V) model with ∼1B parameters, which consists of a 3D Variational Auto-Encoder (VAE) [24], T5 text encoder [38] and a latent diffusion transformer (DiT) model [36]. Each basic DiT block contains a sequence including a 2D spatial self-attention module, a 3D spatiotemporal self-attention module, a text crossattention module and a feed-forward network (FFN). We define a straight path from clean data z0 to noised data zτ at timestep τ using Rectified Flow [11]: zt = (1 − τ)z0 + τϵ, where ϵ ∈ N(0,I). The denoising process follows the ordinary differential equation: dzτ = vΘ(zτ,τ,ctext)dτ, where vΘ is the denoising network. The training objective is to regress velocity [28]:

0 ∥(z1 − z0) − vΘ(zτ,τ,ctext)∥22 (1)

LLCM = Eτ,ϵ,z

As shown in Fig. 2, to adapt the input from single-shot to multi-shot videos with the sudden content changes at shot boundaries, we encode each shot separately through 3D VAE and then concatenate the video latents. During temporal attention, the original 3D-RoPE assigns sequential indices along the temporal dimension, leading to a critical issue: the model cannot distinguish between intra-shot consecutive frames with inter-shot frames across shot boundaries. To explicitly help the model perceive shot boundaries, we propose a Multi-Shot Narrative RoPE mechanism that introduces an angular phase shift into the original 3D-RoPE for each transition. The query (Q) of i-th shot is computed as follows, and similarly for key (K):

Qi = RoPE((t + iϕ) · f, h · f, w · f) ⊙ Q˜i (2)

where (t,h,w) are spatiotemporal position indices, ϕ is the angular phase shift factor, f is the decreasing base frequency vector, and ⊙ denotes the element-wise rotary transformation of query embeddings Q˜i via complex rotations. This design not only maintains the narrative shooting order of inter-shot frames, but also leverages RoPE’s inherent rotational properties to mark shot boundaries through fixed phase shifts, requiring no additional trainable parameters. It

enables users to flexibly configure both the number of shots and their respective durations.

Considering that providing users with the capability to customize each shot individually could facilitate content creation, we design a hierarchical prompt structure. It includes a global caption that describes subject appearances and environments, and per-shot captions that detail subject actions, backgrounds, and camera, following [15]. For each shot, we combine the global caption with the corresponding per-shot caption. In the vanilla T2V model, T5 encoder encodes the text prompts as text embeddings which are then replicated along the temporal dimension to align with the video frame sequence for text-frame cross attention. Accordingly, we replicate each shot’s text embeddings to align with the corresponding shot frame count, enabling shot-level cross-attention.

###### 3.2. Spatiotemporal-Grounded Reference Injection

Users typically require the capability to provide reference images (e.g., subjects and backgrounds) and motion control signal for creating customized video content. To address this requirement, we propose Spatiotemporal Position-Aware RoPE to improve in-context learning for spatiotemporal-grounded reference injection. Specifically, we individually encode each reference image into the latent space through 3D VAE and concatenate them with the noised video latents via token concatenation. In temporal attention, clean reference tokens propagate visual information to noisy video tokens for reference injection. Furthermore, 3D-RoPE enables tokens with closer spatiotemporal distance to attend more to each other. Inspired by this mechanism, we apply 3D-RoPE from specified regions to the corresponding reference tokens, thereby establishing strong correlations between region-specified video tokens and reference tokens, as shown in Fig. 2. Since the subject bounding box region (x1,y1,x2,y2) at t-th frame is smaller than the spatial dimensions (H,W) of reference tokens, we sample the 3D-RoPE by:

Qref = RoPE((t + iϕ) · f, href · f, wref · f) ⊙ Q˜ref, href = y1 +

(y2 − y1)

H · j, j ∈ [0,H − 1], (3) wref = x1 +

(x2 − x1) W · k, k ∈ [0,W − 1]

In this manner, we can control the subject to appear at a specified spatiotemporal position. To control the motion trajectory of a subject, we create multiple copies of the subject tokens and assign different spatiotemporal RoPE to each copy. The temporal attention then transfers the subject motion embedded in these copies to video tokens at the corresponding spatiotemporal positions. The copied tokens of each subject will be averaged after attention. Similarly,

Sample Sample Sample

[Figure 89]

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

#### ···

[Figure 147]

[Figure 148]

[Figure 149]

TransNet v2

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

[Figure 183]

Long Video Data Collection

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

[Figure 197]

Scene 1

Scene42 Scenen ···

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Shot Transition Detection Scene Segmentation

235k Multi-Shot Samples

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Global Caption

Based on:

Shot 1: Subjectin what appears1 is standingto be ancloseintimateto Subjector intense2, facingconversation….each other

- 1. {Globel Caption}
- 2. {Each Shot}

Subject 1: A young woman with curly reddish-brown hair, wearing a dark blue dress or coat; Subject 2: A young man with blonde hair, wearing a dark jacket over a light-colored shirt. The whole scene takes place in a classical indoor with arched windows, creating an intimate and dramatic atmosphere.

[Figure 217]

###### ···

[Figure 218]

[Figure 219]

Shot n: Subjectof the frame.2 is positionedHe appearsintoprofile,be in agazingcontemplative….toward the left side

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

###### Reasoning the per-shot captions

Per-Shot Captions

[Figure 225]

Shot 1 Subjects Shot 2 Subjects

Subject 1 Subject 2

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

YOLOv11

Merge Subjects and Tracking Results

[Figure 242]

[Figure 243]

[Figure 244]

SAM

###### ··· ···

[Figure 245]

###### Track_ID1 Track_ID2 Track_ID2 Track_ID1

[Figure 246]

··· ···

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

OmniEraser

ByteTrack

[Figure 262]

[Figure 263]

Multi-Shot Videos

First Frames & Masks

Backgrounds

- Figure 3. Data Curation Pipeline: (1) We employ a shot transition detection model [43] to cut the collected long videos into short clips, use a scene segmentation model [54] to cluster clips within the same scene, and then sample multi-shot videos. (2) We introduce a hierarchical caption structure and use Gemini-2.5 [10] in a two-stage process to produce global caption and per-shot captions. (3) We integrate YOLOv11 [23], ByteTrack [66] and SAM [25] to detect, track and segment subject images. Then we use Gemini-2.5 to merge the per-shot tracking results by subject appearance. We obtain clean backgrounds by using OmniEraser [52].

to achieve multi-shot scene customization, we copy the 3DRoPE from the first frame of each shot and apply it to the corresponding background tokens. We further clarify the details in temporal attention in Appendix A.1.

tion. As shown in Fig. 2, we maintain full attention across all multi-shot video tokens for global consistency, and limit each shot to access only the reference tokens within its own shot. Accordingly, the reference tokens of each shot can only attend to each other and the video tokens within the same shot. This attention mask strategy effectively ensures that each shot focuses on the intra-shot reference injection, and keeps global consistency by inter-shot full attention.

By incorporating spatiotemporal-controllable multireference injection, we significantly expand the functional boundaries of multi-shot video generation, providing users with more flexible and powerful video creation capabilities. It allows users to customize characters using subject images and control their positions and movements, which improves the practicality of multi-shot video generation. In addition, using multiple background images of a scene can achieve customized multi-shot scene consistency, reducing the necessity for location shooting.

###### 3.4. Training and Inference Paradigm

Typically, controllable capabilities are trained based on a foundation generation model. Considering that the reference injection task requires training on large-scale data to learn diverse subjects, and the construction cost of multishot & multi-reference data is relatively high, we first train spatiotemporal-specified reference injection on 300k single-shot data. We sample bounding boxes with random starting points and 1-second intervals, where each bounding box has a 0.5 drop probability. This sparse bounding box sequence allows users to control subjects easily. In the second stage, we train on the constructed multi-shot & multireference data. To enable controllable multi-shot video generation with text-, subject-, background-, and joint-driven modes, we randomly drop the subject and background, each with a 0.5 probability. We notice that the training objective Eq. 1 focuses on global consistency and ignores details. Therefore, we propose a cross-shot subject-focused post-

###### 3.3. Multi-Shot & Multi-Reference Attention Mask

The user-provided reference images and subject copies may lead to excessively long contexts with high computational costs. On the other hand, there are unnecessary interactions between in-context tokens. For example, Subject 0 may only appear in shot 2, but unconstrained token concatenation allows other shots to access Subject 1 as well. Although 3D-RoPE can guide spatiotemporal-specified video tokens to attend to reference tokens, the small but non-zero attention weights still pose a content leakage risk. Therefore, we design a multi-shot & multi-reference attention mask to constrain information flow and optimize attention alloca-

training that assigns (2×) loss weight to subject regions versus (1×) to backgrounds. It not only improves subject consistency but also enables the model to better comprehend how the subjects change across different shots. More details could be found in Appendix A.2.

During inference, our framework supports controllable multi-shot video generation with text-driven inter-shot consistency, customized subject with motion control, and background-driven customized scene consistency. Both shot count and duration are flexibly configurable. This versatile framework opens new possibilities for diverse multishot video content creation, enabling users to craft highly customized video narratives.

###### 3.5. Multi-Shot & Multi-Reference Data Curation

Multi-Shot Videos: To construct a multi-shot video dataset, as shown in Fig. 3, we first crawl long videos from the Internet (including diverse types: movies, television series, documentaries, cooking demonstrations, sports and fitness). Then we use TransNet V2 [43] to detect shot transitions and crop out massive single-shot videos. To merge the single-shot videos captured in the same scene, we employ a scene segmentation method [54] that could understand the storyline of the video to figure out where a scene starts and ends. It might cluster videos spanning tens of minutes within the same scene into a single group. We further sample multi-shot videos using the following strategy: shot count ranges from 1 to 5, frame count ranges from 77 to 308 (i.e., 5-20 seconds at 15 fps), with priority given to samples with higher frame counts and more shots.

Caption Definition: To provide users with the ability to define characters and customize each shot, we employ a hierarchical caption structure: global caption and per-shot captions, following [15]. We first use Gemini-2.5 [10] to understand the entire multi-shot video and produce a global caption, where each subject is denoted by “Subject X,X ∈ [1,2,3,···]”. Subsequently, based on the global caption and each shot video, we employ Gemini-2.5 to reason the per-shot captions, using the predefined “Subject X” across all captions. It helps the model understand crossshot subject consistency. More details could be found in Appendix A.3.

Reference Images Collection: We first apply YOLOv11 [23], ByteTrack [66] and SAM [25] to detect, track and segment subject images. Due to the presence of shot transitions, the tracking process is conducted shot-by-shot. This process produces the shot-level track IDs and their bounding box sequences. To merge the tracking results across all shots, we choose the largest subject image of each shot-level track ID from each shot, and group them using Gemini2.5 (Details could be found in Appendix A.4). In this way, we obtain complete multi-shot tracking results and the corresponding subject images. In addition, we feed the

first frames of each shot and its foreground mask into OmniEraser [52] to obtain clean backgrounds.

##### 4. Experiment

###### 4.1. Experimental Setup

Implementation Details. Our framework is based on a pretrained single-shot T2V model with only ∼1B parameters at the resolution of 384 × 672. We conduct experiments for controllable multi-shot video generation on narrative videos containing 77-308 frames at 15 fps (i.e., 5-20 seconds), with each video comprising 1-5 shots. We encode each shot separately through 3D VAE [24], and employ a sliding window strategy to encode and decode shots with > 77 frames, which maintains the alignment between pixel space and latent space for multi-shot videos. We train the model on 32 GPUs, with a learning rate of 1 × 10−5, batch size 1. The angular phase shift factor of Multi-Shot Narrative RoPE is set to 0.5 by default. During inference, we set the classifierfree guidance scale [18] as 7.5 and DDIM [42] steps as 50. More details can be found in the supplementary materials.

Baselines. We compare our work with two multi-shot video generation methods [46, 57]. CineTrans [57] is the latest open-source multi-shot narrative method. EchoShot [46] focuses on identity-consistent multi-shot portrait videos, rather than narrative coherence. In addition, considering that there is no controllable multi-shot method, we employ single-shot reference-to-video methods Phantom [29] and VACE [21] to generate multiple single-shot videos with individual text prompts from a story for comparison. The competing baselines are all based on Wan2.1-T2V1.3B [45] at the resolution 480 × 832.

Evaluation. To comprehensively evaluate our work, we design 100 multi-shot prompts using Gemini-2.5 [10]. To ensure fairness, we process the text prompts with the corresponding style for each baseline. Given that both subject consistency and scene consistency are crucial for multi-shot reference injection, we construct 90 cases encompassing three settings: subject injection, background injection, and joint injection, with 30 cases for each setting.

Metrics. We evaluate multi-shot narrative video generation from four perspectives: (1) Text Alignment (TA): we calculate the similarity between text features and shot features extracted by ViCLIP [50]. (2) Inter-Shot Consistency: first, we calculate the holistic semantic similarity between ViCLIP shot features. Then, we apply YOLOv11 [23] and SAM [25] to detect and crop subjects and backgrounds from keyframes (first, middle, and last frames), and subsequently employ DINOv2 [35] to measure subject consistency and scene consistency. (3) Transition Deviation: we employ TransNet V2 [43] to detect transitions in the generated videos and calculate the frame count deviation from the ground-truth transition timestamps. (4)

[Figure 264]

[Figure 265]

[Figure 266]

|[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>Shot 1: [ ] and [ ] talks.|
|---|

|[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>Shot 2: [ ] is writing beside [ ].|
|---|

|[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>Shot 3: Close-up, [ ] is writing.|
|---|

|[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>Shot 4: [ ] and [ ] talks.|
|---|

###### Character Descriptions

###### Ours(w/oref)EchoShotCineTransPhantomVACEOurs(w/ref)

[ ]:An older man with curly

grey hair and glasses, wearing a light shirt and patterned tie.

[ ]:An older woman with blonde hair and sunglasses, wearing a blue patterned dress,

|[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>Shot 2: [ ] is writing beside [ ].|
|---|

|[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>Shot 4: [ ] and [ ] talks.|
|---|

|[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>Shot 1: [ ] and [ ] talks.|
|---|

|[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>Shot 3: Close-up, [ ] is writing.|
|---|

[Figure 299]

I need to repeat character descriptions in every shot caption.

Global Caption

- Subject 1: An older man with curly grey hair and glasses, wearing a light shirt and a patterned tie.
- Subject 2: An older woman with blonde hair and sunglasses, wearing a blue patterned dress. …

|[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>Shot 1: Subject 1 and Subject 2 talks.|
|---|

|[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>Shot 2: Subject 1 is writing beside Subject 2.|
|---|

|[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>[Figure 311]<br><br>Shot 3: Close-up, Subject 1 is writing.|
|---|

|[Figure 312]<br><br>[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]<br><br>Shot 4: Subject 1 and Subject 2 talks.|
|---|

[Figure 316]

[Figure 317]

[Figure 318]

|[Figure 319]<br><br>[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]<br><br>Shot 1: [ ]is talking to [ ].|
|---|

|[Figure 323]<br><br>[Figure 324]<br><br>[Figure 325]<br><br>[Figure 326]<br><br>Shot 2: [ ]is talking to [ ].|
|---|

|[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>Shot 3: [ ]is thinking beside [ ].|
|---|

|[Figure 331]<br><br>[Figure 332]<br><br>[Figure 333]<br><br>[Figure 334]<br><br>Shot 4: [ ]is smiling at [ ].|
|---|

[Figure 335]

[Figure 336]

###### ···

|[Figure 337]<br><br>[Figure 338]<br><br>[Figure 339]<br><br>[Figure 340]<br><br>Shot 1: [ ]is talking to [ ].|
|---|

|[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]<br><br>Shot 2: [ ]is talking to [ ].|
|---|

|[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]<br><br>[Figure 348]<br><br>Shot 3: [ ]is thinking beside [ ].|
|---|

|[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]<br><br>[Figure 352]<br><br>Shot 4: [ ]is smiling at [ ].|
|---|

Backgrounds

[Figure 353]

[Figure 354]

Subject 1 Subject 2 Reference Images

|[Figure 355]<br><br>[Figure 356]<br><br>[Figure 357]<br><br>[Figure 358]<br><br>Shot 1: Subject 1 is talking to Subject 2.|
|---|

|[Figure 359]<br><br>[Figure 360]<br><br>[Figure 361]<br><br>[Figure 362]<br><br>Shot 2: Subject 2 is talking to Subject 1.|
|---|

|[Figure 363]<br><br>[Figure 364]<br><br>[Figure 365]<br><br>[Figure 366]<br><br>Shot 3: Subject 1 is thinking beside Subject 2.|
|---|

|[Figure 367]<br><br>[Figure 368]<br><br>[Figure 369]<br><br>[Figure 370]<br><br>Shot 4: Subject 2 is smiling at Subject 1.|
|---|

[Figure 371]

t=0

[Figure 372]

t=41

Subject 1 Subject 2

Subject 1 Subject 2

###### ···

Grounding Signals

- Figure 4. Qualitative Comparisons. We compare with two multi-shot video generation methods [46, 57] in the upper part, and two singleshot reference-to-video methods [21, 29] under multi-shot setting in the lower part. [ ] denotes the placeholder of character descriptions for baselines. The character introductions of the bottom part are omitted for brevity.

Narrative Coherence: we employ Gemini-2.5 [10] to evaluate the narrative logic of multi-shot videos (Details of this metric could be found in Appendix A.5). In addition, we evaluate the Reference Injection Consistency from two perspectives: (1) We detect and crop the generated subjects and backgrounds, and calculate DINO similarity with the provided references. (2) Grounding: we detect the 2D object bounding boxes and calculate the mean Intersection over Union (mIoU) across keyframes to measure the spatiotemporal-grounded accuracy.

individual caption manner for each shot. All these methods need to repeat the character descriptions in every shot caption, which makes it inconvenient for users. In comparison, we adopt the user-friendly hierarchical caption structure that describes subject appearance in the global caption and uses the indexed nouns in per-shot captions.

In the upper part, CineTrans [57] shows limited variation in camera positioning across shot clips and fails to preserve character identity consistency. This stems from that CineTrans manipulates the attention score for shot transitions, impeding the original token interactions in pretrained attention. EchoShot [46] also designs RoPE-based shot transition for generating multiple portrait video clips that mainly focuses on identity consistency, ignoring other narrative details such as inconsistent clothing colors. Our method implements text-driven cross-shot subject consistency and scene consistency. Notably, the vehicle roof occupies a small area within Shot 3, yet it still maintains consistent

###### 4.2. Qualitative Comparison

As shown in Fig. 4, we present two different feature comparisons for multi-shot text-to-video generation and multishot reference-to-video generation. The simplified prompts for each shot are shown in the subtitle. CineTrans [57] uses a global caption that focuses on the scene and camera transitions, and per-shot captions. Other baselines follow the

Table 1. Quantitative Evaluations. ✗ denotes that this feature is not supported. The upper part compares multi-shot text-to-video generation, and the lower part compares multi-shot reference-to-video generation. In comparison, we achieve superior performance across all evaluation metrics, and further provide the exceptional spatiotemporal-grounded reference injection capabilities.

Inter-Shot Consistency↑ Transition Deviation↓

Narrative Coherence↑

Reference Consistency↑

Text Align.↑

Semantic Subject Scene Subject Background Grounding

CineTrans [57] 0.174 0.683 0.437 0.389 5.27 0.496 ✗ ✗ ✗ EchoShot [46] 0.183 0.617 0.425 0.346 3.54 0.213 ✗ ✗ ✗ Ours (w/o Ref) 0.196 0.697 0.491 0.447 1.72 0.695 ✗ ✗ ✗ VACE [21] 0.201 0.599 0.468 0.273 ✗ 0.325 0.475 0.361 ✗ Phantom [29] 0.224 0.585 0.462 0.279 ✗ 0.362 0.490 0.328 ✗ Ours (w/ Ref) 0.227 0.702 0.495 0.472 1.41 0.825 0.493 0.456 0.594

VACE [21] and Phantom [29] are implemented by performing multiple independent inferences for multi-shot reference-to-video generation, so we do not calculate their transition deviation. The inter-shot consistency stems only from the text prompts and reference images, leading to suboptimal inter-shot consistency and poor narrative coherence. These methods struggle to preserve user-provided backgrounds, resulting in inferior scene consistency. In comparison, we deliver outstanding performance across all evaluation metrics while providing additional support for spatiotemporal-grounded reference injection capabilities.

###### Bad Generation Result

| |[Figure 373]<br><br>[Figure 374]<br><br>[Figure 375]<br><br>|
|---|---|
| |[Figure 376]<br><br>[Figure 377]<br><br>[Figure 378]<br><br>Good Generation Result|

[Figure 379]

[Figure 380]

Background

[Figure 381]

Subject

[Figure 382]

- Figure 5. Limitation visualization. We only explicitly control the subject motion, while the camera position is controlled by text prompts, which might cause the motion coupling issue.

Due to space constraints, we present ablation studies about key components and training strategy in Appendix B.

color with the vehicle roof in shot 2.

In the lower part, we feed subject images and background images into VACE [21] and Phantom [29] for multishot reference-to-video generation and perform inference multiple times using individual shot captions for each shot. Since all shots are generated independently, they fail to maintain inter-shot subject consistency. For instance, in the fourth row, the woman wears different clothing between Shot 1 and Shot 3. And these methods fail to fully preserve the user-provided background reference images. In contrast, we achieve satisfactory reference-driven subject consistency and scene consistency, and further support grounding signals to control the subject injection into specified regions and background injection into specified shots.

##### 5. Conclusion

In this work, we propose MultiShotMaster, the first controllable multi-shot video generation framework. We extend a pretrained text-to-video model through two key RoPE improvements: Multi-Shot Narrative RoPE for recognizing shot boundaries and enabling controllable transitions, and Spatiotemporal Position-Aware RoPE for injecting reference tokens (subjects and backgrounds) into specific spatiotemporal regions. We also propose an automatic multishot & multi-reference data curation pipeline to extract multi-shot videos, captions, cross-shot grounding signals and reference images. Our method leverages the intrinsic architectural properties to integrate text prompts, subjects, grounding signals, and backgrounds for flexible multi-shot video generation with superior controllability. We anticipate that this work could inspire future research in controllable multi-shot video generation.

###### 4.3. Quantitative Comparison

We report the quantitative comparison results in Table 1. Since CineTrans [57] adds a mask matrix to the attention score, weakening the correlations across different shots, which results in unsatisfactory inter-shot consistency. On the other hand, as shown in the row 1 of Fig. 4, its shot transitions are not significant, leading to inferior transition deviation score and text alignment. EchoShot [46] is designed for generating multiple portrait video clips rather than creating narrative content, therefore it exhibits limited narrative coherence. Benefiting from the effectiveness of the proposed framework, we achieve superior inter-shot consistency, transition deviation score and narrative coherence.

Limitation and Future Works. Although our method exhibits strong controllability, several key limitations require additional research to address: (1) We experiment on a pretrained single-shot T2V model with only ∼1B parameters at the resolution of 384 × 672, which lags behind current open-source models like the WAN family of models [45] at the resolution 480 × 832 used by baselines. Therefore, the generation quality still needs improvement. In the future,

we will implement our work on WAN 2.1/2.2 and release code. (2) We explicitly control the subject motion, while the camera position is controlled by text prompts. As shown in Fig 5, although the generated video aligns the grounding signals, this is a consequence of the camera and object moving together. We leave this coupling issue as future work.

##### References

- [1] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multicamera video generation from diverse viewpoints. arXiv preprint arXiv:2412.07760, 2024. 3
- [2] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Cameracontrolled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025.
- [3] Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yijin Li, Fu-Yun Wang, and Hongsheng Li. Gs-dit: Advancing video generation with dynamic 3d gaussian fields through efficient dense 3d point tracking. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21717–21727, 2025. 3
- [4] Weikang Bian, Xiaoyu Shi, Zhaoyang Huang, Jianhong Bai, Qinghe Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Hongsheng Li. Relightmaster: Precise video relighting with multi-plane light images. arXiv preprint arXiv:2511.06271, 2025. 3
- [5] Yuxuan Bian, Xin Chen, Zenan Li, Tiancheng Zhi, Shen Sang, Linjie Luo, and Qiang Xu. Video-asprompt: Unified semantic control for video generation. arXiv preprint arXiv:2510.20888, 2025. 3
- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. 2, 3
- [7] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025. 2, 3
- [8] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets fullsequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 3
- [9] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Shortto-long video diffusion model for generative transition

- and prediction. In The Twelfth International Conference on Learning Representations, 2023. 2
- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 5, 6, 7, 13, 14, 17, 18
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 4
- [12] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025. 3
- [13] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Yusuf Aytar, Michael Rubinstein, Chen Sun, et al. Motion prompting: Controlling video generation with motion trajectories. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1–12, 2025. 2, 3
- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725,

2023. 2

- [15] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025. 2, 3, 4, 6
- [16] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3
- [17] Jingwen He, Hongbo Liu, Jiajun Li, Ziqi Huang, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Cut2next: Generating next shot via in-context tuning. arXiv preprint arXiv:2508.08244, 2025. 3
- [18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598,

2022. 6

- [19] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198, 2025. 3

- [20] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 3
- [21] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17191–17202, 2025. 2, 3, 6, 7, 8
- [22] Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M Rehg, and Tobias Hinz. Shotadapter: Text-to-multi-shot video generation with diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28405–28415,

2025. 2, 3

- [23] Rahima Khanam and Muhammad Hussain. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725, 2024. 5, 6
- [24] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4, 6
- [25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 5, 6
- [26] Baolu Li, Yiming Zhang, Qinghe Wang, Liqian Ma, Xiaoyu Shi, Xintao Wang, Pengfei Wan, Zhenfei Yin, Yunzhi Zhuge, Huchuan Lu, et al. Vfxmaster: Unlocking dynamic visual effect generation via incontext learning. arXiv preprint arXiv:2510.25772,

2025. 3

- [27] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 2, 3
- [28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4
- [29] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025. 2, 3, 4, 6, 7, 8
- [30] Yawen Luo, Jianhong Bai, Xiaoyu Shi, Menghan Xia, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Tianfan Xue. Camclonemaster: Enabling referencebased camera control for video generation. arXiv preprint arXiv:2506.03140, 2025. 3
- [31] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow

- your pose: Pose-guided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4117–4125, 2024. 3
- [32] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869, 2025. 2
- [33] Yue Ma, Yulong Liu, Qiyuan Zhu, Ayden Yang, Kunyu Feng, Xinhua Zhang, Zhifeng Li, Sirui Han, Chenyang Qi, and Qifeng Chen. Followyour-motion: Video motion transfer via efficient spatial-temporal decoupled finetuning. arXiv preprint arXiv:2506.05207, 2025. 3
- [34] Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Yixuan Li, Cheng Chen, Yanhong Zeng, et al. Holocine: Holistic generation of cinematic multi-shot long video narratives. arXiv preprint arXiv:2510.20822, 2025. 3
- [35] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6
- [36] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 2, 3, 4
- [37] Tianhao Qi, Jianlong Yuan, Wanquan Feng, Shancheng Fang, Jiawei Liu, SiYu Zhou, Qian He, Hongtao Xie, and Yongdong Zhang. Maskˆ 2dit: Dual mask-based diffusion transformer for multi-scene long video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18837–18846, 2025. 2, 3
- [38] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67,

2020. 4

- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [40] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-

- scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 14
- [41] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-tovideo generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers, pages 1– 11, 2024. 3
- [42] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 6
- [43] Tom´as Soucek and Jakub Lokoc. Transnet v2: An effective deep network architecture for fast shot transition detection. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11218– 11221, 2024. 5, 6
- [44] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 2
- [45] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 6, 8
- [46] Jiahao Wang, Hualian Sheng, Sijia Cai, Weizhan Zhang, Caixia Yan, Yachuang Feng, Bing Deng, and Jieping Ye. Echoshot: Multi-shot portrait video generation. arXiv preprint arXiv:2506.15838, 2025. 6, 7, 8
- [47] Qinghe Wang, Xu Jia, Xiaomin Li, Taiqing Li, Liqian Ma, Yunzhi Zhuge, and Huchuan Lu. Stableidentity: Inserting anybody into anywhere at first sight. IEEE Transactions on Multimedia, 2025. 3
- [48] Qinghe Wang, Baolu Li, Xiaomin Li, Bing Cao, Liqian Ma, Huchuan Lu, and Xu Jia. Characterfactory: Sampling consistent characters with gans for diffusion models. IEEE Transactions on Image Processing, 2025. 3
- [49] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3daware and controllable framework for cinematic textto-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1– 10, 2025. 2, 4
- [50] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text

- dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 6
- [51] Cong Wei, Bo Sun, Haoyu Ma, Ji Hou, Felix JuefeiXu, Zecheng He, Xiaoliang Dai, Luxin Zhang, Kunpeng Li, Tingbo Hou, et al. Mocha: Towards moviegrade talking character synthesis. arXiv preprint arXiv:2503.23307, 2025. 3
- [52] Runpu Wei, Zijin Yin, Shuo Zhang, Lanxiang Zhou, Xueyi Wang, Chao Ban, Tianwei Cao, Hao Sun, Zhongjiang He, Kongming Liang, et al. Omnieraser: Remove objects and their effects in images with paired video-frame data. arXiv preprint arXiv:2501.07397,

2025. 5, 6

- [53] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Feng Liu, Zhizhong Huang, Jiaxin Ye, Yingya Zhang, and Hongming Shan. Dreamvideo-2: Zero-shot subjectdriven video customization with precise motion control. arXiv preprint arXiv:2410.13830, 2024. 3
- [54] Haoqian Wu, Keyu Chen, Yanan Luo, Ruizhi Qiao, Bo Ren, Haozhe Liu, Weicheng Xie, and Linlin Shen. Scene consistency representation learning for video scene segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14021–14030, 2022. 5, 6
- [55] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized textto-video generation. NeurIPS, 2024. 3
- [56] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tunea-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7623–7633, 2023. 2
- [57] Xiaoxue Wu, Bingjie Gao, Yu Qiao, Yaohui Wang, and Xinyuan Chen. Cinetrans: Learning to generate videos with cinematic transitions via masked diffusion models. arXiv preprint arXiv:2508.11484, 2025. 2, 3, 6, 7, 8
- [58] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. arXiv preprint arXiv:2507.18634, 2025. 2, 3
- [59] Jinbo Xing, Long Mai, Cusuh Ham, Jiahui Huang, Aniruddha Mahapatra, Chi-Wing Fu, Tien-Tsin Wong, and Feng Liu. Motioncanvas: Cinematic shot design with controllable image-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025. 3

- [60] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072,

2024. 2, 3

- [61] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025. 2
- [62] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwaxl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023. 2
- [63] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025. 3
- [64] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626,

2025. 3

- [65] Ruihan Zhang, Borou Yu, Jiajian Min, Yetong Xin, Zheng Wei, Juncheng Nemo Shi, Mingzhen Huang, Xianghao Kong, Nix Liu Xin, Shanshan Jiang, et al. Generative ai for film creation: A survey of recent advances. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6267–6279,

2025. 2

- [66] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In European conference on computer vision, pages 1–21. Springer,

2022. 5, 6

- [67] Yuang Zhang, Junqi Cheng, Haoyu Zhao, Jiaxi Gu, Fangyuan Zou, Zenghui Lu, and Peng Shu. Shouldershot: Generating over-the-shoulder dialogue videos. arXiv preprint arXiv:2508.07597, 2025. 3
- [68] Zhenghao Zhang, Junchao Liao, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2063–2073, 2025. 3
- [69] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 2, 3

[70] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems, 37:110315–110340, 2024. 2, 3

# Appendix

##### A. More Implementation Details

###### A.1. Details in Temporal Attention

To clarify the designs in temporal attention, including Multi-Shot Narrative RoPE and Spatiotemporal PositionAware RoPE, we provide an Algorithm 1. Specifically, the complete in-context latents Z contain multi-shot video latents z = [zi]N

i=1 and reference latents zref = [zm]Nmref=1. Nshot represents shot count, Nref represents the number of input reference images (subjects and backgrounds). The input bounding box sequences of references [boxes]N

shot

box

b

contain Nbox bounding boxes. Each bounding box is represented as [(m,t,x1,y1,x2,y2)], indicating the bounding box of m-th reference at t-th frame. Note that for background references, the bounding boxes are fixed as (m,t,0,0,H,W), where t is the first frame of the corresponding shot.

In temporal attention, the linear projections to q, to k, to v first transform in-context latents Z to Q˜, K˜, V˜. Then, by applying the Multi-Shot Narrative RoPE (i.e., Eq. 2 in the main paper), the query and key of each shot are introduced explicit shot transition signals, while keeping the narrative temporal order. For m-th reference containing Nboxm boxes, we copy the query and key of the m-th reference Nboxm times. Each copy is then applied with a Spatiotemporal Position-Aware RoPE based on the corresponding box in [boxes]N

m box

b . Since RoPE is not applied to value component in attention mechanism, we copy the value for attention computation. After the attention computation with the proposed multi-shot & multi-reference attention mask, we aggregate the Nboxm reference copies for m-th reference by taking their mean. Finally, the multi-shot video zˆ and the reference latents z¯ref are concatenated along the token dimension and fed into the linear projection to out. The attention output Z∗ maintains the same dimension with the input in-context latents Z.

###### A.2. Training Paradigm

The three-stage training paradigm consists of: (1) we finetune the temporal attention for spatiotemporal-specified reference injection on 300k single-shot video data with 30 epochs, batch size 8, while keeping other model parameters frozen. (2) we finetune temporal attention, cross attention and FFN on 235k multi-shot & multi-reference data with 3 epochs, batch size 1. (3) following the second stage, we assign (2×) loss weight to subject regions and (1×) to backgrounds to train 0.5 epoch. We conduct ablation study for the training paradigm in Sec B.2 and Table 4.

Algorithm 1 Temporal Attention with Multi-Shot Narrative RoPE and Spatiotemporal Position-Aware RoPE. Input:

- • In-context latents Z containing:

- – Multi-shot video latents z = [zi]N

shot

i=1

- – Reference latents zref = [zm]Nmref=1

- • Bounding box sequences of references (subjects and

backgrounds) [boxes]N

b = [(m,t,x1,y1,x2,y2)]N

box

box

b Output: In-context latents Z∗ after temporal attention

- 1: Q˜ = to q(Z), K˜ = to k(Z), V˜ = to v(Z) // Apply Multi-Shot Narrative RoPE

- 2: Q = [Qi]N

shot

i=1 = Eq. 2([Q˜i]N

shot

i=1 ) K = [Ki]N

shot

i=1 = Eq. 2([K˜i]N

shot

i=1 ) V = [V˜i]N

shot

i=1

// Apply Spatiotemporal Position-Aware RoPE

- 3: Qref = [Qrefb ]Nb=0box = Eq. 3(Copy(Q˜ref), [boxes]Nb box) Kref = [Kbref]Nb=0box = Eq. 3(Copy(K˜ref), [boxes]Nb box) V ref = [V˜bref]Nb=0box = Copy(V˜ref, [boxes]Nb box) // Attention Computataion
- 4: Zˆ = Attention([Q,Qref],[K,Kref],[V,V ref], Mask) // Reference Aggregation
- 5: z¯ref = [¯zm]Nmref=1 = [ mean([ˆzm]N

m box

b ) ]Nmref=1

- 6: Z∗ = to out([ˆz,z¯ref])

- 7: return Z∗

###### A.3. Labeling Hierarchical Captions

As introduced in Sec 3.5 of the main paper, we employ Gemini-2.5 [10] to label the global caption and per-shot captions. The prompt template is shown in Fig. 6. We begin by proportionally sampling 20 frames from the multishot video, ensuring at least one frame is extracted from each shot, and use Gemini-2.5 to produce a comprehensive global caption. Then we employ Gemini-2.5 to reason the per-shot captions based on the global caption and each shot video (with a sampling frame stride of 15). Each subject is denoted by “Subject X,X ∈ [1,2,3]”. As shown in Fig. 7, the cross-shot consistency of subject annotations is satisfactory due to the powerful Gemini-2.5 and our carefullydesigned prompt template.

###### A.4. Merge Cross-Shot Tracking Annotations

As introduced in Sec 3.5 of the main paper, we conduct the tracking process shot-by-shot to obtain the bounding box sequence of each subject. To merge the cross-shot tracking results, we use Gemini-2.5 [10] to group the subject images by prompting with our carefully-designed prompt template

Table 2. Ablation study for Multi-Shot RoPE. We experiment on multi-shot text-to-video generation without reference input.

Narrative Semantic Subject Scene Coherence↑

Inter-Shot Consistency↑ Transition Deviation↓

w/o MS RoPE 0.702 0.486 0.455 4.68 0.645 Ours (w/o Ref) 0.697 0.491 0.447 1.72 0.695

- as shown in Fig. 8.

- A.5. Narrative Coherence

To comprehensively assess the narrative coherence of generated multi-shot videos, we employ Gemini-2.5 [10] to construct an automated evaluation metric. We begin by proportionally sampling 20 frames from the multi-shot video, ensuring at least one frame is extracted from each shot. Subsequently, we input these frames and the hierarchical captions as a pair into Gemini-2.5. We require Gemini-2.5 to strictly adhere to cinematic narrative logic and scrutinize cross-shot content across four core dimensions: Scene Consistency, Subject Consistency, Action Coherence, and Spatial Consistency, by the constructed elaborate instructions as shown in Fig. 9.

Specifically, Scene Consistency verifies the stability of the background, lighting, and atmosphere during transitions to ensure all shots depict the same setting; Subject Consistency strictly scrutinizes identity features and appearance attributes by comparing core objects across different viewpoints to detect unintended deviations; Action Coherence focuses on evaluating the temporal logic of dynamic behaviors to determine whether actions in subsequent shots constitute reasonable continuations of preceding ones; and Spatial Consistency examines whether the topological structure of relative positional relationships between subjects remains constant in accordance with cinematic language. Functioning as a binary classifier, the model outputs a “True” or “False” verdict for each dimension, thereby quantifying the generative model’s capability in handling complex multishot spatiotemporal consistency.

- B. Ablation Study

###### B.1. Ablation Study for Network Design

We experiment with different settings to validate the effectiveness of the proposed designs in our framework:

- • “w/o MS RoPE”: without Multi-Shot Narrative RoPE, the shot transitions rely only on the per-shot captions.
- • “w/o Mean”: this setting randomly selects one copy from multiple copies of subject tokens after 3D attention, instead of averaging.
- • “w/o Attn Mask”: without Multi-Shot & Multi-Reference Attention Mask, this setting uses full attention along the

Table 3. Ablation study for reference injection. We experiment on multi-shot reference-to-video generation.

Aesthetic Score↑

Narrative Coherence↑

Reference Consistency↑ Subject Scene Grounding

w/o Mean 3.84 0.796 0.482 0.452 0.557 w/o Attn Mask 3.72 0.787 0.468 0.414 0.561 w/o STPA RoPE 3.79 0.761 0.425 0.363 ✗

Ours (w/ Ref) 3.86 0.825 0.493 0.456 0.594

temporal dimension.

- • “w/o STPA RoPE”: without the Spatiotemporal PositionAware RoPE, this setting directly concatenates the reference tokens along the temporal dimension and applies the RoPE(t=0,h,w) to each reference.
- • “Ours (w/o Ref)”: this setting is trained using all the proposed designs, and infers multi-shot text-to-video generation without reference input.
- • “Ours (w/ Ref)”: this setting uses the same trained checkpoint as “Ours (w/o Ref)” and infers multi-shot referenceto-video generation.

Since the spatiotemporal-grounded reference injection might facilitate shot transitions, we do not provide reference input to compare with “w/o MS RoPE” setting. It relies only on the variations between per-shot captions to guide shot transitions, and uses the continuous RoPE to all frames of multi-shot videos in the temporal order. This setting cannot implement precise shot transitions by text prompts only, leading to unsatisfactory transition deviation score, as shown in Table 2. Due to the lack of shot transitions, there is almost no change between shots, resulting in higher semantic and scene consistency scores. With the proposed Multi-Shot Narrative RoPE, we can perform shot transition at user-specified timestamps with superior deviation score.

We further evaluate the designs in spatiotemporalgrounded reference injection. In addition to the mentioned metrics in the main paper, we further introduce Aesthetic Score [40] to measure the aesthetic quality of the generated multi-shot videos. “w/o Mean” might cause information loss, showing suboptimal results, as shown in Table 3. the excessively long contexts in “w/o Attn Mask” setting have unnecessary interactions between in-context tokens, leading to weak aesthetic score and reference consistency. “w/o STPA RoPE” cannot designate the specific shot or exact spatiotemporal position where the subjects and backgrounds appear, relying only on text prompts for positioning. It shows poor performance on reference consistency. Taking advantage of the effectiveness of the proposed designs, our method shows best performance on all metrics.

###### B.2. Ablation Study for Training Paradigm

We conduct ablation study for the three-stage training paradigm introduced in Sec 3.4 of the main paper. We

Table 4. Ablation Study for training paradigm. We experiment on multi-shot reference-to-video generation. The 1st/2nd best results of settings are indicated in underline/bold.

Inter-Shot Consistency↑ Reference Consistency↑

Text Align.↑

Semantic Subject Scene Subject Background Grounding I: Multi-Shot+Ref. Injection 0.211 0.671 0.464 0.415 0.454 0.426 0.477

- I: Multi-Shot
- II: Multi-Shot+Ref. Injection

0.219 0.695 0.481 0.433 0.472 0.451 0.578

- I: Ref. Injection
- II: Multi-Shot+Ref. Injection

0.222 0.692 0.484 0.437 0.485 0.454 0.583

- I: Ref. Injection
- II: Multi-Shot+Ref. Injection
- III: Multi-Shot+Subject-Focused Ref. Injection

###### 0.227 0.702 0.495 0.472 0.493 0.456 0.594

first explore the order of multi-shot video generation and reference-to-video generation, then shows the performance of the subject-focused post-training. The first setting involves fintuning the pretrained text-to-video generation model to learn both multi-shot task and reference-to-video task simultaneously. However, because the diffusion loss is computed across all frames to optimize global consistency, this unified training paradigm shows inadequate for effectively learning both tasks. The second setting is first learning multi-shot text-to-video generation, followed by multishot reference-to-video generation, both using the curated multi-shot & multi-reference data. This setting achieves slightly lower subject consistency due to insufficient exposure to diverse subjects during training. Since the construction cost of multi-shot & multi-reference data is relatively high, we first train the model to learn spatiotemporalgrounded reference injection task on single-shot data, and then learn both tasks using the curated multi-shot & multireference data. It achieves better results on most metrics. Furthermore, we introduce subject-focused post-training that guides the model to prioritize subjects requiring higher consistency, which also promotes the modeling of crossshot subject variations.

###### Prompt for Labeling Global Caption

System Instruction: You are an multi-shot video understanding expert that only outputs video captions. Context Information

[Figure 383]

###### User:

Task Overview: Your task is to analyze the number of subjects that appear in this multi-shot video, and describe the appearance of each subject in the video with one sentence. And describe the video scene roughly.

###### Task Requirements:

- 1. People, vehicles, animals, motor vehicles, food, and other independent objects are all subjects that can be described.
- 2. Subjects who appear only a few times and are not important to the storyline should be omitted.
- 3. Note that the video might have multiple shots, describe the same person no more than once.
- 4. Describe no more than four subjects based on importance.
- 5. Do not describe subjects that are too far away or too small.
- 6. Use no more than 20 words per description for each subject.

Expected Output Format: "Subject 1: A young man with blonde hair, wearing a dark jacket over a light-colored shirt; Subject 2: A yellow dog with a big mouth; Subject 3: A brand-new white car features bright headlights and graceful curves. The whole scene takes place in the parking lot.”

###### Input Multi-Shot Video: These are the frames from the video:{sampled_frames}.

###### Prompt for Labeling Per-Shot Caption

System Instruction: You are an video understanding expert that only outputs video captions based on the input story and video. Context Information

[Figure 384]

###### User:

Task Overview: Your task is to describe the video content in terms of the subjects' expression and actions, scene background, and camera movement in a single paragraph, based on the subject descriptions in the story setting '{global_caption}'. The description should not exceed 80 words.

###### Task Requirements:

- 1. Analyze which subjects in the story setting are present in the current video. And use only existing subject numbers in story setting (e.g., 'Subject 1') to denote the visible subject. Some subjects in the story may not appear in the video.
- 2. Do not describe the subject's appearance. Focus on subjects' expression and actions, scene background, and camera movement.
- 3. First describe the subject facing the camera, then describe the other subjects. When describing camera position, specify which subjects are facing to the camera.
- 4. Subjects who appear only a few times and are not important to the storyline should be omitted.
- 5. Do not describe subjects that are far away or too small or too blurry.
- 6. Do not create new subject number.
- 7. Do not repeat the content in the story setting.

Expected Output Format: "Subject 1 is walking through a dense forest, carrying a large plastic bag. Subject 2 is holding a gun, following behind Subject 1. Subject 3 is running across the ground, with its paws rhythmically hitting the surface. Subject 1 walks slowly and occasionally bends down to pick up items from the ground. The forest is filled with tall, slender trees, and the ground is covered with a mix of grass and fallen leaves. The camera follows them from behind, maintaining a steady and consistent view of their movements. The camera view is a medium shot, capturing the subjects and the surrounding forest. The camera movement is smooth and follows the man's path, maintaining a steady and consistent view of his actions."

###### Input Each Shot Video: These are the frames from the video:{sampled_frames}.

- Figure 6. Prompts of labeling global caption and per-shot captions. We first label the global caption by sampling frames from the input multi-shot video. Then we label the per-shot caption one by one.

###### Global Caption:

- Subject 1: A woman with long brown hair, wearing a purple top and a dark red leather jacket;
- Subject 2: A woman with long blonde hair, wearing a vibrant red dress with a black pattern;
- Subject 3: A woman with dark hair in a bun, wearing a black sleeveless dress and black boots. The whole scene takes place in a modern apartment living room.

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Shot1: Subject 2 is holding a glass of wine, smiling and talking to Subject 1, then drinks from her glass. Subject 1 is pouring wine into a glass, then holds her glass, looking at Subject 2 and adjusting her leg. They are seated on a grey couch in a modern apartment living room with a brick wall and a yellow vase in the background. The camera is static, providing a medium shot of the two subjects.

Shot2: Subject 1 is seen in profile, looking to the right, and occasionally sips from her wine glass. Subject 2 is seated next to Subject 1, holding two wine glasses, and looks towards Subject 1 with an engaged expression, appearing to listen or react. The scene takes place in a modern apartment living room with a brick wall visible in the background. The camera remains static, providing a medium shot of the two women.

Shot3: Subject 3 stands holding a wine

Shot4: Subject 2 is smiling while pouring wine from a bottle into a glass. Subject 1, holding a wine glass, watches attentively. They are seated on a grey couch in a modern apartment living room with a brick wall background. The camera remains static, capturing a medium shot of the two subjects.

Shot5: Subject 1 is seated on a couch in a modern apartment living room, initially smiling broadly and laughing. She then tilts her head back, looking upwards with an amused expression. The background features a brick wall with a shelf holding a yellow vase and a statue. The camera remains static, providing a medium close-up of Subject 1.

glass with a serious expression, then bends down before sitting in a chair, gesturing animatedly while speaking.

Subject 1 and Subject 2 are seated, holding wine glasses, mostly seen from behind. The modern apartment living room features a kitchen area, wall art, and a patterned rug. The camera maintains a medium, mostly static shot, with a slight pan.

- Figure 7. Multi-shot video data example. By employing Gemini-2.5 [10] with the carefully-designed prompts as shown in Fig. 6, the labeled subjects could be consistent in global and per-shot captions.

###### Prompt for Merge Subjects

System Instruction: You are an subject image matching expert that only outputs JSON format. Context Information

[Figure 390]

###### User:

Task Overview: Your task is to group these '{frame_num}' images, The image names are listed as '{all_shot_all_id_name_list}'. You need to place image name with **the same identity or similar appearance** into the same group. The output group name is named as "new_id_0", "new_id_1"...

###### Task Requirements:

- 1. These images might contain diverse categories, including people, vehicles, animals, food, and other subjects.
- 2. The images from the same subject might be shot from different angles. For example, you may see a person's front view and back view.
- 3. Each image could only be assigned to one group.
- 4. If the number of groups exceeds 4, only output the first 4 clearest subjects.
- 5. Only return the json format and strictly follow the template below.

Expected Output Format: {

- "new_id_0": ["shot_0-id_0", "shot_1-id_2"],
- "new_id_1": ["shot_0-id_1", "shot_1-id_1", "shot_2-id_3"],
- "new_id_2": ["shot_0-id_2", "shot_1-id_0", "shot_2-id_5"],
- "new_id_3": ["shot_3-id_0"] }

###### Input Subject Images from All Shots: These are the subject images:{subject_images}.

Figure 8. By employing Gemini-2.5 [10] to group the subject images, we obtain complete multi-shot tracking results.

###### Narrative Coherence Metric

System Instruction: You are a professional AI video evaluation assistant. Context Information

[Figure 391]

###### User:

Your task is to strictly evaluate the continuity and consistency of an AI-generated Multi-shot video. This video should depict the same scene, just shown from different camera angles (shot changes). The input will include:

- 1. Global Caption: An overall description of the entire video scene.
- 2. Multiple sets of visual and text inputs: The video is divided by shots. For each shot, you will receive:

- * Chronologically ordered sampled frames (at least 1 per shot).
- * A description for that shot (Shot Caption).

Your evaluation includes four core principles. You must evaluate the consistency of the entire video (across all shots) based on the following four criteria:

###### 1. Scene Consistency:

- * Objective: Evaluate whether all shots take place in the same scene.
- * Checkpoints: Background environment, object placement, lighting conditions, and overall atmosphere.
- * Judgment Criteria:

* True: The scene, lighting, and atmosphere remain consistent across all shots, clearly indicating the same location and time.

- * False: The background, lighting, or atmosphere changes abruptly and illogically (e.g., suddenly jumping from indoors to outdoors, or from day to night).

###### 2. Subject Consistency:

- * Objective: Evaluate whether the core subjects (people, animals, or key objects) in the video remain consistent after shot changes.
- * Checkpoint (Identity): Is it the same subject after the cut? (e.g., the same person, the same dog).
- * Checkpoint (Appearance): Does the subject's appearance remain unchanged? (e.g., the same person's clothes, hairstyle, accessories).
- * Judgment Criteria:
- * True: The subject's identity and appearance remain consistent across all shots.
- * False: The subject's identity changes (A becomes B), or the subject's appearance (e.g., clothing color) changes illogically after a cut. 3. Action Coherence:
- * Objective: (Only applies to dynamic subjects, like people or animals) Evaluate whether the subject's actions are logically coherent in time across shot changes.

- * Checkpoint: Is the action in the new shot a reasonable continuation of the action from the previous shot?
- * Judgment Criteria:

- * True: The action in the next shot is a reasonable continuation of the action from the previous shot (e.g., Shot 1 shows a hand halfway raised, Shot 2 shows the hand fully raised).
- * False: The action is reset (e.g., Shot 1 shows a hand raised, Shot 2 shows the hand back at the starting position; a person who was running is suddenly standing still), or it jumps to a completely unrelated action, breaking the temporal sequence.

###### 4. Spatial Consistency:

- * Objective: Evaluate whether the relative spatial layout of subjects and their environment remains reasonable after a shot change.
- * Checkpoint (Relative Position): If "A is to the left of B" in Shot 1, does this relative relationship hold in a close-up in Shot 2?
- * Checkpoint (180-Degree Axis): Do the shot changes follow basic cinematic spatial logic (e.g., not "crossing the axis," which could confuse the audience about spatial

relationships)?

* Judgment Criteria:

- * True: Spatial relationships remain consistent (e.g., A is on B's left in Shot 1, and this relative position is maintained in the close-up).
- * False: Relative positions are disordered (e.g., A suddenly teleports from B's left to B's right), or the shot change causes complete spatial disorientation.

Strict Output Format: You will receive four evaluation questions. You must and only answer 'True' or 'False' for each question. Do not add any explanations, justifications, or extra text. Your final response must strictly adhere to this format: Scene Consistency: True/False Subject Consistency: True/False Action Coherence: True/False Spatial Consistency: True/False

Input: Global Caption: {global_caption}

- --- SHOT 1 --{keyframe_images_from_shot_1}

- Shot 1 Caption:{shot_1_caption}

--- SHOT 2 --{keyframe_images_from_shot_2}

- Shot 2 Caption:{shot_2_caption}

……

--- Evaluation Questions --Based on all provided shots and descriptions, please answer the following questions with only 'True' or 'False’.

- Q1: Scene Consistency: The background, lighting, and atmosphere remain consistent across all shots, indicating they are in the same location and time.
- Q2: Subject Consistency: The subjects (people, animals, key objects) in the video maintain the same identity and appearance (e.g., same clothes, features) across all shots.
- Q3: Action Coherence: The actions of dynamic subjects show logical continuity after shot changes, rather than suddenly resetting or jumping.
- Q4: Spatial Consistency: The relative spatial relationships between subjects (e.g., A is to the left of B) and the layout with the environment remain reasonable and non-confusing after shot changes (e.g., not violating the 180-degree axis principle).

Figure 9. We require Gemini-2.5 [10] to strictly adhere to cinematic narrative logic and scrutinize cross-shot content across four core dimensions: Scene Consistency, Subject Consistency, Action Coherence, and Spatial Consistency.

