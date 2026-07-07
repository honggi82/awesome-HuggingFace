# arXiv:2504.02542v3[cs.CV]7Apr2025

## Audio-visual Controlled Video Diffusion with Masked Selective State Spaces Modeling for Natural Talking Head Generation

Fa-Ting Hong1,2 Zunnan Xu2,3 Zixiang Zhou2 Jun Zhou2 Xiu Li3 Qin Lin2 Qinglin Lu2 Dan Xu1,

1HKUST 2Tencent 3Tsinghua University

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

|[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]|
|---|

Audio Source

|[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|

Generated Results

|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]|
|---|

Motion Source

|[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>Only Audio-driven Generation<br><br>[Figure 45]|
|---|

|[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>Only Motion-driven Generation|
|---|

Generated Results

Driven Signal

Figure 1. In this work, we aim to develop a framework that not only generates videos driven by multiple signals without causing control conflicts in the facial region (first three rows) but also supports video generation driven by a single signal (last two rows).

#### Abstract

Talking head synthesis is vital for virtual avatars and human-computer interaction. However, most existing methods are typically limited to accepting control from a single primary modality, restricting their practical utility. To this end, we introduce ACTalker, an end-to-end video diffusion framework that supports both multi-signals control and single-signal control for talking head video generation. For multiple control, we design a parallel mamba structure with multiple branches, each utilizing a separate driving signal to control specific facial regions. A gate mechanism is applied across all branches, providing flexible control over video generation. To ensure natural coordination of the controlled video both temporally and spa-

tially, we employ the mamba structure, which enables driving signals to manipulate feature tokens across both dimensions in each branch. Additionally, we introduce a maskdrop strategy that allows each driving signal to independently control its corresponding facial region within the mamba structure, preventing control conflicts. Experimental results demonstrate that our method produces naturallooking facial videos driven by diverse signals and that the mamba layer seamlessly integrates multiple driving modalities without conflict. The project website can be found at HERE.

#### 1. Introduction

Talking head generation [4, 23–25, 32, 33, 42, 59, 66, 69, 70] aims to create realistic portrait videos driven by specific

input signals. Audio and facial motion are the two primary driving signals for the talking head generation task. In this work, we aim to develop a framework capable of generating portrait videos with either single signal control or simultaneous control of both signals.

Most existing methods typically use a single primary signal to control video generation. They either use audio to control lip movements [15, 27, 29, 38, 47, 69], or rely on facial motion to govern overall facial dynamics [23–25, 42]. Furthermore, some studies [8, 60] have focused on developing unified frameworks that support various control signals for video generation. However, they still allow only one signal to drive the generation at a time during inference.

Therefore, generating a portrait video driven by both audio and facial motion remains a significant challenge. Two critical issues must be addressed for effective multi-control: 1) Control conflicts. Audio signals usually have a strong influence on the mouth region and slightly affect the expression of the face, while the facial motion signals can accurately control the facial expression. When both signals are applied simultaneously without resolving their conflicts, the resulting facial expression tends to favor the strongest one. And when signals are applied in a sequential manner, the model may prioritize the more recent one, especially if the signals are in conflict or affect overlapping facial areas. Solving control conflicts is difficult because it requires balancing and blending these two distinct types of signals—one that controls the lower face (mouth) and the other that governs the entire facial expression—without allowing one to dominate the other. 2) Control signals aggregation. Current video diffusion models [29, 46, 56] typically use attention modules [49] to integrate control signals with intermediate features along the temporal and spatial dimensions separately. This separate processing can miss the interactions between temporal and spatial dimensions, leading to less coherent transitions and spatial inconsistencies. Moreover, when control signals are integrated with flattened spatio-temporal features, the attention map becomes extremely large due to the high number of tokens, especially for longer videos. Thus, finding an efficient way to combine these signals both temporally and spatially remains a critical challenge.

To address the challenges outlined above, we propose the Audio-visual Controlled Video Diffusion model, coined as ACTalker, an end-to-end framework that integrates spatialtemporal features with multiple control signals for photorealistic and expressive talking head generation. To enable the control signals to interact with intermediate video features in both the temporal and spatial dimensions simultaneously, we introduce a selective state-space model (SSM) to aggregate the flattened temporal-spatial feature tokens with the control signals, providing a more computationally efficient alternative to the attention mechanism [49]. Furthermore, to facilitate learning, we employ a mask-drop strat-

egy that discards irrelevant feature tokens outside the control regions, enhancing the effectiveness of the driving signals and improving the generated content within the control regions. Importantly, each driving signal is responsible only for the facial regions indicated by a manually specified mask, addressing the control conflict issues among audio and visual control. The SSM structure and mask-drop strategy together form the Mask-SSM unit in our framework, which handles facial control with a single signal in specific regions. To enable control by multiple signals, we design a parallel-control mamba layer (PCM) consisting of multiple parallel Mask-SSMs. The PCM layer aggregates intermediate features with each driving signal in a spatio-temporal manner within a single branch. To allow simultaneous control by both audio and facial motion while maintaining the flexibility to control by a single signal when needed, we introduce a gating mechanism in each branch, which is randomly set to open during training. This provides flexible control over the generated video, as the gate can be opened or closed during inference, enabling manipulation based on any chosen signals.

We conduct extensive experiments and ablation studies to validate the effectiveness of our proposed method. The experimental results show that our approach not only outperforms existing methods in single-signals control talking head video generation, but also resolves the condition conflict problem to achieve multiple signals control. Our ablation studies demonstrate that our designed mamba structure effectively integrates multiple driving signals with different feature tokens across distinct facial regions, enabling finegrained signals control without conflicts. Our contributions can be summarized as follows:

- • We propose the audio-visual controlled video diffusion model for talking head generation, which enables seamless and simultaneous control of generated videos using both audio and fine-grained facial motion signals, leading to more realistic and expressive outputs.
- • We introduce the parallel-control mamba layer (PCM), which effectively coordinates multiple driving signals without conflicts, ensuring smooth integration of audio and facial motion signals. Additionally, we incorporate a mask-drop strategy that directs the model’s focus to the relevant facial regions for each control signal, improving both the quality and computational efficiency of the generated video.
- • We perform extensive experiments, including evaluations on challenging datasets, demonstrating that our method generates natural-looking talking head videos with precise control over multiple signals, achieving superior results in multiple signals video synthesis.

#### 2. Related Work

Talking Head Generation. Talking head generation has been a longstanding challenge in the fields of computer vi-

Audio Embedding

Audio Mask

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Whisper

[Figure 63]

[Figure 64]

Identity Encoder

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

VAE Encoder

| | |
|---|---|
| | |

Parallel-ControlMamba

Parallel-ControlMamba

TemporalConvolution

TemporalConvolution

SpatialConvolution

TemporalAttention

TemporalAttention

Mask-SSM

Mask-SSM

SpatialAttention

SpatialAttention

[Figure 83]

VAEDecoder

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

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

...

[Figure 116]

Noise Location

| | |
|---|---|
| | |

Guider

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

[Figure 149]

| | | |
|---|---|---|
| | | |

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Facial motion Encoder

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Motion Mask

Motion Embedding

- Figure 2. Illustration of our ACTalker framework. ACTalker takes multiple signals inputs (i.e., audio and visual facial motion) to drive the generation of talking head videos. In addition to the standard layers (e.g., spatial convolution, temporal convolution, spatial attention, and temporal attention) in the stable video diffusion model, we introduce a parallel-control mamba layer to harness the power of multiple signals control. Audio and facial motion signals are fed into this parallel-control mamba layer, along with their corresponding masks, which indicates the regions to focus on for manipulation.

sion and graphics. Recent advancements in the field of talking head generation can be divided into two subcategories: non-diffusion-based and diffusion-based methods. Non-diffusion-based methods [19, 28, 33, 64] are known for their ability to achieve realistic facial animations and fidelity to motion. Some expression-driven methods [23– 25] employ Taylor approximation to estimate the motion flow between two face and then warping the source image. Some audio-driven talking head methods [19, 63] map the audio to the spatial expression landmarks and then control the facial expression and lip movement by audio following the pipeline of expression-driven methods [67]. With the development of diffusion models, recent works [2, 29, 30, 36, 47, 52, 56, 57] have adopted stable diffusion [40] and motion modules [17] for talking head generation within a two-stage training paradigm. Follow-YourEmoji [36] leverages landmarks as motion representations to guide video generation. X-Portrait [55] first constructs cross-identity training pairs using a pretrained talking head model, and then employs a ControlNet-style network to predict the results. Hallo [56] introduces a hierarchical mask that enables audio-driven control of portrait videos.

However, most previous methods only allow singlesignal control at a time. Therefore, we propose a novel framework based on the mamba structure that can generate videos driven by either multiple signals or a single signal at a time. Moreover, our architectural advancements enhance the model’s ability to simultaneously learn spatial and temporal relationships within the mamba structure.

Selective State Space Models. State Space Models (SSMs) have recently been proposed to integrate deep learning for state space transformation [9, 12]. Inspired by continuous state space models in control systems, SSMs, when combined with HiPPO initialization [11], show great potential in addressing long-range dependency issues, as demonstrated in LSSL [13]. However, the computational and memory demands of state representation make LSSL impractical for real-world applications. To address this, S4 [12] introduces parameter normalization into a diagonal structure. This has led to the development of various structured SSMs with different configurations, such as complex-diagonal structures [14, 18], MIMO support [44], diagonal-plus-low-rank decomposition [20], and selection mechanisms [10], which have been integrated into large-scale frameworks [35, 37]. Recently, SSMs have been applied to language understanding [35, 37, 53], content-based reasoning [10], motion generation [58] and one-dimensional image classification at the pixel level [12], leading to significant improvements.

In this work, we integrate SSMs into 2D talking head generation to address the challenge of aggregating features with control signals. By applying SSMs, our framework efficiently integrates contextual information from audio, video, and spatiotemporal features, demonstrating the potential of ACTalker for talking heads.

#### 3. Methodology

In this work, we aim to develop a novel video diffusion model for one-shot talking head video generation driven

[Figure 161]

[Figure 162]

Audio Embedding Audio Mask

| | |
|---|---|
| | |

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Identity Embedding

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Audio Branch

[Figure 175]

- G1
- G2

[Figure 176]

Mask-SSM

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

[Figure 195]

[Figure 196]

[Figure 197]

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

[Figure 210]

[Figure 211]

[Figure 212]

Temporal-Spatial Flatten

[Figure 213]

[Figure 214]

Norm f✓

### f✓

1 Concat

###### G1,G2 2 {0,1} ^ G1 + G2 > 0

…

2

[Figure 215]

[Figure 216]

[Figure 217]

Mask-SSM

z 2 Rb⇥(f⇥h⇥w)⇥c

Visual Branch

| | |
|---|---|
| | |

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Motion Embedding Motion Mask

- Figure 3. Illustration of parallel-control mamba layer. There are two parallel branches in this layer, one for audio control and the other is for expression control. We utilize a gate in each branch to control the accessing of control signal during training. During inference, we can manually modify the statue of gates to enable single signal control or multiple signals control.

by multiple signals. We propose an audio-visual controlled video diffusion model that provides flexible control over the availability of each driving signal, enabling effective multiple signals control of the generated video.

- 3.1. Overview In this work, we adopt the stable video diffusion model (SVD) [1] as our codebase. As shown in Figure 2, in addition to the regular inputs of the source image and pose image, our ACTalker also accepts multiple driving signals, such as audio and visual expressions, to guide video generation. Given a source image Is, a face mask image Mface,

facial motion sequences {Iiexp}Ni=1, and an audio segment A, we first use a VAE encoder to encode the source image into latent space, which is then concatenated with noise latent. Next, we use Whisper [39] to extract the audio embedding ea from the audio A. Similarly, we utilize a pre-trained motion encoder [57] to extract an implicit facial motion embedding emtn from a sequence of face images, and an identity encoder [7] to obtain the identity embedding eid from the source image Is.

In addition to the regular layers in SVD, such as spatial convolution, temporal convolution, appearance attention, and temporal attention, we design a parallel-control mamba layer (PCM) in each block to enable multi-signal control. The PCM layer consists of multiple branches, each containing a Mask-SSM unit. Specifically, in each branch, the Mask-SSM takes one driven signal and its corresponding mask as input to manipulate the selected spatial-temporal feature tokens by aggregating them in the SSM structure, achieving facial control. Additionally, a gate mechanism is used in the PCM to manage the control of each driven signal. Each branch maintains a gate to decide the availability of the corresponding branch.

- 3.2. Parallel-control Mamba Layer In this work, we aim to develop a method for generating portrait videos controlled by either multiple signals with-

out conflict or a single signal. To this end, we propose a novel parallel-control Mamba layer (Figure 3) that leverages driven signals to manipulate the temporal-spatial features via the Mamba structure, achieving fine-grained control over facial synthesis. The layer consists of two primary branches, each controlling different facial regions with different conditioning signals: audio and facial motion. Additionally, we designed a gate mechanism to achieve flexible control by controlling the activation of each branch.

Identity Preservation. As illustrated in Figure 3, to enable the driving signal to influence the intermediate noise feature both spatially and temporally, we first flatten the spatiotemporal intermediate feature across its spatial and temporal dimensions. This produces a flattened feature z ∈ Rb×(f×h×w)×c, where f represents the number of frames, and h and w are the height and width of the original spatiotemporal feature. Furthermore, to preserve identity during face manipulation driven by the signal, we also aggregate the identity embedding eid with the noise feature:

1×c, (1) where fθ

###### z′ = Concat(eid,fθ

###### (z)), z′ ∈ Rb×n

1

, parameterized by θ1, is an MLP that transforms the noise feature to integrate with the identity embedding.

1

Multiple Signals Control with Gate Mechanism. To enable multiple signals control, we feed the concatenated feature z′ into parallel branches, where each branch is responsible for controlling a specific facial region. As shown in Figure 3, we have two branches to handle audio-driven and motion-driven control, respectively. In our setup, we expect our model to generate portrait videos driven by both signals simultaneously, while still maintaining the capability to be driven by a single signals. To achieve this, we randomly set up gate variables in each branch (G1 and G2) to control the driving mode during training. For the gates in both branches, we impose the constraint:

###### G1,G2 ∈ {0,1} ∧ G1 + G2 > 0, (2)

[Figure 224]

| | |
|---|---|
| | |

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

SSM

[Figure 231]

[Figure 232]

Audio Embedding

[Figure 233]

[Figure 234]

[Figure 235]

⇥c2

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

###### …

[Figure 241]

⇥bnR2z1

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

Mask Paste

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Mask Drop

…

…

z0 2 Rb⇥n

1⇥c

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

…

…

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Brocadcasting

zssmin 2 Rb⇥n

2⇥c

[Figure 263]

Flatten

…

Audio Mask

[Figure 264]

- Figure 4. The illustrating of the Mask-SSM in audio branch of parallel-control mamba layer. The visual branch is the same but replace with the motion embedding and motion mask

There are three possible configurations under this constraint: {G1 = 0,G2 = 1}, {G1 = 1,G2 = 0}, and {G1 = 1,G2 = 1}. The first two configurations mean that only one signal is used for control during training, while the last configuration indicates that both signals are used simultaneously. During our training, we randomly select one of these three gate statuses. Therefore, our model is able to generate portrait video under the single signals or both audio-visual signals control.

Multi-control Aggregation. For the outputs of each branch (z1 and z2, with the branch structure detailed later), we concatenate them and apply normalization to the concatenated results in order to improve training stability:

###### o1 = Norm(z1 + z2). (3)

Next, we apply a residual connection between the aggregated output oagg and the original flattened noise feature z:

(o1) + z, (4) where fθ

o2 = fθ

2

is an MLP that transforms the aggregated feature o1, parameterized by θ2. Finally, the entire parallel-control mamba layer outputs o2, which is passed to the next block in our framework.

2

The audio and motion-manipulated features are then aggregated and pass the control signal information throughout the framework without conflicts.

###### 3.3. Mask-SSM

As shown in Figure 3, in each branch, we design a Mask State Space Model (Mask-SSM) to process the input spatiotemporal noise feature and control signal. Since we flatten the noise feature volume along both the spatial and temporal dimensions, the number of tokens increases dramatically (frames × width × height) becomes much larger in the shallow layers). To efficiently fuse each token with the driving signal, we adopt the state space model in our framework. To solve the control conflict problem, we design a specific mask (see Supplementary Material) for each driven signal to indicate their control regions. Based on that mask, we design a mask-drop strategy to not only reduce the number of noise feature tokens by dropping irrelevant ones but

also distinguish the tokens across spatial and temporal dimensions that need to be manipulated by the corresponding signals, as indicated by the input mask. The mask-drop strategy mainly consists of two steps: Mask Drop and Mask Paste. Each branch in PCM shares the same architecture and specifies the control region of the driven signals using different masks, i.e., the audio mask and the motion mask. In this section, we take the audio branch as an example to demonstrate the details. Mask Drop. As shown in Figure 4, given an audio mask Maudio, where the control region is set to 1 and all other regions are set to 0, we first flatten the mask and broadcast it to match the shape of the input noise feature z′. Then, we apply this flattened mask to drop the noise features (we omit the identity tokens concatenated in Eq. 1 for simplicity). This process is expressed as:

zssmin = D(z′,Maudio), (5) where zssmin ∈ Rb×n

1×c, and n1 > n2. Here, D represents the drop operation, which removes the tokens where the corresponding position in the audio mask Maudio is zero.

2×c, z′ ∈ Rb×n

Mask Paste. After obtaining the masked tokens zssmin , we concatenate them with the driving signal, i.e., the audio embedding, and then pass the concatenated result into an SSM unit to enable each token to interact with the driving signal:

zssmout = SSM(Concat(zssmin ,ea)). (6) Consequently, we drop the audio and identity tokens in the result zssmout . The resulting tokens only maintain the facial information of the controlled region after processing by the driven signal. To cooperate with other regions, such as the background, we paste the audio-aggregated tokens back to the original noise feature z according to the mask M:

z1 ← z[Maudio == 1] = zssmout . (7) Therefore, we replace the tokens of the control region in z with the aggregated feature tokens zssmout to obtain the audio-aggregated feature z1. We can also get the motionaggregated feature z2 in the same way but in a different Mask-SSM branch.

In this way, Mask-SSM can resolve control conflicts by distributing different tokens to each driven signal using the mask-drop strategy. By applying the mask-drop strategy in each control branch, we can aggregate features spatiotemporally with reduced computational complexity through the Mamba structure while improving the model’s focus on signal-specific regions of the face, leading to more accurate control of video generation.

###### 3.4. Training and Inference

Training. To train our video diffusion model, we apply the general training objective of the video diffusion model:

L = Et,z,ϵ[||ϵ − ϵθ(C,z,t)||], (8)

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

[Figure 276]

EDTalkerOursEchominicMemoGroundTruthVExpress

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

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

- Figure 5. Comparison of different methods for audio-driven talking head generation. Our method can produce more natural and accurate lip-synced videos. Due to the page limitation, the results of SadTalker [63] and Hallo [56] are reported in Supplementary Material

where z denotes the latent embedding of the training sample, ϵ and ϵθ are the ground truth noise at the corresponding timestep t and the predicted noise by our ACTalker, respectively. C is the condition set, which includes the audio embedding, motion embedding, identity embedding, and pose location. During training, we randomly select one of the three gate configurations in our parallel-control mamba layer to enable flexible controllability. In the step of training the model with single-signal control, we repurpose the pose mask Ip as an audio/expression mask, allowing the driving signal to control the entire face.

Inference. During inference, we manually set the gate status to enable different types of control (audio-only, expression-only, and audio-expression). Our ACTalker is capable of generating videos of arbitrary length within memory constraints, given reference images and audio or facial motion driving signals. We also apply classifier-free guidance (CFG) [22] to achieve better results.

#### 4. Experiments

In this section, we present both quantitative and qualitative experiments to validate the effectiveness of ACTalker. More implementation details, additional results and video demonstrations can be found in the Supplementary Materials. We strongly recommend watching the video demos.

###### 4.1. Experimental Settings

Dataset. We use publicly available datasets, such as HDTF [65], VFHQ [54], VoxCeleb2 [5], CelebV-Text [61], along with self-collected videos, to create a diverse training

dataset. Since our method is capable of both audio-driven talking head generation and expression-driven face reenactment, we conduct comparisons on these two tasks. For audio-driven talking head generation, we follow the settings of Loopy [29], sampling 100 videos from CelebV-HQ [71] and RAVDESS (Kaggle). Additionally, we test our face reenactment capabilities using the VFHQ dataset [54].

###### 4.2. Quantitative and Qualitative Analysis

We conduct a quantitative and qualitative comparison with other methods on audio-driven talking head generation and face reenactment tasks. During inference, we set the gate value to either 0 or 1 to control whether the video is generated using the corresponding signal. The quantitative results are reported in Table. 1 and Table. 2. Also, we visualize the qualitative comparison in Figure. 5 and Figure. 6.

Audio-driven Talking Head Results. We first compare our method with other audio-driven talking head generation methods. The results reported in Table 1 and Table 1 strongly demonstrate the superiority of our approach compared with existing works. Our method achieves the best Sync-C and Sync-D scores on the CelebV-HD dataset (5.317 for Sync-C and 7.869 for Sync-D), verifying that it can produce audio-synchronized talking head videos. In terms of video quality, our method also shows significant improvement. For example, our approach obtains an FVD-Inc score of 232.374, outperforming the second-best method Memo [68] by roughly 32 points. These results demonstrate that our specific design in the stable video diffusion model brings notable benefits. Additionally, Figure 5 visualizes several samples of audio-driven talking head gen-

###### Model Sync-C↑ Sync-D ↓ FVD-Res ↓ FVD-Inc ↓ FID ↓ Smooth ↑ Sync-C↑ Sync-D ↓ FVD-Res ↓ FVD-Inc ↓ FID ↓ Smooth ↑

SadTalker[63] 3.814 8.824 18.484 352.296 51.804 0.9963 3.899 7.895 16.642 264.065 44.965 0.9953 Hallo[56] 4.316 9.020 13.317 342.965 37.400 0.9946 3.963 8.125 6.888 266.920 23.157 0.9941 VExpress[50] 3.612 9.165 37.657 539.920 58.427 0.9959 4.888 7.898 14.950 517.880 26.753 0.9954 EDTalk [45] 5.124 8.438 16.723 430.906 50.428 0.9972 4.759 8.375 14.114 477.147 50.135 0.9954 EchoMimic [2] 2.989 10.188 16.897 366.007 45.489 0.9938 3.239 9.411 46.038 450.798 41.357 0.9923 Memo [68] 3.958 9.118 7.992 264.596 31.134 0.9954 5.093 7.854 5.098 194.570 18.837 0.9945 Ours (Only Audio) 5.317 7.869 7.328 232.374 30.721 0.9978 5.334 7.569 4.754 193.120 16.730 0.9955 Ours (Audio-Visual) 5.737 7.510 7.074 230.125 29.977 0.9979 5.511 7.311 4.574 190.125 15.977 0.9955

Table 1. Audio-driven comparison of different methods on Celebv-HQ dataset (left) and RAVDESS dataset (right).

[Figure 337]

[Figure 338]

Reference Driving Ours X-Portrait Follow-Your-Emoji AniPortrait LivePortrait

Figure 6. Comparison of different methods on VFHQ. Self reenactment (first row) and cross reenactment (last row).

eration. It can be observed that our results exhibit accurate lip motion and fewer artifacts compared with other methods. These findings confirm that our mamba structure design is beneficial for lip-sync by directly manipulating the selected tokens.

Face Reenactment. In addition to audio-driven talking head video generation, our framework is also capable of performing expression-driven talking head video generation, i.e., face reenactment. Existing methods [23, 24] typically evaluate face reenactment in two scenarios: selfreenactment and cross-reenactment. In self-reenactment, the driving video and reference share the same identity, whereas in cross-reenactment they have different identities. As shown in Table 2, our expression-driven method achieves superior results compared with existing state-ofthe-art approaches. Specifically, our method outperforms X-Portrait [55] by 9% in expression similarity during cross reenactment, while also maintaining the best ID similarity (8.64 for cross reenactment). These results validate that our mamba structure effectively manipulates facial region tokens to perform accurate facial expression animation. We further visualize sample outputs in Figure 6. Compared with other methods, our approach captures more subtle micro-motions—such as the mouth movements in the first two self-reenactment samples—and produces more precise expression animations in the last two cross-reenactment samples. It verifies that our designed Mask-SSM effectively enhances the generated content in the controlled region.

###### 4.3. Ablation Study

In this section, we evaluate each design in our framework to verify its effectiveness. The results are reported in Table 3,

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

AudioSourceOursOursw/oMDOursOursw/oMDMotionSource

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

Figure 7. Visualization of multiple signals control. Our generated video accurately replicates the lip movements driven by the audio source and captures the head motion—particularly the eye movements and pose—as guided by the motion source. Once we remove the masks in both Mask-SSMs and generate the video using multiple driving signals, the motion source can also affect the mouth movement (“Ours w/o MD”), causing a control conflict.

Figure 7, and Figure 8.

Multiple Signals Control. Benefiting from our gating mechanism and Mask-SSM, our framework can generate

Model LMD(×10−2) ↓ FID ↓ FVD-Inc ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Pose Distance ↓ Expression Similarity ↑ ID Similarity(×10−1)↑ Smooth (×10−2)↑

LivePortrait [16] 0.49 82.69 483.42 40.37 0.92 0.31 26.99 0.38 8.55 99.53 AniPortrait [52] 0.68 81.89 430.27 39.29 0.85 0.36 21.31 0.46 8.50 99.36 FollowYourEmoji [36] 0.65 77.17 417.51 39.67 0.86 0.35 20.94 0.48 8.59 98.99 X-Portrait [55] 0.24 82.92 416.42 39.64 0.92 0.27 20.38 0.48 8.57 99.39 Ours 0.14 75.47 358.82 40.65 0.94 0.24 20.32 0.57 8.64 99.48

Table 2. Comparison of different methods on VFHQ. Self reenactment (left) and cross reenactment (right).

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Oursw/oIDBaselineOursOursw/oMDGroundTruth

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

Figure 8. The visualization of ablation studies driven by audio. Our full method can produce more natural videos.

Model Sync-C↑ Sync-D ↓ FVD-Res ↓ FVD-Inc ↓ FID ↓ Smooth ↑

Baseline 4.592 8.523 16.983 268.512 32.483 0.9967 Ours w/o MD 4.953 8.184 7.456 240.651 31.268 0.9969 Ours w/o ID 5.241 7.748 8.364 247.933 31.170 0.9978 Ours (Only Audio) 5.317 7.869 7.328 232.374 30.721 0.9978 Ours (Audio-Visual) 5.737 7.510 7.074 230.125 29.977 0.9979

Table 3. Ablation studies on Celebv-HQ dataset for audio-driven.

videos driven either by a single signal (audio or expression) or by multiple signals simultaneously. We conduct multiple signals driven experiments as part of our ablation studies. As shown in Figure 7, we generate three samples using the same audio and expression signals. We observe that the generated videos exhibit consistent expressions (e.g., similar eye status) and synchronized mouth movements. Quantitative results further demonstrate that videos driven by both signals yield superior performance compared to those generated using a single signal. These findings confirm that our parallel-control mamba layer effectively enables different signals to control disentangled facial regions, achieving robust multiple signals control.

Mamba Structure. To evaluate the effectiveness of our mamba structure, we integrate it into the Stable Video Diffusion model and construct a baseline by replacing the parallel-control mamba layer with a spatial cross-attention layer. As shown in Table 3 and Figure 8, without our mamba structure, both lip synchronization and overall video quality deteriorate dramatically. These results confirm that our mamba structure effectively captures the core information from the driving signal and broadcasts it across temporal

and spatial dimensions, resulting in more natural portrait video generation.

Mask-Drop and Control Conflict. We employ a maskdrop strategy in our mamba structure not only to reduce the number of processed tokens but also to enhance the focus of the driving signal on the control regions. We conduct an ablation study (labeled “Ours w/o MD” in Table 3 and illustrated in Figure 8) to verify its effectiveness. As shown in Table 3, without the mask-drop strategy, the model is distracted by irrelevant tokens, resulting in a performance drop (the Sync-C score is 4.953 compared to 5.317 with the full method). Moreover, the generated outputs appear less natural without the mask-drop strategy. Also, we show the samples that are driven by both signals in Figure 7. We can observe that, without the mask-drop strategy, the mouth region is affected by the motion signals, which is not what we expected. These results confirm that the mask-drop strategy significantly improves the controllability of the driving signal over the target regions and resolves the control conflict. Identity Embedding in PCM. In our parallel-control mamba layer (PCM), we inject an identity embedding and aggregate it within the Mask-SSM to preserve identity while manipulating the selected tokens. We also perform an ablation study by removing the identity embedding from the PCM (reported as “Ours w/o ID” in Table 3 and Figure 8). As shown in Figure 8, without the identity embedding, some frames fail to maintain the subject’s identity, resulting in poorer quantitative performance. These findings underscore the necessity of identity embedding in our PCM layer.

#### 5. Conclusion

In this work, we introduce the audio-visual controlled video diffusion (ACTalker) model, a novel end-to-end framework for talking head generation that achieves seamless and simultaneous control using both audio and fine-grained expression signals. Our method leverages a parallel-control mamba (PCM) layer to effectively integrate multiple driving modalities without conflict. By incorporating a maskdrop strategy, the model can focus on the relevant facial regions for each control signal, thereby enhancing video quality and preventing control conflicts in the generated videos. Extensive experiments on challenging datasets demonstrate that our approach produces natural-looking talking head videos with precise multiple signals control, achieving superior results compared to existing methods. Ablation studies verify the effectiveness of our mask-drop strategy in enhancing generated content and the gating mechanism in providing flexible control over the video generation process.

#### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 4, 12
- [2] Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, and Chenguang Ma. Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions. arXiv preprint arXiv:2407.08136, 2024. 3, 7
- [3] Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In ACCV Workshops, 2017. 12
- [4] Joon Son Chung, Andrew Senior, Oriol Vinyals, and Andrew Zisserman. Lip reading sentences in the wild. In CVPR,

2017. 1

- [5] Joon Son Chung, Arsha Nagrani, and Andrew Zisserman. Voxceleb2: Deep speaker recognition. arXiv, 2018. 6
- [6] Radek Danˇeˇcek, Michael J Black, and Timo Bolkart. Emoca: Emotion driven monocular face capture and animation. In CVPR, 2022. 12
- [7] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019. 4, 12
- [8] Nikita Drobyshev, Antoni Bigata Casademunt, Konstantinos Vougioukas, Zoe Landgraf, Stavros Petridis, and Maja Pantic. Emoportraits: Emotion-enhanced multimodal one-shot head avatars. In CVPR, 2024. 2
- [9] Daniel Y Fu, Tri Dao, Khaled Kamal Saab, Armin W Thomas, Atri Rudra, and Christopher Re. Hungry hungry hippos: Towards language modeling with state space models. In ICLR, 2021. 3
- [10] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 3
- [11] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher R´e. Hippo: Recurrent memory with optimal polynomial projections. NeurIPS, 2020. 3
- [12] Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces. In ICLR,

2021. 3

- [13] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher R´e. Combining recurrent, convolutional, and continuous-time models with linear state space layers. NeurIPS, 2021. 3
- [14] Albert Gu, Karan Goel, Ankit Gupta, and Christopher R´e. On the parameterization and initialization of diagonal state space models. NeurIPS, 2022. 3
- [15] Jiazhi Guan, Zhanwang Zhang, Hang Zhou, Tianshu Hu, Kaisiyuan Wang, Dongliang He, Haocheng Feng, Jingtuo Liu, Errui Ding, Ziwei Liu, et al. Stylesync: High-fidelity generalized and personalized lip sync in style-based generator. In CVPR, 2023. 2
- [16] Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou Zhong, Yuan Zhang, Pengfei Wan, and Di Zhang. Liveportrait: Efficient portrait animation with stitching and retargeting control. arXiv, 2024. 8

- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [18] Ankit Gupta, Albert Gu, and Jonathan Berant. Diagonal state spaces are as effective as structured state spaces. NeurIPS,

2022. 3

- [19] Siddharth Gururani, Arun Mallya, Ting-Chun Wang, Rafael Valle, and Ming-Yu Liu. Space: Speech-driven portrait animation with controllable expression. In ICCV, 2023. 3
- [20] Ramin Hasani, Mathias Lechner, Tsun-Hsuan Wang, Makram Chahine, Alexander Amini, and Daniela Rus. Liquid structural state-space models. In ICLR, 2022. 3
- [21] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 12
- [22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6
- [23] Fa-Ting Hong and Dan Xu. Implicit identity representation conditioned memory compensation network for talking head video generation. In ICCV, 2023. 1, 2, 3, 7
- [24] Fa-Ting Hong, Longhao Zhang, Li Shen, and Dan Xu. Depth-aware generative adversarial network for talking head video generation. In CVPR, 2022. 7
- [25] Fa-Ting Hong, Li Shen, and Dan Xu. Dagan++: Depthaware generative adversarial network for talking head video generation. arXiv preprint arXiv:2305.06225, 2023. 1, 2, 3
- [26] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 12
- [27] Xinya Ji, Hang Zhou, Kaisiyuan Wang, Wayne Wu, Chen Change Loy, Xun Cao, and Feng Xu. Audio-driven emotional video portraits. In CVPR, 2021. 2
- [28] Xinya Ji, Hang Zhou, Kaisiyuan Wang, Qianyi Wu, Wayne Wu, Feng Xu, and Xun Cao. Eamm: One-shot emotional talking face via audio-based emotion-aware motion model. In SIGGRAPH, 2022. 3
- [29] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. arXiv preprint arXiv:2409.02634, 2024. 2, 3, 6
- [30] Xiaoyu Jin, Zunnan Xu, Mingwen Ou, and Wenming Yang. Alignment is all you need: A training-free augmentation strategy for pose-guided video generation. arXiv preprint arXiv:2408.16506, 2024. 3
- [31] D. P. Kingma and M. Welling. Auto-encoding variational bayes. In ICLR, 2014. 12
- [32] Yukang Lin, Hokit Fung, Jianjin Xu, Zeping Ren, Adela SM Lau, Guosheng Yin, and Xiu Li. Mvportrait: Text-guided motion and emotion control for multi-view vivid portrait animation. arXiv preprint arXiv:2503.19383, 2025. 1
- [33] Yunfei Liu, Lijian Lin, Fei Yu, Changyin Zhou, and Yu Li. Moda: Mapping-once audio-driven portrait animation with dual attentions. In ICCV, 2023. 1, 3

- [34] Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris McClanahan, Esha Uboweja, Michael Hays, Fan Zhang, ChuoLing Chang, Ming Guang Yong, Juhyun Lee, et al. Mediapipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172, 2019. 12
- [35] Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer. Mega: Moving average equipped gated attention. In ICLR, 2022. 3
- [36] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. arXiv preprint arXiv:2406.01900, 2024. 3, 8, 13
- [37] Harsh Mehta, Ankit Gupta, Ashok Cutkosky, and Behnam Neyshabur. Long range language modeling via gated state spaces. In ICLR, 2023. 3
- [38] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In ACM MM, 2020. 2, 12
- [39] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In ICML,

2023. 4

- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [41] Sefik Ilkin Serengil and Alper Ozpinar. Lightface: A hybrid deep face recognition framework. In ASYU, 2020. 12
- [42] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. NeurIPS, 2019. 1, 2
- [43] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 12
- [44] Jimmy TH Smith, Andrew Warrington, and Scott Linderman. Simplified state space layers for sequence modeling. In ICLR, 2022. 3
- [45] Shuai Tan, Bin Ji, Mengxiao Bi, and Ye Pan. Edtalk: Efficient disentanglement for emotional talking head synthesis. In ECCV, 2024. 7
- [46] Shixiang Tang, Yizhou Wang, Lu Chen, Yuan Wang, Sida Peng, Dan Xu, and Wanli Ouyang. Human-centric foundation models: Perception, generation and agentic modeling. arXiv preprint arXiv:2502.08556, 2025. 2
- [47] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In ECCV, 2025. 2, 3
- [48] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 12
- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017. 2

- [50] Cong Wang, Kuan Tian, Jun Zhang, Yonghang Guan, Feng Luo, Fei Shen, Zhiwei Jiang, Qing Gu, Xiao Han, and Wei Yang. V-express: Conditional dropout for progressive training of portrait video generation. arXiv, 2024. 7
- [51] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. TIP, 13(4):600–612, 2004. 12
- [52] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694, 2024. 3, 8, 13
- [53] Yicheng Xiao, Lin Song, Shaoli Huang, Jiangshan Wang, Siyu Song, Yixiao Ge, Xiu Li, and Ying Shan. Grootvl: Tree topology is all you need in state space model. arXiv preprint arXiv:2406.02395, 2024. 3
- [54] Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong, and Ying Shan. Vfhq: A high-quality dataset and benchmark for video face super-resolution. In CVPR, 2022. 6
- [55] You Xie, Hongyi Xu, Guoxian Song, Chao Wang, Yichun Shi, and Linjie Luo. X-portrait: Expressive portrait animation with hierarchical motion attention. In SIGGRAPH,

2024. 3, 7, 8, 14

- [56] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 2, 3, 6, 7
- [57] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang, Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong, and Baining Guo. Vasa-1: Lifelike audio-driven talking faces generated in real time. arXiv preprint arXiv:2404.10667,

2024. 3, 4

- [58] Zunnan Xu, Yukang Lin, Haonan Han, Sicheng Yang, Ronghui Li, Yachao Zhang, and Xiu Li. Mambatalk: Efficient holistic gesture synthesis with selective state space models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3
- [59] Zunnan Xu, Zhentao Yu, Zixiang Zhou, Jun Zhou, Xiaoyu Jin, Fa-Ting Hong, Xiaozhong Ji, Junwei Zhu, Chengfei Cai, Shiyu Tang, Qin Lin, Xiu Li, and Qinglin Lu. Hunyuanportrait: Implicit condition control for enhanced portrait animation. arXiv preprint arXiv:2503.18860, 2025. 1
- [60] Fei Yin, Yong Zhang, Xiaodong Cun, Mingdeng Cao, Yanbo Fan, Xuan Wang, Qingyan Bai, Baoyuan Wu, Jue Wang, and Yujiu Yang. Styleheat: One-shot high-resolution editable talking face generation via pre-trained stylegan. In ECCV,

2022. 2

- [61] Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, and Wayne Wu. CelebV-Text: A large-scale facial text-video dataset. In CVPR, 2023. 6
- [62] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 12
- [63] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In CVPR, 2023. 3, 6, 7
- [64] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker:

- Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In CVPR, 2023. 3
- [65] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with a highresolution audio-visual dataset. In CVPR, 2021. 6
- [66] Shuling Zhao, Fating Hong, Xiaoshui Huang, and Dan Xu. Synergizing motion and appearance: Multi-scale compensatory codebooks for talking head video generation. In CVPR, 2025. 1
- [67] Shuling Zhao, Fa-Ting Hong, Xiaoshui Huang, and Dan Xu. Synergizing motion and appearance: Multi-scale compensatory codebooks for talking head video generation. In CVPR, 2025. 3
- [68] Longtao Zheng, Yifan Zhang, Hanzhong Guo, Jiachun Pan, Zhenxiong Tan, Jiahao Lu, Chuanxin Tang, Bo An, and Shuicheng Yan. Memo: Memory-guided diffusion for expressive talking video generation. arXiv preprint arXiv:2412.04448, 2024. 6, 7
- [69] Hang Zhou, Yasheng Sun, Wayne Wu, Chen Change Loy, Xiaogang Wang, and Ziwei Liu. Pose-controllable talking face generation by implicitly modularized audio-visual representation. In CVPR, 2021. 1, 2
- [70] Jun Zhou, Jiahao Li, Zunnan Xu, Hanhui Li, Yiji Cheng, Fa-Ting Hong, Qin Lin, Qinglin Lu, and Xiaodan Liang. Fireedit: Fine-grained instruction-based image editing via region-aware vision language model. arXiv preprint arXiv:2503.18860, 2025. 1
- [71] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. Celebvhq: A large-scale video facial attributes dataset. In ECCV,

2022. 6

#### A. Experiment Detail

###### A.1. Implementation.

During the training process, we resize all images and videos to 640 × 640. To optimize the framework, we use the AdamW optimizer with a learning rate of 1 × 10−5. The Identity encoder [7] and VAE [31] are kept fixed, with their weights initialized from Stable Video Diffusion [1]. During training, we randomly select the gate states in the parallelcontrol mamba layer and manually set them during inference to enable flexible control.

###### A.2. Metrics

In this work, we evaluate our method and compare it with other approaches using comprehensive quantitative metrics. Our evaluation framework consists of three main categories: (1) audio-visual synchronization, (2) visual quality assessment, and (3) facial motion accuracy. Additionally, we assess temporal smoothness and identity preservation through specialized measures.

Audio-Visual Synchronization. We employ Sync-C (synchronization confidence) and Sync-D (synchronization distance) metrics from wav2lip [38] using a pretrained SyncNet [3]. Sync-C measures the confidence level of lip-audio alignment through classifier outputs, where higher values indicate better synchronization. Sync-D calculates the L2 distance between audio and visual features, with lower values representing superior alignment.

Visual Quality Assessment. We utilize four complementary metrics:

- • PSNR: Peak Signal-to-Noise Ratio quantifies pixel-level fidelity through a logarithmic decibel scale, where higher values reflect better reconstruction accuracy.
- • SSIM [51]: Structural Similarity Index measures structural information preservation between generated and reference frames, ranging from 0 to 1, with higher values indicating better quality.
- • LPIPS [62]: Learned Perceptual Image Patch Similarity evaluates perceptual differences using VGG [43] features:

LPIPS(Igtt ,Igent ) =

l

wl Fl(Igtt ) − Fl(Igent ) 2 (9)

- • Fr´echet Inception Distance (FID) [21]: Measures feature distribution similarity between generated and real images using Inception-v3 features, with lower scores indicating better perceptual quality.
- • Fr´echet Video Distance (FVD) [48]: Assesses temporal coherence through pretrained network features:

FVD = ∥µgen − µgt∥2 + Tr Σgen + Σgt − 2(ΣgenΣgt)1/2

(10)

Facial Motion Accuracy. For expression and pose evaluation:

- • Landmark Mean Distance (LMD): Computes the average L2 distance between facial landmarks [34] of generated and reference frames, with lower values indicating better geometric accuracy.
- • Pose Distance: Measures head pose discrepancies using EMOCA [6]-derived parameters through the mean L1 distance between generated and driving frames.
- • Expression Similarity: Calculates the cosine similarity of expression parameters from EMOCA [6], with higher values indicating better emotional consistency.

Identity Similarity. We employ ArcFace [7] scores to measure identity similarity between generated frames and reference images through deep face recognition features, where higher scores indicate better identity preservation.

Temporal Smoothness. We evaluate motion temporal smoothness by computing the optical flow consistency using VBench metrics [26], where lower variance in motion vectors indicates smoother transitions.

###### A.3. Mask Design

Face Mask Audio Mask Motion Mask

Figure 9. The type of masks we used in our framework.

In our framework, we utilize masks to indicate the control regions of each signal. Three types of masks are used in our framework. As illustrated in Figure 9, the face mask is used to indicate the rough position of the face in the source image. During training, we use RetinaFace [41] to calculate the bounding box for all frames in the ground truth segments and obtain the smallest enclosing rectangle of these bounding boxes. We then draw the face mask based on that rectangle to indicate the facial location in the desired video. Similarly, the audio mask is obtained by detecting the mouth bounding boxes, and the motion mask is generated by using the face mask to minimize the audio mask. During the inference stage, we detect the bounding box of the source image and apply the appropriate extension.

#### B. Visualization B.1. Face reenacment

Figure 10 demonstrates that our approach achieves enhanced precision in replicating portrait motions that align closely with the driving video’s dynamics. For selfreenactment, the results generated by our framework better

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

Reference Driving Ours X-Portrait Follow-Your-Emoji AniPortrait LivePortrait

Figure 10. The results generated by our method under facial motion control.

preserve intricate facial behaviors, particularly in eye movement patterns, ocular orientation, and lip articulation accuracy.

tions in capturing detailed facial expressions, make previous state-of-the-art methods (e.g., AniPortrait [52], Follow Your Emoji [36])less effective than our method in reconstructing facial contours, gaze direction, and lip synchronization accuracy. These keypoint-dependent methodologies remain susceptible to interference from driving video subjects’ facial geometries, resulting in incomplete motionidentity separation. These methods face challenges in identity preservation due to changes in facial geometry resulting from misalignment of key points. Our framework overcomes these limitations through a parallel-control mamba layer (PCM), with an improved separation of facial identity characteristics from motion parameters, as evidenced in Figure 10. This enhanced decoupling enables superior iden-

As illustrated in the third, fifth, and sixth rows of Figure 10, our method can achieve tracking of the overall rotation of the head, which cannot be achieved by previous top-performing warping-based methods, such as LivePortrait.

While previous diffusion-based methods demonstrate notable advantages in output fidelity, their reliance on facial keypoint tracking introduces limitations. As shown in the third and fifth rows of Figure 10, discrepancies in facial geometry between source and target identities, combined with the inherent limitations of keypoint representa-

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

EDTalkerSadTalkerHelloVExpressEchominicMemoOursGroundTruth

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

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

Figure 11. The results generated by our method under audio control.

tity retention while capturing nuanced facial dynamics. Although X-Portrait [55] utilizes a non-explicit keypoint control method, it does not completely decouple motion and appearance information. This limitation results in noticeable flaws in the generated results, particularly evident in the fourth and fifth lines of Figure 10.

Moreover, frameworks built upon Stable Diffusion’s image generation architecture typically under-perform our method in temporal coherence metrics. By integrating the stable video diffusion model with our framework, we achieve significant improvements in three critical aspects: identity consistency preservation, visual quality optimization, and micro-expression reproduction. This collectively produces more natural-looking and temporally stable animations. We provide video demos in the supplementary materials. In these video demos, we compare our method with other methods, and our method obviously achieves better results. Additionally, we found that when some of the reference images provide more details, the results can be even more realistic.

###### B.2. Audio Driven Talker Head Generation

We present a comprehensive comparison with all baseline methods in Figure 11. As shown in the figure, our method is

able to produce accurate lip motion while containing fewer artifacts. Notably, our method generates natural head poses and expressions similar to the ground truth (please refer to the video demo in Supplementary Material), whereas other methods mainly manipulate the mouth shape and leave other regions static. These results confirm that our mamba design effectively aggregates audio signals with facial tokens to produce natural expressions and accurate lip synchronization, as we use the face mask as an audio mask to incorporate nearly all facial tokens in an audio-driven manner.

###### B.3. Audio-visual Joint Driven

We also present additional demonstrations in Figure 12, which displays the results produced by our method under audio-visual joint control. Our approach effectively maintains lip synchronization with the audio while accurately reflecting the expressions of the Motion Driving sources. We highly recommend watching the video demonstrations. Additional results can be found in the supplementary materials, where video demonstrations are also available.

##### Generated Motion Driving

Motion Driving Generated

[Figure 507]

[Figure 508]

[Figure 509]

shoe /ʃuː/

with /wɪð/

[Figure 510]

[Figure 511]

[Figure 512]

out /aʊt/

little /ˈlɪtl/

[Figure 513]

[Figure 514]

[Figure 515]

cup /kʌp /

always /ˈɔːlweɪz/

[Figure 516]

[Figure 517]

[Figure 518]

and /ənd/

alcohol /ˈælkəhɒl/

Figure 12. The results generated by our method under audio-visual joint control.

#### C. Ethics Considerations and AI Responsibility

This study aims to develop artificial intelligence-driven virtual avatars with enhanced visual emotional expression capabilities, utilizing audio or visual inputs, for applications in positive and constructive domains. The technology is designed specifically for ethical purposes, focusing on applications that are beneficial to society, and is not intended for generating deceptive or harmful media content.

However, as with all generative approaches in this field, there remains a theoretical concern about potential misuse for identity replication or malicious purposes. The research team strongly condemns any attempts to use the technology

for creating fraudulent, harmful, or misleading representations of real individuals. Rigorous technical evaluations of the current system indicate that the generated outputs exhibit clear artificial features, and quantitative comparisons with genuine human recordings show measurable discrepancies, ensuring that the results remain distinguishable from authentic human expressions.

