## X-Dancer: Expressive Music to Human Dance Video Generation

Zeyuan Chen1,2 Hongyi Xu2 Guoxian Song2 You Xie2 Chenxu Zhang2 Xin Chen2 Chao Wang2 Di Chang2,3 Linjie Luo2

1UC San Diego 2ByteDance 3University of Southern California

# arXiv:2502.17414v2[cs.CV]11Jul2025

[Figure 1]

Figure 1. We present X-Dancer, a unified transformer-diffusion framework for zero-shot, music-driven human image animation from a single static image. Our approach accommodates diverse body forms and appearances, generating diverse and highly expressive synchronized full-body dance motions. It captures both fine-grained movements of the head and hands and large-scale actions such as body rotations and jumps, seamlessly translating them into vivid and lifelike dance videos.

### Abstract

We present X-Dancer, a novel zero-shot music-driven image animation pipeline that creates diverse and long-range lifelike human dance videos from a single static image. As its core, we introduce a unified transformer-diffusion framework, featuring an autoregressive transformer model that synthesize extended and music-synchronized token sequences for 2D body, head and hands poses, which then guide a diffusion model to produce coherent and realistic dance video frames. Unlike traditional methods that primarily generate human motion in 3D, X-Dancer addresses data limitations and enhances scalability by modeling a

wide spectrum of 2D dance motions, capturing their nuanced alignment with musical beats through readily available monocular videos. To achieve this, we first build a spatially compositional token representation from 2D human pose labels associated with keypoint confidences, encoding both large articulated body movements (e.g., upper and lower body) and fine-grained motions (e.g., head and hands). We then design a music-to-motion transformer model that autoregressively generates music-aligned dance pose token sequences, incorporating global attention to both musical style and prior motion context. Finally we leverage a diffusion backbone to animate the reference image with these synthesized pose tokens through AdaIN,

forming a fully differentiable end-to-end framework. Experimental results demonstrate that X-Dancer is able to produce both diverse and characterized dance videos, substantially outperforming state-of-the-art methods in term of diversity, expressiveness and realism. See our project page for more results: zeyuan-chen.com/X-Dancer/

### 1. Introduction

Dance is a universal form of self-expression and social communication, deeply embedded in human behavior and culture. With the rise of social media platforms like TikTok and YouTube Shorts, people increasingly share self-expressive dance videos online. However, creating expressive choreography typically demands practice and even professional training. From a computational aspect, generating realistic dance movements is challenging due to the freeform, personalized nature of dance and its alignment with musical rhythm and structure. In this work, we tackle the challenge of creating continuous, expressive and lifelike dance videos from a single static image, driven solely by a music track.

This study addresses two key challenges in music-driven human image animation: (1) generating smooth, diverse full-body movements at finer scales that capture the complex, non-linear synchronization with music inputs, and (2) translating these generated body movements into highfidelity video outputs that maintain visual consistency with the reference image and ensure temporal smoothness. Prior approaches [38, 44] mainly focus on computationally generating 3D human poses, such as SMPL skeletons [30], from music inputs, utilizing diffusion- or GPT-based frameworks. While these methods excel in producing highquality, clean motions, they are constrained by limited training datasets (primarily the multi-view AIST dataset [25, 45]), which lack diversity, and contain 3D body poses only (excluding head and hands motions). Expanding these datasets with widely available 2D monocular dance videos would require 3D human pose estimation, which is often error-prone and risks degraded motion quality and consistency. Moreover, we target at photorealistic dance video generation rather than 3D skeleton or mesh animations.

With recent advances in diffusion models, numerous works have leveraged their generative capabilities to synthesize visually compelling videos by animating a reference image with motion signals such as 2D skeletons [7, 17, 18], dense poses [54], and depth maps [66]. Unlike motion transfer settings that derive motion from a driving video, our goal is to generate motion signals that are consistent with the reference body shape and aligned with the input musical beats. Recently, a few studies [42, 53] have attempted to synthesize visual outputs end-to-end from audio inputs using diffusion models. While these methods have advanced in realism and dynamic quality, they still struggle to capture

long-range motion and audio context due to high computational demands. Moreover, these frameworks have primarily been validated on audio-driven talking heads, leaving uncertain whether they can accommodate the complexities of full-body kinematics and rapid motion transitions.

To this end, we propose X-Dancer, a unified framework that integrates an autoregressive transformer model for generating extended dance sequences attuned to input music, coupled with a diffusion model to produce high-resolution, lifelike videos. In contrast to prior methods focused on 3D music-to-dance motion generation, our approach models 2D human motion, leveraging widely accessible dance videos where 2D pose estimation is both reliable and scalable. For effective autoregressive motion generation, we develop a multi-part tokenization scheme for per-frame 2D whole-body poses, incorporating detected keypoint confidences to capture multi-scale human motions with motion blur and various occlusions. Thereafter we train a crossmodal transformer that auto-regressively predicts future Nframe pose tokens, paired with per-frame music features extracted with Jukebox [11] and Librosa [22]. Our design enables the model to capture a broader diversity of expressive, music-aligned movements with enhanced scalability in both model complexity and data scale. We then leverage a T2I diffusion model with temporal layers to animate the reference image by implicitly translating the generated confidence-aware motion tokens into spatial guidance via a trainable motion token decoder. Specifically, it integrates motion tokens through AdaIN [19] along upsampling from a learnable feature map into multi-scale spatial feature guidance. By co-learning pose translation with temporal and appearance reference modules [6, 14], the diffusion backbone interprets motion tokens with temporal context and body shape reference, leading to better shape-disentangled pose control, smoother and more robust visual outputs even under low-confidence or jittering pose sequences. This design establishes an end-to-end transformer-diffusion learning framework, merging the transformer’s strengths in longcontext understanding and generation with the diffusion model’s capability in high-quality visual synthesis.

To the best of our knowledge, X-Dancer represents the first music-driven human image animation framework. Trained on a large curated music-dance video dataset, our method excels at generating diverse, expressive and detailed whole-body dance videos attuned to input music, adaptable to both realistic and stylized human images across various body types. We extensively evaluate our model on challenging benchmarks, and X-Dancer outperforms state-of-the-art baselines both quantitatively and qualitatively. Additionally, we highlight its scalability and customization capabilities, showcasing scalability across varying model and data scales, as well as fine-tuning to characterized choreography. We summarize our contributions as follows,

- • A novel transformer-diffusion based music-to-dance human image animation, achieving state-of-the-art performance in terms of motion diversity, expressiveness, music alignment and video quality.
- • A cross-modal transformer model that captures longrange synchronized dance movements with music features, employing a multi-scale tokenization scheme of whole-body 2D human poses with keypoint confidence.
- • A diffusion-based human image animation model that interprets temporal pose tokens and translates them into consistent high-resolution video outputs.
- • Demonstration of captivating zero-shot music-driven human image animations, along with characterized choreography generation.

### 2. Related Work

#### 2.1. Music to Dance Generation

Significant progress has been made in realistic human motion generation [9, 31, 38, 44, 61, 65] from various inputs, such as speech [2, 13, 29, 36, 60] or text [20, 35, 41, 63]. However, the task of music-to-dance generation [1, 24, 27, 28, 32, 38, 44, 51, 55, 59, 67] presents unique challenges: (1) ensuring the generated dance rhythmically aligns with the music, and (2) producing intricate motions with diverse styles and speeds. There has been several 3D human pose datasets [24–26, 32] proposed. The AIST++ dataset [25] is a 3D music pose dataset containing 1,408 dance motions tailored to various music genres. FineDance [26], a 3D dataset focusing on fine-grained hand motions, includes 14 hours of dance data. Models like Bailando [38] and EDGE [44] have leveraged AIST++ for training. Bailando [38] pioneered using a VQ-VAE for 3D pose encoding, followed by an Actor-Critic GPT model to generate body poses conditioned on music. EDGE [44] applied a diffusion framework to predict human poses from a noisy sequence, conditioning on music and employing long-form sampling for extended dance generation. DabFusion [55] explores image-to-video generation conditioned on music. Despite these advancements, existing approaches trained on 3D datasets often generate dance poses with limited diversity. X-Dancer overcomes these data constraints by leveraging a broad spectrum of 2D dance motions, aligning complex movements with musical beats, and utilizing widely available video data to enhance scalability.

#### 2.2. Human Image Animation

With the advancements in diffusion models [5, 14, 16, 48, 52], generating high-quality human videos has become feasible. The introduction of ControlNet [50, 62] and PoseGuider [17, 18] has further empowered prior methods [7, 8, 17, 54, 66] to create realistic dance videos from a reference image using motion signals such as 2D skele-

tons, dense poses, and depth maps. However, generating expressive, temporally smooth 2D motion sequences that maintain consistency with reference body shapes and align well with music remains an open challenge. Several methods [21, 42, 53] adopt an end-to-end pipeline for generating realistic talking head videos from audio by injecting audio features directly into the network via cross-attention layers. While these methods excel in capturing micro-expressions and lip movements, they fall short in handling highly dynamic and articulated dance movements. X-Dancer implicitly incorporates generated motion tokens into a diffusion backbone to animate the reference image, enabling highquality and shape-consistent dance video generation.

### 3. Method

Given a single portrait as the reference image IR and a conditioning music sequence represented as Mt, our objective is to generate a sequence of dance frames IM

, where t = 1,...,T denotes the frame index. The generated sequence IM

t

seeks to maintain the appearance and background context depicted in IR while present expressive dance movements in harmony with the provided musical rhythm and beats. As illustrated in Fig. 2, our model is trained in two stages: transformer-based 2D human dance motion generation (Section. 3.2), and diffusion-based video synthesis (Section. 3.3) from the generated motion sequence.

t

#### 3.1. Data Representations

Video Generation. To generate human dance videos, we employ Latent Diffusion Models [37] that synthesize samples in the image latent space, facilitated by a pretrained image auto-encoder. During training, latent features of images are progressively corrupted with Gaussian noise ϵ, following the Denoising Diffusion Probabilistic Model (DDPM) framework [15, 39]. A UNet-based denoising framework, enhanced with spatial and temporal attention blocks, is then trained to learn the reverse denoising process.

We apply human-centric cropping to the training videos of half and full-body dances, yielding a unified resolution of 896 × 512. Rather than modeling intricate pixel-wise movements directly on music, we first establish the correlation between music and human dance movement, which subsequently guides the final visual synthesis.

Motion Modeling. In contrast to prior methods that generate 3D human motions, we represent diverse dance motions as 2D pose sequences. Compared to its 3D counterparts, 2D human dance motions are widely accessible from large collections of monocular videos, eliminating the need for complex multiview capture setups or labor-intensive 3D animations. Furthermore, the 2D pose detection is significantly more reliable and robust. To enhance the realism and expressiveness, we model not only large body articulations but also finer details on head movements and hand gestures.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

|𝑀𝑡|
|---|

[Figure 14]

[Figure 15]

…

|𝐹𝑇𝐿|
|---|
| |
|𝐹𝑇𝐽|
| |
|𝐶𝑇1:𝐵|

𝐹𝑡𝐿

𝐹1𝐿

[Figure 16]

[Figure 17]

𝐹1:𝑇𝐿 𝐹1:𝑇𝐽

…

…

[Figure 18]

[Figure 19]

[Figure 20]

𝐹𝑡𝐽

𝐹1𝐽

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

motion context

𝐹𝑔

𝐶𝑡1:𝐵

𝐶11:𝐵

[Figure 25]

| | |
|---|---|
| | |

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

|𝑛𝑒𝑥𝑡 𝑡𝑜𝑘𝑒𝑛 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛|
|---|

Transformer

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

|23|
|---|

[Figure 35]

𝑐𝑎𝑢𝑠𝑎𝑙 𝑎𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛

|𝐸1| |
|---|---|
| | |

…

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

…

…

|9|
|---|

|𝐶11:𝐵|
|---|

|𝐶𝑡1:𝐵|
|---|

|𝐶𝑇1:𝐵|
|---|

AdaIN

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

|12|
|---|

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

|𝐼𝑀𝑇|
|---|

[Figure 51]

[Figure 52]

|𝑙𝑒𝑎𝑟𝑛𝑎𝑏𝑙𝑒 𝑓𝑒𝑎𝑡𝑢𝑟𝑒 𝑚𝑎𝑝|
|---|

𝑀𝑜𝑡𝑖𝑜𝑛 𝐷𝑒𝑐𝑜𝑑𝑒𝑟

|𝐸2| |
|---|---|
| | |

[Figure 53]

…

[Figure 54]

[Figure 55]

|37|
|---|

|𝐷|
|---|

[Figure 56]

|𝐼𝑀𝑡|
|---|

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

|24|
|---|

|𝐸3| |
|---|---|
| | |

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

…

[Figure 72]

|𝐼𝑀1|
|---|

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

|19|
|---|

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

|56|
|---|

|𝐸4|
|---|

…

[Figure 89]

|32|
|---|

[Figure 90]

[Figure 91]

[Figure 92]

|𝑈𝑁𝑒𝑡|
|---|

[Figure 93]

[Figure 94]

|18|
|---|

[Figure 95]

|𝐼𝑅|
|---|

|𝐸5| |
|---|---|
| | |

[Figure 96]

…

[Figure 97]

Self-Attention

[Figure 98]

𝐶𝑡1:𝐵

|9|
|---|

|𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒𝑁𝑒𝑡|
|---|

[Figure 99]

Cross-Attention

[Figure 100]

Temporal Attention

- Figure 2. Overview of X-Dancer. We propose a cross-conditional transformer model to autoregressively generate 2D human poses synchronized with input music, followed by a diffusion model to produce high-fidelity videos from a single reference image IR. First, we develop a multi-part compositional tokenization for 2D poses, encoding and quantizing each body part independently with keypoint confidence. A shared decoder merges these tokens into a whole-body, confidence-aware pose. Next, a GPT-based transformer autoregressively predicts future pose tokens with causal attention, conditioned on past poses and aligned music embeddings, as well as global music styles and prior motion context. A motion decoder is trained to generate multi-scale spatial pose guidance upsampled from a learned feature map, integrating the generated motion tokens within a temporal window (16 frames) via AdaIN. By co-training the motion decoder and temporal modules, our diffusion model is capable of synthesizing temporally smooth and high-fidelity video frames, while maintaining consistent appearance with the reference image with a trained reference net.

Notably, we incorporate keypoint confidence into our pose representation, allowing the model to account for motion blur and occlusions. Each per-frame whole-body pose with P joints is thus represented as p ∈ RP×3, where the last dimension encodes the keypoint confidence.

Music Embedding. Inspired from [44], we utilize the pretrained Jukebox model [11] to extract rich musical features, supplemented by rhythmic information with one-hot encoding of music beats using an audio processing toolbox Librosa [22]. We resample and synchronize these extracted embeddings—denoted as F1:JT for Jukebox and F1:LT for Librosa—to the video frame rate, ensuring per-frame alignment between the music and visual elements.

#### 3.2. Transformer-Based Music to Dance Motion

Given a collection of monocular, single-person, musicsynchronized dance videos with paired 2D whole-body pose detections, we aim to model the intricate, non-linear correlation between skeletal dance motion and musical features. To achieve this, we first introduce a compositional, confidence-aware VQ-VAE, which captures diverse and nuanced human dance poses across different body parts. Next, we leverage a GPT-like transformer to autoregressively predict future pose tokens, modeling temporal motion token transitions in synchronization with music embeddings.

Compositional Confidence-Aware Human Pose Tok-

enization. Our approach builds on the standard VQ-VAE framework, trained in a self-supervised manner. Given whole-body 2D poses with associated keypoint confidences p, a 1D convolutional encoder maps p into a latent pose embedding ze(p) = E(p), which is then quantized by mapping to its nearest representation zq(p) within a learnable codebook, and finally the decoder D reconstructs the pose pˆ from zq(p). However, as prior studies [38, 49, 58] suggest, the dependencies between spatial keypoints are complex, and a vanilla VQ-VAE often struggles to capture subtle pose details, such as finger movements and head tilts, due to information loss during quantization and multi-frequency nature of pose variations across different body parts.

To improve expressive coverage, we train independent 2D pose encoders Ej and learn B = 5 separate codebooks Zqj for upper and lower half bodies, left and right hands, and head respectively, allowing the model to spatially decompose 2D whole-body pose variations across different frequencies. With pose partition, distinct body-part codes can be flexibly combined, enhancing the range of expressiveness represented within each individual codebook. To capture part-wise spatial correlations and ensure information flow across body parts, we concatenate the quantized pose latents and feed them into a shared decoder. The resulting embedding is then mapped to reconstructed keypoint coordinates through separate projection heads, enabling joint re-

construction while preserving nuanced part-specific details.

We train encoder and decoder simultaneously with the compositional codebooks with the following loss function:

B

B

∥pˆj −pj||2+β

∥sg zej(p) −zqj(p)∥2, (1)

LVQ =

j=1

j=1

where sg is a stop gradient operation, and the second term is a commitment loss with a trade-off β. We utilize exponential moving average (EMA) [47] to update codebook vectors and remove the embedding loss Bj=1 ∥zej(p) − sg zqj(p) ∥2 for more stable training.

Cross-Conditional Autoregressive Motion Modeling. After training the compositional quantized multi-part codebooks, 2D human poses detected in our training videos can be represented as sequences of codebook tokens via encoding and quantization. Specifically, each detected pose is mapped to a sequence of corresponding codebook indices, structured as one-hot vectors indicating the nearest codebook entry for each element. We denote this as C1:j T = ((cj1,1,...,cjK,1),...,(cj1,T,...,cjK,T)), where K is the number of tokens per body part j in each frame.

With this quantized motion representation, we design a temporal autoregressive model to predict the multinomial distributions of possible subsequent poses for each body part, conditioned on both the input music embeddings F1:JT and F1:LT and the preceding motion tokens. Our motion generation transformer is conditioned on the music inputs in two ways. First, we combine the Jukebox and Librosa music embeddings to form starting tokens, which serves as a global music context, such as styles and genres, that informs the entire motion sequence generation. Second, inspired by [38], we concatenate the frame-wise projected music embeddings with the corresponding motion tokens as inputs to the transformer model, ensuring precise synchronization between the motion and music features. This dual conditioning allows our model to produce both globally coherent and locally synchronized dance movements. We use T = 64 frames for our autoregressive model training. To handle extended motion sequence generation with consistent motion styles and smooth transitions, we additionally incorporate cross-segment motion context into the transformer model. Specifically, we uniformly sample a subset of 8 frames from the previous motion segment as the motion context, and append them after the global music context inputs. We denote the combined context as Fg.

Since we model body parts independently, maintaining coherence in the assembled whole-body poses is essential to avoid asynchronous movements (e.g., upper and lower body moving in different directions). To address this, we leverage mutual information across multi-part motions, designing our model with cross-conditioning across body parts. Specifically, we employ a GPT model to estimate the joint distributions of C1:j T as follows,

T

B

K

ϕ(C1:1:TB|Fg) =

t=1

j=1

k=1

ϕ cjk,t|c1:1:BK,<t,c<j1:K,t,cj<k,t,F≤Jt,F≤Lt,Fg (2)

We structure the cross-conditioning between body parts in two ways: (1) the current motion token is conditioned on all preceding motion information from all body parts, ensuring inter-part temporal coherence; (2) by ordering body parts as upper and lower body, head, and hands, we build the hierarchical dependencies from primary components (upper/lower body) to finer, high-frequency movements (head and hands). Since each body part’s pose is represented with a small set of tokens, we empirically observe that causal attention is sufficient to model the next-token distribution within each part. This modeling strategy preserves motion coherence of each body part as a whole, producing expressive and plausible dance movements.

Our GPT model is optimized through supervised training using a cross-entropy loss on the next-token probabilities. Notably, because our pose tokenization includes associated keypoint confidences, the transformer also learns to model temporal variations in confidence, such as those caused by motion blur and occlusions, enabling it to capture more realistic motion distributions observed in videos.

#### 3.3. Diffusion-Based Dance Video Synthesis

We employ a diffusion model to synthesize high-resolution, realistic human dance videos, conditioned on the generated motions from our trained transformer together with a given reference image. To achieve this, we leverage a pretrained T2I diffusion backbone [37] and incorporate additional temporal modules [14] for temporal consistency. For transferring the reference image context, a reference network, as a trainable copy of the backbone UNet, extracts reference features of identity appearance and background which are cross-queried by the self-attentions within the backbone UNet. Motion control is achieved through an addition module, often configured in ControlNet [62] or light-weight PoseGuider [17, 18], which translates the motion conditions into 2D spatial guidance additive to the UNet features.

To incorporate the generated motion tokens into human image animation, one approach is to utilize the trained VQVAE decoder D to decode pose tokens into keypoint coordinates, which are then visualized as 2D skeleton maps for motion conditioning. These skeleton maps can provide motion guidance to the final diffusion synthesis through a PoseGuider module. While effective to some extent, this approach introduces a non-differentiable process due to the skeleton map visualization, which impedes end-toend training and often results in the loss of keypoint confidence information. Additionally, since the temporal module is trained on real video data where pose sequences are typically smooth, it may struggle with jittery or inconsistent

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

Reference X-Dancer Hallo Bailando + Pose Guider EDGE + Pose Guider

- Figure 3. Qualitative Comparisons. Among all the methods, X-Dancer achieves the most expressive and high-fidelity human dance video synthesis, maintaining the highest consistency with both the reference human characteristics and the background scene.

motions generated by the transformer model at inference.

In place of explicit translation of motion tokens into 2D skeleton maps as pose guider conditions, we introduce a trainable motion decoder that implicitly and directly translates the 1D pose tokens into 2D spatial guidance, added towards the intermediate features of the denoising UNet. Starting from a learnable 2D feature map, the decoder injects the 1D pose token sequences including the keypoint confidences with AdaIN layers [19], progressively upsampling this feature map into multiple scales aligned with the resolutions of the denoising UNet features. The motion decoder is trained alongside the temporal module within a 16-frame sequence window, effectively decoding token sequences into continuous pose guidance that integrates temporal context from adjacent frames. Moreover, by incorporating reference image context during training, we observe empirically that the decoded pose guidance retains minimal identity and body shape information compared to explicit 2D poses, enabling generated pose tokens to adapt seamlessly to subjects with varied body shapes and appearances.

- 4. Experiments

VAE is trained for 40k steps with a batch size of 2048 at a learning rate of 2 × 10−4. Next, we train an autoregressive model for pose token prediction over 300k steps, initialized with pretrained GPT-2 weights, using a batch size of 24 and a learning rate of 1 × 10−4. The model operates on 64frame pose sequences with a 2224-token context window. Lastly, we fine-tune the denoising UNet from SD1.5 [37] and ReferenceNet with two randomly selected frames per video, followed by co-training the motion token decoder and temporal module [14] with diffusion losses on consecutive 16 frames. The diffusion stage is trained for 90k steps at a learning rate of 1 × 10−5 and a batch size of 16.

Inference. We initiate our autoregressive dance motion generation from pose tokens encoded from the reference image pose. Extended dance sequences are then generated in 64-frame sliding segments with a 12-frame overlap, while 8 uniformly sampled frames from last segment serve as global motion context. We synthesize all video frames simultaneously with the full pose token sequence, applying prompt traveling [43] to improve temporal smoothness.

#### 4.2. Evaluations and Comparisons

To the best of our knowledge, no existing approach addresses music-driven 2D video generation from a single human image. We adapted and combined established models to create two baselines for comparisons. First, we adapted the audio-driven, diffusion-based portrait animation model Hallo [53], retraining it for our task by substituting its audio features with our music embeddings to animate human images via cross-attention modules. For the second baseline, we utilize 3D music-to-dance generation models like Bailando [38] and EDGE [44] for motion synthesis, projecting their outputs into 2D pose maps, which are then fed into a diffusion model with a pose guider for controlled image animation. For fair comparison, we also train our motion transformer (stage 2, X-Dancer-AIST) on the AIST dataset [45], consistent with Bailando and EDGE, but using detected 2D

#### 4.1. Implementation Details

Dataset. Our model is trained on a curated in-house visualaudio dataset of 76,818 monocular dance recordings in indoor and outdoor settings, averaging 15 seconds per clip. Each video is cropped to 896 × 512 resolution and resampled at 30 fps, covering half- to full-body dances across diverse music and performer characteristics. Details on the dataset are provided in the supplementary paper.

Training. We train our full pipeline in three stages on 8 A100 GPUs using the AdamW optimizer [57]. First, we train a multi-part pose VQ-VAE to encode and quantize 60 joint coordinates and keypoint confidences, into 5-part pose tokens. Each body part’s pose uses 6 tokens, each with a unique 512-entry codebook of 6D embeddings. This VQ-

Table 1. Quantitative comparison on motion generation.

Table 2. Quantitative comparison on visual synthesis.

AIST++ Dataset / In-House Dataset

In-House Dataset FVD ↓ FID-VID ↓ ID-SIM ↑

Metrics

Metrics

FVD ↓ DIV ↑ BAS ↑

Ground Truth 509.58/129.75 34.10/29.67 0.24/0.22 Hallo [53] 548.81/249.12 28.66/28.98 0.16/0.20 Bailando [38] 621.22/534.02 22.34/24.05 0.19/0.19 EDGE [44] 639.46/303.36 24.87/27.29 0.26/0.24

Hallo [53] 609.08 76.99 0.4870 Bailando [38] + PG 583.26 100.02 0.3392 EDGE [44] + PG 613.81 93.73 0.3034 Our motion + PG 735.05 72.71 0.4894

X-Dancer-AIST 620.73/309.18 25.25/24.31 0.26/0.22 X-Dancer 531.52/238.22 25.61/28.08 0.23/0.21

X-Dancer-AIST 549.38 74.06 0.3652 X-Dancer 507.06 61.94 0.5317

Table 3. Ablation on our transformer model designs.

FVD ↓ DIV ↑ BAS ↑

w/o global music context 265.73 27.04 0.2142 w/o global motion context 247.54 26.42 0.2154 sub-dataset + GPT-medium 402.63 24.40 0.2112 sub-dataset + GPT-small 332.93 24.58 0.2046

##### X-Dancer 238.22 28.08 0.2182

poses instead of 3D skeletons. However, since AIST has limited diversity in identities and appearances, we use our diffusion model—trained on our curated dataset—for comparison on synthesized videos. We conduct separate evaluations of all models on the test split of AIST (40 videos) as well as our curated music-video dataset (230 videos).

Quantitative Evaluation. We numerically compare XDancer with baselines in terms of quality of both motion generation and video synthesis. Specifically, we calculate the Fr´echet Video Distance (FVD) [12, 46] between generated dance motions and ground-truth training sequences for assessment of motion fidelity. For motion diversity, we compute the distributional spread of generated pose features (DIV) [25, 38], To numerically evaluate the alignment between the music and generated dancing poses, we follow [25, 38] to measure the Beat Align Score (BAS) by the average temporal distance between each music beat and its closest dancing beat. These evaluations are conducted in 2D pose space, where we detect poses over synthesized videos from the retrained Hallo model, and project the 3D skeleton motion of Bailando and EDGE into 2D via orthographic projection, using camera parameters that best align with the human bounding box in the reference image.

As shown in Tab. 1, our method surpasses all baselines in terms of motion fidelity (FVD) and music beat alignment (BAS), while achieving the second-best diversity (DIV). Notably, Bailando and EDGE is trained on professional dances (AIST++ [25]), whereas our dataset comprises videos of everyday individuals, reflected in lower BAS even for ground-truth videos. Our method trained on AIST (XDancer-AIST) achieves the highest BAS whereas our model trained on our curated videos still significantly outperforms

Table 4. Ablation on VQ-VAE and video diffusion. PG and MD denotes PoseGuider and Motion Decoder respectively.

Pose L1 FullBody/Head/Hands ↓

Single-Part VQVAE 0.83 / 0.64 / 0.52 Multi-Part VQVAE 0.50 / 0.40 / 0.42

Video PSNR ↑ / LPSIS ↓ / FVD ↓

GT Pose + PG 19.465 / 0.197 / 294.52 Single-Part + MD 17.079 / 0.258 / 350.73 Multi-Part + PG 18.836 / 0.207 / 384.16 Multi-Part + MD 19.148 / 0.207 / 295.87

Bailando and Hallo for beat alignment. Hallo entangles motion generation and video synthesis, and achieves a higher DIV score than X-Dancer primarily due to its extremely noisy video outputs which results in jittering and chaotic skeleton motions following pose projection.

For evaluation of video synthesis fidelity, we measure the FVD and FID-VID [4] score between the ground-truth and generated dance videos. Additionally, we assess identity preservation using the ArcFace score [10], which measures the cosine similarity of face identity features (ID-SIM). All metrics are evaluated on our test video dataset. As an extra baseline, we replace the motion token decoder in our full pipeline with a pose guider (Our motion + PG). As shown in Tab. 2, our method achieves the highest visual quality and identity preservation, which we attribute to our disentangled design of motion generation and video synthesis (compared to Hallo) and the use of an implicit motion token decoder rather than an explicit pose guider.

Qualitative Comparisons. We present visual comparisons between our method and the baselines in Fig. 3. For more dynamic comparisons and user study, please refer to our supplementary material. The modified Hallo [53] represents an end-to-end diffusion pipeline that directly synthesizes the final video without intermediate motion generation. However, it exhibits noticeable artifacts, particularly in large articulated body motions, and often fails to preserve the human body’s articulation topology. Bilando [38] and EDGE [44] generate body motion in 3D space without con-

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

###### GT Pose w. single-part VQ-VAE w. multi-part VQ-VAE GT Pose w. single-part VQ-VAE w. multi-part VQ-VAE

Figure 4. Our Multi-part VQ-VAE captures intricate poses such as hand gesture (left), arm bending, head tilting and leg lifting(right) whereas single-part pose tokenization fails to fully replicate them.

sidering the scene context or the human shape in the reference image. Despite post-processing for skeleton alignment in the pose guider input, these methods still struggle with significant identity and shape distortions, often producing unnatural interactions with the background scene. Furthermore, they do not model head and hand motions, leading to more rigid and less expressive dance movements compared to X-Dancer. We also note that our model trained on monocular dance videos is efficient and versatile, easily adapting to specific dances. This is not easily achievable with EDGE or Bailando, which require extensive effort in crafting 3D skeleton dances (supplementary paper Sec.3)

#### 4.3. Ablation Study

We conduct ablation studies by systematically removing individual design components from our full training pipeline. Multi-part VQ-VAE. As shown in Tab. 4, our compositional multi-part VQ-VAE significantly reduces pose reconstruction error compared to single-part whole-body pose tokenization. The multi-part VQ-VAE better captures finescale pose variations, while single-part tokenization often loses high-frequency details, resulting in more rigid and less expressive motion generation. To further assess its impact, we train a video generation diffusion model using single-part pose tokenization. As illustrated in Figure 4, the single-part representation struggles to control fine-grained motions like hip swaying, leg lifting and head tilting, leading to perceptually lower-quality videos. This is further supported by its inferior video reconstruction metrics (PSNR and LPIPS [64]) in Tab. 4.

Motion Generation. Next, we assess the impact of global music and motion context on motion generation. As presented in Tab. 3, both contexts contribute to producing more consistent, plausible, and music-synchronized motions. We further analyze the benefits of 2D motion modeling and transformer-based autoregressive generation by scaling both model parameters and dataset size. Across all metrics, we observe significant performance gains (Tab. 3) as the number of training parameters increases from 117M (GPT-small) to 345M (GPT-medium) and data scale from 10k to 100k videos, underscoring the scalability potential of

monocular dance video data with our architecture, sheding light on further performance improvements as they scale.

Pose-Guided Video Synthesis. In Tab. 2, we compare our motion token decoder (X-Dancer) against a pose guider using explicitly decoded skeleton map (Our motion + PG). While our transformer-generated motion exhibits jittering, our motion token decoder significantly reduces jitter and enhances temporal consistency by leveraging temporal motion context. Additionally, it demonstrates superior identity and body shape preservation compared to the pose guider.

In Tab.4, we evaluate the effectiveness of our motion token decoder (Multi-Part + MD) against the pose guider (Multi-Part + PG) in self-driven video synthesis. Our motion decoder directly infers body pose coordinates and confidences from pose tokens, achieving lower reconstruction errors (PSNR and LPIPS) compared to the pose guider, which relies on explicitly decoded pose maps. However, due to motion blurriness in fast dance motions and finegrained structural variations in small pixel regions like the face and hands, visual rendering artifacts remain present in both methods. Notably, these artifacts stem primarily from the diffusion model’s limitations rather than errors in pose reconstruction, as evidenced by the similar video error observed in GT Pose+PG (Tab. 4).

### 5. Conclusion

We present X-Dancer, a novel framework that unites an autoregressive transformer with a diffusion model to generate high-quality, music-driven human dance videos from a single reference image. Unlike prior works, X-Dancer models and generates dance movements in 2D space, harnessing widely accessible 2D poses from monocular dance videos to capture diverse, expressive whole-body motions. Our method achieves state-of-the-art results in video quality, motion diversity and expressiveness, providing a scalable and adaptable solution for creating vivid, music-aligned dance videos across various human forms and styles.

Limitations and Future Work. Our model is trained solely on curated real-human daily dance videos, which can be noisy and lack the motion precision found in professional dancer videos. Consequently, out-of-domain human images may lead to rendering artifacts, and the generated dance motions may occasionally lack music alignment. More failure cases are present in the supplementary paper. While we designed our pipeline to be end-to-end trainable and scalable, we currently implement it in stages due to memory limitations. Future work will explore large-scale, multi-machine training to further enhance performance and efficiency.

Ethics Statement. Our work aims to improve human image animation from a technical perspective and is not intended for malicious use like fake videos. Therefore, synthesized videos should clearly indicate their artificial nature.

### References

- [1] Simon Alexanderson, Rajmund Nagy, Jonas Beskow, and Gustav Eje Henter. Listen, denoise, action! audio-driven motion synthesis with diffusion models. ACM TOG, 2023. 3
- [2] Tenglong Ao, Qingzhe Gao, Yuke Lou, Baoquan Chen, and Libin Liu. Rhythmic gesticulator: Rhythm-aware co-speech gesture synthesis with hierarchical neural embeddings. ACM TOG, 2022. 3
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1
- [4] Yogesh Balaji, Martin Renqiang Min, Bing Bai, Rama Chellappa, and Hans Peter Graf. Conditional gan with discriminative filter generation for text-to-video synthesis. In IJCAI,

2019. 7

- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [6] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023. 2
- [7] Di Chang, Yichun Shi, Quankai Gao, Hongyi Xu, Jessica Fu, Guoxian Song, Qing Yan, Yizhe Zhu, Xiao Yang, and Mohammad Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identity-aware diffusion. In ICML, 2023. 2, 3
- [8] Di Chang, Hongyi Xu, You Xie, Yipeng Gao, Zhengfei Kuang, Shengqu Cai, Chenxu Zhang, Guoxian Song, Chao Wang, Yichun Shi, et al. X-dyna: Expressive dynamic human image animation. arXiv preprint arXiv:2501.10021,

2025. 3

- [9] Ling-Hao Chen, Jiawei Zhang, Yewen Li, Yiren Pang, Xiaobo Xia, and Tongliang Liu. Humanmac: Masked motion completion for human motion prediction. In ICCV, 2023. 3
- [10] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019. 7
- [11] Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A generative model for music. arXiv preprint arXiv:2005.00341,

2020. 2, 4

- [12] Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, JunYan Zhu, and Jia-Bin Huang. On the content bias in fr´echet video distance. In CVPR, 2024. 7
- [13] Shiry Ginosar, Amir Bar, Gefen Kohavi, Caroline Chan, Andrew Owens, and Jitendra Malik. Learning individual styles of conversational gesture. In CVPR, 2019. 3
- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your

- personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 3, 5, 6
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arxiv:2006.11239,

2020. 3

- [16] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 35, 2022. 3
- [17] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In CVPR, 2024. 2, 3, 5
- [18] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117, 2023. 2, 3, 5
- [19] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV,

2017. 2, 6

- [20] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. NeurIPS, 2023. 3
- [21] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. arXiv preprint arXiv:2409.02634, 2024. 3
- [22] Yanghua Jin, Jiakai Zhang, Minjun Li, Yingtao Tian, Huachun Zhu, and Zhihao Fang. Towards the automatic anime characters creation with generative adversarial networks. arXiv preprint arXiv:1708.05509, 2017. 2, 4
- [23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [24] Nhat Le, Thang Pham, Tuong Do, Erman Tjiputra, Quang D Tran, and Anh Nguyen. Music-driven group choreography. In CVPR, 2023. 3
- [25] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. Ai choreographer: Music conditioned 3d dance generation with aist++. In ICCV, 2021. 2, 3, 7
- [26] Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li. Finedance: A fine-grained choreography dataset for 3d full body dance generation. In ICCV, 2023. 3, 1
- [27] Ronghui Li, Yuqin Dai, Yachao Zhang, Jun Li, Jian Yang, Jie Guo, and Xiu Li. Exploring multi-modal control in musicdriven dance generation. In ICASSP, 2024. 3
- [28] Ronghui Li, YuXiang Zhang, Yachao Zhang, Hongwen Zhang, Jie Guo, Yan Zhang, Yebin Liu, and Xiu Li. Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives. In CVPR,

2024. 3

- [29] Xian Liu, Qianyi Wu, Hang Zhou, Yinghao Xu, Rui Qian, Xinyi Lin, Xiaowei Zhou, Wayne Wu, Bo Dai, and Bolei Zhou. Learning hierarchical cross-modal association for cospeech gesture generation. In CVPR, 2022. 3

- [30] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851–866. 2023. 2
- [31] Thomas Lucas, Fabien Baradel, Philippe Weinzaepfel, and Gr´egory Rogez. Posegpt: Quantization-based 3d human motion generation and forecasting. In ECCV, 2022. 3
- [32] Zhenye Luo, Min Ren, Xuecai Hu, Yongzhen Huang, and Li Yao. Popdg: Popular 3d dance generation with popdanceset. In CVPR, 2024. 3
- [33] Supreeth Narasimhaswamy, Uttaran Bhattacharya, Xiang Chen, Ishita Dasgupta, Saayan Mitra, and Minh Hoai. Handiffuser: Text-to-image generation with realistic hand appearances. In CVPR, 2024. 2
- [34] OpenAI. Sora: Creating video from text. https://openai.com/sora/, 2024. 2
- [35] Mathis Petrovich, Michael J Black, and G¨ul Varol. Temos: Generating diverse human motions from textual descriptions. In ECCV, 2022. 3
- [36] Shenhan Qian, Zhi Tu, Yihao Zhi, Wen Liu, and Shenghua Gao. Speech drives templates: Co-speech gesture synthesis with learned templates. In ICCV, 2021. 3
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3, 5, 6
- [38] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In CVPR, 2022. 2, 3, 4, 5, 6, 7, 1
- [39] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [40] Shaolin Su, Qingsen Yan, Yu Zhu, Cheng Zhang, Xin Ge, Jinqiu Sun, and Yanning Zhang. Blindly assess image quality in the wild guided by a self-adaptive hyper network. In CVPR, 2020. 1
- [41] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In ICLR, 2023. 3
- [42] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive-generating expressive portrait videos with audio2video diffusion model under weak conditions. arXiv preprint arXiv:2402.17485, 2024. 2, 3
- [43] Jonathan Tseng, Rodrigo Castellon, and C. Karen Liu. Edge: Editable dance generation from music, 2022. 6
- [44] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. Edge: Editable dance generation from music. In CVPR, 2023. 2, 3, 4, 6, 7, 1
- [45] Shuhei Tsuchida, Satoru Fukayama, Masahiro Hamasaki, and Masataka Goto. Aist dance video database: Multi-genre, multi-dancer, and multi-camera database for dance information processing. In ISMIR, 2019. 2, 6, 1
- [46] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7

- [47] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. NeurIPS, 30, 2017. 5
- [48] Yilin Wang, Zeyuan Chen, Liangjun Zhong, Zheng Ding, and Zhuowen Tu. Dolfin: Diffusion layout transformers without autoencoder. In ECCV, 2024. 3
- [49] Yuan Wang, Zhao Wang, Junhao Gong, Di Huang, Tong He, Wanli Ouyang, Jile Jiao, Xuetao Feng, Qi Dou, Shixiang Tang, and Dan Xu. Holistic-motion2d: Scalable whole-body human motion generation in 2d space. arXiv preprint arXiv:,

2024. 4

- [50] Yilin Wang, Haiyang Xu, Xiang Zhang, Zeyuan Chen, Zhizhou Sha, Zirui Wang, and Zhuowen Tu. Omnicontrolnet: Dual-stage integration for conditional image generation. In CVPR Workshops, 2024. 3
- [51] Shuang Wu, Shijian Lu, and Li Cheng. Music-todance generation with optimal transport. arXiv preprint arXiv:2112.01806, 2021. 3
- [52] Haiyang Xu, Yu Lei, Zeyuan Chen, Xiang Zhang, Yue Zhao, Yilin Wang, and Zhuowen Tu. Bayesian diffusion models for 3d shape reconstruction. In CVPR, 2024. 3
- [53] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Luc Van Gool, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 2, 3, 6, 7, 1
- [54] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In CVPR, 2024. 2, 3
- [55] Wang Xuanchen, Wang Heng, Liu Dongnan, and Weidong Cai. Dance any beat: Blending beats with visuals in dance video generation. In WACV, 2025. 3
- [56] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [57] Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa, Kurt Keutzer, and Michael Mahoney. Adahessian: An adaptive second order optimizer for machine learning. In AAAI,

2021. 6

- [58] Hongwei Yi, Hualin Liang, Yifei Liu, Qiong Cao, Yandong Wen, Timo Bolkart, Dacheng Tao, and Michael J Black. Generating holistic 3d human motion from speech. In CVPR, pages 469–480, 2023. 4
- [59] Canyu Zhang, Youbao Tang, Ning Zhang, Ruei-Sung Lin, Mei Han, Jing Xiao, and Song Wang. Bidirectional autoregessive diffusion model for dance generation. In CVPR,

2024. 3

- [60] Chenxu Zhang, Chao Wang, Yifan Zhao, Shuo Cheng, Linjie Luo, and Xiaohu Guo. Dr2: Disentangled recurrent representation learning for data-efficient speech video synthesis. In WACV, 2024. 3
- [61] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In CVPR, 2023. 3

- [62] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 3, 5
- [63] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. IEEE TPAMI, 2024. 3
- [64] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 8
- [65] Yihao Zhi, Xiaodong Cun, Xuelin Chen, Xi Shen, Wen Guo, Shaoli Huang, and Shenghua Gao. Livelyspeaker: Towards semantic-aware co-speech gesture generation. In ICCV,

- 2023. 3

[66] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. arXiv preprint arXiv:2403.14781,

- 2024. 2, 3

- [67] Wenlin Zhuang, Congyi Wang, Jinxiang Chai, Yangang Wang, Ming Shao, and Siyu Xia. Music2dance: Dancenet for music-driven dance generation. ACM TOMM, 2022. 3

## X-Dancer: Expressive Music to Human Dance Video Generation Supplementary Material

In the supplementary material, we provide additional details on our curated in-house dataset (Section 6). In Section 7, we provide a user study to further compare X-Dancer to all baselines. We discuss more use cases of X-Dancer in Section 8, and present some failure cases and discuss some limitations of our method (Section 9). For more dynamic visual results, please refer to our offline webpage.

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Figure 5. Human image animation after finetuning motion transformer with 30 dance videos of Subject Three.

### 6. Dataset

Our in-house video dataset is sourced from a third-party curation service, featuring monocular recordings of everyday dance performances by diverse individuals worldwide. To ensure quality, we filter out videos shorter than 5 seconds, those with low resolution or poor quality (HyperIQA score [40] below 40), videos containing multiple people or with a human bounding box smaller than 27.5% of the frame, and footage with non-static cameras (corner pixel standard deviation exceeding 20). After filtering, the dataset comprises 76,818 dance videos, totaling around 360 hours. All filtered videos are center-cropped to 896 × 512 resolution and resampled to 30 fps.

may exhibit weaker beat alignment, mostly frontal movements and less distinct genre characteristics, often leaning towards freestyle movement.

### 7. User Study

In addition to our quantitative and qualitative comparisons to various baseline methods provided in Section 4.2 of our main paper, including Hallo [53], Bailando [38] + PoseGudier, and EDGE [44] + PoseGuider. We conduct a user study to further assess perceptual quality, motion fidelity and consistency with the music across all methods.

To analyze the dataset, we employ the state-of-the-art vision-language model (VLM) Qwen2.5-VL [3], yielding the following statistics:

We generated 25 dance videos for each method conditioned on randomly selected reference images with music tracks. Each video is 10 seconds long, recorded at 30 FPS, totaling 300 frames per video. Evaluation was conducted using a questionnaire distributed via Google Sheets. Participants were asked to choose the best-performing video among candidates based on five specific metrics: (1) Human Identity Consistency, (2) Music Beat Alignment, (3) Music Style Alignment, (4) Motion Consistency and Naturalness, and (5) Overall Quality. In total, 375 responses were gathered from 15 independent participants with no prior knowledge of video generation techniques. For each question sample, the videos from different methods were randomly permuted. The reference images were also provided to assist participants in assessing identity preservation.

- • Gender distribution: 78% female, 22% male.
- • Age distribution: 68% of dancers are classified as young, with the rest as middle-aged or older.
- • Ethnicity distribution: 49% White, 26% Asian, 11.2% Latino, 9% Black, with the rest others.
- • Video duration distribution: 10.04% of videos are between 5-10 seconds, 33.06% between 10-15 seconds, 49.72% between 15-20 seconds, 6.25% between 20-25 seconds, and 0.94% exceed 25 seconds.
- • Recording environment: 88% of videos are recorded indoors.
- • Dance styles: 89% are categorized as freestyle/hip-hop, while the remaining consist of popping and locking.
- • Motion characteristics: 52% feature medium-level movements (e.g., hip swaying, arm waving), while 47.4% exhibit strong motion variations (e.g., leg lifts, body rotations, and translations). 96% of videos do not contain large body turning.

Our method substantially surpasses all the baselines across different metrics, as evidenced in Table 5.

### 8. More Visual Results.

Please refer to our webpage for all the dynamic visual results.

Different from the AIST [45] and FineDance [26] dataset which capture professional dancers in a multiview setup, our dataset consists of monocular recordings of everyday individuals, offering wider accessibility and greater diversity in both dance motions and identity features. However, compared to professional dancers, our curated performances

Single Reference, Multiple Music. Given a single reference image, X-Dancer demonstrates the ability to generate diverse and expressive dance motions, maintaining consistent movement styles that adapt to different music genres and beat flows. This underscores X-Dancer’s capability to

Table 5. User study. The values indicate user preference ratios (%).

##### Method ID Consistency Beat Alignment Style Alignment Motion Consistency Overall

Hallo [53] 18.67 2.13 1.60 1.87 1.87 Bailando [38]+PG 3.73 10.93 6.40 3.73 3.47 EDGE [44] + PG 4.80 24.80 17.87 11.47 9.87

##### X-Dancer 72.80 52.13 74.13 82.93 84.80

effectively interpret the global music context while synchronizing seamlessly with local rhythmic beats.

Single Music, Multiple References. We present diverse dance videos generated by X-Dancer for various reference images, all driven by the same music track. While maintaining a consistent dance style synchronized to the shared music, each generated video also reflects the personalized attributes derived from its corresponding reference image, showcasing X-Dancer’s adaptability and attention to individual identity details.

Single Music, Single Reference. We demonstrate a variety of dance movements generated from a single reference image, all driven by the same music track. While all dance movements are well-aligned with the music beats, our model exhibits the ability to produce diverse and dynamic dance motions, highlighting its versatility and creativity.

Finetuning for Characterized Choreography. While our method operates as a zero-shot pipeline, generalizing seamlessly to new reference images and music inputs, it can also be fine-tuned for characterized choreography using only a few sample dance videos. This adaptability is challenging for 3D motion generation models like EDGE [44] and Bailando [38], which require intricate multiview captures or extensive effort in creating 3D dance movements. As shown in Fig. 5 and our supplementary video, our method successfully captures and mimics the specific choreography after fine-tuning with only 30 dance videos from diverse performers, showcasing its efficiency and versatility in adapting to specific dance styles.

Additional Results. We provide additional results including baseline comparisons and results of X-Dancer-AIST (discussed in line 449 - 455 of the main paper).

### 9. Limitations and Failure Cases.

In addition to the limitations and future work discussed in Section 5 of our main paper, we would like to discuss some additional limitations and failure cases. Specifically, noticeable rendering artifacts remain observable, particularly in the face and hands, and color flickering or over-saturation can occasionally occur. However, as indicated in Table 4 of the main paper (GT Pose + PG), these artifacts persist even when using ground-truth poses, and the overall video quality improvement remains marginal. Such issues are chal-

lenges across human image animation models and are orthogonal to the core contributions of our work . Future work could explore stronger video diffusion base models such as [23, 34, 56], and incorporate specialized embeddings or priors for these fine-grained regions [33] to mitigate these problems. Furthermore, while our method effectively generates motions informed by the identity and shape of the reference image, there are cases where the generated motions become inconsistent with the background scene or reference identity. This misalignment can lead to unnatural motions and, in severe cases, pronounced rendering artifacts (Figure 6).

[Figure 130]

Figure 6. Failure cases. Please note the rendering artifacts on the face and hands, as well as the unnatural renderings under challenging poses.

