## Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation

Bowen Xue1,∗ Zheng-Peng Duan2,1,∗ Qixin Yan1,# Wenjing Wang1 Hao Liu1 Chun-Le Guo2 Chongyi Li2 Chen Li1 Jing LYU1 1WeChat Vision, Tencent Inc. 2VCIP, CS, Nankai University

{bowenxue2005,adamduan0211}@gmail.com, {qixinyan,augustawang,leweshaoliu,chaselli,eckolv}@tencent.com, {guochunle,lichongyi}@nankai.edu.cn, https://github.com/WeChatCV/Stand-In

# arXiv:2508.07901v4[cs.CV]20Mar2026

Identity-Preserving Text-to-Video Generation

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|
|---|---|---|---|

[Figure 5]

“A young man, a streamer, is wearing a green sleeveless top and red headphones. The background is illuminated by vibrant neon lights. The setting is a well-lit room with a curtain and a lamp visible in the background. His expression and body language suggest that he is speaking passionately into the microphone.”

|[Figure 6]|[Figure 7]|[Figure 8]|[Figure 9]|
|---|---|---|---|

[Figure 10]

“A woman is sitting in front of a pottery wheel, her hands covered in wet clay. She pauses her work and looks up at the camera, her face beaming with a proud smile as she displays the pottery she has just shaped. In the background, shelves are filled with ceramic works and tools.”

Plug-and-Play Across Various Applications

Non-Human Subjects Video Stylization

[Figure 11]

|[Figure 12]|[Figure 13]|
|---|---|

|[Figure 14]|x<br><br>[Figure 15]|[Figure 16]|
|---|---|---|

[Figure 17]

|[Figure 18]|
|---|

LoRA: The Tale of the Princess Kaguya

Pose-Guided Video Generation

Video Face Swapping

[Figure 19]

|[Figure 20]|
|---|

| |
|---|

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

VACE

Ours

|[Figure 26]|
|---|

[Figure 27]

[Figure 28]

[Figure 29]

Figure 1. Given a reference image, our method generates videos with strong identity preservation. Furthermore, the framework’s plug-andplay design enables seamless integration into diverse applications for enhanced identity consistency.

### Abstract

In, a lightweight and plug-and-play framework for identity preservation in video generation. Specifically, we introduce a conditional image branch into the pre-trained video generation model. Identity control is achieved through restricted self-attentions with conditional position mapping. Thanks to these designs, which greatly preserve the pretrained prior of the video generation model, our approach is able to outperform other full-parameter training methods in video quality and identity preservation, even with just

Generating high-fidelity human videos that match userspecified identities is important yet challenging in the field of generative AI. Existing methods often rely on an excessive number of training parameters and lack compatibility with other AIGC tools. In this paper, we propose Stand-

∗ Equal Contribution. # Corresponding Author.

∼1% additional parameters and only 2000 training pairs. Moreover, our framework can be seamlessly integrated for other tasks, such as subject-driven video generation, posereferenced video generation, stylization, and face swapping.

| | | | | | | | | | | | | | | | | | | |St|a|nd|-|In|(|O|ur|s|)|0.|1|5B| | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |P|ha 1|[Figure 30]<br><br>n 4<br><br>[Figure 31]|to B|m| | | | | | | | |A|[Figure 32]<br><br>C|E|3|B| | | | | | | |
| | |[Figure 33]<br><br>P<br><br>[Figure 34]|h<br><br>1|an .3|to B|m| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | |[Figure 35]| |k|y<br><br>[Figure 36]|Re 1|e 4B|ls|[Figure 37]<br><br>-A|2|C|[Figure 38]<br><br>[Figure 39]<br><br>H u|u st|n om|yu|a 1|n 3B| | | | | | | | | |
|[Figure 40]|C|on|s 5<br><br>[Figure 41]|is B|tID| | | | | | | | | | | | | | |[Figure 42]| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

3.9

Naturalness↑

3.7

3.5

3.3

### 1. Introduction

3.1

With the rapid advancement of diffusion models [11, 28,

0.4 0.45 0.5 0.55 0.6 0.65 0.7

- 30], video generation [12, 17, 29, 52] has become a pivotal aspect of generative AI. Among its diverse applications, identity-preserving video generation holds profound significance. The goal of this task is to generate high-quality videos that consistently maintain the identity of a given reference image containing a face. It has widespread utility across film, advertising and gaming industries, etc.

Face Similarity ↑

Figure 2. Comparison with SOTA identity-preserving video generation methods. The size of bubbles represents the number of need-to-train parameters for identity preservation. Our approach achieves the highest performance in both face similarity and naturalness, while utilizing the fewest parameters.

Existing methods can be roughly classified into two categories. Early methods [10, 50] use an explicit face encoder for identity feature extraction, while recent methods [15, 21] fully train the diffusion transformer. However, face-encoder-based methods lack flexibility and struggle to capture fine facial details essential for high-quality video generation. The full-parameter training methods require huge training resources and lack compatibility with other applications. Achieving robust identity preservation in a lightweight way remains critical yet challenging.

tity preservation, video quality, and prompt following.

- • To inject identity information without explicit face feature extractors, we introduce a conditional image branch to the video generation model. The image and video branches share information through restricted self-attention with conditional position mapping. With these designs, identity preservation can be learned well with a small dataset.
- • The proposed framework exhibits high compatibility and generalizability. Although trained only on real-people data, our method generalizes to other subjects, such as cartoons and objects. Moreover, our method can be plugand-play applied to other tasks, such as pose-guided video generation, video stylization and face swapping.

To overcome these limitations, we leverage the pretrained VAE from the video generation model itself, enabling the conditional image to be mapped directly into the same latent space as the video. This approach naturally utilizes the model’s inherent capabilities to extract rich and detailed facial features, offering a more integrated and effective solution. Specifically, we employ restricted selfattention with conditional position mapping to merge the features of the reference image into the video. On the one hand, by preserving the core functionality of self-attention and the pretrained prior of the video generation model, our method achieves the highest facial similarity and naturalness in identity-preserving video generation with only ∼1% additional parameters and 2000 training pairs, which is shown in Figure 2. On the other hand, our method does not alter the architecture of the main video generation model and thus can be used in a plug-and-play manner for other applications. As shown in Figure 1, our framework can be extended to various tasks, including subject-driven generation, video stylization, and face swapping, all guaranteeing identity consistency. Additionally, benefit from integrating compatibility with VACE [16], our approach significantly enhances facial similarity in pose-guided video generation. Our main contributions can be summarized as follows:

### 2. Related Work

Video Generation Current video generation models are predominantly built on diffusion frameworks [11], with an evolution in architecture from U-Net-based designs [1] to DiT-based approaches [2, 17, 19, 24, 25, 38, 40]. In the era of U-Net-based diffusion models, text-to-image (T2I) frameworks [30, 32] were extended to video generation by introducing 3D convolutions and temporal attention [1]. AnimateDiff [6] further advanced this direction by reusing pre-trained text-to-image model weights to leverage their strong spatial generation capabilities by adding temporal layers. Latte [25] introduced a spatial-temporal separation mechanism, assigning distinct DiT blocks to process spatial and temporal information independently. This approach was later replaced by 3D full attention mechanisms, which offered more integrated processing. CogVideoX [45] and HunyuanVideo [17] combined 3D-VAE [48] with MM-DiT [4] to enhance video generation capabilities. WAN2.1 [38] employs a 3D-VAE and adopts a DiT backbone for denoising, injecting semantic information into the diffusion process through cross-attention.

• We present Stand-In, a lightweight and plug-and-play framework designed for identity-preserving video generation. By incorporating and training just ∼1% additional parameters, our approach achieves SOTA results in iden-

Identity-Preserving Generation Prior to the advent of zero-shot identity-preserving algorithms, generating content with consistent identity typically relied on case-by-case fine-tuning.[5, 13, 18, 33, 41, 43, 44], In contrast, trainingfree identity-preserving image generation methods[7–9, 27,

[Figure 43]

|[Figure 44]|
|---|

[Figure 45]

[Figure 46]

Original Video Branch

Conditional Image Branch

[Figure 47]

Latent noise

VAE

[Figure 48]

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |

- 31, 34, 35, 39, 46] offer a zero-shot personalization solution by integrating identity features into pre-trained foundation models. These methods typically introduce parameterized plug-in modules or adapters to adjust and inject identity features into the generation process. A popular solution is to use a Face Encoder to extract face embeddings and inject

Timestep (s, sref)

DiT Block

|Restricted Self-Attention|
|---|

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Text Tokens

|Others Layers<br><br>[Figure 53]|
|---|

…

VAE

[Figure 54]

- them into the generation process via cross-attention. For example, InstantID[39] and PuLID[7] can generate highquality, identity-preserving images.

| | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Discard

Figure 3. The overview of our identity-preserving text-to-video generation framework. We introduce a conditional image branch alongside the original video branch. Given the conditional image, the VAE encoder maps it into tokens, which are concatenated with the video latent tokens and then sent to the DiT. Within the DiT blocks, identity information is incorporated into the video features through restricted self-attention.

In the field of identity-preserving video generation [23, 51], early methods commonly rely on explicit face encoders for facial feature extraction to generate videos with identitypreserving. ID-animator [10] leverages a pre-trained textto-video diffusion model in conjunction with a lightweight face adapter to encode ID-relevant embeddings from adaptable facial latent queries. ConsistID [50] aims to maintain identity consistency through frequency decomposition in diffusion transformer. Phantom [21] can also preserve identity consistency in the human domain as a unified subjectconsistent video generation framework. HunyuanCustom [15] is a multi-modal customized video generation framework that emphasizes identity consistency while supporting diverse input modalities. By introducing advanced condition injection mechanisms and identity-preserving strategies, it achieves excellent performance in high-quality video generation. They employed a full fine-tuning for the diffusion transformer, requiring huge training resources.

it into the latent space using the pre-trained VAE encoder. The image latents undergo the same patchification and encoding procedures as the video latents. Then, the image tokens are concatenated with video tokens along the sequence dimension and processed jointly through successive blocks. Finally, image tokens are discarded at the final layer.

To preserve the static nature of the reference image, which serves as a conditioning input rather than undergoing denoising, we maintain its temporal invariance. This is done by fixing its timestep to zero, i.e. sref = 0, where s denotes the denoising timestep in diffusion. Now that we have encoded the conditional image into the same feature space as the video, the next challenge is: How can the video features effectively refer to the image information in a way that is lightweight and easy to learn?

### 3. Method

In this section, we first introduce the overall framework of the proposed method in Section 3.1. Next, we detail the restricted self-attention mechanism in Section 3.2 and conditional position mapping in Section 3.3. Finally, we present the data collection process in Section 3.4.

#### 3.2. Restricted Self-Attention

In the aforementioned DiT blocks, reference image and video tokens are processed independently through most modules (including layer normalization, cross-attention, and feed-forward networks), except for the self-attention layer. The self-attention layer enables information exchange among all tokens, naturally allowing video tokens to refer to identity information. One intuitive solution is to direct concatenate the reference-image tokens with the generated video tokens, and pass them through the Vanilla SelfAttention. However, this approach has two main drawbacks. First, since the reference image serves as a static condition, it should remain unaffected by the dynamic contents of the video. Vanilla Self-Attention lets image queries attend to video contents, making it challenging to maintain the identity. Second, this joint self-attention provides no guarantee that video tokens will actually refer to the image tokens. The

- 3.1. Overall Framework To extract facial features, early methods [10, 50] rely on explicit face encoders, which lack flexibility and often fail to preserve fine facial details critical for high-quality reconstruction. In contrast, we propose using the pre-trained VAE from the video generation model. This strategy maps the conditional image directly into the same latent space as the video and allows us to naturally take advantage of the builtin ability of the pre-trained video generation model to extract rich facial features.

The overall framework is illustrated in Figure 3. We use Wan2.1-14B-T2V [38] as the video generation base model, which adopts a Diffusion Transformer (DiT) architecture. Given a reference image containing a face, we first encode

Conditional Position Mapping

Frozen Trainable

[Figure 55]

[Figure 56]

′

KI

′ ′

height pI

TI

QI KI VI

QI KI

|QKV Proj.<br><br>[Figure 57]|
|---|

| | | | |
|---|---|---|---|

[Figure 58]

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

′

OutI

QI

VI

pV

LoRA

[Figure 59]

′ ′

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |
| |

KV KI

QV KV VV width frame

′ ′

QV KV

TV

VV

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |

Q I′ = Q I · p I ; K I′ = K I · p I Q′ V = Q V · p V ; K V′ = K V · p V

|QKV Proj.<br><br>[Figure 60]| |
|---|---|
| | |

′

OutV

QV

VI

Figure 4. Design of our Restricted Self-Attention: For the input video and image tokens, we compute their Query, Key, and Value matrices independently. Next, we apply Conditional Position Mapping to the Query and Key matrices. Finally, the image matrices operate independently, while the video Query performs attention using the concatenation of the image and video Key and Value matrices.

[Figure 61]

[Figure 62]

[Figure 63]

model may ignore the reference image and generate scenes without the target identity, which is shown in Figure 5.

Therefore, to incorporate identity information while preserving its independence, we propose replacing the vanilla self-attention layers in DiT with a restricted version that explicitly prevents image queries from attending to video keys. As shown in Figure 4, for a self-attention layer, we first independently compute the Query, Key, and Value for image and video tokens, denoted as QI, KI, VI and QV , KV , VV respectively. Then, we concatenate KV with KI and VV with VI for QV . To enhance the model’s ability to utilize identity-related information while preserving its inherent generative robustness, we incorporate Low-Rank Adaptation (LoRA) into the QKV projection of image tokens. For analysis, we visualize attention maps specifically for video queries attending to the reference image by averaging attention maps across all DiT blocks. As shown in Figure 5, in contrast to the vanilla baseline, our Restricted Self-Attention concentrates attention on facial regions of the reference image and yields prompt-faithful frames that preserve the subject’s identity.

Ref. Img Vanilla Self-Attention

[Figure 64]

[Figure 65]

A man gently clutching a bouquet of vibrant flowers … The scene is set in a lush garden,

brimming with colorful

blooms and verdant foliage …

Prompt

Restricted Self-Attention (Ours)

Figure 5. Effect of Restricted Self-Attention (RSA). Given the reference image and the prompt (left column), we visualize attention map to reference-image tokens. Under Vanilla Self-Attention (top row), attention diffuses into background regions and the output skews toward a garden scene. With RSA (bottom row), attention concentrates on facial regions, maintaining the subject’s identity.

ral index of -1 to the reference image tokens, while mapping video tokens to nonnegative temporal positions. This assignment establishes image tokens as temporally invariant conditional inputs. In this way, the model is encouraged to treat the identity information from the reference image as a constant guide throughout the entire denoising process, rather than conflating it with transient, frame-specific features in the temporal sequence.

Given that the timestep for the conditional image is fixed at sref = 0, its Key and Value matrices remain constant throughout the diffusion denoising process. Therefore, during inference, we can cache KI and VI to accelerate computation, named KV Caching. These matrices are computed and stored during the first denoising step, eliminating the need for redundant recalculations in subsequent steps.

For spatial dimensions, we employ a disjoint coordinate strategy to enforce spatial decoupling between reference image and video content. While video frames occupy coordinates within the domain (h,w) ∈ [0,HV ) × [0,WV ), we map reference image tokens to a dedicated coordinate subspace [HV ,HV + HI) × [WV ,WV + WI), where HI and WI represent the reference image dimensions.

- 3.3. Conditional Position Mapping To effectively differentiate image and video tokens in the restricted self-attention, we use a specialized conditional position mapping strategy. Specifically, we employ 3D Rotary Positional Embedding (RoPE) [36], where all tokens associated with the reference image are assigned a distinct and dedicated coordinate space. This ensures clear separation and facilitates precise modeling of interactions between the reference image and video tokens.

Denoting pI as the coordinate for image tokens and pV for video tokens, we apply 3D RoPE as follows:

Q′I = QI · pI, KI′ = KI · pI, (1) Q′V = QV · pV , KV′ = KV · pV . (2)

For the temporal dimension, we assign a fixed tempo-

where · denotes the Hadamard product. The restricted self-

|[Figure 66]|
|---|

|[Figure 67]|[Figure 68]|[Figure 69]|
|---|---|---|

[Figure 70]

[Figure 71]

[Figure 72]

A woman with blonde hair and glasses si ts in a room with a piano in the backgrou nd, speaking to the camera as her facial expressions change. She wears a blue sle eveless top and is illuminated by soft, na tural lighting. The camera is focused on her face, capturing her reactions and em otions.

|[Figure 73]|
|---|

|[Figure 74]|[Figure 75]|[Figure 76]|
|---|---|---|

A man with a beard sits on a bench in a park, wearing a green and white striped s hirt and speaking to another man in a blu e shirt. The speaker uses hand gestures a s he talks, while the background features

trees and grass under natural lighting, cr eating a calm and relaxed atmosphere. T he is shot using a mix of medium shots a nd close-ups, with a realistic style.

Ref. Img Shared Position Mapping Conditional Position Mapping (Ours)

Ref. Img Video

Prompt

Figure 6. Effect of Conditional Position Mapping (CPM). Compared with Shared Postion Mapping, our CPM, where the reference tokens are mapped to a disjoint spatial space, better preserves the pretrained positional prior and yields more stable scenes.

Figure 7. Examples from our human-centric video dataset.

Table 1. Quantitative comparison with state-of-the-art identity-preserving video generation methods. We evaluate across three key metrics: Face Similarity, Naturalness, and Prompt Following. For all metrics, higher values indicate better performance. The best and second-best results in each column are highlighted in bold and underlined, respectively.

attention outputs are computed as:

OutI = Attention(Q′I,KI′,VI), (3)

OutV = Attention(Q′V ,[KV′ ,KI′],[VV ,VI]), (4) where [·,·] denotes concatenation.

Prompt-

FaceSimilarity ↑ Naturalness ↑

Method

Following↑ Closed-Source Methods

This non-overlapping spatial allocation achieves two main goals through geometric separation. By separating the reference tokens from the video coordinate grid, the design naturally reduces false spatial correlations and better preserves the backbone’s pretrained positional prior. Compared with Shared Position Mapping, our Conditional Position Mapping generates more reliable video with more stable proportions, which is shown in Figure 6. At the same time, this separate coordinate system maintains the reference image’s semantic meaning by keeping it as a global identity prior. Consequently, the model is guided to focus on extracting overall semantic features from the reference tokens, rather than treating them as spatially localized patterns that need to align positionally with the video content.

Kling 0.410 3.900 19.921 Hailuo 0.577 3.750 20.649 Pika-2.1 0.323 3.644 20.649 Vidu-2.0 0.361 3.600 18.998

Open-Source Methods

ID-Animator [10] 0.316 3.211 16.677 SkyReels-A2-P14B [37] 0.546 3.411 19.110 EchoVideo [42] 0.487 3.456 19.263 ConcatID-CogVideoX [53] 0.439 3.372 19.359 ConcatID-WAN [53] 0.501 3.650 19.671 Hunyuan-Custom [14] 0.622 3.367 19.853 ConsistID [50] 0.432 3.233 20.552 VACE-P1.3B [16] 0.180 3.567 20.591 VACE-1.3B [16] 0.223 3.611 20.527 VACE-14B [16] 0.647 3.728 19.520 Phantom-1.3B [21] 0.440 3.567 20.364 Phantom-14B [21] 0.519 3.828 20.476 Stand-In (Ours) 0.724 3.922 20.594

- 3.4. Dataset Collection and Processing We construct a human-centric video dataset containing 2,000 high-resolution sequences from publicly available sources. The dataset guarantees a diverse and comprehensive representation, comprising various ethnic groups, age ranges, gender identities, and a wide array of actions. Using the VILA [20] multimodal captioning framework, we automatically generate dense textual annotations for each video, establishing strong text-video alignment.

4. BiSeNet [47] is used for face parsing, and the background is replaced with a solid white color to prevent any leakage of background information.

Examples of the final image-text-video pairs for our training can be found in Figure 7.

### 4. Experiments

To align the dataset with the pre-training distribution of our video generation base model [38] and to mitigate potential degradation in generation quality, we preprocess the videos as follows: each video is resampled to 25 FPS,

#### 4.1. Implementation Details

We adopt LoRA with rank 128, applied only to the QKV projections for image tokens in each DiT block. For the 14B-parameter Wan2.1 model, this adds just 153M trainable parameters (1% of the base model), increasing feedforward time by 3.6% and FLOPs by 2.6%. During inference with KV caching, overhead is minimal: runtime rises by only 2.3% and FLOPs by 0.07% compared to the video generation base model. This negligible cost shows our identity-preserving method is lightweight.

- then cropped and resized to a resolution of 832×480 pixels. From these processed videos, we randomly sample clips of 81 consecutive frames for training.

For each video clip, the corresponding reference facial image is extracted from the original, pre-resampled video. The extraction pipeline is as follows:

- 1. 5 frames are randomly selected from the original video.
- 2. Faces are detected and cropped using RetinaFace[3].
- 3. The cropped face images are resized to 512×512 pixels.

The model is trained over 3000 steps on Nvidia H20

|[Figure 77]|
|---|

|[Figure 78]|
|---|

“The video features a man standing at an easel, focused intently as his brush dances across the canvas. His expression is one of deep concentration, with a hint of satisfaction as each brushstroke adds color and form. He wears a paint-splattered apron, and his hands move with confident precision…”

“The video features a man with dark-haired hair, wearing a blue tank top and holding a pink tank top on a hanger. he appears to be in a clothing store or a similar retail environment, as there are racks of clothes visible in the background. The man is speaking to the camera, possibly providing a review or discussing the tank top he is holding…”

&Prompt

CustomVACE-14BOurs Ref.Img

|[Figure 79]| |[Figure 80]<br><br>[Figure 81]|
|---|---|---|

|[Figure 82]|[Figure 83]|[Figure 84]|
|---|---|---|

Kling

|[Figure 85]|[Figure 86]|[Figure 87]|
|---|---|---|

|[Figure 88]|[Figure 89]|[Figure 90]|
|---|---|---|

ConsistID Phantom-

|[Figure 91]|[Figure 92]|[Figure 93]|
|---|---|---|

|[Figure 94]|[Figure 95]|[Figure 96]|
|---|---|---|

14B Hunyuan-

|[Figure 97]|[Figure 98]|[Figure 99]|
|---|---|---|

|[Figure 100]|[Figure 101]|[Figure 102]|
|---|---|---|

|[Figure 103]|[Figure 104]|[Figure 105]|
|---|---|---|

|[Figure 106]|[Figure 107]|[Figure 108]|
|---|---|---|

|[Figure 109]|[Figure 110]|[Figure 111]|
|---|---|---|

|[Figure 112]|[Figure 113]|[Figure 114]|
|---|---|---|

Figure 8. Comparison on identity-preserving video generation. Please refer to the supplementary material for full prompts.

Table 2. User study results for subjective evaluation. The best and second-best results in each column are highlighted in bold and underlined, respectively.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

##### Method Face Similarity ↑ Video Quality ↑

[Figure 121]

[Figure 122]

[Figure 123]

Hunyuan-Custom[14] 3.34 2.92 VACE-14B [16] 3.00 3.07 Phantom-14B [21] 2.37 2.92 ConsistID [50] 2.25 2.46 Kling 2.21 3.09 Stand-In (Ours) 4.10 4.08

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Figure 9. Our model generalizes to unseen ordinary individuals across diverse ethnicities and age groups, despite being trained with only ∼1% additional parameters and just 2000 training pairs.

GPUs with a batch size of 48. For inference, BiSeNet is adopted as an automatic preprocessing step.

#### 4.2. Quantitative Analysis

trained with only ∼1% additional parameters and just 2000 training pairs, our model also demonstrates strong generalization across individuals of diverse ethnicities and age groups.

To evaluate identity preservation and visual quality, we use the two most important and heavily weighted evaluation metrics from the OpenS2V benchmark [49]: facial similarity and naturalness. To evaluate the relevance between generated video and textual description, we use X-CLIP [26], a pre-trained video-text multimodal model. Results are shown in Table 1 and Figure 8. As shown in Figure 9, despite being

Face Similarity This metric evaluates the model’s ability to maintain identity consistency. It is calculated as the av-

“…a girl, approximately seven or eight years old, stands in the center. She has long black hair and wears a light blue dress, her expression focused and gentle. Holding a doll in both hands, she presents her beloved toy to the camera. As the camera slowly zooms in…”

“…an anime girl standing on a busy street, surrounded by a hurried crowd. The buildings and shops in the background create a classic cityscape. The girl smiles as she puts her headphones on, her movements smooth and natural. Her expression is playful and relaxed…”

|[Figure 133]|[Figure 134]|[Figure 135]|[Figure 136]|
|---|---|---|---|

|[Figure 137]<br><br>[Figure 138]|[Figure 139]|[Figure 140]|[Figure 141]|
|---|---|---|---|

Ref. Img Ours Ref. Img Ours

Figure 10. Our results on subjects other than real-person. Please refer to the supplementary material for full prompts.

|[Figure 142]|[Figure 143]|[Figure 144]|
|---|---|---|
|[Figure 145]|[Figure 146]|[Figure 147]|

|[Figure 148]|[Figure 149]|[Figure 150]|
|---|---|---|
|[Figure 151]|[Figure 152]|[Figure 153]|

“A woman walking on the road”

|[Figure 154]|
|---|

Ref. Img & Prompt VACE VACE w/ Ours

Figure 11. Comparison on pose-guided video generation against VACE.

outperforms the comparative methods.

erage cosine similarity between the CurricularFace embeddings of the reference image and the faces detected in the generated video frames. As shown in Table 1, our proposed method, Stand-In, achieves a score of 0.724, outperforming all other compared methods. This result demonstrates the effectiveness of our approach in generating facial features that remain highly consistent with the source identity.

#### 4.4. Plug-and-Play to Other Applications

Subject-Driven Video Generation Although trained only with human data, our framework can be zero-shot applied to non-human subjects without any additional finetuning. This is because we use the pretrained VAE and video generation model to extract rich features, and learn alignment through paired data and an effective attention mechanism. This zero-shot ability can hardly be achieved by traditional identity-preserving methods relying on face encoders. As shown in Figure 10, our method exhibits strong subject consistency on the teddy bear object and the cartoon character.

Naturalness This metric is primarily designed to evaluate the naturalness of the generated videos. It leverages GPT-4o

- to approximate human judgment of video realism, taking into account factors such as physical plausibility and the absence of noticeable AI artifacts. Following the OpenS2V protocol, a holistic score ranging from 1 to 5 is assigned. In this evaluation, our method achieves a score of 3.922, demonstrating that the improvements in identity fidelity are achieved without compromising the overall visual realism of the generated videos.

Pose-Guided Video Generation The proposed conditional image branch is designed based on the LoRA module, which ensures inherent compatibility with other DiTbased models. To validate this, we conducted experiments on the pose-guided video generation task using the VACE framework [16]. As illustrated in Figure 11, integrating our method significantly improves the facial identity similarity in the generated videos. This not only demonstrates the plug-and-play nature of our approach but also highlights its robustness in preserving identity consistency.

Prompt Following As shown in Table 1, our method ranks second among all compared methods and first among open-source methods. This result demonstrates that our identity-preserving ability can be achieved without hurting the prompt-following performance.

- 4.3. User Study The user study involves 20 participants. We randomly select 10 test videos from the benchmark and ask the participants to rate each video across two dimensions: facial similarity and video quality. The latter dimension encompasses aspects such as naturalness, visual aesthetics, and alignment with the provided text descriptions. Ratings are given on a 5-point scale (1 to 5), and the final scores for each dimension are obtained by averaging the ratings across all participants and test videos. As shown in Table 2, our method

Video Stylization By applying our framework in conjunction with video stylization LoRA, we demonstrate its ability to achieve effective style transfer while maintaining strong identity consistency. As shown in Figure 13, our method successfully renders the artistic style as well as preserving the facial features of the reference image, further showing its versatility and robustness.

|[Figure 155]|[Figure 156]|[Figure 157]|
|---|---|---|

|[Figure 158]|[Figure 159]|[Figure 160]|
|---|---|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|[Figure 164]|[Figure 165]|
|---|---|---|

|[Figure 166]|[Figure 167]|[Figure 168]|
|---|---|---|

Ref. Img Source Video Ours

Figure 12. Application of our model in video face swapping.

[Figure 169]

Table 3. Ablation study on the core components of our method. Replacing Restricted Self-Attention (RSA) with Vanilla SelfAttention (VSA) or Conditional Position Mapping (CPM) with Shared Position Mapping (SPM) degrades both Face Similarity and Naturalness.

“A woman sits on a boat, gazing at the camera with a gentle smile. Behind her is the endless sea, waves crashing against the side of the boat, and a lighthouse in the distance stands tall under the bright sky.”

Ref. Img & Prompt

[Figure 170]

[Figure 171]

[Figure 172]

Original

Method Face Similarity ↑ Naturalness ↑

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

RSA → VSA 0.422 3.855 CPM → SPM 0.536 3.755

LoRA: Ghibli

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Full Model 0.724 3.922

LoRA: Redline

with a Shared Position Mapping (SPM), which places reference and video tokens within the same positional coordinate system. As illustrated in Figure 6, SPM breaks the pretrained positional prior of the backbone model, resulting in unstable spatial layouts. By geometrically separating the coordinate spaces, CPM preserves the pretrained positional prior while maintaining the reference as a global identity prior. In Table 3, CPM increases face similarity from 0.536 to 0.724 and improves naturalness from 3.755 to 3.922, demonstrating its effectiveness in stabilizing video synthesis and enhancing identity fidelity.

Figure 13. Our model applied with stylization LoRA.

Video Face Swapping Our framework is also capable of video face swapping, which can be achieved via zero-shot inpainting [22]. Figure 12 shows that our method not only achieves high-quality facial identity transfer but also maintains strong temporal consistency across frames, resulting in coherent and high-quality videos.

#### 4.5. Ablation Study

Effectiveness of Restricted Self-Attention (RSA) To verify the role of RSA, we replaced it with Vanilla SelfAttention (VSA), which concatenates the reference-image and video tokens and allows full bidirectional information flow. Because VSA treats the reference tokens as part of the dynamic video context, the model often fails to refer to the reference image. As shown in Figure 5, this naive design leads to diffused attention and unreliable identity preservation. In contrast, RSA constrains attention so that image queries cannot attend to video tokens. This design keeps the reference representation static while still providing identity cues. As shown in Table 3, RSA improves face similarity from 0.422 to 0.724 and also slightly enhances naturalness, confirming its essential role in maintaining both identity consistency and visual quality.

### 5. Conclusion

We propose Stand-In, a lightweight, plug-and-play framework for high-fidelity, identity-preserving video generation. We introduce a conditional image branch into a pre-trained video generation model, and propose a restricted attention mechanism with conditional positional encoding to enable cross-branch information exchange. Despite training only 1% of the model’s additional parameters on a limited dataset of 2,000 pairs, our approach achieves high-quality video generation while maintaining strong identity fidelity. Experimental results demonstrate that Stand-In achieves stateof-the-art performance in identity-preserving text-to-video generation. Furthermore, it exhibits excellent performance on other tasks, including pose-guided video generation, stylization, and face swapping, proving its strong compatibility and broad application potential.

Effectiveness of Conditional Position Mapping (CPM) We further investigate the effect of CPM by Replacing CPM

### 6. Acknowledgment

This work was supported in part by the National Natural Science Foundation of China (62306153, 62225604), Tianjin Natural Science Foundation Project (25ZXRGGX00290, 24JCJQJC00020, 25JCQNJC01390), the Young Elite Scientists Sponsorship Program by CAST (YESS20240686), the Fundamental Research Funds for the Central Universities (Nankai University, 63253223, 63253219), and Shenzhen Science and Technology Program (JCYJ20240813114237048). The computational devices is supported by the Supercomputing Center of Nankai University (NKSC).

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. ArXiv preprint,

2023. 2

- [2] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Conference on Computer Vision and Pattern Recognition, 2024. 2
- [3] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multi-level face localisation in the wild. In Conference on Computer Vision and Pattern Recognition, 2020. 5
- [4] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024. 2
- [5] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In International Conference on Learning Representations, 2023. 3
- [6] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. In International Conference on Learning Representations, 2024. 2
- [7] Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. Pulid: Pure and lightning id customization via contrastive alignment. In Conference and Workshop on Neural Information Processing Systems, 2025. 3
- [8] Yue Han, Junwei Zhu, Keke He, Xu Chen, Yanhao Ge, Wei Li, Xiangtai Li, Jiangning Zhang, Chengjie Wang, and Yong Liu. Face-adapter for pre-trained diffusion models with finegrained id and attribute control. In European Conference on Computer Vision. Springer, 2024.
- [9] Junjie He, Yifeng Geng, and Liefeng Bo. Uniportrait: A unified framework for identity-preserving single-

- and multi-human image personalization. arXiv preprint arXiv:2408.05939, 2024. 3
- [10] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. ArXiv preprint,

2024. 2, 3, 5

- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Conference and Workshop on Neural Information Processing Systems, 2020. 2
- [12] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In International Conference on Learning Representations, 2023. 2
- [13] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 3
- [14] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. ArXiv preprint, 2025. 5, 6
- [15] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. ArXiv preprint, 2025. 2, 3
- [16] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. ArXiv preprint, 2025. 2, 5, 6, 7
- [17] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. ArXiv preprint, 2024. 2
- [18] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of textto-image diffusion. In Conference on Computer Vision and Pattern Recognition, 2023. 3
- [19] Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback. In Conference and Workshop on Neural Information Processing Systems, 2024. 2
- [20] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: on pre-training for visual language models. In Conference on Computer Vision and Pattern Recognition, 2024. 5
- [21] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via crossmodal alignment. ArXiv preprint, 2025. 2, 3, 5, 6
- [22] Andreas Lugmayr, Martin Danelljan, Andr´es Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Conference on Computer Vision and Pattern Recognition, 2022. 8
- [23] Boyao Ma, Yuanping Cao, and Lei Zhang. Decoupled twostage talking head generation via gaussian-landmark-based neural radiance fields. Computational Visual Media, 2025. 3

- [24] Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, 2024. 2
- [25] Xin Ma, Yaohui Wang, Xinyuan Chen, Gengyun Jia, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. Transactions on Machine Learning Research, 2025. 2
- [26] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. X-CLIP: end-to-end multi-grained contrastive learning for video-text retrieval. In ACM International Conference on Multimedia, 2022. 6
- [27] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, Mengtian Li, Mingcong Liu, Yi Zhang, Shaojin Wu, Songtao Zhao, Jian Zhang, Qian He, and Xinglong Wu. Dreamo: A unified framework for image customization, 2025. 3
- [28] William Peebles and Saining Xie. Scalable diffusion models with transformers. In International Conference on Computer Vision, 2023. 2
- [29] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, Yuhui Wang, Anbang Ye, Gang Ren, Qianran Ma, Wanying Liang, Xiang Lian, Xiwen Wu, Yuting Zhong, Zhuangyan Li, Chaoyu Gong, Guojun Lei, Leijun Cheng, Limin Zhang, Minghao Li, Ruijie Zhang, Silan Hu, Shijie Huang, Xiaokang Wang, Yuanheng Zhao, Yuqi Wang, Ziang Wei, and Yang You. Open-sora 2.0: Training a commercial-level video generation model in 200k. ArXiv preprint, 2025. 2
- [30] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024. 2
- [31] Guocheng Qian, Kuan-Chieh Wang, Or Patashnik, Negin Heravi, Daniil Ostashev, Sergey Tulyakov, Daniel CohenOr, and Kfir Aberman. Omni-id: Holistic identity representation designed for generative tasks. arXiv preprint arXiv:2412.09694, 2024. 3
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Conference on Computer Vision and Pattern Recognition, 2022. 2
- [33] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Conference on Computer Vision and Pattern Recognition, 2023. 3
- [34] Rinon Gal Daneil Cohen-Ro Sara Dorfman, Dana Cohen Bar. Ip-composer: Semantic composition of visual concepts. arXiv preprint arXiv:2502.13951, 2025. 3
- [35] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. In Conference on Computer Vision and Pattern Recognition, 2024. 3

- [36] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024. 4
- [37] SkyReels Team. Skyreels-v2: Infinite-length film generative model. ArXiv preprint, 2025. 5
- [38] Wan Team. Wan: Open and advanced large-scale video generative models. ArXiv preprint, 2025. 2, 3, 5
- [39] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. ArXiv preprint, 2024. 3
- [40] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, Yuwei Guo, Tianxing Wu, Chenyang Si, Yuming Jiang, Cunjian Chen, Chen Change Loy, Bo Dai, Dahua Lin, Yu Qiao, and Ziwei Liu. Lavie: High-quality video generation with cascaded latent diffusion models. ArXiv preprint, 2023. 2
- [41] Zhao Wang, Aoxue Li, Enze Xie, Lingting Zhu, Yong Guo, Qi Dou, and Zhenguo Li. Customvideo: Customizing textto-video generation with multiple subjects. ArXiv preprint,

2024. 3

- [42] Jiangchuan Wei, Shiyue Yan, Wenfeng Lin, Boyuan Liu, Renjie Chen, and Mingyu Guo. Echovideo: Identitypreserving human video generation by multimodal feature fusion. ArXiv preprint, 2025. 5
- [43] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dream video: Composing your dream videos with customized subject and motion. In Conference on Computer Vision and Pattern Recognition, 2024. 3
- [44] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. In Conference and Workshop on Neural Information Processing Systems, 2024. 3
- [45] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, 2025. 2
- [46] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [47] Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. Bisenet: Bilateral segmentation network for real-time semantic segmentation. In European Conference on Computer Vision, 2018. 5
- [48] Lijun Yu, Jos´e Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In International Conference on Learning Representations, 2024. 2

- [49] Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subjectto-video generation. ArXiv preprint, 2025. 6
- [50] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. In Conference on Computer Vision and Pattern Recognition, 2025. 2, 3, 5, 6
- [51] Chenxu Zhang, Chao Wang, Jianfeng Zhang, Hongyi Xu, Guoxian Song, You Xie, Linjie Luo, Yapeng Tian, Jiashi Feng, and Xiaohu Guo. Magictalk: Implicit and explicit correlation learning for diffusion-based emotional talking face generation. Computational Visual Media, 2025. 3
- [52] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. ArXiv preprint, 2024. 2
- [53] Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Chongxuan Li. Concat-id: Towards universal identitypreserving video synthesis. ArXiv preprint, 2025. 5

