## arXiv:2603.20192v1[cs.CV]20Mar2026

# LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation

Jiazheng Xing∗1,4,2, Fei Du∗2,3 Hangjie Yuan∗2,3,1, Pengwei Liu1,2, Hongbin Xu4, Hai Ci4, Ruigang Niu2,3, Weihua Chen†2,3, Fan Wang2, Yong Liu †1

1Zhejiang University, 2DAMO Academy, Alibaba Group, 3Hupan Lab, 4National University of Singapore

* Equal contribution, † Corresponding authors. jiazhengxing@zju.edu.cn, kugang.cwh@alibaba-inc.com, yongliu@iipc.zju.edu.cn Project Page: https://jiazheng-xing.github.io/lumosx-home/

Recent advances in diffusion models have significantly improved text-to-video generation, enabling personalized content creation with fine-grained control over both foreground and background elements. However, precise face–attribute alignment across subjects remains challenging, as existing methods lack explicit mechanisms to ensure intra-group consistency. Addressing this gap requires both explicit modeling strategies and face-attribute-aware data resources. We therefore propose LumosX, a framework that advances both data and model design. On the data side, a tailored collection pipeline orchestrates captions and visual cues from independent videos, while multimodal large language models (MLLMs) infer and assign subject-specific dependencies. These extracted relational priors impose a finer-grained structure that amplifies the expressive control of personalized video generation and enables the construction of a comprehensive benchmark. On the modeling side, Relational Self-Attention and Relational Cross-Attention intertwine position-aware embeddings with refined attention dynamics to inscribe explicit subject–attribute dependencies, enforcing disciplined intra-group cohesion and amplifying the separation between distinct subject clusters. Comprehensive evaluations on our benchmark demonstrate that LumosX achieves state-of-the-art performance in fine-grained, identity-consistent, and semantically aligned personalized multi-subject video generation.

[Figure 1]

[Figure 2]

Date: March 23, 2026

1 Introduction

In recent years, diffusion models [11, 12, 33] have driven remarkable progress, establishing new performance standards in text-to-video generation [13, 3, 28, 51], particularly through the adoption of Diffusion Transformer (DiT) architectures [31]. These advances have laid a solid groundwork for customized video generation [27, 48, 52, 5, 15, 8, 24], where high-degree-of-freedom personalization unlocks transformative applications ranging from virtual theatrical production to e-commerce—enabling fine-grained control over both backgrounds and foregrounds, including multiple interacting subjects. Yet, realizing open-set personalized multi-subject video generation under such flexible and complex conditions remains profoundly challenging. The task requires not only the precise integration of diverse and interrelated conditioning signals but also the preservation of temporal coherence and identity fidelity across all subjects.

In the realm of open-set personalized video generation, prior studies have pushed the field forward from distinct angles. Certain approaches [27, 10, 48, 50, 52] concentrate narrowly on foreground facial customization, preserving identity fidelity from reference images yet affording only limited flexibility in input specification. In contrast, more recent methods [5, 15, 8, 24] enable highly versatile multi-subject video personalization with controllable foregrounds and backgrounds, but they largely neglect the intrinsic dependency structures that govern multi-subject conditions. Crucially, during fine-grained multi-condition injection, conditioning signals for each subject are typically decomposed into facial exemplars and attribute descriptors (e.g., man: blond hair, white T-shirt, sunglasses). Absent an explicit mechanism to bind identity with its associated attributes, such formulations are inherently fragile and frequently yield attribute entanglement or face–attribute misalignment

man on the left

man on the right

Two men are seated next to each other in a casual indoor setting, captured from an eye-level camera angle. The man on the left has long, wavy hair and a beard, wearing a green button-up shirt. The man on the right is wearing a black t-shirt and a black baseball cap. Between them are two green glass bottles, placed on a table. Behind them, a framed painting with an abstract design hangs on a light-colored wall.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

LumosX

green tent park

man woman black shirt gray shirt

A man in a black shirt with graying hair and a woman in a gray shirt sit close to each other under a green tent in a park, engaged in a serious conversation.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

LumosX

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- Figure 1 LumosX supports flexible personalized multi-subject video generation.

across subjects.

Although implicit modeling via textual captions can capture simple multi-subject dependencies during video generation, ambiguity often arises when captions contain similar subject nouns, such as “A man on the left with ... and a man on the right with ...," leading to confusion in subject–attribute associations. To overcome this limitation, under fine-grained multi-subject inputs, explicit constraints must be imposed at both the data and model levels. (1) Data level: When visual references are provided, the correspondence between each face and its associated attributes should be clearly specified. (2) Model level: During generation, each face-attribute pair is explicitly bound into an independent subject group, with intra-group correlation enhanced and inter-group interference suppressed.

To address the challenge of modeling face-attribute dependencies in multi-subject video generation, we present LumosX, a novel framework for personalized multi-subject synthesis. On the data side, the absence of public datasets with annotated dependency structures motivates us to construct a collection pipeline that supports open-set entities. This pipeline extracts captions and foreground–background visual conditions from independent videos, while multimodal large language models (MLLMs) infer and assign subject-specific dependencies. In particular, it produces customized single- and multi-subject data with explicit face–attribute correspondences, which not only enhance personalization during modeling but also enable the construction of a comprehensive benchmark. On this basis, the benchmark further defines two evaluation tasks, identityconsistent and subject-consistent generation, which allow a systematic assessment of a model’s ability to preserve identity and align multi-subject relationships. On the modeling side, LumosX explicitly encodes face-attribute bindings into coherent subject groups through two dedicated modules: Relational Self-Attention and Relational Cross-Attention. The Relational Self-Attention module incorporates Relational Rotary Position Embedding (R2PE) and a Causal Self-Attention Mask (CSAM) to model dependencies at the positional encoding and spatio-temporal self-attention stages. In addition, the Relational Cross-Attention module introduces a Multilevel Cross-Attention Mask (MCAM), which reinforces intra-group coherence, suppresses cross-group interference, and refines the semantic representation of visual condition tokens. LumosX is built upon the Wan2.1 [40] text-to-video backbone, with our modules seamlessly integrated to support flexible and high-fidelity personalized multi-subject generation, as shown in Fig. 1. Extensive experiments demonstrate Lumos-X’s strong capability in producing fine-grained, identity-consistent, and semantically aligned personalized videos, achieving state-of-the-art results across diverse benchmarks.

The contributions of our LumosX can be summarized as follows:

- • Data Side. We build a collection pipeline for open-set multi-subject generation that extracts captions and foreground–background condition images with explicit face–attribute dependencies from independent videos. This yields finer-grained relational priors that enhance personalized video customization and enable the construction of reliable benchmarks.

- • Model Side. We introduce Relational Self-Attention and Relational Cross-Attention, which integrate relational positional encodings with structured attention masks to explicitly encode face–attribute bindings. This reinforces intra-group coherence, mitigates cross-group interference, and ensures semantically consistent multi-subject video generation.
- • Overall Performance. Through extensive experiments and comparative evaluations, LumosX achieves state-of-the-art results in generating fine-grained, identity-consistent, and semantically aligned personalized multi-subject videos, decisively outperforming advanced open-source approaches including Phantom and SkyReels-A2.

##### 2 Related Works

Video Generation. Video generation has advanced rapidly in recent years, becoming one of the most dynamic research areas. Early works based on generative adversarial networks (GANs) [39, 38] demonstrated initial video synthesis but struggled with temporal coherence and fidelity. Latent diffusion models (LDMs) [33], powered by UNet [34], marked a significant milestone by enabling high-quality video generation through denoising in compressed latent spaces. These works typically add a temporal module to an image generation model, such as Make-A-Video [36] and Animatediff [9]. However, these models often face scalability challenges when scaling to larger parameter sizes or higher resolutions. Diffusion Transformers (DiTs) [31], replacing the Unet backbone with Transformer blocks, have shown superior performance in visual generation. By incorporating spatio-temporal attention mechanisms, video DiTs achieved unprecedented performance in modeling long-range dependencies across both spatial and temporal dimensions, significantly enhancing video realism and consistency. Models like Hunyuan Video [20], Wan2.1 [40], and MAGI-1 [35] have scaled the parameters of video DiTs to more than 10 billion, achieving significant advancement. Despite these advances, controllability remains a critical bottleneck: text-driven generation often fails to precisely align with user intentions due to ambiguities in natural language descriptions. This work focuses on multi-subject video customization to address this limitation, enabling precise content-control video generation.

Multi-Subject Video Customization. In recent years, subject-driven video generation has attracted growing interest. Several works focus on ID-consistent video generation, such as Magic-Me [27], ID-Animator [10], ConsisID [48], Magic Mirror [49], FantasyID [50], and Concat-ID [52]. These works generate videos that show consistent identity with the reference images, mainly focusing on facial identity. For arbitrary subject customization, VideoBooth [17] incorporates high-level and fine-level visual cues from an image prompt to the video generation model via cross attention and cross-frame attention. DreamVideo [44] customizes both subject and motion, with motion extracted from a reference video. Although they have demonstrated capabilities in generating single-subject consistent videos, neither the data processing nor the model can be easily transferred to the more challenging multi-subject customization. CustomVideo [43] generates multisubject identity-preserving videos by composing multiple subjects in a single image and designs an attention control strategy to disentangle them. However, it requires test-time finetuning for different subjects. Recently, several works [5, 15, 8, 24] propose to customize multiple subjects in the video DiTs. Different subjects are usually concatenated and fed into the video DiT network without distinguishing between them. This lack of differentiation can lead to semantic ambiguity, especially when there are numerous targets and hierarchical relationships among them. In this work, we design several strategies to differentiate between various subjects and their hierarchical relationships, achieving consistent customization while ensuring harmony among different objectives and the text’s adherence capabilities.

##### 3 Methods

- 3.1 Preliminary

In this work, we bulid upon the latest text-to-video generative model, Wan2.1 [40], which comprises the 3D variational autoencoder (VAE) E, the text encoder T , and the denoising DiT [31] backbone ϵθ combined with Flow Matching [23]. Within the DiT architecture, full spatio-temporal Self-Attention is used to capture complex dynamics, while Cross-Attention is employed to incorporate text conditions. Specifically, given a video X = {xi}Ni=1 with N frames, E compresses it into a latent representation z ∈ RT×HW×C along the

###### Step 1 Step 2 Step 3

Subjects

###### Original Video

[Figure 19]

[Figure 20]

[Figure 21]

white top

woman

[Figure 22]

[Figure 23]

[Figure 24]

Background

Objects

###### Video Frames Extraction

[Figure 25]

lush garden

[Figure 26]

[Figure 27]

utensils

[Figure 28]

[Figure 29]

[Figure 30]

0.05 0.50 0.95

[Figure 31]

black watach

man

black shirt

[Figure 32]

[Figure 33]

[Figure 34]

Human Detection

Caption Generation

Random Sampling

Face Cropping +SAM

GroundingDINO + SAM

Subject & Object Masking +Inpainting

[Figure 35]

[Figure 36]

A man and a woman are seated at a table in a lush garden, sharing a meal. The man, wearing a black shirt and a black watch on his left wrist, gestures with his hands while speaking, while the woman, in a white top, listens attentively and occasionally responds. Utensils are set neatly on the tables as they enjoy their meal together...

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Subjects man: black shirt, black watch woman: white top

###### 0.05 0.50 0.95

###### Objects

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

utensils

###### Background

Entity Words Retrieval & Subject-Attributes Matching

lush garden

Word Tags

- Figure 2 Dataset construction pipeline for personalized multi-subject video generation. We build the training dataset from raw videos in three steps: (1) generate a caption and detect human subjects in extracted frames; (2) retrieve entity words from the caption and match subjects with their attributes; (3) use these entity tags to localize and segment target subjects and objects, producing a clean background image.

spatiotemporal dimensions, where T, HW, and C denote the temporal, spatial, and channel dimensions, respectively. The text encoder T takes the text prompt and encodes it into a textual embedding ctext. The denoising DiT ϵθ then processes the latent representation z and the textual representation ctext to predict the distribution of video content. Each DiT Block incorporates 3D Rotary Position Embedding (3D-RoPE [37]) within the full spatio-temporal attention module to better capture both temporal and spatial dependencies.

- 3.2 Dataset Construction

As illustrated in Fig. 2, our training dataset and inference benchmark for personalized multi-subject video generation are constructed from raw videos through the following three steps.

Caption Generation and Human Detection. To obtain richer textual descriptions for downstream tasks, we replace the original video captions with captions generated by the large vision–language model VILA [22]. We sample three frames from the beginning, middle, and end of each video (5%, 50%, and 95% positions) and apply human detection [41] to extract human subjects for subsequent face–attribute matching.

Entity Words Retrieval and Face-Attribute Matching. In this step, our goal is to retrieve entity words from the caption, which can be classified into three categories: human subjects with attributes (e.g., man: black shirt, black watch), objects (e.g., utensils), and background (e.g., lush garden). During this process, if multiple human subjects are present, we need to assign different attributes to the corresponding subjects. In particular, when the caption contains multiple instances of the same subject noun (e.g., woman), we rely on visual information to assist in distinguishing between them. Therefore, we employ the multimodal large language model Qwen2.5-VL [1] to retrieve multiple entity words from the caption, while leveraging prior visual information from human detection results to achieve precise face-attribute matching.

Obtaining Condition Images. For subjects, we apply face detection [41] within human detection boxes to extract face crops and use SAM [19] to segment attribute masks. For objects, GroundingDINO [25] combined with SAM segments each entity within the global image. For backgrounds, we remove subjects and objects using the crops and masks, then apply the diffusion inpainting model FLUX [21] to generate a clean background. Finally, from the valid results of the three key frames, we randomly select one per entity as its condition

[Figure 45]

[Figure 46]

[Figure 47]

###### Frozen Trainable

[Figure 48]

###### Caption

In a cozy kitchen setting, a woman in a yellow scarf is mixing ingredients in a large blue bowl, while a young woman in a black hoodie watches her attentively. A small metal measuring cup is also visible on the counter...

Width

|(0,0,0)|(0,0,1)|
|---|---|
|(0,1,0)|(0,1,1)|

|(T-1, 0,0)|(T-1, 0,1)|
|---|---|
|(T-1, 1,0)|(T-1, 1,1)|

[Figure 49]

3D VAE Decoder

Height

Video Tokens

###### ...

[Figure 50]

###### DiT Block

|(T,0,0)|(T,0,1)|
|---|---|
|(T,1,0)|(T,1,1)|

...

[Figure 51]

Background Tokens

Text Encoder

Feed Forward

Relational Cross-Attention

|(T+1, 0,0)|(T+1, 0,1)|
|---|---|
|(T+1, 1,0)|(T+1, 1,1)|

|(T+2, 0,0)|(T+2, 0,1)|
|---|---|
|(T+2, 1,0)|(T+2, 1,1)|

Subject A Subject B Objects

Relational Self-Attention

Objects Tokens

[Figure 52]

[Figure 53]

[Figure 54]

Relational Rotary Position Embedding (R2PE)

Background

[Figure 55]

[Figure 56]

...

|(T+3, 0,0)<br><br>(T+3, 0,1)|
|---|
|(T+3, 1,0)<br><br>(T+3, 1,1)|

|(T+3, 2,2)|(T+3, 2,3)|
|---|---|
|(T+3, 3,2)|(T+3, 3,3)|

[Figure 57]

[Figure 58]

[Figure 59]

- Subject A Tokens
- Subject B Tokens

[Figure 60]

DiT Block

|(T+4, 0,0)|(T+4, 0,1)|
|---|---|
|(T+4, 1,0)|(T+4, 1,1)|

|(T+4, 2,2)|(T+4, 2,3)|
|---|---|
|(T+4, 3,2)|(T+4, 3,3)|

[Figure 61]

3D VAE Encoder

Video Tokens

- Figure 3 Overview of LumosX. Built on the T2V model Wan2.1 [40], our framework encodes all condition images into image tokens via a VAE encoder, concatenates them with denoising video tokens, and feeds the result into DiT [31] blocks. Within each block, the proposed Relational Self-Attention and Relational Cross-Attention enable causal conditional modeling, enhance visual token representations, and ensure precise face–attribute alignment.

image—matching the inference process, where each condition uses a single reference image—while ensuring data diversity by preventing all selections from a single frame.

Through these three steps, we obtain the visual condition images for the subjects, objects, and background, along with their paired word tags derived from the input text caption. Note that a subject is defined as a single human face paired with its corresponding attributes. The face is expected to present clear facial features without significant occlusion, and the associated attributes can include clothing (top or bottom), accessories (e.g., glasses, earrings, or necklaces), or hairstyle.

- 3.3 LumosX

As shown in Fig. 3, our framework builds on the T2V model Wan2.1 [40]. To enable personalized multi-subject video generation, all condition images are encoded into image tokens via a VAE encoder, concatenated with denoising video tokens, and fed into DiT [31] blocks. Within each block, we introduce Relational SelfAttention with Relational Rotary Position Embedding (R2PE) and a Causal Self-Attention Mask to support spatio-temporal and causal conditional modeling. Additionally, Relational Cross-Attention with a Multilevel Cross-Attention Mask (MCAM) incorporates textual conditions, strengthens visual token representations, and aligns face–attribute relationships.

- 3.3.1 Relational Self-Attention

Relational Rotary Position Embedding (R2PE). In T2V models like Wan2.1 [40], utilizing 3D Rotary Position Embedding (3D-RoPE) to assign position indices (i,j,k) to video tokens is necessary, which can affect the interaction among these tokens. In T2V tasks, the original 3D-RoPE assigns position indices (i,j,k) sequentially to the video tokens z ∈ RT×HW×C, where i ∈ [0,T), j ∈ [0,W), and k ∈ [0,H). In personalized multi-subject video generation tasks, it is essential to not only extend 3D-RoPE to the reference condition images but also to preserve the face-attribute dependency throughout this process.

Textual Tokens

largebluebowl

measuringcup

cozykitchen

youngwoman

Visual Tokens

yellowscarf

smallmetal

blackhoddie

setting

Key

- Attr A: Attribute A (yellow scarf)

- Face A: Human Face A

- Subject A (woman)

- Subject B

: True : False

Key

Query

(a) Causal Self-Attention Mask (CSAM)

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

- Face B: Human Face B (young woman)

- Attr B: Atrribute B (black hoodie)

AttrB FaceB

FaceA

AttrA

ObjB

Query

ObjA

woman

Other Text Tokens

Video Noise

Bg

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

VideoNoise

VideoNoise

VisualTokens

VisualTokens

- Face A

- Attr A

Face B

- Attr B Obj A

- Face A

- Attr A

Face B

- Attr B

- Obj A: Object A

(small metal measuring cup)

[Figure 66]

[Figure 67]

- Obj B: Object B (large blue bowl)

Strong Correlation

Objects

Correlation

- Obj A
- Obj B

[Figure 68]

Bg: Background Obj B (cozy kitchen setting)

Weak Correlation

Bg

Bg

(b) Multilevel Cross-Attention Mask (MCAM)

- Figure 4 Illustration of Attention Mask Design: Causal Self-Attention Mask (CSAM) and Multilevel Cross-Attention Mask (MCAM). We present a specific customized task case as an example.

c×HW×C represents the condition tokens, we introduce the Relational Rotary Position Embedding (R2PE), as illustrated in Fig. 3. The condition tokens zc are composed of subject tokens zsub, object tokens zobj, and background tokens zbg, i.e., zc = [zsub;zobj;zbg]. In R2PE, for the video tokens z, we adopt the standard 3D-RoPE (i,j,k) position assignment method, while for the background zbg and object zobj tokens, we sequentially extend each entity along the i-index. For the subject tokens zsub, which are composed of human face tokens zface and human attribute tokens zattr, we strictly adhere to the face-attribute dependency when assigning position indices to the subject tokens. Therefore, for the human face tokens and their corresponding attribute tokens within the same group, they share the same i-index and are extended along the j-index and k-index. Specifically, the position index for the condition tokens zc is defined as:

Given the concatenated VAE tokens z′ = [z;zc] ∈ R(T+N

c)×HW×C, where zc ∈ RN

ibg/obj + T,j,k , when zbg and zobj isub + T + Nbg/obj,j + W ∗ Nig

(i′,j′,k′) =

(1)

,k + H ∗ Nig

, when zsub

sub

sub

where ibg/obj ∈ [0,Nbg/obj), with Nbg/obj denoting the total number of background and object entity. isub ∈ [0,Nsub) where Nsub represents the total number of face-attribute subject groups. And, Nig

), where Ni

∈ [0,Ni

sub

sub

denotes the total number of face and attribute entity within the isub subject group. The proposed R2PE effectively inherits and extends the implicit positional correspondence of the original Wan2.1 model, while preserving the face-attribute dependency within each group of the subject condition.

sub

Causal Self-Attention Mask (CSAM). The Causal Self-Attention Mask is a boolean matrix, as illustrated in Fig. 4 (a), with the following two rules governing its mechanism: (I) Calculations are performed within each conditional branch, where the human face and its corresponding attributes are treated as a unified subject condition branch; (II) Video denoising tokens apply unidirectional attention to the condition tokens only. Given the concatenated tokens z′ ∈ R(T+N

c)×HW×C, the mask can be formulated as:

True, if q ∈ z or q == k or q,k ∈ zgsub False, otherwise

MSAq,k =

(2)

where q and k denote the categories of the tokens corresponding to the query-key matrix in Self-Attention, both of which belong to the visual concatenated tokens z′. z and zgsub represent the denoising video tokens and the face/attribute tokens within the same subject group. This causal mask enforces constraints on the range of interactions during the Self-Attention process. This design efficiently prevents unidirectional attention from the conditional branch to the denoising branch, while enabling the denoising branch to independently aggregate conditional signals and efficiently bind the face-attribute dependencies within the conditional branch. To enable efficient computation, we employ the MagiAttention mechanism proposed in [35].

- 3.3.2 Relational Cross-Attention

Multilevel Cross-Attention Mask (MCAM). In the Cross-Attention process of the T2V task, all visual tokens interact with all textual tokens. However, the requirements may differ for customized video generation tasks. Intuitively, all textual tokens are of equal importance for video denoising tokens. However, for visual condition tokens in customized tasks, each has a corresponding textual token, such as: face image → “man". Therefore, we aim to enhance the interaction of visual condition tokens with the corresponding textual tokens in the cross-attention process to improve the semantic representation of visual tokens. Furthermore, for subject condition tokens, we seek to strengthen the face-attribute dependency within the same subject group in the Cross-Attention process, while reducing the mutual influence between different subject groups.

Based on the aforementioned motivation, we propose the Multilevel Cross-Attention Mask (MCAM), as shown in Fig. 4(b). MCAM is a numerical mask in which we have defined three levels of correlation: Strong Correlation (1), Correlation (0), and Weak Correlation (-1). Specifically, Strong Correlation applies to the interaction between the visual condition tokens and their corresponding textual tokens, as well as between visual subject (face & attribute) condition tokens and all textual tokens within the same subject group. Weak Correlation applies to the interaction between visual subject tokens and the textual tokens from different subject groups. And all other cases remain as Correlation. Therefore, this mask MCAq,k can be formulated as:

 

1 (Strong Correlation), if q,k belong to the same semantic entity or subject group −1 (Weak Correlation), if q,k belongto the different subject group 0(Correlation), otherwise

MCAq,k =

(3)



where q and k denote the categories of the tokens corresponding to the query-key matrix in Cross-Attention, with the query and key representing the visual and textual tokens, respectively, in this context. Subsequently, we inject this constraint mask MCAq,k into the Cross-Attention as follows:

Cross-Attention(Q,K,V) = Softmax

QK⊤ + MCAq,k · s · r √dK

V (4)

where Q denotes concatenated visual features, and K and V are textual features. The hyperparameter r controls the strength of the MCAq,k constraint. Because similarity scores between query and key tokens vary across positions, a uniform mask template cannot be applied directly. To address this, we introduce a dynamic scaling factor s to adjust MCAq,k at each position. The most straightforward strategy is to use the absolute value of the similarity matrix itself as s. However, existing accelerated Attention computation modules based on Pytorch do not support customized numerical masks like this, and recomputing the similarity scores between Q and K outside the Attention module would incur significant computational overhead. To balance accuracy and efficiency, we propose an approximate method to compute the similarity matrix and derive s outside the Attention module as:

s = Repeat QdsK⊤ ,shape QK⊤ (5)

where Qds denotes Q downsampled by a factor of d × d via local average pooling on its spatial dimensions. The Repeat(·,shape(·)) operation restores the downsampled similarity matrix to its original size. Overall, the proposed MACM effectively strengthens relational dependency consistency and enhances the semantic-level representation of visual condition tokens.

##### 4 Experiments

- 4.1 Experimental settings

Datasets. Our personalized multi-subject video generation training dataset is built on Panda70M [4]. After the cleaning and processing steps described in Sec. 3.2, we obtain 1.57M samples: 1.31M single-subject, 0.23M two-subject, and 0.03M three-subject videos.

Benchmark. For the testing benchmark, 500 videos crawled from YouTube are processed using the method described in Sec. 3.2, including 220 single-subject, 230 two-subject, and 50 three-subject videos. To rigorously

assess personalized multi-subject video generation, we establish two tasks in this benchmark: identity-consistent and subject-consistent video generation. For identity-consistent video generation, 1 to 3 facial reference images are provided, with the facial similarity between the reference images and generated videos assessed using FaceSim-Arc (ArcSim), based on ArcFace [6], and FaceSim-Cur (CurSim), based on CurricularFace [14], following [52] and [48]. We also employ VideoCLIPXL (ViCLIP-T) [42] to measure semantic similarity between generated videos and text prompts. For subject-consistent video generation, the model takes multiple reference images, including faces, attributes, objects, and the background, as input. To comprehensively evaluate this task, we consider two aspects: 1) Evaluation of the entire video, where we use ViCLIP-T [42] and ViCLIP-V [42] to assess semantic similarity between the generated videos and text prompts, as well as between the generated videos and the ground-truth videos, respectively. Additionally, to prevent the generated videos from exhibiting copy-paste artifacts, we assess their dynamic degree, following [16]. 2) Evaluation on subjects, where we first use Florence-2 [47] to detect the person subjects based on the text prompts, and then apply OWLv2 [29] to detect the bounding boxes of the person’s attributes and other objects. Once all subjects are located, we compute CLIP-T [32] between the cropped image regions and the corresponding text prompts, as well as DINO-I [30] and CLIP-I [32] between these regions and the corresponding reference images. ArcSim [6] is also used here to assess the identity similarities. If a subject is not detected in the generated video, the score is set to zero.

Implementation Details. LumosX is fine-tuned from Wan2.1’s T2V(1.3B) [40] model, built on the DiT [31] architecture. Video generation is performed at 480p resolution, with each training clip containing 81 frames (5 seconds at 16 FPS). In MACM, the downsampling factor d and hyperparameter r are set to 8 and 0.5, respectively. During training, each subject’s face is associated with up to three attributes. Therefore, we recommend providing no more than three attributes per subject group during inference to maintain consistency with the training setup. LumosX is trained in two phases: 15k iterations on single-subject data, followed by 16k iterations on mixed multi-subject data. Training uses the Adam [18] optimizer with a learning rate of 1e-5, EMA decay of 0.99, weight decay of 1e-4, gradient clipping at 1.0, a batch size of 64, and random text-conditioning dropout at 10%. Our full training process required approximately 883 GPU-days on H20 GPUs. During inference, we use 50 steps and set the CFG scale to 6.

- 4.2 Main Results

In the following experiments, all baseline configurations are fixed when compared against our LumosX (Wan2.1-1.3B), including ConsisID (CogVideoX-5B) [48], Concat-ID (Wan2.1-1.3B) [52], SkyReels-A2 (Wan2.1-14B) [8], and Phantom (Wan2.1-1.3B) [24]. For the Identity-Consistent Video Generation setting, all methods use the same inputs: each subject’s face image and a shared global text prompt. For the Subject-Consistent Video Generation setting, we also enforce strict input parity. All methods receive exactly the same inputs as LumosX, including each subject’s face image, all associated attribute images, object reference images, the background image, and the shared global prompt.

Identity-consistent video generation. In this experiment, we use only face reference images as input to evaluate identity preservation with ArcSim and CurSim. We first compare our LumosX with face-specific customization methods, ConsisID [48] and Concat-ID [52], on a single-face test set of 220 videos, as ConsisID supports only single-face customization and Concat-ID has only released weights for the single-face setting, with results presented in Tab. 1. Additionally, we compare LumosX with SkyReels-A2 [8] and Phantom [24], two general multi-subject video customization methods, on the full test set of 500 videos, as shown in Tab. 2. Across both settings, LumosX achieves SOTA identity similarity scores, demonstrating its strong ability to preserve identity consistency in video generation. The qualitative comparison shown in Fig. 5 (a) further highlights the advanced performance of our LumosX.

Subject-consistent video generation. This experiment focuses on comprehensive multi-subject video customization, with reference inputs including faces, attributes, objects, and background images. For the evaluation of the entire video, we adopt ViCLIP-T and ViCLIP-V to assess the semantic similarity, while Dynamics detects potential copy-paste artifacts in the motion consistency. For evaluation on subjects, CLIP-T, CLIP-I, and DINO-I measure the semantic similarity, while ArcSim and CurSim evaluate facial similarity within the extracted subject regions, collectively capturing the accuracy of face-attribute associations in multi-subject

Table 1 Comparison of different methods for single-face identity-consistent video generation.

###### Table 2 Comparison of different methods for identityconsistent video generation.

Identity Consistency Prompt Following ArcSim ↑ CurSim ↑ ViCLIP-T ↑

Identity Consistency Prompt Following ArcSim ↑ CurSim ↑ ViCLIP-T ↑

Methods

Methods

ConsisID [48] 0.458 0.474 0.263 Concat-ID [52] 0.467 0.485 0.261 LumosX 0.542 0.575 0.262

SkyReels-A2 [8] 0.382 0.401 0.261 Phantom [24] 0.508 0.536 0.264 LumosX 0.510 0.540 0.262

- Table 3 Comparison of different methods for subject-consistent video generation, including evaluation of the entire video and evaluation on the extracted subjects.

Methods

Entire Video Extracted Subjects Dynamic ↑ ViCLIP-T ↑ ViCLIP-V↑ CLIP-T ↑ CLIP-I ↑ DINO-I ↑ ArcSim ↑ CurSim ↑

SkyReels-A2 [8] 0.671 0.251 0.839 0.178 0.606 0.192 0.271 0.290 Phantom [24] 0.661 0.254 0.865 0.185 0.647 0.216 0.444 0.477 LumosX 0.723 0.260 0.932 0.201 0.692 0.261 0.454 0.483

video generation. As shown in Tab. 3, our method achieves SOTA performance in both the entire video generation quality and the accuracy of face-attribute associations within the subject regions, outperforming the advanced models SkyReels-A2 [8] and Phantom [24], thus demonstrating the robustness of our LumosX. For qualitative comparison, Fig. 5(b) shows that our method supports flexible multi-subject foreground and background customization while maintaining accurate face–attribute matching across multiple subjects. In contrast, competing methods exhibit incorrect face–attribute pairings at a relatively high frequency, such as the mismatches visible in the lower-left SkyReels-A2 and Phantom results and the lower-right SkyReels-A2 result in Fig. 5(b), which further underscores the robustness of our approach.

4.3 Ablation Study

In this paper, we focus on enabling flexible foreground-background customization while addressing the dependency of face-attribute across multiple subjects. Therefore, we conduct the ablation study of our LumosX on subject-consistent video generation. We conduct experiments under a relatively lightweight setting, where the training dataset consists of 300K video samples while maintaining the same subject distribution as the full dataset, and the video generation resolution is set to 240p.

- Table 4 Ablation study for the subject-consistent video generation. The value in the MACM parentheses is r.

We ablate each component in LumosX, with the results presented in Tab. 4. The results show that R2PE (Row 2) significantly improves ArcSim, thanks to the binding of faces and attributes during the positional encoding, which helps the model avoid face confusion in generation. But CLIP-T shows a slight decrease, which we believe may be due to the shared T-idx within the subject group, affecting the semantic representation of individual entities. Furthermore, our CSAM (Row 3)

Methods CLIP-T ↑ ArcSim ↑ None 0.184 0.316

+R2PE 0.178 0.363 +R2PE + CSAM 0.182 0.363 +R2PE + CSAM + MCAM (0.1) 0.182 0.364

- +R2PE + CSAM + MCAM (0.5) 0.186 0.429
- +R2PE + CSAM + MCAM (1.0) 0.187 0.384

shows an improvement over CLIP-T, as it effectively blocks the interaction between different conditional signals and allows the denoising branch to independently aggregate these signals. For MCAM (Rows 4–6), we evaluated performance under different values of the hyperparameter r. The results indicate that MCAM yields substantial improvements in both CLIP-T and ArcSim, as it not only enhances the semantic representation of each visual condition but also effectively optimizes intra-group and inter-group correlations within subject groups. ArcSim achieves its best performance at r = 0.5 (Row 5), while CLIP-T peaks at r = 1.0 (Row 6). Since the improvement in ArcSim is more significant at r = 0.5, and considering that ArcSim better reflects the accuracy of face-attribute affiliation matching, we ultimately choose r = 0.5 in LumosX.

Caption

###### Reference Images

Caption

Reference Image

A man with dark hair and a beard, wearing a light brown zip-up sweater over a black shirt, stands in front of a dark, solid-colored wall. To his left, there is a large, industrial-looking spotlight mounted on a black tripod stand. To his right, there is a wooden ukulele with a glossy finish, featuring a sound hole, a bridge, and four tuning pegs.

A man with a beard, wearing a white t-shirt, and a woman with her hair tied back, wearing a black top, stand together in a modern, minimalistic interior. The man gestures with his hands while speaking, and the woman listens attentively. In the background, there is a white wall, a round mirror with a thin, dark frame, a kitchen counter with cabinets, and a potted plant with broad, green leaves.

[Figure 69]

[Figure 70]

[Figure 71]

man man woman

SkyReels-A2PhantomLumosX

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

ConsisIDConcat-IDLumosX

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

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

- (a) Identity-consistent video generation

- (b) Subject-consistent video generation

Reference Images Caption

###### Caption

Reference Images

man on the right

gray collared shirt

Two men are seated in a cozy room. The man on the right wears a gray collared shirt, and is speaking. The man on the left wears a blue polo shirt, listening attentively. In the background, there is a bookshelf with books and a small bottle.

small bottle

pink curtain with a decorative pattern

A man in a dark shirt stands in front of a brightly lit "XMAS" sign with speakers and a pink curtain with a decorative pattern in the background, pointing and speaking to the camera.

cozy room

man

[Figure 96]

dark shirt speakers

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

blue polo shirt

[Figure 104]

man on the left

[Figure 105]

PhantomSkyReels-A2LumosX

PhantomLumosXSkyReels-A2

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

Caption

Reference Images

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

###### Reference Images

Three individuals are in a modern kitchen, engaged in a cooking demonstration. The person on the left, wearing a headband, is smiling at the middle person in a black top holding a utensil. The person on the right in a striped shirt is holding food. In front of them are a wooden cutting board, a metal mixing bowl, and a white baking dish.

###### Caption

[Figure 134]

[Figure 135]

###### red long skirt

wooden cutting board

white baking dish

In a warmly lit room, a woman in a white shirt discusses a book with another woman wearing a striking red long skirt, both seated comfortably across from each other.

white shirt

person on the left

head -band

middle person

black top

[Figure 136]

woman

woman

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

metal mixing person on bowl

the right striped shirt

modern kitchen

PhantomLumosXSkyReels-A2

PhantomSkyReels-A2LumosX

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

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

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

- Figure 5 Qualitative comparison for identity-consistent and subject-consistent video generation.

- 5 Conclusion

This paper proposes LumosX, a novel framework designed for personalized multi-subject video generation that explicitly models face-attribute dependencies. To solve the lack of annotated data tailored for multisubject generation, we develop a data collection pipeline that supports open-set entities with subject-specific dependencies. Built upon Wan2.1’s T2V model, LumosX introduces Relational Self-Attention and Relational Cross-Attention, which incorporate position embedding and attention mechanisms to explicitly bind faceattribute pairs into coherent subject groups and optimize both intra-group and inter-group correlations. Extensive experiments validate the effectiveness of LumosX in generating fine-grained and personalized multi-subject videos, achieving state-of-the-art performance across diverse benchmarks.

Acknowledgments

This work was supported by the State Key Laboratory of Industrial Control Technology, China (Grant No. ICT2024A09).

References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Baidu. Paddleocr. https://github.com/PaddlePaddle/PaddleOCR, 2025.
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023.
- [4] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024.
- [5] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multi-subject open-set personalization in video generation. arXiv preprint arXiv:2501.06187, 2025.
- [6] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019.
- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [8] Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Chen, Yang Li, et al. Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436, 2025.
- [9] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [10] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024.
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [12] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [13] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.
- [14] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5901–5910, 2020.

- [15] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025.
- [16] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [17] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6689–6700, 2024.
- [18] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [19] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.
- [20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [21] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2023.
- [22] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26689–26699, 2024.
- [23] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [24] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025.
- [25] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024.
- [26] Xiaoran Liu, Hang Yan, Shuo Zhang, Chenxin An, Xipeng Qiu, and Dahua Lin. Scaling laws of rope-based extrapolation. arXiv preprint arXiv:2310.05209, 2023.
- [27] Ze Ma, Daquan Zhou, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong, Kurt Keutzer, and Jiashi Feng. Magic-me: Identity-specific video customized diffusion. arXiv preprint arXiv:2402.09368, 2024.
- [28] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7038–7048, 2024.
- [29] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 36:72983–73007, 2023.
- [30] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [31] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from

- natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [34] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.
- [35] Sand-AI. Magi-1: Autoregressive video generation at scale, 2025.
- [36] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.
- [37] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [38] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1526–1535, 2018.
- [39] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. Advances in neural information processing systems, 29, 2016.
- [40] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [41] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. Yolov9: Learning what you want to learn using programmable gradient information. In European conference on computer vision, pages 1–21. Springer, 2024.
- [42] Jiapeng Wang, Chengyu Wang, Kunzhe Huang, Jun Huang, and Lianwen Jin. Videoclip-xl: Advancing long description understanding for video clip models. arXiv preprint arXiv:2410.00741, 2024.
- [43] Zhao Wang, Aoxue Li, Lingting Zhu, Yong Guo, Qi Dou, and Zhenguo Li. Customvideo: Customizing text-to-video generation with multiple subjects. arXiv preprint arXiv:2401.09962, 2024.
- [44] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024.
- [45] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023.
- [46] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025.
- [47] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024.
- [48] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identity-preserving text-to-video generation by frequency decomposition. arXiv preprint arXiv:2411.17440, 2024.

- [49] Yuechen Zhang, Yaoyang Liu, Bin Xia, Bohao Peng, Zexin Yan, Eric Lo, and Jiaya Jia. Magic mirror: Id-preserved video generation in video diffusion transformers. arXiv preprint arXiv:2501.03931, 2025.
- [50] Yunpeng Zhang, Qiang Wang, Fan Jiang, Yaqi Fan, Mu Xu, and Yonggang Qi. Fantasyid: Face knowledge enhanced id-preserving video generation. arXiv preprint arXiv:2502.13995, 2025.
- [51] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [52] Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Chongxuan Li. Concat-id: Towards universal identity-preserving video synthesis. arXiv preprint arXiv:2503.14151, 2025.

### Appendix of LumosX

###### In this Appendix, we provide additional content organized as follows:

- • Sec. A details the data collection pipeline and training dataset, including:

- – Sec. A.1 Entity words retrieval and subject–attributes matching.
- – Sec. A.2 Obtaining condition images.
- – Sec. A.3 Data cleaning and filtering for training dataset.
- – Sec. A.4 Robustness analysis of external modules in the data collection pipeline.

- • Sec. B provides additional details on LumosX, including:

- – Sec. B.1 Training objective.
- – Sec. B.2 Data augmentation details during training.
- – Sec. B.3 Generalization of LumosX to diverse T2V architectures.

- • Sec. C presents extended ablation studies and evaluations, including:

- – Sec. C.1 Fine-grained quantitative comparisons of identity-consistent video generation.
- – Sec. C.2 Fine-grained quantitative comparison of subject-consistent video generation.
- – Sec. C.3 Additional quantitative results from various online sources.
- – Sec. C.4 Discussion on LumosX’s capability for customized control for 4+ subjects.
- – Sec. C.5 Quantitative assessment of temporal coherence among different methods.
- – Sec. C.6 Evaluation on public personalization benchmark.
- – Sec. C.7 Visual effectiveness of individual components of LumosX.
- – Sec. C.8 Qualitative comparison under varying hyperparameters r in MCAM.
- – Sec. C.9 Quantitative analysis of text-based attribute control.
- – Sec. C.10 Quantitative comparison with image-personalization–based initialization.
- – Sec. C.11 Discussion of the importance of the inpainting model in data collection pipeline.
- – Sec. C.12 Human study for multi-subject video customization.
- – Sec. C.13 Analysis of computational overhead and latency in LumosX.

- • Sec. D provides more visualization results, including:

- – Sec. D.1 Additional results of identity-consistent video generation.
- – Sec. D.2 Additional results of subject-consistent video generation.

- • Sec. E discusses the limitations and future work.

##### A Details of Data Collection Pipeline and Training Dataset

- A.1 Entity Words Retrieval and Subject-Attributes Matching

In Sec. 3.2 of the manuscript, we propose utilizing Qwen2.5-VL-32B [1] to retrieve multiple entity words from the caption, while leveraging prior visual information from human detection results to achieve precise face-attribute matching. The process of processing a human-detecting frame with the corresponding caption can be divided into four steps, as shown in Fig. 6. The result of Data Organization (step 2) serves as the input to Qwen2.5-VL and includes: captions (text), bounding boxes (text), and frames with colored bounding boxes (visual). During the Data Processing (third step) phase, we design a tailored prompt for Qwen2.5-VL to

###### better address the task requirements, as shown below. The results of Data Processing, as shown in the fourth column (“Data Annotation”) of Fig. 6, involve extracting key entity words from the captions and categorizing them into Subjects, Objects, and Background. For Subjects, we obtain human-attribute dependency relations, which strictly follow the matching structure presented in the human-detecting frame.

###### Data Collection

###### Data Organization

###### Data Processing Data Annotation

Caption

A man and a woman are seated at a table in a lush garden, sharing a meal. The man, wearing a black shirt and a black watch on his left wrist, gestures with his hands while speaking, while the woman, in a white top, listens attentively and occasionally responds. Utensils are set neatly on the tables as they enjoy their meal together...

Caption: "A man and a woman are seated at a table in a lush garden, sharing a meal. The man, wearing a black shirt and a black watch on his left wrist..." Bboxes: {{"bbox1": {{"coordinate": [0, 0, 365, 480], "color": "periwinkle"}}, "bbox2": {{"coordinate": [375, 64, 694, 480], "color": "orange"}}}} Frame: <image with man in blue bbox1, woman in orange bbox2>

[Figure 168]

Subjects man: black shirt, black watch woman: white top

You are an information extraction assistant. Analyze the given video caption, frame (with colored bounding boxes), and bboxes data.

###### Objects

utensils

###### Background

[Figure 169]

lush garden

- Figure 6 Process of entity words retrieval and subject-attributes matching with four steps: Data Collection, Data Organization, Data Processing, and Data Annotation.

###### Prompt design for entity words retrieval and subject-attribute matching in Qwen2.5-VL

PROMPT = """ You are an information extraction assistant. Analyze the given video caption, frame (with colored bounding boxes), and

bounding box data. Rules:

- 1. For PERSONS:

- - Identify how many persons by the caption and bboxes data, list ALL human-related terms (human/person/man/woman/girl/ boy etc.). Merge the same person to avoid redundancy.

- - Extract ONLY concrete attributes for each person: [OK] Pick maximum THREE most frequent concrete attributes. [X] The attibutes can be clothing descriptions (e.g., "black suit", "red dress", "glasses", "hat") or physical

features (e.g., "blonde hair", "beard") [X] REJECT abstract attributes (e.g., "skin tone", emotions/actions like "smiling" or "running") [X] REJECT additional descriptions of the attributes (e.g., red hoodie with a black design on the front -> red

hoodie) [X] REJECT long phrases: keep under 5 words per attribute

- - Crucially, use the Frame and BBoxes input: Visually match each person described in the Caption to the person inside the corresponding colored bbox (red, green, blue) depicted in the Frame. The BBoxes variable maps color names (e. g., "red") to bbox identifiers (e.g., "bbox1"). Assign the correct bbox identifier to the matched person to prevent mismatches.

- 2. For OTHER SUBJECTS: [OK] Pick maximum THREE most frequent concrete objects [OK] Use exact object names (e.g., "dog", "oak tree", "wooden chair") [X] REJECT objects that cannot be directly described with specific nouns (e.g., small rectangular object) [X] REJECT words that may refer to multiple objects (e.g., "leaves") [X] REJECT long phrases: keep under 5 words per attribute

- 3. For BACKGROUND: Nouns indicating the overall environment or backdrop of the video [OK] Extract the exact environmental phrase (e.g., "park", "coffee shop") [X] REJECT inferred locations

- 4. NOTES:

- - The extracted phrases must strictly come from the caption, without adding, removing, or rewriting words and punctuation marks.

- - Ensure bbox references correspond to the correct colored bbox from the input.

- - Bbox coordinate format: [xmin, ymin, xmax, ymax]. """

###### Subjects

white top

woman

###### Background

Objects lush garden

[Figure 170]

[Figure 171]

[Figure 172]

utensils

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

man

black shirt

Random Sampling

Subject & Object Masking +Inpainting

GroundingDINO + SAM

Face Cropping

SAM

[Figure 177]

[Figure 178]

Subjects

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

###### Background

###### Objects

man: black shirt, black watch woman: white top

lush garden

utensils

- Figure 7 Process of acquiring condition images of subjects, objects, and the background.

- A.2 Obtaining condition images

The detailed process for obtaining condition images, described in Sec. 3.2, is illustrated in Fig. 7. For subjects, we first apply face detection [41] within the bounding boxes obtained from human detection to extract facial crops. Then, guided by the extracted word tags from Qwen2.5-VL [1], we use SAM [19] to segment the corresponding attribute regions within these human bounding boxes. For objects, we first employ GroundingDINO [25], guided by object tags, to detect bounding boxes from the global image. We then apply SAM within these bounding boxes to segment the mask regions corresponding to each object entity. As for the background, we first remove the subjects and objects based on the previously obtained crops and masks in face cropping and SAM. Since SAM occasionally produces imprecise boundaries, we dilate the foreground masks before inpainting. We then utilize the diffusion-based inpainting model FLUX [21], guided by the prompt: “{background}, empty, nothing, there is nothing,” where {background} corresponds to the extracted background tag. Finally, based on the valid outputs from the three key frames, we randomly select one result for each entity to serve as its corresponding condition image. This design aligns with the inference process, where each condition typically corresponds to a single reference image. Additionally, the random selection prevents all condition images from originating from the same frame, thereby enhancing data diversity.

- A.3 Data Cleaning and Filtering for Training Dataset

Our personalized multi-subject video generation training dataset is constructed based on Panda70M [4]. However, not all data in the Panda70M dataset meets the quality standards required for video generation training. To ensure the suitability of the dataset, we perform a filtering and cleaning process based on the following criteria and operations:

- • Removing subtitles using a text detection pipeline based on PaddleOCR [2].
- • Cropping black-and-white borders using a simple Hough transform-based approach.
- • Excluding grayscale or black-and-white videos to ensure sufficient visual quality and aesthetic appeal.
- • Retaining only samples with QAlign quality > 3.5 [45] and aesthetic score > 2.0 [45], enforcing constraints on semantic alignment and visual aesthetics.
- • Retaining only samples with flow strength within the range of 0.05 to 2.0, constraining motion intensity to an appropriate level for video generation.

- • Retaining only videos in which 1 to 3 people are detected by Yolov9 [41], as videos with more than 3 individuals typically exhibit lower visual and semantic quality.
- • Removing duplicate videos by clustering based on VideoCLIP [42] embeddings.

After data cleaning and filtering, we collect a total of 1.57M video samples, comprising 1.31M single-subject, 0.23M two-subject, and 0.03M three-subject videos.

- A.4 Robustness Analysis of External Modules in the Data Collection Pipeline

To ensure the quality and robustness of the data collection pipeline, we adopt deliberate component choices and strict filtering strategies to reduce upstream prediction errors:

- • Careful Selection of High-Reliability Components: 1) For face–attribute tag extraction, we used QwenVL-32B, which incorporates visual priors, instead of the pure language model Qwen-2.5-32B, improving accuracy (see Sec. A); 2) For background inpainting, we selected FLUX over Stable Diffusion 2.0 due to its superior realism.
- • Strict Data Filtering Policies: 1) Qwen-VL predictions were accepted only if the predicted entities appeared in the captions; 2) Conservative thresholds were used in detection modules (box_threshold=0.5, text_threshold=0.45 in Grounding DINO and SAM) to eliminate unreliable region-text matches; 3) For background inpainting, we retained only samples where the foreground occupied less than 50% of the image to maintain consistency and plausibility. Additional filters are detailed in Sec. A.3.

While isolating the robustness of each external module is computationally expensive, we conduct a simple test of Qwen’s robustness in word tag extraction. A case was marked as incorrect if either the face or attribute word was missing from the caption, or if the pairing was semantically invalid. With visual priors, Qwen-VL-32B achieved 95.2% accuracy, compared to 78.4% for Qwen-2.5-32B without visual input. Overall, our stringent quality-driven filtering is evident in the data selection ratio: from 70 million Panda70M samples, only 1.57 million videos were retained (2.2%). This highlights our strong emphasis on quality control and minimizing error propagation.

#### B Details and Discussions of LumosX

- B.1 Training Objective

Due to our LumosX built upon Wan2.1’s [40] T2V model, we adopt Flow Matching[23] to formulate the training objective. Flow Matching is a generative paradigm that learns continuous-time dynamics using ordinary differential equations (ODEs). It directly estimates optimal transport paths between distributions, enabling stable training without iterative denoising steps. Given a video latent representation z, a random noise z0, and a timestep t ∈ [0,1] drawn from a logit-normal distribution, an intermediate latent zt is obtained as the training input. Following Rectified Flows (RF) [7], the intermediate state zt is defined via linear interpolation between z and z0, formulated as:

zt = (1 − t) · z0 + t · z, (6) The corresponding ground-truth velocity is given by the time derivative of xt:

dzt dt

= z − z0. (7)

vt =

The model is trained to estimate this velocity field. The training objective is defined as the Mean Squared Error (MSE) between the predicted velocity and the ground-truth velocity:

0,z,ctext,zc,t ∥u(zt,ctext,zc,t;θ) − vt∥2 , (8)

L = Ez

where ctext and zc denote the umT5 text embedding and reference visual embedding, respectively. θ is the model weights, and u(zt,ctext,zc,t;θ) denotes the output velocity predicted by the generation model.

- B.2 Data Augmentation Details During Training

During training, all conditional reference images are resized to match the video aspect ratio. Subject and object reference images are augmented with both numerical transformations (e.g., brightness, blur) and geometric transformations (e.g., rotation, horizontal flip), while background images are augmented only with numerical augmentations.

- B.3 Generalization of LumosX to Diverse T2V Architectures

While our current implementation is based on Wan2.1-T2V with a single-tower DiT architecture, the proposed modules in LumosX are architecture-agnostic and compatible with other DiT-style backbones, such as the dual-tower MM-DiT in HunyuanVideo [20] and the Parallel Attention in MAGI-1 [35]. Specifically, R2PE simply reorders relative position indices and can be seamlessly integrated into any DiT-based model, as it is independent of tower design. CSAM and MCAM function as attention masks for Self-Attention (among visual tokens) and Cross-Attention (between visual and textual tokens), respectively. In MM-DiT, these operations occur after dual-stream fusion and can adopt the same masking logic. Similarly, Parallel Attention includes equivalent attention interfaces that support our mask-based modules directly. In essence, regardless of whether the base model is Wan2.1, HunyuanVideo, or MAGI-1, all DiT-based T2V architectures share two key components: intra-modal interactions among visual tokens and cross-modal attention between visual and textual tokens. Our modules are explicitly designed to operate at these points, enabling straightforward extension to diverse architectures.

##### C More Ablation Discussion

C.1 Fine-Grained Quantitative Comparisons of Identity-Consistent Video Generation

We have added comparisons with Phantom and SkyReels-A2 in the single-face and multi-faces identityconsistent video generation setting, as shown in Tab 5. In the single-face setting (rows 1-3), our method underperforms Phantom in terms of identity preservation metrics (ArcSim and CurSim). However, our LumosX demonstrates a clear performance advantage in multi-face scenarios for identity preservation (rows 4–6).

- Table 5 Fine-grained quantitative comparison of identity-consistent video generation across different numbers of faces.

C.2 Fine-Grained Quantitative Comparison of SubjectConsistent Video Generation

Identity Consistency Prompt Following ArcSim ↑ CurSim ↑ ViCLIP-T ↑

Methods Numbers

To further demonstrate the superiority of our model in handling multi-subject settings, we conduct a more fine-grained quantitative evaluation of subject-consistent video generation across varying numbers of subjects (1-3 subjects), as shown in Tab. 6. It can be observed that as the number of subjects increases, LumosX demon-

SkyReels-A2 [8] 1 face 0.509 0.540 0.258 Phantom [24] 1 face 0.602 0.637 0.261 LumosX 1 face 0.542 0.575 0.262

SkyReels-A2 [8] ≥ 2 faces 0.282 0.292 0.263 Phantom [24] ≥ 2 faces 0.434 0.457 0.266 LumosX ≥ 2 faces 0.485 0.513 0.263

strates increasingly superior performance across the majority of metrics, with the performance gap over other methods becoming more pronounced. These experimental results further validate the effectiveness of the module we designed in LumosX for modeling multi-subject face-attribute dependency relationships.

- C.3 Additional Quantitative Results from Various Online Sources

We also provide additional qualitative results using condition images collected from various online sources. The test includes 50 sample cases (20 single-subject, 20 two-subject, and 10 three-subject cases). Since ground truth videos are not available for these samples, ViCLIP-V cannot be measured. As shown in Tab. 7, our

- Table 6 Fine-grained quantitative comparison of subject-consistent video generation across different numbers of subjects. (·) denotes that the respective method yields a lower value than our LumosX on this metric, whereas (·) indicates better performance. Values in parentheses are recalibrated differences computed as (Method − LumosX) for the same setting and metric.

Methods Numbers

Entire Video Extracted Subjects Dynamic ↑ ViCLIP-T ↑ ViCLIP-V↑ CLIP-T ↑ CLIP-I ↑ DINO-I ↑ ArcSim ↑ CurSim ↑

- SkyReels-A2 1 subject 0.772 (0.080) 0.252 (0.007) 0.870 (0.064) 0.177 (0.022) 0.604 (0.082) 0.199 (0.069) 0.356 (0.133) 0.380 (0.136)

- Phantom 1 subject 0.848 (0.004) 0.254 (0.005) 0.880 (0.054) 0.186 (0.013) 0.646 (0.040) 0.215 (0.053) 0.539 (0.050) 0.581 (0.065)

- LumosX 1 subject 0.852 0.259 0.934 0.199 0.686 0.268 0.489 0.516

SkyReels-A2 2 subjects 0.655 (0.009) 0.251 (0.011) 0.820 (0.110) 0.182 (0.021) 0.620 (0.079) 0.185 (0.064) 0.216 (0.227) 0.233 (0.241) Phantom 2 subjects 0.526 (0.138) 0.255 (0.007) 0.856 (0.074) 0.188 (0.015) 0.658 (0.041) 0.215 (0.034) 0.396 (0.047) 0.424 (0.050)

- LumosX 2 subjects 0.664 0.262 0.930 0.203 0.699 0.249 0.443 0.474

SkyReels-A2 3 subjects 0.268 (0.156) 0.243 (0.017) 0.795 (0.141) 0.161 (0.039) 0.546 (0.144) 0.189 (0.097) 0.150 (0.202) 0.161 (0.219) Phantom 3 subjects 0.411 (0.013) 0.250 (0.010) 0.840 (0.096) 0.169 (0.031) 0.601 (0.089) 0.220 (0.066) 0.247 (0.105) 0.269 (0.111)

- LumosX 3 subjects 0.424 0.260 0.936 0.200 0.690 0.286 0.352 0.380

SkyReels-A2 4 subjects 0.257 (0.047) 0.241 (0.022) 0.761 (0.165) 0.111 (0.095) 0.375 (0.329) 0.116 (0.149) 0.122 (0.167) 0.129 (0.190) Phantom 4 subjects 0.286 (0.018) 0.240 (0.023) 0.790 (0.136) 0.165 (0.041) 0.615 (0.089) 0.200 (0.065) 0.191 (0.098) 0.200 (0.119)

- LumosX 4 subjects 0.304 0.263 0.926 0.206 0.704 0.265 0.289 0.319

- Table 7 Comparison of different methods for subject-consistent video generation, including evaluation of the entire video and evaluation on subjects.

Entire Video Extracted Subjects Dynamic ↑ ViCLIP-T ↑ CLIP-T ↑ CLIP-I ↑ DINO-I ↑ ArcSim ↑ CurSim ↑

Methods

SkyReels-A2 [8] 0.839 0.235 0.207 0.614 0.173 0.160 0.166 Phantom [24] 0.929 0.255 0.226 0.623 0.195 0.271 0.305 LumosX 0.804 0.247 0.232 0.668 0.336 0.317 0.346

Caption

"10 people looking directly at the camera with fully visible frontal faces. They toss and catch glowing objects between each other in a high-tech arena. Fast coordinated motion, but faces always unobstructed and well-lit."

Wan2.1-T2v-1.3B

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

- Figure 8 Visualization of 10-person T2V results of our baseline model Wan2.1-T2V-1.3B.

method continues to outperform other approaches on most metrics, particularly those related to subject-specific customization (CLIP-T, CLIP-I, DINO-I, ArcSim, CurSim).

- C.4 Discussion on LumosX’s Capability for Customized Control for 4+ Subjects

LumosX is designed with scalability in mind. Although the training dataset contains videos with up to three subjects, the model architecture, including R2PE, CSAM, and MCAM, is inherently scalable and capable of handling more subjects during inference without retraining. To validate this capability, we conduct quantitative comparisons under a four-subject setting using 50 videos, without any retraining. The results in Tab. 6 show that even in the four-subject setting, our method consistently outperforms the baselines. Moreover, performance remains stable compared to the three-subject setting, showing no significant drop. It is worth noting that although R2PE in LumosX inherits the extrapolation capability of RoPE, increasing the number of subjects slightly beyond what was seen during training (e.g., from 3 to 4) does not lead to a noticeable drop in performance during inference without re-training. However, when the subject count increases substantially (e.g. from 3 to 10+), the risk of extrapolation instability rises significantly: higher RoPE dimensions may not have seen a full rotation period during training [26], causing positional encodings to become Out-of-Distribution (O.O.D), which may degrade attention alignment. To support long-range inference without retraining, we plan to try to integrate NTK-Aware Scaled RoPE (NTK-RoPE) [26], a training-free length extrapolation method that extends RoPE’s context range by adjusting the base of its sinusoidal encoding rather than learning new positional parameters. This change enables the model to handle positions beyond the training window with

- Table 8 Quantitative comparison of temporal coherence metrics on subject-consistent video generation.

Methods Subject Consistency ↑ Background Consistency ↑ Motion Smoothness ↑ Face Consistency ↑ SkyReels-A2 [8] 0.654 0.798 0.979 0.739

Phantom [24] 0.768 0.853 0.986 0.883 LumosX 0.962 0.946 0.988 0.895

minimal perplexity degradation and no fine-tuning required. In addition to RoPE extrapolation, we also find that the generation capability of the underlying base model itself plays a crucial role in determining the upper limit of how many subjects the customization model can realistically handle. For example, Wan2.1-1.3B-T2V, which we rely on, also struggles significantly when generating videos containing more than ten distinct human subjects. Fig. 8 presents visualizations of Wan2.1-1.3B-T2V under a 10-subject setting, where the facial quality is severely degraded and the model fails to strictly follow the prompt, generating only nine people instead of ten. This reinforces our point that the base model’s capability is the primary bottleneck when extending LumosX to scenarios involving substantially more subjects. These results indicate that current video-generation models like Wan-2.1 lack the capacity to learn stable representations for such large numbers of subjects, largely because high-quality 10+ subject videos are extremely rare.

- C.5 Quantitative Assessment of Temporal Coherence Among Different Methods

To evaluate temporal coherence, we additionally report four metrics: Subject Consistency, Background Consistency, Motion Smoothness, and Face Consistency. The first three are adopted from VBench [16]. Subject Consistency measures how consistently the subject (e.g., person or object) appears across consecutive frames. Background Consistency evaluates the temporal stability of the background, while Motion Smoothness quantifies the fluidity and natural continuity of motion. In contrast, Face Consistency is a metric we define to capture frame-to-frame facial similarity. Specifically, we use ArcFace embeddings to compute the similarity of the same face across consecutive frames. The results are shown in Tab. 8, evaluated on the test set for subject-consistent video generation, which contains 500 videos. Our LumosX outperforms other methods across all metrics.

Table 9 Quantitative comparison on MSRVTT-personalization [5] benchmark. boldfacen and underline font indicate the highest and the second highest results.

Method #Base Model Text-S ↑ Vid-S↑ Subj-S↑ Dync-D↑

SkyReels-A2 [8] Wan2.1-14B-T2V 0.253 0.781 0.554 0.783 Phantom [24] Wan2.1-1.3B-T2V 0.270 0.696 0.534 0.461 LumosX Wan2.1-1.3B-T2V 0.258 0.707 0.549 0.786

- C.6 Evaluation on PublicPersonalizationBenchmark

To further evaluate our method, we conducted a quantitative comparison between our LumosX and other approaches, including SkyReels-A2 and Phantom, on the MSRVTT-personalization [5] benchmark. The MSRVTT-personalization benchmark provides two evaluation modes: subject-mode and face-mode. Face-mode focuses solely on facesimilarity metrics, whereas we are interested in assessing the overall subject performance (face + attributes). Therefore, we conduct our evaluation under the subject-mode setting. Our evaluation is conducted using one reference image for the subject and one for the background, and, the results are reported in Tab. 9. It is worth noting that MSRVTT-personalization is a single-subject benchmark, and its definition of ’subject’ differs from ours: a subject may refer to either a human category (e.g., man, woman) or an object category (e.g., car, horse, clothes, hat). Moreover, for human subjects under the benchmark’s subject-mode setting, the subject is provided as a holistic entity without decoupled face and attribute references, and therefore does not involve the face–attribute matching problem that our method is designed to address. As a result, our method does not benefit from its explicit face–attribute relational modeling in this setting. In addition, both LumosX and Phantom are built on the Wan2.1-1.3B-T2V base model, whereas SkyReels-A2 adopts the much stronger Wan2.1-14B-T2V base model and therefore further benefits from a significantly more capable backbone. Despite these disadvantages, our LumosX still achieves the second-best overall performance, demonstrating the robustness and generalization ability of our approach even outside its primary multi-subject setting.

Caption

In a sunny park, an old man in a white shirt enjoys a leisurely walk beside a woman dressed in a vibrant yellow long skirt, both sharing light-hearted conversations.

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Reference Images

###### NoneR2PER2PE+CSAM

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

old man

white shirt

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

woman yellow long skirt

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

R2PE+CSAM

+MCAM(0.5)

- Figure 9 Visual effectiveness of individual components proposed in our LumosX.

- C.7 Visual Effectiveness of Individual Components of LumosX

To assess the impact of each proposed component in LumosX, we conduct a qualitative evaluation, as illustrated in Fig. 9. Without incorporating our proposed modules (first row), the generated video exhibits noticeable identity confusion. Specifically, an old white man is misrepresented as black, and a young black girl is erroneously depicted as old. With the introduction of our designed R2PE (second row), coarse-grained identity attributes, including age and skin tone, are effectively rectified. While the introduction of the CSAM (third row) further improves fine-grained facial identity preservation, certain artifacts still persist in the generated videos. Ultimately, the incorporation of the MCAM (fourth row) leads to improved identity consistency in the generated videos, while also enhancing video quality to a certain degree. To further validate the effectiveness of our CSAM and MCAM, we also visualize the Attention Similarity Scores averaged over all 12 heads in the last DiT layer for the case shown in Fig. 9. The results are presented in Fig. 10. It is worth noting that our generated videos contain 81 frames, and after VAE encoding, the temporal dimension is downsampled by a factor of four, resulting in a video noise token length of 21. The results in this figure can be compared with those in Fig. 4 in the main paper. From this comparison, we observe that CSAM (Fig. 10(a) vs. Fig. 10(b)) enables the denoising branch to independently aggregate conditional signals and to effectively bind face–attribute dependencies within the conditional branch. Moreover, MCAM (Fig. 10(c) vs. Fig. 10(d)) explicitly enhances intra-group and inter-group correlations among subject groups and strengthens the semantic-level representation of the visual condition tokens.

- C.8 Qualitative Comparison under Varying Hyperparameters r in MCAM

When MCAM with r = 0.1 is introduced (third row), the artifacts in the generated videos are significantly alleviated. Nevertheless, identity consistency is still unsatisfactory; for instance, a short-sleeved T-shirt may appear as long-sleeved in the video, and facial similarity is still insufficient. At r = 0.5 (fourth row), we observe a notable improvement in identity consistency, especially in the resemblance between facial appearances in the generated videos and the reference image, along with a further enhancement in video quality. Increasing r to 1.0 (fifth row) results in continued improvements in video quality, though it introduces a slight decline in identity similarity. The above findings align with the quantitative results reported in Sec. 4.3 of the manuscript. Accordingly, we choose r = 0.5 as the final setting based on a balanced trade-off between identity consistency and video quality.

(i) Self-Attention Similarity Score (ii) Cross-Attention Similarity Score

[Figure 210]

[Figure 211]

###### Visual

0-20: Video Tokens

[Figure 212]

[Figure 213]

21 22

[Figure 214]

[Figure 215]

Visual

Visual

23

24

Texual

Visual

- (a) w/o CSAM
- (b) w/ CSAM

(c) w/o MCAM

[Figure 216]

[Figure 217]

###### Textual

7-8: old man

11-12: white shirt

22: woman

28-30: yellow long skirt

Visual

Visual

Visual (d) w/ MCAM

Texual

- Figure 10 Visualization of attention maps. (i): Self-Attention similarity score. (ii): Cross-Attention similarity score.

man

Reference Images

Caption

A man in a yellow shirt and a woman in a black shirt sit across from each other at a coffee shop, engaged in a lively conversation.

yellow shirt

woman black shirt

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

None R2PE+CSAM

- +MCAM(0.1)

R2PE+CSAM

- +MCAM(1.0)

R2PE+CSAM

R2PE+CSAM+MCAM(0.5)

[Figure 222]

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

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

- Figure 11 Qualitative comparison under varying control hyperparameters r in MCAM.

- C.9 Quantitative Analysis of Text-Based Attribute Control Without Images in Multi-Subject Video Customization

Our method supports attribute control using text alone, and this setting is already included in both the main paper and the appendix under Identity-Consistent Video Generation (Sec. 4.1, main paper). To more comprehensively evaluate this setting in multi-subject video customization scenarios, we perform subject-region

quantitative evaluations under this setting, using the same metrics as in Subject-Consistent Video Generation, to compare text-based attribute control across different methods. The results are presented in the Tab. 11. Overall, the CLIP-T scores are higher under text-only control, which is expected since the attributes are directly specified via text prompts, naturally yielding higher text–video similarity. CLIP-I trends largely mirror those of CLIP-T because both measure semantic similarity. Notably, for LumosX (rows 6-7, col. 4), the performance gap between text-only control and text&visual control is very small, whereas for the other methods (rows 2-5, col. 4), performance under text&visual control drops noticeably compared to text-only control. This suggests that those methods become more prone to attribute confusion once visual conditions are introduced, likely due to the absence of explicit face–attribute relational modeling, whereas LumosX avoids this issue. For DINO-I, text-only control is clearly weaker than text &visual control in LumosX. At the same time, the substantial improvement under text&visual control indicates that our face–attribute relational constraints are accurately matched and effectively utilized. In comparison, Phantom and SkyReels-A2 do not show notable DINO-I improvements when visual attribute conditions are added, suggesting that their face–attribute alignment is less reliable. Meanwhile, under the text-based attribute control setting, LumosX performs on par with Phantom and clearly outperforms SkyReels-A2, demonstrating its strong generalization ability even without attribute images.

- C.10 Quantitative Comparison with Image-Personalization–Based Initialization

To evaluate whether multi-subject image personalization models can serve as an initial image generator prior to video synthesis, we conducted an additional experiment using UNO [46] to produce a multi-subject customized image, which was then fed into Wan2.1-14B-I2V [40] for video generation. We quantitatively compared this pipeline (UNO + Wan-I2V) against our method on our benchmark. The results are shown in Tab. 12. Overall, our method outperforms UNO + Wan2.1-I2V-14B, particularly on ViCLIP-V, DINO-I, FaceSim and CurSim, which reflects facial identity consistency. This improvement is expected, as UNO is not specifically designed for face-aware attribute binding, whereas our approach explicitly models face–attribute relational constraints, leading to significantly better identity preservation in video personalization. Our ViCLIP-T and CLIP-T scores are slightly weaker than those of UNO + Wan2.1-I2V-14B, which may be attributed to the latter’s use of a substantially larger base model (Wan-14B).

- C.11 Discussion of the Importance of the Inpainting Model in Data Collection Pipeline

To quantitatively assess the realism of FLUX compared with other inpainting models, we conducted a large-scale evaluation on 2,130 test cases, applying both FLUX [21] and Stable-Diffusion-2 [33] for inpainting. We then computed FID scores against the COCO 2017 Val set to measure distributional realism. In addition, we randomly sampled 100 cases and asked GPT-4o to independently judge which inpainting result appeared more realistic. The instruction provided to GPT-4o is as follows:

Instruction design for inpainting model assessment in GPT-4o

PROMPT = """

- - I will give you three images. The first two images are background inpainting results obtained by masking out the foreground in the original image and then applying two different inpainting models. The third image is the corresponding foreground mask. Please evaluate which of the first two images provides a better inpainting result based on pixel-level consistency, scene continuity, and the overall plausibility of the inpainted background. You

should output only "1" or "2", indicating whether the first or the second image is better. Do not output anything else.

- - Note that this is background completion, so the masked region should be inpainted with appropriate background content

. Pay particular attention to the continuity and coherence of the background in the inpainted region. """

The results of both evaluations are summarized in the Tab. 10. The results show that FLUX achieves better performance under both evaluation metrics. To further understand how background quality affects downstream video generation, we performed qualitative comparison using the two inpainting outputs as inputs to video generation. As shown in Fig. 12, artifacts introduced during background inpainting are clearly propagated into the generated video, degrading overall video quality. This confirms that inpainting realism plays a

Reference Images

Caption

[Figure 242]

[Figure 243]

A woman with red lipstick and a leopard-print top, seated in a modern office setting, makes a peace sign, and holds up a white box.

leopard print top

woman

[Figure 244]

[Figure 245]

[Figure 246]

| |
|---|

| |
|---|

Background Image with Mask Stable-Diffusion-2-Inpainting FLUX-Inpainting

modern office setting

Stable-Diffusion-2

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

FLUX-Inpainting-Inpainting

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

- Figure 12 Qualitative comparison between Stable-Diffusion-2 and FLUX for background inpainting.

crucial role in high-quality video generation and motivates our choice of FLUX as the inpainting module.

###### Method FID ↓ AI Judgement ↑ Stable-Diffusion-2-Inpainting 96.32 36%

###### C.12 Human Study for Multi-subject Video Customization

FLUX-Inpainting 92.83 64%

To complement automated metrics, we conducted a user study on a randomly sampled set of 24 video cases, including 6 singlesubject, 12 two-subject, and 6 three-subject customization scenarios. The comparison includes LumosX, SkyReels-A2, and Phantom, with a total of 30 participants. Each participant evaluated videos along four dimensions:

Table 10 Comparison between Stable-Diffusion-2 [33] and FLUX [21] for background inpainting.

- • Face–Attribute Alignment: whether each face is correctly matched with its corresponding attributes (e.g., clothing, accessories, or hairstyle).
- • Face Similarity: how closely each generated face resembles the provided visual reference.
- • Video Naturalness: the overall visual quality and coherence of the generated video.
- • Prompt Adherence: whether the generated video follows the instructions specified in the text prompt.

For each case, participants ranked the three methods, and scores were assigned as follows: 1 point for first place, 0.5 points for second place, and 0 points for third place. Final scores for each dimension were computed as the weighted average across all participants. The results are shown below Fig 13. Overall, our method achieves superior performance across all four evaluation dimensions.

###### C.13 Analysis of Computational Overhead and Latency in LumosX

To evaluate the computational impact of our relational attention mechanisms, we report detailed inferencestage statistics on computational overhead and latency. Specifically, we present the average latency for each Self-Attention and Cross-Attention operation, the average per-step latency, per-step FLOPs, and GPU

[Figure 255]

[Figure 256]

(a) (b)

- Figure 13 Human study results for multi-subject video customization. (a) Results across the four evaluation dimensions. (b) Average scores computed over all four dimensions.

- Table 11 Quantitative comparison of text-only vs. text&visual attribute control across methods. Methods Attribute Control CLIP-T CLIP-I DINO-I

SkyReels-A2 [8]

text-only 0.192 0.643 0.192

text&visual 0.178 0.606 0.192 Phantom [24]

text-only 0.207 0.687 0.209

text&visual 0.185 0.647 0.216 LumosX

text-only 0.205 0.684 0.210 text&visual 0.193 0.681 0.265

- Table 12 Comparison of image-personalization–based method and LumosX for subject-consistent video generation, including evaluation of the entire video and evaluation on subjects.

Entire Video Extracted Subjects Dynamic ↑ ViCLIP-T ↑ ViCLIP-V↑ CLIP-T ↑ CLIP-I ↑ DINO-I ↑ ArcSim ↑ CurSim ↑

Methods

UNO + Wan2.1-I2V-14B 0.547 0.261 0.880 0.201 0.650 0.197 0.237 0.244 LumosX 0.723 0.260 0.932 0.201 0.692 0.261 0.454 0.483

memory consumption. All measurements are obtained on an H20 GPU under the same video customization scenario. The results are shown in Tab. 13 and the key observations are summarized below:

- • R2PE incurs no extra compute or memory overhead (row 1 vs. row 2), since it reorders relative position indices rather than introducing new parameters or operations.
- • The original implementation of the Wan2.1-T2V model utilizes FlashAttention 2.0 for acceleration, which does not support custom masks. In our CSAM module, we replace it with MagiAttention [35] (see Sec. 3.3.1 in the main paper), which supports custom masking and achieves faster inference (row 1 vs. row 3). The integration of CSAM results in only a modest overhead (row 4 vs. row 5).
- • For MCAM (Cross-Attention), MagiAttention cannot handle numeric masks, so we use PyTorch’s native implementation. Since the key comes from relatively short T5 token sequences (512 tokens), the computation remains lightweight. As a result, MCAM does not noticeably affect Cross-Attention latency (row 5 vs. row 6, col 2), and the additional cost of computing the dynamic scaling matrix s is relatively small and well within acceptable bounds (row 5 vs. row 6, cols 3–5).

In summary, through the use of efficient modules (MagiAttention) and optimized strategies (e.g., lightweight scaling matrix design), LumosX achieves high computational efficiency with minimal additional inference overhead (row 6 vs. rows 1 and 4).

- Table 13 Inference-stage computational overhead and latency statistics for each module under the same video customization case on an H20 GPU.

Methods Self-Attn Latency Cross-Attn Latency Latency/step FLOPs/step GPU Memory Usage

None 0.1440s 0.0026s 8.66s 195.44 T 21.5G +R2PE 0.1441s 0.0025s 8.66s 195.44 T 21.5G None (MagiAttention in Self-Attn) 0.0935s 0.0025s 5.79s 195.44 T 21.5G +R2PE (MagiAttention in Self-Attn) 0.0936s 0.0026s 5.79s 195.44 T 21.5G +R2PE+CSAM 0.0966s 0.0026s 5.81s 195.44 T 21.5G +R2PE+CSAM+MCAM 0.0965s 0.0045s 6.11s 195.46 T 22.7G

- D More Visualization Results

- D.1 Additional Results of Identity-Consistent Video Generation

The qualitative identity-consistent video generation results are shown in Fig. 14. Under the single-subject setting (Case 1 and Case 2), our method demonstrates significantly superior identity preservation performance compared to ConsisID [48] and Concat-ID [52]. Under the multi-subject setting (Case 3 and Case 4), our approach demonstrates superior performance over SkyReels-A2 [8] and Phantom [24] in both identity preservation and the alignment between multiple subjects and captions, effectively preventing character confusion (Case 3 SkyReels-A2) and positional confusion (Case 3 Phantom).

- D.2 Additional Results of Subject-Consistent Video Generation

The qualitative comparison of subject-consistent video generation is shown in Fig. 15 and Fig. 16. The experimental results demonstrate that our model supports flexible multi-subject foreground-background video customization. Compared to SkyReels-A2 [8] and Phantom [24], our approach achieves superior subject consistency, accurately matching the human faces and their corresponding attributes, and maintaining the reference identity throughout the generation process. In contrast, SkyReels-A2 struggles when handling multiple customized subjects, often exhibiting character confusion (Cases 2, 4 in Fig. 15 and Case 1 in Fig. 16) and subject disappearance (Case 3 in Fig. 15). Similarly, Phantom also suffers from character confusion (Cases 1,2 in Fig. 16) in comparable scenarios. Furthermore, the videos generated by our LumosX are natural and realistic. Unlike Phantom, which suffers from noticeable quality degradation as the number of reference condition images increases, resulting in visual artifacts (Case 1 in Fig. 15 and Cases 2, 3, 4 in Fig. 16) or an unintended cartoon-like style (Cases 2, 3, 4 in Fig. 15).

- E Limitations and Future Work

Although our LumosX effectively addresses the key challenge of multi-subject video personalization by explicitly modeling the dependency of face–attribute within the subject group, it remains constrained by limitations in model size as well as the diversity and scale of training data. Consequently, the performance of LumosX has not yet reached its full potential. Looking ahead, we plan to deploy LumosX on Wan2.1-14B-T2V model [40] and train it on a larger-scale, higher-quality, and more diverse dataset to further enhance its performance and generalization capabilities. Moreover, to push the boundaries of dynamic behavior understanding and better capture complex motion patterns, we also recognize the value of incorporating motion-aware constraints—for instance, augmenting data collection with motion descriptions (e.g., walking, running) and integrating motion cues within the MCAM module to strengthen correlations between visual tokens and motion-aware textual tokens, which would improve alignment for dynamic behaviors and multi-subject interactions (e.g., hugging, handshaking, passing objects).

A woman with shoulder-length wavy hair, wearing a white off-the-shoulder top and a black choker necklace with small, round, brown beads, is seen speaking and gesturing with her hands. She has a neutral expression and is wearing makeup, including eyeshadow and lipstick. Behind her, there is a black metal shelf with three tiers. On the top shelf, there is a small wooden box with a heart-shaped cutout on the lid.

Caption

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

ConsisIDConcat-IDLumosXPhantomLumosXSkyReels-A2

Reference Image

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

woman

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

A man with a beard, wearing a brown cap and a light brown shirt, stands in front of a white shed with a red roof, speaking and occasionally pointing to his right. The shed is adorned with a yellow wheelbarrow, a red pot, and a green hose coiled on the ground. To the right of the shed, a wooden fence with a red ball attached to it is visible, along with a tree featuring a thick trunk and green leaves. The ground is covered with brown mulch, and the scene is set in a sunny outdoor environment.

Caption

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

ConsisIDConcat-IDLumosX

Reference Image

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

man

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

Two men sit side by side on a light gray sofa, engaged in a conversation. The man on the left, wearing a dark blue shirt, smiles and looks directly at the camera, while the man on the right, in a light blue shirt, occasionally looks at his companion with a subtle smile. Behind them, a brick wall serves as the backdrop, with a dark brown wooden shelf mounted on it. On the shelf, two white candles are placed side by side.

Caption

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

Reference Images

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

man on the left

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

man on the right

Three people are standing in a kitchen in front of a stainless steel refrigerator. On the left, a blonde woman in a light blue top looks slightly to the side. In the center, a young woman with dark hair tied in a top knot, wearing a light gray sweatshirt, holds a piece of bread. On the right, a man in a dark long-sleeve shirt smiles while looking at her. The refrigerator behind them has magnets and papers, with white cabinets above it.

Caption

Reference Images

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

LumosXSkyReels-A2Phantom

[Figure 301]

woman

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

young woman

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

man

- Figure 14 Qualitative comparison for identity-consistent video generation.

Caption

A young man with short brown hair, wearing a light gray hoodie and a black watch, drives a sleek black convertible through a scenic area with trees and greenery. The side mirror is rectangular with a curved edge.

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

###### SkyReels-A2PhantomLumosX

Reference Images

[Figure 316]

[Figure 317]

light gray hoodie

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

young man

black watch

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

scenic area with trees and greenery

Two men are seated at a round wooden table, engaged in a casual conversation. The man on the left, wearing a green plaid shirt, is laughing and clapping his hands, while the man on the right, dressed in a dark green shirt, holds a white mug. A black smartphone is near the man on the left. In the background, a film reel is placed on a shelf. The room is warmly lit, creating a relaxed and friendly atmosphere.

Caption

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

Reference Images

###### SkyReels-A2PhantomLumosX

[Figure 332]

[Figure 333]

[Figure 334]

film reel

man on the left

green plaid shirt

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

white mug

man on the right

dark green shirt

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

room

Caption

A man in a white shirt and denim jacket converses with a woman in a light brown sweater on a bed in a cozy, warmly lit room. A wooden bedside table with a white lamp is in the background.

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Reference Images

###### SkyReels-A2PhantomLumosX

[Figure 351]

[Figure 352]

man

denim jacket

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

light brown sweater

woman

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

cozy, warmly lit room

Caption

A bald man in a black jacket and gray shirt speaks to a woman with short blonde hair in a dark dress and a man in a dark suit in a television studio. The small gray table next to them holds a black cup and two red dice.

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

Reference Images

###### SkyReels-A2PhantomLumosX

[Figure 368]

[Figure 369]

[Figure 370]

black jacket

gray shirt

blad man

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

dark suit

dark dress

woman

man

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

black cup

[Figure 385]

red dice

television studio

- Figure 15 Qualitative comparison for subject-consistent video generation.

Caption

Two men stand at a bustling city crosswalk. The first man wears a crisp white jacket, while the second man contrasts sharply in a sleek black jacket, both deep in conversation.

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Reference Images

###### SkyReels-A2PhantomLumosX

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

man

white jacket

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

man black jacket

Caption

A man in a white shirt and a woman wearing a red long skirt walk side by side through a bustling outdoor market, discussing the items on sale.

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

Reference Images

###### SkyReels-A2PhantomLumosX

[Figure 406]

[Figure 407]

man

white shirt

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

woman

red long skirt

Caption

A man wearing a vibrant red hat and a woman adorned in a flowing yellow skirt are leisurely strolling under the trees by seaside. The gentle sea breeze tugs playfully at their attire, adding a dynamic flutter to the woman's skirt and causing the man's hat to tilt charmingly. Their serene silhouettes against the backdrop of the tranquil blue ocean paint a picturesque scene of peaceful leisure.

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

Reference Images

###### SkyReels-A2PhantomLumosX

[Figure 422]

[Figure 423]

3

woman

yellow skirt

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

man red hat

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

under the trees by seaside

Caption

At the entrance of the zoo, a young girl in a bright pink tulle dress stands beside an old man dressed in a denim shirt, khaki belt, and a worn cowboy hat. They both face a calm capybara nearby, creating a warm, tranquil moment. The lush greenery and rustic structures in the background suggest a peaceful, nature-filled setting, emphasizing the gentle connection between the three figures.

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

###### SkyReels-A2PhantomLumosX

Reference Images

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

cowboy hat

old man

[Figure 445]

[Figure 446]

girl

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

capbara

###### Figure 16 Qualitative comparison for subject-consistent video generation.

