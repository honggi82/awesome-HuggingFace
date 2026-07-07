## Identity-Preserving Text-to-Video Generation by Frequency Decomposition

# arXiv:2411.17440v3[cs.CV]25Mar2025

Shenghai Yuan1,4, Jinfa Huang3, Xianyi He1,4, Yunyang Ge1,4, Yujun Shi5, Liuhan Chen1,4, Jiebo Luo3, Li Yuan1,2,†

1 Peking University, 2 Peng Cheng Laboratory, 3 University of Rochester, 4 Rabbitpre Intelligence, 5 National University of Singapore

{yuanshenghai,HeXianyi,yunyang,chenliuhan}@stu.pku.edu.cn, shi.yujun@u.nus.edu, yuanli-ece@pku.edu.cn, {jhuang90@ur,jluo@cs}.rochester.edu

|https://github.com/PKU-YuanGroup/ConsisID<br><br>[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>a man standing next to an airplane, engaged in a conversation on his cell phone ... wearingsunglasses and a black top .. talking seriously ... airplane has a green<br><br>striperunning along its side, and there is a large engine visible behind his ...<br><br>[Figure 10]<br><br>... a man ... his gaze focused and intense as he holds the basketball ... dressed in athletic gear ... The basketball is positioned firmly in his hands as he gradually puts it down ... urban street court, with the fading light of dusk casting a soft glow over the scene. The man's serious expression and steady grip on the ball ...<br><br>a young boy sitting at a table, eating a piece of food. He appears to be enjoying his meal, as he takes a bite and chews it. The boy is wearing a blue shirtand has short hair ...background is dark, with some light coming from the left side of the frame. There is a straw visible on the right side of the frame ...<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>Text<br><br>Reference Image<br><br>Reference Image<br><br>Reference Image<br><br>Reference Image<br><br>a young woman with long blonde hair ... front of a lush, green bush...white flowers ... wearing a black top ... smiling and looking at the camera while gently touching the flowers on the bush ... then bends downslightly and smells one of theflower...<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>Figure 1. Examples of identity-preserving video generation (IPT2V) by our ConsisID. Given a reference image, our method can generate realistic and personalized human-centered videos while preserving identity. Red indicates that attributes in long instructions.<br><br>Abstract<br><br>Identity-preserving text-to-video (IPT2V) generation aims to create high-fidelity videos with consistent human<br><br>have not been resolved in the literature: (1) A tuning-free pipeline without tedious case-by-case finetuning, and (2) A frequency-aware heuristic identity-preserving Diffusion Transformer (DiT)-based control scheme. To achieve these|
|---|

goals, we propose ConsisID, a tuning-free DiT-based controllable IPT2V model to keep human-identity consistent in the generated video. Inspired by prior findings in fre-

identity. It is an important task in video generation but remains an open problem for generative models. This paper pushes the technical frontier of IPT2V in two directions that

quency analysis of vision/diffusion transformers, it employs identity-control signals base on frequency domain, since facial features can be decomposed into low-frequency global features (e.g., profile, proportions) and high-frequency intrinsic features (e.g., identity markers that remain unaffected by pose changes). Extensive experiments demonstrate that our frequency-aware heuristic scheme provides an optimal control solution for DiT-based models, making strides toward more effective IPT2V.

#### 1. Introduction

Large-scale pre-trained video diffusion models [37, 83, 96, 97] have facilitated a variety of downstream applications [60, 65, 86, 87, 89, 91], particularly in identity-preserving text-to-video (IPT2V) [10, 46, 70, 72, 73]. However, existing methods face significant challenges, particularly the high overhead associated with the need for case-by-case finetuning, which diminishes their applicability. Within the open-source community, only the ID-Animator [19] can implement tuning-free IPT2V, but it can only generate videos similar to talking head [69] and has poor id preservation.

Additionally, the above efforts are predominantly based on U-Net and cannot be adapted to the emerging DiT-based video model [37, 76, 83, 96, 97]. This challenge may stem from the inherent limitations of DiT compared to U-Net, including greater difficulty in training convergence and weakness in perceiving facial details. From some prior findings in frequency analysis of vision/diffusion transformers [3–

- 5, 53, 61, 66, 88], we can know that the reason is: Finding 1: Shallow (e.g., low-level, low-frequency) features are essential for pixel-level prediction tasks in diffusion models, as they ease model training. U-Net facilitates model convergence by aggregating shallow features to the decoder via long skip connections, a mechanism that DiT does not incorporate; Finding 2: Transformers have limited perception of high-frequency information, which is important for preserving facial features. The encoder-decoder architecture of U-Net naturally possesses multi-scale features (e.g., richness in high-frequency), while DiT lacks a comparable structure. To develop a DiT-based control model, these must be addressed first. Please see Appendix for more details.

For ID-preserving video generation, the challenges stem from the requirement for each frame to incorporate both high-frequency (e.g., age- and make-up-independent identity markers) and low-frequency information (e.g., facial shape) derived from the reference image, which can just be used to make up for the DiT defects mentioned above. Therefore, we propose ConsisID, to keep the identity consistency in video generation by frequency decomposition, based on the previously Findings of DiT in frequency analysis. Thanks to the large-scale pre-trained DiT, we can use its powerful capabilities to achieve tuning-free effects.

ConsisID decouples identity features into high- and lowfrequency signals, which are injected into specific locations within the DiT, facilitating efficient IPT2V generation. Specifically, in line with Finding 1, we first convert the reference image and the facial key points to the low-frequency signal, then concatenate them with input noise latent to ease the training. Following Finding 2, we utilize a dual-tower feature extractor to capture high-frequency facial information, which is integrated with vision tokens within the transformer block, thereby enhancing the DiT’s high-frequency perception capabilities. Finally, to transform the pre-trained model into an IPT2V model and improve its generalization, we further introduce a hierarchical training strategy.

Our contributions can be summarized as follows:

- • We introduce ConsisID, a tuning-free identity-preserving DiT-based IPT2V model, which preserves the identity of the main subject of the video using control signals from frequency decomposition.
- • We propose a hierarchical training strategy, including coarse-to-fine training, dynamic mask loss, and dynamic cross-face loss, which work together to facilitate training and enhance generalization effectively.
- • Extensive experiments demonstrate that our ConsisID can generate high-quality, editable, consistent identitypreserving videos, benefiting from our frequency-aware identity-preserving T2V DiT-based control scheme.

#### 2. Related Work

Tuning-based Identity-preserving T2V Models. Diffusion models are widely recognized for their strong generative capabilities [23, 42–45, 52, 89, 90], significantly advancing the development of identity-preserving generative models [11, 47, 71, 91]. Initially, the researchers used tuning-based methods to generate content that matched the input ID. This process requires finetuning pretrained model for each new person during inference. For example, DreamBooth [57] introduced a novel loss function to fine-tune the entire network, embedding identity information while preserving the original generative capabilities. LoRA [25], similar to DreamBooth [57], requires training only a small subset of network parameters. In contrast, Textual Inversion [15] freezes the pretrained network and embeds identity information into a trainable word embedding. Subsequent tuning-based methods, including both image and video models based on U-Net or DiT architectures [10, 33, 58, 70, 72, 73, 75, 82, 95], generally follow three main approaches. While these models demonstrate substantial effectiveness, the requirement to fine-tune for each new identity restricts their practical applicability.

Tuning-free Identity-preserving T2V Models. To address the issue of high resource consumption, several tuning-free diffusion models have recently emerged in the field of image generation [17, 18, 36, 68, 84]. These models

|“a woman in a dynamic action pose, dressed in a ... ”<br><br>|Sample|Face|
|---|---|
| | |
|Key Poi|nts (eye,ear, nose|
<br><br>Text Token<br><br>ψθ<br><br>Vision Token<br><br>|[Figure 25]|
|---|
<br><br>τθ<br><br>Attention<br><br>Cross<br><br>FFN<br><br>MM-DiT Block<br><br>×N<br><br>Global Facial Extractor<br><br>Generated Video<br><br>Text<br><br>Vision<br><br>Consistent Strategy<br><br>MLP<br><br>MLP Dropout<br><br>Local Facial Extractor<br><br>Trainable Frozen<br><br>Element-wise Addition C<br><br>C Element Concatenate<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>Cross Loss<br><br>Mask Loss<br><br>MSE Loss<br><br>)<br><br>[Figure 29]<br><br>[Figure 30]<br><br>CLIP Face<br><br>Extractor<br><br>Multi-scale features<br><br>Low-Freq Token<br><br>...<br><br>Q-Former<br><br>withDropout<br><br>[Figure 31]<br><br>...<br><br>High-Freq Token<br><br>ℒc<br><br>ℒe<br>ℒf<br><br><br>Learnable Queries<br><br>Figure 2. Overview of the proposed ConsisID. Based on Findings of DiT, low-frequency facial information is embedded into the shallow layers, while high-frequency information is incorporated into the vision tokens within the attention blocks. The ID-preserving Recipe is applied to ease training and improve generalization. The cross face, DropToken, and Dropout are executed based on probability.<br><br>not require finetuning parameters for newly introduced<br><br>during inference. For instance, IP-Adapter [84] utilizes the CLIP [51] features of the identity image through cross-attention to guide the pretrained model in generating identity-preserving images. InstantID [68] extends this approach by replacing CLIP [51] features with Arcface [13]<br><br>and integrating a pose network to adjust facial pro-<br><br>89]. The optimization process is defined:<br><br>La = Ex<br><br>0,t,y,ϵ ∥ϵ − ϵθ (x0,t,τθ(y))∥22 , (1)<br><br>where y is text condition, ϵ is sampled from a standard normal distribution (e.g., ϵ ∼ N(0,1)), and τθ(·) is the text|
|---|

Attention

FFN

Cross

ormer

###### ...

Dropout

Fi lay ap

do IDs liz cr id pr features portions. Unlike these initial methods, which introduce control signals via visual tokens, PhotoMaker [36] and Imagine Yourself [20] leverage text tokens. Specifically, PhotoMaker [36] concatenates identity features obtained from the CLIP encoder [51] to the text embedding, while Imagine Yourself [20] uses element-wise addition for feature fusion. In the domain of video generation, only MovieGen [50] and ID-Animator [19] currently support ID-preserving textto-video (IPT2V) generation. MovieGen is closed-source, whereas ID-Animator is open-source but uses a methodology similar to image models, leading to lower-quality identity preservation in the generated videos. We select the emerging DiT architecture [37, 41, 83, 96] and optimize it for IPT2V, drawing on conclusions from prior frequency analyses [3–5, 53, 61, 66]. This enables high-quality, editable, and consistent ID-preserving video generation.

encoder. By replacing x0 with E (x0), the latent diffusion is derived, which is used by ConsisID.

Diffusion Transformer. The DiT-based video generation model shows significant potential in simulating the physical world [8, 83, 97]. Despite being a novel architecture, research on controllable generation has been limited, and current methods [12, 17, 50, 94] largely resemble U-Net based approaches [15, 47, 91]. However, no study has yet examined why this approach works with DiT. Drawing from prior analyses of Diffusion and Transformer from a frequency domain perspective [3–5, 53, 61, 66], we conclude that: (1) Low-frequency (e.g., shallow-layer) features are essential for pixel-level prediction tasks in diffusion models, which helps facilitate model training; (2) Transformers have limited perception for high-frequency information, which is important for controllable generation. Based on these, we decouple ID features into high- and low-frequency parts and inject them into specific locations, achieving effective identity-preserving text-to-video generation.

#### 3. Methodology

##### 3.1. Preliminaries

##### 3.2. ConsisID: Keep Your Identity Consistent

Diffusion Model. Text-to-video generation models usually utilize the diffusion paradigm, which gradually transforms noise ϵ into a video x0. Originally, denoising was conducted directly within the pixel space [24, 62, 63]; however, due to significant computational overheads, recent methods predominantly employ latent space [16, 32, 56,

The overview is illustrated in Figure 2. Given a reference image, the global facial extractor and local facial extractor inject both high- and low-frequency facial information into model, which then generates identity-preserving videos with the assistance of the consistency training strategy.

###### 3.2.1. Low-frequency View: Global Facial Extractor

- In light of Finding 1, enhancing low-level (e.g., shallow, low-frequency) features accelerates model convergence. To easily adapt a pre-trained model for the IPT2V task, the most direct approach is concatenating the reference face with the noise input latent [7]. However, the reference face contains both high-frequency details (e.g., eye and lip textures) and low-frequency information (e.g., facial proportions and contours). From Finding 2, prematurely injecting high-frequency information into the Transformer is inefficient and may hinder the model’s processing of lowfrequency information, as the Transformer focuses primarily on low-frequency features. In addition, feeding the reference face directly into the model could introduce irrelevant noise such as lighting and shadows. To mitigate this, we extract facial key points, convert them to an RGB image, and then concatenate it with the reference image, as shown in

- Figure 2. This strategy focuses the model’s attention on the low-frequency signals in the face, while minimizing the impact of extraneous features. We found that when this component is discarded, the model is difficult to convergen. The objective function is changed to:

Lb = Ex0,t,y,f,ϵ ∥ϵ − ϵθ (x0, t, τθ(y), ψθ(f))∥22 , (2)

where ψθ(·) is a variational autoencoder, f represents the reference image, we ignore key points here for simplicity.

- 3.2.2. High-frequency View: Local Facial Extractor

- In light of Finding 2, we recognize that Transformers have limited sensitivity to high-frequency information. It can be concluded that relying solely on global facial extractor is insufficient for IPT2V generation, as global facial features lack of high-frequency information. So we use a face recognition backbone [13] to extract high-frequency features, as these are invariant to non-ID attributes (e.g., expression, posture, and shape). We refer to these features as intrinsic identity features (e.g., high-frequency), since age and makeup do not alter an individual’s core identity. Following [18], we utilize the penultimate layer of the backbone, rather than its output, as it retains more spatial information pertinent to identity. However, our experiments reveal that while the face recognition backbone improves identity consistency, it lacks the semantic features required for editing. This task demands not only maintaining identity consistency but also incorporating the ability to edit, such as generating videos of faces with the same identity but varying age and makeup. Previous research [19, 20] relies solely on the CLIP encoder [51] to enable editing capabilities. However, since CLIP is not specifically trained on face datasets, the features it extracts include irrelevant non-face information, which can compromise identity fidelity [36, 68, 84].

To address these challenges, we first use a facial recognition backbone to extract features that strongly represent intrinsic identity, and a CLIP image encoder to capture semantically rich features. We then employ the Q-Former

[34, 35, 77] to fuse these features, resulting in intrinsic identity representations enriched with high-frequency semantic information. To mitigate the impact of irrelevant features from CLIP, dropout [2, 26] is applied post-processing. Additionally, we follow [18] to concatenate the shallow, multiscale features from the facial recognition backbone (after interpolation) with the CLIP features. This approach ensures that the model captures essential intrinsic identity features while filtering out extraneous noise unrelated to identity. Finally, we apply cross-attention to facilitate interaction between this feature set and the visual tokens produced by each attention block of the pre-trained model, thereby enhancing the high-frequency information in the DiT:

###### Zi′ = Zi + Attention(Qvi ,Kif,Vif), (3)

where i represents the layer number of the attention block, Qv = ZiWiq, Kf = FWik, and V f = FWiv, where Zi is the visual token, F represents the intrinsic identity features, and Wq, Wk, and Wv are trainable parameters. The objective function is changed to:

Lc = Ex0,t,y,f,ϵ ∥ϵ − ϵθ (x0, t, τθ(y), ψθ(f), φθ(f))∥22 , (4)

where φθ(·) is the local facial extractor. 3.2.3. Consistency Training Strategy

During training, we randomly select a frame from the training frames and apply the Crop & Align [13] to extract the facial region as reference images, which is subsequently used as an identity-control signal, alongside the text as control.

Coarse-to-Fine Training. Compared to Identitypreserving image generation, video generation requires maintaining consistency in both spatial and temporal dimensions, ensuring that high and low-frequency facial information matches the reference image. To mitigate the complexity of training, we propose a hierarchical strategy where the model learns information globally before refining it locally. In the coarse-grained phase (e.g., corresponding Finding 1), we employ the global facial extractor, enabling the model to prioritize low-frequency features, such as facial contours and proportions, thereby ensuring rapid acquisition of identity information from the reference image and consistency across the video sequence. In the fine-grained phase (e.g. corresponding to Finding 2), the local facial extractor shifts the model’s focus to high-frequency details, such as the texture details of eyes and lips (e.g., intrinsic identification), improving the fidelity of facial expressions and the overall similarity of the generated face.

Dynamic Mask Loss. The objective of our task is to ensure that the identity of the person in the generated video remains consistent with the input reference image. However, Equation 1 considers the entire scene, encompassing both high- and low-frequency identity information as well as redundant background content, which introduces noise that

[Figure 32]

Ours ID-Animator

FaceSim-Arc ↑ FaceSim-Cur ↑ CLIPScore ↑ FID ↓ ID-Animator [19] 0.32 0.33 24.97 117.46 ConsisID 0.58 0.60 27.93 151.82

100%

90%

80%

UserPreference

70%

Table 1. Quantitative comparison with state-of-the-art methods. ConsisID achieve well-aligned results across most metrics. "↓" denotes lower is better. "↑" denotes higher is better.

60%

50%

40%

30%

20%

dataset for training, which differs from previous datasets [48, 69, 85] that focus only on the face. In the training phase, we set the resolution to 480×720 and extracted 49 consecutive frames at a stride of 3 from each video as training data. We set the batch size to 80, the learning rate to 3 × 10−6, and the total number of training steps to 1.8k. The randomly discarded text rate is set to 0.1, with AdamW serving as an optimizer and cosine_with_restarts as a learning rate scheduler. The training strategy is the same as Section 3.2.3. We set α and β in the dynamic cross-face loss (Le) and dynamic mask loss (Lf) to 0.5, respectively. In the inference phase, we employ DPM [62] with a sampling step of 50, and a classifier free guidance ratio of 6.0. For more detials and results, please refer to Appendix.

10%

0%

Identity Preservation

Visual Quality

Text Relevance

Motion Amplitude

- Figure 3. User Study between ConsisID and state-of-the-art methods. ConsisID is preferred by voters in all dimensions.

interferes with model training. To address this, we propose to focus the model’s attention on face regions. Specifically, we first extract the facial mask from the video, apply trilinear interpolation to map it to the latent space, and finally use this mask to constrain the computation of Lc:

Ld = M ⊙ Lc, (5)

where M represents a mask with the same shape as ϵ. However, if Equation 5 is used as the supervisory signal for all training data, the model may fail to generate a natural background during inference. To mitigate this issue, we apply Equation 5 with a probability p of α, resulting in:

Le = Ld, if p > α Lc, if p ≤ α

(6)

Dynamic Cross-Face Loss. After training with Equation 6, we observed that the model struggled to generate satisfactory results for persons not present in the data domain during inference. This issue arises because the model, trained exclusively on faces from the training frames, tends to overfit by adopting a "copy-paste" shortcut—essentially replicating the reference image without alteration. To improve the model’s generalization capability, we introduce slight Gaussian noise ζ to the reference images and use cross-face (e.g., reference images are sourced from video frames outside the training frames) as inputs with probability β:

Lf = Le where x0 · ζ, if p > β Le where xc · ζ, if p ≤ β

(7)

where x0 is the reference image extracted from the training frames, and xc is extracted from outside the training frames.

- 4. Experiments

Benchmark. Since there is an absence of an evaluation dataset, we select 30 persons who were not included in the training data and sourced five high-quality images for each ID from the internet. We then design 90 distinct prompts, encompassing a variety of expressions, actions, and backgrounds for evaluation. Building on previous works [19, 50], we evaluate four dimensions: (1). Identity Preservation: We use FaceSim-Arc [13] and introduce FaceSim-Cur, which assesses identity preservation by measuring feature differences between face regions in the generated videos and those in real face images within the ArcFace [13] and CurricularFace [27] feature spaces. (2) Visual Quality: We utilize FID [22] by calculating feature differences in the face regions between the generated frames and real face images within the InceptionV3 [64] feature space. (3) Text Relevance: We utilize CLIPScore [21] to measure the similarity between the generated videos and the input prompts. (4). Motion Amplitude: Due to the lack of reliable metrics [29, 90], we evaluate through the user study.

##### 4.2. Qualitative Analysis

In this section, we compare our method, ConsisID, with IDAnimator [19] (e.g., the only available open-source model) for tuning-free IPT2V tasks. We randomly select images and text prompts of four individuals for qualitative analysis, all of which are absent from the training data. As shown in Figure 4, ID-Animator cannot generate human body parts beyond the face and is unable to generate complex actions or backgrounds in response to text prompts (e.g., action, attribute, background), which significantly limits its practical application. In addition, the preservation of the identity is inadequate; for example, in case 1, the reference image

##### 4.1. Setup

Implementation details. ConsisID selects DiT-based generation architectures CogVideoX-5B [83] as our baseline for validation. We use an in-house human-centric

|ID-AnimatorID-AnimatorID-AnimatorID-AnimatorConsisIDConsisIDConsisIDConsisID<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>|[Figure 62]|
|---|
<br><br>A man gazing thoughtfully at far away, with a serene expression that reveals a slight furrowing of the brow and a softening around the eyes. ... The open field around him is expansive, covered in wildflowers that sway<br><br>gently in the breeze, under a vast, clear blue<br><br>sky ...<br><br>Reference Image<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>|[Figure 67]|
|---|
<br><br>... a woman sitting at an office desk ... dressed in a formal suit ... focused on her computer screen ... with shelves filled with binders ... The woman is holding a red cup ... which she drinks from before setting it down on the desk ... she then proceeds to type on the keyboard ...<br><br>Reference Image<br><br>[Figure 68]<br><br>[Figure 69]<br><br>|[Figure 70]|
|---|
<br><br>... a young woman sitting at a wooden desk, deeply engrossed in her work ... wearing glasses and has long hair that falls over her shoulders. ... as she writes with a pen ... The desk is cluttered with numerous stacks of papers and documents ...<br><br>Reference Image<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>|[Figure 76]|
|---|
<br><br>... a man sitting at a desk in front of a large screen displaying an American flag ... wearing a plaid shirt and appears to be delivering a news report … speaking,<br><br>gesturing with his hands as he talks ... a<br><br>newsroom or studio environment ...<br><br>Reference Image<br><br>4. Qualitative analysis between ConsisID and ID-Animator [19]. ID-Animator can only generate videos of the face region, the identity Preservation is poor (e.g., shape, texture). Additionally, it cannot generate specified content according to the text prompt|
|---|

ID-AnimatorConsisID

Figure and (e.g., action, decoration, background). ConsisID achieves advantages in identity preservation, visual quality, motion amplitude, and text relevance. Moreover, our ConsistID can generate more frames rather than ID-Animator (49 480×720p frames v.s. 16 512×512p frames).

appears to be processed with skin smoothing. In case 2, wrinkles have been introduced which detract from the aesthetic quality. In cases 3 and 4, the face is distorted due to the lack of low frequency information, which compromises identity consistency. In contrast, the proposed ConsisID consistently produces high-quality, realistic videos that accurately match the reference identity and adhere to prompt.

##### 4.3. Quantitative Analysis

We present a comprehensive quantitative evaluation of different methods, with results displayed in Table 1. Consistent with Figure 4, our method outperforms state-of-theart methods across five metrics. For identity preservation, ConsisID achieves a higher score by designing appropri-

ate identity signals for DiT from a frequency perspective. By contrast, ID-Animator [19] is not optimized for IPT2V and only partially retains facial features, resulting in lower FaceSim-Arc [13] and FaceSim-Cur scores. For Text Relevance, ConsisID not only controls expressions via prompts but also adjusts actions and backgrounds, achieving higher CLIPScore [21]. Regarding visual quality, the FID is presented solely as a reference due to its limited alignment [39, 40, 49, 90] with human perception. Please refer to Figure 4 and 3 for qualitative analysis of the visual quality.

##### 4.4. User Study

Building on previous work, we conduct a human evaluation using a binary voting strategy, with each questionnaire con-

[Figure 77]

[Figure 78]

... a woman sitting by a window in a cozy cafe, enjoying a cup of coffee. Her eyes softly gaze out the window, reflecting a sense of contentment ... Her lips part slightly into a serene smile, the corners of her mouth gently lifting ...

… a woman standing next to an airplane, engaged in a conversation on her cell phone. She is wearing sunglasses and a black top, and she appears to be talking seriously ...

[Figure 79]

[Figure 80]

[Figure 81]

|[Figure 82]|[Figure 83]<br><br>fail to converge|[Figure 84]|
|---|---|---|
|[Figure 85]|[Figure 86]<br><br>| |
|---|
<br><br>lack of intrinsic high-frequency feature|[Figure 87]|
|[Figure 88]|[Figure 89]|[Figure 90]|
|[Figure 91]|[Figure 92]<br><br>| |
|---|
<br><br>lack of high-frequency facial detail|[Figure 93]|
|[Figure 94]<br><br>| |
|---|
<br><br>lack of low-frequency facial contours|[Figure 95]|[Figure 96]|
|gradient explosion|gradient explosion|gradient explosion|

- (a)ConsisID

onlyhigh-freq

onlylow-freq (a)

- (b)w/oGFE

failtoconverge

[Figure 97]

[Figure 98]

[Figure 99]

- (c)w/oLFE

lackhigh-freq

[Figure 100]

[Figure 101]

[Figure 102]

- (d)w/oCFT

lacklow-freq

[Figure 103]

[Figure 104]

[Figure 105]

|(e)w/oDML|
|---|

- (e)w/oDML

lackpreservation

[Figure 106]

[Figure 107]

[Figure 108]

- (f)w/oDCL

high&lowfreq (b)

high**&lowfreq (c)

- (d)

high&low*freq

- (e)

high*&lowfreq

[Figure 109]

[Figure 110]

[Figure 111]

lackgeneralize

(f)

Figure 5. Effect of Different Components via Qualitative Analysis. Removing any component may result in the loss of high- or low-frequency facial information, or hinder the ability to modify video content based on the text prompt.

Figure 6. Effect of Different Control Signal Injection Way via Qualitative Analysis. Only (c), which injects both high & lowfreq face signals into the suitable location, performs best.

FaceSim-Arc ↑ FaceSim-Cur ↑ CLIPScore ↑ FID ↓

taining only 80 questions. Participants are required to view 40 video clips, a setup designed to improve both engagement and questionnaire validity. For the IPT2V task, each question requires participants to separately judge which option performs better in terms of Identity Preservation, Visual Quality, Text Alignment, and Motion Amplitude. This composition ensures the accuracy of the human evaluation. Owing to the extensive participant base required for this evaluation, we successfully gathered 103 valid questionnaires. The results, depicted in Figure 3, demonstrate a significant superiority of our method over ID-Animator [19], verifying the effectiveness of the designed DiT for IPT2V generation.

w/o GFE 0.05 0.05 34.86 269.88 w/o LFE 0.66 0.68 34.48 104.34 w/o CFT 0.54 0.58 34.47 144.62 w/o DML 0.62 0.67 34.23 187.78 w/o DCL 0.65 0.69 32.21 117.80 ConsisID 0.73 0.75 36.77 127.42

Table 2. Effect of Local Facial Extractor (LFE), Global Facial Extractor (GFE), coarse-to-fine training (CFT), dynamic mask loss (DML) and dynamic cross-face loss (DCL) by Automatic Metrics. Removing any of the above methods significantly reduces identity preservation, text relevance, and visual quality.

##### 4.5. Effect of the Identity Signal Injection in DiT

To assess the effectiveness of Finding 1 and Finding 2, we perform ablation studies on different methods of injecting control signals into DiT. Specifically, these experiments involved (a) injecting only low-frequency face information with key points into the noise latent, (b) injecting only highfrequency face signals within the attention block, (c) combining (a) and (b), (d) based on (c), but the low-frequency

face information does not contain key points, and (e - f) based on (c), but the high-frequency signal is injected at the output or input of the attention block. (g) injecting only high-frequency face signals before the attention block. To reduce overhead, for each identity, we only select 2 reference images each with 90 text prompts for the evaluation. The results are shown in Figure 6 and Table 3. For Finding 1, we observe that only injecting high-frequency signals (a)

Plan FaceSim-Arc ↑ FaceSim-Cur ↑ CLIPScore ↑ FID ↓

- a 0.05 0.05 34.86 269.88
- b 0.66 0.68 34.48 104.34

- c 0.73 0.75 36.77 127.42

- d 0.64 0.68 30.69 177.65
- e 0.62 0.66 33.61 164.15
- f unstable training process
- g unstable training process

Table 3. Effect of Different Control Signal Injection Way via Quantitative Analysis. Only plan c, which injects both high and low-frequency face information into the model, performs best.

greatly increases the training difficulty, causing the model to fail to converge due to the lack of low-frequency signal injection. In addition, the inclusion of facial key points (d) allows a greater focus on low-frequency information, thereby facilitating training and improving model performance. For Finding 2, when only low-frequency signals are injected

- (b), the model lacks high-frequency information. This reliance on low-frequency signals causes the generated face in the video to copy the reference image, making it difficult to control facial expressions, movements, and other features through prompts. Furthermore, injecting identity signals into the attention block input (f - g) disrupts the intended frequency domain distribution of DiT, resulting in a gradient explosion. Embedding control signals in the attention block (c) is preferable to embedding them in the output (e) because attention block processes predominantly lowfrequency information. By embedding high-frequency information internally, the attention block is guided to highlight intrinsic facial features, whereas injecting it into the output merely concatenates features without directing focus, reducing DiT’s modeling capacity. Moreover, we apply a Fourier transform to the generated videos (only the face region) to visually compare the influence of different components to extract facial information. As shown in Figure 7, the Fourier spectrum and the log amplitude of the Fourier transform reveal that injecting high or low-frequency signals can indeed enhance the corresponding frequency information of the generated face. Moreover, the low-frequency signal can be further enhanced by matching with the face key points, and injecting the high-frequency signal into the attention block has the highest feature utilization rate. Our method (c) shows strongest high and low frequency, further validating the efficiency benefit from Findings 1 and 2.

##### 4.6. Ablation on the Consistency Training Strategy

To reduce overhead, for each identity, we only select 2 reference images for the following experiments. To demonstrate the benefits of the proposed consistency training strategy, we perform ablation experiments on coarse-to-fine training (CFT), dynamic mask loss Le (DML), and dynamic crossface loss Lf (DCL), with the results presented in Figure

|[Figure 112]<br><br>(c) high & low freq<br><br>[Figure 113]<br><br>(d) high & low† freq<br><br>[Figure 114]<br><br>(a) only high-freq (b) only low-freq<br><br>[Figure 115]<br><br>[Figure 116]<br><br>(e) high† & low freq (f) Relative log amplitudes of Fourier transformed videos<br><br><br>[Figure 117]|
|---|

Figure 7. (a - e) Fourier spectrum of different id signal injection. The center area represents low frequencies and the surrounding area represents high frequencies. (f) Relative log amplitudes of Fourier transformed generated videos. A larger response value indicates a higher inclusion of frequency information. (a - f) verify the effect of our frequency decomposition.

5 and Table 2. When CFT is removed, GFE and LFE exhibit competing behaviors, complicating the model’s ability to prioritize high and low-frequency information accurately, leading to convergence at suboptimal points. Removing DML required the model to simultaneously focus on both foreground and background elements, with background noise negatively affecting training and reducing facial consistency. Similarly, the exclusion of DCL impaired the generalization capability, reducing fidelity for faces, not in the training set and reducing its effectiveness in generating identity-preserving videos as intended.

#### 5. Conclusion

In this paper, we present ConsisID, a unified framework for keeping faces consistent in video generation by frequency decomposition. It can seamlessly integrate into existing DiT-based text-to-video models, for generating highquality, editable, consistent identity-preserving videos. Extensive experiments show that ConsisID outperforms the current state-of-the-art identity-preserving T2V models. It reveals that our frequency-aware heuristic DiT-based control scheme is an optimal solution for IPT2V generation.

#### 6. Acknowledgments

We thank all the anonymous reviewers for their constructive comments. This work was supported in part by the Natural Science Foundation of China (No. 62202014, 62332002, 62425101, 62088102).

#### References

- [1] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Botsort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.
- [2] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. NeurIPS, 34:24206–24221, 2021.
- [3] Dosovitskiy Alexey. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv: 2010.11929, 2020.
- [4] Jiawang Bai, Li Yuan, Shu-Tao Xia, Shuicheng Yan, Zhifeng Li, and Wei Liu. Improving vision transformers by revisiting high-frequency components. In ECCV, pages 1–18. Springer, 2022.
- [5] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, pages 22669–22679, 2023.
- [6] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [8] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. In openai, 2024.
- [9] Qiong Cao, Li Shen, Weidi Xie, Omkar M Parkhi, and Andrew Zisserman. Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018), pages 67–74. IEEE, 2018.
- [10] Hila Chefer, Shiran Zada, Roni Paiss, Ariel Ephrat, Omer Tov, Michael Rubinstein, Lior Wolf, Tali Dekel, Tomer Michaeli, and Inbar Mosseri. Still-moving: Customized video generation without customized video data. arXiv preprint arXiv:2407.08674, 2024.
- [11] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023.
- [12] Soon Yau Cheong, Duygu Ceylan, Armin Mustafa, Andrew Gilbert, and Chun-Hao Paul Huang. Boosting camera motion control for video diffusion transformers. arXiv preprint arXiv:2410.10802, 2024.
- [13] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, pages 4690–4699, 2019.
- [14] Donglin Di, He Feng, Wenzhang Sun, Yongjia Ma, Hao Li, Wei Chen, Xiaofei Gou, Tonghua Su, and Xun Yang.

- Facevid-1k: A large-scale high-quality multiracial human face video dataset. arXiv preprint arXiv:2410.07151, 2024.
- [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. ICLR, 2024.
- [17] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. arXiv preprint arXiv:2404.16022, 2024.
- [18] Junjie He, Yifeng Geng, and Liefeng Bo. Uniportrait: A unified framework for identity-preserving singleand multi-human image personalization. arXiv preprint arXiv:2408.05939, 2024.
- [19] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, Man Zhou, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024.
- [20] Zecheng He, Bo Sun, Felix Juefei-Xu, Haoyu Ma, Ankit Ramchandani, Vincent Cheung, Siddharth Shah, Anmol Kalia, Harihar Subramanyam, Alireza Zareian, et al. Imagine yourself: Tuning-free personalized image generation. arXiv preprint arXiv:2409.13346, 2024.
- [21] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.
- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017.
- [23] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020.
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [26] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 646–661. Springer, 2016.
- [27] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In CVPR, pages 5901–5910, 2020.
- [28] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In Proceedings of the ECCV (ECCV), 2022.

- [29] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [30] Glenn Jocher, Ayush Chaurasia, Alex Stoken, Jirka Borovec, Yonghye Kwon, Kalen Michael, Jiacong Fang, Zeng Yifu, Colin Wong, Diego Montes, et al. ultralytics/yolov5: v7. 0yolov5 sota realtime instance segmentation. Zenodo, 2022.
- [31] Tero Karras. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.
- [32] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.
- [33] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, pages 1931–1941, 2023.
- [34] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, pages 12888–12900. PMLR, 2022.
- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023.
- [36] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In CVPR, pages 8640–8650, 2024.
- [37] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [38] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. arXiv preprint arXiv:2411.19108, 2024.
- [39] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440, 2023.
- [40] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-tovideo generation. NeurIPS, 36, 2024.
- [41] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [42] Yue Ma, Xiaodong Cun, Yingqing He, Chenyang Qi, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Magic-

- stick: Controllable video editing via control handle transformations. arXiv preprint arXiv:2312.03047, 2023.
- [43] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4117–4125, 2024.
- [44] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Chenyang Qi, Chengfei Cai, Xiu Li, Zhifeng Li, HeungYeung Shum, Wei Liu, et al. Follow-your-click: Opendomain regional image animation via short prompts. arXiv preprint arXiv:2403.08268, 2024.
- [45] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12, 2024.
- [46] Ze Ma, Daquan Zhou, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong, Kurt Keutzer, and Jiashi Feng. Magic-me: Identity-specific video customized diffusion. arXiv preprint arXiv:2402.09368, 2024.
- [47] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.
- [48] Arsha Nagrani, Joon Son Chung, Weidi Xie, and Andrew Zisserman. Voxceleb: Large-scale speaker verification in the wild. Computer Speech & Language, 60:101027, 2020.
- [49] Mayu Otani, Riku Togashi, Yu Sawai, Ryosuke Ishigami, Yuta Nakashima, Esa Rahtu, Janne Heikkilä, and Shin’ichi Satoh. Toward verifiable and reproducible human evaluation for text-to-image generation. In CVPR, pages 14277–14286,

- 2023.

[50] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

- 2024.

- [51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.
- [52] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pages 8821– 8831, 2021.
- [53] Chen Rao, Guangyuan Li, Zehua Lan, Jiakai Sun, Junsheng Luan, Wei Xing, Lei Zhao, Huaizhong Lin, Jianfeng Dong, and Dalong Zhang. Rethinking video deblurring with wavelet-aware dynamic transformer and diffusion model. arXiv preprint arXiv:2408.13459, 2024.
- [54] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

- [55] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.
- [56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [57] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023.
- [58] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In CVPR, pages 6527–6536, 2024.
- [59] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [60] Yujun Shi, Jun Hao Liew, Hanshu Yan, Vincent YF Tan, and Jiashi Feng. Instadrag: Lightning fast and accurate dragbased image editing emerging from videos. arXiv preprint arXiv:2405.13722, 2024.
- [61] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In CVPR, pages 4733– 4743, 2024.
- [62] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pages 2256–

2265. PMLR, 2015.

- [63] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv, 2020.
- [64] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, pages 2818–2826, 2016.
- [65] Zhenyu Tang, Junwu Zhang, Xinhua Cheng, Wangbo Yu, Chaoran Feng, Yatian Pang, Bin Lin, and Li Yuan. Cycle3d: High-quality and consistent image-to-3d generation via generation-reconstruction cycle. arXiv preprint arXiv:2407.19548, 2024.
- [66] Yuchuan Tian, Zhijun Tu, Hanting Chen, Jie Hu, Chao Xu, and Yunhe Wang. U-dits: Downsample tokens in u-shaped diffusion transformers. arXiv preprint arXiv:2405.02730, 2024.
- [67] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [68] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.

- [69] Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. One-shot free-view neural talking-head synthesis for video conferencing. In CVPR, pages 10039–10049, 2021.
- [70] Zhao Wang, Aoxue Li, Enze Xie, Lingting Zhu, Yong Guo, Qi Dou, and Zhenguo Li. Customvideo: Customizing textto-video generation with multiple subjects. arXiv preprint arXiv:2401.09962, 2024.
- [71] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In ICCV, pages 15943–15953, 2023.
- [72] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In CVPR, pages 6537– 6549, 2024.
- [73] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. arXiv preprint arXiv:2406.17758, 2024.
- [74] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2025.
- [75] Jing Xiong, Gongye Liu, Lun Huang, Chengyue Wu, Taiqiang Wu, Yao Mu, Yuan Yao, Hui Shen, Zhongwei Wan, Jinfa Huang, et al. Autoregressive models in vision: A survey. arXiv preprint arXiv:2411.05902, 2024.
- [76] Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. Easyanimate: A high-performance long video generation method based on transformer architecture. arXiv preprint arXiv:2405.18991, 2024.
- [77] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024.
- [78] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22412–22423, 2023.
- [79] Zhiyuan Yan, Yuhao Luo, Siwei Lyu, Qingshan Liu, and Baoyuan Wu. Transcending forgery specificity with latent space augmentation for generalizable deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8984–8994, 2024.
- [80] Zhiyuan Yan, Taiping Yao, Shen Chen, Yandan Zhao, Xinghe Fu, Junwei Zhu, Donghao Luo, Li Yuan, Chengjie Wang, Shouhong Ding, et al. Df40: Toward next-generation deepfake detection. arXiv preprint arXiv:2406.13495, 2024.
- [81] Zhiyuan Yan, Yandan Zhao, Shen Chen, Mingyi Guo, Xinghe Fu, Taiping Yao, Shouhong Ding, and Li Yuan. Generalizing deepfake video detection with plug-and-play:

- Video-level blending and spatiotemporal adapter tuning. arXiv preprint arXiv:2408.17065, 2024.
- [82] Shuo Yang, Kun-Peng Ning, Yu-Yang Liu, Jia-Yu Yao, Yong-Hong Tian, Yi-Bing Song, and Li Yuan. Is parameter collision hindering continual learning in llms? arXiv preprint arXiv:2410.10179, 2024.
- [83] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [84] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [85] Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, and Wayne Wu. Celebv-text: A large-scale facial text-video dataset. In CVPR, pages 14805–14814, 2023.
- [86] Wangbo Yu, Chaoran Feng, Jiye Tang, Xu Jia, Li Yuan, and Yonghong Tian. Evagaussians: Event stream assisted gaussian splatting from blurry images. arXiv preprint arXiv:2405.20224, 2024.
- [87] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.
- [88] Yongsheng Yu, Ziyun Zeng, Hang Hua, Jianlong Fu, and Jiebo Luo. Promptfix: You prompt and we fix the photo. arXiv preprint arXiv:2405.16785, 2024.
- [89] Shenghai Yuan, Jinfa Huang, Yujun Shi, Yongqi Xu, Ruijie Zhu, Bin Lin, Xinhua Cheng, Li Yuan, and Jiebo Luo. Magictime: Time-lapse video generation models as metamorphic simulators. arXiv preprint arXiv:2404.05014, 2024.
- [90] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Rui-Jie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. NeurIPS, 37:21236–21270, 2025.
- [91] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pages 3836–3847, 2023.
- [92] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In ECCV, pages 1–21. Springer, 2022.
- [93] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with a highresolution audio-visual dataset. In CVPR, pages 3661–3670, 2021.
- [94] Zhenghao Zhang, Junchao Liao, Menghao Li, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. arXiv preprint arXiv:2407.21705, 2024.
- [95] Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng

- Chen, Harry Yang, et al. Videogen-of-thought: A collaborative framework for multi-shot video generation. arXiv preprint arXiv:2412.02259, 2024.
- [96] Zangwei Zheng, Xiangyu Peng, and Yang You. Open-sora: Democratizing efficient video production for all. In Github, 2024.
- [97] Yuan Zhou, Qiuyue Wang, Yuxuan Cai, and Huan Yang. Allegro: Open the black box of commercial-level video generation model. arXiv preprint arXiv:2410.15458, 2024.
- [98] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. Celebvhq: A large-scale video facial attributes dataset. In ECCV, pages 650–667. Springer, 2022.

## Identity-Preserving Text-to-Video Generation by Frequency Decomposition Supplementary Material

- 1. ConsisID Dataset 1
- 2. Additional Experimental Results 2

- 2.1. Comparison with Closed-source Method . . 2
- 2.2. Comparison with Tuning-based Methods . . 3
- 2.3. Comparison with I2V Methods . . . . . . . 4
- 2.4. Fine-grained Ablation Study . . . . . . . . . 4
- 2.5. Ablation on the Number of Inference Steps . 5
- 2.6. Increasing Inference Speed . . . . . . . . . . 6
- 2.7. Generating Higher FPS Videos . . . . . . . . 6
- 2.8. Style Transfer Applications . . . . . . . . . 6
- 2.9. More Cases on ID-preserving Videos . . . . 6

- 3. Additional Experimental Details 6

- 3.1. Visualization of Different Injection Methods 6
- 3.2. Validation of the Automatic Metrics . . . . . 6
- 3.3. Details of Resource . . . . . . . . . . . . . . 6
- 3.4. Details of Evaluation Models . . . . . . . . 6
- 3.5. Details of Implementation . . . . . . . . . . 8
- 3.6. Details of Human Evaluation . . . . . . . . . 8

- 4. Additional Statement 8

- 4.1. Support for Key Findings . . . . . . . . . . 8
- 4.2. Justification of the Fine-grained Feature . . . 9
- 4.3. The feature of Diffusion Transformer . . . . 9
- 4.4. Generalization to Image Generation . . . . . 9
- 4.5. Ethics Statement . . . . . . . . . . . . . . . 9
- 4.6. Reproducibility Statement . . . . . . . . . . 9
- 4.7. Copyright Statement . . . . . . . . . . . . . 9
- 4.8. Limitations and Future Work . . . . . . . . . 9

#### 1. ConsisID Dataset

We propose a data pipeline to process and construct a highquality ID-preservation video dataset as shown in Figure 2. Data Curation Most existing identity-preserving datasets are image-centric [9, 31, 48, 59], with only a few focusing on videos [14, 85, 98]. However, these datasets primarily target facial regions, often cropping out relevant background content (e.g., talking heads [69, 93]), which limits their broader applicability. To address this, we propose a pipeline to construct identity-preserving videos suitable for daily-life scenarios, as depicted in Figure 1. In line with the approach used by [89], we first constructed a set of search keywords (e.g., "human," "woman," "man") and used them to retrieve videos from the internet. During this process, we excluded videos with few views or likes to ensure quality. Next, following [90], we apply PySceneDetect, Aesthetic, and Motion Score to filter out low-quality clips, ultimately getting a dataset of 130K high-quality clips.

||[Figure 118]|[Figure 119]|[Figure 120]|[Figure 121]|
|---|---|---|---|
|[Figure 122]|[Figure 123]|[Figure 124]|[Figure 125]|
<br><br>|[Figure 126]|[Figure 127]|
|---|---|
|[Figure 128]|[Figure 129]|
<br><br>(a) Current Datset<br>(b) Our Dataset<br><br><br>Content: only face region<br><br>Content: Flexible<br><br>[Figure 130]<br><br>[Figure 131]|
|---|

Figure 1. Comparison between our in-house data and current Human Centric Video Dataset [19, 48, 85]. Our dataset is more flexible and diverse compared to previous ones.

Multi-view Face Filtering The purity of internet-sourced data is typically low, as full videos often include only brief segments featuring facial content. To address this, we first apply YOLO-Box [30] to extract frame-by-frame bounding box (bbox) information for the categories "face," "head," and "person". Video clips containing all three bounding boxes are retained. We then use YOLO-Pose [30] to detect facial keypoints (e.g. left eye, right eye, left ear, right ear, nose) and filter out video clips with low keypoint density to obtain clean ID-preservation video clips. To mitigate potential YOLO [30] errors, we set a tolerance threshold α. For instance, if β frames among frames 0 to 49 lack any of the three bounding boxes or valid keypoints, and β < α, frames 0 to 49 are still retained as a complete video clip. To further enhance data quality, we discard video clips in which the "face" bbox occupies less than 6% of the frame.

Identical Verification A video may include multiple individuals, necessitating the assignment of a unique identifier to each person for subsequent training. Existing video tracking algorithms [1, 55, 92] lack robustness, often resulting in missed or incorrect detections. To address this, we utilize the previously obtained frame-by-frame bound-

||IDID VerificationVerification<br><br>| | |
|---|---|---|
|[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>| |
|---|
<br><br>1<br><br>| |
|---|
<br><br>1<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>1<br><br>1<br><br>|| |
|---|
<br><br>22<br><br>[Figure 138]<br><br>[Figure 139]|[Figure 140]<br><br>| |
|---|
<br><br>3<br><br>| |
|---|
<br><br>2 1<br><br>| |
|---|
<br><br>| |
|---|
<br><br>2 3<br><br>[Figure 141]<br><br>2 1<br><br>2|
<br><br>|Multi-Multi-view Filtering<br><br>| | | | |
|---|---|---|---|---|
|[Figure 142]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>[Figure 143]<br><br>99.3%<br><br>99.8%|[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>|| |
|---|
<br><br>[Figure 150]<br><br>99.2%<br><br>[Figure 151]|| |
|---|
<br><br>99.5%<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>99.5% 99.5%<br><br>99.4% 98.8%<br><br>99.5%|[Figure 152]<br><br>99.1%<br><br>[Figure 153]<br><br>|
<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>| |
|---|
<br><br>1<br><br>| |
|---|
<br><br>2<br><br>[Figure 179]<br><br>| |
|---|
<br><br>2 1<br><br>[Figure 180]<br><br>| |
|---|
<br><br>2<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>1 2<br><br>3 1<br><br>Multi-<br><br>ID Verification<br><br>| |
|---|
<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>Data Curation<br><br>Refine the ID that<br><br>exceed the total people depicted in the video.<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>1<br><br>2 2 1<br><br>[Figure 218]<br><br>[Figure 219]<br><br>2<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>1 2<br><br>Segmentation according to ID<br><br>3 1<br><br>Multi-view<br><br>Face Filtering<br><br>ID Verification<br><br>Segmentation<br><br>and Captioning<br><br>Cutting Pose<br><br>Bounding Box<br><br>Deletion Track<br><br>Recover Fail<br><br>Track Refine<br><br>Figure 2. Overview of the proposed Identity-Preserving Video Data Processing Pipeline. First, we identify video clips with high facial density using bounding boxes (bbox) and key points. Next, we implement a tracking algorithm for ID verification, optimizing individual|
|---|

tracking IDs through the previous bounding box. Finally, we generate masks for each individual according to their unique IDs.

Video Resolution

Word Count Range

###### Distribution of Aesthetic Scores

Video Durations

60

6%

7%

14%

50

Percentage

40

31%

1%

0.1%

30

20

10

56%

0

85%

99.9%

Score Interval

<4 4-5 5-6 >6

0-80 80-110 110-140 >140

720P 1080P

0-15 15-30 30+

- Figure 3. Video Clips Statistics of our Dataset. The dataset includes a diverse range of categories, durations and caption lengths, with most of the videos being in 1080P resolution.

ing box (bbox) to compute a unique identifier for each individual. Specifically, using the "face" bbox as an example, we first determine the maximum number of individuals, m, present in the video based on the bbox information, and then assign a unique identifier (not exceeding m) to each "face" bbox in the initial frame. We then apply forward propagation: for the n-th frame, each bbox is assigned a unique identifier, corresponds to those from frame n − 1, based on the Intersection over Union (IoU) between all bboxes of frames n − 1 and n. After completing forward propagation for all frames, we perform backward propagation: for each bbox in frame n, we assign a unique identifier that corresponds to frames n − 1 and n + 1, again based on the IoU between all bboxes of frames n and n + 1. Ultimately, each bbox is assigned a unique identifier, facilitating precise tracking of each person’s location across video.

responding masks for each person’s "face," "head," and "person." Subsequently, SAM2’s [54] tracking signals are used to further refine the unique identifiers assigned earlier. We then employ Time-Aware Annotation [90], leveraging Qwen2-VL-72B [67], to produce high-quality captions for the video clips. Data statistics are presented in Figure 3.

#### 2. Additional Experimental Results

##### 2.1. Comparison with Closed-source Method

In this section, we compare our ConsisID with Vidu 1.5 [6], the only available video generation model capable of performing the Identity-Preserving Text-to-Video (IPT2V) task. Notably, Vidu 1.5 [6] is a closed-source model, and it remains unclear whether it is tuning-free or tuning-based. As Vidu’s API is not publicly available, we can only manually collect output to evaluate its performance. Due to constraints on large-scale generation, we randomly select 60 prompts from the evaluation dataset. Each prompt is paired with a unique reference image, resulting in the generation of

Segmentation and Captioning To facilitate the application of dynamic mask loss, we first input the highestconfidence bounding box (bbox) for each category obtained in the previous step into SAM2 [54] to generate the cor-

||Empty|
|---|
<br><br>|[Figure 232]|
|---|
<br><br>|[Figure 233]|
|---|
<br><br>|[Figure 234]|
|---|
<br><br>|[Figure 235]|
|---|
<br><br>|[Figure 236]|
|---|
<br><br>|Empty|
|---|
<br><br>|[Figure 237]|
|---|
<br><br>|[Figure 238]|
|---|
<br><br>|[Figure 239]|
|---|
<br><br>|[Figure 240]|
|---|
<br><br>|[Figure 241]|
|---|
<br><br>|Empty|
|---|
<br><br>|[Figure 242]|
|---|
<br><br>|Empty|
|---|
<br><br>|[Figure 243]|
|---|
<br><br>|[Figure 244]<br><br>[Figure 245]|
|---|
<br><br>Sample 1<br><br>|[Figure 246]<br><br>[Figure 247]|
|---|
<br><br>|[Figure 248]<br><br>[Figure 249]|
|---|
<br><br>|[Figure 250]<br><br>[Figure 251]|
|---|
<br><br>|[Figure 252]<br><br>[Figure 253]|
|---|
<br><br>|[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]|
|---|
<br><br>Sample 2|
|---|

- Figure 4. Seamlessly integrate the frame interpolation model [28] into ConsisID. The newly added frames are clear, which shows that the original video generated by ConsisID is coherent.

FaceSim-Arc ↑ FaceSim-Cur ↑ CLIPScore ↑ FID ↓ Vidu 1.5 [6] 0.36 0.39 32.87 215.42 ConsisID 0.52 0.54 33.08 163.68

Table 1. Quantitative Comparison with Close-source Method. ConsisID performs best on most metrics, especially FaceSim, which is the most important metric of IPT2V. "↓" denotes lower is better. "↑" denotes higher is better.

60 videos by each method. The quantitative results are presented in Table 1, indicating that ConsisID consistently outperforms Vidu 1.5 [6] across all four automatic metrics. The qualitative analysis, illustrated in Figure 13, further supports these findings. Both ConsisID and Vidu 1.5 follow prompts effectively to generate the specified actions, backgrounds, and attributes. However, Vidu’s outputs contain noticeable artifacts (e.g., the flower in case 1 and the tree in case 2) and exhibit lower visual quality compared to ConsisID. Moreover, in terms of preserving identity features, Vidu 1.5 retains only high-frequency facial characteristics (e.g., hair color and hairstyle in case 1, facial shape in case 2), while failing to maintain intrinsic ID features essential for IPT2V. Consequently, individuals generated by Vidu 1.5 do not align consistently with the reference images.

##### 2.2. Comparison with Tuning-based Methods

In this section, we compare ConsisID with several tuningbased methods, including DreamVideo [72], MotionBooth [73] and Magic-Me [46]. These methods require parameter tuning for each new identity input prior to inference. Due to computational and time constraints, we randomly select

FaceSim-Arc ↑ FaceSim-Cur ↑ CLIPScore ↑ FID ↓ Speed (s) ↓

DreamVideo [72] 0.03 0.03 26.03 237.91 3600+ MotionBooth [73] 0.05 0.06 24.42 287.90 600+

Magic-Me [46] 0.09 0.10 23.14 237.35 500+ ConsisID 0.46 0.47 27.45 181.97 100+

- Table 2. Quantitative Comparison with Tuning-based Methods (on single Nvidia H100). ConsisID can generate high-quality idpreserving videos in a very short time, achieving the best balance among methods. "↓" denotes lower is better. "↑" higher is better.

FaceSim-Arc ↑ FaceSim-Cur ↑ CLIPScore ↑ FID ↓ CogVideoX-5B-I2V [83] 0.37 0.38 28.53 201.69

EasyAnimate v4 [76] 0.15 0.15 27.95 235.68 OpenSora-Plan v1.3 [37] 0.31 0.32 27.25 224.99

DynamiCrafter512 [74] 0.25 0.26 29.76 212.13 ConsisID 0.46 0.47 27.45 181.97

- Table 3. Quantitative Comparison with I2V Methods. End-toend methods yield higher-quality id-preserving videos than twostage methods. "↓" denotes lower is better. "↑" higher is better.

1 reference image per ID from our evaluation dataset and each image is evaluated on 45 randomly selected prompts, resulting in a total of 1,350 video sequences generated per method. The results are presented in Figure 13 and Table 2. As shown, ConsisID consistently outperforms existing tuning-based methods despite having a shorter inference time (on single Nvidia H100). This superior performance is likely due to the fact that the latter are designed for opendomain tasks (e.g., people, objects, animals, plants), which encompass a wide range and consequently fail to effectively capture the nuanced distinctions of individual identity features, leading to suboptimal results.

||[Figure 258]|
|---|
|[Figure 259]|
|[Figure 260]|
|[Figure 261]|
|[Figure 262]|
<br><br>|[Figure 263]|
|---|
|[Figure 264]|
|[Figure 265]|
|[Figure 266]|
|[Figure 267]|
<br><br>|[Figure 268]<br><br>[Figure 269]<br><br>CogVideoX-5B-I2V|
|---|
|[Figure 270]<br><br>[Figure 271]<br><br>EasyAnimate v4|
|[Figure 272]<br><br>[Figure 273]<br><br>OpenSora-Plan v1.3|
|[Figure 274]<br><br>[Figure 275]<br><br>DynamiCrafter512|
|[Figure 276]<br><br>[Figure 277]<br><br>ConsisID|
<br><br>|[Figure 278]|
|---|
<br><br>The woman, elegantly posed and smiling towards the camera, stands amidst a breathtaking landscape, with rolling hills bathed in the warm glow of the golden hour creating an idyllic backdrop. Bathed in soft light and the play of shadows, the scene captures a sense of serenity and momentary bliss. A gentle breeze ruffles through the woman's<br><br>hair, adding a sense of movement and vitality to the harmonious composition.|
|---|

- Figure 5. Qualitative comparison with I2V Methods. It is clear that standard I2V models encounter challenges in generating highquality identity-preserving videos.

FaceSim-Arc ↑ FaceSim-Cur ↑ CLIPScore ↑ FID ↓

w/o CLIP 0.40 0.38 26.87 142.23 w/o FaceExtractor 0.36 0.37 27.76 193.99

w/o Noise ζ 0.37 0.38 27.52 193.46

Loss † 0.35 0.37 26.53 167.11 Loss †† 0.39 0.38 26.89 216.69

Loss ‡ 0.29 0.30 24.77 150.80 ConsisID † 0.40 0.40 27.38 256.29

ConsisID 0.46 0.47 27.45 181.97

- Table 4. Fine-grained Ablation Study. Each component of ConsisID plays a crucial role in generating high-quality videos. "↓" denotes lower is better. "↑" higher is better.

##### 2.3. Comparison with I2V Methods

In this section, we compare ConsisID with several imageto-video methods, leveraging the identity-preserving image model [36]. As shown in Figure 5, Table 3 and Table 5, the I2V foundation models [74, 76, 83] clearly demonstrates considerable temporal decay in identity preservation. While OpenSora-Plan [37] achieves higher fidelity due to its lower motion amplitude, it does not align with the real video. In contrast, only the proposed ConsisID consistently preserves identity throughout the entire video.

||[Figure 279]<br><br>[Figure 280]<br><br>ConsisID|
|---|
|[Figure 281]<br><br>[Figure 282]<br><br>ConsisID+TeaCache†|
|[Figure 283]<br><br>[Figure 284]<br><br>ConsisID+TeaCache‡|
<br><br>|[Figure 285]|
|---|
|[Figure 286]|
|[Figure 287]|
<br><br>|[Figure 288]|
|---|
|[Figure 289]|
|[Figure 290]|
<br><br>|[Figure 291]|
|---|
<br><br>A woman adorned with a delicate flower crown, is standing amidst a field of gently swaying wildflowers. Her eyes sparkle with a serene gaze, and a faint smile graces her lips, suggesting a moment of peaceful contentment. The shot is framed from the waist up, highlighting the gentle breeze lightly tousling her hair. The background reveals an expansive meadow under a bright blue sky, capturing the tranquility of a sunny afternoon.|
|---|

Figure 6. Increasing Inference Speed with the help of TeaCache [38]. ConsisID can seamlessly integrate into existing inference acceleration frameworks without much quality degradation, demonstrating its strong scalability.

##### 2.4. Fine-grained Ablation Study

In this section, we conduct more detailed ablation studies. Due to limited computational resources, we extract approximately 30K video samples from the ConsisID-Dataset for the following experiments. The batch size is reduced to 16, and the training duration is set to one epoch, with all other settings remaining unchanged. The experiments including:

Ablation on the Local Facial Extractor : This experiment aims to assess the distinct roles of CLIP and the Face Extractor in capturing high-frequency facial features, specifically w/o CLIP and w/o Face Extractor.

Ablation on Noise in Cross-Face Loss : This experiment examines whether introducing subtle noise into the input image improves the model’s generalization capability, specifically w/o Noise ζ.

Ablation on the Weight Ratio of Loss : This experiment investigates the individual roles of MSE Loss, Dynamic Mask Loss, and Dynamic Cross Loss to the training process. Loss † corresponds to a mix ratio of 2:1:1, Loss †† to a ratio of 1:2:1, and Loss ‡ to a ratio of 1:1:2.

The results are shown in Figure 7 and Table 4, where ConsisID † represents the complete model trained on the subset data. It can be concluded that CLIP is essential for acquiring the semantic information necessary for video editing, as removing it leads to a significant decrease in the CLIPScore. FaceExtractor plays a critical role in maintaining facial consistency, with its removal resulting in a drop in FaceSim scores. Both Noise ζ and Dynamic Cross Loss contribute positively to the model’s generalization performance; however, an overemphasis on the latter may prevent convergence. MSE Loss accelerates convergence, while

||[Figure 292]|
|---|
<br><br>The video shows a confident man riding a horse, his posture straight and assured as he expertly guides the animal. He adjusts his wide-brimmed hat with a swift, practiced motion, his eyes narrowing slightly against the sun. His jaw is set in determination, and his lips curl into a faint smirk, exuding control and poise. The horse's mane flows in the breeze, while the man maintains a steady grip on the reins, his calm demeanor contrasting with the energy of the ride. The scene conveys a sense of mastery and confidence in the open landscape.<br><br>|[Figure 293]<br><br>[Figure 294]<br><br>LossLoss ††<br><br>[Figure 295]<br><br>[Figure 296]|
|---|
|[Figure 297]<br><br>[Figure 298]<br><br>LossLoss ‡<br><br>[Figure 299]<br><br>[Figure 300]|
|[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>ConsisID †|
|[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]<br><br>ConsisID|
<br><br>|[Figure 309]<br><br>[Figure 310]|
|---|
|[Figure 311]<br><br>[Figure 312]|
|[Figure 313]<br><br>[Figure 314]|
|[Figure 315]<br><br>[Figure 316]|
<br><br>|[Figure 317]<br><br>[Figure 318]|
|---|
|[Figure 319]<br><br>[Figure 320]|
|[Figure 321]<br><br>[Figure 322]|
|[Figure 323]<br><br>[Figure 324]|
<br><br>|[Figure 325]<br><br>[Figure 326]|
|---|
|[Figure 327]<br><br>[Figure 328]|
|[Figure 329]<br><br>[Figure 330]|
|[Figure 331]<br><br>[Figure 332]|
<br><br>|[Figure 333]<br><br>[Figure 334]|
|---|
|[Figure 335]<br><br>[Figure 336]|
|[Figure 337]<br><br>[Figure 338]|
|[Figure 339]<br><br>[Figure 340]|
<br><br>|[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]<br><br>w/o Noise ζ|
|---|
|[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]<br><br>[Figure 348]<br><br>Loss †|
<br><br>|[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]<br><br>[Figure 352]<br><br>w/o CLIP|
|---|
|[Figure 353]<br><br>[Figure 354]<br><br>[Figure 355]<br><br>[Figure 356]<br><br>w/o FaceExtractor|
|
|---|

- Figure 7. Fine-grained Ablation Study. After the removal of CLIP, the model loses essential semantic information necessary for editing. FaceExtractor and MSE Loss play a critical role in maintaining consistency of facial features. Over-reliance on Dynamic Mask Loss may result in the loss of background information. Noise ζ and Dynamic Cross Loss are vital for the model’s generalization; without them, the model struggles to produce results beyond the training data.

Model Memory Paramaters Speed ID-Animator [19] 8GB 1.5B ~11s

CogVideoX-5B-I2V †† [83] 42GB 5.2B ~210s+7s EasyAnimate v4 †† [76] 15GB 1.8B ~78s+7s OpenSora-Plan v1.3 †† [37] 37GB 2.6B ~282s+7s

DynamiCrafter †† [74] 17GB 2.4B ~26s+7s CogVideoX-5B-I2V [83] 42GB 5.2B ~210s

ConsisID 44GB 5.7B ~214s

ConsisID+TeaCache † [38] 44GB 5.7B ~137s ConsisID+TeaCache ‡ [38] 44GB 5.7B ~103s

- Table 5. Computation Overhead of Different Methods (on single Nvidia A100). Compared to the baseline, ConsisID introduces only a minimal overhead to achieve the IPT2V task. †† means generating video with the help of PhotomakerV2 [36].

Dynamic Mask Loss enhances focus on facial features, thereby improving identity consistency. However, excessive reliance on Dynamic Mask Loss may lose the ability to generate background content. The complete model, integrating all components, yields optimal performance.

##### 2.5. Ablation on the Number of Inference Steps

To assess the impact of varying the number of inference steps on model performance, we conduct an ablation study within the inference phase of ConsisID. Given constraints on computing resources, 60 prompts are randomly selected from the evaluation dataset. Each prompt is paired with

||[Figure 357]|
|---|
<br><br>|[Figure 358]|
|---|
<br><br>|[Figure 359]|
|---|
<br><br>|[Figure 360]|
|---|
<br><br>|[Figure 361]|
|---|
<br><br>|[Figure 362]|
|---|
<br><br>|[Figure 363]|
|---|
<br><br>|[Figure 364]|
|---|
<br><br>|[Figure 365]|
|---|
<br><br>|[Figure 366]|
|---|
<br><br>|[Figure 367]|
|---|
<br><br>|[Figure 368]|
|---|
<br><br>The video, in a beautifully crafted animated style, features a confident woman riding a horse through a lush forest clearing. Her expression is focused yet serene as she adjusts her wide-brimmed hat with a practiced hand. She wears a flowy bohemian dress ......<br><br>The animation features a whimsical portrait of a balloon seller standing in a gentle breeze, captured with soft, hazy brushstrokes that evoke the feel of a serene spring day. His face is framed by a gentle smile, his eyes ......<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>The video, in a beautifully crafted animated style, features a confident woman riding a horse through a lush forest clearing. Her expression is focused yet serene as she adjusts her wide-brimmed hat with a practiced hand. She wears a flowy bohemian dress ......|
|---|

Figure 8. Style Transfer Applications. Despite being trained on real facial data, ConsisID demonstrates a remarkable generalization by generating anime-style videos in a zero-shot manner.

a unique reference image, leading to the generation of 60 videos for each setting. Using a fixed random seed, we vary the inversion step parameter t across values of 25, 50, 75, 100, 125, 150, 175, and 200. The results are illustrated in Figure 9 and Table 6. Although theoretical expectations [24, 62, 63] suggest that increasing the number of inference steps would continuously enhance the generation quality, our findings indicate a non-linear relationship where quality peaks at t = 50 and subsequently declines. Specifi-

FaceSim-Arc ↑ FaceSim-Cur ↑ CLIPScore ↑ FID ↓ Speed (s) ↓

t = 25 0.50 0.53 30.43 184.44 50+ t = 50 0.52 0.54 33.08 163.68 100+ t = 75 0.43 0.52 31.92 200.86 160+

t = 100 0.46 0.55 32.25 212.74 220+ t = 125 0.42 0.51 32.38 185.85 270+ t = 150 0.34 0.40 32.41 186.56 330+ t = 175 0.35 0.42 29.98 186.99 390+ t = 200 0.33 0.39 31.18 166.79 440+

- Table 6. Effect of the Inference Steps by Quantitative Analysis (on Nvidia H100). "↓" denotes lower is better. "↑" higher is better.

cally, at t = 25, the model produces incomplete garlands; at t = 75, it fails to generate upper body clothing; beyond t = 125, it loses critical low-frequency facial information, resulting in distorted facial features; and beyond t = 150, the visual clarity progressively deteriorates. We infer that the initial stages of denoising process are dominated by lowfrequency information, such as generating the outline of a face, while the later stages focus on high-frequency details, such as intrinsic facial features. t = 50 is just the optimal setting to balance these two stages.

##### 2.6. Increasing Inference Speed

As shown in Table 5, ConsisID requires about 44 GB of GPU memory to decode 49 frames with output resolution 720x480 (W × H), while baseline needs 42 GB of GPU memory. The inference time of ConsisID is almost identical to the baseline, with only an additional 0.5B parameters, yet it achieves the IPT2V functionality that the baseline cannot, demonstrating the efficiency of the proposed method. In addition, ConsisID can seamlessly integrate with training-free inference acceleration strategies, achieving minimal degradation in visual quality, as illustrated in Figure 6. Specifically, TeaCache† corresponds to setting rel_l1_thresh = 0.1, while TeaCache‡ corresponds to setting rel_l1_thresh = 0.15. The rel_l1_thresh regulates the trade-off between generation quality and speed.

##### 2.7. Generating Higher FPS Videos

Due to limited resources, ConsisID can only generate 49 frames at 8 frames per second (fps). Although the resulting video is coherent, the frame rate falls below the human perceptual threshold for smoothness, which is approximately 16 fps. Therefore, a frame interpolation model [28] can be applied to post-process the output video, increasing the frame rate to 16 fps, as illustrated in Figure 4. The results indicate that after interpolation, the video maintains a high level of clarity, suggesting that the original frames generated at fps 8 are sufficiently coherent.

##### 2.8. Style Transfer Applications

- Figure 8 demonstrates the generalization capability of ConsisID. Beyond generating realistic, customized videos, the

framework effectively processes stylized prompts while preserving the identity of animated characters in a zero-shot manner. These capabilities are expected to significantly advance video content creation.

- 2.9. More Cases on ID-preserving Videos

Due to space constraints, to assess the robustness and generalizability of our method, we present more ID-preserving video generation results in Figures 14, 15 and 16, covering different people and different text prompts. ConsisID not only generates faces that match the identity of the reference image but also adheres to the text prompt, allowing for control over the character’s expressions, attire, actions, age, background, and even camera angles (e.g., detailed closeups, wide panoramic views). These results substantiate the effectiveness of the Global / Local Facial Extractors, and the Consistency Training Recipe can enhance performance.

- 3. Additional Experimental Details

- 3.1. Visualization of Different Injection Methods

To enhance the explanation of Identity Signal Injection in DiT presented in the main text, we visualized various schemes, as shown in Figure 10. For simplicity, the visualization of the text branch is omitted. ConsisID employs scheme (c), which injects high-frequency information between the Attention and FFN modules, while integrating low-frequency signals (with facial key points) into the shallow layers of the network, achieving optimal result.

- 3.2. Validation of the Automatic Metrics

In order to assess the effectiveness of the different metrics, we preform a cross-validation using the results of the user study. Specifically, we obtain FaceSim-Arc [13], FaceSimCur, CLIPScore [21] and FID [22] scores for each video in the questionnaire. We then identify the highest scoring option for each metric and compared these results with the questionnaire responses, as shown in Figure 11. Although CLIPScore [21] reflect model performance reasonably well, their alignment with human perception remains limited. In particular, FID [22] showed an inverse relationship with human perception, with the lower quality ID-Animator [19] receiving a higher score. Therefore, the quantitative results presented in the main text should be interpreted cautiously.

- 3.3. Details of Resource

We employ Nvidia H100 (x40) and A100 (x8) for all the experiments. All implementations are conducted on the basis of the official code using the PyTorch framework.

- 3.4. Details of Evaluation Models

Vidu [6]. Vidu1.5 is currently the only closed-source model supporting tuning-free IPT2V. It can generate videos

|[Figure 369]|
|---|

A woman adorned with a delicate flower crown, is standing amidst a field of gently swaying wildflowers. Her eyes sparkle with a serene gaze, and a faint smile graces her lips, suggesting a moment of peaceful contentment. The shot is framed from the waist up, highlighting the gentle breeze lightly tousling her hair. The background reveals an expansive meadow under a bright blue sky, capturing the tranquility of a sunny afternoon.

|[Figure 370]|[Figure 371]|[Figure 372]|
|---|---|---|
|[Figure 373]|[Figure 374]|[Figure 375]|
|[Figure 376]|[Figure 377]|[Figure 378]|
|[Figure 379]|[Figure 380]|[Figure 381]|

|[Figure 382]|[Figure 383]|[Figure 384]|
|---|---|---|
|[Figure 385]|[Figure 386]|[Figure 387]|
|[Figure 388]|[Figure 389]|[Figure 390]|
|[Figure 391]|[Figure 392]|[Figure 393]|

(b)step=50

(a)step=25

(d)step=100

(c)step=75

(e)step=125

(f)step=150

(g)step=175

(h)step=200

- Figure 9. Effect of the Inference Steps t. Overall quality does not improve consistently as t increases, but first improves and then declines. This may be because the early steps are dominated by low frequency, whereas the later steps are dominated by high frequency.

|[Figure 394]|
|---|

FFN

Attention

face

(e) high† & low freq

FFN

face

Cross

(f) high†† & low freq

FFN

Attention

(g) only high-freq†

FFN

Attention

face

Cross

(c) high & low freq

FFN

Attention

face

Cross

(d) high & low† freq

FFN

Attention

(b) only low-freq

|[Figure 395]|
|---|

FFN

Attention

face

Cross

(a) only high-freq

face

|[Figure 396]|
|---|

face point

|[Figure 397]|
|---|

face point

|[Figure 398]|
|---|

face point

|[Figure 399]|
|---|

face point

|[Figure 400]|
|---|

Cross

Attention

face

Cross

- Figure 10. Visualization of Different Methods of Injecting Control Signals into DiT. Only (c), which injects high-frequency information between the Attention and FFN modules, while incorporating low-frequency signals, including facial key points, into the shallow layers of the network, resulting in optimal performance.

90%

80%

AlignmentPercentage

70%

60%

50%

40%

30%

20%

10%

0%

FaceSim-Arc FaceSim-Cur CLIPScore FID

Figure 11. Cross Validation between Automatic Metrics and Human Perception. Existing metrics show limited alignment with human, particularly CLIPScore and FID.

alisticVisionV60B1) to generate videos for comparison.

DreamVideo [72]. DreamVideo is an open-source model supporting tuning-based IPT2V. It utilizes a UNet-based architecture and is designed to generate 4-second (32-frame) videos at a resolution of 256 × 256. For fairness, we use only one reference image and the official default settings (e.g., steps, learning rate) to train the model, and then generate videos for comparison.

of 4 or 8 seconds in length, with resolutions of 480p, 720p, or 1280p, and aspect ratios of 16:9, 9:16, or 1:1. We used its official default settings to generate 4-second, 480p, 16:9 videos for best comparison.

MotionBooth [73]. MotionBooth is an open-source model supporting tuning-based IPT2V. It utilizes a UNetbased architecture and is designed to generate 576×320×24 and 512×320×16 videos, respectively. For fairness, we use only one reference image and the official default settings (e.g., 512×320×16, steps) to train the model, and then

ID-Animator [19]. ID-Animator is the sole open-source model currently supporting tuning-free IPT2V. It utilizes a UNet-based architecture and is designed to generate 2second (16-frame) videos at a resolution of 512×512. We utilized its official default configuration of DreamBooth (re-

generate videos for comparison. Due to the requirement to specify the direction of camera movement, we fix the camera to move to the left.

Magic-Me [46]. Magic-Me is an open-source model supporting tuning-based IPT2V. It utilizes a UNet-based architecture and is designed to generate 8-second (16-frame) videos at a resolution of 1024×1024. For fairness, we use only one reference image and the official default settings (e.g., steps, learning rate) to train the model, and then generate videos for comparison.

CogVideoX [83], EasyAnimate [76], OpenSora-Plan [37], DynamiCrafter [74]. These models are opensource foundational generation models supporting imageto-video generation. While CogVideoX, EasyAnimate, and OpenSora-Plan are based on DiT architecture, DynamiCrafter employs a UNet-based architecture. Due to the lack of support for IPT2V in all these models, the process begins by generating the initial frame using PhotomakerV2 [36]. Subsequently, the respective models are used to generate the subsequent frames: CogVideoX-5B-I2V, EasyAnimate v4, OpenSora-Plan v1.3, and DynamiCrafter512. Due to the differences in supported resolution and length, we use the official default settings to ensure optimal performance.

##### 3.5. Details of Implementation

(1) The section of Quantitative Analysis requires each model to generate 13,500 videos (30×5×90). To minimize computational overhead, we select only 2 reference images per ID, each with 90 text prompts in the section Effect of the Identity Signal Injection in DiT (30×2×90); 60 IDs each with 1 text prompt in the section Comparison with Tuningbased Methods and Ablation on the Number of Inference Steps (60×1×1); and select only 1 reference image per ID, each with 45 text prompts for the remaining experiments (30×1×45), including the Comparison with I2V Methods, Fine-grained Ablation Study, etc. (2) For all the baselines used in this paper, including Vidu [6], ID-Animator [19], DreamVideo [72], MotionBooth [73] and Magic-Me [46], we use the default settings from their official websites (e.g., resolution, fps, inference steps, training steps, etc.) to ensure optimal results. For MotionBooth [73], since it requires the direction of camera movement to be specified, we set the camera to move to the left, which is also the official setting. (3) The evaluation dataset consists of 30 individuals, including celebrities, ordinary people, and those of diverse skin tones, as demonstrated by the qualitative results presented in this paper. This diversity enhances the comprehensiveness and reliability of the experimental data. Furthermore, the text prompts cover a wide range of expressions, actions, and backgrounds, providing a thorough assessment of the generalizability of IPT2V.

||[Figure 401]|
|---|
<br><br>|[Figure 402]|
|---|
<br><br>Figure 12. Visualization of the Questionnaire for User Study.<br><br>3.6. Details of Human Evaluation<br><br>To illustrate the user study intuitively, we provide a visualisation of the questionnaire in Figure 12. In addition, to|
|---|

##### 3

T a increase the reliability and diversity of the questionnaire, we implement the following rules:

- • The presentation order of different videos is randomized to reduce cognitive bias among respondents.
- • A sliding verification upon submission is required to confirm that all responses are submitted manually and not by automated bots.
- • Each IP address is restricted to a single submission, and users are required to log in before voting to ensure each individual could submit only once.
- • Questionnaires where 90% of responses selected the same option (all A or all B) are discarded.
- • The primary voting population consisted of undergraduate, master’s, and PhD students from universities, along with a portion of the general public outside the field.
- • The validity of data is assessed based on the time spent completing the questionnaire; responses with completion times of less than two minutes are excluded, as it typically takes 2–5 minutes to complete.
- • Validity is also evaluated based on response distribution. Since the A/B options are randomly assigned for each question, we discarded questionnaires where 90% of responses favored a single option (either all A or all B).

#### 4. Additional Statement 4.1. Support for Key Findings

These findings are not merely our observations but are synthesized from and validated by existing diffusion and ViT literature, with additional support provided by our experiments in Main Sec. 4.5. Finding 1) [[53] (Sec. 3.2), [61] (Sec. 1)] highlight that diffusion models’ noise prediction is fundamentally low-level and benefits from a U-Net bias. In Main Sec. 4.5, Main Table 3 (f–g) shows training instability when low-frequency features are omitted, reinforcing their crucial role. Finding 2) [[88] (Sec. 1)] shows the im-

portance of high-frequency features in the diffusion model. [[5] (Abs), [3] (Sec. 1)] note that vision transformers’ challenges with capturing high-frequency detail. Convergent evidence in Main Figure 7 (f) indicates transformers’ highfrequency handling warrants deeper investigation. In Main Sec. 4.5, Main Figure 7 confirms improvements when decoupling high- and low-frequency signals, despite a legend error we have corrected.

##### 4.2. Justification of the Fine-grained Feature

For how to ensure that the feature output by Local Facial Extractor remains fine-grained: Q-Former serve as a fusion mechanism rather than an extractor. Among the input into it, Face Extractor, as a face recognition backbone, plays a central role by inherently extracting fine-grained features invariant to non-identity attributes (e.g., expression, posture). CLIP provides secondary semantic features for editing, while Dropout [2, 26] are employed to it to maintain the Q-Former’s fine-grained feature dominance. Main Table 3 shows both modules are distinct and complementary.

##### 4.3. The feature of Diffusion Transformer

Sora [8], based on the DiT architecture, exhibits significant potential in simulating the physical world. Recently, foundational models for visual generation have shifted from UNet [16, 74] to DiT [50, 76, 83, 96, 97], owing to the latter’s scalability and superior performance. Accordingly, our ConsisID is based on DiT instead of UNet architecture.

##### 4.4. Generalization to Image Generation

Despite being trained exclusively on video data, ConsisID can leverage the generalization capabilities of CogVideoX [83] to generate high-quality, identity-preserving images. This is achieved by either setting the frame parameter to 1 or extracting the first frame of a video as an image.

##### 4.5. Ethics Statement

ConsisID is capable of generating high-quality, realistic human videos. However, it also presents potential negative impacts, as the technique may be utilized to produce deceptive content for fraudulent activities. It is important to recognize that any technology is susceptible to misuse [78–81].

##### 4.6. Reproducibility Statement

First, we have explained the implementation of ConsisID in detail in section 3. Second, we have explained the details of training and inference in section 4. Finally, the data and codes used in this work will be open-source online.

##### 4.7. Copyright Statement

The training data is sourced from in-house datasets, and only a subset of the data (CC BY 4.0 license) will be made publicly available. The video content exclusively features

humans, and any NSFW content is detected and removed based on the video captions. The videos come from different regions of the world to ensure they are representative.

##### 4.8. Limitations and Future Work

Existing metrics fail to accurately assess the capabilities of different ID preservation models. While ConsisID can generate realistic and natural videos based on a text prompt, commonly used metrics such as CLIPScore [21] and FID [22] exhibit minimal differences compared to previous methods. A promising approach is to develop a metric that better aligns with human perception.

|[Figure 403]<br><br>[Figure 404]<br><br>[Figure 405]<br><br>[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]<br><br>[Figure 410]<br><br>[Figure 411]<br><br>ConsisIDVidu<br><br>clutchingabouquetofvibrantflowersslightlyupturnedlips...mangentlyconvey...His<br><br>asenseofcalmjoy,accompaniedbyafainttwinkleinhiseye.Thesceneissetinalush<br><br>,brimmingwithcolorfulbloomsverdantfoliagegardenand...<br><br>[Figure 412]<br><br>[Figure 413]<br><br>[Figure 414]<br><br>[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]<br><br>[Figure 418]<br><br>MotionBoothDreamVideoConsisIDViduMagic-Me<br><br>[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]<br><br>[Figure 422]<br><br>[Figure 423]<br><br>[Figure 424]<br><br>[Figure 425]<br><br>[Figure 426]<br><br>[Figure 427]<br><br>[Figure 428]<br><br>ofacar.heiswearing...Thebackgroundinthedriver'sseat...amansittingglasses<br><br>,suggestingthatthecarisparkednearaparkoragarden.outsidethecarshowsgreenery<br><br>,possiblyobservingsomethingoutside...lookingoutthewindowThemanseemstobe<br><br>MotionBoothDreamVideoMagic-Me<br><br>[Figure 429]<br><br>[Figure 430]<br><br>[Figure 431]<br><br>[Figure 432]<br><br>[Figure 433]<br><br>[Figure 434]<br><br>[Figure 435]<br><br>[Figure 436]<br><br>[Figure 437]<br><br>[Figure 438]<br><br>[Figure 439]<br><br>[Figure 440]<br><br>[Figure 441]<br><br>[Figure 442]<br><br>[Figure 443]<br><br>[Figure 444]<br><br>[Figure 445]<br><br>[Figure 446]<br><br>[Figure 447]<br><br>[Figure 448]<br><br>[Figure 449]<br><br>[Figure 450]<br><br>[Figure 451]<br><br>[Figure 452]<br><br>13. Quantitative comparison with Closed-source and Tuning-based Identity-Preserving Videos Generation methods. Conachieves advantages in identity preservation, visual quality, motion amplitude, and text relevance. Moreover, only ConsisID, Vidu<br><br>MotionBooth [73] can generate aesthetically pleasing horizontal videos, while the others [46, 72] can only produce square videos.|
|---|

###### ConsisIDVidu

ightlyupturnedlipsconvey

e.Thesceneissetinalush

dantfoliage...

oth

Figure 1 sisID ac [6] and M

|[Figure 453]<br><br>[Figure 454]<br><br>[Figure 455]<br><br>[Figure 456]<br><br>[Figure 457]<br><br>[Figure 458]<br><br>[Figure 459]<br><br>[Figure 460]<br><br>... a young boy sitting at a table, eating a piece of food. He appears to be enjoying his meal, as he takes a bite and chews it. The boy is wearing a blue shirt and has short hair ... background is dark, with some light coming from the left side of the frame. There is a straw visible on the right side of the frame ...<br><br>... a young man who appears to be a content creator or streamer. he is wearing a green sleeveless top and red headphones. The background is illuminated with vibrant neon lights ... The man is seated in front of a microphone ... The setting appears to be a well-lit room with a curtain and a lamp visible ...<br><br>.. a man sitting in a red armchair, enjoying a cup of coffee or tea. he is dressed in a light-colored outfit and has long dark-haired hair. The setting appears to be indoors, with large windows providing a view of a misty or foggy coastal landscape outside ...<br><br>... a news reporter who is walking down a city street at night while holding a microphone and speaking to the camera. The reporter is wearing a white coat and a blue tie ... background shows a brightly lit cityscape with tall buildings and streetlights ... speech is accompanied by various text overlays ...<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>[Figure 461]<br><br>... a woman with blonde hair, wearing a blue tank top and holding a pink tank ... in a clothing store ... as there are racks of clothes visible in the background. ... speaking to the camera ... She has colorful bracelets on her wrist and is wearing a necklace with multiple beads ...<br><br>... a woman with blonde hair standing on a beach near the water's edge. She is wearing a black swimsuit ... The sky above is clear with some clouds, and the ocean waves gently lap against the shore ... seems to be holding something white in her hand ...<br><br>A woman wearing a colorful scarf and cozy sweater, her eyes sparkling with a hint of wonder as she looks around at the falling leaves. Her lips curl into a slight, content smile ... Golden and orange leaves cascade softly around her, with the trees forming a vibrant canopy overhead ...<br><br>A woman with an anticipatory smile, her eyes twinkling with excitement as she holds a camera ... Her face is animated with enthusiasm; her lips slightly parted as she concentrates on framing the shot. The scene is set in a picturesque landscape, perhaps a mountain range in the background ...<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>[Figure 462]<br><br>[Figure 463]<br><br>[Figure 464]<br><br>[Figure 465]<br><br>[Figure 466]<br><br>[Figure 467]<br><br>[Figure 468]<br><br>[Figure 469]<br><br>[Figure 470]<br><br>[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]<br><br>[Figure 474]<br><br>[Figure 475]<br><br>[Figure 476]<br><br>[Figure 477]<br><br>[Figure 478]<br><br>[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]<br><br>[Figure 482]<br><br>[Figure 483]<br><br>[Figure 484]<br><br>[Figure 485]<br><br>[Figure 486]<br><br>[Figure 487]<br><br>[Figure 488]<br><br>[Figure 489]<br><br>[Figure 490]<br><br>[Figure 491]<br><br>[Figure 492]<br><br>[Figure 493]<br><br>[Figure 494]<br><br>[Figure 495]<br><br>[Figure 496]<br><br>[Figure 497]<br><br>[Figure 498]<br><br>[Figure 499]<br><br>[Figure 500]|
|---|

###### Figure 14. Showcases of Identity-Preserving Videos Generated by ConsisID. Our method consistently generates realistic videos that match the input identity while enabling precise control through text prompts, demonstrating significant practical utility.

|[Figure 501]<br><br>[Figure 502]<br><br>[Figure 503]<br><br>[Figure 504]<br><br>[Figure 505]<br><br>... a woman ... leaning against a tall, white column ... Her arms are crossed, and she appears to be posing for the camera. The background is lush with greenery, including trees and bushes, suggesting that the location is a park or garden ...<br><br>Input Iamge<br><br>[Figure 506]<br><br>[Figure 507]<br><br>[Figure 508]<br><br>[Figure 509]<br><br>[Figure 510]<br><br>... a young man ... in a snowy park ... wearing a colorful winter jacket with a floral pattern ... shows a snowy landscape with trees, benches, and a metal fence ... he smiles and gives a thumbs-up gesture towards the camera ...<br><br>Input Iamge<br><br>[Figure 511]<br><br>[Figure 512]<br><br>[Figure 513]<br><br>[Figure 514]<br><br>[Figure 515]<br><br>[Figure 516]<br><br>... a woman standing in front of a large screen ... She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting ...<br><br>Input Iamge<br><br>[Figure 517]<br><br>[Figure 518]<br><br>[Figure 519]<br><br>[Figure 520]<br><br>[Figure 521]<br><br>[Figure 522]<br><br>... a man walking down a city street at night, engrossed in his smartphone. He is dressed in a formal suit and tie ... street is illuminated with various neon lights and signs ... background shows tall buildings with lit-up windows ...<br><br>Input Iamge<br><br>[Figure 523]<br><br>[Figure 524]<br><br>[Figure 525]<br><br>[Figure 526]<br><br>[Figure 527]<br><br>[Figure 528]<br><br>[Figure 529]<br><br>[Figure 530]<br><br>[Figure 531]<br><br>[Figure 532]<br><br>[Figure 533]<br><br>[Figure 534]<br><br>[Figure 535]<br><br>[Figure 536]<br><br>[Figure 537]<br><br>[Figure 538]<br><br>[Figure 539]<br><br>[Figure 540]<br><br>[Figure 541]<br><br>[Figure 542]<br><br>[Figure 543]<br><br>[Figure 544]<br><br>[Figure 545]<br><br>[Figure 546]<br><br>[Figure 547]<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>... a woman dressed as a mermaid, swimming underwater. She is wearing a silver tail and a matching top, which is adorned with colorful patterns. The woman has long, wavy hair that flows freely underwater. The water around her is clear and blue, with sunlight filtering through the surface ...<br><br>... a man jogging along a grassy path next to a body of water, likely a lake or river. he is wearing a white sports cloth and appears to be focused on his run. The background shows a serene outdoor setting with green trees and a clear sky ...<br><br>... a young man engaged in an intense gaming session. he is seated in a gaming chair ... The chair has a distinctive design, featuring a logo or emblem on the backrest. The man is wearing a gray sweatshirt and glasses, and he is equipped with a headset that includes a microphone ...<br><br>... a young man walking through a park during sunset ... he is wearing a sleeveless top with a geometric pattern and denim shorts ... long, dark hair that falls over his shoulders … holds a skateboard ... orange and green in color. The park is lush with greenery, and the sunlight filters through the trees ...|
|---|

[Figure 548]

|[Figure 549]<br><br>[Figure 550]<br><br>[Figure 551]<br><br>[Figure 552]<br><br>[Figure 553]<br><br>[Figure 554]<br><br>[Figure 555]<br><br>[Figure 556]<br><br>[Figure 557]<br><br>[Figure 558]<br><br>[Figure 559]<br><br>[Figure 560]<br><br>[Figure 561]<br><br>[Figure 562]<br><br>[Figure 563]<br><br>[Figure 564]<br><br>[Figure 565]<br><br>[Figure 566]<br><br>... a woman in exquisite hybrid armor adorned with iridescent gemstones, standing amidst gently falling cherry blossoms. Her piercing yet serene gaze hints at quiet determination ... a tranquil courtyard framed by moss-covered stone walls and wooden arches ... The petals swirl around her ...<br><br>Input Iamge<br><br>... a baby wearing a bright superhero cape, standing confidently with arms raised in a powerful pose ... with eyes wide and lips pursed in concentration, as if ready to take on a challenge. The setting appears playful, with colorful toys scattered around and a soft rug underfoot ...<br><br>Input Iamge<br><br>... a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation ... the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint ...<br><br>Input Iamge<br><br>... a man standing at an easel, focused intently as his brush dances across the canvas. He wears a paint-splattered apron ... The setting, filled with scattered art supplies, open paint tubes, and unfinished sketches pinned to the wall ... A large window on one side allows sunlight to stream in ...<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>Input Iamge<br><br>... a little girl with pigtails, a backpack, and a bright smile, skipping joyfully down a bustling city street. She holds a bunch of colorful balloons that sway as she moves, her eyes wide with excitement. The backdrop of vibrant buildings and street vendors ...<br><br>... a girl sitting on a stool, playing a guitar with her fingers moving skillfully across the strings, creating a soulful melody. Her eyes are closed, and there's a slight smile on her face, conveying a sense of deep connection to the music ...<br><br>... a man with a rugged beard, wearing a leather jacket, riding a vintage motorcycle along a desert highway ... eyes narrowed slightly against the wind, as the setting sun casts a warm glow over the landscape ... arid land with occasional cacti and rocky outcrops ...<br><br>... a man celebrating his birthday, holding a piece of cake while he prepares to blow out the candles. He is smiling warmly, and as he closes his eyes to make a wish ... Around him, soft, ambient lighting from nearby candles casts a warm glow, and there are a few balloons and decorations in the background ...<br><br>[Figure 567]<br><br>[Figure 568]<br><br>[Figure 569]<br><br>[Figure 570]<br><br>[Figure 571]<br><br>[Figure 572]<br><br>[Figure 573]<br><br>[Figure 574]<br><br>[Figure 575]<br><br>[Figure 576]<br><br>[Figure 577]<br><br>[Figure 578]<br><br>[Figure 579]<br><br>[Figure 580]<br><br>[Figure 581]<br><br>[Figure 582]<br><br>[Figure 583]<br><br>[Figure 584]<br><br>[Figure 585]<br><br>[Figure 586]<br><br>[Figure 587]<br><br>[Figure 588]<br><br>[Figure 589]<br><br>[Figure 590]<br><br>[Figure 591]<br><br>[Figure 592]<br><br>[Figure 593]<br><br>[Figure 594]<br><br>[Figure 595]<br><br>[Figure 596]|
|---|

