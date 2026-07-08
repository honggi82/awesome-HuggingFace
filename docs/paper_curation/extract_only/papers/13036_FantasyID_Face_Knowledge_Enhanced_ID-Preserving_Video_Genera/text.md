## FantasyID: Face Knowledge Enhanced ID-Preserving Video Generation

Yunpeng Zhang1,2∗ , Qiang Wang1∗ , Fan Jiang1† , Yaqi Fan2 , Mu Xu1† , Yonggang Qi2‡ 1AMAP, Alibaba Group 2Beijing University of Posts and Telecommunications {yijing.wq, frank.jf, xumu.xm}@alibaba-inc.com, {bryan2233, yqfan, qiyg}@bupt.edu.cn

# arXiv:2502.13995v1[cs.GR]19Feb2025

"A woman in an elegant gown at a ball ... crystal chandeliers casting glow over the marble floors ... holds a champagne glass ... her movements graceful ..."

Reference Image

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|
|---|---|---|---|

[Figure 5]

"A man in a cozy room, strumming guitar ... the warm lighting casting shadows ... books on shelves ... his fingers moving across strings ... foot tapping to rhythm ..."

Reference Image

[Figure 6]

|[Figure 7]|[Figure 8]|[Figure 9]|[Figure 10]|
|---|---|---|---|

"A woman rides a bicycle through city streets ... hair flowing in breeze ... surrounded by colorful storefronts ... expression shows enjoyment ..."

Reference Image

[Figure 11]

|[Figure 12]|[Figure 13]|[Figure 14]|[Figure 15]|
|---|---|---|---|

"A man at a night market ... bright lights overhead ... vendors calling out ... displaying street food and crafts ... atmosphere full of energy ..."

Reference Image

|[Figure 16]|[Figure 17]|[Figure 18]|[Figure 19]|
|---|---|---|---|

[Figure 20]

Figure 1: Examples of FantasyID. Given a human face image, FantasyID generates ID-preserving videos with enhanced motion dynamics and more stable facial structures.

### Abstract

preserving text-to-video generation (IPT2V) have gained popularity recently due to their efficacy and scalability. However, significant challenges remain to achieve satisfied facial dynamics while keeping the identity unchanged. In this work, we present a novel tuning-free IPT2V framework by enhancing face knowledge of the pre-trained video model built

Tuning-free approaches adapting large-scale pretrained video diffusion models for identity-

∗Equal Contribution †Project Leaders ‡Corresponding authors

on diffusion transformers (DiT), dubbed FantasyID. Essentially, 3D facial geometry prior is incorporated to ensure plausible facial structures during video synthesis. To prevent the model from learning “copy-paste” shortcuts that simply replicate reference face across frames, a multi-view face augmentation strategy is devised to capture diverse 2D facial appearance features, hence increasing the dynamics over the facial expressions and head poses. Additionally, after blending the 2D and 3D features as guidance, instead of naively employing adapter to inject guidance cues into DiT layers, a learnable layer-aware adaptive mechanism is employed to selectively inject the fused features into each individual DiT layers, facilitating balanced modeling of identity preservation and motion dynamics. Experimental results validate our model’s superiority over the current tuning-free IPT2V methods. Our project page: https://fantasy-amap.github. io/fantasy-id/.

### 1 Introduction

Identity-preserving text-to-video generation (IPT2V) aims to generate videos from a reference image while consistently maintaining the identity across frames [44, 34, 40, 42]. Solving this problem provides valuable insights into developing compelling applications, such as personalized avatars, immersive try-ons, interactive storytelling, and more. Upon the powerful generative capability of large-scale pre-trained video diffusion models, recent works on IPT2V, such as IDAnimator [37] and ConsisID [47], shifted to model adaptation, thus avoiding case-by-case tuning during inference.

Despite notable advancements, current identity-preserving video diffusion models face three critical challenges rooted in their architectural and training paradigms. First, current methods exhibit limited knowledge of facial structures, making them vulnerable when confronted with intricate facial movements. Besides, tuning pre-trained T2V models with a single-view reference face may encounter the ”copy-paste” issue [42, 37], that excessive reliance on a monocular static image could restrict the desired diversity of facial expressions in the video.

Moreover, the intrinsic hierarchical nature of DiT causes layer-specific sensitivity to control signals [46, 47], calling for dedicated conditioning strategies to harmonize identity preservation and temporal coherence throughout generation.

To tackle the first challenge, we propose integrating 3D facial geometry priors into our model to ensure stable and consistent structures of the human face during generation.

Specifically, DECA [14] is employed to extract the essential identity-specific 3D feature, i.e., a shape point cloud, which is found effective for identity preservation. The identity-irrelevant features, e.g., pose and expression, are discarded. Moreover, this introduced 3D shape prior is conveniently manageable by varying the 3D points’ locations so the generated human face could change accordingly.

To mitigate the issue of static motion, we devise a multiview face adaptation strategy to avoid learning shortcuts that

directly replicate the static face across frames in the generated video. Namely, we augment a monocular reference face image with its variants from different viewpoints, forming a face pool for the same identity obtained from the training human video. We then randomly select any of them as input for pre-trained video model adaptation.

It turns out that this can enforce the adapted IPT2V model to capture detailed diverse 2D facial appearance features, thereby improving the dynamics performance of the generated video.

Given the obtained 3D and 2D facial features from the reference face image, we further blend them together using a transformer-based feature fusion module to guide the pretrained video diffusion model to produce identity-specific human video. However, we figure that it is inefficient to inject the fused feature into the pre-trained DiT layers naively by adapter as it is known that the DiT lower layers tend to capture the overall structure and upper layers for details. Therefore, we introduce a layer-aware injection mechanism to allow the model to adaptively select the most beneficial cues from the fused features.

To this end, our contributions are three-fold. (i) To our best knowledge, this is the first attempt that 3D facial priors extracted from a single-view reference image are employed to enhance the facial structure stability, thus benefiting ID preservation throughout video generation.

- (ii) By employing a multi-view facial augmentation strat-

egy, we can significantly enhance the perception of 2D facial appearance across a wide range of viewpoints, thus benefiting the motion dynamics associated with facial expression and head pose.

- (iii) A learnable layer-aware feature guidance mechanism

is devised to facilitate precise control for a balanced IDpreserving and dynamics modeling, offering high-fidelity human video with better temporal coherence and identity consistency.

### 2 Related Work

Personalized Diffusion Models. Recent advancements in identity-preserving image generation have seen rapid development, with several approaches employing Textual Inversion [18], DreamBooth [28], and LoRA [15] for fine-tuning models on specific IDs, achieving impressive results. However, these methods lack flexibility and the capability for real-time inference. In contrast, training-free methods effectively eliminate the dependency on parameter fine-tuning. For instance, the IP-Adapter [16] leverages CLIP features [16] to guide pre-trained models, ensuring identity preservation. PuLID [35] adopts EvaClip [29] to maintain identity consistency, while InstantID [43] integrates ArcFace [7] with pose networks to achieve flexible ID retention. Within the realm of identity-preserving video generation, maintaining smooth character motion alongside accurate preservation of identity features represents a key challenge. The ID-Animator [37], built upon the AnimateDiff [24] base model, has successfully preserved identity characteristics but exhibits noticeable limitations in the fluidity of character movements. The emerging DiT architecture [45, 48] shows promise for enhancing

[Figure 21]

- Figure 2: Overview of FantasyID. The framework constructs a multi-view face collection, randomly selects one face as the reference input, and employs face abstractor to extract 2D visual tokens while using DECA to extract 3D face vertex tokens. We fuse both the 2D and 3D tokens with fusion transformer layers and guide DiT-based model via a layer-aware signal injection method.

consistency in video output. ConsisID [47], which builds on CogVideoX[45], employs a frequency-aware control scheme to enhance identity consistency without the need for identityspecific tuning. However, existing methods still face challenges such as low motion amplitude and facial instability during movement, which can result in limited dynamic movements and unnatural changes or distortions in facial features across frames.

Portrait Animation. Portrait animation techniques have made significant strides in animating static images by mapping motions and expressions from a guiding video while preserving the portrait’s identity and background. [19, 6, 12, 26, 17] Research primarily focuses on 3D morphable models [22], such as DECA [13] and FLAME [4], which excel in detailed 3D face modeling but largely concentrate on facial features without extending to full-body animations or scene elements. In rendering, volumetric methods provide high detail but are computationally intense, while CVTHead [39] offers a more efficient, yet still facially focused, approach. Animation methods like EchoMimic [33], which relies on Mediapipe [9], and SadTalker [31], which uses audio inputs to generate 3D motion coefficients, also emphasize facial regions. Despite their advancements, these methods generally lack the ability to generate or control complete scenes through text-based inputs, highlighting a gap in creating broader narrative or environmental elements through such interactions.

### 3 Methodology

Given a reference face image, FantasyID is designed to generate a video that faithfully preserves the individual’s identity characteristics. An overview of FantasyID is illustrated in Figure 2. For each training video, we construct a multiview face collection and randomly select a reference image as the input condition. Then, we utilize face abstractor to extract 2D clip tokens (Sec. 3.2), employ DECA to disentangle

features unrelated to the core ID (such as expressions, pose) and to extract 3D structural information (Sec. 3.3), and use fusion transformer to fuse the 2D tokens and 3D tokens into face descriptor embeddings (Sec. 3.4). Additionally, we exert control over the DiT-based model by employing a layer-aware signal injection method, ensuring precise modulation at each layer (Sec. 3.5). The following section (Sec. 3.1) elaborates on the diffusion model and the preliminaries of our method.

#### 3.1 Preliminary

Latent Diffusion Models. Latent diffusion models are efficient diffusion models that operate in the latent space rather than the pixel space. We use an encoder ε from a pre-trained variational autoencoder to compress video data x into a latent code z = ε(x). The encoder ε is a video compression module based on 3D variational autoencoders [30]. It incorporates three-dimensional convolutions to compress videos both spatially and temporally. During the diffusion stage, Gaussian noise ϵ is added to z to create zt = √αtz + √1 − αtϵ, with ϵ ∼ N(0,I), over T stages. Here, αt serves as the noise scheduler, while t represents the timestep. The denoising process employs the conditional probability pθ(zt−1|zt) = N(µθ(zt),Σθ(zt)) to predict the previous state zt−1. Here, µθ implemented using a denoising model ϵθ, while Σθ represents the learned covariance of the reverse diffusion process. The training objective typically involves a reconstruction loss that aims to minimize the discrepancy between the added noise and the network’s predicted noise:

L = Et,z∼p(z),ϵ∼N(0,I) ∥ϵθ(zt,t) − ϵ∥22 (1)

Diffusion Transformer(DiT). The Diffusion Transformer [27] is a transformer-based architecture designed for efficient denoising in latent diffusion models. In contrast to to UNet [2] for processing spatiotemporal latent representations,DiTbased models demonstrate superior capabilities in modeling

long-range dependencies and ensuring cross-temporal coherence. This has led to significant advancements in motion coherence and overall video quality [45, 38, 48]. For our denoising model ϵθ, we opted for the MM-DiT from CogVideoX [45].

#### 3.2 Multi-View Collection and Face Abstractor

Multi-View Face Collection. Obtaining effective ID references during the training stage is crucial. To ensure the model focuses on critical areas, we first crop the facial region from each frame of the video, eliminating the background distractions. Following MovieGen [42], a single reference image can lead the model to learn shortcuts that involve directly replicating the face, so we have constructed a collection of face images I from different viewpoints in the training stage. We utilize RetinaFace [11] to extract geometric relationships among facial landmarks to calculate the head pose angles and select six images with the most significant viewpoint differences to form a multi-view face set. By providing the model with a diverse range of perspectives, it gains a more comprehensive understanding of the subject, thereby enhancing its ability to maintain identity consistency across various poses and expressions.

Face Abstractor. After acquiring the face image from the multi-view face collection, we aim to effectively extract facial features from it. Previous works [47, 36] on identitypreserving generation has employed Q-Former [25] to transform the face clip embeddings, but this approach can disrupt the spatial structure among features [32]. Recognizing the critical importance of local correlations for face characteristics, we introduce the use of C-Abstractor [32] to transform face clip embeddings to Xf ∈ Rh×w×C. This module, composed of two-dimensional convolutions and average pooling, leverages spatial locality to enhance feature extraction. By effectively capturing comprehensive facial information while preserving the key spatial relationships essential for accurate video generation.

#### 3.3 3D Constraints

Face Vertex. A stable facial geometric constraint is critical to ensuring high-quality generation. We utilize the priors obtained from 3D reconstruction to both constrain and enhance the model’s understanding of the reference image, thereby improving the quality and fidelity of the generated output. We employ the DECA framework [14] to capture the threedimensional geometric structure of faces, which provides vertex coordinates Vs ∈ R(N×3) for the reference image, where N is the number of vertices in FLAME model [4]. This approach distinguishes between intrinsic facial features and extrinsic factors such as pose, expression, and lighting. This separation mechanism enhances the model’s comprehension of identity-specific features while effectively suppressing interference from non-identity characteristics.

3D Vertex Representation. We employ SpiralNet++ [8] to extract 3D features from Vs, represented as XV′ ∈ RN

′×C′, where N′ denotes the number of vertices after downsampling and C′ indicates the channel dimension of the feature descriptor. To encode positional information from the 3D point cloud, we incorporate a positional encoding Edep, derived

from the depth map of the projected vertices. This results in the enhanced vertex features is XV = XV′ + Edep.

#### 3.4 Fusion Transformer

To effectively integrate 3D point cloud features with complementary 2D descriptors, we design the fusion transformer. Specifically, We utilize MLP to transform the 2D features Xf ∈ Rh×w×C into XF ∈ Rh×w×C

′

, aligning the feature dimension from C to C′. The aligned 2D features are then concatenated with the 3D vertex features, formulated as X = [XV ,XF], where the combined feature X ∈ R(N

′+h×w)×C′). Here, h × w denotes the dimensions of the 2D feature space. The fusion transformer consists of L layers. Finally, we utilize a series of residual 1D convolutions, referred to as ResConv1D, to align the hidden dimensions of DiT, thereby obtaining the id descriptor VF ∈ R(N

′×C). Through this fusion, we effectively integrates highdimensional data into a integrated features, capturing rich facial representations while ensuring the preservation of 3D structural priors in the generated facial representations.

#### 3.5 Layer-Aware Control Signal Injection

Inspired by the observation that each layer in the DiT architecture contributes uniquely to the overall performance [46], we adopt a similar approach for controlling facial video generation using DiT. Specifically, we recognize that different layers exhibit varying sensitivities to control signals. To address this, we propose a layer-aware control signal injection mechanism that dynamically adjusts the integration of control signals based on the role of each layer.

Particularly, for each MM-DiT block, we employ a lightweight model Fl to learn the optimal feature representation. This lightweight network comprises a convolution block followed by normalization. Independent weights for each layer enhance fidelity and diversity, aligning control signals precisely with the needs of each layer. This ensures stability and expressive potential in outputs. The process is defined by the formula:

Zˆl = Zl + Fl(Attention(l,Ql,Klid,Vlid)) (2)

where Ql,Klid,Vlid are the query, key, and values matrices of the attention operation, Ql = ZlWlq, Klid = VFWlk, Vlid = VFWlv. Here, Zl represents the hidden states, and l is the layer index of the MM-DiT block, and Wlq,Wlk,Wlv are trainable weights.

### 4 Experiments

#### 4.1 Setups

Implementation Details. In our experiments, we utilize a diverse dataset comprising full body and portrait data from ConsisID-Data [47], CelebV-HQ[21], and Open-Vid[41]. Subsequently, following the approach in SVD [23], we employed PaddleOCR [5, 20] to eliminate any videos containing subtitles. Furthermore, we used InsightFace [7, 10] to exclude videos with a face confidence score below 0.9, resulting in a final selection of approximately 50,000 clips. We optimize with a batch size of 16 and a learning rate of 3 × 10−6,

[Figure 22]

[Figure 23]

Case2: "A man speaking into a microphone...He is wearing a white shirt and has a beard... with his right hand. The background is dark... he is engaged in delivering a speech or presentation."

###### Case1: "A young man with bleached blonde hair... wearing a black leather jacket over a white t-shirt... Dramatic lighting in blue and pink hues... creates a moody atmosphere. He strikes a pose, suggesting flair..."

FantasyIDID-AnimatorInputInputID-AnimatorConsisIDConsisIDFantasyID

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

[Figure 45]

Case4: "A man stands in a room... wearing a gray cap, tshirt, and glasses. He holds an open book with colorful, abstract art and multilingual text... Smiling at the camera, presenting it. The modern, minimalist room features a green plant... The atmosphere is creative."

Case3: "A woman in a kitchen... wearing an apron and denim shirt... stands by a stove with a pot and vegetables... The modern kitchen features white countertops... talking to the camera... Her confident demeanor reflects her cooking style."

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

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

[Figure 64]

[Figure 65]

- Figure 3: Qualitiative Comparision between our methods and ConsisID, ID-Animator. Please refer our supplementary materials for the video results.

completing 90,000 training steps, which takes approximately 36 hours using 16 A100 GPUs. Our methodology incorporates classification-free guidance with a random null text ratio of 0.1, utilizing AdamW as the optimizer, and employs a cosine with restarts as the learning rate scheduler. During the inference stage, we utilize DECA’s coarse FLAME [4] parameters to construct a 3D point cloud from the input image. The denoising step is set to 50. The fusion transformer is designed to 6 layers, and the ResConv1D comprises 4 residual 1D convolutional blocks. The downsampling factor of the face abstractor is set to 4, and the number of 3D tokens N′ is

314.

Evaluation Metrics. We employ ArcFace [7] embedding similarity to assess two key aspects. First, Reference Similarity(RS) calculates the similarity between the reference image and frames to evaluate identity preservation. Second, InterFrame Similarity(IFS) measures the similarity between consecutive video frames to evaluate the stability of identity features during motion. Additionally, we analyze the Fr´echet Inception Distance (FID) [3] of the face region to assess video quality and utilize Face Motion (FM), measured by average dense optical flow [1], to evaluate the degree of motion. We

used 50 richly detailed portrait reference images. To more accurately measure the identity preservation capability, we cropped the facial regions from each video for quantitative evaluation.

- 4.2 Qualitative Analysis

For the qualitative evaluation, we present comparison results with diffusion-based identity preservation models, ConsisID and ID-Animator. Other models, including VideoMaker [44] and MagicMirror [34], are not open-source and therefore not included in our direct comparisons.

Figure 3 demonstrates that ID-Animator struggles with generating human body parts beyond the face and exhibits noticeable “copy-paste” artifacts. Moreover, the generated content often appears overly static, lacking natural motion. These limitations significantly restrict its practical application in scenarios requiring dynamic and realistic human behavior or interactions. Regarding ConsisID, while the overall visual quality is high, there are still issues with structural instability during facial movements, as seen in Case 1. Although ConsisID retains features such as skin texture from the reference image, it fails to accurately reproduce the overall facial structure in Case 3 and 4. In contrast, our method achieves the best results in terms of visual quality, preservation of the subject’s identity from the reference image, and maintaining consistent facial structure across frames during motion.

To further validate the effectiveness of our proposed method, we conducted a comprehensive user study involving 32 participants. Each participant was tasked with assessing four critical aspects: Overall Quality(OQ), Face Similarity(FSim), Facial Structure(F-Str), and Facial Dynamics(FD), rating each aspect on a scale from 0 to 10. As shown in Table 1, the scores indicate that FantasyID consistently outperforms baseline methods across all evaluated dimensions, demonstrating its superior perceived quality in human assessments.

OQ F-Sim F-Str FD

ID-Animator 4.38 6.20 5.82 3.28 ConsisID 7.85 7.79 6.44 7.12 Ours 8.39 8.68 8.10 7.94

- Table 1: User Study results. Higher scores indicate better performance.

4.3 Quantitative Analysis

- Table 2 presents a comprehensive quantitative evaluation of various face video generation methods. ID-Animator achieves an impressive FID score and a higher IFS score. However, this performance can be attributed to its tendency to generate more static content, thereby ensuring high quality and excellent identity consistency. This focus on static representations likely limits its ability to produce diverse and dynamic facial motions. In contrast, while our method provides a slightly highest FID score, it excels in capturing dynamic expressions, as evidenced by the leading face motion score of 0.61, and achieves the highest RS score of 0.57, reflecting superior identity preservation. Notably, our model outperforms

[Figure 66]

[Figure 67]

[Figure 68]

Face Vertex

###### Face details

|[Figure 69]|[Figure 70]| |
|---|---|---|
|[Figure 71]<br><br>Refer|[Figure 72]<br><br>Refer| |
|[Figure 73]|[Figure 74]| |

|[Figure 75]| |
|---|---|
| | |

[Figure 76]

[Figure 77]

[Figure 78]

| |
|---|

- Vertex 1
- Vertex 2

Differ

- 1. Facial Widths
- 2.Jawline Sharpness

[Figure 79]

[Figure 80]

[Figure 81]

|[Figure 82]| |
|---|---|
| | |

[Figure 83]

[Figure 84]

[Figure 85]

Figure 4: Effect of 3D Constraints. By altering the facial widths and jawline sharpness of face vertex, the generated facial videos exhibit noticeable structural changes.

ConsisID across all metrics, reflecting a superior ability in dynamism and identity preservation.

###### FID ↓ RS↑ IFS↑ FM↑

ID-Animator 138.27 0.35 0.98 0.18 ConsisID 149.70 0.47 0.93 0.54 Ours 142.50 0.57 0.95 0.61

##### Table 2: Quantitative evaluation of different methods. The best results are highlighted in bold.

- 4.4 Ablation Studies

To comprehensively evaluate the contribution of each module within the FantasyID framework, we conducted a series of ablation studies. These experiments systematically removed individual components to assess their impact on the overall performance of the model in Table 3. Specifically, we examined the effects of excluding the Multi-View Face Collection(MFC), Face Abstractor(FA), Face Vertex(FV), and Layer-Aware Control Signal Injection (LACSI). Additionally, we modify the face vertex data of different inputs to validate the effectiveness of 3D constraints.

FID ↓ RS ↑ IFS ↑ FM ↑

w/o FA 145.82 0.55 0.93 0.49 w/o MFC 130.77 0.54 0.98 0.33 w/o FV 172.51 0.42 0.90 0.36 w/o LACSI 235.46 0.33 0.93 0.22 FantasyID 142.50 0.57 0.95 0.61

Table 3: Quantitative results of removing individual modules from FantasyID framework.

- 4.5 Qualitative Analysis

Effect of 3D Constraints. To verify the efficacy of our 3D constraint control mechanism, we modify the 3D face vertex to generate videos with different facial widths and jawline sharpness. The qualitative results presented in Figure 4 showcase the significant variations in facial structure, thereby

Video generated by Reference 1 Video generated by Reference 2

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

###### w/o3DFVwith3DFV

[Figure 94]

[Figure 95]

[Figure 96]

| |
|---|

|[Figure 97]|
|---|

| |
|---|

Enlarge Details during Motion

Structural integrity Smooth facial dynamics

[Figure 98]

Reference 1 Reference 2

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

| |
|---|

| |
|---|

|[Figure 103]|
|---|

Structural instability Distorted facial dynamics

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

| |
|---|

| |
|---|

- Figure 5: Ablation study on Face Vertex(FV). Without the face vertex leading to distorted facial structures during motion.

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

w/oFAw/oMFCw/oLACSIFantasyID

[Figure 122]

[Figure 123]

[Figure 124]

"A woman swims energetically in the ocean, her strokes powerful and fluid as she navigates the waves. She wears a swimsuit... The water is a deep blue... Her expression is one of joy and focus... The horizon stretches endlessly...

- Figure 6: Ablation studies on Multi-View Face Collction(MFC), Face Abstractor(FA) and LayerAware Control Signal Injection(LACSI).

These results indicate that the critical role of 3D vertex integration in preserving structural integrity and ensuring smooth facial dynamics.

w/o Multi-View Face Collection. We replace the multi-view face collection with a single face image during the training stage. As shown in Figure 6, this approach significantly reduces the range of captured facial motions, limiting the model’s ability to understand and represent different angles. However, this approach achieves the best FID and IFS scores

- as shown in Table 3. This performance can be attributed to the model’s tendency to take a shortcut by prioritizing higher similarity to the reference image, thereby maintaining consistency at the expense of dynamic range. w/o Face Abstractor. We replaced the face abstractor with Q-Former, which, as shown in Figure 6, leads to some facial distortions. These distortions are likely due to Q-Former’s tendency to disrupt the spatial characteristics of face CLIP features. Additionally, the results presented in Table 2 indicate that this approach achieves lower FID, RS, and IFS scores. This suggests that Face Abstractor is more effective
- at capturing comprehensive and spatially coherent facial information. w/o Layer-Aware Control Signal Injection. By removing

the layer-aware control signal injection module Fl, we observed a significant decrease in face similarity, as shown in Figure 6, along with a decline in all metric scores, as detailed in Table 3. These results indicate a decline in both video quality and identity preserving. In contrast, the layeraware control method adapts more effectively to the unique feature distributions between different DiT blocks by learning the most suitable feature control signals for each layer. This approach ensures optimal performance and fidelity in generating ID features.

confirming that our 3D constraints effectively guide the generation of facial features. This demonstrates the flexibility and precision of our approach in controlling facial characteristics.

w/o Face Vertex. We evaluated the importance of 3D constraints by excluding the face vertex of our framework. The qualitative results in Figure 5 demonstrates that the absence of

### 5 Conclusion

- 3D face vertex data causes the model to rely solely on 2D feature extraction, leading to distortions in facial structure during motion. The quantitative results in Table 2 show a decline across all metrics, which suggest more erratic facial motion.

FantasyID presents a groundbreaking approach for identitypreserving human video generation, overcoming the limitations of traditional methods. By employing a multi-view face

collection, face abstractor, 3D constraints, and layer-aware control signal injection, it significantly enhances video quality, identity preserving, and temporal coherence. This scalable, training-free solution maintains high-fidelity representations during complex motions. Future work will focus on optimizing multi-identity retention and expanding FantasyID’s role in dynamic video production and personalized content creation.

### References

- [1] Gunnar Farneb¨ack. “Two-frame motion estimation based on polynomial expansion”. In: Image Analysis: 13th Scandinavian Conference, SCIA 2003 Halmstad, Sweden, June 29–July 2, 2003 Proceedings 13. Springer. 2003, pp. 363–370.
- [2] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. “U-net: Convolutional networks for biomedical image segmentation”. In: Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer. 2015, pp. 234–241.
- [3] Martin Heusel et al. “Gans trained by a two time-scale update rule converge to a local nash equilibrium”. In: Advances in neural information processing systems 30

(2017).

- [4] Tianye Li et al. “Learning a model of facial shape and expression from 4D scans.” In: ACM Trans. Graph. 36.6 (2017), pp. 194–1.
- [5] Xinyu Zhou et al. “East: an efficient and accurate scene text detector”. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2017, pp. 5551–5560.
- [6] Olivia Wiles, A Koepke, and Andrew Zisserman. “X2face: A network for controlling face generation using images, audio, and pose codes”. In: Proceedings of the European conference on computer vision (ECCV). 2018, pp. 670–686.
- [7] Jiankang Deng et al. “Arcface: Additive angular margin loss for deep face recognition”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019, pp. 4690–4699.
- [8] Shunwang Gong et al. “Spiralnet++: A fast and highly efficient mesh convolution operator”. In: Proceedings of the IEEE/CVF international conference on computer vision workshops. 2019, pp. 0–0.
- [9] Camillo Lugaresi et al. “Mediapipe: A framework for building perception pipelines”. In: arXiv preprint arXiv:1906.08172 (2019).
- [10] Jiankang Deng et al. “RetinaFace: Single-Shot MultiLevel Face Localisation in the Wild”. In: CVPR. 2020.
- [11] Jiankang Deng et al. “Retinaface: Single-shot multilevel face localisation in the wild”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020, pp. 5203–5212.
- [12] Guangming Yao et al. “Mesh guided one-shot face reenactment using graph convolutional networks”. In: Proceedings of the 28th ACM international conference on multimedia. 2020, pp. 1773–1781.
- [13] Yao Feng et al. “Learning an Animatable Detailed 3D Face Model from In-The-Wild Images”. In: vol. 40. 8.

2021. URL: https://doi.org/10.1145/3450626.3459936.

- [14] Yao Feng et al. “Learning an animatable detailed 3D face model from in-the-wild images”. In: ACM Transactions on Graphics (ToG) 40.4 (2021), pp. 1–13.

- [15] Edward J Hu et al. “Lora: Low-rank adaptation of large language models”. In: arXiv preprint arXiv:2106.09685 (2021).
- [16] Alec Radford et al. “Learning transferable visual models from natural language supervision”. In: International conference on machine learning. PMLR. 2021, pp. 8748–8763.
- [17] Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. “One-shot free-view neural talking-head synthesis for video conferencing”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021, pp. 10039–10049.
- [18] Rinon Gal et al. “An image is worth one word: Personalizing text-to-image generation using textual inversion”. In: arXiv preprint arXiv:2208.01618 (2022).
- [19] Taras Khakhulin et al. “Realistic one-shot mesh-based head avatars”. In: European Conference on Computer Vision. Springer. 2022, pp. 345–362.
- [20] Minghui Liao et al. “Real-time scene text detection with differentiable binarization and adaptive scale fusion”. In: IEEE transactions on pattern analysis and machine intelligence 45.1 (2022), pp. 919–931.
- [21] Hao Zhu et al. “CelebV-HQ: A Large-Scale Video Facial Attributes Dataset”. In: ECCV. 2022.
- [22] Volker Blanz and Thomas Vetter. “A morphable model for the synthesis of 3D faces”. In: Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 2023, pp. 157–164.
- [23] Andreas Blattmann et al. “Stable video diffusion: Scaling latent video diffusion models to large datasets”. In: arXiv preprint arXiv:2311.15127 (2023).
- [24] Yuwei Guo et al. “Animatediff: Animate your personalized text-to-image diffusion models without specific tuning”. In: arXiv preprint arXiv:2307.04725 (2023).
- [25] Junnan Li et al. “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models”. In: International conference on machine learning. PMLR. 2023, pp. 19730–19742.
- [26] Youxin Pang et al. “Dpe: Disentanglement of pose and expression for general video portrait editing”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, pp. 427–436.
- [27] William Peebles and Saining Xie. “Scalable diffusion models with transformers”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023, pp. 4195–4205.
- [28] Nataniel Ruiz et al. “Dreambooth: Fine tuning text-toimage diffusion models for subject-driven generation”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023, pp. 22500– 22510.
- [29] Quan Sun et al. “Eva-clip: Improved training techniques for clip at scale”. In: arXiv preprint arXiv:2303.15389 (2023).
- [30] Lijun Yu et al. “Language Model Beats Diffusion– Tokenizer is Key to Visual Generation”. In: arXiv preprint arXiv:2310.05737 (2023).

- [31] Wenxuan Zhang et al. “Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pat-

- tern Recognition. 2023, pp. 8652–8661.

[32] Junbum Cha et al. “Honeybee: Locality-enhanced projector for multimodal llm”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pat-

- tern Recognition. 2024, pp. 13817–13827.

- [33] Zhiyuan Chen et al. “Echomimic: Lifelike audiodriven portrait animations through editable landmark conditions”. In: arXiv preprint arXiv:2407.08136

(2024).

- [34] Armand Comas-Massagu´e et al. “MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space”. In: arXiv preprint arXiv:2404.01296 (2024).
- [35] Zinan Guo et al. “Pulid: Pure and lightning id customization via contrastive alignment”. In: arXiv preprint arXiv:2404.16022 (2024).
- [36] Junjie He, Yifeng Geng, and Liefeng Bo. “UniPortrait: A Unified Framework for Identity-Preserving Singleand Multi-Human Image Personalization”. In: arXiv preprint arXiv:2408.05939 (2024).
- [37] Xuanhua He et al. “Id-animator: Zero-shot identitypreserving human video generation”. In: arXiv preprint arXiv:2404.15275 (2024).
- [38] Weijie Kong et al. “HunyuanVideo: A Systematic Framework For Large Video Generative Models”. In: arXiv preprint arXiv:2412.03603 (2024).
- [39] Haoyu Ma et al. “CVTHead: One-shot Controllable Head Avatar with Vertex-feature Transformer”. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2024, pp. 6131– 6141.
- [40] Ze Ma et al. “Magic-me: Identity-specific video customized diffusion”. In: arXiv preprint arXiv:2402.09368 (2024).
- [41] Kepan Nan et al. “OpenVid-1M: A Large-Scale HighQuality Dataset for Text-to-video Generation”. In: arXiv preprint arXiv:2407.02371 (2024).
- [42] Adam Polyak et al. “Movie gen: A cast of media foundation models”. In: arXiv preprint arXiv:2410.13720

(2024).

- [43] Qixun Wang et al. “Instantid: Zero-shot identitypreserving generation in seconds”. In: arXiv preprint arXiv:2401.07519 (2024).
- [44] Tao Wu et al. “VideoMaker: Zero-shot Customized Video Generation with the Inherent Force of Video Diffusion Models”. In: arXiv preprint arXiv:2412.19645 (2024).
- [45] Zhuoyi Yang et al. “Cogvideox: Text-to-video diffusion models with an expert transformer”. In: arXiv preprint arXiv:2408.06072 (2024).
- [46] Sihyun Yu et al. “Representation alignment for generation: Training diffusion transformers is easier than you think”. In: arXiv preprint arXiv:2410.06940 (2024).

- [47] Shenghai Yuan et al. “Identity-Preserving Text-toVideo Generation by Frequency Decomposition”. In: arXiv preprint arXiv:2411.17440 (2024).
- [48] Zangwei Zheng et al. Open-Sora: Democratizing Efficient Video Production for All. Mar. 2024. URL: https: //github.com/hpcaitech/Open-Sora.

