# MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling

Yifang Men, Yuan Yao, Miaomiao Cui, Liefeng Bo

Tongyi Lab, Alibaba Group https://menyifang.github.io/projects/MIMO/index.html

arXiv:2409.16160v2[cs.CV]11Jun2025

[Figure 1]

Figure 1. Given a single reference image of character, MIMO can synthesize animated avatars in driving 3D poses (visualized as skeleton sequences) retrieved from motion datasets (left) or extracted from in-the-wild videos (right). Real-world scenes from driving videos can also be integrated into the synthesis with natural human-object interactions. MIMO simultaneously achieves advanced scalability to arbitrary characters, generality to novel 3D motions, and applicability to interactive real-world scenes in a unified framework.

## Abstract

Character video synthesis aims to produce realistic videos of animatable characters within lifelike scenes. As a fundamental problem in the computer vision and graphics community, 3D works typically require multi-view captures for per-case training, which severely limits their applicability of modeling arbitrary characters in a short time. Recent 2D methods break this limitation via pre-trained diffusion models, but they struggle for flexible controls, pose generality and scene interaction. To this end, we propose MIMO, a novel framework which can not only synthesize realistic character videos with controllable attributes (i.e., character, motion and scene) provided by simple user in-

puts, but also simultaneously achieve advanced scalability to arbitrary characters, generality to novel 3D motions, and applicability to interactive real-world scenes in a unified framework. The core idea is to encode the 2D video to compact spatial codes, considering the inherent 3D nature of video occurrence. Concretely, we lift the 2D frame pixels into 3D using monocular depth estimators, and decompose the video clip into three spatial components (i.e., main human, underlying scene, and floating occlusion) in hierarchical layers based on the 3D depth. These components are further encoded to canonical identity code, structured motion code and full scene code, which are utilized as control signals of the synthesis process. The design of spatial decomposed modeling enables flexible user control, com-

plex motion expression, as well as 3D-aware synthesis for scene interactions. Experimental results show that the proposed method outperforms prior works by a large margin in character animation synthesis and is effective in providing a high degree of controllability (i.e., arbitrary characters, novel 3D motions, interactive scenes), thus enabling brandnew editing tasks (e.g., video character replacement).

## 1. Introduction

Character video synthesis, an essential topic in areas of Computer Vision and Computer Graphics, has huge potential applications for movie production, virtual reality, and animation. While recent video generative models [2, 8, 10, 18, 42, 49] have achieved great progress with text or image guidance, none of them fully captures the underlying attributes (e.g., appearance and motion of instance and scene) in a video and provides flexible user controls. Meanwhile, they still struggle for reasonable character synthesis in challenging scenarios, such as extreme 3D motions and complex object interactions accompanied by occlusions.

The aim of this paper is to propose a brand-new and boosting method for controllable video synthesis, which can not only synthesize character videos with controllable attributes (i.e., character, motion and scene) provided by very simple user inputs, but also achieve advanced scalability to arbitrary characters, generality to novel 3D motions, and applicability to interactive real-world scenes in a unified framework (see Figure 1). In other words, the proposed method is capable of mimicking anyone anywhere with complex motions and object interactions, thus named MIMO. As more concretely illustrated in Figure 2, users are allowed to feed multiple inputs (e.g., a single image for character, a pose sequence for motion, and a single video even an image for scene) to provide desired attributes respectively or a direct driving video as input. The proposed model can embed target attributes into the latent space to construct target codes or encode the driving video with spatial-aware decomposition as spatial codes, thus enabling intuitive attribute control of the synthesis by freely integrating latent codes in a specific order.

Our task setting significantly decreases the cost of video creation and enables wide applications for not only character animation, but also video attribute editing (e.g., character replacement, motion transfer and scene insertion). However, it is extremely challenging due to the simplicity of user inputs, the complexity of real-world scenarios and the absence of 2D video annotations. With the great progress of 3D neural representations (e.g., NeRF [30] and 3D Gaussian splatting [19]), a series of works [13, 22, 25, 31, 41] tend to represent the dynamic human as a pose-conditioned NeRF or Gaussian to learn animatable avatars in highfidelity rendering quality. However, they typically require

[Figure 2]

Figure 2. The basic idea of MIMO. Controllable character video synthesis with desired attributes provided by multiple inputs (e.g., a single image for character, a pose sequence for motion, and a single video even an image for scene) or a driving video. Target attributes are embedded into the latent space as the target codes and the driving video is spatially decomposed as the spatial codes. Target character videos can be generated in user control with the combined attribute codes.

fitting a neural field to multi-view captures or a monocular video of dynamic performers, which severely limits their applicability due to inefficient training and expensive data acquisition. Another 3D works explored faster and cheaper solutions by directly inferring 3D models from single human images, following by rigged animation and physical rendering [14, 15, 23, 29]. Unfortunately, the realism of the renderings is marginally compromised due to cumulative errors in sequential processes. Recently, several efforts [12, 39, 44, 53] have investigated the potential of 2D diffusion models on image-guided character video synthesis, named character animation. They show that highfidelity character video can be synthesized by inserting image feature extracted from the reference-net [12, 53] or control-net [44, 47] into a pretrained diffusion model. However, they only focus on character synthesis in simple 2D motions (e.g., frontal dancing) and are less effective for articulated human motion in 3D space due to limited pose generality. Moreover, they fail to produce lifelike video for complicated scenes accompanied by human-object interactions and large camera movements. We argue that the cause for these difficulties stems from insufficient video attribute parser considered only in 2D feature space, thereby disre-

garding the inherent 3D nature of video occurrence.

To tackle these challenges, we propose a novel framework for controllable character video synthesis via spatial decomposed modeling. The core idea is to decompose and

- encode the 2D video in 3D-aware manner and employ more adequate expressions (e.g., 3D representations) for articulated properties. In contrast to previous works [12, 50] directly learn the whole 2D feature of each video frame, we lift the 2D frame pixels into 3D, and construct the decomposed spatial representations in 3D space, which are equipped with richer contextual information and can be used for control signals of the synthesis process. Specifically, we decompose the video clip to three spatial components (scene, human and occlusion) in hierarchical layers based on 3D depth. In particular, human represents the main object in the video, scene represents the underlying background, and occlusion traces floating foreground objects. For the human component, we further disentangle the identity property via canonical appearance transfer and
- encode the 3D motion representation via structural body codes. The scene and occlusion components are embedded with a shared VAE encoder and re-organized as a full scene code. The decomposed latent codes are inserted as conditions of a diffusion-based decoder to reconstruction the video clip. In this way, the network learns not only controllable synthesis of various attributes, but also 3D-aware layer composition of main object, foreground and background. Thereby, it enables flexible user controls as well as challenging cases of complicated 3D motions and natural object interactions. In summary, our contributions are threefold:

- • We propose a brand-new task that synthesizes character videos with controllable attributes by directly providing simple user inputs, and solve it with a novel approach to simultaneously achieve advanced scalability to arbitrary characters, generality to novel 3D motions, and applicability to interactive scenes in a unified framework.
- • We introduce the spatial decomposed modeling, an effective architecture to simulate intricate video observations by encoding the inherent spatial components. It enables not only flexible user control, but also 3D-aware synthesis in human-object interaction contexts.
- • We tackle the challenge of inadequate pose representation for articulated human by introducing structured motion codes. It provides better expressive ability to handle complicated motions in spatial space, thus enabling advanced generality of generative model to novel 3D motions.

## 2. Related Work

3D Human Modeling. Since the introduction of Neural Radiance Fields (NeRF) [30], neural human representation has achieved remarkable success in obtaining articulated human models by fitting implicit neural fields to multi-view captures [25, 32, 36] or a monocular video [16, 17, 41]. Hu-

manNeRF [41] proposes to represent human from a single video of a moving person by optimizing canonical volume and motion fields. NeuMan [17] jointly learns the decomposition of the human and the scene capable of novel pose rendering and animation of human in the scene. HOSNERF [24] extend to support human-environment interactions by introduce a dynamic human-object model. Recent works [13, 22] further introduce 3D Gaussian splatting [19] for realistic renderings. Despite achieving promising results, these methods typically require expensive data acquisition with precise camera/pose estimates and are less efficient for training and rendering, severely limiting their realworld applications. Another series of work explored faster and cheaper human modeling solutions via generative models [3, 9, 29, 46]. En3D [29] learns an enhanced 3D human generator with efficient refiner to directly infer 3D models from single images or text prompts in few minutes. These generated avatars are rendered in a canonical body pose and aligned to an underlying 3D skeleton which allows for easy animation and the generation of motion videos. However, the fidelity of the driven results is marginally compromised due to cumulative errors inherent in the rendering processes. Diffusion-based Character Video Synthesis. The remarkable progress in diffusion models has demonstrated promising results in image and video generation [2, 7, 8, 10, 35]. A plethora of methodologies proposed to incorporate these pre-trained diffusion models for human-centric video synthesis [4, 12, 27, 38, 44, 50, 53], enabling the transformation of character images into animated videos controlled by desired pose sequences. MagicAnimate [44] utilizes ControlNet [47] and an appearance encoder for identity preservation and pose guidance, building upon a video diffusion model. Animate Anyone [12] employs a UNet-based ReferenceNet to extract detailed features from reference images. Champ [53] further introduces a 3D parametric model to extract motion guidance as conditions. MimicMotion [50] presents a confidence-aware pose guidance approach to enhance generation quality and temporal smoothness. Despite producing visually appealing results, these methods encounter quality degradation issues in complex motion scenarios and are incapable of handling occlusion-aware generation in human-object interaction contexts. Our method built on diffusion models overcomes these challenges by a novel generative architecture with spatial decomposed modeling, considering inherent 3D nature for 2D videos.

## 3. Method Description

Our goal is to synthesize high-quality character videos with user-controlled visual attributes, such as character, motion and scenes. The desired attributes can be automatically extracted from an in-the-wild character video or simply provided by a single image, a pose sequence, and a single video, respectively. Different from previous methods us-

[Figure 3]

Figure 3. An overview of the proposed framework. The video clip is decomposed to three spatial components (i.e., main human, underlying scene, and floating occlusion) in hierarchical layers based on 3D depth. The human component is further disentangled for properties of identity and motion via canonical appearance transfer and structured body codes, and encoded to identity code Cid and motion code Cmo. The scene and occlusion components are embedded with a shared VAE encoder and re-organized as a full scene code Cso. These latent codes are inserted into a diffusion-based decoder as conditions for video reconstruction.

ing only weak control signals (e.g., text prompt) [27, 51] or insufficient 2D expressions [12, 50], our model achieves automatic and unsupervised separation of spatial components and encodes them into compact latent codes considering inherent 3D nature to control the synthesis. Thus, our dataset can only contain 2D character videos {v ∈ RN×H×W} without any annotations.

The overview of the proposed framework is illustrated in Figure 3. Given a video clip v, MIMO learns a reconstruction process with automatic attribute encoding and composed condition decoding. Considering 3D nature of video occurrence, we extract the three spatial components in hierarchical layers based on 3D depth (Section 3.1). The first component of human is encoded with disentangled properties of identity and motion (Section 3.2). The last two components of scene and occlusion are embedded with a shared encoder and re-organized as a scene code (Section 3.3). These latent codes C are inserted into a diffusion decoder D as composed conditions (Section 3.4). C, D are jointly learned by minimizing the difference between the generated frames and input frames via noise prediction (Section 3.5).

### 3.1. Hierarchically spatial layer decomposition

Considering the inherent 3D elements of video composition, we split a video v = {It|t = 1,...,N} into three

main components: human as a core performer, scene as the underlying background, and occluded object as the floating foreground. To automatically decompose them, we lift 2D pixels into 3D and track detected objects in hierarchical layers based on corresponding depth values.

To start with, for each frame It ∈ v, we obtain its monocular depth map using a pretrained monocular depth estimator [45]. The human layer is firstly extracted with human detection [43], and propagate to video volume via video tracking method [33], thus obtaining Mh ∈ RN∗H∗W, a binary mask sequence along the time axis (i.e., masklet). Subsequently, we extract the occlusion layer with objects whose mean depth values are smaller than the human layer, and generate masklet predictions Mo via a video tracker. The scene layer can be obtained by removing human and occlusion objects, defined by scene masklet Ms. With predicted masklets, we can compute the decomposed human video of component i by multiplying the original source video with component masklet Mi:

vi = v ⊙ Mi,i = {h,o,s}, (1)

where ⊙ denotes element-wise product. vi is then fed into the corresponding branch for human, scene and occlusion encoding, respectively.

### 3.2. Disentangled human encoding

This branch aims to encode the human component vh into the latent space as disentangled codes Cid and Cmo of identity and motion. Previous works [12, 44, 50] typically random select one frame from the video clip as appearance representation, and employ extracted 2D skeleton with keypoints as the pose representation. Essentially, this design exists two core issues which may limit networks’ performance: 1) It is hard for 2D pose to adequately express motions which take place in 3D spatial space, especially for articulated ones accompanied by exaggerated deformations and frequent self-occlusions. 2) The postures of frames across a video are highly similar, and there inevitably exists the entanglement between appearance frame and target frame both retrieved from the posed video. Thereby, we introduce new 3D representations of motion and identity for adequate expression and full disentanglement.

Structured motion code. We define a set of latent codes Z = {z1,z2,...,z6890}, and anchor them to corresponding vertices of a deformable human body model (SMPL)

- [26]. For the frame t, SMPL parameters St and camera parameters Ct are estimated from the monocular video frame vth using [5]. The spatial locations of the latent codes are then transformed based on the human pose St and projected to the 2D plane based on the camera setting Ct. Using a differentiable rasterizer [21] with vertex interpolation, the 2D feature map Ft in continuous values can be obtained. {Ft,t = 1,..,N} will be stacked along the time axis and embedded into the latent space as the motion code Cmo by a pose encoder Ep. In this way, we establish correspondences between the same set of identifiable latent codes on underlying 3D body surface and posed 2D renderings at different frames of arbitrary videos. This structured motion code enables clearer and more dense pose representation for articulated 3D motions in spatial space. Canonical appearance transfer. To fully disentangle the appearance from posed video frames, an ideal solution is to learn the dynamic human representation from the monocular video and transform it from the posed space to the canonical space. Considering the efficiency, we employ a simplified method that directly transforms the posed human image to the canonical result in standard A-pose using a pretrained human repose model. The synthesized canonical appearance image is fed to ID encoders to obtain the iden-

tity code Cid. This simple design enables full disentanglement of identity and motion attributes. Following [12], the ID encoders include a CLIP image encoder and a referencenet architecture to embed for the global and local feature, respectively, which composes Cid.

### 3.3. Scene and occlusion encoding

In scene and occlusion branches, we use a shared and fixed VAE encoder [20] to embed the vs and vo into the latent

[Figure 4]

Figure 4. The architecture of the diffusion-based decoder.

space as the scene code Cs and occlusion code Co, respectively. Before vs input, we pre-recover it by a video inpainting method [52] as R(vs) to avoid the interference brought by mask contours. Then the scene code Cs and the occlusion code Co are concatenated together in order to get the full scene code Cso for composed synthesis. The independent encoding of spatial components (i.e., middle human, underlying scene, and floating occlusion) enable the network to learn an automatic layer composition, thus achieving natural character insertion in complicated scenes even with occluded object interactions.

### 3.4. Composed decoding

Given the latent codes of decomposed attributes, we recompose them as conditions of the diffusion-based decoder for video reconstruction. As shown in Figure 4, we adapt denoising U-Net backbone built upon Stable Diffusion (SD) [34] with temporal layers from [6]. The full scene code Cso is concatenated with the latent noise, and is fed into a 3D convolution layer for fusion and alignment. The motion code Cmo is added to the fused feature and input to the denoising U-Net. For identity code Cid, its local feature and global feature are inserted into the U-Net via self-attention layers and cross-attention layers as [12], respectively. Finally, the denoised result is converted into the video clip vˆ via a pretrained VAE decoder [20].

### 3.5. Training

For the training, we employ the diffusion noise-prediction loss to simulate video reconstruction process:

0,cid,cso,cmo,t,ϵ∈N(0,1)[||ϵ−ϵθ(xt,cid,cso,cmo,t)||22]

L = Ex

(2) where x0 is the augmented input sample, t denotes the diffusion timestep, xt is the noised sample at t, and ϵθ represents the function of the denoising UNet.

Implementation details. Our method is implemented in PyTorch using 8 NVIDIA Tesla-A100 GPUs with 80GB memory. We initialize the model of denoising U-Net and reference-net based on the pre-trained weights from SD 1.5 [34], whereas the motion module is initialized with the weights of AnimateDiff [6]. During training, the weights of VAE encoder and decoder, as well as the CLIP image

[Figure 5]

Figure 5. Results of animating diverse characters (e.g., realistic humans, cartoon characters and personified ones) with novel 3D motions retrieved from the motion database (a) or extracted from the driving video (b), and interactive scenes from in-the-wild videos (c).

encoder are frozen. We optimize the denoising U-Net, pose encoder and reference-net with the diffusion noiseprediction loss. It takes around 50k iterations with 24 video frames and a batch size of 8 for converge.

## 4. Experimental Results

Dataset. We create a human video dataset called HUD-7K to train our model. This dataset consists of 5K real character videos and 2K synthetic character animations. The former does not require any annotations and can be automatically decomposed to various spatial attributes via our scheme. To enlarge the range of the real dataset, we also synthesize 2K videos by rendering character animations in complex motions under multiple camera views, utilizing En3D [29]. These synthetic videos are equipped with accurate annotations due to completely controlled production. For the evaluation, we collect 100 in-the-wild human videos covering diverse contents (e.g., dancing, sports and movie) and randomly truncate them to 150-frame clips as test set.

Metrics. We follow [12] to evaluate our method using four standard metrics: Peak Signal-to-Noise Ratio (PSNR) [11], Structural Similarity Index Measure (SSIM) [40], Learned Perceptual Image Patch Similarity (LPIPS) [48] for imagelevel quality, and Fr´echet Video Distance (FVD) [37] for video-level evaluation.

### 4.1. Controllable character video synthesis

Given the target attributes of character, motion and scene, our method can generate realistic video results with their latent codes combined for guided synthesis. The target attributes can be provided by simple user inputs (e.g., single images/videos for character/scene, pose sequences from large database [1, 28] for motion) or flexibly extracted from the real-world videos, involving complicated scenes of occluded object interactions and extreme articulated motions. In the following, MIMO demonstrates that it can simultaneously achieve advanced scalability to arbitrary characters, generality to novel 3D motions, and applicability to in-thewild scenes in a unified framework. More results can be found in the supplemental materials (Supp).

Arbitrary character control. As shown in Figure 5, our method can animate arbitrary characters, including realistic humans, cartoon characters and personified ones. Various body shapes of characters can be faithfully preserved due to the decoupled pose and shape parameters in structured motion representation.

Novel 3D motion control. To verify the generality to novel 3D motions, we test MIMO using challenging outof-distribution pose sequences from the AMASS [28] and Mixamo [1] database, including dancing, playing and

[Figure 6]

Figure 6. Qualitative comparison with three state-of-the-art methods: Animate Anyone [12], Mimic-Motion [50] and Champ [53].

climbing (Figure 5 (a)). We also try complex spatial motions by extracting them from in-the-wild videos for driving (Figure 5 (b, c)). Our method exhibits high robustness for these novel 3D motions under different viewpoints.

Interactive scene control. We validate the applicability of our model to complicated real scenes by extracting both scene and motion attributes from in-the-wild videos for character animation, as a brand-new task of video character replacement. Figure 5 (c) shows that the characters can be seamlessly inserted to the real scenes with natural humanobject interactions.

### 4.2. Comparison with state-of-the-arts

Qualitative comparison. In Figure 6, we compare the synthesis results of our method with three state-of-the-art character animation methods: Animate Anyone [12], MimicMotion [50] and Champ [53]. All the results of these methods are obtained by using the source codes and trained models released by authors or popular re-implements, following by fine-tuning in our training dataset. As we can see, all previous methods fail to produce extreme articulated human motions with exaggerated deformations and frequent self-

occlusions (Figure 6 (a)). They also cannot handle complicated scenes of object interaction (Figure 6 (b)) and large camera movement (Figure 6 (c)). In contrast, our method tackles these challenges and gives more realistic results in both global structures and detailed textures. More video results can be found in Supp. Furthermore, our method shows its superiority that it enables directly inferring animatable avatars in free-viewpoint with inter-frame consistency to some extent, which are comparable to the results of SOTA training-based 3D method, as presented in Supp.

Quantitative comparison. Table 1 shows the comparison of our method with [12, 50, 53] in terms of the PSNR, SSIM, LPIPS and FVD metrics, respectively. Due to the presence of a certain quantity of complex cases (e.g., including spatial motions, scene interactions, and camera movements) in the test set, it’s extremely challenging to model the intricate interplay of real-world scenarios. Even so, our method demonstrated the best performance in these metrics among all methods. It outperforms previous works by a margin of at least 4.16 in terms of PSNR and 0.152 in terms of SSIM, etc. In contrast to directly learning the entire 2D video frame with only inadequate human pose annota-

Table 1. Quantitative comparison with state-of-the-art methods in terms of PSNR, SSIM, LPIPS and FVD.

Method PSNR↑ SSIM↑ LPIPS↓ FVD↓ Animate Anyone [12] 21.003 0.722 0.264 304.3

Mimic-Motion [50] 20.688 0.731 0.343 289.2 Champ [53] 21.044 0.724 0.312 412.5 Ours 25.210 0.883 0.125 221.4

[Figure 7]

Figure 7. Effects of spatial decomposed modeling.

tions, MIMO decompose 2D frames into hierarchically spatial components with more expressive 3D representations. The results indicate that our method better simulates video observations of the real physical world. Considering insufficient scene modeling of previous methods, we also provide additional quantitative comparison by removing background and object for only character synthesis in Supp.

### 4.3. Ablation study

Spatial decomposed modeling. We assess the impact of this design by training a model via randomly selecting one frame from videos as the appearance reference without decomposed layers (w/o SDM). In this way, it fails to produce faithful background and interactive foreground, easily suffering from unstable texture distortions for large camera movements (Figure 7 (a, b)). Essentially, this instability stems from the absent guidance of scene generation, relying only on weak correlation between the scene and character movement revealed by data distribution. We also attempt to model the human and mixed scene without independent occlusion encoding (w/o occ.), and Figure 7 (c, d) shows that this variation cannot synthesize reasonable occluded objects for scene interaction without the ability to comprehend spatial layers. Figure 7 (d) and Table 2 also indicates that this decomposed strategy yield more realistic results with facial details and clothing wrinkles.

Structured motion code. To verify the effectiveness of the proposed structured motion representation (SMR), we evaluate the performance of several variants of our method by employing alternative motion formats: commonly used 2D skeleton in [12] and 3D maps in [53]. As shown in Figure 8 and Table 2, 2D skeleton ignores the occlusion relationship in bones and muscles, resulting in ambiguity for

Table 2. Results of models trained by removing specific modules or replacing with alternative designs for ablation study.

Method PSNR↑ SSIM↑ LPIPS↓ FVD↓ Ours-w/o SDM 22.148 0.762 0.231 268.5

- Ours-w/ 2D skeleton 24.326 0.842 0.186 237.2
- Ours-w/ 3D maps 24.402 0.844 0.203 278.1 Ours-w/o CA 24.918 0.871 0.148 223.1

#### Ours 25.210 0.883 0.125 221.4

[Figure 8]

Figure 8. Effects of structured motion representation and canonical appearance transfer.

spatial motions. The 3D maps, consisting of normal map, depth map, etc., improve the pose representation capability, but still struggle for highly complex spatial motions due to undefined labels of dense body parts. Our SMR, inserting identifiable codes into structured body surfaces and projecting for 2D correspondence, provides strong articulation ability of motion in spatial space and significantly improves the model’s generalizability to novel 3D motions.

Canonical appearance transfer. This design (CA) further disentangles motion and identity in consecutive video frames with high posture correlation. It leads to more effective learning of SMR and obviously alleviates the issue of synthesis confusion between hands and feet (Figure 8 (d)).

## 5. Conclusions

In this paper, we presented MIMO, a novel framework for controllable character video synthesis, which allows for flexible user control with simple attribute inputs. Our method introduces a new generative architecture which decomposes the 2D video to various spatial components, and embeds their latent codes as the condition of decoder to reconstruct the video. Experimental results demonstrated that our method enables not only flexible character, motion and scene control, but also advanced scalability to arbitrary characters, generality to novel 3D motions, and applicability to interactive scenes. We also believed that our solution, which considers inherent 3D nature of video occurrence and automatically encodes the 2D video to hierarchical spatial components could inspire future researches for 3D-aware video modeling. Furthermore, our framework is not only well suited to generate character videos but also can be potentially adapted to common video synthesis tasks.

## References

- [1] Mixamo. https://www.mixamo.com. 6
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2, 3
- [3] Zijian Dong, Xu Chen, Jinlong Yang, Michael J Black, Otmar Hilliges, and Andreas Geiger. Ag3d: Learning to generate 3d avatars from 2d image collections. arXiv preprint arXiv:2305.02312, 2023. 3
- [4] Mengyang Feng, Jinlin Liu, Kai Yu, Yuan Yao, Zheng Hui, Xiefan Guo, Xianhui Lin, Haolan Xue, Chen Shi, Xiaowen Li, et al. Dreamoving: A human dance video generation framework based on diffusion models. arXiv preprint arXiv:2312.05107, 2023. 3
- [5] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4d: Reconstructing and tracking humans with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14783–14794, 2023. 5
- [6] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 5
- [7] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [8] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2, 3
- [9] Fangzhou Hong, Zhaoxi Chen, Yushi Lan, Liang Pan, and Ziwei Liu. Eva3d: Compositional 3d human generation from 2d image collections. arXiv preprint arXiv:2210.04888,

2022. 3

- [10] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2, 3
- [11] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010. 6
- [12] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 2, 3, 4, 5, 6, 7, 8
- [13] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 634–644, 2024. 2, 3
- [14] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies. Tech: Text-guided

- reconstruction of lifelike clothed humans. In 2024 International Conference on 3D Vision (3DV), pages 1531–1542. IEEE, 2024. 2
- [15] Zeng Huang, Yuanlu Xu, Christoph Lassner, Hao Li, and Tony Tung. Arch: Animatable reconstruction of clothed humans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3093–3102,

2020. 2

- [16] Boyi Jiang, Yang Hong, Hujun Bao, and Juyong Zhang. Selfrecon: Self reconstruction your digital avatar from monocular video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5605– 5615, 2022. 3
- [17] Wei Jiang, Kwang Moo Yi, Golnoosh Samei, Oncel Tuzel, and Anurag Ranjan. Neuman: Neural human radiance field from a single video. In European Conference on Computer Vision, pages 402–418. Springer, 2022. 3
- [18] Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman. Dreampose: Fashion image-to-video synthesis via stable diffusion. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 22623–22633. IEEE, 2023. 2
- [19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 2, 3

- [20] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 5
- [21] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Transactions on Graphics, 39(6), 2020. 5
- [22] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19711–19722, 2024. 2, 3
- [23] Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxiang Tang, Yangyi Huang, Justus Thies, and Michael J Black. Tada! text to animatable digital avatars. In 2024 International Conference on 3D Vision (3DV), pages 1508–1519. IEEE, 2024. 2
- [24] Jia-Wei Liu, Yan-Pei Cao, Tianyuan Yang, Zhongcong Xu, Jussi Keppo, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Hosnerf: Dynamic human-object-scene neural radiance fields from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18483–18494, 2023. 3
- [25] Lingjie Liu, Marc Habermann, Viktor Rudnev, Kripasindhu Sarkar, Jiatao Gu, and Christian Theobalt. Neural actor: Neural free-view synthesis of human actors with pose control. ACM transactions on graphics (TOG), 40(6):1–16,

2021. 2, 3

- [26] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, 2015. 5

- [27] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4117–4125, 2024. 3, 4
- [28] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5442–5451, 2019. 6
- [29] Yifang Men, Biwen Lei, Yuan Yao, Miaomiao Cui, Zhouhui Lian, and Xuansong Xie. En3d: An enhanced generative model for sculpting 3d humans from 2d synthetic data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9981–9991, 2024. 2, 3, 6
- [30] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2, 3
- [31] Sida Peng, Junting Dong, Qianqian Wang, Shangzhan Zhang, Qing Shuai, Xiaowei Zhou, and Hujun Bao. Animatable neural radiance fields for modeling dynamic human bodies. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14314–14323, 2021. 2
- [32] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9054–9063, 2021. 3
- [33] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 4
- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 5
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [36] Shih-Yang Su, Frank Yu, Michael Zollh¨ofer, and Helge Rhodin. A-nerf: Articulated neural radiance fields for learning human shape, appearance, and pose. Advances in neural information processing systems, 34:12278–12291, 2021. 3
- [37] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [38] Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Disco: Disentangled control for referring human dance generation in real world. arXiv e-prints, pages arXiv–2307, 2023. 3

- [39] Tan Wang, Linjie Li, Kevin Lin, Yuanhao Zhai, ChungChing Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Disco: Disentangled control for realistic human dance generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9326–9336, 2024. 2
- [40] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [41] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, pages 16210–16220, 2022. 2, 3
- [42] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 2
- [43] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github. com/facebookresearch/detectron2, 2019. 4
- [44] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1481–1490, 2024. 2, 3, 5
- [45] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024. 4
- [46] Zhuoqian Yang, Shikai Li, Wayne Wu, and Bo Dai. 3dhumangan: 3d-aware human image generation with 3d pose mapping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23008–23019, 2023. 3
- [47] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2, 3
- [48] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [49] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2
- [50] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation

- with confidence-aware pose guidance. arXiv preprint arXiv:2406.19680, 2024. 3, 4, 5, 7, 8
- [51] Yuyang Zhao, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Make-a-protagonist: Generic video editing with an ensemble of experts. arXiv preprint arXiv:2305.08850, 2023. 4
- [52] Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. Propainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10477–10486, 2023. 5
- [53] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. arXiv preprint arXiv:2403.14781,

2024. 2, 3, 7, 8

