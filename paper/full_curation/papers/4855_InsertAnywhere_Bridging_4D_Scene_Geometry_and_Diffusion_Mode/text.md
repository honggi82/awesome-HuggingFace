# arXiv:2512.17504v2[cs.CV]29Jun2026

## InsertAnywhere: Geometrically Grounded and Optics-Aware Video Object Insertion

Hoiyeong Jin1⋆ , Hyojin Jang1⋆ , Junha Hyung1⋆ , Jeongho Kim1⋆ , Kinam Kim1 , Dongjin Kim1 , Huijin Choi2 , Hyeonji Kim2 , and Jaegul Choo1

- 1 KAIST AI
- 2 SK Telecom

{hy.jin, wkdgywlsrud, rlawjdghek, sharpeeee, kinamplify, dj_kim, jchoo}@kaist.ac.kr {astehelen, hyeonji}@sk.com

[Figure 1]

[Figure 2]

[Figure 3]

Fig. 1: InsertAnywhere achieves realistic video object insertion by 4D scene understanding and optics-aware video synthesis. From a single user-specified 3D placement, our method seamlessly propagates the reference object to produce geometrically and photometrically consistent insertions across complex camera motions, occlusions, and viewpoints, while naturally synthesizing optical effects such as shadows and reflections.

Abstract. Recent advances in diffusion models have enabled impressive video editing capabilities, yet production-grade Video Object Insertion (VOI) remains challenging due to inadequate 4D scene understanding and a lack of proper optical interactions, such as shadows and reflections. To address these limitations, we present InsertAnywhere, a comprehensive VOI framework that achieves geometrically grounded object placement and optics-aware video synthesis. Our approach first leverages a 4D-aware mask generation module that allows users to anchor an object’s 3D pose in a single frame. The framework automatically propagates this placement across the video, accurately handling local scene dynamics and occlusions. To synthesize realistic physical lighting interactions, we introduce Optics-Aware Representation Alignment, a novel

⋆ Authors contributed equally to this work.

2 H. Jin, H. Jang, J. Hyung, J. Kim et al.

strategy that utilizes an extended mask to guide feature extraction, enabling optical effects to seamlessly extend beyond the inserted object’s boundary. Finally, to overcome the lack of training data for such phenomena, we construct and open-source ROSE++, a specialized quadruplet dataset tailored for the supervised learning of optical effects. Extensive experiments demonstrate that InsertAnywhere produces geometrically plausible and photometrically realistic insertions in complex real-world scenarios, significantly outperforming existing research and commercial generative tools.

### 1 Introduction

Recent advances in diffusion-based generative models have driven significant progress in user-controllable video editing, propelling Video Object Insertion (VOI) into a prominent role for applications like content creation, advertising, and film post-production. The goal of VOI is to seamlessly integrate new objects into existing scenes while maintaining strict spatial, temporal, and photometric consistency. While recent research [2,11,12,21,23] and commercial video generation tools such as Kling [22] and Pika-Pro [18] have made strides in temporally coherent object insertion, existing models still fall short in two critical aspects: 4D-aware object placement and optics-aware video synthesis. These limitations severely restrict their ability to achieve production-grade quality.

Achieving accurate object placement requires both user-guided control and a robust 4D geometric understanding of the scene. Since a single 2D reference image lacks scale and depth context, users must be able to explicitly specify the object’s initial position, size, and pose. However, manually annotating this across all video frames is highly impractical. Therefore, a robust framework must automatically propagate the initial placement throughout the sequence while gracefully handling complex real-world dynamics—such as moving support surfaces and occlusions caused by foreground elements—as demonstrated in Fig. 1 and Fig. 9.

Beyond placement, the generative model must faithfully synthesize the object’s appearance alongside local optical variations induced by the insertion, such as cast shadows and reflections. While mask-free VOI models [3,11,12,15,29] can hallucinate these effects, they often fail to preserve the unedited background regions. Conversely, mask-conditioned models [8,23,28] rigidly confine edits within the provided boundary, struggling to synthesize optical interactions that naturally extend into the surrounding scene. A major bottleneck in solving this is the lack of open-source training data; learning these effects requires a dataset containing a paired tuple of [source video, target video, object mask video, reference image], where the target video explicitly contains the optically integrated object.

To address these challenges, we introduce InsertAnywhere, a comprehensive VOI framework that combines 4D-aware object placement with optics-aware video generation. First, our 4D-aware mask generation module reconstructs the input video into a 4D scene representation. Through an intuitive user interface,

users can anchor the object’s 3D pose and scale in a single reference frame. Our module then automatically propagates this placement temporally, accurately tracking local scene dynamics (e.g., an object moving synchronously with the luggage cart it rests upon) while preserving occlusion boundaries. The 3D object is then reprojected into 2D to extract a temporally coherent mask video, which serves as the spatial condition for the subsequent video generation stage.

For optics-aware video synthesis, we construct and open-source ROSE++, a synthetic dataset specifically tailored to train models on complex optical interactions. We demonstrate that models fine-tuned on ROSE++ generalize highly effectively to real-world VOI tasks. To maximize visual fidelity, our generation pipeline leverages the strong priors of image-based insertion models via firstframe anchoring. Furthermore, we propose Optics-Aware Representation Alignment to enhance photometric consistency. By aligning the intermediate features of an extended mask with those of the primary input mask, it enables maskconditioned video diffusion models to synthesize realistic shadows and reflections that seamlessly extend beyond the immediate boundaries of the inserted object.

Our contributions are summarized as follows:

- – We propose a geometrically grounded mask generation module that propagates object masks across all frames via a 4D scene representation. Based on a single user-specified placement, it produces reliable, temporally coherent masks even under complex camera motions and occlusions.
- – We introduce an optics-aware video generation strategy, incorporating a firstframe anchoring technique and Optics-Aware Representation Alignment, which enables the model to synthesize physical lighting interactions, such as shadows and reflections, that extend beyond the object mask.
- – We construct and open-source ROSE++, a specialized dataset for video object insertion comprising quadruplets of source videos, target videos, mask sequences, and reference images, explicitly tailored for the supervised training of optical effects.
- – We demonstrate that InsertAnywhere achieves geometrically plausible and photometrically realistic object insertions across diverse real-world scenarios, significantly outperforming existing commercial and research-based generative tools.

### 2 Related Work

Mask-Free Video Object Insertion Mask-free models [3,11,12,15,29] typically drive insertion using text prompts and an edited first frame. Crucially, the absence of spatial mask conditioning deprives users of the ability to explicitly control the precise location, scale, and boundaries of the inserted object. While their unbounded generation region theoretically allows them to hallucinate natural optical interactions like shadows, this lack of strict spatial constraints frequently degrades unedited background regions. Furthermore, without explicit geometric guidance, these methods struggle to maintain temporal consistency and fail under complex, dynamic occlusion patterns.

[Figure 4]

[Figure 5]

[Figure 6]

4 H. Jin, H. Jang, J. Hyung, J. Kim et al.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

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

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

- Fig. 2: The InsertAnywhere architecture. Our framework first utilizes 4D scene reconstruction and scene flow tracking to propagate a single user-specified 3D placement into a geometrically grounded mask sequence (a, b). This mask then conditions an opticsaware diffusion model (c), which leverages first-frame anchoring and LoRA fine-tuning to seamlessly synthesize the inserted object and its surrounding optical effects.

Mask-Conditioned Video Object Insertion To strictly preserve the background, mask-conditioned models [4,8,21,23,28] formulate VOI as localized video inpainting governed by an explicit spatial mask. However, this rigid confinement prevents them from synthesizing optical interactions (e.g., cast shadows and reflections) that naturally extend into the surrounding scene. Additionally, these methods typically lack 4D geometric understanding, resulting in brittle and incoherent insertions when the target object interacts with moving support surfaces or becomes occluded.

In contrast to both paradigms, InsertAnywhere achieves production-grade VOI by automatically generating a 4D-aware mask sequence for geometrically grounded, occlusion-robust placement, while simultaneously employing an opticsaware synthesis strategy to render photorealistic lighting interactions that naturally expand beyond the primary object boundary.

### 3 Method

##### 3.1 Overview

Video Object Insertion (VOI) is the task of seamlessly integrating a novel object into an existing video sequence. Formally, given a source video Vsrc = {I1,I2,...,IT}, a reference object image Iref, and a user-defined initial 3D placement, the goal is to generate a photorealistic output video Vˆ where the object is geometrically and photometrically embedded into the scene.

To achieve production-grade VOI, a framework must overcome two critical limitations present in prior works: the lack of robust 4D scene understanding for controllable object placement, and the rigid, tightly-bound nature of standard mask-conditioned generative models that fail to synthesize scene-wide optical interactions.

To simultaneously address both spatial controllability and photometric realism, we propose InsertAnywhere. As depicted in Figure 2, our two-stage framework consists of:

- – Geometrically Grounded Mask Propagation (Section 3.2): Based on a user-specified object placement within a 3D-reconstructed scene, the object is automatically propagated and reprojected using the underlying 4D spatio-temporal dynamics. This yields a reliable, temporally consistent

mask sequence {Mt}Tt=1 that accurately handles complex camera motions and occlusions.

- – Optics-Aware Video Synthesis (Section 3.3): Given the propagated mask sequence, source video, and reference image, our diffusion-based generation pipeline synthesizes the final photorealistic video. By leveraging our ROSE++ dataset and a novel Optics-Aware Representation Alignment strategy, the model seamlessly expands object-induced optical effects (e.g., shadows and reflections) beyond the strict boundaries of the primary mask.

##### 3.2 Geometrically Grounded Mask Propagation

This stage generates a highly controllable, geometrically grounded mask sequence based on the source video and a reference object. The pipeline executes three main steps: 4D scene reconstruction, user-controlled object placement, and scene flow-based temporal propagation. This ensures that the resulting mask sequence accurately aligns with the scene’s geometry, camera motion, and temporal continuity.

- 4D Scene Reconstruction. We first reconstruct a temporally consistent 4D scene representation from the input monocular video by building upon the Uni4D framework [26]. By orchestrating multiple pretrained vision foundation models including depth estimation [17], optical flow [9], camera pose estimation, and segmentation [5, 10, 13], this stage infers the underlying scene geometry and dynamics. Specifically, the 4D representation consists of spatio-temporal point clouds explicitly decoupled into static components for global structure and dynamic components for moving entities. This decoupling allows our framework to handle complex real-world occlusions and perspective changes with high geometric fidelity.

User-Controlled Object Placement. To ensure precise spatial control, the target object is introduced into the 4D scene as a standalone rigid entity. First, the reference object image Iref is lifted into a local 3D point cloud y using TRELLIS [25], an image-to-3D generation model. Through an interactive interface, users align this object point cloud with the reconstructed scene space by

applying a rigid transformation: ys = sobjRobjy + tobj (1)

where the rotation Robj, translation tobj, and global scale factor sobj are intuitively adjusted. This interface provides real-time visualization within the anchor frame s, allowing users to verify the object’s perspective and scale relative to the supporting scene geometry before committing to video generation.

Scene Flow-Based Object Propagation. To maintain physical realism in dynamic scenarios (e.g., an object resting on a moving luggage cart), the object’s trajectory must synchronize with local scene dynamics. Simply fixing the initial

- 3D pose in world coordinates often results in physically implausible drifting relative to moving surfaces. To model these physical interactions, we propagate the object motion using a K-Nearest Neighbor (KNN) scene flow aggregation strategy.

We first estimate the dense 2D optical flow of scene points around the object using SEA-RAFT [24]. The depth value of each displaced pixel is back-projected into 3D world coordinates using the estimated camera intrinsics and extrinsics, yielding the 3D displacement ∆pti for each tracked scene point at frame t:

∆pti = pti,world+1 − pti,world (2) For each frame, we select the K-nearest scene points to the object centroid in 3D space and aggregate their displacement vectors to estimate the object’s motion:

1 K k∈TopK

∆ptobj =

∆ptk (3)

This aggregated displacement drives the object to faithfully follow the dominant movement of nearby scene surfaces while suppressing noisy flow estimates. To propagate the object in 4D space, the 3D coordinates yt′ are sequentially updated:

yt′ = yt′−1 + ∆ptobj−1 (4) where the sequence is initialized with ys′ = ys from Equation (1).

Camera-Aligned Reprojection. The updated 3D object points yt′ are projected onto the image plane of frame t using the estimated camera intrinsics K

and extrinsics Pt = [Rt|tt]:

 

  ∼ K Rtyj,t′ + tt (5)

uj,t vj,t 1

By projecting and rasterizing all visible points, we obtain the object’s silhouette for each frame. This reprojection naturally accounts for camera motion, parallax, and occlusion, producing geometrically consistent renderings from real camera viewpoints. The projected silhouettes are further refined using SAM 2 [20] to yield a temporally coherent binary mask sequence {Mt}Tt=1, serving as the foundational spatial condition for synthesis.

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

- Fig. 3: ROSE++ dataset generation pipeline. Reference object images are synthesized using an image editing model [1] and selected through a DINO-guided rejection sampling pipeline to ensure high appearance fidelity.

##### 3.3 Optics-Aware Video Synthesis

Large-scale mask-conditioned video generation models (e.g., VACE [8]) provide an efficient foundation for object insertion, as they are trained to strongly adhere to the given spatial condition. However, their localized nature strictly confines generation within the provided mask, leaving the surrounding region untouched. Consequently, they fail to synthesize scene-wide optical interactions, such as cast shadows and reflections, that naturally extend beyond the object’s strict boundary.

ROSE++ Dataset Construction. To supervise the generation of these physical lighting effects, we construct ROSE++, a specialized dataset derived from the ROSE object removal benchmark [14]. We reorganize the dataset’s structure by treating the object-removed video as the source Vsrc and the object-present video as the target Vtgt. This inherently embeds insertion-induced illumination changes into the supervision signal, as shadows and lighting variations absent in the source safely appear in the target.

Since the original dataset lacks isolated reference images, we generate them using an image editing model [1] coupled with a rejection sampling pipeline. Multi-view object crops fj are extracted from Vtgt using the ground-truth masks and provided to the editing model to synthesize candidate reference images on white backgrounds, with scene-dependent lighting cues removed. This ensures

that a model trained with ROSE++ infers illumination from the target scene rather than relying on copy-and-paste shortcuts from the reference image. To filter out candidate images that deviate from the true object’s appearance, we employ a rejection sampling strategy. The generated candidates are ranked using a DINO-based similarity metric [16] against the cropped frames:

N

1 N

sim(ΦDINO(ˆok),ΦDINO(fj)) (6)

sk =

j=1

where oˆk denotes the k-th generated candidate, ΦDINO(·) extracts the feature embeddings, and sim(·,·) represents cosine similarity. The highest-scoring candidate is selected as the final reference image.

##### Optics-Aware Representation Alignment. To encourage the generative model

to synthesize optical variations beyond the primary object mask, we propose the representation alignment technique. As illustrated in Fig. 4(a), we compute an optics-aware extended mask (Mext) by extracting the pixel-wise difference between Vsrc and Vtgt. After thresholding and morphological post-processing, this mask successfully captures both the object region and its associated photometric footprint (i.e., shadows and reflections), such that M ⊂ Mext.

During training, both the mask and the extended-mask are tokenized and passed through a shared diffusion backbone (Fig. 4(b)). We extract intermediate features from multiple transformer blocks respectively and enforce alignment between the mask features (Fl) and the extended-mask features (Flext):

Lalign =

l

||Fl − sg(Flext)||22 (7)

Crucially, a stop-gradient operator (sg) is applied to the extended-mask branch, allowing it to act as a fixed, optics-aware teacher. This alignment forces the standard fine-mask input to anticipate and generate extended optical variations, even during inference when only the tightly bound fine mask is provided.

First-Frame Anchoring and LoRA Fine-Tuning. We fine-tune the pretrained mask-conditioned video diffusion model [8] on ROSE++ using LoRA adapters [6], preserving its core generative prior while adapting it to opticsaware object insertion. To further maximize visual fidelity, we employ an optional but highly effective first-frame anchoring technique. Because image-level object insertion benefits from highly mature generative priors compared to the more complex VOI task, we first synthesize a high-quality edited image using a dedicated image insertion model [27]. We then provide this synthesized image to our video diffusion model as an explicit first-frame conditioning signal. This establishes highly reliable object appearance and local illumination cues, which are naturally propagated to subsequent frames, ensuring robust temporal consistency in color, texture, and lighting throughout the final synthesized video.

[Figure 62]

[Figure 63]

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

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

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

- Fig. 4: (a) We extract an extended mask containing the object and its optical footprint. (b) By explicitly aligning the intermediate representations of the standard fine mask to those of the extended mask via an alignment loss, the model learns to synthesize physical lighting effects (e.g., cast shadows) that extend beyond the strict object mask boundary.

Subject Consistency VBench Type Method CLIP-I ↑ DINO-I ↑

Background Consistency ↑

Subject Consistency ↑

Motion Smoothness ↑

Imaging Quality ↑

Pika-Pro [18] 0.4940 0.3856 0.9080 0.8720 0.9889 0.6546 Kling [22] 0.6349 0.5028 0.9335 0.9494 0.9940 0.7069

Commercial

AnyV2V [11] 0.7033 0.2217 0.8884 0.8699 0.9795 0.5973 ReVideo [15] 0.7385 0.3651 0.9391 0.9403 0.9906 0.6526 Señorita [29] 0.7499 0.3982 0.9262 0.9266 0.9902 0.6333

Open Source

VACEour mask [8] 0.7368 0.5060 0.9011 0.8855 0.9887 0.6046 Ours 0.8132 0.5669 0.9503 0.9534 0.9925 0.7473

Table 1: Quantitative comparisons with baseline methods.

### 4 Experiments

##### 4.1 Experimental Setup

Evaluation Dataset. We introduce VOIBench, a new benchmark designed to rigorously assess video object insertion. VOIBench spans three axes that jointly stress geometric and photometric robustness: scene (indoor / outdoor / natural), occlusion (none / dynamic / initially-occluded), and lighting (daylight / indoor / low-light), giving the benchmark broad coverage and evaluative power. The dataset consists of 200 video clips covering a broad spectrum of settings, such as indoor scenes, outdoor environments, and natural landscapes, with each video containing a pair of objects. We crawled contextually relevant objects for each scene.

Baselines and Evaluation Metrics. We evaluate against top commercial tools (Pika-Pro [18], Kling [22]) and representative open-source models (AnyV2V [11], ReVideo [15], Señorita [29]), as well as a VACE-based variant (VACEour mask), conditioned on our generated mask, to ensure fair VOI evaluation. We also report ablation results to isolate the impact of our proposed components.

We assess three key aspects: (1) Subject Consistency via CLIP-I [19] and DINO-I [16]; (2) Video Quality using VBench [7] (Image Quality, Background/

[Figure 93]

10 H. Jin, H. Jang, J. Hyung, J. Kim et al.

[Figure 94]

[Figure 95]

- Fig. 5: Qualitative comparisons with baseline methods. Best viewed when zoomed in.

Subject Consistency, Motion Smoothness); and (3) Multi-View Consistency [7] to gauge how reliably the object is preserved across viewpoint changes and dynamic occlusions.

##### 4.2 Qualitative Results

- Fig. 5 qualitatively highlights two key factors: 1) a 4D-aware mask for occlusionrobust insertion and 2) Optics-Aware Representation Alignment for realistic illumination effects. Without a 4D-aware mask, Kling [22] and Señorita [29] fail to maintain a stable separation between the insertion region and the rest of the scene, leading to background drift and deformation when a dynamic occluder (the moving tube) crosses the target area. In particular, even with a detailed prompt specifying the insertion location (“Add a paper boat to the right rear of the river in the video background.”), Kling often misinterprets the scene and transforms the tube into a paper boat. Señorita, which propagates edits from an edited first frame, degrades over time and cannot sustain consistent insertions under occlusions. In contrast, methods equipped with a 4D-aware mask (VACE [8] and ours) preserve both the tube geometry and the original background, producing consistent insertions. However, VACE lacks optics-aware modeling, making the inserted boat appear pasted with implausible reflections and weak cast shadows. Additional qualitative results for other baselines are provided in the supplementary material.

##### 4.3 Quantitative Results

To assess subject consistency, we uniformly sample 10 frames from each generated video and measure how closely they match the reference subject using

Subject Consistency VBench Method CLIP-I ↑ DINO-I ↑

Background Consistency ↑

Subject Consistency ↑

Motion Smoothness ↑

Imaging Quality ↑

VACE - , our 4D mask [8] 0.7368 0.5060 0.9011 0.8855 0.9887 0.6046 VACEROSE++, bbox mask [8] 0.6383 0.4856 0.9026 0.8818 0.9864 0.5824 VACEROSE++, our 4D mask [8] 0.7485 0.5103 0.9035 0.8895 0.9888 0.6583 Ours 0.8132 0.5669 0.9503 0.9534 0.9925 0.7473

Table 2: Fair comparison against VACE fine-tuned on ROSE++ with our 4D-aware mask and with a coarse bounding-box mask, evaluated on the VOIBench. Both finetuned VACE variants underperform our full model.

subject-masked regions only. As shown in Table 1, InsertAnywhere achieves the highest CLIP-I [19] and DINO-I [16] scores among all compared methods, demonstrating superior identity preservation. Moreover, InsertAnywhere attains the best overall VBench score, with clear advantages in Background Consistency, Subject Consistency, and Imaging Quality. Overall, the results suggest that our model not only provides a reliable 4D-aware mask for subject control but also improves full-video quality without sacrificing visual fidelity. In contrast, Kling and Pika-Pro often introduce undesirable changes to the original background during insertion. To verify that our gains stem from our design rather than from data exposure, we additionally fine-tune VACE on ROSE++ with our 4D-aware mask and with a coarse bounding-box mask (Table 2). Both variants underperform our full model, confirming that the improvement comes from our optics-aware design rather than merely from training on ROSE++. Notably, our method is best on five of the six metrics; the only exception is Motion Smoothness, where we closely match Kling (0.9925 vs. 0.9940), a gap within VBench’s per-clip noise.

##### 4.4 Ablation Study

We evaluate the effectiveness of our proposed components through a progressive ablation study, with qualitative and quantitative results detailed in Fig. 6 and Tab. 3. First, introducing our 4D-aware mask sequence successfully resolves dynamic occlusion failures (e.g., orange box, row 2), though basic object fidelity remains low. Next, incorporating first-frame anchoring significantly enhances identity preservation, yet temporal inconsistencies in shape and color still emerge post-occlusion. While subsequent LoRA fine-tuning on the ROSE++ dataset successfully stabilizes temporal consistency and appearance, the network still struggles to hallucinate external optical interactions. Finally, integrating our Optics-Aware Representation Alignment (Lalign) naturally synthesizes insertioninduced lighting effects, such as shadows. Combining all components yields geometrically accurate and photometrically realistic insertions with high object fidelity across the entire sequence.

Effect of Alignment Loss on Optics-Aware Rendering. Table 4 (right) reports quantitative comparisons for rendering quality using LPIPS/PSNR/SSIM against the ground truth. To enable a paired evaluation with ground truth,

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

- Fig. 6: Qualitative ablation study. Sequentially adding our proposed components progressively improves occlusion handling, object fidelity, and realistic optics-aware generation. Best viewed zoomed in.

Subject Consistency VBench Method CLIP-I ↑ DINO-I ↑

Background Consistency ↑

Subject Consistency ↑

Motion Smoothness ↑

Multi-View Consistency ↑

Imaging Quality ↑

- Config (a) 0.7532 0.3861 0.9232 0.9380 0.9913 0.6298 0.5308

- Config (b) 0.7880 0.5135 0.9175 0.9290 0.9911 0.6318 0.5436

- Config (c) 0.8122 0.5678 0.9429 0.9520 0.9916 0.7101 0.5857 Ours 0.8132 0.5669 0.9503 0.9534 0.9925 0.7473 0.5865

Table 3: Quantitative results of the ablation study.

we hold out 25 samples from ROSE++ as a test set and compute all metrics on this split. We compare VACE, our ROSE++ LoRA fine-tuned model without the alignment loss (Ours w/oL

), and the full model trained with the proposed alignment loss (Ours). We evaluate results under two complementary regions: Object region covers only the precise foreground area of the inserted object, while Optics region isolates the area affected by the object-induced optical effects such as shadows, specular highlights, and contact-dependent illumination changes. Using Object Mask, the two Ours variants achieve similar

align

[Figure 113]

- Fig. 7: Impact of Optics-Aware Representation Alignment. Attention maps for (a) a

baseline model (w/o Lalign) with a fine mask, (b) a baseline model with an extended mask, and (c) our model with a fine mask. Despite training on the same ROSE++ dataset, the baseline (a) remains strictly confined to the inserted object. In contrast, our aligned model (c) exhibits strong external activations closely matching (b). This confirms that our alignment strategy effectively trains the network to synthesize surrounding optical effects (e.g., shadows) without requiring an extended input mask.

performance, indicating that the object appearance fidelity is largely insensitive to Lalign. In contrast, under Optics Mask, Ours yields clear gains with higher PSNR/SSIM and lower LPIPS, demonstrating that Lalign effectively improves the preservation and consistency of optics-aware effects. We further verify that Lalign does not degrade fidelity in the unedited region: across the held-out ROSE++ split, the unedited-region PSNR is nearly identical with and without Lalign (VACEour mask: 33.27, Oursw/o L

: 33.31, Ours: 33.30), while the optics-region PSNR rises by +7.3dB. The mild color drift occasionally observed thus reflects sample-to-sample variation rather than a systematic effect of Lalign. Qualitative results in Fig. 8 corroborate this trend: in both the desk-top mouse and the floating tube examples, VACE shows noticeable appearance mismatch to the reference and reduced overall image quality, whereas both Ours variants maintain comparable object quality, with Ours producing more realistic and coherent specular highlights and shadows than Ours w/o L

align

.

align

- Fig. 7 visualizes the network’s attention maps during generation. We compare

three configurations: (a) a baseline model trained without Lalign conditioned on a fine mask, (b) baseline model conditioned on an extended mask, and (c) our full model conditioned on a fine mask. Despite being fine-tuned on the same ROSE++ dataset, the attention of the baseline model (a) remains strictly confined to the inserted object’s boundaries. In contrast, when provided only the tightly-bound fine mask, our aligned model (c) exhibits strong external activations that closely mirror the extended mask conditions of (b). This visual evidence confirms that our alignment strategy effectively teaches the network to anticipate and synthesize surrounding optical effects, such as cast shadows, without requiring an explicitly expanded mask during inference.

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

- Fig. 8: Qualitative ablation for Optics-Aware Representation Alignment.

Multi-View Consistency ↑ Oursrandom 0.5295

|Methods<br><br>|Object Mask LPIPS ↓ PSNR ↑ SSIM ↑<br><br>|Optics Mask LPIPS ↓ PSNR ↑ SSIM ↑|
|---|---|---|
|VACEour mask [8] Ours w/o Lalign Ours<br><br>|0.0250 21.1980 0.9795 0.0137 26.0282 0.9843<br><br>0.0161 25.8785 0.9872<br><br>|0.0130 19.8345 0.9877 0.0121 19.9065 0.9883 0.0089 27.2362 0.9920<br><br>|

Methods

###### Ours 0.5865

Table 4: Left: Multi-view Consistency evaluation. Models trained with random-frame references vs. our ROSE++ reference. Right: Quantitative evaluation of shadow rendering quality on object and optics masks, comparing VACE, Oursw/o Lalign, and Ours.

Effectiveness of ROSE++ We evaluate whether the generated reference images in ROSE++ successfully prevent copy-and-paste artifacts. To do so, we compare our model against a baseline trained with reference objects trivially cropped from the target video. As reported in Table 4 (left), our method achieves substantially higher multi-view consistency. This confirms that our dataset construction forces the network to genuinely infer 4D geometry and lighting, rather than relying on direct image copying.

### 5 Conclusion

We introduced InsertAnywhere, a novel framework for production-grade Video Object Insertion (VOI). By combining Geometrically Grounded Mask Propagation with an optics-aware diffusion model, our approach simultaneously guarantees precise spatial control and photorealistic rendering. From a single 3D anchor, our method automatically handles complex camera motions and dynamic occlusions. Furthermore, by leveraging our custom ROSE++ dataset and Optics-Aware Representation Alignment, the network successfully synthesizes object-induced physical lighting effects, such as shadows, outside the immediate

[Figure 125]

[Figure 126]

Fig. 9: Effectiveness of scene flow-based object propagation in Section 3.2. Unlike the static mask (second row), our proposed propagation ensures temporally consistent and 4D-aware alignment with moving objects (third row).

mask boundary. Extensive evaluations confirm that InsertAnywhere significantly outperforms existing baselines in both geometric consistency and photometric realism, paving the way for advanced practical applications in virtual product placement and visual effects.

Limitations. Even when the propagated mask is not perfectly pixel-accurate, our pipeline re-synthesizes regions beyond the boundary and absorbs small errors into a plausible output. Nonetheless, complex physics-aware interactions (e.g., dropping a heavy object onto a soft sofa) lie outside our current scope and are left for future work.

### Acknowledgement

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2019-II190075, Artificial Intelligence Graduate School Program (KAIST)). This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2025-02263841, Development of a Real-time Multimodal Framework for Comprehensive Deepfake Detection Incorporating Common Sense Error Analysis). This work was supported by and carried out in close collaboration with SK Telecom (SKT). This research was supported by Culture, Sports, and Tourism R&D Program through the Korea Creative Content Agency (KOCCA) grant funded by the Ministry of Culture, Sports, and Tourism (MCST) in 2024 (Project Name: Development of Technology for Convergence Performance Planning and Production Platform to Revitalize the Production of Convergence Performance by Traditional Artist Dance Music, Project Number: RS-202400398536, Contribution Rate: 30%).

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 2. Bai, C., Shao, Z., Zhang, G., Liang, D., Yang, J., Zhang, Z., Guo, Y., Zhong, C., Qiu, Y., Wang, Z., et al.: Anything in any scene: Photorealistic video object insertion. arXiv preprint arXiv:2401.17509 (2024)
- 3. Chen, J., Li, X., Bai, X., Ma, T., Zhang, P., Chen, Z., Li, G., Liu, L., Zhao, S., Li, B., et al.: Omniinsert: Mask-free video insertion of any reference via diffusion transformer models. arXiv preprint arXiv:2509.17627 (2025)
- 4. Chen, X., Huang, L., Liu, Y., Shen, Y., Zhao, D., Zhao, H.: Anydoor: Zero-shot object-level image customization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6593–6602 (2024)
- 5. Cheng, H.K., Oh, S.W., Price, B., Schwing, A., Lee, J.Y.: Tracking anything with decoupled video segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1316–1326 (2023)
- 6. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR (2022)
- 7. Huang, Z., Zhang, F., Xu, X., He, Y., Yu, J., Dong, Z., Ma, Q., Chanpaisit, N., Si, C., Jiang, Y., et al.: Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503 (2024)
- 8. Jiang, Z., Han, Z., Mao, C., Zhang, J., Pan, Y., Liu, Y.: Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598 (2025)
- 9. Karaev, N., Makarov, Y., Wang, J., Neverova, N., Vedaldi, A., Rupprecht, C.: Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6013–6022 (2025)
- 10. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4015–4026 (2023)
- 11. Ku, M., Wei, C., Ren, W., Yang, H., Chen, W.: Anyv2v: A tuning-free framework for any video-to-video editing tasks. Transactions on Machine Learning Research

(2024)

- 12. Liu, S., Wang, T., Wang, J.H., Liu, Q., Zhang, Z., Lee, J.Y., Li, Y., Yu, B., Lin, Z., Kim, S.Y., et al.: Generative video propagation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17712–17722 (2025)
- 13. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In: European conference on computer vision. pp. 38–55. Springer (2024)
- 14. Miao, C., Feng, Y., Zeng, J., Gao, Z., Liu, H., Yan, Y., Qi, D., Chen, X., Wang, B., Zhao, H.: Rose: Remove objects with side effects in videos. arXiv preprint arXiv:2508.18633 (2025)
- 15. Mou, C., Cao, M., Wang, X., Zhang, Z., Shan, Y., Zhang, J.: Revideo: Remake a video with motion and content control. Advances in Neural Information Processing Systems 37, 18481–18505 (2024)
- 16. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

- 17. Piccinelli, L., Sakaridis, C., Yang, Y.H., Segu, M., Li, S., Abbeloos, W., Van Gool, L.: Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110 (2025)
- 18. Pika: Pika additions. https://pika.art/pikadditions (2025), accessed: 2025-1114
- 19. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)
- 20. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)
- 21. Saini, N., Bodla, N., Shrivastava, A., Ravichandran, A., Zhang, X., Shrivastava, A., Singh, B.: Invi: Object insertion in videos using off-the-shelf diffusion models. arXiv preprint arXiv:2407.10958 (2024)
- 22. Team, K.A.: Klingai (2025), https://app.klingai.com/global/multimodal-tovideo/add-object/new
- 23. Tu, Y., Luo, H., Chen, X., Ji, S., Bai, X., Zhao, H.: Videoanydoor: High-fidelity video object insertion with precise motion control. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. pp. 1–11 (2025)
- 24. Wang, Y., Lipson, L., Deng, J.: Sea-raft: Simple, efficient, accurate raft for optical flow. In: European Conference on Computer Vision. pp. 36–54. Springer (2024)
- 25. Xiang, J., Lv, Z., Xu, S., Deng, Y., Wang, R., Zhang, B., Chen, D., Tong, X., Yang, J.: Structured 3d latents for scalable and versatile 3d generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21469–21480 (2025)
- 26. Yao, D.Y., Zhai, A.J., Wang, S.: Uni4d: Unifying visual foundation models for 4d modeling from a single video. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1116–1126 (2025)
- 27. Yu, Y., Zeng, Z., Zheng, H., Luo, J.: Omnipaint: Mastering object-oriented editing via disentangled insertion-removal inpainting. arXiv preprint arXiv:2503.08677

(2025)

- 28. Zhao, Q., Ma, Z., Zhou, P.: Dreaminsert: Zero-shot image-to-video object insertion from a single image. arXiv preprint arXiv:2503.10342 (2025)
- 29. Zi, B., Ruan, P., Chen, M., Qi, X., Hao, S., Zhao, S., Huang, Y., Liang, B., Xiao, R., Wong, K.F.: Señorita-2m: A high-quality instruction-based dataset for general video editing by video specialists. In: NeurIPS D&B (2025)

## InsertAnywhere: Geometrically Grounded and Optics-Aware Video Object Insertion

#### Supplementary Material

This supplementary material covers: Implementation Details (Sec. A), User Study (Sec. B), Retrieval Prompt Details (Sec. C), and Additional Qualitative Comparisons (Sec. D).

### A Implementation Detail.

Training Details. Our video generation is performed with Wan2.1-VACE14B [8], a diffusion-based video generation model fine-tuned with LoRA to adapt to our insertion-specific domain. The model is fine-tuned for 6,500 iterations with a learning rate of 1×10−4 and a LoRA rank of 128. The entire training process, conducted on a single NVIDIA H200 GPU, takes approximately 60 hours. All videos are trained and generated at a spatial resolution of 832 × 480 with 81 frames per clip.

Inference Cost. On a single NVIDIA B200 GPU, our pipeline takes about 4 minutes for the 4D-mask stage and 4.5 minutes for video synthesis at 832 × 480 × 81 (peak memory 48GB), comparable to VACE-14B (4.5 minutes) and orders of magnitude faster than manual graphics pipelines. As VOI is an offline production task, this footprint is well within practical limits.

Extended-mask noise. Because ROSE is simulator-rendered, the noise in the extended (optics) mask is small. We further reduce it by frame-differencing the source and target edits, retaining only connected components near the groundtruth object mask by Euclidean-distance-transform (EDT) proximity, and applying morphological closing to merge the object with its associated shadow.

### B User Study

We conducted a user study with 20 participants by randomly sampling 10% of the videos in the test set. For each video, we prepared six evaluation questions, and participants were asked to choose the best-performing model for each question from four candidates: our method and three baselines selected based on the top scores in Table 1, namely Kling from the commercial group and Señorita and VACEour mask from the open-source group. To avoid positional bias, the order of the four candidate results was randomly shuffled for each participant. As illustrated in the Fig. B1, each question presented the source video and the reference object together with four candidate results, labeled (a)-(d), whose ordering was randomized across participants.

The evaluation questions are as follows:

– Object Realism. Assesses whether the inserted object exhibits physically plausible geometry, scale, and appearance without distortions or implausible configurations.

Object Realism

Lighting Consistency

Occlusion Integrity

Object-Video Consistency

Background Preservation

Overall Naturalness

Method (%)

Kling [22] 19.09 34.55 13.64 20.45 11.82 21.82 Señorita [29] 6.82 15.45 6.36 7.73 10.91 10.91 VACEour mask [8] 10.45 13.64 38.18 11.36 37.27 8.64 Ours 63.64 36.36 41.82 60.45 40.00 58.64

- Table B1: User study preference percentages across six evaluation criteria for four compared methods.

- – Lighting Consistency. Measures how well the inserted object matches the illumination and shading conditions of the surrounding scene.
- – Occlusion Integrity. Evaluates whether occlusion relationships with surrounding objects are handled correctly, without causing disappearance or distortion of existing scene elements.
- – Object–Video Consistency. Determines how consistent the inserted object is with visually or semantically related objects present in the original video.
- – Background Preservation. Checks whether regions unrelated to the inserted object remain faithful to the original video without unnecessary alterations or artifacts.
- – Overall Naturalness. Captures the overall perceptual realism of the result, including the absence of visible artifacts and the user’s preferred choice for real-world deployment.

Table B1 reports the user preference percentages aggregated over the randomly sampled 10% subset of the test videos and all 20 participants. For each evaluation criterion, participants selected the method they judged to perform best, and the percentages indicate the proportion of total votes received by each method for that criterion.

Optics-Focused User Study. To more directly assess the perceptual benefit of our optics-aware synthesis, we conducted an additional user study with 30 participants on 50 clips explicitly selected for shadow- and reflection-rich scenes. As reported in Table B2, our method is again preferred across all six criteria, confirming that the optics-aware gain, not only the overall preference, is perceptually salient.

### C ROSE++ Prompt Details.

We use the prompt described in Fig. C2 to generate object images for constructing the ROSE++ dataset. For each video, we use this prompt to generate multiple object candidates with consistent visual appearance and select as the reference image the one with the highest DINO score.

Object Realism

Lighting Consistency

Occlusion Integrity

Object-Video Consistency

Background Preservation

Overall Naturalness

Method (%)

Kling [22] 24.5 26.0 12.0 16.4 9.7 15.0 Señorita [29] 7.9 22.7 6.0 6.7 8.9 8.5

VACEour mask [8] 9.7 5.3 26.0 10.3 28.5 18.5 VACEROSE++, our mask [8] 14.2 14.0 27.0 25.5 21.5 23.0 Ours 43.7 32.0 29.0 41.1 31.5 35.0

- Table B2: Optics-focused user study with 30 participants on 50 shadow/reflection-rich clips. Our method is preferred across all six criteria.

[Figure 127]

Fig. B1: User study example

### D Additional Qualitative Comparisons.

Fig. D4, D5, D6 and D7 present additional qualitative results of VOI in real-world scenarios. We compare our method against two commercial models, Pika-Pro [18] and Kling [22], and four open-source approaches: AnyV2V [11], ReVideo [15], Señorita [29], and VACE [8]. Given the varying input modalities of these models, we adapt the evaluation setup to ensure a fair comparison. Since Pika-Pro and Kling do not support mask inputs, we use text prompts to describe the target insertion areas as precisely as possible; the specific prompts used are detailed in each figure’s caption. For AnyV2V, ReVideo, and Señorita, the models do not support mask or reference image inputs. Instead, they accept a source video paired with an edited first frame; following their intended pipeline, we provide the source video and the same edited first frame used by our method. Finally, for VACE, which serves as our primary baseline and supports all relevant inputs, we provide the source video and reference image, along with our 4D-aware mask sequence. To ensure a fair comparison, we additionally supply VACE with the same edited first frame used by our method.

As shown in the figures, Pika-Pro and Kling tend to modify the original scene rather than performing localized insertion. For instance, in Fig. D5, Pika-Pro removes the penguin lying behind the insertion region. Similarly, Kling swaps

Fig. C2: Prompt for object retrieval in constructing the ROSE++ dataset

the existing tube with the target object in Fig. D4, while in Fig. D6, it removes the original sofa and generates a new black sofa instead. In addition, Pika-Pro frequently alters the global appearance of the video, including brightness and color tone. These models also struggle to faithfully preserve the appearance of the given object: Pika-Pro distorts the flag details of the paper boat in Fig. D4, while Kling generates a mug with a different shape from the reference in Fig. D7.

For AnyV2V, ReVideo, and Señorita, the main limitation appears in scenes involving occlusion. Across all four figures, these methods either fail to preserve the original content of the source video or fail to generate the target object accurately under occlusion, highlighting their limited ability to handle complex spatial relationships.

VACE, benefiting from our 4D mask as input, produces the most stable results among the baselines. However, it still occasionally distorts the appearance of the inserted object. Most notably, VACE completely fails to synthesize side effects such as shadows and reflections, which are critical for achieving realistic and coherent object insertion.

In contrast, our InsertAnywhere preserves the original scene content while accurately inserting the reference object into the target region, and naturally synthesizes optical effects such as shadows and reflections, resulting in more realistic and coherent compositions.

Furthermore, Fig. D3 visualizes the 4D-aware mask sequences generated by our geometrically grounded mask propagation module. Our masks are temporally propagated according to the reconstructed 4D scene geometry and camera motion, maintaining consistent alignment with the insertion region across frames. This temporally coherent guidance provides a more reliable conditioning signal for the generative model during video synthesis.

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

[Figure 150]

[Figure 151]

###### Fig. D3: Visualization of 4D-aware mask sequences generated by our geometrically grounded mask propagation module.

[Figure 152]

[Figure 153]

[Figure 154]

###### Fig. D4: Additional qualitative results. (Kling / Pika-Pro prompt: “Add a paper boat to the right rear of the river in the video background.”)

[Figure 155]

[Figure 156]

[Figure 157]

ACEV

Fig. D5: Additional qualitative results. (Kling / Pika-Pro prompt: “Add a coca cola behind the moving penguin in video.”)

[Figure 158]

[Figure 159]

ACEV

Fig. D6: Additional qualitative results. (Kling / Pika-Pro prompt: “Add a teddy bear right sofa in video background.”)

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

ACEV

Fig. D7: Additional qualitative results. (Kling / Pika-Pro prompt: “Add a stanley mug on the table back of pillar in video.”)

