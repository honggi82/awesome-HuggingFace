## VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence

# arXiv:2312.02087v2[cs.CV]5Dec2023

Yuchao Gu1,2, Yipin Zhou2, Bichen Wu2, Licheng Yu2, Jia-Wei Liu1, Rui Zhao1, Jay Zhangjie Wu1, David Junhao Zhang1, Mike Zheng Shou1*, Kevin Tang2

1Show Lab, National University of Singapore 2GenAI, Meta https://videoswap.github.io/

Figure 1. Customized video subject swapping results with VideoSwap. VideoSwap supports shape change in the swapped results while aligning with the source motion trajectory. The swapped target can be either a predefined concept from a pretrained model (e.g., helicopter) or a customized concept (denoted by V ∗). Previous methods based on implicit motion encoding and dense correspondence do not perform well in subject swapping with shape changes. We encourage readers to click and play the video clips in this figure using Adobe Acrobat.

#### Abstract

Current diffusion-based video editing primarily focuses on structure-preserved editing by utilizing various dense correspondences to ensure temporal consistency and motion alignment. However, these approaches are often ineffective when the target edit involves a shape change. To embark on video editing with shape change, we explore customized video subject swapping in this work, where we aim to replace the main subject in a source video with a target

∗Corresponding Author

subject having a distinct identity and potentially different shape. In contrast to previous methods that rely on dense correspondences, we introduce the VideoSwap framework that exploits semantic point correspondences, inspired by our observation that only a small number of semantic points are necessary to align the subject’s motion trajectory and modify its shape. We also introduce various user-point interactions (e.g., removing points and dragging points) to address various semantic point correspondence. Extensive experiments demonstrate state-of-the-art video subject swapping results across a variety of real-world videos.

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>V catA V dogA V dogB V porsche V carA V yacht V jet<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>Reference Images for Customized Concepts<br><br>V sailboat<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|

- Figure 2. Customized video subject swapping results of VideoSwap on various concepts. The swapping target can either be a predefined concept in the pretrained model (e.g., helicopter) or a customized concept created by ED-LoRA [15] (denoted as V ∗). We encourage readers to click and play the video clips in this figure using Adobe Acrobat. For legal issues, we cannot display the human swap results.

#### 1. Introduction

Diffusion-based video editing [5, 11, 29, 34, 42, 57, 58, 64] is an emerging field that harnesses the capabilities of pretrained text-to-image/video diffusion models [16, 19, 43, 49] to facilitate a range of video editing tasks, including style change and subject/background swapping. The main challenge in video editing lies in how to extract mo-

tion from the source video and transfer it to the edited video while ensuring temporal consistency. Pioneer TuneA-Video [57] implicitly encodes source motion in the diffusion model weights by tuning from the source video. While it demonstrates versatile applications for video editing, the temporal consistency is far from satisfactory. Subsequent works make use of various dense correspondences extracted

from the source video, including attention maps [30, 42], edge/depth maps [11, 29, 60, 64], optical flows [60], and deformation fields [5, 7] for video editing. While achieving better temporal consistency, dense correspondence imposes strict shape constraints on the target edit, which makes it ineffective for video editing with shape changes.

To embark on video editing with shape change, we delve into a challenging task: customized video subject swapping. Unlike the conventional video subject swapping addressed in previous works [5, 60, 64], where the swapped subject conforms to the shape of source subject, customized subjects have a clearly defined identity in terms of both appearance and shape. These distinctive characteristics should be preserved in the target edit. Therefore, previous structurepreserved video editing methods are often ineffective for this problem, as shown in Fig. 1(b).

To address customized video subject swapping, our primary insight is that the subject’s motion trajectory can be effectively described using a small number of semantic points. As shown in Fig. 1(a), the motion trajectory of an airplane can be depicted by semantic points located at its wings, nose, and tail. This insight naturally leads us to the following question: Can we utilize these semantic points as correspondences to align the motion trajectory while relaxing the shape constraints in video editing? To answer this question, we conduct a toy experiments and observe that it is possible to learn semantic point correspondence for a specific video subject using just a small number of source video frames. Users can interact with learned semantic points to generate unseen poses or modify the shape of the video subject. These observations suggest the potential for integrating semantic point correspondence into video editing, provided that we can obtain an accurate semantic point sequence for the target edit.

To unleash the potential of semantic point correspondence, we introduce the VideoSwap framework for customized video subject swapping, which comprises the following primary designs: 1) Integrating the motion layer into the image diffusion model to ensure essential temporal consistency. 2) Registering semantic points on the source video and utilizing them to transfer the motion trajectory of source subject to the target edit. 3) Introducing user-point interactions (e.g., removing or dragging points) for various semantic point correspondence.

Our contributions are summarized as follows:

- • Empirical observations that reveal the potential of semantic point correspondence for aligning motion trajectories and changing shapes in video editing.
- • The VideoSwap framework, which minimizes user intervention while unleashing the potential of semantic point correspondence in customized video subject swapping.
- • State-of-the-art results in customized video subject swapping, as demonstrated in Fig. 2.

#### 2. Related Work

##### 2.1. Diffusion-Based Video Editing

Structure-Preserved Video Editing. FateZero [42] and Video-P2P [30] extract cross- and self-attention from the source video to control spatial layout. To achieve stricter alignment of temporal consistency with the source video, Rerender-A-Video [60], Gen-1 [11], ControlVideo [64], and TokenFlow [13] extract and align optical flow, depth/edge maps, and nearest-neighbour field from the source video respectively, resulting in improved temporal consistency. StableVideo [5], VidEdit [7], and CoDEF [38] learn the canonical space for editing following the Layered Neural Atlas [23] or the deformation field in Dynamic Nerf [33, 41]. While achieving promising results in structure-preserved video editing, the above methods based on various dense correspondence are not suitable for handling subject swapping involving shape changes.

Video Editing with Shape Change. Tune-A-Video (TAV) [57] and FateZero [42] with TAV checkpoint can be utilized for video editing with shape change, as they implicitly encode the motion in model weights through tuning on the source video. However, they suffer from structure and appearance leakage due to model tuning. Shape-aware editing [27] is built on the Layered Neural Atlas [23], employing semantic correspondences to estimate shape deformation and warp the atlas. Nevertheless, texture warping will lead to unrealistic textures.

Video Diffusion Models. Previous video editing primarily relies on the text-to-image diffusion model [43]. However, recent advancements are occurring in video foundation models [3, 16, 55, 62]. In this work, we add a motion layer [16] to the image diffusion model to provide essential temporal consistency for video editing, and we focus on exploiting semantic point correspondence to align the motion trajectory of the video subject.

##### 2.2. Point Correspondence

Point Correspondence in Diffusion Models. DIFT [52] initially uncovers robust semantic point correspondences in diffusion models. Building upon the observation in DIFT, subsequent works, DragDiffusion [48] and DragonDiffusion [35] extend DragGAN [39] to support interactive pointbased image editing in diffusion models. However, point correspondences and interactive drag-based editing are seldom investigated for video editing.

Tracking Any Point in Video. TAP-Vid [9] first introduces the problem of tracking any point (TAP). Unlike optical flow estimation, TAP requires the establishment of longrange correspondence for all points in a video. In our work, we use TAP to reduce human intervention in annotating subject keypoints and acquire long-range motion estimation. Although several works [9, 10, 22, 56] address the TAP

- Figure 3. Toy experiment exploring semantic point correspondence. We encourage readers to click and play the video clips in this figure using Adobe Acrobat.

problem, we choose to employ Co-Tracker [22], which is the most efficient solution available.

##### 2.3. Concept Customization

Concept customization is mainly categorized into tuningbased approaches [12, 15, 26, 44, 54] and tuning-free solutions [21, 45, 47, 59, 61]. Tuning-free solutions are fast but typically adhere closely to the provided reference image and lack variation. On the other hand, tuning-based solutions can leverage multi-view images to ensure variation in the given concepts and maintain the same inference behavior to pretrained diffusion models. In this paper, we employ the tuning-based ED-LoRA [15] for encoding subject identity.

In addition to image customization for generation (i.e., noise to image), several works employ customization techniques for subject-driven image editing (i.e., image to image). CustomEdit [6] and Photoswap [14] propose the invert concept identity to text token and utilize attention swapping to preserve the layout and pose of the source image. While these methods achieve promising swapping results, the injection of attention maps tends to constrain the shape and leak color information to the target swapped result, as observed in DreamEdit [28]. In contrast to them, we only align the semantic points’ correspondence with the source subject and thus relax the shape constraints, better revealing the concept identity.

[Figure 25]

Figure 4. Overview of the VideoSwap pipeline for customized video subject swapping.

#### 3. VideoSwap

In this section, we start by presenting a toy experiment to illustrate our motivation to explore semantic point correspondence in Sec. 3.1. Subsequently, we offer an overview of the VideoSwap pipeline for customized video subject swapping in Sec. 3.2. Following that, we explain the process of injecting semantic point correspondence in Sec. 3.3 to Sec. 3.5.

##### 3.1. Motivation

Dense correspondences explored in previous video editing methods [5, 11, 13, 38, 60] restrict the subject’s shape change in the edited video. Therefore, our goal is to find a more flexible correspondence that can transfer the source subject’s motion trajectory without imposing strict shape constraints. Motivated by this, we investigate sparse semantic points as correspondences. Unlike dense correspondences such as depth, edge, and optical flow, which are lowlevel cues shared across all video subjects, semantic points vary with different open-world concepts. Therefore, it is not feasible to train a general condition model for injecting semantic point correspondence. Instead, the question we aim to address in this section is, is it possible to learn semantic point control for a specific source video subject using only a small number of source video frames?

Toy Experiment Setting. To address the above question, we perform a toy experiment. Firstly, for a given video, we manually define a set of semantic points. Next, we annotate the same set of points on eight frames of this video. Finally, we train a T2I-Adapter [36] on these data pairs, as illustrated in Fig. 3(a, b), to determine whether these semantic points can be used to control the source video subject.

Observation 1: Semantic points optimized on source video frames have the potential to align the subject’s motion trajectory and change the subject’s shape. As shown in Fig. 3(c, left), we drag the points on the cat’s face. Despite this edited point map is not in the training data, the re-

[Figure 26]

- Figure 5. Pipelines for semantic point extraction (Sec. 3.3) and semantic point registration (Sec. 3.4) in VideoSwap. In semantic point extraction, users define semantic points at a keyframe. We then extract the trajectory and embedding of those semantic points from the video. In semantic point registration, the semantic point embedding is projected by multiple 2-layer learnable MLPs, placed in empty features based on their coordinates, and then added element-wise to the diffusion model as motion guidance.

sulting image closely follows the adjusted semantic points, effectively generating the unseen pose of the subject. As shown in Fig. 3(c, right), when we drag the boundary semantic points of the car, the edited subject will also reshape to align with the semantic point. This suggests the potential of utilizing semantic points to align the motion trajectory or change the shape when a sequence of dragged points for all video frames is accessible.

Observation 2: Semantic points optimized on source video frames can transfer across semantic and low-level changes. As demonstrated in Fig. 3(d), when we replace the source subject with different semantics or modify the low-level information in the prompts, the optimized semantic point can also control the pose or shape of the target concept. This suggests that the semantic point can be transferred across both semantic and low-level changes.

##### 3.2. Overview of VideoSwap

###### 3.2.1 Task Formulation

In this paper, we focus on customized video subject swapping, with the goal of subject replacement and background preservation. Subject replacement requires preserving the identity of the target subject in the swapped results, encompassing both its appearance and shape. Simultaneously, background preservation requires the unedited background area to remain the same with the source video. The primary challenge of this task lies in aligning the motion trajectory of the source subject while preserving the identity of the target concept, particularly its shape.

###### 3.2.2 Overall Pipeline

The VideoSwap pipeline is illustrated in Fig. 4. Following the latent diffusion model [43], we encode the source video with a VAE encoder to obtain the latent space representation z0. Subsequently, DDIM inversion [8, 50] is applied to transform the clean latent z0 back to the noisy latent zT. After obtaining the DDIM inverted noise zT, we replace the source subject in the text prompt with the target subject and denoise it using the DDIM scheduler [50]. In this denoising process, we introduce semantic point correspondence to guide the subject’s motion trajectory, as detailed in Sec. 3.3Sec. 3.5. To preserve the unedited background, we leverage the concept of latent blending [1, 2], further explained in the Sec. 6.1. Additionally, we incorporate the following designs:

Adopting Motion Layers. We integrate the motion layers [3, 16] into the image diffusion model to ensure essential temporal consistency for video editing.

Supporting Predefined and Customized Concepts. We support both predefined concepts from the pretrained model and customized concepts. To create customized concepts, we train ED-LoRA [15] on a set of representative images to encode their identity. After training, these concept EDLoRAs can be used at the inference time.

##### 3.3. Semantic Point Extraction

We first extract the trajectories of semantic points and their associated semantic embeddings from the source video.

Point Trajectory Extraction. As depicted in Fig. 5, for a video containing N frames, users specify K semantic points at a keyframe i. These user-defined semantic points

[Figure 27]

- Figure 6. Point displacement propagation based on layered neural atlas (LNA) [20, 23]. Once a trained LNA is provided, users can drag a semantic point at the keyframe, and this displacement is consistently propagated to every frame through the canonical space of the LNA.

are then propagated to the remaining N − 1 frames using a point tracker [22] or detector [4]. Subsequently, the motion trajectory of all semantic points in the entire video is obtained and represented as Pcoord = {Tra(k)|k = 1...K}, where Tra(k) = {(xkn,ynk,n)|n = 1...N} represents the motion trajectory of semantic point k across all N frames.

Point Embedding Extraction. To leverage semantic point correspondence, it is crucial to associate each point with its semantics. Specifically, we extract the DIFT [52] feature Dn for each frame n. Subsequently, the point embedding for semantic point k at frame n is obtained as vnk = Dn(xkn,ynk), where (xkn,ynk) is retrieved from Pcoord. Following this, we aggregate the point embeddings acquired from all N frames to obtain the final embedding for each semantic point k by vk = N1 Nn=1 vnk. Finally, we obtain the semantic embedding for all semantic points, succinctly represented as Pemb = {vk|k = 1...K}.

##### 3.4. Semantic Point Registration

After acquiring the motion trajectory Pcoord and the embedding Pemb for semantic points, we register these semantic points on the source video to enable them to offer motion guidance for the video subject.

Sparse Motion Feature Creation. To utilize semantic points as guidance, we generate sparse motion features infused with semantic point embeddings, making them compatible with the Unet encoder. We denote Fenc = {F1enc,F2enc,F3enc,F4enc} as the multi-scale intermediate feature of the Unet encoder. For an input VAE-encoded latent with spatial-temporal size (H,W,N), the feature size of Flenc for each Unet stage l ∈ [1,4] can be computed as (H/2l,W/2l,N).

As depicted in Fig. 5 and detailed in Algorithm. 1, we create a series of multi-scale conditional features Fc = {F1c,F2c,F3c,F4c}. Notably, Fc shares the same size as Fenc and is initialized with zero vectors. And we introduce a series of learnable MLPs ϕ = {ϕl|l ∈ {1,2,3,4}}, each of ϕl project the point embedding to match the feature dimension of the corresponding Flc. Then, for each point (xkn,ynk,n) ∈ Pcoord, we compute its corresponding

spatial-temporal coordinate (x,y,n) at l-th Unet stage and assign the projected embedding based on the coordinate by Flc(x,y,n) = ϕl(Pemb(k)).

It is crucial to emphasize that the motion feature Fc demonstrates high sparsity, with only the semantic point trajectories containing the feature embeddings. Finally, Fc are added element-wise into the intermediate feature Fenc of Unet encoder as motion guidance:

Flenc = Flenc + Flc,l ∈ {1,2,3,4}. (1)

Algorithm 1: Sparse Motion Feature Creation

- 1 Input:
- 2 1. Point Trajectory Pcoord,
- 3 2. Point Embedding Pemb,
- 4 3. Learnable MLPs ϕ = {ϕl|l ∈ {1,2,3,4}}
- 5 Output:
- 6 Motion Feature Fc = {Flc|l ∈ {1,2,3,4}}
- 7 Initialize: Fc = {Flc = 0|l ∈ {1,2,3,4}};
- 8 for l ← 1 to 4 do

- 9 foreach (xkn,ynk,n) in Pcoord do

- 10 x,y = round(xkn/2l),round(ynk/2l);
- 11 Flc(x,y,n) = ϕl(Pemb(k))
- 12 end
- 13 end

Semantic Point Registration on Source Video. Our objective is to optimize the projection MLPs (ϕ) to facilitate better motion guidance from semantic points. This optimization objective is defined as

Eϵ∼N(0,I),t∼U(Tmin,T)||[ϵ−ϵθ(zt, t, p, ϕ(Pemb))]⊙Ω(Pcoord)||22,

min

ϕ

(2)

where p represents the embedding for the text prompt, and Ω(Pcoord) denotes the binary mask that only turns on around the semantic point.

We adopt two techniques to enhance the learning of semantic point correspondences in Eq. (2). The first technique is semantic-enhanced schedule, controlled by Tmin. We set Tmin = T/2 to enhance the learning at the higher

- Figure 7. Comparison of VideoSwap with several baselines built upon the same foundational model. The only difference lies in adopting different motion guidance. We encourage readers to click and play the video clips in this figure using Adobe Acrobat.

timestep, which prevents overfitting to low-level details and improves semantic point alignment. The second technique, point patch loss, constrains the computation of the loss to a local patch near each semantic point, which reduces structure leakage into the target swap. This is implemented by the a loss mask Ω in Eq. (2).

##### 3.5. User-Point Interaction at Inference Time

In Sec. 3.3, users define semantic points at the source video keyframe, and we subsequently extract their trajectory and semantic embedding in the video. To utilize these semantic points as correspondences, we register them on the source video in Sec. 3.4. Following these two steps, these semantic points become applicable for controlling the motion of the target object. In this section, we introduce user-point interaction to address various semantic point correspondence.

Adopting Source Point Sequence. If there exist one-toone semantic point correspondence between the source subject and the target subject, such as swapping the dog with V catA as illustrated in Fig. 1, we directly use the source point sequence as the motion guidance.

Removing Parts of Semantic Point. When there only exists partial semantic point correspondence between the source subject and the target subject, we can remove redundant points to loosen shape constraints. For example, in scenarios such as swapping an airplane for a helicopter, as depicted in Fig. 1, we remove semantic points associated with the airplane’s wings, given that helicopters typically lack wings. The remaining semantic points on the airplane’s nose and tail are retained as motion guidance.

Dragging of Semantic Point. In situations where there exists semantic point correspondence between the source and target subjects, but misalignment occurs due to shape morphing, we provide users the option to manually drag the semantic points on a keyframe for better alignment of the shape changes. For instance, when swapping a jeep (tall and narrow) for a VcarA (sports car, low and wide) as illustrated in Fig. 1, users can drag the semantic points to accurately

reflect the shape change.

Editing the shape change by dragging semantic points on a single frame is straightforward. However, consistently propagating these semantic point displacement over time is non-trivial, mainly because of the complex camera and subject motion in the video. Therefore, we introduce point displacement propagation to solve this problem.

As illustrated in Fig. 6, we follow the Layered Neural Atlas (LNA) [20, 23] to learn the canonical space, as detailed in the Sec. 6.2. Following LNA [20, 23], we establish a forward coordinate mapping from the video to the canonical space, denoted as M: (x,y,f) → (u,v), along with its corresponding backward mapping B: (u,v,f) → (x,y).

Given a semantic point, with coordinates at the keyframe fkey represented as (x,y,fkey), its trajectory over time can be expressed as a function of time f: (x(f),y(f)) = P(f). Suppose a user drag it to a new position at (x + dx,y + dy,fkey), we aim to estimate the edited trajectory P′(f) for f = {0,...,N}. We resort to LNA’s representation, and first compute a linearized estimation of its shifted position on the canonical coordinate:

[du,dv]T = JM(x,y,fkey)[dx,dy]T, (3) where JM denote the Jacobian matrix with respect to (x,y). Next, at a given time f, we estimate the edited coordinate in the pixel space as

P′(f) = P(f) + JB(u,v,f)[du,dv]T, (4) where (u,v) = B(x,y,fkey) and JB denote the Jacobian matrix with respect to (u,v). In practice, we approximate the Jacobian computation by

T 1/ε 1/ε

Ms(x + ε,y,f) − Ms(x,y,f) Ms(x,y + ε,f) − Ms(x,y,f)

, (5)

JM =

T 1/ε 1/ε

Bs(u + ε,v,f) − Bs(u,v,f) Bs(u,v + ε,f) − Bs(u,v,f)

, (6)

JB =

where ε represents the small coordinate shift. We then use this edited trajectory P′(f) for the dragged semantic point during inference.

- Figure 8. Ablation Study of our VideoSwap. We encourage readers to click and play the video clips in this figure using Adobe Acrobat.

|Methods/Metrics<br><br>| |Subject Identity<br><br>Motion Alignment<br><br>Temporal Consistency<br><br>Overall Preference|
|---|---|---|
|VideoSwapv.s.<br><br>|Compare to Previous Video Editing Methods<br><br>| |
| |Tune-A-Video<br><br>FateZero<br><br>Text2Video-Zero<br><br>Rerender-A-Video|80% v.s. 20% 72% v.s. 28% 80% v.s. 20% 78% v.s. 22%<br><br>74% v.s. 26% 67% v.s. 33% 73% v.s. 27% 70% v.s. 30% 76% v.s. 24% 71% v.s. 29% 80% v.s. 20% 80% v.s. 20%<br><br>81% v.s. 19% 72% v.s. 28% 84% v.s. 16% 84% v.s. 16%<br>|
| |Compare to Baselines on AnimateDiff| |
| |w/ DDIM w/ DDIM+TAV w/ DDIM+T2I-Adapter|70% v.s. 30% 74% v.s. 26% 69% v.s. 31% 73% v.s. 27% 68% v.s. 32% 60% v.s. 40% 66% v.s. 34% 69% v.s. 31% 66% v.s. 34% 59% v.s. 41% 65% v.s. 35% 66% v.s. 34%<br><br>|

VideoSwapv.s.

Table 1. Human Evaluation on Video Subject Swapping Results.

#### 4. Experiments

We implement our method using the Latent Diffusion Model [43] and adopt the motion layer in AnimateDiff [16]

- as the foundational model. All videos consist of 16 frames. The primary time cost is registering semantic points in the video, which requires about 3 minutes per video. Additional implementation details, as well as time and memory cost analyses, are included in the Sec. 7.2.

##### 4.1. Qualitative Comparison

Comparison with the State-of-the-Art. We qualitatively compare to Tune-A-Video [57], FateZero [42], Rerender-AVideo [60], TokenFlow [13] and StableVideo [5] in Fig. 1. Previous methods are less effective in revealing the correct shape of the target subject. Compared to them, VideoSwap can achieve a significant shape change while aligning the source motion trajectory.

Comparison with Baselines on AnimateDiff. As most state-of-the-art methods are based on image diffusion models, we also compare VideoSwap to several baselines on the AnimateDiff. The only distinctions from VideoSwap lie in different motion guidance, as shown in the results in Fig. 7:

- • DDIM: The DDIM sampling without other motion guidance cannot produce the correct motion trajectory.
- • DDIM+Tune-A-Video: If we tune the model as [57] to inject source motion, it achieves correct motion but suffers from severe structure and appearance leakage.
- • DDIM+T2I-Adapter: If we add spatial controls [36], such as depth, to control the editing, we observe that 1) the shape is restricted by the source, and 2) the deformable motion cannot follow the source video.

Compared to all constructed baselines, our VideoSwap with semantic point correspondence can effectively align the motion trajectory while preserving the target concept’s identity.

##### 4.2. Quantitative Comparison

We conduct both automatic and human evaluations to quantitatively compare VideoSwap with previous state-of-the-art methods and several baselines on AnimateDiff. Detailed evaluation settings and automatic evaluation results are provided in the Sec. 8. For human evaluation, we distribute 1000 questionnaires on Amazon Mturk to assess various criteria in customized video subject swapping. From the human evaluation results in Table. 1, our method achieves a clear advantage over the compared methods.

##### 4.3. Ablation Study

Sparse Motion Feature. There are several variants for encoding semantic points to generate motion guidance. The most straightforward approach is to encode the point map of semantic points using T2I-Adapter [36]. However, this method leads to severe overfitting, as illustrated in Fig. 8(a). The issue arises because the encoded feature added to the

diffusion model is non-sparse, and the background also contains features that may overfit to the source video. We use MLPs to encode point embeddings, positioning them in the empty feature to create sparse motion features for guidance without compromising video quality. Regarding point embeddings, we opt for DIFT embeddings, which inherently carry robust semantics. Compared to randomly initialized learnable embeddings, our approach achieves similar results with 3× less time for point registration step.

Point Patch Loss. In the semantic point registration, we employ a point patch loss to reconstruct the local patch surrounding each semantic point. If we omit the point patch loss and opt to directly reconstruct the entire video, we observe that the source structure leaks into the target swapped results and thus produce artifacts, as depicted in Fig. 8(b).

Semantic-Enhanced Schedule. Our goal is to employ semantic points to transfer motion trajectories, acting as a linkage between the source and target subjects. Therefore, we aim for the semantic points to emphasize high-level semantic alignment without transferring low-level details. This objective is achieved by registering points only during the early sampling steps, i.e., Tmin = T2 in Eq. (2). As shown in Fig. 8(c), this technique prevents the model from learning excessive low-level details and enhances semantic point alignment.

Drag-based Point Control. As illustrated in Fig. 8(d), our goal is to swap the black swan to the duck. If we directly use the source point sequence as guidance, the duck’s neck conforms to the shape of the black swan, resulting in an inferior identity. However, by employing the proposed point displacement propagation, we can drag the semantic point

- at the keyframe, ensuring a consistent motion trajectory after dragging. Utilizing the dragged semantic point trajectory as motion guidance allows us to accurately establish the identity of the duck.

#### 5. Conclusion

This paper uncovers the potential of semantic point correspondence in aligning motion trajectories and altering the subject’s identity in video editing. From there, we present VideoSwap, a framework that minimizes human intervention while utilizing semantic point correspondences for customized video subject swapping. Through user-point interactions like point removal or dragging, we address various semantic point correspondence. VideoSwap facilitates shape changes in the target swap while aligning the motion trajectory with the source subject, demonstrating state-ofthe-art results in customized video subject swapping.

#### References

[1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision

and Pattern Recognition (CVPR), pages 18208–18218, 2022. 5

- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM Trans. Graph., 42(4), 2023. 5
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 3, 5, 1
- [4] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7291–7299, 2017. 6
- [5] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. arXiv preprint arXiv:2308.09592, 2023. 2, 3, 4, 8
- [6] Jooyoung Choi, Yunjey Choi, Yunji Kim, Junho Kim, and Sungroh Yoon. Custom-edit: Text-guided image editing with customized diffusion models. arXiv preprint

- arXiv:2305.15779, 2023. 4

[7] Paul Couairon, Cl´ement Rambour, Jean-Emmanuel Haugeard, and Nicolas Thome. Videdit: Zero-shot and spatially aware text-driven video editing. arXiv preprint

- arXiv:2306.08707, 2023. 3

- [8] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 5
- [9] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adri`a Recasens, Lucas Smaira, Yusuf Aytar, Jo˜ao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems, 35:13610–13626, 2022. 3
- [10] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. arXiv preprint arXiv:2306.08637,

2023. 3

- [11] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 2, 3, 4
- [12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 4
- [13] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 3, 4, 8
- [14] Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, et al. Photoswap: Personalized subject swapping in images. arXiv preprint arXiv:2305.18286, 2023. 4
- [15] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning

- Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. arXiv preprint arXiv:2305.18292, 2023. 2, 4, 5
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 3, 5, 8, 1
- [17] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 1
- [18] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: a reference-free evaluation metric for image captioning. In EMNLP, 2021. 2
- [19] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [20] Jiahui Huang, Leonid Sigal, Kwang Moo Yi, Oliver Wang, and Joon-Young Lee. Inve: Interactive neural video editing. arXiv preprint arXiv:2307.07663, 2023. 6, 7, 1, 4
- [21] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 4
- [22] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. arXiv preprint arXiv:2307.07635, 2023. 3, 4, 6
- [23] Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG), 40(6):1–12, 2021. 3, 6, 7, 1, 4
- [24] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (ToG), 42(4):1–14, 2023. 4
- [25] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 2
- [26] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. arXiv preprint arXiv:2212.04488,

2022. 4, 2

- [27] Yao-Chih Lee, Ji-Ze Genevieve Jang, Yi-Ting Chen, Elizabeth Qiu, and Jia-Bin Huang. Shape-aware text-driven layered video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14317–14326, 2023. 3
- [28] Tianle Li, Max Ku, Cong Wei, and Wenhu Chen. Dreamedit: Subject-driven image editing. arXiv preprint arXiv:2306.12624, 2023. 4
- [29] Jun Hao Liew, Hanshu Yan, Jianfeng Zhang, Zhongcong Xu, and Jiashi Feng. Magicedit: High-fidelity and temporally

coherent video editing. arXiv preprint arXiv:2308.14749,

2023. 2, 3

- [30] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761, 2023. 3
- [31] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440, 2023. 2
- [32] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 4
- [33] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 3
- [34] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 2
- [35] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421,

2023. 3

- [36] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 4, 8, 1
- [37] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1–15, 2022. 4
- [38] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926, 2023. 3, 4
- [39] Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 3, 4
- [40] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732,

2016. 2

- [41] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318–10327, 2021. 3

- [42] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535, 2023. 2, 3, 8
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 5, 8, 1
- [44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. arXiv preprint arXiv:2208.12242, 2022. 4
- [45] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 4
- [46] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 4
- [47] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023. 4
- [48] Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv preprint arXiv:2306.14435, 2023. 3
- [49] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2

- [50] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 5
- [51] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023. 4
- [52] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. arXiv preprint arXiv:2306.03881,

2023. 3, 6

- [53] Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc N Tran, and Anh Tran. Anti-dreambooth: Protecting users from personalized text-to-image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2116–2127, 2023. 5
- [54] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 4
- [55] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 3
- [56] Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski, and Noah Snavely.

- Tracking everything everywhere all at once. arXiv preprint arXiv:2306.05422, 2023. 3
- [57] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 2, 3, 8
- [58] Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, et al. Cvpr 2023 text guided video editing competition. arXiv preprint arXiv:2310.16003, 2023. 2
- [59] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 4
- [60] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. arXiv preprint arXiv:2306.07954, 2023. 3, 4, 8, 2
- [61] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 4

- [62] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023. 3
- [63] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 1
- [64] Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. arXiv preprint arXiv:2305.17098,

2023. 2, 3

## VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence

### Supplementary Material

#### 6. Additional Details about Methods

##### 6.1. Latent Blend

Given our focus on subject swapping, where the objective is to maintain the unedited background region identical to the source video, this is achieved through latent blend [3, 16], as shown in Fig. 4.

The key idea is that the latent noise in DDIM denoising and DDIM inversion provides information for the swapped subject and background, respectively. These two latent noises can be blended using a mask that indicates the foreground region, thus blending the swapped target with the source background.

To initiate the process, we acquire the foreground mask for timestep t as Mt = Mti ∪ Mtd, formed by merging the subject masks Mi during inversion and Md during denoising at the same timestep t. This subject mask is automatically generated through the cross-attention of the concept token, following the approach of Prompt2Prompt [17].

Subsequently, the foreground mask is used to blend the latent features, resulting in zt = (1 − Mt) · zit + Mt · zdt, where zit and zdt represent the latent features of timestep t in DDIM inversion and DDIM denoising, respectively. Through latent blend, we can effectively preserve the unedited background in the source video.

##### 6.2. Layered Neural Atlas Training

As mentioned in Sec. 3.5 of the main paper, we introduce interactive dragging on the key frame for handling point correspondence with shape morphing in customized video subject swapping. This function is supported by the learned canonical space of Layered Neural Atlas [23] (LNA). Here, we present a detailed formulation of LNA.

LNA [23] represents a video through the following three sets of parameterized MLPs:

- 1. Coordinate Mapping MLPs. The coordinate mapping MLPs map the spatial-temporal coordinates of video pixels to the 2D canonical space (i.e., the UV map), denoted as M: (x,y,f) → (u,v). We employ separate

mappings, Ms and Mb, for the foreground subject and background, respectively. Additionally, following the approach of INVE [20], we include a background mapping Bs: (u,v) → (x,y,f) to learn the coordinate mapping of the foreground subject from the canonical space back to the video pixel.

- 2. Atlas MLPs. The atlas MLPs, denoted as A: (u,v) → (r,g,b), learn to predict the color of the coordinates on

the UV map.

3. Alpha MLPs. The alpha MLPs, denoted as Mα: (x,y,f) → α, predict the blending ratio α of the color value from the subject atlas and background atlas.

Based on these sets of learnable MLPs, the training objective of LNA is to reconstruct the RGB values of the source video, accompanied by the following regularization losses:

- 1. Rigidity Loss. The rigidity loss encourages the learned mapping from pixel coordinates in the video to the 2D canonical space to exhibit local rigidity.
- 2. Consistency Loss. The consistency loss encourages the mapping of corresponding video pixels across consecutive frames to be consistent, with correspondence estimated through pre-computed optical flow.
- 3. Sparsity Loss. The sparsity loss encourages the manyto-one mapping from the video coordinates to the canonical coordinates, penalizing duplicate contents in the canonical space.

We refer the reader to the LNA [23] paper for the complete formulation.

##### 6.3. Discussion the Relation to Human Keypoint

The ControlNet [63] and T2I-Adapter [36] also incorporate control over human keypoints. These human keypoints can be viewed as a type of sparse semantic points, where the semantic position and total number of human keypoints are predefined by the existing pose detectors, and their semantic embedding for controlling the diffusion model is implicitly aligned through large-scale paired data. However, defining keypoints or collecting paired data for open-set concepts proves challenging due to the variability in semantic points. Therefore, our method provides a more generic framework for point-based video editing, with human keypoints serving as a specific use case within our framework.

#### 7. Experimental Details 7.1. Implementation Details

We implement our method using the Latent Diffusion Model [43] and incorporate the pretrained motion layer from AnimateDiff [16] as the foundational model. All experiments are conducted on an Nvidia A100 (40GB) GPU. All video samples consist of 16 frames with a time stride of 4, matching the temporal window of the motion layer in AnimateDiff. We crop the videos to two alternate resolutions (H × W): 512 × 512 and 448 × 768. For all experiments, we employ the Adam optimizer with a learning rate of 5e-

5, optimizing for 100 iterations. Regarding the point patch loss, we use a patch size of 4×4 around the semantic point.

##### 7.2. Time Cost Analysis

In this section, we analyze the time cost of editing a video in VideoSwap. All time costs are calculated on an Nvidia A100 GPU to process a 16 frame video clip.

Time Cost of Preprocess. The preprocessing step involves (1) extracting point trajectories and their DIFT embeddings, and (2) registering those semantic points to guide the diffusion model, and (3) generate DDIM-inverted noise. The extraction of trajectories and embeddings takes approximately 30 seconds. The registration step requires 100 iterations, taking about 3 minutes. And the DDIM inversion of 50 steps takes approximately 30 seconds. To summarize, it takes about 4 minutes to pre-process a video for editing.

Time Cost of Each Edit. Then for each edit, the time cost of VideoSwap remain the similar to AnimateDiff [16], necessitating 50 seconds with the latent blend technique. The introduction of semantic point correspondence does not notably increase the time cost, given its lightweight computation.

Time Cost of User-Point Interaction. The time cost for user-point interaction (e.g., removing or dragging a point) can be negligible. Dragging a point at the keyframe only takes 1 seconds to propagate to all other frames through a learned layered neural altas (LNA).

Extra Time Cost in Training LNA. Our support for dragbased editing is built upon a learned LNA of the given video. In contrast to the original LNA, which necessitates approximately 10 hours of training, we do not require full training as we only adopt the forward/backward coordinate mapping. This training process takes about 2 hours for a video.

7.3. Memory Cost Analysis

The overall memory cost is similar to AnimateDiff, where we don’t incur significant additional memory costs, as our semantic points and MLPs are lightweight. It only requires a memory cost of 16/12 GB for point registration and inference, respectively.

8. Quantitative Evaluation

##### 8.1. Dataset and Evaluation Setting

We collect 30 videos from Shutterstock and DAVIS [40]. Each category—human, animal, and object—comprises 10 videos. Besides, we gather 13 customized concepts: 5 for human characters, 3 for animals, and 5 for objects. Due to legal concerns, we cannot demonstrate qualitative results involving human characters. For each source video, we adopt 8 predefined concepts and 2-5 customized concepts as swap

Temporal Consistency (↑) Compare to Previous Video Editing Methods

Text Alignment (↑)

Image Alignment (↑)

Methods/Metrics

Tune-A-Video [57] 25.34 - 95.79 FateZero [42] 24.39 - 95.49 Text2Video-Zero [25] 24.85 - 95.02 Rerender-A-Video [60] 24.99 - 92.28 VideoSwap (Ours) 26.87 - 95.93

###### Compare to Baselines on AnimateDiff

w/ DDIM 27.36 79.79 95.89 w/ DDIM + TAV 24.75 75.93 95.49 w/ DDIM + T2I-Adapter 25.86 77.54 95.50 VideoSwap (Ours) 26.87 79.87 95.93

Table 2. Automatic Quantitative Evaluation on Video Subject Swapping Results.

targets, yielding approximately 300 edited results. For comparison to previous video-editing methods that don’t support customized concepts, we only compute the metric on predefined concepts. In comparison to the baselines built upon AnimateDiff [16], we compute the metric on both predefined concepts and customized concepts.

##### 8.2. Automatic Evaluation by CLIP-Score

We conduct a quantitative evaluation using the automatic metric, CLIP-Score [18]. The metric includes text alignment and temporal consistency, following [58]. Additionally, for customized concepts, we follow Custom Diffusion [26] to compute pairwise image alignment between each edited frame and each reference concept image. The results are summarized in Table. 2. In comparison to previous video editing methods, VideoSwap demonstrates the best text alignment and temporal consistency. Moreover, when compared to baselines built on AnimateDiff, we achieve superior image alignment and temporal consistency. However, it is important to note that CLIP-Score is primarily based on frame-wise computation and may not align well with human perception, as discussed in EvalCrafter [31]. Therefore, we present these results for reference purposes and primarily evaluate and compare using human evaluation.

##### 8.3. Human Evaluation Interface

We primarily conduct human evaluations to compare different methods based on several criteria: subject identity, motion alignment, temporal consistency, and overall swapping preference. As depicted in Fig. 9, we present the source video and reference images for the target concept in the interface and ask users to select their preferred video based on various criteria related to customized video subject swapping. We distribute 1000 questionnaires on Amazon Mturk. The human evaluation results in Table. 1 of the main paper clearly demonstrate our advantage.

[Figure 28]

[Figure 29]

[Figure 30]

- Figure 9. Human evaluation interface on Amazon Mturk. We provide the source video and reference images for target concept and ask user to select favorable video in terms of different criteria of video subject swapping.

|Methods/Metrics<br><br>| |Subject Identity<br><br>Motion Alignment<br><br>Temporal Consistency<br><br>Overall Preference|
|---|---|---|
|VideoSwapv.s.<br><br>|Ablation of Sparse Motion Feature 87% 90% 87% 90%<br><br>| |
| |Point Map + T2I-Adapter (100 iters) Learnable Embedding + MLP (100 iters) Learnable Embedding + MLP (300 iters)|v.s. 13% v.s. 10% v.s. 13% v.s. 10% 87% v.s. 13% 95% v.s. 5% 90% v.s. 10% 90% v.s. 10% 52% v.s. 48% 52% v.s. 48% 52% v.s. 48% 55% v.s. 45%<br><br>|
| |Ablation of Point Patch Loss| |
| |w/o. Point Patch Loss<br><br>|73% v.s. 27% 73% v.s. 27% 78% v.s. 22% 78% v.s. 22%|
| |Ablation of Semantic-Enhanced Schedule| |
| |w/o. Semantic-Enhanced Schedule<br><br>|85% v.s. 15% 90% v.s. 10% 90% v.s. 10% 87% v.s. 13%|

Table 3. Human Evaluation for Ablation Study in VideoSwap. VideoSwap utilizes DIFT embedding + MLP (100 iterations) and incorporates the point patch loss and a semantic-enhanced schedule to improve the learning of semantic point correspondence.

##### 8.4. Human Evaluation for Ablation Study

We employ human evaluation to quantitatively assess various variants of our methods, and the results are summarized in Table. 3. In terms of creating sparse motion features, our DIFT embedding significantly outperforms point map + T2I-Adapter and the learnable embedding + MLP with the same registration iterations. In comparison to the learnable embedding and MLP, our explicit DIFT embedding already contains sufficient semantic information, requiring 3× less time to achieve similar preference. The introduction of the

point patch loss and semantic-enhanced schedule further enhances VideoSwap, leading to higher preferences compared to variants without these enhancements.

#### 9. Qualitative Evaluation

All our qualitative results and analysis are presented in the https : / / videoswap . github . io / supplementary/videoswap_supp.html. Please visit this webpage for more details.

[Figure 31]

[Figure 32]

1-st frame (keyframe) 16-th frame 32-th frame 48-th frame

64-th frame

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

###### (a) Tracking Error with Self-Occlusion

1-st frame 16-th frame 32-th frame (keyframe) 48-th frame

64-th frame

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

| |
|---|

[Figure 44]

| |
|---|

###### (b) Tracking Error with View-Change

- Figure 10. Limitations in point tracking inherited from Co-Tracker [22] in scenarios involving self-occlusion and significant view changes.

#### 10. Limitation and Future Works

##### 10.2. Future Works

##### 10.1. Limitation Analysis

The limitation of VideoSwap is inherited from inaccurate point tracking and an imperfect canonical space representation of Layered Neural Atlas.

Inaccurate Point Tracking by Co-Tracker. VideoSwap relies on accurate point trajectory extraction. However, the existing point tracking method Co-Tracker [22] is not stable enough when the video contains self-occlusion and large view changes, as shown in Fig. 10(a) and Fig. 10(b). To address this issue, users may choose to remove inaccurate semantic points; however, this would result in less motion alignment. Nevertheless, since tracking any point is a newly formed problem, any progress in this area can seamlessly integrate into VideoSwap.

Imperfect Canonical Space by Layer Neural Atlas. As discussed in Layered Neural Atlas (LNA) [23], LNA fails to represent videos involving 3D rotations and non-rigid motion with self-occlusion. VideoSwap resorts to LNA to propagate the dragged point displacement. Therefore, due to the limitations of LNA, we cannot support drag-based interaction in such cases. Improvement in LNA representation will further broaden support for drag-based video editing.

Time Cost for Interactive Editing. The time cost of VideoSwap prohibits its use for real-time interactive editing. Setting up semantic points for a video takes approximately 4 minutes. And to support drag-based editing, an additional 2 hours are required to prepare the LNA for the given video. Furthermore, constrained by diffusion model sampling, it takes about 50 seconds to perform an edit, falling short of real-time editing. We anticipate that advancements in neural field acceleration [20, 24, 37] and diffusion model distillation [32, 46, 51] will significantly reduce the preprocess cost and enhance speed for real-time interactive editing.

VideoSwap embarks on video editing with shape change. With semantic points as correspondence, VideoSwap can support interactive editing for large shape changes while aligning motion trajectories. We list several promising directions motivated by VideoSwap.

Interactive Video Editing. VideoSwap supports dragbased interaction at the keyframe, propagating the dragged displacement throughout the entire video and obtaining the source and dragged trajectories with similar motion. As we can obtain the source point trajectory and target point trajectory, future work may extend the idea of DragGAN [39] to the video domain for drag-based real video editing.

Video Editing with Shape Change. VideoSwap has demonstrated promising results in swapping the subject in the source video with a target concept that may have a different shape. In our paper, we focus on the swapping foreground subject, without considering background swapping or stylization. Further research could delve into a more general framework for video editing involving shape changes, thereby enhancing the flexibility of the video editing.

Application Based on Customized Video Editing. VideoSwap has shown promising results in swapping the subject in the source video with a target concept with customized identity. Future work may further investigate its application in movie generation and storytelling by fixing subjects’ identities.

##### 10.3. Potential Negative Social Impact

This project aims to provide the community with an effective method to swap their customized concept into the video. However, a risk exists wherein malicious entities could exploit this framework to create deceptive video with real-world figures, potentially misleading the public. This concern is not unique to our approach but rather a shared consideration in other concept customization methodologies. One potential solution to mitigate such risks involves

adopting methods similar to anti-dreambooth [53], which introduce subtle noise perturbations to the published images to mislead the customization process. Additionally, applying unseen watermarking to the generated video could deter misuse and prevent them from being used without proper recognition.

