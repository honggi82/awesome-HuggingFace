# arXiv:2604.28130v3[cs.CV]19Jun2026

## MoCapAnything V2: End-to-End Motion Capture for Arbitrary Skeletons

KEHONG GONG∗, Huawei Technologies Co., Ltd, Singapore ZHENGYU WEN∗, Huawei Central Research Institute, Singapore DAO THIEN PHONG∗, Huawei Central Research Institute, Singapore MINGXI XU, Huawei Central Research Institute, Singapore WEIXIA HE, Huawei Central Research Institute, Singapore QI WANG, Huawei Central Research Institute, Singapore NING ZHANG, Huawei Central Research Institute, Singapore ZHENGYU LI, Huawei Central Research Institute, Singapore GUANLI HOU, Huawei Central Research Institute, Singapore DONGZE LIAN, Huawei Central Research Institute, Singapore XIAOYU HE, Huawei Central Research Institute, Singapore MINGYUAN ZHANG†, Huawei Central Research Institute, Singapore HANWANG ZHANG, Huawei Central Research Institute, Singapore

[Figure 1]

Fig. 1. Overview of MoCapAnything V2. Given an input video of a human or an animal, our method infers a topology-agnostic skeleton sequence across diverse skeleton topologies. Conditioned on a reference asset, the model predicts animation-ready rotations via an end-to-end framework, enabling the reference asset to perform the input motion.

Recent methods for arbitrary-skeleton motion capture from monocular video [Gong et al. 2026] follow a factorized pipeline, where a Video-toPose network predicts joint positions and an analytical inverse-kinematics (IK) stage recovers joint rotations. While effective, this design is inherently limited, since joint positions do not fully determine rotations and leave degrees of freedom such as bone-axis twist ambiguous, and meanwhile the

∗Equal contribution. †Corresponding author.

non-differentiable IK stage prevents the system from adapting to noisy predictions or optimizing for the final animation objective. In this work, we instead present the first fully end-to-end framework, where both Video-toPose and Pose-to-Rotation are learnable and jointly optimized. Crucially, we observe that the ambiguity in pose-to-rotation mapping arises from missing coordinate-system information, as the same joint positions can correspond to different rotations under different rest poses and local axis conventions, and to resolve this we introduce a reference pose–rotation pair from the target asset, which together with the rest pose not only anchors the mapping but also

defines the underlying rotation coordinate system. This turns rotation prediction into a well-constrained conditional problem and enables effective learning. In addition, our model predicts joint positions directly from video without relying on mesh intermediates, which improves both robustness and efficiency, and both stages share a skeleton-aware Global-Local Graph-guided Multi-Head Attention (GL-GMHA) module for joint-level local reasoning and global coordination. Experiments on Truebones Zoo and Objaverse show that our method reduces rotation error from ∼17◦ to ∼10◦, and to 6.54◦ on unseen skeletons, while achieving ∼20× faster inference than mesh-based pipelines. Project page: https://animotionlab.github.io/MoCapAnythingV2/.

Additional Key Words and Phrases: Motion capture, arbitrary skeleton, monocular video, end-to-end learning, inverse kinematics, character animation, cross-skeleton generalization

1 Introduction

Recovering 3D character animation from monocular video, commonly referred to as motion capture, is a long-standing problem at the intersection of computer vision and computer graphics. While recent deep learning approaches have achieved strong performance for human motion capture [Goel et al. 2023; Kanazawa et al. 2018a; Kocabas et al. 2020; Pavllo et al. 2019], extending this capability to the arbitrary-skeleton setting remains fundamentally challenging. In this setting, the goal is to recover motion directly in the parameter space of a target rigged asset given only a monocular video and a reference skeleton [Gong et al. 2026]. The difficulty arises from the large variation in skeleton topology, joint count, and especially local coordinate conventions across different assets.

A central challenge lies in recovering joint rotations. Unlike joint positions, rotations are defined with respect to a skeleton-specific coordinate system determined by the rest pose and local axis conventions. As a result, the same joint positions can correspond to different rotations under different skeletons, making direct videoto-rotation prediction prone to overfitting and poor generalization.

To mitigate this issue, existing methods adopt a factorized design. A learned Video-to-Pose (V→P) network first predicts 3D joint positions, and an analytical inverse-kinematics (IK) solver then converts these positions into joint rotations. This design leverages the fact that joint positions are largely shared across skeletons performing the same motion, thereby improving generalization at the pose level. However, this factorization introduces a fundamental limitation. Since joint positions do not fully determine rotations, the pose-to-rotation mapping is inherently under-constrained, and analytical solvers cannot resolve degrees of freedom such as boneaxis twist, nor can they adapt to the noise distribution of predicted poses. Moreover, the non-differentiable IK stage prevents the two components from being jointly optimized, forcing the pose predictor to ignore the downstream rotation objective.

In this work, we present the first fully end-to-end framework for arbitrary-skeleton motion capture from monocular video, in which both the Video-to-Pose and Pose-to-Rotation (P→R) stages are learnable and jointly optimized. The key challenge lies in the ill-posed nature of the P→R mapping: the same 3D joint positions can correspond to different rotation values under different rest poses and local coordinate frames, and therefore pose alone is insufficient to determine rotations, as it does not specify how the target skeleton defines its rotation coordinate system.

To address this issue, we extend the reference signal. While prior work relies on static geometry (skeleton, mesh, and rendered views) [Gong et al. 2026], we additionally introduce a single reference pose–rotation pair from the target asset, which is naturally available for any rigged skeleton. Together with the rest pose, this reference acts as an explicit coordinate-system anchor, turning the ill-posed P→R mapping into a well-constrained conditional prediction problem that can be modeled by a neural decoder.

This design enables the P→R stage to be learned, and more importantly, unlocks end-to-end training of the entire pipeline. Gradients from the rotation objective can now flow back through the pose intermediate and into the visual encoder, so that the pose representation is no longer optimized solely for positional accuracy, but is adaptively reshaped into a representation that better supports rotation recovery.

In addition, we remove the mesh intermediate used in prior work [Gong et al. 2026]. While mesh representations can be beneficial when accurate, predicted mesh sequences often introduce noise and lead to error accumulation. By directly predicting joint positions from video, our approach improves robustness and significantly reduces computational cost, achieving approximately 20× faster inference.

To support both stages, we introduce a skeleton-aware attention mechanism, Global-Local Graph-guided Multi-Head Attention (GL-GMHA), which alternates between local kinematic-chain reasoning and global cross-branch coordination. This design enables effective modeling of both structural constraints and global motion dependencies across diverse skeletons.

We evaluate our method on the Truebones Zoo dataset, which covers seen, rare, and unseen skeletons, as well as the Objaverse benchmark. Our approach significantly improves rotation accuracy across all settings, especially on unseen skeletons, while also providing substantial gains in efficiency.

Our main contributions are as follows:

- • End-to-endlearnableVideo-to-Pose-to-Rotation framework.We presentthefirstfullyend-to-end trainable pipeline for arbitrary-skeleton motion capture, enabling joint optimization of pose and rotation.
- • Reference-conditioned rotation modeling. We introduce a reference pose–rotation pair that, together with the rest pose, defines the rotation coordinate system and resolves the ambiguity of pose-to-rotation mapping.
- • Efficient direct video-to-pose prediction. We eliminate the mesh intermediate, improving robustness and achieving approximately 20× faster inference.
- • Skeleton-aware attention mechanism. We propose GLGMHA, which integrates local kinematic reasoning with global skeleton coordination in a unified attention framework.

2 Related Work

Recovering animation-ready motion from monocular video requires bridging the gap between observable visual cues (e.g., joint positions) and skeleton-specific motion parameters (e.g., rotations). Existing approaches address different aspects of this problem, but none fully

[Figure 2]

- Fig. 2. Comparison of MoCapAnything V1 and V2. Unlike V1, which depends on mesh-conditioned video-to-pose estimation and analytical inverse kinematics (IK) for rotation recovery, V2 eliminates mesh conditioning and introduces a fully learnable Pose2Rot module. The entire pipeline is optimized end-to-end, enabling bidirectional coupling between pose and rotation for improved robustness and animation-ready motion synthesis.

resolve the ambiguity of mapping pose to rotation under varying skeleton definitions.

Pose estimation. Pose estimation methods localize anatomical keypoints from images or videos, ranging from heatmap-based architectures [Sun et al. 2019] and DETR-style end-to-end models [Carion et al. 2020; Shi et al. 2022; Xiao et al. 2022; Xu et al. 2022b] to categoryagnostic frameworks [Hirschorn and Avidan 2024; Rusanovsky et al. 2025; Shi et al. 2023; Xu et al. 2022a] that generalize to unseen objects via support–query matching. However, these methods operate in 2D or in a fixed keypoint space and produce neither 3D motion trajectories nor skeleton-specific rotation parameters; in particular, they do not address how a joint configuration should be interpreted under different skeleton coordinate systems.

Motion capture. Monocular motion capture has been extensively studied for parametric human models [Loper et al. 2015; Pavlakos et al. 2019], with feed-forward networks [Kocabas et al. 2020] and transformer-based variants regressing pose and shape parameters directly from video. Beyond humans, model-free reconstruction methods [Kanazawa et al. 2018b; Li et al. 2024; Wu et al. 2023] and video-based extensions [Yang et al. 2021, 2022] recover deformable surfaces, while category-specific parametric models such as SMAL [Zuffi et al. 2017] target animals. These approaches either operate in a fixed parameter space tied to predefined skeletons, or lack explicit skeletal parameterization, and therefore do not generalize to the arbitrary skeletons required for animation. The closest line of work, MoCapAnything [Gong et al. 2026], conditions on a reference asset and adopts a factorized design: a learned Video-to-Pose network followed by an analytical IK solver. While pose as a shared intermediate improves cross-skeleton generalization, the mapping from positions to rotations is inherently under-constrained, since the same pose can correspond to different rotations under different rest poses and local coordinate systems, and the non-differentiable IK stage prevents end-to-end optimization. In contrast, we resolve

the P→R ambiguity with a reference pose–rotation pair that defines the underlying coordinate system, casting P→R as a learnable conditional problem and training the entire V→P→R pipeline endto-end.

3 Method

- 3.1 Problem Formulation

Given an input video V = {𝐼1, . . .,𝐼𝑇 } and a target skeleton S defined by parent indices 𝝅, bone offsets o ∈ R𝐽 ×3 (the rest pose), and per-joint semantic labels, we seek per-frame joint rotations R = {r1, . . ., r𝑇 } in 6D representation [Zhou et al. 2019], r𝑡 ∈ R𝐽 ×6, that drive S to perform the observed motion. S is assumed to be a treestructured rigged skeleton with a single root; joint count 𝐽 and topology are otherwise arbitrary, padded up to 150 to accommodate our largest training skeleton (143 joints).

To improve modeling stability and cross-skeleton generalization, we decompose the problem into two stages mediated by an explicit intermediate representation:

Video −−−−−−→Stage 1 Pose −−−−−−→Stage 2 Rotation, (1)

where pose P = {p𝑡}, p𝑡 ∈ R𝐽 ×3, denotes root-relative 3D joint positions in the camera coordinate frame, and rotation denotes local joint rotations in S’s coordinate system.

- 3.2 Overview

Unlike prior arbitrary-skeleton methods [Gong et al. 2026], which adopt a factorized design, a learned Video-to-Pose stage followed by a non-differentiable analytical IK stage for Pose-to-Rotation (See Fig. 2), our framework casts both stages as learnable neural modules and trains them jointly end-to-end (see Fig. 3). This unified design is what makes joint optimization of the full V→P→R pipeline possible; its technical enabler is a reference pose–rotation pair that resolves the ill-posedness of a learnable P→R mapping (§3.5).

Concretely, our framework comprises two core modules:

[Figure 3]

- Fig. 3. Framework of MoCapAnything V2. Our method unifies video-to-pose and pose-to-rotation within a single end-to-end trainable architecture. The video-to-pose stage consists of a reference-conditioned pose prompt encoder (A), which encodes skeleton and image cues into joint prompt, and a unified pose decoder (B), which predicts temporally coherent joint positions via cross-attention with video features. The pose-to-rotation stage is formulated as a learnable inverse kinematics module, composed of a rotation prompt encoder (C) that maps predicted poses into rot prompt, an anchor encoder (D) that encodes reference pose–rotation pairs to establish a consistent rotation coordinate space, and a unified rotation decoder (E) that generates animation-ready joint rotations conditioned on the anchor through cross-attention.

Video-to-Pose module (§3.4) predicts a sequence of joint positions from the input video, conditioned on a reference frame that establishes the skeleton-specific joint layout.

Pose-to-Rotation module (§3.5) maps the predicted joint positions to per-joint rotations, conditioned on the target skeleton’s rest pose and a reference pose–rotation pair that anchors the local coordinate system.

Both modules are conditioned on the same reference frame, providing a consistent anchor throughout the pipeline: during training, this frame is randomly sampled from the ground-truth sequence, while at test time it is naturally provided together with the target skeleton asset (e.g., one frame from the rigged animation supplied with the asset).

- 3.3 Global-Local Graph-Guided Multi-Head Attention

We adopt a topology-aware attention mechanism shared by both the Video-to-Pose and Pose-to-Rotation modules. Building on Graphguided Multi-Head Attention (GMHA) [Gat et al. 2025], which uses graph-derived joint relations (distance and kinematic connectivity)

- as attention bias, we introduce a Global-Local variant (GL-GMHA) that alternates between local layers restricted to kinematic chains (capturing intra-limb dependencies) and global layers with full connectivity (capturing cross-limb coordination). This complementary

design models both structural constraints and whole-body dynamics without additional parameters and naturally generalizes across diverse skeleton topologies.

3.4 Video-to-Pose Module

This module predicts joint positions directly from the input video, without explicit geometric intermediates such as mesh or surface normals: predicted-mesh noise propagates through downstream stages and degrades stability (§4.3).

Reference query encoder. To ground the joint layout in the target skeleton, we encode a reference frame for which joint positions and image features are both known. Reference joint positions pref ∈ R𝐽 ×3 are mapped through a frequency-based positional embedding [Mildenhall et al. 2021] and projected to dimension 𝑑. Perjoint semantic embeddings, obtained by encoding joint names with the T5 [Raffel et al. 2020] text encoder, are added to provide categoryagnostic joint identity that generalizes across arbitrary joint counts and naming conventions. Reference image features zref ∈ R𝑃×𝑑img (from a frozen DINOv2 [Oquab et al. 2023] encoder) are fused via a stack of RefFusionBlocks, each comprising GL-GMHA self-attention over joints, vanilla self-attention, and cross-attention to image features. The output is a set of reference joint queries Qref ∈ R𝐽 ×𝑑 that encode both skeletal structure and reference appearance.

Temporal pose decoder. Given Qref and per-frame image features z𝑡 (same frozen DINOv2 encoder), a temporal transformer uses GL-GMHA for spatial reasoning across joints and windowed perjoint temporal attention with RoPE [Su et al. 2024] across frames, yielding Pˆ = {pˆ𝑡} ∈ R𝑇×𝐽 ×3. Joints flagged as position-static during preprocessing are overwritten with the reference position to ensure structural consistency; an analogous rotation-static flag is handled in §3.5.3.

We treat joint positions as a skeleton-shared canonical representation: different skeletons performing the same motion share similar position patterns, decoupling skeleton-specific conventions from motion content (§4.6). Because P→R is also learnable (§3.5), this representation is co-adapted during end-to-end training (§3.6) toward the final rotation objective rather than frozen by a positionalaccuracyloss. Suchco-adaptationis unattainable in factorized pipelines with non-differentiable IK.

- 3.5 Reference-Conditioned Pose-to-Rotation Module

- 3.5.1 The Ill-Posedness of Pose-to-Rotation Mapping. Joint rotations are always defined relative to a skeleton’s rest pose and a choice of local coordinate frames. Geometrically, the rest pose fixes the origin of each joint’s local frame, but not its axes: the same positional trajectory can be expressed by many different rotation sequences under different axis conventions, and bone-axis twist is unconstrained by positions altogether. Conditioning on rest pose alone, the static-geometry signal used in prior work [Gong et al. 2026], therefore supplies only the origin, leaving the learning target R = 𝑓 (P, o) multi-valued and difficult to learn directly. To make P→R learnable, we need an explicit axis anchor; we show below that a single reference pose–rotation pair from the target asset provides exactly that.
- 3.5.2 Reference-ConditionedModeling. Weextendthestatic-geometry reference signal of prior work [Gong et al. 2026] with a single pose– rotation pair sampled from the same asset. Intuitively, rest pose supplies the coordinate origin while the reference pose–rotation pair supplies the coordinate axes: together they fully specify the local frame convention, telling the model “for this skeleton’s coordinate definition, this joint configuration corresponds to these rotations,” which converts the multi-valued P→R mapping into a well-constrained conditional prediction task. We use a single reference pair by default; the effect of additional pairs is analyzed in our ablation study (§4.5). Concretely, the rotation prediction is conditioned on three encoders, all built from GL-GMHA layers: (i) a Rest Pose Encoder takes bone offsets o ∈ R𝐽 ×3 and per-joint semantic embeddings, producing a rest-pose feature Erest ∈ R𝐽 ×𝑑 that captures static geometry and topology; (ii) a Reference Encoder jointly embeds the reference position pref and 6D rotation rref, modulated by Erest via FiLM [Perez et al. 2018], producing the coordinate-system anchor Cref ∈ R𝐽 ×𝑑; (iii) a Pose Encoder processes the predicted (or ground-truth) pose sequence P ∈ R𝑇×𝐽 ×3 via alternating GLGMHA and per-joint windowed temporal attention with RoPE [Su et al. 2024], optionally modulated by Erest, yielding a pose feature sequence Q ∈ R𝑇×𝐽 ×𝑑.

- 3.5.3 Rotation Decoder. The rotation decoder predicts per-frame, per-joint 6D rotations from the pose features Q. Each of its 𝐿 blocks applies, in order: FiLM modulation by Erest for skeleton-specific conditioning, per-joint temporal self-attention (windowed with RoPE) for temporal coherence, GL-GMHA spatial attention (alternating local/global masking) for cross-joint reasoning, per-joint cross-attention to the reference anchor Cref (applied in the first 𝐿cross ≤ 𝐿 layers; the rest rely on already-integrated reference information), and a feed-forward residual; the final layer projects to 6D rotation via a two-layer MLP. Joints flagged as rotation-static (determined independently from the position-static flag in §3.4) are overwritten with the reference rotation, mirroring the static-joint handling in the pose stage. By conditioning jointly on the rest pose, the reference pair, and the input pose, the decoder turns the ill-posed rotation recovery problem into a well-constrained conditional prediction task and learns statistical motion priors that resolve degrees of freedom (e.g., twist) inaccessible to analytical IK.
- 3.6 End-to-End Training Strategy

Joint optimization. With P→R cast as a learnable neural module (§3.5), end-to-end joint training of the entire V→P→R pipeline becomes possible for the first time in arbitrary-skeleton motion capture. We train the Video-to-Pose and Pose-to-Rotation modules jointly: gradients from the rotation loss flow back through the predicted pose and into the visual encoder, so the intermediate pose representation is reshaped not by a positional-accuracy objective alone, but by what best serves the final rotation objective. This is the key departure from factorized pipelines, whose non-differentiable IK handoff between V→P and P→R precludes any such co-adaptation; we quantify the effect in §4.4.

Loss function. The total loss combines four terms spanning both the pose and rotation stages:

L = 𝜆pos Lpos + 𝜆rot Lrot + 𝜆rot_v Lrot_v + 𝜆root Lroot, (2)

where Lpos is the per-joint position error between predicted and ground-truth joint positions. For the rotation stage, Lrot measures the geodesic angular error between predicted and ground-truth rotations averaged over all joints, and Lrot_v penalizes the angular velocity difference to promote temporal consistency in the rotation sequence. Lroot additionally re-weights the root joint’s rotation error, which we found accelerates convergence of the global orientation. All losses are computed with per-joint masking to handle the variable number of joints across different skeletons. We set 𝜆pos=𝜆rot=𝜆rot_v=1.0 and 𝜆root=0.1.

Mixed-pose training. A distribution gap exists between training, where the Pose-to-Rotation module can receive ground-truth poses, and inference, where it must operate on noisy predicted poses. To bridge this gap, we employ a mixed-pose training strategy that stochastically selects, for each sample in a batch, whether to feed ground-truth or predicted poses to the rotation module. The probability of using predicted poses follows a schedule:

𝑝pred(𝑒) = 𝑝start + (𝑝end − 𝑝start) · min 1,

𝑒 𝐸warmup

, (3)

where𝑒 is the current epoch and 𝐸warmup controls the transition rate. We set 𝑝start=0.1 and 𝑝end=1.0, so that early in training ground-truth poses dominate to ensure stable convergence, and the proportion of predicted poses increases gradually until the model is fully trained on its own pose predictions by the end of the warm-up phase. The choice of 𝐸warmup is studied in Appendix B.

- 4 Experiments

- 4.1 Dataset and Evaluation Protocol

Datasets. We evaluate on two benchmarks spanning diverse skeleton structures:

- • Truebones Zoo [Truebones nd]: 1,038 animal motion sequences (104,715 frames) covering a broad range of species and kinematic topologies. The test set (60 sequences) is stratified into Seen (species with abundant training data), Rare (species with limited training data), and Unseen (species never seen during training).
- • Objaverse: 1,000samplesfromObjaverse[Deitke et al.2023a,b], containing structurally distinct humanoid and non-animal targetsunseenduringtraining, serving as an out-of-distribution stress test.

We additionally evaluate on in-the-wild videos collected from the Internet to assess real-world robustness.

Evaluation metrics. We report four metrics spanning spatial accuracy and rotation quality: MPJPE (Mean Per Joint Position Error, cm), MPJVE (Mean Per Joint Velocity Error, cm), Ang. Err (geodesic angle error, ◦), and AngV Err (angular velocity error, ◦). To handle large inter-species scale variations, all samples are normalized to [−1, 1]3 for training and rescaled to a unified 1m3 cube for evaluation. All metrics are computed with per-joint masking to accommodate varying joint counts.

Baselines. We compare against HRNet [Sun et al. 2019], ViTPose [Xu et al. 2022b], VIBE [Kocabas et al. 2020], and GLoT [Shen et al. 2023]. For each baseline, we instantiate both the Video-to-Pose and Pose-to-Rotation modules with that method’s architecture and train them jointly end-to-end on the same training set with unified skeleton representations. Our most direct point of comparison is MoCapAnything V1 [Gong et al. 2026], which shares the same problem setting and training data, and differs only in that V1 adopts a factorized design: a learned Video-to-Pose stage with a 4D mesh intermediate, followed by a constraint-aware analytical IK stage for Pose-to-Rotation, with the two stages optimized independently.

Implementation details. We train the Video-to-Pose and Pose-toRotation modules jointly end-to-end with a frozen DINOv2 [Oquab et al. 2023] ViT visual backbone, sequence length 𝑇=48, an 8-layer rotation decoder with reference cross-attention in the first 6 layers, and the mixed-pose warm-up of §3.6; full hyperparameters and training schedule are reported in Appendix A.

- 4.2 Comparison with Baselines

Table 1 presents a comprehensive comparison across all four evaluation splits.

- Table 1. Main results on Zoo (Seen/Rare/Unseen) and Obj. Position in cm (↓); rotation in degrees (↓). All baselines are trained jointly end-to-end with learnable rotation modules; only V1 uses traditional IK. Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen Obj

Method JP JV An AV JP JV An AV JP JV An AV JP JV An AV HRNet 32.63 0.53 19.86 0.51 34.44 0.59 24.72 0.63 37.87 1.45 24.59 0.64 37.70 1.56 31.37 0.75 GLoT 19.66 0.60 20.24 0.52 21.60 0.70 26.13 0.65 26.13 1.47 25.95 0.66 22.95 1.70 29.07 0.69 ViTPose 19.77 0.59 20.90 0.52 21.12 0.68 25.48 0.63 26.17 1.45 24.46 0.63 24.16 1.85 29.30 0.70 VIBE 19.69 0.54 19.67 0.51 20.77 0.63 25.06 0.63 26.29 1.45 25.74 0.65 23.51 1.63 28.72 0.70 Ours 2.34 0.53 10.73 0.29 2.98 0.61 14.38 0.37 3.39 0.99 6.54 0.17 3.84 1.05 11.06 0.30

- Table 2. V1 vs. Ours under different mesh configurations. “GT Mesh” = ground-truth mesh; “Pred Mesh” = predicted mesh; “Ours” removes mesh entirely. For fair comparison with V1 (which was trained on Zoo only), all models in this table are both trained and evaluated on Zoo only; the Ours numbers therefore differ slightly from those in Table 1, where the model is trained on Zoo+Obj. Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen

Config JP JV An AV JP JV An AV JP JV An AV V1 (GT Mesh+IK) 1.06 0.44 17.47 2.18 1.28 0.37 18.52 2.08 1.76 0.36 20.56 3.10 V1 (Pred Mesh+IK) 3.30 0.67 20.02 2.51 4.19 0.55 19.82 2.18 4.78 0.93 22.04 3.37 Ours 2.20 0.53 10.91 0.30 2.86 0.59 14.36 0.37 3.73 0.98 6.68 0.18

Our method achieves consistently strong results across all splits, with the largest gains on rotation. Although the learned baselines receive the same reference inputs and adopt an end-to-end V→P→R formulation, their architectures do not effectively leverage reference and topology cues to resolve coordinate-axis ambiguity on arbitrary skeletons, cappingrotationquality near20◦ angleerror withartifacts such as joint spinning. Our reference-conditioned design halves this to ∼10◦ with substantially lower angular velocity error. The improvement is most pronounced on Zoo-Unseen, where the angle error (6.54◦) is actually lower than Zoo-Seen (10.73◦) and Zoo-Rare (14.38◦): this split is dominated by common locomotion motions (e.g., walking, running) whose rotations become straightforward once the coordinate axes are anchored by the reference pair. V1 is compared separately in §4.3.

4.3 End-to-End vs. Factorized Design: Comparison with V1

V1 [Gong et al. 2026] differs from our method along two simultaneous axes: (i) presence of a 4D mesh intermediate between video and pose, and (ii) a factorized learned-V→P + analytical-IK P→R design versus our end-to-end learnable V→P→R. We compare against V1 under different mesh configurations in Table 2 to disentangle the two effects.

V1 with ground-truth mesh achieves the lowest positional error, confirming geometry helps when accurate; but GT mesh is unavailable at inference. With predicted mesh, V1’s factorized design propagates noise through the mesh bridge, performing markedly worse than our mesh-free model. Our end-to-end V→P→R reaches pose accuracy competitive with V1 (GT Mesh) without any mesh, and substantially outperforms all V1 variants on rotation, reducing angle error from V1’s 17◦–22◦ range (with frequent joint spinning and limb flipping on twist-heavy joints; see Fig. 4) to ∼10◦ with much lower angular velocity. Two factors drive this gain, analyzed next: end-to-end joint training adapts the pose to the rotation objective

[Figure 4]

- Fig. 4. MoCap V1 vs. V2. Row 1: V1 (traditional IK-based optimization). Row 2: V2 (our learning-based rotation recovery). Each cell stacks three keyframes (diagonal offset, alpha fade) to convey motion. V1 suffers from joint spinning artifacts visible across the ghosted frames, whereas V2 produces stable, temporally consistent rotations.

Table 3. Ablation of training strategies on Zoo (Seen/Rare/Unseen) and Obj. Position in cm; rotation in degrees (↓). Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen Obj

Strategy JP JV An AV JP JV An AV JP JV An AV JP JV An AV Mixed (gradient detached) 2.05 0.43 11.67 0.32 2.43 0.50 14.80 0.38 2.91 0.89 7.82 0.21 3.96 1.01 11.96 0.29 GT pose only 2.10 0.43 12.68 0.35 2.71 0.55 14.74 0.39 3.39 0.86 13.28 0.38 3.59 1.01 16.39 0.41 Pred pose only 3.47 0.65 11.91 0.32 4.23 0.70 15.32 0.39 4.42 1.14 9.58 0.25 4.57 1.18 11.75 0.30 Mixed (with joint opt. ours) 2.34 0.53 10.73 0.29 2.98 0.61 14.38 0.37 3.39 0.99 6.54 0.17 3.84 1.05 11.06 0.30

Table 4. Ablation of reference conditioning (Ref) and rest pose (Rest). Position in cm; rotation in degrees (↓). Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen Obj

Ref Rest JP JV An AV JP JV An AV JP JV An AV JP JV An AV

- – – 2.23 0.54 11.25 0.31 2.97 0.61 14.83 0.40 3.07 0.98 24.26 0.64 4.22 1.10 18.00 0.46

- – ✓ 2.24 0.53 11.15 0.31 3.11 0.61 13.92 0.37 3.18 0.96 24.05 0.64 3.61 1.05 16.14 0.41

✓ – 2.39 0.56 11.31 0.31 2.98 0.59 14.95 0.39 3.22 0.98 7.37 0.19 3.71 1.08 11.02 0.27 ✓ ✓ 2.34 0.53 10.73 0.29 2.98 0.61 14.38 0.37 3.39 0.99 6.54 0.17 3.84 1.05 11.06 0.30

(§4.4), and the reference pose–rotation pair anchors the coordinate system to make learnable P→R feasible (§4.5). A stronger mesh predictor could narrow the positional gap but at significant overhead (§4.8).

4.4 Training Strategy Analysis: Validating the End-to-End Claim

This experiment directly tests the central claim of our framework: making P→R learnable is not sufficient on its own; its benefit arises from enabling end-to-end gradient coupling with V→P. We therefore compare four training regimes: a gradient-detached variant (where gradients from P→R do not flow back to V→P), end-to-end training with ground-truth poses only, end-to-end training with predicted poses only, and our default mixed-pose end-to-end training. Results are shown in Table 3.

Enabling gradient flow is critical: joint training reduces the ZooUnseen angle error from 7.82◦ (detached) to 6.54◦, showing the benefit of a learnable P→R module comes from end-to-end adaptation rather than standalone modeling. GT-only training suffers a distribution gap and fails on Zoo-Unseen (13.28◦), while Pred-only is unstable due to noisy early predictions; the mixed-pose schedule achieves the best balance (warm-up sensitivity in Appendix B).

- 4.5 Pose-to-Rotation Module Analysis

Reference pair and rest pose. Table 4 examines the contributions of the reference pose–rotation pair and the rest-pose encoding.

This empirically confirms the origin/axes decomposition of §3.5.1. On Zoo-Seen and Zoo-Rare, whose axis conventions are well represented in training, all four configurations reach comparable angle errors in the 10–15◦ range: the axis convention is effectively memorized, so the explicit anchor offers little extra gain. The picture changes dramatically on Zoo-Unseen: without the reference pair, error jumps to 24.05◦–24.26◦ (rest pose supplies only the coordinate origin, leaving axes ambiguous); adding the reference pair drops it to 7.37◦, and combining both yields the best 6.54◦. Rest pose thus provides a smaller but consistent benefit on top of the reference as complementary structural context.

- 4.6 Role of Pose as Intermediate Representation

To validate the necessity of explicit joint positions as an intermediate representation, we compare three architectural variants in Table 5.

Both Direct (V→R, 23.73◦) and Latent+Aux (23.57◦) match our Seen/Rare performance (9.32◦/12.71◦ and 9.06◦/11.85◦ respectively) but collapse on Zoo-Unseen, indicating that without an explicit pose intermediate, even when the latent pose is supervised, the model fails to generalize across unseen skeleton topologies. Our explicitpose intermediate substantially improves Zoo-Unseen (6.54◦) while

[Figure 5]

### Fig. 5. MoCap demo across domains. Row 1: Objaverse assets; Row 2: Truebones Zoo; Rows 3–4: in-the-wild videos. Each cell stacks three keyframes (diagonal offset, alpha fade) to convey motion across the sequence. Results are shown from multiple viewpoints, demonstrating accurate mocap on arbitrary subjects.

[Figure 6]

### Fig. 6. Dance demo A. A single input video (center) is mocapped and retargeted to a diverse set of humanoid and animal skeletons, all driven from the same source motion. Each cell stacks three keyframes (diagonal offset, alpha fade) to convey motion.

[Figure 7]

- Fig. 7. Dance demo B. A single input video (center) is mocapped and retargeted to a diverse set of humanoid and animal skeletons, all driven from the same source motion. Each cell stacks three keyframes (diagonal offset, alpha fade) to convey motion.

- Table 5. Ablation of the intermediate pose representation. Position in cm; rotation in degrees (↓). “Direct (V→R)” regresses rotations directly from video without a pose branch; joint positions for this variant are recovered by applying forward kinematics to the predicted rotations. “Latent + Aux” uses a latent intermediate supervised by an auxiliary pose loss. “Full” (Ours) uses an explicit joint-position intermediate as the first-stage output. Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen Obj

Architecture JP JV An AV JP JV An AV JP JV An AV JP JV An AV Direct (V→R) – – 9.32 0.26 – – 12.71 0.34 – – 23.73 0.62 – – 11.16 0.28 Latent + Aux 2.07 0.47 9.06 0.25 2.40 0.53 11.85 0.31 2.88 0.89 23.57 0.62 3.82 1.04 10.71 0.27 Full (explicit pose, Ours) 2.34 0.53 10.73 0.29 2.98 0.61 14.38 0.37 3.39 0.99 6.54 0.17 3.84 1.05 11.06 0.30

- Table 6. Ablation of attention mechanisms. Position in cm; rotation in degrees (↓). All variants use the same backbone; only the spatial attention pattern differs. “GMHA (all-global)” is the original uniform formulation from [Gat et al. 2025] in which every layer attends over all joints; “All-local” applies the ancestor mask at every layer; “GL-GMHA (Ours)” alternates between local and global layers. Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen Obj

Attention JP JV An AV JP JV An AV JP JV An AV JP JV An AV Full Attn (no graph bias) 2.83 0.48 12.57 0.33 3.57 0.57 16.11 0.41 3.98 1.01 11.92 0.31 4.64 1.07 12.02 0.30 GMHA (all-global) [Gat et al. 2025] 2.17 0.53 11.18 0.31 3.17 0.65 14.51 0.39 3.46 0.98 6.69 0.18 3.76 1.08 11.21 0.28 All-local (ancestor mask every layer) 2.47 0.55 12.55 0.33 3.09 0.65 16.97 0.43 3.39 1.02 11.60 0.31 4.28 1.10 13.59 0.34 GL-GMHA (Ours) 2.34 0.53 10.73 0.29 2.98 0.61 14.38 0.37 3.39 0.99 6.54 0.17 3.84 1.05 11.06 0.30

remaining competitive on Seen/Rare. This confirms that joint positions are skeleton-shared while rotations are skeleton-dependent: the explicit pose acts as a structural bottleneck that separates transferable motion patterns from skeleton-specific parameterization.

4.7 GL-GMHA Attention Analysis

- Table 6 evaluates the effectiveness of the Global-Local extension to graph-guided attention.

The “All-local” variant, which restricts attention to ancestor paths

- at every layer, performs worst overall: it captures intra-branch dependencies but loses cross-branch coordination, with angle error

jumping to 16.97◦ on Zoo-Rare and 11.60◦ on Zoo-Unseen. Removing graph biases entirely (“Full Attn”) trails uniform GMHA [Gat et al. 2025], indicating that the structural inductive bias on joint relationships matters; uniform GMHA attends globally in every layer but does not explicitly model kinematic-chain locality. Our alternating GL-GMHA consistently outperforms uniform GMHA across all four splits, validating that kinematic-chain locality and skeleton-global coordination are complementary rather than redundant.

- 4.8 Efficiency Analysis

Our efficiency gain comes from eliminating both the mesh prediction stage and the analytical IK solver. For a 120-frame input sequence, V1 takes over 20 minutes (feature extraction ∼40 s, mesh reconstruction ∼15min, pose estimation ∼20s, IK optimization ∼5min); our method shares the same feature-extraction cost but predicts pose and rotation in a single forward pass within ∼10s, totaling under 1 minute and corresponding to a ∼20× speedup. Removing the mesh avoids a heavy, error-prone reconstruction stage, while a learned rotation decoder replaces iterative per-frame IK with efficient batched computation. This speedup does not cost accuracy: averaged over Zoo-Seen/Rare/Unseen (Table 2), our method achieves 10.6◦ angle error vs. 18.9◦ for V1 (GT Mesh) and 20.63◦ (Pred Mesh).

- 4.9 Qualitative Results

Qualitative figures (Figs. 4, 5, 6, 7) are placed at the end of the paper for layout reasons. Each cell stacks three keyframes via diagonal offset and alpha fade to convey motion in a static figure; full animations are available on our project homepage.

Rotation quality: V1 vs. V2. Figure 4 compares our learning-based rotation recovery (V2) against the traditional IK-based optimization used in V1. V1 frequently exhibits joint spinning and limb

flipping, particularly on joints with large twist components, since per-frame IK lacks temporal context and reference priors. In contrast, V2 produces natural, temporally continuous rotations, confirming the effectiveness of reference-conditioned rotation modeling.

MoCap across Objaverse, Zoo, and in-the-wild. Figure 5 shows mocap results across three domains: Objaverse assets (row 1), Truebones Zoo (row 2), and in-the-wild Internet videos (rows 3–4), rendered from multiple viewpoints. Despite large variations in appearance, shape, and capture conditions, our method produces accurate and temporally consistent 3D motion, demonstrating that it generalizes to arbitrary subjects rather than being tied to a specific training distribution.

Unified mocap and cross-skeleton retargeting. A distinctive feature of our framework is that a single input video can simultaneously drive mocap and retargeting onto many different skeletons, without any skeleton-specific training. Figures 6 and 7 illustrate this on two dance clips: given the input in the center, the surrounding cells show the same motion retargeted onto a diverse set of humanoid and animal characters. The retargeted motion preserves the rhythm and semantics of the source while respecting each target skeleton’s topology, enabling flexible cross-species motion transfer from a single video.

- 5 Limitations

We see a few natural directions for further improvement. First, the Pose-to-Rotation decoder implicitly learns plausible per-skeleton motion priors from data, so for unnatural retargeting cases (e.g., transferring a bird’s flapping motion to a dog so that it “flies” with its forelegs spread out) the predicted rotations tend to drift toward more typical configurations for that skeleton, even when Video-to-Pose recovers the intended pose; in the opposite direction, the human skeleton can indeed reproduce such unusual motions. Supporting more unnatural retargeting cases would mainly call for augmenting the training set with such configurations. Second, we have not yet explored occlusion case, which is an interesting direction for future work. Third, our current animal dataset contains only ∼1,000 sequences, and we expect results to improve further as this scales up.

- 6 Conclusion

We have presented the first fully end-to-end framework for arbitraryskeleton motion capture from monocular video, in which both Videoto-Pose and Pose-to-Rotation are learnable and jointly trained. The key enabler is a single reference pose–rotation pair from the target asset, which anchors the coordinate axes of each joint’s local frame and thereby converts the ill-posed P→R mapping into a wellconstrained conditional prediction task. Once P→R is learnable, joint training lets the intermediate pose representation reshape itself to serve the final rotation objective. This is something factorized pipelines, with their non-differentiable IK handoff, cannot achieve.

On Truebones Zoo and Objaverse, our approach reduces the average rotation angle error from ∼17◦ (V1’s factorized learned-V→P + analytical-IK pipeline) to ∼10◦, with the largest improvements on unseen skeletons whose axis conventions rest-pose structure alone

cannot disambiguate. Removing the mesh intermediate used by prior work as a video-to-joint bridge further yields ∼20× faster inference, since predicted-mesh errors compound through the pipeline rather than providing useful information.

References

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-End Object Detection with Transformers. In Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I (Glasgow, United Kingdom). Springer-Verlag, Berlin, Heidelberg, 213–229. doi:10.1007/978-3-030-58452-8_13

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. 2023a. Objaverse-XL: a universe of 10M+ 3D objects. In Proceedings of the 37th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’23). Curran Associates Inc., Red Hook, NY, USA, Article 1554, 15 pages.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsanit, Aniruddha Kembhavi, and Ali Farhadi. 2023b. Objaverse: A Universe of Annotated 3D Objects . In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 13142–13153. doi:10.1109/CVPR52729.2023.01263

Inbar Gat, Sigal Raab, Guy Tevet, Yuval Reshef, Amit Haim Bermano, and Daniel Cohen-Or. 2025. AnyTop: Character Animation Diffusion with Any Topology. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers ’25). Association for Computing Machinery, New York, NY, USA, Article 13, 10 pages. doi:10.1145/3721238.3730621

Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. 2023. Humans in 4D: Reconstructing and Tracking Humans with Transformers . In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society, Los Alamitos, CA, USA, 14737–14748. doi:10.1109/ICCV51070. 2023.01358

Kehong Gong, Zhengyu Wen, Weixia He, Mingxi Xu, Qi Wang, Ning Zhang, Zhengyu Li, Dongze Lian, Wei Zhao, Xiaoyu He, and Mingyuan Zhang. 2026. MoCapAnything: Unified 3D Motion Capture for Arbitrary Skeletons from Monocular Videos. arXiv:2512.10881 [cs.CV] https://arxiv.org/abs/2512.10881

Or Hirschorn and Shai Avidan. 2024. A Graph-Based Approach for Category-Agnostic Pose Estimation. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LXIII (Milan, Italy). SpringerVerlag, Berlin, Heidelberg, 469–485. doi:10.1007/978-3-031-73036-8_27

Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. 2018a. End-toend recovery of human shape and pose. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, Washington, DC, USA, 7122–7131.

Angjoo Kanazawa, Shubham Tulsiani, Alexei A. Efros, and Jitendra Malik. 2018b. Learning Category-Specific Mesh Reconstruction from Image Collections. In Computer Vision – ECCV 2018: 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part XV (Munich, Germany). Springer-Verlag, Berlin, Heidelberg, 386–402. doi:10.1007/978-3-030-01267-0_23

Muhammed Kocabas, Nikos Athanasiou, and Michael J. Black. 2020. VIBE: Video Inference for Human Body Pose and Shape Estimation . In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 5252–5262. doi:10.1109/CVPR42600.2020.00530

Zizhang Li, Dor Litvak, Ruining Li, Yunzhi Zhang, Tomas Jakab, Christian Rupprecht, Shangzhe Wu, Andrea Vedaldi, and Jiajun Wu. 2024. Learning the 3D Fauna of the Web . In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 9752–9762. doi:10.1109/ CVPR52733.2024.00931

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. 2015. SMPL: a skinned multi-person linear model. ACM Trans. Graph. 34, 6, Article 248 (Nov. 2015), 16 pages. doi:10.1145/2816795.2818013

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2021. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. Commun. ACM 65, 1 (Dec. 2021), 99–106. doi:10.1145/3503250

Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2023. DINOv2: Learning Robust Visual Features without Supervision.

Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. Osman, Dimitrios Tzionas, and Michael J. Black. 2019. Expressive Body Capture: 3D Hands,

Face, and Body From a Single Image . In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 10967–10977. doi:10.1109/CVPR.2019.01123

Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. 2019. 3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training . In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 7745–7754. doi:10.1109/ CVPR.2019.00794

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. 2018. FiLM: visual reasoning with a general conditioning layer. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence (New Orleans, Louisiana, USA) (AAAI’18/IAAI’18/EAAI’18). AAAI Press, Palo Alto, California, USA, Article 483, 10 pages.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res. 21, 1, Article 140 (Jan. 2020), 67 pages.

Matan Rusanovsky, Or Hirschorn, and Shai Avidan. 2025. CapeX: CategoryAgnostic Pose Estimation from Textual Point Explanation. In International Conference on Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (Eds.), Vol. 2025. International Conference on Learning Representations, Singapore, 32015–32032. https://proceedings.iclr.cc/paper_files/paper/2025/file/ 4f5aeaee95e528a0ec5040bfa2fe9303-Paper-Conference.pdf

Xiaolong Shen, Zongxin Yang, Xiaohan Wang, Jianxin Ma, Chang Zhou, and Yi Yang. 2023. Global-to-Local Modeling for Video-Based 3D Human Pose and Shape Estimation . In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 8887–8896. doi:10.1109/ CVPR52729.2023.00858

Dahu Shi, Xing Wei, Liangqi Li, Ye Ren, and Wenming Tan. 2022. End-to-End MultiPerson Pose Estimation with Transformers . In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 11059–11068. doi:10.1109/CVPR52688.2022.01079

Min Shi, Zihao Huang, Xianzheng Ma, Xiaowei Hu, and Zhiguo Cao. 2023. Matching Is Not Enough: A Two-Stage Framework for Category-Agnostic Pose Estimation . In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 7308–7317. doi:10.1109/CVPR52729.2023. 00706

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. RoFormer: Enhanced transformer with Rotary Position Embedding. Neurocomput. 568, C (Feb. 2024), 12 pages. doi:10.1016/j.neucom.2023.127063

Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. 2019. Deep High-Resolution Representation Learning for Human Pose Estimation . In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 5686–5696. doi:10.1109/CVPR.2019.00584

Truebones. n.d.. Truebones Motion Capture – mocap files. https://truebones.gumroad. com/l/skZMC Accessed: 2025-05-22.

Shangzhe Wu, Ruining Li, Tomas Jakab, Christian Rupprecht, and Andrea Vedaldi. 2023. MagicPony: Learning Articulated 3D Animals in the Wild . In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 8792–8802. doi:10.1109/CVPR52729.2023.00849 Yabo Xiao, Kai Su, Xiaojuan Wang, Dongdong Yu, Lei Jin, Mingshu He, and Zehuan Yuan.

2022. Querypose: Sparse multi-person pose regression via spatial-aware part-level query. Advances in Neural Information Processing Systems 35 (2022), 12464–12477.

Lumin Xu, Sheng Jin, Wang Zeng, Wentao Liu, Chen Qian, Wanli Ouyang, Ping Luo, and Xiaogang Wang. 2022a. Pose for Everything: Towards Category-Agnostic Pose Estimation. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VI (Tel Aviv, Israel). Springer-Verlag, Berlin, Heidelberg, 398–416. doi:10.1007/978-3-031-20068-7_23

Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. 2022b. ViTPose: simple vision transformer baselines for human pose estimation. In Proceedings of the 36th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’22). Curran Associates Inc., Red Hook, NY, USA, Article 2795, 14 pages.

Gengshan Yang, Deqing Sun, Varun Jampani, Daniel Vlasic, Forrester Cole, Huiwen Chang, Deva Ramanan, William T. Freeman, and Ce Liu. 2021. LASR: Learning Articulated Shape Reconstruction from a Monocular Video . In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 15975–15984. doi:10.1109/CVPR46437.2021.01572

Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ramanan, Andrea Vedaldi, and Hanbyul Joo. 2022. BANMo: Building Animatable 3D Neural Models from Many Casual Videos . In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 2853–2863. doi:10.1109/CVPR52688.2022.00288

Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. 2019. On the Continuity of Rotation Representations in Neural Networks . In 2019 IEEE/CVF Conference

on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 5738–5746. doi:10.1109/CVPR.2019.00589

Silvia Zuffi, Angjoo Kanazawa, David W. Jacobs, and Michael J. Black. 2017. 3D Menagerie: Modeling the 3D Shape and Pose of Animals . In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 5524–5532. doi:10.1109/CVPR.2017.586

- A Implementation Details

We use a frozen DINOv2 [Oquab et al. 2023] ViT encoder as the visual backbone. The Video-to-Pose and Pose-to-Rotation modules are trained jointly end-to-end, with the rotation decoder using 𝐿=8 blocks and reference cross-attention in the first 𝐿cross=6 layers. Sequences are processed with a length of 𝑇=48 frames and per-joint temporal attention uses a window size of 5. The model is trained with Adam on 8× V100 for 60 epochs with a batch size of 8, taking roughly one day. Loss weights are𝜆pos=𝜆rot=𝜆rot_v=1.0 and𝜆root=0.1. The mixed-pose schedule uses 𝑝start=0.1, 𝑝end=1.0, and 𝐸warmup=30. The maximum joint count is set to 150 to accommodate the largest skeleton in our training data (143 joints); this limit is straightforward to adjust for other datasets.

- B Warm-up Schedule Sensitivity

We study the sensitivity of the mixed-pose warm-up length 𝐸𝑤 in the end-to-end training strategy (§4.4). Table 7 reports angle and position metrics for 𝐸𝑤 ∈ {10, 20, 30, 40, 50}. Performance is consistent across 𝐸𝑤 ∈ [20, 50], indicating that the method is robust to the mixing schedule; we use 𝐸𝑤=30 by default.

- C Model Depth

We study the effect of model depth by jointly scaling both the Videoto-Pose and Pose-to-Rotation modules (Table 8). Increasing the depth from 6 to 8 layers improves rotation accuracy, while further increasing to 12 layers degrades performance across all splits. The 8-layer model achieves the best results (6.54◦ on Zoo-Unseen), suggesting a good balance between model capacity and optimization.

- D Cross-Attention Depth

We study the effect of reference cross-attention depth (𝐿cross) in the rotation decoder, while keeping the overall model depth fixed to 8 layers (Table 9). Without cross-attention (𝐿cross=0), the model fails on Zoo-Unseen (23.49◦), showing that reference conditioning is essential. Introducing cross-attention significantly improves performance, with the best results achieved at𝐿cross=6 (6.54◦ on ZooUnseen). Applying it in all layers (𝐿cross=8) brings no further gain and slightly degrades performance, indicating diminishing returns when over-conditioning the decoder.

- Table 8. Effect of model depth (Video-to-Pose and Pose-to-Rotation jointly scaled). Position in cm; rotation in degrees (↓). Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen Obj

Depth JP JV An AV JP JV An AV JP JV An AV JP JV An AV 6 2.26 0.52 10.84 0.30 3.08 0.70 14.42 0.38 3.37 0.99 7.00 0.19 3.92 1.05 11.20 0.28 8 (Ours) 2.34 0.53 10.73 0.29 2.98 0.61 14.38 0.37 3.39 0.99 6.54 0.17 3.84 1.05 11.06 0.30 12 3.07 0.64 11.35 0.31 3.72 0.71 15.17 0.39 4.26 1.18 7.66 0.19 4.28 1.13 11.98 0.30

- Table 9. Effect of reference cross-attention depth (𝐿cross) in an 8-layer rotation decoder. Position in cm; rotation in degrees (↓). Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen Obj

𝐿cross JP JV An AV JP JV An AV JP JV An AV JP JV An AV 0 4.56 0.87 12.32 0.33 5.30 0.89 16.32 0.43 6.40 1.57 23.49 0.63 5.37 1.39 18.34 0.46 2 2.24 0.53 10.95 0.30 2.92 0.60 14.90 0.38 3.27 1.02 7.21 0.19 3.79 1.07 11.40 0.29 4 2.34 0.53 10.83 0.30 2.81 0.57 14.46 0.37 3.07 0.97 7.60 0.21 3.78 1.05 11.47 0.29 6 2.34 0.53 10.73 0.29 2.98 0.61 14.38 0.37 3.39 0.99 6.54 0.17 3.84 1.05 11.06 0.30 8 2.58 0.57 11.11 0.30 3.08 0.63 13.94 0.37 3.43 1.02 7.47 0.20 4.22 1.16 11.22 0.29

- Table 7. Sensitivity to the warm-up length 𝐸𝑤 of the mixed-pose schedule. Position in cm; rotation in degrees (↓). Best angle error per split in bold.

Zoo-Seen Zoo-Rare Zoo-Unseen Obj

𝐸𝑤 JP JV An AV JP JV An AV JP JV An AV JP JV An AV 10 2.99 0.64 11.73 0.32 3.61 0.69 15.18 0.40 3.74 1.09 7.71 0.20 4.38 1.33 11.46 0.29 20 2.42 0.56 11.18 0.31 3.51 0.66 14.93 0.39 3.32 1.00 6.48 0.17 3.96 1.04 11.06 0.28 30 (Ours) 2.34 0.53 10.73 0.29 2.98 0.61 14.38 0.37 3.39 0.99 6.54 0.17 3.84 1.05 11.06 0.30 40 2.59 0.53 11.08 0.30 3.46 0.62 15.42 0.39 4.23 1.00 7.10 0.19 4.09 1.15 11.09 0.28 50 2.28 0.54 10.75 0.30 3.01 0.61 14.42 0.38 3.43 0.99 6.61 0.18 4.14 1.10 11.86 0.29

