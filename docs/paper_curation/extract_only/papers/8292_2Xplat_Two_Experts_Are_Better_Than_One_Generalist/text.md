# arXiv:2603.21064v2[cs.CV]24Mar2026

## 2Xplat: Two Experts Are Better Than One Generalist

Hwasik Jeong1⋆ Seungryong Lee2⋆ Gyeongjin Kang2 Seungkwon Yang1 Xiangyu Sun2 Seungtae Nam1 Eunbyung Park1†

1 Yonsei University 2 Sungkyunkwan University

[Figure 1]

Fig. 1: Left: An illustration of our 2Xplat on a DL3DV scene with 32 input views. Right: PSNR vs. Inference speed (DL3DV, 224 × 224 resolution, 12 input views).

Abstract. Pose-free feed-forward 3D Gaussian Splatting (3DGS) has opened a new frontier for rapid 3D modeling, enabling high-quality Gaussian representations to be generated from uncalibrated multi-view images in a single forward pass. The dominant approach in this space adopts unified monolithic architectures, often built on geometry-centric 3D foundation models, to jointly estimate camera poses and synthesize 3DGS representations within a single network. While architecturally streamlined, such “all-in-one” designs may be suboptimal for high-fidelity 3DGS generation, as they entangle geometric reasoning and appearance modeling within a shared representation. In this work, we introduce 2Xplat, a pose-free feed-forward 3DGS framework based on a two-expert design that explicitly separates geometry estimation from Gaussian generation. A dedicated geometry expert first predicts camera poses, which are then explicitly passed to a powerful appearance expert that synthesizes 3D Gaussians. Despite its conceptual simplicity, being largely underexplored in prior works, the proposed approach proves highly effective. In fewer than 5K training iterations, the proposed two-experts pipeline substantially outperforms prior pose-free feed-forward 3DGS approaches and achieves performance on par with state-of-the-art posed methods. These results challenge the prevailing unified paradigm and

† Corresponding author ⋆ Equal contribution

suggest the potential advantages of modular design principles for complex 3D geometric estimation and appearance synthesis tasks. Project page: https://hwasikjeong.github.io/2Xplat/.

Keywords: 3D Gaussian Splatting · Feed-Forward 3D Gaussian Splatting · Novel-view Synthesis

### 1 Introduction

3D Gaussian Splatting (3DGS) [32] has recently emerged as a powerful representation for high-quality, real-time novel-view synthesis, enabling a wide range of practical applications, including AR/XR [26,27], immersive telepresence [20,48], volumetric video production [53,57], robotics [9,41], and autonomous driving [15, 78], to name a few. However, conventional 3DGS pipelines rely on computationally intensive iterative optimization procedures, often requiring tens of minutes to hours per scene [18,32,42,44,73,79], thereby limiting their broader applicability. To address this bottleneck, feed-forward 3DGS methods have been actively studied to directly predict Gaussian parameters from multi-view images in a single pass [3,4,19,58,68,74,76], reducing reconstruction time to a few seconds, even for large numbers of high-resolution inputs, while achieving view synthesis quality comparable to optimization-based methods.

Despite these advances, many feed-forward approaches assume access to accurate camera poses, which limits their applicability in unconstrained settings. While calibrated setups such as camera rigs, studio capture systems, or multicamera systems in autonomous vehicles can provide reliable pose estimates, many real-world scenarios lack such information. In principle, camera poses can be estimated using inertial sensors [12,49] or SfM/SLAM pipelines [6,13,45,51,52], but obtaining sufficiently accurate estimates can be time-consuming and may incur pose errors, leading to noticeable degradation in reconstruction quality. Consequently, the overall efficiency advantage of feed-forward 3DGS diminishes when pose estimation becomes the dominant computational cost or failure point. These limitations have motivated the development of pose-free feed-forward 3DGS methods [16, 19, 25, 31, 70, 71], which aim to reconstruct Gaussian representations directly from uncalibrated multi-view images.

Most existing pose-free feed-forward approaches adopt a monolithic design, in which a single network jointly predicts camera poses and 3DGS parameters using shared features with task-specific output heads (or post optimization) [25,70,71]. For example, recent methods augment a geometry estimation backbone with Gaussian prediction heads, producing both camera poses and per-pixel 3DGS attributes in a single forward pass. While this unified architecture is conceptually appealing, we argue that it may be inherently limited in achieving state-of-theart performance in both geometry and appearance modeling.

First, in appearance modeling, particularly with 3DGS representations, strict adherence to accurate scene geometry may not be essential for achieving highquality novel-view synthesis. Indeed, enforcing strong geometric constraints can

sometimes degrade visual fidelity, as small geometric inaccuracies may be perceptually negligible, while strict consistency can limit the model’s ability to reproduce complex appearance effects such as translucency, thin or high-frequency structures, and view-dependent shading. Consequently, a unified model that simultaneously produces geometrically accurate structure and visually optimized Gaussian parameters faces inherently conflicting objectives.

Second, achieving high-fidelity 3DGS reconstruction requires a dedicated appearance expert rather than a unified monolithic architecture or a minimally extended geometry network. This is reflected in state-of-the-art posed feed-forward 3DGS methods, which employ sophisticated architectural designs that explicitly leverage known camera poses throughout the pipeline. In particular, a substantial body of work has developed mechanisms to inject pose information into multiview transformers, such as Epipolar Transformer [14], PRoPE [37], GTA [43], CaPE [66], and RayRoPE [65], consistently demonstrating that tightly coupling visual features with camera poses leads to performance gains. By aligning features according to camera poses, they reduce the burden on the network to learn geometry from scratch. In contrast, unified monolithic architectures that jointly infer camera poses and appearance must rely on implicitly estimated geometric knowledge during synthesis, limiting their ability to fully incorporate advanced pose-conditioned architectural mechanisms.

Third, generating high-quality 3DGS attributes is not merely a minor refinement of predicted geometry; it demands substantial representational capacity and sophisticated spatial reasoning. High-fidelity Gaussian attributes must capture multi-view consistency, fine-grained structural details, and complex viewdependent appearance effects across images. Put differently, the appearance expert is expected to generate high-fidelity 3D Gaussians and their attributes in a single forward pass, an outcome that conventionally requires tens of thousands of gradient-based optimization iterations. Such complexity is unlikely to be adequately handled by a lightweight extension of a geometry-centric backbone.

As an alternative, “geometry-first, appearance synthesis-second” approaches have been explored in several prior works [24, 31, 35, 55, 76]. These approaches primarily focus on self-supervised learning paradigms, training geometry and appearance jointly without explicit 3D supervision such as ground-truth camera poses or depth. While promising, their emphasis largely lies in training strategies and geometry estimation, with comparatively less attention devoted to fully exploiting recent advances in high-capacity appearance models and poseconditioned architectures. As a result, their novel-view synthesis quality remains limited compared to state-of-the-art posed feed-forward 3DGS methods. In this work, we revisit this paradigm from a different perspective: rather than emphasizing self-supervised training alone, we investigate how far high-quality pose-free novel-view synthesis can be pushed by explicitly combining a strong geometry estimator with a powerful, pose-conditioned 3DGS generator.

While this two-stage design may appear to introduce an information bottleneck between geometry estimation and appearance synthesis, it in fact provides a significant practical advantage in training efficiency. In monolithic architectures,

although the backbone can be initialized from pretrained weights, additional task-specific modules and prediction heads are typically randomly initialized and learned jointly with the pretrained components. This makes optimization more challenging and often requires longer training, sometimes needing large-scale datasets similar to those used for the original foundation models. In contrast, our framework directly reuses two mature pretrained experts without introducing newly initialized modules. As a result, the entire pipeline can be optimized efficiently through lightweight end-to-end fine-tuning. In practice, the full model converges in fewer than 5K iterations, highlighting the remarkable training efficiency of the proposed modular design.

Despite its conceptual simplicity, this framework has been surprisingly underexplored, to the best of our knowledge. Nevertheless, it delivers substantial improvements over prior pose-free feed-forward 3DGS methods and achieves stateof-the-art performance by a large margin. In addition, the proposed approach performs on par with state-of-the-art posed feed-forward 3DGS methods in novel view synthesis, paving the way toward eliminating the need for explicit camera pose information in many practical applications. In sum, our key contributions can be summarized as follows:

- – We explore an end-to-end two-expert framework that decomposes pose-free feed-forward 3DGS into a dedicated geometry expert and an appearance expert.
- – By explicitly conditioning the appearance expert on predicted camera poses, our design enables the incorporation of advanced pose-aware architectural mechanisms.
- – Through end-to-end joint optimization, our appearance expert becomes robust to noisy camera pose estimates, mitigating the sensitivity of 3DGS generation to geometric errors.
- – Our approach significantly outperforms prior pose-free feed-forward 3DGS methods and performs on par with state-of-the-art posed models in novel view synthesis.

### 2 Related Works

#### 2.1 Feed-forward 3D Foundation Models

Traditional 3D reconstruction methods rely on per-scene optimization pipelines such as Structure-from-Motion [51] (SfM) followed by Multi-View Stereo [13,52] (MVS), which are computationally expensive and brittle to sparse or unstructured inputs. Recent efforts have shifted toward data-driven, feed-forward approaches that amortize reconstruction cost across large-scale training, enabling inference-time generalization without per-scene optimization.

A particularly influential line of work builds on Vision Transformers [7] to directly regress 3D structure from images. DUSt3R [62] and MASt3R [36] pioneered this paradigm by framing pairwise reconstruction as a dense pointmap

regression problem, allowing unconstrained camera pose estimation and geometry prediction in a single forward pass. While these models demonstrate impressive generalization to in-the-wild images, they operate primarily on image pairs, and scaling to multi-view inputs requires a global alignment postprocessing step that aggregates pairwise predictions. More recent work relaxes this two-view constraint by operating directly over arbitrary numbers of input views [38, 59, 60, 63, 69]. These multi-view methods leverage attention mechanisms across view tokens to jointly reason about geometry and camera parameters, achieving strong performance on standard benchmarks while significantly reducing inference latency compared to global alignment-based pipelines.

#### 2.2 Posed Feed-forward 3D Models

A large body of feed-forward 3D reconstruction methods conditions on known camera poses at test time, offloading the pose estimation problem to an external system such as SfM [51]. LRM [17] introduced a large-scale transformer that maps image to a neural radiance field [11] in a single forward pass, establishing a foundation for subsequent feed-forward approaches.

These methods can be broadly categorized by their choice of 3D representation. Explicit methods directly predict 3D primitives (e.g. Gaussians) from posed input views, using a variety of strategies ranging from geometry-guided approaches that leverage epipolar constraints [3] or cost-volume-based feature matching [4,68], to iterative feedback-driven refinement schemes [29,46,67], to purely data-driven transformer architectures that learn to regress primitives endto-end [21,30,74]. Implicit methods, on the other hand, eschew explicit 3D representations entirely, instead training large-scale transformers to directly perform neural rendering and synthesize novel views from posed images [10,28,50]. While these methods achieve impressive reconstruction quality and fast inference, they fundamentally assume that accurate camera poses are available at test time.

#### 2.3 Pose-free Feed-forward 3D Models

To remove the dependency on known camera poses, a growing line of work explores feed-forward 3D reconstruction from unposed images, jointly inferring scene geometry, appearance, and camera parameters in a single pass. This paradigm is particularly appealing in practice, as acquiring accurate camera poses requires careful calibration procedures. Representative approaches span a range of scene representations, including neural field [23,55,61] and 3D Gaussian Splatting [16,25,31,56,70,71]. These methods demonstrate that accurate geometry and photorealistic appearance can be recovered directly from unposed image collections, without any pose inputs at inference time.

Despite this progress, prevailing pose-free reconstruction pipelines [25, 56, 70,71] share a common architectural bottleneck: a single monolithic network is tasked with simultaneously estimating camera poses and Gaussian parameters using shared features, thereby entangling two fundamentally distinct objectives within a single representational bottleneck. We argue that this design imposes an

[Figure 2]

Fig. 2: Two experts vs. one generalist: (a) The prevailing monolithic architecture employs a shared backbone with task-specific heads to jointly predict camera poses and 3DGS representations in a single forward pass. (b) The two-expert architecture explicitly decomposes the process into sequential stages: a geometry expert first estimates camera poses from multi-view input images, and a dedicated appearance (posed feedforward 3DGS) expert subsequently generates 3D Gaussian representations conditioned on the predicted poses and input images.

inherent performance ceiling for both tasks. Our approach presents a two-expert framework in which specialized modules handle each objective independently, yet remain tightly coupled through end-to-end joint optimization.

### 3 Method

#### 3.1 Problem Formulation

We consider the pose-free feed-forward 3DGS task, where the goal is to generate a 3D Gaussian representation directly from unposed multi-view images. Optionally, the model may also estimate the camera pose associated with each input view. Formally, let {Ii ∈ RH×W×3}Ni=1 denotes a set of N input images with height H and width W. A pose-free feed-forward model F(·) maps input images to a set of pixel-aligned 3D Gaussians and camera parameters.

{pˆi}Ni=1,{Gj}N

j=1 = F({Ii}Ni=1), (1) where N = Nc + Nt, and Nc and Nt denote the numbers of context and target views, respectively. Gj ∈ RH×W×d

c

g denotes pixel-aligned 3D Gaussian for each context view image. The camera parameters pi = [Ki,Ri,ti] ∈ Rd

p represent the intrinsic Ki and extrinsic components (Ri,ti) corresponding to the image Ii, and dg and dp are the dimensions of the 3D Gaussian attributes and camera parameters, respectively.

#### 3.2 Monolithic vs. Two-Experts Architecture

A common architectural paradigm for pose-free feed-forward 3DGS adopts a monolithic design, where a single network jointly predicts camera poses and 3D

Gaussian parameters using a shared backbone with task-specific output heads. As discussed earlier (Sec. 1), monolithic architectures may entangle geometry estimation and appearance modeling within a shared representation, thereby limiting representational specialization. Also, it is not straightforward to fully exploit the recent pose-conditioned architectural designs and may lack the capacity required for high-fidelity 3DGS generation, which demands sophisticated multi-view reasoning beyond a lightweight extension of a geometry backbone.

Recent approaches further extend this paradigm by allowing camera poses to be optionally provided as input, enabling a single model to handle both posed and pose-free settings [22,38,70]. While flexible and appealing in principle, this unified formulation introduces additional complexity. Internally, the network must implicitly switch between two operational modes: predicting camera poses when ground-truth poses are unavailable, and bypassing pose prediction when they are provided. Learning such a sophisticated switching mechanism in a shared representation is non-trivial. Moreover, incorporating advanced pose-conditioned architectural mechanisms into this fused structure is challenging, as explicit camera pose information is not cleanly separated from the learned features.

We explore a two-experts framework that explicitly decomposes geometry estimation and 3DGS generation into sequential modules. The framework consists of a pose expert Fpose, which estimates camera parameters from input images, and an appearance expert F3dgs, which generates pixel-aligned 3D Gaussian representations conditioned on the context view images and the corresponding predicted poses. The detailed formulation is described in Sec. 3.5.

The entire pipeline remains end-to-end trainable, enabling the 3DGS generator to become robust to pose estimation errors through joint optimization. In addition, when ground-truth camera parameters are available, the pose expert can be simply bypassed, and the appearance expert can directly operate in the posed setting. This modular design naturally accommodates both scenarios. Furthermore, it enables independent incorporation of architectural advancements from both geometry estimation and posed feed-forward 3DGS.

Separating the pose and appearance modules raises concerns about redundant processing, as pose estimation and multi-view consistent appearance modeling may share certain low-level visual reasoning. However, our empirical results indicate that such redundancy does not compromise efficiency and, in fact, proves beneficial. With comparable, and in some cases even fewer (Tab. 6), parameters than monolithic counterparts, the proposed two-experts framework consistently achieves significantly better performance. Nevertheless, exploring more principled ways to share low-level geometric and visual reasoning between the two experts remains an interesting direction, and we leave it to future work.

#### 3.3 Geometry Expert

Recent advances in large-scale 3D geometry foundation models such as DUSt3R [62], VGGT [60], π3 [63], and Depth Anything 3 (DA3) [38], have significantly improved multi-view geometry estimation. These models are trained on extensive

synthetic and real-world datasets, requiring sophisticated data curation that includes dense depth, point maps, ray maps, and camera pose annotations. A consistent finding across recent works is that jointly learning multiple geometric tasks, even when some tasks are theoretically convertible (e.g., depth, point maps, and camera poses), leads to improved performance due to shared supervision and synergetic multitask training. In particular, DA3 demonstrates that training with depth, pointmap, ray maps, and auxiliary camera pose objectives yields state-of-the-art results in both pose accuracy and geometry reconstruction. Given its strong performance, we adopt DA3 as our geometry expert. For a fair comparison with prior pose-free feed-forward 3DGS methods [70,71], we additionally evaluate alternative geometry backbones to ensure that the benefits of our two-expert framework are not tied to a single geometry model (Tab. 6).

#### 3.4 Appearance Expert

For the 3DGS expert, we adopt the recent Multi-view Pyramid Transformer (MVP) architecture [30], which currently represents the state of the art among posed feed-forward 3D Gaussian Splatting models in both reconstruction quality and inference efficiency. MVP integrates several advanced architectural components, including the PRoPE-based camera pose conditioning mechanism [37], register tokens for stabilizing transformer representations [5], and an Alternating Attention design [60] with a dual hierarchical framework. Notably, the model is trained entirely from scratch without relying on pre-trained DINO features [2, 47, 54], demonstrating the strength of its architectural design. Owing to its computational efficiency and scalable dual-attention hierarchy, it enables the use of more fine-grained spatial tokens (i.e., smaller patch sizes), leading to improved reconstruction fidelity without prohibitive cost. Given its superior performance–efficiency trade-off and strong camera pose conditioning, we employ MVP as our 3DGS expert in the proposed two-experts framework.

#### 3.5 Joint Training

We initialize our framework from two pretrained experts—a geometry (camera pose) expert and an appearance (3DGS) expert—and fine-tune the entire system

end-to-end. Given a set of context and target views {Ii}Ni=1, where N = Nc+Nt, the pose expert first predicts camera parameters for all views,

{pˆi}Ni=1 = Fpose({Ii}Ni=1). (2)

The 3DGS expert then takes the context images together with their predicted camera parameters to generate pixel-aligned 3D Gaussian representations,

{Gi}N

i=1 = F3dgs({Ii}N

i=1,{pˆi}N

i=1). (3)

c

c

c

Using a differentiable 3DGS renderer, we render each target view from the predicted Gaussian representation and compute an image reconstruction loss against

the corresponding ground-truth image. To regularize pose prediction and prevent geometric drift, we additionally supervise the predicted poses with ground-truth camera parameters. The overall training objective L is defined as

Nt

N

1 N

1 Nt

Lrender(Iˆi,IN

Lcam(ˆpj,pj). (4)

L =

c+i) +

i=1

j=1

Here, Iˆi and IN

c+i denote the rendered target view and its corresponding groundtruth image, while pˆj and pj are the predicted and ground-truth camera parameters. For the rendering loss Lrender, we adopt a weighted combination of ℓ2 reconstruction loss and perceptual loss to balance pixel-wise accuracy and perceptual fidelity,

Lrender = LMSE(I,Iˆ ) + λpercLperc(I,Iˆ ) (5)

where λperc controls the contribution of the perceptual term. For the camera supervision term Lcam, we adopt a relative pose loss following [63,70] to address the ambiguity in the global reference frame between the predicted and groundtruth poses. Specifically, the predicted relative pose Tˆi←j from view j to i is computed as

R ˆi←j tˆi←j 0 1

Tˆi←j = Tˆi−1Tˆj =

(6)

where Tˆi←j consists of the relative rotation Rˆi←j and translation tˆi←j. The relative rotation loss LR(i,j) and translation loss Lt(i,j) are defined as

 

 , Lt(i,j) = Hδ t ˆi←j − ti←j (7)

tr Ri⊤←jRˆi←j − 1 2

LR(i,j) = arccos

where tr(·) denotes the trace of a matrix and Hδ(·) represents the Huber loss with threshold δ. The overall pose loss is the formulated as

N

1 N(N − 1) i̸=j

λK N

LK(j) (8)

Lcam =

(λRLR(i,j) + λtLt(i,j)) +

j=1

where LK denotes l2 loss between the predicted and ground-truth camera intrinsics, and λR, λt and λK are weighting factors that balance between the contributions of each component.

#### 3.6 Information Bottleneck and Training Efficiency

The proposed two-stage decomposition introduces camera pose as an intermediate interface. This design may raise concerns about potential information bottlenecks or redundant low-level visual processing across the geometry and 3DGS experts, yet it brings significant advantages in training efficiency and stability.

###### Table 1: Novel view synthesis on DL3DV with 6, 12, and 24 input views. p and k denote the use of ground-truth poses and intrinsics. † indicates evaluation with EPA.

6v 12v 24v PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ MVSplat ✓ ✓ 22.659 0.760 0.173 21.289 0.709 0.224 19.975 0.662 0.269

Method p k

- DepthSplat ✓ ✓ 23.418 0.797 0.136 21.911 0.753 0.179 20.088 0.690 0.240 YoNoSplat ✓ ✓ 24.717 0.817 0.139 23.285 0.773 0.177 22.664 0.758 0.192 Ours ✓ ✓ 26.631 0.856 0.121 27.240 0.866 0.115 27.413 0.877 0.109 NoPoSplat† ✓ 22.766 0.743 0.179 19.380 0.563 0.318 17.860 0.495 0.397

- YoNoSplat† ✓ 24.887 0.819 0.138 23.149 0.758 0.183 22.354 0.731 0.205

###### Ours† ✓ 26.673 0.855 0.122 26.971 0.855 0.122 27.094 0.865 0.116

AnySplat 19.027 0.554 0.235 18.940 0.549 0.262 19.703 0.596 0.249 YoNoSplat 22.290 0.695 0.173 20.383 0.602 0.229 19.711 0.572 0.255 YoNoSplat† 24.531 0.804 0.142 22.933 0.746 0.187 22.174 0.720 0.209 Ours 26.007 0.839 0.126 26.015 0.826 0.129 25.894 0.832 0.125

- Ours† 26.670 0.855 0.122 26.963 0.854 0.121 27.083 0.865 0.116

Because we directly use the final predicted poses from a fully pretrained geometry expert, we can leverage its large-scale pretrained weights without architectural modification. Importantly, our framework does not introduce any newly initialized modules into the pipeline.

In contrast, monolithic architectures typically initialize a shared backbone with pretrained weights but append randomly initialized additional layers and prediction heads [25,70,71]. While the pretrained backbone provides strong initialization, these newly introduced parameters do not directly benefit from the large-scale pretraining used for the underlying geometry models. When finetuned on smaller or domain-specific datasets, such parameters are more susceptible to overfitting, as they learn task-specific mappings with limited supervision.

[Figure 3]

Fig. 3: Qualitative comparison on DL3DV with varying numbers of input views.

### 4 Experiments

Dataset. We train our model on the RealEstate10K (RE10K) [77] and DL3DV [39] datasets using their official data splits. For benchmarking on RE10K, we

###### Table 2: Quantitative comparison on the DL3DV dataset under varying numbers of input views (16, 32, 64, and 128) for high-resolution (960 × 540) novel view synthesis.

16 views 32 views 64 views 128 views PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Method Optim. Pose

3D-GS30k ✓ ✓ 21.48 0.753 0.252 24.43 0.827 0.191 27.34 0.883 0.146 29.43 0.914 0.123 Long-LRM ✓ 21.05 0.708 0.297 23.97 0.778 0.267 23.60 0.789 0.260 21.24 0.739 0.308 iLRM ✓ 21.92 0.748 0.316 24.30 0.803 0.256 24.44 0.819 0.240 22.98 0.807 0.249 MVP ✓ 23.76 0.798 0.239 25.96 0.847 0.187 27.73 0.881 0.154 29.02 0.903 0.134 Ours 22.90 0.754 0.259 24.75 0.800 0.208 26.11 0.830 0.180 27.16 0.853 0.162 Ours† 23.61 0.786 0.248 25.66 0.832 0.198 27.12 0.859 0.171 28.30 0.880 0.153

[Figure 4]

Fig. 4: Qualitative comparison on high-resolution DL3DV (960 × 540).

retain only test sequences with at least 200 frames, yielding a total of 1,580 sequences. For a fair comparison, we follow prior work [70] that uses 6 context views. For DL3DV, we evaluate model performance using 6, 12, and 24 input views, with maximum frame intervals of 50, 100, and 150, respectively. To assess generalization, we evaluate the model trained on DL3DV on the ScanNet++ dataset [72]. For each scene, we sample 32, 64, and 128 views with a fixed target view. Input views are selected using farthest point sampling over camera centers, while 8 views are randomly held out for validation, following the [70]. For the high-resolution DL3DV evaluation, we use the undistorted version of the dataset following [30,80] and adopt the same evaluation protocol.

Baselines. For novel view synthesis, we compare with pose-dependent methods (MVSplat [4], DepthSplat [68], Long-LRM [80], iLRM [29], MVP [30]) and pose-free methods (NoPoSplat [71], AnySplat [25], YonoSplat [70]). For pose estimation, we compare with MASt3R [36], VGGT [60], π3 [63] and DA3 [38].

Evaluation Protocol. For novel view synthesis, we adopt standard image quality metrics, including PSNR, SSIM [64], and LPIPS [75]. For pose estimation, we report the cumulative angular pose error curve (AUC) evaluated at thresholds of 5◦, 10◦, and 20◦ [8]. We report results under both pose-dependent and pose-free evaluation protocols. In the pose-dependent protocol, target views are rendered

using the corresponding ground-truth camera poses, whereas in the pose-free protocol, rendering is performed using the predicted camera poses. Since several prior pose-free methods [70, 71] adopt evaluation-time pose alignment (EPA) during evaluation, we additionally report results with EPA for fair comparison. Results computed with EPA are marked with a † symbol.

Table 3: NVS on the RE10K.

[Figure 5]

Method p k PSNR↑ SSIM↑ LPIPS↓

- DepthSplat ✓ ✓ 24.156 0.846 0.145 YoNoSplat† ✓ ✓ 25.037 0.848 0.134 NoPoSplat† ✓ 22.175 0.750 0.207

- YoNoSplat† ✓ 25.395 0.857 0.131

###### Ours† ✓ 27.108 0.877 0.128

YoNoSplat 19.723 0.613 0.229 YoNoSplat† 24.571 0.823 0.144 Ours 26.161 0.859 0.132

- Ours† 27.239 0.881 0.126

Fig. 5: Qualitative comparison on RE10K.

Implementation Details. We use the pretrained Depth Anything 3 [38] and Multi-view Pyramid Transformer [30] as the geometry and appearance experts, respectively. We use the same image resolution to ensure a fair comparison with [70]. For RE10K, we train with an image resolution of 224 × 224, using 6 context views and 8 target views with a batch size of 8 per GPU. For DL3DV, we trained with varying numbers of context and target views (6-32 context views and 1-4 target views), with the per-GPU batch size adjusted accordingly. For the high-resolution DL3DV setting, we use images of resolution 540×960 for the appearance expert and 280 × 504 for the geometry expert, and train the model with 16 and 32 context views. All models are trained on 8 H200 GPUs for 2K-5K iterations. YoNoSplat [70] requires 16 GH200 GPUs and 150K iterations, highlighting the superior computation efficiency of our approach. Unless otherwise specified, we use the DA3-Giant.

#### 4.1 Results

Novel View Synthesis. We evaluate our method on the DL3DV dataset in a low-resolution setting under varying input view settings. As shown in Tab. 1, our pose-free and intrinsic-free model significantly outperforms all state-of-theart baselines, including methods that rely on ground-truth camera poses or intrinsics. These results demonstrate the effectiveness of the proposed two-expert framework for pose-free novel view synthesis. Furthermore, our model consistently improves as the number of input views increases, whereas the performance of competing methods degrades notably under the same setting. This robustness highlights the effectiveness of our design in leveraging multi-view information, which can be largely attributed to our appearance modeling. Qualitative com-

parisons in Fig. 3 further illustrate that our method produces sharper structures and more consistent renderings than prior approaches.

Moreover, prior approaches such as YoNoSplat [70] and NoPoSplat [71] rely heavily on evaluation-time pose alignment (EPA) to achieve competitive performance. Our method, however, already surpasses these baselines without EPA and achieves further gains when pose alignment is applied. On the indoor RE10K dataset shown in Tab. 7, we observe a similar trend. Our method consistently outperforms both pose-free and pose-dependent state-of-the-art methods, demonstrating strong performance across both indoor and outdoor scenarios. Qualitative results are shown in Fig. 5.

We further evaluate our method on the DL3DV dataset in a high-resolution setting , as shown in Tab. 2. We compare our method with the optimizationbased 3D Gasussian Splatting [32] (30K iterations) and feed-forward reconstruction methods [29, 30, 80]. Across all evaluated settings from 16 to 128 input views, ours is the only method that performs pose-free inference. Nevertheless, our method achieves competitive performance.

Pose Estimation. Although our primary focus lies in learning a high-quality 3DGS representation, our framework also enables accurate camera pose estimation as a byproduct. Notably, our method achieves competitive AUC performance despite being fine-tuned on only a small subset of the dataset, whereas prior state-of-the-art approaches [70, 71] rely on extensive pose supervision (Tab. 4). This result suggests that the ambiguity in the global world-frame reference, which naturally arises from the geometry expert, can be effectively resolved through our two-expertise pipeline. By jointly leveraging complementary geometric and appearance modeling capabilities, our framework produces reliable pose estimates without requiring large-scale pose-specific training.

Table 4: Pose estimation comparison. Although our method is trained with only 2k iterations on RE10K, it achieves performance comparable to state-of-the-art approaches that employ substantially more intensive pose training stages.

Method 5◦ ↑ 10◦ ↑ 20◦ ↑ Backbone

MASt3R 518×288 0.609 0.776 0.878 VGGT 518×280 0.566 0.753 0.867 π3 518×280 0.705 0.841 0.916 DA3 504×504 0.694 0.826 0.900 -

NoPoSplat256×256 0.443 0.627 0.755 MASt3R YoNoSplat224×224 0.722 0.852 0.923 π3 Ours224×224 0.718 0.843 0.912 DA3

Table 5: Cross-dataset generalization from DL3DV to ScanNet++.

32v 64v 128v PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Method

AnySplat 14.054 0.494 0.468 15.982 0.551 0.412 16.988 0.583 0.386 YoNoSplat† w/o GT k 16.886 0.600 0.432 17.368 0.608 0.413 17.641 0.617 0.405 YoNoSplat† w/ GT k 17.935 0.659 0.380 18.833 0.688 0.342 19.284 0.701 0.325 Ours† w/o GT k 18.021 0.660 0.395 20.194 0.723 0.305 21.896 0.764 0.252 Ours† w/ GT k 18.136 0.667 0.390 20.446 0.734 0.297 22.302 0.781 0.240

[Figure 6]

Fig. 6: Qualitative cross-dataset generalization from DL3DV to ScanNet++.

Cross Dataset Generalization. We evaluate cross-dataset generalization by training the model on DL3DV and directly testing it on ScanNet++ without any fine-tuning. As shown in Tab. 5, our method consistently achieves strong performance across all evaluation metrics under this cross-dataset setting. Qualitative results are shown in Sec. 4.1, where our method produces sharper and more consistent renderings. These results suggest that the proposed two-expert framework can generalize well across different scene distributions.

#### 4.2 Analyses

Different Backbones. We evaluate three backbone choices for camera pose estimation: π3, DA3-L, and DA3-G (Tab. 6). Our method performs consistently well across all backbone choices, demonstrating its robustness. Notably, even with DA3-L, which has fewer parameters, our method still achieves competitive performance with faster inference speed, indicating that our approach does not rely heavily on backbone capacity. Furthermore, replacing the backbone with larger variants such as DA3-G leads to only moderate improvements, suggesting that the performance gain mainly stems from our architecture rather than increased model size. This highlights the efficiency of our design in effectively leveraging the geometry expert.

Pose Supervision. We compare three pose loss configurations in Tab. 7: relative loss, absolute loss, and without pose loss. While training without pose loss achieves marginally better rendering quality, it significantly degrades pose estimation accuracy across all angular thresholds. Therefore, we adopt relative loss, which achieves the best balance between rendering quality and pose accuracy.

##### Table 6: Different backbones.

Method size speed PSNR↑ LPIPS↓

YoNoSplat (π3) 1B 0.33s 19.723 0.229 Ours-L 0.5B 0.15s 25.758 0.135 Ours-G 1.3B 0.31s 26.161 0.132 Ours-π3 1.1B 0.27s 26.017 0.138

##### Table 7: Pose supervision.

Loss type PSNR↑ LPIPS↓ 5◦ ↑ 10◦ ↑ 20◦ ↑ Rel. loss 26.161 0.131 0.718 0.843 0.912 Abs. loss 25.704 0.137 0.641 0.797 0.888 W/O loss 26.369 0.129 0.686 0.836 0.905

### 5 Conclusion

In this work, we present 2Xplat, a two-expert framework for pose-free feedforward 3D Gaussian Splatting that decouples pose estimation from appearance synthesis. Despite its conceptual simplicity, our approach substantially outperforms prior pose-free methods and achieves performance on par with state-ofthe-art posed approaches, all within fewer than 5K training iterations, demonstrating strong reconstruction quality with remarkable training efficiency. These findings challenge the assumption that entangling geometric reasoning and appearance modeling within a shared representation is necessary or beneficial, and instead highlight the potential of modular design principles for complex 3D reconstruction tasks. We hope this work motivates further exploration of expertdecomposed architectures in 3D generation and beyond.

### References

- 1. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR (2022)
- 2. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the International Conference on Computer Vision (ICCV) (2021)
- 3. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19457– 19467 (2024)
- 4. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In: European conference on computer vision. pp. 370–386. Springer (2024)
- 5. Darcet, T., Oquab, M., Mairal, J., Bojanowski, P.: Vision transformers need registers. arXiv preprint arXiv:2309.16588 (2023)
- 6. Davison, A.J., Reid, I.D., Molton, N.D., Stasse, O.: Monoslam: Real-time single camera slam. IEEE Transactions on Pattern Analysis and Machine Intelligence 29(6), 1052–1067 (2007). https://doi.org/10.1109/TPAMI.2007.1049
- 7. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)
- 8. Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M., Felsberg, M.: Roma: Robust dense feature matching. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19790–19800 (2024)

- 9. Escontrela, A., Kerr, J., Allshire, A., Frey, J., Duan, R., Sferrazza, C., Abbeel, P.: Gaussgym: An open-source real-to-sim framework for learning locomotion from pixels. arXiv preprint arXiv:2510.15352 (2025)
- 10. Flynn, J., Broxton, M., Murmann, L., Chai, L., DuVall, M., Godard, C., Heal, K., Kaza, S., Lombardi, S., Luo, X., et al.: Quark: Real-time, high-resolution, and general neural view synthesis. ACM Transactions on Graphics (TOG) 43(6), 1–20

(2024)

- 11. Fridovich-Keil, S., Meanti, G., Warburg, F.R., Recht, B., Kanazawa, A.: K-planes: Explicit radiance fields in space, time, and appearance. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12479– 12488 (2023)
- 12. Geneva, P., Eckenhoff, K., Lee, W., Yang, Y., Huang, G.: Openvins: A research platform for visual-inertial estimation. In: 2020 IEEE International Conference on Robotics and Automation (ICRA). pp. 4666–4672 (2020). https://doi.org/10. 1109/ICRA40945.2020.9196524
- 13. Hartley, R., Zisserman, A.: Multiple view geometry in computer vision. Cambridge university press (2003)
- 14. He, Y., Yan, R., Fragkiadaki, K., Yu, S.I.: Epipolar transformers. In: Proceedings of the ieee/cvf conference on computer vision and pattern recognition. pp. 7779–7788

(2020)

- 15. Hess, G., Lindström, C., Fatemi, M., Petersson, C., Svensson, L.: Splatad: Realtime lidar and camera rendering with 3d gaussian splatting for autonomous driving. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 11982–11992 (2025)
- 16. Hong, S., Jung, J., Shin, H., Han, J., Yang, J., Luo, C., Kim, S.: Pf3plat: Pose-free feed-forward 3d gaussian splatting. arXiv preprint arXiv:2410.22128 (2024)
- 17. Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., Tan, H.: Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023)
- 18. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. In: ACM SIGGRAPH 2024 conference papers. pp. 1–11 (2024)
- 19. Huang, R., Mikolajczyk, K.: No pose at all: Self-supervised pose-free 3d gaussian splatting from sparse views. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 27947–27957 (2025)
- 20. Huang, X., Frehlich, D., Xia, Z., Gholami, P., Xiao, R.: Gaussiannexus: Room-scale real-time ar/vr telepresence with gaussian splatting. Association for Computing Machinery, New York, NY, USA (2025)
- 21. Imtiaz, T., Chai, L., Heal, K., Luo, X., Park, J., Dy, J., Flynn, J.: Lvt: Largescale scene reconstruction via local view transformers. In: Proceedings of the SIGGRAPH Asia 2025 Conference Papers. pp. 1–12 (2025)
- 22. Jang, W., Weinzaepfel, P., Leroy, V., Agapito, L., Revaud, J.: Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1071–1081 (2025)
- 23. Jiang, H., Jiang, Z., Zhao, Y., Huang, Q.: Leap: Liberate sparse-view 3d modeling from camera poses. ArXiv 2310.01410 (2023)
- 24. Jiang, H., Tan, H., Wang, P., Jin, H., Zhao, Y., Bi, S., Zhang, K., Luan, F., Sunkavalli, K., Huang, Q., et al.: Rayzer: A self-supervised large view synthesis model. arXiv preprint arXiv:2505.00702 (2025)

- 25. Jiang, L., Mao, Y., Xu, L., Lu, T., Ren, K., Jin, Y., Xu, X., Yu, M., Pang, J., Zhao, F., et al.: Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on Graphics (TOG) 44(6), 1–16 (2025)
- 26. Jiang, Y., Yu, C., Xie, T., Li, X., Feng, Y., Wang, H., Li, M., Lau, H., Gao, F., Yang, Y., et al.: Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In: ACM SIGGRAPH 2024 conference papers. pp. 1–1

(2024)

- 27. Jiang, Y., Shen, Z., Hong, Y., Guo, C., Wu, Y., Zhang, Y., Yu, J., Xu, L.: Robust dual gaussian splatting for immersive human-centric volumetric videos. ACM Transactions on Graphics (TOG) 43(6), 1–15 (2024)
- 28. Jin, H., Jiang, H., Tan, H., Zhang, K., Bi, S., Zhang, T., Luan, F., Snavely, N., Xu, Z.: Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242 (2024)
- 29. Kang, G., Nam, S., Yang, S., Sun, X., Khamis, S., Mohamed, A., Park, E.: ilrm: An iterative large 3d reconstruction model. arXiv preprint arXiv:2507.23277 (2025)
- 30. Kang, G., Yang, S., Nam, S., Lee, Y., Kim, J., Park, E.: Multi-view pyramid transformer: Look coarser to see broader. arXiv preprint arXiv:2512.07806 (2025)
- 31. Kang, G., Yoo, J., Park, J., Nam, S., Im, H., Shin, S., Kim, S., Park, E.: Selfsplat: Pose-free and 3d prior-free generalizable 3d gaussian splatting. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22012–22022 (2025)
- 32. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., et al.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)
- 33. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)
- 34. Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V.: Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics 36(4) (2017)
- 35. Lai, Z., Liu, S., Efros, A.A., Wang, X.: Video autoencoder: self-supervised disentanglement of static 3d structure and motion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9730–9740 (2021)
- 36. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r. In: European conference on computer vision. pp. 71–91. Springer (2024)
- 37. Li, R., Yi, B., Liu, J., Gao, H., Ma, Y., Kanazawa, A.: Cameras as relative positional encoding. Advances in Neural Information Processing Systems (2025)
- 38. Lin, H., Chen, S., Liew, J., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647 (2025)
- 39. Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al.: Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22160–22169 (2024)
- 40. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 41. Lu, G., Zhang, S., Wang, Z., Liu, C., Lu, J., Tang, Y.: Manigaussian: Dynamic gaussian splatting for multi-task robotic manipulation. In: European Conference on Computer Vision. pp. 349–366. Springer (2024)
- 42. Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., Dai, B.: Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20654–20664 (2024)
- 43. Miyato, T., Jaeger, B., Welling, M., Geiger, A.: Gta: A geometry-aware attention mechanism for multi-view transformers. arXiv preprint arXiv:2310.10375 (2023)

- 44. Moenne-Loccoz, N., Mirzaei, A., Perel, O., De Lutio, R., Martinez Esturo, J., State, G., Fidler, S., Sharp, N., Gojcic, Z.: 3d gaussian ray tracing: Fast tracing of particle scenes. ACM Transactions on Graphics (TOG) 43(6), 1–19 (2024)
- 45. Mur-Artal, R., Montiel, J.M.M., Tardos, J.D.: Orb-slam: A versatile and accurate monocular slam system. IEEE transactions on robotics 31(5), 1147–1163 (2015)
- 46. Nam, S., Sun, X., Kang, G., Lee, Y., Oh, S., Park, E.: Generative densification: Learning to densify gaussians for high-fidelity generalizable 3d reconstruction. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 26683–26693 (2025)
- 47. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Howes, R., Huang, P.Y., Xu, H., Sharma, V., Li, S.W., Galuba, W., Rabbat, M., Assran, M., Ballas, N., Synnaeve, G., Misra, I., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision (2023)
- 48. Pan, Z., Zhang, Y., Lin, T.: Telegs: End-to-end monocular gaussian head for immersive telepresence. Association for Computing Machinery, New York, NY, USA

(2025), https://doi.org/10.1145/3746441.3748230

- 49. Qin, T., Li, P., Shen, S.: Vins-mono: A robust and versatile monocular visualinertial state estimator. IEEE transactions on robotics 34(4), 1004–1020 (2018)
- 50. Sajjadi, M.S., Meyer, H., Pot, E., Bergmann, U., Greff, K., Radwan, N., Vora, S., Lučić, M., Duckworth, D., Dosovitskiy, A., et al.: Scene representation transformer: Geometry-free novel view synthesis through set-latent scene representations. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6229–6238 (2022)
- 51. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2016)
- 52. Schönberger, J.L., Zheng, E., Pollefeys, M., Frahm, J.M.: Pixelwise view selection for unstructured multi-view stereo. In: European Conference on Computer Vision (ECCV) (2016)
- 53. Shen, Q., Yi, X., Lin, M., Zhang, H., Yan, S., Wang, X.: Seeing world dynamics in a nutshell. arXiv preprint arXiv:2502.03465 (2025)
- 54. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)
- 55. Smith, C., Du, Y., Tewari, A., Sitzmann, V.: Flowcam: Training generalizable 3d radiance fields without camera poses via pixel-aligned scene flow. arXiv preprint arXiv:2306.00180 (2023)
- 56. Sun, X., Jiang, H., Liu, L., Nam, S., Kang, G., Wang, X., Sui, W., Su, Z., Liu, W., Wang, X., et al.: Uni3r: Unified 3d reconstruction and semantic understanding via generalizable gaussian splatting from unposed multi-view images. arXiv preprint arXiv:2508.03643 (2025)
- 57. Sun, Y.T., Huang, Y., Ma, L., Lyu, X., Cao, Y.P., Qi, X.: Splatter a video: Video gaussian representation for versatile processing. Advances in Neural Information Processing Systems 37, 50401–50425 (2024)
- 58. Szymanowicz, S., Rupprecht, C., Vedaldi, A.: Splatter image: Ultra-fast singleview 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10208–10217 (2024)
- 59. Tang, Z., Fan, Y., Wang, D., Xu, H., Ranjan, R., Schwing, A., Yan, Z.: Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5283–5293 (2025)

- 60. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)
- 61. Wang, P., Tan, H., Bi, S., Xu, Y., Luan, F., Sunkavalli, K., Wang, W., Xu, Z., Zhang, K.: Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction. arXiv preprint arXiv:2311.12024 (2023)
- 62. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20697–20709 (2024)
- 63. Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., He, T.: π3: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347 (2025)
- 64. Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861
- 65. Wu, Y., Jeon, M., Chang, J.H.R., Tuzel, O., Tulsiani, S.: Rayrope: Projective ray positional encoding for multi-view attention. arXiv preprint arXiv:2601.15275

(2026)

- 66. Xiong, K., Gong, S., Ye, X., Tan, X., Wan, J., Ding, E., Wang, J., Bai, X.: Cape: Camera view position embedding for multi-view 3d object detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 21570–21579 (2023)
- 67. Xu, H., Barath, D., Geiger, A., Pollefeys, M.: Resplat: Learning recurrent gaussian splats. arXiv preprint arXiv:2510.08575 (2025)
- 68. Xu, H., Peng, S., Wang, F., Blum, H., Barath, D., Geiger, A., Pollefeys, M.: Depthsplat: Connecting gaussian splatting and depth. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16453–16463 (2025)
- 69. Yang, J., Sax, A., Liang, K.J., Henaff, M., Tang, H., Cao, A., Chai, J., Meier, F., Feiszli, M.: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21924–21935 (2025)
- 70. Ye, B., Chen, B., Xu, H., Barath, D., Pollefeys, M.: Yonosplat: You only need one model for feedforward 3d gaussian splatting. arXiv preprint arXiv:2511.07321

(2025)

- 71. Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.H., Peng, S.: No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207 (2024)
- 72. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12–22 (2023)
- 73. Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A.: Mip-splatting: Alias-free 3d gaussian splatting. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19447–19456 (2024)
- 74. Zhang, K., Bi, S., Tan, H., Xiangli, Y., Zhao, N., Sunkavalli, K., Xu, Z.: Gs-lrm: Large reconstruction model for 3d gaussian splatting. In: European Conference on Computer Vision. pp. 1–19. Springer (2024)
- 75. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

- 76. Zhao, Q., Tan, H., Wang, Q., Bi, S., Zhang, K., Sunkavalli, K., Tulsiani, S., Jiang, H.: E-rayzer: Self-supervised 3d reconstruction as spatial visual pre-training. arXiv preprint arXiv:2512.10950 (2025)
- 77. Zhou, T., Tucker, R., Flynn, J., Fyffe, G., Snavely, N.: Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817 (2018)
- 78. Zhou, X., Lin, Z., Shan, X., Wang, Y., Sun, D., Yang, M.H.: Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 21634–21643 (2024)
- 79. Zielonka, W., Bagautdinov, T., Saito, S., Zollhöfer, M., Thies, J., Romero, J.: Drivable 3d gaussian avatars. In: 2025 International Conference on 3D Vision (3DV). pp. 979–990. IEEE (2025)
- 80. Ziwen, C., Tan, H., Zhang, K., Bi, S., Luan, F., Hong, Y., Fuxin, L., Xu, Z.: Longlrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4349–4359 (2025)

2Xplat: Two Experts Are Better Than One Generalist Supplementary Material

- A Additional Details

Implementation details. All input images are resized such that the shorter side matches the target resolution, followed by a center square crop. During finetuning, the model is optimized using a combination of rendering and relative pose losses, where the rendering loss includes a perceptual term with weight λperc = 0.5, and the relative pose loss uses weights λR = 0.1, λt = 10, and λK = 0.5 for rotation, translation, and intrinsic parameters, respectively. We train the model using the AdamW [40] optimizer with a learning rate of 2×10−5, β1 = 0.9, β2 = 0.99, and a weight decay of 0.05, and apply gradient clipping based on gradient norm to stabilize end-to-end training. For evaluation-time pose alignment (EPA), we further refine all camera parameters for 100 iterations using the Adam [33] optimizer with a learning rate of 1×10−4, following YoNoSplat [70] for a fair comparison. To address the scale ambiguity between the predicted poses from the geometry expert and the ground-truth poses obtained from SfM, we normalize both scenes by dividing the translations by the maximum translation magnitude.

For DL3DV [39] training at 224 × 224 resolution, we use 6, 12, 24, and 32 input views with corresponding target views of 6, 6, 6, and 1, respectively. The batch size per GPU is set to 4, 2, 1, and 1 for each configuration. For RE10K [77], we train with an image resolution of 224×224, using 6 context views and 8 target views with a batch size of 8 per GPU. For DL3DV, we additionally train at a higher resolution of 540×960, using 16 and 32 context views with corresponding target views of 4 and 1, respectively, and batch sizes of 2 and 1 per GPU.

Datasets. For zero-shot inference, we additionally evaluate on the train and truck scenes from Tanks&Templates [34], as well as 9 scenes (bicycle, bonsai, counter, garden, kitchen, room, stump, flower, and treehill) from the MipNeRF360 [1] dataset. For fair comparison across methods, all images are downsampled to a resolution closest to, but not smaller than, 960 × 540.

Table 8: Different backbones.

6v 12v 24v speed PSNR ↑ LPIPS ↓ speed PSNR ↑ LPIPS ↓ speed PSNR ↑ LPIPS ↓

Method size

YoNoSplat (π3) 1B 0.354s 22.290 0.173 0.410s 20.383 0.229 0.646s 19.711 0.255 Ours-L 0.5B 0.175s 25.400 0.133 0.255s 25.085 0.141 0.456s 24.762 0.144 Ours-G 1.3B 0.326s 26.007 0.126 0.462s 26.015 0.129 0.801s 25.894 0.125 Ours-π3 1.1B 0.309s 25.777 0.130 0.401s 26.135 0.128 0.707s 25.649 0.131

[Figure 7]

- Fig. 7: PSNR vs. inference speed on DL3DV (224×224 resolution). 6, 12, and 24 input views from left to right. The circle sizes are proportional to the model sizes, and the numbers inside indicate the number of parameters.

### B Additional Results

Additional analyses. We further compare different backbone choices under varying numbers of input views (6v, 12v, and 24v) on DL3DV, evaluating both rendering quality and efficiency in Fig. 7 and Tab. 8. Inference speed is measured end-to-end on a single RTX 3090 GPU. Across all settings, our methods consistently outperform the current state-of-the-art method, YoNoSplat [70] by a large margin in PSNR and LPIPS, demonstrating the effectiveness of our two-expert design for geometry estimation and appearance modeling. Among our variants, Ours-G achieves the best rendering quality, obtaining the highest PSNR and the lowest LPIPS across nearly all view configurations, indicating that a larger backbone further improves camera pose estimation, thereby enhancing reconstruction fidelity. Meanwhile, Ours-L offers the fastest inference speed while still maintaining strong rendering performance, highlighting an efficient quality–speed tradeoff. Notably, our models remain stable as the number of input views increases from 6 to 24, whereas the baseline exhibits clear performance degradation. These results demonstrate that our framework scales well with additional views and that the performance gains are consistent across different backbone capacities.

For evaluating zero-shot generalization performance, we further assess our model on the Tanks&Templates and Mip-NeRF360 datasets, and compare against several pose-dependent baselines. As shown in Tab. 9, despite not relying on camera pose supervision, our method achieves competitive—and in some cases superior—performance across all metrics. Notably, our approach maintains stable performance as the number of input views increases, whereas prior methods exhibit larger performance variations. This indicates that our model generalizes effectively across diverse scenes and view configurations. Overall, these results demonstrate that our pose-free formulation does not hinder generalization; instead, it enables robust performance comparable to pose-dependent methods, highlighting the strength of our architecture in learning view-consistent representations.

##### Table 9: Cross Dataset Generalization.

Tanks & Temples Mip-NeRF360

Method Pose Views

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Long-LRM ✓

18.59 0.614 0.366 21.08 0.484 0.445

- iLRM ✓ 18.58 0.631 0.385 21.09 0.495 0.466 MVP ✓ 19.54 0.708 0.277 22.21 0.587 0.355

- Ours 19.28 0.681 0.298 21.15 0.509 0.405 Long-LRM ✓

64

19.44 0.651 0.334 21.30 0.499 0.431 iLRM ✓ 19.82 0.692 0.318 21.60 0.522 0.444

- MVP ✓ 21.24 0.761 0.221 23.72 0.656 0.302

Ours 20.65 0.710 0.256 21.85 0.535 0.375 Long-LRM ✓

128

18.47 0.613 0.375 19.82 0.484 0.457 iLRM ✓ 19.22 0.696 0.319 21.32 0.551 0.424

- MVP ✓ 22.36 0.804 0.184 25.12 0.736 0.248

- Ours 21.59 0.743 0.227 22.34 0.557 0.258

32

Additional qualitative results. We provide additional qualitative comparisons on RE10K [77], DL3DV [39], high-resolution DL3DV (960 × 540), and cross-dataset generalization results on ScanNet++ [72], as shown in Figs. 8 to 11. These results further highlight the strong reconstruction quality of our method across different datasets and resolutions.

### C Limitations

While our primary goal is to achieve high-quality rendering for the novel view synthesis (NVS) task, our method focuses primarily on improving appearance modeling within the pose-free feed-forward 3DGS framework. As a result, our approach achieves state-of-the-art rendering performance and demonstrates strong scalability to a large number of input views, as well as good generalization across diverse scenes. However, since camera pose supervision is used mainly as a regularization signal rather than a primary optimization objective, the pose estimation accuracy of our model is slightly lower than that of methods specifically designed for pose prediction. Nevertheless, our results remain comparable to prior approaches, suggesting that our framework still learns meaningful geometric structure while prioritizing rendering quality.

[Figure 8]

###### Fig. 8: Qualitative comparison on DL3DV (224 × 224 resolution, 1-3 rows: 6 context views, 4-6 rows: 12 context views, 7-9 rows: 24 context views).

[Figure 9]

###### Fig. 9: Qualitative comparison on RE10K dataset with 6 context views.

[Figure 10]

###### Fig. 10: Qualitative cross-dataset generalization results from DL3DV to ScanNet++ (224 × 224 resolution, 1-2 rows: 32 context views, 3-4 rows: 64 context views, 5-6 rows: 128 context views).

[Figure 11]

###### Fig. 11: Qualitative comparison on high-resolution DL3DV (960 × 540, 1-2 rows: 16 context views, 3-4 rows: 32 context views, 5-6 rows: 64 context views, 7-8 rows: 128 context views).

